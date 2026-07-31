/**
 * IBL Specular Prefilter Fragment Shader
 *
 * GGX-filtered importance sampling of the sky-capture cube (Karis split-sum,
 * first sum): one pass per (face, mip) of the specular cube; render::IBL maps
 * mip -> roughness = mip / (mips - 1). N = V = R approximation. Source
 * samples are mip-biased by the ratio of the sample's solid angle to a source
 * texel's solid angle, so a low sample count stays noise-free (quality over
 * noise at 64px).
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

// Mirrors gpu::IblFilterUniforms (gpu_uniforms.hpp).
[[vk::binding(0, 3)]]
cbuffer IblFilterUniforms {
    float3 faceForward;
    float roughness;   // this mip's perceptual roughness
    float3 faceRight;
    float sourceSize;  // mip-0 face size of the source capture cube
    float3 faceUp;
    float _pad0;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
TextureCube<float4> sourceCube;
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState sourceSampler;

static const float PI_IBL = 3.14159265358979323846;
static const uint SPEC_SAMPLE_COUNT = 32u;

// Van der Corput radical inverse (base 2). CPU mirror: ibl_radical_inverse
// (render/ibl.hpp) — keep in sync.
float radicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

float2 hammersley(uint i, uint n) {
    return float2(float(i) / float(n), radicalInverseVdC(i));
}

// GGX-distributed half vector around N (tangent-space cosine-lobe mapping).
float3 importanceSampleGGX(float2 xi, float3 N, float rough) {
    float a = rough * rough;
    float phi = 2.0 * PI_IBL * xi.x;
    float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float3 H = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    float3 upAxis = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(upAxis, N));
    float3 B = cross(N, T);
    return normalize(T * H.x + B * H.y + N * H.z);
}

float4 PSMain(PSInput input) : SV_Target {
    float2 st = input.texCoord * 2.0 - 1.0;
    float3 N = normalize(faceForward + st.x * faceRight + st.y * faceUp);

    // Mip 0 (roughness 0): the lobe is a delta — pass the source through.
    if (roughness < 1e-3) {
        return float4(sourceCube.SampleLevel(sourceSampler, N, 0).rgb, 1.0);
    }

    float3 V = N; // N = V = R approximation (Karis)
    float3 color = float3(0.0, 0.0, 0.0);
    float totalWeight = 0.0;

    // Solid angle of one source mip-0 texel, for the mip-bias heuristic.
    float saTexel = 4.0 * PI_IBL / (6.0 * sourceSize * sourceSize);

    [loop]
    for (uint i = 0u; i < SPEC_SAMPLE_COUNT; ++i) {
        float2 xi = hammersley(i, SPEC_SAMPLE_COUNT);
        float3 H = importanceSampleGGX(xi, N, roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);
        float NdotL = saturate(dot(N, L));
        if (NdotL > 0.0) {
            // Sample pdf -> solid angle -> source mip (Karis GPU Gems-style
            // filtered importance sampling).
            float NdotH = saturate(dot(N, H));
            float HdotV = saturate(dot(H, V));
            float a = roughness * roughness;
            float a2 = a * a;
            float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
            float D = a2 / max(PI_IBL * d * d, 1e-7);
            float pdf = D * NdotH / max(4.0 * HdotV, 1e-4) + 1e-4;
            float saSample = 1.0 / (float(SPEC_SAMPLE_COUNT) * pdf);
            float mip = max(0.5 * log2(saSample / saTexel), 0.0);

            color += sourceCube.SampleLevel(sourceSampler, L, mip).rgb * NdotL;
            totalWeight += NdotL;
        }
    }

    return float4(color / max(totalWeight, 1e-4), 1.0);
}
