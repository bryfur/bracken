/**
 * IBL Diffuse Irradiance Fragment Shader
 *
 * Cosine-hemisphere convolution of the sky-capture cube into the small
 * irradiance cube, one pass per face. Stores (1/pi) * integral(L * cos) —
 * the "multiply by albedo directly" convention the hemisphere ambient always
 * used (a uniform sky of radiance L0 convolves to exactly L0). Samples a
 * deep source mip so a modest sample count is noise-free.
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

// Mirrors gpu::IblFilterUniforms (gpu_uniforms.hpp); roughness is unused here.
[[vk::binding(0, 3)]]
cbuffer IblFilterUniforms {
    float3 faceForward;
    float roughness;
    float3 faceRight;
    float sourceSize;
    float3 faceUp;
    float _pad0;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
TextureCube<float4> sourceCube;
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState sourceSampler;

static const float PI_IBL = 3.14159265358979323846;
static const uint IRR_SAMPLE_COUNT = 128u;
// 64px capture mip 2 = 16px: plenty of angular detail for a cosine lobe.
static const float IRR_SOURCE_MIP = 2.0;

// Van der Corput radical inverse (base 2) — same sequence as ibl_prefilter.
float radicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float4 PSMain(PSInput input) : SV_Target {
    float2 st = input.texCoord * 2.0 - 1.0;
    float3 N = normalize(faceForward + st.x * faceRight + st.y * faceUp);

    float3 upAxis = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(upAxis, N));
    float3 B = cross(N, T);

    // Cosine-weighted hemisphere sampling: pdf = cos/pi cancels the
    // integrand, so the estimator is the plain average of the samples.
    float3 irradiance = float3(0.0, 0.0, 0.0);
    [loop]
    for (uint i = 0u; i < IRR_SAMPLE_COUNT; ++i) {
        float2 xi = float2(float(i) / float(IRR_SAMPLE_COUNT), radicalInverseVdC(i));
        float phi = 2.0 * PI_IBL * xi.x;
        float cosTheta = sqrt(1.0 - xi.y);
        float sinTheta = sqrt(xi.y);
        float3 L = normalize(T * (sinTheta * cos(phi)) + B * (sinTheta * sin(phi)) + N * cosTheta);
        irradiance += sourceCube.SampleLevel(sourceSampler, L, IRR_SOURCE_MIP).rgb;
    }

    return float4(irradiance / float(IRR_SAMPLE_COUNT), 1.0);
}
