/**
 * SSAO (Screen-Space Ambient Occlusion) Fragment Shader
 *
 * Simpler than GTAO — uses precomputed hemisphere kernel samples around each
 * fragment to estimate occlusion. The kernel is generated once on the CPU
 * (see AmbientOcclusion::render_ssao_pass); the shader only applies a cheap
 * per-pixel IGN rotation instead of 3 hashes + acos/sin/cos per sample.
 * Outputs a single-channel AO value (1.0 = no occlusion, 0.0 = full occlusion).
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

#define KERNEL_SIZE 24

[[vk::binding(0, 3)]]
cbuffer GTAOUniforms : register(b0, space3) {
    float4x4 projection;
    float4x4 invProjection;
    float2 screenSize;
    float2 invScreenSize;
    float radius;
    float bias;
    int numDirections;
    int numSteps;
    float radiusScale;
    // Scalars (not a float array) so the kernel lands at byte 176, matching
    // the CPU struct: HLSL would pad float[3] to one register per element.
    float _padding0;
    float _padding1;
    float _padding2;
    // xyz = hemisphere direction (z >= 0) * per-sample length in [0,1]
    float4 sampleKernel[KERNEL_SIZE];
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D<float> depthTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState depthSampler : register(s0, space2);

static const float PI = 3.14159265359;

float3 viewPosFromDepth(float2 uv, float depth) {
    float4 clipPos = float4(uv * 2.0 - 1.0, depth, 1.0);
    clipPos.y = -clipPos.y;
    float4 viewPos = mul(invProjection, clipPos);
    return viewPos.xyz / viewPos.w;
}

float interleavedGradientNoise(float2 pos) {
    return frac(52.9829189 * frac(dot(pos, float2(0.06711056, 0.00583715))));
}

float PSMain(PSInput input) : SV_Target {
    float depth = depthTexture.SampleLevel(depthSampler, input.texCoord, 0);

    if (depth >= 0.9999) {
        return 1.0;
    }

    float3 viewPos = viewPosFromDepth(input.texCoord, depth);

    // Reconstruct normal from screen-space derivatives
    float3 dpdx = ddx(viewPos);
    float3 dpdy = ddy(viewPos);
    float3 viewNormal = normalize(cross(dpdy, dpdx));

    // Per-pixel IGN rotation: one sincos for the whole pixel, kernel rotates
    // around the surface normal (tangent-space z).
    float rotationAngle = interleavedGradientNoise(input.position.xy) * 2.0 * PI;
    float ca = cos(rotationAngle);
    float sa = sin(rotationAngle);

    // Build a tangent basis around the view-space normal.
    float3 up = abs(viewNormal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, viewNormal));
    float3 bitangent = cross(viewNormal, tangent);

    int totalSamples = min(numDirections * numSteps, KERNEL_SIZE);
    float occlusion = 0.0;
    float effectiveRadius = radius * radiusScale;

    for (int i = 0; i < totalSamples; ++i) {
        float3 k = sampleKernel[i].xyz;
        // Rotate kernel xy by the per-pixel angle, then orient to the normal hemisphere.
        float3 rk = float3(k.x * ca - k.y * sa, k.x * sa + k.y * ca, k.z);
        float3 sampleDir = tangent * rk.x + bitangent * rk.y + viewNormal * rk.z;

        // Kernel length already encodes per-sample distance in [0,1].
        float3 samplePos = viewPos + sampleDir * effectiveRadius;

        // Project sample to screen space
        float4 clipPos = mul(projection, float4(samplePos, 1.0));
        clipPos.xy /= clipPos.w;
        clipPos.y = -clipPos.y;
        float2 sampleUV = clipPos.xy * 0.5 + 0.5;

        if (any(sampleUV < 0.0) || any(sampleUV > 1.0)) {
            continue;
        }

        float sampleDepth = depthTexture.SampleLevel(depthSampler, sampleUV, 0);
        float3 actualPos = viewPosFromDepth(sampleUV, sampleDepth);

        // Check if the actual surface is closer than our sample (occluding)
        float rangeCheck = smoothstep(0.0, 1.0, effectiveRadius / max(abs(viewPos.z - actualPos.z), 0.001));
        occlusion += (actualPos.z >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck;
    }

    occlusion /= max(float(totalSamples), 1.0);
    return saturate(1.0 - occlusion);
}
