/**
 * Bilateral Blur Fragment Shader for AO
 *
 * Performs a depth-aware Gaussian blur in one direction (horizontal or vertical).
 * Run twice (H then V) for a separable 2D blur.
 * Edge rejection weights the RELATIVE linearized view-Z difference: raw
 * non-linear depth differences collapse with distance, which used to let AO
 * bleed across far silhouettes (halos). Linear view-Z relative to the center
 * sample keeps edge preservation uniform across the depth range.
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer BlurUniforms : register(b0, space3) {
    float2 direction;       // (1, 0) for horizontal, (0, 1) for vertical
    float2 invScreenSize;
    float sharpness;
    // Projection constants for depth linearization (glm RH_ZO):
    // viewZ = depthB / (depth + depthA), depthA = proj[2][2], depthB = proj[3][2]
    float depthA;
    float depthB;
    float _padding;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D<float> aoTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState aoSampler : register(s0, space2);

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D<float> depthTexture : register(t1, space2);
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerState depthSampler : register(s1, space2);

// Positive linear view depth from non-linear [0,1] depth.
float linearizeDepth(float depth) {
    return abs(depthB / (depth + depthA));
}

float PSMain(PSInput input) : SV_Target {
    float centerAO = aoTexture.SampleLevel(aoSampler, input.texCoord, 0);
    float centerDepth = depthTexture.SampleLevel(depthSampler, input.texCoord, 0);

    // Skip skybox
    if (centerDepth >= 0.9999) {
        return 1.0;
    }

    float centerZ = linearizeDepth(centerDepth);

    static const int KERNEL_RADIUS = 3;
    static const float weights[7] = {0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125};

    float totalWeight = weights[KERNEL_RADIUS]; // center weight
    float blurredAO = centerAO * totalWeight;

    float2 step = direction * invScreenSize;

    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; ++i) {
        if (i == 0) continue;

        float2 sampleUV = input.texCoord + step * float(i);
        float sampleAO = aoTexture.SampleLevel(aoSampler, sampleUV, 0);
        float sampleZ = linearizeDepth(depthTexture.SampleLevel(depthSampler, sampleUV, 0));

        // Depth-aware weight on relative linear view-Z difference: a fixed
        // percentage of the center depth counts as "same surface" at any range.
        float relDiff = abs(sampleZ - centerZ) / max(centerZ, 1e-3);
        float depthWeight = exp(-relDiff * sharpness);

        float w = weights[i + KERNEL_RADIUS] * depthWeight;
        blurredAO += sampleAO * w;
        totalWeight += w;
    }

    return blurredAO / totalWeight;
}
