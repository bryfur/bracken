/**
 * Bilateral Blur Fragment Shader for AO
 *
 * Performs a depth-aware Gaussian blur in one direction (horizontal or vertical).
 * Run twice (H then V) for a separable 2D blur.
 * Edge rejection weights the RELATIVE linearized view-Z difference: raw
 * non-linear depth differences collapse with distance, which used to let AO
 * bleed across far silhouettes (halos). Linear view-Z relative to the center
 * sample keeps edge preservation uniform across the depth range.
 *
 * Since R5a it filters all four channels of the AO texture
 * (shaders/bent_cone.hlsli): AO and aperture with the bilateral weights
 * directly, and the bent normal as a DIRECTION — every tap decoded, summed
 * with the same weights, renormalised and re-encoded. Both source textures in
 * the chain (the RGBA16F history for pass 1, the RGBA8 ping for pass 2) carry
 * the same channel layout, so one pipeline serves both passes.
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
    // Projection constants for depth linearization (glm RH_ZO, standard or
    // reversed-Z — the formula holds for both as long as depthA/depthB come
    // from the actual projection matrix):
    // viewZ = |depthB / (depth + depthA)|, depthA = proj[2][2], depthB = proj[3][2]
    float depthA;
    float depthB;
    float maxDistance; // view-space far cutoff, matches the AO passes
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D<float4> aoTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState aoSampler : register(s0, space2);

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D<float> depthTexture : register(t1, space2);
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerState depthSampler : register(s1, space2);

#include "bent_cone.hlsli"
#include "depth_linear.hlsli"

// Positive linear view depth from non-linear [0,1] depth.
float linearizeDepth(float depth) {
    return brackenLinearViewDepth(depth, depthA, depthB);
}

float4 PSMain(PSInput input) : SV_Target {
    float4 center = aoTexture.SampleLevel(aoSampler, input.texCoord, 0);
    float centerDepth = depthTexture.SampleLevel(depthSampler, input.texCoord, 0);

    // Sky: reversed-Z cleared/skybox depth is exactly 0.0 (same test as the AO
    // passes and volumetric_fog.frag). Pass the value through (it is the
    // fully-open escape there) rather than returning a constant that could
    // disagree with the AO pass.
    if (centerDepth <= 0.0) {
        return center;
    }

    float centerZ = linearizeDepth(centerDepth);

    // Same explicit far cutoff as the AO passes: past maxDistance AO has been
    // faded to 1.0, so pass it through instead of blurring. The old
    // `depth >= 0.9999` test put this boundary at a near-plane-dependent
    // distance and returned a constant across it.
    if (centerZ > maxDistance) {
        return center;
    }

    static const int KERNEL_RADIUS = 3;
    static const float weights[7] = {0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125};

    float3 centerDir = bentOctDecode(center.gb);

    float totalWeight = weights[KERNEL_RADIUS]; // center weight
    float blurredAO = center.r * totalWeight;
    float blurredAperture = center.a * totalWeight;
    float3 blurredDir = centerDir * totalWeight;

    float2 step = direction * invScreenSize;

    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; ++i) {
        if (i == 0) continue;

        float2 sampleUV = input.texCoord + step * float(i);
        float4 s = aoTexture.SampleLevel(aoSampler, sampleUV, 0);
        float sampleZ = linearizeDepth(depthTexture.SampleLevel(depthSampler, sampleUV, 0));

        // Depth-aware weight on relative linear view-Z difference: a fixed
        // percentage of the center depth counts as "same surface" at any range.
        float relDiff = abs(sampleZ - centerZ) / max(centerZ, 1e-3);
        float depthWeight = exp(-relDiff * sharpness);

        float w = weights[i + KERNEL_RADIUS] * depthWeight;
        blurredAO += s.r * w;
        blurredAperture += s.a * w;
        blurredDir += bentOctDecode(s.gb) * w;
        totalWeight += w;
    }

    float inv = 1.0 / totalWeight;
    return float4(blurredAO * inv, bentOctEncode(bentNormalize(blurredDir, centerDir)),
                  blurredAperture * inv);
}
