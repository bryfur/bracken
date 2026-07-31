/**
 * GTAO Temporal Accumulation Fragment Shader.
 *
 * Blends this frame's raw GTAO (rotating per-frame sample pattern) with the
 * reprojected AO history. Runs at the AO working resolution; the history is
 * RG16F with R = accumulated AO and G = the frame's linear view depth (used
 * for disocclusion rejection next frame).
 *
 * Reprojection follows the engine velocity convention (prepass RG16F,
 * UV-space delta, current minus previous: prev_uv = uv - velocity,
 * scene/velocity_math.hpp). Rejection is on RELATIVE linear-depth mismatch
 * against the depth stored in the history — feedback fades to zero as
 * |prevZ - currZ| / currZ approaches 1 / depthRejectScale, so disocclusions
 * fall back to the current frame instead of smearing AO behind movers.
 *
 * historyValid == 0 (first frame / resize / AO-mode gap / prepass-off frame)
 * outputs the current AO unblended — the same invalidation policy as TAA.
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer GtaoTemporalUniforms : register(b0, space3) {
    float2 invScreenSize;
    float depthA; // viewZ = |depthB / (depth + depthA)| (glm RH_ZO)
    float depthB;
    float historyValid;
    float feedbackMax;
    float depthRejectScale;
    float _pad0;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D<float> aoTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState aoSampler : register(s0, space2);

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D<float2> historyTexture : register(t1, space2);
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerState historySampler : register(s1, space2);

[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
Texture2D<float2> velocityTexture : register(t2, space2);
[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
SamplerState velocitySampler : register(s2, space2);

[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
Texture2D<float> depthTexture : register(t3, space2);
[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
SamplerState depthSampler : register(s3, space2);

// Positive linear view depth from non-linear [0,1] depth.
float linearizeDepth(float depth) {
    return abs(depthB / (depth + depthA));
}

float2 PSMain(PSInput input) : SV_Target {
    float2 uv = input.texCoord;
    float currAO = aoTexture.SampleLevel(aoSampler, uv, 0);
    float deviceDepth = depthTexture.SampleLevel(depthSampler, uv, 0);
    float currZ = linearizeDepth(deviceDepth);

    // Sky (reversed-Z cleared depth == 0.0): AO is 1 there and history never
    // helps — write through.
    if (deviceDepth <= 0.0) {
        return float2(currAO, currZ);
    }

    float feedback = 0.0;
    float prevAO = currAO;
    if (historyValid > 0.5) {
        float2 velocity = velocityTexture.SampleLevel(velocitySampler, uv, 0).xy;
        float2 prevUV = uv - velocity;
        if (all(prevUV >= 0.0) && all(prevUV <= 1.0)) {
            float2 history = historyTexture.SampleLevel(historySampler, prevUV, 0).xy;
            prevAO = history.x;
            float prevZ = history.y;
            float relDiff = abs(prevZ - currZ) / max(currZ, 1e-3);
            feedback = feedbackMax * saturate(1.0 - relDiff * depthRejectScale);
        }
    }

    return float2(lerp(currAO, prevAO, feedback), currZ);
}
