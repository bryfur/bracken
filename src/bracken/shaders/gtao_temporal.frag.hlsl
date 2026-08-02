/**
 * GTAO Temporal Accumulation Fragment Shader.
 *
 * Blends this frame's raw GTAO (rotating per-frame sample pattern) with the
 * reprojected history. Runs at the AO working resolution.
 *
 * TWO TARGETS since R5a, because the pass now carries five linearly filtered
 * quantities and four channels:
 *   target 0  RGBA16F  AO, octahedral world bent normal (GB), cone aperture
 *   target 1  RG16F    R = this frame's linear view depth, for next frame's
 *                      disocclusion test (G unused)
 * Nothing here can be bit-packed: the history is sampled with a LINEAR filter
 * at arbitrary sub-texel reprojection positions, and a packed channel does not
 * survive that. RGBA16F rather than RGBA8 because a 0.9-feedback accumulation
 * against an 8-bit grid stalls — a per-frame delta under half an LSB never
 * lands and the history freezes.
 *
 * The bent normal is blended as a DIRECTION (decode both ends, lerp, normalise,
 * re-encode), never as a pair of octahedral coordinates: those are
 * discontinuous across the fold seam, and averaging across it points the bent
 * normal at neither input. The aperture is a projected solid angle and blends
 * linearly as it stands.
 *
 * Reprojection follows the engine velocity convention (prepass RG16F,
 * UV-space delta, current minus previous: prev_uv = uv - velocity,
 * scene/velocity_math.hpp). Rejection is on RELATIVE linear-depth mismatch
 * against the depth history — feedback fades to zero as
 * |prevZ - currZ| / currZ approaches 1 / depthRejectScale, so disocclusions
 * fall back to the current frame instead of smearing AO behind movers.
 *
 * historyValid == 0 (first frame / resize / AO-mode gap / prepass-off frame)
 * outputs the current frame unblended — the same invalidation policy as TAA.
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

struct PSOutput {
    float4 history : SV_Target0; // AO, oct.x, oct.y, aperture
    float2 depth : SV_Target1;   // R = linear view depth
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
Texture2D<float4> aoTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState aoSampler : register(s0, space2);

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D<float4> historyTexture : register(t1, space2);
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

[[vk::combinedImageSampler]][[vk::binding(4, 2)]]
Texture2D<float2> depthHistoryTexture : register(t4, space2);
[[vk::combinedImageSampler]][[vk::binding(4, 2)]]
SamplerState depthHistorySampler : register(s4, space2);

#include "bent_cone.hlsli"

// Positive linear view depth from non-linear [0,1] depth.
float linearizeDepth(float depth) {
    return abs(depthB / (depth + depthA));
}

PSOutput PSMain(PSInput input) {
    float2 uv = input.texCoord;
    float4 curr = aoTexture.SampleLevel(aoSampler, uv, 0);
    float deviceDepth = depthTexture.SampleLevel(depthSampler, uv, 0);
    float currZ = linearizeDepth(deviceDepth);

    PSOutput output;
    output.depth = float2(currZ, 0.0);

    // Sky (reversed-Z cleared depth == 0.0): the raw pass already wrote the
    // fully-open escape value and history never helps — write through.
    if (deviceDepth <= 0.0) {
        output.history = curr;
        return output;
    }

    float feedback = 0.0;
    float4 prev = curr;
    if (historyValid > 0.5) {
        float2 velocity = velocityTexture.SampleLevel(velocitySampler, uv, 0).xy;
        float2 prevUV = uv - velocity;
        if (all(prevUV >= 0.0) && all(prevUV <= 1.0)) {
            prev = historyTexture.SampleLevel(historySampler, prevUV, 0);
            float prevZ = depthHistoryTexture.SampleLevel(depthHistorySampler, prevUV, 0).x;
            float relDiff = abs(prevZ - currZ) / max(currZ, 1e-3);
            feedback = feedbackMax * saturate(1.0 - relDiff * depthRejectScale);
        }
    }

    float ao = lerp(curr.r, prev.r, feedback);
    float aperture = lerp(curr.a, prev.a, feedback);
    float3 currDir = bentOctDecode(curr.gb);
    float3 dir = bentNormalize(lerp(currDir, bentOctDecode(prev.gb), feedback), currDir);

    output.history = float4(ao, bentOctEncode(dir), aperture);
    return output;
}
