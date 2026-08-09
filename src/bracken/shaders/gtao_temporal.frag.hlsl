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
 * Nothing here can be bit-packed: the history is filtered at arbitrary
 * sub-texel reprojection positions, and a packed channel does not survive that.
 * RGBA16F rather than RGBA8 because a 0.9-feedback accumulation against an
 * 8-bit grid stalls — a per-frame delta under half an LSB never lands and the
 * history freezes.
 *
 * The history sampler is NEAREST and the bilinear filter is built BY HAND from
 * 4 taps, because GB is a pair of OCTAHEDRAL COORDINATES: those are
 * discontinuous across the fold seam, and a hardware filter averaging them
 * across it points the bent normal at neither input — for a wall facing world
 * -Z it can land 73 degrees off, and at the -Z pole it returns the exact
 * opposite direction. So the bent normal is filtered and then blended as a
 * DIRECTION (decode every tap, weight, normalise, lerp, re-encode), which is
 * what blur_ao and gtao_upsample already do and what bent_cone.hlsli requires
 * of every stage in the chain. AO and the aperture are linear quantities, so
 * the same manual weights reproduce what the hardware filter gave them; the
 * depth history keeps its linear sampler and is tapped at those same texel
 * CENTRES, which is the same thing again.
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
#include "depth_linear.hlsli"

// Positive linear view depth from non-linear [0,1] depth.
float linearizeDepth(float depth) {
    return brackenLinearViewDepth(depth, depthA, depthB);
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

    float3 currDir = bentOctDecode(curr.gb);

    float feedback = 0.0;
    float prevAo = curr.r;
    float prevAperture = curr.a;
    float3 prevDir = currDir;
    if (historyValid > 0.5) {
        float2 velocity = velocityTexture.SampleLevel(velocitySampler, uv, 0).xy;
        float2 prevUV = uv - velocity;
        if (all(prevUV >= 0.0) && all(prevUV <= 1.0)) {
            // The 4 history texels around the reprojected position, with the
            // bilinear weights the hardware would have used. Done by hand
            // because GB is an OCTAHEDRAL PAIR: the hardware would average the
            // coordinates, and across the fold seam that points at neither
            // input (bent_cone.hlsli). AO / aperture / depth reproduce the
            // hardware filter; the direction is decoded per tap, weighted and
            // renormalised, exactly as blur_ao and gtao_upsample already do.
            // Taps can land half a texel outside [0,1] at the frame border,
            // which is what CLAMP_TO_EDGE is for.
            float2 pos = prevUV / invScreenSize - 0.5;
            float2 base = floor(pos);
            float2 f = pos - base;
            float2 uv00 = (base + 0.5) * invScreenSize;
            float2 tapUV[4];
            tapUV[0] = uv00;
            tapUV[1] = uv00 + float2(invScreenSize.x, 0.0);
            tapUV[2] = uv00 + float2(0.0, invScreenSize.y);
            tapUV[3] = uv00 + invScreenSize;

            float4 w = float4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y),
                              (1.0 - f.x) * f.y, f.x * f.y);

            prevAo = 0.0;
            prevAperture = 0.0;
            float3 dirSum = float3(0.0, 0.0, 0.0);
            float prevZ = 0.0;
            [unroll]
            for (int i = 0; i < 4; ++i) {
                float4 s = historyTexture.SampleLevel(historySampler, tapUV[i], 0);
                prevAo += s.r * w[i];
                prevAperture += s.a * w[i];
                dirSum += bentOctDecode(s.gb) * w[i];
                prevZ += depthHistoryTexture.SampleLevel(depthHistorySampler, tapUV[i], 0).x * w[i];
            }
            // Four taps that cancel is exactly the case the seam used to
            // produce; the current frame is the honest fallback, never zero.
            prevDir = bentNormalize(dirSum, currDir);

            float relDiff = abs(prevZ - currZ) / max(currZ, 1e-3);
            feedback = feedbackMax * saturate(1.0 - relDiff * depthRejectScale);
        }
    }

    float ao = lerp(curr.r, prevAo, feedback);
    float aperture = lerp(curr.a, prevAperture, feedback);
    float3 dir = bentNormalize(lerp(currDir, prevDir, feedback), currDir);

    output.history = float4(ao, bentOctEncode(dir), aperture);
    return output;
}
