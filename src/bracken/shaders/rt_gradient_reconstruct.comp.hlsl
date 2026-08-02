// rt_gradient_reconstruct.comp.hlsl — dense reconstruction of the sparse
// A-SVGF gradient. Phase R5b of docs/rt_lighting_plan.md §5.6.
//
// One thread per STRATUM, in and out. The gradient trace before it filled at
// most one verdict per 3x3 stratum of the traced image; this spreads each
// verdict over its neighbourhood so that the temporal pass — which reads the
// result per PIXEL — sees a continuous field rather than a scatter of isolated
// ones.
//
// A WEIGHTED MAXIMUM, NOT AN AVERAGE, and the deviation from A-SVGF's b3-spline
// a-trous blur is deliberate. Schied's gradient is a continuous luminance
// ratio, and blurring it is the right way to pool independent estimates of a
// smooth quantity. Ours is a BINARY visibility comparison: the raw verdict is 0
// or 1, and a b3 average of one fired stratum among its 24 quiet neighbours is
// 0.14 — which resets nothing, at exactly the pixels that most needed it. What
// the signal means is "something at this surface changed", and the correct
// spatial operator for a statement like that is a soft DILATION.
//
// So: the weighted max over the 5x5 stratum window, weights falling off by
// Chebyshev ring (render/rt_denoise_math.hpp's `gradient_dilation_weight`, the
// CPU source of truth and where the ring constants are justified). The fired
// stratum resets its own 3x3 region completely, the ring around it heavily, the
// ring beyond it partially, and nothing further. Monotone, idempotent under
// repeated application to an already-dilated field, and impossible to make a
// gradient LARGER than the largest raw verdict it came from.
//
// Both visibility signals run this same pipeline against their own targets,
// exactly as they share the classify/temporal/filter three.

#include "rt_denoise.hlsli"

[[vk::binding(0, 2)]]
cbuffer RtGradientReconstructUniforms : register(b0, space2) {
    uint2 stratumCount;
    uint2 _pad0;
};

/// The raw per-stratum verdicts. A read-only STORAGE texture rather than a
/// sampled one: every tap is an exact texel at an integer offset, so a sampler
/// slot would buy nothing but the chance of a half-texel error.
[[vk::binding(0, 0)]] Texture2D<float> gradientRaw : register(t0, space0);

[[vk::binding(0, 1)]] RWTexture2D<float> gradientOut : register(u0, space1);

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= stratumCount.x || tid.y >= stratumCount.y) {
        return;
    }
    const int2 coord = int2(tid.xy);
    const int2 last = int2(stratumCount) - 1;

    float dilated = 0.0;
    [unroll]
    for (int dy = -2; dy <= 2; ++dy) {
        [unroll]
        for (int dx = -2; dx <= 2; ++dx) {
            // Clamp rather than skip: the alternative is an edge stratum seeing
            // fewer neighbours and therefore dilating less, which would make the
            // screen border systematically slower to reset than its interior.
            const int2 s = clamp(coord + int2(dx, dy), int2(0, 0), last);
            dilated = max(dilated, gradientRaw.Load(int3(s, 0)) * rtGradientDilationWeight(dx, dy));
        }
    }

    gradientOut[coord] = dilated;
}
