// gi_probe_blend_visibility.comp.hlsl - fold one update's ray DISTANCES into
// the visibility atlas: mean and mean-squared, for the Chebyshev leak test
// gi_sample.hlsli performs.
//
// Same shape as gi_probe_blend.comp.hlsl (one threadgroup per probe, one thread
// per texel of the 16x16 tile, border replicated out of groupshared) and the
// same reasoning behind it. Two things differ, both deliberate:
//
//  * THE DISTANCE IS CLAMPED to GI_VISIBILITY_DISTANCE_SCALE * spacing. The
//    Chebyshev test only ever compares a shading point's distance-to-probe
//    against the stored mean, and a point is at most about one spacing away
//    from the probe that answers for it. Letting one ray that flew 4000 units
//    into open sky drag the mean up would make the test declare every nearby
//    wall visible, which is exactly the leak the moments exist to prevent.
//    Clamping also folds misses in for free: a miss recorded rayMaxDistance,
//    which clamps to "as far as anything matters", i.e. no occluder.
//
//  * THE WEIGHTING LOBE IS SHARP (cos^50 rather than cos^1). Visibility is a
//    high-frequency signal — a wall is there or it is not — so a texel must
//    take only the rays that went essentially its way. The price is that at
//    32-96 rays most texels get NO ray inside their lobe on any given update
//    and simply keep their previous value; the per-frame rotation of the ray
//    set is what makes every texel get its turn over successive updates. That
//    is the DDGI trade, and it is why the ray set is rotated rather than fixed.

#include "gi_probe.hlsli"

struct GiProbeUpdate {
    float3 worldPosition;
    uint slot;
    uint flags;
    float3 _pad;
};
#define GI_UPDATE_FLAG_UNINITIALIZED 1u

[[vk::binding(0, 2)]]
cbuffer GiProbeBlendUniforms : register(b0, space2) {
    uint giProbeCount;
    uint giRayCount;
    float giDistanceClamp;
    float giRayMaxDistance;
    uint giFrameIndex;
    uint giCascadeIndex; // which BAND of the shared atlas this dispatch writes
    /// R5b: the analytic sun gradient's global hysteresis floor, [0, 1]. The
    /// renderer knows on the CPU how far the sun moved this frame, so a
    /// time-of-day scrub or a lightning spike collapses the field's
    /// accumulation window BEFORE any probe ray reports the change. Used by the
    /// IRRADIANCE blend only — the visibility moments describe geometry, which
    /// the sun moving does not alter — but the struct is shared by both passes,
    /// so the field is declared in both. 0 is the identity.
    float giLightingReset;
    uint _giPad0;
};

// SET 0 ORDER: READ-ONLY storage textures, then READ-ONLY storage buffers.
[[vk::binding(0, 0)]] Texture2D<float4> giProbeOffset : register(t0, space0);
[[vk::binding(1, 0)]] StructuredBuffer<float4> giRayResults : register(t1, space0);
[[vk::binding(2, 0)]] StructuredBuffer<GiProbeUpdate> giUpdates : register(t2, space0);

// The EMA reads this target as well as writing it. An explicit image format is
// what makes that legal without the optional `shaderStorageImageReadWithoutFormat`
// device feature — a storage image declared Unknown may only be written on
// hardware that does not advertise it.
[[vk::image_format("rg16f")]]
[[vk::binding(0, 1)]] RWTexture2D<float2> giVisibility : register(u0, space1);

groupshared float2 gPayload[GI_VISIBILITY_PAYLOAD][GI_VISIBILITY_PAYLOAD];
/// THE ROTATED DIRECTION SET, computed once per group instead of once per
/// (texel, ray). It depends only on `r`, `giRayCount` and `giFrameIndex`, all
/// group-uniform, and this pass has 196 payload texels — 196 x 64 x 1728 x 2
/// regenerations of a frac/divide/sqrt/sin/cos chain on the rota's worst frame,
/// against 96 that are actually distinct.
groupshared float3 gRayDir[GI_MAX_PROBE_RAYS];

[numthreads(GI_VISIBILITY_TILE, GI_VISIBILITY_TILE, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID) {
    uint probeIndex = tid.y / uint(GI_VISIBILITY_TILE);
    if (probeIndex >= giProbeCount) {
        return;
    }
    GiProbeUpdate probe = giUpdates[probeIndex];
    bool fresh = (probe.flags & GI_UPDATE_FLAG_UNINITIALIZED) != 0u;

    float4 offset = giProbeOffset.Load(int3(giProbeStateTexel(int(probe.slot), int(giCascadeIndex)), 0));
    if (!fresh && offset.w == GI_CLASS_INSIDE_GEOMETRY) {
        return;
    }

    // THE PRELOAD SITS BETWEEN THE TWO GROUP-UNIFORM EARLY RETURNS AND THE
    // isPayload SPLIT, so every surviving thread reaches this barrier. 16x16
    // threads cover GI_MAX_PROBE_RAYS one-to-one, so no stride is needed here.
    const uint rayCount = min(giRayCount, (uint)GI_MAX_PROBE_RAYS);
    {
        float3 c0, c1, c2;
        giRayRotation(giFrameIndex, c0, c1, c2);
        const uint flat = lid.y * uint(GI_VISIBILITY_TILE) + lid.x;
        if (flat < rayCount) {
            float3 d = giSphericalFibonacci(flat, rayCount);
            gRayDir[flat] = normalize(c0 * d.x + c1 * d.y + c2 * d.z);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    int2 tileLocal = int2(lid.xy);
    int2 tileOrigin = giProbeAtlasOrigin(int(probe.slot), GI_VISIBILITY_TILE, int(giCascadeIndex));
    bool isPayload = tileLocal.x >= GI_BORDER && tileLocal.x < GI_VISIBILITY_TILE - GI_BORDER &&
                     tileLocal.y >= GI_BORDER && tileLocal.y < GI_VISIBILITY_TILE - GI_BORDER;

    if (isPayload) {
        int2 local = tileLocal - GI_BORDER;
        float3 texelDir = giTexelDirection(local, GI_VISIBILITY_PAYLOAD);

        // DISTANCES ARE STORED NORMALIZED BY CASCADE SPACING, and that is a
        // correctness requirement rather than tidiness. The atlas is
        // R16G16_FLOAT, whose maximum is 65504, while `giDistanceClamp` is
        // 1.5 * spacing over spacings of 150 / 300 / 600 — so a raw second
        // moment reaches 50625 / 202500 / 810000 and cascades 1 and 2 SATURATE
        // TO INFINITY outright. On the clamping branch
        // `moments.y - moments.x * moments.x` goes negative, pins at the
        // variance floor, and the Chebyshev bound collapses into exactly the
        // hard binary test that SAMPLE_CHEBYSHEV_MIN_VARIANCE's comment says it
        // exists to prevent. Cascade 0 does not overflow but loses the variance
        // to catastrophic cancellation: at dist 200 the second moment is ~40000
        // where the fp16 ULP is 32, so the subtraction carries an absolute error
        // of ~20 against a floor of 2.25, and one ULP of drift in the EMA swings
        // that probe's cage weight by two orders of magnitude. Probes then drop
        // in and out of the 8-probe cage while giSampleCascade renormalizes over
        // the survivors, which redistributes the swing instead of damping it.
        //
        // Normalized, the stored pair is bounded by (1.5, 2.25) and every term
        // in the Chebyshev test is in units of spacing, which makes the test
        // exactly scale-invariant — the bias and variance-floor constants keep
        // meaning "a fraction of spacing" and simply stop being multiplied by it
        // on both sides.
        const float spacing = giCascadeSpacing(int(giCascadeIndex));
        const float invSpacing = 1.0 / max(spacing, 1e-6);

        float2 sum = float2(0, 0);
        float wsum = 0.0;
        for (uint r = 0; r < rayCount; ++r) {
            // The compare is EXACTLY the pow test that follows, for the shipped
            // sharpness: pow(c, 50) <= 1e-8 iff c <= GI_VISIBILITY_WEIGHT_CUTOFF.
            // At 64 directions over the sphere the mean nearest-neighbour
            // separation is ~25 degrees against this lobe's 46, so ~85% of pairs
            // used to evaluate a log2/exp2 pair purely to discard it. The
            // original test is kept below rather than replaced: it is free once
            // the pow has been computed, and keeping it is what makes this hoist
            // bit-identical rather than merely equivalent.
            float c = dot(texelDir, gRayDir[r]);
            if (c <= GI_VISIBILITY_WEIGHT_CUTOFF) {
                continue;
            }
            float w = pow(c, GI_VISIBILITY_SHARPNESS);
            if (w <= 1e-8) {
                continue;
            }
            // abs(): a backface hit is still an occluder at that distance. The
            // sign carries the classification signal, not a distance.
            float dist =
                min(abs(giRayResults[probeIndex * giRayCount + r].w), giDistanceClamp) * invSpacing;
            sum += w * float2(dist, dist * dist);
            wsum += w;
        }

        float2 previous = giVisibility[tileOrigin + tileLocal];
        float2 result;
        if (wsum <= 1e-8) {
            // No ray went this way on this update. An unfilled probe has no
            // previous value worth keeping, so seed it with "nothing in the
            // way" rather than with whatever its slot's last occupant saw.
            const float clampN = giDistanceClamp * invSpacing;
            result = fresh ? float2(clampN, clampN * clampN) : previous;
        } else {
            float2 estimate = sum / wsum;
            // A FIXED EMA, not the irradiance blend's change-adaptive one, and
            // the comment that used to sit here is the argument for the change:
            // these moments describe GEOMETRY, and the sun moving does not move
            // a wall.
            //
            // The adaptive curve exists to let a probe catch up quickly when the
            // LIGHTING changes, by collapsing the hysteresis in proportion to
            // `|new - old| / |old|`. Applied to geometry there is no such event
            // to catch up to, so every delta it reacts to is the Monte Carlo
            // noise of a different ray subset landing in this texel's narrow
            // cos^50 cone — and reacting to that noise by shortening the window
            // makes the next estimate noisier still. That is positive feedback,
            // and it is worst exactly where the estimator is weakest: `delta` is
            // normalized by `max(|old|, 1e-4)`, so a texel whose stored distance
            // is small sees the largest relative deltas from the smallest
            // absolute wobble.
            //
            // Holding the window open at GI_HYSTERESIS_MAX lets the moments
            // converge monotonically, which is all a static wall ever needs. A
            // wall that genuinely moves is a scrolled-in probe (`fresh`) or a
            // rebuilt scene, and both take the left-hand branch.
            result = fresh ? estimate : lerp(estimate, previous, GI_HYSTERESIS_MAX);
        }

        giVisibility[tileOrigin + tileLocal] = result;
        gPayload[local.y][local.x] = result;
    }

    GroupMemoryBarrierWithGroupSync();

    if (!isPayload) {
        int2 src = giBorderSource(tileLocal, GI_VISIBILITY_PAYLOAD);
        giVisibility[tileOrigin + tileLocal] = gPayload[src.y][src.x];
    }
}
