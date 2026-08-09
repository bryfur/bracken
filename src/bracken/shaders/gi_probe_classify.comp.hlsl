// gi_probe_classify.comp.hlsl - one thread per probe updated this frame:
// decide whether it is buried in geometry, nudge it out if so, and write off
// the ones that cannot be saved.
//
// WHY A SPARSE GRID NEEDS THIS AT ALL. A 150-unit lattice anchored at the world
// origin does not know where the walls are. Some probes land inside a building
// shell, a rock, or the terrain; those see backfaces in most directions, and a
// probe that shades what it sees from inside a wall reads as brightly lit when
// it should read as dark. Left alone it leaks that brightness into every
// surface whose trilinear cage touches it. Relocation and classification are
// what make a sparse grid survive interiors — they arrive WITH the structure in
// every shipping DDGI implementation, and this is ours.
//
// THE SIGNAL is the sign of the ray-result distance: the trace negates it when
// the ray struck a triangle from behind. Nothing else in the pipeline can tell
// "inside a wall" from "in a dark alley" — both are black.
//
// THE VERDICT lands in the per-probe offset texture: xyz = the sample-position
// offset in world units (clamped to GI_MAX_OFFSET_FRACTION of the spacing),
// w = GI_CLASS_OK or GI_CLASS_INSIDE_GEOMETRY. It is deliberately NOT in
// probe_state, which has a CPU writer that would erase it on the next scroll —
// see render/gi/gi_math.hpp.
//
// RELOCATION LAGS BY ONE UPDATE, on purpose. The rays that motivate a nudge
// were traced from the OLD position, so applying a full jump on their evidence
// would overshoot; the probe steps a quarter of a spacing per update and
// converges over a few turns of the rota. The trace pass reads this texture
// when it fires the probe's next ray set, which is where the offset takes
// effect.

#include "gi_probe.hlsli"

struct GiProbeUpdate {
    float3 worldPosition;
    uint slot;
    uint flags;
    float3 _pad;
};
#define GI_UPDATE_FLAG_UNINITIALIZED 1u

[[vk::binding(0, 2)]]
cbuffer GiProbeClassifyUniforms : register(b0, space2) {
    uint giProbeCount;
    uint giRayCount;
    uint giCascadeIndex;
    float giMaxOffset;
    float giProbeSpacing;
    uint giFrameIndex;
    float _giPad0;
    float _giPad1;
};

// SET 0: READ-ONLY storage buffers only (no sampled or read-only storage
// textures), so they start at t0.
[[vk::binding(0, 0)]] StructuredBuffer<float4> giRayResults : register(t0, space0);
[[vk::binding(1, 0)]] StructuredBuffer<GiProbeUpdate> giUpdates : register(t1, space0);

// SET 1: read-write storage TEXTURES first, then read-write storage BUFFERS —
// the two runs are numbered contiguously in one `u` space (gpu_compute.hpp).
[[vk::image_format("rgba16f")]]
[[vk::binding(0, 1)]] RWTexture2D<float4> giProbeOffset : register(u0, space1);
/// Four uints per cascade, mirroring gpu::GiProbeCounters. The CPU zeroes them
/// each frame, so they describe THIS frame's scheduled probes and not a running
/// total that could never be attributed to anything.
[[vk::binding(1, 1)]] RWStructuredBuffer<uint> giCounters : register(u1, space1);

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= giProbeCount) {
        return;
    }
    GiProbeUpdate probe = giUpdates[tid.x];
    int2 texel = giProbeStateTexel(int(probe.slot), int(giCascadeIndex));

    // A freshly scrolled-in probe inherits nothing: its slot's offset belongs
    // to a different point in the world.
    float3 previousOffset = float3(0, 0, 0);
    // The previous VERDICT, for the dwell bands below. A scrolled-in probe has
    // no history of its own, so it starts OK and has to earn its way in — the
    // same rule as its offset.
    float previousClass = GI_CLASS_OK;
    if ((probe.flags & GI_UPDATE_FLAG_UNINITIALIZED) == 0u) {
        const float4 prev = giProbeOffset[texel];
        previousOffset = prev.xyz;
        previousClass = prev.w;
    }

    float3 c0, c1, c2;
    giRayRotation(giFrameIndex, c0, c1, c2);

    uint backfaces = 0;
    float3 backfaceSum = float3(0, 0, 0);
    for (uint r = 0; r < giRayCount; ++r) {
        if (giRayResults[tid.x * giRayCount + r].w >= 0.0) {
            continue;
        }
        float3 d = giSphericalFibonacci(r, giRayCount);
        float3 rayDir = normalize(c0 * d.x + c1 * d.y + c2 * d.z);
        // Unweighted sum: the resultant of the backface directions points at
        // the bulk of the occluding shell, and its negation is the shortest way
        // out. Weighting by 1/distance was tried in DDGI's literature and makes
        // a probe wedged in a corner chase the nearest face instead of the
        // opening.
        backfaceSum += rayDir;
        ++backfaces;
    }

    float fraction = giRayCount > 0u ? float(backfaces) / float(giRayCount) : 0.0;
    float3 offset = previousOffset;
    float classification = GI_CLASS_OK;
    bool relocated = false;

    // BOTH VERDICTS GET A DWELL BAND (gi_math.hpp carries the derivation).
    // `fraction` is a 64-ray binomial estimate whose sigma at the 0.75 knee is
    // ~0.054, so a bare threshold flips a probe's classification on consecutive
    // updates with nothing in the world changing — and one probe entering or
    // leaving GI_CLASS_INSIDE_GEOMETRY adds or removes a whole member of every
    // sampler's 8-probe cage, whose weights are then renormalized over the
    // survivors. That is why a single oscillating probe modulates a whole cell.
    const bool wasInside = previousClass == GI_CLASS_INSIDE_GEOMETRY;
    const bool wasRelocated = length(previousOffset) > 1e-4;
    const float relocateThreshold =
        wasRelocated ? GI_RELOCATE_BACKFACE_LEAVE : GI_RELOCATE_BACKFACE_FRACTION;
    const float insideThreshold =
        wasInside ? GI_INSIDE_BACKFACE_LEAVE : GI_INSIDE_BACKFACE_FRACTION;

    // IS THERE A DIRECTION TO GO AT ALL? `-normalize(backfaceSum)` is only
    // meaningful while the backfaces occupy PART of the sphere. A probe fully
    // enclosed in geometry flags every ray, and the resultant of the whole
    // rotated Fibonacci set is not zero — it is numerical residue of length
    // ~0.06-0.10 over 32-96 unit vectors, three orders of magnitude clear of any
    // "is it zero" epsilon, pointing wherever giRayRotation put it THIS frame.
    // Nudging along that is a random walk, and gi_sample.hlsli reads the walking
    // offset as the probe's position for both the Chebyshev distance and the
    // backface weight. gi_math.hpp's PROBE_RELOCATE_MIN_COHERENCE carries the
    // derivation of the threshold.
    const float len = length(backfaceSum);
    const bool coherent = backfaces > 0u && len > GI_RELOCATE_MIN_COHERENCE * float(backfaces);

    if (fraction > relocateThreshold) {
        if (coherent) {
            float3 away = -backfaceSum / max(len, 1e-6);
            offset = previousOffset + away * (0.25 * giProbeSpacing);
            float offsetLen = length(offset);
            if (offsetLen > giMaxOffset) {
                offset *= giMaxOffset / max(offsetLen, 1e-6);
            }
            relocated = true;
        }

        // Relocation cannot help this probe: either it has already gone as far
        // as it is allowed to go and is STILL mostly looking at backfaces, or
        // there is no direction to go in. Both are "inside something". Mark it;
        // the blend passes stop writing it and gi_sample.hlsli drops it from the
        // trilinear cage rather than letting it leak a wall's interior into the
        // street.
        //
        // The !coherent arm is not an optimization: with the guard above, an
        // enclosed probe's offset never reaches the clamp, so gating the verdict
        // on the clamp alone would leave it GI_CLASS_OK forever — at full weight
        // in every cage, contributing the black radiance a buried probe measures.
        if (fraction > insideThreshold &&
            (!coherent || length(previousOffset) >= giMaxOffset * 0.99)) {
            classification = GI_CLASS_INSIDE_GEOMETRY;
        }
    }

    giProbeOffset[texel] = float4(offset, classification);

    uint base = giCascadeIndex * 4u;
    if (relocated) {
        InterlockedAdd(giCounters[base + 0u], 1u);
    }
    if (classification == GI_CLASS_INSIDE_GEOMETRY) {
        InterlockedAdd(giCounters[base + 1u], 1u);
    }
}
