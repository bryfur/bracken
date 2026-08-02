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
    if ((probe.flags & GI_UPDATE_FLAG_UNINITIALIZED) == 0u) {
        previousOffset = giProbeOffset[texel].xyz;
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

    if (fraction > GI_RELOCATE_BACKFACE_FRACTION) {
        float len = length(backfaceSum);
        float3 away = len > 1e-4 ? -backfaceSum / len : float3(0.0, 1.0, 0.0);
        offset = previousOffset + away * (0.25 * giProbeSpacing);
        float offsetLen = length(offset);
        if (offsetLen > giMaxOffset) {
            offset *= giMaxOffset / max(offsetLen, 1e-6);
        }
        relocated = true;

        // Already nudged as far as it is allowed to go and STILL mostly looking
        // at backfaces: this probe is inside something, and no offset inside
        // half a cell will get it out. Mark it; the blend passes stop writing
        // it and gi_sample.hlsli drops it from the trilinear cage rather than
        // letting it leak a wall's interior into the street.
        if (fraction > GI_INSIDE_BACKFACE_FRACTION && length(previousOffset) >= giMaxOffset * 0.99) {
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
