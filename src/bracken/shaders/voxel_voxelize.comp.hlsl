// voxel_voxelize.comp.hlsl - fill one cascade's ALBEDO+OPACITY volume from the
// scene's triangles. Phase RG1 of the raster GI path.
//
// WHY THIS IS A COMPUTE SHADER AND NOT A RASTER PASS. Classic VXGI voxelizes by
// rasterizing the scene with an orthographic camera and having the fragment
// shader image-store into the volume, using a GEOMETRY shader to swizzle each
// triangle onto its dominant axis (so a triangle edge-on to the chosen axis
// still covers texels) and CONSERVATIVE rasterization (so a triangle that
// clips a voxel's corner is not dropped). SDL3 GPU exposes neither. Doing it in
// compute sidesteps both: a thread owns a triangle, so it decides its own
// coverage and no rasterizer has to be persuaded to be conservative.
//
// COVERAGE BY AREA-PROPORTIONAL SURFACE SAMPLING, not by exact triangle-box
// overlap. The textbook alternative is a separating-axis test against every
// voxel in the triangle's AABB, which is exact but costs the AABB's VOLUME per
// triangle — and terrain triangles here span hundreds of voxels, so the cost is
// dominated by the largest primitive in the scene rather than by the average.
// Sampling the surface at a density of a few samples per voxel area is O(area)
// instead of O(volume), degrades gracefully (a missed voxel is a slightly thin
// wall, not a hole in the world), and at a 25-unit voxel is indistinguishable
// from exact. What it cannot do is represent a surface thinner than a voxel as
// opaque, which is the same limitation the exact test has.
//
// ATOMIC MAX RATHER THAN AVERAGE. Several triangles land in one voxel and the
// order they arrive in is undefined, so a running average would be
// nondeterministic frame to frame — and this feeds a probe field whose whole
// value is temporal stability. `max` on the packed value is order-independent
// and therefore reproducible, and it biases a voxel toward its brightest,
// most-opaque contributor, which is the right way to lose information when a
// wall and a leaf share a cell.

#include "voxel_common.hlsli"

[[vk::binding(0, 2)]]
cbuffer VoxelVoxelizeUniforms : register(b0, space2) {
    /// Which cascade this dispatch fills.
    int voxelCascade;
    /// Its base corner in whole voxels — the window every write is tested
    /// against, because torus addressing would otherwise happily wrap a
    /// triangle from the far side of the map onto a texel near the camera.
    int3 voxelBase;
    /// Total (instance, triangle) pairs, i.e. the dispatch's flat extent.
    uint voxelWorkCount;
    /// Instances in `voxelInstances`, for the binary search below.
    uint voxelInstanceCount;
    uint _voxelPad0;
    uint _voxelPad1;
};

/// One model-space triangle plus the material proxy the RT path already bakes
/// per triangle. Mirrors `render::gi::VoxelTriangle`.
struct VoxelTriangle {
    float3 v0;
    uint packedAlbedo;   // RGBA8, the baked per-triangle base colour
    float3 e1;
    uint _triPad0;
    float3 e2;
    float _triPad1;
};

/// Mirrors `render::gi::VoxelInstance`. `firstTriangle`/`triangleCount` name a
/// window of the shared triangle buffer, exactly as a BLAS does, so one model's
/// geometry is stored once however many times it is placed.
struct VoxelInstance {
    float4x4 modelToWorld;
    uint firstTriangle;
    uint triangleCount;
    /// Prefix sum of `triangleCount` over preceding instances — what turns the
    /// flat dispatch index into an (instance, triangle) pair without a second
    /// indirection buffer.
    uint workOffset;
    uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<VoxelTriangle> voxelTriangles : register(t0, space0);
[[vk::binding(1, 0)]] StructuredBuffer<VoxelInstance> voxelInstances : register(t1, space0);

[[vk::image_format("r32ui")]]
[[vk::binding(0, 1)]] RWTexture3D<uint> voxelAlbedo : register(u0, space1);

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------
//
// The volume is declared r32ui rather than rgba8 because the only
// order-independent write available is an integer atomic, and `max` over a
// packed RGBA8 with OPACITY IN THE HIGH BYTE orders by opacity first — so the
// winner of a contested voxel is the most opaque contributor, and colour only
// breaks ties among equally opaque ones. Reading it back as colour is a shift
// and a mask; the injection pass does that once per voxel.

uint voxelPack(float3 albedo, float opacity) {
    uint3 c = uint3(saturate(albedo) * 255.0 + 0.5);
    uint a = uint(saturate(opacity) * 255.0 + 0.5);
    return (a << 24) | (c.b << 16) | (c.g << 8) | c.r;
}

// ---------------------------------------------------------------------------
// Work decomposition
// ---------------------------------------------------------------------------

/// Flat work index -> which instance owns it. Binary search over the prefix
/// sums, which is log2(instances) — ~12 steps at the playground's 2570 — and
/// costs far less than the per-instance dispatch it replaces would waste on
/// threads that have no triangle.
uint voxelFindInstance(uint work) {
    uint lo = 0;
    uint hi = voxelInstanceCount - 1;
    [loop] while (lo < hi) {
        uint mid = (lo + hi + 1) / 2;
        if (voxelInstances[mid].workOffset <= work) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const uint work = tid.x;
    if (work >= voxelWorkCount || voxelInstanceCount == 0) {
        return;
    }

    const uint instanceIndex = voxelFindInstance(work);
    const VoxelInstance inst = voxelInstances[instanceIndex];
    const uint local = work - inst.workOffset;
    if (local >= inst.triangleCount) {
        return; // padding at the tail of the last group
    }

    const VoxelTriangle tri = voxelTriangles[inst.firstTriangle + local];

    // Model space -> world space, once, for the three corners.
    const float3 p0 = mul(inst.modelToWorld, float4(tri.v0, 1.0)).xyz;
    const float3 p1 = mul(inst.modelToWorld, float4(tri.v0 + tri.e1, 1.0)).xyz;
    const float3 p2 = mul(inst.modelToWorld, float4(tri.v0 + tri.e2, 1.0)).xyz;

    // Reject before doing any work: most of the map is outside any one
    // cascade's window, and this is the test that makes voxelizing a whole
    // village affordable at the finest cascade.
    const float vs = voxelSize(voxelCascade);
    const float3 lo = min(p0, min(p1, p2));
    const float3 hi = max(p0, max(p1, p2));
    const int3 loCoord = voxelCoordAt(voxelCascade, lo);
    const int3 hiCoord = voxelCoordAt(voxelCascade, hi);
    if (any(hiCoord < voxelBase) || any(loCoord >= voxelBase + VOXEL_RESOLUTION)) {
        return;
    }

    const float3 albedo = float3(float((tri.packedAlbedo >> 0) & 0xFFu),
                                 float((tri.packedAlbedo >> 8) & 0xFFu),
                                 float((tri.packedAlbedo >> 16) & 0xFFu)) /
                          255.0;
    const uint packed = voxelPack(albedo, 1.0);

    // SAMPLE COUNT FROM AREA. One sample per voxel-face area, floored at 1 (a
    // triangle smaller than a voxel still needs to exist) and capped so a single
    // enormous terrain triangle cannot stall the whole dispatch. The cap is a
    // real limit and it is why coverage is documented as approximate: a triangle
    // spanning more than `VOXEL_MAX_SAMPLES` voxel-areas is sampled sparsely and
    // may leave gaps, which read as thin spots in the occlusion rather than as
    // holes because neighbouring triangles overlap it.
    const float3 cross_e = cross(p1 - p0, p2 - p0);
    const float area = 0.5 * length(cross_e);
    const float voxelArea = vs * vs;
    const uint MAX_SAMPLES = 4096u;
    const uint samples = clamp(uint(ceil(4.0 * area / max(voxelArea, 1e-6))), 1u, MAX_SAMPLES);

    // A fixed low-discrepancy sequence, warped onto the triangle by the
    // uniform-area sqrt map. Deterministic, which the probe field downstream
    // requires: a voxelization that differed run to run would make the smoke
    // gate meaningless.
    for (uint s = 0; s < samples; ++s) {
        const float fs = float(s) + 0.5;
        const float u = frac(fs * 0.7548776662466927);
        const float v = frac(fs * 0.5698402909980532);
        const float su = sqrt(u);
        const float3 b = float3(1.0 - su, su * (1.0 - v), su * v);
        const float3 world = p0 * b.x + p1 * b.y + p2 * b.z;

        const int3 coord = voxelCoordAt(voxelCascade, world);
        if (!voxelInWindow(coord, voxelBase)) {
            continue;
        }
        // Torus wrap AFTER the window test, never instead of it.
        InterlockedMax(voxelAlbedo[voxelWrap(coord)], packed);
    }
}
