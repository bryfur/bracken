// gi_screen_probe_trace.comp.hlsl - stage B of the screen-probe gather: the
// rays the screen could not answer, resumed as SHORT BVH rays, INDIRECTLY
// dispatched over the compacted miss buffer.
//
// Phase R4b of docs/rt_lighting_plan.md §5.5.
//
// WAVEFRONT, NOT BRANCH. The naive shape is one dispatch over every probe texel
// with an `if (screenHit) return;` at the top, and it is exactly the shape §1's
// limit 6 of the plan already measured as the reflection pass's problem:
// scattered active lanes leave nearly every wave paying full traversal latency
// for one of its 32 threads. Compacting the misses first means every wave in
// this dispatch is a wave of live rays.
//
// SHORT rays, deliberately: GI_SP_WORLD_RAY_CLAMP (600 world units = the
// coarsest cascade's probe spacing). Beyond that distance the world probe field
// is the better estimator and the fallback below hands over to it — the screen
// probes buy SHARPNESS inside one coarse cell, not range.
//
// Hit shading is `rt_hit_shade.hlsli` (R3), the same recipe the probe field and
// the reflection trace use. There is exactly one answer in this engine to "a
// ray hit a triangle; how bright is it?".

#include "gi_screen_probe.hlsli"
#include "gi_probe.hlsli"

// The BVH structs first — the binding block needs them to declare its buffers.
#define RT_TRACE_TYPES
#include "rt_trace.hlsli"

// TriangleAttributes, phase 1.
#define RT_HIT_SHADE_TYPES
#include "rt_hit_shade.hlsli"

// ---------------------------------------------------------------------------
// Uniforms (set 2, b0) — mirrors gpu::GiScreenProbeTraceUniforms.
//
// The FIRST TEN MEMBERS are ShadowDataUniforms verbatim, under exactly the
// names shadow_common.hlsli requires, so this pass calls the forward pass's own
// calcShadow() rather than transliterating the standard-Z cascade math. Same
// argument as gi_probe_trace.comp.hlsl's, and docs/rt_lighting_plan.md §8 names
// the alternative as a trap.
// ---------------------------------------------------------------------------
[[vk::binding(0, 2)]]
cbuffer GiScreenProbeTraceUniforms : register(b0, space2) {
    float4x4 shadowLightViewProjection[4];
    float4 cascadeSplits;
    float4 cascadeTexelWorld;
    float4 cascadeDepthRange;
    float shadowMapResolution;
    float lightSize;
    float shadowEnabled;
    float lastActiveSplit;
    float fadeStart;
    float fadeEnd;
    float _shadowPad0;
    float _shadowPad1;

    float3 cameraPosition;
    float rayMaxDistance;
    float3 cameraForward;
    float normalBias;
    float3 sunDirection; // direction light TRAVELS
    float bounceScale;
    float3 sunColor;
    float useSkyCube;
    float3 ambientSky;
    float giEnabled;
    float3 ambientGround;
    float giAmbientFallback;
    int4 giCascadeBase[GI_CASCADE_COUNT];
    uint tlasNodeCount;
    uint giLightCount;
    uint giLightCapacity;
    uint missCapacity;
    int2 probeGrid;
    uint frameIndex;
    float horizonBlend;
};

// ---------------------------------------------------------------------------
// Bindings. SET 0 IS ORDER-SENSITIVE (shaders/README.md).
//
// BUDGET, counted because it is exactly at the ceiling: 4 sampled textures,
// 1 read-only storage texture, and EIGHT read-only storage buffers — TLAS
// nodes, TLAS instances, BLAS nodes, BLAS triangles, triangle attributes, the
// clustered-light byte buffer, the probe headers and the miss buffer. SDL3 GPU
// allows eight. That is why the miss COUNT lives in the miss buffer's own
// header rather than in the counters buffer: a ninth binding for one uint would
// not have fit, and the counters buffer is read-WRITE here anyway.
// ---------------------------------------------------------------------------

// --- sampled (4) ---
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D shadowAtlas : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerComparisonState shadowCmpSampler : register(s0, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D shadowAtlasRaw : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState shadowRawSampler : register(s1, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
TextureCube giSkyCube : register(t2, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
SamplerState giSkySampler : register(s2, space0);
[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
Texture2D<float4> giIrradiance : register(t3, space0);
[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
SamplerState giIrradianceSampler : register(s3, space0);

// --- read-only storage textures (1) ---
[[vk::binding(4, 0)]] Texture2D<uint> giProbeState : register(t4, space0);

// --- read-only storage buffers (8) ---
[[vk::binding(5, 0)]] StructuredBuffer<BvhNodeGpu> tlasNodes : register(t5, space0);
[[vk::binding(6, 0)]] StructuredBuffer<TlasInstance> tlasInstances : register(t6, space0);
[[vk::binding(7, 0)]] StructuredBuffer<BvhNodeGpu> blasNodes : register(t7, space0);
[[vk::binding(8, 0)]] StructuredBuffer<BvhTriangle> blasTriangles : register(t8, space0);
[[vk::binding(9, 0)]] StructuredBuffer<TriangleAttributes> blasAttributes : register(t9, space0);
[[vk::binding(10, 0)]] ByteAddressBuffer lightDataBuf : register(t10, space0);
[[vk::binding(11, 0)]] StructuredBuffer<GiScreenProbe> probes : register(t11, space0);
[[vk::binding(12, 0)]] ByteAddressBuffer missBuf : register(t12, space0);

// --- read-write (set 1) ---
[[vk::binding(0, 1)]] RWTexture2D<float4> radianceAtlas : register(u0, space1);
[[vk::binding(1, 1)]] RWByteAddressBuffer counters : register(u1, space1);

#define RT_TRACE_TRAVERSAL
#include "rt_trace.hlsli"

#define RT_HIT_SHADE_FUNCTIONS
#include "rt_hit_shade.hlsli"

groupshared uint gsWorldHits;
groupshared uint gsFallbacks;

/**
 * What a ray that reached the clamp, or missed the whole BVH, is worth.
 *
 * Sky above, world field below, blended across the horizon. The reasoning:
 * a ray still travelling after 6 m without hitting anything is either heading
 * for open sky (upward — nothing between it and the atmosphere) or into
 * geometry the clamp could not reach (downward and horizontal — the village
 * continues, the ray just ran out of budget). The sky function is exact for the
 * first case; the field is the engine's own estimate for the second.
 *
 * AN HONEST APPROXIMATION, twice over, and neither is hidden:
 *
 *  1. The field's octahedral tap in direction d is the COSINE-CONVOLVED mean
 *     radiance about d, not the radiance along d. Using it as a radiance is a
 *     low-pass of the truth — directionally right, angularly blurred. The bar
 *     here is energy-reasonable, not exact: this is the fallback of a fallback,
 *     it feeds an L1 projection that keeps four coefficients of the whole
 *     sphere, and it goes through a temporal EMA on the way.
 *  2. The horizon term is a heuristic, not a visibility test. A ray heading up
 *     from inside a covered market hall gets sky it cannot see. The probe
 *     field's own Chebyshev visibility is what answers that correctly, and the
 *     forward gather's weight floor is what hands such a pixel back to it.
 */
float3 screenProbeFallback(float3 anchor, float3 dir) {
    float3 sky = rtHitSkyRadiance(dir);
    if (giEnabled <= 0.5 || horizonBlend <= 0.0) {
        return sky;
    }
    bool covered;
    float3 field = rtHitProbeIrradiance(anchor, dir, covered);
    if (!covered) {
        return sky;
    }
    // d.y = +1 -> pure sky, d.y = -1 -> pure field.
    float below = saturate(0.5 - 0.5 * dir.y) * saturate(horizonBlend);
    return lerp(sky, field, below);
}

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex) {
    if (groupIndex == 0) {
        gsWorldHits = 0;
        gsFallbacks = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // The count the args pass dispatched against, clamped the same way. The
    // last group is partial by construction.
    uint count = min(missBuf.Load(0), missCapacity);
    bool live = tid.x < count;

    uint probeIndex = 0;
    uint texel = 0;
    float3 radiance = float3(0, 0, 0);
    float distance = 0.0;

    if (live) {
        uint4 record = missBuf.Load4(giSpMissRecordOffset(tid.x));
        probeIndex = record.x;
        texel = record.y;
        float resume = asfloat(record.z);

        GiScreenProbe probe = probes[probeIndex];
        float3 dir = giSpTexelDirection(int(texel), frameIndex);
        float3 anchor = probe.worldPosition + probe.normal * normalBias;
        // Resume where the screen gave up. The screen march already proved
        // there is nothing between the anchor and here, so re-tracing it would
        // be work whose answer is known — and worse, a self-hit risk against
        // the anchor's own surface.
        float3 origin = anchor + dir * resume;

        Hit hit = sceneClosest(origin, dir, rayMaxDistance);
        if (hit.valid) {
            TriangleAttributes attr = blasAttributes[hit.triIndex];
            float w = 1.0 - hit.bary.x - hit.bary.y;
            float3 nModel = normalize(attr.n0 * w + attr.n1 * hit.bary.x + attr.n2 * hit.bary.y);
            TlasInstance inst = tlasInstances[hit.instanceIndex];
            float3 N = normalize(mul((float3x3)transpose(inst.worldToModel), nModel));
            // A backface here is shaded on the FLIPPED normal, not written off:
            // unlike a world probe, a screen probe is not trying to work out
            // whether it is buried — its anchor came off the depth buffer, so it
            // is on a visible surface by construction. What it hit backwards is
            // a single-sided wall seen from inside, and the light bouncing off
            // that wall is real.
            if (dot(N, dir) > 0.0) {
                N = -N;
            }
            float3 P = origin + dir * hit.t;
            radiance = rtShadeHit(P, N, attr.baseColor.rgb, float2(tid.x & 63u, texel));
            distance = resume + hit.t;
            InterlockedAdd(gsWorldHits, 1u);
        } else {
            radiance = screenProbeFallback(anchor, dir);
            distance = resume + rayMaxDistance;
            InterlockedAdd(gsFallbacks, 1u);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (groupIndex == 0) {
        if (gsWorldHits > 0u) {
            counters.InterlockedAdd(4, gsWorldHits);
        }
        if (gsFallbacks > 0u) {
            counters.InterlockedAdd(8, gsFallbacks);
        }
    }

    if (live) {
        int2 probeCoord = giSpCoord(int(probeIndex), probeGrid);
        radianceAtlas[giSpRadianceTexel(probeCoord, int(texel))] = float4(radiance, distance);
    }
}
