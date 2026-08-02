// gi_probe_trace.comp.hlsl - one thread per (probe scheduled this frame, ray).
//
// Phase R1b of docs/rt_lighting_plan.md §5.2: the probe field stops being an
// empty structure and starts holding radiance. Since R1c the forward pass and
// the fog march sample what this writes (shaders/gi_sample.hlsli), so an error
// here is now visible rather than merely stored.
//
// THE BUDGET IS THE POINT. Every other traced signal in this engine costs rays
// per PIXEL; this one costs rays per FRAME, fixed, chosen by GiQuality and
// spread over the probes by a round-robin rota (render/gi/probe_schedule.hpp).
// That is the single property that makes diffuse GI affordable on hardware that
// cannot trace one ray per pixel.
//
// HIT SHADING LIVES IN `rt_hit_shade.hlsli` (R3). It used to live here, and the
// reflection pass had its own worse copy; the recipe, the units argument and
// the reasoning for each term all moved to that file verbatim. What is left
// below is this pass's own business: which probe, which direction, and what a
// BACKFACE hit means to the classifier.
//
// UNITS. The atlas stores COSINE-WEIGHTED MEAN INCOMING RADIANCE, i.e.
// irradiance/pi — deliberately the same convention as the IBL irradiance cube
// (`ibl_irradiance.frag.hlsl` stores (1/pi) * integral of L*cos), which is what
// lets gi_sample.hlsli substitute one for the other without a scale factor
// appearing anywhere.

#include "gi_probe.hlsli"

// The BVH structs first — the binding block below needs them to declare its
// storage buffers. See rt_trace.hlsli for the two-phase include contract.
#define RT_TRACE_TYPES
#include "rt_trace.hlsli"

// TriangleAttributes + the hit-shading model, phase 1 (the struct alone). The
// binding block below needs the struct to declare its attribute buffer.
#define RT_HIT_SHADE_TYPES
#include "rt_hit_shade.hlsli"

// One probe scheduled this frame. Mirrors gpu::GiProbeUpdateRecord (32 B).
struct GiProbeUpdate {
    float3 worldPosition;
    uint slot;
    uint flags;
    float3 _pad;
};
#define GI_UPDATE_FLAG_UNINITIALIZED 1u

// ---------------------------------------------------------------------------
// Uniforms (set 2, b0) — mirrors gpu::GiProbeTraceUniforms.
//
// The FIRST TEN MEMBERS are ShadowDataUniforms verbatim, under exactly the
// names shadow_common.hlsli requires, so this pass can include that file and
// call calcShadow() instead of reimplementing the cascade math. The atlas is
// deliberately STANDARD-Z while the scene is reversed-Z, and
// docs/rt_lighting_plan.md §8 names reimplementing it here as a trap; sharing
// the code is the only formulation in which the two paths cannot drift.
// ---------------------------------------------------------------------------
[[vk::binding(0, 2)]]
cbuffer GiProbeTraceUniforms : register(b0, space2) {
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
    float probeSpacing;
    float3 sunDirection;   // direction light TRAVELS
    float normalBias;
    float3 sunColor;
    float bounceScale;
    float3 ambientSky;
    float useSkyCube;
    float3 ambientGround;
    float maxOffset;
    int4 giCascadeBase[GI_CASCADE_COUNT];
    uint giCascadeIndex;
    uint giRayCount;
    uint giProbeCount;
    uint tlasNodeCount;
    uint giLightCount;
    uint giLightCapacity;
    uint giFrameIndex;
    uint _giPad0;
    // rt_hit_shade.hlsli's two ambient flags, pinned by ProbeVolume::trace to
    // (1, 0): the probe field is by definition available here, and a probe's
    // own bounce term must NOT fall back to the hemisphere where the field has
    // no answer — see rtHitAmbient for why that would double-count sky and
    // break the bounce series.
    float giEnabled;
    float giAmbientFallback;
    float _giPad1;
    float _giPad2;
};

// ---------------------------------------------------------------------------
// Bindings. SET 0 IS ORDER-SENSITIVE: sampled textures, then READ-ONLY storage
// textures, then READ-ONLY storage buffers, each run numbered contiguously
// (shaders/README.md). Declaring them in any other order compiles, reflects and
// validates cleanly and then binds the wrong resource to every slot.
// ---------------------------------------------------------------------------

// --- sampled (4) ---
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D shadowAtlas : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerComparisonState shadowCmpSampler : register(s0, space0);
// The PCSS blocker search's raw pair. `shadowEnabled` is pinned to 1 (PCF) for
// probe rays, so these are never sampled — but shadow_common.hlsli names them,
// and the CSM atlas is bound here so the slot is valid rather than dangling.
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D shadowAtlasRaw : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState shadowRawSampler : register(s1, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
TextureCube giSkyCube : register(t2, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
SamplerState giSkySampler : register(s2, space0);
// The BANDED irradiance atlas: every cascade in one texture, cascade c at rows
// [c * GI_ATLAS_ROWS, (c+1) * GI_ATLAS_ROWS). R1c's repack — see gi_math.hpp.
[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
Texture2D<float4> giIrradiance : register(t3, space0);
[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
SamplerState giIrradianceSampler : register(s3, space0);

// --- read-only storage textures (2) ---
// Per-probe relocation offset (xyz) + classification (w), all cascades banded.
[[vk::binding(4, 0)]] Texture2D<float4> giProbeOffset : register(t4, space0);
// Per-probe scheduling state, all cascades banded, so the bounce sample can
// reject a probe whose storage still holds the previous occupant of its slot.
[[vk::binding(5, 0)]] Texture2D<uint> giProbeState : register(t5, space0);

// --- read-only storage buffers (7) ---
[[vk::binding(6, 0)]] StructuredBuffer<BvhNodeGpu> tlasNodes : register(t6, space0);
[[vk::binding(7, 0)]] StructuredBuffer<TlasInstance> tlasInstances : register(t7, space0);
[[vk::binding(8, 0)]] StructuredBuffer<BvhNodeGpu> blasNodes : register(t8, space0);
[[vk::binding(9, 0)]] StructuredBuffer<BvhTriangle> blasTriangles : register(t9, space0);
[[vk::binding(10, 0)]] StructuredBuffer<TriangleAttributes> blasAttributes : register(t10, space0);
// The clustered-lighting buffer, read WITHOUT the cluster grid: a probe-ray hit
// is a world-space point and the froxel grid is screen-space, so the only
// correct traversal here is the flat list. Fine at probe-ray counts — the loop
// is <= 256 range-rejected tests, against a BVH descent that costs far more.
[[vk::binding(11, 0)]] ByteAddressBuffer lightDataBuf : register(t11, space0);
[[vk::binding(12, 0)]] StructuredBuffer<GiProbeUpdate> giUpdates : register(t12, space0);

// --- read-write (set 1) ---
// rgb = outgoing radiance at the hit (or the sky on a miss); w = hit distance,
// NEGATED when the ray struck a backface. That sign is the classification
// signal: a probe buried in a wall sees the wall's inside faces, and nothing
// else in the pipeline can tell the difference between "inside" and "dark".
[[vk::binding(0, 1)]] RWStructuredBuffer<float4> giRayResults : register(u0, space1);

// The traversal walks bind to the buffers above by name.
#define RT_TRACE_TRAVERSAL
#include "rt_trace.hlsli"

// Hit shading, phase 2: rtHitSkyRadiance / rtShadeHit over the resources and
// cbuffer members declared above. It includes shadow_common.hlsli itself, so
// the calcShadow() this pass's hits get is the same one the forward pass calls.
// Included AFTER the bindings for the same reason rt_trace.hlsli's second phase
// is: it reads shadowAtlas/giIrradiance/lightDataBuf by name.
#define RT_HIT_SHADE_FUNCTIONS
#include "rt_hit_shade.hlsli"

// ---------------------------------------------------------------------------

[numthreads(32, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= giRayCount || tid.y >= giProbeCount) {
        return;
    }
    GiProbeUpdate probe = giUpdates[tid.y];

    // A probe that was uninitialized when it was scheduled has a stale offset
    // in its slot — whatever the previous occupant was nudged by. Start it at
    // the lattice position; the classify pass writes it a fresh one.
    float3 offset = float3(0, 0, 0);
    if ((probe.flags & GI_UPDATE_FLAG_UNINITIALIZED) == 0u) {
        offset = giProbeOffset.Load(int3(giProbeStateTexel(int(probe.slot), int(giCascadeIndex)), 0)).xyz;
    }
    float3 origin = probe.worldPosition + offset;
    float3 dir = giRayDirection(tid.x, giRayCount, giFrameIndex);

    Hit hit = sceneClosest(origin, dir, rayMaxDistance);

    float3 radiance;
    float distance;
    if (!hit.valid) {
        radiance = rtHitSkyRadiance(dir);
        distance = rayMaxDistance;
    } else {
        TriangleAttributes attr = blasAttributes[hit.triIndex];
        float w = 1.0 - hit.bary.x - hit.bary.y;
        float3 nModel = normalize(attr.n0 * w + attr.n1 * hit.bary.x + attr.n2 * hit.bary.y);
        // Vertex attributes are model-space; lighting is world-space. The
        // inverse-transpose of the model matrix is transpose(worldToModel),
        // exactly as rt_reflection.comp.hlsl does it.
        TlasInstance inst = tlasInstances[hit.instanceIndex];
        float3 N = normalize(mul((float3x3)transpose(inst.worldToModel), nModel));

        if (dot(N, dir) > 0.0) {
            // BACKFACE. Do not shade it — a probe inside a wall would otherwise
            // pick up the wall's own lit exterior and read as brightly lit. The
            // negative distance is what the classify pass counts.
            radiance = float3(0, 0, 0);
            distance = -hit.t;
        } else {
            float3 P = origin + dir * hit.t;
            // The noise argument only rotates calcShadow's PCSS poisson disk,
            // which shadowEnabled = 1 (PCF) never reaches; feeding it the
            // ray/probe pair keeps it deterministic either way.
            radiance = rtShadeHit(P, N, attr.baseColor.rgb, float2(tid.xy));
            distance = hit.t;
        }
    }

    giRayResults[tid.y * giRayCount + tid.x] = float4(radiance, distance);
}
