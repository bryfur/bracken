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
// HIT SHADING v1, the recipe the plan specifies:
//
//   L_out = albedo_flat * ( sunColor * max(0, N.L) * sunVisibility
//                         + local point/spot lights, range-windowed
//                         + pi * previous-frame probe irradiance ) / pi
//         | on a miss: the sky radiance in the ray direction
//
// Three of those terms are things this engine could already answer and no other
// renderer's probe pass gets for free:
//
//  * `albedo_flat` is TriangleAttributes::baseColor, baked from the real albedo
//    texture's centroid-UV sample at load. SDL3 GPU has no bindless, so a ray
//    hit cannot sample a texture — the flat table is the answer, and it was
//    already built for reflections.
//  * `sunVisibility` is a CSM ATLAS FETCH, not a second ray. The cascades are
//    already rendered this frame, so the probe field re-lights itself for free
//    whenever the sun moves: next frame's probe rays see the new atlas, and the
//    response lag is the accumulation hysteresis alone.
//  * the bounce term is LAST FRAME'S OWN FIELD sampled at the hit. That is the
//    standard DDGI recursion and it is what makes the result an infinite-bounce
//    solution out of a single-bounce trace: update n contains n-1 bounces, so
//    the series converges to sum(albedo^k) = 1/(1-albedo) — bounded for any
//    physical albedo below 1, which is why it converges rather than ringing.
//    `bounceScale` below 1 shortens the series if a scene is bright enough to
//    make that a problem.
//
// UNITS. The atlas stores COSINE-WEIGHTED MEAN INCOMING RADIANCE, i.e.
// irradiance/pi — deliberately the same convention as the IBL irradiance cube
// (`ibl_irradiance.frag.hlsl` stores (1/pi) * integral of L*cos), which is what
// lets gi_sample.hlsli substitute one for the other without a scale factor
// appearing anywhere. That is why the shading below divides by pi once and multiplies the
// sampled bounce by pi once.

#include "gi_probe.hlsli"

// The BVH structs first — the binding block below needs them to declare its
// storage buffers. See rt_trace.hlsli for the two-phase include contract.
#define RT_TRACE_TYPES
#include "rt_trace.hlsli"

// Only reflections and this pass bind per-triangle attributes; shadows must not
// pay for them. Same declaration as rt_reflection.comp.hlsl.
struct TriangleAttributes {
    float3 n0;
    float pad0;
    float3 n1;
    float pad1;
    float3 n2;
    float pad2;
    float4 baseColor;
};

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

// calcShadow() + selectShadowCascade() over the cbuffer members declared above.
// Included AFTER the bindings for the same reason rt_trace.hlsli's second phase
// is: it reads shadowAtlas/shadowCmpSampler by name.
#include "shadow_common.hlsli"

// ---------------------------------------------------------------------------
// Sky
// ---------------------------------------------------------------------------

/// Radiance of a ray that escaped the scene.
///
/// The IBL sky-capture cube is the SAME radiance function the skybox draws
/// (sky_common.hlsli, shared by skybox.frag and sky_capture.frag), so probes
/// carry the real time-of-day sky WITH occlusion — blue in a courtyard, dark
/// under an eave, which is exactly what a flat `ambient_sky` constant could
/// never do. When IBL is off the cube slot holds the engine's cube dummy, so
/// `useSkyCube` gates it and the EnvironmentState hemisphere takes over.
float3 giSkyRadiance(float3 dir) {
    if (useSkyCube > 0.5) {
        return giSkyCube.SampleLevel(giSkySampler, dir, 0).rgb;
    }
    return lerp(ambientGround, ambientSky, saturate(dir.y * 0.5 + 0.5));
}

// ---------------------------------------------------------------------------
// Previous-frame probe irradiance — the bounce
// ---------------------------------------------------------------------------

uint giLoadState(int cascade, int slot) {
    return giProbeState.Load(int3(giProbeStateTexel(slot, cascade), 0)).x;
}

float3 giLoadIrradiance(int cascade, int slot, float3 dir) {
    // Payload texel centres live at origin + border + oct * payload, so oct = 0
    // lands exactly on the seam between the border texel and payload texel 0 —
    // which is the whole reason the border exists, and why this is a bilinear
    // sample rather than a Load.
    float2 texel = float2(giProbeAtlasOrigin(slot, GI_IRRADIANCE_TILE, cascade)) + float(GI_BORDER) +
                   giOctEncode(dir) * float(GI_IRRADIANCE_PAYLOAD);
    return giIrradiance.SampleLevel(giIrradianceSampler, texel / giAtlasSize(GI_IRRADIANCE_TILE), 0).rgb;
}

/// Cosine-weighted mean incoming radiance at `P` for a surface facing `N`,
/// from the FINEST cascade that covers `P` with a one-probe margin.
///
/// Trilinear over the 8 surrounding probes, each weighted by a smooth
/// backface term ((dot + 1)/2)^2 so a probe on the far side of the surface
/// contributes almost nothing. R1b deliberately does NOT apply the Chebyshev
/// visibility test here: this is a secondary bounce feeding a heavily filtered
/// EMA, the visibility moments are themselves still converging while the field
/// fills, and the leak it prevents is one bounce deep. The PRIMARY sampler
/// (gi_sample.hlsli) — the one whose leaks a player would see — applies it.
float3 giSampleIrradiance(float3 P, float3 N) {
    for (int c = 0; c < GI_CASCADE_COUNT; ++c) {
        float sp = giCascadeSpacing(c);
        float3 local = P / sp - float3(giCascadeBase[c].xyz);
        // One-probe margin: the 8-tap cage must lie wholly inside the window,
        // or a tap wraps around the torus and reads the opposite side of the
        // world.
        if (any(local < 1.0) || any(local > float3(GI_GRID_X - 2, GI_GRID_Y - 2, GI_GRID_Z - 2))) {
            continue;
        }
        float3 baseF = floor(local);
        float3 f = local - baseF;
        int3 b = int3(baseF);

        float3 sum = float3(0, 0, 0);
        float wsum = 0.0;
        [unroll]
        for (int i = 0; i < 8; ++i) {
            int3 o = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            int3 absolute = giCascadeBase[c].xyz + b + o;
            int slot = giWrappedProbeIndex(absolute);
            if (giLoadState(c, slot) == GI_PROBE_UNINITIALIZED) {
                continue;
            }
            float3 tri = lerp(1.0 - f, f, float3(o));
            float3 toProbe = float3(absolute) * sp - P;
            float len = length(toProbe);
            float wrapped = len > 1e-4 ? (dot(toProbe / len, N) + 1.0) * 0.5 : 1.0;
            // The +0.02 floor keeps a probe cage from collapsing to zero weight
            // on a surface that faces away from all eight, which would leave
            // the hit unlit rather than dimly lit.
            float w = tri.x * tri.y * tri.z * (wrapped * wrapped + 0.02);
            sum += w * giLoadIrradiance(c, slot, N);
            wsum += w;
        }
        return wsum > 1e-6 ? sum / wsum : float3(0, 0, 0);
    }
    return float3(0, 0, 0);
}

// ---------------------------------------------------------------------------
// Local (point/spot) lights
// ---------------------------------------------------------------------------
//
// The byte layout is render::lighting's, and the offsets below are the same
// ones cluster_lighting.hlsli hardcodes:
//   [u32 headerCount][u32 pointCount][u32 spotCount][u32 pad]
//   [LightHeader * capacity][PointLight * capacity][SpotLight * capacity]

uint giPointLightsBase() { return 16u + giLightCapacity * 8u; }
uint giSpotLightsBase() { return 16u + giLightCapacity * (8u + 48u); }

/// Smooth windowed inverse-square falloff. Operation-identical to
/// `pointFalloff` in clusters.hlsli — a probe and a pixel must agree about how
/// far a lamp reaches, or the bounce brightens at a distance the direct term
/// has already cut off.
float giPointFalloff(float dist, float radius) {
    float r = saturate(dist / max(radius, 1e-4));
    float r2 = r * r;
    float win = saturate(1.0 - r2 * r2);
    return (win * win) / max(dist * dist, 1e-4);
}

/// IRRADIANCE (not radiance) at `P` from every packet light in range.
/// No shadow test: local-light ray shadows are R2, and an unshadowed lamp in a
/// secondary bounce is a far smaller error than no lamp at all.
float3 giLocalLights(float3 P, float3 N) {
    float3 acc = float3(0, 0, 0);
    uint count = min(giLightCount, 256u);
    for (uint i = 0; i < count; ++i) {
        uint2 header = lightDataBuf.Load2(16u + i * 8u);
        if (header.x == 0u) {
            uint base = giPointLightsBase() + header.y * 48u;
            uint4 a = lightDataBuf.Load4(base);
            uint4 b = lightDataBuf.Load4(base + 16u);
            float3 position = asfloat(a.xyz);
            float radius = asfloat(a.w);
            float3 toLight = position - P;
            float dist = length(toLight);
            if (dist > radius || dist < 1e-4) {
                continue;
            }
            float3 L = toLight / dist;
            float ndotl = saturate(dot(N, L));
            if (ndotl <= 0.0) {
                continue;
            }
            acc += asfloat(b.xyz) * asfloat(b.w) * giPointFalloff(dist, radius) * ndotl;
        } else {
            uint base = giSpotLightsBase() + header.y * 64u;
            uint4 a = lightDataBuf.Load4(base);
            uint4 b = lightDataBuf.Load4(base + 16u);
            uint4 c = lightDataBuf.Load4(base + 32u);
            uint4 d = lightDataBuf.Load4(base + 48u);
            float3 position = asfloat(a.xyz);
            float radius = asfloat(a.w);
            float3 toLight = position - P;
            float dist = length(toLight);
            if (dist > radius || dist < 1e-4) {
                continue;
            }
            float3 L = toLight / dist;
            float ndotl = saturate(dot(N, L));
            if (ndotl <= 0.0) {
                continue;
            }
            float cone = smoothstep(asfloat(c.w), asfloat(b.w), dot(asfloat(b.xyz), -L));
            if (cone <= 0.0) {
                continue;
            }
            acc += asfloat(c.xyz) * asfloat(d.x) * giPointFalloff(dist, radius) * ndotl * cone;
        }
    }
    return acc;
}

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
        radiance = giSkyRadiance(dir);
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
            float3 toSun = -sunDirection;
            float ndotl = saturate(dot(N, toSun));

            // calcShadow selects a cascade by VIEW depth, so the hit needs one.
            float viewDepth = dot(P - cameraPosition, cameraForward);
            // The screenPos argument only rotates the PCSS poisson disk, which
            // shadowEnabled = 1 (PCF) never reaches; feeding it the ray/probe
            // pair keeps it deterministic either way.
            float vis = calcShadow(P, N, toSun, viewDepth, float2(tid.xy));

            // IRRADIANCE at the hit. The bounce is stored as mean radiance, so
            // it becomes irradiance by multiplying by pi (see the units note at
            // the top of the file).
            float3 E = sunColor * ndotl * vis;
            E += giLocalLights(P, N);
            E += GI_PI * bounceScale * giSampleIrradiance(P + N * normalBias, N);

            // Lambertian outgoing radiance.
            radiance = attr.baseColor.rgb * E / GI_PI;
            distance = hit.t;
        }
    }

    giRayResults[tid.y * giRayCount + tid.x] = float4(radiance, distance);
}
