// cluster_lighting.hlsli - bindings + accumulator for clustered forward+ lights.
// Including shader must NOT also declare ClusterParams cbuffer / SSBO bindings;
// these slots (set 3, b2 + four set-2 bindings after the sampled textures)
// are reserved for clustered lighting.
//
// SDL GPU lays out fragment set 2 as sampled textures, then storage textures,
// then storage buffers. The SSBO bindings therefore start after BOTH: define
// CLUSTER_SSBO_BINDING_BASE to the including shader's fragment sampler count
// PLUS its storage-texture count BEFORE including this file, and
// CLUSTER_SAMPLER_COUNT to the sampler count alone when the two differ.
// SDL_BindGPUFragmentStorageBuffers(pass, 0, bufs, 4) then maps slots 0..3
// onto these bindings.
//
// This header also owns the LOCAL (point/spot) shadow bindings: the shadow
// atlas + comparison sampler pair is expected at the including shader's LAST
// sampler slot (CLUSTER_SAMPLER_COUNT - 1), and the per-tile slot data
// rides the 4th storage buffer. Local shadow maps are standard-Z perspective
// depth (near -> 0, far -> 1, LESS_OR_EQUAL compare) rendered by
// render::lighting::LocalShadowMap — see local_shadows.hpp for the CPU side.
//
// CLUSTER_SAMPLER_COUNT AND CLUSTER_SSBO_BINDING_BASE USED TO BE THE SAME
// NUMBER, and stopped being so in R5c. Fragment set 2 is laid out as sampled
// textures, then STORAGE textures, then storage buffers, so a shader that
// declares storage textures pushes its storage buffers up past its sampler
// count — and the Full-DI tier declares two (local_di.hlsli), because these
// shaders sit at SDL's 16-sampler ceiling and there is no seventeenth to be
// had. The atlas slot follows the SAMPLERS and the SSBO base follows the
// BUFFERS, which are now two different numbers. A shader with no storage
// textures may still define only CLUSTER_SSBO_BINDING_BASE; the fallback below
// keeps that spelling meaning what it always meant.

#ifndef CLUSTER_LIGHTING_HLSLI
#define CLUSTER_LIGHTING_HLSLI

#include "clusters.hlsli"

#ifndef CLUSTER_SSBO_BINDING_BASE
#error "Define CLUSTER_SSBO_BINDING_BASE (= fragment storage-buffer base) before including cluster_lighting.hlsli"
#endif

#ifndef CLUSTER_SAMPLER_COUNT
#define CLUSTER_SAMPLER_COUNT CLUSTER_SSBO_BINDING_BASE
#endif

[[vk::binding(2, 3)]]
cbuffer ClusterParamsCB { ClusterParams clusterParams; };

[[vk::binding(CLUSTER_SSBO_BINDING_BASE + 0, 2)]]
ByteAddressBuffer lightDataBuf;

[[vk::binding(CLUSTER_SSBO_BINDING_BASE + 1, 2)]]
StructuredBuffer<uint2> clusterOffsetsBuf;

[[vk::binding(CLUSTER_SSBO_BINDING_BASE + 2, 2)]]
StructuredBuffer<uint> lightIndicesBuf;

// ---------------------------------------------------------------------------
// Local (point/spot) shadows
// ---------------------------------------------------------------------------

// D32 local-shadow atlas (8x8 tiles) at the including shader's last sampler
// slot, sampled through a LESS_OR_EQUAL comparison sampler (each
// SampleCmpLevelZero tap is a free 2x2 hardware PCF).
[[vk::combinedImageSampler]][[vk::binding(CLUSTER_SAMPLER_COUNT - 1, 2)]]
Texture2D localShadowAtlas;
[[vk::combinedImageSampler]][[vk::binding(CLUSTER_SAMPLER_COUNT - 1, 2)]]
SamplerComparisonState localShadowCmpSampler;

// Mirrors render::lighting::LocalShadowSlotGPU (local_shadows.hpp), 96 bytes.
struct LocalShadowSlot {
    float4x4 viewProj; // world -> light clip (standard-Z perspective)
    float4 rect;       // atlas UV: xy = tile origin, zw = tile scale
    float4 params;     // x = world texel per unit distance (2*tan(fov/2)/tileRes)
                       // y = tile-UV clamp inset for PCF, z = near plane,
                       // w = atlas texel size (1 / atlas resolution)
};

#define LOCAL_SHADOW_MAX_SLOTS 64u

[[vk::binding(CLUSTER_SSBO_BINDING_BASE + 3, 2)]]
StructuredBuffer<LocalShadowSlot> localShadowSlots;

// Cube-face index from the light->fragment vector (dominant axis):
// 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z. CPU mirror: lighting::local_shadow_cube_face.
uint localShadowCubeFace(float3 d) {
    float3 ad = abs(d);
    if (ad.x >= ad.y && ad.x >= ad.z) return d.x > 0.0 ? 0u : 1u;
    if (ad.y >= ad.z) return d.y > 0.0 ? 2u : 3u;
    return d.z > 0.0 ? 4u : 5u;
}

// Visibility [0,1] from one local-shadow tile. Receiver bias is applied in
// WORLD space — a normal offset plus a slope-scaled push toward the light,
// both proportional to the tile's world texel size at this distance — so no
// perspective-depth-unit conversion is needed (the caster pipelines add a
// small rasterizer depth bias on top). 4 comparison taps on a half-texel
// square = an effective 3x3 PCF footprint (each tap is a hardware 2x2).
float sampleLocalShadowSlot(uint slot, float3 P, float3 N, float3 L, float dist) {
    LocalShadowSlot s = localShadowSlots[min(slot, LOCAL_SHADOW_MAX_SLOTS - 1u)];
    float texelWorld = max(dist, s.params.z) * s.params.x;
    float NdotL = saturate(dot(N, L));
    float tanTheta = min(sqrt(1.0 - NdotL * NdotL) / max(NdotL, 1e-2), 4.0);
    float3 samplePos = P + N * (texelWorld * 1.5) + L * (texelWorld * (1.0 + tanTheta));
    float4 lp = mul(s.viewProj, float4(samplePos, 1.0));
    if (lp.w <= 1e-4) return 1.0;
    float3 sc = lp.xyz / lp.w;
    float2 uv = float2(sc.x * 0.5 + 0.5, 1.0 - (sc.y * 0.5 + 0.5));
    if (min(uv.x, uv.y) < 0.0 || max(uv.x, uv.y) > 1.0 || sc.z <= 0.0 || sc.z >= 1.0)
        return 1.0;
    // Clamp inside the tile so PCF taps never bleed into a neighbor.
    uv = clamp(uv, s.params.y, 1.0 - s.params.y);
    float2 atlasUV = s.rect.xy + uv * s.rect.zw;
    float t = 0.5 * s.params.w;
    float sum = localShadowAtlas.SampleCmpLevelZero(localShadowCmpSampler, atlasUV + float2(-t, -t), sc.z);
    sum += localShadowAtlas.SampleCmpLevelZero(localShadowCmpSampler, atlasUV + float2(t, -t), sc.z);
    sum += localShadowAtlas.SampleCmpLevelZero(localShadowCmpSampler, atlasUV + float2(-t, t), sc.z);
    sum += localShadowAtlas.SampleCmpLevelZero(localShadowCmpSampler, atlasUV + float2(t, t), sc.z);
    return sum * 0.25;
}

// Shadow visibility for a local light: face select (points) + tile sample +
// smooth fade to unshadowed near the light radius so the shadow never pops
// at the shadow map's far plane.
float localLightShadow(int shadow_slot, bool is_point, float3 light_pos, float radius,
                       float3 P, float3 N, float3 L, float dist) {
    if (shadow_slot < 0) return 1.0;
    uint slot = uint(shadow_slot);
    if (is_point) slot += localShadowCubeFace(P - light_pos);
    float shadow = sampleLocalShadowSlot(slot, P, N, L, dist);
    return lerp(shadow, 1.0, smoothstep(0.75, 1.0, dist / max(radius, 1e-4)));
}

// ---------------------------------------------------------------------------
// Light data loads — the decode is cluster_light_load.hlsli, shared verbatim
// with the compute path. It reads `lightDataBuf` and `clusterParams` by name,
// both declared above, which is why the include sits HERE and not at the top.
// ---------------------------------------------------------------------------

#include "cluster_light_load.hlsli"

// ---------------------------------------------------------------------------
// Ray-traced local shadows, Ratio tier (R2b, docs/rt_lighting_plan.md §5.3).
//
// `traced_visibility` is the denoised R8 mask rt_local_shadow.comp.hlsl +
// the three shadow denoiser passes produced for this frame, sampled by the
// INCLUDING shader at its own slot and passed in — this file does not declare
// it, because the four including shaders sit at four different slot numbers and
// a fifth #define would be one more thing to keep in step.
//
// It is a Heitz ratio estimator: the whole point+spot subtotal — the ANALYTIC
// sum, unchanged, still sharp — is multiplied by one stochastic visibility
// ratio. That is what buys soft shadows for every local light at one ray per
// pixel; the alternative (a per-light traced term) would need a ray per light.
//
// The atlas path below is not deleted and not disabled: in Ratio mode the
// engine simply never assigns a shadow_slot, so `localLightShadow` returns 1.0
// through its existing `shadow_slot < 0` branch and the two cannot double-darken
// a pixel. That is also what makes `Off` byte-identical — with the mask
// argument left at its default the multiply is not merely by 1.0, it is absent.
//
// IN THE FULL-DI TIER (R5c) THIS FUNCTION IS NOT CALLED AT ALL. Every point and
// spot light is traced there, so there is no analytic subtotal to multiply and
// nothing for a ratio to correct; the four scene shaders branch to
// `localDiLighting` (local_di.hlsli) instead of calling this. That is a
// stronger statement than "the mask is 1": the loop below does not run, so a
// light the traced pass could not reach contributes nothing rather than
// contributing unshadowed. Both tiers are estimates of the same integral, which
// is what the energy identity in local_di.test.cpp pins.
// ---------------------------------------------------------------------------
float3 accumulateClusterLights(float3 P, float3 N, float3 V,
                                float3 baseColor, float metallic, float roughness, float ao,
                                float view_depth, float2 screen_uv,
                                float traced_visibility = -1.0) {
    if (clusterParams.gridDim.w == 0u) return float3(0, 0, 0);
    uint cidx = clusterIndexFromScreen(screen_uv, view_depth, clusterParams);
    uint2 oc = clusterOffsetsBuf[cidx];
    uint offset = oc.x;
    // Never trust the SSBO count alone: clamp to the CPU-enforced per-cluster
    // cap (MAX_LIGHTS_PER_CLUSTER) so a stale or partially-uploaded offsets
    // buffer cannot spin this loop for up to 2^32 iterations.
    uint count  = min(oc.y, clusterParams.maxPerCluster.x);
    float3 acc = float3(0, 0, 0);
    for (uint i = 0; i < count; ++i) {
        uint header_idx = lightIndicesBuf[offset + i];
        LightHeader h = loadLightHeader(header_idx);
        if (h.type == 0u) {
            PointLightGPU pl = loadPointLight(h.payload_index);
            float3 to_light = pl.position - P;
            float dist = length(to_light);
            if (dist > pl.radius) continue;
            float3 L = to_light / max(dist, 1e-4);
            float att = pointFalloff(dist, pl.radius);
            float shadow = localLightShadow(pl.shadow_slot, true, pl.position, pl.radius, P, N, L, dist);
            if (shadow <= 0.0) continue;
            acc += evaluatePBRPoint(N, V, L, pl.color * pl.intensity, att,
                                    baseColor, metallic, roughness, ao) * shadow;
        } else {
            SpotLightGPU sl = loadSpotLight(h.payload_index);
            float3 to_light = sl.position - P;
            float dist = length(to_light);
            if (dist > sl.radius) continue;
            float3 L = to_light / max(dist, 1e-4);
            float att = pointFalloff(dist, sl.radius);
            float cone = spotConeFactor(L, sl.direction, sl.inner_cos, sl.outer_cos);
            if (cone <= 0.0) continue;
            float shadow = localLightShadow(sl.shadow_slot, false, sl.position, sl.radius, P, N, L, dist);
            if (shadow <= 0.0) continue;
            acc += evaluatePBRSpot(N, V, L, sl.color * sl.intensity, att, cone,
                                   baseColor, metallic, roughness, ao) * shadow;
        }
    }
    if (traced_visibility >= 0.0) {
        acc *= traced_visibility;
    }
    return acc;
}

#endif // CLUSTER_LIGHTING_HLSLI
