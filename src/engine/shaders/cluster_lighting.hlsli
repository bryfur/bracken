// cluster_lighting.hlsli - bindings + accumulator for clustered forward+ lights.
// Including shader must NOT also declare ClusterParams cbuffer / SSBO bindings;
// these slots (set 3, b2 + three set-2 bindings after the sampled textures)
// are reserved for clustered lighting.
//
// SDL GPU lays out fragment set 2 as sampled textures, then storage textures,
// then storage buffers. The SSBO bindings therefore start right after the
// including shader's sampled textures: define CLUSTER_SSBO_BINDING_BASE to
// that shader's fragment sampler count BEFORE including this file.
// SDL_BindGPUFragmentStorageBuffers(pass, 0, bufs, 3) then maps slots 0..2
// onto these bindings.

#ifndef CLUSTER_LIGHTING_HLSLI
#define CLUSTER_LIGHTING_HLSLI

#include "clusters.hlsli"

#ifndef CLUSTER_SSBO_BINDING_BASE
#error "Define CLUSTER_SSBO_BINDING_BASE (= fragment sampler count) before including cluster_lighting.hlsli"
#endif

[[vk::binding(2, 3)]]
cbuffer ClusterParamsCB { ClusterParams clusterParams; };

[[vk::binding(CLUSTER_SSBO_BINDING_BASE + 0, 2)]]
ByteAddressBuffer lightDataBuf;

[[vk::binding(CLUSTER_SSBO_BINDING_BASE + 1, 2)]]
StructuredBuffer<uint2> clusterOffsetsBuf;

[[vk::binding(CLUSTER_SSBO_BINDING_BASE + 2, 2)]]
StructuredBuffer<uint> lightIndicesBuf;

// Light-data buffer section offsets. ClusterGrid::upload() (light_cluster.cpp)
// packs the buffer as:
//   [LightDataHeader 16B][LightHeader * cap][PointLight * cap][SpotLight * cap]
// where cap = clusterParams.maxPerCluster.y — the CPU-side buffer capacity
// (ClusterGrid max_lights_) — so the offsets follow the CPU layout instead of
// hardcoding a light count. Strides: LightHeader 8B, PointLight 32B, SpotLight 64B.
uint pointLightsBase() { return 16u + clusterParams.maxPerCluster.y * 8u; }
uint spotLightsBase()  { return 16u + clusterParams.maxPerCluster.y * (8u + 32u); }

LightHeader loadLightHeader(uint i) {
    uint base = 16u + i * 8u; // headers start right after the 16B LightDataHeader
    uint2 raw = lightDataBuf.Load2(base);
    LightHeader h;
    h.type = raw.x;
    h.payload_index = raw.y;
    return h;
}

PointLightGPU loadPointLight(uint i) {
    uint base = pointLightsBase() + i * 32;
    uint4 a = lightDataBuf.Load4(base);
    uint4 b = lightDataBuf.Load4(base + 16);
    PointLightGPU p;
    p.position  = asfloat(a.xyz);
    p.radius    = asfloat(a.w);
    p.color     = asfloat(b.xyz);
    p.intensity = asfloat(b.w);
    return p;
}

SpotLightGPU loadSpotLight(uint i) {
    uint base = spotLightsBase() + i * 64;
    uint4 a = lightDataBuf.Load4(base);
    uint4 b = lightDataBuf.Load4(base + 16);
    uint4 c = lightDataBuf.Load4(base + 32);
    uint4 d = lightDataBuf.Load4(base + 48);
    SpotLightGPU s;
    s.position  = asfloat(a.xyz);
    s.radius    = asfloat(a.w);
    s.direction = asfloat(b.xyz);
    s.inner_cos = asfloat(b.w);
    s.color     = asfloat(c.xyz);
    s.outer_cos = asfloat(c.w);
    s.intensity = asfloat(d.x);
    s._pad      = float3(0, 0, 0);
    return s;
}

float3 accumulateClusterLights(float3 P, float3 N, float3 V,
                                float3 baseColor, float metallic, float roughness, float ao,
                                float view_depth, float2 screen_uv) {
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
            acc += evaluatePBRPoint(N, V, L, pl.color * pl.intensity, att,
                                    baseColor, metallic, roughness, ao);
        } else {
            SpotLightGPU sl = loadSpotLight(h.payload_index);
            float3 to_light = sl.position - P;
            float dist = length(to_light);
            if (dist > sl.radius) continue;
            float3 L = to_light / max(dist, 1e-4);
            float att = pointFalloff(dist, sl.radius);
            float cone = spotConeFactor(L, sl.direction, sl.inner_cos, sl.outer_cos);
            if (cone <= 0.0) continue;
            acc += evaluatePBRSpot(N, V, L, sl.color * sl.intensity, att, cone,
                                   baseColor, metallic, roughness, ao);
        }
    }
    return acc;
}

#endif // CLUSTER_LIGHTING_HLSLI
