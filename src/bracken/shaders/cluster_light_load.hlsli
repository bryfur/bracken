// cluster_light_load.hlsli - decoding the packed light-data ByteAddressBuffer.
//
// This file NAMES resources it does not own: `lightDataBuf` (the packed light
// buffer) and `clusterParams` (the cbuffer). The including file must declare
// both, with whatever set/binding its stage requires, BEFORE including this —
// the same shape as cluster_lighting.hlsli's own contract. That indirection is
// deliberate: the fragment path binds these on set 2/3 and the compute path on
// set 0/2, but the byte offsets they decode must be one copy of the code.
//
// CPU side: ClusterGrid::upload() (render/lighting/light_cluster.cpp).

#ifndef CLUSTER_LIGHT_LOAD_HLSLI
#define CLUSTER_LIGHT_LOAD_HLSLI

#include "cluster_index.hlsli"

// Light-data buffer section offsets. ClusterGrid::upload() packs the buffer as:
//   [LightDataHeader 16B][LightHeader * cap][PointLight * cap][SpotLight * cap]
// where cap = clusterParams.maxPerCluster.y — the CPU-side buffer capacity
// (ClusterGrid max_lights_) — so the offsets follow the CPU layout instead of
// hardcoding a light count. Strides: LightHeader 8B, PointLight 48B, SpotLight 64B.
uint pointLightsBase() { return 16u + clusterParams.maxPerCluster.y * 8u; }
uint spotLightsBase()  { return 16u + clusterParams.maxPerCluster.y * (8u + 48u); }

LightHeader loadLightHeader(uint i) {
    uint base = 16u + i * 8u; // headers start right after the 16B LightDataHeader
    uint2 raw = lightDataBuf.Load2(base);
    LightHeader h;
    h.type = raw.x;
    h.payload_index = raw.y;
    return h;
}

PointLightGPU loadPointLight(uint i) {
    uint base = pointLightsBase() + i * 48;
    uint4 a = lightDataBuf.Load4(base);
    uint4 b = lightDataBuf.Load4(base + 16);
    uint4 c = lightDataBuf.Load4(base + 32);
    PointLightGPU p;
    p.position    = asfloat(a.xyz);
    p.radius      = asfloat(a.w);
    p.color       = asfloat(b.xyz);
    p.intensity   = asfloat(b.w);
    // c.x is the CPU-side casts_shadows byte + padding (never read here);
    // shadow_slot is engine-assigned each frame (-1 = unshadowed).
    p.shadow_slot = asint(c.y);
    // c.z is stable_id, resolved during binning; c.w is the remaining pad.
    p.stable_id   = c.z;
    return p;
}

SpotLightGPU loadSpotLight(uint i) {
    uint base = spotLightsBase() + i * 64;
    uint4 a = lightDataBuf.Load4(base);
    uint4 b = lightDataBuf.Load4(base + 16);
    uint4 c = lightDataBuf.Load4(base + 32);
    uint4 d = lightDataBuf.Load4(base + 48);
    SpotLightGPU s;
    s.position    = asfloat(a.xyz);
    s.radius      = asfloat(a.w);
    s.direction   = asfloat(b.xyz);
    s.inner_cos   = asfloat(b.w);
    s.color       = asfloat(c.xyz);
    s.outer_cos   = asfloat(c.w);
    s.intensity   = asfloat(d.x);
    // d.y is the CPU-side casts_shadows byte + padding (never read here).
    s.shadow_slot = asint(d.z);
    s.stable_id   = d.w;
    return s;
}

// ---------------------------------------------------------------------------
// The RIS target function (R2b, docs/rt_lighting_plan.md §5.3).
//
// p̂ for a candidate light = luminance of its UNSHADOWED DIFFUSE contribution.
// It lives HERE, next to the decode and above the fragment path's accumulator,
// for exactly the reason the froxel index math lives in cluster_index.hlsli: the
// compute pass that PICKS a light and the fragment pass that SUMS the lights
// must agree about which of them contribute nothing, or the ratio estimator
// stops being unbiased. Both zeros below are `continue`s in
// accumulateClusterLights — `dist > radius`, and `cone <= 0` — and the
// attenuation is the same pointFalloff()/spotConeFactor() that function calls,
// not a second copy of the curve.
//
// What it deliberately is NOT: the albedo is absent (identical for every
// candidate at a pixel, so it cancels out of the reservoir's normalized
// weights) and so is the specular lobe (no view-dependent material term is
// worth the divergence here; leaving it out only shifts which light a glancing
// highlight prefers). A target function is allowed to approximate — that is
// what RIS is for — as long as its SUPPORT matches.
//
// CPU mirror: render/lighting/light_target.hpp, pinned by light_target.test.cpp.
// ---------------------------------------------------------------------------

float clusterTargetLuminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

/// p̂ of a point light at surface (P, N). 0 = contributes nothing here.
float clusterPointLightTarget(PointLightGPU pl, float3 P, float3 N) {
    float3 to_light = pl.position - P;
    float dist = length(to_light);
    if (dist > pl.radius) return 0.0;
    float3 L = to_light / max(dist, 1e-4);
    float ndotl = max(dot(N, L), 0.0);
    if (ndotl <= 0.0) return 0.0;
    return clusterTargetLuminance(pl.color * pl.intensity) * pointFalloff(dist, pl.radius) * ndotl;
}

/// p̂ of a spot light — the point target times the cone factor.
float clusterSpotLightTarget(SpotLightGPU sl, float3 P, float3 N) {
    float3 to_light = sl.position - P;
    float dist = length(to_light);
    if (dist > sl.radius) return 0.0;
    float3 L = to_light / max(dist, 1e-4);
    float cone = spotConeFactor(L, sl.direction, sl.inner_cos, sl.outer_cos);
    if (cone <= 0.0) return 0.0;
    float ndotl = max(dot(N, L), 0.0);
    if (ndotl <= 0.0) return 0.0;
    return clusterTargetLuminance(sl.color * sl.intensity) * pointFalloff(dist, sl.radius) * cone * ndotl;
}

/// Where a candidate light SITS and how big its emitter is, for the shadow ray.
/// Both light types answer through one struct so the trace has no branch.
struct ClusterLightSample {
    float3 position;
    float radius;   ///< the light's RANGE, not its emitter size
    bool valid;
};

ClusterLightSample clusterLightGeometry(uint packed_index) {
    ClusterLightSample s;
    s.position = float3(0, 0, 0);
    s.radius = 0.0;
    s.valid = false;
    LightHeader h = loadLightHeader(packed_index);
    if (h.type == 0u) {
        PointLightGPU pl = loadPointLight(h.payload_index);
        s.position = pl.position;
        s.radius = pl.radius;
    } else {
        SpotLightGPU sl = loadSpotLight(h.payload_index);
        s.position = sl.position;
        s.radius = sl.radius;
    }
    s.valid = true;
    return s;
}

/// p̂ of whichever light `packed_index` names, dispatched on the header type.
float clusterLightTarget(uint packed_index, float3 P, float3 N) {
    LightHeader h = loadLightHeader(packed_index);
    if (h.type == 0u) {
        return clusterPointLightTarget(loadPointLight(h.payload_index), P, N);
    }
    return clusterSpotLightTarget(loadSpotLight(h.payload_index), P, N);
}

#endif // CLUSTER_LIGHT_LOAD_HLSLI
