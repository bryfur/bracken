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

#endif // CLUSTER_LIGHT_LOAD_HLSLI
