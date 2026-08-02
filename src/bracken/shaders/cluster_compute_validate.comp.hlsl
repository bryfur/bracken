// cluster_compute_validate.comp.hlsl - the compile-time guarantee for
// cluster_compute.hlsli.
//
// An .hlsli that no .hlsl includes is never compiled, and an include that is
// never compiled rots. R2b is the real consumer of the froxel-access header;
// until it lands, THIS is what makes the header a build artifact rather than a
// text file — it declares the bindings, walks a froxel's light list, decodes
// both record types and touches the identity side-channel, so any drift
// between the shared includes and the CPU layout is a build failure now
// instead of a debugging session later.
//
// It is deliberately not registered in the pipeline catalogue: nothing
// dispatches it, and its only job is to be compiled. Delete it when R2b's real
// pass includes the same header.

// This shader binds no sampled textures and no read-only storage textures, so
// the cluster storage buffers start at set-0 slot 0; its only uniform is
// ClusterParams, at set-2 slot 0.
#define CLUSTER_COMPUTE_SSBO_BINDING_BASE 0
#define CLUSTER_COMPUTE_PARAMS_BINDING 0
#include "cluster_compute.hlsli"

[[vk::binding(0, 1)]]
RWStructuredBuffer<uint> outDigest;

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    uint2 pixel = uint2(tid.x & 0xFFu, tid.x >> 8);
    float2 uv = (float2(pixel) + 0.5) * clusterParams.screenSize.zw;

    uint froxel = clusterFroxelIndex(uv, 10.0);
    uint count = clusterFroxelLightCount(froxel);

    uint digest = clusterTotalLights();
    for (uint i = 0; i < count; ++i) {
        uint packed = clusterFroxelLight(froxel, i);
        LightHeader h = loadLightHeader(packed);
        if (h.type == 0u) {
            PointLightGPU pl = loadPointLight(h.payload_index);
            digest += asuint(pointFalloff(1.0, pl.radius)) + uint(pl.shadow_slot + 1) + pl.stable_id;
        } else {
            SpotLightGPU sl = loadSpotLight(h.payload_index);
            digest += asuint(spotConeFactor(float3(0, 0, 1), sl.direction, sl.inner_cos, sl.outer_cos)) +
                      uint(sl.shadow_slot + 1) + sl.stable_id;
        }
        digest += clusterLightStableId(packed);
        digest += clusterRemapPrevLight(packed) == LIGHT_REMAP_INVALID ? 1u : 0u;
    }
    outDigest[tid.x] = digest;
}
