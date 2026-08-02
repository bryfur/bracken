// clusters.hlsli - clustered forward+ shader-side helpers, PBR half.
// Layout matches render::lighting::ClusterGrid. See light_cluster.cpp for
// the C++ side that builds + uploads these buffers.
//
// The grid constants, ClusterParams, the light records and the froxel index
// math moved to cluster_index.hlsli, which is binding-free AND pbr-free so the
// compute path (cluster_compute.hlsli) can share the exact same arithmetic.
// What is left here is what needs pbr.hlsli.

#ifndef CLUSTERS_HLSLI
#define CLUSTERS_HLSLI

#include "cluster_index.hlsli"
#include "pbr.hlsli"

// Generic point/spot PBR contribution. Caller supplies surface params and the
// per-light attenuation factor; the BRDF is identical to evaluatePBR.
float3 evaluatePBRPoint(float3 N, float3 V, float3 L, float3 lightColor, float attenuation,
                        float3 baseColor, float metallic, float roughness, float ao) {
    return evaluatePBR(N, V, L, lightColor * attenuation,
                       baseColor, metallic, roughness, ao);
}

float3 evaluatePBRSpot(float3 N, float3 V, float3 L, float3 lightColor,
                       float attenuation, float cone,
                       float3 baseColor, float metallic, float roughness, float ao) {
    return evaluatePBR(N, V, L, lightColor * (attenuation * cone),
                       baseColor, metallic, roughness, ao);
}

#endif // CLUSTERS_HLSLI
