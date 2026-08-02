// cluster_index.hlsli - the froxel index math and the light record layouts.
//
// BINDING-FREE by construction: this file names no resource, so it can be
// included from a fragment shader (via clusters.hlsli) and from a compute
// shader (via cluster_compute.hlsli) alike. That is the whole point — the two
// paths must land on the SAME froxel for the same pixel, and the only way to
// guarantee that is to give them one copy of the arithmetic.
//
// CPU mirror: render/lighting/cluster_index.hpp, which light_cluster.cpp's
// binner itself calls, and which light_cluster.test.cpp pins against the
// binner's cluster bounds across a grid sweep.

#ifndef CLUSTER_INDEX_HLSLI
#define CLUSTER_INDEX_HLSLI

#define CLUSTER_DIM_X 16
#define CLUSTER_DIM_Y 9
#define CLUSTER_DIM_Z 24
#define CLUSTER_COUNT (CLUSTER_DIM_X * CLUSTER_DIM_Y * CLUSTER_DIM_Z)

struct ClusterParams {
    float4x4 view;
    float4x4 invProjection;
    float4 screenSize;     // (w, h, 1/w, 1/h)
    float4 zPlanes;        // (near, far, log(far/near), 1/log(far/near))
    uint4 gridDim;         // (X, Y, Z, totalLightCount)
    uint4 maxPerCluster;   // (max_lights_per_cluster, light buffer capacity, _, _)
};

struct PointLightGPU {
    float3 position; float radius;
    float3 color;    float intensity;
    int shadow_slot; // base tile of the light's 6 faces in the local-shadow atlas; -1 = unshadowed
    uint stable_id;  // persistent identity (contracts/light.hpp); never 0 once the engine resolved it
};

struct SpotLightGPU {
    float3 position;  float radius;
    float3 direction; float inner_cos;
    float3 color;     float outer_cos;
    float intensity;
    int shadow_slot;  // the light's tile in the local-shadow atlas; -1 = unshadowed
    uint stable_id;   // persistent identity (contracts/light.hpp)
};

struct LightHeader {
    uint type;          // 0 = point, 1 = spot
    uint payload_index;
};

// Maps positive view-space depth (distance forward of camera) to a Z slice.
uint clusterZSlice(float view_depth, ClusterParams cp) {
    float depth = max(view_depth, cp.zPlanes.x);
    float t = log(depth / cp.zPlanes.x) * cp.zPlanes.w;
    uint z = (uint)clamp(floor(t * (float)cp.gridDim.z), 0.0, (float)cp.gridDim.z - 1.0);
    return z;
}

uint clusterIndexFromScreen(float2 uv, float view_depth, ClusterParams cp) {
    // uv comes from SV_Position * screenSize.zw, so uv.y = 0 is the TOP of the
    // screen. CPU binning (light_cluster.cpp, unproject_corner) maps cluster
    // row cy = 0 to NDC y = -1 — the BOTTOM of the screen — so flip Y here to
    // match that convention. X needs no flip (cx = 0 is NDC x = -1 = left on
    // both sides).
    uint cx = (uint)clamp(floor(uv.x * (float)cp.gridDim.x), 0.0, (float)cp.gridDim.x - 1.0);
    uint cy = (uint)clamp(floor((1.0 - uv.y) * (float)cp.gridDim.y), 0.0, (float)cp.gridDim.y - 1.0);
    uint cz = clusterZSlice(view_depth, cp);
    return (cz * cp.gridDim.y + cy) * cp.gridDim.x + cx;
}

// Smoothly fades light to zero at the radius and applies inverse-square law.
// Lives here rather than in clusters.hlsli so the compute path gets the exact
// attenuation the forward loop applies — R2's target function has to agree
// with the forward pass or "culled => zero" stops being bias-free.
float pointFalloff(float dist, float radius) {
    float r = saturate(dist / max(radius, 1e-4));
    float r2 = r * r;
    float win = saturate(1.0 - r2 * r2);  // (1 - (d/R)^4) clamped
    return (win * win) / max(dist * dist, 1e-4);
}

float spotConeFactor(float3 L_to_surface, float3 spotDir, float inner_cos, float outer_cos) {
    float cd = dot(spotDir, -L_to_surface);
    return smoothstep(outer_cos, inner_cos, cd);
}

#endif // CLUSTER_INDEX_HLSLI
