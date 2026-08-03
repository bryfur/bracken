#pragma once

// The froxel index math, once.
//
// Three places have to agree about which cluster a fragment/pixel/light falls
// in: the fragment path (`shaders/cluster_index.hlsli`, included by
// `clusters.hlsli`), the compute path (`shaders/cluster_compute.hlsli`, which
// includes the same file), and the CPU binner (`light_cluster.cpp`). The first
// two share their source outright; this header is the CPU mirror, and
// `light_cluster.test.cpp` pins it against the binner's own cluster bounds
// across a full grid sweep.
//
// GPU-free on purpose: no `gpu/`, no SDL, nothing but glm — so a test can
// exercise the exact function the renderer calls with no device.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>

namespace bracken::render::lighting {

// Std140-compatible layout shared with shaders/cluster_index.hlsli.
// All fields packed contiguously; HLSL cbuffer layout requires 16-byte alignment.
struct alignas(16) ClusterParams {
    glm::mat4 view;
    glm::mat4 invProjection;
    glm::vec4 screenSize;     // (width, height, 1/width, 1/height)
    glm::vec4 zPlanes;        // (near, far, log(far/near), 1/log(far/near))
    glm::uvec4 gridDim;       // (X, Y, Z, totalLightCount)
    // .x = per-cluster light cap (shader clamps its loop with it);
    // .y = light-data buffer capacity in lights (max_lights_) — the shader
    //      derives its payload section offsets from it (cluster_lighting.hlsli).
    glm::uvec4 maxPerCluster;
};

/**
 * One froxel's VIEW-SPACE axis-aligned bounds — the memoized table
 * `ClusterGrid::begin_frame` rebuilds whenever the projection or the near/far
 * pair changes, and the thing every froxel binner in the tree tests against.
 *
 * Here rather than inside `ClusterGrid` because it has two binners now: the
 * light one that owns the table, and `render::rt::EmissiveClusterGrid`, which
 * borrows it. A view-space AABB needs no device and no SDL, so it belongs in
 * the GPU-free half with the index math it is used alongside.
 */
struct ClusterBounds {
    glm::vec3 view_min{0.0f};
    glm::vec3 view_max{0.0f};
};

/// Positive view-space depth (distance forward of the camera) -> Z slice.
/// Mirror of `clusterZSlice` in shaders/cluster_index.hlsli, operation for
/// operation — including the `max(depth, near)` clamp and the floor/clamp
/// order, which is what makes the two land on the same slice at a boundary.
inline uint32_t cluster_z_slice(float view_depth, const ClusterParams& cp) {
    const float depth = std::max(view_depth, cp.zPlanes.x);
    const float t = std::log(depth / cp.zPlanes.x) * cp.zPlanes.w;
    const float dim = static_cast<float>(cp.gridDim.z);
    return static_cast<uint32_t>(std::clamp(std::floor(t * dim), 0.0f, dim - 1.0f));
}

/// Screen uv (SV_Position * screenSize.zw — uv.y = 0 is the TOP of the screen)
/// + positive view depth -> linear cluster index. Mirror of
/// `clusterIndexFromScreen`, including the Y flip: CPU binning maps cluster row
/// cy = 0 to NDC y = -1, the BOTTOM of the screen.
inline uint32_t cluster_index_from_screen(glm::vec2 screen_uv, float view_depth, const ClusterParams& cp) {
    const auto tile = [](float v, uint32_t dim) {
        return static_cast<uint32_t>(
            std::clamp(std::floor(v * static_cast<float>(dim)), 0.0f, static_cast<float>(dim) - 1.0f));
    };
    const uint32_t cx = tile(screen_uv.x, cp.gridDim.x);
    const uint32_t cy = tile(1.0f - screen_uv.y, cp.gridDim.y);
    const uint32_t cz = cluster_z_slice(view_depth, cp);
    return (cz * cp.gridDim.y + cy) * cp.gridDim.x + cx;
}

} // namespace bracken::render::lighting
