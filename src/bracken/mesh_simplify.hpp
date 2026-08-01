#pragma once

#include "bracken/gpu/gpu_types.hpp"

#include <cstdint>
#include <vector>

namespace bracken {

/** One simplified level: the same surface at fewer triangles. */
struct SimplifiedMesh {
    std::vector<gpu::Vertex3D> vertices;
    std::vector<uint32_t> indices;
};

/**
 * Vertex-clustering mesh simplification.
 *
 * The mesh's bounding box is divided into a grid; every vertex falling in a
 * cell collapses to one representative, and triangles whose corners end up in
 * the same cell disappear. Coarser grid, fewer triangles.
 *
 * WHY CLUSTERING AND NOT QUADRIC ERROR METRICS. QEM produces visibly better
 * silhouettes and is what an offline tool should use. But it needs edge
 * collapse with a priority queue and topology bookkeeping, it can fail on the
 * non-manifold geometry that game art is full of, and it is slow enough that
 * you would not run it at load time. Clustering is a handful of lines, cannot
 * fail, is linear, and runs on every mesh a game loads without a bake step —
 * and at the distances an LOD actually switches, the silhouette difference is
 * a few pixels. If a project later wants QEM output it can author LOD meshes
 * and hand them to the engine directly; this exists so that LOD works at all
 * without an asset pipeline.
 *
 * The representative is the cell's AVERAGE position, not an arbitrary member:
 * averaging keeps the simplified surface centred on the original instead of
 * letting it shrink toward whichever vertex happened to be found first.
 * Normals and UVs are averaged likewise, and normals renormalized.
 *
 * Deterministic: the same input always produces the same output, so a
 * generated LOD is stable across runs and machines.
 */
[[nodiscard]] SimplifiedMesh simplify_by_clustering(const std::vector<gpu::Vertex3D>& vertices,
                                                    const std::vector<uint32_t>& indices,
                                                    int grid_resolution);

/**
 * Pick a grid resolution that lands near `target_ratio` of the input triangle
 * count, by bisection on the actual result.
 *
 * Triangle count is not a smooth function of grid resolution — it depends on
 * how the mesh happens to fill its cells — so guessing a resolution from a
 * formula misses badly on exactly the meshes that matter (long thin props,
 * dense clusters). Measuring is cheap here because simplification is linear.
 */
[[nodiscard]] SimplifiedMesh simplify_to_ratio(const std::vector<gpu::Vertex3D>& vertices,
                                               const std::vector<uint32_t>& indices,
                                               float target_ratio);

} // namespace bracken
