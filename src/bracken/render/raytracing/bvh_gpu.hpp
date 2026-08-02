#pragma once

#include "bracken/render/raytracing/bvh.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <vector>

namespace bracken::render::rt {

/**
 * One BVH node in TRAVERSAL form: 64 bytes holding BOTH children's AABBs and
 * links (the Aila-Laine layout).
 *
 * This is the UPLOAD format, not the build format. `BvhNode` (bvh.hpp) stays
 * 32 bytes and stays what the builder emits; this is produced from it by
 * `convert_to_gpu_nodes` and is all the GPU ever sees.
 *
 * WHY, since the 32-byte node was itself chosen for bandwidth: with one AABB
 * per node, resolving a single binary decision costs three dependent fetches —
 * the node, then each child, each 32 bytes and each on the critical path
 * because the address of the second depends on nothing but is issued after the
 * first anyway once the walk needs both boxes to order them. Here one 64-byte
 * fetch resolves the whole step: both boxes, both links, no dependent load to
 * decide where to go next. Node COUNT halves at the same time (only interior
 * nodes become nodes here, and there are leaves-1 of those against 2*leaves-1
 * total), so the buffers do not grow — 64 B x N/2 is 32 B x N.
 *
 * Per side: `count == 0` means the side is an interior node and `child` is the
 * index of that child's own 64-byte node; `count > 0` means the side is a leaf
 * and `child` is its first primitive index, exactly as `BvhNode::left_first`
 * meant. Leaves therefore never occupy a node of their own and never go on the
 * traversal stack — the parent already holds everything needed to intersect
 * them.
 *
 * Indices are TREE-RELATIVE, like the 32-byte tree they come from: a BLAS's
 * node indices start at 0 for that BLAS and the shader adds the instance's
 * `blas_node_offset`. The root is always index 0.
 */
struct BvhNodeGpu {
    glm::vec3 lmin{0.0f};
    uint32_t lchild = 0;
    glm::vec3 lmax{0.0f};
    uint32_t lcount = 0;
    glm::vec3 rmin{0.0f};
    uint32_t rchild = 0;
    glm::vec3 rmax{0.0f};
    uint32_t rcount = 0;
};
static_assert(sizeof(BvhNodeGpu) == 64,
              "BvhNodeGpu must stay 64 bytes: it mirrors the HLSL BvhNodeGpu in rt_trace.hlsli");

/**
 * Coordinate of the sentinel box used for the unused side of a degenerate root
 * (see `convert_to_gpu_nodes`). A degenerate POINT box this far out fails the
 * slab test for every ray: with min == max the near and far distances along
 * each axis are equal, so either they exceed the ray's tMax (tNear > tFar) or
 * they are negative (tFar < 0 <= tNear). An INVERTED box would not do — the
 * branchless min/max slab test reads min > max as an infinite box and lets
 * every ray in.
 */
inline constexpr float k_empty_side_coord = 1e30f;

/**
 * Convert a built 32-byte tree to the 64-byte traversal layout.
 *
 * Only the source tree's INTERIOR nodes become nodes here; a leaf is folded
 * into its parent's side. Two linear passes over the source: one to number the
 * interior nodes (a child's new index is needed before the parent is written,
 * and children always sit at higher indices than their parent), one to write.
 * O(n) with a small constant — the TLAS path runs whenever anything in the
 * scene moves.
 *
 * DEGENERATE ROOT: a tree whose root is itself a leaf (a one-triangle BLAS, a
 * one-instance TLAS) has no interior node at all, so there is nothing to fold
 * that leaf into. Rather than teach every walk about a "the root is a leaf"
 * case, one node is emitted whose LEFT side is the real leaf and whose RIGHT
 * side is an empty leaf: `rcount > 0` so no walk ever mistakes it for an
 * interior link into a node that does not exist, with the sentinel box above
 * so no ray ever reaches its primitive window. The window is `[0, 1)`, which
 * is inside the real leaf's own range anyway, so even entering it could not
 * change an answer.
 *
 * @param nodes   the built tree (`Blas::nodes` or `Tlas::nodes`).
 * @param out     receives the converted tree; cleared first.
 * @param scratch old-index to new-index map, kept caller-side so a per-frame
 *                caller does not reallocate it every rebuild.
 */
void convert_to_gpu_nodes(std::span<const BvhNode> nodes, std::vector<BvhNodeGpu>& out,
                          std::vector<uint32_t>& scratch);

/** Convenience overload for callers with no per-frame cost to care about. */
[[nodiscard]] std::vector<BvhNodeGpu> convert_to_gpu_nodes(std::span<const BvhNode> nodes);

} // namespace bracken::render::rt
