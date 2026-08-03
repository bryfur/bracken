#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <vector>

namespace bracken::render::rt {

/**
 * One BVH node in the BUILD format — what the builder emits and what every
 * CPU-side consumer here (`sah_cost`, the tests, the conversion below) reads.
 *
 * NOT what the GPU traverses. Nodes are converted to the 64-byte two-children
 * `BvhNodeGpu` (`bvh_gpu.hpp`) on the upload side, so a traversal step fetches
 * one node instead of a node and both its children. This format stays 32 bytes
 * because it is the format the RECURSIVE BUILD writes, where a node is touched
 * once and its sibling may not exist yet.
 *
 * 32 bytes exactly, so two nodes share a cache line and a build step touching
 * a node and its sibling costs one fetch. Every field placement here is a
 * bandwidth decision, not a style one:
 *
 *  - The right child is NOT stored. Siblings are emitted adjacently, so
 *    `right == left_first + 1`. That is 4 bytes saved per node, and 4 bytes is
 *    the difference between 32 and 36, which is the difference between two
 *    nodes per cache line and one.
 *  - `count == 0` marks an interior node, so the leaf test is a compare
 *    against zero rather than a flag bit that has to be masked out.
 */
struct BvhNode {
    glm::vec3 bounds_min{0.0f};
    /// Interior: index of the LEFT child (right is left+1). Leaf: first
    /// primitive index.
    uint32_t left_first = 0;
    glm::vec3 bounds_max{0.0f};
    /// 0 = interior node; otherwise the number of primitives in this leaf.
    uint32_t count = 0;
};
static_assert(sizeof(BvhNode) == 32, "BvhNode must stay 32 bytes: two per cache line");

/**
 * A triangle in traversal form: origin plus two edges.
 *
 * Stored as v0/e1/e2 rather than v0/v1/v2 because Möller–Trumbore wants the
 * edges, and precomputing them removes two vector subtractions from the
 * innermost loop of the whole system. Same 36 bytes either way, so it is free.
 *
 * The trailing floats pad to 48 bytes (three 16-byte vectors), which keeps
 * every triangle 16-byte aligned for the GPU. The padding is not wasted: it
 * carries the source primitive id, which reflections need to look up a
 * material and shadows simply ignore.
 */
struct BvhTriangle {
    glm::vec3 v0{0.0f};
    float pad0 = 0.0f;
    glm::vec3 e1{0.0f}; ///< v1 - v0
    uint32_t prim_id = 0;
    glm::vec3 e2{0.0f}; ///< v2 - v0
    float pad2 = 0.0f;
};
static_assert(sizeof(BvhTriangle) == 48, "BvhTriangle must stay 16-byte aligned for the GPU");

/**
 * A bottom-level acceleration structure: one BVH over one model's triangles,
 * in the model's own space.
 *
 * BLAS-per-model is the decision that makes software ray tracing fit in memory
 * here at all. The playground draws 6.2M triangles, but only ~600k of them are
 * unique — the rest is instancing. A flat world BVH would be ~300 MB of
 * triangles alone; one BLAS per model is ~29 MB, built once at load and shared
 * by every instance.
 */
struct Blas {
    std::vector<BvhNode> nodes;
    std::vector<BvhTriangle> triangles;
    glm::vec3 bounds_min{0.0f};
    glm::vec3 bounds_max{0.0f};

    [[nodiscard]] bool empty() const noexcept { return nodes.empty(); }
    [[nodiscard]] std::size_t node_bytes() const noexcept { return nodes.size() * sizeof(BvhNode); }
    [[nodiscard]] std::size_t triangle_bytes() const noexcept {
        return triangles.size() * sizeof(BvhTriangle);
    }
};

struct BvhBuildStats {
    uint32_t node_count = 0;
    uint32_t leaf_count = 0;
    uint32_t max_depth = 0;
    uint32_t triangle_count = 0;
    float average_leaf_size = 0.0f;
    /// Surface-area heuristic cost of the finished tree. Comparable between
    /// builds of the same geometry: lower is a better tree.
    float sah_cost = 0.0f;
};

struct BvhBuildOptions {
    /// Stop splitting at or below this many triangles. 4 is the usual sweet
    /// spot for software traversal: fewer means more nodes to walk, more means
    /// wasted intersection tests.
    uint32_t max_leaf_triangles = 4;
    /// SAH bins per axis. 16 gets within ~1% of a full sweep at a fraction of
    /// the cost; the build runs at load time and should not dominate it.
    uint32_t sah_bins = 16;
    uint32_t max_depth = 64;
};

/**
 * Build a BLAS from an indexed triangle mesh.
 *
 * Binned SAH: for each axis the triangles are bucketed by centroid, the bins
 * are swept to find the cheapest split, and the best axis wins. Median splits
 * are simpler and produce trees that are roughly 1.5-2x slower to traverse —
 * and traversal cost is paid every frame by every ray, while build cost is
 * paid once, so the trade is not close.
 *
 * Deterministic: the same mesh always produces the same tree, so a traced
 * frame is reproducible and can sit inside the regression gate.
 */
[[nodiscard]] Blas build_blas(std::span<const glm::vec3> positions,
                              std::span<const uint32_t> indices,
                              const BvhBuildOptions& options = {},
                              BvhBuildStats* out_stats = nullptr);

/**
 * One instance in the top-level structure: which BLAS, and where.
 *
 * The inverse transform is stored rather than derived because traversal needs
 * it per ray per instance — transforming the RAY into model space is far
 * cheaper than transforming the model's geometry into world space, which is
 * the entire reason a two-level structure exists.
 */
/**
 * SKINNED FLAG, carried in the high bit of `TlasInstance::blas_node_offset`.
 *
 * A skinned instance's BLAS is its model's BIND POSE — the acceleration
 * structure is built once from CPU geometry and the instance carries only the
 * character's root transform. That is fine for the closest-hit paths
 * (reflections, GI probe rays), which need something shadeable where the
 * character is and cannot see the pose error at probe scale. It is NOT fine
 * for the any-hit visibility paths, whose result REPLACES the cascade shadow:
 * there the bind pose is a rigid statue dragged around the world
 * (docs/rt_lighting_plan.md §1 limit 3). Those walks skip flagged instances
 * and test the frame's ShadowCapsules instead (§5.8).
 *
 * WHY A FLAG BIT AND NOT A FIELD. `TlasInstance` is uploaded to the GPU by raw
 * memcpy — the CPU struct IS the shader struct — and it is exactly 96 bytes
 * with no spare word. A new field would grow the per-instance record by a full
 * 16-byte register for one bit, across every instance of every frame. The
 * offset it rides in is an index into the converted BLAS node array, whose size
 * is bounded by the loaded model set: the playground's is ~230k nodes, and the
 * static_assert below pins the ceiling this trick assumes.
 *
 * EVERY READ MASKS. `blas_node_offset` must never be used raw again — see
 * `tlas_blas_node_offset()` here, and the identical masking in
 * `shaders/rt_trace.hlsli`'s two instance loops.
 */
inline constexpr uint32_t k_tlas_skinned_flag = 0x80000000u;
inline constexpr uint32_t k_tlas_node_offset_mask = ~k_tlas_skinned_flag;
/// The largest BLAS node array the flag leaves room for. 2^31 converted
/// 64-byte nodes is 128 GB of node buffer — three orders of magnitude past any
/// GPU's storage-buffer limit — so the bit is free in every sense that matters.
inline constexpr uint32_t k_tlas_max_blas_nodes = k_tlas_node_offset_mask;
static_assert(k_tlas_skinned_flag == (1u << 31), "the skinned flag is the sign bit of the offset");

struct TlasInstance {
    glm::mat4 world_to_model{1.0f};
    glm::vec3 bounds_min{0.0f}; ///< world-space AABB of this instance
    /// Where this instance's BLAS starts in the CONVERTED (64-byte) node
    /// array the shaders index — see `bvh_gpu.hpp` — OR'd with
    /// `k_tlas_skinned_flag` when the instance came from a skinned draw.
    /// `RtScene` assigns both halves; the builder only copies it around.
    /// Read it through `tlas_blas_node_offset()`, never raw.
    uint32_t blas_node_offset = 0;
    glm::vec3 bounds_max{0.0f};
    uint32_t blas_triangle_offset = 0;
};

/// The node offset with the skinned flag masked off — the value traversal
/// indexes `blasNodes` with.
[[nodiscard]] inline constexpr uint32_t tlas_blas_node_offset(const TlasInstance& inst) noexcept {
    return inst.blas_node_offset & k_tlas_node_offset_mask;
}

/// Whether this instance came from a skinned draw, and therefore traces its
/// bind pose. See `k_tlas_skinned_flag`.
[[nodiscard]] inline constexpr bool tlas_instance_is_skinned(const TlasInstance& inst) noexcept {
    return (inst.blas_node_offset & k_tlas_skinned_flag) != 0u;
}

/** A TLAS is itself a BVH, over instance AABBs rather than triangles. */
struct Tlas {
    std::vector<BvhNode> nodes;
    std::vector<TlasInstance> instances;
    /// For each entry of `instances`, its index in the list that was handed to
    /// `build_tlas`. The builder PERMUTES instances into leaf order (a leaf's
    /// primitives have to be contiguous), so a caller that keeps its own list
    /// in its own order has no other way to find the slot holding a given
    /// instance — which is exactly what `refit_tlas` needs before it runs.
    std::vector<uint32_t> source_index;

    [[nodiscard]] bool empty() const noexcept { return nodes.empty(); }
};

/**
 * Build a TLAS over world-space instance bounds.
 *
 * Rebuilt only when the set of instances CHANGES. When the same instances have
 * merely moved, `refit_tlas` updates the existing tree's bounds instead — see
 * `RtScene::set_instances` for the three-case policy.
 */
[[nodiscard]] Tlas build_tlas(std::span<const TlasInstance> instances,
                              const BvhBuildOptions& options = {},
                              BvhBuildStats* out_stats = nullptr);

/**
 * Recompute an existing TLAS's node bounds in place, keeping its topology.
 *
 * The caller has already written this frame's world bounds into
 * `tlas.instances` (using `Tlas::source_index` to find each slot); this walks
 * the tree bottom-up and re-unions every node from its children, or from its
 * instance window for a leaf. O(nodes), no partitioning, no SAH, no
 * allocation — the difference between ~0.5 ms and ~0.02 ms on the playground's
 * 2570 instances.
 *
 * WHAT IT COSTS: a refitted tree is topologically the tree the instances had
 * when they were LAST built, so bounds only ever grow relative to a fresh
 * build and traversal slowly gets worse as things drift apart. That is the
 * caller's problem to bound — `RtScene` watches `sah_cost` and forces a real
 * rebuild when it degrades (with a refit-count backstop).
 *
 * Correctness is not at stake either way: refitted bounds are conservative
 * supersets of the true ones, so the hit SET is unchanged; only the number of
 * boxes a ray walks through moves.
 */
void refit_tlas(Tlas& tlas);

/** SAH cost of a finished tree, for comparing builds. Traversal-cost 1.0,
 *  intersection-cost 1.0, which is the usual normalization. */
[[nodiscard]] float sah_cost(std::span<const BvhNode> nodes);

} // namespace bracken::render::rt
