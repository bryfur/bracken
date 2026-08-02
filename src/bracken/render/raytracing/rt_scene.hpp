#pragma once

#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render/raytracing/bvh.hpp"
#include "bracken/render/raytracing/bvh_gpu.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace bracken::render::rt {

/** Extra per-triangle data reflections need and shadows do not. */
struct TriangleAttributes {
    glm::vec3 n0{0.0f, 1.0f, 0.0f};
    float pad0 = 0.0f;
    glm::vec3 n1{0.0f, 1.0f, 0.0f};
    float pad1 = 0.0f;
    glm::vec3 n2{0.0f, 1.0f, 0.0f};
    float pad2 = 0.0f;
    glm::vec4 base_color{1.0f};
};
static_assert(sizeof(TriangleAttributes) == 64, "attributes must stay 16-byte aligned");

struct RtSceneStats {
    uint32_t models = 0;
    uint32_t instances = 0;
    uint32_t unique_triangles = 0;  ///< triangles actually stored
    uint32_t drawn_triangles = 0;   ///< triangles after instancing, for comparison
    /// Counts of what is actually UPLOADED, i.e. 64-byte `BvhNodeGpu`s, not
    /// the 32-byte build nodes they were converted from (roughly half as many:
    /// only interior nodes survive the conversion).
    uint32_t blas_nodes = 0;
    uint32_t tlas_nodes = 0;
    float build_ms = 0.0f;
    /// The three components of `gpu_bytes`, tracked separately because they
    /// have separate levers: BLAS follows the loaded model set, attributes
    /// exist only when reflections asked for them (and are the largest item
    /// when they do), and TLAS follows the per-frame INSTANCE count — the part
    /// GraphicsSettings::rt_max_instance_range moves.
    std::size_t blas_bytes = 0;      ///< concatenated BLAS nodes + triangles
    std::size_t tlas_bytes = 0;      ///< TLAS nodes + instance records
    std::size_t attribute_bytes = 0; ///< per-triangle shading attributes; 0 without reflections
    /// blas_bytes + tlas_bytes + attribute_bytes.
    std::size_t gpu_bytes = 0;
    /// How many times the TLAS has actually been rebuilt. Exposed because the
    /// dirty check is a PERFORMANCE contract, not just an internal detail: a
    /// static scene must show 1 here no matter how many frames have gone by,
    /// and the only way to keep that true is to be able to assert on it.
    uint32_t tlas_builds = 0;
    /// How many times the TLAS was REFITTED instead — same instances, new
    /// transforms. Reported next to `tlas_builds` everywhere, because the two
    /// only mean something as a pair: a live scene with one animated character
    /// should read builds≈1, refits≈frames, and any drift back towards
    /// builds≈frames is the topology check failing to engage.
    uint32_t tlas_refits = 0;
};

/// A refit tree keeps the topology the instances had when they were last
/// built, so its bounds only ever grow. Two triggers force a real rebuild:
/// the measured one (SAH cost against the last full build) and this counter as
/// a backstop, so a pathological motion pattern that keeps the cost flat while
/// the tree quietly stops matching the scene still gets a fresh build.
inline constexpr uint32_t k_tlas_max_refits_between_builds = 64;
/// SAH cost growth, relative to the last full build, that forces a rebuild.
/// 1.5 is deliberately loose: a rebuild costs ~25x a refit here, so paying one
/// to recover a few percent of traversal cost is a bad trade, whereas a tree
/// half again as expensive to walk as a fresh one is worth 0.5 ms.
inline constexpr float k_tlas_refit_sah_rebuild_ratio = 1.5f;

/**
 * The GPU-side acceleration structure: every model's BLAS concatenated into
 * one pair of buffers, plus a TLAS over the frame's instances.
 *
 * CONCATENATED, NOT PER-MODEL. One buffer per model would mean one bind per
 * instance during traversal, which a compute shader cannot do — it has a fixed
 * binding set. So every BLAS is appended into a shared node buffer and a
 * shared triangle buffer, and each TLAS instance carries the offsets where its
 * model's data begins. Traversal adds the offset; nothing is rebound.
 *
 * ATTRIBUTES ARE OPTIONAL. Normals and colours are needed to SHADE a hit,
 * which only reflections do — a shadow ray asks a yes/no question. They are a
 * separate buffer, uploaded only when reflections are enabled, so a game using
 * ray-traced shadows alone pays 48 bytes per triangle instead of 112.
 *
 * Tier: main-thread. build_blas_for_models() fans the per-model builds out
 * over the job pool internally, but GPU upload is main-thread as always.
 */
class RtScene {
public:
    RtScene() = default;
    ~RtScene();
    RtScene(const RtScene&) = delete;
    RtScene& operator=(const RtScene&) = delete;

    /**
     * Build a BLAS for every model that has geometry, on the job pool.
     *
     * Call once after models are loaded. Models whose CPU-side geometry was
     * released at GPU upload are skipped with a warning rather than silently
     * producing an empty structure — the same trap load_model_with_lods
     * documents.
     *
     * @param want_attributes also gather normals/colours for reflections.
     */
    void build_blas_for_models(const ModelManager& models, bool want_attributes);

    /**
     * Take this frame's instances, doing the least work they permit.
     *
     * THREE CASES, decided from two hashes computed in one pass over the list.
     * Neither is a caller promise — the instance list is rebuilt from the frame
     * packet every frame and there is nobody to make such a promise.
     *
     *  a. **Content hash unchanged** — nothing moved, nothing appeared. Keep
     *     the TLAS, keep the buffers, do no work. Load-bearing rather than an
     *     optimization: without it a village that never moves paid a full
     *     binned-SAH build (2569 instances ≈ 0.49 ms at -O2, ~18 ms in an
     *     unoptimized tree) on the main thread every frame, forever.
     *  b. **Topology hash unchanged, content hash changed** — the same models
     *     in the same order, moved. REFIT: recompute the world bounds of the
     *     instances whose transform actually differs from last frame, then
     *     re-union the existing tree's node bounds bottom-up (`refit_tlas`).
     *     No SAH, no partition, no allocation.
     *  c. **Topology hash changed** — an instance was added, removed, or the
     *     order changed. Full rebuild; there is no tree to refit onto.
     *
     * Case (b) is why this is not a footnote: case (a) sounded like it covered
     * the playground and did not. A single animated character changes the
     * content hash every frame, so before the refit path landed the "static
     * scene builds once" contract engaged in exactly zero live frames and
     * `RTInstances` was the most expensive CPU pass in the frame.
     *
     * A refit tree keeps the topology the instances had when they were last
     * built, so it degrades as they drift; see
     * `k_tlas_refit_sah_rebuild_ratio` and `k_tlas_max_refits_between_builds`
     * for the two triggers that force a real rebuild back.
     */
    void set_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances);

    /**
     * Upload whatever changed. Main-thread; no-op when nothing is dirty.
     *
     * @param cmd The frame's command buffer. The TLAS lives on persistent
     *   dynamic buffers and is recorded into it, so a moving scene costs one
     *   copy rather than two buffer creations plus its own command-buffer
     *   submit in the middle of frame recording. Passing null falls back to
     *   immediate create-and-submit, which is what a host calling this outside
     *   a frame needs.
     */
    void upload(gpu::GPUDevice& device, SDL_GPUCommandBuffer* cmd = nullptr);

    [[nodiscard]] bool ready() const noexcept {
        return tlas_node_buffer_ != nullptr && !tlas_gpu_nodes_.empty();
    }
    /// Nodes in the UPLOADED (64-byte) TLAS — what the shaders index.
    [[nodiscard]] uint32_t tlas_node_count() const noexcept {
        return static_cast<uint32_t>(tlas_gpu_nodes_.size());
    }
    [[nodiscard]] const RtSceneStats& stats() const noexcept { return stats_; }
    [[nodiscard]] bool has_attributes() const noexcept { return attribute_buffer_ != nullptr; }

    // Buffers, in the order the traversal shaders bind them.
    [[nodiscard]] SDL_GPUBuffer* tlas_nodes() const;
    [[nodiscard]] SDL_GPUBuffer* tlas_instances() const;
    [[nodiscard]] SDL_GPUBuffer* blas_nodes() const;
    [[nodiscard]] SDL_GPUBuffer* blas_triangles() const;
    [[nodiscard]] SDL_GPUBuffer* triangle_attributes() const;

private:
    struct ModelEntry {
        /// Index of this model's root in the concatenated CONVERTED node array.
        uint32_t node_offset = 0;
        uint32_t triangle_offset = 0;
        uint32_t node_count = 0;
        uint32_t triangle_count = 0;
        glm::vec3 bounds_min{0.0f};
        glm::vec3 bounds_max{0.0f};
        bool valid = false;
    };

    /// Case (c) of set_instances: throw the tree away and build a new one.
    void rebuild_tlas(std::span<const std::pair<ModelHandle, glm::mat4>> instances);
    /// Case (b): update the moved instances and re-union the existing tree.
    /// Returns false when the result degraded past
    /// `k_tlas_refit_sah_rebuild_ratio`, meaning the caller should rebuild
    /// instead — the refit is left in place and simply overwritten.
    [[nodiscard]] bool refit_tlas_in_place(
        std::span<const std::pair<ModelHandle, glm::mat4>> instances);

    std::unordered_map<ModelHandle, ModelEntry> entries_;
    /// The concatenated BLAS nodes in UPLOAD form. The 32-byte build nodes are
    /// converted per model as they are appended and are not kept: nothing
    /// after the build reads them.
    std::vector<BvhNodeGpu> blas_nodes_;
    std::vector<BvhTriangle> blas_triangles_;
    std::vector<TriangleAttributes> attributes_;
    Tlas tlas_;
    /// The TLAS in upload form, reconverted on every rebuild. A member rather
    /// than a local so the per-rebuild path allocates nothing once warm — same
    /// reason as `convert_scratch_`.
    std::vector<BvhNodeGpu> tlas_gpu_nodes_;
    std::vector<uint32_t> convert_scratch_;

    std::unique_ptr<gpu::GPUBuffer> blas_node_buffer_;
    std::unique_ptr<gpu::GPUBuffer> blas_triangle_buffer_;
    std::unique_ptr<gpu::GPUBuffer> attribute_buffer_;
    std::unique_ptr<gpu::GPUBuffer> tlas_node_buffer_;
    std::unique_ptr<gpu::GPUBuffer> tlas_instance_buffer_;
    /// Byte capacity of the two persistent TLAS buffers above. They are grown
    /// (reallocated) only when the instance count outgrows them, so a scene
    /// whose instance count is stable never reallocates.
    std::size_t tlas_node_capacity_ = 0;
    std::size_t tlas_instance_capacity_ = 0;

    bool blas_dirty_ = false;
    bool tlas_dirty_ = false;
    /// Content hash of the last instance list accepted by set_instances, and
    /// whether one has been seen at all. An empty list is a legitimate state
    /// distinct from "never called", hence the flag rather than a sentinel.
    uint64_t instance_hash_ = 0;
    /// Hash of the same list's COUNT AND HANDLES only — everything that decides
    /// the tree's shape-eligibility, and nothing that merely decides where the
    /// boxes are. Equal topology hash + unequal content hash is precisely the
    /// refit case, and both come out of the same single pass over the list.
    uint64_t topology_hash_ = 0;
    bool has_instance_hash_ = false;
    /// Every instance's transform as it arrived last time, indexed by INPUT
    /// index (i.e. before the no-BLAS filter). The refit path compares against
    /// this so it recomputes an inverse and eight corner transforms only for
    /// the instances that actually moved — in the playground, the character
    /// and nothing else, out of 2570.
    std::vector<glm::mat4> last_transforms_;
    /// For each slot of `tlas_.instances`, the INPUT index it came from.
    /// Composed at build time from the builder's `Tlas::source_index` (slot ->
    /// filtered index) and the filter's own mapping (filtered -> input), so
    /// the refit path can walk slots and never needs an inverse permutation.
    std::vector<uint32_t> input_index_of_slot_;
    /// Scratch for the rebuild path's filtered instance list; a member so a
    /// rebuild allocates nothing once warm.
    std::vector<TlasInstance> tlas_build_scratch_;
    std::vector<uint32_t> filtered_to_input_scratch_;
    /// `sah_cost` of the tree the last FULL build produced, and how many refits
    /// have been stacked on it since. The pair drives the two rebuild triggers.
    float last_build_sah_ = 0.0f;
    uint32_t refits_since_build_ = 0;
    RtSceneStats stats_{};
};

} // namespace bracken::render::rt
