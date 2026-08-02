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
};

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
     * Rebuild the TLAS from this frame's instances.
     *
     * SKIPPED ENTIRELY WHEN NOTHING MOVED, and that is load-bearing rather than
     * an optimization: the renderer hands this the whole model-command list off
     * every traced frame, so without the check a village that never moves paid a
     * full binned-SAH build (2569 instances ≈ 0.49 ms at -O2, and ~18 ms in an
     * unoptimized tree) on the main thread, every frame, forever. The docs used
     * to claim this skip existed; it did not.
     *
     * The check is a content hash over (handle, transform) — not a caller
     * promise — because the instance list is rebuilt from the packet each frame
     * and there is nobody to make such a promise.
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
    bool has_instance_hash_ = false;
    RtSceneStats stats_{};
};

} // namespace bracken::render::rt
