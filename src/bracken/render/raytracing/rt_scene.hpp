#pragma once

#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render/raytracing/bvh.hpp"

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
    uint32_t blas_nodes = 0;
    uint32_t tlas_nodes = 0;
    float build_ms = 0.0f;
    std::size_t gpu_bytes = 0;
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

    /** Rebuild the TLAS from this frame's instances. Cheap (1024 instances in
     *  ~134 us) but skipped entirely when nothing moved. */
    void set_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances);

    /** Upload whatever changed. Main-thread; no-op when nothing is dirty. */
    void upload(gpu::GPUDevice& device);

    [[nodiscard]] bool ready() const noexcept {
        return tlas_node_buffer_ != nullptr && !tlas_.empty();
    }
    [[nodiscard]] uint32_t tlas_node_count() const noexcept {
        return static_cast<uint32_t>(tlas_.nodes.size());
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
        uint32_t node_offset = 0;
        uint32_t triangle_offset = 0;
        uint32_t node_count = 0;
        glm::vec3 bounds_min{0.0f};
        glm::vec3 bounds_max{0.0f};
        bool valid = false;
    };

    std::unordered_map<ModelHandle, ModelEntry> entries_;
    std::vector<BvhNode> blas_nodes_;
    std::vector<BvhTriangle> blas_triangles_;
    std::vector<TriangleAttributes> attributes_;
    Tlas tlas_;

    std::unique_ptr<gpu::GPUBuffer> blas_node_buffer_;
    std::unique_ptr<gpu::GPUBuffer> blas_triangle_buffer_;
    std::unique_ptr<gpu::GPUBuffer> attribute_buffer_;
    std::unique_ptr<gpu::GPUBuffer> tlas_node_buffer_;
    std::unique_ptr<gpu::GPUBuffer> tlas_instance_buffer_;

    bool blas_dirty_ = false;
    bool tlas_dirty_ = false;
    RtSceneStats stats_{};
};

} // namespace bracken::render::rt
