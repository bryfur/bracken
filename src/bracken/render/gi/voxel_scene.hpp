#pragma once

/**
 * The geometry the voxelizer reads (RG1) — a shared model-space triangle buffer
 * plus a per-instance placement list.
 *
 * DELIBERATELY NOT `RtScene`. The two gather the same source (retained CPU
 * geometry plus the per-triangle material proxies `model_material_proxy.hpp`
 * bakes at load) and then diverge completely: `RtScene` builds a BVH so a RAY
 * can find the nearest triangle, and this builds nothing at all, because a
 * voxelizer visits every triangle exactly once and never searches. Sharing an
 * acceleration structure between them would couple the raster GI path to the
 * ray-traced one, and the whole point of this path is that it works with
 * `rt_shadows` and `rt_reflections` off and no acceleration structure in
 * memory.
 *
 * WHAT IT COSTS. One `VoxelTriangle` (48 B) per source triangle across all
 * loaded models — the same per-model, not per-instance, storage a BLAS uses, so
 * a village placing one crate mesh two hundred times stores it once. Instances
 * are 80 B each.
 *
 * Tier: main-thread for `upload`; `rebuild` is device-free and could move to the
 * Prepare stage if it ever measures.
 */

#include "bracken/contracts/model_handle.hpp"
#include "bracken/gpu/gpu_buffer.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace bracken {
class ModelManager;
}
namespace bracken::gpu {
class GPUDevice;
}

namespace bracken::render::gi {

/**
 * One model-space triangle, stored as origin + two edges so the voxelizer does
 * not repeat the subtractions per sample. Mirrors `VoxelTriangle` in
 * `shaders/voxel_voxelize.comp.hlsl` byte for byte — uploaded by raw memcpy.
 */
struct VoxelTriangle {
    glm::vec3 v0{0.0f};
    /// The triangle's baked material, packed into the word the voxelizer stores
    /// VERBATIM: base colour in the low three bytes, HIGH BYTE = emissive
    /// intensity as a fraction of `VOXEL_EMISSIVE_SCALE` (voxel_scene.cpp),
    /// which must equal `VOXEL_EMISSIVE_SCALE` in voxel_inject.comp.hlsl — the
    /// two halves of one quantization and there is no test that pins them.
    ///
    /// The default is white and NON-emissive. `0xFFFFFFFF` would be a full-scale
    /// emitter, which is what the high byte meaning emission costs the obvious
    /// spelling — the same mistake, made once for real, took the night frame
    /// from 35.8 to 96.8 mean luma.
    std::uint32_t packed_albedo = 0x00FFFFFFu;
    glm::vec3 e1{0.0f};
    std::uint32_t _pad0 = 0;
    glm::vec3 e2{0.0f};
    float _pad1 = 0.0f;
};
static_assert(sizeof(VoxelTriangle) == 48, "VoxelTriangle is uploaded by memcpy");

/**
 * One placement. `first_triangle`/`triangle_count` name a window of the shared
 * triangle buffer, exactly as a BLAS does.
 */
struct VoxelInstance {
    glm::mat4 model_to_world{1.0f};
    std::uint32_t first_triangle = 0;
    std::uint32_t triangle_count = 0;
    /// Prefix sum of `triangle_count` over PRECEDING instances. This is what
    /// turns the voxelizer's flat dispatch index into an (instance, triangle)
    /// pair with a binary search and no second indirection buffer.
    std::uint32_t work_offset = 0;
    std::uint32_t _pad = 0;
};
static_assert(sizeof(VoxelInstance) == 80, "VoxelInstance is uploaded by memcpy");

struct VoxelSceneStats {
    std::uint32_t models_with_geometry = 0;
    std::uint32_t triangles = 0;
    std::uint32_t instances = 0;
    /// Total (instance, triangle) pairs — the voxelizer's dispatch extent, and
    /// the number that actually predicts its cost.
    std::uint64_t work_items = 0;
    std::uint64_t triangle_bytes = 0;
    std::uint64_t instance_bytes = 0;
    /// Instances dropped because their model had no retained CPU geometry.
    /// Counted rather than ignored — a silently unvoxelized building is a hole
    /// in the GI that looks like a bug in the cone tracer.
    std::uint32_t instances_without_geometry = 0;
};

class VoxelScene {
public:
    /// Gather every model's triangles. Idempotent per model set: re-running with
    /// the same `ModelManager` contents rebuilds identical buffers, which the
    /// deterministic smoke gate requires.
    void build_models(const ModelManager& models);

    /// Refresh the placement list. Cheap — no geometry is touched, so this may
    /// run every frame; it early-outs when the list is unchanged.
    void set_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances);

    /// Push whatever is dirty to the GPU. Main-thread.
    void upload(gpu::GPUDevice& device);

    void shutdown();

    [[nodiscard]] bool ready() const {
        return triangle_buffer_ != nullptr && instance_buffer_ != nullptr && !instances_.empty();
    }
    [[nodiscard]] gpu::GPUBuffer* triangle_buffer() const { return triangle_buffer_.get(); }
    [[nodiscard]] gpu::GPUBuffer* instance_buffer() const { return instance_buffer_.get(); }
    [[nodiscard]] std::uint32_t instance_count() const {
        return static_cast<std::uint32_t>(instances_.size());
    }
    [[nodiscard]] std::uint64_t work_count() const { return stats_.work_items; }
    [[nodiscard]] const VoxelSceneStats& stats() const { return stats_; }

private:
    /// Where each model's triangles start in `triangles_`, indexed by handle.
    struct ModelRange {
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };

    std::vector<VoxelTriangle> triangles_;
    std::vector<ModelRange> model_ranges_;
    std::vector<VoxelInstance> instances_;

    std::unique_ptr<gpu::GPUBuffer> triangle_buffer_;
    std::unique_ptr<gpu::GPUBuffer> instance_buffer_;

    bool triangles_dirty_ = false;
    bool instances_dirty_ = false;
    /// Content hash of the last accepted instance list, so an unchanged frame
    /// does not re-upload 200 KB for nothing.
    std::uint64_t instance_hash_ = 0;
    bool has_instance_hash_ = false;

    VoxelSceneStats stats_{};
};

} // namespace bracken::render::gi
