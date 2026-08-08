#pragma once

/**
 * The RASTER GI path's frame driver — clear, voxelize, inject, mip, cone-trace.
 *
 * Owns the clipmap (`VoxelVolume`) and the geometry it is built from
 * (`VoxelScene`), and issues the four compute passes in order. It produces
 * exactly one thing: the probe field's `ray_results_` buffer, filled to the same
 * contract `gi_probe_trace.comp.hlsl` fills it to. Everything the probe field
 * does with that buffer is unchanged and shared with the ray-traced path.
 *
 * ---------------------------------------------------------------------------
 * THE CADENCE, which is most of what makes this affordable
 * ---------------------------------------------------------------------------
 *
 * Voxelizing is O(instances x triangles) and the finest cascade scrolls
 * whenever the camera moves a voxel, so revoxelizing everything that moved
 * every frame would dominate the frame. Instead AT MOST ONE CASCADE
 * revoxelizes per frame, chosen round-robin among those that are dirty — the
 * same trade the probe field's own rota makes, and for the same reason: a
 * cascade that is one frame stale is wrong by one voxel at its own scale, which
 * is invisible, while the cost of pretending otherwise is not.
 *
 * Injection is O(volume) and cheap, but it only has to re-run when the LIGHTING
 * changed or the cascade was revoxelized, so it follows the same schedule.
 *
 * The cone trace runs every frame for whichever probes the field scheduled,
 * because that is the probe field's own cadence and this is standing in for its
 * ray source.
 *
 * Tier: main-thread.
 */

#include "bracken/render/gi/voxel_scene.hpp"
#include "bracken/render/gi/voxel_volume.hpp"

#include <SDL3/SDL_gpu.h>
#include <glm/glm.hpp>

#include <cstdint>

namespace bracken {
class ModelManager;
}
namespace bracken::gpu {
class GPUDevice;
class PipelineRegistry;
} // namespace bracken::gpu

namespace bracken::render::gi {

/// What the driver needs from the frame that it cannot derive itself. Deliberately
/// a plain struct of values — no renderer type is reachable from here.
struct VoxelGiFrameParams {
    glm::vec3 camera_position{0.0f};
    glm::vec3 camera_forward{0.0f, 0.0f, -1.0f};

    /// The CSM the injection pass shadows the sun with — the same atlas the
    /// raster path already rendered, which is the entire economy of raster GI.
    SDL_GPUTexture* shadow_atlas = nullptr;
    SDL_GPUSampler* shadow_cmp_sampler = nullptr;
    SDL_GPUSampler* shadow_raw_sampler = nullptr;
    /// The shadow_common.hlsli uniform block, already assembled by the caller
    /// (it is the same block every shadow consumer receives).
    const void* shadow_block = nullptr;
    std::size_t shadow_block_size = 0;

    glm::vec3 sun_direction{0.0f, -1.0f, 0.0f};
    glm::vec3 sun_color{1.0f};
    float sun_intensity = 1.0f;
    glm::vec3 sky_color{0.0f};
    float sky_floor = 0.0f;

    /// True when the sun or sky moved enough that injected radiance is stale.
    bool lighting_changed = false;
};

/// The cone-trace half's inputs, which come from the probe field rather than
/// from the frame.
struct VoxelConeTraceParams {
    /// Cascade selection is camera-relative, so the cone tracer needs the same
    /// position the clipmap was scrolled with. Passing a stale one silently
    /// selects the wrong cascade and every cone reads the wrong scale of the
    /// world.
    glm::vec3 camera_position{0.0f};
    SDL_GPUBuffer* probe_updates = nullptr;
    SDL_GPUBuffer* ray_results = nullptr;
    std::uint32_t probe_count = 0;
    std::uint32_t ray_count = 0;
    std::uint32_t frame_index = 0;
    /// Must be the ray-traced path's value — the visibility moments downstream
    /// would otherwise change meaning between GI modes.
    float ray_max_distance = 6000.0f;
    float bounce_scale = 1.0f;
};

struct VoxelGiStats {
    /// Which cascade revoxelized this frame, or -1.
    int cascade_revoxelized = -1;
    std::uint32_t cascades_dirty = 0;
    std::uint64_t voxelize_work_items = 0;
    std::uint32_t cone_probes = 0;
    bool injected = false;
};

class VoxelGi {
public:
    /**
     * Allocate and arm the driver.
     *
     * CURRENTLY ALWAYS FAILS, deliberately and loudly — 3D storage-image writes
     * fault on this SDL3 backend. The implementation carries the full
     * investigation: what was ruled out, the three real bugs found on the way,
     * and what the next step is. `init_unchecked` is the same function without
     * the refusal, kept so the path can be exercised the moment the platform
     * question is settled.
     */
    [[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipelines);
    /// `init` without the platform refusal. See `init`.
    [[nodiscard]] bool init_unchecked(gpu::GPUDevice& device, gpu::PipelineRegistry& pipelines);
    void shutdown();
    [[nodiscard]] bool ready() const { return volume_.ready(); }

    /// Gather the scene's triangles. Call when the model set changes.
    void build_models(const ModelManager& models);
    /// Refresh placements. Cheap and safe to call every frame.
    void set_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances);
    /// Push dirty geometry buffers. Main-thread, outside a pass.
    void upload(gpu::GPUDevice& device);

    /**
     * Scroll, then revoxelize + reinject AT MOST ONE cascade. Returns false when
     * nothing could run (not ready, no geometry, missing pipeline), which the
     * caller should treat as "the clipmap is not usable this frame".
     */
    bool update(SDL_GPUCommandBuffer* cmd, const VoxelGiFrameParams& params);

    /**
     * Fill `params.ray_results` by cone-tracing the clipmap. This is the drop-in
     * replacement for `ProbeVolume::trace`'s dispatch.
     */
    bool cone_trace(SDL_GPUCommandBuffer* cmd, const VoxelConeTraceParams& params);

    [[nodiscard]] const VoxelVolume& volume() const { return volume_; }
    [[nodiscard]] const VoxelScene& scene() const { return scene_; }
    [[nodiscard]] const VoxelGiStats& stats() const { return stats_; }

private:
    /// Clear + voxelize + inject one cascade.
    bool rebuild_cascade(SDL_GPUCommandBuffer* cmd, int cascade,
                         const VoxelGiFrameParams& params);

    gpu::PipelineRegistry* pipelines_ = nullptr;
    VoxelVolume volume_;
    VoxelScene scene_;

    /// Per-cascade "contents no longer match the window or the lighting".
    std::array<bool, VOXEL_CASCADE_COUNT> dirty_{};
    /// Round-robin cursor, so a permanently-scrolling cascade 0 cannot starve
    /// the coarser ones.
    int next_cascade_ = 0;
    VoxelGiStats stats_{};
};

} // namespace bracken::render::gi
