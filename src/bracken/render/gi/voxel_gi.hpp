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

    /// The frame's clustered lights, so lanterns light the bounce. WITHOUT
    /// these a voxel's only illumination is the sun, so after dusk the whole
    /// clipmap goes black and takes the GI term with it.
    SDL_GPUBuffer* light_data = nullptr;
    std::uint32_t light_count = 0;
    std::uint32_t light_capacity = 0;

    /// IBL sky CAPTURE (raw radiance) for the ambient term — the same texture
    /// the traced probe path samples on a miss, so the two modes agree about
    /// what the sky is worth. `sky_cube_fallback` must always be valid: a
    /// binding slot cannot be left empty.
    SDL_GPUTexture* sky_cube = nullptr;
    SDL_GPUTexture* sky_cube_fallback = nullptr;
    SDL_GPUSampler* sky_sampler = nullptr;

    /// A HARD CUT in the lighting — a teleport, a level change, a host slamming
    /// the environment. An immediate override, NOT the primary staleness
    /// signal: ordinary time-of-day drift is far too slow to register on a
    /// per-frame detector, so the drift case is detected inside `VoxelGi`
    /// against the lighting each cascade was last injected with.
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
    /// Radiance a cone returns for the fraction of itself that escaped the
    /// clipmap. Must be what the traced source's `rtHitSkyRadiance` returns for
    /// a miss, or the two ray sources write different units into the SAME
    /// `ray_results_` buffer. Direction-independent, where the traced path's
    /// miss is not — a cone escaping DOWNWARD past the coarsest cascade gets
    /// sky rather than ground.
    glm::vec3 sky_color{0.0f};
    float bounce_scale = 1.0f;
};

struct VoxelGiStats {
    /// Which cascade revoxelized this frame, or -1.
    int cascade_revoxelized = -1;
    /// Which cascade was re-lit this frame, or -1. A relight without a
    /// revoxelization is the steady state under a moving sun.
    int cascade_relit = -1;
    std::uint32_t cascades_dirty = 0;
    std::uint64_t voxelize_work_items = 0;
    std::uint32_t cone_probes = 0;
    bool injected = false;
};

class VoxelGi {
public:
    [[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipelines);
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
    /// Clear + voxelize (only when `revoxelize`) + inject + mip one cascade.
    bool rebuild_cascade(SDL_GPUCommandBuffer* cmd, int cascade,
                         const VoxelGiFrameParams& params, bool revoxelize);

    gpu::PipelineRegistry* pipelines_ = nullptr;
    VoxelVolume volume_;
    VoxelScene scene_;

    /// Per-cascade "the GEOMETRY no longer matches the window".
    std::array<bool, VOXEL_CASCADE_COUNT> dirty_{};
    /// Per-cascade "the geometry is fine but the RADIANCE is stale". Kept apart
    /// from `dirty_` because a sun move needs only reinjection, which is
    /// O(volume) and cheap, where revoxelization is O(instances x triangles) and
    /// is not.
    std::array<bool, VOXEL_CASCADE_COUNT> relight_{};

    /// The lighting each cascade was last INJECTED with, which is what staleness
    /// is measured against. A PER-FRAME delta cannot do this job: a 24-minute
    /// day moves the sun 0.004 degrees per frame, under any threshold worth
    /// having, so a per-frame test never fires and the clipmap's radiance is
    /// only ever refreshed when a cascade happens to scroll.
    struct InjectedLighting {
        glm::vec3 sun_dir{0.0f};
        glm::vec3 sun_radiance{0.0f};
        glm::vec3 sky_color{0.0f};
        bool valid = false;
    };
    std::array<InjectedLighting, VOXEL_CASCADE_COUNT> injected_{};
    /// Round-robin cursor, so a permanently-scrolling cascade 0 cannot starve
    /// the coarser ones.
    int next_cascade_ = 0;
    VoxelGiStats stats_{};
};

} // namespace bracken::render::gi
