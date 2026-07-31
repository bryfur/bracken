#pragma once

#include "bracken/contracts/environment.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <SDL3/SDL_gpu.h>

namespace bracken::render {

namespace gpu = bracken::gpu;

/**
 * WorldRenderer draws the procedural sky (a fullscreen triangle — gradient,
 * sun disc, clouds, and starfield are computed analytically in
 * skybox.frag.hlsl; there is no cubemap) and holds the per-frame
 * EnvironmentState copy the sky shader reads its atmosphere colors from.
 */
class WorldRenderer {
public:
    WorldRenderer();
    ~WorldRenderer();

    // Non-copyable
    WorldRenderer(const WorldRenderer&) = delete;
    WorldRenderer& operator=(const WorldRenderer&) = delete;
    WorldRenderer(WorldRenderer&&) = delete;
    WorldRenderer& operator=(WorldRenderer&&) = delete;

    /**
     * Initialize world rendering resources.
     * @param device GPU device for resource creation
     * @param pipeline_registry Pipeline registry for shader pipelines
     */
    [[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry);

    /**
     * Clean up resources.
     */
    void shutdown();

    /**
     * Set the atmosphere state (sun/sky/fog/ambient single source of truth).
     * Called once per frame by the scene layer before rendering.
     */
    void set_environment(const EnvironmentState& env) { env_ = env; }
    const EnvironmentState& environment() const { return env_; }

    /**
     * Set the sky-time clock. sky_time() is the accumulated cloud scroll
     * PHASE (dt integrated scaled by EnvironmentState::cloud_speed) and
     * drives only the skybox's cloud scroll (the `time` uniform in
     * skybox.frag) and the IBL capture's frozen cloud snapshot; grass wind
     * runs on the separately accumulated scene clock. The SIM stage advances
     * the phase and carries it in the FramePacket's engine section
     * (docs/multithreading_plan.md Phase 4); SceneRenderer adopts it here —
     * the renderer no longer integrates dt itself.
     */
    void set_sky_time(float time) { skybox_time_ = time; }
    float sky_time() const { return skybox_time_; }

    /**
     * Render skybox.
     * @param pass Active render pass
     * @param cmd Command buffer for uniform uploads
     */
    void render_skybox(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view,
                       const glm::mat4& projection);

    // Accessors (derived from the environment state)
    glm::vec3 sun_direction() const { return -env_.sun_dir; } // direction TO the sun
    const glm::vec3& light_dir() const { return env_.sun_dir; } // direction light travels

private:
    void create_skybox_mesh();

    gpu::GPUDevice* device_ = nullptr;
    gpu::PipelineRegistry* pipeline_registry_ = nullptr;

    // Skybox
    std::unique_ptr<gpu::GPUBuffer> skybox_vertex_buffer_;
    float skybox_time_ = 0.0f;

    // Cached skybox inverse view-projection (rotation-only view); recomputed
    // only when the camera rotation or projection changes.
    glm::mat4 cached_sky_view_{0.0f};
    glm::mat4 cached_sky_projection_{0.0f};
    glm::mat4 cached_sky_inv_vp_{1.0f};

    // Atmosphere (sun/sky/fog/ambient single source of truth)
    EnvironmentState env_{};
};

} // namespace bracken::render
