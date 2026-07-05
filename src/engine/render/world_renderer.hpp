#pragma once

#include "engine/gpu/gpu_buffer.hpp"
#include "engine/gpu/gpu_device.hpp"
#include "engine/gpu/pipeline_registry.hpp"
#include "engine/model_loader.hpp"
#include "engine/scene/frustum.hpp"
#include "engine/render/environment.hpp"
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <SDL3/SDL_gpu.h>
#include <vector>

namespace mmo::engine::render {

namespace gpu = mmo::engine::gpu;

/**
 * WorldRenderer handles environmental world rendering:
 * - Skybox
 * - Grid
 *
 * Note: Rocks and trees are now rendered as server-side entities
 *
 * SDL3 GPU Migration: This class now uses SDL3 GPU API instead of OpenGL.
 * - VAO/VBO replaced with GPUBuffer
 * - GL shaders replaced with PipelineRegistry
 * - Draw calls use SDL_DrawGPUPrimitives
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
    [[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry, float world_width, float world_height,
              ModelManager* model_manager);

    /**
     * Clean up resources.
     */
    void shutdown();

    /**
     * Set terrain height callback for proper object placement.
     */
    void set_terrain_height_func(std::function<float(float, float)> func) { terrain_height_func_ = std::move(func); }

    /**
     * Set the atmosphere state (sun/sky/fog/ambient single source of truth).
     * Called once per frame by the scene layer before rendering.
     */
    void set_environment(const EnvironmentState& env) { env_ = env; }
    const EnvironmentState& environment() const { return env_; }

    /**
     * Update time-based effects. WorldRenderer is the single owner of sky
     * time; callers needing the shared clock (grass wind, cloud scroll) read
     * sky_time() instead of accumulating their own copy.
     */
    void update(float dt);
    float sky_time() const { return skybox_time_; }

    /**
     * Render skybox.
     * @param pass Active render pass
     * @param cmd Command buffer for uniform uploads
     */
    void render_skybox(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view,
                       const glm::mat4& projection);

    /**
     * Render debug grid.
     * @param pass Active render pass
     * @param cmd Command buffer for uniform uploads
     */
    void render_grid(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view,
                     const glm::mat4& projection);

    // Accessors (derived from the environment state)
    glm::vec3 sun_direction() const { return -env_.sun_dir; } // direction TO the sun
    const glm::vec3& light_dir() const { return env_.sun_dir; } // direction light travels

private:
    void create_skybox_mesh();
    void create_grid_mesh();

    float get_terrain_height(float x, float z) const;

    float world_width_ = 0.0f;
    float world_height_ = 0.0f;

    gpu::GPUDevice* device_ = nullptr;
    gpu::PipelineRegistry* pipeline_registry_ = nullptr;
    ModelManager* model_manager_ = nullptr;
    std::function<float(float, float)> terrain_height_func_;

    // Skybox
    std::unique_ptr<gpu::GPUBuffer> skybox_vertex_buffer_;
    float skybox_time_ = 0.0f;

    // Cached skybox inverse view-projection (rotation-only view); recomputed
    // only when the camera rotation or projection changes.
    glm::mat4 cached_sky_view_{0.0f};
    glm::mat4 cached_sky_projection_{0.0f};
    glm::mat4 cached_sky_inv_vp_{1.0f};

    // Grid
    std::unique_ptr<gpu::GPUBuffer> grid_vertex_buffer_;
    uint32_t grid_vertex_count_ = 0;

    // Atmosphere (sun/sky/fog/ambient single source of truth)
    EnvironmentState env_{};

    // Sampler for textures
    SDL_GPUSampler* sampler_ = nullptr;
};

} // namespace mmo::engine::render
