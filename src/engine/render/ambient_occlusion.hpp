#pragma once

#include "engine/gpu/gpu_device.hpp"
#include "engine/gpu/gpu_texture.hpp"
#include "engine/gpu/gpu_uniforms.hpp"
#include "engine/gpu/pipeline_registry.hpp"
#include <array>
#include <glm/glm.hpp>
#include <memory>

namespace mmo::engine::render {

namespace gpu = mmo::engine::gpu;

namespace detail {

/**
 * Base directions for the "GTAO" pass (ao_mode == 2): evenly spaced
 * unit-circle directions in xy, entry i at angle (i % num_directions) /
 * num_directions * 2π (num_directions is clamped to >= 1). Rebuilt on every
 * call so a changed num_directions takes effect immediately — a previous
 * version latched the first call's count in a function-local static.
 * Exposed here for testing.
 */
std::array<glm::vec4, gpu::GTAOUniforms::KERNEL_SIZE> gtao_direction_kernel(int num_directions);

} // namespace detail

/**
 * Ambient Occlusion manager (ao_mode 1 = SSAO, 2 = "GTAO" — the latter is an
 * HBAO-style directional variant, not true ground-truth AO; the mode name is
 * kept for settings compatibility).
 *
 * When enabled, the scene is rendered to an offscreen color+depth target.
 * AO is computed from the depth buffer at half resolution, bilaterally
 * blurred, then upsampled depth-aware (joint bilateral) and composited with
 * the scene color onto the swapchain.
 */
class AmbientOcclusion {
public:
    AmbientOcclusion();
    ~AmbientOcclusion();

    AmbientOcclusion(const AmbientOcclusion&) = delete;
    AmbientOcclusion& operator=(const AmbientOcclusion&) = delete;
    AmbientOcclusion(AmbientOcclusion&&) = delete;
    AmbientOcclusion& operator=(AmbientOcclusion&&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device, int width, int height);
    void shutdown();
    void resize(int width, int height);

    /**
     * Set the linear clear color for the offscreen scene target. Should track
     * EnvironmentState::fog_color_horizon so uncovered pixels blend into the
     * sky/fog.
     */
    void set_clear_color(glm::vec3 linear) { clear_color_ = linear; }

    // Offscreen pass (replaces begin_main_pass when AO is on)
    SDL_GPURenderPass* begin_offscreen_pass(SDL_GPUCommandBuffer* cmd);
    void end_offscreen_pass();

    // Post-processing passes
    void render_ssao_pass(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, const glm::mat4& projection,
                          const glm::mat4& inv_projection, float radius_scale = 1.0f);
    void render_gtao_pass(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, const glm::mat4& projection,
                          const glm::mat4& inv_projection, float radius_scale = 1.0f);
    void render_blur_pass(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines);

    struct CompositeParams {
        float bloom_strength = 0.35f;
        float ao_strength = 1.0f;
        float exposure = 1.0f;
        int tonemap_mode = 0;
        float contrast = 1.0f;
        float saturation = 1.0f;
        // Projection constants for the depth-aware AO/fog upsample
        // (viewZ = depth_b / (depth + depth_a), glm RH_ZO): depth_a =
        // proj[2][2], depth_b = proj[3][2]. Zeros degrade to plain bilinear.
        float depth_a = 0.0f;
        float depth_b = 0.0f;
    };

    /**
     * Composite the HDR scene (+AO/bloom/fog) onto `target` with tonemapping.
     * Callers pass the swapchain texture directly (AA off) or ldr_target()
     * (then run render_fxaa_pass to resolve onto the swapchain).
     */
    void render_composite_pass(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, SDL_GPUTexture* target,
                               SDL_GPUTexture* bloom_texture, SDL_GPUTexture* fog_texture,
                               const CompositeParams& params);

    /**
     * Fullscreen FXAA pass: samples ldr_target() (the gamma-encoded composite
     * output) and writes the anti-aliased image to the swapchain. Must run
     * after render_composite_pass in the same frame (the composite pass caches
     * the pipeline registry used here).
     */
    void render_fxaa_pass(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchain_target);

    bool is_ready() const;

    // Offscreen targets for other systems (bloom, shadows, etc.)
    gpu::GPUTexture* offscreen_color() { return offscreen_color_.get(); }
    gpu::GPUTexture* offscreen_depth() { return offscreen_depth_.get(); }

    /**
     * Lazily (re)created swapchain-format, swapchain-sized LDR color target.
     * Used as the composite destination when FXAA is enabled.
     */
    SDL_GPUTexture* ldr_target();

private:
    void create_textures(int width, int height);
    void create_samplers();
    void clear_ao_textures();

    gpu::GPUDevice* device_ = nullptr;

    // Full-resolution offscreen targets
    std::unique_ptr<gpu::GPUTexture> offscreen_color_;
    std::unique_ptr<gpu::GPUTexture> offscreen_depth_;

    // Half-resolution AO textures (ping-pong for blur)
    std::unique_ptr<gpu::GPUTexture> ao_texture_;
    std::unique_ptr<gpu::GPUTexture> ao_blurred_;

    // Swapchain-format LDR target for the composite->FXAA chain (lazy).
    std::unique_ptr<gpu::GPUTexture> ldr_target_;
    SDL_GPUTextureFormat ldr_format_ = SDL_GPU_TEXTUREFORMAT_INVALID;

    SDL_GPUSampler* nearest_clamp_sampler_ = nullptr;
    SDL_GPUSampler* linear_clamp_sampler_ = nullptr;

    SDL_GPURenderPass* current_pass_ = nullptr;

    // Linear HDR clear for uncovered pixels; defaults to the EnvironmentState
    // fog_color_horizon default. scene_renderer plumbs the live value.
    glm::vec3 clear_color_{0.45f, 0.56f, 0.74f};

    // Cached from the AO/composite passes: projection constants for the
    // depth-linearizing bilateral blur, and the registry for render_fxaa_pass.
    float blur_depth_a_ = 0.0f;
    float blur_depth_b_ = 0.0f;
    gpu::PipelineRegistry* cached_pipelines_ = nullptr;

    int width_ = 0;
    int height_ = 0;
    int ao_width_ = 0;
    int ao_height_ = 0;
};

} // namespace mmo::engine::render
