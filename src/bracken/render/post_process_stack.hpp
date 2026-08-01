#pragma once

#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include <glm/glm.hpp>
#include <memory>

namespace bracken::render {

namespace gpu = bracken::gpu;

/**
 * The frame's HDR post-processing stack (formerly `AmbientOcclusion`, which
 * despite the name owned this whole chain — the AO passes themselves now
 * live in render::GTAO). In the normal frame path regardless of any AO
 * setting. It owns:
 *
 * - the full-resolution linear-HDR offscreen scene color + depth targets
 *   (the scene's render destination; bloom, volumetric fog, TAA and GTAO
 *   fetch their inputs from offscreen_color()/offscreen_depth()),
 * - the composite pass — the frame's tonemapping/exposure/contrast/
 *   saturation plus the depth-aware (joint bilateral) volumetric-fog
 *   upsample (screen-space AO is applied in the forward pass, not here),
 * - the FXAA resolve with its lazily created LDR intermediate target,
 * - the shared nearest/linear clamp samplers used by these passes.
 */
class PostProcessStack {
public:
    PostProcessStack();
    ~PostProcessStack();

    PostProcessStack(const PostProcessStack&) = delete;
    PostProcessStack& operator=(const PostProcessStack&) = delete;
    PostProcessStack(PostProcessStack&&) = delete;
    PostProcessStack& operator=(PostProcessStack&&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device, int width, int height);
    void shutdown();
    void resize(int width, int height);

    /**
     * Set the linear clear color for the offscreen scene target. Should track
     * EnvironmentState::fog_color_horizon so uncovered pixels blend into the
     * sky/fog.
     */
    void set_clear_color(glm::vec3 linear) { clear_color_ = linear; }

    // Offscreen pass (replaces begin_main_pass when the HDR path is on).
    // load_depth = true keeps the depth contents laid down by the depth
    // prepass (LOADOP_LOAD) instead of clearing to the reversed-Z far plane.
    SDL_GPURenderPass* begin_offscreen_pass(SDL_GPUCommandBuffer* cmd, bool load_depth = false);
    void end_offscreen_pass();

    struct CompositeParams {
        // Defaults track GraphicsSettings (the source of truth — the normal
        // frame path overwrites every field from it; see scene_renderer).
        // A test pins bloom_strength == GraphicsSettings{}.bloom_strength.
        float bloom_strength = 0.25f;
        float exposure = 1.0f;
        int tonemap_mode = 0;
        float contrast = 1.0f;
        float saturation = 1.0f;
        // Projection constants for the depth-aware fog upsample
        // (viewZ = depth_b / (depth + depth_a), glm RH_ZO): depth_a =
        // proj[2][2], depth_b = proj[3][2]. Zeros degrade to plain bilinear.
        float depth_a = 0.0f;
        float depth_b = 0.0f;
        // Auto exposure: when enabled AND a state buffer is supplied
        // (render::AutoExposure::state_buffer()), the shader multiplies
        // exp2(adapted EV) into `exposure` (which stays the manual bias).
        // Null buffer -> a zeroed dummy keeps the SSBO slot valid, flag off.
        bool auto_exposure = false;
        SDL_GPUBuffer* exposure_buffer = nullptr;
        // Color grading; all-neutral defaults are PIXEL-IDENTICAL to the
        // ungraded path (the pass gates each block on non-neutral values —
        // pinned by post_process_stack tests). White balance applies
        // pre-tonemap in linear light (render::white_balance_coefficients);
        // the CDL triple post-tonemap, before contrast/saturation.
        glm::vec3 white_balance{1.0f, 1.0f, 1.0f};
        glm::vec3 cdl_slope{1.0f, 1.0f, 1.0f};
        glm::vec3 cdl_offset{0.0f, 0.0f, 0.0f};
        glm::vec3 cdl_power{1.0f, 1.0f, 1.0f};
    };

    /**
     * Composite the HDR scene (+bloom/fog) onto `target` with tonemapping.
     * Callers pass the swapchain texture directly (AA off / TAA) or
     * ldr_target() (then run render_fxaa_pass to resolve onto the swapchain).
     */
    void render_composite_pass(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, SDL_GPUTexture* target,
                               SDL_GPUTexture* bloom_texture, SDL_GPUTexture* fog_texture,
                               SDL_GPUTexture* rt_reflection_texture,
                               const CompositeParams& params);

    /**
     * Fullscreen FXAA pass: samples ldr_target() (the gamma-encoded composite
     * output) and writes the anti-aliased image to the swapchain. Must run
     * after render_composite_pass in the same frame (the composite pass caches
     * the pipeline registry used here).
     */
    void render_fxaa_pass(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchain_target);

    bool is_ready() const;

    // Offscreen targets for other systems (bloom, fog, TAA, GTAO, prepass).
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

    gpu::GPUDevice* device_ = nullptr;

    // Full-resolution offscreen targets
    std::unique_ptr<gpu::GPUTexture> offscreen_color_;
    std::unique_ptr<gpu::GPUTexture> offscreen_depth_;

    // Swapchain-format LDR target for the composite->FXAA chain (lazy).
    std::unique_ptr<gpu::GPUTexture> ldr_target_;
    SDL_GPUTextureFormat ldr_format_ = SDL_GPU_TEXTUREFORMAT_INVALID;

    // Zeroed 16-byte SSBO filler for the composite shader's exposure-state
    // slot when no AutoExposure buffer is supplied (the slot is always
    // declared; it is only read when the autoExposureEnabled flag is set).
    std::unique_ptr<gpu::GPUBuffer> dummy_exposure_buffer_;

    SDL_GPUSampler* nearest_clamp_sampler_ = nullptr;
    SDL_GPUSampler* linear_clamp_sampler_ = nullptr;

    SDL_GPURenderPass* current_pass_ = nullptr;

    // Linear HDR clear for uncovered pixels; defaults to the EnvironmentState
    // fog_color_horizon default. scene_renderer plumbs the live value.
    glm::vec3 clear_color_{0.45f, 0.56f, 0.74f};

    // Cached from the composite pass: the registry for render_fxaa_pass.
    gpu::PipelineRegistry* cached_pipelines_ = nullptr;

    int width_ = 0;
    int height_ = 0;
};

} // namespace bracken::render
