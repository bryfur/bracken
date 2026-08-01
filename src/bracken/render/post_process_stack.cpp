#include "post_process_stack.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include <algorithm>
#include <SDL3/SDL_gpu.h>

namespace bracken::render {

namespace {
constexpr const char* LOG_TAG = "post";
} // namespace

PostProcessStack::PostProcessStack() = default;

PostProcessStack::~PostProcessStack() {
    shutdown();
}

bool PostProcessStack::init(gpu::GPUDevice& device, int width, int height) {
    device_ = &device;
    width_ = width;
    height_ = height;

    create_textures(width, height);
    create_samplers();

    // Zeroed filler for the composite shader's always-declared exposure SSBO
    // slot (read only when the auto-exposure flag is set).
    const float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    dummy_exposure_buffer_ = gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::Storage, zeros, sizeof(zeros));

    if (!is_ready()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PostProcessStack: Failed to initialize");
        return false;
    }

    BRACKEN_LOG_INFO(LOG_TAG, "PostProcessStack: Initialized {}x{}", width, height);
    return true;
}

void PostProcessStack::shutdown() {
    offscreen_color_.reset();
    offscreen_depth_.reset();
    ldr_target_.reset();
    ldr_format_ = SDL_GPU_TEXTUREFORMAT_INVALID;
    dummy_exposure_buffer_.reset();
    cached_pipelines_ = nullptr;

    if (device_) {
        if (nearest_clamp_sampler_) {
            device_->release_sampler(nearest_clamp_sampler_);
            nearest_clamp_sampler_ = nullptr;
        }
        if (linear_clamp_sampler_) {
            device_->release_sampler(linear_clamp_sampler_);
            linear_clamp_sampler_ = nullptr;
        }
    }

    device_ = nullptr;
}

void PostProcessStack::resize(int width, int height) {
    if (width == width_ && height == height_) {
        return;
    }
    if (!device_) {
        return;
    }

    width_ = width;
    height_ = height;

    create_textures(width, height);
    BRACKEN_LOG_INFO(LOG_TAG, "PostProcessStack: Resized to {}x{}", width, height);
}

void PostProcessStack::create_textures(int width, int height) {
    // HDR (linear) offscreen scene target. Lighting/bloom/fog operate in linear HDR,
    // tonemapping happens only in the composite pass to the LDR swapchain.
    offscreen_color_ = gpu::GPUTexture::create_2d(*device_, width, height, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                                                  SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);

    // Full-resolution depth
    offscreen_depth_ = gpu::GPUTexture::create_depth(*device_, width, height);

    // LDR target is lazily recreated at the new size on next ldr_target() call.
    ldr_target_.reset();
}

SDL_GPUTexture* PostProcessStack::ldr_target() {
    if (!device_ || width_ <= 0 || height_ <= 0) {
        return nullptr;
    }

    const SDL_GPUTextureFormat format = device_->swapchain_format();
    const bool stale = !ldr_target_ || ldr_target_->width() != width_ || ldr_target_->height() != height_ ||
                       ldr_format_ != format;
    if (stale) {
        ldr_target_ = gpu::GPUTexture::create_2d(*device_, width_, height_, format,
                                                 SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);
        ldr_format_ = format;
    }

    return ldr_target_ ? ldr_target_->handle() : nullptr;
}

void PostProcessStack::create_samplers() {
    SDL_GPUSamplerCreateInfo nearest_info = {};
    nearest_info.min_filter = SDL_GPU_FILTER_NEAREST;
    nearest_info.mag_filter = SDL_GPU_FILTER_NEAREST;
    nearest_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    nearest_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    nearest_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    nearest_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    nearest_clamp_sampler_ = device_->create_sampler(nearest_info);

    SDL_GPUSamplerCreateInfo linear_info = nearest_info;
    linear_info.min_filter = SDL_GPU_FILTER_LINEAR;
    linear_info.mag_filter = SDL_GPU_FILTER_LINEAR;
    linear_clamp_sampler_ = device_->create_sampler(linear_info);
}

bool PostProcessStack::is_ready() const {
    return offscreen_color_ && offscreen_depth_ && dummy_exposure_buffer_ && (nearest_clamp_sampler_ != nullptr) &&
           (linear_clamp_sampler_ != nullptr);
}

// =============================================================================
// Render Passes
// =============================================================================

SDL_GPURenderPass* PostProcessStack::begin_offscreen_pass(SDL_GPUCommandBuffer* cmd, bool load_depth) {
    if (!is_ready()) {
        return nullptr;
    }

    SDL_GPUColorTargetInfo color_target = {};
    color_target.texture = offscreen_color_->handle();
    color_target.load_op = SDL_GPU_LOADOP_CLEAR;
    color_target.store_op = SDL_GPU_STOREOP_STORE;
    // Linear HDR clear; scene_renderer plumbs EnvironmentState::fog_color_horizon
    // via set_clear_color() so uncovered pixels blend into the sky/fog.
    color_target.clear_color = {clear_color_.r, clear_color_.g, clear_color_.b, 1.0f};

    SDL_GPUDepthStencilTargetInfo depth_target = {};
    depth_target.texture = offscreen_depth_->handle();
    // After a depth prepass the buffer already holds this frame's scene depth
    // (the forward opaques depth-test EQUAL against it) — load, don't clear.
    depth_target.load_op = load_depth ? SDL_GPU_LOADOP_LOAD : SDL_GPU_LOADOP_CLEAR;
    depth_target.store_op = SDL_GPU_STOREOP_STORE;
    depth_target.clear_depth = 0.0f; // reversed-Z: 0.0 is the far plane
    depth_target.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
    depth_target.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;

    current_pass_ = SDL_BeginGPURenderPass(cmd, &color_target, 1, &depth_target);
    return current_pass_;
}

void PostProcessStack::end_offscreen_pass() {
    if (current_pass_) {
        SDL_EndGPURenderPass(current_pass_);
        current_pass_ = nullptr;
    }
}

void PostProcessStack::render_composite_pass(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines,
                                             SDL_GPUTexture* target, SDL_GPUTexture* bloom_texture,
                                             SDL_GPUTexture* fog_texture,
                                             SDL_GPUTexture* rt_reflection_texture,
                                             const CompositeParams& params) {
    BRACKEN_PROFILE_ZONE_COLOR("PostProcessStack::render_composite_pass", 0x16A085);
    auto* pipeline = pipelines.get_pipeline(gpu::PipelineType::Composite);
    if (!pipeline || !target) {
        return;
    }
    cached_pipelines_ = &pipelines;

    // Auto exposure needs the caller-supplied state buffer; without one the
    // dummy filler keeps the SSBO slot valid and the flag stays off.
    SDL_GPUBuffer* exposure_buffer = params.exposure_buffer ? params.exposure_buffer
                                                            : dummy_exposure_buffer_->handle();
    const bool auto_exposure_on = params.auto_exposure && params.exposure_buffer != nullptr;
    // Grading blocks are gated on non-neutral parameters so the all-neutral
    // default is pixel-identical to the ungraded path (matrix/pow round-trips
    // are not bit-exact even at neutral values).
    const bool wb_on = params.white_balance != glm::vec3(1.0f);
    const bool cdl_on = params.cdl_slope != glm::vec3(1.0f) || params.cdl_offset != glm::vec3(0.0f) ||
                        params.cdl_power != glm::vec3(1.0f);

    gpu::CompositeUniforms uniforms = {};
    uniforms.bloomStrength = bloom_texture ? params.bloom_strength : 0.0f;
    uniforms.volumetricFogEnabled = fog_texture ? 1.0f : 0.0f;
    uniforms.exposure = params.exposure;
    uniforms.tonemapMode = params.tonemap_mode;
    uniforms.contrast = params.contrast;
    uniforms.saturation = params.saturation;
    // Volumetric fog renders at half resolution (VolumetricFog::resize) —
    // one texel size drives its joint bilateral upsample here.
    uniforms.lowResTexelSize =
        glm::vec2(1.0f / static_cast<float>(std::max(1, width_ / 2)), 1.0f / static_cast<float>(std::max(1, height_ / 2)));
    uniforms.depthA = params.depth_a;
    uniforms.depthB = params.depth_b;
    uniforms.autoExposureEnabled = auto_exposure_on ? 1.0f : 0.0f;
    uniforms.whiteBalanceEnabled = wb_on ? 1.0f : 0.0f;
    uniforms.whiteBalance = params.white_balance;
    uniforms.cdlEnabled = cdl_on ? 1.0f : 0.0f;
    uniforms.cdlSlope = params.cdl_slope;
    uniforms.cdlOffset = params.cdl_offset;
    uniforms.cdlPower = params.cdl_power;

    SDL_GPUColorTargetInfo color_target = {};
    color_target.texture = target;
    color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
    color_target.store_op = SDL_GPU_STOREOP_STORE;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
    if (!pass) {
        return;
    }

    pipeline->bind(pass);
    SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

    // Absent bloom/fog fall back to the scene color texture so every sampler
    // slot stays valid — the shader reads bloom with strength 0 and skips fog
    // via the enabled flag.
    SDL_GPUTexture* bloom_tex = bloom_texture ? bloom_texture : offscreen_color_->handle();
    SDL_GPUTexture* fog_tex = fog_texture ? fog_texture : offscreen_color_->handle();

    SDL_GPUTextureSamplerBinding bindings[5] = {};
    bindings[0].texture = offscreen_color_->handle();
    bindings[0].sampler = linear_clamp_sampler_;
    bindings[1].texture = bloom_tex;
    bindings[1].sampler = linear_clamp_sampler_;
    bindings[2].texture = fog_tex;
    bindings[2].sampler = linear_clamp_sampler_;
    // Full-res depth drives the joint bilateral upsample of the fog.
    bindings[3].texture = offscreen_depth_->handle();
    bindings[3].sampler = nearest_clamp_sampler_;
    // Traced reflections, or the transparent dummy when ray tracing is off —
    // the composite adds rgb*a, so a zero alpha makes it a no-op.
    bindings[4].texture = rt_reflection_texture ? rt_reflection_texture
                                                : offscreen_color_->handle();
    bindings[4].sampler = linear_clamp_sampler_;
    SDL_BindGPUFragmentSamplers(pass, 0, bindings, 5);

    // Exposure state (or the zeroed dummy) as the pass's only fragment SSBO.
    SDL_BindGPUFragmentStorageBuffers(pass, 0, &exposure_buffer, 1);

    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
}

void PostProcessStack::render_fxaa_pass(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchain_target) {
    BRACKEN_PROFILE_ZONE_COLOR("PostProcessStack::render_fxaa_pass", 0x16A085);
    // Pipeline registry is cached by render_composite_pass; FXAA always runs
    // right after composite-to-LDR within the same frame.
    if (!cached_pipelines_ || !swapchain_target || !ldr_target_) {
        return;
    }
    auto* pipeline = cached_pipelines_->get_pipeline(gpu::PipelineType::Fxaa);
    if (!pipeline) {
        return;
    }

    gpu::FxaaUniforms uniforms = {};
    uniforms.invScreenSize = glm::vec2(1.0f / static_cast<float>(width_), 1.0f / static_cast<float>(height_));

    SDL_GPUColorTargetInfo color_target = {};
    color_target.texture = swapchain_target;
    color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
    color_target.store_op = SDL_GPU_STOREOP_STORE;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
    if (!pass) {
        return;
    }

    pipeline->bind(pass);
    SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

    SDL_GPUTextureSamplerBinding binding = {};
    binding.texture = ldr_target_->handle();
    binding.sampler = linear_clamp_sampler_;
    SDL_BindGPUFragmentSamplers(pass, 0, &binding, 1);

    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
}

} // namespace bracken::render
