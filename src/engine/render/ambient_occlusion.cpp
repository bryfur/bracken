#include "ambient_occlusion.hpp"
#include "engine/core/profiler.hpp"
#include "engine/gpu/gpu_pipeline.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <glm/gtc/matrix_inverse.hpp>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_log.h>

namespace mmo::engine::render {

namespace {

constexpr float TWO_PI = 6.28318530717958647692f;

// Low-discrepancy sequence for deterministic, well-distributed kernel samples.
float halton(int index, int base) {
    float f = 1.0f;
    float r = 0.0f;
    while (index > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(index % base);
        index /= base;
    }
    return r;
}

// SSAO hemisphere kernel, computed once: xyz = direction (z >= 0) scaled by a
// per-sample length in [0.1, 1] clustered toward the center. Replaces the old
// per-sample in-shader 3x hash + acos/sin/cos.
const std::array<glm::vec4, gpu::GTAOUniforms::KERNEL_SIZE>& ssao_hemisphere_kernel() {
    static const auto kernel = [] {
        std::array<glm::vec4, gpu::GTAOUniforms::KERNEL_SIZE> k{};
        for (int i = 0; i < gpu::GTAOUniforms::KERNEL_SIZE; ++i) {
            const float u1 = halton(i + 1, 2);
            const float u2 = halton(i + 1, 3);
            const float u3 = halton(i + 1, 5);
            const float theta = u1 * TWO_PI;
            const float phi = std::acos(1.0f - 2.0f * u2) * 0.5f; // hemisphere (z >= 0)
            const glm::vec3 dir(std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta), std::cos(phi));
            const float len = 0.1f + 0.9f * u3 * u3; // cluster samples near the shading point
            k[i] = glm::vec4(dir * len, 0.0f);
        }
        return k;
    }();
    return kernel;
}

} // namespace

namespace detail {

// "GTAO"-pass base directions: evenly spaced unit-circle directions in xy. The
// shader rotates them per-pixel via IGN (one sincos per pixel total). Built
// fresh on every call (24 sincos, trivial) so a changed num_directions takes
// effect immediately — this used to be a function-local static that latched
// the first call's count forever.
std::array<glm::vec4, gpu::GTAOUniforms::KERNEL_SIZE> gtao_direction_kernel(int num_directions) {
    std::array<glm::vec4, gpu::GTAOUniforms::KERNEL_SIZE> k{};
    const int dirs = std::max(num_directions, 1);
    for (int i = 0; i < gpu::GTAOUniforms::KERNEL_SIZE; ++i) {
        const float angle = (static_cast<float>(i % dirs) / static_cast<float>(dirs)) * TWO_PI;
        k[i] = glm::vec4(std::cos(angle), std::sin(angle), 0.0f, 0.0f);
    }
    return k;
}

} // namespace detail

AmbientOcclusion::AmbientOcclusion() = default;

AmbientOcclusion::~AmbientOcclusion() {
    shutdown();
}

bool AmbientOcclusion::init(gpu::GPUDevice& device, int width, int height) {
    device_ = &device;
    width_ = width;
    height_ = height;
    // Run AO at half resolution: 4x less fragment work for AO compute + blur.
    // The composite pass upsamples depth-aware (joint bilateral) to avoid
    // silhouette halos from mixing foreground/background AO.
    ao_width_ = std::max(1, width / 2);
    ao_height_ = std::max(1, height / 2);

    create_textures(width, height);
    create_samplers();

    if (!is_ready()) {
        SDL_Log("AmbientOcclusion: Failed to initialize");
        return false;
    }

    // The composite pass always samples the AO texture, even when ao_mode == 0
    // skips the AO passes entirely — initialize to white (no occlusion).
    clear_ao_textures();

    SDL_Log("AmbientOcclusion: Initialized %dx%d (AO half-res: %dx%d)", width, height, ao_width_, ao_height_);
    return true;
}

void AmbientOcclusion::shutdown() {
    offscreen_color_.reset();
    offscreen_depth_.reset();
    ao_texture_.reset();
    ao_blurred_.reset();
    ldr_target_.reset();
    ldr_format_ = SDL_GPU_TEXTUREFORMAT_INVALID;
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

void AmbientOcclusion::resize(int width, int height) {
    if (width == width_ && height == height_) {
        return;
    }
    if (!device_) {
        return;
    }

    width_ = width;
    height_ = height;
    ao_width_ = std::max(1, width / 2);
    ao_height_ = std::max(1, height / 2);

    create_textures(width, height);
    clear_ao_textures();
    SDL_Log("AmbientOcclusion: Resized to %dx%d (AO half-res: %dx%d)", width, height, ao_width_, ao_height_);
}

void AmbientOcclusion::create_textures(int width, int height) {
    // HDR (linear) offscreen scene target. Lighting/bloom/fog operate in linear HDR,
    // tonemapping happens only in the composite pass to the LDR swapchain.
    offscreen_color_ = gpu::GPUTexture::create_2d(*device_, width, height, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                                                  SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);

    // Full-resolution depth
    offscreen_depth_ = gpu::GPUTexture::create_depth(*device_, width, height);

    // Half-resolution AO textures
    ao_texture_ = gpu::GPUTexture::create_2d(*device_, ao_width_, ao_height_, SDL_GPU_TEXTUREFORMAT_R8_UNORM,
                                             SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);

    ao_blurred_ = gpu::GPUTexture::create_2d(*device_, ao_width_, ao_height_, SDL_GPU_TEXTUREFORMAT_R8_UNORM,
                                             SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);

    // LDR target is lazily recreated at the new size on next ldr_target() call.
    ldr_target_.reset();
}

void AmbientOcclusion::clear_ao_textures() {
    if (!device_ || !ao_texture_ || !ao_blurred_) {
        return;
    }

    // One-off command buffer: init/resize happen outside the frame loop.
    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_->handle());
    if (!cmd) {
        SDL_Log("AmbientOcclusion: Failed to acquire command buffer for AO clear: %s", SDL_GetError());
        return;
    }

    for (gpu::GPUTexture* tex : {ao_texture_.get(), ao_blurred_.get()}) {
        SDL_GPUColorTargetInfo color_target = {};
        color_target.texture = tex->handle();
        color_target.load_op = SDL_GPU_LOADOP_CLEAR;
        color_target.store_op = SDL_GPU_STOREOP_STORE;
        color_target.clear_color = {1.0f, 1.0f, 1.0f, 1.0f}; // 1.0 = no occlusion

        if (SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr)) {
            SDL_EndGPURenderPass(pass);
        }
    }

    SDL_SubmitGPUCommandBuffer(cmd);
}

SDL_GPUTexture* AmbientOcclusion::ldr_target() {
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

void AmbientOcclusion::create_samplers() {
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

bool AmbientOcclusion::is_ready() const {
    return offscreen_color_ && offscreen_depth_ && ao_texture_ && ao_blurred_ && (nearest_clamp_sampler_ != nullptr) &&
           (linear_clamp_sampler_ != nullptr);
}

// =============================================================================
// Render Passes
// =============================================================================

SDL_GPURenderPass* AmbientOcclusion::begin_offscreen_pass(SDL_GPUCommandBuffer* cmd) {
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
    depth_target.load_op = SDL_GPU_LOADOP_CLEAR;
    depth_target.store_op = SDL_GPU_STOREOP_STORE;
    depth_target.clear_depth = 0.0f; // reversed-Z: 0.0 is the far plane
    depth_target.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
    depth_target.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;

    current_pass_ = SDL_BeginGPURenderPass(cmd, &color_target, 1, &depth_target);
    return current_pass_;
}

void AmbientOcclusion::end_offscreen_pass() {
    if (current_pass_) {
        SDL_EndGPURenderPass(current_pass_);
        current_pass_ = nullptr;
    }
}

void AmbientOcclusion::render_ssao_pass(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines,
                                        const glm::mat4& projection, const glm::mat4& inv_projection,
                                        float radius_scale) {
    ENGINE_PROFILE_ZONE_COLOR("AmbientOcclusion::render_ssao_pass", 0x16A085);
    auto* pipeline = pipelines.get_ssao_pipeline();
    if (!pipeline) {
        return;
    }
    cached_pipelines_ = &pipelines;
    blur_depth_a_ = projection[2][2];
    blur_depth_b_ = projection[3][2];

    gpu::GTAOUniforms uniforms = {};
    uniforms.projection = projection;
    uniforms.invProjection = inv_projection;
    uniforms.screenSize = glm::vec2(static_cast<float>(ao_width_), static_cast<float>(ao_height_));
    uniforms.invScreenSize = glm::vec2(1.0f / ao_width_, 1.0f / ao_height_);
    uniforms.radiusScale = radius_scale;
    const auto& kernel = ssao_hemisphere_kernel();
    std::copy(kernel.begin(), kernel.end(), uniforms.sampleKernel);

    SDL_GPUColorTargetInfo color_target = {};
    color_target.texture = ao_texture_->handle();
    color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
    color_target.store_op = SDL_GPU_STOREOP_STORE;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
    if (!pass) {
        return;
    }

    pipeline->bind(pass);
    SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

    SDL_GPUTextureSamplerBinding depth_binding = {};
    depth_binding.texture = offscreen_depth_->handle();
    depth_binding.sampler = nearest_clamp_sampler_;
    SDL_BindGPUFragmentSamplers(pass, 0, &depth_binding, 1);

    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
}

void AmbientOcclusion::render_gtao_pass(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines,
                                        const glm::mat4& projection, const glm::mat4& inv_projection,
                                        float radius_scale) {
    ENGINE_PROFILE_ZONE_COLOR("AmbientOcclusion::render_gtao_pass", 0x16A085);
    auto* pipeline = pipelines.get_gtao_pipeline();
    if (!pipeline) {
        return;
    }
    cached_pipelines_ = &pipelines;
    blur_depth_a_ = projection[2][2];
    blur_depth_b_ = projection[3][2];

    gpu::GTAOUniforms uniforms = {};
    uniforms.projection = projection;
    uniforms.invProjection = inv_projection;
    uniforms.screenSize = glm::vec2(static_cast<float>(ao_width_), static_cast<float>(ao_height_));
    uniforms.invScreenSize = glm::vec2(1.0f / ao_width_, 1.0f / ao_height_);
    uniforms.radiusScale = radius_scale;
    const auto kernel = detail::gtao_direction_kernel(uniforms.numDirections);
    std::copy(kernel.begin(), kernel.end(), uniforms.sampleKernel);

    SDL_GPUColorTargetInfo color_target = {};
    color_target.texture = ao_texture_->handle();
    color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
    color_target.store_op = SDL_GPU_STOREOP_STORE;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
    if (!pass) {
        return;
    }

    pipeline->bind(pass);
    SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

    SDL_GPUTextureSamplerBinding depth_binding = {};
    depth_binding.texture = offscreen_depth_->handle();
    depth_binding.sampler = nearest_clamp_sampler_;
    SDL_BindGPUFragmentSamplers(pass, 0, &depth_binding, 1);

    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
}

void AmbientOcclusion::render_blur_pass(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines) {
    ENGINE_PROFILE_ZONE_COLOR("AmbientOcclusion::render_blur_pass", 0x16A085);
    auto* pipeline = pipelines.get_blur_ao_pipeline();
    if (!pipeline) {
        return;
    }

    glm::vec2 inv_screen(1.0f / ao_width_, 1.0f / ao_height_);

    // Pass 1: Horizontal blur (ao_texture_ → ao_blurred_)
    {
        gpu::BlurUniforms uniforms = {};
        uniforms.direction = glm::vec2(1.0f, 0.0f);
        uniforms.invScreenSize = inv_screen;
        uniforms.depthA = blur_depth_a_;
        uniforms.depthB = blur_depth_b_;

        SDL_GPUColorTargetInfo color_target = {};
        color_target.texture = ao_blurred_->handle();
        color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
        color_target.store_op = SDL_GPU_STOREOP_STORE;

        SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
        if (!pass) {
            return;
        }

        pipeline->bind(pass);
        SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

        SDL_GPUTextureSamplerBinding bindings[2] = {};
        bindings[0].texture = ao_texture_->handle();
        bindings[0].sampler = nearest_clamp_sampler_;
        bindings[1].texture = offscreen_depth_->handle();
        bindings[1].sampler = nearest_clamp_sampler_;
        SDL_BindGPUFragmentSamplers(pass, 0, bindings, 2);

        SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
        SDL_EndGPURenderPass(pass);
    }

    // Pass 2: Vertical blur (ao_blurred_ → ao_texture_)
    {
        gpu::BlurUniforms uniforms = {};
        uniforms.direction = glm::vec2(0.0f, 1.0f);
        uniforms.invScreenSize = inv_screen;
        uniforms.depthA = blur_depth_a_;
        uniforms.depthB = blur_depth_b_;

        SDL_GPUColorTargetInfo color_target = {};
        color_target.texture = ao_texture_->handle();
        color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
        color_target.store_op = SDL_GPU_STOREOP_STORE;

        SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
        if (!pass) {
            return;
        }

        pipeline->bind(pass);
        SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

        SDL_GPUTextureSamplerBinding bindings[2] = {};
        bindings[0].texture = ao_blurred_->handle();
        bindings[0].sampler = nearest_clamp_sampler_;
        bindings[1].texture = offscreen_depth_->handle();
        bindings[1].sampler = nearest_clamp_sampler_;
        SDL_BindGPUFragmentSamplers(pass, 0, bindings, 2);

        SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
        SDL_EndGPURenderPass(pass);
    }
}

void AmbientOcclusion::render_composite_pass(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines,
                                             SDL_GPUTexture* target, SDL_GPUTexture* bloom_texture,
                                             SDL_GPUTexture* fog_texture, const CompositeParams& params) {
    ENGINE_PROFILE_ZONE_COLOR("AmbientOcclusion::render_composite_pass", 0x16A085);
    auto* pipeline = pipelines.get_composite_pipeline();
    if (!pipeline || !target) {
        return;
    }
    cached_pipelines_ = &pipelines;

    gpu::CompositeUniforms uniforms = {};
    uniforms.aoStrength = params.ao_strength;
    uniforms.bloomStrength = bloom_texture ? params.bloom_strength : 0.0f;
    uniforms.volumetricFogEnabled = fog_texture ? 1.0f : 0.0f;
    uniforms.exposure = params.exposure;
    uniforms.tonemapMode = params.tonemap_mode;
    uniforms.contrast = params.contrast;
    uniforms.saturation = params.saturation;
    // AO and volumetric fog share the same half-res grid (both width/2 x
    // height/2), so one texel size drives the joint bilateral upsample.
    uniforms.lowResTexelSize = glm::vec2(1.0f / ao_width_, 1.0f / ao_height_);
    uniforms.depthA = params.depth_a;
    uniforms.depthB = params.depth_b;

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

    // Bind bloom texture (or AO texture as fallback for slot 2 - shader reads 0.0 strength)
    SDL_GPUTexture* bloom_tex = bloom_texture ? bloom_texture : ao_texture_->handle();
    // Bind fog texture (or AO texture as fallback for slot 3 - shader checks enabled flag)
    SDL_GPUTexture* fog_tex = fog_texture ? fog_texture : ao_texture_->handle();

    SDL_GPUTextureSamplerBinding bindings[5] = {};
    bindings[0].texture = offscreen_color_->handle();
    bindings[0].sampler = linear_clamp_sampler_;
    bindings[1].texture = ao_texture_->handle(); // After blur, result is back in ao_texture_
    bindings[1].sampler = linear_clamp_sampler_;
    bindings[2].texture = bloom_tex;
    bindings[2].sampler = linear_clamp_sampler_;
    bindings[3].texture = fog_tex;
    bindings[3].sampler = linear_clamp_sampler_;
    // Full-res depth drives the joint bilateral upsample of AO + fog.
    bindings[4].texture = offscreen_depth_->handle();
    bindings[4].sampler = nearest_clamp_sampler_;
    SDL_BindGPUFragmentSamplers(pass, 0, bindings, 5);

    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
}

void AmbientOcclusion::render_fxaa_pass(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchain_target) {
    ENGINE_PROFILE_ZONE_COLOR("AmbientOcclusion::render_fxaa_pass", 0x16A085);
    // Pipeline registry is cached by render_composite_pass; FXAA always runs
    // right after composite-to-LDR within the same frame.
    if (!cached_pipelines_ || !swapchain_target || !ldr_target_) {
        return;
    }
    auto* pipeline = cached_pipelines_->get_fxaa_pipeline();
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

} // namespace mmo::engine::render
