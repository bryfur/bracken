#include "gtao.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include <SDL3/SDL_gpu.h>

namespace bracken::render {

namespace {
constexpr const char* LOG_TAG = "gtao";

// Golden-ratio low-discrepancy sequence for the per-frame temporal jitter:
// slice rotation and step offset never repeat and cover [0,1) uniformly,
// which the temporal accumulation integrates over ~1/(1-feedbackMax) frames.
constexpr float GOLDEN_RATIO_CONJUGATE = 0.61803398874989485f;

float temporal_sequence(uint64_t frame_index, float scale) {
    const float x = static_cast<float>(frame_index % 4096u) * scale;
    return x - std::floor(x);
}

} // namespace

GTAO::GTAO() = default;

GTAO::~GTAO() {
    shutdown();
}

bool GTAO::init(gpu::GPUDevice& device, int width, int height, bool half_res) {
    device_ = &device;
    width_ = width;
    height_ = height;
    half_res_ = half_res;

    create_textures();

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

    if (!is_ready()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GTAO: Failed to initialize");
        return false;
    }

    clear_full_res_texture();

    BRACKEN_LOG_INFO(LOG_TAG, "GTAO: Initialized {}x{} (working res {}x{})", width, height, ao_width_, ao_height_);
    return true;
}

void GTAO::shutdown() {
    ao_raw_.reset();
    blur_tmp_.reset();
    ao_blurred_.reset();
    history_[0].reset();
    history_[1].reset();
    ao_full_.reset();
    history_valid_ = false;

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

void GTAO::resize(int width, int height) {
    if ((width == width_ && height == height_) || !device_) {
        return;
    }
    width_ = width;
    height_ = height;
    create_textures();
    clear_full_res_texture();
    BRACKEN_LOG_INFO(LOG_TAG, "GTAO: Resized to {}x{} (working res {}x{}, history invalidated)", width, height,
                     ao_width_, ao_height_);
}

void GTAO::set_half_res(bool half_res) {
    if (half_res == half_res_ || !device_) {
        half_res_ = half_res;
        return;
    }
    half_res_ = half_res;
    create_textures();
    clear_full_res_texture();
    BRACKEN_LOG_INFO(LOG_TAG, "GTAO: Working res now {}x{} (history invalidated)", ao_width_, ao_height_);
}

void GTAO::create_textures() {
    ao_width_ = half_res_ ? std::max(1, width_ / 2) : std::max(1, width_);
    ao_height_ = half_res_ ? std::max(1, height_ / 2) : std::max(1, height_);

    constexpr SDL_GPUTextureUsageFlags USAGE = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;

    ao_raw_ = gpu::GPUTexture::create_2d(*device_, ao_width_, ao_height_, SDL_GPU_TEXTUREFORMAT_R8_UNORM, USAGE);
    blur_tmp_ = gpu::GPUTexture::create_2d(*device_, ao_width_, ao_height_, SDL_GPU_TEXTUREFORMAT_R8_UNORM, USAGE);
    ao_blurred_ = gpu::GPUTexture::create_2d(*device_, ao_width_, ao_height_, SDL_GPU_TEXTUREFORMAT_R8_UNORM, USAGE);
    for (auto& slot : history_) {
        slot = gpu::GPUTexture::create_2d(*device_, ao_width_, ao_height_, SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT, USAGE);
    }
    ao_full_ = gpu::GPUTexture::create_2d(*device_, width_, height_, SDL_GPU_TEXTUREFORMAT_R8_UNORM, USAGE);

    history_valid_ = false;
}

void GTAO::clear_full_res_texture() {
    if (!device_ || !ao_full_) {
        return;
    }

    // One-off command buffer: init/resize/quality changes happen outside the
    // frame loop's recording. White = no occlusion, so the forward pass can
    // bind the texture before the first render() (e.g. AO mode Off).
    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_->handle());
    if (!cmd) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GTAO: Failed to acquire command buffer for AO clear: {}", SDL_GetError());
        return;
    }

    SDL_GPUColorTargetInfo color_target = {};
    color_target.texture = ao_full_->handle();
    color_target.load_op = SDL_GPU_LOADOP_CLEAR;
    color_target.store_op = SDL_GPU_STOREOP_STORE;
    color_target.clear_color = {1.0f, 1.0f, 1.0f, 1.0f};

    if (SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr)) {
        SDL_EndGPURenderPass(pass);
    }

    SDL_SubmitGPUCommandBuffer(cmd);
}

bool GTAO::is_ready() const {
    return ao_raw_ && blur_tmp_ && ao_blurred_ && history_[0] && history_[1] && ao_full_ &&
           (nearest_clamp_sampler_ != nullptr) && (linear_clamp_sampler_ != nullptr);
}

void GTAO::render(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, gpu::GPUTexture& depth,
                  gpu::GPUTexture& velocity, const FrameParams& params) {
    BRACKEN_PROFILE_ZONE_COLOR("GTAO::render", 0x16A085);
    if (!is_ready() || !cmd) {
        return;
    }
    auto* gtao_pipeline = pipelines.get_gtao_pipeline();
    auto* temporal_pipeline = pipelines.get_gtao_temporal_pipeline();
    auto* blur_pipeline = pipelines.get_blur_ao_pipeline();
    auto* upsample_pipeline = pipelines.get_gtao_upsample_pipeline();
    if (!gtao_pipeline || !temporal_pipeline || !blur_pipeline || !upsample_pipeline) {
        return;
    }

    const glm::vec2 working_size(static_cast<float>(ao_width_), static_cast<float>(ao_height_));
    const glm::vec2 inv_working(1.0f / static_cast<float>(ao_width_), 1.0f / static_cast<float>(ao_height_));
    // Depth linearization constants (viewZ = |B / (depth + A)|, glm RH_ZO).
    const float depth_a = params.projection[2][2];
    const float depth_b = params.projection[3][2];

    // ---- 1. GTAO: depth -> raw AO ----
    {
        gpu::GTAOUniforms uniforms = {};
        uniforms.projection = params.projection;
        uniforms.invProjection = params.inv_projection;
        uniforms.screenSize = working_size;
        uniforms.invScreenSize = inv_working;
        uniforms.radiusScale = params.radius_scale;
        uniforms.temporalRotation = temporal_sequence(params.frame_index, GOLDEN_RATIO_CONJUGATE);
        uniforms.temporalOffset = temporal_sequence(params.frame_index, 1.0f - GOLDEN_RATIO_CONJUGATE);
        // radius / falloffRange / thinOccluderCompensation / maxDistance keep
        // their NSDMI tuning defaults.

        SDL_GPUColorTargetInfo color_target = {};
        color_target.texture = ao_raw_->handle();
        color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
        color_target.store_op = SDL_GPU_STOREOP_STORE;

        SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
        if (!pass) {
            return;
        }

        gtao_pipeline->bind(pass);
        SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

        SDL_GPUTextureSamplerBinding depth_binding = {};
        depth_binding.texture = depth.handle();
        depth_binding.sampler = nearest_clamp_sampler_;
        SDL_BindGPUFragmentSamplers(pass, 0, &depth_binding, 1);

        SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
        SDL_EndGPURenderPass(pass);
    }

    gpu::GPUTexture* history_write = history_[history_write_].get();
    gpu::GPUTexture* history_read = history_[history_write_ ^ 1].get();

    // ---- 2. Temporal accumulation: raw AO + history + velocity + depth -> new history ----
    {
        gpu::GtaoTemporalUniforms uniforms = {};
        uniforms.invScreenSize = inv_working;
        uniforms.depthA = depth_a;
        uniforms.depthB = depth_b;
        uniforms.historyValid = history_valid_ ? 1.0f : 0.0f;
        // feedbackMax / depthRejectScale keep their NSDMI tuning defaults.

        SDL_GPUColorTargetInfo color_target = {};
        color_target.texture = history_write->handle();
        color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
        color_target.store_op = SDL_GPU_STOREOP_STORE;

        SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
        if (!pass) {
            return;
        }

        temporal_pipeline->bind(pass);
        SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

        SDL_GPUTextureSamplerBinding bindings[4] = {};
        bindings[0].texture = ao_raw_->handle();
        bindings[0].sampler = nearest_clamp_sampler_;
        // History reprojects to arbitrary sub-texel positions: linear filter.
        // Contents are ignored while historyValid == 0.
        bindings[1].texture = history_read->handle();
        bindings[1].sampler = linear_clamp_sampler_;
        bindings[2].texture = velocity.handle();
        bindings[2].sampler = nearest_clamp_sampler_;
        bindings[3].texture = depth.handle();
        bindings[3].sampler = nearest_clamp_sampler_;
        SDL_BindGPUFragmentSamplers(pass, 0, bindings, 4);

        SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
        SDL_EndGPURenderPass(pass);
    }

    // ---- 3+4. Depth-aware separable blur of the accumulated AO ----
    // Reads the RG16F history's R channel; the history itself stays
    // unfiltered so the blur kernel never compounds across frames.
    const auto blur_pass = [&](gpu::GPUTexture* src, gpu::GPUTexture* dst, glm::vec2 direction) {
        gpu::BlurUniforms uniforms = {};
        uniforms.direction = direction;
        uniforms.invScreenSize = inv_working;
        uniforms.depthA = depth_a;
        uniforms.depthB = depth_b;

        SDL_GPUColorTargetInfo color_target = {};
        color_target.texture = dst->handle();
        color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
        color_target.store_op = SDL_GPU_STOREOP_STORE;

        SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
        if (!pass) {
            return;
        }

        blur_pipeline->bind(pass);
        SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

        SDL_GPUTextureSamplerBinding bindings[2] = {};
        bindings[0].texture = src->handle();
        bindings[0].sampler = nearest_clamp_sampler_;
        bindings[1].texture = depth.handle();
        bindings[1].sampler = nearest_clamp_sampler_;
        SDL_BindGPUFragmentSamplers(pass, 0, bindings, 2);

        SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
        SDL_EndGPURenderPass(pass);
    };
    blur_pass(history_write, blur_tmp_.get(), glm::vec2(1.0f, 0.0f));
    blur_pass(blur_tmp_.get(), ao_blurred_.get(), glm::vec2(0.0f, 1.0f));

    // ---- 5. Joint-bilateral upsample/finalize to the full-res AO texture ----
    // At full working res (GTAO_Full) the filter is an exact passthrough;
    // ao_strength applies here either way.
    {
        gpu::GtaoUpsampleUniforms uniforms = {};
        uniforms.lowResTexelSize = inv_working;
        uniforms.depthA = depth_a;
        uniforms.depthB = depth_b;
        uniforms.strength = params.strength;

        SDL_GPUColorTargetInfo color_target = {};
        color_target.texture = ao_full_->handle();
        color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
        color_target.store_op = SDL_GPU_STOREOP_STORE;

        SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
        if (!pass) {
            return;
        }

        upsample_pipeline->bind(pass);
        SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

        SDL_GPUTextureSamplerBinding bindings[2] = {};
        bindings[0].texture = ao_blurred_->handle();
        bindings[0].sampler = nearest_clamp_sampler_;
        bindings[1].texture = depth.handle();
        bindings[1].sampler = nearest_clamp_sampler_;
        SDL_BindGPUFragmentSamplers(pass, 0, bindings, 2);

        SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
        SDL_EndGPURenderPass(pass);
    }

    history_write_ ^= 1;
    history_valid_ = true;
}

} // namespace bracken::render
