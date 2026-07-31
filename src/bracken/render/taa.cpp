#include "taa.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include <SDL3/SDL_gpu.h>

namespace bracken::render {

namespace {
constexpr const char* LOG_TAG = "taa";
} // namespace

TAA::TAA() = default;

TAA::~TAA() {
    shutdown();
}

bool TAA::init(gpu::GPUDevice& device, int width, int height) {
    device_ = &device;
    width_ = width;
    height_ = height;

    create_textures(width, height);

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
        BRACKEN_LOG_ERROR(LOG_TAG, "TAA: Failed to initialize");
        return false;
    }

    BRACKEN_LOG_INFO(LOG_TAG, "TAA: Initialized {}x{} (RGBA16F double-buffered history)", width, height);
    return true;
}

void TAA::shutdown() {
    history_[0].reset();
    history_[1].reset();
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

void TAA::resize(int width, int height) {
    if ((width == width_ && height == height_) || !device_) {
        return;
    }
    width_ = width;
    height_ = height;
    create_textures(width, height);
    history_valid_ = false;
    BRACKEN_LOG_INFO(LOG_TAG, "TAA: Resized to {}x{} (history invalidated)", width, height);
}

void TAA::create_textures(int width, int height) {
    // Same linear-HDR format as the offscreen scene target: the history must
    // hold pre-tonemap values so bloom reads a stabilized HDR image.
    for (auto& slot : history_) {
        slot = gpu::GPUTexture::create_2d(*device_, width, height, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                                          SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);
    }
    history_valid_ = false;
}

bool TAA::is_ready() const {
    return history_[0] && history_[1] && (nearest_clamp_sampler_ != nullptr) && (linear_clamp_sampler_ != nullptr);
}

void TAA::render(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, gpu::GPUTexture& scene_color,
                 gpu::GPUTexture& velocity, gpu::GPUTexture& depth, const ResolveParams& params) {
    BRACKEN_PROFILE_ZONE_COLOR("TAA::render", 0x16A085);
    if (!is_ready()) {
        return;
    }
    auto* resolve_pipeline = pipelines.get_taa_resolve_pipeline();
    auto* sharpen_pipeline = pipelines.get_taa_sharpen_pipeline();
    if (!resolve_pipeline || !sharpen_pipeline) {
        return;
    }

    gpu::GPUTexture* history_write = history_[history_write_].get();
    gpu::GPUTexture* history_read = history_[history_write_ ^ 1].get();

    // ---- Resolve: scene + history + velocity + depth -> new history ----
    {
        gpu::TaaResolveUniforms uniforms = {};
        uniforms.currInvViewProjUnjittered = params.curr_inv_view_proj_unjittered;
        uniforms.prevViewProjUnjittered = params.prev_view_proj_unjittered;
        uniforms.screenSize = glm::vec2(static_cast<float>(width_), static_cast<float>(height_));
        uniforms.invScreenSize = glm::vec2(1.0f / static_cast<float>(width_), 1.0f / static_cast<float>(height_));
        uniforms.cameraPos = params.camera_pos;
        uniforms.historyValid = history_valid_ ? 1.0f : 0.0f;
        // feedbackMax / varianceGamma keep their NSDMI tuning defaults.

        SDL_GPUColorTargetInfo color_target = {};
        color_target.texture = history_write->handle();
        color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
        color_target.store_op = SDL_GPU_STOREOP_STORE;

        SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
        if (!pass) {
            return;
        }

        resolve_pipeline->bind(pass);
        SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

        SDL_GPUTextureSamplerBinding bindings[4] = {};
        bindings[0].texture = scene_color.handle();
        bindings[0].sampler = nearest_clamp_sampler_;
        // History uses the linear sampler: the Catmull-Rom kernel is built
        // from bilinear taps. Contents are ignored while historyValid == 0.
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

    // ---- Sharpen/copy: new history -> scene color (bloom/composite input) ----
    // Always runs (strength 0 degrades to a copy) so downstream passes keep
    // their usual input texture; the history itself stays unsharpened.
    {
        gpu::TaaSharpenUniforms uniforms = {};
        uniforms.invScreenSize = glm::vec2(1.0f / static_cast<float>(width_), 1.0f / static_cast<float>(height_));
        uniforms.sharpenStrength = sharpen_strength_;

        SDL_GPUColorTargetInfo color_target = {};
        color_target.texture = scene_color.handle();
        color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
        color_target.store_op = SDL_GPU_STOREOP_STORE;

        SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
        if (!pass) {
            return;
        }

        sharpen_pipeline->bind(pass);
        SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

        SDL_GPUTextureSamplerBinding binding = {};
        binding.texture = history_write->handle();
        binding.sampler = nearest_clamp_sampler_;
        SDL_BindGPUFragmentSamplers(pass, 0, &binding, 1);

        SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
        SDL_EndGPURenderPass(pass);
    }

    history_write_ ^= 1;
    history_valid_ = true;
}

} // namespace bracken::render
