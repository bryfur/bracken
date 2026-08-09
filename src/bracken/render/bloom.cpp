#include "bloom.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include <algorithm>
#include <cmath>
#include <SDL3/SDL_gpu.h>

namespace bracken::render {

namespace {
constexpr const char* LOG_TAG = "bloom";
} // namespace

Bloom::Bloom() = default;

Bloom::~Bloom() {
    shutdown();
}

bool Bloom::init(gpu::GPUDevice& device, int width, int height) {
    device_ = &device;
    width_ = width;
    height_ = height;

    create_textures(width, height);
    create_sampler();

    // Zeroed filler for the prefilter's always-declared exposure SSBO slot
    // (read only when the auto-exposure flag is set), mirroring the composite
    // pass's own dummy.
    const float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    dummy_exposure_buffer_ =
        gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::Storage, zeros, sizeof(zeros));

    if (!is_ready()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "Bloom: Failed to initialize");
        return false;
    }

    BRACKEN_LOG_INFO(LOG_TAG, "Bloom: Initialized {}x{} ({} mip levels)", width, height, mip_count_);
    return true;
}

void Bloom::shutdown() {
    for (int i = 0; i < MAX_MIPS; ++i) {
        mip_textures_[i].reset();
    }
    dummy_exposure_buffer_.reset();

    if (device_ && linear_sampler_) {
        device_->release_sampler(linear_sampler_);
        linear_sampler_ = nullptr;
    }

    device_ = nullptr;
}

void Bloom::resize(int width, int height) {
    if (width == width_ && height == height_) {
        return;
    }
    if (!device_) {
        return;
    }

    width_ = width;
    height_ = height;

    create_textures(width, height);
    BRACKEN_LOG_INFO(LOG_TAG, "Bloom: Resized to {}x{} ({} mip levels)", width, height, mip_count_);
}

int Bloom::compute_mip_count(int width, int height) {
    // Scale mip count with resolution: clamp(log2(min(w,h)) - 7, 4, 6).
    // 1080p/1440p -> 4 mips, 4K -> 4, 8K -> 5+. More mips = wider glow radius
    // in screen-space without extra cost at lower resolutions.
    const int min_dim = std::max(1, std::min(width, height));
    const int mips = static_cast<int>(std::log2(static_cast<float>(min_dim))) - 7;
    return std::clamp(mips, 4, static_cast<int>(MAX_MIPS));
}

void Bloom::create_textures(int width, int height) {
    mip_count_ = compute_mip_count(width, height);

    // Each mip is half the resolution of the previous
    int w = std::max(1, width / 2);
    int h = std::max(1, height / 2);

    for (int i = 0; i < MAX_MIPS; ++i) {
        if (i < mip_count_) {
            mip_textures_[i] =
                gpu::GPUTexture::create_2d(*device_, w, h, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                                           SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);
        } else {
            mip_textures_[i].reset();
        }

        w = std::max(1, w / 2);
        h = std::max(1, h / 2);
    }
}

void Bloom::create_sampler() {
    SDL_GPUSamplerCreateInfo info = {};
    info.min_filter = SDL_GPU_FILTER_LINEAR;
    info.mag_filter = SDL_GPU_FILTER_LINEAR;
    info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    linear_sampler_ = device_->create_sampler(info);
}

bool Bloom::is_ready() const {
    if (!linear_sampler_ || !dummy_exposure_buffer_) {
        return false;
    }
    for (int i = 0; i < mip_count_; ++i) {
        if (!mip_textures_[i]) {
            return false;
        }
    }
    return true;
}

SDL_GPUTexture* Bloom::bloom_texture() const {
    if (mip_textures_[0]) {
        return mip_textures_[0]->handle();
    }
    return nullptr;
}

void Bloom::render(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& registry, SDL_GPUTexture* scene_color,
                   float threshold, float exposure, SDL_GPUBuffer* exposure_buffer) {
    BRACKEN_PROFILE_ZONE_COLOR("Bloom::render", 0x16A085);
    if (!is_ready() || !cmd || !scene_color) {
        return;
    }

    auto* downsample_pipeline = registry.get_pipeline(gpu::PipelineType::BloomDownsample);
    auto* upsample_pipeline = registry.get_pipeline(gpu::PipelineType::BloomUpsample);
    if (!downsample_pipeline || !upsample_pipeline) {
        return;
    }

    // The prefilter's exposure SSBO. Bound on EVERY downsample iteration, not
    // just the thresholding one: SDL3 GPU requires a slot the shader DECLARES
    // to be bound whether or not the branch that reads it runs, and an unbound
    // descriptor is undefined rather than merely unread.
    SDL_GPUBuffer* exposure_ssbo =
        exposure_buffer != nullptr ? exposure_buffer : dummy_exposure_buffer_->handle();

    // =========================================================================
    // Downsample chain: scene_color -> mip[0] -> mip[1] -> ... -> mip[N-1]
    // =========================================================================
    for (int i = 0; i < mip_count_; ++i) {
        // Source is either the scene color (first pass) or previous mip
        SDL_GPUTexture* src_tex = (i == 0) ? scene_color : mip_textures_[i - 1]->handle();
        int src_w = (i == 0) ? width_ : mip_textures_[i - 1]->width();
        int src_h = (i == 0) ? height_ : mip_textures_[i - 1]->height();

        gpu::BloomDownsampleUniforms uniforms = {};
        uniforms.srcTexelSize = glm::vec2(1.0f / src_w, 1.0f / src_h);
        uniforms.threshold = threshold;
        uniforms.isFirstPass = (i == 0) ? 1.0f : 0.0f;
        // The prefilter thresholds POST-exposure luminance, which is the manual
        // bias times exp2(adapted EV) — the same product composite applies.
        uniforms.exposure = exposure;
        uniforms.autoExposureEnabled = (exposure_buffer != nullptr) ? 1.0f : 0.0f;

        SDL_GPUColorTargetInfo color_target = {};
        color_target.texture = mip_textures_[i]->handle();
        color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
        color_target.store_op = SDL_GPU_STOREOP_STORE;

        SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
        if (!pass) {
            continue;
        }

        downsample_pipeline->bind(pass);
        SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

        SDL_GPUTextureSamplerBinding binding = {};
        binding.texture = src_tex;
        binding.sampler = linear_sampler_;
        SDL_BindGPUFragmentSamplers(pass, 0, &binding, 1);
        SDL_BindGPUFragmentStorageBuffers(pass, 0, &exposure_ssbo, 1);

        SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
        SDL_EndGPURenderPass(pass);
    }

    // =========================================================================
    // Progressive upsample chain: mip[N-1] -> mip[N-2] -> ... -> mip[0]
    // Each pass tent-filters the smaller mip and ADDS it onto the larger mip
    // (LOADOP_LOAD + additive pipeline), so the result accumulates every scale:
    // tight glow core from the large mips plus wide falloff from the small ones.
    // Previously LOADOP_DONT_CARE + an opaque pipeline overwrote each mip,
    // leaving only the smallest mip blurred up (no core).
    // =========================================================================
    for (int i = mip_count_ - 1; i > 0; --i) {
        // Source is the current (smaller) mip, destination is the next larger mip
        SDL_GPUTexture* src_tex = mip_textures_[i]->handle();
        int src_w = mip_textures_[i]->width();
        int src_h = mip_textures_[i]->height();

        gpu::BloomUpsampleUniforms uniforms = {};
        uniforms.srcTexelSize = glm::vec2(1.0f / src_w, 1.0f / src_h);
        uniforms.bloomRadius = 1.0f;

        SDL_GPUColorTargetInfo color_target = {};
        color_target.texture = mip_textures_[i - 1]->handle();
        color_target.load_op = SDL_GPU_LOADOP_LOAD;
        color_target.store_op = SDL_GPU_STOREOP_STORE;

        SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
        if (!pass) {
            continue;
        }

        upsample_pipeline->bind(pass);
        SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

        SDL_GPUTextureSamplerBinding binding = {};
        binding.texture = src_tex;
        binding.sampler = linear_sampler_;
        SDL_BindGPUFragmentSamplers(pass, 0, &binding, 1);

        SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
        SDL_EndGPURenderPass(pass);
    }
}

} // namespace bracken::render
