#include "auto_exposure.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include <array>
#include <cstring>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_log.h>

namespace bracken::render {

AutoExposure::AutoExposure() = default;

AutoExposure::~AutoExposure() {
    shutdown();
}

bool AutoExposure::init(gpu::GPUDevice& device) {
    device_ = &device;

    // Both buffers start zero-filled: the histogram's per-frame reset is the
    // adapt pass zeroing bins after reading them, so frame 0 must arrive
    // clean; the state buffer's zero "valid" marker just means "never
    // written" until the first adapt dispatch.
    const std::array<uint32_t, EXPOSURE_HISTOGRAM_BINS> zero_bins{};
    histogram_ = gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::ComputeStorage, zero_bins.data(),
                                               zero_bins.size() * sizeof(uint32_t));

    const std::array<float, 4> zero_state{};
    state_ = gpu::GPUBuffer::create_static(device, gpu::GPUBuffer::Type::ComputeStorage, zero_state.data(),
                                           zero_state.size() * sizeof(float));

    SDL_GPUSamplerCreateInfo sampler_info = {};
    sampler_info.min_filter = SDL_GPU_FILTER_NEAREST;
    sampler_info.mag_filter = SDL_GPU_FILTER_NEAREST;
    sampler_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    sampler_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sampler_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sampler_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    nearest_clamp_sampler_ = device.create_sampler(sampler_info);

    SDL_GPUTransferBufferCreateInfo transfer_info = {};
    transfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
    transfer_info.size = 4 * sizeof(float);
    readback_transfer_ = device.create_transfer_buffer(transfer_info);

    snap_frames_left_ = SNAP_FRAMES;

    if (!is_ready()) {
        SDL_Log("AutoExposure: Failed to initialize");
        return false;
    }

    SDL_Log("AutoExposure: Initialized (%d-bin log2 histogram, EV %.0f..%.0f)", EXPOSURE_HISTOGRAM_BINS,
            EXPOSURE_MIN_LOG2_LUMA, EXPOSURE_MAX_LOG2_LUMA);
    return true;
}

void AutoExposure::shutdown() {
    histogram_.reset();
    state_.reset();

    if (device_) {
        if (readback_fence_) {
            SDL_ReleaseGPUFence(device_->handle(), readback_fence_);
            readback_fence_ = nullptr;
        }
        if (readback_transfer_) {
            device_->release_transfer_buffer(readback_transfer_);
            readback_transfer_ = nullptr;
        }
        if (nearest_clamp_sampler_) {
            device_->release_sampler(nearest_clamp_sampler_);
            nearest_clamp_sampler_ = nullptr;
        }
    }

    latest_state_.reset();
    device_ = nullptr;
}

bool AutoExposure::is_ready() const {
    return device_ != nullptr && histogram_ && state_ && nearest_clamp_sampler_ != nullptr;
}

void AutoExposure::set_percentiles(float low, float high) {
    low_percentile_ = std::clamp(low, 0.0f, 1.0f);
    high_percentile_ = std::clamp(high, 0.0f, 1.0f - low_percentile_);
}

void AutoExposure::render(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, gpu::GPUTexture& scene_color,
                          const FrameParams& params) {
    BRACKEN_PROFILE_ZONE_COLOR("AutoExposure::render", 0x16A085);
    if (!is_ready() || !cmd) {
        return;
    }

    auto* histogram_pipeline = pipelines.get_compute_pipeline(gpu::ComputePipelineType::ExposureHistogram);
    auto* adapt_pipeline = pipelines.get_compute_pipeline(gpu::ComputePipelineType::ExposureAdapt);
    if (!histogram_pipeline || !adapt_pipeline) {
        return;
    }

    const uint32_t width = static_cast<uint32_t>(scene_color.width());
    const uint32_t height = static_cast<uint32_t>(scene_color.height());
    if (width == 0 || height == 0) {
        return;
    }

    // ---- Pass 1: log2-luminance histogram (one thread per pixel; LDS
    // accumulation then a per-bin atomic merge). The histogram buffer arrives
    // zeroed (init upload / last frame's adapt pass) — cycle must stay false
    // on both passes or the accumulated state would be swapped away.
    {
        SDL_GPUStorageBufferReadWriteBinding rw = {};
        rw.buffer = histogram_->handle();
        rw.cycle = false;

        auto pass = gpu::ComputePass::begin(cmd, {}, {&rw, 1});
        if (pass) {
            pass.bind_pipeline(*histogram_pipeline);

            SDL_GPUTextureSamplerBinding scene_binding = {};
            scene_binding.texture = scene_color.handle();
            scene_binding.sampler = nearest_clamp_sampler_;
            pass.bind_samplers(0, {&scene_binding, 1});

            gpu::ExposureHistogramUniforms uniforms = {};
            uniforms.width = width;
            uniforms.height = height;
            uniforms.minLog2Luma = EXPOSURE_MIN_LOG2_LUMA;
            uniforms.invLog2Range = 1.0f / (EXPOSURE_MAX_LOG2_LUMA - EXPOSURE_MIN_LOG2_LUMA);
            pass.push_uniforms(0, &uniforms, sizeof(uniforms));

            pass.dispatch_for(*histogram_pipeline, width, height);
            pass.end();
        }
    }

    // ---- Pass 2: reduce + temporal adaptation (single workgroup). Also
    // zeroes the histogram for the next frame.
    {
        SDL_GPUStorageBufferReadWriteBinding rw[2] = {};
        rw[0].buffer = histogram_->handle();
        rw[0].cycle = false;
        rw[1].buffer = state_->handle();
        rw[1].cycle = false;

        auto pass = gpu::ComputePass::begin(cmd, {}, rw);
        if (pass) {
            pass.bind_pipeline(*adapt_pipeline);

            gpu::ExposureAdaptUniforms uniforms = {};
            uniforms.minLog2Luma = EXPOSURE_MIN_LOG2_LUMA;
            uniforms.log2Range = EXPOSURE_MAX_LOG2_LUMA - EXPOSURE_MIN_LOG2_LUMA;
            uniforms.deltaTime = params.dt;
            uniforms.firstFrame = snap_frames_left_ > 0 ? 1.0f : 0.0f;
            uniforms.speedUp = params.speed_up;
            uniforms.speedDown = params.speed_down;
            uniforms.minEv = params.min_ev;
            uniforms.maxEv = params.max_ev;
            uniforms.lowPercentile = low_percentile_;
            uniforms.highPercentile = high_percentile_;
            uniforms.log2TargetGray = log2_target_gray_;
            pass.push_uniforms(0, &uniforms, sizeof(uniforms));

            pass.dispatch(1, 1, 1);
            pass.end();
        }
    }

    if (snap_frames_left_ > 0) {
        --snap_frames_left_;
    }

    // ---- Debug readback upkeep (tiny fenced download of last frame's state
    // at most every readback_interval_ seconds; polled non-blocking).
    since_readback_ += params.dt;
    poll_readback();
    if (!readback_fence_ && since_readback_ >= readback_interval_) {
        request_readback();
    }
}

void AutoExposure::poll_readback() {
    if (!readback_fence_ || !device_) {
        return;
    }
    if (!SDL_QueryGPUFence(device_->handle(), readback_fence_)) {
        return; // still in flight
    }
    SDL_ReleaseGPUFence(device_->handle(), readback_fence_);
    readback_fence_ = nullptr;

    void* mapped = device_->map_transfer_buffer(readback_transfer_, false);
    if (!mapped) {
        return;
    }
    float values[4];
    std::memcpy(values, mapped, sizeof(values));
    device_->unmap_transfer_buffer(readback_transfer_);

    if (values[3] != 0.0f) { // valid marker — adapt has written at least once
        latest_state_ = ExposureState{values[0], values[1], values[2]};
    }
}

void AutoExposure::request_readback() {
    if (!device_ || !readback_transfer_) {
        return;
    }
    // Own tiny command buffer: submitted before the current frame's command
    // buffer, so it reads the PREVIOUS frame's adapt result — fine for a
    // diagnostic. SDL orders and synchronizes submissions on the queue.
    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_->handle());
    if (!cmd) {
        return;
    }
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
    if (!copy) {
        SDL_CancelGPUCommandBuffer(cmd);
        return;
    }
    SDL_GPUBufferRegion src = {};
    src.buffer = state_->handle();
    src.offset = 0;
    src.size = 4 * sizeof(float);
    SDL_GPUTransferBufferLocation dst = {};
    dst.transfer_buffer = readback_transfer_;
    dst.offset = 0;
    SDL_DownloadFromGPUBuffer(copy, &src, &dst);
    SDL_EndGPUCopyPass(copy);

    readback_fence_ = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    since_readback_ = 0.0f;
}

} // namespace bracken::render
