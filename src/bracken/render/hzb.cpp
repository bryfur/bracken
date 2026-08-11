#include "hzb.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"

#include <SDL3/SDL_gpu.h>
#include <array>

namespace bracken::render {

namespace {
constexpr const char* LOG_TAG = "hzb";

constexpr SDL_GPUTextureFormat HZB_FORMAT = SDL_GPU_TEXTUREFORMAT_R32_FLOAT;

// SAMPLER because R4b's screen march wants an explicit-LOD fetch;
// SIMULTANEOUS_READ_WRITE because hzb_reduce.comp.hlsl reads one mip and writes
// the next of this same texture, and SDL3 documents that reading a texture
// inside a compute pass requires exactly this flag (plain READ | WRITE means
// "one shader writes, a DIFFERENT shader reads").
constexpr SDL_GPUTextureUsageFlags HZB_USAGE =
    SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE;

} // namespace

HzbPyramid::~HzbPyramid() {
    shutdown();
}

bool HzbPyramid::init(gpu::GPUDevice& device, int width, int height, HzbReduction reduction) {
    device_ = &device;
    width_ = width;
    height_ = height;
    reduction_ = reduction;
    mip_count_ = hzb_mip_count(width, height);

    if (!device.supports_format(HZB_FORMAT, SDL_GPU_TEXTURETYPE_2D, HZB_USAGE)) {
        // Checked rather than assumed: SDL's own header says SIMULTANEOUS
        // read-write "is only supported by a limited number of texture
        // formats", and a create that fails this check would otherwise show up
        // as a null texture with no explanation.
        BRACKEN_LOG_WARN(LOG_TAG,
                         "R32_FLOAT does not support SAMPLER | COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE "
                         "on this device; the Hi-Z pyramid will not be built");
        width_ = height_ = mip_count_ = 0;
        return false;
    }

    create_texture();

    SDL_GPUSamplerCreateInfo nearest_info = {};
    nearest_info.min_filter = SDL_GPU_FILTER_NEAREST;
    nearest_info.mag_filter = SDL_GPU_FILTER_NEAREST;
    nearest_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    nearest_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    nearest_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    nearest_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    nearest_clamp_sampler_ = device_->create_sampler(nearest_info);

    if (!texture_ || nearest_clamp_sampler_ == nullptr) {
        BRACKEN_LOG_ERROR(LOG_TAG, "HzbPyramid: failed to initialize at {}x{}", width, height);
        shutdown();
        return false;
    }

    BRACKEN_LOG_DEBUG(LOG_TAG, "HzbPyramid: {}x{} R32_FLOAT, {} mips, {} surface", width_, height_, mip_count_,
                      reduction_ == HzbReduction::Farthest ? "farthest" : "closest");
    return true;
}

void HzbPyramid::shutdown() {
    texture_.reset();
    if (device_ != nullptr && nearest_clamp_sampler_ != nullptr) {
        device_->release_sampler(nearest_clamp_sampler_);
    }
    nearest_clamp_sampler_ = nullptr;
    width_ = 0;
    height_ = 0;
    mip_count_ = 0;
}

void HzbPyramid::create_texture() {
    if (device_ == nullptr || width_ <= 0 || height_ <= 0) {
        return;
    }
    texture_ = gpu::GPUTexture::create_2d(*device_, width_, height_, HZB_FORMAT, HZB_USAGE,
                                          static_cast<uint32_t>(mip_count_));
    if (!texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "HzbPyramid: could not create the {}x{} pyramid texture", width_, height_);
    }
}

void HzbPyramid::resize(int width, int height) {
    if (width == width_ && height == height_) {
        return;
    }
    if (width <= 0 || height <= 0) {
        return;
    }
    width_ = width;
    height_ = height;
    mip_count_ = hzb_mip_count(width, height);
    // No history to preserve: every level is rewritten from this frame's depth
    // before anything reads it.
    create_texture();
}

void HzbPyramid::build(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, gpu::GPUTexture& depth) {
    if (cmd == nullptr || !is_ready() || nearest_clamp_sampler_ == nullptr) {
        return;
    }
    if (depth.width() != width_ || depth.height() != height_) {
        BRACKEN_LOG_WARN(LOG_TAG, "HzbPyramid::build: depth is {}x{} but the pyramid is {}x{}; skipped",
                         depth.width(), depth.height(), width_, height_);
        return;
    }

    BRACKEN_PROFILE_ZONE_COLOR("HZB", 0x2C81C4);

    // ---- mip 0: copy the prepass depth -------------------------------------
    gpu::GPUComputePipeline* copy = pipelines.get_compute_pipeline(gpu::ComputePipelineType::HzbCopy);
    if (copy == nullptr) {
        return;
    }
    {
        gpu::HzbCopyUniforms u{};
        u.mipWidth = static_cast<uint32_t>(width_);
        u.mipHeight = static_cast<uint32_t>(height_);

        SDL_GPUStorageTextureReadWriteBinding rw{};
        rw.texture = texture_->handle();
        rw.mip_level = 0;
        rw.layer = 0;
        // cycle=false: the pyramid is fully overwritten every frame and nothing
        // reads it across frames, so a fresh allocation would buy nothing and
        // would invalidate the mip chain's other levels' backing.
        rw.cycle = false;

        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (!pass) {
            return;
        }
        pass.bind_pipeline(*copy);
        const std::array<SDL_GPUTextureSamplerBinding, 1> textures{{{depth.handle(), nearest_clamp_sampler_}}};
        pass.bind_samplers(0, textures);
        pass.push_uniforms(0, &u, sizeof(u));
        pass.dispatch_for(*copy, static_cast<uint32_t>(width_), static_cast<uint32_t>(height_));
    }

    // ---- mips 1..N-1: conservative reduction, one PASS each -----------------
    //
    // A pass per mip, not a dispatch per mip inside one pass: SDL3 GPU offers
    // no intra-pass synchronization, and its documentation is explicit that a
    // dispatch depending on a previous dispatch's output must sit in a new
    // pass. Collapsing these would read half-written mips.
    // The reduction is baked into the level chain, so the pipeline is chosen
    // once from what this pyramid IS, never per call site. Wiring a culler to
    // the closest-surface chain is the failure hzb_reduce_far's comment is
    // about.
    gpu::GPUComputePipeline* reduce = pipelines.get_compute_pipeline(
        reduction_ == HzbReduction::Farthest ? gpu::ComputePipelineType::HzbReduceFar
                                             : gpu::ComputePipelineType::HzbReduce);
    if (reduce == nullptr) {
        return;
    }
    for (int level = 1; level < mip_count_; ++level) {
        const int src_w = hzb_mip_size(width_, level - 1);
        const int src_h = hzb_mip_size(height_, level - 1);
        const int dst_w = hzb_mip_size(width_, level);
        const int dst_h = hzb_mip_size(height_, level);

        gpu::HzbReduceUniforms u{};
        u.dstWidth = static_cast<uint32_t>(dst_w);
        u.dstHeight = static_cast<uint32_t>(dst_h);
        u.srcWidth = static_cast<uint32_t>(src_w);
        u.srcHeight = static_cast<uint32_t>(src_h);

        std::array<SDL_GPUStorageTextureReadWriteBinding, 2> rw{};
        rw[0].texture = texture_->handle();
        rw[0].mip_level = static_cast<Uint32>(level - 1); // source
        rw[0].layer = 0;
        rw[0].cycle = false;
        rw[1].texture = texture_->handle();
        rw[1].mip_level = static_cast<Uint32>(level); // destination
        rw[1].layer = 0;
        rw[1].cycle = false;

        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, rw, {});
        if (!pass) {
            return;
        }
        pass.bind_pipeline(*reduce);
        pass.push_uniforms(0, &u, sizeof(u));
        pass.dispatch_for(*reduce, static_cast<uint32_t>(dst_w), static_cast<uint32_t>(dst_h));
    }
}

std::size_t HzbPyramid::level_readback_bytes(int level) const {
    if (level < 0 || level >= mip_count_) {
        return 0;
    }
    const std::size_t w = static_cast<std::size_t>(hzb_mip_size(width_, level));
    const std::size_t h = static_cast<std::size_t>(hzb_mip_size(height_, level));
    return w * h * sizeof(float);
}

SDL_GPUFence* HzbPyramid::request_level_readback(int level, SDL_GPUTransferBuffer* dst) {
    BRACKEN_ASSERT_MAIN_THREAD();
    if (device_ == nullptr || !is_ready() || dst == nullptr) {
        return nullptr;
    }
    if (level < 1 || level >= mip_count_) {
        // Level 0 is refused rather than clamped: it is the whole screen, and a
        // caller asking for it has almost certainly mixed up "the pyramid" with
        // "a coarse level of the pyramid".
        BRACKEN_LOG_WARN(LOG_TAG, "HzbPyramid::request_level_readback: level {} is out of range [1, {})", level,
                         mip_count_);
        return nullptr;
    }

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_->handle());
    if (cmd == nullptr) {
        return nullptr;
    }
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
    if (copy == nullptr) {
        SDL_CancelGPUCommandBuffer(cmd);
        return nullptr;
    }

    SDL_GPUTextureRegion src = {};
    src.texture = texture_->handle();
    src.mip_level = static_cast<Uint32>(level);
    src.layer = 0;
    src.w = static_cast<Uint32>(hzb_mip_size(width_, level));
    src.h = static_cast<Uint32>(hzb_mip_size(height_, level));
    src.d = 1;

    SDL_GPUTextureTransferInfo info = {};
    info.transfer_buffer = dst;
    info.offset = 0;
    // Zero means "tightly packed" per SDL's own documentation; spelling it out
    // rather than leaving it implicit, because a wrong pitch here reads as a
    // sheared depth image on the CPU and looks like a projection bug.
    info.pixels_per_row = src.w;
    info.rows_per_layer = src.h;

    SDL_DownloadFromGPUTexture(copy, &src, &info);
    SDL_EndGPUCopyPass(copy);
    return SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
}

} // namespace bracken::render
