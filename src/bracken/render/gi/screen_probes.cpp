#include "bracken/render/gi/screen_probes.hpp"

#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/render/raytracing/rt_scene.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace bracken::render::gi {

namespace {

constexpr const char* LOG_TAG = "gi";

/// Every atlas is written by compute and read by compute or by the forward
/// pass, so they all carry the same usage set the probe field's atlases do.
constexpr SDL_GPUTextureUsageFlags ATLAS_USAGE = SDL_GPU_TEXTUREUSAGE_SAMPLER |
                                                 SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                                                 SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;

/// How often the diagnostic counter download is requested, in frames. Fenced
/// and polled, so doing it more often costs a command buffer and doing it less
/// often costs staleness — nothing in the frame waits either way.
constexpr std::uint64_t COUNTER_READBACK_INTERVAL = 8;

/// Bytes the readback transfer holds: the miss buffer's 16-byte header
/// (word 0 = the record count) followed by the 16-byte counters struct.
constexpr std::uint32_t READBACK_BYTES = 32;

/// The BVH pass's [numthreads].x. Sent to the args pass as a uniform rather
/// than hardcoded there — see GiScreenProbeArgsUniforms::threadsPerGroup.
constexpr std::uint32_t TRACE_THREADS_PER_GROUP = 64;

SDL_GPUStorageTextureReadWriteBinding rw_texture(SDL_GPUTexture* texture, bool cycle) {
    return SDL_GPUStorageTextureReadWriteBinding{.texture = texture,
                                                 .mip_level = 0,
                                                 .layer = 0,
                                                 .cycle = cycle,
                                                 .padding1 = 0,
                                                 .padding2 = 0,
                                                 .padding3 = 0};
}

SDL_GPUStorageBufferReadWriteBinding rw_buffer(SDL_GPUBuffer* buffer, bool cycle) {
    return SDL_GPUStorageBufferReadWriteBinding{
        .buffer = buffer, .cycle = cycle, .padding1 = 0, .padding2 = 0, .padding3 = 0};
}

std::size_t texture_bytes(glm::ivec2 size, std::size_t bytes_per_texel) {
    return static_cast<std::size_t>(size.x) * static_cast<std::size_t>(size.y) * bytes_per_texel;
}

} // namespace

ScreenProbeGather::~ScreenProbeGather() { shutdown(); }

bool ScreenProbeGather::init(gpu::GPUDevice& device, int width, int height) {
    shutdown();
    device_ = &device;
    width_ = width;
    height_ = height;
    grid_ = screen_probe_grid(width, height);
    if (grid_.x <= 0 || grid_.y <= 0) {
        BRACKEN_LOG_ERROR(LOG_TAG, "ScreenProbeGather: refusing to allocate for a {}x{} framebuffer",
                          width, height);
        device_ = nullptr;
        return false;
    }

    create_resources();
    if (!ready_) {
        shutdown();
        return false;
    }

    const glm::ivec2 radiance_size = screen_probe_radiance_atlas_size(grid_);
    BRACKEN_LOG_INFO(LOG_TAG,
                     "screen probes: {}x{} tiles ({} probes) at {}px, radiance {}x{} x3, "
                     "SH {} B/probe, miss capacity {}, {:.2f} MB",
                     grid_.x, grid_.y, screen_probe_count(grid_), SCREEN_PROBE_TILE, radiance_size.x,
                     radiance_size.y, sizeof(gpu::GiScreenProbeSh),
                     screen_probe_miss_capacity(grid_),
                     static_cast<double>(stats_.bytes) / (1024.0 * 1024.0));
    return true;
}

void ScreenProbeGather::create_resources() {
    ready_ = false;
    const glm::ivec2 radiance_size = screen_probe_radiance_atlas_size(grid_);
    const int probes = screen_probe_count(grid_);
    const int miss_capacity = screen_probe_miss_capacity(grid_);

    raw_ = gpu::GPUTexture::create_2d(*device_, radiance_size.x, radiance_size.y,
                                      SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, ATLAS_USAGE);
    for (auto& slot : filtered_) {
        slot = gpu::GPUTexture::create_2d(*device_, radiance_size.x, radiance_size.y,
                                          SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, ATLAS_USAGE);
    }
    if (!raw_ || !filtered_[0] || !filtered_[1]) {
        BRACKEN_LOG_ERROR(LOG_TAG, "ScreenProbeGather: could not allocate the probe atlases");
        return;
    }

    for (auto& slot : probes_) {
        slot = gpu::GPUBuffer::create_dynamic(
            *device_, gpu::GPUBuffer::Type::ComputeStorage,
            static_cast<std::size_t>(probes) * sizeof(gpu::GiScreenProbeHeader));
    }
    miss_ = gpu::GPUBuffer::create_dynamic(
        *device_, gpu::GPUBuffer::Type::ComputeStorage,
        static_cast<std::size_t>(gpu::GI_SCREEN_PROBE_MISS_HEADER_BYTES) +
            static_cast<std::size_t>(miss_capacity) * sizeof(gpu::GiScreenProbeMiss));
    counters_ = gpu::GPUBuffer::create_dynamic(*device_, gpu::GPUBuffer::Type::ComputeStorage,
                                               sizeof(gpu::GiScreenProbeCounters));
    // ComputeStorage, not Storage: the projection pass WRITES it from compute
    // and the scene fragment shaders READ it, which is exactly the usage triple
    // that type carries (COMPUTE_STORAGE_READ | WRITE | GRAPHICS_STORAGE_READ).
    sh_ = gpu::GPUBuffer::create_dynamic(*device_, gpu::GPUBuffer::Type::ComputeStorage,
                                         static_cast<std::size_t>(probes) *
                                             sizeof(gpu::GiScreenProbeSh));
    // Type::Indirect is INDIRECT | COMPUTE_STORAGE_WRITE: the args pass writes
    // it and SDL_DispatchGPUComputeIndirect reads it. Nothing else may.
    args_ = gpu::GPUBuffer::create_dynamic(*device_, gpu::GPUBuffer::Type::Indirect, 16);
    if (!probes_[0] || !probes_[1] || !miss_ || !counters_ || !args_ || !sh_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "ScreenProbeGather: could not allocate the gather buffers");
        return;
    }

    SDL_GPUSamplerCreateInfo info = {};
    info.min_filter = SDL_GPU_FILTER_NEAREST;
    info.mag_filter = SDL_GPU_FILTER_NEAREST;
    info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    // NEAREST on the pyramid too, and that is not laziness: a mip texel is a
    // conservative MAX over its footprint (render/hzb.hpp), and averaging two
    // of them produces a depth no surface has, which the march would then walk
    // straight through.
    point_clamp_sampler_ = device_->create_sampler(info);
    info.min_filter = SDL_GPU_FILTER_LINEAR;
    info.mag_filter = SDL_GPU_FILTER_LINEAR;
    linear_clamp_sampler_ = device_->create_sampler(info);
    if (point_clamp_sampler_ == nullptr || linear_clamp_sampler_ == nullptr) {
        BRACKEN_LOG_ERROR(LOG_TAG, "ScreenProbeGather: could not create the samplers");
        return;
    }

    SDL_GPUTransferBufferCreateInfo transfer_info = {};
    transfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
    transfer_info.size = READBACK_BYTES;
    readback_transfer_ = SDL_CreateGPUTransferBuffer(device_->handle(), &transfer_info);
    if (readback_transfer_ == nullptr) {
        BRACKEN_LOG_WARN(LOG_TAG, "ScreenProbeGather: no counter readback buffer; ray stats stay 0");
    }

    stats_ = GiScreenProbeStats{};
    stats_.grid = grid_;
    stats_.probes = static_cast<std::uint32_t>(probes);
    stats_.bytes = texture_bytes(radiance_size, 8) * 3 +
                   static_cast<std::size_t>(probes) * sizeof(gpu::GiScreenProbeSh) +
                   static_cast<std::size_t>(probes) * sizeof(gpu::GiScreenProbeHeader) * 2 +
                   static_cast<std::size_t>(miss_capacity) * sizeof(gpu::GiScreenProbeMiss);
    history_valid_ = false;
    write_ = 0;
    ready_ = true;
}

void ScreenProbeGather::shutdown() {
    if (device_ != nullptr) {
        if (readback_fence_ != nullptr) {
            // Wait rather than release blind: the fence guards a download into
            // memory about to be freed.
            SDL_WaitForGPUFences(device_->handle(), true, &readback_fence_, 1);
            SDL_ReleaseGPUFence(device_->handle(), readback_fence_);
            readback_fence_ = nullptr;
        }
        if (readback_transfer_ != nullptr) {
            SDL_ReleaseGPUTransferBuffer(device_->handle(), readback_transfer_);
            readback_transfer_ = nullptr;
        }
        if (point_clamp_sampler_ != nullptr) {
            device_->release_sampler(point_clamp_sampler_);
            point_clamp_sampler_ = nullptr;
        }
        if (linear_clamp_sampler_ != nullptr) {
            device_->release_sampler(linear_clamp_sampler_);
            linear_clamp_sampler_ = nullptr;
        }
    }
    readback_fence_ = nullptr;
    readback_transfer_ = nullptr;
    point_clamp_sampler_ = nullptr;
    linear_clamp_sampler_ = nullptr;

    raw_.reset();
    filtered_[0].reset();
    filtered_[1].reset();
    sh_.reset();
    probes_[0].reset();
    probes_[1].reset();
    miss_.reset();
    args_.reset();
    counters_.reset();

    device_ = nullptr;
    ready_ = false;
    width_ = 0;
    height_ = 0;
    grid_ = glm::ivec2(0);
    write_ = 0;
    history_valid_ = false;
    frame_counter_ = 0;
    stats_ = GiScreenProbeStats{};
}

void ScreenProbeGather::resize(int width, int height) {
    if ((width == width_ && height == height_) || device_ == nullptr) {
        return;
    }
    if (width <= 0 || height <= 0) {
        return;
    }
    gpu::GPUDevice& device = *device_;
    const int probes_before = screen_probe_count(grid_);
    shutdown();
    if (!init(device, width, height)) {
        BRACKEN_LOG_WARN(LOG_TAG, "ScreenProbeGather: resize to {}x{} failed; the gather is off",
                         width, height);
        return;
    }
    BRACKEN_LOG_DEBUG(LOG_TAG, "ScreenProbeGather: resized to {}x{} ({} -> {} probes, history dropped)",
                      width, height, probes_before, screen_probe_count(grid_));
}

void ScreenProbeGather::render(SDL_GPUCommandBuffer* cmd, const ScreenProbeParams& params) {
    stats_.active = false;
    stats_.grid = grid_;
    if (!ready_) {
        return;
    }
    poll_counter_readback();

    if (cmd == nullptr || params.pipelines == nullptr || params.hzb == nullptr ||
        params.gbuffer_normal == nullptr || params.scene == nullptr || !params.scene->ready() ||
        !params.scene->has_attributes() || params.velocity == nullptr ||
        params.shadow_atlas == nullptr || params.light_data == nullptr ||
        params.sky_cube_fallback == nullptr || params.sky_sampler == nullptr ||
        // The probe-state plane has no typed dummy here, and a null read-only
        // storage texture is undefined rather than empty. `GiMode::ProbesScreen`
        // always builds the world field, so this is a can't-happen guard rather
        // than a supported configuration.
        params.gi_probe_state == nullptr) {
        // A prerequisite is missing. Leaving the SH atlas at last frame's
        // contents would be worse than an empty gather: the forward pass would
        // light this frame's geometry with the previous frame's probes at
        // whatever the camera has since done. Drop the history so the next
        // successful frame starts clean, and report inactive so the caller
        // leaves giScreenEnabled at 0.
        history_valid_ = false;
        return;
    }
    if (params.screen_width != width_ || params.screen_height != height_) {
        return; // the caller resizes first; a mismatch would index the wrong tiles
    }

    auto* place = params.pipelines->get_compute_pipeline(gpu::ComputePipelineType::GiScreenProbePlace);
    auto* march = params.pipelines->get_compute_pipeline(gpu::ComputePipelineType::GiScreenProbeMarch);
    auto* args = params.pipelines->get_compute_pipeline(gpu::ComputePipelineType::GiScreenProbeArgs);
    auto* trace = params.pipelines->get_compute_pipeline(gpu::ComputePipelineType::GiScreenProbeTrace);
    auto* filter = params.pipelines->get_compute_pipeline(gpu::ComputePipelineType::GiScreenProbeFilter);
    auto* project =
        params.pipelines->get_compute_pipeline(gpu::ComputePipelineType::GiScreenProbeProject);
    if (place == nullptr || march == nullptr || args == nullptr || trace == nullptr ||
        filter == nullptr || project == nullptr) {
        history_valid_ = false;
        return;
    }

    BRACKEN_PROFILE_ZONE("ScreenProbeGather::render");

    const std::uint32_t probe_count = static_cast<std::uint32_t>(screen_probe_count(grid_));
    const std::uint32_t miss_capacity = static_cast<std::uint32_t>(screen_probe_miss_capacity(grid_));
    const std::uint32_t frame_index = static_cast<std::uint32_t>(frame_counter_);
    // The history is only usable when TAA actually resolved a previous frame.
    // Without it the march has no radiance source at all and hands EVERY ray to
    // the BVH pass, which is slower and correct rather than fast and invented.
    const bool history_ready = params.scene_history != nullptr;

    gpu::GPUBuffer* probes_write = probes_[static_cast<std::size_t>(write_)].get();
    gpu::GPUBuffer* probes_read = probes_[static_cast<std::size_t>(write_ ^ 1)].get();
    gpu::GPUTexture* filtered_write = filtered_[static_cast<std::size_t>(write_)].get();
    gpu::GPUTexture* filtered_read = filtered_[static_cast<std::size_t>(write_ ^ 1)].get();

    // ---- 1. placement (also zeroes the miss header + counters) -------------
    {
        gpu::GiScreenProbePlaceUniforms u{};
        u.invViewProj = params.inv_view_projection;
        u.cameraPosition = params.camera_position;
        u.frameIndex = frame_index;
        u.probeGrid = grid_;
        u.screenSize = glm::ivec2(width_, height_);
        u.missCapacity = miss_capacity;

        const std::array<SDL_GPUStorageBufferReadWriteBinding, 3> rw{
            rw_buffer(probes_write->handle(), false), rw_buffer(miss_->handle(), false),
            rw_buffer(counters_->handle(), false)};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {}, rw);
        if (!pass) {
            return;
        }
        pass.bind_pipeline(*place);
        const std::array<SDL_GPUTextureSamplerBinding, 2> textures{
            {{params.hzb, point_clamp_sampler_}, {params.gbuffer_normal, point_clamp_sampler_}}};
        pass.bind_samplers(0, textures);
        pass.push_uniforms(0, &u, sizeof(u));
        pass.dispatch_for(*place, probe_count);
    }

    // ---- 2. screen march + miss compaction ---------------------------------
    {
        gpu::GiScreenProbeMarchUniforms u{};
        u.viewProj = params.view_projection;
        u.prevViewProjUnjittered = params.prev_view_projection_unjittered;
        u.cameraPosition = params.camera_position;
        u.depthA = params.depth_a;
        u.depthB = params.depth_b;
        u.probeGrid = grid_;
        u.screenSize = glm::ivec2(width_, height_);
        u.rayMaxDistance = params.screen_ray_max_distance;
        u.normalBias = SCREEN_PROBE_NORMAL_BIAS;
        u.historyValid = history_ready ? 1.0f : 0.0f;
        u.frameIndex = frame_index;
        u.mipCount = static_cast<std::uint32_t>(std::max(1, params.hzb_mip_count));
        u.missCapacity = miss_capacity;

        const auto rw_tex = rw_texture(raw_->handle(), false);
        const std::array<SDL_GPUStorageBufferReadWriteBinding, 2> rw_buf{
            rw_buffer(miss_->handle(), false), rw_buffer(counters_->handle(), false)};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw_tex, 1}, rw_buf);
        if (!pass) {
            return;
        }
        pass.bind_pipeline(*march);
        // History slot is never left dangling: with no TAA history the pyramid
        // stands in as a typed filler and `historyValid = 0` is what stops it
        // being read.
        const std::array<SDL_GPUTextureSamplerBinding, 2> textures{
            {{params.hzb, point_clamp_sampler_},
             {history_ready ? params.scene_history : params.hzb, linear_clamp_sampler_}}};
        pass.bind_samplers(0, textures);
        const std::array<SDL_GPUBuffer*, 1> buffers{probes_write->handle()};
        pass.bind_storage_buffers(0, buffers);
        pass.push_uniforms(0, &u, sizeof(u));
        // One threadgroup per probe: the shader's [numthreads] is the probe's
        // whole 64-texel payload, and its groupshared compaction requires it.
        pass.dispatch(probe_count, 1, 1);
    }

    // ---- 3. indirect dispatch arguments ------------------------------------
    {
        gpu::GiScreenProbeArgsUniforms u{};
        u.missCapacity = miss_capacity;
        u.threadsPerGroup = TRACE_THREADS_PER_GROUP;

        const auto rw = rw_buffer(args_->handle(), false);
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {}, {&rw, 1});
        if (!pass) {
            return;
        }
        pass.bind_pipeline(*args);
        const std::array<SDL_GPUBuffer*, 1> buffers{miss_->handle()};
        pass.bind_storage_buffers(0, buffers);
        pass.push_uniforms(0, &u, sizeof(u));
        pass.dispatch(1, 1, 1);
    }

    // ---- 4. the handed-off rays, INDIRECTLY dispatched ----------------------
    //
    // Its own pass, and it has to be: SDL provides no intra-pass ordering, so
    // the args pass must have ENDED before the dispatch that reads what it
    // wrote (gpu/gpu_compute.hpp records the requirement). This is the engine's
    // first runtime use of ComputePass::dispatch_indirect.
    {
        gpu::GiScreenProbeTraceUniforms u{};
        u.shadow = params.shadow;
        // PCF, never PCSS — the same call the probe field makes, and for the
        // same reason: a handed-off ray's visibility is integrated by the
        // probe's temporal EMA, so twenty atlas taps to soften one sample buy
        // nothing. `shadowEnabled == 0` still means no shadows and is honoured.
        if (u.shadow.shadowEnabled > 1.5f) {
            u.shadow.shadowEnabled = 1.0f;
        }
        u.cameraPosition = params.camera_position;
        u.cameraForward = params.camera_forward;
        u.rayMaxDistance = SCREEN_PROBE_WORLD_RAY_CLAMP;
        u.normalBias = SCREEN_PROBE_NORMAL_BIAS;
        u.sunDirection = params.environment.sun_dir;
        u.sunColor = params.environment.sun_color * params.sun_intensity;
        u.bounceScale = 1.0f;
        u.ambientSky = params.environment.ambient_sky;
        u.ambientGround = params.environment.ambient_ground;
        u.useSkyCube = params.sky_cube != nullptr ? 1.0f : 0.0f;
        u.giEnabled = params.gi_irradiance != nullptr ? params.gi_enabled : 0.0f;
        // 1: a ray that escaped the 600-unit clamp outside the field's coverage
        // has to be lit by something, and the engine's documented no-GI ambient
        // tier is the hemisphere. See rtHitAmbient.
        u.giAmbientFallback = 1.0f;
        for (std::size_t c = 0; c < CASCADE_COUNT; ++c) {
            u.cascadeBase[c] = params.gi_cascade_base[c];
        }
        u.tlasNodeCount = params.scene->tlas_node_count();
        u.lightCount = params.light_count;
        u.lightCapacity = params.light_capacity;
        u.missCapacity = miss_capacity;
        u.probeGrid = grid_;
        u.frameIndex = frame_index;
        u.horizonBlend = 1.0f;

        const auto rw_tex = rw_texture(raw_->handle(), false);
        const auto rw_buf = rw_buffer(counters_->handle(), false);
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw_tex, 1}, {&rw_buf, 1});
        if (!pass) {
            return;
        }
        pass.bind_pipeline(*trace);
        const std::array<SDL_GPUTextureSamplerBinding, 4> textures{{
            {params.shadow_atlas, params.shadow_compare_sampler},
            {params.shadow_atlas, params.shadow_raw_sampler},
            {params.sky_cube != nullptr ? params.sky_cube : params.sky_cube_fallback,
             params.sky_sampler},
            {params.gi_irradiance != nullptr ? params.gi_irradiance : raw_->handle(),
             params.gi_irradiance_sampler != nullptr ? params.gi_irradiance_sampler
                                                     : linear_clamp_sampler_},
        }};
        pass.bind_samplers(0, textures);
        const std::array<SDL_GPUTexture*, 1> storage_textures{params.gi_probe_state};
        pass.bind_storage_textures(0, storage_textures);
        const std::array<SDL_GPUBuffer*, 8> buffers{
            params.scene->tlas_nodes(),          params.scene->tlas_instances(),
            params.scene->blas_nodes(),          params.scene->blas_triangles(),
            params.scene->triangle_attributes(), params.light_data,
            probes_write->handle(),              miss_->handle()};
        pass.bind_storage_buffers(0, buffers);
        pass.push_uniforms(0, &u, sizeof(u));
        pass.dispatch_indirect(*args_);
    }

    // ---- 5. probe-space temporal filter -------------------------------------
    {
        gpu::GiScreenProbeFilterUniforms u{};
        u.probeGrid = grid_;
        u.screenSize = glm::ivec2(width_, height_);
        u.alpha = SCREEN_PROBE_TEMPORAL_ALPHA;
        u.historyValid = history_valid_ ? 1.0f : 0.0f;
        u.fireflyMax = SCREEN_PROBE_FIREFLY_MAX;
        u.frameIndex = frame_index;

        const auto rw = rw_texture(filtered_write->handle(), false);
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (!pass) {
            return;
        }
        pass.bind_pipeline(*filter);
        const std::array<SDL_GPUTextureSamplerBinding, 3> textures{
            {{raw_->handle(), point_clamp_sampler_},
             {filtered_read->handle(), point_clamp_sampler_},
             {params.velocity, point_clamp_sampler_}}};
        pass.bind_samplers(0, textures);
        const std::array<SDL_GPUBuffer*, 2> buffers{probes_write->handle(), probes_read->handle()};
        pass.bind_storage_buffers(0, buffers);
        pass.push_uniforms(0, &u, sizeof(u));
        pass.dispatch(probe_count, 1, 1);
    }

    // ---- 6. spatial filter + L1 projection ----------------------------------
    {
        gpu::GiScreenProbeProjectUniforms u{};
        u.probeGrid = grid_;
        u.normalPower = SCREEN_PROBE_NORMAL_POWER;
        u.frameIndex = frame_index;

        const auto rw = rw_buffer(sh_->handle(), false);
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {}, {&rw, 1});
        if (!pass) {
            return;
        }
        pass.bind_pipeline(*project);
        const std::array<SDL_GPUTextureSamplerBinding, 1> textures{
            {{filtered_write->handle(), point_clamp_sampler_}}};
        pass.bind_samplers(0, textures);
        const std::array<SDL_GPUBuffer*, 1> buffers{probes_write->handle()};
        pass.bind_storage_buffers(0, buffers);
        pass.push_uniforms(0, &u, sizeof(u));
        pass.dispatch(probe_count, 1, 1);
    }

    write_ ^= 1;
    history_valid_ = true;
    stats_.active = true;
    if (frame_counter_ % COUNTER_READBACK_INTERVAL == 0) {
        request_counter_readback();
    }
    ++frame_counter_;
}

void ScreenProbeGather::poll_counter_readback() {
    if (readback_fence_ == nullptr || device_ == nullptr) {
        return;
    }
    if (!SDL_QueryGPUFence(device_->handle(), readback_fence_)) {
        return; // still in flight — never block a frame on a diagnostic
    }
    SDL_ReleaseGPUFence(device_->handle(), readback_fence_);
    readback_fence_ = nullptr;

    void* mapped = device_->map_transfer_buffer(readback_transfer_, false);
    if (mapped == nullptr) {
        return;
    }
    std::array<std::uint32_t, 4> header{};
    gpu::GiScreenProbeCounters counters{};
    std::memcpy(header.data(), mapped, sizeof(header));
    std::memcpy(&counters, static_cast<const std::uint8_t*>(mapped) + sizeof(header),
                sizeof(counters));
    device_->unmap_transfer_buffer(readback_transfer_);

    stats_.world_rays = header[0];
    stats_.screen_hits = counters.screenHits;
    stats_.world_hits = counters.worldHits;
    stats_.fallbacks = counters.fallbacks;
    stats_.dropped = counters.dropped;
}

void ScreenProbeGather::request_counter_readback() {
    if (device_ == nullptr || readback_transfer_ == nullptr || readback_fence_ != nullptr) {
        return;
    }
    // Its own tiny command buffer, submitted before the frame's — so it reads
    // the PREVIOUS frame's counters. Fine for a diagnostic, and it keeps the
    // frame's command buffer free of a fence. Same shape as ProbeVolume's.
    SDL_GPUCommandBuffer* readback_cmd = SDL_AcquireGPUCommandBuffer(device_->handle());
    if (readback_cmd == nullptr) {
        return;
    }
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(readback_cmd);
    if (copy == nullptr) {
        SDL_CancelGPUCommandBuffer(readback_cmd);
        return;
    }
    SDL_GPUBufferRegion src = {};
    SDL_GPUTransferBufferLocation dst = {};
    dst.transfer_buffer = readback_transfer_;

    // The miss count lives in the miss buffer's own header (see
    // GiScreenProbeMiss); the rest live in the counters buffer. Two downloads
    // into one transfer, one fence.
    src.buffer = miss_->handle();
    src.offset = 0;
    src.size = 16;
    dst.offset = 0;
    SDL_DownloadFromGPUBuffer(copy, &src, &dst);

    src.buffer = counters_->handle();
    src.offset = 0;
    src.size = static_cast<Uint32>(sizeof(gpu::GiScreenProbeCounters));
    dst.offset = 16;
    SDL_DownloadFromGPUBuffer(copy, &src, &dst);

    SDL_EndGPUCopyPass(copy);
    readback_fence_ = SDL_SubmitGPUCommandBufferAndAcquireFence(readback_cmd);
}

} // namespace bracken::render::gi
