#include "bracken/render/gi/probe_volume.hpp"

#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/render/raytracing/rt_scene.hpp"

#include <algorithm>
#include <cstring>

namespace bracken::render::gi {

namespace {

constexpr const char* LOG_TAG = "gi";

/// Every atlas is both sampled and written by compute, so they all carry the
/// same usage set the traced targets do (`RtRenderer::ensure_target`).
constexpr SDL_GPUTextureUsageFlags ATLAS_USAGE = SDL_GPU_TEXTUREUSAGE_SAMPLER |
                                                 SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                                                 SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;

/// How often the diagnostic counter download is requested, in frames. It is
/// fenced and polled, so the only cost of doing it more often is a command
/// buffer; the only cost of doing it less often is a staler number.
constexpr std::uint64_t COUNTER_READBACK_INTERVAL = 8;

std::size_t texture_bytes(int width, int height, std::size_t bytes_per_texel) {
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * bytes_per_texel;
}

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

} // namespace

ProbeVolume::~ProbeVolume() { shutdown(); }

bool ProbeVolume::init(gpu::GPUDevice& device) {
    shutdown();
    device_ = &device;

    const int irr_w = atlas_width(IRRADIANCE_PROBE_TEXELS);
    const int irr_h = atlas_height(IRRADIANCE_PROBE_TEXELS);
    const int vis_w = atlas_width(VISIBILITY_PROBE_TEXELS);
    const int vis_h = atlas_height(VISIBILITY_PROBE_TEXELS);
    const int plane_w = PROBES_PER_ATLAS_ROW;
    const int plane_h = ATLAS_ROWS_ALL_CASCADES;

    stats_ = GiStats{};
    // One texture each for the whole field; cascade c is a band of rows.
    irradiance_ = gpu::GPUTexture::create_2d(device, irr_w, irr_h,
                                             SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, ATLAS_USAGE);
    visibility_ =
        gpu::GPUTexture::create_2d(device, vis_w, vis_h, SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT, ATLAS_USAGE);
    state_ = gpu::GPUTexture::create_2d(device, plane_w, plane_h, SDL_GPU_TEXTUREFORMAT_R8_UINT, ATLAS_USAGE);
    // One texel per probe: xyz = relocation offset in world units,
    // w = classification verdict. Half floats are ample for an offset
    // bounded at 0.45 * 600 = 270 units.
    offset_ = gpu::GPUTexture::create_2d(device, plane_w, plane_h,
                                         SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, ATLAS_USAGE);
    if (!irradiance_ || !visibility_ || !state_ || !offset_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "ProbeVolume: could not allocate the field atlases");
        shutdown();
        return false;
    }

    for (int c = 0; c < CASCADE_COUNT; ++c) {
        cascades_[static_cast<std::size_t>(c)].cpu.reset(c);

        // One update list per cascade — see the member's comment for why they
        // cannot share one buffer.
        update_lists_[static_cast<std::size_t>(c)] = gpu::GPUBuffer::create_dynamic(
            device, gpu::GPUBuffer::Type::Storage,
            static_cast<std::size_t>(gi_max_probes_per_dispatch()) * sizeof(gpu::GiProbeUpdateRecord));
        if (!update_lists_[static_cast<std::size_t>(c)]) {
            BRACKEN_LOG_ERROR(LOG_TAG, "ProbeVolume: could not allocate cascade {} update list", c);
            shutdown();
            return false;
        }

        GiCascadeStats& cs = stats_.per_cascade[static_cast<std::size_t>(c)];
        cs.spacing = cascade_spacing(c);
        cs.probes = static_cast<std::uint32_t>(PROBES_PER_CASCADE);
        // One cascade's SHARE of the banded textures — the totals below are
        // unchanged by the banding, so the per-cascade figure stays comparable
        // with R1a's.
        cs.atlas_bytes = texture_bytes(irr_w, irr_h / CASCADE_COUNT, 8) +
                         texture_bytes(vis_w, vis_h / CASCADE_COUNT, 4) +
                         texture_bytes(plane_w, ATLAS_ROWS, 1) + texture_bytes(plane_w, ATLAS_ROWS, 8);
        stats_.atlas_bytes += cs.atlas_bytes;
        stats_.probes += cs.probes;
    }
    state_mirror_.assign(static_cast<std::size_t>(plane_w) * static_cast<std::size_t>(plane_h),
                         PROBE_UNINITIALIZED);

    // Sized for the largest single-cascade turn at ANY quality, so switching
    // GiQuality mid-session never reallocates and never resizes a dispatch's
    // backing store while the previous frame is still reading it.
    ray_results_ = gpu::GPUBuffer::create_dynamic(
        device, gpu::GPUBuffer::Type::ComputeStorage,
        static_cast<std::size_t>(gi_max_rays_per_dispatch()) * sizeof(glm::vec4));
    counters_ = gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::ComputeStorage,
                                               sizeof(gpu::GiProbeCounters) * CASCADE_COUNT);
    if (!ray_results_ || !counters_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "ProbeVolume: could not allocate the trace buffers");
        shutdown();
        return false;
    }

    SDL_GPUSamplerCreateInfo sampler_info = {};
    sampler_info.min_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mag_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    sampler_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sampler_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sampler_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    linear_clamp_sampler_ = device.create_sampler(sampler_info);
    // The per-probe planes are one texel per probe: a linear tap on them blends
    // two unrelated probes' relocation offsets into a position that is neither.
    sampler_info.min_filter = SDL_GPU_FILTER_NEAREST;
    sampler_info.mag_filter = SDL_GPU_FILTER_NEAREST;
    point_clamp_sampler_ = device.create_sampler(sampler_info);
    if (linear_clamp_sampler_ == nullptr || point_clamp_sampler_ == nullptr) {
        BRACKEN_LOG_ERROR(LOG_TAG, "ProbeVolume: could not create the atlas samplers");
        shutdown();
        return false;
    }

    SDL_GPUTransferBufferCreateInfo transfer_info = {};
    transfer_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
    transfer_info.size = sizeof(gpu::GiProbeCounters) * CASCADE_COUNT;
    readback_transfer_ = SDL_CreateGPUTransferBuffer(device.handle(), &transfer_info);
    // A missing readback transfer costs the two diagnostic counters, nothing
    // else — do not fail the field over it.
    if (readback_transfer_ == nullptr) {
        BRACKEN_LOG_WARN(LOG_TAG, "ProbeVolume: no counter readback buffer; relocation stats stay 0");
    }

    // SEED THE OFFSET PLANE with PROBE_CLASS_INSIDE_GEOMETRY, once, before any
    // GPU pass has written a byte of it. This is what makes `offset.w` a
    // complete "do not use this probe" signal for R1c's samplers rather than
    // just a buried-probe verdict: a probe reads as unusable from allocation
    // until the classify pass — which only ever runs on probes the blend passes
    // filled in the same dispatch chain — replaces it with PROBE_CLASS_OK. It
    // is also what closes the last path to reading undefined texture memory,
    // which the never-clear-the-atlases policy leaves open the moment something
    // outside the update passes starts sampling them. See gi_math.hpp.
    //
    // Its own command buffer: init() has none, and this is a one-time 162 KB
    // upload rather than anything the frame should carry.
    {
        constexpr std::uint16_t HALF_ZERO = 0x0000; // 0.0h
        constexpr std::uint16_t HALF_TWO = 0x4000;  // 2.0h == PROBE_CLASS_INSIDE_GEOMETRY
        static_assert(PROBE_CLASS_INSIDE_GEOMETRY == 2.0f,
                      "HALF_TWO below encodes PROBE_CLASS_INSIDE_GEOMETRY literally");
        std::vector<std::uint16_t> seed(static_cast<std::size_t>(plane_w) *
                                            static_cast<std::size_t>(plane_h) * 4u,
                                        HALF_ZERO);
        for (std::size_t i = 3; i < seed.size(); i += 4) {
            seed[i] = HALF_TWO;
        }
        SDL_GPUCommandBuffer* seed_cmd = SDL_AcquireGPUCommandBuffer(device.handle());
        if (seed_cmd == nullptr) {
            BRACKEN_LOG_ERROR(LOG_TAG, "ProbeVolume: could not seed the probe offset plane");
            shutdown();
            return false;
        }
        offset_->upload(seed_cmd, seed.data(), plane_w, plane_h);
        SDL_SubmitGPUCommandBuffer(seed_cmd);
    }

    scratch_slots_.reserve(gi_max_probes_per_dispatch());
    scratch_records_.reserve(gi_max_probes_per_dispatch());

    stats_.cascades = static_cast<std::uint32_t>(CASCADE_COUNT);
    stats_.active = true;
    ready_ = true;
    frame_counter_ = 0;

    BRACKEN_LOG_INFO(LOG_TAG,
                     "probe field: {} cascades x {}x{}x{} probes (banded into one atlas each), "
                     "spacing {:.0f}..{:.0f} world units, irradiance {}x{} visibility {}x{}, {:.1f} MB",
                     CASCADE_COUNT, GRID_X, GRID_Y, GRID_Z, cascade_spacing(0),
                     cascade_spacing(CASCADE_COUNT - 1), irr_w, irr_h, vis_w, vis_h,
                     static_cast<double>(stats_.atlas_bytes) / (1024.0 * 1024.0));
    return true;
}

void ProbeVolume::shutdown() {
    if (device_ != nullptr) {
        if (readback_fence_ != nullptr) {
            // Wait rather than release blind: the fence guards a download into
            // memory we are about to free.
            SDL_WaitForGPUFences(device_->handle(), true, &readback_fence_, 1);
            SDL_ReleaseGPUFence(device_->handle(), readback_fence_);
            readback_fence_ = nullptr;
        }
        if (readback_transfer_ != nullptr) {
            SDL_ReleaseGPUTransferBuffer(device_->handle(), readback_transfer_);
            readback_transfer_ = nullptr;
        }
        if (linear_clamp_sampler_ != nullptr) {
            device_->release_sampler(linear_clamp_sampler_);
            linear_clamp_sampler_ = nullptr;
        }
        if (point_clamp_sampler_ != nullptr) {
            device_->release_sampler(point_clamp_sampler_);
            point_clamp_sampler_ = nullptr;
        }
    }
    readback_fence_ = nullptr;
    readback_transfer_ = nullptr;
    linear_clamp_sampler_ = nullptr;
    point_clamp_sampler_ = nullptr;

    irradiance_.reset();
    visibility_.reset();
    state_.reset();
    offset_.reset();
    state_mirror_.clear();
    for (Cascade& cascade : cascades_) {
        cascade.cpu.reset(cascade.cpu.cascade());
    }
    for (auto& list : update_lists_) {
        list.reset();
    }
    ray_results_.reset();
    counters_.reset();
    device_ = nullptr;
    ready_ = false;
    frame_counter_ = 0;
    stats_ = GiStats{};
}

void ProbeVolume::update(SDL_GPUCommandBuffer* cmd, const glm::vec3& camera_position) {
    if (!ready_) {
        return;
    }
    stats_.probes_invalidated = 0;
    bool any_dirty = false;
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        Cascade& cascade = cascades_[static_cast<std::size_t>(c)];
        const std::uint32_t invalidated = cascade.cpu.update(camera_position);

        GiCascadeStats& cs = stats_.per_cascade[static_cast<std::size_t>(c)];
        cs.base = cascade.cpu.base();
        cs.probes_invalidated = invalidated;
        stats_.probes_invalidated += invalidated;
        stats_.probes_invalidated_total += invalidated;
        any_dirty = any_dirty || cascade.cpu.state_dirty();
    }

    // The whole banded state plane is 20.25 KB, so ANY cascade going dirty
    // re-uploads all three bands rather than carrying a dirty-rect list. That
    // is a measured trade, not laziness: 20.25 KB was already R1a's worst case
    // (all three cascades scrolling in one frame), and against it a rect scheme
    // would have to describe an L-shaped slab — a diagonal scroll enters two
    // overlapping arms — i.e. two or three partial copies plus the code that
    // computes them, to save a few kilobytes on a bus that moves megabytes of
    // instance data every frame. The banded layout is also why it is one upload
    // and not three: `GPUTexture::upload` writes a whole texture.
    //
    // THE UPLOAD IS ALSO WHAT MAKES THE FIELD SELF-CONSISTENT, and the ordering
    // is load-bearing: trace() promotes a scheduled probe to PROBE_ACTIVE on
    // the CPU, but that reaches the GPU only here, on the NEXT frame — by which
    // time the blend passes have filled its tile. So a probe reads as ACTIVE to
    // the bounce sampler exactly one frame after its atlas texels were written,
    // and no pass ever samples a texel that no pass has written.
    if (cmd != nullptr && any_dirty && state_) {
        for (int c = 0; c < CASCADE_COUNT; ++c) {
            Cascade& cascade = cascades_[static_cast<std::size_t>(c)];
            const std::vector<std::uint8_t>& src = cascade.cpu.state();
            std::memcpy(state_mirror_.data() + static_cast<std::size_t>(c) * src.size(), src.data(),
                        src.size());
            cascade.cpu.clear_state_dirty();
        }
        state_->upload(cmd, state_mirror_.data(), PROBES_PER_ATLAS_ROW, ATLAS_ROWS_ALL_CASCADES);
    }
}

void ProbeVolume::trace(SDL_GPUCommandBuffer* cmd, const ProbeUpdateParams& params) {
    if (!ready_) {
        return;
    }
    stats_.rays_this_frame = 0;
    stats_.probes_updated = 0;
    for (GiCascadeStats& cs : stats_.per_cascade) {
        cs.probes_updated = 0;
    }
    poll_counter_readback();

    if (cmd == nullptr || params.pipelines == nullptr || params.scene == nullptr ||
        !params.scene->ready() || !params.scene->has_attributes()) {
        // GI needs the acceleration structure WITH attributes, like every other
        // traced feature. The caller builds it; if it is not there yet the
        // honest answer is to trace nothing this frame rather than to shade
        // hits with no material.
        return;
    }
    BRACKEN_PROFILE_ZONE("ProbeVolume::trace");

    const GiBudget budget = gi_budget(params.quality);

    // Zero the diagnostic counters before any classify dispatch adds to them,
    // so what comes back describes one frame's scheduled probes rather than an
    // unattributable running total. Its own copy pass, outside the compute
    // passes below.
    const std::array<gpu::GiProbeCounters, CASCADE_COUNT> zero{};
    counters_->update(cmd, zero.data(), sizeof(zero));

    for (int c = 0; c < CASCADE_COUNT; ++c) {
        if (!gi_cascade_updates_on_frame(c, frame_counter_)) {
            continue;
        }
        const std::uint32_t traced =
            trace_cascade(cmd, params, c, budget.probes_per_turn, budget.rays_per_probe);
        stats_.per_cascade[static_cast<std::size_t>(c)].probes_updated = traced;
        stats_.probes_updated += traced;
        stats_.rays_this_frame += traced * budget.rays_per_probe;
    }

    stats_.probes_updated_total += stats_.probes_updated;
    stats_.rays_total += stats_.rays_this_frame;

    if (frame_counter_ % COUNTER_READBACK_INTERVAL == 0) {
        request_counter_readback();
    }
    ++frame_counter_;
}

std::uint32_t ProbeVolume::trace_cascade(SDL_GPUCommandBuffer* cmd, const ProbeUpdateParams& params,
                                         int cascade_index, std::uint32_t probe_budget,
                                         std::uint32_t rays_per_probe) {
    Cascade& cascade = cascades_[static_cast<std::size_t>(cascade_index)];
    if (!cascade.cpu.placed()) {
        return 0;
    }

    scratch_slots_.clear();
    const std::uint32_t count = cascade.cpu.select_updates(probe_budget, scratch_slots_);
    if (count == 0) {
        return 0;
    }

    scratch_records_.clear();
    for (const ProbeUpdateSlot& slot : scratch_slots_) {
        gpu::GiProbeUpdateRecord record{};
        record.worldPosition = cascade.cpu.slot_world_position(slot.slot);
        record.slot = static_cast<std::uint32_t>(slot.slot);
        record.flags = slot.was_uninitialized ? gpu::GI_UPDATE_FLAG_UNINITIALIZED : 0u;
        scratch_records_.push_back(record);
    }

    gpu::GPUBuffer* list = update_lists_[static_cast<std::size_t>(cascade_index)].get();
    list->update(cmd, scratch_records_.data(),
                 scratch_records_.size() * sizeof(gpu::GiProbeUpdateRecord));

    const auto* trace_pipeline =
        params.pipelines->get_compute_pipeline(gpu::ComputePipelineType::GiProbeTrace);
    const auto* blend_pipeline =
        params.pipelines->get_compute_pipeline(gpu::ComputePipelineType::GiProbeBlend);
    const auto* visibility_pipeline =
        params.pipelines->get_compute_pipeline(gpu::ComputePipelineType::GiProbeBlendVisibility);
    const auto* classify_pipeline =
        params.pipelines->get_compute_pipeline(gpu::ComputePipelineType::GiProbeClassify);
    if (trace_pipeline == nullptr || blend_pipeline == nullptr || visibility_pipeline == nullptr ||
        classify_pipeline == nullptr) {
        return 0;
    }

    const float spacing = cascade_spacing(cascade_index);
    const float max_offset = PROBE_MAX_OFFSET_FRACTION * spacing;

    // ---- 1. trace -------------------------------------------------------
    {
        gpu::GiProbeTraceUniforms u{};
        u.shadow = params.shadow;
        // PCF, never PCSS: a probe ray's visibility is integrated over dozens
        // of updates by the EMA, so paying 20 atlas taps per ray to soften one
        // sample buys nothing the accumulation does not already give.
        // `shadowEnabled` of 0 still means "no shadows at all", which is
        // honoured — a host with shadows off gets an unshadowed field, exactly
        // as it gets an unshadowed image.
        if (u.shadow.shadowEnabled > 1.5f) {
            u.shadow.shadowEnabled = 1.0f;
        }
        u.cameraPosition = params.camera_position;
        u.cameraForward = params.camera_forward;
        u.rayMaxDistance = params.ray_max_distance;
        u.probeSpacing = spacing;
        u.maxOffset = max_offset;
        u.sunDirection = params.environment.sun_dir;
        u.sunColor = params.environment.sun_color * params.sun_intensity;
        u.normalBias = 1.0f;
        u.bounceScale = 1.0f;
        u.ambientSky = params.environment.ambient_sky;
        u.ambientGround = params.environment.ambient_ground;
        u.useSkyCube = params.sky_cube != nullptr ? 1.0f : 0.0f;
        for (int i = 0; i < CASCADE_COUNT; ++i) {
            const glm::ivec3 base = cascades_[static_cast<std::size_t>(i)].cpu.base();
            u.cascadeBase[i] = glm::ivec4(base, 0);
        }
        u.cascadeIndex = static_cast<std::uint32_t>(cascade_index);
        u.rayCount = rays_per_probe;
        u.probeCount = count;
        u.tlasNodeCount = params.scene->tlas_node_count();
        u.lightCount = params.light_count;
        u.lightCapacity = params.light_capacity;
        u.frameIndex = static_cast<std::uint32_t>(frame_counter_);

        // Fully rewritten every dispatch, so cycling costs nothing and spares
        // the driver a write-after-read barrier against the previous cascade.
        const auto rw = rw_buffer(ray_results_->handle(), true);
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {}, {&rw, 1});
        if (pass) {
            pass.bind_pipeline(*trace_pipeline);
            // ORDER IS THE BINDING CONTRACT (shaders/README.md): sampled
            // textures, then read-only storage textures, then read-only storage
            // buffers, each run numbered from 0 and matching the shader's
            // declaration order exactly. Banding cost this pass four bindings:
            // three per-cascade irradiance atlases and three state planes
            // collapsed into one of each.
            const std::array<SDL_GPUTextureSamplerBinding, 4> textures{{
                {params.shadow_atlas, params.shadow_compare_sampler},
                {params.shadow_atlas, params.shadow_raw_sampler},
                {params.sky_cube != nullptr ? params.sky_cube : params.sky_cube_fallback,
                 params.sky_sampler},
                {irradiance_->handle(), linear_clamp_sampler_},
            }};
            pass.bind_samplers(0, textures);
            const std::array<SDL_GPUTexture*, 2> storage_textures{offset_->handle(), state_->handle()};
            pass.bind_storage_textures(0, storage_textures);
            const std::array<SDL_GPUBuffer*, 7> buffers{
                params.scene->tlas_nodes(),    params.scene->tlas_instances(),
                params.scene->blas_nodes(),    params.scene->blas_triangles(),
                params.scene->triangle_attributes(), params.light_data,
                list->handle()};
            pass.bind_storage_buffers(0, buffers);
            pass.push_uniforms(0, &u, sizeof(u));
            pass.dispatch_for(*trace_pipeline, rays_per_probe, count);
        }
    }

    gpu::GiProbeBlendUniforms blend{};
    blend.probeCount = count;
    blend.rayCount = rays_per_probe;
    blend.distanceClamp = PROBE_VISIBILITY_DISTANCE_SCALE * spacing;
    blend.rayMaxDistance = params.ray_max_distance;
    blend.frameIndex = static_cast<std::uint32_t>(frame_counter_);
    blend.cascadeIndex = static_cast<std::uint32_t>(cascade_index);

    // ---- 2. irradiance blend + border -----------------------------------
    {
        // cycle = false: this is an accumulator. A fresh allocation would
        // discard every probe the rota did not touch this frame — i.e. all but
        // a quarter of the cascade.
        const auto rw = rw_texture(irradiance_->handle(), false);
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (pass) {
            pass.bind_pipeline(*blend_pipeline);
            const std::array<SDL_GPUTexture*, 1> storage_textures{offset_->handle()};
            pass.bind_storage_textures(0, storage_textures);
            const std::array<SDL_GPUBuffer*, 2> buffers{ray_results_->handle(), list->handle()};
            pass.bind_storage_buffers(0, buffers);
            pass.push_uniforms(0, &blend, sizeof(blend));
            // One threadgroup per probe: the shader's [numthreads] equals the
            // tile size, so covering (tile) x (tile * probes) elements lands
            // exactly one group on each probe's tile — which is what its
            // groupshared border fill requires.
            pass.dispatch_for(*blend_pipeline, static_cast<std::uint32_t>(IRRADIANCE_PROBE_TEXELS),
                              static_cast<std::uint32_t>(IRRADIANCE_PROBE_TEXELS) * count);
        }
    }

    // ---- 3. visibility moments + border ---------------------------------
    {
        const auto rw = rw_texture(visibility_->handle(), false);
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (pass) {
            pass.bind_pipeline(*visibility_pipeline);
            const std::array<SDL_GPUTexture*, 1> storage_textures{offset_->handle()};
            pass.bind_storage_textures(0, storage_textures);
            const std::array<SDL_GPUBuffer*, 2> buffers{ray_results_->handle(), list->handle()};
            pass.bind_storage_buffers(0, buffers);
            pass.push_uniforms(0, &blend, sizeof(blend));
            pass.dispatch_for(*visibility_pipeline, static_cast<std::uint32_t>(VISIBILITY_PROBE_TEXELS),
                              static_cast<std::uint32_t>(VISIBILITY_PROBE_TEXELS) * count);
        }
    }

    // ---- 4. classification + relocation ---------------------------------
    // AFTER the blends, deliberately: a probe's rays were traced from its OLD
    // sample position, so they are the right evidence for this update's
    // radiance and only a hint about where the probe should sit next time.
    {
        gpu::GiProbeClassifyUniforms u{};
        u.probeCount = count;
        u.rayCount = rays_per_probe;
        u.cascadeIndex = static_cast<std::uint32_t>(cascade_index);
        u.maxOffset = max_offset;
        u.probeSpacing = spacing;
        u.frameIndex = static_cast<std::uint32_t>(frame_counter_);

        const auto rw_tex = rw_texture(offset_->handle(), false);
        const auto rw_buf = rw_buffer(counters_->handle(), false);
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw_tex, 1}, {&rw_buf, 1});
        if (pass) {
            pass.bind_pipeline(*classify_pipeline);
            const std::array<SDL_GPUBuffer*, 2> buffers{ray_results_->handle(), list->handle()};
            pass.bind_storage_buffers(0, buffers);
            pass.push_uniforms(0, &u, sizeof(u));
            pass.dispatch_for(*classify_pipeline, count);
        }
    }

    return count;
}

void ProbeVolume::poll_counter_readback() {
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
    std::array<gpu::GiProbeCounters, CASCADE_COUNT> counters{};
    std::memcpy(counters.data(), mapped, sizeof(counters));
    device_->unmap_transfer_buffer(readback_transfer_);

    std::uint32_t relocated = 0;
    std::uint32_t inside = 0;
    for (const gpu::GiProbeCounters& c : counters) {
        relocated += c.relocated;
        inside += c.insideGeometry;
    }
    stats_.probes_relocated = relocated;
    stats_.probes_inside_geometry = inside;
}

void ProbeVolume::request_counter_readback() {
    if (device_ == nullptr || readback_transfer_ == nullptr || counters_ == nullptr ||
        readback_fence_ != nullptr) {
        return;
    }
    // Its own tiny command buffer, submitted before the frame's — so it reads
    // the PREVIOUS frame's counters. Fine for a diagnostic, and it is what
    // keeps the frame's command buffer free of a fence. Same shape as
    // render::AutoExposure's debug readback.
    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_->handle());
    if (cmd == nullptr) {
        return;
    }
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd);
    if (copy == nullptr) {
        SDL_CancelGPUCommandBuffer(cmd);
        return;
    }
    SDL_GPUBufferRegion src = {};
    src.buffer = counters_->handle();
    src.offset = 0;
    src.size = static_cast<Uint32>(sizeof(gpu::GiProbeCounters) * CASCADE_COUNT);
    SDL_GPUTransferBufferLocation dst = {};
    dst.transfer_buffer = readback_transfer_;
    dst.offset = 0;
    SDL_DownloadFromGPUBuffer(copy, &src, &dst);
    SDL_EndGPUCopyPass(copy);

    readback_fence_ = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
}

std::array<glm::ivec3, CASCADE_COUNT> ProbeVolume::cascade_bases() const {
    std::array<glm::ivec3, CASCADE_COUNT> bases{};
    for (int c = 0; c < CASCADE_COUNT; ++c) {
        bases[static_cast<std::size_t>(c)] = cascades_[static_cast<std::size_t>(c)].cpu.base();
    }
    return bases;
}

const ProbeCascade& ProbeVolume::cascade(int index) const {
    const int clamped = index < 0 ? 0 : (index >= CASCADE_COUNT ? CASCADE_COUNT - 1 : index);
    return cascades_[static_cast<std::size_t>(clamped)].cpu;
}

} // namespace bracken::render::gi
