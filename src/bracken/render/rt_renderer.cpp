#include "bracken/render/rt_renderer.hpp"

#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_debug.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/render/raytracing/emissive_clusters.hpp"
#include "bracken/render/rt_denoise_math.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace bracken::render {

RtRenderer::~RtRenderer() {
    shutdown();
}

bool RtRenderer::init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipelines) {
    device_ = &device;
    pipelines_ = &pipelines;
    SDL_GPUSamplerCreateInfo info = {};
    info.min_filter = SDL_GPU_FILTER_NEAREST;
    info.mag_filter = SDL_GPU_FILTER_NEAREST;
    info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    nearest_clamp_sampler_ = device.create_sampler(info);

    // The denoiser's history reprojection lands between texels essentially
    // always, so it needs a bilinear tap — a nearest one quantizes the motion
    // to whole texels and makes the accumulated shadow jitter in exactly the
    // way the denoiser exists to remove.
    info.min_filter = SDL_GPU_FILTER_LINEAR;
    info.mag_filter = SDL_GPU_FILTER_LINEAR;
    linear_clamp_sampler_ = device.create_sampler(info);

    // Four uint slots per signal in the shared diagnostics block. Assigned once,
    // here, so the two signals cannot be given the same base by a caller.
    sun_shadow_.counter_base = 0;
    local_shadow_.counter_base = 4;

    return nearest_clamp_sampler_ != nullptr && linear_clamp_sampler_ != nullptr;
}

// The traced-resolution targets only. The full-res mask survives, because it
// is what the forward shaders are still bound to this frame; only a resize
// (which reallocates it anyway) invalidates that.
void RtRenderer::VisibilityDenoiser::release_targets() {
    bits[0].reset();
    bits[1].reset();
    tile_class.reset();
    history[0].reset();
    history[1].reset();
    fast_history[0].reset();
    fast_history[1].reset();
    gradient[0].reset();
    gradient[1].reset();
    history_valid = false;
    gradient_valid = false;
}

void RtRenderer::VisibilityDenoiser::release() {
    release_targets();
    mask.reset();
}

void RtRenderer::RadianceDenoiser::release() {
    for (auto& tex : raw) {
        tex.reset();
    }
    for (auto& stage : history) {
        for (auto& tex : stage) {
            tex.reset();
        }
    }
    for (auto& tex : filtered) {
        tex.reset();
    }
    // The 1x1 stand-ins deliberately survive `release()`'s callers that only
    // mean "the resolution changed": they are size-independent and creating
    // them again would be work for nothing. shutdown() drops them explicitly.
    history_valid = false;
    width = 0;
    height = 0;
}

void RtRenderer::shutdown() {
    reflection_.reset();
    reflection_hit_distance_.reset();
    gi_dummy_irradiance_.reset();
    gi_dummy_probe_state_.reset();
    if (device_ != nullptr && nearest_clamp_sampler_ != nullptr) {
        device_->release_sampler(nearest_clamp_sampler_);
    }
    if (device_ != nullptr && linear_clamp_sampler_ != nullptr) {
        device_->release_sampler(linear_clamp_sampler_);
    }
    nearest_clamp_sampler_ = nullptr;
    linear_clamp_sampler_ = nullptr;
    sun_shadow_.release();
    local_shadow_.release();
    local_di_.release();
    local_di_.dummy[0].reset();
    local_di_.dummy[1].reset();
    local_di_ready_ = false;
    local_reservoirs_[0].reset();
    local_reservoirs_[1].reset();
    local_reservoir_capacity_ = 0;
    capsule_buffer_.reset();
    capsule_dummy_.reset();
    capsule_capacity_ = 0;
    capsule_count_ = 0;
    reflection_history_[0].reset();
    reflection_history_[1].reset();
    reflection_history_valid_ = false;
    denoise_counters_.reset();
    exposure_dummy_.reset();
    if (device_ != nullptr && counter_fence_ != nullptr) {
        // Wait rather than release: the fence guards a download into the
        // transfer buffer released on the next line.
        SDL_WaitForGPUFences(device_->handle(), true, &counter_fence_, 1);
        SDL_ReleaseGPUFence(device_->handle(), counter_fence_);
    }
    counter_fence_ = nullptr;
    if (device_ != nullptr && counter_transfer_ != nullptr) {
        device_->release_transfer_buffer(counter_transfer_);
    }
    counter_transfer_ = nullptr;
    counter_previous_valid_ = false;
    device_ = nullptr;
    pipelines_ = nullptr;
}

bool RtRenderer::ensure_target(gpu::GPUDevice& device, std::unique_ptr<gpu::GPUTexture>& tex,
                               int width, int height, SDL_GPUTextureFormat format,
                               const char* label) {
    if (width <= 0 || height <= 0) {
        return false;
    }
    if (tex && tex->width() == width && tex->height() == height) {
        return true;
    }
    // The packed-bits and tile-class targets are R32_UINT / R8_UINT, and SDL3
    // GPU forbids SAMPLER on an integer format — filtering an integer texel is
    // undefined, so `SDL_CreateGPUTexture` asserts and, under
    // BRACKEN_GPU_DEBUG=1, poisons the device for every creation after it.
    // Neither is ever sampled: the denoiser passes bind both as storage
    // textures and `Load()` them. Only the float history/mask targets are
    // sampled, so the flag follows the format rather than the call site.
    SDL_GPUTextureUsageFlags usage = SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                                     SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
    if (!gpu::GPUTexture::is_integer_format(format)) {
        usage |= SDL_GPU_TEXTUREUSAGE_SAMPLER;
    }
    tex = gpu::GPUTexture::create_2d(device, width, height, format, usage);
    if (!tex) {
        BRACKEN_LOG_ERROR("rt", "RtRenderer: could not create {} ({}x{})", label, width, height);
        return false;
    }
    return true;
}

// 1x1 stand-ins for the probe field, so the reflection pass's GI slots are
// always bound even with gi_mode Off. Created on first use rather than in
// init(): a game that never enables reflections should not pay for them, small
// as they are.
//
// `cmd` writes zeros into both. The state texel then reads
// GI_PROBE_UNINITIALIZED and the irradiance texel reads black, so a shader that
// ignored `giEnabled` would get no contribution rather than undefined memory —
// which is the same argument ProbeVolume::init() makes for seeding its whole
// offset plane.
bool RtRenderer::ensure_gi_dummies(SDL_GPUCommandBuffer* cmd) {
    if (gi_dummy_irradiance_ && gi_dummy_probe_state_) {
        return true;
    }
    if (device_ == nullptr || cmd == nullptr) {
        return false;
    }
    gi_dummy_irradiance_ = gpu::GPUTexture::create_2d(
        *device_, 1, 1, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, SDL_GPU_TEXTUREUSAGE_SAMPLER);
    // Storage-read only, no SAMPLER: it stands in for ProbeVolume's state
    // plane, which the reflection pass binds as a read-only storage texture and
    // `Load()`s — and R8_UINT with SAMPLER is an SDL assertion that poisons the
    // device for every later texture creation (gpu_texture.hpp).
    gi_dummy_probe_state_ = gpu::GPUTexture::create_2d(
        *device_, 1, 1, SDL_GPU_TEXTUREFORMAT_R8_UINT, SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ);
    if (!gi_dummy_irradiance_ || !gi_dummy_probe_state_) {
        BRACKEN_LOG_ERROR("rt", "RtRenderer: could not create the GI dummy textures");
        gi_dummy_irradiance_.reset();
        gi_dummy_probe_state_.reset();
        return false;
    }
    const std::array<uint16_t, 4> black{0, 0, 0, 0};
    gi_dummy_irradiance_->upload(cmd, black.data(), 1, 1);
    const uint8_t uninitialized = 0;
    gi_dummy_probe_state_->upload(cmd, &uninitialized, 1, 1);
    return true;
}

// The TWELVE monotone counters the two temporal dispatches and R7's DI noise
// probe accumulate into, and the 1x1 stand-in bound wherever a pass wants the
// exposure state and the caller has none. Created on first use rather than in
// init(): a game that never enables a traced visibility signal should not
// allocate either.
//
// Four per visibility signal, then three for the probe and one spare. The probe
// shares this buffer rather than owning one because it wants exactly the
// property this buffer already has — monotone accumulation differenced across a
// lagged readback — and a second buffer would mean a second fence, a second
// transfer and a second polling path for three uints.
bool RtRenderer::ensure_denoise_diagnostics() {
    if (device_ == nullptr) {
        return false;
    }
    if (!denoise_counters_) {
        const std::array<uint32_t, 12> zero{};
        denoise_counters_ = gpu::GPUBuffer::create_static(
            *device_, gpu::GPUBuffer::Type::ComputeStorage, zero.data(), sizeof(zero));
        if (!denoise_counters_) {
            BRACKEN_LOG_ERROR("rt", "RtRenderer: could not create the denoiser counter buffer");
            return false;
        }
    }
    if (!exposure_dummy_) {
        // w = 0 is AutoExposure's own "not adapted yet" flag, so a shader that
        // ignored `exposureEnabled` would fall back rather than read garbage.
        const std::array<float, 4> zero{0.0f, 0.0f, 0.0f, 0.0f};
        exposure_dummy_ = gpu::GPUBuffer::create_static(
            *device_, gpu::GPUBuffer::Type::ComputeStorage, zero.data(), sizeof(zero));
        if (!exposure_dummy_) {
            BRACKEN_LOG_ERROR("rt", "RtRenderer: could not create the exposure stand-in buffer");
            return false;
        }
    }
    if (counter_transfer_ == nullptr) {
        SDL_GPUTransferBufferCreateInfo info = {};
        info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
        info.size = 12 * sizeof(uint32_t);
        counter_transfer_ = device_->create_transfer_buffer(info);
    }
    return true;
}

// Poll the lagged readback; when one lands, DIFFERENCE it against the previous
// snapshot. The counters are never reset, so every reported figure is an
// average over the interval between two readbacks — and unsigned subtraction
// makes that exact even across a wraparound.
void RtRenderer::update_denoise_stats() {
    if (device_ == nullptr || !denoise_counters_ || counter_transfer_ == nullptr) {
        return;
    }

    if (counter_fence_ != nullptr) {
        if (!SDL_QueryGPUFence(device_->handle(), counter_fence_)) {
            return; // still in flight; a diagnostic never blocks the frame
        }
        SDL_ReleaseGPUFence(device_->handle(), counter_fence_);
        counter_fence_ = nullptr;

        std::array<uint32_t, 12> now{};
        if (void* mapped = device_->map_transfer_buffer(counter_transfer_, false)) {
            std::memcpy(now.data(), mapped, sizeof(now));
            device_->unmap_transfer_buffer(counter_transfer_);

            if (counter_previous_valid_) {
                const auto delta = [&](std::size_t i) {
                    return static_cast<uint32_t>(now[i] - counter_previous_[i]);
                };
                RtStats::DenoiseStats& d = stats_.denoise;
                const uint32_t sun_accum_px = delta(1);
                const uint32_t sun_px = delta(3);
                d.shadow_valid = sun_px > 0;
                d.shadow_mean_accum =
                    sun_accum_px > 0 ? static_cast<float>(delta(0)) / static_cast<float>(sun_accum_px)
                                     : 0.0f;
                d.shadow_gradient_fire =
                    sun_px > 0 ? static_cast<float>(delta(2)) / static_cast<float>(sun_px) : 0.0f;

                const uint32_t loc_accum_px = delta(5);
                const uint32_t loc_px = delta(7);
                d.local_valid = loc_px > 0;
                d.local_mean_accum =
                    loc_accum_px > 0 ? static_cast<float>(delta(4)) / static_cast<float>(loc_accum_px)
                                     : 0.0f;
                d.local_gradient_fire =
                    loc_px > 0 ? static_cast<float>(delta(6)) / static_cast<float>(loc_px) : 0.0f;

                // R7's probe. The shader accumulates fixed point at two
                // different scales — see rt_di_noise.comp.hlsl for why the delta
                // and the level cannot share one — so dividing by the texel
                // count and by the matching scale is what turns three uints back
                // into two radiance figures. `di_noise_valid` is false whenever
                // the probe did not run in the window, which is absence of
                // evidence and must not be reported as a noise floor of zero.
                constexpr float DI_NOISE_SCALE = 1024.0f;
                const uint32_t di_px = delta(10);
                d.di_noise_valid = di_px > 0;
                d.di_noise_mean_delta =
                    di_px > 0 ? static_cast<float>(delta(8)) /
                                    (static_cast<float>(di_px) * DI_NOISE_SCALE)
                              : 0.0f;
                d.di_noise_mean_level =
                    di_px > 0 ? static_cast<float>(delta(9)) /
                                    (static_cast<float>(di_px) * DI_NOISE_SCALE)
                              : 0.0f;
            }
            counter_previous_ = now;
            counter_previous_valid_ = true;
        }
    }

    if (counter_fence_ != nullptr || ++frames_since_counter_readback_ < COUNTER_READBACK_FRAMES) {
        return;
    }
    frames_since_counter_readback_ = 0;

    // Own tiny command buffer, submitted ahead of the frame's, so it reads the
    // totals as of the PREVIOUS frame — which is what "lagged" means here and
    // is fine for a statistic averaged over ten frames.
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
    src.buffer = denoise_counters_->handle();
    src.offset = 0;
    src.size = 12 * sizeof(uint32_t);
    SDL_GPUTransferBufferLocation dst = {};
    dst.transfer_buffer = counter_transfer_;
    dst.offset = 0;
    SDL_DownloadFromGPUBuffer(copy, &src, &dst);
    SDL_EndGPUCopyPass(copy);
    counter_fence_ = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
}

int RtRenderer::gradient_stratum_count(int traced) {
    return std::max(1, (traced + GRADIENT_STRATUM - 1) / GRADIENT_STRATUM);
}

bool RtRenderer::ensure_gradient_targets(VisibilityDenoiser& d, int traced_w, int traced_h,
                                         const char* label) {
    const int sx = gradient_stratum_count(traced_w);
    const int sy = gradient_stratum_count(traced_h);
    return ensure_target(*device_, d.fast_history[0], traced_w, traced_h,
                         SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT, label) &&
           ensure_target(*device_, d.fast_history[1], traced_w, traced_h,
                         SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT, label) &&
           ensure_target(*device_, d.gradient[0], sx, sy, SDL_GPU_TEXTUREFORMAT_R16_FLOAT, label) &&
           ensure_target(*device_, d.gradient[1], sx, sy, SDL_GPU_TEXTUREFORMAT_R16_FLOAT, label);
}

void RtRenderer::resize(gpu::GPUDevice& device, int width, int height) {
    screen_width_ = width;
    screen_height_ = height;
    // Targets are (re)created lazily in render(), where the resolution setting
    // is known — a game that never enables ray tracing should not allocate a
    // screen-sized target for it.
    sun_shadow_.release();
    local_shadow_.release();
    local_di_.release();
    local_di_ready_ = false;
    local_reservoirs_[0].reset();
    local_reservoirs_[1].reset();
    local_reservoir_capacity_ = 0;
    reflection_.reset();
    reflection_hit_distance_.reset();
    (void)device;
}

void RtRenderer::build_scene(gpu::GPUDevice& device, const ModelManager& models,
                             bool want_attributes) {
    scene_.build_blas_for_models(models, want_attributes);
    scene_.upload(device);
    built_with_attributes_ = want_attributes;
    publish_scene_stats();
}

void RtRenderer::set_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances,
                               SDL_GPUCommandBuffer* cmd, uint32_t skinned_first) {
    scene_.set_instances(instances, skinned_first);
    if (device_ != nullptr) {
        scene_.upload(*device_, cmd);
        publish_scene_stats();
    }
}

void RtRenderer::upload_shadow_capsules(SDL_GPUCopyPass* copy_pass,
                                        std::span<const rt::ShadowCapsule> capsules) {
    // The count is latched even on the paths that upload nothing, so a frame
    // that goes from "some capsules" to "none" actually clears the occluders
    // instead of tracing against last frame's pose forever.
    capsule_count_ = 0;
    if (device_ == nullptr || copy_pass == nullptr || capsules.empty()) {
        return;
    }

    // Grow-only, geometric, with a floor — the bone-palette buffer's pattern.
    // The floor is the frame budget itself: FramePacket::MAX_SHADOW_CAPSULES is
    // 256 records = 8 KB, which is small enough that sizing for the worst case
    // up front is cheaper than ever thinking about it again.
    const std::size_t required = capsules.size() * sizeof(rt::ShadowCapsule);
    if (!capsule_buffer_ || capsule_capacity_ < required) {
        capsule_capacity_ = std::max(required * 2, std::size_t{256} * sizeof(rt::ShadowCapsule));
        capsule_buffer_ = gpu::GPUBuffer::create_dynamic(
            *device_, gpu::GPUBuffer::Type::ComputeStorage, capsule_capacity_);
        if (!capsule_buffer_) {
            BRACKEN_LOG_ERROR("rt", "RtRenderer: could not create the shadow-capsule buffer ({} bytes)",
                              capsule_capacity_);
            capsule_capacity_ = 0;
            return;
        }
    }

    gpu::GPUBuffer::Mapping mapping = capsule_buffer_->map();
    if (!mapping) {
        BRACKEN_LOG_ERROR("rt", "RtRenderer: could not map the shadow-capsule transfer buffer");
        return;
    }
    std::memcpy(mapping.data(), capsules.data(), required);
    mapping.upload(copy_pass, required);
    capsule_count_ = static_cast<uint32_t>(capsules.size());
}

bool RtRenderer::ensure_capsule_dummy() {
    if (capsule_dummy_) {
        return true;
    }
    if (device_ == nullptr) {
        return false;
    }
    // One zeroed capsule. ComputeStorage for the same reason the real buffer is:
    // two of the four consumers bind it to a read-write slot.
    const rt::ShadowCapsule zero{};
    capsule_dummy_ = gpu::GPUBuffer::create_static(*device_, gpu::GPUBuffer::Type::ComputeStorage,
                                                   &zero, sizeof(zero));
    if (!capsule_dummy_) {
        BRACKEN_LOG_ERROR("rt", "RtRenderer: could not create the shadow-capsule stand-in buffer");
        return false;
    }
    return true;
}

SDL_GPUBuffer* RtRenderer::capsule_binding() {
    // A storage-buffer descriptor may never be empty, so a capsule-less frame
    // still binds SOMETHING; `shadowCapsuleCount == 0` is what guarantees it is
    // never indexed. Returning the stand-in whenever the real buffer is missing
    // (not only when the count is zero) also covers the failed-allocation path
    // above, which leaves the count at zero but may leave the pointer null.
    if (capsule_count_ > 0 && capsule_buffer_) {
        return capsule_buffer_->handle();
    }
    return ensure_capsule_dummy() ? capsule_dummy_->handle() : nullptr;
}

// One place that copies the structure's accounting out of RtScene, so the
// split figures and the total can never disagree about which upload they came
// from.
void RtRenderer::publish_scene_stats() {
    const rt::RtSceneStats& s = scene_.stats();
    stats_.instances = s.instances;
    stats_.tlas_builds = s.tlas_builds;
    stats_.tlas_refits = s.tlas_refits;
    stats_.blas_bytes = s.blas_bytes;
    stats_.tlas_bytes = s.tlas_bytes;
    stats_.attribute_bytes = s.attribute_bytes;
    stats_.emissive_bytes = s.emissive_bytes;
    stats_.acceleration_bytes = s.gpu_bytes;
    stats_.emissive_triangles = s.emissive_triangles;
    stats_.emissive_records = s.emissive_records;
    stats_.emissive_records_dropped = s.emissive_records_dropped;
}

void RtRenderer::render(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                        SDL_GPUTexture* normal_rough, SDL_GPUTexture* velocity,
                        const FrameParams& params,
                        const RtShadowSettings& shadows, const RtReflectionSettings& reflections,
                        const RtLocalShadowSettings& local_shadows,
                        const RtClusterBindings& clusters) {
    stats_.shadows_traced = false;
    stats_.reflections_traced = false;
    stats_.local_shadows_traced = false;
    stats_.local_di_traced = false;
    // Cleared here rather than only in the Full path: an early return below
    // (nothing enabled, no acceleration structure) must not leave last frame's
    // DI targets advertised as this frame's.
    local_di_ready_ = false;
    stats_.shadow_rays = 0;
    stats_.reflection_rays = 0;
    stats_.local_shadow_rays = 0;
    stats_.shadow_gradient_rays = 0;
    stats_.local_shadow_gradient_rays = 0;

    const bool want_any = shadows.enabled || reflections.enabled || local_shadows.enabled;
    if (!want_any || cmd == nullptr || device_ == nullptr || pipelines_ == nullptr) {
        return; // both off: not a single byte of work
    }
    if (!scene_.ready()) {
        return; // no acceleration structure yet
    }

    // The counter buffer and the exposure stand-in, plus the lagged readback
    // that turns the former into RtStats::denoise. Both are cheap and neither
    // blocks; failing to create them only costs the diagnostics.
    const bool diagnostics_ready = ensure_denoise_diagnostics();
    if (diagnostics_ready) {
        update_denoise_stats();
    }

    // ---- shadows ---------------------------------------------------------
    if (shadows.enabled) {
        BRACKEN_PROFILE_ZONE("RtRenderer::shadows");
        // Trace + the three denoiser dispatches, as one entry in the scene
        // renderer's RenderStats::pass_times_ms. The timer lives HERE rather
        // than at the call site because the call site makes one render() call
        // for both signals and cannot tell them apart. String LITERALS only:
        // PassTimeline interns the name by pointer.
        BRACKEN_GPU_TIMER(timers_, cmd, "RTShadows");
        const int div = rt_resolution_divisor(shadows.resolution);
        const int w = std::max(1, screen_width_ / div);
        const int h = std::max(1, screen_height_ / div);
        if (sun_shadow_.resolution != shadows.resolution) {
            // Every traced-resolution target changes shape, and the history
            // describes a different image; keeping it would smear the old
            // resolution across the first frames of the new one.
            sun_shadow_.release_targets();
            sun_shadow_.resolution = shadows.resolution;
        }

        // Tile grid over the traced resolution. 8x4 pixels per uint.
        const int tiles_x = (w + 7) / 8;
        const int tiles_y = (h + 3) / 4;

        const bool targets_ok =
            ensure_target(*device_, sun_shadow_.bits[0], tiles_x, tiles_y,
                          SDL_GPU_TEXTUREFORMAT_R32_UINT, "shadow bits 0") &&
            ensure_target(*device_, sun_shadow_.bits[1], tiles_x, tiles_y,
                          SDL_GPU_TEXTUREFORMAT_R32_UINT, "shadow bits 1") &&
            ensure_target(*device_, sun_shadow_.tile_class, tiles_x, tiles_y,
                          SDL_GPU_TEXTUREFORMAT_R8_UINT, "shadow tile class") &&
            ensure_target(*device_, sun_shadow_.history[0], w, h,
                          SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, "shadow history 0") &&
            ensure_target(*device_, sun_shadow_.history[1], w, h,
                          SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, "shadow history 1") &&
            ensure_gradient_targets(sun_shadow_, w, h, "shadow gradient/fast history") &&
            // The mask is FULL resolution now: the denoiser's last pass fuses
            // the upsample into the filter, so the forward shaders sample it
            // 1:1 instead of stretching a half-res binary signal bilinearly.
            ensure_target(*device_, sun_shadow_.mask, screen_width_, screen_height_,
                          SDL_GPU_TEXTUREFORMAT_R8_UNORM, "shadow mask");

        // The bits ping-pong: this frame writes the half the gradient pass is
        // not reading. Advanced only after the gradient has consumed the other.
        const int bits_next = 1 - sun_shadow_.bits_index;

        const gpu::GPUComputePipeline* pipeline =
            pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtShadow);
        if (targets_ok && pipeline != nullptr) {
            gpu::RtShadowUniforms u{};
            u.invViewProj = params.inv_view_proj;
            u.sunDirection = params.environment.sun_dir;
            u.rayMaxDistance = shadows.max_distance;
            u.cameraPosition = params.camera_position;
            u.normalBias = shadows.normal_bias;
            u.outputSize = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
            u.tlasNodeCount = scene_.tlas_node_count();
            u.depthBias = shadows.depth_bias;
            u.sunCosThetaMax =
                std::cos(glm::radians(std::max(shadows.sun_angular_radius_degrees, 0.0f)));
            u.frameIndex = sun_shadow_.frame_index;
            u.hasGBuffer = normal_rough != nullptr ? 1u : 0u;
            // §5.8. The gradient replay below reads the SAME latched count, so
            // the two dispatches cannot disagree about the occluder set.
            u.shadowCapsuleCount = capsule_count_;

            // cycle = false: the OTHER half of this ping-pong is the previous
            // frame's bits and the gradient pass is about to read it, so a
            // cycled allocation would hand this dispatch a fresh buffer and
            // leave the pair pointing at the wrong pair of images.
            const SDL_GPUStorageTextureReadWriteBinding rw{.texture = sun_shadow_.bits[bits_next]->handle(),
                                                           .mip_level = 0,
                                                           .layer = 0,
                                                           .cycle = false,
                                                           .padding1 = 0, .padding2 = 0, .padding3 = 0};
            gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
            if (pass) {
                pass.bind_pipeline(*pipeline);
                // Slot 1 must always be a valid texture even with no G-buffer;
                // hasGBuffer is what decides whether it is read, not the bind.
                const std::array<SDL_GPUTextureSamplerBinding, 2> textures{{
                    {depth.handle(), nearest_clamp_sampler_},
                    {normal_rough != nullptr ? normal_rough : depth.handle(),
                     nearest_clamp_sampler_},
                }};
                pass.bind_samplers(0, textures);
                // Four acceleration buffers, then the capsule set at t6 (§5.8).
                // This pass has read-only headroom; rt_local_shadow does not.
                const std::array<SDL_GPUBuffer*, 5> buffers{
                    scene_.tlas_nodes(), scene_.tlas_instances(), scene_.blas_nodes(),
                    scene_.blas_triangles(), capsule_binding()};
                pass.bind_storage_buffers(0, buffers);
                pass.push_uniforms(0, &u, sizeof(u));
                // One group PER TILE — the group is the 8x4 tile whose bits it
                // packs, so this dispatches over the tile grid, not the pixels.
                pass.dispatch(static_cast<uint32_t>(tiles_x), static_cast<uint32_t>(tiles_y), 1);
                stats_.shadows_traced = true;
                stats_.shadow_rays = static_cast<uint32_t>(w * h);
            }
        }

        if (stats_.shadows_traced) {
            // ---- the A-SVGF gradient, between the trace and the denoiser ----
            //
            // It reads the PREVIOUS frame's bits (`bits_index`, not
            // `bits_next`) and the previous frame's history, so it must run
            // before either is overwritten. Failure at any step leaves
            // `gradient_valid` false and the temporal pass falls back to
            // motion-only invalidation — a degradation, never a skipped frame.
            sun_shadow_.gradient_valid =
                trace_sun_gradient(cmd, sun_shadow_, depth, normal_rough, velocity, params, shadows,
                                   w, h) &&
                reconstruct_gradient(cmd, sun_shadow_, w, h);

            sun_shadow_.bits_index = bits_next;
            denoise_visibility(cmd, sun_shadow_, depth, normal_rough, velocity, params,
                               shadows.max_accum_frames, shadows.filter_radius, w, h);
            sun_shadow_.frame_index++;
        }
    } else {
        // Shadows off this frame: the next frame that turns them back on must
        // not reproject through a history describing an unknown number of
        // frames ago. The gradient goes with it — its whole premise is that the
        // retained bits are exactly one frame old.
        sun_shadow_.history_valid = false;
        sun_shadow_.gradient_valid = false;
    }

    // ---- local (point/spot) light shadows --------------------------------
    if (local_shadows.enabled) {
        BRACKEN_PROFILE_ZONE("RtRenderer::local_shadows");
        // Trace + the same three denoiser dispatches, as one entry in the
        // scene renderer's RenderStats::pass_times_ms. String LITERALS only:
        // PassTimeline interns the name by pointer.
        BRACKEN_GPU_TIMER(timers_, cmd, "RTLocalShadows");
        if (render_local_shadows(cmd, depth, normal_rough, velocity, params, local_shadows, clusters)) {
            local_shadow_.frame_index++;
        }
    } else {
        local_shadow_.history_valid = false;
        local_shadow_.gradient_valid = false;
        local_di_.history_valid = false;
        local_di_ready_ = false;
    }

    // ---- opaque reflections ---------------------------------------------
    reflection_denoised_this_frame_ = false;
    if (reflections.enabled) {
        // The trace + its temporal denoise. The full-res resolve is a separate
        // pass on the scene renderer's own timeline ("RTReflectionResolve").
        BRACKEN_GPU_TIMER(timers_, cmd, "RTReflectionTrace");
        if (trace_reflections(cmd, depth, normal_rough, params, reflections, nullptr)) {
            denoise_reflections(cmd, depth, normal_rough, velocity, params, reflections);
            reflection_frame_index_++;
        }
    } else {
        // Same reasoning as the shadow history: a gap of unknown length makes
        // the stored accumulation describe a different image.
        reflection_history_valid_ = false;
    }
}

// The denoiser, for EITHER stochastic visibility signal.
//
// Nothing below reads the sun, the light list, or anything else that would make
// it one signal's property: it takes packed bits at a traced resolution and
// produces a full-res R8 mask. That is why the local-light pass reuses these
// three pipelines rather than forking them — see VisibilityDenoiser's comment.
void RtRenderer::denoise_visibility(SDL_GPUCommandBuffer* cmd, VisibilityDenoiser& d,
                                    gpu::GPUTexture& depth, SDL_GPUTexture* normal_rough,
                                    SDL_GPUTexture* velocity, const FrameParams& params,
                                    float max_accum_frames, float filter_radius, int traced_w,
                                    int traced_h) {
    BRACKEN_PROFILE_ZONE("RtRenderer::denoise_visibility");

    const auto tiles_x = static_cast<uint32_t>((traced_w + 7) / 8);
    const auto tiles_y = static_cast<uint32_t>((traced_h + 3) / 4);
    const glm::uvec2 tile_count{tiles_x, tiles_y};
    const glm::uvec2 traced_size{static_cast<uint32_t>(traced_w), static_cast<uint32_t>(traced_h)};

    const gpu::GPUComputePipeline* classify =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtShadowClassify);
    const gpu::GPUComputePipeline* temporal =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtShadowTemporal);
    const gpu::GPUComputePipeline* filter =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtShadowFilter);
    // The counter buffer is a hard requirement of the temporal pass's binding
    // block, not of its arithmetic — but a descriptor may not be left empty, so
    // it joins the pipelines in the same precondition rather than being
    // silently skipped.
    if (classify == nullptr || temporal == nullptr || filter == nullptr || !denoise_counters_) {
        return;
    }

    // ---- 1. tile classification -----------------------------------------
    {
        gpu::RtShadowClassifyUniforms u{};
        u.tileCount = tile_count;

        const SDL_GPUStorageTextureReadWriteBinding rw{.texture = d.tile_class->handle(),
                                                       .mip_level = 0, .layer = 0, .cycle = true,
                                                       .padding1 = 0, .padding2 = 0, .padding3 = 0};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (pass) {
            pass.bind_pipeline(*classify);
            const std::array<SDL_GPUTexture*, 1> ro{d.bits[d.bits_index]->handle()};
            pass.bind_storage_textures(0, ro);
            pass.push_uniforms(0, &u, sizeof(u));
            pass.dispatch_for(*classify, tiles_x, tiles_y);
        }
    }

    // ---- 2. temporal accumulation ---------------------------------------
    const int prev = d.history_index;
    const int next = 1 - d.history_index;
    {
        gpu::RtShadowTemporalUniforms u{};
        u.tracedSize = traced_size;
        u.tileCount = tile_count;
        u.stratumCount = {static_cast<uint32_t>(gradient_stratum_count(traced_w)),
                          static_cast<uint32_t>(gradient_stratum_count(traced_h))};
        u.depthA = params.depth_a;
        u.depthB = params.depth_b;
        u.maxAccumFrames = std::max(max_accum_frames, 1.0f);
        u.depthRejectRelative = 0.05f;
        // Composed here rather than in the shader: the pass reconstructs no
        // world position for anything else, so one matrix and one `mul` per
        // pixel do what `invViewProj` + `prevViewProjUnjittered` would. This is
        // what makes the disocclusion test measure the SURFACE rather than the
        // camera — the same correction the gradient replays took, applied to the
        // accumulator that was left behind. See the shader's comment at the test.
        u.prevClipFromClip = params.prev_view_proj_unjittered * params.inv_view_proj;
        // The analytic sun gradient, measured on the main thread before any ray
        // was cast. 0 on every frame the sun did not move, which is what makes
        // the pre-R5b path bit-identical.
        u.lightingReset = lighting_reset_floor(params.lighting_reset);
        // No velocity buffer means no reprojection is possible; saying the
        // history is valid anyway would accumulate through camera motion and
        // smear the whole image.
        u.historyValid = (d.history_valid && velocity != nullptr) ? 1u : 0u;
        u.gradientValid = d.gradient_valid ? 1u : 0u;
        u.counterBase = d.counter_base;

        const std::array<SDL_GPUStorageTextureReadWriteBinding, 2> rw{{
            {.texture = d.history[next]->handle(), .mip_level = 0, .layer = 0, .cycle = true,
             .padding1 = 0, .padding2 = 0, .padding3 = 0},
            // cycle = false on the fast history for the same reason the
            // reservoirs use it: the other half of the pair is being read this
            // dispatch, and a cycled allocation would break the ping-pong.
            {.texture = d.fast_history[next]->handle(), .mip_level = 0, .layer = 0, .cycle = false,
             .padding1 = 0, .padding2 = 0, .padding3 = 0},
        }};
        const std::array<SDL_GPUStorageBufferReadWriteBinding, 1> rw_bufs{{
            {.buffer = denoise_counters_->handle(), .cycle = false,
             .padding1 = 0, .padding2 = 0, .padding3 = 0},
        }};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, rw, rw_bufs);
        if (pass) {
            pass.bind_pipeline(*temporal);
            const std::array<SDL_GPUTextureSamplerBinding, 4> textures{{
                {depth.handle(), nearest_clamp_sampler_},
                {velocity != nullptr ? velocity : depth.handle(), nearest_clamp_sampler_},
                {d.history[prev]->handle(), linear_clamp_sampler_},
                {d.fast_history[prev]->handle(), linear_clamp_sampler_},
            }};
            pass.bind_samplers(0, textures);
            // Slot 2 must hold a valid texture even with no gradient this
            // frame; `gradientValid` is what decides whether it is read.
            const std::array<SDL_GPUTexture*, 3> ro{d.bits[d.bits_index]->handle(),
                                                    d.tile_class->handle(),
                                                    d.gradient[1]->handle()};
            pass.bind_storage_textures(0, ro);
            pass.push_uniforms(0, &u, sizeof(u));
            pass.dispatch_for(*temporal, static_cast<uint32_t>(traced_w),
                              static_cast<uint32_t>(traced_h));
        }
    }

    // ---- 3. variance-guided filter, fused with the upsample --------------
    {
        gpu::RtShadowFilterUniforms u{};
        u.outputSize = {static_cast<uint32_t>(screen_width_), static_cast<uint32_t>(screen_height_)};
        u.tracedSize = traced_size;
        u.tileCount = tile_count;
        u.depthA = params.depth_a;
        u.depthB = params.depth_b;
        u.maxRadius = std::max(filter_radius, 0.0f);
        u.depthSigma = 0.05f;
        u.normalPower = 8.0f;
        u.hasGBuffer = normal_rough != nullptr ? 1u : 0u;

        const SDL_GPUStorageTextureReadWriteBinding rw{.texture = d.mask->handle(),
                                                       .mip_level = 0, .layer = 0, .cycle = true,
                                                       .padding1 = 0, .padding2 = 0, .padding3 = 0};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (pass) {
            pass.bind_pipeline(*filter);
            const std::array<SDL_GPUTextureSamplerBinding, 3> textures{{
                // Linear: the filter's uniform-tile path samples the history
                // at a sub-texel UV to upsample it continuously.
                {d.history[next]->handle(), linear_clamp_sampler_},
                {depth.handle(), nearest_clamp_sampler_},
                // NEAREST, and the slot holds a valid texture even with no
                // G-buffer: a filtered octahedral pair across a silhouette
                // decodes to a direction neither surface has, and `hasGBuffer`
                // is what decides whether it is decoded at all.
                {normal_rough != nullptr ? normal_rough : depth.handle(), nearest_clamp_sampler_},
            }};
            pass.bind_samplers(0, textures);
            const std::array<SDL_GPUTexture*, 1> ro{d.tile_class->handle()};
            pass.bind_storage_textures(0, ro);
            pass.push_uniforms(0, &u, sizeof(u));
            pass.dispatch_for(*filter, static_cast<uint32_t>(screen_width_),
                              static_cast<uint32_t>(screen_height_));
        }
    }

    d.history_index = next;
    d.history_valid = true;
}

// ---------------------------------------------------------------------------
// The A-SVGF sparse gradient (docs/rt_lighting_plan.md §5.6, R5b)
// ---------------------------------------------------------------------------
//
// Two traces, one reconstruction. Each trace is one thread per 3x3 stratum of
// the traced resolution — the 1-in-9 ray budget — and each RE-TRACES this
// frame's scene with the previous frame's sample, comparing against the
// previous frame's retained raw bit. The reconstruction is shared, exactly as
// the classify/temporal/filter three are: nothing in it is signal-specific.
//
// Both are recorded between the trace and the denoiser, and both read state the
// denoiser is about to overwrite — the other half of the bits ping-pong, and
// the history the temporal pass is about to replace. Ordering is therefore not
// a preference here; it is the contract.

bool RtRenderer::trace_sun_gradient(SDL_GPUCommandBuffer* cmd, VisibilityDenoiser& d,
                                    gpu::GPUTexture& depth, SDL_GPUTexture* normal_rough,
                                    SDL_GPUTexture* velocity, const FrameParams& params,
                                    const RtShadowSettings& shadows, int traced_w, int traced_h) {
    BRACKEN_PROFILE_ZONE("RtRenderer::sun_gradient");
    const gpu::GPUComputePipeline* pipeline =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtShadowGradient);
    if (pipeline == nullptr || !d.gradient[0]) {
        return false;
    }

    const int sx = gradient_stratum_count(traced_w);
    const int sy = gradient_stratum_count(traced_h);

    gpu::RtShadowGradientUniforms u{};
    u.invViewProj = params.inv_view_proj;
    u.sunDirection = params.environment.sun_dir;
    u.rayMaxDistance = shadows.max_distance;
    u.tracedSize = {static_cast<uint32_t>(traced_w), static_cast<uint32_t>(traced_h)};
    u.stratumCount = {static_cast<uint32_t>(sx), static_cast<uint32_t>(sy)};
    u.normalBias = shadows.normal_bias;
    u.depthBias = shadows.depth_bias;
    u.sunCosThetaMax = std::cos(glm::radians(std::max(shadows.sun_angular_radius_degrees, 0.0f)));
    u.frameIndex = d.frame_index;
    u.tlasNodeCount = scene_.tlas_node_count();
    u.hasGBuffer = normal_rough != nullptr ? 1u : 0u;
    // Replay needs BOTH a previous frame's bits and a previous frame's history
    // (for the depth it recorded). `history_valid` is cleared in every case that
    // invalidates either, and a null velocity buffer makes reprojection — and
    // therefore the whole comparison — impossible.
    u.historyValid = (d.history_valid && velocity != nullptr) ? 1u : 0u;
    u.depthA = params.depth_a;
    u.depthB = params.depth_b;
    u.depthRejectRelative = SUN_GRADIENT_DEPTH_REJECT_RELATIVE;
    // The replay's depth test asks where this world point WAS, so it needs the
    // camera it was seen with — not this frame's.
    u.prevViewProjUnjittered = params.prev_view_proj_unjittered;
    // Exactly what the trace used, one dispatch ago (§5.8).
    u.shadowCapsuleCount = capsule_count_;

    const SDL_GPUStorageTextureReadWriteBinding rw{.texture = d.gradient[0]->handle(),
                                                   .mip_level = 0, .layer = 0, .cycle = true,
                                                   .padding1 = 0, .padding2 = 0, .padding3 = 0};
    gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
    if (!pass) {
        return false;
    }
    pass.bind_pipeline(*pipeline);
    // NEAREST on the history: the replay compares one texel's recorded depth
    // against one texel's current depth, and a filtered tap would average two
    // surfaces across a silhouette and reject a valid comparison.
    const std::array<SDL_GPUTextureSamplerBinding, 4> textures{{
        {depth.handle(), nearest_clamp_sampler_},
        {normal_rough != nullptr ? normal_rough : depth.handle(), nearest_clamp_sampler_},
        {velocity != nullptr ? velocity : depth.handle(), nearest_clamp_sampler_},
        {d.history[d.history_index]->handle(), nearest_clamp_sampler_},
    }};
    pass.bind_samplers(0, textures);
    // The PREVIOUS frame's bits — `bits_index` still points at them here,
    // because render() advances it only after this returns.
    const std::array<SDL_GPUTexture*, 1> ro_tex{d.bits[d.bits_index]->handle()};
    pass.bind_storage_textures(0, ro_tex);
    // The capsule set as a fifth, and the SAME buffer the trace bound: a replay
    // that cannot see the trace's occluders fires a gradient on every
    // capsule-shadowed pixel, every frame, forever.
    const std::array<SDL_GPUBuffer*, 5> buffers{scene_.tlas_nodes(), scene_.tlas_instances(),
                                                scene_.blas_nodes(), scene_.blas_triangles(),
                                                capsule_binding()};
    pass.bind_storage_buffers(0, buffers);
    pass.push_uniforms(0, &u, sizeof(u));
    pass.dispatch_for(*pipeline, static_cast<uint32_t>(sx), static_cast<uint32_t>(sy));
    stats_.shadow_gradient_rays = static_cast<uint32_t>(sx * sy);
    return true;
}

bool RtRenderer::trace_local_gradient(SDL_GPUCommandBuffer* cmd, VisibilityDenoiser& d,
                                      gpu::GPUTexture& depth, SDL_GPUTexture* normal_rough,
                                      SDL_GPUTexture* velocity, const FrameParams& params,
                                      const RtLocalShadowSettings& settings,
                                      const RtClusterBindings& clusters, int prev_reservoir,
                                      int traced_w, int traced_h) {
    BRACKEN_PROFILE_ZONE("RtRenderer::local_gradient");
    const gpu::GPUComputePipeline* pipeline =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtLocalShadowGradient);
    if (pipeline == nullptr || !d.gradient[0] || !local_reservoirs_[prev_reservoir]) {
        return false;
    }

    const int sx = gradient_stratum_count(traced_w);
    const int sy = gradient_stratum_count(traced_h);

    gpu::RtLocalShadowGradientUniforms u{};
    u.invViewProj = params.inv_view_proj;
    u.tracedSize = {static_cast<uint32_t>(traced_w), static_cast<uint32_t>(traced_h)};
    u.stratumCount = {static_cast<uint32_t>(sx), static_cast<uint32_t>(sy)};
    u.normalBias = settings.normal_bias;
    u.depthBias = settings.depth_bias;
    u.lightSourceRadius = std::max(settings.source_radius, 0.0f);
    u.frameIndex = d.frame_index;
    u.tlasNodeCount = scene_.tlas_node_count();
    u.hasGBuffer = normal_rough != nullptr ? 1u : 0u;
    u.historyValid = (d.history_valid && velocity != nullptr) ? 1u : 0u;
    // Exactly what the trace resolved a dispatch ago — see the members.
    u.emissiveCount = local_emissive_count_;
    u.emissiveHistoryValid = local_emissive_history_valid_;
    u.shadowCapsuleCount = capsule_count_; // exactly what the trace used
    u.depthRejectRelative = LOCAL_GRADIENT_DEPTH_REJECT_RELATIVE;
    // The replay asks where this world point WAS, so it needs the camera it was
    // seen with. Passing this frame's made the test measure camera motion.
    u.prevView = params.prev_view;
    if (local_emissive_bound_ == nullptr && !ensure_emissive_dummy()) {
        return false;
    }
    SDL_GPUBuffer* capsule_buf = capsule_binding();
    if (capsule_buf == nullptr) {
        return false;
    }

    const SDL_GPUStorageTextureReadWriteBinding rw_tex{.texture = d.gradient[0]->handle(),
                                                       .mip_level = 0, .layer = 0, .cycle = true,
                                                       .padding1 = 0, .padding2 = 0, .padding3 = 0};
    // cycle = false: this is the previous frame's reservoir state, read only.
    // The emissive records ride the second slot for the same budget reason the
    // trace's third does, and are the SAME buffer the trace bound — a replay
    // aimed at a different record set is not a replay.
    // The capsule set rides a third slot for the same budget reason, and is the
    // SAME buffer the trace bound — a replay aimed at a different occluder set
    // is not a replay, and would fire a gradient on every capsule-shadowed
    // stratum forever.
    const std::array<SDL_GPUStorageBufferReadWriteBinding, 3> rw_bufs{{
        {.buffer = local_reservoirs_[prev_reservoir]->handle(), .cycle = false,
         .padding1 = 0, .padding2 = 0, .padding3 = 0},
        {.buffer = local_emissive_bound_ != nullptr ? local_emissive_bound_
                                                    : local_emissive_dummy_->handle(),
         .cycle = false, .padding1 = 0, .padding2 = 0, .padding3 = 0},
        {.buffer = capsule_buf, .cycle = false, .padding1 = 0, .padding2 = 0, .padding3 = 0},
    }};
    gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw_tex, 1}, rw_bufs);
    if (!pass) {
        return false;
    }
    pass.bind_pipeline(*pipeline);
    const std::array<SDL_GPUTextureSamplerBinding, 3> textures{{
        {depth.handle(), nearest_clamp_sampler_},
        {normal_rough != nullptr ? normal_rough : depth.handle(), nearest_clamp_sampler_},
        {velocity != nullptr ? velocity : depth.handle(), nearest_clamp_sampler_},
    }};
    pass.bind_samplers(0, textures);
    const std::array<SDL_GPUTexture*, 1> ro_tex{d.bits[d.bits_index]->handle()};
    pass.bind_storage_textures(0, ro_tex);
    // Same eight the trace binds, in the same order — four acceleration, four
    // cluster. See rt_local_shadow.comp.hlsl's binding note for the ceiling.
    const std::array<SDL_GPUBuffer*, 8> buffers{
        scene_.tlas_nodes(),     scene_.tlas_instances(),  scene_.blas_nodes(),
        scene_.blas_triangles(), clusters.light_data,      clusters.cluster_offsets,
        clusters.light_indices,  clusters.light_remap};
    pass.bind_storage_buffers(0, buffers);
    pass.push_uniforms(0, &u, sizeof(u));
    pass.push_uniforms(1, clusters.params, clusters.params_size);
    pass.dispatch_for(*pipeline, static_cast<uint32_t>(sx), static_cast<uint32_t>(sy));
    stats_.local_shadow_gradient_rays = static_cast<uint32_t>(sx * sy);
    return true;
}

bool RtRenderer::reconstruct_gradient(SDL_GPUCommandBuffer* cmd, VisibilityDenoiser& d,
                                      int traced_w, int traced_h) {
    const gpu::GPUComputePipeline* pipeline =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtGradientReconstruct);
    if (pipeline == nullptr || !d.gradient[0] || !d.gradient[1]) {
        return false;
    }
    const int sx = gradient_stratum_count(traced_w);
    const int sy = gradient_stratum_count(traced_h);

    gpu::RtGradientReconstructUniforms u{};
    u.stratumCount = {static_cast<uint32_t>(sx), static_cast<uint32_t>(sy)};

    const SDL_GPUStorageTextureReadWriteBinding rw{.texture = d.gradient[1]->handle(),
                                                   .mip_level = 0, .layer = 0, .cycle = true,
                                                   .padding1 = 0, .padding2 = 0, .padding3 = 0};
    gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
    if (!pass) {
        return false;
    }
    pass.bind_pipeline(*pipeline);
    const std::array<SDL_GPUTexture*, 1> ro{d.gradient[0]->handle()};
    pass.bind_storage_textures(0, ro);
    pass.push_uniforms(0, &u, sizeof(u));
    pass.dispatch_for(*pipeline, static_cast<uint32_t>(sx), static_cast<uint32_t>(sy));
    return true;
}

// Froxel-RIS local-light shadows (docs/rt_lighting_plan.md §5.3, Ratio tier).
//
// One dispatch, then the shared denoiser. The dispatch is over the TILE grid
// rather than the pixel grid — the shader's group IS the 8x4 tile whose bits it
// packs, exactly as the sun trace's is.
// The Full-DI tier's six RGBA16F targets, plus the two 1x1 stand-ins the Ratio
// path binds into the trace's DI slots (R5c, docs/rt_lighting_plan.md §5.3).
//
// The usage flags are the interesting part and they differ per role:
//  - `raw` and `history` never leave the compute stage, so compute storage
//    read/write plus SAMPLER (the temporal pass reprojects its history with a
//    bilinear tap, which needs a real sampler). Since R7c `raw` is ALSO the
//    a-trous scratch the first spatial iteration writes and the second reads,
//    which needs no usage the trace did not already give it;
//  - `filtered` is written by compute and read by the FRAGMENT stage as a
//    read-only storage texture, so it carries GRAPHICS_STORAGE_READ instead of
//    SAMPLER — there is no seventeenth fragment sampler to bind it to, which is
//    the whole reason this signal is a storage texture downstream.
bool RtRenderer::ensure_di_dummies() {
    if (local_di_.dummy[0] && local_di_.dummy[1] && local_di_.dummy[2]) {
        return true;
    }
    if (device_ == nullptr) {
        return false;
    }
    // TYPED TO MATCH, and that is not decoration: a storage image is declared
    // in the shader with a channel count, and binding an RGBA one where the
    // shader says RG is undefined rather than merely wasteful. R7b's direction
    // slot is RG16F, so its stand-in is too.
    for (int i = 0; i < 3; ++i) {
        local_di_.dummy[i] = gpu::GPUTexture::create_2d(
            *device_, 1, 1,
            i == 2 ? SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT : SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
            SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE);
        if (!local_di_.dummy[i]) {
            BRACKEN_LOG_ERROR("rt", "RtRenderer: could not create the local-DI stand-in textures");
            return false;
        }
    }
    return true;
}

bool RtRenderer::ensure_emissive_dummy() {
    if (local_emissive_dummy_) {
        return true;
    }
    if (device_ == nullptr) {
        return false;
    }
    // One zeroed record. ComputeStorage because it binds to a READ-WRITE slot,
    // for the reason the record buffer itself does (rt_local_shadow.comp.hlsl's
    // eight read-only slots were spent before R6 existed).
    const rt::EmissiveTriangle zero{};
    local_emissive_dummy_ = gpu::GPUBuffer::create_static(
        *device_, gpu::GPUBuffer::Type::ComputeStorage, &zero, sizeof(zero));
    if (!local_emissive_dummy_) {
        BRACKEN_LOG_ERROR("rt", "RtRenderer: could not create the emissive-record stand-in buffer");
        return false;
    }
    return true;
}

bool RtRenderer::ensure_emissive_froxel_dummy() {
    if (local_emissive_froxel_dummy_) {
        return true;
    }
    if (device_ == nullptr) {
        return false;
    }
    // One zeroed froxel block (R8). Its own buffer rather than a reuse of the
    // record stand-in above, because the two slots declare different element
    // types and a stand-in that only works by nothing ever indexing it is the
    // kind of thing that stops being true quietly.
    const std::array<uint32_t, rt::k_emissive_froxel_stride> zero{};
    local_emissive_froxel_dummy_ = gpu::GPUBuffer::create_static(
        *device_, gpu::GPUBuffer::Type::ComputeStorage, zero.data(), sizeof(zero));
    if (!local_emissive_froxel_dummy_) {
        BRACKEN_LOG_ERROR("rt", "RtRenderer: could not create the emissive froxel-list stand-in buffer");
        return false;
    }
    return true;
}

bool RtRenderer::ensure_di_targets(int traced_w, int traced_h) {
    if (device_ == nullptr || traced_w <= 0 || traced_h <= 0) {
        return false;
    }
    if (local_di_.width == traced_w && local_di_.height == traced_h && local_di_.filtered[2]) {
        return true;
    }

    local_di_.release();
    const SDL_GPUTextureUsageFlags internal_usage = SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                                                    SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE |
                                                    SDL_GPU_TEXTUREUSAGE_SAMPLER;
    const SDL_GPUTextureUsageFlags output_usage = SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                                                  SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE |
                                                  SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ;
    bool ok = true;
    for (int i = 0; i < 3; ++i) {
        // Signal 2 is R7b's DIRECTION: two channels, not four. An octahedral
        // unit vector needs exactly two, and at half res the difference is a
        // megabyte per copy across four copies.
        const SDL_GPUTextureFormat format = i == 2 ? SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT
                                                   : SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
        local_di_.raw[i] =
            gpu::GPUTexture::create_2d(*device_, traced_w, traced_h, format, internal_usage);
        local_di_.filtered[i] =
            gpu::GPUTexture::create_2d(*device_, traced_w, traced_h, format, output_usage);
        ok = ok && local_di_.raw[i] != nullptr && local_di_.filtered[i] != nullptr;
        for (int p = 0; p < 2; ++p) {
            local_di_.history[p][i] =
                gpu::GPUTexture::create_2d(*device_, traced_w, traced_h, format, internal_usage);
            ok = ok && local_di_.history[p][i] != nullptr;
        }
    }
    if (!ok) {
        BRACKEN_LOG_ERROR("rt", "RtRenderer: could not create the local-DI targets ({}x{})", traced_w,
                          traced_h);
        local_di_.release();
        return false;
    }
    local_di_.width = traced_w;
    local_di_.height = traced_h;
    // A history at a different size describes a different image, and the new
    // allocations are uninitialized besides.
    local_di_.history_valid = false;
    return true;
}

// The DI denoiser chain: one temporal dispatch, then one or two a-trous spatial
// iterations, then (only when asked) R7's noise probe. Deliberately NOT folded
// into denoise_visibility: that
// function turns packed bits into an R8 mask through three shared pipelines and
// nothing in it is sun-specific, which is exactly why the local VISIBILITY
// signal could reuse it verbatim in R2b. A radiance estimate shares none of
// that machinery — no bit packing, no tile classification, no second moment —
// so reusing it would have meant a flag through every one of those passes.
void RtRenderer::denoise_local_di(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                                  SDL_GPUTexture* normal_rough, SDL_GPUTexture* velocity,
                                  const FrameParams& params,
                                  const RtLocalShadowSettings& settings, int traced_w,
                                  int traced_h) {
    BRACKEN_PROFILE_ZONE("RtRenderer::denoise_local_di");
    local_di_ready_ = false;

    const gpu::GPUComputePipeline* temporal =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtDiTemporal);
    const gpu::GPUComputePipeline* spatial =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtDiSpatial);
    if (temporal == nullptr || spatial == nullptr || !local_shadow_.gradient[1]) {
        return;
    }

    const auto groups_x = static_cast<uint32_t>((traced_w + 7) / 8);
    const auto groups_y = static_cast<uint32_t>((traced_h + 7) / 8);
    const int hist_in = local_di_.history_index;
    const int hist_out = 1 - local_di_.history_index;

    // ---- temporal --------------------------------------------------------
    {
        gpu::RtDiTemporalUniforms u{};
        u.tracedSize = {static_cast<uint32_t>(traced_w), static_cast<uint32_t>(traced_h)};
        u.stratumCount = {static_cast<uint32_t>(gradient_stratum_count(traced_w)),
                          static_cast<uint32_t>(gradient_stratum_count(traced_h))};
        u.maxAccumFrames = std::max(settings.di_max_accum_frames, 1.0f);
        u.clampSigma = std::max(settings.di_clamp_sigma, 0.0f);
        u.lightingReset = params.lighting_reset;
        // No velocity buffer means no reprojection is possible, and claiming
        // otherwise would carry every texel straight through camera motion.
        // No velocity buffer means no reprojection is possible, and claiming
        // otherwise would carry every texel straight through camera motion.
        u.historyValid = (local_di_.history_valid && velocity != nullptr) ? 1u : 0u;
        // The LOCAL signal's gradient, shared with its visibility denoiser: a
        // reservoir that switched light is a lighting change for both.
        u.gradientValid = (local_shadow_.gradient_valid && local_shadow_.gradient[1]) ? 1u : 0u;
        u.exposureEnabled = params.exposure_state != nullptr ? 1u : 0u;
        u.manualExposureEv = params.manual_exposure_ev;

        const std::array<SDL_GPUStorageTextureReadWriteBinding, 3> rw{{
            {.texture = local_di_.history[hist_out][0]->handle(), .mip_level = 0, .layer = 0,
             .cycle = false, .padding1 = 0, .padding2 = 0, .padding3 = 0},
            {.texture = local_di_.history[hist_out][1]->handle(), .mip_level = 0, .layer = 0,
             .cycle = false, .padding1 = 0, .padding2 = 0, .padding3 = 0},
            {.texture = local_di_.history[hist_out][2]->handle(), .mip_level = 0, .layer = 0,
             .cycle = false, .padding1 = 0, .padding2 = 0, .padding3 = 0},
        }};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, rw, {});
        if (!pass) {
            return;
        }
        pass.bind_pipeline(*temporal);
        const std::array<SDL_GPUTextureSamplerBinding, 7> textures{{
            {local_di_.raw[0]->handle(), nearest_clamp_sampler_},
            {local_di_.raw[1]->handle(), nearest_clamp_sampler_},
            {local_di_.raw[2]->handle(), nearest_clamp_sampler_},
            // Slot 3 must hold a valid texture even with no velocity buffer;
            // historyValid is what decides whether it is read.
            {velocity != nullptr ? velocity : depth.handle(), linear_clamp_sampler_},
            {local_di_.history[hist_in][0]->handle(), linear_clamp_sampler_},
            {local_di_.history[hist_in][1]->handle(), linear_clamp_sampler_},
            // NEAREST for the direction history, unlike the two radiance ones:
            // a bilinear tap on octahedral coordinates crosses the fold seam at
            // exactly the silhouettes it is meant to preserve.
            {local_di_.history[hist_in][2]->handle(), nearest_clamp_sampler_},
        }};
        pass.bind_samplers(0, textures);
        // Always the gradient target, never a stand-in of another format: it is
        // R16F and this slot is a typed storage image, so an RGBA16F filler
        // would be a format mismatch rather than merely unread data. It is
        // allocated whenever this function can be reached (ensure_gradient_
        // targets ran in render_local_shadows), and `gradientValid` is what
        // decides whether its contents mean anything.
        SDL_GPUTexture* gradient = local_shadow_.gradient[1]->handle();
        pass.bind_storage_textures(0, {&gradient, 1});
        SDL_GPUBuffer* exposure =
            params.exposure_state != nullptr ? params.exposure_state : exposure_dummy_->handle();
        pass.bind_storage_buffers(0, {&exposure, 1});
        pass.push_uniforms(0, &u, sizeof(u));
        pass.dispatch(groups_x, groups_y, 1);
        pass.end();
    }

    // ---- spatial: one or two a-trous iterations --------------------------
    //
    // The intermediate target is `raw`, which the temporal dispatch above has
    // already consumed and which nothing reads again until next frame's trace
    // overwrites it. Reusing it is what makes the second iteration cost no
    // memory at all — the alternative was three more RGBA16F/RG16F targets at
    // traced resolution to hold a value that lives for one dispatch.
    const int iterations =
        std::clamp(static_cast<int>(settings.di_spatial_iterations), 1, 2);
    for (int it = 0; it < iterations; ++it) {
        const bool final_iteration = it == iterations - 1;
        gpu::RtDiSpatialUniforms u{};
        u.tracedSize = {static_cast<uint32_t>(traced_w), static_cast<uint32_t>(traced_h)};
        u.depthA = params.depth_a;
        u.depthB = params.depth_b;
        u.maxRadius = std::max(settings.di_filter_radius, 0.0f);
        // Step doubling: 1, then 2. The kernel keeps its tap count and doubles
        // its footprint, which is the whole reason a second iteration is worth
        // dispatching instead of widening the first.
        u.stepSize = static_cast<uint32_t>(1 << it);
        // Only the last iteration may convert the diffuse alpha from the
        // accumulation count to the view depth — the next iteration needs the
        // count to size its radius and to make its converged-pixel decision.
        u.isFinal = final_iteration ? 1u : 0u;

        const auto& dst = final_iteration ? local_di_.filtered : local_di_.raw;
        const auto& src = it == 0 ? local_di_.history[hist_out] : local_di_.raw;

        const std::array<SDL_GPUStorageTextureReadWriteBinding, 3> rw{{
            {.texture = dst[0]->handle(), .mip_level = 0, .layer = 0,
             .cycle = false, .padding1 = 0, .padding2 = 0, .padding3 = 0},
            {.texture = dst[1]->handle(), .mip_level = 0, .layer = 0,
             .cycle = false, .padding1 = 0, .padding2 = 0, .padding3 = 0},
            {.texture = dst[2]->handle(), .mip_level = 0, .layer = 0,
             .cycle = false, .padding1 = 0, .padding2 = 0, .padding3 = 0},
        }};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, rw, {});
        if (!pass) {
            return;
        }
        pass.bind_pipeline(*spatial);
        const std::array<SDL_GPUTextureSamplerBinding, 5> textures{{
            {src[0]->handle(), nearest_clamp_sampler_},
            {src[1]->handle(), nearest_clamp_sampler_},
            {src[2]->handle(), nearest_clamp_sampler_},
            {depth.handle(), nearest_clamp_sampler_},
            {normal_rough != nullptr ? normal_rough : depth.handle(), nearest_clamp_sampler_},
        }};
        pass.bind_samplers(0, textures);
        pass.push_uniforms(0, &u, sizeof(u));
        pass.dispatch(groups_x, groups_y, 1);
        pass.end();
    }

    local_di_.history_index = hist_out;
    local_di_.history_valid = true;
    local_di_ready_ = true;

    // ---- R7's temporal-noise probe, only when asked ----------------------
    if (settings.di_noise_probe) {
        probe_di_noise(cmd, traced_w, traced_h);
    } else {
        // The stored scalar is now a measurement of some frame that is not the
        // previous one. Saying so is cheaper than clearing the texture, and
        // reporting a delta against it would be a fabricated number.
        di_noise_history_valid_ = false;
    }
}

// R7's temporal-noise probe. One dispatch over the traced resolution, comparing
// the FILTERED targets against last frame's scalar and accumulating into the
// shared denoiser counters. See rt_di_noise.comp.hlsl for why the metric is what
// it is and why it does not measure the composited frame.
void RtRenderer::probe_di_noise(SDL_GPUCommandBuffer* cmd, int traced_w, int traced_h) {
    BRACKEN_PROFILE_ZONE("RtRenderer::probe_di_noise");
    const gpu::GPUComputePipeline* probe =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtDiNoise);
    if (probe == nullptr || !denoise_counters_) {
        return;
    }
    if (!di_noise_prev_ || di_noise_prev_->width() != traced_w ||
        di_noise_prev_->height() != traced_h) {
        di_noise_prev_ = gpu::GPUTexture::create_2d(*device_, traced_w, traced_h,
                                                    SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT,
                                                    SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                                                        SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE);
        // A fresh allocation holds nothing, and a resized one held a different
        // image. Either way this frame only establishes the baseline.
        di_noise_history_valid_ = false;
        if (!di_noise_prev_) {
            BRACKEN_LOG_ERROR("rt", "RtRenderer: could not create the DI noise probe target");
            return;
        }
    }

    gpu::RtDiNoiseUniforms u{};
    u.tracedSize = {static_cast<uint32_t>(traced_w), static_cast<uint32_t>(traced_h)};
    u.counterBase = 8;
    u.historyValid = di_noise_history_valid_ ? 1u : 0u;

    const std::array<SDL_GPUStorageTextureReadWriteBinding, 1> rw_tex{{
        {.texture = di_noise_prev_->handle(), .mip_level = 0, .layer = 0, .cycle = false,
         .padding1 = 0, .padding2 = 0, .padding3 = 0},
    }};
    const std::array<SDL_GPUStorageBufferReadWriteBinding, 1> rw_bufs{{
        {.buffer = denoise_counters_->handle(), .cycle = false,
         .padding1 = 0, .padding2 = 0, .padding3 = 0},
    }};
    gpu::ComputePass pass = gpu::ComputePass::begin(cmd, rw_tex, rw_bufs);
    if (!pass) {
        return;
    }
    pass.bind_pipeline(*probe);
    const std::array<SDL_GPUTextureSamplerBinding, 2> textures{{
        {local_di_.filtered[0]->handle(), nearest_clamp_sampler_},
        {local_di_.filtered[1]->handle(), nearest_clamp_sampler_},
    }};
    pass.bind_samplers(0, textures);
    pass.push_uniforms(0, &u, sizeof(u));
    pass.dispatch(static_cast<uint32_t>((traced_w + 7) / 8),
                  static_cast<uint32_t>((traced_h + 7) / 8), 1);
    pass.end();
    di_noise_history_valid_ = true;
}

bool RtRenderer::render_local_shadows(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                                      SDL_GPUTexture* normal_rough, SDL_GPUTexture* velocity,
                                      const FrameParams& params,
                                      const RtLocalShadowSettings& settings,
                                      const RtClusterBindings& clusters) {
    // A scene lit only by emissive meshes has zero BINNED analytic lights and
    // is not dark, so `complete()` (which requires total_lights > 0) is the
    // wrong gate for the Full tier. `emissiveCount` is not resolved until the
    // DI targets are allocated further down, so this is deliberately the
    // CONSERVATIVE predicate: if those targets then fail to allocate, the
    // emissive count resolves to 0, the shader's `shaded` is false everywhere
    // and the pass writes the identity mask — one wasted frame, never a wrong
    // one.
    const bool emissive_possible = settings.full_di && settings.emissive_candidate_count > 0 &&
                                   scene_.emissive_record_count() > 0 &&
                                   scene_.emissive_records() != nullptr &&
                                   scene_.emissive_total_power() > 0.0f;
    if (!clusters.has_grid() || (clusters.total_lights == 0 && !emissive_possible)) {
        // No binned lights this frame, or the caller had no cluster buffers to
        // give. Not an error and not worth a log line every frame: a scene with
        // no local lights has nothing for this pass to shadow. But the history
        // must go, because the next frame that DOES have lights would otherwise
        // remap reservoirs against a frame that is not its predecessor.
        local_shadow_.history_valid = false;
        return false;
    }
    if (normal_rough == nullptr) {
        // The G-buffer is what supplies the surface normal, and the normal is
        // what the target function weights candidates by. Reconstructing one
        // from depth taps is good enough for a binary sun test and is not good
        // enough to CHOOSE A LIGHT with, so this is skipped rather than
        // degraded — the atlas is already retired, so the frame simply shows
        // unshadowed local lights instead of wrongly shadowed ones.
        local_shadow_.history_valid = false;
        return false;
    }

    const int div = rt_resolution_divisor(settings.resolution);
    const int w = std::max(1, screen_width_ / div);
    const int h = std::max(1, screen_height_ / div);
    if (local_shadow_.resolution != settings.resolution) {
        local_shadow_.release_targets();
        local_shadow_.resolution = settings.resolution;
    }

    const int tiles_x = (w + 7) / 8;
    const int tiles_y = (h + 3) / 4;

    const bool targets_ok =
        ensure_target(*device_, local_shadow_.bits[0], tiles_x, tiles_y,
                      SDL_GPU_TEXTUREFORMAT_R32_UINT, "local shadow bits 0") &&
        ensure_target(*device_, local_shadow_.bits[1], tiles_x, tiles_y,
                      SDL_GPU_TEXTUREFORMAT_R32_UINT, "local shadow bits 1") &&
        ensure_target(*device_, local_shadow_.tile_class, tiles_x, tiles_y,
                      SDL_GPU_TEXTUREFORMAT_R8_UINT, "local shadow tile class") &&
        ensure_target(*device_, local_shadow_.history[0], w, h,
                      SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, "local shadow history 0") &&
        ensure_target(*device_, local_shadow_.history[1], w, h,
                      SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, "local shadow history 1") &&
        ensure_gradient_targets(local_shadow_, w, h, "local shadow gradient/fast history") &&
        ensure_target(*device_, local_shadow_.mask, screen_width_, screen_height_,
                      SDL_GPU_TEXTUREFORMAT_R8_UNORM, "local shadow mask");
    if (!targets_ok) {
        return false;
    }

    // The reservoir ping-pong, plus R7a's stage-0 scratch. ComputeStorage rather
    // than Storage: each of them is bound read-only to one dispatch and
    // read-write to another, so all three need both usages.
    //
    // THREE BUFFERS, NOT TWO, and the third is not a ping-pong member: stage 0
    // writes it and stage 1 reads it, both within one frame. The pair keeps
    // meaning exactly "last frame's final" and "this frame's final", which is
    // what the temporal merge and the gradient replay both depend on.
    const int reservoir_count = w * h;
    if (local_reservoir_capacity_ != reservoir_count) {
        const auto bytes = static_cast<std::size_t>(reservoir_count) * 16u;
        local_reservoirs_[0] =
            gpu::GPUBuffer::create_dynamic(*device_, gpu::GPUBuffer::Type::ComputeStorage, bytes);
        local_reservoirs_[1] =
            gpu::GPUBuffer::create_dynamic(*device_, gpu::GPUBuffer::Type::ComputeStorage, bytes);
        local_reservoir_temp_ =
            gpu::GPUBuffer::create_dynamic(*device_, gpu::GPUBuffer::Type::ComputeStorage, bytes);
        local_reservoir_capacity_ =
            (local_reservoirs_[0] && local_reservoirs_[1] && local_reservoir_temp_) ? reservoir_count
                                                                                   : 0;
        // Reservoir N describes a different pixel at a different size, and the
        // buffers are uninitialized besides.
        local_shadow_.history_valid = false;
    }
    if (local_reservoir_capacity_ != reservoir_count) {
        BRACKEN_LOG_ERROR("rt", "RtRenderer: could not create the local-shadow reservoirs ({}x{})", w, h);
        return false;
    }

    const gpu::GPUComputePipeline* pipeline =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtLocalShadow);
    if (pipeline == nullptr) {
        return false;
    }

    const int prev = local_reservoir_index_;
    const int next = 1 - local_reservoir_index_;

    gpu::RtLocalShadowUniforms u{};
    u.invViewProj = params.inv_view_proj;
    u.outputSize = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
    u.tlasNodeCount = scene_.tlas_node_count();
    u.normalBias = settings.normal_bias;
    u.depthBias = settings.depth_bias;
    u.lightSourceRadius = std::max(settings.source_radius, 0.0f);
    u.historyMaxMRatio = std::max(settings.history_max_m_ratio, 1.0f);
    u.frameIndex = local_shadow_.frame_index;
    u.hasGBuffer = 1u;
    // Same rule as the denoiser's: no velocity buffer means no reprojection is
    // possible, and claiming otherwise would carry every reservoir straight
    // through camera motion.
    u.historyValid = (local_shadow_.history_valid && velocity != nullptr) ? 1u : 0u;
    u.candidateCount = std::max(settings.candidate_count, 1u);
    // The Full-DI tier, if its targets came up. A failed allocation degrades to
    // Ratio for the frame rather than dropping the pass: the reservoir, the ray
    // and the visibility mask are all still correct, and the forward shaders
    // read `local_di_ready_` to decide which tier they are in.
    if (!ensure_di_dummies()) {
        return false; // the trace's DI slots have to hold SOMETHING in either tier
    }
    const bool want_di = settings.full_di && ensure_di_targets(w, h);
    if (!want_di) {
        // Ratio (or a failed DI allocation): the next frame that turns Full
        // back on must not accumulate against a history from an unknown number
        // of frames ago.
        local_di_.history_valid = false;
        local_di_ready_ = false;
    }
    u.diMode = want_di ? 2u : 0u;
    u.cameraPosition = params.camera_position;

    // R6, emissive mesh lights. FULL TIER ONLY, and the gate is here rather
    // than in the shader so the Ratio path is not merely equal but byte-for-byte
    // the R5c one: with `emissiveCount` 0 every RIS weight in the pass reduces
    // to its pre-R6 expression exactly. rt_local_shadow.comp.hlsl's header
    // carries the argument for why a ratio estimator cannot represent a light
    // the forward loop does not evaluate.
    const uint32_t emissive_count =
        (want_di && settings.emissive_candidate_count > 0) ? scene_.emissive_record_count() : 0u;
    const bool emissive_live = emissive_count > 0 && scene_.emissive_records() != nullptr &&
                               scene_.emissive_total_power() > 0.0f;
    u.emissiveCount = emissive_live ? emissive_count : 0u;
    u.emissiveTotalPower = scene_.emissive_total_power();
    u.emissiveCandidateCount = emissive_live ? settings.emissive_candidate_count : 0u;
    // A record's index is stable only within one topology epoch. A rebuild
    // renumbers them, so an emissive index carried over from last frame names a
    // different triangle and the history must be dropped — which is correct, not
    // a gap: reusing it would shadow against an unrelated emitter, the exact
    // failure the light remap prevents on the analytic side.
    const uint32_t epoch = scene_.emissive_epoch();
    u.emissiveHistoryValid = (emissive_live && local_emissive_epoch_ == epoch) ? 1u : 0u;
    local_emissive_epoch_ = epoch;
    local_emissive_count_ = u.emissiveCount;
    u.shadowCapsuleCount = capsule_count_; // §5.8; the replay reads the same latch
    // R7a. Both tiers get the spatial reuse — see the field's own comment for
    // why it is not gated to Full. 0 makes stage 1's merge the identity.
    u.spatialNeighbors = settings.spatial_neighbors;
    u.spatialRadius = std::max(settings.spatial_radius, 1.0f);
    local_emissive_history_valid_ = u.emissiveHistoryValid;
    stats_.emissive_di_sampled = emissive_live;

    // R8, the per-froxel emissive lists. `emissiveFroxelShare` is the whole
    // switch: 0 collapses the shader's mixture to R6's global alias draw bit for
    // bit, and it is what a frame with no table (binning skipped, allocation
    // failed, the caller's A/B lever at 0) sends. There is deliberately no
    // fallback that draws through a table which is not THIS frame's — a stale
    // list names records that may have been renumbered.
    if (!ensure_emissive_froxel_dummy()) {
        return false;
    }
    const bool froxel_lists_live = emissive_live && clusters.emissive_froxels != nullptr &&
                                   clusters.emissive_froxel_entries > 0 &&
                                   settings.emissive_froxel_share > 0.0f;
    u.emissiveFroxelShare = froxel_lists_live ? std::clamp(settings.emissive_froxel_share, 0.0f, 1.0f) : 0.0f;
    SDL_GPUBuffer* emissive_froxel_buf = froxel_lists_live ? clusters.emissive_froxels
                                                           : local_emissive_froxel_dummy_->handle();
    stats_.emissive_froxel_local = froxel_lists_live;
    stats_.emissive_froxels_filled = clusters.emissive_froxels_filled;
    stats_.emissive_froxel_entries = clusters.emissive_froxel_entries;
    stats_.emissive_froxel_dropped = clusters.emissive_froxel_dropped;

    // The bits ping-pong (R5b): this frame writes the half the gradient pass is
    // not reading. cycle = false for exactly the reason the reservoirs use it.
    const int bits_next = 1 - local_shadow_.bits_index;
    // Three read-write storage textures: the packed bits, and the Full-DI
    // tier's two demodulated targets. In Ratio mode the latter two are the 1x1
    // stand-ins — a storage-image descriptor may not be empty, and `diMode` is
    // what stops the shader writing through them.
    const std::array<SDL_GPUStorageTextureReadWriteBinding, 4> rw_tex{{
        {.texture = local_shadow_.bits[bits_next]->handle(),
         .mip_level = 0, .layer = 0, .cycle = false,
         .padding1 = 0, .padding2 = 0, .padding3 = 0},
        {.texture = want_di ? local_di_.raw[0]->handle() : local_di_.dummy[0]->handle(),
         .mip_level = 0, .layer = 0, .cycle = false,
         .padding1 = 0, .padding2 = 0, .padding3 = 0},
        {.texture = want_di ? local_di_.raw[1]->handle() : local_di_.dummy[1]->handle(),
         .mip_level = 0, .layer = 0, .cycle = false,
         .padding1 = 0, .padding2 = 0, .padding3 = 0},
        {.texture = want_di ? local_di_.raw[2]->handle() : local_di_.dummy[2]->handle(),
         .mip_level = 0, .layer = 0, .cycle = false,
         .padding1 = 0, .padding2 = 0, .padding3 = 0},
    }};
    // TWO reservoir buffers bind here per dispatch, and the second is only ever
    // READ. SDL3 GPU allows a compute pipeline eight read-only storage buffers
    // and the eight above are spoken for (four acceleration, four cluster), so
    // the reservoirs being read ride a read-write slot instead of a read-only
    // one. Which two buffers they are depends on the R7a stage — see the loop
    // below.
    //
    // cycle = false on both: this is persistent state the NEXT frame reads
    // back, and a cycled allocation would hand the dispatch a fresh, undefined
    // buffer while leaving the one the ping-pong points at untouched.
    //
    // The third is R6's emissive record buffer, read-only in the shader and in
    // a read-write slot for the same budget reason. A scene with no emitter
    // binds the 1-element stand-in — a storage-buffer descriptor may not be
    // empty — and `emissiveCount = 0` is what stops anything indexing it.
    //
    // The FOURTH is §5.8's capsule set, read-only in the shader and in a
    // read-write slot for the third time and the same budget reason. Its
    // `capsule_binding()` substitutes the 1-element stand-in on a frame with no
    // capsules, exactly as the emissive dummy does.
    //
    // The FIFTH is R8's per-froxel emissive candidate table, same story again:
    // read-only, read-write slot, 1-block stand-in when the table is not this
    // frame's — with `emissiveFroxelShare = 0` stopping the shader indexing it.
    if (!ensure_emissive_dummy()) {
        return false;
    }
    SDL_GPUBuffer* emissive_buf =
        emissive_live ? scene_.emissive_records() : local_emissive_dummy_->handle();
    local_emissive_bound_ = emissive_buf;
    SDL_GPUBuffer* capsule_buf = capsule_binding();
    if (capsule_buf == nullptr) {
        return false;
    }
    const std::array<SDL_GPUTextureSamplerBinding, 3> textures{{
        {depth.handle(), nearest_clamp_sampler_},
        {normal_rough, nearest_clamp_sampler_},
        // Slot 2 must always be a valid texture; historyValid is what decides
        // whether it is read, not the bind.
        {velocity != nullptr ? velocity : depth.handle(), nearest_clamp_sampler_},
    }};
    // Binding ORDER is the shader's declaration order: the four acceleration
    // buffers, then the four cluster_compute.hlsli claims. The packed stable
    // ids are the fifth thing that header CAN declare and this pass does not
    // ask for them — see the catalogue row for why eight is the whole budget.
    const std::array<SDL_GPUBuffer*, 8> buffers{
        scene_.tlas_nodes(),     scene_.tlas_instances(),  scene_.blas_nodes(),
        scene_.blas_triangles(), clusters.light_data,      clusters.cluster_offsets,
        clusters.light_indices,  clusters.light_remap};

    // ---- R7a: two dispatches of one pipeline ----------------------------
    //
    // Stage 0 RESAMPLES (froxel candidates + the temporal merge) from LAST
    // frame's final reservoirs into the scratch; stage 1 REUSES the scratch
    // spatially, traces the ray at the surviving winner and writes the DI
    // targets into THIS frame's final. The bindings are identical — only the
    // two reservoir slots and one uniform change — which is exactly why this is
    // one pipeline. rt_local_shadow.comp.hlsl's header carries the argument for
    // why a spatial pass cannot live inside a single dispatch, and for why the
    // ray had to move behind the reuse.
    //
    // The four read-write TEXTURES are bound to both dispatches even though
    // stage 0 writes none of them: a storage-image descriptor may not be empty,
    // and `stage` is what stops the writes. That costs nothing — the images are
    // the same ones stage 1 will write.
    struct StageBinding {
        uint32_t stage;
        SDL_GPUBuffer* write;
        SDL_GPUBuffer* read;
    };
    const std::array<StageBinding, 2> stages{{
        {0u, local_reservoir_temp_->handle(), local_reservoirs_[prev]->handle()},
        {1u, local_reservoirs_[next]->handle(), local_reservoir_temp_->handle()},
    }};

    for (const StageBinding& sb : stages) {
        u.stage = sb.stage;
        const std::array<SDL_GPUStorageBufferReadWriteBinding, 5> rw_bufs{{
            {.buffer = sb.write, .cycle = false,
             .padding1 = 0, .padding2 = 0, .padding3 = 0},
            {.buffer = sb.read, .cycle = false,
             .padding1 = 0, .padding2 = 0, .padding3 = 0},
            {.buffer = emissive_buf, .cycle = false,
             .padding1 = 0, .padding2 = 0, .padding3 = 0},
            {.buffer = capsule_buf, .cycle = false,
             .padding1 = 0, .padding2 = 0, .padding3 = 0},
            {.buffer = emissive_froxel_buf, .cycle = false,
             .padding1 = 0, .padding2 = 0, .padding3 = 0},
        }};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, rw_tex, rw_bufs);
        if (!pass) {
            return false;
        }
        pass.bind_pipeline(*pipeline);
        pass.bind_samplers(0, textures);
        pass.bind_storage_buffers(0, buffers);
        pass.push_uniforms(0, &u, sizeof(u));
        pass.push_uniforms(1, clusters.params, clusters.params_size);
        pass.dispatch(static_cast<uint32_t>(tiles_x), static_cast<uint32_t>(tiles_y), 1);
        pass.end();
    }

    stats_.local_shadows_traced = true;
    stats_.local_shadow_rays = static_cast<uint32_t>(w * h);
    stats_.local_di_traced = want_di;

    // The gradient replays against `prev`'s reservoirs and the retained bits,
    // so it must run before either index advances.
    local_shadow_.gradient_valid =
        trace_local_gradient(cmd, local_shadow_, depth, normal_rough, velocity, params, settings,
                             clusters, prev, w, h) &&
        reconstruct_gradient(cmd, local_shadow_, w, h);

    local_reservoir_index_ = next;
    local_shadow_.bits_index = bits_next;

    denoise_visibility(cmd, local_shadow_, depth, normal_rough, velocity, params,
                       settings.max_accum_frames, settings.filter_radius, w, h);

    // The radiance denoiser last, and after denoise_visibility rather than
    // before it, because it reads the LOCAL signal's reconstructed gradient —
    // which trace_local_gradient + reconstruct_gradient produced above and
    // which denoise_visibility only consumes.
    if (want_di) {
        denoise_local_di(cmd, depth, normal_rough, velocity, params, settings, w, h);
        stats_.local_di_traced = local_di_ready_;
    }
    return true;
}

void RtRenderer::denoise_reflections(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                                     SDL_GPUTexture* normal_rough, SDL_GPUTexture* velocity,
                                     const FrameParams& params,
                                     const RtReflectionSettings& reflections) {
    BRACKEN_PROFILE_ZONE("RtRenderer::denoise_reflections");
    // The exposure stand-in is a binding requirement, not an arithmetic one:
    // slot 0 of this pass's read-only storage buffers must hold something even
    // when the caller has no auto exposure to offer.
    if (reflection_ == nullptr || !exposure_dummy_) {
        return;
    }
    const int w = reflection_->width();
    const int h = reflection_->height();

    if (!ensure_target(*device_, reflection_history_[0], w, h,
                       SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, "reflection history 0") ||
        !ensure_target(*device_, reflection_history_[1], w, h,
                       SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, "reflection history 1")) {
        return;
    }

    const gpu::GPUComputePipeline* pipeline =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtReflectionTemporal);
    if (pipeline == nullptr) {
        return;
    }

    const int prev = reflection_history_index_;
    const int next = 1 - reflection_history_index_;

    // R3's virtual-hit-point reprojection needs all three of the hit distance,
    // the depth buffer and the G-buffer to place the reflected point. Any one
    // missing and the pass falls back to pure surface motion, which is exactly
    // what it did before R3 — a degradation, not a skip.
    const bool virtual_ready = reflection_hit_distance_ != nullptr && normal_rough != nullptr;

    gpu::RtReflectionTemporalUniforms u{};
    u.invViewProj = params.inv_view_proj;
    u.viewProjUnjittered = params.view_proj_unjittered;
    u.prevViewProjUnjittered = params.prev_view_proj_unjittered;
    u.cameraPosition = params.camera_position;
    u.rayMaxDistance = reflections.max_distance;
    u.tracedSize = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
    u.maxAccumFrames = std::max(reflections.max_accum_frames, 1.0f);
    u.clampSigma = std::max(reflections.clamp_sigma, 0.0f);
    u.historyValid = (reflection_history_valid_ && velocity != nullptr) ? 1u : 0u;
    u.virtualValid = virtual_ready ? 1u : 0u;
    // R5b: the sigma floor and the new firefly ceiling are absolute
    // POST-EXPOSURE luminances, so they have to be told which exposure. With no
    // exposure buffer the pass binds this renderer's 1x1 stand-in and reads the
    // manual bias instead, so the two configurations clamp identically in
    // perceptual terms rather than one of them silently clamping in scene units.
    const bool exposure_ready = params.exposure_state != nullptr && exposure_dummy_ != nullptr;
    u.exposureEnabled = exposure_ready ? 1u : 0u;
    u.manualExposureEv = params.manual_exposure_ev;

    const SDL_GPUStorageTextureReadWriteBinding rw{.texture = reflection_history_[next]->handle(),
                                                   .mip_level = 0, .layer = 0, .cycle = true,
                                                   .padding1 = 0, .padding2 = 0, .padding3 = 0};
    gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
    if (!pass) {
        return;
    }
    pass.bind_pipeline(*pipeline);
    // Slots 3-5 must be valid textures even when `virtualValid` is 0; the flag,
    // not the bind, is what decides whether they are read.
    const std::array<SDL_GPUTextureSamplerBinding, 6> textures{{
        {reflection_->handle(), nearest_clamp_sampler_},
        {velocity != nullptr ? velocity : reflection_->handle(), nearest_clamp_sampler_},
        {reflection_history_[prev]->handle(), linear_clamp_sampler_},
        {virtual_ready ? reflection_hit_distance_->handle() : reflection_->handle(),
         nearest_clamp_sampler_},
        {depth.handle(), nearest_clamp_sampler_},
        {virtual_ready ? normal_rough : depth.handle(), nearest_clamp_sampler_},
    }};
    pass.bind_samplers(0, textures);
    const std::array<SDL_GPUBuffer*, 1> buffers{
        exposure_ready ? params.exposure_state : exposure_dummy_->handle()};
    pass.bind_storage_buffers(0, buffers);
    pass.push_uniforms(0, &u, sizeof(u));
    pass.dispatch_for(*pipeline, static_cast<uint32_t>(w), static_cast<uint32_t>(h));

    reflection_history_index_ = next;
    reflection_history_valid_ = true;
    reflection_denoised_this_frame_ = true;
}

bool RtRenderer::trace_reflections(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                                   SDL_GPUTexture* receiver_data, const FrameParams& params,
                                   const RtReflectionSettings& reflections,
                                   const RtWaterReceiver* water) {
    BRACKEN_PROFILE_ZONE("RtRenderer::reflections");
    if (cmd == nullptr || device_ == nullptr || pipelines_ == nullptr || receiver_data == nullptr ||
        !scene_.ready()) {
        return false;
    }
    if (!scene_.has_attributes()) {
        // Attributes are gathered at build time; enabling reflections
        // afterwards cannot conjure them. Say so once rather than rendering
        // black reflections forever.
        if (!warned_missing_attributes_) {
            BRACKEN_LOG_ERROR("rt",
                              "Reflections are enabled but the acceleration structure was "
                              "built without triangle attributes. Rebuild with "
                              "build_scene(..., want_attributes=true).");
            warned_missing_attributes_ = true;
        }
        return false;
    }

    const int div = rt_resolution_divisor(reflections.resolution);
    const int w = std::max(1, screen_width_ / div);
    const int h = std::max(1, screen_height_ / div);
    if (reflection_res_ != reflections.resolution) {
        reflection_.reset();
        reflection_hit_distance_.reset();
        reflection_res_ = reflections.resolution;
    }
    if (!ensure_target(*device_, reflection_, w, h, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                       "reflection target")) {
        return false;
    }
    if (!ensure_target(*device_, reflection_hit_distance_, w, h, SDL_GPU_TEXTUREFORMAT_R16_FLOAT,
                       "reflection hit distance")) {
        return false;
    }
    const gpu::GPUComputePipeline* pipeline =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtReflection);
    if (pipeline == nullptr) {
        return false;
    }

    const RtHitShadeBindings& hit = params.hit_shade;
    const bool gi_ready = hit.gi_ready();
    if (!gi_ready && !ensure_gi_dummies(cmd)) {
        return false;
    }

    gpu::RtReflectionUniforms u{};
    u.shadow = hit.shadow;
    // PCF, never PCSS: a reflection is temporally accumulated over up to
    // max_accum_frames, so paying 20 atlas taps per ray to soften ONE sample
    // buys nothing the accumulation does not already give. The same clamp
    // ProbeVolume::trace applies, for the same reason. `shadowEnabled` of 0
    // still means "no shadows at all", which is honoured — a host with shadows
    // off gets unshadowed reflections, exactly as it gets an unshadowed image.
    if (u.shadow.shadowEnabled > 1.5f) {
        u.shadow.shadowEnabled = 1.0f;
    }
    u.invViewProj = params.inv_view_proj;
    u.cameraPosition = params.camera_position;
    u.cameraForward = params.camera_forward;
    u.rayMaxDistance = reflections.max_distance;
    u.sunDirection = params.environment.sun_dir;
    u.maxRoughness = reflections.max_roughness;
    u.gateFadeStart = reflections.fade_start;
    // WITH sun_intensity. Without it a reflected facade is three times darker
    // than the same facade drawn directly — invisible before R3 only because
    // the missing Lambertian 1/pi cancelled it at the playground's value of 3.
    u.sunColor = params.environment.sun_color * params.sun_intensity;
    u.normalBias = reflections.normal_bias;
    u.ambientSky = params.environment.ambient_sky;
    u.ambientGround = params.environment.ambient_ground;
    u.useSkyCube = hit.sky_cube != nullptr ? 1.0f : 0.0f;
    u.bounceScale = 1.0f;
    u.giEnabled = gi_ready ? 1.0f : 0.0f;
    // Always: a reflected point the field does not cover has to be lit by the
    // hemisphere rather than by nothing. Probe rays pass 0 — see rtHitAmbient.
    u.giAmbientFallback = 1.0f;
    static_assert(std::size(gpu::RtReflectionUniforms{}.giCascadeBase) ==
                      std::tuple_size_v<decltype(RtHitShadeBindings::gi_cascade_base)>,
                  "cascade count must agree between the binding struct and the cbuffer");
    for (std::size_t i = 0; i < hit.gi_cascade_base.size(); ++i) {
        u.giCascadeBase[i] = hit.gi_cascade_base[i];
    }
    u.giLightCount = hit.light_data != nullptr ? hit.light_count : 0;
    u.giLightCapacity = hit.light_capacity;
    u.frameIndex = reflection_frame_index_;
    u.outputSize = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
    u.tlasNodeCount = scene_.tlas_node_count();
    if (water != nullptr) {
        const float extent = std::max(water->sim_extent, 1.0f);
        u.waterBounds = {water->bounds_min.x, water->bounds_min.y,
                         water->bounds_max.x, water->bounds_max.y};
        u.waterSurface = {water->height, water->roughness,
                          water->sim_displacement, water->sim_normal_strength};
        u.waterSimWindow = {water->sim_min.x, water->sim_min.y, 1.0f / extent, 0.06f};
        u.waterFlowTime = {water->flow.x, water->flow.y, water->flow_speed, water->time};
        u.waterDetail = {water->detail_wave_strength, 1.0f, 0.5f, 0.0f};
    }

    SDL_GPUTexture* const shadow_atlas =
        hit.shadow_atlas != nullptr ? hit.shadow_atlas : depth.handle();
    SDL_GPUSampler* const shadow_cmp = hit.shadow_compare_sampler != nullptr
                                           ? hit.shadow_compare_sampler
                                           : nearest_clamp_sampler_;
    SDL_GPUSampler* const shadow_raw =
        hit.shadow_raw_sampler != nullptr ? hit.shadow_raw_sampler : nearest_clamp_sampler_;
    SDL_GPUTexture* const sky_cube =
        hit.sky_cube != nullptr ? hit.sky_cube : hit.sky_cube_fallback;
    SDL_GPUSampler* const sky_sampler =
        hit.sky_sampler != nullptr ? hit.sky_sampler : linear_clamp_sampler_;
    // A ray miss samples a CUBE, and no 2D texture can stand in for one. The
    // caller always has the engine's dummy black cube to offer; without it
    // there is nothing valid to bind, so skip the pass rather than bind
    // garbage.
    if (sky_cube == nullptr) {
        BRACKEN_LOG_ERROR("rt", "Reflections need a sky cube (or the engine's dummy) to bind");
        return false;
    }

    // Order is the binding order: u0 = radiance, u1 = hit distance.
    const std::array<SDL_GPUStorageTextureReadWriteBinding, 2> rw{{
        {.texture = reflection_->handle(), .mip_level = 0, .layer = 0, .cycle = true,
         .padding1 = 0, .padding2 = 0, .padding3 = 0},
        {.texture = reflection_hit_distance_->handle(), .mip_level = 0, .layer = 0, .cycle = true,
         .padding1 = 0, .padding2 = 0, .padding3 = 0},
    }};
    gpu::ComputePass pass = gpu::ComputePass::begin(cmd, rw, {});
    if (!pass) {
        return false;
    }
    pass.bind_pipeline(*pipeline);
    // ORDER IS THE BINDING CONTRACT (shaders/README.md): sampled textures, then
    // read-only storage textures, then read-only storage buffers, each run
    // numbered from 0 and matching the shader's declaration order exactly.
    //
    // Every slot must hold a VALID resource even where the shader will not read
    // it — an unbound descriptor is undefined, not empty — which is why the
    // shadow atlas falls back to the depth texture (also D32, also comparison-
    // sampleable) and the probe atlases fall back to this renderer's dummies.
    // What decides whether they are READ is `shadowEnabled` and `giEnabled`.
    const std::array<SDL_GPUTextureSamplerBinding, 6> textures{{
        {depth.handle(), nearest_clamp_sampler_},
        {receiver_data, nearest_clamp_sampler_},
        {shadow_atlas, shadow_cmp},
        {shadow_atlas, shadow_raw},
        // A miss with no cube at all falls back to the hemisphere via
        // useSkyCube = 0, but the slot still needs a cube; the caller supplies
        // the engine's dummy. Without even that, the depth texture is not a
        // cube and the bind would be invalid, so bail rather than guess.
        {sky_cube, sky_sampler},
        {gi_ready ? hit.gi_irradiance : gi_dummy_irradiance_->handle(),
         gi_ready ? hit.gi_irradiance_sampler : linear_clamp_sampler_},
    }};
    pass.bind_samplers(0, textures);
    const std::array<SDL_GPUTexture*, 1> storage_textures{
        gi_ready ? hit.gi_probe_state : gi_dummy_probe_state_->handle()};
    pass.bind_storage_textures(0, storage_textures);
    // SIX of SDL3 GPU's eight read-only storage buffers. The light buffer is
    // optional: with no binned lights the shader's loop count is zero, but the
    // slot still has to hold something, so it borrows an acceleration buffer.
    const std::array<SDL_GPUBuffer*, 6> buffers{
        scene_.tlas_nodes(),      scene_.tlas_instances(),
        scene_.blas_nodes(),      scene_.blas_triangles(),
        scene_.triangle_attributes(),
        hit.light_data != nullptr ? hit.light_data : scene_.blas_triangles()};
    pass.bind_storage_buffers(0, buffers);
    pass.push_uniforms(0, &u, sizeof(u));
    pass.dispatch_for(*pipeline, static_cast<uint32_t>(w), static_cast<uint32_t>(h));
    stats_.reflections_traced = true;
    stats_.reflection_rays += static_cast<uint32_t>(w * h);
    return true;
}

bool RtRenderer::render_water_reflections(SDL_GPUCommandBuffer* cmd,
                                          gpu::GPUTexture& opaque_depth,
                                          gpu::GPUTexture& simulation_state,
                                          const FrameParams& params,
                                          const RtReflectionSettings& reflections,
                                          const RtWaterReceiver& water) {
    if (!reflections.enabled) {
        return false;
    }
    return trace_reflections(cmd, opaque_depth, simulation_state.handle(), params,
                             reflections, &water);
}

bool RtRenderer::resolve_reflections(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& hdr_scene,
                                     gpu::GPUTexture& depth, SDL_GPUTexture* normal_rough,
                                     const FrameParams& params,
                                     const RtReflectionSettings& reflections) {
    if (cmd == nullptr || pipelines_ == nullptr || reflection_ == nullptr ||
        normal_rough == nullptr || nearest_clamp_sampler_ == nullptr ||
        !stats_.reflections_traced) {
        return false;
    }
    gpu::GPUPipeline* pipeline =
        pipelines_->get_pipeline(gpu::PipelineType::RtReflectionResolve);
    if (pipeline == nullptr) {
        return false;
    }

    SDL_GPUColorTargetInfo target = {};
    target.texture = hdr_scene.handle();
    target.load_op = SDL_GPU_LOADOP_LOAD;
    target.store_op = SDL_GPU_STOREOP_STORE;
    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &target, 1, nullptr);
    if (pass == nullptr) {
        return false;
    }

    pipeline->bind(pass);
    gpu::RtReflectionResolveUniforms u{};
    u.invViewProj = params.inv_view_proj;
    u.cameraPosition = params.camera_position;
    u.maxRoughness = reflections.max_roughness;
    u.outputSize = {static_cast<float>(std::max(params.screen_width, 1)),
                    static_cast<float>(std::max(params.screen_height, 1))};
    u.reflectionSize = {static_cast<float>(reflection_->width()),
                        static_cast<float>(reflection_->height())};
    u.depthA = params.depth_a;
    u.depthB = params.depth_b;
    u.fadeStart = reflections.fade_start;
    u.intensity = reflections.intensity;
    SDL_PushGPUFragmentUniformData(cmd, 0, &u, sizeof(u));

    const std::array<SDL_GPUTextureSamplerBinding, 3> bindings{{
        // The DENOISED signal when the temporal pass ran this frame, the raw
        // trace otherwise. GGX-sampled reflections are one lobe sample per
        // pixel and are not viewable un-accumulated; the resolve's own
        // material-aware upsample is a spatial filter, not a temporal one.
        {reflection_denoised_this_frame_ && reflection_history_[reflection_history_index_]
             ? reflection_history_[reflection_history_index_]->handle()
             : reflection_->handle(),
         nearest_clamp_sampler_},
        {depth.handle(), nearest_clamp_sampler_},
        {normal_rough, nearest_clamp_sampler_},
    }};
    SDL_BindGPUFragmentSamplers(pass, 0, bindings.data(),
                                static_cast<Uint32>(bindings.size()));
    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
    return true;
}

gpu::GPUTexture* RtRenderer::shadow_mask() const {
    return sun_shadow_.mask.get();
}

gpu::GPUTexture* RtRenderer::local_shadow_mask() const {
    return local_shadow_.mask.get();
}

gpu::GPUTexture* RtRenderer::local_di_diffuse() const {
    return local_di_ready_ ? local_di_.filtered[0].get() : nullptr;
}

gpu::GPUTexture* RtRenderer::local_di_specular() const {
    return local_di_ready_ ? local_di_.filtered[1].get() : nullptr;
}

gpu::GPUTexture* RtRenderer::local_di_direction() const {
    return local_di_ready_ ? local_di_.filtered[2].get() : nullptr;
}

gpu::GPUTexture* RtRenderer::reflection_target() const {
    return reflection_.get();
}

gpu::GPUTexture* RtRenderer::reflection_hit_distance() const {
    return reflection_hit_distance_.get();
}

} // namespace bracken::render
