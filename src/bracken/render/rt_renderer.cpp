#include "bracken/render/rt_renderer.hpp"

#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_debug.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"

#include <algorithm>
#include <array>

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

    return nearest_clamp_sampler_ != nullptr && linear_clamp_sampler_ != nullptr;
}

// The traced-resolution targets only. The full-res mask survives, because it
// is what the forward shaders are still bound to this frame; only a resize
// (which reallocates it anyway) invalidates that.
void RtRenderer::VisibilityDenoiser::release_targets() {
    bits.reset();
    tile_class.reset();
    history[0].reset();
    history[1].reset();
    history_valid = false;
}

void RtRenderer::VisibilityDenoiser::release() {
    release_targets();
    mask.reset();
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
    local_reservoirs_[0].reset();
    local_reservoirs_[1].reset();
    local_reservoir_capacity_ = 0;
    reflection_history_[0].reset();
    reflection_history_[1].reset();
    reflection_history_valid_ = false;
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
    tex = gpu::GPUTexture::create_2d(device, width, height, format,
                                     SDL_GPU_TEXTUREUSAGE_SAMPLER |
                                         SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                                         SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE);
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
    gi_dummy_probe_state_ = gpu::GPUTexture::create_2d(
        *device_, 1, 1, SDL_GPU_TEXTUREFORMAT_R8_UINT,
        SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ);
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

void RtRenderer::resize(gpu::GPUDevice& device, int width, int height) {
    screen_width_ = width;
    screen_height_ = height;
    // Targets are (re)created lazily in render(), where the resolution setting
    // is known — a game that never enables ray tracing should not allocate a
    // screen-sized target for it.
    sun_shadow_.release();
    local_shadow_.release();
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
                               SDL_GPUCommandBuffer* cmd) {
    scene_.set_instances(instances);
    if (device_ != nullptr) {
        scene_.upload(*device_, cmd);
        publish_scene_stats();
    }
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
    stats_.acceleration_bytes = s.gpu_bytes;
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
    stats_.shadow_rays = 0;
    stats_.reflection_rays = 0;
    stats_.local_shadow_rays = 0;

    const bool want_any = shadows.enabled || reflections.enabled || local_shadows.enabled;
    if (!want_any || cmd == nullptr || device_ == nullptr || pipelines_ == nullptr) {
        return; // both off: not a single byte of work
    }
    if (!scene_.ready()) {
        return; // no acceleration structure yet
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
            ensure_target(*device_, sun_shadow_.bits, tiles_x, tiles_y,
                          SDL_GPU_TEXTUREFORMAT_R32_UINT, "shadow bits") &&
            ensure_target(*device_, sun_shadow_.tile_class, tiles_x, tiles_y,
                          SDL_GPU_TEXTUREFORMAT_R8_UINT, "shadow tile class") &&
            ensure_target(*device_, sun_shadow_.history[0], w, h,
                          SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, "shadow history 0") &&
            ensure_target(*device_, sun_shadow_.history[1], w, h,
                          SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, "shadow history 1") &&
            // The mask is FULL resolution now: the denoiser's last pass fuses
            // the upsample into the filter, so the forward shaders sample it
            // 1:1 instead of stretching a half-res binary signal bilinearly.
            ensure_target(*device_, sun_shadow_.mask, screen_width_, screen_height_,
                          SDL_GPU_TEXTUREFORMAT_R8_UNORM, "shadow mask");

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

            const SDL_GPUStorageTextureReadWriteBinding rw{.texture = sun_shadow_.bits->handle(),
                                                           .mip_level = 0,
                                                           .layer = 0,
                                                           .cycle = true,
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
                const std::array<SDL_GPUBuffer*, 4> buffers{
                    scene_.tlas_nodes(), scene_.tlas_instances(), scene_.blas_nodes(),
                    scene_.blas_triangles()};
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
            denoise_visibility(cmd, sun_shadow_, depth, velocity, params, shadows.max_accum_frames,
                               shadows.filter_radius, w, h);
            sun_shadow_.frame_index++;
        }
    } else {
        // Shadows off this frame: the next frame that turns them back on must
        // not reproject through a history describing an unknown number of
        // frames ago.
        sun_shadow_.history_valid = false;
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
                                    gpu::GPUTexture& depth, SDL_GPUTexture* velocity,
                                    const FrameParams& params, float max_accum_frames,
                                    float filter_radius, int traced_w, int traced_h) {
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
    if (classify == nullptr || temporal == nullptr || filter == nullptr) {
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
            const std::array<SDL_GPUTexture*, 1> ro{d.bits->handle()};
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
        u.depthA = params.depth_a;
        u.depthB = params.depth_b;
        u.maxAccumFrames = std::max(max_accum_frames, 1.0f);
        u.depthRejectRelative = 0.05f;
        // No velocity buffer means no reprojection is possible; saying the
        // history is valid anyway would accumulate through camera motion and
        // smear the whole image.
        u.historyValid = (d.history_valid && velocity != nullptr) ? 1u : 0u;

        const SDL_GPUStorageTextureReadWriteBinding rw{.texture = d.history[next]->handle(),
                                                       .mip_level = 0, .layer = 0, .cycle = true,
                                                       .padding1 = 0, .padding2 = 0, .padding3 = 0};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (pass) {
            pass.bind_pipeline(*temporal);
            const std::array<SDL_GPUTextureSamplerBinding, 3> textures{{
                {depth.handle(), nearest_clamp_sampler_},
                {velocity != nullptr ? velocity : depth.handle(), nearest_clamp_sampler_},
                {d.history[prev]->handle(), linear_clamp_sampler_},
            }};
            pass.bind_samplers(0, textures);
            const std::array<SDL_GPUTexture*, 2> ro{d.bits->handle(), d.tile_class->handle()};
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

        const SDL_GPUStorageTextureReadWriteBinding rw{.texture = d.mask->handle(),
                                                       .mip_level = 0, .layer = 0, .cycle = true,
                                                       .padding1 = 0, .padding2 = 0, .padding3 = 0};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (pass) {
            pass.bind_pipeline(*filter);
            const std::array<SDL_GPUTextureSamplerBinding, 2> textures{{
                // Linear: the filter's uniform-tile path samples the history
                // at a sub-texel UV to upsample it continuously.
                {d.history[next]->handle(), linear_clamp_sampler_},
                {depth.handle(), nearest_clamp_sampler_},
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

// Froxel-RIS local-light shadows (docs/rt_lighting_plan.md §5.3, Ratio tier).
//
// One dispatch, then the shared denoiser. The dispatch is over the TILE grid
// rather than the pixel grid — the shader's group IS the 8x4 tile whose bits it
// packs, exactly as the sun trace's is.
bool RtRenderer::render_local_shadows(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                                      SDL_GPUTexture* normal_rough, SDL_GPUTexture* velocity,
                                      const FrameParams& params,
                                      const RtLocalShadowSettings& settings,
                                      const RtClusterBindings& clusters) {
    if (!clusters.complete()) {
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
        ensure_target(*device_, local_shadow_.bits, tiles_x, tiles_y,
                      SDL_GPU_TEXTUREFORMAT_R32_UINT, "local shadow bits") &&
        ensure_target(*device_, local_shadow_.tile_class, tiles_x, tiles_y,
                      SDL_GPU_TEXTUREFORMAT_R8_UINT, "local shadow tile class") &&
        ensure_target(*device_, local_shadow_.history[0], w, h,
                      SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, "local shadow history 0") &&
        ensure_target(*device_, local_shadow_.history[1], w, h,
                      SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, "local shadow history 1") &&
        ensure_target(*device_, local_shadow_.mask, screen_width_, screen_height_,
                      SDL_GPU_TEXTUREFORMAT_R8_UNORM, "local shadow mask");
    if (!targets_ok) {
        return false;
    }

    // The reservoir ping-pong. ComputeStorage rather than Storage: one of the
    // pair is bound read-only to this dispatch and the other read-write, and
    // they swap every frame, so both need both usages.
    const int reservoir_count = w * h;
    if (local_reservoir_capacity_ != reservoir_count) {
        const auto bytes = static_cast<std::size_t>(reservoir_count) * 16u;
        local_reservoirs_[0] =
            gpu::GPUBuffer::create_dynamic(*device_, gpu::GPUBuffer::Type::ComputeStorage, bytes);
        local_reservoirs_[1] =
            gpu::GPUBuffer::create_dynamic(*device_, gpu::GPUBuffer::Type::ComputeStorage, bytes);
        local_reservoir_capacity_ =
            (local_reservoirs_[0] && local_reservoirs_[1]) ? reservoir_count : 0;
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

    const SDL_GPUStorageTextureReadWriteBinding rw_tex{.texture = local_shadow_.bits->handle(),
                                                       .mip_level = 0, .layer = 0, .cycle = true,
                                                       .padding1 = 0, .padding2 = 0, .padding3 = 0};
    // BOTH halves of the reservoir ping-pong bind here, and the second one is
    // only ever READ. SDL3 GPU allows a compute pipeline eight read-only
    // storage buffers and the eight above are spoken for (four acceleration,
    // four cluster), so the previous frame's reservoirs ride a read-write slot
    // instead of a read-only one.
    //
    // cycle = false on both: this is persistent state the NEXT frame reads
    // back, and a cycled allocation would hand the dispatch a fresh, undefined
    // buffer while leaving the one the ping-pong points at untouched.
    const std::array<SDL_GPUStorageBufferReadWriteBinding, 2> rw_bufs{{
        {.buffer = local_reservoirs_[next]->handle(), .cycle = false,
         .padding1 = 0, .padding2 = 0, .padding3 = 0},
        {.buffer = local_reservoirs_[prev]->handle(), .cycle = false,
         .padding1 = 0, .padding2 = 0, .padding3 = 0},
    }};
    gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw_tex, 1}, rw_bufs);
    if (!pass) {
        return false;
    }
    pass.bind_pipeline(*pipeline);
    const std::array<SDL_GPUTextureSamplerBinding, 3> textures{{
        {depth.handle(), nearest_clamp_sampler_},
        {normal_rough, nearest_clamp_sampler_},
        // Slot 2 must always be a valid texture; historyValid is what decides
        // whether it is read, not the bind.
        {velocity != nullptr ? velocity : depth.handle(), nearest_clamp_sampler_},
    }};
    pass.bind_samplers(0, textures);
    // Binding ORDER is the shader's declaration order: the four acceleration
    // buffers, then the four cluster_compute.hlsli claims. The packed stable
    // ids are the fifth thing that header CAN declare and this pass does not
    // ask for them — see the catalogue row for why eight is the whole budget.
    const std::array<SDL_GPUBuffer*, 8> buffers{
        scene_.tlas_nodes(),     scene_.tlas_instances(),  scene_.blas_nodes(),
        scene_.blas_triangles(), clusters.light_data,      clusters.cluster_offsets,
        clusters.light_indices,  clusters.light_remap};
    pass.bind_storage_buffers(0, buffers);
    pass.push_uniforms(0, &u, sizeof(u));
    pass.push_uniforms(1, clusters.params, clusters.params_size);
    pass.dispatch(static_cast<uint32_t>(tiles_x), static_cast<uint32_t>(tiles_y), 1);
    pass.end();

    local_reservoir_index_ = next;
    stats_.local_shadows_traced = true;
    stats_.local_shadow_rays = static_cast<uint32_t>(w * h);

    denoise_visibility(cmd, local_shadow_, depth, velocity, params, settings.max_accum_frames,
                       settings.filter_radius, w, h);
    return true;
}

void RtRenderer::denoise_reflections(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                                     SDL_GPUTexture* normal_rough, SDL_GPUTexture* velocity,
                                     const FrameParams& params,
                                     const RtReflectionSettings& reflections) {
    BRACKEN_PROFILE_ZONE("RtRenderer::denoise_reflections");
    if (reflection_ == nullptr) {
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

gpu::GPUTexture* RtRenderer::reflection_target() const {
    return reflection_.get();
}

gpu::GPUTexture* RtRenderer::reflection_hit_distance() const {
    return reflection_hit_distance_.get();
}

} // namespace bracken::render
