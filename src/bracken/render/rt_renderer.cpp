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

void RtRenderer::shutdown() {
    shadow_mask_.reset();
    reflection_.reset();
    reflection_hit_distance_.reset();
    if (device_ != nullptr && nearest_clamp_sampler_ != nullptr) {
        device_->release_sampler(nearest_clamp_sampler_);
    }
    if (device_ != nullptr && linear_clamp_sampler_ != nullptr) {
        device_->release_sampler(linear_clamp_sampler_);
    }
    nearest_clamp_sampler_ = nullptr;
    linear_clamp_sampler_ = nullptr;
    shadow_bits_.reset();
    shadow_tile_class_.reset();
    shadow_history_[0].reset();
    shadow_history_[1].reset();
    shadow_history_valid_ = false;
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

void RtRenderer::resize(gpu::GPUDevice& device, int width, int height) {
    screen_width_ = width;
    screen_height_ = height;
    // Targets are (re)created lazily in render(), where the resolution setting
    // is known — a game that never enables ray tracing should not allocate a
    // screen-sized target for it.
    shadow_mask_.reset();
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
    stats_.blas_bytes = s.blas_bytes;
    stats_.tlas_bytes = s.tlas_bytes;
    stats_.attribute_bytes = s.attribute_bytes;
    stats_.acceleration_bytes = s.gpu_bytes;
}

void RtRenderer::render(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                        SDL_GPUTexture* normal_rough, SDL_GPUTexture* velocity,
                        const FrameParams& params,
                        const RtShadowSettings& shadows, const RtReflectionSettings& reflections) {
    stats_.shadows_traced = false;
    stats_.reflections_traced = false;
    stats_.shadow_rays = 0;
    stats_.reflection_rays = 0;

    const bool want_any = shadows.enabled || reflections.enabled;
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
        if (shadow_res_ != shadows.resolution) {
            // Every traced-resolution target changes shape, and the history
            // describes a different image; keeping it would smear the old
            // resolution across the first frames of the new one.
            shadow_bits_.reset();
            shadow_tile_class_.reset();
            shadow_history_[0].reset();
            shadow_history_[1].reset();
            shadow_history_valid_ = false;
            shadow_res_ = shadows.resolution;
        }

        // Tile grid over the traced resolution. 8x4 pixels per uint.
        const int tiles_x = (w + 7) / 8;
        const int tiles_y = (h + 3) / 4;

        const bool targets_ok =
            ensure_target(*device_, shadow_bits_, tiles_x, tiles_y,
                          SDL_GPU_TEXTUREFORMAT_R32_UINT, "shadow bits") &&
            ensure_target(*device_, shadow_tile_class_, tiles_x, tiles_y,
                          SDL_GPU_TEXTUREFORMAT_R8_UINT, "shadow tile class") &&
            ensure_target(*device_, shadow_history_[0], w, h,
                          SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, "shadow history 0") &&
            ensure_target(*device_, shadow_history_[1], w, h,
                          SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, "shadow history 1") &&
            // The mask is FULL resolution now: the denoiser's last pass fuses
            // the upsample into the filter, so the forward shaders sample it
            // 1:1 instead of stretching a half-res binary signal bilinearly.
            ensure_target(*device_, shadow_mask_, screen_width_, screen_height_,
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
            u.frameIndex = shadow_frame_index_;
            u.hasGBuffer = normal_rough != nullptr ? 1u : 0u;

            const SDL_GPUStorageTextureReadWriteBinding rw{.texture = shadow_bits_->handle(),
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
            denoise_shadows(cmd, depth, velocity, params, shadows, w, h);
            shadow_frame_index_++;
        }
    } else {
        // Shadows off this frame: the next frame that turns them back on must
        // not reproject through a history describing an unknown number of
        // frames ago.
        shadow_history_valid_ = false;
    }

    // ---- opaque reflections ---------------------------------------------
    reflection_denoised_this_frame_ = false;
    if (reflections.enabled) {
        // The trace + its temporal denoise. The full-res resolve is a separate
        // pass on the scene renderer's own timeline ("RTReflectionResolve").
        BRACKEN_GPU_TIMER(timers_, cmd, "RTReflectionTrace");
        if (trace_reflections(cmd, depth, normal_rough, params, reflections, nullptr)) {
            denoise_reflections(cmd, velocity, reflections);
            reflection_frame_index_++;
        }
    } else {
        // Same reasoning as the shadow history: a gap of unknown length makes
        // the stored accumulation describe a different image.
        reflection_history_valid_ = false;
    }
}

void RtRenderer::denoise_shadows(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                                 SDL_GPUTexture* velocity, const FrameParams& params,
                                 const RtShadowSettings& shadows, int traced_w, int traced_h) {
    BRACKEN_PROFILE_ZONE("RtRenderer::denoise_shadows");

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

        const SDL_GPUStorageTextureReadWriteBinding rw{.texture = shadow_tile_class_->handle(),
                                                       .mip_level = 0, .layer = 0, .cycle = true,
                                                       .padding1 = 0, .padding2 = 0, .padding3 = 0};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (pass) {
            pass.bind_pipeline(*classify);
            const std::array<SDL_GPUTexture*, 1> ro{shadow_bits_->handle()};
            pass.bind_storage_textures(0, ro);
            pass.push_uniforms(0, &u, sizeof(u));
            pass.dispatch_for(*classify, tiles_x, tiles_y);
        }
    }

    // ---- 2. temporal accumulation ---------------------------------------
    const int prev = shadow_history_index_;
    const int next = 1 - shadow_history_index_;
    {
        gpu::RtShadowTemporalUniforms u{};
        u.tracedSize = traced_size;
        u.tileCount = tile_count;
        u.depthA = params.depth_a;
        u.depthB = params.depth_b;
        u.maxAccumFrames = std::max(shadows.max_accum_frames, 1.0f);
        u.depthRejectRelative = 0.05f;
        // No velocity buffer means no reprojection is possible; saying the
        // history is valid anyway would accumulate through camera motion and
        // smear the whole image.
        u.historyValid = (shadow_history_valid_ && velocity != nullptr) ? 1u : 0u;

        const SDL_GPUStorageTextureReadWriteBinding rw{.texture = shadow_history_[next]->handle(),
                                                       .mip_level = 0, .layer = 0, .cycle = true,
                                                       .padding1 = 0, .padding2 = 0, .padding3 = 0};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (pass) {
            pass.bind_pipeline(*temporal);
            const std::array<SDL_GPUTextureSamplerBinding, 3> textures{{
                {depth.handle(), nearest_clamp_sampler_},
                {velocity != nullptr ? velocity : depth.handle(), nearest_clamp_sampler_},
                {shadow_history_[prev]->handle(), linear_clamp_sampler_},
            }};
            pass.bind_samplers(0, textures);
            const std::array<SDL_GPUTexture*, 2> ro{shadow_bits_->handle(),
                                                    shadow_tile_class_->handle()};
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
        u.maxRadius = std::max(shadows.filter_radius, 0.0f);
        u.depthSigma = 0.05f;

        const SDL_GPUStorageTextureReadWriteBinding rw{.texture = shadow_mask_->handle(),
                                                       .mip_level = 0, .layer = 0, .cycle = true,
                                                       .padding1 = 0, .padding2 = 0, .padding3 = 0};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (pass) {
            pass.bind_pipeline(*filter);
            const std::array<SDL_GPUTextureSamplerBinding, 2> textures{{
                // Linear: the filter's uniform-tile path samples the history
                // at a sub-texel UV to upsample it continuously.
                {shadow_history_[next]->handle(), linear_clamp_sampler_},
                {depth.handle(), nearest_clamp_sampler_},
            }};
            pass.bind_samplers(0, textures);
            const std::array<SDL_GPUTexture*, 1> ro{shadow_tile_class_->handle()};
            pass.bind_storage_textures(0, ro);
            pass.push_uniforms(0, &u, sizeof(u));
            pass.dispatch_for(*filter, static_cast<uint32_t>(screen_width_),
                              static_cast<uint32_t>(screen_height_));
        }
    }

    shadow_history_index_ = next;
    shadow_history_valid_ = true;
}

void RtRenderer::denoise_reflections(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* velocity,
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

    gpu::RtReflectionTemporalUniforms u{};
    u.tracedSize = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
    u.maxAccumFrames = std::max(reflections.max_accum_frames, 1.0f);
    u.clampSigma = std::max(reflections.clamp_sigma, 0.0f);
    u.historyValid = (reflection_history_valid_ && velocity != nullptr) ? 1u : 0u;

    const SDL_GPUStorageTextureReadWriteBinding rw{.texture = reflection_history_[next]->handle(),
                                                   .mip_level = 0, .layer = 0, .cycle = true,
                                                   .padding1 = 0, .padding2 = 0, .padding3 = 0};
    gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
    if (!pass) {
        return;
    }
    pass.bind_pipeline(*pipeline);
    const std::array<SDL_GPUTextureSamplerBinding, 3> textures{{
        {reflection_->handle(), nearest_clamp_sampler_},
        {velocity != nullptr ? velocity : reflection_->handle(), nearest_clamp_sampler_},
        {reflection_history_[prev]->handle(), linear_clamp_sampler_},
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

    gpu::RtReflectionUniforms u{};
    u.invViewProj = params.inv_view_proj;
    u.cameraPosition = params.camera_position;
    u.rayMaxDistance = reflections.max_distance;
    u.sunDirection = params.environment.sun_dir;
    u.maxRoughness = reflections.max_roughness;
    u.fadeStart = reflections.fade_start;
    u.sunColor = params.environment.sun_color;
    u.normalBias = reflections.normal_bias;
    u.skyColor = params.environment.sky_horizon;
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
    const std::array<SDL_GPUTextureSamplerBinding, 2> textures{{
        {depth.handle(), nearest_clamp_sampler_},
        {receiver_data, nearest_clamp_sampler_},
    }};
    pass.bind_samplers(0, textures);
    const std::array<SDL_GPUBuffer*, 5> buffers{
        scene_.tlas_nodes(), scene_.tlas_instances(), scene_.blas_nodes(),
        scene_.blas_triangles(), scene_.triangle_attributes()};
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
    return shadow_mask_.get();
}

gpu::GPUTexture* RtRenderer::reflection_target() const {
    return reflection_.get();
}

gpu::GPUTexture* RtRenderer::reflection_hit_distance() const {
    return reflection_hit_distance_.get();
}

} // namespace bracken::render
