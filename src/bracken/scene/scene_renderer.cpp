#include "scene_renderer.hpp"
#include "scene_renderer_internal.hpp"
#include "bracken/animation/animation_types.hpp" // MAX_BONES
#include "bracken/core/assert.hpp"
#include "bracken/core/color.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/core/jobs/task_group.hpp"
#include "bracken/core/log_throttle.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_debug.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/heightmap.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render/gtao.hpp"
#include "bracken/render/ibl.hpp"
#include "bracken/render/auto_exposure.hpp"
#include "bracken/render/color_grading.hpp"
#include "bracken/render/post_process_stack.hpp"
#include "bracken/render/bloom.hpp"
#include "bracken/render/effect_renderer.hpp"
#include "bracken/render/grass_renderer.hpp"
#include "bracken/render/render_context.hpp"
#include "bracken/render/shadow_map.hpp"
#include "bracken/render/taa.hpp"
#include "bracken/render/water_renderer.hpp"
#include "bracken/render/terrain_renderer.hpp"
#include "bracken/render/ui_renderer.hpp"
#include "bracken/render/volumetric_fog.hpp"
#include "bracken/render/world_renderer.hpp"
#include "frustum.hpp"
#include "SDL3/SDL_gpu.h"
#include "SDL3/SDL_video.h"
#include <SDL3_image/SDL_image.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

#include "bracken/graphics_settings.hpp"
#include "bracken/scene/camera_jitter.hpp"
#include "bracken/scene/camera_state.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/render_plan.hpp"
#include "bracken/scene/scene_prep.hpp"
#include "bracken/scene/transform_math.hpp"
#include "glm/common.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"
#include "glm/matrix.hpp"

namespace bracken::scene {

namespace gpu = bracken::gpu;
namespace render = bracken::render;

SceneRenderer::SceneRenderer()
    : pipeline_registry_(std::make_unique<gpu::PipelineRegistry>()),
      terrain_(std::make_unique<render::TerrainRenderer>()),
      world_(std::make_unique<render::WorldRenderer>()),
      ui_(std::make_unique<render::UIRenderer>()),
      effects_(std::make_unique<render::EffectRenderer>()),
      model_manager_(std::make_unique<ModelManager>()),
      grass_renderer_(std::make_unique<render::GrassRenderer>()),
      shadow_map_(std::make_unique<render::ShadowMap>()),
      post_(std::make_unique<render::PostProcessStack>()),
      gtao_(std::make_unique<render::GTAO>()),
      bloom_(std::make_unique<render::Bloom>()),
      volumetric_fog_(std::make_unique<render::VolumetricFog>()),
      taa_(std::make_unique<render::TAA>()),
      auto_exposure_(std::make_unique<render::AutoExposure>()),
      ibl_(std::make_unique<render::IBL>()),
      water_(std::make_unique<render::WaterRenderer>()),
      local_shadows_(std::make_unique<render::lighting::LocalShadowMap>()) {}

SceneRenderer::~SceneRenderer() {
    shutdown();
}

// ========== Accessors (out-of-line for forward-declared types) ==========

render::RenderContext* SceneRenderer::context() {
    return context_;
}
render::TerrainRenderer& SceneRenderer::terrain() {
    return *terrain_;
}
ModelManager& SceneRenderer::models() {
    return *model_manager_;
}
render::GrassRenderer* SceneRenderer::grass() {
    return grass_renderer_.get();
}
render::WaterRenderer* SceneRenderer::water() {
    return water_.get();
}
float SceneRenderer::get_terrain_height(float x, float z) {
    return terrain_->get_height(x, z);
}
int SceneRenderer::text_width(std::string_view text, float scale) const {
    return ui_ ? ui_->text_width(text, scale) : 0;
}
int SceneRenderer::text_height(float scale) const {
    return ui_ ? ui_->text_height(scale) : 0;
}

bool SceneRenderer::init(render::RenderContext& context, float world_width, float world_height) {
    // The local mirror exists because scene_renderer.hpp only forward-declares
    // ShadowMap; this TU sees both definitions, so keep them in lockstep here.
    static_assert(ScenePrep::MAX_SHADOW_CASCADES == render::CSM_MAX_CASCADES,
                  "ScenePrep::MAX_SHADOW_CASCADES must match render::CSM_MAX_CASCADES");
    context_ = &context;

    if (!pipeline_registry_->init(context_->device())) {
        BRACKEN_LOG_ERROR(LOG_TAG, "Failed to initialize pipeline registry");
        return false;
    }
    pipeline_registry_->set_swapchain_format(context_->swapchain_format());

    model_manager_->set_device(&context_->device());

    if (!terrain_->init(context_->device(), *pipeline_registry_, world_width, world_height)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "Failed to initialize terrain renderer");
        return false;
    }

    if (!world_->init(context_->device(), *pipeline_registry_)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "Failed to initialize world renderer");
        return false;
    }

    int w = context_->width();
    int h = context_->height();

    if (!ui_->init(context_->device(), *pipeline_registry_, w, h)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "Failed to initialize UI renderer");
        return false;
    }

    if (!effects_->init(context_->device(), *pipeline_registry_, model_manager_.get())) {
        BRACKEN_LOG_ERROR(LOG_TAG, "Failed to initialize effect renderer");
        return false;
    }

    depth_texture_ = gpu::GPUTexture::create_depth(context_->device(), w, h);
    if (!depth_texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "Failed to create depth texture");
        return false;
    }

    init_pipelines();

    // Preload all pipelines upfront to avoid hitching during gameplay
    if (!pipeline_registry_->preload_all_pipelines()) {
        BRACKEN_LOG_WARN(LOG_TAG, "Some pipelines failed to preload");
    }

    if (grass_renderer_ && !grass_renderer_->init(context_->device(), *pipeline_registry_, world_width, world_height)) {
        BRACKEN_LOG_WARN(LOG_TAG, "Failed to initialize grass renderer (grass disabled)");
    }

    const GraphicsSettings& gfx = graphics_ ? *graphics_ : default_graphics_;
    const int shadow_res = shadow_map_texels(gfx.shadow_resolution);
    shadow_map_->set_active_cascades(cascade_count(gfx.shadow_cascades));
    if (!shadow_map_->init(context_->device(), shadow_res)) {
        BRACKEN_LOG_WARN(LOG_TAG, "Failed to initialize shadow map (shadows disabled)");
    }

    const int local_shadow_res = local_shadow_atlas_texels(gfx.local_shadow_resolution);
    if (!local_shadows_->init(context_->device(), local_shadow_res)) {
        BRACKEN_LOG_WARN(LOG_TAG, "Failed to initialize local shadow map (point/spot shadows disabled)");
    }

    if (!post_->init(context_->device(), w, h)) {
        BRACKEN_LOG_WARN(LOG_TAG, "Failed to initialize post-process stack (HDR path disabled)");
    }

    if (!gtao_->init(context_->device(), w, h, gfx.ao_mode != AOMode::GTAO_Full)) {
        BRACKEN_LOG_WARN(LOG_TAG, "Failed to initialize GTAO (AO disabled)");
    }

    if (!bloom_->init(context_->device(), w, h)) {
        BRACKEN_LOG_WARN(LOG_TAG, "Failed to initialize Bloom (bloom disabled)");
    }

    if (!volumetric_fog_->init(context_->device(), w, h)) {
        BRACKEN_LOG_WARN(LOG_TAG, "Failed to initialize Volumetric Fog (fog disabled)");
    }

    if (!taa_->init(context_->device(), w, h)) {
        BRACKEN_LOG_WARN(LOG_TAG, "Failed to initialize TAA (temporal AA disabled)");
    }

    if (!auto_exposure_->init(context_->device())) {
        BRACKEN_LOG_WARN(LOG_TAG, "Failed to initialize AutoExposure (histogram exposure disabled)");
    }

    if (!ibl_->init(context_->device(), *pipeline_registry_)) {
        BRACKEN_LOG_WARN(LOG_TAG, "Failed to initialize IBL (sky image-based lighting disabled)");
    }

    if (!water_->init(context_->device(), *pipeline_registry_)) {
        BRACKEN_LOG_WARN(LOG_TAG, "Failed to initialize WaterRenderer (water disabled)");
    }

    // Timestamp pool is a documented stub (returns zeros); failure is harmless.
    (void)gpu_timer_pool_.init(context_->device(), 64);

    cluster_grid_ready_ = cluster_grid_.init(context_->device(), render::lighting::MAX_LIGHTS);
    if (!cluster_grid_ready_) {
        BRACKEN_LOG_WARN(LOG_TAG, "Failed to initialize ClusterGrid (dynamic lighting disabled)");
    }

    return true;
}

void SceneRenderer::shutdown() {
    if (model_manager_) {
        model_manager_->unload_all();
    }

    if (grass_renderer_) {
        grass_renderer_->shutdown();
    }

    depth_texture_.reset();
    velocity_texture_.reset();

    if (default_sampler_ && context_) {
        context_->device().release_sampler(default_sampler_);
        default_sampler_ = nullptr;
    }

    gpu_timer_pool_.shutdown();
    water_->shutdown();
    water_scene_copy_.reset();
    water_depth_copy_.reset();
    ibl_->shutdown();
    auto_exposure_->shutdown();
    taa_->shutdown();
    volumetric_fog_->shutdown();
    bloom_->shutdown();
    gtao_->shutdown();
    post_->shutdown();
    local_shadows_->shutdown();
    shadow_map_->shutdown();
    effects_->shutdown();
    ui_->shutdown();
    world_->shutdown();
    terrain_->shutdown();
    pipeline_registry_->shutdown();
}

void SceneRenderer::init_pipelines() {
    auto* model_pipeline = pipeline_registry_->get_model_pipeline();
    auto* skinned_pipeline = pipeline_registry_->get_skinned_model_pipeline();

    if (!model_pipeline || !skinned_pipeline) {
        BRACKEN_LOG_WARN(LOG_TAG, "Some pipelines failed to preload");
    }

    SDL_GPUSamplerCreateInfo sampler_info = {};
    sampler_info.min_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mag_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
    sampler_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sampler_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sampler_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sampler_info.max_anisotropy = 16.0f;
    sampler_info.enable_anisotropy = true;
    // Vulkan clamps sampling to [min_lod, max_lod]; leaving max_lod at 0 would
    // pin every model texture to mip 0 and defeat the generated mip chains.
    sampler_info.max_lod = 1000.0f;
    default_sampler_ = context_->device().create_sampler(sampler_info);

    if (!default_sampler_) {
        BRACKEN_LOG_WARN(LOG_TAG, "Failed to create default GPU sampler");
    }

    // Create 1x1 white dummy texture for binding slots that require a texture
    {
        uint32_t white_pixel = 0xFFFFFFFF;
        dummy_white_texture_ =
            gpu::GPUTexture::create_2d(context_->device(), 1, 1, gpu::TextureFormat::RGBA8, &white_pixel);
    }

    // 1x1 flat tangent-space normal: (0,0,1) packed as UNORM (128,128,255).
    // Bound on the normal-map slot when the material has no normal map, purely
    // to keep the binding valid for SDL3 GPU: the model shaders branch on
    // hasNormalMap and never sample it on the no-map path, so the contents are
    // inert (flat values chosen only for debuggability).
    {
        uint8_t flat_normal[4] = {128, 128, 255, 255};
        default_normal_texture_ =
            gpu::GPUTexture::create_2d(context_->device(), 1, 1, gpu::TextureFormat::RGBA8, flat_normal);
    }

    // 1x1 black for the emissive slot when the material has no emissive map.
    // Also a safe filler for the metallic-roughness slot (sampling is gated by
    // hasMRMap, so the contents never reach the BRDF) — but white is bound there.
    {
        uint32_t black_pixel = 0xFF000000;
        dummy_black_texture_ =
            gpu::GPUTexture::create_2d(context_->device(), 1, 1, gpu::TextureFormat::RGBA8, &black_pixel);
    }

    // 1x1x6 cube for the IBL cube slots when the IBL renderer is unavailable:
    // a TextureCube-declared slot must be fed a cube-typed texture. Never
    // sampled (the shaders branch on iblEnabled == 0 in that case), so the
    // contents stay undefined by design.
    dummy_black_cube_ = gpu::GPUTexture::create_cube(context_->device(), 1, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
                                                     SDL_GPU_TEXTUREUSAGE_SAMPLER);
}

// ============================================================================
// Configuration
// ============================================================================

void SceneRenderer::set_screen_size(int width, int height) {
    ui_->set_screen_size(width, height);
    if (post_->is_ready()) {
        post_->resize(width, height);
    }
    if (gtao_->is_ready()) {
        gtao_->resize(width, height);
    }
    if (volumetric_fog_->is_ready()) {
        volumetric_fog_->resize(width, height);
    }
}

void SceneRenderer::set_graphics_settings(const GraphicsSettings& settings) {
    // Apply shadow settings unconditionally, comparing against the shadow
    // map's current state, so saved settings take effect on the first call
    // at boot (reinit is a no-op before init since there's no device yet).
    const int new_res = shadow_map_texels(settings.shadow_resolution);
    const int new_cascades = cascade_count(settings.shadow_cascades);

    if (new_res != shadow_map_->resolution()) {
        shadow_map_->reinit(new_res);
    }
    if (new_cascades != shadow_map_->active_cascades()) {
        shadow_map_->set_active_cascades(new_cascades);
    }

    // Local (point/spot) shadow atlas resolution — reinit drops every cached
    // tile, so lights re-render under the per-frame cap over the next frames.
    const int new_local_res = local_shadow_atlas_texels(settings.local_shadow_resolution);
    if (new_local_res != local_shadows_->atlas_resolution()) {
        local_shadows_->reinit(new_local_res);
    }

    default_graphics_ = settings;
    graphics_ = &default_graphics_;

    // Forward lighting tunables to renderers that own their own uniform builders.
    if (terrain_) {
        terrain_->set_lighting_tunables(settings.ambient_strength, settings.sun_intensity);
    }
    if (grass_renderer_) {
        grass_renderer_->ambient_strength = settings.ambient_strength;
        grass_renderer_->sun_intensity = settings.sun_intensity;
    }
}

void SceneRenderer::set_vsync_mode(int mode) {
    if (context_) {
        context_->set_vsync_mode(mode);
    }
}

int SceneRenderer::max_vsync_mode() const {
    return context_ ? context_->max_vsync_mode() : 1;
}

void SceneRenderer::set_anisotropic_filter(int level) {
    float aniso_value = 1.0f;
    if (level > 0) {
        aniso_value = static_cast<float>(1 << level);
    }
    aniso_value = std::min(aniso_value, 16.0f);

    terrain_->set_anisotropic_filter(aniso_value);

    if (default_sampler_ && context_) {
        context_->device().release_sampler(default_sampler_);

        SDL_GPUSamplerCreateInfo sampler_info = {};
        sampler_info.min_filter = SDL_GPU_FILTER_LINEAR;
        sampler_info.mag_filter = SDL_GPU_FILTER_LINEAR;
        sampler_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
        sampler_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
        sampler_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
        sampler_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
        sampler_info.max_anisotropy = aniso_value;
        sampler_info.enable_anisotropy = (level > 0);
        // Keep mip chains reachable (max_lod = 0 clamps sampling to mip 0).
        sampler_info.max_lod = 1000.0f;
        default_sampler_ = context_->device().create_sampler(sampler_info);

        if (!default_sampler_) {
            BRACKEN_LOG_WARN(LOG_TAG, "Failed to recreate default GPU sampler (anisotropic level: {})", level);
        }
    }
}

void SceneRenderer::set_heightmap(const Heightmap& heightmap) {
    terrain_->set_heightmap(heightmap);

    if (grass_renderer_ && terrain_->heightmap_texture()) {
        render::HeightmapParams hm_params;
        hm_params.world_origin_x = heightmap.world_origin_x;
        hm_params.world_origin_z = heightmap.world_origin_z;
        hm_params.world_size = heightmap.world_size;
        hm_params.min_height = heightmap.min_height;
        hm_params.max_height = heightmap.max_height;
        grass_renderer_->set_heightmap(terrain_->heightmap_texture(), hm_params);
    }

    BRACKEN_LOG_DEBUG(LOG_TAG, "Heightmap set for terrain rendering");
}

// ============================================================================
// Frame Lifecycle
// ============================================================================

void SceneRenderer::begin_frame() {
    context_->begin_frame();
    had_main_pass_this_frame_ = false;
    ui_->set_screen_size(context_->width(), context_->height());
}

void SceneRenderer::end_frame() {
    current_swapchain_ = nullptr;
    context_->end_frame();
}

void SceneRenderer::begin_main_pass() {
    SDL_GPUCommandBuffer* cmd = context_->current_command_buffer();
    if (!cmd) {
        BRACKEN_LOG_ERROR(LOG_TAG, "begin_main_pass: No active command buffer");
        return;
    }

    uint32_t sw_width = 0;
    uint32_t sw_height = 0;
    current_swapchain_ = context_->acquire_swapchain_texture(cmd, &sw_width, &sw_height);
    if (!current_swapchain_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "begin_main_pass: Failed to acquire swapchain texture");
        return;
    }

    if (depth_texture_ && (depth_texture_->width() != static_cast<int>(sw_width) ||
                           depth_texture_->height() != static_cast<int>(sw_height))) {
        depth_texture_ = gpu::GPUTexture::create_depth(context_->device(), sw_width, sw_height);
        if (!depth_texture_) {
            BRACKEN_LOG_ERROR(LOG_TAG, "begin_main_pass: Failed to resize depth texture");
            return;
        }
    }

    SDL_GPUColorTargetInfo color_target = {};
    color_target.texture = current_swapchain_;
    color_target.load_op = SDL_GPU_LOADOP_CLEAR;
    color_target.store_op = SDL_GPU_STOREOP_STORE;
    color_target.clear_color = {0.35f, 0.45f, 0.6f, 1.0f};

    SDL_GPUDepthStencilTargetInfo depth_target = {};
    depth_target.texture = depth_texture_ ? depth_texture_->handle() : nullptr;
    // Explicit CLEAR: zero-init would silently mean LOADOP_LOAD (== 0).
    depth_target.load_op = SDL_GPU_LOADOP_CLEAR;
    depth_target.store_op = SDL_GPU_STOREOP_STORE;
    depth_target.clear_depth = 0.0f; // reversed-Z: 0.0 is the far plane
    depth_target.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
    depth_target.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;

    main_render_pass_ = SDL_BeginGPURenderPass(cmd, &color_target, 1, depth_texture_ ? &depth_target : nullptr);
    if (!main_render_pass_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "begin_main_pass: Failed to begin render pass");
        return;
    }

    had_main_pass_this_frame_ = true;
}

void SceneRenderer::end_main_pass() {
    if (main_render_pass_) {
        SDL_EndGPURenderPass(main_render_pass_);
        main_render_pass_ = nullptr;
    }
}

void SceneRenderer::begin_ui() {
    SDL_GPUCommandBuffer* cmd = context_->current_command_buffer();
    if (!cmd) {
        BRACKEN_LOG_ERROR(LOG_TAG, "begin_ui: No active command buffer");
        return;
    }

    if (!current_swapchain_) {
        uint32_t sw_width = 0;
        uint32_t sw_height = 0;
        current_swapchain_ = context_->acquire_swapchain_texture(cmd, &sw_width, &sw_height);
        if (!current_swapchain_) {
            return;
        }
    }

    ui_->begin(cmd);
}

void SceneRenderer::end_ui() {
    ui_->end();

    // Sampled before execute(), which clears the batch.
    if (collect_stats_) {
        render_stats_.ui_vertices = static_cast<uint32_t>(ui_->batch_vertex_count());
        render_stats_.ui_draws = static_cast<uint32_t>(ui_->batch_draw_count());
    }

    SDL_GPUCommandBuffer* cmd = context_->current_command_buffer();
    if (cmd && current_swapchain_) {
        bool clear_background = !had_main_pass_this_frame_;
        ui_->execute(cmd, current_swapchain_, clear_background);
    }
}

// ============================================================================
// Main Render Frame
// ============================================================================

void SceneRenderer::render_frame(const FramePacket& frame) {
    BRACKEN_PROFILE_ZONE("SceneRenderer::render_frame");
    // Swapchain acquire + every sub-renderer assume the SDL main thread.
    BRACKEN_ASSERT_MAIN_THREAD();
    // Diagnostics reset unconditionally: they are drop counters, not opt-in
    // instrumentation (see SceneRenderer::frame_diagnostics()).
    frame_diagnostics_ = {};
    if (collect_stats_) {
        render_stats_ = {};
        render_stats_.jobs_pending = core::jobs::JobSystem::instance().pending_count();
        render_stats_.packet_bytes_used = frame.bytes_used();
        render_stats_.packet_bytes_capacity = frame.bytes_capacity();
        render_stats_.bone_palettes_used = static_cast<uint32_t>(frame.palettes().size());
        render_stats_.bone_palettes_budget = FramePacket::MAX_BONE_PALETTES;
    }
    // Emit-time drops the FrameBuilder already counted on the packet.
    const FramePacket::BuildDiagnostics build_diag = frame.build_diagnostics();
    frame_diagnostics_.draws_dropped_unknown_model += build_diag.draws_dropped_invalid_handle;
    frame_diagnostics_.skinned_dropped_no_skeleton += build_diag.skinned_dropped_invalid_palette;
    frame_diagnostics_.palettes_over_budget += build_diag.palettes_over_budget;
    const GraphicsSettings& gfx = graphics_ ? *graphics_ : default_graphics_;
    const float dt = frame.dt();

    // Camera: per-frame optional. Adopt the packet camera when the host set
    // one; a camera-less frame reuses the previous frame's camera, and until
    // any camera was ever set the 3D passes are skipped below.
    if (frame.has_camera()) {
        frame_camera_state_ = frame.camera();
        have_frame_camera_ = true;
    }
    const CameraState& camera = frame_camera_state_;

    // Environment: adopt the packet snapshot when the host set one this
    // frame; otherwise env_ (live-editable via environment(), e.g. the dev
    // overlay) is the frame's environment unchanged.
    if (frame.has_environment()) {
        env_ = frame.environment();
    }

    // ===== The frame's pass plan =====
    // Which passes run, and the frame-scoped cull constants — decided ONCE,
    // here, from the settings + the sealed packet + what the sub-renderers
    // can currently do (scene/render_plan.hpp). Everything below reads the
    // plan instead of re-deriving a predicate, so there is exactly one
    // definition of "is TAA on this frame". The plan is pure: the two
    // invalidations it reports are performed at their original sites below,
    // AFTER the minimized-window early-out, so a minimized frame still drops
    // nothing.
    RendererCaps caps;
    caps.post_ready = post_->is_ready();
    caps.taa_ready = taa_->is_ready();
    caps.ibl_ready = ibl_ && ibl_->is_ready();
    caps.camera_seen = have_frame_camera_;
    caps.shadows_ready = shadow_map_->is_ready();
    caps.shadow_cascades = shadow_map_->active_cascades();
    const RenderPlan plan = plan_frame(gfx, frame, caps);

    // Cache frame-level state once (avoids repeated ternary/function calls)
    frame_fog_active_ = plan.fog_active;
    frame_draw_dist_sq_ = plan.draw_dist_sq;
    frame_do_frustum_cull_ = plan.do_frustum_cull;

    // Scene/sky clocks are advanced by the SIM stage and carried in the
    // packet's engine section (docs/multithreading_plan.md Phase 4): adopt
    // them — the renderer no longer integrates dt. Hand-built packets
    // without an engine section keep the previous clocks (frozen, never
    // wrong-rate). The sky clock advances even when the skybox is toggled
    // off, so the sky never freezes; grass wind runs off the scene clock.
    if (frame.has_engine_frame()) {
        scene_time_ = frame.scene_time();
        world_->set_sky_time(frame.sky_time());
    }
    frame_effect_count_ = frame.effect_instance_count();
    frame_particle_count_ = static_cast<uint32_t>(frame.effect_particles().size());

    // Propagate the environment to every renderer that builds its own lighting
    // uniforms; env_ is the single source of truth for sun/sky/fog this frame.
    light_dir_ = env_.sun_dir;
    world_->set_environment(env_);
    terrain_->set_environment(env_);
    if (grass_renderer_) {
        grass_renderer_->set_environment(env_);
    }
    if (water_) {
        water_->set_environment(env_);
    }

    // Pre-upload every model referenced by the packet's effect-instance
    // snapshot BEFORE any render pass (including the minimized-window
    // early-out below) — uploads cannot run inside a render pass, and the
    // mid-pass upload path in EffectRenderer is a logged skip, never a
    // submit hitch. The snapshot is the render side's ONLY view of the
    // particle system: spawns were applied and the live EffectSystem updated
    // on the sim side before this packet sealed.
    effects_->ensure_models_uploaded(frame.effect_emitters());
    frame_diagnostics_.effect_models_unresolved += effects_->models_unresolved();

    begin_frame();

    // Minimized windows can't acquire a swapchain image; skip all recording but
    // still submit the command buffer so the frame lifecycle stays balanced.
    if (context_->window() && (SDL_GetWindowFlags(context_->window()) & SDL_WINDOW_MINIMIZED) != 0) {
        end_frame();
        return;
    }

    gpu_timer_pool_.begin_frame();

    // 3D needs both content and a camera (no camera ever set = skip 3D,
    // conservative fallback for the packet's optional camera). A frame that
    // describes geometry and never set a camera renders NOTHING — that is the
    // single most common "I built a frame and see nothing" report, so name it
    // and name the fix. Once: it is a standing property of the host's code.
    if (frame_needs_camera(frame, have_frame_camera_)) {
        frame_diagnostics_.camera_missing = true;
        BRACKEN_LOG_WARN_ONCE("render",
                              "frame has 3D content but no camera was ever set - call frame().set_camera(); "
                              "the entire 3D pass is being skipped");
    }
    const bool has_content = plan.has_content;
    frame_prepass_active_ = false; // set by the HDR path when the prepass records
    // Reset the frame's AO texture to the white dummy: forward pipelines
    // always bind the slot, and GTAO's textures may have been recreated
    // since last frame (resize / quality change).
    frame_ao_texture_ = dummy_white_texture_ ? dummy_white_texture_->handle() : nullptr;
    // Sky IBL drives every scene shader's iblEnabled flag (the IBL sampler
    // slots are bound either way — dummies when inactive).
    frame_ibl_active_ = plan.ibl_active;
    // Always render 3D through the HDR offscreen target so scene lighting stays
    // in linear HDR; the composite pass tonemaps + gamma-corrects to the swapchain.
    const bool use_hdr_path = plan.use_hdr_path;
    // TAA drives the sub-pixel camera jitter: jitter without a resolve to
    // integrate the samples would just shimmer, so every other mode renders
    // unjittered.
    const bool taa_active = plan.taa_active;
    taa_jitter_enabled_ = taa_active;
    if (plan.invalidate_taa_history) {
        // Any gap (mode switch, fallback path) stales the accumulated
        // history; drop it so re-enabling TAA restarts cleanly.
        taa_->invalidate_history();
    }
    // Auto exposure follows the same gap policy: any frame its passes cannot
    // record (setting off, fallback path, no 3D content) stales the adapted
    // EV — snap to the metered target when it comes back.
    if (plan.invalidate_auto_exposure) {
        auto_exposure_->invalidate();
    }
    // Set when a late swapchain acquire fails: composite + UI are skipped but
    // the command buffer is still submitted normally.
    bool swapchain_unavailable = false;

    // Draw camera defaults to the host camera; the HDR path below may swap in
    // the jittered projection (culling and velocity always stay unjittered).
    CameraState draw_camera = camera;

    if (has_content) {
        // Extract main camera frustum once (reused in render_3d_scene for
        // skinned models). Always from the UNJITTERED host matrices.
        frame_frustum_.extract_from_matrix(camera.view_projection);

        // ===== Frame camera matrices (motion vectors / TAA jitter) =====
        // Capture unjittered copies FIRST, then apply the optional sub-pixel
        // jitter to the draw projection only.
        frame_cam_.view = camera.view;
        frame_cam_.proj_unjittered = camera.projection;
        frame_cam_.view_proj_unjittered = camera.projection * camera.view;
        if (taa_jitter_enabled_) {
            frame_cam_.jitter_pixels = halton_jitter(frame_index_);
            frame_cam_.proj_jittered =
                jitter_projection(camera.projection, frame_cam_.jitter_pixels,
                                  static_cast<float>(context_->width()), static_cast<float>(context_->height()));
            frame_cam_.view_proj_jittered = frame_cam_.proj_jittered * camera.view;
        } else {
            frame_cam_.jitter_pixels = glm::vec2(0.0f);
            frame_cam_.proj_jittered = camera.projection;
            frame_cam_.view_proj_jittered = frame_cam_.view_proj_unjittered;
        }
        if (!have_prev_view_proj_) {
            // First rendered frame: previous == current => zero camera velocity.
            prev_view_proj_unjittered_ = frame_cam_.view_proj_unjittered;
        }
        draw_camera.projection = frame_cam_.proj_jittered;
        draw_camera.view_projection = frame_cam_.view_proj_jittered;

        if (draw_camera.projection != cached_projection_) {
            cached_projection_ = draw_camera.projection;
            cached_inv_projection_ = glm::inverse(draw_camera.projection);
        }

        // Update cascade matrices FIRST so we can extract per-cascade light frustums
        // and use them to cull shadow casters per cascade in build_instance_batches.
        // plan.active_cascades is the single "shadows on" answer (it is 0
        // exactly when the atlas is unavailable or the mode is Off).
        const bool shadows_on = plan.active_cascades > 0;
        if (shadows_on) {
            // Near plane recovered from the reversed-Z RH_ZO projection
            // (n = m32 / (m22 + 1); under standard-Z it was m32/m22).
            // Shadow range is a quality setting (clamped by draw distance);
            // the shader fades shadows out over the tail instead of popping
            // them off.
            const float cam_near = std::max(camera.projection[3][2] / (camera.projection[2][2] + 1.0f), 0.1f);
            const float shadow_far = std::min(gfx.get_draw_distance(), gfx.shadow_distance);
            shadow_map_->update(camera.view, camera.projection, light_dir_, cam_near, shadow_far);
            for (int c = 0; c < plan.active_cascades; ++c) {
                shadow_cascade_frustums_[c].extract_from_matrix(shadow_map_->cascades()[c].light_view_projection);
            }
        }

        // ===== CPU prep for all per-frame GPU uploads =====

        // Build instance batches (cull for main + per-cascade shadows) and
        // resolve the skinned commands against the packet-owned palettes.
        build_instance_batches(frame, camera, frame_frustum_, plan.active_cascades);
        pack_bone_palettes(frame, camera);

        // Grass chunk culling/visibility prep (upload happens below).
        const bool grass_active = frame.should_draw_grass() && gfx.grass_enabled && grass_renderer_;
        if (grass_active) {
            // Cap grass at the scene draw distance; the host-set
            // grass_view_distance itself is respected (no longer overwritten).
            grass_renderer_->set_view_distance_cap(gfx.get_draw_distance());
            grass_renderer_->update(scene_time_);
        }

        // Build cluster light grid on the CPU. Skip the build + upload when the
        // scene has zero dynamic lights — the per-frag cluster path early-outs
        // via gridDim.w==0. We still need to bind the SSBOs to terrain/grass
        // pipelines below so those binding slots are valid; the buffers retain
        // their last-uploaded state (initially zero).
        const bool any_dyn_lights = !frame.point_lights().empty() || !frame.spot_lights().empty();
        const bool cluster_upload_needed = cluster_grid_ready_ && any_dyn_lights;
        frame_local_shadows_active_ =
            cluster_upload_needed && gfx.local_shadows && local_shadows_ && local_shadows_->is_ready();
        if (cluster_upload_needed) {
            // Local (point/spot) light shadows: the selector picks the
            // budgeted shadow casters, assigns atlas tiles, decides which
            // tiles re-render this frame, and writes shadow_slot into these
            // per-frame light copies (scene commands stay untouched).
            frame_point_lights_.assign(frame.point_lights().begin(), frame.point_lights().end());
            frame_spot_lights_.assign(frame.spot_lights().begin(), frame.spot_lights().end());
            if (frame_local_shadows_active_) {
                // Dynamic casters (moving statics + every skinned command):
                // cached tiles whose light sphere they touch must re-render.
                local_dynamic_casters_.clear();
                for (const auto& mc : frame.model_commands()) {
                    if (mc.motion_id == 0) {
                        continue;
                    }
                    Model* model = model_manager_->get_model(mc.model_handle);
                    if (!model) {
                        continue;
                    }
                    const glm::vec3 wc = glm::vec3(mc.transform * glm::vec4(model->bounding_center, 1.0f));
                    const float wr = model->bounding_half_diag * max_scale_factor(mc.transform);
                    local_dynamic_casters_.push_back({wc, wr});
                }
                for (const auto& rs : skinned_resolved_) {
                    local_dynamic_casters_.push_back({rs.world_center, rs.world_radius});
                }
                render::lighting::LocalShadowConfig ls_cfg;
                ls_cfg.point_budget = gfx.local_shadow_point_budget;
                ls_cfg.spot_budget = gfx.local_shadow_spot_budget;
                local_shadows_->update(frame_point_lights_, frame_spot_lights_, camera.position,
                                       local_dynamic_casters_, ls_cfg);
            } else {
                // Local shadows off/unavailable: force every slot unshadowed
                // so stray host-side values can never index the atlas.
                for (auto& l : frame_point_lights_) l.shadow_slot = -1;
                for (auto& l : frame_spot_lights_) l.shadow_slot = -1;
            }

            uint32_t cw = static_cast<uint32_t>(std::max(context_->width(), 1));
            uint32_t ch = static_cast<uint32_t>(std::max(context_->height(), 1));
            cluster_grid_.begin_frame(camera, cw, ch, 0.1f, gfx.get_draw_distance());
            for (const auto& l : frame_point_lights_) {
                (void)cluster_grid_.add_point_light(l); // refusals are counted below
            }
            for (const auto& l : frame_spot_lights_) {
                (void)cluster_grid_.add_spot_light(l);
            }
            cluster_grid_.build();
            frame_cluster_params_ = cluster_grid_.params();
            frame_diagnostics_.lights_dropped += cluster_grid_.lights_dropped();
            frame_diagnostics_.cluster_lights_dropped += cluster_grid_.cluster_lights_dropped();
            if (collect_stats_) {
                render_stats_.lights_submitted +=
                    static_cast<uint32_t>(frame_point_lights_.size() + frame_spot_lights_.size());
                render_stats_.lights_binned += cluster_grid_.total_lights();
            }
        } else if (cluster_grid_ready_) {
            // No dynamic lights: publish zeroed params so the shaders see
            // gridDim.w == 0 instead of last frame's stale light count.
            frame_cluster_params_ = {};
        }

        // ===== One copy pass for every renderer-owned per-frame upload =====
        // Instances, shadow instances, bone palettes, grass chunks and cluster
        // light buffers all batch into a single SDL_GPUCopyPass, recorded
        // before any render pass begins (copy passes cannot nest in them).
        if (SDL_GPUCommandBuffer* up_cmd = context_->current_command_buffer()) {
            BRACKEN_PROFILE_ZONE_COLOR("FrameUploads", 0x27AE60);
            BRACKEN_GPU_SCOPE(up_cmd, "FrameUploads");
            BRACKEN_GPU_TIMER(gpu_timer_pool_, up_cmd, "FrameUploads");
            SDL_GPUCopyPass* copy_pass = SDL_BeginGPUCopyPass(up_cmd);
            if (copy_pass) {
                upload_instance_buffers(copy_pass);
                upload_bone_palettes(copy_pass, frame);
                effects_->upload_particle_instances(copy_pass, frame.effect_emitters(), frame.effect_particles());
                upload_debug_lines(frame, copy_pass);
                if (grass_active) {
                    grass_renderer_->upload_chunks(copy_pass, camera.position, &frame_frustum_);
                }
                if (cluster_upload_needed) {
                    cluster_grid_.upload(copy_pass);
                }
                if (frame_local_shadows_active_) {
                    local_shadows_->upload(copy_pass); // slot SSBO, only when dirty
                }
                SDL_EndGPUCopyPass(copy_pass);
            } else {
                BRACKEN_LOG_ERROR(LOG_TAG, "render_frame: Failed to begin frame upload copy pass");
            }
        }

        // ===== Sky IBL upkeep: refresh the prefilter chain when the sky-
        // relevant environment fields changed, amortized over frames; a
        // no-change frame records nothing. Runs before the forward pass so
        // same-frame draws sample the newest faces. =====
        if (frame_ibl_active_) {
            if (SDL_GPUCommandBuffer* ibl_cmd = context_->current_command_buffer()) {
                BRACKEN_PROFILE_ZONE_COLOR("IBL", 0x16A085);
                BRACKEN_GPU_SCOPE(ibl_cmd, "IBL");
                BRACKEN_GPU_TIMER(gpu_timer_pool_, ibl_cmd, "IBL");
                ibl_->update(ibl_cmd, *pipeline_registry_, env_, world_->sky_time());
            }
        }

        // ===== Water simulation: fixed-substep wave equation (compute).
        // Outside every render pass; before the forward pass so this frame's
        // surface uses this frame's ripples. =====
        if (water_ && water_->active()) {
            if (SDL_GPUCommandBuffer* sim_cmd = context_->current_command_buffer()) {
                BRACKEN_PROFILE_ZONE_COLOR("WaterSim", 0x2980B9);
                BRACKEN_GPU_TIMER(gpu_timer_pool_, sim_cmd, "WaterSim");
                water_->simulate(sim_cmd, dt);
            }
        }

        if (shadows_on) {
            render_shadow_passes();
        }

        // Local (point/spot) shadow tiles: independent of the directional
        // CSM toggle; only the tiles the selector marked dirty re-render.
        if (frame_local_shadows_active_) {
            render_local_shadow_passes(frame, camera);
        }

        // Cache shadow uniforms (cascade matrices are stable after update()).
        if (shadow_map_->is_ready()) {
            frame_shadow_uniforms_ = shadow_map_->get_shadow_uniforms(static_cast<int>(gfx.shadow_mode));
        }

        if (cluster_grid_ready_) {
            SDL_GPUBuffer* local_slots =
                (local_shadows_ && local_shadows_->is_ready()) ? local_shadows_->slot_buffer() : nullptr;
            terrain_->set_cluster_lighting(cluster_grid_.light_data_buffer(), cluster_grid_.cluster_offsets_buffer(),
                                           cluster_grid_.light_indices_buffer(), &frame_cluster_params_,
                                           sizeof(frame_cluster_params_), local_slots);
            if (grass_renderer_) {
                grass_renderer_->set_cluster_lighting(cluster_grid_.light_data_buffer(),
                                                      cluster_grid_.cluster_offsets_buffer(),
                                                      cluster_grid_.light_indices_buffer(), &frame_cluster_params_,
                                                      sizeof(frame_cluster_params_), local_slots);
            }
        }

        SDL_GPUCommandBuffer* cmd = context_->current_command_buffer();

        if (use_hdr_path) {
            // === HDR PATH: prepass + GTAO + offscreen 3D + post chain;
            // swapchain acquired late ===

            // Resize post targets from the window size; the swapchain isn't
            // acquired until composite so the GPU can start on the offscreen
            // pass without waiting for a presentable image.
            post_->resize(context_->width(), context_->height());
            gtao_->set_half_res(gfx.ao_mode != AOMode::GTAO_Full); // invalidates history on change
            gtao_->resize(context_->width(), context_->height());  // invalidates history on change
            bloom_->resize(context_->width(), context_->height());
            volumetric_fog_->resize(context_->width(), context_->height());
            taa_->resize(context_->width(), context_->height()); // invalidates history on change

            // Velocity target tracks the offscreen depth target's size (both
            // are the depth-prepass MRT attachments).
            {
                const int vw = context_->width();
                const int vh = context_->height();
                if (!velocity_texture_ || velocity_texture_->width() != vw || velocity_texture_->height() != vh) {
                    velocity_texture_ = gpu::GPUTexture::create_2d(
                        context_->device(), vw, vh, pipeline_registry_->get_velocity_format(),
                        SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);
                    if (!velocity_texture_) {
                        BRACKEN_LOG_WARN(LOG_TAG, "render_frame: failed to create velocity texture (prepass disabled)");
                    }
                }
            }

            // Offscreen clear tracks the environment's horizon fog color so
            // unrendered sky pixels melt into the atmosphere (linear HDR).
            post_->set_clear_color(env_.fog_color_horizon);

            // ===== Depth prepass: velocity + scene depth for opaque draws =====
            // (MASK/BLEND/skybox/particles keep their forward depth behavior.)
            frame_prepass_active_ = render_depth_prepass(frame, draw_camera);

            // ===== GTAO: computed from the prepass depth BEFORE the forward
            // pass — the forward shaders sample the full-res AO texture in
            // their ambient term. Chain: GTAO -> temporal accumulation ->
            // bilateral blur -> joint-bilateral upsample (render::GTAO). =====
            const bool ao_active = gfx.ao_mode != AOMode::Off && gtao_->is_ready() && frame_prepass_active_ &&
                                   velocity_texture_ && post_->offscreen_depth() && cmd;
            if (ao_active) {
                BRACKEN_PROFILE_ZONE_COLOR("GTAO", 0x16A085);
                BRACKEN_GPU_SCOPE(cmd, "GTAO");
                BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "GTAO");
                render::GTAO::FrameParams ao_params;
                // The draw (jittered) projection: it must match the depth
                // buffer the prepass just wrote.
                ao_params.projection = draw_camera.projection;
                ao_params.inv_projection = cached_inv_projection_;
                ao_params.radius_scale = gfx.ao_radius;
                ao_params.strength = gfx.ao_strength;
                ao_params.frame_index = frame_index_;
                gtao_->render(cmd, *pipeline_registry_, *post_->offscreen_depth(), *velocity_texture_, ao_params);
                frame_ao_texture_ = gtao_->ao_texture();
            } else {
                // AO off / prepass unavailable: stale history would smear when
                // AO comes back — drop it (mirrors TAA's invalidation policy).
                gtao_->invalidate_history();
            }

            {
                BRACKEN_PROFILE_ZONE_COLOR("HDROffscreen", 0xC0392B);
                BRACKEN_GPU_SCOPE(cmd, "HDROffscreen");
                BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "HDROffscreen");
                // After the prepass the depth buffer already holds this
                // frame's opaque depth — load it (forward opaques test EQUAL).
                main_render_pass_ = post_->begin_offscreen_pass(cmd, frame_prepass_active_);
                had_main_pass_this_frame_ = (main_render_pass_ != nullptr);

                if (main_render_pass_ && cmd) {
                    render_3d_scene(frame, draw_camera);
                    post_->end_offscreen_pass();
                    main_render_pass_ = nullptr;
                }
            }

            // ===== Water: its own pass after the opaque scene + skybox.
            // The HDR color/depth are snapshotted first (what refraction and
            // shoreline softness sample); the live targets are then re-bound
            // with LOAD so the surface depth-tests against — and writes —
            // real scene depth, keeping TAA reprojection and the fog's depth
            // reads consistent. Runs before TAA so the water is temporally
            // stabilized like everything else. Known limitation: transparents
            // and particles drew before this pass, so water overdraws them
            // where they overlap on screen. =====
            if (water_ && water_->active() && cmd && post_->offscreen_color() && post_->offscreen_depth()) {
                BRACKEN_PROFILE_ZONE_COLOR("Water", 0x2980B9);
                BRACKEN_GPU_SCOPE(cmd, "Water");
                BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "Water");
                const int w = context_->width();
                const int h = context_->height();
                auto ensure_copy = [&](std::unique_ptr<gpu::GPUTexture>& tex, SDL_GPUTextureFormat fmt) {
                    if (!tex || tex->width() != w || tex->height() != h) {
                        tex = gpu::GPUTexture::create_2d(context_->device(), w, h, fmt,
                                                         SDL_GPU_TEXTUREUSAGE_SAMPLER);
                    }
                    return tex != nullptr;
                };
                if (ensure_copy(water_scene_copy_, pipeline_registry_->get_scene_color_format()) &&
                    ensure_copy(water_depth_copy_, SDL_GPU_TEXTUREFORMAT_D32_FLOAT)) {
                    if (SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(cmd)) {
                        auto copy_tex = [&](SDL_GPUTexture* src, SDL_GPUTexture* dst) {
                            SDL_GPUTextureLocation from = {};
                            from.texture = src;
                            SDL_GPUTextureLocation to = {};
                            to.texture = dst;
                            SDL_CopyGPUTextureToTexture(copy, &from, &to, static_cast<Uint32>(w),
                                                        static_cast<Uint32>(h), 1, false);
                        };
                        copy_tex(post_->offscreen_color()->handle(), water_scene_copy_->handle());
                        copy_tex(post_->offscreen_depth()->handle(), water_depth_copy_->handle());
                        SDL_EndGPUCopyPass(copy);
                    }

                    SDL_GPUColorTargetInfo color = {};
                    color.texture = post_->offscreen_color()->handle();
                    color.load_op = SDL_GPU_LOADOP_LOAD;
                    color.store_op = SDL_GPU_STOREOP_STORE;
                    SDL_GPUDepthStencilTargetInfo depth = {};
                    depth.texture = post_->offscreen_depth()->handle();
                    depth.load_op = SDL_GPU_LOADOP_LOAD;
                    depth.store_op = SDL_GPU_STOREOP_STORE;
                    depth.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
                    depth.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;
                    if (SDL_GPURenderPass* water_pass = SDL_BeginGPURenderPass(cmd, &color, 1, &depth)) {
                        SDL_GPUTextureSamplerBinding ibl_bindings[3];
                        const bool have_ibl = fill_ibl_bindings(ibl_bindings);
                        const SDL_GPUTextureSamplerBinding specular =
                            have_ibl ? ibl_bindings[1]
                                     : SDL_GPUTextureSamplerBinding{dummy_black_cube_->handle(), default_sampler_};
                        water_->render(water_pass, cmd, draw_camera.view_projection, draw_camera.projection,
                                       camera.position, scene_time_, w, h, water_scene_copy_->handle(),
                                       water_depth_copy_->handle(), specular, frame_ibl_active_,
                                       gfx.sun_intensity);
                        SDL_EndGPURenderPass(water_pass);
                        if (collect_stats_) {
                            render_stats_.draw_calls += 1;
                        }
                    }
                }
            }

            // ===== TAA resolve (before bloom/composite so the whole post
            // chain reads the temporally stabilized image). Reads the jittered
            // HDR offscreen color + prepass velocity + scene depth, accumulates
            // into the TAA history, then writes the (sharpened) result back
            // over the offscreen color target. =====
            if (taa_active) {
                if (frame_prepass_active_ && velocity_texture_ && cmd && post_->offscreen_color() &&
                    post_->offscreen_depth()) {
                    BRACKEN_PROFILE_ZONE_COLOR("TAA", 0x16A085);
                    BRACKEN_GPU_SCOPE(cmd, "TAA");
                    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "TAA");
                    render::TAA::ResolveParams taa_params;
                    // prev_view_proj_unjittered_ still holds frame N-1 here —
                    // the motion-vector bookkeeping below updates it after.
                    taa_params.curr_inv_view_proj_unjittered = glm::inverse(frame_cam_.view_proj_unjittered);
                    taa_params.prev_view_proj_unjittered = prev_view_proj_unjittered_;
                    taa_params.camera_pos = camera.position;
                    taa_->render(cmd, *pipeline_registry_, *post_->offscreen_color(), *velocity_texture_,
                                 *post_->offscreen_depth(), taa_params);
                } else {
                    // Jittered frame without a resolve (prepass/velocity
                    // unavailable): history would ghost against it next frame.
                    taa_->invalidate_history();
                }
            }

            // ===== Auto exposure (compute): 256-bin log2-luminance histogram
            // + adaptation over the TAA-resolved (pre-bloom) HDR scene color.
            // The composite pass reads the adapted EV from the state SSBO. =====
            const bool auto_exposure_active =
                gfx.auto_exposure && auto_exposure_->is_ready() && post_->offscreen_color() && cmd;
            if (auto_exposure_active) {
                BRACKEN_PROFILE_ZONE_COLOR("AutoExposure", 0x16A085);
                BRACKEN_GPU_SCOPE(cmd, "AutoExposure");
                BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "AutoExposure");
                render::AutoExposure::FrameParams ae_params;
                ae_params.dt = dt;
                ae_params.speed_up = gfx.auto_exposure_speed_up;
                ae_params.speed_down = gfx.auto_exposure_speed_down;
                ae_params.min_ev = gfx.auto_exposure_min_ev;
                ae_params.max_ev = gfx.auto_exposure_max_ev;
                auto_exposure_->render(cmd, *pipeline_registry_, *post_->offscreen_color(), ae_params);
            } else {
                // Off/unavailable for a frame: the adapted EV is stale history
                // — snap to the metered target when it comes back (mirrors
                // the TAA/GTAO invalidation policy).
                auto_exposure_->invalidate();
            }

            // Bloom: downsample + upsample chain on offscreen color
            SDL_GPUTexture* bloom_tex = nullptr;
            if (gfx.bloom_enabled && bloom_->is_ready()) {
                SDL_GPUTexture* scene_in = post_->offscreen_color() ? post_->offscreen_color()->handle() : nullptr;
                if (cmd && scene_in) {
                    BRACKEN_PROFILE_ZONE_COLOR("Bloom", 0x16A085);
                    BRACKEN_GPU_SCOPE(cmd, "Bloom");
                    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "Bloom");
                    bloom_->render(cmd, *pipeline_registry_, scene_in, gfx.bloom_threshold, gfx.exposure);
                }
                bloom_tex = bloom_->bloom_texture();
            }

            // Volumetric fog: ray march through fog volume using depth + shadow map.
            // God rays read the shadow atlas, which only has valid contents when
            // the shadow passes actually ran this frame (shadows_on) — with
            // shadow_mode == 0 the atlas is never rendered, so force them off.
            SDL_GPUTexture* fog_tex = nullptr;
            const bool god_rays_active = gfx.god_rays && shadows_on;
            if ((gfx.volumetric_fog || god_rays_active) && volumetric_fog_->is_ready() && shadow_map_->is_ready() &&
                post_->offscreen_depth()) {
                BRACKEN_PROFILE_ZONE_COLOR("VolumetricFog", 0x16A085);
                BRACKEN_GPU_SCOPE(cmd, "VolumetricFog");
                BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "VolumetricFog");
                // Default fog_density is 0.05; treat as a multiplier on the
                // environment's base density (env_.fog_volumetric_density).
                float fog_density_mul = gfx.fog_density / 0.05f;
                volumetric_fog_->render(cmd, *pipeline_registry_, post_->offscreen_depth()->handle(), *shadow_map_,
                                        draw_camera, env_, god_rays_active, gfx.volumetric_fog, fog_density_mul);
                fog_tex = volumetric_fog_->fog_texture();
            }

            // Late swapchain acquire: only composite/FXAA + UI need the image.
            current_swapchain_ = context_->acquire_swapchain_texture(cmd);
            if (current_swapchain_ && cmd) {
                render::PostProcessStack::CompositeParams composite_params;
                composite_params.bloom_strength = gfx.bloom_enabled ? gfx.bloom_strength : 0.0f;
                composite_params.exposure = gfx.exposure;
                composite_params.tonemap_mode = static_cast<int>(gfx.tonemap_mode);
                composite_params.contrast = gfx.contrast;
                composite_params.saturation = gfx.saturation;
                // Depth linearization constants for the joint bilateral fog
                // upsample in the composite shader. viewZ = |B / (depth + A)|
                // holds for both standard and reversed-Z RH_ZO projections as
                // long as A/B come from the actual projection matrix.
                composite_params.depth_a = draw_camera.projection[2][2];
                composite_params.depth_b = draw_camera.projection[3][2];
                // Auto exposure: adapted-EV SSBO from the compute passes above
                // (gfx.exposure stays the manual bias on top).
                composite_params.auto_exposure = auto_exposure_active;
                composite_params.exposure_buffer =
                    auto_exposure_active ? auto_exposure_->state_buffer() : nullptr;
                // Color grading is a LOOK — it lives in the environment (live-
                // editable like the sky/fog colors), not in GraphicsSettings.
                composite_params.white_balance =
                    render::white_balance_coefficients(env_.grade_temperature, env_.grade_tint);
                composite_params.cdl_slope = env_.grade_slope;
                composite_params.cdl_offset = env_.grade_offset;
                composite_params.cdl_power = env_.grade_power;

                // FXAA runs on the tonemapped LDR image: composite into the
                // intermediate LDR target, then FXAA resolves to the swapchain.
                // aa_mode == 0 composites straight to the swapchain as before.
                SDL_GPUTexture* ldr = (gfx.aa_mode == AAMode::FXAA) ? post_->ldr_target() : nullptr;
                SDL_GPUTexture* composite_target = ldr ? ldr : current_swapchain_;

                {
                    BRACKEN_PROFILE_ZONE_COLOR("Composite", 0x16A085);
                    BRACKEN_GPU_SCOPE(cmd, "Composite");
                    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "Composite");
                    post_->render_composite_pass(cmd, *pipeline_registry_, composite_target, bloom_tex, fog_tex,
                                               composite_params);
                }

                if (ldr) {
                    BRACKEN_PROFILE_ZONE_COLOR("FXAA", 0x16A085);
                    BRACKEN_GPU_SCOPE(cmd, "FXAA");
                    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "FXAA");
                    post_->render_fxaa_pass(cmd, current_swapchain_);
                }
            } else {
                swapchain_unavailable = true;
            }
        } else {
            // === NORMAL PATH: render directly to swapchain ===
            // (No depth prepass here: frame_prepass_active_ stays false, so
            // every forward draw keeps its GREATER_OR_EQUAL write-on pipeline.)

            BRACKEN_PROFILE_ZONE_COLOR("MainScene", 0xC0392B);
            BRACKEN_GPU_SCOPE(cmd, "MainScene");
            BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "MainScene");
            begin_main_pass();

            if (main_render_pass_ && context_->current_command_buffer()) {
                render_3d_scene(frame, draw_camera);
            }

            end_main_pass();
        }

        // ===== Motion-vector bookkeeping (CPU, once per rendered frame) =====
        // Record every submitted command carrying a motion_id; ids not
        // re-submitted this frame are pruned by end_frame(). Skinned entries
        // record this frame's palette offset — after the double-buffer swap
        // below it indexes next frame's *previous* palette buffer.
        for (const auto& mc : frame.model_commands()) {
            if (mc.motion_id != 0) {
                motion_history_.record(mc.motion_id, mc.transform);
            }
        }
        for (const auto& rs : skinned_resolved_) {
            if (rs.cmd->motion_id != 0) {
                motion_history_.record(rs.cmd->motion_id, rs.cmd->transform, rs.palette_offset);
            }
        }
        motion_history_.end_frame();
        prev_view_proj_unjittered_ = frame_cam_.view_proj_unjittered;
        have_prev_view_proj_ = true;
        // Palette double-buffer swap: this frame's palettes become next
        // frame's "previous" palettes (only valid if anything was uploaded).
        prev_palette_buffer_valid_ =
            bone_palette_matrix_count_ > 0 && bone_palette_buffers_[bone_palette_current_] != nullptr;
        bone_palette_current_ ^= 1;
        ++frame_index_; // advances the Halton jitter sequence
    }

    if (!swapchain_unavailable) {
        if (SDL_GPUCommandBuffer* ui_cmd = context_->current_command_buffer()) {
            BRACKEN_PROFILE_ZONE_COLOR("UI", 0x16A085);
            BRACKEN_GPU_SCOPE(ui_cmd, "UI");
            BRACKEN_GPU_TIMER(gpu_timer_pool_, ui_cmd, "UI");
            begin_ui();
            render_ui_commands(frame);
            for (const auto& billboard : frame.billboards()) {
                draw_billboard_3d(billboard, camera);
            }
            end_ui();
        }

        // current_swapchain_ can still be null here (begin_ui acquire failure
        // on the no-content / fallback paths); the callback renders into the
        // swapchain, so skip it rather than hand it a null target.
        if (post_ui_callback_ && current_swapchain_) {
            if (SDL_GPUCommandBuffer* post_cmd = context_->current_command_buffer()) {
                BRACKEN_PROFILE_ZONE_COLOR("PostUICallback", 0x16A085);
                BRACKEN_GPU_SCOPE(post_cmd, "PostUICallback");
                BRACKEN_GPU_TIMER(gpu_timer_pool_, post_cmd, "PostUICallback");
                post_ui_callback_(post_cmd, current_swapchain_);
            }
        }
    }

    if (SDL_GPUCommandBuffer* end_cmd = context_->current_command_buffer()) {
        gpu_timer_pool_.end_frame(end_cmd);
    }
    if (collect_stats_) {
        if (local_shadows_) {
            render_stats_.local_shadow_tiles_rendered =
                static_cast<uint32_t>(local_shadows_->selector().tiles_rendered_last_frame());
        }
        gpu_timer_pool_.read_results(render_stats_.pass_times_ms);
        BRACKEN_PROFILE_PLOT("render.draw_calls", static_cast<int64_t>(render_stats_.draw_calls));
        BRACKEN_PROFILE_PLOT("render.triangles", static_cast<int64_t>(render_stats_.triangle_count));
        BRACKEN_PROFILE_PLOT("render.entities", static_cast<int64_t>(render_stats_.entities_rendered));
    }
    if (!pending_screenshot_path_.empty() && current_swapchain_ && context_->current_command_buffer()) {
        capture_screenshot_and_end_frame();
    } else {
        end_frame();
    }
}

void SceneRenderer::capture_screenshot_and_end_frame() {
    const std::string path = std::move(pending_screenshot_path_);
    pending_screenshot_path_.clear();

    const uint32_t w = static_cast<uint32_t>(context_->width());
    const uint32_t h = static_cast<uint32_t>(context_->height());
    gpu::GPUDevice& device = context_->device();

    SDL_GPUTransferBufferCreateInfo tb_info{};
    tb_info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
    tb_info.size = w * h * 4;
    SDL_GPUTransferBuffer* transfer = device.create_transfer_buffer(tb_info);
    if (!transfer) {
        BRACKEN_LOG_ERROR(LOG_TAG, "Screenshot: failed to create download buffer: {}", SDL_GetError());
        end_frame();
        return;
    }

    // Record the download on this frame's command buffer, then submit with a
    // fence and block — the transfer buffer is only coherent after the GPU is
    // done. Copy passes can't nest in render passes; all passes ended above.
    SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(context_->current_command_buffer());
    SDL_GPUTextureRegion src{};
    src.texture = current_swapchain_;
    src.w = w;
    src.h = h;
    src.d = 1;
    SDL_GPUTextureTransferInfo dst{};
    dst.transfer_buffer = transfer;
    SDL_DownloadFromGPUTexture(copy, &src, &dst);
    SDL_EndGPUCopyPass(copy);

    const SDL_GPUTextureFormat gpu_format = context_->swapchain_format();
    current_swapchain_ = nullptr;
    if (!context_->end_frame_and_wait()) {
        device.release_transfer_buffer(transfer);
        return;
    }

    SDL_PixelFormat pixel_format;
    switch (gpu_format) {
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
        pixel_format = SDL_PIXELFORMAT_BGRA32;
        break;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
        pixel_format = SDL_PIXELFORMAT_RGBA32;
        break;
    default:
        BRACKEN_LOG_WARN(LOG_TAG, "Screenshot: unexpected swapchain format {}, assuming RGBA8",
                         static_cast<int>(gpu_format));
        pixel_format = SDL_PIXELFORMAT_RGBA32;
        break;
    }

    bool saved = false;
    if (void* pixels = device.map_transfer_buffer(transfer, false)) {
        SDL_Surface* surface =
            SDL_CreateSurfaceFrom(static_cast<int>(w), static_cast<int>(h), pixel_format, pixels, static_cast<int>(w * 4));
        if (surface) {
            // RGB24 drops the alpha channel — swapchain alpha is undefined and
            // would render the PNG transparent in viewers.
            if (SDL_Surface* rgb = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_RGB24)) {
                saved = IMG_SavePNG(rgb, path.c_str());
                SDL_DestroySurface(rgb);
            }
            SDL_DestroySurface(surface);
        }
        device.unmap_transfer_buffer(transfer);
    }
    device.release_transfer_buffer(transfer);

    if (saved) {
        BRACKEN_LOG_INFO(LOG_TAG, "SceneRenderer: screenshot saved to '{}' ({}x{})", path.c_str(), w, h);
    } else {
        BRACKEN_LOG_ERROR(LOG_TAG, "Screenshot: failed to save '{}': {}", path, SDL_GetError());
    }
}

// ============================================================================
// 3D Scene Rendering (shared between the HDR and fallback paths)
// ============================================================================

// ============================================================================
// Model Rendering
// ============================================================================

// ============================================================================
// Instanced Rendering
// ============================================================================

void SceneRenderer::build_instance_batches(const FramePacket& frame, const CameraState& camera,
                                           const Frustum& frustum, int active_cascades) {
    BRACKEN_PROFILE_ZONE_COLOR("SceneRenderer::build_instance_batches", 0x27AE60);

    // `active_cascades` comes from the frame's RenderPlan, which is also what
    // gates the shadow passes themselves — with shadows off the cascade
    // frustums are stale, so nothing may be binned, packed or uploaded.
    //
    // Thin adapter, and that is the whole job: gather the renderer's members
    // into ScenePrep's params/outputs and call across the tier boundary. This
    // function touches renderer state; nothing on the other side may.
    ScenePrep::PrepParams p;
    p.models = model_manager_.get();
    p.motion_history = &motion_history_;
    p.camera_pos = camera.position;
    p.main_frustum = &frustum;
    p.cascade_frustums = shadow_cascade_frustums_.data();
    p.active_cascades = active_cascades;
    p.draw_dist_sq = frame_draw_dist_sq_;
    p.do_frustum_cull = frame_do_frustum_cull_;
    p.collect_stats = collect_stats_;

    // Cull stats accumulate into per-chunk partials, merged once after the
    // fan-out (per-chunk merge discipline — see FrameStatsPartial).
    FrameStatsPartial stats_partial;
    const ScenePrep::InstanceBinOutputs out{instance_batches_,       instance_batches_touched_,
                                            shadow_instance_batches_, shadow_batches_touched_,
                                            non_instanced_commands_,  stats_partial};
    ScenePrep::prep_instance_batches(core::jobs::JobSystem::instance(), frame.model_commands(), p, prep_scratch_,
                                     out);
    stats_partial.merge_into(render_stats_, frame_diagnostics_);
}

// ============================================================================
// Bone Palettes
// ============================================================================

void SceneRenderer::pack_bone_palettes(const FramePacket& frame, const CameraState& camera) {
    BRACKEN_PROFILE_ZONE_COLOR("SceneRenderer::pack_bone_palettes", 0x8E44AD);
    const std::span<const SkinnedDrawCommand> cmds = frame.skinned_commands();
    skinned_resolved_.clear();
    bone_palette_matrix_count_ = 0;
    if (cmds.empty()) {
        return;
    }

    // The packet owns the palettes: matrices live in packet memory, offsets
    // were prefix-summed at seal, and the MAX_BONE_PALETTES budget was
    // enforced at build time (FrameBuilder::alloc_bone_palette) — no
    // renderer-side dedupe, cap, or copy plan anymore. The whole palette
    // stream uploads in palettes() order (upload_bone_palettes), so each
    // draw's GPU palette offset is exactly its record's matrix_offset.
    bone_palette_matrix_count_ = frame.palette_matrix_total();

    ScenePrep::PrepParams p;
    p.models = model_manager_.get();
    p.motion_history = &motion_history_;
    p.camera_pos = camera.position;
    p.main_frustum = &frame_frustum_;
    p.draw_dist_sq = frame_draw_dist_sq_;
    p.do_frustum_cull = frame_do_frustum_cull_;
    p.collect_stats = collect_stats_;
    p.prev_palette_valid = prev_palette_buffer_valid_ && prev_bone_palette_buffer() != nullptr;

    // Skinned cull stats accumulate into per-chunk partials, merged once
    // after the fan-out (per-chunk merge discipline — see FrameStatsPartial).
    FrameStatsPartial stats_partial;
    ScenePrep::prep_resolve_skinned(core::jobs::JobSystem::instance(), cmds, frame.palettes(), p,
                                    prep_scratch_.skinned, skinned_resolved_, stats_partial);
    stats_partial.merge_into(render_stats_, frame_diagnostics_);
}

} // namespace bracken::scene
