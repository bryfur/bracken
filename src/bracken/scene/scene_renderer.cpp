#include "scene_renderer.hpp"
#include "bracken/animation/animation_types.hpp" // MAX_BONES
#include "bracken/core/assert.hpp"
#include "bracken/core/color.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/core/jobs/task_group.hpp"
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
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>

#include "bracken/graphics_settings.hpp"
#include "bracken/scene/camera_jitter.hpp"
#include "bracken/scene/camera_state.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/transform_math.hpp"
#include "glm/common.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"
#include "glm/matrix.hpp"

namespace bracken::scene {

namespace gpu = bracken::gpu;
namespace render = bracken::render;

// Fill the per-mesh material fields shared by ModelLightingUniforms and
// InstancedLightingUniforms (identical field names in both structs).
template <typename U>
static inline void apply_mesh_material(U& u, const Mesh& mesh) {
    const PBRMaterial& mat = mesh.material;
    u.hasTexture = (mesh.has_texture && mesh.texture) ? 1 : 0;
    u.hasNormalMap = mat.normal_texture ? 1 : 0;
    u.normalScale = mat.normal_scale;
    u.baseColorFactor = mat.base_color_factor;
    u.metallicFactor = mat.metallic_factor;
    u.roughnessFactor = mat.roughness_factor;
    u.occlusionStrength = mat.occlusion_strength;
    u.hasMRMap = mat.metallic_roughness_texture ? 1.0f : 0.0f;
    u.emissiveFactor = mat.emissive_factor;
    u.hasEmissiveMap = mat.emissive_texture ? 1.0f : 0.0f;
    u.hasPackedOcclusion = mat.has_packed_occlusion ? 1.0f : 0.0f;
    // glTF alphaMode: only MASK materials alpha-test; OPAQUE/BLEND must not
    // discard, which a cutoff of 0 guarantees in the shader.
    u.alphaCutoff = (mat.alpha_mode == AlphaMode::Mask) ? mat.alpha_cutoff : 0.0f;
    // Screen-space AO comes from the depth prepass. OPAQUE meshes are always
    // in it, and MASK meshes join via the alpha-tested prepass variants (so
    // foliage both receives and casts screen AO — leaf-on-leaf occlusion is
    // what separates a canopy from a flat blob). BLEND meshes write no
    // prepass depth, so the AO texel under them describes the geometry
    // behind them; they skip the AO term. Skinned MASK meshes are the one
    // gap (no masked skinned prepass variant) — they'd sample the AO of
    // whatever is behind them, which for a moving character is benign.
    u.screenAOEnabled = (mat.alpha_mode != AlphaMode::Blend) ? 1.0f : 0.0f;
}

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
    static_assert(MAX_SHADOW_CASCADES == render::CSM_MAX_CASCADES,
                  "SceneRenderer::MAX_SHADOW_CASCADES must match render::CSM_MAX_CASCADES");
    context_ = &context;

    if (!pipeline_registry_->init(context_->device())) {
        std::cerr << "Failed to initialize pipeline registry" << '\n';
        return false;
    }
    pipeline_registry_->set_swapchain_format(context_->swapchain_format());

    model_manager_->set_device(&context_->device());

    if (!terrain_->init(context_->device(), *pipeline_registry_, world_width, world_height)) {
        std::cerr << "Failed to initialize terrain renderer" << '\n';
        return false;
    }

    if (!world_->init(context_->device(), *pipeline_registry_)) {
        std::cerr << "Failed to initialize world renderer" << '\n';
        return false;
    }

    int w = context_->width();
    int h = context_->height();

    if (!ui_->init(context_->device(), *pipeline_registry_, w, h)) {
        std::cerr << "Failed to initialize UI renderer" << '\n';
        return false;
    }

    if (!effects_->init(context_->device(), *pipeline_registry_, model_manager_.get())) {
        std::cerr << "Failed to initialize effect renderer" << '\n';
        return false;
    }

    depth_texture_ = gpu::GPUTexture::create_depth(context_->device(), w, h);
    if (!depth_texture_) {
        std::cerr << "Failed to create depth texture" << '\n';
        return false;
    }

    init_pipelines();

    // Preload all pipelines upfront to avoid hitching during gameplay
    if (!pipeline_registry_->preload_all_pipelines()) {
        std::cerr << "Warning: Some pipelines failed to preload" << '\n';
    }

    if (grass_renderer_ && !grass_renderer_->init(context_->device(), *pipeline_registry_, world_width, world_height)) {
        std::cerr << "Warning: Failed to initialize grass renderer (grass disabled)" << '\n';
    }

    const GraphicsSettings& gfx = graphics_ ? *graphics_ : default_graphics_;
    static constexpr int resolution_table[] = {512, 1024, 2048, 4096};
    int shadow_res = resolution_table[std::clamp(gfx.shadow_resolution, 0, 3)];
    shadow_map_->set_active_cascades(gfx.shadow_cascades + 1);
    if (!shadow_map_->init(context_->device(), shadow_res)) {
        std::cerr << "Warning: Failed to initialize shadow map (shadows disabled)" << '\n';
    }

    static constexpr int local_shadow_resolution_table[] = {1024, 2048, 4096};
    int local_shadow_res = local_shadow_resolution_table[std::clamp(gfx.local_shadow_resolution, 0, 2)];
    if (!local_shadows_->init(context_->device(), local_shadow_res)) {
        std::cerr << "Warning: Failed to initialize local shadow map (point/spot shadows disabled)" << '\n';
    }

    if (!post_->init(context_->device(), w, h)) {
        std::cerr << "Warning: Failed to initialize post-process stack (HDR path disabled)" << '\n';
    }

    if (!gtao_->init(context_->device(), w, h, gfx.ao_mode != AOMode::GTAO_Full)) {
        std::cerr << "Warning: Failed to initialize GTAO (AO disabled)" << '\n';
    }

    if (!bloom_->init(context_->device(), w, h)) {
        std::cerr << "Warning: Failed to initialize Bloom (bloom disabled)" << '\n';
    }

    if (!volumetric_fog_->init(context_->device(), w, h)) {
        std::cerr << "Warning: Failed to initialize Volumetric Fog (fog disabled)" << '\n';
    }

    if (!taa_->init(context_->device(), w, h)) {
        std::cerr << "Warning: Failed to initialize TAA (temporal AA disabled)" << '\n';
    }

    if (!auto_exposure_->init(context_->device())) {
        std::cerr << "Warning: Failed to initialize AutoExposure (histogram exposure disabled)" << '\n';
    }

    if (!ibl_->init(context_->device(), *pipeline_registry_)) {
        std::cerr << "Warning: Failed to initialize IBL (sky image-based lighting disabled)" << '\n';
    }

    if (!water_->init(context_->device(), *pipeline_registry_)) {
        std::cerr << "Warning: Failed to initialize WaterRenderer (water disabled)" << '\n';
    }

    // Timestamp pool is a documented stub (returns zeros); failure is harmless.
    (void)gpu_timer_pool_.init(context_->device(), 64);

    cluster_grid_ready_ = cluster_grid_.init(context_->device(), render::lighting::MAX_LIGHTS);
    if (!cluster_grid_ready_) {
        std::cerr << "Warning: Failed to initialize ClusterGrid (dynamic lighting disabled)" << '\n';
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
        std::cerr << "Warning: Some pipelines failed to preload" << '\n';
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
        std::cerr << "Warning: Failed to create default GPU sampler" << '\n';
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
    static constexpr int resolution_table[] = {512, 1024, 2048, 4096};
    int new_res = resolution_table[std::clamp(settings.shadow_resolution, 0, 3)];
    int new_cascades = settings.shadow_cascades + 1;

    if (new_res != shadow_map_->resolution()) {
        shadow_map_->reinit(new_res);
    }
    if (new_cascades != shadow_map_->active_cascades()) {
        shadow_map_->set_active_cascades(new_cascades);
    }

    // Local (point/spot) shadow atlas resolution — reinit drops every cached
    // tile, so lights re-render under the per-frame cap over the next frames.
    static constexpr int local_shadow_resolution_table[] = {1024, 2048, 4096};
    int new_local_res = local_shadow_resolution_table[std::clamp(settings.local_shadow_resolution, 0, 2)];
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
            std::cerr << "Warning: Failed to recreate default GPU sampler (anisotropic level: " << level << ")" << '\n';
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

    std::cout << "[Renderer] Heightmap set for terrain rendering" << '\n';
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
        std::cerr << "begin_main_pass: No active command buffer" << '\n';
        return;
    }

    uint32_t sw_width = 0;
    uint32_t sw_height = 0;
    current_swapchain_ = context_->acquire_swapchain_texture(cmd, &sw_width, &sw_height);
    if (!current_swapchain_) {
        std::cerr << "begin_main_pass: Failed to acquire swapchain texture" << '\n';
        return;
    }

    if (depth_texture_ && (depth_texture_->width() != static_cast<int>(sw_width) ||
                           depth_texture_->height() != static_cast<int>(sw_height))) {
        depth_texture_ = gpu::GPUTexture::create_depth(context_->device(), sw_width, sw_height);
        if (!depth_texture_) {
            std::cerr << "begin_main_pass: Failed to resize depth texture" << '\n';
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
        std::cerr << "begin_main_pass: Failed to begin render pass" << '\n';
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
        std::cerr << "begin_ui: No active command buffer" << '\n';
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
    if (collect_stats_) {
        render_stats_ = {};
        render_stats_.jobs_pending = core::jobs::JobSystem::instance().pending_count();
    }
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

    // Cache frame-level state once (avoids repeated ternary/function calls)
    frame_fog_active_ = gfx.fog_enabled;
    frame_draw_dist_sq_ = gfx.get_draw_distance() * gfx.get_draw_distance();
    frame_do_frustum_cull_ = gfx.frustum_culling;

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

    begin_frame();

    // Minimized windows can't acquire a swapchain image; skip all recording but
    // still submit the command buffer so the frame lifecycle stays balanced.
    if (context_->window() && (SDL_GetWindowFlags(context_->window()) & SDL_WINDOW_MINIMIZED) != 0) {
        end_frame();
        return;
    }

    gpu_timer_pool_.begin_frame();

    // 3D needs both content and a camera (no camera ever set = skip 3D,
    // conservative fallback for the packet's optional camera).
    bool has_content = frame.has_3d_content() && have_frame_camera_;
    frame_prepass_active_ = false; // set by the HDR path when the prepass records
    // Reset the frame's AO texture to the white dummy: forward pipelines
    // always bind the slot, and GTAO's textures may have been recreated
    // since last frame (resize / quality change).
    frame_ao_texture_ = dummy_white_texture_ ? dummy_white_texture_->handle() : nullptr;
    // Sky IBL: active when the setting is on and the prefilter chain is
    // alive; drives every scene shader's iblEnabled flag (the IBL sampler
    // slots are bound either way — dummies when inactive).
    frame_ibl_active_ = gfx.ibl_enabled && ibl_ && ibl_->is_ready();
    // Always render 3D through the HDR offscreen target so scene lighting stays
    // in linear HDR; the composite pass tonemaps + gamma-corrects to the swapchain.
    bool use_hdr_path = post_->is_ready();
    // TAA is active exactly when the settings ask for it AND the resolve can
    // actually run (offscreen HDR path + history targets alive). It drives the
    // sub-pixel camera jitter: jitter without a resolve to integrate the
    // samples would just shimmer, so every other mode renders unjittered.
    const bool taa_active = gfx.aa_mode == AAMode::TAA && use_hdr_path && taa_->is_ready();
    taa_jitter_enabled_ = taa_active;
    if (!taa_active) {
        // Any gap (mode switch, fallback path) stales the accumulated
        // history; drop it so re-enabling TAA restarts cleanly.
        taa_->invalidate_history();
    }
    // Auto exposure follows the same gap policy: any frame its passes cannot
    // record (setting off, fallback path, no 3D content) stales the adapted
    // EV — snap to the metered target when it comes back.
    if (!gfx.auto_exposure || !use_hdr_path || !has_content) {
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
        const bool shadows_on = shadow_map_->is_ready() && gfx.shadow_mode != ShadowMode::Off;
        if (shadows_on) {
            // Near plane recovered from the reversed-Z RH_ZO projection
            // (n = m32 / (m22 + 1); under standard-Z it was m32/m22).
            // Shadow range is a quality setting (clamped by draw distance);
            // the shader fades shadows out over the tail instead of popping
            // them off.
            const float cam_near = std::max(camera.projection[3][2] / (camera.projection[2][2] + 1.0f), 0.1f);
            const float shadow_far = std::min(gfx.get_draw_distance(), gfx.shadow_distance);
            shadow_map_->update(camera.view, camera.projection, light_dir_, cam_near, shadow_far);
            const int active = shadow_map_->active_cascades();
            for (int c = 0; c < active; ++c) {
                shadow_cascade_frustums_[c].extract_from_matrix(shadow_map_->cascades()[c].light_view_projection);
            }
        }

        // ===== CPU prep for all per-frame GPU uploads =====

        // Build instance batches (cull for main + per-cascade shadows) and
        // resolve the skinned commands against the packet-owned palettes.
        build_instance_batches(frame, camera, frame_frustum_);
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
            for (const auto& l : frame_point_lights_) cluster_grid_.add_point_light(l);
            for (const auto& l : frame_spot_lights_) cluster_grid_.add_spot_light(l);
            cluster_grid_.build();
            frame_cluster_params_ = cluster_grid_.params();
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
                std::cerr << "render_frame: Failed to begin frame upload copy pass" << '\n';
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
                        std::cerr << "render_frame: failed to create velocity texture (prepass disabled)" << '\n';
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
        std::cerr << "Screenshot: failed to create download buffer: " << SDL_GetError() << '\n';
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
        std::cerr << "Screenshot: unexpected swapchain format " << static_cast<int>(gpu_format)
                  << ", assuming RGBA8\n";
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
        SDL_Log("SceneRenderer: screenshot saved to '%s' (%ux%u)", path.c_str(), w, h);
    } else {
        std::cerr << "Screenshot: failed to save '" << path << "': " << SDL_GetError() << '\n';
    }
}

// ============================================================================
// 3D Scene Rendering (shared between the HDR and fallback paths)
// ============================================================================

void SceneRenderer::render_3d_scene(const FramePacket& frame, const CameraState& camera) {
    BRACKEN_PROFILE_ZONE_COLOR("SceneRenderer::render_3d_scene", 0xC0392B);
    const GraphicsSettings& gfx = graphics_ ? *graphics_ : default_graphics_;
    SDL_GPUCommandBuffer* cmd = context_->current_command_buffer();

    // BLEND meshes are queued by the opaque walks below and drawn sorted after
    // the skybox; drop anything stale from an aborted previous pass.
    transparent_draws_.clear();

    // Prepare shadow bindings for terrain/grass: the atlas bound twice —
    // comparison sampler (hardware PCF) then raw sampler (PCSS blocker search).
    SDL_GPUTextureSamplerBinding shadow_bindings[2] = {};
    int shadow_binding_count = 0;
    if (shadow_map_->is_ready()) {
        shadow_bindings[0] = {shadow_map_->atlas_texture(), shadow_map_->shadow_compare_sampler()};
        shadow_bindings[1] = {shadow_map_->atlas_texture(), shadow_map_->shadow_sampler()};
        shadow_binding_count = 2;
    }

    // Screen-space AO binding for terrain/grass (the frame's GTAO output or
    // the white dummy) + the SV_Position -> AO UV texel size.
    SDL_GPUTexture* ao_tex =
        frame_ao_texture_ ? frame_ao_texture_ : (dummy_white_texture_ ? dummy_white_texture_->handle() : nullptr);
    SDL_GPUTextureSamplerBinding ao_binding = {ao_tex, default_sampler_};
    const bool have_ao_binding = ao_tex && default_sampler_;
    const glm::vec2 ao_texel(1.0f / static_cast<float>(std::max(context_->width(), 1)),
                             1.0f / static_cast<float>(std::max(context_->height(), 1)));

    // Sky IBL bindings for terrain/grass (real chain or typed dummies) — the
    // model-family paths bind the same set via bind_ibl().
    SDL_GPUTextureSamplerBinding ibl_bindings[3];
    const bool have_ibl_bindings = fill_ibl_bindings(ibl_bindings);

    // Local (point/spot) shadow atlas + comparison sampler for terrain/grass
    // (the model-family paths bind the same pair via bind_local_shadow_atlas).
    // Falls back to the CSM atlas as a typed filler when unavailable — the
    // slot is only sampled for lights with shadow_slot >= 0.
    SDL_GPUTextureSamplerBinding local_shadow_binding = {};
    bool have_local_shadow_binding = false;
    if (local_shadows_ && local_shadows_->is_ready()) {
        local_shadow_binding = {local_shadows_->atlas_texture(), local_shadows_->compare_sampler()};
        have_local_shadow_binding = true;
    } else if (shadow_map_->is_ready()) {
        local_shadow_binding = {shadow_map_->atlas_texture(), shadow_map_->shadow_compare_sampler()};
        have_local_shadow_binding = true;
    }

    if (frame.should_draw_ground()) {
        SDL_PushGPUFragmentUniformData(cmd, 1, &frame_shadow_uniforms_, sizeof(frame_shadow_uniforms_));
        terrain_->render(main_render_pass_, cmd, camera.view, camera.projection, camera.position, light_dir_,
                         shadow_binding_count > 0 ? shadow_bindings : nullptr, shadow_binding_count,
                         have_ao_binding ? &ao_binding : nullptr, ao_texel, &frame_frustum_, frame_prepass_active_,
                         have_ibl_bindings ? ibl_bindings : nullptr, frame_ibl_active_,
                         have_local_shadow_binding ? &local_shadow_binding : nullptr);
    }

    if (frame.should_draw_grass() && gfx.grass_enabled && grass_renderer_) {
        // view_distance + chunk upload already performed before render pass began.
        SDL_PushGPUFragmentUniformData(cmd, 1, &frame_shadow_uniforms_, sizeof(frame_shadow_uniforms_));
        grass_renderer_->render(main_render_pass_, cmd, camera.view, camera.projection, camera.position, light_dir_,
                                shadow_binding_count > 0 ? shadow_bindings : nullptr, shadow_binding_count,
                                have_ao_binding ? &ao_binding : nullptr, ao_texel, frame_prepass_active_,
                                have_ibl_bindings ? ibl_bindings : nullptr, frame_ibl_active_,
                                have_local_shadow_binding ? &local_shadow_binding : nullptr);
    }

    // Render instanced static models (rocks, trees, etc.) - uses same frustum
    render_instanced_models(camera);

    // Render non-instanced static models (attack animations, no-fog, etc.)
    if (!non_instanced_commands_.empty() && main_render_pass_ && bind_model_pipeline(false, frame_prepass_active_)) {
        for (const auto* draw_cmd : non_instanced_commands_) {
            render_model_command_inner(*draw_cmd, camera);
        }
    }

    // Render skinned models individually (they have per-instance bone data).
    // pack_bone_palettes() already resolved Model*, world bounds, palette
    // offset and main-view visibility per drawable command (commands without
    // a model/skeleton/palette were excluded there; cull stats were counted
    // there), so nothing is re-resolved or re-culled here. Pipeline, shadow
    // data and the bone-palette SSBO bind once before the loop.
    {
        bool skinned_pipeline_bound = false;

        for (const auto& rs : skinned_resolved_) {
            if (!rs.main_visible) {
                continue;
            }

            // Bind pipeline + shadow data + bone palettes once for all skinned models
            if (!skinned_pipeline_bound) {
                if (!bind_skinned_pipeline(false, frame_prepass_active_)) {
                    break;
                }
                skinned_pipeline_bound = true;
            }

            render_skinned_model_command_inner(*rs.cmd, *rs.model, camera, rs.palette_offset);
        }
    }

    // Skybox last among opaques: depth writes are off and the VS emits z == 1
    // with a LESS_OR_EQUAL compare, so only sky pixels shade after the opaque
    // passes. (The sky clock is adopted from the packet unconditionally in
    // render_frame, so the scroll phase advances even while toggled off.)
    if (frame.should_draw_skybox() && gfx.skybox_enabled) {
        world_->render_skybox(main_render_pass_, cmd, camera.view, camera.projection);
    }

    // Sorted transparent pass: glTF alphaMode BLEND meshes queued by the
    // opaque walks, drawn back-to-front over opaques + sky with depth-test on
    // and depth-write off.
    render_transparent_models(camera);

    // Particle effects after the skybox: additive HDR with depth-test-on /
    // depth-write-off, so they composite over sky + opaques in any order.
    // Instance data came from the packet's effect-instance snapshot, packed +
    // uploaded in the consolidated frame copy pass.
    effects_->draw_particle_effects(main_render_pass_, context_->current_command_buffer(), camera.view,
                                    camera.projection, camera.position);

    // Debug line rendering (only if game submitted any; vertex data was
    // uploaded in the consolidated frame copy pass)
    render_debug_lines(camera);
}

// ============================================================================
// Model Rendering
// ============================================================================

void SceneRenderer::render_model_command(const ModelDrawCommand& cmd, const CameraState& camera) {
    // Standalone path: binds pipeline + shadows itself
    if (!main_render_pass_) {
        return;
    }
    if (!bind_model_pipeline(false)) {
        return;
    }
    render_model_command_inner(cmd, camera);
    // Flush any BLEND meshes queued by the command right away (this path runs
    // outside render_3d_scene's sorted transparent phase).
    render_transparent_models(camera);
}

bool SceneRenderer::bind_model_pipeline(bool double_sided, bool depth_equal) {
    gpu::PipelineType type;
    if (depth_equal) {
        type = double_sided ? gpu::PipelineType::ModelDoubleSidedDepthEqual : gpu::PipelineType::ModelDepthEqual;
    } else {
        type = double_sided ? gpu::PipelineType::ModelDoubleSided : gpu::PipelineType::Model;
    }
    gpu::GPUPipeline* pipeline = pipeline_registry_->get_pipeline(type);
    if (!pipeline || !main_render_pass_) {
        return false;
    }
    pipeline->bind(main_render_pass_);
    reset_model_bind_state();
    bind_shadow_data(main_render_pass_, context_->current_command_buffer(), 1);
    bind_cluster_lights(main_render_pass_, context_->current_command_buffer(), 2);
    bind_screen_ao(6);
    bind_ibl(7);
    bind_local_shadow_atlas(10);
    model_pipeline_double_sided_ = double_sided;
    model_pipeline_depth_equal_ = depth_equal;
    return true;
}

bool SceneRenderer::bind_skinned_pipeline(bool double_sided, bool depth_equal) {
    if (!main_render_pass_ || !current_bone_palette_buffer()) {
        return false;
    }
    gpu::PipelineType type;
    if (depth_equal) {
        type = double_sided ? gpu::PipelineType::SkinnedModelDoubleSidedDepthEqual
                            : gpu::PipelineType::SkinnedModelDepthEqual;
    } else {
        type = double_sided ? gpu::PipelineType::SkinnedModelDoubleSided : gpu::PipelineType::SkinnedModel;
    }
    gpu::GPUPipeline* pipeline = pipeline_registry_->get_pipeline(type);
    if (!pipeline) {
        return false;
    }
    pipeline->bind(main_render_pass_);
    reset_model_bind_state();
    bind_shadow_data(main_render_pass_, context_->current_command_buffer(), 1);
    bind_cluster_lights(main_render_pass_, context_->current_command_buffer(), 2);
    bind_screen_ao(6);
    bind_ibl(7);
    bind_local_shadow_atlas(10);
    SDL_GPUBuffer* palette_buf = current_bone_palette_buffer()->handle();
    SDL_BindGPUVertexStorageBuffers(main_render_pass_, 0, &palette_buf, 1);
    skinned_pipeline_double_sided_ = double_sided;
    skinned_pipeline_depth_equal_ = depth_equal;
    return true;
}

void SceneRenderer::reset_model_bind_state() {
    last_frag_tex_.fill(nullptr);
    model_lighting_valid_ = false;
}

void SceneRenderer::bind_screen_ao(uint32_t slot, bool enabled) {
    // BLEND draws pass enabled = false: screen AO belongs to the opaque
    // surface behind a transparent, so applying it would darken transparents
    // with occlusion that isn't theirs.
    SDL_GPUTexture* white = dummy_white_texture_ ? dummy_white_texture_->handle() : nullptr;
    SDL_GPUTexture* tex = (enabled && frame_ao_texture_) ? frame_ao_texture_ : white;
    if (!tex || !default_sampler_ || !main_render_pass_) {
        return;
    }
    SDL_GPUTextureSamplerBinding binding = {tex, default_sampler_};
    SDL_BindGPUFragmentSamplers(main_render_pass_, slot, &binding, 1);
}

bool SceneRenderer::fill_ibl_bindings(SDL_GPUTextureSamplerBinding out[3]) {
    if (frame_ibl_active_ && ibl_ && ibl_->is_ready()) {
        out[0] = {ibl_->irradiance_map(), ibl_->cube_sampler()};
        out[1] = {ibl_->specular_map(), ibl_->cube_sampler()};
        out[2] = {ibl_->brdf_lut(), ibl_->lut_sampler()};
        return true;
    }
    // IBL unavailable/off: keep the slots valid with typed dummies (the
    // shaders never sample them while iblEnabled == 0).
    if (!dummy_black_cube_ || !dummy_white_texture_ || !default_sampler_) {
        return false;
    }
    out[0] = {dummy_black_cube_->handle(), default_sampler_};
    out[1] = {dummy_black_cube_->handle(), default_sampler_};
    out[2] = {dummy_white_texture_->handle(), default_sampler_};
    return true;
}

void SceneRenderer::bind_ibl(uint32_t base_slot) {
    SDL_GPUTextureSamplerBinding bindings[3];
    if (!fill_ibl_bindings(bindings) || !main_render_pass_) {
        return;
    }
    SDL_BindGPUFragmentSamplers(main_render_pass_, base_slot, bindings, 3);
}

void SceneRenderer::bind_material_texture(uint32_t slot, SDL_GPUTexture* texture) {
    if (!texture || !default_sampler_ || last_frag_tex_[slot] == texture) {
        return;
    }
    SDL_GPUTextureSamplerBinding binding = {texture, default_sampler_};
    SDL_BindGPUFragmentSamplers(main_render_pass_, slot, &binding, 1);
    last_frag_tex_[slot] = texture;
}

void SceneRenderer::bind_mesh_material_textures(const Mesh& mesh) {
    const PBRMaterial& mat = mesh.material;
    // Always bind slot 0: the shader declares baseColorTexture regardless of
    // the hasTexture branch, and leaving the slot unbound (or carrying a
    // previous pipeline's incompatible descriptor) is undefined behavior.
    bind_material_texture(0, (mesh.has_texture && mesh.texture)
                                 ? mesh.texture->handle()
                                 : (dummy_white_texture_ ? dummy_white_texture_->handle() : nullptr));
    bind_material_texture(3, mat.normal_texture
                                 ? mat.normal_texture->handle()
                                 : (default_normal_texture_ ? default_normal_texture_->handle() : nullptr));
    bind_material_texture(4, mat.metallic_roughness_texture
                                 ? mat.metallic_roughness_texture->handle()
                                 : (dummy_white_texture_ ? dummy_white_texture_->handle() : nullptr));
    bind_material_texture(5, mat.emissive_texture
                                 ? mat.emissive_texture->handle()
                                 : (dummy_black_texture_ ? dummy_black_texture_->handle() : nullptr));
}

gpu::ModelLightingUniforms SceneRenderer::make_model_lighting(const CameraState& camera, bool fog_active) const {
    const GraphicsSettings& gfx = graphics_ ? *graphics_ : default_graphics_;
    gpu::ModelLightingUniforms u = {};
    u.lightDir = env_.sun_dir;
    u.lightColor = env_.sun_color;
    u.ambientSky = env_.ambient_sky;
    u.ambientGround = env_.ambient_ground;
    u.tintColor = glm::vec4(1.0f);
    u.fogColor = env_.fog_color_horizon;
    u.fogSunColor = env_.fog_color_sun;
    u.fogStart = env_.fog_start;
    u.fogEnd = env_.fog_end;
    u.fogHeightFalloff = env_.fog_height_falloff;
    u.fogEnabled = fog_active ? 1 : 0;
    u.cameraPos = camera.position;
    u.ambientStrength = gfx.ambient_strength;
    u.sunIntensity = gfx.sun_intensity;
    // SV_Position -> screen-AO UV mapping (the AO texture is render-target
    // sized; the values are inert when the white dummy is bound).
    u.screenAOTexelX = 1.0f / static_cast<float>(std::max(context_->width(), 1));
    u.screenAOTexelY = 1.0f / static_cast<float>(std::max(context_->height(), 1));
    u.iblEnabled = frame_ibl_active_ ? 1.0f : 0.0f;
    return u;
}

void SceneRenderer::render_model_command_inner(const ModelDrawCommand& cmd, const CameraState& camera) {
    Model* model = model_manager_->get_model(cmd.model_handle);
    if (!model || !main_render_pass_) {
        return;
    }

    SDL_GPUCommandBuffer* gpu_cmd = context_->current_command_buffer();
    if (!gpu_cmd) {
        return;
    }

    const glm::mat4& model_mat = cmd.transform;

    gpu::ModelTransformUniforms transform_uniforms = {};
    transform_uniforms.model = model_mat;
    transform_uniforms.view = camera.view;
    transform_uniforms.projection = camera.projection;
    transform_uniforms.cameraPos = camera.position;
    transform_uniforms.normalMatrix = compute_normal_matrix(model_mat);

    bool fog_active = !cmd.no_fog && frame_fog_active_;

    gpu::ModelLightingUniforms lighting_uniforms = make_model_lighting(camera, fog_active);
    lighting_uniforms.tintColor = cmd.tint;

    SDL_PushGPUVertexUniformData(gpu_cmd, 0, &transform_uniforms, sizeof(transform_uniforms));

    // Per-section stats partial, merged into render_stats_ once at the end
    // (per-chunk merge discipline — see FrameStatsPartial in render_stats.hpp).
    FrameStatsPartial stats_partial;
    for (auto& mesh : model->meshes) {
        if (mesh.shadow_only) {
            continue; // canopy proxy etc. — shadow-pass only
        }
        BRACKEN_ASSERT(mesh.uploaded, "Mesh not uploaded before non-instanced render");
        if (!mesh.uploaded) {
            continue;
        }

        if (!mesh.vertex_buffer || !mesh.index_buffer) {
            continue;
        }
        if (mesh.index_count() == 0) {
            continue;
        }

        // glTF alphaMode BLEND: defer to the sorted transparent pass.
        if (mesh.material.alpha_mode == AlphaMode::Blend) {
            queue_transparent_mesh(mesh, model_mat, transform_uniforms.normalMatrix, cmd.tint, cmd.no_fog,
                                   INVALID_PALETTE_OFFSET, *model, camera);
            continue;
        }

        // glTF doubleSided selects the no-cull variant; after a depth prepass
        // OPAQUE meshes use the EQUAL/write-off variant while MASK meshes
        // keep GREATER_OR_EQUAL + write (their depth wasn't prepassed).
        // Pushed uniform data survives pipeline binds; sampler/uniform
        // tracking is conservatively reset by bind_model_pipeline.
        const bool wants_equal = frame_prepass_active_ && mesh.material.alpha_mode == AlphaMode::Opaque;
        if ((mesh.material.double_sided != model_pipeline_double_sided_ ||
             wants_equal != model_pipeline_depth_equal_) &&
            !bind_model_pipeline(mesh.material.double_sided, wants_equal)) {
            continue;
        }

        apply_mesh_material(lighting_uniforms, mesh);
        if (!model_lighting_valid_ ||
            std::memcmp(&last_model_lighting_, &lighting_uniforms, sizeof(lighting_uniforms)) != 0) {
            SDL_PushGPUFragmentUniformData(gpu_cmd, 0, &lighting_uniforms, sizeof(lighting_uniforms));
            last_model_lighting_ = lighting_uniforms;
            model_lighting_valid_ = true;
        }

        bind_mesh_material_textures(mesh);

        mesh.bind_buffers(main_render_pass_);
        if (collect_stats_) {
            stats_partial.draw_calls++;
            stats_partial.triangle_count += mesh.index_count() / 3;
        }
        SDL_DrawGPUIndexedPrimitives(main_render_pass_, mesh.index_count(), 1, 0, 0, 0);
    }
    stats_partial.merge_into(render_stats_);
}

// ============================================================================
// Instanced Rendering
// ============================================================================

// ---------------------------------------------------------------------------
// Prep fan-out chunk bodies (docs/multithreading_plan.md §4.3). Each binning
// body is destination-agnostic: the single-chunk inline path passes the FINAL
// output containers and the fan-out path passes a chunk's private scratch —
// the exact same per-command logic runs either way (granularity control, not
// a second code path), which is what makes chunk-order concatenation
// bit-identical to the serial walk.
// ---------------------------------------------------------------------------

// Main-view walk over cmds[begin, end): distance cull → bounds → main frustum
// cull → scatter into dense per-handle bins / the non-instanced list, with
// cull counters in `stats` (main view owns ALL model-command stats — the
// shadow views recompute the same pure predicates but count nothing).
static void bin_main_chunk(std::span<const ModelDrawCommand> cmds, size_t begin, size_t end,
                           const SceneRenderer::PrepParams& p, std::vector<std::vector<gpu::InstanceData>>& bins,
                           std::vector<ModelHandle>& touched,
                           std::vector<const ModelDrawCommand*>& non_instanced, FrameStatsPartial& stats) {
    for (size_t i = begin; i < end; ++i) {
        const ModelDrawCommand& cmd = cmds[i];
        const glm::mat4& t = cmd.transform;

        const float dx = t[3][0] - p.camera_pos.x;
        const float dz = t[3][2] - p.camera_pos.z;
        if (dx * dx + dz * dz > p.draw_dist_sq) {
            if (p.collect_stats) {
                stats.entities_distance_culled++;
            }
            continue;
        }

        const ModelHandle handle = cmd.model_handle;

        // Main visibility only needs bounds when frustum culling is on; a
        // command whose model is missing has no bounds and stays visible
        // (matches the serial walk, where the frustum test required bounds).
        bool main_visible = true;
        if (p.do_frustum_cull) {
            if (const Model* model = p.models->get_model(handle)) {
                const glm::vec3 world_center = glm::vec3(t * glm::vec4(model->bounding_center, 1.0f));
                const float world_radius = model->bounding_half_diag * max_scale_factor(t);
                main_visible = p.main_frustum->intersects_sphere(world_center, world_radius);
            }
        }

        if (!main_visible) {
            if (p.collect_stats) {
                stats.entities_frustum_culled++;
            }
            continue; // shadow binning for main-culled casters is the shadow views' job
        }

        if (p.collect_stats) {
            stats.entities_rendered++;
        }

        if (cmd.force_non_instanced) {
            non_instanced.push_back(&cmd);
            continue;
        }

        gpu::InstanceData instance{};
        instance.model = cmd.transform;
        instance.normalMatrix = compute_normal_matrix(cmd.transform);
        instance.tint = cmd.tint;
        instance.noFog = cmd.no_fog ? 1.0f : 0.0f;
        // Motion vectors: previous-frame transform from the motion history
        // (per-instance motion_id); statics fall back to the current
        // transform, i.e. camera-only velocity.
        const MotionHistory::Entry* hist = p.motion_history->find_prev(cmd.motion_id);
        instance.prevModel = hist ? hist->model : cmd.transform;

        // Dense-bin push: grow the bin array to cover the handle (handles are
        // dense indices into ModelManager's slot array, so this stays small)
        // and record first touch for O(touched) clearing/merging.
        if (handle >= bins.size()) {
            bins.resize(handle + 1);
        }
        auto& bin = bins[handle];
        if (bin.empty()) {
            touched.push_back(handle);
        }
        bin.push_back(instance);
    }
}

// One cascade's walk over cmds[begin, end): recomputes the pure main-view
// predicates (distance cull, bounds, main frustum test — cheap and pure, the
// price of view independence), then bins casters that intersect THIS
// cascade's light frustum. Semantics mirror the serial walk exactly:
// - distance-culled commands cast nothing;
// - main-frustum-culled commands cast only with bounds and only when
//   instanced (force_non_instanced casters draw via the non-instanced path);
// - main-visible instanced commands cast when boundless or intersecting.
static void bin_shadow_chunk(std::span<const ModelDrawCommand> cmds, size_t begin, size_t end,
                             const SceneRenderer::PrepParams& p, int cascade,
                             std::vector<std::vector<gpu::ShadowInstanceData>>& bins,
                             std::vector<ModelHandle>& touched) {
    const Frustum& light_frustum = p.cascade_frustums[cascade];
    for (size_t i = begin; i < end; ++i) {
        const ModelDrawCommand& cmd = cmds[i];
        const glm::mat4& t = cmd.transform;

        const float dx = t[3][0] - p.camera_pos.x;
        const float dz = t[3][2] - p.camera_pos.z;
        if (dx * dx + dz * dz > p.draw_dist_sq) {
            continue;
        }

        const ModelHandle handle = cmd.model_handle;

        glm::vec3 world_center{};
        float world_radius = 0.0f;
        bool have_bounds = false;
        if (const Model* model = p.models->get_model(handle)) {
            world_center = glm::vec3(t * glm::vec4(model->bounding_center, 1.0f));
            world_radius = model->bounding_half_diag * max_scale_factor(t);
            have_bounds = true;
        }

        const bool main_visible =
            !(p.do_frustum_cull && have_bounds && !p.main_frustum->intersects_sphere(world_center, world_radius));

        bool casts = false;
        if (!main_visible) {
            casts = !cmd.force_non_instanced && have_bounds &&
                    light_frustum.intersects_sphere(world_center, world_radius);
        } else if (!cmd.force_non_instanced) {
            casts = !have_bounds || light_frustum.intersects_sphere(world_center, world_radius);
        }
        if (!casts) {
            continue;
        }

        gpu::ShadowInstanceData shadow_instance{};
        shadow_instance.model = cmd.transform;
        if (handle >= bins.size()) {
            bins.resize(handle + 1);
        }
        auto& bin = bins[handle];
        if (bin.empty()) {
            touched.push_back(handle);
        }
        bin.push_back(shadow_instance);
    }
}

// Shared per-view chunk fan-out driver: run `chunk_fn(chunk, begin, end)`
// over [0, total) in PREP_COMMAND_GRAIN chunks on the pool, then
// `merge_fn(chunk)` single-threaded in chunk-index order. `inline_fn()` is
// the one-chunk path (same chunk body aimed at the final outputs). The guard
// mirrors parallel_for's exactly (initialized, >1 worker, more than one
// grain of work), so the fan-out branch is taken iff parallel_for will
// actually chunk at [k * grain, ...) boundaries — which is what makes
// `begin / grain` the chunk index.
template<typename InlineFn, typename ChunkFn, typename MergeFn>
static void prep_view_fan_out(core::jobs::JobSystem& js, size_t total, SceneRenderer::PrepViewScratch& scratch,
                              InlineFn&& inline_fn, ChunkFn&& chunk_fn, MergeFn&& merge_fn) {
    constexpr size_t grain = SceneRenderer::PREP_COMMAND_GRAIN;
    const bool fan_out = js.is_initialized() && js.worker_count() > 1 && total > grain;
    if (!fan_out) {
        inline_fn();
        return;
    }

    const size_t chunk_count = (total + grain - 1) / grain;
    // Grow-only: per-chunk capacity survives across frames (see
    // PrepViewScratch — the no-per-frame-allocs discipline).
    if (scratch.chunks.size() < chunk_count) {
        scratch.chunks.resize(chunk_count);
    }
    core::jobs::parallel_for(
        js, 0, total,
        [&](size_t begin, size_t end) { chunk_fn(scratch.chunks[begin / grain], begin, end); }, grain);
    // Single-threaded merge in chunk-index order (§6.2): contiguous chunks
    // concatenated in index order reproduce the serial command order.
    for (size_t k = 0; k < chunk_count; ++k) {
        merge_fn(scratch.chunks[k]);
    }
}

// Main-view job body: chunked cull+bin into instance bins + non-instanced
// list + stats, merged in chunk order.
static void prep_main_view(core::jobs::JobSystem& js, std::span<const ModelDrawCommand> cmds,
                           const SceneRenderer::PrepParams& p, SceneRenderer::PrepViewScratch& scratch,
                           const SceneRenderer::InstanceBinOutputs& out) {
    prep_view_fan_out(
        js, cmds.size(), scratch,
        [&] {
            bin_main_chunk(cmds, 0, cmds.size(), p, out.instance_batches, out.instance_batches_touched,
                           out.non_instanced, out.stats);
        },
        [&](SceneRenderer::PrepChunkScratch& ch, size_t begin, size_t end) {
            // Clear ONLY this core's scratch group, O(touched of last use).
            for (ModelHandle h : ch.touched) ch.bins[h].clear();
            ch.touched.clear();
            ch.non_instanced.clear();
            ch.stats = {};
            bin_main_chunk(cmds, begin, end, p, ch.bins, ch.touched, ch.non_instanced, ch.stats);
        },
        [&](SceneRenderer::PrepChunkScratch& ch) {
            for (ModelHandle h : ch.touched) {
                if (h >= out.instance_batches.size()) {
                    out.instance_batches.resize(h + 1);
                }
                auto& bin = out.instance_batches[h];
                if (bin.empty()) {
                    out.instance_batches_touched.push_back(h);
                }
                bin.insert(bin.end(), ch.bins[h].begin(), ch.bins[h].end());
            }
            out.non_instanced.insert(out.non_instanced.end(), ch.non_instanced.begin(), ch.non_instanced.end());
            out.stats += ch.stats;
        });
}

// One cascade view's job body: chunked light-frustum binning into that
// cascade's bins, merged in chunk order. No stats (main view owns them).
static void prep_shadow_view(core::jobs::JobSystem& js, std::span<const ModelDrawCommand> cmds,
                             const SceneRenderer::PrepParams& p, int cascade,
                             SceneRenderer::PrepViewScratch& scratch,
                             std::vector<std::vector<gpu::ShadowInstanceData>>& bins,
                             std::vector<ModelHandle>& touched) {
    prep_view_fan_out(
        js, cmds.size(), scratch, [&] { bin_shadow_chunk(cmds, 0, cmds.size(), p, cascade, bins, touched); },
        [&](SceneRenderer::PrepChunkScratch& ch, size_t begin, size_t end) {
            for (ModelHandle h : ch.shadow_touched) ch.shadow_bins[h].clear();
            ch.shadow_touched.clear();
            bin_shadow_chunk(cmds, begin, end, p, cascade, ch.shadow_bins, ch.shadow_touched);
        },
        [&](SceneRenderer::PrepChunkScratch& ch) {
            for (ModelHandle h : ch.shadow_touched) {
                if (h >= bins.size()) {
                    bins.resize(h + 1);
                }
                auto& bin = bins[h];
                if (bin.empty()) {
                    touched.push_back(h);
                }
                bin.insert(bin.end(), ch.shadow_bins[h].begin(), ch.shadow_bins[h].end());
            }
        });
}

void SceneRenderer::prep_instance_batches(core::jobs::JobSystem& js, std::span<const ModelDrawCommand> cmds,
                                          const PrepParams& p, PrepScratch& scratch,
                                          const InstanceBinOutputs& out) {
    BRACKEN_ASSERT(p.active_cascades <= MAX_SHADOW_CASCADES, "active_cascades {} exceeds MAX_SHADOW_CASCADES",
                   p.active_cascades);
    // Clear only the bins touched last frame (capacity preserved); empty bins
    // for never/rarely used models cost nothing to skip at iteration time.
    for (ModelHandle h : out.instance_batches_touched) out.instance_batches[h].clear();
    out.instance_batches_touched.clear();
    for (int c = 0; c < MAX_SHADOW_CASCADES; ++c) {
        for (ModelHandle h : out.shadow_batches_touched[c]) out.shadow_instance_batches[c][h].clear();
        out.shadow_batches_touched[c].clear();
    }
    out.non_instanced.clear();
    if (cmds.empty()) {
        return;
    }

    // View fan-out: the views write DISJOINT outputs (main bins + stats vs
    // each cascade's shadow bins), so each is an independent job. Small
    // scenes (below one chunk of commands — each view job would walk only a
    // handful of items) and 1-worker pools run the views sequentially inline
    // instead; the view bodies apply the chunk grain guard themselves, so
    // this is granularity control, not a separate code path.
    const bool fan_out_views = js.is_initialized() && js.worker_count() > 1 && p.active_cascades > 0 &&
                               cmds.size() > PREP_COMMAND_GRAIN;
    if (!fan_out_views) {
        prep_main_view(js, cmds, p, scratch.main, out);
        for (int c = 0; c < p.active_cascades; ++c) {
            prep_shadow_view(js, cmds, p, c, scratch.cascades[c], out.shadow_instance_batches[c],
                             out.shadow_batches_touched[c]);
        }
        return;
    }

    core::jobs::TaskGroup views;
    try {
        for (int c = 0; c < p.active_cascades; ++c) {
            js.submit(views, [&js, cmds, &p, &scratch, &out, c] {
                prep_shadow_view(js, cmds, p, c, scratch.cascades[c], out.shadow_instance_batches[c],
                                 out.shadow_batches_touched[c]);
            });
        }
        // Main view runs on the caller, which then helps execute the cascade
        // jobs (and their chunks) inside the join.
        prep_main_view(js, cmds, p, scratch.main, out);
    } catch (...) {
        // Join the cascade jobs before unwinding — they reference this
        // frame's stack state (mirrors parallel_for's exception contract).
        try {
            js.wait(views);
        } catch (...) { // NOLINT(bugprone-empty-catch)
        }
        throw;
    }
    js.wait(views);
}

void SceneRenderer::build_instance_batches(const FramePacket& frame, const CameraState& camera,
                                           const Frustum& frustum) {
    BRACKEN_PROFILE_ZONE_COLOR("SceneRenderer::build_instance_batches", 0x27AE60);

    // Shadow state: which cascades are active, and their light frustums.
    // Mirror render_frame's shadows_on gate — with shadow_mode == 0 the shadow
    // passes never run (and the cascade frustums are stale), so don't bin,
    // pack, or upload any shadow instances.
    const GraphicsSettings& gfx = graphics_ ? *graphics_ : default_graphics_;
    const int active_cascades =
        (shadow_map_->is_ready() && gfx.shadow_mode != ShadowMode::Off) ? shadow_map_->active_cascades() : 0;

    PrepParams p;
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
    const InstanceBinOutputs out{instance_batches_,   instance_batches_touched_, shadow_instance_batches_,
                                 shadow_batches_touched_, non_instanced_commands_, stats_partial};
    prep_instance_batches(core::jobs::JobSystem::instance(), frame.model_commands(), p, prep_scratch_, out);
    stats_partial.merge_into(render_stats_);
}

void SceneRenderer::upload_instance_buffers(SDL_GPUCopyPass* copy_pass) {
    if (!copy_pass) {
        return;
    }

    // Upload main instance buffer
    size_t total_instances = 0;
    for (ModelHandle h : instance_batches_touched_) {
        total_instances += instance_batches_[h].size();
    }

    if (total_instances > 0) {
        size_t required_size = total_instances * sizeof(gpu::InstanceData);

        if (!instance_storage_buffer_ || instance_storage_capacity_ < required_size) {
            instance_storage_capacity_ = required_size * 2; // over-allocate
            instance_storage_buffer_ = gpu::GPUBuffer::create_dynamic(context_->device(), gpu::GPUBuffer::Type::Storage,
                                                                      instance_storage_capacity_);
            if (!instance_storage_buffer_) {
                std::cerr << "[SceneRenderer] Failed to create instance storage buffer" << '\n';
                instance_storage_capacity_ = 0;
            }
        }

        if (instance_storage_buffer_) {
            // Pack in ascending handle order — must match the iteration order in
            // render_instanced_models so base_instance offsets line up. Bins are
            // written straight into the mapped transfer buffer (one pass over
            // the data instead of bin -> pack scratch -> transfer buffer).
            if (gpu::GPUBuffer::Mapping mapping = instance_storage_buffer_->map()) {
                auto* dst = static_cast<gpu::InstanceData*>(mapping.data());
                for (const auto& instances : instance_batches_) {
                    if (instances.empty()) {
                        continue;
                    }
                    std::memcpy(dst, instances.data(), instances.size() * sizeof(gpu::InstanceData));
                    dst += instances.size();
                }
                mapping.upload(copy_pass, required_size);
            }
        }
    }

    // Upload shadow instance buffer (per-cascade sections packed contiguously).
    // Layout: [cascade 0 instances][cascade 1 instances]... each cascade in ascending handle order.
    size_t total_shadow = 0;
    for (int c = 0; c < MAX_SHADOW_CASCADES; ++c) {
        for (ModelHandle h : shadow_batches_touched_[c]) {
            total_shadow += shadow_instance_batches_[c][h].size();
        }
    }

    if (total_shadow > 0) {
        size_t required_size = total_shadow * sizeof(gpu::ShadowInstanceData);

        if (!shadow_instance_storage_buffer_ || shadow_instance_storage_capacity_ < required_size) {
            shadow_instance_storage_capacity_ = required_size * 2;
            shadow_instance_storage_buffer_ = gpu::GPUBuffer::create_dynamic(
                context_->device(), gpu::GPUBuffer::Type::Storage, shadow_instance_storage_capacity_);
            if (!shadow_instance_storage_buffer_) {
                std::cerr << "[SceneRenderer] Failed to create shadow instance storage buffer" << '\n';
                shadow_instance_storage_capacity_ = 0;
                shadow_cascade_base_instance_ = {0, 0, 0, 0};
                return;
            }
        }

        // Cascade sections packed contiguously, straight into the mapped
        // transfer buffer (same single-pass packing as the main instances).
        if (gpu::GPUBuffer::Mapping mapping = shadow_instance_storage_buffer_->map()) {
            auto* dst = static_cast<gpu::ShadowInstanceData*>(mapping.data());
            uint32_t packed = 0;
            for (int c = 0; c < MAX_SHADOW_CASCADES; ++c) {
                shadow_cascade_base_instance_[c] = packed;
                for (const auto& instances : shadow_instance_batches_[c]) {
                    if (instances.empty()) {
                        continue;
                    }
                    std::memcpy(dst, instances.data(), instances.size() * sizeof(gpu::ShadowInstanceData));
                    dst += instances.size();
                    packed += static_cast<uint32_t>(instances.size());
                }
            }
            mapping.upload(copy_pass, required_size);
        }
    } else {
        shadow_cascade_base_instance_ = {0, 0, 0, 0};
    }
}

// ============================================================================
// Bone Palettes
// ============================================================================

// Skinned-resolve walk over cmds[begin, end): the pure per-command work —
// model lookup, joint clamp, world bounds, main-view cull, motion-history
// pairing — appended to `out` in command order. Destination-agnostic like the
// binning chunk bodies: final output vector inline, chunk scratch in the
// fan-out.
static void resolve_skinned_chunk(std::span<const SkinnedDrawCommand> cmds, size_t begin, size_t end,
                                  std::span<const BonePalette> palettes, const SceneRenderer::PrepParams& p,
                                  std::vector<SceneRenderer::ResolvedSkinnedEntry>& out, FrameStatsPartial& stats) {
    for (size_t i = begin; i < end; ++i) {
        const SkinnedDrawCommand& cmd = cmds[i];
        // Sealed commands carry dense palette indices (invalid ids were
        // dropped with a warning at emit time) — an out-of-range id here is
        // a builder/packet bug, not host error.
        BRACKEN_ASSERT(cmd.palette.valid() && cmd.palette.value < palettes.size(),
                       "SkinnedDrawCommand palette id {} out of range ({} palettes)", cmd.palette.value,
                       palettes.size());
        if (!cmd.palette.valid() || cmd.palette.value >= palettes.size()) {
            continue;
        }
        const BonePalette& rec = palettes[cmd.palette.value];
        Model* model = p.models->get_model(cmd.model_handle);
        uint32_t joint_count = 0;
        if (model && model->has_skeleton) {
            // Drawn joints clamp to min(model joints, palette joints) — same
            // clamp as the old span path, now against the packet palette.
            joint_count = std::min({static_cast<uint32_t>(model->skeleton.joints.size()), rec.joint_count,
                                    static_cast<uint32_t>(bracken::animation::MAX_BONES)});
        }
        if (joint_count == 0) {
            continue; // no model/skeleton/palette joints: not drawable as skinned
        }

        // Resolve world bounds once here; the main view and every shadow
        // cascade reuse them instead of re-fetching the model per pass.
        const glm::mat4& t = cmd.transform;
        const glm::vec3 world_center = glm::vec3(t * glm::vec4(model->bounding_center, 1.0f));
        const float world_radius = model->bounding_half_diag * max_scale_factor(t);

        // Main-view cull once, shared by the depth prepass and the forward
        // pass (shadow cascades cull independently against light frustums).
        // Cull stats for skinned entities are counted here, nowhere else.
        bool main_visible = true;
        const float dx = t[3][0] - p.camera_pos.x;
        const float dz = t[3][2] - p.camera_pos.z;
        if (dx * dx + dz * dz > p.draw_dist_sq) {
            main_visible = false;
            if (p.collect_stats) {
                stats.entities_distance_culled++;
            }
        } else if (p.do_frustum_cull && !p.main_frustum->intersects_sphere(world_center, world_radius)) {
            main_visible = false;
            if (p.collect_stats) {
                stats.entities_frustum_culled++;
            }
        } else if (p.collect_stats) {
            stats.entities_rendered++;
        }

        // Motion vectors: previous-frame transform + palette offset. The
        // palette offset indexes the PREVIOUS palette buffer (double-buffered)
        // and is only usable when that buffer actually holds last frame's data.
        const MotionHistory::Entry* hist = p.motion_history->find_prev(cmd.motion_id);
        const glm::mat4 prev_model = hist ? hist->model : t;
        const uint32_t prev_palette =
            (hist && hist->palette_offset != MotionHistory::NO_PALETTE && p.prev_palette_valid)
                ? hist->palette_offset
                : gpu::PrepassModelUniforms::NO_PREV_PALETTE;

        // Palette GPU offset: the record's prefix-summed matrix_offset —
        // sharing (many draws, one palette) already happened at build time
        // via a shared BonePaletteId, replacing the old pointer-dedupe.
        out.push_back(
            {&cmd, model, world_center, world_radius, rec.matrix_offset, main_visible, prev_model, prev_palette});
    }
}

void SceneRenderer::prep_resolve_skinned(core::jobs::JobSystem& js, std::span<const SkinnedDrawCommand> cmds,
                                         std::span<const BonePalette> palettes, const PrepParams& p,
                                         PrepViewScratch& scratch, std::vector<ResolvedSkinnedEntry>& out,
                                         FrameStatsPartial& stats) {
    out.clear();
    if (cmds.empty()) {
        return;
    }
    out.reserve(cmds.size());
    prep_view_fan_out(
        js, cmds.size(), scratch, [&] { resolve_skinned_chunk(cmds, 0, cmds.size(), palettes, p, out, stats); },
        [&](PrepChunkScratch& ch, size_t begin, size_t end) {
            ch.skinned.clear();
            ch.stats = {};
            resolve_skinned_chunk(cmds, begin, end, palettes, p, ch.skinned, ch.stats);
        },
        [&](PrepChunkScratch& ch) {
            out.insert(out.end(), ch.skinned.begin(), ch.skinned.end());
            stats += ch.stats;
        });
}

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

    PrepParams p;
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
    prep_resolve_skinned(core::jobs::JobSystem::instance(), cmds, frame.palettes(), p, prep_scratch_.skinned,
                         skinned_resolved_, stats_partial);
    stats_partial.merge_into(render_stats_);
}

void SceneRenderer::upload_bone_palettes(SDL_GPUCopyPass* copy_pass, const FramePacket& frame) {
    if (!copy_pass || bone_palette_matrix_count_ == 0) {
        return;
    }

    // Upload into the CURRENT slot of the double buffer; the other slot still
    // holds last frame's palettes for the skinned prepass' prev-pose reads.
    auto& buffer = bone_palette_buffers_[bone_palette_current_];
    auto& capacity = bone_palette_capacities_[bone_palette_current_];

    const size_t required_size = bone_palette_matrix_count_ * sizeof(glm::mat4);
    if (!buffer || capacity < required_size) {
        // Grow geometrically with a floor of 64 KB (~16 palettes of 64 joints).
        capacity = std::max(required_size * 2, static_cast<size_t>(64 * 1024));
        buffer = gpu::GPUBuffer::create_dynamic(context_->device(), gpu::GPUBuffer::Type::Storage, capacity);
        if (!buffer) {
            std::cerr << "[SceneRenderer] Failed to create bone palette storage buffer" << '\n';
            capacity = 0;
            return;
        }
    }

    // Copy each palette straight from packet memory into the mapped transfer
    // buffer at its prefix-summed matrix_offset — the offsets were assigned
    // serially at seal in palettes() order, so destination offsets line up
    // with every draw's palette offset by construction (and the copies
    // parallelize over disjoint ranges — see prep_copy_palettes).
    gpu::GPUBuffer::Mapping mapping = buffer->map();
    if (!mapping) {
        std::cerr << "[SceneRenderer] Failed to map bone palette transfer buffer" << '\n';
        return;
    }
    prep_copy_palettes(core::jobs::JobSystem::instance(), frame.palettes(),
                       static_cast<glm::mat4*>(mapping.data()));
    mapping.upload(copy_pass, required_size);
}

void SceneRenderer::prep_copy_palettes(core::jobs::JobSystem& js, std::span<const BonePalette> palettes,
                                       glm::mat4* dst) {
    // Every chunk writes only [matrix_offset, matrix_offset + joint_count)
    // ranges of the palettes it owns — disjoint by the seal-time prefix sum,
    // so there are no partials to merge and the result is bit-identical at
    // any worker count (the bytes and their destinations are fixed; only the
    // copying thread varies). Explicit grain: palette items are heavy (up to
    // an 8 KB memcpy each); <= one grain of palettes stays inline.
    core::jobs::parallel_for(
        js, 0, palettes.size(),
        [palettes, dst](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                const BonePalette& rec = palettes[i];
                std::memcpy(dst + rec.matrix_offset, rec.matrices,
                            static_cast<size_t>(rec.joint_count) * sizeof(glm::mat4));
            }
        },
        PREP_PALETTE_GRAIN);
}

void SceneRenderer::render_instanced_models(const CameraState& camera) {
    if (instance_batches_touched_.empty() || !main_render_pass_) {
        return;
    }

    SDL_GPUCommandBuffer* gpu_cmd = context_->current_command_buffer();
    if (!gpu_cmd) {
        return;
    }

    if (!pipeline_registry_->get_instanced_model_pipeline() || !instance_storage_buffer_) {
        return;
    }

    bool fog_active = frame_fog_active_;

    // Push shared camera uniforms (once)
    gpu::InstancedCameraUniforms camera_uniforms = {};
    camera_uniforms.view = camera.view;
    camera_uniforms.projection = camera.projection;
    camera_uniforms.cameraPos = camera.position;

    // Shared lighting uniforms; per-mesh material fields filled in the loop.
    const GraphicsSettings& gfx_inst = graphics_ ? *graphics_ : default_graphics_;
    gpu::InstancedLightingUniforms lighting_uniforms = {};
    lighting_uniforms.lightDir = env_.sun_dir;
    lighting_uniforms.lightColor = env_.sun_color;
    lighting_uniforms.ambientSky = env_.ambient_sky;
    lighting_uniforms.ambientGround = env_.ambient_ground;
    lighting_uniforms.fogColor = env_.fog_color_horizon;
    lighting_uniforms.fogSunColor = env_.fog_color_sun;
    lighting_uniforms.fogStart = env_.fog_start;
    lighting_uniforms.fogEnd = env_.fog_end;
    lighting_uniforms.fogHeightFalloff = env_.fog_height_falloff;
    lighting_uniforms.fogEnabled = fog_active ? 1 : 0;
    lighting_uniforms.cameraPos = camera.position;
    lighting_uniforms.ambientStrength = gfx_inst.ambient_strength;
    lighting_uniforms.sunIntensity = gfx_inst.sun_intensity;
    lighting_uniforms.screenAOTexelX = 1.0f / static_cast<float>(std::max(context_->width(), 1));
    lighting_uniforms.screenAOTexelY = 1.0f / static_cast<float>(std::max(context_->height(), 1));
    lighting_uniforms.iblEnabled = frame_ibl_active_ ? 1.0f : 0.0f;

    SDL_GPUBuffer* storage_buf = instance_storage_buffer_->handle();

    // Track last-pushed uniforms to skip redundant per-mesh pushes
    gpu::InstancedLightingUniforms last_pushed = {};
    bool lighting_pushed = false;

    // Pipeline variant selection: glTF doubleSided selects the no-cull
    // variant; after a depth prepass OPAQUE meshes use the EQUAL/write-off
    // variant while MASK meshes keep GREATER_OR_EQUAL + write. Binding a
    // variant re-establishes the full pass state that follows a bind.
    gpu::PipelineType bound_type = gpu::PipelineType::Count;
    auto select_type = [](bool double_sided, bool depth_equal) {
        if (depth_equal) {
            return double_sided ? gpu::PipelineType::InstancedModelDoubleSidedDepthEqual
                                : gpu::PipelineType::InstancedModelDepthEqual;
        }
        return double_sided ? gpu::PipelineType::InstancedModelDoubleSided : gpu::PipelineType::InstancedModel;
    };
    auto ensure_pipeline = [&](gpu::PipelineType want) -> bool {
        if (want == bound_type) {
            return true;
        }
        gpu::GPUPipeline* variant = pipeline_registry_->get_pipeline(want);
        if (!variant) {
            return false;
        }
        variant->bind(main_render_pass_);
        reset_model_bind_state();
        SDL_PushGPUVertexUniformData(gpu_cmd, 0, &camera_uniforms, sizeof(camera_uniforms));
        SDL_BindGPUVertexStorageBuffers(main_render_pass_, 0, &storage_buf, 1);
        bind_shadow_data(main_render_pass_, gpu_cmd, 1);
        bind_cluster_lights(main_render_pass_, gpu_cmd, 2);
        bind_screen_ao(6);
        bind_ibl(7);
        bind_local_shadow_atlas(10);
        lighting_pushed = false;
        bound_type = want;
        return true;
    };

    // Draw each model type as a single instanced draw call per mesh.
    // Ascending handle order matches the packing order in upload_instance_buffers.
    // Draw stats accumulate into a section-local partial, merged once after
    // the walk (per-chunk merge discipline — see FrameStatsPartial).
    FrameStatsPartial stats_partial;
    uint32_t base_instance = 0;
    for (ModelHandle batch_handle = 0; batch_handle < instance_batches_.size(); ++batch_handle) {
        const auto& instances = instance_batches_[batch_handle];
        if (instances.empty()) {
            continue;
        }

        Model* model = (batch_handle != INVALID_MODEL_HANDLE) ? model_manager_->get_model(batch_handle) : nullptr;
        if (!model) {
            base_instance += static_cast<uint32_t>(instances.size());
            continue;
        }

        uint32_t instance_count = static_cast<uint32_t>(instances.size());

        for (auto& mesh : model->meshes) {
            if (mesh.shadow_only) {
                continue;
            }
            BRACKEN_ASSERT(mesh.uploaded, "Mesh not uploaded before instanced render");
            if (!mesh.uploaded) {
                continue;
            }
            if (!mesh.vertex_buffer || !mesh.index_buffer || mesh.index_count() == 0) {
                continue;
            }

            // glTF alphaMode BLEND: one instanced draw can't be depth-sorted
            // per instance, so route each instance through the sorted
            // per-draw transparent path instead.
            if (mesh.material.alpha_mode == AlphaMode::Blend) {
                for (const auto& inst : instances) {
                    queue_transparent_mesh(mesh, inst.model, inst.normalMatrix, inst.tint, inst.noFog > 0.5f,
                                           INVALID_PALETTE_OFFSET, *model, camera);
                }
                continue;
            }

            const bool wants_equal = frame_prepass_active_ && mesh.material.alpha_mode == AlphaMode::Opaque;
            if (!ensure_pipeline(select_type(mesh.material.double_sided, wants_equal))) {
                continue;
            }

            apply_mesh_material(lighting_uniforms, mesh);
            if (!lighting_pushed ||
                std::memcmp(&last_pushed, &lighting_uniforms, sizeof(lighting_uniforms)) != 0) {
                SDL_PushGPUFragmentUniformData(gpu_cmd, 0, &lighting_uniforms, sizeof(lighting_uniforms));
                last_pushed = lighting_uniforms;
                lighting_pushed = true;
            }

            bind_mesh_material_textures(mesh);

            mesh.bind_buffers(main_render_pass_);
            if (collect_stats_) {
                stats_partial.draw_calls++;
                stats_partial.triangle_count += mesh.index_count() / 3 * instance_count;
            }
            SDL_DrawGPUIndexedPrimitives(main_render_pass_, mesh.index_count(), instance_count, 0, 0, base_instance);
        }

        base_instance += instance_count;
    }
    stats_partial.merge_into(render_stats_);
}

// ============================================================================
// Depth Prepass (velocity + scene depth)
// ============================================================================
// Covers every glTF alphaMode OPAQUE draw: terrain, static models, instanced
// models, skinned models, grass. MASK/BLEND are excluded (they keep forward
// GREATER_OR_EQUAL depth behavior). Vertex work is position-only (+ skinning);
// the shared fragment shader writes UV-space velocity (current - previous,
// unjittered matrices) into the RG16F target while depth writes lay down the
// scene depth the forward opaques then test EQUAL against.

bool SceneRenderer::render_depth_prepass(const FramePacket& frame, const CameraState& draw_camera) {
    SDL_GPUCommandBuffer* cmd = context_->current_command_buffer();
    if (!cmd || !velocity_texture_ || !post_->offscreen_depth()) {
        return false;
    }
    if (velocity_texture_->width() != post_->offscreen_depth()->width() ||
        velocity_texture_->height() != post_->offscreen_depth()->height()) {
        return false; // mid-resize mismatch: skip the prepass this frame
    }

    BRACKEN_PROFILE_ZONE_COLOR("DepthPrepass", 0xC0392B);
    BRACKEN_GPU_SCOPE(cmd, "DepthPrepass");
    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "DepthPrepass");

    SDL_GPUColorTargetInfo color_target = {};
    color_target.texture = velocity_texture_->handle();
    color_target.load_op = SDL_GPU_LOADOP_CLEAR;
    color_target.store_op = SDL_GPU_STOREOP_STORE;
    color_target.clear_color = {0.0f, 0.0f, 0.0f, 0.0f}; // zero velocity (sky pixels reproject via depth in TAA)

    SDL_GPUDepthStencilTargetInfo depth_target = {};
    depth_target.texture = post_->offscreen_depth()->handle();
    depth_target.load_op = SDL_GPU_LOADOP_CLEAR;
    depth_target.store_op = SDL_GPU_STOREOP_STORE; // the forward pass loads this depth
    depth_target.clear_depth = 0.0f;               // reversed-Z: 0.0 is the far plane
    depth_target.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
    depth_target.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, &depth_target);
    if (!pass) {
        std::cerr << "render_depth_prepass: failed to begin render pass" << '\n';
        return false;
    }

    const GraphicsSettings& gfx = graphics_ ? *graphics_ : default_graphics_;

    // Shared camera uniforms for the terrain/instanced prepass shaders.
    gpu::PrepassCameraUniforms camera_uniforms = {};
    camera_uniforms.view = draw_camera.view;
    camera_uniforms.projection = draw_camera.projection; // jittered when TAA jitter is on
    camera_uniforms.currViewProjectionNoJitter = frame_cam_.view_proj_unjittered;
    camera_uniforms.prevViewProjectionNoJitter = prev_view_proj_unjittered_;

    if (frame.should_draw_ground()) {
        terrain_->render_prepass(pass, cmd, camera_uniforms, &frame_frustum_);
    }

    prepass_draw_instanced(pass, cmd, camera_uniforms);
    prepass_draw_static_models(pass, cmd, draw_camera);
    prepass_draw_skinned_models(pass, cmd, draw_camera);

    if (frame.should_draw_grass() && gfx.grass_enabled && grass_renderer_) {
        grass_renderer_->render_prepass(pass, cmd, draw_camera.view, draw_camera.projection, draw_camera.position,
                                        frame_cam_.view_proj_unjittered, prev_view_proj_unjittered_);
    }

    SDL_EndGPURenderPass(pass);
    return true;
}

void SceneRenderer::prepass_draw_instanced(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                                           const gpu::PrepassCameraUniforms& camera_uniforms) {
    if (instance_batches_touched_.empty() || !instance_storage_buffer_) {
        return;
    }

    SDL_GPUBuffer* storage_buf = instance_storage_buffer_->handle();

    gpu::PipelineType bound_type = gpu::PipelineType::Count;
    auto ensure_pipeline_type = [&](gpu::PipelineType want) -> bool {
        if (want == bound_type) {
            return true;
        }
        gpu::GPUPipeline* pipeline = pipeline_registry_->get_pipeline(want);
        if (!pipeline) {
            return false;
        }
        pipeline->bind(pass);
        SDL_PushGPUVertexUniformData(cmd, 0, &camera_uniforms, sizeof(camera_uniforms));
        SDL_BindGPUVertexStorageBuffers(pass, 0, &storage_buf, 1);
        bound_type = want;
        return true;
    };
    auto ensure_pipeline = [&](bool double_sided) -> bool {
        return ensure_pipeline_type(double_sided ? gpu::PipelineType::DepthPrepassInstancedModelDoubleSided
                                                 : gpu::PipelineType::DepthPrepassInstancedModel);
    };

    // Ascending handle order matches the packing order in upload_instance_buffers.
    // Draw stats accumulate into a section-local partial, merged once after
    // the walk (per-chunk merge discipline — see FrameStatsPartial).
    FrameStatsPartial stats_partial;
    uint32_t base_instance = 0;
    for (ModelHandle batch_handle = 0; batch_handle < instance_batches_.size(); ++batch_handle) {
        const auto& instances = instance_batches_[batch_handle];
        if (instances.empty()) {
            continue;
        }

        Model* model = (batch_handle != INVALID_MODEL_HANDLE) ? model_manager_->get_model(batch_handle) : nullptr;
        if (!model) {
            base_instance += static_cast<uint32_t>(instances.size());
            continue;
        }

        const uint32_t instance_count = static_cast<uint32_t>(instances.size());

        for (auto& mesh : model->meshes) {
            if (mesh.shadow_only || !mesh.uploaded || !mesh.vertex_buffer || !mesh.index_buffer ||
                mesh.index_count() == 0) {
                continue;
            }
            // BLEND writes no depth; MASK goes through the alpha-tested
            // masked variant so foliage participates in GTAO + velocity.
            if (mesh.material.alpha_mode == AlphaMode::Blend) {
                continue;
            }
            if (mesh.material.alpha_mode == AlphaMode::Mask) {
                if (!ensure_pipeline_type(gpu::PipelineType::DepthPrepassInstancedModelMasked)) {
                    continue;
                }
                SDL_GPUTextureSamplerBinding tex_binding = {
                    (mesh.has_texture && mesh.texture) ? mesh.texture->handle()
                                                       : dummy_white_texture_->handle(),
                    default_sampler_};
                SDL_BindGPUFragmentSamplers(pass, 0, &tex_binding, 1);
                gpu::PrepassMaskedUniforms masked = {};
                masked.alphaCutoff = mesh.material.alpha_cutoff;
                SDL_PushGPUFragmentUniformData(cmd, 0, &masked, sizeof(masked));
            } else if (!ensure_pipeline(mesh.material.double_sided)) {
                continue;
            }

            mesh.bind_buffers(pass);
            if (collect_stats_) {
                stats_partial.draw_calls++;
                stats_partial.triangle_count += mesh.index_count() / 3 * instance_count;
            }
            SDL_DrawGPUIndexedPrimitives(pass, mesh.index_count(), instance_count, 0, 0, base_instance);
        }

        base_instance += instance_count;
    }
    stats_partial.merge_into(render_stats_);
}

void SceneRenderer::prepass_draw_static_models(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                                               const CameraState& draw_camera) {
    if (non_instanced_commands_.empty()) {
        return;
    }

    gpu::PipelineType bound_type = gpu::PipelineType::Count;
    auto ensure_pipeline_type = [&](gpu::PipelineType want) -> bool {
        if (want == bound_type) {
            return true;
        }
        gpu::GPUPipeline* pipeline = pipeline_registry_->get_pipeline(want);
        if (!pipeline) {
            return false;
        }
        pipeline->bind(pass);
        bound_type = want;
        return true;
    };
    auto ensure_pipeline = [&](bool double_sided) -> bool {
        return ensure_pipeline_type(double_sided ? gpu::PipelineType::DepthPrepassModelDoubleSided
                                                 : gpu::PipelineType::DepthPrepassModel);
    };

    // Draw stats accumulate into a section-local partial, merged once after
    // the walk (per-chunk merge discipline — see FrameStatsPartial).
    FrameStatsPartial stats_partial;
    for (const auto* mc : non_instanced_commands_) {
        Model* model = model_manager_->get_model(mc->model_handle);
        if (!model) {
            continue;
        }

        gpu::PrepassModelUniforms u = {};
        u.model = mc->transform;
        u.view = draw_camera.view;
        u.projection = draw_camera.projection;
        const MotionHistory::Entry* hist = motion_history_.find_prev(mc->motion_id);
        u.prevModel = hist ? hist->model : mc->transform;
        u.currViewProjectionNoJitter = frame_cam_.view_proj_unjittered;
        u.prevViewProjectionNoJitter = prev_view_proj_unjittered_;

        bool uniforms_pushed = false;
        for (auto& mesh : model->meshes) {
            if (mesh.shadow_only || !mesh.uploaded || !mesh.vertex_buffer || !mesh.index_buffer ||
                mesh.index_count() == 0) {
                continue;
            }
            if (mesh.material.alpha_mode == AlphaMode::Blend) {
                continue; // BLEND writes no depth
            }
            if (mesh.material.alpha_mode == AlphaMode::Mask) {
                // Alpha-tested masked variant: foliage joins GTAO + velocity.
                if (!ensure_pipeline_type(gpu::PipelineType::DepthPrepassModelMasked)) {
                    continue;
                }
                SDL_GPUTextureSamplerBinding tex_binding = {
                    (mesh.has_texture && mesh.texture) ? mesh.texture->handle()
                                                       : dummy_white_texture_->handle(),
                    default_sampler_};
                SDL_BindGPUFragmentSamplers(pass, 0, &tex_binding, 1);
                gpu::PrepassMaskedUniforms masked = {};
                masked.alphaCutoff = mesh.material.alpha_cutoff;
                SDL_PushGPUFragmentUniformData(cmd, 0, &masked, sizeof(masked));
            } else if (!ensure_pipeline(mesh.material.double_sided)) {
                continue;
            }
            if (!uniforms_pushed) {
                // Pushed uniform data survives pipeline binds within the pass.
                SDL_PushGPUVertexUniformData(cmd, 0, &u, sizeof(u));
                uniforms_pushed = true;
            }

            mesh.bind_buffers(pass);
            if (collect_stats_) {
                stats_partial.draw_calls++;
                stats_partial.triangle_count += mesh.index_count() / 3;
            }
            SDL_DrawGPUIndexedPrimitives(pass, mesh.index_count(), 1, 0, 0, 0);
        }
    }
    stats_partial.merge_into(render_stats_);
}

void SceneRenderer::prepass_draw_skinned_models(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                                                const CameraState& draw_camera) {
    if (skinned_resolved_.empty() || !current_bone_palette_buffer()) {
        return;
    }

    // Both palette buffers bound once: slot 0 = current, slot 1 = previous
    // (falls back to current when there is no previous buffer yet — draws
    // then carry the NO_PREV_PALETTE sentinel and never read slot 1's data).
    SDL_GPUBuffer* curr_buf = current_bone_palette_buffer()->handle();
    SDL_GPUBuffer* prev_buf =
        (prev_palette_buffer_valid_ && prev_bone_palette_buffer()) ? prev_bone_palette_buffer()->handle() : curr_buf;
    SDL_GPUBuffer* palette_bufs[2] = {curr_buf, prev_buf};

    gpu::PipelineType bound_type = gpu::PipelineType::Count;
    auto ensure_pipeline = [&](bool double_sided) -> bool {
        const gpu::PipelineType want = double_sided ? gpu::PipelineType::DepthPrepassSkinnedModelDoubleSided
                                                    : gpu::PipelineType::DepthPrepassSkinnedModel;
        if (want == bound_type) {
            return true;
        }
        gpu::GPUPipeline* pipeline = pipeline_registry_->get_pipeline(want);
        if (!pipeline) {
            return false;
        }
        pipeline->bind(pass);
        SDL_BindGPUVertexStorageBuffers(pass, 0, palette_bufs, 2);
        bound_type = want;
        return true;
    };

    // Draw stats accumulate into a section-local partial, merged once after
    // the walk (per-chunk merge discipline — see FrameStatsPartial).
    FrameStatsPartial stats_partial;
    for (const auto& rs : skinned_resolved_) {
        if (!rs.main_visible) {
            continue;
        }

        gpu::PrepassModelUniforms u = {};
        u.model = rs.cmd->transform;
        u.view = draw_camera.view;
        u.projection = draw_camera.projection;
        u.prevModel = rs.prev_model;
        u.currViewProjectionNoJitter = frame_cam_.view_proj_unjittered;
        u.prevViewProjectionNoJitter = prev_view_proj_unjittered_;
        u.paletteOffset = rs.palette_offset;
        u.prevPaletteOffset = rs.prev_palette_offset;

        bool uniforms_pushed = false;
        for (auto& mesh : rs.model->meshes) {
            if (mesh.shadow_only || !mesh.uploaded || !mesh.vertex_buffer || !mesh.index_buffer ||
                mesh.index_count() == 0) {
                continue;
            }
            // Opaque only: no masked skinned prepass variant (static/instanced
            // MASK foliage has one; skinned MASK meshes are rare and stay out).
            if (mesh.material.alpha_mode != AlphaMode::Opaque) {
                continue;
            }
            if (!ensure_pipeline(mesh.material.double_sided)) {
                continue;
            }
            if (!uniforms_pushed) {
                SDL_PushGPUVertexUniformData(cmd, 0, &u, sizeof(u));
                uniforms_pushed = true;
            }

            mesh.bind_buffers(pass);
            if (collect_stats_) {
                stats_partial.draw_calls++;
                stats_partial.triangle_count += mesh.index_count() / 3;
            }
            SDL_DrawGPUIndexedPrimitives(pass, mesh.index_count(), 1, 0, 0, 0);
        }
    }
    stats_partial.merge_into(render_stats_);
}

void SceneRenderer::render_instanced_shadow_models(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                                                   const glm::mat4& light_view_projection, int cascade) {
    if (cascade < 0 || cascade >= 4) {
        return;
    }
    const auto& cascade_batches = shadow_instance_batches_[cascade];
    if (shadow_batches_touched_[cascade].empty() || !pass || !shadow_instance_storage_buffer_) {
        return;
    }

    auto* pipeline = pipeline_registry_->get_instanced_shadow_model_pipeline();
    if (!pipeline) {
        return;
    }

    pipeline->bind(pass);

    gpu::InstancedShadowUniforms shadow_uniforms = {};
    shadow_uniforms.lightViewProjection = light_view_projection;
    SDL_PushGPUVertexUniformData(cmd, 0, &shadow_uniforms, sizeof(shadow_uniforms));

    SDL_GPUBuffer* storage_buf = shadow_instance_storage_buffer_->handle();
    SDL_BindGPUVertexStorageBuffers(pass, 0, &storage_buf, 1);

    // Track last state to skip redundant pushes/binds for back-to-back meshes.
    SDL_GPUTexture* last_tex = nullptr;
    int last_has_tex = -1;

    // Ascending handle order matches the per-cascade packing order in upload_instance_buffers.
    // Draw stats accumulate into a section-local partial, merged once after
    // the walk (per-chunk merge discipline — see FrameStatsPartial).
    FrameStatsPartial stats_partial;
    uint32_t base_instance = shadow_cascade_base_instance_[cascade];
    for (ModelHandle batch_handle = 0; batch_handle < cascade_batches.size(); ++batch_handle) {
        const auto& instances = cascade_batches[batch_handle];
        if (instances.empty()) {
            continue;
        }

        Model* model = (batch_handle != INVALID_MODEL_HANDLE) ? model_manager_->get_model(batch_handle) : nullptr;
        if (!model) {
            base_instance += static_cast<uint32_t>(instances.size());
            continue;
        }

        uint32_t instance_count = static_cast<uint32_t>(instances.size());

        for (auto& mesh : model->meshes) {
            if (!mesh.cast_shadows) {
                continue; // skip leaves / transparent fluff
            }
            BRACKEN_ASSERT(mesh.uploaded, "Mesh not uploaded before instanced shadow render");
            if (!mesh.uploaded) {
                continue;
            }
            if (!mesh.vertex_buffer || !mesh.index_buffer || mesh.index_count() == 0) {
                continue;
            }

            int has_tex = (mesh.has_texture && mesh.texture) ? 1 : 0;
            if (has_tex != last_has_tex) {
                gpu::ShadowFragUniforms shadow_frag = {};
                shadow_frag.hasTexture = has_tex;
                SDL_PushGPUFragmentUniformData(cmd, 0, &shadow_frag, sizeof(shadow_frag));
                last_has_tex = has_tex;
            }

            if (has_tex && default_sampler_) {
                SDL_GPUTexture* th = mesh.texture->handle();
                if (th != last_tex) {
                    SDL_GPUTextureSamplerBinding tb = {th, default_sampler_};
                    SDL_BindGPUFragmentSamplers(pass, 0, &tb, 1);
                    last_tex = th;
                }
            } else if (default_sampler_ && dummy_white_texture_ && last_tex != dummy_white_texture_->handle()) {
                SDL_GPUTextureSamplerBinding dummy = {dummy_white_texture_->handle(), default_sampler_};
                SDL_BindGPUFragmentSamplers(pass, 0, &dummy, 1);
                last_tex = dummy_white_texture_->handle();
            }

            mesh.bind_buffers(pass);
            if (collect_stats_) {
                stats_partial.draw_calls++;
                stats_partial.triangle_count += mesh.index_count() / 3 * instance_count;
            }
            SDL_DrawGPUIndexedPrimitives(pass, mesh.index_count(), instance_count, 0, 0, base_instance);
        }

        base_instance += instance_count;
    }
    stats_partial.merge_into(render_stats_);
}

void SceneRenderer::render_skinned_model_command_inner(const SkinnedDrawCommand& cmd, Model& model,
                                                       const CameraState& camera, uint32_t palette_offset) {
    // Model was resolved once in pack_bone_palettes(); no per-draw lookup.
    if (!main_render_pass_) {
        return;
    }

    SDL_GPUCommandBuffer* gpu_cmd = context_->current_command_buffer();
    if (!gpu_cmd) {
        return;
    }

    const glm::mat4& model_mat = cmd.transform;

    gpu::ModelTransformUniforms transform_uniforms = {};
    transform_uniforms.model = model_mat;
    transform_uniforms.view = camera.view;
    transform_uniforms.projection = camera.projection;
    transform_uniforms.cameraPos = camera.position;
    transform_uniforms.normalMatrix = compute_normal_matrix(model_mat);
    transform_uniforms.paletteOffset = palette_offset;

    bool fog_active = !cmd.no_fog && frame_fog_active_;

    gpu::ModelLightingUniforms lighting_uniforms = make_model_lighting(camera, fog_active);
    lighting_uniforms.tintColor = cmd.tint;

    SDL_PushGPUVertexUniformData(gpu_cmd, 0, &transform_uniforms, sizeof(transform_uniforms));

    // Per-section stats partial, merged into render_stats_ once at the end
    // (per-chunk merge discipline — see FrameStatsPartial in render_stats.hpp).
    FrameStatsPartial stats_partial;
    for (auto& mesh : model.meshes) {
        if (mesh.shadow_only) {
            continue;
        }
        BRACKEN_ASSERT(mesh.uploaded, "Mesh not uploaded before skinned render");
        if (!mesh.uploaded) {
            continue;
        }

        if (!mesh.vertex_buffer || !mesh.index_buffer) {
            continue;
        }
        if (mesh.index_count() == 0) {
            continue;
        }

        // glTF alphaMode BLEND: defer to the sorted transparent pass.
        if (mesh.material.alpha_mode == AlphaMode::Blend) {
            queue_transparent_mesh(mesh, model_mat, transform_uniforms.normalMatrix, cmd.tint, cmd.no_fog,
                                   palette_offset, model, camera);
            continue;
        }

        // glTF doubleSided selects the no-cull skinned variant; after a depth
        // prepass OPAQUE meshes use EQUAL/write-off, MASK keeps GE + write.
        const bool wants_equal = frame_prepass_active_ && mesh.material.alpha_mode == AlphaMode::Opaque;
        if ((mesh.material.double_sided != skinned_pipeline_double_sided_ ||
             wants_equal != skinned_pipeline_depth_equal_) &&
            !bind_skinned_pipeline(mesh.material.double_sided, wants_equal)) {
            continue;
        }

        apply_mesh_material(lighting_uniforms, mesh);
        if (!model_lighting_valid_ ||
            std::memcmp(&last_model_lighting_, &lighting_uniforms, sizeof(lighting_uniforms)) != 0) {
            SDL_PushGPUFragmentUniformData(gpu_cmd, 0, &lighting_uniforms, sizeof(lighting_uniforms));
            last_model_lighting_ = lighting_uniforms;
            model_lighting_valid_ = true;
        }

        bind_mesh_material_textures(mesh);

        mesh.bind_buffers(main_render_pass_);
        if (collect_stats_) {
            stats_partial.draw_calls++;
            stats_partial.triangle_count += mesh.index_count() / 3;
        }
        SDL_DrawGPUIndexedPrimitives(main_render_pass_, mesh.index_count(), 1, 0, 0, 0);
    }
    stats_partial.merge_into(render_stats_);
}

// ============================================================================
// Transparent (glTF alphaMode BLEND) Rendering
// ============================================================================

void SceneRenderer::queue_transparent_mesh(const Mesh& mesh, const glm::mat4& transform,
                                           const glm::mat4& normal_matrix, const glm::vec4& tint, bool no_fog,
                                           uint32_t palette_offset, const Model& model, const CameraState& camera) {
    const glm::vec3 world_center = glm::vec3(transform * glm::vec4(model.bounding_center, 1.0f));
    const glm::vec3 d = world_center - camera.position;
    transparent_draws_.push_back(
        {&mesh, transform, normal_matrix, tint, palette_offset, no_fog, d.x * d.x + d.y * d.y + d.z * d.z});
}

// Draws the queued BLEND meshes back-to-front (per queued draw — the model's
// bounding-sphere center stands in for each of its meshes) with depth test on
// and depth writes off. Limitations: no OIT, no intra-mesh sorting; see
// docs/architecture.md.
void SceneRenderer::render_transparent_models(const CameraState& camera) {
    if (transparent_draws_.empty()) {
        return;
    }
    SDL_GPUCommandBuffer* gpu_cmd = context_->current_command_buffer();
    if (!main_render_pass_ || !gpu_cmd) {
        transparent_draws_.clear();
        return;
    }

    std::sort(transparent_draws_.begin(), transparent_draws_.end(),
              [](const TransparentDraw& a, const TransparentDraw& b) { return a.view_dist_sq > b.view_dist_sq; });

    // Draw stats accumulate into a section-local partial, merged once after
    // the walk (per-chunk merge discipline — see FrameStatsPartial).
    FrameStatsPartial stats_partial;
    bool pipeline_bound = false;
    gpu::PipelineType bound_type = gpu::PipelineType::Count;
    for (const TransparentDraw& draw : transparent_draws_) {
        const Mesh& mesh = *draw.mesh;
        const bool skinned = draw.palette_offset != INVALID_PALETTE_OFFSET;
        const gpu::PipelineType want =
            skinned ? (mesh.material.double_sided ? gpu::PipelineType::SkinnedModelBlendDoubleSided
                                                  : gpu::PipelineType::SkinnedModelBlend)
                    : (mesh.material.double_sided ? gpu::PipelineType::ModelBlendDoubleSided
                                                  : gpu::PipelineType::ModelBlend);
        if (!pipeline_bound || want != bound_type) {
            if (skinned && !current_bone_palette_buffer()) {
                continue;
            }
            gpu::GPUPipeline* pipeline = pipeline_registry_->get_pipeline(want);
            if (!pipeline) {
                continue;
            }
            pipeline->bind(main_render_pass_);
            reset_model_bind_state();
            bind_shadow_data(main_render_pass_, gpu_cmd, 1);
            bind_cluster_lights(main_render_pass_, gpu_cmd, 2);
            // Transparent surfaces never receive screen AO (it belongs to
            // the opaque surface behind them) — bind the white dummy. IBL is
            // view/normal-based, not screen-space, so it stays on for BLEND;
            // local light shadows are world-space too and stay on as well.
            bind_screen_ao(6, /*enabled=*/false);
            bind_ibl(7);
            bind_local_shadow_atlas(10);
            if (skinned) {
                SDL_GPUBuffer* palette_buf = current_bone_palette_buffer()->handle();
                SDL_BindGPUVertexStorageBuffers(main_render_pass_, 0, &palette_buf, 1);
            }
            bound_type = want;
            pipeline_bound = true;
        }

        gpu::ModelTransformUniforms transform_uniforms = {};
        transform_uniforms.model = draw.transform;
        transform_uniforms.view = camera.view;
        transform_uniforms.projection = camera.projection;
        transform_uniforms.cameraPos = camera.position;
        transform_uniforms.normalMatrix = draw.normal_matrix;
        transform_uniforms.paletteOffset = skinned ? draw.palette_offset : 0;
        SDL_PushGPUVertexUniformData(gpu_cmd, 0, &transform_uniforms, sizeof(transform_uniforms));

        gpu::ModelLightingUniforms lighting_uniforms =
            make_model_lighting(camera, !draw.no_fog && frame_fog_active_);
        lighting_uniforms.tintColor = draw.tint;
        apply_mesh_material(lighting_uniforms, mesh);
        if (!model_lighting_valid_ ||
            std::memcmp(&last_model_lighting_, &lighting_uniforms, sizeof(lighting_uniforms)) != 0) {
            SDL_PushGPUFragmentUniformData(gpu_cmd, 0, &lighting_uniforms, sizeof(lighting_uniforms));
            last_model_lighting_ = lighting_uniforms;
            model_lighting_valid_ = true;
        }

        bind_mesh_material_textures(mesh);
        mesh.bind_buffers(main_render_pass_);
        if (collect_stats_) {
            stats_partial.draw_calls++;
            stats_partial.triangle_count += mesh.index_count() / 3;
        }
        SDL_DrawGPUIndexedPrimitives(main_render_pass_, mesh.index_count(), 1, 0, 0, 0);
    }
    stats_partial.merge_into(render_stats_);
    transparent_draws_.clear();
}

// ============================================================================
// UI Rendering
// ============================================================================

void SceneRenderer::render_ui_commands(const FramePacket& frame) {
    // Packet UI text/labels are string_views over packet-arena copies —
    // valid through end_ui()'s execute() because the caller retires the
    // packet only after render_frame returns.
    for (const auto& cmd : frame.ui_commands()) {
        std::visit(
            [this](const auto& data) {
                using T = std::decay_t<decltype(data)>;

                if constexpr (std::is_same_v<T, UiFilledRectCommand>) {
                    ui_->draw_filled_rect(data.x, data.y, data.w, data.h, data.color);
                } else if constexpr (std::is_same_v<T, UiRectOutlineCommand>) {
                    ui_->draw_rect_outline(data.x, data.y, data.w, data.h, data.color, data.line_width);
                } else if constexpr (std::is_same_v<T, UiCircleCommand>) {
                    ui_->draw_circle(data.x, data.y, data.radius, data.color, data.segments);
                } else if constexpr (std::is_same_v<T, UiCircleOutlineCommand>) {
                    ui_->draw_circle_outline(data.x, data.y, data.radius, data.color, data.line_width, data.segments);
                } else if constexpr (std::is_same_v<T, UiLineCommand>) {
                    ui_->draw_line(data.x1, data.y1, data.x2, data.y2, data.color, data.line_width);
                } else if constexpr (std::is_same_v<T, UiTextCommand>) {
                    ui_->draw_text(data.text, data.x, data.y, data.color, data.scale);
                } else if constexpr (std::is_same_v<T, UiButtonCommand>) {
                    // draw_button still takes std::string; the label copy is
                    // per-button-per-frame and buttons are rare.
                    ui_->draw_button(data.x, data.y, data.w, data.h, std::string(data.label), data.color,
                                     data.selected);
                }
            },
            cmd.data);
    }
}

void SceneRenderer::draw_billboard_3d(const BillboardDrawCommand& cmd, const CameraState& camera) {
    glm::vec4 world_pos(cmd.world_x, cmd.world_y, cmd.world_z, 1.0f);
    glm::vec4 clip_pos = camera.projection * camera.view * world_pos;
    if (clip_pos.w <= 0.01f) {
        return;
    }

    glm::vec3 ndc = glm::vec3(clip_pos) / clip_pos.w;
    if (ndc.x < -1.5f || ndc.x > 1.5f || ndc.y < -1.5f || ndc.y > 1.5f || ndc.z < -1.0f || ndc.z > 1.0f) {
        return;
    }

    float screen_x = (ndc.x * 0.5f + 0.5f) * context_->width();
    float screen_y = (1.0f - (ndc.y * 0.5f + 0.5f)) * context_->height();

    float distance_scale = 100.0f / clip_pos.w;
    distance_scale = glm::clamp(distance_scale, 0.3f, 1.5f);

    float bar_w = cmd.width * 2.0f * distance_scale;
    float bar_h = cmd.width * 0.4f * distance_scale;

    float x = screen_x - bar_w * 0.5f;
    float y = screen_y - bar_h * 0.5f;

    ui_->draw_filled_rect(x - 1, y - 1, bar_w + 2, bar_h + 2, cmd.frame_color);
    ui_->draw_filled_rect(x, y, bar_w, bar_h, cmd.bg_color);
    float fill_w = bar_w * cmd.fill_ratio;
    ui_->draw_filled_rect(x, y, fill_w, bar_h, cmd.fill_color);
}

// ============================================================================
// Debug Line Rendering
// ============================================================================

// Builds the debug-line vertex data and uploads it inside the caller-owned
// consolidated frame copy pass (copy passes cannot begin inside a render pass,
// so the upload must happen before any render pass starts).
void SceneRenderer::upload_debug_lines(const FramePacket& frame, SDL_GPUCopyPass* copy_pass) {
    debug_line_vertex_count_ = 0;
    const std::span<const DebugLineDrawCommand> lines = frame.debug_lines();
    if (lines.empty() || !copy_pass) {
        return;
    }

    const size_t line_count = lines.size();
    const size_t vertex_count = line_count * 2;
    // Grid vertex format: float3 position + float4 color = 7 floats per vertex
    const size_t vertex_stride = sizeof(float) * 7;
    const size_t buffer_size = vertex_count * vertex_stride;

    // Recreate buffer if too small
    if (!debug_line_vertex_buffer_ || debug_line_buffer_capacity_ < line_count) {
        // Round up to avoid frequent reallocations
        size_t new_capacity = std::max(line_count, static_cast<size_t>(256));
        if (debug_line_buffer_capacity_ > 0) {
            new_capacity = std::max(new_capacity, debug_line_buffer_capacity_ * 2);
        }
        debug_line_vertex_buffer_ = gpu::GPUBuffer::create_dynamic(context_->device(), gpu::GPUBuffer::Type::Vertex,
                                                                   new_capacity * 2 * vertex_stride);
        if (!debug_line_vertex_buffer_) {
            return;
        }
        debug_line_buffer_capacity_ = new_capacity;
    }

    // Build vertex data: position(3) + color(4) per vertex.
    // Colors are packed ARGB (engine-wide convention, bracken/core/color.hpp).
    // Reuse scratch buffer to avoid per-frame alloc.
    debug_line_scratch_.resize(vertex_count * 7);
    float* dst = debug_line_scratch_.data();
    for (const auto& line : lines) {
        const glm::vec4 c = core::unpack_color(line.color);

        // Start vertex
        *dst++ = line.start.x;
        *dst++ = line.start.y;
        *dst++ = line.start.z;
        *dst++ = c.r;
        *dst++ = c.g;
        *dst++ = c.b;
        *dst++ = c.a;

        // End vertex
        *dst++ = line.end.x;
        *dst++ = line.end.y;
        *dst++ = line.end.z;
        *dst++ = c.r;
        *dst++ = c.g;
        *dst++ = c.b;
        *dst++ = c.a;
    }

    // Upload vertex data within the consolidated copy pass
    debug_line_vertex_buffer_->update(copy_pass, debug_line_scratch_.data(), buffer_size);
    debug_line_vertex_count_ = vertex_count;
}

// Binds + draws the lines uploaded by upload_debug_lines this frame.
void SceneRenderer::render_debug_lines(const CameraState& camera) {
    if (debug_line_vertex_count_ == 0 || !debug_line_vertex_buffer_ || !main_render_pass_) {
        return;
    }

    SDL_GPUCommandBuffer* cmd = context_->current_command_buffer();
    if (!cmd) {
        return;
    }

    gpu::GPUPipeline* pipeline = pipeline_registry_->get_grid_pipeline();
    if (!pipeline) {
        return;
    }

    // Bind pipeline and draw
    pipeline->bind(main_render_pass_);

    // Push view-projection matrix as vertex uniform (same as grid rendering)
    glm::mat4 vp = camera.projection * camera.view;
    SDL_PushGPUVertexUniformData(cmd, 0, &vp, sizeof(vp));

    // Bind vertex buffer
    SDL_GPUBufferBinding binding = {};
    binding.buffer = debug_line_vertex_buffer_->handle();
    binding.offset = 0;
    SDL_BindGPUVertexBuffers(main_render_pass_, 0, &binding, 1);

    // Draw all lines
    SDL_DrawGPUPrimitives(main_render_pass_, static_cast<uint32_t>(debug_line_vertex_count_), 1, 0, 0);

    if (collect_stats_) {
        render_stats_.draw_calls++;
    }
}

// ============================================================================
// Shadow Rendering
// ============================================================================

void SceneRenderer::render_shadow_passes() {
    SDL_GPUCommandBuffer* cmd = context_->current_command_buffer();
    if (!cmd) {
        return;
    }

    BRACKEN_PROFILE_ZONE_COLOR("ShadowPass", 0xE67E22);
    BRACKEN_GPU_SCOPE(cmd, "ShadowPass");
    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "ShadowPass");

    // Note: shadow_map_->update() was already called in render_frame() so that the
    // per-cascade light frustums could be used during build_instance_batches.

    // Skinned entities were already resolved (Model*, world bounds, palette
    // offset) by pack_bone_palettes(); the cascade loops below reuse that
    // scratch directly instead of re-resolving per shadow pass.
    const auto& skinned_scratch = skinned_resolved_;

    // Pre-resolve Model* for non-instanced static commands too.
    auto& static_scratch = shadow_static_scratch_;
    static_scratch.clear();
    static_scratch.reserve(non_instanced_commands_.size());
    for (const auto* model_cmd : non_instanced_commands_) {
        Model* model = model_manager_->get_model(model_cmd->model_handle);
        if (model) {
            static_scratch.push_back({model_cmd, model});
        }
    }

    SDL_GPUTextureSamplerBinding dummy_binding = {};
    if (default_sampler_ && dummy_white_texture_) {
        dummy_binding = {dummy_white_texture_->handle(), default_sampler_};
    }

    const gpu::ShadowFragUniforms frag_notex = {.hasTexture = 0};
    const gpu::ShadowFragUniforms frag_tex = {.hasTexture = 1};

    auto cascade_label = [](int c) -> const char* {
        switch (c & 3) {
            case 0:
                return "ShadowPass:Cascade0";
            case 1:
                return "ShadowPass:Cascade1";
            case 2:
                return "ShadowPass:Cascade2";
            default:
                return "ShadowPass:Cascade3";
        }
    };

    // Single render pass over the whole atlas; each cascade renders into its
    // own quadrant viewport.
    SDL_GPURenderPass* shadow_pass = shadow_map_->begin_shadow_pass(cmd);
    if (!shadow_pass) {
        return;
    }

    // Draw stats for the non-instanced static + skinned cascade walks below
    // accumulate into a section-local partial, merged once after the cascade
    // loop (per-chunk merge discipline — see FrameStatsPartial). The
    // instanced walk merges inside render_instanced_shadow_models.
    FrameStatsPartial stats_partial;
    for (int cascade = 0; cascade < shadow_map_->active_cascades(); ++cascade) {
        const char* cascade_name = cascade_label(cascade);
        BRACKEN_GPU_SCOPE(cmd, cascade_name);
        BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, cascade_name);
        shadow_map_->set_cascade_viewport(cascade);

        const auto& cascade_data = shadow_map_->cascades()[cascade];

        // All cascades now draw entities — trees use a 20-triangle canopy proxy
        // (instead of 2000 leaf quads) so even the far cascade cost is trivial.
        // Per-cascade frustum culling below still skips anything outside the
        // cascade's light frustum.

        // Render static models into shadow map (instanced)
        render_instanced_shadow_models(shadow_pass, cmd, cascade_data.light_view_projection, cascade);

        // Render non-instanced static models into shadow map.
        // Per-cascade frustum cull so near cascades don't draw distant casters.
        if (!static_scratch.empty()) {
            auto* shadow_pipeline = pipeline_registry_->get_shadow_model_pipeline();
            if (shadow_pipeline) {
                shadow_pipeline->bind(shadow_pass);
                SDL_GPUTexture* last_tex = nullptr;
                int last_has_tex = -1;
                for (const auto& rs : static_scratch) {
                    // Per-cascade cull: skip if not in this cascade's light frustum.
                    const glm::mat4& t = rs.cmd->transform;
                    glm::vec3 wc = glm::vec3(t * glm::vec4(rs.model->bounding_center, 1.0f));
                    float wr = rs.model->bounding_half_diag * max_scale_factor(t);
                    if (!shadow_cascade_frustums_[cascade].intersects_sphere(wc, wr)) {
                        continue;
                    }

                    gpu::ShadowTransformUniforms shadow_uniforms = {};
                    shadow_uniforms.lightViewProjection = cascade_data.light_view_projection;
                    shadow_uniforms.model = rs.cmd->transform;
                    SDL_PushGPUVertexUniformData(cmd, 0, &shadow_uniforms, sizeof(shadow_uniforms));

                    for (auto& mesh : rs.model->meshes) {
                        if (!mesh.cast_shadows) {
                            continue;
                        }
                        BRACKEN_ASSERT(mesh.uploaded, "Mesh not uploaded before non-instanced shadow render");
                        if (!mesh.uploaded) {
                            continue;
                        }
                        if (!mesh.vertex_buffer || !mesh.index_buffer || mesh.index_count() == 0) {
                            continue;
                        }

                        int has_tex = (mesh.has_texture && mesh.texture) ? 1 : 0;
                        if (has_tex != last_has_tex) {
                            SDL_PushGPUFragmentUniformData(cmd, 0, has_tex ? &frag_tex : &frag_notex,
                                                           sizeof(gpu::ShadowFragUniforms));
                            last_has_tex = has_tex;
                        }

                        if (has_tex && default_sampler_) {
                            SDL_GPUTexture* th = mesh.texture->handle();
                            if (th != last_tex) {
                                SDL_GPUTextureSamplerBinding tb = {th, default_sampler_};
                                SDL_BindGPUFragmentSamplers(shadow_pass, 0, &tb, 1);
                                last_tex = th;
                            }
                        } else if (default_sampler_ && dummy_white_texture_ && last_tex != dummy_binding.texture) {
                            SDL_BindGPUFragmentSamplers(shadow_pass, 0, &dummy_binding, 1);
                            last_tex = dummy_binding.texture;
                        }

                        mesh.bind_buffers(shadow_pass);
                        if (collect_stats_) {
                            stats_partial.draw_calls++;
                            stats_partial.triangle_count += mesh.index_count() / 3;
                        }
                        SDL_DrawGPUIndexedPrimitives(shadow_pass, mesh.index_count(), 1, 0, 0, 0);
                    }
                }
            }
        }

        // Render skinned models into shadow map. The skinned shadow pipeline
        // is depth-only opaque (no alpha test), so no fragment state at all.
        // Bone palettes were uploaded once in the frame copy pass; each draw
        // only pushes the small transform uniform with its palette offset.
        if (!skinned_scratch.empty() && current_bone_palette_buffer()) {
            auto* shadow_skinned_pipeline = pipeline_registry_->get_shadow_skinned_model_pipeline();
            if (shadow_skinned_pipeline) {
                shadow_skinned_pipeline->bind(shadow_pass);
                SDL_GPUBuffer* palette_buf = current_bone_palette_buffer()->handle();
                SDL_BindGPUVertexStorageBuffers(shadow_pass, 0, &palette_buf, 1);

                for (const auto& rs : skinned_scratch) {
                    // Per-cascade cull: skip if not in this cascade's light frustum.
                    if (!shadow_cascade_frustums_[cascade].intersects_sphere(rs.world_center, rs.world_radius)) {
                        continue;
                    }

                    const SkinnedDrawCommand* data = rs.cmd;
                    Model* model = rs.model;

                    gpu::ShadowSkinnedTransformUniforms shadow_uniforms = {};
                    shadow_uniforms.lightViewProjection = cascade_data.light_view_projection;
                    shadow_uniforms.model = data->transform;
                    shadow_uniforms.paletteOffset = rs.palette_offset;
                    SDL_PushGPUVertexUniformData(cmd, 0, &shadow_uniforms, sizeof(shadow_uniforms));

                    for (auto& mesh : model->meshes) {
                        if (!mesh.cast_shadows) {
                            continue;
                        }
                        BRACKEN_ASSERT(mesh.uploaded, "Mesh not uploaded before skinned shadow render");
                        if (!mesh.uploaded) {
                            continue;
                        }
                        if (!mesh.vertex_buffer || !mesh.index_buffer || mesh.index_count() == 0) {
                            continue;
                        }

                        mesh.bind_buffers(shadow_pass);
                        if (collect_stats_) {
                            stats_partial.draw_calls++;
                            stats_partial.triangle_count += mesh.index_count() / 3;
                        }
                        SDL_DrawGPUIndexedPrimitives(shadow_pass, mesh.index_count(), 1, 0, 0, 0);
                    }
                }
            }
        }

        // Render terrain into shadow map. The terrain shadow pipeline is also
        // depth-only opaque — no fragment state needed.
        terrain_->render_shadow(shadow_pass, cmd, cascade_data.light_view_projection,
                                &shadow_cascade_frustums_[cascade]);
    }
    stats_partial.merge_into(render_stats_);

    shadow_map_->end_shadow_pass();
}


void SceneRenderer::bind_local_shadow_atlas(uint32_t slot) {
    // Local atlas + comparison sampler; the CSM atlas is a typed filler when
    // the local system is unavailable (D32 + comparison sampler keeps the
    // declared slot valid — it is only sampled for shadow_slot >= 0 lights,
    // which never happens in that case).
    SDL_GPUTexture* tex = nullptr;
    SDL_GPUSampler* smp = nullptr;
    if (local_shadows_ && local_shadows_->is_ready()) {
        tex = local_shadows_->atlas_texture();
        smp = local_shadows_->compare_sampler();
    } else if (shadow_map_->is_ready()) {
        tex = shadow_map_->atlas_texture();
        smp = shadow_map_->shadow_compare_sampler();
    }
    if (!tex || !smp || !main_render_pass_) {
        return;
    }
    SDL_GPUTextureSamplerBinding binding = {tex, smp};
    SDL_BindGPUFragmentSamplers(main_render_pass_, slot, &binding, 1);
}

// Draws this frame's dirty local-shadow tiles. Caster set per light: every
// model command (instanced ones draw individually here — the per-light caster
// sets are small, a handful of props inside the light radius) + every skinned
// command, filtered by mesh.cast_shadows exactly like the CSM passes. Terrain
// and grass deliberately do NOT cast local shadows (grass also doesn't cast
// in the CSM path; terrain within a local light's radius is almost always the
// receiving floor, and skipping it keeps tile cost bounded).
void SceneRenderer::render_local_shadow_passes(const FramePacket& frame, const CameraState& camera) {
    (void)camera;
    const auto& requests = local_shadows_->selector().tile_requests();
    if (requests.empty()) {
        return;
    }
    SDL_GPUCommandBuffer* cmd = context_->current_command_buffer();
    if (!cmd) {
        return;
    }

    BRACKEN_PROFILE_ZONE_COLOR("LocalShadowPass", 0xE67E22);
    BRACKEN_GPU_SCOPE(cmd, "LocalShadowPass");
    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "LocalShadowPass");

    auto* clear_pipeline = pipeline_registry_->get_pipeline(gpu::PipelineType::LocalShadowClear);
    auto* model_pipeline = pipeline_registry_->get_shadow_model_pipeline();
    auto* skinned_pipeline = pipeline_registry_->get_shadow_skinned_model_pipeline();
    if (!clear_pipeline || !model_pipeline) {
        return;
    }

    SDL_GPURenderPass* pass = local_shadows_->begin_pass(cmd);
    if (!pass) {
        return;
    }

    const gpu::ShadowFragUniforms frag_notex = {.hasTexture = 0};
    const gpu::ShadowFragUniforms frag_tex = {.hasTexture = 1};
    SDL_GPUTextureSamplerBinding dummy_binding = {};
    if (default_sampler_ && dummy_white_texture_) {
        dummy_binding = {dummy_white_texture_->handle(), default_sampler_};
    }

    // Per-light caster lists, rebuilt when the request stream moves to a new
    // light (a point light's 6 face requests arrive consecutively).
    glm::vec3 casters_light_pos(std::numeric_limits<float>::max());
    float casters_light_radius = -1.0f;
    Frustum tile_frustum;

    // Draw stats accumulate into a section-local partial, merged once after
    // the tile walk (per-chunk merge discipline — see FrameStatsPartial).
    FrameStatsPartial stats_partial;
    for (const auto& req : requests) {
        if (req.light_pos != casters_light_pos || req.light_radius != casters_light_radius) {
            casters_light_pos = req.light_pos;
            casters_light_radius = req.light_radius;
            local_static_casters_.clear();
            for (const auto& mc : frame.model_commands()) {
                Model* model = model_manager_->get_model(mc.model_handle);
                if (!model) {
                    continue;
                }
                const glm::vec3 wc = glm::vec3(mc.transform * glm::vec4(model->bounding_center, 1.0f));
                const float wr = model->bounding_half_diag * max_scale_factor(mc.transform);
                const glm::vec3 d = wc - req.light_pos;
                const float rr = wr + req.light_radius;
                if (glm::dot(d, d) <= rr * rr) {
                    local_static_casters_.push_back({&mc, model, wc, wr});
                }
            }
            local_skinned_casters_.clear();
            for (const auto& rs : skinned_resolved_) {
                const glm::vec3 d = rs.world_center - req.light_pos;
                const float rr = rs.world_radius + req.light_radius;
                if (glm::dot(d, d) <= rr * rr) {
                    local_skinned_casters_.push_back(&rs);
                }
            }
        }

        tile_frustum.extract_from_matrix(req.view_proj);

        // Clear the tile to far-plane depth: fullscreen triangle with the
        // viewport depth range pinned to [1,1] (cached tiles persist, so the
        // pass LOADs and each dirty tile clears itself).
        local_shadows_->set_tile_viewport(req.tile, /*clear_depth_range=*/true);
        clear_pipeline->bind(pass);
        SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
        local_shadows_->set_tile_viewport(req.tile, /*clear_depth_range=*/false);

        // Static / instanced-model casters (alpha-tested where textured).
        if (!local_static_casters_.empty()) {
            model_pipeline->bind(pass);
            SDL_GPUTexture* last_tex = nullptr;
            int last_has_tex = -1;
            for (const auto& caster : local_static_casters_) {
                if (!tile_frustum.intersects_sphere(caster.center, caster.radius)) {
                    continue;
                }
                gpu::ShadowTransformUniforms u = {};
                u.lightViewProjection = req.view_proj;
                u.model = caster.cmd->transform;
                SDL_PushGPUVertexUniformData(cmd, 0, &u, sizeof(u));
                for (auto& mesh : caster.model->meshes) {
                    if (!mesh.cast_shadows || !mesh.uploaded || !mesh.vertex_buffer || !mesh.index_buffer ||
                        mesh.index_count() == 0) {
                        continue;
                    }
                    const int has_tex = (mesh.has_texture && mesh.texture) ? 1 : 0;
                    if (has_tex != last_has_tex) {
                        SDL_PushGPUFragmentUniformData(cmd, 0, has_tex ? &frag_tex : &frag_notex,
                                                       sizeof(gpu::ShadowFragUniforms));
                        last_has_tex = has_tex;
                    }
                    if (has_tex && default_sampler_) {
                        SDL_GPUTexture* th = mesh.texture->handle();
                        if (th != last_tex) {
                            SDL_GPUTextureSamplerBinding tb = {th, default_sampler_};
                            SDL_BindGPUFragmentSamplers(pass, 0, &tb, 1);
                            last_tex = th;
                        }
                    } else if (dummy_binding.texture && last_tex != dummy_binding.texture) {
                        SDL_BindGPUFragmentSamplers(pass, 0, &dummy_binding, 1);
                        last_tex = dummy_binding.texture;
                    }
                    mesh.bind_buffers(pass);
                    if (collect_stats_) {
                        stats_partial.draw_calls++;
                        stats_partial.triangle_count += mesh.index_count() / 3;
                    }
                    SDL_DrawGPUIndexedPrimitives(pass, mesh.index_count(), 1, 0, 0, 0);
                }
            }
        }

        // Skinned casters (depth-only, palettes already uploaded this frame).
        if (!local_skinned_casters_.empty() && skinned_pipeline && current_bone_palette_buffer()) {
            skinned_pipeline->bind(pass);
            SDL_GPUBuffer* palette_buf = current_bone_palette_buffer()->handle();
            SDL_BindGPUVertexStorageBuffers(pass, 0, &palette_buf, 1);
            for (const auto* rs : local_skinned_casters_) {
                if (!tile_frustum.intersects_sphere(rs->world_center, rs->world_radius)) {
                    continue;
                }
                gpu::ShadowSkinnedTransformUniforms u = {};
                u.lightViewProjection = req.view_proj;
                u.model = rs->cmd->transform;
                u.paletteOffset = rs->palette_offset;
                SDL_PushGPUVertexUniformData(cmd, 0, &u, sizeof(u));
                for (auto& mesh : rs->model->meshes) {
                    if (!mesh.cast_shadows || !mesh.uploaded || !mesh.vertex_buffer || !mesh.index_buffer ||
                        mesh.index_count() == 0) {
                        continue;
                    }
                    mesh.bind_buffers(pass);
                    if (collect_stats_) {
                        stats_partial.draw_calls++;
                        stats_partial.triangle_count += mesh.index_count() / 3;
                    }
                    SDL_DrawGPUIndexedPrimitives(pass, mesh.index_count(), 1, 0, 0, 0);
                }
            }
        }
    }
    stats_partial.merge_into(render_stats_);

    local_shadows_->end_pass();
}

void SceneRenderer::bind_shadow_data(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, int sampler_slot) {
    if (!shadow_map_->is_ready()) {
        return;
    }

    // Atlas bound twice starting at sampler_slot: comparison sampler for
    // hardware PCF, then raw sampler for the PCSS blocker search. Material
    // textures occupy slots 3-5 and are bound per draw.
    SDL_GPUTextureSamplerBinding shadow_bindings[2];
    shadow_bindings[0] = {shadow_map_->atlas_texture(), shadow_map_->shadow_compare_sampler()};
    shadow_bindings[1] = {shadow_map_->atlas_texture(), shadow_map_->shadow_sampler()};
    SDL_BindGPUFragmentSamplers(pass, sampler_slot, shadow_bindings, 2);

    // Use frame-cached shadow uniforms (computed once in render_frame)
    SDL_PushGPUFragmentUniformData(cmd, 1, &frame_shadow_uniforms_, sizeof(frame_shadow_uniforms_));
}

void SceneRenderer::bind_cluster_lights(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, int uniform_slot) {
    if (!cluster_grid_ready_) {
        return;
    }
    // 4th SSBO = local-shadow slot data (cluster_lighting.hlsli). When the
    // local shadow system is unavailable, substitute the light-data buffer:
    // the slot must hold a valid buffer, but it is never read while every
    // light's shadow_slot is -1.
    SDL_GPUBuffer* local_slots =
        (local_shadows_ && local_shadows_->is_ready()) ? local_shadows_->slot_buffer() : nullptr;
    SDL_GPUBuffer* bufs[4] = {
        cluster_grid_.light_data_buffer(),
        cluster_grid_.cluster_offsets_buffer(),
        cluster_grid_.light_indices_buffer(),
        local_slots ? local_slots : cluster_grid_.light_data_buffer(),
    };
    if (!bufs[0] || !bufs[1] || !bufs[2]) {
        return;
    }
    SDL_BindGPUFragmentStorageBuffers(pass, 0, bufs, 4);
    // frame_cluster_params_ (not cluster_grid_.params()) so frames with zero
    // dynamic lights publish gridDim.w == 0 rather than stale counts.
    SDL_PushGPUFragmentUniformData(cmd, uniform_slot, &frame_cluster_params_, sizeof(frame_cluster_params_));
}

} // namespace bracken::scene
