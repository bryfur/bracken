#include "scene_renderer.hpp"
#include "scene_renderer_internal.hpp"
#include "bracken/contracts/frustum.hpp"
#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/log_throttle.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/effects/effect_renderer.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_debug.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/heightmap.hpp"
#include "bracken/heightmap_grid.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render/auto_exposure.hpp"
#include "bracken/render/bloom.hpp"
#include "bracken/render/color_grading.hpp"
#include "bracken/render/grass_renderer.hpp"
#include "bracken/render/gtao.hpp"
#include "bracken/render/ibl.hpp"
#include "bracken/render/post_process_stack.hpp"
#include "bracken/render/render_context.hpp"
#include "bracken/render/rt_denoise_math.hpp"
#include "bracken/render/shadow_map.hpp"
#include "bracken/render/taa.hpp"
#include "bracken/render/terrain_renderer.hpp"
#include "bracken/render/ui_renderer.hpp"
#include "bracken/render/volumetric_fog.hpp"
#include "bracken/render/water_renderer.hpp"
#include "bracken/render/world_renderer.hpp"
#include "SDL3/SDL_gpu.h"
#include "SDL3/SDL_video.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

#include "bracken/contracts/camera_state.hpp"
#include "bracken/contracts/depth_linear.hpp"
#include "bracken/contracts/graphics_settings.hpp"
#include "bracken/scene/camera_jitter.hpp"
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

bracken::UiTextureId SceneRenderer::register_ui_texture(SDL_GPUTexture* texture) {
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    return ui_ ? ui_->register_texture(texture) : bracken::UiTextureId{};
}

void SceneRenderer::unregister_ui_texture(bracken::UiTextureId id) {
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (ui_) {
        ui_->unregister_texture(id);
    }
}

const bracken::FontMetrics& SceneRenderer::font_metrics() const {
    // Tier: not sim-stage. This reaches into the LIVE renderer for a table the
    // sim stage is supposed to read from its own frozen copy
    // (`SimFrame::font_metrics()`) — which is the entire reason
    // Application::init_renderer() freezes one. Asserting here is what stops a
    // host "just asking the renderer" from on_simulate and getting away with
    // it right up until the renderer is torn down mid-frame.
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    // A single frozen empty table, so a host that asks before init() gets a
    // valid()==false answer rather than a null reference.
    static const bracken::FontMetrics k_empty{};
    return ui_ ? ui_->metrics() : k_empty;
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

    // Ray tracing. Created unconditionally but inert: with both features off
    // it allocates nothing, builds no acceleration structure, and its render()
    // returns before touching the GPU.
    rt_renderer_ = std::make_unique<render::RtRenderer>();
    if (!rt_renderer_->init(context_->device(), *pipeline_registry_)) {
        BRACKEN_LOG_WARN("scene", "ray tracing unavailable; rt_shadows/rt_reflections will no-op");
        rt_renderer_.reset();
    } else {
        rt_renderer_->resize(context_->device(), w, h);
        // Lend it this renderer's pass timeline so the traced passes show up in
        // RenderStats::pass_times_ms (and the dev Stats panel) next to every
        // other pass. Pointer to a member of this object, so the lifetime
        // requirement is met by construction.
        rt_renderer_->set_timer_pool(&gpu_timer_pool_);
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
    // Allocates nothing here: the table's buffer is created on the first frame
    // that actually bins records, so a scene with no emitter never pays for it.
    emissive_clusters_ready_ = emissive_clusters_.init(context_->device());

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
    // Before the device goes: the probe atlases and the Hi-Z pyramid's mip
    // chain + sampler are released through it.
    probe_volume_.reset();
    screen_probes_.reset();
    hzb_.reset();
    release_occlusion_resources();

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
    auto* model_pipeline = pipeline_registry_->get_pipeline(gpu::PipelineType::Model);
    auto* skinned_pipeline = pipeline_registry_->get_pipeline(gpu::PipelineType::SkinnedModel);

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

    // 1x1 zeros for the Full-DI tier's READ-ONLY STORAGE texture slots when
    // that tier is off (R5c; a third slot since R7b). Storage-read usage, not
    // SAMPLER: the four scene shaders Load() these rather than sampling them,
    // because they have no seventeenth sampler to spare
    // (shaders/local_di.hlsli). Alpha 0 is the "no answer" localDiSample
    // rejects, so even a shader that ignored rtLocalShadowMode would read no
    // contribution rather than garbage.
    //
    // TWO FORMATS, because the direction target is RG16F where the radiance
    // pair is RGBA16F, and a storage image is declared in the shader with a
    // channel count: binding a four-channel stand-in into a two-channel slot is
    // undefined rather than merely wasteful.
    {
        const std::array<uint16_t, 4> zero{0, 0, 0, 0};
        dummy_di_texture_ = gpu::GPUTexture::create_2d(
            context_->device(), 1, 1, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
            SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ);
        dummy_di_dir_texture_ = gpu::GPUTexture::create_2d(
            context_->device(), 1, 1, SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT,
            SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ);
        if (dummy_di_texture_ && dummy_di_dir_texture_) {
            // One-shot upload on its own command buffer: init() has no frame
            // command buffer to ride, and this is a 24-byte copy that happens
            // once per process.
            if (SDL_GPUCommandBuffer* upload_cmd =
                    SDL_AcquireGPUCommandBuffer(context_->device().handle())) {
                dummy_di_texture_->upload(upload_cmd, zero.data(), 1, 1);
                dummy_di_dir_texture_->upload(upload_cmd, zero.data(), 1, 1);
                SDL_SubmitGPUCommandBuffer(upload_cmd);
            }
        } else {
            BRACKEN_LOG_WARN(LOG_TAG, "Failed to create the 1x1 local-DI stand-in textures");
        }
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
    // THE 1x1 ADAPTER. Every pre-paging host call site lands here and the
    // grid path below reproduces the old behaviour exactly: from_single keeps
    // the map's origin/size/range, TerrainRenderer keeps its init-time world
    // rect for a single page, and the grass binding below resolves to that
    // page's own texture and the map's own five scalars.
    set_heightmap_grid(HeightmapGrid::from_single(heightmap));
}

void SceneRenderer::set_heightmap_grid(const HeightmapGrid& grid) {
    terrain_->set_heightmap_grid(grid);

    if (grass_renderer_ && terrain_->heightmap_texture()) {
        // Whatever texture the terrain decided grass should sample — the
        // single page's for a 1x1 grid, the coarse world proxy for a paged
        // one — described by the params that decode it.
        const render::TerrainRenderer::HeightmapSampleParams src = terrain_->heightmap_sample_params();
        render::HeightmapParams hm_params;
        hm_params.world_origin_x = src.world_origin_x;
        hm_params.world_origin_z = src.world_origin_z;
        hm_params.world_size = src.world_size;
        hm_params.min_height = src.min_height;
        hm_params.max_height = src.max_height;
        grass_renderer_->set_heightmap(terrain_->heightmap_texture(), hm_params);
    }

    BRACKEN_LOG_DEBUG(LOG_TAG, "Heightmap set for terrain rendering ({} of {} page(s) resident)",
                      terrain_->resident_pages(), terrain_->heightmap_grid().page_count());
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
// Host render passes (scene/render_hook.hpp)
// ============================================================================

void SceneRenderer::set_post_ui_callback(PostUICallback cb) {
    BRACKEN_ASSERT_MAIN_THREAD();
    // ONE assignable slot, and Application::init_dev_ui takes it. A game that
    // set it used to turn F10 into a no-op with no diagnostic anywhere — the
    // failure this whole extension surface exists to close. The slot is kept for
    // the overlay's sake, so the least it can do is say when it is taken over.
    if (post_ui_callback_ && cb) {
        BRACKEN_LOG_WARN(LOG_TAG,
                         "SceneRenderer::set_post_ui_callback replaced an already-installed callback — if that was "
                         "the developer overlay (Application::init_dev_ui installs it), F10 will now draw nothing. "
                         "Hosts should use render_hooks().add(RenderHookSlot::AfterUi, ...), which composes.");
    }
    post_ui_callback_ = std::move(cb);
}

void SceneRenderer::count_skipped_render_hooks(std::initializer_list<RenderHookSlot> slots) {
    // One bool test is what an engine with no host passes pays here — the same
    // early-out every dispatch site uses.
    if (render_hooks_.empty()) {
        return;
    }
    uint32_t skipped = 0;
    for (RenderHookSlot slot : slots) {
        skipped += static_cast<uint32_t>(render_hooks_.entries(slot).size());
    }
    frame_diagnostics_.render_hooks_skipped += skipped;
}

void SceneRenderer::run_render_hooks(RenderHookSlot slot, SDL_GPUCommandBuffer* cmd, const CameraState& draw_camera,
                                     float dt, SDL_GPUTexture* swapchain, bool scene_targets_valid) {
    // Hooks record into the frame's command buffer alongside every engine pass.
    BRACKEN_ASSERT_MAIN_THREAD();
    const std::span<RenderHookRegistry::Entry* const> hooks = render_hooks_.entries(slot);
    if (hooks.empty()) {
        return;
    }
    if (cmd == nullptr) {
        // Nothing to record into. Counted, not skipped in silence — see
        // FrameDiagnostics::render_hooks_skipped.
        frame_diagnostics_.render_hooks_skipped += static_cast<uint32_t>(hooks.size());
        return;
    }

    RenderHookContext ctx;
    ctx.cmd = cmd;
    ctx.pipelines = pipeline_registry_.get();
    ctx.slot = slot;
    // `scene_targets_valid` is false on the fallback direct-to-swapchain path
    // and on a frame with no 3D content: the offscreen textures may still
    // EXIST there, holding the last frame that did render into them, and
    // handing a host a stale image is worse than handing it null.
    ctx.scene_color =
        (scene_targets_valid && post_ && post_->offscreen_color()) ? post_->offscreen_color()->handle() : nullptr;
    ctx.scene_depth =
        (scene_targets_valid && post_ && post_->offscreen_depth()) ? post_->offscreen_depth()->handle() : nullptr;
    ctx.velocity = (frame_prepass_active_ && velocity_texture_) ? velocity_texture_->handle() : nullptr;
    ctx.normal_roughness = normal_rough_texture_ ? normal_rough_texture_->handle() : nullptr;
    ctx.ao = frame_ao_texture_;
    ctx.swapchain = swapchain;
    ctx.scene_color_format =
        pipeline_registry_ ? pipeline_registry_->get_scene_color_format() : SDL_GPU_TEXTUREFORMAT_INVALID;
    ctx.depth_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    ctx.swapchain_format =
        pipeline_registry_ ? pipeline_registry_->get_swapchain_format() : SDL_GPU_TEXTUREFORMAT_INVALID;
    ctx.width = context_ ? context_->width() : 0;
    ctx.height = context_ ? context_->height() : 0;
    ctx.camera = draw_camera;
    ctx.view_projection_unjittered = frame_cam_.view_proj_unjittered;
    // NOT prev_view_proj_unjittered_ / frame_index_ directly: the AfterUi slot
    // runs AFTER the end-of-frame advance that overwrites both, so reading the
    // members there would hand a hook this frame's VP as the "previous" one
    // (zero reprojection) and a frame index one past the frame it is drawing
    // into. hook_prev_view_proj_ / hook_frame_index_ are captured before that
    // advance, so every slot sees the same pair.
    ctx.prev_view_projection_unjittered = hook_prev_view_proj_;
    ctx.depth_prepass_active = frame_prepass_active_;
    ctx.scene_time = scene_time_;
    ctx.dt = dt;
    ctx.frame_index = hook_frame_index_;

    // While this scope is alive the registry refuses add/remove/clear: a hook
    // that registered another hook would reallocate the span being walked.
    RenderHookRegistry::DispatchScope guard(render_hooks_);
    for (RenderHookRegistry::Entry* entry : hooks) {
        // entry->stable_name, NOT entry->name.c_str(): the timer pool keeps the
        // pointer past this frame and a host may remove the hook at the frame
        // fence (RenderHookRegistry::Entry::stable_name).
        BRACKEN_PROFILE_ZONE_COLOR("RenderHook", 0x2C82C9);
        BRACKEN_GPU_SCOPE(cmd, entry->stable_name);
        BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, entry->stable_name);
        entry->fn(ctx);
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
    frame_diagnostics_.shadow_capsules_over_budget += build_diag.shadow_capsules_over_budget;
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
    // Adopt the previous frame's occlusion readback BEFORE the plan is made:
    // whether a usable field exists is one of the plan's inputs, and the plan
    // has to stay a pure function of facts that are already settled.
    if (gfx.occlusion_culling) {
        resolve_occlusion_readback();
    } else if (hzb_far_ || occlusion_transfer_ || occlusion_fence_) {
        // Released the moment the setting goes off, mirroring hzb_/probe_volume_:
        // a screen-sized mip chain is not something to keep resident for a
        // feature nobody asked for.
        release_occlusion_resources();
        hzb_far_init_failed_ = false;
    }

    RendererCaps caps;
    caps.post_ready = post_->is_ready();
    caps.occlusion_depth_ready = occlusion_depth_ready_;
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
    frame_occlusion_active_ = plan.occlusion_active;

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

    // ---- R5b: the analytic sun gradient (docs/rt_lighting_plan.md §5.6) ----
    //
    // Measured HERE, once, on the frame's final environment, and handed to
    // every temporal accumulator later in this function. Two dot products and
    // an acos on the main thread — free next to a single ray — and exactly zero
    // for a sun that did not move, so nothing downstream changes on a frame
    // where nothing changed.
    //
    // The first frame reports 0 rather than a full reset: there is no previous
    // sun to have moved from, and every history it would reset is invalid for
    // its own reasons already.
    {
        const glm::vec3 sun_radiance = env_.sun_color * gfx.sun_intensity;
        lighting_reset_ = sun_state_valid_
                              ? render::lighting_reset_floor(render::sun_change_signal(
                                    prev_sun_dir_, env_.sun_dir, prev_sun_radiance_, sun_radiance))
                              : 0.0f;
        prev_sun_dir_ = env_.sun_dir;
        prev_sun_radiance_ = sun_radiance;
        sun_state_valid_ = true;
    }

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
        // Nothing records this frame, so no slot can run. Counted, because a
        // host whose pass never ran must be able to tell that apart from a pass
        // that ran and drew nothing.
        count_skipped_render_hooks({RenderHookSlot::PreForward, RenderHookSlot::AfterOpaque,
                                    RenderHookSlot::BeforeTaa, RenderHookSlot::AfterComposite,
                                    RenderHookSlot::AfterUi});
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
    // Ray-tracing state is per-frame and must be cleared here, not merely set
    // when the passes run: these are read by every forward draw, and a stale
    // "shadows active" plus a stale mask handle would keep applying last
    // frame's traced shadows after ray tracing is switched off.
    frame_rt_shadows_active_ = false;
    frame_rt_shadow_texture_ = nullptr;
    frame_rt_local_shadows_active_ = false;
    frame_rt_local_shadow_texture_ = nullptr;
    frame_rt_local_di_active_ = false;
    frame_rt_local_di_diffuse_ = nullptr;
    frame_rt_local_di_specular_ = nullptr;
    frame_rt_local_di_direction_ = nullptr;
    frame_rt_reflection_max_roughness_ = 0.0f;
    frame_rt_reflection_fade_start_ = 0.0f;
    // Sky IBL drives every scene shader's iblEnabled flag (the IBL sampler
    // slots are bound either way — dummies when inactive).
    frame_ibl_active_ = plan.ibl_active;
    // Same reasoning as the ray-tracing state above: a stale `enabled` plus a
    // stale atlas handle would keep lighting the scene from a field that has
    // been destroyed. Cleared here, filled by refresh_gi_bindings() below.
    frame_gi_ = render::gi::GiBindings{};
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

    // Host-hook context values that the END-OF-FRAME advance below overwrites.
    // The AfterUi slot runs after that advance, so it has to read the captured
    // copies or it sees this frame's VP as the "previous" one and a frame index
    // one past the frame it is drawing into (scene/render_hook.hpp documents
    // both fields as this frame's).
    hook_prev_view_proj_ = prev_view_proj_unjittered_;
    hook_frame_index_ = frame_index_;

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
            prev_view_ = frame_cam_.view;
            hook_prev_view_proj_ = prev_view_proj_unjittered_;
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
            // Near plane recovered from the projection: it is the linearized
            // view depth of device depth 1, which under reversed Z is the near
            // plane (contracts/depth_linear.hpp).
            // Shadow range is a quality setting (clamped by draw distance);
            // the shader fades shadows out over the tail instead of popping
            // them off.
            const float cam_near =
                std::max(scene::linear_view_depth(1.0f, scene::depth_linear_terms(camera.projection)), 0.1f);
            const float shadow_far = std::min(gfx.get_draw_distance(), gfx.shadow_distance);
            shadow_map_->update(camera.view, camera.projection, light_dir_, cam_near, shadow_far);
            for (int c = 0; c < plan.active_cascades; ++c) {
                shadow_cascade_frustums_[c].extract_from_matrix(shadow_map_->cascades()[c].light_view_projection);
            }
        }

        // ===== CPU prep for all per-frame GPU uploads =====

        // Build instance batches (cull for main + per-cascade shadows) and
        // resolve the skinned commands against the packet-owned palettes.
        // The occlusion field's motion gates, measured against the camera that
        // actually rasterized the depth it holds. They are the difference
        // between "conservative" and "conservative for a still camera": the
        // depth is a frame old, so a big translation or rotation means it
        // describes a view this one no longer resembles, and the field is
        // simply not consulted that frame. Both inputs are scene state, so the
        // decision is reproducible run to run.
        if (frame_occlusion_active_) {
            const glm::vec3 forward =
                glm::normalize(-glm::vec3(camera.view[0][2], camera.view[1][2], camera.view[2][2]));
            occlusion_field_.camera_travel = glm::length(camera.position - occlusion_camera_position_);
            occlusion_field_.camera_rotation_cos = glm::dot(forward, occlusion_camera_forward_);
        }

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
        // R6/R8: AN EMISSIVE MESH IS A LIGHT THE CLUSTER GRID KNOWS NOTHING
        // ABOUT. The Full-DI pass still needs the froxel BOUNDS — which are
        // light-independent — and a real ClusterParams to index them with, so
        // gating the whole grid on "are there any binned analytic lights"
        // silently switched emissive DI off in exactly the scene that needs it
        // most: one lit only by glowing windows has zero binned lights and is
        // not dark. rt_local_shadow.comp.hlsl says so in its own comments and,
        // until this, could never reach the code that says it.
        //
        // Settings-level rather than record-level on purpose: the RT scene's
        // records are not refreshed until ensure_raytracing_scene() several
        // hundred lines below, so a record-count test here would read last
        // frame's answer.
        const bool emissive_di_wanted = gfx.rt_local_shadows == LocalRtShadowMode::Full &&
                                        rt_renderer_ != nullptr &&
                                        rt_local_tuning_.emissive_candidates > 0;
        const bool cluster_upload_needed =
            cluster_grid_ready_ && (any_dyn_lights || emissive_di_wanted);
        // ATLAS RETIREMENT. In Ratio mode the traced pass owns local shadowing
        // outright: the selector never runs, no tile is ever rendered, and
        // every shadow_slot stays -1 — which makes the atlas branch in
        // cluster_lighting.hlsli naturally dead rather than disabled, so the
        // two can never double-darken a pixel. The atlas CODE is untouched and
        // is still exactly what `Off` runs.
        //
        // Decided on the SETTING, not on whether the trace succeeded, and that
        // is deliberate: the alternative is deciding here and discovering three
        // hundred lines later that the acceleration structure was not ready, by
        // which time the tile renders have either happened or not. A frame that
        // asks for traced local shadows and cannot have them shows unshadowed
        // local lights for that frame — visible, recoverable, and honest —
        // where the alternative would be a frame that pays for both.
        const bool rt_local_shadows_wanted =
            gfx.rt_local_shadows != LocalRtShadowMode::Off && rt_renderer_ != nullptr;
        // Stays on the ANALYTIC condition, not on the widened
        // `cluster_upload_needed` above: the rasterized atlas has nothing to
        // render for an emissive mesh, so a scene lit only by glowing windows
        // must not switch it on.
        frame_local_shadows_active_ = cluster_grid_ready_ && any_dyn_lights && gfx.local_shadows &&
                                      !rt_local_shadows_wanted && local_shadows_ &&
                                      local_shadows_->is_ready();
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
            // Binning is skipped entirely on this path, so the grid's light
            // identity history would otherwise survive the gap and let the
            // next build remap against a frame that is no longer "previous".
            cluster_grid_.reset_light_history();
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
                if (rt_renderer_) {
                    // The frame's analytic capsule occluders, straight off the
                    // packet (docs/rt_lighting_plan.md §5.8). Unconditional on
                    // the RT toggles: it is one small copy, it is what the four
                    // any-hit passes bind, and gating it on which traced
                    // features happen to be on this frame is exactly how a
                    // buffer ends up holding another frame's pose.
                    rt_renderer_->upload_shadow_capsules(copy_pass, frame.shadow_capsules());
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

            // Normal+roughness G-buffer: allocated while EITHER ray-traced
            // feature is on and released the moment both go off, so a game that
            // never enables them never holds a screen-sized target. It shares
            // the prepass with velocity, hence the identical size.
            //
            // Shadows need it as much as reflections do, and used not to get it.
            // The shadow pass reconstructed its normal from four depth taps
            // instead; on foliage and any high-frequency silhouette that
            // reconstruction flips between neighbouring pixels and between
            // frames, and since the pass hard-cuts to "shadowed" at N.L <= 0,
            // the flip is full-amplitude. That was the speckle, and no denoiser
            // can undo it — the signal is wrong before it is noisy.
            //
            // The local-light pass needs it for a THIRD reason, and a stricter
            // one: the sun pass can fall back to a depth-tap normal because it
            // only asks a binary question, but the froxel-RIS pass weights
            // candidate LIGHTS by N.L and picks one. A normal that flickers
            // between neighbouring pixels there does not add noise to an
            // answer, it changes which question was asked — so that pass is
            // skipped outright without a G-buffer rather than degraded.
            //
            // AND THE SCREEN PROBES NEED IT TOO (R4b). Their anchors' NORMALS
            // are what every bilateral weight in the gather is measured
            // against — the temporal history test, the 3x3 spatial filter and
            // the forward 4-probe interpolation all ask "is this the same
            // surface?" and answer with a dot product. A depth-tap
            // reconstruction there would not add noise to an answer, it would
            // move the plane the answer is about. So GiMode::ProbesScreen
            // switches the prepass to the G-buffer set exactly as the traced
            // effects do.
            if (gfx.rt_reflections || gfx.rt_shadows ||
                gfx.rt_local_shadows != LocalRtShadowMode::Off ||
                gfx.gi_mode == GiMode::ProbesScreen) {
                const int gw = context_->width();
                const int gh = context_->height();
                if (!normal_rough_texture_ || normal_rough_texture_->width() != gw ||
                    normal_rough_texture_->height() != gh) {
                    normal_rough_texture_ = gpu::GPUTexture::create_2d(
                        context_->device(), gw, gh, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
                        SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);
                    if (!normal_rough_texture_) {
                        BRACKEN_LOG_WARN(LOG_TAG,
                                         "render_frame: could not create the reflection G-buffer; "
                                         "ray-traced reflections will not run");
                    }
                }
            } else if (normal_rough_texture_) {
                normal_rough_texture_.reset();
            }

            // Offscreen clear tracks the environment's horizon fog color so
            // unrendered sky pixels melt into the atmosphere (linear HDR).
            post_->set_clear_color(env_.fog_color_horizon);

            // ===== Depth prepass: velocity + scene depth for opaque draws =====
            // (MASK/BLEND/skybox/particles keep their forward depth behavior.)
            frame_prepass_active_ = render_depth_prepass(frame, draw_camera);

            // ===== Hierarchical-Z pyramid (R4a, docs/rt_lighting_plan.md
            // §5.5): a conservative closest-surface mip chain over the depth
            // the prepass just wrote. It sits HERE, immediately after the
            // prepass and before GTAO, because that is the earliest point its
            // only input exists and because every future consumer (R4b's screen
            // march, and occlusion culling if it ever lands) runs later in the
            // frame.
            //
            // GATED ON ITS CONSUMER. `GiMode::ProbesScreen` is the one mode that
            // reads the pyramid — R4b's screen-probe march traces it before it
            // touches the BVH — so it is the one mode that builds it. Building
            // it unconditionally would be a screen-sized allocation and eleven
            // dispatches no other configuration benefits from. Every other
            // configuration, including the default, is untouched: not a byte
            // allocated, one enum compare. =====
            const bool hzb_wanted = gfx.gi_mode == GiMode::ProbesScreen && frame_prepass_active_ &&
                                    post_->offscreen_depth() != nullptr && cmd != nullptr;
            if (hzb_wanted) {
                if (!hzb_ && !hzb_init_failed_) {
                    auto pyramid = std::make_unique<render::HzbPyramid>();
                    if (pyramid->init(context_->device(), context_->width(), context_->height())) {
                        hzb_ = std::move(pyramid);
                    } else {
                        // init() logged the reason. Latch it: retrying a
                        // format-support check and a mip-chain allocation every
                        // frame would be a per-frame stall and a log flood.
                        hzb_init_failed_ = true;
                    }
                }
                if (hzb_) {
                    hzb_->resize(context_->width(), context_->height());
                    BRACKEN_PROFILE_ZONE_COLOR("HZB", 0x2C81C4);
                    BRACKEN_GPU_SCOPE(cmd, "HZB");
                    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "HZB");
                    hzb_->build(cmd, *pipeline_registry_, *post_->offscreen_depth());
                }
            } else if (gfx.gi_mode != GiMode::ProbesScreen && (hzb_ || hzb_init_failed_)) {
                // Released the moment the mode leaves ProbesScreen, mirroring
                // probe_volume_. A frame that merely lost its prepass keeps the
                // pyramid allocated — that is a transient, not a mode change.
                hzb_.reset();
                hzb_init_failed_ = false;
            }

            // ===== The FARTHEST-surface pyramid, for next frame's draw
            // occlusion cull. Same input as the block above, opposite reduction
            // (render/hzb.hpp hzb_reduce_far), and its own texture — the two
            // answers cannot be recovered from one another.
            //
            // It sits here for the same reason: this is the earliest point its
            // only input exists. What it produces is consumed a WHOLE FRAME
            // LATER, in the Prepare stage, because Prepare runs before this
            // frame's prepass and SDL3 has no non-blocking map. That one-frame
            // lag is the source of every conservatism knob on OcclusionField.
            //
            // Off by default: one enum compare and not a byte allocated. =====
            if (gfx.occlusion_culling && frame_prepass_active_ && cmd != nullptr) {
                build_occlusion_pyramid(cmd, draw_camera);
            }

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
                // The pass publishes a WORLD-space bent normal, so it needs
                // view->world. Not cached like the projection: the view matrix
                // changes every frame anyway.
                ao_params.inv_view = glm::inverse(draw_camera.view);
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

            // ===== The acceleration structure, for EVERY consumer of it.
            //
            // GI IS A RAY-TRACED FEATURE. `gi_mode != Off` builds the BVH — with
            // per-triangle attributes, because probe rays SHADE what they hit —
            // even when rt_shadows and rt_reflections are both off, and it
            // therefore inherits every constraint the other two carry: models
            // must have kept their CPU geometry (see build_raytracing_scene),
            // the first frame after enabling it pays a full BLAS build, and
            // ~30 MB of BLAS plus the attribute buffer stay resident. That is
            // documented in docs/user_guide.md and it is not a detail a host
            // can discover from the setting's name.
            //
            // Owned here rather than asked of the host: a toggle flipped at
            // runtime would otherwise leave the structure unbuilt, every trace
            // would return immediately, and the setting would silently do
            // nothing. =====
            const bool gi_active = gfx.gi_mode != GiMode::Off;
            // ...but the RASTER GI mode is exempt from every sentence above.
            // GiMode::VoxelProbes fills the same probe field from a voxel
            // clipmap, so it needs no BVH, no attributes and none of the ~30 MB
            // — which is the entire reason it exists. Keeping it out of
            // `gi_wants_rt` is what makes "GI without ray tracing" true rather
            // than merely differently spelled.
            const bool gi_raster = gfx.gi_mode == GiMode::VoxelProbes;
            const bool gi_wants_rt = gi_active && !gi_raster;
            // Local-light shadows join the same gate: they trace the same BVH
            // and therefore need the same acceleration structure built. They do
            // NOT need triangle ATTRIBUTES — a shadow ray asks a yes/no
            // question and never shades what it hits — so want_attributes below
            // stays driven by reflections and GI alone.
            const bool rt_traces_active =
                gfx.rt_shadows || gfx.rt_reflections || gfx.rt_local_shadows != LocalRtShadowMode::Off;
            render::RtRenderer::FrameParams rt_frame_params;
            render::RtReflectionSettings rt_reflection_settings;
            bool rt_reflection_resolve_ready = false;
            if ((rt_traces_active || gi_wants_rt) && rt_renderer_ && cmd) {
                ensure_raytracing_scene(gfx.rt_reflections || gi_wants_rt);
                {
                    // Timed separately from the traces: this is the CPU-side
                    // TLAS work (range filter, content hash, and the rebuild
                    // when the hash moved), which is exactly the number
                    // rt_max_instance_range trades against.
                    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "RTInstances");
                    refresh_raytracing_instances(frame, cmd, draw_camera.position,
                                                 rt_range_units(gfx.rt_max_instance_range));
                }
                // R8: bin those records into THIS frame's froxel grid
                // (docs/rt_lighting_plan.md §5.7 follow-up 2). It has to be here
                // and not in the frame's consolidated copy pass: the records are
                // world-space and were only just brought up to date by the refit
                // above, and a table describing last frame's records is exactly
                // what the trace refuses to draw through.
                //
                // Timed on its own row rather than folded into RTInstances,
                // because "how much main-thread time does the localization cost"
                // is the question this package has to answer. The camera-relative
                // grid is why it reruns every frame where the records themselves
                // mostly do not.
                if (emissive_clusters_ready_ && cluster_upload_needed &&
                    gfx.rt_local_shadows == LocalRtShadowMode::Full &&
                    rt_local_tuning_.emissive_froxel_share > 0.0f &&
                    rt_local_tuning_.emissive_candidates > 0) {
                    BRACKEN_PROFILE_ZONE_COLOR("RTEmissiveBins", 0x27AE60);
                    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "RTEmissiveBins");
                    emissive_clusters_.build(rt_renderer_->scene().emissive_record_data(),
                                             cluster_grid_.params(), cluster_grid_.cluster_bounds(),
                                             rt_renderer_->scene().emissive_generation());
                    emissive_clusters_.upload(cmd);
                } else {
                    emissive_clusters_.invalidate();
                }
            } else {
                emissive_clusters_.invalidate();
            }

            // ===== Probe-field GI: scroll the clipmap to follow the camera,
            // then trace/blend/classify this frame's share of it.
            //
            // Everything the trace reads is already in place at this point in
            // the frame, which is why the block sits HERE and not earlier: the
            // CSM atlas was rendered, the cluster light buffer uploaded, the IBL
            // sky-capture chain refreshed, and the TLAS brought up to date
            // immediately above. Nothing SAMPLES the field yet (R1c), so this
            // must leave the image bit-identical while doing all the work.
            //
            // With gi_mode == Off — the default — not one byte is allocated and
            // this is a single enum compare. =====
            if (gi_active && cmd && context_) {
                if (!probe_volume_ && !gi_init_failed_) {
                    auto volume = std::make_unique<render::gi::ProbeVolume>();
                    if (volume->init(context_->device())) {
                        probe_volume_ = std::move(volume);
                    } else {
                        // init() already logged the reason. Latch the failure:
                        // retrying a 30 MB allocation every frame would turn
                        // one bad allocation into a per-frame stall and a log
                        // flood. Toggling gi_mode off and on again clears it.
                        gi_init_failed_ = true;
                        BRACKEN_LOG_WARN(LOG_TAG, "GI probe field unavailable; gi_mode has no effect");
                    }
                }
                // ===== The RASTER ray source (GiMode::VoxelProbes). =====
                //
                // Allocated on the same lazy/latched policy as the probe field
                // and torn down the moment the mode leaves VoxelProbes: ~35 MB
                // of volumes is not worth keeping resident for a mode the host
                // is not using.
                if (gi_raster && !voxel_gi_ && !voxel_gi_init_failed_) {
                    auto voxel = std::make_unique<render::gi::VoxelGi>();
                    if (voxel->init(context_->device(), *pipeline_registry_)) {
                        voxel_gi_ = std::move(voxel);
                        voxel_gi_model_count_ = 0; // force the first gather
                    } else {
                        voxel_gi_init_failed_ = true;
                        BRACKEN_LOG_WARN(LOG_TAG,
                                         "voxel GI unavailable; gi_mode=VoxelProbes falls back to "
                                         "an untraced field");
                    }
                } else if (!gi_raster && (voxel_gi_ || voxel_gi_init_failed_)) {
                    voxel_gi_.reset();
                    voxel_gi_init_failed_ = false;
                    voxel_gi_model_count_ = 0;
                }

                if (voxel_gi_) {
                    BRACKEN_PROFILE_ZONE_COLOR("VoxelGI", 0x16A085);
                    BRACKEN_GPU_SCOPE(cmd, "VoxelGI");
                    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "VoxelGI");

                    // The geometry gather is expensive and the model set almost
                    // never changes, so it is keyed off the model count rather
                    // than run per frame. Placements are re-hashed every frame
                    // and early-out when unchanged.
                    if (model_manager_ && model_manager_->model_count() != voxel_gi_model_count_) {
                        voxel_gi_->build_models(*model_manager_);
                        voxel_gi_model_count_ = model_manager_->model_count();
                    }
                    // Gathered here rather than reused from the ray-tracing
                    // block: that block does not run in this mode, so sharing
                    // its scratch would leave the clipmap empty in exactly the
                    // configuration VoxelProbes exists for. Unfiltered by range
                    // — the clipmap does its own culling per cascade window, and
                    // the coarsest one wants the whole map.
                    voxel_instances_scratch_.clear();
                    const auto voxel_models = frame.model_commands();
                    voxel_instances_scratch_.reserve(voxel_models.size());
                    for (const ModelDrawCommand& draw : voxel_models) {
                        if (draw.model_handle != INVALID_MODEL_HANDLE) {
                            voxel_instances_scratch_.emplace_back(draw.model_handle, draw.transform);
                        }
                    }
                    // Skinned draws contribute their BIND POSE, which is the
                    // right call at a 25-unit voxel: a character's own bounce is
                    // negligible and re-voxelizing it per frame is not.
                    for (const SkinnedDrawCommand& draw : frame.skinned_commands()) {
                        if (draw.model_handle != INVALID_MODEL_HANDLE) {
                            voxel_instances_scratch_.emplace_back(draw.model_handle, draw.transform);
                        }
                    }
                    voxel_gi_->set_instances(voxel_instances_scratch_);
                    voxel_gi_->upload(context_->device());

                    render::gi::VoxelGiFrameParams vp;
                    vp.camera_position = draw_camera.position;
                    vp.camera_forward = -glm::vec3(draw_camera.view[0][2], draw_camera.view[1][2],
                                                   draw_camera.view[2][2]);
                    vp.shadow_atlas = shadow_map_->atlas_texture();
                    vp.shadow_cmp_sampler = shadow_map_->shadow_compare_sampler();
                    vp.shadow_raw_sampler = shadow_map_->shadow_sampler();
                    vp.shadow_block = &frame_shadow_uniforms_;
                    vp.shadow_block_size = sizeof(frame_shadow_uniforms_);
                    vp.sun_direction = env_.sun_dir;
                    vp.sun_color = env_.sun_color;
                    vp.sun_intensity = gfx.sun_intensity;
                    vp.sky_color = env_.ambient_sky;
                    // A small floor so a voxel the sun cannot reach is not
                    // perfectly black; the probe field applies the real sky
                    // term, and this only stops the volume from zeroing it.
                    vp.sky_floor = 0.15f;
                    // Lanterns light the bounce. Without these the clipmap has
                    // only the sun in it, so after dusk it goes black and takes
                    // the GI term with it.
                    vp.light_data = cluster_grid_.light_data_buffer();
                    vp.light_count = cluster_grid_.total_lights();
                    vp.light_capacity = render::lighting::MAX_LIGHTS;
                    // The sky CAPTURE, not the irradiance convolution — the same
                    // texture the traced probe path samples on a ray miss, so
                    // the two GI modes agree about what the sky is worth.
                    vp.sky_cube = (ibl_ && ibl_->is_ready()) ? ibl_->capture_map() : nullptr;
                    vp.sky_cube_fallback =
                        dummy_black_cube_ ? dummy_black_cube_->handle() : nullptr;
                    vp.sky_sampler = (ibl_ && ibl_->cube_sampler() != nullptr)
                                         ? ibl_->cube_sampler()
                                         : default_sampler_;
                    // The same signal the denoisers use for a sun jump: injected
                    // radiance is stale the moment the sun moves.
                    vp.lighting_changed = lighting_reset_ > 0.01f;
                    voxel_gi_->update(cmd, vp);
                }

                if (probe_volume_) {
                    BRACKEN_PROFILE_ZONE_COLOR("GIProbes", 0x27AE60);
                    BRACKEN_GPU_SCOPE(cmd, "GIProbes");
                    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "GIProbes");
                    probe_volume_->update(cmd, draw_camera.position);

                    render::gi::ProbeUpdateParams gi_params;
                    gi_params.pipelines = pipeline_registry_.get();
                    gi_params.scene = rt_renderer_ ? &rt_renderer_->scene() : nullptr;
                    gi_params.quality = gfx.gi_quality;
                    gi_params.camera_position = draw_camera.position;
                    // Third ROW of the view matrix is the camera's -forward in
                    // world space; calcShadow needs a view depth for the hit.
                    gi_params.camera_forward = -glm::vec3(draw_camera.view[0][2], draw_camera.view[1][2],
                                                          draw_camera.view[2][2]);
                    gi_params.environment = env_;
                    gi_params.sun_intensity = gfx.sun_intensity;
                    // R5b: the same number the visibility denoisers get, so a
                    // sun jump collapses the field's blend window on the frame
                    // it happened rather than several probe rotas later.
                    gi_params.lighting_reset = lighting_reset_;
                    gi_params.shadow = frame_shadow_uniforms_;
                    gi_params.shadow_atlas = shadow_map_->atlas_texture();
                    gi_params.shadow_compare_sampler = shadow_map_->shadow_compare_sampler();
                    gi_params.shadow_raw_sampler = shadow_map_->shadow_sampler();
                    // The sky-CAPTURE cube (raw radiance), not the irradiance
                    // convolution: a probe-ray miss is a look straight at the
                    // sky in one direction.
                    gi_params.sky_cube = (ibl_ && ibl_->is_ready()) ? ibl_->capture_map() : nullptr;
                    gi_params.sky_cube_fallback =
                        dummy_black_cube_ ? dummy_black_cube_->handle() : nullptr;
                    gi_params.sky_sampler =
                        (ibl_ && ibl_->cube_sampler() != nullptr) ? ibl_->cube_sampler() : default_sampler_;
                    gi_params.light_data = cluster_grid_.light_data_buffer();
                    gi_params.light_count = cluster_grid_.total_lights();
                    gi_params.light_capacity = render::lighting::MAX_LIGHTS;
                    gi_params.voxel = voxel_gi_.get();
                    // The sky cube, the sky sampler and the light buffer are
                    // the RAY source's inputs — it shades what it hits. The
                    // raster source shades nothing (the radiance is already in
                    // the volume), so requiring them would make VoxelProbes fail
                    // for the wrong reasons on a frame with no IBL yet.
                    const bool ray_inputs_ready =
                        gi_params.sky_cube_fallback != nullptr && gi_params.sky_sampler != nullptr &&
                        gi_params.light_data != nullptr;
                    if (gi_params.shadow_atlas != nullptr &&
                        (ray_inputs_ready || (voxel_gi_ && voxel_gi_->ready()))) {
                        probe_volume_->trace(cmd, gi_params);
                    }
                }
            } else if (probe_volume_ || gi_init_failed_) {
                probe_volume_.reset();
                gi_init_failed_ = false;
            }

            // ===== Screen-probe gather (R4b, docs/rt_lighting_plan.md §5.5):
            // one 8x8 octahedral probe per 16x16 screen tile, traced against
            // the Hi-Z pyramid first and the BVH second, filtered in probe
            // space, published as an L1 SH atlas the scene shaders sample.
            //
            // AFTER the probe field's own update, deliberately and necessarily:
            // the handed-off rays shade their hits with the field (R3's
            // rt_hit_shade), and the rays that escape the 600-unit clamp fall
            // back to it. A gather that ran first would be lit by the previous
            // frame's field while claiming to be this frame's.
            //
            // BEFORE the forward pass, equally necessarily: the forward shaders
            // read the SH atlas this block writes.
            //
            // The TAA history it samples is the PREVIOUS frame's resolve —
            // `TAA::render` does not run until after the offscreen pass, which
            // is what makes `previous_result()` mean what its name says here
            // and not two hundred lines later. =====
            const bool screen_probes_wanted =
                gfx.gi_mode == GiMode::ProbesScreen && cmd != nullptr && hzb_ && hzb_->is_ready() &&
                frame_prepass_active_ && normal_rough_texture_ && velocity_texture_ &&
                probe_volume_ && probe_volume_->is_ready();
            if (screen_probes_wanted) {
                if (!screen_probes_ && !screen_probes_init_failed_) {
                    auto gather = std::make_unique<render::gi::ScreenProbeGather>();
                    if (gather->init(context_->device(), context_->width(), context_->height())) {
                        screen_probes_ = std::move(gather);
                    } else {
                        // init() logged the reason. Latch it: retrying a
                        // multi-megabyte allocation every frame would be a
                        // per-frame stall and a log flood.
                        screen_probes_init_failed_ = true;
                        BRACKEN_LOG_WARN(LOG_TAG,
                                         "screen-probe gather unavailable; gi_mode=ProbesScreen "
                                         "behaves as Probes");
                    }
                }
                if (screen_probes_) {
                    screen_probes_->resize(context_->width(), context_->height());
                    BRACKEN_PROFILE_ZONE_COLOR("GIScreenProbes", 0x1E8449);
                    BRACKEN_GPU_SCOPE(cmd, "GIScreenProbes");
                    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "GIScreenProbes");

                    render::gi::ScreenProbeParams sp;
                    sp.pipelines = pipeline_registry_.get();
                    sp.scene = rt_renderer_ ? &rt_renderer_->scene() : nullptr;
                    sp.hzb = hzb_->texture();
                    sp.hzb_mip_count = hzb_->mip_count();
                    sp.gbuffer_normal = normal_rough_texture_->handle();
                    // Only when TAA actually resolved a previous frame. Without
                    // it every ray goes to the BVH — slower, and correct,
                    // rather than fast and invented.
                    gpu::GPUTexture* taa_prev =
                        (taa_active && taa_) ? taa_->previous_result() : nullptr;
                    sp.scene_history = taa_prev ? taa_prev->handle() : nullptr;
                    sp.velocity = velocity_texture_->handle();
                    sp.screen_width = context_->width();
                    sp.screen_height = context_->height();
                    // The DRAW (jittered) pair — the depth buffer's matrices.
                    sp.view_projection = draw_camera.view_projection;
                    sp.inv_view_projection = glm::inverse(draw_camera.view_projection);
                    sp.prev_view_projection_unjittered = prev_view_proj_unjittered_;
                    sp.camera_position = draw_camera.position;
                    sp.camera_forward = -glm::vec3(draw_camera.view[0][2], draw_camera.view[1][2],
                                                   draw_camera.view[2][2]);
                    const auto sp_depth = scene::depth_linear_terms(draw_camera.projection);
                    sp.depth_a = sp_depth.a;
                    sp.depth_b = sp_depth.b;
                    sp.environment = env_;
                    // R5b: the probe-space firefly ceiling's absolute floor.
                    sp.exposure_state =
                        (gfx.auto_exposure && auto_exposure_ && auto_exposure_->is_ready())
                            ? auto_exposure_->state_buffer()
                            : nullptr;
                    sp.manual_exposure_ev = std::log2(std::max(gfx.exposure, 1e-6f));
                    sp.sun_intensity = gfx.sun_intensity;
                    sp.shadow = frame_shadow_uniforms_;
                    sp.shadow_atlas = shadow_map_->atlas_texture();
                    sp.shadow_compare_sampler = shadow_map_->shadow_compare_sampler();
                    sp.shadow_raw_sampler = shadow_map_->shadow_sampler();
                    sp.sky_cube = (ibl_ && ibl_->is_ready()) ? ibl_->capture_map() : nullptr;
                    sp.sky_cube_fallback = dummy_black_cube_ ? dummy_black_cube_->handle() : nullptr;
                    sp.sky_sampler = (ibl_ && ibl_->cube_sampler() != nullptr) ? ibl_->cube_sampler()
                                                                              : default_sampler_;
                    sp.light_data = cluster_grid_.light_data_buffer();
                    sp.light_count = cluster_grid_.total_lights();
                    sp.light_capacity = render::lighting::MAX_LIGHTS;
                    sp.gi_irradiance = probe_volume_->irradiance()->handle();
                    sp.gi_irradiance_sampler = probe_volume_->linear_sampler();
                    sp.gi_probe_state = probe_volume_->probe_state()->handle();
                    const std::array<glm::ivec3, render::gi::CASCADE_COUNT> bases =
                        probe_volume_->cascade_bases();
                    for (std::size_t c = 0; c < bases.size(); ++c) {
                        sp.gi_cascade_base[c] = glm::ivec4(bases[c], 0);
                    }
                    sp.gi_enabled = probe_volume_->cascade(0).placed() ? 1.0f : 0.0f;
                    screen_probes_->render(cmd, sp);
                }
            } else {
                if (screen_probes_ && gfx.gi_mode == GiMode::ProbesScreen) {
                    // A transient (no prepass this frame, no G-buffer yet):
                    // keep the allocation, drop the temporal history so the
                    // next good frame does not EMA against a stale camera.
                    screen_probes_->invalidate_history();
                } else if (screen_probes_ || screen_probes_init_failed_) {
                    // A real mode change. Released the moment the mode leaves
                    // ProbesScreen, mirroring probe_volume_ and hzb_.
                    screen_probes_.reset();
                    screen_probes_init_failed_ = false;
                }
            }

            // ONE snapshot of the field for the whole frame (R1c). Taken after
            // the update passes rather than before, so the cascade bases a
            // forward draw uses are the ones the atlases were just written
            // against; taken unconditionally, because the slots are declared by
            // every scene shader and SDL3 GPU leaves an unbound fragment
            // sampler descriptor undefined.
            refresh_gi_bindings();

            // Ray-traced sun shadows. The mask is its own R8 texture on its
            // own fragment sampler slot; it does NOT touch frame_ao_texture_.
            // It used to: the pass copied GTAO through into R of an RG8 mask
            // so the lighting stage could read both from one slot. That copy
            // indexed the AO texture with a coordinate scaled from the DEPTH
            // texture's dimensions, so with AO off (a 1x1 white dummy) every
            // thread read out of bounds, got 0, and the frame rendered fully
            // ambient-occluded. Un-multiplexing also stops full-res GTAO being
            // point-downsampled to the mask's resolution on every traced frame.
            if (rt_traces_active && rt_renderer_ && post_->offscreen_depth() && cmd) {
                BRACKEN_PROFILE_ZONE_COLOR("RT shadows", 0x8E44AD);
                BRACKEN_GPU_SCOPE(cmd, "RT shadows");
                rt_frame_params.inv_view_proj = glm::inverse(draw_camera.view_projection);
                // UNJITTERED, both — the pair the velocity buffer was built
                // from, which is what makes the reflection denoiser's virtual
                // motion vector directly comparable with it.
                rt_frame_params.view_proj_unjittered = frame_cam_.view_proj_unjittered;
                rt_frame_params.prev_view_proj_unjittered = prev_view_proj_unjittered_;
                // The local gradient's replay compares true view depths, so it
                // needs the previous VIEW rather than the previous VP pair.
                rt_frame_params.prev_view = prev_view_;
                rt_frame_params.camera_position = draw_camera.position;
                // Third ROW of the view matrix is the camera's -forward in
                // world space; calcShadow needs a view depth for the hit.
                rt_frame_params.camera_forward = -glm::vec3(
                    draw_camera.view[0][2], draw_camera.view[1][2], draw_camera.view[2][2]);
                rt_frame_params.environment = env_;
                rt_frame_params.sun_intensity = gfx.sun_intensity;
                rt_frame_params.lighting_reset = lighting_reset_;
                // R5b's exposure-aware clamps. The state buffer exists whenever
                // AutoExposure initialized, but it only holds a LIVE adapted EV
                // when the setting is on — so the flag the pass reads is the
                // pointer being non-null, and the pointer is null whenever the
                // setting is off. The manual bias rides alongside in both cases
                // because the composite multiplies it in both cases.
                rt_frame_params.exposure_state =
                    (gfx.auto_exposure && auto_exposure_ && auto_exposure_->is_ready())
                        ? auto_exposure_->state_buffer()
                        : nullptr;
                rt_frame_params.manual_exposure_ev = std::log2(std::max(gfx.exposure, 1e-6f));
                rt_frame_params.screen_width = context_->width();
                rt_frame_params.screen_height = context_->height();
                const auto rt_depth = scene::depth_linear_terms(draw_camera.projection);
                rt_frame_params.depth_a = rt_depth.a;
                rt_frame_params.depth_b = rt_depth.b;

                // What the reflection trace shades its hits with (R3). Every
                // one of these is already built and bound elsewhere this frame
                // — the CSM atlas the forward pass filters, the sky capture the
                // skybox draws, the light buffer the cluster loop reads, the
                // probe field the ambient term samples — which is the whole
                // argument for sharing them rather than approximating them.
                render::RtHitShadeBindings& hit_shade = rt_frame_params.hit_shade;
                hit_shade.shadow = frame_shadow_uniforms_;
                hit_shade.shadow_atlas = shadow_map_->atlas_texture();
                hit_shade.shadow_compare_sampler = shadow_map_->shadow_compare_sampler();
                hit_shade.shadow_raw_sampler = shadow_map_->shadow_sampler();
                // The sky-CAPTURE cube (raw radiance), not the irradiance
                // convolution: a reflection-ray miss is a look straight at the
                // sky in one direction.
                hit_shade.sky_cube = (ibl_ && ibl_->is_ready()) ? ibl_->capture_map() : nullptr;
                hit_shade.sky_cube_fallback =
                    dummy_black_cube_ ? dummy_black_cube_->handle() : nullptr;
                hit_shade.sky_sampler =
                    (ibl_ && ibl_->cube_sampler() != nullptr) ? ibl_->cube_sampler() : default_sampler_;
                hit_shade.light_data = cluster_grid_.light_data_buffer();
                hit_shade.light_count = cluster_grid_.total_lights();
                hit_shade.light_capacity = render::lighting::MAX_LIGHTS;
                // The probe field, when there is one. Not frame_gi_'s three
                // slots: the hit shader wants the probe STATE plane (has this
                // slot ever been filled?) where the forward sampler wants the
                // OFFSET plane, and it has no use for the visibility moments —
                // a secondary bounce does not run the Chebyshev test.
                if (probe_volume_ && probe_volume_->is_ready() && frame_gi_.enabled > 0.0f) {
                    hit_shade.gi_irradiance = probe_volume_->irradiance()->handle();
                    hit_shade.gi_irradiance_sampler = probe_volume_->linear_sampler();
                    hit_shade.gi_probe_state = probe_volume_->probe_state()->handle();
                    hit_shade.gi_cascade_base = frame_gi_.cascade_base;
                }

                render::RtShadowSettings rs;
                rs.enabled = gfx.rt_shadows;
                rs.resolution = gfx.rt_shadow_resolution;
                rs.sun_angular_radius_degrees = gfx.rt_sun_softness_deg;
                rt_reflection_settings.enabled = gfx.rt_reflections;
                rt_reflection_settings.resolution = gfx.rt_reflection_resolution;
                rt_reflection_settings.max_roughness = gfx.rt_reflection_max_roughness;
                // fade_start is the START of the same handover; keep it below
                // the ceiling or rtReflectionWeight's smoothstep is undefined.
                rt_reflection_settings.fade_start =
                    std::min(rt_reflection_settings.fade_start, gfx.rt_reflection_max_roughness * 0.5f);

                render::RtLocalShadowSettings ls;
                ls.enabled = gfx.rt_local_shadows != LocalRtShadowMode::Off;
                // Full needs exactly what Ratio needs — the acceleration
                // structure and the prepass G-buffer — so there is no second
                // ensure path here, only a second flag.
                ls.full_di = gfx.rt_local_shadows == LocalRtShadowMode::Full;
                ls.resolution = gfx.rt_local_shadow_resolution;
                // R7's developer knobs (SceneRenderer::RtLocalTuning), which
                // default to the shipped configuration — see the struct for why
                // they are not GraphicsSettings fields.
                ls.spatial_neighbors = rt_local_tuning_.spatial_neighbors;
                ls.spatial_radius = rt_local_tuning_.spatial_radius;
                ls.di_spatial_iterations = rt_local_tuning_.di_spatial_iterations;
                ls.di_noise_probe = rt_local_tuning_.di_noise_probe;
                // Emissive mesh lights are DIRECT lighting for the reservoir
                // (R6) and, at the same time, first-hit radiance in the probe
                // field: a probe ray that lands on a glowing window deposits
                // that window's emission into the atlas (rt_hit_shade.hlsli),
                // and the forward pass reads the field as the WHOLE diffuse
                // ambient. rt_hit_shade's "this is a SECONDARY hit, so it cannot
                // double-count" argument holds for a REFLECTION ray and not for
                // a probe ray — a probe is an interpolation point on the same
                // emitter -> receiver path the reservoir evaluates, not a bounce
                // vertex. So with the field live the two estimators are additive
                // over one transport and emissive geometry is counted twice.
                //
                // Partition it the cheap exact way: while the field is live the
                // FIELD owns emissive transport and this stream drops it. That
                // is the mechanism the Ratio tier already uses — emissiveCount 0
                // collapses cleanly in rt_local_shadow.comp.hlsl — and it costs
                // the per-pixel SHADOWED emissive direct term: blurry
                // probe-interpolated emissive instead of a crisp contact shadow
                // from a lit window. Scaling the probe DEPOSIT to zero instead
                // would remove emissive bounce, the emissive fog in-scatter and
                // R8's far-field fill along with the double count, which is
                // worse; §5.7's "give probe hits an emissive-record irradiance
                // term" is the partition that costs neither, and it is a phase.
                //
                // Keyed on the field being live rather than on gi_mode, so a
                // frame whose probe volume failed to allocate keeps its emissive
                // direct lighting. R8's froxel binning above still runs in that
                // case and its table simply goes unsampled — wasted work, not a
                // wrong picture.
                const bool gi_owns_emissive = gi_active && frame_gi_.enabled > 0.0f;
                ls.emissive_candidate_count =
                    gi_owns_emissive ? 0 : rt_local_tuning_.emissive_candidates;
                ls.emissive_froxel_share = std::clamp(rt_local_tuning_.emissive_froxel_share, 0.0f, 1.0f);

                // The frame's binned light buffers. Handed over as one struct
                // that knows when it is incomplete, because "trace against a
                // stale froxel list" is a wrong picture rather than a missing
                // one: the reservoir would sample lights the frame does not
                // have. frame_cluster_params_ (not cluster_grid_.params()) for
                // the same reason bind_cluster_lights uses it — a frame with no
                // dynamic lights publishes gridDim.w == 0 there.
                render::RtClusterBindings cluster_bindings;
                if (ls.enabled && cluster_upload_needed) {
                    cluster_bindings.light_data = cluster_grid_.light_data_buffer();
                    cluster_bindings.cluster_offsets = cluster_grid_.cluster_offsets_buffer();
                    cluster_bindings.light_indices = cluster_grid_.light_indices_buffer();
                    cluster_bindings.light_remap = cluster_grid_.light_remap_buffer();
                    cluster_bindings.params = &frame_cluster_params_;
                    cluster_bindings.params_size = sizeof(frame_cluster_params_);
                    cluster_bindings.total_lights = frame_cluster_params_.gridDim.w;
                    // R8's emissive froxel table, only when THIS frame binned it.
                    // `live()` false leaves the pointer null, which is what makes
                    // the trace fall back to R6's global draw rather than sample
                    // through a list that describes a different record set.
                    if (emissive_clusters_.live()) {
                        const auto& ec = emissive_clusters_.stats();
                        cluster_bindings.emissive_froxels = emissive_clusters_.buffer();
                        cluster_bindings.emissive_froxel_entries = ec.entries;
                        cluster_bindings.emissive_froxels_filled = ec.froxels_filled;
                        cluster_bindings.emissive_froxel_dropped = ec.entries_dropped;
                    }
                }

                // Reflections need the prepass G-buffer; without it (allocation
                // failed, or the prepass did not run this frame) the pass has
                // no way to tell a mirror from a wall, so it is skipped rather
                // than run against a texture that means something else.
                SDL_GPUTexture* normal_rough =
                    (frame_prepass_active_ && normal_rough_texture_) ? normal_rough_texture_->handle() : nullptr;
                render::RtReflectionSettings opaque_reflection_settings = rt_reflection_settings;
                opaque_reflection_settings.enabled =
                    opaque_reflection_settings.enabled && normal_rough != nullptr;
                rt_renderer_->render(cmd, *post_->offscreen_depth(), normal_rough,
                                     velocity_texture_ ? velocity_texture_->handle() : nullptr,
                                     rt_frame_params, rs, opaque_reflection_settings, ls,
                                     cluster_bindings);
                if (rt_renderer_->stats().shadows_traced && rt_renderer_->shadow_mask() != nullptr) {
                    frame_rt_shadow_texture_ = rt_renderer_->shadow_mask()->handle();
                    frame_rt_shadows_active_ = true;
                }
                if (rt_renderer_->stats().local_shadows_traced &&
                    rt_renderer_->local_shadow_mask() != nullptr) {
                    frame_rt_local_shadow_texture_ = rt_renderer_->local_shadow_mask()->handle();
                    frame_rt_local_shadows_active_ = true;
                }
                // The Full-DI tier, only if BOTH targets came out of the
                // denoiser. Half a pair would mean the forward shaders
                // remodulate a diffuse estimate against a stale specular one,
                // which is a wrong picture rather than a missing one — so this
                // falls back to Ratio, whose mask is live either way.
                if (rt_renderer_->stats().local_di_traced &&
                    rt_renderer_->local_di_diffuse() != nullptr &&
                    rt_renderer_->local_di_specular() != nullptr &&
                    rt_renderer_->local_di_direction() != nullptr) {
                    frame_rt_local_di_diffuse_ = rt_renderer_->local_di_diffuse()->handle();
                    frame_rt_local_di_specular_ = rt_renderer_->local_di_specular()->handle();
                    frame_rt_local_di_direction_ = rt_renderer_->local_di_direction()->handle();
                    frame_rt_local_di_active_ = true;
                }
                if (rt_renderer_->stats().reflections_traced &&
                    rt_renderer_->reflection_target() != nullptr) {
                    frame_rt_reflection_max_roughness_ = rt_reflection_settings.max_roughness;
                    frame_rt_reflection_fade_start_ = rt_reflection_settings.fade_start;
                    rt_reflection_resolve_ready = true;
                }
            }

            // Host slot: after GTAO/GI/RT, before the forward pass. Scene
            // colour is about to be cleared by begin_offscreen_pass, so this is
            // for compute the forward pass will sample — see RenderHookSlot.
            if (!render_hooks_.empty(RenderHookSlot::PreForward)) {
                run_render_hooks(RenderHookSlot::PreForward, cmd, draw_camera, dt, nullptr,
                                 /*scene_targets_valid=*/true);
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

            // Host slot: the forward pass has ended and nothing is bound, so
            // scene colour (HDR) and scene depth (reversed-Z) are both readable.
            if (!render_hooks_.empty(RenderHookSlot::AfterOpaque)) {
                run_render_hooks(RenderHookSlot::AfterOpaque, cmd, draw_camera, dt, nullptr,
                                 /*scene_targets_valid=*/true);
            }

            // Resolve sparse reflections into the linear HDR scene BEFORE
            // water, TAA, auto exposure and bloom. The full-resolution pass is
            // guided by depth + material normal/roughness/metalness, so a
            // quarter-res stone texel cannot bleed onto grass or foliage; TAA
            // then stabilizes the traced signal with all other lighting.
            if (rt_reflection_resolve_ready && rt_renderer_ && cmd &&
                post_->offscreen_color() && post_->offscreen_depth() &&
                normal_rough_texture_) {
                BRACKEN_PROFILE_ZONE_COLOR("RT reflection resolve", 0x8E44AD);
                BRACKEN_GPU_SCOPE(cmd, "RT reflection resolve");
                BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "RTReflectionResolve");
                (void)rt_renderer_->resolve_reflections(
                    cmd, *post_->offscreen_color(), *post_->offscreen_depth(),
                    normal_rough_texture_->handle(), rt_frame_params,
                    rt_reflection_settings);
            }

            // ===== Water: its own pass after the opaque scene + skybox.
            // The optional analytic RT receiver runs before the HDR
            // color/depth snapshot (what refraction and shoreline softness
            // sample); the live targets are then re-bound
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

                // Opaque reflections have already been resolved, so the raw
                // quarter-resolution target can be recycled for a dedicated
                // water receiver. This avoids a second target and, crucially,
                // avoids writing water into opaque depth before refraction has
                // captured the scene behind it.
                bool water_rt_reflection_ready = false;
                if (gfx.rt_reflections && rt_renderer_) {
                    if (gpu::GPUTexture* sim = water_->simulation_state_texture()) {
                        const auto& region = water_->region();
                        render::RtWaterReceiver receiver;
                        receiver.bounds_min = {region.min_x, region.min_z};
                        receiver.bounds_max = {region.max_x, region.max_z};
                        receiver.height = region.height;
                        receiver.roughness = region.reflection_roughness;
                        receiver.sim_extent = std::max(region.sim_extent, 1.0f);
                        receiver.sim_min = region.sim_center - glm::vec2(0.5f * receiver.sim_extent);
                        receiver.sim_displacement = region.sim_displacement;
                        receiver.sim_normal_strength = region.sim_normal_strength;
                        receiver.flow = region.flow;
                        receiver.flow_speed = region.flow_speed;
                        receiver.detail_wave_strength = region.detail_wave_strength;
                        receiver.time = scene_time_;
                        water_rt_reflection_ready = rt_renderer_->render_water_reflections(
                            cmd, *post_->offscreen_depth(), *sim, rt_frame_params,
                            rt_reflection_settings, receiver);
                    }
                }

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
                        SDL_GPUTexture* traced_reflection =
                            water_rt_reflection_ready && rt_renderer_->reflection_target()
                                ? rt_renderer_->reflection_target()->handle()
                                : dummy_black_texture_->handle();
                        water_->render(water_pass, cmd, draw_camera.view_projection, draw_camera.projection,
                                       draw_camera.position, scene_time_, w, h, water_scene_copy_->handle(),
                                       water_depth_copy_->handle(), specular, frame_ibl_active_,
                                       traced_reflection, water_rt_reflection_ready,
                                       rt_reflection_settings.intensity,
                                       gfx.sun_intensity);
                        SDL_EndGPURenderPass(water_pass);
                        if (collect_stats_) {
                            render_stats_.draw_calls += 1;
                        }
                    }
                }
            }

            // Host slot: after water, before the TAA resolve — anything drawn
            // here is temporally stabilized with the rest of the frame.
            if (!render_hooks_.empty(RenderHookSlot::BeforeTaa)) {
                run_render_hooks(RenderHookSlot::BeforeTaa, cmd, draw_camera, dt, nullptr,
                                 /*scene_targets_valid=*/true);
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
                    // The prefilter's cut is POST-exposure, and the adapted EV
                    // is the larger half of that product — so it gets the same
                    // state buffer under the same gate the composite pass uses.
                    // Ordering holds by construction: AutoExposure::render was
                    // recorded into this command buffer immediately above.
                    bloom_->render(cmd, *pipeline_registry_, scene_in, gfx.bloom_threshold, gfx.exposure,
                                   auto_exposure_active ? auto_exposure_->state_buffer() : nullptr);
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
                                        draw_camera, env_, god_rays_active, gfx.volumetric_fog, fog_density_mul,
                                        &frame_gi_);
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
                // upsample in the composite shader (contracts/depth_linear.hpp:
                // viewZ = |B / (depth + A)|, standard and reversed Z alike, as
                // long as A/B come from the actual projection matrix).
                const auto composite_depth = scene::depth_linear_terms(draw_camera.projection);
                composite_params.depth_a = composite_depth.a;
                composite_params.depth_b = composite_depth.b;
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
                    post_->render_composite_pass(cmd, *pipeline_registry_, composite_target,
                                                 bloom_tex, fog_tex, composite_params);
                }

                if (ldr) {
                    BRACKEN_PROFILE_ZONE_COLOR("FXAA", 0x16A085);
                    BRACKEN_GPU_SCOPE(cmd, "FXAA");
                    BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "FXAA");
                    post_->render_fxaa_pass(cmd, current_swapchain_);
                }

                // Host slot: the swapchain holds the tonemapped LDR image and
                // scene colour still holds the pre-tonemap HDR one.
                if (!render_hooks_.empty(RenderHookSlot::AfterComposite)) {
                    run_render_hooks(RenderHookSlot::AfterComposite, cmd, draw_camera, dt, current_swapchain_,
                                     /*scene_targets_valid=*/true);
                }
            } else {
                swapchain_unavailable = true;
                // No swapchain image this frame: the composite slot cannot run.
                count_skipped_render_hooks({RenderHookSlot::AfterComposite});
            }
        } else {
            // The fallback direct-to-swapchain path has no HDR offscreen
            // targets, so the three HDR-only slots have nothing to hand a
            // callback. Counted rather than silently skipped — a host whose
            // pass never ran needs to be able to tell that apart from a pass
            // that ran and drew nothing.
            count_skipped_render_hooks({RenderHookSlot::PreForward, RenderHookSlot::AfterOpaque,
                                        RenderHookSlot::BeforeTaa, RenderHookSlot::AfterComposite});

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
        prev_view_ = frame_cam_.view;
        have_prev_view_proj_ = true;
        // Palette double-buffer swap: this frame's palettes become next
        // frame's "previous" palettes (only valid if anything was uploaded).
        prev_palette_buffer_valid_ =
            bone_palette_matrix_count_ > 0 && bone_palette_buffers_[bone_palette_current_] != nullptr;
        bone_palette_current_ ^= 1;
        ++frame_index_; // advances the Halton jitter sequence
    } else {
        // No 3D content (or no camera was ever set): none of the four scene
        // slots has a pass to sit between. AfterUi still runs below — the UI
        // pass does not need 3D content.
        count_skipped_render_hooks({RenderHookSlot::PreForward, RenderHookSlot::AfterOpaque,
                                    RenderHookSlot::BeforeTaa, RenderHookSlot::AfterComposite});
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

        // Host slot: after the engine's UI pass, on the swapchain. Runs BEFORE
        // post_ui_callback_ (the dev overlay), so a host overlay ends up UNDER
        // the developer overlay rather than replacing it — which is the exact
        // failure the single-slot callback had.
        if (!render_hooks_.empty(RenderHookSlot::AfterUi)) {
            if (current_swapchain_ != nullptr) {
                run_render_hooks(RenderHookSlot::AfterUi, context_->current_command_buffer(), draw_camera, dt,
                                 current_swapchain_, /*scene_targets_valid=*/has_content && use_hdr_path);
            } else {
                count_skipped_render_hooks({RenderHookSlot::AfterUi});
            }
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
    } else {
        // A late swapchain-acquire failure retires the UI pass too, so the one
        // slot that normally survives every path does not run either.
        count_skipped_render_hooks({RenderHookSlot::AfterUi});
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

    // AFTER the frame's command buffer is submitted, never before: the readback
    // rides its own buffer and command buffers execute in submission order, so
    // requesting it next to the build would download the previous frame's
    // pyramid while recording this frame's camera. No-op unless a pyramid was
    // actually built this frame.
    submit_occlusion_readback();
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
    // Null unless the plan said yes. The field is a frame-constant read-only
    // input, so handing it across does not weaken the Prepare stage's
    // bit-identity contract (scene_prep.hpp PrepParams::occlusion).
    p.occlusion = frame_occlusion_active_ ? &occlusion_field_ : nullptr;

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

namespace bracken::scene {

// ============================================================================
// Hi-Z draw occlusion culling (GraphicsSettings::occlusion_culling)
// ============================================================================

void SceneRenderer::release_occlusion_resources() {
    if (context_ != nullptr) {
        if (occlusion_fence_ != nullptr) {
            // Waited, not abandoned: the fence guards a copy into
            // occlusion_transfer_, and releasing the buffer with the copy still
            // in flight is a use-after-free inside the driver.
            SDL_WaitForGPUFences(context_->device().handle(), true, &occlusion_fence_, 1);
            SDL_ReleaseGPUFence(context_->device().handle(), occlusion_fence_);
        }
        if (occlusion_transfer_ != nullptr) {
            context_->device().release_transfer_buffer(occlusion_transfer_);
        }
    }
    occlusion_fence_ = nullptr;
    occlusion_transfer_ = nullptr;
    occlusion_transfer_bytes_ = 0;
    occlusion_depth_.clear();
    occlusion_depth_ready_ = false;
    occlusion_field_ = OcclusionField{};
    // A build recorded on a command buffer we are abandoning owes no readback.
    occlusion_build_recorded_ = false;
    hzb_far_.reset();
}

void SceneRenderer::resolve_occlusion_readback() {
    BRACKEN_ASSERT_MAIN_THREAD();
    if (occlusion_fence_ == nullptr || context_ == nullptr || !hzb_far_) {
        return;
    }
    SDL_GPUDevice* dev = context_->device().handle();

    // DETERMINISTIC wait, deliberately. The alternative — "skip the test if the
    // fence has not signalled" — makes the culled set depend on GPU timing, so
    // two runs of the same scene render different frames and every pinned
    // screenshot baseline stops meaning anything. This fence was submitted a
    // full frame ago and is normally already signalled; if it ever is not, the
    // cost shows up as render time, not as present_wait.
    SDL_WaitForGPUFences(dev, true, &occlusion_fence_, 1);
    SDL_ReleaseGPUFence(dev, occlusion_fence_);
    occlusion_fence_ = nullptr;

    const int mip_w = render::hzb_mip_size(hzb_far_->width(), k_occlusion_level);
    const int mip_h = render::hzb_mip_size(hzb_far_->height(), k_occlusion_level);
    const std::size_t texels = static_cast<std::size_t>(mip_w) * static_cast<std::size_t>(mip_h);
    if (texels == 0 || occlusion_transfer_ == nullptr) {
        return;
    }

    void* mapped = context_->device().map_transfer_buffer(occlusion_transfer_, false);
    if (mapped == nullptr) {
        return;
    }
    occlusion_depth_.resize(texels);
    std::memcpy(occlusion_depth_.data(), mapped, texels * sizeof(float));
    context_->device().unmap_transfer_buffer(occlusion_transfer_);

    // The camera that rasterized THIS depth, promoted from the pending slot.
    occlusion_view_proj_ = pending_occlusion_view_proj_;
    occlusion_camera_position_ = pending_occlusion_camera_position_;
    occlusion_camera_forward_ = pending_occlusion_camera_forward_;

    occlusion_field_.depth = occlusion_depth_;
    occlusion_field_.mip_w = mip_w;
    occlusion_field_.mip_h = mip_h;
    occlusion_field_.screen_w = hzb_far_->width();
    occlusion_field_.screen_h = hzb_far_->height();
    occlusion_field_.level = k_occlusion_level;
    occlusion_field_.view_proj = occlusion_view_proj_;
    occlusion_depth_ready_ = true;
}

void SceneRenderer::build_occlusion_pyramid(SDL_GPUCommandBuffer* cmd, const CameraState& camera) {
    BRACKEN_ASSERT_MAIN_THREAD();
    if (cmd == nullptr || context_ == nullptr || post_->offscreen_depth() == nullptr) {
        return;
    }
    if (!hzb_far_ && !hzb_far_init_failed_) {
        auto pyramid = std::make_unique<render::HzbPyramid>();
        if (pyramid->init(context_->device(), context_->width(), context_->height(),
                          render::HzbReduction::Farthest)) {
            hzb_far_ = std::move(pyramid);
        } else {
            // init() logged why. Latched: retrying a format-support check and a
            // mip-chain allocation every frame is a per-frame stall plus a log
            // flood, exactly as for hzb_.
            hzb_far_init_failed_ = true;
        }
    }
    if (!hzb_far_) {
        return;
    }
    if (hzb_far_->width() != context_->width() || hzb_far_->height() != context_->height()) {
        // A resize retires the pending readback AND the adopted depth: the old
        // level describes a different frame at a different resolution, and
        // reusing it would address a smaller buffer with larger indices.
        if (occlusion_fence_ != nullptr) {
            SDL_WaitForGPUFences(context_->device().handle(), true, &occlusion_fence_, 1);
            SDL_ReleaseGPUFence(context_->device().handle(), occlusion_fence_);
            occlusion_fence_ = nullptr;
        }
        hzb_far_->resize(context_->width(), context_->height());
        occlusion_depth_ready_ = false;
        occlusion_field_ = OcclusionField{};
    }

    {
        BRACKEN_PROFILE_ZONE_COLOR("HZB far", 0x8E6FC4);
        BRACKEN_GPU_SCOPE(cmd, "HZBFar");
        BRACKEN_GPU_TIMER(gpu_timer_pool_, cmd, "HZBFar");
        hzb_far_->build(cmd, *pipeline_registry_, *post_->offscreen_depth());
    }

    // The download is NOT requested here — see submit_occlusion_readback(). It
    // has to wait until `cmd` itself has been submitted, because the dispatches
    // above are only RECORDED at this point. All that is recorded now is which
    // camera rasterized what the pyramid will hold once `cmd` runs.
    //
    // The UNJITTERED matrix, deliberately, even though the prepass rasterized
    // with the jittered one. The discrepancy is at most half a pixel at mip 0,
    // i.e. 1/32 of a texel at level 4, and the footprint walk dilates by a
    // whole texel on every side — so it is inside the conservatism the test
    // already carries, and using the jittered matrix would make the cull depend
    // on the TAA phase, which is a frame-index input nobody wants in a culling
    // decision.
    occlusion_build_view_proj_ = frame_cam_.view_proj_unjittered;
    occlusion_build_camera_position_ = camera.position;
    occlusion_build_camera_forward_ =
        glm::normalize(-glm::vec3(camera.view[0][2], camera.view[1][2], camera.view[2][2]));
    occlusion_build_recorded_ = true;
}

void SceneRenderer::submit_occlusion_readback() {
    BRACKEN_ASSERT_MAIN_THREAD();
    // ORDER IS THE WHOLE POINT OF THIS FUNCTION EXISTING. A readback rides its
    // OWN command buffer, submitted the instant it is recorded, and SDL3 GPU
    // executes command buffers in submission order — so a download requested
    // next to the build (which is merely RECORDED on the frame's buffer,
    // submitted at present) runs BEFORE that build and captures the PREVIOUS
    // frame's pyramid. The depth would then be two frames old while the camera
    // matrix stored beside it was one, and a projection tested against depth
    // rasterized with a different matrix is not conservative at all — which is
    // exactly the false-cull failure this feature must not have.
    //
    // Called after the frame's command buffer has been submitted, so the copy
    // observes this frame's pyramid and pairs with this frame's camera.
    const bool recorded = occlusion_build_recorded_;
    occlusion_build_recorded_ = false;
    if (!recorded || !hzb_far_ || context_ == nullptr) {
        return;
    }
    // One readback in flight at a time. A second request while the first is
    // pending would race two copies into one transfer buffer.
    if (occlusion_fence_ != nullptr) {
        return;
    }
    const std::size_t need = hzb_far_->level_readback_bytes(k_occlusion_level);
    if (need == 0) {
        return;
    }
    if (occlusion_transfer_ == nullptr || occlusion_transfer_bytes_ < need) {
        if (occlusion_transfer_ != nullptr) {
            context_->device().release_transfer_buffer(occlusion_transfer_);
        }
        SDL_GPUTransferBufferCreateInfo info = {};
        info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD;
        info.size = static_cast<Uint32>(need);
        occlusion_transfer_ = context_->device().create_transfer_buffer(info);
        occlusion_transfer_bytes_ = occlusion_transfer_ != nullptr ? need : 0;
    }
    if (occlusion_transfer_ == nullptr) {
        return;
    }

    occlusion_fence_ = hzb_far_->request_level_readback(k_occlusion_level, occlusion_transfer_);
    if (occlusion_fence_ != nullptr) {
        // Remember what rasterized it; the next frame's travel/rotation gates
        // are measured against exactly this camera, not against "last frame".
        pending_occlusion_view_proj_ = occlusion_build_view_proj_;
        pending_occlusion_camera_position_ = occlusion_build_camera_position_;
        pending_occlusion_camera_forward_ = occlusion_build_camera_forward_;
    }
}

void SceneRenderer::build_raytracing_scene(const ModelManager& models, bool want_attributes) {
    if (!rt_renderer_ || context_ == nullptr) {
        return;
    }
    rt_renderer_->build_scene(context_->device(), models, want_attributes);
    rt_scene_built_ = true;
    rt_scene_has_attributes_ = want_attributes;
}

void SceneRenderer::ensure_raytracing_scene(bool want_attributes) {
    if (!rt_renderer_ || context_ == nullptr || model_manager_ == nullptr) {
        return;
    }
    // Reflections need per-triangle attributes; a structure built for shadows
    // alone does not have them, so enabling reflections later has to rebuild.
    const bool need_rebuild = !rt_scene_built_ || (want_attributes && !rt_scene_has_attributes_);
    if (!need_rebuild) {
        return;
    }
    // Building every model's BLAS is on the order of a second for a scene this
    // size. That is a visible hitch when it happens because someone ticked a
    // checkbox, so say what is going on rather than looking like a hang.
    // "reflections or GI": both need the attribute buffer to shade a hit, and a
    // host that only ticked `gi_mode` deserves to see why its frame time just
    // grew by a second and its memory by ~27 MB.
    BRACKEN_LOG_INFO(LOG_TAG, "ray tracing enabled: building the acceleration structure{}",
                     want_attributes ? " with triangle attributes (reflections/GI)" : "");
    build_raytracing_scene(*model_manager_, want_attributes);
}

void SceneRenderer::refresh_raytracing_instances(const FramePacket& frame,
                                                 SDL_GPUCommandBuffer* cmd,
                                                 const glm::vec3& camera_pos, float max_range) {
    if (!rt_renderer_) {
        return;
    }

    // GraphicsSettings::rt_max_instance_range, in world units. Unlimited maps
    // to infinity precisely so this is a single test rather than a comparison
    // against a sentinel on every instance of every frame.
    const bool range_limited = std::isfinite(max_range) && model_manager_ != nullptr;

    // Distance is measured to the instance's world-space BOUNDS, not its
    // origin, from the same Model::bounding_* pair the main-view cull uses
    // (scene_prep.cpp's bin_main_chunk) — so a cathedral whose origin sits
    // 30 m past the cutoff but whose wall is 2 m in front of the camera is
    // still traced, and nothing pops as the camera creeps up on it. A model
    // the manager cannot resolve has no bounds and is KEPT, matching the
    // culling path's "no bounds, stays visible" rule: a missing handle is a
    // diagnostic elsewhere, not a licence to silently drop a caster here.
    //
    // COST TRADE, stated because it is the knob's whole price: with a finite
    // range the instance list becomes a function of camera position, so a
    // camera crossing the threshold changes the list, changes the content
    // hash, and correctly re-triggers a TLAS rebuild that Unlimited (the
    // default) never pays on a static scene. That is the deal — traversal and
    // per-frame TLAS size traded against rebuild frequency.
    auto within_range = [&](ModelHandle handle, const glm::mat4& t) {
        if (!range_limited) {
            return true;
        }
        const Model* model = model_manager_->get_model(handle);
        if (model == nullptr) {
            return true;
        }
        const glm::vec3 world_center = glm::vec3(t * glm::vec4(model->bounding_center, 1.0f));
        const float world_radius = model->bounding_half_diag * max_scale_factor(t);
        // Squared compare against (range + radius): exactly "are the bounds
        // entirely beyond the cutoff", with no square root per instance.
        const glm::vec3 to_camera = world_center - camera_pos;
        const float reach = max_range + world_radius;
        return glm::dot(to_camera, to_camera) <= reach * reach;
    };

    // Straight off the packet: whatever the host described this frame is what
    // the rays see. There is no second list for a host to forget to update.
    rt_instances_scratch_.clear();
    const auto models = frame.model_commands();
    rt_instances_scratch_.reserve(models.size());
    for (const ModelDrawCommand& draw : models) {
        if (draw.model_handle != INVALID_MODEL_HANDLE && within_range(draw.model_handle, draw.transform)) {
            rt_instances_scratch_.emplace_back(draw.model_handle, draw.transform);
        }
    }

    // SKINNED DRAWS CAST TOO. They were absent from this list entirely, which
    // meant every animated character was invisible to the rays — and because
    // traced visibility REPLACES the cascade result rather than combining with
    // it, the effect was not "falls back to CSM" but "has no sun shadow at
    // all". The player character in the playground was the obvious victim.
    //
    // THE BIND-POSE LIMITATION, and what now answers it (§5.8 of
    // docs/rt_lighting_plan.md). The BLAS is built from the model's bind-pose
    // CPU geometry, so a skinned instance's TRIANGLES are its bind pose,
    // translated and rotated into place — a rigid statue that tracks the
    // character around the world without animating. That was visibly wrong for
    // the two VISIBILITY signals, whose answer replaces the cascade shadow, and
    // acceptable for the closest-hit ones, which need something shadeable where
    // the character is and cannot see the pose error at reflection or probe
    // scale.
    //
    // So the split is made explicit rather than suffered: the skinned instances
    // go at the END of the list, `skinned_first` marks where they start, and
    // RtScene stamps `k_tlas_skinned_flag` into their node offsets. The any-hit
    // walks skip flagged instances and test the frame's ShadowCapsules — which
    // the HOST derives from its live skeletons and emits through the packet —
    // while the closest-hit walks are untouched.
    //
    // The append order IS the partition; there is no sort. Everything above
    // came from model_commands(), everything below from skinned_commands().
    const auto skinned_first = static_cast<uint32_t>(rt_instances_scratch_.size());
    for (const SkinnedDrawCommand& draw : frame.skinned_commands()) {
        if (draw.model_handle != INVALID_MODEL_HANDLE && within_range(draw.model_handle, draw.transform)) {
            rt_instances_scratch_.emplace_back(draw.model_handle, draw.transform);
        }
    }

    // The filter runs BEFORE this, deliberately: set_instances hashes the list
    // it is given, so what the TLAS skip-check sees is the filtered content.
    rt_renderer_->set_instances(rt_instances_scratch_, cmd, skinned_first);
}

void SceneRenderer::set_raytracing_instances(
    std::span<const std::pair<ModelHandle, glm::mat4>> instances) {
    if (rt_renderer_) {
        rt_renderer_->set_instances(instances);
    }
}

const render::RtStats& SceneRenderer::raytracing_stats() const {
    static const render::RtStats k_empty{};
    return rt_renderer_ ? rt_renderer_->stats() : k_empty;
}

const render::gi::GiStats& SceneRenderer::gi_stats() const {
    static const render::gi::GiStats k_empty{};
    return probe_volume_ ? probe_volume_->stats() : k_empty;
}

} // namespace bracken::scene
