#include "scene_renderer.hpp"
#include "engine/core/assert.hpp"
#include "engine/core/jobs/job_system.hpp"
#include "engine/core/profiler.hpp"
#include "engine/gpu/gpu_buffer.hpp"
#include "engine/gpu/gpu_debug.hpp"
#include "engine/gpu/gpu_pipeline.hpp"
#include "engine/gpu/gpu_texture.hpp"
#include "engine/gpu/gpu_uniforms.hpp"
#include "engine/gpu/pipeline_registry.hpp"
#include "engine/heightmap.hpp"
#include "engine/model_loader.hpp"
#include "engine/render/ambient_occlusion.hpp"
#include "engine/render/bloom.hpp"
#include "engine/render/effect_renderer.hpp"
#include "engine/render/grass_renderer.hpp"
#include "engine/render/render_context.hpp"
#include "engine/render/shadow_map.hpp"
#include "engine/render/terrain_renderer.hpp"
#include "engine/render/ui_renderer.hpp"
#include "engine/render/volumetric_fog.hpp"
#include "engine/render/world_renderer.hpp"
#include "engine/systems/effect_system.hpp"
#include "frustum.hpp"
#include "SDL3/SDL_gpu.h"
#include "SDL3/SDL_video.h"
#include <SDL3_image/SDL_image.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <type_traits>

#include "engine/graphics_settings.hpp"
#include "engine/scene/camera_state.hpp"
#include "engine/scene/render_scene.hpp"
#include "engine/scene/transform_math.hpp"
#include "engine/scene/ui_scene.hpp"
#include "glm/common.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"
#include "glm/matrix.hpp"

namespace mmo::engine::scene {

namespace gpu = mmo::engine::gpu;
namespace render = mmo::engine::render;

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
}

SceneRenderer::SceneRenderer()
    : pipeline_registry_(std::make_unique<gpu::PipelineRegistry>()),
      terrain_(std::make_unique<render::TerrainRenderer>()),
      world_(std::make_unique<render::WorldRenderer>()),
      ui_(std::make_unique<render::UIRenderer>()),
      effects_(std::make_unique<render::EffectRenderer>()),
      effect_system_(std::make_unique<mmo::engine::systems::EffectSystem>()),
      model_manager_(std::make_unique<ModelManager>()),
      grass_renderer_(std::make_unique<render::GrassRenderer>()),
      shadow_map_(std::make_unique<render::ShadowMap>()),
      ao_(std::make_unique<render::AmbientOcclusion>()),
      bloom_(std::make_unique<render::Bloom>()),
      volumetric_fog_(std::make_unique<render::VolumetricFog>()) {}

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
float SceneRenderer::get_terrain_height(float x, float z) {
    return terrain_->get_height(x, z);
}

bool SceneRenderer::init(render::RenderContext& context, float world_width, float world_height) {
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

    if (!world_->init(context_->device(), *pipeline_registry_, world_width, world_height, model_manager_.get())) {
        std::cerr << "Failed to initialize world renderer" << '\n';
        return false;
    }

    world_->set_terrain_height_func([this](float x, float z) { return terrain_->get_height(x, z); });

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

    if (grass_renderer_) {
        grass_renderer_->init(context_->device(), *pipeline_registry_, world_width, world_height);
    }

    const GraphicsSettings& gfx = graphics_ ? *graphics_ : default_graphics_;
    static constexpr int resolution_table[] = {512, 1024, 2048, 4096};
    int shadow_res = resolution_table[std::clamp(gfx.shadow_resolution, 0, 3)];
    shadow_map_->set_active_cascades(gfx.shadow_cascades + 1);
    if (!shadow_map_->init(context_->device(), shadow_res)) {
        std::cerr << "Warning: Failed to initialize shadow map (shadows disabled)" << '\n';
    }

    if (!ao_->init(context_->device(), w, h)) {
        std::cerr << "Warning: Failed to initialize GTAO (AO disabled)" << '\n';
    }

    if (!bloom_->init(context_->device(), w, h)) {
        std::cerr << "Warning: Failed to initialize Bloom (bloom disabled)" << '\n';
    }

    if (!volumetric_fog_->init(context_->device(), w, h)) {
        std::cerr << "Warning: Failed to initialize Volumetric Fog (fog disabled)" << '\n';
    }

    gpu_timer_pool_.init(context_->device(), 64);

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

    if (default_sampler_ && context_) {
        context_->device().release_sampler(default_sampler_);
        default_sampler_ = nullptr;
    }

    gpu_timer_pool_.shutdown();
    volumetric_fog_->shutdown();
    bloom_->shutdown();
    ao_->shutdown();
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
    // Bound on the normal-map slot when the material has no normal map, so the
    // shader's branch-free path produces N == N_geo.
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
}

// ============================================================================
// Configuration
// ============================================================================

void SceneRenderer::set_screen_size(int width, int height) {
    ui_->set_screen_size(width, height);
    if (ao_->is_ready()) {
        ao_->resize(width, height);
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
    depth_target.clear_depth = 1.0f;
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

void SceneRenderer::render_frame(RenderScene& scene, const UIScene& ui_scene, const CameraState& camera, float dt) {
    ENGINE_PROFILE_ZONE("SceneRenderer::render_frame");
    if (collect_stats_) {
        render_stats_ = {};
        render_stats_.jobs_pending = core::jobs::JobSystem::instance().pending_count();
    }
    const GraphicsSettings& gfx = graphics_ ? *graphics_ : default_graphics_;

    // Cache frame-level state once (avoids repeated ternary/function calls)
    frame_fog_active_ = gfx.fog_enabled;
    frame_draw_dist_sq_ = gfx.get_draw_distance() * gfx.get_draw_distance();
    frame_do_frustum_cull_ = gfx.frustum_culling;
    scene_time_ += dt;

    // Propagate the environment to every renderer that builds its own lighting
    // uniforms; env_ is the single source of truth for sun/sky/fog this frame.
    light_dir_ = env_.sun_dir;
    world_->set_environment(env_);
    terrain_->set_environment(env_);
    if (grass_renderer_) {
        grass_renderer_->set_environment(env_);
    }

    // Sky/cloud time must advance even when the skybox is toggled off or the
    // frame has no 3D content, so the sky never freezes (grass wind runs off
    // scene_time_, accumulated above).
    world_->update(dt);

    // Update particle effect system
    auto get_terrain_height = [this](float x, float z) -> float { return terrain_->get_height(x, z); };
    effect_system_->update(dt, get_terrain_height);

    // Consume queued particle-effect spawn commands BEFORE any render pass so
    // newly referenced effect models can be GPU-uploaded in the pre-pass
    // section below (uploads cannot run inside a render pass).
    for (const auto& spawn_cmd : scene.particle_effect_spawns()) {
        if (spawn_cmd.definition) {
            effect_system_->spawn_effect(spawn_cmd.definition, spawn_cmd.position, spawn_cmd.direction,
                                         spawn_cmd.range);
        }
    }
    scene.clear_particle_effect_spawns();

    // Pre-upload every model referenced by active effects; the mid-pass upload
    // path in EffectRenderer is now a logged skip, never a submit hitch.
    effects_->ensure_models_uploaded(*effect_system_);

    begin_frame();

    // Minimized windows can't acquire a swapchain image; skip all recording but
    // still submit the command buffer so the frame lifecycle stays balanced.
    if (context_->window() && (SDL_GetWindowFlags(context_->window()) & SDL_WINDOW_MINIMIZED) != 0) {
        end_frame();
        return;
    }

    gpu_timer_pool_.begin_frame();

    bool has_content = scene.has_3d_content();
    // Always render 3D through the HDR offscreen target so scene lighting stays
    // in linear HDR; the composite pass tonemaps + gamma-corrects to the swapchain.
    // ao_mode == 0 simply skips the AO pass (aoStrength=0 in composite).
    bool use_ao = ao_->is_ready();
    // Set when a late swapchain acquire fails: composite + UI are skipped but
    // the command buffer is still submitted normally.
    bool swapchain_unavailable = false;

    if (has_content) {
        // Extract main camera frustum once (reused in render_3d_scene for skinned models).
        frame_frustum_.extract_from_matrix(camera.view_projection);

        if (camera.projection != cached_projection_) {
            cached_projection_ = camera.projection;
            cached_inv_projection_ = glm::inverse(camera.projection);
        }

        // Update cascade matrices FIRST so we can extract per-cascade light frustums
        // and use them to cull shadow casters per cascade in build_instance_batches.
        const bool shadows_on = shadow_map_->is_ready() && gfx.shadow_mode > 0;
        if (shadows_on) {
            // Near plane recovered from the RH_ZO projection (n = m32/m22).
            // Shadow range is capped well below draw distance; the shader
            // fades shadows out over the tail instead of popping them off.
            const float cam_near = std::max(camera.projection[3][2] / camera.projection[2][2], 0.1f);
            const float shadow_far = std::min(gfx.get_draw_distance(), 1200.0f);
            shadow_map_->update(camera.view, camera.projection, light_dir_, cam_near, shadow_far);
            const int active = shadow_map_->active_cascades();
            for (int c = 0; c < active; ++c) {
                shadow_cascade_frustums_[c].extract_from_matrix(shadow_map_->cascades()[c].light_view_projection);
            }
        }

        // ===== CPU prep for all per-frame GPU uploads =====

        // Build instance batches (cull for main + per-cascade shadows) and
        // pack all skinned bone palettes tightly into CPU scratch.
        build_instance_batches(scene, camera, frame_frustum_);
        pack_bone_palettes(scene);

        // Grass chunk culling/visibility prep (upload happens below).
        const bool grass_active = scene.should_draw_grass() && gfx.grass_enabled && grass_renderer_;
        if (grass_active) {
            grass_renderer_->grass_view_distance =
                std::min(gfx.get_draw_distance(), render::GrassRenderer::MAX_VIEW_DISTANCE);
            grass_renderer_->update(dt, scene_time_);
        }

        // Build cluster light grid on the CPU. Skip the build + upload when the
        // scene has zero dynamic lights — the per-frag cluster path early-outs
        // via gridDim.w==0. We still need to bind the SSBOs to terrain/grass
        // pipelines below so those binding slots are valid; the buffers retain
        // their last-uploaded state (initially zero).
        const bool any_dyn_lights = !scene.point_lights().empty() || !scene.spot_lights().empty();
        const bool cluster_upload_needed = cluster_grid_ready_ && any_dyn_lights;
        if (cluster_upload_needed) {
            uint32_t cw = static_cast<uint32_t>(std::max(context_->width(), 1));
            uint32_t ch = static_cast<uint32_t>(std::max(context_->height(), 1));
            cluster_grid_.begin_frame(camera, cw, ch, 0.1f, gfx.get_draw_distance());
            for (const auto& l : scene.point_lights()) cluster_grid_.add_point_light(l);
            for (const auto& l : scene.spot_lights()) cluster_grid_.add_spot_light(l);
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
            ENGINE_PROFILE_ZONE_COLOR("FrameUploads", 0x27AE60);
            ENGINE_GPU_SCOPE(up_cmd, "FrameUploads");
            ENGINE_GPU_TIMER(gpu_timer_pool_, up_cmd, "FrameUploads");
            SDL_GPUCopyPass* copy_pass = SDL_BeginGPUCopyPass(up_cmd);
            if (copy_pass) {
                upload_instance_buffers(copy_pass);
                upload_bone_palettes(copy_pass);
                effects_->upload_particle_instances(copy_pass, *effect_system_);
                upload_debug_lines(scene, copy_pass);
                if (grass_active) {
                    grass_renderer_->upload_chunks(copy_pass, camera.position, &frame_frustum_);
                }
                if (cluster_upload_needed) {
                    cluster_grid_.upload(copy_pass);
                }
                SDL_EndGPUCopyPass(copy_pass);
            } else {
                std::cerr << "render_frame: Failed to begin frame upload copy pass" << '\n';
            }
        }

        if (shadows_on) {
            render_shadow_passes(scene, camera);
        }

        // Cache shadow uniforms (cascade matrices are stable after update()).
        if (shadow_map_->is_ready()) {
            frame_shadow_uniforms_ = shadow_map_->get_shadow_uniforms(gfx.shadow_mode);
        }

        if (cluster_grid_ready_) {
            terrain_->set_cluster_lighting(cluster_grid_.light_data_buffer(), cluster_grid_.cluster_offsets_buffer(),
                                           cluster_grid_.light_indices_buffer(), &frame_cluster_params_,
                                           sizeof(frame_cluster_params_));
            if (grass_renderer_) {
                grass_renderer_->set_cluster_lighting(
                    cluster_grid_.light_data_buffer(), cluster_grid_.cluster_offsets_buffer(),
                    cluster_grid_.light_indices_buffer(), &frame_cluster_params_, sizeof(frame_cluster_params_));
            }
        }

        SDL_GPUCommandBuffer* cmd = context_->current_command_buffer();

        if (use_ao) {
            // === GTAO PATH: offscreen 3D + post chain; swapchain acquired late ===

            // Resize post targets from the window size; the swapchain isn't
            // acquired until composite so the GPU can start on the offscreen
            // pass without waiting for a presentable image.
            ao_->resize(context_->width(), context_->height());
            bloom_->resize(context_->width(), context_->height());
            volumetric_fog_->resize(context_->width(), context_->height());

            // Offscreen clear tracks the environment's horizon fog color so
            // unrendered sky pixels melt into the atmosphere (linear HDR).
            ao_->set_clear_color(env_.fog_color_horizon);

            {
                ENGINE_PROFILE_ZONE_COLOR("AOOffscreen", 0xC0392B);
                ENGINE_GPU_SCOPE(cmd, "AOOffscreen");
                ENGINE_GPU_TIMER(gpu_timer_pool_, cmd, "AOOffscreen");
                main_render_pass_ = ao_->begin_offscreen_pass(cmd);
                had_main_pass_this_frame_ = (main_render_pass_ != nullptr);

                if (main_render_pass_ && cmd) {
                    render_3d_scene(scene, camera);
                    ao_->end_offscreen_pass();
                    main_render_pass_ = nullptr;
                }
            }

            // AO computation + blur + bloom + fog (none of these touch the swapchain)
            glm::mat4 inv_proj = cached_inv_projection_;
            if (gfx.ao_mode > 0 && cmd) {
                ENGINE_PROFILE_ZONE_COLOR("SSAO/GTAO", 0x16A085);
                const char* ao_name = (gfx.ao_mode == 1) ? "SSAO" : "GTAO";
                ENGINE_GPU_SCOPE(cmd, ao_name);
                ENGINE_GPU_TIMER(gpu_timer_pool_, cmd, ao_name);
                if (gfx.ao_mode == 1) {
                    ao_->render_ssao_pass(cmd, *pipeline_registry_, camera.projection, inv_proj, gfx.ao_radius);
                } else {
                    ao_->render_gtao_pass(cmd, *pipeline_registry_, camera.projection, inv_proj, gfx.ao_radius);
                }
                {
                    ENGINE_PROFILE_ZONE_COLOR("BlurAO", 0x16A085);
                    ENGINE_GPU_SCOPE(cmd, "BlurAO");
                    ENGINE_GPU_TIMER(gpu_timer_pool_, cmd, "BlurAO");
                    ao_->render_blur_pass(cmd, *pipeline_registry_);
                }
            }

            // Bloom: downsample + upsample chain on offscreen color
            SDL_GPUTexture* bloom_tex = nullptr;
            if (gfx.bloom_enabled && bloom_->is_ready()) {
                SDL_GPUTexture* scene_in = ao_->offscreen_color() ? ao_->offscreen_color()->handle() : nullptr;
                if (cmd && scene_in) {
                    ENGINE_PROFILE_ZONE_COLOR("Bloom", 0x16A085);
                    ENGINE_GPU_SCOPE(cmd, "Bloom");
                    ENGINE_GPU_TIMER(gpu_timer_pool_, cmd, "Bloom");
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
                ao_->offscreen_depth()) {
                ENGINE_PROFILE_ZONE_COLOR("VolumetricFog", 0x16A085);
                ENGINE_GPU_SCOPE(cmd, "VolumetricFog");
                ENGINE_GPU_TIMER(gpu_timer_pool_, cmd, "VolumetricFog");
                // Default fog_density is 0.05; treat as a multiplier on the
                // environment's base density (env_.fog_volumetric_density).
                float fog_density_mul = gfx.fog_density / 0.05f;
                volumetric_fog_->render(cmd, *pipeline_registry_, ao_->offscreen_depth()->handle(), *shadow_map_,
                                        camera, env_, god_rays_active, gfx.volumetric_fog, fog_density_mul);
                fog_tex = volumetric_fog_->fog_texture();
            }

            // Late swapchain acquire: only composite/FXAA + UI need the image.
            current_swapchain_ = context_->acquire_swapchain_texture(cmd);
            if (current_swapchain_ && cmd) {
                render::AmbientOcclusion::CompositeParams composite_params;
                composite_params.bloom_strength = gfx.bloom_enabled ? gfx.bloom_strength : 0.0f;
                composite_params.ao_strength = gfx.ao_mode > 0 ? gfx.ao_strength : 0.0f;
                composite_params.exposure = gfx.exposure;
                composite_params.tonemap_mode = gfx.tonemap_mode;
                composite_params.contrast = gfx.contrast;
                composite_params.saturation = gfx.saturation;
                // Depth linearization constants for the joint bilateral AO/fog
                // upsample in the composite shader (glm RH_ZO).
                composite_params.depth_a = camera.projection[2][2];
                composite_params.depth_b = camera.projection[3][2];

                // FXAA runs on the tonemapped LDR image: composite into the
                // intermediate LDR target, then FXAA resolves to the swapchain.
                // aa_mode == 0 composites straight to the swapchain as before.
                SDL_GPUTexture* ldr = (gfx.aa_mode == 1) ? ao_->ldr_target() : nullptr;
                SDL_GPUTexture* composite_target = ldr ? ldr : current_swapchain_;

                {
                    ENGINE_PROFILE_ZONE_COLOR("Composite", 0x16A085);
                    ENGINE_GPU_SCOPE(cmd, "Composite");
                    ENGINE_GPU_TIMER(gpu_timer_pool_, cmd, "Composite");
                    ao_->render_composite_pass(cmd, *pipeline_registry_, composite_target, bloom_tex, fog_tex,
                                               composite_params);
                }

                if (ldr) {
                    ENGINE_PROFILE_ZONE_COLOR("FXAA", 0x16A085);
                    ENGINE_GPU_SCOPE(cmd, "FXAA");
                    ENGINE_GPU_TIMER(gpu_timer_pool_, cmd, "FXAA");
                    ao_->render_fxaa_pass(cmd, current_swapchain_);
                }
            } else {
                swapchain_unavailable = true;
            }
        } else {
            // === NORMAL PATH: render directly to swapchain ===

            ENGINE_PROFILE_ZONE_COLOR("MainScene", 0xC0392B);
            ENGINE_GPU_SCOPE(cmd, "MainScene");
            ENGINE_GPU_TIMER(gpu_timer_pool_, cmd, "MainScene");
            begin_main_pass();

            if (main_render_pass_ && context_->current_command_buffer()) {
                render_3d_scene(scene, camera);
            }

            end_main_pass();
        }
    }

    if (!swapchain_unavailable) {
        if (SDL_GPUCommandBuffer* ui_cmd = context_->current_command_buffer()) {
            ENGINE_PROFILE_ZONE_COLOR("UI", 0x16A085);
            ENGINE_GPU_SCOPE(ui_cmd, "UI");
            ENGINE_GPU_TIMER(gpu_timer_pool_, ui_cmd, "UI");
            begin_ui();
            render_ui_commands(ui_scene, camera);
            for (const auto& billboard : scene.billboards()) {
                draw_billboard_3d(billboard, camera);
            }
            end_ui();
        }

        // current_swapchain_ can still be null here (begin_ui acquire failure
        // on the no-content / non-AO paths); the callback renders into the
        // swapchain, so skip it rather than hand it a null target.
        if (post_ui_callback_ && current_swapchain_) {
            if (SDL_GPUCommandBuffer* post_cmd = context_->current_command_buffer()) {
                ENGINE_PROFILE_ZONE_COLOR("PostUICallback", 0x16A085);
                ENGINE_GPU_SCOPE(post_cmd, "PostUICallback");
                ENGINE_GPU_TIMER(gpu_timer_pool_, post_cmd, "PostUICallback");
                post_ui_callback_(post_cmd, current_swapchain_);
            }
        }
    }

    if (SDL_GPUCommandBuffer* end_cmd = context_->current_command_buffer()) {
        gpu_timer_pool_.end_frame(end_cmd);
    }
    if (collect_stats_) {
        gpu_timer_pool_.read_results(render_stats_.pass_times_ms);
        ENGINE_PROFILE_PLOT("render.draw_calls", static_cast<int64_t>(render_stats_.draw_calls));
        ENGINE_PROFILE_PLOT("render.triangles", static_cast<int64_t>(render_stats_.triangle_count));
        ENGINE_PROFILE_PLOT("render.entities", static_cast<int64_t>(render_stats_.entities_rendered));
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
// 3D Scene Rendering (shared between normal and AO paths)
// ============================================================================

void SceneRenderer::render_3d_scene(const RenderScene& scene, const CameraState& camera) {
    ENGINE_PROFILE_ZONE_COLOR("SceneRenderer::render_3d_scene", 0xC0392B);
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

    if (scene.should_draw_ground()) {
        SDL_PushGPUFragmentUniformData(cmd, 1, &frame_shadow_uniforms_, sizeof(frame_shadow_uniforms_));
        terrain_->render(main_render_pass_, cmd, camera.view, camera.projection, camera.position, light_dir_,
                         shadow_binding_count > 0 ? shadow_bindings : nullptr, shadow_binding_count, &frame_frustum_);
    }

    if (scene.should_draw_grass() && gfx.grass_enabled && grass_renderer_) {
        // view_distance + chunk upload already performed before render pass began.
        SDL_PushGPUFragmentUniformData(cmd, 1, &frame_shadow_uniforms_, sizeof(frame_shadow_uniforms_));
        grass_renderer_->render(main_render_pass_, cmd, camera.view, camera.projection, camera.position, light_dir_,
                                shadow_binding_count > 0 ? shadow_bindings : nullptr, shadow_binding_count);
    }

    // Use pre-cached frame state
    const Frustum& frustum = frame_frustum_;
    bool do_frustum_cull = frame_do_frustum_cull_;
    float draw_dist_sq = frame_draw_dist_sq_;

    // Render instanced static models (rocks, trees, etc.) - uses same frustum
    render_instanced_models(camera);

    // Render non-instanced static models (attack animations, no-fog, etc.)
    if (!non_instanced_commands_.empty() && main_render_pass_ && bind_model_pipeline(false)) {
        for (const auto* cmd : non_instanced_commands_) {
            render_model_command_inner(*cmd, camera);
        }
    }

    // Render skinned models individually (they have per-instance bone data)
    // Bind pipeline, shadow data and the bone-palette SSBO once before the loop
    {
        bool skinned_pipeline_bound = false;

        const auto& skinned_cmds = scene.skinned_commands();
        for (size_t i = 0; i < skinned_cmds.size(); ++i) {
            const SkinnedModelCommand& data = skinned_cmds[i];
            // Palettes were packed in command order by pack_bone_palettes();
            // INVALID means no model/skeleton or dropped over the palette cap.
            const uint32_t palette_offset =
                i < skinned_palette_offsets_.size() ? skinned_palette_offsets_[i] : INVALID_PALETTE_OFFSET;
            if (palette_offset == INVALID_PALETTE_OFFSET) {
                continue;
            }

            const glm::mat4& t = data.transform;
            glm::vec3 world_pos(t[3][0], t[3][1], t[3][2]);

            float dx = world_pos.x - camera.position.x;
            float dz = world_pos.z - camera.position.z;
            if (dx * dx + dz * dz > draw_dist_sq) {
                if (collect_stats_) {
                    render_stats_.entities_distance_culled++;
                }
                continue;
            }

            if (do_frustum_cull) {
                Model* model = model_manager_->get_model(data.model_handle);
                if (model) {
                    glm::vec3 world_center = glm::vec3(t * glm::vec4(model->bounding_center, 1.0f));
                    float max_scale = max_scale_factor(t);
                    if (!frustum.intersects_sphere(world_center, model->bounding_half_diag * max_scale)) {
                        if (collect_stats_) {
                            render_stats_.entities_frustum_culled++;
                        }
                        continue;
                    }
                }
            }

            if (collect_stats_) {
                render_stats_.entities_rendered++;
            }

            // Bind pipeline + shadow data + bone palettes once for all skinned models
            if (!skinned_pipeline_bound) {
                if (!bind_skinned_pipeline(false)) {
                    break;
                }
                skinned_pipeline_bound = true;
            }

            render_skinned_model_command_inner(data, camera, palette_offset);
        }
    }

    // Skybox last among opaques: depth writes are off and the VS emits z == 1
    // with a LESS_OR_EQUAL compare, so only sky pixels shade after the opaque
    // passes. (world_->update(dt) runs unconditionally in render_frame.)
    if (scene.should_draw_skybox() && gfx.skybox_enabled) {
        world_->render_skybox(main_render_pass_, cmd, camera.view, camera.projection);
    }

    // Sorted transparent pass: glTF alphaMode BLEND meshes queued by the
    // opaque walks, drawn back-to-front over opaques + sky with depth-test on
    // and depth-write off.
    render_transparent_models(camera);

    // Particle effects after the skybox: additive HDR with depth-test-on /
    // depth-write-off, so they composite over sky + opaques in any order.
    // Spawn commands were consumed pre-pass in render_frame; instance data
    // was packed + uploaded in the consolidated frame copy pass.
    effects_->draw_particle_effects(main_render_pass_, context_->current_command_buffer(), camera.view,
                                    camera.projection, camera.position);

    // Debug line rendering (only if game submitted any; vertex data was
    // uploaded in the consolidated frame copy pass)
    render_debug_lines(camera);
}

// ============================================================================
// Model Rendering
// ============================================================================

void SceneRenderer::render_model_command(const ModelCommand& cmd, const CameraState& camera) {
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

bool SceneRenderer::bind_model_pipeline(bool double_sided) {
    gpu::GPUPipeline* pipeline = pipeline_registry_->get_pipeline(
        double_sided ? gpu::PipelineType::ModelDoubleSided : gpu::PipelineType::Model);
    if (!pipeline || !main_render_pass_) {
        return false;
    }
    pipeline->bind(main_render_pass_);
    reset_model_bind_state();
    bind_shadow_data(main_render_pass_, context_->current_command_buffer(), 1);
    bind_cluster_lights(main_render_pass_, context_->current_command_buffer(), 2);
    model_pipeline_double_sided_ = double_sided;
    return true;
}

bool SceneRenderer::bind_skinned_pipeline(bool double_sided) {
    if (!main_render_pass_ || !bone_palette_buffer_) {
        return false;
    }
    gpu::GPUPipeline* pipeline = pipeline_registry_->get_pipeline(
        double_sided ? gpu::PipelineType::SkinnedModelDoubleSided : gpu::PipelineType::SkinnedModel);
    if (!pipeline) {
        return false;
    }
    pipeline->bind(main_render_pass_);
    reset_model_bind_state();
    bind_shadow_data(main_render_pass_, context_->current_command_buffer(), 1);
    bind_cluster_lights(main_render_pass_, context_->current_command_buffer(), 2);
    SDL_GPUBuffer* palette_buf = bone_palette_buffer_->handle();
    SDL_BindGPUVertexStorageBuffers(main_render_pass_, 0, &palette_buf, 1);
    skinned_pipeline_double_sided_ = double_sided;
    return true;
}

void SceneRenderer::reset_model_bind_state() {
    last_frag_tex_.fill(nullptr);
    model_lighting_valid_ = false;
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
    return u;
}

void SceneRenderer::render_model_command_inner(const ModelCommand& cmd, const CameraState& camera) {
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
    transform_uniforms.useSkinning = 0;

    bool fog_active = !cmd.no_fog && frame_fog_active_;

    gpu::ModelLightingUniforms lighting_uniforms = make_model_lighting(camera, fog_active);
    lighting_uniforms.tintColor = cmd.tint;

    SDL_PushGPUVertexUniformData(gpu_cmd, 0, &transform_uniforms, sizeof(transform_uniforms));

    for (auto& mesh : model->meshes) {
        if (mesh.shadow_only) {
            continue; // canopy proxy etc. — shadow-pass only
        }
        ENGINE_ASSERT(mesh.uploaded, "Mesh not uploaded before non-instanced render");
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

        // glTF doubleSided: switch to the no-cull variant (and back). Pushed
        // uniform data survives pipeline binds; sampler/uniform tracking is
        // conservatively reset by bind_model_pipeline.
        if (mesh.material.double_sided != model_pipeline_double_sided_ &&
            !bind_model_pipeline(mesh.material.double_sided)) {
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
            render_stats_.draw_calls++;
            render_stats_.triangle_count += mesh.index_count() / 3;
        }
        SDL_DrawGPUIndexedPrimitives(main_render_pass_, mesh.index_count(), 1, 0, 0, 0);
    }
}

// ============================================================================
// Instanced Rendering
// ============================================================================

void SceneRenderer::build_instance_batches(const RenderScene& scene, const CameraState& camera,
                                           const Frustum& frustum) {
    ENGINE_PROFILE_ZONE_COLOR("SceneRenderer::build_instance_batches", 0x27AE60);
    // Clear only the bins touched last frame (capacity preserved); empty bins
    // for never/rarely used models cost nothing to skip at iteration time.
    for (ModelHandle h : instance_batches_touched_) instance_batches_[h].clear();
    instance_batches_touched_.clear();
    for (int c = 0; c < 4; ++c) {
        for (ModelHandle h : shadow_batches_touched_[c]) shadow_instance_batches_[c][h].clear();
        shadow_batches_touched_[c].clear();
    }
    non_instanced_commands_.clear();

    // Dense-bin push helpers: grow the bin array to cover the handle (handles
    // are dense indices into ModelManager's slot array, so this stays small)
    // and record first touch for O(touched) clearing next frame.
    auto add_instance = [this](ModelHandle h, const gpu::InstanceData& inst) {
        if (h >= instance_batches_.size()) {
            instance_batches_.resize(h + 1);
        }
        auto& bin = instance_batches_[h];
        if (bin.empty()) {
            instance_batches_touched_.push_back(h);
        }
        bin.push_back(inst);
    };
    auto add_shadow_instance = [this](int c, ModelHandle h, const gpu::ShadowInstanceData& inst) {
        auto& bins = shadow_instance_batches_[c];
        if (h >= bins.size()) {
            bins.resize(h + 1);
        }
        auto& bin = bins[h];
        if (bin.empty()) {
            shadow_batches_touched_[c].push_back(h);
        }
        bin.push_back(inst);
    };

    // Shadow state: which cascades are active, and their light frustums.
    // Mirror render_frame's shadows_on gate — with shadow_mode == 0 the shadow
    // passes never run (and the cascade frustums are stale), so don't bin,
    // pack, or upload any shadow instances.
    const GraphicsSettings& gfx = graphics_ ? *graphics_ : default_graphics_;
    const int active_cascades =
        (shadow_map_->is_ready() && gfx.shadow_mode > 0) ? shadow_map_->active_cascades() : 0;

    bool do_frustum_cull = frame_do_frustum_cull_;
    float draw_dist_sq = frame_draw_dist_sq_;

    for (const auto& cmd : scene.model_commands()) {
        const glm::mat4& t = cmd.transform;
        glm::vec3 world_pos(t[3][0], t[3][1], t[3][2]);

        float dx = world_pos.x - camera.position.x;
        float dz = world_pos.z - camera.position.z;
        if (dx * dx + dz * dz > draw_dist_sq) {
            if (collect_stats_) {
                render_stats_.entities_distance_culled++;
            }
            continue;
        }

        ModelHandle handle = cmd.model_handle;

        glm::vec3 world_center{};
        float world_radius = 0.0f;
        bool have_bounds = false;
        if (do_frustum_cull || active_cascades > 0) {
            Model* model = model_manager_->get_model(handle);
            if (model) {
                world_center = glm::vec3(t * glm::vec4(model->bounding_center, 1.0f));
                world_radius = model->bounding_half_diag * max_scale_factor(t);
                have_bounds = true;
            }
        }

        bool main_visible = true;
        if (do_frustum_cull && have_bounds) {
            if (!frustum.intersects_sphere(world_center, world_radius)) {
                main_visible = false;
            }
        }

        if (!main_visible) {
            if (collect_stats_) {
                render_stats_.entities_frustum_culled++;
            }
            if (!cmd.force_non_instanced && have_bounds) {
                gpu::ShadowInstanceData shadow_instance{};
                shadow_instance.model = cmd.transform;
                for (int c = 0; c < active_cascades; ++c) {
                    if (shadow_cascade_frustums_[c].intersects_sphere(world_center, world_radius)) {
                        add_shadow_instance(c, handle, shadow_instance);
                    }
                }
            }
            continue;
        }

        if (cmd.force_non_instanced) {
            if (collect_stats_) {
                render_stats_.entities_rendered++;
            }
            non_instanced_commands_.push_back(&cmd);
            continue;
        }

        if (collect_stats_) {
            render_stats_.entities_rendered++;
        }

        gpu::InstanceData instance{};
        instance.model = cmd.transform;
        instance.normalMatrix = compute_normal_matrix(cmd.transform);
        instance.tint = cmd.tint;
        instance.noFog = cmd.no_fog ? 1.0f : 0.0f;
        add_instance(handle, instance);

        gpu::ShadowInstanceData shadow_instance{};
        shadow_instance.model = cmd.transform;
        for (int c = 0; c < active_cascades; ++c) {
            if (!have_bounds || shadow_cascade_frustums_[c].intersects_sphere(world_center, world_radius)) {
                add_shadow_instance(c, handle, shadow_instance);
            }
        }
    }
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
            // render_instanced_models so base_instance offsets line up.
            packed_instances_.clear();
            packed_instances_.reserve(total_instances);
            for (const auto& instances : instance_batches_) {
                packed_instances_.insert(packed_instances_.end(), instances.begin(), instances.end());
            }

            instance_storage_buffer_->update(copy_pass, packed_instances_.data(), required_size);
        }
    }

    // Upload shadow instance buffer (per-cascade sections packed contiguously).
    // Layout: [cascade 0 instances][cascade 1 instances]... each cascade in ascending handle order.
    size_t total_shadow = 0;
    for (int c = 0; c < 4; ++c) {
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

        packed_shadow_instances_.clear();
        packed_shadow_instances_.reserve(total_shadow);
        for (int c = 0; c < 4; ++c) {
            shadow_cascade_base_instance_[c] = static_cast<uint32_t>(packed_shadow_instances_.size());
            for (const auto& instances : shadow_instance_batches_[c]) {
                packed_shadow_instances_.insert(packed_shadow_instances_.end(), instances.begin(), instances.end());
            }
        }

        shadow_instance_storage_buffer_->update(copy_pass, packed_shadow_instances_.data(), required_size);
    } else {
        shadow_cascade_base_instance_ = {0, 0, 0, 0};
    }
}

// ============================================================================
// Bone Palettes
// ============================================================================

void SceneRenderer::pack_bone_palettes(const RenderScene& scene) {
    ENGINE_PROFILE_ZONE_COLOR("SceneRenderer::pack_bone_palettes", 0x8E44AD);
    const auto& cmds = scene.skinned_commands();
    skinned_palette_offsets_.clear();
    bone_palette_scratch_.clear();
    palette_dedupe_.clear();
    if (cmds.empty()) {
        return;
    }
    skinned_palette_offsets_.reserve(cmds.size());

    uint32_t palette_count = 0;
    uint32_t dropped = 0;
    for (const auto& cmd : cmds) {
        Model* model = model_manager_->get_model(cmd.model_handle);
        uint32_t joint_count = 0;
        if (model && model->has_skeleton) {
            joint_count = std::min(static_cast<uint32_t>(model->skeleton.joints.size()),
                                   static_cast<uint32_t>(mmo::engine::animation::MAX_BONES));
        }
        if (joint_count == 0 || !cmd.bone_matrices) {
            skinned_palette_offsets_.push_back(INVALID_PALETTE_OFFSET);
            continue;
        }

        // Dedupe shared bone arrays (notably the static identity palette used
        // for unbound skeletons): reuse the packed palette when it covers at
        // least this model's joint count. Linear scan — skinned counts are
        // small and this avoids per-frame map-node allocations.
        PaletteDedupeEntry* found = nullptr;
        for (auto& entry : palette_dedupe_) {
            if (entry.key == cmd.bone_matrices) {
                found = &entry;
                break;
            }
        }
        if (found && found->joint_count >= joint_count) {
            skinned_palette_offsets_.push_back(found->offset);
            continue;
        }

        if (palette_count >= MAX_BONE_PALETTES) {
            ++dropped;
            skinned_palette_offsets_.push_back(INVALID_PALETTE_OFFSET);
            continue;
        }

        const uint32_t offset = static_cast<uint32_t>(bone_palette_scratch_.size());
        bone_palette_scratch_.insert(bone_palette_scratch_.end(), cmd.bone_matrices->begin(),
                                     cmd.bone_matrices->begin() + joint_count);
        if (found) {
            // Repacked with more joints: point the entry at the new copy.
            found->offset = offset;
            found->joint_count = joint_count;
        } else {
            palette_dedupe_.push_back({cmd.bone_matrices, offset, joint_count});
        }
        skinned_palette_offsets_.push_back(offset);
        ++palette_count;
    }

    if (dropped > 0) {
        std::cerr << "[SceneRenderer] Bone palette cap (" << MAX_BONE_PALETTES << ") exceeded; dropped " << dropped
                  << " skinned draws this frame" << '\n';
    }
}

void SceneRenderer::upload_bone_palettes(SDL_GPUCopyPass* copy_pass) {
    if (!copy_pass || bone_palette_scratch_.empty()) {
        return;
    }

    const size_t required_size = bone_palette_scratch_.size() * sizeof(glm::mat4);
    if (!bone_palette_buffer_ || bone_palette_capacity_ < required_size) {
        // Grow geometrically with a floor of 64 KB (~16 palettes of 64 joints).
        bone_palette_capacity_ = std::max(required_size * 2, static_cast<size_t>(64 * 1024));
        bone_palette_buffer_ = gpu::GPUBuffer::create_dynamic(context_->device(), gpu::GPUBuffer::Type::Storage,
                                                              bone_palette_capacity_);
        if (!bone_palette_buffer_) {
            std::cerr << "[SceneRenderer] Failed to create bone palette storage buffer" << '\n';
            bone_palette_capacity_ = 0;
            return;
        }
    }

    bone_palette_buffer_->update(copy_pass, bone_palette_scratch_.data(), required_size);
}

void SceneRenderer::render_instanced_models(const CameraState& camera) {
    if (instance_batches_touched_.empty() || !main_render_pass_) {
        return;
    }

    SDL_GPUCommandBuffer* gpu_cmd = context_->current_command_buffer();
    if (!gpu_cmd) {
        return;
    }

    gpu::GPUPipeline* pipeline = pipeline_registry_->get_instanced_model_pipeline();
    if (!pipeline || !instance_storage_buffer_) {
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

    pipeline->bind(main_render_pass_);
    reset_model_bind_state();
    SDL_PushGPUVertexUniformData(gpu_cmd, 0, &camera_uniforms, sizeof(camera_uniforms));

    // Bind instance storage buffer
    SDL_GPUBuffer* storage_buf = instance_storage_buffer_->handle();
    SDL_BindGPUVertexStorageBuffers(main_render_pass_, 0, &storage_buf, 1);

    // Bind shadow map data
    bind_shadow_data(main_render_pass_, gpu_cmd, 1);
    bind_cluster_lights(main_render_pass_, gpu_cmd, 2);

    // Track last-pushed uniforms to skip redundant per-mesh pushes
    gpu::InstancedLightingUniforms last_pushed = {};
    bool lighting_pushed = false;

    // Cull variant currently bound (glTF doubleSided selects the no-cull one).
    bool double_sided_bound = false;

    // Draw each model type as a single instanced draw call per mesh.
    // Ascending handle order matches the packing order in upload_instance_buffers.
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
            ENGINE_ASSERT(mesh.uploaded, "Mesh not uploaded before instanced render");
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

            // glTF doubleSided: switch to the no-cull variant (and back),
            // re-establishing the pass state that follows a pipeline bind.
            if (mesh.material.double_sided != double_sided_bound) {
                gpu::GPUPipeline* variant =
                    pipeline_registry_->get_pipeline(mesh.material.double_sided
                                                         ? gpu::PipelineType::InstancedModelDoubleSided
                                                         : gpu::PipelineType::InstancedModel);
                if (!variant) {
                    continue;
                }
                variant->bind(main_render_pass_);
                reset_model_bind_state();
                SDL_PushGPUVertexUniformData(gpu_cmd, 0, &camera_uniforms, sizeof(camera_uniforms));
                SDL_BindGPUVertexStorageBuffers(main_render_pass_, 0, &storage_buf, 1);
                bind_shadow_data(main_render_pass_, gpu_cmd, 1);
                bind_cluster_lights(main_render_pass_, gpu_cmd, 2);
                lighting_pushed = false;
                double_sided_bound = mesh.material.double_sided;
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
                render_stats_.draw_calls++;
                render_stats_.triangle_count += mesh.index_count() / 3 * instance_count;
            }
            SDL_DrawGPUIndexedPrimitives(main_render_pass_, mesh.index_count(), instance_count, 0, 0, base_instance);
        }

        base_instance += instance_count;
    }
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
            ENGINE_ASSERT(mesh.uploaded, "Mesh not uploaded before instanced shadow render");
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
                render_stats_.draw_calls++;
                render_stats_.triangle_count += mesh.index_count() / 3 * instance_count;
            }
            SDL_DrawGPUIndexedPrimitives(pass, mesh.index_count(), instance_count, 0, 0, base_instance);
        }

        base_instance += instance_count;
    }
}

void SceneRenderer::render_skinned_model_command_inner(const SkinnedModelCommand& cmd, const CameraState& camera,
                                                       uint32_t palette_offset) {
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
    transform_uniforms.useSkinning = 1;
    transform_uniforms.paletteOffset = palette_offset;

    bool fog_active = frame_fog_active_;

    gpu::ModelLightingUniforms lighting_uniforms = make_model_lighting(camera, fog_active);
    lighting_uniforms.tintColor = cmd.tint;

    SDL_PushGPUVertexUniformData(gpu_cmd, 0, &transform_uniforms, sizeof(transform_uniforms));

    for (auto& mesh : model->meshes) {
        if (mesh.shadow_only) {
            continue;
        }
        ENGINE_ASSERT(mesh.uploaded, "Mesh not uploaded before skinned render");
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
            queue_transparent_mesh(mesh, model_mat, transform_uniforms.normalMatrix, cmd.tint, /*no_fog=*/false,
                                   palette_offset, *model, camera);
            continue;
        }

        // glTF doubleSided: switch to the no-cull skinned variant (and back).
        if (mesh.material.double_sided != skinned_pipeline_double_sided_ &&
            !bind_skinned_pipeline(mesh.material.double_sided)) {
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
            render_stats_.draw_calls++;
            render_stats_.triangle_count += mesh.index_count() / 3;
        }
        SDL_DrawGPUIndexedPrimitives(main_render_pass_, mesh.index_count(), 1, 0, 0, 0);
    }
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
            if (skinned && !bone_palette_buffer_) {
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
            if (skinned) {
                SDL_GPUBuffer* palette_buf = bone_palette_buffer_->handle();
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
        transform_uniforms.useSkinning = skinned ? 1 : 0;
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
            render_stats_.draw_calls++;
            render_stats_.triangle_count += mesh.index_count() / 3;
        }
        SDL_DrawGPUIndexedPrimitives(main_render_pass_, mesh.index_count(), 1, 0, 0, 0);
    }
    transparent_draws_.clear();
}

// ============================================================================
// UI Rendering
// ============================================================================

void SceneRenderer::render_ui_commands(const UIScene& ui_scene, const CameraState& camera) {
    for (const auto& cmd : ui_scene.commands()) {
        std::visit(
            [this, &camera](const auto& data) {
                using T = std::decay_t<decltype(data)>;

                if constexpr (std::is_same_v<T, FilledRectCommand>) {
                    ui_->draw_filled_rect(data.x, data.y, data.w, data.h, data.color);
                } else if constexpr (std::is_same_v<T, RectOutlineCommand>) {
                    ui_->draw_rect_outline(data.x, data.y, data.w, data.h, data.color, data.line_width);
                } else if constexpr (std::is_same_v<T, CircleCommand>) {
                    ui_->draw_circle(data.x, data.y, data.radius, data.color, data.segments);
                } else if constexpr (std::is_same_v<T, CircleOutlineCommand>) {
                    ui_->draw_circle_outline(data.x, data.y, data.radius, data.color, data.line_width, data.segments);
                } else if constexpr (std::is_same_v<T, LineCommand>) {
                    ui_->draw_line(data.x1, data.y1, data.x2, data.y2, data.color, data.line_width);
                } else if constexpr (std::is_same_v<T, TextCommand>) {
                    ui_->draw_text(data.text, data.x, data.y, data.color, data.scale);
                } else if constexpr (std::is_same_v<T, ButtonCommand>) {
                    ui_->draw_button(data.x, data.y, data.w, data.h, data.label, data.color, data.selected);
                }
            },
            cmd.data);
    }
}

void SceneRenderer::draw_billboard_3d(const Billboard3DCommand& cmd, const CameraState& camera) {
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
void SceneRenderer::upload_debug_lines(const RenderScene& scene, SDL_GPUCopyPass* copy_pass) {
    debug_line_vertex_count_ = 0;
    const auto& lines = scene.debug_lines();
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

    // Build vertex data: position(3) + color(4) per vertex
    // Unpack RGBA packed uint32 to float4. Reuse scratch buffer to avoid per-frame alloc.
    debug_line_scratch_.resize(vertex_count * 7);
    float* dst = debug_line_scratch_.data();
    for (const auto& line : lines) {
        float r = static_cast<float>((line.color >> 24) & 0xFF) / 255.0f;
        float g = static_cast<float>((line.color >> 16) & 0xFF) / 255.0f;
        float b = static_cast<float>((line.color >> 8) & 0xFF) / 255.0f;
        float a = static_cast<float>((line.color >> 0) & 0xFF) / 255.0f;

        // Start vertex
        *dst++ = line.start.x;
        *dst++ = line.start.y;
        *dst++ = line.start.z;
        *dst++ = r;
        *dst++ = g;
        *dst++ = b;
        *dst++ = a;

        // End vertex
        *dst++ = line.end.x;
        *dst++ = line.end.y;
        *dst++ = line.end.z;
        *dst++ = r;
        *dst++ = g;
        *dst++ = b;
        *dst++ = a;
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

void SceneRenderer::render_shadow_passes(const RenderScene& scene, const CameraState& camera) {
    SDL_GPUCommandBuffer* cmd = context_->current_command_buffer();
    if (!cmd) {
        return;
    }

    ENGINE_PROFILE_ZONE_COLOR("ShadowPass", 0xE67E22);
    ENGINE_GPU_SCOPE(cmd, "ShadowPass");
    ENGINE_GPU_TIMER(gpu_timer_pool_, cmd, "ShadowPass");

    // Note: shadow_map_->update() was already called in render_frame() so that the
    // per-cascade light frustums could be used during build_instance_batches.

    // Pre-resolve Model* + world bounds once per skinned entity (outside the cascade loop).
    // Avoids N cascades * M entities hash lookups per frame. Palette offsets
    // come from pack_bone_palettes() (parallel to skinned_commands order);
    // entities without a palette (no skeleton / dropped over cap) are skipped.
    auto& skinned_scratch = shadow_skinned_scratch_;
    skinned_scratch.clear();
    const auto& skinned_cmds = scene.skinned_commands();
    skinned_scratch.reserve(skinned_cmds.size());
    for (size_t i = 0; i < skinned_cmds.size(); ++i) {
        const auto& skinned_cmd = skinned_cmds[i];
        const uint32_t palette_offset =
            i < skinned_palette_offsets_.size() ? skinned_palette_offsets_[i] : INVALID_PALETTE_OFFSET;
        if (palette_offset == INVALID_PALETTE_OFFSET) {
            continue;
        }
        Model* model = model_manager_->get_model(skinned_cmd.model_handle);
        if (!model) {
            continue;
        }
        const glm::mat4& t = skinned_cmd.transform;
        glm::vec3 wc = glm::vec3(t * glm::vec4(model->bounding_center, 1.0f));
        float wr = model->bounding_half_diag * max_scale_factor(t);
        skinned_scratch.push_back({&skinned_cmd, model, wc, wr, palette_offset});
    }

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

    for (int cascade = 0; cascade < shadow_map_->active_cascades(); ++cascade) {
        const char* cascade_name = cascade_label(cascade);
        ENGINE_GPU_SCOPE(cmd, cascade_name);
        ENGINE_GPU_TIMER(gpu_timer_pool_, cmd, cascade_name);
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
                        ENGINE_ASSERT(mesh.uploaded, "Mesh not uploaded before non-instanced shadow render");
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
                            render_stats_.draw_calls++;
                            render_stats_.triangle_count += mesh.index_count() / 3;
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
        if (!skinned_scratch.empty() && bone_palette_buffer_) {
            auto* shadow_skinned_pipeline = pipeline_registry_->get_shadow_skinned_model_pipeline();
            if (shadow_skinned_pipeline) {
                shadow_skinned_pipeline->bind(shadow_pass);
                SDL_GPUBuffer* palette_buf = bone_palette_buffer_->handle();
                SDL_BindGPUVertexStorageBuffers(shadow_pass, 0, &palette_buf, 1);

                for (const auto& rs : skinned_scratch) {
                    // Per-cascade cull: skip if not in this cascade's light frustum.
                    if (!shadow_cascade_frustums_[cascade].intersects_sphere(rs.world_center, rs.world_radius)) {
                        continue;
                    }

                    const SkinnedModelCommand* data = rs.cmd;
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
                        ENGINE_ASSERT(mesh.uploaded, "Mesh not uploaded before skinned shadow render");
                        if (!mesh.uploaded) {
                            continue;
                        }
                        if (!mesh.vertex_buffer || !mesh.index_buffer || mesh.index_count() == 0) {
                            continue;
                        }

                        mesh.bind_buffers(shadow_pass);
                        if (collect_stats_) {
                            render_stats_.draw_calls++;
                            render_stats_.triangle_count += mesh.index_count() / 3;
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

    shadow_map_->end_shadow_pass();
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
    SDL_GPUBuffer* bufs[3] = {
        cluster_grid_.light_data_buffer(),
        cluster_grid_.cluster_offsets_buffer(),
        cluster_grid_.light_indices_buffer(),
    };
    if (!bufs[0] || !bufs[1] || !bufs[2]) {
        return;
    }
    SDL_BindGPUFragmentStorageBuffers(pass, 0, bufs, 3);
    // frame_cluster_params_ (not cluster_grid_.params()) so frames with zero
    // dynamic lights publish gridDim.w == 0 rather than stale counts.
    SDL_PushGPUFragmentUniformData(cmd, uniform_slot, &frame_cluster_params_, sizeof(frame_cluster_params_));
}

} // namespace mmo::engine::scene
