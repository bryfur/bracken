// SceneRenderer: the main pass — everything drawn into the scene colour
// target, in the order render_3d_scene records it.
//
// Terrain and grass (delegated to their sub-renderers), instanced statics,
// non-instanced statics, skinned models, the skybox, the sorted transparent
// (glTF alphaMode BLEND) queue, particle effects and debug lines; plus the
// model-family pipeline binds and the lazy material/lighting bind state they
// invalidate, and the 2D UI / billboard drawing that runs after the main pass.
//
// The pipeline-variant flags this file reads (model_pipeline_double_sided_ and
// friends) are written by binds in this same file, but their correctness rests
// on an ENTRY-ORDER dependency documented at their declaration in
// scene_renderer.hpp: render_3d_scene binds unconditionally before the first
// draw of every frame. That is asserted here rather than assumed.

#include "scene_renderer.hpp"
#include "scene_renderer_internal.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/color.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/effects/effect_renderer.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render/grass_renderer.hpp"
#include "bracken/render/ibl.hpp"
#include "bracken/render/lighting/local_shadows.hpp"
#include "bracken/render/render_context.hpp"
#include "bracken/render/shadow_map.hpp"
#include "bracken/render/terrain_renderer.hpp"
#include "bracken/render/ui_renderer.hpp"
#include "bracken/render/world_renderer.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/scene_prep.hpp"
#include "bracken/scene/transform_math.hpp"

#include <SDL3/SDL_gpu.h>
#include <algorithm>
#include <cstring>
#include <glm/glm.hpp>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace bracken::scene {

namespace gpu = bracken::gpu;
namespace render = bracken::render;

void SceneRenderer::render_3d_scene(const FramePacket& frame, const CameraState& camera) {
    BRACKEN_PROFILE_ZONE_COLOR("SceneRenderer::render_3d_scene", 0xC0392B);
    const GraphicsSettings& gfx = graphics_ ? *graphics_ : default_graphics_;
    SDL_GPUCommandBuffer* cmd = context_->current_command_buffer();

    // BLEND meshes are queued by the opaque walks below and drawn sorted after
    // the skybox; drop anything stale from an aborted previous pass.
    transparent_draws_.clear();

    // The model/skinned pipeline-variant flags describe the last bind of THIS
    // pass only (survey S16 — see their declaration). This is the pass entry
    // point, so nothing is bound yet; the walks below bind unconditionally
    // before their first draw, and the readers assert it.
    model_pipeline_bound_ = false;
    skinned_pipeline_bound_ = false;

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

    SDL_GPUTexture* rt_shadow_tex = frame_rt_shadow_texture_
                                        ? frame_rt_shadow_texture_
                                        : (dummy_white_texture_ ? dummy_white_texture_->handle() : nullptr);
    SDL_GPUTextureSamplerBinding rt_shadow_binding = {rt_shadow_tex, default_sampler_};
    const bool have_rt_shadow_binding = rt_shadow_tex && default_sampler_;

    SDL_GPUTexture* rt_local_tex = frame_rt_local_shadow_texture_
                                       ? frame_rt_local_shadow_texture_
                                       : (dummy_white_texture_ ? dummy_white_texture_->handle() : nullptr);
    SDL_GPUTextureSamplerBinding rt_local_binding = {rt_local_tex, default_sampler_};
    const bool have_rt_local_binding = rt_local_tex && default_sampler_;

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
                         have_rt_shadow_binding ? &rt_shadow_binding : nullptr, frame_rt_shadows_active_,
                         have_rt_local_binding ? &rt_local_binding : nullptr, frame_rt_local_shadows_active_,
                         have_local_shadow_binding ? &local_shadow_binding : nullptr, &frame_gi_);
    }

    if (frame.should_draw_grass() && gfx.grass_enabled && grass_renderer_) {
        // view_distance + chunk upload already performed before render pass began.
        SDL_PushGPUFragmentUniformData(cmd, 1, &frame_shadow_uniforms_, sizeof(frame_shadow_uniforms_));
        grass_renderer_->render(main_render_pass_, cmd, camera.view, camera.projection, camera.position, light_dir_,
                                shadow_binding_count > 0 ? shadow_bindings : nullptr, shadow_binding_count,
                                have_ao_binding ? &ao_binding : nullptr, ao_texel, frame_prepass_active_,
                                have_ibl_bindings ? ibl_bindings : nullptr, frame_ibl_active_,
                                have_rt_shadow_binding ? &rt_shadow_binding : nullptr, frame_rt_shadows_active_,
                                have_rt_local_binding ? &rt_local_binding : nullptr, frame_rt_local_shadows_active_,
                                have_local_shadow_binding ? &local_shadow_binding : nullptr, &frame_gi_);
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
    bind_rt_shadow_mask(10);
    bind_gi(11);
    bind_rt_local_shadow_mask(14);
    bind_local_shadow_atlas(15);
    model_pipeline_double_sided_ = double_sided;
    model_pipeline_depth_equal_ = depth_equal;
    model_pipeline_bound_ = true;
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
    bind_rt_shadow_mask(10);
    bind_gi(11);
    bind_rt_local_shadow_mask(14);
    bind_local_shadow_atlas(15);
    SDL_GPUBuffer* palette_buf = current_bone_palette_buffer()->handle();
    SDL_BindGPUVertexStorageBuffers(main_render_pass_, 0, &palette_buf, 1);
    skinned_pipeline_double_sided_ = double_sided;
    skinned_pipeline_depth_equal_ = depth_equal;
    skinned_pipeline_bound_ = true;
    return true;
}

void SceneRenderer::reset_model_bind_state() {
    last_frag_tex_.fill(nullptr);
    model_lighting_valid_ = false;
}

void SceneRenderer::bind_rt_shadow_mask(uint32_t slot) {
    // Always bound, even with ray tracing off: model.frag / model_instanced
    // .frag declare the slot unconditionally, and SDL3 GPU leaves an unbound
    // fragment sampler descriptor undefined. The white dummy reads 1.0 from
    // any UV, which is "fully lit" — and rtShadowEnabled is 0 anyway, so it is
    // never actually sampled.
    SDL_GPUTexture* white = dummy_white_texture_ ? dummy_white_texture_->handle() : nullptr;
    SDL_GPUTexture* tex = frame_rt_shadow_texture_ ? frame_rt_shadow_texture_ : white;
    if (!tex || !default_sampler_ || !main_render_pass_) {
        return;
    }
    SDL_GPUTextureSamplerBinding binding = {tex, default_sampler_};
    SDL_BindGPUFragmentSamplers(main_render_pass_, slot, &binding, 1);
}

void SceneRenderer::bind_rt_local_shadow_mask(uint32_t slot) {
    // Always bound, for the same reason bind_rt_shadow_mask always is: the
    // slot is declared unconditionally and SDL3 GPU leaves an unbound fragment
    // sampler descriptor undefined. The white dummy reads 1.0 from any UV,
    // which for a visibility RATIO is "nothing is shadowed" — and
    // rtLocalShadowEnabled is 0 anyway, so it is never sampled.
    SDL_GPUTexture* white = dummy_white_texture_ ? dummy_white_texture_->handle() : nullptr;
    SDL_GPUTexture* tex = frame_rt_local_shadow_texture_ ? frame_rt_local_shadow_texture_ : white;
    if (!tex || !default_sampler_ || !main_render_pass_) {
        return;
    }
    SDL_GPUTextureSamplerBinding binding = {tex, default_sampler_};
    SDL_BindGPUFragmentSamplers(main_render_pass_, slot, &binding, 1);
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

void SceneRenderer::refresh_gi_bindings() {
    frame_gi_ = render::gi::GiBindings{};

    // The dummy set. Black for the two atlases and the metadata plane alike:
    // with `enabled == 0` nothing samples them, and a black metadata texel
    // reads as GI_CLASS_OK rather than as "skip", which is deliberately the
    // LESS defensive choice — if a shader ever taps this while disabled, it
    // should produce visibly black GI rather than silently fall back to the
    // hemisphere and hide the bug.
    SDL_GPUTexture* black = dummy_black_texture_ ? dummy_black_texture_->handle() : nullptr;
    if (black != nullptr && default_sampler_ != nullptr) {
        for (SDL_GPUTextureSamplerBinding& b : frame_gi_.textures) {
            b = {black, default_sampler_};
        }
    }

    if (!probe_volume_ || !probe_volume_->is_ready()) {
        return;
    }
    SDL_GPUSampler* linear = probe_volume_->linear_sampler();
    SDL_GPUSampler* point = probe_volume_->point_sampler();
    if (linear == nullptr || point == nullptr) {
        return;
    }
    // Slot order IS the binding contract — the three are bound as one run and
    // gi_sample.hlsli declares them in exactly this order.
    frame_gi_.textures[0] = {probe_volume_->irradiance()->handle(), linear};
    frame_gi_.textures[1] = {probe_volume_->visibility()->handle(), linear};
    frame_gi_.textures[2] = {probe_volume_->probe_offset()->handle(), point};

    const std::array<glm::ivec3, render::gi::CASCADE_COUNT> bases = probe_volume_->cascade_bases();
    for (std::size_t c = 0; c < bases.size(); ++c) {
        frame_gi_.cascade_base[c] = glm::ivec4(bases[c], 0);
    }
    // Only now: a volume that exists but has never been placed has no window to
    // address probes against, and every cascade would answer for the origin.
    frame_gi_.enabled = probe_volume_->cascade(0).placed() ? 1.0f : 0.0f;

    // R4b's screen-probe tier rides on top. `stats().active` is the gather's own
    // statement that it ran all six passes THIS frame — not that the object
    // exists — so a frame that lost its prepass, its G-buffer or a pipeline
    // leaves giScreenEnabled at 0 and the field answers alone.
    if (screen_probes_ && screen_probes_->is_ready() && screen_probes_->stats().active &&
        screen_probes_->sh_buffer() != nullptr) {
        frame_gi_.screen_sh = screen_probes_->sh_buffer();
        const glm::ivec2 grid = screen_probes_->grid();
        frame_gi_.screen_probes = glm::vec2(static_cast<float>(grid.x), static_cast<float>(grid.y));
        frame_gi_.screen_enabled = frame_gi_.enabled;
    }
}

void SceneRenderer::bind_gi(uint32_t base_slot) {
    if (!main_render_pass_ || frame_gi_.textures[0].texture == nullptr) {
        return;
    }
    SDL_BindGPUFragmentSamplers(main_render_pass_, base_slot, frame_gi_.textures.data(), 3);
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
    bind_material_texture(0, mesh.has_albedo()
                                 ? mesh.albedo_handle()
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
    // Ray-traced sun shadows: sampled from rtShadowTexture (fragment slot 10,
    // bound by bind_rt_shadow_mask) with the same screenAOTexel* UV scale. The
    // mask is at the traced pass's own resolution, so the sampler does the
    // magnification; screenAOTexel* only converts SV_Position to [0,1].
    u.rtShadowEnabled = frame_rt_shadows_active_ ? 1.0f : 0.0f;
    u.rtLocalShadowEnabled = frame_rt_local_shadows_active_ ? 1.0f : 0.0f;
    u.rtReflectionMaxRoughness = frame_rt_reflection_max_roughness_;
    u.rtReflectionFadeStart = frame_rt_reflection_fade_start_;
    // Probe-field GI: the frame's ONE snapshot of the field (refresh_gi_bindings).
    // giEnabled == 0 leaves every ambient branch on its pre-R1c arithmetic.
    for (std::size_t c = 0; c < frame_gi_.cascade_base.size(); ++c) {
        u.giCascadeBase[c] = frame_gi_.cascade_base[c];
    }
    u.giEnabled = frame_gi_.enabled;
    u.giScreenEnabled = frame_gi_.screen_enabled;
    u.giScreenProbesX = frame_gi_.screen_probes.x;
    u.giScreenProbesY = frame_gi_.screen_probes.y;

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

    const bool no_fog = has(cmd.flags, DrawFlags::NoFog);
    bool fog_active = !no_fog && frame_fog_active_;

    gpu::ModelLightingUniforms lighting_uniforms = make_model_lighting(camera, fog_active);
    lighting_uniforms.tintColor = cmd.tint;

    SDL_PushGPUVertexUniformData(gpu_cmd, 0, &transform_uniforms, sizeof(transform_uniforms));

    // The variant comparison below is only meaningful if a model-family bind
    // happened in this pass (survey S16, entry-order dependency documented at
    // the flags' declaration). Every caller binds first; assert it rather
    // than trust a comment in another translation unit.
    BRACKEN_ASSERT(model_pipeline_bound_,
                   "render_model_command_inner: no model pipeline bound in this pass — the cull/depth variant "
                   "flags describe an earlier pass's bind");

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
            queue_transparent_mesh(mesh, model_mat, transform_uniforms.normalMatrix, cmd.tint, no_fog,
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

        apply_mesh_material(lighting_uniforms, mesh, frame_rt_reflection_max_roughness_,
                            frame_rt_reflection_fade_start_);
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
    stats_partial.merge_into(render_stats_, frame_diagnostics_);
}

void SceneRenderer::render_instanced_models(const CameraState& camera) {
    if (instance_batches_touched_.empty() || !main_render_pass_) {
        return;
    }

    SDL_GPUCommandBuffer* gpu_cmd = context_->current_command_buffer();
    if (!gpu_cmd) {
        return;
    }

    if (!pipeline_registry_->get_pipeline(gpu::PipelineType::InstancedModel) || !instance_storage_buffer_) {
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
    lighting_uniforms.rtShadowEnabled = frame_rt_shadows_active_ ? 1.0f : 0.0f;
    lighting_uniforms.rtLocalShadowEnabled = frame_rt_local_shadows_active_ ? 1.0f : 0.0f;
    lighting_uniforms.rtReflectionMaxRoughness = frame_rt_reflection_max_roughness_;
    lighting_uniforms.rtReflectionFadeStart = frame_rt_reflection_fade_start_;
    for (std::size_t c = 0; c < frame_gi_.cascade_base.size(); ++c) {
        lighting_uniforms.giCascadeBase[c] = frame_gi_.cascade_base[c];
    }
    lighting_uniforms.giEnabled = frame_gi_.enabled;
    lighting_uniforms.giScreenEnabled = frame_gi_.screen_enabled;
    lighting_uniforms.giScreenProbesX = frame_gi_.screen_probes.x;
    lighting_uniforms.giScreenProbesY = frame_gi_.screen_probes.y;

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
        bind_rt_shadow_mask(10);
        bind_gi(11);
        bind_rt_local_shadow_mask(14);
        bind_local_shadow_atlas(15);
        lighting_pushed = false;
        bound_type = want;
        return true;
    };

    // Draw each model type as a single instanced draw call per mesh, walking
    // the bins in the one packing order (for_each_drawable_instance_batch).
    // Draw stats accumulate into a section-local partial, merged once after
    // the walk (per-chunk merge discipline — see FrameStatsPartial).
    FrameStatsPartial stats_partial;
    (void)for_each_drawable_instance_batch(
        instance_batches_, 0,
        [&](Model& model, const std::vector<gpu::InstanceData>& instances, uint32_t base_instance) {
            const uint32_t instance_count = static_cast<uint32_t>(instances.size());

            for (auto& mesh : model.meshes) {
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
                                               INVALID_PALETTE_OFFSET, model, camera);
                    }
                    continue;
                }

                const bool wants_equal = frame_prepass_active_ && mesh.material.alpha_mode == AlphaMode::Opaque;
                if (!ensure_pipeline(select_type(mesh.material.double_sided, wants_equal))) {
                    continue;
                }

                apply_mesh_material(lighting_uniforms, mesh, frame_rt_reflection_max_roughness_,
                                    frame_rt_reflection_fade_start_);
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
                SDL_DrawGPUIndexedPrimitives(main_render_pass_, mesh.index_count(), instance_count, 0, 0,
                                             base_instance);
            }
        });
    stats_partial.merge_into(render_stats_, frame_diagnostics_);
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

    const bool no_fog = has(cmd.flags, DrawFlags::NoFog);
    bool fog_active = !no_fog && frame_fog_active_;

    gpu::ModelLightingUniforms lighting_uniforms = make_model_lighting(camera, fog_active);
    lighting_uniforms.tintColor = cmd.tint;

    SDL_PushGPUVertexUniformData(gpu_cmd, 0, &transform_uniforms, sizeof(transform_uniforms));

    // Same entry-order dependency as the static path (survey S16): the
    // skinned variant flags describe the last bind of THIS pass, and
    // render_3d_scene binds before the first skinned draw.
    BRACKEN_ASSERT(skinned_pipeline_bound_,
                   "render_skinned_model_command_inner: no skinned pipeline bound in this pass — the cull/depth "
                   "variant flags describe an earlier pass's bind");

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
            queue_transparent_mesh(mesh, model_mat, transform_uniforms.normalMatrix, cmd.tint, no_fog,
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

        apply_mesh_material(lighting_uniforms, mesh, frame_rt_reflection_max_roughness_,
                            frame_rt_reflection_fade_start_);
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
    stats_partial.merge_into(render_stats_, frame_diagnostics_);
}

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
            bind_rt_shadow_mask(10);
            bind_gi(11);
            bind_rt_local_shadow_mask(14);
            bind_local_shadow_atlas(15);
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
        apply_mesh_material(lighting_uniforms, mesh, frame_rt_reflection_max_roughness_,
                            frame_rt_reflection_fade_start_);
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
    stats_partial.merge_into(render_stats_, frame_diagnostics_);
    transparent_draws_.clear();
}

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
        const glm::vec4 c = line.color.to_linear();

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

    gpu::GPUPipeline* pipeline = pipeline_registry_->get_pipeline(gpu::PipelineType::Grid);
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

} // namespace bracken::scene
