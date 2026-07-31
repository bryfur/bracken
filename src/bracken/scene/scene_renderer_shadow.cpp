// SceneRenderer: shadow rendering — the directional CSM cascade passes, the
// budgeted local (point/spot) shadow tiles, and the two binds that publish
// their results to the main pass.
//
// render_instanced_shadow_models travels with these passes rather than with
// the forward draw walk: it is the third of the three walks that index the
// packed instance buffer, and the only one seeded with a per-cascade section
// offset (shadow_cascade_base_instance_). The order it walks in is defined
// once, by SceneRenderer::for_each_instance_batch.

#include "scene_renderer.hpp"
#include "scene_renderer_internal.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_debug.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render/lighting/local_shadows.hpp"
#include "bracken/render/render_context.hpp"
#include "bracken/render/shadow_map.hpp"
#include "bracken/render/terrain_renderer.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/frustum.hpp"
#include "bracken/scene/scene_prep.hpp"
#include "bracken/scene/transform_math.hpp"

#include <SDL3/SDL_gpu.h>
#include <limits>
#include <vector>

namespace bracken::scene {

namespace gpu = bracken::gpu;
namespace render = bracken::render;

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

    // Same walk as the forward/prepass paths, seeded with this cascade's
    // section offset (for_each_drawable_instance_batch). Draw stats
    // accumulate into a section-local partial, merged once after the walk
    // (per-chunk merge discipline — see FrameStatsPartial).
    FrameStatsPartial stats_partial;
    (void)for_each_drawable_instance_batch(
        cascade_batches, shadow_cascade_base_instance_[cascade],
        [&](Model& model, const std::vector<gpu::ShadowInstanceData>& instances, uint32_t base_instance) {
            const uint32_t instance_count = static_cast<uint32_t>(instances.size());

            for (auto& mesh : model.meshes) {
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
        });
    stats_partial.merge_into(render_stats_, frame_diagnostics_);
}

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
    stats_partial.merge_into(render_stats_, frame_diagnostics_);

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
    stats_partial.merge_into(render_stats_, frame_diagnostics_);

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
