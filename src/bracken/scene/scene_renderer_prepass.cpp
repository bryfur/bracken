// SceneRenderer: the depth prepass — velocity (RG16F) + scene depth for every
// glTF alphaMode OPAQUE draw, recorded before the forward pass.
//
// Covers terrain, static models, instanced models, skinned models and grass.
// MASK joins through the alpha-tested variants (so foliage participates in
// GTAO and motion vectors); BLEND is excluded and keeps its forward
// GREATER_OR_EQUAL depth behaviour. Vertex work is position-only (+ skinning);
// the shared fragment shader writes UV-space velocity (current - previous,
// UNJITTERED matrices) while depth writes lay down the depth the forward
// opaques then test EQUAL against.

#include "scene_renderer.hpp"
#include "scene_renderer_internal.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_debug.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render/grass_renderer.hpp"
#include "bracken/render/post_process_stack.hpp"
#include "bracken/render/render_context.hpp"
#include "bracken/render/terrain_renderer.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/scene_prep.hpp"
#include "bracken/scene/transform_math.hpp"

#include <SDL3/SDL_gpu.h>
#include <vector>

namespace bracken::scene {

namespace gpu = bracken::gpu;
namespace render = bracken::render;

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
        BRACKEN_LOG_ERROR(LOG_TAG, "render_depth_prepass: failed to begin render pass");
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

    // Same packing order as upload_instance_buffers / the forward walk
    // (for_each_drawable_instance_batch). Draw stats accumulate into a
    // section-local partial, merged once after the walk (per-chunk merge
    // discipline — see FrameStatsPartial).
    FrameStatsPartial stats_partial;
    (void)for_each_drawable_instance_batch(
        instance_batches_, 0,
        [&](Model& model, const std::vector<gpu::InstanceData>& instances, uint32_t base_instance) {
            const uint32_t instance_count = static_cast<uint32_t>(instances.size());

            for (auto& mesh : model.meshes) {
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
                        mesh.has_albedo() ? mesh.albedo_handle()
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
        });
    stats_partial.merge_into(render_stats_, frame_diagnostics_);
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
                    mesh.has_albedo() ? mesh.albedo_handle()
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
    stats_partial.merge_into(render_stats_, frame_diagnostics_);
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
    stats_partial.merge_into(render_stats_, frame_diagnostics_);
}

} // namespace bracken::scene
