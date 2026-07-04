#include "effect_renderer.hpp"
#include "../gpu/gpu_texture.hpp"
#include "../gpu/gpu_uniforms.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/profiler.hpp"
#include "engine/gpu/gpu_device.hpp"
#include "engine/gpu/pipeline_registry.hpp"
#include "engine/model_loader.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"
#include "SDL3/SDL_error.h"
#include "SDL3/SDL_gpu.h"
#include <cstdint>

namespace mmo::engine::render {

namespace gpu = mmo::engine::gpu;

namespace {

/// Unlit additive HDR particles: emitted RGB is pushed above the bloom
/// threshold so spell effects glow. Opacity fades by scaling RGB (additive
/// blend ignores alpha), so particles dissolve into the scene smoothly.
constexpr float PARTICLE_EMISSIVE_BOOST = 2.0f;

glm::mat4 build_particle_transform(const mmo::engine::systems::Particle& particle, const Model& model) {
    glm::mat4 model_mat = glm::mat4(1.0f);

    // Position
    model_mat = glm::translate(model_mat, particle.position);

    // Rotation (euler angles in radians)
    model_mat = glm::rotate(model_mat, particle.rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
    model_mat = glm::rotate(model_mat, particle.rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
    model_mat = glm::rotate(model_mat, particle.rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));

    // Scale
    model_mat = glm::scale(model_mat, glm::vec3(particle.scale));

    // Center the model at origin
    float cx = (model.min_x + model.max_x) / 2.0f;
    float cy = (model.min_y + model.max_y) / 2.0f;
    float cz = (model.min_z + model.max_z) / 2.0f;
    return model_mat * glm::translate(glm::mat4(1.0f), glm::vec3(-cx, -cy, -cz));
}

} // anonymous namespace

EffectRenderer::EffectRenderer() = default;

EffectRenderer::~EffectRenderer() {
    shutdown();
}

bool EffectRenderer::init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry,
                          ModelManager* model_manager) {
    device_ = &device;
    pipeline_registry_ = &pipeline_registry;
    model_manager_ = model_manager;

    // Create sampler for effect textures
    SDL_GPUSamplerCreateInfo sampler_info = {};
    sampler_info.min_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mag_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
    sampler_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sampler_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sampler_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sampler_info.max_anisotropy = 4.0f;
    sampler_info.enable_anisotropy = true;
    // Keep mip chains reachable (max_lod = 0 clamps sampling to mip 0).
    sampler_info.max_lod = 1000.0f;
    sampler_ = SDL_CreateGPUSampler(device_->handle(), &sampler_info);
    if (!sampler_) {
        ENGINE_LOG_ERROR("effect", "Failed to create sampler: {}", SDL_GetError());
        return false;
    }

    // 1x1 white fallback for the base-color slot so particle bindings are
    // fully self-contained (no reliance on textures left by earlier passes).
    uint32_t white_pixel = 0xFFFFFFFF;
    dummy_white_texture_ = gpu::GPUTexture::create_2d(device, 1, 1, gpu::TextureFormat::RGBA8, &white_pixel);
    if (!dummy_white_texture_) {
        ENGINE_LOG_ERROR("effect", "Failed to create dummy white texture");
        return false;
    }

    return true;
}

void EffectRenderer::shutdown() {
    if (sampler_ && device_) {
        SDL_ReleaseGPUSampler(device_->handle(), sampler_);
        sampler_ = nullptr;
    }
    dummy_white_texture_.reset();
    emitter_model_cache_.clear();

    device_ = nullptr;
    pipeline_registry_ = nullptr;
    model_manager_ = nullptr;
}

ModelHandle EffectRenderer::resolve_emitter_model(const mmo::engine::EmitterDefinition& def) {
    auto it = emitter_model_cache_.find(&def);
    if (it != emitter_model_cache_.end()) {
        return it->second;
    }
    ModelHandle handle = INVALID_MODEL_HANDLE;
    if (model_manager_ && !def.model.empty()) {
        handle = model_manager_->get_handle(def.model);
        if (handle == INVALID_MODEL_HANDLE) {
            ENGINE_LOG_WARN("effect", "Effect emitter '{}' references unknown model '{}'", def.name, def.model);
        }
    }
    emitter_model_cache_.emplace(&def, handle);
    return handle;
}

void EffectRenderer::ensure_models_uploaded(const mmo::engine::systems::EffectSystem& effect_system) {
    ENGINE_PROFILE_ZONE_COLOR("EffectRenderer::ensure_models_uploaded", 0x3D6E9E);
    if (!device_ || !model_manager_) {
        return;
    }

    for (const auto& effect : effect_system.get_effects()) {
        for (const auto& emitter : effect.emitters) {
            if (!emitter.definition) {
                continue;
            }
            ModelHandle handle = resolve_emitter_model(*emitter.definition);
            if (handle == INVALID_MODEL_HANDLE) {
                continue;
            }
            Model* model = model_manager_->get_model(handle);
            if (model && !model->meshes.empty() && !model->meshes[0].uploaded) {
                ModelLoader::upload_to_gpu(*device_, *model);
            }
        }
    }
}

void EffectRenderer::draw_particle_effects(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                                           const mmo::engine::systems::EffectSystem& effect_system,
                                           const glm::mat4& view, const glm::mat4& projection,
                                           const glm::vec3& camera_pos) {
    ENGINE_PROFILE_ZONE_COLOR("EffectRenderer::draw_particle_effects", 0xC0392B);
    if (!pass || !cmd || !device_ || !pipeline_registry_ || !model_manager_) {
        return;
    }
    if (effect_system.particle_count() == 0) {
        return;
    }

    auto* pipeline = pipeline_registry_->get_pipeline(gpu::PipelineType::Particle);
    if (!pipeline) {
        return;
    }
    pipeline->bind(pass);

    // Shared per-particle vertex uniforms; only `model` changes per draw.
    gpu::ModelTransformUniforms vertex_uniforms = {};
    vertex_uniforms.view = view;
    vertex_uniforms.projection = projection;
    vertex_uniforms.cameraPos = camera_pos;
    vertex_uniforms.normalMatrix = glm::mat4(1.0f);

    SDL_GPUTexture* last_bound_tex = nullptr;

    // Particles within one emitter all share the emitter's model, so iterating
    // emitter-by-emitter groups draws naturally: buffers/texture bind once per
    // (emitter, mesh), then each particle is a uniform push + draw.
    for (const auto& effect : effect_system.get_effects()) {
        for (const auto& emitter : effect.emitters) {
            if (!emitter.definition || emitter.particles.empty()) {
                continue;
            }

            ModelHandle handle = resolve_emitter_model(*emitter.definition);
            Model* model = (handle != INVALID_MODEL_HANDLE) ? model_manager_->get_model(handle) : nullptr;
            if (!model) {
                continue;
            }

            // The pre-pass upload (ensure_models_uploaded) should have handled
            // this; never upload mid-render-pass — skip and log instead.
            if (!model->meshes.empty() && !model->meshes[0].uploaded) {
                ENGINE_LOG_WARN("effect", "Effect model '{}' not GPU-uploaded; skipping particles this frame",
                                emitter.definition->model);
                continue;
            }

            for (const auto& mesh : model->meshes) {
                if (!mesh.uploaded || !mesh.vertex_buffer || !mesh.index_buffer || mesh.index_count() == 0) {
                    continue;
                }

                // Bind buffers + texture once for every particle of this mesh.
                SDL_GPUBufferBinding vb_binding = {mesh.vertex_buffer->handle(), 0};
                SDL_BindGPUVertexBuffers(pass, 0, &vb_binding, 1);
                SDL_GPUBufferBinding ib_binding = {mesh.index_buffer->handle(), 0};
                SDL_BindGPUIndexBuffer(pass, &ib_binding, SDL_GPU_INDEXELEMENTSIZE_32BIT);

                SDL_GPUTexture* tex =
                    (mesh.has_texture && mesh.texture) ? mesh.texture->handle() : dummy_white_texture_->handle();
                if (tex != last_bound_tex) {
                    SDL_GPUTextureSamplerBinding tex_binding = {tex, sampler_};
                    SDL_BindGPUFragmentSamplers(pass, 0, &tex_binding, 1);
                    last_bound_tex = tex;
                }

                for (const auto& particle : emitter.particles) {
                    vertex_uniforms.model = build_particle_transform(particle, *model);
                    SDL_PushGPUVertexUniformData(cmd, 0, &vertex_uniforms, sizeof(vertex_uniforms));

                    gpu::ParticleFragmentUniforms frag_uniforms = {};
                    frag_uniforms.tintColor = particle.color;
                    frag_uniforms.emissiveBoost = PARTICLE_EMISSIVE_BOOST;
                    frag_uniforms.opacity = particle.opacity;
                    SDL_PushGPUFragmentUniformData(cmd, 0, &frag_uniforms, sizeof(frag_uniforms));

                    SDL_DrawGPUIndexedPrimitives(pass, mesh.index_count(), 1, 0, 0, 0);
                }
            }
        }
    }
}

} // namespace mmo::engine::render
