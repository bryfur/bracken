#include "engine/render/effect_renderer.hpp"
#include "engine/gpu/gpu_texture.hpp"
#include "engine/gpu/gpu_uniforms.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/profiler.hpp"
#include "engine/gpu/gpu_device.hpp"
#include "engine/gpu/pipeline_registry.hpp"
#include "engine/model_loader.hpp"
#include "glm/ext/matrix_float3x3.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"
#include "SDL3/SDL_error.h"
#include "SDL3/SDL_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace mmo::engine::render {

namespace gpu = mmo::engine::gpu;

namespace {

/// Unlit additive HDR particles: emitted RGB is pushed above the bloom
/// threshold so spell effects glow. Opacity fades by scaling RGB (additive
/// blend ignores alpha), so particles dissolve into the scene smoothly.
constexpr float PARTICLE_EMISSIVE_BOOST = 2.0f;

} // anonymous namespace

glm::mat4 build_particle_transform(const mmo::engine::systems::Particle& particle, const glm::vec3& center_offset) {
    // Equivalent to translate(position) * rotate(Y) * rotate(X) * rotate(Z) *
    // scale(s) * translate(-center_offset), composed directly: one sin/cos
    // pair per axis instead of three chained 4x4 rotate multiplies
    // (effect_renderer.test.cpp pins equivalence with the glm chain).
    const float sx = std::sin(particle.rotation.x), cx = std::cos(particle.rotation.x);
    const float sy = std::sin(particle.rotation.y), cy = std::cos(particle.rotation.y);
    const float sz = std::sin(particle.rotation.z), cz = std::cos(particle.rotation.z);

    // R = Ry * Rx * Rz, column-major (glm), column vectors.
    const glm::mat3 rotation(
        // column 0
        cy * cz + sy * sx * sz, cx * sz, -sy * cz + cy * sx * sz,
        // column 1
        -cy * sz + sy * sx * cz, cx * cz, sy * sz + cy * sx * cz,
        // column 2
        sy * cx, -sx, cy * cx);

    const float s = particle.scale;
    glm::mat4 model_mat(1.0f);
    model_mat[0] = glm::vec4(rotation[0] * s, 0.0f);
    model_mat[1] = glm::vec4(rotation[1] * s, 0.0f);
    model_mat[2] = glm::vec4(rotation[2] * s, 0.0f);
    // T * R * S * T(-center): translation = position - R * (center * s).
    model_mat[3] = glm::vec4(particle.position - rotation * (center_offset * s), 1.0f);
    return model_mat;
}

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
    batches_.clear();
    instance_scratch_.clear();
    instance_buffer_.reset();
    instance_buffer_capacity_ = 0;

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

void EffectRenderer::upload_particle_instances(SDL_GPUCopyPass* copy_pass,
                                               const mmo::engine::systems::EffectSystem& effect_system) {
    ENGINE_PROFILE_ZONE_COLOR("EffectRenderer::upload_particle_instances", 0x27AE60);
    batches_.clear();
    instance_scratch_.clear();
    if (!copy_pass || !device_ || !model_manager_) {
        return;
    }

    // Particles within one emitter all share the emitter's model, so each
    // emitter packs into one contiguous instance range = one draw batch.
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
            // this; never upload from the frame copy pass — skip and log instead.
            if (!model->meshes.empty() && !model->meshes[0].uploaded) {
                ENGINE_LOG_WARN("effect", "Effect model '{}' not GPU-uploaded; skipping particles this frame",
                                emitter.definition->model);
                continue;
            }

            // Model-center recentering offset is constant per model; hoisted
            // out of the per-particle loop.
            const glm::vec3 center((model->min_x + model->max_x) * 0.5f, (model->min_y + model->max_y) * 0.5f,
                                   (model->min_z + model->max_z) * 0.5f);

            const uint32_t first_instance = static_cast<uint32_t>(instance_scratch_.size());
            for (const auto& particle : emitter.particles) {
                instance_scratch_.push_back(
                    {build_particle_transform(particle, center),
                     glm::vec4(particle.color.r, particle.color.g, particle.color.b, particle.opacity)});
            }
            batches_.push_back(
                {model, first_instance, static_cast<uint32_t>(instance_scratch_.size()) - first_instance});
        }
    }

    if (instance_scratch_.empty()) {
        return;
    }

    const size_t required_size = instance_scratch_.size() * sizeof(gpu::ParticleInstanceData);
    if (!instance_buffer_ || instance_buffer_capacity_ < required_size) {
        // Grow geometrically with a floor of 16 KB (~200 particles). The old
        // buffer may still be referenced by last frame's commands; SDL defers
        // the actual release until the GPU is done with it.
        instance_buffer_capacity_ = std::max(required_size * 2, static_cast<size_t>(16 * 1024));
        instance_buffer_ =
            gpu::GPUBuffer::create_dynamic(*device_, gpu::GPUBuffer::Type::Storage, instance_buffer_capacity_);
        if (!instance_buffer_) {
            ENGINE_LOG_ERROR("effect", "Failed to create particle instance storage buffer");
            instance_buffer_capacity_ = 0;
            batches_.clear();
            return;
        }
    }
    instance_buffer_->update(copy_pass, instance_scratch_.data(), required_size);
}

void EffectRenderer::draw_particle_effects(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view,
                                           const glm::mat4& projection, const glm::vec3& camera_pos) {
    ENGINE_PROFILE_ZONE_COLOR("EffectRenderer::draw_particle_effects", 0xC0392B);
    if (!pass || !cmd || !device_ || !pipeline_registry_) {
        return;
    }
    // Batches + instance data were packed and uploaded this frame by
    // upload_particle_instances() (SceneRenderer's FrameUploads copy pass).
    if (batches_.empty() || !instance_buffer_) {
        return;
    }

    auto* pipeline = pipeline_registry_->get_pipeline(gpu::PipelineType::Particle);
    if (!pipeline) {
        return;
    }
    pipeline->bind(pass);

    // Frame-invariant camera + fragment uniforms: pushed once per pass.
    // Per-particle transform/tint/opacity ride in the instance buffer.
    gpu::InstancedCameraUniforms camera_uniforms = {};
    camera_uniforms.view = view;
    camera_uniforms.projection = projection;
    camera_uniforms.cameraPos = camera_pos;
    SDL_PushGPUVertexUniformData(cmd, 0, &camera_uniforms, sizeof(camera_uniforms));

    gpu::ParticleFragmentUniforms frag_uniforms = {};
    frag_uniforms.emissiveBoost = PARTICLE_EMISSIVE_BOOST;
    SDL_PushGPUFragmentUniformData(cmd, 0, &frag_uniforms, sizeof(frag_uniforms));

    // Instance data was uploaded once for the whole frame; bind it once.
    SDL_GPUBuffer* instance_buf = instance_buffer_->handle();
    SDL_BindGPUVertexStorageBuffers(pass, 0, &instance_buf, 1);

    SDL_GPUTexture* last_bound_tex = nullptr;

    // One instanced draw per (emitter batch, mesh); the 16-byte batch uniform
    // points the vertex shader at the batch's first instance record.
    for (const auto& batch : batches_) {
        gpu::ParticleBatchUniforms batch_uniforms = {};
        batch_uniforms.instanceOffset = batch.first_instance;
        SDL_PushGPUVertexUniformData(cmd, 1, &batch_uniforms, sizeof(batch_uniforms));

        for (const auto& mesh : batch.model->meshes) {
            if (!mesh.uploaded || !mesh.vertex_buffer || !mesh.index_buffer || mesh.index_count() == 0) {
                continue;
            }

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

            SDL_DrawGPUIndexedPrimitives(pass, mesh.index_count(), batch.instance_count, 0, 0, 0);
        }
    }
}

} // namespace mmo::engine::render
