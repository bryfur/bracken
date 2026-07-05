#pragma once

#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/systems/effect_system.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <SDL3/SDL_gpu.h>
#include <unordered_map>
#include <vector>

namespace bracken::render {

namespace gpu = bracken::gpu;

/// Compose a particle's world transform: translate(position) * rotY * rotX *
/// rotZ * scale * translate(-center_offset). The rotation is built directly
/// from the Euler angles (one sin/cos pair per axis) instead of chained
/// glm::rotate 4x4 multiplies; exposed as a free function so tests can pin
/// equivalence with the naive glm chain.
glm::mat4 build_particle_transform(const bracken::systems::Particle& particle, const glm::vec3& center_offset);

/**
 * EffectRenderer draws particle-based spell/attack effects.
 *
 * Particles are small meshes drawn instanced through the dedicated Particle
 * pipeline: per-particle {transform, tint+opacity} records are packed into a
 * per-frame storage buffer (uploaded in SceneRenderer's consolidated
 * FrameUploads copy pass), then each (emitter, mesh) batch is ONE instanced
 * draw with a small instance-offset uniform. Shading is unlit additive HDR
 * (ONE, ONE), depth test LESS_OR_EQUAL against opaques, depth write off.
 * Opacity fades by scaling the emitted RGB, and the emissive boost pushes
 * bright particles over the bloom threshold so spell glow feeds the bloom
 * chain.
 */
class EffectRenderer {
public:
    EffectRenderer();
    ~EffectRenderer();

    // Non-copyable
    EffectRenderer(const EffectRenderer&) = delete;
    EffectRenderer& operator=(const EffectRenderer&) = delete;
    EffectRenderer(EffectRenderer&&) = delete;
    EffectRenderer& operator=(EffectRenderer&&) = delete;

    /**
     * Initialize effect rendering resources (SDL3 GPU API version).
     * @param device GPU device for resource creation
     * @param pipeline_registry Pipeline registry for shader pipelines
     * @param model_manager Model manager for effect models
     */
    [[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry, ModelManager* model_manager);

    /**
     * Clean up resources.
     */
    void shutdown();

    /**
     * Ensure every model referenced by the effect system's active emitters is
     * resolved + GPU-uploaded. Must be called OUTSIDE any render pass (uploads
     * submit their own copy work); SceneRenderer calls it in the pre-pass
     * section of render_frame so first spell use never hitches mid-pass.
     */
    void ensure_models_uploaded(const bracken::systems::EffectSystem& effect_system);

    /**
     * Pack per-particle instance data ({model matrix, tint+opacity}) for
     * every active emitter into the per-frame instance storage buffer and
     * record its upload. Particles of one emitter share the emitter's model,
     * so each emitter becomes one contiguous batch. Models that missed the
     * ensure_models_uploaded window are skipped (never uploaded from here).
     * Must be called from the frame's consolidated copy pass before
     * draw_particle_effects — the draw consumes the batches built here.
     * @param copy_pass Open copy pass to record the upload into
     * @param effect_system Effect system containing all active effects and particles
     */
    void upload_particle_instances(SDL_GPUCopyPass* copy_pass,
                                   const bracken::systems::EffectSystem& effect_system);

    /**
     * Draw the particle batches packed by upload_particle_instances() through
     * the Particle pipeline: camera + fragment uniforms are pushed once per
     * pass, then each (emitter batch, mesh) pair is a single instanced draw.
     * @param pass Active render pass
     * @param cmd Command buffer for uniform uploads
     * @param view View matrix
     * @param projection Projection matrix
     * @param camera_pos Camera position
     */
    void draw_particle_effects(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view,
                               const glm::mat4& projection, const glm::vec3& camera_pos);

private:
    /// Resolve (and cache) the ModelHandle for an emitter definition.
    /// Definitions are loaded once and have stable addresses, so a
    /// pointer-keyed cache turns per-particle string hashing into a single
    /// pointer-hash lookup per emitter.
    ModelHandle resolve_emitter_model(const bracken::EmitterDefinition& def);

    /// One instanced draw group: all particles of one emitter (they share the
    /// emitter's model), packed contiguously into the instance buffer.
    struct ParticleBatch {
        const Model* model = nullptr;
        uint32_t first_instance = 0;
        uint32_t instance_count = 0;
    };

    gpu::GPUDevice* device_ = nullptr;
    gpu::PipelineRegistry* pipeline_registry_ = nullptr;
    ModelManager* model_manager_ = nullptr;
    SDL_GPUSampler* sampler_ = nullptr;
    std::unique_ptr<gpu::GPUTexture> dummy_white_texture_; // bound when a mesh has no base color texture

    // EmitterDefinition* -> ModelHandle, resolved once per definition.
    std::unordered_map<const bracken::EmitterDefinition*, ModelHandle> emitter_model_cache_;

    // Per-frame instancing state, rebuilt by upload_particle_instances().
    // Model pointers stay valid frame-locally (same contract as the bone
    // palette scratch in SceneRenderer). Vectors keep capacity across frames.
    std::vector<ParticleBatch> batches_;
    std::vector<gpu::ParticleInstanceData> instance_scratch_;
    std::unique_ptr<gpu::GPUBuffer> instance_buffer_;
    size_t instance_buffer_capacity_ = 0;
};

} // namespace bracken::render
