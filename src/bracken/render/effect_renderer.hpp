#pragma once

#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render/particle_transform.hpp"
#include "bracken/scene/frame_packet.hpp" // Effect{Emitter,Particle}Snapshot
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <SDL3/SDL_gpu.h>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bracken::render {

namespace gpu = bracken::gpu;

/// The render path's particle transform: identical math over a frame's
/// EffectParticleSnapshot (particles reach the renderer only through the
/// sealed packet's effect-instance snapshot). Out-of-line so the hot inner
/// loop below is not the only thing pinning the composition; the generic form
/// is `build_particle_transform_of` in particle_transform.hpp, which is what
/// the test instantiates against a live systems::Particle.
glm::mat4 build_particle_transform(const bracken::scene::EffectParticleSnapshot& particle,
                                   const glm::vec3& center_offset);

/**
 * EffectRenderer draws particle-based spell/attack effects.
 *
 * Particles are small meshes drawn instanced through the dedicated Particle
 * pipeline: per-particle {transform, tint+opacity} records are packed into a
 * per-frame storage buffer (uploaded in SceneRenderer's consolidated
 * FrameUploads copy pass), then each (emitter, mesh) batch is ONE instanced
 * draw with a small instance-offset uniform. Shading is unlit additive HDR —
 * the blend is configured (SRC_ALPHA, ONE) and the fragment shader writes
 * alpha 1.0, so the effective color blend is (ONE, ONE) — with depth test
 * GREATER_OR_EQUAL (reversed-Z) against opaques, depth write off.
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
     * Ensure every model referenced by the frame's effect-instance snapshot
     * (the sealed packet's effect_emitters()) is resolved + GPU-uploaded.
     * Must be called OUTSIDE any render pass (uploads submit their own copy
     * work); SceneRenderer calls it in the pre-pass section of render_frame
     * so first spell use never hitches mid-pass. Main thread (GPU create).
     */
    void ensure_models_uploaded(std::span<const bracken::scene::EffectEmitterSnapshot> emitters);

    /** Emitters in the last ensure_models_uploaded() whose EmitterDefinition
     *  named a model the ModelManager does not have — those particles draw
     *  nothing. Folded into FrameDiagnostics::effect_models_unresolved. */
    [[nodiscard]] uint32_t models_unresolved() const { return models_unresolved_; }

    /**
     * Pack per-particle instance data ({model matrix, tint+opacity}) from
     * the sealed packet's effect-instance snapshot into the per-frame
     * instance storage buffer and record its upload. Particles of one
     * emitter share the emitter's model, so each emitter becomes one
     * contiguous batch. Models that missed the ensure_models_uploaded window
     * are skipped (never uploaded from here). Must be called from the
     * frame's consolidated copy pass before draw_particle_effects — the draw
     * consumes the batches built here.
     * @param copy_pass Open copy pass to record the upload into
     * @param emitters  Sealed packet's effect_emitters()
     * @param particles Sealed packet's effect_particles()
     */
    void upload_particle_instances(SDL_GPUCopyPass* copy_pass,
                                   std::span<const bracken::scene::EffectEmitterSnapshot> emitters,
                                   std::span<const bracken::scene::EffectParticleSnapshot> particles);

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
    friend struct EffectRendererTestAccess; // effect_renderer.test.cpp

    /// Resolve (and cache) the ModelHandle for an emitter definition.
    /// Definitions are loaded once and have stable addresses, so a
    /// pointer-keyed cache turns per-particle string hashing into a single
    /// pointer-hash lookup per emitter. Failed lookups are NOT cached — a
    /// model registered after an emitter's first use resolves on the next
    /// call (the miss is warned once per definition to avoid per-frame spam).
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
    // Only successful lookups are stored (see resolve_emitter_model).
    std::unordered_map<const bracken::EmitterDefinition*, ModelHandle> emitter_model_cache_;

    // Definitions already warned about for an unresolved model name, so the
    // uncached retry doesn't log every frame.
    std::unordered_set<const bracken::EmitterDefinition*> warned_missing_models_;
    // Emitters whose model name resolved to nothing during the last
    // ensure_models_uploaded() — folded into FrameDiagnostics by the renderer.
    uint32_t models_unresolved_ = 0;

    // Per-frame instancing state, rebuilt by upload_particle_instances().
    // Model pointers stay valid frame-locally (same contract as the bone
    // palette scratch in SceneRenderer). Vectors keep capacity across frames.
    std::vector<ParticleBatch> batches_;
    std::vector<gpu::ParticleInstanceData> instance_scratch_;
    std::unique_ptr<gpu::GPUBuffer> instance_buffer_;
    size_t instance_buffer_capacity_ = 0;
};

} // namespace bracken::render
