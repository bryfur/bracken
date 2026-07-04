#pragma once

#include "../gpu/gpu_device.hpp"
#include "../gpu/gpu_texture.hpp"
#include "../gpu/pipeline_registry.hpp"
#include "../model_loader.hpp"
#include "engine/systems/effect_system.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <SDL3/SDL_gpu.h>
#include <unordered_map>

namespace mmo::engine::render {

namespace gpu = mmo::engine::gpu;

/**
 * EffectRenderer draws particle-based spell/attack effects.
 *
 * Particles are small meshes drawn through the dedicated Particle pipeline:
 * unlit additive HDR (ONE, ONE), depth test LESS_OR_EQUAL against opaques,
 * depth write off. Opacity fades by scaling the emitted RGB, and the
 * emissive boost pushes bright particles over the bloom threshold so spell
 * glow feeds the bloom chain.
 */
class EffectRenderer {
public:
    EffectRenderer();
    ~EffectRenderer();

    // Non-copyable
    EffectRenderer(const EffectRenderer&) = delete;
    EffectRenderer& operator=(const EffectRenderer&) = delete;

    /**
     * Initialize effect rendering resources (SDL3 GPU API version).
     * @param device GPU device for resource creation
     * @param pipeline_registry Pipeline registry for shader pipelines
     * @param model_manager Model manager for effect models
     */
    bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry, ModelManager* model_manager);

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
    void ensure_models_uploaded(const mmo::engine::systems::EffectSystem& effect_system);

    /**
     * Draw all particles from an effect system through the Particle pipeline.
     * Particles are grouped per emitter (all particles of an emitter share one
     * model) so buffers/textures bind once per group. Models that missed the
     * pre-upload window are skipped (never uploaded mid-pass).
     * @param pass Active render pass
     * @param cmd Command buffer for uniform uploads
     * @param effect_system Effect system containing all active effects and particles
     * @param view View matrix
     * @param projection Projection matrix
     * @param camera_pos Camera position
     */
    void draw_particle_effects(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                               const mmo::engine::systems::EffectSystem& effect_system, const glm::mat4& view,
                               const glm::mat4& projection, const glm::vec3& camera_pos);

private:
    /// Resolve (and cache) the ModelHandle for an emitter definition.
    /// Definitions are loaded once and have stable addresses, so a
    /// pointer-keyed cache turns per-particle string hashing into a single
    /// pointer-hash lookup per emitter.
    ModelHandle resolve_emitter_model(const mmo::engine::EmitterDefinition& def);

    gpu::GPUDevice* device_ = nullptr;
    gpu::PipelineRegistry* pipeline_registry_ = nullptr;
    ModelManager* model_manager_ = nullptr;
    SDL_GPUSampler* sampler_ = nullptr;
    std::unique_ptr<gpu::GPUTexture> dummy_white_texture_; // bound when a mesh has no base color texture

    // EmitterDefinition* -> ModelHandle, resolved once per definition.
    std::unordered_map<const mmo::engine::EmitterDefinition*, ModelHandle> emitter_model_cache_;
};

} // namespace mmo::engine::render
