#pragma once

#include "engine/gpu/gpu_device.hpp"
#include "engine/gpu/gpu_texture.hpp"
#include "engine/gpu/gpu_uniforms.hpp"
#include "engine/gpu/pipeline_registry.hpp"
#include <array>
#include <glm/glm.hpp>
#include <memory>

namespace mmo::engine::render {

namespace gpu = mmo::engine::gpu;

/// Shadow map configuration
static constexpr int CSM_MAX_CASCADES = 4;

/// Per-cascade data
struct CascadeData {
    glm::mat4 light_view_projection;
    float split_depth; // view-space far distance for this cascade
    float texel_world; // world units covered by one shadow texel
    float depth_range; // light-space depth extent in world units
};

/**
 * Cascaded Shadow Map manager.
 *
 * All cascades render into one D32 atlas (2*resolution x 2*resolution);
 * cascade i occupies quadrant ((i%2)*res, (i/2)*res). SDL3 GPU can't render
 * to depth-array layers, and a single atlas lets all cascades share one
 * render pass and one pair of sampler bindings in receivers.
 *
 * Cascade fitting uses the bounding sphere of each frustum slice with the
 * sphere center snapped to whole shadow texels in a rotation-only light
 * view, so static shadow edges stay pixel-stable under camera translation
 * and rotation.
 */
class ShadowMap {
public:
    ShadowMap();
    ~ShadowMap();

    ShadowMap(const ShadowMap&) = delete;
    ShadowMap& operator=(const ShadowMap&) = delete;
    ShadowMap(ShadowMap&&) = delete;
    ShadowMap& operator=(ShadowMap&&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device, int resolution = 2048);
    void shutdown();

    /// Reinitialize with a new per-cascade resolution (recreates the atlas)
    bool reinit(int resolution);

    /// Set number of active cascades (1-4), does not require reinit
    void set_active_cascades(int count);
    int active_cascades() const { return active_cascades_; }
    int resolution() const { return resolution_; }

    void update(const glm::mat4& camera_view, const glm::mat4& camera_proj, const glm::vec3& light_dir,
                float near_plane, float far_plane);

    /// Begin the single atlas render pass (clears all cascades)
    SDL_GPURenderPass* begin_shadow_pass(SDL_GPUCommandBuffer* cmd);
    /// Set viewport + scissor to a cascade's atlas quadrant
    void set_cascade_viewport(int cascade_index);
    void end_shadow_pass();

    // Accessors
    const std::array<CascadeData, CSM_MAX_CASCADES>& cascades() const { return cascades_; }

    /// The shared depth atlas containing all cascades
    SDL_GPUTexture* atlas_texture() const;

    /// Returns true if the atlas and samplers are ready
    bool is_ready() const;

    /// Raw NEAREST sampler (PCSS blocker search, volumetric fog)
    SDL_GPUSampler* shadow_sampler() const { return shadow_sampler_; }
    /// LINEAR comparison sampler (LESS_EQUAL) for hardware PCF
    SDL_GPUSampler* shadow_compare_sampler() const { return shadow_compare_sampler_; }

    gpu::ShadowDataUniforms get_shadow_uniforms(int shadow_mode = 2) const;

    // tan(sun angular radius): world penumbra width per world unit of
    // blocker-receiver distance. ~0.02 reads as a slightly soft sun.
    float light_size = 0.02f;
    float split_lambda = 0.85f;

private:
    void compute_cascade_splits(float near_plane, float far_plane);
    void fit_cascade(int cascade_index, const glm::mat4& camera_view, const glm::mat4& camera_proj,
                     const glm::vec3& light_dir, float near_split, float far_split);

    gpu::GPUDevice* device_ = nullptr;
    std::unique_ptr<gpu::GPUTexture> atlas_texture_;
    SDL_GPUSampler* shadow_sampler_ = nullptr;
    SDL_GPUSampler* shadow_compare_sampler_ = nullptr;
    SDL_GPURenderPass* current_shadow_pass_ = nullptr;

    int active_cascades_ = CSM_MAX_CASCADES;
    int resolution_ = 2048; // per-cascade quadrant resolution (atlas is 2x)
    float shadow_far_ = 1200.0f;
    std::array<CascadeData, CSM_MAX_CASCADES> cascades_ = {};
};

} // namespace mmo::engine::render
