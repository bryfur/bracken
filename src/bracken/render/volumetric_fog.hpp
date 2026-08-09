#pragma once

#include "bracken/contracts/camera_state.hpp"
#include "bracken/contracts/environment.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/render/shadow_map.hpp"
#include <glm/glm.hpp>
#include <memory>

namespace bracken::render {

namespace gi {
struct GiBindings;
} // namespace gi


namespace gpu = bracken::gpu;

/**
 * Volumetric fog with god rays (light shafts) post-processing effect.
 *
 * Renders at half resolution for performance. Ray-marches through a
 * height-based fog volume, sampling the shadow map to produce
 * inscattered light (god rays) where sunlight penetrates fog.
 *
 * Output is an RGBA16F texture:
 *   RGB = inscattered fog light
 *   A   = fog opacity (1 - transmittance)
 *
 * Composited in the final composite pass via premultiplied alpha blend.
 */
class VolumetricFog {
public:
    VolumetricFog();
    ~VolumetricFog();

    VolumetricFog(const VolumetricFog&) = delete;
    VolumetricFog& operator=(const VolumetricFog&) = delete;
    VolumetricFog(VolumetricFog&&) = delete;
    VolumetricFog& operator=(VolumetricFog&&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device, int width, int height);
    void shutdown();
    void resize(int width, int height);

    bool is_ready() const;

    /**
     * Run the volumetric fog ray march pass.
     * Must be called after the offscreen scene render and shadow passes,
     * but before the composite pass. No render pass should be active.
     *
     * Sun direction/color, fog color, base density, and the height-falloff /
     * march-range parameters all come from the EnvironmentState
     * (fog_volumetric_* fields + sun_dir + fog_color_horizon). God rays
     * sample every CSM cascade — the shader selects one per march sample by
     * camera distance — and sky pixels are marched too, so shafts silhouette
     * against the sky. Cost knob: the fixed step count in
     * volumetric_fog.frag.hlsl (numSteps, 24) — every half-res pixel pays it,
     * plus one shadow tap per step when god rays are on.
     *
     * @param cmd         Active command buffer
     * @param registry    Pipeline registry
     * @param depth_texture  Offscreen scene depth texture
     * @param shadow_map  Shadow map manager (all cascade atlas quadrants)
     * @param camera      Camera state for the frame
     * @param env         Environment (sun, fog colors, volumetric fog params)
     */
    void render(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& registry, SDL_GPUTexture* depth_texture,
                const ShadowMap& shadow_map, const scene::CameraState& camera, const EnvironmentState& env,
                bool god_rays_enabled = true, bool fog_enabled = true, float density_multiplier = 1.0f,
                const gi::GiBindings* gi = nullptr);

    /**
     * Build the uniform block render() pushes. Pure CPU (no GPU state
     * touched); public so the environment plumbing is unit-testable.
     */
    gpu::VolumetricFogUniforms build_uniforms(const ShadowMap& shadow_map, const scene::CameraState& camera,
                                              const EnvironmentState& env, bool god_rays_enabled, bool fog_enabled,
                                              float density_multiplier, const gi::GiBindings* gi = nullptr);

    /**
     * Get the fog result texture for the composite pass.
     */
    SDL_GPUTexture* fog_texture() const;

private:
    void create_textures(int width, int height);
    void create_sampler();

    gpu::GPUDevice* device_ = nullptr;
    std::unique_ptr<gpu::GPUTexture> fog_texture_; // Half-res RGBA16F
    // NEAREST, clamped: the pass's one sampler, and its one consumer is the
    // full-res depth texture read at half-res pixel centres. See create_sampler().
    SDL_GPUSampler* depth_sampler_ = nullptr;

    int width_ = 0;
    int height_ = 0;
    int fog_width_ = 0;
    int fog_height_ = 0;

    // Cache view_projection inverse: skips ~80 flops on frames where the camera is idle.
    glm::mat4 cached_view_projection_{0.0f};
    glm::mat4 cached_inv_view_projection_{1.0f};
};

} // namespace bracken::render
