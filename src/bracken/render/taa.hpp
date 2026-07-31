#pragma once

#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include <array>
#include <glm/glm.hpp>
#include <memory>

namespace bracken::render {

namespace gpu = bracken::gpu;

/**
 * CPU reference for the TAA resolve's depth-based reprojection — the fallback
 * used for every pixel the velocity prepass did not cover (sky, glTF
 * alphaMode MASK foliage, blended draws). Keep in sync with
 * taa_resolve.frag.hlsl. For prepass-covered pixels the shader instead uses
 * `prev_uv = uv - velocity` (scene/velocity_math.hpp); for a static object
 * both paths must agree, which taa.test.cpp pins.
 *
 * `uv` is top-left-origin (u = ndc.x*0.5+0.5, v = 0.5-ndc.y*0.5);
 * `device_depth` is the reversed-Z depth-buffer value (near = 1, far = 0).
 * Device depth <= 0 (the cleared far plane / sky) reprojects the view-ray
 * DIRECTION with w = 0 so camera translation drops out — exact for points at
 * infinity. Returns uv unchanged when the point lands behind the previous
 * camera (w <= 0), matching the shader's "no history" bail-out.
 */
[[nodiscard]] inline glm::vec2 depth_reproject_prev_uv(const glm::mat4& curr_inv_view_proj_unjittered,
                                                       const glm::mat4& prev_view_proj_unjittered,
                                                       const glm::vec3& camera_pos, glm::vec2 uv,
                                                       float device_depth) {
    const glm::vec2 ndc(uv.x * 2.0f - 1.0f, 1.0f - 2.0f * uv.y);
    glm::vec4 prev_clip;
    if (device_depth <= 0.0f) {
        // Any depth in (0,1] unprojects to a point on the pixel's view ray;
        // only the direction matters for the w = 0 projection below.
        const glm::vec4 world = curr_inv_view_proj_unjittered * glm::vec4(ndc, 0.5f, 1.0f);
        const glm::vec3 dir = glm::vec3(world) / world.w - camera_pos;
        prev_clip = prev_view_proj_unjittered * glm::vec4(dir, 0.0f);
    } else {
        glm::vec4 world = curr_inv_view_proj_unjittered * glm::vec4(ndc, device_depth, 1.0f);
        world /= world.w;
        prev_clip = prev_view_proj_unjittered * world;
    }
    if (prev_clip.w <= 0.0f) {
        return uv;
    }
    const glm::vec2 prev_ndc = glm::vec2(prev_clip) / prev_clip.w;
    return glm::vec2(prev_ndc.x * 0.5f + 0.5f, 0.5f - prev_ndc.y * 0.5f);
}

/**
 * Temporal anti-aliasing resolve. Consumes the jittered linear-HDR offscreen
 * scene color plus the depth-prepass velocity target and offscreen depth,
 * accumulates into a double-buffered RGBA16F history, and writes the
 * (optionally sharpened) result back over the scene color texture — bloom,
 * volumetric fog and composite then read the stabilized image through their
 * existing inputs. Runs pre-tonemap so bloom flicker is damped too.
 *
 * Resolve quality (taa_resolve.frag.hlsl): 3x3 closest-depth velocity
 * dilation, depth reprojection for velocity-less pixels (translation-free at
 * the far plane), YCoCg variance clipping, 5-tap Catmull-Rom history
 * sampling, Karis luminance-weighted blend with disocclusion-scaled feedback.
 *
 * History is INVALID (resolve outputs the current frame) on the first frame
 * and after resize() or invalidate_history() — the SceneRenderer invalidates
 * whenever TAA was not active on the previous rendered frame.
 */
class TAA {
public:
    TAA();
    ~TAA();

    TAA(const TAA&) = delete;
    TAA& operator=(const TAA&) = delete;
    TAA(TAA&&) = delete;
    TAA& operator=(TAA&&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device, int width, int height);
    void shutdown();
    void resize(int width, int height); // no-op at same size; otherwise invalidates history

    bool is_ready() const;

    /** Drop accumulated history: the next render() outputs the current frame
     *  unblended. Called on resize and whenever TAA was inactive for a frame. */
    void invalidate_history() { history_valid_ = false; }
    bool history_valid() const { return history_valid_; }

    /** Post-resolve unsharp-mask strength (0 = plain copy-back). The default
     *  is a deliberately subtle counter to residual temporal softness. */
    static constexpr float DEFAULT_SHARPEN_STRENGTH = 0.25f;
    void set_sharpen_strength(float strength) { sharpen_strength_ = strength; }
    float sharpen_strength() const { return sharpen_strength_; }

    struct ResolveParams {
        // Inverse of this frame's / last frame's UNJITTERED view-projection
        // (SceneRenderer::current/previous_view_projection_unjittered()).
        glm::mat4 curr_inv_view_proj_unjittered{1.0f};
        glm::mat4 prev_view_proj_unjittered{1.0f};
        glm::vec3 camera_pos{0.0f};
    };

    /**
     * Record the resolve + sharpen passes (no render pass may be active).
     * Reads `scene_color` / `velocity` / `depth`, writes the new history,
     * then writes the sharpened history back over `scene_color`. All three
     * textures must be screen-sized; velocity contents must be this frame's
     * (SceneRenderer::depth_prepass_active()).
     */
    void render(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, gpu::GPUTexture& scene_color,
                gpu::GPUTexture& velocity, gpu::GPUTexture& depth, const ResolveParams& params);

private:
    void create_textures(int width, int height);

    gpu::GPUDevice* device_ = nullptr;

    // Double-buffered RGBA16F history: [history_write_] receives this frame's
    // resolve while the other slot supplies last frame's accumulation.
    std::array<std::unique_ptr<gpu::GPUTexture>, 2> history_;
    int history_write_ = 0;
    bool history_valid_ = false;

    float sharpen_strength_ = DEFAULT_SHARPEN_STRENGTH;

    SDL_GPUSampler* nearest_clamp_sampler_ = nullptr;
    SDL_GPUSampler* linear_clamp_sampler_ = nullptr;

    int width_ = 0;
    int height_ = 0;
};

} // namespace bracken::render
