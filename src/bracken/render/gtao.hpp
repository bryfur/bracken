#pragma once

#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>

namespace bracken::render {

namespace gpu = bracken::gpu;

namespace detail {

/**
 * CPU reference for the GTAO cosine-weighted horizon arc integration
 * (gtao.frag.hlsl) — keep in sync with the shader. Per slice, the shader
 * finds two horizon cosines (one per march direction), converts them to
 * angles clamped to the hemisphere around the projected normal, and
 * integrates visibility analytically (Jimenez et al. 2016, eq. 9 / XeGTAO).
 *
 * `horizon_cos0/1` are the horizon cosines relative to the view vector for
 * the negative/positive slice direction; `n` is the signed angle of the
 * projected normal from the view vector; returns visibility in [0, 1]
 * BEFORE the projected-normal-length weighting the shader applies per slice.
 * Unoccluded horizons (cos == cos(n ± π/2)) yield 1; horizons raised to the
 * view vector (cos == 1) yield 0.
 */
[[nodiscard]] inline float gtao_slice_visibility(float horizon_cos0, float horizon_cos1, float n) {
    constexpr float HALF_PI = 1.57079632679489661923f;
    const float cos_n = std::cos(n);
    float h0 = -std::acos(std::clamp(horizon_cos0, -1.0f, 1.0f));
    float h1 = std::acos(std::clamp(horizon_cos1, -1.0f, 1.0f));
    h0 = n + std::clamp(h0 - n, -HALF_PI, HALF_PI);
    h1 = n + std::clamp(h1 - n, -HALF_PI, HALF_PI);
    const float sin_n = std::sin(n);
    const float iarc0 = (cos_n + 2.0f * h0 * sin_n - std::cos(2.0f * h0 - n)) * 0.25f;
    const float iarc1 = (cos_n + 2.0f * h1 * sin_n - std::cos(2.0f * h1 - n)) * 0.25f;
    return iarc0 + iarc1;
}

} // namespace detail

/**
 * Screen-space ground-truth ambient occlusion (GTAO, Jimenez et al. 2016).
 *
 * Runs AFTER the depth prepass and BEFORE the forward pass, from the prepass
 * depth alone (glTF alphaMode MASK/BLEND geometry is not in the prepass, so
 * it neither receives its own depth nor occludes). The forward shaders
 * sample the full-res result (`ao_texture()`) and multiply it into their
 * AMBIENT term only — direct sun light and speculars are never attenuated.
 *
 * Per-frame pass chain (render(), all raster fullscreen passes at the AO
 * working resolution = half res for AOMode::GTAO_Half, full res for
 * AOMode::GTAO_Full):
 *   1. GTAO           depth -> raw AO (R8); horizon search + cosine-weighted
 *                     arc integration, per-pixel IGN + per-frame golden-ratio
 *                     jitter, depth-reconstructed edge-aware normals.
 *   2. GTAOTemporal   raw AO + history + velocity + depth -> new history
 *                     (RG16F: AO + linear depth); velocity reprojection with
 *                     relative-depth disocclusion rejection.
 *   3-4. BlurAO       depth-aware separable blur of the accumulated AO
 *                     (history stays unfiltered so the blur never compounds).
 *   5. GTAOUpsample   joint-bilateral upsample to the full-res R8 AO texture
 *                     (+ ao_strength); exact passthrough at full working res.
 *
 * History is INVALID (temporal pass outputs the current frame) on the first
 * frame and after resize()/set_half_res()/invalidate_history() — the
 * SceneRenderer invalidates whenever AO was off or the prepass didn't run,
 * mirroring TAA's policy.
 */
class GTAO {
public:
    GTAO();
    ~GTAO();

    GTAO(const GTAO&) = delete;
    GTAO& operator=(const GTAO&) = delete;
    GTAO(GTAO&&) = delete;
    GTAO& operator=(GTAO&&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device, int width, int height, bool half_res = true);
    void shutdown();
    void resize(int width, int height); // no-op at same size; otherwise invalidates history
    void set_half_res(bool half_res);   // no-op at same quality; otherwise invalidates history
    bool half_res() const { return half_res_; }

    bool is_ready() const;

    /** Drop accumulated history: the next render() outputs the current frame
     *  unblended. Called on resize/quality change and by the SceneRenderer
     *  whenever AO was inactive for a frame. */
    void invalidate_history() { history_valid_ = false; }
    bool history_valid() const { return history_valid_; }

    /** Full-res R8 AO texture the forward pass samples (1 = unoccluded).
     *  Cleared to white at init/resize, so it is safe to bind before the
     *  first render(). Null until init(). */
    SDL_GPUTexture* ao_texture() const { return ao_full_ ? ao_full_->handle() : nullptr; }

    struct FrameParams {
        // The DRAW projection (jittered under TAA — it must match the depth
        // buffer contents) and its inverse.
        glm::mat4 projection{1.0f};
        glm::mat4 inv_projection{1.0f};
        float radius_scale = 1.0f; // GraphicsSettings::ao_radius
        float strength = 1.0f;     // GraphicsSettings::ao_strength
        uint64_t frame_index = 0;  // drives the temporal rotation/offset jitter
    };

    /**
     * Record the AO pass chain (no render pass may be active). `depth` is the
     * full-res prepass scene depth; `velocity` the full-res prepass RG16F
     * velocity target — both must hold THIS frame's contents
     * (SceneRenderer::depth_prepass_active()).
     */
    void render(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, gpu::GPUTexture& depth,
                gpu::GPUTexture& velocity, const FrameParams& params);

private:
    void create_textures();
    void clear_full_res_texture();

    gpu::GPUDevice* device_ = nullptr;

    // Working-resolution chain.
    std::unique_ptr<gpu::GPUTexture> ao_raw_;     // R8: this frame's raw GTAO
    std::unique_ptr<gpu::GPUTexture> blur_tmp_;   // R8: horizontal blur ping
    std::unique_ptr<gpu::GPUTexture> ao_blurred_; // R8: blurred accumulated AO
    // Double-buffered RG16F history: [history_write_] receives this frame's
    // accumulation while the other slot supplies last frame's.
    std::array<std::unique_ptr<gpu::GPUTexture>, 2> history_;
    int history_write_ = 0;
    bool history_valid_ = false;

    // Full-resolution output the forward pass samples.
    std::unique_ptr<gpu::GPUTexture> ao_full_;

    SDL_GPUSampler* nearest_clamp_sampler_ = nullptr;
    SDL_GPUSampler* linear_clamp_sampler_ = nullptr;

    int width_ = 0;  // full resolution
    int height_ = 0;
    int ao_width_ = 0; // working resolution (== full res unless half_res_)
    int ao_height_ = 0;
    bool half_res_ = true;
};

} // namespace bracken::render
