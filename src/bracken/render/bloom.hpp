#pragma once

#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include <memory>

namespace bracken::render {

namespace gpu = bracken::gpu;

/**
 * Bloom post-processing effect.
 *
 * When enabled (alongside AO offscreen rendering), extracts bright pixels
 * from the scene color buffer, progressively downsamples through a mip chain,
 * then progressively upsamples back (each mip ACCUMULATES the tent-filtered
 * smaller mip via additive blending) to produce a glow with both a tight core
 * and a wide falloff. The final bloom texture is composited with the scene
 * color in the composite pass.
 *
 * Uses a 13-tap downsample filter and 9-tap tent upsample (Jimenez 2014).
 */
class Bloom {
public:
    Bloom();
    ~Bloom();

    Bloom(const Bloom&) = delete;
    Bloom& operator=(const Bloom&) = delete;
    Bloom(Bloom&&) = delete;
    Bloom& operator=(Bloom&&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device, int width, int height);
    void shutdown();
    void resize(int width, int height);

    bool is_ready() const;

    /**
     * Run the bloom downsample + upsample chain.
     * Must be called after the offscreen scene render but before the composite pass.
     *
     * @param cmd         Active command buffer (no render pass should be active)
     * @param registry    Pipeline registry for bloom pipelines
     * @param scene_color The offscreen scene color texture to extract bloom from
     * @param threshold   POST-exposure luminance above which pixels bloom. The
     *                    prefilter thresholds scene luminance against the SAME
     *                    product the composite applies — the manual bias times
     *                    `exp2(adapted EV)` — so what blooms is what will be
     *                    bright on screen rather than what is bright in the
     *                    scene buffer.
     * @param exposure    The composite pass's manual exposure bias.
     * @param exposure_buffer AutoExposure's state SSBO, or null when auto
     *                    exposure is off/unavailable. Non-null IS the flag: a
     *                    zeroed filler is bound in its place so the shader's
     *                    always-declared slot stays valid. Only the threshold
     *                    decision uses either factor — bloom output stays in
     *                    pre-exposure scene units, because composite applies
     *                    exposure to the summed bloom.
     */
    void render(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& registry, SDL_GPUTexture* scene_color,
                float threshold = 1.0f, float exposure = 1.0f,
                SDL_GPUBuffer* exposure_buffer = nullptr);

    /**
     * Get the final bloom result texture for the composite pass.
     * Returns the first (largest) mip in the chain after upsampling.
     */
    SDL_GPUTexture* bloom_texture() const;

private:
    void create_textures(int width, int height);
    void create_sampler();
    static int compute_mip_count(int width, int height);

    gpu::GPUDevice* device_ = nullptr;

    // Downsample/upsample mip chain. The active count scales with resolution
    // (clamp(log2(min(w,h)) - 7, 4, 6)): wider glow needs more mips at 4K+
    // without paying for them at lower resolutions.
    static constexpr int MAX_MIPS = 6;
    std::unique_ptr<gpu::GPUTexture> mip_textures_[MAX_MIPS];
    int mip_count_ = 4;

    SDL_GPUSampler* linear_sampler_ = nullptr;

    /// Zeroed filler for the prefilter's always-declared exposure SSBO slot,
    /// read only when `autoExposureEnabled` is set. Bloom owns its own rather
    /// than borrowing PostProcessStack's: that one is private to another unit,
    /// and 16 bytes is cheaper than an accessor coupling the two passes.
    std::unique_ptr<gpu::GPUBuffer> dummy_exposure_buffer_;

    int width_ = 0;
    int height_ = 0;
};

} // namespace bracken::render
