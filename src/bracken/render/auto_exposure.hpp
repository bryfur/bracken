#pragma once

#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace bracken::render {

namespace gpu = bracken::gpu;

// =============================================================================
// CPU reference for the auto-exposure compute shaders. Keep in sync with
// exposure_histogram.comp.hlsl / exposure_adapt.comp.hlsl — auto_exposure
// tests pin this math, and the shaders replicate it operation-for-operation
// (the adapt reduce runs serially on one thread for exactly this reason).
// =============================================================================

/// Histogram size (fixed by the shaders' [numthreads] and LDS arrays).
inline constexpr int EXPOSURE_HISTOGRAM_BINS = 256;

/// log2 luminance mapped to bin 1 / bin 255. Bin 0 collects everything below
/// exp2(EXPOSURE_MIN_LOG2_LUMA) ("black": counted toward the percentile
/// total, contributes no luminance). Range EV -10..+16 -> ~0.102 EV per bin.
inline constexpr float EXPOSURE_MIN_LOG2_LUMA = -10.0f;
inline constexpr float EXPOSURE_MAX_LOG2_LUMA = 16.0f;

/** Histogram bin for a linear-HDR luminance (mirror of binFromLuma()). */
[[nodiscard]] inline uint32_t exposure_bin_from_luma(float luma) {
    if (luma < std::exp2(EXPOSURE_MIN_LOG2_LUMA)) {
        return 0;
    }
    const float t = std::clamp(
        (std::log2(luma) - EXPOSURE_MIN_LOG2_LUMA) / (EXPOSURE_MAX_LOG2_LUMA - EXPOSURE_MIN_LOG2_LUMA), 0.0f, 1.0f);
    return 1 + static_cast<uint32_t>(t * 254.0f);
}

/** log2 luminance at the lower edge of a non-black bin (mirror of
 *  binLog2Luma(); bins 1..255). */
[[nodiscard]] inline float exposure_bin_log2_luma(uint32_t bin) {
    return EXPOSURE_MIN_LOG2_LUMA +
           (static_cast<float>(bin) - 1.0f) / 254.0f * (EXPOSURE_MAX_LOG2_LUMA - EXPOSURE_MIN_LOG2_LUMA);
}

/**
 * Percentile-clipped histogram reduce -> UNCLAMPED target EV (mirror of the
 * exposure_adapt reduce; the shader clamps the result to [minEv, maxEv]).
 *
 * Pixels below the `low_percentile` and above the (1 - high_percentile)
 * cumulative marks are ignored; a bin straddling a mark contributes only its
 * inside portion. The remaining counts average the bins' log2 luminance and
 * target_ev = log2_target_gray - avg (so applying exp2(target_ev) maps the
 * metered average onto the target gray). Empty/all-black histograms return 0.
 */
[[nodiscard]] inline float exposure_target_ev(std::span<const uint32_t> bins, float low_percentile,
                                              float high_percentile, float log2_target_gray) {
    float total = 0.0f;
    for (uint32_t c : bins) {
        total += static_cast<float>(c);
    }
    if (total <= 0.0f) {
        return 0.0f;
    }
    const float clip_lo = total * low_percentile;
    const float clip_hi = total * (1.0f - high_percentile);
    float cumulative = 0.0f;
    float weight_sum = 0.0f;
    float weighted_log2 = 0.0f;
    for (size_t bin = 0; bin < bins.size(); ++bin) {
        const float count = static_cast<float>(bins[bin]);
        const float inside = std::max(0.0f, std::min(cumulative + count, clip_hi) - std::max(cumulative, clip_lo));
        cumulative += count;
        if (bin == 0) {
            continue; // "black" bin: eats percentile budget, adds no luma
        }
        weight_sum += inside;
        weighted_log2 += inside * exposure_bin_log2_luma(static_cast<uint32_t>(bin));
    }
    if (weight_sum <= 0.0f) {
        return 0.0f;
    }
    return log2_target_gray - weighted_log2 / weight_sum;
}

/**
 * One temporal-adaptation step (mirror of the exposure_adapt tail): move
 * current toward target at most speed_up EV/s upward / speed_down EV/s
 * downward, landing exactly on the target when within one step.
 */
[[nodiscard]] inline float exposure_adapt(float current_ev, float target_ev, float dt, float speed_up,
                                          float speed_down) {
    const float delta = target_ev - current_ev;
    return current_ev + std::clamp(delta, -speed_down * dt, speed_up * dt);
}

/**
 * Histogram-based auto exposure — the engine's first compute-shader consumer.
 *
 * Two dispatches per frame over the TAA-resolved (pre-bloom) HDR scene color:
 * exposure_histogram.comp (16x16 tiles -> LDS -> one atomic merge per
 * non-zero bin) then exposure_adapt.comp (single 256-thread workgroup:
 * reduce + temporal adaptation into a tiny persistent state buffer). The
 * composite pass reads state_buffer() as a read-only fragment SSBO and
 * multiplies exp2(current EV) into its exposure factor; the GraphicsSettings
 * `exposure` value stays as a manual bias on top.
 *
 * No screen-size resources: the histogram dispatch is sized purely by
 * uniforms, so there is no resize() to call.
 *
 * The state persists across frames; invalidate() makes the next render()
 * SNAP to the metered target instead of adapting toward it (used on init and
 * whenever auto exposure was inactive for a frame, mirroring the TAA/GTAO
 * history-invalidation policy).
 */
class AutoExposure {
public:
    /// Percentile clipping defaults: ignore the darkest 60% and brightest 5%.
    static constexpr float DEFAULT_LOW_PERCENTILE = 0.60f;
    static constexpr float DEFAULT_HIGH_PERCENTILE = 0.05f;

    /// Scene log2 luminance the exposure converges to. Calibrated so the
    /// default playground scene (midday sun, exposure bias 1.0) converges to
    /// ~EV 0, i.e. auto exposure ON approximately matches the
    /// pre-auto-exposure brightness (the scene's steady-state
    /// percentile-clipped average meters -1.91; -1.80 sits it and the smoke
    /// screenshot's dimmed-sun framing symmetrically within +-0.1 EV).
    static constexpr float DEFAULT_LOG2_TARGET_GRAY = -1.80f;

    /// render() calls that SNAP to the metered target after invalidate()
    /// (instead of rate-limited adaptation). More than one because the first
    /// frames after init/re-enable routinely meter a half-warmed scene
    /// (pending uploads); by the third frame the metering is representative.
    static constexpr int SNAP_FRAMES = 3;

    /// Default cadence of the fenced debug readback of the state buffer.
    static constexpr float DEFAULT_READBACK_INTERVAL = 1.0f;

    AutoExposure();
    ~AutoExposure();

    AutoExposure(const AutoExposure&) = delete;
    AutoExposure& operator=(const AutoExposure&) = delete;
    AutoExposure(AutoExposure&&) = delete;
    AutoExposure& operator=(AutoExposure&&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device);
    void shutdown();
    bool is_ready() const;

    /** Snap to the metered target on the next SNAP_FRAMES render() calls
     *  (first frames / re-enable after a gap — adaptation history is
     *  meaningless then, and the first metered frames may not be
     *  representative yet). */
    void invalidate() { snap_frames_left_ = SNAP_FRAMES; }

    struct FrameParams {
        float dt = 0.0f;         // frame delta time, seconds
        float speed_up = 1.5f;   // EV/s toward brighter (defaults track GraphicsSettings)
        float speed_down = 2.5f; // EV/s toward darker
        float min_ev = -4.0f;    // clamp on the adapted EV
        float max_ev = 4.0f;
    };

    /**
     * Record the histogram + adapt dispatches (no render pass may be open).
     * `scene_color` is the HDR image to meter — the offscreen scene color
     * AFTER the TAA block, so exposure meters what composite will see.
     * Also polls/requests the periodic debug readback.
     */
    void render(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, gpu::GPUTexture& scene_color,
                const FrameParams& params);

    /** Persistent {current EV, target EV, avg log2 luma, valid} float4 buffer
     *  (GRAPHICS_STORAGE_READ — bind as the composite pass's fragment SSBO).
     *  Null before init(). */
    SDL_GPUBuffer* state_buffer() const { return state_ ? state_->handle() : nullptr; }

    // ---- Metering tunables (engine API, not persisted settings) ----
    void set_percentiles(float low, float high);
    void set_log2_target_gray(float value) { log2_target_gray_ = value; }
    float log2_target_gray() const { return log2_target_gray_; }

    // ---- Debug readback (DevUI display / smoke logging) ----
    struct ExposureState {
        float current_ev = 0.0f;
        float target_ev = 0.0f;
        float avg_log2_luma = 0.0f;
    };

    /** Most recent completed readback (1+ frames stale; empty until the first
     *  one lands). The readback is a tiny fenced 16-byte download recorded at
     *  most every readback interval — cost is unmeasurable. */
    const std::optional<ExposureState>& latest_state() const { return latest_state_; }

    /** Readback cadence in seconds (0 = every render() call). */
    void set_readback_interval(float seconds) { readback_interval_ = std::max(0.0f, seconds); }

private:
    void poll_readback();
    void request_readback();

    gpu::GPUDevice* device_ = nullptr;

    // 256 x uint32 histogram; zeroed at creation and re-zeroed by the adapt
    // pass each frame after reading.
    std::unique_ptr<gpu::GPUBuffer> histogram_;
    // float4 {current EV, target EV, avg log2 luma, valid}; persistent.
    std::unique_ptr<gpu::GPUBuffer> state_;

    SDL_GPUSampler* nearest_clamp_sampler_ = nullptr;

    int snap_frames_left_ = SNAP_FRAMES;

    float low_percentile_ = DEFAULT_LOW_PERCENTILE;
    float high_percentile_ = DEFAULT_HIGH_PERCENTILE;
    float log2_target_gray_ = DEFAULT_LOG2_TARGET_GRAY;

    // Debug readback plumbing (own tiny command buffer + fence, polled
    // non-blocking; reads the state written by the PREVIOUS frame).
    SDL_GPUTransferBuffer* readback_transfer_ = nullptr;
    SDL_GPUFence* readback_fence_ = nullptr;
    float readback_interval_ = DEFAULT_READBACK_INTERVAL;
    float since_readback_ = DEFAULT_READBACK_INTERVAL; // request one immediately
    std::optional<ExposureState> latest_state_;
};

} // namespace bracken::render
