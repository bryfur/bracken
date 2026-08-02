#ifndef BRACKEN_RT_DENOISE_HLSLI
#define BRACKEN_RT_DENOISE_HLSLI

// rt_denoise.hlsli — the R5b denoiser core, transliterated.
//
// TRANSLITERATION of render/rt_denoise_math.hpp, which is the source of truth,
// carries the reasoning for every constant below, and is tested with no GPU.
// Change one, change both — `rt_denoise_math.test.cpp` pins the C++ half and
// the numbers here are copied from it verbatim.
//
// Nothing in this file declares a resource or reads a uniform, so any pass can
// include it: the sun and local visibility denoisers, the reflection temporal
// pass, and the screen-probe filter all do.

// For `rtSeed` (the stratum picker below) and `luminance`. Both files are
// include-guarded and declare no resources, so taking them here costs a
// consumer nothing but saves every consumer from having to know which of the
// helpers below came from where.
#include "rt_sampling.hlsli"

// ---------------------------------------------------------------------------
// 1. The analytic sun gradient (the CPU computes the signal; this is the shape)
// ---------------------------------------------------------------------------

#define RT_SUN_GRADIENT_ANGLE_DEADZONE_DEG 0.05
#define RT_SUN_GRADIENT_ANGLE_SATURATION_DEG 0.35
#define RT_SUN_GRADIENT_COLOR_DEADZONE 0.02
#define RT_SUN_GRADIENT_COLOR_SATURATION 0.25

/// Dead-zone-to-saturation ramp. CPU mirror: `rt_ramp`.
float rtRamp(float x, float deadzone, float saturation) {
    if (saturation <= deadzone) {
        return x > deadzone ? 1.0 : 0.0;
    }
    return saturate((x - deadzone) / (saturation - deadzone));
}

/// CPU mirror: `lighting_reset_floor`. Identity, and therefore worth a name
/// only because the identity is the CONTRACT: a floor of 0 must leave every
/// accumulator bit-identical to the pre-R5b tree.
float rtLightingResetFloor(float signal) {
    return saturate(signal);
}

/// Apply a reset to an accumulated sample count. 0 keeps the count exactly
/// (bit-identical); 1 collapses it to a single sample, i.e. this frame only.
float rtApplyReset(float count, float reset) {
    return lerp(count, 1.0, saturate(reset));
}

// ---------------------------------------------------------------------------
// 2. The traced sparse gradient
// ---------------------------------------------------------------------------

/// Traced pixels per gradient stratum, per axis (the 1-in-9 ray budget).
#define RT_GRADIENT_STRATUM 3

#define RT_GRADIENT_DILATION_RING0 1.0
#define RT_GRADIENT_DILATION_RING1 0.6
#define RT_GRADIENT_DILATION_RING2 0.3

/// Soft-dilation weight by Chebyshev ring. CPU mirror:
/// `gradient_dilation_weight` — read its comment for why this is a weighted MAX
/// and not an a-trous average.
float rtGradientDilationWeight(int dx, int dy) {
    int ring = max(abs(dx), abs(dy));
    if (ring == 0) { return RT_GRADIENT_DILATION_RING0; }
    if (ring == 1) { return RT_GRADIENT_DILATION_RING1; }
    if (ring == 2) { return RT_GRADIENT_DILATION_RING2; }
    return 0.0;
}

/// See `GRADIENT_DEPTH_REJECT_RELATIVE` for why this is 100x tighter than the
/// denoiser's own depth test, and for the measurement that set it.
#define RT_GRADIENT_DEPTH_REJECT_RELATIVE 0.0005

#define RT_GRADIENT_FIRE_THRESHOLD 0.5

/// Which pixel of a 3x3 stratum carries this frame's gradient ray. A pure
/// function of the stratum coordinate and the frame counter, so the choice is
/// bit-identical across runs and at any worker count — the same rule every
/// stochastic pass in this engine follows.
uint2 rtGradientPixelInStratum(uint2 stratum, uint frameIndex) {
    uint h = rtSeed(stratum, frameIndex) % 9u;
    return uint2(h % 3u, h / 3u);
}

// ---------------------------------------------------------------------------
// 3. Dual fast/slow history
// ---------------------------------------------------------------------------

#define RT_FAST_HISTORY_WINDOW 4.0
#define RT_FAST_HISTORY_CLAMP_SIGMA 2.0
#define RT_FAST_HISTORY_CLAMP_FLOOR 0.05

/// CPU mirror: `fast_history_margin`.
float rtFastHistoryMargin(float fastMean, float fastCount) {
    float p = saturate(fastMean);
    float n = max(fastCount, 1.0);
    float sigma = sqrt(max(p * (1.0 - p), 0.0) / n);
    return RT_FAST_HISTORY_CLAMP_SIGMA * sigma + RT_FAST_HISTORY_CLAMP_FLOOR;
}

// ---------------------------------------------------------------------------
// 4. Exposure-aware clamps
// ---------------------------------------------------------------------------

#define RT_DENOISE_PERCEPTUAL_EPSILON 0.004
#define RT_REFLECTION_FIREFLY_SIGMA 4.0
#define RT_REFLECTION_FIREFLY_PERCEPTUAL_FLOOR 2.0
#define RT_SCREEN_PROBE_FIREFLY_PERCEPTUAL_FLOOR 0.25

/// Scene-radiance units per unit of post-exposure luminance. CPU mirror:
/// `exposure_scene_scale`.
float rtExposureSceneScale(float adaptedEv) {
    return exp2(-adaptedEv);
}

/// CPU mirror: `clamp_sigma_floor`.
float rtClampSigmaFloor(float adaptedEv) {
    return RT_DENOISE_PERCEPTUAL_EPSILON * rtExposureSceneScale(adaptedEv);
}

/// CPU mirror: `reflection_firefly_ceiling`.
float rtReflectionFireflyCeiling(float meanLuma, float sigmaLuma, float adaptedEv) {
    return max(meanLuma + RT_REFLECTION_FIREFLY_SIGMA * sigmaLuma,
               RT_REFLECTION_FIREFLY_PERCEPTUAL_FLOOR * rtExposureSceneScale(adaptedEv));
}

/// CPU mirror: `screen_probe_firefly_ceiling`.
float rtScreenProbeFireflyCeiling(float probeMeanLuma, float fireflyMax, float adaptedEv) {
    return max(fireflyMax * probeMeanLuma,
               RT_SCREEN_PROBE_FIREFLY_PERCEPTUAL_FLOOR * rtExposureSceneScale(adaptedEv));
}

// ---------------------------------------------------------------------------
// The exposure state buffer
// ---------------------------------------------------------------------------
//
// render::AutoExposure::state_buffer() is a float4 {current EV, target EV,
// avg log2 luma, valid}, created as a ComputeStorage buffer and therefore
// bindable as a compute read-only storage buffer with no change to it at all.
// A pass that wants the adapted EV declares:
//
//   [[vk::binding(N, 0)]] StructuredBuffer<float4> exposureState : register(tN, space0);
//
// and calls this. `exposureEnabled` is 0 when auto exposure is off, where the
// caller binds the buffer anyway (a descriptor may not be empty) and passes the
// manual bias instead — so the clamps mean the same thing either way.
float rtAdaptedEv(float4 state, uint exposureEnabled, float manualEv) {
    // state.w is the adapt pass's own validity flag: 0 before the first
    // histogram has been reduced, where current EV is still whatever the buffer
    // was created with.
    //
    // The two terms ADD rather than one replacing the other, because that is
    // what the composite pass does: `GraphicsSettings::exposure` is a manual
    // BIAS multiplied on top of exp2(adapted EV) when auto exposure is on, and
    // the whole exposure when it is off. `manualEv` is its log2, so this is the
    // exact total in both configurations.
    float autoEv = (exposureEnabled != 0u && state.w > 0.5) ? state.x : 0.0;
    return autoEv + manualEv;
}

#endif // BRACKEN_RT_DENOISE_HLSLI
