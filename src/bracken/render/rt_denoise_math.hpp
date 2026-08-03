#pragma once

/**
 * The denoiser core of phase R5b (`docs/rt_lighting_plan.md` §5.6): every curve
 * that decides HOW MUCH HISTORY SURVIVES A FRAME, as the CPU source of truth
 * for the shader transliterations that cannot otherwise be tested.
 *
 * Three mechanisms live here, and they answer three different questions:
 *
 *  1. `sun_change_signal` / `lighting_reset_floor` — the ANALYTIC sun gradient.
 *     The renderer knows the sun moved before any ray is cast, so a
 *     time-of-day scrub or a lightning-style intensity spike collapses every
 *     accumulation window in the frame for exactly the frames it changes. Free
 *     (two dot products on the main thread), global, and inert for a static sun.
 *  2. `gradient_dilation_weight` — the reconstruction kernel of the TRACED
 *     sparse gradient (A-SVGF, Schied et al. 2018). One re-traced sample per
 *     3x3 stratum is a per-region "the lighting at this surface changed"
 *     detector; this is how one stratum's verdict is spread over its
 *     neighbourhood.
 *  3. `fast_history_margin` — the DUAL fast/slow history clamp. A four-frame
 *     mean beside the main accumulator, and the margin the main mean is allowed
 *     to sit outside it. Catches what gradients miss: sub-stratum changes, and
 *     the local signal's reservoir switching to a different light.
 *
 * Mirrors (change one, change both — every constant below is `#define`d
 * verbatim in `shaders/rt_denoise.hlsli`):
 *   lighting_reset_floor     -> rtLightingResetFloor
 *   gradient_dilation_weight -> rtGradientDilationWeight
 *   fast_history_margin      -> rtFastHistoryMargin
 *   exposure_scene_scale     -> rtExposureSceneScale
 *   clamp_sigma_floor        -> rtClampSigmaFloor
 *
 * Tier: pure (no state, no device, callable from anywhere).
 */

#include <algorithm>
#include <cmath>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

namespace bracken::render {

// ---------------------------------------------------------------------------
// 1. The analytic sun gradient
// ---------------------------------------------------------------------------

/**
 * Per-frame angular sun motion below which nothing happens, in DEGREES.
 *
 * Real time-of-day is 0.0042 deg/s (360 degrees in a day), i.e. 7e-5 deg at
 * 60 Hz — five hundred times under this dead zone. A static or realistically
 * moving sun therefore costs exactly nothing, which is the property that lets
 * this run unconditionally.
 */
inline constexpr float SUN_GRADIENT_ANGLE_DEADZONE_DEG = 0.05f;

/**
 * Per-frame angular sun motion at which the reset is total, in DEGREES.
 *
 * This is not a taste constant: 0.35 is `RtShadowSettings::sun_angular_radius_
 * degrees`, the solar disk's own angular RADIUS. The stochastic sun ray is
 * jittered across that disk, so once the disk has moved by its own radius in
 * one frame, this frame's samples and the previous frame's are drawn from
 * essentially disjoint sets of directions — the accumulated mean is no longer
 * an estimate of the same integral, and averaging it in is not conservatism but
 * error. Saturating exactly there is the physically honest place to saturate.
 */
inline constexpr float SUN_GRADIENT_ANGLE_SATURATION_DEG = 0.35f;

/// Relative sun-radiance change (|new - old| / max(|old|, eps)) below which
/// nothing happens. 2% per frame is under any dawn ramp and over float noise.
inline constexpr float SUN_GRADIENT_COLOR_DEADZONE = 0.02f;
/// ...and at which the reset is total. A 25%-per-frame swing is a lightning
/// flash, a cloud edge crossing the sun, or a scripted cut — never weather
/// drift, which the playground's own system moves over seconds.
inline constexpr float SUN_GRADIENT_COLOR_SATURATION = 0.25f;

/// Shared dead-zone-to-saturation ramp. Linear rather than smoothstepped: the
/// quantity being ramped is already a physical ratio, and a smoothstep here
/// would only make the response to a CONSTANT-RATE sweep non-constant.
[[nodiscard]] inline float rt_ramp(float x, float deadzone, float saturation) noexcept {
    if (saturation <= deadzone) {
        return x > deadzone ? 1.0f : 0.0f;
    }
    return std::clamp((x - deadzone) / (saturation - deadzone), 0.0f, 1.0f);
}

/**
 * How much the sun changed between two frames, as a scale-free [0, 1] signal.
 *
 * `dir` is the direction light TRAVELS (the engine's `EnvironmentState::sun_dir`
 * convention); its magnitude is irrelevant and both are normalized here.
 * `radiance` is `sun_color * sun_intensity` — the product, not the colour
 * alone, because a host that dims the sun by scaling intensity and a host that
 * dims it by darkening the colour have done the same thing to the image.
 *
 * The two channels are combined with MAX, not a sum: either one alone is
 * sufficient reason to distrust history, and adding them would make a sun that
 * moved a little AND dimmed a little look like a cut.
 */
[[nodiscard]] inline float sun_change_signal(const glm::vec3& prev_dir, const glm::vec3& dir,
                                             const glm::vec3& prev_radiance,
                                             const glm::vec3& radiance) noexcept {
    const float prev_len = glm::length(prev_dir);
    const float len = glm::length(dir);
    float angle_deg = 0.0f;
    if (prev_len > 1e-6f && len > 1e-6f) {
        const float cos_theta = std::clamp(glm::dot(prev_dir / prev_len, dir / len), -1.0f, 1.0f);
        angle_deg = std::acos(cos_theta) * (180.0f / 3.14159265358979323846f);
    }
    const float prev_mag = glm::length(prev_radiance);
    const float color_rel = glm::length(radiance - prev_radiance) / std::max(prev_mag, 1e-4f);

    return std::max(rt_ramp(angle_deg, SUN_GRADIENT_ANGLE_DEADZONE_DEG,
                            SUN_GRADIENT_ANGLE_SATURATION_DEG),
                    rt_ramp(color_rel, SUN_GRADIENT_COLOR_DEADZONE, SUN_GRADIENT_COLOR_SATURATION));
}

/**
 * The GLOBAL HYSTERESIS FLOOR the sun signal pushes into every temporal
 * accumulator this frame: 0 = keep all history, 1 = keep none.
 *
 * Identity at signal 0, which is what makes a static sun bit-identical to the
 * pre-R5b tree rather than merely close to it. The mapping is the identity for
 * the same reason `rt_ramp` is linear — the signal is already a normalized
 * fraction of "one frame's worth of change that invalidates a frame's worth of
 * history", so squaring or smoothing it would only decouple the response from
 * the sweep rate.
 */
[[nodiscard]] inline float lighting_reset_floor(float signal) noexcept {
    return std::clamp(signal, 0.0f, 1.0f);
}

// ---------------------------------------------------------------------------
// 2. The traced sparse gradient
// ---------------------------------------------------------------------------

/// Traced pixels per gradient stratum, per axis. 3 gives the 1-in-9 sample
/// budget §5.6 asks for: one re-traced ray per stratum per enabled signal.
inline constexpr int GRADIENT_STRATUM = 3;

/**
 * Reconstruction weight for a stratum `ring` steps (Chebyshev distance) from
 * the one being written.
 *
 * NOT an a-trous AVERAGE, and the deviation is deliberate. A-SVGF's kernel is
 * a b3-spline blur because its gradient is a continuous luminance ratio. Ours
 * is a BINARY visibility comparison — one re-traced bit against one stored bit
 * — so a stratum's raw verdict is 0 or 1, and blurring a lone 1 among its 24
 * quiet neighbours with a b3 kernel yields 0.14 and resets nothing. What the
 * signal actually means is "something at this surface changed", and the correct
 * spatial operator for that is a soft DILATION: the fired stratum resets its
 * own region completely, and its neighbourhood partially, monotone in the
 * distance. The pass takes the weighted MAX over the 5x5 stratum window.
 *
 * Three rings, so the pass is a 5x5 window at stratum resolution — 15x15 traced
 * pixels. Wider would smear a moving character's reset across the wall behind
 * it. The falloff is steep on purpose: a stratum that fired says something
 * about ITS 3x3 pixels and progressively less about its neighbours', and the
 * measured cost of a shallow falloff is that one noisy stratum resets a 15x15
 * region outright.
 */
inline constexpr float GRADIENT_DILATION_RING0 = 1.0f;
inline constexpr float GRADIENT_DILATION_RING1 = 0.6f;
inline constexpr float GRADIENT_DILATION_RING2 = 0.3f;

[[nodiscard]] inline float gradient_dilation_weight(int dx, int dy) noexcept {
    const int ring = std::max(std::abs(dx), std::abs(dy));
    switch (ring) {
    case 0: return GRADIENT_DILATION_RING0;
    case 1: return GRADIENT_DILATION_RING1;
    case 2: return GRADIENT_DILATION_RING2;
    default: return 0.0f;
    }
}

/**
 * Relative depth tolerance the gradient's own reprojection test uses, and TWO
 * ORDERS OF MAGNITUDE tighter than the denoiser's own 0.05.
 *
 * The two tests look identical and are asking different questions. The
 * denoiser's asks "is this history REUSABLE?" — a neighbouring point on the
 * same wall is a fine answer. The gradient's asks "is this LITERALLY the point
 * the previous frame traced?", because anything looser makes a difference in
 * the traced bit attributable to the surface having moved rather than to the
 * lighting having changed.
 *
 * The gap was measured, not chosen. With TAA on, the jitter moves each pixel's
 * reconstructed world position by up to half a pixel every frame; in geometry
 * as high-frequency as the playground's canopies and grass tufts that lands the
 * replay on a different leaf. At the denoiser's own 0.05 the static
 * `--rt-bench` run reports a 3.3% gradient fire rate that is ENTIRELY that
 * effect — the same run with TAA off reports 0.14% — and it drags the sun's
 * mean accumulation window from 26.6 frames down to 10.3 in exchange for no
 * signal at all.
 *
 * Tighter still keeps shrinking the floor and shrinks real detections with it:
 * a grazing ground plane has a large per-pixel depth derivative and
 * legitimately fails a very tight test. This is the knee of that curve, not an
 * optimum.
 *
 * RETUNED 0.0005 -> 0.0004 on 2026-08-02, and it is the ONE threshold in that
 * package that moved. The knee above was measured against a depth
 * linearization whose `depthA` sign was inverted (contracts/depth_linear.hpp),
 * which made the EFFECTIVE threshold `0.0005 * (1 - 2t/far)` — a curve, not a
 * constant, and always tighter than the nominal. Correcting the linearization
 * loosened it, and the static `--rt-bench 120` run — where the sun does not
 * move, so EVERY gradient fired is spurious — showed exactly that: fire
 * 0.0086 -> 0.0100, sun accumulation 17.84 -> 17.19 frames. Rather than scale
 * by a representative distance (there is no single one), the knee was
 * RE-MEASURED on the same run:
 *
 *     nominal   fire     sun_accum
 *     0.0005    0.0100   17.19
 *     0.0004    0.0086   17.95   <- reproduces the pre-fix calibration
 *     0.0003    0.0068   19.06
 *     0.0002    0.0047   20.70
 *
 * 0.0004 restores the measured behaviour to within a rounding of the statistic,
 * and the implied effective depth (1 - 2t/far = 0.8, i.e. t ~ 400 units) is the
 * near-field canopy and grass geometry the doc above says the fire rate is
 * dominated by. The 0.05 thresholds the same package touched were NOT retuned
 * (see gpu_uniforms.hpp): they document an INTENT — 5% relative — and this one
 * documents a MEASUREMENT, and only a measurement has to be re-taken when the
 * ruler changes.
 */
inline constexpr float GRADIENT_DEPTH_REJECT_RELATIVE = 0.0004f;

/**
 * Reconstructed gradient above which a pixel counts as "fired" for the bench
 * harness's fire-fraction statistic.
 *
 * 0.5 is not a taste value: `rtApplyReset` lerps the accumulated count toward
 * 1, so a gradient of 0.5 is exactly "this pixel's accumulation window was cut
 * at least in half". That is a statement a report can carry, where "the
 * gradient exceeded some number" is not.
 *
 * Where it sits relative to the dilation weights is load-bearing, and was
 * measured rather than reasoned about. It must be ABOVE
 * `GRADIENT_DILATION_RING2`: with a threshold under the weakest ring, one raw
 * stratum verdict lights its whole 5x5 footprint above the line and the
 * statistic reports the kernel's width instead of the scene's activity — on the
 * playground bench that inflated the figure ~25x and flattened its response to
 * the sweep rate to nothing. At or below `GRADIENT_DILATION_RING1` it counts
 * the 3x3 CORE of each detection: the strata that lost most of their
 * accumulation, which is the region a reader of the number cares about, and not
 * the ring-2 tail that was only nudged.
 */
inline constexpr float GRADIENT_FIRE_THRESHOLD = 0.5f;

// ---------------------------------------------------------------------------
// 3. Dual fast/slow history
// ---------------------------------------------------------------------------

/// Frames the FAST accumulator averages over. Four is short enough to follow a
/// step within the time a viewer reads as "immediate" and long enough that its
/// own sampling error (below) is not larger than the signal it is policing.
inline constexpr float FAST_HISTORY_WINDOW = 4.0f;

/// Multiples of the fast mean's own sampling standard deviation the main mean
/// may sit outside it before being clamped back. Two sigma: a main mean that is
/// where the fast mean says it should be, to within what four binary samples
/// can actually resolve, is left alone.
inline constexpr float FAST_HISTORY_CLAMP_SIGMA = 2.0f;

/// Absolute floor on that margin, in visibility units. Without it a region that
/// has been unanimously shadowed (or lit) for four frames would have a margin
/// of exactly zero and would pin the main mean to a binary value, undoing the
/// penumbra the accumulator exists to build.
inline constexpr float FAST_HISTORY_CLAMP_FLOOR = 0.05f;

/**
 * How far the MAIN accumulated mean may sit from the FAST mean before it is
 * clamped back toward it.
 *
 * The fast mean is the mean of `count` Bernoulli samples of the same
 * visibility, so its standard error is sqrt(p(1-p)/count) with p estimated by
 * the mean itself. Deriving the margin from that rather than fixing it is what
 * makes the clamp do its job at both ends of the range at once:
 *
 *  - a fully shadowed region (p = 0) has zero sampling error, so the margin
 *    collapses to the floor and a stale bright main history is dragged down
 *    within the fast window instead of over `max_accum_frames`;
 *  - a genuine half-penumbra (p = 0.5, count 4) has an error of 0.25, so the
 *    margin opens to 0.55 and the main mean — which is a BETTER estimate of the
 *    same p, from more samples — is not clamped toward a noisier one.
 *
 * That asymmetry is the whole point: the fast history is a change DETECTOR, not
 * a better estimator, and the margin is what stops it being mistaken for one.
 */
[[nodiscard]] inline float fast_history_margin(float fast_mean, float fast_count) noexcept {
    const float p = std::clamp(fast_mean, 0.0f, 1.0f);
    const float n = std::max(fast_count, 1.0f);
    const float sigma = std::sqrt(std::max(p * (1.0f - p), 0.0f) / n);
    return FAST_HISTORY_CLAMP_SIGMA * sigma + FAST_HISTORY_CLAMP_FLOOR;
}

// ---------------------------------------------------------------------------
// 4. Exposure-aware clamps
// ---------------------------------------------------------------------------

/**
 * Scene-radiance units per unit of POST-EXPOSURE (perceptual) luminance.
 *
 * The composite pass multiplies scene radiance by exp2(adapted EV) before
 * tonemapping, so a threshold that is meant to be constant to the VIEWER is a
 * scene-radiance threshold of `perceptual / exp2(EV)`. The day-to-night swing
 * is ~10^3 in scene radiance and ~0 in perceptual terms; every absolute
 * threshold in a denoiser therefore has to be multiplied by this or it means
 * something different at each end of the cycle.
 */
[[nodiscard]] inline float exposure_scene_scale(float adapted_ev) noexcept {
    return std::exp2(-adapted_ev);
}

/**
 * Post-exposure luminance difference below which two radiances are the same
 * colour to a viewer — one 255th of the tonemapper's mid-grey anchor. Anything
 * a denoiser is asked to distinguish below this is noise by definition.
 */
inline constexpr float DENOISE_PERCEPTUAL_EPSILON = 0.004f;

/**
 * Floor added to the reflection denoiser's neighbourhood standard deviation
 * before the sigma-clamp box is built.
 *
 * The sigma-clamp is scale-free and therefore already exposure-invariant — but
 * only where sigma is nonzero. In a flat region (a smooth wall reflecting an
 * even sky) sigma collapses toward zero, the box collapses onto this frame's
 * one stochastic sample, and the accumulation is thrown away every frame at
 * exactly the pixels where it would have converged best. A floor fixes that,
 * and the floor is the one part of the clamp that is an ABSOLUTE luminance and
 * therefore has to be told what the exposure is.
 */
[[nodiscard]] inline float clamp_sigma_floor(float adapted_ev) noexcept {
    return DENOISE_PERCEPTUAL_EPSILON * exposure_scene_scale(adapted_ev);
}

/// Multiples of the neighbourhood standard deviation above its mean at which
/// this frame's sample is treated as a firefly and rescaled. Wider than the
/// history clamp's sigma, because rejecting a bright sample the neighbourhood
/// genuinely contains costs energy that never comes back.
inline constexpr float REFLECTION_FIREFLY_SIGMA = 4.0f;

/**
 * Absolute post-exposure luminance below which the firefly clamp does not
 * engage at all, whatever the neighbourhood says.
 *
 * A dark neighbourhood has a tiny mean and a tiny sigma, so a purely relative
 * ceiling would crush the first genuinely lit sample to arrive — a lantern
 * coming into view in a reflection would be denoised out of existence.
 *
 * 2.0 is deliberately high: post-exposure, that is already past what the
 * tonemapper resolves as white, so a sample has to be both far outside its
 * neighbourhood's statistics AND brighter than anything a viewer could
 * distinguish before this engages. A firefly in a 1-spp GGX trace is two to
 * three orders of magnitude above the surrounding lobe, so it clears this
 * easily; a legitimately bright reflection does not. That asymmetry is what
 * lets the clamp be added to a shipping image without moving it.
 */
inline constexpr float REFLECTION_FIREFLY_PERCEPTUAL_FLOOR = 2.0f;

/**
 * Ceiling this frame's reflection sample luminance is rescaled to.
 *
 * `max` of the relative and absolute terms, so neither can make the other
 * stricter: a bright neighbourhood raises the ceiling by its own statistics,
 * and a dark one is prevented from lowering it below what a viewer could see.
 */
[[nodiscard]] inline float reflection_firefly_ceiling(float mean_luma, float sigma_luma,
                                                      float adapted_ev) noexcept {
    const float relative = mean_luma + REFLECTION_FIREFLY_SIGMA * sigma_luma;
    const float absolute = REFLECTION_FIREFLY_PERCEPTUAL_FLOOR * exposure_scene_scale(adapted_ev);
    return std::max(relative, absolute);
}

/**
 * Floor under the screen-probe filter's firefly ceiling
 * (`shaders/gi_screen_probe_filter.comp.hlsl`), which until R5b clamped purely
 * against the probe's own mean and said so in a comment naming this phase.
 *
 * Same argument as the reflection floor, one tier lower: a probe in a dark
 * interior has a mean near zero, and `fireflyMax * mean` would rescale the one
 * ray that found the doorway down to the darkness around it. This is the
 * absolute term that keeps the clamp honest at low probe means, and it is
 * exposure-relative so that the day and night ends of the cycle clamp the same
 * fraction of rays.
 */
inline constexpr float SCREEN_PROBE_FIREFLY_PERCEPTUAL_FLOOR = 0.25f;

[[nodiscard]] inline float screen_probe_firefly_ceiling(float probe_mean_luma, float firefly_max,
                                                        float adapted_ev) noexcept {
    return std::max(firefly_max * probe_mean_luma,
                    SCREEN_PROBE_FIREFLY_PERCEPTUAL_FLOOR * exposure_scene_scale(adapted_ev));
}

} // namespace bracken::render
