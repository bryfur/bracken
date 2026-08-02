#include "bracken/render/rt_denoise_math.hpp"

#include "bracken/render/gi/gi_math.hpp"

#include <gtest/gtest.h>

#include <cmath>

// The R5b denoiser core (docs/rt_lighting_plan.md §5.6). Every curve here is
// transliterated into shaders/rt_denoise.hlsli, where it cannot be tested; this
// is the half that can be. The properties pinned below are the ones the
// transliteration would silently break — identity at rest, the saturation
// anchors, and the exposure invariance that is the entire point of §5.6's
// exposure-aware clamps.

namespace bracken::render {
namespace {

constexpr float PI = 3.14159265358979323846f;

/// A unit direction at (elevation, azimuth) in degrees, in the engine's Y-up
/// frame. Returned as the direction light TRAVELS, i.e. sun_dir's convention.
glm::vec3 sun_dir_at(float elevation_deg, float azimuth_deg) {
    const float el = elevation_deg * PI / 180.0f;
    const float az = azimuth_deg * PI / 180.0f;
    return -glm::vec3(std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az));
}

// ---------------------------------------------------------------------------
// 1. The analytic sun gradient
// ---------------------------------------------------------------------------

TEST(RtDenoiseMath, StaticSunProducesExactlyZero) {
    const glm::vec3 dir = sun_dir_at(55.0f, 20.0f);
    const glm::vec3 radiance{3.0f, 2.85f, 2.7f};
    // Not "small": ZERO. The whole bit-identical-when-nothing-changed claim
    // rests on this being an exact 0 that lerps every accumulator by 0.
    EXPECT_EQ(sun_change_signal(dir, dir, radiance, radiance), 0.0f);
    EXPECT_EQ(lighting_reset_floor(sun_change_signal(dir, dir, radiance, radiance)), 0.0f);
}

TEST(RtDenoiseMath, RealTimeOfDayRatesAreInsideTheDeadZone) {
    // 360 degrees in 24 hours is 0.0042 deg/s; at 60 Hz that is 7e-5 deg/frame,
    // three orders of magnitude under the dead zone. A game with a real day
    // cycle must never pay for this mechanism.
    const glm::vec3 radiance{3.0f, 3.0f, 3.0f};
    const float per_frame = 360.0f / (24.0f * 3600.0f) / 60.0f;
    EXPECT_EQ(sun_change_signal(sun_dir_at(55.0f, 0.0f), sun_dir_at(55.0f, per_frame), radiance,
                                radiance),
              0.0f);
}

TEST(RtDenoiseMath, SunSignalSaturatesAtTheSolarDiskRadius) {
    const glm::vec3 radiance{3.0f, 3.0f, 3.0f};
    // Elevation change, so the angular delta IS the parameter (an azimuth change
    // at high elevation subtends less).
    const auto signal_at = [&](float delta_deg) {
        return sun_change_signal(sun_dir_at(0.0f, 0.0f), sun_dir_at(delta_deg, 0.0f), radiance,
                                 radiance);
    };
    EXPECT_EQ(signal_at(SUN_GRADIENT_ANGLE_DEADZONE_DEG * 0.5f), 0.0f);
    EXPECT_NEAR(signal_at(SUN_GRADIENT_ANGLE_SATURATION_DEG), 1.0f, 1e-3f);
    EXPECT_EQ(signal_at(SUN_GRADIENT_ANGLE_SATURATION_DEG * 4.0f), 1.0f);
    // Strictly increasing across the ramp, which is what makes the response to a
    // constant-rate sweep proportional to the rate.
    EXPECT_LT(signal_at(0.10f), signal_at(0.20f));
    EXPECT_LT(signal_at(0.20f), signal_at(0.30f));
}

TEST(RtDenoiseMath, SunSignalIsInvariantToDirectionMagnitude) {
    const glm::vec3 radiance{1.0f, 1.0f, 1.0f};
    const glm::vec3 a = sun_dir_at(30.0f, 0.0f);
    const glm::vec3 b = sun_dir_at(30.3f, 0.0f);
    // Loose on purpose: the ramp's input is an acos of a cosine within 1e-5 of
    // 1, where single-precision acos loses about half its digits. The claim is
    // that the SCALE does not matter, not that the arithmetic is exact — a
    // scale-sensitive implementation would differ by the ratio, i.e. by ~1.
    EXPECT_NEAR(sun_change_signal(a, b, radiance, radiance),
                sun_change_signal(a * 17.0f, b * 0.03f, radiance, radiance), 5e-3f);
}

TEST(RtDenoiseMath, IntensityAndColourDimmingAreTheSameEvent) {
    const glm::vec3 dir = sun_dir_at(55.0f, 20.0f);
    const glm::vec3 base{2.0f, 2.0f, 2.0f};
    // Halving the intensity and halving the colour produce the same radiance,
    // and must therefore produce the same signal — which is why the caller
    // passes the PRODUCT.
    EXPECT_EQ(sun_change_signal(dir, dir, base, base * 0.5f),
              sun_change_signal(dir, dir, base, glm::vec3(1.0f, 1.0f, 1.0f)));
}

TEST(RtDenoiseMath, LightningSpikeSaturatesTheColourChannel) {
    const glm::vec3 dir = sun_dir_at(20.0f, 0.0f);
    const glm::vec3 dim{0.2f, 0.2f, 0.25f};
    EXPECT_EQ(sun_change_signal(dir, dir, dim, dim * 6.0f), 1.0f);
    // ...and a 1% weather nudge does not.
    EXPECT_EQ(sun_change_signal(dir, dir, dim, dim * 1.01f), 0.0f);
}

TEST(RtDenoiseMath, ChannelsCombineWithMaxNotSum) {
    const glm::vec3 radiance{1.0f, 1.0f, 1.0f};
    const glm::vec3 a = sun_dir_at(0.0f, 0.0f);
    const glm::vec3 b = sun_dir_at(0.2f, 0.0f);
    const float angle_only = sun_change_signal(a, b, radiance, radiance);
    const float colour_only = sun_change_signal(a, a, radiance, radiance * 1.12f);
    const float both = sun_change_signal(a, b, radiance, radiance * 1.12f);
    EXPECT_GT(angle_only, 0.0f);
    EXPECT_GT(colour_only, 0.0f);
    EXPECT_FLOAT_EQ(both, std::max(angle_only, colour_only));
    EXPECT_LT(both, angle_only + colour_only); // a sum would have over-reacted
}

TEST(RtDenoiseMath, ResetFloorClampsAndIsTheIdentityInRange) {
    EXPECT_EQ(lighting_reset_floor(0.0f), 0.0f);
    EXPECT_EQ(lighting_reset_floor(0.37f), 0.37f);
    EXPECT_EQ(lighting_reset_floor(1.0f), 1.0f);
    EXPECT_EQ(lighting_reset_floor(4.0f), 1.0f);
    EXPECT_EQ(lighting_reset_floor(-1.0f), 0.0f);
}

// ---------------------------------------------------------------------------
// 2. The traced sparse gradient
// ---------------------------------------------------------------------------

TEST(RtDenoiseMath, GradientStratumIsTheOneInNineBudget) {
    // The plan's budget is "~1/9 extra rays per enabled signal", and the stratum
    // is the only thing that sets it.
    EXPECT_EQ(GRADIENT_STRATUM * GRADIENT_STRATUM, 9);
}

TEST(RtDenoiseMath, DilationWeightsFallOffByChebyshevRingAndStopAtTwo) {
    EXPECT_EQ(gradient_dilation_weight(0, 0), GRADIENT_DILATION_RING0);
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            EXPECT_EQ(gradient_dilation_weight(dx, dy), GRADIENT_DILATION_RING1);
        }
    }
    EXPECT_EQ(gradient_dilation_weight(2, 0), GRADIENT_DILATION_RING2);
    EXPECT_EQ(gradient_dilation_weight(-2, 2), GRADIENT_DILATION_RING2);
    EXPECT_EQ(gradient_dilation_weight(3, 0), 0.0f);
    EXPECT_EQ(gradient_dilation_weight(0, -9), 0.0f);

    EXPECT_GT(GRADIENT_DILATION_RING0, GRADIENT_DILATION_RING1);
    EXPECT_GT(GRADIENT_DILATION_RING1, GRADIENT_DILATION_RING2);
    // A dilation may never manufacture signal: the largest weight is 1, so the
    // reconstructed gradient is bounded by the largest raw verdict it saw.
    EXPECT_EQ(GRADIENT_DILATION_RING0, 1.0f);
}

TEST(RtDenoiseMath, FireThresholdBracketsTheDilationCore) {
    // Load-bearing, and measured. Below the WEAKEST ring weight, one raw stratum
    // verdict lights its whole 5x5 footprint above the line and the reported
    // fire fraction measures the kernel's width instead of the scene's activity
    // — on the playground bench that inflated the figure ~25x and flattened its
    // response to the sweep rate to nothing. At or below the ring-1 weight it
    // counts the 3x3 core of each detection, which is the region that actually
    // lost its accumulation.
    EXPECT_GT(GRADIENT_FIRE_THRESHOLD, GRADIENT_DILATION_RING2);
    EXPECT_LE(GRADIENT_FIRE_THRESHOLD, GRADIENT_DILATION_RING1);
    EXPECT_LE(GRADIENT_FIRE_THRESHOLD, GRADIENT_DILATION_RING0);
}

TEST(RtDenoiseMath, GradientDepthTestIsFarTighterThanTheDenoisers) {
    // 0.05 is the temporal pass's own `depthRejectRelative`. The gradient asks a
    // strictly stronger question and must not inherit the looser answer.
    EXPECT_LT(GRADIENT_DEPTH_REJECT_RELATIVE, 0.05f / 10.0f);
    EXPECT_GT(GRADIENT_DEPTH_REJECT_RELATIVE, 0.0f);
}

// ---------------------------------------------------------------------------
// 3. Dual fast/slow history
// ---------------------------------------------------------------------------

TEST(RtDenoiseMath, FastMarginCollapsesToTheFloorAtAUnanimousMean) {
    // The case the mechanism exists for: four frames all shadowed, so the fast
    // mean has no sampling error at all and a stale bright main history is
    // dragged down immediately instead of over max_accum_frames.
    EXPECT_FLOAT_EQ(fast_history_margin(0.0f, FAST_HISTORY_WINDOW), FAST_HISTORY_CLAMP_FLOOR);
    EXPECT_FLOAT_EQ(fast_history_margin(1.0f, FAST_HISTORY_WINDOW), FAST_HISTORY_CLAMP_FLOOR);
}

TEST(RtDenoiseMath, FastMarginOpensWidestAtAHalfPenumbra) {
    const float half = fast_history_margin(0.5f, FAST_HISTORY_WINDOW);
    EXPECT_GT(half, fast_history_margin(0.1f, FAST_HISTORY_WINDOW));
    EXPECT_GT(half, fast_history_margin(0.9f, FAST_HISTORY_WINDOW));
    // Four binary samples cannot resolve a p of 0.5 to better than 0.25, so the
    // margin must exceed that or the clamp would pull the MAIN mean — the better
    // estimate, from more samples — toward the noisier one.
    EXPECT_GT(half, 0.25f);
}

TEST(RtDenoiseMath, FastMarginNarrowsAsTheFastCountGrows) {
    EXPECT_GT(fast_history_margin(0.5f, 1.0f), fast_history_margin(0.5f, 4.0f));
    EXPECT_GT(fast_history_margin(0.5f, 4.0f), fast_history_margin(0.5f, 16.0f));
    // Never below the floor, whatever the count.
    EXPECT_GE(fast_history_margin(0.5f, 1e6f), FAST_HISTORY_CLAMP_FLOOR);
}

TEST(RtDenoiseMath, FastMarginIsRobustToOutOfRangeInputs) {
    // A half-float history round-trip can land marginally outside [0, 1]; the
    // margin must not become NaN from a negative variance.
    EXPECT_TRUE(std::isfinite(fast_history_margin(-0.01f, 4.0f)));
    EXPECT_TRUE(std::isfinite(fast_history_margin(1.01f, 4.0f)));
    EXPECT_TRUE(std::isfinite(fast_history_margin(0.5f, 0.0f)));
}

TEST(RtDenoiseMath, FastWindowIsShorterThanTheMainOne) {
    // 32 is RtShadowSettings::max_accum_frames. A "fast" history that was not
    // materially faster would detect nothing the main one had not already.
    EXPECT_LT(FAST_HISTORY_WINDOW, 32.0f / 4.0f);
}

// ---------------------------------------------------------------------------
// 4. Exposure-aware clamps
// ---------------------------------------------------------------------------

TEST(RtDenoiseMath, ExposureSceneScaleInvertsTheCompositeMultiply) {
    EXPECT_FLOAT_EQ(exposure_scene_scale(0.0f), 1.0f);
    EXPECT_FLOAT_EQ(exposure_scene_scale(1.0f), 0.5f);
    EXPECT_FLOAT_EQ(exposure_scene_scale(-2.0f), 4.0f);
    // The property that matters: a scene radiance times exp2(EV) is the
    // post-exposure luminance, so a post-exposure threshold T maps to
    // T * exposure_scene_scale(EV) in scene units, and back again exactly.
    for (float ev : {-8.0f, -1.5f, 0.0f, 2.25f, 9.0f}) {
        const float scene = 0.7f * exposure_scene_scale(ev);
        EXPECT_NEAR(scene * std::exp2(ev), 0.7f, 1e-4f);
    }
}

TEST(RtDenoiseMath, ClampAndFireflyThresholdsAreConstantInPerceptualTerms) {
    // The day/night swing is ~10 EV. Every absolute threshold must land on the
    // same POST-EXPOSURE number at both ends, which is the whole claim of §5.6's
    // exposure-aware clamps.
    const float day = 1.0f;
    const float night = -9.0f;
    EXPECT_NEAR(clamp_sigma_floor(day) * std::exp2(day), DENOISE_PERCEPTUAL_EPSILON, 1e-5f);
    EXPECT_NEAR(clamp_sigma_floor(night) * std::exp2(night), DENOISE_PERCEPTUAL_EPSILON, 1e-5f);

    // A dark neighbourhood, so the ceiling is the absolute term at both ends.
    EXPECT_NEAR(reflection_firefly_ceiling(0.0f, 0.0f, day) * std::exp2(day),
                REFLECTION_FIREFLY_PERCEPTUAL_FLOOR, 1e-3f);
    EXPECT_NEAR(reflection_firefly_ceiling(0.0f, 0.0f, night) * std::exp2(night),
                REFLECTION_FIREFLY_PERCEPTUAL_FLOOR, 1e-3f);
}

TEST(RtDenoiseMath, FireflyCeilingTakesWhicheverTermIsLooser) {
    // A bright neighbourhood raises the ceiling by its own statistics...
    const float bright = reflection_firefly_ceiling(40.0f, 5.0f, 0.0f);
    EXPECT_FLOAT_EQ(bright, 40.0f + REFLECTION_FIREFLY_SIGMA * 5.0f);
    // ...and a dark one cannot lower it below what a viewer could see.
    EXPECT_FLOAT_EQ(reflection_firefly_ceiling(1e-6f, 0.0f, 0.0f),
                    REFLECTION_FIREFLY_PERCEPTUAL_FLOOR);
    // Never stricter than either term alone.
    for (float mean : {0.0f, 0.1f, 1.0f, 30.0f}) {
        for (float ev : {-4.0f, 0.0f, 3.0f}) {
            const float c = reflection_firefly_ceiling(mean, 0.25f, ev);
            EXPECT_GE(c, mean + REFLECTION_FIREFLY_SIGMA * 0.25f);
            EXPECT_GE(c, REFLECTION_FIREFLY_PERCEPTUAL_FLOOR * exposure_scene_scale(ev));
        }
    }
}

TEST(RtDenoiseMath, ScreenProbeCeilingOnlyEverLoosensTheRelativeOne) {
    // R4b's clamp was `fireflyMax * mean` with a `mean > 1e-6` guard. R5b adds a
    // floor and removes the guard, so the new ceiling must never be BELOW the
    // old one — otherwise the screen-probe image would have moved.
    for (float mean : {0.0f, 1e-7f, 0.01f, 0.5f, 12.0f}) {
        EXPECT_GE(screen_probe_firefly_ceiling(mean, 8.0f, 0.5f), 8.0f * mean);
    }
    EXPECT_GT(screen_probe_firefly_ceiling(0.0f, 8.0f, 0.0f), 0.0f);
}

// ---------------------------------------------------------------------------
// 5. The probe field's share of the same reset floor
// ---------------------------------------------------------------------------

TEST(RtDenoiseMath, ProbeBlendResetIsAnUpperBoundNotASecondTerm) {
    using gi::PROBE_HYSTERESIS_MAX;
    using gi::PROBE_HYSTERESIS_MIN;
    using gi::probe_blend;
    using gi::probe_blend_hysteresis;

    // Reset 0 is the pre-R5b curve exactly, at every delta.
    for (float delta : {0.0f, 0.05f, 0.4f, 1.0f, 3.0f}) {
        EXPECT_FLOAT_EQ(probe_blend_hysteresis(delta, 0.0f), probe_blend_hysteresis(delta));
    }
    EXPECT_FLOAT_EQ(probe_blend_hysteresis(0.0f, 0.0f), PROBE_HYSTERESIS_MAX);

    // A full reset pins the floor however small the probe's own observed delta
    // is — the case a probe that has not yet re-traced under the new sun is in.
    EXPECT_FLOAT_EQ(probe_blend_hysteresis(0.0f, 1.0f), PROBE_HYSTERESIS_MIN);
    // ...and never UNDOES a collapse the probe's own delta already caused.
    EXPECT_FLOAT_EQ(probe_blend_hysteresis(2.0f, 0.0f), PROBE_HYSTERESIS_MIN);

    // Monotone in the reset, and the blend follows it toward the new estimate.
    const glm::vec3 old_value{1.0f, 1.0f, 1.0f};
    const glm::vec3 estimate{1.02f, 1.02f, 1.02f};
    const glm::vec3 slow = probe_blend(old_value, estimate, 0.0f);
    const glm::vec3 fast = probe_blend(old_value, estimate, 1.0f);
    EXPECT_GT(fast.x, slow.x);
    EXPECT_LT(fast.x, estimate.x);
}

} // namespace
} // namespace bracken::render
