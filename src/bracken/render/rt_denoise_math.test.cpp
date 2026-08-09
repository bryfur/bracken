#include "bracken/render/rt_denoise_math.hpp"

#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/render/gi/gi_math.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>

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
    for (const float knee :
         {SUN_GRADIENT_DEPTH_REJECT_RELATIVE, LOCAL_GRADIENT_DEPTH_REJECT_RELATIVE}) {
        EXPECT_LT(knee, 0.05f / 10.0f);
        EXPECT_GT(knee, 0.0f);
    }
}

TEST(RtDenoiseMath, SunAndLocalGradientKneesAreDeliberatelyDifferent) {
    // Pinned as exact values, not bounds, because the divergence is a decision
    // and the failure mode it guards against is silent drift back together.
    //
    // The sun replay linearizes device depth through depth_linear.hlsli, so the
    // 2026-08-02 sign fix changed its ruler and its knee was RE-MEASURED to
    // 0.0004. The local replay compares two true view depths taken straight from
    // the view matrix and never touched that formula, so its 0.0005 was never
    // mis-scaled and re-syncing it would tighten an unbroken test by 20%.
    EXPECT_FLOAT_EQ(SUN_GRADIENT_DEPTH_REJECT_RELATIVE, 0.0004f);
    EXPECT_FLOAT_EQ(LOCAL_GRADIENT_DEPTH_REJECT_RELATIVE, 0.0005f);
    EXPECT_LT(SUN_GRADIENT_DEPTH_REJECT_RELATIVE, LOCAL_GRADIENT_DEPTH_REJECT_RELATIVE);
}

TEST(RtDenoiseMath, GradientUniformDefaultsCarryTheirOwnConstants) {
    // gpu/ sits below render/ and cannot include this header, so both gradient
    // cbuffers carry their knee as a literal default. That is two definitions of
    // a measured constant, and the drift it invites is not hypothetical: the
    // sun's default was the TEMPORAL pass's 0.05 — the one value that field must
    // never hold — kept alive only because rt_renderer.cpp always overwrites it.
    EXPECT_FLOAT_EQ(gpu::RtShadowGradientUniforms{}.depthRejectRelative,
                    SUN_GRADIENT_DEPTH_REJECT_RELATIVE);
    EXPECT_FLOAT_EQ(gpu::RtLocalShadowGradientUniforms{}.depthRejectRelative,
                    LOCAL_GRADIENT_DEPTH_REJECT_RELATIVE);
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
// 3b. The visibility filter's radius driver
// ---------------------------------------------------------------------------
//
// The property these pin is the one the shader lacked for its whole life: for a
// ONE-BIT sample the stored moments are bit-identical, so `moment2 - mean^2` is
// p(1-p) and does not move as history accumulates. Anything that reintroduces
// that — dropping the count division, or "simplifying" the expression back to
// the two moments — fails here rather than silently pinning the filter's radius
// at its maximum again.

/// The temporal pass's recursion, exactly: seeded from one binary sample and
/// advanced with alpha = 1/count from the same operands for both moments.
struct BinaryAccumulator {
    float mean = 0.0f;
    float moment2 = 0.0f;
    float count = 0.0f;

    void push(float sample, float max_accum_frames) {
        if (count <= 0.0f) {
            mean = sample;
            moment2 = sample * sample;
            count = 1.0f;
            return;
        }
        count = std::min(count + 1.0f, max_accum_frames);
        const float alpha = 1.0f / count;
        mean += (sample - mean) * alpha;
        moment2 += (sample * sample - moment2) * alpha;
    }
};

TEST(RtDenoiseMath, FilterVarianceFallsAsHistoryAccumulates) {
    // A half-lit penumbra texel: alternating bits, so the mean converges to 0.5
    // and the SAMPLE variance parks at 0.25 forever. The estimator's must not.
    BinaryAccumulator acc;
    float previous = std::numeric_limits<float>::infinity();
    for (int frame = 0; frame < 64; ++frame) {
        acc.push(static_cast<float>(frame % 2), 32.0f);
        const float v = visibility_filter_variance(acc.mean, acc.moment2, acc.count);
        if (frame >= 2 && acc.count < 32.0f) {
            EXPECT_LT(v, previous) << "frame " << frame;
        }
        previous = v;
    }
    // The raw second-moment difference is what the shader used to read: still
    // ~p(1-p) after 64 frames, which is why the radius never came down.
    EXPECT_NEAR(acc.moment2 - acc.mean * acc.mean, 0.25f, 0.02f);
    // Divided by the accumulated count it is an order of magnitude smaller, and
    // below the 1/16 knee where the shader's `saturate(sqrt(v) * 4)` stops
    // saturating — i.e. the radius is finally free to move.
    EXPECT_LT(previous, 0.0625f);
}

TEST(RtDenoiseMath, FilterVarianceIsTheStandardErrorOfTheMean) {
    // p(1-p)/n, the same statistic fast_history_margin derives for the fast
    // window — expressed there as its square root.
    for (const float count : {1.0f, 4.0f, 32.0f}) {
        EXPECT_FLOAT_EQ(visibility_filter_variance(0.5f, 0.5f, count), 0.25f / count);
        const float sigma = fast_history_margin(0.5f, count) - FAST_HISTORY_CLAMP_FLOOR;
        EXPECT_NEAR(std::sqrt(visibility_filter_variance(0.5f, 0.5f, count)),
                    sigma / FAST_HISTORY_CLAMP_SIGMA, 1e-6f);
    }
}

TEST(RtDenoiseMath, FilterVarianceIsRobustToOutOfRangeInputs) {
    // The moments ride an RGBA16F history, so a round-trip can make `moment2`
    // marginally smaller than `mean^2`; a count of 0 is the first frame.
    EXPECT_FLOAT_EQ(visibility_filter_variance(0.5f, 0.24f, 8.0f), 0.0f);
    EXPECT_TRUE(std::isfinite(visibility_filter_variance(0.5f, 0.5f, 0.0f)));
    EXPECT_TRUE(std::isfinite(visibility_filter_variance(0.5f, 0.5f, -1.0f)));
    // A unanimous texel: zero variance at any count, so it is never blurred on
    // the strength of its moments — only the count term can widen it.
    EXPECT_FLOAT_EQ(visibility_filter_variance(1.0f, 1.0f, 1.0f), 0.0f);
    EXPECT_FLOAT_EQ(visibility_filter_variance(0.0f, 0.0f, 1.0f), 0.0f);
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

namespace {

/// Mean and standard deviation of `values`, exactly as the denoisers' 3x3
/// windows compute them: population moments, `sqrt(max(m2 - m1*m1, 0))`.
struct Window {
    float mean;
    float sigma;
};
[[nodiscard]] Window window_stats(std::span<const float> values) {
    float m1 = 0.0f;
    float m2 = 0.0f;
    for (const float v : values) {
        m1 += v;
        m2 += v * v;
    }
    const auto n = static_cast<float>(values.size());
    const float mean = m1 / n;
    return {.mean = mean, .sigma = std::sqrt(std::max(m2 / n - mean * mean, 0.0f))};
}

} // namespace

TEST(RtDenoiseMath, FireflyCeilingCannotFireAgainstAWindowContainingTheSample) {
    // The defect this pins, and the reason the firefly clamp has to take
    // LEAVE-ONE-OUT statistics (rt_di_denoise_temporal.comp.hlsl's RtDiWindow,
    // rt_reflection_denoise_temporal.comp.hlsl's `looMean`/`looSigma`).
    //
    // Samuelson's inequality: every member of an n-sample set lies within
    // `mean +- sigma*sqrt(n-1)`. For the 9 taps of a 3x3 window that is
    // `mean + 2.83*sigma`, which is strictly below the `mean + 4*sigma` the
    // relative ceiling asks for. So a sample tested against a window it is
    // itself a member of can NEVER be judged a firefly, whatever its value.
    const float samuelson = std::sqrt(8.0f); // sqrt(n - 1), n = 9
    EXPECT_LT(samuelson, REFLECTION_FIREFLY_SIGMA)
        << "the relative firefly test is only satisfiable if the sample is excluded";

    // The worst case the clamp exists for: one enormous sample among 8 dark
    // neighbours. Exposure 0 and a firefly far above the perceptual floor, so
    // the absolute term is not what decides this.
    const float firefly = 5000.0f;
    const std::array<float, 9> with_centre{firefly, 0.0f, 0.0f, 0.0f, 0.0f,
                                           0.0f,    0.0f, 0.0f, 0.0f};
    const std::array<float, 8> without_centre{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const Window full = window_stats(with_centre);
    const Window loo = window_stats(without_centre);

    // Included in its own statistics, the firefly sits comfortably under the
    // ceiling -- the branch is dead.
    EXPECT_GE(reflection_firefly_ceiling(full.mean, full.sigma, 0.0f), firefly);
    // Excluded, it is caught, and rescaled to the absolute perceptual floor.
    EXPECT_LT(reflection_firefly_ceiling(loo.mean, loo.sigma, 0.0f), firefly);
}

TEST(RtDenoiseMath, LeaveOneOutCeilingStillPassesASampleItsNeighboursSupport) {
    // The other half of the contract: excluding the centre must not turn the
    // clamp into an energy sink. A sample consistent with its neighbourhood
    // survives, so a genuinely bright reflection is not denoised out of
    // existence -- which is what REFLECTION_FIREFLY_PERCEPTUAL_FLOOR's comment
    // promises and what the `max` with the absolute term protects.
    const std::array<float, 8> neighbours{38.0f, 42.0f, 40.0f, 39.0f,
                                          41.0f, 40.0f, 43.0f, 37.0f};
    const Window loo = window_stats(neighbours);
    // mean 40, sigma sqrt(3.5) ~ 1.871, so the relative ceiling sits at
    // 40 + 4*1.871 ~ 47.5 -- comfortably above the window's own spread, which is
    // the headroom REFLECTION_FIREFLY_SIGMA's comment is buying.
    EXPECT_NEAR(loo.mean, 40.0f, 1e-4f);
    EXPECT_NEAR(loo.sigma, std::sqrt(3.5f), 1e-4f);
    for (const float sample : {38.0f, 40.0f, 44.0f, 46.0f}) {
        EXPECT_GE(reflection_firefly_ceiling(loo.mean, loo.sigma, 0.0f), sample)
            << "sample " << sample << " is within its neighbourhood and must not be clamped";
    }
    // Two orders of magnitude out is not, and that is the case R5b built it for.
    EXPECT_LT(reflection_firefly_ceiling(loo.mean, loo.sigma, 0.0f), 4000.0f);
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
