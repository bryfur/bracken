#include "bracken/render/auto_exposure.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>

using namespace bracken::render;

namespace {

// Convenience: a zeroed histogram with `count` pixels dropped into the bin
// for `luma`.
std::array<uint32_t, EXPOSURE_HISTOGRAM_BINS> histogram_of(float luma, uint32_t count) {
    std::array<uint32_t, EXPOSURE_HISTOGRAM_BINS> bins{};
    bins[exposure_bin_from_luma(luma)] = count;
    return bins;
}

} // namespace

// =============================================================================
// Bin mapping (mirror of exposure_histogram.comp.hlsl binFromLuma)
// =============================================================================

TEST(ExposureBinMapping, BlackAndBelowRangeGoToBinZero) {
    EXPECT_EQ(exposure_bin_from_luma(0.0f), 0u);
    // Just below the bottom of the range (EV -10).
    EXPECT_EQ(exposure_bin_from_luma(std::exp2(-10.0f) * 0.99f), 0u);
}

TEST(ExposureBinMapping, RangeEndpoints) {
    // Bottom of the range maps to bin 1, top (EV +16) and beyond clamp to 255.
    EXPECT_EQ(exposure_bin_from_luma(std::exp2(-10.0f)), 1u);
    EXPECT_EQ(exposure_bin_from_luma(std::exp2(16.0f)), 255u);
    EXPECT_EQ(exposure_bin_from_luma(1e9f), 255u);
}

TEST(ExposureBinMapping, MidRangeRoundTripsThroughBinLog2) {
    // For any in-range luma, the bin's lower-edge log2 must be within one bin
    // width (26 EV / 254 bins) below the true log2.
    constexpr float bin_width_ev = 26.0f / 254.0f;
    for (float ev = -9.5f; ev <= 15.5f; ev += 0.7f) {
        const uint32_t bin = exposure_bin_from_luma(std::exp2(ev));
        ASSERT_GE(bin, 1u);
        const float edge = exposure_bin_log2_luma(bin);
        EXPECT_LE(edge, ev + 1e-4f) << "ev=" << ev;
        EXPECT_GT(edge, ev - bin_width_ev - 1e-4f) << "ev=" << ev;
    }
}

TEST(ExposureBinMapping, MonotonicInLuma) {
    uint32_t prev = 0;
    for (float ev = -11.0f; ev <= 17.0f; ev += 0.25f) {
        const uint32_t bin = exposure_bin_from_luma(std::exp2(ev));
        EXPECT_GE(bin, prev);
        prev = bin;
    }
}

// =============================================================================
// Percentile-clipped reduce (mirror of exposure_adapt.comp.hlsl)
// =============================================================================

TEST(ExposureTargetEv, UniformSceneConvergesLumaToTargetGray) {
    // Every pixel at one luminance: target EV must map that luminance onto
    // the target gray (up to the bin quantization, one bin = ~0.102 EV).
    const float log2_target_gray = -2.0f;
    for (float scene_ev : {-4.0f, -1.0f, 0.0f, 2.5f}) {
        auto bins = histogram_of(std::exp2(scene_ev), 100000);
        const float target =
            exposure_target_ev(bins, AutoExposure::DEFAULT_LOW_PERCENTILE, AutoExposure::DEFAULT_HIGH_PERCENTILE,
                               log2_target_gray);
        EXPECT_NEAR(target, log2_target_gray - scene_ev, 26.0f / 254.0f + 1e-3f) << "scene_ev=" << scene_ev;
    }
}

TEST(ExposureTargetEv, EmptyHistogramReturnsZero) {
    std::array<uint32_t, EXPOSURE_HISTOGRAM_BINS> bins{};
    EXPECT_FLOAT_EQ(exposure_target_ev(bins, 0.6f, 0.05f, -1.0f), 0.0f);
}

TEST(ExposureTargetEv, AllBlackReturnsZero) {
    // Bin 0 counts toward the percentile total but never contributes
    // luminance; an all-black frame has no signal -> neutral EV.
    std::array<uint32_t, EXPOSURE_HISTOGRAM_BINS> bins{};
    bins[0] = 500000;
    EXPECT_FLOAT_EQ(exposure_target_ev(bins, 0.6f, 0.05f, -1.0f), 0.0f);
}

TEST(ExposureTargetEv, HighPercentileClipsBrightOutliers) {
    // 96% of pixels at EV 0, 4% at EV +12 (a sun disc): with the default 5%
    // bright clip the outliers are fully ignored.
    auto bins = histogram_of(1.0f, 96000);
    bins[exposure_bin_from_luma(std::exp2(12.0f))] = 4000;
    const float with_clip = exposure_target_ev(bins, 0.0f, 0.05f, 0.0f);
    // Reference: the same scene without the outliers at all.
    auto clean = histogram_of(1.0f, 96000);
    const float reference = exposure_target_ev(clean, 0.0f, 0.05f, 0.0f);
    EXPECT_NEAR(with_clip, reference, 1e-3f);

    // Without clipping the outliers drag the average up (target EV down).
    const float no_clip = exposure_target_ev(bins, 0.0f, 0.0f, 0.0f);
    EXPECT_LT(no_clip, with_clip - 0.3f);
}

TEST(ExposureTargetEv, LowPercentileClipsDarkPixels) {
    // 55% deep-shadow pixels at EV -8, 45% subject at EV 0: the default 60%
    // dark clip swallows the shadows entirely (55% < 60% budget, and the
    // remaining 5% of the budget eats into the subject).
    auto bins = histogram_of(std::exp2(-8.0f), 55000);
    bins[exposure_bin_from_luma(1.0f)] = 45000;
    const float target = exposure_target_ev(bins, 0.6f, 0.0f, 0.0f);
    // All surviving weight is at EV 0 -> target == log2_target_gray - 0 == 0
    // (up to bin quantization).
    EXPECT_NEAR(target, 0.0f, 26.0f / 254.0f + 1e-3f);
}

TEST(ExposureTargetEv, StraddlingBinContributesInsidePortionOnly) {
    // Two equal bins; clip exactly the first one via low_percentile = 0.5.
    auto bins = histogram_of(std::exp2(-4.0f), 1000);
    bins[exposure_bin_from_luma(std::exp2(4.0f))] = 1000;
    const float target = exposure_target_ev(bins, 0.5f, 0.0f, 0.0f);
    // Only the EV+4 bin survives.
    EXPECT_NEAR(target, -exposure_bin_log2_luma(exposure_bin_from_luma(std::exp2(4.0f))), 1e-3f);
}

// =============================================================================
// Temporal adaptation (mirror of the exposure_adapt.comp.hlsl tail)
// =============================================================================

TEST(ExposureAdapt, RateLimitedInBothDirections) {
    // Brightening (target above current) moves at speed_up EV/s.
    EXPECT_FLOAT_EQ(exposure_adapt(0.0f, 3.0f, 0.1f, 1.5f, 2.5f), 0.15f);
    // Darkening moves at speed_down EV/s.
    EXPECT_FLOAT_EQ(exposure_adapt(0.0f, -3.0f, 0.1f, 1.5f, 2.5f), -0.25f);
}

TEST(ExposureAdapt, LandsExactlyOnTargetWithinOneStep) {
    EXPECT_FLOAT_EQ(exposure_adapt(0.0f, 0.05f, 0.1f, 1.5f, 2.5f), 0.05f);
    EXPECT_FLOAT_EQ(exposure_adapt(0.0f, -0.1f, 0.1f, 1.5f, 2.5f), -0.1f);
    // Already converged: no movement.
    EXPECT_FLOAT_EQ(exposure_adapt(1.2f, 1.2f, 0.1f, 1.5f, 2.5f), 1.2f);
}

TEST(ExposureAdapt, ConvergesWithinExpectedTime) {
    // A -1 EV scene change (sun dimmed) at the default 1.5 EV/s up-speed must
    // re-converge in ~0.67 s of simulated time; the smoke driver's sun swing
    // relies on "visibly re-converged within ~2 s".
    float current = 0.0f;
    const float target = 1.0f;
    float t = 0.0f;
    const float dt = 1.0f / 60.0f;
    while (std::abs(target - current) > 0.01f && t < 5.0f) {
        current = exposure_adapt(current, target, dt, 1.5f, 2.5f);
        t += dt;
    }
    EXPECT_NEAR(current, target, 0.01f);
    EXPECT_LT(t, 0.75f);
}

TEST(ExposureAdapt, ZeroSpeedFreezesExposure) {
    EXPECT_FLOAT_EQ(exposure_adapt(0.5f, 3.0f, 0.1f, 0.0f, 0.0f), 0.5f);
}
