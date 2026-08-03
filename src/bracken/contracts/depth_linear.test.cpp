#include "bracken/contracts/depth_linear.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

using bracken::scene::depth_linear_terms;
using bracken::scene::linear_view_depth;

namespace {

/// The engine's projection: glm RH_ZO with near/far SWAPPED, so near -> device
/// depth 1 and far -> 0 (CLAUDE.md, systems/camera_system.cpp).
glm::mat4 reversed_z_projection(float near_plane, float far_plane, float fov_deg = 60.0f) {
    return glm::perspective(glm::radians(fov_deg), 16.0f / 9.0f, far_plane, near_plane);
}

/// GROUND TRUTH, and deliberately a different algorithm: unproject the device
/// depth through the inverse projection, exactly as lighting.hlsli's
/// `viewPosFromDepth` does in the shaders that get depth right the expensive
/// way. Returns positive view depth.
float reference_view_depth(const glm::mat4& projection, float device_depth) {
    const glm::mat4 inv = glm::inverse(projection);
    const glm::vec4 clip(0.0f, 0.0f, device_depth, 1.0f);
    const glm::vec4 view = inv * clip;
    return -view.z / view.w;
}

/// The device depth a surface at positive view depth `t` rasterizes to.
float device_depth_for(const glm::mat4& projection, float t) {
    const glm::vec4 clip = projection * glm::vec4(0.0f, 0.0f, -t, 1.0f);
    return clip.z / clip.w;
}

/// The formula four traced passes carried until 2026-08-02: the sign of the
/// `a` term inverted, with the clamp that used to hide the blow-up.
float old_broken_linearize(float device_depth, float a, float b) {
    return b / std::max(device_depth - a, 1e-6f);
}

constexpr std::array<float, 4> REPRESENTATIVE_DEPTHS = {10.0f, 100.0f, 1000.0f, 4000.0f};

} // namespace

// ---------------------------------------------------------------------------
// The fix: the helper IS the inverse projection, to well under 0.1%.
// ---------------------------------------------------------------------------

TEST(DepthLinear, MatchesInverseProjectionAcrossTheDepthRange) {
    // The engine default (camera_controller.hpp near_plane = 5,
    // DrawDistance::D4000) plus the two extremes of the DrawDistance table.
    const std::array<glm::mat4, 3> projections = {
        reversed_z_projection(5.0f, 4000.0f),
        reversed_z_projection(0.1f, 500.0f),
        reversed_z_projection(5.0f, 32000.0f),
    };

    for (const glm::mat4& projection : projections) {
        const auto terms = depth_linear_terms(projection);
        for (float t : REPRESENTATIVE_DEPTHS) {
            const float device = device_depth_for(projection, t);
            if (device < 0.0f || device > 1.0f) {
                continue; // outside this projection's range
            }
            const float reference = reference_view_depth(projection, device);
            const float linearized = linear_view_depth(device, terms);
            EXPECT_NEAR(linearized, reference, reference * 1e-3f)
                << "t=" << t << " device=" << device;
            // And it round-trips the depth it was asked about.
            EXPECT_NEAR(linearized, t, t * 1e-3f) << "t=" << t;
        }
    }
}

TEST(DepthLinear, PinsTheProjectionPlanes) {
    const glm::mat4 projection = reversed_z_projection(5.0f, 4000.0f);
    const auto terms = depth_linear_terms(projection);
    // Reversed Z: device depth 1 is the near plane, 0 the far plane. This is
    // the identity scene_renderer.cpp recovers the shadow near plane from.
    EXPECT_NEAR(linear_view_depth(1.0f, terms), 5.0f, 5.0f * 1e-4f);
    EXPECT_NEAR(linear_view_depth(0.0f, terms), 4000.0f, 4000.0f * 1e-4f);
    // b / a == far exactly, which is the identity the error curve below leans on.
    EXPECT_NEAR(terms.b / terms.a, 4000.0f, 4000.0f * 1e-4f);
}

TEST(DepthLinear, HoldsForStandardZToo) {
    // Not swapped: near -> 0, far -> 1. Same two terms, same formula — that is
    // why the engine needs only one.
    const glm::mat4 projection = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 5.0f, 4000.0f);
    const auto terms = depth_linear_terms(projection);
    for (float t : REPRESENTATIVE_DEPTHS) {
        const float device = device_depth_for(projection, t);
        EXPECT_NEAR(linear_view_depth(device, terms), t, t * 1e-3f) << "t=" << t;
    }
}

TEST(DepthLinear, DegenerateTermsReturnZeroNotNaN) {
    // A pass that never filled its uniforms: the depth-aware upsamples want
    // equal weights (plain bilinear), not a NaN through their weight sum.
    EXPECT_EQ(linear_view_depth(0.5f, 0.0f, 0.0f), 0.0f);
    EXPECT_EQ(linear_view_depth(0.0f, 0.0f, 0.0f), 0.0f);
    EXPECT_TRUE(std::isfinite(linear_view_depth(0.0f, 0.0f, 1.0f)));
}

// ---------------------------------------------------------------------------
// The defect, recorded executably: what the old formula did and by how much.
// ---------------------------------------------------------------------------

/**
 * The magnitude claim, as a test rather than a commit message.
 *
 * Substituting d = b/t - a into `b / (d - a)` and using a/b = 1/far gives
 *
 *     t_wrong = t / (1 - 2 * t / far)
 *
 * — an overestimate that GROWS with distance, blows up at far/2 and is pinned
 * to one constant beyond it by the old `max(.., 1e-6)` clamp. R2b measured
 * "~33% at typical scene distances"; that is the far/8 point, 500 units at the
 * default draw distance.
 */
TEST(DepthLinear, OldFormulaErrorCurveIsTOverOneMinusTwoTOverFar) {
    constexpr float near_plane = 5.0f;
    constexpr float far_plane = 4000.0f;
    const glm::mat4 projection = reversed_z_projection(near_plane, far_plane);
    const auto terms = depth_linear_terms(projection);

    for (float t : {10.0f, 100.0f, 500.0f, 1000.0f, 1900.0f}) {
        const float device = device_depth_for(projection, t);
        const float fixed = linear_view_depth(device, terms);
        const float broken = old_broken_linearize(device, terms.a, terms.b);
        const float predicted = t / (1.0f - 2.0f * t / far_plane);

        EXPECT_NEAR(fixed, t, t * 1e-3f);
        EXPECT_NEAR(broken, predicted, predicted * 5e-3f) << "t=" << t;

        // Emitted so the magnitude is readable in the test log, not only
        // asserted: `ctest -R DepthLinear --output-on-failure` / --verbose.
        testing::Test::RecordProperty("t_" + std::to_string(static_cast<int>(t)),
                                      std::to_string(broken / t));
        std::printf("[depth_linear] t=%7.1f  fixed=%9.2f  old=%12.2f  old/true=%.4f\n",
                    static_cast<double>(t), static_cast<double>(fixed),
                    static_cast<double>(broken), static_cast<double>(broken / t));
    }

    // The four headline numbers, pinned.
    const auto ratio_at = [&](float t) {
        return old_broken_linearize(device_depth_for(projection, t), terms.a, terms.b) / t;
    };
    EXPECT_NEAR(ratio_at(10.0f), 1.0050f, 1e-3f);   //   +0.5%
    EXPECT_NEAR(ratio_at(100.0f), 1.0526f, 1e-3f);  //   +5.3%
    EXPECT_NEAR(ratio_at(500.0f), 1.3333f, 2e-3f);  //  +33.3%  <- R2b's figure
    EXPECT_NEAR(ratio_at(1000.0f), 2.0000f, 5e-3f); // +100.0%
}

TEST(DepthLinear, OldFormulaSaturatedEverythingPastHalfTheDrawDistance) {
    constexpr float far_plane = 4000.0f;
    const glm::mat4 projection = reversed_z_projection(5.0f, far_plane);
    const auto terms = depth_linear_terms(projection);

    // Past far/2 the denominator turns non-positive and the clamp pins every
    // surface to the SAME number. Two surfaces 1500 units apart compared equal,
    // which is why the traced denoisers' depth rejection quietly stopped
    // discriminating out there rather than misbehaving visibly.
    const float a = old_broken_linearize(device_depth_for(projection, 2500.0f), terms.a, terms.b);
    const float b = old_broken_linearize(device_depth_for(projection, 3900.0f), terms.a, terms.b);
    EXPECT_FLOAT_EQ(a, b);
    EXPECT_GT(a, 1e5f);

    // The fixed helper keeps them 1400 units apart, as they are.
    const float fixed_a = linear_view_depth(device_depth_for(projection, 2500.0f), terms);
    const float fixed_b = linear_view_depth(device_depth_for(projection, 3900.0f), terms);
    EXPECT_NEAR(fixed_b - fixed_a, 1400.0f, 5.0f);
}
