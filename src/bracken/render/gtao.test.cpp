#include "bracken/render/gtao.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace bracken::render;
using bracken::gpu::BlurUniforms;
using bracken::gpu::GTAOUniforms;

namespace {

constexpr float PI = 3.14159265358979323846f;
constexpr float HALF_PI = PI / 2.0f;
constexpr float TOLERANCE = 1e-5f;

} // namespace

// =============================================================================
// detail::gtao_slice_visibility — CPU reference for the shader's
// cosine-weighted horizon arc integration (gtao.frag.hlsl). Analytic
// expectations follow from the closed-form integral in Jimenez et al. 2016.
// =============================================================================

// Per-slice "no occlusion" horizon cosines: the hemisphere around the
// projected normal ends at angle n - pi/2 on the negative march side and
// n + pi/2 on the positive side (matches lowHorizonCos0/1 in gtao.frag.hlsl).
float unoccluded_low0(float n) {
    return std::cos(n - HALF_PI);
}
float unoccluded_low1(float n) {
    return std::cos(n + HALF_PI);
}

// Surface facing the camera head-on (projected normal along V, n = 0) with
// unraised horizons: the full hemisphere is visible, visibility = 1.
TEST(GtaoSliceVisibility, UnoccludedFacingSurfaceIsOne) {
    EXPECT_NEAR(detail::gtao_slice_visibility(unoccluded_low0(0.0f), unoccluded_low1(0.0f), 0.0f), 1.0f, TOLERANCE);
}

// Both horizons raised all the way to the view vector (cos = 1): the arc
// collapses, nothing is visible.
TEST(GtaoSliceVisibility, FullyRaisedHorizonsAreZero) {
    EXPECT_NEAR(detail::gtao_slice_visibility(1.0f, 1.0f, 0.0f), 0.0f, TOLERANCE);
}

// One side fully occluded, the other fully open, facing surface: exactly half
// the cosine-weighted visibility remains.
TEST(GtaoSliceVisibility, HalfOccludedIsHalf) {
    EXPECT_NEAR(detail::gtao_slice_visibility(1.0f, 0.0f, 0.0f), 0.5f, TOLERANCE);
    EXPECT_NEAR(detail::gtao_slice_visibility(0.0f, 1.0f, 0.0f), 0.5f, TOLERANCE);
}

// Raising a horizon must never increase visibility (monotonicity of the arc
// integral in each horizon angle).
TEST(GtaoSliceVisibility, MonotonicInHorizons) {
    float prev = 2.0f;
    for (int i = 0; i <= 10; ++i) {
        const float horizon_cos = static_cast<float>(i) / 10.0f; // 0 (open) .. 1 (closed)
        const float v = detail::gtao_slice_visibility(horizon_cos, 0.0f, 0.0f);
        EXPECT_LE(v, prev + TOLERANCE) << "horizon_cos " << horizon_cos;
        prev = v;
    }
}

// Tilted projected normal (n != 0), unraised horizons: the closed-form
// per-slice unoccluded value is cos(n) + n*sin(n) (Jimenez et al. eq. 9 with
// h0 = n - pi/2, h1 = n + pi/2). Per-slice values can exceed 1 — the paper's
// normalization holds after integrating projLen-weighted slices over phi.
TEST(GtaoSliceVisibility, UnoccludedTiltedSurfaceMatchesClosedForm) {
    for (const float n : {-0.8f, -0.3f, 0.3f, 0.8f}) {
        const float expected = std::cos(n) + n * std::sin(n);
        EXPECT_NEAR(detail::gtao_slice_visibility(unoccluded_low0(n), unoccluded_low1(n), n), expected, 1e-4f)
            << "n " << n;
    }
}

// Horizons are clamped to the hemisphere around the projected normal: any
// horizon inputs must land between zero and the unoccluded value for that n
// (raising a horizon can only remove visibility; lowering it past the
// hemisphere edge cannot add any).
TEST(GtaoSliceVisibility, ClampedToHemisphereStaysInRange) {
    for (const float n : {-1.2f, 0.0f, 1.2f}) {
        const float unoccluded = detail::gtao_slice_visibility(unoccluded_low0(n), unoccluded_low1(n), n);
        for (int i = 0; i <= 8; ++i) {
            const float c = -1.0f + 2.0f * static_cast<float>(i) / 8.0f;
            const float v = detail::gtao_slice_visibility(c, c, n);
            EXPECT_GE(v, -TOLERANCE) << "n " << n << " cos " << c;
            EXPECT_LE(v, unoccluded + 1e-4f) << "n " << n << " cos " << c;
        }
    }
}

// =============================================================================
// Uniform tuning invariants
// =============================================================================

// The AO pass fades to 1.0 past GTAOUniforms::maxDistance and the bilateral
// blur passes through beyond BlurUniforms::maxDistance — the two compile-time
// constants must stay equal or the blur breaks at the fade boundary. Neither
// is assigned at runtime (only radiusScale/strength/temporal jitter are), so
// the NSDMI defaults are the live values.
TEST(GtaoUniformDefaults, AoAndBlurMaxDistanceAgree) {
    const GTAOUniforms ao{};
    const BlurUniforms blur{};
    EXPECT_FLOAT_EQ(ao.maxDistance, blur.maxDistance);
}
