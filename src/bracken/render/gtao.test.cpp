#include "bracken/render/gtao.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace bracken::render;
using bracken::gpu::BlurUniforms;
using bracken::gpu::GTAOUniforms;

namespace {

constexpr float PI = 3.14159265358979323846f;
constexpr float HALF_PI = PI / 2.0f;
constexpr float TOLERANCE = 1e-5f;
constexpr int SECTORS = detail::GTAO_SECTOR_COUNT;

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
// SSILVB sector weights — the discrete form of the SAME cosine integrand.
// If these stop summing to the arc integral, the bitmask pass has silently
// become a differently-normalised AO and every tuning constant beside it lies.
// =============================================================================

TEST(GtaoSectorWeights, SumMatchesArcIntegral) {
    for (const float n : {-1.4f, -0.8f, -0.3f, 0.0f, 0.3f, 0.8f, 1.4f}) {
        float sum = 0.0f;
        for (int k = 0; k < SECTORS; ++k) {
            sum += detail::gtao_sector_weight(k, n);
        }
        EXPECT_NEAR(sum, detail::gtao_slice_visibility(unoccluded_low0(n), unoccluded_low1(n), n), 1e-4f)
            << "n " << n;
    }
}

// A weight is an integral of a non-negative integrand over a sector, so no
// sector may ever subtract visibility.
TEST(GtaoSectorWeights, AllStrictlyPositive) {
    for (const float n : {-1.4f, -0.5f, 0.0f, 0.5f, 1.4f}) {
        for (int k = 0; k < SECTORS; ++k) {
            EXPECT_GT(detail::gtao_sector_weight(k, n), 0.0f) << "n " << n << " sector " << k;
        }
    }
}

// With the projected normal along the view vector the hemicircle is symmetric
// about it, so sector k and its mirror must weigh the same.
TEST(GtaoSectorWeights, SymmetricAtZeroTilt) {
    for (int k = 0; k < SECTORS; ++k) {
        EXPECT_NEAR(detail::gtao_sector_weight(k, 0.0f), detail::gtao_sector_weight(SECTORS - 1 - k, 0.0f),
                    1e-6f)
            << "sector " << k;
    }
}

// =============================================================================
// detail::gtao_mark_sectors — which bits an angular span sets
// =============================================================================

TEST(GtaoMarkSectors, WholeHemicircleSetsEveryBit) {
    EXPECT_EQ(detail::gtao_mark_sectors(0.0f, 1.0f), 0xFFFFFFFFu);
}

// Zero width marks nothing. This is the path taken by a sample at the radius
// edge (the falloff has collapsed its span), by a coplanar neighbour, and by a
// background sample below the tangent plane — all three must be free.
TEST(GtaoMarkSectors, ZeroWidthMarksNothing) {
    EXPECT_EQ(detail::gtao_mark_sectors(0.0f, 0.0f), 0u);
    EXPECT_EQ(detail::gtao_mark_sectors(0.5f, 0.5f), 0u);
    EXPECT_EQ(detail::gtao_mark_sectors(1.0f, 1.0f), 0u);
    // Inverted (clamped past each other) is also nothing, never everything.
    EXPECT_EQ(detail::gtao_mark_sectors(0.7f, 0.3f), 0u);
}

TEST(GtaoMarkSectors, AlignedSpanSetsExactlyItsSectors) {
    // [0.5, 0.75] = sectors 16..23.
    const uint32_t bits = detail::gtao_mark_sectors(0.5f, 0.75f);
    EXPECT_EQ(bits, 0x00FF0000u);
    // [0, 1/32] = sector 0 alone.
    EXPECT_EQ(detail::gtao_mark_sectors(0.0f, 1.0f / static_cast<float>(SECTORS)), 0x1u);
    // The last sector alone.
    EXPECT_EQ(detail::gtao_mark_sectors(31.0f / static_cast<float>(SECTORS), 1.0f), 0x80000000u);
}

// A sub-sector occluder still costs a whole sector: the width rounds outward.
// (The START rounds inward, so a span can also land up to one sector short at
// its far end — the ±1-sector quantisation the brute-force test bounds.)
TEST(GtaoMarkSectors, SubSectorOccluderStillCostsASector) {
    const uint32_t bits = detail::gtao_mark_sectors(0.501f, 0.502f);
    EXPECT_EQ(bits, 1u << 16);
}

// A span pinned at the far edge must not wrap around to sector 0 — that would
// darken the wrong side of the hemisphere.
TEST(GtaoMarkSectors, EdgeSpanDoesNotWrap) {
    const uint32_t bits = detail::gtao_mark_sectors(0.99f, 1.0f);
    EXPECT_EQ(bits & 0x1u, 0u);
    EXPECT_NE(bits & 0x80000000u, 0u);
}

// =============================================================================
// detail::gtao_slab_span — the constant-thickness occluder model
// =============================================================================

namespace {

// Signed angle (from the view vector, positive toward the +omega march) of a
// point at `angle` from V on `side`, and the same point pushed `thickness`
// units further along the view ray. Mirrors what gtao.frag.hlsl computes in
// view space; the test drives the slice plane directly in 2D (V, tangent).
struct SlabSample {
    float front = 0.0f; // unsigned angle from V to the surface point
    float back = 0.0f;  // unsigned angle from V to the slab's far face
    float side = 1.0f;
};

SlabSample make_slab(float signed_angle, float distance, float thickness) {
    SlabSample s;
    s.side = signed_angle >= 0.0f ? 1.0f : -1.0f;
    s.front = std::abs(signed_angle);
    // Point at (distance, s.front) in (V, tangent) polar coordinates, pushed
    // by -V * thickness.
    const float along_v = distance * std::cos(s.front) - thickness;
    const float along_t = distance * std::sin(s.front);
    s.back = std::atan2(along_t, along_v);
    return s;
}

} // namespace

// A neighbour in the tangent plane sits exactly on the hemicircle edge, and
// its slab extends AWAY from the normal, off the end of the hemicircle. It must
// mark nothing at all — otherwise every flat surface would occlude itself.
TEST(GtaoSlabSpan, CoplanarNeighbourMarksNothing) {
    const float n = 0.0f; // normal along V, so the tangent plane is at ±90°
    for (const float sign : {-1.0f, 1.0f}) {
        const SlabSample s = make_slab(sign * HALF_PI, 3.0f, 1.0f);
        const auto span = detail::gtao_slab_span(s.front, s.back, s.side, n);
        EXPECT_EQ(detail::gtao_mark_sectors(span.lo, span.hi), 0u) << "sign " << sign;
    }
}

// A sample BELOW the tangent plane is background seen through a gap; it is not
// an occluder and never was, and the bitmask gets that for free rather than
// through a heuristic.
TEST(GtaoSlabSpan, BackgroundBelowTheTangentPlaneMarksNothing) {
    const float n = 0.0f;
    const SlabSample s = make_slab(HALF_PI + 0.4f, 8.0f, 1.0f);
    const auto span = detail::gtao_slab_span(s.front, s.back, s.side, n);
    EXPECT_EQ(detail::gtao_mark_sectors(span.lo, span.hi), 0u);
}

// An occluder above the tangent plane blocks a BAND: from its own angle out
// toward the horizon, ending where the slab ends. Not everything past it —
// that is the whole point of the upgrade.
TEST(GtaoSlabSpan, OccluderBlocksABandNotEverythingPastIt) {
    const float n = 0.0f;
    const SlabSample s = make_slab(0.9f, 4.0f, 1.0f);
    const auto span = detail::gtao_slab_span(s.front, s.back, s.side, n);
    const uint32_t bits = detail::gtao_mark_sectors(span.lo, span.hi);
    EXPECT_NE(bits, 0u);
    // Bits live strictly inside the upper half (the +omega side of a normal
    // aligned with V) and stop short of the horizon.
    EXPECT_EQ(bits & 0x0000FFFFu, 0u) << "leaked onto the other march side";
    EXPECT_LT(std::popcount(bits), SECTORS / 2) << "a finite slab must not swallow the whole side";
}

// The two march directions are mirror images: the same occluder geometry on
// the other side must occlude the mirrored sectors, and just as many of them.
TEST(GtaoSlabSpan, SidesMirror) {
    const float n = 0.0f;
    const SlabSample pos = make_slab(0.7f, 5.0f, 1.0f);
    const SlabSample neg = make_slab(-0.7f, 5.0f, 1.0f);
    const auto span_pos = detail::gtao_slab_span(pos.front, pos.back, pos.side, n);
    const auto span_neg = detail::gtao_slab_span(neg.front, neg.back, neg.side, n);
    EXPECT_NEAR(span_pos.lo - 0.5f, 0.5f - span_neg.hi, 1e-5f);
    EXPECT_NEAR(span_pos.hi - 0.5f, 0.5f - span_neg.lo, 1e-5f);
    EXPECT_EQ(std::popcount(detail::gtao_mark_sectors(span_pos.lo, span_pos.hi)),
              std::popcount(detail::gtao_mark_sectors(span_neg.lo, span_neg.hi)));
}

// THE REGRESSION THE PACKAGE EXISTS FOR: a thin occluder must occlude strictly
// less than a thick one at the same place. The pre-R5a horizon pair could not
// express this at all — every occluder blocked everything behind it, and
// `thinOccluderCompensation` guessed at a correction with no length in it.
TEST(GtaoSlabSpan, ThinnerOccluderOccludesStrictlyLess) {
    const float n = 0.0f;
    int previous = SECTORS + 1;
    for (const float thickness : {8.0f, 4.0f, 2.0f, 1.0f, 0.5f}) {
        const SlabSample s = make_slab(0.8f, 4.0f, thickness);
        const auto span = detail::gtao_slab_span(s.front, s.back, s.side, n);
        const int marked = std::popcount(detail::gtao_mark_sectors(span.lo, span.hi));
        EXPECT_LE(marked, previous) << "thickness " << thickness;
        previous = marked;
    }
    EXPECT_LT(previous, SECTORS / 2);
}

// =============================================================================
// detail::gtao_slice_visibility_bitmask
// =============================================================================

TEST(GtaoBitmaskVisibility, EmptyMaskMatchesTheUnoccludedArcIntegral) {
    for (const float n : {-1.2f, -0.4f, 0.0f, 0.4f, 1.2f}) {
        EXPECT_NEAR(detail::gtao_slice_visibility_bitmask(0u, n),
                    detail::gtao_slice_visibility(unoccluded_low0(n), unoccluded_low1(n), n), 1e-4f)
            << "n " << n;
    }
}

TEST(GtaoBitmaskVisibility, FullMaskIsZero) {
    for (const float n : {-1.2f, 0.0f, 1.2f}) {
        EXPECT_NEAR(detail::gtao_slice_visibility_bitmask(0xFFFFFFFFu, n), 0.0f, TOLERANCE) << "n " << n;
    }
}

// One march side fully blocked with the normal along V: exactly half.
TEST(GtaoBitmaskVisibility, HalfMaskIsHalfAtZeroTilt) {
    EXPECT_NEAR(detail::gtao_slice_visibility_bitmask(0x0000FFFFu, 0.0f), 0.5f, 1e-4f);
    EXPECT_NEAR(detail::gtao_slice_visibility_bitmask(0xFFFF0000u, 0.0f), 0.5f, 1e-4f);
}

// Setting a bit is occlusion; it can never brighten a pixel.
TEST(GtaoBitmaskVisibility, MonotonicInSetBits) {
    for (const float n : {-0.6f, 0.0f, 0.6f}) {
        uint32_t mask = 0u;
        float previous = detail::gtao_slice_visibility_bitmask(mask, n);
        for (int k = 0; k < SECTORS; ++k) {
            mask |= (1u << k);
            const float v = detail::gtao_slice_visibility_bitmask(mask, n);
            EXPECT_LE(v, previous + TOLERANCE) << "n " << n << " after sector " << k;
            previous = v;
        }
        EXPECT_NEAR(previous, 0.0f, TOLERANCE);
    }
}

// =============================================================================
// Brute force over synthetic depth profiles
//
// The reference path (slab spans -> bitmask -> weighted sum) integrates the
// cosine lobe over a QUANTISED visible set. The brute force integrates the same
// lobe over the EXACT visible set with 20k samples. They may differ only by the
// sector quantisation, which is bounded: marking rounds the width outward and
// the start inward, so each occluder can shift the boundary by at most one
// sector in either direction.
// =============================================================================

namespace {

float brute_force_visibility(const std::vector<std::pair<float, float>>& blocked_angles, float n) {
    constexpr int STEPS = 20000;
    const float lo = n - HALF_PI;
    const float step = PI / static_cast<float>(STEPS);
    float total = 0.0f;
    for (int i = 0; i < STEPS; ++i) {
        const float theta = lo + (static_cast<float>(i) + 0.5f) * step;
        bool occluded = false;
        for (const auto& [a, b] : blocked_angles) {
            if (theta >= a && theta <= b) {
                occluded = true;
                break;
            }
        }
        if (!occluded) {
            total += std::abs(std::sin(theta)) * std::cos(theta - n) * step;
        }
    }
    return total;
}

// Worst-case error budget: one sector of weight at each end of each occluder.
float quantisation_budget(float n, size_t occluders) {
    float widest = 0.0f;
    for (int k = 0; k < SECTORS; ++k) {
        widest = std::max(widest, detail::gtao_sector_weight(k, n));
    }
    return 2.0f * widest * static_cast<float>(occluders) + 2e-3f;
}

} // namespace

TEST(GtaoBitmaskBruteForce, MatchesExactIntegrationOverSyntheticProfiles) {
    std::mt19937 rng(20260802u); // fixed seed: this test is a gate, not a sampler
    std::uniform_real_distribution<float> angle(-HALF_PI + 0.05f, HALF_PI - 0.05f);
    std::uniform_real_distribution<float> dist(1.0f, 6.0f);
    std::uniform_real_distribution<float> tilt(-1.2f, 1.2f);
    std::uniform_int_distribution<int> count(1, 4);

    for (int trial = 0; trial < 200; ++trial) {
        const float n = tilt(rng);
        const int occluders = count(rng);

        uint32_t mask = 0u;
        std::vector<std::pair<float, float>> blocked;
        for (int i = 0; i < occluders; ++i) {
            const SlabSample s = make_slab(angle(rng), dist(rng), 1.0f);
            const auto span = detail::gtao_slab_span(s.front, s.back, s.side, n);
            mask |= detail::gtao_mark_sectors(span.lo, span.hi);
            // The same interval in raw signed angles, for the exact integral.
            blocked.emplace_back(s.side > 0.0f ? s.front : -s.back,
                                 s.side > 0.0f ? s.back : -s.front);
        }

        const float from_bitmask = detail::gtao_slice_visibility_bitmask(mask, n);
        const float exact = brute_force_visibility(blocked, n);
        EXPECT_NEAR(from_bitmask, exact, quantisation_budget(n, blocked.size()))
            << "trial " << trial << " n " << n << " occluders " << occluders;
    }
}

// No occluders at all: the two paths must agree to the numerical integration's
// own accuracy, with no quantisation slack spent.
TEST(GtaoBitmaskBruteForce, UnoccludedProfileNeedsNoSlack) {
    for (const float n : {-1.0f, 0.0f, 0.5f}) {
        EXPECT_NEAR(detail::gtao_slice_visibility_bitmask(0u, n), brute_force_visibility({}, n), 1e-3f)
            << "n " << n;
    }
}

// =============================================================================
// Bent cone: aperture from the visibility lobe's first moment
// =============================================================================

// A fully open cosine-weighted hemisphere has mean resultant length 2/3, and
// its aperture is the whole hemisphere — the value that also serves as the
// encoding's "no directional data" escape.
TEST(GtaoConeAperture, OpenHemisphereIsFullyOpen) {
    EXPECT_NEAR(detail::gtao_cone_cos_from_resultant(2.0f / 3.0f), 0.0f, 1e-5f);
    EXPECT_NEAR(detail::gtao_cone_aperture(2.0f / 3.0f), 1.0f, 1e-5f);
    // A SPLIT visible set (resultant below 2/3) has no cone; the widest one is
    // the honest answer, never a negative or wrapped-around aperture.
    EXPECT_NEAR(detail::gtao_cone_aperture(0.0f), 1.0f, 1e-5f);
    EXPECT_NEAR(detail::gtao_cone_aperture(0.4f), 1.0f, 1e-5f);
}

TEST(GtaoConeAperture, PerfectlyConcentratedLobeIsClosed) {
    EXPECT_NEAR(detail::gtao_cone_cos_from_resultant(1.0f), 1.0f, 1e-5f);
    EXPECT_NEAR(detail::gtao_cone_aperture(1.0f), 0.0f, 1e-5f);
}

// Tighter lobe -> narrower cone -> less ambient. Monotone with no inversion, or
// the gate would brighten pixels as they became more occluded.
TEST(GtaoConeAperture, NarrowsMonotonicallyWithConcentration) {
    float previous = 2.0f;
    for (int i = 0; i <= 20; ++i) {
        const float r = 2.0f / 3.0f + (1.0f / 3.0f) * static_cast<float>(i) / 20.0f;
        const float aperture = detail::gtao_cone_aperture(r);
        EXPECT_LE(aperture, previous + TOLERANCE) << "resultant " << r;
        EXPECT_GE(aperture, -TOLERANCE);
        EXPECT_LE(aperture, 1.0f + TOLERANCE);
        previous = aperture;
    }
}

// The fit inverts its own model: build the resultant length a cone of half
// angle alpha would produce, and the aperture must come back as sin^2(alpha).
TEST(GtaoConeAperture, InvertsTheConeModelItWasDerivedFrom) {
    for (int i = 1; i <= 17; ++i) {
        const float alpha = HALF_PI * static_cast<float>(i) / 18.0f;
        const float c = std::cos(alpha);
        const float r = (2.0f / 3.0f) * (1.0f + c + c * c) / (1.0f + c);
        EXPECT_NEAR(detail::gtao_cone_aperture(r), std::sin(alpha) * std::sin(alpha), 1e-4f)
            << "alpha " << alpha;
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
