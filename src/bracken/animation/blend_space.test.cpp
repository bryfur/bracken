// Blend-space weight math. Every degenerate case named in blend_space.hpp gets
// an assertion here, because "defined rather than UB" is only a claim until
// something checks it.

#include "bracken/animation/blend_space.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

using namespace bracken::animation;

namespace {

[[nodiscard]] float weight_of(std::span<const WeightedClip> out, int clip) {
    float w = 0.0f;
    for (const WeightedClip& c : out) {
        if (c.clip_index == clip) {
            w += c.weight;
        }
    }
    return w;
}

[[nodiscard]] float sum_of(std::span<const WeightedClip> out) {
    float w = 0.0f;
    for (const WeightedClip& c : out) {
        w += c.weight;
    }
    return w;
}

/// The canonical locomotion set the polar formulation exists for: a centre
/// sample meaning "not moving", plus eight directions on the unit circle.
BlendSpace2D make_locomotion_space() {
    BlendSpace2D s;
    EXPECT_TRUE(s.add_sample(0, glm::vec2(0.0f, 0.0f)));
    for (int k = 0; k < 7; ++k) {
        const float a = static_cast<float>(k) * (2.0f * glm::pi<float>() / 7.0f);
        EXPECT_TRUE(s.add_sample(1 + k, glm::vec2(std::cos(a), std::sin(a))));
    }
    return s; // 8 samples == k_max_blend_samples
}

} // namespace

// ---------------------------------------------------------------------------
// 1D
// ---------------------------------------------------------------------------

TEST(BlendSpace1D, EmptySpaceResolvesToNothing) {
    BlendSpace1D s;
    std::array<WeightedClip, 2> out{};
    EXPECT_EQ(s.resolve(0.0f, out), 0u);
    EXPECT_EQ(s.sample_count(), 0);
}

TEST(BlendSpace1D, ASingleSampleWinsEverywhere) {
    BlendSpace1D s;
    ASSERT_TRUE(s.add_sample(7, 100.0f, 1.25f));
    s.finalize();
    std::array<WeightedClip, 2> out{};
    for (const float x : {-1e6f, 0.0f, 100.0f, 1e6f}) {
        const std::uint32_t n = s.resolve(x, out);
        ASSERT_EQ(n, 1u) << "x=" << x;
        EXPECT_EQ(out[0].clip_index, 7);
        EXPECT_FLOAT_EQ(out[0].weight, 1.0f);
        EXPECT_FLOAT_EQ(out[0].speed_scale, 1.25f);
    }
}

TEST(BlendSpace1D, IsExactAtEverySample) {
    BlendSpace1D s;
    ASSERT_TRUE(s.add_sample(2, 400.0f)); // deliberately out of order
    ASSERT_TRUE(s.add_sample(0, 0.0f));
    ASSERT_TRUE(s.add_sample(1, 150.0f));
    s.finalize();
    ASSERT_EQ(s.sample_count(), 3);
    EXPECT_EQ(s.sample(0).clip_index, 0) << "finalize() must sort by position";
    EXPECT_EQ(s.sample(2).clip_index, 2);

    std::array<WeightedClip, 2> out{};
    for (int i = 0; i < s.sample_count(); ++i) {
        const std::uint32_t n = s.resolve(s.sample(i).position, out);
        ASSERT_GE(n, 1u);
        ASSERT_LE(n, 2u);
        EXPECT_NEAR(weight_of(std::span<const WeightedClip>(out.data(), n), s.sample(i).clip_index), 1.0f, 1e-6f)
            << "sample " << i;
        EXPECT_NEAR(sum_of(std::span<const WeightedClip>(out.data(), n)), 1.0f, 1e-6f);
    }
}

TEST(BlendSpace1D, InterpolatesTheBracketingPairAndPartitionsUnity) {
    BlendSpace1D s;
    ASSERT_TRUE(s.add_sample(0, 0.0f));
    ASSERT_TRUE(s.add_sample(1, 100.0f));
    ASSERT_TRUE(s.add_sample(2, 400.0f));
    s.finalize();

    std::array<WeightedClip, 2> out{};
    std::uint32_t n = s.resolve(25.0f, out);
    ASSERT_EQ(n, 2u);
    EXPECT_NEAR(weight_of(std::span<const WeightedClip>(out.data(), n), 0), 0.75f, 1e-6f);
    EXPECT_NEAR(weight_of(std::span<const WeightedClip>(out.data(), n), 1), 0.25f, 1e-6f);

    n = s.resolve(250.0f, out);
    ASSERT_EQ(n, 2u);
    EXPECT_NEAR(weight_of(std::span<const WeightedClip>(out.data(), n), 1), 0.5f, 1e-6f);
    EXPECT_NEAR(weight_of(std::span<const WeightedClip>(out.data(), n), 2), 0.5f, 1e-6f);

    for (int i = 0; i <= 1000; ++i) {
        const float x = -100.0f + static_cast<float>(i) * 0.6f; // spans well past both ends
        n = s.resolve(x, out);
        ASSERT_GE(n, 1u) << "x=" << x;
        EXPECT_NEAR(sum_of(std::span<const WeightedClip>(out.data(), n)), 1.0f, 1e-5f) << "x=" << x;
    }
}

TEST(BlendSpace1D, ClampsRatherThanExtrapolatingOutsideTheAuthoredRange) {
    BlendSpace1D s;
    ASSERT_TRUE(s.add_sample(0, 10.0f));
    ASSERT_TRUE(s.add_sample(1, 20.0f));
    s.finalize();
    std::array<WeightedClip, 2> out{};

    ASSERT_EQ(s.resolve(-500.0f, out), 1u);
    EXPECT_EQ(out[0].clip_index, 0);
    EXPECT_FLOAT_EQ(out[0].weight, 1.0f);

    ASSERT_EQ(s.resolve(500.0f, out), 1u);
    EXPECT_EQ(out[0].clip_index, 1);
    EXPECT_FLOAT_EQ(out[0].weight, 1.0f);
}

TEST(BlendSpace1D, CoincidentSamplesDoNotDivideByZero) {
    BlendSpace1D s;
    ASSERT_TRUE(s.add_sample(0, 5.0f));
    ASSERT_TRUE(s.add_sample(1, 5.0f));
    ASSERT_TRUE(s.add_sample(2, 9.0f));
    s.finalize();
    std::array<WeightedClip, 2> out{};
    const std::uint32_t n = s.resolve(5.0f, out);
    ASSERT_GE(n, 1u);
    EXPECT_TRUE(std::isfinite(sum_of(std::span<const WeightedClip>(out.data(), n))));
    EXPECT_NEAR(sum_of(std::span<const WeightedClip>(out.data(), n)), 1.0f, 1e-6f);
}

TEST(BlendSpace1D, RefusesSamplesPastTheBudgetAndNegativeClipIndices) {
    BlendSpace1D s;
    for (int i = 0; i < k_max_blend_samples; ++i) {
        ASSERT_TRUE(s.add_sample(i, static_cast<float>(i)));
    }
    EXPECT_FALSE(s.add_sample(99, 99.0f)) << "budget must be enforced";
    EXPECT_EQ(s.sample_count(), k_max_blend_samples);

    BlendSpace1D t;
    EXPECT_FALSE(t.add_sample(-1, 0.0f));
    EXPECT_EQ(t.sample_count(), 0);
}

// ---------------------------------------------------------------------------
// 2D
// ---------------------------------------------------------------------------

TEST(BlendSpace2D, EmptyAndSingleSampleSpacesAreDefined) {
    BlendSpace2D empty;
    std::array<WeightedClip, k_max_blend_samples> out{};
    EXPECT_EQ(empty.resolve(glm::vec2(1.0f, 2.0f), out), 0u);

    BlendSpace2D one;
    ASSERT_TRUE(one.add_sample(4, glm::vec2(3.0f, -1.0f)));
    for (const glm::vec2 p : {glm::vec2(0.0f), glm::vec2(3.0f, -1.0f), glm::vec2(-900.0f, 900.0f)}) {
        const std::uint32_t n = one.resolve(p, out);
        ASSERT_EQ(n, 1u);
        EXPECT_EQ(out[0].clip_index, 4);
        EXPECT_FLOAT_EQ(out[0].weight, 1.0f);
    }
}

TEST(BlendSpace2D, IsExactAtEverySample) {
    // The property the whole estimator is justified by: at p == p_k, v_kp is the
    // zero vector so w_k == 1, and for every other i the pair term against k
    // evaluates to exactly 0.
    const BlendSpace2D s = make_locomotion_space();
    std::array<WeightedClip, k_max_blend_samples> out{};
    for (int k = 0; k < s.sample_count(); ++k) {
        const std::uint32_t n = s.resolve(s.sample(k).position, out);
        ASSERT_GE(n, 1u) << "sample " << k;
        EXPECT_NEAR(weight_of(std::span<const WeightedClip>(out.data(), n), s.sample(k).clip_index), 1.0f, 1e-6f)
            << "sample " << k;
        for (std::uint32_t i = 0; i < n; ++i) {
            if (out[i].clip_index != s.sample(k).clip_index) {
                EXPECT_LT(out[i].weight, k_blend_weight_epsilon) << "sample " << k << " leaked into " << i;
            }
        }
    }
}

TEST(BlendSpace2D, TheCentreSampleWinsAtTheOrigin) {
    // The case that motivates DIRECTIONAL over cartesian: (0,0) means "not
    // moving", and the eight direction samples must contribute nothing there.
    const BlendSpace2D s = make_locomotion_space();
    std::array<WeightedClip, k_max_blend_samples> out{};
    const std::uint32_t n = s.resolve(glm::vec2(0.0f), out);
    ASSERT_GE(n, 1u);
    EXPECT_NEAR(weight_of(std::span<const WeightedClip>(out.data(), n), 0), 1.0f, 1e-5f);
}

TEST(BlendSpace2D, PartitionsUnityAcrossAGridWiderThanTheSamples) {
    const BlendSpace2D s = make_locomotion_space();
    std::array<WeightedClip, k_max_blend_samples> out{};
    for (int gy = 0; gy < 64; ++gy) {
        for (int gx = 0; gx < 64; ++gx) {
            const glm::vec2 p(-1.5f + 3.0f * static_cast<float>(gx) / 63.0f,
                              -1.5f + 3.0f * static_cast<float>(gy) / 63.0f);
            const std::uint32_t n = s.resolve(p, out);
            ASSERT_GE(n, 1u) << "(" << p.x << "," << p.y << ") produced no weights at all";
            const float sum = sum_of(std::span<const WeightedClip>(out.data(), n));
            ASSERT_TRUE(std::isfinite(sum)) << "(" << p.x << "," << p.y << ")";
            EXPECT_NEAR(sum, 1.0f, 1e-5f) << "(" << p.x << "," << p.y << ")";
        }
    }
}

TEST(BlendSpace2D, IsC0ButTheBoundIsLooseBecauseItIsNotC1) {
    // min() over pair constraints creates creases: the estimator is continuous
    // but its DERIVATIVE is not, so this asserts a bound on the step rather than
    // smoothness. The origin is excluded because polar coordinates are genuinely
    // singular there — the angular term's gradient goes as 1/|p| — which is a
    // property of the parameterisation, not a defect in the code.
    const BlendSpace2D s = make_locomotion_space();
    std::array<WeightedClip, k_max_blend_samples> a{}, b{};
    constexpr float k_delta = 1e-4f;
    float worst = 0.0f;

    for (int gy = 0; gy < 48; ++gy) {
        for (int gx = 0; gx < 48; ++gx) {
            const glm::vec2 p(-1.5f + 3.0f * static_cast<float>(gx) / 47.0f,
                              -1.5f + 3.0f * static_cast<float>(gy) / 47.0f);
            if (glm::length(p) < 0.05f) {
                continue;
            }
            const std::uint32_t na = s.resolve(p, a);
            const std::uint32_t nb = s.resolve(p + glm::vec2(k_delta, k_delta), b);
            for (int clip = 0; clip < s.sample_count(); ++clip) {
                const float wa = weight_of(std::span<const WeightedClip>(a.data(), na), clip);
                const float wb = weight_of(std::span<const WeightedClip>(b.data(), nb), clip);
                worst = std::max(worst, std::abs(wa - wb));
            }
        }
    }
    EXPECT_LT(worst, 0.05f) << "a step of " << k_delta << " moved a weight by " << worst
                            << " — the estimator has become discontinuous, not merely non-smooth";
}

TEST(BlendSpace2D, CoincidentSamplesShareTheirWeightDeterministically) {
    BlendSpace2D s;
    ASSERT_TRUE(s.add_sample(0, glm::vec2(1.0f, 0.0f)));
    ASSERT_TRUE(s.add_sample(1, glm::vec2(1.0f, 0.0f)));
    std::array<WeightedClip, k_max_blend_samples> out{};

    const std::uint32_t n = s.resolve(glm::vec2(1.0f, 0.0f), out);
    ASSERT_EQ(n, 2u);
    EXPECT_NEAR(out[0].weight, 0.5f, 1e-6f);
    EXPECT_NEAR(out[1].weight, 0.5f, 1e-6f);

    // Deterministic: the same query twice gives the same answer, bit for bit.
    std::array<WeightedClip, k_max_blend_samples> again{};
    ASSERT_EQ(s.resolve(glm::vec2(1.0f, 0.0f), again), 2u);
    EXPECT_EQ(again[0].weight, out[0].weight);
    EXPECT_EQ(again[1].weight, out[1].weight);
}

TEST(BlendSpace2D, EverySampleAtTheOriginGivesUniformWeights) {
    BlendSpace2D s;
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(s.add_sample(i, glm::vec2(0.0f)));
    }
    std::array<WeightedClip, k_max_blend_samples> out{};
    const std::uint32_t n = s.resolve(glm::vec2(0.3f, -0.2f), out);
    ASSERT_EQ(n, 4u);
    for (std::uint32_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i].weight, 0.25f, 1e-6f);
    }
}

TEST(BlendSpace2D, CollinearSamplesBehaveLikeAOneDimensionalSpace) {
    // The case a Delaunay triangulation cannot triangulate at all.
    BlendSpace2D s;
    ASSERT_TRUE(s.add_sample(0, glm::vec2(-2.0f, 0.0f)));
    ASSERT_TRUE(s.add_sample(1, glm::vec2(0.0f, 0.0f)));
    ASSERT_TRUE(s.add_sample(2, glm::vec2(2.0f, 0.0f)));
    std::array<WeightedClip, k_max_blend_samples> out{};

    for (int i = -30; i <= 30; ++i) {
        const glm::vec2 p(static_cast<float>(i) * 0.1f, 0.0f);
        const std::uint32_t n = s.resolve(p, out);
        ASSERT_GE(n, 1u);
        const float sum = sum_of(std::span<const WeightedClip>(out.data(), n));
        ASSERT_TRUE(std::isfinite(sum));
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }

    const std::uint32_t at_end = s.resolve(glm::vec2(2.0f, 0.0f), out);
    ASSERT_GE(at_end, 1u);
    EXPECT_NEAR(weight_of(std::span<const WeightedClip>(out.data(), at_end), 2), 1.0f, 1e-6f);
}

TEST(BlendSpace2D, AQueryFarOutsideTheCoverIsDominatedByTheSampleItPointsAt) {
    // This space is DIRECTIONAL, not nearest-neighbour, and the distinction is
    // the whole point of the test. There is no "outside the cover" fallback and
    // there must not be one: a nearest-sample snap would put a discontinuity
    // wherever the query crosses the perpendicular bisector, and gradient-band
    // interpolation is C0 everywhere precisely because it never snaps.
    BlendSpace2D s;
    ASSERT_TRUE(s.add_sample(0, glm::vec2(1.0f, 0.0f)));
    ASSERT_TRUE(s.add_sample(1, glm::vec2(-1.0f, 0.0f)));
    std::array<WeightedClip, k_max_blend_samples> out{};

    const std::uint32_t n = s.resolve(glm::vec2(1000.0f, 3.0f), out);
    ASSERT_GE(n, 1u) << "resolve must never return an empty set";
    EXPECT_NEAR(sum_of(std::span<const WeightedClip>(out.data(), n)), 1.0f, 1e-5f);

    // (1000, 3) is 0.003 rad off the +x axis, and the angular band between two
    // opposed samples spans pi — so sample 1 keeps exactly 0.003/pi of the
    // weight. Asserting 1.0 here would be asserting a snap that does not and
    // should not happen; what the direction buys is DOMINANCE, not exclusivity.
    const float w0 = weight_of(std::span<const WeightedClip>(out.data(), n), 0);
    EXPECT_GT(w0, 0.999f) << "(1000, 3) points essentially straight at (1, 0)";
    EXPECT_NEAR(w0, 1.0f - (3.0f / 1000.0f) / std::numbers::pi_v<float>, 2e-4f);

    // Exactly on the axis it IS exclusive, and that is the property a caller
    // can rely on: a query collinear with a sample resolves to that sample.
    const std::uint32_t on_axis = s.resolve(glm::vec2(1000.0f, 0.0f), out);
    ASSERT_EQ(on_axis, 1u);
    EXPECT_FLOAT_EQ(weight_of(std::span<const WeightedClip>(out.data(), on_axis), 0), 1.0f);
}

TEST(BlendSpace2D, RefusesSamplesPastTheBudget) {
    BlendSpace2D s = make_locomotion_space();
    EXPECT_EQ(s.sample_count(), k_max_blend_samples);
    EXPECT_FALSE(s.add_sample(42, glm::vec2(5.0f, 5.0f)));
    EXPECT_EQ(s.sample_count(), k_max_blend_samples);
}

// ---------------------------------------------------------------------------
// effective_duration / find_clip_index
// ---------------------------------------------------------------------------

TEST(BlendSpace, EffectiveDurationIsTheWeightedMeanDividedByEachSpeedScale) {
    std::array<WeightedClip, 2> a{};
    a[0] = WeightedClip{0, 0.75f, 1.0f, 1.0f};
    a[1] = WeightedClip{1, 0.25f, 2.0f, 2.0f}; // effective length 1.0
    EXPECT_NEAR(effective_duration(std::span<const WeightedClip>(a.data(), 2)), 1.0f, 1e-6f);

    a[1] = WeightedClip{1, 0.25f, 1.0f, 2.0f};
    EXPECT_NEAR(effective_duration(std::span<const WeightedClip>(a.data(), 2)), 0.75f * 1.0f + 0.25f * 2.0f,
                1e-6f);
}

TEST(BlendSpace, EffectiveDurationIsZeroWhenThereIsNothingToAdvance) {
    EXPECT_FLOAT_EQ(effective_duration({}), 0.0f);
    std::array<WeightedClip, 1> a{};
    a[0] = WeightedClip{0, 0.0f, 1.0f, 5.0f};
    EXPECT_FLOAT_EQ(effective_duration(std::span<const WeightedClip>(a.data(), 1)), 0.0f);
    a[0] = WeightedClip{0, 1.0f, 1.0f, 0.0f};
    EXPECT_FLOAT_EQ(effective_duration(std::span<const WeightedClip>(a.data(), 1)), 0.0f);
}

TEST(BlendSpace, FindClipIndexResolvesByNameOrReturnsMinusOne) {
    std::vector<AnimationClip> clips(2);
    clips[0].name = "Idle";
    clips[1].name = "Run";
    EXPECT_EQ(find_clip_index(clips, "Idle"), 0);
    EXPECT_EQ(find_clip_index(clips, "Run"), 1);
    EXPECT_EQ(find_clip_index(clips, "Sprint"), -1);
}

// ---------------------------------------------------------------------------
// The invariant `resolve` needs is now maintained by `add_sample`, not by the
// host remembering `finalize()`.
//
// Before this, a space built out of order and never finalized was a debug-only
// BRACKEN_ASSERT whose documented release behaviour was "garbage weights" — a
// host-boundary precondition with no defined answer in a shipped build.
// ---------------------------------------------------------------------------

TEST(BlendSpace1D, IsResolvableWithoutFinalizeEvenWhenBuiltOutOfOrder) {
    BlendSpace1D s;
    ASSERT_TRUE(s.add_sample(2, 400.0f));
    ASSERT_TRUE(s.add_sample(0, 0.0f));
    ASSERT_TRUE(s.add_sample(1, 150.0f));
    // Deliberately NO finalize() call.
    EXPECT_TRUE(s.finalized()) << "add_sample must leave the space resolvable";
    EXPECT_EQ(s.sample(0).clip_index, 0);
    EXPECT_EQ(s.sample(1).clip_index, 1);
    EXPECT_EQ(s.sample(2).clip_index, 2);

    std::array<WeightedClip, 2> out{};
    const std::uint32_t n = s.resolve(75.0f, out);
    ASSERT_EQ(n, 2u);
    EXPECT_NEAR(weight_of(std::span<const WeightedClip>(out.data(), n), 0), 0.5f, 1e-6f);
    EXPECT_NEAR(weight_of(std::span<const WeightedClip>(out.data(), n), 1), 0.5f, 1e-6f);
}

TEST(BlendSpace1D, FinalizeIsIdempotentAndAgreesWithTheSortedInsert) {
    // The sorted insert has to reproduce what the stable_sort produced for the
    // same insertion order, coincident positions included — that ordering is
    // what decides which of two equal-position samples wins a query.
    BlendSpace1D inserted;
    ASSERT_TRUE(inserted.add_sample(0, 5.0f));
    ASSERT_TRUE(inserted.add_sample(1, 5.0f));
    ASSERT_TRUE(inserted.add_sample(2, 1.0f));
    ASSERT_TRUE(inserted.add_sample(3, 5.0f));

    BlendSpace1D sorted = inserted;
    sorted.finalize();
    ASSERT_EQ(inserted.sample_count(), sorted.sample_count());
    for (int i = 0; i < inserted.sample_count(); ++i) {
        EXPECT_EQ(inserted.sample(i).clip_index, sorted.sample(i).clip_index) << "slot " << i;
        EXPECT_FLOAT_EQ(inserted.sample(i).position, sorted.sample(i).position) << "slot " << i;
    }
    // Equal positions keep insertion order: 2 (at 1.0) then 0, 1, 3 (all at 5.0).
    EXPECT_EQ(inserted.sample(0).clip_index, 2);
    EXPECT_EQ(inserted.sample(1).clip_index, 0);
    EXPECT_EQ(inserted.sample(3).clip_index, 3);
}

TEST(BlendSpace, EffectiveDurationUsesTheMAGNITUDEOfSpeedScale) {
    // A negative speed_scale means "play this sample backwards", not "negative
    // length". The signed division let one such sample cancel the others out,
    // drive the sum to <= 0, and freeze the whole layer — `effective_duration`
    // returning 0 is the caller's "do not advance" signal.
    std::array<WeightedClip, 2> a{};
    a[0] = WeightedClip{0, 0.5f, 1.0f, 2.0f};
    a[1] = WeightedClip{1, 0.5f, -1.0f, 2.0f};
    const float d = effective_duration(std::span<const WeightedClip>(a.data(), 2));
    EXPECT_NEAR(d, 2.0f, 1e-6f) << "a backwards sample must contribute its positive length";
    EXPECT_GT(d, 0.0f) << "the layer would be frozen";
}
