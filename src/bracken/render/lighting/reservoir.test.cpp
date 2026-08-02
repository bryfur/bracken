// Weighted reservoir sampling + RIS merge: the maths, pinned.
//
// The shipped pass is a shader, and a shader cannot be unit-tested here. What
// CAN be — and what is actually at risk — is the estimator itself: an off-by-one
// in the acceptance test, an M that is not summed on merge, a W that divides by
// the wrong pdf. None of those crash and none of them look obviously wrong on
// screen; they make the shadows a little too dark or a little too light,
// forever. So they are checked as numbers.
//
// render/lighting/reservoir.hpp is transliterated into rt_local_shadow.comp.hlsl
// operation for operation; keeping the two in step is a review obligation these
// tests make meaningful.

#include <gtest/gtest.h>

#include "bracken/render/lighting/reservoir.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using bracken::render::lighting::clamp_reservoir_history;
using bracken::render::lighting::remap_reservoir;
using bracken::render::lighting::Reservoir;
using bracken::render::lighting::ReservoirBuilder;
using bracken::render::lighting::RESERVOIR_NO_LIGHT;

/// A deterministic uniform stream. Deterministic on purpose: the smoke gate
/// depends on the shipped pass being replayable, and a test that leans on a
/// seeded generator is testing the same property.
class Uniforms {
public:
    explicit Uniforms(uint32_t seed) : rng_(seed) {}
    float next() { return dist_(rng_); }

private:
    std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_{0.0f, 1.0f};
};

} // namespace

// The defining property: after streaming a candidate set through a reservoir,
// P(candidate i survives) == w_i / sum(w). Anything else and the estimator
// weights the wrong lights.
TEST(Reservoir, SelectionIsProportionalToWeight) {
    const std::array<float, 4> weights{1.0f, 3.0f, 0.5f, 5.5f};
    const float total = 10.0f;

    constexpr int TRIALS = 200000;
    std::array<int, 4> hits{};
    Uniforms u(12345);
    for (int t = 0; t < TRIALS; ++t) {
        ReservoirBuilder r;
        for (uint32_t i = 0; i < weights.size(); ++i) {
            r.update(i, weights[i], weights[i], u.next());
        }
        ASSERT_NE(r.r.light_index, RESERVOIR_NO_LIGHT);
        ++hits[r.r.light_index];
    }
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const double observed = static_cast<double>(hits[i]) / TRIALS;
        const double expected = weights[i] / total;
        EXPECT_NEAR(observed, expected, 0.01) << "candidate " << i;
    }
}

// A zero-weight candidate can never be selected, but it MUST still advance M.
// Skipping it entirely would be the tempting optimization and it biases W:
// M is the divisor of the contribution weight, so a dropped candidate inflates
// every estimate at that pixel.
TEST(Reservoir, ZeroWeightCandidatesCountTowardMButAreNeverSelected) {
    Uniforms u(7);
    ReservoirBuilder r;
    r.update(10u, 0.0f, 0.0f, u.next());
    r.update(11u, 2.0f, 2.0f, u.next());
    r.update(12u, 0.0f, 0.0f, u.next());
    EXPECT_EQ(r.r.light_index, 11u);
    EXPECT_FLOAT_EQ(r.r.sample_count, 3.0f);
    EXPECT_FLOAT_EQ(r.w_sum, 2.0f);
}

// A NaN weight is discarded rather than propagated. It cannot arise from the
// shipped target function, but the comparison it fails (`weight > 0`) is the
// one line that decides, and a reservoir that adopts a NaN W poisons every
// subsequent frame through the temporal chain.
TEST(Reservoir, NanWeightIsDiscarded) {
    Uniforms u(9);
    ReservoirBuilder r;
    r.update(4u, std::nanf(""), std::nanf(""), u.next());
    r.update(5u, 1.0f, 1.0f, u.next());
    r.finalize();
    EXPECT_EQ(r.r.light_index, 5u);
    EXPECT_FALSE(std::isnan(r.r.weight));
    EXPECT_FLOAT_EQ(r.r.sample_count, 2.0f);
}

// W = w_sum / (M * p̂(y)), and it is what makes the estimate unbiased. Checked
// against the closed form rather than against a remembered number.
TEST(Reservoir, FinalizeComputesTheUnbiasedContributionWeight) {
    // Force the selection by giving one candidate all the weight.
    ReservoirBuilder r;
    r.update(0u, 4.0f, 2.0f, 0.0f); // rand 0 always accepts
    r.update(1u, 0.0f, 0.0f, 0.9f);
    r.update(2u, 0.0f, 0.0f, 0.9f);
    r.finalize();
    ASSERT_EQ(r.r.light_index, 0u);
    // w_sum = 4, M = 3, p̂(y) = 2  =>  W = 4 / (3 * 2)
    EXPECT_FLOAT_EQ(r.r.weight, 4.0f / 6.0f);
}

// Nothing selected (every candidate zero) must yield W = 0, not a division by
// zero. A NaN here would reach the shader's ratio output and then the denoiser.
TEST(Reservoir, FinalizeWithNoSelectionYieldsZeroWeight) {
    ReservoirBuilder r;
    r.update(0u, 0.0f, 0.0f, 0.5f);
    r.update(1u, 0.0f, 0.0f, 0.5f);
    r.finalize();
    EXPECT_EQ(r.r.light_index, RESERVOIR_NO_LIGHT);
    EXPECT_FLOAT_EQ(r.r.weight, 0.0f);
    EXPECT_FALSE(std::isnan(r.r.weight));
}

// THE CLAIM THAT MAKES TEMPORAL REUSE LEGAL: streaming candidates A then B
// through one reservoir, and merging a reservoir over A with a reservoir over
// B, must select from the same distribution. This is the whole reason a
// per-pixel reservoir can be carried across frames instead of recomputed.
TEST(Reservoir, StreamingMergeMatchesBatchWeights) {
    const std::array<float, 3> first{2.0f, 1.0f, 1.0f};   // indices 0,1,2
    const std::array<float, 3> second{4.0f, 0.5f, 1.5f};  // indices 3,4,5
    const float total = 10.0f;

    constexpr int TRIALS = 200000;
    std::array<int, 6> batch_hits{};
    std::array<int, 6> merged_hits{};

    Uniforms ub(2024);
    for (int t = 0; t < TRIALS; ++t) {
        ReservoirBuilder r;
        for (uint32_t i = 0; i < 3; ++i) {
            r.update(i, first[i], first[i], ub.next());
        }
        for (uint32_t i = 0; i < 3; ++i) {
            r.update(i + 3, second[i], second[i], ub.next());
        }
        ++batch_hits[r.r.light_index];
    }

    Uniforms um(2025);
    for (int t = 0; t < TRIALS; ++t) {
        ReservoirBuilder a;
        for (uint32_t i = 0; i < 3; ++i) {
            a.update(i, first[i], first[i], um.next());
        }
        a.finalize();
        ReservoirBuilder b;
        for (uint32_t i = 0; i < 3; ++i) {
            b.update(i + 3, second[i], second[i], um.next());
        }
        b.finalize();

        // Merge b into a, evaluating b's survivor under a's target function —
        // which here is the same function, because both halves are the same
        // candidate domain.
        ReservoirBuilder merged;
        merged.merge(a.r, a.target_pdf, um.next());
        merged.merge(b.r, b.target_pdf, um.next());
        merged.finalize();
        ASSERT_NE(merged.r.light_index, RESERVOIR_NO_LIGHT);
        ++merged_hits[merged.r.light_index];
        // M must be the total candidate count, not the reservoir count.
        EXPECT_FLOAT_EQ(merged.r.sample_count, 6.0f);
    }

    for (std::size_t i = 0; i < 6; ++i) {
        const double expected = (i < 3 ? first[i] : second[i - 3]) / total;
        EXPECT_NEAR(static_cast<double>(batch_hits[i]) / TRIALS, expected, 0.01) << "batch " << i;
        EXPECT_NEAR(static_cast<double>(merged_hits[i]) / TRIALS, expected, 0.01) << "merged " << i;
    }
}

// Merging an empty reservoir is a no-op, including for M. A history that was
// dropped (vanished light, disocclusion) arrives here as exactly this.
TEST(Reservoir, MergingAnEmptyReservoirChangesNothing) {
    ReservoirBuilder r;
    r.update(3u, 2.0f, 2.0f, 0.0f);
    const float m_before = r.r.sample_count;
    r.merge(Reservoir{}, 5.0f, 0.0f);
    EXPECT_EQ(r.r.light_index, 3u);
    EXPECT_FLOAT_EQ(r.r.sample_count, m_before);
}

// A merged candidate the current surface cannot see contributes nothing but M.
// This is the structural stand-in for a normal-similarity test: the target
// function evaluated HERE is zero, so the history cannot win.
TEST(Reservoir, MergedCandidateWithZeroTargetHereCannotWin) {
    ReservoirBuilder r;
    r.update(1u, 1.0f, 1.0f, 0.0f);
    Reservoir hist{};
    hist.light_index = 99u;
    hist.weight = 1000.0f; // enormous history confidence...
    hist.sample_count = 120.0f;
    r.merge(hist, /*target_pdf_here=*/0.0f, /*rand01=*/0.0f); // ...but invisible here
    EXPECT_EQ(r.r.light_index, 1u);
    EXPECT_FLOAT_EQ(r.r.sample_count, 121.0f);
}

// ---------------------------------------------------------------------------
// The remap: history against a rebuilt light array.
// ---------------------------------------------------------------------------

TEST(ReservoirRemap, SurvivingLightIsRepointedAtItsNewIndex) {
    const std::vector<uint32_t> remap{2u, 0u, 1u};
    Reservoir prev{};
    prev.light_index = 0u;
    prev.weight = 1.5f;
    prev.sample_count = 30.0f;
    prev.depth = 42.0f;

    const Reservoir out = remap_reservoir(prev, remap);
    EXPECT_EQ(out.light_index, 2u);
    // Everything else rides through untouched: the light is the SAME light.
    EXPECT_FLOAT_EQ(out.weight, 1.5f);
    EXPECT_FLOAT_EQ(out.sample_count, 30.0f);
    EXPECT_FLOAT_EQ(out.depth, 42.0f);
}

// THE FAILURE THE REMAP EXISTS TO PREVENT. A vanished light's slot is very
// likely occupied by a different light this frame; carrying the reservoir would
// shadow against a light that is not there, using an index that now names
// someone else.
TEST(ReservoirRemap, VanishedLightDropsTheHistoryEntirely) {
    const std::vector<uint32_t> remap{RESERVOIR_NO_LIGHT, 0u};
    Reservoir prev{};
    prev.light_index = 0u;
    prev.weight = 9.0f;
    prev.sample_count = 120.0f;
    prev.depth = 7.0f;

    const Reservoir out = remap_reservoir(prev, remap);
    EXPECT_EQ(out.light_index, RESERVOIR_NO_LIGHT);
    EXPECT_FLOAT_EQ(out.weight, 0.0f);
    EXPECT_FLOAT_EQ(out.sample_count, 0.0f);
}

TEST(ReservoirRemap, EmptyAndOutOfRangeHistoriesAreDropped) {
    const std::vector<uint32_t> remap{0u, 1u};
    EXPECT_EQ(remap_reservoir(Reservoir{}, remap).light_index, RESERVOIR_NO_LIGHT);

    Reservoir past_end{};
    past_end.light_index = 5u; // the table shrank since the reservoir was written
    past_end.sample_count = 10.0f;
    const Reservoir out = remap_reservoir(past_end, remap);
    EXPECT_EQ(out.light_index, RESERVOIR_NO_LIGHT);
    EXPECT_FLOAT_EQ(out.sample_count, 0.0f);
}

// ---------------------------------------------------------------------------
// The M clamp.
// ---------------------------------------------------------------------------

TEST(ReservoirHistoryClamp, CapsMAtTheConfiguredMultipleAndLeavesWAlone) {
    Reservoir prev{};
    prev.light_index = 1u;
    prev.weight = 3.25f;
    prev.sample_count = 500.0f;

    const Reservoir out = clamp_reservoir_history(prev, /*current_m=*/6.0f, /*max_ratio=*/20.0f);
    EXPECT_FLOAT_EQ(out.sample_count, 120.0f);
    // W is normalized BY M already, so the estimate is unchanged — only its
    // inertia is. Clamping W too would be the bug this pins against.
    EXPECT_FLOAT_EQ(out.weight, 3.25f);
}

TEST(ReservoirHistoryClamp, UnderTheCapIsUntouched) {
    Reservoir prev{};
    prev.light_index = 1u;
    prev.sample_count = 42.0f;
    EXPECT_FLOAT_EQ(clamp_reservoir_history(prev, 6.0f, 20.0f).sample_count, 42.0f);
}

// The clamp makes a light that changes REACHABLE within a bounded number of
// frames: with M pinned at 20x the per-frame candidate count, a new candidate
// carrying all the weight wins with probability 1 immediately, and even a
// modest one is not drowned. Without it M grows without bound and this
// probability tends to zero.
TEST(ReservoirHistoryClamp, BoundedMKeepsANewCandidateAbleToWin) {
    Reservoir hist{};
    hist.light_index = 1u;
    hist.weight = 1.0f;
    hist.sample_count = 100000.0f; // an unclamped history after ~a minute

    const auto win_probability = [](const Reservoir& h, float new_weight) {
        ReservoirBuilder r;
        r.merge(h, 1.0f, 1.0f); // never accepted at rand == 1, but w_sum records it
        const float hist_w = 1.0f * h.weight * h.sample_count;
        return new_weight / (hist_w + new_weight);
    };

    const float unclamped = win_probability(hist, 10.0f);
    const float clamped = win_probability(clamp_reservoir_history(hist, 6.0f, 20.0f), 10.0f);
    EXPECT_LT(unclamped, 1e-3f);
    EXPECT_GT(clamped, 0.07f);
}
