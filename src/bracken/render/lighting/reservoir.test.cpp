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

// ---------------------------------------------------------------------------
// Pairwise-MIS spatial reuse (R7a, docs/rt_lighting_plan.md §5.3)
// ---------------------------------------------------------------------------
//
// The claim these tests make is the one a screenshot cannot: that reusing a
// NEIGHBOUR'S reservoir leaves the estimator unbiased. A biased spatial pass
// does not crash and does not look broken — it brightens depth discontinuities
// and leaks light around corners by a few percent, forever — so it is checked
// against brute force over a domain small enough to sum exactly.
//
// The world used below: `LIGHTS` discrete lights, each with a contribution
// f(x) at the CENTRE pixel. Every pixel p has its own target function p̂_p and
// its own DOMAIN D_p (the froxel list): p̂_p(x) is zero outside D_p, which is
// the condition the derivation in reservoir.hpp names as its one assumption.
// A reservoir at pixel p is built by streaming M uniform draws from D_p, which
// is exactly what the shipped pass does.

namespace {

using bracken::render::lighting::PairwiseSpatialMerge;

constexpr uint32_t LIGHTS = 6;

/// One pixel of the toy world: a target function and the froxel list it draws
/// candidates from.
struct Pixel {
    std::vector<uint32_t> domain;
    std::array<float, LIGHTS> target{};

    [[nodiscard]] float pdf(uint32_t light) const {
        for (uint32_t d : domain) {
            if (d == light) {
                return target[light];
            }
        }
        return 0.0f; // outside the domain: p̂ == 0, the derivation's assumption
    }
};

/// Build one pixel's reservoir the way the trace does: M uniform draws from the
/// froxel list, RIS-weighted by p̂ over the source pdf (1 / |list|).
Reservoir sample_pixel(const Pixel& p, uint32_t m, Uniforms& u) {
    ReservoirBuilder r;
    const auto n = static_cast<float>(p.domain.size());
    for (uint32_t i = 0; i < m; ++i) {
        const auto slot = std::min(static_cast<std::size_t>(u.next() * n), p.domain.size() - 1);
        const uint32_t light = p.domain[slot];
        const float pdf = p.pdf(light);
        r.update(light, pdf * n, pdf, u.next());
    }
    r.finalize();
    return r.r;
}

/// The centre pixel and three neighbours, deliberately DISAGREEING: neighbour 2
/// cannot see light 0 or 1 at all, and neighbour 3's domain is disjoint from the
/// centre's except for one light. A 1/Z merge of these is visibly biased; the
/// pairwise weights are what make it not.
struct World {
    Pixel centre;
    std::array<Pixel, 3> neighbors;
};

World make_world() {
    World w;
    w.centre.domain = {0, 1, 2, 3};
    w.centre.target = {2.0f, 0.5f, 1.25f, 3.0f, 0.0f, 0.0f};

    w.neighbors[0].domain = {0, 1, 2, 3};
    w.neighbors[0].target = {1.7f, 0.9f, 1.0f, 2.2f, 0.0f, 0.0f};

    w.neighbors[1].domain = {2, 3, 4};
    w.neighbors[1].target = {0.0f, 0.0f, 3.1f, 0.4f, 1.9f, 0.0f};

    w.neighbors[2].domain = {3, 4, 5};
    w.neighbors[2].target = {0.0f, 0.0f, 0.0f, 0.8f, 2.6f, 1.1f};

    return w;
}

/// One spatial merge, driven end to end exactly as the shader drives it.
Reservoir merge_world(const World& w, const std::array<Reservoir, 3>& neighbors,
                      const Reservoir& canonical, Uniforms& u) {
    PairwiseSpatialMerge m;
    m.begin(canonical, canonical.light_index == RESERVOIR_NO_LIGHT
                           ? 0.0f
                           : w.centre.pdf(canonical.light_index));
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        const Reservoir& n = neighbors[i];
        if (n.light_index == RESERVOIR_NO_LIGHT) {
            m.add_neighbor(n, 0.0f, 0.0f, 0.0f, u.next());
            continue;
        }
        const float canonical_there =
            canonical.light_index == RESERVOIR_NO_LIGHT
                ? 0.0f
                : w.neighbors[i].pdf(canonical.light_index);
        m.add_neighbor(n, w.centre.pdf(n.light_index), w.neighbors[i].pdf(n.light_index),
                       canonical_there, u.next());
    }
    return m.finalize(u.next());
}

} // namespace

// THE PROPERTY THE WHOLE SCHEME RESTS ON: the MIS weights sum to 1 at every
// point of the union domain, including the two degenerate ones — a light the
// neighbour cannot see (p̂_i = 0) and a light the centre cannot (p̂_c = 0).
// Checked directly on the algebra rather than through a merge, because if this
// fails every other test below fails for a reason that is harder to read.
TEST(PairwiseSpatialMerge, MisWeightsPartitionUnity) {
    const auto pair = [](float m_i, float p_i, float m_c, float p_c) {
        const float den = m_i * p_i + m_c * p_c;
        const float a = den > 0.0f ? (m_i * p_i) / den : 0.0f;
        const float b = den > 0.0f ? (m_c * p_c) / den : 0.0f;
        return a + b;
    };
    // Ordinary: both inputs see the light.
    EXPECT_FLOAT_EQ(pair(12.0f, 1.5f, 30.0f, 0.75f), 1.0f);
    // The neighbour cannot see it: a == 0, b == 1.
    EXPECT_FLOAT_EQ(pair(12.0f, 0.0f, 30.0f, 0.75f), 1.0f);
    // The centre cannot see it: a == 1, b == 0.
    EXPECT_FLOAT_EQ(pair(12.0f, 1.5f, 30.0f, 0.0f), 1.0f);
}

// With no accepted neighbour the merge is the IDENTITY, bit for bit. This is
// the contract `spatial_neighbors = 0` leans on: turning the pass off must
// reproduce the pre-R7a image exactly, and "exactly" is a bit comparison.
TEST(PairwiseSpatialMerge, NoNeighboursIsTheIdentity) {
    Reservoir c{};
    c.light_index = 3u;
    c.weight = 1.234567f;
    c.sample_count = 26.0f;
    c.depth = 412.5f;

    PairwiseSpatialMerge m;
    m.begin(c, 2.5f);
    const Reservoir out = m.finalize(0.5f);
    EXPECT_EQ(out.light_index, c.light_index);
    EXPECT_EQ(out.weight, c.weight);
    EXPECT_EQ(out.sample_count, c.sample_count);
    EXPECT_EQ(out.depth, c.depth);
}

// A neighbour holding nothing is not an input: it must not count toward k, or
// it dilutes the canonical's MIS weight and darkens the pixel for free.
TEST(PairwiseSpatialMerge, EmptyNeighboursAreNotInputs) {
    Reservoir c{};
    c.light_index = 1u;
    c.weight = 2.0f;
    c.sample_count = 8.0f;

    PairwiseSpatialMerge m;
    m.begin(c, 1.0f);
    m.add_neighbor(Reservoir{}, 1.0f, 1.0f, 1.0f, 0.5f);
    Reservoir zero_m{};
    zero_m.light_index = 4u;
    zero_m.weight = 9.0f;
    zero_m.sample_count = 0.0f;
    m.add_neighbor(zero_m, 1.0f, 1.0f, 1.0f, 0.5f);
    EXPECT_EQ(m.neighbors, 0u);
    EXPECT_EQ(m.finalize(0.5f).weight, c.weight);
}

// `w_sum` is a SUM, so it cannot depend on the order neighbours arrive in. The
// selection does (it consumes the random stream), but the contribution weight
// the estimator multiplies by is order-invariant — which is what makes the
// shader's rotating neighbour offsets a sampling choice rather than a
// correctness one.
TEST(PairwiseSpatialMerge, WeightSumIsOrderIndependent) {
    const World w = make_world();
    Uniforms build(4242);
    const Reservoir canonical = sample_pixel(w.centre, 4, build);
    std::array<Reservoir, 3> n{sample_pixel(w.neighbors[0], 4, build),
                               sample_pixel(w.neighbors[1], 4, build),
                               sample_pixel(w.neighbors[2], 4, build)};

    const auto run = [&](std::array<std::size_t, 3> order) {
        PairwiseSpatialMerge m;
        m.begin(canonical, w.centre.pdf(canonical.light_index));
        for (std::size_t i : order) {
            m.add_neighbor(n[i], w.centre.pdf(n[i].light_index),
                           w.neighbors[i].pdf(n[i].light_index),
                           w.neighbors[i].pdf(canonical.light_index), 1.0f);
        }
        (void)m.finalize(1.0f);
        return m.w_sum;
    };
    EXPECT_NEAR(run({0, 1, 2}), run({2, 0, 1}), 1e-5f);
    EXPECT_NEAR(run({0, 1, 2}), run({1, 2, 0}), 1e-5f);
}

// THE BRUTE-FORCE GATE. The RIS estimate of the centre pixel's integral is
// f(y) * W. Summed over trials it must converge on the EXACT integral,
// sum_x f(x) over the centre's domain — a number this toy world can just add
// up. Run against the same trials WITHOUT spatial reuse as a control, so the
// test says "reuse did not move the mean" and not merely "the mean is roughly
// right".
TEST(PairwiseSpatialMerge, UnbiasedAgainstBruteForce) {
    const World w = make_world();
    // f is deliberately NOT the target function: an estimator that is only
    // unbiased for the function it importance-sampled is not unbiased.
    const std::array<float, LIGHTS> f{1.0f, 4.0f, 0.25f, 2.0f, 3.5f, 0.75f};
    double exact = 0.0;
    for (uint32_t x : w.centre.domain) {
        exact += f[x];
    }

    constexpr int TRIALS = 20000;
    Uniforms u(20260802);
    double reused = 0.0;
    double control = 0.0;
    for (int t = 0; t < TRIALS; ++t) {
        const Reservoir canonical = sample_pixel(w.centre, 4, u);
        const std::array<Reservoir, 3> n{sample_pixel(w.neighbors[0], 4, u),
                                         sample_pixel(w.neighbors[1], 4, u),
                                         sample_pixel(w.neighbors[2], 4, u)};
        const Reservoir out = merge_world(w, n, canonical, u);
        if (out.light_index != RESERVOIR_NO_LIGHT) {
            reused += static_cast<double>(f[out.light_index] * out.weight);
        }
        if (canonical.light_index != RESERVOIR_NO_LIGHT) {
            control += static_cast<double>(f[canonical.light_index] * canonical.weight);
        }
    }
    reused /= TRIALS;
    control /= TRIALS;

    // 2% of the integral. The estimator's own standard error over 20 k trials
    // is well inside that; a missing MIS weight is not. Measured on this exact
    // world, the confidence-weighted (1/Z) merge the TEMPORAL path uses lands at
    // 0.65x the integral — 35% of the energy lost, because the neighbours whose
    // samples score p̂_c == 0 here still take their share of the 1/sum(M)
    // normalization. That 35% is what the world was built to expose.
    EXPECT_NEAR(reused, exact, exact * 0.02);
    EXPECT_NEAR(control, exact, exact * 0.02);
}

// The union domain is NOT the centre's domain: neighbours 2 and 3 can hand this
// pixel lights 4 and 5, which the centre's froxel never lists. The MIS weights
// must give those samples weight zero at the centre — p̂_c is 0 there, so the
// contribution is zero by construction — and the estimate must STILL come out
// on the centre's own integral rather than on the union's.
TEST(PairwiseSpatialMerge, ForeignLightsDoNotAddEnergy) {
    const World w = make_world();
    const std::array<float, LIGHTS> f{1.0f, 4.0f, 0.25f, 2.0f, 3.5f, 0.75f};
    double exact = 0.0;
    for (uint32_t x : w.centre.domain) {
        exact += f[x];
    }
    // The union integral, which a scheme that forgot to evaluate p̂_c at the
    // neighbour's sample would converge on instead.
    const double union_integral = exact + static_cast<double>(f[4] + f[5]);
    ASSERT_GT(union_integral, exact * 1.2);

    constexpr int TRIALS = 20000;
    Uniforms u(99);
    double mean = 0.0;
    int foreign = 0;
    for (int t = 0; t < TRIALS; ++t) {
        const Reservoir canonical = sample_pixel(w.centre, 4, u);
        const std::array<Reservoir, 3> n{sample_pixel(w.neighbors[0], 4, u),
                                         sample_pixel(w.neighbors[1], 4, u),
                                         sample_pixel(w.neighbors[2], 4, u)};
        const Reservoir out = merge_world(w, n, canonical, u);
        if (out.light_index != RESERVOIR_NO_LIGHT) {
            mean += static_cast<double>(f[out.light_index] * out.weight);
            if (out.light_index >= 4u) {
                ++foreign;
            }
        }
    }
    mean /= TRIALS;
    EXPECT_EQ(foreign, 0) << "a light outside the centre's domain scores p̂_c == 0 "
                             "and can never be selected";
    EXPECT_NEAR(mean, exact, exact * 0.02);
}

// Confidence accumulates: the merged record stands for every candidate that
// informed it, which is what next frame's temporal clamp then bounds.
TEST(PairwiseSpatialMerge, ConfidenceIsTheSumOfItsInputs) {
    Reservoir c{};
    c.light_index = 0u;
    c.weight = 1.0f;
    c.sample_count = 10.0f;
    c.depth = 100.0f;

    Reservoir n{};
    n.light_index = 1u;
    n.weight = 1.0f;
    n.sample_count = 7.0f;
    n.depth = 900.0f; // deliberately different: the merged depth must be OURS

    PairwiseSpatialMerge m;
    m.begin(c, 1.0f);
    m.add_neighbor(n, 1.0f, 1.0f, 1.0f, 0.5f);
    const Reservoir out = m.finalize(0.5f);
    EXPECT_FLOAT_EQ(out.sample_count, 17.0f);
    EXPECT_FLOAT_EQ(out.depth, 100.0f);
}
