#include "bracken/render/lighting/alias_table.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

using bracken::render::lighting::AliasEntry;
using bracken::render::lighting::build_alias_table;
using bracken::render::lighting::sample_alias;

namespace {

/// EXACT selection probabilities, by integrating the sampler over its own two
/// uniforms rather than by drawing randoms.
///
/// This is the whole reason these tests can assert equality instead of a
/// confidence band: `sample_alias` is a deterministic function of (u1, u2), so
/// summing the measure of the region that maps to each index gives the
/// distribution exactly. u1 picks a bucket (each of measure 1/n); inside it, u2
/// keeps the bucket with measure `prob` and takes the alias with `1 - prob`.
std::vector<double> exact_probabilities(const std::vector<AliasEntry>& table) {
    const std::size_t n = table.size();
    std::vector<double> p(n, 0.0);
    if (n == 0) {
        return p;
    }
    const double bucket = 1.0 / static_cast<double>(n);
    for (std::size_t i = 0; i < n; ++i) {
        p[i] += bucket * static_cast<double>(table[i].prob);
        p[table[i].alias] += bucket * (1.0 - static_cast<double>(table[i].prob));
    }
    return p;
}

} // namespace

TEST(AliasTable, EmptyInputProducesEmptyTable) {
    std::vector<AliasEntry> table;
    EXPECT_EQ(build_alias_table({}, table), 0.0f);
    EXPECT_TRUE(table.empty());
    // A sample from an empty table must not read out of bounds. Zero is the
    // only defensible answer and the shader never asks (emissiveCount is 0).
    EXPECT_EQ(sample_alias(table, 0.5f, 0.5f), 0u);
}

TEST(AliasTable, SingleEntryAlwaysSelectsItself) {
    const std::vector<float> w{7.5f};
    std::vector<AliasEntry> table;
    EXPECT_FLOAT_EQ(build_alias_table(w, table), 7.5f);
    ASSERT_EQ(table.size(), 1u);
    EXPECT_FLOAT_EQ(table[0].prob, 1.0f);
    EXPECT_EQ(table[0].alias, 0u);
    for (int i = 0; i < 16; ++i) {
        const float u = static_cast<float>(i) / 16.0f;
        EXPECT_EQ(sample_alias(table, u, u), 0u);
    }
}

TEST(AliasTable, AllZeroWeightsDegradeToUniformRatherThanUnsampleable) {
    // A scene whose emitters all collapsed (degenerate instance transforms) is
    // a real state. The table must stay INDEXABLE — an out-of-range alias is a
    // GPU read past the buffer — even though the total says it carries no power
    // and the caller will therefore not draw from it.
    const std::vector<float> w{0.0f, 0.0f, 0.0f};
    std::vector<AliasEntry> table;
    EXPECT_EQ(build_alias_table(w, table), 0.0f);
    ASSERT_EQ(table.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(table[i].prob, 1.0f);
        EXPECT_EQ(table[i].alias, i);
    }
}

TEST(AliasTable, NonFiniteAndNegativeWeightsAreTreatedAsZero) {
    const std::vector<float> w{std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity(), -3.0f, 4.0f};
    std::vector<AliasEntry> table;
    // Only the 4.0 survives, so it is the whole distribution.
    EXPECT_FLOAT_EQ(build_alias_table(w, table), 4.0f);
    const auto p = exact_probabilities(table);
    EXPECT_NEAR(p[3], 1.0, 1e-9);
    EXPECT_NEAR(p[0] + p[1] + p[2], 0.0, 1e-9);
}

TEST(AliasTable, SelectionProbabilityIsProportionalToWeight) {
    // The claim the whole file exists to make. Weights deliberately span two
    // orders of magnitude and are neither sorted nor a nice ratio.
    const std::vector<float> w{3.0f, 1.0f, 12.0f, 0.25f, 7.0f, 2.5f, 0.5f};
    std::vector<AliasEntry> table;
    const float total = build_alias_table(w, table);
    ASSERT_EQ(table.size(), w.size());
    EXPECT_FLOAT_EQ(total, std::accumulate(w.begin(), w.end(), 0.0f));

    const auto p = exact_probabilities(table);
    for (std::size_t i = 0; i < w.size(); ++i) {
        EXPECT_NEAR(p[i], static_cast<double>(w[i] / total), 1e-6)
            << "entry " << i << " is not drawn in proportion to its power";
    }
    // And nothing leaks: the probabilities are a distribution.
    EXPECT_NEAR(std::accumulate(p.begin(), p.end(), 0.0), 1.0, 1e-9);
}

TEST(AliasTable, EveryAliasIndexIsInRangeAndEveryProbIsAProbability) {
    // A hostile shape: one dominant entry and a long tail, which is what a
    // village of dim windows plus a handful of bright orbs actually looks like
    // and is the case that stresses Vose's small/large pairing.
    std::vector<float> w(64, 0.01f);
    w[7] = 500.0f;
    std::vector<AliasEntry> table;
    ASSERT_GT(build_alias_table(w, table), 0.0f);
    for (const AliasEntry& e : table) {
        EXPECT_GE(e.prob, 0.0f);
        EXPECT_LE(e.prob, 1.0f);
        EXPECT_LT(e.alias, table.size());
    }
    const auto p = exact_probabilities(table);
    EXPECT_NEAR(p[7], 500.0 / (500.0 + 63.0 * 0.01), 1e-6);
}

TEST(AliasTable, ConstructionIsDeterministic) {
    // Bit-identical run to run for the same weight sequence — the property the
    // smoke gate depends on, and the reason the two worklists are index-ordered
    // stacks rather than any container whose order is an implementation detail.
    const std::vector<float> w{5.0f, 5.0f, 5.0f, 1.0f, 9.0f, 5.0f};
    std::vector<AliasEntry> a;
    std::vector<AliasEntry> b;
    build_alias_table(w, a);
    build_alias_table(w, b);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].prob, b[i].prob);
        EXPECT_EQ(a[i].alias, b[i].alias);
    }
}

TEST(AliasTable, SampleIsInRangeForBoundaryUniforms) {
    const std::vector<float> w{1.0f, 2.0f, 3.0f};
    std::vector<AliasEntry> table;
    build_alias_table(w, table);
    // u1 == 1.0 would index one past the end without the clamp; the shader's
    // rtNextFloat is [0,1) but a float rounding up at the top of the range is
    // exactly the kind of thing that reads one record past the buffer.
    EXPECT_LT(sample_alias(table, 1.0f, 0.0f), 3u);
    EXPECT_LT(sample_alias(table, 0.999999f, 0.999999f), 3u);
    EXPECT_LT(sample_alias(table, 0.0f, 0.0f), 3u);
}
