#include <gtest/gtest.h>

#include "bracken/core/random.hpp"

#include <cmath>
#include <set>
#include <vector>

using bracken::core::Rng;

TEST(Rng, SameSeedGivesTheSameStream) {
    Rng a(12345);
    Rng b(12345);
    for (int i = 0; i < 256; ++i) {
        EXPECT_EQ(a.next_u64(), b.next_u64()) << "at draw " << i;
    }
}

TEST(Rng, DifferentSeedsDiverge) {
    Rng a(1);
    Rng b(2);
    int same = 0;
    for (int i = 0; i < 64; ++i) {
        same += (a.next_u64() == b.next_u64()) ? 1 : 0;
    }
    EXPECT_EQ(same, 0);
}

TEST(Rng, UnitIsInRange) {
    Rng r(99);
    for (int i = 0; i < 10000; ++i) {
        const float v = r.next_unit();
        ASSERT_GE(v, 0.0f);
        ASSERT_LT(v, 1.0f);
    }
}

TEST(Rng, RangeRespectsBounds) {
    Rng r(7);
    for (int i = 0; i < 10000; ++i) {
        const float v = r.next_range(-3.0f, 5.0f);
        ASSERT_GE(v, -3.0f);
        ASSERT_LT(v, 5.0f);
    }
}

TEST(Rng, BelowZeroDoesNotDivideByZero) {
    Rng r(1);
    EXPECT_EQ(r.next_below(0), 0u);
}

TEST(Rng, BelowStaysUnderTheBound) {
    Rng r(4);
    for (int i = 0; i < 1000; ++i) {
        ASSERT_LT(r.next_below(10), 10u);
    }
}

// Consecutive draws must actually differ. This is the property the playground's
// footstep scatter needed and the old position-derived version failed to give.
TEST(Rng, ConsecutiveDrawsVary) {
    Rng r(0xABCDEF);
    std::set<float> distinct;
    for (int i = 0; i < 64; ++i) {
        distinct.insert(r.next_range(0.92f, 1.08f));
    }
    EXPECT_GT(distinct.size(), 55u) << "a scatter source must actually scatter";
}

// The bug this replaced, stated as a test so it cannot come back: deriving a
// "random" value from one coordinate is constant whenever that coordinate is.
// A player running along Z holds x fixed, so every footstep got an identical
// pitch — reproducible, and audibly wrong.
TEST(Rng, BeatsDerivingScatterFromASingleCoordinate) {
    auto position_derived = [](float x) {
        return 0.92f + 0.16f * std::fabs(std::fmod(x * 0.013f, 1.0f));
    };

    // Walk along Z: x never changes.
    std::set<float> derived;
    for (int step = 0; step < 32; ++step) {
        derived.insert(position_derived(100.0f)); // x constant
    }
    EXPECT_EQ(derived.size(), 1u) << "this is the defect: no variation at all";

    Rng r(0x5EED);
    std::set<float> seeded;
    for (int step = 0; step < 32; ++step) {
        seeded.insert(r.next_range(0.92f, 1.08f));
    }
    EXPECT_GT(seeded.size(), 28u) << "a seeded stream varies regardless of position";
}

TEST(Rng, ReseedRestartsTheStream) {
    Rng r(42);
    const uint64_t first = r.next_u64();
    r.next_u64();
    r.reseed(42);
    EXPECT_EQ(r.next_u64(), first);
}

// Roughly uniform: a scatter that clumps would still sound repetitive.
TEST(Rng, IsRoughlyUniform) {
    Rng r(2024);
    int buckets[10] = {};
    constexpr int k_draws = 100000;
    for (int i = 0; i < k_draws; ++i) {
        buckets[static_cast<int>(r.next_unit() * 10.0f)]++;
    }
    for (int b : buckets) {
        // Each decile should hold ~10%; allow a generous ±2%.
        EXPECT_NEAR(static_cast<double>(b) / k_draws, 0.1, 0.02);
    }
}

TEST(Rng, IsUsableInAConstantExpression) {
    // constexpr so a host can build lookup tables at compile time.
    constexpr uint64_t v = [] {
        Rng r(1);
        return r.next_u64();
    }();
    static_assert(v != 0);
    SUCCEED();
}
