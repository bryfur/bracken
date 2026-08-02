// The probe budget and the cascade rota — the two tables that decide what GI
// costs per frame.
//
// These are settings-driven numbers with no GPU anywhere near them, and getting
// one wrong is invisible: the field still fills, just faster or slower than the
// tier it was sold as. So the arithmetic is pinned here, including the peak —
// which is the number the feature's cost should ever be quoted against, and
// which is NOT the per-cascade budget.

#include "bracken/render/gi/probe_schedule.hpp"

#include <gtest/gtest.h>

#include <array>
#include <set>

using namespace bracken::render::gi;
using bracken::GiQuality;

TEST(ProbeSchedule, BudgetsFollowTheTierTable) {
    // Rays/probe {32, 64, 96} and update fraction {1/8, 1/4, 1/4},
    // docs/rt_lighting_plan.md §5.2.
    EXPECT_EQ(gi_budget(GiQuality::Low).rays_per_probe, 32u);
    EXPECT_EQ(gi_budget(GiQuality::Medium).rays_per_probe, 64u);
    EXPECT_EQ(gi_budget(GiQuality::High).rays_per_probe, 96u);

    EXPECT_EQ(gi_budget(GiQuality::Low).probes_per_turn, PROBES_PER_CASCADE / 8);
    EXPECT_EQ(gi_budget(GiQuality::Medium).probes_per_turn, PROBES_PER_CASCADE / 4);
    EXPECT_EQ(gi_budget(GiQuality::High).probes_per_turn, PROBES_PER_CASCADE / 4);

    EXPECT_EQ(gi_budget(GiQuality::Low).probes_per_turn, 864u);
    EXPECT_EQ(gi_budget(GiQuality::Medium).probes_per_turn, 1728u);
}

TEST(ProbeSchedule, BudgetsAreMonotonicInQuality) {
    // Nobody should ever be able to make GI cheaper by turning quality UP.
    const GiBudget low = gi_budget(GiQuality::Low);
    const GiBudget medium = gi_budget(GiQuality::Medium);
    const GiBudget high = gi_budget(GiQuality::High);
    EXPECT_LE(low.rays_per_probe, medium.rays_per_probe);
    EXPECT_LE(medium.rays_per_probe, high.rays_per_probe);
    EXPECT_LE(low.probes_per_turn, medium.probes_per_turn);
    EXPECT_LE(medium.probes_per_turn, high.probes_per_turn);
    EXPECT_LT(gi_peak_rays_per_frame(GiQuality::Low), gi_peak_rays_per_frame(GiQuality::Medium));
    EXPECT_LT(gi_peak_rays_per_frame(GiQuality::Medium), gi_peak_rays_per_frame(GiQuality::High));
}

TEST(ProbeSchedule, PeakRayCountsMatchThePlansTierBudgets) {
    // 2 cascades x budget, because the rota never puts more than two in one
    // frame — pinned separately below.
    EXPECT_EQ(gi_peak_rays_per_frame(GiQuality::Low), 2u * 864u * 32u);     // 55,296
    EXPECT_EQ(gi_peak_rays_per_frame(GiQuality::Medium), 2u * 1728u * 64u); // 221,184
    EXPECT_EQ(gi_peak_rays_per_frame(GiQuality::High), 2u * 1728u * 96u);   // 331,776

    // §5.2's Low tier is "iGPU-viable at ~85 k rays" and High is "~500 k".
    // Ours come in under both, which is the direction that is allowed to drift.
    EXPECT_LT(gi_peak_rays_per_frame(GiQuality::Low), 85000u);
    EXPECT_LT(gi_peak_rays_per_frame(GiQuality::High), 500000u);
}

TEST(ProbeSchedule, BufferSizingCoversEveryQuality) {
    // The trace buffers are allocated once at the worst case so a quality
    // change never reallocates while the previous frame is still reading.
    for (const GiQuality q : {GiQuality::Low, GiQuality::Medium, GiQuality::High}) {
        const GiBudget b = gi_budget(q);
        EXPECT_LE(b.probes_per_turn, gi_max_probes_per_dispatch());
        EXPECT_LE(b.probes_per_turn * b.rays_per_probe, gi_max_rays_per_dispatch());
    }
}

TEST(ProbeSchedule, RotaGivesTheFineCascadeEveryFrame) {
    for (std::uint64_t frame = 0; frame < 64; ++frame) {
        EXPECT_TRUE(gi_cascade_updates_on_frame(0, frame)) << "frame " << frame;
    }
}

TEST(ProbeSchedule, RotaHalvesTheRateWithEachCascade) {
    // The exact schedule, spelled out, because "every 2nd / every 4th" has two
    // readings and only one of them keeps the peak at two cascades:
    //   c1 on odd frames, c2 on frames == 2 (mod 4).
    const std::array<bool, 8> expected_c1{false, true, false, true, false, true, false, true};
    const std::array<bool, 8> expected_c2{false, false, true, false, false, false, true, false};
    for (std::uint64_t frame = 0; frame < 8; ++frame) {
        EXPECT_EQ(gi_cascade_updates_on_frame(1, frame), expected_c1[frame]) << "frame " << frame;
        EXPECT_EQ(gi_cascade_updates_on_frame(2, frame), expected_c2[frame]) << "frame " << frame;
    }

    // Over a long run the rates are exactly 1, 1/2 and 1/4.
    std::array<int, CASCADE_COUNT> counts{};
    constexpr int FRAMES = 4096;
    for (std::uint64_t frame = 0; frame < FRAMES; ++frame) {
        for (int c = 0; c < CASCADE_COUNT; ++c) {
            counts[static_cast<std::size_t>(c)] += gi_cascade_updates_on_frame(c, frame) ? 1 : 0;
        }
    }
    EXPECT_EQ(counts[0], FRAMES);
    EXPECT_EQ(counts[1], FRAMES / 2);
    EXPECT_EQ(counts[2], FRAMES / 4);
}

TEST(ProbeSchedule, AtMostTwoCascadesTraceInOneFrame) {
    // This is what GI_MAX_CASCADES_PER_FRAME asserts, and the whole reason the
    // coarse cascades are PHASED rather than all landing on frame 0. Without
    // it the peak frame would be 50% more expensive than the table says.
    for (std::uint64_t frame = 0; frame < 4096; ++frame) {
        int active = 0;
        for (int c = 0; c < CASCADE_COUNT; ++c) {
            active += gi_cascade_updates_on_frame(c, frame) ? 1 : 0;
        }
        EXPECT_LE(active, GI_MAX_CASCADES_PER_FRAME) << "frame " << frame;
    }
}

TEST(ProbeSchedule, RotaIgnoresCascadesOutsideTheField) {
    EXPECT_FALSE(gi_cascade_updates_on_frame(CASCADE_COUNT, 0));
    EXPECT_FALSE(gi_cascade_updates_on_frame(CASCADE_COUNT + 5, 12));
    // Negative is "cascade 0 or finer" — there is nothing finer, so it is
    // treated as cascade 0 rather than silently skipped.
    EXPECT_TRUE(gi_cascade_updates_on_frame(-1, 3));
}

TEST(ProbeSchedule, AFullRefreshTakesTheDocumentedNumberOfFrames) {
    // Cascade 0 at Medium refreshes fully every 4 frames; cascade 2 every 16.
    // Those two numbers are what "the field lags" actually means, so they are
    // stated as a test rather than as a comment.
    const GiBudget b = gi_budget(GiQuality::Medium);
    const std::uint32_t turns_for_full = PROBES_PER_CASCADE / b.probes_per_turn;
    EXPECT_EQ(turns_for_full, 4u);

    for (int c = 0; c < CASCADE_COUNT; ++c) {
        std::uint32_t turns = 0;
        std::uint64_t frame = 0;
        while (turns < turns_for_full) {
            if (gi_cascade_updates_on_frame(c, frame)) {
                ++turns;
            }
            ++frame;
        }
        const std::uint64_t expected_period = 1ull << static_cast<unsigned>(c);
        EXPECT_LE(frame, turns_for_full * expected_period + expected_period)
            << "cascade " << c << " full refresh took " << frame << " frames";
    }
}
