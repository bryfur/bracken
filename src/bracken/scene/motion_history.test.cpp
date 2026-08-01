#include "bracken/scene/motion_history.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

using namespace bracken::scene;

namespace {
glm::mat4 translation(float x) {
    return glm::translate(glm::mat4(1.0f), glm::vec3(x, 0.0f, 0.0f));
}
} // namespace

TEST(MotionHistory, EmptyHistoryFindsNothing) {
    MotionHistory h;
    EXPECT_EQ(h.find_prev(1), nullptr);
    EXPECT_EQ(h.size(), 0u);
}

TEST(MotionHistory, MotionIdZeroIsNeverRecordedOrFound) {
    MotionHistory h;
    h.record(0, translation(5.0f));
    h.end_frame();
    EXPECT_EQ(h.find_prev(0), nullptr);
    EXPECT_EQ(h.size(), 0u);
}

TEST(MotionHistory, RecordedStateBecomesVisibleAfterEndFrame) {
    MotionHistory h;
    h.record(7, translation(1.0f), 12);

    // Not visible within the same frame — find_prev returns frame N-1 state.
    EXPECT_EQ(h.find_prev(7), nullptr);

    h.end_frame();
    const MotionHistory::Entry* e = h.find_prev(7);
    ASSERT_NE(e, nullptr);
    EXPECT_EQ(e->model, translation(1.0f));
    EXPECT_EQ(e->palette_offset, 12u);
}

TEST(MotionHistory, PreviousStateLagsByExactlyOneFrame) {
    MotionHistory h;
    // Frame 1: object at x=1
    h.record(3, translation(1.0f));
    h.end_frame();
    // Frame 2: object at x=2; prev must be x=1
    ASSERT_NE(h.find_prev(3), nullptr);
    EXPECT_EQ(h.find_prev(3)->model, translation(1.0f));
    h.record(3, translation(2.0f));
    h.end_frame();
    // Frame 3: prev must be x=2
    ASSERT_NE(h.find_prev(3), nullptr);
    EXPECT_EQ(h.find_prev(3)->model, translation(2.0f));
}

TEST(MotionHistory, EntriesNotResubmittedArePruned) {
    MotionHistory h;
    h.record(1, translation(1.0f));
    h.record(2, translation(2.0f));
    h.end_frame();
    EXPECT_EQ(h.size(), 2u);

    // Next frame only id 1 is submitted.
    h.record(1, translation(1.5f));
    h.end_frame();
    EXPECT_EQ(h.size(), 1u);
    EXPECT_NE(h.find_prev(1), nullptr);
    EXPECT_EQ(h.find_prev(2), nullptr) << "id 2 was not re-submitted and must be pruned";
}

TEST(MotionHistory, DefaultPaletteOffsetIsNoPalette) {
    MotionHistory h;
    h.record(9, translation(4.0f)); // static model: no palette
    h.end_frame();
    ASSERT_NE(h.find_prev(9), nullptr);
    EXPECT_EQ(h.find_prev(9)->palette_offset, MotionHistory::NO_PALETTE);
}

TEST(MotionHistory, DuplicateRecordInFrameKeepsLast) {
    MotionHistory h;
    h.record(5, translation(1.0f), 3);
    h.record(5, translation(2.0f), 8);
    h.end_frame();
    ASSERT_NE(h.find_prev(5), nullptr);
    EXPECT_EQ(h.find_prev(5)->model, translation(2.0f));
    EXPECT_EQ(h.find_prev(5)->palette_offset, 8u);
}

TEST(MotionHistory, ClearDropsEverything) {
    MotionHistory h;
    h.record(1, translation(1.0f));
    h.end_frame();
    h.record(1, translation(2.0f));
    h.clear();
    EXPECT_EQ(h.size(), 0u);
    h.end_frame();
    EXPECT_EQ(h.find_prev(1), nullptr);
}

// ============================================================================
// Open-addressed storage (phase 10d). The public contract above is the real
// specification; these pin the behaviours the table implementation could get
// wrong and the maps could not: growth, probe-run collisions, and the stamp
// bump that replaced the per-frame clear.
// ============================================================================

TEST(MotionHistory, GrowsPastInitialCapacityAndKeepsEveryEntry) {
    MotionHistory h;
    // Well past the initial 64 slots, so the table rehashes several times.
    constexpr uint32_t COUNT = 5000;
    for (uint32_t i = 1; i <= COUNT; ++i) {
        h.record(i, translation(static_cast<float>(i)), i);
    }
    h.end_frame();

    EXPECT_EQ(h.size(), COUNT);
    for (uint32_t i = 1; i <= COUNT; ++i) {
        const MotionHistory::Entry* e = h.find_prev(i);
        ASSERT_NE(e, nullptr) << "id " << i;
        EXPECT_EQ(e->model, translation(static_cast<float>(i))) << "id " << i;
        EXPECT_EQ(e->palette_offset, i) << "id " << i;
    }
    EXPECT_EQ(h.find_prev(COUNT + 1), nullptr); // absent id, after a full table
}

TEST(MotionHistory, IdsCollidingInTheSameProbeRunStayDistinct) {
    MotionHistory h;
    // Fibonacci hashing takes the high bits of key * 2654435769, so ids one
    // apart land in adjacent slots — consecutive ids ARE the collision case.
    for (uint32_t i = 100; i < 140; ++i) {
        h.record(i, translation(static_cast<float>(i)));
    }
    h.end_frame();
    for (uint32_t i = 100; i < 140; ++i) {
        const MotionHistory::Entry* e = h.find_prev(i);
        ASSERT_NE(e, nullptr) << "id " << i;
        EXPECT_EQ(e->model, translation(static_cast<float>(i))) << "id " << i;
    }
    // A miss whose probe run starts inside that cluster must still terminate.
    EXPECT_EQ(h.find_prev(139u + 4096u), nullptr);
}

TEST(MotionHistory, SurvivesManyFramesOfChurnWithoutLeakingStaleEntries) {
    MotionHistory h;
    // Each frame records a disjoint id window; the previous window must be
    // pruned by end_frame() alone (the stamp bump), never resurface, and the
    // table must not grow without bound.
    for (uint32_t frame = 0; frame < 200; ++frame) {
        for (uint32_t k = 0; k < 32; ++k) {
            h.record(frame * 1000 + k + 1, translation(static_cast<float>(k)));
        }
        h.end_frame();
        EXPECT_EQ(h.size(), 32u) << "frame " << frame;
        if (frame > 0) {
            EXPECT_EQ(h.find_prev((frame - 1) * 1000 + 1), nullptr) << "frame " << frame;
        }
        EXPECT_NE(h.find_prev(frame * 1000 + 1), nullptr) << "frame " << frame;
    }
}
