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
