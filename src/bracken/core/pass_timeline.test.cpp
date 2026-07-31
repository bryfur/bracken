#include "bracken/core/pass_timeline.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

namespace bracken::core {
namespace {

// The point of the whole file: 19 BRACKEN_GPU_TIMER scopes used to report
// 0.0 ms because SDL3 has no timestamp query API. They now report a REAL
// number — of the CPU record cost, which is a different quantity and is
// labelled as such everywhere it surfaces.
TEST(PassTimeline, ReportsNothingUntilAPassCompletes) {
    PassTimeline t;
    t.begin_frame();
    t.reserve(0, "Shadows");
    EXPECT_FALSE(t.has_sample(0));
    EXPECT_EQ(t.pass_ms(0), 0.0f);

    t.begin_pass(0);
    EXPECT_FALSE(t.has_sample(0)); // still open
    t.end_pass(0);
    EXPECT_TRUE(t.has_sample(0));
}

TEST(PassTimeline, MeasuresRealElapsedTime) {
    PassTimeline t;
    t.begin_frame();
    t.reserve(0, "Sleep");
    t.begin_pass(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    t.end_pass(0);

    // Generous bounds: this asserts the clock is wired up, not the scheduler.
    EXPECT_GT(t.pass_ms(0), 1.0f);
    EXPECT_LT(t.pass_ms(0), 500.0f);
}

TEST(PassTimeline, FirstSampleIsNotRampedUpFromZero) {
    PassTimeline t;
    t.begin_frame();
    t.reserve(0, "Sleep");
    t.begin_pass(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    t.end_pass(0);
    const float first = t.pass_ms(0);

    // An EMA seeded at 0 would report ~20% of the truth on frame one, which is
    // exactly when someone is looking at it.
    EXPECT_GT(first, 1.0f);
}

TEST(PassTimeline, AverageConvergesTowardTheSteadyValue) {
    PassTimeline t;
    // Frame 1: an outlier. Frames 2..N: a fast pass. The average must move
    // toward the fast value rather than staying pinned to the hitch.
    t.begin_frame();
    t.reserve(0, "P");
    t.begin_pass(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    t.end_pass(0);
    const float after_hitch = t.pass_ms(0);

    for (int i = 0; i < 40; ++i) {
        t.begin_frame();
        t.reserve(0, "P");
        t.begin_pass(0);
        t.end_pass(0); // ~0 ms
    }
    EXPECT_LT(t.pass_ms(0), after_hitch * 0.5f);
}

TEST(PassTimeline, ANameChangeDiscardsTheSlotsAverage) {
    PassTimeline t;
    t.begin_frame();
    t.reserve(0, "Slow");
    t.begin_pass(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    t.end_pass(0);
    ASSERT_TRUE(t.has_sample(0));

    // A conditional pass came or went and shifted the ordering: slot 0 is now
    // a DIFFERENT pass, and blending the two would describe neither.
    t.begin_frame();
    t.reserve(0, "Fast");
    EXPECT_FALSE(t.has_sample(0));
    EXPECT_EQ(t.pass_ms(0), 0.0f);
}

TEST(PassTimeline, EndWithoutBeginProducesNoSample) {
    PassTimeline t;
    t.begin_frame();
    t.reserve(0, "P");
    t.end_pass(0);
    EXPECT_FALSE(t.has_sample(0));
}

TEST(PassTimeline, BeginFrameClosesAnyStrandedOpenPass) {
    PassTimeline t;
    t.begin_frame();
    t.reserve(0, "P");
    t.begin_pass(0); // never ended (an early-out inside the pass)
    t.begin_frame();
    t.end_pass(0);   // stale end from the previous frame
    EXPECT_FALSE(t.has_sample(0));
}

TEST(PassTimeline, OutOfRangeSlotsAreIgnoredNotUndefined) {
    PassTimeline t;
    t.begin_frame();
    const std::size_t over = PassTimeline::k_max_passes + 7;
    t.reserve(over, "P");
    t.begin_pass(over);
    t.end_pass(over);
    EXPECT_FALSE(t.has_sample(over));
    EXPECT_EQ(t.pass_ms(over), 0.0f);
}

} // namespace
} // namespace bracken::core
