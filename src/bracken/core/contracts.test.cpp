// The host-boundary contracts, proven to abort — IN RELEASE BUILDS TOO, which
// is the whole point of BRACKEN_CONTRACT existing separately from
// BRACKEN_ASSERT. There is no switch that compiles them out.
//
// Each case below is a mistake a GAME makes against the engine, not a bug
// inside the engine. Without the contract, every one of them is a null
// dereference or a silent write into memory another stage is reading — the
// kind of failure that surfaces three subsystems away from its cause, in a
// shipped build, where nobody has a debugger.
//
// Suite names end in "DeathTest" so gtest schedules them before threaded
// suites (a forked death test in a threaded process is unreliable).

#include <gtest/gtest.h>

#include "bracken/core/assert.hpp"
#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/testing/frame_test.hpp"

#include <glm/glm.hpp>
#include <thread>

using bracken::testing::FrameAccess;
using namespace bracken;
using namespace bracken::scene;

namespace {


TEST(ContractsDeathTest, ReadingAPacketBeforeSealAborts) {
    EXPECT_DEATH(
        {
            FramePacket packet{1 << 16};
            FrameBuilder fb(packet);
            fb.draw_model(static_cast<ModelHandle>(1), glm::mat4(1.0f));
            (void)packet.model_commands(); // still Filling
        },
        "FramePacket read before seal");
}

TEST(ContractsDeathTest, SealingTwiceAborts) {
    EXPECT_DEATH(
        {
            FramePacket packet{1 << 16};
            FrameBuilder fb(packet);
            FrameAccess::seal(fb);
            FrameAccess::seal(fb);
        },
        "seal called twice");
}

TEST(ContractsDeathTest, EmittingAfterSealAborts) {
    EXPECT_DEATH(
        {
            FramePacket packet{1 << 16};
            FrameBuilder fb(packet);
            FrameAccess::seal(fb);
            fb.draw_model(static_cast<ModelHandle>(1), glm::mat4(1.0f));
        },
        "outside the Filling state");
}

TEST(ContractsDeathTest, OutOfRangeLaneIndexAborts) {
    EXPECT_DEATH(
        {
            FramePacket packet{1 << 16};
            FrameBuilder fb(packet, 4);
            fb.lane(4).draw_model(static_cast<ModelHandle>(1), glm::mat4(1.0f));
        },
        "index 4 out of range");
}


// The legal counterpart of the lane contract, and the reason the lane guard
// detects OVERLAP rather than binding an owner: a lane changes threads all the
// time — emit from a job, join, then continue emitting from the joining
// thread. Application's own sim tail does exactly this. It must NOT abort.
TEST(Contracts, LaneMayChangeThreadsAcrossAJoin) {
    FramePacket packet{1 << 16};
    FrameBuilder fb(packet, 2);

    std::thread worker([&fb] { fb.lane(0).draw_model(static_cast<ModelHandle>(7), glm::mat4(1.0f)); });
    worker.join(); // the happens-before edge that makes the handoff safe

    fb.lane(0).draw_model(static_cast<ModelHandle>(8), glm::mat4(1.0f));
    FrameAccess::seal(fb);

    ASSERT_EQ(packet.model_commands().size(), 2u);
    EXPECT_EQ(packet.model_commands()[0].model_handle, 7u);
    EXPECT_EQ(packet.model_commands()[1].model_handle, 8u);
}

TEST(Contracts, ConditionIsEvaluatedExactlyOnce) {
    int calls = 0;
    auto observed = [&calls]() {
        ++calls;
        return true;
    };
    BRACKEN_CONTRACT(observed(), "never fires");
    EXPECT_EQ(calls, 1);
}

} // namespace
