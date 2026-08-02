// The public headless frame fixture, and the boundary it exists to guard.
//
// Two jobs here. The first is to prove the fixture actually works the way a
// game developer's test suite would use it — build a frame through a
// FrameSink, seal, read back. The second is the phase-8 gate: that a
// host holding a FrameSink CANNOT reach the engine's sim-stage tail. That used
// to be a sentence in a doc comment backed by a debug assert; here it is a
// compile-time fact.

#include <gtest/gtest.h>

#include "bracken/scene/frame_sink.hpp"
#include "bracken/testing/frame_test.hpp"

#include <concepts>
#include <cstdint>
#include <glm/glm.hpp>
#include <string_view>

using bracken::testing::FrameAccess;
using bracken::testing::TestFrame;
using namespace bracken::scene;

namespace {

// Detectors. A member that is absent AND a member that is private both make
// the requires-expression ill-formed, so one concept covers both halves of the
// split: FrameSink does not declare the tail, FrameBuilder declares it private.
template<typename T> concept CanSeal = requires(T& t) { t.seal(); };
template<typename T> concept CanStampTiming = requires(T& t) { t.set_frame_timing(0.0f, 0.0, uint64_t{0}); };
template<typename T> concept CanAllocSnapshot = requires(T& t) { t.alloc_effect_snapshot(0u, 0u); };
template<typename T> concept CanSetEngineFrame = requires(T& t) { t.set_engine_frame(0.0f, 0.0f, 0u); };
template<typename T> concept CanReadStagedEnv = requires(const T& t) { t.staged_environment(); };
template<typename T> concept CanReachPacket = requires(T& t) { t.packet(); };
template<typename T> concept CanBeginFill = requires(T& t) { t.begin_fill(); };
template<typename T> concept CanResetPacket = requires(T& t) { t.reset(); };
template<typename T> concept CanReserveBlock = requires(T& t) { t.reserve_block(16u, 16u); };

// The host surface a FrameSink DOES carry — if these ever stop compiling the
// negatives below would pass vacuously.
template<typename T> concept CanDraw = requires(T& t) { t.draw_model(bracken::ModelHandle{1}, glm::mat4(1.0f)); };
template<typename T> concept CanSetCamera = requires(T& t) { t.set_camera(CameraState{}); };

static_assert(CanDraw<FrameSink>);
static_assert(CanSetCamera<FrameSink>);
static_assert(CanDraw<FrameBuilder>);

// THE GATE. Neither the host's view nor the builder itself lets host code run
// the engine tail.
static_assert(!CanSeal<FrameSink>);
static_assert(!CanStampTiming<FrameSink>);
static_assert(!CanAllocSnapshot<FrameSink>);
static_assert(!CanSetEngineFrame<FrameSink>);
static_assert(!CanReadStagedEnv<FrameSink>);
static_assert(!CanReachPacket<FrameSink>);

static_assert(!CanSeal<FrameBuilder>);
static_assert(!CanStampTiming<FrameBuilder>);
static_assert(!CanAllocSnapshot<FrameBuilder>);
static_assert(!CanSetEngineFrame<FrameBuilder>);
static_assert(!CanReadStagedEnv<FrameBuilder>);
static_assert(!CanReachPacket<FrameBuilder>);

static_assert(!CanBeginFill<FramePacket>);
static_assert(!CanResetPacket<FramePacket>);
static_assert(!CanReserveBlock<FramePacket>);

// A FrameSink is a handle, not a frame: it must stay cheap enough that passing
// it by value through a host's own helpers costs nothing.
static_assert(sizeof(FrameSink) == sizeof(void*));
static_assert(std::is_trivially_copyable_v<FrameSink>);

// A stand-in for the game code under test: it only ever sees a FrameSink.
void describe_frame(FrameSink& frame, int orb_count) {
    CameraState cam{};
    cam.position = {0.0f, 10.0f, 20.0f};
    frame.set_camera(cam);
    frame.set_draw_grass(false);

    for (int i = 0; i < orb_count; ++i) {
        glm::mat4 xf(1.0f);
        xf[3][0] = static_cast<float>(i);
        frame.draw_model(static_cast<bracken::ModelHandle>(i + 1), xf, {1, 1, 1, 1}, DrawFlags::NoFog,
                         static_cast<uint32_t>(100 + i));
        frame.add_point_light({{static_cast<float>(i), 0.0f, 0.0f}, 50.0f, {1.0f, 0.5f, 0.25f}, 100.0f, true});
    }
    frame.ui().add_text("orbs", 8.0f, 8.0f);
}

} // namespace

TEST(TestFrame, DrivesHostFrameCodeAndReadsItBack) {
    TestFrame frame;
    describe_frame(frame.sink(), 3);
    frame.seal();

    ASSERT_EQ(frame.draws().size(), 3u);
    EXPECT_EQ(frame.draws()[0].model_handle, 1u);
    EXPECT_EQ(frame.draws()[2].motion_id, 102u);
    EXPECT_TRUE(has(frame.draws()[2].flags, DrawFlags::NoFog));

    ASSERT_EQ(frame.point_lights().size(), 3u);
    EXPECT_EQ(frame.point_lights()[0].radius, 50.0f);
    EXPECT_TRUE(frame.point_lights()[0].casts_shadows);

    ASSERT_EQ(frame.ui_text().size(), 1u);
    EXPECT_EQ(frame.ui_text()[0], "orbs");

    EXPECT_TRUE(frame.packet().has_camera());
    EXPECT_FALSE(frame.packet().should_draw_grass());
    EXPECT_TRUE(frame.sealed());
}

TEST(TestFrame, BorrowsACallerOwnedPacket) {
    FramePacket packet;
    {
        TestFrame frame(packet);
        frame.set_frame_timing(0.016f, 1.0, 42);
        frame.sink().draw_model(7, glm::mat4(1.0f));
        frame.seal();
        EXPECT_EQ(frame.draws().size(), 1u);
    }
    // The packet outlives the fixture and is still readable — the fixture owns
    // the builder, not the frame.
    EXPECT_EQ(packet.state(), FramePacket::State::Sealed);
    EXPECT_EQ(packet.frame_index(), 42u);
    FrameAccess::reset(packet);
}

TEST(TestFrame, ReportsWhatTheFillRefused) {
    TestFrame frame;
    frame.sink().draw_model(bracken::INVALID_MODEL_HANDLE, glm::mat4(1.0f));
    frame.sink().draw_model(3, glm::mat4(1.0f));
    frame.seal();

    EXPECT_EQ(frame.draws().size(), 1u);
    EXPECT_EQ(frame.build_diagnostics().draws_dropped_invalid_handle, 1u);
}

TEST(TestFrame, LanesAreReachableForFannedOutEmission) {
    TestFrame frame(4);
    ASSERT_EQ(frame.sink().lane_count(), 4u);
    for (uint32_t i = 0; i < 4; ++i) {
        frame.sink().lane(i).draw_model(static_cast<bracken::ModelHandle>(i + 1), glm::mat4(1.0f));
    }
    frame.seal();

    // Lane-major merge order, the same guarantee the engine's own frames get.
    ASSERT_EQ(frame.draws().size(), 4u);
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(frame.draws()[i].model_handle, i + 1);
    }
}

// The R2a packet-API change, from a host's side of the fence: PointLightDesc
// and SpotLightDesc grew a `stable_id`, and the field was APPENDED with a
// default precisely so that every positional aggregate initializer a game
// already wrote keeps compiling and keeps meaning what it meant. If this test
// ever needs editing to compile, the change was not source-compatible.
TEST(TestFrame, LightDescsStaySourceCompatibleAndDefaultToEngineAssignedIds) {
    TestFrame frame;
    auto& sink = frame.sink();

    // Pre-R2a spellings, verbatim: positional, no stable_id in sight.
    sink.add_point_light({{1.0f, 2.0f, 3.0f}, 50.0f, {1.0f, 0.5f, 0.25f}, 100.0f, true});
    sink.add_spot_light({{4.0f, 5.0f, 6.0f},
                         30.0f,
                         {0.0f, -1.0f, 0.0f},
                         bracken::core::Degrees{18.0f},
                         bracken::core::Degrees{25.0f},
                         {0.2f, 0.4f, 1.0f},
                         80.0f,
                         false});
    // ...and the new spelling, for a host that owns its own identity space.
    bracken::render::lighting::PointLightDesc tagged;
    tagged.position = {7.0f, 8.0f, 9.0f};
    tagged.radius = 10.0f;
    tagged.intensity = 5.0f;
    tagged.stable_id = 0xABCDEFu;
    sink.add_point_light(tagged);

    frame.seal();

    ASSERT_EQ(frame.point_lights().size(), 2u);
    ASSERT_EQ(frame.spot_lights().size(), 1u);
    // 0 = "engine assigns"; the packet carries the host's answer unchanged and
    // the renderer resolves it during binning, not here.
    EXPECT_EQ(frame.point_lights()[0].stable_id, 0u);
    EXPECT_EQ(frame.spot_lights()[0].stable_id, 0u);
    EXPECT_EQ(frame.point_lights()[1].stable_id, 0xABCDEFu);
    // The fields the old initializers named still land where they landed.
    EXPECT_EQ(frame.point_lights()[0].radius, 50.0f);
    EXPECT_TRUE(frame.point_lights()[0].casts_shadows);
    EXPECT_EQ(frame.spot_lights()[0].radius, 30.0f);
    EXPECT_FALSE(frame.spot_lights()[0].casts_shadows);
}
