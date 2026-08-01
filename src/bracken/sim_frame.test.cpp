#include "bracken/sim_frame.hpp"

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/task_group.hpp"
#include "bracken/effects/effect_definition.hpp"
#include "bracken/effects/effect_system.hpp"
#include "bracken/input_handler.hpp"
#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/testing/frame_test.hpp"

#include <atomic>
#include <cstddef>
#include <gtest/gtest.h>
#include <SDL3/SDL_events.h>
#include <vector>

using bracken::testing::FrameAccess;
using bracken::InputHandler;
using bracken::InputSnapshot;
using bracken::SimFeedback;
using bracken::SimFrame;
using bracken::core::jobs::JobSystem;
using bracken::core::jobs::TaskGroup;
using bracken::scene::FrameBuilder;
using bracken::scene::FramePacket;

// All tests are headless: hand-built packets and builders, no Application,
// no window, no GPU. SimFrame's threading surface services the canonical
// singleton pool, so tests that exercise it initialize the singleton (the
// same pattern as parallel_for.test.cpp); accessor-only tests leave the pool
// alone (an uninitialized singleton runs parallel_for inline — same code
// path, one chunk).

namespace {

class SimFrameJobsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!JobSystem::instance().is_initialized()) {
            JobSystem::instance().init();
        }
    }
};

} // namespace

// --- Accessors over a hand-built packet ---

TEST(SimFrame, AccessorsExposeFrameContexts) {
    FramePacket packet;
    FrameBuilder fb(packet);
    InputSnapshot input;
    input.camera_yaw = 42.0f;
    input.mouse_left_pressed = true;

    SimFrame sim(fb, input, 0.016f, 1.25, 7);

    EXPECT_FLOAT_EQ(sim.dt(), 0.016f);
    EXPECT_DOUBLE_EQ(sim.time(), 1.25);
    EXPECT_EQ(sim.frame_index(), 7u);
    // sim.frame() is a FrameSink over the builder handed in — the host half
    // of it and nothing else (seal()/set_frame_timing() are not reachable).
    sim.frame().draw_model(bracken::ModelHandle{9}, glm::mat4(1.0f));
    EXPECT_EQ(fb.lane_count(), 1u);
    // input() is the exact snapshot handed in — by reference, no copy.
    EXPECT_EQ(&sim.input(), &input);
    EXPECT_FLOAT_EQ(sim.input().camera_yaw, 42.0f);
    EXPECT_TRUE(sim.input().mouse_left_pressed);

    FrameAccess::seal(fb);
    FrameAccess::reset(packet);
}

TEST(SimFrame, EmissionThroughFrameReachesSealedPacket) {
    FramePacket packet;
    FrameBuilder fb(packet);
    InputSnapshot input;
    SimFrame sim(fb, input, 0.016f, 0.0, 0);

    sim.frame().draw_model(bracken::ModelHandle{3}, glm::mat4(1.0f));
    sim.frame().ui().add_text("hello", 1.0f, 2.0f);
    FrameAccess::seal(fb);

    EXPECT_EQ(packet.model_commands().size(), 1u);
    EXPECT_EQ(packet.ui_commands().size(), 1u);
    FrameAccess::reset(packet);
}

// --- Feedback (set_camera_forward) ---

TEST(SimFrame, SetCameraForwardRecordsFeedback) {
    FramePacket packet;
    FrameBuilder fb(packet);
    InputSnapshot input;
    SimFeedback feedback;
    SimFrame sim(fb, input, 0.016f, 0.0, 0, &feedback);

    EXPECT_FALSE(feedback.has_camera_forward);
    sim.set_camera_forward({0.6f, -0.8f});
    EXPECT_TRUE(feedback.has_camera_forward);
    EXPECT_FLOAT_EQ(feedback.camera_forward.x, 0.6f);
    EXPECT_FLOAT_EQ(feedback.camera_forward.y, -0.8f);

    FrameAccess::seal(fb);
    FrameAccess::reset(packet);
}

// --- Viewport (P9: sim code must not read RenderContext) ---

TEST(SimFrame, CarriesTheKickTimeViewport) {
    FramePacket packet;
    FrameBuilder fb(packet);
    InputSnapshot input;
    SimFrame sim(fb, input, 0.016f, 0.0, 0, nullptr, 1600, 900);

    // Snapshotted on the main thread at kick: the swapchain may be
    // reconfiguring under the next frame's sim, so Application::screen_width()
    // is not readable from here.
    EXPECT_EQ(sim.screen_width(), 1600);
    EXPECT_EQ(sim.screen_height(), 900);

    FrameAccess::seal(fb);
    FrameAccess::reset(packet);
}

TEST(SimFrame, ViewportIsZeroWithoutARenderer) {
    FramePacket packet;
    FrameBuilder fb(packet);
    InputSnapshot input;
    SimFrame sim(fb, input, 0.016f, 0.0, 0);
    EXPECT_EQ(sim.screen_width(), 0);
    EXPECT_EQ(sim.screen_height(), 0);
    FrameAccess::seal(fb);
    FrameAccess::reset(packet);
}

TEST(SimFrame, SetCameraForwardWithoutFeedbackIsNoop) {
    FramePacket packet;
    FrameBuilder fb(packet);
    InputSnapshot input;
    SimFrame sim(fb, input, 0.016f, 0.0, 0, nullptr);
    sim.set_camera_forward({1.0f, 0.0f}); // must not crash
    FrameAccess::seal(fb);
    FrameAccess::reset(packet);
}

// --- Threading surface ---

TEST_F(SimFrameJobsTest, ParallelForIndexFormCoversRangeExactlyOnce) {
    FramePacket packet;
    FrameBuilder fb(packet);
    InputSnapshot input;
    SimFrame sim(fb, input, 0.016f, 0.0, 0);

    std::vector<int> hits(1024, 0);
    sim.parallel_for(std::size_t{0}, hits.size(), [&](std::size_t i) { hits[i] += 1; });
    for (std::size_t i = 0; i < hits.size(); ++i) {
        ASSERT_EQ(hits[i], 1) << "index " << i;
    }

    FrameAccess::seal(fb);
    FrameAccess::reset(packet);
}

TEST_F(SimFrameJobsTest, ParallelForRangeFormMutatesElements) {
    FramePacket packet;
    FrameBuilder fb(packet);
    InputSnapshot input;
    SimFrame sim(fb, input, 0.016f, 0.0, 0);

    std::vector<int> values(700, 2);
    sim.parallel_for(values, [](int& v) { v *= 3; }, /*grain=*/64);
    for (std::size_t i = 0; i < values.size(); ++i) {
        ASSERT_EQ(values[i], 6) << "index " << i;
    }

    FrameAccess::seal(fb);
    FrameAccess::reset(packet);
}

TEST_F(SimFrameJobsTest, RunAsyncJoinsThroughHostOwnedGroup) {
    FramePacket packet;
    FrameBuilder fb(packet);
    InputSnapshot input;
    SimFrame sim(fb, input, 0.016f, 0.0, 0);

    std::atomic<int> ran{0};
    TaskGroup group;
    sim.run_async(group, [&ran] { ran.fetch_add(1, std::memory_order_relaxed); });
    sim.run_async(group, [&ran] { ran.fetch_add(1, std::memory_order_relaxed); });
    JobSystem::instance().wait(group);
    EXPECT_EQ(ran.load(std::memory_order_relaxed), 2);

    FrameAccess::seal(fb);
    FrameAccess::reset(packet);
}

// --- Engine-internal builder API (the sim-stage tail's tools) ---

TEST(FrameBuilderEngine, ForEachEffectSpawnWalksLaneMajorOrder) {
    bracken::EffectDefinition def_a; // distinct addresses are all that matters
    bracken::EffectDefinition def_b;

    FramePacket packet;
    FrameBuilder fb(packet, /*lane_count=*/2);
    // Lane 1 emits FIRST in time, lane 0 second — the walk must still yield
    // lane 0 before lane 1 (deterministic lane-major order, matching seal).
    fb.lane(1).spawn_effect(&def_b, {4.0f, 5.0f, 6.0f});
    fb.lane(0).spawn_effect(&def_a, {1.0f, 2.0f, 3.0f});

    std::vector<const bracken::EffectDefinition*> order;
    FrameAccess::for_each_effect_spawn(fb, [&](const bracken::scene::EffectSpawnCommand& s) { order.push_back(s.definition); });
    ASSERT_EQ(order.size(), 2u);
    EXPECT_EQ(order[0], &def_a);
    EXPECT_EQ(order[1], &def_b);

    FrameAccess::seal(fb);
    // seal() merges the same lane-major order into the packet view.
    ASSERT_EQ(packet.effect_spawns().size(), 2u);
    EXPECT_EQ(packet.effect_spawns()[0].definition, &def_a);
    EXPECT_EQ(packet.effect_spawns()[1].definition, &def_b);
    FrameAccess::reset(packet);
}

TEST(FrameBuilderEngine, EngineFrameSectionRoundTripsThroughSeal) {
    FramePacket packet;
    FrameBuilder fb(packet);

    auto alloc = FrameAccess::alloc_effect_snapshot(fb, 1, 2);
    ASSERT_EQ(alloc.emitters.size(), 1u);
    ASSERT_EQ(alloc.particles.size(), 2u);
    alloc.emitters[0] = {nullptr, 0, 2};
    alloc.particles[0].position = {1.0f, 2.0f, 3.0f};
    alloc.particles[0].opacity = 0.5f;
    alloc.particles[1].position = {4.0f, 5.0f, 6.0f};
    alloc.particles[1].scale = 2.0f;
    FrameAccess::set_engine_frame(fb, /*scene_time=*/1.5f, /*sky_time=*/0.25f, /*effect_instance_count=*/3);
    FrameAccess::seal(fb);

    EXPECT_TRUE(packet.has_engine_frame());
    EXPECT_FLOAT_EQ(packet.scene_time(), 1.5f);
    EXPECT_FLOAT_EQ(packet.sky_time(), 0.25f);
    EXPECT_EQ(packet.effect_instance_count(), 3u);
    ASSERT_EQ(packet.effect_emitters().size(), 1u);
    ASSERT_EQ(packet.effect_particles().size(), 2u);
    EXPECT_EQ(packet.effect_emitters()[0].particle_count, 2u);
    EXPECT_FLOAT_EQ(packet.effect_particles()[0].opacity, 0.5f);
    EXPECT_FLOAT_EQ(packet.effect_particles()[1].scale, 2.0f);

    FrameAccess::reset(packet);
    // The engine section resets with the packet.
    FrameBuilder fb2(packet);
    FrameAccess::seal(fb2);
    EXPECT_FALSE(packet.has_engine_frame());
    EXPECT_EQ(packet.effect_emitters().size(), 0u);
    EXPECT_EQ(packet.effect_particles().size(), 0u);
    EXPECT_EQ(packet.effect_instance_count(), 0u);
    FrameAccess::reset(packet);
}

TEST(FrameBuilderEngine, EmptyEffectSnapshotRegistersEmptySpans) {
    FramePacket packet;
    FrameBuilder fb(packet);
    auto alloc = FrameAccess::alloc_effect_snapshot(fb, 0, 0);
    EXPECT_TRUE(alloc.emitters.empty());
    EXPECT_TRUE(alloc.particles.empty());
    FrameAccess::seal(fb);
    EXPECT_TRUE(packet.effect_emitters().empty());
    EXPECT_TRUE(packet.effect_particles().empty());
    FrameAccess::reset(packet);
}

// --- EffectSystem::snapshot_into (the sim-stage tail itself) ---

TEST(EffectSystemSnapshot, SnapshotMatchesLiveSystem) {
    using bracken::systems::EffectSystem;

    // One burst emitter that spawns 4 particles immediately.
    bracken::EffectDefinition def;
    def.name = "test_burst";
    bracken::EmitterDefinition emitter;
    emitter.name = "burst";
    emitter.spawn_mode = bracken::SpawnMode::BURST;
    emitter.spawn_count = 4;
    emitter.particle_lifetime = 10.0f;
    emitter.duration = 1.0f;
    def.emitters.push_back(emitter);

    EffectSystem system;
    ASSERT_GE(system.spawn_effect(&def, {1.0f, 2.0f, 3.0f}), 0);
    system.update(0.016f); // services the (possibly uninitialized) singleton inline

    ASSERT_GT(system.particle_count(), 0u);

    FramePacket packet;
    FrameBuilder fb(packet);
    system.snapshot_into(fb);
    FrameAccess::seal(fb);

    ASSERT_EQ(packet.effect_emitters().size(), 1u);
    ASSERT_EQ(packet.effect_particles().size(), system.particle_count());
    const auto& snap_emitter = packet.effect_emitters()[0];
    EXPECT_EQ(snap_emitter.definition, &def.emitters[0]);
    EXPECT_EQ(snap_emitter.first_particle, 0u);
    EXPECT_EQ(snap_emitter.particle_count, static_cast<uint32_t>(system.particle_count()));

    // Per-particle fields mirror the live system's particles, in order.
    const auto& live = system.get_effects()[0].emitters[0].particles;
    for (std::size_t i = 0; i < live.size(); ++i) {
        const auto& s = packet.effect_particles()[i];
        EXPECT_EQ(s.position, live[i].position) << i;
        EXPECT_EQ(s.rotation, live[i].rotation) << i;
        EXPECT_FLOAT_EQ(s.scale, live[i].scale) << i;
        EXPECT_FLOAT_EQ(s.opacity, live[i].opacity) << i;
        EXPECT_EQ(s.color, live[i].color) << i;
    }
    FrameAccess::reset(packet);
}

// --- InputSnapshot copy (headless: synthesized events, no SDL video) ---

TEST(InputSnapshotCopy, SnapshotMirrorsHandlerStateByValue) {
    InputHandler h;

    SDL_Event key{};
    key.type = SDL_EVENT_KEY_DOWN;
    key.key.scancode = SDL_SCANCODE_F3;
    key.key.down = true;
    key.key.repeat = false;
    h.process_event(key);

    SDL_Event click{};
    click.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
    click.button.button = SDL_BUTTON_LEFT;
    click.button.x = 10.0f;
    click.button.y = 20.0f;
    h.process_event(click);

    InputSnapshot s = h.snapshot();
    EXPECT_TRUE(s.was_key_just_pressed(SDL_SCANCODE_F3));
    EXPECT_FALSE(s.was_key_just_pressed(SDL_SCANCODE_F4));
    EXPECT_TRUE(s.mouse_left_pressed);
    EXPECT_TRUE(s.mouse_left_held);
    EXPECT_FLOAT_EQ(s.camera_yaw, h.get_camera_yaw());
    EXPECT_FLOAT_EQ(s.camera_pitch, h.get_camera_pitch());

    // By value: clearing the handler's edges must not touch the snapshot.
    h.clear_key_edges();
    h.clear_mouse_edges();
    EXPECT_TRUE(s.was_key_just_pressed(SDL_SCANCODE_F3));
    EXPECT_TRUE(s.mouse_left_pressed);
    EXPECT_FALSE(h.was_key_just_pressed(SDL_SCANCODE_F3));
}

TEST(InputSnapshotCopy, SnapshotDrainsTextInput) {
    InputHandler h;
    h.set_text_input_enabled(true);

    SDL_Event text{};
    text.type = SDL_EVENT_TEXT_INPUT;
    text.text.text = "hi";
    h.process_event(text);

    InputSnapshot s1 = h.snapshot();
    EXPECT_EQ(s1.text_input, "hi");
    // Drained: the next snapshot carries only the delta since this one.
    InputSnapshot s2 = h.snapshot();
    EXPECT_TRUE(s2.text_input.empty());
}
