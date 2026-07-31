#include <gtest/gtest.h>

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/task_group.hpp"
#include "bracken/core/jobs/task_handle.hpp"
#include "bracken/effects/effect_definition.hpp"
#include "bracken/effects/effect_system.hpp"
#include "bracken/input_handler.hpp"
#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/frame_packet_ring.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <SDL3/SDL_events.h>
#include <set>
#include <stdexcept>
#include <string>

using bracken::InputHandler;
using bracken::InputSnapshot;
using bracken::core::jobs::JobSystem;
using bracken::core::jobs::TaskGroup;
using bracken::core::jobs::TaskHandle;
using bracken::scene::FrameBuilder;
using bracken::scene::FramePacket;
using bracken::scene::PacketRing;

// ============================================================================
// Pipelined-frame tests, complementing frame_packet_ring.test.cpp (raw-thread
// SPSC handoff) and application.test.cpp (run()'s schedule): here the sim
// stage runs AS A POOL JOB the way Application::kick_sim/join_sim drive it —
// kick(i+1) after join(i), handoff riding the TaskHandle kick/join edges with
// no ring publish — while the "render" side (this test's main thread) walks
// the previous sealed packet CONCURRENTLY with the in-flight fill.
//
// All headless: no window, no GPU, no Application. Synchronization is only
// the jobs primitives (submit/wait, TaskGroup) — no sleeps, no yields.
// ============================================================================

namespace {

class FramePipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!JobSystem::instance().is_initialized()) {
            JobSystem::instance().init();
        }
    }
};

constexpr uint32_t k_lanes = 4;
constexpr float k_dt = 0.016f;

/// Sim-job body for `frame`: fan emission out across k_lanes lanes (one
/// TaskGroup job per lane — the documented lane-per-job contract), join,
/// emit the singular tail from the job root, seal.
void fill_and_seal(FrameBuilder& fb, uint64_t frame) {
    JobSystem& js = JobSystem::instance();
    TaskGroup group;
    for (uint32_t j = 0; j < k_lanes; ++j) {
        js.submit(group, [&fb, frame, j] {
            auto& lane = fb.lane(j);
            glm::mat4 xf(1.0f);
            xf[0][0] = static_cast<float>(frame + 1);
            xf[1][1] = static_cast<float>(j + 1);
            lane.draw_model(static_cast<bracken::ModelHandle>(100 + j), xf, {1.0f, 1.0f, 1.0f, 1.0f},
                            bracken::scene::DrawFlags::None,
                            static_cast<uint32_t>(frame * k_lanes + j + 1));
            lane.add_debug_line({static_cast<float>(frame), static_cast<float>(j), 0.0f},
                                {static_cast<float>(j), static_cast<float>(frame), 1.0f},
                                0xFF000000u | static_cast<uint32_t>(frame * k_lanes + j));
        });
    }
    js.wait(group); // all emitting jobs joined before any further lane-0 use / seal

    // Singular tail on the job root (single writer), lane 0 for the palette.
    auto pal = fb.alloc_bone_palette(3);
    ASSERT_TRUE(pal.valid());
    for (uint32_t k = 0; k < 3; ++k) {
        pal.matrices[k] = glm::mat4(static_cast<float>(frame + k));
    }
    fb.draw_skinned(static_cast<bracken::ModelHandle>(200), glm::mat4(1.0f), pal.id);
    fb.ui().add_text("frame " + std::to_string(frame), 4.0f, 8.0f);
    fb.set_draw_skybox(frame % 2 == 0);
    fb.seal();
}

/// Walk one sealed packet and count every deviation from frame `frame`'s
/// expected content. Zero mismatches == the packet is exactly frame's
/// description: bit-stable (no tearing from the concurrent fill of the other
/// slot) and free of any leakage from the slot's previous occupant.
int count_mismatches(const FramePacket& p, uint64_t frame) {
    int bad = 0;
    if (p.state() != FramePacket::State::Sealed) {
        ++bad;
    }
    if (p.frame_index() != frame || p.dt() != k_dt || p.time() != static_cast<double>(frame) * k_dt) {
        ++bad;
    }
    if (p.should_draw_skybox() != (frame % 2 == 0)) {
        ++bad;
    }

    const auto models = p.model_commands();
    if (models.size() != k_lanes) {
        ++bad;
    } else {
        for (uint32_t j = 0; j < k_lanes; ++j) { // lane-major merge order
            const auto& m = models[j];
            glm::mat4 xf(1.0f);
            xf[0][0] = static_cast<float>(frame + 1);
            xf[1][1] = static_cast<float>(j + 1);
            if (m.model_handle != 100 + j || m.motion_id != frame * k_lanes + j + 1 ||
                std::memcmp(&m.transform, &xf, sizeof(xf)) != 0) {
                ++bad;
            }
        }
    }

    const auto lines = p.debug_lines();
    if (lines.size() != k_lanes) {
        ++bad;
    } else {
        for (uint32_t j = 0; j < k_lanes; ++j) {
            if (lines[j].color != (0xFF000000u | static_cast<uint32_t>(frame * k_lanes + j)) ||
                lines[j].start != glm::vec3(static_cast<float>(frame), static_cast<float>(j), 0.0f)) {
                ++bad;
            }
        }
    }

    const auto skinned = p.skinned_commands();
    if (skinned.size() != 1 || !skinned[0].palette.valid()) {
        ++bad;
    } else {
        const auto mats = p.palette_matrices(skinned[0].palette);
        if (mats.size() != 3) {
            ++bad;
        } else {
            for (uint32_t k = 0; k < 3; ++k) {
                const glm::mat4 expect(static_cast<float>(frame + k));
                if (std::memcmp(&mats[k], &expect, sizeof(expect)) != 0) {
                    ++bad;
                }
            }
        }
    }

    const auto ui = p.ui_commands();
    if (ui.size() != 1 || !ui[0].is<bracken::scene::UiTextCommand>() ||
        ui[0].get<bracken::scene::UiTextCommand>().text != "frame " + std::to_string(frame)) {
        ++bad;
    }
    return bad;
}

} // namespace

// ---------------------------------------------------------------------------
// 1. Pipelined overlap, Application-shaped: main thread walks sealed packet i
// (twice — once before kicking sim i+1, once while it runs) and must see the
// identical, complete frame-i description both times, for every frame.
// ---------------------------------------------------------------------------

TEST_F(FramePipelineTest, SimJobFillsNextPacketWhileMainReadsSealedPrevious) {
    constexpr uint64_t k_frames = 200;
    JobSystem& js = JobSystem::instance();
    PacketRing ring(2);

    std::unique_ptr<FrameBuilder> builder; // outlives the job; replaced at each kick
    FramePacket* in_flight = nullptr;
    TaskHandle task;

    auto kick = [&](uint64_t frame) {
        FramePacket& p = ring.acquire_for_fill();
        builder = std::make_unique<FrameBuilder>(p, k_lanes);
        builder->set_frame_timing(k_dt, static_cast<double>(frame) * k_dt, frame); // stamped at kick, like run()
        in_flight = &p;
        FrameBuilder& fb = *builder;
        task = js.submit([&fb, frame] { fill_and_seal(fb, frame); });
    };

    kick(0); // bootstrap frame, no overlap (run()'s shape)
    for (uint64_t frame = 0; frame < k_frames; ++frame) {
        js.wait(task); // join(frame) — the handoff edge; packet is sealed
        FramePacket* cur = in_flight;
        ASSERT_NE(cur, nullptr);
        ASSERT_EQ(cur->state(), FramePacket::State::Sealed);

        // Walk A: quiescent (no sim in flight).
        ASSERT_EQ(count_mismatches(*cur, frame), 0) << "walk A, frame " << frame;

        if (frame + 1 < k_frames) {
            kick(frame + 1); // Simulate(i+1) now fills the OTHER slot on the pool
            ASSERT_NE(in_flight, cur); // depth-2 ring: never the packet being read
        }

        // Walk B: concurrent with the in-flight fill — must be bit-identical
        // to walk A (slot isolation; the fill touches only the other packet).
        ASSERT_EQ(count_mismatches(*cur, frame), 0) << "walk B, frame " << frame;

        ring.retire(*cur); // frame fence: happens-before the slot's next acquire
    }

    // Everything joined and retired: the ring ends quiescent, slots reusable.
    EXPECT_EQ(ring.packet(0).state(), FramePacket::State::Retired);
    EXPECT_EQ(ring.packet(1).state(), FramePacket::State::Retired);
}

// ---------------------------------------------------------------------------
// 2. Sim-error path at the ring level (application.test.cpp covers the same
// policy through run()): a sim job that throws mid-fill propagates out of the
// join on the coordinating thread, and the abandoned Filling packet resets
// back to Retired so the slot cycles cleanly — no poisoned ring state.
// ---------------------------------------------------------------------------

TEST_F(FramePipelineTest, ThrowingSimJobRethrowsAtJoinAndPacketIsAbandonable) {
    JobSystem& js = JobSystem::instance();
    PacketRing ring(2);

    FramePacket& p = ring.acquire_for_fill();
    auto builder = std::make_unique<FrameBuilder>(p, k_lanes);
    builder->set_frame_timing(k_dt, 0.0, 0);
    FrameBuilder& fb = *builder;
    TaskHandle task = js.submit([&fb] {
        fb.draw_model(static_cast<bracken::ModelHandle>(1), glm::mat4(1.0f)); // partial fill
        throw std::runtime_error("sim boom");
    });
    EXPECT_THROW(js.wait(task), std::runtime_error);

    // Abandon the partial packet exactly as Application::run does: drop the
    // builder, reset the packet (any state -> Retired).
    EXPECT_EQ(p.state(), FramePacket::State::Filling);
    builder.reset();
    p.reset();
    EXPECT_EQ(p.state(), FramePacket::State::Retired);

    // Both slots (the untouched one at frame 0, then the ABANDONED one at
    // frame 1 — the round-robin cursor wraps back to it) run a clean frame
    // afterwards. count_mismatches checks exact counts, so any residue from
    // the partial fill would surface as an extra command.
    for (uint64_t frame = 0; frame < 2; ++frame) {
        FramePacket& q = ring.acquire_for_fill();
        if (frame == 1) {
            ASSERT_EQ(&q, &p); // the abandoned slot, cycled back for reuse
        }
        FrameBuilder clean(q, k_lanes);
        clean.set_frame_timing(k_dt, static_cast<double>(frame) * k_dt, frame);
        TaskHandle t = js.submit([&clean, frame] { fill_and_seal(clean, frame); });
        js.wait(t);
        EXPECT_EQ(count_mismatches(q, frame), 0) << "post-abandon frame " << frame;
        ring.retire(q);
    }
}

// ---------------------------------------------------------------------------
// 3. Effect-instance snapshot stability: the sim job advances the LIVE
// EffectSystem and snapshots it into packet i+1 while the reader walks packet
// i's snapshot — the previous frame's snapshot must be bit-stable even though
// the system it was taken from is mutating concurrently.
// ---------------------------------------------------------------------------

namespace {

/// FNV-1a over the packet's whole effect snapshot (emitter records + raw
/// particle bytes) — any torn/mutated read changes the digest.
uint64_t effect_snapshot_digest(const FramePacket& p) {
    static_assert(std::is_trivially_copyable_v<bracken::scene::EffectParticleSnapshot>);
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const void* data, std::size_t n) {
        const auto* b = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < n; ++i) {
            h = (h ^ b[i]) * 1099511628211ull;
        }
    };
    const auto emitters = p.effect_emitters();
    const auto particles = p.effect_particles();
    const uint64_t counts[3] = {emitters.size(), particles.size(), p.effect_instance_count()};
    mix(counts, sizeof(counts));
    for (const auto& e : emitters) {
        const auto def = reinterpret_cast<uintptr_t>(e.definition);
        mix(&def, sizeof(def));
        mix(&e.first_particle, sizeof(e.first_particle));
        mix(&e.particle_count, sizeof(e.particle_count));
    }
    for (const auto& s : particles) {
        mix(&s, sizeof(s));
    }
    return h;
}

} // namespace

TEST_F(FramePipelineTest, PreviousPacketEffectSnapshotIsBitStableWhileSimAdvances) {
    constexpr uint64_t k_frames = 120;
    JobSystem& js = JobSystem::instance();

    // A continuously churning emitter: particles spawn and die every frame,
    // so consecutive snapshots genuinely differ.
    bracken::EffectDefinition def;
    def.name = "pipeline_churn";
    bracken::EmitterDefinition emitter;
    emitter.name = "churn";
    emitter.spawn_mode = bracken::SpawnMode::CONTINUOUS;
    emitter.spawn_rate = 400.0f;
    emitter.particle_lifetime = 0.25f;
    emitter.duration = 1000.0f;
    def.emitters.push_back(emitter);

    bracken::systems::EffectSystem system; // sim-stage owned: touched only inside the sim job
    ASSERT_GE(system.spawn_effect(&def, {1.0f, 2.0f, 3.0f}), 0);

    PacketRing ring(2);
    std::unique_ptr<FrameBuilder> builder;
    FramePacket* in_flight = nullptr;
    TaskHandle task;

    auto kick = [&](uint64_t frame) {
        FramePacket& p = ring.acquire_for_fill();
        builder = std::make_unique<FrameBuilder>(p);
        builder->set_frame_timing(k_dt, static_cast<double>(frame) * k_dt, frame);
        in_flight = &p;
        FrameBuilder& fb = *builder;
        // The Application sim-job tail: update the live system, snapshot into
        // THIS frame's packet, stamp the engine section, seal.
        task = js.submit([&fb, &system, frame] {
            system.update(k_dt); // nested fork-join on the pool
            system.snapshot_into(fb);
            fb.set_engine_frame(static_cast<float>(frame + 1) * k_dt, static_cast<float>(frame + 1) * k_dt,
                                static_cast<uint32_t>(system.effect_count()));
            fb.seal();
        });
    };

    std::set<std::size_t> particle_counts;
    kick(0);
    for (uint64_t frame = 0; frame < k_frames; ++frame) {
        js.wait(task);
        FramePacket* cur = in_flight;
        ASSERT_EQ(cur->state(), FramePacket::State::Sealed);
        ASSERT_TRUE(cur->has_engine_frame());

        const uint64_t digest_quiescent = effect_snapshot_digest(*cur);
        particle_counts.insert(cur->effect_particles().size());

        if (frame + 1 < k_frames) {
            kick(frame + 1); // sim now mutates the live system + fills the other slot
        }

        // Concurrent re-walk: packet i's snapshot must not move a bit while
        // the system it snapshotted is being updated and re-snapshotted.
        ASSERT_EQ(effect_snapshot_digest(*cur), digest_quiescent) << "frame " << frame;

        ring.retire(*cur);
    }

    // The guard that makes the stability assertion meaningful: the live
    // system really did change across frames (snapshots weren't all equal
    // because nothing was happening).
    EXPECT_GT(particle_counts.size(), 1u);
}

// ---------------------------------------------------------------------------
// 4. Input-snapshot independence under post-snapshot MUTATION (the
// complementary direction to sim_frame.test.cpp's edge-clearing tests):
// feeding the live handler new events after snapshot() must not move any
// captured field.
// ---------------------------------------------------------------------------

TEST(InputSnapshotIndependence, NewEventsAfterSnapshotDoNotAlterSnapshot) {
    InputHandler h;

    // Build non-default state through the event path (headless-safe events).
    SDL_Event motion{};
    motion.type = SDL_EVENT_MOUSE_MOTION;
    motion.motion.x = 50.0f;
    motion.motion.y = 60.0f;
    h.process_event(motion);

    SDL_Event wheel{};
    wheel.type = SDL_EVENT_MOUSE_WHEEL;
    wheel.wheel.y = 1.0f;
    h.process_event(wheel);

    SDL_Event key{};
    key.type = SDL_EVENT_KEY_DOWN;
    key.key.scancode = SDL_SCANCODE_F5;
    key.key.down = true;
    h.process_event(key);

    SDL_Event click{};
    click.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
    click.button.button = SDL_BUTTON_LEFT;
    click.button.x = 50.0f;
    click.button.y = 60.0f;
    h.process_event(click);

    const InputSnapshot snap = h.snapshot();
    // Reference values captured from the handler at snapshot time.
    const float yaw_before = h.get_camera_yaw();
    const float zoom_before = h.get_camera_zoom_delta();

    // Mutate the live handler every way the events above can.
    motion.motion.x = 300.0f;
    motion.motion.y = 400.0f;
    h.process_event(motion);
    wheel.wheel.y = -3.0f;
    h.process_event(wheel);
    key.type = SDL_EVENT_KEY_UP;
    key.key.down = false;
    h.process_event(key);
    SDL_Event key2{};
    key2.type = SDL_EVENT_KEY_DOWN;
    key2.key.scancode = SDL_SCANCODE_F6;
    key2.key.down = true;
    h.process_event(key2);
    SDL_Event release{};
    release.type = SDL_EVENT_MOUSE_BUTTON_UP;
    release.button.button = SDL_BUTTON_LEFT;
    h.process_event(release);
    h.clear_key_edges();
    h.clear_mouse_edges();
    h.reset_camera_deltas();

    // The live handler moved...
    EXPECT_FLOAT_EQ(h.mouse_x(), 300.0f);
    EXPECT_FALSE(h.mouse_left_held());
    EXPECT_FALSE(h.was_key_just_pressed(SDL_SCANCODE_F5));
    EXPECT_FLOAT_EQ(h.get_camera_zoom_delta(), 0.0f);

    // ...and the snapshot did not: every field still reads as captured.
    EXPECT_FLOAT_EQ(snap.mouse_x, 50.0f);
    EXPECT_FLOAT_EQ(snap.mouse_y, 60.0f);
    EXPECT_TRUE(snap.mouse_left_held);
    EXPECT_TRUE(snap.mouse_left_pressed);
    EXPECT_FALSE(snap.mouse_left_released);
    EXPECT_TRUE(snap.was_key_just_pressed(SDL_SCANCODE_F5));
    EXPECT_FALSE(snap.was_key_just_pressed(SDL_SCANCODE_F6));
    EXPECT_FLOAT_EQ(snap.camera_yaw, yaw_before);
    EXPECT_FLOAT_EQ(snap.camera_zoom_delta, zoom_before);
}

TEST(InputSnapshotIndependence, SnapshotCopiesAreIndependentValues) {
    InputHandler h;
    h.set_text_input_enabled(true);
    SDL_Event text{};
    text.type = SDL_EVENT_TEXT_INPUT;
    text.text.text = "abc";
    h.process_event(text);

    InputSnapshot a = h.snapshot();
    InputSnapshot b = a; // copy: worker-safe immutable value semantics
    a.text_input.clear();
    a.key_just_pressed[SDL_SCANCODE_A] = true;
    EXPECT_EQ(b.text_input, "abc");
    EXPECT_FALSE(b.was_key_just_pressed(SDL_SCANCODE_A));
}
