#pragma once

// bracken/testing/frame_test.hpp — headless frame testing, for GAMES.
//
// FrameBuilder/FramePacket are a host's entire output surface, and they are
// perfectly testable with no GPU, no window and no SDL: build a frame, seal
// it, read back exactly what your game described. The engine has always used
// that internally; nothing shipped it, and phase 8 makes MORE of the surface
// private, so the escape hatch has to be part of the public API rather than an
// accident of everything being public.
//
// Two things live here:
//
//   TestFrame    — the fixture. Owns (or borrows) a packet, hands you a
//                  FrameSink to give to your own frame code, seals, and reads
//                  back typed views.
//   FrameAccess  — the raw friend hatch: the engine's sim-stage tail and the
//                  packet's lifecycle, as free functions. TestFrame is built
//                  on it. Reach for it directly only when you are testing the
//                  machinery itself (the engine's own frame_packet.test.cpp
//                  does exactly that).
//
// Typical use in a game's test suite:
//
//     bracken::testing::TestFrame frame;
//     my_game.describe_frame(frame.sink());     // your code, unchanged
//     frame.seal();
//     EXPECT_EQ(frame.draws().size(), 3u);
//
// Header-only and GPU-free. Nothing in the engine's production path includes
// it (checked by core/module_layering.test.cpp: no header outside testing/
// depends on this directory).

#include "bracken/contracts/light.hpp"
#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/frame_sink.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace bracken::testing {

/**
 * FrameAccess — the friend hatch for FrameBuilder's sim-stage tail and
 * FramePacket's lifecycle.
 *
 * Everything here is private on the type it forwards to (see the friend list
 * in frame_builder.hpp): a host calling seal() mid-frame, or reset() on a
 * packet the renderer is consuming, corrupts a frame in release with no
 * diagnostic. Calling it from a TEST is a different matter — a test owns the
 * whole lifecycle — so the capability is published, named for what it is, and
 * kept out of the surface a game's frame code sees.
 */
struct FrameAccess {
    // ---- FrameBuilder's sim-stage tail ----
    static void seal(scene::FrameBuilder& fb) { fb.seal(); }
    static void set_frame_timing(scene::FrameBuilder& fb, float dt, double time, uint64_t frame_index) {
        fb.set_frame_timing(dt, time, frame_index);
    }
    [[nodiscard]] static scene::FrameBuilder::EffectSnapshotAllocation
    alloc_effect_snapshot(scene::FrameBuilder& fb, uint32_t emitter_count, uint32_t particle_count) {
        return fb.alloc_effect_snapshot(emitter_count, particle_count);
    }
    static void set_engine_frame(scene::FrameBuilder& fb, float scene_time, float sky_time,
                                 uint32_t effect_instance_count) {
        fb.set_engine_frame(scene_time, sky_time, effect_instance_count);
    }
    template<typename Fn> static void for_each_effect_spawn(const scene::FrameBuilder& fb, Fn&& fn) {
        fb.for_each_effect_spawn(std::forward<Fn>(fn));
    }
    [[nodiscard]] static bool has_staged_environment(const scene::FrameBuilder& fb) {
        return fb.has_staged_environment();
    }
    [[nodiscard]] static const bracken::render::EnvironmentState& staged_environment(const scene::FrameBuilder& fb) {
        return fb.staged_environment();
    }
    [[nodiscard]] static scene::FramePacket& packet(scene::FrameBuilder& fb) { return fb.packet(); }

    // ---- FramePacket's lifecycle + raw allocator ----
    static void begin_fill(scene::FramePacket& p) { p.begin_fill(); }
    static void begin_consume(scene::FramePacket& p) { p.begin_consume(); }
    static void reset(scene::FramePacket& p) { p.reset(); }
    [[nodiscard]] static void* reserve_block(scene::FramePacket& p, std::size_t size, std::size_t align) {
        return p.reserve_block(size, align);
    }
    [[nodiscard]] static bool take_palette_slot(scene::FramePacket& p) { return p.take_palette_slot(); }
    [[nodiscard]] static uint32_t palette_slots_taken(const scene::FramePacket& p) { return p.palette_slots_taken(); }
};

/**
 * TestFrame — one headless frame, start to finish.
 *
 * Construct it, hand `sink()` to the frame code under test, `seal()`, then
 * read back. The default constructor owns a packet; the borrowing constructor
 * binds to one you already have (a PacketRing slot, say). Either way the
 * builder is bound at construction, exactly as `Application` binds one per
 * frame, so the state machine sees the same transitions a real frame does.
 */
class TestFrame {
public:
    explicit TestFrame(uint32_t lane_count = 1,
                       std::size_t arena_bytes = scene::FramePacket::k_default_arena_bytes)
        : owned_(std::make_unique<scene::FramePacket>(arena_bytes)), packet_(owned_.get()) {
        builder_.emplace(*packet_, lane_count);
        sink_.emplace(*builder_);
    }

    /** Bind to a caller-owned packet (must be Retired). */
    explicit TestFrame(scene::FramePacket& packet, uint32_t lane_count = 1) : packet_(&packet) {
        builder_.emplace(*packet_, lane_count);
        sink_.emplace(*builder_);
    }

    TestFrame(const TestFrame&) = delete;
    TestFrame& operator=(const TestFrame&) = delete;

    /** What you hand to your own frame code — the host surface, nothing else. */
    [[nodiscard]] scene::FrameSink& sink() { return *sink_; }

    /** The builder underneath, for the multi-lane emission API. */
    [[nodiscard]] scene::FrameBuilder& builder() { return *builder_; }

    [[nodiscard]] scene::FramePacket& packet() { return *packet_; }
    [[nodiscard]] const scene::FramePacket& packet() const { return *packet_; }

    /** Stamp the timing the engine would have stamped at kick. */
    void set_frame_timing(float dt, double time, uint64_t frame_index) {
        FrameAccess::set_frame_timing(*builder_, dt, time, frame_index);
    }

    /** Filling → Sealed. Exactly once, after every emitting job is joined. */
    void seal() { FrameAccess::seal(*builder_); }
    [[nodiscard]] bool sealed() const { return builder_->sealed(); }

    // ---- Typed read-back (all require a sealed packet) ----

    [[nodiscard]] std::span<const scene::ModelDrawCommand> draws() const { return packet_->model_commands(); }
    [[nodiscard]] std::span<const scene::SkinnedDrawCommand> skinned_draws() const {
        return packet_->skinned_commands();
    }
    [[nodiscard]] std::span<const bracken::render::lighting::PointLight> point_lights() const {
        return packet_->point_lights();
    }
    [[nodiscard]] std::span<const bracken::render::lighting::SpotLight> spot_lights() const {
        return packet_->spot_lights();
    }
    [[nodiscard]] std::span<const scene::BillboardDrawCommand> billboards() const { return packet_->billboards(); }
    [[nodiscard]] std::span<const scene::DebugLineDrawCommand> debug_lines() const { return packet_->debug_lines(); }
    [[nodiscard]] std::span<const scene::EffectSpawnCommand> effect_spawns() const {
        return packet_->effect_spawns();
    }
    [[nodiscard]] std::span<const scene::UiCommand> ui_commands() const { return packet_->ui_commands(); }

    /** Every UI text/button string of the frame, in emission order. The views
     *  point into packet memory and are valid while this TestFrame lives. */
    [[nodiscard]] std::vector<std::string_view> ui_text() const {
        std::vector<std::string_view> out;
        for (const scene::UiCommand& cmd : packet_->ui_commands()) {
            if (cmd.is<scene::UiTextCommand>()) {
                out.push_back(cmd.get<scene::UiTextCommand>().text);
            } else if (cmd.is<scene::UiButtonCommand>()) {
                out.push_back(cmd.get<scene::UiButtonCommand>().label);
            }
        }
        return out;
    }

    /** What the fill refused: invalid handles, invalid palette ids, palettes
     *  over the frame budget. Readable before seal() too. */
    [[nodiscard]] scene::FramePacket::BuildDiagnostics build_diagnostics() const {
        return packet_->build_diagnostics();
    }

private:
    std::unique_ptr<scene::FramePacket> owned_; // null when borrowing
    scene::FramePacket* packet_;
    std::optional<scene::FrameBuilder> builder_;
    std::optional<scene::FrameSink> sink_;
};

} // namespace bracken::testing
