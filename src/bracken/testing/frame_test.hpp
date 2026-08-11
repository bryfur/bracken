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
#include "bracken/contracts/shadow_capsule.hpp"
#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/frame_sink.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
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
    /** The frame's analytic shadow occluders, in (lane, emission) order — see
     *  scene::FrameSink::add_shadow_capsules. */
    [[nodiscard]] std::span<const bracken::render::rt::ShadowCapsule> shadow_capsules() const {
        return packet_->shadow_capsules();
    }

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

    // ---- The equality instrument ----

    /**
     * FNV-1a over everything the HOST described this frame, in packet order.
     *
     * WHY IT IS ORDER-SENSITIVE, and why that is the point. Packet order is
     * what the renderer consumes — `seal()` merges lanes in (lane-index,
     * emission-order) and the merged frame is byte-identical at any worker
     * count (`scene/frame_builder.hpp`). So two frames holding the same three
     * draws in a different order are two DIFFERENT frames, and a digest that
     * called them equal would be useless as a round-trip proof: it would pass
     * for a level loader that shuffled its instance table.
     *
     * WHAT IS FOLDED, exactly: model draws, skinned draws (palette id and the
     * palette's own matrices), point and spot lights, billboards, debug lines,
     * effect spawns, UI commands (including the text bytes), shadow capsules,
     * the camera, the environment, the three world-element toggles and the
     * build diagnostics.
     *
     * WHAT IS DELIBERATELY NOT FOLDED: the engine-stamped timing and clock
     * section — `dt`, `time`, `frame_index`, `scene_time`, `sky_time` — and
     * the effect-instance snapshot derived from them. Those are functions of
     * wall clock, not of what the host asked for; folding them would make the
     * digest differ between two runs of the same scene, which is the one thing
     * it must not do.
     *
     * Two POINTER fields ride the packet by documented exception
     * (`EffectSpawnCommand::definition`, host-owned). A pointer VALUE is not
     * stable across runs, so only its null-ness is folded — a digest that
     * hashed the address would be nondeterministic by construction.
     *
     * Stable across runs and across `BRACKEN_WORKERS` counts, because every
     * input to it is. Requires a sealed packet (each accessor asserts).
     *
     * BLIND SPOT, stated rather than discovered: this folds the streams that
     * exist today. A feature that adds a new command stream to FramePacket and
     * does not extend this function leaves the digest silently insensitive to
     * it. Extend it in the same change that adds the stream.
     */
    [[nodiscard]] uint64_t digest() const {
        Fnv1a h;
        const scene::FramePacket& p = *packet_;

        h.count(p.model_commands().size());
        for (const scene::ModelDrawCommand& c : p.model_commands()) {
            h.u32(c.model_handle);
            h.u32(c.motion_id);
            h.mat4(c.transform);
            h.vec4(c.tint);
            h.u32(static_cast<uint32_t>(c.flags));
        }

        h.count(p.skinned_commands().size());
        for (const scene::SkinnedDrawCommand& c : p.skinned_commands()) {
            h.u32(c.model_handle);
            h.u32(c.motion_id);
            h.mat4(c.transform);
            h.vec4(c.tint);
            h.u32(c.palette.value);
            h.u32(static_cast<uint32_t>(c.flags));
        }

        h.count(p.palettes().size());
        for (const scene::BonePalette& bp : p.palettes()) {
            h.u32(bp.joint_count);
            h.u32(bp.matrix_offset);
            for (uint32_t i = 0; i < bp.joint_count; ++i) {
                h.mat4(bp.matrices[i]);
            }
        }

        h.count(p.point_lights().size());
        for (const bracken::render::lighting::PointLight& l : p.point_lights()) {
            h.vec3(l.position);
            h.f32(l.radius);
            h.vec3(l.color);
            h.f32(l.intensity);
            h.u32(l.casts_shadows ? 1u : 0u);
            h.u32(static_cast<uint32_t>(l.shadow_slot));
            h.u32(l.stable_id);
            h.f32(l.source_radius);
        }

        h.count(p.spot_lights().size());
        for (const bracken::render::lighting::SpotLight& l : p.spot_lights()) {
            h.vec3(l.position);
            h.f32(l.radius);
            h.vec3(l.direction);
            h.f32(l.inner_cos);
            h.vec3(l.color);
            h.f32(l.outer_cos);
            h.f32(l.intensity);
            h.u32(l.casts_shadows ? 1u : 0u);
            h.u32(static_cast<uint32_t>(l.shadow_slot));
            h.u32(l.stable_id);
            h.f32(l.source_radius);
        }

        h.count(p.billboards().size());
        for (const scene::BillboardDrawCommand& b : p.billboards()) {
            h.f32(b.world_x);
            h.f32(b.world_y);
            h.f32(b.world_z);
            h.f32(b.width);
            h.f32(b.fill_ratio);
            h.u32(b.fill_color.argb);
            h.u32(b.bg_color.argb);
            h.u32(b.frame_color.argb);
        }

        h.count(p.debug_lines().size());
        for (const scene::DebugLineDrawCommand& l : p.debug_lines()) {
            h.vec3(l.start);
            h.vec3(l.end);
            h.u32(l.color.argb);
        }

        h.count(p.effect_spawns().size());
        for (const scene::EffectSpawnCommand& e : p.effect_spawns()) {
            h.u32(e.definition != nullptr ? 1u : 0u);
            h.vec3(e.position);
            h.vec3(e.direction);
        }

        h.count(p.shadow_capsules().size());
        for (const bracken::render::rt::ShadowCapsule& c : p.shadow_capsules()) {
            h.vec3(c.p0);
            h.f32(c.radius);
            h.vec3(c.p1);
        }

        h.count(p.ui_commands().size());
        for (const scene::UiCommand& cmd : p.ui_commands()) {
            h.u32(static_cast<uint32_t>(cmd.data.index()));
            digest_ui(h, cmd);
        }

        h.u32(p.has_camera() ? 1u : 0u);
        if (p.has_camera()) {
            const scene::CameraState& cam = p.camera();
            h.mat4(cam.view);
            h.mat4(cam.projection);
            h.mat4(cam.view_projection);
            h.vec3(cam.position);
        }

        h.u32(p.has_environment() ? 1u : 0u);
        if (p.has_environment()) {
            // EnvironmentState is float-only (every member is a float or a
            // glm::vec3, all 4-byte aligned), so it has no padding and hashing
            // it as a run of floats is exact. If a non-float member is ever
            // added, hash the fields individually instead — padding bytes are
            // unspecified after a copy and would make this nondeterministic.
            const bracken::render::EnvironmentState& env = p.environment();
            static_assert(sizeof(bracken::render::EnvironmentState) % sizeof(float) == 0,
                          "EnvironmentState is expected to be a run of floats");
            const auto* words = reinterpret_cast<const unsigned char*>(&env);
            for (std::size_t i = 0; i < sizeof(env); ++i) {
                h.byte(words[i]);
            }
        }

        h.u32(p.should_draw_skybox() ? 1u : 0u);
        h.u32(p.should_draw_ground() ? 1u : 0u);
        h.u32(p.should_draw_grass() ? 1u : 0u);

        const scene::FramePacket::BuildDiagnostics diag = p.build_diagnostics();
        h.u32(diag.draws_dropped_invalid_handle);
        h.u32(diag.skinned_dropped_invalid_palette);
        h.u32(diag.palettes_over_budget);
        h.u32(diag.shadow_capsules_over_budget);

        return h.value;
    }

private:
    /** FNV-1a, byte at a time. Everything is fed through the typed helpers so
     *  no struct padding is ever hashed. */
    struct Fnv1a {
        uint64_t value = 0xcbf29ce484222325ull;

        void byte(unsigned char b) {
            value ^= static_cast<uint64_t>(b);
            value *= 0x100000001b3ull;
        }
        void u32(uint32_t v) {
            for (int i = 0; i < 4; ++i) {
                byte(static_cast<unsigned char>((v >> (i * 8)) & 0xFFu));
            }
        }
        void u64(uint64_t v) {
            for (int i = 0; i < 8; ++i) {
                byte(static_cast<unsigned char>((v >> (i * 8)) & 0xFFull));
            }
        }
        /** Bit pattern, not value: -0.0f and 0.0f are different frames, and a
         *  NaN tint is a bug the digest should surface rather than smooth. */
        void f32(float v) {
            uint32_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            u32(bits);
        }
        void count(std::size_t n) { u64(static_cast<uint64_t>(n)); }
        void vec3(const glm::vec3& v) {
            f32(v.x);
            f32(v.y);
            f32(v.z);
        }
        void vec4(const glm::vec4& v) {
            f32(v.x);
            f32(v.y);
            f32(v.z);
            f32(v.w);
        }
        void mat4(const glm::mat4& m) {
            for (int c = 0; c < 4; ++c) {
                vec4(m[c]);
            }
        }
        void text(std::string_view s) {
            count(s.size());
            for (char ch : s) {
                byte(static_cast<unsigned char>(ch));
            }
        }
    };

    static void digest_ui(Fnv1a& h, const scene::UiCommand& cmd) {
        if (cmd.is<scene::UiFilledRectCommand>()) {
            const auto& c = cmd.get<scene::UiFilledRectCommand>();
            h.f32(c.x); h.f32(c.y); h.f32(c.w); h.f32(c.h); h.u32(c.color.argb);
        } else if (cmd.is<scene::UiRectOutlineCommand>()) {
            const auto& c = cmd.get<scene::UiRectOutlineCommand>();
            h.f32(c.x); h.f32(c.y); h.f32(c.w); h.f32(c.h); h.u32(c.color.argb); h.f32(c.line_width);
        } else if (cmd.is<scene::UiCircleCommand>()) {
            const auto& c = cmd.get<scene::UiCircleCommand>();
            h.f32(c.x); h.f32(c.y); h.f32(c.radius); h.u32(c.color.argb);
            h.u32(static_cast<uint32_t>(c.segments));
        } else if (cmd.is<scene::UiCircleOutlineCommand>()) {
            const auto& c = cmd.get<scene::UiCircleOutlineCommand>();
            h.f32(c.x); h.f32(c.y); h.f32(c.radius); h.u32(c.color.argb); h.f32(c.line_width);
            h.u32(static_cast<uint32_t>(c.segments));
        } else if (cmd.is<scene::UiLineCommand>()) {
            const auto& c = cmd.get<scene::UiLineCommand>();
            h.f32(c.x1); h.f32(c.y1); h.f32(c.x2); h.f32(c.y2); h.u32(c.color.argb); h.f32(c.line_width);
        } else if (cmd.is<scene::UiTextCommand>()) {
            const auto& c = cmd.get<scene::UiTextCommand>();
            h.text(c.text);
            h.f32(c.x); h.f32(c.y); h.f32(c.scale); h.u32(c.color.argb);
        } else if (cmd.is<scene::UiButtonCommand>()) {
            const auto& c = cmd.get<scene::UiButtonCommand>();
            h.f32(c.x); h.f32(c.y); h.f32(c.w); h.f32(c.h);
            h.text(c.label);
            h.u32(c.color.argb);
            h.u32(c.selected ? 1u : 0u);
        }
    }

    std::unique_ptr<scene::FramePacket> owned_; // null when borrowing
    scene::FramePacket* packet_;
    std::optional<scene::FrameBuilder> builder_;
    std::optional<scene::FrameSink> sink_;
};

} // namespace bracken::testing
