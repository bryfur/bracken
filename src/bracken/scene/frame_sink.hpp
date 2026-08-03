#pragma once

#include "bracken/scene/frame_builder.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <span>

namespace bracken::scene {

/**
 * FrameSink — everything a HOST may do to a frame, and nothing else.
 *
 * `SimFrame::frame()` hands out one of these. It is a thin, non-owning view of
 * the frame's FrameBuilder: every call forwards, there is no state, and it
 * costs nothing.
 *
 * Why it exists. FrameBuilder is two APIs welded together: the host's
 * description surface, and the engine's sim-stage tail (`seal`,
 * `set_frame_timing`, `alloc_effect_snapshot`, `set_engine_frame`,
 * `for_each_effect_spawn`, the staged-environment reads, `packet()`). Until
 * now the second one was public and the contract was PROSE — "Do NOT call
 * seal() or set_frame_timing()" in a doc comment, backed only by a
 * BRACKEN_ASSERT. A shipped game that sealed the packet early got a frame the
 * renderer was already consuming, in release, with no diagnostic. Splitting
 * the surfaces makes the mistake unrepresentable instead of undiagnosed: the
 * tail is private to FrameBuilder now, and a host holding a FrameSink cannot
 * name it.
 *
 * Thread-safety tier: `sim-stage`, exactly like the builder underneath.
 * Multi-lane emission is unchanged — `lane(i)` returns the same
 * `FrameBuilder::Lane` (tier `owned`), and the lane API is entirely
 * host-facing, so it is exposed as-is.
 */
class FrameSink {
public:
    using BonePaletteAllocation = FrameBuilder::BonePaletteAllocation;
    using Lane = FrameBuilder::Lane;
    using UiLane = FrameBuilder::UiLane;

    explicit FrameSink(FrameBuilder& builder) noexcept : builder_(&builder) {}

    // ---- Multi-lane emission ------------------------------------------------

    [[nodiscard]] uint32_t lane_count() const { return builder_->lane_count(); }
    [[nodiscard]] Lane& lane(uint32_t index) { return builder_->lane(index); }

    // ---- Single-lane conveniences (forward to lane 0) ------------------------

    void draw_model(bracken::ModelHandle handle, const glm::mat4& transform,
                    const glm::vec4& tint = {1.0f, 1.0f, 1.0f, 1.0f}, DrawFlags flags = DrawFlags::None,
                    uint32_t motion_id = 0) {
        builder_->draw_model(handle, transform, tint, flags, motion_id);
    }
    void draw_skinned(bracken::ModelHandle handle, const glm::mat4& transform, BonePaletteId palette,
                      const glm::vec4& tint = {1.0f, 1.0f, 1.0f, 1.0f}, DrawFlags flags = DrawFlags::None,
                      uint32_t motion_id = 0) {
        builder_->draw_skinned(handle, transform, palette, tint, flags, motion_id);
    }
    [[nodiscard]] BonePaletteAllocation alloc_bone_palette(uint32_t joint_count) {
        return builder_->alloc_bone_palette(joint_count);
    }
    void add_point_light(const bracken::render::lighting::PointLightDesc& light) { builder_->add_point_light(light); }
    void add_spot_light(const bracken::render::lighting::SpotLightDesc& light) { builder_->add_spot_light(light); }
    void add_billboard_3d(float world_x, float world_y, float world_z, float width, float fill_ratio,
                          bracken::core::Color fill_color, bracken::core::Color bg_color,
                          bracken::core::Color frame_color) {
        builder_->add_billboard_3d(world_x, world_y, world_z, width, fill_ratio, fill_color, bg_color, frame_color);
    }
    void add_debug_line(const glm::vec3& start, const glm::vec3& end, bracken::core::Color color) {
        builder_->add_debug_line(start, end, color);
    }
    void add_debug_sphere(const glm::vec3& center, float radius, bracken::core::Color color, int segments = 16) {
        builder_->add_debug_sphere(center, radius, color, segments);
    }
    void add_debug_box(const glm::vec3& min, const glm::vec3& max, bracken::core::Color color) {
        builder_->add_debug_box(min, max, color);
    }
    /** Analytic shadow occluders for the traced visibility passes — see
     *  FrameBuilder::Lane::add_shadow_capsules for the contract and the budget.
     *  Copied into packet memory; emit them fresh every frame. */
    void add_shadow_capsules(std::span<const bracken::render::rt::ShadowCapsule> capsules) {
        builder_->add_shadow_capsules(capsules);
    }
    void spawn_effect(const bracken::EffectDefinition* definition, const glm::vec3& position,
                      const glm::vec3& direction = {1, 0, 0}) {
        builder_->spawn_effect(definition, position, direction);
    }
    [[nodiscard]] UiLane ui() { return builder_->ui(); }

    // ---- Singular frame state (single writer; last write wins) ---------------

    void set_camera(const CameraState& camera) { builder_->set_camera(camera); }
    void set_environment(const bracken::render::EnvironmentState& environment) {
        builder_->set_environment(environment);
    }
    void set_draw_skybox(bool draw) { builder_->set_draw_skybox(draw); }
    void set_draw_ground(bool draw) { builder_->set_draw_ground(draw); }
    void set_draw_grass(bool draw) { builder_->set_draw_grass(draw); }

private:
    FrameBuilder* builder_;
};

} // namespace bracken::scene
