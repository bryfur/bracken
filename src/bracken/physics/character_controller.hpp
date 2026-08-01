#pragma once

#include "bracken/physics/physics_world.hpp"

#include <glm/glm.hpp>

namespace bracken::physics {

/** Tuning for a walking character. All distances are world units. */
struct CharacterConfig {
    float radius = 30.0f;        ///< capsule radius
    float stand_height = 150.0f; ///< total height, feet to crown
    float crouch_height = 80.0f;

    /** Slopes steeper than this are walls: you slide, you do not walk up. The
     *  single most important number here — too low and stairs feel sticky,
     *  too high and the character walks up cliffs. */
    float max_slope_degrees = 50.0f;

    /** Obstacles up to this tall are stepped over rather than blocking. Curbs,
     *  stairs and doorway sills all live here; without it a character stops
     *  dead on a 5-unit lip and players report it as "invisible wall". */
    float step_height = 40.0f;

    float gravity = -2000.0f;
    float jump_speed = 780.0f;

    /** Walking off a small lip should not launch the character into an arc.
     *  While grounded and moving down, the controller looks this far below the
     *  feet and sticks to what it finds. */
    float ground_snap = 45.0f;

    /** Air control as a fraction of ground control. 0 = no steering while
     *  airborne, 1 = full. */
    float air_control = 0.35f;

    /** How long after leaving ground a jump is still allowed. Players press
     *  jump a frame or two late constantly; without this the game feels
     *  unresponsive and nobody can say why. */
    float coyote_time = 0.12f;

    /** A jump pressed this long before landing still fires on touchdown. */
    float jump_buffer = 0.15f;
};

/** What the character is doing right now. */
struct CharacterState {
    glm::vec3 position{0.0f}; ///< FEET position, not centre
    glm::vec3 velocity{0.0f};
    glm::vec3 ground_normal{0.0f, 1.0f, 0.0f};
    bool grounded = false;
    bool crouching = false;
    /// Standing on something too steep to walk on: sliding, not walking.
    bool on_steep_slope = false;
    /// Horizontal movement was blocked this frame (a wall, not a slope).
    bool blocked = false;
    /// Set for the frame a step-up happened; useful for footstep audio.
    bool stepped_up = false;
};

/** Per-frame intent from the game. */
struct CharacterInput {
    /** Desired horizontal velocity in world units/sec. Magnitude is the speed;
     *  the controller does not know about walk vs run. */
    glm::vec3 desired_velocity{0.0f};
    bool jump = false;   ///< edge, not held
    bool crouch = false; ///< held
};

/**
 * A capsule character controller: slopes, step-up, ground snapping, crouch and
 * jump, resolved against the physics world.
 *
 * WHY IT IS IN THE ENGINE. This is the single most rewritten thing in games,
 * and every rewrite reproduces the same handful of bugs — catching on
 * doorframes, launching off ramps, sticking to walls in mid-air, jumps that
 * feel unresponsive because the player pressed a frame early. None of that is
 * game-specific.
 *
 * WHAT IT IS NOT. It is not a rigid body. The character is moved kinematically
 * and swept against the world, because a dynamic body makes a character that
 * is shoved by every crate and slides down every ramp — physically honest and
 * uniformly hated. Push the world FROM the character (the playground drives a
 * kinematic pusher capsule alongside this) rather than letting the world push
 * the character.
 *
 * COORDINATES. `position` is the FEET, which is what gameplay code and
 * animation actually want; the capsule centre is derived internally. Getting
 * this backwards is a persistent source of half-height-off bugs.
 *
 * Tier: owned / sim-stage. It calls only PhysicsWorld queries, which are
 * sim-stage safe; give each character its own controller.
 */
class CharacterController {
public:
    CharacterController() = default;
    explicit CharacterController(const CharacterConfig& config) : config_(config) {}

    void configure(const CharacterConfig& config) { config_ = config; }
    [[nodiscard]] const CharacterConfig& config() const noexcept { return config_; }

    /** Teleport. Clears velocity — a teleport that keeps momentum is almost
     *  never what the caller meant. */
    void set_position(const glm::vec3& feet) {
        state_.position = feet;
        state_.velocity = glm::vec3(0.0f);
    }

    /** Move the character without clearing velocity — for a host that clamps
     *  the position to world bounds after the fact, where set_position's
     *  momentum reset would kill movement every frame at the edge. */
    void set_position_preserving_velocity(const glm::vec3& feet) { state_.position = feet; }

    [[nodiscard]] const CharacterState& state() const noexcept { return state_; }

    /** Current capsule height, accounting for crouch. */
    [[nodiscard]] float height() const noexcept {
        return state_.crouching ? config_.crouch_height : config_.stand_height;
    }

    /**
     * Advance one step: gravity, horizontal move with wall sliding and
     * step-up, then ground resolution.
     *
     * @param self_body   the character's own physics body, if it has one (games
     *        usually pair a controller with a kinematic capsule so the
     *        character can shove dynamic props). Excluded from EVERYTHING —
     *        without it the ground probe hits the character's own capsule and
     *        it stands on itself, climbing a little each frame.
     * @param ground_body the terrain / ground collider. Excluded from
     *        DEPENETRATION only, and deliberately still visible to the ground
     *        ray. A capsule resting on a heightfield always overlaps it
     *        slightly; pushing out of that every frame walks the character
     *        into the sky, while the ray is exactly how the ground should be
     *        found. Walls are depenetration's job, floors are the ray's.
     */
    void update(const PhysicsWorld& world, const CharacterInput& input, float dt,
                BodyId self_body = INVALID_BODY_ID, BodyId ground_body = INVALID_BODY_ID);

private:
    [[nodiscard]] float capsule_half_cylinder() const noexcept {
        // Capsule total height = cylinder + two hemispheres. Clamped so a
        // crouch height below 2*radius degenerates to a sphere rather than a
        // negative cylinder.
        return glm::max(height() * 0.5f - config_.radius, 0.0f);
    }
    [[nodiscard]] glm::vec3 capsule_center(const glm::vec3& feet) const noexcept {
        return feet + glm::vec3(0.0f, height() * 0.5f, 0.0f);
    }

    /** Push the capsule out of anything it overlaps at `feet`. Returns true if
     *  it had to move, and accumulates the surface normals it pushed off. */
    bool depenetrate(const PhysicsWorld& world, glm::vec3& feet, BodyId ignore_body,
                     glm::vec3* out_wall_normal) const;

    CharacterConfig config_{};
    CharacterState state_{};
    float time_since_grounded_ = 1e9f;
    float jump_buffered_for_ = 0.0f;
};

} // namespace bracken::physics
