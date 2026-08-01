#include "bracken/physics/character_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace bracken::physics {
namespace {

/** Depenetration passes per move. Contacts interact — pushing out of a wall
 *  can push into the floor — so one pass is not enough in a corner. Four
 *  resolves every case a character actually stands in; more is wasted work. */
constexpr int k_depenetration_passes = 4;
constexpr std::size_t k_max_contacts = 8;

/** Remove the component of `v` heading into a surface, so the character slides
 *  along a wall instead of stopping dead against it. */
glm::vec3 slide_along(const glm::vec3& v, const glm::vec3& normal) {
    const float into = glm::dot(v, normal);
    return into < 0.0f ? v - normal * into : v;
}

} // namespace

bool CharacterController::depenetrate(const PhysicsWorld& world, glm::vec3& feet,
                                      BodyId ignore_body, glm::vec3* out_wall_normal) const {
    bool moved = false;
    const float cos_max_slope = std::cos(glm::radians(config_.max_slope_degrees));

    for (int pass = 0; pass < k_depenetration_passes; ++pass) {
        std::array<PhysicsWorld::ShapeContact, k_max_contacts> contacts{};
        const std::size_t n =
            world.collide_capsule(capsule_center(feet), capsule_half_cylinder(), config_.radius,
                                  contacts, /*include_moving=*/false, ignore_body);
        if (n == 0) {
            break;
        }

        for (std::size_t i = 0; i < n; ++i) {
            const auto& c = contacts[i];
            if (c.penetration <= 0.0f) {
                continue;
            }

            // HORIZONTAL ONLY. Vertical placement belongs to ground
            // resolution, which uses a downward ray and knows what a floor is.
            // Pushing out of a contact along its full normal looks right until
            // the character is inside a large enclosing volume — a whole
            // building's AABB, say — and then the shortest way out is UP, and
            // the character is levitated a frame at a time until it clears the
            // roof. That is not a hypothetical: it is what this did to the
            // playground's village, a single model with a 1337-unit AABB.
            glm::vec3 push = c.normal * c.penetration;
            push.y = 0.0f;
            if (glm::length(push) <= 1e-5f) {
                continue; // a pure floor/ceiling contact: nothing to do here
            }
            feet += push;
            moved = true;

            // A near-vertical surface is a wall; remember it so the caller can
            // slide velocity along it. Floors are handled by ground
            // resolution, not here.
            if (out_wall_normal != nullptr && c.normal.y < cos_max_slope) {
                *out_wall_normal += c.normal;
            }
        }
    }
    return moved;
}

void CharacterController::update(const PhysicsWorld& world, const CharacterInput& input, float dt,
                                 BodyId self_body, BodyId ground_body) {
    // Depenetration ignores the ground; the ground ray ignores only ourselves.
    const BodyId ignore_body = ground_body != INVALID_BODY_ID ? ground_body : self_body;
    if (dt <= 0.0f) {
        return;
    }

    state_.blocked = false;
    state_.stepped_up = false;

    // ---- crouch ----------------------------------------------------------
    // Standing up is refused while something is directly overhead, otherwise
    // the character grows into the ceiling and gets ejected.
    if (input.crouch) {
        state_.crouching = true;
    } else if (state_.crouching) {
        const float standing_half = glm::max(config_.stand_height * 0.5f - config_.radius, 0.0f);
        const glm::vec3 standing_center =
            state_.position + glm::vec3(0.0f, config_.stand_height * 0.5f, 0.0f);
        std::array<PhysicsWorld::ShapeContact, k_max_contacts> contacts{};
        const std::size_t n = world.collide_capsule(standing_center, standing_half, config_.radius,
                                                    contacts, false, ignore_body);
        if (n == 0) {
            state_.crouching = false;
        }
    }

    // ---- jump intent -----------------------------------------------------
    // Buffered and coyote-timed. Both exist because players are not frame
    // accurate, and a jump that silently does nothing reads as a broken game
    // rather than a mistimed press.
    jump_buffered_for_ = input.jump ? config_.jump_buffer
                                    : glm::max(jump_buffered_for_ - dt, 0.0f);
    time_since_grounded_ = state_.grounded ? 0.0f : time_since_grounded_ + dt;

    const bool may_jump = time_since_grounded_ <= config_.coyote_time && !state_.on_steep_slope;
    if (jump_buffered_for_ > 0.0f && may_jump) {
        state_.velocity.y = config_.jump_speed;
        state_.grounded = false;
        jump_buffered_for_ = 0.0f;
        time_since_grounded_ = config_.coyote_time + 1.0f; // consume it
    }

    // ---- horizontal velocity --------------------------------------------
    const glm::vec3 desired{input.desired_velocity.x, 0.0f, input.desired_velocity.z};
    if (state_.grounded) {
        state_.velocity.x = desired.x;
        state_.velocity.z = desired.z;
    } else {
        // Partial air control: steer, but do not turn on a dime mid-jump.
        const glm::vec3 current{state_.velocity.x, 0.0f, state_.velocity.z};
        const glm::vec3 blended = glm::mix(current, desired, glm::clamp(config_.air_control, 0.0f, 1.0f));
        state_.velocity.x = blended.x;
        state_.velocity.z = blended.z;
    }

    state_.velocity.y += config_.gravity * dt;
    // Terminal velocity: an unbounded fall eventually tunnels through the
    // world in one step, and no game needs a 400 m/s character.
    state_.velocity.y = glm::max(state_.velocity.y, -4000.0f);

    // ---- horizontal move, with wall slide and step-up --------------------
    const glm::vec3 before = state_.position;
    glm::vec3 horizontal_step{state_.velocity.x * dt, 0.0f, state_.velocity.z * dt};

    glm::vec3 candidate = state_.position + horizontal_step;
    glm::vec3 wall_normal{0.0f};
    const bool hit_something = depenetrate(world, candidate, ignore_body, &wall_normal);

    if (hit_something && glm::length(wall_normal) > 1e-4f) {
        wall_normal = glm::normalize(wall_normal);

        // Step-up: retry the same horizontal move from step_height higher. If
        // the raised position is clear, the obstacle was a step rather than a
        // wall, and we drop back down during ground resolution.
        glm::vec3 raised = state_.position + glm::vec3(0.0f, config_.step_height, 0.0f) +
                           horizontal_step;
        glm::vec3 raised_wall{0.0f};
        const bool raised_blocked = depenetrate(world, raised, ignore_body, &raised_wall);

        if (!raised_blocked && state_.grounded) {
            state_.position = raised;
            state_.stepped_up = true;
        } else {
            // A real wall: slide along it rather than stopping, and cancel the
            // velocity heading into it so we do not push every frame.
            state_.position = candidate;
            state_.velocity = slide_along(state_.velocity, wall_normal);
            state_.blocked = true;
        }
    } else {
        state_.position = candidate;
    }

    // ---- vertical move ---------------------------------------------------
    // Clamped to the probe's reach below the feet: moving further than the
    // ground ray can see in a single step is how a character tunnels through
    // the floor at terminal velocity.
    const float max_fall_step = config_.ground_snap + config_.step_height;
    const float vertical_step =
        glm::clamp(state_.velocity.y * dt, -max_fall_step, max_fall_step);
    state_.position.y += vertical_step;
    glm::vec3 ignored_normal{0.0f};
    depenetrate(world, state_.position, ignore_body, &ignored_normal);

    // ---- ground resolution ----------------------------------------------
    // A downward ray from inside the capsule. Length covers the snap distance
    // plus the step we may have just taken.
    // Cast from the capsule CENTRE, not from just above the feet. A fast fall
    // moves the character further in one step than a short probe is long, and
    // once the feet dip below the surface a ray starting there points away
    // from the ground and finds nothing — so the character falls forever
    // through solid terrain. Starting half a body up means the origin is still
    // above the surface in every case depenetration has not already fixed.
    // Origin just above the feet, NOT half a body up. cast_ray returns the
    // CLOSEST hit, so a high origin finds the interior face of any volume the
    // character happens to be standing inside (the playground's village is one
    // model with a 1337-unit AABB) and the real ground below is never seen.
    // Tunnelling is prevented by clamping the fall step above instead.
    const float above = config_.radius;
    const float probe = above + config_.ground_snap + config_.step_height;
    const glm::vec3 origin = state_.position + glm::vec3(0.0f, above, 0.0f);
    const PhysicsWorld::RayHit hit =
        world.cast_ray(origin, glm::vec3(0.0f, -probe, 0.0f), self_body);

    const float cos_max_slope = std::cos(glm::radians(config_.max_slope_degrees));
    const bool falling = state_.velocity.y <= 0.0f;

    if (hit.hit && falling) {
        const float ground_y = hit.position.y;
        const float gap = state_.position.y - ground_y; // >0: ground below the feet

        // The ray starts half a body ABOVE the feet so a fast fall cannot
        // outrun it, which means it can also hit something above the feet —
        // a ceiling, or an interior face when the character is inside a large
        // volume. Snapping to that would teleport the character UP, and doing
        // it every frame levitates them out through the roof. Ground is only
        // ground if it is at or just below the feet; `radius` of slack covers
        // having sunk slightly into a surface.
        const bool ground_is_below = gap >= -config_.radius;

        if (ground_is_below && gap <= config_.ground_snap) {
            state_.ground_normal = hit.normal;
            state_.on_steep_slope = hit.normal.y < cos_max_slope;

            if (!state_.on_steep_slope) {
                // Stick to the surface: this is what stops a character
                // launching off every small crest it walks over.
                state_.position.y = ground_y;
                state_.velocity.y = 0.0f;
                state_.grounded = true;
            } else {
                // Too steep to stand on: keep sliding, but do not sink in.
                state_.position.y = glm::max(state_.position.y, ground_y);
                state_.velocity = slide_along(state_.velocity, hit.normal);
                state_.grounded = false;
            }
        } else {
            state_.grounded = false;
            state_.on_steep_slope = false;
        }
    } else {
        state_.grounded = false;
        state_.on_steep_slope = false;
    }

    // Report what actually happened rather than what was requested: gameplay
    // (footsteps, animation speed) should react to real motion.
    const glm::vec3 moved = state_.position - before;
    if (dt > 0.0f && state_.grounded) {
        state_.velocity.x = moved.x / dt;
        state_.velocity.z = moved.z / dt;
    }
}

} // namespace bracken::physics
