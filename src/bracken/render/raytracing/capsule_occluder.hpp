#pragma once

#include "bracken/contracts/shadow_capsule.hpp"

#include <cmath>
#include <glm/glm.hpp>
#include <span>

namespace bracken::render::rt {

/**
 * Analytic ray-capsule occlusion — the CPU SOURCE OF TRUTH for the test the
 * any-hit traversal runs in `shaders/rt_trace.hlsli` (`capsuleOccluded` /
 * `capsulesOccluded`).
 *
 * The two are written to be read side by side, statement for statement. That
 * is the whole reason this file exists: the GPU copy is inside a compute
 * shader that only reports a boolean per pixel, so a sign error in it is
 * indistinguishable from a denoiser artefact, whereas here the graze, cap,
 * parallel and degenerate cases are unit-testable (capsule_occluder.test.cpp).
 * Edit one, edit the other.
 *
 * ANY-HIT SEMANTICS. Everything here answers "is anything in the way", not
 * "what is nearest": no distance, no normal, no surface. That is what a shadow
 * ray asks, and it is why an analytic proxy can stand in for a skinned mesh at
 * all (docs/rt_lighting_plan.md §5.8).
 */

/// Rays whose direction is this close to parallel with the capsule axis make
/// the cylinder quadratic's leading coefficient vanish. Rather than divide by
/// it, such rays are answered by the two cap spheres alone, which is exact:
/// a ray parallel to the axis can only enter through an end cap.
inline constexpr float k_capsule_parallel_epsilon = 1e-8f;

/// Does the ray hit the sphere `center`/`radius_sq` somewhere in [t_min,
/// t_max]? `dd` is dot(dir, dir), carried in so a non-unit direction (the
/// model-space rays a scaled instance produces elsewhere in this traversal)
/// stays correct instead of quietly rescaling t.
[[nodiscard]] inline bool sphere_occluded(const glm::vec3& origin, const glm::vec3& dir, float dd,
                                          const glm::vec3& center, float radius_sq, float t_min,
                                          float t_max) {
    const glm::vec3 oc = origin - center;
    const float b = glm::dot(dir, oc);
    const float c = glm::dot(oc, oc) - radius_sq;
    const float disc = b * b - dd * c;
    if (disc < 0.0f || dd <= 0.0f) {
        return false;
    }
    const float sq = std::sqrt(disc);
    // BOTH roots, not just the near one: an origin sitting between them is a
    // ray that starts inside the sphere, and the far root is where it leaves.
    // (The caller's inside-rejection below usually removes that case, but this
    // routine is also called for a cap sphere the origin is genuinely outside
    // of while being inside the OTHER one.)
    const float t0 = (-b - sq) / dd;
    const float t1 = (-b + sq) / dd;
    return (t0 >= t_min && t0 <= t_max) || (t1 >= t_min && t1 <= t_max);
}

/**
 * Does the ray hit `capsule` somewhere in [t_min, t_max]?
 *
 * THE METHOD, in three parts, because a capsule is three surfaces welded
 * together:
 *
 *  1. **Inside-rejection.** If the ray ORIGIN lies within the capsule, this
 *     returns false. That is not a numerical dodge, it is the contract: a
 *     capsule is a PROXY for a body, and the points inside it are that body's
 *     own surface. Occluding them would paint every character's own skin
 *     black — the torso proxy is necessarily fatter than the mesh in one axis
 *     and thinner in another (you cannot fit a cylinder to a box cross
 *     section), so a chest pixel genuinely does sit inside its own capsule.
 *     Those pixels are handled correctly anyway: a surface facing away from
 *     the light is already unlit by N·L before a ray is ever cast, and a
 *     surface facing the light is not occluded by the body behind it. Note
 *     this rejects only THIS capsule — an arm capsule still shadows the chest.
 *
 *  2. **The infinite cylinder**, as a quadratic in t, with both roots tested
 *     and each accepted only when its axial coordinate falls inside the
 *     segment. Testing both roots (rather than the near one alone, as the
 *     usual closest-hit formulation does) is what makes the answer right for
 *     a ray that starts inside the cylinder but beyond a cap — straight above
 *     a head, say.
 *
 *  3. **The two cap spheres.** They are always tested, not selected by which
 *     side the cylinder root fell on. That costs a little and buys every
 *     degenerate case for free: `p0 == p1` (the axis vanishes, both caps are
 *     the same sphere, and the answer is a pure sphere test), and a ray
 *     parallel to the axis (the quadratic degenerates, and a parallel ray can
 *     only enter through a cap anyway).
 *
 * A capsule with radius <= 0 occludes nothing, which is the useful reading of
 * a degenerate emit rather than an assert in a per-item hot loop.
 */
[[nodiscard]] inline bool capsule_occluded(const glm::vec3& origin, const glm::vec3& dir,
                                           float t_min, float t_max, const ShadowCapsule& capsule) {
    const float radius_sq = capsule.radius * capsule.radius;
    if (radius_sq <= 0.0f) {
        return false;
    }
    const glm::vec3 ba = capsule.p1 - capsule.p0;
    const glm::vec3 oa = origin - capsule.p0;
    const float baba = glm::dot(ba, ba);
    const float dd = glm::dot(dir, dir);

    // ---- 1. inside-rejection ------------------------------------------------
    // Distance from the origin to the SEGMENT, via the clamped axial parameter.
    float h = 0.0f;
    if (baba > 0.0f) {
        h = glm::dot(oa, ba) / baba;
        h = h < 0.0f ? 0.0f : (h > 1.0f ? 1.0f : h);
    }
    const glm::vec3 radial = oa - ba * h;
    if (glm::dot(radial, radial) <= radius_sq) {
        return false;
    }

    // ---- 2. infinite cylinder, clamped to the segment -----------------------
    const float bard = glm::dot(ba, dir);
    const float baoa = glm::dot(ba, oa);
    const float rdoa = glm::dot(dir, oa);
    const float oaoa = glm::dot(oa, oa);
    // a t^2 + 2 b t + c = 0, all three scaled by baba so the axial coordinate
    // below comes out in the same units and no division by |ba| is needed.
    const float a = baba * dd - bard * bard;
    const float b = baba * rdoa - baoa * bard;
    const float c = baba * oaoa - baoa * baoa - radius_sq * baba;
    if (a > k_capsule_parallel_epsilon) {
        const float disc = b * b - a * c;
        if (disc >= 0.0f) {
            const float sq = std::sqrt(disc);
            const float t0 = (-b - sq) / a;
            const float y0 = baoa + t0 * bard;
            if (y0 >= 0.0f && y0 <= baba && t0 >= t_min && t0 <= t_max) {
                return true;
            }
            const float t1 = (-b + sq) / a;
            const float y1 = baoa + t1 * bard;
            if (y1 >= 0.0f && y1 <= baba && t1 >= t_min && t1 <= t_max) {
                return true;
            }
        }
    }

    // ---- 3. the caps --------------------------------------------------------
    return sphere_occluded(origin, dir, dd, capsule.p0, radius_sq, t_min, t_max) ||
           sphere_occluded(origin, dir, dd, capsule.p1, radius_sq, t_min, t_max);
}

/**
 * The whole set, as one any-hit query. A LINEAR loop with an early-out on the
 * first hit, deliberately: the set is a dozen capsules per character and a
 * couple of hundred at the frame budget, which is far below where any
 * acceleration structure pays for the pointer chasing it costs.
 */
[[nodiscard]] inline bool capsules_occluded(std::span<const ShadowCapsule> capsules,
                                            const glm::vec3& origin, const glm::vec3& dir,
                                            float t_min, float t_max) {
    for (const ShadowCapsule& capsule : capsules) {
        if (capsule_occluded(origin, dir, t_min, t_max, capsule)) {
            return true;
        }
    }
    return false;
}

} // namespace bracken::render::rt
