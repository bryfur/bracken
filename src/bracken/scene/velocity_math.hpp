#pragma once

#include <glm/glm.hpp>

namespace bracken::scene {

/**
 * CPU reference for the velocity (motion-vector) convention written by the
 * depth-prepass shaders (prepass_velocity.frag.hlsl). Keep the two in sync —
 * TAA (and any other consumer) resolves history with:
 *
 *     prev_uv = uv - velocity
 *
 * Convention:
 * - Velocity is a UV-space delta, **current minus previous**, stored RG16F.
 * - UV origin is top-left (render-target convention): u = ndc.x*0.5+0.5,
 *   v = 0.5-ndc.y*0.5, so the NDC delta maps through (0.5, -0.5).
 * - Both clip positions are computed with UNJITTERED view-projection
 *   matrices; TAA jitter must never leak into velocity.
 * - Statics: prev_model == curr_model (velocity = camera motion only).
 * - Points behind either clip plane (w <= 0) return zero velocity, matching
 *   the shader guard.
 */
[[nodiscard]] inline glm::vec2 compute_uv_velocity(const glm::mat4& curr_view_proj_unjittered,
                                                   const glm::mat4& prev_view_proj_unjittered,
                                                   const glm::mat4& curr_model, const glm::mat4& prev_model,
                                                   const glm::vec3& local_pos) {
    const glm::vec4 curr_clip = curr_view_proj_unjittered * (curr_model * glm::vec4(local_pos, 1.0f));
    const glm::vec4 prev_clip = prev_view_proj_unjittered * (prev_model * glm::vec4(local_pos, 1.0f));
    if (curr_clip.w <= 0.0f || prev_clip.w <= 0.0f) {
        return glm::vec2(0.0f);
    }
    const glm::vec2 curr_ndc = glm::vec2(curr_clip) / curr_clip.w;
    const glm::vec2 prev_ndc = glm::vec2(prev_clip) / prev_clip.w;
    return (curr_ndc - prev_ndc) * glm::vec2(0.5f, -0.5f);
}

} // namespace bracken::scene
