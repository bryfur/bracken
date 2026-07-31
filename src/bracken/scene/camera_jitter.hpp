#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace bracken::scene {

/**
 * TAA camera-jitter infrastructure (Phase: motion-vector foundation).
 *
 * The jitter is a sub-pixel translation applied to the projection matrix
 * AFTER the frame captures its unjittered copies: velocity/motion-vector math
 * and frustum culling always use the unjittered matrices; only rasterization
 * (SV_Position) sees the jittered projection. Driven per frame by
 * SceneRenderer when TAA jitter is enabled (off by default until the TAA
 * resolve lands).
 */

/// Halton radical-inverse for `base` at 1-based `index`. halton(i, 2) and
/// halton(i, 3) form the classic low-discrepancy (2,3) sequence in [0, 1).
[[nodiscard]] inline float halton(uint32_t index, uint32_t base) {
    float f = 1.0f;
    float r = 0.0f;
    while (index > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(index % base);
        index /= base;
    }
    return r;
}

/// Length of the repeating jitter sequence (frames).
inline constexpr uint32_t JITTER_SEQUENCE_LENGTH = 8;

/// Halton(2,3) sub-pixel jitter for a frame, in pixel units, each component
/// in [-0.5, 0.5). The sequence repeats every JITTER_SEQUENCE_LENGTH frames.
[[nodiscard]] inline glm::vec2 halton_jitter(uint64_t frame_index) {
    const uint32_t i = static_cast<uint32_t>(frame_index % JITTER_SEQUENCE_LENGTH) + 1; // Halton is 1-based
    return {halton(i, 2) - 0.5f, halton(i, 3) - 0.5f};
}

/// Apply a sub-pixel screen-space translation (in pixels) to a perspective
/// projection matrix. Works for any projection whose clip w is produced from
/// view z (proj[2][3] = ±1), which covers the engine's reversed-Z RH_ZO scene
/// projections: NDC shifts by exactly 2*jitter/viewport independent of depth.
[[nodiscard]] inline glm::mat4 jitter_projection(const glm::mat4& proj, glm::vec2 jitter_pixels, float viewport_width,
                                                 float viewport_height) {
    if (viewport_width <= 0.0f || viewport_height <= 0.0f) {
        return proj;
    }
    const glm::vec2 jitter_ndc(2.0f * jitter_pixels.x / viewport_width, 2.0f * jitter_pixels.y / viewport_height);
    glm::mat4 jittered = proj;
    // clip.xy += jitter_ndc * clip.w; clip.w = proj[2][3] * view.z for a
    // perspective projection, so fold the offset into the z column.
    jittered[2][0] += jitter_ndc.x * proj[2][3];
    jittered[2][1] += jitter_ndc.y * proj[2][3];
    return jittered;
}

} // namespace bracken::scene
