#pragma once

#include <glm/glm.hpp>

namespace bracken::render {

/**
 * White-balance temperature/tint -> linear-RGB gains applied in LMS space
 * (CPU side of the composite pass's white-balance block; the shader owns the
 * fixed linear-RGB <-> LMS matrices, this function owns the white-point math).
 *
 * Standard CIE-based formulation (as popularized by Unity's post stack):
 * shift the D65 white point along the Planckian locus by `temperature` and
 * perpendicular to it by `tint`, then return the per-channel LMS gains that
 * map the shifted white back to D65. Both inputs are in [-1, 1]:
 *   temperature > 0 -> warmer (R up, B down), < 0 -> cooler
 *   tint        > 0 -> magenta, < 0 -> green
 *
 * temperature == 0 && tint == 0 returns EXACTLY (1,1,1) — the numerical
 * round-trip is only approximately neutral, and the composite shader gates
 * the whole white-balance block on coefficients != 1 so the ungraded path
 * stays pixel-identical (pinned by color_grading tests).
 */
[[nodiscard]] inline glm::vec3 white_balance_coefficients(float temperature, float tint) {
    if (temperature == 0.0f && tint == 0.0f) {
        return glm::vec3(1.0f);
    }

    // CIE xy chromaticity of the shifted illuminant. The asymmetric
    // temperature scale matches the reference formulation (the locus is
    // steeper toward the blues).
    const float x = 0.31271f - temperature * (temperature < 0.0f ? 0.1f : 0.05f);
    const float standard_illuminant_y = 2.87f * x - 3.0f * x * x - 0.27509507f;
    const float y = standard_illuminant_y + tint * 0.05f;

    // CIE xy -> XYZ (Y = 1) -> LMS (CIECAM02 / CAT02 transform).
    const float big_x = x / y;
    const float big_z = (1.0f - x - y) / y;
    const glm::vec3 w2{0.7328f * big_x + 0.4296f - 0.1624f * big_z,
                       -0.7036f * big_x + 1.6975f + 0.0061f * big_z,
                       0.0030f * big_x + 0.0136f + 0.9834f * big_z};

    // D65 white in LMS; gains map the shifted white back onto it.
    const glm::vec3 w1{0.949237f, 1.03542f, 1.08728f};
    return w1 / w2;
}

/** True when a CDL triple is the identity grade (slope 1, offset 0, power 1).
 *  The composite shader skips the CDL block entirely then — pow() is not
 *  bit-exact even at power 1. */
[[nodiscard]] inline bool cdl_is_neutral(const glm::vec3& slope, const glm::vec3& offset, const glm::vec3& power) {
    return slope == glm::vec3(1.0f) && offset == glm::vec3(0.0f) && power == glm::vec3(1.0f);
}

} // namespace bracken::render
