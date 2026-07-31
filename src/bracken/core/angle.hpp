#pragma once

#include <glm/trigonometric.hpp>

namespace bracken::core {

/**
 * Degrees — an angle, with its unit in the type.
 *
 * Exists for one reason: engine APIs that want a cone half-angle used to take
 * a bare `float` holding the COSINE of one, leaving every host to write
 * `std::cos(glm::radians(18.0f))` at the call site and every reader to guess
 * whether a given float was degrees, radians or a cosine. A host should say
 * what it means; the conversion belongs to the engine.
 *
 * Deliberately not implicitly constructible from `float`: `Degrees{18.0f}` at
 * the call site is the entire point.
 */
struct Degrees {
    float value = 0.0f;

    [[nodiscard]] constexpr float radians() const noexcept { return glm::radians(value); }

    friend constexpr bool operator==(Degrees, Degrees) noexcept = default;
};

} // namespace bracken::core
