#include <gtest/gtest.h>

#include "bracken/core/angle.hpp"

#include <glm/trigonometric.hpp>
#include <type_traits>

using bracken::core::Degrees;

// Degrees exists so a host says what it means and the engine does the
// conversion. The one property that matters is that it converts EXACTLY the
// way every call site used to do it by hand — a different constant here would
// shift every spot-light cone by a hair, which is precisely the kind of change
// no image gate would attribute to this file.

TEST(Degrees, RadiansMatchesGlmExactly) {
    for (float d : {0.0f, 1.0f, 18.0f, 32.0f, 45.0f, 90.0f, 180.0f, 360.0f, -12.5f}) {
        EXPECT_EQ(Degrees{d}.radians(), glm::radians(d)) << d;
    }
}

TEST(Degrees, IsAValueTypeAndConstexpr) {
    static_assert(Degrees{}.value == 0.0f);
    static_assert(Degrees{90.0f} == Degrees{90.0f});
    static_assert(Degrees{90.0f} != Degrees{45.0f});
    // Not implicitly constructible from a bare float: writing Degrees{18.0f}
    // at the call site is the entire point of the type.
    static_assert(!std::is_convertible_v<float, Degrees>);
    SUCCEED();
}
