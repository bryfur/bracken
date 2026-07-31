#include "bracken/render/color_grading.hpp"
#include <gtest/gtest.h>

using namespace bracken::render;

// =============================================================================
// White balance coefficients (CPU side of the composite white-balance block)
// =============================================================================

TEST(WhiteBalance, NeutralIsExactlyOne) {
    // The composite shader gates the whole LMS round-trip on coeffs != 1, so
    // the neutral case must be EXACT (the numerical white-point math only
    // lands within ~4e-6 of 1.0) — this is what keeps default output
    // pixel-identical to the ungraded path.
    EXPECT_EQ(white_balance_coefficients(0.0f, 0.0f), glm::vec3(1.0f));
}

TEST(WhiteBalance, PositiveTemperatureWarms) {
    // Warm = L (long/red) gain up, S (short/blue) gain down.
    const glm::vec3 warm = white_balance_coefficients(0.5f, 0.0f);
    EXPECT_GT(warm.x, 1.0f);
    EXPECT_LT(warm.z, 1.0f);
    // Cooler = the opposite.
    const glm::vec3 cool = white_balance_coefficients(-0.5f, 0.0f);
    EXPECT_LT(cool.x, 1.0f);
    EXPECT_GT(cool.z, 1.0f);
}

TEST(WhiteBalance, TintShiftsGreenMagenta) {
    // Positive tint = magenta = M (green-sensitive) gain down.
    const glm::vec3 magenta = white_balance_coefficients(0.0f, 0.5f);
    EXPECT_LT(magenta.y, 1.0f);
    const glm::vec3 green = white_balance_coefficients(0.0f, -0.5f);
    EXPECT_GT(green.y, 1.0f);
}

TEST(WhiteBalance, StrengthGrowsWithTemperature) {
    const glm::vec3 mild = white_balance_coefficients(0.25f, 0.0f);
    const glm::vec3 strong = white_balance_coefficients(1.0f, 0.0f);
    EXPECT_GT(strong.x / strong.z, mild.x / mild.z);
}

// =============================================================================
// CDL neutrality check (gates the post-tonemap pow in the shader)
// =============================================================================

TEST(CdlNeutral, IdentityTripleIsNeutral) {
    EXPECT_TRUE(cdl_is_neutral(glm::vec3(1.0f), glm::vec3(0.0f), glm::vec3(1.0f)));
}

TEST(CdlNeutral, AnyDeviationIsNotNeutral) {
    EXPECT_FALSE(cdl_is_neutral(glm::vec3(1.1f, 1.0f, 1.0f), glm::vec3(0.0f), glm::vec3(1.0f)));
    EXPECT_FALSE(cdl_is_neutral(glm::vec3(1.0f), glm::vec3(0.0f, -0.02f, 0.0f), glm::vec3(1.0f)));
    EXPECT_FALSE(cdl_is_neutral(glm::vec3(1.0f), glm::vec3(0.0f), glm::vec3(1.0f, 1.0f, 0.9f)));
}
