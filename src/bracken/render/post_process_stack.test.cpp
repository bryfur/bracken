#include "bracken/render/post_process_stack.hpp"
#include "bracken/graphics_settings.hpp"
#include "bracken/render/color_grading.hpp"
#include "bracken/render/environment.hpp"
#include <gtest/gtest.h>

using namespace bracken::render;
using bracken::gpu::CompositeUniforms;

// GraphicsSettings is the single source of truth for the default bloom
// strength: the normal frame path overwrites CompositeParams from it every
// frame, so the struct defaults are live only for hosts driving the composite
// pass directly — those hosts must see the same default. Pins the L30 fix
// (the defaults used to disagree: 0.35 vs the persisted 0.25).
TEST(CompositeDefaults, BloomStrengthMatchesGraphicsSettings) {
    const bracken::GraphicsSettings settings;
    const PostProcessStack::CompositeParams params;
    const CompositeUniforms uniforms;

    EXPECT_FLOAT_EQ(params.bloom_strength, settings.bloom_strength);
    EXPECT_FLOAT_EQ(uniforms.bloomStrength, settings.bloom_strength);
}

// Default CompositeParams must be a NEUTRAL grade with auto exposure off:
// the composite pass gates the white-balance/CDL/auto-exposure blocks on
// these exact values, so the defaults render pixel-identical to the
// pre-grading composite (the shader-side gating exists because LMS/pow
// round-trips are not bit-exact even at neutral parameters).
TEST(CompositeDefaults, GradingDefaultsAreNeutral) {
    const PostProcessStack::CompositeParams params;

    EXPECT_FALSE(params.auto_exposure);
    EXPECT_EQ(params.exposure_buffer, nullptr);
    EXPECT_EQ(params.white_balance, glm::vec3(1.0f));
    EXPECT_EQ(params.cdl_slope, glm::vec3(1.0f));
    EXPECT_EQ(params.cdl_offset, glm::vec3(0.0f));
    EXPECT_EQ(params.cdl_power, glm::vec3(1.0f));
}

// The default EnvironmentState grading fields must map onto that neutral
// CompositeParams state through the same functions the frame path uses.
TEST(CompositeDefaults, EnvironmentGradingDefaultsMapToNeutral) {
    const EnvironmentState env;
    EXPECT_EQ(white_balance_coefficients(env.grade_temperature, env.grade_tint), glm::vec3(1.0f));
    EXPECT_TRUE(cdl_is_neutral(env.grade_slope, env.grade_offset, env.grade_power));
}
