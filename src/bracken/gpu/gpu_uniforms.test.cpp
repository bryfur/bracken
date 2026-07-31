// Cross-module DEFAULT-VALUE agreement for the GPU uniform structs.
//
// Layout (sizeof / alignof / every field offset / the 16-byte-register
// straddle rule) is NO LONGER TESTED HERE: it is pinned by the
// BRACKEN_HLSL_CBUFFER / BRACKEN_HLSL_FIELD static_assert blocks that follow
// each struct in gpu_uniforms.hpp, so a layout edit is a compile error in the
// same translation unit as the edit rather than a red test in a build nobody
// ran yet. The ~225 offsetof EXPECT_EQs that used to live in this file were
// deleted in exchange for those blocks; agreement with the *shaders* is
// checked by gpu/uniform_layout.test.cpp against the compiled SPIR-V.
//
// What remains is the part a static_assert cannot express: NSDMI defaults that
// must track a value owned by ANOTHER module (EnvironmentState,
// GraphicsSettings, render/auto_exposure.hpp constants), and the neutral-
// default invariants the shaders rely on to stay pixel-identical.

#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/render/environment.hpp"
#include <glm/glm.hpp>
#include <gtest/gtest.h>

using namespace bracken::gpu;

// =============================================================================
// SkyboxFragmentUniforms — cloud/haze defaults mirror EnvironmentState
// =============================================================================

TEST(UniformDefaults, SkyboxCloudDefaultsMatchEnvironmentState) {
    // The uniform defaults exist only as a safety net; the real source of
    // truth is EnvironmentState — keep them in lockstep.
    const SkyboxFragmentUniforms u{};
    const bracken::render::EnvironmentState env{};
    EXPECT_EQ(u.cloudCoverage, env.cloud_coverage);
    EXPECT_EQ(u.cloudDensity, env.cloud_density);
    EXPECT_EQ(u.cloudScale, env.cloud_scale);
    EXPECT_EQ(u.hazeStrength, env.haze_strength);
}

// The IBL sky-capture pass renders the same atmosphere as the skybox, so its
// cloud/haze snapshot defaults must start from the same values.
TEST(UniformDefaults, SkyCaptureCloudDefaultsMatchSkybox) {
    const SkyCaptureUniforms capture{};
    const SkyboxFragmentUniforms skybox{};
    EXPECT_EQ(capture.cloudCoverage, skybox.cloudCoverage);
    EXPECT_EQ(capture.cloudDensity, skybox.cloudDensity);
    EXPECT_EQ(capture.cloudScale, skybox.cloudScale);
    EXPECT_EQ(capture.hazeStrength, skybox.hazeStrength);
}

// =============================================================================
// CompositeUniforms — bloom strength tracks GraphicsSettings; grading neutral
// =============================================================================

TEST(UniformDefaults, CompositeDefaultsAreNeutral) {
    CompositeUniforms u;
    // Must track GraphicsSettings::bloom_strength (agreement is pinned in
    // post_process_stack.test.cpp).
    EXPECT_FLOAT_EQ(u.bloomStrength, 0.25f);
    EXPECT_FLOAT_EQ(u.volumetricFogEnabled, 0.0f);
    EXPECT_FLOAT_EQ(u.exposure, 1.0f);
    EXPECT_EQ(u.tonemapMode, 0);
    EXPECT_FLOAT_EQ(u.contrast, 1.0f);
    EXPECT_FLOAT_EQ(u.saturation, 1.0f);
    // depthA/depthB of 0 degrade to plain bilinear weights in the shader.
    EXPECT_FLOAT_EQ(u.depthA, 0.0f);
    EXPECT_FLOAT_EQ(u.depthB, 0.0f);
    // Exposure/grading defaults must be NEUTRAL (flags off, identity grade):
    // the ungraded path is pixel-identical to pre-P6 output.
    EXPECT_FLOAT_EQ(u.autoExposureEnabled, 0.0f);
    EXPECT_FLOAT_EQ(u.whiteBalanceEnabled, 0.0f);
    EXPECT_EQ(u.whiteBalance, glm::vec3(1.0f));
    EXPECT_FLOAT_EQ(u.cdlEnabled, 0.0f);
    EXPECT_EQ(u.cdlSlope, glm::vec3(1.0f));
    EXPECT_EQ(u.cdlOffset, glm::vec3(0.0f));
    EXPECT_EQ(u.cdlPower, glm::vec3(1.0f));
}

// =============================================================================
// Auto exposure — defaults mirror render/auto_exposure.hpp + GraphicsSettings
// =============================================================================

TEST(UniformDefaults, ExposureAdaptTracksSettingsAndConstants) {
    const ExposureAdaptUniforms u;
    // Speeds/clamps mirror the GraphicsSettings defaults (the frame path
    // overwrites them from settings each frame); percentile/range defaults
    // mirror render/auto_exposure.hpp.
    EXPECT_FLOAT_EQ(u.minLog2Luma, -10.0f);
    EXPECT_FLOAT_EQ(u.log2Range, 26.0f);
    EXPECT_FLOAT_EQ(u.speedUp, 1.5f);
    EXPECT_FLOAT_EQ(u.speedDown, 2.5f);
    EXPECT_FLOAT_EQ(u.minEv, -4.0f);
    EXPECT_FLOAT_EQ(u.maxEv, 4.0f);
    EXPECT_FLOAT_EQ(u.lowPercentile, 0.60f);
    EXPECT_FLOAT_EQ(u.highPercentile, 0.05f);
}

// The histogram pass builds the bins the adapt pass reads: both must agree on
// the log2-luma range or every adapted EV is silently wrong.
TEST(UniformDefaults, ExposureHistogramAndAdaptShareTheLumaRange) {
    const ExposureHistogramUniforms histogram;
    const ExposureAdaptUniforms adapt;
    EXPECT_FLOAT_EQ(histogram.minLog2Luma, adapt.minLog2Luma);
    EXPECT_FLOAT_EQ(histogram.invLog2Range, 1.0f / adapt.log2Range);
}

// =============================================================================
// Volumetric fog — defaults mirror the EnvironmentState volumetric defaults
// =============================================================================

TEST(UniformDefaults, VolumetricFogDefaults) {
    VolumetricFogUniforms u;
    EXPECT_FLOAT_EQ(u.fogDensity, 0.0015f);
    EXPECT_FLOAT_EQ(u.scatterStrength, 0.25f);
    EXPECT_FLOAT_EQ(u.fogHeight, 100.0f);
    EXPECT_FLOAT_EQ(u.fogFalloff, 0.005f);
    EXPECT_FLOAT_EQ(u.startDist, 100.0f);
    EXPECT_FLOAT_EQ(u.maxDist, 3000.0f);
    EXPECT_FLOAT_EQ(u.godRaysEnabled, 1.0f);
}

// =============================================================================
// GTAO — tuning constants, and the far cutoff shared with the blur pass
// =============================================================================

TEST(UniformDefaults, GtaoDefaults) {
    GTAOUniforms u;
    EXPECT_FLOAT_EQ(u.radius, 4.0f);
    EXPECT_FLOAT_EQ(u.radiusScale, 1.0f);
    // Falloff spans a sane fraction of the radius; thin-occluder decay is a
    // blend factor in [0, 1].
    EXPECT_GT(u.falloffRange, 0.0f);
    EXPECT_LE(u.falloffRange, 1.0f);
    EXPECT_GE(u.thinOccluderCompensation, 0.0f);
    EXPECT_LE(u.thinOccluderCompensation, 1.0f);
    EXPECT_FLOAT_EQ(u.maxDistance, 12000.0f);
    // The blur pass must agree with the AO pass on the far cutoff.
    EXPECT_FLOAT_EQ(u.maxDistance, BlurUniforms{}.maxDistance);
}

TEST(UniformDefaults, GtaoTemporalDefaults) {
    GtaoTemporalUniforms u;
    // History invalid until the CPU says otherwise (first frame outputs the
    // current AO unblended). Tuning constants are NSDMI live values.
    EXPECT_FLOAT_EQ(u.historyValid, 0.0f);
    EXPECT_FLOAT_EQ(u.feedbackMax, 0.9f);
    EXPECT_GT(u.depthRejectScale, 0.0f);
}

// =============================================================================
// TAA / bloom / particles — neutral-by-default invariants
// =============================================================================

TEST(UniformDefaults, TaaResolveDefaults) {
    TaaResolveUniforms u;
    // History invalid until the CPU says otherwise (first frame outputs
    // the current color unblended).
    EXPECT_FLOAT_EQ(u.historyValid, 0.0f);
    // Engine tuning constants — the NSDMI defaults are the live values.
    EXPECT_FLOAT_EQ(u.feedbackMax, 0.9f);
    EXPECT_FLOAT_EQ(u.varianceGamma, 1.0f);
}

TEST(UniformDefaults, BloomDownsampleDefaults) {
    BloomDownsampleUniforms u;
    // exposure defaults to neutral: post-exposure thresholding is then
    // byte-identical to the old pre-exposure behavior.
    EXPECT_FLOAT_EQ(u.exposure, 1.0f);
    EXPECT_FLOAT_EQ(u.isFirstPass, 1.0f);
}

TEST(UniformDefaults, ParticleFragmentDefaults) {
    ParticleFragmentUniforms u;
    EXPECT_FLOAT_EQ(u.emissiveBoost, 1.0f);
}

// =============================================================================
// Prepass — the "no previous palette" sentinel
// =============================================================================

TEST(UniformDefaults, PrepassPrevPaletteIsSentinel) {
    PrepassModelUniforms u;
    EXPECT_EQ(u.prevPaletteOffset, PrepassModelUniforms::NO_PREV_PALETTE);
}
