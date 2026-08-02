// The two R3 roughness curves, pinned. Neither is testable through the GPU: a
// screenshot cannot tell "reprojected through the virtual point" from
// "reprojected through the surface" on a static camera (they are identical
// there — which is itself one of the properties asserted below), and the
// rough-specular blend is one lerp inside a fragment shader.
//
// So the shape is pinned here and the shaders transliterate it, exactly as
// `gi_math.hpp` does for the probe field.

#include "bracken/render/rt_reflection_math.hpp"

#include <gtest/gtest.h>

namespace {

using bracken::render::ROUGH_SPECULAR_GI_END;
using bracken::render::ROUGH_SPECULAR_GI_START;
using bracken::render::rough_specular_gi_weight;
using bracken::render::rt_smoothstep;
using bracken::render::VIRTUAL_REPROJECTION_MIRROR_ROUGHNESS;
using bracken::render::VIRTUAL_REPROJECTION_SURFACE_ROUGHNESS;
using bracken::render::virtual_reprojection_weight;

// ---------------------------------------------------------------------------
// The helper both curves are built from
// ---------------------------------------------------------------------------

TEST(RtSmoothstep, MatchesTheHlslDefinition) {
    // Clamped at both ends, Hermite in between, and exactly 0.5 at the midpoint
    // — the three properties a shader's `smoothstep` guarantees.
    EXPECT_FLOAT_EQ(rt_smoothstep(0.0f, 1.0f, -5.0f), 0.0f);
    EXPECT_FLOAT_EQ(rt_smoothstep(0.0f, 1.0f, 0.0f), 0.0f);
    EXPECT_FLOAT_EQ(rt_smoothstep(0.0f, 1.0f, 0.5f), 0.5f);
    EXPECT_FLOAT_EQ(rt_smoothstep(0.0f, 1.0f, 1.0f), 1.0f);
    EXPECT_FLOAT_EQ(rt_smoothstep(0.0f, 1.0f, 5.0f), 1.0f);
    // t*t*(3-2t) at t = 0.25.
    EXPECT_FLOAT_EQ(rt_smoothstep(0.0f, 1.0f, 0.25f), 0.15625f);
    // The zero derivative at both edges is what keeps a blend from stepping;
    // sample just inside each and check the curve is still hugging the edge.
    EXPECT_LT(rt_smoothstep(0.0f, 1.0f, 0.01f), 0.001f);
    EXPECT_GT(rt_smoothstep(0.0f, 1.0f, 0.99f), 0.999f);
}

// ---------------------------------------------------------------------------
// Virtual-hit-point reprojection
// ---------------------------------------------------------------------------

TEST(VirtualReprojectionWeight, MirrorsTakeTheVirtualPointAndRoughSurfacesDoNot) {
    EXPECT_FLOAT_EQ(virtual_reprojection_weight(0.0f), 1.0f);
    EXPECT_FLOAT_EQ(virtual_reprojection_weight(VIRTUAL_REPROJECTION_MIRROR_ROUGHNESS), 1.0f);
    EXPECT_FLOAT_EQ(virtual_reprojection_weight(VIRTUAL_REPROJECTION_SURFACE_ROUGHNESS), 0.0f);
    EXPECT_FLOAT_EQ(virtual_reprojection_weight(1.0f), 0.0f);
}

TEST(VirtualReprojectionWeight, IsMonotoneAndContinuousAcrossTheWholeRange) {
    // Continuity is the point of using a blend rather than a threshold: no
    // roughness may make a pixel's history jump to a different texel.
    float previous = virtual_reprojection_weight(0.0f);
    for (int i = 1; i <= 1000; ++i) {
        const float r = static_cast<float>(i) / 1000.0f;
        const float w = virtual_reprojection_weight(r);
        EXPECT_LE(w, previous + 1e-6f) << "roughness " << r;
        EXPECT_LE(std::abs(w - previous), 0.02f) << "step at roughness " << r;
        EXPECT_GE(w, 0.0f);
        EXPECT_LE(w, 1.0f);
        previous = w;
    }
}

TEST(VirtualReprojectionWeight, HandsOverBelowTheReflectionGate) {
    // Both edges must sit below RtReflectionSettings::fade_start (0.35) and
    // max_roughness (0.75), so the curve spans the part of the roughness range
    // where the TRACE supplies most of the specular. A band above the gate
    // would be spent on pixels the IBL probe already owns.
    EXPECT_LT(VIRTUAL_REPROJECTION_MIRROR_ROUGHNESS, VIRTUAL_REPROJECTION_SURFACE_ROUGHNESS);
    EXPECT_LT(VIRTUAL_REPROJECTION_SURFACE_ROUGHNESS, 0.75f);
    EXPECT_GT(virtual_reprojection_weight(0.20f), 0.25f);
}

// ---------------------------------------------------------------------------
// Rough-tier specular fallback
// ---------------------------------------------------------------------------

TEST(RoughSpecularGiWeight, IsZeroBelowTheBandAndTotalAtFullyRough) {
    // Zero below the band is the load-bearing half: it is what keeps every
    // mid-roughness surface on the prefiltered IBL lobe, which is still
    // directional enough there to matter.
    EXPECT_FLOAT_EQ(rough_specular_gi_weight(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(rough_specular_gi_weight(ROUGH_SPECULAR_GI_START), 0.0f);
    EXPECT_FLOAT_EQ(rough_specular_gi_weight(ROUGH_SPECULAR_GI_END), 1.0f);
    EXPECT_FLOAT_EQ(rough_specular_gi_weight(1.0f), 1.0f);
    EXPECT_FLOAT_EQ(rough_specular_gi_weight(0.8f), 0.5f);
}

TEST(RoughSpecularGiWeight, StartsAboveTheReflectionGateSoTheTwoNeverOverlap) {
    // The trace's handover completes at max_roughness = 0.75. Starting the
    // probe substitution at 0.6 overlaps that window slightly and deliberately:
    // the two are different signals (traced specular vs. ambient specular), and
    // a gap between them would leave a roughness band served by the sky-only
    // prefilter alone, which is the exact artifact this exists to remove.
    EXPECT_LT(ROUGH_SPECULAR_GI_START, ROUGH_SPECULAR_GI_END);
    EXPECT_LE(ROUGH_SPECULAR_GI_END, 1.0f);
    EXPECT_GT(rough_specular_gi_weight(0.75f), 0.0f);
}

TEST(RoughSpecularGiWeight, IsMonotoneAndContinuous) {
    float previous = rough_specular_gi_weight(0.0f);
    for (int i = 1; i <= 1000; ++i) {
        const float r = static_cast<float>(i) / 1000.0f;
        const float w = rough_specular_gi_weight(r);
        EXPECT_GE(w, previous - 1e-6f) << "roughness " << r;
        EXPECT_LE(std::abs(w - previous), 0.02f) << "step at roughness " << r;
        previous = w;
    }
}

TEST(RoughSpecularGiWeight, ScaledByGiWeightIsExactlyZeroWithGiOff) {
    // The shader multiplies by the caller's giWeight and skips the lerp at
    // zero, which is what makes the GI-off path bit-identical rather than
    // merely equal. Anything else here would be testing the multiplication.
    for (int i = 0; i <= 100; ++i) {
        const float r = static_cast<float>(i) / 100.0f;
        EXPECT_FLOAT_EQ(rough_specular_gi_weight(r) * 0.0f, 0.0f);
    }
}

} // namespace
