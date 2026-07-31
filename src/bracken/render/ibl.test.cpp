#include "bracken/render/ibl.hpp"
#include <cmath>
#include <glm/glm.hpp>
#include <gtest/gtest.h>

using namespace bracken::render;

namespace {

// Reference implementation of the Vulkan/D3D cube-lookup convention: map a
// direction to (face, u, v) with v growing downward. Used only to verify the
// engine's face-basis table round-trips against the spec.
struct FaceUV {
    int face;
    float u;
    float v;
};

FaceUV direction_to_face_uv(const glm::vec3& d) {
    const glm::vec3 a = glm::abs(d);
    float ma = 0.0f;
    float sc = 0.0f;
    float tc = 0.0f;
    int face = 0;
    if (a.x >= a.y && a.x >= a.z) {
        ma = a.x;
        if (d.x > 0) { face = 0; sc = -d.z; tc = -d.y; }
        else         { face = 1; sc = d.z;  tc = -d.y; }
    } else if (a.y >= a.z) {
        ma = a.y;
        if (d.y > 0) { face = 2; sc = d.x; tc = d.z; }
        else         { face = 3; sc = d.x; tc = -d.z; }
    } else {
        ma = a.z;
        if (d.z > 0) { face = 4; sc = d.x;  tc = -d.y; }
        else         { face = 5; sc = -d.x; tc = -d.y; }
    }
    return {face, 0.5f * (sc / ma + 1.0f), 0.5f * (tc / ma + 1.0f)};
}

} // namespace

// ============================================================================
// Face basis / direction math (must match the sky_capture / ibl_* shaders and
// the Vulkan cube lookup convention)
// ============================================================================

TEST(IblFaceBasis, CentersMapToMajorAxes) {
    const glm::vec3 expected[6] = {
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
    };
    for (int face = 0; face < 6; ++face) {
        const glm::vec3 dir = ibl_face_direction(face, 0.5f, 0.5f);
        EXPECT_NEAR(glm::dot(dir, expected[face]), 1.0f, 1e-6f) << "face " << face;
    }
}

TEST(IblFaceBasis, BasesAreOrthonormal) {
    for (int face = 0; face < 6; ++face) {
        const IblFaceBasis b = ibl_face_basis(face);
        EXPECT_NEAR(glm::length(b.forward), 1.0f, 1e-6f);
        EXPECT_NEAR(glm::length(b.right), 1.0f, 1e-6f);
        EXPECT_NEAR(glm::length(b.up), 1.0f, 1e-6f);
        EXPECT_NEAR(glm::dot(b.forward, b.right), 0.0f, 1e-6f);
        EXPECT_NEAR(glm::dot(b.forward, b.up), 0.0f, 1e-6f);
        EXPECT_NEAR(glm::dot(b.right, b.up), 0.0f, 1e-6f);
    }
}

TEST(IblFaceBasis, RoundTripsAgainstCubeLookupConvention) {
    // A direction rendered at face texel (u, v) must read back from the same
    // (face, u, v) through the standard cube lookup — otherwise the prefilter
    // writes and the TextureCube reads would disagree about orientation.
    for (int face = 0; face < 6; ++face) {
        for (float u = 0.1f; u < 1.0f; u += 0.2f) {
            for (float v = 0.1f; v < 1.0f; v += 0.2f) {
                const glm::vec3 dir = ibl_face_direction(face, u, v);
                const FaceUV back = direction_to_face_uv(dir);
                ASSERT_EQ(back.face, face) << "face " << face << " u " << u << " v " << v;
                EXPECT_NEAR(back.u, u, 1e-5f);
                EXPECT_NEAR(back.v, v, 1e-5f);
            }
        }
    }
}

// ============================================================================
// Hammersley radical inverse (CPU mirror of the shader implementation)
// ============================================================================

TEST(IblRadicalInverse, MatchesKnownValues) {
    EXPECT_FLOAT_EQ(ibl_radical_inverse(0u), 0.0f);
    EXPECT_FLOAT_EQ(ibl_radical_inverse(1u), 0.5f);
    EXPECT_FLOAT_EQ(ibl_radical_inverse(2u), 0.25f);
    EXPECT_FLOAT_EQ(ibl_radical_inverse(3u), 0.75f);
    EXPECT_FLOAT_EQ(ibl_radical_inverse(4u), 0.125f);
    EXPECT_FLOAT_EQ(ibl_radical_inverse(5u), 0.625f);
}

TEST(IblRadicalInverse, StaysInUnitInterval) {
    for (uint32_t i = 0; i < 1024; ++i) {
        const float x = ibl_radical_inverse(i * 2654435761u); // scrambled indices
        EXPECT_GE(x, 0.0f);
        EXPECT_LT(x, 1.0f);
    }
}

// ============================================================================
// Refresh plan (unit decode + roughness schedule)
// ============================================================================

TEST(IblPlan, UnitCountArithmetic) {
    EXPECT_EQ(IBL::SPECULAR_UNITS, static_cast<int>(IBL::SPECULAR_MIPS) * 6);
    EXPECT_EQ(IBL::TOTAL_UNITS, IBL::CAPTURE_UNITS + 1 + IBL::SPECULAR_UNITS + IBL::IRRADIANCE_UNITS);
    // The amortization contract: a full refresh completes within a handful of
    // frames (~7 at the current budget) — never a single-frame hitch, never
    // longer than a second at any playable frame rate.
    const int frames = (IBL::TOTAL_UNITS + IBL::UNITS_PER_FRAME - 1) / IBL::UNITS_PER_FRAME;
    EXPECT_GT(frames, 1);
    EXPECT_LE(frames, 16);
}

TEST(IblPlan, DecodeCoversEveryFaceAndMipExactlyOnce) {
    int capture_faces[6] = {};
    int irradiance_faces[6] = {};
    int specular_cells[IBL::SPECULAR_MIPS][6] = {};
    int mipgen = 0;

    for (int unit = 0; unit < IBL::TOTAL_UNITS; ++unit) {
        const IBL::UnitDesc d = IBL::decode_unit(unit);
        switch (d.kind) {
            case IBL::UnitKind::CaptureFace:
                ASSERT_GE(d.face, 0);
                ASSERT_LT(d.face, 6);
                ++capture_faces[d.face];
                EXPECT_EQ(mipgen, 0) << "capture faces must precede mip generation";
                break;
            case IBL::UnitKind::CaptureMips:
                ++mipgen;
                break;
            case IBL::UnitKind::SpecularFaceMip:
                ASSERT_GE(d.face, 0);
                ASSERT_LT(d.face, 6);
                ASSERT_GE(d.mip, 0);
                ASSERT_LT(d.mip, static_cast<int>(IBL::SPECULAR_MIPS));
                ++specular_cells[d.mip][d.face];
                EXPECT_EQ(mipgen, 1) << "specular prefilter needs the capture mips";
                break;
            case IBL::UnitKind::IrradianceFace:
                ASSERT_GE(d.face, 0);
                ASSERT_LT(d.face, 6);
                ++irradiance_faces[d.face];
                EXPECT_EQ(mipgen, 1);
                break;
        }
    }

    EXPECT_EQ(mipgen, 1);
    for (int face = 0; face < 6; ++face) {
        EXPECT_EQ(capture_faces[face], 1);
        EXPECT_EQ(irradiance_faces[face], 1);
        for (uint32_t mip = 0; mip < IBL::SPECULAR_MIPS; ++mip) {
            EXPECT_EQ(specular_cells[mip][face], 1);
        }
    }
}

TEST(IblPlan, MipRoughnessSpansUnitInterval) {
    EXPECT_FLOAT_EQ(IBL::mip_roughness(0), 0.0f);
    EXPECT_FLOAT_EQ(IBL::mip_roughness(static_cast<int>(IBL::SPECULAR_MIPS) - 1), 1.0f);
    for (uint32_t mip = 1; mip < IBL::SPECULAR_MIPS; ++mip) {
        EXPECT_GT(IBL::mip_roughness(static_cast<int>(mip)), IBL::mip_roughness(static_cast<int>(mip) - 1));
    }
}

// ============================================================================
// Sky-change detection
// ============================================================================

TEST(IblSkyParams, EqualityTracksOnlySkyRelevantFields) {
    bracken::render::EnvironmentState env{};
    const IblSkyParams base = IblSkyParams::from_environment(env);

    // Non-sky fields (fog distances, ambient hemisphere, volumetrics) must
    // NOT dirty the chain. cloud_speed is deliberately in this set: it is
    // time-like (it only scales the scroll-phase accumulation), and the
    // scroll itself never triggers a refresh — reflected clouds stay frozen
    // between refreshes.
    env.fog_start = 1.0f;
    env.fog_end = 2.0f;
    env.ambient_sky = {9.0f, 9.0f, 9.0f};
    env.fog_volumetric_density = 1.0f;
    env.cloud_speed = 4.0f;
    EXPECT_EQ(IblSkyParams::from_environment(env), base);

    // Every sky-relevant field must dirty it.
    bracken::render::EnvironmentState e2{};
    e2.sun_dir = glm::normalize(glm::vec3(0.3f, -0.5f, 0.1f));
    EXPECT_NE(IblSkyParams::from_environment(e2), base);

    bracken::render::EnvironmentState e3{};
    e3.sky_zenith *= 0.5f;
    EXPECT_NE(IblSkyParams::from_environment(e3), base);

    bracken::render::EnvironmentState e4{};
    e4.fog_color_sun *= 0.5f;
    EXPECT_NE(IblSkyParams::from_environment(e4), base);

    // The cloud/haze SHAPING params relight the sky, so they must dirty it
    // (unlike the scroll phase / cloud_speed above).
    bracken::render::EnvironmentState e5{};
    e5.cloud_coverage = 0.9f;
    EXPECT_NE(IblSkyParams::from_environment(e5), base);

    bracken::render::EnvironmentState e6{};
    e6.cloud_density = 0.2f;
    EXPECT_NE(IblSkyParams::from_environment(e6), base);

    bracken::render::EnvironmentState e7{};
    e7.cloud_scale = 2.0f;
    EXPECT_NE(IblSkyParams::from_environment(e7), base);

    bracken::render::EnvironmentState e8{};
    e8.haze_strength = 0.0f;
    EXPECT_NE(IblSkyParams::from_environment(e8), base);
}

TEST(IblSkyParams, EnvironmentCloudDefaultsPreserveClassicLook) {
    // The defaults are the "classic" sky: coverage 0.35 (the old compile-time
    // CLOUD_COVERAGE), all multipliers neutral. Changing a default here
    // changes every consumer's out-of-the-box look — do it deliberately.
    const bracken::render::EnvironmentState env{};
    EXPECT_FLOAT_EQ(env.cloud_coverage, 0.35f);
    EXPECT_FLOAT_EQ(env.cloud_density, 1.0f);
    EXPECT_FLOAT_EQ(env.cloud_scale, 1.0f);
    EXPECT_FLOAT_EQ(env.cloud_speed, 1.0f);
    EXPECT_FLOAT_EQ(env.haze_strength, 1.0f);
}
