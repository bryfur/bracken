// Unit tests for the CPU side of the volumetric fog pass: EnvironmentState
// plumbing into VolumetricFogUniforms and the per-cascade god-ray data
// (audit findings H4 / M1 / L3 in docs/audits/graphics.md).

#include "bracken/render/volumetric_fog.hpp"

#include "bracken/contracts/camera_state.hpp"
#include "bracken/contracts/environment.hpp"
#include "bracken/render/shadow_map.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

using bracken::render::EnvironmentState;
using bracken::render::CSM_MAX_CASCADES;
using bracken::render::ShadowMap;
using bracken::render::VolumetricFog;
using bracken::scene::CameraState;

namespace {

CameraState make_camera(const glm::vec3& position = glm::vec3(0.0f, 10.0f, 0.0f)) {
    CameraState camera;
    camera.position = position;
    camera.view = glm::lookAt(position, position + glm::vec3(0, 0, -1), glm::vec3(0, 1, 0));
    camera.projection = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 5000.0f);
    camera.view_projection = camera.projection * camera.view;
    return camera;
}

} // namespace

// H4/L3: the values that used to be hardcoded inside VolumetricFog::render
// are now EnvironmentState defaults — default visuals must not change.
TEST(VolumetricFogEnvironment, DefaultsMatchLegacyHardcodedValues) {
    EnvironmentState env;
    EXPECT_EQ(env.fog_volumetric_sun_color, glm::vec3(1.0f, 0.9f, 0.7f));
    EXPECT_FLOAT_EQ(env.fog_volumetric_scatter, 0.25f);
    EXPECT_FLOAT_EQ(env.fog_volumetric_density, 0.0015f);
    EXPECT_FLOAT_EQ(env.fog_volumetric_height, 100.0f);
    EXPECT_FLOAT_EQ(env.fog_volumetric_falloff, 0.005f);
    EXPECT_FLOAT_EQ(env.fog_volumetric_start, 100.0f);  // L3: near dead zone
    EXPECT_FLOAT_EQ(env.fog_volumetric_end, 3000.0f);   // L3: march cap
}

// H4: every environment parameter must land in the uniform block.
TEST(VolumetricFogUniformsBuild, PlumbsEnvironmentState) {
    VolumetricFog fog;
    ShadowMap shadow_map;
    CameraState camera = make_camera(glm::vec3(3.0f, 25.0f, -7.0f));

    EnvironmentState env;
    env.sun_dir = glm::normalize(glm::vec3(0.3f, -0.9f, 0.1f));
    env.fog_color_horizon = glm::vec3(0.9f, 0.5f, 0.3f); // sunset
    env.fog_volumetric_sun_color = glm::vec3(2.0f, 0.8f, 0.4f);
    env.fog_volumetric_scatter = 0.5f;
    env.fog_volumetric_density = 0.01f;
    env.fog_volumetric_height = 40.0f;
    env.fog_volumetric_falloff = 0.02f;
    env.fog_volumetric_start = 5.0f;
    env.fog_volumetric_end = 800.0f;

    auto u = fog.build_uniforms(shadow_map, camera, env, true, true, 2.0f);

    EXPECT_EQ(u.lightDir, env.sun_dir);
    EXPECT_EQ(u.lightColor, env.fog_volumetric_sun_color);
    EXPECT_EQ(u.fogColor, env.fog_color_horizon);
    EXPECT_FLOAT_EQ(u.scatterStrength, 0.5f);
    EXPECT_FLOAT_EQ(u.fogDensity, 0.01f);
    EXPECT_FLOAT_EQ(u.fogHeight, 40.0f);
    EXPECT_FLOAT_EQ(u.fogFalloff, 0.02f);
    EXPECT_FLOAT_EQ(u.startDist, 5.0f);
    EXPECT_FLOAT_EQ(u.maxDist, 800.0f);
    EXPECT_FLOAT_EQ(u.godRaysEnabled, 1.0f);
    EXPECT_FLOAT_EQ(u.densityMultiplier, 2.0f);
    EXPECT_EQ(u.cameraPos, camera.position);

    // invViewProjection must invert the camera's view-projection.
    glm::mat4 identity = u.invViewProjection * camera.view_projection;
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            EXPECT_NEAR(identity[c][r], (c == r) ? 1.0f : 0.0f, 1e-3f);
        }
    }
}

TEST(VolumetricFogUniformsBuild, DisabledFogZeroesDensityOnly) {
    VolumetricFog fog;
    ShadowMap shadow_map;
    EnvironmentState env;

    auto u = fog.build_uniforms(shadow_map, make_camera(), env, false, false, 1.0f);
    EXPECT_FLOAT_EQ(u.fogDensity, 0.0f);
    EXPECT_FLOAT_EQ(u.godRaysEnabled, 0.0f);
    // The rest of the environment still flows through.
    EXPECT_FLOAT_EQ(u.fogHeight, env.fog_volumetric_height);
}

// M1: god rays use all cascades — the uniforms must carry the full matrix
// array, ascending splits, per-cascade bias, and the last-active split.
TEST(VolumetricFogUniformsBuild, FillsAllFourCascades) {
    VolumetricFog fog;
    ShadowMap shadow_map; // cascade fitting is pure CPU math (no GPU init)
    CameraState camera = make_camera();
    glm::vec3 light_dir = glm::normalize(glm::vec3(-0.5f, -0.8f, -0.3f));
    shadow_map.update(camera.view, camera.projection, light_dir, 0.1f, 1200.0f);

    EnvironmentState env;
    auto u = fog.build_uniforms(shadow_map, camera, env, true, true, 1.0f);

    const auto& cascades = shadow_map.cascades();
    for (int i = 0; i < CSM_MAX_CASCADES; ++i) {
        EXPECT_EQ(u.shadowLightViewProjection[i], cascades[i].light_view_projection) << "cascade " << i;
        EXPECT_FLOAT_EQ(u.cascadeSplits[i], cascades[i].split_depth) << "cascade " << i;
        EXPECT_GT(u.shadowBias[i], 0.0f) << "cascade " << i;
        EXPECT_FLOAT_EQ(u.shadowBias[i], 1.5f * cascades[i].texel_world / cascades[i].depth_range) << "cascade " << i;
    }
    // Splits ascend so distance-based selection walks near -> far.
    EXPECT_LT(u.cascadeSplits[0], u.cascadeSplits[1]);
    EXPECT_LT(u.cascadeSplits[1], u.cascadeSplits[2]);
    EXPECT_LT(u.cascadeSplits[2], u.cascadeSplits[3]);
    EXPECT_FLOAT_EQ(u.lastActiveSplit, cascades[CSM_MAX_CASCADES - 1].split_depth);
    EXPECT_FLOAT_EQ(u.shadowAtlasTexel, 0.5f / static_cast<float>(shadow_map.resolution()));
}

// Inactive cascades clone the last active one (same scheme as
// ShadowMap::get_shadow_uniforms) so dynamic indexing is always safe.
TEST(VolumetricFogUniformsBuild, InactiveCascadesCloneLastActive) {
    VolumetricFog fog;
    ShadowMap shadow_map;
    shadow_map.set_active_cascades(3);
    CameraState camera = make_camera();
    glm::vec3 light_dir = glm::normalize(glm::vec3(-0.5f, -0.8f, -0.3f));
    shadow_map.update(camera.view, camera.projection, light_dir, 0.1f, 1200.0f);

    EnvironmentState env;
    auto u = fog.build_uniforms(shadow_map, camera, env, true, true, 1.0f);

    const auto& cascades = shadow_map.cascades();
    EXPECT_EQ(u.shadowLightViewProjection[3], cascades[2].light_view_projection);
    EXPECT_FLOAT_EQ(u.cascadeSplits[3], cascades[2].split_depth);
    EXPECT_FLOAT_EQ(u.lastActiveSplit, cascades[2].split_depth);
}
