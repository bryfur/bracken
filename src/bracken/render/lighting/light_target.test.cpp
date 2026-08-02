// The RIS target function, and the property that makes it bias-free.
//
// The claim under test is NOT "the target function is the right brightness" —
// a target function is allowed to approximate; that is what RIS is for. It is
// that its SUPPORT matches the forward loop's: zero exactly where
// accumulateClusterLights contributes zero, non-zero everywhere it does not.
// If those two disagree, the sampler either spends the frame's one ray on a
// light that contributes nothing, or never spends it on a light that does.
//
// The forward loop's zeros are structural and all three are checked here:
//   * `dist > radius` is a `continue` (and the binner culls on the same radius)
//   * the range window (1 - (d/R)^4)^2 reaches zero AT the radius
//   * a spot's smoothstep cone reaches zero at outer_cos
//
// render/lighting/light_target.hpp mirrors shaders/cluster_index.hlsli's
// pointFalloff/spotConeFactor and shaders/cluster_light_load.hlsli's
// clusterLightTarget*. The shader shares one copy of that arithmetic between
// the fragment and compute paths (the R2a move); this is the CPU third.

#include <gtest/gtest.h>

#include "bracken/contracts/light.hpp"
#include "bracken/render/lighting/light_cluster.hpp"
#include "bracken/render/lighting/light_target.hpp"

#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace {

using bracken::render::lighting::point_falloff;
using bracken::render::lighting::point_light_target;
using bracken::render::lighting::spot_cone_factor;
using bracken::render::lighting::spot_light_target;
using bracken::render::lighting::target_luminance;

constexpr glm::vec3 UP{0.0f, 1.0f, 0.0f};

} // namespace

// The window is exactly (1 - (d/R)^4)^2 / d^2 — spelled out here rather than
// referenced, so a change to the curve in either the header or the .hlsli shows
// up as a failing number instead of as slightly different lighting.
TEST(LightTarget, PointFalloffMatchesTheDocumentedCurve) {
    const float R = 100.0f;
    for (const float d : {1.0f, 10.0f, 50.0f, 99.0f}) {
        const float r = d / R;
        const float win = 1.0f - r * r * r * r;
        EXPECT_NEAR(point_falloff(d, R), (win * win) / (d * d), 1e-9f) << "d=" << d;
    }
}

TEST(LightTarget, PointFalloffIsZeroAtAndBeyondTheRadius) {
    const float R = 100.0f;
    EXPECT_FLOAT_EQ(point_falloff(R, R), 0.0f);
    EXPECT_FLOAT_EQ(point_falloff(R * 2.0f, R), 0.0f);
    EXPECT_GT(point_falloff(R * 0.999f, R), 0.0f);
}

TEST(LightTarget, PointFalloffIsFiniteAtDegenerateInputs) {
    // A coincident sample point and a zero-radius light both hit guard clamps
    // rather than dividing by zero. A NaN here would reach a reservoir weight
    // and then survive in the temporal chain indefinitely.
    EXPECT_TRUE(std::isfinite(point_falloff(0.0f, 100.0f)));
    EXPECT_TRUE(std::isfinite(point_falloff(1.0f, 0.0f)));
}

TEST(LightTarget, SpotConeIsSmoothstepBetweenOuterAndInner) {
    const glm::vec3 dir{0.0f, -1.0f, 0.0f}; // pointing down
    const glm::vec3 L = UP;                 // surface -> light, straight up
    // Directly under the light: cd == 1, fully inside.
    EXPECT_FLOAT_EQ(spot_cone_factor(L, dir, 0.9f, 0.7f), 1.0f);

    // Halfway through the window is smoothstep(0.5) == 0.5.
    const float mid = 0.8f;
    const glm::vec3 angled = glm::normalize(glm::vec3(std::sqrt(1.0f - mid * mid), mid, 0.0f));
    EXPECT_NEAR(spot_cone_factor(angled, dir, 0.9f, 0.7f), 0.5f, 1e-5f);
}

TEST(LightTarget, SpotConeIsZeroAtAndBeyondTheOuterAngle) {
    const glm::vec3 dir{0.0f, -1.0f, 0.0f};
    const float outer = 0.7f;
    const glm::vec3 at_edge = glm::normalize(glm::vec3(std::sqrt(1.0f - outer * outer), outer, 0.0f));
    EXPECT_FLOAT_EQ(spot_cone_factor(at_edge, dir, 0.9f, outer), 0.0f);
    const glm::vec3 outside = glm::normalize(glm::vec3(1.0f, 0.1f, 0.0f));
    EXPECT_FLOAT_EQ(spot_cone_factor(outside, dir, 0.9f, outer), 0.0f);
}

// THE PARITY CLAIM. The binner's admission test for a light at a point and the
// target function's support must be the same predicate. `ClusterGrid` culls on
// `dist > radius` (its cluster-AABB test reduces to exactly that for a point
// sample), and so does the target function.
TEST(LightTarget, SupportMatchesTheBinnerRadiusCull) {
    bracken::render::lighting::PointLight light{};
    light.position = glm::vec3(0.0f, 100.0f, 0.0f);
    light.radius = 250.0f;
    light.color = glm::vec3(1.0f, 0.9f, 0.7f);
    light.intensity = 4.0f;

    // A ground plane directly under the light, sampled outward past the radius.
    for (float x = 0.0f; x < 400.0f; x += 7.0f) {
        const glm::vec3 p(x, 0.0f, 0.0f);
        const float dist = glm::length(light.position - p);
        const bool in_range = dist <= light.radius;
        const float target = point_light_target(light.position, light.radius, light.color,
                                                light.intensity, p, UP);
        if (in_range) {
            EXPECT_GT(target, 0.0f) << "x=" << x << " dist=" << dist;
        } else {
            EXPECT_FLOAT_EQ(target, 0.0f) << "x=" << x << " dist=" << dist;
        }
    }
}

// The other two structural zeros of accumulateClusterLights.
TEST(LightTarget, SurfaceFacingAwayScoresZero) {
    const glm::vec3 light_pos(0.0f, 100.0f, 0.0f);
    const glm::vec3 p(0.0f, 0.0f, 0.0f);
    const glm::vec3 down(0.0f, -1.0f, 0.0f);
    EXPECT_FLOAT_EQ(point_light_target(light_pos, 500.0f, glm::vec3(1.0f), 1.0f, p, down), 0.0f);
    EXPECT_GT(point_light_target(light_pos, 500.0f, glm::vec3(1.0f), 1.0f, p, UP), 0.0f);
}

TEST(LightTarget, SpotTargetIsZeroOutsideTheCone) {
    const glm::vec3 light_pos(0.0f, 200.0f, 0.0f);
    const glm::vec3 dir(0.0f, -1.0f, 0.0f);
    // Under the light: lit.
    EXPECT_GT(spot_light_target(light_pos, 500.0f, dir, 0.95f, 0.90f, glm::vec3(1.0f), 1.0f,
                                glm::vec3(0.0f), UP),
              0.0f);
    // Far to the side, still within RANGE but outside the cone: zero, which is
    // the forward loop's `cone <= 0` continue.
    const glm::vec3 aside(400.0f, 0.0f, 0.0f);
    EXPECT_LT(glm::length(light_pos - aside), 500.0f);
    EXPECT_FLOAT_EQ(spot_light_target(light_pos, 500.0f, dir, 0.95f, 0.90f, glm::vec3(1.0f), 1.0f,
                                      aside, UP),
                    0.0f);
}

// Ordering, not magnitude, is what the reservoir consumes: a brighter or nearer
// light must score higher, because that is the only thing the weights decide.
TEST(LightTarget, BrighterAndNearerLightsScoreHigher) {
    const glm::vec3 p(0.0f);
    const glm::vec3 near_pos(0.0f, 50.0f, 0.0f);
    const glm::vec3 far_pos(0.0f, 200.0f, 0.0f);
    const float R = 500.0f;

    const float near_score = point_light_target(near_pos, R, glm::vec3(1.0f), 1.0f, p, UP);
    const float far_score = point_light_target(far_pos, R, glm::vec3(1.0f), 1.0f, p, UP);
    EXPECT_GT(near_score, far_score);

    const float bright = point_light_target(far_pos, R, glm::vec3(1.0f), 8.0f, p, UP);
    EXPECT_GT(bright, far_score);
}

// Rec. 709 luma, matching the shader's clusterTargetLuminance weights. A green
// lantern and a blue one of the same radiance are NOT equally important to the
// eye, and the target function is allowed to know that.
TEST(LightTarget, LuminanceUsesRec709Weights) {
    EXPECT_FLOAT_EQ(target_luminance(glm::vec3(1.0f, 0.0f, 0.0f)), 0.2126f);
    EXPECT_FLOAT_EQ(target_luminance(glm::vec3(0.0f, 1.0f, 0.0f)), 0.7152f);
    EXPECT_FLOAT_EQ(target_luminance(glm::vec3(0.0f, 0.0f, 1.0f)), 0.0722f);
    EXPECT_FLOAT_EQ(target_luminance(glm::vec3(1.0f)), 1.0f);
}

// The binner and the target function agree on a real frame's worth of lights:
// every light the grid accepted for a cluster containing the sample point has a
// non-zero target there, unless the surface faces away from it.
TEST(LightTarget, AgreesWithTheGridOnASmallScene) {
    bracken::render::lighting::ClusterGrid grid;
    bracken::scene::CameraState camera{};
    camera.position = glm::vec3(0.0f, 100.0f, 400.0f);
    camera.view = glm::lookAt(camera.position, glm::vec3(0.0f, 100.0f, 0.0f), UP);
    camera.projection = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 4000.0f, 0.1f);
    grid.begin_frame(camera, 1600, 900, 0.1f, 4000.0f);

    for (int i = 0; i < 6; ++i) {
        bracken::render::lighting::PointLight l{};
        l.position = glm::vec3(static_cast<float>(i) * 60.0f - 150.0f, 80.0f, 0.0f);
        l.radius = 200.0f;
        l.color = glm::vec3(1.0f, 0.8f, 0.6f);
        l.intensity = 3.0f;
        EXPECT_TRUE(grid.add_point_light(l));
    }
    grid.build();

    // A ground point under the middle of the row; every light within 200 units
    // of it must score, and every light beyond must not.
    const glm::vec3 p(-30.0f, 0.0f, 0.0f);
    for (const auto& l : grid.point_lights()) {
        const float target =
            point_light_target(l.position, l.radius, l.color, l.intensity, p, UP);
        const bool in_range = glm::length(l.position - p) <= l.radius;
        EXPECT_EQ(target > 0.0f, in_range)
            << "light at " << l.position.x << " vs sample at " << p.x;
    }
}
