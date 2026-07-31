#include "bracken/render/taa.hpp"
#include "bracken/scene/velocity_math.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

using namespace bracken::render;

namespace {

// Engine-convention reversed-Z RH_ZO projection (near/far swapped).
glm::mat4 projection() {
    return glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 15000.0f, 5.0f);
}

glm::mat4 look_at(const glm::vec3& eye, const glm::vec3& dir = {0.0f, 0.0f, -1.0f}) {
    return glm::lookAt(eye, eye + dir, glm::vec3(0.0f, 1.0f, 0.0f));
}

// Project a world point to (top-left-origin uv, device depth) under `vp`.
struct Projected {
    glm::vec2 uv;
    float depth;
};

Projected project(const glm::mat4& vp, const glm::vec3& world) {
    const glm::vec4 clip = vp * glm::vec4(world, 1.0f);
    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    return {glm::vec2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f), ndc.z};
}

} // namespace

TEST(TaaDepthReproject, StaticCameraIsIdentity) {
    const glm::mat4 vp = projection() * look_at({0.0f, 0.0f, 10.0f});
    const glm::vec3 world(1.0f, 2.0f, -40.0f);
    const auto p = project(vp, world);

    const glm::vec2 prev_uv =
        depth_reproject_prev_uv(glm::inverse(vp), vp, {0.0f, 0.0f, 10.0f}, p.uv, p.depth);
    EXPECT_NEAR(prev_uv.x, p.uv.x, 1e-5f);
    EXPECT_NEAR(prev_uv.y, p.uv.y, 1e-5f);
}

TEST(TaaDepthReproject, RecoversPreviousUVForStaticGeometry) {
    // A static world point seen from two camera positions: reprojection must
    // land on the point's uv under the PREVIOUS camera.
    const glm::vec3 world(-3.0f, 1.5f, -60.0f);
    const glm::vec3 curr_eye(0.5f, -0.25f, 2.0f);
    const glm::vec3 prev_eye(0.0f, 0.0f, 0.0f);
    const glm::mat4 curr_vp = projection() * look_at(curr_eye);
    const glm::mat4 prev_vp = projection() * look_at(prev_eye);

    const auto curr = project(curr_vp, world);
    const auto expected_prev = project(prev_vp, world);

    const glm::vec2 prev_uv =
        depth_reproject_prev_uv(glm::inverse(curr_vp), prev_vp, curr_eye, curr.uv, curr.depth);
    EXPECT_NEAR(prev_uv.x, expected_prev.uv.x, 1e-4f);
    EXPECT_NEAR(prev_uv.y, expected_prev.uv.y, 1e-4f);
}

TEST(TaaDepthReproject, AgreesWithPrepassVelocityConvention) {
    // For a prepass-covered static pixel the shader uses prev_uv = uv -
    // velocity (velocity_math.hpp); for the same pixel the depth-reprojection
    // fallback must produce the same answer, or velocity-less neighbors (MASK
    // foliage) would swim against their velocity-covered surroundings.
    const glm::vec3 world(4.0f, -1.0f, -80.0f);
    const glm::vec3 curr_eye(1.0f, 0.5f, 3.0f);
    const glm::mat4 curr_vp = projection() * look_at(curr_eye);
    const glm::mat4 prev_vp = projection() * look_at({0.0f, 0.0f, 0.0f});

    const auto curr = project(curr_vp, world);
    const glm::vec2 velocity =
        bracken::scene::compute_uv_velocity(curr_vp, prev_vp, glm::mat4(1.0f), glm::mat4(1.0f), world);
    const glm::vec2 via_velocity = curr.uv - velocity;

    const glm::vec2 via_depth =
        depth_reproject_prev_uv(glm::inverse(curr_vp), prev_vp, curr_eye, curr.uv, curr.depth);
    EXPECT_NEAR(via_depth.x, via_velocity.x, 1e-4f);
    EXPECT_NEAR(via_depth.y, via_velocity.y, 1e-4f);
}

TEST(TaaDepthReproject, SkyIgnoresCameraTranslation) {
    // Device depth 0 (reversed-Z cleared far plane) reprojects the view-ray
    // direction with w = 0: pure translation must not move the sky.
    const glm::vec3 curr_eye(100.0f, 25.0f, -300.0f);
    const glm::mat4 curr_vp = projection() * look_at(curr_eye);
    const glm::mat4 prev_vp = projection() * look_at({0.0f, 0.0f, 0.0f}); // same orientation

    const glm::vec2 uv(0.3f, 0.4f);
    const glm::vec2 prev_uv = depth_reproject_prev_uv(glm::inverse(curr_vp), prev_vp, curr_eye, uv, 0.0f);
    EXPECT_NEAR(prev_uv.x, uv.x, 1e-4f);
    EXPECT_NEAR(prev_uv.y, uv.y, 1e-4f);
}

TEST(TaaDepthReproject, SkyFollowsCameraRotation) {
    // A yawing camera pans the sky: the reprojected uv must land where the
    // same view direction appeared last frame.
    const glm::vec3 eye(0.0f, 10.0f, 0.0f);
    const glm::vec3 curr_dir = glm::normalize(glm::vec3(0.1f, 0.0f, -1.0f));
    const glm::mat4 curr_vp = projection() * look_at(eye, curr_dir);
    const glm::mat4 prev_vp = projection() * look_at(eye); // looking straight down -Z

    // Pick the uv of a fixed direction under the current camera, then check
    // the reprojection matches projecting that direction with the previous
    // camera (w = 0: a point at infinity).
    const glm::vec3 sky_dir = glm::normalize(glm::vec3(0.05f, 0.2f, -1.0f));
    const glm::vec4 curr_clip = curr_vp * glm::vec4(sky_dir, 0.0f);
    const glm::vec2 curr_ndc = glm::vec2(curr_clip) / curr_clip.w;
    const glm::vec2 curr_uv(curr_ndc.x * 0.5f + 0.5f, 0.5f - curr_ndc.y * 0.5f);

    const glm::vec4 prev_clip = prev_vp * glm::vec4(sky_dir, 0.0f);
    const glm::vec2 prev_ndc = glm::vec2(prev_clip) / prev_clip.w;
    const glm::vec2 expected(prev_ndc.x * 0.5f + 0.5f, 0.5f - prev_ndc.y * 0.5f);

    const glm::vec2 prev_uv = depth_reproject_prev_uv(glm::inverse(curr_vp), prev_vp, eye, curr_uv, 0.0f);
    EXPECT_NEAR(prev_uv.x, expected.x, 1e-4f);
    EXPECT_NEAR(prev_uv.y, expected.y, 1e-4f);
}

TEST(TaaDepthReproject, BehindPreviousCameraReturnsInputUV) {
    // 180-degree turn: the point is behind last frame's camera (w <= 0) —
    // return the input uv (shader treats it as "no history").
    const glm::vec3 eye(0.0f, 0.0f, 0.0f);
    const glm::mat4 curr_vp = projection() * look_at(eye, {0.0f, 0.0f, -1.0f});
    const glm::mat4 prev_vp = projection() * look_at(eye, {0.0f, 0.0f, 1.0f});

    const glm::vec3 world(0.0f, 0.0f, -50.0f);
    const auto curr = project(curr_vp, world);
    const glm::vec2 prev_uv = depth_reproject_prev_uv(glm::inverse(curr_vp), prev_vp, eye, curr.uv, curr.depth);
    EXPECT_EQ(prev_uv, curr.uv);
}

TEST(Taa, HistoryInvalidBeforeInit) {
    TAA taa;
    EXPECT_FALSE(taa.is_ready());
    EXPECT_FALSE(taa.history_valid());
    taa.invalidate_history(); // safe pre-init
    EXPECT_FALSE(taa.history_valid());
}

TEST(Taa, DefaultSharpenStrengthIsSubtle) {
    TAA taa;
    EXPECT_FLOAT_EQ(taa.sharpen_strength(), TAA::DEFAULT_SHARPEN_STRENGTH);
    EXPECT_FLOAT_EQ(TAA::DEFAULT_SHARPEN_STRENGTH, 0.25f);
    taa.set_sharpen_strength(0.0f);
    EXPECT_FLOAT_EQ(taa.sharpen_strength(), 0.0f);
}
