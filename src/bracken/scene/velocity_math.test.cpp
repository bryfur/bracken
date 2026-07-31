#include "bracken/scene/velocity_math.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

using namespace bracken::scene;

namespace {

// Engine-convention reversed-Z RH_ZO projection (near/far swapped).
glm::mat4 projection() {
    return glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 15000.0f, 5.0f);
}

glm::mat4 look_at(const glm::vec3& eye) {
    return glm::lookAt(eye, eye + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::vec2 ndc_of(const glm::mat4& vp, const glm::mat4& model, const glm::vec3& p) {
    const glm::vec4 clip = vp * model * glm::vec4(p, 1.0f);
    return glm::vec2(clip) / clip.w;
}

} // namespace

TEST(VelocityMath, StaticObjectStaticCameraIsZero) {
    const glm::mat4 vp = projection() * look_at({0.0f, 0.0f, 10.0f});
    const glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, -5.0f));
    const glm::vec2 v = compute_uv_velocity(vp, vp, model, model, glm::vec3(0.3f, 0.1f, 0.0f));
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
}

TEST(VelocityMath, ObjectMovingRightHasPositiveUVelocity) {
    // Camera at origin looking down -Z; object moved +X since last frame.
    const glm::mat4 vp = projection() * look_at({0.0f, 0.0f, 0.0f});
    const glm::mat4 prev_model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -50.0f));
    const glm::mat4 curr_model = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, -50.0f));

    const glm::vec2 v = compute_uv_velocity(vp, vp, curr_model, prev_model, glm::vec3(0.0f));
    EXPECT_GT(v.x, 0.0f) << "moving right on screen => u increases => +u velocity";
    EXPECT_FLOAT_EQ(v.y, 0.0f);

    // current-minus-previous in UV units: exactly half the NDC delta.
    const float expected_u = (ndc_of(vp, curr_model, {0, 0, 0}).x - ndc_of(vp, prev_model, {0, 0, 0}).x) * 0.5f;
    EXPECT_FLOAT_EQ(v.x, expected_u);
}

TEST(VelocityMath, ObjectMovingUpHasNegativeVVelocity) {
    // +Y in world/NDC is *up*, but UV v grows downward: moving up => v decreases.
    const glm::mat4 vp = projection() * look_at({0.0f, 0.0f, 0.0f});
    const glm::mat4 prev_model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -50.0f));
    const glm::mat4 curr_model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 3.0f, -50.0f));

    const glm::vec2 v = compute_uv_velocity(vp, vp, curr_model, prev_model, glm::vec3(0.0f));
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_LT(v.y, 0.0f);
}

TEST(VelocityMath, StaticObjectMovingCameraGetsCameraVelocity) {
    // Statics pass prev_model == curr_model; a strafing camera still produces
    // screen-space velocity (this is what the prepass writes for all statics).
    const glm::mat4 model(1.0f);
    const glm::mat4 curr_vp = projection() * look_at({1.0f, 0.0f, 20.0f});
    const glm::mat4 prev_vp = projection() * look_at({0.0f, 0.0f, 20.0f});

    const glm::vec2 v = compute_uv_velocity(curr_vp, prev_vp, model, model, glm::vec3(0.0f));
    // Camera moved right => object appears to move left => u decreases.
    EXPECT_LT(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
}

TEST(VelocityMath, TAAReprojectionRecoversPreviousUV) {
    // The documented consumer contract: prev_uv = uv - velocity.
    const glm::mat4 curr_vp = projection() * look_at({0.5f, -0.25f, 30.0f});
    const glm::mat4 prev_vp = projection() * look_at({0.0f, 0.0f, 31.0f});
    const glm::mat4 prev_model = glm::translate(glm::mat4(1.0f), glm::vec3(-1.0f, 0.5f, -40.0f));
    const glm::mat4 curr_model = glm::translate(glm::mat4(1.0f), glm::vec3(-0.5f, 0.75f, -41.0f));
    const glm::vec3 local(0.2f, -0.3f, 0.4f);

    auto to_uv = [](glm::vec2 ndc) { return glm::vec2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f); };
    const glm::vec2 curr_uv = to_uv(ndc_of(curr_vp, curr_model, local));
    const glm::vec2 prev_uv = to_uv(ndc_of(prev_vp, prev_model, local));

    const glm::vec2 v = compute_uv_velocity(curr_vp, prev_vp, curr_model, prev_model, local);
    EXPECT_NEAR(curr_uv.x - v.x, prev_uv.x, 1e-6f);
    EXPECT_NEAR(curr_uv.y - v.y, prev_uv.y, 1e-6f);
}

TEST(VelocityMath, BehindCameraReturnsZero) {
    const glm::mat4 vp = projection() * look_at({0.0f, 0.0f, 0.0f});
    const glm::mat4 behind = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 50.0f)); // +Z = behind
    const glm::mat4 in_front = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -50.0f));
    EXPECT_EQ(compute_uv_velocity(vp, vp, behind, in_front, glm::vec3(0.0f)), glm::vec2(0.0f));
    EXPECT_EQ(compute_uv_velocity(vp, vp, in_front, behind, glm::vec3(0.0f)), glm::vec2(0.0f));
}
