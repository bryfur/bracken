#include "bracken/systems/camera_system.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace bracken::systems;

namespace {

constexpr float kDt = 1.0f / 60.0f;
constexpr int kConvergeSteps = 600; // 10 seconds at 60 Hz

// Mirrors the spherical-offset math in CameraSystem::update_camera_position
// for a fully converged camera (zero velocity, no terrain, no shake).
// Note: the look-at height is terrain_y + height_offset; the target's own y
// is intentionally ignored by the implementation.
glm::vec3 expected_camera_pos(const glm::vec3& target, const CameraModeConfig& config, float yaw_deg,
                              float pitch_deg) {
    glm::vec3 look_at(target.x, config.height_offset, target.z);
    float yaw = glm::radians(yaw_deg);
    float pitch = glm::radians(pitch_deg);
    float horizontal = config.distance * std::cos(pitch);
    float vertical = config.distance * std::sin(pitch);
    float off_x = std::sin(yaw) * horizontal + std::cos(yaw) * config.shoulder_offset;
    float off_z = std::cos(yaw) * horizontal - std::sin(yaw) * config.shoulder_offset;
    return glm::vec3(look_at.x + off_x, look_at.y + vertical, look_at.z + off_z);
}

} // namespace

class CameraSystemTest : public ::testing::Test {
protected:
    CameraSystem camera;

    void step(int n) {
        for (int i = 0; i < n; ++i) {
            camera.update(kDt);
        }
    }
};

// ===== Target following =====

TEST_F(CameraSystemTest, UpdateConvergesSmoothedTargetToSetTarget) {
    glm::vec3 target(100.0f, 0.0f, 200.0f);
    camera.set_target(target);
    step(kConvergeSteps);

    glm::vec3 followed = camera.get_target_position();
    EXPECT_NEAR(followed.x, target.x, 0.1f);
    EXPECT_NEAR(followed.y, target.y, 0.1f);
    EXPECT_NEAR(followed.z, target.z, 0.1f);
}

TEST_F(CameraSystemTest, CameraPositionConvergesToSphericalOffsetFromTarget) {
    glm::vec3 target(100.0f, 0.0f, 200.0f);
    camera.set_target(target);
    step(kConvergeSteps);

    // Default orientation: yaw 0, pitch 25 (initial values in camera_system.hpp).
    glm::vec3 expected = expected_camera_pos(target, camera.get_config(), camera.get_yaw(), camera.get_pitch());
    glm::vec3 pos = camera.get_position();
    EXPECT_NEAR(pos.x, expected.x, 0.5f);
    EXPECT_NEAR(pos.y, expected.y, 0.5f);
    EXPECT_NEAR(pos.z, expected.z, 0.5f);
}

// ===== Look-ahead =====

TEST_F(CameraSystemTest, TargetVelocityLookAheadShiftsCameraForward) {
    CameraModeConfig config;
    config.auto_center_enabled = false; // keep yaw fixed while the target moves
    camera.set_config(config);
    camera.set_target(glm::vec3(0.0f));
    step(kConvergeSteps);
    glm::vec3 baseline = camera.get_position();

    // Fast +X velocity: look-ahead factor saturates, so the offset settles at
    // exactly config.look_ahead_dist along the velocity direction.
    camera.set_target_velocity(glm::vec3(300.0f, 0.0f, 0.0f));
    step(kConvergeSteps);
    glm::vec3 shifted = camera.get_position();

    EXPECT_NEAR(shifted.x - baseline.x, config.look_ahead_dist, 1.0f);
    EXPECT_NEAR(shifted.y, baseline.y, 1.0f);
    EXPECT_NEAR(shifted.z, baseline.z, 1.0f);
}

// ===== Camera shake =====

TEST_F(CameraSystemTest, AddShakePerturbsCameraPosition) {
    camera.set_target(glm::vec3(0.0f));
    step(kConvergeSteps);
    glm::vec3 baseline = camera.get_position();

    camera.add_shake(ShakeType::Impact, 20.0f, 0.3f);
    camera.update(kDt);
    EXPECT_GT(glm::distance(camera.get_position(), baseline), 0.5f);
}

TEST_F(CameraSystemTest, ShakeDecaysToZeroAfterDuration) {
    camera.set_target(glm::vec3(0.0f));
    step(kConvergeSteps);
    glm::vec3 baseline = camera.get_position();

    camera.add_shake(ShakeType::Impact, 20.0f, 0.3f);
    step(60); // 1 second, well past the 0.3 s duration

    glm::vec3 pos = camera.get_position();
    EXPECT_NEAR(pos.x, baseline.x, 0.01f);
    EXPECT_NEAR(pos.y, baseline.y, 0.01f);
    EXPECT_NEAR(pos.z, baseline.z, 0.01f);
}

TEST_F(CameraSystemTest, DirectionalShakePerturbsAlongDirectionOnly) {
    camera.set_target(glm::vec3(0.0f));
    step(kConvergeSteps);
    glm::vec3 baseline = camera.get_position();

    camera.add_directional_shake(glm::vec3(0.0f, 1.0f, 0.0f), 20.0f, 0.2f);
    camera.update(kDt);

    glm::vec3 pos = camera.get_position();
    EXPECT_NEAR(pos.x, baseline.x, 0.01f);
    EXPECT_NEAR(pos.z, baseline.z, 0.01f);
    EXPECT_GT(std::abs(pos.y - baseline.y), 0.5f);
}

TEST_F(CameraSystemTest, DirectionalShakeDecaysToZeroAfterDuration) {
    camera.set_target(glm::vec3(0.0f));
    step(kConvergeSteps);
    glm::vec3 baseline = camera.get_position();

    camera.add_directional_shake(glm::vec3(0.0f, 1.0f, 0.0f), 20.0f, 0.2f);
    step(60); // 1 second, well past the 0.2 s duration

    glm::vec3 pos = camera.get_position();
    EXPECT_NEAR(pos.x, baseline.x, 0.01f);
    EXPECT_NEAR(pos.y, baseline.y, 0.01f);
    EXPECT_NEAR(pos.z, baseline.z, 0.01f);
}

// ===== Zoom =====

TEST_F(CameraSystemTest, AdjustZoomClampsToMinDistance) {
    // Zoom limits are the CameraSystem defaults (min 80, max 600), not part of
    // CameraModeConfig.
    camera.adjust_zoom(-100000.0f);
    step(kConvergeSteps);
    EXPECT_NEAR(camera.get_current_distance(), 80.0f, 0.1f);
}

TEST_F(CameraSystemTest, AdjustZoomClampsToMaxDistance) {
    camera.adjust_zoom(100000.0f);
    step(kConvergeSteps);
    EXPECT_NEAR(camera.get_current_distance(), 600.0f, 0.1f);
}

// ===== Pitch clamping =====

TEST_F(CameraSystemTest, SetPitchClampsToConfigLimits) {
    camera.set_pitch(500.0f);
    EXPECT_FLOAT_EQ(camera.get_input_pitch(), camera.get_config().pitch_max);

    camera.set_pitch(-500.0f);
    EXPECT_FLOAT_EQ(camera.get_input_pitch(), camera.get_config().pitch_min);
}

TEST_F(CameraSystemTest, RotatePitchClampsToConfigLimits) {
    camera.rotate_pitch(500.0f);
    EXPECT_FLOAT_EQ(camera.get_input_pitch(), camera.get_config().pitch_max);

    camera.rotate_pitch(-1000.0f);
    EXPECT_FLOAT_EQ(camera.get_input_pitch(), camera.get_config().pitch_min);
}

TEST_F(CameraSystemTest, SmoothedPitchConvergesToClampedInput) {
    camera.set_pitch(-500.0f);
    step(kConvergeSteps);
    EXPECT_NEAR(camera.get_pitch(), camera.get_config().pitch_min, 0.1f);
}

// ===== Terrain collision =====

TEST_F(CameraSystemTest, TerrainHeightCallbackKeepsCameraAboveGround) {
    constexpr float kGroundHeight = 500.0f;
    camera.set_terrain_height_func([](float, float) { return kGroundHeight; });
    camera.set_target(glm::vec3(0.0f));
    camera.set_pitch(-60.0f); // aim the camera arm well below the terrain plane
    step(kConvergeSteps);
    EXPECT_GE(camera.get_position().y, kGroundHeight);

    // Sanity: without the callback the same setup settles far below ground.
    CameraSystem uncollided;
    uncollided.set_target(glm::vec3(0.0f));
    uncollided.set_pitch(-60.0f);
    for (int i = 0; i < kConvergeSteps; ++i) {
        uncollided.update(kDt);
    }
    EXPECT_LT(uncollided.get_position().y, kGroundHeight);
}

// ===== Screen size / projection =====

TEST_F(CameraSystemTest, SetScreenSizeChangesProjectionAspect) {
    // glm::perspective: m[1][1] / m[0][0] == aspect ratio.
    camera.set_screen_size(1920, 1080);
    camera.update(kDt);
    glm::mat4 proj = camera.get_projection_matrix();
    EXPECT_NEAR(proj[1][1] / proj[0][0], 1920.0f / 1080.0f, 1e-4f);

    camera.set_screen_size(720, 720);
    camera.update(kDt);
    proj = camera.get_projection_matrix();
    EXPECT_NEAR(proj[1][1] / proj[0][0], 1.0f, 1e-4f);
}

TEST_F(CameraSystemTest, SetScreenSizeClampsDegenerateDimensions) {
    camera.set_screen_size(0, -5); // clamped to 1x1 -> aspect 1, no NaNs
    camera.update(kDt);
    glm::mat4 proj = camera.get_projection_matrix();
    EXPECT_FALSE(std::isnan(proj[0][0]));
    EXPECT_NEAR(proj[1][1] / proj[0][0], 1.0f, 1e-4f);
}

// ===== Forward vector =====

TEST_F(CameraSystemTest, ForwardIsUnitLength) {
    camera.set_target(glm::vec3(50.0f, 0.0f, -30.0f));
    step(kConvergeSteps);
    EXPECT_NEAR(glm::length(camera.get_forward()), 1.0f, 1e-4f);
}

TEST_F(CameraSystemTest, ForwardIsConsistentWithYaw) {
    CameraModeConfig config;
    config.shoulder_offset = 0.0f; // keep the camera directly behind the target
    camera.set_config(config);
    camera.set_target(glm::vec3(0.0f));
    camera.set_pitch(0.0f);

    camera.set_yaw(0.0f);
    step(kConvergeSteps);
    glm::vec3 fwd = camera.get_forward();
    EXPECT_NEAR(fwd.x, 0.0f, 0.01f);
    EXPECT_NEAR(fwd.y, 0.0f, 0.01f);
    EXPECT_NEAR(fwd.z, -1.0f, 0.01f);

    camera.set_yaw(90.0f);
    step(kConvergeSteps);
    fwd = camera.get_forward();
    EXPECT_NEAR(fwd.x, -1.0f, 0.01f);
    EXPECT_NEAR(fwd.y, 0.0f, 0.01f);
    EXPECT_NEAR(fwd.z, 0.0f, 0.01f);
}

// ===== Yaw normalization =====

TEST_F(CameraSystemTest, SetYawWrapsIntoZeroTo360) {
    camera.set_yaw(450.0f);
    EXPECT_FLOAT_EQ(camera.get_input_yaw(), 90.0f);

    camera.set_yaw(-30.0f);
    EXPECT_FLOAT_EQ(camera.get_input_yaw(), 330.0f);
}

// ===== Configuration =====

TEST_F(CameraSystemTest, SetConfigUpdatesDistance) {
    CameraModeConfig config;
    config.distance = 400.0f;
    camera.set_config(config);
    EXPECT_FLOAT_EQ(camera.get_config().distance, 400.0f);

    step(kConvergeSteps);
    EXPECT_NEAR(camera.get_current_distance(), 400.0f, 0.1f);
}

// ===== Dynamic FOV =====

TEST_F(CameraSystemTest, FovBiasShiftsSmoothedFov) {
    camera.set_fov_bias(10.0f);
    step(kConvergeSteps);
    EXPECT_NEAR(camera.get_current_fov(), camera.get_config().fov + 10.0f, 0.1f);
}

// ===== View matrix =====

TEST_F(CameraSystemTest, ViewMatrixMapsCameraPositionToOrigin) {
    camera.set_target(glm::vec3(25.0f, 0.0f, -40.0f));
    step(kConvergeSteps);

    glm::vec4 eye = camera.get_view_matrix() * glm::vec4(camera.get_position(), 1.0f);
    EXPECT_NEAR(eye.x, 0.0f, 1e-3f);
    EXPECT_NEAR(eye.y, 0.0f, 1e-3f);
    EXPECT_NEAR(eye.z, 0.0f, 1e-3f);
}
