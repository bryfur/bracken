#pragma once

#include "bracken/systems/camera_controller.hpp"
#include <deque>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace bracken::systems {

// Third-person camera system with smooth following and configuration support

struct CameraShake {
    ShakeType type;
    float intensity;
    float duration;
    float elapsed;
    glm::vec3 direction; // For directional shakes
};

class CameraSystem : public CameraController {
public:
    CameraSystem();

    // Main update - call every frame with delta time
    void update(float dt) override;

    // Set the target to follow (player position in world space)
    void set_target(const glm::vec3& position) override;
    void set_target_velocity(const glm::vec3& velocity) override;

    // Manual camera control (from mouse/controller input)
    void rotate_yaw(float delta_degrees) override;
    void rotate_pitch(float delta_degrees) override;
    void set_yaw(float degrees) override;   // Direct set (no delta)
    void set_pitch(float degrees) override; // Direct set (no delta)
    void adjust_zoom(float delta) override;
    float get_input_yaw() const { return input_yaw_; }
    float get_input_pitch() const { return input_pitch_; }

    // Configuration
    void set_config(const CameraModeConfig& config) override;
    const CameraModeConfig& get_config() const override { return current_config_; }

    // Soft-focus target (e.g. for action-cam soft-lock). Engine biases yaw
    // toward focus_target_ when focus_strength_ > 0. Strength is clamped
    // to [0, 1]; values above 1 would over-rotate and oscillate.
    void set_focus_target(const glm::vec3* target) override { focus_target_ = target; }
    void set_focus_strength(float strength_0_1) override {
        focus_strength_ = glm::clamp(strength_0_1, 0.0f, 1.0f);
    }
    float get_focus_strength() const override { return focus_strength_; }

    // Additive FOV bias (degrees). Caller manages over-time variation;
    // camera smooths the result toward target_fov + fov_bias_.
    void set_fov_bias(float fov_delta) override { fov_bias_ = fov_delta; }
    float get_fov_bias() const override { return fov_bias_; }

    // Camera shake
    void add_shake(ShakeType type, float intensity, float duration = 0.3f) override;
    void add_directional_shake(const glm::vec3& direction, float intensity, float duration = 0.2f) override;

    // Terrain collision callback - returns height at world XZ position
    void set_terrain_height_func(TerrainHeightFunc func) override { get_terrain_height_ = func; }

    // Obstacle/wall collision callback - returns true if line of sight is blocked
    using CollisionCheckFunc = std::function<bool(const glm::vec3& from, const glm::vec3& to, glm::vec3& hit_point)>;
    void set_collision_func(CollisionCheckFunc func) { check_collision_ = func; }

    // Get computed matrices and values
    glm::mat4 get_view_matrix() const override { return view_matrix_; }
    glm::mat4 get_projection_matrix() const override { return projection_matrix_; }
    glm::vec3 get_position() const override { return final_camera_pos_; }
    glm::vec3 get_target_position() const { return smoothed_target_; } // Player/follow target position
    glm::vec3 get_forward() const override { return camera_forward_; }
    glm::vec3 get_right() const { return camera_right_; }
    float get_yaw() const { return current_yaw_; }
    float get_pitch() const { return current_pitch_; }
    float get_current_fov() const { return current_fov_; }
    float get_current_distance() const { return current_distance_; }

    // Set screen dimensions for projection matrix
    void set_screen_size(int width, int height) override;

private:
    // Internal update stages (in call order). Shake is computed before
    // collision avoidance so the clamps apply to the shaken position and
    // large shakes cannot push the camera through terrain/obstacles.
    void update_input_smoothing(float dt);
    void update_look_ahead(float dt);
    void update_auto_centering(float dt);
    void update_soft_lock(float dt);
    void update_camera_position(float dt);
    void update_camera_shake(float dt);
    void update_collision_avoidance(float dt);
    void update_dynamic_fov(float dt);
    void compute_matrices();

    // Smooth damping helper (critically-damped spring)
    static glm::vec3 smooth_damp(const glm::vec3& current, const glm::vec3& target, glm::vec3& velocity,
                                 float smoothTime, float dt);
    static float smooth_damp_angle(float current, float target, float& velocity, float smoothTime, float dt);
    static float smooth_damp_float(float current, float target, float& velocity, float smoothTime, float dt);

    // Current configuration
    CameraModeConfig current_config_;

    // Target tracking
    glm::vec3 target_position_ = glm::vec3(0.0f);
    glm::vec3 target_velocity_ = glm::vec3(0.0f);
    glm::vec3 smoothed_target_ = glm::vec3(0.0f);
    glm::vec3 target_smooth_vel_ = glm::vec3(0.0f);

    // Look-ahead
    glm::vec3 look_ahead_offset_ = glm::vec3(0.0f);
    glm::vec3 look_ahead_vel_ = glm::vec3(0.0f);

    // Camera angles (degrees)
    float input_yaw_ = 0.0f;      // Raw input yaw
    float input_pitch_ = 25.0f;   // Raw input pitch
    float current_yaw_ = 0.0f;    // Smoothed yaw
    float current_pitch_ = 25.0f; // Smoothed pitch
    float yaw_velocity_ = 0.0f;
    float pitch_velocity_ = 0.0f;

    // Auto-centering (delay comes from current_config_.auto_center_delay)
    float time_since_input_ = 0.0f;
    bool auto_centering_active_ = false;

    // Distance/zoom (limits come from current_config_.zoom_min/zoom_max)
    float input_distance_ = 250.0f;
    float current_distance_ = 250.0f;
    float distance_velocity_ = 0.0f;

    // Collision avoidance (ground clearance comes from current_config_)
    float collision_pull_in_speed_ = 15.0f;
    float collision_push_out_speed_ = 5.0f;
    float collision_distance_offset_ = 0.0f;

    // Camera positions
    glm::vec3 ideal_camera_pos_ = glm::vec3(0.0f);
    glm::vec3 current_camera_pos_ = glm::vec3(0.0f);
    glm::vec3 final_camera_pos_ = glm::vec3(0.0f);
    glm::vec3 camera_pos_velocity_ = glm::vec3(0.0f);

    // Look target
    glm::vec3 look_at_target_ = glm::vec3(0.0f);
    glm::vec3 current_look_at_ = glm::vec3(0.0f);
    glm::vec3 look_at_velocity_ = glm::vec3(0.0f);

    // Camera orientation vectors
    glm::vec3 camera_forward_ = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 camera_right_ = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 camera_up_ = glm::vec3(0.0f, 1.0f, 0.0f);

    // Soft-lock smoothing
    float soft_lock_strength_ = 0.0f;
    float soft_lock_blend_speed_ = 3.0f;
    float soft_lock_velocity_ = 0.0f;

    // Dynamic FOV (sprint bonus comes from current_config_)
    float current_fov_ = 55.0f;
    float fov_velocity_ = 0.0f;
    float fov_bias_ = 0.0f;

    // Soft-focus bias (action-cam soft-lock)
    const glm::vec3* focus_target_ = nullptr;
    float focus_strength_ = 0.0f;

    // Camera shake
    std::deque<CameraShake> active_shakes_;
    glm::vec3 shake_offset_ = glm::vec3(0.0f);
    static constexpr size_t MAX_ACTIVE_SHAKES = 8;

    // Screen dimensions
    int screen_width_ = 1280;
    int screen_height_ = 720;

    // Output matrices
    glm::mat4 view_matrix_ = glm::mat4(1.0f);
    glm::mat4 projection_matrix_ = glm::mat4(1.0f);

    // Callbacks
    TerrainHeightFunc get_terrain_height_ = nullptr;
    CollisionCheckFunc check_collision_ = nullptr;

    // Input tracking
    bool had_input_this_frame_ = false;
};

} // namespace bracken::systems
