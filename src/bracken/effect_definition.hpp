#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace bracken {

// Curve types for animating properties over particle lifetime
enum class CurveType {
    CONSTANT,      // Fixed value
    LINEAR,        // Linear interpolation from start to end
    EASE_IN,       // Slow start, accelerate
    EASE_OUT,      // Fast start, decelerate
    EASE_IN_OUT,   // Slow start and end
    FADE_OUT_LATE, // Stay at start value until fade_start, then fade to end
};

struct Curve {
    CurveType type = CurveType::CONSTANT;
    float start_value = 1.0f;
    float end_value = 1.0f;
    float fade_start = 0.8f; // For FADE_OUT_LATE curve type

    // Evaluate curve at time t (0.0 to 1.0)
    float evaluate(float t) const;
};

// Spawn behavior for emitters
enum class SpawnMode {
    BURST,      // Spawn all particles at once
    CONTINUOUS, // Spawn particles over time
};

// Velocity behavior types
enum class VelocityType {
    DIRECTIONAL, // Move in a direction (forward, up, etc.)
    RADIAL,      // Move outward from origin
    ORBITAL,     // Orbit around origin
    CUSTOM,      // Use explicit velocity vector
    ARC,         // Move in an arc (for melee slashes)
};

struct VelocityDefinition {
    VelocityType type = VelocityType::DIRECTIONAL;
    float speed = 100.0f;            // Units per second
    glm::vec3 direction = {1, 0, 0}; // For DIRECTIONAL/CUSTOM
    float spread_angle = 0.0f;       // Random spread in degrees
    glm::vec3 gravity = {0, 0, 0};   // Gravity acceleration
    float drag = 0.0f;               // Velocity damping (0-1)

    // For ORBITAL type
    float orbit_radius = 50.0f;
    float orbit_speed = 2.0f;       // Rotations per second
    float orbit_height_base = 0.0f; // Base height above origin
    float height_variation = 0.0f;  // Vertical oscillation amplitude

    // For ARC type (melee slash)
    float arc_radius = 36.0f;           // Radius of the arc swing
    float arc_height_base = 25.0f;      // Base height above ground
    float arc_height_amplitude = 15.0f; // Height oscillation amplitude
    float arc_tilt_amplitude = 0.8f;    // Tilt angle amplitude (radians)
};

// Rotation behavior
struct RotationDefinition {
    glm::vec3 initial_rotation = {0, 0, 0}; // Initial euler angles (degrees)
    glm::vec3 rotation_rate = {0, 0, 0};    // Degrees per second
    bool face_velocity = false;             // Rotate to face movement direction
};

// Particle appearance
struct AppearanceDefinition {
    Curve scale_over_lifetime;
    Curve opacity_over_lifetime;
    glm::vec4 color_tint = {1, 1, 1, 1};
    glm::vec4 color_end = {1, 1, 1, 1}; // For color interpolation
    bool use_color_gradient = false;
};

// Emitter definition - describes how particles are spawned and behave
struct EmitterDefinition {
    std::string name;

    // Particle model, looked up by name in the ModelManager
    std::string model;

    // Spawn behavior
    SpawnMode spawn_mode = SpawnMode::BURST;
    int spawn_count = 1;
    float spawn_rate = 10.0f; // Particles per second (for CONTINUOUS; must be
                              // > 0 — non-positive rates are rejected with a
                              // warning at spawn and the emitter stays inert)
    glm::vec3 spawn_volume{0.0f}; // Box half-extents around the emitter origin;
                                  // each particle spawns at origin + a uniform
                                  // random offset within the box. {0,0,0}
                                  // (default) = point emitter. Area effects
                                  // (rain, snow, fog banks) set this instead of
                                  // spawning many point effects.

    // Particle lifetime
    float particle_lifetime = 1.0f; // How long each particle lives

    // Behavior
    VelocityDefinition velocity;
    RotationDefinition rotation;
    AppearanceDefinition appearance;

    // Emitter timing
    float delay = 0.0f;     // Delay before emitter starts
    float duration = -1.0f; // How long emitter runs (-1 = use particle_lifetime)
};

// Effect definition - collection of emitters that make up a complete effect
struct EffectDefinition {
    std::string name;
    std::vector<EmitterDefinition> emitters;

    // Effect-level duration override. When >= 0, emitters stop spawning once
    // the effect has been alive this long, and the effect completes (and is
    // compacted away) as soon as its remaining particles die. -1 (default) =
    // emitter-driven: each emitter runs its own duration and the effect
    // completes when every emitter is done and every particle is dead.
    float duration = -1.0f;
};

// Helper functions for curve evaluation
inline float Curve::evaluate(float t) const {
    t = glm::clamp(t, 0.0f, 1.0f);

    switch (type) {
        case CurveType::CONSTANT:
            return start_value;

        case CurveType::LINEAR:
            return glm::mix(start_value, end_value, t);

        case CurveType::EASE_IN:
            return glm::mix(start_value, end_value, t * t);

        case CurveType::EASE_OUT:
            return glm::mix(start_value, end_value, 1.0f - (1.0f - t) * (1.0f - t));

        case CurveType::EASE_IN_OUT: {
            float smoothed = t < 0.5f ? 2.0f * t * t : 1.0f - 2.0f * (1.0f - t) * (1.0f - t);
            return glm::mix(start_value, end_value, smoothed);
        }

        case CurveType::FADE_OUT_LATE:
            if (t < fade_start) {
                return start_value;
            } else {
                float fade_t = (t - fade_start) / (1.0f - fade_start);
                return glm::mix(start_value, end_value, fade_t);
            }
    }

    return start_value;
}

} // namespace bracken
