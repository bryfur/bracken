#pragma once

#include <glm/glm.hpp>

namespace mmo::engine {

/**
 * Single source of truth for the atmosphere: sun, sky gradient, hemisphere
 * ambient, and distance fog. The scene layer pushes one EnvironmentState into
 * each renderer (WorldRenderer / TerrainRenderer / GrassRenderer) once per
 * frame via set_environment().
 *
 * All colors are LINEAR; the scene renders into an RGBA16F target and ACES
 * tonemapping happens at composite. Defaults model a clear summer midday.
 */
struct EnvironmentState {
    glm::vec3 sun_dir{glm::normalize(glm::vec3(-0.5f, -0.8f, -0.3f))}; // direction light TRAVELS
    glm::vec3 sun_color{1.0f, 0.95f, 0.9f};      // linear tint; sun_intensity setting multiplies at use site
    glm::vec3 sky_zenith{0.20f, 0.37f, 0.66f};   // linear; rich midday blue overhead
    glm::vec3 sky_horizon{0.45f, 0.56f, 0.74f};  // linear; pale band at the horizon
    glm::vec3 ground_color{0.16f, 0.14f, 0.11f}; // linear; sky color below the horizon line
    glm::vec3 ambient_sky{0.35f, 0.42f, 0.55f};  // hemisphere upper, pre ambient_strength scale
    glm::vec3 ambient_ground{0.22f, 0.20f, 0.17f}; // hemisphere lower (bounce off terrain)
    glm::vec3 fog_color_horizon{0.45f, 0.56f, 0.74f}; // should track sky_horizon so fog melts into the sky
    glm::vec3 fog_color_sun{0.95f, 0.75f, 0.50f};     // fog tint when looking toward the sun
    float fog_start = 1000.0f;         // world units; fog factor 0 at/below this view distance
    float fog_end = 8000.0f;           // world units; fog factor saturates approaching this
    float fog_height_falloff = 0.002f; // exp falloff on world Y (thinner fog at altitude)
};

} // namespace mmo::engine
