#pragma once

#include <glm/glm.hpp>

namespace bracken {

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

    // Volumetric fog / god rays (render/volumetric_fog.hpp, half-res ray march).
    // The ambient fog tint reuses fog_color_horizon; sun direction reuses
    // sun_dir. Defaults reproduce the pass's original built-in look.
    glm::vec3 fog_volumetric_sun_color{1.0f, 0.9f, 0.7f}; // linear god-ray inscatter color; sun_intensity is NOT
                                                          // applied — day/night hosts should track their sun tint here
    float fog_volumetric_scatter = 0.25f;   // inscatter strength along the forward-scatter phase lobe
    float fog_volumetric_density = 0.0015f; // base extinction per world unit at ground level (y = 0)
    float fog_volumetric_height = 100.0f;   // height scale of the fog volume (world Y units)
    float fog_volumetric_falloff = 0.005f;  // exponential falloff shaping with height
    float fog_volumetric_start = 100.0f;    // near dead zone: the ray march starts at this camera distance
    float fog_volumetric_end = 3000.0f;     // ray-march distance cap (world units)
};

} // namespace bracken
