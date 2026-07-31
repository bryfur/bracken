#pragma once

#include <glm/glm.hpp>

namespace bracken::render {

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

    // Procedural sky clouds + horizon haze (sky_common.hlsli — shared by the
    // skybox pass AND the IBL sky capture, so edits relight the whole scene).
    // cloud_coverage / cloud_density / cloud_scale / haze_strength dirty the
    // amortized IBL refresh (IblSkyParams); cloud_speed only scales the
    // sky-time (scroll phase) accumulation in the sim stage's sky clock, so like
    // the scroll itself it NEVER triggers a refresh — reflected clouds stay
    // frozen between refreshes. Defaults reproduce the classic look.
    float cloud_coverage = 0.35f; // sky fraction reading as cloud; 0 = clear, ~1 = overcast
    float cloud_density = 1.0f;   // opacity/edge weight; <1 wispy sheets, >1 heavy solid decks
    float cloud_scale = 1.0f;     // noise footprint multiplier; <1 = larger formations
    float cloud_speed = 1.0f;     // scroll rate multiplier (live-safe: phase accumulates, no jump)
    float haze_strength = 1.0f;   // horizon haze amount; haze color comes from the fog colors

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

    // Color grading (composite pass). Part of the environment — grading is a
    // LOOK, live-editable like the sky/fog colors (DevUI Environment tab),
    // not a persisted quality setting; GraphicsSettings keeps only the
    // display-side trims (exposure bias / contrast / saturation / tonemap).
    // All-neutral defaults render pixel-identical to no grading at all.
    float grade_temperature = 0.0f;  // white balance, -1 (cool/blue) .. +1 (warm/orange); pre-tonemap
    float grade_tint = 0.0f;         // white balance, -1 (green) .. +1 (magenta); pre-tonemap
    glm::vec3 grade_slope{1.0f};     // ASC CDL slope (gain), per channel; post-tonemap
    glm::vec3 grade_offset{0.0f};    // ASC CDL offset (lift), per channel; post-tonemap
    glm::vec3 grade_power{1.0f};     // ASC CDL power (gamma), per channel; post-tonemap
};

} // namespace bracken::render
