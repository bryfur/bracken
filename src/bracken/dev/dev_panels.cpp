#include "bracken/dev/dev_panels.hpp"

#if defined(BRACKEN_DEV_UI) && BRACKEN_DEV_UI

#include "bracken/graphics_settings.hpp"
#include "bracken/render/environment.hpp"
#include "bracken/render_stats.hpp"
#include "bracken/scene/scene_renderer.hpp"
#include "bracken/systems/camera_controller.hpp"
#include "bracken/systems/effect_system.hpp"
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <glm/glm.hpp>
#include <imgui.h>

namespace bracken::dev::detail {

namespace {

// Colors in EnvironmentState are linear HDR (tonemapped at composite), so the
// pickers must edit raw floats without the 0..1 clamp.
constexpr ImGuiColorEditFlags HDR_COLOR_FLAGS = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR;

/** Combo over an `enum class E : int` with contiguous 0-based values. */
template <typename E>
bool combo_enum(const char* label, E& value, const char* const items[], int count) {
    int index = std::clamp(static_cast<int>(value), 0, count - 1);
    const bool changed = ImGui::Combo(label, &index, items, count);
    if (changed) {
        value = static_cast<E>(index);
    }
    return changed;
}

// ========== Stats ==========

void draw_stats_tab(const DevUI::Bindings& bindings) {
    if (bindings.fps != nullptr) {
        const float fps = *bindings.fps;
        // 120-sample scrolling history. Static is fine: the overlay is
        // main-thread only and drawn at most once per frame.
        static std::array<float, 120> history{};
        static size_t next_sample = 0;
        history[next_sample] = fps;
        next_sample = (next_sample + 1) % history.size();

        char overlay[32];
        (void)std::snprintf(overlay, sizeof(overlay), "%.1f fps", static_cast<double>(fps));
        ImGui::PlotLines("##fps_history", history.data(), static_cast<int>(history.size()),
                         static_cast<int>(next_sample), overlay, 0.0f, FLT_MAX, ImVec2(-1.0f, 64.0f));
        ImGui::Text("Frame time: %.2f ms", fps > 0.0f ? static_cast<double>(1000.0f / fps) : 0.0);
    }

    if (bindings.scene_renderer != nullptr) {
        // The counters are only accumulated while collection is on; keep it
        // enabled while this tab is drawn so the readouts are live (the cost
        // is a handful of CPU-side increments per frame).
        bindings.scene_renderer->set_collect_stats(true);
        const RenderStats& stats = bindings.scene_renderer->render_stats();

        ImGui::SeparatorText("Render");
        ImGui::Text("Draw calls: %u", stats.draw_calls);
        ImGui::Text("Triangles: %u", stats.triangle_count);
        ImGui::Text("Entities rendered: %u", stats.entities_rendered);
        ImGui::Text("Entities culled: %u distance, %u frustum", stats.entities_distance_culled,
                    stats.entities_frustum_culled);
        if (const auto* effects = bindings.scene_renderer->effect_system()) {
            ImGui::Text("Particles: %u (in %u effects)", static_cast<unsigned>(effects->particle_count()),
                        static_cast<unsigned>(effects->effect_count()));
        }
        ImGui::Spacing();
        ImGui::TextDisabled("Per-pass GPU times are stubbed (SDL3 limitation);");
        ImGui::TextDisabled("use Tracy (CPU) or RenderDoc/Nsight (GPU) for real numbers.");
    }
}

// ========== Environment ==========

void draw_environment_tab(scene::SceneRenderer& renderer) {
    // Mutating EnvironmentState in place is the intended live-edit path:
    // render_frame re-propagates env_ to every sub-renderer each frame.
    EnvironmentState& env = renderer.environment();

    ImGui::SeparatorText("Sun");
    // sun_dir is the direction light TRAVELS; edit as azimuth/elevation of
    // the direction TO the sun, re-decomposed each frame so external changes
    // (e.g. a host day/night cycle) keep the sliders in sync.
    glm::vec3 to_sun = -env.sun_dir;
    const float to_sun_len = glm::length(to_sun);
    if (to_sun_len > 1e-5f) {
        to_sun /= to_sun_len;
    } else {
        to_sun = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    float elevation_deg = glm::degrees(std::asin(std::clamp(to_sun.y, -1.0f, 1.0f)));
    float azimuth_deg = glm::degrees(std::atan2(to_sun.z, to_sun.x));
    bool sun_moved = false;
    sun_moved |= ImGui::SliderFloat("Azimuth", &azimuth_deg, -180.0f, 180.0f, "%.1f deg");
    sun_moved |= ImGui::SliderFloat("Elevation", &elevation_deg, -89.0f, 89.0f, "%.1f deg");
    if (sun_moved) {
        const float az = glm::radians(azimuth_deg);
        const float el = glm::radians(elevation_deg);
        const glm::vec3 dir_to_sun{std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)};
        env.sun_dir = -glm::normalize(dir_to_sun);
    }
    ImGui::ColorEdit3("Sun color", &env.sun_color.x, HDR_COLOR_FLAGS);

    ImGui::SeparatorText("Sky & ambient");
    ImGui::ColorEdit3("Sky zenith", &env.sky_zenith.x, HDR_COLOR_FLAGS);
    ImGui::ColorEdit3("Sky horizon", &env.sky_horizon.x, HDR_COLOR_FLAGS);
    ImGui::ColorEdit3("Ground color", &env.ground_color.x, HDR_COLOR_FLAGS);
    ImGui::ColorEdit3("Ambient sky", &env.ambient_sky.x, HDR_COLOR_FLAGS);
    ImGui::ColorEdit3("Ambient ground", &env.ambient_ground.x, HDR_COLOR_FLAGS);

    ImGui::SeparatorText("Distance fog");
    ImGui::ColorEdit3("Fog horizon", &env.fog_color_horizon.x, HDR_COLOR_FLAGS);
    ImGui::ColorEdit3("Fog sun tint", &env.fog_color_sun.x, HDR_COLOR_FLAGS);
    ImGui::DragFloat("Fog start", &env.fog_start, 10.0f, 0.0f, 32000.0f, "%.0f");
    ImGui::DragFloat("Fog end", &env.fog_end, 10.0f, 0.0f, 32000.0f, "%.0f");
    ImGui::DragFloat("Height falloff", &env.fog_height_falloff, 0.0001f, 0.0f, 0.05f, "%.4f");

    ImGui::SeparatorText("Volumetric fog / god rays");
    ImGui::ColorEdit3("Volumetric sun color", &env.fog_volumetric_sun_color.x, HDR_COLOR_FLAGS);
    ImGui::DragFloat("Scatter", &env.fog_volumetric_scatter, 0.005f, 0.0f, 2.0f, "%.3f");
    ImGui::DragFloat("Density", &env.fog_volumetric_density, 0.0001f, 0.0f, 0.02f, "%.5f");
    ImGui::DragFloat("Height", &env.fog_volumetric_height, 1.0f, 1.0f, 2000.0f, "%.0f");
    ImGui::DragFloat("Falloff", &env.fog_volumetric_falloff, 0.0001f, 0.0f, 0.05f, "%.4f");
    ImGui::DragFloat("March start", &env.fog_volumetric_start, 5.0f, 0.0f, 4000.0f, "%.0f");
    ImGui::DragFloat("March end", &env.fog_volumetric_end, 10.0f, 100.0f, 32000.0f, "%.0f");
    ImGui::TextDisabled("Volumetric pass runs only when 'Volumetric fog' is on (Graphics tab).");
}

// ========== Graphics ==========

void draw_graphics_tab(const DevUI::Bindings& bindings) {
    GraphicsSettings& settings = *bindings.settings;
    bool changed = false;

    ImGui::SeparatorText("Modes");
    static const char* const SHADOW_MODES[] = {"Off", "PCF", "PCSS"};
    changed |= combo_enum("Shadows", settings.shadow_mode, SHADOW_MODES, 3);
    static const char* const CASCADE_COUNTS[] = {"1", "2", "3", "4"};
    changed |= ImGui::Combo("Shadow cascades", &settings.shadow_cascades, CASCADE_COUNTS, 4);
    static const char* const SHADOW_RESOLUTIONS[] = {"512", "1024", "2048", "4096"};
    changed |= ImGui::Combo("Shadow resolution", &settings.shadow_resolution, SHADOW_RESOLUTIONS, 4);
    static const char* const AO_MODES[] = {"Off", "SSAO", "GTAO"};
    changed |= combo_enum("Ambient occlusion", settings.ao_mode, AO_MODES, 3);
    static const char* const AA_MODES[] = {"Off", "FXAA"};
    changed |= combo_enum("Anti-aliasing", settings.aa_mode, AA_MODES, 2);
    static const char* const TONEMAP_MODES[] = {"ACES (fitted)", "ACES (Narkowicz)", "Reinhard", "None"};
    changed |= combo_enum("Tonemap", settings.tonemap_mode, TONEMAP_MODES, 4);
    static const char* const ANISO_LEVELS[] = {"Off", "2x", "4x", "8x", "16x"};
    changed |= ImGui::Combo("Anisotropic filter", &settings.anisotropic_filter, ANISO_LEVELS, 5);
    static const char* const VSYNC_MODES[] = {"Immediate", "VSync", "Mailbox"};
    changed |= ImGui::Combo("VSync", &settings.vsync_mode, VSYNC_MODES, 3);
    static const char* const WINDOW_MODES[] = {"Windowed", "Borderless fullscreen", "Exclusive fullscreen"};
    changed |= ImGui::Combo("Window mode", &settings.window_mode, WINDOW_MODES, 3);

    ImGui::SeparatorText("Features");
    changed |= ImGui::Checkbox("Bloom", &settings.bloom_enabled);
    changed |= ImGui::Checkbox("Volumetric fog", &settings.volumetric_fog);
    changed |= ImGui::Checkbox("God rays", &settings.god_rays);
    changed |= ImGui::Checkbox("Distance fog", &settings.fog_enabled);
    changed |= ImGui::Checkbox("Grass", &settings.grass_enabled);
    changed |= ImGui::Checkbox("Skybox", &settings.skybox_enabled);
    changed |= ImGui::Checkbox("Trees", &settings.trees_enabled);
    changed |= ImGui::Checkbox("Rocks", &settings.rocks_enabled);
    changed |= ImGui::Checkbox("Frustum culling", &settings.frustum_culling);
    changed |= ImGui::Checkbox("Show FPS", &settings.show_fps);
    changed |= ImGui::Checkbox("Debug HUD", &settings.show_debug_hud);

    ImGui::SeparatorText("Culling");
    static const char* const DRAW_DISTANCES[] = {"500", "1000", "2000", "4000", "8000", "16000", "32000"};
    changed |= ImGui::Combo("Draw distance", &settings.draw_distance, DRAW_DISTANCES, 7);

    ImGui::SeparatorText("Tone & grading");
    changed |= ImGui::SliderFloat("Exposure", &settings.exposure, 0.1f, 4.0f, "%.2f");
    changed |= ImGui::SliderFloat("Contrast", &settings.contrast, 0.5f, 2.0f, "%.2f");
    changed |= ImGui::SliderFloat("Saturation", &settings.saturation, 0.0f, 2.0f, "%.2f");
    changed |= ImGui::SliderFloat("Ambient strength", &settings.ambient_strength, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("Sun intensity", &settings.sun_intensity, 0.0f, 10.0f, "%.2f");

    ImGui::SeparatorText("Bloom / AO / fog");
    changed |= ImGui::SliderFloat("Bloom strength", &settings.bloom_strength, 0.0f, 1.0f, "%.2f");
    changed |= ImGui::SliderFloat("Bloom threshold", &settings.bloom_threshold, 0.0f, 4.0f, "%.2f");
    changed |= ImGui::SliderFloat("AO strength", &settings.ao_strength, 0.0f, 2.0f, "%.2f");
    changed |= ImGui::SliderFloat("AO radius", &settings.ao_radius, 0.1f, 4.0f, "%.2f");
    changed |= ImGui::SliderFloat("Fog density", &settings.fog_density, 0.0f, 1.0f, "%.3f");

    // Push edits through the host's apply path only on frames where a widget
    // actually changed something (the host diffs internally, but there is no
    // reason to spam it every frame).
    if (changed && bindings.apply_settings) {
        bindings.apply_settings(settings);
    }
}

// ========== Camera ==========

void draw_camera_tab(systems::CameraController& camera) {
    const glm::vec3 position = camera.get_position();
    const glm::vec3 forward = camera.get_forward();
    // The abstract controller exposes no yaw/pitch getters; derive both from
    // the forward vector (yaw convention matches camera_system.cpp).
    const float pitch_deg = glm::degrees(std::asin(std::clamp(forward.y, -1.0f, 1.0f)));
    const float yaw_deg = glm::degrees(std::atan2(-forward.x, -forward.z));
    const systems::CameraModeConfig& config = camera.get_config();

    ImGui::SeparatorText("Readout");
    ImGui::Text("Position: %.1f, %.1f, %.1f", static_cast<double>(position.x), static_cast<double>(position.y),
                static_cast<double>(position.z));
    ImGui::Text("Forward: %.2f, %.2f, %.2f", static_cast<double>(forward.x), static_cast<double>(forward.y),
                static_cast<double>(forward.z));
    ImGui::Text("Yaw: %.1f deg  Pitch: %.1f deg", static_cast<double>(yaw_deg), static_cast<double>(pitch_deg));
    ImGui::SameLine();
    ImGui::TextDisabled("(derived from forward)");
    ImGui::Text("Distance: %.0f  FOV: %.1f deg", static_cast<double>(config.distance),
                static_cast<double>(config.fov));

    ImGui::SeparatorText("Shake test");
    static float shake_intensity = 1.0f;
    static float shake_duration = 0.3f;
    ImGui::SliderFloat("Intensity", &shake_intensity, 0.0f, 3.0f, "%.2f");
    ImGui::SliderFloat("Duration", &shake_duration, 0.05f, 2.0f, "%.2f s");
    if (ImGui::Button("Impact")) {
        camera.add_shake(systems::ShakeType::Impact, shake_intensity, shake_duration);
    }
    ImGui::SameLine();
    if (ImGui::Button("Heavy")) {
        camera.add_shake(systems::ShakeType::Heavy, shake_intensity, shake_duration);
    }
    ImGui::SameLine();
    if (ImGui::Button("Subtle")) {
        camera.add_shake(systems::ShakeType::Subtle, shake_intensity, shake_duration);
    }
    ImGui::SameLine();
    if (ImGui::Button("Directional (fwd)")) {
        camera.add_directional_shake(forward, shake_intensity, shake_duration);
    }

    ImGui::SeparatorText("Framing");
    static float fov_bias = 0.0f;
    if (ImGui::SliderFloat("FOV bias", &fov_bias, -30.0f, 30.0f, "%.1f deg")) {
        camera.set_fov_bias(fov_bias);
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        fov_bias = 0.0f;
        camera.set_fov_bias(0.0f);
    }
    static float focus_strength = 0.0f;
    if (ImGui::SliderFloat("Focus strength", &focus_strength, 0.0f, 1.0f, "%.2f")) {
        camera.set_focus_strength(focus_strength);
    }
}

} // namespace

void draw_builtin_panels(const DevUI::Bindings& bindings) {
    ImGui::SetNextWindowSize(ImVec2(420.0f, 540.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Bracken Dev")) {
        ImGui::End();
        return;
    }
    if (ImGui::BeginTabBar("##bracken_dev_tabs")) {
        if ((bindings.fps != nullptr || bindings.scene_renderer != nullptr) && ImGui::BeginTabItem("Stats")) {
            ImGui::PushID("stats");
            draw_stats_tab(bindings);
            ImGui::PopID();
            ImGui::EndTabItem();
        }
        if (bindings.scene_renderer != nullptr && ImGui::BeginTabItem("Environment")) {
            ImGui::PushID("environment");
            draw_environment_tab(*bindings.scene_renderer);
            ImGui::PopID();
            ImGui::EndTabItem();
        }
        if (bindings.settings != nullptr && ImGui::BeginTabItem("Graphics")) {
            ImGui::PushID("graphics");
            draw_graphics_tab(bindings);
            ImGui::PopID();
            ImGui::EndTabItem();
        }
        if (bindings.camera != nullptr && ImGui::BeginTabItem("Camera")) {
            ImGui::PushID("camera");
            draw_camera_tab(*bindings.camera);
            ImGui::PopID();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

} // namespace bracken::dev::detail

#else // stub configuration: BRACKEN_DEV_UI not defined

namespace bracken::dev::detail {

void draw_builtin_panels(const DevUI::Bindings& /*bindings*/) {}

} // namespace bracken::dev::detail

#endif // BRACKEN_DEV_UI
