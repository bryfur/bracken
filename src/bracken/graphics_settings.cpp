#include "bracken/graphics_settings.hpp"

#include <fstream>
#include <unordered_map>

namespace bracken {

namespace {

// Clamp helper for the enum modes: clamps the raw integer to the enum's
// inclusive [lo, hi] underlying range.
template<typename E> E clamp_enum(E value, E lo, E hi) {
    return static_cast<E>(std::clamp(static_cast<int>(value), static_cast<int>(lo), static_cast<int>(hi)));
}

} // namespace

void GraphicsSettings::clamp_to_valid_ranges() {
    draw_distance = std::clamp(draw_distance, 0, 6);
    anisotropic_filter = std::clamp(anisotropic_filter, 0, 4);
    vsync_mode = std::clamp(vsync_mode, 0, 2);
    shadow_mode = clamp_enum(shadow_mode, ShadowMode::Off, ShadowMode::PCSS);
    shadow_cascades = std::clamp(shadow_cascades, 0, 3);
    shadow_resolution = std::clamp(shadow_resolution, 0, 3);
    ao_mode = clamp_enum(ao_mode, AOMode::Off, AOMode::GTAO);
    aa_mode = clamp_enum(aa_mode, AAMode::Off, AAMode::FXAA);
    window_mode = std::clamp(window_mode, 0, 2);
    // Upper bound depends on the runtime display-mode list; RenderContext
    // re-clamps at apply time.
    resolution_index = std::max(resolution_index, 0);
    tonemap_mode = clamp_enum(tonemap_mode, TonemapMode::ACESFitted, TonemapMode::None);
}

bool GraphicsSettings::save(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) {
        return false;
    }

    // Always written so load() can tell new files from pre-versioning ones.
    out << "settings_version=" << CURRENT_SETTINGS_VERSION << "\n";
    out << "fog_enabled=" << fog_enabled << "\n";
    out << "grass_enabled=" << grass_enabled << "\n";
    out << "skybox_enabled=" << skybox_enabled << "\n";
    out << "trees_enabled=" << trees_enabled << "\n";
    out << "rocks_enabled=" << rocks_enabled << "\n";
    out << "show_fps=" << show_fps << "\n";
    out << "show_debug_hud=" << show_debug_hud << "\n";
    out << "draw_distance=" << draw_distance << "\n";
    out << "frustum_culling=" << frustum_culling << "\n";
    out << "anisotropic_filter=" << anisotropic_filter << "\n";
    out << "vsync_mode=" << vsync_mode << "\n";
    out << "shadow_mode=" << static_cast<int>(shadow_mode) << "\n";
    out << "shadow_cascades=" << shadow_cascades << "\n";
    out << "shadow_resolution=" << shadow_resolution << "\n";
    out << "ao_mode=" << static_cast<int>(ao_mode) << "\n";
    out << "aa_mode=" << static_cast<int>(aa_mode) << "\n";
    out << "bloom_enabled=" << bloom_enabled << "\n";
    out << "bloom_strength=" << bloom_strength << "\n";
    out << "volumetric_fog=" << volumetric_fog << "\n";
    out << "god_rays=" << god_rays << "\n";
    out << "window_mode=" << window_mode << "\n";
    out << "resolution_index=" << resolution_index << "\n";
    out << "exposure=" << exposure << "\n";
    out << "tonemap_mode=" << static_cast<int>(tonemap_mode) << "\n";
    out << "contrast=" << contrast << "\n";
    out << "saturation=" << saturation << "\n";
    out << "ambient_strength=" << ambient_strength << "\n";
    out << "sun_intensity=" << sun_intensity << "\n";
    out << "bloom_threshold=" << bloom_threshold << "\n";
    out << "ao_strength=" << ao_strength << "\n";
    out << "ao_radius=" << ao_radius << "\n";
    out << "fog_density=" << fog_density << "\n";

    return out.good();
}

bool GraphicsSettings::load(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }

    std::unordered_map<std::string, std::string> kvs;
    std::string line;
    while (std::getline(in, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        kvs[line.substr(0, eq)] = line.substr(eq + 1);
    }

    auto read_bool = [&](const char* key, bool& val) {
        auto it = kvs.find(key);
        if (it != kvs.end()) {
            val = (it->second != "0");
        }
    };
    auto read_int = [&](const char* key, int& val) {
        auto it = kvs.find(key);
        if (it != kvs.end()) {
            try {
                val = std::stoi(it->second);
            } catch (...) { // NOLINT(bugprone-empty-catch): keep defaults on malformed value
            }
        }
    };
    auto read_enum = [&]<typename E>(const char* key, E& val) {
        int raw = static_cast<int>(val);
        read_int(key, raw);
        val = static_cast<E>(raw); // normalized by clamp_to_valid_ranges() below
    };
    auto read_float = [&](const char* key, float& val) {
        auto it = kvs.find(key);
        if (it != kvs.end()) {
            try {
                val = std::stof(it->second);
            } catch (...) { // NOLINT(bugprone-empty-catch): keep defaults on malformed value
            }
        }
    };

    read_bool("fog_enabled", fog_enabled);
    read_bool("grass_enabled", grass_enabled);
    read_bool("skybox_enabled", skybox_enabled);
    read_bool("trees_enabled", trees_enabled);
    read_bool("rocks_enabled", rocks_enabled);
    read_bool("show_fps", show_fps);
    read_bool("show_debug_hud", show_debug_hud);
    read_int("draw_distance", draw_distance);
    read_bool("frustum_culling", frustum_culling);
    read_int("anisotropic_filter", anisotropic_filter);
    read_int("vsync_mode", vsync_mode);
    read_enum("shadow_mode", shadow_mode);
    read_int("shadow_cascades", shadow_cascades);
    read_int("shadow_resolution", shadow_resolution);
    read_enum("ao_mode", ao_mode);
    read_enum("aa_mode", aa_mode);
    read_bool("bloom_enabled", bloom_enabled);
    read_float("bloom_strength", bloom_strength);
    read_bool("volumetric_fog", volumetric_fog);
    read_bool("god_rays", god_rays);
    read_int("window_mode", window_mode);
    read_int("resolution_index", resolution_index);
    read_float("exposure", exposure);
    read_enum("tonemap_mode", tonemap_mode);
    read_float("contrast", contrast);
    read_float("saturation", saturation);
    read_float("ambient_strength", ambient_strength);
    read_float("sun_intensity", sun_intensity);
    read_float("bloom_threshold", bloom_threshold);
    read_float("ao_strength", ao_strength);
    read_float("ao_radius", ao_radius);
    read_float("fog_density", fog_density);

    // Migration: a settings file WITHOUT settings_version was authored
    // against the old lighting model (sun 1.0 / ambient 0.5 / flat AO /
    // bloom tuned for LDR-ish output). Those values look wrong under the
    // wave-3 lighting, so reset the lighting-related knobs to the new
    // defaults; user prefs unrelated to lighting (draw distance, toggles,
    // window mode, ...) are preserved.
    int file_version = 0;
    read_int("settings_version", file_version);
    if (file_version < CURRENT_SETTINGS_VERSION) {
        const GraphicsSettings defaults;
        sun_intensity = defaults.sun_intensity;
        ambient_strength = defaults.ambient_strength;
        exposure = defaults.exposure;
        bloom_strength = defaults.bloom_strength;
        bloom_threshold = defaults.bloom_threshold;
        shadow_mode = defaults.shadow_mode;
        shadow_cascades = defaults.shadow_cascades;
        ao_radius = defaults.ao_radius;
        vsync_mode = defaults.vsync_mode;
    }
    settings_version = CURRENT_SETTINGS_VERSION;

    // Normalize everything enum-like once at the trust boundary.
    clamp_to_valid_ranges();

    return true;
}

} // namespace bracken
