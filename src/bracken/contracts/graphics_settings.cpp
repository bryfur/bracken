#include "bracken/contracts/graphics_settings.hpp"

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
    draw_distance = clamp_enum(draw_distance, DrawDistance::D500, DrawDistance::D32000);
    anisotropic_filter = clamp_enum(anisotropic_filter, AnisotropicFilter::Off, AnisotropicFilter::X16);
    vsync_mode = clamp_enum(vsync_mode, VSyncMode::Immediate, VSyncMode::Mailbox);
    shadow_mode = clamp_enum(shadow_mode, ShadowMode::Off, ShadowMode::PCSS);
    shadow_cascades = clamp_enum(shadow_cascades, ShadowCascades::One, ShadowCascades::Four);
    shadow_resolution = clamp_enum(shadow_resolution, ShadowResolution::R512, ShadowResolution::R4096);
    rt_shadow_resolution = clamp_enum(rt_shadow_resolution, RtResolution::Full, RtResolution::Quarter);
    rt_reflection_resolution =
        clamp_enum(rt_reflection_resolution, RtResolution::Full, RtResolution::Quarter);
    // 0 is meaningful (a hard shadow); the upper bound stops a typo turning the
    // sun into a dome light and the denoiser into a blur.
    rt_sun_softness_deg = std::clamp(rt_sun_softness_deg, 0.0f, 10.0f);
    rt_reflection_max_roughness = std::clamp(rt_reflection_max_roughness, 0.0f, 1.0f);
    rt_max_instance_range = clamp_enum(rt_max_instance_range, RtRange::Near, RtRange::Unlimited);
    // `Full` is a real enumerator since R5c, so a file holding 2 now survives.
    // It could not before, and deliberately: until the tier existed, a 2 on
    // disk clamped DOWN to Ratio rather than arriving as a silent opt-in to a
    // tier nobody chose. 3 and up still clamp to Full, and -1 to Off.
    rt_local_shadows = clamp_enum(rt_local_shadows, LocalRtShadowMode::Off, LocalRtShadowMode::Full);
    rt_local_shadow_resolution =
        clamp_enum(rt_local_shadow_resolution, RtResolution::Full, RtResolution::Quarter);
    // ProbesScreen is a real enumerator and clamps like one, even though it
    // behaves as Probes until phase R4 — a settings file that already holds it
    // must survive the version that implements it unchanged.
    gi_mode = clamp_enum(gi_mode, GiMode::Off, GiMode::ProbesScreen);
    gi_quality = clamp_enum(gi_quality, GiQuality::Low, GiQuality::High);
    shadow_distance = std::clamp(shadow_distance, 200.0f, 20000.0f);
    // Budget bounds mirror render::lighting::LOCAL_SHADOW_MAX_{POINT,SPOT}_BUDGET
    // (the 8x8 atlas holds 8*6 point-face tiles + 16 spot tiles exactly).
    local_shadow_point_budget = std::clamp(local_shadow_point_budget, 0, 8);
    local_shadow_spot_budget = std::clamp(local_shadow_spot_budget, 0, 16);
    local_shadow_resolution =
        clamp_enum(local_shadow_resolution, LocalShadowResolution::R1024, LocalShadowResolution::R4096);
    ao_mode = clamp_enum(ao_mode, AOMode::Off, AOMode::GTAO_Full);
    aa_mode = clamp_enum(aa_mode, AAMode::Off, AAMode::TAA);
    window_mode = clamp_enum(window_mode, WindowMode::Windowed, WindowMode::ExclusiveFullscreen);
    // Only the lower bound is enforced here — the upper bound depends on the
    // runtime display-mode list, which this struct never sees. An index past
    // that list is handled at apply time: RenderContext::set_window_mode
    // warns and falls back to borderless fullscreen.
    resolution_index = std::max(resolution_index, 0);
    tonemap_mode = clamp_enum(tonemap_mode, TonemapMode::ACESFitted, TonemapMode::None);
    // Auto exposure: EV clamp window kept ordered and within the histogram's
    // meaningful correction range; adaptation speeds non-negative and sane.
    auto_exposure_min_ev = std::clamp(auto_exposure_min_ev, -12.0f, 12.0f);
    auto_exposure_max_ev = std::clamp(auto_exposure_max_ev, auto_exposure_min_ev, 12.0f);
    auto_exposure_speed_up = std::clamp(auto_exposure_speed_up, 0.0f, 20.0f);
    auto_exposure_speed_down = std::clamp(auto_exposure_speed_down, 0.0f, 20.0f);
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
    out << "draw_distance=" << static_cast<int>(draw_distance) << "\n";
    out << "frustum_culling=" << frustum_culling << "\n";
    out << "anisotropic_filter=" << static_cast<int>(anisotropic_filter) << "\n";
    out << "vsync_mode=" << static_cast<int>(vsync_mode) << "\n";
    out << "shadow_mode=" << static_cast<int>(shadow_mode) << "\n";
    out << "shadow_cascades=" << static_cast<int>(shadow_cascades) << "\n";
    out << "shadow_resolution=" << static_cast<int>(shadow_resolution) << "\n";
    out << "shadow_distance=" << shadow_distance << "\n";
    out << "local_shadows=" << local_shadows << "\n";
    out << "local_shadow_point_budget=" << local_shadow_point_budget << "\n";
    out << "local_shadow_spot_budget=" << local_shadow_spot_budget << "\n";
    out << "local_shadow_resolution=" << static_cast<int>(local_shadow_resolution) << "\n";
    out << "ao_mode=" << static_cast<int>(ao_mode) << "\n";
    out << "aa_mode=" << static_cast<int>(aa_mode) << "\n";
    out << "ibl_enabled=" << ibl_enabled << "\n";
    out << "bloom_enabled=" << bloom_enabled << "\n";
    out << "bloom_strength=" << bloom_strength << "\n";
    out << "rt_shadows=" << rt_shadows << "\n";
    out << "rt_reflections=" << rt_reflections << "\n";
    out << "rt_shadow_resolution=" << static_cast<int>(rt_shadow_resolution) << "\n";
    out << "rt_reflection_resolution=" << static_cast<int>(rt_reflection_resolution) << "\n";
    out << "rt_sun_softness_deg=" << rt_sun_softness_deg << "\n";
    out << "rt_reflection_max_roughness=" << rt_reflection_max_roughness << "\n";
    out << "rt_max_instance_range=" << static_cast<int>(rt_max_instance_range) << "\n";
    out << "rt_local_shadows=" << static_cast<int>(rt_local_shadows) << "\n";
    out << "rt_local_shadow_resolution=" << static_cast<int>(rt_local_shadow_resolution) << "\n";
    out << "gi_mode=" << static_cast<int>(gi_mode) << "\n";
    out << "gi_quality=" << static_cast<int>(gi_quality) << "\n";
    out << "volumetric_fog=" << volumetric_fog << "\n";
    out << "god_rays=" << god_rays << "\n";
    out << "window_mode=" << static_cast<int>(window_mode) << "\n";
    out << "resolution_index=" << resolution_index << "\n";
    out << "exposure=" << exposure << "\n";
    out << "tonemap_mode=" << static_cast<int>(tonemap_mode) << "\n";
    out << "contrast=" << contrast << "\n";
    out << "saturation=" << saturation << "\n";
    out << "auto_exposure=" << auto_exposure << "\n";
    out << "auto_exposure_min_ev=" << auto_exposure_min_ev << "\n";
    out << "auto_exposure_max_ev=" << auto_exposure_max_ev << "\n";
    out << "auto_exposure_speed_up=" << auto_exposure_speed_up << "\n";
    out << "auto_exposure_speed_down=" << auto_exposure_speed_down << "\n";
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
    read_enum("draw_distance", draw_distance);
    read_bool("frustum_culling", frustum_culling);
    read_enum("anisotropic_filter", anisotropic_filter);
    read_enum("vsync_mode", vsync_mode);
    read_enum("shadow_mode", shadow_mode);
    read_enum("shadow_cascades", shadow_cascades);
    read_enum("shadow_resolution", shadow_resolution);
    read_float("shadow_distance", shadow_distance);
    read_bool("local_shadows", local_shadows);
    read_int("local_shadow_point_budget", local_shadow_point_budget);
    read_int("local_shadow_spot_budget", local_shadow_spot_budget);
    read_enum("local_shadow_resolution", local_shadow_resolution);
    read_enum("ao_mode", ao_mode);
    read_enum("aa_mode", aa_mode);
    read_bool("ibl_enabled", ibl_enabled);
    read_bool("bloom_enabled", bloom_enabled);
    read_float("bloom_strength", bloom_strength);
    read_bool("rt_shadows", rt_shadows);
    read_enum("rt_shadow_resolution", rt_shadow_resolution);
    read_enum("rt_reflection_resolution", rt_reflection_resolution);
    read_float("rt_sun_softness_deg", rt_sun_softness_deg);
    read_float("rt_reflection_max_roughness", rt_reflection_max_roughness);
    read_enum("rt_max_instance_range", rt_max_instance_range);
    read_enum("rt_local_shadows", rt_local_shadows);
    read_enum("rt_local_shadow_resolution", rt_local_shadow_resolution);
    read_enum("gi_mode", gi_mode);
    read_enum("gi_quality", gi_quality);
    read_bool("rt_reflections", rt_reflections);
    read_bool("volumetric_fog", volumetric_fog);
    read_bool("god_rays", god_rays);
    read_enum("window_mode", window_mode);
    read_int("resolution_index", resolution_index);
    read_float("exposure", exposure);
    read_enum("tonemap_mode", tonemap_mode);
    read_float("contrast", contrast);
    read_float("saturation", saturation);
    read_bool("auto_exposure", auto_exposure);
    read_float("auto_exposure_min_ev", auto_exposure_min_ev);
    read_float("auto_exposure_max_ev", auto_exposure_max_ev);
    read_float("auto_exposure_speed_up", auto_exposure_speed_up);
    read_float("auto_exposure_speed_down", auto_exposure_speed_down);
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
