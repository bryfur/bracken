#pragma once

#include <algorithm>
#include <string>

namespace mmo::engine {

// Rendering mode enums. The explicit underlying values are the integers
// written by save()/load() — keep them stable so existing settings files
// keep their meaning. Serialize with static_cast<int> at the file and
// GPU-uniform boundaries only.

enum class ShadowMode : int {
    Off = 0,
    PCF = 1,
    PCSS = 2,
};

enum class AOMode : int {
    Off = 0,
    SSAO = 1,
    GTAO = 2,
};

enum class AAMode : int {
    Off = 0,
    FXAA = 1,
};

enum class TonemapMode : int {
    ACESFitted = 0,
    ACESNarkowicz = 1,
    Reinhard = 2,
    None = 3,
};

struct GraphicsSettings {
    // Bumped when default semantics change (e.g. the wave-3 lighting model
    // retuned sun/ambient/exposure/bloom). Files saved without this key were
    // authored against the old lighting model and get their lighting-related
    // values migrated to the new defaults on load().
    static constexpr int CURRENT_SETTINGS_VERSION = 2;
    int settings_version = CURRENT_SETTINGS_VERSION;

    bool fog_enabled = true;
    bool grass_enabled = true;
    bool skybox_enabled = true;
    bool trees_enabled = true;
    bool rocks_enabled = true;
    bool show_fps = false;
    bool show_debug_hud = false;

    // Culling settings
    int draw_distance = 3; // 0=500, 1=1000, 2=2000, 3=4000, 4=8000, 5=16000, 6=32000
    bool frustum_culling = true;

    // Quality settings
    int anisotropic_filter = 4;               // 0=off, 1=2x, 2=4x, 3=8x, 4=16x
    int vsync_mode = 2;                       // 0=immediate, 1=vsync, 2=mailbox (clamped to max supported at apply)
    ShadowMode shadow_mode = ShadowMode::PCSS;
    int shadow_cascades = 3;                  // 0=1, 1=2, 2=3, 3=4
    int shadow_resolution = 2;                // 0=512, 1=1024, 2=2048, 3=4096
    AOMode ao_mode = AOMode::SSAO;
    AAMode aa_mode = AAMode::FXAA;
    bool bloom_enabled = true;                // Bloom post-processing
    float bloom_strength = 0.25f;             // Bloom intensity (0.0 - 1.0)
    bool volumetric_fog = false;              // Atmospheric fog haze (expensive, off by default)
    bool god_rays = false;                    // Light shafts through shadows (requires volumetric_fog)
    int window_mode = 0;                      // 0=windowed, 1=borderless fullscreen, 2=exclusive fullscreen
    int resolution_index = 0;                 // index into available native display modes

    // Tone, exposure, color grading
    float exposure = 1.0f; // linear scene multiplier before tonemap; 1.0 = neutral
    TonemapMode tonemap_mode = TonemapMode::ACESFitted;
    float contrast = 1.0f;   // post-tonemap contrast; 1.0 = neutral, 1.2 = punchy
    float saturation = 1.0f; // post-tonemap saturation; 1.0 = neutral, 0.0 = grayscale

    // Lighting feel
    float ambient_strength = 0.45f; // shadow-region lift; 0.0 = full black in shadow, 1.0 = fully lit
    float sun_intensity = 3.0f;     // multiplier on directional sun light (HDR: >1 so bloom/tonemap have headroom)

    // Bloom
    // POST-exposure HDR luminance above which bloom contributes: the prefilter
    // thresholds scene luminance * exposure, so what blooms stays stable when
    // exposure changes (identical to the old pre-exposure behavior at
    // exposure = 1.0).
    float bloom_threshold = 1.1f;

    // AO
    float ao_strength = 1.0f; // AO darkening intensity (0..1+, 1 = stock)
    float ao_radius = 1.0f;   // multiplier on AO sample radius

    // Volumetric fog
    float fog_density = 0.05f; // density multiplier

    float get_draw_distance() const {
        constexpr float distances[] = {500.0f, 1000.0f, 2000.0f, 4000.0f, 8000.0f, 16000.0f, 32000.0f};
        // Clamped so a corrupt/hand-edited settings file can never index out
        // of bounds (the field is also public, so hosts can set it directly).
        return distances[std::clamp(draw_distance, 0, 6)];
    }

    /**
     * Clamp every enum-like/index field to its valid range. Applied by load()
     * so file contents are normalized once at the trust boundary; call it
     * yourself after mutating fields from untrusted input.
     */
    void clamp_to_valid_ranges();

    [[nodiscard]] bool save(const std::string& path) const;
    [[nodiscard]] bool load(const std::string& path);
};

} // namespace mmo::engine
