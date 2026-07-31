#include "bracken/graphics_settings.hpp"
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <gtest/gtest.h>

using namespace bracken;

static const std::string TEST_FILE = "/tmp/bracken_test_settings.cfg";

class GraphicsSettingsTest : public ::testing::Test {
protected:
    void TearDown() override { std::remove(TEST_FILE.c_str()); }
};

TEST_F(GraphicsSettingsTest, DefaultValues) {
    GraphicsSettings s;
    EXPECT_EQ(s.settings_version, GraphicsSettings::CURRENT_SETTINGS_VERSION);
    EXPECT_EQ(GraphicsSettings::CURRENT_SETTINGS_VERSION, 2);
    EXPECT_TRUE(s.fog_enabled);
    EXPECT_TRUE(s.grass_enabled);
    EXPECT_TRUE(s.skybox_enabled);
    EXPECT_EQ(s.shadow_mode, ShadowMode::PCSS);
    EXPECT_EQ(s.shadow_cascades, 3);
    EXPECT_EQ(s.ao_mode, AOMode::GTAO_Half); // half-res GTAO on by default for fresh settings
    EXPECT_EQ(s.aa_mode, AAMode::TAA); // TAA on by default for fresh settings
    EXPECT_EQ(s.vsync_mode, 2);
    EXPECT_TRUE(s.ibl_enabled); // sky image-based lighting on by default
    EXPECT_TRUE(s.local_shadows); // point/spot light shadows on by default
    EXPECT_EQ(s.local_shadow_point_budget, 4);
    EXPECT_EQ(s.local_shadow_spot_budget, 2);
    EXPECT_EQ(s.local_shadow_resolution, 1); // 2048 atlas
    EXPECT_TRUE(s.bloom_enabled);
    EXPECT_FLOAT_EQ(s.bloom_strength, 0.25f);
    EXPECT_FLOAT_EQ(s.bloom_threshold, 1.1f);
    EXPECT_FLOAT_EQ(s.sun_intensity, 3.0f);
    EXPECT_FLOAT_EQ(s.ambient_strength, 0.45f);
    EXPECT_FLOAT_EQ(s.exposure, 1.0f);
    EXPECT_FLOAT_EQ(s.ao_radius, 1.0f);
    EXPECT_FALSE(s.volumetric_fog);
    EXPECT_FALSE(s.god_rays);
    EXPECT_FALSE(s.show_fps);
    EXPECT_TRUE(s.auto_exposure); // histogram auto exposure on by default
    EXPECT_FLOAT_EQ(s.auto_exposure_min_ev, -4.0f);
    EXPECT_FLOAT_EQ(s.auto_exposure_max_ev, 4.0f);
    EXPECT_FLOAT_EQ(s.auto_exposure_speed_up, 1.5f);
    EXPECT_FLOAT_EQ(s.auto_exposure_speed_down, 2.5f);
}

TEST_F(GraphicsSettingsTest, GetDrawDistanceMapping) {
    GraphicsSettings s;
    s.draw_distance = 0;
    EXPECT_FLOAT_EQ(s.get_draw_distance(), 500.0f);
    s.draw_distance = 1;
    EXPECT_FLOAT_EQ(s.get_draw_distance(), 1000.0f);
    s.draw_distance = 3;
    EXPECT_FLOAT_EQ(s.get_draw_distance(), 4000.0f);
    s.draw_distance = 6;
    EXPECT_FLOAT_EQ(s.get_draw_distance(), 32000.0f);
}

TEST_F(GraphicsSettingsTest, SaveCreatesFile) {
    GraphicsSettings s;
    EXPECT_TRUE(s.save(TEST_FILE));
    EXPECT_TRUE(std::filesystem::exists(TEST_FILE));
}

TEST_F(GraphicsSettingsTest, LoadNonexistentFileReturnsFalse) {
    GraphicsSettings s;
    EXPECT_FALSE(s.load("/tmp/does_not_exist_12345.cfg"));
    // Defaults should be preserved
    EXPECT_TRUE(s.fog_enabled);
    EXPECT_EQ(s.shadow_mode, ShadowMode::PCSS);
}

TEST_F(GraphicsSettingsTest, SaveLoadRoundTripsAllFields) {
    GraphicsSettings original;
    original.fog_enabled = false;
    original.grass_enabled = false;
    original.skybox_enabled = false;
    original.trees_enabled = false;
    original.rocks_enabled = false;
    original.show_fps = true;
    original.show_debug_hud = true;
    original.draw_distance = 5;
    original.frustum_culling = false;
    original.shadow_mode = ShadowMode::PCF;
    original.shadow_cascades = 3;
    original.shadow_resolution = 3;
    original.shadow_distance = 6500.0f;
    original.ao_mode = AOMode::GTAO_Full;
    original.ibl_enabled = false;
    original.local_shadows = false;
    original.local_shadow_point_budget = 7;
    original.local_shadow_spot_budget = 5;
    original.local_shadow_resolution = 2;
    original.bloom_enabled = false;
    original.bloom_strength = 0.77f;
    original.volumetric_fog = true;
    original.god_rays = true;
    original.vsync_mode = 1;
    original.window_mode = 1;
    original.anisotropic_filter = 3;
    original.aa_mode = AAMode::Off;
    original.auto_exposure = false;
    original.auto_exposure_min_ev = -2.0f;
    original.auto_exposure_max_ev = 6.0f;
    original.auto_exposure_speed_up = 3.0f;
    original.auto_exposure_speed_down = 0.5f;

    ASSERT_TRUE(original.save(TEST_FILE));

    GraphicsSettings loaded;
    ASSERT_TRUE(loaded.load(TEST_FILE));

    EXPECT_EQ(loaded.fog_enabled, false);
    EXPECT_EQ(loaded.grass_enabled, false);
    EXPECT_EQ(loaded.skybox_enabled, false);
    EXPECT_EQ(loaded.trees_enabled, false);
    EXPECT_EQ(loaded.rocks_enabled, false);
    EXPECT_EQ(loaded.show_fps, true);
    EXPECT_EQ(loaded.show_debug_hud, true);
    EXPECT_EQ(loaded.draw_distance, 5);
    EXPECT_EQ(loaded.frustum_culling, false);
    EXPECT_EQ(loaded.shadow_mode, ShadowMode::PCF);
    EXPECT_EQ(loaded.shadow_cascades, 3);
    EXPECT_EQ(loaded.shadow_resolution, 3);
    EXPECT_NEAR(loaded.shadow_distance, 6500.0f, 0.5f);
    EXPECT_EQ(loaded.ao_mode, AOMode::GTAO_Full);
    EXPECT_EQ(loaded.ibl_enabled, false);
    EXPECT_EQ(loaded.local_shadows, false);
    EXPECT_EQ(loaded.local_shadow_point_budget, 7);
    EXPECT_EQ(loaded.local_shadow_spot_budget, 5);
    EXPECT_EQ(loaded.local_shadow_resolution, 2);
    EXPECT_EQ(loaded.bloom_enabled, false);
    EXPECT_NEAR(loaded.bloom_strength, 0.77f, 0.01f);
    EXPECT_EQ(loaded.volumetric_fog, true);
    EXPECT_EQ(loaded.god_rays, true);
    EXPECT_EQ(loaded.vsync_mode, 1);
    EXPECT_EQ(loaded.window_mode, 1);
    EXPECT_EQ(loaded.anisotropic_filter, 3);
    EXPECT_EQ(loaded.aa_mode, AAMode::Off);
    EXPECT_EQ(loaded.auto_exposure, false);
    EXPECT_FLOAT_EQ(loaded.auto_exposure_min_ev, -2.0f);
    EXPECT_FLOAT_EQ(loaded.auto_exposure_max_ev, 6.0f);
    EXPECT_FLOAT_EQ(loaded.auto_exposure_speed_up, 3.0f);
    EXPECT_FLOAT_EQ(loaded.auto_exposure_speed_down, 0.5f);
    EXPECT_EQ(loaded.settings_version, GraphicsSettings::CURRENT_SETTINGS_VERSION);
}

TEST_F(GraphicsSettingsTest, FloatPrecisionPreserved) {
    GraphicsSettings original;
    original.bloom_strength = 0.123456f;
    ASSERT_TRUE(original.save(TEST_FILE));

    GraphicsSettings loaded;
    ASSERT_TRUE(loaded.load(TEST_FILE));
    EXPECT_NEAR(loaded.bloom_strength, 0.123456f, 0.001f);
}

// --- Edge case tests ---

TEST_F(GraphicsSettingsTest, LoadFileWithUnknownKeys) {
    // Unknown keys should be silently ignored
    {
        std::ofstream out(TEST_FILE);
        out << "settings_version=2\n";
        out << "fog_enabled=0\n";
        out << "totally_unknown_key=999\n";
        out << "shadow_mode=1\n";
        out << "another_fake=hello\n";
    }
    GraphicsSettings s;
    ASSERT_TRUE(s.load(TEST_FILE));
    EXPECT_FALSE(s.fog_enabled);
    EXPECT_EQ(s.shadow_mode, ShadowMode::PCF);
    // Other fields remain at defaults
    EXPECT_TRUE(s.grass_enabled);
    EXPECT_TRUE(s.bloom_enabled);
}

TEST_F(GraphicsSettingsTest, LoadFileWithMalformedValues) {
    // Malformed integer values should leave defaults intact
    {
        std::ofstream out(TEST_FILE);
        out << "settings_version=2\n";
        out << "shadow_mode=abc\n";
        out << "draw_distance=!@#\n";
        out << "bloom_strength=not_a_float\n";
        out << "ao_mode=2\n"; // this one is valid (2 = GTAO_Full on disk)
    }
    GraphicsSettings s;
    ASSERT_TRUE(s.load(TEST_FILE));
    EXPECT_EQ(s.shadow_mode, ShadowMode::PCSS);              // default, "abc" failed to parse
    EXPECT_EQ(s.draw_distance, 3);            // default, "!@#" failed to parse
    EXPECT_FLOAT_EQ(s.bloom_strength, 0.25f); // default, "not_a_float" failed
    EXPECT_EQ(s.ao_mode, AOMode::GTAO_Full);  // successfully parsed
}

TEST_F(GraphicsSettingsTest, LoadFileWithComments) {
    // Lines starting with '#' should be skipped
    {
        std::ofstream out(TEST_FILE);
        out << "# This is a comment\n";
        out << "settings_version=2\n";
        out << "fog_enabled=0\n";
        out << "# Another comment\n";
        out << "shadow_mode=0\n";
    }
    GraphicsSettings s;
    ASSERT_TRUE(s.load(TEST_FILE));
    EXPECT_FALSE(s.fog_enabled);
    EXPECT_EQ(s.shadow_mode, ShadowMode::Off);
}

TEST_F(GraphicsSettingsTest, LoadEmptyFileKeepsDefaults) {
    {
        std::ofstream out(TEST_FILE);
        // Write nothing
    }
    GraphicsSettings s;
    ASSERT_TRUE(s.load(TEST_FILE));
    // All defaults should be intact
    EXPECT_TRUE(s.fog_enabled);
    EXPECT_TRUE(s.grass_enabled);
    EXPECT_TRUE(s.skybox_enabled);
    EXPECT_TRUE(s.trees_enabled);
    EXPECT_TRUE(s.rocks_enabled);
    EXPECT_FALSE(s.show_fps);
    EXPECT_FALSE(s.show_debug_hud);
    EXPECT_EQ(s.draw_distance, 3);
    EXPECT_TRUE(s.frustum_culling);
    EXPECT_EQ(s.anisotropic_filter, 4);
    EXPECT_EQ(s.vsync_mode, 2);
    EXPECT_EQ(s.shadow_mode, ShadowMode::PCSS);
    EXPECT_EQ(s.shadow_cascades, 3);
    EXPECT_EQ(s.shadow_resolution, 2);
    EXPECT_EQ(s.ao_mode, AOMode::GTAO_Half);
    EXPECT_EQ(s.aa_mode, AAMode::TAA);
    EXPECT_TRUE(s.bloom_enabled);
    EXPECT_FLOAT_EQ(s.bloom_strength, 0.25f);
    EXPECT_FALSE(s.volumetric_fog);
    EXPECT_FALSE(s.god_rays);
    EXPECT_EQ(s.window_mode, 0);
    EXPECT_EQ(s.resolution_index, 0);
}

TEST_F(GraphicsSettingsTest, SaveOverwritesExistingFile) {
    // Save once with one set of values
    GraphicsSettings s1;
    s1.shadow_mode = ShadowMode::Off;
    s1.fog_enabled = false;
    ASSERT_TRUE(s1.save(TEST_FILE));

    // Save again with different values
    GraphicsSettings s2;
    s2.shadow_mode = ShadowMode::PCF;
    s2.fog_enabled = true;
    ASSERT_TRUE(s2.save(TEST_FILE));

    // Load should reflect the second save
    GraphicsSettings loaded;
    ASSERT_TRUE(loaded.load(TEST_FILE));
    EXPECT_EQ(loaded.shadow_mode, ShadowMode::PCF);
    EXPECT_TRUE(loaded.fog_enabled);
}

TEST_F(GraphicsSettingsTest, LoadWithExtraWhitespaceAroundValues) {
    // The current parser does not trim whitespace, so "shadow_mode" with spaces
    // in the key won't match. But trailing whitespace in value is passed to stoi
    // which does handle leading whitespace. Test what the parser actually handles.
    {
        std::ofstream out(TEST_FILE);
        // stoi/stof skip leading whitespace in the value
        out << "settings_version=2\n";
        out << "shadow_mode= 1\n";
        out << "bloom_strength= 0.55\n";
        out << "ao_mode=2\n";
    }
    GraphicsSettings s;
    ASSERT_TRUE(s.load(TEST_FILE));
    // stoi(" 1") = 1, stof(" 0.55") = 0.55 - leading space in value is fine
    EXPECT_EQ(s.shadow_mode, ShadowMode::PCF);
    EXPECT_NEAR(s.bloom_strength, 0.55f, 0.01f);
    EXPECT_EQ(s.ao_mode, AOMode::GTAO_Full);
}

TEST_F(GraphicsSettingsTest, BooleanValuesZeroAndOne) {
    {
        std::ofstream out(TEST_FILE);
        out << "fog_enabled=1\n";
        out << "grass_enabled=0\n";
        out << "show_fps=1\n";
        out << "frustum_culling=0\n";
    }
    GraphicsSettings s;
    ASSERT_TRUE(s.load(TEST_FILE));
    EXPECT_TRUE(s.fog_enabled);
    EXPECT_FALSE(s.grass_enabled);
    EXPECT_TRUE(s.show_fps);
    EXPECT_FALSE(s.frustum_culling);
}

TEST_F(GraphicsSettingsTest, BooleanNonZeroTreatedAsTrue) {
    // The parser treats anything != "0" as true for booleans
    {
        std::ofstream out(TEST_FILE);
        out << "fog_enabled=true\n";
        out << "grass_enabled=yes\n";
        out << "show_fps=42\n";
    }
    GraphicsSettings s;
    s.fog_enabled = false;
    s.grass_enabled = false;
    s.show_fps = false;
    ASSERT_TRUE(s.load(TEST_FILE));
    EXPECT_TRUE(s.fog_enabled);
    EXPECT_TRUE(s.grass_enabled);
    EXPECT_TRUE(s.show_fps);
}

TEST_F(GraphicsSettingsTest, NegativeIntegerValuesClampedOnLoad) {
    {
        std::ofstream out(TEST_FILE);
        out << "settings_version=2\n";
        out << "shadow_mode=-1\n";
        out << "draw_distance=-5\n";
    }
    GraphicsSettings s;
    ASSERT_TRUE(s.load(TEST_FILE));
    // load() normalizes at the trust boundary: negative values clamp to the
    // bottom of each range instead of being stored as-is.
    EXPECT_EQ(s.shadow_mode, ShadowMode::Off);
    EXPECT_EQ(s.draw_distance, 0);
}

// --- Settings-version migration tests ---

TEST_F(GraphicsSettingsTest, MigrationResetsLightingValuesFromVersionlessFile) {
    // A file without settings_version was authored against the old lighting
    // model: lighting-related values must be reset to the new defaults.
    {
        std::ofstream out(TEST_FILE);
        out << "sun_intensity=1.0\n";
        out << "ambient_strength=0.5\n";
        out << "exposure=1.3\n";
        out << "bloom_strength=0.6\n";
        out << "bloom_threshold=0.8\n";
        out << "shadow_mode=1\n";
        out << "shadow_cascades=1\n";
        out << "ao_radius=2.0\n";
        out << "vsync_mode=0\n";
    }
    GraphicsSettings s;
    ASSERT_TRUE(s.load(TEST_FILE));

    const GraphicsSettings defaults;
    EXPECT_FLOAT_EQ(s.sun_intensity, defaults.sun_intensity);
    EXPECT_FLOAT_EQ(s.ambient_strength, defaults.ambient_strength);
    EXPECT_FLOAT_EQ(s.exposure, defaults.exposure);
    EXPECT_FLOAT_EQ(s.bloom_strength, defaults.bloom_strength);
    EXPECT_FLOAT_EQ(s.bloom_threshold, defaults.bloom_threshold);
    EXPECT_EQ(s.shadow_mode, defaults.shadow_mode);
    EXPECT_EQ(s.shadow_cascades, defaults.shadow_cascades);
    EXPECT_FLOAT_EQ(s.ao_radius, defaults.ao_radius);
    EXPECT_EQ(s.vsync_mode, defaults.vsync_mode);
    EXPECT_EQ(s.settings_version, GraphicsSettings::CURRENT_SETTINGS_VERSION);
}

TEST_F(GraphicsSettingsTest, MigrationPreservesNonLightingValues) {
    // Non-lighting preferences in a versionless file survive the migration.
    {
        std::ofstream out(TEST_FILE);
        out << "fog_enabled=0\n";
        out << "draw_distance=5\n";
        out << "window_mode=1\n";
        out << "ao_strength=1.5\n";
        out << "sun_intensity=1.0\n"; // migrated away
    }
    GraphicsSettings s;
    ASSERT_TRUE(s.load(TEST_FILE));

    EXPECT_FALSE(s.fog_enabled);
    EXPECT_EQ(s.draw_distance, 5);
    EXPECT_EQ(s.window_mode, 1);
    EXPECT_FLOAT_EQ(s.ao_strength, 1.5f);
    EXPECT_FLOAT_EQ(s.sun_intensity, 3.0f);
}

TEST_F(GraphicsSettingsTest, VersionedFileSkipsMigration) {
    // Files that carry the current settings_version keep their lighting values.
    {
        std::ofstream out(TEST_FILE);
        out << "settings_version=2\n";
        out << "sun_intensity=1.7\n";
        out << "vsync_mode=0\n";
        out << "shadow_mode=1\n";
    }
    GraphicsSettings s;
    ASSERT_TRUE(s.load(TEST_FILE));

    EXPECT_FLOAT_EQ(s.sun_intensity, 1.7f);
    EXPECT_EQ(s.vsync_mode, 0);
    EXPECT_EQ(s.shadow_mode, ShadowMode::PCF);
    EXPECT_EQ(s.settings_version, GraphicsSettings::CURRENT_SETTINGS_VERSION);
}

TEST_F(GraphicsSettingsTest, SaveAlwaysWritesSettingsVersion) {
    GraphicsSettings s;
    ASSERT_TRUE(s.save(TEST_FILE));

    std::ifstream in(TEST_FILE);
    bool found = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line == "settings_version=2") {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(GraphicsSettingsTest, AaModeRoundTrips) {
    // Persisted files keep their stored mode (the TAA default only applies to
    // fresh settings / files without an aa_mode key).
    for (AAMode mode : {AAMode::Off, AAMode::FXAA, AAMode::TAA}) {
        GraphicsSettings original;
        original.aa_mode = mode;
        ASSERT_TRUE(original.save(TEST_FILE));

        GraphicsSettings loaded;
        loaded.aa_mode = AAMode::Off; // any pre-load value must be overwritten
        ASSERT_TRUE(loaded.load(TEST_FILE));
        EXPECT_EQ(loaded.aa_mode, mode);
    }
}

TEST_F(GraphicsSettingsTest, AaModeTAASerializesAsTwo) {
    // The on-disk format is numeric: TAA must stay 2 forever (files depend on it).
    GraphicsSettings s;
    s.aa_mode = AAMode::TAA;
    ASSERT_TRUE(s.save(TEST_FILE));

    std::ifstream in(TEST_FILE);
    bool found = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line == "aa_mode=2") {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(GraphicsSettingsTest, IblEnabledSerializesNumerically) {
    // The on-disk format is numeric: the IBL toggle writes 0/1 like every
    // other bool, and a stored 0 must survive a round trip past the
    // default-true initializer.
    GraphicsSettings s;
    s.ibl_enabled = false;
    ASSERT_TRUE(s.save(TEST_FILE));

    std::ifstream in(TEST_FILE);
    bool found = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line == "ibl_enabled=0") {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);

    GraphicsSettings loaded; // defaults to true
    ASSERT_TRUE(loaded.load(TEST_FILE));
    EXPECT_FALSE(loaded.ibl_enabled);
}

TEST_F(GraphicsSettingsTest, SaveToVeryLongFilePath) {
    // Attempting to save to a deeply nested nonexistent directory should fail
    std::string long_path = "/tmp";
    for (int i = 0; i < 50; i++) {
        long_path += "/nonexistent_dir_" + std::to_string(i);
    }
    long_path += "/settings.cfg";

    GraphicsSettings s;
    EXPECT_FALSE(s.save(long_path));
}

// --- Corrupt-settings-file robustness (clamp_to_valid_ranges) ---

TEST_F(GraphicsSettingsTest, GetDrawDistanceClampsOutOfRangeIndex) {
    GraphicsSettings s;
    // A corrupt file (or a buggy host) can leave the public field out of
    // range; the accessor must never index out of bounds.
    s.draw_distance = 9;
    EXPECT_FLOAT_EQ(s.get_draw_distance(), 32000.0f);
    s.draw_distance = -3;
    EXPECT_FLOAT_EQ(s.get_draw_distance(), 500.0f);
}

TEST_F(GraphicsSettingsTest, LoadClampsOutOfRangeDrawDistance) {
    {
        std::ofstream out(TEST_FILE);
        out << "settings_version=2\n";
        out << "draw_distance=9\n";
    }
    GraphicsSettings s;
    ASSERT_TRUE(s.load(TEST_FILE));
    EXPECT_EQ(s.draw_distance, 6);
    EXPECT_FLOAT_EQ(s.get_draw_distance(), 32000.0f);
}

TEST_F(GraphicsSettingsTest, LoadClampsOutOfRangeModes) {
    {
        std::ofstream out(TEST_FILE);
        out << "settings_version=2\n";
        out << "shadow_mode=99\n";
        out << "ao_mode=77\n";
        out << "aa_mode=9\n"; // clamps to the highest mode (TAA)
        out << "tonemap_mode=42\n";
        out << "vsync_mode=9\n";
        out << "window_mode=5\n";
        out << "anisotropic_filter=12\n";
        out << "shadow_resolution=99\n";
        out << "shadow_cascades=-2\n";
        out << "resolution_index=-7\n";
    }
    GraphicsSettings s;
    ASSERT_TRUE(s.load(TEST_FILE));
    EXPECT_EQ(s.shadow_mode, ShadowMode::PCSS);
    EXPECT_EQ(s.ao_mode, AOMode::GTAO_Full); // clamps to the highest mode
    EXPECT_EQ(s.aa_mode, AAMode::TAA);
    EXPECT_EQ(s.tonemap_mode, TonemapMode::None);
    EXPECT_EQ(s.vsync_mode, 2);
    EXPECT_EQ(s.window_mode, 2);
    EXPECT_EQ(s.anisotropic_filter, 4);
    EXPECT_EQ(s.shadow_resolution, 3);
    EXPECT_EQ(s.shadow_cascades, 0);
    EXPECT_EQ(s.resolution_index, 0);
}

// The explicit underlying values ARE the on-disk format. The AO values were
// deliberately redefined in the AO rearchitecture: 1 (formerly "SSAO") is now
// GTAO_Half and 2 (formerly the fake "GTAO") is GTAO_Full, preserving the
// relative cost of old settings files.
TEST_F(GraphicsSettingsTest, AoModeOnDiskValuesArePinned) {
    EXPECT_EQ(static_cast<int>(AOMode::Off), 0);
    EXPECT_EQ(static_cast<int>(AOMode::GTAO_Half), 1);
    EXPECT_EQ(static_cast<int>(AOMode::GTAO_Full), 2);
}

TEST_F(GraphicsSettingsTest, ClampToValidRangesNormalizesDirectMutation) {
    GraphicsSettings s;
    s.draw_distance = 100;
    s.shadow_mode = static_cast<ShadowMode>(7);
    s.ao_mode = static_cast<AOMode>(-1);
    s.aa_mode = static_cast<AAMode>(5);
    s.tonemap_mode = static_cast<TonemapMode>(-2);
    s.local_shadow_point_budget = 99;
    s.local_shadow_spot_budget = -3;
    s.local_shadow_resolution = 7;
    s.clamp_to_valid_ranges();
    EXPECT_EQ(s.draw_distance, 6);
    EXPECT_EQ(s.shadow_mode, ShadowMode::PCSS);
    EXPECT_EQ(s.ao_mode, AOMode::Off);
    EXPECT_EQ(s.aa_mode, AAMode::TAA);
    EXPECT_EQ(s.tonemap_mode, TonemapMode::ACESFitted);
    // Budgets clamp to the 8x8 atlas capacity (8 point groups, 16 spot tiles).
    EXPECT_EQ(s.local_shadow_point_budget, 8);
    EXPECT_EQ(s.local_shadow_spot_budget, 0);
    EXPECT_EQ(s.local_shadow_resolution, 2);
}

TEST_F(GraphicsSettingsTest, LocalShadowsSerializeNumerically) {
    // The on-disk format is numeric: the toggle writes 0/1, the budgets and
    // atlas-resolution index write plain integers; a stored 0 must survive a
    // round trip past the default-true initializer.
    GraphicsSettings s;
    s.local_shadows = false;
    s.local_shadow_point_budget = 6;
    s.local_shadow_spot_budget = 3;
    s.local_shadow_resolution = 0;
    ASSERT_TRUE(s.save(TEST_FILE));

    std::ifstream in(TEST_FILE);
    bool found_toggle = false;
    bool found_point = false;
    bool found_spot = false;
    bool found_res = false;
    std::string line;
    while (std::getline(in, line)) {
        found_toggle |= (line == "local_shadows=0");
        found_point |= (line == "local_shadow_point_budget=6");
        found_spot |= (line == "local_shadow_spot_budget=3");
        found_res |= (line == "local_shadow_resolution=0");
    }
    EXPECT_TRUE(found_toggle);
    EXPECT_TRUE(found_point);
    EXPECT_TRUE(found_spot);
    EXPECT_TRUE(found_res);

    GraphicsSettings loaded;
    ASSERT_TRUE(loaded.load(TEST_FILE));
    EXPECT_FALSE(loaded.local_shadows);
    EXPECT_EQ(loaded.local_shadow_point_budget, 6);
    EXPECT_EQ(loaded.local_shadow_spot_budget, 3);
    EXPECT_EQ(loaded.local_shadow_resolution, 0);
}

TEST_F(GraphicsSettingsTest, AutoExposureClampedOnLoad) {
    // The EV window clamps to [-12, 12] and stays ordered (max >= min);
    // adaptation speeds clamp to [0, 20].
    GraphicsSettings s;
    s.auto_exposure_min_ev = -50.0f;
    s.auto_exposure_max_ev = 50.0f;
    s.auto_exposure_speed_up = -1.0f;
    s.auto_exposure_speed_down = 999.0f;
    s.clamp_to_valid_ranges();
    EXPECT_FLOAT_EQ(s.auto_exposure_min_ev, -12.0f);
    EXPECT_FLOAT_EQ(s.auto_exposure_max_ev, 12.0f);
    EXPECT_FLOAT_EQ(s.auto_exposure_speed_up, 0.0f);
    EXPECT_FLOAT_EQ(s.auto_exposure_speed_down, 20.0f);

    // Inverted window: max is pulled up to min.
    s.auto_exposure_min_ev = 3.0f;
    s.auto_exposure_max_ev = -3.0f;
    s.clamp_to_valid_ranges();
    EXPECT_FLOAT_EQ(s.auto_exposure_min_ev, 3.0f);
    EXPECT_FLOAT_EQ(s.auto_exposure_max_ev, 3.0f);
}

TEST_F(GraphicsSettingsTest, AutoExposureSerializesNumerically) {
    // On-disk format: the toggle writes 0/1, the EV window and speeds write
    // plain floats; a stored 0 must survive a round trip past the
    // default-true initializer.
    GraphicsSettings s;
    s.auto_exposure = false;
    s.auto_exposure_min_ev = -1.5f;
    s.auto_exposure_max_ev = 2.5f;
    s.auto_exposure_speed_up = 4.0f;
    s.auto_exposure_speed_down = 0.25f;
    ASSERT_TRUE(s.save(TEST_FILE));

    std::ifstream in(TEST_FILE);
    bool found_toggle = false;
    bool found_min = false;
    bool found_max = false;
    bool found_up = false;
    bool found_down = false;
    std::string line;
    while (std::getline(in, line)) {
        found_toggle |= (line == "auto_exposure=0");
        found_min |= (line == "auto_exposure_min_ev=-1.5");
        found_max |= (line == "auto_exposure_max_ev=2.5");
        found_up |= (line == "auto_exposure_speed_up=4");
        found_down |= (line == "auto_exposure_speed_down=0.25");
    }
    EXPECT_TRUE(found_toggle);
    EXPECT_TRUE(found_min);
    EXPECT_TRUE(found_max);
    EXPECT_TRUE(found_up);
    EXPECT_TRUE(found_down);

    GraphicsSettings loaded;
    ASSERT_TRUE(loaded.load(TEST_FILE));
    EXPECT_FALSE(loaded.auto_exposure);
    EXPECT_FLOAT_EQ(loaded.auto_exposure_min_ev, -1.5f);
    EXPECT_FLOAT_EQ(loaded.auto_exposure_max_ev, 2.5f);
    EXPECT_FLOAT_EQ(loaded.auto_exposure_speed_up, 4.0f);
    EXPECT_FLOAT_EQ(loaded.auto_exposure_speed_down, 0.25f);
}
