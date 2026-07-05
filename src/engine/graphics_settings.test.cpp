#include "engine/graphics_settings.hpp"
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <gtest/gtest.h>

using namespace mmo::engine;

static const std::string TEST_FILE = "/tmp/mmo_test_settings.cfg";

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
    EXPECT_EQ(s.ao_mode, AOMode::SSAO);
    EXPECT_EQ(s.aa_mode, AAMode::FXAA); // FXAA on by default
    EXPECT_EQ(s.vsync_mode, 2);
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
    original.ao_mode = AOMode::GTAO;
    original.bloom_enabled = false;
    original.bloom_strength = 0.77f;
    original.volumetric_fog = true;
    original.god_rays = true;
    original.vsync_mode = 1;
    original.window_mode = 1;
    original.anisotropic_filter = 3;
    original.aa_mode = AAMode::Off;

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
    EXPECT_EQ(loaded.ao_mode, AOMode::GTAO);
    EXPECT_EQ(loaded.bloom_enabled, false);
    EXPECT_NEAR(loaded.bloom_strength, 0.77f, 0.01f);
    EXPECT_EQ(loaded.volumetric_fog, true);
    EXPECT_EQ(loaded.god_rays, true);
    EXPECT_EQ(loaded.vsync_mode, 1);
    EXPECT_EQ(loaded.window_mode, 1);
    EXPECT_EQ(loaded.anisotropic_filter, 3);
    EXPECT_EQ(loaded.aa_mode, AAMode::Off);
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
        out << "ao_mode=2\n"; // this one is valid
    }
    GraphicsSettings s;
    ASSERT_TRUE(s.load(TEST_FILE));
    EXPECT_EQ(s.shadow_mode, ShadowMode::PCSS);              // default, "abc" failed to parse
    EXPECT_EQ(s.draw_distance, 3);            // default, "!@#" failed to parse
    EXPECT_FLOAT_EQ(s.bloom_strength, 0.25f); // default, "not_a_float" failed
    EXPECT_EQ(s.ao_mode, AOMode::GTAO);       // successfully parsed
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
    EXPECT_EQ(s.ao_mode, AOMode::SSAO);
    EXPECT_EQ(s.aa_mode, AAMode::FXAA);
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
    EXPECT_EQ(s.ao_mode, AOMode::GTAO);
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
    GraphicsSettings original;
    original.aa_mode = AAMode::Off;
    ASSERT_TRUE(original.save(TEST_FILE));

    GraphicsSettings loaded;
    ASSERT_TRUE(loaded.load(TEST_FILE));
    EXPECT_EQ(loaded.aa_mode, AAMode::Off);
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
        out << "aa_mode=9\n";
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
    EXPECT_EQ(s.ao_mode, AOMode::GTAO);
    EXPECT_EQ(s.aa_mode, AAMode::FXAA);
    EXPECT_EQ(s.tonemap_mode, TonemapMode::None);
    EXPECT_EQ(s.vsync_mode, 2);
    EXPECT_EQ(s.window_mode, 2);
    EXPECT_EQ(s.anisotropic_filter, 4);
    EXPECT_EQ(s.shadow_resolution, 3);
    EXPECT_EQ(s.shadow_cascades, 0);
    EXPECT_EQ(s.resolution_index, 0);
}

TEST_F(GraphicsSettingsTest, ClampToValidRangesNormalizesDirectMutation) {
    GraphicsSettings s;
    s.draw_distance = 100;
    s.shadow_mode = static_cast<ShadowMode>(7);
    s.ao_mode = static_cast<AOMode>(-1);
    s.aa_mode = static_cast<AAMode>(3);
    s.tonemap_mode = static_cast<TonemapMode>(-2);
    s.clamp_to_valid_ranges();
    EXPECT_EQ(s.draw_distance, 6);
    EXPECT_EQ(s.shadow_mode, ShadowMode::PCSS);
    EXPECT_EQ(s.ao_mode, AOMode::Off);
    EXPECT_EQ(s.aa_mode, AAMode::FXAA);
    EXPECT_EQ(s.tonemap_mode, TonemapMode::ACESFitted);
}
