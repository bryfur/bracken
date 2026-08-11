// The level format, and the one claim it exists to make: a level written to a
// file and read back describes the SAME FRAME, byte for byte, not "a frame
// that looks the same".
//
// The headline is FramePacketDigestSurvivesARoundTrip. Everything above it is
// the machinery that has to hold for the headline to mean anything — the
// transform identity, field-by-field round trips, the stride rule, and the
// refusals — and everything below it is the diagnostics contract (the engine
// counts what it drops).

#include <gtest/gtest.h>

#include "bracken/model_loader.hpp"
#include "bracken/render/water_renderer.hpp"
#include "bracken/save/save_document.hpp"
#include "bracken/scene/level.hpp"
#include "bracken/testing/frame_test.hpp"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using bracken::Model;
using bracken::ModelHandle;
using bracken::ModelManager;
using bracken::Mesh;
using bracken::testing::TestFrame;
using namespace bracken::scene;

namespace {

std::string temp_path(const char* leaf) {
    return (std::filesystem::temp_directory_path() / leaf).string();
}

struct TempFile {
    std::string path;
    explicit TempFile(const char* leaf) : path(temp_path(leaf)) { remove(); }
    ~TempFile() { remove(); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    void remove() const {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        std::filesystem::remove(path + ".tmp", ec);
    }
};

// register_model refuses a model with no geometry, so the synthetic models
// carry one throwaway triangle. Nothing here reads mesh contents.
std::unique_ptr<Model> make_model() {
    auto m = std::make_unique<Model>();
    Mesh mesh;
    mesh.vertices.resize(3);
    mesh.indices = {0, 1, 2};
    m->meshes.push_back(std::move(mesh));
    m->bounds.min = {-1.0f, -1.0f, -1.0f};
    m->bounds.max = {1.0f, 1.0f, 1.0f};
    m->compute_bounding_sphere();
    return m;
}

void register_all(ModelManager& mm, std::initializer_list<const char*> names) {
    for (const char* n : names) {
        auto r = mm.register_model(n, make_model());
        ASSERT_TRUE(r.is_ok()) << "could not register " << n;
    }
}

/// A level with every section populated, every optional engaged, and no field
/// left at its default — a round trip that only exercises defaults proves
/// nothing about the encoding.
Level make_full_level() {
    Level level;
    level.name = "riverside";
    level.content_version = 17;

    EXPECT_EQ(level.intern_model("prop_crate", "models/prop_crate.glb"), 0u);
    EXPECT_EQ(level.intern_model("tree_oak", "models/tree_oak.glb"), 1u);
    EXPECT_EQ(level.intern_model("procedural_rock", ""), 2u);
    // Interning twice must not append.
    EXPECT_EQ(level.intern_model("tree_oak"), 1u);

    EXPECT_EQ(level.intern_string("player_spawn"), 0u);
    EXPECT_EQ(level.intern_string("chimney"), 1u);

    level.add_instance("prop_crate", {10.0f, 20.0f, 30.0f}, 0.5f, 2.0f, {0.2f, 0.4f, 0.6f, 0.8f},
                       DrawFlags::NoFog, 7u, 3u);
    level.add_instance("tree_oak", {-1.5f, 0.25f, 900.0f}, -2.25f, 0.75f);
    level.add_instance("procedural_rock", {0.0f, 0.0f, 0.0f}, 3.14159f, 12.5f, {1, 1, 1, 1},
                       DrawFlags::ForceNonInstanced | DrawFlags::NoFog);
    level.add_instance("tree_oak", {5.0f, 6.0f, 7.0f}, 1.0f, 1.0f);
    level.add_instance("prop_crate", {5.0f, 106.0f, 7.0f}, 1.0f, 1.0f);

    bracken::render::lighting::PointLightDesc pl;
    pl.position = {1.0f, 2.0f, 3.0f};
    pl.radius = 400.0f;
    pl.color = {0.9f, 0.5f, 0.2f};
    pl.intensity = 12.5f;
    pl.casts_shadows = true;
    pl.stable_id = 0xBEEFu;
    pl.source_radius = 8.0f;
    level.point_lights.push_back(pl);

    bracken::render::lighting::SpotLightDesc sl;
    sl.position = {-40.0f, 90.0f, 12.0f};
    sl.radius = 1200.0f;
    sl.direction = {0.0f, -0.7f, 0.7f};
    sl.inner_angle = bracken::core::Degrees{14.0f};
    sl.outer_angle = bracken::core::Degrees{28.5f};
    sl.color = {0.3f, 0.6f, 1.0f};
    sl.intensity = 60.0f;
    sl.casts_shadows = true;
    sl.stable_id = 99u;
    sl.source_radius = 60.0f;
    level.spot_lights.push_back(sl);

    level.add_marker("player_spawn", {30.0f, 12.0f, -40.0f}, 1.25f);
    level.add_marker("chimney", {100.0f, 400.0f, 100.0f});
    level.add_marker("chimney", {200.0f, 410.0f, 120.0f});

    LevelWater w;
    w.min_x = 1.0f;
    w.min_z = 2.0f;
    w.max_x = 3.0f;
    w.max_z = 4.0f;
    w.height = 5.0f;
    w.flow = {0.5f, -0.5f};
    w.flow_speed = 33.0f;
    w.deep_color = {0.01f, 0.02f, 0.03f};
    w.shallow_color = {0.11f, 0.12f, 0.13f};
    w.absorption = 0.5f;
    w.foam_depth = 6.0f;
    w.reflection_roughness = 0.31f;
    w.reflection_strength = 0.32f;
    w.refraction_strength = 0.33f;
    w.ssr_strength = 0.34f;
    w.ssr_max_distance = 35.0f;
    w.ssr_thickness = 36.0f;
    w.detail_wave_strength = 0.37f;
    w.sim_displacement = 38.0f;
    w.sim_normal_strength = 0.39f;
    w.ripple_foam_gain = 0.40f;
    w.sim_center = {41.0f, 42.0f};
    w.sim_extent = 43.0f;
    w.wave_speed = 44.0f;
    w.wave_damping = 0.45f;
    level.water.push_back(w);

    level.scatter_models.push_back(LevelScatterModel{0u, 3.0f, 0.45f, 0.75f});
    level.scatter_models.push_back(LevelScatterModel{1u, 1.0f, 0.60f, 0.90f});
    LevelScatterRule rule;
    rule.name = level.intern_string("ground_cover");
    rule.model_first = 0;
    rule.model_count = 2;
    rule.seed = 0x2f6e2b1u;
    rule.center_x = 12.0f;
    rule.center_z = -8.0f;
    rule.inner_radius = 3.0f;
    rule.outer_radius = 20.0f;
    rule.step = 4.0f;
    rule.jitter = 0.8f;
    rule.sink = 3.0f;
    rule.baked = 0;
    level.scatter_rules.push_back(rule);

    LevelTerrain terrain;
    terrain.resolution = 4;
    terrain.origin_x = -100.0f;
    terrain.origin_z = -200.0f;
    terrain.size = 6400.0f;
    terrain.min_height = -160.0f;
    terrain.max_height = 150.0f;
    terrain.samples = {0, 1, 2, 3, 100, 200, 300, 400, 5000, 6000, 7000, 8000, 65535, 0, 12, 34};
    terrain.splat_resolution = 2;
    terrain.splat = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    terrain.grass_exclusions.push_back(LevelRect{-1.0f, -2.0f, 3.0f, 4.0f});
    terrain.grass_exclusions.push_back(LevelRect{10.0f, 20.0f, 30.0f, 40.0f});
    level.terrain = std::move(terrain);

    bracken::render::EnvironmentState env;
    env.sun_dir = {0.1f, -0.9f, 0.3f};
    env.sun_color = {1.0f, 0.6f, 0.3f};
    env.sky_zenith = {0.05f, 0.06f, 0.07f};
    env.fog_start = 111.0f;
    env.fog_end = 2222.0f;
    env.cloud_coverage = 0.77f;
    env.grade_power = {1.1f, 1.2f, 1.3f};
    env.fog_volumetric_end = 4321.0f;
    level.environment = env;

    bracken::nav::NavGridConfig navcfg;
    navcfg.origin = {-3200.0f, 0.0f, -3200.0f};
    navcfg.cell_size = 125.0f;
    navcfg.width = 51;
    navcfg.depth = 52;
    navcfg.max_slope_degrees = 38.5f;
    navcfg.max_step = 44.0f;
    level.nav = navcfg;

    return level;
}

void expect_levels_equal(const Level& a, const Level& b) {
    EXPECT_EQ(a.name, b.name);
    EXPECT_EQ(a.content_version, b.content_version);
    EXPECT_EQ(a.models, b.models);
    EXPECT_EQ(a.model_sources, b.model_sources);
    EXPECT_EQ(a.strings, b.strings);

    ASSERT_EQ(a.instances.size(), b.instances.size());
    EXPECT_EQ(0, std::memcmp(a.instances.data(), b.instances.data(),
                             a.instances.size() * sizeof(LevelInstance)));

    ASSERT_EQ(a.markers.size(), b.markers.size());
    EXPECT_EQ(0, std::memcmp(a.markers.data(), b.markers.data(), a.markers.size() * sizeof(LevelMarker)));

    ASSERT_EQ(a.water.size(), b.water.size());
    EXPECT_EQ(0, std::memcmp(a.water.data(), b.water.data(), a.water.size() * sizeof(LevelWater)));

    ASSERT_EQ(a.scatter_models.size(), b.scatter_models.size());
    EXPECT_EQ(0, std::memcmp(a.scatter_models.data(), b.scatter_models.data(),
                             a.scatter_models.size() * sizeof(LevelScatterModel)));
    ASSERT_EQ(a.scatter_rules.size(), b.scatter_rules.size());
    EXPECT_EQ(0, std::memcmp(a.scatter_rules.data(), b.scatter_rules.data(),
                             a.scatter_rules.size() * sizeof(LevelScatterRule)));

    ASSERT_EQ(a.point_lights.size(), b.point_lights.size());
    for (std::size_t i = 0; i < a.point_lights.size(); ++i) {
        const auto& x = a.point_lights[i];
        const auto& y = b.point_lights[i];
        EXPECT_EQ(x.position, y.position);
        EXPECT_EQ(x.radius, y.radius);
        EXPECT_EQ(x.color, y.color);
        EXPECT_EQ(x.intensity, y.intensity);
        EXPECT_EQ(x.casts_shadows, y.casts_shadows);
        EXPECT_EQ(x.stable_id, y.stable_id);
        EXPECT_EQ(x.source_radius, y.source_radius);
    }
    ASSERT_EQ(a.spot_lights.size(), b.spot_lights.size());
    for (std::size_t i = 0; i < a.spot_lights.size(); ++i) {
        const auto& x = a.spot_lights[i];
        const auto& y = b.spot_lights[i];
        EXPECT_EQ(x.position, y.position);
        EXPECT_EQ(x.radius, y.radius);
        EXPECT_EQ(x.direction, y.direction);
        EXPECT_EQ(x.inner_angle.value, y.inner_angle.value);
        EXPECT_EQ(x.outer_angle.value, y.outer_angle.value);
        EXPECT_EQ(x.color, y.color);
        EXPECT_EQ(x.intensity, y.intensity);
        EXPECT_EQ(x.casts_shadows, y.casts_shadows);
        EXPECT_EQ(x.stable_id, y.stable_id);
        EXPECT_EQ(x.source_radius, y.source_radius);
    }

    ASSERT_EQ(a.terrain.has_value(), b.terrain.has_value());
    if (a.terrain.has_value()) {
        EXPECT_EQ(a.terrain->resolution, b.terrain->resolution);
        EXPECT_EQ(a.terrain->origin_x, b.terrain->origin_x);
        EXPECT_EQ(a.terrain->origin_z, b.terrain->origin_z);
        EXPECT_EQ(a.terrain->size, b.terrain->size);
        EXPECT_EQ(a.terrain->min_height, b.terrain->min_height);
        EXPECT_EQ(a.terrain->max_height, b.terrain->max_height);
        EXPECT_EQ(a.terrain->samples, b.terrain->samples);
        EXPECT_EQ(a.terrain->splat_resolution, b.terrain->splat_resolution);
        EXPECT_EQ(a.terrain->splat, b.terrain->splat);
        ASSERT_EQ(a.terrain->grass_exclusions.size(), b.terrain->grass_exclusions.size());
        EXPECT_EQ(0, std::memcmp(a.terrain->grass_exclusions.data(), b.terrain->grass_exclusions.data(),
                                 a.terrain->grass_exclusions.size() * sizeof(LevelRect)));
    }

    ASSERT_EQ(a.environment.has_value(), b.environment.has_value());
    if (a.environment.has_value()) {
        EXPECT_EQ(0, std::memcmp(&*a.environment, &*b.environment,
                                 sizeof(bracken::render::EnvironmentState)));
    }

    ASSERT_EQ(a.nav.has_value(), b.nav.has_value());
    if (a.nav.has_value()) {
        EXPECT_EQ(a.nav->origin, b.nav->origin);
        EXPECT_EQ(a.nav->cell_size, b.nav->cell_size);
        EXPECT_EQ(a.nav->width, b.nav->width);
        EXPECT_EQ(a.nav->depth, b.nav->depth);
        EXPECT_EQ(a.nav->max_slope_degrees, b.nav->max_slope_degrees);
        EXPECT_EQ(a.nav->max_step, b.nav->max_step);
    }
}

/// A hand-built container carrying an instance table of the caller's stride.
/// The only way to test the forward-compatibility rule is to write a file this
/// build's encoder cannot produce.
std::vector<std::byte> hand_built_level(int64_t count, int64_t stride, std::size_t data_bytes,
                                        const std::vector<std::byte>& rows) {
    bracken::save::SaveDocument doc;
    doc.write("level/version", static_cast<int64_t>(k_level_format_version));
    doc.write("level/name", std::string_view("handmade"));
    doc.write("models/count", int64_t{1});
    doc.write("models/0/name", std::string_view("prop_crate"));
    doc.write("models/0/source", std::string_view(""));
    doc.write("instances/count", count);
    doc.write("instances/stride", stride);
    std::vector<std::byte> blob = rows;
    blob.resize(data_bytes);
    doc.write_blob("instances/data", blob);
    return doc.serialize();
}

} // namespace

// ---------------------------------------------------------------------------
// The identity argument
// ---------------------------------------------------------------------------

// If this ever fails, nothing else in this file means anything: the format's
// whole claim is that recomposing three stored floats reproduces the matrix a
// host built with these three glm calls.
TEST(Level, InstanceTransformIsExactlyTheThreeGlmCalls) {
    const glm::vec3 p{123.456f, -78.9f, 0.125f};
    const float yaw = 2.3456f;
    const float scale = 0.75f;

    glm::mat4 expected = glm::translate(glm::mat4(1.0f), p);
    expected = glm::rotate(expected, yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    expected = glm::scale(expected, glm::vec3(scale));

    const glm::mat4 got = level_instance_transform(p, yaw, scale);
    EXPECT_EQ(0, std::memcmp(&expected[0][0], &got[0][0], sizeof(glm::mat4)))
        << "level_instance_transform is no longer the same three calls, in the same order";

    LevelInstance inst;
    inst.position = p;
    inst.yaw = yaw;
    inst.scale = scale;
    const glm::mat4 via_instance = inst.transform();
    EXPECT_EQ(0, std::memcmp(&expected[0][0], &via_instance[0][0], sizeof(glm::mat4)));
}

TEST(Level, RecordStridesArePinned) {
    // The stride IS the on-disk row width. A silent sizeof change makes every
    // existing file unreadable, so it is pinned here as well as in the header.
    EXPECT_EQ(sizeof(LevelInstance), 52u);
    EXPECT_EQ(sizeof(LevelMarker), 20u);
    EXPECT_EQ(sizeof(LevelRect), 16u);
    EXPECT_EQ(sizeof(LevelWater), 124u);
    EXPECT_EQ(sizeof(LevelScatterModel), 16u);
    EXPECT_EQ(sizeof(LevelScatterRule), 48u);
}

// ---------------------------------------------------------------------------
// Round trips
// ---------------------------------------------------------------------------

TEST(Level, RoundTripPreservesEveryField) {
    TempFile f("bracken_level_full.level");
    const Level original = make_full_level();
    ASSERT_TRUE(save_level(original, f.path));

    auto loaded = load_level(f.path);
    ASSERT_TRUE(loaded.is_ok()) << to_string(loaded.error());
    expect_levels_equal(original, loaded.value());
}

TEST(Level, EmptyOptionalsRoundTripAsEmpty) {
    TempFile f("bracken_level_sparse.level");
    Level level;
    level.name = "bare";
    level.add_instance("only_model", {1.0f, 2.0f, 3.0f}, 0.0f, 1.0f);

    ASSERT_TRUE(save_level(level, f.path));
    auto loaded = load_level(f.path);
    ASSERT_TRUE(loaded.is_ok()) << to_string(loaded.error());
    EXPECT_FALSE(loaded.value().terrain.has_value());
    EXPECT_FALSE(loaded.value().environment.has_value());
    EXPECT_FALSE(loaded.value().nav.has_value());
    EXPECT_TRUE(loaded.value().markers.empty());
    EXPECT_TRUE(loaded.value().water.empty());
    EXPECT_TRUE(loaded.value().point_lights.empty());
    EXPECT_EQ(loaded.value().instances.size(), 1u);
}

// The AssetPack path: a packed level arrives as bytes, never as a path.
TEST(Level, SerializeParseMatchesTheFileRoundTrip) {
    TempFile f("bracken_level_bytes.level");
    const Level original = make_full_level();

    ASSERT_TRUE(save_level(original, f.path));
    auto from_file = load_level(f.path);
    ASSERT_TRUE(from_file.is_ok()) << to_string(from_file.error());

    const std::vector<std::byte> bytes = serialize_level(original);
    auto from_bytes = parse_level(bytes);
    ASSERT_TRUE(from_bytes.is_ok()) << to_string(from_bytes.error());

    expect_levels_equal(from_file.value(), from_bytes.value());

    // And the two encoders are literally the same one: the file's bytes are
    // what serialize_level produced.
    std::ifstream in(f.path, std::ios::binary);
    ASSERT_TRUE(in);
    const std::vector<char> on_disk((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    ASSERT_EQ(on_disk.size(), bytes.size());
    EXPECT_EQ(0, std::memcmp(on_disk.data(), bytes.data(), bytes.size()));
}

// The promise SaveDocument makes, extended to the level schema: an older build
// that loads a newer file and re-saves it must not delete what it could not
// read.
TEST(Level, UnknownRecordsSurviveARoundTrip) {
    TempFile f("bracken_level_extras.level");
    Level level = make_full_level();
    level.extras().write("mygame/quest/state", int64_t{42});
    level.extras().write("mygame/quest/label", std::string_view("find the goat"));
    level.extras().write("futureengine/decals/count", int64_t{9});

    ASSERT_TRUE(save_level(level, f.path));
    auto once = load_level(f.path);
    ASSERT_TRUE(once.is_ok()) << to_string(once.error());
    EXPECT_EQ(once.value().extras().read_or<int64_t>("mygame/quest/state", 0), 42);
    EXPECT_EQ(once.value().extras().read_or<std::string>("mygame/quest/label", ""), "find the goat");
    EXPECT_EQ(once.value().extras().read_or<int64_t>("futureengine/decals/count", 0), 9);
    // The schema's own records must NOT leak into extras — otherwise a
    // re-save would carry a stale copy of every table.
    EXPECT_FALSE(once.value().extras().has("instances/count"));
    EXPECT_FALSE(once.value().extras().has("level/version"));

    // Save the loaded level again and confirm nothing was dropped in between.
    TempFile g("bracken_level_extras2.level");
    ASSERT_TRUE(save_level(once.value(), g.path));
    auto twice = load_level(g.path);
    ASSERT_TRUE(twice.is_ok()) << to_string(twice.error());
    EXPECT_EQ(twice.value().extras().read_or<int64_t>("mygame/quest/state", 0), 42);
    EXPECT_EQ(twice.value().extras().read_or<std::string>("mygame/quest/label", ""), "find the goat");
    expect_levels_equal(once.value(), twice.value());
}

// ---------------------------------------------------------------------------
// Forward compatibility: the stride rule
// ---------------------------------------------------------------------------

TEST(Level, ShorterStrideLeavesTheTailAtDefaults) {
    // The first 28 bytes of a 52-byte row: model, motion_id, flags, lod_hint,
    // position — i.e. an older writer that had not yet grown yaw/scale/tint.
    // Those three must keep THIS build's defaults rather than being read from
    // the next row's bytes.
    constexpr int64_t k_short = 28;
    std::vector<std::byte> rows(k_short * 2, std::byte{0});
    const uint32_t model = 0;
    const uint32_t motion = 5;
    const glm::vec3 pos{1.0f, 2.0f, 3.0f};
    std::memcpy(rows.data() + 0, &model, 4);
    std::memcpy(rows.data() + 4, &motion, 4);
    std::memcpy(rows.data() + 16, &pos, 12); // LevelInstance::position lives at 16
    std::memcpy(rows.data() + k_short + 0, &model, 4);

    auto parsed = parse_level(hand_built_level(2, k_short, static_cast<std::size_t>(k_short) * 2, rows));
    ASSERT_TRUE(parsed.is_ok()) << to_string(parsed.error());
    ASSERT_EQ(parsed.value().instances.size(), 2u);

    const LevelInstance& a = parsed.value().instances[0];
    EXPECT_EQ(a.motion_id, 5u);
    EXPECT_EQ(a.position, pos);
    EXPECT_FLOAT_EQ(a.yaw, 0.0f);
    EXPECT_FLOAT_EQ(a.scale, 1.0f) << "the tail past a short stride must stay at its DEFAULT, not zero";
    EXPECT_EQ(a.tint, glm::vec4(1.0f));
    EXPECT_EQ(parsed.value().instances[1].motion_id, 0u);
}

TEST(Level, LongerStrideIsSkippedNotMisread) {
    // A newer writer appended 8 bytes to the row. This build must read its own
    // 52 and step over the extra 8 — not walk off the blob, and not shift.
    constexpr int64_t k_long = 60;
    std::vector<std::byte> rows(k_long * 2, std::byte{0xAB});
    LevelInstance a;
    a.model = 0;
    a.motion_id = 11;
    a.position = {9.0f, 8.0f, 7.0f};
    a.yaw = 1.5f;
    a.scale = 4.0f;
    a.tint = {0.1f, 0.2f, 0.3f, 0.4f};
    LevelInstance b = a;
    b.motion_id = 22;
    b.scale = 9.0f;
    std::memcpy(rows.data(), &a, sizeof(a));
    std::memcpy(rows.data() + k_long, &b, sizeof(b));

    auto parsed = parse_level(hand_built_level(2, k_long, static_cast<std::size_t>(k_long) * 2, rows));
    ASSERT_TRUE(parsed.is_ok()) << to_string(parsed.error());
    ASSERT_EQ(parsed.value().instances.size(), 2u);
    EXPECT_EQ(0, std::memcmp(&parsed.value().instances[0], &a, sizeof(a)));
    EXPECT_EQ(0, std::memcmp(&parsed.value().instances[1], &b, sizeof(b)));
}

TEST(Level, MismatchedTableLengthIsBadTableNotACrash) {
    // 10 rows of 52 bytes declared, 100 bytes supplied.
    auto parsed = parse_level(hand_built_level(10, 52, 100, {}));
    ASSERT_TRUE(parsed.is_err());
    EXPECT_EQ(parsed.error(), LevelError::BadTable);

    // A zero stride is not a table either.
    auto zero_stride = parse_level(hand_built_level(4, 0, 0, {}));
    ASSERT_TRUE(zero_stride.is_err());
    EXPECT_EQ(zero_stride.error(), LevelError::BadTable);
}

// ---------------------------------------------------------------------------
// Refusals — none of these may crash, all of them must name what went wrong
// ---------------------------------------------------------------------------

// The string tables are path-addressed, so their length is a RECORD and
// nothing but the guard bounds it. A checksum-valid file can still claim a
// billion models; that must be a diagnosis, not a bad_alloc.
TEST(Level, AnAbsurdStringTableCountIsRefusedRatherThanAllocated) {
    bracken::save::SaveDocument doc;
    doc.write("level/version", static_cast<int64_t>(k_level_format_version));
    doc.write("models/count", int64_t{1} << 40);
    auto r = parse_level(doc.serialize());
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.error(), LevelError::BadTable);
}

TEST(Level, MissingFileIsFileNotFound) {
    auto r = load_level(temp_path("bracken_level_absent.level"));
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.error(), LevelError::FileNotFound);
}

TEST(Level, NotAContainerIsNotALevelFile) {
    TempFile f("bracken_level_garbage.level");
    {
        std::ofstream out(f.path, std::ios::binary);
        out << "this is not a bracken container at all, not even close";
    }
    auto r = load_level(f.path);
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.error(), LevelError::NotALevelFile);
}

TEST(Level, AContainerWithoutALevelVersionIsNotALevelFile) {
    // A perfectly valid SAVE file. It parses; it is simply not a level.
    TempFile f("bracken_level_savegame.level");
    bracken::save::SaveDocument doc;
    doc.write("player/health", int64_t{87});
    ASSERT_TRUE(doc.save(f.path));

    auto r = load_level(f.path);
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.error(), LevelError::NotALevelFile);
}

TEST(Level, ACorruptPayloadIsRefused) {
    TempFile f("bracken_level_corrupt.level");
    ASSERT_TRUE(save_level(make_full_level(), f.path));

    // Flip one byte deep inside the payload; the checksum must catch it.
    std::vector<char> bytes;
    {
        std::ifstream in(f.path, std::ios::binary);
        bytes.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }
    ASSERT_GT(bytes.size(), 200u);
    bytes[bytes.size() / 2] = static_cast<char>(bytes[bytes.size() / 2] ^ 0x5A);
    {
        std::ofstream out(f.path, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    auto r = load_level(f.path);
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.error(), LevelError::Corrupt);
}

TEST(Level, ATruncatedFileIsRefused) {
    TempFile f("bracken_level_truncated.level");
    ASSERT_TRUE(save_level(make_full_level(), f.path));

    std::vector<char> bytes;
    {
        std::ifstream in(f.path, std::ios::binary);
        bytes.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }
    bytes.resize(bytes.size() / 2);
    {
        std::ofstream out(f.path, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    auto r = load_level(f.path);
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.error(), LevelError::Corrupt);
}

TEST(Level, AFutureSchemaIsRefusedRatherThanGuessedAt) {
    bracken::save::SaveDocument doc;
    doc.write("level/version", static_cast<int64_t>(k_level_format_version) + 1);
    doc.write("level/name", std::string_view("from the future"));
    const std::vector<std::byte> bytes = doc.serialize();

    auto r = parse_level(bytes);
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.error(), LevelError::FutureVersion);
}

TEST(Level, EveryErrorHasAMessage) {
    for (auto e : {LevelError::FileNotFound, LevelError::NotALevelFile, LevelError::Corrupt,
                   LevelError::FutureVersion, LevelError::WriteFailed, LevelError::BadTable,
                   LevelError::BadValue}) {
        EXPECT_STRNE(to_string(e), "unknown");
    }
}

// ---------------------------------------------------------------------------
// Resolution against a ModelManager
// ---------------------------------------------------------------------------

TEST(Level, UnknownModelIsCountedNotSilentlyDropped) {
    ModelManager mm;
    register_all(mm, {"prop_crate", "tree_oak", "procedural_rock"});

    Level level = make_full_level();
    // A fourth name nothing registered — two of the five instances use it.
    level.add_instance("tree_ghost", {0.0f, 0.0f, 0.0f}, 0.0f, 1.0f);
    level.add_instance("tree_ghost", {1.0f, 0.0f, 0.0f}, 0.0f, 1.0f);

    ResolvedLevel resolved;
    const LevelResolveReport report = resolved.resolve(level, mm);

    EXPECT_EQ(report.models_unresolved, 1u);
    EXPECT_EQ(report.instances_dropped_unknown_model, 2u);
    EXPECT_EQ(report.instances_dropped_bad_index, 0u);
    EXPECT_EQ(report.instances_placed, 5u);
    EXPECT_EQ(resolved.size(), 5u);
    EXPECT_TRUE(report.any_dropped());

    const auto missing = missing_models(level, mm);
    ASSERT_EQ(missing.size(), 1u);
    EXPECT_EQ(missing[0].first, "tree_ghost");
}

TEST(Level, AnOutOfRangeModelIndexIsCountedSeparately) {
    ModelManager mm;
    register_all(mm, {"prop_crate"});

    Level level;
    level.add_instance("prop_crate", {0.0f, 0.0f, 0.0f}, 0.0f, 1.0f);
    // Only reachable from a hand-edited or truncated file — but reachable.
    LevelInstance bad;
    bad.model = 99;
    level.instances.push_back(bad);

    ResolvedLevel resolved;
    const LevelResolveReport report = resolved.resolve(level, mm);
    EXPECT_EQ(report.instances_dropped_bad_index, 1u);
    EXPECT_EQ(report.instances_dropped_unknown_model, 0u);
    EXPECT_EQ(report.instances_placed, 1u);
}

TEST(Level, MissingModelsCarriesTheSourceForTheLoader) {
    ModelManager mm;
    register_all(mm, {"prop_crate"});

    Level level = make_full_level();
    const auto missing = missing_models(level, mm);
    ASSERT_EQ(missing.size(), 2u);
    EXPECT_EQ(missing[0].first, "tree_oak");
    EXPECT_EQ(missing[0].second, "models/tree_oak.glb");
    EXPECT_EQ(missing[1].first, "procedural_rock");
    EXPECT_EQ(missing[1].second, "") << "an empty source means the host registers this one itself";
}

TEST(Level, EmitPreservesFileOrder) {
    ModelManager mm;
    register_all(mm, {"prop_crate", "tree_oak", "procedural_rock"});

    const Level level = make_full_level();
    ResolvedLevel resolved;
    ASSERT_EQ(resolved.resolve(level, mm).instances_placed, 5u);

    TestFrame frame;
    resolved.emit(frame.sink());
    frame.seal();

    ASSERT_EQ(frame.draws().size(), resolved.size());
    for (std::size_t i = 0; i < resolved.size(); ++i) {
        const ResolvedLevel::Draw& d = resolved.draws()[i];
        const ModelDrawCommand& c = frame.draws()[i];
        EXPECT_EQ(c.model_handle, d.model);
        EXPECT_EQ(c.motion_id, d.motion_id);
        EXPECT_EQ(c.flags, d.flags);
        EXPECT_EQ(c.tint, d.tint);
        EXPECT_EQ(0, std::memcmp(&c.transform[0][0], &d.transform[0][0], sizeof(glm::mat4)));
    }
}

// ---------------------------------------------------------------------------
// THE HEADLINE
// ---------------------------------------------------------------------------

// A level built in memory, and the same level after a full save/load, must
// describe the same frame — not a similar one. The digest is order-sensitive
// and field-sensitive (see testing/frame_test.test.cpp), so this is a real
// equality and not a coincidence of counts.
//
// ONE ModelManager on purpose. ModelHandle is a registration-order dense
// index, so re-registering between the two frames would be testing handle
// assignment rather than the format.
TEST(Level, FramePacketDigestSurvivesARoundTrip) {
    TempFile f("bracken_level_digest.level");
    ModelManager mm;
    register_all(mm, {"prop_crate", "tree_oak", "procedural_rock"});

    Level authored = make_full_level();
    // Enough rows that an off-by-one in the blob decode cannot hide.
    for (int i = 0; i < 500; ++i) {
        const float t = static_cast<float>(i);
        authored.add_instance(i % 2 == 0 ? "tree_oak" : "prop_crate", {t * 3.5f, t * 0.25f, -t},
                              t * 0.017453292f, 0.5f + t * 0.001f,
                              {0.5f, 0.25f, 1.0f, 1.0f}, i % 3 == 0 ? DrawFlags::NoFog : DrawFlags::None,
                              static_cast<uint32_t>(i % 7));
    }

    ResolvedLevel a;
    ASSERT_EQ(a.resolve(authored, mm).instances_placed, 505u);
    TestFrame frame_a;
    a.emit(frame_a.sink());
    frame_a.seal();

    ASSERT_TRUE(save_level(authored, f.path));
    auto reloaded = load_level(f.path);
    ASSERT_TRUE(reloaded.is_ok()) << to_string(reloaded.error());

    ResolvedLevel b;
    ASSERT_EQ(b.resolve(reloaded.value(), mm).instances_placed, 505u);
    TestFrame frame_b;
    b.emit(frame_b.sink());
    frame_b.seal();

    EXPECT_EQ(frame_a.digest(), frame_b.digest())
        << "a round-tripped level describes a DIFFERENT frame than the one it was written from";
}

// ---------------------------------------------------------------------------
// Scatter rules
// ---------------------------------------------------------------------------

TEST(Level, ScatterIsDeterministicAndSeedSensitive) {
    Level level = make_full_level();
    const LevelScatterRule& rule = level.scatter_rules[0];
    const auto models = level.scatter_slice(rule);
    ASSERT_EQ(models.size(), 2u);

    const auto flat = [](float, float) { return std::optional<float>(100.0f); };
    const std::vector<LevelInstance> first = scatter_ground_cover(rule, models, flat);
    const std::vector<LevelInstance> second = scatter_ground_cover(rule, models, flat);
    ASSERT_FALSE(first.empty());
    ASSERT_EQ(first.size(), second.size());
    EXPECT_EQ(0, std::memcmp(first.data(), second.data(), first.size() * sizeof(LevelInstance)));

    LevelScatterRule other = rule;
    other.seed = rule.seed + 1u;
    const std::vector<LevelInstance> third = scatter_ground_cover(other, models, flat);
    // Sizes MAY differ: the jitter moves candidates across the radius test, so
    // a new seed changes the count as well as the rows.
    const bool differs = third.size() != first.size() ||
                         std::memcmp(first.data(), third.data(), first.size() * sizeof(LevelInstance)) != 0;
    EXPECT_TRUE(differs) << "a different seed produced identical rows";

    // Sink and ground both reach the stored Y.
    for (const LevelInstance& inst : first) {
        EXPECT_FLOAT_EQ(inst.position.y, 100.0f - rule.sink);
        EXPECT_GE(inst.scale, 0.45f);
        EXPECT_LE(inst.scale, 0.90f);
    }
}

// The property that makes a seed worth storing at all: five draws per CELL,
// unconditionally, so a blocked cell does not shift every later one.
TEST(Level, ScatterStreamDoesNotShiftWhenTheGroundRefusesACell) {
    Level level = make_full_level();
    const LevelScatterRule& rule = level.scatter_rules[0];
    const auto models = level.scatter_slice(rule);

    const auto all = [](float, float) { return std::optional<float>(0.0f); };
    // Refuse a band; everything outside it must be untouched.
    const auto banded = [](float x, float) {
        return x > 20.0f ? std::optional<float>() : std::optional<float>(0.0f);
    };

    const std::vector<LevelInstance> full = scatter_ground_cover(rule, models, all);
    const std::vector<LevelInstance> holed = scatter_ground_cover(rule, models, banded);
    ASSERT_FALSE(holed.empty());
    ASSERT_LT(holed.size(), full.size());

    std::size_t matched = 0;
    for (const LevelInstance& kept : holed) {
        for (const LevelInstance& original : full) {
            if (std::memcmp(&kept, &original, sizeof(LevelInstance)) == 0) {
                ++matched;
                break;
            }
        }
    }
    EXPECT_EQ(matched, holed.size())
        << "a rejected cell shifted the RNG stream; the surviving rows are no longer the same rows";
}

TEST(Level, ScatterRefusesAnAbsurdGridInsteadOfHanging) {
    LevelScatterRule rule;
    rule.outer_radius = 1.0e7f;
    rule.step = 0.01f;
    const LevelScatterModel model{0u, 1.0f, 1.0f, 1.0f};
    const std::span<const LevelScatterModel> models(&model, 1);
    EXPECT_TRUE(scatter_ground_cover(rule, models, {}).empty());
}

TEST(Level, ScatterSliceRefusesAnOutOfRangeRule) {
    Level level;
    level.scatter_models.push_back(LevelScatterModel{});
    LevelScatterRule rule;
    rule.model_first = 5;
    rule.model_count = 2;
    EXPECT_TRUE(level.scatter_slice(rule).empty());
}

// ---------------------------------------------------------------------------
// Queries and the dump
// ---------------------------------------------------------------------------

TEST(Level, MarkersAreQueryableByRole) {
    const Level level = make_full_level();
    const auto chimneys = level.markers_named("chimney");
    ASSERT_EQ(chimneys.size(), 2u);
    EXPECT_EQ(chimneys[0].position, glm::vec3(100.0f, 400.0f, 100.0f));
    EXPECT_EQ(chimneys[1].position, glm::vec3(200.0f, 410.0f, 120.0f));

    EXPECT_EQ(level.markers_named("player_spawn").size(), 1u);
    EXPECT_TRUE(level.markers_named("no_such_role").empty());
}

TEST(Level, OutOfRangeTableIndicesAnswerRatherThanAbort) {
    const Level level = make_full_level();
    EXPECT_EQ(level.model_name(0), "prop_crate");
    EXPECT_TRUE(level.model_name(9999).empty());
    EXPECT_TRUE(level.string_at(9999).empty());
}

TEST(Level, TerrainConvertsToAndFromAHeightmap) {
    const Level level = make_full_level();
    const bracken::Heightmap hm = level.terrain->to_heightmap();
    EXPECT_TRUE(hm.valid());
    EXPECT_EQ(hm.resolution, 4u);
    EXPECT_EQ(hm.min_height, -160.0f);
    EXPECT_EQ(hm.height_data, level.terrain->samples);

    LevelTerrain back;
    back.from_heightmap(hm);
    EXPECT_EQ(back.resolution, level.terrain->resolution);
    EXPECT_EQ(back.origin_x, level.terrain->origin_x);
    EXPECT_EQ(back.size, level.terrain->size);
    EXPECT_EQ(back.samples, level.terrain->samples);
}

// A binary format nobody can read is a format nobody debugs. This does not
// pin the wording — it pins that every section is represented.
TEST(Level, DumpNamesEverySection) {
    const std::string text = level_dump(make_full_level());
    EXPECT_NE(text.find("riverside"), std::string::npos);
    EXPECT_NE(text.find("prop_crate"), std::string::npos);
    EXPECT_NE(text.find("player_spawn"), std::string::npos);
    EXPECT_NE(text.find("scatter rule"), std::string::npos);
    EXPECT_NE(text.find("terrain:"), std::string::npos);
    EXPECT_NE(text.find("environment:"), std::string::npos);
    EXPECT_NE(text.find("nav:"), std::string::npos);
    EXPECT_NE(text.find("instances"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Hostile input
//
// A `.level` is untrusted the moment it can arrive from anywhere but the build
// that wrote it. The contract these pin, stated in level.hpp: never a crash,
// never an allocation on a number the file made up, always a named LevelError.
//
// Every test below builds its bad file through the REAL encoder or the real
// container, so none of them can pass by accident against a decoder that
// simply refuses everything — the good-file cases above prove it accepts.
// ---------------------------------------------------------------------------

// The whole point of BadValue: a NaN encodes perfectly well and is refused on
// the way back in, before it can become a matrix.
TEST(Level, ANaNInAnInstanceIsRefused) {
    Level level;
    level.add_instance("prop_crate", {0.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f}, 0.0f, 1.0f);
    auto r = parse_level(serialize_level(level));
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.error(), LevelError::BadValue);

    // An infinity is the same class of poison and is refused the same way.
    Level infinite;
    infinite.add_instance("prop_crate", {0.0f, 0.0f, 0.0f}, 0.0f,
                          std::numeric_limits<float>::infinity());
    auto inf = parse_level(serialize_level(infinite));
    ASSERT_TRUE(inf.is_err());
    EXPECT_EQ(inf.error(), LevelError::BadValue);

    // ... and a finite level of the same shape is accepted, so the refusal is
    // about the value and not about the shape.
    Level good;
    good.add_instance("prop_crate", {0.0f, 1.0f, 0.0f}, 0.0f, 1.0f);
    EXPECT_TRUE(parse_level(serialize_level(good)).is_ok());
}

TEST(Level, ANaNAnywhereElseIsRefusedToo) {
    const float nan = std::numeric_limits<float>::quiet_NaN();

    {
        Level level = make_full_level();
        level.environment->sun_dir.y = nan;
        auto r = parse_level(serialize_level(level));
        ASSERT_TRUE(r.is_err());
        EXPECT_EQ(r.error(), LevelError::BadValue) << "a NaN sun direction reached the frame";
    }
    {
        Level level = make_full_level();
        level.terrain->origin_x = nan;
        auto r = parse_level(serialize_level(level));
        ASSERT_TRUE(r.is_err());
        EXPECT_EQ(r.error(), LevelError::BadValue);
    }
    {
        Level level = make_full_level();
        level.nav->cell_size = nan;
        auto r = parse_level(serialize_level(level));
        ASSERT_TRUE(r.is_err());
        EXPECT_EQ(r.error(), LevelError::BadValue);
    }
    {
        Level level = make_full_level();
        level.point_lights[0].intensity = nan;
        auto r = parse_level(serialize_level(level));
        ASSERT_TRUE(r.is_err());
        EXPECT_EQ(r.error(), LevelError::BadValue);
    }
    {
        Level level = make_full_level();
        level.water[0].wave_damping = nan;
        auto r = parse_level(serialize_level(level));
        ASSERT_TRUE(r.is_err());
        EXPECT_EQ(r.error(), LevelError::BadValue);
    }
    {
        Level level = make_full_level();
        level.markers[1].position.z = nan;
        auto r = parse_level(serialize_level(level));
        ASSERT_TRUE(r.is_err());
        EXPECT_EQ(r.error(), LevelError::BadValue);
    }
    {
        Level level = make_full_level();
        level.scatter_rules[0].step = nan;
        auto r = parse_level(serialize_level(level));
        ASSERT_TRUE(r.is_err());
        EXPECT_EQ(r.error(), LevelError::BadValue);
    }
}

// A model INDEX that happens to look like a NaN when read as a float must not
// be refused — which is why the finiteness checks are per-field and not a
// blanket byte scan of the row.
TEST(Level, AnIntegerFieldIsNotMistakenForANaN) {
    Level level;
    level.models.assign(1, "prop_crate");
    level.model_sources.assign(1, "");
    LevelInstance inst;
    inst.model = 0;
    // 0x7FC00001 is a quiet NaN bit pattern. As a motion id it is just a
    // number, and a decoder that scanned the row as floats would reject it.
    inst.motion_id = 0x7FC00001u;
    inst.lod_hint = 0xFF800000u; // -infinity's bit pattern
    inst.flags = DrawFlags::NoFog;
    level.instances.push_back(inst);

    auto r = parse_level(serialize_level(level));
    ASSERT_TRUE(r.is_ok()) << to_string(r.error());
    EXPECT_EQ(r.value().instances[0].motion_id, 0x7FC00001u);
    EXPECT_EQ(r.value().instances[0].lod_hint, 0xFF800000u);
}

// The heightmap's header and its blob must agree, because `Heightmap` indexes
// at `z * resolution + x` and a resolution larger than the data is an
// out-of-bounds read at the first sample.
TEST(Level, TerrainThatContradictsItsOwnResolutionIsRefused) {
    {
        Level level = make_full_level();
        level.terrain->resolution = 64; // 16 samples on hand
        auto r = parse_level(serialize_level(level));
        ASSERT_TRUE(r.is_err());
        EXPECT_EQ(r.error(), LevelError::BadTable);
    }
    {
        Level level = make_full_level();
        level.terrain->splat_resolution = 9; // 16 bytes on hand, wants 324
        auto r = parse_level(serialize_level(level));
        ASSERT_TRUE(r.is_err());
        EXPECT_EQ(r.error(), LevelError::BadTable);
    }
    {
        // An odd number of sample BYTES is not a whole number of samples.
        bracken::save::SaveDocument doc;
        doc.write("level/version", static_cast<int64_t>(k_level_format_version));
        doc.write("terrain/present", true);
        doc.write("terrain/resolution", int64_t{2});
        const std::array<std::byte, 7> odd{};
        doc.write_blob("terrain/samples", odd);
        auto r = parse_level(doc.serialize());
        ASSERT_TRUE(r.is_err());
        EXPECT_EQ(r.error(), LevelError::BadTable);
    }
}

TEST(Level, AnAbsurdTerrainResolutionIsRefusedRatherThanSquared) {
    for (const int64_t resolution : {int64_t{-1}, int64_t{k_max_terrain_resolution} + 1, int64_t{1} << 40}) {
        bracken::save::SaveDocument doc;
        doc.write("level/version", static_cast<int64_t>(k_level_format_version));
        doc.write("terrain/present", true);
        doc.write("terrain/resolution", resolution);
        auto r = parse_level(doc.serialize());
        ASSERT_TRUE(r.is_err()) << "resolution " << resolution << " was accepted";
        EXPECT_EQ(r.error(), LevelError::BadTable);
    }
}

// NavGrid::build allocates width * depth cells, so a file's word on the matter
// is the one number that must not be believed.
TEST(Level, AnAbsurdNavGridIsRefused) {
    struct Case {
        int64_t width;
        int64_t depth;
    };
    // The last pair is the overflow case: multiplied as int64 it wraps, so a
    // bounds check written as `width * depth > cap` alone would ACCEPT it.
    for (const Case c : {Case{-1, 10}, Case{10, -1}, Case{1 << 20, 1 << 20}, Case{1LL << 40, 1LL << 40}}) {
        bracken::save::SaveDocument doc;
        doc.write("level/version", static_cast<int64_t>(k_level_format_version));
        doc.write("nav/present", true);
        doc.write("nav/width", c.width);
        doc.write("nav/depth", c.depth);
        auto r = parse_level(doc.serialize());
        ASSERT_TRUE(r.is_err()) << "nav " << c.width << "x" << c.depth << " was accepted";
        EXPECT_EQ(r.error(), LevelError::BadTable);
    }
}

// The tight bound on the two path-addressed tables: a table of N names needs 2N
// records to exist, so a count larger than the whole document is a lie the
// file's own size disproves. Without this, a 200-byte file can ask for a
// gigabyte of empty strings.
TEST(Level, ANameCountLargerThanTheDocumentIsRefused) {
    bracken::save::SaveDocument doc;
    doc.write("level/version", static_cast<int64_t>(k_level_format_version));
    doc.write("models/count", int64_t{4096}); // three records in the whole file
    auto r = parse_level(doc.serialize());
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.error(), LevelError::BadTable);

    bracken::save::SaveDocument strings;
    strings.write("level/version", static_cast<int64_t>(k_level_format_version));
    strings.write("strings/count", int64_t{k_max_name_table_rows} + 1);
    auto s = parse_level(strings.serialize());
    ASSERT_TRUE(s.is_err());
    EXPECT_EQ(s.error(), LevelError::BadTable);
}

// The amplification bound. The blob's length alone only bounds the row count by
// `length / stride`, and the stride comes from the file too — a declared stride
// of 1 turns a small blob into millions of 52-byte rows.
TEST(Level, ATinyStrideCannotAmplifyASmallBlobIntoAHugeTable) {
    constexpr int64_t k_rows = int64_t{k_max_table_rows} + 1;
    auto parsed = parse_level(hand_built_level(k_rows, 1, static_cast<std::size_t>(k_rows), {}));
    ASSERT_TRUE(parsed.is_err());
    EXPECT_EQ(parsed.error(), LevelError::BadTable);

    // The same shape one row under the ceiling is structurally fine — the
    // refusal is about the ceiling, not about short strides, which are the
    // format's forward-compatibility mechanism.
    constexpr int64_t k_ok_rows = 64;
    auto small = parse_level(hand_built_level(k_ok_rows, 1, static_cast<std::size_t>(k_ok_rows), {}));
    ASSERT_TRUE(small.is_ok()) << to_string(small.error());
    EXPECT_EQ(small.value().instances.size(), static_cast<std::size_t>(k_ok_rows));
}

// The blunt instrument, and the one that finds what a case-by-case test does
// not: EVERY prefix of a valid file must be refused with a named error, and
// none of them may read out of bounds (run this under ASan and it is the whole
// fuzz corpus a format this small needs).
TEST(Level, EveryTruncationOfAValidLevelIsRefusedRatherThanCrashing) {
    const std::vector<std::byte> whole = serialize_level(make_full_level());
    ASSERT_GT(whole.size(), 512u);

    for (std::size_t n = 0; n < whole.size(); ++n) {
        auto r = parse_level(std::span<const std::byte>(whole.data(), n));
        ASSERT_TRUE(r.is_err()) << "a " << n << "-byte prefix of a " << whole.size()
                                << "-byte level parsed as a complete level";
        // Which error is not the point — that it IS an error, every time, is.
        EXPECT_TRUE(r.error() == LevelError::NotALevelFile || r.error() == LevelError::Corrupt ||
                    r.error() == LevelError::FutureVersion || r.error() == LevelError::BadTable ||
                    r.error() == LevelError::BadValue)
            << "prefix " << n << " produced an unexpected error";
    }
    EXPECT_TRUE(parse_level(whole).is_ok());
}

// Every single-byte corruption must be caught. The payload checksum is what
// does it; the header cases fall out as NotASaveFile / FutureVersion / Corrupt.
TEST(Level, EverySingleByteCorruptionIsRefused) {
    const std::vector<std::byte> whole = serialize_level(make_full_level());
    ASSERT_GT(whole.size(), 512u);

    // Bytes 12..15 are the container's `game_version` — host metadata it
    // round-trips and deliberately does not validate, so corrupting one
    // produces a level that loads with a different content_version and no
    // error. That is the format's stated behaviour, not a hole in this sweep.
    constexpr std::size_t k_game_version_offset = 12;

    // Every 17th byte: a full sweep is tens of thousands of parses of a
    // 500-record document and buys nothing a coprime stride does not.
    for (std::size_t i = 0; i < whole.size(); i += 17) {
        if (i >= k_game_version_offset && i < k_game_version_offset + 4) {
            continue;
        }
        std::vector<std::byte> damaged = whole;
        damaged[i] = static_cast<std::byte>(std::to_integer<uint8_t>(damaged[i]) ^ 0x5Au);
        auto r = parse_level(damaged);
        EXPECT_TRUE(r.is_err()) << "flipping byte " << i << " of " << whole.size()
                                << " produced a level that loaded cleanly";
    }
}

// Indices into the two name tables are the format's only "pointers". None of
// them may be followed off the end — a level whose instance names model 4000
// of 2 is counted as a drop, and a marker naming string 9999 answers "".
TEST(Level, OutOfRangeIndicesAreRefusedByTheReaderNotFollowed) {
    Level level;
    (void)level.intern_model("prop_crate");
    (void)level.intern_string("spawn");

    LevelInstance inst;
    inst.model = 4000u;
    level.instances.push_back(inst);
    LevelMarker marker;
    marker.name = 9999u;
    level.markers.push_back(marker);
    LevelScatterRule rule;
    rule.name = 12345u;
    rule.model_first = 77u;
    rule.model_count = 88u;
    level.scatter_rules.push_back(rule);

    auto r = parse_level(serialize_level(level));
    ASSERT_TRUE(r.is_ok()) << to_string(r.error());
    const Level& back = r.value();

    EXPECT_TRUE(back.model_name(back.instances[0].model).empty());
    EXPECT_TRUE(back.string_at(back.markers[0].name).empty());
    EXPECT_TRUE(back.scatter_slice(back.scatter_rules[0]).empty());

    ModelManager mm;
    register_all(mm, {"prop_crate"});
    ResolvedLevel resolved;
    const LevelResolveReport report = resolved.resolve(back, mm);
    EXPECT_EQ(report.instances_placed, 0u);
    EXPECT_EQ(report.instances_dropped_bad_index, 1u);
    EXPECT_TRUE(resolved.draws().empty());

    // And the dump — the thing you reach for when a file is already suspect —
    // must survive all of it.
    EXPECT_FALSE(level_dump(back, 0).empty());
}

// The output bound. The cells-per-axis ceiling alone still permits 8192^2
// candidates; without this a rule whose ground accepts everything allocates
// until the process dies.
TEST(Level, ScatterOutputIsCappedRatherThanUnbounded) {
    LevelScatterRule rule;
    rule.outer_radius = 600.0f;
    rule.step = 1.0f;  // 1200 cells per axis, ~1.13M inside the circle
    rule.jitter = 0.0f;
    const LevelScatterModel model{0u, 1.0f, 1.0f, 1.0f};
    const std::span<const LevelScatterModel> models(&model, 1);

    const auto flat = [](float, float) { return std::optional<float>(0.0f); };
    const std::vector<LevelInstance> out = scatter_ground_cover(rule, models, flat);
    EXPECT_EQ(out.size(), static_cast<std::size_t>(k_max_scatter_instances));
}

// ---------------------------------------------------------------------------
// The engine-side exporter surface
// ---------------------------------------------------------------------------

TEST(Level, DigestIsFieldSensitiveAndIgnoresHostExtras) {
    const Level base = make_full_level();
    const uint64_t reference = level_digest(base);
    EXPECT_EQ(reference, level_digest(make_full_level())) << "the digest is not a function of the content";

    const auto moved = [&](auto&& mutate) {
        Level copy = base;
        mutate(copy);
        return level_digest(copy) != reference;
    };
    EXPECT_TRUE(moved([](Level& l) { l.instances[2].position.x += 0.001f; }));
    EXPECT_TRUE(moved([](Level& l) { l.instances[2].tint.a = 0.5f; }));
    EXPECT_TRUE(moved([](Level& l) { l.instances[0].flags = DrawFlags::None; }));
    EXPECT_TRUE(moved([](Level& l) { std::swap(l.instances[0], l.instances[1]); }))
        << "the digest is order-insensitive, so it cannot prove draw order survived";
    EXPECT_TRUE(moved([](Level& l) { l.name = "elsewhere"; }));
    EXPECT_TRUE(moved([](Level& l) { l.models[0] = "prop_barrel"; }));
    EXPECT_TRUE(moved([](Level& l) { l.model_sources[0] = "other.glb"; }));
    EXPECT_TRUE(moved([](Level& l) { l.markers[0].yaw += 0.1f; }));
    EXPECT_TRUE(moved([](Level& l) { l.point_lights[0].casts_shadows = false; }));
    EXPECT_TRUE(moved([](Level& l) { l.spot_lights[0].outer_angle.value += 1.0f; }));
    EXPECT_TRUE(moved([](Level& l) { l.water[0].wave_speed += 1.0f; }));
    EXPECT_TRUE(moved([](Level& l) { l.terrain->samples[3] += 1; }));
    EXPECT_TRUE(moved([](Level& l) { l.terrain->splat[3] += 1; }));
    EXPECT_TRUE(moved([](Level& l) { l.environment->cloud_density += 0.01f; }));
    EXPECT_TRUE(moved([](Level& l) { l.nav->width += 1; }));
    EXPECT_TRUE(moved([](Level& l) { l.scatter_rules[0].seed += 1; }));
    EXPECT_TRUE(moved([](Level& l) { l.terrain.reset(); }));

    // extras() is the HOST's channel. A game that writes a frame counter into
    // it must not make every level unequal to itself.
    Level with_extras = base;
    with_extras.extras().write("mygame/frame", int64_t{12345});
    EXPECT_EQ(level_digest(with_extras), reference);
}

TEST(Level, VerifyRoundTripAcceptsAGoodLevelAndNamesABadOne) {
    Level level = make_full_level();
    level.extras().write("mygame/checkpoint", std::string_view("bridge"));
    auto ok = verify_level_roundtrip(level);
    EXPECT_TRUE(ok) << to_string(ok.error());

    // The case this exists to catch at EXPORT time rather than at load time.
    Level poisoned = make_full_level();
    poisoned.instances[1].position.z = std::numeric_limits<float>::quiet_NaN();
    auto bad = verify_level_roundtrip(poisoned);
    ASSERT_TRUE(bad.is_err());
    EXPECT_EQ(bad.error(), LevelError::BadValue);

    // A terrain header that disagrees with its own blob is likewise an
    // export-time failure, not a mystery in the shipped build.
    Level inconsistent = make_full_level();
    inconsistent.terrain->resolution = 5;
    auto table = verify_level_roundtrip(inconsistent);
    ASSERT_TRUE(table.is_err());
    EXPECT_EQ(table.error(), LevelError::BadTable);
}

// Host records survive; host records that SQUAT on the schema's own prefixes do
// not, because the encoder would otherwise leave `models/9/name` in a
// three-model file for the next reader to trip over.
TEST(Level, HostRecordsUnderASchemaPrefixAreDroppedNotSmuggled) {
    Level level = make_full_level();
    level.extras().write("mygame/save_slot", int64_t{3});
    level.extras().write("instances/count", int64_t{999999});
    level.extras().write("models/9/name", std::string_view("ghost"));

    auto r = parse_level(serialize_level(level));
    ASSERT_TRUE(r.is_ok()) << to_string(r.error());
    const Level& back = r.value();

    EXPECT_EQ(back.instances.size(), level.instances.size()) << "a squatted record overrode the schema";
    EXPECT_EQ(back.models.size(), level.models.size());
    EXPECT_TRUE(back.extras().has("mygame/save_slot"));
    EXPECT_FALSE(back.extras().has("instances/count"));
    EXPECT_FALSE(back.extras().has("models/9/name"));
    EXPECT_EQ(back.extras().record_count(), 1u);
}

TEST(Level, WaterConvertsBothWaysFieldForField) {
    // Every field perturbed away from its default: a conversion that forgot one
    // would otherwise pass by inheriting the same default on both sides.
    const LevelWater original = make_full_level().water[0];
    const bracken::render::WaterRegion region = to_water_region(original);
    const LevelWater back = to_level_water(region);
    EXPECT_EQ(0, std::memcmp(&original, &back, sizeof(LevelWater)))
        << "to_water_region / to_level_water lost or transposed a field";

    // Spot-check the two that are easiest to transpose.
    EXPECT_EQ(region.min_z, original.min_z);
    EXPECT_EQ(region.max_x, original.max_x);
    EXPECT_EQ(region.sim_center, original.sim_center);
}

// A tripwire for a field ADDED to EnvironmentState without a record in
// write_env/read_env: the struct is float-only, so filling every float slot
// with a distinct value catches a new member that nobody wired up.
TEST(Level, EveryEnvironmentFloatSurvivesTheRoundTrip) {
    constexpr std::size_t k_floats = sizeof(bracken::render::EnvironmentState) / sizeof(float);
    static_assert(sizeof(bracken::render::EnvironmentState) % sizeof(float) == 0,
                  "EnvironmentState gained a member that is not a float; this test needs revisiting");

    std::array<float, k_floats> pattern{};
    for (std::size_t i = 0; i < k_floats; ++i) {
        pattern[i] = 0.125f + static_cast<float>(i) * 1.5f; // exact in binary32
    }
    // Written slot by slot through a byte pointer: EnvironmentState has default
    // member initializers, so a whole-struct memcpy into it is a
    // -Wclass-memaccess error even though its layout is exactly these floats.
    bracken::render::EnvironmentState env;
    auto* slots = reinterpret_cast<unsigned char*>(&env);
    for (std::size_t i = 0; i < k_floats; ++i) {
        std::memcpy(slots + i * sizeof(float), &pattern[i], sizeof(float));
    }

    Level level;
    level.environment = env;
    auto r = parse_level(serialize_level(level));
    ASSERT_TRUE(r.is_ok()) << to_string(r.error());
    ASSERT_TRUE(r.value().environment.has_value());
    EXPECT_EQ(0, std::memcmp(&env, &*r.value().environment, sizeof(env)))
        << "an EnvironmentState field is not carried by write_env/read_env";
}

// ---------------------------------------------------------------------------
// Inspection
// ---------------------------------------------------------------------------

TEST(Level, DumpShowsEveryRowWhenAskedAndTruncatesByDefault) {
    Level level = make_full_level();
    for (int i = 0; i < 40; ++i) {
        level.add_instance("prop_crate", {static_cast<float>(i), 0.0f, 0.0f}, 0.0f, 1.0f);
    }
    level.extras().write("mygame/save_slot", int64_t{3});

    const std::string preview = level_dump(level);
    const std::string full = level_dump(level, 0);
    EXPECT_LT(preview.size(), full.size());
    EXPECT_NE(preview.find("showing 8 of 45"), std::string::npos) << preview.substr(0, 400);
    EXPECT_NE(full.find("showing 45 of 45"), std::string::npos);

    // Every section a reader needs to answer "what is in this file".
    for (const char* needle : {"riverside", "digest 0x", "prop_crate", "player_spawn", "scatter rule",
                               "terrain:", "environment:", "nav:", "point [", "spot  [", "water [0]",
                               "mygame/save_slot"}) {
        EXPECT_NE(full.find(needle), std::string::npos) << "the dump does not mention " << needle;
    }
}

TEST(Level, DumpOfAnEmptyLevelIsStillAnswerable) {
    const std::string text = level_dump(Level{});
    EXPECT_NE(text.find("instances     0"), std::string::npos);
    EXPECT_EQ(text.find("terrain:"), std::string::npos);
    EXPECT_EQ(text.find("extras"), text.rfind("extras")); // the counts line only
}

// The exporter entry point: the ORDER is the policy it owns. A level that
// cannot survive its own encoder must not become a file, so a failed
// verification must leave nothing on disk.
TEST(Level, ExportVerifiesBeforeItWritesAndDropsTheSidecar) {
    {
        TempFile f("bracken_level_export_ok.level");
        std::error_code sidecar_ec;
        const Level level = make_full_level();
        auto r = export_level(level, f.path);
        ASSERT_TRUE(r) << to_string(r.error());
        EXPECT_TRUE(std::filesystem::exists(f.path));
        EXPECT_TRUE(std::filesystem::exists(f.path + ".txt"));
        auto back = load_level(f.path);
        ASSERT_TRUE(back.is_ok()) << to_string(back.error());
        EXPECT_EQ(level_digest(back.value()), level_digest(level));
        std::filesystem::remove(f.path + ".txt", sidecar_ec);
    }
    {
        TempFile f("bracken_level_export_bad.level");
        Level poisoned = make_full_level();
        poisoned.instances[0].scale = std::numeric_limits<float>::quiet_NaN();
        auto r = export_level(poisoned, f.path);
        ASSERT_TRUE(r.is_err());
        EXPECT_EQ(r.error(), LevelError::BadValue);
        EXPECT_FALSE(std::filesystem::exists(f.path))
            << "a level that fails verification still reached the disk";
    }
    {
        // The sidecar is opt-out, and skipping it is not an error.
        TempFile f("bracken_level_export_nodump.level");
        std::error_code stale_ec;
        std::filesystem::remove(f.path + ".txt", stale_ec); // left by a previous run
        auto r = export_level(make_full_level(), f.path, /*write_dump=*/false);
        ASSERT_TRUE(r) << to_string(r.error());
        EXPECT_TRUE(std::filesystem::exists(f.path));
        EXPECT_FALSE(std::filesystem::exists(f.path + ".txt"));
    }
}

TEST(Level, DumpSidecarIsWrittenAndReadable) {
    TempFile f("bracken_level_dump.txt");
    const Level level = make_full_level();
    auto r = save_level_dump(level, f.path);
    ASSERT_TRUE(r) << to_string(r.error());

    std::ifstream in(f.path);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(text, level_dump(level, 0)) << "the sidecar is not the full dump";

    // A path that cannot be opened is a WriteFailed, not an abort.
    auto bad = save_level_dump(level, temp_path("no_such_directory_here/level.txt"));
    ASSERT_TRUE(bad.is_err());
    EXPECT_EQ(bad.error(), LevelError::WriteFailed);
}
