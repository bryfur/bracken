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
#include "bracken/save/save_document.hpp"
#include "bracken/scene/level.hpp"
#include "bracken/testing/frame_test.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <string>
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
                   LevelError::FutureVersion, LevelError::WriteFailed, LevelError::BadTable}) {
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
