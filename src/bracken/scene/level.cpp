#include "bracken/scene/level.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_stage.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/scene/frame_sink.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <glm/gtc/matrix_transform.hpp>

namespace bracken::scene {
namespace {

constexpr const char* LOG_TAG = "level";

/// Everything under these belongs to the level schema and is regenerated on
/// every save. A record whose path starts with none of them is carried
/// verbatim in Level::extras(). Kept as one list so the save side and the
/// load side cannot disagree about what "mine" means.
constexpr std::string_view k_owned_prefixes[] = {
    "level/", "models/", "strings/", "instances/", "markers/", "lights/",
    "water/", "terrain/", "env/",    "nav/",       "scatter/",
};

[[nodiscard]] bool is_owned_path(std::string_view path) noexcept {
    for (std::string_view prefix : k_owned_prefixes) {
        if (path.size() >= prefix.size() && path.compare(0, prefix.size(), prefix) == 0) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Blob tables
//
// SaveDocument::find is a linear scan over a vector, so every write is O(n) in
// the records already present. Path-addressing 2550 instances at five records
// each is ~81M string comparisons on write; one blob with an explicit stride is
// three. That is not an optimization, it is the reason the schema looks the way
// it does — see docs/gap_level.md.
// ---------------------------------------------------------------------------

template <typename T>
void write_table(save::SaveDocument& doc, std::string_view prefix, const std::vector<T>& rows) {
    static_assert(std::is_trivially_copyable_v<T>, "table rows are memcpy'd");
    doc.write(std::format("{}/count", prefix), static_cast<int64_t>(rows.size()));
    doc.write(std::format("{}/stride", prefix), static_cast<int64_t>(sizeof(T)));
    const auto* bytes = reinterpret_cast<const std::byte*>(rows.data());
    doc.write_blob(std::format("{}/data", prefix),
                   std::span<const std::byte>(bytes, rows.size() * sizeof(T)));
}

/**
 * Decode a blob table. Returns false for BadTable — never reads past the blob.
 *
 * THE FORWARD-COMPATIBILITY RULE lives here: each row copies
 * `min(stride, sizeof(T))` bytes and leaves the rest of the row at its
 * default. A newer writer that APPENDED a field is readable (the tail is
 * skipped); an older file with a shorter stride is readable (the tail keeps
 * this build's defaults). Both only work because rows are append-only and
 * padding-free, which the static_asserts in level.hpp pin.
 */
template <typename T>
[[nodiscard]] bool read_table(const save::SaveDocument& doc, std::string_view prefix, std::vector<T>& out) {
    out.clear();
    const std::string count_key = std::format("{}/count", prefix);
    if (!doc.has(count_key)) {
        return true; // an absent table is an empty table, not a corrupt one
    }
    int64_t count = 0;
    int64_t stride = 0;
    (void)doc.read(count_key, count);
    (void)doc.read(std::format("{}/stride", prefix), stride);
    std::vector<std::byte> data;
    (void)doc.read_blob(std::format("{}/data", prefix), data);

    if (count < 0 || stride <= 0) {
        BRACKEN_LOG_ERROR(LOG_TAG, "table '{}': count {} / stride {} is not a table", prefix, count, stride);
        return false;
    }
    if (count == 0) {
        return data.empty();
    }
    const auto ustride = static_cast<uint64_t>(stride);
    // Overflow-free: never form count * stride, divide instead.
    if (data.size() % ustride != 0 || static_cast<uint64_t>(count) != data.size() / ustride) {
        BRACKEN_LOG_ERROR(LOG_TAG, "table '{}': {} rows x {} bytes does not match a {}-byte blob", prefix,
                          count, stride, data.size());
        return false;
    }

    const auto copy = static_cast<std::size_t>(std::min<uint64_t>(ustride, sizeof(T)));
    out.assign(static_cast<std::size_t>(count), T{});
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::memcpy(&out[i], data.data() + i * static_cast<std::size_t>(ustride), copy);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Light rows
//
// PointLightDesc/SpotLightDesc carry a `bool` and therefore PADDING, and
// padding bytes are unspecified after a copy. memcpy'ing them into a blob
// would make two saves of the same level differ in bytes (and in checksum) for
// no reason a reader could see, so lights go through explicit all-scalar rows.
// The cone half-angles stay in DEGREES on disk, matching core::Degrees — the
// engine takes the cosine, and storing the cosine here would make the file
// unreadable to a human and unrepresentable back to a Degrees.
// ---------------------------------------------------------------------------

struct PointLightRow {
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    float radius = 0.0f;
    float cr = 1.0f, cg = 1.0f, cb = 1.0f;
    float intensity = 0.0f;
    float source_radius = 0.0f;
    uint32_t stable_id = 0;
    uint32_t casts_shadows = 0;
};
static_assert(sizeof(PointLightRow) == 44 && std::is_trivially_copyable_v<PointLightRow>);

struct SpotLightRow {
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    float radius = 0.0f;
    float dx = 0.0f, dy = -1.0f, dz = 0.0f;
    float inner_deg = 0.0f, outer_deg = 0.0f;
    float cr = 1.0f, cg = 1.0f, cb = 1.0f;
    float intensity = 0.0f;
    float source_radius = 0.0f;
    uint32_t stable_id = 0;
    uint32_t casts_shadows = 0;
};
static_assert(sizeof(SpotLightRow) == 64 && std::is_trivially_copyable_v<SpotLightRow>);

[[nodiscard]] PointLightRow to_row(const bracken::render::lighting::PointLightDesc& l) {
    return PointLightRow{l.position.x, l.position.y, l.position.z, l.radius, l.color.r,
                         l.color.g,    l.color.b,    l.intensity,  l.source_radius,
                         l.stable_id,  l.casts_shadows ? 1u : 0u};
}

[[nodiscard]] bracken::render::lighting::PointLightDesc from_row(const PointLightRow& r) {
    bracken::render::lighting::PointLightDesc l;
    l.position = {r.px, r.py, r.pz};
    l.radius = r.radius;
    l.color = {r.cr, r.cg, r.cb};
    l.intensity = r.intensity;
    l.casts_shadows = r.casts_shadows != 0;
    l.stable_id = r.stable_id;
    l.source_radius = r.source_radius;
    return l;
}

[[nodiscard]] SpotLightRow to_row(const bracken::render::lighting::SpotLightDesc& l) {
    return SpotLightRow{l.position.x,          l.position.y,          l.position.z, l.radius,
                        l.direction.x,         l.direction.y,         l.direction.z,
                        l.inner_angle.value,   l.outer_angle.value,   l.color.r,    l.color.g,
                        l.color.b,             l.intensity,           l.source_radius,
                        l.stable_id,           l.casts_shadows ? 1u : 0u};
}

[[nodiscard]] bracken::render::lighting::SpotLightDesc from_row(const SpotLightRow& r) {
    bracken::render::lighting::SpotLightDesc l;
    l.position = {r.px, r.py, r.pz};
    l.radius = r.radius;
    l.direction = {r.dx, r.dy, r.dz};
    l.inner_angle = core::Degrees{r.inner_deg};
    l.outer_angle = core::Degrees{r.outer_deg};
    l.color = {r.cr, r.cg, r.cb};
    l.intensity = r.intensity;
    l.casts_shadows = r.casts_shadows != 0;
    l.stable_id = r.stable_id;
    l.source_radius = r.source_radius;
    return l;
}

// ---------------------------------------------------------------------------
// Environment: one record per field, path-addressed.
//
// 29 records rather than one blob, deliberately: EnvironmentState is the one
// section a host edits by hand in a dump, it grows a field every few releases,
// and path-addressing is what makes a field added later round-trip through an
// older build by KEY. It is also small enough that the linear-scan cost does
// not matter.
// ---------------------------------------------------------------------------

void write_env(save::SaveDocument& doc, const bracken::render::EnvironmentState& e) {
    doc.write("env/present", true);
    doc.write("env/sun_dir", e.sun_dir);
    doc.write("env/sun_color", e.sun_color);
    doc.write("env/sky_zenith", e.sky_zenith);
    doc.write("env/sky_horizon", e.sky_horizon);
    doc.write("env/ground_color", e.ground_color);
    doc.write("env/ambient_sky", e.ambient_sky);
    doc.write("env/ambient_ground", e.ambient_ground);
    doc.write("env/fog_color_horizon", e.fog_color_horizon);
    doc.write("env/fog_color_sun", e.fog_color_sun);
    doc.write("env/fog_start", e.fog_start);
    doc.write("env/fog_end", e.fog_end);
    doc.write("env/fog_height_falloff", e.fog_height_falloff);
    doc.write("env/cloud_coverage", e.cloud_coverage);
    doc.write("env/cloud_density", e.cloud_density);
    doc.write("env/cloud_scale", e.cloud_scale);
    doc.write("env/cloud_speed", e.cloud_speed);
    doc.write("env/haze_strength", e.haze_strength);
    doc.write("env/fog_volumetric_sun_color", e.fog_volumetric_sun_color);
    doc.write("env/fog_volumetric_scatter", e.fog_volumetric_scatter);
    doc.write("env/fog_volumetric_density", e.fog_volumetric_density);
    doc.write("env/fog_volumetric_height", e.fog_volumetric_height);
    doc.write("env/fog_volumetric_falloff", e.fog_volumetric_falloff);
    doc.write("env/fog_volumetric_start", e.fog_volumetric_start);
    doc.write("env/fog_volumetric_end", e.fog_volumetric_end);
    doc.write("env/grade_temperature", e.grade_temperature);
    doc.write("env/grade_tint", e.grade_tint);
    doc.write("env/grade_slope", e.grade_slope);
    doc.write("env/grade_offset", e.grade_offset);
    doc.write("env/grade_power", e.grade_power);
}

[[nodiscard]] bracken::render::EnvironmentState read_env(const save::SaveDocument& doc) {
    // Start from the engine defaults: a key this file does not carry keeps the
    // default rather than becoming zero, which is what makes an older file
    // readable after a field is added.
    bracken::render::EnvironmentState e;
    (void)doc.read("env/sun_dir", e.sun_dir);
    (void)doc.read("env/sun_color", e.sun_color);
    (void)doc.read("env/sky_zenith", e.sky_zenith);
    (void)doc.read("env/sky_horizon", e.sky_horizon);
    (void)doc.read("env/ground_color", e.ground_color);
    (void)doc.read("env/ambient_sky", e.ambient_sky);
    (void)doc.read("env/ambient_ground", e.ambient_ground);
    (void)doc.read("env/fog_color_horizon", e.fog_color_horizon);
    (void)doc.read("env/fog_color_sun", e.fog_color_sun);
    (void)doc.read("env/fog_start", e.fog_start);
    (void)doc.read("env/fog_end", e.fog_end);
    (void)doc.read("env/fog_height_falloff", e.fog_height_falloff);
    (void)doc.read("env/cloud_coverage", e.cloud_coverage);
    (void)doc.read("env/cloud_density", e.cloud_density);
    (void)doc.read("env/cloud_scale", e.cloud_scale);
    (void)doc.read("env/cloud_speed", e.cloud_speed);
    (void)doc.read("env/haze_strength", e.haze_strength);
    (void)doc.read("env/fog_volumetric_sun_color", e.fog_volumetric_sun_color);
    (void)doc.read("env/fog_volumetric_scatter", e.fog_volumetric_scatter);
    (void)doc.read("env/fog_volumetric_density", e.fog_volumetric_density);
    (void)doc.read("env/fog_volumetric_height", e.fog_volumetric_height);
    (void)doc.read("env/fog_volumetric_falloff", e.fog_volumetric_falloff);
    (void)doc.read("env/fog_volumetric_start", e.fog_volumetric_start);
    (void)doc.read("env/fog_volumetric_end", e.fog_volumetric_end);
    (void)doc.read("env/grade_temperature", e.grade_temperature);
    (void)doc.read("env/grade_tint", e.grade_tint);
    (void)doc.read("env/grade_slope", e.grade_slope);
    (void)doc.read("env/grade_offset", e.grade_offset);
    (void)doc.read("env/grade_power", e.grade_power);
    return e;
}

// ---------------------------------------------------------------------------
// The scatter RNG.
//
// A named 32-bit LCG rather than anything from <random>: the sequence is part
// of the file format's meaning (a seed is worthless if it produces different
// rows on a different standard library), so it is written out here where it
// can be read and cannot drift. Numerical Recipes' constants, top 24 bits
// taken so the low-order-bit weakness of an LCG never reaches the output.
// ---------------------------------------------------------------------------

struct ScatterRng {
    uint32_t state;

    explicit ScatterRng(uint32_t seed) noexcept : state(seed * 747796405u + 2891336453u) {}

    [[nodiscard]] float next01() noexcept {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
    }
};

/// Two pi as a float literal rather than a computed constant: the yaw it
/// produces is stored in the file, so it must not depend on <numbers>.
constexpr float k_two_pi = 6.283185307179586f;

/// Cells per axis a scatter rule may ask for. A rule with a metre-scale step
/// over a kilometre-scale radius is a typo, and expanding it silently is how a
/// level load turns into a two-minute hang.
constexpr int k_max_scatter_cells_per_axis = 8192;

/// Upper bound on the two PATH-ADDRESSED tables (models, strings). Their length
/// comes from a record, not from a blob size, so nothing else bounds it.
constexpr std::size_t k_max_string_table_rows = 1u << 20;

} // namespace

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

const char* to_string(LevelError e) noexcept {
    switch (e) {
    case LevelError::FileNotFound: return "level file not found";
    case LevelError::NotALevelFile: return "not a bracken level file";
    case LevelError::Corrupt: return "level file is corrupt or truncated";
    case LevelError::FutureVersion: return "level written by a newer schema than this build";
    case LevelError::WriteFailed: return "could not write the level file";
    case LevelError::BadTable: return "a level table's count/stride disagree with its data";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// The instance transform
// ---------------------------------------------------------------------------

glm::mat4 level_instance_transform(const glm::vec3& position, float yaw, float scale) noexcept {
    glm::mat4 t = glm::translate(glm::mat4(1.0f), position);
    t = glm::rotate(t, yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    t = glm::scale(t, glm::vec3(scale));
    return t;
}

glm::mat4 LevelInstance::transform() const noexcept {
    return level_instance_transform(position, yaw, scale);
}

// ---------------------------------------------------------------------------
// Terrain conversion
// ---------------------------------------------------------------------------

Heightmap LevelTerrain::to_heightmap() const {
    Heightmap hm;
    hm.resolution = resolution;
    hm.world_origin_x = origin_x;
    hm.world_origin_z = origin_z;
    hm.world_size = size;
    hm.min_height = min_height;
    hm.max_height = max_height;
    hm.height_data = samples;
    return hm;
}

void LevelTerrain::from_heightmap(const Heightmap& hm) {
    resolution = hm.resolution;
    origin_x = hm.world_origin_x;
    origin_z = hm.world_origin_z;
    size = hm.world_size;
    min_height = hm.min_height;
    max_height = hm.max_height;
    samples = hm.height_data;
}

// ---------------------------------------------------------------------------
// Scatter
// ---------------------------------------------------------------------------

std::vector<LevelInstance> scatter_ground_cover(const LevelScatterRule& rule,
                                                std::span<const LevelScatterModel> models,
                                                const ScatterGroundFn& ground) {
    std::vector<LevelInstance> out;
    if (models.empty() || !(rule.step > 0.0f) || !(rule.outer_radius > 0.0f)) {
        return out;
    }

    float total_weight = 0.0f;
    for (const LevelScatterModel& m : models) {
        total_weight += std::max(0.0f, m.weight);
    }
    if (!(total_weight > 0.0f)) {
        BRACKEN_LOG_WARN(LOG_TAG, "scatter rule (seed {}) has no positively weighted model; nothing placed",
                         rule.seed);
        return out;
    }

    const float span = 2.0f * rule.outer_radius;
    const double cells = std::ceil(static_cast<double>(span) / static_cast<double>(rule.step));
    if (!(cells >= 1.0) || cells > static_cast<double>(k_max_scatter_cells_per_axis)) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "scatter rule (seed {}) asks for {} cells per axis (radius {}, step {}); refusing",
                          rule.seed, cells, rule.outer_radius, rule.step);
        return out;
    }
    const int n = static_cast<int>(cells);

    ScatterRng rng(rule.seed);
    // x outer, z inner — the emission order IS the draw order, so it is part
    // of the rule's meaning and not an implementation detail.
    for (int ix = 0; ix < n; ++ix) {
        const float gx = rule.center_x - rule.outer_radius + static_cast<float>(ix) * rule.step;
        for (int iz = 0; iz < n; ++iz) {
            const float gz = rule.center_z - rule.outer_radius + static_cast<float>(iz) * rule.step;

            // THE FIXED FIVE. Drawn before any rejection so the stream stays
            // aligned no matter what the ground callback answers — see
            // LevelScatterRule's comment for why that is the whole point.
            const float jitter_x = rng.next01();
            const float jitter_z = rng.next01();
            const float yaw01 = rng.next01();
            const float pick01 = rng.next01();
            const float scale01 = rng.next01();

            const float x = gx + (jitter_x - 0.5f) * rule.jitter * rule.step;
            const float z = gz + (jitter_z - 0.5f) * rule.jitter * rule.step;
            const float dx = x - rule.center_x;
            const float dz = z - rule.center_z;
            const float rr = std::sqrt(dx * dx + dz * dz);
            if (rr < rule.inner_radius || rr > rule.outer_radius) {
                continue;
            }
            const std::optional<float> height = ground ? ground(x, z) : std::optional<float>(0.0f);
            if (!height.has_value()) {
                continue;
            }

            std::size_t selected = models.size() - 1;
            float remaining = pick01 * total_weight;
            for (std::size_t i = 0; i < models.size(); ++i) {
                remaining -= std::max(0.0f, models[i].weight);
                if (remaining < 0.0f) {
                    selected = i;
                    break;
                }
            }
            const LevelScatterModel& m = models[selected];

            LevelInstance inst;
            inst.model = m.model;
            inst.position = {x, *height - rule.sink, z};
            inst.yaw = yaw01 * k_two_pi;
            inst.scale = m.min_scale + (m.max_scale - m.min_scale) * scale01;
            out.push_back(inst);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Level authoring / queries
// ---------------------------------------------------------------------------

uint32_t Level::intern_model(std::string_view model_name, std::string_view source) {
    for (std::size_t i = 0; i < models.size(); ++i) {
        if (models[i] == model_name) {
            return static_cast<uint32_t>(i);
        }
    }
    models.emplace_back(model_name);
    model_sources.resize(models.size());
    model_sources.back() = std::string(source);
    return static_cast<uint32_t>(models.size() - 1);
}

uint32_t Level::intern_string(std::string_view s) {
    for (std::size_t i = 0; i < strings.size(); ++i) {
        if (strings[i] == s) {
            return static_cast<uint32_t>(i);
        }
    }
    strings.emplace_back(s);
    return static_cast<uint32_t>(strings.size() - 1);
}

uint32_t Level::add_instance(std::string_view model_name, const glm::vec3& position, float yaw, float scale,
                             const glm::vec4& tint, DrawFlags flags, uint32_t motion_id, uint32_t lod_hint) {
    LevelInstance inst;
    inst.model = intern_model(model_name);
    inst.motion_id = motion_id;
    inst.flags = flags;
    inst.lod_hint = lod_hint;
    inst.position = position;
    inst.yaw = yaw;
    inst.scale = scale;
    inst.tint = tint;
    instances.push_back(inst);
    return static_cast<uint32_t>(instances.size() - 1);
}

uint32_t Level::add_marker(std::string_view role, const glm::vec3& position, float yaw) {
    LevelMarker m;
    m.name = intern_string(role);
    m.position = position;
    m.yaw = yaw;
    markers.push_back(m);
    return static_cast<uint32_t>(markers.size() - 1);
}

std::vector<LevelMarker> Level::markers_named(std::string_view role) const {
    std::vector<LevelMarker> out;
    uint32_t index = 0;
    bool found = false;
    for (std::size_t i = 0; i < strings.size(); ++i) {
        if (strings[i] == role) {
            index = static_cast<uint32_t>(i);
            found = true;
            break;
        }
    }
    if (!found) {
        return out;
    }
    for (const LevelMarker& m : markers) {
        if (m.name == index) {
            out.push_back(m);
        }
    }
    return out;
}

std::string_view Level::model_name(uint32_t index) const noexcept {
    return index < models.size() ? std::string_view(models[index]) : std::string_view{};
}

std::string_view Level::string_at(uint32_t index) const noexcept {
    return index < strings.size() ? std::string_view(strings[index]) : std::string_view{};
}

std::span<const LevelScatterModel> Level::scatter_slice(const LevelScatterRule& rule) const noexcept {
    const auto first = static_cast<std::size_t>(rule.model_first);
    const auto count = static_cast<std::size_t>(rule.model_count);
    if (first > scatter_models.size() || count > scatter_models.size() - first) {
        return {};
    }
    return std::span<const LevelScatterModel>(scatter_models.data() + first, count);
}

// ---------------------------------------------------------------------------
// Encode / decode
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] save::SaveDocument encode(const Level& level) {
    // Start from the carried-forward records so a newer build's data survives
    // an older build loading and re-saving. Known keys are written over the
    // top; a stale key under an owned prefix cannot be here because the
    // decoder never puts one there.
    save::SaveDocument doc = level.extras();
    doc.set_game_version(level.content_version);

    doc.write("level/version", static_cast<int64_t>(k_level_format_version));
    doc.write("level/name", level.name);

    doc.write("models/count", static_cast<int64_t>(level.models.size()));
    for (std::size_t i = 0; i < level.models.size(); ++i) {
        doc.write(save::index_path("models", i, "name"), level.models[i]);
        const std::string_view source = i < level.model_sources.size()
                                            ? std::string_view(level.model_sources[i])
                                            : std::string_view{};
        doc.write(save::index_path("models", i, "source"), source);
    }

    doc.write("strings/count", static_cast<int64_t>(level.strings.size()));
    for (std::size_t i = 0; i < level.strings.size(); ++i) {
        doc.write(std::format("strings/{}", i), level.strings[i]);
    }

    write_table(doc, "instances", level.instances);
    write_table(doc, "markers", level.markers);
    write_table(doc, "water", level.water);
    write_table(doc, "scatter/models", level.scatter_models);
    write_table(doc, "scatter/rules", level.scatter_rules);

    std::vector<PointLightRow> point_rows;
    point_rows.reserve(level.point_lights.size());
    for (const auto& l : level.point_lights) {
        point_rows.push_back(to_row(l));
    }
    write_table(doc, "lights/point", point_rows);

    std::vector<SpotLightRow> spot_rows;
    spot_rows.reserve(level.spot_lights.size());
    for (const auto& l : level.spot_lights) {
        spot_rows.push_back(to_row(l));
    }
    write_table(doc, "lights/spot", spot_rows);

    doc.write("terrain/present", level.terrain.has_value());
    if (level.terrain.has_value()) {
        const LevelTerrain& t = *level.terrain;
        doc.write("terrain/resolution", static_cast<int64_t>(t.resolution));
        doc.write("terrain/origin_x", t.origin_x);
        doc.write("terrain/origin_z", t.origin_z);
        doc.write("terrain/size", t.size);
        doc.write("terrain/min_height", t.min_height);
        doc.write("terrain/max_height", t.max_height);
        doc.write_blob("terrain/samples",
                       std::span<const std::byte>(reinterpret_cast<const std::byte*>(t.samples.data()),
                                                  t.samples.size() * sizeof(uint16_t)));
        doc.write("terrain/splat_resolution", static_cast<int64_t>(t.splat_resolution));
        doc.write_blob("terrain/splat",
                       std::span<const std::byte>(reinterpret_cast<const std::byte*>(t.splat.data()),
                                                  t.splat.size()));
        write_table(doc, "terrain/grass_exclusions", t.grass_exclusions);
    }

    doc.write("env/present", level.environment.has_value());
    if (level.environment.has_value()) {
        write_env(doc, *level.environment);
    }

    doc.write("nav/present", level.nav.has_value());
    if (level.nav.has_value()) {
        const nav::NavGridConfig& n = *level.nav;
        doc.write("nav/origin", n.origin);
        doc.write("nav/cell_size", n.cell_size);
        doc.write("nav/width", static_cast<int64_t>(n.width));
        doc.write("nav/depth", static_cast<int64_t>(n.depth));
        doc.write("nav/max_slope_degrees", n.max_slope_degrees);
        doc.write("nav/max_step", n.max_step);
    }

    return doc;
}

[[nodiscard]] core::Result<Level, LevelError> decode(const save::SaveDocument& doc) {
    if (!doc.has("level/version")) {
        BRACKEN_LOG_ERROR(LOG_TAG, "document has no 'level/version' record; not a level file");
        return core::Err(LevelError::NotALevelFile);
    }
    const auto version = doc.read_or<int64_t>("level/version", 0);
    if (version > static_cast<int64_t>(k_level_format_version)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "level schema version {} is newer than this build ({})", version,
                          k_level_format_version);
        return core::Err(LevelError::FutureVersion);
    }

    Level level;
    level.content_version = doc.game_version();
    (void)doc.read("level/name", level.name);

    // The string tables are path-addressed, so their length comes from a record
    // rather than from a blob's size — nothing bounds it but this. A file whose
    // checksum passes can still name a billion models (hand-edited, or written
    // by something with its own bug), and `resize` would answer that with
    // bad_alloc rather than a diagnosis.
    const std::size_t model_count = doc.read_count("models");
    const std::size_t string_count_raw = doc.read_count("strings");
    if (model_count > k_max_string_table_rows || string_count_raw > k_max_string_table_rows) {
        BRACKEN_LOG_ERROR(LOG_TAG, "level declares {} models and {} strings; refusing", model_count,
                          string_count_raw);
        return core::Err(LevelError::BadTable);
    }
    level.models.resize(model_count);
    level.model_sources.resize(model_count);
    for (std::size_t i = 0; i < model_count; ++i) {
        (void)doc.read(save::index_path("models", i, "name"), level.models[i]);
        (void)doc.read(save::index_path("models", i, "source"), level.model_sources[i]);
    }

    level.strings.resize(string_count_raw);
    for (std::size_t i = 0; i < string_count_raw; ++i) {
        (void)doc.read(std::format("strings/{}", i), level.strings[i]);
    }

    std::vector<PointLightRow> point_rows;
    std::vector<SpotLightRow> spot_rows;
    if (!read_table(doc, "instances", level.instances) || !read_table(doc, "markers", level.markers) ||
        !read_table(doc, "water", level.water) || !read_table(doc, "scatter/models", level.scatter_models) ||
        !read_table(doc, "scatter/rules", level.scatter_rules) ||
        !read_table(doc, "lights/point", point_rows) || !read_table(doc, "lights/spot", spot_rows)) {
        return core::Err(LevelError::BadTable);
    }
    level.point_lights.reserve(point_rows.size());
    for (const PointLightRow& r : point_rows) {
        level.point_lights.push_back(from_row(r));
    }
    level.spot_lights.reserve(spot_rows.size());
    for (const SpotLightRow& r : spot_rows) {
        level.spot_lights.push_back(from_row(r));
    }

    if (doc.read_or<bool>("terrain/present", false)) {
        LevelTerrain t;
        (void)doc.read("terrain/resolution", t.resolution);
        (void)doc.read("terrain/origin_x", t.origin_x);
        (void)doc.read("terrain/origin_z", t.origin_z);
        (void)doc.read("terrain/size", t.size);
        (void)doc.read("terrain/min_height", t.min_height);
        (void)doc.read("terrain/max_height", t.max_height);
        std::vector<std::byte> samples;
        (void)doc.read_blob("terrain/samples", samples);
        if (samples.size() % sizeof(uint16_t) != 0) {
            BRACKEN_LOG_ERROR(LOG_TAG, "terrain/samples is {} bytes, not a whole number of 16-bit samples",
                              samples.size());
            return core::Err(LevelError::BadTable);
        }
        t.samples.resize(samples.size() / sizeof(uint16_t));
        if (!t.samples.empty()) {
            std::memcpy(t.samples.data(), samples.data(), samples.size());
        }
        (void)doc.read("terrain/splat_resolution", t.splat_resolution);
        std::vector<std::byte> splat;
        (void)doc.read_blob("terrain/splat", splat);
        t.splat.resize(splat.size());
        if (!t.splat.empty()) {
            std::memcpy(t.splat.data(), splat.data(), splat.size());
        }
        if (!read_table(doc, "terrain/grass_exclusions", t.grass_exclusions)) {
            return core::Err(LevelError::BadTable);
        }
        level.terrain = std::move(t);
    }

    if (doc.read_or<bool>("env/present", false)) {
        level.environment = read_env(doc);
    }

    if (doc.read_or<bool>("nav/present", false)) {
        nav::NavGridConfig n;
        (void)doc.read("nav/origin", n.origin);
        (void)doc.read("nav/cell_size", n.cell_size);
        int64_t width = 0;
        int64_t depth = 0;
        (void)doc.read("nav/width", width);
        (void)doc.read("nav/depth", depth);
        n.width = static_cast<int>(width);
        n.depth = static_cast<int>(depth);
        (void)doc.read("nav/max_slope_degrees", n.max_slope_degrees);
        (void)doc.read("nav/max_step", n.max_step);
        level.nav = n;
    }

    // Everything this build did not consume, carried verbatim. This is the
    // unknown-chunk guarantee: an older engine that loads a newer level and
    // re-saves it must not delete what it could not read.
    for (std::string_view path : doc.paths()) {
        if (is_owned_path(path)) {
            continue;
        }
        if (const save::SaveDocument::Value* v = doc.value_at(path)) {
            level.extras().write_value(path, *v);
        }
    }

    return level;
}

[[nodiscard]] LevelError map_error(save::SaveError e) noexcept {
    switch (e) {
    case save::SaveError::FileNotFound: return LevelError::FileNotFound;
    case save::SaveError::NotASaveFile: return LevelError::NotALevelFile;
    case save::SaveError::Corrupt: return LevelError::Corrupt;
    case save::SaveError::FutureVersion: return LevelError::FutureVersion;
    case save::SaveError::WriteFailed: return LevelError::WriteFailed;
    }
    return LevelError::Corrupt;
}

} // namespace

// ---------------------------------------------------------------------------
// Files and bytes
// ---------------------------------------------------------------------------

std::vector<std::byte> serialize_level(const Level& level) { return encode(level).serialize(); }

core::Result<void, LevelError> save_level(const Level& level, const std::string& path) {
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    const save::SaveDocument doc = encode(level);
    if (auto r = doc.save(path); r.is_err()) {
        return core::Err(map_error(r.error()));
    }
    BRACKEN_LOG_INFO(LOG_TAG, "Wrote level '{}' to {} ({} instances, {} models, {} markers)", level.name, path,
                     level.instances.size(), level.models.size(), level.markers.size());
    return {};
}

core::Result<Level, LevelError> load_level(const std::string& path) {
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    auto doc = save::SaveDocument::load(path);
    if (doc.is_err()) {
        return core::Err(map_error(doc.error()));
    }
    auto level = decode(doc.value());
    if (level.is_ok()) {
        BRACKEN_LOG_INFO(LOG_TAG, "Loaded level '{}' from {} ({} instances, {} models)",
                         level.value().name, path, level.value().instances.size(),
                         level.value().models.size());
    }
    return level;
}

core::Result<Level, LevelError> parse_level(std::span<const std::byte> bytes) {
    auto doc = save::SaveDocument::parse(bytes);
    if (doc.is_err()) {
        return core::Err(map_error(doc.error()));
    }
    return decode(doc.value());
}

// ---------------------------------------------------------------------------
// Dump
// ---------------------------------------------------------------------------

std::string level_dump(const Level& level) {
    constexpr std::size_t k_preview_rows = 8;
    std::string out;

    out += std::format("level '{}'  schema v{}  content v{}\n", level.name, k_level_format_version,
                       level.content_version);
    out += std::format("  instances     {}\n", level.instances.size());
    out += std::format("  models        {}\n", level.models.size());
    out += std::format("  strings       {}\n", level.strings.size());
    out += std::format("  point lights  {}\n", level.point_lights.size());
    out += std::format("  spot lights   {}\n", level.spot_lights.size());
    out += std::format("  markers       {}\n", level.markers.size());
    out += std::format("  water regions {}\n", level.water.size());
    out += std::format("  scatter rules {} (over {} candidate models)\n", level.scatter_rules.size(),
                       level.scatter_models.size());
    out += std::format("  extras        {} carried records\n", level.extras().record_count());

    out += "\nmodels:\n";
    for (std::size_t i = 0; i < level.models.size(); ++i) {
        const std::string_view source =
            i < level.model_sources.size() ? std::string_view(level.model_sources[i]) : std::string_view{};
        out += std::format("  [{:4}] {}{}{}\n", i, level.models[i], source.empty() ? "" : "  <- ", source);
    }

    if (!level.strings.empty()) {
        out += "\nstrings:\n";
        for (std::size_t i = 0; i < level.strings.size(); ++i) {
            out += std::format("  [{:4}] {}\n", i, level.strings[i]);
        }
    }

    out += std::format("\ninstances (first {} of {}):\n", std::min(k_preview_rows, level.instances.size()),
                       level.instances.size());
    for (std::size_t i = 0; i < level.instances.size() && i < k_preview_rows; ++i) {
        const LevelInstance& n = level.instances[i];
        out += std::format("  [{:5}] {:24} pos({:.2f}, {:.2f}, {:.2f}) yaw {:.4f} scale {:.4f} flags 0x{:x}\n", i,
                           level.model_name(n.model), n.position.x, n.position.y, n.position.z, n.yaw, n.scale,
                           static_cast<uint32_t>(n.flags));
    }

    if (!level.markers.empty()) {
        out += "\nmarkers:\n";
        for (std::size_t i = 0; i < level.markers.size() && i < k_preview_rows; ++i) {
            const LevelMarker& m = level.markers[i];
            out += std::format("  [{:4}] {:20} ({:.2f}, {:.2f}, {:.2f})\n", i, level.string_at(m.name),
                               m.position.x, m.position.y, m.position.z);
        }
    }

    for (std::size_t i = 0; i < level.scatter_rules.size(); ++i) {
        const LevelScatterRule& r = level.scatter_rules[i];
        out += std::format("\nscatter rule [{}] '{}': seed {} grid step {:.3f} radius {:.1f}..{:.1f} "
                           "jitter {:.2f} sink {:.2f} models [{}..{}) {}\n",
                           i, level.string_at(r.name), r.seed, r.step, r.inner_radius, r.outer_radius,
                           r.jitter, r.sink, r.model_first, r.model_first + r.model_count,
                           r.baked != 0 ? "(BAKED into instances)" : "(expanded at load)");
    }

    if (level.terrain.has_value()) {
        const LevelTerrain& t = *level.terrain;
        out += std::format("\nterrain: {}^2 samples, origin ({:.1f}, {:.1f}) size {:.1f}, height "
                           "[{:.1f}, {:.1f}], splat {}^2, {} grass exclusions\n",
                           t.resolution, t.origin_x, t.origin_z, t.size, t.min_height, t.max_height,
                           t.splat_resolution, t.grass_exclusions.size());
    }
    if (level.environment.has_value()) {
        const bracken::render::EnvironmentState& e = *level.environment;
        out += std::format("\nenvironment: sun dir ({:.3f}, {:.3f}, {:.3f}) fog {:.0f}..{:.0f} "
                           "cloud coverage {:.2f}\n",
                           e.sun_dir.x, e.sun_dir.y, e.sun_dir.z, e.fog_start, e.fog_end, e.cloud_coverage);
    }
    if (level.nav.has_value()) {
        const nav::NavGridConfig& n = *level.nav;
        out += std::format("\nnav: {}x{} cells of {:.1f}, origin ({:.1f}, {:.1f}, {:.1f}), max slope {:.1f} "
                           "deg, max step {:.1f}\n",
                           n.width, n.depth, n.cell_size, n.origin.x, n.origin.y, n.origin.z,
                           n.max_slope_degrees, n.max_step);
    }

    return out;
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

LevelResolveReport ResolvedLevel::resolve(const Level& level, const bracken::ModelManager& models) {
    BRACKEN_ASSERT_NOT_SIM_STAGE();

    LevelResolveReport report;
    draws_.clear();
    draws_.reserve(level.instances.size());

    // One lookup per distinct NAME, not per instance: 2550 instances over 47
    // models is 47 hash lookups here and none in the loop.
    std::vector<bracken::ModelHandle> handles(level.models.size(), bracken::INVALID_MODEL_HANDLE);
    for (std::size_t i = 0; i < level.models.size(); ++i) {
        handles[i] = models.get_handle(level.models[i]);
        if (handles[i] == bracken::INVALID_MODEL_HANDLE) {
            ++report.models_unresolved;
            BRACKEN_LOG_WARN(LOG_TAG, "level '{}': model '{}' is not registered; its instances are dropped",
                             level.name, level.models[i]);
        }
    }

    for (const LevelInstance& inst : level.instances) {
        if (inst.model >= handles.size()) {
            ++report.instances_dropped_bad_index;
            continue;
        }
        const bracken::ModelHandle handle = handles[inst.model];
        if (handle == bracken::INVALID_MODEL_HANDLE) {
            ++report.instances_dropped_unknown_model;
            continue;
        }
        draws_.push_back(Draw{handle, inst.transform(), inst.tint, inst.flags, inst.motion_id, inst.lod_hint});
    }
    report.instances_placed = static_cast<uint32_t>(draws_.size());

    if (report.any_dropped()) {
        // One summary, not one line per instance: a level whose art is missing
        // is missing it for thousands of rows at a time.
        BRACKEN_LOG_WARN(LOG_TAG,
                         "level '{}': placed {} instances, dropped {} (unknown model) + {} (bad model index) "
                         "over {} unresolved names",
                         level.name, report.instances_placed, report.instances_dropped_unknown_model,
                         report.instances_dropped_bad_index, report.models_unresolved);
    }
    return report;
}

void ResolvedLevel::emit(FrameSink& sink) const {
    for (const Draw& d : draws_) {
        sink.draw_model(d.model, d.transform, d.tint, d.flags, d.motion_id);
    }
}

std::vector<std::pair<std::string, std::string>> missing_models(const Level& level,
                                                                const bracken::ModelManager& models) {
    std::vector<std::pair<std::string, std::string>> out;
    for (std::size_t i = 0; i < level.models.size(); ++i) {
        if (models.get_handle(level.models[i]) != bracken::INVALID_MODEL_HANDLE) {
            continue;
        }
        const std::string source = i < level.model_sources.size() ? level.model_sources[i] : std::string{};
        out.emplace_back(level.models[i], source);
    }
    return out;
}

} // namespace bracken::scene
