#include "bracken/scene/level.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_stage.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render/water_renderer.hpp"
#include "bracken/scene/frame_sink.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
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
    // BEFORE the blob check, because this is the bound the blob check cannot
    // provide: `data.size() / stride` is proportional to a stride the file also
    // chose. See k_max_table_rows.
    if (count > static_cast<int64_t>(k_max_table_rows)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "table '{}': {} rows is past the {}-row ceiling", prefix, count,
                          k_max_table_rows);
        return false;
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

// ---------------------------------------------------------------------------
// Finiteness
//
// Every float a level carries is checked at LOAD. See LevelError::BadValue for
// why here and not at use: a NaN transform does not fail loudly anywhere
// downstream, it quietly makes a frame wrong in a place that has no idea a
// file was involved.
//
// The checks are explicit per record type rather than a blanket byte scan,
// because these rows are float/uint32 MIXTURES — reinterpreting a model index
// as a float and finding a NaN bit pattern would refuse a perfectly good file.
// `all_floats_finite` is the exception and is used only on structs that are
// float all the way through, which their size static_asserts pin.
// ---------------------------------------------------------------------------

struct FiniteScan {
    bool ok = true;
    void operator()(float v) noexcept { ok = ok && std::isfinite(v); }
    void operator()(const glm::vec2& v) noexcept { (*this)(v.x); (*this)(v.y); }
    void operator()(const glm::vec3& v) noexcept { (*this)(v.x); (*this)(v.y); (*this)(v.z); }
    void operator()(const glm::vec4& v) noexcept {
        (*this)(v.x);
        (*this)(v.y);
        (*this)(v.z);
        (*this)(v.w);
    }
};

/// For FLOAT-ONLY trivially copyable structs, and nothing else. Adding a
/// non-float member to one of these makes this wrong in the silent direction
/// (an integer that happens to look like a NaN), which is why each caller sits
/// next to a size static_assert that a new member would break first.
template <typename T>
[[nodiscard]] bool all_floats_finite(const T& v) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(sizeof(T) % sizeof(float) == 0);
    float scratch[sizeof(T) / sizeof(float)];
    std::memcpy(scratch, &v, sizeof(T));
    for (float f : scratch) {
        if (!std::isfinite(f)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool row_finite(const LevelInstance& r) noexcept {
    FiniteScan s;
    s(r.position);
    s(r.yaw);
    s(r.scale);
    s(r.tint);
    return s.ok;
}

[[nodiscard]] bool row_finite(const LevelMarker& r) noexcept {
    FiniteScan s;
    s(r.position);
    s(r.yaw);
    return s.ok;
}

[[nodiscard]] bool row_finite(const LevelRect& r) noexcept { return all_floats_finite(r); }
[[nodiscard]] bool row_finite(const LevelWater& r) noexcept { return all_floats_finite(r); }

[[nodiscard]] bool row_finite(const LevelScatterModel& r) noexcept {
    FiniteScan s;
    s(r.weight);
    s(r.min_scale);
    s(r.max_scale);
    return s.ok;
}

[[nodiscard]] bool row_finite(const LevelScatterRule& r) noexcept {
    FiniteScan s;
    s(r.center_x);
    s(r.center_z);
    s(r.inner_radius);
    s(r.outer_radius);
    s(r.step);
    s(r.jitter);
    s(r.sink);
    return s.ok;
}

[[nodiscard]] bool row_finite(const PointLightRow& r) noexcept {
    FiniteScan s;
    s(r.px);
    s(r.py);
    s(r.pz);
    s(r.radius);
    s(r.cr);
    s(r.cg);
    s(r.cb);
    s(r.intensity);
    s(r.source_radius);
    return s.ok;
}

[[nodiscard]] bool row_finite(const SpotLightRow& r) noexcept {
    FiniteScan s;
    s(r.px);
    s(r.py);
    s(r.pz);
    s(r.radius);
    s(r.dx);
    s(r.dy);
    s(r.dz);
    s(r.inner_deg);
    s(r.outer_deg);
    s(r.cr);
    s(r.cg);
    s(r.cb);
    s(r.intensity);
    s(r.source_radius);
    return s.ok;
}

/// The first offending ROW is named. A level with one bad crate among 2550 is
/// the realistic case, and "instances[1873]" is the difference between a fix
/// and a bisect.
template <typename T>
[[nodiscard]] bool rows_finite(const std::vector<T>& rows, std::string_view what) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (!row_finite(rows[i])) {
            BRACKEN_LOG_ERROR(LOG_TAG, "{}[{}] carries a NaN or an infinity; refusing the level", what, i);
            return false;
        }
    }
    return true;
}

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
    case LevelError::BadValue: return "a level value is not a finite number";
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
// Water conversion
//
// Field for field, in declaration order, both directions. The only correct way
// to read these is against `render::WaterRegion` side by side — which is the
// argument for their being here, once, instead of in every host that ships a
// river.
// ---------------------------------------------------------------------------

LevelWater to_level_water(const bracken::render::WaterRegion& region) noexcept {
    LevelWater w;
    w.min_x = region.min_x;
    w.min_z = region.min_z;
    w.max_x = region.max_x;
    w.max_z = region.max_z;
    w.height = region.height;
    w.flow = region.flow;
    w.flow_speed = region.flow_speed;
    w.deep_color = region.deep_color;
    w.shallow_color = region.shallow_color;
    w.absorption = region.absorption;
    w.foam_depth = region.foam_depth;
    w.reflection_roughness = region.reflection_roughness;
    w.reflection_strength = region.reflection_strength;
    w.refraction_strength = region.refraction_strength;
    w.ssr_strength = region.ssr_strength;
    w.ssr_max_distance = region.ssr_max_distance;
    w.ssr_thickness = region.ssr_thickness;
    w.detail_wave_strength = region.detail_wave_strength;
    w.sim_displacement = region.sim_displacement;
    w.sim_normal_strength = region.sim_normal_strength;
    w.ripple_foam_gain = region.ripple_foam_gain;
    w.sim_center = region.sim_center;
    w.sim_extent = region.sim_extent;
    w.wave_speed = region.wave_speed;
    w.wave_damping = region.wave_damping;
    return w;
}

bracken::render::WaterRegion to_water_region(const LevelWater& water) noexcept {
    bracken::render::WaterRegion r;
    r.min_x = water.min_x;
    r.min_z = water.min_z;
    r.max_x = water.max_x;
    r.max_z = water.max_z;
    r.height = water.height;
    r.flow = water.flow;
    r.flow_speed = water.flow_speed;
    r.deep_color = water.deep_color;
    r.shallow_color = water.shallow_color;
    r.absorption = water.absorption;
    r.foam_depth = water.foam_depth;
    r.reflection_roughness = water.reflection_roughness;
    r.reflection_strength = water.reflection_strength;
    r.refraction_strength = water.refraction_strength;
    r.ssr_strength = water.ssr_strength;
    r.ssr_max_distance = water.ssr_max_distance;
    r.ssr_thickness = water.ssr_thickness;
    r.detail_wave_strength = water.detail_wave_strength;
    r.sim_displacement = water.sim_displacement;
    r.sim_normal_strength = water.sim_normal_strength;
    r.ripple_foam_gain = water.ripple_foam_gain;
    r.sim_center = water.sim_center;
    r.sim_extent = water.sim_extent;
    r.wave_speed = water.wave_speed;
    r.wave_damping = water.wave_damping;
    return r;
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

    // `last_positive` is the selection walk's fall-through pick, and it must NOT
    // be `models.size() - 1`. The walk subtracts float weights one at a time, so
    // accumulated rounding can leave `remaining` at or just above zero after the
    // last positively weighted entry — and "the last row" as a fallback then
    // hands that draw to a row of weight <= 0, which LevelScatterModel documents
    // as never picked.
    //
    // Honest scope: an exhaustive sweep of all 2^24 values `next01()` can return
    // found that branch reachable only for contrived weight sets and only for a
    // handful of draws in 2^24. This makes the documented contract
    // unconditional; it is not a bug anyone was hitting. The one that WAS
    // reachable is the infinite total below.
    float total_weight = 0.0f;
    std::size_t last_positive = 0;
    bool any_positive = false;
    for (std::size_t i = 0; i < models.size(); ++i) {
        const float w = std::max(0.0f, models[i].weight);
        total_weight += w;
        if (w > 0.0f) {
            last_positive = i;
            any_positive = true;
        }
    }
    if (!any_positive || !(total_weight > 0.0f)) {
        BRACKEN_LOG_WARN(LOG_TAG, "scatter rule (seed {}) has no positively weighted model; nothing placed",
                         rule.seed);
        return out;
    }
    // An INFINITE total is the case the load-time finiteness sweep cannot see:
    // `row_finite(LevelScatterModel)` checks each weight on its own, and two
    // perfectly finite 3e38 weights sum to infinity. `pick01 * inf` is then inf
    // for every draw but the zeroth (which is NaN), no subtraction ever goes
    // negative, and the walk falls through on EVERY candidate — so the rule
    // silently ignores its weights and places one single model a million times.
    // That is not a degraded carpet, it is a different level, and it is refused.
    if (!std::isfinite(total_weight)) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "scatter rule (seed {}) has {} models whose weights sum to infinity; the "
                          "selection would be meaningless, so nothing is placed",
                          rule.seed, models.size());
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

            std::size_t selected = last_positive;
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
            if (out.size() >= k_max_scatter_instances) {
                // The cell bound alone permits 8192^2 candidates; this is what
                // bounds the OUTPUT when a ground callback accepts all of
                // them. Refuse the rest of the rule rather than truncate
                // silently, and say so once — a partially expanded carpet that
                // nobody was told about is worse than none.
                BRACKEN_LOG_ERROR(LOG_TAG,
                                  "scatter rule (seed {}) reached the {}-instance ceiling at cell ({}, {}); "
                                  "the rest of the rule is not expanded",
                                  rule.seed, k_max_scatter_instances, ix, iz);
                return out;
            }
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
    // Seed from the carried-forward records so a newer build's data survives an
    // older build loading and re-saving — but copy only what is NOT ours.
    //
    // The decoder never files an owned key into extras(), so the only way one
    // gets there is a host writing into the schema's own namespace. Copying it
    // would leave a record this encoder does not overwrite (`models/9/name` in
    // a three-model level) sitting in the file to confuse the next reader.
    // Dropping it loses the host nothing it was entitled to keep, and saying so
    // once is what stops the loss from being a mystery.
    save::SaveDocument doc;
    doc.set_game_version(level.content_version);
    std::size_t squatters = 0;
    for (std::string_view path : level.extras().paths()) {
        if (is_owned_path(path)) {
            ++squatters;
            continue;
        }
        if (const save::SaveDocument::Value* v = level.extras().value_at(path)) {
            doc.write_value(path, *v);
        }
    }
    if (squatters != 0) {
        BRACKEN_LOG_WARN(LOG_TAG,
                         "level '{}': {} extra record(s) sit under a schema-owned prefix and are dropped; "
                         "namespace host records under your own prefix",
                         level.name, squatters);
    }

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

    // The name tables are path-addressed, so their length comes from a RECORD
    // rather than from a blob's size — nothing bounds it but this. A file whose
    // checksum passes can still name a billion models (hand-edited, or written
    // by something with its own bug), and `resize` would answer that with
    // bad_alloc rather than a diagnosis.
    //
    // TWO bounds, and the tight one is the second. A table of N names occupies
    // 2N records in this container, so a declared count larger than the
    // container's ENTIRE record count is a lie the file's own size disproves —
    // that is what makes the allocation bounded by the bytes on disk rather
    // than by a constant somebody picked. The ceiling is the coarse backstop
    // for a small file that packs its lie densely.
    const std::size_t records = doc.record_count();
    const std::size_t model_count = doc.read_count("models");
    const std::size_t string_count = doc.read_count("strings");
    if (model_count > k_max_name_table_rows || string_count > k_max_name_table_rows ||
        model_count > records || string_count > records) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "level declares {} models and {} strings over {} records (ceiling {}); refusing",
                          model_count, string_count, records, k_max_name_table_rows);
        return core::Err(LevelError::BadTable);
    }
    level.models.resize(model_count);
    level.model_sources.resize(model_count);
    for (std::size_t i = 0; i < model_count; ++i) {
        (void)doc.read(save::index_path("models", i, "name"), level.models[i]);
        (void)doc.read(save::index_path("models", i, "source"), level.model_sources[i]);
    }

    level.strings.resize(string_count);
    for (std::size_t i = 0; i < string_count; ++i) {
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
    // Finiteness, before any of it becomes a matrix. See LevelError::BadValue.
    if (!rows_finite(level.instances, "instances") || !rows_finite(level.markers, "markers") ||
        !rows_finite(level.water, "water") || !rows_finite(level.scatter_models, "scatter/models") ||
        !rows_finite(level.scatter_rules, "scatter/rules") ||
        !rows_finite(point_rows, "lights/point") || !rows_finite(spot_rows, "lights/spot")) {
        return core::Err(LevelError::BadValue);
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
        // Read the two resolutions WIDE and range-check them. `read(uint32_t&)`
        // narrows an int64 silently, so a file saying "-1" would otherwise
        // arrive as 4294967295 and be squared.
        const int64_t resolution = doc.read_or<int64_t>("terrain/resolution", 0);
        const int64_t splat_resolution = doc.read_or<int64_t>("terrain/splat_resolution", 0);
        if (resolution < 0 || resolution > static_cast<int64_t>(k_max_terrain_resolution) ||
            splat_resolution < 0 || splat_resolution > static_cast<int64_t>(k_max_terrain_resolution)) {
            BRACKEN_LOG_ERROR(LOG_TAG, "terrain resolution {} / splat resolution {} is out of range (0..{})",
                              resolution, splat_resolution, k_max_terrain_resolution);
            return core::Err(LevelError::BadTable);
        }
        t.resolution = static_cast<uint32_t>(resolution);
        t.splat_resolution = static_cast<uint32_t>(splat_resolution);
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
        std::vector<std::byte> splat;
        (void)doc.read_blob("terrain/splat", splat);
        t.splat.resize(splat.size());
        if (!t.splat.empty()) {
            std::memcpy(t.splat.data(), splat.data(), splat.size());
        }
        // The header and the blob must agree. A heightmap whose declared
        // resolution does not square to its sample count is the one shape that
        // would sail past every other check here and then be INDEXED by
        // `Heightmap::get_height_local` at `z * resolution + x` — an out-of-
        // bounds read the moment the level is applied. `Heightmap::valid`
        // catches it too, but only if the host remembers to ask; refusing the
        // file means it cannot be reached.
        const auto samples_wanted = static_cast<std::size_t>(t.resolution) * t.resolution;
        const auto splat_wanted = static_cast<std::size_t>(t.splat_resolution) * t.splat_resolution * 4u;
        if (t.samples.size() != samples_wanted || t.splat.size() != splat_wanted) {
            BRACKEN_LOG_ERROR(LOG_TAG,
                              "terrain: {} samples for a {}^2 map (want {}), {} splat bytes for {}^2 RGBA "
                              "(want {})",
                              t.samples.size(), t.resolution, samples_wanted, t.splat.size(),
                              t.splat_resolution, splat_wanted);
            return core::Err(LevelError::BadTable);
        }
        if (!read_table(doc, "terrain/grass_exclusions", t.grass_exclusions)) {
            return core::Err(LevelError::BadTable);
        }
        FiniteScan scan;
        scan(t.origin_x);
        scan(t.origin_z);
        scan(t.size);
        scan(t.min_height);
        scan(t.max_height);
        if (!scan.ok || !rows_finite(t.grass_exclusions, "terrain/grass_exclusions")) {
            BRACKEN_LOG_ERROR(LOG_TAG, "terrain header carries a NaN or an infinity; refusing the level");
            return core::Err(LevelError::BadValue);
        }
        level.terrain = std::move(t);
    }

    if (doc.read_or<bool>("env/present", false)) {
        const bracken::render::EnvironmentState e = read_env(doc);
        // EnvironmentState is float all the way through — every member is a
        // float or a glm vector of them — which is what makes the blanket scan
        // legal here. A NaN sun direction is the classic version of this bug:
        // it survives every normalize, reaches the shadow-cascade fit, and
        // turns the whole frame black.
        if (!all_floats_finite(e)) {
            BRACKEN_LOG_ERROR(LOG_TAG, "environment carries a NaN or an infinity; refusing the level");
            return core::Err(LevelError::BadValue);
        }
        level.environment = e;
    }

    if (doc.read_or<bool>("nav/present", false)) {
        nav::NavGridConfig n;
        (void)doc.read("nav/origin", n.origin);
        (void)doc.read("nav/cell_size", n.cell_size);
        const int64_t width = doc.read_or<int64_t>("nav/width", 0);
        const int64_t depth = doc.read_or<int64_t>("nav/depth", 0);
        // `NavGrid::build` allocates width * depth cells and walks them, so a
        // negative or astronomical pair is a hang or an allocation failure one
        // call after the load returns.
        // Each side is bounded BEFORE the product is formed: two int64s from a
        // file multiplied together overflow, and signed overflow is undefined
        // — the bounds check would be the bug.
        const auto cell_ceiling = static_cast<int64_t>(k_max_nav_cells);
        if (width < 0 || depth < 0 || width > cell_ceiling || depth > cell_ceiling ||
            width * depth > cell_ceiling) {
            BRACKEN_LOG_ERROR(LOG_TAG, "nav grid {}x{} is out of range (0..{} cells)", width, depth,
                              k_max_nav_cells);
            return core::Err(LevelError::BadTable);
        }
        n.width = static_cast<int>(width);
        n.depth = static_cast<int>(depth);
        (void)doc.read("nav/max_slope_degrees", n.max_slope_degrees);
        (void)doc.read("nav/max_step", n.max_step);
        FiniteScan scan;
        scan(n.origin);
        scan(n.cell_size);
        scan(n.max_slope_degrees);
        scan(n.max_step);
        if (!scan.ok) {
            BRACKEN_LOG_ERROR(LOG_TAG, "nav config carries a NaN or an infinity; refusing the level");
            return core::Err(LevelError::BadValue);
        }
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
// Digest and self-verification
// ---------------------------------------------------------------------------

namespace {

/// FNV-1a, byte at a time. The same construction `testing::TestFrame::digest`
/// uses, deliberately: when both numbers appear in one log line they should be
/// the same KIND of number, computed the same way, so a reader is comparing
/// like with like.
struct Fnv1a {
    uint64_t value = 0xcbf29ce484222325ull;

    void bytes(const void* p, std::size_t n) noexcept {
        const auto* q = static_cast<const unsigned char*>(p);
        for (std::size_t i = 0; i < n; ++i) {
            value ^= static_cast<uint64_t>(q[i]);
            value *= 0x100000001b3ull;
        }
    }
    void u32(uint32_t v) noexcept { bytes(&v, sizeof(v)); }
    /// Lengths are folded as a fixed 64-bit width so that concatenating two
    /// tables can never collide with one table of their combined length.
    void count(std::size_t n) noexcept {
        const uint64_t wide = n;
        bytes(&wide, sizeof(wide));
    }
    void text(std::string_view s) noexcept {
        count(s.size());
        if (!s.empty()) { // a default-constructed view has a NULL data()
            bytes(s.data(), s.size());
        }
    }
    /// Whole-table fold. Legal only for the padding-free row types — their
    /// `sizeof` static_asserts in level.hpp are what make the padding bytes
    /// this reads non-existent rather than indeterminate.
    template <typename T>
    void rows(const std::vector<T>& r) noexcept {
        count(r.size());
        if (!r.empty()) {
            bytes(r.data(), r.size() * sizeof(T));
        }
    }
};

/// Records in `extras` that this schema does not own — the ones a save is
/// obliged to carry forward. Used by the round-trip check, which must not
/// count a host's squatted records against the encoder that correctly dropped
/// them.
[[nodiscard]] std::size_t carried_record_count(const save::SaveDocument& extras) {
    std::size_t n = 0;
    for (std::string_view path : extras.paths()) {
        if (!is_owned_path(path)) {
            ++n;
        }
    }
    return n;
}

} // namespace

uint64_t level_digest(const Level& level) noexcept {
    Fnv1a h;
    h.u32(k_level_format_version);
    h.u32(level.content_version);
    h.text(level.name);

    h.count(level.models.size());
    for (std::size_t i = 0; i < level.models.size(); ++i) {
        h.text(level.models[i]);
        h.text(i < level.model_sources.size() ? std::string_view(level.model_sources[i])
                                              : std::string_view{});
    }
    h.count(level.strings.size());
    for (const std::string& s : level.strings) {
        h.text(s);
    }

    h.rows(level.instances);
    h.rows(level.markers);
    h.rows(level.water);
    h.rows(level.scatter_models);
    h.rows(level.scatter_rules);

    // Lights go through their on-disk rows rather than their descs: a desc
    // carries a `bool` and therefore PADDING, and padding bytes after a copy
    // are unspecified — hashing them would make a level's digest differ from
    // itself between two runs of the same binary.
    h.count(level.point_lights.size());
    for (const auto& l : level.point_lights) {
        const PointLightRow r = to_row(l);
        h.bytes(&r, sizeof(r));
    }
    h.count(level.spot_lights.size());
    for (const auto& l : level.spot_lights) {
        const SpotLightRow r = to_row(l);
        h.bytes(&r, sizeof(r));
    }

    h.u32(level.terrain.has_value() ? 1u : 0u);
    if (level.terrain.has_value()) {
        const LevelTerrain& t = *level.terrain;
        h.u32(t.resolution);
        // Field by field rather than one span over the five: a span would be
        // assuming a layout nothing in the header pins.
        h.bytes(&t.origin_x, sizeof(float));
        h.bytes(&t.origin_z, sizeof(float));
        h.bytes(&t.size, sizeof(float));
        h.bytes(&t.min_height, sizeof(float));
        h.bytes(&t.max_height, sizeof(float));
        h.count(t.samples.size());
        if (!t.samples.empty()) {
            h.bytes(t.samples.data(), t.samples.size() * sizeof(uint16_t));
        }
        h.u32(t.splat_resolution);
        h.count(t.splat.size());
        if (!t.splat.empty()) {
            h.bytes(t.splat.data(), t.splat.size());
        }
        h.rows(t.grass_exclusions);
    }

    h.u32(level.environment.has_value() ? 1u : 0u);
    if (level.environment.has_value()) {
        // Float all the way through; see the decode's finiteness scan.
        static_assert(sizeof(bracken::render::EnvironmentState) % sizeof(float) == 0);
        h.bytes(&*level.environment, sizeof(bracken::render::EnvironmentState));
    }

    h.u32(level.nav.has_value() ? 1u : 0u);
    if (level.nav.has_value()) {
        const nav::NavGridConfig& n = *level.nav;
        h.bytes(&n.origin, sizeof(glm::vec3));
        h.bytes(&n.cell_size, sizeof(float));
        h.u32(static_cast<uint32_t>(n.width));
        h.u32(static_cast<uint32_t>(n.depth));
        h.bytes(&n.max_slope_degrees, sizeof(float));
        h.bytes(&n.max_step, sizeof(float));
    }
    return h.value;
}

core::Result<void, LevelError> verify_level_roundtrip(const Level& level) {
    const std::vector<std::byte> bytes = serialize_level(level);
    auto parsed = parse_level(bytes);
    if (parsed.is_err()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "level '{}' does not survive its own encoder: {}", level.name,
                          to_string(parsed.error()));
        return core::Err(parsed.error());
    }

    const uint64_t authored = level_digest(level);
    const uint64_t reloaded = level_digest(parsed.value());
    if (authored != reloaded) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "level '{}' round trip changed its content: digest 0x{:016x} -> 0x{:016x} "
                          "({} bytes)",
                          level.name, authored, reloaded, bytes.size());
        return core::Err(LevelError::Corrupt);
    }
    const std::size_t carried = carried_record_count(level.extras());
    if (carried != parsed.value().extras().record_count()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "level '{}' round trip lost extra records: {} carried, {} came back",
                          level.name, carried, parsed.value().extras().record_count());
        return core::Err(LevelError::Corrupt);
    }
    return {};
}

// ---------------------------------------------------------------------------
// Dump
// ---------------------------------------------------------------------------

std::string level_dump(const Level& level, std::size_t max_rows) {
    // 0 means "all of them" — a tool inspecting a file wants the whole table,
    // and the alternative spelling (SIZE_MAX at every call site) is the kind of
    // thing that gets typed wrong once and silently truncates a report.
    const auto limit = [max_rows](std::size_t n) { return max_rows == 0 ? n : std::min(max_rows, n); };
    std::string out;

    out += std::format("level '{}'  schema v{}  content v{}  digest 0x{:016x}\n", level.name,
                       k_level_format_version, level.content_version, level_digest(level));
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

    const std::size_t instance_rows = limit(level.instances.size());
    out += std::format("\ninstances (showing {} of {}):\n", instance_rows, level.instances.size());
    for (std::size_t i = 0; i < instance_rows; ++i) {
        const LevelInstance& n = level.instances[i];
        // model_name() answers "" for an out-of-range index instead of reading
        // one — dumping is exactly what you do to a file you already suspect.
        out += std::format("  [{:5}] {:24} pos({:.2f}, {:.2f}, {:.2f}) yaw {:.4f} scale {:.4f} "
                           "tint({:.2f}, {:.2f}, {:.2f}, {:.2f}) flags 0x{:x} motion {} lod {}\n",
                           i, level.model_name(n.model), n.position.x, n.position.y, n.position.z, n.yaw,
                           n.scale, n.tint.r, n.tint.g, n.tint.b, n.tint.a, static_cast<uint32_t>(n.flags),
                           n.motion_id, n.lod_hint);
    }

    if (!level.markers.empty()) {
        const std::size_t marker_rows = limit(level.markers.size());
        out += std::format("\nmarkers (showing {} of {}):\n", marker_rows, level.markers.size());
        for (std::size_t i = 0; i < marker_rows; ++i) {
            const LevelMarker& m = level.markers[i];
            out += std::format("  [{:4}] {:20} ({:.2f}, {:.2f}, {:.2f}) yaw {:.4f}\n", i,
                               level.string_at(m.name), m.position.x, m.position.y, m.position.z, m.yaw);
        }
    }

    if (!level.point_lights.empty() || !level.spot_lights.empty()) {
        out += "\nlights:\n";
        for (std::size_t i = 0; i < limit(level.point_lights.size()); ++i) {
            const auto& l = level.point_lights[i];
            out += std::format("  point [{:3}] ({:.1f}, {:.1f}, {:.1f}) r {:.1f} intensity {:.2f} "
                               "source_r {:.1f} shadows {} id {}\n",
                               i, l.position.x, l.position.y, l.position.z, l.radius, l.intensity,
                               l.source_radius, l.casts_shadows ? "yes" : "no", l.stable_id);
        }
        for (std::size_t i = 0; i < limit(level.spot_lights.size()); ++i) {
            const auto& l = level.spot_lights[i];
            out += std::format("  spot  [{:3}] ({:.1f}, {:.1f}, {:.1f}) r {:.1f} cone {:.1f}/{:.1f} deg "
                               "intensity {:.2f} shadows {} id {}\n",
                               i, l.position.x, l.position.y, l.position.z, l.radius, l.inner_angle.value,
                               l.outer_angle.value, l.intensity, l.casts_shadows ? "yes" : "no",
                               l.stable_id);
        }
    }

    for (std::size_t i = 0; i < level.water.size(); ++i) {
        const LevelWater& w = level.water[i];
        out += std::format("\nwater [{}]: x [{:.1f}, {:.1f}] z [{:.1f}, {:.1f}] at y {:.1f}, flow "
                           "({:.2f}, {:.2f}) x {:.1f}, sim {:.0f} around ({:.0f}, {:.0f})\n",
                           i, w.min_x, w.max_x, w.min_z, w.max_z, w.height, w.flow.x, w.flow.y,
                           w.flow_speed, w.sim_extent, w.sim_center.x, w.sim_center.y);
    }

    for (std::size_t i = 0; i < level.scatter_rules.size(); ++i) {
        const LevelScatterRule& r = level.scatter_rules[i];
        out += std::format("\nscatter rule [{}] '{}': seed {} grid step {:.3f} radius {:.1f}..{:.1f} "
                           "jitter {:.2f} sink {:.2f} models [{}..{}) {}\n",
                           i, level.string_at(r.name), r.seed, r.step, r.inner_radius, r.outer_radius,
                           r.jitter, r.sink, r.model_first,
                           static_cast<uint64_t>(r.model_first) + r.model_count,
                           r.baked != 0 ? "(BAKED: its rows are already in instances)"
                                        : "(NOT baked: the host expands it with scatter_ground_cover)");
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

    // The carried records, by path. Not their values: they are the HOST's, of
    // types this build may not know, and the point of listing them is to prove
    // they survived — which is a question about presence, not about content.
    if (level.extras().record_count() != 0) {
        out += "\nextras (carried verbatim, not interpreted):\n";
        for (std::string_view path : level.extras().paths()) {
            out += std::format("  {}\n", path);
        }
    }

    return out;
}

core::Result<void, LevelError> export_level(const Level& level, const std::string& path, bool write_dump) {
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (auto verified = verify_level_roundtrip(level); verified.is_err()) {
        return core::Err(verified.error());
    }
    if (auto written = save_level(level, path); written.is_err()) {
        return core::Err(written.error());
    }
    if (write_dump) {
        const std::string dump_path = path + ".txt";
        if (auto dumped = save_level_dump(level, dump_path); dumped.is_err()) {
            // A missing sidecar is a lost convenience, not a lost level.
            BRACKEN_LOG_WARN(LOG_TAG, "level '{}' was written but its dump sidecar '{}' was not ({})",
                             level.name, dump_path, to_string(dumped.error()));
        }
    }
    return {};
}

core::Result<void, LevelError> save_level_dump(const Level& level, const std::string& path) {
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    const std::string text = level_dump(level, /*max_rows=*/0);
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        BRACKEN_LOG_ERROR(LOG_TAG, "could not open '{}' for the level dump", path);
        return core::Err(LevelError::WriteFailed);
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.flush();
    if (!out) {
        BRACKEN_LOG_ERROR(LOG_TAG, "writing the level dump to '{}' failed", path);
        return core::Err(LevelError::WriteFailed);
    }
    return {};
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
    // Same tier as ResolvedLevel::resolve, and for the same reason: this reads
    // the manager's name table, which a concurrent load would be mutating.
    BRACKEN_ASSERT_NOT_SIM_STAGE();

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
