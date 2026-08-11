// The playground's level exporter, and the one composition site every placed
// prop goes through.
//
// This is the demonstration half of `bracken/scene/level.hpp`: it takes the
// village the demo builds in C++ — 629 hand-authored placements plus 1921
// procedurally filled ground-cover tufts — and writes it as data, so the same
// scene can be rebuilt from a file rather than from a compile.
//
// Kept out of playground.cpp for the reason that file is 3300 lines: the only
// things that had to stay there are the two call sites and `load_level_props`,
// which depends on the collider policy (`classify_collider`) that is game
// code by design.

#include "playground.hpp"

#include "bracken/core/logger.hpp"
#include "bracken/render/water_renderer.hpp"
#include "bracken/scene/level.hpp"
#include "bracken/scene/scene_renderer.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

using bracken::ModelHandle;
namespace level = bracken::scene;

namespace playground {
namespace {

constexpr const char* LOG_TAG = "playground";

// The ground-cover fill parameters, MIRRORED from scatter_wilds() in
// playground.cpp (FLORA_R0 / FLORA_R1 / FLORA_STEP and the two model arrays).
//
// Mirrored rather than shared because the rule this produces is PROVENANCE:
// it is written with `baked = 1`, meaning the 1921 rows it describes are
// already in the instance table and no loader will ever run it. A drift here
// mislabels the file; it cannot change what loads. The rows are baked because
// the demo's fill draws a VARIABLE number of randoms per cell and shares its
// LCG with the rest of on_init, so regenerating it would need the host's exact
// stream — which is the trap `LevelScatterRule`'s fixed-five-draws contract
// exists to avoid for levels authored against the engine's own scatter.
constexpr float k_flora_r0_metres = 13.5f;
constexpr float k_flora_r1_metres = 42.5f;
constexpr float k_flora_step_metres = 1.2f;
constexpr float k_flora_sink = 3.0f;
constexpr float k_village_scale = 100.0f; // village::VILLAGE_SCALE
constexpr float k_world = 8000.0f;        // WORLD in playground.cpp
constexpr uint32_t k_scatter_seed = 0x2f6e2b1u; // the demo's rng_ seed

constexpr const char* k_tuft_names[] = {
    "nature_grass_common_short", "nature_grass_common_tall", "nature_grass_wispy_short",
    "nature_grass_wispy_tall",   "nature_grass_wide_short",  "nature_grass_wide_tall"};
constexpr const char* k_clover_names[] = {"nature_clover_1", "nature_clover_2"};

/// FNV-1a, byte at a time. Same construction as
/// `bracken::testing::TestFrame::digest()`, over the demo's own draw list —
/// the playground cannot include the testing header (it is not part of the
/// supported surface), and props_ is exactly what becomes the packet's model
/// commands, in the same order.
struct Fnv1a {
    uint64_t value = 0xcbf29ce484222325ull;
    void byte(unsigned char b) {
        value ^= static_cast<uint64_t>(b);
        value *= 0x100000001b3ull;
    }
    void bytes(const void* p, std::size_t n) {
        const auto* q = static_cast<const unsigned char*>(p);
        for (std::size_t i = 0; i < n; ++i) {
            byte(q[i]);
        }
    }
};

level::LevelWater to_level_water(const bracken::render::WaterRegion& r) {
    level::LevelWater w;
    w.min_x = r.min_x;
    w.min_z = r.min_z;
    w.max_x = r.max_x;
    w.max_z = r.max_z;
    w.height = r.height;
    w.flow = r.flow;
    w.flow_speed = r.flow_speed;
    w.deep_color = r.deep_color;
    w.shallow_color = r.shallow_color;
    w.absorption = r.absorption;
    w.foam_depth = r.foam_depth;
    w.reflection_roughness = r.reflection_roughness;
    w.reflection_strength = r.reflection_strength;
    w.refraction_strength = r.refraction_strength;
    w.ssr_strength = r.ssr_strength;
    w.ssr_max_distance = r.ssr_max_distance;
    w.ssr_thickness = r.ssr_thickness;
    w.detail_wave_strength = r.detail_wave_strength;
    w.sim_displacement = r.sim_displacement;
    w.sim_normal_strength = r.sim_normal_strength;
    w.ripple_foam_gain = r.ripple_foam_gain;
    w.sim_center = r.sim_center;
    w.sim_extent = r.sim_extent;
    w.wave_speed = r.wave_speed;
    w.wave_damping = r.wave_damping;
    return w;
}

} // namespace

// ---------------------------------------------------------------------------
// The one composition site
// ---------------------------------------------------------------------------

glm::mat4 PlaygroundGame::place_prop(const char* model_name, ModelHandle handle, const glm::vec3& position,
                                     float yaw, float scale) {
    // ONE function builds this matrix, here and in a loaded level. That is the
    // whole reason the export can claim bit-identity rather than "looks the
    // same": composing it twice, in two places, is how the last ulp drifts.
    const glm::mat4 transform = level::level_instance_transform(position, yaw, scale);
    props_.push_back({handle, transform});
    prop_placements_.push_back({model_name, position, yaw, scale});
    return transform;
}

// ---------------------------------------------------------------------------
// The digest instrument
// ---------------------------------------------------------------------------

void PlaygroundGame::log_prop_digest(const char* label) const {
    Fnv1a h;
    for (const Prop& p : props_) {
        const uint32_t handle = p.model;
        h.bytes(&handle, sizeof(handle));
        h.bytes(&p.transform[0][0], sizeof(glm::mat4));
    }
    // The A/B gate. `--export-level v.level` then `--level v.level` must print
    // the same props= and digest= as a plain run; if it does not, the format
    // moved something the screenshot's mean luma would never show.
    BRACKEN_LOG_INFO(LOG_TAG, "[level] source={} props={} digest=0x{:016x}", label, props_.size(), h.value);
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

bool PlaygroundGame::export_level_now() {
    level::Level out;
    out.name = has_village_ ? "playground_village" : "playground_procedural";
    out.content_version = 1;

    // Instances, in props_ order — which IS draw order, and therefore packet
    // order. Reordering this table would produce a pixel-identical frame with
    // a different digest and no way to explain it from the image.
    out.instances.reserve(prop_placements_.size());
    for (const PropPlacement& p : prop_placements_) {
        (void)out.add_instance(p.model, p.position, p.yaw, p.scale);
    }
    // Every model the level references also gets its source path, so a host
    // that does not already have them can feed missing_models() straight into
    // AsyncModelLoader.
    for (std::size_t i = 0; i < out.models.size(); ++i) {
        out.model_sources[i] = "assets/models/" + out.models[i] + ".glb";
    }

    (void)out.add_marker("player_spawn", player_pos_);
    for (const glm::vec3& spot : orb_spots_) {
        (void)out.add_marker("orb_spot", spot);
    }
    for (const glm::vec3& top : chimney_tops_) {
        (void)out.add_marker("chimney", top);
    }

    // Terrain: the heightfield verbatim (a copy, not a requantization).
    //
    // NO SPLATMAP. The demo's 512^2 splat is a pure function of
    // village_layout.h and the river constants, regenerated in ~30 ms by
    // setup_village(); baking it would add 1 MB to a 130 KB file to store
    // something the host recomputes anyway. LevelTerrain carries the field for
    // hosts whose splat IS authored — this one's is not.
    level::LevelTerrain terrain;
    terrain.from_heightmap(heightmap_);
    out.terrain = std::move(terrain);

    out.environment = env_;
    out.nav = nav_.config();

    if (const auto* water = scene_renderer().water(); water != nullptr && water->active()) {
        out.water.push_back(to_level_water(water->region()));
    }

    // The ground-cover layer as a RULE as well as rows — see the mirrored
    // constants above for why `baked` is 1.
    if (has_village_) {
        const uint32_t first = static_cast<uint32_t>(out.scatter_models.size());
        for (const char* name : k_tuft_names) {
            out.scatter_models.push_back(
                level::LevelScatterModel{out.intern_model(name), 0.88f / 6.0f, 0.45f, 0.75f});
        }
        for (const char* name : k_clover_names) {
            out.scatter_models.push_back(
                level::LevelScatterModel{out.intern_model(name), 0.12f / 2.0f, 0.60f, 0.90f});
        }
        level::LevelScatterRule rule;
        rule.name = out.intern_string("ground_cover");
        rule.model_first = first;
        rule.model_count = static_cast<uint32_t>(out.scatter_models.size()) - first;
        rule.seed = k_scatter_seed;
        rule.center_x = k_world * 0.5f;
        rule.center_z = k_world * 0.5f;
        rule.inner_radius = k_flora_r0_metres * k_village_scale;
        rule.outer_radius = k_flora_r1_metres * k_village_scale;
        rule.step = k_flora_step_metres * k_village_scale;
        rule.jitter = 1.0f;
        rule.sink = k_flora_sink;
        rule.baked = 1;
        out.scatter_rules.push_back(rule);
    }

    // The demo's own state, in the level's unknown-record channel: the engine
    // round-trips it and never looks inside. See k_level_rng_key.
    out.extras().write(k_level_rng_key, static_cast<int64_t>(rng_after_scatter_));

    if (auto r = level::save_level(out, export_level_path_); r.is_err()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "--export-level {}: {}", export_level_path_,
                          level::to_string(r.error()));
        return false;
    }

    // The text sidecar. A binary format nobody can read is a format nobody
    // debugs; this one is diagnostic only and has no parser on purpose.
    const std::string dump_path = export_level_path_ + ".txt";
    {
        std::ofstream dump(dump_path, std::ios::trunc);
        if (!dump) {
            BRACKEN_LOG_WARN(LOG_TAG, "--export-level: could not write the dump sidecar {}", dump_path);
        } else {
            dump << level::level_dump(out);
        }
    }

    BRACKEN_LOG_INFO(LOG_TAG, "--export-level {}: {} instances, {} models, {} markers (dump in {})",
                     export_level_path_, out.instances.size(), out.models.size(), out.markers.size(),
                     dump_path);
    return true;
}

} // namespace playground
