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
#include <iostream>
#include <string>
#include <vector>

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

} // namespace

// ---------------------------------------------------------------------------
// Inspection — `--dump-level FILE`
// ---------------------------------------------------------------------------

bool dump_level_file(const std::string& path) {
    // No window, no device, no assets: this is the mode that still works when
    // the file is the thing that is broken.
    auto loaded = level::load_level(path);
    if (loaded.is_err()) {
        std::cerr << "--dump-level " << path << ": " << level::to_string(loaded.error()) << "\n";
        return false;
    }
    std::cout << level::level_dump(loaded.value(), /*max_rows=*/0);
    return true;
}

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

uint64_t PlaygroundGame::prop_digest() const {
    Fnv1a h;
    for (const Prop& p : props_) {
        const uint32_t handle = p.model;
        h.bytes(&handle, sizeof(handle));
        h.bytes(&p.transform[0][0], sizeof(glm::mat4));
    }
    return h.value;
}

void PlaygroundGame::log_prop_digest(const char* label) const {
    // The A/B gate. `--export-level v.level` then `--level v.level` must print
    // the same props= and digest= as a plain run; if it does not, the format
    // moved something the screenshot's mean luma would never show.
    BRACKEN_LOG_INFO(LOG_TAG, "[level] source={} props={} digest=0x{:016x}", label, props_.size(),
                     prop_digest());
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

bool PlaygroundGame::export_level_now() {
    // `--export-level` with `--level` is refused rather than half-done.
    // load_level_props() fills props_ but CANNOT fill prop_placements_ — a
    // PropPlacement holds a `const char*` model name and a loaded level's names
    // die with the local Level — so the export would write a level with zero
    // instances. And because export_level() writes before verify_exported_level
    // runs, pointing both flags at one path would truncate the very file being
    // read. Refuse before a byte moves.
    if (!level_path_.empty()) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "--export-level cannot be combined with --level: a level loaded from '{}' carries "
                          "no authoring triples to re-export, and exporting to that same path would destroy "
                          "it. Run --export-level against a hand-authored scene.",
                          level_path_);
        return false;
    }

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
        // The engine's conversion, not a hand-written copy of 25 fields — a
        // field added to WaterRegion is then wrong in one place instead of two.
        out.water.push_back(level::to_level_water(water->region()));
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

    // Model sources LAST, after every intern_model above — including the
    // scatter block's, which appends any candidate model the instance table did
    // not already name. Filling them earlier left those rows with an empty
    // source, which missing_models() would then hand to a host's loader as
    // "you register this one yourself".
    for (std::size_t i = 0; i < out.models.size(); ++i) {
        out.model_sources[i] = "assets/models/" + out.models[i] + ".glb";
    }

    // The demo's own state, in the level's unknown-record channel: the engine
    // round-trips it and never looks inside. See k_level_rng_key.
    out.extras().write(k_level_rng_key, static_cast<int64_t>(rng_after_scatter_));

    // ONE engine call: it verifies the level against its own encoder before a
    // byte reaches the disk, writes it atomically, and drops the readable
    // `.txt` sidecar beside it. The demo does not own that policy — a game's
    // own export tool gets the same three steps in the same order for free.
    if (auto r = level::export_level(out, export_level_path_); r.is_err()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "--export-level {}: {}", export_level_path_,
                          level::to_string(r.error()));
        return false;
    }

    BRACKEN_LOG_INFO(LOG_TAG, "--export-level {}: {} instances, {} models, {} markers (dump in {}.txt)",
                     export_level_path_, out.instances.size(), out.models.size(), out.markers.size(),
                     export_level_path_);

    return verify_exported_level(out, export_level_path_);
}

// ---------------------------------------------------------------------------
// The round-trip proof
// ---------------------------------------------------------------------------

bool PlaygroundGame::verify_exported_level(const level::Level& authored, const std::string& path) {
    auto reloaded = level::load_level(path);
    if (reloaded.is_err()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "[level] roundtrip FAILED: the file just written will not load ({})",
                          level::to_string(reloaded.error()));
        return false;
    }
    if (level::level_digest(reloaded.value()) != level::level_digest(authored)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "[level] roundtrip FAILED: the file's content digest does not match "
                                   "the level it was written from");
        return false;
    }

    // Resolve through the SAME ModelManager the live scene used. That is what
    // makes the comparison meaningful: ModelHandle is a registration-order
    // index, so a resolution against a differently loaded manager would be
    // testing handle assignment rather than the format.
    level::ResolvedLevel resolved;
    const level::LevelResolveReport report = resolved.resolve(reloaded.value(), models());
    if (report.any_dropped()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "[level] roundtrip FAILED: {} instances dropped on reload ({} unknown "
                                   "model, {} bad index)",
                          report.instances_dropped_unknown_model + report.instances_dropped_bad_index,
                          report.instances_dropped_unknown_model, report.instances_dropped_bad_index);
        return false;
    }
    if (resolved.size() != props_.size()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "[level] roundtrip FAILED: {} draws reloaded against {} live props",
                          resolved.size(), props_.size());
        return false;
    }

    // Matrix by matrix, byte for byte. NOT epsilon-compared: the whole claim of
    // level_instance_transform is that both sides run the same three glm calls
    // on the same three floats, so anything but equality is a real defect —
    // and one that no screenshot would ever show.
    Fnv1a h;
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < props_.size(); ++i) {
        const level::ResolvedLevel::Draw& draw = resolved.draws()[i];
        const uint32_t handle = draw.model;
        h.bytes(&handle, sizeof(handle));
        h.bytes(&draw.transform[0][0], sizeof(glm::mat4));
        if (draw.model != props_[i].model ||
            std::memcmp(&draw.transform[0][0], &props_[i].transform[0][0], sizeof(glm::mat4)) != 0) {
            if (mismatches < 4) {
                BRACKEN_LOG_ERROR(LOG_TAG, "[level] roundtrip: instance {} differs (model {} vs {})", i,
                                  draw.model, props_[i].model);
            }
            ++mismatches;
        }
    }

    const uint64_t live = prop_digest();
    if (mismatches != 0 || h.value != live) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "[level] roundtrip FAILED: {} of {} instances differ; digest 0x{:016x} "
                          "reloaded vs 0x{:016x} live",
                          mismatches, props_.size(), h.value, live);
        return false;
    }

    BRACKEN_LOG_INFO(LOG_TAG, "[level] roundtrip OK: {} props reload bit-identical, digest=0x{:016x}",
                     props_.size(), live);
    return true;
}

} // namespace playground
