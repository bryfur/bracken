#pragma once

// bracken/scene/level.hpp — a level as DATA.
//
// Before this, a "level" in this engine was C++: the playground's village is
// 700 lines of `scene_layout.cpp`, a machine-generated `village_layout.h` and
// ~500 lines of `playground.cpp` that turn those into world state. That works
// exactly once — you cannot ship a second level without a second compile, you
// cannot inspect what a scene contains without reading its constructor, and
// nothing outside the demo can consume it.
//
// What a level is here: an ordered table of placed model instances, plus the
// static things around them (lights, markers, terrain, water, environment,
// navigation, ground-cover fill rules). It is a DESCRIPTION, not a runtime:
// nothing in this header owns a GPU resource, opens a device, or touches a
// frame. `ResolvedLevel` binds it to a `ModelManager` once and emits the draws
// into a `FrameSink`; that is the whole runtime surface.
//
// THREE DECISIONS WORTH KNOWING BEFORE READING FURTHER.
//
//   1. A level stores model NAMES, never handles. `bracken::ModelHandle` is a
//      dense registration-order index (`contracts/model_handle.hpp`), so a
//      handle written to a file means something different the moment a host
//      loads its models in another order. Names are the stable identity;
//      resolution happens at load, against a manager, and every name it cannot
//      resolve is COUNTED (`LevelResolveReport`) rather than skipped quietly.
//
//   2. An instance is a uniform TRS with a Y-only rotation, not a mat4. That
//      is not a simplification of the reference scene, it is a description of
//      it: every one of the playground's ~2550 placements is
//      translate -> rotate(+Y) -> scale(uniform). Storing the three inputs and
//      recomposing them through ONE function (`level_instance_transform`) is
//      what makes a reload bit-identical; storing a decomposed mat4 would not
//      be, and storing the mat4 itself costs 64 bytes where 24 do.
//
//   3. The container is `save::SaveDocument`, not a new format. It already has
//      the atomic temp+rename write, the payload checksum, the version refusal
//      and the unknown-record round trip, and it is already tested. A second
//      container would have been the same four features with a second set of
//      bugs. What a level adds on top is BLOB TABLES: SaveDocument's lookup is
//      a linear scan, so path-addressing 2550 instances at five records each
//      is quadratic. Bulk rows go in one blob with an explicit stride; only
//      the ~60 scalars are path-addressed. See docs/gap_level.md.
//
// Tier: `owned`. Build, query and mutate a Level on whatever thread owns it.
// `save_level` / `load_level` do blocking file IO — loading time, never a
// frame. `ResolvedLevel::emit` is `sim-stage` and allocates nothing.

#include "bracken/contracts/environment.hpp"
#include "bracken/contracts/light.hpp"
#include "bracken/contracts/model_handle.hpp"
#include "bracken/core/result.hpp"
#include "bracken/heightmap.hpp"
#include "bracken/navigation/nav_grid.hpp"
#include "bracken/save/save_document.hpp"
#include "bracken/scene/draw_flags.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace bracken {
class ModelManager;
}

namespace bracken::scene {

class FrameSink;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

enum class LevelError : uint8_t {
    FileNotFound,
    NotALevelFile,  ///< a readable container, but not one carrying `level/version`
    Corrupt,        ///< checksum failed, or the payload is truncated
    FutureVersion,  ///< written by a newer level schema than this build knows
    WriteFailed,
    /// A blob table's `count` * `stride` disagrees with the blob's length. The
    /// checksum passed, so this is a WRITER bug or a hand-edited file, not bit
    /// rot — and it is refused rather than half-read.
    BadTable,
};

[[nodiscard]] const char* to_string(LevelError e) noexcept;

/**
 * Bump ONLY when the record set changes shape in a way an older build cannot
 * ignore.
 *
 * Adding a key does NOT bump it (an older reader skips what it does not ask
 * for). Growing a table's stride does NOT bump it either — every table decode
 * reads `min(stride, sizeof(record))` bytes per row and leaves the tail at its
 * default, which is the only forward-compatibility mechanism available here:
 * `save::SaveDocument` stops parsing dead on an unknown value TAG, so a new
 * tag is never an option, and a new key or a wider row always is.
 */
inline constexpr uint32_t k_level_format_version = 1;

// ---------------------------------------------------------------------------
// The instance transform — ONE definition, deliberately out of line
// ---------------------------------------------------------------------------

/**
 * The placed-instance matrix: `translate -> rotate(yaw, +Y) -> scale(uniform)`,
 * composed by exactly those three glm calls in exactly that order.
 *
 * This is the identity argument for the whole format, so it is one function
 * and not a pattern. A host that composes its own placements with the same
 * three calls, exports them, and loads them back gets the same matrix BIT for
 * bit — not "visually the same", the same 16 floats — because both sides
 * execute this function. Recompose it as three separate matrices multiplied
 * together, or re-derive the yaw from degrees on the way in, and the products
 * differ in the last ulp while the image does not: the digest breaks and
 * nothing on screen explains why.
 *
 * It is deliberately NOT inline. An inline definition can be instantiated
 * differently in two translation units (different inlining context, different
 * floating-point contraction), which is exactly the class of difference this
 * function exists to rule out.
 */
[[nodiscard]] glm::mat4 level_instance_transform(const glm::vec3& position, float yaw, float scale) noexcept;

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

/**
 * One placed model instance.
 *
 * `position` is the RESOLVED world position — ground snap, terrain height and
 * any authoring-space sink already applied. Storing authoring coordinates plus
 * "look the height up at load" would make every reload depend on the loader
 * owning the same heightmap, sampled the same way; storing the answer makes it
 * depend on nothing.
 *
 * Field order is the on-disk row order and the stride is `sizeof`. Keep it
 * padding-free (the static_assert below) and APPEND ONLY: a reader with a
 * shorter stride leaves the tail at these defaults, which is only correct if
 * the tail is what was added.
 */
struct LevelInstance {
    uint32_t model = 0;               ///< index into Level::models
    uint32_t motion_id = 0;           ///< host-unique nonzero id for MOVING instances; 0 = static
    DrawFlags flags = DrawFlags::None;
    /**
     * Opaque host hint for level-of-detail / streaming selection. The engine
     * NEVER reads it — it is carried so a streaming or LOD layer has somewhere
     * to put a cell id or a quality tier without a parallel side table. 0 is
     * "no hint", which is what every instance an exporter writes today has.
     */
    uint32_t lod_hint = 0;
    glm::vec3 position{0.0f};         ///< world units
    float yaw = 0.0f;                 ///< radians about +Y
    float scale = 1.0f;               ///< uniform
    glm::vec4 tint{1.0f, 1.0f, 1.0f, 1.0f};

    /// `level_instance_transform(position, yaw, scale)`. See it for why this
    /// forwards instead of composing.
    [[nodiscard]] glm::mat4 transform() const noexcept;
};

static_assert(sizeof(LevelInstance) == 52, "LevelInstance grew padding or a field; the on-disk stride is sizeof");
static_assert(std::is_trivially_copyable_v<LevelInstance>, "LevelInstance rows are memcpy'd into a blob");

/** Axis-aligned world-space XZ rect (grass exclusions, clearings). */
struct LevelRect {
    float min_x = 0.0f;
    float min_z = 0.0f;
    float max_x = 0.0f;
    float max_z = 0.0f;
};
static_assert(sizeof(LevelRect) == 16 && std::is_trivially_copyable_v<LevelRect>);

/**
 * A named point: player spawn, pickup anchor, patrol node, effect emitter site,
 * door. `name` indexes Level::strings, so markers are queryable by ROLE rather
 * than by position in the table — `markers_named("player_spawn")`.
 *
 * Roles are host vocabulary. The engine defines none of them and never will;
 * "spawn point" is a gameplay concept and this header is not the place for it.
 */
struct LevelMarker {
    uint32_t name = 0;
    glm::vec3 position{0.0f};
    float yaw = 0.0f;
};
static_assert(sizeof(LevelMarker) == 20 && std::is_trivially_copyable_v<LevelMarker>);

/**
 * Baked ground.
 *
 * `samples` is `Heightmap::height_data` VERBATIM — row-major `z * resolution +
 * x`, 16-bit normalized against `[min_height, max_height]` — so a load is a
 * copy rather than a requantization. The range is carried explicitly for the
 * reason `heightmap.hpp` gives: a scanned range makes every sample's encoding
 * depend on the extremes, so a reproducible level must pin it.
 *
 * OPTIONAL ON PURPOSE. The playground's 257^2 heightmap is 132 KB and its
 * 512^2 splatmap is 1 MB, against ~130 KB for all 2550 instances: a level that
 * bakes its terrain is an order of magnitude bigger than one that lets the
 * host generate it. Bake it when the terrain is authored; leave it empty when
 * a formula produces it.
 */
struct LevelTerrain {
    uint32_t resolution = 0;
    float origin_x = 0.0f;
    float origin_z = 0.0f;
    float size = 0.0f;
    float min_height = 0.0f;
    float max_height = 0.0f;
    std::vector<uint16_t> samples;

    uint32_t splat_resolution = 0;
    std::vector<uint8_t> splat; ///< RGBA8, the layer weights TerrainRenderer::update_splatmap takes

    std::vector<LevelRect> grass_exclusions;

    /** The engine heightfield this describes. Invalid (see `Heightmap::valid`)
     *  when `resolution < 2` or `samples` is the wrong length. */
    [[nodiscard]] Heightmap to_heightmap() const;

    /** The inverse: copy an engine heightfield in, leaving splat/exclusions
     *  untouched. */
    void from_heightmap(const Heightmap& hm);
};

/**
 * A water region, MIRRORED field for field from `render::WaterRegion`.
 *
 * Mirrored and not included: `render/water_renderer.hpp` pulls in `gpu_buffer`,
 * `gpu_texture` and `<SDL3/SDL_gpu.h>`, and a level header that dragged the GPU
 * layer into every host TU would be the same mistake `application.hpp` is
 * pinned against. Convert at the call site — the field names match one to one.
 */
struct LevelWater {
    float min_x = 0.0f;
    float min_z = 0.0f;
    float max_x = 0.0f;
    float max_z = 0.0f;
    float height = 0.0f;
    glm::vec2 flow{0.0f, -1.0f};
    float flow_speed = 50.0f;
    glm::vec3 deep_color{0.02f, 0.07f, 0.09f};
    glm::vec3 shallow_color{0.20f, 0.42f, 0.40f};
    float absorption = 0.008f;
    float foam_depth = 20.0f;
    float reflection_roughness = 0.14f;
    float reflection_strength = 0.75f;
    float refraction_strength = 0.035f;
    float ssr_strength = 1.0f;
    float ssr_max_distance = 2400.0f;
    float ssr_thickness = 30.0f;
    float detail_wave_strength = 0.12f;
    float sim_displacement = 6.0f;
    float sim_normal_strength = 0.7f;
    float ripple_foam_gain = 0.25f;
    glm::vec2 sim_center{0.0f};
    float sim_extent = 2048.0f;
    float wave_speed = 90.0f;
    float wave_damping = 0.994f;
};
static_assert(sizeof(LevelWater) == 124 && std::is_trivially_copyable_v<LevelWater>);

/** One candidate model of a scatter rule, with its share of the draw. */
struct LevelScatterModel {
    uint32_t model = 0;      ///< index into Level::models
    float weight = 1.0f;     ///< relative selection weight; <= 0 is never picked
    float min_scale = 1.0f;  ///< uniform scale range, inclusive of both ends
    float max_scale = 1.0f;
};
static_assert(sizeof(LevelScatterModel) == 16 && std::is_trivially_copyable_v<LevelScatterModel>);

/**
 * A procedural ground-cover fill, as a RULE plus its seed.
 *
 * WHY A RULE AND NOT ROWS. The playground's tuft carpet is 1921 instances,
 * ~100 KB of table, and none of it is composition: a lawn is texture. Carrying
 * the forty bytes that generate it, plus the seed, is the difference between a
 * level file you can read and one you can only diff.
 *
 * THE FIXED FIVE DRAWS, which is the part that makes a seed worth carrying.
 * `scatter_ground_cover` consumes exactly five random numbers per CANDIDATE
 * CELL — jitter x, jitter z, yaw, model pick, scale — unconditionally, before
 * it asks whether the cell is usable. The playground's hand-written loop draws
 * a variable number (two, then three or four only if the cell survives), which
 * means its output cannot be regenerated after ANY change to its blocking
 * predicate, the terrain under it, or the clearing rects: every later cell
 * shifts. A fixed draw count makes the stream independent of the predicate, so
 * a rule reproduces its own output for as long as the seed and the grid hold,
 * and a predicate change moves only the cells it actually touches.
 *
 * `baked != 0` means the instances this rule produced are ALSO in
 * `Level::instances` and the rule is provenance, not a build step — a loader
 * must not run it again. That is the honest state for an exporter capturing a
 * host loop it did not generate.
 */
struct LevelScatterRule {
    uint32_t name = 0;         ///< index into Level::strings — the layer's role
    uint32_t model_first = 0;  ///< first row of this rule's slice of Level::scatter_models
    uint32_t model_count = 0;
    uint32_t seed = 0;         ///< the whole reason a rule is worth storing
    float center_x = 0.0f;
    float center_z = 0.0f;
    float inner_radius = 0.0f; ///< candidates nearer the centre than this are dropped
    float outer_radius = 0.0f; ///< grid half-extent AND the outer rejection radius
    float step = 1.0f;         ///< grid pitch, world units; <= 0 produces nothing
    float jitter = 1.0f;       ///< per-candidate offset, +/- 0.5 * jitter * step
    float sink = 0.0f;         ///< world units pushed down along -Y after the ground snap
    uint32_t baked = 0;        ///< 1 = already expanded into Level::instances
};
static_assert(sizeof(LevelScatterRule) == 48 && std::is_trivially_copyable_v<LevelScatterRule>);

/**
 * The ground query a scatter rule runs against: the surface height at a world
 * XZ, or `nullopt` for "nothing grows here".
 *
 * One callback for both questions on purpose. A rule that took a height
 * function and a separate blocking predicate would invite a host to answer
 * them inconsistently — a cell blocked by the predicate but still given a
 * height, or the reverse — and the engine cannot tell which one is the lie.
 */
using ScatterGroundFn = std::function<std::optional<float>(float x, float z)>;

/**
 * Expand a scatter rule to instances, in deterministic grid order.
 *
 * Pure: same rule, same models, same callback answers -> same rows, bit for
 * bit, on any thread and at any worker count. It calls nothing but the
 * callback and its own LCG (specified in level.cpp so it cannot drift with a
 * standard-library version).
 *
 * `models` is the rule's slice of `Level::scatter_models`; pass
 * `level.scatter_slice(rule)`.
 */
[[nodiscard]] std::vector<LevelInstance> scatter_ground_cover(const LevelScatterRule& rule,
                                                              std::span<const LevelScatterModel> models,
                                                              const ScatterGroundFn& ground);

// ---------------------------------------------------------------------------
// The level
// ---------------------------------------------------------------------------

/**
 * A level: everything above, plus the string tables the indices point into.
 *
 * `instances` is in DRAW ORDER and that order is part of the contract. The
 * frame packet merges lanes in (lane-index, emission-order) and the renderer
 * consumes exactly that sequence, so a loader that reordered the table would
 * produce a different frame — pixel-identical, digest-different, and
 * impossible to explain from the image.
 */
class Level {
public:
    /** The HOST's own revision of this level's content. The engine round-trips
     *  it and never interprets it — it rides `SaveDocument::game_version`. */
    uint32_t content_version = 0;
    std::string name;

    /// ModelManager registration names — the stable identity (see the header
    /// comment). Parallel to `model_sources`.
    std::vector<std::string> models;
    /// Where each model comes from: a pack entry name or a filesystem path,
    /// exactly the shape `AsyncModelLoader::Request` wants. "" means "the host
    /// registers this one itself" (a procedural mesh, say).
    std::vector<std::string> model_sources;
    /// Marker roles, scatter-layer names — anything indexed by a `uint32_t`.
    std::vector<std::string> strings;

    std::vector<LevelInstance> instances;
    std::vector<bracken::render::lighting::PointLightDesc> point_lights;
    std::vector<bracken::render::lighting::SpotLightDesc> spot_lights;
    std::vector<LevelMarker> markers;
    std::vector<LevelWater> water;
    std::vector<LevelScatterModel> scatter_models;
    std::vector<LevelScatterRule> scatter_rules;

    std::optional<LevelTerrain> terrain;
    std::optional<bracken::render::EnvironmentState> environment;
    std::optional<nav::NavGridConfig> nav;

    /**
     * Records this build did not recognize, carried verbatim and re-emitted on
     * save — the same promise `save::SaveDocument` makes, extended to the
     * level schema: an OLDER build that loads a newer level and re-saves it
     * must not delete the newer build's data.
     *
     * A host may also write its own records here. Namespace them under your
     * own prefix: everything under `level/`, `models/`, `strings/`,
     * `instances/`, `markers/`, `lights/`, `water/`, `terrain/`, `env/`,
     * `nav/` and `scatter/` belongs to this schema and is regenerated on every
     * save.
     */
    [[nodiscard]] const save::SaveDocument& extras() const noexcept { return extras_; }
    [[nodiscard]] save::SaveDocument& extras() noexcept { return extras_; }

    // ---- Authoring ------------------------------------------------------

    /** Index of `name` in `models`, appending it if new. `source` is recorded
     *  on first insert only — a second call with a different source keeps the
     *  first, because two placements of one model cannot disagree about where
     *  it comes from. */
    [[nodiscard]] uint32_t intern_model(std::string_view model_name, std::string_view source = {});

    /** Index of `s` in `strings`, appending it if new. */
    [[nodiscard]] uint32_t intern_string(std::string_view s);

    /** Append a placement, interning its model name. Returns its index in
     *  `instances` — which is its draw position. */
    uint32_t add_instance(std::string_view model_name, const glm::vec3& position, float yaw, float scale,
                          const glm::vec4& tint = {1.0f, 1.0f, 1.0f, 1.0f}, DrawFlags flags = DrawFlags::None,
                          uint32_t motion_id = 0, uint32_t lod_hint = 0);

    /** Append a marker, interning its role. */
    uint32_t add_marker(std::string_view role, const glm::vec3& position, float yaw = 0.0f);

    // ---- Queries --------------------------------------------------------

    /** Markers carrying `role`, in file order. Empty when the role was never
     *  interned — a level with no player spawn is a level, not an error. */
    [[nodiscard]] std::vector<LevelMarker> markers_named(std::string_view role) const;

    /** `models[index]`, or "" when the index is out of range. Out of range is
     *  reachable from a hand-edited file, so it answers rather than aborts. */
    [[nodiscard]] std::string_view model_name(uint32_t index) const noexcept;
    [[nodiscard]] std::string_view string_at(uint32_t index) const noexcept;

    /** A scatter rule's slice of `scatter_models`. Empty when the rule's slice
     *  runs past the table (again: hand-edited files exist). */
    [[nodiscard]] std::span<const LevelScatterModel> scatter_slice(const LevelScatterRule& rule) const noexcept;

private:
    save::SaveDocument extras_;
};

// ---------------------------------------------------------------------------
// Files and bytes
//
// Tier: main-thread / loading. `save_level` and `load_level` do blocking IO —
// never call them from `on_simulate`. `parse_level` and `serialize_level` are
// worker-safe: they touch no filesystem.
// ---------------------------------------------------------------------------

/** Atomic (temp + rename) and checksummed, inherited from
 *  `save::SaveDocument::save` — an interrupted write leaves the previous file
 *  intact. */
[[nodiscard]] core::Result<void, LevelError> save_level(const Level& level, const std::string& path);

[[nodiscard]] core::Result<Level, LevelError> load_level(const std::string& path);

/** For a level shipped inside a pack: `core::asset::AssetPack::read` hands
 *  back exactly this span. */
[[nodiscard]] core::Result<Level, LevelError> parse_level(std::span<const std::byte> bytes);

[[nodiscard]] std::vector<std::byte> serialize_level(const Level& level);

/**
 * A human-readable dump: counts, the model and string tables, the terrain and
 * environment headers, and the first rows of each table.
 *
 * A binary format nobody can read is a format nobody debugs, and this is the
 * answer to that — write it next to the `.level` as a sidecar, or print it
 * from a tool. It is DIAGNOSTIC, not a second serialization: there is no
 * parser for it and there should not be one.
 */
[[nodiscard]] std::string level_dump(const Level& level);

// ---------------------------------------------------------------------------
// Resolution against a ModelManager
// ---------------------------------------------------------------------------

/**
 * What resolution refused. The engine does not drop work silently, and a level
 * whose art did not load is the single most common thing to go wrong here — a
 * renamed .glb, a partial checkout, a pack built before the level was.
 */
struct LevelResolveReport {
    uint32_t instances_placed = 0;
    /// Instances whose model name the manager did not know.
    uint32_t instances_dropped_unknown_model = 0;
    /// Instances whose `model` index was outside `Level::models` — a
    /// hand-edited or truncated file, counted separately because the fix is
    /// different.
    uint32_t instances_dropped_bad_index = 0;
    /// DISTINCT unresolved names, each logged once at WARN.
    uint32_t models_unresolved = 0;

    [[nodiscard]] bool any_dropped() const noexcept {
        return instances_dropped_unknown_model != 0 || instances_dropped_bad_index != 0;
    }
};

/**
 * A level bound to a ModelManager: names resolved to handles once, transforms
 * composed once, ready to emit every frame.
 *
 * Rebuild it when the level or the model table changes, not per frame — that
 * is the entire reason it is a type and not a function.
 */
class ResolvedLevel {
public:
    struct Draw {
        bracken::ModelHandle model = bracken::INVALID_MODEL_HANDLE;
        glm::mat4 transform{1.0f};
        glm::vec4 tint{1.0f, 1.0f, 1.0f, 1.0f};
        DrawFlags flags = DrawFlags::None;
        uint32_t motion_id = 0;
        /// Carried through from `LevelInstance::lod_hint`, untouched.
        uint32_t lod_hint = 0;
    };

    /**
     * Resolve every instance, in file order. Replaces any previous contents.
     *
     * Main-thread / loading tier: it reads the ModelManager's name table, which
     * a concurrent load would be mutating. Asserted.
     */
    LevelResolveReport resolve(const Level& level, const bracken::ModelManager& models);

    /**
     * Emit every resolved draw into the frame, in file order.
     *
     * Tier: `sim-stage`. Allocates nothing and touches no shared state — it is
     * the same loop a host writes over its own prop vector, which is why a
     * host that already has one should keep it and call `draws()` instead.
     */
    void emit(FrameSink& sink) const;

    [[nodiscard]] std::span<const Draw> draws() const noexcept { return draws_; }
    [[nodiscard]] std::size_t size() const noexcept { return draws_.size(); }
    void clear() noexcept { draws_.clear(); }

private:
    std::vector<Draw> draws_;
};

/**
 * (name, source) for every model the level references that `models` does not
 * already hold, in level table order.
 *
 * The shape `AsyncModelLoader::Request` takes, so a host's load path is two
 * lines. Note the order: it is the LEVEL's model order, not the host's, and a
 * host that cares about handle stability across a hand-authored and a
 * level-driven run must register in one fixed order and use it for both —
 * handles are registration-order indices.
 */
[[nodiscard]] std::vector<std::pair<std::string, std::string>>
missing_models(const Level& level, const bracken::ModelManager& models);

} // namespace bracken::scene
