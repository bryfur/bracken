# Levels — a serializable `bracken::scene::Level`

A level in bracken used to be C++. The playground's village is 700 lines of
`scene_layout.cpp`, a machine-generated `village_layout.h`, and ~500 lines of
`playground.cpp` that turn those into world state. That works exactly once: you
cannot ship a second level without a second compile, you cannot inspect what a
scene contains without reading its constructor, and nothing outside the demo can
consume it.

This adds the data form of that: an ordered table of placed model instances plus
the static things around them (lights, markers, terrain, water, environment,
navigation, ground-cover fill rules), a versioned checksummed container with an
atomic write, a loader that resolves model names against a `ModelManager` and
counts what it cannot place, and a text dump so the binary is debuggable.

The claim it has to earn is stronger than "it loads": a level written to a file
and read back must describe **the same frame**, not a similar one. The
instrument for that is `bracken::testing::TestFrame::digest()`, which is also
new here.

---

## User guide

### Authoring a level in code

```cpp
#include "bracken/scene/level.hpp"

using namespace bracken::scene;

Level level;
level.name = "riverside";
level.content_version = 3;                  // your revision; the engine never reads it

// Placements. The model name is interned on first use; the index it returns is
// what the instance row stores.
level.add_instance("prop_crate", {1200.0f, 40.0f, -800.0f}, glm::radians(35.0f), 1.4f);
level.add_instance("tree_oak",   {1600.0f, 38.0f, -790.0f}, 0.0f, 9.0f);

// Where the model comes from, so a loader can fetch it (see "Loading" below).
level.model_sources[0] = "assets/models/prop_crate.glb";
level.model_sources[1] = "assets/models/tree_oak.glb";

// Named points. Roles are YOUR vocabulary — the engine defines none of them.
level.add_marker("player_spawn", {1000.0f, 40.0f, -1000.0f}, glm::radians(180.0f));
level.add_marker("orb_spot",     {1100.0f, 40.0f, -1020.0f});

// Static lights, using the same descriptors FrameSink::add_point_light takes.
bracken::render::lighting::PointLightDesc lantern;
lantern.position = {1180.0f, 120.0f, -805.0f};
lantern.radius = 600.0f;
lantern.color = {1.0f, 0.72f, 0.36f};
lantern.intensity = 9.0f;
lantern.source_radius = 40.0f;
level.point_lights.push_back(lantern);

// export_level is the entry point an authoring tool wants: it VERIFIES the
// level against its own encoder first (a level that cannot survive a round
// trip must not become a file), writes it atomically, and drops a readable
// `riverside.level.txt` beside it. save_level is the same write without the
// verification, for a caller that has already done it.
if (auto r = export_level(level, "riverside.level"); r.is_err()) {
    BRACKEN_LOG_ERROR(LOG_TAG, "export failed: {}", to_string(r.error()));
}
```

### Loading and drawing one

```cpp
auto loaded = load_level("riverside.level");
if (loaded.is_err()) {
    BRACKEN_LOG_ERROR(LOG_TAG, "{}", to_string(loaded.error()));
    return false;
}
const Level& level = loaded.value();

// Anything the ModelManager does not have yet, in the level's own table order.
// This is exactly AsyncModelLoader::Request's shape.
std::vector<bracken::AsyncModelLoader::Request> requests;
for (auto& [name, source] : missing_models(level, models())) {
    requests.push_back({name, source});
}
async_loader.request_all(requests);
async_loader.drain();

// Resolve once, at load time. Never per frame.
ResolvedLevel resolved;
const LevelResolveReport report = resolved.resolve(level, models());
if (report.any_dropped()) {
    // Already logged one WARN per unresolved NAME and one summary line; the
    // counts are here if you want them on screen.
}

// Then, every frame, from on_simulate:
void MyGame::on_simulate(bracken::SimFrame& sim) {
    resolved_.emit(sim.frame());     // sim-stage, allocates nothing
}
```

`ResolvedLevel::draws()` is there for a host that already keeps its own prop
vector — copy out of it and keep your own loop; `emit()` is that loop.

### Markers, terrain, water, environment

```cpp
for (const LevelMarker& m : level.markers_named("orb_spot")) {
    spawn_pickup(m.position);
}

if (level.terrain.has_value()) {
    set_heightmap(level.terrain->to_heightmap());
    if (!level.terrain->splat.empty()) {
        scene_renderer().terrain().update_splatmap(level.terrain->splat.data(),
                                                   level.terrain->splat_resolution);
    }
}

if (level.environment.has_value()) {
    env_ = *level.environment;
}

if (!level.water.empty()) {
    const LevelWater& w = level.water.front();
    bracken::render::WaterRegion region;   // MIRRORED field for field — copy across
    region.min_x = w.min_x;  region.max_x = w.max_x;
    region.min_z = w.min_z;  region.max_z = w.max_z;
    region.height = w.height;
    // ... the remaining 26 fields have the same names ...
    scene_renderer().water()->set_region(region);
}
```

### Levels inside an asset pack

`AssetPack::read` hands back bytes, not a path, so a packed level goes through
`parse_level`:

```cpp
auto bytes = pack.read("levels/riverside.level");
if (bytes.is_ok()) {
    auto level = parse_level(bytes.value());
}
```

`serialize_level` is the other half, and produces the exact bytes `save_level`
writes.

### Your own data in a level file

```cpp
level.extras().write("mygame/weather/preset", std::string_view("storm"));
level.extras().write("mygame/quest/act", int64_t{2});
```

Anything under a prefix the schema does not own round-trips untouched, including
through a build that has never heard of it. Namespace it: `level/`, `models/`,
`strings/`, `instances/`, `markers/`, `lights/`, `water/`, `terrain/`, `env/`,
`nav/` and `scatter/` belong to the schema and are regenerated on every save.

### Ground cover as a rule instead of rows

```cpp
const uint32_t first = static_cast<uint32_t>(level.scatter_models.size());
level.scatter_models.push_back({level.intern_model("grass_tuft"), 0.9f, 0.45f, 0.75f});
level.scatter_models.push_back({level.intern_model("clover"),     0.1f, 0.60f, 0.90f});

LevelScatterRule rule;
rule.name         = level.intern_string("ground_cover");
rule.model_first  = first;
rule.model_count  = 2;
rule.seed         = 0x2f6e2b1u;
rule.center_x     = 4000.0f;
rule.center_z     = 4000.0f;
rule.inner_radius = 1350.0f;
rule.outer_radius = 4250.0f;
rule.step         = 120.0f;
rule.jitter       = 1.0f;
rule.sink         = 3.0f;
level.scatter_rules.push_back(rule);

// At load: expand it. One callback answers both "is there ground here" and
// "how high" — two callbacks would let a host answer them inconsistently.
for (const LevelScatterRule& r : level.scatter_rules) {
    if (r.baked != 0) { continue; }          // already in level.instances
    auto rows = scatter_ground_cover(r, level.scatter_slice(r),
        [this](float x, float z) -> std::optional<float> {
            if (is_paved(x, z)) { return std::nullopt; }
            return terrain_height(x, z);
        });
    level.instances.insert(level.instances.end(), rows.begin(), rows.end());
}
```

Forty bytes instead of a hundred kilobytes of rows, and the same rows every
time.

### Reading a level file

```cpp
BRACKEN_LOG_INFO(LOG_TAG, "{}", level_dump(level));        // first 8 rows per table
auto wrote = save_level_dump(level, path + ".txt");        // every row, to a file
```

`level_dump` is diagnostic and has no parser — deliberately. Its second
parameter is a per-table row cap; `0` means all of them, which is what a tool
inspecting a file wants and what `save_level_dump` passes. Every index it
prints goes through `model_name` / `string_at`, which answer `""` for an
out-of-range index rather than reading one: dumping is exactly what you do to a
file you already suspect.

The shipped tool is a mode of the demo, because it needs neither a window nor a
GPU nor the 50 MB of art:

```
playground --dump-level village.level
```

### Verifying an export

`export_level` already does this for you — reach for `verify_level_roundtrip`
directly only when you are not writing a file (a level built in memory and
shipped some other way):

```cpp
if (auto ok = verify_level_roundtrip(level); ok.is_err()) {
    BRACKEN_LOG_ERROR(LOG_TAG, "level will not survive its own encoder: {}",
                      to_string(ok.error()));
    return false;                       // do not ship this file
}
```

`verify_level_roundtrip` serializes, parses the result back, and compares
`level_digest` on both sides plus the count of carried extra records. It is the
round-trip proof as a CALLABLE, so an export tool gates on it instead of a
human comparing two log lines. The failure it actually catches is `BadValue`: a
NaN written by a broken authoring step encodes fine and is refused on the way
back in — at export time, in the tool, rather than six weeks later in a
player's install.

`level_digest` is FNV-1a over the level's canonical content in table order. It
excludes `extras()`, which belongs to the host: a game that writes a frame
counter in there must not thereby make every level unequal to itself.

### The playground

```
playground --export-level village.level     # build the scene, write it, verify, exit
playground --smoke 30 --level village.level # rebuild the props from the file
playground --dump-level village.level       # read it as text
```

`--export-level` proves itself and exits non-zero if it cannot. After writing
the file it reads it back OFF DISK, resolves it against the same
`ModelManager` the live scene used, and compares the result to the live prop
list matrix by matrix — not epsilon-compared, `memcmp`'d, because the whole
claim of `level_instance_transform` is that both sides ran the same three glm
calls on the same three floats:

```
[level] roundtrip OK: 2550 props reload bit-identical, digest=0x…
```

`--level` is the separate, weaker-but-broader gate: the export's own check
proves the DATA survives, and this proves the RENDERER never notices the
difference. Both runs print

```
[level] source=authored props=2550 digest=0x…
[level] source=file     props=2550 digest=0x…
```

and the two digests must match. `--smoke 30` must still report the pinned
`mean_luma` from CLAUDE.md's current column.

---

## Spec

### Files added

| File | What |
|---|---|
| `src/bracken/scene/level.hpp` | the data model, errors, free functions, `ResolvedLevel` |
| `src/bracken/scene/level.cpp` | transform, encode/decode, validation, scatter, dump, digest, resolution |
| `src/bracken/scene/level.test.cpp` | 44 tests, listed at the end of this document |
| `examples/playground/level_export.cpp` | `place_prop`, `export_level_now`, `verify_exported_level`, `prop_digest`, `dump_level_file` |

### Files modified

| File | What |
|---|---|
| `src/bracken/save/save_document.hpp/.cpp` | `serialize()` / `parse(span)` / `value_at` / `write_value`, all additive |
| `src/bracken/testing/frame_test.hpp` | `TestFrame::digest()` |
| `src/bracken/testing/frame_test.test.cpp` | 5 digest tests |
| `src/bracken/save/save_document.test.cpp` | 4 tests for the new methods |
| `src/bracken/bracken.hpp` | one umbrella line for `scene/level.hpp` |
| `examples/playground/{main,playground}.cpp`, `playground.hpp`, `CMakeLists.txt` | the two flags and the load path |

### Records

```cpp
struct LevelInstance {                    // 52 bytes, padding-free, trivially copyable
    uint32_t  model;        // index into Level::models
    uint32_t  motion_id;    // 0 = static
    DrawFlags flags;
    uint32_t  lod_hint;     // opaque host hint; the engine never reads it
    glm::vec3 position;     // RESOLVED world position
    float     yaw;          // radians about +Y
    float     scale;        // uniform
    glm::vec4 tint;
    glm::mat4 transform() const noexcept;   // -> level_instance_transform
};

struct LevelRect        { float min_x, min_z, max_x, max_z; };            // 16
struct LevelMarker      { uint32_t name; glm::vec3 position; float yaw; };// 20
struct LevelWater       { /* mirrors render::WaterRegion, 31 floats */ }; // 124
struct LevelScatterModel{ uint32_t model; float weight, min_scale, max_scale; }; // 16
struct LevelScatterRule { /* name, model slice, seed, grid, jitter, sink, baked */ }; // 48
struct LevelTerrain     { /* resolution, origin, size, height range, samples,
                             splat_resolution, splat, grass_exclusions */ };
```

Every fixed-width record is `static_assert`ed for both size and trivial
copyability, in the header and again in the test — the size **is** the on-disk
row stride.

### `level_instance_transform`

```cpp
glm::mat4 level_instance_transform(const glm::vec3& position, float yaw, float scale) noexcept;
```

`translate -> rotate(yaw, +Y) -> scale(uniform)`, those three glm calls, that
order. It is the format's identity argument, so it exists exactly once, and it
is deliberately **not** `inline`: an inline definition can be instantiated
differently in two translation units (different inlining context, different
floating-point contraction), which is the class of difference it exists to rule
out. `LevelInstance::transform()` and the playground's `place_prop` both call
this one out-of-line function.

Why a TRS and not a stored `mat4`: every one of the reference scene's ~2550
placements is a uniform TRS with a Y-only rotation, so 24 bytes reproduce what
64 would store. Why the *resolved* world position rather than authoring
coordinates plus a terrain lookup: a reload must not depend on the loader owning
the same heightmap sampled the same way.

### The container

A `save::SaveDocument` — magic `BRKSAVE`, format version, host version, payload
length, FNV-1a payload checksum, temp-file + rename write. Reused rather than
reinvented: the atomic write, the checksum, the version refusal and the
unknown-record round trip already exist there and are already tested, and a
second container would have been those four features with a second set of bugs.
(`docs/engine_design_notes.md` decision principle #1.)

What a level adds on top is **blob tables**. `SaveDocument::find` is a linear
scan over a vector and `set` calls it on every write, so path-addressing 2550
instances at five records each is ~81M string comparisons on write. Bulk rows go
into one blob with an explicit stride; only the ~60 scalars and the (small)
model/string tables are path-addressed. That is not an optimization, it is why
the schema looks the way it does.

### On-disk record set

```
level/version            i64    schema version (k_level_format_version)
level/name               str
models/count             i64
models/<i>/name          str    ~52 records — cheap, and readable in a dump
models/<i>/source        str
strings/count            i64
strings/<i>              str
instances/{count,stride,data}        stride 52
markers/{count,stride,data}          stride 20
water/{count,stride,data}            stride 124
scatter/models/{count,stride,data}   stride 16
scatter/rules/{count,stride,data}    stride 48
lights/point/{count,stride,data}     stride 44   (explicit all-scalar row)
lights/spot/{count,stride,data}      stride 64   (explicit all-scalar row)
terrain/present          bool
terrain/{resolution,origin_x,origin_z,size,min_height,max_height}
terrain/samples          blob   resolution^2 * 2 bytes
terrain/splat_resolution i64
terrain/splat            blob   splat_resolution^2 * 4 bytes
terrain/grass_exclusions/{count,stride,data}   stride 16
env/present              bool
env/<field>              29 records, one per EnvironmentState field
nav/present              bool
nav/{origin,cell_size,width,depth,max_slope_degrees,max_step}
```

`Level::content_version` rides `SaveDocument::game_version`.

Lights get an **explicit** row type rather than a memcpy of
`PointLightDesc` / `SpotLightDesc`: those carry a `bool` and therefore padding,
and padding bytes are unspecified after a copy, so memcpy'ing them would make
two saves of the same level differ in bytes and in checksum for no reason a
reader could see. Spot cone half-angles stay in **degrees** on disk, matching
`core::Degrees`.

The environment is 29 path-addressed records rather than one blob: it is the one
section a human edits in a dump, it gains a field every few releases, and
path-addressing is what makes a field added later round-trip by key through an
older build.

### Forward compatibility

Two mechanisms, and the choice between them is forced.

1. **New keys.** An older reader asks for what it knows and ignores the rest.
2. **Wider table rows.** Every table decode copies `min(stride, sizeof(row))`
   bytes per row and leaves the remainder at this build's defaults. A newer
   writer that *appended* a field is readable here; an older file with a shorter
   stride is readable too. Rows are therefore **append-only** and must stay
   padding-free.

What is *not* available: a new value TAG. `SaveDocument`'s parser stops dead on
an unknown tag because it cannot compute the record's length to skip it
(`save_document.cpp`, the `default:` arm). So the level schema never introduces
one.

`k_level_format_version` is bumped only when the record set changes shape in a
way an older build cannot ignore. Adding a key does not bump it; growing a
stride does not bump it. A file whose `level/version` is newer than this build's
is refused (`LevelError::FutureVersion`) rather than guessed at.

### Errors

```cpp
enum class LevelError : uint8_t {
    FileNotFound, NotALevelFile, Corrupt, FutureVersion, WriteFailed,
    BadTable, BadValue
};
const char* to_string(LevelError) noexcept;
```

`BadTable` is a structural contradiction: a `count * stride` that disagrees
with the blob length, a non-positive stride, a terrain sample blob that is not
a whole number of 16-bit samples, a heightmap whose declared resolution does
not square to its sample count, a declared name-table length larger than the
whole document, an out-of-range terrain resolution or nav grid. The checksum
passed in every one of those cases, so it is a writer bug or a hand-edited file
— refused rather than half-read.

`BadValue` is a number that survives the container and could not survive the
renderer: a NaN or an infinity in a position, a scale, a colour, a fog
distance, a cell size. It is refused at LOAD because that is the only place it
is cheap to refuse. A NaN that reaches a transform poisons the whole matrix,
and a NaN matrix in the frame packet propagates into culling (every comparison
against it is false), the bounding-sphere merge and the depth pyramid — the
symptom is a blank or flickering frame several subsystems away from the file
that caused it.

Every fallible entry point returns `core::Result<T, LevelError>` and is
`[[nodiscard]]`.

### Hardening: what a malformed file may do to you

A `.level` is untrusted input the moment it can arrive from anywhere but the
build that wrote it — a mod directory, a download, a partially written file, a
hand-edited one. The contract, case by case in `level.test.cpp`:

* **It never reads outside the bytes it was handed.** The container's cursor
  bounds-checks every take, and every table decode divides rather than
  multiplying, so a `count` can never be trusted into a pointer.
* **It never allocates on a number it read** without first bounding that number
  by something the file cannot inflate. The bulk tables are bounded by their
  blob's actual length; the two path-addressed name tables are bounded by the
  container's own record count (a table of N names occupies 2N records, so a
  count larger than the whole document is a lie the file's own size disproves)
  as well as by `k_max_name_table_rows`; terrain, nav and scatter have explicit
  ceilings.
* **Every refusal is a named `LevelError` with one log line saying which record
  was wrong**, and a decode either returns a complete `Level` or an error —
  never a half-populated one.
* **Indices are the format's only pointers and none of them is followed off the
  end.** `model_name` / `string_at` / `scatter_slice` answer empty for an
  out-of-range index; an instance whose model index is past the table is
  counted in `LevelResolveReport::instances_dropped_bad_index`.

What it deliberately does not do is validate GAMEPLAY: a level may place a
crate under the terrain, aim a spotlight at nothing, or reference a model the
host never registers. Those are counted or simply allowed. Structural
integrity is the engine's business; whether the level is any good is the
game's.

| Limit | Value | What it bounds |
|---|---|---|
| `k_max_name_table_rows` | 65536 | rows in the two path-addressed tables (models, strings) |
| `k_max_table_rows` | 2097152 | rows in ANY blob table. The blob's length bounds the count only by `length / stride`, and the stride comes from the file too: a declared stride of 1 turns a 10 MB blob into ten million 52-byte rows. This is what makes the amplification bounded rather than proportional |
| `k_max_terrain_resolution` | 8192 | heightmap and splatmap resolution, so `res * res` cannot overflow or allocate on a file's say-so |
| `k_max_nav_cells` | 4194304 | `width * depth`, which `nav::NavGrid::build` allocates a cell per |
| `k_max_scatter_cells_per_axis` | 8192 | candidate cells a scatter rule may walk |
| `k_max_scatter_instances` | 1048576 | instances one scatter rule may PRODUCE — the cell bound alone still permits 8192² candidates |

Two holes, stated rather than discovered. The container's `game_version` field
(which carries `Level::content_version`) sits outside the payload checksum, so
a corrupted one loads as a different content version with no error — that is
the container's documented behaviour and the level schema inherits it. And a
finite-but-absurd number (a scale of `1e30`) is accepted: finiteness is
checkable, plausibility is the game's business.

### Resolution and diagnostics

```cpp
struct LevelResolveReport {
    uint32_t instances_placed;
    uint32_t instances_dropped_unknown_model;
    uint32_t instances_dropped_bad_index;    // model index outside Level::models
    uint32_t models_unresolved;              // DISTINCT names, each logged once
    bool any_dropped() const noexcept;
};
```

`resolve` looks each distinct NAME up once (47 hash lookups for 2550 instances,
none in the loop), logs one WARN per unresolved name and one summary line, and
returns the counts. It never `continue`s silently — a level whose art did not
load is the single most common thing to go wrong here.

Model **names**, never handles: `bracken::ModelHandle` is a dense
registration-order index, so a handle written to a file means something
different the moment a host registers its models in another order.

### Scatter rules

```cpp
using ScatterGroundFn = std::function<std::optional<float>(float x, float z)>;
std::vector<LevelInstance> scatter_ground_cover(const LevelScatterRule&,
                                                std::span<const LevelScatterModel>,
                                                const ScatterGroundFn&);
```

Pure and deterministic: same rule, same models, same callback answers → the same
rows, bit for bit, on any thread. The RNG is a named 32-bit LCG written out in
`level.cpp` (`state = state * 1664525 + 1013904223`, top 24 bits taken) rather
than anything from `<random>`, because the sequence is part of the file's
meaning and a seed is worthless if it produces different rows on a different
standard library.

**Five draws per candidate cell — jitter x, jitter z, yaw, model pick, scale —
unconditionally, before any rejection.** That is the property that makes the
seed worth storing. The playground's hand-written fill draws a variable number
(two, then three or four only if the cell survives), so its output cannot be
regenerated after any change to its blocking predicate, the terrain under it, or
the clearing rects: every later cell shifts. A fixed draw count makes the stream
independent of the predicate. `level.test.cpp`'s
`ScatterStreamDoesNotShiftWhenTheGroundRefusesACell` is the pin.

Grid order is x-outer, z-inner, and emission order is draw order, so it is part
of the rule's meaning. A rule with more than 8192 cells per axis is refused with
an ERROR rather than expanded — a metre-scale step over a kilometre-scale radius
is a typo, and expanding it silently turns a level load into a two-minute hang.

`baked != 0` means the rows are already in `Level::instances` and the rule is
provenance; a loader must not run it again.

### `TestFrame::digest()`

```cpp
[[nodiscard]] uint64_t digest() const;   // bracken/testing/frame_test.hpp
```

FNV-1a over everything the **host** described this frame, in packet order:
model draws, skinned draws (palette id plus the palette's own matrices), point
and spot lights, billboards, debug lines, effect spawns, UI commands including
the text bytes, shadow capsules, the camera, the environment, the three
world-element toggles and the build diagnostics. Stream lengths are folded too,
so an empty stream and an absent one cannot collide.

Order-sensitive by design: packet order is what the renderer consumes, so two
frames holding the same three draws in a different order are two different
frames. A digest that called them equal would pass for a level loader that
shuffled its instance table.

Deliberately **not** folded: the engine-stamped timing and clock section (`dt`,
`time`, `frame_index`, `scene_time`, `sky_time`) and the effect-instance
snapshot derived from them. Those are functions of wall clock, not of what the
host asked for. Also not folded: the *value* of the two host-owned definition
pointers the packet carries by documented exception — only their null-ness,
because an address is not stable across runs.

Floats are folded by **bit pattern**, so `-0.0f` and `0.0f` are different frames
and a NaN tint surfaces rather than smoothing away.

**Known blind spot, stated rather than discovered:** it folds the streams that
exist today. A feature that adds a new command stream to `FramePacket` and does
not extend `digest()` leaves it silently insensitive to that stream. Extend it
in the same change that adds the stream.

---

## Architecture

### Where it sits

`scene/` — above `contracts/`, `core/`, `heightmap.hpp`, `save/` and
`navigation/`, below nothing. Three new header edges, all acyclic (`save/` and
`navigation/` include only `core/`, glm and the standard library):

```
scene/level.hpp -> save/save_document.hpp
                -> navigation/nav_grid.hpp
                -> heightmap.hpp
                -> contracts/{environment,light,model_handle}.hpp
                -> core/result.hpp
                -> scene/draw_flags.hpp
```

It does **not** reach `render/`. `render::WaterRegion` is mirrored as
`LevelWater` rather than included, because `render/water_renderer.hpp` pulls
`gpu_buffer`, `gpu_texture` and `<SDL3/SDL_gpu.h>`, and a level header that
dragged the GPU layer into every host TU would be the same mistake
`application.hpp` is pinned against. `ModelManager` is forward-declared in the
header and included only in `level.cpp`, for the same reason.

`scene/level.hpp` is in the `bracken.hpp` umbrella, so it also compiles
standalone in the generated `bracken_public_headers` TU.

### Threading tiers

| Entry point | Tier | Why |
|---|---|---|
| `Level` construction, mutation, queries | `owned` | plain data; touches no engine global |
| `save_level` / `load_level` / `export_level` / `save_level_dump` | main-thread / loading | blocking file IO; `BRACKEN_ASSERT_NOT_SIM_STAGE()` |
| `serialize_level` / `parse_level` | worker-safe | no filesystem, no shared state |
| `ResolvedLevel::resolve` | main-thread / loading | reads the ModelManager name table, which a concurrent load mutates; `BRACKEN_ASSERT_NOT_SIM_STAGE()` |
| `ResolvedLevel::emit` | `sim-stage` | one loop of `FrameSink::draw_model`; allocates nothing, touches nothing shared |
| `scatter_ground_cover` | worker-safe | pure; the only state is a local LCG |
| `level_dump` / `level_digest` | `owned` | pure reads; `level_dump` builds a string |
| `verify_level_roundtrip` | worker-safe | serialize + parse in memory; no filesystem |
| `to_level_water` / `to_water_region` | `owned` | field copies |

Nothing here is on the frame path except `emit`, and `emit` is exactly the loop
a host writes by hand over its own prop vector.

### Lifetimes

`Level` owns every byte it holds — strings, tables, blobs. `ResolvedLevel` owns
its `Draw` vector and holds no pointer into the `Level` it was resolved from, so
the level may be destroyed after resolution. `Level::extras()` is a
`SaveDocument` by value. No API here hands out a pointer into anything with a
shorter lifetime than the object it came from.

### The determinism argument, end to end

1. The host composes a placement through `level_instance_transform`.
2. The exporter stores the three *inputs* to that call, captured at the
   composition site (`PlaygroundGame::place_prop`), never recovered by
   decomposing the matrix afterwards — decomposition is lossy in the last ulp
   and produces a scene that is invisibly different and digest-different.
3. The loader calls the same out-of-line function with the same three floats.
4. `ResolvedLevel::emit` writes them into the packet in file order.
5. `FrameBuilder::seal` merges lanes in (lane-index, emission-order) — already
   byte-identical at any worker count.
6. `TestFrame::digest()` folds the sealed packet in packet order.

Steps 1–4 are pinned by `Level.InstanceTransformIsExactlyTheThreeGlmCalls`,
`Level.EmitPreservesFileOrder` and `Level.FramePacketDigestSurvivesARoundTrip`.
Step 6 is pinned by the five digest tests in `frame_test.test.cpp`.

### The playground path

* `place_prop` is now the single composition site for every placed prop, in both
  the village and the procedural-fallback scenes. It fills `props_` (unchanged
  in shape and order) and `prop_placements_` (the authoring triple, ~60 KB,
  written once at init and read only by the exporter).
* `--export-level <path>` runs at the **tail of `on_init`**, not at the frame
  fence: that is the last moment where every piece is final and no sim job
  exists yet. It writes the `.level` and a `.level.txt` dump, then `quit()`s.
* `--level <path>` replaces `scatter_wilds()` + `build_orb_spots()` with
  `load_level_props()`. Everything else — heightmap, splatmap, water, effects,
  state machine — is generated exactly as without the flag.
* The demo's RNG state after the scatter rides the level's **host-extras**
  channel (`playground/rng_after_scatter`). Without it the orb spawn draws that
  follow in `on_init` would land on different anchors, because a `--level` run
  skips the 1921-cell fill that consumed the stream. This is what
  `Level::extras()` is for, and it doubles as the demonstration that unknown
  records survive.

---

## Deliberate omissions, and why

* **The playground does not export its splatmap.** The demo's 512² splat is a
  pure function of `village_layout.h` and the river constants, regenerated in
  ~30 ms; baking it would add 1 MB to a ~130 KB file to store something the host
  recomputes anyway. `LevelTerrain::splat` exists for hosts whose splat *is*
  authored.
* **The playground bakes its ground cover** (`LevelScatterRule::baked = 1`) and
  writes the rule as provenance. Its fill draws a variable number of randoms per
  cell from an LCG shared with the rest of `on_init`, so regenerating it would
  require reproducing the host's exact stream — the trap the engine's own
  fixed-five-draws contract exists to avoid for levels authored against
  `scatter_ground_cover`.
* **Physics colliders are not level data.** `classify_collider` and
  `nature_trunk_half` are string matches on model names, i.e. game policy, and
  they stay in the host. A level loaded by a *different* host therefore gets no
  colliders. An explicit per-instance collider field is the fix if this ever
  needs to be portable; smuggling the string matching into the engine is not.
* **No editor, no GUI, no runtime streaming.** `LevelInstance::lod_hint` is the
  one concession: an opaque `uint32_t` the engine never reads, so a streaming or
  LOD layer has somewhere to put a cell id without a parallel side table.
* **No hot reload.** `core::asset::FileWatcher` could watch a `.level`; nothing
  in this change needs it.

---

## CLAUDE.md delta

*Module table — replace the `scene/` row's contents cell with:*

> transforms, motion history, `FramePacket` (immutable per-frame command
> snapshot) + `FrameBuilder` (the only write interface) + `FrameSink` (its
> host-facing half, what `SimFrame::frame()` returns) + `DrawFlags` +
> `PacketRing` (SPSC packet ring), `RenderPlan` (which passes run this frame;
> pure, tested headless), `ScenePrep` (the Prepare-stage tier boundary as a
> type: device-free, stateless, cannot reach a main-thread API),
> `SceneRenderer` (frame orchestrator), and **`Level`** — the serializable scene
> (`scene/level.hpp`): placed instances, markers, static lights, terrain, water,
> environment, nav config and procedural ground-cover rules, on top of
> `save::SaveDocument` (atomic write, checksum, version refusal, unknown-record
> round trip). A level stores model **names**, never handles, because
> `ModelHandle` is a registration-order index. `ResolvedLevel` binds one to a
> `ModelManager` and emits it into a `FrameSink` in file order.

*Key patterns — add:*

> - **A level is data, not a compile.** `scene::Level` is an ordered instance
>   table plus the static scene around it, written through
>   `save::SaveDocument` — bulk rows as ONE blob per table with an explicit
>   stride, because `SaveDocument::find` is a linear scan and path-addressing
>   2550 instances is quadratic. Forward compatibility is by KEY (a new record)
>   and by STRIDE (a decode reads `min(stride, sizeof(row))` and leaves the tail
>   at its defaults); a new value TAG is never an option, because the
>   SaveDocument parser stops dead on one. Every placement is a uniform TRS with
>   a Y-only rotation composed by ONE out-of-line function
>   (`scene::level_instance_transform`) — that is what makes a reload
>   bit-identical rather than merely similar, and it is why the function is not
>   `inline`.

*Load-bearing constants — add:*

> - `k_level_format_version = 1` (`scene/level.hpp`) — the level SCHEMA version,
>   distinct from `save::k_save_format_version` (the container's). Bump only
>   when the record set changes shape in a way an older build cannot ignore;
>   adding a key or growing a table's stride does NOT bump it. Row strides are
>   the format: `LevelInstance` 52 B, `LevelMarker` 20 B, `LevelRect` 16 B,
>   `LevelWater` 124 B, `LevelScatterModel` 16 B, `LevelScatterRule` 48 B, all
>   `static_assert`ed padding-free and APPEND-ONLY.
> - The loader's ceilings, all in `scene/level.hpp`, all of them the answer to
>   "what stops a hostile file allocating": `k_max_name_table_rows = 65536`
>   (rows in the two path-addressed tables — the tighter bound is the
>   container's own record count), `k_max_table_rows = 1 << 21` (rows in any
>   blob table, because a declared stride of 1 would otherwise amplify a small
>   blob 52x), `k_max_terrain_resolution = 8192`,
>   `k_max_nav_cells = 4194304` (`width * depth`),
>   `k_max_scatter_cells_per_axis = 8192` (candidate cells a rule may walk) and
>   `k_max_scatter_instances = 1048576` (instances it may produce). Every one
>   is a refusal with a named error, never a truncation.

*Gotchas — add:*

> - **`testing::TestFrame::digest()` folds only the streams that existed when it
>   was written.** It is the equality instrument for the level round trip
>   (`scene/level.test.cpp`), and it covers draws, lights, billboards, debug
>   lines, effect spawns, UI, shadow capsules, palettes, camera, environment and
>   the draw toggles — but NOT the engine-stamped clock section, deliberately.
>   A new packet command stream that does not extend `digest()` leaves it
>   silently blind to that stream.
> - **A level file stores model NAMES.** `ModelHandle` is a dense
>   registration-order index, so two runs that register the same models in
>   different orders produce different handles, a different `FramePacket`
>   digest, and a pixel-identical image. If a digest comparison fails and the
>   screenshot does not move, check the registration order first.
> - **A malformed level is a `LevelError`, never a crash — and `BadValue` is
>   the one that surprises people.** A NaN encodes perfectly well, so a broken
>   authoring step produces a file that saves cleanly and is refused on the way
>   back IN. Run `scene::verify_level_roundtrip` in your export tool and the
>   refusal happens there instead of in a player's install; that is what
>   `playground --export-level` does before it claims success.
> - **`playground --dump-level FILE` needs no window, no GPU and no assets.**
>   It is the mode that still works when the file is the thing that is broken,
>   and it is the only way to see what is actually in a `.level` without
>   writing a parser.
> - **Do not decompose a matrix to author a level instance.** Capture the
>   translate/yaw/scale inputs at the composition site
>   (`PlaygroundGame::place_prop` is the worked example). Recovering them from
>   the product differs in the last ulp, and the result is a scene that looks
>   identical and digests differently.

*Building & testing — add next to `--build-pack`:*

> `playground --export-level village.level` writes the hand-authored village out
> as a `.level` (plus a readable `.level.txt` dump) and exits. **It verifies
> itself**: the level is round-tripped through its own encoder, then read back
> off disk, resolved against the same model manager, and compared to the live
> prop list matrix by matrix — `[level] roundtrip OK: N props reload
> bit-identical` and exit 0, or a `roundtrip FAILED` line and exit 1. The proof
> is a build step, not an eyeball.
>
> `playground --smoke 30 --level village.level` rebuilds the prop / marker /
> collider layer from that file instead of running the C++ layout; that is the
> broader gate, because it proves the RENDERER cannot tell the difference. Both
> runs print `[level] source=… props=N digest=0x…` at init and **the digests
> must match** — that line, not `mean_luma`, is what sees a single prop shifting
> by a ulp. The smoke screenshot pins must also still hold.
>
> `playground --dump-level village.level` prints the file as text and exits;
> it needs no display, no GPU and no assets.

---

## Tests

`src/bracken/scene/level.test.cpp` (51):

| Test | Asserts |
|---|---|
| `InstanceTransformIsExactlyTheThreeGlmCalls` | `memcmp` equality against the literal `translate/rotate/scale` expression, via both the free function and `LevelInstance::transform()` |
| `RecordStridesArePinned` | the six `sizeof`s that are the on-disk strides |
| `RoundTripPreservesEveryField` | a level with every section populated and every optional engaged survives save+load field for field |
| `EmptyOptionalsRoundTripAsEmpty` | absent terrain/env/nav come back absent, not defaulted-present |
| `SerializeParseMatchesTheFileRoundTrip` | bytes path == file path, AND the file's bytes are exactly what `serialize_level` produced (one encoder, not two) |
| `UnknownRecordsSurviveARoundTrip` | foreign records survive two round trips; schema records do NOT leak into `extras()` |
| `ShorterStrideLeavesTheTailAtDefaults` | a 24-byte row leaves `scale` at 1.0, not 0.0, and does not shift the next row |
| `LongerStrideIsSkippedNotMisread` | a 60-byte row is read as 52 and stepped over |
| `MismatchedTableLengthIsBadTableNotACrash` | 10×52 declared against 100 bytes → `BadTable`; stride 0 → `BadTable` |
| `MissingFileIsFileNotFound` | |
| `NotAContainerIsNotALevelFile` | random bytes |
| `AContainerWithoutALevelVersionIsNotALevelFile` | a valid SAVE file is refused as a level |
| `ACorruptPayloadIsRefused` | one flipped payload byte → `Corrupt` |
| `ATruncatedFileIsRefused` | half a file → `Corrupt` |
| `AFutureSchemaIsRefusedRatherThanGuessedAt` | `level/version + 1` → `FutureVersion` |
| `EveryErrorHasAMessage` | no enumerator falls through `to_string` |
| `UnknownModelIsCountedNotSilentlyDropped` | exact drop counts and survivor count against a manager missing one of four names |
| `AnOutOfRangeModelIndexIsCountedSeparately` | a hand-edited index is a different counter |
| `MissingModelsCarriesTheSourceForTheLoader` | (name, source) pairs, in level table order, empty source preserved |
| `EmitPreservesFileOrder` | the packet's draws match `ResolvedLevel::draws()` one for one, transform by `memcmp` |
| **`FramePacketDigestSurvivesARoundTrip`** | 505 instances, one `ModelManager`, in-memory vs save+load → identical digest |
| `ScatterIsDeterministicAndSeedSensitive` | same seed → identical rows; different seed → different; sink and scale range honoured |
| `ScatterStreamDoesNotShiftWhenTheGroundRefusesACell` | every surviving row of a holed run is bit-identical to a row of the unholed run |
| `ScatterRefusesAnAbsurdGridInsteadOfHanging` | 2e9 cells per axis → empty, not a hang |
| `ScatterSliceRefusesAnOutOfRangeRule` | |
| `MarkersAreQueryableByRole`, `OutOfRangeTableIndicesAnswerRatherThanAbort`, `TerrainConvertsToAndFromAHeightmap`, `DumpNamesEverySection` | |

Hostile input (second pass):

| Test | Asserts |
|---|---|
| `ANaNInAnInstanceIsRefused` | a NaN position and an infinite scale each → `BadValue`; the same level with finite values → `is_ok`, so the refusal is about the value and not the shape |
| `ANaNAnywhereElseIsRefusedToo` | seven sections one at a time — env sun dir, terrain origin, nav cell size, a point light's intensity, water damping, a marker position, a scatter step |
| `AnIntegerFieldIsNotMistakenForANaN` | `motion_id = 0x7FC00001` and `lod_hint = 0xFF800000` (NaN and -inf bit patterns) load unchanged — the reason finiteness is checked per FIELD and not as a blanket byte scan |
| `TerrainThatContradictsItsOwnResolutionIsRefused` | resolution that does not square to the sample count, splat resolution that does not match the byte count, and an odd number of sample bytes → `BadTable`. Without this a `Heightmap` indexes at `z * resolution + x` past the end of its data |
| `AnAbsurdTerrainResolutionIsRefusedRatherThanSquared` | −1, `k_max_terrain_resolution + 1`, 2⁴⁰ |
| `AnAbsurdNavGridIsRefused` | negative sides, an over-budget product, AND `2⁴⁰ × 2⁴⁰` — the case a bounds check written as `width * depth > cap` alone would ACCEPT, because the product overflows |
| `ATinyStrideCannotAmplifyASmallBlobIntoAHugeTable` | a 1-byte stride over `k_max_table_rows + 1` rows → `BadTable`; the same shape at 64 rows still loads, because a short stride is the format's forward-compatibility mechanism and not the thing being refused |
| `ANameCountLargerThanTheDocumentIsRefused` | 4096 models declared in a three-record file; `k_max_name_table_rows + 1` strings |
| **`EveryTruncationOfAValidLevelIsRefusedRatherThanCrashing`** | every prefix of a valid file, byte by byte, is a named error and never a parse — the whole fuzz corpus a format this small needs, and the test to run under ASan |
| `EverySingleByteCorruptionIsRefused` | every 17th byte flipped is caught, skipping only the container's unchecked `game_version` field |
| `OutOfRangeIndicesAreRefusedByTheReaderNotFollowed` | model index 4000 of 1, marker name 9999, a scatter slice past the table: each answers empty, the instance is counted as `instances_dropped_bad_index`, and the dump still renders |
| `ScatterOutputIsCappedRatherThanUnbounded` | a rule whose ground accepts every one of 1.44M candidates stops at `k_max_scatter_instances` |

The exporter surface and inspection:

| Test | Asserts |
|---|---|
| `DigestIsFieldSensitiveAndIgnoresHostExtras` | seventeen single-field mutations each move `level_digest`, including SWAPPING two instances (order is content); a record written to `extras()` does not |
| `VerifyRoundTripAcceptsAGoodLevelAndNamesABadOne` | a good level passes; a NaN → `BadValue`; a terrain header contradicting its blob → `BadTable` |
| `HostRecordsUnderASchemaPrefixAreDroppedNotSmuggled` | a host record under `instances/` or `models/` is dropped rather than left in the file to override the schema; its own-prefix records survive |
| `WaterConvertsBothWaysFieldForField` | `to_water_region` → `to_level_water` is `memcmp`-identical over a fully perturbed region |
| `EveryEnvironmentFloatSurvivesTheRoundTrip` | every float slot of `EnvironmentState` filled with a distinct value survives — the tripwire for a field added without a record in `write_env`/`read_env` |
| `DumpShowsEveryRowWhenAskedAndTruncatesByDefault` | `max_rows = 0` shows all 45 instances, the default shows 8, and every section (lights, water, extras, digest) appears |
| `ExportVerifiesBeforeItWritesAndDropsTheSidecar` | a good level writes both files and reloads to the same digest; a NaN level fails verification and leaves NOTHING on disk; `write_dump = false` skips the sidecar and is not an error |
| `DumpOfAnEmptyLevelIsStillAnswerable`, `DumpSidecarIsWrittenAndReadable` | an empty level dumps; the sidecar equals the full dump; an unopenable path is `WriteFailed`, not an abort |

`src/bracken/testing/frame_test.test.cpp` (5 added):
`DigestIsStableForIdenticalFrames`, `DigestIsOrderSensitive` (same two draws,
swapped → different digest), `DigestIsFieldSensitive` (one tint component, one
flag bit, the camera, the environment, a draw toggle, UI text bytes — each moves
it), `DigestIsIndependentOfLaneCount`, `DigestDistinguishesAnEmptyFrame`.

`src/bracken/save/save_document.test.cpp` (4 added):
`SerializeProducesExactlyTheFileBytes` (the gate against a second encoder),
`ParseAcceptsWhatSerializeProduces`, `ParseRefusesGarbageWithoutReadingPastTheSpan`,
`ValueAtCarriesRecordsAcrossDocumentsWithoutKnowingTheirType`.

### What the tests do not cover

**The playground round trip is now self-checking, but it has still never been
run.** `--export-level` performs the whole proof in-process — encoder round
trip, reload off disk, resolve against the live `ModelManager`, `memcmp` every
matrix against `props_` — and exits non-zero if any of it disagrees, so it no
longer depends on a human comparing two log lines. It cannot run in the unit
suite (it needs a GPU, a display and the 50 MB asset set), so the in-tree proof
remains the engine-side digest test above, and this is the gate to run by hand:

```
playground --export-level village.level     # must print [level] roundtrip OK, exit 0
playground --dump-level village.level        # must show 2550 instances, 47 models
playground --smoke 30                        # the pinned mean_luma / stddev
playground --smoke 30 --level village.level  # same [level] digest=, same pins
```

None of the four has been executed: this change was developed against
`-fsyntax-only`, in a worktree with no build tree.

Also uncovered, deliberately:

* **the scatter instance cap under a realistic rule** — the cap test forces it
  with a 1.44M-candidate grid, which is a plausible authoring mistake but not a
  plausible level;
* **`k_max_name_table_rows`, `k_max_terrain_resolution` and `k_max_nav_cells`
  as VALUES.** They are ceilings picked to be far past anything this engine
  renders, not measured limits. The property under test is that a file cannot
  push past them, not that these are the right numbers;
* **concurrent load.** `load_level` is main-thread/loading tier and asserts
  `NOT_SIM_STAGE`; two threads loading into one `ModelManager` is the
  ModelManager's contract, not this format's.
