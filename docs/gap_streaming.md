# Streaming gap: paged terrain, asset residency, occlusion culling

Three landings, in the order the survey brief lists them:

* **(a) paged terrain** — `HeightmapGrid` + per-page GPU terrain. Landed first;
  documented in the first half of this file.
* **(b) asset residency** — a byte budget, a distance band with hysteresis, and
  per-asset residency state over the existing `AsyncModelLoader`.
* **(c) occlusion culling for draws** — a farthest-surface Hi-Z pyramid and a
  conservative CPU test in the Prepare stage. **Default off, and it must stay
  off until somebody measures it.**

---

# Part 1 — Paged terrain (stage 1)

The world was exactly one square heightmap. `Heightmap` was already a *tile*
type in shape — it carries its own origin, its own size and its own
quantization range — the engine simply never made a second one, and
`TerrainRenderer` baked that single map into one vertex buffer, one index
buffer, one R16 texture and one tile list.

Stage 1 of the streaming gap makes the terrain an **N x M grid of pages**:
`HeightmapGrid` on the CPU side, one GPU page per grid cell in the renderer,
and a page-AABB reject in front of the tile culling at all three draw sites.
`set_heightmap(const Heightmap&)` becomes the 1x1 adapter, so every existing
host keeps working, untouched, and keeps producing the identical frame.

Residency, eviction and occlusion culling are **not** in this landing — see
"What is not here" at the end.

---

## User guide

### The single-map world is unchanged

Nothing to do. `Application::set_heightmap` still takes a `Heightmap` and
still means what it meant; it is now spelled as a 1x1 grid internally.

```cpp
bracken::Heightmap map = bracken::Heightmap::from_floats(
    heights, 257, {0.0f, 0.0f, 8000.0f}, bracken::HeightRange{-160.0f, 150.0f});
set_heightmap(map);          // exactly as before
```

### Authoring a paged world

Build the grid, then fill pages. **Do not** pack pages yourself with the
scanning `Heightmap::from_floats` overload — see the trap below.

```cpp
#include "bracken/bracken.hpp"   // exports heightmap_grid.hpp

constexpr uint32_t PAGES_X = 4, PAGES_Z = 4;
constexpr float    PAGE    = 4000.0f;   // world units per page side
constexpr uint32_t RES     = 257;       // samples per page side
constexpr bracken::HeightRange RANGE{-400.0f, 600.0f};   // SHARED by every page

bracken::HeightmapGrid grid =
    bracken::HeightmapGrid::create(PAGES_X, PAGES_Z, PAGE, /*origin_x=*/0.0f,
                                   /*origin_z=*/0.0f, RANGE);

std::vector<float> samples(static_cast<std::size_t>(RES) * RES);
for (uint32_t pz = 0; pz < PAGES_Z; ++pz) {
    for (uint32_t px = 0; px < PAGES_X; ++px) {
        const float ox = static_cast<float>(px) * PAGE;
        const float oz = static_cast<float>(pz) * PAGE;
        for (uint32_t iz = 0; iz < RES; ++iz) {
            for (uint32_t ix = 0; ix < RES; ++ix) {
                // NOTE the divisor: RES - 1, so the FIRST and LAST sample of a
                // page sit ON its boundary and the neighbour repeats them.
                const float x = ox + PAGE * static_cast<float>(ix) / static_cast<float>(RES - 1);
                const float z = oz + PAGE * static_cast<float>(iz) / static_cast<float>(RES - 1);
                samples[static_cast<std::size_t>(iz) * RES + ix] = my_terrain_height(x, z);
            }
        }
        if (!grid.set_page(px, pz, samples, RES)) {
            /* logged; the page stays absent */
        }
    }
}

set_heightmap_grid(grid);    // Application::set_heightmap_grid
```

`set_page` derives the page's world bounds from the grid and packs against the
grid's `range`, so a host never computes a page origin and never chooses a
range — the two ways to get this wrong.

### The trap this API exists to close

`Heightmap::from_floats` has two overloads. The scanning one derives the
16-bit quantization range from the data it is handed, which is the most
precise choice for one map and the **wrong** choice for a page: two pages
authored independently quantize against different extremes, so the heights
they store for the samples on the line they share disagree, and the seam
**steps** — with nothing logged anywhere. The header has warned about this
since before there was a grid.

`HeightmapGrid::validate` therefore makes it a hard refusal rather than a
preference. A grid whose resident pages disagree about their range is not
`valid()`, and `Application::set_heightmap_grid` /
`SceneRenderer::set_heightmap_grid` / `TerrainRenderer::set_heightmap_grid`
log the offending page and **keep the previous terrain**. There is no
tolerance on the range comparison: a range is copied between pages, never
recomputed, so any difference at all is an authoring mistake. Page geometry
(origin, world size) gets a small tolerance, because a host may legitimately
compute `origin + px * page_size` by a different but equivalent route.

### Partly-streamed grids

A page may be absent. That is not an error — it is what a streamed world looks
like most of the time:

```cpp
grid.clear_page(2, 1);                        // page unloaded
grid.resident_at(9000.0f, 5000.0f);           // false
grid.get_height_world(9000.0f, 5000.0f, -1e30f);   // returns the fallback
```

An absent page renders as a flat placeholder mesh (the same placeholder the
renderer draws before any heightmap arrives) and samples as `fallback`. The
grid still `valid()`s.

### Known limitation: grass follows a smoothed terrain on a paged world

`sampleTerrainHeight` (`src/bracken/shaders/grass_blade.hlsli`) hard-codes ONE
height texture, ONE origin and ONE square world size, and the offsets of the
matching `HeightmapParams` (`src/bracken/render/grass_renderer.hpp`) are
static-asserted on the CPU side and SPIR-V-reflected by the uniform-layout
test. Rather than move that cbuffer in this landing, a **paged** grid gets a
coarse world-spanning R16 **proxy** texture resampled through the grid
sampler, capped at 1024 x 1024. Grass therefore sits on a smoothed version of
the terrain when the world is paged.

A **1x1** grid does not use the proxy at all: grass gets that page's own
texture and the map's own five scalars, i.e. exactly the pre-paging binding.

Per-page grass sampling wants a `Texture2DArray` plus a page index in the
shader, which is a two-sided cbuffer change; it is deliberately not in this
landing.

---

## Spec

### `HeightmapGrid` (`src/bracken/heightmap_grid.hpp`)

Root layer, namespace `bracken`, exported from `src/bracken/bracken.hpp`.
Plain data plus const sampling; no GPU, no device, no allocation on the read
path.

| Field | Meaning |
|---|---|
| `pages_x`, `pages_z` | grid shape, row-major |
| `page_size` | world units per page side (pages are SQUARE) |
| `origin_x`, `origin_z` | world position of page (0,0)'s -x/-z corner |
| `range` | the `HeightRange` every resident page must be packed against |
| `pages` | `pages_z * pages_x` `Heightmap`s; invalid entries are absent pages |

**Page coverage.** Page (px,pz) covers the CLOSED world square
`[origin + px*page_size, origin + (px+1)*page_size]` on both axes. Adjacent
pages therefore SHARE their boundary sample line — page (0,0)'s last column
and page (1,0)'s first column are at the same world x and must hold the same
heights. Continuity across a seam is a property of the authored data plus the
shared quantization range, not of any blending at sample time.

| API | Contract |
|---|---|
| `from_single(const Heightmap&)` | the 1x1 grid; carries the map's origin, size and range |
| `create(pages_x, pages_z, page_size, origin_x, origin_z, range)` | empty grid of the given shape |
| `set_page(px, pz, heights, resolution)` | packs row-major world-unit heights against the grid's range and derived bounds; `false` + log on a bad index or sample count |
| `clear_page(px, pz)` | makes a page absent |
| `validate()` | `core::Result<void, std::string>`; the error names the page and the disagreement |
| `valid()` | `validate().is_ok()` |
| `page_at(x, z)` | page index, or `{-1,-1}` strictly outside the grid (the outer boundary line is inside) |
| `page(px, pz)` | the resident page, or null (out of range **or** absent) |
| `page_min_xz` / `page_max_xz` | a page's world XZ corners |
| `get_height_world(x, z, fallback = 0)` | bilinear, page index CLAMPED to the grid so outside queries return the grid edge height; `fallback` when the containing page is absent |
| `get_normal_world(x, z)` | central differences, `eps = page_size / (resolution - 1)`, taps taken THROUGH `get_height_world` so a seam-straddling stencil reads both pages; `(0,1,0)` when degenerate or nothing resident |
| `resident_at(x, z)` | is the containing (clamped) page loaded |
| `resident_pages()`, `page_resolution()`, `world_size_x()`, `world_size_z()`, `page_count()` | inspection |

**What `validate()` refuses**

1. `pages_x == 0` or `pages_z == 0`; `page_size <= 0`.
2. `pages.size() != pages_x * pages_z`.
3. A resident page whose `min_height`/`max_height` is not EXACTLY the grid's
   `range` — the seam trap. No tolerance.
4. A resident page whose `resolution` differs from a sibling's.
5. A resident page whose `world_size` or origin is not the one the grid's
   addressing assumes, beyond `geometry_tolerance(reference)` =
   `max(0.01, 1e-5 * |reference|)` world units.

A grid with **no** resident pages is valid.

**Constants**

| Constant | Value | Why |
|---|---|---|
| `HeightmapGrid::k_recommended_min_page_size` | `2000.0f` | advisory floor on page size; below it `TerrainRenderer` warns once. A 2000-unit page is 80x80 cells of the 25-unit terrain grid — large enough that the tile-run merging inside a page still does real work |
| `TerrainRenderer::k_page_margin` | `500.0f` | the terrain skirt outside the world, applied on the grid's OUTER edges only. This is the value both mesh generators used privately before paging |
| `TerrainRenderer::k_grass_proxy_max_resolution` | `1024` | cap on the paged-world grass proxy's side (2 MB of R16, ~1M grid samples at load) |

### `TerrainRenderer` (`src/bracken/render/terrain_renderer.hpp`)

New public surface:

```cpp
void set_heightmap(const bracken::Heightmap&);          // 1x1 adapter
void set_heightmap_grid(const bracken::HeightmapGrid&); // refuses an invalid grid
const bracken::HeightmapGrid& heightmap_grid() const;
std::size_t resident_pages() const;
std::size_t page_bytes() const;
struct HeightmapSampleParams { float world_origin_x, world_origin_z, world_size, min_height, max_height; };
HeightmapSampleParams heightmap_sample_params() const;
gpu::GPUTexture* heightmap_texture();   // the ONE texture grass samples
```

Private `TerrainPage`, one per grid cell: its own vertex/index buffers, its
own low-LOD shadow buffers, its own R16 heightmap texture, its own
`tiles`/`rows` (and `shadow_tiles`/`shadow_rows`), and the union AABB of each.

**The mesh rect rule.** A page's mesh spans:

* **1x1 grid** — `[-500, world_width + 500] x [-500, world_height + 500]`,
  i.e. the renderer's init-time world rect plus the margin, exactly as before
  paging. This rect has never been derived from the heightmap (a map may be
  any size; sampling clamps), and keeping it is what makes the adapter
  bit-identical.
* **paged grid** — the page's own square from the grid, with the 500-unit
  margin added only where the page is on the grid's outer edge. Adding it at
  an interior seam would overlap the neighbour and z-fight.

Everything else in the generators is unchanged: cell size 25, 32-cell tiles
for the main mesh; cell size 100, 16-cell tiles, the conservative 5x5 MIN
probe and the 5-unit sink for the shadow mesh; the distance/height vertex tint
still measured from the renderer's world centre.

**The page reject.** At each of the three draw sites (`render`,
`render_prepass`, `render_shadow`) the loop is:

```
for each page:
    skip if the page has no buffers
    skip if a frustum was supplied and it misses the page's union AABB
    bind the page's vertex + index buffers
    cull_tile_runs(...) on the page's own tiles/rows   // UNCHANGED
    one draw per run
```

`cull_tile_runs` and `build_tile_rows` are untouched — still static, still
pure, still deterministic at any worker count. The reject is safe because a
page's box is the union of its tile boxes: a frustum that misses the union
misses every tile in it, so the reject can only skip work the per-tile cull
would have discarded anyway. That is the property that makes a 1x1 grid
produce the identical draw stream.

**Draw-call arithmetic.** Runs are index-buffer ranges and each page owns its
own index buffer, so runs never merge across pages: the floor is one draw per
visible page in each of the six terrain walks a frame (main, depth prepass,
four shadow cascades), i.e. `6 * P`. The ceiling is `6 * sum(runs per page)`.
The total is **not** ordered against the unpaged one — a page renumbers its
tiles into its own buffer, so tiles from consecutive page rows are contiguous
there while in the world-sized buffer they were separated by the neighbouring
page's columns, and the page's runs can merge where the world's could not.
This was measured, not assumed (`TerrainPageCulling` in
`src/bracken/render/terrain_renderer.test.cpp`). What is guaranteed is the
floor and that the same triangles are drawn.

### `SceneRenderer` / `Application`

```cpp
void SceneRenderer::set_heightmap(const Heightmap&);        // = set_heightmap_grid(from_single(m))
void SceneRenderer::set_heightmap_grid(const HeightmapGrid&);
void Application::set_heightmap_grid(const HeightmapGrid&); // CONTRACT: init_renderer() first
```

`Application::set_heightmap_grid` validates and refuses with a logged error,
the same policy `Application::set_heightmap` already applies to an invalid
`Heightmap`.

---

## Architecture

### Where it sits in the frame

Nowhere new. Paging is entirely an **init/fence-time** restructuring of the
terrain's GPU state plus a per-page loop inside the three existing terrain
draw calls. There is no new pass, no new pipeline, no new cbuffer, no new
shader, and no change to `RenderPlan`, `ScenePrep` or the frame packet.

* `TerrainRenderer::render` — main forward pass, inside the main render pass.
* `TerrainRenderer::render_prepass` — depth prepass. The forward pass
  depth-tests EQUAL against it, so the two walks MUST draw the same set: the
  page reject is the identical predicate on the identical AABB in both, not an
  approximation.
* `TerrainRenderer::render_shadow` — once per CSM cascade.

### Threading tiers

| Operation | Tier |
|---|---|
| `HeightmapGrid` const sampling (`get_height_world`, `get_normal_world`, `page_at`, `resident_at`) | **worker-safe** — const reads of an immutable value; this is what keeps `Application::get_terrain_height` callable from `on_simulate` (the effect system's terrain sampler does exactly that) |
| `HeightmapGrid::set_page` / `clear_page` / `create` | owned by the host's builder thread; not called by the engine |
| `TerrainRenderer::set_heightmap_grid` and everything it calls (mesh build, buffer + texture creation, the grass proxy) | **main-thread** — creates and destroys GPU resources |
| `TerrainRenderer::get_height` / `get_normal` | worker-safe const reads, unchanged |
| the three draw entry points | main-thread, inside a render pass |

`TerrainRenderer::get_height` is called once per mesh vertex — millions of
times at init — so the "is anything resident" answer is cached in a bool
rather than re-walking the page vector per sample.

### Lifetimes

* `HeightmapGrid` is copied into the renderer by `set_heightmap_grid`; the
  caller's copy is theirs.
* `TerrainPage` owns its buffers and its texture through `unique_ptr`;
  `pages_` is rebuilt wholesale on every `set_heightmap_grid`, so a page
  pointer never outlives a grid change.
* `heightmap_texture()` returns a NON-owning pointer, either into page 0 or to
  the grass proxy. `SceneRenderer::set_heightmap_grid` re-fetches it after
  every install, which is the same rewiring the pre-paging path did.

### Additivity

With one page — which is what every host in the tree has, including the
playground — the intent is a bit-identical frame:

* same mesh rect, same cell size, same tiles, same rows, same index numbering;
* same early-outs (the per-page `has_buffers()` predicate reproduces the old
  `vertex_buffer_ && index_buffer_` gate; the prepass's adds the index-count
  term the old prepass had);
* page reject can only agree with the tile cull (union AABB);
* `get_height`/`get_normal` are `Heightmap`'s own functions with no arithmetic
  in front of them — pinned bit-for-bit over 10k random probes in
  `src/bracken/heightmap_grid.test.cpp`;
* grass binds the same texture with the same five scalars.

**Not verified by running the engine.** This worktree has no build tree, so
the claim above is argued from the code and pinned by unit tests, NOT by a
`--smoke 30` run. The integrator must confirm the RT-off baseline
(`86.3 / 45.6`) and diff the screenshot — a mean-luma-blind change is exactly
what CLAUDE.md warns about.

Two deliberate behaviour changes on degenerate input, both in the safe
direction: `SceneRenderer::set_heightmap` with a map whose `world_size` is 0
is now REFUSED with a logged error instead of installing a heightmap that
divides by zero at sample time; and a map with `resolution < 2` installs as a
non-resident page (flat placeholder) rather than as a map whose
`resolution - 1` underflows an unsigned. `Application::set_heightmap` already
rejected both before reaching the renderer.

---

## CLAUDE.md delta

**Module table** — replace the `render/` row's terrain mention with, or append
to the row: *"terrain (now PAGED: `render/terrain_renderer.hpp` keeps one GPU
page — vertex/index buffers, low-LOD shadow buffers, R16 texture, tiles and
rows — per `HeightmapGrid` cell, and rejects a page's union AABB in front of
the unchanged `cull_tile_runs` at all three draw sites)"*.

**"Architecture" prose, after the module table** — add:

> Terrain is **paged**. `HeightmapGrid` (`src/bracken/heightmap_grid.hpp`) is
> an N x M grid of `Heightmap` pages that all share ONE quantization range,
> and the single-map world every host has today is a 1x1 grid:
> `set_heightmap(map)` is `set_heightmap_grid(HeightmapGrid::from_single(map))`
> and nothing else. Pages share their boundary sample lines, so seam
> continuity is a property of the authored data plus that shared range — which
> is why `validate()` REFUSES a grid whose pages were packed against different
> ranges instead of preferring matched ones. See docs/gap_streaming.md.

**Load-bearing constants** — add:

> - `HeightmapGrid::k_recommended_min_page_size = 2000.0f`
>   (`src/bracken/heightmap_grid.hpp`) — advisory, not enforced: below it
>   `TerrainRenderer::set_heightmap_grid` warns. Tile runs never merge across
>   pages (each page owns its index buffer), so every frustum-intersecting
>   page costs at least one draw in each of the six terrain walks a frame
>   (main + prepass + 4 cascades). `TerrainRenderer::k_page_margin = 500.0f`
>   is the terrain skirt, now applied only on the grid's OUTER edges;
>   `k_grass_proxy_max_resolution = 1024` caps the paged-world grass proxy.

**Gotchas** — add:

> - **Two pages packed by the scanning `Heightmap::from_floats` overload step
>   at their shared seam.** The scanning overload derives its 16-bit range
>   from the data it is given, so independently authored pages encode their
>   shared samples differently and the seam shows a hard step with nothing
>   logged. `HeightmapGrid::validate()` refuses such a grid outright (no
>   tolerance on the range comparison); build grids with
>   `HeightmapGrid::create` + `set_page`, which packs against the grid's range
>   for you.
> - **Grass follows a smoothed terrain on a PAGED world.** `grass_blade.hlsli`
>   hard-codes one height texture, one origin and one square world size, so a
>   paged grid feeds it a coarse world-spanning proxy (<= 1024^2 R16) instead
>   of per-page textures. A 1x1 grid is unaffected — it binds the page's own
>   texture. Per-page grass sampling moves `GrassVertexUniforms`, which the
>   uniform-layout test pins on both sides.

---

## What is not here (stage 1)

Deliberately out of scope for the terrain landing:

* **Terrain-page residency** — nothing loads or unloads PAGES by distance; a
  host builds the whole grid up front (or calls `set_page`/`clear_page` and
  re-installs). Part 2 below is asset (model) residency, not page residency.
* **Per-page grass sampling** — see the limitation above.
* **A paged playground** — `examples/playground/playground.cpp` is untouched
  and still builds one 257x257 map, which is what keeps the pinned smoke
  baselines meaningful for this change. Exercising paging end-to-end on the
  GPU (a 2x2 playground grid) is the obvious next verification step and has
  NOT been done.
* **Splatmap paging** — `splat_params` still spans the renderer's world rect
  from the origin, unchanged.

---

# Part 2 — Asset residency (stage 2, piece b)

`AsyncModelLoader` already did the hard half of streaming — decode on the job
pool, budgeted main-thread upload in `pump()` — but nothing above it ever gave
anything back. `TextureCache` had only `clear()`, `ModelManager` had only
`unload_all()`, and a draw whose model was missing was counted as
`draws_dropped_unknown_model`, which is a *bug* diagnosis for what is really a
*budget* condition.

Stage 2 adds the missing half: a byte budget, a distance band with hysteresis,
per-asset residency state, per-model unload/reupload, texture reference counts,
and a distinct diagnostic for a draw whose asset is simply not here yet.

## User guide

### Nothing changes unless you construct one

`ResidencyManager` is host-owned and entirely opt-in. Without one,
`ModelManager` and `TextureCache` behave exactly as before: nothing is ever
unloaded, no texture is ever destroyed, and `draws_deferred_not_resident` is
always zero.

### Streaming a world by distance

```cpp
#include "bracken/residency_manager.hpp"

// Owned by the game, alongside its ModelManager and AsyncModelLoader.
bracken::AsyncModelLoader loader(models(), /*pack=*/nullptr);

bracken::ResidencyManager::Config cfg;
cfg.budget_bytes = 384ull << 20;
cfg.load_radius = 3000.0f;   // inside  -> request
cfg.evict_radius = 5000.0f;  // outside -> eligible to evict
cfg.min_frames_resident = 120;
bracken::ResidencyManager residency(models(), loader, cfg);

// Pumped at the frame fence, and the invalidator is wired for you.
set_residency(&residency);

// One entry per streamable thing, with where in the world it is needed.
std::vector<bracken::ResidencyManager::StreamId> ids;
for (const Prop& prop : world_props) {
    bracken::ResidencyManager::Item item;
    item.name = prop.name;              // ModelManager registration name
    item.source = prop.glb_path;        // pack entry or filesystem path
    item.anchor = prop.position;
    item.cull_radius = prop.radius;     // added to BOTH radii
    item.proxy = low_detail_handle_;    // drawn while not resident
    ids.push_back(residency.add(std::move(item)));
}

// In on_simulate: set_viewer is worker-safe, and handle() resolves a draw.
void on_simulate(bracken::SimFrame& sim) override {
    residency_->set_viewer(player_position_);
    auto& frame = sim.frame();
    for (std::size_t i = 0; i < world_props.size(); ++i) {
        const bracken::ModelHandle h = residency_->handle(ids[i]);
        if (h != bracken::INVALID_MODEL_HANDLE) {
            frame.draw_model(h, world_props[i].transform);
        }
    }
}
```

`handle()` returns the real handle when resident and the item's `proxy`
otherwise, so the draw loop needs no branch on residency state. With no proxy it
returns `INVALID_MODEL_HANDLE` and the draw is simply not emitted.

### Reading the numbers

`ResidencyManager::stats()` reports `resident_bytes` against `budget_bytes`, the
per-state counts, and two numbers worth watching:

* `budget_overruns` — pumps that ended over budget with nothing evictable. Every
  resident item was either inside the band or younger than
  `min_frames_resident`. The budget is not achievable from where the viewer is
  standing; raise it or shrink `load_radius`.
* `evictions_blocked_by_hysteresis` — candidates the frame count refused. A
  large steady value means `min_frames_resident` is fighting the budget.

On the render side, `Application::frame_diagnostics().draws_deferred_not_resident`
counts draws whose model is registered but not resident this frame. It is
separate from `draws_dropped_unknown_model` on purpose — see below.

### The two diagnostics are NOT the same thing

`draws_dropped_unknown_model` means the handle names no model at all: a host
bug, permanent, and a reason to look at the code. `draws_deferred_not_resident`
means the model is registered and its slot is intact but its payload is not
here: a budget condition, transient, and a reason to look at the config. A
streaming frame conflates them at the cost of a wall of false bug reports, which
is exactly why the counter was added rather than reusing the existing one.

## Spec

### `ResidencyManager` (`src/bracken/residency_manager.hpp`)

Root layer, namespace `bracken`, beside `async_model_loader.hpp` — it knows
`ModelManager`, so `core/` is illegal for it (the module-layering test caught
that arrangement once already).

| Type | Meaning |
|---|---|
| `StreamId` | opaque, stable per-item id; 0 is invalid |
| `State` | `Absent`, `Loading`, `Resident`, `Evicted`, `Failed` |
| `Config` | budget, the two radii, `min_frames_resident`, three per-pump caps |
| `Item` | name, source, `anchor`, `cull_radius`, `proxy` |
| `Stats` | bytes, per-state counts, cumulative totals, `budget_overruns` |

**The pure decision core**, static and testable with no ModelManager, no job
system and no GPU:

| Function | Contract |
|---|---|
| `wants_load(distance, cull_radius, cfg)` | inside `load_radius + cull_radius`, inclusive |
| `outside_evict_band(distance, cull_radius, cfg)` | strictly beyond `evict_radius + cull_radius` |
| `in_hysteresis_band(...)` | neither — the dead zone where nothing ever happens |
| `config_is_valid(cfg)` | `load_radius > 0 && evict_radius > load_radius` |
| `plan_evictions(candidates, resident_bytes, cfg, out, blocked)` | farthest first, ties by ascending index, stops at the budget or `max_evicts_per_pump`, skips candidates younger than `min_frames_resident`; a zero budget plans nothing |

**`pump()`**, in order: adopt what the loader finished → age the resident set →
request what entered the load band (nearest first, ties by index) → evict while
over budget → republish the stats. Main-thread, at the frame fence.

**Defaults**: `budget_bytes = 512 MB`, `load_radius = 4000`,
`evict_radius = 6000`, `min_frames_resident = 120`, `max_uploads_per_pump = 2`,
`max_requests_per_pump = 4`, `max_evicts_per_pump = 4`. Every one of these is a
plausible starting value, **not a measured one** — see "Tuned constants with no
measurement" below.

### `ModelManager` additions (`src/bracken/model_loader.hpp`)

| API | Contract |
|---|---|
| `unload_model(handle)` | frees buffers, releases texture references, drops retained CPU geometry, clears `Model::loaded`. KEEPS the slot, the name mapping, the source path, the bounds and the LOD links. Returns the byte estimate released. Does NOT follow the LOD chain |
| `reupload_model(handle, decoded)` | move-assigns fresh contents into the same slot, preserving the handle, the `Model*` and the slot's LOD chain; re-uploads and recomputes the bounding sphere |
| `gpu_bytes(handle)` | buffers exactly, plus the FULL estimated size of every texture slot — deliberately over-counting shared atlases, because an upper bound is the safe direction for a budget |

`ModelLoader::free_gpu_resources(TextureCache&, Model&)` is the counted form of
the existing free: one `release()` per non-null texture slot, mirroring
`upload_to_gpu`'s one `acquire()` per slot. `reload_model` now uses it too, so a
hot-reload no longer pins its outgoing images forever.

### `TextureCache` reference counts (`src/bracken/texture_cache.hpp`)

`acquire()` keeps its exact signature and now also increments. `release()`
decrements. `bytes_resident()`, `refcount()`, `texture_bytes()` and `stats()`
report.

**Destruction at zero is OFF by default** (`set_destroy_on_release`), and that
is the single most important sentence in this section. `Mesh` and `PBRMaterial`
hold four non-owning `gpu::GPUTexture*` each and there is no central registry of
who borrowed what; one missed pairing with destruction on leaves a live
descriptor pointing at a freed texture, which is a GPU fault or garbage pixels
three subsystems away, not an assert. With destruction off the counts are exact
and observable, so a project can VERIFY its own pairing on its own content
before flipping the switch. Leaving it off reproduces today's never-evicted
behaviour byte for byte — the only cost is that eviction returns geometry VRAM
and not image VRAM.

`texture_bytes_estimate(w, h, format, mipmapped)` is the size model: the EXACT
sum over mip levels, not the 4/3 limit (a 1024x1 strip's chain is nowhere near
4/3 of its base). It is an estimate and says so — SDL3 exposes no allocation
size — but it is stable, content-derived and monotone, which is all a budget
needs.

### `AsyncModelLoader::Request::reupload_into`

New field, defaulting to `INVALID_MODEL_HANDLE` (the original behaviour). A
valid handle routes the finished decode to `ModelManager::reupload_model`
instead of `register_model`. It exists because `register_model` REFUSES a name
it already knows, so streaming a model back in after an eviction has no route
through the registering path — and inventing one would mean either orphaning the
old handle or renumbering, both of which break handle stability.

## Architecture

### Where it sits in the frame

`ResidencyManager::pump()` runs at `Application::frame_fence`, the pipeline's
quiescent point, immediately after the hot-reload poll and for a stronger
version of the same reason: hot-reload replaces a resource's *contents*, and
`pump()` can FREE one. A packet being built or rendered against a freed vertex
buffer is not a stale image, it is a crash.

Nothing else in the frame changes. The Prepare stage gains one branch —
`model->loaded` — on a path that already had the model pointer in hand.

### Eviction versus the ray-tracing acceleration structure

This is the sharp edge of the whole feature. `RtScene` rebuilds its TLAS when
the instance SET changes (a hash of the count and the handle sequence), and an
eviction changes NEITHER, so nothing fires on its own and the BLAS keeps
pointing at the CPU geometry `unload_model` just freed. It is a use-after-free,
not a visual glitch.

Three things close it:

1. `ResidencyManager` calls a geometry invalidator on every eviction and every
   completed load, and warns once (loudly) if none is installed.
2. `Application::set_residency` wires that invalidator to
   `SceneRenderer::invalidate_raytracing_scene()` automatically when the
   renderer exists, so a host cannot forget.
3. `RtScene::build_blas_for_models` skips a slot whose `Model::loaded` is false,
   which is also why `unload_model` must leave `model_count()` unchanged — that
   walk is indexed from 0 to `model_count()`.

### Threading tiers

| Operation | Tier |
|---|---|
| `ResidencyManager::add` / `set_config` / `set_geometry_invalidator` / `pump` | **main-thread** (asserted); `pump` additionally requires the frame fence |
| `ResidencyManager::set_viewer` | **worker-safe** — three relaxed atomic stores; callable from `on_simulate` |
| `handle` / `resident` / `state` / `stats` | worker-safe reads of a table only `pump()` writes; safe from `on_simulate` because the fence and the sim never overlap |
| `ModelManager::unload_model` / `reupload_model` | **main-thread** (asserted) — destroys and creates GPU resources |
| `TextureCache::release` | **main-thread** (asserted) |

### Lifetimes

* The `ResidencyManager` is host-owned and must outlive the `set_residency`
  detach.
* `AsyncModelLoader` must outlive the manager (it is held by reference).
* An evicted model's `Model*` stays valid — that is the point of keeping the
  slot. Pointers INTO it (meshes, vertices) do not.

## Tests

| File | What it asserts |
|---|---|
| `src/bracken/residency_manager.test.cpp` | the band's three regions and their boundaries; **oscillating across the band 100 times produces zero events**; an inverted band is refused by both the constructor and `set_config`; `plan_evictions` is farthest-first, tie-broken by index (order-independent), respects the budget, the per-pump cap and `min_frames_resident`, and plans nothing on a zero budget; adoption of an already-loaded model; duplicate/empty names refused; the per-state counts sum to `item_count()` |
| `src/bracken/model_loader.test.cpp` | `unload_model` keeps the slot, the name, the handle, the neighbour's pointer and `model_count()`; it drops retained CPU geometry; unloading twice or an unknown handle is a no-op; `reupload_model` restores the SAME slot and pointer, recomputes the bounding sphere, and PRESERVES the slot's LOD chain; refusals leave the slot loaded |
| `src/bracken/texture_cache.test.cpp` | `texture_bytes_estimate` is exact unmipped, the exact mip sum (not 4/3) mipped, monotone in every dimension, zero on degenerate input; destruction-on-release is off by default; releasing null is a silent no-op; releasing a foreign pointer is counted and survived; `clear()` resets every counter |
| `src/bracken/scene/scene_renderer_prep.test.cpp` | an evicted model is counted as `draws_deferred_not_resident`, NOT as `draws_dropped_unknown_model`; it is binned into neither the main view nor any cascade; the count is identical at 0/1/4 workers |
| `src/bracken/contracts/render_stats.test.cpp` | the new counters fold through `operator+=` and `merge_into` like every other one |

---

# Part 3 — Hi-Z occlusion culling for draws (stage 2, piece c)

**Default OFF** (`GraphicsSettings::occlusion_culling`), and the honest reason
is in "One-sidedness, and exactly where it ends" below.

## User guide

```cpp
auto settings = graphics_settings_edit();   // scoped, committed at the fence
settings->occlusion_culling = true;
```

or tick "Occlusion culling (experimental)" in the dev overlay's Graphics panel.

`Application::render_stats()->entities_occlusion_culled` is the readout. Zero in
a dense scene means the depth never arrived (the first frame after enabling,
after a resize, or a frame whose prepass did not run) — not that there was
nothing to cull.

**How to evaluate it, and the trap.** Turn it on and off and diff the
SCREENSHOTS pixel-wise. A correct occlusion cull removes only invisible draws,
so the two images must be identical; any pixel difference is a false cull.
`mean_luma` is structurally blind to this — CLAUDE.md records twice that a
change of a few hundred pixels moves no luma digit — and a missing prop behind a
wall is exactly a few hundred pixels. For the timing half, use
`--rt-bench 3000` A/B back to back in a RelWithDebInfo tree; a `--smoke 30`
line's fps figure measures GPU clock state, not the renderer.

## Spec

### `OcclusionField` / `OcclusionBounds` (`src/bracken/contracts/occlusion.hpp`)

POD plus glm, in the leaf `contracts/` layer, so `ScenePrep` can consume it
without gaining a device or a renderer pointer.

| Field | Meaning |
|---|---|
| `depth`, `mip_w`, `mip_h` | the downloaded level of the FARTHEST-surface pyramid |
| `screen_w`, `screen_h`, `level` | mip 0's dimensions; footprints are addressed at `screen / 2^level` |
| `view_proj` | the unjittered matrix the depth was rasterized with |
| `camera_travel`, `max_camera_travel` | translation since then; past the max the field is unusable |
| `camera_rotation_cos`, `min_camera_rotation_cos` | rotation since then; below the min the field is unusable |
| `max_tested_screen_fraction` | never test anything larger — the occluder feedback loop |
| `depth_slack` | extra device-depth margin; 0 is already conservative |

`project_aabb(view_proj, min, max)` returns `OcclusionBounds`: the axis-aligned
hull of the eight projected corners, plus `nearest_depth` (the MAX device depth
over them), and `valid == false` whenever any corner is at or behind the near
plane. `occluded_by_field(field, bounds)` is the footprint walk;
`occlusion_cull_aabb(field, min, max)` is inflate-project-test in one call, so
the inflation cannot be forgotten at a call site.

### The second pyramid (`src/bracken/render/hzb.hpp`)

`hzb_reduce` keeps the CLOSEST surface (MAX under reversed-Z) because the
screen-probe march needs that. Culling needs the FARTHEST — a box is hidden only
when its nearest point is behind the farthest occluder ANYWHERE in its
footprint, or a box peeking between two occluders that share a texel gets culled.
`hzb_reduce_far` is that reduction, `HzbReduction` selects it at `init()`, and
`shaders/hzb_reduce_far.comp.hlsl` is the GPU half (a copy of
`hzb_reduce.comp.hlsl` with one operator flipped, plus its own
`ComputePipelineType::HzbReduceFar` row).

They are opposite reductions of the same buffer, so neither can be recovered
from the other after the fact: this is a second texture, not a flag at read
time. `HzbPyramid::request_level_readback(level, dst)` downloads one coarse mip
on its own command buffer and returns a fence, following the idiom
`render/auto_exposure.cpp` established.

### Constants

| Constant | Value | Status |
|---|---|---|
| `SceneRenderer::k_occlusion_level` | `4` | 100x57 floats at 1600x900 (23 KB). Chosen so the download and the footprint walk are both trivial while a prop still covers a texel or two. **Not measured.** |
| `OcclusionField::max_tested_screen_fraction` | `0.15f` | The occluder-feedback guard. **A guess.** |
| `OcclusionField::max_camera_travel` | `250.0f` world units | **A guess.** |
| `OcclusionField::min_camera_rotation_cos` | `0.99905f` (cos 2.5 deg) | **A guess**, sized as roughly one frame of a fast flick at 60 Hz. |
| `OcclusionField::depth_slack` | `0.0f` | Zero is already conservative; the dial exists for depth precision at range. |

## Architecture

### Where it sits in the frame

```
frame i:   ... depth prepass -> [closest HZB] -> [FARTHEST HZB + readback submit] -> GTAO -> ...
frame i+1: resolve readback (deterministic fence wait) -> plan_frame -> Prepare (the cull) -> ...
```

The pyramid is built where its only input first exists. What it produces is
consumed a WHOLE FRAME LATER, because Prepare runs before this frame's prepass
and SDL3 has no non-blocking map. That one-frame lag is the source of every
conservatism knob above; there is no arrangement of this feature in which a CPU
cull reads current depth.

### The determinism rule, and which side of it this lands on

**The readback fence is waited on DETERMINISTICALLY**, at the top of
`render_frame`, before the plan is made. The alternative — "skip the test if the
fence has not signalled" — makes the culled set depend on GPU timing, so two
runs of the same scene render different frames and every pinned screenshot
baseline stops meaning anything. The fence is a full frame old and normally
already signalled; if it ever is not, the cost shows up as render time.

`RenderPlan::occlusion_active` is therefore a pure function of the setting, the
HDR path and `RendererCaps::occlusion_depth_ready`, and the last of those is
false only for reasons that are facts about the renderer (first frame after
enabling, after a resize, after a frame with no prepass) — never about timing.

### Determinism inside the Prepare stage

`PrepParams::occlusion` is a frame-constant, read-only input, exactly like
`main_frustum`, so the per-command predicate stays a pure function of (command,
params) and the chunked walk still reproduces the serial one byte for byte. The
design that would break this is the obvious one — a per-instance "was visible
last frame" latch — because chunk order would decide who writes it. There is
deliberately no such latch, and
`src/bracken/scene/scene_prep_parity.test.cpp` re-runs the whole 0/1/4-worker
memcmp with a populated field.

The test sits between the frustum reject and `entities_rendered++`, reusing the
one `get_model` lookup already there, and it is applied to the main view ONLY:
`bin_shadow_chunk` never sees it, because a camera-occluded object still casts
into the view.

### One-sidedness, and exactly where it ends

**For a static scene under a static camera**, with a correct farthest-surface
pyramid, a false cull is impossible, and `src/bracken/contracts/occlusion.test.cpp`
brute-forces every link in the chain:

1. any corner at or behind the near plane => `valid == false` => never culled;
2. the projected rect CONTAINS the projection of interior points (checked over
   random interior samples), because with all `w > 0` the projection of a convex
   hull lies inside the hull of the projected vertices;
3. `nearest_depth` (the corner maximum) bounds every interior point's device
   depth, because device depth is monotone in view depth and view depth attains
   its extremes at vertices — this is what makes testing one scalar sound;
4. the footprint is DILATED by one texel on every side and addressed at
   `screen / 2^level`, so it is a strict superset of the true footprint under
   any rounding convention, and a superset can only refuse to cull;
5. the field texel is the MIN over its whole mip-0 region (pinned by
   `render/hzb.test.cpp`), so "behind this texel" really means "behind
   everything in it";
6. the comparison is STRICT, so coplanar is not behind;
7. `camera_travel` inflation, `depth_slack` and a wider footprint are each
   monotone in the safe direction — asserted by sweeping them, not assumed.

**Under camera motion it is a heuristic, not a proof.** The depth is a frame
old, so a box the previous view hid may be visible now, and no amount of
inflating the box fixes that: the box did not move, the camera did.
`camera_travel` inflation, `max_camera_travel` and `min_camera_rotation_cos`
bound the error; they do not eliminate it. A moving OCCLUDER has the same
problem in the other direction.

That is why the feature ships off, why the three motion constants are labelled
as guesses above, and why the acceptance test is a pixel-wise screenshot diff
rather than a luma number.

### The occluder feedback loop

An object culled this frame writes no prepass depth, so it cannot occlude next
frame. For a pebble that is free; for a cathedral it means everything behind it
pops back in and the scene oscillates. `max_tested_screen_fraction` is the only
guard, and 0.15 is a plausible number with no measured knee behind it — exactly
the kind of unitless guess CLAUDE.md's R5a note criticises. Measuring it is the
first thing to do before this feature is ever turned on by default.

## Tests

| File | What it asserts |
|---|---|
| `src/bracken/contracts/occlusion.test.cpp` | every conservatism property listed above, including the near-plane straddle, the random-interior-point containment sweep, the one-nearer-texel case, the screen-fraction gate (and that raising it culls the same box, proving the refusal came from the gate), the OVERSIZED last texel of an odd mip axis, the unusable-field gates, and the monotonicity sweeps for `camera_travel` and `depth_slack` |
| `src/bracken/render/hzb.test.cpp` | `hzb_reduce_far` keeps the farthest surface; a far mip texel is the MIN of its immediate footprint AND of its whole mip-0 region; and **`far <= near` texel for texel** — the cheap guard against wiring the culler to the wrong pyramid |
| `src/bracken/scene/render_plan.test.cpp` | `occlusion_active` is false by default, requires all three inputs, is false on the first frame after enabling, and changes NO other field of the plan |
| `src/bracken/scene/scene_renderer_prep.test.cpp` | a constant wall culls every bounded draw and nothing else; an empty depth buffer culls nothing; a command with no resolvable model is never tested; the shadow cascade bins are byte-identical with and without the field; the result is bit-identical at 0/1/4 workers |
| `src/bracken/scene/scene_prep_parity.test.cpp` | the full 0/1/4-worker memcmp with a populated field at four depths, plus an explicit-null run reproducing the baseline exactly |

---

# Honest limitations across parts 2 and 3

Read this section before trusting anything above.

* **Nothing here was run.** This worktree has no build tree, so every claim is
  argued from the code and pinned by unit tests, not by a `--smoke 30` run. The
  integrator must confirm the RT-off pin (`86.3 / 45.6`) is unmoved and, with
  occlusion ON, diff the screenshot pixel-wise against the OFF run.
* **Texture destruction is deliberately disabled.** `set_destroy_on_release`
  defaults to false, so an eviction returns geometry VRAM and NOT image VRAM.
  The refcount is exact and observable; enabling destruction is a measured step
  for a project that has verified its own pairing.
* **`unload_model` does not follow the LOD chain.** Levels are separate models
  with their own handles; evicting a base leaves its levels resident. A host
  that wants the chain gone must evict each handle.
* **Skinned draws are not residency-aware.** `resolve_skinned_chunk` does not
  test `Model::loaded`; a skinned draw against an evicted model resolves through
  its (still-present) skeleton and draws nothing, counted nowhere. Static model
  draws are handled; skinned ones were left alone to limit the blast radius.
* **The reload round-trip is not covered by a test.** It goes through
  `AsyncModelLoader`, which needs a real `.glb`. The two halves are covered
  separately (`reupload_model` on the ModelManager side, the routing branch in
  `async_model_loader.cpp`) but not end to end.
* **Every `ResidencyManager::Config` default is a plausible value, not a
  measured one**, and so are all four occlusion constants in the table above.
* **The occlusion GPU path has never executed.** The far pyramid's build, the
  texture readback (`SDL_DownloadFromGPUTexture` pitch and region setup), the
  transfer-buffer sizing and the fence lifecycle are all written but unrun. The
  pure math either side of them is heavily tested; the plumbing between is not.
* **No playground integration.** `examples/playground/playground.cpp` is
  untouched: there is no `PLAYGROUND_OCCLUSION` pin and no residency-driven
  scene, which is what keeps the existing smoke baselines meaningful for this
  change and also what leaves both features unexercised end to end.

# CLAUDE.md delta (parts 2 and 3)

**Module table** — append to the root-layer description, or add beside
`async_model_loader.hpp`:

> `residency_manager.hpp` — `ResidencyManager`: distance-driven asset residency
> over `AsyncModelLoader` and `ModelManager`. A byte budget, a load/evict band
> with dead space between the radii, and per-item state; `pump()` runs at the
> frame fence, the same quiescent point hot-reload uses and for a stronger
> reason (it can FREE a model, not just replace its contents). Host-owned and
> entirely opt-in — without one, nothing is ever unloaded.

**Module table, `render/`** — append to the HZB mention:

> …hierarchical-Z pyramid (`render/hzb.hpp`) — now in TWO reductions:
> `HzbReduction::Closest` (MAX under reversed-Z) for the screen-probe march and
> `HzbReduction::Farthest` (MIN) for draw occlusion culling. Opposite reductions
> of the same buffer, so they are two textures, not one with a read-time flag.

**Load-bearing constants** — add:

> - `SceneRenderer::k_occlusion_level = 4` (`src/bracken/scene/scene_renderer.hpp`)
>   — the Hi-Z mip the CPU occlusion cull downloads; 100x57 floats at 1600x900.
>   `OcclusionField`'s four conservatism knobs
>   (`max_tested_screen_fraction = 0.15`, `max_camera_travel = 250`,
>   `min_camera_rotation_cos = 0.99905`, `depth_slack = 0`) live in
>   `src/bracken/contracts/occlusion.hpp` and are all UNMEASURED guesses.
> - `ResidencyManager::Config` defaults (`src/bracken/residency_manager.hpp`) —
>   512 MB budget, load 4000 / evict 6000, 120 frames minimum residency. The
>   two radii must satisfy `evict > load`; `set_config` refuses otherwise rather
>   than clamping, because a silently-clamped band thrashes anyway.

**Diagnostics bullet** — add `draws_deferred_not_resident` to the list of
`RenderStats::diagnostics` counters, and `entities_occlusion_culled` to
`RenderStats`.

**Gotchas** — add:

> - **Evicting a model does NOT invalidate the ray-tracing acceleration
>   structure on its own.** `RtScene` rebuilds when the instance SET changes —
>   a hash of the count and the handle sequence — and an eviction changes
>   neither, so the BLAS keeps pointing at the CPU geometry
>   `ModelManager::unload_model` just freed. Call
>   `SceneRenderer::invalidate_raytracing_scene()`;
>   `Application::set_residency` wires it for you.
> - **`TextureCache` counts references but does not destroy at zero.**
>   `set_destroy_on_release` is off by default. `Mesh`/`PBRMaterial` hold four
>   non-owning `gpu::GPUTexture*` each with no registry of who borrowed what, so
>   one missed pairing is a live descriptor pointing at a freed texture — a GPU
>   fault, not an assert. Verify with `TextureCache::stats()` before enabling it.
> - **Occlusion culling is conservative for a STILL camera and a heuristic for a
>   moving one.** The depth it tests is a frame old by construction (Prepare runs
>   before the prepass; SDL3 has no non-blocking map), so a box the previous view
>   hid may be visible now. Default off. Judge it by a PIXEL-WISE screenshot diff
>   against the off run, never by `mean_luma` — a missing prop behind a wall is a
>   few hundred pixels.
