# Paged terrain (streaming gap, stage 1)

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

## What is not here

Deliberately out of scope for this landing, in the order the brief lists them:

* **Residency and eviction** — nothing loads or unloads pages by distance; a
  host builds the whole grid up front (or calls `set_page`/`clear_page` and
  re-installs). `TextureCache` still has no refcount and `ModelManager` still
  has no `unload_model`.
* **Occlusion culling** — the HZB is still the CLOSEST-surface reduction the
  screen-probe march needs; there is no second farthest-surface pyramid and no
  `OcclusionField`.
* **Per-page grass sampling** — see the limitation above.
* **A paged playground** — `examples/playground/playground.cpp` is untouched
  and still builds one 257x257 map, which is what keeps the pinned smoke
  baselines meaningful for this change. Exercising paging end-to-end on the
  GPU (a 2x2 playground grid) is the obvious next verification step and has
  NOT been done.
* **Splatmap paging** — `splat_params` still spans the renderer's world rect
  from the origin, unchanged.
