#pragma once
/**
 * A rectangular grid of Heightmap PAGES — the paged half of the streaming
 * gap (docs/gap_streaming.md).
 *
 * The world the engine has always rendered is exactly a 1x1 grid, which is
 * what makes this additive: `Heightmap` is unchanged and already carries its
 * own origin (it was a tile type in shape all along, the engine simply never
 * made a second one). `HeightmapGrid::from_single` wraps today's map and
 * every sampling path below reproduces `Heightmap`'s answer BIT for BIT in
 * that configuration — see heightmap_grid.test.cpp, which pins it.
 *
 * THE SEAM TRAP, and why this type refuses rather than prefers. The scanning
 * overload of `Heightmap::from_floats` derives its 16-bit quantization range
 * from the data it is handed (heightmap.cpp's `from_floats`), so two pages
 * authored independently quantize against DIFFERENT ranges and the height
 * they store for their shared edge samples disagrees — a visible step along
 * the whole seam, with no error anywhere. heightmap.hpp's from_floats comment
 * has warned about this since before there was a grid. `validate()` therefore
 * makes a range disagreement a hard refusal: an inconsistent grid is not
 * valid(), and the renderer will not install one. Use `create()` +
 * `set_page()` and the question cannot come up at all — set_page packs
 * against the grid's own `range`.
 */

#include "bracken/core/result.hpp"
#include "bracken/heightmap.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <vector>

namespace bracken {

/**
 * Pages are SQUARE, all the same size and resolution, laid out row-major, and
 * they SHARE their boundary samples: page (px,pz) covers the closed world
 * square [origin + px*page_size, origin + (px+1)*page_size] on both axes, so
 * its last sample column is at the same world position as its right
 * neighbour's first. Height continuity across a seam is therefore a property
 * of the authored data (equal heights on the shared line) plus the shared
 * quantization range this type enforces — not of any blending done at sample
 * time.
 *
 * A page may be absent (`Heightmap::valid() == false`); that is the state a
 * streamed world spends most of its life in, and it is what `resident_at()`
 * answers and what the `fallback` argument of `get_height_world` covers.
 */
struct HeightmapGrid {
    uint32_t pages_x = 0;
    uint32_t pages_z = 0;
    /// World units per page side. Pages are square; a non-square WORLD is a
    /// non-square page COUNT.
    float page_size = 0.0f;
    /// World position of page (0,0)'s -x/-z corner.
    float origin_x = 0.0f;
    float origin_z = 0.0f;
    /// The quantization range EVERY resident page must be packed against.
    HeightRange range{};
    /// Row-major, `pages_z * pages_x` entries; invalid entries are pages that
    /// are not resident.
    std::vector<Heightmap> pages;

    /**
     * Below this page size the per-page draw-call arithmetic stops paying:
     * tile runs cannot merge across pages (each page owns its own index
     * buffer), so every frustum-intersecting page costs at least one draw in
     * each of the six terrain walks (main, prepass, 4 shadow cascades). At
     * 2000 units a page is 80x80 cells of the 25-unit terrain grid — big
     * enough that the tile-run merging inside it still does the work. This is
     * ADVISORY: a smaller page is correct, just more expensive, so
     * TerrainRenderer warns once instead of refusing.
     */
    static constexpr float k_recommended_min_page_size = 2000.0f;

    /**
     * One page grid over the single map the engine has today. Zero copies is
     * not the point; call-site compatibility is —
     * `SceneRenderer::set_heightmap` is exactly this call plus the grid
     * overload.
     */
    [[nodiscard]] static HeightmapGrid from_single(const Heightmap& map);

    /**
     * An empty grid of the given shape with every page absent. Fill it with
     * `set_page`, which packs against `range` and therefore cannot produce
     * the seam step described at the top of this header.
     */
    [[nodiscard]] static HeightmapGrid create(uint32_t pages_x, uint32_t pages_z, float page_size, float origin_x,
                                              float origin_z, HeightRange range);

    /**
     * Pack `heights` (row-major `z * resolution + x`, world units) into page
     * (px,pz) against the grid's shared `range` and the page's derived world
     * bounds. Returns false and logs on an out-of-range page index or a
     * sample count that is not `resolution * resolution`.
     *
     * The FIRST and LAST sample rows/columns sit ON the page boundary, so a
     * neighbouring page must repeat them — see the type comment.
     */
    [[nodiscard]] bool set_page(uint32_t px, uint32_t pz, std::span<const float> heights, uint32_t resolution);

    /// Drop a page's data (it becomes non-resident). No-op out of range.
    void clear_page(uint32_t px, uint32_t pz);

    /**
     * Why this grid cannot be used, or ok. The error string names the
     * offending page and the disagreement, because "the terrain is flat" is
     * not a debuggable symptom.
     *
     * Refused: a degenerate shape; a `pages` vector that is not
     * `pages_x * pages_z` long; a resident page whose quantization range is
     * not EXACTLY the grid's (the seam trap — no tolerance, deliberately); a
     * resident page whose resolution differs from its siblings'; a resident
     * page whose world size or origin is not the one the grid's addressing
     * will assume (tolerance below).
     *
     * A grid with NO resident pages is ok: that is a streamed world before
     * anything has arrived.
     */
    [[nodiscard]] core::Result<void, std::string> validate() const;

    /// `validate().is_ok()`, for call sites that only need the verdict.
    [[nodiscard]] bool valid() const { return validate().is_ok(); }

    /**
     * Slack allowed on a page's world_size/origin against the value the
     * grid's addressing derives. Ranges get NO slack; geometry gets this
     * much, because a host that computes `origin + px * page_size` by a
     * different but equivalent route can land a float ulp away, while a real
     * authoring mistake is half a page or a whole page off.
     */
    [[nodiscard]] static float geometry_tolerance(float reference);

    [[nodiscard]] float world_size_x() const { return page_size * static_cast<float>(pages_x); }
    [[nodiscard]] float world_size_z() const { return page_size * static_cast<float>(pages_z); }
    [[nodiscard]] std::size_t page_count() const { return pages.size(); }
    /// Pages with data. 0 for a grid nothing has streamed into yet.
    [[nodiscard]] std::size_t resident_pages() const;
    /// The shared per-page sample resolution, or 0 when no page is resident.
    [[nodiscard]] uint32_t page_resolution() const;

    /// Page containing a world XZ, or {-1,-1} when it lies outside the grid.
    [[nodiscard]] glm::ivec2 page_at(float world_x, float world_z) const;

    /// The resident page at (px,pz), or null when out of range or absent.
    [[nodiscard]] const Heightmap* page(int px, int pz) const;

    /// World AABB corners of page (px,pz) on the XZ plane; zeros out of range.
    [[nodiscard]] glm::vec2 page_min_xz(int px, int pz) const;
    [[nodiscard]] glm::vec2 page_max_xz(int px, int pz) const;

    /**
     * Same contract as `Heightmap::get_height_world`, across pages: the page
     * index is CLAMPED to the grid, and the containing page then clamps the
     * sample to its own square, so a query outside the world returns the grid
     * edge height exactly as a single map returns its edge height. Returns
     * `fallback` when the containing page is not resident — the one answer a
     * lone `Heightmap` cannot give.
     *
     * worker-safe: const reads only.
     *
     * For a 1x1 grid this is `Heightmap::get_height_world` with no arithmetic
     * added in front of it, which is what makes the adapter bit-identical.
     */
    [[nodiscard]] float get_height_world(float world_x, float world_z, float fallback = 0.0f) const;

    /**
     * Unit surface normal by central differences over one grid cell, the same
     * stencil `Heightmap::get_normal_world` uses (eps = page_size /
     * (resolution - 1)) but sampled THROUGH `get_height_world`, so a stencil
     * straddling a seam reads both pages instead of clamping against a page
     * edge that is not a world edge. (0,1,0) where the surface is degenerate
     * or nothing is resident. worker-safe.
     */
    [[nodiscard]] glm::vec3 get_normal_world(float world_x, float world_z) const;

    /// Is the page containing this XZ loaded? Outside the grid answers for
    /// the clamped edge page, matching get_height_world's clamping.
    [[nodiscard]] bool resident_at(float world_x, float world_z) const;
};

} // namespace bracken
