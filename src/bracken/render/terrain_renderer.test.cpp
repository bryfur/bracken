// Unit tests for the CPU side of TerrainRenderer's tile culling: row
// construction (build_tile_rows) and frustum-culled run merging
// (cull_tile_runs), including the deterministic chunked fan-out parity
// contract of multithreading_plan §4.3/§6.2 — bit-identical outputs at any
// worker count. Everything here is GPU-free (static pure functions on
// synthetic tile grids).

#include "bracken/render/terrain_renderer.hpp"

#include "bracken/contracts/frustum.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

using bracken::core::jobs::JobSystem;
using bracken::render::TerrainRenderer;
using bracken::scene::Frustum;

using TerrainTile = TerrainRenderer::TerrainTile;
using TileRow = TerrainRenderer::TileRow;
using TileRun = TerrainRenderer::TileRun;

namespace {

constexpr float TILE_SIZE = 100.0f;

// Synthetic row-major tile grid mirroring generate_terrain_mesh's output:
// contiguous first_index ranges, per-row shared aabb_min.z bands, varied
// index counts and heights so merge bugs can't hide behind uniformity.
std::vector<TerrainTile> make_tile_grid(int rows, int cols) {
    std::vector<TerrainTile> tiles;
    tiles.reserve(static_cast<size_t>(rows) * cols);
    uint32_t next_index = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            TerrainTile tile;
            tile.first_index = next_index;
            tile.index_count = 6 * (1 + static_cast<uint32_t>((r + c) % 3));
            next_index += tile.index_count;
            const float max_y = 10.0f + static_cast<float>((r * 31 + c * 17) % 50);
            tile.aabb_min = glm::vec3(static_cast<float>(c) * TILE_SIZE, -10.0f, static_cast<float>(r) * TILE_SIZE);
            tile.aabb_max =
                glm::vec3(static_cast<float>(c + 1) * TILE_SIZE, max_y, static_cast<float>(r + 1) * TILE_SIZE);
            tiles.push_back(tile);
        }
    }
    return tiles;
}

// Engine-style reversed-Z frustum (near/far swapped in glm::perspective)
// placed mid-grid looking down -Z, so it cuts the grid non-trivially.
Frustum make_fixed_frustum(const glm::vec3& cam) {
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 2000.0f, 0.1f);
    glm::mat4 view = glm::lookAt(cam, cam + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    Frustum frustum;
    frustum.extract_from_matrix(proj * view);
    return frustum;
}

// Reference for cull_tile_runs semantics: per-tile frustum test in storage
// order, contiguous visible index ranges merged — the original serial
// algorithm. Row acceleration must never change the result (a row AABB
// contains its tiles, so a rejected row implies every tile is rejected).
std::vector<TileRun> reference_runs(const Frustum& frustum, const std::vector<TerrainTile>& tiles) {
    std::vector<TileRun> runs;
    for (const TerrainTile& tile : tiles) {
        if (!frustum.intersects_aabb(tile.aabb_min, tile.aabb_max)) {
            continue;
        }
        if (!runs.empty() && tile.first_index == runs.back().first_index + runs.back().index_count) {
            runs.back().index_count += tile.index_count;
        } else {
            runs.push_back({tile.first_index, tile.index_count});
        }
    }
    return runs;
}

// One PAGE's tile list carved out of a world-sized grid: the same tiles, the
// same world AABBs, but renumbered into the page's OWN index buffer starting
// at 0 — which is exactly what generate_page_mesh produces and the reason
// runs can never merge across pages.
std::vector<TerrainTile> carve_page(const std::vector<TerrainTile>& world, int cols, int row0, int row1, int col0,
                                    int col1) {
    std::vector<TerrainTile> page;
    uint32_t next_index = 0;
    for (int r = row0; r < row1; ++r) {
        for (int c = col0; c < col1; ++c) {
            TerrainTile tile = world[static_cast<size_t>(r) * cols + c];
            tile.first_index = next_index;
            next_index += tile.index_count;
            page.push_back(tile);
        }
    }
    return page;
}

// The page-level reject box: the union of the page's tile AABBs, exactly as
// TerrainRenderer::generate_page_mesh computes it.
std::pair<glm::vec3, glm::vec3> page_bounds(const std::vector<TerrainTile>& tiles) {
    if (tiles.empty()) {
        return {glm::vec3(0.0f), glm::vec3(0.0f)};
    }
    glm::vec3 lo = tiles.front().aabb_min;
    glm::vec3 hi = tiles.front().aabb_max;
    for (const TerrainTile& tile : tiles) {
        lo = glm::min(lo, tile.aabb_min);
        hi = glm::max(hi, tile.aabb_max);
    }
    return {lo, hi};
}

uint32_t drawn_indices(const std::vector<TileRun>& runs) {
    uint32_t total = 0;
    for (const TileRun& run : runs) {
        total += run.index_count;
    }
    return total;
}

void expect_rows_equal(const std::vector<TileRow>& a, const std::vector<TileRow>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].first_tile, b[i].first_tile) << "row " << i;
        EXPECT_EQ(a[i].tile_count, b[i].tile_count) << "row " << i;
        EXPECT_EQ(a[i].aabb_min, b[i].aabb_min) << "row " << i;
        EXPECT_EQ(a[i].aabb_max, b[i].aabb_max) << "row " << i;
    }
}

} // namespace

// ============================================================================
// build_tile_rows
// ============================================================================

TEST(TerrainTileRows, GroupsRowMajorTilesByZBand) {
    JobSystem js; // uninitialized -> inline single-chunk path
    const int rows = 7;
    const int cols = 9;
    auto tiles = make_tile_grid(rows, cols);

    std::vector<TileRow> tile_rows;
    TerrainRenderer::build_tile_rows(js, tiles, tile_rows);

    ASSERT_EQ(tile_rows.size(), static_cast<size_t>(rows));
    for (int r = 0; r < rows; ++r) {
        const TileRow& row = tile_rows[r];
        EXPECT_EQ(row.first_tile, static_cast<size_t>(r) * cols);
        EXPECT_EQ(row.tile_count, static_cast<size_t>(cols));
        // Row AABB = union of its tiles' AABBs.
        EXPECT_FLOAT_EQ(row.aabb_min.x, 0.0f);
        EXPECT_FLOAT_EQ(row.aabb_max.x, static_cast<float>(cols) * TILE_SIZE);
        EXPECT_FLOAT_EQ(row.aabb_min.z, static_cast<float>(r) * TILE_SIZE);
        EXPECT_FLOAT_EQ(row.aabb_max.z, static_cast<float>(r + 1) * TILE_SIZE);
        float max_y = 0.0f;
        for (size_t i = row.first_tile; i < row.first_tile + row.tile_count; ++i) {
            max_y = std::max(max_y, tiles[i].aabb_max.y);
        }
        EXPECT_FLOAT_EQ(row.aabb_max.y, max_y);
    }
}

TEST(TerrainTileRows, EmptyTilesYieldNoRows) {
    JobSystem js;
    std::vector<TileRow> tile_rows{{3, 4, glm::vec3(1.0f), glm::vec3(2.0f)}}; // stale content must be cleared
    TerrainRenderer::build_tile_rows(js, {}, tile_rows);
    EXPECT_TRUE(tile_rows.empty());
}

// Parity: the tile-chunked fan-out (ROW_BUILD_TILE_GRAIN = 512 -> 2+ chunks
// at 1000 tiles, with a z-band spanning the chunk seam) must reproduce the
// serial grouping exactly at any worker count.
TEST(TerrainTileRows, ParallelParityAcrossWorkerCounts) {
    JobSystem js1;
    js1.init(1);
    JobSystem jsN;
    jsN.init(4);

    // 40 x 25 = 1000 tiles; 512-tile chunk boundary falls mid-row (row 20,
    // tile 512 = row 20 col 12), exercising the seam-coalescing merge.
    auto tiles = make_tile_grid(40, 25);
    ASSERT_GT(tiles.size(), TerrainRenderer::ROW_BUILD_TILE_GRAIN);

    std::vector<TileRow> rows_serial;
    TerrainRenderer::build_tile_rows(js1, tiles, rows_serial);
    ASSERT_EQ(rows_serial.size(), 40u);

    for (int iteration = 0; iteration < 8; ++iteration) {
        std::vector<TileRow> rows_parallel;
        TerrainRenderer::build_tile_rows(jsN, tiles, rows_parallel);
        expect_rows_equal(rows_serial, rows_parallel);
    }
}

// ============================================================================
// cull_tile_runs
// ============================================================================

TEST(TerrainTileCulling, MatchesPerTileReferenceAndMergesRuns) {
    JobSystem js; // uninitialized -> inline single-chunk path
    auto tiles = make_tile_grid(40, 25);
    std::vector<TileRow> rows;
    TerrainRenderer::build_tile_rows(js, tiles, rows);

    const Frustum frustum = make_fixed_frustum(glm::vec3(1250.0f, 80.0f, 3500.0f));

    std::vector<TileRun> runs;
    TerrainRenderer::cull_tile_runs(js, frustum, tiles, rows, runs);
    const std::vector<TileRun> expected = reference_runs(frustum, tiles);

    ASSERT_FALSE(runs.empty());
    EXPECT_EQ(runs, expected);

    // Culling actually rejected something, and contiguous tiles merged
    // (fewer runs than visible tiles).
    uint32_t visible_indices = 0;
    for (const TileRun& run : runs) {
        visible_indices += run.index_count;
    }
    uint32_t total_indices = 0;
    for (const TerrainTile& tile : tiles) {
        total_indices += tile.index_count;
    }
    EXPECT_LT(visible_indices, total_indices);
    EXPECT_LT(runs.size(), tiles.size());
}

TEST(TerrainTileCulling, EmptyRowsFallsBackToPlainTileWalk) {
    JobSystem js;
    auto tiles = make_tile_grid(6, 6);
    const Frustum frustum = make_fixed_frustum(glm::vec3(300.0f, 50.0f, 550.0f));

    std::vector<TileRun> runs;
    TerrainRenderer::cull_tile_runs(js, frustum, tiles, /*rows=*/{}, runs);
    EXPECT_EQ(runs, reference_runs(frustum, tiles));
}

// Parity: the row-chunked fan-out (CULL_ROW_GRAIN = 32 -> 2 chunks at 40
// rows) must produce a BIT-identical run list at any worker count — per-chunk
// partial run lists merged in chunk-index order, runs coalesced across the
// chunk seam.
TEST(TerrainTileCulling, ParallelParityAcrossWorkerCounts) {
    JobSystem js1;
    js1.init(1);
    JobSystem jsN;
    jsN.init(4);

    auto tiles = make_tile_grid(40, 25);
    std::vector<TileRow> rows;
    TerrainRenderer::build_tile_rows(js1, tiles, rows);
    ASSERT_GT(rows.size(), TerrainRenderer::CULL_ROW_GRAIN);

    // A frustum whose visible band crosses the row-chunk boundary (rows ~32
    // +/- a few): camera deep in the grid looking across it.
    const Frustum frustum = make_fixed_frustum(glm::vec3(1250.0f, 80.0f, 3400.0f));

    std::vector<TileRun> runs_serial;
    TerrainRenderer::cull_tile_runs(js1, frustum, tiles, rows, runs_serial);
    ASSERT_FALSE(runs_serial.empty());

    for (int iteration = 0; iteration < 8; ++iteration) {
        std::vector<TileRun> runs_parallel;
        TerrainRenderer::cull_tile_runs(jsN, frustum, tiles, rows, runs_parallel);
        EXPECT_EQ(runs_parallel, runs_serial);
    }
}

// ============================================================================
// Paged terrain: the page-AABB reject that sits in FRONT of cull_tile_runs at
// all three draw sites (main, prepass, shadow). cull_tile_runs itself is
// untouched — these tests are about the reject being safe and about what
// paging costs in draw calls.
// ============================================================================

// THE SAFETY PROPERTY, and it is the whole reason the reject can be added
// without moving a pixel: a page's box is the UNION of its tile boxes, so a
// frustum that misses the union misses every tile in it. The reject can
// therefore only skip work the per-tile cull would have thrown away anyway —
// which is what makes a 1x1 grid bit-identical to the pre-paging path.
TEST(TerrainPageCulling, PageRejectOnlySkipsPagesThatWouldDrawNothing) {
    JobSystem js;
    const auto tiles = make_tile_grid(12, 12);
    std::vector<TileRow> rows;
    TerrainRenderer::build_tile_rows(js, tiles, rows);
    const auto [lo, hi] = page_bounds(tiles);

    int rejected = 0;
    int drawn = 0;
    // Sweep the camera around and through the grid so both verdicts occur.
    for (int i = 0; i < 24; ++i) {
        const float t = static_cast<float>(i);
        const Frustum frustum =
            make_fixed_frustum(glm::vec3(-400.0f + t * 130.0f, 60.0f, -600.0f + t * 190.0f));
        std::vector<TileRun> runs;
        TerrainRenderer::cull_tile_runs(js, frustum, tiles, rows, runs);
        if (!frustum.intersects_aabb(lo, hi)) {
            EXPECT_TRUE(runs.empty()) << "page " << i << " was rejected but its tiles were not";
            ++rejected;
        } else if (!runs.empty()) {
            ++drawn;
        }
    }
    EXPECT_GT(rejected, 0) << "the sweep never rejected a page — the fixture proves nothing";
    EXPECT_GT(drawn, 0) << "the sweep never drew a page — the fixture proves nothing";
}

// What paging actually changes: the SET of triangles drawn is identical, and
// each page's cull is the untouched algorithm run on that page's own tiles.
//
// NOTE ON DRAW COUNTS, measured here rather than assumed: a paged frame does
// NOT always issue more draws than the single-buffer frame. A page renumbers
// its tiles into its own index buffer, so tiles from consecutive PAGE rows
// are contiguous there while the same tiles in the world-sized buffer are
// separated by the columns of the neighbouring page — the page's runs can
// merge where the world's could not. The honest statement is that runs never
// merge ACROSS pages (different buffers), so the floor is one draw per
// visible page per walk; the total is not ordered either way.
TEST(TerrainPageCulling, PagesDrawTheSameTrianglesWithPageLocalRuns) {
    JobSystem js;
    constexpr int ROWS = 12;
    constexpr int COLS = 12;
    const auto world = make_tile_grid(ROWS, COLS);
    std::vector<TileRow> world_rows;
    TerrainRenderer::build_tile_rows(js, world, world_rows);

    // 2x2 pages of 6x6 tiles.
    std::vector<std::vector<TerrainTile>> pages;
    std::vector<std::vector<TileRow>> page_rows;
    for (int pz = 0; pz < 2; ++pz) {
        for (int px = 0; px < 2; ++px) {
            pages.push_back(carve_page(world, COLS, pz * 6, pz * 6 + 6, px * 6, px * 6 + 6));
            page_rows.emplace_back();
            TerrainRenderer::build_tile_rows(js, pages.back(), page_rows.back());
        }
    }

    int compared = 0;
    for (int i = 0; i < 16; ++i) {
        const float t = static_cast<float>(i);
        const Frustum frustum = make_fixed_frustum(glm::vec3(100.0f + t * 70.0f, 60.0f, 200.0f + t * 110.0f));

        std::vector<TileRun> single;
        TerrainRenderer::cull_tile_runs(js, frustum, world, world_rows, single);

        uint32_t paged_indices = 0;
        std::size_t visible_pages = 0;
        std::size_t paged_runs = 0;
        for (std::size_t p = 0; p < pages.size(); ++p) {
            const auto [lo, hi] = page_bounds(pages[p]);
            if (!frustum.intersects_aabb(lo, hi)) {
                continue; // the reject, exactly as the draw path applies it
            }
            std::vector<TileRun> runs;
            TerrainRenderer::cull_tile_runs(js, frustum, pages[p], page_rows[p], runs);
            // Per page, the culler is the unmodified one on the page's own
            // tiles — the reject in front of it changes nothing else.
            EXPECT_EQ(runs, reference_runs(frustum, pages[p])) << "camera step " << i << ", page " << p;
            paged_indices += drawn_indices(runs);
            paged_runs += runs.size();
            if (!runs.empty()) {
                ++visible_pages;
            }
        }

        // Same geometry, page-split or not. This is the claim that matters:
        // paging changes how the draws are packaged, never what is drawn.
        EXPECT_EQ(paged_indices, drawn_indices(single)) << "camera step " << i;
        // The floor the draw-call arithmetic is stated against.
        EXPECT_GE(paged_runs, visible_pages) << "camera step " << i;
        if (!single.empty()) {
            ++compared;
        }
    }
    EXPECT_GT(compared, 4) << "the sweep barely drew anything — the fixture proves nothing";
}

// A page with no tiles (nothing streamed in, or a mesh that failed to build)
// must be rejected by its own degenerate box rather than draw the world.
TEST(TerrainPageCulling, AnEmptyPageDrawsNothing) {
    JobSystem js;
    const std::vector<TerrainTile> empty;
    const auto [lo, hi] = page_bounds(empty);
    EXPECT_EQ(lo, glm::vec3(0.0f));
    EXPECT_EQ(hi, glm::vec3(0.0f));

    std::vector<TileRun> runs;
    const Frustum frustum = make_fixed_frustum(glm::vec3(300.0f, 50.0f, 550.0f));
    TerrainRenderer::cull_tile_runs(js, frustum, empty, /*rows=*/{}, runs);
    EXPECT_TRUE(runs.empty());
}
