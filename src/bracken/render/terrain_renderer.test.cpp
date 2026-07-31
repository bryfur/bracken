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
