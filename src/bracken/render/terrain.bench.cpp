// Terrain tile culling — the CPU cost of ONE cull_tile_runs pass.
//
// This is the measurement Phase 10c is decided on: the main view culls the
// same frustum twice per frame (the forward pass and the depth prepass), so
// caching the main-view run list saves exactly one of these calls per frame.
// The grid matches the playground's real terrain (a 42000-unit world tiles
// into ~52x52 = ~2700 tiles, terrain_renderer.cpp:329-334), so the number
// here is the per-frame saving, not a synthetic one.
//
// GPU-free: build_tile_rows/cull_tile_runs are public static pure functions.

#include "bracken/render/terrain_renderer.hpp"

#include "bracken/contracts/frustum.hpp"
#include "bracken/core/jobs/job_system.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

using bracken::core::jobs::JobSystem;
using bracken::render::TerrainRenderer;
using bracken::scene::Frustum;

namespace {

using TerrainTile = TerrainRenderer::TerrainTile;
using TileRow = TerrainRenderer::TileRow;
using TileRun = TerrainRenderer::TileRun;

constexpr float k_tile_size = 800.0f; // TILE_CELLS(32) * cell_size(25)

// Row-major grid with the playground's tile footprint and varied heights, so
// row AABBs are not degenerate and the per-tile tests actually run.
std::vector<TerrainTile> make_grid(int rows, int cols) {
    std::vector<TerrainTile> tiles;
    tiles.reserve(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    uint32_t next_index = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            TerrainTile tile;
            tile.first_index = next_index;
            tile.index_count = 6144 + 6 * static_cast<uint32_t>((r + c) % 3);
            next_index += tile.index_count;
            const float max_y = 40.0f + static_cast<float>((r * 31 + c * 17) % 200);
            tile.aabb_min = {static_cast<float>(c) * k_tile_size, -40.0f, static_cast<float>(r) * k_tile_size};
            tile.aabb_max = {static_cast<float>(c + 1) * k_tile_size, max_y,
                             static_cast<float>(r + 1) * k_tile_size};
            tiles.push_back(tile);
        }
    }
    return tiles;
}

// Engine-style reversed-Z frustum (near/far swapped) standing mid-world and
// looking across it — the realistic mix where 80-95% of tiles are rejected.
Frustum make_camera_frustum(int rows, int cols) {
    const glm::vec3 cam(static_cast<float>(cols) * k_tile_size * 0.5f, 200.0f,
                        static_cast<float>(rows) * k_tile_size * 0.75f);
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 8000.0f, 0.1f);
    glm::mat4 view = glm::lookAt(cam, cam + glm::vec3(0.0f, -0.15f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    Frustum f;
    f.extract_from_matrix(proj * view);
    return f;
}

} // namespace

static void BM_TerrainCullTileRuns(benchmark::State& state) {
    auto& js = JobSystem::instance();
    if (!js.is_initialized()) {
        js.init();
    }
    const int side = static_cast<int>(state.range(0));
    const std::vector<TerrainTile> tiles = make_grid(side, side);
    std::vector<TileRow> rows;
    TerrainRenderer::build_tile_rows(js, tiles, rows);
    const Frustum frustum = make_camera_frustum(side, side);

    std::vector<TileRun> runs;
    for (auto _ : state) {
        TerrainRenderer::cull_tile_runs(js, frustum, tiles, rows, runs);
        benchmark::DoNotOptimize(runs.data());
        benchmark::ClobberMemory();
    }
    state.counters["runs"] = static_cast<double>(runs.size());
    state.counters["tiles"] = static_cast<double>(tiles.size());
}
// 52 = the playground's terrain; 16 = a small world that stays a single chunk.
BENCHMARK(BM_TerrainCullTileRuns)->Arg(16)->Arg(52);
