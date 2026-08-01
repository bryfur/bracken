#include <benchmark/benchmark.h>

#include "bracken/core/random.hpp"
#include "bracken/render/raytracing/bvh.hpp"

using namespace bracken::render::rt;

namespace {

/** Terrain-like geometry: a tessellated heightfield, which is what most of a
 *  scene's triangles actually are. */
void make_heightfield(int n, std::vector<glm::vec3>& pos, std::vector<uint32_t>& idx) {
    bracken::core::Rng rng(7);
    for (int z = 0; z <= n; ++z) {
        for (int x = 0; x <= n; ++x) {
            pos.emplace_back(static_cast<float>(x) * 10.0f, rng.next_range(-40.0f, 40.0f),
                             static_cast<float>(z) * 10.0f);
        }
    }
    const int stride = n + 1;
    for (int z = 0; z < n; ++z) {
        for (int x = 0; x < n; ++x) {
            const auto a = static_cast<uint32_t>(z * stride + x);
            idx.insert(idx.end(), {a, a + static_cast<uint32_t>(stride), a + 1, a + 1,
                                   a + static_cast<uint32_t>(stride),
                                   a + static_cast<uint32_t>(stride) + 1});
        }
    }
}

} // namespace

/**
 * BLAS build cost against triangle count.
 *
 * This is the number that decides whether a BVH can be built at load time or
 * needs an offline bake. The engine builds one BLAS per MODEL, so the range
 * that matters is a few hundred to a few hundred thousand triangles — not the
 * 6.2M the playground draws, because instancing means those are the same
 * meshes over and over.
 */
static void BM_BuildBlas(benchmark::State& state) {
    const int grid = static_cast<int>(state.range(0));
    std::vector<glm::vec3> pos;
    std::vector<uint32_t> idx;
    make_heightfield(grid, pos, idx);

    const auto triangles = static_cast<int64_t>(idx.size() / 3);
    for (auto _ : state) {
        BvhBuildStats stats;
        Blas blas = build_blas(pos, idx, {}, &stats);
        benchmark::DoNotOptimize(blas.nodes.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * triangles);
    state.counters["triangles"] = static_cast<double>(triangles);

    BvhBuildStats stats;
    (void)build_blas(pos, idx, {}, &stats);
    state.counters["nodes"] = stats.node_count;
    state.counters["leaf_avg"] = stats.average_leaf_size;
    state.counters["sah"] = stats.sah_cost;
    state.counters["MB"] =
        static_cast<double>(stats.node_count * sizeof(BvhNode) +
                            stats.triangle_count * sizeof(BvhTriangle)) /
        (1024.0 * 1024.0);
}
BENCHMARK(BM_BuildBlas)->Arg(32)->Arg(128)->Arg(256)->Arg(400)->Unit(benchmark::kMillisecond);

/** SAH bin count against build cost and tree quality — the knob that trades
 *  load time for every-frame traversal time. */
static void BM_BuildBlasBins(benchmark::State& state) {
    std::vector<glm::vec3> pos;
    std::vector<uint32_t> idx;
    make_heightfield(200, pos, idx);

    BvhBuildOptions opt;
    opt.sah_bins = static_cast<uint32_t>(state.range(0));

    for (auto _ : state) {
        Blas blas = build_blas(pos, idx, opt);
        benchmark::DoNotOptimize(blas.nodes.data());
    }
    BvhBuildStats stats;
    (void)build_blas(pos, idx, opt, &stats);
    state.counters["sah_cost"] = stats.sah_cost;
}
BENCHMARK(BM_BuildBlasBins)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Unit(benchmark::kMillisecond);

/** TLAS build over instance counts. Rebuilt when instances move, so unlike the
 *  BLAS this can be per-frame work and its cost is a frame-time question. */
static void BM_BuildTlas(benchmark::State& state) {
    const auto count = static_cast<std::size_t>(state.range(0));
    bracken::core::Rng rng(3);
    std::vector<TlasInstance> instances(count);
    for (std::size_t i = 0; i < count; ++i) {
        const glm::vec3 p{rng.next_range(0.0f, 8000.0f), rng.next_range(0.0f, 200.0f),
                          rng.next_range(0.0f, 8000.0f)};
        instances[i].bounds_min = p;
        instances[i].bounds_max = p + glm::vec3(60.0f);
        instances[i].blas_node_offset = static_cast<uint32_t>(i);
    }

    for (auto _ : state) {
        Tlas tlas = build_tlas(instances);
        benchmark::DoNotOptimize(tlas.nodes.data());
    }
    state.counters["instances"] = static_cast<double>(count);
}
BENCHMARK(BM_BuildTlas)->Arg(256)->Arg(1024)->Arg(4096)->Unit(benchmark::kMicrosecond);
