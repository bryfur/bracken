#include "engine/model_loader.hpp"
#include "engine/model_utils.hpp"
#include "engine/procedural/rock_generator.hpp"
#include "engine/procedural/tree_generator.hpp"
#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>

using namespace mmo::engine::procedural;
using namespace mmo::engine;

namespace {

uint64_t total_vertices(const Model& model) {
    uint64_t count = 0;
    for (const auto& mesh : model.meshes) {
        count += mesh.vertices.size();
    }
    return count;
}

} // namespace

static void BM_TreeOak(benchmark::State& state) {
    uint64_t vertices = 0;
    for (auto _ : state) {
        auto model = TreeGenerator::generate_oak(42);
        vertices = total_vertices(*model);
        benchmark::DoNotOptimize(model);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * vertices));
}
BENCHMARK(BM_TreeOak)->Unit(benchmark::kMillisecond);

static void BM_TreePine(benchmark::State& state) {
    uint64_t vertices = 0;
    for (auto _ : state) {
        auto model = TreeGenerator::generate_pine(42);
        vertices = total_vertices(*model);
        benchmark::DoNotOptimize(model);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * vertices));
}
BENCHMARK(BM_TreePine)->Unit(benchmark::kMillisecond);

static void BM_RockBoulder(benchmark::State& state) {
    // Boulder preset parameters (RockParams defaults match generate_boulder),
    // with the marching-cubes grid resolution driven by the benchmark arg.
    RockParams params;
    params.seed = 880;
    params.resolution = static_cast<int>(state.range(0));
    uint64_t vertices = 0;
    for (auto _ : state) {
        auto model = RockGenerator::generate(params);
        vertices = total_vertices(*model);
        benchmark::DoNotOptimize(model);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * vertices));
}
BENCHMARK(BM_RockBoulder)->Arg(40)->Arg(80)->Unit(benchmark::kMillisecond);

static void BM_ComputeTangents(benchmark::State& state) {
    // Reuse one generated tree's branch mesh as a realistic input (thousands
    // of vertices with real UVs). compute_tangents_from_uvs only overwrites
    // Vertex3D::tangent, so a single working copy can be re-processed.
    auto model = TreeGenerator::generate_oak(42);
    std::vector<gpu::Vertex3D> vertices = model->meshes[0].vertices;
    const std::vector<uint32_t>& indices = model->meshes[0].indices;
    for (auto _ : state) {
        compute_tangents_from_uvs(vertices, indices);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(vertices.size()));
}
BENCHMARK(BM_ComputeTangents)->Unit(benchmark::kMillisecond);
