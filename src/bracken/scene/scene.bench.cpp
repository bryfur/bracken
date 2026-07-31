#include "bracken/contracts/frustum.hpp"
#include "bracken/heightmap.hpp"
#include "bracken/scene/transform_math.hpp"
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

using namespace bracken;
using namespace bracken::scene;

namespace {

// Minimal deterministic LCG (numerical-recipes constants) so benchmark inputs
// are reproducible without <random> setup cost.
struct Lcg {
    uint32_t state = 0x12345678u;

    uint32_t next_u32() {
        state = state * 1664525u + 1013904223u;
        return state;
    }

    // Uniform float in [0, 1).
    float next_float01() { return static_cast<float>(next_u32() >> 8) * (1.0f / 16777216.0f); }

    float next_float(float lo, float hi) { return lo + next_float01() * (hi - lo); }
};

// Typical gameplay camera: perspective, wide FOV, long far plane.
glm::mat4 make_view_proj() {
    glm::mat4 view = glm::lookAt(glm::vec3(0, 50, 100), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    glm::mat4 proj = glm::perspective(glm::radians(70.0f), 16.0f / 9.0f, 0.1f, 1000.0f);
    return proj * view;
}

struct Aabb {
    glm::vec3 min;
    glm::vec3 max;
};

struct Sphere {
    glm::vec3 center;
    float radius;
};

constexpr size_t kShapeCount = 10000;

// Centers span a box straddling the frustum (including behind the camera and
// beyond the side planes) so culling sees a realistic mix of inside/outside.
glm::vec3 random_center(Lcg& rng) {
    return {rng.next_float(-600.0f, 600.0f), rng.next_float(-200.0f, 200.0f), rng.next_float(-1200.0f, 400.0f)};
}

Heightmap make_heightmap() {
    Heightmap map;
    map.resolution = 257;
    map.world_origin_x = 0.0f;
    map.world_origin_z = 0.0f;
    map.world_size = 1024.0f;
    map.height_data.resize(static_cast<size_t>(map.resolution) * map.resolution);
    Lcg rng;
    for (uint16_t& h : map.height_data) {
        h = static_cast<uint16_t>(rng.next_u32() & 0xFFFFu);
    }
    return map;
}

// Power of two so the benchmark loop can wrap with a mask instead of a modulo.
constexpr size_t kCoordCount = 4096;

std::vector<glm::vec2> make_world_coords(const Heightmap& map) {
    Lcg rng{0x9e3779b9u};
    std::vector<glm::vec2> coords(kCoordCount);
    for (glm::vec2& c : coords) {
        c = {rng.next_float(0.0f, map.world_size), rng.next_float(0.0f, map.world_size)};
    }
    return coords;
}

} // namespace

// ============================================================================
// Frustum plane extraction
// ============================================================================

static void BM_FrustumExtract(benchmark::State& state) {
    glm::mat4 vp = make_view_proj();
    Frustum frustum{};
    for (auto _ : state) {
        benchmark::DoNotOptimize(vp);
        frustum.extract_from_matrix(vp);
        benchmark::DoNotOptimize(frustum);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_FrustumExtract);

// ============================================================================
// Frustum culling over a pre-generated shape set
// ============================================================================

static void BM_FrustumCullAABB(benchmark::State& state) {
    Frustum frustum{};
    frustum.extract_from_matrix(make_view_proj());

    Lcg rng;
    std::vector<Aabb> boxes;
    boxes.reserve(kShapeCount);
    for (size_t i = 0; i < kShapeCount; ++i) {
        glm::vec3 center = random_center(rng);
        glm::vec3 half(rng.next_float(0.5f, 8.0f), rng.next_float(0.5f, 8.0f), rng.next_float(0.5f, 8.0f));
        boxes.push_back({center - half, center + half});
    }

    for (auto _ : state) {
        size_t visible = 0;
        for (const Aabb& box : boxes) {
            visible += frustum.intersects_aabb(box.min, box.max) ? 1 : 0;
        }
        benchmark::DoNotOptimize(visible);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(kShapeCount));
}
BENCHMARK(BM_FrustumCullAABB);

static void BM_FrustumCullSphere(benchmark::State& state) {
    Frustum frustum{};
    frustum.extract_from_matrix(make_view_proj());

    Lcg rng;
    std::vector<Sphere> spheres;
    spheres.reserve(kShapeCount);
    for (size_t i = 0; i < kShapeCount; ++i) {
        spheres.push_back({random_center(rng), rng.next_float(0.5f, 8.0f)});
    }

    for (auto _ : state) {
        size_t visible = 0;
        for (const Sphere& sphere : spheres) {
            visible += frustum.intersects_sphere(sphere.center, sphere.radius) ? 1 : 0;
        }
        benchmark::DoNotOptimize(visible);
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(kShapeCount));
}
BENCHMARK(BM_FrustumCullSphere);

// ============================================================================
// Normal matrix: uniform-scale fast path vs inverse-transpose slow path
// ============================================================================

static void BM_ComputeNormalMatrixUniform(benchmark::State& state) {
    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(10, 2, -5));
    model = glm::rotate(model, 0.7f, glm::vec3(0, 1, 0));
    model = glm::scale(model, glm::vec3(2.0f));
    for (auto _ : state) {
        benchmark::DoNotOptimize(model);
        glm::mat4 n = compute_normal_matrix(model);
        benchmark::DoNotOptimize(n);
    }
}
BENCHMARK(BM_ComputeNormalMatrixUniform);

static void BM_ComputeNormalMatrixNonUniform(benchmark::State& state) {
    glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(10, 2, -5));
    model = glm::rotate(model, 0.7f, glm::vec3(0, 1, 0));
    model = glm::scale(model, glm::vec3(2.0f, 1.0f, 0.5f));
    for (auto _ : state) {
        benchmark::DoNotOptimize(model);
        glm::mat4 n = compute_normal_matrix(model);
        benchmark::DoNotOptimize(n);
    }
}
BENCHMARK(BM_ComputeNormalMatrixNonUniform);

// ============================================================================
// Heightmap sampling (257x257, bilinear) at pre-generated random coordinates
// ============================================================================

static void BM_HeightmapSample(benchmark::State& state) {
    const Heightmap map = make_heightmap();
    const std::vector<glm::vec2> coords = make_world_coords(map);

    size_t i = 0;
    for (auto _ : state) {
        const glm::vec2& c = coords[i++ & (kCoordCount - 1)];
        float h = map.get_height_world(c.x, c.y);
        benchmark::DoNotOptimize(h);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HeightmapSample);

static void BM_HeightmapNormal(benchmark::State& state) {
    const Heightmap map = make_heightmap();
    const std::vector<glm::vec2> coords = make_world_coords(map);

    size_t i = 0;
    for (auto _ : state) {
        const glm::vec2& c = coords[i++ & (kCoordCount - 1)];
        float nx = 0.0f;
        float ny = 0.0f;
        float nz = 0.0f;
        map.get_normal_world(c.x, c.y, nx, ny, nz);
        benchmark::DoNotOptimize(nx);
        benchmark::DoNotOptimize(ny);
        benchmark::DoNotOptimize(nz);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HeightmapNormal);
