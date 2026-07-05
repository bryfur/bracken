#include <benchmark/benchmark.h>

#include "bracken/render/lighting/light.hpp"
#include "bracken/render/lighting/light_cluster.hpp"
#include "bracken/scene/camera_state.hpp"

#include <glm/ext/matrix_clip_space.hpp>

#include <cstdint>
#include <random>
#include <vector>

namespace {

using namespace bracken::render::lighting;
using bracken::scene::CameraState;

constexpr uint32_t SCREEN_W = 1920;
constexpr uint32_t SCREEN_H = 1080;
constexpr float NEAR_PLANE = 0.1f;
constexpr float FAR_PLANE = 100.0f;

CameraState make_camera(float fov_deg = 60.0f, float aspect = 16.0f / 9.0f, float near_plane = NEAR_PLANE,
                        float far_plane = FAR_PLANE) {
    CameraState c;
    c.view = glm::mat4(1.0f); // camera at origin looking down -Z
    c.projection = glm::perspective(glm::radians(fov_deg), aspect, near_plane, far_plane);
    c.view_projection = c.projection * c.view;
    c.position = glm::vec3(0.0f);
    return c;
}

// Lights scattered through the view volume (camera at origin, -Z forward) so
// binning intersects real cluster AABBs instead of trivially rejecting.
std::vector<PointLight> make_point_lights(std::size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> px(-30.0f, 30.0f);
    std::uniform_real_distribution<float> py(-10.0f, 10.0f);
    std::uniform_real_distribution<float> pz(-80.0f, -1.0f);
    std::uniform_real_distribution<float> pr(0.5f, 6.0f);

    std::vector<PointLight> lights;
    lights.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        PointLight p{};
        p.position = glm::vec3(px(rng), py(rng), pz(rng));
        p.radius = pr(rng);
        p.color = glm::vec3(1.0f);
        p.intensity = 1.0f;
        lights.push_back(p);
    }
    return lights;
}

std::vector<SpotLight> make_spot_lights(std::size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> px(-30.0f, 30.0f);
    std::uniform_real_distribution<float> py(-10.0f, 10.0f);
    std::uniform_real_distribution<float> pz(-80.0f, -1.0f);
    std::uniform_real_distribution<float> pr(0.5f, 6.0f);
    std::uniform_real_distribution<float> pd(-1.0f, 1.0f);

    std::vector<SpotLight> lights;
    lights.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        SpotLight s{};
        s.position = glm::vec3(px(rng), py(rng), pz(rng));
        s.radius = pr(rng);
        glm::vec3 dir(pd(rng), pd(rng), pd(rng));
        s.direction = glm::dot(dir, dir) > 1e-6f ? glm::normalize(dir) : glm::vec3(0.0f, -1.0f, 0.0f);
        s.inner_cos = glm::cos(glm::radians(15.0f));
        s.outer_cos = glm::cos(glm::radians(25.0f));
        s.color = glm::vec3(1.0f);
        s.intensity = 1.0f;
        lights.push_back(s);
    }
    return lights;
}

} // namespace

// Per-frame CPU cost of clustered light binning: begin_frame() + add_*_light()
// + build(). ClusterGrid is constructed without init() — begin_frame/add/build
// never touch the GPU buffers (see light_cluster.test.cpp), so setup stays
// GPU-free. Reusing one grid across iterations matches steady-state frame
// behavior: cluster AABBs are cached while the projection is unchanged.
static void BM_ClusterBuild(benchmark::State& state) {
    const std::size_t light_count = static_cast<std::size_t>(state.range(0));
    const std::vector<PointLight> lights = make_point_lights(light_count, 12345);
    const CameraState cam = make_camera();
    ClusterGrid grid;

    for (auto _ : state) {
        grid.begin_frame(cam, SCREEN_W, SCREEN_H, NEAR_PLANE, FAR_PLANE);
        for (const auto& p : lights) {
            grid.add_point_light(p);
        }
        grid.build();
        benchmark::DoNotOptimize(grid.cluster_offsets().data());
        benchmark::DoNotOptimize(grid.light_indices().data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(light_count));
}
BENCHMARK(BM_ClusterBuild)->Arg(16)->Arg(64)->Arg(256);

// Mixed point + spot workload (half each) — exercises both header types and
// the per-type payload lookup in build().
static void BM_ClusterBuildMixed(benchmark::State& state) {
    const std::size_t light_count = static_cast<std::size_t>(state.range(0));
    const std::vector<PointLight> points = make_point_lights(light_count / 2, 12345);
    const std::vector<SpotLight> spots = make_spot_lights(light_count - light_count / 2, 67890);
    const CameraState cam = make_camera();
    ClusterGrid grid;

    for (auto _ : state) {
        grid.begin_frame(cam, SCREEN_W, SCREEN_H, NEAR_PLANE, FAR_PLANE);
        for (const auto& p : points) {
            grid.add_point_light(p);
        }
        for (const auto& s : spots) {
            grid.add_spot_light(s);
        }
        grid.build();
        benchmark::DoNotOptimize(grid.cluster_offsets().data());
        benchmark::DoNotOptimize(grid.light_indices().data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(light_count));
}
BENCHMARK(BM_ClusterBuildMixed)->Arg(16)->Arg(64)->Arg(256);
