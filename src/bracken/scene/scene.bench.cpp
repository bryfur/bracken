#include "bracken/contracts/frustum.hpp"
#include "bracken/contracts/render_stats.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/heightmap.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/motion_history.hpp"
#include "bracken/scene/scene_prep.hpp"
#include "bracken/scene/transform_math.hpp"
#include "bracken/testing/frame_test.hpp"
#include <array>
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <string>
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
        glm::vec3 n = map.get_normal_world(c.x, c.y);
        benchmark::DoNotOptimize(n);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HeightmapNormal);

// ============================================================================
// Prepare stage — instance binning over a sealed synthetic packet
// (Phase 10b's gate). GPU-free: ScenePrep is a set of device-free statics and
// ModelManager needs no device to hold bounds. The world mirrors
// scene_prep_parity.test.cpp's (same LCG, same mix of unregistered handles /
// non-instanced / motion ids) so a bin-order regression shows up in both.
// ============================================================================

namespace {

void bench_add_token_mesh(Model& m) {
    Mesh mesh;
    mesh.vertices.resize(3);
    mesh.indices = {0, 1, 2};
    m.meshes.push_back(std::move(mesh));
}

std::unique_ptr<Model> bench_make_model(Lcg& rng) {
    auto m = std::make_unique<Model>();
    bench_add_token_mesh(*m);
    m->bounds.min = {-rng.next_float(0.5f, 8.0f), 0.0f, -rng.next_float(0.5f, 8.0f)};
    m->bounds.max = {rng.next_float(0.5f, 8.0f), rng.next_float(1.0f, 12.0f), rng.next_float(0.5f, 8.0f)};
    m->compute_bounding_sphere();
    return m;
}

glm::mat4 bench_make_transform(Lcg& rng) {
    glm::mat4 t = glm::translate(glm::mat4(1.0f), glm::vec3(rng.next_float(-900.0f, 900.0f),
                                                            rng.next_float(-50.0f, 80.0f),
                                                            rng.next_float(-1200.0f, 500.0f)));
    if ((rng.next_u32() >> 20) % 8u == 0) {
        t = glm::rotate(t, rng.next_float(0.0f, 6.28f), glm::vec3(0, 1, 0));
        t = glm::scale(t, glm::vec3(rng.next_float(0.5f, 3.0f), rng.next_float(0.5f, 3.0f),
                                    rng.next_float(0.5f, 3.0f)));
    }
    return t;
}

// Held by pointer and built once per benchmark instance (packet arenas are
// megabytes; rebuilding per iteration would time the builder, not prep).
struct PrepBenchWorld {
    ModelManager models;
    std::vector<ModelHandle> handles;
    FramePacket packet{1 << 22};
    MotionHistory history;
    Frustum main_frustum{};
    std::array<Frustum, ScenePrep::MAX_SHADOW_CASCADES> cascade_frustums{};
    int active_cascades = 3;
    glm::vec3 camera_pos{0, 50, 100};

    explicit PrepBenchWorld(size_t command_count) {
        Lcg rng{0xC0FFEEu};
        // 64 distinct models — the playground registers ~60 (7 procedural +
        // 52 pack models), and the model count is what decides whether the
        // per-command Model* chase in the cull misses cache.
        for (int i = 0; i < 64; ++i) {
            handles.push_back(models.register_model("bench_static_" + std::to_string(i), bench_make_model(rng))
                                  .value_or(static_cast<ModelHandle>(0)));
        }

        glm::mat4 view = glm::lookAt(camera_pos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        glm::mat4 proj = glm::perspective(glm::radians(70.0f), 16.0f / 9.0f, 0.1f, 1000.0f);
        main_frustum.extract_from_matrix(proj * view);
        for (int c = 0; c < active_cascades; ++c) {
            const float ext = 60.0f * static_cast<float>(1 << c);
            glm::mat4 lview =
                glm::lookAt(glm::vec3(100, 300, 100), glm::vec3(0, 0, -100.0f * c), glm::vec3(0, 1, 0));
            glm::mat4 lproj = glm::ortho(-ext, ext, -ext, ext, 1.0f, 800.0f);
            cascade_frustums[c].extract_from_matrix(lproj * lview);
        }

        for (uint32_t id = 1; id <= 200; id += 2) {
            Lcg r{id * 0x9E3779B9u};
            history.record(id, bench_make_transform(r), MotionHistory::NO_PALETTE);
        }
        history.end_frame();

        FrameBuilder fb(packet);
        for (size_t i = 0; i < command_count; ++i) {
            const bool unregistered = ((rng.next_u32() >> 20) & 15u) == 0;
            const ModelHandle h =
                unregistered ? static_cast<ModelHandle>(4321) : handles[rng.next_u32() % handles.size()];
            const bool non_instanced = ((rng.next_u32() >> 20) & 7u) == 0;
            const uint32_t motion_id = (((rng.next_u32() >> 20) & 3u) == 0) ? 1u + (rng.next_u32() % 200u) : 0u;
            fb.draw_model(h, bench_make_transform(rng), glm::vec4(1.0f),
                          non_instanced ? DrawFlags::ForceNonInstanced : DrawFlags::None, motion_id);
        }
        testing::FrameAccess::seal(fb);
    }

    [[nodiscard]] ScenePrep::PrepParams params() const {
        ScenePrep::PrepParams p;
        p.models = &models;
        p.motion_history = &history;
        p.camera_pos = camera_pos;
        p.main_frustum = &main_frustum;
        p.cascade_frustums = cascade_frustums.data();
        p.active_cascades = active_cascades;
        p.draw_dist_sq = 700.0f * 700.0f;
        p.do_frustum_cull = true;
        p.collect_stats = true;
        p.prev_palette_valid = true;
        return p;
    }
};

} // namespace

static void BM_PrepInstanceBatches(benchmark::State& state) {
    auto& js = core::jobs::JobSystem::instance();
    if (!js.is_initialized()) {
        js.init();
    }
    const size_t command_count = static_cast<size_t>(state.range(0));
    PrepBenchWorld world(command_count);

    ScenePrep::PrepScratch scratch;
    std::vector<std::vector<gpu::InstanceData>> bins;
    std::vector<ModelHandle> touched;
    std::array<std::vector<std::vector<gpu::ShadowInstanceData>>, ScenePrep::MAX_SHADOW_CASCADES> shadow_bins;
    std::array<std::vector<ModelHandle>, ScenePrep::MAX_SHADOW_CASCADES> shadow_touched;
    std::vector<const ModelDrawCommand*> non_instanced;
    FrameStatsPartial stats;
    const ScenePrep::InstanceBinOutputs out{bins, touched, shadow_bins, shadow_touched, non_instanced, stats};

    const ScenePrep::PrepParams p = world.params();
    for (auto _ : state) {
        stats = {};
        ScenePrep::prep_instance_batches(js, world.packet.model_commands(), p, scratch, out);
        benchmark::DoNotOptimize(touched.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(command_count));
}
BENCHMARK(BM_PrepInstanceBatches)->Arg(2048)->Arg(16384);

// ============================================================================
// MotionHistory record/find — one frame's worth of traffic (Phase 10d's gate).
// The per-frame shape is: find_prev() once per command, record() once per
// command, then a single end_frame(). All three are timed together because
// end_frame()'s cost is the point (a node-destroying clear vs a stamp bump).
// ============================================================================

static void BM_MotionHistoryRecordFind(benchmark::State& state) {
    const size_t count = static_cast<size_t>(state.range(0));
    std::vector<uint32_t> ids(count);
    std::vector<glm::mat4> transforms(count);
    Lcg rng{0x5EEDu};
    for (size_t i = 0; i < count; ++i) {
        // Sparse, non-contiguous ids: hosts hand out id ranges per subsystem.
        ids[i] = static_cast<uint32_t>(1 + i * 7);
        transforms[i] = glm::translate(glm::mat4(1.0f), glm::vec3(rng.next_float(-100.0f, 100.0f), 0.0f, 0.0f));
    }

    MotionHistory history;
    // Prime `prev_` so find_prev() hits rather than misses (the steady state).
    for (size_t i = 0; i < count; ++i) {
        history.record(ids[i], transforms[i]);
    }
    history.end_frame();

    for (auto _ : state) {
        for (size_t i = 0; i < count; ++i) {
            const MotionHistory::Entry* prev = history.find_prev(ids[i]);
            benchmark::DoNotOptimize(prev);
            history.record(ids[i], transforms[i], static_cast<uint32_t>(i));
        }
        history.end_frame();
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(count));
}
BENCHMARK(BM_MotionHistoryRecordFind)->Arg(256)->Arg(4096);
