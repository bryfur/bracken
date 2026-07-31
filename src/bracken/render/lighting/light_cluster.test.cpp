#include <gtest/gtest.h>

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/render/lighting/light.hpp"
#include "bracken/render/lighting/light_cluster.hpp"
#include "bracken/scene/camera_state.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <random>

namespace {

using namespace bracken::render::lighting;
using bracken::scene::CameraState;

// Reflective subclass exposing internals for CPU-only testing without needing a GPU.
class TestableClusterGrid : public ClusterGrid {
public:
    using ClusterGrid::ClusterGrid;

    // CPU-only init: reproduces the buffer setup but skips GPU allocation.
    void init_cpu_only() {
        // Touch the public scaffold via begin_frame; default identity camera with
        // a projective camera would be set per test.
    }
};

CameraState make_camera(uint32_t /*w*/, uint32_t /*h*/, float fov_deg = 60.0f, float aspect = 16.0f / 9.0f,
                        float near_plane = 0.1f, float far_plane = 100.0f) {
    CameraState c;
    c.view = glm::mat4(1.0f); // camera at origin looking down -Z
    c.projection = glm::perspective(glm::radians(fov_deg), aspect, near_plane, far_plane);
    c.view_projection = c.projection * c.view;
    c.position = glm::vec3(0.0f);
    return c;
}

// Public-only smoke shim: drives the CPU pipeline by accessing visible state.
// We can't allocate GPU buffers here, so we construct a ClusterGrid without
// init() and exercise the public methods that don't depend on the SSBO handles.
struct CpuOnlyClusterGrid {
    ClusterGrid g;
    bool inited = false;

    // Simulate the begin/add/build pipeline using a stub allocator: skip init().
    void run(const CameraState& cam, uint32_t w, uint32_t h, float n, float f, const std::vector<PointLight>& points,
             const std::vector<SpotLight>& spots) {
        // begin_frame() doesn't touch GPU buffers - safe even without init().
        g.begin_frame(cam, w, h, n, f);
        for (const auto& p : points) g.add_point_light(p);
        for (const auto& s : spots) g.add_spot_light(s);
        g.build();
    }
};

// Replicates the shader-side lookup in clusters.hlsli (clusterZSlice +
// clusterIndexFromScreen, INCLUDING the `1 - uv.y` flip) so the CPU<->shader
// screen-orientation contract is pinned by a CPU test. screen_uv is
// SV_Position-style: (0,0) = top-left of the screen.
uint32_t shader_cluster_index(glm::vec2 screen_uv, float view_depth, const ClusterParams& cp) {
    auto clamp_dim = [](float v, uint32_t dim) {
        return static_cast<uint32_t>(std::clamp(v, 0.0f, static_cast<float>(dim - 1)));
    };
    float depth = std::max(view_depth, cp.zPlanes.x);
    float t = std::log(depth / cp.zPlanes.x) * cp.zPlanes.w;
    uint32_t cz = clamp_dim(std::floor(t * static_cast<float>(cp.gridDim.z)), cp.gridDim.z);
    uint32_t cx = clamp_dim(std::floor(screen_uv.x * static_cast<float>(cp.gridDim.x)), cp.gridDim.x);
    uint32_t cy = clamp_dim(std::floor((1.0f - screen_uv.y) * static_cast<float>(cp.gridDim.y)), cp.gridDim.y);
    return (cz * cp.gridDim.y + cy) * cp.gridDim.x + cx;
}

// Unprojects an NDC (x, y) + positive forward depth to a view-space position,
// mirroring light_cluster.cpp's unproject_corner().
glm::vec3 view_pos_from_ndc(const glm::mat4& inv_proj, float ndc_x, float ndc_y, float depth) {
    glm::vec4 v = inv_proj * glm::vec4(ndc_x, ndc_y, 0.0f, 1.0f);
    v /= v.w;
    float t = -depth / v.z;
    return {v.x * t, v.y * t, -depth};
}

} // namespace

TEST(ClusterGrid, EmptySceneAllOffsetsZero) {
    CpuOnlyClusterGrid cg;
    cg.run(make_camera(1920, 1080), 1920, 1080, 0.1f, 100.0f, {}, {});
    const auto& offsets = cg.g.cluster_offsets();
    ASSERT_EQ(offsets.size(), CLUSTER_COUNT * 2u);
    for (uint32_t i = 0; i < CLUSTER_COUNT; ++i) {
        EXPECT_EQ(offsets[i * 2 + 0], 0u);
        EXPECT_EQ(offsets[i * 2 + 1], 0u);
    }
    EXPECT_TRUE(cg.g.light_indices().empty());
    EXPECT_EQ(cg.g.total_lights(), 0u);
}

TEST(ClusterGrid, SingleLightAtOriginTouchesNearClusters) {
    PointLight p{};
    p.position = glm::vec3(0.0f, 0.0f, -5.0f);
    p.radius = 3.0f;
    p.color = glm::vec3(1.0f);
    p.intensity = 1.0f;

    CpuOnlyClusterGrid cg;
    cg.run(make_camera(1920, 1080), 1920, 1080, 0.1f, 100.0f, {p}, {});

    const auto& offsets = cg.g.cluster_offsets();
    uint32_t total_count = 0;
    for (uint32_t i = 0; i < CLUSTER_COUNT; ++i) {
        total_count += offsets[i * 2 + 1];
    }
    // A 3-unit-radius point light at z=-5 in a 100m frustum must touch at least
    // one cluster.
    EXPECT_GT(total_count, 0u);
    // All emitted indices reference our single header (index 0).
    for (uint32_t idx : cg.g.light_indices()) {
        EXPECT_EQ(idx, 0u);
    }
}

TEST(ClusterGrid, LightFarBehindCameraTouchesNoClusters) {
    PointLight p{};
    // Place light far behind camera (positive Z is behind for default view).
    p.position = glm::vec3(0.0f, 0.0f, 500.0f);
    p.radius = 1.0f;
    p.color = glm::vec3(1.0f);
    p.intensity = 1.0f;

    CpuOnlyClusterGrid cg;
    cg.run(make_camera(1920, 1080), 1920, 1080, 0.1f, 100.0f, {p}, {});

    uint32_t total_count = 0;
    const auto& offsets = cg.g.cluster_offsets();
    for (uint32_t i = 0; i < CLUSTER_COUNT; ++i) {
        total_count += offsets[i * 2 + 1];
    }
    EXPECT_EQ(total_count, 0u);
    EXPECT_TRUE(cg.g.light_indices().empty());
}

TEST(ClusterGrid, RandomLightsAllIndicesInBounds) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> px(-30.0f, 30.0f);
    std::uniform_real_distribution<float> py(-10.0f, 10.0f);
    std::uniform_real_distribution<float> pz(-80.0f, -1.0f);
    std::uniform_real_distribution<float> pr(0.5f, 6.0f);

    std::vector<PointLight> lights;
    for (int i = 0; i < 100; ++i) {
        PointLight p{};
        p.position = glm::vec3(px(rng), py(rng), pz(rng));
        p.radius = pr(rng);
        p.color = glm::vec3(1.0f);
        p.intensity = 1.0f;
        lights.push_back(p);
    }

    CpuOnlyClusterGrid cg;
    cg.run(make_camera(1920, 1080), 1920, 1080, 0.1f, 100.0f, lights, {});

    const uint32_t header_count = cg.g.total_lights();
    EXPECT_EQ(header_count, 100u);
    for (uint32_t idx : cg.g.light_indices()) {
        EXPECT_LT(idx, header_count);
    }
    const auto& offsets = cg.g.cluster_offsets();
    ASSERT_EQ(offsets.size(), CLUSTER_COUNT * 2u);
    for (uint32_t c = 0; c < CLUSTER_COUNT; ++c) {
        uint32_t off = offsets[c * 2 + 0];
        uint32_t cnt = offsets[c * 2 + 1];
        EXPECT_LE(off + cnt, cg.g.light_indices().size());
        EXPECT_LE(cnt, MAX_LIGHTS_PER_CLUSTER);
    }
}

TEST(ClusterGrid, LightLimitTruncatesPastMaxLights) {
    CpuOnlyClusterGrid cg;
    CameraState cam = make_camera(1920, 1080);

    cg.g.begin_frame(cam, 1920, 1080, 0.1f, 100.0f);
    for (uint32_t i = 0; i < MAX_LIGHTS + 50; ++i) {
        PointLight p{};
        p.position = glm::vec3(0.0f, 0.0f, -2.0f);
        p.radius = 0.5f;
        p.color = glm::vec3(1.0f);
        p.intensity = 1.0f;
        cg.g.add_point_light(p);
    }
    cg.g.build();
    EXPECT_EQ(cg.g.total_lights(), MAX_LIGHTS);
}

TEST(ClusterGrid, BuildIsDeterministic) {
    // Build twice with identical inputs; offsets and indices must match exactly.
    std::mt19937 rng(987654);
    std::uniform_real_distribution<float> px(-30.0f, 30.0f);
    std::uniform_real_distribution<float> py(-10.0f, 10.0f);
    std::uniform_real_distribution<float> pz(-80.0f, -1.0f);
    std::uniform_real_distribution<float> pr(0.5f, 6.0f);
    std::vector<PointLight> lights;
    for (int i = 0; i < 200; ++i) {
        PointLight p{};
        p.position = glm::vec3(px(rng), py(rng), pz(rng));
        p.radius = pr(rng);
        p.color = glm::vec3(1.0f);
        p.intensity = 1.0f;
        lights.push_back(p);
    }

    CpuOnlyClusterGrid a;
    CpuOnlyClusterGrid b;
    a.run(make_camera(1920, 1080), 1920, 1080, 0.1f, 100.0f, lights, {});
    b.run(make_camera(1920, 1080), 1920, 1080, 0.1f, 100.0f, lights, {});

    EXPECT_EQ(a.g.cluster_offsets(), b.g.cluster_offsets());
    EXPECT_EQ(a.g.light_indices(), b.g.light_indices());
}

TEST(LightStruct, SizesMatchSpec) {
    // Strides hardcoded in cluster_lighting.hlsli (loadPointLight/loadSpotLight).
    EXPECT_EQ(sizeof(PointLight), 48u);
    EXPECT_EQ(sizeof(SpotLight), 64u);
    EXPECT_EQ(sizeof(LightHeader), 8u);
    // shadow_slot byte offsets the shader reads (Load4 lanes c.y / d.z).
    EXPECT_EQ(offsetof(PointLight, shadow_slot), 36u);
    EXPECT_EQ(offsetof(SpotLight, shadow_slot), 56u);
    // Host-facing defaults: opt-out of shadows, engine slot unassigned.
    PointLight p{};
    SpotLight s{};
    EXPECT_FALSE(p.casts_shadows);
    EXPECT_FALSE(s.casts_shadows);
    EXPECT_EQ(p.shadow_slot, -1);
    EXPECT_EQ(s.shadow_slot, -1);
}

TEST(ClusterGrid, ParamsCarryShaderContract) {
    // cluster_lighting.hlsli clamps its per-cluster loop with maxPerCluster.x
    // and derives the light-payload section offsets from maxPerCluster.y (the
    // light-data buffer capacity). Pin both here so the uniform the shader
    // depends on is always populated.
    CpuOnlyClusterGrid cg;
    cg.run(make_camera(1920, 1080), 1920, 1080, 0.1f, 100.0f, {}, {});
    const auto& mp = cg.g.params().maxPerCluster;
    EXPECT_EQ(mp.x, MAX_LIGHTS_PER_CLUSTER);
    EXPECT_EQ(mp.y, MAX_LIGHTS); // default capacity when init() wasn't given one
}

TEST(ClusterGrid, ShaderScreenLookupMatchesCpuBinningOrientation) {
    // Regression for the vertical flip between CPU binning (cy = 0 <=> NDC
    // y = -1 <=> BOTTOM of screen) and the shader's SV_Position-derived uv
    // (y = 0 = TOP of screen): clusterIndexFromScreen must flip Y.
    CameraState cam = make_camera(1920, 1080); // near 0.1, far 100
    const glm::mat4 inv_proj = glm::inverse(cam.projection);

    // A point light near the BOTTOM of the screen: NDC y = -0.9 at depth 10.
    const float ndc_x = 0.0f;
    const float ndc_y = -0.9f;
    const float depth = 10.0f;
    PointLight p{};
    p.position = view_pos_from_ndc(inv_proj, ndc_x, ndc_y, depth); // view == world (identity view)
    p.radius = 0.5f;
    p.color = glm::vec3(1.0f);
    p.intensity = 1.0f;

    CpuOnlyClusterGrid cg;
    cg.run(cam, 1920, 1080, 0.1f, 100.0f, {p}, {});
    const auto& offsets = cg.g.cluster_offsets();
    ASSERT_EQ(offsets.size(), CLUSTER_COUNT * 2u);

    // A fragment at the light's center: SV_Position uv has y = 0 at the top,
    // so NDC y = -0.9 (bottom) is uv.y = 0.95.
    const glm::vec2 screen_uv(ndc_x * 0.5f + 0.5f, 1.0f - (ndc_y * 0.5f + 0.5f));
    const uint32_t cidx = shader_cluster_index(screen_uv, depth, cg.g.params());
    ASSERT_LT(cidx, CLUSTER_COUNT);
    EXPECT_GE(offsets[cidx * 2 + 1], 1u)
        << "shader-side screen lookup must land in the cluster the CPU binned the light into";

    // The pre-fix (unflipped) lookup mirrored the row to the TOP of the screen;
    // feeding the vertically mirrored uv reproduces it and must find nothing.
    const uint32_t cidx_mirrored = shader_cluster_index({screen_uv.x, 1.0f - screen_uv.y}, depth, cg.g.params());
    EXPECT_EQ(offsets[cidx_mirrored * 2 + 1], 0u)
        << "a bottom-of-screen light must not be visible through the top-row (unflipped) lookup";
}

namespace {

// Contract of the per-light cluster-range insertion: any fragment within a
// light's radius must find that light via the shader-side cluster lookup.
// Parameterized on the camera so both depth conventions are pinned: cluster
// binning consumes the LINEAR near/far passed to begin_frame and only uses
// invProjection for frustum rays, so it must behave identically whether the
// projection is standard-Z or reversed-Z.
void check_binned_clusters_cover_lit_fragments(const CameraState& cam, float near_plane, float far_plane) {
    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> px(-30.0f, 30.0f);
    std::uniform_real_distribution<float> py(-10.0f, 10.0f);
    std::uniform_real_distribution<float> pz(-80.0f, -1.0f);
    std::uniform_real_distribution<float> pr(0.5f, 6.0f);
    std::vector<PointLight> lights;
    for (int i = 0; i < 64; ++i) {
        PointLight p{};
        p.position = glm::vec3(px(rng), py(rng), pz(rng));
        p.radius = pr(rng);
        p.color = glm::vec3(1.0f);
        p.intensity = 1.0f;
        lights.push_back(p);
    }

    CpuOnlyClusterGrid cg;
    cg.run(cam, 1920, 1080, near_plane, far_plane, lights, {});

    const glm::mat4 inv_proj = glm::inverse(cam.projection);
    const auto& offsets = cg.g.cluster_offsets();
    const auto& indices = cg.g.light_indices();

    std::uniform_real_distribution<float> ndc(-0.9f, 0.9f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    int checked = 0;
    for (int s = 0; s < 2000; ++s) {
        const float nx = ndc(rng);
        const float ny = ndc(rng);
        // Log-uniform depth strictly inside [near, far].
        const float depth = near_plane * 1.5f * std::pow(far_plane * 0.9f / (near_plane * 1.5f), unit(rng));
        const glm::vec3 frag = view_pos_from_ndc(inv_proj, nx, ny, depth);

        const glm::vec2 screen_uv(nx * 0.5f + 0.5f, 1.0f - (ny * 0.5f + 0.5f));
        const uint32_t cidx = shader_cluster_index(screen_uv, depth, cg.g.params());
        const uint32_t off = offsets[cidx * 2 + 0];
        const uint32_t cnt = offsets[cidx * 2 + 1];

        for (uint32_t j = 0; j < lights.size(); ++j) {
            // Strictly-inside margin avoids FP disagreement at the exact radius.
            if (glm::distance(frag, lights[j].position) >= lights[j].radius * 0.95f) {
                continue;
            }
            const auto begin = indices.begin() + off;
            const bool found = std::find(begin, begin + cnt, j) != begin + cnt;
            EXPECT_TRUE(found) << "light " << j << " lights fragment at ndc(" << nx << ", " << ny << ") depth "
                               << depth << " but is missing from cluster " << cidx;
            ++checked;
        }
    }
    // The sample set must actually have exercised lit fragments.
    EXPECT_GT(checked, 0);
}

} // namespace

TEST(ClusterGrid, BinnedClustersCoverAllLitFragments) {
    const float near_plane = 0.1f;
    const float far_plane = 100.0f;
    CameraState cam = make_camera(1920, 1080, 60.0f, 16.0f / 9.0f, near_plane, far_plane);
    check_binned_clusters_cover_lit_fragments(cam, near_plane, far_plane);
}

TEST(ClusterGrid, BinnedClustersCoverAllLitFragmentsReversedZ) {
    // Same scene under the engine's runtime reversed-Z projection (near/far
    // swapped in the perspective call). begin_frame still receives the linear
    // near/far distances — those are z-slice parameters, not device depths.
    const float near_plane = 0.1f;
    const float far_plane = 100.0f;
    CameraState cam;
    cam.view = glm::mat4(1.0f);
    cam.projection = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, far_plane, near_plane);
    cam.view_projection = cam.projection * cam.view;
    cam.position = glm::vec3(0.0f);
    check_binned_clusters_cover_lit_fragments(cam, near_plane, far_plane);
}

namespace {

// Seeded light sets for the parallel parity tests — fixed seed constants, no
// runtime RNG (docs/multithreading_plan.md §6.2).
constexpr uint32_t k_parity_point_seed = 424242;
constexpr uint32_t k_parity_spot_seed = 971131;

std::vector<PointLight> make_seeded_point_lights(std::size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> px(-30.0f, 30.0f);
    std::uniform_real_distribution<float> py(-10.0f, 10.0f);
    std::uniform_real_distribution<float> pz(-80.0f, -1.0f);
    std::uniform_real_distribution<float> pr(0.5f, 6.0f);
    std::vector<PointLight> lights(count);
    for (auto& p : lights) {
        p.position = glm::vec3(px(rng), py(rng), pz(rng));
        p.radius = pr(rng);
        p.color = glm::vec3(1.0f);
        p.intensity = 1.0f;
    }
    return lights;
}

std::vector<SpotLight> make_seeded_spot_lights(std::size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> px(-30.0f, 30.0f);
    std::uniform_real_distribution<float> py(-10.0f, 10.0f);
    std::uniform_real_distribution<float> pz(-80.0f, -1.0f);
    std::uniform_real_distribution<float> pr(0.5f, 6.0f);
    std::uniform_real_distribution<float> pd(-1.0f, 1.0f);
    std::vector<SpotLight> lights(count);
    for (auto& s : lights) {
        s.position = glm::vec3(px(rng), py(rng), pz(rng));
        s.radius = pr(rng);
        glm::vec3 dir(pd(rng), pd(rng), pd(rng));
        s.direction = glm::dot(dir, dir) > 1e-6f ? glm::normalize(dir) : glm::vec3(0.0f, -1.0f, 0.0f);
        s.inner_cos = glm::cos(glm::radians(15.0f));
        s.outer_cos = glm::cos(glm::radians(25.0f));
        s.color = glm::vec3(1.0f);
        s.intensity = 1.0f;
    }
    return lights;
}

// Drives begin_frame/add/build against a caller-owned pool. Two frames so the
// second build also exercises the packing dirty-diff over populated bins.
void run_frames(bracken::core::jobs::JobSystem& js, ClusterGrid& g, const CameraState& cam,
                const std::vector<PointLight>& points, const std::vector<SpotLight>& spots) {
    for (int frame = 0; frame < 2; ++frame) {
        g.begin_frame(js, cam, 1920, 1080, 0.1f, 100.0f);
        for (const auto& p : points) g.add_point_light(p);
        for (const auto& s : spots) g.add_spot_light(s);
        g.build(js);
    }
}

} // namespace

TEST(ClusterGrid, ParallelBuildBitIdenticalAcrossWorkerCounts) {
    // Determinism contract (§6.2): the cluster-chunked binning fan-out must
    // produce bins (observable through the packed offset/count table) and
    // packed index arrays BIT-identical at every worker count. Caller-owned
    // pools: 1 worker (parallel_for collapses to one inline chunk) vs 4.
    const auto points = make_seeded_point_lights(96, k_parity_point_seed);
    const auto spots = make_seeded_spot_lights(32, k_parity_spot_seed);
    const CameraState cam = make_camera(1920, 1080);

    bracken::core::jobs::JobSystem js1;
    js1.init(1);
    bracken::core::jobs::JobSystem js4;
    js4.init(4);
    bracken::core::jobs::JobSystem js_uninit; // never init(): guaranteed inline path

    ClusterGrid g1;
    ClusterGrid g4;
    ClusterGrid g0;
    run_frames(js1, g1, cam, points, spots);
    run_frames(js4, g4, cam, points, spots);
    run_frames(js_uninit, g0, cam, points, spots);

    ASSERT_FALSE(g1.light_indices().empty());
    // Bins are fully captured by (offset, count) + the packed index stream.
    EXPECT_EQ(g1.cluster_offsets(), g4.cluster_offsets());
    EXPECT_EQ(g1.light_indices(), g4.light_indices());
    EXPECT_EQ(g0.cluster_offsets(), g4.cluster_offsets());
    EXPECT_EQ(g0.light_indices(), g4.light_indices());
}

TEST(ClusterGrid, ParallelBuildTruncationBitIdentical) {
    // 150 identical overlapping lights force the MAX_LIGHTS_PER_CLUSTER
    // cutoff: the parallel per-cluster form must keep exactly the same lowest
    // light indices as the inline path.
    std::vector<PointLight> lights(150);
    for (auto& p : lights) {
        p.position = glm::vec3(0.0f, 0.0f, -10.0f);
        p.radius = 3.0f;
        p.color = glm::vec3(1.0f);
        p.intensity = 1.0f;
    }
    const CameraState cam = make_camera(1920, 1080);

    bracken::core::jobs::JobSystem js4;
    js4.init(4);
    bracken::core::jobs::JobSystem js_uninit;

    ClusterGrid g4;
    ClusterGrid g0;
    run_frames(js4, g4, cam, lights, {});
    run_frames(js_uninit, g0, cam, lights, {});

    EXPECT_EQ(g0.cluster_offsets(), g4.cluster_offsets());
    EXPECT_EQ(g0.light_indices(), g4.light_indices());

    uint32_t max_count = 0;
    for (uint32_t c = 0; c < CLUSTER_COUNT; ++c) {
        max_count = std::max(max_count, g4.cluster_offsets()[c * 2 + 1]);
    }
    EXPECT_EQ(max_count, MAX_LIGHTS_PER_CLUSTER);
}

TEST(ClusterGrid, PerClusterCountNeverExceedsCap) {
    // 150 identical overlapping lights: every touched cluster must truncate at
    // MAX_LIGHTS_PER_CLUSTER, keeping the lowest light indices.
    std::vector<PointLight> lights;
    for (int i = 0; i < 150; ++i) {
        PointLight p{};
        p.position = glm::vec3(0.0f, 0.0f, -10.0f);
        p.radius = 3.0f;
        p.color = glm::vec3(1.0f);
        p.intensity = 1.0f;
        lights.push_back(p);
    }

    CpuOnlyClusterGrid cg;
    cg.run(make_camera(1920, 1080), 1920, 1080, 0.1f, 100.0f, lights, {});

    const auto& offsets = cg.g.cluster_offsets();
    uint32_t max_count = 0;
    for (uint32_t c = 0; c < CLUSTER_COUNT; ++c) {
        const uint32_t cnt = offsets[c * 2 + 1];
        EXPECT_LE(cnt, MAX_LIGHTS_PER_CLUSTER);
        max_count = std::max(max_count, cnt);
    }
    EXPECT_EQ(max_count, MAX_LIGHTS_PER_CLUSTER);
}
