#include <gtest/gtest.h>

#include "bracken/contracts/camera_state.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/render/lighting/cluster_index.hpp"
#include "bracken/render/lighting/light.hpp"
#include "bracken/render/lighting/light_cluster.hpp"
#include "bracken/render/lighting/light_target.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <vector>

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
        for (const auto& p : points) (void)g.add_point_light(p);
        for (const auto& s : spots) (void)g.add_spot_light(s);
        g.build();
    }
};

// The shader-side lookup, as the ENGINE ships it: cluster_index.hpp is the CPU
// mirror of shaders/cluster_index.hlsli, and light_cluster.cpp's binner calls
// its clusterZSlice half itself. These tests used to carry a hand-copied
// transliteration here — which pinned the contract only as well as the copy
// was maintained.
uint32_t shader_cluster_index(glm::vec2 screen_uv, float view_depth, const ClusterParams& cp) {
    return cluster_index_from_screen(screen_uv, view_depth, cp);
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
        (void)cg.g.add_point_light(p);
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
    // Strides hardcoded in cluster_light_load.hlsli (loadPointLight/loadSpotLight).
    // SpotLight is 80 since R6: source_radius found no spare padding there and
    // the stride literal moved with it. PointLight had a declared pad and did
    // not grow.
    EXPECT_EQ(sizeof(PointLight), 48u);
    EXPECT_EQ(sizeof(SpotLight), 80u);
    EXPECT_EQ(sizeof(LightHeader), 8u);
    // shadow_slot byte offsets the shader reads (Load4 lanes c.y / d.z).
    EXPECT_EQ(offsetof(PointLight, shadow_slot), 36u);
    EXPECT_EQ(offsetof(SpotLight, shadow_slot), 56u);
    // R6's source_radius, at the offsets the shader loads it from.
    EXPECT_EQ(offsetof(PointLight, source_radius), 44u);
    EXPECT_EQ(offsetof(SpotLight, source_radius), 64u);
    // 0 means "the host did not author one"; the traced pass substitutes its
    // own default. It is NOT "a point light" — see contracts/light.hpp.
    EXPECT_EQ(PointLight{}.source_radius, 0.0f);
    EXPECT_EQ(SpotLight{}.source_radius, 0.0f);
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
void check_point_light_coverage(const CameraState& cam, float near_plane, float far_plane,
                                const std::vector<PointLight>& lights) {
    std::mt19937 rng(2024);
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
    check_point_light_coverage(cam, near_plane, far_plane, lights);
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

TEST(ClusterGrid, BinnedClustersCoverLitFragmentsForLightsStraddlingTheNearPlane) {
    // The range precompute clips the light's view-space AABB to z <= -near
    // before projecting it, which is what lets the "outside the frustum
    // sideways" reject fire for a light BESIDE the camera. Getting that clip
    // wrong narrows the X/Y tile range and punches a rectangular hole in the
    // light pool, so pin the coverage property on exactly those lights:
    // every one of these has view depth under its own radius, so every one
    // takes the clipped path.
    const float near_plane = 0.1f;
    const float far_plane = 100.0f;
    const CameraState cam = make_camera(1920, 1080, 60.0f, 16.0f / 9.0f, near_plane, far_plane);

    std::vector<PointLight> lights;
    const auto add = [&lights](glm::vec3 pos, float radius) {
        PointLight p{};
        p.position = pos;
        p.radius = radius;
        p.color = glm::vec3(1.0f);
        p.intensity = 1.0f;
        lights.push_back(p);
    };
    add({0.0f, 0.0f, -2.0f}, 50.0f);    // camera inside the sphere, on axis
    add({0.0f, 0.0f, 1.0f}, 30.0f);     // apex slightly BEHIND the camera
    add({40.0f, 0.0f, -5.0f}, 60.0f);   // beside the camera, far off axis right
    add({-40.0f, 0.0f, -5.0f}, 60.0f);  // ... and left
    add({0.0f, 25.0f, -3.0f}, 40.0f);   // ... and above
    add({18.0f, -12.0f, -0.5f}, 25.0f); // straddling the near plane diagonally
    check_point_light_coverage(cam, near_plane, far_plane, lights);
}

TEST(ClusterGrid, SlicingTheCameraPlanesLeavesNoUnreachableSliceAtEitherEnd) {
    // begin_frame's near/far are Z-SLICE parameters, and the exponential
    // distribution spends slices where they are given. Handing it a near that
    // is far in front of the camera's own near plane buys 8 of 24 slices that
    // no fragment can ever occupy, and a far short of the camera's far stops
    // binning local lights at that distance while the same geometry is still
    // rasterized past it. This pins the property the CALLER
    // (SceneRenderer::render, which owns the begin_frame call) has to preserve:
    // fed the camera's own planes, both ends of the grid are reachable.
    const float near_plane = 5.0f;      // CameraModeConfig::near_plane
    const float far_plane = 15000.0f;   // CameraModeConfig::far_plane
    CameraState cam;
    cam.view = glm::mat4(1.0f);
    // Reversed-Z, as the engine builds it (near/far swapped in the call).
    cam.projection = glm::perspective(glm::radians(55.0f), 16.0f / 9.0f, far_plane, near_plane);
    cam.view_projection = cam.projection * cam.view;
    cam.position = glm::vec3(0.0f);

    CpuOnlyClusterGrid cg;
    cg.run(cam, 1600, 900, near_plane, far_plane, {}, {});
    const ClusterParams& cp = cg.g.params();
    EXPECT_EQ(cluster_z_slice(near_plane * 1.001f, cp), 0u)
        << "slice 0 must start AT the camera near plane, not in front of it";
    EXPECT_EQ(cluster_z_slice(far_plane * 0.999f, cp), CLUSTER_DIM_Z - 1)
        << "the last slice must reach the camera far plane";

    // ...and a light out past a typical draw distance is still binned, because
    // the geometry it lights is still being drawn out there.
    PointLight distant{};
    distant.position = glm::vec3(0.0f, 0.0f, -6000.0f);
    distant.radius = 400.0f;
    distant.color = glm::vec3(1.0f);
    distant.intensity = 1.0f;
    cg.run(cam, 1600, 900, near_plane, far_plane, {distant}, {});
    EXPECT_FALSE(cg.g.light_indices().empty()) << "a light at depth 6000 inside a 15000-unit grid must reach a froxel";
}

namespace {

// The spot-cone half of the coverage contract, and the one that matters most:
// the binner now culls a spot to its CONE rather than its bounding sphere, and
// a cone test that is not conservative produces a froxel-shaped hole in the
// light pool — the failure mode this subsystem is worst at diagnosing.
void check_spot_light_coverage(const CameraState& cam, float near_plane, float far_plane,
                               const std::vector<SpotLight>& lights) {
    std::mt19937 rng(90210);
    CpuOnlyClusterGrid cg;
    cg.run(cam, 1920, 1080, near_plane, far_plane, {}, lights);

    const glm::mat4 inv_proj = glm::inverse(cam.projection);
    const auto& offsets = cg.g.cluster_offsets();
    const auto& indices = cg.g.light_indices();

    std::uniform_real_distribution<float> ndc(-0.9f, 0.9f);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    int checked = 0;
    for (int s = 0; s < 4000; ++s) {
        const float nx = ndc(rng);
        const float ny = ndc(rng);
        const float depth = near_plane * 1.5f * std::pow(far_plane * 0.9f / (near_plane * 1.5f), unit(rng));
        const glm::vec3 frag = view_pos_from_ndc(inv_proj, nx, ny, depth);

        const glm::vec2 screen_uv(nx * 0.5f + 0.5f, 1.0f - (ny * 0.5f + 0.5f));
        const uint32_t cidx = shader_cluster_index(screen_uv, depth, cg.g.params());
        const uint32_t off = offsets[cidx * 2 + 0];
        const uint32_t cnt = offsets[cidx * 2 + 1];

        for (uint32_t j = 0; j < lights.size(); ++j) {
            const SpotLight& sl = lights[j];
            const glm::vec3 to_light = sl.position - frag;
            const float dist = glm::length(to_light);
            if (dist >= sl.radius * 0.95f) {
                continue;
            }
            // The forward loop's own predicate (light_target.hpp mirrors
            // shaders/cluster_index.hlsli): a spot lights this fragment iff the
            // cone factor is non-zero. Margin for the same FP reason the radius
            // test has one.
            const glm::vec3 l = to_light / std::max(dist, 1e-4f);
            if (spot_cone_factor(l, sl.direction, sl.inner_cos, sl.outer_cos) <= 0.0f ||
                glm::dot(sl.direction, -l) <= sl.outer_cos + 1e-3f) {
                continue;
            }
            const auto begin = indices.begin() + off;
            const bool found = std::find(begin, begin + cnt, j) != begin + cnt;
            EXPECT_TRUE(found) << "spot " << j << " lights fragment at ndc(" << nx << ", " << ny << ") depth " << depth
                               << " but the cone cull removed it from cluster " << cidx;
            ++checked;
        }
    }
    EXPECT_GT(checked, 0);
}

SpotLight make_cone(glm::vec3 pos, glm::vec3 dir, float radius, float outer_deg, float inner_deg) {
    SpotLight s{};
    s.position = pos;
    s.radius = radius;
    s.direction = glm::normalize(dir);
    s.inner_cos = std::cos(glm::radians(inner_deg));
    s.outer_cos = std::cos(glm::radians(outer_deg));
    s.color = glm::vec3(1.0f);
    s.intensity = 1.0f;
    return s;
}

} // namespace

TEST(ClusterGrid, SpotConeCullRemovesFroxelsTheConeMissesAndKeepsTheOnesItHits) {
    // A spot binned as its bounding SPHERE lands in ~12x the froxels its cone
    // can reach, and every one of those costs the fragment loop an 80-byte
    // SpotLight load plus a spotConeFactor that returns zero.
    const float near_plane = 0.1f;
    const float far_plane = 100.0f;
    const CameraState cam = make_camera(1920, 1080, 60.0f, 16.0f / 9.0f, near_plane, far_plane);

    // Probe froxel: 15 units to the RIGHT of the light, same depth.
    const glm::vec3 light_pos(0.0f, 0.0f, -20.0f);
    const float depth = 20.0f;
    // NDC of the probe point, so the shader-side lookup names the same froxel.
    const glm::vec4 clip = cam.projection * glm::vec4(15.0f, 0.0f, -depth, 1.0f);
    ASSERT_GT(clip.w, 0.0f);
    const glm::vec2 probe_ndc = glm::vec2(clip) / clip.w;
    ASSERT_LT(std::abs(probe_ndc.x), 1.0f) << "the probe must be on screen for the froxel to exist";
    const glm::vec2 probe_uv(probe_ndc.x * 0.5f + 0.5f, 1.0f - (probe_ndc.y * 0.5f + 0.5f));

    const auto count_at_probe = [&](const SpotLight& s) {
        CpuOnlyClusterGrid cg;
        cg.run(cam, 1920, 1080, near_plane, far_plane, {}, {s});
        const uint32_t cidx = shader_cluster_index(probe_uv, depth, cg.g.params());
        return cg.g.cluster_offsets()[cidx * 2 + 1];
    };

    // Same light, same 30-degree cone, three directions. Aimed at the probe it
    // must be binned there; aimed elsewhere it must not.
    EXPECT_GE(count_at_probe(make_cone(light_pos, {1.0f, 0.0f, 0.0f}, 20.0f, 30.0f, 18.0f)), 1u)
        << "a spot whose cone contains the froxel must still be binned into it";
    EXPECT_EQ(count_at_probe(make_cone(light_pos, {-1.0f, 0.0f, 0.0f}, 20.0f, 30.0f, 18.0f)), 0u)
        << "a spot aimed away from the froxel contributes exactly zero there and must be culled";
    EXPECT_EQ(count_at_probe(make_cone(light_pos, {0.0f, -1.0f, 0.0f}, 20.0f, 30.0f, 18.0f)), 0u)
        << "...and so must one aimed perpendicular to it";

    // The cull only fires on a WELL-FORMED cone. A default-constructed
    // SpotLight carries inner_cos == outer_cos == 0 (a 90-degree cutoff with no
    // falloff band), which has no usable cone test: it must bin exactly as it
    // did before the cull existed, so no pre-existing test can move.
    SpotLight defaults{};
    defaults.position = light_pos;
    defaults.radius = 20.0f;
    defaults.direction = glm::vec3(-1.0f, 0.0f, 0.0f);
    defaults.color = glm::vec3(1.0f);
    defaults.intensity = 1.0f;
    EXPECT_EQ(defaults.outer_cos, 0.0f);
    EXPECT_GE(count_at_probe(defaults), 1u) << "a default-cone spot must bin exactly as the bounding sphere did";
}

TEST(ClusterGrid, TransposedSpotConeFallsBackToSphereBinning) {
    // FrameBuilder::Lane::add_spot_light normalizes a transposed pair, but a
    // SpotLight reaching the grid another way might carry one. Its lit set is
    // then the COMPLEMENT of the cone (the shader's smoothstep runs with
    // edge0 > edge1), so culling to the cone would remove froxels the forward
    // loop lights. The binner must notice and bin the sphere instead.
    const float near_plane = 0.1f;
    const float far_plane = 100.0f;
    const CameraState cam = make_camera(1920, 1080, 60.0f, 16.0f / 9.0f, near_plane, far_plane);
    const float depth = 20.0f;

    SpotLight inverted = make_cone({0.0f, 0.0f, -20.0f}, {-1.0f, 0.0f, 0.0f}, 20.0f, 20.0f, 12.0f);
    std::swap(inverted.inner_cos, inverted.outer_cos); // inner 20 deg, outer 12 deg

    CpuOnlyClusterGrid cg;
    cg.run(cam, 1920, 1080, near_plane, far_plane, {}, {inverted});
    const glm::vec4 clip = cam.projection * glm::vec4(15.0f, 0.0f, -depth, 1.0f);
    const glm::vec2 probe_ndc = glm::vec2(clip) / clip.w;
    const glm::vec2 probe_uv(probe_ndc.x * 0.5f + 0.5f, 1.0f - (probe_ndc.y * 0.5f + 0.5f));
    const uint32_t cidx = shader_cluster_index(probe_uv, depth, cg.g.params());
    EXPECT_GE(cg.g.cluster_offsets()[cidx * 2 + 1], 1u);
}

TEST(ClusterGrid, SpotBinsCoverEveryFragmentInsideTheCone) {
    const float near_plane = 0.1f;
    const float far_plane = 100.0f;
    const CameraState cam = make_camera(1920, 1080, 60.0f, 16.0f / 9.0f, near_plane, far_plane);

    std::mt19937 rng(31337);
    std::uniform_real_distribution<float> px(-25.0f, 25.0f);
    std::uniform_real_distribution<float> py(-12.0f, 12.0f);
    std::uniform_real_distribution<float> pz(-70.0f, -3.0f);
    std::uniform_real_distribution<float> pr(6.0f, 30.0f);
    std::uniform_real_distribution<float> pd(-1.0f, 1.0f);
    std::uniform_real_distribution<float> outer(10.0f, 75.0f);

    std::vector<SpotLight> lights;
    for (int i = 0; i < 48; ++i) {
        glm::vec3 dir(pd(rng), pd(rng), pd(rng));
        if (glm::dot(dir, dir) < 1e-4f) {
            dir = glm::vec3(0.0f, -1.0f, 0.0f);
        }
        const float outer_deg = outer(rng);
        lights.push_back(make_cone(glm::vec3(px(rng), py(rng), pz(rng)), dir, pr(rng), outer_deg, outer_deg * 0.6f));
    }
    check_spot_light_coverage(cam, near_plane, far_plane, lights);
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
        for (const auto& p : points) (void)g.add_point_light(p);
        for (const auto& s : spots) (void)g.add_spot_light(s);
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
    // cutoff: the parallel per-cluster form must keep exactly the same lights
    // as the inline path. Identical lights all score identically at any froxel
    // and the eviction rule is STRICTLY greater, so "the same" is still "the
    // lowest indices" here — which is the point of using identical lights.
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
    // 150 identical overlapping lights: every touched cluster must stop at
    // MAX_LIGHTS_PER_CLUSTER. Identical lights tie on contribution and the
    // eviction is strictly-greater, so the survivors are the lowest indices.
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

// ===========================================================================
// Over-budget eviction policy (MAX_LIGHTS_PER_CLUSTER)
// ===========================================================================

TEST(ClusterGrid, OverBudgetBinKeepsTheBrightestLightsNotTheEarliest) {
    // Submission order is uncorrelated with how much a light matters, so
    // arrival-order truncation can drop the brightest light in the frame. With
    // 150 coincident lights of strictly increasing intensity the only correct
    // survivors are the top MAX_LIGHTS_PER_CLUSTER by intensity — everything
    // else about them is equal, so contribution reduces to brightness here.
    std::vector<PointLight> lights;
    for (uint32_t i = 0; i < 150; ++i) {
        PointLight p{};
        p.position = glm::vec3(0.0f, 0.0f, -10.0f);
        p.radius = 3.0f;
        p.color = glm::vec3(1.0f);
        p.intensity = 1.0f + static_cast<float>(i); // strictly increasing
        lights.push_back(p);
    }

    CpuOnlyClusterGrid cg;
    cg.run(make_camera(1920, 1080), 1920, 1080, 0.1f, 100.0f, lights, {});

    const auto& offsets = cg.g.cluster_offsets();
    const auto& indices = cg.g.light_indices();
    const uint32_t dropped = 150u - MAX_LIGHTS_PER_CLUSTER; // 22
    int full_bins = 0;
    for (uint32_t c = 0; c < CLUSTER_COUNT; ++c) {
        const uint32_t cnt = offsets[c * 2 + 1];
        if (cnt != MAX_LIGHTS_PER_CLUSTER) {
            continue;
        }
        ++full_bins;
        const uint32_t off = offsets[c * 2 + 0];
        std::set<uint32_t> kept(indices.begin() + off, indices.begin() + off + cnt);
        EXPECT_EQ(kept.size(), MAX_LIGHTS_PER_CLUSTER) << "eviction must not duplicate a light into a bin";
        EXPECT_EQ(*kept.begin(), dropped) << "the " << dropped << " dimmest lights are the ones that lose their slot";
        EXPECT_EQ(*kept.rbegin(), 149u) << "the brightest light in the frame must never be the one dropped";
    }
    EXPECT_GT(full_bins, 0) << "the test scene must actually overflow a bin";
    EXPECT_GT(cg.g.cluster_lights_dropped(), 0u) << "refused insertions are still counted";
}

TEST(ClusterGrid, EvictionRanksByContributionHereNotByTotalPower) {
    // The metric has to be per-FROXEL, not a scene-wide power proxy: what a
    // froxel loses when a light is dropped is that light's contribution THERE.
    // 128 distant floodlights (100x the intensity, 12x the radius) fill the
    // bin, and then a dim torch arrives sitting inside the froxel. Total power
    // ranks the floodlights far above the torch; contribution at this froxel
    // ranks the torch four orders of magnitude above them, and the torch is the
    // one the player would notice going out.
    const float depth = 11.0f;
    const CameraState cam = make_camera(1920, 1080);

    std::vector<PointLight> lights;
    for (int i = 0; i < static_cast<int>(MAX_LIGHTS_PER_CLUSTER); ++i) {
        PointLight flood{};
        flood.position = glm::vec3(0.0f, 0.0f, -60.0f);
        flood.radius = 60.0f;
        flood.color = glm::vec3(1.0f);
        flood.intensity = 100.0f;
        lights.push_back(flood);
    }
    PointLight torch{};
    torch.position = glm::vec3(0.0f, 0.0f, -depth);
    torch.radius = 5.0f;
    torch.color = glm::vec3(1.0f);
    torch.intensity = 1.0f;
    lights.push_back(torch); // arrives LAST — arrival order drops it outright
    const uint32_t torch_index = static_cast<uint32_t>(lights.size() - 1);

    CpuOnlyClusterGrid cg;
    cg.run(cam, 1920, 1080, 0.1f, 100.0f, lights, {});

    // The froxel the torch sits in, named through the shader-side lookup.
    const glm::vec2 probe_uv(0.5f, 0.5f);
    const uint32_t cidx = shader_cluster_index(probe_uv, depth, cg.g.params());
    const auto& offsets = cg.g.cluster_offsets();
    const uint32_t off = offsets[cidx * 2 + 0];
    const uint32_t cnt = offsets[cidx * 2 + 1];
    ASSERT_EQ(cnt, MAX_LIGHTS_PER_CLUSTER) << "the froxel must be over budget for eviction to run at all";
    const auto& indices = cg.g.light_indices();
    const auto begin = indices.begin() + off;
    EXPECT_NE(std::find(begin, begin + cnt, torch_index), begin + cnt)
        << "a near dim light must outrank a far bright one at the froxel it is standing in";
}

// ===========================================================================
// Froxel index parity: the shared math vs. the binner's own cluster bounds
// ===========================================================================

namespace {

// The Z-slice boundary formula light_cluster.cpp bins against
// (`exp_z_slice`), restated from the documented exponential distribution
// rather than borrowed from the implementation — a mirror that copied the
// code would agree with a typo.
float slice_depth(uint32_t z, float near_z, float far_z) {
    return near_z * std::pow(far_z / near_z, static_cast<float>(z) / static_cast<float>(CLUSTER_DIM_Z));
}

} // namespace

// The refactored index math (render/lighting/cluster_index.hpp — the CPU
// mirror of shaders/cluster_index.hlsli, which BOTH the fragment path and the
// new compute path include) must land on exactly the cluster the binner's
// AABB path binned a light into, everywhere in the grid. This is the guarantee
// that lets a compute pass read a pixel's froxel light list and get the same
// answer the forward pass gets for the same pixel.
TEST(ClusterGrid, FroxelIndexMathAgreesWithBinnerAcrossTheGrid) {
    const float near_plane = 0.1f;
    const float far_plane = 100.0f;
    CameraState cam = make_camera(1920, 1080, 60.0f, 16.0f / 9.0f, near_plane, far_plane);
    const glm::mat4 inv_proj = glm::inverse(cam.projection);

    // Sweep the whole grid, strided so one pass stays under MAX_LIGHTS; every
    // cluster index in the sweep is checked in both directions.
    struct Probe {
        uint32_t expected_cluster;
        glm::vec2 screen_uv;
        float depth;
        glm::vec3 view_pos;
    };
    std::vector<Probe> probes;
    for (uint32_t z = 0; z < CLUSTER_DIM_Z; ++z) {
        for (uint32_t y = 0; y < CLUSTER_DIM_Y; ++y) {
            for (uint32_t x = 0; x < CLUSTER_DIM_X; ++x) {
                // Tile center in NDC (cx = 0 starts at NDC -1, per
                // unproject_corner) and the GEOMETRIC mean depth of the slice
                // — the exact midpoint of an exponentially distributed slice.
                const float ndc_x = ((static_cast<float>(x) + 0.5f) / CLUSTER_DIM_X) * 2.0f - 1.0f;
                const float ndc_y = ((static_cast<float>(y) + 0.5f) / CLUSTER_DIM_Y) * 2.0f - 1.0f;
                const float depth =
                    std::sqrt(slice_depth(z, near_plane, far_plane) * slice_depth(z + 1, near_plane, far_plane));
                Probe p;
                p.expected_cluster = (z * CLUSTER_DIM_Y + y) * CLUSTER_DIM_X + x;
                // SV_Position-style uv: y = 0 is the TOP of the screen.
                p.screen_uv = glm::vec2(ndc_x * 0.5f + 0.5f, 1.0f - (ndc_y * 0.5f + 0.5f));
                p.depth = depth;
                p.view_pos = view_pos_from_ndc(inv_proj, ndc_x, ndc_y, depth);
                probes.push_back(p);
            }
        }
    }
    ASSERT_EQ(probes.size(), CLUSTER_COUNT);

    CpuOnlyClusterGrid cg;
    constexpr std::size_t BATCH = 128; // well under MAX_LIGHTS
    for (std::size_t base = 0; base < probes.size(); base += BATCH) {
        const std::size_t end = std::min(base + BATCH, probes.size());
        std::vector<PointLight> lights;
        for (std::size_t i = base; i < end; ++i) {
            PointLight p{};
            p.position = probes[i].view_pos; // identity view => view == world
            // Small but not degenerate: it must reach the cluster's AABB,
            // which is the conservative box around the frustum cell.
            p.radius = 0.05f;
            p.color = glm::vec3(1.0f);
            p.intensity = 1.0f;
            lights.push_back(p);
        }
        cg.run(cam, 1920, 1080, near_plane, far_plane, lights, {});

        const auto& offsets = cg.g.cluster_offsets();
        const auto& indices = cg.g.light_indices();
        for (std::size_t i = base; i < end; ++i) {
            const Probe& probe = probes[i];
            SCOPED_TRACE(::testing::Message() << "cluster " << probe.expected_cluster);
            // (a) the pure math: shared index function -> the intended cluster.
            const uint32_t cidx = cluster_index_from_screen(probe.screen_uv, probe.depth, cg.g.params());
            ASSERT_EQ(cidx, probe.expected_cluster);
            // (b) parity with the binner's independent AABB path: the light
            // placed at that point is in THAT cluster's bin.
            const uint32_t offset = offsets[cidx * 2 + 0];
            const uint32_t count = offsets[cidx * 2 + 1];
            const uint32_t light_index = static_cast<uint32_t>(i - base);
            bool found = false;
            for (uint32_t k = 0; k < count; ++k) {
                if (indices[offset + k] == light_index) {
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found) << "binner and froxel index math disagree about this cluster";
        }
    }
}

// The Z-slice half on its own, at the boundaries the geometric-mean sweep
// above deliberately stays away from.
TEST(ClusterGrid, ZSliceMathCoversTheWholeRangeMonotonically) {
    CpuOnlyClusterGrid cg;
    cg.run(make_camera(1920, 1080, 60.0f, 16.0f / 9.0f, 0.1f, 100.0f), 1920, 1080, 0.1f, 100.0f, {}, {});
    const ClusterParams& cp = cg.g.params();

    // Clamping at both ends: behind the near plane and past the far plane.
    EXPECT_EQ(cluster_z_slice(-5.0f, cp), 0u);
    EXPECT_EQ(cluster_z_slice(0.0f, cp), 0u);
    EXPECT_EQ(cluster_z_slice(1e6f, cp), CLUSTER_DIM_Z - 1);

    uint32_t previous = 0;
    for (uint32_t z = 0; z < CLUSTER_DIM_Z; ++z) {
        const float d = std::sqrt(slice_depth(z, 0.1f, 100.0f) * slice_depth(z + 1, 0.1f, 100.0f));
        const uint32_t got = cluster_z_slice(d, cp);
        EXPECT_EQ(got, z);
        EXPECT_GE(got, previous);
        previous = got;
    }
}

// ===========================================================================
// Persistent light identity (contracts/light.hpp) and the prev->current remap
// ===========================================================================

namespace {

PointLight make_point(glm::vec3 pos, glm::vec3 color = glm::vec3(1.0f), float intensity = 1.0f) {
    PointLight p{};
    p.position = pos;
    p.radius = 3.0f;
    p.color = color;
    p.intensity = intensity;
    return p;
}

SpotLight make_spot(glm::vec3 pos, glm::vec3 color = glm::vec3(1.0f), float intensity = 1.0f) {
    SpotLight s{};
    s.position = pos;
    s.radius = 3.0f;
    s.direction = glm::vec3(0.0f, -1.0f, 0.0f);
    s.inner_cos = 0.9f;
    s.outer_cos = 0.7f;
    s.color = color;
    s.intensity = intensity;
    return s;
}

// One frame through the CPU pipeline; returns the resolved stable ids.
std::vector<uint32_t> run_frame(CpuOnlyClusterGrid& cg, const std::vector<PointLight>& points,
                                const std::vector<SpotLight>& spots = {}) {
    cg.run(make_camera(1920, 1080), 1920, 1080, 0.1f, 100.0f, points, spots);
    return cg.g.stable_ids();
}

} // namespace

TEST(LightIdentity, SameSceneEmittedTwiceGetsIdenticalIds) {
    const std::vector<PointLight> scene = {make_point({0.0f, 1.0f, -10.0f}), make_point({12.0f, 2.0f, -30.0f}),
                                           make_point({-20.0f, 0.5f, -45.0f}, glm::vec3(1.0f, 0.4f, 0.1f), 8.0f)};
    CpuOnlyClusterGrid cg;
    const std::vector<uint32_t> first = run_frame(cg, scene);
    const std::vector<uint32_t> second = run_frame(cg, scene);
    ASSERT_EQ(first.size(), 3u);
    EXPECT_EQ(first, second);
    for (uint32_t id : first) {
        EXPECT_NE(id, 0u) << "0 is the 'engine assigns' sentinel and must never survive assignment";
    }

    // ...and a second grid, from scratch, agrees: assignment is a pure
    // function of the emission sequence, not of accumulated state.
    CpuOnlyClusterGrid fresh;
    EXPECT_EQ(run_frame(fresh, scene), first);
}

TEST(LightIdentity, ReorderedEmissionOfDistinctLightsMovesIdsWithTheLights) {
    const PointLight a = make_point({0.0f, 1.0f, -10.0f});
    const PointLight b = make_point({40.0f, 2.0f, -30.0f});
    const PointLight c = make_point({-40.0f, 3.0f, -60.0f});

    CpuOnlyClusterGrid cg;
    const std::vector<uint32_t> forward = run_frame(cg, {a, b, c});
    const std::vector<uint32_t> reversed = run_frame(cg, {c, b, a});
    ASSERT_EQ(forward.size(), 3u);
    ASSERT_EQ(reversed.size(), 3u);
    // Identity follows the LIGHT, not the slot.
    EXPECT_EQ(reversed[0], forward[2]);
    EXPECT_EQ(reversed[1], forward[1]);
    EXPECT_EQ(reversed[2], forward[0]);

    // And the remap says exactly that: prev slot 0 (light a) is now slot 2.
    const auto& remap = cg.g.index_remap();
    ASSERT_EQ(remap.size(), MAX_LIGHTS);
    EXPECT_EQ(remap[0], 2u);
    EXPECT_EQ(remap[1], 1u);
    EXPECT_EQ(remap[2], 0u);
    for (uint32_t i = 3; i < MAX_LIGHTS; ++i) {
        EXPECT_EQ(remap[i], LIGHT_REMAP_INVALID);
    }
}

TEST(LightIdentity, SurvivorsKeepIdsAndTheRemapNamesTheVanished) {
    const PointLight a = make_point({0.0f, 1.0f, -10.0f});
    const PointLight b = make_point({40.0f, 2.0f, -30.0f});
    const PointLight c = make_point({-40.0f, 3.0f, -60.0f});
    const PointLight d = make_point({0.0f, 30.0f, -80.0f});

    CpuOnlyClusterGrid cg;
    const std::vector<uint32_t> before = run_frame(cg, {a, b, c});
    // b is removed, d appears; a and c survive and shift down a slot.
    const std::vector<uint32_t> after = run_frame(cg, {a, c, d});
    ASSERT_EQ(after.size(), 3u);
    EXPECT_EQ(after[0], before[0]); // a
    EXPECT_EQ(after[1], before[2]); // c
    EXPECT_NE(after[2], before[0]);
    EXPECT_NE(after[2], before[1]);
    EXPECT_NE(after[2], before[2]); // d is new

    const auto& remap = cg.g.index_remap();
    EXPECT_EQ(remap[0], 0u);                    // a: slot 0 -> slot 0
    EXPECT_EQ(remap[1], LIGHT_REMAP_INVALID);   // b: gone
    EXPECT_EQ(remap[2], 1u);                    // c: slot 2 -> slot 1
    EXPECT_EQ(remap[3], LIGHT_REMAP_INVALID);   // nothing was there last frame
}

TEST(LightIdentity, FirstFrameAndResetHistoryRemapToNothing) {
    CpuOnlyClusterGrid cg;
    (void)run_frame(cg, {make_point({0.0f, 1.0f, -10.0f})});
    // Nothing preceded frame 1, so no previous index maps anywhere.
    for (uint32_t id : cg.g.index_remap()) {
        EXPECT_EQ(id, LIGHT_REMAP_INVALID);
    }

    (void)run_frame(cg, {make_point({0.0f, 1.0f, -10.0f})});
    EXPECT_EQ(cg.g.index_remap()[0], 0u) << "a static light must remap to itself";

    // The renderer calls this on any frame it skips binning; the next build
    // must then behave like a first frame rather than reach across the gap.
    cg.g.reset_light_history();
    for (uint32_t id : cg.g.index_remap()) {
        EXPECT_EQ(id, LIGHT_REMAP_INVALID);
    }
    (void)run_frame(cg, {make_point({0.0f, 1.0f, -10.0f})});
    for (uint32_t id : cg.g.index_remap()) {
        EXPECT_EQ(id, LIGHT_REMAP_INVALID);
    }
}

TEST(LightIdentity, MovementBelowTheQuantumKeepsTheIdAndATeleportDoesNot) {
    // Base sits at a quantum CENTER (10 = 2 * 5) so the sub-quantum walk below
    // cannot straddle a rounding boundary.
    const glm::vec3 base(10.0f, 10.0f, -10.0f);
    CpuOnlyClusterGrid cg;
    const uint32_t id = run_frame(cg, {make_point(base)}).front();

    // Jitter well inside the quantum, in every axis: same identity.
    for (float d : {0.5f, -0.5f, 1.9f, -1.9f}) {
        const std::vector<uint32_t> ids = run_frame(cg, {make_point(base + glm::vec3(d, -d, d))});
        EXPECT_EQ(ids.front(), id) << "sub-quantum motion of " << d << " must not change identity";
        EXPECT_EQ(cg.g.index_remap()[0], 0u);
    }

    // A teleport past the quantum is a different light as far as temporal
    // history is concerned, and the remap says so: nothing carries over.
    const std::vector<uint32_t> moved =
        run_frame(cg, {make_point(base + glm::vec3(4.0f * LIGHT_ID_POSITION_QUANTUM, 0.0f, 0.0f))});
    EXPECT_NE(moved.front(), id);
    EXPECT_EQ(cg.g.index_remap()[0], LIGHT_REMAP_INVALID);
}

TEST(LightIdentity, IntensityFlickerBelowAnOctaveKeepsTheId) {
    const glm::vec3 pos(10.0f, 10.0f, -10.0f);
    CpuOnlyClusterGrid cg;
    const uint32_t id = run_frame(cg, {make_point(pos, glm::vec3(1.0f), 4.0f)}).front();
    // A torch flickering +-20% stays in its octave — the reason intensity is
    // quantized in log2 rather than by an absolute step.
    for (float k : {1.1f, 0.9f, 1.18f, 0.85f}) {
        EXPECT_EQ(run_frame(cg, {make_point(pos, glm::vec3(1.0f), 4.0f * k)}).front(), id);
    }
    // Two octaves up is a different light.
    EXPECT_NE(run_frame(cg, {make_point(pos, glm::vec3(1.0f), 16.0f)}).front(), id);
}

TEST(LightIdentity, IdenticalLightsGetDistinctDeterministicIds) {
    // Four lanterns with byte-identical content: the hash collides four ways
    // and the occurrence counter has to break it, the same way every run.
    const PointLight lantern = make_point({10.0f, 10.0f, -10.0f});
    const std::vector<PointLight> four(4, lantern);

    CpuOnlyClusterGrid cg;
    const std::vector<uint32_t> ids = run_frame(cg, four);
    ASSERT_EQ(ids.size(), 4u);
    std::set<uint32_t> unique(ids.begin(), ids.end());
    EXPECT_EQ(unique.size(), 4u) << "identical lights must still get distinct identities";
    EXPECT_EQ(unique.count(0u), 0u);

    // Deterministic: same sequence, same ids, on a fresh grid.
    CpuOnlyClusterGrid fresh;
    EXPECT_EQ(run_frame(fresh, four), ids);
    // And stable frame over frame, so their reservoirs survive.
    EXPECT_EQ(run_frame(cg, four), ids);
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(cg.g.index_remap()[i], i);
    }
}

TEST(LightIdentity, HostSuppliedIdsPassThroughUntouchedAndWinOverHashing) {
    PointLight tagged = make_point({10.0f, 10.0f, -10.0f});
    tagged.stable_id = 0xC0FFEEu;
    PointLight untagged = make_point({10.0f, 10.0f, -10.0f}); // identical content

    CpuOnlyClusterGrid cg;
    const std::vector<uint32_t> ids = run_frame(cg, {tagged, untagged});
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 0xC0FFEEu) << "a host-supplied id must survive verbatim";
    EXPECT_NE(ids[1], 0xC0FFEEu);

    // Order must not matter for the host's id: emit it second and it still
    // wins, with the auto-assigned light forced to move.
    const std::vector<uint32_t> swapped = run_frame(cg, {untagged, tagged});
    EXPECT_EQ(swapped[1], 0xC0FFEEu);
    EXPECT_NE(swapped[0], 0xC0FFEEu);

    // A host id that teleports keeps its identity — that is the whole point of
    // supplying one.
    PointLight moved = make_point({80.0f, 4.0f, -70.0f});
    moved.stable_id = 0xC0FFEEu;
    const std::vector<uint32_t> after = run_frame(cg, {moved});
    EXPECT_EQ(after.front(), 0xC0FFEEu);
    EXPECT_EQ(cg.g.index_remap()[1], 0u) << "prev slot 1 held the host-tagged light; it is slot 0 now";
}

TEST(LightIdentity, PointsAndSpotsWithMatchingContentAreDifferentLights) {
    const glm::vec3 pos(10.0f, 10.0f, -10.0f);
    CpuOnlyClusterGrid cg;
    cg.run(make_camera(1920, 1080), 1920, 1080, 0.1f, 100.0f, {make_point(pos)}, {make_spot(pos)});
    const auto& ids = cg.g.stable_ids();
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_NE(ids[0], ids[1]) << "light TYPE is part of the identity hash";
}

TEST(LightIdentity, ResolvedIdsAreWrittenBackIntoTheUploadedLightData) {
    // The parallel GPU buffer is the fast path; the light records carry the
    // same value so "who is this light" is answerable from the light data too.
    CpuOnlyClusterGrid cg;
    cg.run(make_camera(1920, 1080), 1920, 1080, 0.1f, 100.0f,
           {make_point({10.0f, 10.0f, -10.0f}), make_point({40.0f, 2.0f, -30.0f})},
           {make_spot({-20.0f, 5.0f, -25.0f})});
    const auto& ids = cg.g.stable_ids();
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(cg.g.point_lights()[0].stable_id, ids[0]);
    EXPECT_EQ(cg.g.point_lights()[1].stable_id, ids[1]);
    EXPECT_EQ(cg.g.spot_lights()[0].stable_id, ids[2]);
}

TEST(LightIdentity, RemapIsSizedToTheFrameCapAndStaysInBounds) {
    std::vector<PointLight> many;
    for (uint32_t i = 0; i < MAX_LIGHTS; ++i) {
        many.push_back(make_point({static_cast<float>(i) * 7.0f, 1.0f, -20.0f}));
    }
    CpuOnlyClusterGrid cg;
    const std::vector<uint32_t> ids = run_frame(cg, many);
    EXPECT_EQ(ids.size(), MAX_LIGHTS);
    EXPECT_EQ(std::set<uint32_t>(ids.begin(), ids.end()).size(), MAX_LIGHTS);

    (void)run_frame(cg, many);
    const auto& remap = cg.g.index_remap();
    ASSERT_EQ(remap.size(), MAX_LIGHTS);
    for (uint32_t i = 0; i < MAX_LIGHTS; ++i) {
        ASSERT_NE(remap[i], LIGHT_REMAP_INVALID);
        EXPECT_LT(remap[i], MAX_LIGHTS);
        EXPECT_EQ(remap[i], i) << "an unchanged light list must remap to itself";
    }
}
