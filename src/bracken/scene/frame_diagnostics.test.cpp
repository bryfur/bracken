// Phase 4: every drop path names itself.
//
// The four failures below used to be indistinguishable from "nothing to
// draw": a `return;` at record time, a `continue;` in a prep chunk, an early
// return in the cluster grid. Each test drives the real production path (no
// mocks, no test-only branches) and asserts the corresponding counter moved
// by exactly the expected amount — the counters are what a host reads through
// Application::frame_diagnostics(), and what the smoke run prints.
//
// Everything here is device-free: the prep cores are static on ScenePrep
// (see scene_prep_parity.test.cpp) and ClusterGrid::begin_frame/add_*/build
// touch no GPU buffers, so the whole drop surface is testable headless.

#include <gtest/gtest.h>

#include "bracken/contracts/frustum.hpp"
#include "bracken/contracts/render_stats.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/render/lighting/light_cluster.hpp"
#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/motion_history.hpp"
#include "bracken/scene/scene_prep.hpp"
#include "bracken/scene/scene_renderer.hpp"
#include "bracken/testing/frame_test.hpp"

#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <string>
#include <vector>

using bracken::testing::FrameAccess;
using namespace bracken;
using namespace bracken::scene;
namespace jobs = bracken::core::jobs;

namespace {

// register_model refuses a model with no geometry, so the synthetic models
// below carry one throwaway triangle. Prep works from bounds and handles, not
// mesh contents, so it changes nothing these tests measure.
void add_token_mesh(Model& m) {
    Mesh mesh;
    mesh.vertices.resize(3);
    mesh.indices = {0, 1, 2};
    m.meshes.push_back(std::move(mesh));
}

std::unique_ptr<Model> make_model(bool with_skeleton, uint32_t joints = 0) {
    auto m = std::make_unique<Model>();
    add_token_mesh(*m);
    m->bounds.min.x = m->bounds.min.y = m->bounds.min.z = -1.0f;
    m->bounds.max.x = m->bounds.max.y = m->bounds.max.z = 1.0f;
    m->compute_bounding_sphere();
    if (with_skeleton) {
        m->has_skeleton = true;
        m->skeleton.joints.resize(joints);
    }
    return m;
}

// A prep environment with culling disabled, so nothing is dropped for a
// reason other than the one under test.
struct DiagWorld {
    ModelManager models;
    FramePacket packet{1 << 20};
    MotionHistory history;
    Frustum main_frustum{};
    std::array<Frustum, ScenePrep::MAX_SHADOW_CASCADES> cascade_frustums{};

    ScenePrep::PrepParams params() const {
        ScenePrep::PrepParams p;
        p.models = &models;
        p.motion_history = &history;
        p.camera_pos = glm::vec3(0.0f);
        p.main_frustum = &main_frustum;
        p.cascade_frustums = cascade_frustums.data();
        p.active_cascades = 0;
        p.draw_dist_sq = 1.0e12f; // nothing distance-culled
        p.do_frustum_cull = false;
        p.collect_stats = false; // diagnostics must NOT depend on this
        p.prev_palette_valid = true;
        return p;
    }
};

FrameStatsPartial run_model_prep(DiagWorld& w) {
    jobs::JobSystem js; // uninitialized: serial inline path
    ScenePrep::PrepScratch scratch;
    std::vector<std::vector<gpu::InstanceData>> bins;
    std::vector<ModelHandle> touched;
    std::array<std::vector<std::vector<gpu::ShadowInstanceData>>, ScenePrep::MAX_SHADOW_CASCADES> shadow_bins;
    std::array<std::vector<ModelHandle>, ScenePrep::MAX_SHADOW_CASCADES> shadow_touched;
    std::vector<const ModelDrawCommand*> non_instanced;
    FrameStatsPartial stats;
    const ScenePrep::InstanceBinOutputs out{bins, touched, shadow_bins, shadow_touched, non_instanced, stats};
    ScenePrep::prep_instance_batches(js, w.packet.model_commands(), w.params(), scratch, out);
    return stats;
}

FrameStatsPartial run_skinned_prep(DiagWorld& w) {
    jobs::JobSystem js;
    ScenePrep::PrepScratch scratch;
    std::vector<ScenePrep::ResolvedSkinnedEntry> resolved;
    FrameStatsPartial stats;
    ScenePrep::prep_resolve_skinned(js, w.packet.skinned_commands(), w.packet.palettes(), w.params(),
                                        scratch.skinned, resolved, stats);
    return stats;
}

// ---------------------------------------------------------------------------
// (a) a draw with an unregistered ModelHandle
// ---------------------------------------------------------------------------

TEST(FrameDiagnosticsTest, UnknownModelHandleIsCountedOncePerDraw) {
    DiagWorld w;
    const ModelHandle good = w.models.register_model("good", make_model(false)).value_or(0);
    const ModelHandle bogus = static_cast<ModelHandle>(9999); // never registered

    FrameBuilder fb(w.packet);
    fb.draw_model(good, glm::mat4(1.0f));
    fb.draw_model(bogus, glm::mat4(1.0f));
    fb.draw_model(bogus, glm::translate(glm::mat4(1.0f), glm::vec3(1.0f)));
    FrameAccess::seal(fb);

    // The two bogus draws SURVIVE the packet (an out-of-range handle is not
    // INVALID_MODEL_HANDLE, so the builder cannot know) and die at prep.
    ASSERT_EQ(w.packet.model_commands().size(), 3u);

    const FrameStatsPartial stats = run_model_prep(w);
    EXPECT_EQ(stats.diagnostics.draws_dropped_unknown_model, 2u);
    // Diagnostics are unconditional; stats (collect_stats=false) are not.
    EXPECT_EQ(stats.entities_rendered, 0u);
}

TEST(FrameDiagnosticsTest, InvalidHandleIsRefusedAtEmitTime) {
    DiagWorld w;
    FrameBuilder fb(w.packet);
    fb.draw_model(INVALID_MODEL_HANDLE, glm::mat4(1.0f));
    fb.draw_model(INVALID_MODEL_HANDLE, glm::mat4(1.0f));
    FrameAccess::seal(fb);

    // Never became a command: named at the frame that emitted it, not three
    // stages downstream.
    EXPECT_TRUE(w.packet.model_commands().empty());
    EXPECT_EQ(w.packet.build_diagnostics().draws_dropped_invalid_handle, 2u);
    EXPECT_TRUE(w.packet.build_diagnostics().any());
}

TEST(FrameDiagnosticsTest, CleanFrameCountsNothing) {
    DiagWorld w;
    const ModelHandle good = w.models.register_model("good", make_model(false)).value_or(0);
    FrameBuilder fb(w.packet);
    for (int i = 0; i < 8; ++i) {
        fb.draw_model(good, glm::mat4(1.0f));
    }
    FrameAccess::seal(fb);

    EXPECT_FALSE(w.packet.build_diagnostics().any());
    EXPECT_FALSE(run_model_prep(w).diagnostics.any());
}

// ---------------------------------------------------------------------------
// (b) a skinned draw against a skeleton-less model
// ---------------------------------------------------------------------------

TEST(FrameDiagnosticsTest, SkinnedDrawWithNoSkeletonIsCounted) {
    DiagWorld w;
    const ModelHandle rigged = w.models.register_model("rigged", make_model(true, 8)).value_or(0);
    const ModelHandle rigid = w.models.register_model("rigid", make_model(false)).value_or(0);

    FrameBuilder fb(w.packet);
    for (const ModelHandle h : {rigged, rigid, rigid}) {
        auto alloc = fb.alloc_bone_palette(8);
        ASSERT_TRUE(alloc.valid());
        for (glm::mat4& m : alloc.matrices) {
            m = glm::mat4(1.0f);
        }
        fb.draw_skinned(h, glm::mat4(1.0f), alloc.id);
    }
    FrameAccess::seal(fb);

    const FrameStatsPartial stats = run_skinned_prep(w);
    EXPECT_EQ(stats.diagnostics.skinned_dropped_no_skeleton, 2u);
}

TEST(FrameDiagnosticsTest, PaletteBudgetOverrunIsCountedAndSummarized) {
    DiagWorld w;
    const ModelHandle rigged = w.models.register_model("rigged", make_model(true, 4)).value_or(0);

    FrameBuilder fb(w.packet);
    constexpr uint32_t k_over = 5;
    for (uint32_t i = 0; i < FramePacket::MAX_BONE_PALETTES + k_over; ++i) {
        auto alloc = fb.alloc_bone_palette(4);
        if (alloc.valid()) {
            for (glm::mat4& m : alloc.matrices) {
                m = glm::mat4(1.0f);
            }
        }
        // The over-budget ids are invalid, so these draws are refused too.
        fb.draw_skinned(rigged, glm::mat4(1.0f), alloc.id);
    }
    FrameAccess::seal(fb);

    const auto build_diag = w.packet.build_diagnostics();
    EXPECT_EQ(build_diag.palettes_over_budget, k_over);
    EXPECT_EQ(build_diag.skinned_dropped_invalid_palette, k_over);
    EXPECT_EQ(w.packet.palettes().size(), FramePacket::MAX_BONE_PALETTES);
    EXPECT_EQ(w.packet.skinned_commands().size(), FramePacket::MAX_BONE_PALETTES);
}

// ---------------------------------------------------------------------------
// (c) 3D content with no camera
// ---------------------------------------------------------------------------

TEST(FrameDiagnosticsTest, ThreeDContentWithoutACameraIsDetected) {
    DiagWorld w;
    const ModelHandle good = w.models.register_model("good", make_model(false)).value_or(0);
    FrameBuilder fb(w.packet);
    fb.draw_model(good, glm::mat4(1.0f));
    FrameAccess::seal(fb);

    ASSERT_TRUE(w.packet.has_3d_content());
    EXPECT_TRUE(SceneRenderer::frame_needs_camera(w.packet, /*have_camera=*/false));
    // A camera set on an EARLIER frame still counts: the renderer keeps it.
    EXPECT_FALSE(SceneRenderer::frame_needs_camera(w.packet, /*have_camera=*/true));
}

TEST(FrameDiagnosticsTest, EmptyFrameWithoutACameraIsNotAProblem) {
    DiagWorld w;
    FrameBuilder fb(w.packet);
    fb.set_draw_skybox(false);
    fb.set_draw_ground(false);
    fb.set_draw_grass(false);
    FrameAccess::seal(fb);

    ASSERT_FALSE(w.packet.has_3d_content());
    EXPECT_FALSE(SceneRenderer::frame_needs_camera(w.packet, /*have_camera=*/false));
}

// ---------------------------------------------------------------------------
// (d) 300 lights against MAX_LIGHTS = 256
// ---------------------------------------------------------------------------

TEST(FrameDiagnosticsTest, LightsPastTheCapAreRefusedAndCounted) {
    using namespace bracken::render::lighting;
    ClusterGrid grid;
    const glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0));
    const glm::mat4 proj = glm::perspective(glm::radians(70.0f), 16.0f / 9.0f, 0.1f, 100.0f);
    grid.begin_frame(CameraState{view, proj, proj * view, glm::vec3(0.0f)}, 1920, 1080, 0.1f, 100.0f);

    constexpr uint32_t k_submitted = 300;
    uint32_t accepted = 0;
    uint32_t refused = 0;
    for (uint32_t i = 0; i < k_submitted; ++i) {
        PointLight p{};
        p.position = glm::vec3(0.0f, 0.0f, -2.0f);
        p.radius = 0.5f;
        p.color = glm::vec3(1.0f);
        p.intensity = 1.0f;
        if (grid.add_point_light(p)) {
            ++accepted;
        } else {
            ++refused;
        }
    }

    EXPECT_EQ(accepted, MAX_LIGHTS);
    EXPECT_EQ(refused, k_submitted - MAX_LIGHTS);
    EXPECT_EQ(grid.lights_dropped(), k_submitted - MAX_LIGHTS);
    EXPECT_EQ(grid.total_lights(), MAX_LIGHTS);
}

TEST(FrameDiagnosticsTest, DroppedLightCountResetsEachFrame) {
    using namespace bracken::render::lighting;
    ClusterGrid grid;
    const glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0));
    const glm::mat4 proj = glm::perspective(glm::radians(70.0f), 16.0f / 9.0f, 0.1f, 100.0f);
    const CameraState cam{view, proj, proj * view, glm::vec3(0.0f)};

    grid.begin_frame(cam, 1920, 1080, 0.1f, 100.0f);
    for (uint32_t i = 0; i < MAX_LIGHTS + 10; ++i) {
        PointLight p{};
        p.position = glm::vec3(0.0f, 0.0f, -2.0f);
        p.radius = 0.5f;
        (void)grid.add_point_light(p);
    }
    ASSERT_EQ(grid.lights_dropped(), 10u);

    grid.begin_frame(cam, 1920, 1080, 0.1f, 100.0f);
    EXPECT_EQ(grid.lights_dropped(), 0u);
}

// ---------------------------------------------------------------------------
// Determinism: the drop counters ride the chunk-index-ordered merge, so they
// are bit-identical at any worker count (docs/multithreading_plan.md §6.2).
// ---------------------------------------------------------------------------

TEST(FrameDiagnosticsTest, CountsAreIdenticalAtAnyWorkerCount) {
    DiagWorld w;
    const ModelHandle good = w.models.register_model("good", make_model(false)).value_or(0);
    FrameBuilder fb(w.packet);
    for (uint32_t i = 0; i < 4000; ++i) {
        // Every third draw names a handle nothing ever registered.
        fb.draw_model(i % 3 == 0 ? static_cast<ModelHandle>(7777) : good,
                      glm::translate(glm::mat4(1.0f), glm::vec3(static_cast<float>(i) * 0.01f, 0.0f, 0.0f)));
    }
    FrameAccess::seal(fb);

    const uint32_t expected = 4000u / 3u + 1u; // i % 3 == 0 over [0, 4000)

    auto run_with = [&](unsigned workers) {
        jobs::JobSystem js;
        js.init(workers);
        ScenePrep::PrepScratch scratch;
        std::vector<std::vector<gpu::InstanceData>> bins;
        std::vector<ModelHandle> touched;
        std::array<std::vector<std::vector<gpu::ShadowInstanceData>>, ScenePrep::MAX_SHADOW_CASCADES> shadow_bins;
        std::array<std::vector<ModelHandle>, ScenePrep::MAX_SHADOW_CASCADES> shadow_touched;
        std::vector<const ModelDrawCommand*> non_instanced;
        FrameStatsPartial stats;
        const ScenePrep::InstanceBinOutputs out{bins,          touched,       shadow_bins,
                                                    shadow_touched, non_instanced, stats};
        ScenePrep::prep_instance_batches(js, w.packet.model_commands(), w.params(), scratch, out);
        js.shutdown();
        return stats.diagnostics.draws_dropped_unknown_model;
    };

    EXPECT_EQ(run_with(1), expected);
    EXPECT_EQ(run_with(4), expected);
}

} // namespace
