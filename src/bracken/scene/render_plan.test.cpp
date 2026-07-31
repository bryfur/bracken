// Table tests for the frame's pass-selection tree (scene/render_plan.hpp).
//
// Before this existed the tree lived as ~15 lines of interleaved predicates in
// the middle of render_frame(), reachable only with a GPU device and a window
// — i.e. it had no coverage at all. plan_frame() is pure, so every
// combination of settings x packet content x renderer capability is now a
// headless table row.
//
// The cases that actually bite, and are asserted individually below:
//  - TAA requested but the post chain never initialized -> INACTIVE. Jitter
//    without a resolve to integrate the samples just shimmers.
//  - a frame that describes geometry but no camera was ever set -> no content.
//  - IBL enabled but the prefilter chain is not ready -> inactive.
//  - shadows: every cascade count, and Off collapsing to zero.

#include <gtest/gtest.h>

#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/render_plan.hpp"
#include "bracken/testing/frame_test.hpp"

#include <glm/glm.hpp>
#include <string>

using bracken::testing::FrameAccess;
using namespace bracken;
using namespace bracken::scene;

namespace {

// A sealed packet that describes 3D geometry (the scene flags default on, so
// an untouched packet already has content — assert that rather than assume).
FramePacket& content_packet() {
    static FramePacket p;
    static bool built = false;
    if (!built) {
        FrameBuilder fb(p);
        fb.set_draw_ground(true);
        FrameAccess::seal(fb);
        built = true;
    }
    return p;
}

// A sealed packet that describes nothing 3D: every scene flag off and no
// commands. UI-only frames look like this.
FramePacket& empty_packet() {
    static FramePacket p;
    static bool built = false;
    if (!built) {
        FrameBuilder fb(p);
        fb.set_draw_skybox(false);
        fb.set_draw_ground(false);
        fb.set_draw_grass(false);
        FrameAccess::seal(fb);
        built = true;
    }
    return p;
}

// Everything alive: the shape of a healthy frame after init succeeded.
RendererCaps all_ready() {
    RendererCaps caps;
    caps.post_ready = true;
    caps.taa_ready = true;
    caps.ibl_ready = true;
    caps.camera_seen = true;
    caps.shadows_ready = true;
    caps.shadow_cascades = 4;
    return caps;
}

} // namespace

TEST(RenderPlan, PacketFixturesAreWhatTheyClaim) {
    EXPECT_TRUE(content_packet().has_3d_content());
    EXPECT_FALSE(empty_packet().has_3d_content());
}

// ---------------------------------------------------------------------------
// has_content — the "I built a frame and see nothing" decision
// ---------------------------------------------------------------------------

TEST(RenderPlan, ContentNeedsBothGeometryAndACameraEverSet) {
    const GraphicsSettings gfx;
    RendererCaps caps = all_ready();

    EXPECT_TRUE(plan_frame(gfx, content_packet(), caps).has_content);

    caps.camera_seen = false;
    EXPECT_FALSE(plan_frame(gfx, content_packet(), caps).has_content)
        << "geometry with no camera ever set must not run the 3D passes";

    caps.camera_seen = true;
    EXPECT_FALSE(plan_frame(gfx, empty_packet(), caps).has_content);
}

// ---------------------------------------------------------------------------
// HDR path
// ---------------------------------------------------------------------------

TEST(RenderPlan, HdrPathFollowsPostReadinessOnly) {
    GraphicsSettings gfx;
    RendererCaps caps = all_ready();
    EXPECT_TRUE(plan_frame(gfx, content_packet(), caps).use_hdr_path);

    caps.post_ready = false;
    EXPECT_FALSE(plan_frame(gfx, content_packet(), caps).use_hdr_path);

    // No setting turns the HDR path off — it is a capability, not a taste.
    gfx.aa_mode = AAMode::Off;
    gfx.bloom_enabled = false;
    gfx.auto_exposure = false;
    caps.post_ready = true;
    EXPECT_TRUE(plan_frame(gfx, content_packet(), caps).use_hdr_path);
}

// ---------------------------------------------------------------------------
// TAA — every (mode, post_ready, taa_ready) combination
// ---------------------------------------------------------------------------

TEST(RenderPlan, TaaActiveTable) {
    struct Row {
        AAMode mode;
        bool post_ready;
        bool taa_ready;
        bool expect_active;
    };
    const Row rows[] = {
        {AAMode::TAA, true, true, true},
        {AAMode::TAA, true, false, false},  // resolve targets missing
        {AAMode::TAA, false, true, false},  // no offscreen HDR path to resolve into
        {AAMode::TAA, false, false, false},
        {AAMode::FXAA, true, true, false},
        {AAMode::FXAA, false, false, false},
        {AAMode::Off, true, true, false},
        {AAMode::Off, false, false, false},
    };
    for (const Row& r : rows) {
        GraphicsSettings gfx;
        gfx.aa_mode = r.mode;
        RendererCaps caps = all_ready();
        caps.post_ready = r.post_ready;
        caps.taa_ready = r.taa_ready;
        const RenderPlan plan = plan_frame(gfx, content_packet(), caps);
        EXPECT_EQ(plan.taa_active, r.expect_active)
            << "aa_mode=" << static_cast<int>(r.mode) << " post_ready=" << r.post_ready
            << " taa_ready=" << r.taa_ready;
        // The history drop is the exact complement: any frame TAA does not
        // resolve must stale the history or a re-enable ghosts.
        EXPECT_EQ(plan.invalidate_taa_history, !r.expect_active);
    }
}

// ---------------------------------------------------------------------------
// Auto exposure invalidation
// ---------------------------------------------------------------------------

TEST(RenderPlan, AutoExposureInvalidatesOnEveryGap) {
    struct Row {
        bool setting;
        bool post_ready;
        bool has_geometry;
        bool expect_invalidate;
    };
    const Row rows[] = {
        {true, true, true, false},  // the only frame it records on
        {false, true, true, true},  // setting off
        {true, false, true, true},  // no HDR path
        {true, true, false, true},  // nothing drawn
        {false, false, false, true},
    };
    for (const Row& r : rows) {
        GraphicsSettings gfx;
        gfx.auto_exposure = r.setting;
        RendererCaps caps = all_ready();
        caps.post_ready = r.post_ready;
        const FramePacket& packet = r.has_geometry ? content_packet() : empty_packet();
        EXPECT_EQ(plan_frame(gfx, packet, caps).invalidate_auto_exposure, r.expect_invalidate)
            << "setting=" << r.setting << " post_ready=" << r.post_ready << " geometry=" << r.has_geometry;
    }
}

TEST(RenderPlan, AutoExposureInvalidatesWhenNoCameraWasEverSet) {
    const GraphicsSettings gfx; // auto exposure on by default
    RendererCaps caps = all_ready();
    caps.camera_seen = false;
    EXPECT_TRUE(plan_frame(gfx, content_packet(), caps).invalidate_auto_exposure);
}

// ---------------------------------------------------------------------------
// IBL
// ---------------------------------------------------------------------------

TEST(RenderPlan, IblNeedsSettingAndChain) {
    for (bool setting : {false, true}) {
        for (bool ready : {false, true}) {
            GraphicsSettings gfx;
            gfx.ibl_enabled = setting;
            RendererCaps caps = all_ready();
            caps.ibl_ready = ready;
            EXPECT_EQ(plan_frame(gfx, content_packet(), caps).ibl_active, setting && ready)
                << "setting=" << setting << " ready=" << ready;
        }
    }
}

TEST(RenderPlan, IblIsIndependentOfTheHdrPath) {
    // The IBL prefilter chain feeds the forward shaders directly, so it stays
    // available on the fallback (direct-to-swapchain) path.
    GraphicsSettings gfx;
    RendererCaps caps = all_ready();
    caps.post_ready = false;
    EXPECT_TRUE(plan_frame(gfx, content_packet(), caps).ibl_active);
}

// ---------------------------------------------------------------------------
// Shadows — active_cascades IS the "shadows on" predicate
// ---------------------------------------------------------------------------

TEST(RenderPlan, EveryCascadeCountSurvivesWhenShadowsAreOn) {
    for (int count = 1; count <= 4; ++count) {
        for (ShadowMode mode : {ShadowMode::PCF, ShadowMode::PCSS}) {
            GraphicsSettings gfx;
            gfx.shadow_mode = mode;
            RendererCaps caps = all_ready();
            caps.shadow_cascades = count;
            EXPECT_EQ(plan_frame(gfx, content_packet(), caps).active_cascades, count)
                << "count=" << count << " mode=" << static_cast<int>(mode);
        }
    }
}

TEST(RenderPlan, ShadowsOffOrAtlasMissingCollapsesToZeroCascades) {
    GraphicsSettings gfx;
    gfx.shadow_mode = ShadowMode::Off;
    RendererCaps caps = all_ready();
    EXPECT_EQ(plan_frame(gfx, content_packet(), caps).active_cascades, 0)
        << "with the mode off the cascade frustums are stale and nothing may be binned";

    gfx.shadow_mode = ShadowMode::PCSS;
    caps.shadows_ready = false;
    EXPECT_EQ(plan_frame(gfx, content_packet(), caps).active_cascades, 0);
}

// ---------------------------------------------------------------------------
// Cull constants
// ---------------------------------------------------------------------------

TEST(RenderPlan, DrawDistanceIsSquaredForTheCullCompare) {
    for (DrawDistance d : {DrawDistance::D500, DrawDistance::D1000, DrawDistance::D2000, DrawDistance::D4000,
                           DrawDistance::D8000, DrawDistance::D16000, DrawDistance::D32000}) {
        GraphicsSettings gfx;
        gfx.draw_distance = d;
        const float units = gfx.get_draw_distance();
        EXPECT_FLOAT_EQ(plan_frame(gfx, content_packet(), all_ready()).draw_dist_sq, units * units);
    }
}

TEST(RenderPlan, FogAndFrustumCullingMirrorTheirSettings) {
    for (bool fog : {false, true}) {
        for (bool cull : {false, true}) {
            GraphicsSettings gfx;
            gfx.fog_enabled = fog;
            gfx.frustum_culling = cull;
            const RenderPlan plan = plan_frame(gfx, content_packet(), all_ready());
            EXPECT_EQ(plan.fog_active, fog);
            EXPECT_EQ(plan.do_frustum_cull, cull);
        }
    }
}

// ---------------------------------------------------------------------------
// Purity + the whole grid
// ---------------------------------------------------------------------------

TEST(RenderPlan, PlanningIsPureAndRepeatable) {
    GraphicsSettings gfx;
    gfx.aa_mode = AAMode::TAA;
    const RenderPlan a = plan_frame(gfx, content_packet(), all_ready());
    const RenderPlan b = plan_frame(gfx, content_packet(), all_ready());
    // Field-wise, not memcmp: the struct has padding and plan_frame fills
    // fields, not bytes.
    EXPECT_EQ(a.has_content, b.has_content);
    EXPECT_EQ(a.use_hdr_path, b.use_hdr_path);
    EXPECT_EQ(a.taa_active, b.taa_active);
    EXPECT_EQ(a.ibl_active, b.ibl_active);
    EXPECT_EQ(a.fog_active, b.fog_active);
    EXPECT_EQ(a.do_frustum_cull, b.do_frustum_cull);
    EXPECT_EQ(a.draw_dist_sq, b.draw_dist_sq);
    EXPECT_EQ(a.active_cascades, b.active_cascades);
    EXPECT_EQ(a.invalidate_taa_history, b.invalidate_taa_history);
    EXPECT_EQ(a.invalidate_auto_exposure, b.invalidate_auto_exposure);
}

// Sweep the whole capability cube against the decision rules stated once
// here. This is the regression net: any future edit that couples two of the
// decisions together shows up as a row failure naming its own inputs.
TEST(RenderPlan, FullCapabilityCube) {
    int rows = 0;
    for (bool post : {false, true}) {
        for (bool taa_ready : {false, true}) {
            for (bool ibl_ready : {false, true}) {
                for (bool camera : {false, true}) {
                    for (bool shadows_ready : {false, true}) {
                        for (AAMode mode : {AAMode::Off, AAMode::FXAA, AAMode::TAA}) {
                            for (ShadowMode sm : {ShadowMode::Off, ShadowMode::PCF, ShadowMode::PCSS}) {
                                for (bool geometry : {false, true}) {
                                    GraphicsSettings gfx;
                                    gfx.aa_mode = mode;
                                    gfx.shadow_mode = sm;
                                    RendererCaps caps;
                                    caps.post_ready = post;
                                    caps.taa_ready = taa_ready;
                                    caps.ibl_ready = ibl_ready;
                                    caps.camera_seen = camera;
                                    caps.shadows_ready = shadows_ready;
                                    caps.shadow_cascades = 3;
                                    const FramePacket& packet = geometry ? content_packet() : empty_packet();
                                    const RenderPlan p = plan_frame(gfx, packet, caps);

                                    const std::string where =
                                        "post=" + std::to_string(post) + " taa_ready=" + std::to_string(taa_ready) +
                                        " ibl_ready=" + std::to_string(ibl_ready) + " camera=" +
                                        std::to_string(camera) + " shadows_ready=" + std::to_string(shadows_ready) +
                                        " aa=" + std::to_string(static_cast<int>(mode)) + " sm=" +
                                        std::to_string(static_cast<int>(sm)) + " geo=" + std::to_string(geometry);

                                    EXPECT_EQ(p.has_content, geometry && camera) << where;
                                    EXPECT_EQ(p.use_hdr_path, post) << where;
                                    EXPECT_EQ(p.taa_active, mode == AAMode::TAA && post && taa_ready) << where;
                                    EXPECT_EQ(p.invalidate_taa_history, !p.taa_active) << where;
                                    EXPECT_EQ(p.ibl_active, ibl_ready) << where; // ibl_enabled defaults on
                                    EXPECT_EQ(p.invalidate_auto_exposure, !(post && p.has_content)) << where;
                                    EXPECT_EQ(p.active_cascades,
                                              (shadows_ready && sm != ShadowMode::Off) ? 3 : 0)
                                        << where;
                                    ++rows;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(rows, 2 * 2 * 2 * 2 * 2 * 3 * 3 * 2);
}
