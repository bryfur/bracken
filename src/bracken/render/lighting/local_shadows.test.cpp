#include <gtest/gtest.h>

#include "bracken/render/lighting/local_shadows.hpp"

#include <glm/glm.hpp>

#include <cstddef>
#include <vector>

namespace {

using namespace bracken::render::lighting;

PointLight make_point(glm::vec3 pos, float radius = 300.0f, float intensity = 8.0f, bool casts = true) {
    PointLight l{};
    l.position = pos;
    l.radius = radius;
    l.color = {1.0f, 1.0f, 1.0f};
    l.intensity = intensity;
    l.casts_shadows = casts;
    return l;
}

SpotLight make_spot(glm::vec3 pos, glm::vec3 dir, float radius = 1000.0f, float intensity = 10.0f,
                    bool casts = true) {
    SpotLight l{};
    l.position = pos;
    l.radius = radius;
    l.direction = glm::normalize(dir);
    l.inner_cos = std::cos(glm::radians(14.0f));
    l.outer_cos = std::cos(glm::radians(26.0f));
    l.color = {1.0f, 1.0f, 1.0f};
    l.intensity = intensity;
    l.casts_shadows = casts;
    return l;
}

constexpr int TILE_RES = 256;

// ============================================================================
// Pure helper math
// ============================================================================

TEST(LocalShadowMath, SlotGpuStructIs96Bytes) {
    // cluster_lighting.hlsli LocalShadowSlot mirror.
    EXPECT_EQ(sizeof(LocalShadowSlotGPU), 96u);
    EXPECT_EQ(offsetof(LocalShadowSlotGPU, rect), 64u);
    EXPECT_EQ(offsetof(LocalShadowSlotGPU, params), 80u);
}

TEST(LocalShadowMath, TileRectCoversAtlasGrid) {
    // Tile 0 = top-left origin; tiles walk rows of LOCAL_SHADOW_TILES_PER_ROW.
    const glm::vec4 r0 = local_shadow_tile_rect(0);
    EXPECT_FLOAT_EQ(r0.x, 0.0f);
    EXPECT_FLOAT_EQ(r0.y, 0.0f);
    EXPECT_FLOAT_EQ(r0.z, 1.0f / 8.0f);
    EXPECT_FLOAT_EQ(r0.w, 1.0f / 8.0f);

    const glm::vec4 r9 = local_shadow_tile_rect(9); // col 1, row 1
    EXPECT_FLOAT_EQ(r9.x, 1.0f / 8.0f);
    EXPECT_FLOAT_EQ(r9.y, 1.0f / 8.0f);

    const glm::vec4 rlast = local_shadow_tile_rect(LOCAL_SHADOW_MAX_SLOTS - 1);
    EXPECT_FLOAT_EQ(rlast.x, 7.0f / 8.0f);
    EXPECT_FLOAT_EQ(rlast.y, 7.0f / 8.0f);
}

TEST(LocalShadowMath, CubeFaceSelectsDominantAxis) {
    EXPECT_EQ(local_shadow_cube_face({1.0f, 0.1f, 0.2f}), 0);
    EXPECT_EQ(local_shadow_cube_face({-5.0f, 2.0f, 2.0f}), 1);
    EXPECT_EQ(local_shadow_cube_face({0.1f, 3.0f, 0.2f}), 2);
    EXPECT_EQ(local_shadow_cube_face({0.0f, -1.0f, 0.5f}), 3);
    EXPECT_EQ(local_shadow_cube_face({0.3f, 0.2f, 2.0f}), 4);
    EXPECT_EQ(local_shadow_cube_face({0.3f, 0.2f, -2.0f}), 5);
}

// Property: any point selected onto face F by the dominant-axis rule must
// project inside face F's clip volume (the face FOV is slightly wider than
// 90 deg, so dominant-axis membership implies containment).
TEST(LocalShadowMath, CubeFaceViewProjContainsSelectedPoints) {
    const glm::vec3 light_pos(100.0f, 50.0f, -30.0f);
    const float radius = 400.0f;
    const float near_plane = local_shadow_near_plane(radius);

    // Deterministic direction sweep (including diagonals-ish directions).
    std::vector<glm::vec3> dirs;
    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            for (int z = -2; z <= 2; ++z) {
                if (x == 0 && y == 0 && z == 0) continue;
                dirs.push_back(glm::normalize(glm::vec3(x, y, z)));
            }
        }
    }
    for (const glm::vec3& d : dirs) {
        for (float dist : {near_plane * 2.0f, radius * 0.5f, radius * 0.95f}) {
            const glm::vec3 p = light_pos + d * dist;
            const int face = local_shadow_cube_face(p - light_pos);
            const glm::mat4 vp = local_point_face_view_proj(light_pos, face, near_plane, radius, TILE_RES);
            const glm::vec4 clip = vp * glm::vec4(p, 1.0f);
            ASSERT_GT(clip.w, 0.0f) << "dir " << d.x << "," << d.y << "," << d.z;
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            EXPECT_LE(std::abs(ndc.x), 1.0001f);
            EXPECT_LE(std::abs(ndc.y), 1.0001f);
            EXPECT_GE(ndc.z, -0.0001f); // standard-Z: near -> 0
            EXPECT_LE(ndc.z, 1.0001f);  //             far  -> 1
        }
    }
}

TEST(LocalShadowMath, PointFaceDepthIsStandardZ) {
    const glm::vec3 pos(0.0f);
    const float radius = 200.0f;
    const float near_plane = local_shadow_near_plane(radius);
    const glm::mat4 vp = local_point_face_view_proj(pos, 0, near_plane, radius, TILE_RES);

    auto depth_at = [&](float dist) {
        const glm::vec4 clip = vp * glm::vec4(dist, 0.0f, 0.0f, 1.0f);
        return clip.z / clip.w;
    };
    // near -> ~0, far (radius) -> ~1, monotonically increasing: standard-Z.
    EXPECT_NEAR(depth_at(near_plane), 0.0f, 1e-4f);
    EXPECT_NEAR(depth_at(radius), 1.0f, 1e-4f);
    EXPECT_LT(depth_at(radius * 0.25f), depth_at(radius * 0.75f));
}

TEST(LocalShadowMath, SpotViewProjCoversOuterCone) {
    const glm::vec3 pos(10.0f, 200.0f, 5.0f);
    const glm::vec3 dir = glm::normalize(glm::vec3(0.3f, -1.0f, 0.2f));
    const float outer_deg = 26.0f;
    const float outer_cos = std::cos(glm::radians(outer_deg));
    const float radius = 1000.0f;
    const glm::mat4 vp = local_spot_view_proj(pos, dir, outer_cos, local_shadow_near_plane(radius), radius);

    // Points ON the outer cone edge (in an arbitrary orthonormal frame around
    // dir) must project inside the widened frustum.
    const glm::vec3 side = glm::normalize(glm::cross(dir, glm::vec3(0, 0, 1)));
    const glm::vec3 up = glm::normalize(glm::cross(side, dir));
    for (float angle = 0.0f; angle < 6.28f; angle += 0.5f) {
        const glm::vec3 rim =
            glm::normalize(dir + std::tan(glm::radians(outer_deg)) * (std::cos(angle) * side + std::sin(angle) * up));
        const glm::vec3 p = pos + rim * (radius * 0.8f);
        const glm::vec4 clip = vp * glm::vec4(p, 1.0f);
        ASSERT_GT(clip.w, 0.0f);
        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
        EXPECT_LT(std::abs(ndc.x), 1.0f) << "angle " << angle;
        EXPECT_LT(std::abs(ndc.y), 1.0f) << "angle " << angle;
        EXPECT_GT(ndc.z, 0.0f);
        EXPECT_LT(ndc.z, 1.0f);
    }
}

TEST(LocalShadowMath, ScoreFavorsBrightCloseLights) {
    EXPECT_GT(local_shadow_light_score(10.0f, 300.0f, 100.0f), local_shadow_light_score(10.0f, 300.0f, 500.0f));
    EXPECT_GT(local_shadow_light_score(10.0f, 300.0f, 100.0f), local_shadow_light_score(2.0f, 300.0f, 100.0f));
    // Distance clamps at 1 so a light at the camera doesn't explode.
    EXPECT_FLOAT_EQ(local_shadow_light_score(1.0f, 1.0f, 0.0f), 1.0f);
}

// ============================================================================
// Selection / assignment / caching
// ============================================================================

struct SelectorFixture {
    LocalShadowSelector sel;
    LocalShadowConfig cfg; // defaults: 4 point + 2 spot, 6 tiles/frame
    glm::vec3 cam{0.0f};

    void run(std::vector<PointLight>& points, std::vector<SpotLight>& spots,
             std::vector<LocalShadowDynamicCaster> dyn = {}) {
        sel.update(points, spots, cam, dyn, cfg, TILE_RES);
    }
};

TEST(LocalShadowSelector, AssignsOptedInLightsAndPublishesSlots) {
    SelectorFixture f;
    std::vector<PointLight> points = {make_point({100, 0, 0})};
    std::vector<SpotLight> spots = {make_spot({0, 100, 0}, {0, -1, 0})};

    f.run(points, spots);
    // Fresh assignments: 6 point faces exceed nothing (cap 6), so the point
    // renders fully this frame; the spot needs 1 more tile than the cap
    // leaves — it renders next frame.
    EXPECT_EQ(points[0].shadow_slot, 0);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 6);
    EXPECT_EQ(spots[0].shadow_slot, -1);

    f.run(points, spots);
    EXPECT_EQ(points[0].shadow_slot, 0);
    EXPECT_EQ(spots[0].shadow_slot, LOCAL_SHADOW_SPOT_TILE_BASE);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 1); // just the spot tile

    // Third frame, nothing changed: full cache reuse, zero tiles rendered.
    f.run(points, spots);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 0);
    EXPECT_EQ(points[0].shadow_slot, 0);
    EXPECT_EQ(spots[0].shadow_slot, LOCAL_SHADOW_SPOT_TILE_BASE);
    EXPECT_EQ(f.sel.shadowed_point_lights(), 1);
    EXPECT_EQ(f.sel.shadowed_spot_lights(), 1);
}

TEST(LocalShadowSelector, OptOutLightsAreNeverSelected) {
    SelectorFixture f;
    std::vector<PointLight> points = {make_point({100, 0, 0}, 300.0f, 8.0f, /*casts=*/false)};
    std::vector<SpotLight> spots;
    f.run(points, spots);
    EXPECT_EQ(points[0].shadow_slot, -1);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 0);
}

TEST(LocalShadowSelector, StaticLightReusesTilesMovedLightReRenders) {
    SelectorFixture f;
    std::vector<PointLight> points = {make_point({100, 0, 0})};
    std::vector<SpotLight> spots;
    f.run(points, spots);
    ASSERT_EQ(f.sel.tiles_rendered_last_frame(), 6);

    // Static: cached.
    f.run(points, spots);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 0);

    // Moved slightly: matched to the same slot (positional continuity) but
    // every face re-renders.
    points[0].position.x += 5.0f;
    f.run(points, spots);
    EXPECT_EQ(points[0].shadow_slot, 0);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 6);
}

TEST(LocalShadowSelector, DynamicCasterInsideRadiusDirtiesTiles) {
    SelectorFixture f;
    std::vector<PointLight> points = {make_point({100, 0, 0}, 300.0f)};
    std::vector<SpotLight> spots;
    f.run(points, spots);
    f.run(points, spots);
    ASSERT_EQ(f.sel.tiles_rendered_last_frame(), 0);

    // A skinned/moving caster inside the light sphere forces a re-render...
    f.run(points, spots, {{{200.0f, 0.0f, 0.0f}, 20.0f}});
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 6);

    // ...but one far outside does not.
    f.run(points, spots, {{{2000.0f, 0.0f, 0.0f}, 20.0f}});
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 0);
}

TEST(LocalShadowSelector, TeleportedLightReadsAsNewAndReRenders) {
    SelectorFixture f;
    std::vector<PointLight> points = {make_point({100, 0, 0}, 300.0f)};
    std::vector<SpotLight> spots;
    f.run(points, spots);
    f.run(points, spots);
    ASSERT_EQ(f.sel.tiles_rendered_last_frame(), 0);

    // Far beyond the positional-match threshold (0.5 * radius): the slot is
    // freed and refilled as a fresh assignment.
    points[0].position = {5000.0f, 0.0f, 0.0f};
    f.cam = {4900.0f, 0.0f, 0.0f}; // keep it scoring
    f.run(points, spots);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 6);
    EXPECT_EQ(points[0].shadow_slot, 0);
}

TEST(LocalShadowSelector, BudgetLimitsSelectionToTopScorers) {
    SelectorFixture f;
    f.cfg.point_budget = 2;
    f.cfg.max_tile_updates_per_frame = 100; // isolate the budget from the cap
    // 4 candidates at increasing distance: nearest two must win.
    std::vector<PointLight> points = {
        make_point({100, 0, 0}),
        make_point({200, 0, 0}),
        make_point({300, 0, 0}),
        make_point({400, 0, 0}),
    };
    std::vector<SpotLight> spots;
    f.run(points, spots);
    EXPECT_GE(points[0].shadow_slot, 0);
    EXPECT_GE(points[1].shadow_slot, 0);
    EXPECT_EQ(points[2].shadow_slot, -1);
    EXPECT_EQ(points[3].shadow_slot, -1);
    EXPECT_EQ(f.sel.shadowed_point_lights(), 2);
}

TEST(LocalShadowSelector, HysteresisKeepsIncumbentAgainstMarginalChallenger) {
    SelectorFixture f;
    f.cfg.point_budget = 1;
    f.cfg.max_tile_updates_per_frame = 100;
    f.cfg.hysteresis = 0.25f;

    // Incumbent at distance 100 (score = 8 * 300 / 100 = 24).
    std::vector<PointLight> points = {make_point({100, 0, 0})};
    std::vector<SpotLight> spots;
    f.run(points, spots);
    ASSERT_EQ(points[0].shadow_slot, 0);

    // Challenger scores ~10% higher — inside the 25% margin: incumbent stays.
    points.push_back(make_point({90, 0, 0}, 300.0f, 8.0f)); // score ~26.7
    f.run(points, spots);
    EXPECT_EQ(points[0].shadow_slot, 0);
    EXPECT_EQ(points[1].shadow_slot, -1);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 0); // no thrash

    // Challenger at 2x the score — beats the margin: eviction + re-render.
    points[1] = make_point({50, 0, 0}, 300.0f, 8.0f); // score 48
    f.run(points, spots);
    EXPECT_EQ(points[0].shadow_slot, -1);
    EXPECT_EQ(points[1].shadow_slot, 0);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 6);
}

TEST(LocalShadowSelector, TileCapDegradesToStalenessWithRoundRobinCatchUp) {
    SelectorFixture f;
    f.cfg.point_budget = 3;
    // Cap 6: only one point light's 6 faces fit per frame.
    std::vector<PointLight> points = {
        make_point({100, 0, 0}),
        make_point({150, 0, 0}, 300.0f, 7.0f),
        make_point({200, 0, 0}, 300.0f, 6.0f),
    };
    std::vector<SpotLight> spots;

    f.run(points, spots);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 6);
    int shadowed_frame1 = f.sel.shadowed_point_lights();
    EXPECT_EQ(shadowed_frame1, 1); // the other two wait their turn, unshadowed

    f.run(points, spots);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 6);
    EXPECT_EQ(f.sel.shadowed_point_lights(), 2);

    f.run(points, spots);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 6);
    EXPECT_EQ(f.sel.shadowed_point_lights(), 3); // all caught up

    f.run(points, spots);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 0); // steady state: cached
    EXPECT_EQ(f.sel.shadowed_point_lights(), 3);
}

TEST(LocalShadowSelector, MovingLightsShareTheCapFairly) {
    SelectorFixture f;
    f.cfg.point_budget = 2;
    std::vector<PointLight> points = {
        make_point({100, 0, 0}),
        make_point({150, 0, 0}, 300.0f, 7.0f),
    };
    std::vector<SpotLight> spots;
    // Warm up until both are fully rendered.
    f.run(points, spots);
    f.run(points, spots);
    ASSERT_EQ(f.sel.shadowed_point_lights(), 2);

    // Both move every frame: 12 dirty tiles vs cap 6. Round-robin must
    // alternate which light gets its faces refreshed, and both stay shadowed
    // (stale, never unshadowed).
    uint64_t before = f.sel.tiles_rendered_total();
    for (int i = 0; i < 4; ++i) {
        points[0].position.x += 1.0f;
        points[1].position.x += 1.0f;
        f.run(points, spots);
        EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 6);
        EXPECT_EQ(f.sel.shadowed_point_lights(), 2);
    }
    EXPECT_EQ(f.sel.tiles_rendered_total() - before, 24u);
}

TEST(LocalShadowSelector, SlotDataWrittenForRenderedTiles) {
    SelectorFixture f;
    std::vector<PointLight> points = {make_point({100, 0, 0}, 300.0f)};
    std::vector<SpotLight> spots;
    f.run(points, spots);
    ASSERT_TRUE(f.sel.slot_data_dirty());

    const auto& slots = f.sel.slot_data();
    for (int face = 0; face < 6; ++face) {
        const auto& s = slots[static_cast<size_t>(face)];
        EXPECT_EQ(s.rect, local_shadow_tile_rect(face));
        EXPECT_GT(s.params.x, 0.0f);                          // world texel per unit distance
        EXPECT_FLOAT_EQ(s.params.y, 2.0f / TILE_RES);         // PCF clamp inset
        EXPECT_FLOAT_EQ(s.params.z, local_shadow_near_plane(300.0f));
        EXPECT_FLOAT_EQ(s.params.w, 1.0f / (TILE_RES * 8.0f)); // atlas texel
        // The slot matrix must match the request the renderer draws with.
        EXPECT_EQ(s.view_proj, f.sel.tile_requests()[static_cast<size_t>(face)].view_proj);
    }
    f.sel.clear_slot_data_dirty();
    f.run(points, spots);
    EXPECT_FALSE(f.sel.slot_data_dirty()); // cached frame: no upload
}

TEST(LocalShadowSelector, InvalidateAllTilesForcesFullReRender) {
    SelectorFixture f;
    std::vector<PointLight> points = {make_point({100, 0, 0})};
    std::vector<SpotLight> spots;
    f.run(points, spots);
    f.run(points, spots);
    ASSERT_EQ(f.sel.tiles_rendered_last_frame(), 0);

    f.sel.invalidate_all_tiles(); // e.g. atlas resolution changed
    f.run(points, spots);
    EXPECT_EQ(f.sel.tiles_rendered_last_frame(), 6);
    EXPECT_EQ(points[0].shadow_slot, 0);
}

TEST(LocalShadowSelector, DeterministicAcrossIdenticalRuns) {
    std::vector<PointLight> base_points = {
        make_point({100, 0, 0}), make_point({150, 30, 0}, 250.0f, 5.0f), make_point({-80, 0, 40}, 400.0f, 3.0f),
        make_point({0, 60, -200}, 350.0f, 9.0f),
    };
    std::vector<SpotLight> base_spots = {make_spot({0, 100, 0}, {0, -1, 0}), make_spot({50, 80, 0}, {1, -1, 0})};

    auto run_sequence = [&](LocalShadowSelector& sel) {
        LocalShadowConfig cfg;
        std::vector<PointLight> points = base_points;
        std::vector<SpotLight> spots = base_spots;
        std::vector<int> slots;
        for (int i = 0; i < 6; ++i) {
            for (auto& p : points) p.position.x += 1.0f;
            sel.update(points, spots, {0, 0, 0}, {}, cfg, TILE_RES);
            for (const auto& p : points) slots.push_back(p.shadow_slot);
            for (const auto& s : spots) slots.push_back(s.shadow_slot);
            slots.push_back(sel.tiles_rendered_last_frame());
        }
        return slots;
    };

    LocalShadowSelector a;
    LocalShadowSelector b;
    EXPECT_EQ(run_sequence(a), run_sequence(b));
}

TEST(LocalShadowSelector, ShrinkingBudgetDropsExcessGroups) {
    SelectorFixture f;
    f.cfg.point_budget = 2;
    f.cfg.max_tile_updates_per_frame = 100;
    std::vector<PointLight> points = {make_point({100, 0, 0}), make_point({200, 0, 0})};
    std::vector<SpotLight> spots;
    f.run(points, spots);
    ASSERT_EQ(f.sel.shadowed_point_lights(), 2);

    f.cfg.point_budget = 1;
    f.run(points, spots);
    EXPECT_EQ(f.sel.shadowed_point_lights(), 1);
    EXPECT_GE(points[0].shadow_slot, 0); // the higher scorer keeps its slot
    EXPECT_EQ(points[1].shadow_slot, -1);
}

} // namespace
