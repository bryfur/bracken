// The occlusion test's ONE-SIDEDNESS, brute-forced rather than argued.
//
// A false cull is the worst bug this feature can have and the hardest to see:
// a prop missing behind a wall is a few hundred pixels, which CLAUDE.md records
// twice as being structurally invisible in `mean_luma`. So the properties below
// are checked against references and against random inputs, not against
// recorded values:
//
//   * a box straddling the near plane is NEVER culled (the perspective divide
//     is meaningless there and the corner hull is not the projection);
//   * `project_aabb`'s rect CONTAINS the projection of interior points, over
//     10k random samples — the containment the footprint walk depends on;
//   * `nearest_depth` is the maximum device depth over the whole box, not just
//     its corners;
//   * a box in front of every texel is never culled; behind every texel it IS;
//     behind SOME texels it is not;
//   * the screen-fraction gate refuses a big box (the occluder feedback loop);
//   * an odd mip axis's OVERSIZED last texel is still read — the hole that
//     `uv * mip_w` addressing would open;
//   * `camera_travel` inflation is MONOTONE in the safe direction: more travel
//     can only reduce culling, never increase it.
//
// Pure math: no GPU, no device, no renderer.

#include <gtest/gtest.h>

#include "bracken/contracts/occlusion.hpp"

#include <cmath>
#include <cstdint>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

using namespace bracken;

namespace {

/// A perspective view looking down -Z from the origin. Standard glm RH_ZO
/// (near -> 0), NOT the engine's reversed-Z: the tests below never assume which
/// end is near — they take every depth value from `project_aabb` itself and
/// only ever compare depths against each other.
glm::mat4 perspective_vp() {
    const glm::mat4 view = glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    return glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.5f, 500.0f) * view;
}

/// Field of `w * h` texels all holding `depth`.
struct Field {
    std::vector<float> texels;
    OcclusionField f;

    Field(int w, int h, int level, float depth, const glm::mat4& vp) : texels(static_cast<std::size_t>(w * h), depth) {
        f.depth = texels;
        f.mip_w = w;
        f.mip_h = h;
        f.level = level;
        f.screen_w = w << level;
        f.screen_h = h << level;
        f.view_proj = vp;
        f.max_tested_screen_fraction = 1.0f;
    }
};

/// Deterministic 32-bit LCG — a fixed sequence, so a failure reproduces.
struct Rng {
    uint32_t state = 0x13579BDFu;
    uint32_t next() {
        state = state * 1664525u + 1013904223u;
        return state;
    }
    float unit() { return static_cast<float>(next() >> 8) / static_cast<float>(1u << 24); }
    float range(float lo, float hi) { return lo + (hi - lo) * unit(); }
};

} // namespace

// ---------------------------------------------------------------------------
// project_aabb
// ---------------------------------------------------------------------------

TEST(OcclusionProject, ABoxStraddlingTheNearPlaneIsNeverValid) {
    const glm::mat4 vp = perspective_vp();
    // Spans the camera: some corners are behind it.
    const OcclusionBounds b = project_aabb(vp, glm::vec3(-5.0f, -5.0f, -10.0f), glm::vec3(5.0f, 5.0f, 10.0f));
    EXPECT_FALSE(b.valid);
    // An invalid box reports a full-screen fraction, so even a caller that
    // forgot to check `valid` refuses to test it.
    EXPECT_FLOAT_EQ(b.screen_fraction(), 1.0f);

    Field field(8, 8, 3, 1.0f, vp);
    EXPECT_FALSE(occluded_by_field(field.f, b));
}

TEST(OcclusionProject, ABoxEntirelyBehindTheCameraIsNeverValid) {
    const glm::mat4 vp = perspective_vp();
    const OcclusionBounds b = project_aabb(vp, glm::vec3(-1.0f, -1.0f, 10.0f), glm::vec3(1.0f, 1.0f, 20.0f));
    EXPECT_FALSE(b.valid);
}

TEST(OcclusionProject, UpInTheWorldIsUpInTheFrame) {
    // The convention test, and the one whose absence let a mirrored footprint
    // ship. Every other test in this file is vertically SYMMETRIC about the
    // view axis, so flipping the Y mapping changes none of their answers — and
    // the flip is invisible in mean_luma too. It cost 419 of the playground's
    // 464 entities: distant geometry near the skyline was tested against the
    // near ground at the bottom of the frame and read as hopelessly behind it.
    //
    // glm (GLM_FORCE_DEPTH_ZERO_TO_ONE) leaves NDC +Y pointing UP; texel row 0
    // of a copied depth target is the TOP of the frame. So a box ABOVE the view
    // axis must land at SMALL uv.y. render/taa.hpp:51 is the same mapping.
    const glm::mat4 vp = perspective_vp(); // looks down -Z from the origin

    const OcclusionBounds high = project_aabb(vp, glm::vec3(-1.0f, 20.0f, -101.0f), glm::vec3(1.0f, 22.0f, -99.0f));
    const OcclusionBounds low = project_aabb(vp, glm::vec3(-1.0f, -22.0f, -101.0f), glm::vec3(1.0f, -20.0f, -99.0f));
    ASSERT_TRUE(high.valid);
    ASSERT_TRUE(low.valid);

    EXPECT_LT(high.max_uv.y, 0.5f) << "a box above the view axis must be in the TOP half of the frame";
    EXPECT_GT(low.min_uv.y, 0.5f) << "a box below the view axis must be in the BOTTOM half of the frame";
    EXPECT_LT(high.max_uv.y, low.min_uv.y) << "the Y mapping is mirrored";

    // X is not flipped, and pinning it here is what stops a future 'fix' from
    // flipping the wrong axis.
    const OcclusionBounds right = project_aabb(vp, glm::vec3(20.0f, -1.0f, -101.0f), glm::vec3(22.0f, 1.0f, -99.0f));
    ASSERT_TRUE(right.valid);
    EXPECT_GT(right.min_uv.x, 0.5f) << "+X in the world must be the RIGHT of the frame";
}

TEST(OcclusionProject, RectContainsEveryInteriorPointsProjection) {
    const glm::mat4 vp = perspective_vp();
    Rng rng;
    for (int trial = 0; trial < 200; ++trial) {
        const glm::vec3 centre(rng.range(-40.0f, 40.0f), rng.range(-25.0f, 25.0f), rng.range(-200.0f, -20.0f));
        const glm::vec3 half(rng.range(0.2f, 8.0f), rng.range(0.2f, 8.0f), rng.range(0.2f, 8.0f));
        const glm::vec3 lo = centre - half;
        const glm::vec3 hi = centre + half;
        const OcclusionBounds b = project_aabb(vp, lo, hi);
        if (!b.valid) {
            continue; // straddles the near plane; nothing to contain
        }
        for (int sample = 0; sample < 50; ++sample) {
            const glm::vec3 p(rng.range(lo.x, hi.x), rng.range(lo.y, hi.y), rng.range(lo.z, hi.z));
            const glm::vec4 clip = vp * glm::vec4(p, 1.0f);
            ASSERT_GT(clip.w, 0.0f);
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            // Must mirror project_aabb's mapping, Y flip included — this
            // duplicate is the reason a mirrored footprint survived the suite:
            // it agreed with the implementation instead of with the frame.
            // UpInTheWorldIsUpInTheFrame is what pins the direction; this one
            // only pins containment.
            const glm::vec2 uv(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
            constexpr float eps = 1e-5f;
            EXPECT_GE(uv.x, b.min_uv.x - eps);
            EXPECT_LE(uv.x, b.max_uv.x + eps);
            EXPECT_GE(uv.y, b.min_uv.y - eps);
            EXPECT_LE(uv.y, b.max_uv.y + eps);
            // The one that makes testing a single scalar sound: no interior
            // point may be closer than the corner maximum.
            EXPECT_LE(ndc.z, b.nearest_depth + eps);
        }
    }
}

TEST(OcclusionProject, ScreenFractionGrowsWithTheBox) {
    const glm::mat4 vp = perspective_vp();
    const OcclusionBounds small = project_aabb(vp, glm::vec3(-1.0f, -1.0f, -101.0f), glm::vec3(1.0f, 1.0f, -99.0f));
    const OcclusionBounds big = project_aabb(vp, glm::vec3(-40.0f, -40.0f, -101.0f), glm::vec3(40.0f, 40.0f, -99.0f));
    ASSERT_TRUE(small.valid);
    ASSERT_TRUE(big.valid);
    EXPECT_LT(small.screen_fraction(), big.screen_fraction());
    EXPECT_LE(big.screen_fraction(), 1.0f);
}

// ---------------------------------------------------------------------------
// occluded_by_field — the one-sidedness cases
// ---------------------------------------------------------------------------

namespace {

/// A box small enough on screen to pass the fraction gate, at `z`.
OcclusionBounds box_at(const glm::mat4& vp, float z) {
    return project_aabb(vp, glm::vec3(-1.0f, -1.0f, z - 1.0f), glm::vec3(1.0f, 1.0f, z + 1.0f));
}

} // namespace

TEST(OcclusionField, ABoxBehindEveryTexelIsCulled) {
    const glm::mat4 vp = perspective_vp();
    const OcclusionBounds b = box_at(vp, -100.0f);
    ASSERT_TRUE(b.valid);
    Field field(16, 9, 3, b.nearest_depth + 0.1f, vp);
    EXPECT_TRUE(occluded_by_field(field.f, b));
}

TEST(OcclusionField, ABoxInFrontOfEveryTexelIsNotCulled) {
    const glm::mat4 vp = perspective_vp();
    const OcclusionBounds b = box_at(vp, -100.0f);
    Field field(16, 9, 3, b.nearest_depth - 0.1f, vp);
    EXPECT_FALSE(occluded_by_field(field.f, b));
}

TEST(OcclusionField, ABoxAtExactlyTheTexelDepthIsNotCulled) {
    // Strict inequality, deliberately: coplanar is not "behind".
    const glm::mat4 vp = perspective_vp();
    const OcclusionBounds b = box_at(vp, -100.0f);
    Field field(16, 9, 3, b.nearest_depth, vp);
    EXPECT_FALSE(occluded_by_field(field.f, b));
}

TEST(OcclusionField, OneNearerTexelAnywhereInTheFootprintDefeatsTheCull) {
    // A box hidden by every texel but ONE is visible through that one, and the
    // test must say so. This is the property a max-reduced pyramid or a "most
    // texels" heuristic would silently lose.
    const glm::mat4 vp = perspective_vp();
    const OcclusionBounds b = box_at(vp, -100.0f);
    ASSERT_TRUE(b.valid);
    Field field(16, 9, 3, b.nearest_depth + 0.1f, vp);
    ASSERT_TRUE(occluded_by_field(field.f, b));

    // Punch a hole at the rect's centre texel.
    const int cx = static_cast<int>((b.min_uv.x + b.max_uv.x) * 0.5f * static_cast<float>(field.f.mip_w));
    const int cy = static_cast<int>((b.min_uv.y + b.max_uv.y) * 0.5f * static_cast<float>(field.f.mip_h));
    ASSERT_GE(cx, 0);
    ASSERT_LT(cx, field.f.mip_w);
    ASSERT_GE(cy, 0);
    ASSERT_LT(cy, field.f.mip_h);
    field.texels[static_cast<std::size_t>(cy * field.f.mip_w + cx)] = b.nearest_depth - 1.0f;
    EXPECT_FALSE(occluded_by_field(field.f, b));
}

TEST(OcclusionField, AFullScreenBoxIsRefusedByTheFractionGate) {
    // THE OCCLUDER FEEDBACK LOOP: a big occluder that culls itself stops
    // occluding, and the scene oscillates. The gate is the only guard.
    const glm::mat4 vp = perspective_vp();
    const OcclusionBounds b = project_aabb(vp, glm::vec3(-400.0f, -400.0f, -101.0f), glm::vec3(400.0f, 400.0f, -99.0f));
    ASSERT_TRUE(b.valid);
    Field field(16, 9, 3, b.nearest_depth + 10.0f, vp);
    field.f.max_tested_screen_fraction = 0.15f;
    EXPECT_GT(b.screen_fraction(), 0.15f);
    EXPECT_FALSE(occluded_by_field(field.f, b));
    // Raise the gate above the box's coverage and the same box IS culled —
    // which proves the refusal above came from the gate and not from the depth.
    field.f.max_tested_screen_fraction = 1.0f;
    EXPECT_TRUE(occluded_by_field(field.f, b));
}

TEST(OcclusionField, AnUnusableFieldNeverCulls) {
    const glm::mat4 vp = perspective_vp();
    const OcclusionBounds b = box_at(vp, -100.0f);
    Field field(16, 9, 3, b.nearest_depth + 10.0f, vp);
    ASSERT_TRUE(occluded_by_field(field.f, b));

    OcclusionField empty = field.f;
    empty.depth = {};
    EXPECT_FALSE(empty.usable());
    EXPECT_FALSE(occluded_by_field(empty, b));

    OcclusionField teleported = field.f;
    teleported.camera_travel = teleported.max_camera_travel + 1.0f;
    EXPECT_FALSE(teleported.usable());
    EXPECT_FALSE(occluded_by_field(teleported, b));

    OcclusionField spun = field.f;
    spun.camera_rotation_cos = 0.5f;
    EXPECT_FALSE(spun.usable());
    EXPECT_FALSE(occluded_by_field(spun, b));
}

TEST(OcclusionField, ARectEntirelyOffScreenIsNeverCulled) {
    // "No data here" must never read as "occluded".
    const glm::mat4 vp = perspective_vp();
    Field field(16, 9, 3, 10.0f, vp);
    OcclusionBounds b;
    b.valid = true;
    b.nearest_depth = 0.0f;
    b.min_uv = glm::vec2(1.4f, 0.2f);
    b.max_uv = glm::vec2(1.9f, 0.4f);
    EXPECT_FALSE(occluded_by_field(field.f, b));
}

TEST(OcclusionField, TheOversizedLastTexelOfAnOddAxisIsStillRead) {
    // Mip dimensions are NOT screen >> level once an axis is odd: the last
    // destination texel ABSORBS the trailing source texel (render/hzb.hpp).
    // Here the field is 5 wide over an 11-wide screen, i.e. the level-1 shape
    // of an odd axis, and the rect is pinned to the right-hand edge.
    //
    // WHAT THIS DOES *NOT* PROVE, stated because the obvious reading is wrong:
    // it does not discriminate screen-based addressing from `uv * mip_w`. Since
    // `screen < (mip + 1) * 2^level`, the two differ by at most one texel, and
    // the footprint walk dilates by a whole texel on every side — so no input
    // can separate them while that dilation exists. What this case pins is that
    // the trailing, oversized texel of an odd axis is inside the walked range
    // at all, i.e. that the clamp does not stop short of it.
    const glm::mat4 vp = perspective_vp();
    Field field(5, 4, 1, 0.0f, vp);
    field.f.screen_w = 11; // odd: 11 -> 5, last texel covers columns 8..10
    field.f.screen_h = 8;

    // A rect pinned to the right-hand edge of the frame.
    OcclusionBounds b;
    b.valid = true;
    b.nearest_depth = 0.0f;
    b.min_uv = glm::vec2(0.90f, 0.40f);
    b.max_uv = glm::vec2(1.00f, 0.60f);

    // Everything occludes except the last column: the cull must fail.
    for (float& t : field.texels) {
        t = 1.0f;
    }
    for (int y = 0; y < field.f.mip_h; ++y) {
        field.texels[static_cast<std::size_t>(y * field.f.mip_w + (field.f.mip_w - 1))] = -1.0f;
    }
    EXPECT_FALSE(occluded_by_field(field.f, b))
        << "the oversized last texel was not read: the footprint is addressed by screen/2^level, not by mip dims";

    // With the last column occluding too, the same rect IS culled — the check
    // that the failure above came from that column and not from the walk
    // rejecting the rect outright.
    for (int y = 0; y < field.f.mip_h; ++y) {
        field.texels[static_cast<std::size_t>(y * field.f.mip_w + (field.f.mip_w - 1))] = 1.0f;
    }
    EXPECT_TRUE(occluded_by_field(field.f, b));
}

// ---------------------------------------------------------------------------
// occlusion_cull_aabb — the inflation
// ---------------------------------------------------------------------------

TEST(OcclusionCull, CameraTravelInflationCanOnlyReduceCulling) {
    // Monotone in the safe direction, asserted rather than assumed: a wider box
    // has a wider footprint and a nearer `nearest_depth`, both of which make
    // the cull harder. Swept over a range of travels against a field tuned to
    // sit just past the un-inflated box.
    const glm::mat4 vp = perspective_vp();
    const glm::vec3 lo(-1.0f, -1.0f, -101.0f);
    const glm::vec3 hi(1.0f, 1.0f, -99.0f);
    const OcclusionBounds tight = project_aabb(vp, lo, hi);
    ASSERT_TRUE(tight.valid);

    Field field(32, 18, 3, tight.nearest_depth + 0.0015f, vp);
    OcclusionField f = field.f;
    f.camera_travel = 0.0f;
    // Take the screen-fraction guard out of the picture: an inflated box fills
    // the frame long before it wins on depth, and this test is about depth. If
    // the guard were left at 1.0 the sweep would "defeat" the cull by refusing
    // to test at all, which proves nothing about the inflation.
    f.max_tested_screen_fraction = 1e9f;
    ASSERT_TRUE(occlusion_cull_aabb(f, lo, hi)) << "the un-inflated box must be culled for the sweep to mean anything";

    // The sweep has to reach the crossing, and where that is depends on the
    // depth distribution, not on intuition about world units. Reversed-Z with
    // near = 0.5 compresses z = -99..-59 into 0.0014 of depth, so this box
    // needs ~45 units of travel to clear a 0.0015 slack. Sweeping to 40 (the
    // original bound) stopped one crossing short and read as a failure of the
    // inflation. The cap stays under 99, past which the inflated box would
    // swallow the camera and project_aabb would have a different opinion.
    bool still_culled = true;
    for (float travel = 0.0f; travel <= 60.0f; travel += 0.5f) {
        f.camera_travel = travel;
        const bool culled = occlusion_cull_aabb(f, lo, hi);
        if (!culled) {
            still_culled = false;
        }
        // Once it stops being culled it must never start again: monotone.
        EXPECT_TRUE(still_culled || !culled) << "travel " << travel << " re-enabled a cull that inflation had refused";
    }
    EXPECT_FALSE(still_culled) << "enough travel must eventually defeat this cull";
}

TEST(OcclusionCull, DepthSlackCanOnlyReduceCulling) {
    const glm::mat4 vp = perspective_vp();
    const glm::vec3 lo(-1.0f, -1.0f, -101.0f);
    const glm::vec3 hi(1.0f, 1.0f, -99.0f);
    const OcclusionBounds tight = project_aabb(vp, lo, hi);
    Field field(32, 18, 3, tight.nearest_depth + 0.001f, vp);
    OcclusionField f = field.f;
    EXPECT_TRUE(occlusion_cull_aabb(f, lo, hi));
    f.depth_slack = 0.01f;
    EXPECT_FALSE(occlusion_cull_aabb(f, lo, hi));
}

TEST(OcclusionCull, NoBoxIsCulledUnlessItIsSTRICTLYBeyondEveryTexel) {
    // The blunt end of one-sidedness, swept over random shapes and placements:
    // a field at a fixed device depth may never cull a box whose nearest point
    // is at or beyond that depth. Stated as a comparison between two numbers
    // rather than as "in front", because this fixture's projection is standard
    // glm RH_ZO while the engine's is reversed — the predicate is what is under
    // test, not the depth convention.
    const glm::mat4 vp = perspective_vp();
    constexpr float wall = 0.30f;
    Field field(40, 24, 2, wall, vp);
    Rng rng;
    int tested = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const glm::vec3 centre(rng.range(-30.0f, 30.0f), rng.range(-20.0f, 20.0f), rng.range(-300.0f, -5.0f));
        const glm::vec3 half(rng.range(0.1f, 6.0f), rng.range(0.1f, 6.0f), rng.range(0.1f, 6.0f));
        const glm::vec3 lo = centre - half;
        const glm::vec3 hi = centre + half;
        const OcclusionBounds b = project_aabb(vp, lo, hi);
        if (!b.valid || b.nearest_depth < wall) {
            continue;
        }
        ++tested;
        EXPECT_FALSE(occluded_by_field(field.f, b))
            << "culled a box whose nearest_depth (" << b.nearest_depth << ") is not strictly beyond the field ("
            << wall << ")";
    }
    EXPECT_GT(tested, 100) << "the sweep produced too few qualifying boxes to mean anything";
}
