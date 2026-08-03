#include "bracken/render/raytracing/capsule_occluder.hpp"

#include <gtest/gtest.h>

#include <array>
#include <glm/gtc/constants.hpp>

namespace {

using bracken::render::rt::capsule_occluded;
using bracken::render::rt::capsules_occluded;
using bracken::render::rt::ShadowCapsule;

/// A vertical capsule: axis from y=0 to y=100 at the origin, radius 10.
constexpr ShadowCapsule k_pillar{{0.0f, 0.0f, 0.0f}, 10.0f, {0.0f, 100.0f, 0.0f}, 0.0f};

constexpr float k_far = 1e6f;

bool hit(const glm::vec3& o, const glm::vec3& d, const ShadowCapsule& c = k_pillar,
         float t_max = k_far) {
    return capsule_occluded(o, glm::normalize(d), 0.0f, t_max, c);
}

// ---------------------------------------------------------------------------
// The body: the cylindrical middle.
// ---------------------------------------------------------------------------

TEST(CapsuleOccluder, HitsTheCylinderBroadside) {
    // Straight at the axis, halfway up.
    EXPECT_TRUE(hit({100.0f, 50.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}));
    EXPECT_TRUE(hit({0.0f, 50.0f, -100.0f}, {0.0f, 0.0f, 1.0f}));
}

TEST(CapsuleOccluder, MissesBesideTheCylinder) {
    // Parallel to the axis-facing ray but offset past the radius.
    EXPECT_FALSE(hit({100.0f, 50.0f, 20.0f}, {-1.0f, 0.0f, 0.0f}));
    // Aimed away from the capsule entirely.
    EXPECT_FALSE(hit({100.0f, 50.0f, 0.0f}, {1.0f, 0.0f, 0.0f}));
}

TEST(CapsuleOccluder, MissesAboveAndBelowTheSegmentBand) {
    // Broadside but well past either cap: outside the segment AND outside both
    // cap spheres, so neither part of the test may claim it.
    EXPECT_FALSE(hit({100.0f, 130.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}));
    EXPECT_FALSE(hit({100.0f, -30.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}));
}

TEST(CapsuleOccluder, GrazeInsideAndOutsideTheRadiusDisagree) {
    // The discriminating pair: two parallel rays either side of the radius.
    EXPECT_TRUE(hit({100.0f, 50.0f, 9.5f}, {-1.0f, 0.0f, 0.0f}));
    EXPECT_FALSE(hit({100.0f, 50.0f, 10.5f}, {-1.0f, 0.0f, 0.0f}));
}

// ---------------------------------------------------------------------------
// The caps: the hemispheres the cylinder test cannot see.
// ---------------------------------------------------------------------------

TEST(CapsuleOccluder, HitsTheCapSpheres) {
    // Straight down onto the top cap, from above the segment's end.
    EXPECT_TRUE(hit({0.0f, 200.0f, 0.0f}, {0.0f, -1.0f, 0.0f}));
    // Up into the bottom cap.
    EXPECT_TRUE(hit({0.0f, -50.0f, 0.0f}, {0.0f, 1.0f, 0.0f}));
    // Broadside just above the top joint — inside the cap sphere, outside the
    // cylinder band. This is precisely the case a body-only test drops.
    EXPECT_TRUE(hit({100.0f, 105.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}));
}

TEST(CapsuleOccluder, MissesJustPastTheCapSphere) {
    EXPECT_FALSE(hit({100.0f, 111.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}));
}

// ---------------------------------------------------------------------------
// Degenerate geometry.
// ---------------------------------------------------------------------------

TEST(CapsuleOccluder, DegenerateSegmentIsASphere) {
    // p0 == p1: baba is zero, the cylinder quadratic vanishes entirely and the
    // caps have to carry the whole answer.
    const ShadowCapsule ball{{0.0f, 50.0f, 0.0f}, 10.0f, {0.0f, 50.0f, 0.0f}, 0.0f};
    EXPECT_TRUE(hit({100.0f, 50.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, ball));
    EXPECT_TRUE(hit({100.0f, 50.0f, 9.0f}, {-1.0f, 0.0f, 0.0f}, ball));
    EXPECT_FALSE(hit({100.0f, 50.0f, 11.0f}, {-1.0f, 0.0f, 0.0f}, ball));
    // A sphere has no extent along the (nonexistent) axis, unlike the pillar.
    EXPECT_FALSE(hit({100.0f, 80.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, ball));
}

TEST(CapsuleOccluder, RayParallelToTheAxis) {
    // a == 0: the quadratic is degenerate and the caps must answer. Radially
    // inside -> enters the bottom cap; radially outside -> nothing.
    EXPECT_TRUE(hit({5.0f, -50.0f, 0.0f}, {0.0f, 1.0f, 0.0f}));
    EXPECT_FALSE(hit({15.0f, -50.0f, 0.0f}, {0.0f, 1.0f, 0.0f}));
    // Parallel and aimed away from the capsule.
    EXPECT_FALSE(hit({5.0f, -50.0f, 0.0f}, {0.0f, -1.0f, 0.0f}));
}

TEST(CapsuleOccluder, ZeroRadiusOccludesNothing) {
    const ShadowCapsule thin{{0.0f, 0.0f, 0.0f}, 0.0f, {0.0f, 100.0f, 0.0f}, 0.0f};
    EXPECT_FALSE(hit({100.0f, 50.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, thin));
}

// ---------------------------------------------------------------------------
// Inside-rejection: the self-shadow contract.
// ---------------------------------------------------------------------------

TEST(CapsuleOccluder, OriginInsideIsNotOccluded) {
    // On the axis, in the middle of the body.
    EXPECT_FALSE(hit({0.0f, 50.0f, 0.0f}, {1.0f, 0.0f, 0.0f}));
    EXPECT_FALSE(hit({0.0f, 50.0f, 0.0f}, {0.0f, 1.0f, 0.0f}));
    // Just inside the surface — the case a character's own chest pixel is in.
    EXPECT_FALSE(hit({9.0f, 50.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}));
    // Inside a cap.
    EXPECT_FALSE(hit({0.0f, 105.0f, 0.0f}, {0.0f, -1.0f, 0.0f}));
    // But a point just OUTSIDE the surface still sees it.
    EXPECT_TRUE(hit({11.0f, 50.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}));
}

TEST(CapsuleOccluder, InsideOneCapsuleStillTestsTheOthers) {
    // The rejection is per capsule, which is what lets an arm shadow a chest
    // that lies inside the torso proxy.
    const std::array<ShadowCapsule, 2> set{
        k_pillar,
        ShadowCapsule{{40.0f, 40.0f, 0.0f}, 8.0f, {40.0f, 60.0f, 0.0f}, 0.0f},
    };
    const glm::vec3 inside_pillar{0.0f, 50.0f, 0.0f};
    EXPECT_FALSE(capsule_occluded(inside_pillar, {1.0f, 0.0f, 0.0f}, 0.0f, k_far, set[0]));
    EXPECT_TRUE(capsules_occluded(set, inside_pillar, {1.0f, 0.0f, 0.0f}, 0.0f, k_far));
}

// ---------------------------------------------------------------------------
// The t window.
// ---------------------------------------------------------------------------

TEST(CapsuleOccluder, TMaxClampsTheQuery) {
    // The capsule surface is 90 units away; a shorter ray must not reach it.
    const glm::vec3 o{100.0f, 50.0f, 0.0f};
    const glm::vec3 d{-1.0f, 0.0f, 0.0f};
    EXPECT_TRUE(capsule_occluded(o, d, 0.0f, 95.0f, k_pillar));
    EXPECT_FALSE(capsule_occluded(o, d, 0.0f, 85.0f, k_pillar));
}

TEST(CapsuleOccluder, TMinSkipsNearHits) {
    const glm::vec3 o{100.0f, 50.0f, 0.0f};
    const glm::vec3 d{-1.0f, 0.0f, 0.0f};
    // The whole capsule lives in t = [90, 110]; a t_min past it finds nothing.
    EXPECT_TRUE(capsule_occluded(o, d, 0.0f, k_far, k_pillar));
    EXPECT_FALSE(capsule_occluded(o, d, 120.0f, k_far, k_pillar));
    // A t_min between the two roots still finds the FAR wall, which is what
    // testing both cylinder roots buys.
    EXPECT_TRUE(capsule_occluded(o, d, 100.0f, k_far, k_pillar));
}

TEST(CapsuleOccluder, NonUnitDirectionScalesTConsistently) {
    // Traversal hands model-space rays whose direction may not be unit length;
    // t must stay expressed in units of that direction.
    const glm::vec3 o{100.0f, 50.0f, 0.0f};
    const glm::vec3 half{-0.5f, 0.0f, 0.0f}; // half speed -> twice the t
    EXPECT_TRUE(capsule_occluded(o, half, 0.0f, 190.0f, k_pillar));
    EXPECT_FALSE(capsule_occluded(o, half, 0.0f, 170.0f, k_pillar));
}

// ---------------------------------------------------------------------------
// The set query.
// ---------------------------------------------------------------------------

TEST(CapsuleOccluder, EmptySetOccludesNothing) {
    EXPECT_FALSE(capsules_occluded({}, {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 0.0f, k_far));
}

TEST(CapsuleOccluder, SetIsTheUnionOfItsMembers) {
    const std::array<ShadowCapsule, 3> limbs{
        ShadowCapsule{{-20.0f, 0.0f, 0.0f}, 5.0f, {-20.0f, 90.0f, 0.0f}, 0.0f},
        ShadowCapsule{{20.0f, 0.0f, 0.0f}, 5.0f, {20.0f, 90.0f, 0.0f}, 0.0f},
        ShadowCapsule{{0.0f, 150.0f, 0.0f}, 9.0f, {0.0f, 150.0f, 0.0f}, 0.0f},
    };
    const glm::vec3 d{0.0f, 0.0f, 1.0f};
    EXPECT_TRUE(capsules_occluded(limbs, {-20.0f, 45.0f, -100.0f}, d, 0.0f, k_far));
    EXPECT_TRUE(capsules_occluded(limbs, {20.0f, 45.0f, -100.0f}, d, 0.0f, k_far));
    EXPECT_TRUE(capsules_occluded(limbs, {0.0f, 150.0f, -100.0f}, d, 0.0f, k_far));
    // The gap between the legs, below the head: a union, not a bounding volume.
    EXPECT_FALSE(capsules_occluded(limbs, {0.0f, 45.0f, -100.0f}, d, 0.0f, k_far));
}

// ---------------------------------------------------------------------------
// A capsule that MOVES produces a shadow that moves — the whole point of the
// feature, expressed as a test rather than left to the screenshot.
// ---------------------------------------------------------------------------

TEST(CapsuleOccluder, AnimatingTheSegmentMovesTheOcclusion) {
    // A "limb" swinging about its top joint, sampled at two poses. A ground
    // point under the swung-forward pose is occluded in one and lit in the
    // other; the bind-pose BLAS this feature replaces could not tell them
    // apart at all.
    auto limb = [](float angle) {
        const glm::vec3 root{0.0f, 90.0f, 0.0f};
        const glm::vec3 tip = root + 80.0f * glm::vec3(0.0f, -std::cos(angle), std::sin(angle));
        return ShadowCapsule{root, 6.0f, tip, 0.0f};
    };
    const glm::vec3 sun_up{0.0f, 1.0f, 0.0f}; // ray toward a sun straight overhead
    const glm::vec3 under_swing{0.0f, 0.0f, 60.0f};

    EXPECT_TRUE(capsule_occluded(under_swing, sun_up, 0.0f, k_far, limb(glm::half_pi<float>())));
    EXPECT_FALSE(capsule_occluded(under_swing, sun_up, 0.0f, k_far, limb(0.0f)));
}

} // namespace
