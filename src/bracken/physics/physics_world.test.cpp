// PhysicsWorld tests — CPU-only (no GPU/display). Each fixture owns its own
// JobSystem pool (tests never rely on the singleton being initialized) and
// exercises the Jolt-on-bracken job bridge implicitly: every
// PhysicsWorld::update fans its step jobs out across the caller-owned pool.

#include "bracken/physics/physics_world.hpp"

#include "bracken/core/jobs/job_system.hpp"

#include <gtest/gtest.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

using bracken::physics::BodyDesc;
using bracken::physics::BodyId;
using bracken::physics::BodyType;
using bracken::physics::INVALID_BODY_ID;
using bracken::physics::PhysicsConfig;
using bracken::physics::PhysicsWorld;

constexpr float k_dt = 1.0f / 60.0f;

class PhysicsWorldTest : public ::testing::Test {
protected:
    void SetUp() override {
        js_.init(4);
        ASSERT_TRUE(world_.init(PhysicsConfig{}, js_));
    }

    void TearDown() override {
        world_.shutdown();
        js_.shutdown();
    }

    void step(int n) {
        for (int i = 0; i < n; ++i) {
            world_.update(k_dt);
        }
    }

    BodyId add_floor() {
        BodyDesc desc;
        desc.type = BodyType::Static;
        desc.position = {0.0f, -0.5f, 0.0f}; // top surface at y = 0
        return world_.add_box(desc, {50.0f, 0.5f, 50.0f});
    }

    bracken::core::jobs::JobSystem js_; // caller-owned pool
    PhysicsWorld world_;
};

TEST_F(PhysicsWorldTest, InitRejectsDoubleInitAndBadConfig) {
    EXPECT_FALSE(world_.init(PhysicsConfig{}, js_)); // already initialized

    PhysicsWorld other;
    PhysicsConfig bad;
    bad.fixed_dt = 0.0f;
    EXPECT_FALSE(other.init(bad, js_));
    EXPECT_FALSE(other.is_initialized());
}

TEST_F(PhysicsWorldTest, SphereFreeFallMatchesGravity) {
    BodyDesc desc;
    desc.type = BodyType::Dynamic;
    desc.position = {0.0f, 100.0f, 0.0f};
    const BodyId id = world_.add_sphere(desc, 0.5f);
    ASSERT_NE(id, INVALID_BODY_ID);

    step(60); // 1 second

    // Semi-implicit Euler + Jolt's default linear damping: expect ~g*t^2/2.
    const float drop = 100.0f - world_.position(id).y;
    EXPECT_NEAR(drop, 0.5f * 9.81f, 0.35f);
    EXPECT_LT(world_.linear_velocity(id).y, -9.0f);
    EXPECT_TRUE(world_.is_active(id));

    // Removed bodies answer with identity values.
    world_.remove_body(id);
    EXPECT_EQ(world_.position(id), glm::vec3(0.0f));
    EXPECT_FALSE(world_.is_active(id));
}

TEST_F(PhysicsWorldTest, RestitutionBouncesToSaneFraction) {
    ASSERT_NE(add_floor(), INVALID_BODY_ID);

    BodyDesc desc;
    desc.type = BodyType::Dynamic;
    desc.position = {0.0f, 5.5f, 0.0f}; // rest height 0.5 -> drop height 5.0
    desc.restitution = 0.8f;
    const BodyId id = world_.add_sphere(desc, 0.5f);
    ASSERT_NE(id, INVALID_BODY_ID);

    bool bounced = false;
    float peak_y = 0.0f;
    float prev_vy = 0.0f;
    for (int i = 0; i < 900; ++i) {
        world_.update(k_dt);
        const float vy = world_.linear_velocity(id).y;
        if (!bounced && prev_vy < -1.0f && vy > 0.0f) {
            bounced = true; // impact -> rebound
        }
        if (bounced) {
            peak_y = std::max(peak_y, world_.position(id).y);
            if (vy < 0.0f && peak_y > 0.6f) {
                break; // past the first rebound apex
            }
        }
        prev_vy = vy;
    }
    ASSERT_TRUE(bounced);

    // Ideal rebound fraction = restitution^2 = 0.64 of the drop height;
    // allow generous slack for damping and contact softness.
    const float rebound_fraction = (peak_y - 0.5f) / 5.0f;
    EXPECT_GT(rebound_fraction, 0.35f);
    EXPECT_LT(rebound_fraction, 0.9f);
}

TEST_F(PhysicsWorldTest, BoxStackSettlesAndSleeps) {
    ASSERT_NE(add_floor(), INVALID_BODY_ID);

    std::vector<BodyId> boxes;
    for (int i = 0; i < 3; ++i) {
        BodyDesc desc;
        desc.type = BodyType::Dynamic;
        desc.position = {0.0f, 0.5f + 1.02f * static_cast<float>(i), 0.0f};
        boxes.push_back(world_.add_box(desc, {0.5f, 0.5f, 0.5f}));
        ASSERT_NE(boxes.back(), INVALID_BODY_ID);
    }

    bool all_asleep = false;
    for (int i = 0; i < 1200 && !all_asleep; ++i) {
        world_.update(k_dt);
        all_asleep = std::none_of(boxes.begin(), boxes.end(),
                                  [&](BodyId b) { return world_.is_active(b); });
    }
    EXPECT_TRUE(all_asleep) << "stack did not fall asleep within 20s of sim time";

    // Still stacked (top box roughly three half-extents up), not toppled.
    EXPECT_GT(world_.position(boxes[2]).y, 2.0f);
    EXPECT_LT(glm::length(glm::vec2(world_.position(boxes[2]).x, world_.position(boxes[2]).z)),
              0.5f);
}

TEST_F(PhysicsWorldTest, HeightfieldBowlRayAndSettling) {
    constexpr std::uint32_t k_res = 64;
    const float world_size = 32.0f;
    const glm::vec3 origin{-16.0f, 0.0f, -16.0f};
    const float cell = world_size / static_cast<float>(k_res - 1);

    // Synthetic bowl: h(x,z) = 0.02 * (x^2 + z^2), minimum ~0 at world
    // origin.
    std::vector<float> heights(static_cast<std::size_t>(k_res) * k_res);
    for (std::uint32_t z = 0; z < k_res; ++z) {
        for (std::uint32_t x = 0; x < k_res; ++x) {
            const float wx = origin.x + static_cast<float>(x) * cell;
            const float wz = origin.z + static_cast<float>(z) * cell;
            heights[static_cast<std::size_t>(z) * k_res + x] = 0.02f * (wx * wx + wz * wz);
        }
    }
    const BodyId terrain = world_.add_heightfield(origin, world_size, heights, k_res);
    ASSERT_NE(terrain, INVALID_BODY_ID);

    // Downward ray near the bowl center hits the terrain close to y = 0.
    const auto hit = world_.cast_ray({0.1f, 10.0f, 0.1f}, {0.0f, -20.0f, 0.0f});
    ASSERT_TRUE(hit.hit);
    EXPECT_EQ(hit.body, terrain);
    EXPECT_NEAR(hit.position.y, 0.0f, 0.25f);
    EXPECT_NEAR(hit.fraction, 0.5f, 0.05f);
    EXPECT_GT(hit.normal.y, 0.9f);

    // A dropped sphere rolls inward and stays on the surface.
    BodyDesc desc;
    desc.type = BodyType::Dynamic;
    desc.position = {6.0f, 4.0f, 3.0f};
    const BodyId ball = world_.add_sphere(desc, 0.5f);
    ASSERT_NE(ball, INVALID_BODY_ID);

    const float start_radius = glm::length(glm::vec2(6.0f, 3.0f));
    step(900); // 15 s

    const glm::vec3 pos = world_.position(ball);
    const float end_radius = glm::length(glm::vec2(pos.x, pos.z));
    EXPECT_LT(end_radius, start_radius - 0.5f) << "ball did not roll toward the bowl bottom";

    // Center height tracks surface height + radius while resting/rolling.
    const float surface = 0.02f * (pos.x * pos.x + pos.z * pos.z);
    EXPECT_NEAR(pos.y, surface + 0.5f, 0.5f);
}

TEST_F(PhysicsWorldTest, KinematicCapsulePushesRestingSphere) {
    ASSERT_NE(add_floor(), INVALID_BODY_ID);

    BodyDesc sphere_desc;
    sphere_desc.type = BodyType::Dynamic;
    sphere_desc.position = {0.0f, 0.5f, 0.0f};
    const BodyId sphere = world_.add_sphere(sphere_desc, 0.5f);
    ASSERT_NE(sphere, INVALID_BODY_ID);
    step(120); // let it settle (and possibly sleep)

    BodyDesc capsule_desc;
    capsule_desc.type = BodyType::Kinematic;
    capsule_desc.position = {-3.0f, 1.0f, 0.0f}; // bottom touches the floor
    const BodyId capsule = world_.add_capsule(capsule_desc, 0.5f, 0.5f);
    ASSERT_NE(capsule, INVALID_BODY_ID);

    // Drive the capsule through the sphere's position over 2 seconds.
    const glm::vec3 start{-3.0f, 1.0f, 0.0f};
    const glm::vec3 end{1.0f, 1.0f, 0.0f};
    const glm::quat rot{1.0f, 0.0f, 0.0f, 0.0f};
    const int n = 120;
    for (int i = 1; i <= n; ++i) {
        const glm::vec3 target =
            glm::mix(start, end, static_cast<float>(i) / static_cast<float>(n));
        world_.move_kinematic(capsule, target, rot, k_dt);
        world_.update(k_dt);
    }

    EXPECT_NEAR(world_.position(capsule).x, end.x, 0.02f); // kinematic reached its target
    EXPECT_GT(world_.position(sphere).x, 0.5f) << "sphere was not pushed";
}

TEST_F(PhysicsWorldTest, MoveKinematicOutrunningFixedStepDoesNotDiverge) {
    // Regression: Jolt turns move_kinematic into an UNCLAMPED velocity of
    // (target - position) / dt, consumed in fixed_dt-sized steps. A host
    // whose frame rate outruns the fixed step (frame dt of 1 ms against the
    // 60 Hz step) used to make every step overshoot the target by
    // fixed_dt/dt, oscillating to inf/NaN within a second of sim time. The
    // window is now clamped up to the fixed step, so the body reaches the
    // target and stays there.
    BodyDesc desc;
    desc.type = BodyType::Kinematic;
    desc.position = {0.0f, 5.0f, 0.0f};
    const BodyId id = world_.add_capsule(desc, 0.5f, 0.5f);
    ASSERT_NE(id, INVALID_BODY_ID);

    const glm::vec3 target{1.0f, 5.0f, -2.0f};
    const glm::quat rot{1.0f, 0.0f, 0.0f, 0.0f};
    constexpr float k_frame_dt = 1.0f / 1000.0f;
    for (int i = 0; i < 1200; ++i) { // 1.2 s of sim, ~72 fixed steps
        world_.move_kinematic(id, target, rot, k_frame_dt);
        world_.update(k_frame_dt);
    }

    const glm::vec3 pos = world_.position(id);
    ASSERT_TRUE(std::isfinite(pos.x) && std::isfinite(pos.y) && std::isfinite(pos.z))
        << "kinematic position diverged: " << pos.x << ", " << pos.y << ", " << pos.z;
    EXPECT_NEAR(pos.x, target.x, 0.01f);
    EXPECT_NEAR(pos.y, target.y, 0.01f);
    EXPECT_NEAR(pos.z, target.z, 0.01f);
}

TEST_F(PhysicsWorldTest, ImpulseAndVelocityWakeAndMoveBody) {
    ASSERT_NE(add_floor(), INVALID_BODY_ID);

    BodyDesc desc;
    desc.type = BodyType::Dynamic;
    desc.position = {0.0f, 0.5f, 0.0f};
    desc.mass = 2.0f;
    const BodyId id = world_.add_sphere(desc, 0.5f);
    ASSERT_NE(id, INVALID_BODY_ID);
    step(240); // settle until asleep
    ASSERT_FALSE(world_.is_active(id));

    world_.apply_impulse(id, {0.0f, 12.0f, 0.0f}); // dv = 6 m/s upward
    EXPECT_TRUE(world_.is_active(id));
    step(1);
    EXPECT_GT(world_.linear_velocity(id).y, 3.0f);

    // set_linear_velocity on a sleeping body wakes it too.
    step(600);
    ASSERT_FALSE(world_.is_active(id));
    world_.set_linear_velocity(id, {2.0f, 0.0f, 0.0f});
    EXPECT_TRUE(world_.is_active(id));
    step(30);
    EXPECT_GT(world_.position(id).x, 0.2f);
}

TEST_F(PhysicsWorldTest, DeterministicAcrossWorldsOnSamePool) {
    const auto build = [](PhysicsWorld& w, std::vector<BodyId>& ids) {
        BodyDesc floor;
        floor.type = BodyType::Static;
        floor.position = {0.0f, -0.5f, 0.0f};
        ids.push_back(w.add_box(floor, {20.0f, 0.5f, 20.0f}));
        for (int i = 0; i < 4; ++i) {
            const float fi = static_cast<float>(i);
            BodyDesc d;
            d.type = BodyType::Dynamic;
            d.position = {-3.0f + 2.0f * fi, 2.0f + 0.7f * fi, 0.3f * fi};
            d.restitution = 0.3f;
            d.friction = 0.4f;
            d.mass = 1.0f + fi;
            ids.push_back(w.add_sphere(d, 0.4f));

            BodyDesc b = d;
            b.position.z += 2.0f;
            b.rotation = glm::normalize(glm::quat(1.0f, 0.1f * fi, 0.2f, 0.0f));
            ids.push_back(w.add_box(b, {0.4f, 0.3f, 0.5f}));
        }
    };

    PhysicsWorld a;
    PhysicsWorld b;
    ASSERT_TRUE(a.init(PhysicsConfig{}, js_));
    ASSERT_TRUE(b.init(PhysicsConfig{}, js_));

    std::vector<BodyId> ids_a;
    std::vector<BodyId> ids_b;
    build(a, ids_a);
    build(b, ids_b);
    ASSERT_EQ(ids_a, ids_b); // identical creation order -> identical ids
    ASSERT_EQ(std::count(ids_a.begin(), ids_a.end(), INVALID_BODY_ID), 0);

    for (int s = 0; s < 120; ++s) {
        a.update(k_dt);
        b.update(k_dt);
    }

    // Bit-identical, not approximately equal: Jolt guarantees determinism
    // for identical inputs on the same binary, independent of thread count.
    for (std::size_t i = 0; i < ids_a.size(); ++i) {
        const glm::vec3 pa = a.position(ids_a[i]);
        const glm::vec3 pb = b.position(ids_b[i]);
        EXPECT_EQ(pa.x, pb.x) << "body " << i;
        EXPECT_EQ(pa.y, pb.y) << "body " << i;
        EXPECT_EQ(pa.z, pb.z) << "body " << i;

        const glm::quat ra = a.rotation(ids_a[i]);
        const glm::quat rb = b.rotation(ids_b[i]);
        EXPECT_EQ(ra.w, rb.w) << "body " << i;
        EXPECT_EQ(ra.x, rb.x) << "body " << i;
        EXPECT_EQ(ra.y, rb.y) << "body " << i;
        EXPECT_EQ(ra.z, rb.z) << "body " << i;
    }
}

TEST_F(PhysicsWorldTest, CollideCapsuleReportsStaticOverlapWithPushNormal) {
    BodyDesc desc;
    desc.type = BodyType::Static;
    desc.position = {2.0f, 1.0f, 0.0f}; // -x face at x = 1
    ASSERT_NE(world_.add_box(desc, {1.0f, 1.0f, 1.0f}), INVALID_BODY_ID);

    // Capsule surface reaches x = 0.8 + 0.5 = 1.3 -> 0.3 into the box face.
    PhysicsWorld::ShapeContact contacts[4];
    const std::size_t n =
        world_.collide_capsule({0.8f, 1.0f, 0.0f}, 0.5f, 0.5f, contacts);
    ASSERT_EQ(n, 1u);
    EXPECT_NEAR(contacts[0].normal.x, -1.0f, 1e-3f); // pushes the capsule out along -x
    EXPECT_NEAR(contacts[0].normal.y, 0.0f, 1e-3f);
    EXPECT_NEAR(contacts[0].normal.z, 0.0f, 1e-3f);
    EXPECT_NEAR(contacts[0].penetration, 0.3f, 0.02f);

    // Separated capsule: no contacts.
    EXPECT_EQ(world_.collide_capsule({-1.0f, 1.0f, 0.0f}, 0.5f, 0.5f, contacts), 0u);
}

TEST_F(PhysicsWorldTest, CollideCapsuleFiltersMovingAndIgnoredBodies) {
    BodyDesc dyn;
    dyn.type = BodyType::Dynamic;
    dyn.position = {0.0f, 1.0f, 0.0f};
    const BodyId sphere = world_.add_sphere(dyn, 0.5f);
    ASSERT_NE(sphere, INVALID_BODY_ID);

    PhysicsWorld::ShapeContact contacts[4];
    // A dynamic body dead-center in the capsule is invisible to the default
    // static-only query...
    EXPECT_EQ(world_.collide_capsule({0.0f, 1.0f, 0.0f}, 0.5f, 0.5f, contacts), 0u);
    // ...reported with include_moving...
    ASSERT_EQ(world_.collide_capsule({0.0f, 1.0f, 0.0f}, 0.5f, 0.5f, contacts, true), 1u);
    EXPECT_EQ(contacts[0].body, sphere);
    // ...and excluded again via ignore_body.
    EXPECT_EQ(world_.collide_capsule({0.0f, 1.0f, 0.0f}, 0.5f, 0.5f, contacts, true, sphere), 0u);
}

TEST_F(PhysicsWorldTest, CollideCapsuleTruncatesDeepestFirst) {
    BodyDesc shallow;
    shallow.type = BodyType::Static;
    shallow.position = {1.4f, 1.0f, 0.0f}; // capsule overlaps its face by 0.1
    ASSERT_NE(world_.add_box(shallow, {1.0f, 1.0f, 1.0f}), INVALID_BODY_ID);

    BodyDesc deep;
    deep.type = BodyType::Static;
    deep.position = {-1.0f, 1.0f, 0.0f}; // capsule overlaps its face by 0.5
    const BodyId deep_id = world_.add_box(deep, {1.0f, 1.0f, 1.0f});
    ASSERT_NE(deep_id, INVALID_BODY_ID);

    // A one-slot span keeps the deepest contact, not an arbitrary one.
    PhysicsWorld::ShapeContact one[1];
    ASSERT_EQ(world_.collide_capsule({0.0f, 1.0f, 0.0f}, 0.5f, 0.5f, one), 1u);
    EXPECT_EQ(one[0].body, deep_id);
    EXPECT_NEAR(one[0].penetration, 0.5f, 0.02f);
}

} // namespace
