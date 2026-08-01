#include <gtest/gtest.h>

#include "bracken/physics/character_controller.hpp"

#include "bracken/core/jobs/job_system.hpp"

#include <cmath>

using namespace bracken::physics;

namespace {

constexpr float k_dt = 1.0f / 60.0f;

/** A world with a large flat floor at y = 0.
 *
 *  Owns its own JobSystem rather than using the singleton: the test binary
 *  never brings the global pool up, and PhysicsWorld::init needs a live one.
 *  Mirrors PhysicsWorldTest's fixture. */
struct FloorWorld {
    bracken::core::jobs::JobSystem jobs;
    PhysicsWorld world;
    bool ok = false;

    FloorWorld() {
        jobs.init(4);
        ok = world.init(PhysicsConfig{}, jobs);
        if (ok) {
            BodyDesc floor;
            floor.position = {0.0f, -50.0f, 0.0f};
            floor.type = BodyType::Static;
            (void)world.add_box(floor, {5000.0f, 50.0f, 5000.0f});
            world.update(1.0f / 60.0f); // build the broadphase
        }
    }
    ~FloorWorld() {
        world.shutdown();
        jobs.shutdown();
    }
    FloorWorld(const FloorWorld&) = delete;
    FloorWorld& operator=(const FloorWorld&) = delete;

    /** Add an axis-aligned block whose TOP sits at `top_y`. */
    void add_block(const glm::vec3& center_xz, float top_y, const glm::vec3& half) {
        BodyDesc d;
        d.position = {center_xz.x, top_y - half.y, center_xz.z};
        d.type = BodyType::Static;
        (void)world.add_box(d, half);
    }

    /** Step once so newly added static bodies are in the broadphase. Without
     *  this, collide_capsule finds nothing and a character walks straight
     *  through every wall — which is exactly how this test first failed. */
    void settle() { world.update(1.0f / 60.0f); }
};

CharacterConfig test_config() {
    CharacterConfig c;
    c.radius = 25.0f;
    c.stand_height = 150.0f;
    c.crouch_height = 80.0f;
    c.step_height = 40.0f;
    c.max_slope_degrees = 50.0f;
    c.gravity = -2000.0f;
    c.jump_speed = 800.0f;
    return c;
}

/** Step the controller for `seconds`, holding one input. */
void simulate(CharacterController& cc, const PhysicsWorld& w, const CharacterInput& in,
              float seconds) {
    const int steps = static_cast<int>(seconds / k_dt);
    for (int i = 0; i < steps; ++i) {
        cc.update(w, in, k_dt);
    }
}

} // namespace

TEST(CharacterController, FallsAndLandsOnTheFloor) {
    FloorWorld fw;
    ASSERT_TRUE(fw.ok);

    CharacterController cc(test_config());
    cc.set_position({0.0f, 500.0f, 0.0f});
    EXPECT_FALSE(cc.state().grounded);

    simulate(cc, fw.world, {}, 2.0f);

    EXPECT_TRUE(cc.state().grounded);
    EXPECT_NEAR(cc.state().position.y, 0.0f, 2.0f) << "feet should rest on the floor";
    EXPECT_NEAR(cc.state().velocity.y, 0.0f, 1.0f) << "landing must kill downward velocity";
}

TEST(CharacterController, StandsStillWhenGroundedWithNoInput) {
    FloorWorld fw;
    ASSERT_TRUE(fw.ok);
    CharacterController cc(test_config());
    cc.set_position({0.0f, 10.0f, 0.0f});
    simulate(cc, fw.world, {}, 1.0f);

    const glm::vec3 settled = cc.state().position;
    simulate(cc, fw.world, {}, 1.0f);
    EXPECT_NEAR(cc.state().position.x, settled.x, 0.5f);
    EXPECT_NEAR(cc.state().position.z, settled.z, 0.5f);
    EXPECT_NEAR(cc.state().position.y, settled.y, 0.5f) << "a grounded character must not sink";
}

TEST(CharacterController, WalksAtTheRequestedSpeed) {
    FloorWorld fw;
    ASSERT_TRUE(fw.ok);
    CharacterController cc(test_config());
    cc.set_position({0.0f, 10.0f, 0.0f});
    simulate(cc, fw.world, {}, 0.5f); // settle

    const glm::vec3 start = cc.state().position;
    CharacterInput in;
    in.desired_velocity = {200.0f, 0.0f, 0.0f};
    simulate(cc, fw.world, in, 1.0f);

    EXPECT_NEAR(cc.state().position.x - start.x, 200.0f, 20.0f);
}

TEST(CharacterController, JumpLeavesTheGroundAndComesBack) {
    FloorWorld fw;
    ASSERT_TRUE(fw.ok);
    CharacterController cc(test_config());
    cc.set_position({0.0f, 10.0f, 0.0f});
    simulate(cc, fw.world, {}, 0.5f);
    ASSERT_TRUE(cc.state().grounded);

    CharacterInput jump;
    jump.jump = true;
    cc.update(fw.world, jump, k_dt);
    EXPECT_FALSE(cc.state().grounded);
    EXPECT_GT(cc.state().velocity.y, 0.0f);

    float peak = cc.state().position.y;
    for (int i = 0; i < 200; ++i) {
        cc.update(fw.world, {}, k_dt);
        peak = std::max(peak, cc.state().position.y);
    }
    EXPECT_GT(peak, 100.0f) << "the jump should actually clear some height";
    EXPECT_TRUE(cc.state().grounded) << "and it must land again";
}

TEST(CharacterController, CannotJumpWhileAirborne) {
    FloorWorld fw;
    ASSERT_TRUE(fw.ok);
    CharacterController cc(test_config());
    cc.set_position({0.0f, 800.0f, 0.0f});

    CharacterInput jump;
    jump.jump = true;
    // Well past coyote time, falling from a great height.
    simulate(cc, fw.world, {}, 0.5f);
    const float before = cc.state().velocity.y;
    cc.update(fw.world, jump, k_dt);
    EXPECT_LT(cc.state().velocity.y, before + 100.0f) << "no double jump";
}

// Players press jump a frame or two late constantly. Without coyote time the
// game feels unresponsive and nobody can say why.
TEST(CharacterController, CoyoteTimeAllowsAJumpJustAfterLeavingGround) {
    FloorWorld fw;
    ASSERT_TRUE(fw.ok);
    CharacterConfig cfg = test_config();
    cfg.coyote_time = 0.15f;
    CharacterController cc(cfg);
    cc.set_position({0.0f, 10.0f, 0.0f});
    simulate(cc, fw.world, {}, 0.5f);
    ASSERT_TRUE(cc.state().grounded);

    // Walk off the edge of the world floor... simulated by two frames of fall.
    cc.set_position({0.0f, 200.0f, 0.0f});
    cc.update(fw.world, {}, k_dt); // now airborne, but only just

    CharacterInput jump;
    jump.jump = true;
    cc.update(fw.world, jump, k_dt);
    EXPECT_GT(cc.state().velocity.y, 0.0f) << "a jump within coyote time should fire";
}

// A jump pressed just before landing should fire on touchdown, not be dropped.
TEST(CharacterController, JumpBufferFiresOnLanding) {
    FloorWorld fw;
    ASSERT_TRUE(fw.ok);
    CharacterConfig cfg = test_config();
    cfg.jump_buffer = 0.2f;
    CharacterController cc(cfg);
    // Fall time from 30 units at g=2000 is ~0.17s, inside the 0.2s buffer.
    cc.set_position({0.0f, 30.0f, 0.0f});

    CharacterInput jump;
    jump.jump = true;
    cc.update(fw.world, jump, k_dt); // pressed while still falling

    bool left_ground_again = false;
    for (int i = 0; i < 60; ++i) {
        cc.update(fw.world, {}, k_dt);
        if (cc.state().velocity.y > 100.0f) {
            left_ground_again = true;
            break;
        }
    }
    EXPECT_TRUE(left_ground_again) << "a buffered jump should fire when the character lands";
}

TEST(CharacterController, StepsOntoALowLedge) {
    FloorWorld fw;
    ASSERT_TRUE(fw.ok);
    // A 30-unit step, under the 40-unit step height.
    // Extends well past the ~500 units the character walks, so the assertion
    // samples while it is still ON the ledge rather than after crossing it.
    fw.add_block({1100.0f, 0.0f, 0.0f}, 30.0f, {1000.0f, 200.0f, 400.0f});
    fw.settle();

    CharacterController cc(test_config());
    cc.set_position({0.0f, 10.0f, 0.0f});
    simulate(cc, fw.world, {}, 0.5f);

    CharacterInput in;
    in.desired_velocity = {250.0f, 0.0f, 0.0f};
    simulate(cc, fw.world, in, 2.0f);

    EXPECT_GT(cc.state().position.y, 20.0f)
        << "a low step must be climbed, not treated as a wall";
    EXPECT_GT(cc.state().position.x, 100.0f) << "and the character must get past it";
}

TEST(CharacterController, IsBlockedByATallWall) {
    FloorWorld fw;
    ASSERT_TRUE(fw.ok);
    // 400 units tall: far above step height.
    fw.add_block({300.0f, 0.0f, 0.0f}, 400.0f, {50.0f, 400.0f, 500.0f});
    fw.settle();

    CharacterController cc(test_config());
    cc.set_position({0.0f, 10.0f, 0.0f});
    simulate(cc, fw.world, {}, 0.5f);

    CharacterInput in;
    in.desired_velocity = {300.0f, 0.0f, 0.0f};
    simulate(cc, fw.world, in, 3.0f);

    EXPECT_LT(cc.state().position.x, 300.0f) << "the wall must stop the character";
    EXPECT_LT(cc.state().position.y, 100.0f) << "and must not be climbed";
}

TEST(CharacterController, CrouchLowersHeightAndStandingRestoresIt) {
    FloorWorld fw;
    ASSERT_TRUE(fw.ok);
    CharacterController cc(test_config());
    cc.set_position({0.0f, 10.0f, 0.0f});
    simulate(cc, fw.world, {}, 0.5f);
    EXPECT_FLOAT_EQ(cc.height(), test_config().stand_height);

    CharacterInput crouch;
    crouch.crouch = true;
    cc.update(fw.world, crouch, k_dt);
    EXPECT_TRUE(cc.state().crouching);
    EXPECT_FLOAT_EQ(cc.height(), test_config().crouch_height);

    cc.update(fw.world, {}, k_dt);
    EXPECT_FALSE(cc.state().crouching) << "nothing overhead, so standing should succeed";
}

// Standing up into a ceiling would eject the character through it.
TEST(CharacterController, CannotStandUnderALowCeiling) {
    FloorWorld fw;
    ASSERT_TRUE(fw.ok);
    CharacterController cc(test_config());
    cc.set_position({0.0f, 10.0f, 0.0f});
    simulate(cc, fw.world, {}, 0.5f);

    CharacterInput crouch;
    crouch.crouch = true;
    cc.update(fw.world, crouch, k_dt);
    ASSERT_TRUE(cc.state().crouching);

    // Ceiling whose underside is at 100 — above the 80 crouch height, below
    // the 150 stand height.
    BodyDesc d;
    d.position = {0.0f, 100.0f + 50.0f, 0.0f};
    d.type = BodyType::Static;
    (void)fw.world.add_box(d, {300.0f, 50.0f, 300.0f});
    fw.settle();

    cc.update(fw.world, {}, k_dt); // release crouch
    EXPECT_TRUE(cc.state().crouching) << "must stay crouched while blocked overhead";
}

TEST(CharacterController, ZeroDeltaTimeIsANoOp) {
    FloorWorld fw;
    ASSERT_TRUE(fw.ok);
    CharacterController cc(test_config());
    cc.set_position({10.0f, 20.0f, 30.0f});
    const glm::vec3 before = cc.state().position;
    cc.update(fw.world, {}, 0.0f);
    EXPECT_EQ(cc.state().position, before);
}

TEST(CharacterController, SetPositionClearsVelocity) {
    FloorWorld fw;
    ASSERT_TRUE(fw.ok);
    CharacterController cc(test_config());
    cc.set_position({0.0f, 500.0f, 0.0f});
    simulate(cc, fw.world, {}, 0.3f);
    ASSERT_LT(cc.state().velocity.y, -100.0f);

    cc.set_position({0.0f, 500.0f, 0.0f});
    EXPECT_EQ(cc.state().velocity, glm::vec3(0.0f)) << "a teleport should not keep momentum";
}
