#include "engine/animation/ik_solver.hpp"
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <string>

using namespace mmo::engine::animation;

namespace {

// Bind-pose matrices for a test skeleton: translation-only world transforms,
// identity bone matrices (world * inverse_bind at bind pose). Arrays are
// MAX_BONES wide to satisfy the span contract in ik_solver.hpp.
struct Pose {
    std::array<glm::mat4, MAX_BONES> world;
    std::array<glm::mat4, MAX_BONES> bones;
};

void add_joint(Skeleton& skel, std::string name, int parent, const glm::vec3& pos) {
    Joint j;
    j.name = std::move(name);
    j.parent_index = parent;
    j.inverse_bind_matrix = glm::translate(glm::mat4(1.0f), -pos);
    glm::vec3 parent_pos = parent >= 0 ? -glm::vec3(skel.joints[parent].inverse_bind_matrix[3]) : glm::vec3(0.0f);
    j.local_translation = pos - parent_pos;
    j.local_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    j.local_scale = glm::vec3(1.0f);
    skel.joints.push_back(std::move(j));
}

Pose make_bind_pose(const Skeleton& skel) {
    Pose pose;
    pose.world.fill(glm::mat4(1.0f));
    pose.bones.fill(glm::mat4(1.0f));
    for (size_t i = 0; i < skel.joints.size(); i++) {
        glm::vec3 pos = -glm::vec3(skel.joints[i].inverse_bind_matrix[3]);
        pose.world[i] = glm::translate(glm::mat4(1.0f), pos);
    }
    return pose;
}

glm::vec3 joint_pos(const Pose& pose, int idx) {
    return glm::vec3(pose.world[idx][3]);
}

bool is_finite(const glm::mat4& m) {
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            if (!std::isfinite(m[c][r])) {
                return false;
            }
        }
    }
    return true;
}

void expect_vec_near(const glm::vec3& actual, const glm::vec3& expected, float eps = 1e-4f) {
    EXPECT_NEAR(actual.x, expected.x, eps);
    EXPECT_NEAR(actual.y, expected.y, eps);
    EXPECT_NEAR(actual.z, expected.z, eps);
}

void expect_mat_near(const glm::mat4& actual, const glm::mat4& expected, float eps = 1e-4f) {
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            EXPECT_NEAR(actual[c][r], expected[c][r], eps) << "element [" << c << "][" << r << "]";
        }
    }
}

void expect_pose_unchanged(const Pose& pose, const Pose& before, size_t joint_count) {
    for (size_t i = 0; i < joint_count; i++) {
        EXPECT_EQ(pose.world[i], before.world[i]) << "world_transforms[" << i << "] changed";
        EXPECT_EQ(pose.bones[i], before.bones[i]) << "bone_matrices[" << i << "] changed";
    }
}

void expect_pose_finite(const Pose& pose, size_t joint_count) {
    for (size_t i = 0; i < joint_count; i++) {
        EXPECT_TRUE(is_finite(pose.world[i])) << "world_transforms[" << i << "] is non-finite";
        EXPECT_TRUE(is_finite(pose.bones[i])) << "bone_matrices[" << i << "] is non-finite";
    }
}

} // namespace

// ============================================================================
// solve_two_bone_ik
// ============================================================================

// Straight vertical chain: hip at y=2, knee at y=1, foot at the origin, with a
// toe child hanging off the end effector. Bone lengths are 1 + 1, so max reach
// is 1.99 and min reach 0.01 (the solver keeps a 0.01 margin at both ends).
class TwoBoneIKTest : public ::testing::Test {
protected:
    void SetUp() override {
        add_joint(skeleton, "Hip", -1, {0.0f, 2.0f, 0.0f});
        add_joint(skeleton, "Knee", HIP, {0.0f, 1.0f, 0.0f});
        add_joint(skeleton, "Foot", KNEE, {0.0f, 0.0f, 0.0f});
        add_joint(skeleton, "Toe", FOOT, {0.0f, 0.0f, 0.2f});
        pose = make_bind_pose(skeleton);
    }

    static constexpr int HIP = 0;
    static constexpr int KNEE = 1;
    static constexpr int FOOT = 2;
    static constexpr int TOE = 3;

    Skeleton skeleton;
    Pose pose;
};

TEST_F(TwoBoneIKTest, ReachableTargetPlacesEndEffector) {
    glm::vec3 target(0.5f, 0.8f, 0.0f); // 1.3 from the hip, well inside reach
    solve_two_bone_ik(pose.bones, pose.world, skeleton, HIP, KNEE, FOOT, target, glm::vec3(0.0f, 1.0f, 1.0f));

    expect_vec_near(joint_pos(pose, FOOT), target);
    expect_vec_near(joint_pos(pose, HIP), glm::vec3(0.0f, 2.0f, 0.0f)); // root of the chain never moves
    expect_pose_finite(pose, skeleton.joints.size());
}

TEST_F(TwoBoneIKTest, PreservesBoneLengths) {
    glm::vec3 target(0.5f, 0.8f, 0.0f);
    solve_two_bone_ik(pose.bones, pose.world, skeleton, HIP, KNEE, FOOT, target, glm::vec3(0.0f, 1.0f, 1.0f));

    EXPECT_NEAR(glm::length(joint_pos(pose, KNEE) - joint_pos(pose, HIP)), 1.0f, 1e-3f);
    EXPECT_NEAR(glm::length(joint_pos(pose, FOOT) - joint_pos(pose, KNEE)), 1.0f, 1e-3f);
}

TEST_F(TwoBoneIKTest, TargetBeyondReachClampsToFullExtension) {
    glm::vec3 target(0.0f, -5.0f, 0.0f); // 7 units below the hip, far out of reach
    solve_two_bone_ik(pose.bones, pose.world, skeleton, HIP, KNEE, FOOT, target, glm::vec3(0.0f, 1.0f, 1.0f));

    // Clamped onto the max-reach sphere (1 + 1 - 0.01) along the hip->target ray.
    EXPECT_NEAR(glm::length(joint_pos(pose, FOOT) - joint_pos(pose, HIP)), 1.99f, 1e-3f);
    expect_vec_near(joint_pos(pose, FOOT), glm::vec3(0.0f, 0.01f, 0.0f), 1e-3f);
    expect_pose_finite(pose, skeleton.joints.size());
}

TEST_F(TwoBoneIKTest, BendFollowsPoleHint) {
    glm::vec3 target(0.0f, 0.5f, 0.0f); // 1.5 from the hip: forces a bent knee

    solve_two_bone_ik(pose.bones, pose.world, skeleton, HIP, KNEE, FOOT, target, glm::vec3(0.0f, 1.25f, 5.0f));
    EXPECT_GT(joint_pos(pose, KNEE).z, 0.1f) << "knee should bend toward the +z pole";
    expect_vec_near(joint_pos(pose, FOOT), target);

    pose = make_bind_pose(skeleton);
    solve_two_bone_ik(pose.bones, pose.world, skeleton, HIP, KNEE, FOOT, target, glm::vec3(0.0f, 1.25f, -5.0f));
    EXPECT_LT(joint_pos(pose, KNEE).z, -0.1f) << "knee should bend toward the -z pole";
    expect_vec_near(joint_pos(pose, FOOT), target);
}

TEST(TwoBoneIK, ZeroLengthBoneIsNoOp) {
    Skeleton skeleton;
    add_joint(skeleton, "Hip", -1, {0.0f, 2.0f, 0.0f});
    add_joint(skeleton, "Knee", 0, {0.0f, 2.0f, 0.0f}); // coincident with the hip
    add_joint(skeleton, "Foot", 1, {0.0f, 1.0f, 0.0f});
    Pose pose = make_bind_pose(skeleton);
    Pose before = pose;

    solve_two_bone_ik(pose.bones, pose.world, skeleton, 0, 1, 2, glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 1.0f));

    expect_pose_unchanged(pose, before, skeleton.joints.size());
    expect_pose_finite(pose, skeleton.joints.size());
}

TEST_F(TwoBoneIKTest, StraightChainWithCollinearPoleIsNoOp) {
    Pose before = pose;

    // Chain, target, and pole hint all lie on the y axis: no bend plane can be
    // derived, so the solver must give up without touching the pose.
    solve_two_bone_ik(pose.bones, pose.world, skeleton, HIP, KNEE, FOOT, glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f));

    expect_pose_unchanged(pose, before, skeleton.joints.size());
    expect_pose_finite(pose, skeleton.joints.size());
}

TEST(TwoBoneIK, DegeneratePoleFallsBackToChainBend) {
    Skeleton skeleton;
    add_joint(skeleton, "Hip", -1, {0.0f, 2.0f, 0.0f});
    add_joint(skeleton, "Knee", 0, {0.3f, 1.0f, 0.0f}); // pre-bent toward +x
    add_joint(skeleton, "Foot", 1, {0.0f, 0.0f, 0.0f});
    Pose pose = make_bind_pose(skeleton);

    // Pole hint sits on the hip->target axis (useless), but the chain is bent,
    // so the solver falls back to the existing knee direction and still solves.
    glm::vec3 target(0.0f, 0.4f, 0.0f);
    solve_two_bone_ik(pose.bones, pose.world, skeleton, 0, 1, 2, target, glm::vec3(0.0f, 1.0f, 0.0f));

    expect_vec_near(joint_pos(pose, 2), target, 1e-3f);
    EXPECT_GT(joint_pos(pose, 1).x, 0.1f) << "knee should keep bending toward +x";
    expect_pose_finite(pose, skeleton.joints.size());
}

TEST_F(TwoBoneIKTest, TargetAtHipClampsToMinReachWithoutNaN) {
    glm::vec3 target = joint_pos(pose, HIP); // exactly on the chain root
    solve_two_bone_ik(pose.bones, pose.world, skeleton, HIP, KNEE, FOOT, target, glm::vec3(0.0f, 1.0f, 1.0f));

    expect_pose_finite(pose, skeleton.joints.size());
    // Pulled out to the min-reach shell (|len_ab - len_bc| + 0.01).
    EXPECT_NEAR(glm::length(joint_pos(pose, FOOT) - joint_pos(pose, HIP)), 0.01f, 1e-4f);
    EXPECT_NEAR(glm::length(joint_pos(pose, KNEE) - joint_pos(pose, HIP)), 1.0f, 1e-3f);
}

TEST_F(TwoBoneIKTest, NonFiniteTargetIsNoOp) {
    Pose before = pose;
    float nan = std::numeric_limits<float>::quiet_NaN();

    solve_two_bone_ik(pose.bones, pose.world, skeleton, HIP, KNEE, FOOT, glm::vec3(nan, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 1.0f));

    expect_pose_unchanged(pose, before, skeleton.joints.size());
}

TEST_F(TwoBoneIKTest, ChildOfEndEffectorFollows) {
    glm::vec3 target(0.5f, 0.8f, 0.0f);
    solve_two_bone_ik(pose.bones, pose.world, skeleton, HIP, KNEE, FOOT, target, glm::vec3(0.0f, 1.0f, 1.0f));

    // The toe is carried along by translation, preserving its offset from the foot.
    expect_vec_near(joint_pos(pose, TOE), target + glm::vec3(0.0f, 0.0f, 0.2f));
}

TEST_F(TwoBoneIKTest, BoneMatricesMatchWorldTransforms) {
    glm::vec3 target(0.5f, 0.8f, 0.0f);
    solve_two_bone_ik(pose.bones, pose.world, skeleton, HIP, KNEE, FOOT, target, glm::vec3(0.0f, 1.0f, 1.0f));

    for (size_t i = 0; i < skeleton.joints.size(); i++) {
        expect_mat_near(pose.bones[i], pose.world[i] * skeleton.joints[i].inverse_bind_matrix);
    }
}

// ============================================================================
// apply_foot_ik
// ============================================================================

// Minimal biped: hips + spine + two legs, joint names following the Mixamo
// conventions in FootIKBoneNames so FootIKData::init resolves them. Knees are
// pre-bent toward +z so the leg solve's pole derivation is never degenerate.
class FootIKTest : public ::testing::Test {
protected:
    void SetUp() override {
        add_joint(skeleton, "Hips", -1, {0.0f, 1.0f, 0.0f});
        add_joint(skeleton, "Spine", 0, {0.0f, 1.2f, 0.0f});
        add_joint(skeleton, "LeftUpperLeg", 0, {0.2f, 1.0f, 0.0f});
        add_joint(skeleton, "LeftLowerLeg", 2, {0.2f, 0.5f, 0.1f});
        add_joint(skeleton, "LeftFoot", 3, {0.2f, 0.0f, 0.0f});
        add_joint(skeleton, "RightUpperLeg", 0, {-0.2f, 1.0f, 0.0f});
        add_joint(skeleton, "RightLowerLeg", 5, {-0.2f, 0.5f, 0.1f});
        add_joint(skeleton, "RightFoot", 6, {-0.2f, 0.0f, 0.0f});
        ik.init(skeleton);
        pose = make_bind_pose(skeleton);
    }

    Skeleton skeleton;
    FootIKData ik;
    Pose pose;
};

TEST_F(FootIKTest, InitResolvesMixamoBoneNames) {
    EXPECT_TRUE(ik.valid);
    EXPECT_EQ(ik.hips, 0);
    EXPECT_EQ(ik.spine, 1);
    EXPECT_EQ(ik.left_upper, 2);
    EXPECT_EQ(ik.left_foot, 4);
    EXPECT_EQ(ik.right_upper, 5);
    EXPECT_EQ(ik.right_foot, 7);
}

TEST_F(FootIKTest, SubThresholdOffsetsAreNoOp) {
    Pose before = pose;

    // Both offsets inside the 0.1 deadzone: the whole pass is skipped.
    apply_foot_ik(pose.bones, pose.world, skeleton, ik, glm::mat4(1.0f), 1.0f, 0.05f, -0.05f);

    expect_pose_unchanged(pose, before, skeleton.joints.size());
}

TEST_F(FootIKTest, InvalidDataIsNoOp) {
    // ik.valid == false means the bone-name lookup failed and the joint
    // indices are still -1; apply_foot_ik must bail out before touching them,
    // even when the offsets are far outside the deadzone.
    FootIKData invalid;
    ASSERT_FALSE(invalid.valid);
    Pose before = pose;

    apply_foot_ik(pose.bones, pose.world, skeleton, invalid, glm::mat4(1.0f), 1.0f, 0.0f, 0.0f);
    expect_pose_unchanged(pose, before, skeleton.joints.size());

    apply_foot_ik(pose.bones, pose.world, skeleton, invalid, glm::mat4(1.0f), 1.0f, -5.0f, 3.0f);
    expect_pose_unchanged(pose, before, skeleton.joints.size());
}

TEST_F(FootIKTest, EqualNegativeOffsetsDropWholeSkeleton) {
    apply_foot_ik(pose.bones, pose.world, skeleton, ik, glm::mat4(1.0f), 1.0f, -0.5f, -0.5f);

    // Pelvis drop absorbs the entire correction: every joint moves down by 0.5
    // and no per-leg solve is needed.
    Pose bind = make_bind_pose(skeleton);
    for (size_t i = 0; i < skeleton.joints.size(); i++) {
        expect_vec_near(joint_pos(pose, static_cast<int>(i)),
                        joint_pos(bind, static_cast<int>(i)) + glm::vec3(0.0f, -0.5f, 0.0f));
        expect_mat_near(pose.bones[i], pose.world[i] * skeleton.joints[i].inverse_bind_matrix);
    }
}

TEST_F(FootIKTest, UnevenOffsetsSolveHigherLeg) {
    apply_foot_ik(pose.bones, pose.world, skeleton, ik, glm::mat4(1.0f), 1.0f, -0.5f, 0.0f);

    // The pelvis drops to the lower (left) foot; the right leg is then IK'd
    // back up to its original ground height.
    expect_vec_near(joint_pos(pose, ik.hips), glm::vec3(0.0f, 0.5f, 0.0f));
    expect_vec_near(joint_pos(pose, ik.left_foot), glm::vec3(0.2f, -0.5f, 0.0f));
    expect_vec_near(joint_pos(pose, ik.right_foot), glm::vec3(-0.2f, 0.0f, 0.0f), 1e-3f);
    EXPECT_GT(joint_pos(pose, ik.right_lower).z, 0.05f) << "right knee should bend forward";
    expect_pose_finite(pose, skeleton.joints.size());
}

TEST_F(FootIKTest, PositiveOffsetsRaiseFeetWithoutPelvisShift) {
    Pose before = pose;

    apply_foot_ik(pose.bones, pose.world, skeleton, ik, glm::mat4(1.0f), 1.0f, 0.3f, 0.3f);

    // Feet stepping up onto higher terrain: no pelvis drop, both legs solve.
    EXPECT_EQ(pose.world[ik.hips], before.world[ik.hips]);
    EXPECT_EQ(pose.world[ik.spine], before.world[ik.spine]);
    expect_vec_near(joint_pos(pose, ik.left_foot), glm::vec3(0.2f, 0.3f, 0.0f), 1e-3f);
    expect_vec_near(joint_pos(pose, ik.right_foot), glm::vec3(-0.2f, 0.3f, 0.0f), 1e-3f);
    expect_pose_finite(pose, skeleton.joints.size());
}

TEST_F(FootIKTest, ScaleConvertsWorldOffsetsToModelSpace) {
    apply_foot_ik(pose.bones, pose.world, skeleton, ik, glm::mat4(1.0f), 2.0f, -0.5f, -0.5f);

    // A -0.5 world-space offset on a 2x-scaled model is a -0.25 model-space drop.
    expect_vec_near(joint_pos(pose, ik.hips), glm::vec3(0.0f, 0.75f, 0.0f));
    expect_vec_near(joint_pos(pose, ik.left_foot), glm::vec3(0.2f, -0.25f, 0.0f));
    expect_vec_near(joint_pos(pose, ik.right_foot), glm::vec3(-0.2f, -0.25f, 0.0f));
}

TEST_F(FootIKTest, OffsetsClampToMaxCorrection) {
    apply_foot_ik(pose.bones, pose.world, skeleton, ik, glm::mat4(1.0f), 1.0f, -20.0f, -20.0f);

    // Corrections are clamped to +/-8 before the pelvis drop is computed.
    expect_vec_near(joint_pos(pose, ik.hips), glm::vec3(0.0f, -7.0f, 0.0f));
    expect_vec_near(joint_pos(pose, ik.left_foot), glm::vec3(0.2f, -8.0f, 0.0f));
    expect_vec_near(joint_pos(pose, ik.right_foot), glm::vec3(-0.2f, -8.0f, 0.0f));
    expect_pose_finite(pose, skeleton.joints.size());
}

// ============================================================================
// apply_body_lean
// ============================================================================

// Hips with a spine->head chain plus one leg that is *not* a spine descendant,
// to verify the lean only touches the spine subtree.
class BodyLeanTest : public ::testing::Test {
protected:
    void SetUp() override {
        add_joint(skeleton, "Hips", -1, {0.0f, 1.0f, 0.0f});
        add_joint(skeleton, "Spine", HIPS, {0.0f, 1.2f, 0.0f});
        add_joint(skeleton, "Head", SPINE, {0.0f, 1.8f, 0.0f});
        add_joint(skeleton, "LeftLeg", HIPS, {0.2f, 0.5f, 0.0f});
        pose = make_bind_pose(skeleton);
    }

    static constexpr int HIPS = 0;
    static constexpr int SPINE = 1;
    static constexpr int HEAD = 2;
    static constexpr int LEG = 3;

    Skeleton skeleton;
    Pose pose;
};

TEST_F(BodyLeanTest, ForwardLeanPitchesSpineDescendants) {
    apply_body_lean(pose.bones, pose.world, skeleton, SPINE, 0.2f, 0.0f);

    // The spine is the pivot: its position holds but its basis pitches about +x.
    expect_vec_near(joint_pos(pose, SPINE), glm::vec3(0.0f, 1.2f, 0.0f));
    EXPECT_NEAR(pose.world[SPINE][1][1], std::cos(0.2f), 1e-4f);
    EXPECT_NEAR(pose.world[SPINE][1][2], std::sin(0.2f), 1e-4f);

    // The head (0.6 above the pivot) swings forward on the same arc.
    expect_vec_near(joint_pos(pose, HEAD),
                    glm::vec3(0.0f, 1.2f + 0.6f * std::cos(0.2f), 0.6f * std::sin(0.2f)));
}

TEST_F(BodyLeanTest, LateralLeanRollsSpineDescendants) {
    apply_body_lean(pose.bones, pose.world, skeleton, SPINE, 0.0f, 0.3f);

    expect_vec_near(joint_pos(pose, SPINE), glm::vec3(0.0f, 1.2f, 0.0f));
    expect_vec_near(joint_pos(pose, HEAD),
                    glm::vec3(-0.6f * std::sin(0.3f), 1.2f + 0.6f * std::cos(0.3f), 0.0f));
}

TEST_F(BodyLeanTest, NonDescendantsUnaffected) {
    Pose before = pose;

    apply_body_lean(pose.bones, pose.world, skeleton, SPINE, 0.25f, 0.1f);

    EXPECT_EQ(pose.world[HIPS], before.world[HIPS]);
    EXPECT_EQ(pose.bones[HIPS], before.bones[HIPS]);
    EXPECT_EQ(pose.world[LEG], before.world[LEG]);
    EXPECT_EQ(pose.bones[LEG], before.bones[LEG]);
}

TEST_F(BodyLeanTest, IdentityLeanIsNoOp) {
    Pose before = pose;

    apply_body_lean(pose.bones, pose.world, skeleton, SPINE, 0.0f, 0.0f);
    expect_pose_unchanged(pose, before, skeleton.joints.size());

    // Leans inside the 0.001 deadzone are also skipped.
    apply_body_lean(pose.bones, pose.world, skeleton, SPINE, 0.0005f, -0.0005f);
    expect_pose_unchanged(pose, before, skeleton.joints.size());
}

TEST_F(BodyLeanTest, BoneMatricesMatchWorldAfterLean) {
    apply_body_lean(pose.bones, pose.world, skeleton, SPINE, 0.2f, 0.1f);

    for (size_t i = 0; i < skeleton.joints.size(); i++) {
        expect_mat_near(pose.bones[i], pose.world[i] * skeleton.joints[i].inverse_bind_matrix);
    }
}
