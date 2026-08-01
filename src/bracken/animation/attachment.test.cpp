#include <gtest/gtest.h>

#include "bracken/animation/attachment.hpp"

#include <glm/gtc/matrix_transform.hpp>

using namespace bracken::animation;

namespace {

/** A two-joint skeleton: root at the origin, "hand" offset along +X. */
Skeleton two_joint_skeleton() {
    Skeleton s;
    Joint root;
    root.name = "root";
    root.parent_index = -1;
    root.inverse_bind_matrix = glm::mat4(1.0f);
    root.local_translation = {0.0f, 0.0f, 0.0f};
    root.local_rotation = glm::quat(1, 0, 0, 0);
    root.local_scale = {1.0f, 1.0f, 1.0f};

    Joint hand = root;
    hand.name = "hand";
    hand.parent_index = 0;
    hand.local_translation = {100.0f, 0.0f, 0.0f};

    s.joints = {root, hand};
    return s;
}

/** A one-keyframe clip that reproduces the rest pose.
 *
 *  AnimationPlayer::update() returns early when there are no clips, so a
 *  skeleton with no animation never has its world transforms computed and
 *  every attachment would read identity. A single-keyframe clip is the
 *  smallest thing that drives the real code path — which is also what a
 *  static posed character needs in a game.
 */
std::vector<AnimationClip> rest_pose_clip(const Skeleton& skeleton) {
    AnimationClip clip;
    clip.name = "rest";
    clip.duration = 1.0f;
    for (std::size_t i = 0; i < skeleton.joints.size(); ++i) {
        AnimationChannel ch;
        ch.bone_index = static_cast<int>(i);
        ch.position_times = {0.0f};
        ch.positions = {skeleton.joints[i].local_translation};
        ch.rotation_times = {0.0f};
        ch.rotations = {skeleton.joints[i].local_rotation};
        ch.scale_times = {0.0f};
        ch.scales = {skeleton.joints[i].local_scale};
        clip.channels.push_back(std::move(ch));
    }
    return {clip};
}

AnimationPlayer posed_player(const Skeleton& skeleton, const std::vector<AnimationClip>& clips) {
    AnimationPlayer p;
    p.update(skeleton, clips, 0.0f);
    return p;
}

} // namespace

TEST(Attachment, FindsAJointByName) {
    const Skeleton s = two_joint_skeleton();
    EXPECT_TRUE(find_joint(s, "root").valid());
    EXPECT_EQ(find_joint(s, "root").value, 0);
    EXPECT_EQ(find_joint(s, "hand").value, 1);
}

TEST(Attachment, UnknownJointIsInvalidNotUndefined) {
    const Skeleton s = two_joint_skeleton();
    const JointIndex missing = find_joint(s, "tail");
    EXPECT_FALSE(missing.valid());

    const auto clips = rest_pose_clip(s);
    AnimationPlayer p = posed_player(s, clips);
    // Every accessor must tolerate it.
    EXPECT_EQ(joint_model_transform(p, missing), glm::mat4(1.0f));
}

// A renamed bone in an updated asset should drop the sword, not corrupt the
// frame or teleport it to the world origin.
TEST(Attachment, InvalidJointKeepsTheObjectOnTheCharacter) {
    const Skeleton s = two_joint_skeleton();
    const auto clips = rest_pose_clip(s);
    AnimationPlayer p = posed_player(s, clips);
    const glm::mat4 model = glm::translate(glm::mat4(1.0f), {500.0f, 0.0f, 300.0f});

    const glm::mat4 world = attachment_transform(model, p, JointIndex{}, glm::mat4(1.0f));
    EXPECT_EQ(glm::vec3(world[3]), glm::vec3(500.0f, 0.0f, 300.0f))
        << "it should follow the model, not fall to the origin";
}

TEST(Attachment, TracksTheBoneInModelSpace) {
    const Skeleton s = two_joint_skeleton();
    const auto clips = rest_pose_clip(s);
    AnimationPlayer p = posed_player(s, clips);
    const JointIndex hand = find_joint(s, "hand");
    ASSERT_TRUE(hand.valid());

    const glm::mat4 bone = joint_model_transform(p, hand);
    EXPECT_NEAR(bone[3][0], 100.0f, 1e-3f) << "the hand sits 100 units along +X of the root";
}

TEST(Attachment, CombinesModelTransformWithTheBone) {
    const Skeleton s = two_joint_skeleton();
    const auto clips = rest_pose_clip(s);
    AnimationPlayer p = posed_player(s, clips);
    const JointIndex hand = find_joint(s, "hand");

    const glm::mat4 model = glm::translate(glm::mat4(1.0f), {1000.0f, 50.0f, 0.0f});
    const glm::vec3 world = glm::vec3(attachment_transform(model, p, hand)[3]);

    EXPECT_NEAR(world.x, 1100.0f, 1e-2f) << "model translation plus bone offset";
    EXPECT_NEAR(world.y, 50.0f, 1e-2f);
}

// Rotating the character must carry the attachment around with it — the case
// hand-rolled math usually gets wrong.
TEST(Attachment, FollowsTheCharacterRotation) {
    const Skeleton s = two_joint_skeleton();
    const auto clips = rest_pose_clip(s);
    AnimationPlayer p = posed_player(s, clips);
    const JointIndex hand = find_joint(s, "hand");

    // Yaw 90 degrees: the hand's +X offset should swing onto -Z.
    const glm::mat4 model =
        glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::vec3 world = glm::vec3(attachment_transform(model, p, hand)[3]);

    EXPECT_NEAR(world.x, 0.0f, 1e-2f);
    EXPECT_NEAR(world.z, -100.0f, 1e-2f);
}

TEST(Attachment, LocalOffsetAppliesInBoneSpace) {
    const Skeleton s = two_joint_skeleton();
    const auto clips = rest_pose_clip(s);
    AnimationPlayer p = posed_player(s, clips);
    const JointIndex hand = find_joint(s, "hand");

    const glm::mat4 grip = glm::translate(glm::mat4(1.0f), {0.0f, 20.0f, 0.0f});
    const glm::vec3 world = glm::vec3(attachment_transform(glm::mat4(1.0f), p, hand, grip)[3]);

    EXPECT_NEAR(world.x, 100.0f, 1e-2f);
    EXPECT_NEAR(world.y, 20.0f, 1e-2f);
}

TEST(Attachment, PositionHelperMatchesTheFullTransform) {
    const Skeleton s = two_joint_skeleton();
    const auto clips = rest_pose_clip(s);
    AnimationPlayer p = posed_player(s, clips);
    const JointIndex hand = find_joint(s, "hand");
    const glm::mat4 model = glm::translate(glm::mat4(1.0f), {7.0f, 8.0f, 9.0f});

    const glm::vec3 via_matrix = glm::vec3(attachment_transform(model, p, hand)[3]);
    const glm::vec3 via_helper = attachment_position(model, p, hand);
    EXPECT_NEAR(glm::length(via_matrix - via_helper), 0.0f, 1e-3f);
}

TEST(Attachment, MakeAttachmentResolvesAndCarriesItsOffset) {
    const Skeleton s = two_joint_skeleton();
    const auto clips = rest_pose_clip(s);
    AnimationPlayer p = posed_player(s, clips);

    const Attachment a =
        make_attachment(s, "hand", glm::translate(glm::mat4(1.0f), {0.0f, 0.0f, 5.0f}));
    ASSERT_TRUE(a.valid());

    const glm::vec3 world = glm::vec3(a.world(glm::mat4(1.0f), p)[3]);
    EXPECT_NEAR(world.x, 100.0f, 1e-2f);
    EXPECT_NEAR(world.z, 5.0f, 1e-2f);
}

TEST(Attachment, MakeAttachmentOnAMissingBoneIsInvalidButUsable) {
    const Skeleton s = two_joint_skeleton();
    const auto clips = rest_pose_clip(s);
    AnimationPlayer p = posed_player(s, clips);
    const Attachment a = make_attachment(s, "no_such_bone");
    EXPECT_FALSE(a.valid());
    // Still safe to call every frame.
    const glm::mat4 world = a.world(glm::mat4(1.0f), p);
    EXPECT_EQ(world, glm::mat4(1.0f));
}

TEST(Attachment, EmptySkeletonIsHandled) {
    const Skeleton empty;
    EXPECT_FALSE(find_joint(empty, "anything").valid());
}
