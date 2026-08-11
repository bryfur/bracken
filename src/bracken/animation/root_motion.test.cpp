// Root-motion extraction and stripping, validated entirely against SYNTHETIC
// clips.
//
// That is not laziness: no asset in this repository carries usable root
// translation. The playground character's only position channel is a vertical
// hips bob, and glTF root motion authored above the skin is dropped at load
// (ModelLoader counts it in ModelLoadReport::channels_non_skin_node). See
// root_motion.hpp for what an artist must author instead. One test below
// reproduces the playground's hips channel exactly and asserts it yields a ZERO
// delta, which is the behaviour that keeps this feature from breaking a
// controller-driven character the day someone switches it on.

#include "bracken/animation/root_motion.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

using namespace bracken::animation;

namespace {

constexpr float k_hips_y = 95.0f;

/// Hips -> Spine. Hips is at the ORIGIN in XZ, which root motion requires (the
/// round-trip test below is where that precondition bites).
Skeleton make_skeleton() {
    Skeleton s;
    Joint hips;
    hips.name = "Hips";
    hips.parent_index = -1;
    hips.local_translation = glm::vec3(0.0f, k_hips_y, 0.0f);
    hips.local_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    hips.local_scale = glm::vec3(1.0f);
    hips.inverse_bind_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -k_hips_y, 0.0f));
    s.joints.push_back(hips);

    Joint spine = hips;
    spine.name = "Spine";
    spine.parent_index = 0;
    spine.local_translation = glm::vec3(0.0f, 35.0f, 0.0f);
    s.joints.push_back(spine);
    return s;
}

/// Root translates +Z linearly by `distance` over `duration`, and yaws by
/// `yaw` radians over the same span.
AnimationClip make_travel_clip(float duration, float distance, float yaw) {
    AnimationClip c;
    c.name = "travel";
    c.duration = duration;
    AnimationChannel ch;
    ch.bone_index = 0;
    ch.position_times = {0.0f, duration};
    ch.positions = {glm::vec3(0.0f, k_hips_y, 0.0f), glm::vec3(0.0f, k_hips_y, distance)};
    ch.rotation_times = {0.0f, duration};
    ch.rotations = {glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::angleAxis(yaw, glm::vec3(0, 1, 0))};
    c.channels.push_back(ch);
    return c;
}

/// The playground's hips channel, verbatim in shape: a pure vertical bob and
/// nothing else (examples/playground/character.cpp).
AnimationClip make_bob_only_clip(float duration) {
    AnimationClip c;
    c.name = "walk";
    c.duration = duration;
    AnimationChannel ch;
    ch.bone_index = 0;
    for (int k = 0; k <= 16; ++k) {
        const float p = static_cast<float>(k) / 16.0f;
        ch.position_times.push_back(p * duration);
        ch.positions.push_back(
            glm::vec3(0.0f, k_hips_y + 4.0f * 0.5f * (1.0f - std::cos(4.0f * glm::pi<float>() * p)), 0.0f));
    }
    c.channels.push_back(ch);
    return c;
}

[[nodiscard]] float yaw_of(const glm::quat& q) {
    const glm::quat t = twist_about_y(q);
    return 2.0f * std::atan2(t.y, t.w);
}

} // namespace

// ---------------------------------------------------------------------------
// twist_about_y
// ---------------------------------------------------------------------------

TEST(RootMotion, TwistAboutYIsTheIdentityOnPurePitchAndRoll) {
    for (const glm::vec3 axis : {glm::vec3(1, 0, 0), glm::vec3(0, 0, 1)}) {
        for (const float a : {0.1f, 1.0f, 2.5f}) {
            const glm::quat t = twist_about_y(glm::angleAxis(a, axis));
            EXPECT_NEAR(std::abs(t.w), 1.0f, 1e-6f) << "axis " << axis.x << "," << axis.z << " angle " << a;
            EXPECT_NEAR(t.y, 0.0f, 1e-6f);
        }
    }
}

TEST(RootMotion, TwistAboutYReproducesAPureYaw) {
    for (const float a : {-2.0f, -0.3f, 0.0f, 0.75f, 3.0f}) {
        const glm::quat q = glm::angleAxis(a, glm::vec3(0, 1, 0));
        const glm::quat t = twist_about_y(q);
        EXPECT_NEAR(std::abs(glm::dot(q, t)), 1.0f, 1e-6f) << "angle " << a;
    }
}

TEST(RootMotion, TwistAboutYIsTheIdentityNotNaNAtTheDegenerateCase) {
    // 180 degrees about X: w == 0 and y == 0, so the yaw is genuinely undefined.
    const glm::quat q(0.0f, 1.0f, 0.0f, 0.0f);
    const glm::quat t = twist_about_y(q);
    EXPECT_TRUE(std::isfinite(t.w));
    EXPECT_FLOAT_EQ(t.w, 1.0f);
    EXPECT_FLOAT_EQ(t.x, 0.0f);
    EXPECT_FLOAT_EQ(t.y, 0.0f);
    EXPECT_FLOAT_EQ(t.z, 0.0f);
}

TEST(RootMotion, TheSwingLeftByRemovingTheTwistHasNoYaw) {
    const glm::quat q = glm::angleAxis(1.1f, glm::normalize(glm::vec3(0.4f, 0.7f, -0.3f)));
    const glm::quat swing = glm::inverse(twist_about_y(q)) * q;
    EXPECT_NEAR(yaw_of(swing), 0.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// extract_root_motion
// ---------------------------------------------------------------------------

TEST(RootMotion, AFullPassReturnsTheWholeAuthoredTranslation) {
    const Skeleton s = make_skeleton();
    const std::vector<AnimationClip> clips{make_travel_clip(2.0f, 300.0f, 0.0f)};
    ClipBindings b;
    b.build(s, clips);

    RootMotionConfig cfg;
    cfg.joint = 0;

    const RootMotionDelta whole = extract_root_motion(clips, b, 0, 0.0f, 2.0f, false, cfg);
    EXPECT_NEAR(whole.translation.z, 300.0f, 1e-3f);
    EXPECT_NEAR(whole.translation.x, 0.0f, 1e-6f);
    EXPECT_NEAR(whole.translation.y, 0.0f, 1e-6f) << "translation_y is off by default";

    const RootMotionDelta a = extract_root_motion(clips, b, 0, 0.0f, 1.0f, false, cfg);
    const RootMotionDelta c = extract_root_motion(clips, b, 0, 1.0f, 2.0f, false, cfg);
    EXPECT_NEAR(a.translation.z + c.translation.z, whole.translation.z, 1e-3f)
        << "two half passes must sum to one full pass";
}

TEST(RootMotion, TheWrappedCaseSplitsAtTheClipEndAndComposesPreWrapFirst) {
    const Skeleton s = make_skeleton();
    const std::vector<AnimationClip> clips{make_travel_clip(2.0f, 300.0f, 1.2f)};
    ClipBindings b;
    b.build(s, clips);

    RootMotionConfig cfg;
    cfg.joint = 0;

    const float t_prev = 1.8f;
    const float t_now = 0.2f;
    const RootMotionDelta wrapped = extract_root_motion(clips, b, 0, t_prev, t_now, true, cfg);

    const RootMotionDelta pre = extract_root_motion(clips, b, 0, t_prev, 2.0f, false, cfg);
    const RootMotionDelta post = extract_root_motion(clips, b, 0, 0.0f, t_now, false, cfg);

    EXPECT_NEAR(wrapped.translation.z, pre.translation.z + post.translation.z, 1e-4f);

    const glm::quat expected = post.rotation * pre.rotation;
    EXPECT_NEAR(std::abs(glm::dot(wrapped.rotation, expected)), 1.0f, 1e-6f)
        << "the composition order must be post * pre";
}

TEST(RootMotion, AVerticalHipsBobIsNotLocomotion) {
    // Exactly the playground's authored channel. With the default config this
    // must return nothing, or switching root motion on would freeze a character
    // whose XZ motion comes from the physics controller.
    const Skeleton s = make_skeleton();
    const std::vector<AnimationClip> clips{make_bob_only_clip(1.0f)};
    ClipBindings b;
    b.build(s, clips);

    RootMotionConfig cfg;
    cfg.joint = 0;

    for (int step = 0; step < 20; ++step) {
        const float t0 = static_cast<float>(step) * 0.05f;
        const RootMotionDelta d = extract_root_motion(clips, b, 0, t0, t0 + 0.05f, false, cfg);
        EXPECT_TRUE(d.is_zero()) << "step " << step << " produced translation " << d.translation.y;
    }

    // ...and turning translation_y on DOES see it, which is what proves the
    // previous assertion is about the config and not about a dead code path.
    cfg.translation_y = true;
    const RootMotionDelta lifted = extract_root_motion(clips, b, 0, 0.0f, 0.125f, false, cfg);
    EXPECT_GT(std::abs(lifted.translation.y), 0.5f);
}

TEST(RootMotion, AJointWithNoChannelYieldsAZeroDeltaRatherThanReadingTheRestPose) {
    const Skeleton s = make_skeleton();
    const std::vector<AnimationClip> clips{make_travel_clip(2.0f, 300.0f, 0.0f)};
    ClipBindings b;
    b.build(s, clips);

    RootMotionConfig cfg;
    cfg.joint = 1; // Spine — the clip animates joint 0 only
    EXPECT_TRUE(extract_root_motion(clips, b, 0, 0.0f, 1.0f, false, cfg).is_zero());

    cfg.joint = -1; // disabled
    EXPECT_TRUE(extract_root_motion(clips, b, 0, 0.0f, 1.0f, false, cfg).is_zero());

    cfg.joint = 0;
    EXPECT_TRUE(extract_root_motion(clips, b, 99, 0.0f, 1.0f, false, cfg).is_zero()) << "bad clip index";
}

TEST(RootMotion, ARotationOnlyChannelYieldsYawAndNoTranslation) {
    Skeleton s = make_skeleton();
    AnimationClip c;
    c.name = "turn";
    c.duration = 1.0f;
    AnimationChannel ch;
    ch.bone_index = 0;
    ch.rotation_times = {0.0f, 1.0f};
    ch.rotations = {glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::angleAxis(1.0f, glm::vec3(0, 1, 0))};
    c.channels.push_back(ch);
    const std::vector<AnimationClip> clips{c};

    ClipBindings b;
    b.build(s, clips);
    RootMotionConfig cfg;
    cfg.joint = 0;

    const RootMotionDelta d = extract_root_motion(clips, b, 0, 0.0f, 1.0f, false, cfg);
    EXPECT_EQ(d.translation, glm::vec3(0.0f));
    EXPECT_NEAR(yaw_of(d.rotation), 1.0f, 1e-4f);

    cfg.rotation_yaw = false;
    EXPECT_TRUE(extract_root_motion(clips, b, 0, 0.0f, 1.0f, false, cfg).is_zero());
}

// ---------------------------------------------------------------------------
// strip_root_motion
// ---------------------------------------------------------------------------

TEST(RootMotion, StrippingResetsXZAndYawWhilePreservingEverythingElseBitwise) {
    const Skeleton s = make_skeleton();
    std::array<BoneTransform, 2> reference{};
    rest_pose(s, PoseSpan(reference.data(), reference.size()));

    std::array<BoneTransform, 2> pose = reference;
    pose[0].translation = glm::vec3(120.0f, k_hips_y + 4.0f, -80.0f);
    // A rotation with all three of yaw, pitch and roll in it.
    pose[0].rotation = glm::angleAxis(0.9f, glm::vec3(0, 1, 0)) * glm::angleAxis(0.4f, glm::vec3(1, 0, 0)) *
                       glm::angleAxis(0.2f, glm::vec3(0, 0, 1));
    pose[1].translation = glm::vec3(3.0f, 4.0f, 5.0f);
    pose[1].rotation = glm::angleAxis(1.3f, glm::vec3(0, 1, 0));

    const glm::quat before_root = pose[0].rotation;
    const BoneTransform before_spine = pose[1];

    RootMotionConfig cfg;
    cfg.joint = 0;
    strip_root_motion(PoseSpan(pose.data(), pose.size()), ConstPoseSpan(reference.data(), reference.size()),
                      cfg);

    EXPECT_FLOAT_EQ(pose[0].translation.x, reference[0].translation.x);
    EXPECT_FLOAT_EQ(pose[0].translation.z, reference[0].translation.z);
    EXPECT_FLOAT_EQ(pose[0].translation.y, k_hips_y + 4.0f) << "translation_y is off, so Y must survive";
    EXPECT_NEAR(yaw_of(pose[0].rotation), 0.0f, 1e-5f);

    // Pitch and roll survive: reapplying the removed yaw reconstructs the input.
    const glm::quat rebuilt = twist_about_y(before_root) * pose[0].rotation;
    EXPECT_NEAR(std::abs(glm::dot(rebuilt, before_root)), 1.0f, 1e-5f);

    // Every other joint bitwise untouched.
    EXPECT_EQ(pose[1].translation, before_spine.translation);
    EXPECT_EQ(pose[1].rotation, before_spine.rotation);
    EXPECT_EQ(pose[1].scale, before_spine.scale);
}

TEST(RootMotion, StrippingIsANoOpWhenDisabledOrOutOfRange) {
    const Skeleton s = make_skeleton();
    std::array<BoneTransform, 2> reference{};
    rest_pose(s, PoseSpan(reference.data(), reference.size()));
    std::array<BoneTransform, 2> pose = reference;
    pose[0].translation = glm::vec3(5.0f, 6.0f, 7.0f);
    const glm::vec3 before = pose[0].translation;

    RootMotionConfig cfg;
    cfg.joint = -1;
    strip_root_motion(PoseSpan(pose.data(), 2), ConstPoseSpan(reference.data(), 2), cfg);
    EXPECT_EQ(pose[0].translation, before);

    cfg.joint = 99;
    strip_root_motion(PoseSpan(pose.data(), 2), ConstPoseSpan(reference.data(), 2), cfg);
    EXPECT_EQ(pose[0].translation, before);
}

TEST(RootMotion, ExtractStripAndReapplyReconstructTheUnstrippedRootTransform) {
    // THE no-double-application, no-loss property: the sum of the extracted
    // deltas is EXACTLY what stripping removed, so composing it back on top of
    // the stripped pose rebuilds the original.
    //
    // The accumulation here is in the CLIP's own frame — translations summed
    // plainly, rotations composed. A running game additionally rotates each
    // translation increment by the character's accumulated yaw
    // (`glm::mat3_cast(character_yaw) * d.translation`, as root_motion.hpp
    // documents), which makes a turning walk curve through the world. That
    // curve is the feature, not a loss, and it is why this test does not apply
    // it: the property being pinned is conservation, not trajectory.
    //
    // PRECONDITION this leans on, and the one root_motion.hpp tells artists
    // about: the root joint sits at the ORIGIN in XZ in the reference pose. With
    // a non-zero XZ reference the accumulated yaw rotates that offset and the
    // reconstruction legitimately fails.
    const Skeleton s = make_skeleton();
    const std::vector<AnimationClip> clips{make_travel_clip(2.0f, 300.0f, 1.1f)};
    ClipBindings b;
    b.build(s, clips);

    std::array<BoneTransform, 2> reference{};
    rest_pose(s, PoseSpan(reference.data(), reference.size()));
    ASSERT_FLOAT_EQ(reference[0].translation.x, 0.0f);
    ASSERT_FLOAT_EQ(reference[0].translation.z, 0.0f);

    RootMotionConfig cfg;
    cfg.joint = 0;

    glm::vec3 accumulated_translation(0.0f);
    glm::quat accumulated_yaw(1.0f, 0.0f, 0.0f, 0.0f);
    constexpr int k_steps = 40;
    const float dt = 2.0f / static_cast<float>(k_steps);

    for (int i = 0; i < k_steps; ++i) {
        const float t_prev = static_cast<float>(i) * dt;
        const float t_now = static_cast<float>(i + 1) * dt;
        const RootMotionDelta d = extract_root_motion(clips, b, 0, t_prev, t_now, false, cfg);
        accumulated_translation += d.translation;
        accumulated_yaw = d.rotation * accumulated_yaw;
    }
    EXPECT_NEAR(accumulated_translation.z, 300.0f, 0.05f) << "the deltas must add up to the authored travel";

    std::array<BoneTransform, 2> unstripped{};
    sample_clip_pose(s, clips, b, 0, 2.0f, PoseSpan(unstripped.data(), unstripped.size()));

    std::array<BoneTransform, 2> stripped = unstripped;
    strip_root_motion(PoseSpan(stripped.data(), 2), ConstPoseSpan(reference.data(), 2), cfg);
    // Stripping really did remove something — otherwise this test would pass on
    // a pair of no-ops.
    ASSERT_GT(std::abs(unstripped[0].translation.z - stripped[0].translation.z), 1.0f);

    const glm::mat4 expected =
        glm::translate(glm::mat4(1.0f), unstripped[0].translation) * glm::mat4_cast(unstripped[0].rotation);
    const glm::mat4 character =
        glm::translate(glm::mat4(1.0f), accumulated_translation) * glm::mat4_cast(accumulated_yaw);
    const glm::mat4 rebuilt = character * glm::translate(glm::mat4(1.0f), stripped[0].translation) *
                              glm::mat4_cast(stripped[0].rotation);

    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            EXPECT_NEAR(rebuilt[c][r], expected[c][r], 0.02f) << "column " << c << " row " << r;
        }
    }
}
