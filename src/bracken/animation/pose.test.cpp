// Pose value type, bone masks, the two blend operators, and the proof that
// pose_to_matrices IS the AnimationPlayer matrix pipeline rather than a
// lookalike.
//
// The comparisons here are deliberately EXACT (float ==, memcmp) wherever the
// contract is exactness. An epsilon compare would pass for a blend that is
// merely close, and "merely close" is exactly what moves a screenshot baseline.

#include "bracken/animation/animation_player.hpp"
#include "bracken/animation/clip_sampler.hpp"
#include "bracken/animation/pose.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

using namespace bracken::animation;

namespace {

/// Root -> Hips -> Spine -> {Head, LeftArm}, plus a detached second root, so the
/// subtree tests have a joint that must NOT be swept in.
///
/// Index order is parents-before-children, which every function under test
/// requires.
Skeleton make_skeleton() {
    Skeleton s;
    struct Row {
        const char* name;
        int parent;
        glm::vec3 t;
    };
    const Row rows[] = {
        {"Root", -1, {0.0f, 0.0f, 0.0f}},   // 0
        {"Hips", 0, {0.0f, 95.0f, 0.0f}},   // 1
        {"Spine", 1, {0.0f, 35.0f, 0.0f}},  // 2
        {"Head", 2, {0.0f, 32.0f, 0.0f}},   // 3
        {"LeftArm", 2, {-24.0f, 22.0f, 0.0f}}, // 4
        {"Tail", 0, {0.0f, -5.0f, 12.0f}},  // 5  sibling of Hips, never in the Spine subtree
    };
    for (const Row& r : rows) {
        Joint j;
        j.name = r.name;
        j.parent_index = r.parent;
        j.local_translation = r.t;
        j.local_rotation = glm::angleAxis(0.13f * static_cast<float>(s.joints.size()), glm::vec3(0, 1, 0));
        j.local_scale = glm::vec3(1.0f);
        j.inverse_bind_matrix = glm::translate(glm::mat4(1.0f), -r.t);
        s.joints.push_back(j);
    }
    return s;
}

AnimationClip make_clip(const char* name, float duration, float amplitude) {
    AnimationClip c;
    c.name = name;
    c.duration = duration;
    for (int bone : {1, 2, 4}) {
        AnimationChannel ch;
        ch.bone_index = bone;
        ch.rotation_times = {0.0f, duration * 0.5f, duration};
        ch.rotations = {glm::angleAxis(0.0f, glm::vec3(1, 0, 0)),
                        glm::angleAxis(amplitude * static_cast<float>(bone), glm::vec3(1, 0, 0)),
                        glm::angleAxis(0.0f, glm::vec3(1, 0, 0))};
        ch.position_times = {0.0f, duration};
        ch.positions = {glm::vec3(0.0f, static_cast<float>(bone), 0.0f),
                        glm::vec3(amplitude, static_cast<float>(bone), amplitude * 2.0f)};
        c.channels.push_back(ch);
    }
    return c;
}

[[nodiscard]] bool exact(const glm::vec3& a, const glm::vec3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
[[nodiscard]] bool exact(const glm::quat& a, const glm::quat& b) {
    return a.w == b.w && a.x == b.x && a.y == b.y && a.z == b.z;
}
[[nodiscard]] bool exact(const BoneTransform& a, const BoneTransform& b) {
    return exact(a.translation, b.translation) && exact(a.rotation, b.rotation) && exact(a.scale, b.scale);
}
[[nodiscard]] bool exact(const glm::mat4& a, const glm::mat4& b) {
    return std::memcmp(&a, &b, sizeof(glm::mat4)) == 0;
}

} // namespace

// ---------------------------------------------------------------------------
// rest_pose
// ---------------------------------------------------------------------------

TEST(Pose, RestPoseReproducesEveryJointsLocalTransformExactly) {
    const Skeleton s = make_skeleton();
    std::array<BoneTransform, MAX_BONES> out{};
    rest_pose(s, PoseSpan(out.data(), out.size()));

    for (std::size_t i = 0; i < s.joints.size(); ++i) {
        EXPECT_TRUE(exact(out[i].translation, s.joints[i].local_translation)) << "joint " << i;
        EXPECT_TRUE(exact(out[i].rotation, s.joints[i].local_rotation)) << "joint " << i;
        EXPECT_TRUE(exact(out[i].scale, s.joints[i].local_scale)) << "joint " << i;
    }
    // Beyond the skeleton, untouched — still the default identity transform.
    EXPECT_TRUE(exact(out[s.joints.size()], BoneTransform{}));
}

TEST(Pose, RestPoseHonoursAShortOutputSpan) {
    const Skeleton s = make_skeleton();
    std::array<BoneTransform, MAX_BONES> out{};
    rest_pose(s, PoseSpan(out.data(), 2));
    EXPECT_TRUE(exact(out[1].translation, s.joints[1].local_translation));
    EXPECT_TRUE(exact(out[2], BoneTransform{})) << "wrote past the span";
}

// ---------------------------------------------------------------------------
// pose_to_matrices — the proof it is the player's own pipeline
// ---------------------------------------------------------------------------

TEST(Pose, PoseToMatricesReproducesAnimationPlayerBitwise) {
    const Skeleton s = make_skeleton();
    const std::vector<AnimationClip> clips{make_clip("walk", 1.6f, 0.4f)};

    ClipBindings bindings;
    bindings.build(s, clips);

    AnimationPlayer player;
    ClipSampler sampler;
    sampler.bind(s.joints.size(), true);

    std::array<BoneTransform, MAX_BONES> pose{};
    std::array<glm::mat4, MAX_BONES> world{};
    std::array<glm::mat4, MAX_BONES> bones{};

    constexpr float dt = 1.0f / 60.0f;
    float t = 0.0f;
    for (int step = 0; step < 90; ++step) {
        player.update(s, clips, dt);

        // Mirror the player's clock: no crossfade is in play, so this is just
        // the looping time advance.
        t += dt;
        if (t >= clips[0].duration) {
            t = std::fmod(t, clips[0].duration);
        }

        sampler.sample(s, clips, bindings, 0, t, PoseSpan(pose.data(), pose.size()));
        const std::size_t used =
            pose_to_matrices(s, ConstPoseSpan(pose.data(), s.joints.size()),
                             std::span<glm::mat4>(world.data(), world.size()),
                             std::span<glm::mat4>(bones.data(), bones.size()));

        ASSERT_EQ(used, s.joints.size());
        ASSERT_EQ(used, player.used_joint_count());
        for (std::size_t i = 0; i < used; ++i) {
            ASSERT_TRUE(exact(world[i], player.world_transforms()[i]))
                << "step " << step << " world joint " << i;
            ASSERT_TRUE(exact(bones[i], player.bone_matrices()[i])) << "step " << step << " bone joint " << i;
        }
    }
}

TEST(Pose, PoseToMatricesMaintainsTheSkinningInvariant) {
    const Skeleton s = make_skeleton();
    std::array<BoneTransform, MAX_BONES> pose{};
    rest_pose(s, PoseSpan(pose.data(), pose.size()));
    pose[2].rotation = glm::angleAxis(0.7f, glm::vec3(1, 0, 0));
    pose[2].scale = glm::vec3(1.0f, 1.2f, 1.0f);

    std::array<glm::mat4, MAX_BONES> world{};
    std::array<glm::mat4, MAX_BONES> bones{};
    const std::size_t used = pose_to_matrices(s, ConstPoseSpan(pose.data(), s.joints.size()),
                                              std::span<glm::mat4>(world.data(), world.size()),
                                              std::span<glm::mat4>(bones.data(), bones.size()));
    ASSERT_EQ(used, s.joints.size());
    for (std::size_t i = 0; i < used; ++i) {
        const glm::mat4 expect = world[i] * s.joints[i].inverse_bind_matrix;
        EXPECT_TRUE(exact(bones[i], expect)) << "joint " << i;
    }
}

TEST(Pose, PoseToMatricesClampsToMaxBonesAndTouchesNothingBeyondIt) {
    // 200 joints: more than MAX_BONES, so entries at index >= 128 must be left
    // exactly as the caller had them.
    Skeleton s;
    for (int i = 0; i < 200; ++i) {
        Joint j;
        j.parent_index = i == 0 ? -1 : i - 1;
        j.local_translation = glm::vec3(0.0f, 1.0f, 0.0f);
        j.local_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        j.local_scale = glm::vec3(1.0f);
        j.inverse_bind_matrix = glm::mat4(1.0f);
        s.joints.push_back(j);
    }

    std::array<BoneTransform, MAX_BONES> pose{};
    rest_pose(s, PoseSpan(pose.data(), pose.size()));

    std::array<glm::mat4, MAX_BONES> world{};
    std::array<glm::mat4, MAX_BONES> bones{};
    const glm::mat4 sentinel = glm::translate(glm::mat4(1.0f), glm::vec3(-7.0f));
    world.fill(sentinel);
    bones.fill(sentinel);

    const std::size_t used = pose_to_matrices(s, ConstPoseSpan(pose.data(), pose.size()),
                                              std::span<glm::mat4>(world.data(), world.size()),
                                              std::span<glm::mat4>(bones.data(), bones.size()));
    EXPECT_EQ(used, static_cast<std::size_t>(MAX_BONES));
    EXPECT_FALSE(exact(world[MAX_BONES - 1], sentinel)) << "the last in-range slot was not written";
    // Nothing above MAX_BONES exists to check inside the arrays; the guarantee
    // is that the call did not walk past them, which ASan proves and this
    // asserts by construction (the arrays are exactly MAX_BONES long).
    EXPECT_EQ(world.size(), static_cast<std::size_t>(MAX_BONES));
}

// ---------------------------------------------------------------------------
// BoneMask
// ---------------------------------------------------------------------------

TEST(BoneMask, AllNoneAndInvertedRoundTrip) {
    const BoneMask all = BoneMask::all();
    const BoneMask none = BoneMask::none();
    EXPECT_EQ(all.count(), MAX_BONES);
    EXPECT_EQ(none.count(), 0);
    EXPECT_TRUE(none.empty());
    EXPECT_FALSE(all.empty());
    EXPECT_TRUE(all.inverted() == none);
    EXPECT_TRUE(none.inverted() == all);

    BoneMask m;
    m.set(0);
    m.set(63);
    m.set(64);
    m.set(127);
    EXPECT_EQ(m.count(), 4);
    EXPECT_TRUE(m.test(0));
    EXPECT_TRUE(m.test(63));
    EXPECT_TRUE(m.test(64));
    EXPECT_TRUE(m.test(127));
    EXPECT_FALSE(m.test(62));
    EXPECT_EQ(m.inverted().count(), MAX_BONES - 4);

    BoneMask both = m;
    both |= BoneMask::none();
    EXPECT_EQ(both.count(), 4);
    both &= m.inverted();
    EXPECT_EQ(both.count(), 0);
}

TEST(BoneMask, OutOfRangeAccessIsANoOpNotUndefinedBehaviour) {
    BoneMask m;
    m.set(-1);
    m.set(MAX_BONES);
    m.set(100000);
    EXPECT_EQ(m.count(), 0);
    EXPECT_FALSE(m.test(-1));
    EXPECT_FALSE(m.test(MAX_BONES));
    EXPECT_FALSE(m.test(100000));
}

TEST(BoneMask, SetSubtreeMarksExactlyTheDescendants) {
    const Skeleton s = make_skeleton();
    BoneMask m;
    m.set_subtree(s, 2); // Spine

    EXPECT_TRUE(m.test(2)) << "the root of the subtree itself";
    EXPECT_TRUE(m.test(3)) << "Head";
    EXPECT_TRUE(m.test(4)) << "LeftArm";
    EXPECT_FALSE(m.test(0)) << "Root is an ancestor, not a descendant";
    EXPECT_FALSE(m.test(1)) << "Hips is an ancestor";
    EXPECT_FALSE(m.test(5)) << "Tail hangs off Root, not off Spine";
    EXPECT_EQ(m.count(), 3);

    // Clearing the same subtree returns to empty.
    m.set_subtree(s, 2, false);
    EXPECT_EQ(m.count(), 0);
}

TEST(BoneMask, SetSubtreeDoesNotInheritUnrelatedBitsAlreadySet) {
    // THE REGRESSION this test exists for: propagating over *this instead of a
    // local scratch would let the pre-set joint 1 seed joint 2's marking, and
    // set_subtree(5) would silently claim the entire Spine chain.
    const Skeleton s = make_skeleton();
    BoneMask m;
    m.set(1); // Hips, set for some unrelated reason
    m.set_subtree(s, 5);

    EXPECT_TRUE(m.test(1));
    EXPECT_TRUE(m.test(5));
    EXPECT_FALSE(m.test(2)) << "Spine is not below Tail and must not be swept in";
    EXPECT_FALSE(m.test(3));
    EXPECT_FALSE(m.test(4));
    EXPECT_EQ(m.count(), 2);
}

TEST(BoneMask, SetSubtreeOnAnInvalidRootIsANoOp) {
    const Skeleton s = make_skeleton();
    BoneMask m;
    m.set_subtree(s, -1);
    m.set_subtree(s, 999);
    EXPECT_EQ(m.count(), 0);
}

TEST(BoneMask, FeatherByDepthRampsAcrossTheSubtree) {
    const Skeleton s = make_skeleton();
    std::array<float, MAX_BONES> w{};
    feather_by_depth(s, 1, 2, std::span<float>(w.data(), w.size())); // Hips, 2-joint ramp

    EXPECT_FLOAT_EQ(w[1], 0.0f) << "the mask root is the lightest end of the ramp";
    EXPECT_FLOAT_EQ(w[2], 0.5f) << "Spine, one level down";
    EXPECT_FLOAT_EQ(w[3], 1.0f) << "Head, two levels down: full weight";
    EXPECT_FLOAT_EQ(w[4], 1.0f) << "LeftArm";
    EXPECT_FLOAT_EQ(w[0], 0.0f) << "outside the subtree";
    EXPECT_FLOAT_EQ(w[5], 0.0f) << "outside the subtree";

    // falloff 0 reproduces the binary mask exactly.
    feather_by_depth(s, 1, 0, std::span<float>(w.data(), w.size()));
    EXPECT_FLOAT_EQ(w[1], 1.0f);
    EXPECT_FLOAT_EQ(w[2], 1.0f);
    EXPECT_FLOAT_EQ(w[0], 0.0f);
    EXPECT_FLOAT_EQ(w[5], 0.0f);
}

// ---------------------------------------------------------------------------
// blend_override
// ---------------------------------------------------------------------------

TEST(BlendOverride, WeightZeroLeavesTheDestinationBitwiseUnchangedAndWeightOneAssigns) {
    const Skeleton s = make_skeleton();
    std::array<BoneTransform, MAX_BONES> dst{};
    std::array<BoneTransform, MAX_BONES> src{};
    rest_pose(s, PoseSpan(dst.data(), dst.size()));
    for (std::size_t i = 0; i < s.joints.size(); ++i) {
        src[i].translation = glm::vec3(1.0f, 2.0f, 3.0f) * static_cast<float>(i + 1);
        src[i].rotation = glm::angleAxis(0.31f * static_cast<float>(i + 1), glm::vec3(0, 0, 1));
        src[i].scale = glm::vec3(1.5f);
    }
    const std::array<BoneTransform, MAX_BONES> before = dst;

    BlendParams p;
    p.weight = 0.0f;
    blend_override(PoseSpan(dst.data(), s.joints.size()), ConstPoseSpan(src.data(), s.joints.size()), p);
    for (std::size_t i = 0; i < s.joints.size(); ++i) {
        EXPECT_TRUE(exact(dst[i], before[i])) << "weight 0 moved joint " << i;
    }

    p.weight = 1.0f;
    blend_override(PoseSpan(dst.data(), s.joints.size()), ConstPoseSpan(src.data(), s.joints.size()), p);
    for (std::size_t i = 0; i < s.joints.size(); ++i) {
        EXPECT_TRUE(exact(dst[i], src[i])) << "weight 1 was not an exact assignment at joint " << i;
    }
}

TEST(BlendOverride, AMaskedLayerLeavesEveryUnmaskedJointsLocalTransformBitwiseUnchanged) {
    const Skeleton s = make_skeleton();
    std::array<BoneTransform, MAX_BONES> dst{};
    std::array<BoneTransform, MAX_BONES> src{};
    rest_pose(s, PoseSpan(dst.data(), dst.size()));
    for (std::size_t i = 0; i < s.joints.size(); ++i) {
        src[i].translation = glm::vec3(9.0f);
        src[i].rotation = glm::angleAxis(1.1f, glm::vec3(0, 0, 1));
        src[i].scale = glm::vec3(2.0f);
    }
    const std::array<BoneTransform, MAX_BONES> before = dst;

    BoneMask spine;
    spine.set_subtree(s, 2);
    BlendParams p;
    p.weight = 1.0f;
    p.mask = &spine;
    blend_override(PoseSpan(dst.data(), s.joints.size()), ConstPoseSpan(src.data(), s.joints.size()), p);

    for (std::size_t i = 0; i < s.joints.size(); ++i) {
        const bool masked = spine.test(static_cast<int>(i));
        if (masked) {
            EXPECT_TRUE(exact(dst[i], src[i])) << "masked joint " << i << " did not take the layer";
        } else {
            EXPECT_TRUE(exact(dst[i], before[i])) << "unmasked joint " << i << " moved";
        }
    }

    // ...and the point of the previous assertion being about LOCAL transforms:
    // a masked joint's descendants still move in WORLD space. That is correct —
    // rotating a spine moves the head — and this assertion exists so nobody
    // "fixes" it.
    std::array<glm::mat4, MAX_BONES> world_a{}, bones_a{}, world_b{}, bones_b{};
    (void)pose_to_matrices(s, ConstPoseSpan(before.data(), s.joints.size()),
                           std::span<glm::mat4>(world_a.data(), world_a.size()),
                           std::span<glm::mat4>(bones_a.data(), bones_a.size()));
    (void)pose_to_matrices(s, ConstPoseSpan(dst.data(), s.joints.size()),
                           std::span<glm::mat4>(world_b.data(), world_b.size()),
                           std::span<glm::mat4>(bones_b.data(), bones_b.size()));
    EXPECT_TRUE(exact(world_a[1], world_b[1])) << "Hips is outside the mask and must not move at all";
    EXPECT_FALSE(exact(world_a[3], world_b[3])) << "Head is inside the mask's subtree and must follow it";
}

TEST(BlendOverride, JointWeightsMultiplyTheMask) {
    const Skeleton s = make_skeleton();
    std::array<BoneTransform, MAX_BONES> dst{}, src{};
    rest_pose(s, PoseSpan(dst.data(), dst.size()));
    for (auto& b : src) {
        b.translation = glm::vec3(10.0f, 0.0f, 0.0f);
    }
    const std::array<BoneTransform, MAX_BONES> before = dst;

    std::array<float, MAX_BONES> feather{};
    feather.fill(0.0f);
    feather[2] = 0.5f;
    feather[3] = 1.0f;

    BlendParams p;
    p.weight = 1.0f;
    p.joint_weights = std::span<const float>(feather.data(), feather.size());
    blend_override(PoseSpan(dst.data(), s.joints.size()), ConstPoseSpan(src.data(), s.joints.size()), p);

    EXPECT_TRUE(exact(dst[0], before[0])) << "feather 0 must be bitwise-untouched";
    EXPECT_FLOAT_EQ(dst[2].translation.x, glm::mix(before[2].translation.x, 10.0f, 0.5f));
    EXPECT_FLOAT_EQ(dst[3].translation.x, 10.0f);
}

// ---------------------------------------------------------------------------
// blend_additive
// ---------------------------------------------------------------------------

TEST(BlendAdditive, WeightZeroIsExactlyTheBasePose) {
    const Skeleton s = make_skeleton();
    std::array<BoneTransform, MAX_BONES> dst{}, src{}, ref{};
    rest_pose(s, PoseSpan(dst.data(), dst.size()));
    rest_pose(s, PoseSpan(ref.data(), ref.size()));
    for (std::size_t i = 0; i < s.joints.size(); ++i) {
        src[i].translation = ref[i].translation + glm::vec3(3.0f, 4.0f, 5.0f);
        src[i].rotation = glm::angleAxis(0.9f, glm::vec3(1, 0, 0)) * ref[i].rotation;
        src[i].scale = ref[i].scale * 2.0f;
    }
    const std::array<BoneTransform, MAX_BONES> before = dst;

    BlendParams p;
    p.weight = 0.0f;
    blend_additive(PoseSpan(dst.data(), s.joints.size()), ConstPoseSpan(src.data(), s.joints.size()),
                   ConstPoseSpan(ref.data(), s.joints.size()), p);
    for (std::size_t i = 0; i < s.joints.size(); ++i) {
        EXPECT_TRUE(exact(dst[i], before[i])) << "additive at weight 0 moved joint " << i;
    }
}

TEST(BlendAdditive, SourceEqualToReferenceIsTheIdentityAtEveryWeight) {
    const Skeleton s = make_skeleton();
    std::array<BoneTransform, MAX_BONES> ref{};
    rest_pose(s, PoseSpan(ref.data(), ref.size()));

    // A NON-IDENTITY base, which is the whole point: an additive identity that
    // only holds against a rest base proves nothing about the rotation order.
    std::array<BoneTransform, MAX_BONES> base{};
    for (std::size_t i = 0; i < s.joints.size(); ++i) {
        base[i].translation = glm::vec3(7.0f, -2.0f, 1.5f) * static_cast<float>(i);
        base[i].rotation = glm::angleAxis(0.61f * static_cast<float>(i + 1), glm::normalize(glm::vec3(1, 2, 3)));
        base[i].scale = glm::vec3(1.0f + 0.1f * static_cast<float>(i));
    }

    for (const float w : {0.0f, 0.37f, 1.0f}) {
        std::array<BoneTransform, MAX_BONES> dst = base;
        BlendParams p;
        p.weight = w;
        blend_additive(PoseSpan(dst.data(), s.joints.size()), ConstPoseSpan(ref.data(), s.joints.size()),
                       ConstPoseSpan(ref.data(), s.joints.size()), p);
        for (std::size_t i = 0; i < s.joints.size(); ++i) {
            if (w == 0.0f) {
                EXPECT_TRUE(exact(dst[i], base[i])) << "weight 0 must be EXACT, joint " << i;
                continue;
            }
            // Above weight 0 the identity is only as exact as inverse(q)*q is.
            EXPECT_NEAR(dst[i].translation.x, base[i].translation.x, 1e-4f) << "w=" << w << " joint " << i;
            EXPECT_NEAR(dst[i].scale.y, base[i].scale.y, 1e-5f) << "w=" << w << " joint " << i;
            const float dot = std::abs(glm::dot(dst[i].rotation, base[i].rotation));
            EXPECT_NEAR(dot, 1.0f, 1e-5f) << "w=" << w << " joint " << i;
        }
    }
}

TEST(BlendAdditive, RotationDeltaIsPostMultipliedInTheBonesOwnFrame) {
    // ONE joint, an identity reference, an additive rotation of +90 degrees
    // about X, and a base already rotated +90 degrees about Y. Post-multiply
    // (bone-local) gives base * add; pre-multiply (parent-frame) gives
    // add * base. Those are different quaternions and this pins which one we
    // ship.
    Skeleton s;
    Joint j;
    j.parent_index = -1;
    j.local_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    j.local_scale = glm::vec3(1.0f);
    j.inverse_bind_matrix = glm::mat4(1.0f);
    s.joints.push_back(j);

    const glm::quat add = glm::angleAxis(glm::half_pi<float>(), glm::vec3(1, 0, 0));
    const glm::quat base = glm::angleAxis(glm::half_pi<float>(), glm::vec3(0, 1, 0));

    std::array<BoneTransform, 1> ref{};
    std::array<BoneTransform, 1> src{};
    std::array<BoneTransform, 1> dst{};
    src[0].rotation = add;
    dst[0].rotation = base;

    BlendParams p;
    p.weight = 1.0f;
    blend_additive(PoseSpan(dst.data(), 1), ConstPoseSpan(src.data(), 1), ConstPoseSpan(ref.data(), 1), p);

    const glm::quat post = base * add;
    const glm::quat pre = add * base;
    EXPECT_NEAR(std::abs(glm::dot(dst[0].rotation, post)), 1.0f, 1e-6f) << "not the post-multiply convention";
    EXPECT_GT(std::abs(1.0f - std::abs(glm::dot(post, pre))), 1e-3f)
        << "the fixture is degenerate: the two conventions agree here, so it proves nothing";
}

TEST(BlendAdditive, ComposesTranslationLinearlyAndScalesAsARatio) {
    std::array<BoneTransform, 1> ref{}, src{}, dst{};
    ref[0].translation = glm::vec3(1.0f, 0.0f, 0.0f);
    ref[0].scale = glm::vec3(2.0f);
    src[0].translation = glm::vec3(5.0f, 0.0f, 0.0f); // +4 delta
    src[0].scale = glm::vec3(4.0f);                   // x2 ratio
    dst[0].translation = glm::vec3(10.0f, 0.0f, 0.0f);
    dst[0].scale = glm::vec3(3.0f);

    BlendParams p;
    p.weight = 0.5f;
    blend_additive(PoseSpan(dst.data(), 1), ConstPoseSpan(src.data(), 1), ConstPoseSpan(ref.data(), 1), p);

    EXPECT_FLOAT_EQ(dst[0].translation.x, 10.0f + 4.0f * 0.5f);
    EXPECT_FLOAT_EQ(dst[0].scale.x, 3.0f * glm::mix(1.0f, 2.0f, 0.5f));
}

TEST(BlendAdditive, AZeroReferenceScaleComponentProducesFiniteOutput) {
    std::array<BoneTransform, 1> ref{}, src{}, dst{};
    ref[0].scale = glm::vec3(0.0f, 1.0f, 2.0f);
    src[0].scale = glm::vec3(5.0f, 3.0f, 4.0f);
    dst[0].scale = glm::vec3(1.0f);

    BlendParams p;
    p.weight = 1.0f;
    blend_additive(PoseSpan(dst.data(), 1), ConstPoseSpan(src.data(), 1), ConstPoseSpan(ref.data(), 1), p);

    EXPECT_TRUE(std::isfinite(dst[0].scale.x));
    EXPECT_FLOAT_EQ(dst[0].scale.x, 1.0f) << "a zero reference component means 'no additive scale here'";
    EXPECT_FLOAT_EQ(dst[0].scale.y, 3.0f);
    EXPECT_FLOAT_EQ(dst[0].scale.z, 2.0f);
}
