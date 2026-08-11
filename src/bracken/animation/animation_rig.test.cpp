// AnimationRig: the layer stack, and THE GATE that keeps it a strict superset
// of AnimationPlayer.
//
// The first two tests are the regression that matters. The engine's screenshot
// baselines are pinned against the playground character's final pose, and the
// rig duplicates AnimationPlayer's sampler and clock; if either copy drifts, the
// bitwise comparisons below go red before anything reaches a screenshot.

#include "bracken/animation/animation_player.hpp"
#include "bracken/animation/animation_rig.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <vector>

using namespace bracken::animation;

namespace {

/// Root -> Hips -> Spine -> {Head, LeftArm}; Hips -> LeftLeg. Deep enough for
/// the subtree mask to have a real inside and outside.
Skeleton make_skeleton(int joints = 6) {
    Skeleton s;
    const int parents[] = {-1, 0, 1, 2, 2, 1};
    for (int i = 0; i < joints; ++i) {
        Joint j;
        j.name = "joint" + std::to_string(i);
        j.parent_index = (i < 6) ? parents[i] : i - 1;
        j.local_translation = glm::vec3(0.0f, 10.0f + static_cast<float>(i), static_cast<float>(i) * 0.5f);
        j.local_rotation = glm::angleAxis(0.07f * static_cast<float>(i), glm::normalize(glm::vec3(0, 1, 0.2f)));
        j.local_scale = glm::vec3(1.0f);
        j.inverse_bind_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -10.0f, 0.0f));
        s.joints.push_back(j);
    }
    return s;
}

AnimationClip make_clip(const char* name, float duration, float amplitude, int first_bone = 1) {
    AnimationClip c;
    c.name = name;
    c.duration = duration;
    for (int bone = first_bone; bone < first_bone + 3; ++bone) {
        AnimationChannel ch;
        ch.bone_index = bone;
        ch.rotation_times = {0.0f, duration * 0.33f, duration * 0.66f, duration};
        ch.rotations = {glm::angleAxis(0.0f, glm::vec3(1, 0, 0)),
                        glm::angleAxis(amplitude, glm::vec3(1, 0, 0)),
                        glm::angleAxis(-amplitude, glm::vec3(0, 0, 1)),
                        glm::angleAxis(0.0f, glm::vec3(1, 0, 0))};
        ch.position_times = {0.0f, duration};
        ch.positions = {glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(amplitude * 5.0f, 10.0f, 0.0f)};
        c.channels.push_back(ch);
    }
    return c;
}

std::vector<AnimationClip> make_clips() {
    return {make_clip("A", 1.4f, 0.5f), make_clip("B", 0.9f, -0.8f, 2), make_clip("C", 2.1f, 0.25f)};
}

[[nodiscard]] bool exact(const glm::mat4& a, const glm::mat4& b) {
    return std::memcmp(&a, &b, sizeof(glm::mat4)) == 0;
}
[[nodiscard]] bool exact(const BoneTransform& a, const BoneTransform& b) {
    return a.translation == b.translation && a.rotation == b.rotation && a.scale == b.scale;
}

/// A rig holding one full-weight Override layer — the configuration that must
/// reproduce AnimationPlayer.
struct SingleLayerRig {
    Skeleton skeleton;
    std::vector<AnimationClip> clips;
    ClipBindings bindings;
    AnimationRig rig;
    PoseScratch scratch;

    explicit SingleLayerRig(int joints = 6) : skeleton(make_skeleton(joints)), clips(make_clips()) {
        bindings.build(skeleton, clips);
        rig.bind(skeleton, clips, bindings);
        LayerDesc base;
        base.name = "base";
        EXPECT_EQ(rig.add_layer(std::move(base)), 0);
    }

    void step(float dt) { rig.evaluate(skeleton, clips, dt, scratch); }
};

} // namespace

// ---------------------------------------------------------------------------
// THE GATE
// ---------------------------------------------------------------------------

TEST(AnimationRig, SingleOverrideLayerMatchesAnimationPlayerBitwise) {
    SingleLayerRig fixture;
    AnimationPlayer player;

    constexpr float dt = 1.0f / 60.0f;
    for (int step = 0; step < 120; ++step) {
        player.update(fixture.skeleton, fixture.clips, dt);
        fixture.step(dt);

        ASSERT_EQ(fixture.rig.used_joint_count(), player.used_joint_count()) << "step " << step;
        for (std::size_t i = 0; i < player.used_joint_count(); ++i) {
            ASSERT_TRUE(exact(fixture.rig.bone_matrices()[i], player.bone_matrices()[i]))
                << "step " << step << " bone " << i;
            ASSERT_TRUE(exact(fixture.rig.world_transforms()[i], player.world_transforms()[i]))
                << "step " << step << " world " << i;
        }
    }
}

TEST(AnimationRig, SingleOverrideLayerReproducesACrossfadeBitwise) {
    SingleLayerRig fixture;
    AnimationPlayer player;

    constexpr float dt = 1.0f / 60.0f;
    for (int step = 0; step < 120; ++step) {
        if (step == 30) {
            player.crossfade_to(1, 0.25f);
            fixture.rig.play_clip(0, 1, 0.25f);
        }
        if (step == 75) {
            player.crossfade_to(2, 0.4f);
            fixture.rig.play_clip(0, 2, 0.4f);
        }
        player.update(fixture.skeleton, fixture.clips, dt);
        fixture.step(dt);

        ASSERT_EQ(fixture.rig.used_joint_count(), player.used_joint_count()) << "step " << step;
        for (std::size_t i = 0; i < player.used_joint_count(); ++i) {
            ASSERT_TRUE(exact(fixture.rig.bone_matrices()[i], player.bone_matrices()[i]))
                << "step " << step << " bone " << i;
        }
    }
}

TEST(AnimationRig, WritePaletteMatchesAnimationPlayerIncludingTheIdentityTail) {
    SingleLayerRig fixture;
    AnimationPlayer player;
    for (int step = 0; step < 20; ++step) {
        player.update(fixture.skeleton, fixture.clips, 1.0f / 60.0f);
        fixture.step(1.0f / 60.0f);
    }

    // A destination LARGER than the used count: everything past the prefix must
    // be identity, because alloc_bone_palette hands back uninitialized memory.
    std::vector<glm::mat4> a(40, glm::mat4(7.0f));
    std::vector<glm::mat4> b(40, glm::mat4(7.0f));
    player.write_palette(a);
    fixture.rig.write_palette(b);
    for (std::size_t i = 0; i < a.size(); ++i) {
        ASSERT_TRUE(exact(a[i], b[i])) << "palette slot " << i;
    }
    for (std::size_t i = fixture.rig.used_joint_count(); i < b.size(); ++i) {
        EXPECT_TRUE(exact(b[i], glm::mat4(1.0f))) << "tail slot " << i << " was not identity-filled";
    }

    // ...and a destination SMALLER than the used count must not overrun.
    std::vector<glm::mat4> small(3, glm::mat4(7.0f));
    fixture.rig.write_palette(small);
    EXPECT_TRUE(exact(small[2], fixture.rig.bone_matrices()[2]));
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

TEST(AnimationRig, AnAdditiveLayerAtWeightZeroLeavesTheBaseBitwiseUnchanged) {
    SingleLayerRig fixture;
    for (int step = 0; step < 30; ++step) {
        fixture.step(1.0f / 60.0f);
    }
    std::array<glm::mat4, MAX_BONES> baseline{};
    std::copy(fixture.rig.bone_matrices().begin(), fixture.rig.bone_matrices().end(), baseline.begin());
    const float base_time = fixture.rig.layer_time(0);

    LayerDesc add;
    add.name = "additive";
    add.mode = BlendMode::Additive;
    add.weight = 0.0f;
    const int li = fixture.rig.add_layer(std::move(add));
    ASSERT_EQ(li, 1);
    fixture.rig.play_clip(li, 1, 0.0f);

    // Rewind the base to the same clock so the comparison is against the same
    // frame, then re-evaluate with the muted layer present.
    fixture.rig.set_layer_time(0, base_time);
    fixture.rig.evaluate(fixture.skeleton, fixture.clips, 0.0f, fixture.scratch);

    for (std::size_t i = 0; i < fixture.rig.used_joint_count(); ++i) {
        ASSERT_TRUE(exact(fixture.rig.bone_matrices()[i], baseline[i]))
            << "a weight-0 additive layer moved bone " << i;
    }
}

TEST(AnimationRig, AnAdditiveLayerWhoseSourceIsItsOwnReferenceIsTheIdentity) {
    SingleLayerRig fixture;
    for (int step = 0; step < 17; ++step) {
        fixture.step(1.0f / 60.0f);
    }
    std::array<glm::mat4, MAX_BONES> baseline{};
    std::copy(fixture.rig.bone_matrices().begin(), fixture.rig.bone_matrices().end(), baseline.begin());
    const float base_time = fixture.rig.layer_time(0);

    LayerDesc add;
    add.name = "additive";
    add.mode = BlendMode::Additive;
    add.weight = 1.0f;
    add.additive_reference_clip = 1;
    add.additive_reference_time = 0.31f;
    const int li = fixture.rig.add_layer(std::move(add));
    ASSERT_EQ(li, 1);
    fixture.rig.play_clip(li, 1, 0.0f);
    fixture.rig.set_layer_time(li, 0.31f); // source == reference, exactly

    fixture.rig.set_layer_time(0, base_time);
    fixture.rig.evaluate(fixture.skeleton, fixture.clips, 0.0f, fixture.scratch);

    for (std::size_t i = 0; i < fixture.rig.used_joint_count(); ++i) {
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                ASSERT_NEAR(fixture.rig.bone_matrices()[i][c][r], baseline[i][c][r], 1e-3f)
                    << "bone " << i << " column " << c << " row " << r;
            }
        }
    }
}

TEST(AnimationRig, AMaskedLayerChangesNoLocalTransformOutsideItsSubtree) {
    SingleLayerRig fixture;
    for (int step = 0; step < 12; ++step) {
        fixture.step(1.0f / 60.0f);
    }
    std::array<BoneTransform, MAX_BONES> baseline{};
    std::copy(fixture.rig.local_pose().begin(), fixture.rig.local_pose().end(), baseline.begin());
    const float base_time = fixture.rig.layer_time(0);

    BoneMask spine;
    spine.set_subtree(fixture.skeleton, 2);
    ASSERT_EQ(spine.count(), 3) << "the fixture must have a real subtree to mask";

    LayerDesc upper;
    upper.name = "upper_body";
    upper.mode = BlendMode::Override;
    upper.weight = 1.0f;
    upper.mask = spine;
    const int li = fixture.rig.add_layer(std::move(upper));
    ASSERT_EQ(li, 1);
    fixture.rig.play_clip(li, 1, 0.0f);
    fixture.rig.set_layer_time(li, 0.5f);

    fixture.rig.set_layer_time(0, base_time);
    fixture.rig.evaluate(fixture.skeleton, fixture.clips, 0.0f, fixture.scratch);

    bool something_moved = false;
    for (std::size_t i = 0; i < fixture.rig.used_joint_count(); ++i) {
        if (spine.test(static_cast<int>(i))) {
            something_moved = something_moved || !exact(fixture.rig.local_pose()[i], baseline[i]);
        } else {
            ASSERT_TRUE(exact(fixture.rig.local_pose()[i], baseline[i]))
                << "joint " << i << " is outside the mask and moved";
        }
    }
    EXPECT_TRUE(something_moved) << "the masked layer changed nothing at all — the test proves nothing";
}

TEST(AnimationRig, LayerCompositionIsIndexOrderDependent) {
    // The order is a documented contract, not an accident. Two overlapping
    // Override layers at partial weight must produce different results when
    // swapped.
    auto run = [](int first_clip, int second_clip) {
        SingleLayerRig f;
        LayerDesc a;
        a.name = "a";
        a.weight = 0.5f;
        LayerDesc b;
        b.name = "b";
        b.weight = 0.5f;
        const int ia = f.rig.add_layer(std::move(a));
        const int ib = f.rig.add_layer(std::move(b));
        EXPECT_EQ(ia, 1);
        EXPECT_EQ(ib, 2);
        f.rig.play_clip(ia, first_clip, 0.0f);
        f.rig.play_clip(ib, second_clip, 0.0f);
        f.rig.set_layer_time(ia, 0.4f);
        f.rig.set_layer_time(ib, 0.4f);
        f.step(1.0f / 60.0f);
        std::array<glm::mat4, MAX_BONES> out{};
        std::copy(f.rig.bone_matrices().begin(), f.rig.bone_matrices().end(), out.begin());
        return out;
    };

    const auto ab = run(1, 2);
    const auto ba = run(2, 1);
    bool differs = false;
    for (std::size_t i = 0; i < 6; ++i) {
        differs = differs || !exact(ab[i], ba[i]);
    }
    EXPECT_TRUE(differs) << "swapping two partial-weight layers produced the same pose — order is not respected";
}

TEST(AnimationRig, AddLayerRefusesPastTheBudgetAndTheRigStaysEvaluable) {
    SingleLayerRig fixture; // already holds layer 0
    for (int i = 1; i < k_max_animation_layers; ++i) {
        LayerDesc d;
        d.name = "layer" + std::to_string(i);
        d.weight = 0.0f;
        ASSERT_EQ(fixture.rig.add_layer(std::move(d)), i);
    }
    LayerDesc overflow;
    overflow.name = "overflow";
    EXPECT_EQ(fixture.rig.add_layer(std::move(overflow)), -1);
    EXPECT_EQ(fixture.rig.layer_count(), k_max_animation_layers);

    fixture.step(1.0f / 60.0f);
    EXPECT_EQ(fixture.rig.used_joint_count(), fixture.skeleton.joints.size());
}

TEST(AnimationRig, OutOfRangeLayerAccessorsDegradeRatherThanCorrupt) {
    SingleLayerRig fixture;
    EXPECT_NO_FATAL_FAILURE(fixture.rig.set_layer_weight(99, 0.5f));
    EXPECT_NO_FATAL_FAILURE(fixture.rig.play_clip(-3, 1, 0.1f));
    EXPECT_NO_FATAL_FAILURE(fixture.rig.set_blend_position(99, 1.0f));
    EXPECT_FALSE(fixture.rig.layer_playing(99));
    EXPECT_EQ(fixture.rig.layer_clip(99), -1);
    EXPECT_FLOAT_EQ(fixture.rig.layer_time(99), 0.0f);

    // Writing through the discard reference must not affect a real layer.
    fixture.rig.layer(99).weight = 0.25f;
    EXPECT_FLOAT_EQ(fixture.rig.layer(0).weight, 1.0f);
}

TEST(AnimationRig, ClampsA200JointSkeletonToMaxBones) {
    Skeleton big;
    for (int i = 0; i < 200; ++i) {
        Joint j;
        j.parent_index = i == 0 ? -1 : i - 1;
        j.local_translation = glm::vec3(0.0f, 1.0f, 0.0f);
        j.local_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        j.local_scale = glm::vec3(1.0f);
        j.inverse_bind_matrix = glm::mat4(1.0f);
        big.joints.push_back(j);
    }
    const std::vector<AnimationClip> clips = make_clips();
    ClipBindings bindings;
    bindings.build(big, clips);

    AnimationRig rig;
    rig.bind(big, clips, bindings);
    LayerDesc base;
    base.name = "base";
    ASSERT_EQ(rig.add_layer(std::move(base)), 0);

    auto scratch = std::make_unique<PoseScratch>();
    rig.evaluate(big, clips, 1.0f / 60.0f, *scratch);
    EXPECT_EQ(rig.used_joint_count(), static_cast<std::size_t>(MAX_BONES));
    EXPECT_EQ(rig.bone_matrices().size(), static_cast<std::size_t>(MAX_BONES));
}

TEST(AnimationRig, ReBindingToAShorterSkeletonIdentitiesTheFreedTail) {
    SingleLayerRig fixture(6);
    fixture.step(1.0f / 60.0f);
    ASSERT_EQ(fixture.rig.used_joint_count(), 6u);
    ASSERT_FALSE(exact(fixture.rig.bone_matrices()[5], glm::mat4(1.0f)))
        << "joint 5 must be non-identity before the shrink for the test to mean anything";

    const Skeleton shorter = make_skeleton(3);
    ClipBindings short_bindings;
    short_bindings.build(shorter, fixture.clips);
    fixture.rig.bind(shorter, fixture.clips, short_bindings);
    fixture.rig.evaluate(shorter, fixture.clips, 1.0f / 60.0f, fixture.scratch);

    EXPECT_EQ(fixture.rig.used_joint_count(), 3u);
    for (std::size_t i = 3; i < 6; ++i) {
        EXPECT_TRUE(exact(fixture.rig.bone_matrices()[i], glm::mat4(1.0f)))
            << "freed slot " << i << " was not identity-filled";
        EXPECT_TRUE(exact(fixture.rig.world_transforms()[i], glm::mat4(1.0f))) << "freed world slot " << i;
    }
}

TEST(AnimationRig, ARigWithNoLayersProducesTheRestPose) {
    // AnimationPlayer leaves identity here (its update() returns early), which is
    // why a never-animated skeleton drops its attachments. The rig answers the
    // rest pose instead — a deliberate difference, documented on evaluate().
    const Skeleton s = make_skeleton();
    const std::vector<AnimationClip> clips = make_clips();
    ClipBindings bindings;
    bindings.build(s, clips);
    AnimationRig rig;
    rig.bind(s, clips, bindings);

    auto scratch = std::make_unique<PoseScratch>();
    rig.evaluate(s, clips, 1.0f / 60.0f, *scratch);
    EXPECT_EQ(rig.used_joint_count(), s.joints.size());

    std::array<BoneTransform, MAX_BONES> rest{};
    rest_pose(s, PoseSpan(rest.data(), rest.size()));
    for (std::size_t i = 0; i < s.joints.size(); ++i) {
        EXPECT_TRUE(exact(rig.local_pose()[i], rest[i])) << "joint " << i;
    }
}

// ---------------------------------------------------------------------------
// Blend spaces on a layer
// ---------------------------------------------------------------------------

TEST(AnimationRig, ABlendSpaceLayerAtASampleReproducesThatClipExactly) {
    SingleLayerRig fixture;

    BlendSpace1D space;
    ASSERT_TRUE(space.add_sample(0, 0.0f));
    ASSERT_TRUE(space.add_sample(1, 100.0f));
    ASSERT_TRUE(space.add_sample(2, 400.0f));
    space.finalize();
    fixture.rig.set_blend_space_1d(0, space);
    fixture.rig.layer(0).sync_duration = false;
    fixture.rig.set_blend_position(0, 400.0f);

    // A second rig playing clip 2 straight, at the same clock.
    SingleLayerRig plain;
    plain.rig.play_clip(0, 2, 0.0f);

    for (int step = 0; step < 40; ++step) {
        fixture.step(1.0f / 60.0f);
        plain.step(1.0f / 60.0f);
    }
    for (std::size_t i = 0; i < plain.rig.used_joint_count(); ++i) {
        ASSERT_TRUE(exact(fixture.rig.bone_matrices()[i], plain.rig.bone_matrices()[i]))
            << "blend space pinned to a sample diverged from the plain clip at bone " << i;
    }
}

TEST(AnimationRig, ABlendSpaceBetweenTwoSamplesLandsBetweenThem) {
    SingleLayerRig fixture;
    BlendSpace1D space;
    ASSERT_TRUE(space.add_sample(0, 0.0f));
    ASSERT_TRUE(space.add_sample(1, 100.0f));
    space.finalize();
    fixture.rig.set_blend_space_1d(0, space);
    fixture.rig.layer(0).sync_duration = false;

    auto pose_at = [&](float x) {
        SingleLayerRig f;
        f.rig.set_blend_space_1d(0, space);
        f.rig.layer(0).sync_duration = false;
        f.rig.set_blend_position(0, x);
        f.step(0.25f);
        std::array<BoneTransform, MAX_BONES> out{};
        std::copy(f.rig.local_pose().begin(), f.rig.local_pose().end(), out.begin());
        return out;
    };

    const auto lo = pose_at(0.0f);
    const auto mid = pose_at(50.0f);
    const auto hi = pose_at(100.0f);

    bool between = false;
    for (std::size_t i = 0; i < 6; ++i) {
        if (exact(lo[i], hi[i])) {
            continue; // this joint is the same in both clips; it proves nothing
        }
        EXPECT_FALSE(exact(mid[i], lo[i])) << "joint " << i << " did not leave the low sample";
        EXPECT_FALSE(exact(mid[i], hi[i])) << "joint " << i << " snapped to the high sample";
        between = true;
    }
    EXPECT_TRUE(between) << "the two samples produce identical poses — the fixture proves nothing";
}

TEST(AnimationRig, PhaseSyncKeepsBothSamplesAtTheSameNormalizedPhase) {
    // Clips A (1.4 s) and C (2.1 s) have different lengths. With sync on, a
    // half-and-half blend must advance both to the same fraction of their own
    // duration; without it they drift, which is what makes a walk/run blend
    // skate.
    SingleLayerRig fixture;
    BlendSpace1D space;
    ASSERT_TRUE(space.add_sample(0, 0.0f));
    ASSERT_TRUE(space.add_sample(2, 1.0f));
    space.finalize();
    fixture.rig.set_blend_space_1d(0, space);
    fixture.rig.layer(0).sync_duration = true;
    fixture.rig.set_blend_position(0, 0.5f);

    // effective_duration for a 50/50 blend of 1.4 s and 2.1 s is 1.75 s, so
    // advancing by a quarter of that must land the layer at phase 0.25 —
    // meaning BOTH clips are evaluated a quarter of the way through their own
    // (different) durations.
    constexpr float k_effective = 0.5f * 1.4f + 0.5f * 2.1f;
    for (int i = 0; i < 4; ++i) {
        fixture.step(k_effective / 16.0f);
    }
    EXPECT_NEAR(fixture.rig.layer_phase(0), 0.25f, 1e-4f);
    EXPECT_NEAR(fixture.rig.layer_time(0), 4.0f * k_effective / 16.0f, 1e-5f);
    EXPECT_EQ(fixture.rig.used_joint_count(), fixture.skeleton.joints.size());

    // Phase wraps rather than growing without bound.
    for (int i = 0; i < 40; ++i) {
        fixture.step(k_effective / 8.0f);
    }
    EXPECT_GE(fixture.rig.layer_phase(0), 0.0f);
    EXPECT_LT(fixture.rig.layer_phase(0), 1.0f);

    // With sync off the phase is never touched.
    SingleLayerRig unsynced;
    unsynced.rig.set_blend_space_1d(0, space);
    unsynced.rig.layer(0).sync_duration = false;
    unsynced.rig.set_blend_position(0, 0.5f);
    unsynced.step(0.5f);
    EXPECT_FLOAT_EQ(unsynced.rig.layer_phase(0), 0.0f);
}

// ---------------------------------------------------------------------------
// Root motion through the rig
// ---------------------------------------------------------------------------

TEST(AnimationRig, RootMotionIsExtractedFromLayerZeroAndStrippedFromThePose) {
    Skeleton s = make_skeleton(2);
    s.joints[0].local_translation = glm::vec3(0.0f, 10.0f, 0.0f);

    AnimationClip travel;
    travel.name = "travel";
    travel.duration = 1.0f;
    AnimationChannel ch;
    ch.bone_index = 0;
    ch.position_times = {0.0f, 1.0f};
    ch.positions = {glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(0.0f, 10.0f, 200.0f)};
    travel.channels.push_back(ch);
    const std::vector<AnimationClip> clips{travel};

    ClipBindings bindings;
    bindings.build(s, clips);

    AnimationRig rig;
    rig.bind(s, clips, bindings);
    LayerDesc base;
    base.name = "base";
    ASSERT_EQ(rig.add_layer(std::move(base)), 0);
    RootMotionConfig cfg;
    cfg.joint = 0;
    rig.set_root_motion(cfg);

    auto scratch = std::make_unique<PoseScratch>();
    float travelled = 0.0f;
    for (int i = 0; i < 10; ++i) {
        rig.evaluate(s, clips, 0.05f, *scratch);
        travelled += rig.root_motion().translation.z;
        // Stripped: the root's XZ stays at the rest value every single frame.
        EXPECT_FLOAT_EQ(rig.local_pose()[0].translation.z, 0.0f) << "frame " << i;
        EXPECT_FLOAT_EQ(rig.local_pose()[0].translation.x, 0.0f) << "frame " << i;
    }
    EXPECT_NEAR(travelled, 100.0f, 0.5f) << "half the clip at 0.05 s x 10 == 0.5 s of a 200-unit travel";

    // Reading root_motion() twice must not double-count: it is reset at the top
    // of every evaluate.
    const glm::vec3 once = rig.root_motion().translation;
    EXPECT_EQ(rig.root_motion().translation, once);
}

TEST(AnimationRig, RootMotionOffLeavesThePoseAlone) {
    Skeleton s = make_skeleton(2);
    AnimationClip travel;
    travel.name = "travel";
    travel.duration = 1.0f;
    AnimationChannel ch;
    ch.bone_index = 0;
    ch.position_times = {0.0f, 1.0f};
    ch.positions = {glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(0.0f, 10.0f, 200.0f)};
    travel.channels.push_back(ch);
    const std::vector<AnimationClip> clips{travel};

    ClipBindings bindings;
    bindings.build(s, clips);
    AnimationRig rig;
    rig.bind(s, clips, bindings);
    LayerDesc base;
    base.name = "base";
    ASSERT_EQ(rig.add_layer(std::move(base)), 0);

    auto scratch = std::make_unique<PoseScratch>();
    for (int i = 0; i < 10; ++i) {
        rig.evaluate(s, clips, 0.05f, *scratch);
    }
    EXPECT_TRUE(rig.root_motion().is_zero());
    EXPECT_GT(rig.local_pose()[0].translation.z, 50.0f) << "with root motion off the pose must still travel";
}

TEST(AnimationRig, StripFromPoseOffLeavesTheTravelInThePoseButStillReportsTheDelta) {
    Skeleton s = make_skeleton(2);
    AnimationClip travel;
    travel.name = "travel";
    travel.duration = 1.0f;
    AnimationChannel ch;
    ch.bone_index = 0;
    ch.position_times = {0.0f, 1.0f};
    ch.positions = {glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(0.0f, 10.0f, 200.0f)};
    travel.channels.push_back(ch);
    const std::vector<AnimationClip> clips{travel};

    ClipBindings bindings;
    bindings.build(s, clips);
    AnimationRig rig;
    rig.bind(s, clips, bindings);
    LayerDesc base;
    base.name = "base";
    ASSERT_EQ(rig.add_layer(std::move(base)), 0);
    RootMotionConfig cfg;
    cfg.joint = 0;
    cfg.strip_from_pose = false;
    rig.set_root_motion(cfg);

    auto scratch = std::make_unique<PoseScratch>();
    for (int i = 0; i < 10; ++i) {
        rig.evaluate(s, clips, 0.05f, *scratch);
    }
    EXPECT_GT(rig.local_pose()[0].translation.z, 50.0f);
    EXPECT_GT(rig.root_motion().translation.z, 0.0f) << "the delta is still reported — this is the "
                                                        "double-application configuration the docs warn about";
}
