// The playground character's rig, headless.
//
// WHY THIS FILE EXISTS
// --------------------
// `examples/playground` moved its villager from a bare `AnimationPlayer` to a
// two-layer `AnimationRig`, and the reason was behavioural rather than
// architectural: `Attack` used to be a whole-body Override STATE that every
// locomotion state fell into at priority 10, so swinging stopped the legs dead
// and the villager had to plant his feet to hit anything. It is now a masked,
// feathered, ADDITIVE upper-body LAYER.
//
// "The animation still reads correctly" is a claim about what a person sees on
// screen, and nothing in a test suite can check that. What CAN be checked is
// every property the port rests on, and that is what this file does. The
// skeleton, the two clip shapes and the layer configuration below are
// deliberately the same shapes `examples/playground/character.cpp` and
// `PlaygroundGame::setup_animation_rig` build — the playground is not linked
// into the test binary, so they are mirrored here rather than shared. If the
// playground's rig setup changes, change this fixture with it; a divergence
// makes these tests pass while the game is broken.
//
// The four properties, in the order the port depends on them:
//
//   1. THE LEGS KEEP WALKING. Every joint outside the Spine subtree is bitwise
//      untouched by the upper-body layer, at any weight. This is the entire
//      point of the change.
//   2. A MASKED JOINT WITH NO CHANNEL DOES NOT SNAP. The Attack clip animates
//      the right arm and the spine only. Under an Override layer the left arm
//      would jump to the REST pose the instant the layer faded in, because the
//      sampler's fallback is the rest pose. Under an Additive layer it
//      contributes a zero delta and keeps swinging. This is why the layer is
//      Additive, and it is the property most likely to be broken by someone
//      "simplifying" the mode.
//   3. THE FEATHER IS REAL AND IS A HALF AT THE SPINE. A binary mask pops the
//      chest; the playground feathers by hierarchy depth from the spine's
//      PARENT, which lands the spine itself at 0.5.
//   4. THE LAYER FADES IN AND OUT OF NOTHING. Both keyed channels start and end
//      at the identity rotation, so weight 1 at clip time 0 is the base pose.

#include "bracken/animation/animation_rig.hpp"
#include "bracken/animation/animation_state_machine.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <span>
#include <string>
#include <vector>

using namespace bracken::animation;

namespace {

// Joint indices, fixed by construction order exactly as in
// examples/playground/character.cpp.
enum : int {
    J_HIPS = 0,
    J_SPINE,
    J_HEAD,
    J_LEFT_UPPER_LEG,
    J_LEFT_LOWER_LEG,
    J_LEFT_FOOT,
    J_RIGHT_UPPER_LEG,
    J_RIGHT_LOWER_LEG,
    J_RIGHT_FOOT,
    J_LEFT_ARM,
    J_RIGHT_ARM,
    JOINT_COUNT
};

constexpr float k_two_pi = 6.28318530718f;
constexpr float k_hips_y = 95.0f;
constexpr float k_dt = 1.0f / 60.0f;

glm::quat rot_x(float a) { return glm::angleAxis(a, glm::vec3(1, 0, 0)); }
glm::quat rot_y(float a) { return glm::angleAxis(a, glm::vec3(0, 1, 0)); }

void add_joint(Skeleton& skel, std::vector<glm::vec3>& global_pos, const char* name, int parent,
                const glm::vec3& local_t) {
    Joint j;
    j.name = name;
    j.parent_index = parent;
    j.local_translation = local_t;
    j.local_rotation = glm::quat(1, 0, 0, 0); // identity rest — see the additive note below
    j.local_scale = glm::vec3(1.0f);
    const glm::vec3 global = parent >= 0 ? global_pos[static_cast<std::size_t>(parent)] + local_t : local_t;
    j.inverse_bind_matrix = glm::translate(glm::mat4(1.0f), -global);
    skel.joints.push_back(std::move(j));
    global_pos.push_back(global);
}

/// The villager. Parents strictly before children; every rest rotation is
/// identity, which is what makes an additive delta equal the authored key.
Skeleton make_villager() {
    Skeleton s;
    std::vector<glm::vec3> global;
    add_joint(s, global, "Hips", -1, {0, k_hips_y, 0});
    add_joint(s, global, "Spine", J_HIPS, {0, 35, 0});
    add_joint(s, global, "Head", J_SPINE, {0, 32, 0});
    add_joint(s, global, "LeftUpperLeg", J_HIPS, {-12, -5, 0});
    add_joint(s, global, "LeftLowerLeg", J_LEFT_UPPER_LEG, {0, -45, 0});
    add_joint(s, global, "LeftFoot", J_LEFT_LOWER_LEG, {0, -40, 0});
    add_joint(s, global, "RightUpperLeg", J_HIPS, {12, -5, 0});
    add_joint(s, global, "RightLowerLeg", J_RIGHT_UPPER_LEG, {0, -45, 0});
    add_joint(s, global, "RightFoot", J_RIGHT_LOWER_LEG, {0, -40, 0});
    add_joint(s, global, "LeftArm", J_SPINE, {-24, 22, 0});
    add_joint(s, global, "RightArm", J_SPINE, {24, 22, 0});
    return s;
}

AnimationChannel& new_channel(AnimationClip& clip, int bone) {
    clip.channels.emplace_back();
    clip.channels.back().bone_index = bone;
    return clip.channels.back();
}

/// A locomotion cycle in the playground's shape: legs and arms swinging in
/// opposite phase, a spine pitch, and a hips bob that is PURE VERTICAL (the
/// reason the playground has no root motion to extract).
AnimationClip make_locomotion_clip(const char* name, float duration, float leg_swing, float arm_swing) {
    AnimationClip clip;
    clip.name = name;
    clip.duration = duration;
    constexpr int keys = 16;

    const auto sample_rot = [&](int bone, auto f) {
        auto& ch = new_channel(clip, bone);
        for (int i = 0; i <= keys; ++i) {
            const float t = duration * static_cast<float>(i) / static_cast<float>(keys);
            const float p = (i == keys) ? 0.0f : static_cast<float>(i) / static_cast<float>(keys);
            ch.rotation_times.push_back(t);
            ch.rotations.push_back(f(p));
        }
    };

    sample_rot(J_LEFT_UPPER_LEG, [&](float p) { return rot_x(leg_swing * std::sin(k_two_pi * p)); });
    sample_rot(J_RIGHT_UPPER_LEG,
               [&](float p) { return rot_x(leg_swing * std::sin(k_two_pi * p + glm::pi<float>())); });
    sample_rot(J_LEFT_LOWER_LEG, [&](float p) { return rot_x(0.5f * std::max(0.0f, std::sin(k_two_pi * p))); });
    sample_rot(J_RIGHT_LOWER_LEG,
               [&](float p) { return rot_x(0.5f * std::max(0.0f, -std::sin(k_two_pi * p))); });
    sample_rot(J_LEFT_ARM, [&](float p) { return rot_x(arm_swing * std::sin(k_two_pi * p + glm::pi<float>())); });
    sample_rot(J_RIGHT_ARM, [&](float p) { return rot_x(arm_swing * std::sin(k_two_pi * p)); });
    sample_rot(J_SPINE, [&](float p) { return rot_x(0.05f) * rot_y(0.08f * std::sin(k_two_pi * 2.0f * p)); });

    auto& hips = new_channel(clip, J_HIPS);
    for (int i = 0; i <= keys; ++i) {
        const float t = duration * static_cast<float>(i) / static_cast<float>(keys);
        const float p = (i == keys) ? 0.0f : static_cast<float>(i) / static_cast<float>(keys);
        hips.position_times.push_back(t);
        hips.positions.push_back(glm::vec3(0.0f, k_hips_y + 1.5f * (1.0f - std::cos(k_two_pi * 2.0f * p)), 0.0f));
    }
    return clip;
}

/// The playground's Attack clip, key for key: the right arm and the spine only,
/// both starting and ending at identity.
AnimationClip make_attack_clip() {
    AnimationClip clip;
    clip.name = "Attack";
    clip.duration = 0.45f;

    auto& arm = new_channel(clip, J_RIGHT_ARM);
    arm.rotation_times = {0.0f, 0.12f, 0.24f, 0.45f};
    arm.rotations = {rot_x(0.0f), rot_x(-2.2f), rot_x(1.1f), rot_x(0.0f)};

    auto& spine = new_channel(clip, J_SPINE);
    spine.rotation_times = {0.0f, 0.12f, 0.24f, 0.45f};
    spine.rotations = {rot_y(0.0f), rot_y(-0.35f), rot_y(0.30f), rot_y(0.0f)};
    return clip;
}

enum : int { CLIP_IDLE = 0, CLIP_WALK = 1, CLIP_RUN = 2, CLIP_ATTACK = 3 };

std::vector<AnimationClip> make_clips() {
    std::vector<AnimationClip> clips;
    clips.push_back(make_locomotion_clip("Idle", 3.0f, 0.02f, 0.06f));
    clips.push_back(make_locomotion_clip("Walk", 0.9f, 0.55f, 0.5f));
    clips.push_back(make_locomotion_clip("Run", 0.55f, 0.85f, 0.9f));
    clips.push_back(make_attack_clip());
    return clips;
}

/// The playground's rig, reproduced: layer 0 locomotion, layer 1 a feathered
/// additive upper body masked to the Spine subtree.
struct VillagerRig {
    Skeleton skeleton = make_villager();
    std::vector<AnimationClip> clips = make_clips();
    ClipBindings bindings;
    AnimationRig rig;
    PoseScratch scratch;
    /// MEMBER, not a local: LayerDesc stores the SPAN, so a local would dangle.
    std::array<float, MAX_BONES> feather{};
    int upper = -1;

    /// `with_upper == false` builds the single-layer reference rig the two-layer
    /// one is diffed against. `feathered == false` drops the joint-weight ramp,
    /// which is how the feather's effect is isolated.
    explicit VillagerRig(bool with_upper = true, bool feathered = true) {
        bindings.build(skeleton, clips);
        rig.bind(skeleton, clips, bindings);

        LayerDesc base;
        base.name = "locomotion";
        EXPECT_EQ(rig.add_layer(std::move(base)), 0);

        if (!with_upper) {
            return;
        }

        BoneMask mask;
        mask.set_subtree(skeleton, J_SPINE);

        // Rooted at the spine's PARENT (the hips) so the spine itself lands
        // mid-ramp rather than at zero. falloff 2: hips 0, spine 0.5,
        // head/arms 1.
        feather_by_depth(skeleton, J_HIPS, 2, std::span<float>(feather.data(), feather.size()));

        LayerDesc upper_desc;
        upper_desc.name = "upper_body";
        upper_desc.mode = BlendMode::Additive;
        upper_desc.additive_reference_clip = -1; // rest pose
        upper_desc.weight = 0.0f;
        upper_desc.mask = mask;
        if (feathered) {
            upper_desc.joint_weights = std::span<const float>(feather.data(), feather.size());
        }
        upper = rig.add_layer(std::move(upper_desc));
        EXPECT_EQ(upper, 1);
        rig.play_clip(upper, CLIP_ATTACK, 0.0f);
        rig.set_layer_loop(upper, false);
    }

    void step(float dt) { rig.evaluate(skeleton, clips, dt, scratch); }

    /// Park the upper-body layer at a fixed clip time and hold it there.
    ///
    /// The single step is NOT optional and the trap is worth naming: the
    /// constructor's `play_clip` starts a crossfade with duration 0, and a
    /// crossfade only resolves inside `advance_clock` — which returns early for
    /// a layer that is not playing. Freeze first and the layer stays wedged at
    /// blend factor 0, sampling its PREVIOUS clip forever. One live step
    /// resolves it (blend duration 0 snaps the factor to 1), and only then is
    /// it safe to stop the clock.
    void freeze_attack_at(float time) {
        step(k_dt);
        rig.set_layer_playing(upper, false);
        rig.set_layer_time(upper, time);
    }
};

[[nodiscard]] bool exact(const BoneTransform& a, const BoneTransform& b) {
    return std::memcmp(&a, &b, sizeof(BoneTransform)) == 0;
}

/// Angle between two rotations, in radians, sign-insensitive (q and -q are the
/// same rotation).
[[nodiscard]] float angle_between(const glm::quat& a, const glm::quat& b) {
    // Sign-align first: q and -q are the same rotation, so the short way round
    // is the only meaningful one.
    const glm::quat c = (glm::dot(a, b) < 0.0f) ? -b : b;
    // 2*atan2(|a - c|, |a + c|), NOT 2*acos(|dot|). acos has infinite slope at
    // 1, so it loses half its significant digits exactly where this test looks:
    // at ONE ULP of separation the acos form returns 9.8e-4 rad, a resolution
    // floor two orders of magnitude above the 1e-5 these tests want to assert.
    // The atan2 form is linear in the separation and bottoms out at zero.
    // The factor is 4, not 2: atan2(|a - c|, |a + c|) is HALF the 4-D angle
    // between the quaternions, and the rotation angle is twice that again.
    return 4.0f * std::atan2(glm::length(a - c), glm::length(a + c));
}

} // namespace

// ---------------------------------------------------------------------------
// 1. The legs keep walking
// ---------------------------------------------------------------------------

TEST(VillagerRig, TheUpperBodyLayerLeavesEveryLegJointBITWISEUntouched) {
    VillagerRig with_attack;
    VillagerRig without_attack(/*with_upper=*/false);

    // Both rigs run the same locomotion clip with the same clock, so layer 0 is
    // identical by construction; anything that differs came from layer 1.
    with_attack.rig.play_clip(0, CLIP_WALK, 0.0f);
    without_attack.rig.play_clip(0, CLIP_WALK, 0.0f);
    with_attack.rig.set_layer_weight(with_attack.upper, 1.0f);

    constexpr int legs[] = {J_HIPS, J_LEFT_UPPER_LEG, J_LEFT_LOWER_LEG, J_LEFT_FOOT,
                            J_RIGHT_UPPER_LEG, J_RIGHT_LOWER_LEG, J_RIGHT_FOOT};

    for (int step = 0; step < 120; ++step) {
        with_attack.step(k_dt);
        without_attack.step(k_dt);
        for (const int j : legs) {
            ASSERT_TRUE(exact(with_attack.rig.local_pose()[static_cast<std::size_t>(j)],
                              without_attack.rig.local_pose()[static_cast<std::size_t>(j)]))
                << "leg joint " << j << " moved at step " << step
                << " — the upper-body mask is leaking below the waist";
        }
    }
}

TEST(VillagerRig, TheMaskCoversExactlyTheSpineSubtree) {
    // Pins the shape the port assumes: spine, head and BOTH arms move; hips and
    // both legs do not. If the skeleton ever gains a joint, this is the test
    // that says whether it landed on the right side of the waist.
    const Skeleton skeleton = make_villager();
    BoneMask mask;
    mask.set_subtree(skeleton, J_SPINE);

    EXPECT_TRUE(mask.test(J_SPINE));
    EXPECT_TRUE(mask.test(J_HEAD));
    EXPECT_TRUE(mask.test(J_LEFT_ARM));
    EXPECT_TRUE(mask.test(J_RIGHT_ARM));
    EXPECT_FALSE(mask.test(J_HIPS));
    EXPECT_FALSE(mask.test(J_LEFT_UPPER_LEG));
    EXPECT_FALSE(mask.test(J_LEFT_LOWER_LEG));
    EXPECT_FALSE(mask.test(J_LEFT_FOOT));
    EXPECT_FALSE(mask.test(J_RIGHT_UPPER_LEG));
    EXPECT_FALSE(mask.test(J_RIGHT_LOWER_LEG));
    EXPECT_FALSE(mask.test(J_RIGHT_FOOT));
    EXPECT_EQ(mask.count(), 4);
}

// ---------------------------------------------------------------------------
// 2. A masked joint with no channel does not snap — why the layer is ADDITIVE
// ---------------------------------------------------------------------------

TEST(VillagerRig, TheLeftArmKeepsSwingingThroughASwingWhileTheRightArmDoesNot) {
    VillagerRig with_attack;
    VillagerRig without_attack(/*with_upper=*/false);
    with_attack.rig.play_clip(0, CLIP_WALK, 0.0f);
    without_attack.rig.play_clip(0, CLIP_WALK, 0.0f);
    with_attack.rig.set_layer_weight(with_attack.upper, 1.0f);

    // Let the swing play once through at full weight. The authored right-arm key
    // peaks at -2.2 rad at t = 0.12 s, which the 60 Hz clock straddles.
    float peak_right = 0.0f;
    float worst_left = 0.0f;
    for (int step = 0; step < 60; ++step) {
        with_attack.step(k_dt);
        without_attack.step(k_dt);

        const auto& a = with_attack.rig.local_pose();
        const auto& b = without_attack.rig.local_pose();
        peak_right = std::max(peak_right, angle_between(a[J_RIGHT_ARM].rotation, b[J_RIGHT_ARM].rotation));
        worst_left = std::max(worst_left, angle_between(a[J_LEFT_ARM].rotation, b[J_LEFT_ARM].rotation));
    }

    // The right arm is driven. The bound is 1.8 rather than 2.2 because the
    // sampling grid never lands exactly on the key; anything near zero means the
    // layer is not reaching the arm at all, which is what this actually tests.
    EXPECT_GT(peak_right, 1.8f) << "the attack layer is not driving the right arm";
    // The left arm has NO channel in the Attack clip. Additive turns that into a
    // zero delta; Override would slam it to the rest pose, which on this
    // skeleton is a 0.5 rad error at the peak of the walk's arm swing. The
    // tolerance is float noise from inverse(q) * q, not a fudge factor.
    EXPECT_LT(worst_left, 1e-4f)
        << "the left arm moved — is the upper-body layer Override instead of Additive?";
}

TEST(VillagerRig, AnOverrideUpperBodyLayerWouldHaveFrozenTheLeftArm) {
    // The negative control for the test above. This is not a wish: it is what
    // the playground actually looked like before the port, and the number below
    // is the size of the artifact that motivated the change. If this test ever
    // starts failing because the error got small, the Additive-vs-Override
    // decision has stopped mattering and the comment in character.cpp is stale.
    VillagerRig with_attack;
    VillagerRig without_attack(/*with_upper=*/false);
    with_attack.rig.layer(with_attack.upper).mode = BlendMode::Override;
    with_attack.rig.play_clip(0, CLIP_WALK, 0.0f);
    without_attack.rig.play_clip(0, CLIP_WALK, 0.0f);
    with_attack.rig.set_layer_weight(with_attack.upper, 1.0f);

    float worst_left = 0.0f;
    for (int step = 0; step < 60; ++step) {
        with_attack.step(k_dt);
        without_attack.step(k_dt);
        worst_left = std::max(worst_left, angle_between(with_attack.rig.local_pose()[J_LEFT_ARM].rotation,
                                                        without_attack.rig.local_pose()[J_LEFT_ARM].rotation));
    }
    // The Walk clip swings the left arm 0.5 rad either way, and Override would
    // pin it at the rest pose (identity) for as long as the layer is up.
    EXPECT_GT(worst_left, 0.2f) << "Override no longer snaps an unchanneled joint to the rest pose";
}

// ---------------------------------------------------------------------------
// 3. The feather
// ---------------------------------------------------------------------------

TEST(VillagerRig, TheDepthFeatherPutsTheSpineAtHalfWeightAndTheArmsAtFull) {
    const Skeleton skeleton = make_villager();
    std::array<float, MAX_BONES> w{};
    feather_by_depth(skeleton, J_HIPS, 2, std::span<float>(w.data(), w.size()));

    EXPECT_FLOAT_EQ(w[J_HIPS], 0.0f);   // depth 0 — the ramp's foot
    EXPECT_FLOAT_EQ(w[J_SPINE], 0.5f);  // depth 1
    EXPECT_FLOAT_EQ(w[J_HEAD], 1.0f);   // depth 2, clamped at the top
    EXPECT_FLOAT_EQ(w[J_LEFT_ARM], 1.0f);
    EXPECT_FLOAT_EQ(w[J_RIGHT_ARM], 1.0f);
    // Rooting the ramp at the SPINE instead would put the spine at 0 — the trap
    // the playground comment names. Pinned so nobody "fixes" the root back.
    std::array<float, MAX_BONES> from_spine{};
    feather_by_depth(skeleton, J_SPINE, 2, std::span<float>(from_spine.data(), from_spine.size()));
    EXPECT_FLOAT_EQ(from_spine[J_SPINE], 0.0f);
}

TEST(VillagerRig, TheFeatherHalvesTheSpineDeltaAndLeavesTheArmsAtFullStrength) {
    VillagerRig feathered(/*with_upper=*/true, /*feathered=*/true);
    VillagerRig binary(/*with_upper=*/true, /*feathered=*/false);
    VillagerRig base(/*with_upper=*/false);
    for (VillagerRig* r : {&feathered, &binary, &base}) {
        r->rig.play_clip(0, CLIP_WALK, 0.0f);
    }
    feathered.rig.set_layer_weight(feathered.upper, 1.0f);
    binary.rig.set_layer_weight(binary.upper, 1.0f);

    // Park both attack clocks on the key where the spine twist is largest, so
    // the two rigs differ ONLY in their joint weights. `base` takes the same
    // number of steps so all three layer-0 clocks stay in lockstep.
    feathered.freeze_attack_at(0.12f);
    binary.freeze_attack_at(0.12f);
    base.step(k_dt);
    for (int step = 0; step < 20; ++step) {
        feathered.step(k_dt);
        binary.step(k_dt);
        base.step(k_dt);
    }

    const auto& f = feathered.rig.local_pose();
    const auto& b = binary.rig.local_pose();
    const auto& n = base.rig.local_pose();

    // slerp's angle is linear in t, so a joint weight of exactly 0.5 halves the
    // angular displacement from the base pose. The window is wide enough for
    // float error and narrow enough to catch an off-by-one in the depth ramp.
    const float full = angle_between(b[J_SPINE].rotation, n[J_SPINE].rotation);
    const float half = angle_between(f[J_SPINE].rotation, n[J_SPINE].rotation);
    ASSERT_GT(full, 0.05f) << "the attack clip is not moving the spine at all";
    EXPECT_NEAR(half / full, 0.5f, 0.02f);

    // The arms sit at feather weight 1, so feathering must not have touched them.
    EXPECT_NEAR(angle_between(f[J_RIGHT_ARM].rotation, b[J_RIGHT_ARM].rotation), 0.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
// 4. The layer fades in and out of nothing
// ---------------------------------------------------------------------------

TEST(VillagerRig, AtClipStartAndEndTheAttackLayerIsTheIdentityEvenAtFullWeight) {
    // character.cpp's authoring contract: both keyed channels start and end at
    // the identity rotation, so a swing can be faded in at any moment without a
    // snap. If someone adds a channel holding a non-identity value at t == 0,
    // this is what catches it.
    VillagerRig with_attack;
    VillagerRig without_attack(/*with_upper=*/false);
    with_attack.rig.play_clip(0, CLIP_WALK, 0.0f);
    without_attack.rig.play_clip(0, CLIP_WALK, 0.0f);
    with_attack.rig.set_layer_weight(with_attack.upper, 1.0f);
    with_attack.freeze_attack_at(0.0f);
    without_attack.step(k_dt);

    for (const float t : {0.0f, 0.45f}) {
        with_attack.rig.set_layer_time(with_attack.upper, t);
        for (int step = 0; step < 10; ++step) {
            with_attack.step(k_dt);
            without_attack.step(k_dt);
        }
        for (std::size_t j = 0; j < JOINT_COUNT; ++j) {
            EXPECT_LT(angle_between(with_attack.rig.local_pose()[j].rotation,
                                    without_attack.rig.local_pose()[j].rotation),
                      1e-5f)
                << "joint " << j << " differs at attack clip time " << t;
            EXPECT_LT(glm::length(with_attack.rig.local_pose()[j].translation -
                                  without_attack.rig.local_pose()[j].translation),
                      1e-4f)
                << "joint " << j << " translated at attack clip time " << t;
        }
    }
}

TEST(VillagerRig, RampingTheLayerWeightMovesTheArmContinuously) {
    // The playground follows the weight rather than snapping it, because a
    // binary 0/1 on a masked layer pops. This checks the engine side of that:
    // with both clocks frozen, a linear weight ramp must produce a monotone,
    // step-bounded path — no single frame may account for a disproportionate
    // share of the travel.
    VillagerRig r;
    r.rig.play_clip(0, CLIP_WALK, 0.0f);
    r.freeze_attack_at(0.12f);
    r.rig.set_layer_playing(0, false); // freeze locomotion too, so only w moves

    constexpr int steps = 32;
    glm::vec3 prev(0.0f);
    glm::vec3 first(0.0f);
    float max_step = 0.0f;
    float total = 0.0f;
    for (int i = 0; i <= steps; ++i) {
        r.rig.set_layer_weight(r.upper, static_cast<float>(i) / static_cast<float>(steps));
        r.step(0.0f);
        const glm::vec3 p = glm::vec3(r.rig.world_transforms()[J_RIGHT_ARM][3]);
        if (i == 0) {
            first = p;
        } else {
            const float d = glm::length(p - prev);
            max_step = std::max(max_step, d);
            total += d;
        }
        prev = p;
    }

    ASSERT_GT(total, 1.0f) << "the ramp did not move the arm at all";
    // Perfectly uniform would be total/steps. Allow 2x for slerp's curvature;
    // a POP would be a single step carrying most of the travel.
    EXPECT_LT(max_step, 2.0f * total / static_cast<float>(steps));
    // ...and the ramp must actually end somewhere different from where it began.
    EXPECT_GT(glm::length(prev - first), 1.0f);
}

TEST(VillagerRig, WeightZeroReproducesTheSingleLayerRigBITWISE) {
    // The steady state of a villager who is not swinging, which is most frames.
    // It must cost nothing and change nothing — and "nothing" here is bitwise,
    // because blend_additive leaves a zero-weight joint untouched rather than
    // running an identity blend through it.
    VillagerRig with_attack;
    VillagerRig without_attack(/*with_upper=*/false);
    with_attack.rig.play_clip(0, CLIP_WALK, 0.0f);
    without_attack.rig.play_clip(0, CLIP_WALK, 0.0f);
    with_attack.rig.set_layer_weight(with_attack.upper, 0.0f);

    for (int step = 0; step < 60; ++step) {
        with_attack.step(k_dt);
        without_attack.step(k_dt);
        for (std::size_t j = 0; j < JOINT_COUNT; ++j) {
            ASSERT_TRUE(exact(with_attack.rig.local_pose()[j], without_attack.rig.local_pose()[j]))
                << "joint " << j << " at step " << step;
        }
    }
}

// ---------------------------------------------------------------------------
// The state machine now describes locomotion only
// ---------------------------------------------------------------------------

TEST(VillagerRig, TheStateMachineDrivesLayerZeroAndIgnoresTheUpperBody) {
    // The port deleted the Attack STATE. What is left must still walk and run
    // off `speed` alone, and — the part that used to be impossible — it must do
    // so while the upper-body layer is at full weight.
    VillagerRig r;
    r.rig.set_layer_weight(r.upper, 1.0f);

    AnimationStateMachine sm;
    using TC = TransitionCondition;
    const auto cond = [](const char* p, TC::Op op, float v) {
        TC c;
        c.param_name = p;
        c.op = op;
        c.threshold = v;
        return c;
    };
    const auto to = [](const char* target, TC c, float xfade) {
        StateTransition t;
        t.target_state = target;
        t.conditions.push_back(std::move(c));
        t.crossfade_duration = xfade;
        return t;
    };

    AnimState idle;
    idle.name = "Idle";
    idle.clip_name = "Idle";
    idle.transitions.push_back(to("Walk", cond("speed", TC::Op::GT, 20.0f), 0.15f));
    AnimState walk;
    walk.name = "Walk";
    walk.clip_name = "Walk";
    walk.transitions.push_back(to("Run", cond("speed", TC::Op::GT, 240.0f), 0.15f));
    walk.transitions.push_back(to("Idle", cond("speed", TC::Op::LT, 20.0f), 0.2f));
    AnimState run;
    run.name = "Run";
    run.clip_name = "Run";
    run.transitions.push_back(to("Walk", cond("speed", TC::Op::LT, 220.0f), 0.2f));

    sm.add_state(std::move(idle));
    sm.add_state(std::move(walk));
    sm.add_state(std::move(run));
    sm.set_default_state("Idle");
    ASSERT_TRUE(sm.bind_clips(r.clips));
    const int speed = sm.param_index("speed");

    const auto run_for = [&](float value, int frames) {
        for (int i = 0; i < frames; ++i) {
            sm.set_float(speed, value);
            sm.update(r.rig, 0);
            r.step(k_dt);
        }
    };

    run_for(0.0f, 10);
    EXPECT_EQ(sm.current_state(), "Idle");
    EXPECT_EQ(r.rig.layer_clip(0), CLIP_IDLE);

    run_for(120.0f, 30);
    EXPECT_EQ(sm.current_state(), "Walk");
    EXPECT_EQ(r.rig.layer_clip(0), CLIP_WALK);

    run_for(400.0f, 30);
    EXPECT_EQ(sm.current_state(), "Run");
    EXPECT_EQ(r.rig.layer_clip(0), CLIP_RUN);

    // The upper-body layer never became the base, which is the failure mode of
    // driving the state machine at the wrong layer index.
    EXPECT_EQ(r.rig.layer_clip(r.upper), CLIP_ATTACK);

    run_for(0.0f, 40);
    EXPECT_EQ(sm.current_state(), "Idle");
}

// ---------------------------------------------------------------------------
// Root motion stays OFF here, and that is a decision rather than an omission
// ---------------------------------------------------------------------------

TEST(VillagerRig, TheHipsBobIsNotRootMotion) {
    // The villager's only hips channel is a pure vertical bob, and his XZ travel
    // comes from the capsule controller. Enabling root motion with the default
    // config must therefore report a ZERO delta — if it ever reports otherwise,
    // a clip gained horizontal hips translation and the playground would start
    // sliding at double speed.
    VillagerRig r(/*with_upper=*/false);
    RootMotionConfig cfg;
    cfg.joint = J_HIPS;
    cfg.strip_from_pose = false; // measure extraction alone
    r.rig.set_root_motion(cfg);
    r.rig.play_clip(0, CLIP_WALK, 0.0f);

    float worst = 0.0f;
    for (int step = 0; step < 120; ++step) {
        r.step(k_dt);
        worst = std::max(worst, glm::length(r.rig.root_motion().translation));
    }
    EXPECT_LT(worst, 1e-5f) << "a playground clip has gained horizontal root translation";
}
