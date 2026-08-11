// Evaluating many rigs across the job pool must be BITWISE identical to
// evaluating them one at a time.
//
// This is the test that catches a shared PoseScratch. Two rigs evaluated
// concurrently through one scratch corrupt each other silently — no assert can
// see it, and the symptom in a game is an occasional single-frame limb glitch
// that nobody can reproduce. Here it is a hard failure.
//
// Run it at BRACKEN_WORKERS=1 and unset, and against a -fsanitize=thread build
// per CLAUDE.md's concurrency rule.

#include "bracken/animation/animation_rig.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <vector>

using namespace bracken::animation;
using namespace bracken::core::jobs;

namespace {

constexpr std::size_t k_rigs = 128;

Skeleton make_skeleton() {
    Skeleton s;
    const int parents[] = {-1, 0, 1, 2, 2, 1, 5, 5};
    for (int i = 0; i < 8; ++i) {
        Joint j;
        j.name = "j" + std::to_string(i);
        j.parent_index = parents[i];
        j.local_translation = glm::vec3(0.0f, 8.0f + static_cast<float>(i), static_cast<float>(i));
        j.local_rotation = glm::angleAxis(0.05f * static_cast<float>(i), glm::normalize(glm::vec3(1, 1, 0)));
        j.local_scale = glm::vec3(1.0f);
        j.inverse_bind_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -8.0f, 0.0f));
        s.joints.push_back(j);
    }
    return s;
}

AnimationClip make_clip(const char* name, float duration, float amp, int first_bone) {
    AnimationClip c;
    c.name = name;
    c.duration = duration;
    for (int bone = first_bone; bone < first_bone + 3; ++bone) {
        AnimationChannel ch;
        ch.bone_index = bone;
        ch.rotation_times = {0.0f, duration * 0.5f, duration};
        ch.rotations = {glm::angleAxis(0.0f, glm::vec3(1, 0, 0)), glm::angleAxis(amp, glm::vec3(1, 0, 0)),
                        glm::angleAxis(0.0f, glm::vec3(1, 0, 0))};
        ch.position_times = {0.0f, duration};
        ch.positions = {glm::vec3(0.0f, 8.0f, 0.0f), glm::vec3(amp * 3.0f, 8.0f, amp * 9.0f)};
        c.channels.push_back(ch);
    }
    return c;
}

std::vector<AnimationClip> make_clips() {
    return {make_clip("walk", 1.2f, 0.4f, 1), make_clip("run", 0.8f, 0.9f, 1), make_clip("aim", 1.0f, 0.3f, 2)};
}

/// The three-layer stack the brief asks for: a 1D blend space on the base, a
/// masked additive layer on top, and root motion enabled.
void configure(AnimationRig& rig, const Skeleton& skeleton, const std::vector<AnimationClip>& clips,
               const ClipBindings& bindings, std::size_t seed) {
    rig.bind(skeleton, clips, bindings);

    LayerDesc base;
    base.name = "locomotion";
    base.cache_cursors = (seed % 2) == 0; // both dials exercised, same output demanded
    ASSERT_EQ(rig.add_layer(std::move(base)), 0);

    BlendSpace1D space;
    ASSERT_TRUE(space.add_sample(0, 0.0f));
    ASSERT_TRUE(space.add_sample(1, 1.0f));
    space.finalize();
    rig.set_blend_space_1d(0, space);
    rig.set_blend_position(0, static_cast<float>(seed % 11) / 10.0f);

    BoneMask upper;
    upper.set_subtree(skeleton, 2);
    LayerDesc additive;
    additive.name = "aim";
    additive.mode = BlendMode::Additive;
    additive.weight = 0.25f + 0.05f * static_cast<float>(seed % 7);
    additive.mask = upper;
    ASSERT_EQ(rig.add_layer(std::move(additive)), 1);
    rig.play_clip(1, 2, 0.0f);
    rig.set_layer_time(1, static_cast<float>(seed % 13) * 0.05f);

    LayerDesc idle_override;
    idle_override.name = "spare";
    idle_override.weight = 0.1f;
    ASSERT_EQ(rig.add_layer(std::move(idle_override)), 2);
    rig.play_clip(2, 1, 0.0f);

    RootMotionConfig cfg;
    cfg.joint = 0;
    rig.set_root_motion(cfg);

    rig.set_layer_time(0, static_cast<float>(seed) * 0.003f);
}

[[nodiscard]] bool exact(const glm::mat4& a, const glm::mat4& b) {
    return std::memcmp(&a, &b, sizeof(glm::mat4)) == 0;
}

class AnimationRigParallelTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!JobSystem::instance().is_initialized()) {
            JobSystem::instance().init();
        }
    }
};

} // namespace

TEST_F(AnimationRigParallelTest, ParallelEvaluationIsBitwiseIdenticalToSerial) {
    const Skeleton skeleton = make_skeleton();
    const std::vector<AnimationClip> clips = make_clips();
    ClipBindings bindings;
    bindings.build(skeleton, clips);

    auto serial = std::make_unique<std::vector<AnimationRig>>(k_rigs);
    auto parallel = std::make_unique<std::vector<AnimationRig>>(k_rigs);
    for (std::size_t i = 0; i < k_rigs; ++i) {
        configure((*serial)[i], skeleton, clips, bindings, i);
        configure((*parallel)[i], skeleton, clips, bindings, i);
    }

    constexpr float dt = 1.0f / 60.0f;
    constexpr int steps = 30;

    {
        auto scratch = std::make_unique<PoseScratch>();
        for (int s = 0; s < steps; ++s) {
            for (auto& rig : *serial) {
                rig.evaluate(skeleton, clips, dt, *scratch);
            }
        }
    }

    for (int s = 0; s < steps; ++s) {
        parallel_for(0, k_rigs, [&](std::size_t b, std::size_t e) {
            // ONE scratch per chunk — the whole point. Sharing a single scratch
            // across chunks is what this test exists to make impossible to ship.
            auto scratch = std::make_unique<PoseScratch>();
            for (std::size_t i = b; i < e; ++i) {
                (*parallel)[i].evaluate(skeleton, clips, dt, *scratch);
            }
        });
    }

    for (std::size_t i = 0; i < k_rigs; ++i) {
        const auto sm = (*serial)[i].bone_matrices();
        const auto pm = (*parallel)[i].bone_matrices();
        ASSERT_EQ((*serial)[i].used_joint_count(), (*parallel)[i].used_joint_count()) << "rig " << i;
        for (std::size_t bone = 0; bone < (*serial)[i].used_joint_count(); ++bone) {
            ASSERT_TRUE(exact(sm[bone], pm[bone])) << "rig " << i << " bone " << bone;
        }
        ASSERT_EQ((*serial)[i].root_motion().translation, (*parallel)[i].root_motion().translation)
            << "rig " << i;
    }
}

TEST_F(AnimationRigParallelTest, CursorCachingChangesNoFloat) {
    // cache_cursors is documented as a pure SPEED dial: the cursor fast path and
    // the binary search select the same keyframe interval. If that ever stops
    // being true, an eight-sample blend space silently animates differently from
    // a one-clip layer.
    const Skeleton skeleton = make_skeleton();
    const std::vector<AnimationClip> clips = make_clips();
    ClipBindings bindings;
    bindings.build(skeleton, clips);

    auto build = [&](bool cache) {
        auto rig = std::make_unique<AnimationRig>();
        rig->bind(skeleton, clips, bindings);
        LayerDesc d;
        d.name = "base";
        d.cache_cursors = cache;
        EXPECT_EQ(rig->add_layer(std::move(d)), 0);
        rig->play_clip(0, 1, 0.0f);
        return rig;
    };

    auto cached = build(true);
    auto uncached = build(false);
    auto scratch = std::make_unique<PoseScratch>();

    for (int s = 0; s < 90; ++s) {
        cached->evaluate(skeleton, clips, 1.0f / 60.0f, *scratch);
        uncached->evaluate(skeleton, clips, 1.0f / 60.0f, *scratch);
        for (std::size_t b = 0; b < cached->used_joint_count(); ++b) {
            ASSERT_TRUE(exact(cached->bone_matrices()[b], uncached->bone_matrices()[b]))
                << "step " << s << " bone " << b;
        }
    }
}
