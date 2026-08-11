// AnimationRig: THE ALLOCATION AUDIT.
//
// `AnimationRig::evaluate` runs once per animated character per frame, inside
// the sim stage, on a pool worker. A single heap allocation in there is not a
// micro-optimisation problem: it is a lock in the middle of a fan-out, taken by
// every character on every worker at the same point in the frame. The header
// promises "after bind(), evaluate() allocates nothing", and a promise in a
// comment is not a promise. This file is the proof.
//
// HOW IT WORKS, AND THE ONE THING TO KNOW BEFORE EDITING
// -----------------------------------------------------
// This TU REPLACES the global `operator new` / `operator delete` for the entire
// `bracken_tests` binary. That is the only way to see a `std::vector` growth
// from outside the vector — there is no allocator hook on the default
// allocator. Consequences a future editor must know:
//
//   * There must be exactly ONE such replacement in the whole test binary. If a
//     second test file ever adds its own, the link fails with a duplicate
//     symbol. Count allocations from here, or refactor this into a shared
//     helper — do not add a second copy.
//   * The counter is `thread_local`, so a background thread allocating during a
//     measured window cannot make a test flake. It also means a measurement
//     only sees the thread it runs on, which is exactly what we want: the
//     question is "does evaluate() allocate", not "did anything anywhere".
//   * Everything routes through `std::malloc` / `std::free`, so a sanitizer
//     build still tracks the memory. What ASan loses is its new/delete
//     type-mismatch check, which nothing here depends on.
//
// `TheCounterActuallySeesAllocations` is not filler. Without it, a counter that
// silently stopped counting would make every other test in this file pass
// vacuously, which is the classic way an allocation audit rots into decoration.

#include "bracken/animation/animation_rig.hpp"
#include "bracken/animation/animation_state_machine.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <glm/gtc/matrix_transform.hpp>
#include <new>
#include <span>
#include <string>
#include <vector>

namespace {

/// Allocations seen on THIS thread while `g_counting` is set. Not atomic on
/// purpose: it is thread-local, so there is no cross-thread protocol to
/// document and no contention to pay for.
thread_local std::size_t g_alloc_count = 0;
thread_local bool g_counting = false;

/// Counts every allocation inside its scope. Nested use is not supported and
/// not needed — measure one window at a time.
struct AllocScope {
    AllocScope() {
        g_alloc_count = 0;
        g_counting = true;
    }
    ~AllocScope() { g_counting = false; }
    AllocScope(const AllocScope&) = delete;
    AllocScope& operator=(const AllocScope&) = delete;

    /// Stop counting and report. Called explicitly before any ASSERT_*, because
    /// a gtest failure message allocates and would otherwise be counted.
    [[nodiscard]] std::size_t stop() {
        g_counting = false;
        return g_alloc_count;
    }
};

[[nodiscard]] void* counted_malloc(std::size_t bytes) {
    if (g_counting) {
        ++g_alloc_count;
    }
    // malloc(0) is allowed to return null, which `operator new` may not.
    void* p = std::malloc(bytes != 0 ? bytes : 1);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
}

[[nodiscard]] void* counted_aligned_malloc(std::size_t bytes, std::size_t alignment) {
    if (g_counting) {
        ++g_alloc_count;
    }
    // std::aligned_alloc wants a size that is a multiple of the alignment; the
    // caller's size need not be one.
    const std::size_t rounded = ((bytes != 0 ? bytes : 1) + alignment - 1) / alignment * alignment;
    void* p = std::aligned_alloc(alignment, rounded);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
}

} // namespace

// --- the replacements ------------------------------------------------------
// All four `new` forms and their matching `delete`s, so no allocation shape can
// slip past uncounted.

void* operator new(std::size_t bytes) { return counted_malloc(bytes); }
void* operator new[](std::size_t bytes) { return counted_malloc(bytes); }
void* operator new(std::size_t bytes, std::align_val_t a) {
    return counted_aligned_malloc(bytes, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t bytes, std::align_val_t a) {
    return counted_aligned_malloc(bytes, static_cast<std::size_t>(a));
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

// ---------------------------------------------------------------------------

using namespace bracken::animation;

namespace {

/// A full-size rig: MAX_BONES joints, which is the worst case the constant
/// allows and the size at which a per-joint allocation would be most obvious.
Skeleton make_full_skeleton() {
    Skeleton s;
    for (int i = 0; i < MAX_BONES; ++i) {
        Joint j;
        j.name = "j" + std::to_string(i);
        j.parent_index = (i == 0) ? -1 : (i - 1) / 2; // balanced tree, parents first
        j.local_translation = glm::vec3(0.0f, 1.0f, 0.0f);
        j.local_rotation = glm::angleAxis(0.01f * static_cast<float>(i), glm::vec3(0, 1, 0));
        j.local_scale = glm::vec3(1.0f);
        j.inverse_bind_matrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        s.joints.push_back(j);
    }
    return s;
}

AnimationClip make_clip(const char* name, float duration, float amplitude) {
    AnimationClip c;
    c.name = name;
    c.duration = duration;
    for (int bone = 0; bone < MAX_BONES; bone += 2) {
        AnimationChannel ch;
        ch.bone_index = bone;
        ch.rotation_times = {0.0f, duration * 0.5f, duration};
        ch.rotations = {glm::angleAxis(0.0f, glm::vec3(1, 0, 0)), glm::angleAxis(amplitude, glm::vec3(1, 0, 0)),
                        glm::angleAxis(0.0f, glm::vec3(1, 0, 0))};
        ch.position_times = {0.0f, duration};
        ch.positions = {glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(amplitude, 1.0f, 0.0f)};
        c.channels.push_back(ch);
    }
    return c;
}

std::vector<AnimationClip> make_clips() {
    return {make_clip("Idle", 1.4f, 0.2f), make_clip("Walk", 0.9f, 0.6f), make_clip("Run", 0.7f, 0.9f),
            make_clip("Aim", 1.1f, 0.3f)};
}

constexpr float k_dt = 1.0f / 60.0f;

} // namespace

// ---------------------------------------------------------------------------
// The instrument, checked before it is trusted
// ---------------------------------------------------------------------------

TEST(AnimationRigAlloc, TheCounterActuallySeesAllocations) {
    // A vector growth is one allocation at minimum. If this reads 0 the whole
    // file is meaningless and every test below is a false green.
    std::size_t seen = 0;
    {
        AllocScope scope;
        std::vector<int> v;
        v.reserve(1024);
        seen = scope.stop();
        // Keep `v` observable so nothing elides the reserve.
        EXPECT_GE(v.capacity(), 1024u);
    }
    EXPECT_GE(seen, 1u) << "the counting operator new is not being linked in";
}

TEST(AnimationRigAlloc, SetupIsWhereTheAllocationsLive) {
    // The mirror image of the audit: bind() and add_layer() are ALLOWED to
    // allocate, and they must, because that is where the per-joint cursor
    // storage comes from. A bind() that allocated nothing would mean the
    // cursors were never sized and evaluate() would be doing it lazily.
    Skeleton skeleton = make_full_skeleton();
    std::vector<AnimationClip> clips = make_clips();
    ClipBindings bindings;
    bindings.build(skeleton, clips);

    AnimationRig rig;
    std::size_t during_bind = 0;
    std::size_t during_add = 0;
    {
        AllocScope scope;
        rig.bind(skeleton, clips, bindings);
        during_bind = scope.stop();
    }
    {
        AllocScope scope;
        LayerDesc base;
        base.name = "locomotion";
        const int index = rig.add_layer(std::move(base));
        during_add = scope.stop();
        EXPECT_EQ(index, 0);
    }
    // bind() with no layers yet has nothing to size, so it may legitimately be
    // zero; add_layer() sizes two samplers' cursor vectors and cannot be.
    EXPECT_GE(during_bind + during_add, 1u)
        << "cursor storage is not being sized at setup — check ClipSampler::bind";
}

// ---------------------------------------------------------------------------
// THE AUDIT
// ---------------------------------------------------------------------------

TEST(AnimationRigAlloc, EvaluateAllocatesNothingOnAClipLayer) {
    Skeleton skeleton = make_full_skeleton();
    std::vector<AnimationClip> clips = make_clips();
    ClipBindings bindings;
    bindings.build(skeleton, clips);

    AnimationRig rig;
    rig.bind(skeleton, clips, bindings);
    LayerDesc base;
    base.name = "locomotion";
    ASSERT_EQ(rig.add_layer(std::move(base)), 0);
    PoseScratch scratch;

    // Prime: the first evaluate is allowed to touch anything lazy. Everything
    // after it is the steady state the audit is about.
    rig.evaluate(skeleton, clips, k_dt, scratch);

    std::size_t seen = 0;
    {
        AllocScope scope;
        for (int step = 0; step < 240; ++step) {
            rig.evaluate(skeleton, clips, k_dt, scratch);
        }
        seen = scope.stop();
    }
    EXPECT_EQ(seen, 0u) << "AnimationRig::evaluate allocated on a plain clip layer";
}

TEST(AnimationRigAlloc, PlayClipAndTheCrossfadeThatFollowsAllocateNothing) {
    // A crossfade brings the second sampler into play for the first time. If
    // its cursor storage were sized lazily, THIS is where the allocation would
    // appear — mid-frame, on a worker, only for characters that changed clip.
    Skeleton skeleton = make_full_skeleton();
    std::vector<AnimationClip> clips = make_clips();
    ClipBindings bindings;
    bindings.build(skeleton, clips);

    AnimationRig rig;
    rig.bind(skeleton, clips, bindings);
    LayerDesc base;
    base.name = "locomotion";
    ASSERT_EQ(rig.add_layer(std::move(base)), 0);
    PoseScratch scratch;
    rig.evaluate(skeleton, clips, k_dt, scratch);

    std::size_t seen = 0;
    {
        AllocScope scope;
        for (int step = 0; step < 240; ++step) {
            if (step % 40 == 0) {
                rig.play_clip(0, (step / 40) % 4, 0.2f);
            }
            rig.evaluate(skeleton, clips, k_dt, scratch);
        }
        seen = scope.stop();
    }
    EXPECT_EQ(seen, 0u) << "play_clip or the crossfade that follows it allocated";
}

TEST(AnimationRigAlloc, EvaluateAllocatesNothingWithFourMaskedLayersAndRootMotion) {
    // The realistic worst case for a character: a base plus three masked layers
    // alternating Override and Additive, a caller-owned feather span, and root
    // motion extracting and stripping every frame.
    Skeleton skeleton = make_full_skeleton();
    std::vector<AnimationClip> clips = make_clips();
    ClipBindings bindings;
    bindings.build(skeleton, clips);

    AnimationRig rig;
    rig.bind(skeleton, clips, bindings);

    std::array<float, MAX_BONES> feather{};
    feather_by_depth(skeleton, 1, 3, std::span<float>(feather.data(), feather.size()));

    for (int i = 0; i < 4; ++i) {
        LayerDesc d;
        d.name = "layer" + std::to_string(i);
        d.mode = (i % 2 == 1) ? BlendMode::Additive : BlendMode::Override;
        d.weight = 0.5f;
        if (i > 0) {
            d.mask = BoneMask::none();
            d.mask.set_subtree(skeleton, 1);
            d.joint_weights = std::span<const float>(feather.data(), feather.size());
            // Reference from a CLIP rather than the rest pose, which is the
            // path that samples a whole extra pose per frame.
            d.additive_reference_clip = 3;
            d.additive_reference_time = 0.0f;
        }
        const int index = rig.add_layer(std::move(d));
        ASSERT_EQ(index, i);
        rig.play_clip(index, index % 4, 0.0f);
    }

    RootMotionConfig cfg;
    cfg.joint = 0;
    cfg.translation_xz = true;
    cfg.rotation_yaw = true;
    cfg.strip_from_pose = true;
    rig.set_root_motion(cfg);

    PoseScratch scratch;
    rig.evaluate(skeleton, clips, k_dt, scratch);

    std::size_t seen = 0;
    {
        AllocScope scope;
        for (int step = 0; step < 240; ++step) {
            rig.set_layer_weight(1, 0.5f + 0.4f * static_cast<float>(step % 2));
            rig.evaluate(skeleton, clips, k_dt, scratch);
        }
        seen = scope.stop();
    }
    EXPECT_EQ(seen, 0u) << "a four-layer rig with root motion allocated during evaluate";
}

TEST(AnimationRigAlloc, EvaluateAllocatesNothingOnABlendSpaceLayer) {
    // A blend space samples up to k_max_blend_samples poses per frame and
    // re-resolves its weights every frame as the query moves, so it crosses
    // sample boundaries during the measured window on purpose: a boundary cross
    // changes which samplers are used and is the most likely place for a lazy
    // resize to hide.
    Skeleton skeleton = make_full_skeleton();
    std::vector<AnimationClip> clips = make_clips();
    ClipBindings bindings;
    bindings.build(skeleton, clips);

    AnimationRig rig;
    rig.bind(skeleton, clips, bindings);
    LayerDesc base;
    base.name = "locomotion";
    ASSERT_EQ(rig.add_layer(std::move(base)), 0);

    BlendSpace1D space;
    ASSERT_TRUE(space.add_sample(0, 0.0f));
    ASSERT_TRUE(space.add_sample(1, 150.0f));
    ASSERT_TRUE(space.add_sample(2, 400.0f));
    space.finalize();
    rig.set_blend_space_1d(0, space); // sizes samplers — allowed to allocate

    PoseScratch scratch;
    rig.evaluate(skeleton, clips, k_dt, scratch);

    std::size_t seen = 0;
    {
        AllocScope scope;
        for (int step = 0; step < 240; ++step) {
            rig.set_blend_position(0, static_cast<float>(step) * 2.0f);
            rig.evaluate(skeleton, clips, k_dt, scratch);
        }
        seen = scope.stop();
    }
    EXPECT_EQ(seen, 0u) << "a blend-space layer allocated during evaluate";
}

TEST(AnimationRigAlloc, WritePaletteAllocatesNothing) {
    // write_palette runs on the frame-build path with a span into packet arena
    // memory. An allocation here would be an allocation per skinned draw.
    Skeleton skeleton = make_full_skeleton();
    std::vector<AnimationClip> clips = make_clips();
    ClipBindings bindings;
    bindings.build(skeleton, clips);

    AnimationRig rig;
    rig.bind(skeleton, clips, bindings);
    LayerDesc base;
    base.name = "locomotion";
    ASSERT_EQ(rig.add_layer(std::move(base)), 0);
    PoseScratch scratch;
    rig.evaluate(skeleton, clips, k_dt, scratch);

    std::vector<glm::mat4> dst(MAX_BONES, glm::mat4(0.0f));
    std::size_t seen = 0;
    {
        AllocScope scope;
        for (int i = 0; i < 64; ++i) {
            rig.write_palette(std::span<glm::mat4>(dst.data(), dst.size()));
        }
        seen = scope.stop();
    }
    EXPECT_EQ(seen, 0u) << "write_palette allocated";
}

TEST(AnimationRigAlloc, StateMachineDrivingALayerAllocatesNothingAcrossTransitions) {
    // The playground's per-frame shape: anim_sm_.update(rig, 0) then
    // rig.evaluate(...). The window deliberately covers several Idle/Walk/Run
    // transitions, because enter_state is the one place the state machine
    // touches the rig and `play_clip` is what it calls.
    Skeleton skeleton = make_full_skeleton();
    std::vector<AnimationClip> clips = make_clips();
    ClipBindings bindings;
    bindings.build(skeleton, clips);

    AnimationRig rig;
    rig.bind(skeleton, clips, bindings);
    LayerDesc base;
    base.name = "locomotion";
    ASSERT_EQ(rig.add_layer(std::move(base)), 0);

    AnimationStateMachine sm;
    using TC = TransitionCondition;
    const auto cond = [](const char* p, TC::Op op, float v) {
        TC c;
        c.param_name = p;
        c.op = op;
        c.threshold = v;
        return c;
    };
    AnimState idle;
    idle.name = "Idle";
    idle.clip_name = "Idle";
    {
        StateTransition t;
        t.target_state = "Walk";
        t.conditions.push_back(cond("speed", TC::Op::GT, 20.0f));
        t.crossfade_duration = 0.15f;
        idle.transitions.push_back(std::move(t));
    }
    AnimState walk;
    walk.name = "Walk";
    walk.clip_name = "Walk";
    {
        StateTransition t;
        t.target_state = "Idle";
        t.conditions.push_back(cond("speed", TC::Op::LT, 20.0f));
        t.crossfade_duration = 0.2f;
        walk.transitions.push_back(std::move(t));
    }
    sm.add_state(std::move(idle));
    sm.add_state(std::move(walk));
    sm.set_default_state("Idle");
    ASSERT_TRUE(sm.bind_clips(clips));
    const int speed = sm.param_index("speed");
    ASSERT_GE(speed, 0);

    PoseScratch scratch;
    sm.update(rig, 0);
    rig.evaluate(skeleton, clips, k_dt, scratch);

    std::size_t seen = 0;
    int transitions = 0;
    {
        AllocScope scope;
        for (int step = 0; step < 240; ++step) {
            // Oscillate across the threshold so the machine really transitions.
            sm.set_float(speed, (step / 30) % 2 == 0 ? 0.0f : 100.0f);
            const int before = rig.layer_clip(0);
            sm.update(rig, 0);
            if (rig.layer_clip(0) != before) {
                ++transitions;
            }
            rig.evaluate(skeleton, clips, k_dt, scratch);
        }
        seen = scope.stop();
    }
    EXPECT_EQ(seen, 0u) << "the state machine + rig frame path allocated";
    EXPECT_GE(transitions, 4) << "the window did not actually exercise a transition";
}
