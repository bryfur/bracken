#include "bracken/animation/animation_player.hpp"
#include "bracken/animation/animation_rig.hpp"
#include "bracken/animation/animation_state_machine.hpp"
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <array>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace bracken::animation;

namespace {

constexpr float kClipDuration = 2.0f;
constexpr int kKeysPerChannel = 16;
constexpr float kFrameDt = 0.016f; // ~60 Hz host update
constexpr size_t kNumRandomTimes = 4096; // power of two so the index wraps with a mask

// Balanced-tree skeleton (parent = (i-1)/2) — closer to a real rig than a
// single chain, while keeping parent_index < child as the transform
// propagation pass in compute_bone_matrices requires.
Skeleton make_bench_skeleton(int num_joints) {
    Skeleton skel;
    skel.joints.resize(static_cast<size_t>(num_joints));
    for (int i = 0; i < num_joints; i++) {
        Joint& joint = skel.joints[static_cast<size_t>(i)];
        joint.name = "joint_" + std::to_string(i);
        joint.parent_index = (i == 0) ? -1 : (i - 1) / 2;
        joint.local_translation = glm::vec3(0.0f, 0.25f, 0.0f);
        joint.local_rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        joint.local_scale = glm::vec3(1.0f);
        joint.inverse_bind_matrix =
            glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.25f * static_cast<float>(i), 0.0f));
    }
    return skel;
}

// 16-key rotation + position channels on every joint (no scale keys, matching
// typical exported clips). `phase` de-correlates two clips so a crossfade
// blends genuinely different poses.
AnimationClip make_bench_clip(std::string name, int num_joints, float phase) {
    AnimationClip clip;
    clip.name = std::move(name);
    clip.duration = kClipDuration;
    clip.channels.reserve(static_cast<size_t>(num_joints));
    for (int j = 0; j < num_joints; j++) {
        AnimationChannel ch;
        ch.bone_index = j;
        ch.position_times.reserve(kKeysPerChannel);
        ch.positions.reserve(kKeysPerChannel);
        ch.rotation_times.reserve(kKeysPerChannel);
        ch.rotations.reserve(kKeysPerChannel);
        for (int k = 0; k < kKeysPerChannel; k++) {
            float t = kClipDuration * static_cast<float>(k) / static_cast<float>(kKeysPerChannel - 1);
            float w = t * 3.0f + phase + static_cast<float>(j) * 0.1f;
            ch.position_times.push_back(t);
            ch.positions.push_back(glm::vec3(std::sin(w) * 0.1f, 0.25f + std::cos(w) * 0.05f, 0.0f));
            ch.rotation_times.push_back(t);
            ch.rotations.push_back(glm::angleAxis(std::sin(w) * 0.5f, glm::normalize(glm::vec3(0.3f, 1.0f, 0.2f))));
        }
        clip.channels.push_back(std::move(ch));
    }
    return clip;
}

std::vector<float> make_key_times(int num_keys) {
    std::vector<float> times(static_cast<size_t>(num_keys));
    for (int k = 0; k < num_keys; k++) {
        times[static_cast<size_t>(k)] = kClipDuration * static_cast<float>(k) / static_cast<float>(num_keys - 1);
    }
    return times;
}

std::vector<glm::quat> make_quat_keys(int num_keys) {
    std::vector<glm::quat> values(static_cast<size_t>(num_keys));
    for (int k = 0; k < num_keys; k++) {
        float w = static_cast<float>(k) * 0.37f;
        values[static_cast<size_t>(k)] =
            glm::angleAxis(std::sin(w) * 0.5f, glm::normalize(glm::vec3(0.3f, 1.0f, 0.2f)));
    }
    return values;
}

std::vector<glm::vec3> make_vec3_keys(int num_keys) {
    std::vector<glm::vec3> values(static_cast<size_t>(num_keys));
    for (int k = 0; k < num_keys; k++) {
        float w = static_cast<float>(k) * 0.37f;
        values[static_cast<size_t>(k)] = glm::vec3(std::sin(w) * 0.1f, 0.25f + std::cos(w) * 0.05f, 0.0f);
    }
    return values;
}

std::vector<float> make_random_times() {
    std::vector<float> ts(kNumRandomTimes);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, kClipDuration);
    for (auto& t : ts) {
        t = dist(rng);
    }
    return ts;
}

// 8-state ring: state i -> state (i+1)%8. Conditions are ordered so the "go"
// gate is evaluated last — when it is false, every other condition still runs.
// Even states carry 3 conditions, odd states 2.
AnimationStateMachine make_bench_state_machine() {
    constexpr int num_states = 8;
    AnimationStateMachine sm;
    std::vector<AnimationClip> clips;
    for (int i = 0; i < num_states; i++) {
        AnimState s;
        s.name = "state" + std::to_string(i);
        s.clip_name = "clip" + std::to_string(i);
        s.loop = true;

        StateTransition to_next;
        to_next.target_state = "state" + std::to_string((i + 1) % num_states);
        to_next.crossfade_duration = 0.2f;

        TransitionCondition speed_gt;
        speed_gt.param_name = "speed";
        speed_gt.op = TransitionCondition::Op::GT;
        speed_gt.threshold = 0.5f;
        to_next.conditions.push_back(speed_gt);

        if (i % 2 == 0) {
            TransitionCondition not_stunned;
            not_stunned.param_name = "stunned";
            not_stunned.op = TransitionCondition::Op::IS_FALSE;
            to_next.conditions.push_back(not_stunned);
        }

        TransitionCondition go;
        go.param_name = "go";
        go.op = TransitionCondition::Op::IS_TRUE;
        to_next.conditions.push_back(go);

        s.transitions.push_back(to_next);
        sm.add_state(std::move(s));

        AnimationClip clip;
        clip.name = "clip" + std::to_string(i);
        clip.duration = 1.0f;
        clips.push_back(std::move(clip));
    }
    sm.set_default_state("state0");
    (void)sm.bind_clips(clips); // synthetic clips always bind
    return sm;
}

} // namespace

// ============================================================================
// AnimationPlayer::update — single clip, steady playback
// ============================================================================

static void BM_AnimationPlayerUpdate(benchmark::State& state) {
    const int num_joints = static_cast<int>(state.range(0));
    Skeleton skeleton = make_bench_skeleton(num_joints);
    std::vector<AnimationClip> clips;
    clips.push_back(make_bench_clip("walk", num_joints, 0.0f));

    AnimationPlayer player;
    player.update(skeleton, clips, kFrameDt); // prime channel lookup + cursors

    for (auto _ : state) {
        player.update(skeleton, clips, kFrameDt);
        benchmark::DoNotOptimize(player.bone_matrices().data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * num_joints);
}
BENCHMARK(BM_AnimationPlayerUpdate)->Arg(16)->Arg(64)->Arg(128);

// ============================================================================
// AnimationPlayer::update — mid-crossfade between two clips
// ============================================================================

static void BM_AnimationCrossfade(benchmark::State& state) {
    const int num_joints = static_cast<int>(state.range(0));
    Skeleton skeleton = make_bench_skeleton(num_joints);
    std::vector<AnimationClip> clips;
    clips.push_back(make_bench_clip("walk", num_joints, 0.0f));
    clips.push_back(make_bench_clip("run", num_joints, 1.7f));

    AnimationPlayer player;
    player.update(skeleton, clips, kFrameDt);
    // An enormous blend duration keeps blend_factor < 1 for the entire run,
    // so every measured update samples and blends BOTH clips (the true
    // mid-crossfade hot path) instead of finishing the fade after ~12 frames.
    player.crossfade_to(1, 1.0e9f);

    for (auto _ : state) {
        player.update(skeleton, clips, kFrameDt);
        benchmark::DoNotOptimize(player.bone_matrices().data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * num_joints);
}
BENCHMARK(BM_AnimationCrossfade)->Arg(16)->Arg(64)->Arg(128);

// ============================================================================
// interpolate_keyframes — warm cursor (sequential t) vs cold cursor (random t)
// ============================================================================

static void BM_InterpolateKeyframes_Quat_WarmCursor(benchmark::State& state) {
    const int num_keys = static_cast<int>(state.range(0));
    std::vector<float> times = make_key_times(num_keys);
    std::vector<glm::quat> values = make_quat_keys(num_keys);

    size_t cursor = 0;
    float t = 0.0f;
    for (auto _ : state) {
        t += kFrameDt;
        if (t >= kClipDuration) {
            t -= kClipDuration;
        }
        glm::quat q = interpolate_keyframes(times, values, t, cursor);
        benchmark::DoNotOptimize(q);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_InterpolateKeyframes_Quat_WarmCursor)->Arg(16)->Arg(256);

static void BM_InterpolateKeyframes_Quat_ColdCursor(benchmark::State& state) {
    const int num_keys = static_cast<int>(state.range(0));
    std::vector<float> times = make_key_times(num_keys);
    std::vector<glm::quat> values = make_quat_keys(num_keys);
    std::vector<float> random_ts = make_random_times();

    size_t cursor = 0; // stale relative to each random t -> upper_bound path
    size_t idx = 0;
    for (auto _ : state) {
        glm::quat q = interpolate_keyframes(times, values, random_ts[idx], cursor);
        idx = (idx + 1) & (kNumRandomTimes - 1);
        benchmark::DoNotOptimize(q);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_InterpolateKeyframes_Quat_ColdCursor)->Arg(16)->Arg(256);

static void BM_InterpolateKeyframes_Vec3_WarmCursor(benchmark::State& state) {
    const int num_keys = static_cast<int>(state.range(0));
    std::vector<float> times = make_key_times(num_keys);
    std::vector<glm::vec3> values = make_vec3_keys(num_keys);

    size_t cursor = 0;
    float t = 0.0f;
    for (auto _ : state) {
        t += kFrameDt;
        if (t >= kClipDuration) {
            t -= kClipDuration;
        }
        glm::vec3 v = interpolate_keyframes(times, values, t, cursor);
        benchmark::DoNotOptimize(v);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_InterpolateKeyframes_Vec3_WarmCursor)->Arg(16)->Arg(256);

static void BM_InterpolateKeyframes_Vec3_ColdCursor(benchmark::State& state) {
    const int num_keys = static_cast<int>(state.range(0));
    std::vector<float> times = make_key_times(num_keys);
    std::vector<glm::vec3> values = make_vec3_keys(num_keys);
    std::vector<float> random_ts = make_random_times();

    size_t cursor = 0;
    size_t idx = 0;
    for (auto _ : state) {
        glm::vec3 v = interpolate_keyframes(times, values, random_ts[idx], cursor);
        idx = (idx + 1) & (kNumRandomTimes - 1);
        benchmark::DoNotOptimize(v);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_InterpolateKeyframes_Vec3_ColdCursor)->Arg(16)->Arg(256);

// ============================================================================
// AnimationStateMachine::update — 8-state ring, per-frame parameter feed
// ============================================================================

static void BM_StateMachineUpdate(benchmark::State& state) {
    AnimationStateMachine sm = make_bench_state_machine();
    sm.set_float("speed", 1.0f);
    sm.set_bool("stunned", false);
    sm.set_bool("go", false);
    AnimationPlayer player;

    bool go = false;
    for (auto _ : state) {
        // Flip the gate every frame, mirroring a host feeding parameters:
        // when false, all conditions are evaluated and the transition fails
        // on the final "go"; when true, the transition fires to the next
        // state in the ring (enter_state + player crossfade setup).
        go = !go;
        sm.set_bool("go", go);
        sm.update(player);
        benchmark::DoNotOptimize(player.current_clip());
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_StateMachineUpdate);

// ============================================================================
// AnimationRig::evaluate — the layer stack, 1 / 2 / 4 layers deep
//
// The comparison that matters is against BM_AnimationPlayerUpdate at the same
// joint count: a one-layer rig does the same work the player does, so anything
// beyond noise there is the layer machinery's overhead and should be reported.
// Each additional layer is one more full skeleton sample plus one blend pass.
//
// Meaningless from a default configure (-O0 with live asserts). Reconfigure
// with -DCMAKE_BUILD_TYPE=RelWithDebInfo before quoting a number.
// ============================================================================

namespace {

struct RigBenchFixture {
    Skeleton skeleton;
    std::vector<AnimationClip> clips;
    ClipBindings bindings;
    AnimationRig rig;
    PoseScratch scratch;

    RigBenchFixture(int num_joints, int layers) : skeleton(make_bench_skeleton(num_joints)) {
        clips.push_back(make_bench_clip("walk", num_joints, 0.0f));
        clips.push_back(make_bench_clip("run", num_joints, 1.7f));
        clips.push_back(make_bench_clip("aim", num_joints, 0.4f));
        bindings.build(skeleton, clips);
        rig.bind(skeleton, clips, bindings);

        for (int i = 0; i < layers; ++i) {
            LayerDesc d;
            d.name = "layer" + std::to_string(i);
            // Alternate the modes so the benchmark exercises both operators
            // (Additive is the more expensive one — it samples a reference).
            d.mode = (i % 2 == 1) ? BlendMode::Additive : BlendMode::Override;
            d.weight = 0.5f;
            if (i > 0) {
                // A half-skeleton mask, the realistic upper-body shape.
                d.mask.set_subtree(skeleton, num_joints / 4);
            }
            const int index = rig.add_layer(std::move(d));
            if (index > 0) {
                rig.play_clip(index, index % 3, 0.0f);
            }
        }
    }
};

} // namespace

static void BM_AnimationRigEvaluate(benchmark::State& state) {
    const int num_joints = static_cast<int>(state.range(0));
    const int layers = static_cast<int>(state.range(1));
    RigBenchFixture f(num_joints, layers);
    f.rig.evaluate(f.skeleton, f.clips, kFrameDt, f.scratch); // prime cursors

    for (auto _ : state) {
        f.rig.evaluate(f.skeleton, f.clips, kFrameDt, f.scratch);
        benchmark::DoNotOptimize(f.rig.bone_matrices().data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * num_joints * layers);
}
BENCHMARK(BM_AnimationRigEvaluate)
    ->Args({16, 1})
    ->Args({64, 1})
    ->Args({128, 1})
    ->Args({128, 2})
    ->Args({128, 4});

// ============================================================================
// BlendSpace2D::resolve — the O(n^2) gradient band
//
// 4 samples is 12 pair evaluations, 8 is 56. This is per layer per character
// per frame, so the shape of this curve is what k_max_blend_samples is set
// against.
// ============================================================================

static void BM_BlendSpace2DResolve(benchmark::State& state) {
    const int samples = static_cast<int>(state.range(0));
    BlendSpace2D space;
    (void)space.add_sample(0, glm::vec2(0.0f, 0.0f));
    for (int k = 1; k < samples; ++k) {
        const float a = static_cast<float>(k) * 6.2831853f / static_cast<float>(samples - 1);
        (void)space.add_sample(k, glm::vec2(std::cos(a), std::sin(a)));
    }

    std::array<WeightedClip, k_max_blend_samples> out{};
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.2f, 1.2f);
    std::vector<glm::vec2> queries(kNumRandomTimes);
    for (auto& q : queries) {
        q = glm::vec2(dist(rng), dist(rng));
    }

    size_t i = 0;
    for (auto _ : state) {
        std::uint32_t n = space.resolve(queries[i & (kNumRandomTimes - 1)], out);
        ++i;
        benchmark::DoNotOptimize(n);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * samples);
}
BENCHMARK(BM_BlendSpace2DResolve)->Arg(4)->Arg(8);
