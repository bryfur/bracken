#include "bracken/animation/ik_solver.hpp"

#include "bracken/animation/animation_types.hpp"
#include <array>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>

using namespace bracken::animation;

namespace {

// Cheap deterministic LCG (Numerical Recipes constants) so target
// randomization inside the timed loop costs a couple of integer ops.
struct Lcg {
    uint32_t state = 0x12345678u;

    float next01() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
    }

    float next_range(float lo, float hi) { return lo + (hi - lo) * next01(); }
};

// Joint indices in the benchmark rig. Leg/hip/spine names follow the Mixamo
// conventions in FootIKBoneNames so FootIKData::init resolves them.
constexpr int HIPS = 0;
constexpr int SPINE = 1;
constexpr int L_UPPER = 2;
constexpr int L_LOWER = 3;
constexpr int L_FOOT = 4;
constexpr int R_UPPER = 5;
constexpr int R_LOWER = 6;
constexpr int R_FOOT = 7;
constexpr int L_TOE = 8;
constexpr int R_TOE = 9;

struct BenchRig {
    Skeleton skeleton;
    // Bind-pose world transforms (translation-only), one per joint.
    std::array<glm::mat4, MAX_BONES> bind_world{};
    // world * inverse_bind at bind pose (identity, but computed for realism).
    std::array<glm::mat4, MAX_BONES> bind_bones{};
};

// Biped-like skeleton filling all MAX_BONES joints: hips + spine + two legs
// with toes (so the solver's end-effector child fixup has work to do), plus a
// long spine-descendant chain standing in for chest/arms/head/fingers so
// apply_body_lean touches a realistic share of the skeleton.
BenchRig make_biped_rig() {
    BenchRig rig;
    rig.skeleton.joints.reserve(MAX_BONES);

    auto add = [&rig](std::string name, int parent, const glm::vec3& pos) {
        Joint j;
        j.name = std::move(name);
        j.parent_index = parent;
        j.inverse_bind_matrix = glm::translate(glm::mat4(1.0f), -pos);
        glm::vec3 parent_pos = parent >= 0 ? glm::vec3(rig.bind_world[parent][3]) : glm::vec3(0.0f);
        j.local_translation = pos - parent_pos;
        j.local_rotation = glm::quat(1, 0, 0, 0);
        j.local_scale = glm::vec3(1.0f);
        size_t idx = rig.skeleton.joints.size();
        rig.bind_world[idx] = glm::translate(glm::mat4(1.0f), pos);
        rig.skeleton.joints.push_back(std::move(j));
    };

    add("Hips", -1, {0.0f, 0.90f, 0.0f});
    add("Spine", HIPS, {0.0f, 1.15f, 0.0f});
    add("LeftUpperLeg", HIPS, {0.15f, 0.90f, 0.0f});
    add("LeftLowerLeg", L_UPPER, {0.15f, 0.48f, 0.04f}); // slight knee bend
    add("LeftFoot", L_LOWER, {0.15f, 0.08f, 0.0f});
    add("RightUpperLeg", HIPS, {-0.15f, 0.90f, 0.0f});
    add("RightLowerLeg", R_UPPER, {-0.15f, 0.48f, 0.04f});
    add("RightFoot", R_LOWER, {-0.15f, 0.08f, 0.0f});
    add("LeftToe", L_FOOT, {0.15f, 0.02f, 0.12f});
    add("RightToe", R_FOOT, {-0.15f, 0.02f, 0.12f});

    for (int i = static_cast<int>(rig.skeleton.joints.size()); i < MAX_BONES; i++) {
        int parent = (i == R_TOE + 1) ? SPINE : i - 1;
        float k = static_cast<float>(i - R_TOE);
        add("Bone" + std::to_string(i), parent, glm::vec3(0.01f * k, 1.15f + 0.012f * k, 0.008f * k));
    }

    rig.skeleton.joint_node_indices.resize(MAX_BONES);
    for (int i = 0; i < MAX_BONES; i++) {
        rig.skeleton.joint_node_indices[i] = i;
        rig.bind_bones[i] = rig.bind_world[i] * rig.skeleton.joints[i].inverse_bind_matrix;
    }
    return rig;
}

// ============================================================================
// solve_two_bone_ik
// ============================================================================

// Arg 0: target comfortably inside the reachable band.
// Arg 1: target beyond max reach (clamped onto the fully-extended sphere).
// Arg 2: target inside min reach (clamped outward).
static void BM_SolveTwoBoneIK(benchmark::State& state) {
    const BenchRig rig = make_biped_rig();
    std::array<glm::mat4, MAX_BONES> world = rig.bind_world;
    std::array<glm::mat4, MAX_BONES> bones = rig.bind_bones;

    const glm::vec3 hip = glm::vec3(rig.bind_world[L_UPPER][3]);
    const glm::vec3 foot = glm::vec3(rig.bind_world[L_FOOT][3]);
    const glm::vec3 pole = glm::vec3(rig.bind_world[L_LOWER][3]) + glm::vec3(0.0f, 0.0f, 0.5f);
    const int mode = static_cast<int>(state.range(0));

    Lcg rng;
    for (auto _ : state) {
        // The solve mutates in place; restore only the joints it touches so
        // every iteration starts from bind pose without full-array copies.
        for (int idx : {L_UPPER, L_LOWER, L_FOOT, L_TOE}) {
            world[idx] = rig.bind_world[idx];
            bones[idx] = rig.bind_bones[idx];
        }

        glm::vec3 target;
        switch (mode) {
        case 0:
            target = foot + glm::vec3(rng.next_range(-0.1f, 0.1f), rng.next_range(0.08f, 0.30f),
                                      rng.next_range(-0.1f, 0.1f));
            break;
        case 1:
            target = hip + glm::vec3(rng.next_range(-0.3f, 0.3f), -(1.5f + rng.next01()),
                                     rng.next_range(-0.3f, 0.3f));
            break;
        default:
            target = hip + glm::vec3(0.0f, -(0.005f + 0.015f * rng.next01()), 0.002f);
            break;
        }

        solve_two_bone_ik(bones, world, rig.skeleton, L_UPPER, L_LOWER, L_FOOT, target, pole);
        benchmark::DoNotOptimize(bones.data());
        benchmark::DoNotOptimize(world.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SolveTwoBoneIK)->Arg(0)->Arg(1)->Arg(2);

// ============================================================================
// apply_foot_ik
// ============================================================================

// Arg 0: both feet above terrain — no pelvis drop, two leg solves.
// Arg 1: one foot below terrain — full-skeleton pelvis drop + one leg solve.
static void BM_ApplyFootIK(benchmark::State& state) {
    const BenchRig rig = make_biped_rig();
    FootIKData ik;
    ik.init(rig.skeleton);
    if (!ik.valid) {
        state.SkipWithError("FootIKData failed to resolve biped bone names");
        return;
    }

    std::array<glm::mat4, MAX_BONES> world = rig.bind_world;
    std::array<glm::mat4, MAX_BONES> bones = rig.bind_bones;
    const glm::mat4 model_to_world(1.0f);
    const bool pelvis_drop = state.range(0) != 0;

    Lcg rng;
    for (auto _ : state) {
        // Pelvis drop shifts every joint, so a full restore is required to
        // keep iterations identical. The two 8 KB copies are constant
        // overhead included in the measurement.
        world = rig.bind_world;
        bones = rig.bind_bones;

        float left = rng.next_range(0.3f, 0.8f);
        float right = rng.next_range(0.3f, 0.8f);
        if (pelvis_drop) {
            left = -left;
        }

        apply_foot_ik(bones, world, rig.skeleton, ik, model_to_world, 1.0f, left, right);
        benchmark::DoNotOptimize(bones.data());
        benchmark::DoNotOptimize(world.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * MAX_BONES);
}
BENCHMARK(BM_ApplyFootIK)->Arg(0)->Arg(1);

// ============================================================================
// apply_body_lean
// ============================================================================

static void BM_ApplyBodyLean(benchmark::State& state) {
    const BenchRig rig = make_biped_rig();
    std::array<glm::mat4, MAX_BONES> world = rig.bind_world;
    std::array<glm::mat4, MAX_BONES> bones = rig.bind_bones;

    Lcg rng;
    for (auto _ : state) {
        // No restore: the lean cost is data-independent (descendant flood
        // fill + matrix multiplies, no value-dependent early-outs), and the
        // pivot rotation keeps accumulated transforms bounded.
        float forward = rng.next_range(0.02f, 0.18f);
        float lateral = rng.next_range(-0.15f, 0.15f);

        apply_body_lean(bones, world, rig.skeleton, SPINE, forward, lateral);
        benchmark::DoNotOptimize(bones.data());
        benchmark::DoNotOptimize(world.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * MAX_BONES);
}
BENCHMARK(BM_ApplyBodyLean);

} // namespace
