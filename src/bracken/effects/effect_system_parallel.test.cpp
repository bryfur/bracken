// Worker-count parity for the three-phase EffectSystem::update fan-out
// (docs/multithreading_plan.md §6.2): the same scene stepped against a
// 1-worker and a 4-worker caller-owned pool must produce BIT-IDENTICAL
// particle state — same code path, chunking is granularity control only.
//
// The effect deliberately uses RNG-free spawn paths (CUSTOM velocity,
// DIRECTIONAL with zero spread, ORBITAL). Worker count never changes the RNG
// sequence — SPAWN runs serially on the calling thread in submission order —
// but the two EffectSystem instances in this A/B test share ONE thread's
// thread_local generator (seeded from std::random_device), so any RNG draw
// would diverge the *instances*, not the worker counts. RNG-free spawns make
// the comparison exact.
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/effects/effect_system.hpp"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <cstddef>
#include <string>

using namespace bracken::systems;
using namespace bracken;
using bracken::core::jobs::JobSystem;

namespace {

constexpr float kDt = 1.0f / 60.0f;

// ~5k steady-state particles spread over 20 emitters: INTEGRATE chunks
// (grain 512) straddle emitter boundaries, and COMPACT (emitter grain 16)
// actually fans out instead of running inline.
EffectDefinition make_parity_effect() {
    EffectDefinition def;
    def.name = "parity";

    // 18 continuous CUSTOM-velocity emitters, ~250 steady-state particles
    // each. CUSTOM draws nothing from the RNG; gravity/drag/curves/gradient
    // exercise the full standard integration path.
    for (int i = 0; i < 18; ++i) {
        EmitterDefinition e;
        e.name = "cont_" + std::to_string(i);
        e.spawn_mode = SpawnMode::CONTINUOUS;
        e.spawn_rate = 250.0f;
        e.particle_lifetime = 1.0f;
        e.duration = 100.0f;
        e.model = "parity_model";
        e.velocity.type = VelocityType::CUSTOM;
        e.velocity.speed = 5.0f + static_cast<float>(i);
        e.velocity.direction =
            glm::normalize(glm::vec3(0.3f * static_cast<float>(i) - 2.0f, 1.0f,
                                     0.7f - 0.1f * static_cast<float>(i)));
        e.velocity.gravity = {0.0f, -9.8f, 0.0f};
        e.velocity.drag = 0.05f * static_cast<float>(i % 3);
        e.rotation.rotation_rate = {10.0f * static_cast<float>(i), 20.0f, 5.0f};
        e.appearance.scale_over_lifetime.type = CurveType::LINEAR;
        e.appearance.scale_over_lifetime.start_value = 1.0f;
        e.appearance.scale_over_lifetime.end_value = 0.0f;
        e.appearance.opacity_over_lifetime.type = CurveType::FADE_OUT_LATE;
        e.appearance.use_color_gradient = true;
        e.appearance.color_tint = {1.0f, 0.5f, 0.1f, 1.0f};
        e.appearance.color_end = {0.1f, 0.1f, 0.4f, 0.0f};
        def.emitters.push_back(e);
    }

    // DIRECTIONAL with spread_angle == 0: skips the RNG spread draw;
    // face_velocity exercises the atan2 rotation branch.
    EmitterDefinition dir;
    dir.name = "directional";
    dir.spawn_mode = SpawnMode::CONTINUOUS;
    dir.spawn_rate = 300.0f;
    dir.particle_lifetime = 1.0f;
    dir.duration = 100.0f;
    dir.model = "parity_model";
    dir.velocity.type = VelocityType::DIRECTIONAL;
    dir.velocity.speed = 12.0f;
    dir.velocity.spread_angle = 0.0f;
    dir.velocity.gravity = {0.0f, -4.0f, 0.0f};
    dir.rotation.face_velocity = true;
    def.emitters.push_back(dir);

    // ORBITAL burst: no RNG, exercises the orbit integration branch and a
    // long-lived particle population COMPACT never removes mid-test.
    EmitterDefinition orb;
    orb.name = "orbital";
    orb.spawn_mode = SpawnMode::BURST;
    orb.spawn_count = 400;
    orb.particle_lifetime = 30.0f;
    orb.duration = -1.0f;
    orb.model = "parity_model";
    orb.velocity.type = VelocityType::ORBITAL;
    orb.velocity.orbit_radius = 8.0f;
    orb.velocity.orbit_speed = 1.5f;
    orb.velocity.orbit_height_base = 2.0f;
    orb.velocity.height_variation = 0.5f;
    def.emitters.push_back(orb);

    return def;
}

// Exact (==) field-by-field comparison — the contract is bit-identical
// output, not epsilon-close (float == is bit-exact for the non-NaN values
// this scene produces). One AssertionResult per call keeps failure output
// bounded.
::testing::AssertionResult systems_identical(const EffectSystem& a, const EffectSystem& b) {
    const auto& ea = a.get_effects();
    const auto& eb = b.get_effects();
    if (ea.size() != eb.size()) {
        return ::testing::AssertionFailure()
               << "effect count " << ea.size() << " vs " << eb.size();
    }
    for (std::size_t f = 0; f < ea.size(); ++f) {
        if (ea[f].age != eb[f].age) {
            return ::testing::AssertionFailure() << "effect " << f << " age mismatch";
        }
        if (ea[f].emitters.size() != eb[f].emitters.size()) {
            return ::testing::AssertionFailure() << "effect " << f << " emitter count mismatch";
        }
        for (std::size_t m = 0; m < ea[f].emitters.size(); ++m) {
            const EmitterInstance& ma = ea[f].emitters[m];
            const EmitterInstance& mb = eb[f].emitters[m];
            if (ma.age != mb.age || ma.next_spawn_time != mb.next_spawn_time ||
                ma.has_spawned_burst != mb.has_spawned_burst) {
                return ::testing::AssertionFailure() << "emitter " << m << " state mismatch";
            }
            if (ma.particles.size() != mb.particles.size()) {
                return ::testing::AssertionFailure()
                       << "emitter " << m << " particle count " << ma.particles.size() << " vs "
                       << mb.particles.size();
            }
            for (std::size_t p = 0; p < ma.particles.size(); ++p) {
                const Particle& pa = ma.particles[p];
                const Particle& pb = mb.particles[p];
                const bool same =
                    pa.position == pb.position && pa.velocity == pb.velocity &&
                    pa.rotation == pb.rotation && pa.rotation_rate == pb.rotation_rate &&
                    pa.scale == pb.scale && pa.opacity == pb.opacity && pa.color == pb.color &&
                    pa.age == pb.age && pa.lifetime == pb.lifetime &&
                    pa.orbit_angle == pb.orbit_angle && pa.orbit_index == pb.orbit_index &&
                    pa.orbit_origin == pb.orbit_origin && pa.arc_origin == pb.arc_origin &&
                    pa.arc_direction == pb.arc_direction;
                if (!same) {
                    return ::testing::AssertionFailure()
                           << "emitter " << m << " particle " << p << " diverged";
                }
            }
        }
    }
    return ::testing::AssertionSuccess();
}

} // anonymous namespace

TEST(EffectSystemParallel, ParityWithTerrain1v4Workers) {
    // Caller-owned pools (§6.2 parity harness): same code path, different
    // worker counts. js1 runs every chunk on one worker; js4 fans out.
    JobSystem js1;
    js1.init(1);
    JobSystem js4;
    js4.init(4);

    const EffectDefinition def = make_parity_effect();
    EffectSystem sys1;
    EffectSystem sys4;
    ASSERT_EQ(sys1.spawn_effect(&def, {0, 20, 0}, {1, 0, 0}), 0);
    ASSERT_EQ(sys4.spawn_effect(&def, {0, 20, 0}, {1, 0, 0}), 0);

    // Pure analytic sampler — the thread-safety the callback contract
    // requires (const math over captured constants, no shared writes).
    auto terrain = [](float x, float z) { return 0.05f * x + 0.03f * z; };

    for (int frame = 0; frame < 60; ++frame) {
        sys1.update(js1, kDt, terrain);
        sys4.update(js4, kDt, terrain);
        ASSERT_TRUE(systems_identical(sys1, sys4)) << "frame " << frame;
    }
    // The fan-out must have had real work: ~5k steady-state particles means
    // INTEGRATE ran ~10 chunks and COMPACT ~2 on the 4-worker pool.
    EXPECT_GT(sys1.particle_count(), 4000u);
}

TEST(EffectSystemParallel, ParityNoTerrainOverload) {
    JobSystem js1;
    js1.init(1);
    JobSystem js4;
    js4.init(4);

    const EffectDefinition def = make_parity_effect();
    EffectSystem sys1;
    EffectSystem sys4;
    sys1.spawn_effect(&def, {3, 1, -2});
    sys4.spawn_effect(&def, {3, 1, -2});

    for (int frame = 0; frame < 60; ++frame) {
        sys1.update(js1, kDt);
        sys4.update(js4, kDt);
    }
    ASSERT_TRUE(systems_identical(sys1, sys4));
    EXPECT_GT(sys1.particle_count(), 4000u);
}

TEST(EffectSystemParallel, SmallSceneStaysInlineAndMatches) {
    // ~100 particles: below INTEGRATE_GRAIN and COMPACT_EMITTER_GRAIN, so
    // both pools run every phase inline on the caller — parity is trivial
    // but pins the grain guards (a regression that fanned out tiny scenes
    // would still be *correct*; this documents the intended inline path).
    JobSystem js1;
    js1.init(1);
    JobSystem js4;
    js4.init(4);

    EffectDefinition def;
    def.name = "small";
    EmitterDefinition e;
    e.spawn_mode = SpawnMode::BURST;
    e.spawn_count = 100;
    e.particle_lifetime = 2.0f;
    e.duration = -1.0f;
    e.velocity.type = VelocityType::CUSTOM;
    e.velocity.speed = 3.0f;
    e.velocity.direction = {0, 1, 0};
    e.velocity.gravity = {0, -9.8f, 0};
    def.emitters.push_back(e);

    EffectSystem sys1;
    EffectSystem sys4;
    sys1.spawn_effect(&def, {0, 5, 0});
    sys4.spawn_effect(&def, {0, 5, 0});

    for (int frame = 0; frame < 30; ++frame) {
        sys1.update(js1, kDt);
        sys4.update(js4, kDt);
    }
    ASSERT_TRUE(systems_identical(sys1, sys4));
    EXPECT_EQ(sys1.particle_count(), 100u);
}
