// Micro-benchmarks for the CPU-side particle EffectSystem and the Curve
// evaluation it leans on per particle per frame.
//
// Run: ./build/src/bracken/bracken_benchmarks --benchmark_filter=BM_Effect
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/effects/effect_definition.hpp"
#include "bracken/effects/effect_system.hpp"
#include <benchmark/benchmark.h>
#include <cstdint>

using namespace bracken::systems;
using namespace bracken;

namespace {

constexpr float kDt = 1.0f / 60.0f;
constexpr float kParticleLifetime = 1.0f;
// 3 sim-seconds of warmup; spawns and deaths balance after one particle
// lifetime, so this comfortably reaches steady state.
constexpr int kWarmupSteps = 180;

// Continuous emitter tuned so the steady-state live particle count is
// spawn_rate * particle_lifetime. Uses the standard velocity path (gravity +
// drag) plus lifetime curves and a color gradient, matching a typical
// long-running effect.
EffectDefinition make_continuous_effect(float spawn_rate) {
    EffectDefinition def;
    def.name = "bench_continuous";

    EmitterDefinition emitter;
    emitter.spawn_mode = SpawnMode::CONTINUOUS;
    emitter.spawn_rate = spawn_rate;
    emitter.particle_lifetime = kParticleLifetime;
    emitter.duration = 1.0e9f; // effectively infinite - keeps spawning all run
    emitter.model = "bench_model";
    emitter.velocity.type = VelocityType::CUSTOM;
    emitter.velocity.speed = 10.0f;
    emitter.velocity.direction = {1, 0, 0};
    emitter.velocity.gravity = {0, -9.8f, 0};
    emitter.velocity.drag = 0.1f;
    emitter.appearance.scale_over_lifetime.type = CurveType::LINEAR;
    emitter.appearance.scale_over_lifetime.start_value = 1.0f;
    emitter.appearance.scale_over_lifetime.end_value = 0.0f;
    emitter.appearance.opacity_over_lifetime.type = CurveType::FADE_OUT_LATE;
    emitter.appearance.opacity_over_lifetime.start_value = 1.0f;
    emitter.appearance.opacity_over_lifetime.end_value = 0.0f;
    emitter.appearance.use_color_gradient = true;
    emitter.appearance.color_tint = {1, 0.5f, 0, 1};
    emitter.appearance.color_end = {0.2f, 0.2f, 0.2f, 0};
    def.emitters.push_back(emitter);
    return def;
}

EffectDefinition make_burst_effect(int count) {
    EffectDefinition def;
    def.name = "bench_burst";

    EmitterDefinition emitter;
    emitter.spawn_mode = SpawnMode::BURST;
    emitter.spawn_count = count;
    emitter.particle_lifetime = 5.0f;
    emitter.duration = -1.0f; // use particle_lifetime
    emitter.model = "bench_model";
    emitter.velocity.type = VelocityType::RADIAL; // exercises the RNG spawn path
    emitter.velocity.speed = 50.0f;
    def.emitters.push_back(emitter);
    return def;
}

// ===== update() with a steady-state continuous emitter =====
//
// Arg(N) is the target live particle count at steady state. Each measured
// update spawns ~dt*rate particles, ages/curves every live particle, and
// erase-removes the same number of dead ones.

void BM_EffectUpdate(benchmark::State& state) {
    const auto target_particles = static_cast<float>(state.range(0));
    // The definition must outlive the EffectSystem below - instances hold raw
    // pointers into it.
    const EffectDefinition def = make_continuous_effect(target_particles / kParticleLifetime);

    EffectSystem system;
    system.spawn_effect(&def, {0, 10, 0});

    // Reach steady state outside the measured loop.
    for (int i = 0; i < kWarmupSteps; ++i) {
        system.update(kDt);
    }

    for (auto _ : state) {
        system.update(kDt);
        benchmark::ClobberMemory();
    }

    state.counters["particles"] = static_cast<double>(system.particle_count());
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(system.particle_count()));
}
BENCHMARK(BM_EffectUpdate)->Arg(1000)->Arg(10000)->Unit(benchmark::kMicrosecond);

// Same steady-state workload through the three-phase stage-helper form
// (update(JobSystem&, dt)): SPAWN serial, INTEGRATE parallel_for over the
// flattened particle space (grain 512), COMPACT per-emitter. Args are
// {steady-state particles, workers}: workers==1 runs every chunk inline on
// the caller — that delta vs BM_EffectUpdate is the fan-out bookkeeping tax;
// workers==4 measures the actual speedup at each particle count.
void BM_EffectUpdateParallel(benchmark::State& state) {
    const auto target_particles = static_cast<float>(state.range(0));
    const auto workers = static_cast<unsigned>(state.range(1));
    const EffectDefinition def = make_continuous_effect(target_particles / kParticleLifetime);

    // Caller-owned pool sized by the benchmark arg (the singleton would pin
    // us to one ambient worker count).
    bracken::core::jobs::JobSystem js;
    js.init(workers);

    EffectSystem system;
    system.spawn_effect(&def, {0, 10, 0});
    for (int i = 0; i < kWarmupSteps; ++i) {
        system.update(js, kDt);
    }

    for (auto _ : state) {
        system.update(js, kDt);
        benchmark::ClobberMemory();
    }

    state.counters["particles"] = static_cast<double>(system.particle_count());
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(system.particle_count()));
}
BENCHMARK(BM_EffectUpdateParallel)
    ->Args({1000, 4})
    ->Args({10000, 1})
    ->Args({10000, 4})
    ->Args({50000, 4})
    ->Unit(benchmark::kMicrosecond);

// Same steady-state workload, but through the terrain-callback template
// overload - isolates the per-particle heightmap sample + clamp overhead.
void BM_EffectUpdateWithTerrain(benchmark::State& state) {
    const auto target_particles = static_cast<float>(state.range(0));
    const EffectDefinition def = make_continuous_effect(target_particles / kParticleLifetime);

    // Cheap analytic heightmap; kept below the particles so the clamp branch
    // stays predictable and the callback cost itself is what's measured.
    auto heightmap = [](float x, float z) { return 0.05f * x + 0.03f * z - 50.0f; };

    EffectSystem system;
    system.spawn_effect(&def, {0, 10, 0});

    for (int i = 0; i < kWarmupSteps; ++i) {
        system.update(kDt, heightmap);
    }

    for (auto _ : state) {
        system.update(kDt, heightmap);
        benchmark::ClobberMemory();
    }

    state.counters["particles"] = static_cast<double>(system.particle_count());
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(system.particle_count()));
}
BENCHMARK(BM_EffectUpdateWithTerrain)->Arg(1000)->Arg(10000)->Unit(benchmark::kMicrosecond);

// ===== Burst spawn =====
//
// Measures spawn_effect + the first update (BURST particles are spawned once
// age passes the zero delay), then clears so effects don't accumulate. The
// clear() is inside the timed region but is a single vector teardown, dwarfed
// by 256 particle constructions.
void BM_EffectSpawnBurst(benchmark::State& state) {
    constexpr int kBurstCount = 256;
    const EffectDefinition def = make_burst_effect(kBurstCount);

    EffectSystem system;
    for (auto _ : state) {
        int index = system.spawn_effect(&def, {0, 10, 0});
        system.update(kDt);
        benchmark::DoNotOptimize(index);
        benchmark::ClobberMemory();
        system.clear();
    }
    state.SetItemsProcessed(state.iterations() * kBurstCount);
}
BENCHMARK(BM_EffectSpawnBurst)->Unit(benchmark::kMicrosecond);

// ===== Curve::evaluate per CurveType =====
//
// Sweeps t across [0, 1] so branchy types (FADE_OUT_LATE, EASE_IN_OUT) see
// both sides of their split, matching how particle ages hit the curve.
void BM_CurveEvaluate(benchmark::State& state, CurveType type) {
    Curve curve;
    curve.type = type;
    curve.start_value = 2.0f;
    curve.end_value = 0.25f;
    curve.fade_start = 0.8f;
    // Keep the type from constant-folding the switch away.
    benchmark::DoNotOptimize(curve);

    constexpr int kSamples = 1024;
    constexpr float kStep = 1.0f / static_cast<float>(kSamples - 1);
    for (auto _ : state) {
        float sum = 0.0f;
        for (int i = 0; i < kSamples; ++i) {
            sum += curve.evaluate(static_cast<float>(i) * kStep);
        }
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(state.iterations() * kSamples);
}
BENCHMARK_CAPTURE(BM_CurveEvaluate, constant, CurveType::CONSTANT);
BENCHMARK_CAPTURE(BM_CurveEvaluate, linear, CurveType::LINEAR);
BENCHMARK_CAPTURE(BM_CurveEvaluate, ease_in, CurveType::EASE_IN);
BENCHMARK_CAPTURE(BM_CurveEvaluate, ease_out, CurveType::EASE_OUT);
BENCHMARK_CAPTURE(BM_CurveEvaluate, ease_in_out, CurveType::EASE_IN_OUT);
BENCHMARK_CAPTURE(BM_CurveEvaluate, fade_out_late, CurveType::FADE_OUT_LATE);

} // anonymous namespace
