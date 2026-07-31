#pragma once

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/effect_definition.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bracken::scene {
class FrameBuilder;
}

namespace bracken::systems {

// Types from bracken (parent namespace) are directly accessible
using bracken::AppearanceDefinition;
using bracken::EffectDefinition;
using bracken::EmitterDefinition;
using bracken::RotationDefinition;
using bracken::VelocityDefinition;

// Runtime particle instance
struct Particle {
    // Transform
    glm::vec3 position = {0, 0, 0};
    glm::vec3 velocity = {0, 0, 0};
    glm::vec3 rotation = {0, 0, 0};      // Current euler angles (radians)
    glm::vec3 rotation_rate = {0, 0, 0}; // Angular velocity (radians/sec)

    // Appearance
    float scale = 1.0f;
    float opacity = 1.0f;
    glm::vec4 color = {1, 1, 1, 1};

    // Timing
    float age = 0.0f;      // How long this particle has been alive
    float lifetime = 1.0f; // Total lifetime before death

    // For orbital particles
    float orbit_angle = 0.0f;
    int orbit_index = 0;                // Which object in the orbit (0, 1, 2, etc.)
    glm::vec3 orbit_origin = {0, 0, 0}; // Center point of orbit

    // For arc particles (melee slash)
    glm::vec3 arc_origin = {0, 0, 0};    // Center point of the arc
    glm::vec3 arc_direction = {1, 0, 0}; // Base direction for the arc
};

// Runtime emitter instance (spawns and manages particles)
struct EmitterInstance {
    const EmitterDefinition* definition = nullptr;

    // Particles spawned by this emitter
    std::vector<Particle> particles;

    // Emitter state
    float age = 0.0f;
    float next_spawn_time = 0.0f;   // For continuous spawning
    bool has_spawned_burst = false; // For burst mode

    // Spawn location and direction
    glm::vec3 origin = {0, 0, 0};
    glm::vec3 direction = {1, 0, 0};

    // Per-emitter PRNG counter. Seeded deterministically at spawn from the
    // owning EffectSystem's spawn sequence, so an emitter's particle stream
    // depends only on when it was spawned — not on the wall clock and not on
    // which worker steps it. Advanced by every random draw the emitter makes.
    uint64_t rng = 0;

    // Check if emitter is still active
    bool is_active() const {
        if (!definition) {
            return false;
        }
        float duration = definition->duration < 0 ? definition->particle_lifetime : definition->duration;
        return age < duration;
    }

    // Check if emitter has finished and all particles are dead
    bool is_complete() const { return !is_active() && particles.empty(); }
};

// Runtime effect instance (collection of emitters)
struct EffectInstance {
    const EffectDefinition* definition = nullptr;

    // Emitters in this effect
    std::vector<EmitterInstance> emitters;

    // Effect age in seconds, compared against EffectDefinition::duration.
    float age = 0.0f;

    // Check if effect is complete
    bool is_complete() const {
        // Effect-level duration (>= 0) overrides emitter timing: the effect
        // is complete once its duration has elapsed and every particle died
        // (spawning is gated off past the duration in update()).
        if (definition && definition->duration >= 0.0f && age >= definition->duration) {
            for (const auto& emitter : emitters) {
                if (!emitter.particles.empty()) {
                    return false;
                }
            }
            return true;
        }
        for (const auto& emitter : emitters) {
            if (!emitter.is_complete()) {
                return false;
            }
        }
        return true;
    }
};

// Effect system - manages active effects and updates particles.
//
// Thread-safety tier: `sim-stage` owned (docs/multithreading_plan.md §6.1,
// Phase 4). The Application owns the instance and drives it from the sim-job
// tail each frame: apply the sealed builder's spawn commands, update(), then
// snapshot_into() the packet. update() internally fans out onto the given
// JobSystem and joins before returning (nested fork-join from inside the sim
// job is supported), so nothing here is touched by workers between updates.
// The RENDERER never reads the live system — it consumes the packet's
// effect-instance snapshot (FramePacket::effect_emitters()/effect_particles()),
// because the next frame's sim mutates this object concurrently with
// rendering.
class EffectSystem {
public:
    // Spawn a new effect at a location.
    // Returns index of the spawned effect, or -1 if failed.
    // CONTINUOUS emitters with a non-positive (or non-finite) spawn_rate are
    // rejected here with a warning (once per definition) and stay inert.
    int spawn_effect(const EffectDefinition* definition, const glm::vec3& position,
                     const glm::vec3& direction = {1, 0, 0});

    // Update all active effects and particles — the stage-helper form
    // (docs/multithreading_plan.md §4.3: "spawn applied at sim; integrate is
    // per-particle parallel_for; compaction per-emitter"). Three phases:
    //   SPAWN     — serial on the calling thread, in effect/emitter
    //               submission order (see update_t: RNG-sequence
    //               determinism);
    //   INTEGRATE — parallel_for over a flattened particle index space
    //               across all emitters;
    //   COMPACT   — per-emitter dead-particle removal, parallel over
    //               emitters.
    // Output is bit-identical to a serial run at any worker count: the
    // parallel phases write only disjoint per-particle / per-emitter state —
    // no shared accumulators, no atomics, nothing to merge (§6.2).
    //
    // TerrainHeightFn is invoked per particle (templating avoids
    // std::function indirection in the inner loop) FROM JOB WORKERS during
    // INTEGRATE: it MUST be safe to call concurrently from multiple threads —
    // a pure function of (x, z) over state nothing mutates for the duration
    // of update(). The engine's terrain sampler qualifies (const bilinear
    // reads of an immutable Heightmap — TerrainRenderer::get_height →
    // Heightmap::get_height_world); anything that writes shared state does
    // not.
    template<typename TerrainHeightFn>
    void update(core::jobs::JobSystem& js, float dt, TerrainHeightFn&& get_terrain_height);

    // Update with no terrain sampling (skips the per-particle terrain clamp).
    void update(core::jobs::JobSystem& js, float dt);

    // Compatibility forms servicing the canonical singleton pool — thin
    // delegators, identical semantics (uninitialized singleton ⇒ every
    // parallel_for chunk runs inline on the caller).
    template<typename TerrainHeightFn> void update(float dt, TerrainHeightFn&& get_terrain_height) {
        update(core::jobs::JobSystem::instance(), dt, std::forward<TerrainHeightFn>(get_terrain_height));
    }
    void update(float dt) { update(core::jobs::JobSystem::instance(), dt); }
    void update(float dt, std::nullptr_t) { update(dt); }

    // Write this frame's effect-instance snapshot into the packet through the
    // builder's engine-internal API (alloc_effect_snapshot): live emitters in
    // deterministic (effect, emitter) order with their render-relevant
    // per-particle state. Sim-stage tail, single-threaded, after update();
    // the renderer consumes the sealed spans, never this object.
    void snapshot_into(scene::FrameBuilder& fb) const;

    // Get all active effects (sim-stage/test introspection; the renderer
    // reads the packet snapshot instead)
    const std::vector<EffectInstance>& get_effects() const { return effects_; }

    // Clear all active effects
    void clear() { effects_.clear(); }

    // Get number of active effects
    size_t effect_count() const { return effects_.size(); }

    // Get total number of active particles across all effects
    size_t particle_count() const;

private:
    std::vector<EffectInstance> effects_;

    // Stateless sentinel used by the no-terrain update overload.
    struct NoTerrainFn {
        float operator()(float, float) const { return 0.0f; }
    };

    // INTEGRATE grain: ~512 particles per chunk keeps a chunk in the
    // tens-of-microseconds range (well above dispatch cost), and — via
    // parallel_for's total<=grain inline guard — a small scene (the
    // playground's few hundred particles) runs as ONE chunk on the calling
    // thread: granularity control, not a second code path.
    static constexpr std::size_t INTEGRATE_GRAIN = 512;
    // COMPACT fans out only at >= this many live emitters; typical scenes
    // (a handful of emitters) stay inline on the caller.
    static constexpr std::size_t COMPACT_EMITTER_GRAIN = 16;

    // Shared three-phase update body, templated on the terrain sampler so
    // the particle-level check folds away when sampling is disabled
    // (HasTerrain=false compiles out the per-particle callable invocation).
    template<typename TerrainHeightFn, bool HasTerrain>
    void update_t(core::jobs::JobSystem& js, float dt, TerrainHeightFn& get_terrain_height);

    // SPAWN phase for one emitter: advances its age and applies the
    // burst/continuous spawn rules. Serial by contract — see update_t.
    // allow_spawn is the effect-level duration gate: false once the owning
    // effect has outlived EffectDefinition::duration.
    void spawn_phase(EmitterInstance& emitter, float dt, bool allow_spawn);

    // SPAWN-phase only (serial): both draw from the owning emitter's
    // deterministic RNG counter (EmitterInstance::rng), never a shared or
    // thread_local generator.
    void spawn_particles(EmitterInstance& emitter, int count);
    glm::vec3 calculate_initial_velocity(const VelocityDefinition& vel_def, const glm::vec3& direction,
                                         uint64_t& rng);

    // Pure per-particle step: reads/writes only `particle` plus the const
    // definition — worker-safe; called from the INTEGRATE fan-out.
    static void update_particle(Particle& particle, const EmitterDefinition& emitter_def, float dt);

    // Compacts dead effects after updating - shared between overloads.
    void compact_effects();

    // Effect-level spawn gate shared by both update() overloads.
    static bool effect_allows_spawning(const EffectInstance& effect) {
        return !(effect.definition && effect.definition->duration >= 0.0f &&
                 effect.age >= effect.definition->duration);
    }

    // Definitions already warned about for an invalid CONTINUOUS spawn_rate
    // (warn once per definition; definitions have stable addresses).
    std::unordered_set<const EmitterDefinition*> warned_bad_spawn_rate_;

    // Monotonic emitter-seed source. Spawn order is deterministic, so this
    // gives each emitter a reproducible, mutually decorrelated stream.
    uint64_t rng_seed_counter_ = 0;

    // INTEGRATE/COMPACT scratch, rebuilt every update (members so steady
    // state doesn't reallocate): the frame's live (non-empty, valid) emitters
    // and the exclusive-prefix offset of each emitter's particle range in the
    // flattened index space. Owned: written only by the update() caller
    // BEFORE the fan-out; worker chunks read them, never resize.
    std::vector<EmitterInstance*> live_emitters_;
    std::vector<std::size_t> emitter_offsets_;
};

// ============================================================================
// Template definitions (header-only so any caller can instantiate them)
// ============================================================================

template<typename TerrainHeightFn, bool HasTerrain>
void EffectSystem::update_t(core::jobs::JobSystem& js, float dt, TerrainHeightFn& get_terrain_height) {
    // ---- SPAWN (serial, calling thread) -----------------------------------
    // Runs in effect/emitter submission order on the caller.
    // calculate_initial_velocity draws from the thread_local RNG, so keeping
    // spawn serial here means the RNG sequence is UNCHANGED at any worker
    // count — the parallel phases below never touch the RNG. Particles
    // spawned this frame are integrated below with the full dt, matching the
    // previous single-pass behavior.
    for (auto& effect : effects_) {
        effect.age += dt;
        const bool allow_spawn = effect_allows_spawning(effect);
        for (auto& emitter : effect.emitters) {
            spawn_phase(emitter, dt, allow_spawn);
        }
    }

    // ---- INTEGRATE (parallel over the flattened particle index space) -----
    // Emitters flatten into one global index space through an
    // exclusive-prefix offset table, so chunks are contiguous particle
    // ranges regardless of how particles distribute across emitters. Each
    // particle reads only its own state + the const EmitterDefinition and
    // writes only its own slot — disjoint writes, no partials to merge,
    // bit-identical to serial at any worker count.
    live_emitters_.clear();
    emitter_offsets_.clear();
    std::size_t total_particles = 0;
    for (auto& effect : effects_) {
        for (auto& emitter : effect.emitters) {
            if (emitter.definition != nullptr && !emitter.particles.empty()) {
                live_emitters_.push_back(&emitter);
                emitter_offsets_.push_back(total_particles);
                total_particles += emitter.particles.size();
            }
        }
    }

    core::jobs::parallel_for(
        js, 0, total_particles,
        [&](std::size_t chunk_begin, std::size_t chunk_end) {
            // Locate the emitter containing chunk_begin, then walk forward
            // across emitter boundaries within the chunk.
            std::size_t ei = static_cast<std::size_t>(
                                 std::upper_bound(emitter_offsets_.begin(), emitter_offsets_.end(),
                                                  chunk_begin) -
                                 emitter_offsets_.begin()) -
                             1;
            std::size_t index = chunk_begin;
            while (index < chunk_end) {
                EmitterInstance& emitter = *live_emitters_[ei];
                const EmitterDefinition& def = *emitter.definition;
                const std::size_t base = emitter_offsets_[ei];
                const std::size_t stop = std::min(chunk_end, base + emitter.particles.size());
                for (std::size_t i = index; i < stop; ++i) {
                    Particle& particle = emitter.particles[i - base];
                    update_particle(particle, def, dt);
                    if constexpr (HasTerrain) {
                        // Worker-side call: get_terrain_height must be
                        // thread-safe — see the update() contract.
                        const float terrain_h =
                            get_terrain_height(particle.position.x, particle.position.z);
                        if (particle.position.y < terrain_h) {
                            particle.position.y = terrain_h;
                        }
                    }
                }
                index = stop;
                ++ei;
            }
        },
        INTEGRATE_GRAIN);

    // ---- COMPACT (parallel over emitters) ---------------------------------
    // Each chunk erases dead particles from ITS OWN emitters' vectors —
    // disjoint containers, and std::remove_if is stable, so survivors keep
    // their serial-order positions regardless of chunk completion order.
    core::jobs::parallel_for(
        js, 0, live_emitters_.size(),
        [&](std::size_t chunk_begin, std::size_t chunk_end) {
            for (std::size_t ei = chunk_begin; ei < chunk_end; ++ei) {
                auto& particles = live_emitters_[ei]->particles;
                particles.erase(std::remove_if(particles.begin(), particles.end(),
                                               [](const Particle& p) { return p.age >= p.lifetime; }),
                                particles.end());
            }
        },
        COMPACT_EMITTER_GRAIN);

    compact_effects();
}

template<typename TerrainHeightFn>
void EffectSystem::update(core::jobs::JobSystem& js, float dt, TerrainHeightFn&& get_terrain_height) {
    auto sampler = std::forward<TerrainHeightFn>(get_terrain_height);
    update_t<decltype(sampler), true>(js, dt, sampler);
}

inline void EffectSystem::update(core::jobs::JobSystem& js, float dt) {
    NoTerrainFn sampler;
    update_t<NoTerrainFn, false>(js, dt, sampler);
}

} // namespace bracken::systems
