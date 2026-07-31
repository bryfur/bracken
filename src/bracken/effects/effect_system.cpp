#include "effect_system.hpp"
#include "bracken/core/assert.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/scene/frame_builder.hpp"
#include <algorithm>
#include <cmath>
#include <glm/gtc/constants.hpp>
#include <random>

namespace bracken::systems {

using bracken::SpawnMode;
using bracken::VelocityType;

namespace {

// SplitMix64. The particle streams must be reproducible run-to-run AND
// independent of which worker happens to run the sim job, so the generator
// state lives on the emitter that draws from it rather than in a thread_local:
// a thread_local mt19937 seeded from std::random_device gave a different
// stream on every run and interleaved differently at different worker counts,
// which put particle positions — and therefore the sealed packet — outside any
// regression gate. Advancing a per-emitter counter makes each draw a pure
// function of (emitter seed, draw index).
uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

float random_float(uint64_t& rng, float min, float max) {
    // Top 24 bits → [0,1): exactly representable in a float mantissa, so the
    // mapping carries no rounding surprise across platforms.
    const float unit = static_cast<float>(splitmix64(rng) >> 40) / 16777216.0f;
    return min + unit * (max - min);
}

} // anonymous namespace

int EffectSystem::spawn_effect(const EffectDefinition* definition, const glm::vec3& position,
                               const glm::vec3& direction) {
    BRACKEN_PROFILE_ZONE_COLOR("EffectSystem::spawn_effect", 0x3D6E9E);
    if (!definition) {
        return -1;
    }

    EffectInstance effect;
    effect.definition = definition;
    effect.age = 0.0f;

    // Create emitter instances
    for (const auto& emitter_def : definition->emitters) {
        // Definition intake is the engine's earliest validation boundary
        // (parsing effect files is the host's job — see docs/effects_system.md):
        // a CONTINUOUS spawn_rate that is zero, negative, or non-finite would
        // otherwise hang update()'s catch-up loop. The emitter is kept but
        // stays inert — spawn_phase skips continuous spawning for it.
        if (emitter_def.spawn_mode == SpawnMode::CONTINUOUS &&
            !(std::isfinite(emitter_def.spawn_rate) && emitter_def.spawn_rate > 0.0f)) {
            if (warned_bad_spawn_rate_.insert(&emitter_def).second) {
                BRACKEN_LOG_WARN("effect",
                                 "Effect '{}' emitter '{}': CONTINUOUS spawn_rate {} is invalid "
                                 "(must be > 0 and finite); emitter will not spawn",
                                 definition->name, emitter_def.name, emitter_def.spawn_rate);
            }
        }

        EmitterInstance emitter;
        emitter.definition = &emitter_def;
        emitter.origin = position;
        emitter.direction = direction;
        emitter.age = 0.0f;
        emitter.next_spawn_time = emitter_def.delay;
        emitter.has_spawned_burst = false;
        // Deterministic seed: spawn order is itself deterministic under a
        // fixed timestep, so keying on the monotonic spawn counter gives every
        // emitter its own reproducible stream. The odd multiplier decorrelates
        // adjacent emitters of the same effect.
        emitter.rng = rng_seed_counter_++ * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL;

        effect.emitters.push_back(std::move(emitter));
    }

    effects_.push_back(std::move(effect));
    return static_cast<int>(effects_.size() - 1);
}

void EffectSystem::spawn_phase(EmitterInstance& emitter, float dt, bool allow_spawn) {
    if (!emitter.definition) {
        return;
    }

    const auto& def = *emitter.definition;
    emitter.age += dt;

    if (!allow_spawn || !emitter.is_active()) {
        return;
    }

    if (def.spawn_mode == SpawnMode::BURST && !emitter.has_spawned_burst) {
        if (emitter.age >= def.delay) {
            spawn_particles(emitter, def.spawn_count);
            emitter.has_spawned_burst = true;
        }
    } else if (def.spawn_mode == SpawnMode::CONTINUOUS) {
        // Interval is hoisted and validated: the catch-up loop below only
        // terminates for a positive, finite interval. rate <= 0 gives a
        // negative interval (a negative rate used to hang this loop with
        // unbounded particle growth), rate == 0 gives +inf (used to emit
        // a single stray particle), rate == +inf gives 0 (would spin
        // forever), NaN fails every comparison. spawn_effect() already
        // warns about such definitions; this guard keeps update() robust
        // regardless.
        const float spawn_interval = 1.0f / def.spawn_rate;
        if (spawn_interval > 0.0f && std::isfinite(spawn_interval)) {
            while (emitter.age >= emitter.next_spawn_time) {
                spawn_particles(emitter, 1);
                emitter.next_spawn_time += spawn_interval;
            }
        }
    }
}

void EffectSystem::compact_effects() {
    BRACKEN_PROFILE_ZONE_COLOR("EffectSystem::compact_effects", 0x3D6E9E);
    effects_.erase(
        std::remove_if(effects_.begin(), effects_.end(), [](const EffectInstance& e) { return e.is_complete(); }),
        effects_.end());
    // Both update() overloads (header-inline) end here, so this samples the
    // live particle count once per update.
    BRACKEN_PROFILE_PLOT("effects.particles", static_cast<int64_t>(particle_count()));
}

void EffectSystem::spawn_particles(EmitterInstance& emitter, int count) {
    BRACKEN_PROFILE_ZONE_COLOR("EffectSystem::spawn_particles", 0x3D6E9E);
    if (!emitter.definition) {
        return;
    }

    const auto& def = *emitter.definition;

    for (int i = 0; i < count; ++i) {
        Particle particle;

        // Position: emitter origin, jittered within the spawn volume when one
        // is set (the zero-volume guard keeps point emitters' RNG sequences —
        // and therefore existing effects — bit-identical).
        particle.position = emitter.origin;
        if (def.spawn_volume.x > 0.0f || def.spawn_volume.y > 0.0f || def.spawn_volume.z > 0.0f) {
            particle.position += glm::vec3(random_float(emitter.rng, -def.spawn_volume.x, def.spawn_volume.x),
                                           random_float(emitter.rng, -def.spawn_volume.y, def.spawn_volume.y),
                                           random_float(emitter.rng, -def.spawn_volume.z, def.spawn_volume.z));
        }

        // Lifetime
        particle.lifetime = def.particle_lifetime;
        particle.age = 0.0f;

        // Initial velocity
        particle.velocity = calculate_initial_velocity(def.velocity, emitter.direction, emitter.rng);

        // Initial rotation
        particle.rotation = glm::radians(def.rotation.initial_rotation);
        particle.rotation_rate = glm::radians(def.rotation.rotation_rate);

        // Initial appearance
        particle.scale = def.appearance.scale_over_lifetime.evaluate(0.0f);
        particle.opacity = def.appearance.opacity_over_lifetime.evaluate(0.0f);
        particle.color = def.appearance.color_tint;

        // For orbital particles, store index and initial angle
        if (def.velocity.type == VelocityType::ORBITAL) {
            particle.orbit_index = static_cast<int>(emitter.particles.size());
            particle.orbit_angle = (glm::two_pi<float>() / def.spawn_count) * particle.orbit_index;
            particle.orbit_origin = emitter.origin;
        }

        // For arc particles, store origin and direction
        if (def.velocity.type == VelocityType::ARC) {
            particle.arc_origin = emitter.origin;
            particle.arc_direction = emitter.direction;
        }

        emitter.particles.push_back(std::move(particle));
    }
}

glm::vec3 EffectSystem::calculate_initial_velocity(const VelocityDefinition& vel_def, const glm::vec3& direction,
                                                   uint64_t& rng) {
    glm::vec3 velocity = {0, 0, 0};

    switch (vel_def.type) {
        case VelocityType::DIRECTIONAL: {
            velocity = glm::normalize(direction) * vel_def.speed;

            // Apply spread angle
            if (vel_def.spread_angle > 0) {
                float spread_rad = glm::radians(vel_def.spread_angle);
                float angle_offset = random_float(rng, -spread_rad, spread_rad);

                // Rotate velocity by spread angle (simplified 2D rotation in XZ plane)
                float cos_a = std::cos(angle_offset);
                float sin_a = std::sin(angle_offset);
                velocity = glm::vec3(velocity.x * cos_a - velocity.z * sin_a, velocity.y,
                                     velocity.x * sin_a + velocity.z * cos_a);
            }
            break;
        }

        case VelocityType::RADIAL: {
            // Random direction outward
            float angle = random_float(rng, 0.0f, glm::two_pi<float>());
            velocity = glm::vec3(std::cos(angle) * vel_def.speed, 0, std::sin(angle) * vel_def.speed);
            break;
        }

        case VelocityType::ORBITAL: {
            // Orbital particles don't use initial velocity, they calculate position each frame
            velocity = {0, 0, 0};
            break;
        }

        case VelocityType::ARC: {
            // Arc particles don't use velocity, they calculate position based on progress
            velocity = {0, 0, 0};
            break;
        }

        case VelocityType::CUSTOM: {
            velocity = vel_def.direction * vel_def.speed;
            break;
        }
    }

    return velocity;
}

void EffectSystem::update_particle(Particle& particle, const EmitterDefinition& emitter_def, float dt) {
    particle.age += dt;
    float t = particle.age / particle.lifetime; // Normalized lifetime (0 to 1)

    const auto& vel_def = emitter_def.velocity;

    // Update position based on velocity type
    if (vel_def.type == VelocityType::ARC) {
        // Arc motion - replicate old draw_melee_slash behavior
        // Calculate base angle from direction
        float base_angle = std::atan2(particle.arc_direction.x, particle.arc_direction.z);

        // Swing angle goes from -1 to +1 radians over the lifetime
        float swing_angle = -1.0f + t * 2.0f;
        float rotation = base_angle + swing_angle;

        // Calculate position along the arc
        float arc_radius = vel_def.arc_radius;
        particle.position.x = particle.arc_origin.x + std::sin(rotation) * arc_radius;
        particle.position.z = particle.arc_origin.z + std::cos(rotation) * arc_radius;

        // Height oscillates with a sine wave
        particle.position.y = particle.arc_origin.y + vel_def.arc_height_base +
                              std::sin(t * glm::pi<float>()) * vel_def.arc_height_amplitude;

        // Tilt rotates with progress
        float tilt = std::sin(t * glm::pi<float>()) * vel_def.arc_tilt_amplitude;
        particle.rotation.x = tilt;

        // Face the direction of the arc
        particle.rotation.y = rotation + glm::half_pi<float>(); // +90 degrees
        particle.rotation.z = -0.5f;                            // Fixed roll

    } else if (vel_def.type == VelocityType::ORBITAL) {
        // Orbital motion - calculate position from orbit parameters
        particle.orbit_angle += vel_def.orbit_speed * glm::two_pi<float>() * dt;

        // Calculate position relative to orbit origin
        float radius = vel_def.orbit_radius;
        particle.position.x = particle.orbit_origin.x + std::cos(particle.orbit_angle) * radius;
        particle.position.z = particle.orbit_origin.z + std::sin(particle.orbit_angle) * radius;

        // Height with base offset and variation
        particle.position.y = particle.orbit_origin.y + vel_def.orbit_height_base;
        if (vel_def.height_variation > 0) {
            particle.position.y += std::sin(particle.orbit_angle * 3.0f) * vel_def.height_variation;
        }
    } else {
        // Standard velocity-based motion
        particle.position += particle.velocity * dt;

        // Apply gravity (squared-magnitude test: dot > 0 iff length > 0,
        // without the per-particle sqrt on an emitter constant)
        if (glm::dot(vel_def.gravity, vel_def.gravity) > 0.0f) {
            particle.velocity += vel_def.gravity * dt;
        }

        // Apply drag
        if (vel_def.drag > 0) {
            particle.velocity *= (1.0f - vel_def.drag * dt);
        }
    }

    // Update rotation (squared-magnitude test, 0.01^2 = 1e-4; atan2 is
    // scale-invariant so the old normalize was a redundant second sqrt)
    if (emitter_def.rotation.face_velocity && glm::dot(particle.velocity, particle.velocity) > 1e-4f) {
        // Face the direction of movement
        particle.rotation.y = std::atan2(particle.velocity.x, particle.velocity.z);
    } else {
        // Apply rotation rate
        particle.rotation += particle.rotation_rate * dt;
    }

    // Update appearance based on lifetime curves
    particle.scale = emitter_def.appearance.scale_over_lifetime.evaluate(t);
    particle.opacity = emitter_def.appearance.opacity_over_lifetime.evaluate(t);

    // Color gradient
    if (emitter_def.appearance.use_color_gradient) {
        particle.color = glm::mix(emitter_def.appearance.color_tint, emitter_def.appearance.color_end, t);
    }
}

size_t EffectSystem::particle_count() const {
    size_t count = 0;
    for (const auto& effect : effects_) {
        for (const auto& emitter : effect.emitters) {
            count += emitter.particles.size();
        }
    }
    return count;
}

void EffectSystem::snapshot_into(scene::FrameBuilder& fb) const {
    BRACKEN_PROFILE_ZONE_COLOR("EffectSystem::snapshot_into", 0x3D6E9E);

    // Count live emitters (definition + at least one particle — the same
    // filter the retired render-side walk applied) and their particles.
    uint32_t emitter_total = 0;
    uint32_t particle_total = 0;
    for (const auto& effect : effects_) {
        for (const auto& emitter : effect.emitters) {
            if (emitter.definition != nullptr && !emitter.particles.empty()) {
                ++emitter_total;
                particle_total += static_cast<uint32_t>(emitter.particles.size());
            }
        }
    }

    auto alloc = fb.alloc_effect_snapshot(emitter_total, particle_total);

    // Fill in deterministic (effect, emitter, particle) submission order —
    // the order the retired renderer walk used, so batches are stable.
    uint32_t e = 0;
    uint32_t p = 0;
    for (const auto& effect : effects_) {
        for (const auto& emitter : effect.emitters) {
            if (emitter.definition == nullptr || emitter.particles.empty()) {
                continue;
            }
            alloc.emitters[e++] = {emitter.definition, p, static_cast<uint32_t>(emitter.particles.size())};
            for (const auto& particle : emitter.particles) {
                alloc.particles[p++] = {particle.position, particle.scale, particle.rotation, particle.opacity,
                                        particle.color};
            }
        }
    }
    BRACKEN_ASSERT(e == emitter_total && p == particle_total, "effect snapshot count drift ({}≠{} or {}≠{})", e,
                   emitter_total, p, particle_total);
}

} // namespace bracken::systems
