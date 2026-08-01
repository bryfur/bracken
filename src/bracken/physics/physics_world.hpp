#pragma once

// bracken::physics — rigid-body physics, backed by Jolt Physics.
//
// This header is deliberately Jolt-free: the public surface is glm + POD
// types over a pimpl (`Impl` in physics_world.cpp). Jolt is linked PRIVATE
// into the engine and must never leak into a public bracken header.
//
// Thread-safety tier: **owned / sim-stage** (architecture.md "Threading &
// lifetime contracts"). A PhysicsWorld has a single logical owner — the
// host's simulation code — and is intended to be driven from inside the sim
// stage (`on_simulate`, or any job it fans out that the host serializes
// against). `update()` never touches the GPU, SDL, or any main-thread-tier
// API, so calling it from a pool job is the designed use. It is NOT
// internally synchronized against concurrent host calls: do not call
// update()/add_*/remove_body from two threads at once. Jolt itself
// parallelizes each step internally across the shared engine JobSystem (see
// jolt_job_system.hpp — one pool owns the cores).

#include <cstdint>
#include <memory>
#include <span>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace bracken::core::jobs {
class JobSystem;
}

namespace bracken::physics {

enum class BodyType { Static, Dynamic, Kinematic };

// Opaque body identifier (a Jolt BodyID's raw 32-bit value: index + sequence
// number, so stale ids of destroyed bodies do not alias newly created ones).
using BodyId = std::uint32_t;
inline constexpr BodyId INVALID_BODY_ID = 0xffffffffu;

struct BodyDesc {
    BodyType type = BodyType::Static;
    glm::vec3 position{0};
    glm::quat rotation{1, 0, 0, 0};
    float mass = 1.0f;      // dynamic bodies only (> 0); ignored for static/kinematic
    float friction = 0.5f;
    float restitution = 0.0f;
    bool start_active = true; // ignored for static bodies
};

struct PhysicsConfig {
    glm::vec3 gravity{0.0f, -9.81f, 0.0f};
    std::uint32_t max_bodies = 4096;
    std::uint32_t max_body_pairs = 4096;
    std::uint32_t max_contact_constraints = 2048;
    float fixed_dt = 1.0f / 60.0f;
    int max_steps_per_update = 4;
};

// A rigid-body simulation world. Fixed-timestep (accumulator in update()),
// two collision layers (static / moving — kinematic bodies count as moving),
// deterministic: identical setup + identical update() sequence produces
// bit-identical body transforms, independent of worker count (a Jolt
// guarantee this engine's tests pin down).
class PhysicsWorld {
public:
    PhysicsWorld();
    ~PhysicsWorld(); // RAII — shuts down if still initialized

    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;
    PhysicsWorld(PhysicsWorld&&) noexcept;
    PhysicsWorld& operator=(PhysicsWorld&&) noexcept;

    // Creates the Jolt world. `jobs` must be initialized and must outlive
    // this world — physics steps run as jobs on that pool (never a second
    // thread pool). Registers Jolt's global type factory on first use
    // (process-wide, thread-safe, never unregistered). Returns false (and
    // logs) on misconfiguration or double init.
    [[nodiscard]] bool init(const PhysicsConfig& config, core::jobs::JobSystem& jobs);

    // Destroys all remaining bodies and tears the world down. Safe to call
    // when not initialized. All BodyIds from this world become invalid.
    void shutdown();

    bool is_initialized() const noexcept { return impl_ != nullptr; }

    // ---- Body creation / destruction -------------------------------------
    // All add_* return INVALID_BODY_ID (and log) on failure (world full,
    // invalid shape parameters, uninitialized world).

    BodyId add_sphere(const BodyDesc& desc, float radius);
    BodyId add_box(const BodyDesc& desc, const glm::vec3& half_extents);
    // Total capsule half-height = half_height_of_cylinder + radius.
    BodyId add_capsule(const BodyDesc& desc, float half_height_of_cylinder, float radius);

    // Static terrain from a square height grid. `samples` holds
    // resolution×resolution world-space heights, row-major (index =
    // z * resolution + x); sample (x,z) sits at world position
    // origin + (x*cell, samples[z*resolution+x], z*cell) with
    // cell = world_size / (resolution - 1). resolution must be >= 2 and
    // samples.size() must equal resolution². Jolt constraint: the sample
    // count must be a multiple of its block size (2) and at least twice it —
    // grids that don't comply are padded internally by duplicating the last
    // row/column, which extends the collision surface slightly (< one cell)
    // beyond world_size on +x/+z. Pass an even resolution >= 4 to avoid any
    // padding.
    BodyId add_heightfield(const glm::vec3& origin, float world_size,
                           std::span<const float> samples, std::uint32_t resolution);

    void remove_body(BodyId id);

    // ---- Simulation -------------------------------------------------------

    // Fixed-step accumulator: consumes `dt` in fixed_dt slices, at most
    // max_steps_per_update per call. On overflow (dt debt beyond the cap)
    // the remainder is clamped away — the sim slows down instead of
    // spiraling — and a warning is logged once per world. Tier: sim-stage —
    // no GPU/SDL/main-thread coupling; steps fan out on the engine JobSystem
    // passed to init().
    void update(float dt);

    // ---- State access ------------------------------------------------------
    // Queries on INVALID_BODY_ID / destroyed bodies return identity values.

    glm::vec3 position(BodyId id) const;
    glm::quat rotation(BodyId id) const;
    glm::mat4 transform(BodyId id) const;
    void set_position(BodyId id, const glm::vec3& position, bool activate);

    glm::vec3 linear_velocity(BodyId id) const;
    void set_linear_velocity(BodyId id, const glm::vec3& velocity);

    void apply_impulse(BodyId id, const glm::vec3& impulse);
    void apply_impulse_at(BodyId id, const glm::vec3& impulse, const glm::vec3& world_point);

    // Kinematic bodies: velocity-drives the body so it reaches the target
    // transform in `dt` seconds (pushing dynamic bodies along the way).
    void move_kinematic(BodyId id, const glm::vec3& target_pos, const glm::quat& target_rot,
                        float dt);

    bool is_active(BodyId id) const;

    // ---- Queries ------------------------------------------------------------

    struct RayHit {
        bool hit = false;
        float fraction = 0;   // hit point = origin + fraction * direction_and_length
        glm::vec3 position{0};
        glm::vec3 normal{0};
        BodyId body = INVALID_BODY_ID;
    };

    // Casts a ray of length |direction_and_length| from origin; returns the
    // closest hit against any body.
    //
    // `ignore_body` skips one body. A character controller needs this for its
    // own ground probe: a character usually has a kinematic body tracking it,
    // and without the filter the downward ray hits that body's own capsule
    // instead of the floor — which reads as the character standing on itself
    // and levitating a little further every frame.
    RayHit cast_ray(const glm::vec3& origin, const glm::vec3& direction_and_length,
                    BodyId ignore_body = INVALID_BODY_ID) const;

    struct ShapeContact {
        glm::vec3 normal{0};   // unit push direction for the queried shape (world space)
        float penetration = 0; // move the shape this far along normal to separate
        BodyId body = INVALID_BODY_ID;
    };

    // Overlap-tests an upright capsule (center = capsule center; total
    // half-height = half_height_of_cylinder + radius) and reports one contact
    // per overlapping body, deepest first (ties broken by BodyId — output is
    // deterministic and truncation keeps the contacts that matter). By
    // default only static bodies are tested — the character-controller case:
    // walls and props block, dynamic toys are handled by the sim.
    // include_moving adds dynamic/kinematic bodies; ignore_body skips one
    // body (a terrain heightfield the caller resolves separately, or the
    // caller's own body when include_moving). Writes at most
    // out_contacts.size() contacts and returns the count written; returns 0
    // on an uninitialized world or non-positive dimensions.
    [[nodiscard]] std::size_t collide_capsule(const glm::vec3& center,
                                              float half_height_of_cylinder, float radius,
                                              std::span<ShapeContact> out_contacts,
                                              bool include_moving = false,
                                              BodyId ignore_body = INVALID_BODY_ID) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bracken::physics
