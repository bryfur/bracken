#include "bracken/physics/physics_world.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/physics/internal/jolt_job_system.hpp"

#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <vector>

namespace bracken::physics {

namespace {

constexpr const char* k_log_category = "physics";

// ---- Global Jolt registration ---------------------------------------------
// Allocator, RTTI factory, and collision-type registration are process-wide
// Jolt singletons shared by every PhysicsWorld. Registered exactly once
// (std::call_once — thread-safe against concurrent first init of two
// worlds) and deliberately never unregistered: another world may be alive
// or created later, and the factory costs nothing when idle.

void jolt_trace(const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    BRACKEN_LOG_INFO(k_log_category, "[jolt] {}", buf);
}

#ifdef JPH_ENABLE_ASSERTS
bool jolt_assert_failed(const char* expression, const char* message, const char* file,
                        JPH::uint line) {
    BRACKEN_LOG_ERROR(k_log_category, "[jolt] assert failed: {} ({}) at {}:{}", expression,
                      message != nullptr ? message : "", file, line);
    return true; // trigger a breakpoint when a debugger is attached
}
#endif

void ensure_jolt_registered() {
    static std::once_flag s_once;
    std::call_once(s_once, [] {
        JPH::RegisterDefaultAllocator();
        JPH::Trace = jolt_trace;
        JPH_IF_ENABLE_ASSERTS(JPH::AssertFailed = jolt_assert_failed;)
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    });
}

// ---- glm <-> Jolt conversions ----------------------------------------------
// Single-precision build (JPH_DOUBLE_PRECISION off): RVec3 == Vec3.

JPH::Vec3 to_jolt(const glm::vec3& v) { return {v.x, v.y, v.z}; }
JPH::Quat to_jolt(const glm::quat& q) { return {q.x, q.y, q.z, q.w}; }
glm::vec3 to_glm(JPH::Vec3Arg v) { return {v.GetX(), v.GetY(), v.GetZ()}; }
glm::quat to_glm(JPH::QuatArg q) { return {q.GetW(), q.GetX(), q.GetY(), q.GetZ()}; }

glm::mat4 to_glm(const JPH::RMat44& m) {
    glm::mat4 out; // both are column-major
    for (int c = 0; c < 4; ++c) {
        const JPH::Vec4 col = m.GetColumn4(c);
        out[c] = glm::vec4(col.GetX(), col.GetY(), col.GetZ(), col.GetW());
    }
    return out;
}

JPH::BodyID jolt_id(BodyId id) { return JPH::BodyID(id); }

// ---- Collision layers -------------------------------------------------------
// Two object layers with matching broadphase layers: k_static (static
// bodies) and k_moving (dynamic + kinematic). Static never collides with
// static; everything else collides.

namespace layers {
constexpr JPH::ObjectLayer k_static = 0;
constexpr JPH::ObjectLayer k_moving = 1;
} // namespace layers

namespace bp_layers {
constexpr JPH::BroadPhaseLayer k_static{0};
constexpr JPH::BroadPhaseLayer k_moving{1};
constexpr JPH::uint k_count = 2;
} // namespace bp_layers

class BroadPhaseLayerMap final : public JPH::BroadPhaseLayerInterface {
public:
    JPH::uint GetNumBroadPhaseLayers() const override { return bp_layers::k_count; }

    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override {
        return layer == layers::k_static ? bp_layers::k_static : bp_layers::k_moving;
    }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override {
        return layer == bp_layers::k_static ? "static" : "moving";
    }
#endif
};

class ObjectVsBroadPhaseFilter final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer layer, JPH::BroadPhaseLayer bp_layer) const override {
        // Static objects only need to find moving bodies; moving bodies see
        // everything.
        return layer != layers::k_static || bp_layer != bp_layers::k_static;
    }
};

class ObjectLayerPairFilterImpl final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
        return a != layers::k_static || b != layers::k_static;
    }
};

} // namespace

// ---- Impl -------------------------------------------------------------------

struct PhysicsWorld::Impl {
    PhysicsConfig config{};

    // Per-world scratch for Jolt's step-internal allocations (~16 MB, the
    // size Jolt's own samples use).
    JPH::TempAllocatorImpl temp_allocator{16 * 1024 * 1024};

    // Declaration order is load-bearing: PhysicsSystem::Init stores
    // references to the three layer objects, and Update runs on `jobs` — so
    // `physics` is declared last (destroyed first).
    BroadPhaseLayerMap bp_layer_map;
    ObjectVsBroadPhaseFilter object_vs_bp_filter;
    ObjectLayerPairFilterImpl object_pair_filter;
    JoltJobSystem jobs;
    JPH::PhysicsSystem physics;

    float accumulator = 0.0f;
    bool overflow_logged = false;
    bool step_error_logged = false;
    // Set on body add; consumed by update() to (re)build broadphase quality
    // before stepping instead of paying for incremental degradation.
    bool broadphase_dirty = false;

    explicit Impl(core::jobs::JobSystem& pool)
        : jobs(pool, JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers) {}

    BodyId add_body(const BodyDesc& desc, const JPH::Shape* shape) {
        const bool is_static = desc.type == BodyType::Static;
        JPH::EMotionType motion = JPH::EMotionType::Static;
        switch (desc.type) {
            case BodyType::Static: motion = JPH::EMotionType::Static; break;
            case BodyType::Dynamic: motion = JPH::EMotionType::Dynamic; break;
            case BodyType::Kinematic: motion = JPH::EMotionType::Kinematic; break;
        }
        const JPH::ObjectLayer layer = is_static ? layers::k_static : layers::k_moving;

        JPH::BodyCreationSettings settings(shape, to_jolt(desc.position),
                                           to_jolt(glm::normalize(desc.rotation)), motion, layer);
        settings.mFriction = desc.friction;
        settings.mRestitution = desc.restitution;
        if (desc.type == BodyType::Dynamic && desc.mass > 0.0f) {
            // Keep the shape's inertia distribution, scale to the requested
            // mass.
            settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
            settings.mMassPropertiesOverride.mMass = desc.mass;
        }

        const JPH::EActivation activation = (!is_static && desc.start_active)
                                                ? JPH::EActivation::Activate
                                                : JPH::EActivation::DontActivate;
        const JPH::BodyID id = physics.GetBodyInterface().CreateAndAddBody(settings, activation);
        if (id.IsInvalid()) {
            BRACKEN_LOG_ERROR(k_log_category,
                              "add body failed: world is full (max_bodies = {})",
                              config.max_bodies);
            return INVALID_BODY_ID;
        }
        broadphase_dirty = true;
        return id.GetIndexAndSequenceNumber();
    }
};

// ---- PhysicsWorld -------------------------------------------------------------

PhysicsWorld::PhysicsWorld() = default;
PhysicsWorld::~PhysicsWorld() { shutdown(); }
PhysicsWorld::PhysicsWorld(PhysicsWorld&&) noexcept = default;
PhysicsWorld& PhysicsWorld::operator=(PhysicsWorld&&) noexcept = default;

bool PhysicsWorld::init(const PhysicsConfig& config, core::jobs::JobSystem& jobs) {
    if (impl_) {
        BRACKEN_LOG_ERROR(k_log_category, "PhysicsWorld::init called on an initialized world");
        return false;
    }
    if (!jobs.is_initialized()) {
        BRACKEN_LOG_ERROR(k_log_category,
                          "PhysicsWorld::init requires an initialized JobSystem");
        return false;
    }
    if (config.fixed_dt <= 0.0f || config.max_steps_per_update < 1 || config.max_bodies == 0) {
        BRACKEN_LOG_ERROR(k_log_category,
                          "PhysicsWorld::init: invalid config (fixed_dt = {}, "
                          "max_steps_per_update = {}, max_bodies = {})",
                          config.fixed_dt, config.max_steps_per_update, config.max_bodies);
        return false;
    }

    ensure_jolt_registered();

    impl_ = std::make_unique<Impl>(jobs);
    impl_->config = config;
    // num_body_mutexes = 0 -> Jolt picks its default.
    impl_->physics.Init(config.max_bodies, 0, config.max_body_pairs,
                        config.max_contact_constraints, impl_->bp_layer_map,
                        impl_->object_vs_bp_filter, impl_->object_pair_filter);
    impl_->physics.SetGravity(to_jolt(config.gravity));

    BRACKEN_LOG_INFO(k_log_category,
                     "PhysicsWorld initialized (max_bodies = {}, fixed_dt = {:.4f}s, "
                     "{} pool workers)",
                     config.max_bodies, config.fixed_dt, jobs.worker_count());
    return true;
}

void PhysicsWorld::shutdown() {
    if (!impl_) {
        return;
    }
    // Destroy every remaining body: Jolt's BodyManager asserts on leaked
    // bodies at destruction.
    JPH::BodyIDVector bodies;
    impl_->physics.GetBodies(bodies);
    if (!bodies.empty()) {
        JPH::BodyInterface& bi = impl_->physics.GetBodyInterface();
        bi.RemoveBodies(bodies.data(), static_cast<int>(bodies.size()));
        bi.DestroyBodies(bodies.data(), static_cast<int>(bodies.size()));
    }
    impl_.reset();
}

BodyId PhysicsWorld::add_sphere(const BodyDesc& desc, float radius) {
    if (!impl_ || radius <= 0.0f) {
        BRACKEN_LOG_ERROR(k_log_category, "add_sphere: {} (radius = {})",
                          impl_ ? "invalid radius" : "world not initialized", radius);
        return INVALID_BODY_ID;
    }
    return impl_->add_body(desc, new JPH::SphereShape(radius));
}

BodyId PhysicsWorld::add_box(const BodyDesc& desc, const glm::vec3& half_extents) {
    const float min_extent = std::min({half_extents.x, half_extents.y, half_extents.z});
    if (!impl_ || min_extent <= 0.0f) {
        BRACKEN_LOG_ERROR(k_log_category, "add_box: {}",
                          impl_ ? "invalid half extents" : "world not initialized");
        return INVALID_BODY_ID;
    }
    // Jolt requires convex radius <= min half extent; shrink it for thin
    // boxes.
    const float convex_radius = std::min(JPH::cDefaultConvexRadius, 0.5f * min_extent);
    return impl_->add_body(desc, new JPH::BoxShape(to_jolt(half_extents), convex_radius));
}

BodyId PhysicsWorld::add_capsule(const BodyDesc& desc, float half_height_of_cylinder,
                                 float radius) {
    if (!impl_ || half_height_of_cylinder <= 0.0f || radius <= 0.0f) {
        BRACKEN_LOG_ERROR(k_log_category, "add_capsule: {}",
                          impl_ ? "invalid dimensions" : "world not initialized");
        return INVALID_BODY_ID;
    }
    return impl_->add_body(desc, new JPH::CapsuleShape(half_height_of_cylinder, radius));
}

BodyId PhysicsWorld::add_heightfield(const glm::vec3& origin, float world_size,
                                     std::span<const float> samples, std::uint32_t resolution) {
    if (!impl_) {
        BRACKEN_LOG_ERROR(k_log_category, "add_heightfield: world not initialized");
        return INVALID_BODY_ID;
    }
    if (resolution < 2 || world_size <= 0.0f ||
        samples.size() != static_cast<std::size_t>(resolution) * resolution) {
        BRACKEN_LOG_ERROR(k_log_category,
                          "add_heightfield: invalid grid (resolution = {}, samples = {}, "
                          "world_size = {})",
                          resolution, samples.size(), world_size);
        return INVALID_BODY_ID;
    }

    // Jolt's HeightFieldShape stores samples in block_size×block_size blocks
    // (block size 2 here, the default): the sample count must be a multiple
    // of the block size and count/block_size must be >= 2. Pad by
    // duplicating the last row/column — the duplicated strip is flat and
    // extends the surface by less than one cell beyond world_size on +x/+z
    // (documented in the header).
    constexpr std::uint32_t k_block_size = 2;
    std::uint32_t padded = std::max(resolution, 2 * k_block_size);
    if (padded % k_block_size != 0) {
        padded += k_block_size - (padded % k_block_size);
    }

    std::vector<float> padded_storage;
    const float* data = samples.data();
    if (padded != resolution) {
        padded_storage.resize(static_cast<std::size_t>(padded) * padded);
        for (std::uint32_t z = 0; z < padded; ++z) {
            const std::uint32_t src_z = std::min(z, resolution - 1);
            for (std::uint32_t x = 0; x < padded; ++x) {
                const std::uint32_t src_x = std::min(x, resolution - 1);
                padded_storage[static_cast<std::size_t>(z) * padded + x] =
                    samples[static_cast<std::size_t>(src_z) * resolution + src_x];
            }
        }
        data = padded_storage.data();
    }

    // Grid spacing preserves the caller's world_size over the ORIGINAL
    // resolution; samples are world heights (y-scale 1, offset handled by
    // the body transform at `origin`).
    const float cell = world_size / static_cast<float>(resolution - 1);
    JPH::HeightFieldShapeSettings settings(data, JPH::Vec3::sZero(),
                                           JPH::Vec3(cell, 1.0f, cell), padded);
    settings.mBlockSize = k_block_size;

    JPH::ShapeSettings::ShapeResult result = settings.Create();
    if (result.HasError()) {
        BRACKEN_LOG_ERROR(k_log_category, "add_heightfield: shape creation failed: {}",
                          result.GetError().c_str());
        return INVALID_BODY_ID;
    }

    BodyDesc desc;
    desc.type = BodyType::Static;
    desc.position = origin;
    return impl_->add_body(desc, result.Get());
}

void PhysicsWorld::remove_body(BodyId id) {
    if (!impl_ || id == INVALID_BODY_ID) {
        return;
    }
    JPH::BodyInterface& bi = impl_->physics.GetBodyInterface();
    const JPH::BodyID jid = jolt_id(id);
    bi.RemoveBody(jid);
    bi.DestroyBody(jid);
}

void PhysicsWorld::update(float dt) {
    BRACKEN_ASSERT(impl_, "PhysicsWorld::update on uninitialized world");
    if (!impl_ || dt <= 0.0f) {
        return;
    }
    BRACKEN_PROFILE_ZONE("PhysicsWorld::update");
    Impl& im = *impl_;

    if (im.broadphase_dirty) {
        im.physics.OptimizeBroadPhase();
        im.broadphase_dirty = false;
    }

    const float fixed = im.config.fixed_dt;
    im.accumulator += dt;
    int steps = 0;
    while (im.accumulator >= fixed && steps < im.config.max_steps_per_update) {
        const JPH::EPhysicsUpdateError err =
            im.physics.Update(fixed, 1, &im.temp_allocator, &im.jobs);
        if (err != JPH::EPhysicsUpdateError::None && !im.step_error_logged) {
            BRACKEN_LOG_WARN(k_log_category,
                             "Jolt step error {:#x} — raise PhysicsConfig "
                             "max_body_pairs/max_contact_constraints (logged once)",
                             static_cast<std::uint32_t>(err));
            im.step_error_logged = true;
        }
        im.accumulator -= fixed;
        ++steps;
    }

    // Overflow: more sim debt than max_steps_per_update covers. Clamp the
    // remainder (the sim slows down instead of spiraling: stepping more next
    // frame would make the next frame even longer). Logged once per world.
    if (im.accumulator >= fixed) {
        if (!im.overflow_logged) {
            BRACKEN_LOG_WARN(k_log_category,
                             "update overflow: dropping {:.3f}s of sim debt "
                             "(max_steps_per_update = {}; logged once)",
                             im.accumulator - fixed, im.config.max_steps_per_update);
            im.overflow_logged = true;
        }
        im.accumulator = std::fmod(im.accumulator, fixed);
    }
}

glm::vec3 PhysicsWorld::position(BodyId id) const {
    if (!impl_ || id == INVALID_BODY_ID) {
        return glm::vec3(0);
    }
    return to_glm(impl_->physics.GetBodyInterface().GetPosition(jolt_id(id)));
}

glm::quat PhysicsWorld::rotation(BodyId id) const {
    if (!impl_ || id == INVALID_BODY_ID) {
        return glm::quat(1, 0, 0, 0);
    }
    return to_glm(impl_->physics.GetBodyInterface().GetRotation(jolt_id(id)));
}

glm::mat4 PhysicsWorld::transform(BodyId id) const {
    if (!impl_ || id == INVALID_BODY_ID) {
        return glm::mat4(1.0f);
    }
    return to_glm(impl_->physics.GetBodyInterface().GetWorldTransform(jolt_id(id)));
}

void PhysicsWorld::set_position(BodyId id, const glm::vec3& position, bool activate) {
    if (!impl_ || id == INVALID_BODY_ID) {
        return;
    }
    impl_->physics.GetBodyInterface().SetPosition(
        jolt_id(id), to_jolt(position),
        activate ? JPH::EActivation::Activate : JPH::EActivation::DontActivate);
}

glm::vec3 PhysicsWorld::linear_velocity(BodyId id) const {
    if (!impl_ || id == INVALID_BODY_ID) {
        return glm::vec3(0);
    }
    return to_glm(impl_->physics.GetBodyInterface().GetLinearVelocity(jolt_id(id)));
}

void PhysicsWorld::set_linear_velocity(BodyId id, const glm::vec3& velocity) {
    if (!impl_ || id == INVALID_BODY_ID) {
        return;
    }
    // Jolt activates a sleeping body itself when the new velocity is
    // nonzero (BodyInterface::SetLinearVelocity).
    impl_->physics.GetBodyInterface().SetLinearVelocity(jolt_id(id), to_jolt(velocity));
}

void PhysicsWorld::apply_impulse(BodyId id, const glm::vec3& impulse) {
    if (!impl_ || id == INVALID_BODY_ID) {
        return;
    }
    // AddImpulse activates the body itself.
    impl_->physics.GetBodyInterface().AddImpulse(jolt_id(id), to_jolt(impulse));
}

void PhysicsWorld::apply_impulse_at(BodyId id, const glm::vec3& impulse,
                                    const glm::vec3& world_point) {
    if (!impl_ || id == INVALID_BODY_ID) {
        return;
    }
    impl_->physics.GetBodyInterface().AddImpulse(jolt_id(id), to_jolt(impulse),
                                                 to_jolt(world_point));
}

void PhysicsWorld::move_kinematic(BodyId id, const glm::vec3& target_pos,
                                  const glm::quat& target_rot, float dt) {
    if (!impl_ || id == INVALID_BODY_ID || dt <= 0.0f) {
        return;
    }
    impl_->physics.GetBodyInterface().MoveKinematic(
        jolt_id(id), to_jolt(target_pos), to_jolt(glm::normalize(target_rot)), dt);
}

bool PhysicsWorld::is_active(BodyId id) const {
    if (!impl_ || id == INVALID_BODY_ID) {
        return false;
    }
    return impl_->physics.GetBodyInterface().IsActive(jolt_id(id));
}

PhysicsWorld::RayHit PhysicsWorld::cast_ray(const glm::vec3& origin,
                                            const glm::vec3& direction_and_length) const {
    RayHit out;
    if (!impl_) {
        return out;
    }
    const JPH::RRayCast ray{to_jolt(origin), to_jolt(direction_and_length)};
    JPH::RayCastResult hit;
    if (!impl_->physics.GetNarrowPhaseQuery().CastRay(ray, hit)) {
        return out;
    }
    out.hit = true;
    out.fraction = hit.mFraction;
    const JPH::RVec3 point = ray.GetPointOnRay(hit.mFraction);
    out.position = to_glm(point);
    out.body = hit.mBodyID.GetIndexAndSequenceNumber();
    JPH::BodyLockRead lock(impl_->physics.GetBodyLockInterface(), hit.mBodyID);
    if (lock.Succeeded()) {
        out.normal = to_glm(lock.GetBody().GetWorldSpaceSurfaceNormal(hit.mSubShapeID2, point));
    }
    return out;
}

std::size_t PhysicsWorld::collide_capsule(const glm::vec3& center, float half_height_of_cylinder,
                                          float radius, std::span<ShapeContact> out_contacts,
                                          bool include_moving, BodyId ignore_body) const {
    if (!impl_ || out_contacts.empty()) {
        return 0;
    }
    if (half_height_of_cylinder <= 0.0f || radius <= 0.0f) {
        BRACKEN_LOG_ERROR(k_log_category, "collide_capsule: invalid dimensions");
        return 0;
    }

    // Jolt shapes are refcounted — heap-allocate and hold a Ref for the
    // query's duration (CollideShape internals may take temporary refs).
    const JPH::Ref<JPH::Shape> shape = new JPH::CapsuleShape(half_height_of_cylinder, radius);

    const JPH::CollideShapeSettings settings;
    JPH::AllHitCollisionCollector<JPH::CollideShapeCollector> collector;
    const JPH::SpecifiedBroadPhaseLayerFilter static_bp{bp_layers::k_static};
    const JPH::SpecifiedObjectLayerFilter static_obj{layers::k_static};
    const JPH::BroadPhaseLayerFilter any_bp; // base classes pass everything
    const JPH::ObjectLayerFilter any_obj;
    // BodyID equality never matches for INVALID_BODY_ID, so "no ignore" needs
    // no special case.
    const JPH::IgnoreSingleBodyFilter body_filter{jolt_id(ignore_body)};

    impl_->physics.GetNarrowPhaseQuery().CollideShape(
        shape, JPH::Vec3::sReplicate(1.0f), JPH::RMat44::sTranslation(to_jolt(center)), settings,
        JPH::RVec3::sZero(), collector,
        include_moving ? static_cast<const JPH::BroadPhaseLayerFilter&>(any_bp) : static_bp,
        include_moving ? static_cast<const JPH::ObjectLayerFilter&>(any_obj) : static_obj,
        body_filter);

    // Deepest-first with a BodyId tiebreak: deterministic regardless of
    // broadphase traversal order.
    std::sort(collector.mHits.begin(), collector.mHits.end(),
              [](const JPH::CollideShapeResult& a, const JPH::CollideShapeResult& b) {
                  if (a.mPenetrationDepth != b.mPenetrationDepth) {
                      return a.mPenetrationDepth > b.mPenetrationDepth;
                  }
                  return a.mBodyID2.GetIndexAndSequenceNumber() <
                         b.mBodyID2.GetIndexAndSequenceNumber();
              });

    const std::size_t count = std::min(out_contacts.size(), collector.mHits.size());
    for (std::size_t i = 0; i < count; ++i) {
        const JPH::CollideShapeResult& hit = collector.mHits[i];
        // mPenetrationAxis pushes shape 2 (the hit body) out of the overlap;
        // the queried capsule is shape 1, so its push direction is the
        // negation.
        out_contacts[i].normal = to_glm(-hit.mPenetrationAxis.Normalized());
        out_contacts[i].penetration = hit.mPenetrationDepth;
        out_contacts[i].body = hit.mBodyID2.GetIndexAndSequenceNumber();
    }
    return count;
}

} // namespace bracken::physics
