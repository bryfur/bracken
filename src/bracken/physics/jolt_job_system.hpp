#pragma once

// INTERNAL header — includes Jolt. Only src/bracken/physics/*.cpp may
// include it; the public bracken::physics surface (physics_world.hpp) stays
// Jolt-free by contract (Jolt is linked PRIVATE into the engine).

// Jolt v5.2.0's Core.h names std traits (is_trivial et al.) without
// including <type_traits> itself, relying on transitive std-header leakage
// that libc++ ≥ 22 pruned (Jolt's own build hides this behind its PCH).
// Provide the std headers it assumes BEFORE the Jolt umbrella.
#include <cstdio>
#include <type_traits>
#include <utility>

#include <Jolt/Jolt.h>

#include <Jolt/Core/FixedSizeFreeList.h>
#include <Jolt/Core/JobSystemWithBarrier.h>

#include "bracken/core/jobs/task_group.hpp"

namespace bracken::core::jobs {
class JobSystem;
}

namespace bracken::physics {

// JoltJobSystem — bridges Jolt's job graph onto the engine's
// core::jobs::JobSystem so ONE pool owns the cores
// (docs/multithreading_plan.md §7: no second thread pool, no
// oversubscription — Jolt's own JobSystemThreadPool would add N more
// threads competing with the engine's workers).
//
// Structure mirrors JPH::JobSystemThreadPool: pooled Job storage in a
// lock-free FixedSizeFreeList (CreateJob constructs into it, FreeJob
// destructs), and QueueJob/QueueJobs hand each ready job to the bracken
// pool as a thunk that runs `job->Execute(); job->Release();` — exactly the
// thread-pool worker loop's reference discipline: the queue path owns one
// reference (AddRef at queue time, Release after Execute). Barriers come
// from JPH::JobSystemWithBarrier, whose WaitForJobs *also executes ready
// barrier jobs on the waiting thread* (Job::Execute is internally guarded
// by a dependency-count CAS, so the pool thunk for an already-executed job
// degenerates to a reference drop). Two consequences:
//   - a PhysicsWorld::update() running as a sim job makes progress even if
//     every bracken worker is busy or the pool has one worker;
//   - nested fork-join is safe: bracken's helping wait() lets workers
//     blocked in joins execute queued physics thunks instead of sleeping.
//
// Threading tier: worker-safe — CreateJob/QueueJob(s)/FreeJob are called by
// Jolt from any thread mid-step (jobs spawn jobs); FixedSizeFreeList and
// the bracken submit paths are lock-free.
class JoltJobSystem final : public JPH::JobSystemWithBarrier {
public:
    // `pool` must be initialized and must outlive this object.
    JoltJobSystem(core::jobs::JobSystem& pool, JPH::uint max_jobs, JPH::uint max_barriers);
    ~JoltJobSystem() override;

    // Jolt sizes its per-step job splits from this: the bracken worker
    // count (>= 1).
    int GetMaxConcurrency() const override;

    JobHandle CreateJob(const char* name, JPH::ColorArg color, const JobFunction& job_function,
                        JPH::uint32 num_dependencies = 0) override;

protected:
    void QueueJob(Job* job) override;
    void QueueJobs(Job** jobs, JPH::uint num_jobs) override;
    void FreeJob(Job* job) override;

private:
    void submit_(Job* job);

    core::jobs::JobSystem& pool_;

    // group_ — quiescence tracking for the bracken-side thunks only (Jolt
    // tracks *job* completion through its own barriers; a thunk can still
    // be in flight for an instant after WaitForJobs returns, when the
    // barrier waiter executed the job itself). The destructor joins it with
    // a helping wait so pooled Job storage never outlives an in-flight
    // thunk. Never waited elsewhere; thunks cannot throw (Jolt is built
    // without exceptions), so no error ever accumulates in it.
    core::jobs::TaskGroup group_;

    using AvailableJobs = JPH::FixedSizeFreeList<Job>;
    AvailableJobs jobs_;
};

} // namespace bracken::physics
