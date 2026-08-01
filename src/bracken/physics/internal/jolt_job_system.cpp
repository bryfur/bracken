#include "bracken/physics/internal/jolt_job_system.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_system.hpp"

#include <chrono>
#include <thread>

namespace bracken::physics {

JoltJobSystem::JoltJobSystem(core::jobs::JobSystem& pool, JPH::uint max_jobs,
                             JPH::uint max_barriers)
    : JPH::JobSystemWithBarrier(max_barriers), pool_(pool) {
    BRACKEN_ASSERT(pool.is_initialized(),
                   "JoltJobSystem requires an initialized bracken JobSystem");
    jobs_.Init(max_jobs, max_jobs);
}

JoltJobSystem::~JoltJobSystem() {
    // Quiesce the bracken-side thunks before the FixedSizeFreeList (which
    // asserts all objects freed) is destroyed: each queued thunk holds one
    // Job reference whose Release may be the one that calls FreeJob.
    // Helping wait — safe even when destruction happens on a pool worker
    // (the caller executes remaining thunks itself; 1-worker pools
    // included).
    pool_.wait(group_);
}

int JoltJobSystem::GetMaxConcurrency() const {
    const unsigned n = pool_.worker_count();
    return n > 0 ? static_cast<int>(n) : 1;
}

JPH::JobSystem::JobHandle JoltJobSystem::CreateJob(const char* name, JPH::ColorArg color,
                                                   const JobFunction& job_function,
                                                   JPH::uint32 num_dependencies) {
    // Mirrors JPH::JobSystemThreadPool::CreateJob: loop until a free slot is
    // available. Exhaustion is transient by construction — in-flight jobs
    // release their slots as they complete — and sized away by
    // cMaxPhysicsJobs; assert so a misconfigured world is loud in debug.
    JPH::uint32 index;
    for (;;) {
        index = jobs_.ConstructObject(name, color, this, job_function, num_dependencies);
        if (index != AvailableJobs::cInvalidObjectIndex) {
            break;
        }
        BRACKEN_ASSERT(false, "JoltJobSystem: out of job slots, waiting for jobs to complete");
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    Job* job = &jobs_.Get(index);

    // The handle owns one reference (Ref<Job> semantics).
    JobHandle handle(job);

    // Jobs with no outstanding dependencies are immediately runnable; Jolt
    // queues dependent jobs itself when their counter hits zero.
    if (num_dependencies == 0) {
        QueueJob(job);
    }
    return handle;
}

void JoltJobSystem::QueueJob(Job* job) { submit_(job); }

void JoltJobSystem::QueueJobs(Job** jobs, JPH::uint num_jobs) {
    for (JPH::uint i = 0; i < num_jobs; ++i) {
        submit_(jobs[i]);
    }
}

void JoltJobSystem::FreeJob(Job* job) { jobs_.DestructObject(job); }

void JoltJobSystem::submit_(Job* job) {
    // Queue reference discipline, matching JobSystemThreadPool exactly: the
    // queue (here: the thunk) takes one reference before the job becomes
    // reachable by an executor, and drops it after Execute. If a barrier
    // waiter executed the job first, Execute's internal dependency-count
    // CAS fails and this thunk just drops the reference. Group submit is
    // the zero-handle hot path (job_system.hpp): the pooled node recycles
    // on completion, nothing is allocated per thunk beyond the
    // std::function.
    job->AddRef();
    pool_.submit(group_, [job] {
        job->Execute();
        job->Release();
    });
}

} // namespace bracken::physics
