#pragma once

#include "bracken/core/jobs/cache_line.hpp"
#include "bracken/core/jobs/job_stage.hpp"
#include "bracken/core/jobs/mpsc_queue.hpp"
#include "bracken/core/jobs/task_handle.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace bracken::core::jobs {

class TaskGroup;
class NodePool; // per-worker node pool; defined in job_system.cpp

// JobNode — pooled, fixed-size job record (docs/multithreading_plan.md §7:
// "Job lifetimes — pooled, not heap-allocated"). Public only so the queue
// templates and TaskHandle can name it; hosts never touch nodes directly.
// Padded to a cache line so adjacent pool nodes (touched by different
// executor/waiter threads) never false-share.
struct alignas(k_cache_line) JobNode {
    // Intrusive link with exactly one use at a time: the MPSC injection
    // queue while queued externally, or the owning pool's freelist/free-inbox
    // while recycled. Orderings are those of the containing structure
    // (mpsc_queue.hpp / NodePool in job_system.cpp).
    std::atomic<JobNode*> next{nullptr};

    std::function<void()> fn; // the job body; cleared on recycle
    TaskGroup* group = nullptr; // non-owning; groups outlive their member jobs (joined by wait)
    NodePool* pool = nullptr;   // owning pool; set once at slab creation, never changes

    // Pipeline stage this job inherits from its submitter, installed by the
    // executor around fn() (see core/jobs/job_stage.hpp). Non-atomic: written
    // once at submit, published to the executor by the queue's release edge
    // like the rest of the node payload, and read only by the one thread that
    // won exec_guard.
    Stage stage = Stage::None;

    // done — the completion publish edge (house style: non-atomic payload +
    // one atomic publish edge). The executor writes `error`, then
    // release-stores done=true; readers acquire-load done before touching
    // `error`. Pairing: release store in JobSystem::complete_node_ ↔ acquire
    // loads in detail::node_is_done/node_error and the park loop of
    // JobSystem::wait. Also the futex word external waiters park on
    // (done.wait(false) ↔ done.notify_all() after the release store).
    std::atomic<bool> done{false};
    std::exception_ptr error; // non-atomic; gated by `done` (see above)

    // state — generation (high 32 bits) and refcount (low 32 bits) packed
    // into ONE atomic word so every reference operation is generation-checked
    // in a single CAS. This is the ABA/resurrection guard: a stale TaskHandle
    // (from a previous life of this node) that tries to add or drop a
    // reference fails the generation compare inside the CAS and becomes a
    // no-op — it can never inflate, deflate, or resurrect the refcount of the
    // node's CURRENT task. (Two separate atomics had exactly that bug: a
    // stale add_ref could interleave with submit's refcount initialization
    // and a later stale drop then drove the live task's count to zero while
    // it was still queued/executing — caught by TSan as fn reassignment
    // during invocation.)
    //
    // Lifecycle: a free node holds {g, 0}. submit() (owner thread) blind-
    // stores {g, 1} — safe because no handle carrying generation g exists
    // yet and every stale-generation CAS no-ops. Each live TaskHandle CASes
    // {g, c≥1} → {g, c+1} (fails ⇒ that task already completed). Droppers
    // CAS {g, c>1} → {g, c-1} (release: publishes their node accesses to the
    // eventual recycler). The LAST matching-generation dropper performs the
    // recycle transition {g, 1} → {g+1, 0} (acq_rel: acquires all prior
    // droppers' releases, then owns the node exclusively for the reset) and
    // pushes the node to its pool's free inbox. Once the generation has
    // bumped, no stale operation can touch the word again.
    //
    // Executing-system reference: submit's initial count of 1 (2 with a
    // returned handle) includes the execution reference dropped by
    // complete_node_. Group-submitted jobs have no handle, so they complete
    // with the count at 1 and recycle immediately — the zero-overhead hot
    // path. Generation wrap (2^32 recycles of one node between a handle's
    // birth and its query) is accepted and documented.
    std::atomic<std::uint64_t> state{0};

    static constexpr std::uint64_t k_state_count_mask = 0xffff'ffffull;
    static constexpr unsigned k_state_gen_shift = 32;
    static constexpr std::uint32_t state_gen(std::uint64_t s) noexcept {
        return static_cast<std::uint32_t>(s >> k_state_gen_shift);
    }
    static constexpr std::uint32_t state_count(std::uint64_t s) noexcept {
        return static_cast<std::uint32_t>(s & k_state_count_mask);
    }
    static constexpr std::uint64_t make_state(std::uint32_t gen, std::uint32_t count) noexcept {
        return (static_cast<std::uint64_t>(gen) << k_state_gen_shift) | count;
    }

    // Diagnostic invariant guard (kept in all builds with assertions): a
    // node must have exactly one executor at a time. execute_ CASes 0→(1 +
    // consume source) at entry and complete_node_ stores 0 at the end; a
    // nonzero exchange at entry is a double-take — the value identifies
    // which queue path each consumer came through (1=own-pop, 2=injection,
    // 3=steal, 4=shutdown-drain).
    std::atomic<std::uint32_t> exec_guard{0};
};

// Work-stealing job system (docs/multithreading_plan.md §7).
//
// Internals: each worker owns a Chase–Lev deque (owner push/pop at the
// bottom, thieves CAS-steal the top); external (non-worker) submits go
// through a Vyukov MPSC injection queue consumed by workers; jobs are pooled
// nodes recycled through per-worker freelists — no heap allocation and no
// shared_ptr on the submit hot path. Idle workers spin briefly, then park on
// a global work-epoch atomic (futex wait); each submit bumps the epoch and
// issues one targeted notify_one — notify_all only at shutdown. There are no
// mutexes and no polling anywhere in the frame-hot path (the only lock is the
// external-submitter node-pool mutex, off the worker paths).
class JobSystem {
public:
    // Design note: JobSystem is dual-mode — a Meyers singleton (instance())
    // AND publicly constructible for caller-owned pools (tests). The
    // singleton is canonical: the parallel_for(begin, end, …) convenience
    // overload services JobSystem::instance(); caller-owned pools use the
    // explicit parallel_for(JobSystem&, …) overload.
    static JobSystem& instance();

    // Out-of-line (= default in the .cpp): the inline form would instantiate
    // member cleanup for the incomplete Worker type in every includer.
    JobSystem();
    // Shuts down, then frees all node pools. TaskHandles must not outlive
    // the JobSystem that issued them (they stay valid across
    // shutdown()/init() cycles, but not past destruction).
    ~JobSystem();

    JobSystem(const JobSystem&) = delete;
    JobSystem& operator=(const JobSystem&) = delete;
    JobSystem(JobSystem&&) = delete;
    JobSystem& operator=(JobSystem&&) = delete;

    // Spawns the worker pool. thread_count == 0 means
    // hardware_concurrency() - 1, clamped to >= 1. Idempotent while
    // initialized; may be called again after shutdown() to re-initialize.
    void init(unsigned thread_count = 0);

    // Stops and joins all workers. Jobs still queued (never started) are
    // *discarded*: they complete with a "discarded at shutdown" error, so
    // is_done() flips true, has_error()/error() report the discard, group
    // counters drain, and any thread blocked in wait() wakes (and has the
    // error rethrown) instead of hanging. init() may be called again
    // afterwards to re-initialize the pool. Calling submit() concurrently
    // with shutdown() is unsupported.
    void shutdown();

    bool is_initialized() const noexcept { return initialized_.load(std::memory_order_acquire); }
    unsigned worker_count() const noexcept { return static_cast<unsigned>(workers_.size()); }
    unsigned pending_count() const noexcept {
        const std::int64_t p = pending_.load(std::memory_order_acquire);
        return p > 0 ? static_cast<unsigned>(p) : 0u;
    }

    // Submits a job and returns a handle for done/error queries and wait().
    // Worker-thread submits push to the caller's own deque; external submits
    // go through the injection queue.
    [[nodiscard]] TaskHandle submit(std::function<void()> fn);

    // Submits a job against a TaskGroup: no handle is created and the pooled
    // node recycles immediately on completion — the hot path parallel_for
    // uses. The group must outlive the job (join it with wait(group)).
    // Errors are captured into the group (first one wins) and rethrown by
    // the group join.
    void submit(TaskGroup& group, std::function<void()> fn);

    // Blocks until the task completes, helping in the meantime: the caller
    // drains its own deque (if it is a worker of this pool), consumes the
    // injection queue, and steals from other workers — so nested submit+wait
    // (fork-join) is safe even on a 1-worker pool. When there is nothing to
    // help with, the caller parks on the task's done flag (futex) and is
    // woken by completion. Rethrows the task's stored exception, if any.
    void wait(TaskHandle h);

    // Group join with the same helping behavior as wait(TaskHandle) — use
    // this (not TaskGroup::wait(), which only parks) anywhere the caller
    // could execute jobs instead of sleeping, and always from inside jobs.
    // Rethrows the group's first captured error after ALL members finished.
    void wait(TaskGroup& group);

    // Blocks until pending_count() reaches zero, helping in the meantime.
    void wait_all();

private:
    struct Worker; // deque + node pool; defined in job_system.cpp
    enum class InjectionResult { Got, Empty, Busy };

    void worker_main_(unsigned index);
    JobNode* acquire_node_(std::function<void()>&& fn, TaskGroup* group);
    void enqueue_(JobNode* n);
    // Tries to execute one job: own deque (workers), injection queue, then
    // stealing. Returns true if a job ran. injection_busy reports that the
    // injection consumer token was held elsewhere — callers on the epoch
    // parking path must NOT park then (the token holder is awake and
    // responsible); waiters with their own wake channel (task done flag,
    // group/pending counters) may.
    bool try_run_one_(bool& injection_busy);
    InjectionResult try_consume_injection_(JobNode*& out);
    JobNode* try_steal_();
    void execute_(JobNode* n, std::uint32_t source);
    void complete_node_(JobNode* n, std::exception_ptr err);
    void drain_queues_discarding_();

    std::vector<std::thread> workers_;
    // Worker state (deque + pool) persists across shutdown()/init() cycles:
    // live TaskHandles keep pointing into the pools, and freelists are
    // reused on re-init. Freed only at destruction.
    std::vector<std::unique_ptr<Worker>> worker_state_;

    // External (non-worker) submitters allocate nodes from a shared pool
    // under this mutex — a cold-ish path by design (§7: worker-side submits
    // are the hot path and are lock-free). Lazily created.
    std::mutex external_mutex_;
    std::unique_ptr<NodePool> external_pool_;

    // Injection queue for external submits; consumed by workers (and by
    // helping waiters) under the consumer token below.
    MpscQueue<JobNode> injection_;

    std::atomic<bool> initialized_{false}; // release-stored by init/shutdown ↔ acquire in is_initialized()
    // stopping_ — shutdown request. Release store in shutdown() ↔ acquire
    // loads in the worker loop; the epoch bump/notify_all in shutdown()
    // guarantees parked workers re-read it.
    std::atomic<bool> stopping_{false};
    // Number of live workers (bounds the steal sweep). Relaxed: written only
    // while the pool is quiescent (init before spawn / shutdown after join);
    // thread creation/join provides the ordering.
    std::atomic<unsigned> active_workers_{0};

    // pending_ — count of submitted-but-not-completed jobs. fetch_add
    // relaxed at submit (ordered before any worker by the queue publish
    // edge); fetch_sub acq_rel in complete_node_ BEFORE any waiter wake, so
    // a thread returning from a join observes a drained count
    // deterministically. Also the futex word wait_all() parks on
    // (notify_all when it reaches zero). Padded: decremented by all workers.
    alignas(k_cache_line) std::atomic<std::int64_t> pending_{0};

    // epoch_ — the work-availability epoch idle workers park on
    // (§7 "Parking — futex-style, two-phase"). Each enqueue_ does a release
    // fetch_add then ONE notify_one (targeted wake per job); notify_all only
    // at shutdown. A worker reads the epoch (acquire), re-checks all queues,
    // and only then waits — any submit after the read changes the value (the
    // wait returns immediately), any submit before it is found by the
    // re-check via the queue publish edges. Padded: bumped by every submit.
    alignas(k_cache_line) std::atomic<std::uint64_t> epoch_{0};

    // injection_consumer_busy_ — consumer-election token for the
    // single-consumer MPSC injection queue. exchange(true, acquire) to take,
    // store(false, release) to hand back; the pairing carries the queue's
    // consumer-private tail cursor between consecutive consumers. Padded:
    // CASed by all workers when external jobs are in flight.
    alignas(k_cache_line) std::atomic<bool> injection_consumer_busy_{false};
};

} // namespace bracken::core::jobs
