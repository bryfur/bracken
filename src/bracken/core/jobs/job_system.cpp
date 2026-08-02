#include "bracken/core/jobs/job_system.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/chase_lev_deque.hpp"
#include "bracken/core/jobs/task_group.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <random>
#include <stdexcept>
#include <utility>

namespace bracken::core::jobs {

namespace {

// Identity of the current thread within a pool. t_worker_index_ is only
// meaningful when t_owner_ matches the JobSystem being asked — caller-owned
// pools and the singleton coexist (tests), so the pair disambiguates.
thread_local JobSystem* t_owner_ = nullptr;
thread_local int t_worker_index_ = -1;

// Brief-spin pause between failed work hunts, before parking.
inline void cpu_relax() noexcept {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// Failed work-hunt rounds before a worker parks on the epoch futex. Each
// round already includes a full steal sweep, so this is a short grace window
// for freshly submitted work, not a busy-wait.
constexpr unsigned k_spin_rounds = 64;

} // namespace

// ---------------------------------------------------------------------------
// NodePool — per-worker (or external) JobNode pool with freelists.
//
// Allocation is owner-only (the worker thread that owns the pool, or any
// external submitter holding JobSystem::external_mutex_). Release can happen
// from ANY thread — the last reference dropper recycles — so remote frees go
// through a lock-free Treiber *push-only* inbox the owner detaches wholesale
// (exchange), which is immune to the classic Treiber-pop ABA. Slabs are only
// ever freed by ~NodePool (JobSystem destruction): live TaskHandles may point
// into them across shutdown()/init() cycles.
// ---------------------------------------------------------------------------
class NodePool {
public:
    static constexpr std::size_t k_slab_nodes = 64;

    NodePool() = default;
    NodePool(const NodePool&) = delete;
    NodePool& operator=(const NodePool&) = delete;

    // Owner-only (see class comment). Returns a quiescent node: fn/error
    // cleared, done=false, state {g, 0} with g already bumped by the recycle
    // CAS in detail::node_release_ref (so no stale operation can touch it).
    JobNode* acquire() {
        if (local_free_ == nullptr) {
            // Detach the whole remote-free inbox. Acquire pairs with the
            // release CAS in release() so the releasers' resets (fn/error
            // cleared, generation bumped) are visible.
            local_free_ = free_inbox_.exchange(nullptr, std::memory_order_acquire);
        }
        if (local_free_ == nullptr) {
            grow_();
        }
        JobNode* n = local_free_;
        local_free_ = n->next.load(std::memory_order_relaxed);
        n->next.store(nullptr, std::memory_order_relaxed);
        // Fresh task state. Relaxed: the node is private to this thread
        // until the queue publish edge (deque release fence / MPSC release
        // store) hands it to a worker.
        n->done.store(false, std::memory_order_relaxed);
        return n;
    }

    // Any thread; called exactly once per lifetime, by the winner of the
    // recycle CAS {g, 1} → {g+1, 0} in detail::node_release_ref — which both
    // implies the task is complete and has already invalidated every stale
    // handle (their generation-checked CASes now no-op), so this thread owns
    // the node exclusively.
    static void release(JobNode* n) noexcept {
        n->fn = nullptr; // destroy captures
        n->error = nullptr;
        n->group = nullptr;
        // Treiber push into the owner's inbox. Release publishes the resets
        // above to the owner's acquire exchange in acquire(). Push-only from
        // this side + wholesale detach on the other side ⇒ no ABA.
        NodePool* pool = n->pool;
        JobNode* head = pool->free_inbox_.load(std::memory_order_relaxed);
        do {
            n->next.store(head, std::memory_order_relaxed);
        } while (!pool->free_inbox_.compare_exchange_weak(head, n, std::memory_order_release,
                                                          std::memory_order_relaxed));
    }

private:
    struct Slab {
        std::array<JobNode, k_slab_nodes> nodes;
    };

    void grow_() {
        auto slab = std::make_unique<Slab>();
        for (JobNode& n : slab->nodes) {
            n.pool = this;
            n.next.store(local_free_, std::memory_order_relaxed);
            local_free_ = &n;
        }
        slabs_.push_back(std::move(slab));
    }

    JobNode* local_free_ = nullptr; // owner-only LIFO (links via JobNode::next, relaxed)
    // free_inbox_ — remote-free Treiber stack head. Release CAS push (any
    // thread) ↔ acquire exchange-detach (owner). Padded: remote frees come
    // from arbitrary workers.
    alignas(k_cache_line) std::atomic<JobNode*> free_inbox_{nullptr};
    std::vector<std::unique_ptr<Slab>> slabs_; // owner-only; freed at destruction
};

// ---------------------------------------------------------------------------
// detail:: node accessors used by TaskHandle (declared in task_handle.hpp).
// ---------------------------------------------------------------------------
namespace detail {

bool node_try_add_ref(JobNode* n, std::uint32_t generation) noexcept {
    // Generation-checked increment — see the protocol at JobNode::state.
    // Failure (stale generation, or count already 0) proves the referenced
    // task completed; the initial acquire load pairs with the recycle CAS's
    // release side so that inference is safe to act on.
    std::uint64_t s = n->state.load(std::memory_order_acquire);
    for (;;) {
        if (JobNode::state_gen(s) != generation || JobNode::state_count(s) == 0) {
            return false;
        }
        // Failure acquire: re-derive the done-inference above from the fresh
        // value. Success acquire only because GCC's __atomic builtin still
        // enforces the pre-C++17 failure<=success rule (-Winvalid-memory-model);
        // the success path itself needs nothing — acquiring a reference
        // publishes nothing.
        if (n->state.compare_exchange_weak(s, s + 1, std::memory_order_acquire,
                                           std::memory_order_acquire)) {
            return true;
        }
    }
}

void node_release_ref(JobNode* n, std::uint32_t generation) noexcept {
    std::uint64_t s = n->state.load(std::memory_order_relaxed);
    for (;;) {
        if (JobNode::state_gen(s) != generation) {
            return; // stale drop: that life already recycled — no-op
        }
        BRACKEN_ASSERT(JobNode::state_count(s) > 0, "release without a matching reference");
        if (JobNode::state_count(s) == 1) {
            // Last matching-generation reference: the recycle transition
            // {g, 1} → {g+1, 0}. acq_rel: acquire joins every prior
            // dropper's release decrement (their node accesses — including
            // the executor's fn invocation — happen-before the reset in
            // NodePool::release); release publishes the bump to the acquire
            // loads in node_try_add_ref/node_is_done.
            const std::uint64_t recycled = JobNode::make_state(generation + 1, 0);
            if (n->state.compare_exchange_weak(s, recycled, std::memory_order_acq_rel,
                                               std::memory_order_relaxed)) {
                NodePool::release(n);
                return;
            }
        } else {
            // Plain decrement. Release: publishes this thread's accesses to
            // the eventual recycler's acquire (shared_ptr discipline).
            if (n->state.compare_exchange_weak(s, s - 1, std::memory_order_release,
                                               std::memory_order_relaxed)) {
                return;
            }
        }
    }
}

bool node_is_done(const JobNode* n, std::uint32_t generation) noexcept {
    // Generation mismatch ⇒ the node recycled, which requires the original
    // task to have completed (acquire pairs with the recycle CAS's release
    // in node_release_ref).
    if (JobNode::state_gen(n->state.load(std::memory_order_acquire)) != generation) {
        return true;
    }
    return n->done.load(std::memory_order_acquire);
}

std::exception_ptr node_error(const JobNode* n, std::uint32_t generation) noexcept {
    // Precondition: the caller holds a matching-generation reference (a live
    // TaskHandle), so the node cannot recycle between this check and the
    // error read below.
    if (JobNode::state_gen(n->state.load(std::memory_order_acquire)) != generation) {
        return nullptr; // recycled: the original task's error is gone
    }
    // Acquire on done gates the non-atomic `error` read (house style).
    if (!n->done.load(std::memory_order_acquire)) {
        return nullptr;
    }
    return n->error;
}

namespace {
// Park on a live node's done flag. Only called while the caller holds a
// reference (so the node cannot recycle under the futex wait).
void node_wait_done(const JobNode* n, std::uint32_t generation) noexcept {
    if (JobNode::state_gen(n->state.load(std::memory_order_acquire)) != generation) {
        return;
    }
    // atomic::wait re-checks the value internally, so a completion landing
    // between the caller's is_done() check and this call is not lost;
    // complete_node_ and the shutdown discard both notify_all after their
    // release store of done.
    n->done.wait(false, std::memory_order_acquire);
}
} // namespace

} // namespace detail

// ---------------------------------------------------------------------------
// JobSystem
// ---------------------------------------------------------------------------

struct JobSystem::Worker {
    ChaseLevDeque<JobNode> deque{256};
    NodePool pool;
};

JobSystem& JobSystem::instance() {
    static JobSystem inst;
    return inst;
}

JobSystem::JobSystem() = default; // here, where Worker/NodePool are complete

JobSystem::~JobSystem() {
    shutdown();
    // Node storage dies here — TaskHandles must not outlive the JobSystem.
    worker_state_.clear();
    external_pool_.reset();
}

void JobSystem::init(unsigned thread_count) {
    if (initialized_.load(std::memory_order_acquire)) {
        return;
    }

    unsigned n = thread_count;
    if (n == 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        n = hw > 1 ? hw - 1 : 1;
    }
    n = std::max(1u, n);

    stopping_.store(false, std::memory_order_release);
    pending_.store(0, std::memory_order_release);

    // Extend (never shrink) the persistent worker state: pools may still be
    // referenced by handles from a previous incarnation, and their freelists
    // are reused. Deques were drained by the previous shutdown.
    while (worker_state_.size() < n) {
        worker_state_.push_back(std::make_unique<Worker>());
    }
    active_workers_.store(n, std::memory_order_relaxed); // pre-spawn; thread creation orders it

    workers_.reserve(n);
    for (unsigned i = 0; i < n; ++i) {
        workers_.emplace_back([this, i]() { worker_main_(i); });
    }

    initialized_.store(true, std::memory_order_release);
    BRACKEN_LOG_INFO("jobs", "Initialized {} workers", n);
}

void JobSystem::shutdown() {
    if (!initialized_.load(std::memory_order_acquire)) {
        return;
    }

    stopping_.store(true, std::memory_order_release);

    // First discard pass, *before* joining: complete every queued-but-
    // unstarted job with a "discarded" error so threads blocked in wait() on
    // such tasks (including workers in nested waits) wake and unwind instead
    // of deadlocking the joins below.
    drain_queues_discarding_();

    // Wake every parked worker to observe stopping_. The bump makes any
    // concurrent epoch_.wait(e) return even if the notify lands first.
    epoch_.fetch_add(1, std::memory_order_release);
    epoch_.notify_all(); // the one legitimate notify_all (thundering herd is the point)

    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
    workers_.clear();

    // Second pass: jobs submitted by in-flight work while the workers were
    // winding down. After the joins nothing can enqueue anymore, so this
    // drain is quiescent and exact.
    drain_queues_discarding_();

    active_workers_.store(0, std::memory_order_relaxed);
    pending_.store(0, std::memory_order_release);
    pending_.notify_all(); // belt-and-braces for any straggling wait_all
    initialized_.store(false, std::memory_order_release);
    stopping_.store(false, std::memory_order_release);
}

void JobSystem::drain_queues_discarding_() {
    std::size_t discarded = 0;

    auto discard = [&](JobNode* n) {
        // One exception object per task: concurrent rethrows of a shared
        // exception_ptr would alias one object.
        complete_node_(n, std::make_exception_ptr(std::runtime_error(
                              "job discarded by JobSystem::shutdown() before it ran")));
        ++discarded;
    };

    // Injection queue. Busy means a live consumer holds the token — it will
    // either take a job (fine: it observes stopping_ before running another)
    // or hand the token back momentarily.
    for (;;) {
        JobNode* n = nullptr;
        const InjectionResult r = try_consume_injection_(n);
        if (r == InjectionResult::Got) {
            discard(n);
            continue;
        }
        if (r == InjectionResult::Busy) {
            std::this_thread::yield();
            continue;
        }
        break; // Empty (submit concurrent with shutdown is unsupported, so no mid-push producers)
    }

    // Worker deques: steal-drain (steal is legal from any thread). A steal
    // that loses its CAS means someone else took an element — progress — so
    // just retry until the deque reads empty.
    const unsigned n_workers = active_workers_.load(std::memory_order_relaxed);
    for (unsigned i = 0; i < n_workers && i < worker_state_.size(); ++i) {
        auto& deque = worker_state_[i]->deque;
        for (;;) {
            if (JobNode* n = deque.steal()) {
                discard(n);
            } else if (deque.empty()) {
                break;
            } else {
                std::this_thread::yield(); // lost a race or owner mid-pop; retry
            }
        }
    }

    if (discarded > 0) {
        BRACKEN_LOG_WARN("jobs", "shutdown() discarded {} queued job(s); their handles complete with an error",
                         discarded);
    }
}

JobNode* JobSystem::acquire_node_(std::function<void()>&& fn, TaskGroup* group) {
    JobNode* n = nullptr;
    if (t_owner_ == this && t_worker_index_ >= 0) {
        // Worker submit: lock-free from the worker's own pool.
        n = worker_state_[static_cast<std::size_t>(t_worker_index_)]->pool.acquire();
    } else {
        // External submit: shared pool under a mutex (cold-ish path; §7).
        std::lock_guard<std::mutex> lock(external_mutex_);
        if (!external_pool_) {
            external_pool_ = std::make_unique<NodePool>();
        }
        n = external_pool_->acquire();
    }
    n->fn = std::move(fn);
    n->group = group;
    // The stage travels with the node: a job submitted from inside the sim
    // stage IS sim stage, wherever it later executes and whoever steals it.
    n->stage = current_stage();
    return n;
}

void JobSystem::push_(JobNode* n) {
    if (t_owner_ == this && t_worker_index_ >= 0) {
        // Owner push; the deque's release fence publishes the node payload.
        worker_state_[static_cast<std::size_t>(t_worker_index_)]->deque.push(n);
    } else {
        // MPSC push; its release store publishes the node payload.
        injection_.push(n);
    }
}

void JobSystem::publish_work_(std::size_t count) {
    if (count == 0) {
        return;
    }
    // Work-epoch protocol (see epoch_ declaration): the bump comes AFTER the
    // queue publish completes, so a parker that read the pre-bump epoch and
    // found nothing will have its wait(e) return, and a parker that read the
    // post-bump epoch synchronized with the push(es) and finds the job in its
    // re-check. Release pairs with the workers' acquire epoch loads. A batch
    // bumps by `count` in one RMW — every one of its pushes precedes it, so
    // the protocol holds for all of them at once.
    epoch_.fetch_add(count, std::memory_order_release);
    if (count == 1) {
        epoch_.notify_one(); // one targeted wake per job; notify_all only at shutdown…
    } else {
        epoch_.notify_all(); // …and per BATCH, which made `count` jobs runnable at once
    }
}

void JobSystem::enqueue_(JobNode* n) {
    push_(n);
    publish_work_(1);
}

TaskHandle JobSystem::submit(std::function<void()> fn) {
    JobNode* n = acquire_node_(std::move(fn), nullptr);
    // Arm the execution reference: {g, 0} → {g, 1}. A blind store is safe
    // here and ONLY here: no handle carrying generation g exists yet, and
    // every stale-generation operation no-ops inside its generation-checked
    // CAS (see JobNode::state). Relaxed: the queue publish edge in enqueue_
    // orders it for consumers.
    const std::uint32_t gen = JobNode::state_gen(n->state.load(std::memory_order_relaxed));
    n->state.store(JobNode::make_state(gen, 1), std::memory_order_relaxed);
    // The handle's reference: CAS {g, 1} → {g, 2} — cannot fail (same
    // thread, matching generation, count ≥ 1).
    TaskHandle h(n, gen);
    // Relaxed: ordered before any worker's decrement by the queue publish
    // edge in enqueue_ (see pending_ declaration).
    pending_.fetch_add(1, std::memory_order_relaxed);
    enqueue_(n);
    return h;
}

void JobSystem::submit_deferred_(TaskGroup& group, std::function<void()> fn) {
    JobNode* n = acquire_node_(std::move(fn), &group);
    // Execution reference only — no handle, so completion recycles the node
    // immediately (the parallel_for hot path). Blind store safe as above.
    const std::uint32_t gen = JobNode::state_gen(n->state.load(std::memory_order_relaxed));
    n->state.store(JobNode::make_state(gen, 1), std::memory_order_relaxed);
    group.on_job_submitted();
    pending_.fetch_add(1, std::memory_order_relaxed); // see pending_ declaration
    // Queued, but NOT announced: the caller publishes (submit below for one
    // job, submit_batch once for the whole batch). Until then the node is
    // still REACHABLE — a spinning worker, or a helping waiter inside
    // wait(group), can steal it; the epoch bump only wakes PARKED workers.
    push_(n);
}

void JobSystem::submit(TaskGroup& group, std::function<void()> fn) {
    submit_deferred_(group, std::move(fn));
    publish_work_(1);
}

JobSystem::InjectionResult JobSystem::try_consume_injection_(JobNode*& out) {
    // Consumer election for the single-consumer MPSC queue. Acquire on a
    // successful take pairs with the release hand-back below, carrying the
    // queue's consumer-private tail cursor between consecutive consumers.
    if (injection_consumer_busy_.exchange(true, std::memory_order_acquire)) {
        return InjectionResult::Busy;
    }
    out = injection_.pop();
    injection_consumer_busy_.store(false, std::memory_order_release);
    return out != nullptr ? InjectionResult::Got : InjectionResult::Empty;
}

JobNode* JobSystem::try_steal_() {
    const unsigned n = active_workers_.load(std::memory_order_relaxed);
    if (n == 0) {
        return nullptr;
    }
    thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<unsigned> dist(0, n - 1);
    const int self = (t_owner_ == this) ? t_worker_index_ : -1;
    for (unsigned attempts = 0; attempts < n * 2; ++attempts) {
        const unsigned victim = dist(rng);
        if (static_cast<int>(victim) == self) {
            continue;
        }
        if (JobNode* job = worker_state_[victim]->deque.steal()) {
            return job;
        }
    }
    return nullptr;
}

bool JobSystem::try_run_one_(bool& injection_busy) {
    injection_busy = false;
    // A worker (of THIS pool) drains its own deque first: nested submit()
    // from a worker lands there, and on small pools no one else can steal it
    // — without this, fork-join livelocks on a 1-worker pool.
    if (t_owner_ == this && t_worker_index_ >= 0) {
        if (JobNode* n = worker_state_[static_cast<std::size_t>(t_worker_index_)]->deque.pop()) {
            execute_(n, 1); // source: own-deque pop
            return true;
        }
    }
    JobNode* n = nullptr;
    switch (try_consume_injection_(n)) {
    case InjectionResult::Got:
        execute_(n, 2); // source: injection queue
        return true;
    case InjectionResult::Busy:
        injection_busy = true;
        break;
    case InjectionResult::Empty:
        break;
    }
    if ((n = try_steal_()) != nullptr) {
        execute_(n, 3); // source: steal
        return true;
    }
    return false;
}

void JobSystem::execute_(JobNode* n, std::uint32_t source) {
    // Double-take detector: exactly one executor per node lifetime. A
    // nonzero previous value means two queue paths served the same node —
    // the values identify both paths (see JobNode::exec_guard).
    const std::uint32_t prev = n->exec_guard.exchange(source, std::memory_order_acq_rel);
    if (prev != 0) {
        BRACKEN_LOG_ERROR("jobs", "DOUBLE-TAKE: node {} served by source {} while source {} still executing",
                          static_cast<const void*>(n), source, prev);
        BRACKEN_ASSERT(false, "job node double-take (see log for sources)");
    }
    std::exception_ptr err;
    {
        // Install the submitter's stage for the duration of the body and
        // restore it after — correct under stealing (this may be any worker)
        // and under helping waits (this may be the main thread mid-join).
        StageScope stage_scope(n->stage);
        try {
            n->fn();
        } catch (...) {
            err = std::current_exception();
        }
    }
    complete_node_(n, std::move(err));
}

void JobSystem::complete_node_(JobNode* n, std::exception_ptr err) {
    // Destroy the body (and its captures) at completion, not at recycle —
    // a retained TaskHandle must not pin captured state alive. Only the
    // executing/discarding thread touches fn, so this is race-free.
    n->fn = nullptr;
    TaskGroup* group = n->group;
    if (group != nullptr && err) {
        group->capture_error(err); // first error wins (CAS-guarded slot)
    }
    n->error = std::move(err); // non-atomic; published by the release store below

    // Decrement pending BEFORE publishing done: `done` itself is a
    // handle-waiter's wake condition (is_done polls the value — no notify
    // required), so the decrement must be sequenced before the store or a
    // waiter could return from wait() and still read a stale nonzero
    // pending_count(). This ordering makes "a thread returning from any
    // join observes pending_count() == 0 once its jobs are drained" hold
    // for handle waits, group joins, AND wait_all (and keeps the
    // historically flaky ParallelForTest.PendingCountIsZeroOnceDrained
    // deterministic). acq_rel: release joins the pending release chain
    // wait_all acquires; acquire on the 1→0 edge orders the notify after
    // every prior decrement.
    if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        pending_.notify_all(); // wake wait_all parkers
    }
    n->done.store(true, std::memory_order_release);
    n->done.notify_all(); // wake threads parked on this task in wait()
    if (group != nullptr) {
        group->on_job_finished(); // release fetch_sub; notifies group waiters at zero
    }
    n->exec_guard.store(0, std::memory_order_release); // release the double-take detector
    // Drop the execution reference — last dropper recycles. Group-submitted
    // jobs have no handle, so this recycles immediately (hot path). The
    // generation read is stable: our own execution reference (count ≥ 1)
    // blocks the recycle transition until this very call.
    detail::node_release_ref(n, JobNode::state_gen(n->state.load(std::memory_order_relaxed)));
}

void JobSystem::worker_main_(unsigned index) {
    BRACKEN_PROFILE_THREAD("JobWorker");
    BRACKEN_PROFILE_ZONE("JobWorker");
    t_owner_ = this;
    t_worker_index_ = static_cast<int>(index);

    unsigned idle_rounds = 0;
    auto last_plot = std::chrono::steady_clock::now();

    while (!stopping_.load(std::memory_order_acquire)) {
        bool injection_busy = false;
        if (try_run_one_(injection_busy)) {
            idle_rounds = 0;
            if (index == 0) {
                const auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_plot).count() >= 1) {
                    BRACKEN_PROFILE_PLOT("jobs.pending",
                                         static_cast<int64_t>(pending_.load(std::memory_order_acquire)));
                    last_plot = now;
                }
            }
            continue;
        }

        // Spin briefly before parking (two-phase parking, §7).
        if (++idle_rounds < k_spin_rounds) {
            cpu_relax();
            continue;
        }
        if (injection_busy) {
            // The injection consumer token is held elsewhere; jobs may be in
            // the queue that we must not sleep past — the holder is awake,
            // so just yield and retry. (Waiters with their own wake channel
            // may park here; the epoch path may not.)
            std::this_thread::yield();
            continue;
        }

        // Park. Epoch protocol: read the epoch FIRST, re-check for work,
        // then wait — a submit after our read changes the value (wait
        // returns immediately), a submit before it is visible to the
        // re-check through the queue publish edges (acquire pairs with the
        // enqueue_ release bump).
        const std::uint64_t e = epoch_.load(std::memory_order_acquire);
        if (try_run_one_(injection_busy)) {
            idle_rounds = 0;
            continue;
        }
        if (injection_busy) {
            std::this_thread::yield();
            continue;
        }
        if (stopping_.load(std::memory_order_acquire)) {
            break;
        }
        epoch_.wait(e, std::memory_order_acquire);
        idle_rounds = 0;
    }

    // Shutdown-deadlock guard: a nested submit can land on THIS worker's
    // deque after shutdown's first discard pass; once this worker exits, a
    // random-probe steal is the only route to that job, and another worker
    // parked in a nested wait() on it would block shutdown's join forever
    // (the second drain runs only after the joins). Drain our own deque on
    // the way out, completing each stranded job with the shutdown error —
    // that publishes done and wakes its waiters through the normal
    // complete_node_ path.
    while (JobNode* stranded =
               worker_state_[static_cast<std::size_t>(t_worker_index_)]->deque.pop()) {
        complete_node_(stranded,
                       std::make_exception_ptr(std::runtime_error(
                           "job discarded by JobSystem::shutdown() before it ran")));
    }

    t_owner_ = nullptr;
    t_worker_index_ = -1;
}

void JobSystem::wait(TaskHandle h) {
    if (!h.valid()) {
        return;
    }
    // `h` is a by-value copy: it holds a reference for the whole wait, so
    // the node cannot recycle under the park below.
    while (!h.is_done()) {
        bool injection_busy = false;
        if (try_run_one_(injection_busy)) {
            continue;
        }
        if (injection_busy) {
            cpu_relax(); // token holder is mid-pop; retry rather than syscall-park
            continue;
        }
        // Nothing to help with: park on the task's done flag. Safe even if
        // other queues gain work meanwhile — our wake channel is this task's
        // completion, and queued work always has an awake responsible thread
        // (its owner, the injection token holder, or a notified worker).
        detail::node_wait_done(h.node(), h.generation());
    }
    if (std::exception_ptr e = h.error()) {
        std::rethrow_exception(e);
    }
}

void JobSystem::wait(TaskGroup& group) {
    for (;;) {
        const std::int64_t p = group.pending_.load(std::memory_order_acquire);
        if (p <= 0) {
            break;
        }
        bool injection_busy = false;
        if (try_run_one_(injection_busy)) {
            continue;
        }
        if (injection_busy) {
            cpu_relax();
            continue;
        }
        // Park on the immortal group-join epoch, not on the group's own
        // counter (UAF guard — see detail::GroupJoinEpoch in task_group.hpp).
        // Epoch-read → condition-re-check → wait(old) ⇒ no lost wake.
        const std::uint64_t e = detail::g_group_join_epoch.value.load(std::memory_order_acquire);
        if (group.pending_.load(std::memory_order_acquire) <= 0) {
            break;
        }
        detail::g_group_join_epoch.value.wait(e, std::memory_order_acquire);
    }
    if (std::exception_ptr e = group.consume_error_()) {
        std::rethrow_exception(e);
    }
}

void JobSystem::wait_all() {
    for (;;) {
        const std::int64_t p = pending_.load(std::memory_order_acquire);
        if (p <= 0) {
            break;
        }
        bool injection_busy = false;
        if (try_run_one_(injection_busy)) {
            continue;
        }
        if (injection_busy) {
            cpu_relax();
            continue;
        }
        // Park on pending_; woken by complete_node_'s notify_all when the
        // count reaches zero (and by shutdown's belt-and-braces notify).
        pending_.wait(p, std::memory_order_acquire);
    }
}

} // namespace bracken::core::jobs
