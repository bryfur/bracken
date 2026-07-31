#pragma once

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/cache_line.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <utility>

namespace bracken::core::jobs {

class JobSystem;

namespace detail {

// group_join_epoch — the futex word ALL TaskGroup waiters park on, in static
// storage so it outlives every group. This is the use-after-free guard for
// the join protocol: the completing worker's zero-publication (the group's
// pending_ fetch_sub) is the LAST time it touches group memory — the wake
// that follows (epoch bump + notify) happens on this immortal word, so a
// waiter that observes zero may return and the caller may destroy the
// stack-allocated group immediately, while the completer is still mid-wake.
// (Parking on the group's own counter had exactly that UAF: notify_all on a
// freed TaskGroup.) Bumped only on group-completion events (per batch, not
// per job); spurious wakes of unrelated group waiters just re-check their
// own counter and re-park.
struct GroupJoinEpoch {
    // Protocol: waiters load (acquire) → re-check their group's pending_ →
    // wait(old). Completers bump (release) AFTER the group's pending_ hit
    // zero, then notify_all. Release/acquire pairing carries no payload —
    // the group's own pending_ release/acquire edge publishes job effects —
    // the epoch only has to order the re-check against the park (no lost
    // wake: a bump between re-check and wait() changes the value, so wait
    // returns immediately).
    alignas(k_cache_line) std::atomic<std::uint64_t> value{0};
};
inline GroupJoinEpoch g_group_join_epoch;

} // namespace detail

// TaskGroup — fork-join counter for a batch of jobs (docs/multithreading_plan.md §7).
//
// One cache-line-padded atomic counter and nothing else on the hot path:
// JobSystem::submit(TaskGroup&, fn) increments it, each completing member job
// decrements it with a release fetch_sub, and waiters park on
// std::atomic::wait/notify until it reaches zero. Group-submitted jobs carry
// no TaskHandle and their pooled nodes recycle immediately on completion —
// this is the zero-refcount hot path parallel_for uses.
//
// Errors: the FIRST exception thrown by a member job is captured into a
// CAS-guarded slot; wait() rethrows it only AFTER every member job has
// finished (the join is unconditional — never unwinds past running members).
// When several threads wait() concurrently, exactly one of them (the one that
// wins the consume CAS) rethrows; the others return normally. Consuming the
// error also clears it, which is what makes the group reusable across frames.
//
// Reuse contract: a group may be reused for batch after batch, but join and
// resubmission must be ordered by the caller (the normal frame pattern:
// submit → wait → next frame submit → …). Submitting concurrently with a
// wait() that is supposed to join those submissions is a caller error.
//
// Lifetime: the group must outlive its member jobs; JobSystem joins are the
// mechanism (wait returns only at zero). The destructor asserts quiescence.
//
// wait() here is park-only — safe from any external thread, but it does not
// help execute jobs. From inside a job (or whenever the caller can help),
// use JobSystem::wait(TaskGroup&), which drains/steals work while waiting and
// is therefore fork-join safe even on a 1-worker pool.
class TaskGroup {
public:
    TaskGroup() = default;
    ~TaskGroup() {
        BRACKEN_ASSERT(pending_.load(std::memory_order_acquire) == 0,
                       "TaskGroup destroyed with member jobs still in flight");
    }

    TaskGroup(const TaskGroup&) = delete;
    TaskGroup& operator=(const TaskGroup&) = delete;
    TaskGroup(TaskGroup&&) = delete;
    TaskGroup& operator=(TaskGroup&&) = delete;

    // Number of member jobs not yet finished (racy snapshot).
    std::size_t pending() const noexcept {
        const std::int64_t n = pending_.load(std::memory_order_acquire);
        return n > 0 ? static_cast<std::size_t>(n) : 0u;
    }

    // Blocks (parking, without helping) until every member job has finished,
    // then rethrows the first captured member exception, if any — in exactly
    // one waiter (see class comment). Prefer JobSystem::wait(TaskGroup&)
    // when the caller could help execute jobs instead of sleeping.
    void wait() {
        for (;;) {
            if (pending_.load(std::memory_order_acquire) <= 0) {
                break;
            }
            // Park on the immortal epoch word, not on pending_ (see
            // detail::GroupJoinEpoch — UAF guard). Read epoch, re-check the
            // condition, then wait on the old epoch value: a completion
            // landing between the re-check and the wait bumps the epoch, so
            // the wait returns immediately — no lost wake.
            const std::uint64_t e =
                detail::g_group_join_epoch.value.load(std::memory_order_acquire);
            if (pending_.load(std::memory_order_acquire) <= 0) {
                break;
            }
            detail::g_group_join_epoch.value.wait(e, std::memory_order_acquire);
        }
        if (std::exception_ptr e = consume_error_()) {
            std::rethrow_exception(e);
        }
    }

    // ---- Internal API (called by JobSystem; not for host code) ----

    // Registers one member job. Relaxed is sufficient: the job becomes
    // runnable only through the queue publish edge (deque release fence /
    // MPSC release store), which orders this increment before any worker's
    // decrement; join/submit ordering across threads is the caller's
    // contract (see class comment).
    void on_job_submitted() noexcept { pending_.fetch_add(1, std::memory_order_relaxed); }

    // Completes one member job. Release: everything this job wrote (its
    // side effects, a captured error) happens-before a waiter that
    // acquire-observes the counter afterwards — RMWs continue the release
    // sequence, so the acquire read of the final zero synchronizes with
    // EVERY member's decrement, not just the last.
    void on_job_finished() noexcept {
        if (pending_.fetch_sub(1, std::memory_order_release) == 1) {
            // Zero-observer acquire fence (§7): orders all members' effects
            // before anything this thread does after the join point.
            std::atomic_thread_fence(std::memory_order_acquire);
            // CRITICAL: the fetch_sub above was this thread's LAST access to
            // group memory. A waiter may already have observed zero,
            // returned, and destroyed the (stack) group — so the wake below
            // must touch only the immortal epoch word, never `this`.
            detail::g_group_join_epoch.value.fetch_add(1, std::memory_order_release);
            detail::g_group_join_epoch.value.notify_all();
        }
    }

    // Captures the first member exception. Three-state slot protocol
    // (none → writing → ready) so a consume can NEVER read `error_` while a
    // writer is mid-move — even a batch-boundary straggler (a waiter of the
    // previous batch consuming after the group was re-armed) gets a clean
    // handoff or nothing, not a torn exception_ptr. Publication to the
    // normal joiner still rides the winner's release fetch_sub in
    // on_job_finished(); the release store of `ready` covers stragglers.
    void capture_error(std::exception_ptr e) noexcept {
        int expected = k_no_error;
        if (error_state_.compare_exchange_strong(expected, k_writing_error,
                                                 std::memory_order_relaxed,
                                                 std::memory_order_relaxed)) {
            error_ = std::move(e);
            error_state_.store(k_error_ready, std::memory_order_release);
        }
    }

private:
    friend class JobSystem; // JobSystem::wait(TaskGroup&) parks on pending_ and consumes the error

    static constexpr int k_no_error = 0;
    static constexpr int k_writing_error = 1;
    static constexpr int k_error_ready = 2;

    // Exactly one caller that observed pending_ == 0 takes the error;
    // consuming clears it for reuse. Only a READY slot can be consumed — a
    // slot mid-write (k_writing_error) reads as "no error yet" rather than
    // racing the writer. Acquire on the winning CAS pairs with the release
    // store in capture_error (belt for stragglers whose synchronization did
    // not come through pending_'s edge). Reuse + concurrent-waiter caveat:
    // a straggling waiter from the PREVIOUS batch that consumes after the
    // group was re-armed may win a NEXT-batch error (delivered to exactly
    // one waiter, never torn); callers that reuse a group across batches
    // with multiple concurrent waiters accept that delivery ambiguity.
    std::exception_ptr consume_error_() noexcept {
        if (error_state_.load(std::memory_order_acquire) != k_error_ready) {
            return nullptr;
        }
        int expected = k_error_ready;
        if (!error_state_.compare_exchange_strong(expected, k_no_error, std::memory_order_acquire,
                                                  std::memory_order_relaxed)) {
            return nullptr; // another waiter took it (or a writer re-armed)
        }
        std::exception_ptr e = std::move(error_);
        error_ = nullptr;
        return e;
    }

    // pending_ — the group's only hot-path state. Publishes member-job
    // completion: release fetch_sub in on_job_finished() ↔ acquire loads in
    // wait()/pending()/JobSystem::wait(TaskGroup&). NOT the futex word —
    // waiters park on detail::g_group_join_epoch (see its comment: parking
    // on member memory was a use-after-free). Padded: decremented by many
    // workers at once.
    alignas(k_cache_line) std::atomic<std::int64_t> pending_{0};
    // error_state_ — slot guard: k_no_error → k_writing_error (winner's CAS)
    // → k_error_ready (release store after the payload write) → consumed
    // back to k_no_error (acquire CAS). See capture_error/consume_error_.
    std::atomic<int> error_state_{k_no_error};
    std::exception_ptr error_; // non-atomic; guarded by error_state_ + pending_ publish edge
};

} // namespace bracken::core::jobs
