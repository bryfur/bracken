#pragma once

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/task_group.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace bracken::core::jobs {

// Below this total, dispatch overhead exceeds the work — run serially on caller.
// Tuned conservatively: animation/cull loops with a few dozen entities should
// not pay the job-system tax. Callers that know the work is heavy can pass an
// explicit `grain` to bypass this.
inline constexpr std::size_t PARALLEL_MIN_TOTAL = 256;

// Fn signature: void(size_t chunk_begin, size_t chunk_end).
// Chunk form is used (not per-index) so tight loops avoid per-element call
// overhead. One code path at every scale: below the guards (small total,
// <= 1 worker, uninitialized pool) the whole range runs as ONE chunk via a
// direct call on the caller — granularity control, not a second code path.
//
// Chunks are submitted against a stack TaskGroup (no per-chunk handles, no
// allocation) as ONE batch — every chunk node is queued, then the pool is
// woken once (JobSystem::submit_batch) — and joined with
// JobSystem::wait(group), which helps execute chunks — fork-join safe even
// from inside a job on a 1-worker pool.
//
// Exception contract: ALL chunks are joined before anything propagates. fn
// typically captures caller stack state by reference, so unwinding while
// chunks are still queued or running would be use-after-scope in those
// chunks. The first captured exception is rethrown once every chunk has
// completed; subsequent exceptions are dropped. Prefer non-throwing bodies.
template<typename Fn>
void parallel_for(JobSystem& js, std::size_t begin, std::size_t end, Fn&& fn, std::size_t grain = 0) {
    if (end <= begin) {
        return;
    }
    const std::size_t total = end - begin;

    const unsigned workers = js.is_initialized() ? js.worker_count() : 1u;
    if (!js.is_initialized() || workers <= 1) {
        fn(begin, end);
        return;
    }

    // Auto-grain mode applies the small-workload guard. Explicit grain
    // bypasses it (caller is opting in to dispatch overhead).
    if (grain == 0) {
        if (total < PARALLEL_MIN_TOTAL) {
            fn(begin, end);
            return;
        }
        grain = std::max<std::size_t>(64, total / (static_cast<std::size_t>(workers) * 4));
    }

    if (total <= grain) {
        fn(begin, end);
        return;
    }

    // One node per chunk, but ONE work-epoch bump + wake for the whole
    // dispatch (JobSystem::submit_batch): the per-node bump+notify dominated
    // small-chunk dispatch. Chunk boundaries are unchanged — chunk c covers
    // [begin + c * grain, min(end, ...)), exactly the loop this replaced —
    // so every caller's chunk-index arithmetic and determinism contract
    // still holds.
    const std::size_t chunk_count = (total + grain - 1) / grain;
    TaskGroup group;
    try {
        js.submit_batch(group, chunk_count, [&](std::size_t c) {
            const std::size_t chunk_begin = begin + c * grain;
            const std::size_t chunk_end = std::min(end, chunk_begin + grain);
            // fn captured by reference: it lives in this frame, which the
            // unconditional join below keeps alive past every chunk.
            return [chunk_begin, chunk_end, &fn]() { fn(chunk_begin, chunk_end); };
        });
    } catch (...) {
        // A submit failed: join the chunks already queued before unwinding
        // (they reference `fn`/the caller's stack), then propagate the
        // submit failure, dropping any chunk error.
        try {
            js.wait(group);
        } catch (...) { // NOLINT(bugprone-empty-catch)
        }
        throw;
    }
    // Joins ALL chunks, then rethrows the group's first captured error.
    js.wait(group);
}

// Convenience overload servicing the canonical singleton pool — the form
// engine/host loops use. Thin delegator; identical semantics.
template<typename Fn>
void parallel_for(std::size_t begin, std::size_t end, Fn&& fn, std::size_t grain = 0) {
    parallel_for(JobSystem::instance(), begin, end, std::forward<Fn>(fn), grain);
}

} // namespace bracken::core::jobs
