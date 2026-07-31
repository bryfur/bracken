#pragma once

#include "bracken/core/jobs/cache_line.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// ThreadSanitizer does not model std::atomic_thread_fence, so the paper's
// {release fence; relaxed bottom store} publication edge in push() is
// invisible to it: every steal of a payload published through the fence
// reports as a false-positive data race (verified 2026-07-19 — the
// fence-load-bearing payload test trips TSan while being correct by the
// paper and by hardware). Under TSan we therefore compile push()'s
// publication as a release STORE on `bottom` — the SAME algorithm with a
// strictly stronger (still correct, slightly costlier) ordering that the
// tool models, keeping the TSan lane suppression-free. Production builds
// keep the paper's fence formulation. The seq_cst fences in pop()/steal()
// stay in both modes: they order atomic-atomic (the take/steal exclusion
// proof), carry no data publication, and their invisibility to TSan cannot
// produce false positives on data.
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define BRACKEN_TSAN_FENCE_WORKAROUND 1
#endif
#endif

namespace bracken::core::jobs {

// Chase–Lev work-stealing deque of pointers.
//
// Implemented faithfully from the C11 memory-model formulation in:
//   N. M. Lê, A. Pop, A. Cohen, F. Zappa Nardelli,
//   "Correct and Efficient Work-Stealing for Weak Memory Models",
//   PPoPP 2013 — Figure 1 (take/push/steal and their proven orderings).
// Every atomic operation below carries the ordering from that listing; each
// fence/ordering is annotated with its role in Figure 1. Two deliberate,
// semantics-preserving deviations:
//   * indices are std::int64_t instead of the listing's size_t, so the
//     transient `bottom - 1` in pop() on an empty deque cannot wrap
//     (`t <= b` stays a correct emptiness test);
//   * the thief's memory_order_consume load of the array pointer is
//     strengthened to acquire (consume is deprecated; acquire is strictly
//     stronger, so the proof carries over).
//
// Ownership contract: exactly ONE owner thread calls push()/pop() (bottom
// end); any number of thief threads call steal() (top end, CAS on `top`).
// The only contended operation is a steal — the rare path of a well-grained
// workload (docs/multithreading_plan.md §7).
//
// Growth: the owner grows the circular array on full (push side). Old arrays
// cannot be freed while a stale thief may still be reading through them, so
// they are retired to an owner-only list and reclaimed in the destructor —
// growth is rare (capacity doubles), so the retained memory is bounded and
// tiny. The destructor must only run when no thief is active (quiescence is
// the caller's contract — the JobSystem destroys deques after joining all
// workers).
template<typename T>
class ChaseLevDeque {
public:
    // `initial_capacity` is rounded up to a power of two (minimum 2).
    explicit ChaseLevDeque(std::size_t initial_capacity = 256) {
        std::size_t cap = 2;
        while (cap < initial_capacity) {
            cap *= 2;
        }
        buffers_.push_back(std::make_unique<Buffer>(static_cast<std::int64_t>(cap)));
        array_.store(buffers_.back().get(), std::memory_order_relaxed);
    }

    ChaseLevDeque(const ChaseLevDeque&) = delete;
    ChaseLevDeque& operator=(const ChaseLevDeque&) = delete;
    ChaseLevDeque(ChaseLevDeque&&) = delete;
    ChaseLevDeque& operator=(ChaseLevDeque&&) = delete;
    ~ChaseLevDeque() = default; // frees current + retired buffers; caller guarantees quiescence

    // Owner only. Figure 1, push().
    void push(T* item) {
        const std::int64_t b = bottom_.load(std::memory_order_relaxed);
        // Acquire on top: pairs with the thieves' seq_cst CAS on `top` so the
        // owner's fullness check sees a top no staler than the last steal
        // (Fig. 1 push: load_explicit(&q->top, acquire)).
        const std::int64_t t = top_.load(std::memory_order_acquire);
        Buffer* a = array_.load(std::memory_order_relaxed); // owner wrote it; no ordering needed

        if (b - t > a->capacity - 1) {
            a = grow_(a, b, t); // full — owner-only resize
        }

        a->slots[b & a->mask].store(item, std::memory_order_relaxed);
        // Release fence before publishing the new bottom: makes the slot
        // store above (and the item's payload) visible to a thief that
        // acquire-loads `bottom` in steal() (Fig. 1 push: thread_fence(release)
        // before the bottom store). Under TSan the equivalent-but-tool-visible
        // release store is used instead — see the header note.
#if BRACKEN_TSAN_FENCE_WORKAROUND
        bottom_.store(b + 1, std::memory_order_release);
#else
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);
#endif
    }

    // Owner only. Figure 1, take(). Returns nullptr when the deque is empty
    // or the owner lost the race for the single last element.
    T* pop() {
        const std::int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        Buffer* a = array_.load(std::memory_order_relaxed);
        bottom_.store(b, std::memory_order_relaxed);
        // The load-store barrier of the algorithm: the owner's speculative
        // bottom decrement must be globally ordered against the thieves'
        // top reads (Fig. 1 take: thread_fence(seq_cst) between the bottom
        // store and the top load).
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::int64_t t = top_.load(std::memory_order_relaxed);

        T* x = nullptr;
        if (t <= b) {
            // Non-empty.
            x = a->slots[b & a->mask].load(std::memory_order_relaxed);
            if (t == b) {
                // Single last element: race the thieves with the same
                // seq_cst CAS on top they use (Fig. 1 take:
                // compare_exchange_strong_explicit(seq_cst, relaxed)).
                if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                                  std::memory_order_relaxed)) {
                    x = nullptr; // lost the race
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
            }
        } else {
            // Empty: undo the speculative decrement.
            bottom_.store(b + 1, std::memory_order_relaxed);
        }
        return x;
    }

    // Any thread. Figure 1, steal(). Returns nullptr both when the deque is
    // empty and when the CAS race for the top element was lost (the paper's
    // EMPTY and ABORT) — callers treat both as "look elsewhere and retry".
    T* steal() {
        // Acquire on top: synchronizes with the seq_cst CAS of the previous
        // successful steal/take (Fig. 1 steal: load_explicit(&q->top,
        // acquire)).
        std::int64_t t = top_.load(std::memory_order_acquire);
        // Orders the top load before the bottom load — the mirror of the
        // owner's fence in take() (Fig. 1 steal: thread_fence(seq_cst)).
        std::atomic_thread_fence(std::memory_order_seq_cst);
        // Acquire on bottom: pairs with the owner's release fence in push()
        // so the slot written for index t is visible below (Fig. 1 steal:
        // load_explicit(&q->bottom, acquire)).
        const std::int64_t b = bottom_.load(std::memory_order_acquire);

        T* x = nullptr;
        if (t < b) {
            // Non-empty. Paper uses consume for the array load; strengthened
            // to acquire (see header comment). Pairs with the release store
            // of `array_` in grow_().
            Buffer* a = array_.load(std::memory_order_acquire);
            x = a->slots[t & a->mask].load(std::memory_order_relaxed);
            // Claim element t (Fig. 1 steal:
            // compare_exchange_strong_explicit(seq_cst, relaxed)).
            if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                              std::memory_order_relaxed)) {
                return nullptr; // ABORT: lost the race
            }
        }
        return x;
    }

    // Approximate (racy) emptiness hint — safe from any thread, may be stale
    // in either direction. Used only for drain/park heuristics, never for
    // correctness.
    bool empty() const noexcept {
        return bottom_.load(std::memory_order_relaxed) <= top_.load(std::memory_order_relaxed);
    }

    // Current capacity (owner thread only — reads the unpublished array).
    std::size_t capacity() const noexcept {
        return static_cast<std::size_t>(array_.load(std::memory_order_relaxed)->capacity);
    }

private:
    struct Buffer {
        explicit Buffer(std::int64_t cap)
            : capacity(cap), mask(cap - 1), slots(std::make_unique<std::atomic<T*>[]>(static_cast<std::size_t>(cap))) {}
        const std::int64_t capacity; // power of two
        const std::int64_t mask;
        std::unique_ptr<std::atomic<T*>[]> slots;
    };

    // Owner only: double the array, copy live elements [t, b), publish.
    Buffer* grow_(Buffer* old, std::int64_t b, std::int64_t t) {
        auto next = std::make_unique<Buffer>(old->capacity * 2);
        for (std::int64_t i = t; i != b; ++i) {
            next->slots[i & next->mask].store(old->slots[i & old->mask].load(std::memory_order_relaxed),
                                              std::memory_order_relaxed);
        }
        Buffer* raw = next.get();
        buffers_.push_back(std::move(next)); // old buffer retired, freed in ~ChaseLevDeque
        // Release: publishes the copied slots to thieves that acquire-load
        // `array_` in steal() (the paper's resize publication edge).
        array_.store(raw, std::memory_order_release);
        return raw;
    }

    // top — the steal index. Contended by all thieves (seq_cst CAS) and read
    // by the owner; padded onto its own cache line so thief traffic does not
    // false-share with `bottom`. Ordering protocol: see the per-op comments
    // above (Figure 1 of Lê et al. 2013).
    alignas(k_cache_line) std::atomic<std::int64_t> top_{0};
    // bottom — the owner's push/pop index. Written relaxed by the owner
    // (publication happens via the release fence in push()), acquire-read by
    // thieves. Own cache line: owner-hot.
    alignas(k_cache_line) std::atomic<std::int64_t> bottom_{0};
    // array — current circular buffer. Release-stored by the owner on grow,
    // acquire-loaded by thieves (promoted consume); relaxed for the owner's
    // own loads.
    alignas(k_cache_line) std::atomic<Buffer*> array_{nullptr};

    // Owner-only retire list (also owns the current buffer). Reclaimed at
    // destruction, when the caller guarantees no thief is active.
    std::vector<std::unique_ptr<Buffer>> buffers_;
};

} // namespace bracken::core::jobs
