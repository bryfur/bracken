#pragma once

#include "bracken/core/jobs/cache_line.hpp"

#include <atomic>

namespace bracken::core::jobs {

// Vyukov intrusive MPSC queue (non-blocking producers, single consumer).
//
// Reference algorithm: Dmitry Vyukov, "Intrusive MPSC node-based queue"
// (1024cores.net). Producers push with one wait-free exchange + one store;
// the consumer pops by chasing `next` links through an embedded stub node.
//
// Requirements on T: default-constructible, with a member
// `std::atomic<T*> next` used exclusively by this queue while the node is
// enqueued (the JobSystem reuses the same field for its pool freelist — a
// node is never in both places at once).
//
// Contracts:
//   * push(): any thread, any time.
//   * pop(): exactly ONE consumer at a time (the JobSystem elects one via a
//     token; see JobSystem::try_consume_injection_). `tail_` is plain data —
//     concurrent pops are a data race by construction.
//   * pop() may return nullptr while the queue is logically non-empty, when
//     a producer is stalled inside the two-step push window (head swung, next
//     not yet linked). Callers must treat nullptr as "retry later", never as
//     a proof of emptiness — the JobSystem's work-epoch protocol (a producer
//     bumps the epoch only after its push fully completes) guarantees such a
//     job is never lost across a park.
template<typename T>
class MpscQueue {
public:
    MpscQueue() {
        stub_.next.store(nullptr, std::memory_order_relaxed);
        head_.store(&stub_, std::memory_order_relaxed);
        tail_ = &stub_;
    }

    MpscQueue(const MpscQueue&) = delete;
    MpscQueue& operator=(const MpscQueue&) = delete;
    MpscQueue(MpscQueue&&) = delete;
    MpscQueue& operator=(MpscQueue&&) = delete;
    ~MpscQueue() = default; // does not own nodes; caller drains first

    // Producer side — any thread. Two steps: swing head, then link the
    // previous head to us. Between the two steps the queue is temporarily
    // "cut" — pop() observes this and returns nullptr (see class comment).
    void push(T* node) noexcept {
        node->next.store(nullptr, std::memory_order_relaxed);
        // acq_rel: release publishes the node's payload (written before
        // push) to the consumer that will traverse to it; acquire orders us
        // after the previous producer's exchange so the intrusive chain is
        // consistent.
        T* prev = head_.exchange(node, std::memory_order_acq_rel);
        // Release: publishes `node` (and its payload) to the consumer's
        // acquire load of prev->next in pop(). This completes the push.
        prev->next.store(node, std::memory_order_release);
    }

    // Consumer side — single consumer only. Returns the oldest node, or
    // nullptr (empty, or a producer is mid-push).
    T* pop() noexcept {
        T* tail = tail_;
        // Acquire: pairs with the producers' release store of `next` in
        // push() — makes the returned node's payload visible.
        T* next = tail->next.load(std::memory_order_acquire);

        if (tail == &stub_) {
            if (next == nullptr) {
                return nullptr; // empty (or first real push not yet linked)
            }
            // Skip the stub.
            tail_ = next;
            tail = next;
            next = next->next.load(std::memory_order_acquire);
        }
        if (next != nullptr) {
            tail_ = next;
            return tail;
        }

        // tail is the last linked node. If it is also the last pushed node,
        // re-push the stub behind it so we can detach it; otherwise a
        // producer is mid-push — report "nothing poppable right now".
        // Acquire pairs with producers' head exchange (acq_rel).
        T* head = head_.load(std::memory_order_acquire);
        if (tail != head) {
            return nullptr; // producer mid-push between exchange and link
        }
        push(&stub_);
        next = tail->next.load(std::memory_order_acquire);
        if (next != nullptr) {
            tail_ = next;
            return tail;
        }
        // A producer raced our stub re-push and is mid-link; retry later.
        return nullptr;
    }

private:
    // head — the producers' contention point (exchange per push). Own cache
    // line so producer traffic doesn't false-share with the consumer's tail.
    // Ordering: acq_rel exchange in push() ↔ acquire load in pop().
    alignas(k_cache_line) std::atomic<T*> head_{nullptr};
    // tail — consumer-private cursor. Plain (non-atomic by protocol): only
    // the single elected consumer touches it; the hand-off between
    // consecutive consumers is ordered by the caller's consumer token
    // (release on token clear ↔ acquire on token take).
    alignas(k_cache_line) T* tail_ = nullptr;
    T stub_; // sentinel; never handed out
};

} // namespace bracken::core::jobs
