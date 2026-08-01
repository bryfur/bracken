#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace bracken::audio {

/**
 * Bounded lock-free MPSC ring for audio commands.
 *
 * The audio device callback runs on a thread the engine does not own and
 * cannot make wait: if it blocks, the device underruns and the player hears a
 * click or a dropout. So the game→audio path may not use a mutex, may not
 * allocate, and may not be unbounded. This is the primitive that enforces
 * that — a fixed array of slots, each with a sequence number.
 *
 * It is NOT the engine's job-system MPSC queue (core/jobs/mpsc_queue.hpp):
 * that one is intrusive and node-based, so producing requires owning a node,
 * which means allocating. Here the payload is copied into a pre-existing slot.
 *
 * Overflow drops the NEWEST command and reports it, rather than blocking or
 * overwriting an unread one. A dropped sound effect is a missing sound; an
 * overwritten one could be a missing note-off, which is a stuck note that
 * rings forever. Losing the new thing is always the safer failure.
 *
 * Reference: Dmitry Vyukov's bounded MPMC queue, narrowed to a single
 * consumer (the audio callback) because that is all this needs.
 *
 * Tier: push() is worker-safe (any thread, any time). pop() has exactly one
 * consumer — the audio callback.
 */
template <typename T, std::size_t Capacity>
class CommandRing {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
    static_assert(Capacity >= 2, "Capacity must leave room for at least one item");

public:
    CommandRing() {
        for (std::size_t i = 0; i < Capacity; ++i) {
            slots_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    /** Returns false when the ring is full; the command was NOT enqueued. */
    [[nodiscard]] bool push(const T& item) noexcept {
        Slot* slot = nullptr;
        std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            slot = &slots_[pos & k_mask];
            const std::size_t seq = slot->sequence.load(std::memory_order_acquire);
            const auto diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);
            if (diff == 0) {
                // Slot is free and ours if we win the claim.
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                dropped_.fetch_add(1, std::memory_order_relaxed);
                return false; // full
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
        slot->value = item;
        // Release: publishes the payload write above to the consumer's
        // acquire on this same sequence field.
        slot->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    /** Single consumer only. Returns false when empty. */
    [[nodiscard]] bool pop(T& out) noexcept {
        Slot* slot = nullptr;
        std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            slot = &slots_[pos & k_mask];
            const std::size_t seq = slot->sequence.load(std::memory_order_acquire);
            const auto diff =
                static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos + 1);
            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; // empty
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
        out = slot->value;
        slot->sequence.store(pos + k_mask + 1, std::memory_order_release);
        return true;
    }

    /** Commands lost to a full ring since startup. Surfaced in the audio
     *  diagnostics rather than silently swallowed — a nonzero value here
     *  means the game is out-producing the mixer and should be seen. */
    [[nodiscard]] uint64_t dropped() const noexcept {
        return dropped_.load(std::memory_order_relaxed);
    }

private:
    static constexpr std::size_t k_mask = Capacity - 1;

    struct Slot {
        std::atomic<std::size_t> sequence{0};
        T value{};
    };

    std::array<Slot, Capacity> slots_{};
    std::atomic<std::size_t> enqueue_pos_{0};
    std::atomic<std::size_t> dequeue_pos_{0};
    std::atomic<uint64_t> dropped_{0};
};

} // namespace bracken::audio
