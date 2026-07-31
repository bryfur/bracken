#pragma once

#include "bracken/core/assert.hpp"
#include "bracken/scene/frame_packet.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace bracken::scene {

/**
 * PacketRing — fixed ring of FramePackets implementing the frame-pipeline
 * handoff (docs/multithreading_plan.md §4.2, §7): the fill stage acquires the
 * next retired packet, builds + seals it through a FrameBuilder, and
 * publishes it with ONE release store; the consuming stage takes it with ONE
 * acquire exchange. retire() resets the packet (arena memory retained) so the
 * slot cycles back to the fill side. That publish + acquire pair is the
 * entire cross-stage synchronization cost of a frame.
 *
 * Contract: SPSC — one producing stage per frame (acquire_for_fill →
 * FrameBuilder → seal → publish) and one consuming stage per frame
 * (try_acquire_sealed → const reads → retire). The stages may live on
 * different threads; each individual call is made by one thread at a time.
 * Depth 2 covers today's sequential fill→render alternation and the Phase-4
 * Simulate(k+1) ∥ Render(k) overlap; the depth parameter makes deeper
 * pipelining a constant, not a rewrite.
 *
 * Backpressure is the caller's job: the ring never blocks. Before filling,
 * check `peek_fill().state() == FramePacket::State::Retired` (asserted in
 * acquire_for_fill) — with depth D the fill side may run at most D-1 frames
 * ahead of retirement. Owner-thread safety of the underlying arenas rides the
 * Phase-1 Arena ownership asserts: FramePacket::reset() rebinds the arena to
 * the retiring thread, and the next fill's first block reservation binds the
 * filling side.
 *
 * Sequential (ring-less style) use may skip publish/try_acquire_sealed and
 * read the packet right after seal(); retire() accepts Sealed as well as
 * Consuming for exactly that flow.
 */
class PacketRing {
public:
    explicit PacketRing(uint32_t depth = 2, std::size_t initial_arena_bytes = FramePacket::k_default_arena_bytes) {
        BRACKEN_VERIFY(depth >= 1, "PacketRing: depth must be >= 1");
        packets_.reserve(depth);
        for (uint32_t i = 0; i < depth; ++i) {
            packets_.push_back(std::make_unique<FramePacket>(initial_arena_bytes));
        }
    }

    PacketRing(const PacketRing&) = delete;
    PacketRing& operator=(const PacketRing&) = delete;
    PacketRing(PacketRing&&) = delete;
    PacketRing& operator=(PacketRing&&) = delete;

    [[nodiscard]] uint32_t depth() const noexcept { return static_cast<uint32_t>(packets_.size()); }

    /** Slot-indexed access (tests / stats). Does not affect the ring state. */
    [[nodiscard]] FramePacket& packet(uint32_t slot) {
        BRACKEN_ASSERT(slot < packets_.size(), "PacketRing::packet: slot {} out of range", slot);
        return *packets_[slot];
    }

    /** The packet the next acquire_for_fill() will return, without advancing —
     *  poll `peek_fill().state() == Retired` for backpressure. Fill side only. */
    [[nodiscard]] FramePacket& peek_fill() { return *packets_[fill_cursor_]; }

    /**
     * Take the next fill slot (round-robin). The returned packet is still
     * Retired — constructing a FrameBuilder over it performs the
     * Retired → Filling transition. Asserts the slot has been retired (i.e.
     * the pipeline is not overrunning the ring depth). Fill side only.
     */
    [[nodiscard]] FramePacket& acquire_for_fill() {
        FramePacket& p = *packets_[fill_cursor_];
        BRACKEN_ASSERT(p.state() == FramePacket::State::Retired,
                       "PacketRing::acquire_for_fill: slot {} not Retired (pipeline overrun — retire packets before "
                       "running {} frames ahead)",
                       fill_cursor_, packets_.size());
        fill_cursor_ = (fill_cursor_ + 1) % static_cast<uint32_t>(packets_.size());
        return p;
    }

    /**
     * Publish a sealed packet to the consuming stage: the single release
     * store of the frame handoff. Asserts the packet is Sealed and the
     * previous publication was taken (SPSC mailbox). Fill side only.
     */
    void publish(FramePacket& packet) {
        BRACKEN_ASSERT(packet.state() == FramePacket::State::Sealed, "PacketRing::publish: packet not Sealed");
        BRACKEN_ASSERT(sealed_slot_.load(std::memory_order_relaxed) == k_no_slot,
                       "PacketRing::publish: previous sealed packet not yet acquired (SPSC overrun)");
        sealed_slot_.store(slot_of(packet), std::memory_order_release);
    }

    /** True when a published packet is waiting to be acquired. Either side
     *  may poll it (relaxed load — a scheduling hint, e.g. fill-side
     *  backpressure before publish(); try_acquire_sealed() performs the real
     *  acquire). */
    [[nodiscard]] bool has_sealed() const noexcept {
        return sealed_slot_.load(std::memory_order_relaxed) != k_no_slot;
    }

    /**
     * Take the most recently published packet (Sealed → Consuming), or
     * nullptr when nothing is pending: the single acquire of the frame
     * handoff. Consume side only.
     */
    [[nodiscard]] FramePacket* try_acquire_sealed() {
        const uint32_t slot = sealed_slot_.exchange(k_no_slot, std::memory_order_acquire);
        if (slot == k_no_slot) {
            return nullptr;
        }
        FramePacket& p = *packets_[slot];
        p.begin_consume();
        return &p;
    }

    /**
     * Return a consumed packet to the ring: resets its arena (memory
     * retained) and transitions it to Retired for the fill side to reuse.
     * Accepts Sealed too, for the sequential no-publish flow. Consume side
     * only; must happen-before the fill side's next acquire of this slot
     * (the caller's frame fence).
     */
    void retire(FramePacket& packet) {
        [[maybe_unused]] const FramePacket::State s = packet.state();
        BRACKEN_ASSERT(s == FramePacket::State::Consuming || s == FramePacket::State::Sealed,
                       "PacketRing::retire: packet not Consuming/Sealed (state {})", static_cast<uint32_t>(s));
        packet.reset();
    }

private:
    [[nodiscard]] uint32_t slot_of(const FramePacket& packet) const {
        for (uint32_t i = 0; i < packets_.size(); ++i) {
            if (packets_[i].get() == &packet) {
                return i;
            }
        }
        BRACKEN_VERIFY(false, "PacketRing: packet does not belong to this ring");
        return 0; // unreachable
    }

    static constexpr uint32_t k_no_slot = 0xFFFFFFFFu;

    std::vector<std::unique_ptr<FramePacket>> packets_;

    // fill_cursor_ — next fill slot (round-robin). Fill side only, plain
    // non-atomic: a single producing stage advances it.
    uint32_t fill_cursor_ = 0;

    // sealed_slot_ — the single cross-stage publish word (§7: one publish and
    // one acquire per frame). Producer store(release, slot) in publish(),
    // sequenced after the packet's own Filling→Sealed edge, hands the whole
    // sealed packet across; consumer exchange(k_no_slot, acquire) in
    // try_acquire_sealed() takes it. k_no_slot = empty mailbox. SPSC by
    // contract, so store/exchange never race with themselves; the debug
    // relaxed load in publish() only backs the overrun assert.
    std::atomic<uint32_t> sealed_slot_{k_no_slot};
};

} // namespace bracken::scene
