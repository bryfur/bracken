#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mmo::engine::scene {

/**
 * Open-addressing hash table keyed on a bone-palette data pointer, used to
 * dedupe shared bone arrays during per-frame palette packing.
 *
 * Designed for the per-frame path:
 * - clear() is O(1): it bumps a generation counter instead of touching slots.
 * - No allocations after construction (slots are sized once, at 2x
 *   max_entries rounded up to a power of two, so probes stay short and the
 *   table can never fill up).
 * - Expected O(1) find/insert vs. the O(N) linear scan it replaces.
 */
class BonePaletteTable {
public:
    struct Entry {
        const void* key = nullptr;
        uint32_t offset = 0;      // matrix offset into the packed palette buffer
        uint32_t joint_count = 0; // matrices packed at that offset
    };

    explicit BonePaletteTable(uint32_t max_entries) : max_entries_(max_entries) {
        size_t capacity = 2;
        while (capacity < static_cast<size_t>(max_entries) * 2) {
            capacity *= 2;
        }
        slots_.resize(capacity);
        mask_ = capacity - 1;
    }

    /// O(1) invalidation of all entries (slots become stale, not rewritten).
    void clear() {
        ++generation_;
        count_ = 0;
    }

    uint32_t size() const { return count_; }

    /// Returns the live entry for `key` (mutable, so callers can repoint a
    /// palette at a larger repack), or nullptr when absent.
    Entry* find(const void* key) {
        size_t idx = probe_start(key);
        while (slots_[idx].generation == generation_) {
            if (slots_[idx].entry.key == key) {
                return &slots_[idx].entry;
            }
            idx = (idx + 1) & mask_;
        }
        return nullptr;
    }

    /// Inserts `key` (must not already be present — call find() first) and
    /// returns its entry, or nullptr when the table already holds max_entries.
    Entry* insert(const void* key, uint32_t offset, uint32_t joint_count) {
        if (key == nullptr || count_ >= max_entries_) {
            return nullptr;
        }
        size_t idx = probe_start(key);
        while (slots_[idx].generation == generation_) {
            idx = (idx + 1) & mask_;
        }
        Slot& slot = slots_[idx];
        slot.generation = generation_;
        slot.entry = {key, offset, joint_count};
        ++count_;
        return &slot.entry;
    }

private:
    struct Slot {
        Entry entry{};
        uint64_t generation = 0; // slot is live iff this matches generation_
    };

    size_t probe_start(const void* key) const {
        // Fibonacci hash of the pointer bits: the low bits of a pointer are
        // mostly zero (alignment), so multiply-mix before masking.
        const auto bits = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(key));
        return static_cast<size_t>((bits * 0x9E3779B97F4A7C15ull) >> 32) & mask_;
    }

    std::vector<Slot> slots_;
    uint64_t generation_ = 1; // starts above the Slot default so all slots begin stale
    size_t mask_ = 0;
    uint32_t max_entries_ = 0;
    uint32_t count_ = 0;
};

} // namespace mmo::engine::scene
