#pragma once

#include "bracken/core/assert.hpp"

#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <utility>
#include <vector>

namespace bracken::scene {

/**
 * Previous-frame motion state for velocity (motion-vector) rendering.
 *
 * Scene commands are immediate-mode: nothing persists across frames, so
 * moving objects opt into motion history with a host-assigned, host-unique
 * `motion_id` (any nonzero uint32). Semantics:
 *
 * - motion_id 0 = "no history": the object is treated as static and its
 *   velocity is camera-only (previous model matrix == current model matrix).
 * - During frame N, find_prev() returns the state recorded during frame N-1
 *   (or nullptr if the id wasn't submitted last frame — also treated static).
 * - record() is called once per submitted command per frame; end_frame()
 *   promotes the recorded set to "previous" and prunes every id that was not
 *   re-submitted this frame.
 * - For skinned commands the entry also stores the frame's bone-palette
 *   offset. The palette GPU buffer is double-buffered by the SceneRenderer,
 *   so an offset recorded in frame N-1 indexes the *previous* palette buffer
 *   during frame N. NO_PALETTE marks entries without palette data (static
 *   models, or skinned draws dropped over the palette cap).
 *
 * Storage: two open-addressed tables (linear probing, power-of-two capacity)
 * rather than two std::unordered_maps. The access pattern is a per-frame
 * find + record for every motion-tracked command and then a full prune, which
 * is the worst case for a node-based map: every record() allocated a node and
 * every end_frame() destroyed the whole chain. Here a table slot carries a
 * frame STAMP, so "clear" is `++stamp` — slots from older frames read as
 * empty and get overwritten in place — and the tables' storage is allocated
 * once and reused forever.
 *
 * Threading: find_prev() is a pure read of the previous table and is called
 * from prep workers concurrently; record()/end_frame() mutate only the
 * current table and are single-threaded (sim/main). The two never overlap
 * within a frame.
 */
class MotionHistory {
public:
    static constexpr uint32_t NO_PALETTE = 0xFFFFFFFFu;

    struct Entry {
        glm::mat4 model{1.0f};
        uint32_t palette_offset = NO_PALETTE;
    };

    /// Previous-frame state for `motion_id`, or nullptr when the id is 0 or
    /// was not recorded last frame (both mean: treat as static this frame).
    [[nodiscard]] const Entry* find_prev(uint32_t motion_id) const {
        if (motion_id == 0 || prev_.count == 0) {
            return nullptr;
        }
        const size_t mask = prev_.slots.size() - 1;
        for (size_t i = slot_of(motion_id, prev_.shift);; i = (i + 1) & mask) {
            const Slot& s = prev_.slots[i];
            if (s.stamp != prev_.stamp) {
                return nullptr; // empty slot ends the probe: the id is absent
            }
            if (s.key == motion_id) {
                return &s.entry;
            }
        }
    }

    /// Record this frame's state for `motion_id` (ignored for id 0). Call once
    /// per submitted command; a duplicate id within a frame keeps the last
    /// recorded state.
    void record(uint32_t motion_id, const glm::mat4& model, uint32_t palette_offset = NO_PALETTE) {
        if (motion_id == 0) {
            return;
        }
        // Grow at 3/4 load so probe runs stay short (the table is never
        // shrunk: capacity is the steady-state motion-id count).
        if ((curr_.count + 1) * 4 >= curr_.slots.size() * 3) {
            grow(curr_);
        }
        const size_t mask = curr_.slots.size() - 1;
        for (size_t i = slot_of(motion_id, curr_.shift);; i = (i + 1) & mask) {
            Slot& s = curr_.slots[i];
            if (s.stamp != curr_.stamp) { // empty (or a slot from an older frame)
                s.stamp = curr_.stamp;
                s.key = motion_id;
                s.entry = Entry{model, palette_offset};
                ++curr_.count;
                return;
            }
            if (s.key == motion_id) {
                s.entry = Entry{model, palette_offset};
                return;
            }
        }
    }

    /// Promote this frame's recorded state to "previous"; ids not re-recorded
    /// this frame are pruned.
    void end_frame() {
        std::swap(prev_, curr_);
        clear_table(curr_);
    }

    /// Number of ids with previous-frame state (i.e. recorded last frame).
    [[nodiscard]] size_t size() const { return prev_.count; }

    void clear() {
        clear_table(prev_);
        clear_table(curr_);
    }

private:
    struct Slot {
        uint32_t key = 0;
        uint32_t stamp = 0; // == Table::stamp iff this slot holds a live entry
        Entry entry;
    };

    struct Table {
        // Power-of-two sized; a stamp of 0 never matches a live slot, so a
        // freshly resized table reads as all-empty without a memset.
        std::vector<Slot> slots{k_initial_capacity};
        // Cached hash shift for slots.size() (see slot_of); kept in step by
        // the two places that size the table (here and grow()).
        uint32_t shift = k_initial_shift;
        uint32_t stamp = 1;
        size_t count = 0;
    };

    static constexpr size_t k_initial_capacity = 64;
    static constexpr uint32_t k_initial_shift = 26; // 32 - log2(64)
    static_assert(size_t{1} << (32 - k_initial_shift) == k_initial_capacity,
                  "MotionHistory initial shift and capacity disagree");

    // Fibonacci hashing: multiply and take the HIGH bits. Motion ids are
    // host-assigned and typically dense per subsystem (base + index), which
    // the identity hash would map to one contiguous probe run.
    static size_t slot_of(uint32_t key, uint32_t shift) {
        return static_cast<size_t>((key * 2654435769u) >> shift);
    }

    // "Clear" = bump the stamp; every live slot instantly reads as stale.
    // Only a stamp wrap (2^32 frames — 2.3 years at 60 fps, but free to
    // handle) needs the O(capacity) reset.
    static void clear_table(Table& t) {
        t.count = 0;
        if (t.stamp == 0xFFFFFFFFu) {
            for (Slot& s : t.slots) {
                s.stamp = 0;
            }
            t.stamp = 1;
            return;
        }
        ++t.stamp;
    }

    static void grow(Table& t) {
        Table grown;
        grown.slots.assign(t.slots.size() * 2, Slot{});
        grown.shift = t.shift - 1; // capacity doubled -> one more index bit
        grown.stamp = t.stamp;
        for (const Slot& s : t.slots) {
            if (s.stamp != t.stamp) {
                continue;
            }
            const size_t mask = grown.slots.size() - 1;
            for (size_t i = slot_of(s.key, grown.shift);; i = (i + 1) & mask) {
                Slot& dst = grown.slots[i];
                if (dst.stamp != grown.stamp) {
                    dst = s;
                    ++grown.count;
                    break;
                }
            }
        }
        BRACKEN_ASSERT(grown.count == t.count, "MotionHistory rehash lost entries");
        t = std::move(grown);
    }

    Table prev_;
    Table curr_;
};

} // namespace bracken::scene
