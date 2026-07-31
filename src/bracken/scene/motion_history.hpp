#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <unordered_map>
#include <utility>

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
        if (motion_id == 0) {
            return nullptr;
        }
        auto it = prev_.find(motion_id);
        return it != prev_.end() ? &it->second : nullptr;
    }

    /// Record this frame's state for `motion_id` (ignored for id 0). Call once
    /// per submitted command; a duplicate id within a frame keeps the last
    /// recorded state.
    void record(uint32_t motion_id, const glm::mat4& model, uint32_t palette_offset = NO_PALETTE) {
        if (motion_id == 0) {
            return;
        }
        curr_[motion_id] = Entry{model, palette_offset};
    }

    /// Promote this frame's recorded state to "previous"; ids not re-recorded
    /// this frame are pruned.
    void end_frame() {
        std::swap(prev_, curr_);
        curr_.clear();
    }

    /// Number of ids with previous-frame state (i.e. recorded last frame).
    [[nodiscard]] size_t size() const { return prev_.size(); }

    void clear() {
        prev_.clear();
        curr_.clear();
    }

private:
    std::unordered_map<uint32_t, Entry> prev_;
    std::unordered_map<uint32_t, Entry> curr_;
};

} // namespace bracken::scene
