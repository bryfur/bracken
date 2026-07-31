#pragma once

#include "bracken/animation/animation_types.hpp"
#include <array>
#include <span>
#include <vector>

namespace bracken::animation {

class AnimationPlayer {
public:
    AnimationPlayer() { reset(); }

    // Move-only by design: a copied player sharing keyframe cursors with the
    // original was a historical source of animation glitches.
    AnimationPlayer(const AnimationPlayer&) = delete;
    AnimationPlayer& operator=(const AnimationPlayer&) = delete;
    AnimationPlayer(AnimationPlayer&&) noexcept = default;
    AnimationPlayer& operator=(AnimationPlayer&&) noexcept = default;

    // Precondition: skeleton.joints must be ordered parents-before-children
    // (every parent_index < its joint's index) — the hierarchy is propagated
    // in a single index-order pass. Skeletons produced by ModelLoader::load_glb
    // are normalized to this order at load.
    void update(const Skeleton& skeleton, const std::vector<AnimationClip>& clips, float dt);

    void crossfade_to(int clip_index, float duration = 0.2f);

    void reset();

    void set_speed(float v) { speed_ = v; }
    float speed() const { return speed_; }

    void set_playing(bool v) { playing_ = v; }
    bool is_playing() const { return playing_; }

    void set_loop(bool v) { loop_ = v; }
    bool loops() const { return loop_; }

    void seek(float t) { time_ = t; }
    float current_time() const { return time_; }

    int current_clip() const { return current_clip_; }
    int prev_clip() const { return prev_clip_; }
    float prev_time() const { return prev_time_; }
    float blend_factor() const { return blend_factor_; }
    float blend_duration() const { return blend_duration_; }

    std::span<const glm::mat4> bone_matrices() const {
        return std::span<const glm::mat4>(bone_matrices_.data(), bone_matrices_.size());
    }

    // Joints written by the last update(); slots at/beyond this index hold
    // identity. Consumers packing palettes should read only this prefix.
    size_t used_joint_count() const { return used_joint_count_; }

    /**
     * Copy the current bone palette into caller-provided storage — e.g. the
     * uninitialized packet memory returned by
     * scene::FrameBuilder::alloc_bone_palette (docs/multithreading_plan.md
     * §3.2): the used_joint_count() prefix of bone_matrices(), then identity
     * for any remaining slots, so `dst` is fully written regardless of its
     * size relative to the animated prefix. Call after update() (and any
     * IK/lean patching) so the copy captures the final pose.
     *
     * Thread-safety tier: `sim-stage` — const read of per-instance state;
     * safe from a sim job as long as it doesn't race this player's update().
     */
    void write_palette(std::span<glm::mat4> dst) const;

    // IK/lean integrators need to read (and occasionally patch) world transforms
    // together with the bone matrices. Exposed as spans so callers never own the
    // underlying storage.
    std::span<glm::mat4> mutable_bone_matrices() {
        return std::span<glm::mat4>(bone_matrices_.data(), bone_matrices_.size());
    }
    std::span<glm::mat4> mutable_world_transforms() {
        return std::span<glm::mat4>(world_transforms_.data(), world_transforms_.size());
    }
    std::span<const glm::mat4> world_transforms() const {
        return std::span<const glm::mat4>(world_transforms_.data(), world_transforms_.size());
    }

private:
    void compute_bone_matrices(const Skeleton& skeleton, const std::vector<AnimationClip>& clips);

    int current_clip_ = 0;
    float time_ = 0.0f;
    bool playing_ = true;
    bool loop_ = true;
    float speed_ = 1.0f;

    int prev_clip_ = -1;
    float prev_time_ = 0.0f;
    float blend_factor_ = 1.0f;
    float blend_duration_ = 0.2f;

    std::array<glm::mat4, MAX_BONES> bone_matrices_;
    std::array<glm::mat4, MAX_BONES> world_transforms_;
    size_t used_joint_count_ = 0; // slots at/beyond this index are identity

    std::vector<const AnimationChannel*> channel_lookup_;
    std::vector<const AnimationChannel*> prev_channel_lookup_;
    std::vector<ChannelCursors> current_cursors_;
    std::vector<ChannelCursors> prev_cursors_;
    std::vector<glm::mat4> local_transforms_;
    int cached_clip_index_ = -1;
    int cached_prev_clip_index_ = -1;
};

} // namespace bracken::animation
