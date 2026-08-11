#include "bracken/animation/clip_sampler.hpp"

#include "bracken/core/profiler.hpp"

#include <algorithm>

namespace bracken::animation {

// Character-for-character the body of the file-static `sample_clip_joint` in
// animation_player.cpp. See this file's header comment before editing either.
void sample_clip_joint_into(const AnimationClip* clip,
                            std::span<const AnimationChannel* const> channels,
                            std::span<ChannelCursors> cursors, int joint_idx, float time,
                            const Joint& joint, glm::vec3& out_translation, glm::quat& out_rotation,
                            glm::vec3& out_scale) {
    out_translation = joint.local_translation;
    out_rotation = joint.local_rotation;
    out_scale = joint.local_scale;

    if (!clip) {
        return;
    }
    if (joint_idx < 0 || joint_idx >= static_cast<int>(channels.size())) {
        return;
    }
    const auto* ch = channels[static_cast<std::size_t>(joint_idx)];
    if (!ch) {
        return;
    }

    // The cursorless branch selects the same keyframe interval by binary search,
    // so the two paths agree bitwise. Only the cost differs.
    if (cursors.size() <= static_cast<std::size_t>(joint_idx)) {
        if (!ch->position_times.empty()) {
            out_translation = interpolate_keyframes(ch->position_times, ch->positions, time, ch->position_interp);
        }
        if (!ch->rotation_times.empty()) {
            out_rotation = interpolate_keyframes(ch->rotation_times, ch->rotations, time, ch->rotation_interp);
        }
        if (!ch->scale_times.empty()) {
            out_scale = interpolate_keyframes(ch->scale_times, ch->scales, time, ch->scale_interp);
        }
        return;
    }

    auto& cur = cursors[static_cast<std::size_t>(joint_idx)];

    if (!ch->position_times.empty()) {
        out_translation = interpolate_keyframes(ch->position_times, ch->positions, time, cur.pos, ch->position_interp);
    }
    if (!ch->rotation_times.empty()) {
        out_rotation = interpolate_keyframes(ch->rotation_times, ch->rotations, time, cur.rot, ch->rotation_interp);
    }
    if (!ch->scale_times.empty()) {
        out_scale = interpolate_keyframes(ch->scale_times, ch->scales, time, cur.scale, ch->scale_interp);
    }
}

void ClipBindings::build(const Skeleton& skeleton, const std::vector<AnimationClip>& clips) {
    joints_ = skeleton.joints.size();
    clips_ = clips.size();
    flat_.assign(joints_ * clips_, nullptr);
    for (std::size_t c = 0; c < clips_; ++c) {
        const std::size_t base = c * joints_;
        // Same walk order and same last-wins tie-break as
        // AnimationPlayer::compute_bone_matrices' channel_lookup_ build.
        for (const auto& ch : clips[c].channels) {
            if (ch.bone_index >= 0 && ch.bone_index < static_cast<int>(joints_)) {
                flat_[base + static_cast<std::size_t>(ch.bone_index)] = &ch;
            }
        }
    }
}

void ClipBindings::clear() noexcept {
    flat_.clear();
    joints_ = 0;
    clips_ = 0;
}

std::span<const AnimationChannel* const> ClipBindings::for_clip(int clip_index) const noexcept {
    if (clip_index < 0 || static_cast<std::size_t>(clip_index) >= clips_ || joints_ == 0) {
        return {};
    }
    return std::span<const AnimationChannel* const>(flat_.data() + static_cast<std::size_t>(clip_index) * joints_,
                                                    joints_);
}

void ClipSampler::bind(std::size_t joint_count, bool cache_cursors) {
    joints_ = joint_count;
    cache_cursors_ = cache_cursors;
    cached_clip_ = -1;
    if (cache_cursors) {
        cursors_.assign(joint_count, ChannelCursors{});
    } else {
        cursors_.clear();
        cursors_.shrink_to_fit();
    }
}

void ClipSampler::reset() noexcept {
    cached_clip_ = -1;
    std::fill(cursors_.begin(), cursors_.end(), ChannelCursors{});
}

void ClipSampler::sample(const Skeleton& skeleton, const std::vector<AnimationClip>& clips,
                         const ClipBindings& bindings, int clip_index, float time, PoseSpan out) {
    BRACKEN_PROFILE_ZONE("ClipSampler::sample");
    const std::size_t count = std::min(skeleton.joints.size(), out.size());
    if (count == 0) {
        return;
    }

    if (cache_cursors_ && clip_index != cached_clip_) {
        // Mirrors AnimationPlayer, which reassigns its cursor vector whenever
        // the clip changes rather than carrying stale hints into a new track.
        std::fill(cursors_.begin(), cursors_.end(), ChannelCursors{});
        cached_clip_ = clip_index;
    }

    const AnimationClip* clip = nullptr;
    if (clip_index >= 0 && clip_index < static_cast<int>(clips.size())) {
        clip = &clips[static_cast<std::size_t>(clip_index)];
    }
    const auto channels = bindings.for_clip(clip_index);

    const std::span<ChannelCursors> cursors(cursors_.data(), cursors_.size());
    for (std::size_t i = 0; i < count; ++i) {
        sample_clip_joint_into(clip, channels, cursors, static_cast<int>(i), time, skeleton.joints[i],
                               out[i].translation, out[i].rotation, out[i].scale);
    }
}

void sample_clip_pose(const Skeleton& skeleton, const std::vector<AnimationClip>& clips,
                      const ClipBindings& bindings, int clip_index, float time, PoseSpan out) {
    const std::size_t count = std::min(skeleton.joints.size(), out.size());
    if (count == 0) {
        return;
    }
    const AnimationClip* clip = nullptr;
    if (clip_index >= 0 && clip_index < static_cast<int>(clips.size())) {
        clip = &clips[static_cast<std::size_t>(clip_index)];
    }
    const auto channels = bindings.for_clip(clip_index);
    for (std::size_t i = 0; i < count; ++i) {
        sample_clip_joint_into(clip, channels, {}, static_cast<int>(i), time, skeleton.joints[i],
                               out[i].translation, out[i].rotation, out[i].scale);
    }
}

} // namespace bracken::animation
