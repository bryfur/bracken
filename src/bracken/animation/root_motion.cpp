#include "bracken/animation/root_motion.hpp"

#include <cmath>

namespace bracken::animation {
namespace {

/// Position and rotation of one channel at one time, cursorless. Reports
/// whether the channel actually carries each path, because "no keys" and "keys
/// that happen to read as the rest value" must produce different answers.
struct RootSample {
    glm::vec3 translation{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    bool has_translation = false;
    bool has_rotation = false;
};

[[nodiscard]] RootSample sample_root(const AnimationChannel& ch, float t) noexcept {
    RootSample s;
    if (!ch.position_times.empty()) {
        s.translation = interpolate_keyframes(ch.position_times, ch.positions, t, ch.position_interp);
        s.has_translation = true;
    }
    if (!ch.rotation_times.empty()) {
        s.rotation = interpolate_keyframes(ch.rotation_times, ch.rotations, t, ch.rotation_interp);
        s.has_rotation = true;
    }
    return s;
}

[[nodiscard]] glm::vec3 masked(const glm::vec3& d, const RootMotionConfig& cfg) noexcept {
    return glm::vec3(cfg.translation_xz ? d.x : 0.0f, cfg.translation_y ? d.y : 0.0f,
                     cfg.translation_xz ? d.z : 0.0f);
}

} // namespace

glm::quat twist_about_y(const glm::quat& q) noexcept {
    // Swing-twist about +Y: project (x, y, z) onto the axis, keep w, renormalize.
    const float len_sq = q.w * q.w + q.y * q.y;
    if (len_sq < 1e-12f) {
        // The rotation is a pure 180-degree swing about an axis in XZ; it has no
        // well-defined yaw. Identity, not NaN.
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    }
    const float inv = 1.0f / std::sqrt(len_sq);
    return glm::quat(q.w * inv, 0.0f, q.y * inv, 0.0f);
}

RootMotionDelta extract_root_motion(const std::vector<AnimationClip>& clips, const ClipBindings& bindings,
                                    int clip_index, float t_prev, float t_now, bool wrapped,
                                    const RootMotionConfig& cfg) noexcept {
    RootMotionDelta delta;
    if (cfg.joint < 0) {
        return delta;
    }
    if (clip_index < 0 || static_cast<std::size_t>(clip_index) >= clips.size()) {
        return delta;
    }
    const auto channels = bindings.for_clip(clip_index);
    if (static_cast<std::size_t>(cfg.joint) >= channels.size()) {
        return delta;
    }
    const AnimationChannel* ch = channels[static_cast<std::size_t>(cfg.joint)];
    if (ch == nullptr) {
        return delta; // no channel at all — no motion, and NOT the rest pose
    }

    const float duration = clips[static_cast<std::size_t>(clip_index)].duration;

    auto accumulate = [&](float a, float b) {
        const RootSample sa = sample_root(*ch, a);
        const RootSample sb = sample_root(*ch, b);
        glm::vec3 dt(0.0f);
        glm::quat dr(1.0f, 0.0f, 0.0f, 0.0f);
        if (sa.has_translation && sb.has_translation) {
            dt = masked(sb.translation - sa.translation, cfg);
        }
        if (cfg.rotation_yaw && sa.has_rotation && sb.has_rotation) {
            dr = twist_about_y(sb.rotation) * glm::inverse(twist_about_y(sa.rotation));
        }
        return RootMotionDelta{dt, dr};
    };

    if (wrapped && duration > 0.0f) {
        const RootMotionDelta pre = accumulate(t_prev, duration);
        const RootMotionDelta post = accumulate(0.0f, t_now);
        delta.translation = pre.translation + post.translation;
        // Pre-wrap happens first, so it is the RIGHT factor of the composition.
        delta.rotation = post.rotation * pre.rotation;
        return delta;
    }

    return accumulate(t_prev, t_now);
}

void strip_root_motion(PoseSpan pose, ConstPoseSpan reference, const RootMotionConfig& cfg) noexcept {
    if (cfg.joint < 0) {
        return;
    }
    const std::size_t j = static_cast<std::size_t>(cfg.joint);
    if (j >= pose.size() || j >= reference.size()) {
        return;
    }

    if (cfg.translation_xz) {
        pose[j].translation.x = reference[j].translation.x;
        pose[j].translation.z = reference[j].translation.z;
    }
    if (cfg.translation_y) {
        pose[j].translation.y = reference[j].translation.y;
    }
    if (cfg.rotation_yaw) {
        pose[j].rotation = glm::inverse(twist_about_y(pose[j].rotation)) * pose[j].rotation;
    }
}

} // namespace bracken::animation
