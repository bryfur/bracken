#include "bracken/animation/pose.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/quaternion.hpp>

namespace bracken::animation {
namespace {

/// The per-joint effective weight, clamped to [0, 1]. Factored so the two blend
/// operators cannot disagree about what a mask or a feather means.
///
/// THE CLAMP IS WRITTEN OUT RATHER THAN `std::clamp` ON PURPOSE. `std::clamp(v,
/// 0, 1)` is `v < lo ? lo : (hi < v ? hi : v)`, and every comparison against a
/// NaN is false — so a NaN weight falls straight through the clamp and into
/// `slerp`, which turns the WHOLE SKELETON into NaN for the rest of the run.
/// `set_layer_weight` takes an unvalidated float from the host, so that is one
/// bad division in gameplay code away. `!(w > 0)` is true for NaN, so NaN and
/// every negative weight both read as "this layer does not touch this joint",
/// which is the only answer that degrades instead of corrupting.
[[nodiscard]] inline float joint_weight(const BlendParams& params, std::size_t j) noexcept {
    float w = params.weight;
    if (params.mask != nullptr && !params.mask->test(static_cast<int>(j))) {
        return 0.0f;
    }
    if (!params.joint_weights.empty()) {
        w *= (j < params.joint_weights.size()) ? params.joint_weights[j] : 0.0f;
    }
    if (!(w > 0.0f)) {
        return 0.0f; // NaN, -0, and every negative weight
    }
    return (w > 1.0f) ? 1.0f : w;
}

/// Componentwise src/ref with a zero-reference guard. A reference scale
/// component of 0 has no meaningful ratio, and 1 ("no additive scale on this
/// axis") is the only answer that keeps the result finite.
[[nodiscard]] inline glm::vec3 safe_scale_ratio(const glm::vec3& src, const glm::vec3& ref) noexcept {
    constexpr float k_eps = 1e-8f;
    return glm::vec3(std::abs(ref.x) < k_eps ? 1.0f : src.x / ref.x,
                     std::abs(ref.y) < k_eps ? 1.0f : src.y / ref.y,
                     std::abs(ref.z) < k_eps ? 1.0f : src.z / ref.z);
}

} // namespace

void BoneMask::set_subtree(const Skeleton& skeleton, int root, bool on) noexcept {
    const std::size_t joint_count = std::min(skeleton.joints.size(), static_cast<std::size_t>(MAX_BONES));
    if (root < 0 || static_cast<std::size_t>(root) >= joint_count) {
        return;
    }

    // Descendants are computed into a LOCAL mask so an unrelated bit already
    // set in *this cannot seed the propagation. See the header for why that
    // matters.
    BoneMask sub;
    sub.set(root, true);
    for (std::size_t i = static_cast<std::size_t>(root) + 1; i < joint_count; ++i) {
        const int parent = skeleton.joints[i].parent_index;
        if (parent >= root && parent < static_cast<int>(i) && sub.test(parent)) {
            sub.set(static_cast<int>(i), true);
        }
    }

    if (on) {
        *this |= sub;
    } else {
        *this &= sub.inverted();
    }
}

void feather_by_depth(const Skeleton& skeleton, int root, int falloff, std::span<float> out) noexcept {
    const std::size_t joint_count =
        std::min({skeleton.joints.size(), out.size(), static_cast<std::size_t>(MAX_BONES)});
    if (joint_count == 0) {
        return;
    }

    // -1 == outside the subtree. Depth is measured from the root, which is
    // depth 0 and therefore the LIGHTEST weight of the ramp.
    std::array<std::int16_t, MAX_BONES> depth{};
    for (std::size_t i = 0; i < joint_count; ++i) {
        depth[i] = -1;
        out[i] = 0.0f;
    }
    if (root < 0 || static_cast<std::size_t>(root) >= joint_count) {
        return;
    }
    depth[static_cast<std::size_t>(root)] = 0;

    for (std::size_t i = static_cast<std::size_t>(root) + 1; i < joint_count; ++i) {
        const int parent = skeleton.joints[i].parent_index;
        if (parent >= root && parent < static_cast<int>(i) && depth[static_cast<std::size_t>(parent)] >= 0) {
            depth[i] = static_cast<std::int16_t>(depth[static_cast<std::size_t>(parent)] + 1);
        }
    }

    const float span = static_cast<float>(std::max(falloff, 0));
    for (std::size_t i = 0; i < joint_count; ++i) {
        if (depth[i] < 0) {
            continue;
        }
        out[i] = (span <= 0.0f) ? 1.0f : std::min(1.0f, static_cast<float>(depth[i]) / span);
    }
}

void rest_pose(const Skeleton& skeleton, PoseSpan out) noexcept {
    const std::size_t n = std::min(skeleton.joints.size(), out.size());
    for (std::size_t i = 0; i < n; ++i) {
        const Joint& joint = skeleton.joints[i];
        out[i].translation = joint.local_translation;
        out[i].rotation = joint.local_rotation;
        out[i].scale = joint.local_scale;
    }
}

void blend_override(PoseSpan dst, ConstPoseSpan src, const BlendParams& params) noexcept {
    const std::size_t n = std::min(dst.size(), src.size());
    for (std::size_t i = 0; i < n; ++i) {
        const float w = joint_weight(params, i);
        if (w <= 0.0f) {
            continue; // bitwise untouched — the "weight 0 is exactly the base" contract
        }
        if (w >= 1.0f) {
            dst[i] = src[i]; // exact, not slerp(.., 1.0f), which is only approximate
            continue;
        }
        dst[i].translation = glm::mix(dst[i].translation, src[i].translation, w);
        dst[i].rotation = glm::slerp(dst[i].rotation, src[i].rotation, w);
        dst[i].scale = glm::mix(dst[i].scale, src[i].scale, w);
    }
}

void blend_additive(PoseSpan dst, ConstPoseSpan src, ConstPoseSpan reference,
                    const BlendParams& params) noexcept {
    const std::size_t n = std::min({dst.size(), src.size(), reference.size()});
    for (std::size_t i = 0; i < n; ++i) {
        const float w = joint_weight(params, i);
        if (w <= 0.0f) {
            continue;
        }

        // Rotation: post-multiply, delta in the bone's own frame. See the
        // header — the alternative is a parent-frame delta and they are NOT
        // interchangeable on a rotated base.
        const glm::quat delta = glm::inverse(reference[i].rotation) * src[i].rotation;
        const glm::quat target = dst[i].rotation * delta;
        dst[i].rotation = (w >= 1.0f) ? target : glm::slerp(dst[i].rotation, target, w);

        dst[i].translation += (src[i].translation - reference[i].translation) * w;

        const glm::vec3 ratio = safe_scale_ratio(src[i].scale, reference[i].scale);
        dst[i].scale *= (w >= 1.0f) ? ratio : glm::mix(glm::vec3(1.0f), ratio, w);
    }
}

std::size_t pose_to_matrices(const Skeleton& skeleton, ConstPoseSpan pose,
                             std::span<glm::mat4> world_transforms,
                             std::span<glm::mat4> bone_matrices) noexcept {
    const std::size_t count = std::min({pose.size(), skeleton.joints.size(), world_transforms.size(),
                                        bone_matrices.size(), static_cast<std::size_t>(MAX_BONES)});

    // (4) TRS -> local matrix, built in place. Identical statements, identical
    // order to AnimationPlayer::compute_bone_matrices.
    for (std::size_t i = 0; i < count; ++i) {
        glm::mat4 m = glm::mat4_cast(pose[i].rotation);
        m[0] *= pose[i].scale.x;
        m[1] *= pose[i].scale.y;
        m[2] *= pose[i].scale.z;
        m[3] = glm::vec4(pose[i].translation, 1.0f);
        world_transforms[i] = m;
    }

    // (5) Hierarchy, one ascending pass. world_transforms[i] still holds the
    // LOCAL matrix when this reads it, and world_transforms[parent] is already
    // final because parent < i.
    for (std::size_t i = 0; i < count; ++i) {
        const int parent = skeleton.joints[i].parent_index;
        if (parent >= 0 && parent < static_cast<int>(count)) {
            world_transforms[i] = world_transforms[static_cast<std::size_t>(parent)] * world_transforms[i];
        }
    }

    // (6) Skinning matrices.
    for (std::size_t i = 0; i < count; ++i) {
        bone_matrices[i] = world_transforms[i] * skeleton.joints[i].inverse_bind_matrix;
    }

    return count;
}

} // namespace bracken::animation
