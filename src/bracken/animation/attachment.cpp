#include "bracken/animation/attachment.hpp"

#include "bracken/core/logger.hpp"

namespace bracken::animation {

JointIndex find_joint(const Skeleton& skeleton, std::string_view name) {
    for (std::size_t i = 0; i < skeleton.joints.size(); ++i) {
        if (skeleton.joints[i].name == name) {
            return JointIndex{static_cast<int>(i)};
        }
    }
    return {};
}

glm::mat4 joint_model_transform(const PoseView& pose, JointIndex joint) {
    if (!joint.valid()) {
        return glm::mat4(1.0f);
    }
    const auto index = static_cast<std::size_t>(joint.value);
    // Beyond used_joint_count the slots hold identity rather than stale
    // data, but bounds still matter: a skeleton swapped for a shorter one
    // would otherwise read past the array.
    if (index >= pose.world_transforms.size() || index >= pose.used_joint_count) {
        return glm::mat4(1.0f);
    }
    return pose.world_transforms[index];
}

glm::mat4 attachment_transform(const glm::mat4& model_transform, const PoseView& pose, JointIndex joint,
                               const glm::mat4& local_offset) {
    if (!joint.valid()) {
        // No joint means no attachment. Returning the model transform (rather
        // than identity) keeps the object on the character instead of dropping
        // it at the world origin, which is far easier to notice and diagnose.
        return model_transform * local_offset;
    }
    return model_transform * joint_model_transform(pose, joint) * local_offset;
}

glm::vec3 attachment_position(const glm::mat4& model_transform, const PoseView& pose, JointIndex joint,
                              const glm::vec3& local_offset) {
    const glm::mat4 bone = attachment_transform(model_transform, pose, joint);
    return glm::vec3(bone * glm::vec4(local_offset, 1.0f));
}

// The AnimationPlayer overloads are forwarders, so the player and the rig can
// never take different code paths through this file.
namespace {
[[nodiscard]] PoseView view_of(const AnimationPlayer& player) noexcept {
    return PoseView{player.world_transforms(), player.bone_matrices(), player.used_joint_count()};
}
} // namespace

glm::mat4 joint_model_transform(const AnimationPlayer& player, JointIndex joint) {
    return joint_model_transform(view_of(player), joint);
}

glm::mat4 attachment_transform(const glm::mat4& model_transform, const AnimationPlayer& player,
                               JointIndex joint, const glm::mat4& local_offset) {
    return attachment_transform(model_transform, view_of(player), joint, local_offset);
}

glm::vec3 attachment_position(const glm::mat4& model_transform, const AnimationPlayer& player,
                              JointIndex joint, const glm::vec3& local_offset) {
    return attachment_position(model_transform, view_of(player), joint, local_offset);
}

Attachment make_attachment(const Skeleton& skeleton, std::string_view bone_name,
                           const glm::mat4& local_offset) {
    Attachment a;
    a.joint = find_joint(skeleton, bone_name);
    a.local_offset = local_offset;
    if (!a.joint.valid()) {
        BRACKEN_LOG_WARN("animation",
                         "make_attachment: no bone named '{}' in this skeleton ({} joints); the "
                         "attachment will follow the model origin",
                         bone_name, skeleton.joints.size());
    }
    return a;
}

} // namespace bracken::animation
