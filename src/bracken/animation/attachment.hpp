#pragma once

#include "bracken/animation/animation_player.hpp"
#include "bracken/animation/animation_types.hpp"
#include "bracken/animation/pose.hpp"

#include <glm/glm.hpp>
#include <string_view>

namespace bracken::animation {

/** Index of a joint in a Skeleton, resolved once and reused. */
struct JointIndex {
    static constexpr int k_invalid = -1;
    int value = k_invalid;
    [[nodiscard]] bool valid() const noexcept { return value >= 0; }
};

/**
 * Find a joint by name.
 *
 * Resolve ONCE at load and keep the index: this is a linear scan over joint
 * names, and calling it per frame per attachment turns a trivial feature into
 * a profiler entry. Returns an invalid index when the name is absent, which
 * every function here treats as "no attachment" rather than as undefined
 * behavior — a renamed bone in an updated asset should drop the sword, not
 * corrupt the frame.
 */
[[nodiscard]] JointIndex find_joint(const Skeleton& skeleton, std::string_view name);

/**
 * Where a bone is in MODEL space, from the pose the player last computed.
 *
 * This is the joint's world transform, NOT its skinning matrix. The two are
 * constantly confused and the difference is exactly the inverse bind matrix:
 * the skinning matrix maps bind-pose vertices to posed ones and is meaningless
 * as a placement, while this is the bone's actual position and orientation.
 * Attaching a sword with a skinning matrix puts it somewhere near the origin
 * at a strange angle, which is the classic symptom.
 *
 * Returns identity for an invalid or out-of-range index.
 *
 * NOTE: it reads what the player LAST COMPUTED. AnimationPlayer::update()
 * returns early when a model has no animation clips, so a never-animated
 * skeleton has no world transforms and every attachment on it reads identity.
 * Give a static posed character a single-keyframe clip.
 */
[[nodiscard]] glm::mat4 joint_model_transform(const AnimationPlayer& player, JointIndex joint);

/**
 * The same thing, from any posed thing.
 *
 * `PoseView` (`pose.hpp`) is what an `AnimationRig` and an `AnimationPlayer`
 * have in common, so this overload is what lets a layered character keep its
 * lantern. Everything the `AnimationPlayer` overload's comment says about
 * bone-space versus skinning-space applies here verbatim — this is the joint's
 * WORLD transform, not its skinning matrix.
 *
 * The `AnimationPlayer` overloads on this page are one-line forwarders to these,
 * so the two can never disagree.
 */
[[nodiscard]] glm::mat4 joint_model_transform(const PoseView& pose, JointIndex joint);

/**
 * The world transform for something attached to a bone.
 *
 * `model_transform` is the character's own world matrix — the same one handed
 * to draw_skinned. `local_offset` positions the attachment within the bone's
 * frame, which is where a grip offset and a blade rotation belong.
 *
 *     const glm::mat4 sword = attachment_transform(model, player, hand, grip);
 *     frame.draw_model(sword_model, sword);
 *
 * That is the whole feature: hosts were doing this multiplication by hand, and
 * getting the bone-space vs skinning-space distinction wrong when they did.
 */
[[nodiscard]] glm::mat4 attachment_transform(const glm::mat4& model_transform,
                                             const AnimationPlayer& player, JointIndex joint,
                                             const glm::mat4& local_offset = glm::mat4(1.0f));

/** Just the world-space position of a bone — for spawning an effect at a hand,
 *  aiming from a muzzle, or placing a sound emitter. */
[[nodiscard]] glm::vec3 attachment_position(const glm::mat4& model_transform,
                                            const AnimationPlayer& player, JointIndex joint,
                                            const glm::vec3& local_offset = glm::vec3(0.0f));

/** `PoseView` overloads — see `joint_model_transform(const PoseView&, JointIndex)`. */
[[nodiscard]] glm::mat4 attachment_transform(const glm::mat4& model_transform, const PoseView& pose,
                                             JointIndex joint,
                                             const glm::mat4& local_offset = glm::mat4(1.0f));
[[nodiscard]] glm::vec3 attachment_position(const glm::mat4& model_transform, const PoseView& pose,
                                            JointIndex joint,
                                            const glm::vec3& local_offset = glm::vec3(0.0f));

/**
 * A resolved attachment: a joint plus a fixed offset within it.
 *
 * Holds the joint INDEX, not a pointer into the skeleton, so it stays valid
 * across model reloads and can be copied freely into game state.
 */
struct Attachment {
    JointIndex joint{};
    glm::mat4 local_offset{1.0f};

    [[nodiscard]] bool valid() const noexcept { return joint.valid(); }

    [[nodiscard]] glm::mat4 world(const glm::mat4& model_transform,
                                  const AnimationPlayer& player) const {
        return attachment_transform(model_transform, player, joint, local_offset);
    }

    /// Overload for anything that can produce a `PoseView` — notably
    /// `AnimationRig::pose()`.
    [[nodiscard]] glm::mat4 world(const glm::mat4& model_transform, const PoseView& pose) const {
        return attachment_transform(model_transform, pose, joint, local_offset);
    }
};

/** Resolve a named attachment point. `local_offset` is applied within the
 *  bone's frame. An unknown bone name yields an invalid Attachment and a
 *  warning, so a typo is visible rather than silently placing the object at
 *  the character's origin. */
[[nodiscard]] Attachment make_attachment(const Skeleton& skeleton, std::string_view bone_name,
                                         const glm::mat4& local_offset = glm::mat4(1.0f));

} // namespace bracken::animation
