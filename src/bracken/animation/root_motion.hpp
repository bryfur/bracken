#pragma once

/**
 * @file root_motion.hpp
 * @brief Extracting a character's locomotion out of the animation and handing
 *        it to the host, plus removing it from the pose so the character does
 *        not also travel in place.
 *
 * THE SHIPPED ASSETS CANNOT DEMONSTRATE THIS, AND THAT IS NOT A BUG HERE
 * ---------------------------------------------------------------------
 * No clip in this repository carries usable root translation. The playground's
 * character authors exactly one position channel — a vertical hips bob — and
 * its XZ motion comes from the physics character controller. Worse, glTF root
 * motion is conventionally authored on an Armature node ABOVE the skin, and
 * `ModelLoader` drops any channel whose target node is not in `skin.joints`
 * (counting it in `ModelLoadReport::channels_non_skin_node`), so such a channel
 * never reaches the engine at all.
 *
 * WHAT AN ARTIST MUST AUTHOR for this to work, therefore:
 *  1. the motion must be on a JOINT INSIDE the skin — conventionally `Hips`;
 *  2. that joint must be at the ORIGIN in XZ in the reference/rest pose, because
 *     the host applies the delta in the character's frame and a non-zero
 *     reference offset gets rotated by the accumulated yaw (the round-trip test
 *     in `root_motion.test.cpp` states this precondition explicitly);
 *  3. the clip must start and end at compatible positions if it loops, or the
 *     wrap seam produces a jump.
 *
 * Everything here is therefore validated against SYNTHETIC skeletons and clips
 * built in the test, not against shipped content.
 *
 * Thread-safety tier: `worker-safe`.
 */

#include "bracken/animation/animation_types.hpp"
#include "bracken/animation/clip_sampler.hpp"
#include "bracken/animation/pose.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace bracken::animation {

/**
 * One update's worth of locomotion, in the character's OWN (model) frame.
 *
 * The host applies it — the engine never moves anything:
 *
 *     const auto d = rig.root_motion();
 *     character_yaw = character_yaw * d.rotation;                 // yaw first
 *     world_position += glm::mat3_cast(character_yaw) * d.translation;
 *
 * Order matters and there is no universally right one; the engine documents
 * this one and the round-trip test pins it.
 */
struct RootMotionDelta {
    glm::vec3 translation{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};

    [[nodiscard]] bool is_zero() const noexcept {
        return translation == glm::vec3(0.0f) && rotation == glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    }
};

/**
 * Which components of the root joint's motion are locomotion.
 *
 * `translation_y` is OFF by default and that default is the whole point: a hips
 * bob is a vertical channel on the root joint and it is NOT locomotion. Turn it
 * on only for a clip that genuinely translates the character vertically (a
 * vault, a ledge climb), and expect to turn it off again afterwards.
 *
 * `strip_from_pose` defaults ON. With it off — or with it on and the host
 * ignoring the returned delta — the character either travels at double speed or
 * animates on the spot. Those are the two failure modes and they are
 * indistinguishable from a gameplay bug, so the delta is returned as an
 * explicit value the host must consume rather than applied anywhere internally.
 */
struct RootMotionConfig {
    int joint = -1;                 ///< -1 disables extraction entirely; conventionally the Hips index
    bool translation_xz = true;
    bool translation_y = false;     ///< a hips bob is not locomotion
    bool rotation_yaw = true;
    bool strip_from_pose = true;
};

/**
 * The pure-yaw (twist) component of `q` about +Y, by swing-twist decomposition.
 *
 * Returns identity — never a NaN — for the degenerate case where the twist axis
 * projection vanishes (`q.w == 0 && q.y == 0`, i.e. a 180-degree rotation about
 * an axis in the XZ plane). `inverse(twist_about_y(q)) * q` is the SWING, the
 * part with no yaw in it, which is exactly what stripping leaves behind.
 *
 * Shared by extraction and stripping so the two halves cannot drift apart.
 */
[[nodiscard]] glm::quat twist_about_y(const glm::quat& q) noexcept;

/**
 * How much the root joint moved between two clip times.
 *
 * `wrapped` says the layer's clock looped this update, in which case the delta
 * is the pre-wrap part `(duration - t_prev)` composed with the post-wrap part
 * `(t_now - 0)`, PRE-WRAP FIRST — translations add, rotations compose as
 * `post * pre`.
 *
 * Returns a ZERO delta when: the config is disabled, the joint is out of range,
 * the clip index does not resolve, or the joint has no channel. In particular a
 * joint with no position keys reads as no translation rather than as "the rest
 * pose moved", which is the difference between a still character and one that
 * teleports to the origin every frame.
 *
 * The channel is resolved through `ClipBindings`, never by indexing
 * `clip.channels` positionally — that vector's order is unspecified.
 */
[[nodiscard]] RootMotionDelta extract_root_motion(const std::vector<AnimationClip>& clips,
                                                  const ClipBindings& bindings, int clip_index,
                                                  float t_prev, float t_now, bool wrapped,
                                                  const RootMotionConfig& cfg) noexcept;

/**
 * Remove from a composed LOCAL pose exactly what extraction took out.
 *
 * Root translation XZ (and/or Y, per the config) is reset to the corresponding
 * component of `reference` — normally the rest pose — and the yaw is cancelled
 * with `inverse(twist_about_y(q)) * q`, which preserves pitch and roll. Every
 * other joint is left BITWISE untouched.
 *
 * A no-op when the config is disabled or the joint index is out of range, so it
 * is safe to call unconditionally.
 */
void strip_root_motion(PoseSpan pose, ConstPoseSpan reference, const RootMotionConfig& cfg) noexcept;

} // namespace bracken::animation
