#pragma once

/**
 * @file clip_sampler.hpp
 * @brief Sampling a clip into a `BoneTransform` pose, plus the shared
 *        clip->joint->channel table that keeps a blend space from paying for
 *        one channel lookup per sample per character.
 *
 * WHY THE SAMPLER BODY IS DUPLICATED FROM `animation_player.cpp`
 * -------------------------------------------------------------
 * `AnimationPlayer::compute_bone_matrices` calls a file-static
 * `sample_clip_joint` whose output is pinned by the engine's screenshot
 * baselines: the playground character's final pose feeds `rebuild_shadow_capsules`,
 * which feeds traced sun shadows, so a single differing float moves rows in
 * CLAUDE.md's pin tables. `sample_clip_joint_into` below is a character-for-
 * character copy of that function rather than a refactor of it, because a
 * duplicate that is proven equal by test is cheaper than a moved baseline that
 * has to be re-measured on two backends.
 *
 * The guard against drift is not this comment — it is
 * `AnimationRig, SingleOverrideLayerMatchesAnimationPlayerBitwise` in
 * `animation_rig.test.cpp`, which memcmps 120 frames of rig output against
 * `AnimationPlayer` output. Edit one copy without the other and that test goes
 * red immediately.
 *
 * Thread-safety tier: `worker-safe`. `ClipBindings` is const-shared across
 * instances; `ClipSampler` is per-instance mutable state, exactly like
 * `AnimationPlayer`'s cursor vectors.
 */

#include "bracken/animation/animation_types.hpp"
#include "bracken/animation/pose.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace bracken::animation {

/**
 * Sample one joint out of one clip into a TRS triple.
 *
 * Defaults the outputs to the joint's REST transform and overwrites only the
 * paths the channel actually carries, so a clip that animates rotation only
 * leaves translation and scale at the bind pose. A null clip, an out-of-range
 * joint or a joint with no channel leaves all three at rest.
 *
 * `cursors` must cover `joint_idx`; pass an EMPTY span to use the cursorless
 * `interpolate_keyframes` overload instead (a `std::upper_bound` per path).
 * Cursors are a pure SPEED dial — the cursor fast path and the binary search
 * select the same keyframe interval, so results are bitwise identical either
 * way.
 */
void sample_clip_joint_into(const AnimationClip* clip,
                            std::span<const AnimationChannel* const> channels,
                            std::span<ChannelCursors> cursors, int joint_idx, float time,
                            const Joint& joint, glm::vec3& out_translation, glm::quat& out_rotation,
                            glm::vec3& out_scale);

/**
 * clip -> joint -> channel*, precomputed once per MODEL and shared const by
 * every instance driven by it.
 *
 * `AnimationPlayer` rebuilds this table into a per-instance vector whenever the
 * clip changes, which is fine for one clip but not for a blend space: eight
 * simultaneously-sampled clips would otherwise cost 8 * 8 B per joint per
 * character, and rebuild whenever the blend crossed a sample boundary. Build
 * one of these next to the model and hand it to every `AnimationRig`.
 *
 * Resolution matches the player's exactly, including the tie-break: channels
 * are walked in `clip.channels` order and the LAST channel naming a joint wins.
 * (`clip.channels` order is unspecified — the loader emits them from an
 * `unordered_map` — so never index it positionally.)
 *
 * The bindings alias the clip vector. They must be rebuilt if the clips are
 * reloaded, and must outlive every rig bound to them.
 */
class ClipBindings {
public:
    /// Sizes and fills the table. Safe to call again to rebind.
    void build(const Skeleton& skeleton, const std::vector<AnimationClip>& clips);

    void clear() noexcept;

    /// Channels for one clip, indexed by joint. Empty span for an out-of-range
    /// clip index, which callers treat as "no animation" rather than as an error.
    [[nodiscard]] std::span<const AnimationChannel* const> for_clip(int clip_index) const noexcept;

    [[nodiscard]] std::size_t joint_count() const noexcept { return joints_; }
    [[nodiscard]] std::size_t clip_count() const noexcept { return clips_; }
    [[nodiscard]] bool valid() const noexcept { return joints_ > 0 && clips_ > 0; }

private:
    std::vector<const AnimationChannel*> flat_; ///< clips_ * joints_, row-major by clip
    std::size_t joints_ = 0;
    std::size_t clips_ = 0;
};

/**
 * One clip's playback cursors for one instance.
 *
 * `bind(joint_count, cache_cursors)` sizes the cursor storage; with
 * `cache_cursors == false` it stays empty and every sample does a binary search
 * instead — 24 B/joint saved for about five extra comparisons on a typical
 * 20-60-key track. That dial is what makes an eight-sample blend space
 * affordable: eight cached samplers at 128 joints is 24 KB per layer per
 * character.
 *
 * Cursors are reset when the sampled clip index changes, mirroring
 * `AnimationPlayer`, which reassigns its cursor vector on every clip switch.
 */
class ClipSampler {
public:
    void bind(std::size_t joint_count, bool cache_cursors);
    void reset() noexcept;

    /// Sample `clip_index` at `time` into `out[0, min(joints, out.size()))`.
    /// An out-of-range clip index writes the rest pose.
    void sample(const Skeleton& skeleton, const std::vector<AnimationClip>& clips,
                const ClipBindings& bindings, int clip_index, float time, PoseSpan out);

    [[nodiscard]] bool caches_cursors() const noexcept { return cache_cursors_; }

private:
    std::vector<ChannelCursors> cursors_;
    std::size_t joints_ = 0;
    int cached_clip_ = -1;
    bool cache_cursors_ = true;
};

/**
 * Cursorless one-shot sample — for a pose read at a FIXED time, such as an
 * additive layer's reference frame. Allocates nothing and keeps no state, so it
 * is the right tool when there is no per-frame continuity to exploit.
 */
void sample_clip_pose(const Skeleton& skeleton, const std::vector<AnimationClip>& clips,
                      const ClipBindings& bindings, int clip_index, float time, PoseSpan out);

} // namespace bracken::animation
