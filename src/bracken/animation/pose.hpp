#pragma once

/**
 * @file pose.hpp
 * @brief The local-space pose value type the engine never had, plus bone masks
 *        and the two blend operators every layer stack is built out of.
 *
 * Until this header existed, per-joint TRS lived for four statements as three
 * locals inside `AnimationPlayer::compute_bone_matrices` and was immediately
 * collapsed into a `glm::mat4` — so there was no seam a second animation source
 * could be composed at. Everything here is that seam, and nothing more: a pose
 * is a plain array of parent-relative transforms, and the matrix pipeline that
 * turns one into skinning matrices is `pose_to_matrices`, lifted verbatim from
 * the player so both can be proved to agree.
 *
 * Nothing in this header allocates. Nothing in it touches the GPU. Thread-safety
 * tier: `worker-safe` — every function operates only on its arguments.
 */

#include "bracken/animation/animation_types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace bracken::animation {

/**
 * One joint's LOCAL (parent-relative) transform — the representation every
 * blend happens in.
 *
 * 40 bytes, so `std::array<BoneTransform, MAX_BONES>` is 5,120 B. Blending in
 * TRS rather than in matrices is not a preference: a matrix lerp shears, and a
 * quaternion slerp is the only rotation blend that keeps a limb rigid.
 *
 * The defaults are the identity transform, which is also what an unbound joint
 * reads as.
 */
struct BoneTransform {
    glm::vec3 translation{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f};
};

using PoseSpan = std::span<BoneTransform>;
using ConstPoseSpan = std::span<const BoneTransform>;

/**
 * Read-only view of a COMPUTED pose — the matrix side, not the TRS side.
 *
 * This exists so `attachment.hpp`'s helpers can accept an `AnimationPlayer` or
 * an `AnimationRig` without either type knowing about the other. Both expose
 * exactly these three things, and both maintain the same contract: slots at or
 * beyond `used_joint_count` hold identity.
 *
 * A `PoseView` borrows; it must not outlive the player/rig it was taken from.
 */
struct PoseView {
    std::span<const glm::mat4> world_transforms{};
    std::span<const glm::mat4> bone_matrices{};
    std::size_t used_joint_count = 0;
};

/**
 * Per-joint binary selector: which joints a layer is allowed to touch.
 *
 * Two 64-bit words at `MAX_BONES` == 128, and the extent is written in terms of
 * the constant so raising it later is not a second edit here. 16 bytes, so a
 * mask is cheap enough to store by value in a layer description — which is why
 * this is a bitset and not the `std::vector<uint8_t>` `apply_body_lean`
 * heap-allocates on every call.
 *
 * KNOWN ARTIFACT, and it is inherent rather than a defect: a binary mask makes
 * the joint at the boundary jump between two poses while its parent does not,
 * which reads as a pop at the waist when an upper-body layer fades in. The
 * engine does NOT feather this for you. `BlendParams::joint_weights` is the
 * hook — supply a per-joint weight ramp across the boundary — and
 * `feather_by_depth` builds a serviceable one. Feathering is the host's call
 * because only the host knows how many joints the ramp should span.
 */
class BoneMask {
public:
    /// Words needed to hold `MAX_BONES` bits. 2 at 128 joints.
    static constexpr int k_words = (MAX_BONES + 63) / 64;

    /// Every joint in `[0, MAX_BONES)`. The default for a base layer.
    [[nodiscard]] static BoneMask all() noexcept {
        BoneMask m;
        for (int w = 0; w < k_words; ++w) {
            m.bits_[static_cast<std::size_t>(w)] = ~std::uint64_t{0};
        }
        // Trim the tail when MAX_BONES is not a multiple of 64, so `count()`
        // and `inverted()` never report joints that cannot exist.
        constexpr int k_tail = MAX_BONES % 64;
        if constexpr (k_tail != 0) {
            m.bits_[k_words - 1] = (std::uint64_t{1} << k_tail) - 1;
        }
        return m;
    }

    /// No joints. A layer with an empty mask is skipped entirely.
    [[nodiscard]] static BoneMask none() noexcept { return BoneMask{}; }

    /// Set or clear one joint. An out-of-range index is a NO-OP, never UB —
    /// a mask outliving the skeleton it was built for is a content bug, and
    /// silently ignoring it beats corrupting the neighbouring word.
    void set(int joint, bool on = true) noexcept {
        if (joint < 0 || joint >= MAX_BONES) {
            return;
        }
        const std::size_t w = static_cast<std::size_t>(joint) / 64u;
        const std::uint64_t bit = std::uint64_t{1} << (static_cast<std::size_t>(joint) % 64u);
        if (on) {
            bits_[w] |= bit;
        } else {
            bits_[w] &= ~bit;
        }
    }

    /// Out-of-range reads `false` — see `set`.
    [[nodiscard]] bool test(int joint) const noexcept {
        if (joint < 0 || joint >= MAX_BONES) {
            return false;
        }
        const std::size_t w = static_cast<std::size_t>(joint) / 64u;
        return (bits_[w] >> (static_cast<std::size_t>(joint) % 64u)) & 1u;
    }

    [[nodiscard]] bool empty() const noexcept {
        for (const std::uint64_t w : bits_) {
            if (w != 0) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] int count() const noexcept {
        int n = 0;
        for (std::uint64_t w : bits_) {
            while (w != 0) {
                w &= w - 1;
                ++n;
            }
        }
        return n;
    }

    /**
     * Mark `root` and every joint beneath it in the hierarchy.
     *
     * PRECONDITION: joints are ordered parents-before-children (every
     * `parent_index` < its own index; roots are -1), which every skeleton the
     * model loader produces satisfies. That is what lets this be a single
     * ascending-index pass — the same technique `apply_body_lean`
     * (`ik_solver.cpp`) uses, but allocation-free.
     *
     * The descendant set is computed into a LOCAL mask and then merged, rather
     * than tested against `*this` as it goes. Testing in place looks equivalent
     * and is not: a joint whose parent happened to already be set for an
     * unrelated reason would be swept in, so `set_subtree(Spine)` on a mask that
     * already had one hand bit set would silently claim that hand's children.
     */
    void set_subtree(const Skeleton& skeleton, int root, bool on = true) noexcept;

    BoneMask& operator|=(const BoneMask& rhs) noexcept {
        for (int w = 0; w < k_words; ++w) {
            bits_[static_cast<std::size_t>(w)] |= rhs.bits_[static_cast<std::size_t>(w)];
        }
        return *this;
    }

    BoneMask& operator&=(const BoneMask& rhs) noexcept {
        for (int w = 0; w < k_words; ++w) {
            bits_[static_cast<std::size_t>(w)] &= rhs.bits_[static_cast<std::size_t>(w)];
        }
        return *this;
    }

    /// The complement, restricted to `[0, MAX_BONES)`.
    [[nodiscard]] BoneMask inverted() const noexcept {
        BoneMask m = all();
        for (int w = 0; w < k_words; ++w) {
            m.bits_[static_cast<std::size_t>(w)] &= ~bits_[static_cast<std::size_t>(w)];
        }
        return m;
    }

    [[nodiscard]] friend bool operator==(const BoneMask& a, const BoneMask& b) noexcept {
        return a.bits_ == b.bits_;
    }

private:
    std::array<std::uint64_t, k_words> bits_{};
};

/**
 * Build a per-joint weight ramp that feathers a subtree mask's boundary.
 *
 * `out[j]` is 1 for joints at least `falloff` levels below `root`, 0 outside
 * the subtree, and a linear ramp in between measured in HIERARCHY DEPTH from
 * the root. `falloff == 0` reproduces the binary mask exactly.
 *
 * This is a serviceable default, not a rig-authoring tool: depth is a poor
 * proxy for anatomy on a rig where the spine is three joints and an arm is
 * five. It exists so the mask-boundary pop has an answer in the box.
 *
 * PRECONDITION: parents-before-children (single ascending pass). `out` is
 * written for `[0, min(joints, out.size()))` and untouched beyond.
 */
void feather_by_depth(const Skeleton& skeleton, int root, int falloff, std::span<float> out) noexcept;

/**
 * Fill `out` with the skeleton's rest pose — `Joint::local_translation` /
 * `local_rotation` / `local_scale`, verbatim.
 *
 * This is the default additive reference and the pose an unbound layer falls
 * back to. Writes `min(joints, out.size())` entries.
 */
void rest_pose(const Skeleton& skeleton, PoseSpan out) noexcept;

/**
 * Per-blend knobs, shared by both blend operators.
 *
 * The effective weight for joint `j` is
 *     `weight * (mask ? mask->test(j) : 1) * (joint_weights.empty() ? 1 : joint_weights[j])`
 * clamped to [0, 1]. `joint_weights` is CALLER-OWNED and must cover the joint
 * range being blended; it is how a mask boundary is feathered without every
 * layer paying 512 B for the common binary case.
 *
 * Both operators special-case the ends: an effective weight <= 0 leaves the
 * destination joint BITWISE untouched, and >= 1 assigns rather than
 * interpolates. That is not an optimisation — it is what makes "a layer at
 * weight 0 is exactly the base pose" a bitwise statement instead of an
 * approximate one, and it is what lets a single override layer at weight 1
 * reproduce `AnimationPlayer` to the last bit.
 */
struct BlendParams {
    float weight = 1.0f;
    const BoneMask* mask = nullptr;      ///< null == every joint
    std::span<const float> joint_weights{}; ///< empty == binary mask
};

/**
 * `dst = lerp(dst, src, w_j)` per joint: `mix` / `slerp` / `mix`, in that
 * order.
 *
 * Deliberately the same three calls in the same order as the crossfade blend
 * inside `AnimationPlayer::compute_bone_matrices`, because the single-layer
 * bit-identity regression (`animation_rig.test.cpp`) depends on it. Do not
 * "simplify" this into a matrix blend or reorder the components.
 *
 * Blends `min(dst.size(), src.size())` joints.
 */
void blend_override(PoseSpan dst, ConstPoseSpan src, const BlendParams& params) noexcept;

/**
 * `dst = dst + w_j * (src - reference)`, in the standard additive delta form.
 *
 * ROTATION ORDER IS POST-MULTIPLY, IN THE BONE'S OWN FRAME:
 *
 *     delta  = inverse(reference.r) * src.r      // the additive clip's motion
 *     target = dst.r * delta                     // re-expressed on top of dst
 *     dst.r  = slerp(dst.r, target, w)
 *
 * The alternative — `delta = src.r * inverse(reference.r)` applied as
 * `delta * dst.r` — is a delta in the PARENT's frame, and the two differ
 * whenever the base pose has rotated the joint, which on a running character is
 * always. Post-multiply is chosen because it matches the usual authoring
 * intent: "this additive clip means the same thing regardless of where the base
 * put the bone". An aim offset authored as a 30-degree lift stays a 30-degree
 * lift of the arm even after locomotion has swung the arm somewhere else. If
 * you want "lean 10 degrees in the parent's frame" instead, that is a different
 * operator and this is not it.
 *
 * Translation is a plain difference; scale is a RATIO, guarded so a zero
 * component in the reference scale yields a ratio of 1 rather than an infinity.
 *
 * `src == reference` is the identity for any weight — exactly at weight 0 (the
 * early-out assigns nothing) and to float precision above it, because
 * `inverse(q) * q` is only identity to the precision `q` is normalized to.
 *
 * NON-UNIFORM SCALE IS UNSUPPORTED in additive layers, and is wrong rather than
 * approximate: the pipeline bakes scale into the local matrix by scaling
 * columns, so a scaled parent shears a rotated child. That is pre-existing, but
 * a scale RATIO makes it easy to hit. Keep additive clips rotation- and
 * translation-only unless you have measured the result.
 */
void blend_additive(PoseSpan dst, ConstPoseSpan src, ConstPoseSpan reference,
                    const BlendParams& params) noexcept;

/**
 * Local TRS pose -> world transforms -> skinning matrices.
 *
 * This is steps (4)(5)(6) of `AnimationPlayer::compute_bone_matrices`, hoisted
 * unchanged so the player and the rig can be proved to produce the same floats:
 *
 *   (4) `m = mat4_cast(rotation)`, columns 0..2 scaled, column 3 = translation;
 *   (5) `world[i] = world[parent] * local[i]`, one ascending pass;
 *   (6) `bone[i] = world[i] * inverse_bind_matrix`.
 *
 * PRECONDITION: joints ordered parents-before-children — step (5) is a single
 * pass and reads `world[parent]` expecting it to be final already.
 *
 * The local matrix is built IN PLACE in `world_transforms` and then accumulated
 * over itself, which is why no scratch buffer is needed and why the parents-
 * before-children order is load-bearing twice over.
 *
 * Writes and returns `min(pose.size(), joints, world.size(), bones.size(),
 * MAX_BONES)` entries — the caller's new `used_joint_count`. Slots beyond that
 * are NOT touched; maintaining the "identity past the used count" contract is
 * the caller's job, and doing it only when the count SHRINKS is the point (a
 * per-update tail rewrite of both arrays is ~8 KB of dead stores per instance
 * per frame).
 */
[[nodiscard]] std::size_t pose_to_matrices(const Skeleton& skeleton, ConstPoseSpan pose,
                                           std::span<glm::mat4> world_transforms,
                                           std::span<glm::mat4> bone_matrices) noexcept;

/**
 * Scratch buffers one evaluating THREAD needs — 3 x 5,120 = 15,360 bytes.
 *
 * Deliberately not owned per instance and deliberately not `thread_local`. One
 * per `parallel_for` chunk is the intended shape: per-character memory then
 * stays at the rig's own 21 KB no matter how many layers it runs, and the
 * engine's stage discipline rides the job node rather than TLS
 * (`core/jobs/job_stage.hpp`), so a thread-local here would be the one place
 * that disagreed.
 *
 * TWO RIGS MUST NOT SHARE ONE SCRATCH CONCURRENTLY. There is no cheap assert
 * for it — the corruption is silent and looks like an animation glitch. The
 * guard is `animation_rig_parallel.test.cpp`, which evaluates 256 rigs serially
 * and through `parallel_for` and demands bitwise-equal output.
 */
struct PoseScratch {
    std::array<BoneTransform, MAX_BONES> layer{};     ///< the layer currently being composed
    std::array<BoneTransform, MAX_BONES> sample{};    ///< one blend-space sample
    std::array<BoneTransform, MAX_BONES> reference{}; ///< the additive reference

    [[nodiscard]] PoseSpan layer_span() noexcept { return PoseSpan(layer.data(), layer.size()); }
    [[nodiscard]] PoseSpan sample_span() noexcept { return PoseSpan(sample.data(), sample.size()); }
    [[nodiscard]] PoseSpan reference_span() noexcept {
        return PoseSpan(reference.data(), reference.size());
    }
};

} // namespace bracken::animation
