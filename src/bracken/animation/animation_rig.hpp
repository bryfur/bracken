#pragma once

/**
 * @file animation_rig.hpp
 * @brief `AnimationRig` — N weighted, masked, blend-moded layers composed onto a
 *        base pose, then run through the same hierarchy walk `AnimationPlayer`
 *        uses.
 *
 * TWO WAYS TO ANIMATE A CHARACTER, BOTH CORRECT
 * ---------------------------------------------
 * `AnimationPlayer` is NOT deprecated by this and must not be "unified" away.
 * It is the one-clip fast path: 16 KB of matrices plus two cursor vectors, no
 * pose buffer, no layer table, and it is what every existing doc and the
 * umbrella describe. Reach for `AnimationRig` when you need a second
 * simultaneous animation source — an upper-body action over locomotion, an
 * additive aim offset, a blend space — and for nothing else.
 *
 * A rig with exactly one Override layer at weight 1 and a full mask is BITWISE
 * identical to an `AnimationPlayer` running the same clip, crossfades included.
 * That is not a hope; it is `AnimationRig, SingleOverrideLayerMatchesAnimationPlayerBitwise`
 * in `animation_rig.test.cpp`, which memcmps 120 frames of both. Anything that
 * breaks it has moved the engine's screenshot baselines.
 *
 * MEMORY, per rig instance (`MAX_BONES` == 128, unchanged by this feature):
 *   bone_matrices_     8,192 B
 *   world_transforms_  8,192 B
 *   composed_          5,120 B   (the composed LOCAL pose, readable via local_pose())
 *   layers             ~200 B each + cursor storage
 *   -----------------------------------------------------------------------
 *   21.5 KB fixed, against `AnimationPlayer`'s 16 KB.
 *
 * Cursor storage is the part that scales, and it is a dial. A clip layer holds
 * two `ClipSampler`s (current + crossfade partner); a blend-space layer holds up
 * to `k_max_blend_samples`. With `LayerDesc::cache_cursors` true that is
 * 24 B/joint each — 6 KB for a clip layer at 128 joints, 24 KB for a full
 * eight-sample blend-space layer. Set `cache_cursors = false` on blend-space
 * layers unless a profile says otherwise; cursors are a pure speed dial and
 * change no float (see `clip_sampler.hpp`).
 *
 * The three POSE buffers a composition needs are NOT per instance — they live in
 * a `PoseScratch` the caller supplies, one per evaluating thread.
 *
 * Thread-safety tier: `sim-stage` / `worker-safe` per instance. `evaluate()`
 * touches only this rig, its arguments and the scratch; it performs no
 * allocation and no logging. Two rigs must never share one scratch
 * concurrently.
 */

#include "bracken/animation/animation_types.hpp"
#include "bracken/animation/blend_space.hpp"
#include "bracken/animation/clip_sampler.hpp"
#include "bracken/animation/pose.hpp"
#include "bracken/animation/root_motion.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace bracken::animation {

/// How a layer combines with everything composed beneath it.
enum class BlendMode : std::uint8_t {
    /// Replace, weighted: `dst = lerp(dst, layer, w)`.
    Override,
    /// Accumulate a delta measured against a reference pose:
    /// `dst = dst + w * (layer - reference)`. See `blend_additive`.
    Additive,
};

/// What drives a layer's pose.
enum class LayerSource : std::uint8_t {
    Clip,    ///< one clip plus an optional crossfade partner — `AnimationPlayer` semantics
    Blend1D, ///< a `BlendSpace1D` driven by `set_blend_position(layer, float)`
    Blend2D, ///< a `BlendSpace2D` driven by `set_blend_position(layer, glm::vec2)`
};

/**
 * Layer budget. Eight is generous for a character (base + upper body + additive
 * aim + additive breathing + a one-shot leaves three spare) and it is the array
 * extent of the rig's layer table, so raising it costs memory in every rig
 * whether or not the layers are used.
 */
inline constexpr int k_max_animation_layers = 8;

/**
 * Everything about a layer that is authored rather than clocked.
 *
 * LAYER 0 IS THE BASE and its `mode`, `weight` and `mask` are IGNORED: the pose
 * has to start somewhere, and a base at weight 0.5 blended against uninitialized
 * memory is not a meaningful thing to express. Layer 0 writes the composed pose
 * directly. That is also what makes the single-layer bit-identity guarantee
 * possible — there is no `lerp(x, y, 1.0f)` in the path, which is only
 * approximately `y`.
 *
 * Layers compose in INDEX ORDER, and the order is a documented contract, not an
 * accident: swapping two layers changes the result whenever their masks overlap.
 */
struct LayerDesc {
    std::string name;
    BlendMode mode = BlendMode::Override;
    float weight = 1.0f;
    BoneMask mask = BoneMask::all();

    /// Caller-owned per-joint feather multiplying the mask. Empty means a binary
    /// mask. Must remain valid and cover the joint range for as long as the
    /// layer is evaluated — the rig stores the SPAN, not a copy, so that a
    /// binary layer pays nothing. `feather_by_depth` builds one.
    std::span<const float> joint_weights{};

    /// Additive reference: -1 means the skeleton's rest pose (the usual answer),
    /// otherwise a clip index sampled at `additive_reference_time` (frame 0 of
    /// the additive clip's own source animation is the other usual answer).
    /// Ignored when `mode == BlendMode::Override`.
    int additive_reference_clip = -1;
    float additive_reference_time = 0.0f;

    /// Phase-synchronise a blend space's samples so a walk/run blend does not
    /// double-step. Ignored by clip layers. See `effective_duration`.
    bool sync_duration = true;

    /// Keep per-joint keyframe cursors. Pure speed dial; see the file comment
    /// for the memory it costs on a blend-space layer.
    bool cache_cursors = true;
};

/**
 * A layer stack over one skeleton.
 *
 * Move-only for the same reason `AnimationPlayer` is: a copied rig would share
 * keyframe cursors with the original, which was a historical source of
 * animation glitches.
 */
class AnimationRig {
public:
    /// Fills both matrix arrays with identity, exactly as `AnimationPlayer`'s
    /// constructor does. NOT optional: glm's matrix types do not
    /// self-initialize by default, so without this the "slots at/beyond
    /// used_joint_count() hold identity" contract would be a lie on frame 0 —
    /// and `write_palette` and every attachment lean on it.
    AnimationRig() { reset(); }

    AnimationRig(const AnimationRig&) = delete;
    AnimationRig& operator=(const AnimationRig&) = delete;
    AnimationRig(AnimationRig&&) noexcept = default;
    AnimationRig& operator=(AnimationRig&&) noexcept = default;

    /**
     * Bind to a skeleton and its clips, sizing every per-joint buffer.
     *
     * PRECONDITION: `skeleton.joints` is ordered parents-before-children (every
     * `parent_index` < its own index; roots are -1). Skeletons from
     * `ModelLoader::load_glb` are normalized to this at load.
     *
     * `bindings` is shared CONST across every instance driven by this model and
     * must outlive the rig — build one next to the model, not one per character.
     *
     * Safe to call again to rebind (a hot-reloaded model, a swapped skeleton);
     * layers and their clocks survive, cursor storage is resized. After this,
     * and after any `add_layer` / `set_blend_space_*`, `evaluate()` allocates
     * nothing.
     */
    void bind(const Skeleton& skeleton, const std::vector<AnimationClip>& clips, const ClipBindings& bindings);

    [[nodiscard]] bool is_bound() const noexcept { return bindings_ != nullptr; }

    /// Append a layer. Returns its index, or -1 when the stack is full
    /// (`k_max_animation_layers`) — the rig stays evaluable either way, it just
    /// does not gain the layer.
    [[nodiscard]] int add_layer(LayerDesc desc);

    [[nodiscard]] int layer_count() const noexcept { return layer_count_; }

    /// Mutable access to a layer's description, for changing weight/mask between
    /// frames. Out-of-range indices return a scratch layer whose writes are
    /// discarded, so a stale index degrades rather than corrupting.
    [[nodiscard]] LayerDesc& layer(int index) noexcept;
    [[nodiscard]] const LayerDesc& layer(int index) const noexcept;

    void set_layer_weight(int index, float w) noexcept;

    // ---- layer sources (one per layer, last setter wins) -------------------

    /// `AnimationPlayer::crossfade_to` semantics exactly: a no-op when the layer
    /// is already on `clip_index`, otherwise the current clip becomes the
    /// crossfade partner and the new clip starts from time 0.
    void play_clip(int index, int clip_index, float crossfade = 0.2f);

    /// Install a blend space. Sizes that layer's samplers, so call it at setup
    /// (or accept one allocation at the call site — never inside `evaluate`).
    void set_blend_space_1d(int index, const BlendSpace1D& space);
    void set_blend_space_2d(int index, const BlendSpace2D& space);

    void set_blend_position(int index, float x) noexcept;
    void set_blend_position(int index, glm::vec2 p) noexcept;
    [[nodiscard]] glm::vec2 blend_position(int index) const noexcept;

    void set_layer_time(int index, float t) noexcept;
    [[nodiscard]] float layer_time(int index) const noexcept;
    void set_layer_speed(int index, float s) noexcept;
    void set_layer_loop(int index, bool loop) noexcept;
    void set_layer_playing(int index, bool v) noexcept;
    [[nodiscard]] bool layer_playing(int index) const noexcept;
    [[nodiscard]] int layer_clip(int index) const noexcept;

    /// A blend-space layer's normalized phase in [0, 1). Meaningful only when
    /// `LayerDesc::sync_duration` is set; 0 otherwise. Exposed because "did the
    /// phase sync actually sync" is otherwise unobservable from outside.
    [[nodiscard]] float layer_phase(int index) const noexcept;

    // ---- root motion -------------------------------------------------------

    /// Extraction is always from LAYER 0 — the base is the only layer whose clock
    /// unambiguously represents "the character's locomotion". A `joint` of -1
    /// (the default) disables the whole feature and costs nothing.
    void set_root_motion(RootMotionConfig cfg) noexcept { root_motion_ = cfg; }
    [[nodiscard]] const RootMotionConfig& root_motion_config() const noexcept { return root_motion_; }

    /// Valid after `evaluate()`; reset to zero at the start of every evaluate,
    /// so reading it twice per frame does not double-apply. THE HOST MUST
    /// CONSUME THIS — nothing in the engine moves the character.
    [[nodiscard]] const RootMotionDelta& root_motion() const noexcept { return root_delta_; }

    // ---- the frame ---------------------------------------------------------

    /**
     * Advance every layer's clock, sample it, compose in local TRS space in
     * index order, strip root motion, and produce world transforms and skinning
     * matrices.
     *
     * Thread-safety tier: `sim-stage`. Allocates nothing. `scratch` must not be
     * shared with another rig being evaluated concurrently — see `PoseScratch`.
     *
     * Unlike `AnimationPlayer::update`, this does NOT return early when the
     * model has no clips: the rest pose is a legitimate answer and produces
     * usable world transforms, which is what makes attachments work on a
     * never-animated skeleton (the player leaves identity there — see
     * `attachment.hpp`).
     */
    void evaluate(const Skeleton& skeleton, const std::vector<AnimationClip>& clips, float dt,
                  PoseScratch& scratch);

    // ---- output: deliberately the SAME surface AnimationPlayer exposes, so
    // ---- ik_solver, attachments and the bone-palette path need no change.

    [[nodiscard]] std::span<const glm::mat4> bone_matrices() const noexcept {
        return std::span<const glm::mat4>(bone_matrices_.data(), bone_matrices_.size());
    }
    [[nodiscard]] std::span<const glm::mat4> world_transforms() const noexcept {
        return std::span<const glm::mat4>(world_transforms_.data(), world_transforms_.size());
    }
    [[nodiscard]] std::span<glm::mat4> mutable_bone_matrices() noexcept {
        return std::span<glm::mat4>(bone_matrices_.data(), bone_matrices_.size());
    }
    [[nodiscard]] std::span<glm::mat4> mutable_world_transforms() noexcept {
        return std::span<glm::mat4>(world_transforms_.data(), world_transforms_.size());
    }

    /// Joints written by the last `evaluate()`; slots at/beyond this index hold
    /// identity, and are re-identitied only when the count SHRINKS.
    [[nodiscard]] std::size_t used_joint_count() const noexcept { return used_joint_count_; }

    /// Same contract as `AnimationPlayer::write_palette`: the used prefix, then
    /// identity for the rest, so uninitialized packet memory is fully written.
    void write_palette(std::span<glm::mat4> dst) const;

    /// A view for `attachment.hpp`'s pose-view overloads.
    [[nodiscard]] PoseView pose() const noexcept {
        return PoseView{world_transforms(), bone_matrices(), used_joint_count_};
    }

    /// The composed LOCAL pose, after every layer and after root-motion
    /// stripping. Useful for a host-side procedural pass and for tests.
    [[nodiscard]] ConstPoseSpan local_pose() const noexcept {
        return ConstPoseSpan(composed_.data(), used_joint_count_);
    }

    void reset();

private:
    /// Per-layer playback clock. A field-for-field mirror of `AnimationPlayer`'s
    /// clock state, advanced by the same statements in the same order — that is
    /// the bit-identity guarantee's whole mechanism.
    struct Clock {
        int current_clip = 0;
        float time = 0.0f;
        bool playing = true;
        bool loop = true;
        float speed = 1.0f;

        int prev_clip = -1;
        float prev_time = 0.0f;
        float blend_factor = 1.0f;
        float blend_duration = 0.2f;

        bool wrapped = false; ///< the clock looped during the last advance (root motion reads it)
        float prev_sample_time = 0.0f; ///< `time` before the last advance
    };

    struct Layer {
        LayerDesc desc;
        Clock clock;
        LayerSource source = LayerSource::Clip;
        BlendSpace1D space_1d;
        BlendSpace2D space_2d;
        glm::vec2 blend_position{0.0f};
        float phase = 0.0f; ///< normalized [0,1) blend-space phase when sync_duration

        /// [0] current clip, [1] crossfade partner for a Clip layer; samplers
        /// [0..n) for a blend space.
        std::array<ClipSampler, k_max_blend_samples> samplers;
    };

    void size_layer_storage(Layer& layer);
    void advance_clock(Layer& layer, const std::vector<AnimationClip>& clips, float dt) noexcept;
    void evaluate_layer(Layer& layer, const Skeleton& skeleton, const std::vector<AnimationClip>& clips,
                        PoseSpan out, PoseScratch& scratch);

    std::array<glm::mat4, MAX_BONES> bone_matrices_{};
    std::array<glm::mat4, MAX_BONES> world_transforms_{};
    std::array<BoneTransform, MAX_BONES> composed_{};
    std::size_t used_joint_count_ = 0;

    std::array<Layer, k_max_animation_layers> layers_{};
    int layer_count_ = 0;
    LayerDesc discard_{}; ///< returned by layer() for an out-of-range index

    const ClipBindings* bindings_ = nullptr;
    std::size_t joint_count_ = 0;

    RootMotionConfig root_motion_{};
    RootMotionDelta root_delta_{};
};

} // namespace bracken::animation
