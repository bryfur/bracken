#pragma once

/**
 * @file blend_space.hpp
 * @brief 1D and 2D blend spaces: a position in, a small set of weighted clips
 *        out.
 *
 * A blend space is a pure function of (samples, query point). It owns no
 * playback state, allocates nothing, and is small enough (a few hundred bytes)
 * to be stored by value inside a state description. That is deliberate: the
 * playback clock lives on the `AnimationRig` layer, so the same blend space can
 * be shared by every character in a crowd without any of them racing.
 *
 * Thread-safety tier: `worker-safe` — `resolve` is const and touches nothing
 * but its arguments.
 */

#include "bracken/animation/animation_types.hpp"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string_view>
#include <vector>

namespace bracken::animation {

/**
 * Per-blend-space sample budget, and a load-bearing constant twice over: it
 * bounds the 2D estimator's O(n^2) inner loop (8 samples -> 56 pair
 * evaluations per resolve, per layer, per character) and it sizes the array of
 * `ClipSampler`s an `AnimationRig` layer holds. Raising it costs quadratic CPU
 * in the resolve and linear memory in every rig with a blend-space layer.
 */
inline constexpr int k_max_blend_samples = 8;

/**
 * Weights below this are culled rather than sampled — a clip contributing
 * 0.0001 of a pose costs a full skeleton sample and moves nothing. Culling also
 * makes "exactly at a sample" resolve to exactly one entry.
 */
inline constexpr float k_blend_weight_epsilon = 1e-4f;

/// One resolved contribution. `duration` is filled in by the caller (the rig)
/// from the clip table; `resolve` leaves it at 0 because a blend space does not
/// own the clips.
///
/// WHAT `speed_scale` ACTUALLY REACHES, because it is less than the name
/// suggests: it is read in exactly one place, `effective_duration`, where it
/// divides that sample's contribution to the layer's shared clock. So —
///  * with `LayerDesc::sync_duration` ON it changes how fast the WHOLE LAYER
///    advances, in proportion to the sample's weight. It cannot make one
///    sample play faster than another, because phase synchronisation means all
///    samples are at the same normalized phase by definition;
///  * with `sync_duration` OFF it does NOTHING AT ALL. The unsynchronised
///    sample time is `fmod(|clock|, duration_k)`, which never reads it.
/// Authoring a per-sample playback rate that survives into the sample time
/// would need a per-sample clock, which the layer does not have.
struct WeightedClip {
    int clip_index = -1;
    float weight = 0.0f;
    float speed_scale = 1.0f;
    float duration = 0.0f;
};

/// Resolve a clip name to its index, or -1. Linear scan — call it at setup, not
/// per frame. Blend spaces take clip INDICES precisely so no string work
/// happens anywhere near the sampling path.
[[nodiscard]] int find_clip_index(const std::vector<AnimationClip>& clips, std::string_view name) noexcept;

/**
 * 1D blend space: samples on a line, interpolated between the bracketing pair.
 *
 * The canonical use is a locomotion axis — idle at 0, walk at 150, run at 400 —
 * with the character's measured speed as the query. Behaviour at the edges is
 * CLAMP, not extrapolate: a speed above the fastest sample plays the fastest
 * clip at weight 1 rather than overshooting into a pose no animator authored.
 */
class BlendSpace1D {
public:
    struct Sample {
        int clip_index = -1;
        float position = 0.0f;
        float speed_scale = 1.0f;
    };

    /// Insert a sample in POSITION ORDER. Returns false (and adds nothing) when
    /// the space is full or the clip index is negative. Samples may be added in
    /// any order and the space is resolvable after every call — `finalize()` is
    /// not a precondition, only a habit.
    [[nodiscard]] bool add_sample(int clip_index, float position, float speed_scale = 1.0f) noexcept;

    /// Re-establishes the ascending-position invariant `resolve` needs.
    /// IDEMPOTENT and, since `add_sample` became a sorted insert, redundant;
    /// it is kept because it is the documented setup step and because a space
    /// that was never finalized used to produce garbage weights in release
    /// rather than a defined answer. Calling it costs a sort of at most
    /// `k_max_blend_samples` elements.
    void finalize() noexcept;

    [[nodiscard]] int sample_count() const noexcept { return count_; }
    [[nodiscard]] bool finalized() const noexcept { return finalized_; }
    [[nodiscard]] const Sample& sample(int i) const noexcept { return samples_[static_cast<std::size_t>(i)]; }

    /**
     * Weights for `x`. Writes at most 2 entries; `out` must have room for 2.
     * Returns the number written.
     *
     * Properties, all pinned by `blend_space.test.cpp`:
     *  - exact at every sample (one entry, weight 1);
     *  - partition of unity everywhere (weights sum to 1 within float error);
     *  - C0 continuous, piecewise linear;
     *  - an empty space returns 0 entries, a one-sample space returns that
     *    sample at weight 1 for every `x`.
     */
    [[nodiscard]] std::uint32_t resolve(float x, std::span<WeightedClip> out) const noexcept;

private:
    std::array<Sample, k_max_blend_samples> samples_{};
    int count_ = 0;
    bool finalized_ = false;
};

/**
 * 2D blend space, GRADIENT-BAND FREEFORM DIRECTIONAL (polar).
 *
 * WHY THIS AND NOT A TRIANGULATION
 * --------------------------------
 * Barycentric interpolation over a Delaunay triangulation is the other standard
 * answer and it is a worse fit here for three reasons. It needs a triangulation
 * step whose degenerate cases (collinear samples, duplicate points, a query
 * outside the hull) each need their own defined answer, so the "degenerate cases
 * are defined rather than UB" requirement costs more code than the estimator
 * itself. It behaves badly for the case this feature exists to serve — a
 * locomotion set of one centre sample plus eight on a circle triangulates into
 * long thin slivers, and a query near the centre picks up whichever sliver it
 * happens to land in. And the polar formulation is the one that understands
 * that (0,0) means "not moving" and that the angle between two directions
 * matters more than their cartesian distance, which is exactly the semantics of
 * a locomotion blend.
 *
 * THE ESTIMATOR
 * -------------
 * For each sample i, with r_k = |p_k| and `signed_angle` measured about the
 * origin:
 *
 *     avg   = (r_i + r_j) / 2
 *     v_ij  = ( (r_j - r_i) / avg,  signed_angle(p_i, p_j) * k_direction_scale )
 *     v_ip  = ( (r   - r_i) / avg,  signed_angle(p_i, p)   * k_direction_scale )
 *     h_ij  = clamp01( 1 - dot(v_ip, v_ij) / dot(v_ij, v_ij) )
 *     w_i   = min over j != i of h_ij
 *
 * then cull below `k_blend_weight_epsilon` and normalize. The angle term is
 * ZERO when either operand has ~zero magnitude (a centre sample has no
 * direction), and `dot(v_ij, v_ij) ~ 0` (coincident samples) yields h = 1 — a
 * sample cannot constrain itself.
 *
 * Exactness at a sample falls straight out: at `p == p_k`, `v_kp` is the zero
 * vector so every `h_kj` is 1 and `w_k` is 1, while for every other i the term
 * `h_ik` evaluates to exactly 0 because `v_ip == v_ik`.
 *
 * KNOWN PROPERTY, not a bug: the estimator is C0 but NOT C1. `min()` creates a
 * crease wherever the binding constraint changes, so a character crossing one
 * gets a visible kink in the blend rather than a smooth roll. This is inherent
 * to gradient band (Unity ships the same artifact) and it is the price of not
 * building and maintaining a triangulation. `blend_space.test.cpp` pins the
 * continuity bound so it stays a known property.
 */
class BlendSpace2D {
public:
    struct Sample {
        int clip_index = -1;
        glm::vec2 position{0.0f};
        float speed_scale = 1.0f;
    };

    /**
     * How much an angular difference counts relative to a radial one.
     *
     * TUNED CONSTANT WITH NO MEASURED JUSTIFICATION IN THIS TREE: 2.0 is the
     * value the technique is normally published with (it makes a 90-degree
     * direction change weigh about as much as a doubling of magnitude), and it
     * is what makes a strafe sample stop bleeding into a forward one. Nothing
     * here measured it against a real locomotion set, because this tree has no
     * such set to measure against — see docs/gap_anim.md.
     */
    static constexpr float k_direction_scale = 2.0f;

    [[nodiscard]] bool add_sample(int clip_index, glm::vec2 position, float speed_scale = 1.0f) noexcept;

    [[nodiscard]] int sample_count() const noexcept { return count_; }
    [[nodiscard]] const Sample& sample(int i) const noexcept { return samples_[static_cast<std::size_t>(i)]; }

    /**
     * Weights for `p`. Writes at most `k_max_blend_samples` entries; returns the
     * number written.
     *
     * DEGENERATE CASES ARE DEFINED, all pinned by test:
     *  - zero samples -> 0 entries;
     *  - one sample -> that sample at weight 1, for every `p`;
     *  - coincident samples -> they share the weight evenly, deterministically;
     *  - every sample at the origin -> uniform weights;
     *  - a query so far outside the cover that every weight culls to zero ->
     *    the single NEAREST sample at weight 1, never an empty result and never
     *    a divide by a zero sum.
     */
    [[nodiscard]] std::uint32_t resolve(glm::vec2 p, std::span<WeightedClip> out) const noexcept;

private:
    std::array<Sample, k_max_blend_samples> samples_{};
    int count_ = 0;
};

/**
 * The weighted mean clip duration of an active set — the denominator of the
 * phase-synchronised clock.
 *
 * Without phase sync a walk/run blend double-steps: the two clips have
 * different durations, so their feet drift apart and the character appears to
 * skate. With it, the layer holds ONE normalized phase, advances it by
 * `dt * speed / effective_duration`, and evaluates sample i at
 * `phase * duration_i` — so every sample is at the same point in its stride.
 *
 * `duration_i` is divided by `|speed_scale_i|`, which is how an authored clip
 * that plays at 1.5x contributes its EFFECTIVE length. The magnitude is
 * deliberate: a negative scale means "backwards", not "negative length".
 * Returns 0 when the weights sum to zero or every duration is zero; the caller
 * must treat 0 as "do not advance".
 */
[[nodiscard]] float effective_duration(std::span<const WeightedClip> active) noexcept;

} // namespace bracken::animation
