#pragma once

// Weighted reservoir sampling (WRS) + RIS reservoir merge — the CPU mirror of
// what `shaders/rt_local_shadow.comp.hlsl` does per pixel.
//
// WHY A MIRROR AT ALL. The reservoir is the one part of the froxel-RIS local
// shadow pass whose correctness is a MATHEMATICAL claim rather than a visual
// one: a streaming reservoir must select each candidate with probability
// proportional to its weight, and merging two reservoirs must be equivalent to
// having streamed both candidate sets through one. Neither is observable in a
// screenshot — a subtly biased estimator just makes the shadows slightly the
// wrong darkness — so both are pinned in `reservoir.test.cpp` against a header
// a test can exercise with no GPU, and transliterated into HLSL exactly once.
//
// GPU-free on purpose: no `gpu/`, no SDL — same rule as cluster_index.hpp.

#include <cstdint>
#include <span>

namespace bracken::render::lighting {

/// "This light is not in the frame any more." Mirrors LIGHT_REMAP_INVALID in
/// shaders/cluster_compute.hlsli and render::lighting::LIGHT_REMAP_INVALID.
inline constexpr uint32_t RESERVOIR_NO_LIGHT = 0xFFFFFFFFu;

/**
 * One pixel's STORED reservoir — the 16 bytes that cross the frame boundary.
 *
 * 16 is a budget, not a coincidence: at half res on a 1600x900 frame the
 * ping-pong pair costs 2 x 2.7 MB, and the pass reads one and writes the other
 * every frame.
 *
 * NOTE WHAT IS NOT HERE: the selected candidate's target-function value p̂(y).
 * It looks like it belongs — every ReSTIR write-up carries it — and it is dead
 * weight in a temporal-only reuse scheme, because the only consumer of a stored
 * reservoir is next frame's merge, and that merge must re-evaluate p̂ at the
 * CURRENT surface point anyway (`ReservoirBuilder::merge`'s `target_pdf_here`).
 * A stored p̂ would be the value at last frame's shading point, which is the one
 * number the merge is specifically not allowed to use. The freed lane carries
 * `depth` instead, which the merge genuinely cannot reconstruct.
 *
 * `light_index` is the PACKED light index (the index `loadLightHeader()`
 * speaks), valid only for the frame it was written in — which is exactly why
 * the remap buffer exists: next frame resolves it through `remap_reservoir()`
 * before the history means anything.
 */
struct Reservoir {
    /// Selected candidate, or RESERVOIR_NO_LIGHT when nothing was selected
    /// (empty froxel, or every candidate had zero target weight).
    uint32_t light_index = RESERVOIR_NO_LIGHT;
    /// Unbiased contribution weight: w_sum / (M * p̂(y)). The number the
    /// estimator multiplies by; NOT a probability.
    float weight = 0.0f;
    /// Candidates this reservoir represents. A float rather than a uint because
    /// the temporal clamp scales it and the shader stores it in a float lane.
    float sample_count = 0.0f;
    /// Positive linear view depth at the pixel that wrote it. Purely a
    /// disocclusion test for next frame's reprojection — the same job, and the
    /// same relative-epsilon comparison, as the alpha lane of the sun shadow
    /// denoiser's history.
    float depth = 0.0f;
};

static_assert(sizeof(Reservoir) == 16, "the shader's RtReservoir is 16 bytes; keep the pair in step");

/**
 * Streaming state: the record above plus the two values that never have to
 * cross a frame boundary — the running weight sum and the selected candidate's
 * p̂ under THIS pixel's target function.
 */
struct ReservoirBuilder {
    Reservoir r{};
    float w_sum = 0.0f;
    /// p̂(y) of the current selection, evaluated here. `finalize()` divides by
    /// it; nothing stores it.
    float target_pdf = 0.0f;

    /**
     * Stream one candidate in. `weight` is the RIS weight p̂(x)/p(x) — target
     * function over source pdf — and `target_pdf` is p̂(x) itself.
     *
     * `rand01` must be uniform in [0, 1). The comparison is `< w / w_sum` with
     * the NEW w already folded in — the standard Chao/Efraimidis update, which
     * is what makes P(select x_i) = w_i / sum(w) exactly rather than
     * approximately.
     *
     * A non-positive weight still counts toward M. Dropping it would bias the
     * estimator by pretending the candidate was never drawn.
     */
    void update(uint32_t light_index, float weight, float candidate_pdf, float rand01) {
        r.sample_count += 1.0f;
        if (!(weight > 0.0f)) {
            return; // NaN-safe: a NaN weight fails this test and is discarded
        }
        w_sum += weight;
        if (rand01 * w_sum < weight) {
            r.light_index = light_index;
            target_pdf = candidate_pdf;
        }
    }

    /**
     * Merge another reservoir's candidate in as one sample carrying its whole
     * history.
     *
     * `target_pdf_here` is the other reservoir's candidate re-evaluated under
     * THIS pixel's target function — see the note on Reservoir about why the
     * stored value would be the wrong one. It is also the mechanism that makes
     * a merge across a surface change self-correcting without a normal test:
     * a light that is behind, out of range of, or facing away from the new
     * surface evaluates to zero here, and a zero-weight merge changes nothing
     * but M.
     */
    void merge(const Reservoir& other, float target_pdf_here, float rand01) {
        if (other.light_index == RESERVOIR_NO_LIGHT || other.sample_count <= 0.0f) {
            return;
        }
        const float weight = target_pdf_here * other.weight * other.sample_count;
        const float m_before = r.sample_count;
        update(other.light_index, weight, target_pdf_here, rand01);
        // update() added ONE to M; this candidate stands for other.sample_count
        // of them. Setting M after keeps update()'s contract ("one candidate,
        // one M") intact for the RIS loop that shares it.
        r.sample_count = m_before + other.sample_count;
    }

    /**
     * Close the reservoir: W = w_sum / (M * p̂(y)).
     *
     * Call once, after the last update/merge. A zero p̂ — nothing selected, or a
     * selection the target function evaluates to zero — yields W = 0, which is
     * the correct "this pixel's estimate is zero".
     */
    void finalize() {
        const float denom = r.sample_count * target_pdf;
        r.weight = denom > 0.0f ? w_sum / denom : 0.0f;
    }
};

/**
 * Balance-heuristic scale for one candidate source in a MULTI-SOURCE RIS draw
 * (R6, docs/rt_lighting_plan.md §5.7).
 *
 * WHAT PROBLEM THIS SOLVES. Since R6 the local-light reservoir is fed by two
 * candidate streams: `m_source` analytic lights drawn uniformly from the
 * pixel's froxel, and a second batch drawn from a power-weighted alias table
 * over emissive triangles. Their domains are DISJOINT — no draw from one can
 * ever produce a sample of the other — and the quantity wanted is the SUM of
 * the two integrals, `I_analytic + I_emissive`.
 *
 * Feeding both streams into one reservoir with the shipped `1/M` normalization
 * would instead deliver
 *
 *     (M_a * I_a + M_e * I_e) / (M_a + M_e)
 *
 * a convex COMBINATION of the two halves — i.e. the frame dimmed toward
 * whichever source happened to be drawn more often. That is an energy error,
 * not a noise one, and it is invisible in a screenshot because it looks exactly
 * like "the lighting is a bit dark".
 *
 * (Note the contrast with the TEMPORAL merge, where a convex combination is
 * exactly right: `merge()` above combines two estimates of the SAME integral,
 * so weighting them by confidence and summing to one is the correct thing.)
 *
 * THE FIX. Apply the balance heuristic per candidate:
 * `m_i(x) = p_i(x) / sum_j p_j(x)` over all M draws. With disjoint domains
 * every cross term vanishes, so `m` collapses to `1/M_source` for a candidate
 * of that source. Folding that into each stream weight while KEEPING the
 * shipped finalize (`W = w_sum / (M_total * p̂)`) leaves exactly one factor per
 * candidate, and this function is that factor:
 *
 *     w_i = ris_source_scale(M_source, M_current) * p̂(x_i) / p_source(x_i)
 *
 * with `M_current = M_a + M_e` (this frame's candidates, history excluded — the
 * history carries its own confidence weight through `merge()`).
 *
 * WHY IT IS A NO-OP FOR ONE SOURCE. With `m_source == m_total_current` this
 * returns exactly 1.0f, and multiplying a float by 1.0f is exact — which is why
 * every pre-R6 configuration reproduces its baseline BIT for bit rather than
 * merely closely.
 *
 * Mirrored in `shaders/rt_local_shadow.comp.hlsl` as `sourceScaleA` /
 * `sourceScaleE`.
 */
[[nodiscard]] inline float ris_source_scale(uint32_t m_source, uint32_t m_total_current) {
    if (m_source == 0) {
        return 0.0f;
    }
    return static_cast<float>(m_total_current) / static_cast<float>(m_source);
}

/**
 * SPATIAL reuse of a neighbouring pixel's reservoir, with PAIRWISE MIS
 * (docs/rt_lighting_plan.md §5.3's "optional spatial pass (3-5 neighbours,
 * pairwise MIS)", landed as R7a).
 *
 * WHY THIS IS NOT `merge()`. The temporal merge above combines two estimates of
 * the SAME integral at the SAME point, so weighting them by confidence
 * (`m_j = M_j / sum M`) and letting `finalize()` divide by the total M is both
 * correct and cheap. A spatial neighbour is a different point: the light it
 * picked may be one THIS pixel could never have drawn (a different froxel, a
 * light behind this surface, an occluder between), and the confidence weighting
 * has no way to notice. Reusing it under `m_j = M_j / sum M` is the classic
 * ReSTIR spatial bias — visible as brightening along depth discontinuities and
 * as light leaking around corners.
 *
 * THE WEIGHTS. Write `p̂_c` for this pixel's target function and `p̂_i` for
 * neighbour i's, `M_c` / `M_i` for their confidences, and `k` for the number of
 * neighbours actually accepted. Pair each neighbour against the canonical:
 *
 *     a_i(x) = M_i p̂_i(x) / ( M_i p̂_i(x) + M_c p̂_c(x) )
 *     b_i(x) = 1 - a_i(x) = M_c p̂_c(x) / ( M_i p̂_i(x) + M_c p̂_c(x) )
 *
 *     m_i(x) = a_i(x) / k                 (neighbour i)
 *     m_c(x) = (1/k) * sum_i b_i(x)       (the canonical)
 *
 * These sum to 1 for EVERY x, unconditionally: `sum_i m_i + m_c =
 * (1/k) sum_i (a_i + b_i) = 1`. That is the whole unbiasedness argument, and it
 * survives the two degenerate cases that matter — a light outside neighbour i's
 * domain gives `p̂_i = 0`, hence `a_i = 0` and `b_i = 1`; a light outside the
 * canonical's gives `p̂_c = 0`, hence `a_i = 1` and `b_i = 0`. In both the pair
 * still sums to one and the input that could not have produced x gets weight
 * zero, which is exactly the condition generalized RIS asks for.
 *
 * THE ESTIMATOR. Each input contributes `w_j = m_j(X_j) * p̂_c(X_j) * W_j`, one
 * survives by weighted reservoir sampling, and
 *
 *     W_out = ( sum_j w_j ) / p̂_c(y)
 *
 * with NO division by M — the MIS weights already normalize, which is the
 * structural difference from `ReservoirBuilder::finalize()`. The `1/k` is
 * factored out of every weight and applied once at the end, so it cannot
 * influence the selection and the streaming order cannot influence `w_sum`.
 *
 * THE ONE APPROXIMATION, NAMED. The derivation needs `p̂_j(x) = 0` exactly when
 * x is outside input j's candidate domain. In the shipped pass the domain is the
 * neighbour's froxel light list and the target function is its unshadowed
 * contribution, which vanishes out of range and behind the surface — so the two
 * agree wherever the cluster grid's range binning agrees with the light's own
 * falloff, and disagree for a light that is in range of the neighbour's froxel
 * but was never drawn into its M candidates. That residual is the standard
 * practical bias of every shipped ReSTIR spatial pass; the tests below pin the
 * exact case (explicit disjoint domains, `p̂ = 0` outside) because it is the one
 * the maths actually claims.
 *
 * COST, per neighbour: three target-function evaluations —
 * `p̂_c(X_i)` (its sample here), `p̂_i(X_i)` (its sample there) and `p̂_i(X_c)`
 * (our sample there). The last is what makes it PAIRWISE rather than 1/Z, and
 * it is why the shader reconstructs each neighbour's surface rather than only
 * gating on its depth.
 *
 * Mirrored in `shaders/rt_local_shadow.comp.hlsl`'s spatial stage.
 */
struct PairwiseSpatialMerge {
    /// The running selection. `sample_count` accumulates confidence as inputs
    /// arrive; `light_index` and `weight` are only meaningful after `finalize`.
    Reservoir r{};
    /// Sum of the `m_j * p̂_c * W_j` weights, WITHOUT the shared `1/k`.
    float w_sum = 0.0f;
    /// p̂_c of the current selection. `finalize()` divides by it.
    float target_pdf = 0.0f;
    /// `sum_i b_i(X_c)` — the canonical's MIS weight, accumulated one neighbour
    /// at a time because it cannot be known until every neighbour has been seen.
    float canonical_mis_sum = 0.0f;
    /// Neighbours actually accepted. `k` in the weights above.
    uint32_t neighbors = 0;

    Reservoir canonical{};
    /// p̂_c(X_c) — the canonical's own sample under its own target function.
    float canonical_pdf = 0.0f;

    /**
     * Seat the canonical reservoir. `canonical_pdf_here` is p̂_c(X_c): the
     * centre pixel's selected candidate re-evaluated at the centre pixel, which
     * is the value `ReservoirBuilder::finalize()` already had and did not store.
     */
    void begin(const Reservoir& c, float canonical_pdf_here) {
        canonical = c;
        canonical_pdf = canonical_pdf_here;
        r = Reservoir{};
        w_sum = 0.0f;
        target_pdf = 0.0f;
        canonical_mis_sum = 0.0f;
        neighbors = 0;
    }

    /**
     * Fold one neighbour in.
     *
     * @param other              the neighbour's reservoir, already depth/normal gated.
     * @param other_pdf_here     p̂_c(X_i) — its sample, evaluated at THIS pixel.
     * @param other_pdf_there    p̂_i(X_i) — its sample, evaluated at ITS pixel.
     * @param canonical_pdf_there p̂_i(X_c) — OUR sample, evaluated at its pixel.
     * @param rand01             uniform in [0, 1).
     *
     * A neighbour holding no sample is not an input and does not count toward
     * `k`: it can contribute nothing, and letting it dilute the canonical's MIS
     * weight would darken the pixel for no reason.
     */
    void add_neighbor(const Reservoir& other, float other_pdf_here, float other_pdf_there,
                      float canonical_pdf_there, float rand01) {
        if (other.light_index == RESERVOIR_NO_LIGHT || other.sample_count <= 0.0f) {
            return;
        }
        ++neighbors;

        // a_i(X_i): how much of X_i's weight this pairing gives the neighbour.
        const float a_num = other.sample_count * other_pdf_there;
        const float a_den = a_num + canonical.sample_count * other_pdf_here;
        const float a_i = a_den > 0.0f ? a_num / a_den : 0.0f;

        // b_i(X_c): the canonical's share of ITS OWN sample under this pairing.
        const float b_num = canonical.sample_count * canonical_pdf;
        const float b_den = b_num + other.sample_count * canonical_pdf_there;
        canonical_mis_sum += b_den > 0.0f ? b_num / b_den : 0.0f;

        // Confidence adds whether or not the sample survives the draw: the
        // neighbour's M candidates genuinely inform this pixel now.
        r.sample_count += other.sample_count;

        const float w = a_i * other_pdf_here * other.weight;
        if (!(w > 0.0f)) {
            return; // NaN-safe, as ReservoirBuilder::update is
        }
        w_sum += w;
        if (rand01 * w_sum < w) {
            r.light_index = other.light_index;
            target_pdf = other_pdf_here;
        }
    }

    /**
     * Stream the canonical in LAST — its MIS weight depends on every neighbour —
     * and close the reservoir.
     *
     * With no accepted neighbour this returns the canonical UNCHANGED, bit for
     * bit. That is the contract the shipped `spatial_neighbors = 0` path leans
     * on: turning spatial reuse off must reproduce the pre-R7a image exactly,
     * not merely closely.
     */
    [[nodiscard]] Reservoir finalize(float rand01) {
        if (neighbors == 0) {
            return canonical;
        }
        const float w_c = canonical_mis_sum * canonical_pdf * canonical.weight;
        r.sample_count += canonical.sample_count;
        if (w_c > 0.0f) {
            w_sum += w_c;
            if (rand01 * w_sum < w_c) {
                r.light_index = canonical.light_index;
                target_pdf = canonical_pdf;
            }
        }
        // The shared 1/k, applied once, where it cannot touch the selection.
        const float denom = static_cast<float>(neighbors) * target_pdf;
        r.weight = denom > 0.0f ? w_sum / denom : 0.0f;
        // The record is this PIXEL's from here on, so it carries this pixel's
        // depth — the lane is a disocclusion test for next frame's reprojection
        // and a neighbour's depth would answer the wrong question.
        r.depth = canonical.depth;
        return r;
    }
};

/**
 * Resolve a reservoir carried over from the previous frame against this frame's
 * light set.
 *
 * `remap` is ClusterGrid's prev-packed-index -> current-packed-index table
 * (`light_remap_buffer()`), holding RESERVOIR_NO_LIGHT where the light vanished.
 * A vanished light means the history names a light that is not in the frame, and
 * the only correct answer is to drop it: reusing it would shadow against a light
 * nothing is lit by, and re-pointing it at whatever now occupies that index
 * would shadow against an unrelated one — which is precisely the failure the
 * remap exists to prevent, because packed indices are rebuilt every frame.
 *
 * @return the reservoir with `light_index` in CURRENT packed indices, or a
 *   default-constructed (empty) reservoir when the history cannot be carried.
 */
[[nodiscard]] inline Reservoir remap_reservoir(const Reservoir& prev, std::span<const uint32_t> remap) {
    if (prev.light_index == RESERVOIR_NO_LIGHT || prev.light_index >= remap.size()) {
        return Reservoir{};
    }
    const uint32_t now = remap[prev.light_index];
    if (now == RESERVOIR_NO_LIGHT) {
        return Reservoir{};
    }
    Reservoir out = prev;
    out.light_index = now;
    return out;
}

/**
 * Clamp a carried-over reservoir's M so history cannot outvote the present.
 *
 * `max_ratio` x this frame's candidate count is the cap (20x in the shipped
 * pass, the ReSTIR-standard figure): without it M grows without bound and the
 * temporal estimate stops responding to a light that moved, dimmed or changed
 * colour. Scaling M alone is correct because W is already normalized BY M — the
 * estimate is untouched, only its inertia is.
 */
[[nodiscard]] inline Reservoir clamp_reservoir_history(const Reservoir& prev, float current_m, float max_ratio) {
    Reservoir out = prev;
    const float cap = current_m * max_ratio;
    if (cap > 0.0f && out.sample_count > cap) {
        out.sample_count = cap;
    }
    return out;
}

} // namespace bracken::render::lighting
