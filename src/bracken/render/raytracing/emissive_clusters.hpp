#pragma once

// PER-FROXEL emissive candidate lists — R8, docs/rt_lighting_plan.md §5.7's
// follow-up item 2 ("the ~2.1 ms above is a scene-wide alias draw paying for
// long rays; binning emitters into the cluster grid the way point lights are
// would localize both").
//
// THE PROBLEM, restated in one line. R6 drew every emissive candidate from a
// SCENE-WIDE power-weighted alias table, so a pixel in the north of the village
// regularly won a window triangle in the south and aimed a 2000-unit any-hit ray
// at it. Analytic point and spot candidates have been froxel-local since R2b;
// this makes emissive ones local too, and the measured price of NOT doing so was
// 2.11 ms of device time on the playground village.
//
// ---------------------------------------------------------------------------
// A MIXTURE, NOT A SECOND STREAM — the shape of the thing
// ---------------------------------------------------------------------------
//
// The obvious shape is a THIRD candidate stream (analytic / froxel-emissive /
// global-emissive) under the multi-source RIS weights R6 already derived. It
// needs the three domains to be disjoint, which for the two emissive halves they
// are not: a record bright enough to be in the global fallback is exactly the
// record most likely to also be in the local list. Making them disjoint by
// construction — excluding globally-bright records from froxel lists — is
// possible and is WRONG for the scene this exists to light, because it would
// take the brightest emitter near a pixel out of that pixel's local list.
//
// So the emissive stream stays ONE stream and its SOURCE PDF becomes a mixture
// over the same domain it always had:
//
//     p(x) = q_f * [x in L_f] / |L_f|  +  q_g * power(x) / totalPower
//
// with `q_f = emissive_froxel_share` and `q_g = 1 - q_f`. THE DEDUPE IS THE SUM:
// a record reachable both ways simply carries both terms, which is what "dedupe"
// means for a mixture and why there is no partition to get wrong. The two
// indicator lookups are deliberately asymmetric in cost — a froxel-drawn record
// gets `power/totalPower` out of the record the draw loads anyway, while a
// globally-drawn one pays a bounded scan of the (ascending, early-outing) list.
//
// `q_f = 0` collapses p(x) to `power/totalPower` EXACTLY, which is R6's draw bit
// for bit. That is the A/B control and it is why every pre-R8 measurement is
// reproducible from the shipped binary.
//
// ---------------------------------------------------------------------------
// WHY THE SHIPPED DEFAULT IS q_f = 1, AND WHAT THAT COSTS
// ---------------------------------------------------------------------------
//
// Every intermediate value of q_f is UNBIASED over the full record set — the
// lighthouse no froxel list holds is still drawn, at `q_g * power/total` — and
// every intermediate value RECOVERS NOTHING. That is the central measurement of
// this package and it is not intuitive, so it is written down:
//
//     q_f        night luma      frame render      device cost of the records
//     0 (R6)     37.0            4.724 ms          2.11 ms
//     0.75       37.2            4.240 ms          ~2.0 ms
//     0.90       37.4            4.249 ms          ~2.0 ms
//     0.97       37.9            4.199 ms          ~2.0 ms
//     1.0        36.0            3.78  ms          0.89 ms  (+0.29 ms CPU)
//
// A CLIFF, NOT A SLOPE, and the reason is the dispatch geometry: this pass is
// 8x4 threads per tile, one thread walking the whole acceleration structure sets
// the cost of all 32, and at ANY nonzero global rate nearly every tile contains
// one such thread. Stratifying the global half by TILE (a quarter of tiles pay,
// the rest run local) was implemented and measured WORSE than either end — the
// R7a spatial reservoir reuse reaches across tile boundaries and propagates the
// far records, and the long rays with them.
//
// So the recovery is only available at q_f = 1, and at q_f = 1 a froxel's list
// IS the candidate domain — the same contract analytic point and spot lights
// have had since R2b, where a light not binned into your froxel is not sampled.
// **The honest cost is 1.0 luma of far-field emissive fill at the night pin**:
// a village's glow is a sum over eight thousand dim windows and no bounded
// per-froxel list holds more than a few dozen of them. That was verified rather
// than assumed — packing every froxel to 64 entries measured 35.6, WORSE than
// 16, so it was never a coverage problem — and it is why `emissive_froxel_share`
// stays a documented dial rather than a constant. In exchange the DI temporal
// noise probe drops 3.2x (0.00045 -> 0.00014), because a local list is a far
// better importance function for the near field than a scene-wide power table.
//
// ---------------------------------------------------------------------------
// THE RAY IS NOT TRUNCATED, AND THAT WAS MEASURED TOO
// ---------------------------------------------------------------------------
//
// A record only enters a froxel's list if its unshadowed contribution AT THE
// FROXEL CENTRE clears `k_emissive_froxel_min_contribution` — i.e. if it is
// within `min(sqrt(power / threshold), k_emissive_froxel_max_radius)` of that
// centre. So a list-drawn candidate's shadow ray is bounded by that radius plus
// the froxel's own extent, and the bound is enforced HERE, at binning time.
//
// Re-applying it as a `min()` on the shader's ray length is the obvious
// companion move, it is cheap, and it is wrong: a distant emitter behind a
// building then comes back VISIBLE, and the night village measured 37.0 -> 43.8
// mean luma under it. Light through walls, not a speedup. `rtEmissiveRayLength`
// carries that note where the ray is actually built.
//
// The other named approximation is the CAP: a froxel holds at most
// `k_emissive_froxel_capacity` records, chosen by contribution at its centre.
// Over-cap insertions are COUNTED (`EmissiveClusterStats::entries_dropped`), per
// the engine's no-silent-drops rule.
//
// CPU mirror rules, same as emissive_lights.hpp / reservoir.hpp: the sampling
// arithmetic below is transliterated into `shaders/rt_emissive.hlsli` exactly
// once and pinned by `emissive_clusters.test.cpp`, because "the estimator is
// unbiased over the mixture's domain" is a mathematical claim no screenshot can
// check.

#include "bracken/render/lighting/cluster_index.hpp"
#include "bracken/render/raytracing/emissive_lights.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <vector>

struct SDL_GPUBuffer;
struct SDL_GPUCommandBuffer;

namespace bracken::core::jobs {
class JobSystem;
}
namespace bracken::gpu {
class GPUBuffer;
class GPUDevice;
} // namespace bracken::gpu

namespace bracken::render::rt {

/**
 * Emissive records one froxel may hold.
 *
 * 16, matching nothing else on purpose: `MAX_LIGHTS_PER_CLUSTER` is 128 because
 * a fragment shader LOOPS its froxel's analytic lights and needs them all, while
 * this list is only ever SAMPLED — two draws per pixel per frame. Every entry
 * costs 4 bytes x 3456 froxels of upload per frame plus a step of the membership
 * scan; 16 keeps the table at 235 KB and the scan inside one cache line's worth
 * of loads.
 *
 * 64 was measured against it and lost on both counts: the night pin moved 36.0
 * -> 36.2 luma (inside the run-to-run band), the DI noise probe went the wrong
 * way (0.00014 -> 0.00019), the table grew to 900 KB and the frame did not move.
 * Widening this is NOT the lever for the far-field fill q_f = 1 gives up — see
 * the header — because that fill is a sum over thousands of records and no
 * bounded list reaches it.
 */
inline constexpr uint32_t k_emissive_froxel_capacity = 16;

/**
 * Minimum unshadowed contribution `power / max(d^2, area)` at a froxel's CENTRE
 * for a record to enter that froxel's list.
 *
 * The same quantity `emissive_target()` computes with the two cosines set to 1,
 * so it is directly commensurable with the analytic path's target function. It
 * is a SAMPLE of p̂ at that one point, not a bound over the froxel: a record
 * nearer the froxel than its centre is under-scored, by up to (D / (D - h))^2
 * for a froxel of half-extent h, and can therefore miss the list of the froxel
 * it sits in.
 *
 * Its other face is a RELEVANCE RADIUS, `sqrt(power / threshold)`, which is
 * what makes the bin O(records x touched froxels) instead of
 * O(records x CLUSTER_COUNT) — the same role a point light's `radius` plays for
 * the light binner, derived rather than authored because a triangle has no
 * range field to author.
 *
 * 0.002 in the engine's linear units. Measured against the playground village:
 * a lit-window record (power ~ 10^2) reaches a couple of hundred units, an orb
 * facet (power ~ 10) a few tens, and the brightest emitters run into the horizon
 * below. Lowering it costs bin time in proportion to the touched-froxel count
 * and buys nothing — at 0.00002 every one of the 3456 froxels fills to capacity,
 * the bin takes 30 ms, and the night pin reads 35.6, WORSE than at 0.002.
 * Raising it hands dim near-field emitters back to the global half of the
 * mixture, which is correct and merely noisier.
 */
inline constexpr float k_emissive_froxel_min_contribution = 0.002f;

/**
 * Hard ceiling on that relevance radius, in world units — **the localization
 * horizon, and the ray-length bound this whole package exists to buy**.
 *
 * A record only enters a froxel's list if it is within
 * `min(sqrt(power / threshold), this)` of the froxel centre, so a list-drawn
 * candidate's shadow ray is bounded by this plus the froxel's own extent. That
 * bound is enforced HERE and not as a `min()` on the shader's ray length: an
 * artificial clamp would drop occluders that genuinely sit between the surface
 * and the emitter, and R5b's gradient replay would have to reproduce it from a
 * table it does not bind. Candidates drawn through the GLOBAL half of the
 * mixture keep unbounded rays, as they must.
 *
 * It is also what keeps the bin affordable. The threshold alone gives a
 * sun-bright emitter a relevance radius of thousands of units, whose
 * conservative froxel box is most of the grid — and every froxel in it costs a
 * distance test to conclude what the per-froxel cap was going to conclude
 * anyway. 400 units is where the playground's records stop reaching new froxels
 * in practice: 1200 and uncapped both produce the identical table.
 */
inline constexpr float k_emissive_froxel_max_radius = 400.0f;

/**
 * Share of each emissive candidate draw taken from the froxel list. Shipped
 * default; the live value rides `RtLocalShadowUniforms::emissiveFroxelShare`.
 *
 * 1.0, and the header explains why it is an end of the range rather than a point
 * inside it: every intermediate value is unbiased and recovers no device time,
 * because one long ray sets the cost of its whole 8x4 tile. 0 is R6 exactly and
 * is the A/B control. Values in between trade the recovery back for far-field
 * emissive fill and are a legitimate thing for a host to want — they are just
 * not the default, since the recovery is the point of the package.
 */
inline constexpr float k_emissive_froxel_share = 1.0f;

/**
 * Flat table layout, mirrored by `RT_EMISSIVE_FROXEL_*` in
 * shaders/rt_emissive.hlsli. Fixed stride, so a froxel's block is addressable
 * without a prefix sum — which is what lets the binning fan-out pack its own
 * froxels inside the chunk instead of needing a serial pack afterwards.
 *
 *     [base + 0]        entry count
 *     [base + 1 + i]    record index, ASCENDING
 *
 * ONE INDEX PER ENTRY, AND THE SHADER DRAWS THE LIST UNIFORMLY. Drawing it
 * proportional to the same relevance score that put a record in it is the
 * textbook improvement and it was implemented — a second half-block carrying the
 * cumulative score, a linear CDF search in the shader — then measured and
 * removed: 0.4 ms of device time for no change in mean luma and none in the
 * `di noise` probe. The list is short, the reservoir re-weights every candidate
 * by p̂ regardless of how it was drawn, and a 16-step dependent-load search is
 * expensive next to what it saves. Ascending order is kept because the `q_g`
 * half of the mixture pdf has to answer "is this record in the list", and
 * ascending lets that scan early-out.
 */
inline constexpr uint32_t k_emissive_froxel_stride = k_emissive_froxel_capacity + 1;

/// Entries of the flat table for `froxel_count` froxels.
[[nodiscard]] inline constexpr uint32_t emissive_cluster_table_size(uint32_t froxel_count) {
    return froxel_count * k_emissive_froxel_stride;
}

/**
 * "Do not memoize me" — the record generation a caller passes when it has no
 * generation to offer, which forces `build()` to run.
 *
 * A SENTINEL RATHER THAN A DEFAULT ARGUMENT, deliberately. The table is a pure
 * function of (record CONTENTS, view, projection, z planes, grid), and every
 * one of those is visible in `build()`'s arguments except the first: two
 * different record vectors can occupy the same heap address at the same size,
 * which is exactly what a test or a bench that builds temporaries does. A
 * defaulted generation would make both carry 0 and let the second build be
 * silently skipped against the first one's table. Making the opt-in explicit
 * keeps every CPU-only caller ungated and correct by construction.
 */
inline constexpr uint32_t k_emissive_no_generation = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// The CPU mirror of the shader's mixture draw.
// ---------------------------------------------------------------------------

/// Mirror of `rtEmissiveSampleIndex` — the global power-weighted alias draw,
/// with the Vose row riding inside the record (emissive_lights.hpp).
[[nodiscard]] inline uint32_t emissive_sample_index(std::span<const EmissiveTriangle> records, float u1,
                                                    float u2) {
    const auto n = static_cast<uint32_t>(records.size());
    if (n == 0) {
        return 0;
    }
    const uint32_t bucket = std::min(static_cast<uint32_t>(u1 * static_cast<float>(n)), n - 1u);
    const EmissiveTriangle& r = records[bucket];
    return u2 < r.alias_prob ? bucket : std::min(r.alias_index, n - 1u);
}

/// Mirror of `rtEmissiveFroxelHolds`. `list` is ASCENDING by record index, which
/// is the whole reason this can early-out.
[[nodiscard]] inline bool emissive_froxel_holds(std::span<const uint32_t> list, uint32_t record) {
    for (const uint32_t r : list) {
        if (r == record) {
            return true;
        }
        if (r > record) {
            return false;
        }
    }
    return false;
}

/// q_f — the share of a draw that actually goes to the froxel list. An empty
/// list takes none of it, which is what makes an unbinned froxel behave exactly
/// as R6 did rather than as a division by zero.
[[nodiscard]] inline float emissive_froxel_share(float share, uint32_t list_count) {
    return list_count > 0 ? share : 0.0f;
}

/**
 * The mixture source pdf of one record. Mirror of the `pdf` lane of
 * `rtEmissiveDrawMixture`.
 *
 * @param share       q_f before the empty-list clamp.
 * @param list_count  |L_f|.
 * @param in_list     whether this record is in that froxel's list.
 * @param power       the record's alias weight.
 * @param total_power the alias table's normalizing constant.
 */
[[nodiscard]] inline float emissive_mixture_pdf(float share, uint32_t list_count, bool in_list, float power,
                                                float total_power) {
    const float qf = emissive_froxel_share(share, list_count);
    const float qg = 1.0f - qf;
    const float global = qg * (power / std::max(total_power, 1e-6f));
    return global + (in_list ? qf / static_cast<float>(list_count) : 0.0f);
}

/// One mixture draw: which record, and the pdf it was drawn with.
struct EmissiveDraw {
    uint32_t index = 0;
    float pdf = 0.0f;
};

/**
 * Mirror of `rtEmissiveDrawMixture`, uniform for uniform.
 *
 * `u1` picks the mixture component AND, rescaled onto [0, 1), the entry within
 * it — one uniform doing two jobs, which is exact because `(u1 - a) / (b - a)`
 * is uniform on [0, 1) conditioned on the branch it selected. `u2` is the alias
 * table's second uniform and is unused by the froxel branch, so BOTH branches
 * consume the same two draws and the sample stream keeps the shape R6 gave it.
 */
[[nodiscard]] inline EmissiveDraw emissive_draw_mixture(std::span<const uint32_t> list,
                                                        std::span<const EmissiveTriangle> records,
                                                        float total_power, float share, float u1, float u2) {
    EmissiveDraw d;
    const auto n = static_cast<uint32_t>(records.size());
    if (n == 0) {
        return d;
    }
    const auto list_count = static_cast<uint32_t>(list.size());
    const float qf = emissive_froxel_share(share, list_count);
    const float qg = 1.0f - qf;

    bool in_list = false;
    if (qg <= 0.0f || u1 < qf) {
        const float t = std::clamp(u1 / std::max(qf, 1e-6f), 0.0f, 1.0f);
        const uint32_t slot =
            std::min(static_cast<uint32_t>(t * static_cast<float>(list_count)), list_count - 1u);
        d.index = std::min(list[slot], n - 1u);
        in_list = true;
    } else {
        const float t = std::clamp((u1 - qf) / std::max(qg, 1e-6f), 0.0f, 1.0f);
        d.index = emissive_sample_index(records, t, u2);
        in_list = qf > 0.0f && emissive_froxel_holds(list, d.index);
    }
    d.pdf = emissive_mixture_pdf(share, list_count, in_list, records[d.index].power, total_power);
    return d;
}

/// What the bin cost and refused, per frame. Reported through `RtStats` and the
/// playground's `[smoke] rt` line — the no-silent-drops rule applies to a froxel
/// list exactly as it does to the record cap above it.
struct EmissiveClusterStats {
    /// Froxels holding at least one record.
    uint32_t froxels_filled = 0;
    /// Entries packed across every froxel.
    uint32_t entries = 0;
    /// Record-froxel associations that did not fit: every insertion attempted
    /// into a froxel already at capacity, whether the newcomer was refused or
    /// displaced the weakest entry. Either way one record loses that froxel's
    /// local importance sampling — it stays reachable through the global half of
    /// the mixture, which is why this is a quality figure rather than an error.
    uint32_t entries_dropped = 0;
    /// Records whose relevance sphere reached no froxel of this frame's frustum.
    uint32_t records_culled = 0;
};

/**
 * The froxel binner for emissive records.
 *
 * Deliberately shaped like `render::lighting::ClusterGrid::build()` and for the
 * same reasons: a serial per-record range precompute (fanned out here, because
 * there are thousands of records where there are at most 256 lights), then a
 * fan-out chunked by FROXEL index in which every chunk is the sole writer of its
 * froxels' bins. Records are walked in ascending index inside a chunk, and the
 * top-k insertion replaces only on a STRICTLY greater score, so a tie keeps the
 * lower record index — bit-identical output at any worker count.
 *
 * Thread-safety: externally synchronized, one caller at a time (the Prepare
 * stage / the frame's render thread), like ClusterGrid.
 */
class EmissiveClusterGrid {
public:
    EmissiveClusterGrid() = default;
    ~EmissiveClusterGrid();

    EmissiveClusterGrid(const EmissiveClusterGrid&) = delete;
    EmissiveClusterGrid& operator=(const EmissiveClusterGrid&) = delete;
    EmissiveClusterGrid(EmissiveClusterGrid&&) = delete;
    EmissiveClusterGrid& operator=(EmissiveClusterGrid&&) = delete;

    /// Allocate the flat table's GPU buffer. Safe to skip entirely for a CPU-only
    /// caller (the tests): `build()` needs no device.
    [[nodiscard]] bool init(gpu::GPUDevice& device);
    void shutdown();

    /**
     * Bin `records` into the froxel grid described by `params` + `bounds`.
     *
     * @param bounds CLUSTER_COUNT view-space froxel AABBs —
     *   `ClusterGrid::cluster_bounds()`, borrowed rather than recomputed so the
     *   two binners cannot disagree about where a froxel is.
     *
     * An empty record set, an empty bounds table or a mismatched grid clears the
     * table and leaves `live()` false; nothing else in the frame then reads it.
     *
     * @param record_generation `RtScene::emissive_generation()`, opting this
     *   call into the REBUILD GATE: when it, the record span, and every field
     *   of `params` this table depends on are all unchanged since the last
     *   successful build, the existing table already describes this frame and
     *   the whole pass is skipped — `stats()`, `live()` and the pending upload
     *   all still stand. Pass `k_emissive_no_generation` (what the overloads
     *   without it do) to force the rebuild; see that constant for why it is
     *   not a default argument.
     *
     *   The gate cannot fire on a frame the camera moved, so it recovers
     *   nothing in the playground's own bench — the emissive orbs bob every
     *   frame there, which bumps the generation regardless. It is for a paused
     *   game, a photo mode, a cutscene hold, or a fixed camera over static
     *   emitters.
     */
    void build(core::jobs::JobSystem& js, std::span<const EmissiveTriangle> records,
               const lighting::ClusterParams& params, std::span<const lighting::ClusterBounds> bounds,
               uint32_t record_generation);
    /// Convenience overload servicing the canonical singleton pool.
    void build(std::span<const EmissiveTriangle> records, const lighting::ClusterParams& params,
               std::span<const lighting::ClusterBounds> bounds, uint32_t record_generation);
    /// Ungated forms: always rebuild. What every CPU-only caller wants.
    void build(core::jobs::JobSystem& js, std::span<const EmissiveTriangle> records,
               const lighting::ClusterParams& params, std::span<const lighting::ClusterBounds> bounds);
    void build(std::span<const EmissiveTriangle> records, const lighting::ClusterParams& params,
               std::span<const lighting::ClusterBounds> bounds);

    /// Record the table's transfer onto the frame's command buffer. Unlike
    /// ClusterGrid's this is NOT a copy-pass upload: the records it describes are
    /// refreshed after the frame's consolidated copy pass has already closed
    /// (the TLAS refit runs later), so it rides `cmd` the way RtScene's own
    /// buffers do.
    void upload(SDL_GPUCommandBuffer* cmd);

    /// The flat table, or null before `init()`.
    [[nodiscard]] SDL_GPUBuffer* buffer() const;
    /// True when the last `build()` produced a table describing THIS frame's
    /// records. False makes the trace send `emissiveFroxelShare = 0`, which is
    /// R6's draw — a correct estimator against a stale list is impossible, and
    /// falling back to the global half is exactly the right degradation.
    [[nodiscard]] bool live() const { return live_; }
    /// Declare the table stale without rebuilding it — what a frame that SKIPS
    /// the bin must call. Without it `live()` would still describe the last
    /// frame that did bin, and the trace would sample a list naming records that
    /// may since have been renumbered.
    void invalidate() {
        live_ = false;
        stats_ = {};
    }
    [[nodiscard]] const EmissiveClusterStats& stats() const { return stats_; }

    /// Test accessors — the flat table and one froxel's list, decoded.
    [[nodiscard]] std::span<const uint32_t> table() const { return table_; }
    [[nodiscard]] std::span<const uint32_t> froxel_list(uint32_t froxel) const;

private:
    /**
     * One SURVIVING record's conservative froxel-index box, in the COMPACTED
     * array the binning fan-out walks. Shaped like `ClusterGrid::LightRange`
     * because it answers the identical question about an identical sphere, and
     * packed tighter because there are thousands of these where there are at
     * most 256 lights: the binning chunks re-walk this array once each, so its
     * size is bandwidth the pass pays `chunk_count` times over. 36 bytes rather
     * than 60 measurably matters at 8 k records.
     */
    struct RecordRange {
        glm::vec3 view_center{0.0f};
        float area = 0.0f;
        float power = 0.0f;
        uint32_t record = 0;       ///< index into the caller's record span
        uint16_t min_c = 0, max_c = 0;
        uint8_t x0 = 0, x1 = 0, y0 = 0, y1 = 0, z0 = 0, z1 = 0;
    };

    gpu::GPUDevice* device_ = nullptr;
    std::unique_ptr<gpu::GPUBuffer> buffer_;
    std::size_t buffer_entries_ = 0;

    std::vector<uint32_t> table_;       // flat, froxel_count * stride
    std::vector<RecordRange> ranges_;   // scratch, one per record (pass 1)
    std::vector<uint8_t> range_active_; // scratch, parallel to ranges_
    std::vector<RecordRange> active_;   // scratch, the compacted survivors
    std::vector<uint32_t> bin_index_;   // scratch, froxel_count * capacity
    std::vector<float> bin_score_;      // scratch, froxel_count * capacity
    std::vector<uint32_t> chunk_dropped_;
    std::vector<uint32_t> chunk_filled_;
    std::vector<uint32_t> chunk_entries_;

    uint32_t froxel_count_ = 0;
    glm::mat4 last_inv_projection_{0.0f};
    glm::mat4 projection_{1.0f};
    /// The rebuild gate's memo of the inputs of the last SUCCESSFUL build —
    /// everything the table is a function of. `last_inv_projection_` above is
    /// part of it and is deliberately shared with the projection memo one line
    /// up: that memo's condition is a subset of this one's, so the two cannot
    /// drift. Cleared by shutdown(); untouched by invalidate(), which clears
    /// `live_` and thereby disarms the gate on its own.
    const EmissiveTriangle* last_records_ = nullptr;
    std::size_t last_record_count_ = 0;
    uint32_t last_generation_ = k_emissive_no_generation;
    glm::mat4 last_view_{0.0f};
    glm::vec4 last_z_planes_{0.0f};
    glm::uvec4 last_grid_dim_{0u};
    /// The four SIDE planes of the frustum in VIEW space (Gribb-Hartmann rows of
    /// the projection), normalized. The cheap pre-cull: six dot products reject
    /// a record before it costs eight matrix-vector products to bound its tiles.
    std::array<glm::vec4, 4> side_planes_{};
    bool projection_valid_ = false;
    bool live_ = false;
    bool dirty_ = false;
    bool warned_dropped_ = false;
    EmissiveClusterStats stats_{};
};

} // namespace bracken::render::rt
