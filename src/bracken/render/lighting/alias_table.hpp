#pragma once

// Vose's alias method — O(1) sampling from a discrete distribution, built on
// the CPU once per emissive-light rebuild and consumed by
// `shaders/rt_local_shadow.comp.hlsl` (R6, docs/rt_lighting_plan.md §5.7).
//
// WHY AN ALIAS TABLE AND NOT A CDF. The emissive-light candidate stage draws a
// handful of samples per pixel from a distribution of up to
// `k_max_emissive_records` entries. A CDF costs a binary search — log2(4096) =
// 12 dependent loads per draw, in a shader whose whole cost model is "do not
// add dependent memory traffic to a BVH walk". The alias table answers in TWO
// loads and no branchy loop, at the price of one extra float and one extra uint
// per entry, which ride in the emissive record itself (see
// `render/raytracing/emissive_lights.hpp` — there are exactly as many alias
// buckets as records, so the table needs no buffer of its own and the pass
// spends ONE storage-buffer slot instead of two; SDL3 GPU's read-only ceiling
// was already spent to the last slot before this phase).
//
// GPU-free on purpose: no `gpu/`, no SDL, nothing but `<span>`/`<vector>` —
// same rule as cluster_index.hpp, reservoir.hpp and light_target.hpp, and for
// the same reason: the sampling identity this file claims is a MATHEMATICAL
// one, invisible in a screenshot, and therefore has to be pinned by a test that
// needs no device.
//
// DETERMINISM. Construction is bit-identical run to run for the same weight
// sequence: the two worklists are index-ordered stacks, never a container whose
// iteration order depends on hashing or allocation.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace bracken::render::lighting {

/**
 * One bucket of the table. `prob` is the probability of KEEPING the bucket's
 * own index when it is landed on; `alias` is where the remaining mass goes.
 */
struct AliasEntry {
    float prob = 1.0f;
    uint32_t alias = 0;
};

/**
 * Build the alias table for `weights`.
 *
 * `out` is resized to `weights.size()`. Non-finite and negative weights are
 * treated as zero — a NaN radiance from a broken material must not be able to
 * poison every draw the table serves.
 *
 * @return the sum of the (sanitized) weights, i.e. the normalizing constant the
 *   shader needs to turn a per-entry weight back into a pdf. Zero means the
 *   distribution is degenerate; the table is still filled (uniformly), because
 *   an unsampleable table and an out-of-range index are different failure modes
 *   and only one of them is safe.
 */
inline float build_alias_table(std::span<const float> weights, std::vector<AliasEntry>& out) {
    const std::size_t n = weights.size();
    out.assign(n, AliasEntry{});
    if (n == 0) {
        return 0.0f;
    }

    double total = 0.0;
    std::vector<double> scaled(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const float w = weights[i];
        const double v = (std::isfinite(w) && w > 0.0f) ? static_cast<double>(w) : 0.0;
        scaled[i] = v;
        total += v;
    }

    for (std::size_t i = 0; i < n; ++i) {
        out[i].prob = 1.0f;
        out[i].alias = static_cast<uint32_t>(i);
    }
    if (!(total > 0.0)) {
        return 0.0f; // uniform identity table; every bucket keeps itself
    }

    // Scale so the mean bucket is 1.0. Vose then pairs an under-full bucket with
    // an over-full one until none are left.
    const double norm = static_cast<double>(n) / total;
    for (std::size_t i = 0; i < n; ++i) {
        scaled[i] *= norm;
    }

    std::vector<uint32_t> small;
    std::vector<uint32_t> large;
    small.reserve(n);
    large.reserve(n);
    // Descending push so the stacks POP in ascending index order — the cheapest
    // way to make the pairing a pure function of the weight sequence.
    for (std::size_t i = n; i-- > 0;) {
        (scaled[i] < 1.0 ? small : large).push_back(static_cast<uint32_t>(i));
    }

    while (!small.empty() && !large.empty()) {
        const uint32_t s = small.back();
        small.pop_back();
        const uint32_t l = large.back();
        large.pop_back();

        out[s].prob = static_cast<float>(scaled[s]);
        out[s].alias = l;

        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
        (scaled[l] < 1.0 ? small : large).push_back(l);
    }
    // Whatever is left is 1.0 up to float error; forcing it exactly keeps the
    // "prob == 1 means always keep" reading true.
    for (uint32_t i : large) {
        out[i].prob = 1.0f;
        out[i].alias = i;
    }
    for (uint32_t i : small) {
        out[i].prob = 1.0f;
        out[i].alias = i;
    }

    return static_cast<float>(total);
}

/**
 * Draw one index. `u1`, `u2` uniform in [0, 1) — the exact two-load, no-loop
 * form `rt_local_shadow.comp.hlsl` transliterates.
 */
[[nodiscard]] inline uint32_t sample_alias(std::span<const AliasEntry> table, float u1, float u2) {
    if (table.empty()) {
        return 0;
    }
    const auto n = static_cast<uint32_t>(table.size());
    uint32_t bucket = static_cast<uint32_t>(u1 * static_cast<float>(n));
    if (bucket >= n) {
        bucket = n - 1;
    }
    return u2 < table[bucket].prob ? bucket : table[bucket].alias;
}

} // namespace bracken::render::lighting
