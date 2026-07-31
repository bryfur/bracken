#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace bracken {

struct RenderStats {
    uint32_t draw_calls = 0;
    uint32_t triangle_count = 0;
    uint32_t entities_rendered = 0;
    uint32_t entities_distance_culled = 0;
    uint32_t entities_frustum_culled = 0;
    uint32_t jobs_pending = 0;
    std::vector<std::pair<std::string, float>> pass_times_ms;
};

/**
 * Per-section (and, from Phase 3 of the multithreading plan, per-chunk)
 * accumulator for the counters that culling, batch building, and the draw
 * walks bump per item.
 *
 * Merge discipline: hot loops increment a private FrameStatsPartial — a plain
 * struct, deliberately no atomics (an atomic counter increment inside a
 * per-item loop is a review defect; see docs/multithreading_plan.md §7). Each
 * section merges its partial into the frame's RenderStats exactly once, at a
 * single defined point after its loop(s) finish. When a section is split into
 * parallel chunks, each chunk owns its own partial and the partials are
 * merged single-threaded at the join, in chunk-index order — deterministic at
 * any worker count. (Integer sums commute, so ordering is about discipline
 * and future float fields, not today's values.) If per-chunk partials are
 * ever stored in a shared array written concurrently, pad the slots to the
 * destructive interference size; a stack-local partial per chunk needs no
 * padding.
 *
 * `jobs_pending` and `pass_times_ms` are frame-level values sampled at
 * defined single-threaded points, so they are not part of the partial.
 */
struct FrameStatsPartial {
    uint32_t draw_calls = 0;
    uint32_t triangle_count = 0;
    uint32_t entities_rendered = 0;
    uint32_t entities_distance_culled = 0;
    uint32_t entities_frustum_culled = 0;

    /// Fold another chunk's partial into this one (chunk-order at joins).
    FrameStatsPartial& operator+=(const FrameStatsPartial& other) {
        draw_calls += other.draw_calls;
        triangle_count += other.triangle_count;
        entities_rendered += other.entities_rendered;
        entities_distance_culled += other.entities_distance_culled;
        entities_frustum_culled += other.entities_frustum_culled;
        return *this;
    }

    /// Merge this section's counters into the frame stats — call once per
    /// section, at the section's single defined merge point.
    void merge_into(RenderStats& stats) const {
        stats.draw_calls += draw_calls;
        stats.triangle_count += triangle_count;
        stats.entities_rendered += entities_rendered;
        stats.entities_distance_culled += entities_distance_culled;
        stats.entities_frustum_culled += entities_frustum_culled;
    }
};

} // namespace bracken
