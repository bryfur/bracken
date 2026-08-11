#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace bracken {

/**
 * One pass's per-frame cost.
 *
 * `name` is a view of a string LITERAL owned by the BRACKEN_GPU_TIMER call
 * site (GPUTimestampPool interns the pointer, it never copies), so the view
 * outlives every frame and costs no allocation.
 *
 * `cpu_ms` is the wall time spent RECORDING the pass on the CPU — building
 * and submitting its commands. It is NOT how long the GPU took to execute
 * them; a pass can record in 40 us and cost 4 ms on the device. Present it
 * labelled as "CPU (record)".
 *
 * `gpu_ms` is nullopt whenever GPUTimestampPool::is_supported() is false,
 * which today is always: SDL3 exposes no timestamp query API. An optional
 * says "unknown" where the old `float` said "0.0 ms", which is the difference
 * between an honest number and a lie that looks like a fast pass.
 */
struct PassTiming {
    std::string_view name;
    std::optional<float> cpu_ms;
    std::optional<float> gpu_ms;
};

/**
 * Per-frame render counters. Populated only while
 * SceneRenderer::set_collect_stats(true) — Application::render_stats() returns
 * nullopt otherwise, so an all-zero struct can no longer be mistaken for a
 * genuinely empty frame.
 *
 * The `*_used` / `*_budget` pairs exist because these are the caps that
 * actually get hit in practice and every one of them used to fail silently:
 * a frame that overruns MAX_BONE_PALETTES or fills the packet arena produced
 * no number a host could look at. They are cheap frame-level samples, not
 * per-item counters.
 */
struct RenderStats {
    uint32_t draw_calls = 0;
    uint32_t triangle_count = 0;
    uint32_t entities_rendered = 0;
    uint32_t entities_distance_culled = 0;
    uint32_t entities_frustum_culled = 0;
    uint32_t jobs_pending = 0;

    /// Instances actually written into the main-view instance buffer.
    uint32_t instances_drawn = 0;

    /// Packet arena occupancy of the frame that was rendered.
    std::size_t packet_bytes_used = 0;
    std::size_t packet_bytes_capacity = 0;

    /// Bone palettes the frame allocated, against FramePacket::MAX_BONE_PALETTES.
    uint32_t bone_palettes_used = 0;
    uint32_t bone_palettes_budget = 0;

    /// Lights the host submitted vs. lights the cluster grid accepted.
    uint32_t lights_submitted = 0;
    uint32_t lights_binned = 0;

    /// Local (point/spot) shadow atlas tiles re-rendered this frame.
    uint32_t local_shadow_tiles_rendered = 0;

    /// UI batch size after text expansion.
    uint32_t ui_vertices = 0;
    uint32_t ui_draws = 0;

    std::vector<PassTiming> pass_times_ms;
};

/**
 * Everything the frame threw away, and why.
 *
 * Every counter here used to be a bare `return;` or `continue;`. The day-1
 * failure — "I described a frame and nothing appears" — had no answer inside
 * the engine: an unregistered ModelHandle, a skinned draw against a model
 * with no skeleton, a palette allocation over budget and a 257th light all
 * looked identical from outside, namely like nothing happening.
 *
 * Unlike RenderStats these are accumulated UNCONDITIONALLY (they do not
 * depend on set_collect_stats): a drop is never something a host opted out of
 * knowing about, and the counters cost one increment on a path that is
 * already failing. Read them via Application::frame_diagnostics().
 */
struct FrameDiagnostics {
    /// Model draw whose ModelHandle resolves to no model (never registered,
    /// or stale across an unload_all()).
    uint32_t draws_dropped_unknown_model = 0;
    /// Skinned draw whose model has no skeleton, or whose palette carried
    /// zero joints — not drawable as skinned, and not drawn at all.
    uint32_t skinned_dropped_no_skeleton = 0;
    /// alloc_bone_palette() calls refused because the frame's
    /// MAX_BONE_PALETTES budget was exhausted (each one drops a draw).
    uint32_t palettes_over_budget = 0;
    /// add_shadow_capsules() capsules refused because the frame's
    /// MAX_SHADOW_CAPSULES budget was exhausted — analytic occluders the
    /// traced visibility passes therefore did not test against.
    uint32_t shadow_capsules_over_budget = 0;
    /// Effect emitters naming a model the ModelManager does not have.
    uint32_t effect_models_unresolved = 0;
    /// Lights the cluster grid refused because MAX_LIGHTS was reached.
    uint32_t lights_dropped = 0;
    /// Light-to-cluster insertions refused by MAX_LIGHTS_PER_CLUSTER — the
    /// lighting a cluster silently does not receive. (Counted as insertions
    /// rather than as distinct clusters: an exact distinct-cluster count
    /// needs per-cluster scratch inside the binning hot loop, and the number
    /// a host can act on is how much light data was lost.)
    uint32_t cluster_lights_dropped = 0;
    /// Host render hooks (`scene/render_hook.hpp`) whose slot could not run
    /// this frame. The four HDR-path slots have no targets to hand a callback
    /// on the fallback direct-to-swapchain path, so their hooks are skipped —
    /// counted rather than silently dropped, because "my pass never ran" is
    /// otherwise indistinguishable from "my pass drew nothing".
    uint32_t render_hooks_skipped = 0;
    /// The frame had 3D content but no camera had ever been set, so the whole
    /// 3D pass was skipped.
    bool camera_missing = false;

    /// Fold another accumulator in (chunk-index order at fan-out joins).
    FrameDiagnostics& operator+=(const FrameDiagnostics& other) {
        draws_dropped_unknown_model += other.draws_dropped_unknown_model;
        skinned_dropped_no_skeleton += other.skinned_dropped_no_skeleton;
        palettes_over_budget += other.palettes_over_budget;
        shadow_capsules_over_budget += other.shadow_capsules_over_budget;
        effect_models_unresolved += other.effect_models_unresolved;
        lights_dropped += other.lights_dropped;
        cluster_lights_dropped += other.cluster_lights_dropped;
        render_hooks_skipped += other.render_hooks_skipped;
        camera_missing = camera_missing || other.camera_missing;
        return *this;
    }

    /// True when the frame silently lost something a host would want to know
    /// about — the one-line "should I look at the numbers?" predicate.
    [[nodiscard]] bool any() const {
        return draws_dropped_unknown_model != 0 || skinned_dropped_no_skeleton != 0 || palettes_over_budget != 0 ||
               shadow_capsules_over_budget != 0 || effect_models_unresolved != 0 || lights_dropped != 0 ||
               cluster_lights_dropped != 0 || render_hooks_skipped != 0 || camera_missing;
    }
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
 * `diagnostics` rides the exact same merge — that is why the drop counters
 * need no atomics and are bit-identical at any worker count.
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
    FrameDiagnostics diagnostics;

    /// Fold another chunk's partial into this one (chunk-order at joins).
    FrameStatsPartial& operator+=(const FrameStatsPartial& other) {
        draw_calls += other.draw_calls;
        triangle_count += other.triangle_count;
        entities_rendered += other.entities_rendered;
        entities_distance_culled += other.entities_distance_culled;
        entities_frustum_culled += other.entities_frustum_culled;
        diagnostics += other.diagnostics;
        return *this;
    }

    /// Merge this section's counters into the frame stats + diagnostics —
    /// call once per section, at the section's single defined merge point.
    void merge_into(RenderStats& stats, FrameDiagnostics& diag) const {
        stats.draw_calls += draw_calls;
        stats.triangle_count += triangle_count;
        stats.entities_rendered += entities_rendered;
        stats.entities_distance_culled += entities_distance_culled;
        stats.entities_frustum_culled += entities_frustum_culled;
        diag += diagnostics;
    }
};

} // namespace bracken
