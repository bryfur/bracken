#include "bracken/scene/scene_prep.hpp"

#include "bracken/animation/animation_types.hpp" // MAX_BONES
#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/core/jobs/task_group.hpp"
#include "bracken/model_loader.hpp"
#include "bracken/scene/transform_math.hpp"

#include <algorithm>
#include <cstring>

// This translation unit deliberately includes NOTHING from render/ or gpu/
// beyond the POD uniform layouts: it is the worker-safe side of the renderer's
// only thread-tier boundary (see the class comment in scene_prep.hpp). The
// MAX_SHADOW_CASCADES == render::CSM_MAX_CASCADES static_assert therefore
// lives in scene_renderer.cpp, which already owns that dependency.

namespace bracken::scene {

// ---------------------------------------------------------------------------
// Prep fan-out chunk bodies (docs/multithreading_plan.md §4.3). Each binning
// body is destination-agnostic: the single-chunk inline path passes the FINAL
// output containers and the fan-out path passes a chunk's private scratch —
// the exact same per-command logic runs either way (granularity control, not
// a second code path), which is what makes chunk-order concatenation
// bit-identical to the serial walk.
// ---------------------------------------------------------------------------

// Main-view walk over cmds[begin, end): distance cull → bounds → main frustum
// cull → scatter into dense per-handle bins / the non-instanced list, with
// cull counters in `stats` (main view owns ALL model-command stats — the
// shadow views recompute the same pure predicates but count nothing).
static void bin_main_chunk(std::span<const ModelDrawCommand> cmds, size_t begin, size_t end,
                           const ScenePrep::PrepParams& p, std::vector<std::vector<gpu::InstanceData>>& bins,
                           std::vector<ModelHandle>& touched,
                           std::vector<const ModelDrawCommand*>& non_instanced, FrameStatsPartial& stats) {
    for (size_t i = begin; i < end; ++i) {
        const ModelDrawCommand& cmd = cmds[i];
        const glm::mat4& t = cmd.transform;

        const float dx = t[3][0] - p.camera_pos.x;
        const float dz = t[3][2] - p.camera_pos.z;
        if (dx * dx + dz * dz > p.draw_dist_sq) {
            if (p.collect_stats) {
                stats.entities_distance_culled++;
            }
            continue;
        }

        const ModelHandle handle = cmd.model_handle;

        // ONE lookup, serving both the bounds test and the diagnostic. This
        // is the single point at which every model command of the frame is
        // visited exactly once, so it is where an unresolvable handle gets
        // counted: the command survives culling here and then silently draws
        // nothing at record time, which from outside is indistinguishable
        // from the model being invisible. Diagnostics are NOT gated on
        // p.collect_stats — a drop is not opt-in instrumentation.
        const Model* model = p.models->get_model(handle);
        if (model == nullptr) {
            stats.diagnostics.draws_dropped_unknown_model++;
        }

        // Main visibility only needs bounds when frustum culling is on; a
        // command whose model is missing has no bounds and stays visible
        // (matches the serial walk, where the frustum test required bounds).
        bool main_visible = true;
        if (p.do_frustum_cull && model != nullptr) {
            const glm::vec3 world_center = glm::vec3(t * glm::vec4(model->bounding_center, 1.0f));
            const float world_radius = model->bounding_half_diag * max_scale_factor(t);
            main_visible = p.main_frustum->intersects_sphere(world_center, world_radius);
        }

        if (!main_visible) {
            if (p.collect_stats) {
                stats.entities_frustum_culled++;
            }
            continue; // shadow binning for main-culled casters is the shadow views' job
        }

        if (p.collect_stats) {
            stats.entities_rendered++;
        }

        if (has(cmd.flags, DrawFlags::ForceNonInstanced)) {
            non_instanced.push_back(&cmd);
            continue;
        }

        // LOD: pick a coarser copy of this model when it is small on screen.
        // Levels are separate models, so the swap happens HERE and every stage
        // after this line — binning, upload, draw — is unchanged and unaware.
        // select_lod is pure, so this stays bit-identical at any worker count.
        ModelHandle bin_handle = handle;
        if (model != nullptr && !model->lod_handles.empty()) {
            const glm::vec3 to_camera = glm::vec3(t[3]) - p.camera_pos;
            const float distance = glm::length(to_camera);
            const float world_radius = model->bounding_half_diag * max_scale_factor(t);
            // Apparent size, not distance: the same threshold then behaves
            // identically for a pebble up close and a tower far away.
            const float projected = distance > 1e-3f ? world_radius / distance : 1e6f;
            bin_handle = select_lod(*model, handle, projected);
        }

        gpu::InstanceData instance{};
        instance.model = cmd.transform;
        instance.normalMatrix = compute_normal_matrix(cmd.transform);
        instance.tint = cmd.tint;
        instance.noFog = has(cmd.flags, DrawFlags::NoFog) ? 1.0f : 0.0f;
        // Motion vectors: previous-frame transform from the motion history
        // (per-instance motion_id); statics fall back to the current
        // transform, i.e. camera-only velocity.
        const MotionHistory::Entry* hist = p.motion_history->find_prev(cmd.motion_id);
        instance.prevModel = hist ? hist->model : cmd.transform;

        // Dense-bin push: grow the bin array to cover the handle (handles are
        // dense indices into ModelManager's slot array, so this stays small)
        // and record first touch for O(touched) clearing/merging.
        if (bin_handle >= bins.size()) {
            bins.resize(bin_handle + 1);
        }
        auto& bin = bins[bin_handle];
        if (bin.empty()) {
            touched.push_back(bin_handle);
        }
        bin.push_back(instance);
    }
}

// One cascade's walk over cmds[begin, end): recomputes the pure main-view
// predicates (distance cull, bounds, main frustum test — cheap and pure, the
// price of view independence), then bins casters that intersect THIS
// cascade's light frustum. Semantics mirror the serial walk exactly:
// - distance-culled commands cast nothing;
// - main-frustum-culled commands cast only with bounds and only when
//   instanced (ForceNonInstanced casters draw via the non-instanced path);
// - main-visible instanced commands cast when boundless or intersecting.
static void bin_shadow_chunk(std::span<const ModelDrawCommand> cmds, size_t begin, size_t end,
                             const ScenePrep::PrepParams& p, int cascade,
                             std::vector<std::vector<gpu::ShadowInstanceData>>& bins,
                             std::vector<ModelHandle>& touched) {
    const Frustum& light_frustum = p.cascade_frustums[cascade];
    for (size_t i = begin; i < end; ++i) {
        const ModelDrawCommand& cmd = cmds[i];
        const glm::mat4& t = cmd.transform;

        const float dx = t[3][0] - p.camera_pos.x;
        const float dz = t[3][2] - p.camera_pos.z;
        if (dx * dx + dz * dz > p.draw_dist_sq) {
            continue;
        }

        const ModelHandle handle = cmd.model_handle;

        glm::vec3 world_center{};
        float world_radius = 0.0f;
        bool have_bounds = false;
        if (const Model* model = p.models->get_model(handle)) {
            world_center = glm::vec3(t * glm::vec4(model->bounding_center, 1.0f));
            world_radius = model->bounding_half_diag * max_scale_factor(t);
            have_bounds = true;
        }

        const bool main_visible =
            !(p.do_frustum_cull && have_bounds && !p.main_frustum->intersects_sphere(world_center, world_radius));

        bool casts = false;
        if (!main_visible) {
            casts = !has(cmd.flags, DrawFlags::ForceNonInstanced) && have_bounds &&
                    light_frustum.intersects_sphere(world_center, world_radius);
        } else if (!has(cmd.flags, DrawFlags::ForceNonInstanced)) {
            casts = !have_bounds || light_frustum.intersects_sphere(world_center, world_radius);
        }
        if (!casts) {
            continue;
        }

        gpu::ShadowInstanceData shadow_instance{};
        shadow_instance.model = cmd.transform;
        if (handle >= bins.size()) {
            bins.resize(handle + 1);
        }
        auto& bin = bins[handle];
        if (bin.empty()) {
            touched.push_back(handle);
        }
        bin.push_back(shadow_instance);
    }
}

// Shared per-view chunk fan-out driver: run `chunk_fn(chunk, begin, end)`
// over [0, total) in PREP_COMMAND_GRAIN chunks on the pool, then
// `merge_fn(chunk)` single-threaded in chunk-index order. `inline_fn()` is
// the one-chunk path (same chunk body aimed at the final outputs). The guard
// mirrors parallel_for's exactly (initialized, >1 worker, more than one
// grain of work), so the fan-out branch is taken iff parallel_for will
// actually chunk at [k * grain, ...) boundaries — which is what makes
// `begin / grain` the chunk index.
template<typename InlineFn, typename ChunkFn, typename MergeFn>
static void prep_view_fan_out(core::jobs::JobSystem& js, size_t total, ScenePrep::PrepViewScratch& scratch,
                              InlineFn&& inline_fn, ChunkFn&& chunk_fn, MergeFn&& merge_fn) {
    constexpr size_t grain = ScenePrep::PREP_COMMAND_GRAIN;
    const bool fan_out = js.is_initialized() && js.worker_count() > 1 && total > grain;
    if (!fan_out) {
        inline_fn();
        return;
    }

    const size_t chunk_count = (total + grain - 1) / grain;
    // Grow-only: per-chunk capacity survives across frames (see
    // PrepViewScratch — the no-per-frame-allocs discipline).
    if (scratch.chunks.size() < chunk_count) {
        scratch.chunks.resize(chunk_count);
    }
    core::jobs::parallel_for(
        js, 0, total,
        [&](size_t begin, size_t end) { chunk_fn(scratch.chunks[begin / grain], begin, end); }, grain);
    // Single-threaded merge in chunk-index order (§6.2): contiguous chunks
    // concatenated in index order reproduce the serial command order.
    for (size_t k = 0; k < chunk_count; ++k) {
        merge_fn(scratch.chunks[k]);
    }
}

// Main-view job body: chunked cull+bin into instance bins + non-instanced
// list + stats, merged in chunk order.
static void prep_main_view(core::jobs::JobSystem& js, std::span<const ModelDrawCommand> cmds,
                           const ScenePrep::PrepParams& p, ScenePrep::PrepViewScratch& scratch,
                           const ScenePrep::InstanceBinOutputs& out) {
    prep_view_fan_out(
        js, cmds.size(), scratch,
        [&] {
            bin_main_chunk(cmds, 0, cmds.size(), p, out.instance_batches, out.instance_batches_touched,
                           out.non_instanced, out.stats);
        },
        [&](ScenePrep::PrepChunkScratch& ch, size_t begin, size_t end) {
            // Clear ONLY this core's scratch group, O(touched of last use).
            for (ModelHandle h : ch.touched) ch.bins[h].clear();
            ch.touched.clear();
            ch.non_instanced.clear();
            ch.stats = {};
            bin_main_chunk(cmds, begin, end, p, ch.bins, ch.touched, ch.non_instanced, ch.stats);
        },
        [&](ScenePrep::PrepChunkScratch& ch) {
            for (ModelHandle h : ch.touched) {
                if (h >= out.instance_batches.size()) {
                    out.instance_batches.resize(h + 1);
                }
                auto& bin = out.instance_batches[h];
                if (bin.empty()) {
                    out.instance_batches_touched.push_back(h);
                }
                bin.insert(bin.end(), ch.bins[h].begin(), ch.bins[h].end());
            }
            out.non_instanced.insert(out.non_instanced.end(), ch.non_instanced.begin(), ch.non_instanced.end());
            out.stats += ch.stats;
        });
}

// One cascade view's job body: chunked light-frustum binning into that
// cascade's bins, merged in chunk order. No stats (main view owns them).
static void prep_shadow_view(core::jobs::JobSystem& js, std::span<const ModelDrawCommand> cmds,
                             const ScenePrep::PrepParams& p, int cascade,
                             ScenePrep::PrepViewScratch& scratch,
                             std::vector<std::vector<gpu::ShadowInstanceData>>& bins,
                             std::vector<ModelHandle>& touched) {
    prep_view_fan_out(
        js, cmds.size(), scratch, [&] { bin_shadow_chunk(cmds, 0, cmds.size(), p, cascade, bins, touched); },
        [&](ScenePrep::PrepChunkScratch& ch, size_t begin, size_t end) {
            for (ModelHandle h : ch.shadow_touched) ch.shadow_bins[h].clear();
            ch.shadow_touched.clear();
            bin_shadow_chunk(cmds, begin, end, p, cascade, ch.shadow_bins, ch.shadow_touched);
        },
        [&](ScenePrep::PrepChunkScratch& ch) {
            for (ModelHandle h : ch.shadow_touched) {
                if (h >= bins.size()) {
                    bins.resize(h + 1);
                }
                auto& bin = bins[h];
                if (bin.empty()) {
                    touched.push_back(h);
                }
                bin.insert(bin.end(), ch.shadow_bins[h].begin(), ch.shadow_bins[h].end());
            }
        });
}

void ScenePrep::prep_instance_batches(core::jobs::JobSystem& js, std::span<const ModelDrawCommand> cmds,
                                     const PrepParams& p, PrepScratch& scratch, const InstanceBinOutputs& out) {
    BRACKEN_ASSERT(p.active_cascades <= MAX_SHADOW_CASCADES, "active_cascades {} exceeds MAX_SHADOW_CASCADES",
                   p.active_cascades);
    // Clear only the bins touched last frame (capacity preserved); empty bins
    // for never/rarely used models cost nothing to skip at iteration time.
    for (ModelHandle h : out.instance_batches_touched) out.instance_batches[h].clear();
    out.instance_batches_touched.clear();
    for (int c = 0; c < MAX_SHADOW_CASCADES; ++c) {
        for (ModelHandle h : out.shadow_batches_touched[c]) out.shadow_instance_batches[c][h].clear();
        out.shadow_batches_touched[c].clear();
    }
    out.non_instanced.clear();
    if (cmds.empty()) {
        return;
    }

    // View fan-out: the views write DISJOINT outputs (main bins + stats vs
    // each cascade's shadow bins), so each is an independent job. Small
    // scenes (below one chunk of commands — each view job would walk only a
    // handful of items) and 1-worker pools run the views sequentially inline
    // instead; the view bodies apply the chunk grain guard themselves, so
    // this is granularity control, not a separate code path.
    const bool fan_out_views = js.is_initialized() && js.worker_count() > 1 && p.active_cascades > 0 &&
                               cmds.size() > PREP_COMMAND_GRAIN;
    if (!fan_out_views) {
        prep_main_view(js, cmds, p, scratch.main, out);
        for (int c = 0; c < p.active_cascades; ++c) {
            prep_shadow_view(js, cmds, p, c, scratch.cascades[c], out.shadow_instance_batches[c],
                             out.shadow_batches_touched[c]);
        }
        return;
    }

    core::jobs::TaskGroup views;
    try {
        for (int c = 0; c < p.active_cascades; ++c) {
            js.submit(views, [&js, cmds, &p, &scratch, &out, c] {
                prep_shadow_view(js, cmds, p, c, scratch.cascades[c], out.shadow_instance_batches[c],
                                 out.shadow_batches_touched[c]);
            });
        }
        // Main view runs on the caller, which then helps execute the cascade
        // jobs (and their chunks) inside the join.
        prep_main_view(js, cmds, p, scratch.main, out);
    } catch (...) {
        // Join the cascade jobs before unwinding — they reference this
        // frame's stack state (mirrors parallel_for's exception contract).
        try {
            js.wait(views);
        } catch (...) { // NOLINT(bugprone-empty-catch)
        }
        throw;
    }
    js.wait(views);
}

// Skinned-resolve walk over cmds[begin, end): the pure per-command work —
// model lookup, joint clamp, world bounds, main-view cull, motion-history
// pairing — appended to `out` in command order. Destination-agnostic like the
// binning chunk bodies: final output vector inline, chunk scratch in the
// fan-out.
static void resolve_skinned_chunk(std::span<const SkinnedDrawCommand> cmds, size_t begin, size_t end,
                                  std::span<const BonePalette> palettes, const ScenePrep::PrepParams& p,
                                  std::vector<ScenePrep::ResolvedSkinnedEntry>& out, FrameStatsPartial& stats) {
    for (size_t i = begin; i < end; ++i) {
        const SkinnedDrawCommand& cmd = cmds[i];
        // Sealed commands carry dense palette indices (invalid ids were
        // dropped with a warning at emit time) — an out-of-range id here is
        // a builder/packet bug, not host error.
        BRACKEN_ASSERT(cmd.palette.valid() && cmd.palette.value < palettes.size(),
                       "SkinnedDrawCommand palette id {} out of range ({} palettes)", cmd.palette.value,
                       palettes.size());
        if (!cmd.palette.valid() || cmd.palette.value >= palettes.size()) {
            continue;
        }
        const BonePalette& rec = palettes[cmd.palette.value];
        Model* model = p.models->get_model(cmd.model_handle);
        uint32_t joint_count = 0;
        if (model && model->has_skeleton) {
            // Drawn joints clamp to min(model joints, palette joints) — same
            // clamp as the old span path, now against the packet palette.
            joint_count = std::min({static_cast<uint32_t>(model->skeleton.joints.size()), rec.joint_count,
                                    static_cast<uint32_t>(bracken::animation::MAX_BONES)});
        }
        if (joint_count == 0) {
            // No model, no skeleton, or a zero-joint palette: not drawable as
            // skinned and not drawn at all. Counted here (the one place every
            // skinned command is visited exactly once) rather than dropped in
            // silence.
            stats.diagnostics.skinned_dropped_no_skeleton++;
            continue;
        }

        // Resolve world bounds once here; the main view and every shadow
        // cascade reuse them instead of re-fetching the model per pass.
        const glm::mat4& t = cmd.transform;
        const glm::vec3 world_center = glm::vec3(t * glm::vec4(model->bounding_center, 1.0f));
        const float world_radius = model->bounding_half_diag * max_scale_factor(t);

        // Main-view cull once, shared by the depth prepass and the forward
        // pass (shadow cascades cull independently against light frustums).
        // Cull stats for skinned entities are counted here, nowhere else.
        bool main_visible = true;
        const float dx = t[3][0] - p.camera_pos.x;
        const float dz = t[3][2] - p.camera_pos.z;
        if (dx * dx + dz * dz > p.draw_dist_sq) {
            main_visible = false;
            if (p.collect_stats) {
                stats.entities_distance_culled++;
            }
        } else if (p.do_frustum_cull && !p.main_frustum->intersects_sphere(world_center, world_radius)) {
            main_visible = false;
            if (p.collect_stats) {
                stats.entities_frustum_culled++;
            }
        } else if (p.collect_stats) {
            stats.entities_rendered++;
        }

        // Motion vectors: previous-frame transform + palette offset. The
        // palette offset indexes the PREVIOUS palette buffer (double-buffered)
        // and is only usable when that buffer actually holds last frame's data.
        const MotionHistory::Entry* hist = p.motion_history->find_prev(cmd.motion_id);
        const glm::mat4 prev_model = hist ? hist->model : t;
        const uint32_t prev_palette =
            (hist && hist->palette_offset != MotionHistory::NO_PALETTE && p.prev_palette_valid)
                ? hist->palette_offset
                : gpu::PrepassModelUniforms::NO_PREV_PALETTE;

        // Palette GPU offset: the record's prefix-summed matrix_offset —
        // sharing (many draws, one palette) already happened at build time
        // via a shared BonePaletteId, replacing the old pointer-dedupe.
        out.push_back(
            {&cmd, model, world_center, world_radius, rec.matrix_offset, main_visible, prev_model, prev_palette});
    }
}

void ScenePrep::prep_resolve_skinned(core::jobs::JobSystem& js, std::span<const SkinnedDrawCommand> cmds,
                                         std::span<const BonePalette> palettes, const PrepParams& p,
                                         PrepViewScratch& scratch, std::vector<ResolvedSkinnedEntry>& out,
                                         FrameStatsPartial& stats) {
    out.clear();
    if (cmds.empty()) {
        return;
    }
    out.reserve(cmds.size());
    prep_view_fan_out(
        js, cmds.size(), scratch, [&] { resolve_skinned_chunk(cmds, 0, cmds.size(), palettes, p, out, stats); },
        [&](PrepChunkScratch& ch, size_t begin, size_t end) {
            ch.skinned.clear();
            ch.stats = {};
            resolve_skinned_chunk(cmds, begin, end, palettes, p, ch.skinned, ch.stats);
        },
        [&](PrepChunkScratch& ch) {
            out.insert(out.end(), ch.skinned.begin(), ch.skinned.end());
            stats += ch.stats;
        });
}

void ScenePrep::prep_copy_palettes(core::jobs::JobSystem& js, std::span<const BonePalette> palettes,
                                       glm::mat4* dst) {
    // Every chunk writes only [matrix_offset, matrix_offset + joint_count)
    // ranges of the palettes it owns — disjoint by the seal-time prefix sum,
    // so there are no partials to merge and the result is bit-identical at
    // any worker count (the bytes and their destinations are fixed; only the
    // copying thread varies). Explicit grain: palette items are heavy (up to
    // an 8 KB memcpy each); <= one grain of palettes stays inline.
    core::jobs::parallel_for(
        js, 0, palettes.size(),
        [palettes, dst](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                const BonePalette& rec = palettes[i];
                std::memcpy(dst + rec.matrix_offset, rec.matrices,
                            static_cast<size_t>(rec.joint_count) * sizeof(glm::mat4));
            }
        },
        PREP_PALETTE_GRAIN);
}

} // namespace bracken::scene
