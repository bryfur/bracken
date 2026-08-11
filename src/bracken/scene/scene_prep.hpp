#pragma once

#include "bracken/contracts/frustum.hpp"
#include "bracken/contracts/occlusion.hpp"
#include "bracken/contracts/render_stats.hpp"
#include "bracken/core/jobs/cache_line.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/scene/frame_packet.hpp"
#include "bracken/scene/motion_history.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <vector>

namespace bracken::core::jobs {
class JobSystem;
}

namespace bracken {
class ModelManager;
struct Model;
} // namespace bracken

namespace bracken::scene {

namespace gpu = bracken::gpu;

/**
 * The renderer's CPU Prepare stage: cull, bin, resolve and copy, over a
 * SEALED FramePacket (docs/multithreading_plan.md §4.3).
 *
 * ==================== THE TIER BOUNDARY IS THE POINT ====================
 *
 * This is the only place inside the renderer where work leaves the main
 * thread. Everything here runs on pool workers, and the contract that makes
 * that safe used to be a comment plus a hand-checked PrepParams field list.
 * It is now a type: `ScenePrep` has no GPUDevice, no RenderContext, no
 * sub-renderer pointer and no state at all, so a prep core structurally
 * CANNOT reach a main-thread-tier API. The inputs it does take are exactly
 * the ones that are worker-safe on a sealed frame — const ModelManager
 * lookups, const MotionHistory lookups, spans over packet memory, and
 * by-value frame state.
 *
 * HARD RULE: ScenePrep never gains a RenderContext*, a GPUDevice* or a
 * sub-renderer pointer "for convenience". Doing so silently re-opens the tier
 * and Phase 4's BRACKEN_ASSERT_NOT_SIM_STAGE would start firing from a
 * worker. Whatever needs a device is main-thread work and belongs in
 * scene_renderer_upload.cpp, on the other side of the boundary.
 *
 * DETERMINISM CONTRACT (§6.2): every core produces BIT-IDENTICAL outputs at
 * any worker count. Chunks accumulate into their own per-chunk scratch (no
 * atomics, no shared cache-line writes inside the per-item loops) and
 * partials merge single-threaded in chunk-index order after the join —
 * concatenating contiguous chunks in index order reproduces the serial
 * command order exactly. Below one chunk the same binning code runs inline on
 * the caller, writing the final outputs directly: granularity control, not a
 * second code path. scene/scene_prep_parity.test.cpp is the acceptance test
 * (0/1/4 workers, memcmp on every output).
 *
 * Stateless by construction — all entry points are static and the type is
 * not instantiable.
 */
class ScenePrep {
public:
    ScenePrep() = delete;

    // Mirror of render::CSM_MAX_CASCADES (render/shadow_map.hpp), duplicated
    // because neither this header nor scene_renderer.hpp may include a
    // render/ header; a static_assert in scene_prep.cpp keeps them in
    // lockstep.
    static constexpr int MAX_SHADOW_CASCADES = 4;

    // Chunk grains. Command-walk chunks target ~128 commands so per-chunk
    // dispatch overhead stays well under the cull/bin work; totals <= grain
    // run as one inline chunk, which keeps the 1-character playground
    // effectively serial. Palette-copy items are much heavier per item (up to
    // an 8 KB memcpy each), so they chunk at a smaller grain.
    static constexpr std::size_t PREP_COMMAND_GRAIN = 128;
    static constexpr std::size_t PREP_PALETTE_GRAIN = 32;

    // Skinned commands resolved once per frame by prep_resolve_skinned():
    // Model*, world bounds, palette offset, main-view visibility, and the
    // previous-frame motion state (model matrix + palette offset from
    // MotionHistory), shared by the prepass, main view and every shadow
    // cascade. Commands without a model/skeleton/palette joints are
    // excluded. Cleared per frame, capacity preserved. `cmd` points into the
    // consumed packet — valid only within render_frame (in tests: while the
    // packet stays sealed).
    struct ResolvedSkinnedEntry {
        const SkinnedDrawCommand* cmd;
        Model* model;
        glm::vec3 world_center;
        float world_radius;
        uint32_t palette_offset;
        bool main_visible;            // passed the main-view distance + frustum cull
        glm::mat4 prev_model;         // last frame's transform (== current if no history)
        uint32_t prev_palette_offset; // into the prev palette buffer; NO_PREV_PALETTE if none
    };

    /** Read-only inputs of the prep cores. Everything reachable through it is
     *  worker-safe during the prep fan-out: const ModelManager lookups (no
     *  mutation in flight — the hot-reload fence guarantees quiescence),
     *  const MotionHistory lookups, and by-value/pointer frame state. */
    struct PrepParams {
        const ModelManager* models = nullptr;          // get_model — const, worker-safe after load
        const MotionHistory* motion_history = nullptr; // find_prev — const lookups only
        glm::vec3 camera_pos{0.0f};
        const Frustum* main_frustum = nullptr;     // required when do_frustum_cull
        const Frustum* cascade_frustums = nullptr; // [active_cascades] light frustums
        int active_cascades = 0;                   // 0 = no shadow binning
        float draw_dist_sq = 0.0f;
        bool do_frustum_cull = false;
        bool collect_stats = false;
        bool prev_palette_valid = false; // prev palette buffer holds last frame's data
        /**
         * Frame-old coarse depth for the Hi-Z occlusion test, or null when
         * GraphicsSettings::occlusion_culling is off (the default).
         *
         * DOES NOT WEAKEN THE BIT-IDENTITY CONTRACT. It is a frame-constant,
         * read-only input — exactly like `main_frustum` — so the per-command
         * predicate stays a pure function of (command, params) and the chunked
         * walk still reproduces the serial one at any worker count. What would
         * break the contract is the obvious alternative, a per-instance "was
         * visible last frame" latch: chunk order would decide who writes it.
         * There is deliberately no such latch.
         *
         * With it null the hot path costs one null compare per surviving
         * command and every output is bit-identical to before the feature
         * existed.
         */
        const OcclusionField* occlusion = nullptr;
    };

    /** One chunk's private partial for the prep fan-outs. Each field group is
     *  used by exactly one core (main-view binning / shadow-view binning /
     *  skinned resolve); a chunk job clears ONLY the group it is about to
     *  fill (O(touched), capacity preserved) at the top of the chunk, so
     *  steady-state frames allocate nothing. Padded to the cache line:
     *  adjacent chunk slots in the pool array are written by different
     *  workers concurrently (render_stats.hpp merge discipline). */
    struct alignas(core::jobs::k_cache_line) PrepChunkScratch {
        // Main-view binning partial (dense by ModelHandle + first-touch list).
        std::vector<std::vector<gpu::InstanceData>> bins;
        std::vector<ModelHandle> touched;
        std::vector<const ModelDrawCommand*> non_instanced;
        // Shadow-view binning partial (each cascade view has its own pool).
        std::vector<std::vector<gpu::ShadowInstanceData>> shadow_bins;
        std::vector<ModelHandle> shadow_touched;
        // Skinned-resolve partial.
        std::vector<ResolvedSkinnedEntry> skinned;
        // Cull/render counters of this chunk, merged in chunk-index order.
        FrameStatsPartial stats;
    };

    /** One view's chunk-scratch pool: grown to the frame's chunk count on
     *  demand and never shrunk — per-chunk capacity survives across frames
     *  (the no-per-frame-allocs-at-steady-state discipline). Only chunk slots
     *  [0, chunk_count) of a given frame are written and merged; higher slots
     *  keep stale data and are never read. */
    struct PrepViewScratch {
        std::vector<PrepChunkScratch> chunks;
    };

    /** Per-view scratch pools for the Prepare fan-out. Views run as
     *  concurrent sibling jobs, so each has its own pool: main view,
     *  each CSM cascade, and the skinned resolve. Owned by the CALLER
     *  (the renderer holds one as a member, tests hold locals) so
     *  lifetime/reset stays testable — ScenePrep itself owns nothing. */
    struct PrepScratch {
        PrepViewScratch main;
        std::array<PrepViewScratch, MAX_SHADOW_CASCADES> cascades;
        PrepViewScratch skinned;
    };

    /** Output bindings of prep_instance_batches — references onto the
     *  caller's containers (the renderer's members, or a test's locals).
     *  All previous contents are cleared O(touched) by the call. */
    struct InstanceBinOutputs {
        std::vector<std::vector<gpu::InstanceData>>& instance_batches;
        std::vector<ModelHandle>& instance_batches_touched;
        std::array<std::vector<std::vector<gpu::ShadowInstanceData>>, MAX_SHADOW_CASCADES>& shadow_instance_batches;
        std::array<std::vector<ModelHandle>, MAX_SHADOW_CASCADES>& shadow_batches_touched;
        std::vector<const ModelDrawCommand*>& non_instanced;
        FrameStatsPartial& stats;
    };

    /**
     * Cull + bin the packet's model commands for the main view and every
     * active CSM cascade. Fan-out shape (§4.3): one TaskGroup job per view —
     * the outputs are disjoint (instance_batches vs shadow_instance_batches[c])
     * so views never share a write target; the main view runs on the caller,
     * which then helps execute the cascade jobs inside the join. Within each
     * view the command span is chunked at PREP_COMMAND_GRAIN with per-chunk
     * partial bins merged in chunk order. Bin contents/order, touched lists,
     * the non-instanced list and `out.stats` are bit-identical to the serial
     * walk at any worker count.
     */
    static void prep_instance_batches(core::jobs::JobSystem& js, std::span<const ModelDrawCommand> cmds,
                                      const PrepParams& p, PrepScratch& scratch, const InstanceBinOutputs& out);

    /**
     * Resolve the packet's skinned commands: per-command model lookup, joint
     * clamp, world bounds, main-view cull and motion-history pairing — all
     * pure reads that chunk at PREP_COMMAND_GRAIN with per-chunk partials
     * appended into `out` in chunk-index order (byte-identical to the serial
     * walk). Order-dependent palette work stays out of the fan-out entirely:
     * palette offsets were prefix-summed serially at packet seal, and the
     * old renderer-side dedupe table is gone (palette sharing happens at
     * build time via shared BonePaletteIds).
     */
    static void prep_resolve_skinned(core::jobs::JobSystem& js, std::span<const SkinnedDrawCommand> cmds,
                                     std::span<const BonePalette> palettes, const PrepParams& p,
                                     PrepViewScratch& scratch, std::vector<ResolvedSkinnedEntry>& out,
                                     FrameStatsPartial& stats);

    /**
     * Copy every packet palette into `dst` (the mapped GPU transfer buffer in
     * render_frame; any caller-owned buffer of palette_matrix_total() matrices
     * in tests). Each palette's destination range
     * [matrix_offset, matrix_offset + joint_count) is disjoint by construction
     * (the seal-time prefix sum — offset assignment itself stays serial), so
     * chunks write disjoint ranges: no partials, no merge, bit-identical at
     * any worker count.
     *
     * `dst` is a plain pointer, deliberately: on the renderer's side it
     * happens to be a mapped transfer buffer, but ScenePrep is not allowed to
     * know that — mapping and uploading are main-thread work and live in
     * scene_renderer_upload.cpp.
     */
    static void prep_copy_palettes(core::jobs::JobSystem& js, std::span<const BonePalette> palettes, glm::mat4* dst);
};

} // namespace bracken::scene
