#pragma once

// The optional HIGH-QUALITY BLAS builder: tinybvh's SBVH, converted into our
// own 32-byte build format.
//
// WHY THIS EXISTS (docs/rt_lighting_plan.md §5.0, "Offline HQ BLAS"). Our
// binned-SAH builder (`bvh.cpp`) splits triangles by CENTROID, so a long
// diagonal triangle drags its whole AABB into whichever side its centre lands
// on and every ray that enters that side pays for it. An SBVH additionally
// considers SPATIAL splits: it may cut a primitive's *bounds* at a plane and
// reference the same triangle from both children, which is the only way to get
// tight boxes around geometry that straddles. The price is duplicated
// references — a leaf window is a window over triangle REFERENCES, not over
// distinct triangles — and a build that cannot be refitted.
//
// WHAT THIS FILE IS NOT. It is not a second traversal format. tinybvh's own
// node layouts, its traversal kernels and its GPU formats are all unused: the
// only thing taken from it is the TOPOLOGY of the tree it builds, which is
// re-emitted here into `Blas` exactly as `build_blas` would have emitted it.
// Everything downstream — `convert_to_gpu_nodes`, the 64-byte upload format,
// the shaders, the tests — is unchanged and unaware.
//
// THE INVARIANTS THAT MAKE THAT SAFE, all re-established by the conversion
// rather than assumed of tinybvh:
//   * siblings adjacent (`right == left_first + 1`) — the 32-byte node has no
//     room for a right link and `convert_to_gpu_nodes` reads `left + 1`;
//   * children at HIGHER indices than their parent — `refit_tlas`'s reverse
//     index walk and the conversion's one-pass numbering both depend on it;
//   * `prim_id` is the SOURCE triangle ordinal, so `rt_scene.cpp`'s attribute
//     table (indexed in lockstep with triangles, resolved through `prim_id`)
//     keeps resolving — duplicated references simply produce duplicate,
//     identical attribute rows;
//   * depth within `BvhBuildOptions::max_depth`, which is what sizes the
//     shaders' 64-entry traversal stack.
// Any of them failing is reported as a failure, not fixed up: the caller falls
// back to the binned build, which is always available anyway.
//
// This header deliberately does NOT include tiny_bvh.h. The vendored header
// lives next to it and is included by `hq_blas_builder.cpp` alone, so even
// inside `internal/` the dependency has exactly one TU.

#include "bracken/render/raytracing/bvh.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <span>

namespace bracken::render::rt {

/** What one HQ build did, for the log line and for the tests. */
struct HqBlasStats {
    /// Wall time of tinybvh's SBVH build alone.
    float build_ms = 0.0f;
    /// Wall time of the re-emit into `Blas`, including the validation pass.
    float convert_ms = 0.0f;
    /// Triangles handed in (after the same malformed-index filter `build_blas`
    /// applies), i.e. the number of distinct `prim_id`s.
    uint32_t source_triangles = 0;
    /// Triangle REFERENCES in the finished tree. Greater than
    /// `source_triangles` exactly when spatial splits duplicated something.
    uint32_t triangle_refs = 0;
    uint32_t node_count = 0;
    uint32_t leaf_count = 0;
    uint32_t max_depth = 0;
    float sah_cost = 0.0f;
    /// Null on success; otherwise the invariant that was violated, for the
    /// caller's WARN. Static storage — safe to log after the builder returns.
    const char* failure = nullptr;
};

struct HqBlasOptions {
    /// SBVH bins per axis. tinybvh defaults to 8; 32 is the value its own
    /// documentation suggests for quality builds and is what we pay for here,
    /// since this path is opt-in and load-time.
    uint32_t sbvh_bins = 32;
    /// Iterations of tinybvh's reinsertion optimizer. **0 = off, and off is
    /// the shipped setting** — see the note at `run_optimizer` in the .cpp for
    /// the measurement and the leak that decided it.
    uint32_t optimize_iterations = 0;
};

/**
 * Build a BLAS with tinybvh's SBVH and convert it into `out`.
 *
 * Same inputs as `build_blas`. Returns false and leaves `out` untouched when
 * any conversion invariant fails, with `stats.failure` naming which; the caller
 * is expected to keep the binned-SAH tree it already has.
 *
 * Deterministic: tinybvh's own threading is compiled out (`NO_THREADED_BUILDS`)
 * and the optimizer's stochastic mode — the only `rand()` in the library's
 * build path — is never selected. Same mesh in, same tree out, on any worker.
 */
[[nodiscard]] bool build_blas_hq(std::span<const glm::vec3> positions,
                                 std::span<const uint32_t> indices,
                                 const BvhBuildOptions& options, const HqBlasOptions& hq_options,
                                 Blas& out, HqBlasStats& stats);

/**
 * Whether `BRACKEN_RT_HQ_BLAS=1` is set in the environment.
 *
 * An ENV VAR and not a `GraphicsSettings` field on purpose: the BLAS set is
 * built once, at load, from CPU geometry that is released immediately after.
 * A runtime toggle would have to either rebuild every BLAS mid-frame or lie
 * about what is in the buffers, and neither is worth a settings row for a
 * choice a host makes once. Same reasoning, same spelling, as
 * `BRACKEN_WORKERS`.
 *
 * Read once and cached, so every model in a parallel build sees one answer.
 */
[[nodiscard]] bool hq_blas_requested();

} // namespace bracken::render::rt
