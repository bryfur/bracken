#pragma once

#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace bracken::gpu {
class PipelineRegistry;
} // namespace bracken::gpu

namespace bracken::render {

namespace gpu = bracken::gpu;

// =============================================================================
// The pyramid's index and reduction math, as pure functions.
//
// Everything below is mirrored texel for texel by shaders/hzb_reduce.comp.hlsl.
// It lives here rather than only in HLSL for the same reason gi_math.hpp does:
// a footprint that is wrong at an odd edge is invisible in a screenshot and
// obvious in a brute-force test.
//
// ---------------------------------------------------------------------------
// WHY THE REDUCTION IS A MAX, NOT A MIN
//
// The engine's scene depth is reversed-Z, zero-to-one (see CLAUDE.md and
// docs/architecture.md): the projection swaps near and far, so the NEAR plane
// maps to depth 1 and the FAR plane to depth 0, and a LARGER stored value means
// a CLOSER surface. Scene pipelines therefore compare GREATER_OR_EQUAL and
// clear depth to 0.0.
//
// What a screen-space ray march wants from the pyramid is a conservative
// occlusion test over a whole footprint: it may skip a footprint in one step
// only when the ray is provably IN FRONT of every surface inside it — that is,
// closer to the camera than the footprint's CLOSEST surface. So the value a
// coarse texel has to retain is the closest surface anywhere in its footprint,
// which under reversed-Z is the MAXIMUM of the source values. Take the minimum
// instead and the texel describes the FARTHEST surface: the march would then
// step over a near wall whose depth the pyramid had thrown away, and screen
// rays would pass through geometry.
//
// docs/rt_lighting_plan.md §5.5 calls this a "min-depth pyramid". That is the
// standard-Z spelling of the identical rule — closest-surface-wins — and the
// conversion, not a contradiction. The pyramid is named after what it keeps
// (the closest surface), and the arithmetic follows the depth convention of the
// buffer it is built from, which here is reversed.
//
// ---------------------------------------------------------------------------
// WHY THE FOOTPRINT IS NOT ALWAYS 2x2
//
// Mip N has dimensions max(1, size >> N), so an odd source dimension loses its
// last row/column to integer division: a 5-wide source reduces to 2, and a
// plain 2x2 gather covers source columns 0..3 and silently drops column 4. In a
// conservative pyramid that is not an approximation, it is a hole — the dropped
// column may hold the only near surface in the footprint, and a march that
// trusts the mip walks straight through it.
//
// The fix is to EXTEND the last destination texel to the end of the source
// rather than to clamp its reads (clamping re-reads column 3 and still never
// sees column 4). Because a source dimension is either 2*dst or 2*dst+1, the
// extension is at most one texel, so the gather is 2x2 or, at one edge, 3x2 /
// 2x3 / 3x3. The footprints stay a partition of the source: every source texel
// belongs to exactly one destination texel, which is what makes "mip N texel ==
// max over its mip-0 region" true by induction.
// =============================================================================

/** Dimension of mip `level` of a chain whose mip 0 measures `size0`. */
[[nodiscard]] constexpr int hzb_mip_size(int size0, int level) {
    int s = size0;
    for (int i = 0; i < level; ++i) {
        s = s > 1 ? s / 2 : 1;
    }
    return s < 1 ? 1 : s;
}

/**
 * Levels in the full chain for a `width` x `height` mip 0, i.e. how many mips
 * it takes to reach 1x1: floor(log2(max(width, height))) + 1.
 */
[[nodiscard]] constexpr int hzb_mip_count(int width, int height) {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    int levels = 1;
    int w = width;
    int h = height;
    while (w > 1 || h > 1) {
        w = w > 1 ? w / 2 : 1;
        h = h > 1 ? h / 2 : 1;
        ++levels;
    }
    return levels;
}

/** Half-open source range [begin, end) along one axis. */
struct HzbSpan {
    int begin = 0;
    int end = 0;

    [[nodiscard]] constexpr int count() const { return end - begin; }
    friend constexpr bool operator==(const HzbSpan&, const HzbSpan&) = default;
};

/**
 * Source texels destination texel `dst` covers along one axis, given the
 * source and destination extents of that axis.
 *
 * The last destination texel absorbs an odd source's trailing texel (see the
 * header comment); every other one covers exactly two. Out-of-range `dst`
 * yields an empty span rather than a wrapped one.
 */
[[nodiscard]] constexpr HzbSpan hzb_source_span(int dst, int dst_extent, int src_extent) {
    if (dst < 0 || dst >= dst_extent || dst_extent <= 0 || src_extent <= 0) {
        return {0, 0};
    }
    if (dst_extent >= src_extent) {
        // 1x1 source, or a degenerate chain step that does not halve: one to one.
        return {dst, dst + 1};
    }
    const int begin = dst * 2;
    const int end = (dst == dst_extent - 1) ? src_extent : std::min(begin + 2, src_extent);
    return {begin, end};
}

/**
 * The conservative reduction itself: closest-surface-wins under reversed-Z.
 * A separate name so the tests and the doc argue about a function rather than
 * about an operator that could be flipped without anyone noticing.
 */
[[nodiscard]] constexpr float hzb_reduce(float a, float b) { return a > b ? a : b; }

/**
 * The OPPOSITE reduction: farthest-surface-wins under reversed-Z.
 *
 * WHY THIS IS A SECOND PYRAMID AND NOT A FLAG AT READ TIME. A screen march
 * needs the closest surface in a footprint (above); DRAW OCCLUSION CULLING
 * needs the farthest. A box is hidden only when its nearest point is behind
 * the farthest occluder ANYWHERE in its footprint — anything weaker culls a box
 * that peeks through a gap between two occluders in the same texel. The two
 * answers are different reductions of the same source, so they cannot be
 * recovered from one another after the fact: a stored maximum simply does not
 * contain the minimum.
 *
 * The failure modes are asymmetric, which is why this has its own name and its
 * own test. Cull against the CLOSEST pyramid with the natural comparison and
 * nothing is ever culled — harmless, and it reads as "occlusion culling does
 * not help in this scene". Flip the comparison to make it cull and it culls
 * VISIBLE geometry, which shows up as props missing behind walls and moves the
 * mean luma by nothing at all. `render/hzb.test.cpp` pins `far <= near` texel
 * for texel, which is the cheap guard against wiring the culler to the wrong
 * one.
 */
[[nodiscard]] constexpr float hzb_reduce_far(float a, float b) { return a < b ? a : b; }

/** Which surface a coarse texel retains. See hzb_reduce / hzb_reduce_far. */
enum class HzbReduction : uint8_t {
    Closest, ///< MAX under reversed-Z — the screen-probe march (R4b)
    Farthest ///< MIN under reversed-Z — draw occlusion culling
};

/**
 * Hierarchical-Z pyramid over the depth prepass (docs/rt_lighting_plan.md
 * §5.5, R4a).
 *
 * One R32_FLOAT texture with a full mip chain at the prepass depth's
 * resolution. Mip 0 is a straight copy of the D32 prepass depth — D32 cannot be
 * a compute storage texture, so the copy pass reads it through a SAMPLER and
 * writes the R32F level; every further mip is the conservative reduction of the
 * level above it, one dispatch per mip.
 *
 * BUILT BUT UNCONSUMED. R4a exists to put the pyramid and the gpu/ layer's
 * indirect-dispatch wrapper in the tree; R4b's screen-probe gather is what
 * marches it. It is therefore only built when gi_mode == GiMode::ProbesScreen,
 * so no existing configuration allocates it or spends a dispatch on it.
 *
 * ONE DISPATCH PER MIP, deliberately. A single-pass groupshared ladder (the
 * FidelityFX SPD shape — one dispatch computing up to 12 mips through LDS and a
 * global counter) is the known optimization and is NOT this package: it needs
 * a globally-coherent atomic counter buffer and a tile-tail workgroup, and it
 * would land before anything measures what the naive chain costs. Each mip also
 * needs its own compute pass regardless of the shader, because SDL3 gives no
 * intra-pass synchronization — a dispatch that reads the previous dispatch's
 * output must sit in a later pass.
 */
class HzbPyramid {
public:
    HzbPyramid() = default;
    ~HzbPyramid();

    HzbPyramid(const HzbPyramid&) = delete;
    HzbPyramid& operator=(const HzbPyramid&) = delete;
    HzbPyramid(HzbPyramid&&) = delete;
    HzbPyramid& operator=(HzbPyramid&&) = delete;

    /**
     * Allocate the pyramid at `width` x `height`.
     *
     * Fails (logging why, once) when the device cannot do R32_FLOAT with the
     * usage the build needs — the reduce pass reads one mip and writes the next
     * of the SAME texture, which SDL only permits with
     * COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE, and that flag is documented as
     * supported by a limited set of formats.
     */
    [[nodiscard]] bool init(gpu::GPUDevice& device, int width, int height,
                            HzbReduction reduction = HzbReduction::Closest);
    void shutdown();

    /** Which surface this pyramid's coarse texels retain. Fixed at init: the
     *  reduction is baked into the levels, so it cannot be a read-time choice. */
    [[nodiscard]] HzbReduction reduction() const noexcept { return reduction_; }

    /** Reallocate at a new size; no-op at the same size. Contents are not
     *  preserved — the pyramid is rebuilt from scratch every frame anyway. */
    void resize(int width, int height);

    [[nodiscard]] bool is_ready() const { return texture_ != nullptr; }

    int width() const { return width_; }
    int height() const { return height_; }
    int mip_count() const { return mip_count_; }

    /** The pyramid texture (R32_FLOAT, `mip_count()` levels). Null until
     *  init(). Sampled by R4b; SAMPLER usage is requested for that reason. */
    SDL_GPUTexture* texture() const { return texture_ ? texture_->handle() : nullptr; }

    /**
     * Record the whole chain: one copy pass for mip 0, then one compute pass
     * per further mip. No render/copy/compute pass may be open on `cmd`.
     *
     * `depth` is the full-res prepass scene depth and must hold THIS frame's
     * contents (SceneRenderer::depth_prepass_active()); its dimensions must
     * match the pyramid's — the caller resizes first.
     */
    void build(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, gpu::GPUTexture& depth);

    /** Bytes a `level` readback needs in its destination transfer buffer. */
    [[nodiscard]] std::size_t level_readback_bytes(int level) const;

    /**
     * Record a download of ONE coarse mip into a caller-owned transfer buffer,
     * on the caller's OWN command buffer, and return the fence to wait on.
     *
     * `level` must be >= 1: mip 0 is the full screen (5.8 MB at 1600x900),
     * mip 5 is 5.6 KB, and the CPU cull wants the small one. Returns null when
     * the pyramid is not ready, the level is out of range, or the submit
     * failed; the caller owns the returned fence and must release it.
     *
     * SEPARATE COMMAND BUFFER, deliberately, following the idiom every existing
     * readback in the tree uses (render/auto_exposure.cpp, gi/screen_probes.cpp,
     * gi/probe_volume.cpp): the frame's command buffer is submitted at present
     * time, and a fence acquired from it could not be waited on until then.
     * No pass may be open on `cmd` — this opens a copy pass.
     *
     * Main-thread.
     */
    [[nodiscard]] SDL_GPUFence* request_level_readback(int level, SDL_GPUTransferBuffer* dst);

private:
    void create_texture();

    gpu::GPUDevice* device_ = nullptr;
    std::unique_ptr<gpu::GPUTexture> texture_;
    SDL_GPUSampler* nearest_clamp_sampler_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    int mip_count_ = 0;
    HzbReduction reduction_ = HzbReduction::Closest;
};

} // namespace bracken::render
