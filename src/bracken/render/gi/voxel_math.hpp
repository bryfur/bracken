#pragma once

/**
 * The VOXEL RADIANCE CLIPMAP's addressing, cascade selection and cone-step math
 * — the CPU source of truth for `shaders/voxel_common.hlsli` and every voxel
 * pass that includes it.
 *
 * WHY THIS EXISTS AS A HEADER RATHER THAN AS SHADER CODE. Exactly the reason
 * `gi_math.hpp` does: a clipmap whose CPU and GPU disagree about which texel
 * holds which point in the world produces lighting that is subtly wrong
 * everywhere and looks like a shading bug for a week. Everything here is
 * transliterated into HLSL, and everything here is tested with no GPU, no
 * window and no SDL.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS IS FOR
 * ---------------------------------------------------------------------------
 *
 * A raster alternative to the ray-traced probe field (docs/rt_lighting_plan.md
 * §5.2 traces rays into a BVH; this traces CONES into a voxel volume). The two
 * meet at exactly one place — `ProbeVolume::ray_results_`, a buffer of
 * `float4(radiance, distance)` per (probe, ray) — so everything downstream of
 * that buffer is SHARED: the octahedral irradiance and visibility atlases, the
 * blend, the Chebyshev leak test, probe classification and relocation, the
 * cascade blend, the fog in-scatter, and every fragment shader that samples the
 * field. This module replaces the ray source and nothing else.
 *
 * ---------------------------------------------------------------------------
 * THE CLIPMAP
 * ---------------------------------------------------------------------------
 *
 * `CASCADE_COUNT` nested volumes, all the same texel resolution, each covering
 * twice the world extent of the one before and all centred on the camera. A
 * cone samples the finest cascade that still contains the sample point, so
 * detail is spent where the eye is and the far field costs nothing extra.
 *
 * TORUS-ADDRESSED, like the probe field: the volume does not move in memory
 * when the camera moves. The base corner advances in whole voxels and the
 * addressing wraps, so a voxel still inside the window after a scroll keeps the
 * exact texel it had and only the shell that scrolled in needs revoxelizing.
 * Snapping to whole voxels is what makes that true — a fractional scroll would
 * shift every voxel's world position and invalidate the entire volume.
 *
 * Tier: pure (no state, no device, callable from anywhere).
 */

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace bracken::render::gi {

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

/**
 * Texels per axis in every cascade.
 *
 * 96 rather than 128, and the reason is the memory table rather than taste. Two
 * volumes per cascade survive the design — an RGBA8 albedo+opacity that only the
 * injection pass reads (so it needs no mip chain) and an RGBA16F radiance that
 * the cones sample (so it does). At 128 that is 25 + 57 = 82 MB on top of the
 * probe field's 30, which is a lot to ask of a feature whose entire purpose is
 * to be the CHEAPER alternative to ray tracing. At 96 it is ~35 MB.
 *
 * The cost of the reduction is voxel WORLD size at a fixed cascade extent, and
 * that is the number that decides how thin a wall can be before light leaks
 * through it — so the extent is held by shrinking coverage rather than by
 * growing `VOXEL_BASE_SIZE`. `voxel_cascade_bytes` is the function to re-run
 * before changing this.
 *
 * NO NORMAL VOLUME. Classic VXGI stores one so injection can evaluate N·L, but
 * injection here runs from the TRIANGLE and already has the geometric normal in
 * hand; storing it per voxel would only let a later pass re-derive what the
 * earlier one knew. That is a third of the memory saved for nothing given up.
 */
inline constexpr int VOXEL_RESOLUTION = 96;

/// Nested cascades. Three covers 2^2 = 4x the finest extent, which at the
/// finest voxel size below is a ~40 m radius of full-detail GI around the
/// camera and ~160 m of coarse — past that the sky term dominates anyway.
inline constexpr int VOXEL_CASCADE_COUNT = 3;

/// World units per voxel in cascade 0. The playground is ~6 cm/unit, so 25
/// units is ~1.5 m: small enough that a building wall is several voxels thick
/// (leak-free) and large enough that 128^3 covers a useful radius.
inline constexpr float VOXEL_BASE_SIZE = 25.0f;

/// Voxels of overlap kept when choosing which cascade answers a sample, so a
/// cone crossing a cascade boundary never samples outside the finer volume's
/// valid region. Two voxels covers a trilinear tap (which reaches one voxel) at
/// the coarsest mip a cascade is sampled at.
inline constexpr float VOXEL_CASCADE_MARGIN = 2.0f;

/// World size of one voxel in `cascade`.
[[nodiscard]] inline constexpr float voxel_size(int cascade) noexcept {
    return VOXEL_BASE_SIZE * static_cast<float>(1 << cascade);
}

/// World extent of one whole cascade, edge to edge.
[[nodiscard]] inline constexpr float voxel_cascade_extent(int cascade) noexcept {
    return voxel_size(cascade) * static_cast<float>(VOXEL_RESOLUTION);
}

/// Half-extent — the radius of the axis-aligned box the cascade covers.
[[nodiscard]] inline constexpr float voxel_cascade_half_extent(int cascade) noexcept {
    return voxel_cascade_extent(cascade) * 0.5f;
}

// ---------------------------------------------------------------------------
// Scrolling
// ---------------------------------------------------------------------------

/**
 * The cascade's base corner, in WHOLE VOXELS of that cascade.
 *
 * Snapped with `floor`, not rounded toward zero: the addressing below takes a
 * Euclidean modulo, and a base that flipped sign discontinuously at the origin
 * would tear the volume along the world axes. `floor` is monotone everywhere,
 * so the base advances by one voxel at a time no matter which way the camera
 * crosses zero.
 */
[[nodiscard]] inline glm::ivec3 voxel_cascade_base(int cascade, const glm::vec3& camera) noexcept {
    const float vs = voxel_size(cascade);
    const float half = voxel_cascade_half_extent(cascade);
    const glm::vec3 corner = (camera - glm::vec3(half)) / vs;
    return {static_cast<int>(std::floor(corner.x)), static_cast<int>(std::floor(corner.y)),
            static_cast<int>(std::floor(corner.z))};
}

/// World position of a cascade voxel's CENTRE, from its absolute lattice
/// coordinate (base + local). The half-voxel offset is what makes a trilinear
/// tap at this position return exactly this voxel.
[[nodiscard]] inline glm::vec3 voxel_center(int cascade, const glm::ivec3& absolute) noexcept {
    const float vs = voxel_size(cascade);
    return (glm::vec3(absolute) + glm::vec3(0.5f)) * vs;
}

// ---------------------------------------------------------------------------
// Torus addressing
// ---------------------------------------------------------------------------

/// Euclidean modulo. C++ `%` truncates toward zero and every wrap here crosses
/// the origin, so the naive form would fold negative coordinates onto the wrong
/// texel. Same function, same reason, as `gi_math.hpp`'s.
[[nodiscard]] inline constexpr int voxel_wrap_axis(int v, int n) noexcept {
    const int m = v % n;
    return m < 0 ? m + n : m;
}

/// Absolute lattice coordinate -> storage texel. A voxel still inside the
/// window after a scroll keeps the texel it had, which is the whole point.
[[nodiscard]] inline glm::ivec3 voxel_wrap(const glm::ivec3& absolute) noexcept {
    return {voxel_wrap_axis(absolute.x, VOXEL_RESOLUTION),
            voxel_wrap_axis(absolute.y, VOXEL_RESOLUTION),
            voxel_wrap_axis(absolute.z, VOXEL_RESOLUTION)};
}

/**
 * Is this absolute lattice coordinate inside the cascade's current window?
 *
 * Torus addressing wraps ANY coordinate to a valid texel, so this is the only
 * thing standing between a sample outside the window and a confident answer
 * read from the opposite side of the world. Every read must test it.
 */
[[nodiscard]] inline bool voxel_in_window(const glm::ivec3& absolute,
                                          const glm::ivec3& base) noexcept {
    const glm::ivec3 local = absolute - base;
    return local.x >= 0 && local.y >= 0 && local.z >= 0 && local.x < VOXEL_RESOLUTION &&
           local.y < VOXEL_RESOLUTION && local.z < VOXEL_RESOLUTION;
}

/// World position -> that cascade's absolute lattice coordinate.
[[nodiscard]] inline glm::ivec3 voxel_coord_at(int cascade, const glm::vec3& world) noexcept {
    const glm::vec3 v = world / voxel_size(cascade);
    return {static_cast<int>(std::floor(v.x)), static_cast<int>(std::floor(v.y)),
            static_cast<int>(std::floor(v.z))};
}

/**
 * World position -> normalized volume texture coordinate for that cascade.
 *
 * Returns the coordinate to feed a trilinear sampler. Because storage is a
 * torus, this is `frac` of the world position in voxels divided by the
 * resolution — the wrap the sampler does in REPEAT mode is exactly the wrap the
 * addressing wants, which is why the volumes are created with a repeating
 * sampler rather than clamped.
 */
[[nodiscard]] inline glm::vec3 voxel_sample_uvw(int cascade, const glm::vec3& world) noexcept {
    return world / (voxel_size(cascade) * static_cast<float>(VOXEL_RESOLUTION));
}

// ---------------------------------------------------------------------------
// Cascade selection
// ---------------------------------------------------------------------------

/**
 * The finest cascade whose window contains `world` with `VOXEL_CASCADE_MARGIN`
 * voxels to spare, or -1 when even the coarsest does not.
 *
 * The margin is not politeness: a trilinear tap reaches half a voxel past the
 * sample point and a mip tap reaches further, so a sample accepted right at the
 * window edge would filter against texels belonging to the opposite side of the
 * torus. Selecting by MARGIN rather than by containment is what keeps a cone
 * that crosses a cascade boundary from picking up a seam.
 */
[[nodiscard]] inline int voxel_select_cascade(const glm::vec3& world,
                                              const glm::vec3& camera) noexcept {
    for (int c = 0; c < VOXEL_CASCADE_COUNT; ++c) {
        const float half = voxel_cascade_half_extent(c) - VOXEL_CASCADE_MARGIN * voxel_size(c);
        const glm::vec3 d = glm::abs(world - camera);
        if (d.x <= half && d.y <= half && d.z <= half) {
            return c;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Cone tracing
// ---------------------------------------------------------------------------

/**
 * Diameter of a cone of half-angle `tan_half_angle` at distance `t`, floored at
 * one cascade-0 voxel.
 *
 * This is what turns a cone trace into a MIP SELECTION: the correct mip is the
 * one whose voxels are about as wide as the cone is at that distance, so the
 * single filtered tap taken there stands in for the whole cone cross-section.
 * That substitution is the entire technique — it is why a cone costs a few tens
 * of taps instead of the thousands of rays it represents.
 */
[[nodiscard]] inline float voxel_cone_diameter(float t, float tan_half_angle) noexcept {
    return std::max(2.0f * t * tan_half_angle, VOXEL_BASE_SIZE);
}

/**
 * Mip level whose voxels are `diameter` wide, within `cascade`.
 *
 * Fractional on purpose: the sampler interpolates between mips, so a cone that
 * widens smoothly gets a smoothly widening filter instead of stepping.
 */
[[nodiscard]] inline float voxel_cone_mip(float diameter, int cascade) noexcept {
    return std::max(std::log2(diameter / voxel_size(cascade)), 0.0f);
}

/**
 * Front-to-back alpha compositing, one cone step.
 *
 * `accum` is (rgb premultiplied radiance, a occlusion). Standard over-operator:
 * a sample only contributes what the samples in front of it left unblocked,
 * which is what makes a cone that hits a wall stop reporting whatever is behind
 * it. Returns the updated accumulator.
 */
[[nodiscard]] inline glm::vec4 voxel_cone_accumulate(const glm::vec4& accum,
                                                     const glm::vec4& sample) noexcept {
    const float transmittance = 1.0f - accum.a;
    return {accum.r + transmittance * sample.a * sample.r,
            accum.g + transmittance * sample.a * sample.g,
            accum.b + transmittance * sample.a * sample.b,
            accum.a + transmittance * sample.a};
}

/// Occlusion above which a cone is considered to have hit something, for the
/// purpose of reporting an effective distance back to the probe field's
/// Chebyshev moments. 0.5 rather than 1.0 because a cone that is half occluded
/// has already found the surface the visibility test cares about, and waiting
/// for saturation biases every recorded distance long.
inline constexpr float VOXEL_CONE_HIT_ALPHA = 0.5f;

/// Occlusion at which a cone stops early. Below 1 because the last few percent
/// of transmittance cannot change the result perceptibly and the steps are the
/// pass's whole cost.
inline constexpr float VOXEL_CONE_TERMINATE_ALPHA = 0.98f;

/**
 * Step length along a cone at distance `t`.
 *
 * Proportional to the cone's own width, which is what keeps the sample count
 * bounded: near the apex the cone is thin and steps are short, far away it is
 * wide and steps are long, so the number of steps grows with the LOGARITHM of
 * distance rather than linearly. The 0.5 factor is a half-diameter step, the
 * standard rate at which front-to-back compositing does not under-count a
 * surface it passes through.
 */
[[nodiscard]] inline float voxel_cone_step(float t, float tan_half_angle) noexcept {
    return std::max(0.5f * voxel_cone_diameter(t, tan_half_angle), VOXEL_BASE_SIZE * 0.5f);
}

// ---------------------------------------------------------------------------
// Memory accounting
// ---------------------------------------------------------------------------

/// Bytes one cascade of a volume of `bytes_per_texel` occupies, mip chain
/// included. The chain converges to 8/7 of the base level.
[[nodiscard]] inline std::uint64_t voxel_cascade_bytes(int bytes_per_texel, bool mipmapped) noexcept {
    const auto base = static_cast<std::uint64_t>(VOXEL_RESOLUTION) * VOXEL_RESOLUTION *
                      VOXEL_RESOLUTION * static_cast<std::uint64_t>(bytes_per_texel);
    if (!mipmapped) {
        return base;
    }
    std::uint64_t total = 0;
    for (int r = VOXEL_RESOLUTION; r >= 1; r /= 2) {
        total += static_cast<std::uint64_t>(r) * r * r * static_cast<std::uint64_t>(bytes_per_texel);
    }
    return total;
}

} // namespace bracken::render::gi
