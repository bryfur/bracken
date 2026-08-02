#pragma once

/**
 * @file screen_probe_math.hpp
 * @brief The screen-probe gather's index, direction, march and SH math, as pure
 *        functions — the CPU source of truth for `shaders/gi_screen_probe.hlsli`.
 *
 * Phase R4b of docs/rt_lighting_plan.md §5.5. Same contract as
 * `render/gi/gi_math.hpp` and `render/hzb.hpp`: every function below is
 * transliterated texel for texel into HLSL, and it lives here because the
 * failures this code has are *invisible in a screenshot* — a footprint that is
 * wrong at an odd edge, a jitter that repeats, an SH basis with a sign flipped
 * — and obvious against a brute-force reference.
 *
 * No GPU, no SDL, no device: `screen_probe_math.test.cpp` runs all of it
 * headless.
 */

#include "bracken/render/gi/gi_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>

namespace bracken::render::gi {

// ============================================================================
// The grid
// ============================================================================

/// Screen pixels per probe tile, one probe each.
///
/// 16, not GI-1.0's 8. The engine's probe payload is 8x8 = 64 rays and only the
/// hemisphere above the anchor is traced (see `screen_probe_texel_direction`),
/// so a tile is 32 rays; at 8x8 tiles that is one ray for every two pixels,
/// which is a per-pixel budget wearing a probe's clothes. 16x16 puts it at one
/// ray per eight pixels — an amortized workload, which is the only kind §2 of
/// the plan says fits the hardware floor this engine targets. The price is
/// paid in the interpolation: a 16-pixel probe pitch is where the
/// depth/normal-weighted 4-probe gather and the field fallback earn their
/// keep, and it is why there is a weight floor at all.
inline constexpr int SCREEN_PROBE_TILE = 16;

/// Octahedral payload per probe. NO BORDER RING, unlike the world field's
/// atlases: nothing bilinearly samples this atlas across a payload edge. The
/// spatial filter and the SH projection both read it texel-for-texel at
/// matching indices across probes, and the forward pass reads the SH atlas
/// rather than this one — so the border's only job would be to be written.
inline constexpr int SCREEN_PROBE_OCT = 8;

/// Ray slots per probe = one per octahedral texel.
inline constexpr int SCREEN_PROBE_RAYS = SCREEN_PROBE_OCT * SCREEN_PROBE_OCT;

/// float4 quads in the published SH record: three of L1 spherical harmonics (one
/// per colour channel, four coefficients each) plus one of metadata.
///
/// FOUR, not the three §5.5's sketch assumed. Three quads hold exactly the
/// twelve SH floats and not one bit more, and the forward gather needs the
/// anchor's normal and camera distance to weight its four probes — without them
/// the "bilinear across the 2x2 nearest probes, depth/normal-weighted" step has
/// nothing to weight with. The record is `gpu::GiScreenProbeSh`, 64 bytes, and
/// it is a STORAGE BUFFER rather than a texture because SDL3 GPU caps a shader
/// stage at 16 samplers and the scene shaders are already there.
inline constexpr int SCREEN_PROBE_SH_QUADS = 4;

/// Probe grid covering a `width` x `height` framebuffer. The last row/column of
/// tiles is partial when the framebuffer is not a multiple of the tile; its
/// probe still exists and its anchor search simply skips the pixels that are
/// off-screen.
[[nodiscard]] constexpr glm::ivec2 screen_probe_grid(int width, int height) {
    if (width <= 0 || height <= 0) {
        return {0, 0};
    }
    return {(width + SCREEN_PROBE_TILE - 1) / SCREEN_PROBE_TILE,
            (height + SCREEN_PROBE_TILE - 1) / SCREEN_PROBE_TILE};
}

/// Row-major probe index. The dispatch is one threadgroup per probe, so this is
/// also the group id.
[[nodiscard]] constexpr int screen_probe_index(glm::ivec2 probe, glm::ivec2 grid) {
    return probe.y * grid.x + probe.x;
}

[[nodiscard]] constexpr glm::ivec2 screen_probe_coord(int index, glm::ivec2 grid) {
    return {grid.x > 0 ? index % grid.x : 0, grid.x > 0 ? index / grid.x : 0};
}

/// Probes in a whole grid.
[[nodiscard]] constexpr int screen_probe_count(glm::ivec2 grid) { return grid.x * grid.y; }

// ============================================================================
// Anchor placement
// ============================================================================

/// How many pixels the anchor search may try before giving the tile up.
inline constexpr int SCREEN_PROBE_ANCHOR_ATTEMPTS = 9;

/**
 * The `attempt`-th candidate pixel inside a tile, as an offset in [0, TILE).
 *
 * A FIXED ORDER, not a search: the probe's anchor decides which surface the
 * whole tile's indirect light is measured at, and "the most-common-depth pixel"
 * would need a per-tile histogram to answer — 256 depth taps and a reduction,
 * for a choice the interpolation weights already tolerate being approximate.
 * What must NOT be approximate is that two frames with the same depth buffer
 * pick the same pixel, because a probe that hops between two surfaces inside
 * its tile invalidates its own temporal history every frame and reads as noise.
 *
 * So: the centre, then the four quarter-points, then the four edge midpoints.
 * The first candidate with a valid (non-sky) depth wins. The centre is first
 * because it is the pixel whose interpolation weight is highest for the whole
 * tile; the rest walk outward so a tile whose centre is sky but whose corner is
 * a rooftop still gets a probe rather than falling to the world field.
 */
[[nodiscard]] constexpr glm::ivec2 screen_probe_anchor_offset(int attempt) {
    constexpr int H = SCREEN_PROBE_TILE / 2;      // 8
    constexpr int Q = SCREEN_PROBE_TILE / 4;      // 4
    constexpr int L = SCREEN_PROBE_TILE - 1;      // 15
    constexpr int T = SCREEN_PROBE_TILE - 1 - Q;  // 11
    switch (attempt) {
    case 0: return {H, H};
    case 1: return {Q, Q};
    case 2: return {T, Q};
    case 3: return {Q, T};
    case 4: return {T, T};
    case 5: return {0, H};
    case 6: return {L, H};
    case 7: return {H, 0};
    case 8: return {H, L};
    default: return {H, H};
    }
}

/// Pack an anchor's in-tile offset into the probe record's flag word. Four bits
/// per axis is exactly enough for a 16-pixel tile, which is why the tile size
/// and this packing have to change together.
inline constexpr std::uint32_t SCREEN_PROBE_FLAG_VALID = 1u;
inline constexpr int SCREEN_PROBE_ANCHOR_SHIFT = 8;

[[nodiscard]] constexpr std::uint32_t screen_probe_pack_flags(glm::ivec2 anchor_local, bool valid) {
    const std::uint32_t x = static_cast<std::uint32_t>(anchor_local.x & 0xF);
    const std::uint32_t y = static_cast<std::uint32_t>(anchor_local.y & 0xF);
    return (valid ? SCREEN_PROBE_FLAG_VALID : 0u) |
           (x << SCREEN_PROBE_ANCHOR_SHIFT) | (y << (SCREEN_PROBE_ANCHOR_SHIFT + 4));
}

[[nodiscard]] constexpr glm::ivec2 screen_probe_unpack_anchor(std::uint32_t flags) {
    return {static_cast<int>((flags >> SCREEN_PROBE_ANCHOR_SHIFT) & 0xF),
            static_cast<int>((flags >> (SCREEN_PROBE_ANCHOR_SHIFT + 4)) & 0xF)};
}

// ============================================================================
// Ray directions
// ============================================================================

/**
 * The per-frame sub-texel jitter, in [0,1)^2.
 *
 * The R2 low-discrepancy sequence (`rt_sampling.hlsli`'s `r2Sequence`, the 2D
 * member of the same family `gi_probe.hlsli` uses in 3D), evaluated at the
 * frame counter alone. Deterministic in the frame index is the whole
 * requirement: `--smoke` reproduces bit for bit, and the temporal filter's
 * EMA is what turns "one direction per texel per frame" into a converged
 * integral over the texel's solid angle.
 */
[[nodiscard]] inline glm::vec2 screen_probe_jitter(std::uint32_t frame_index) {
    constexpr double R2_A = 0.7548776662466927; // 1/g
    constexpr double R2_B = 0.5698402909980532; // 1/g^2
    const double f = static_cast<double>(frame_index);
    const auto frac = [](double v) { return static_cast<float>(v - std::floor(v)); };
    // OFFSET BY 0.5, so frame 0 is the texel CENTRE and not its corner. That is
    // not cosmetic: at (0,0) the octahedral decode lands on the fold seam, where
    // encode/decode is not a round trip, and — worse — a whole probe's 64
    // directions all sitting at their texels' corners is a systematically
    // one-sided set whose L1 projection has a bias no amount of temporal
    // accumulation removes. `ScreenProbeDirections` and `ShL1` both fail without
    // this, which is how it was found.
    return {frac(0.5 + R2_A * f), frac(0.5 + R2_B * f)};
}

/**
 * World-space direction of octahedral texel `texel` on frame `frame_index`.
 *
 * A FULL-SPHERE octahedral map, in world space, deliberately: one probe serves
 * every pixel in its tile and those pixels do not share a normal (a tile
 * straddling a wall corner has two). A hemisphere map oriented to the anchor's
 * normal would be denser but would have no answer at all for the other face.
 *
 * The lower hemisphere is not traced — see `screen_probe_traces_texel`.
 */
[[nodiscard]] inline glm::vec3 screen_probe_texel_direction(int texel, std::uint32_t frame_index) {
    const int tx = texel % SCREEN_PROBE_OCT;
    const int ty = texel / SCREEN_PROBE_OCT;
    const glm::vec2 jitter = screen_probe_jitter(frame_index);
    const glm::vec2 oct = (glm::vec2(static_cast<float>(tx), static_cast<float>(ty)) + jitter) /
                          static_cast<float>(SCREEN_PROBE_OCT);
    return oct_decode(oct);
}

/**
 * Whether texel `texel` casts a ray at all this frame, given the anchor normal.
 *
 * Only the hemisphere above the anchor is traced. That is not a saving bolted
 * onto a full-sphere estimator — it is exact for the anchor's own normal:
 * projecting zero radiance over the lower hemisphere onto L1 and convolving
 * with the cosine lobe reproduces the hemisphere integral exactly (proved
 * numerically in screen_probe_math.test.cpp, `ShL1UniformHemisphereIsExact`).
 * A pixel whose normal is tilted away from the anchor's gets an answer that is
 * short of light from below its own horizon — which is precisely the case the
 * normal weight in the forward gather drives toward the world field.
 */
[[nodiscard]] inline bool screen_probe_traces_texel(const glm::vec3& direction,
                                                    const glm::vec3& anchor_normal) {
    return glm::dot(direction, anchor_normal) > 0.0f;
}

// ============================================================================
// Atlas addressing
// ============================================================================

/// Radiance atlas texel of (probe, octahedral texel). RGB = radiance,
/// A = hit distance, or a NEGATIVE alpha meaning "no sample resolved this
/// frame" (dropped miss record) — see gi_screen_probe_march.comp.hlsl.
[[nodiscard]] constexpr glm::ivec2 screen_probe_radiance_texel(glm::ivec2 probe, int texel) {
    return {probe.x * SCREEN_PROBE_OCT + (texel % SCREEN_PROBE_OCT),
            probe.y * SCREEN_PROBE_OCT + (texel / SCREEN_PROBE_OCT)};
}

[[nodiscard]] constexpr glm::ivec2 screen_probe_radiance_atlas_size(glm::ivec2 grid) {
    return {grid.x * SCREEN_PROBE_OCT, grid.y * SCREEN_PROBE_OCT};
}

/**
 * Miss records the buffer must hold for a grid.
 *
 * Worst case is every traced ray missing on screen, and only the hemisphere is
 * traced, so it is probes x RAYS/2. Sizing for the true worst case rather than
 * a guessed fraction is what lets the shader's capacity clamp be a diagnostic
 * (a counter that should read zero) instead of a tuning parameter: at 1600x900
 * the buffer is 5700 x 32 x 16 B = 2.85 MB, which is not worth a heuristic.
 */
[[nodiscard]] constexpr int screen_probe_miss_capacity(glm::ivec2 grid) {
    return screen_probe_count(grid) * (SCREEN_PROBE_RAYS / 2);
}

// ============================================================================
// The hierarchical-Z march
// ============================================================================

/// Mip the screen march starts at. 2 is a 4x4-pixel footprint: coarse enough
/// that a ray crossing an empty courtyard skips it in a handful of steps, fine
/// enough that a ray starting inside a doorway does not need three refinement
/// steps before it can move at all.
inline constexpr int SCREEN_TRACE_START_MIP = 2;

/// Hard step budget. A ray that has not resolved by here is handed to the BVH
/// exactly as one that left the screen is — the budget bounds the pass's cost
/// and the handoff makes exceeding it a correctness non-event.
inline constexpr int SCREEN_TRACE_MAX_STEPS = 48;

/**
 * Assumed thickness of a screen surface, in linear view depth.
 *
 * The depth buffer stores one distance per pixel and says nothing about how far
 * BEHIND it the surface extends, so a screen march has to assume. Too thin and
 * rays tunnel through walls into whatever the depth buffer holds behind them;
 * too thick and every surface casts a shadow volume to infinity and the gather
 * goes dark. The rule is a floor plus a fraction of the distance, because the
 * uncertainty is really about the depth buffer's own resolution: at 40 m a
 * pixel already spans several centimetres.
 *
 * 50 world units = 0.5 m at the playground's 100 units/metre.
 */
inline constexpr float SCREEN_TRACE_THICKNESS_MIN = 50.0f;
inline constexpr float SCREEN_TRACE_THICKNESS_REL = 0.05f;

[[nodiscard]] inline float screen_trace_thickness(float linear_depth) {
    return std::max(SCREEN_TRACE_THICKNESS_MIN, std::abs(linear_depth) * SCREEN_TRACE_THICKNESS_REL);
}

/**
 * How far a handed-off ray may travel through the BVH, in world units.
 *
 * 600 = the COARSEST cascade's probe spacing (`cascade_spacing(2)`), i.e. the
 * scale below which the world probe field cannot resolve anything at all. That
 * is the honest division of labour: the screen probes buy sharpness inside one
 * probe cell of the coarse field and nothing beyond it, so a ray that has
 * travelled a whole coarse cell without hitting has left the region where a
 * traced answer beats the field's, and the fallback takes over. It also bounds
 * the pass: a short ray descends a fraction of the TLAS a 4000-unit ray does.
 */
inline constexpr float SCREEN_PROBE_WORLD_RAY_CLAMP = 600.0f;

/**
 * Offset along the anchor's normal before a ray is cast, world units.
 *
 * 1 unit = 1 cm at the playground's scale. It exists so the ray's first screen
 * sample is not the anchor's own surface and so the handed-off BVH ray does not
 * self-intersect the triangle it started on — the same job `normalBias` does in
 * the probe field's trace, at the same magnitude, and deliberately the same
 * value so a surface does not leak in one path and not the other.
 */
inline constexpr float SCREEN_PROBE_NORMAL_BIAS = 1.0f;

/**
 * The level-`level` cell grid, in cells per unit UV.
 *
 * `screen_size / 2^level`, NOT `hzb_mip_size(screen_size, level)` — and the
 * difference is a real bug the CPU mirror caught before the GPU ever ran.
 *
 * The pyramid's footprints are NOT uniform: `render/hzb.hpp` extends the LAST
 * destination texel of each axis to the end of the source, so that an odd
 * dimension's trailing row is not silently dropped. A 72-row image reduces to 4
 * rows at mip 4 whose footprints are 16, 16, 16 and **24** source rows. Address
 * that mip with `floor(uv * 4)` and every uv above 0.667 asks the wrong texel —
 * which is exactly the case where the march sailed through a ledge that the
 * texel it should have read was reporting.
 *
 * Dividing by 2^level instead puts the boundaries where the reduction actually
 * put them (at source multiples of 2^level), and the resulting cell index is
 * `floor(uv * cells)`, clamped to `hzb_mip_size - 1` FOR THE LOOKUP ONLY. The
 * step function keeps the unclamped grid: a ray in the oversized last cell has
 * its next boundary beyond uv = 1, so it leaves the screen rather than stalling
 * against a boundary that is not there.
 */
[[nodiscard]] inline glm::vec2 screen_trace_cell_grid(glm::ivec2 screen_size, int level) {
    const float scale = static_cast<float>(1 << level);
    return {static_cast<float>(screen_size.x) / scale, static_cast<float>(screen_size.y) / scale};
}

/// The pyramid texel a UV addresses at `level` — see `screen_trace_cell_grid`
/// for why this is not `floor(uv * hzb_mip_size(...))`.
[[nodiscard]] inline glm::ivec2 screen_trace_cell(glm::vec2 uv, glm::vec2 cells,
                                                  glm::ivec2 mip_size) {
    const glm::ivec2 cell = glm::ivec2(glm::floor(uv * cells));
    return glm::clamp(cell, glm::ivec2(0), glm::max(mip_size - 1, glm::ivec2(0)));
}

/**
 * The march's step function: how far along a screen-space ray it is to the next
 * cell boundary of a `cell_count`-resolution grid.
 *
 * `p` is a position in [0,1] screen UV, `d` the ray's UV direction (NOT
 * normalized — it is the whole ray's UV extent, so the returned t is in the
 * ray's own parameterization). The returned t is the smallest positive value
 * for which `p + d*t` lies on a cell boundary; the caller adds an epsilon so
 * the next sample lands strictly inside the next cell.
 *
 * This is the one piece of the march that is pure arithmetic, and it is where a
 * hierarchical march goes wrong: an off-by-one on which boundary plane the ray
 * is heading for makes it either revisit its own cell forever (hang) or skip
 * one (tunnel). `screen_probe_math.test.cpp` checks it against a brute-force
 * scan over the ray.
 *
 * An axis with d == 0 never crosses a boundary and yields infinity for that
 * axis, so the other one decides; both zero yields infinity, which the caller
 * must treat as "this ray cannot advance" and abandon.
 */
[[nodiscard]] inline float screen_trace_cell_step(glm::vec2 p, glm::vec2 d, glm::vec2 cell_count) {
    const glm::vec2 cell = glm::floor(p * cell_count);
    // The boundary the ray is heading for on each axis: the far side of the
    // current cell when moving positively, the near side when moving negatively.
    const glm::vec2 plane =
        cell + glm::vec2(d.x >= 0.0f ? 1.0f : 0.0f, d.y >= 0.0f ? 1.0f : 0.0f);
    const glm::vec2 target = plane / cell_count;
    const float inf = std::numeric_limits<float>::infinity();
    const float tx = std::abs(d.x) > 1e-12f ? (target.x - p.x) / d.x : inf;
    const float ty = std::abs(d.y) > 1e-12f ? (target.y - p.y) / d.y : inf;
    // A ray sitting exactly on a boundary yields 0 on that axis; the caller's
    // epsilon is what carries it across, so clamping to zero here (rather than
    // rejecting it) keeps the step monotone.
    return std::min(std::max(tx, 0.0f), std::max(ty, 0.0f));
}

/**
 * Reversed-Z device depth -> positive linear view depth.
 *
 * `a` and `b` are `projection[2][2]` and `projection[3][2]` of the same
 * (jittered) projection the depth buffer was rendered with, which is what the
 * traced passes already pass around as `depth_a` / `depth_b`. Device depth 0 is
 * the far plane and returns a very large number rather than infinity, so a
 * caller that forgets to test for sky gets an obviously wrong distance instead
 * of a NaN that propagates.
 */
[[nodiscard]] inline float screen_trace_linear_depth(float device_depth, float a, float b) {
    // `abs(depthB / (depth + depthA))`, the engine's one linearization — the
    // same expression composite.frag, blur_ao.frag and rt_reflection_resolve
    // use, and it holds for standard and reversed Z alike as long as the pair
    // comes from the live projection.
    const float denom = device_depth + a;
    if (std::abs(denom) < 1e-9f) {
        return 1e9f;
    }
    return std::abs(b / denom);
}

// ============================================================================
// L1 spherical harmonics
// ============================================================================

/// The four real L0/L1 basis functions evaluated at `n`, in the order the atlas
/// stores them: (Y00, Y1-1, Y10, Y11).
[[nodiscard]] inline glm::vec4 sh_l1_basis(const glm::vec3& n) {
    constexpr float Y0 = 0.28209479177387814f; // 1/(2*sqrt(pi))
    constexpr float Y1 = 0.4886025119029199f;  // sqrt(3)/(2*sqrt(pi))
    return {Y0, Y1 * n.y, Y1 * n.z, Y1 * n.x};
}

/// The solid angle one octahedral texel of a full-sphere `SCREEN_PROBE_OCT`^2
/// map subtends, treated as uniform. The octahedral mapping is equal-area by
/// construction, which is exactly why it is the projection used here and not,
/// say, a lat-long grid whose poles would need a per-texel weight.
inline constexpr float SCREEN_PROBE_TEXEL_SOLID_ANGLE =
    4.0f * 3.14159265358979323846f / static_cast<float>(SCREEN_PROBE_RAYS);

/**
 * Accumulate one sample into an L1 projection.
 *
 * `coeffs[k]` is the RGB of basis function k. The caller drives the loop so the
 * GPU can do the same accumulation across a threadgroup.
 */
inline void sh_l1_accumulate(glm::vec3 coeffs[4], const glm::vec3& direction,
                             const glm::vec3& radiance, float solid_angle) {
    const glm::vec4 y = sh_l1_basis(direction);
    for (int k = 0; k < 4; ++k) {
        coeffs[k] += radiance * (y[k] * solid_angle);
    }
}

/**
 * Cosine-weighted MEAN INCOMING RADIANCE (irradiance / pi) at a surface facing
 * `n`, from an L1 projection of the incident radiance.
 *
 * THE UNIT MATTERS AND IS NOT NEGOTIABLE: it is the same convention
 * `gi_sample.hlsli` returns and the same one the IBL irradiance cube stores, so
 * the screen-probe answer substitutes for the world field's with no scale
 * factor anywhere. Getting it wrong is a uniform brightness error that looks
 * like a tuning problem.
 *
 * Ramamoorthi & Hanrahan's convolution: E(n) = sum_l A_l * L_lm * Y_lm(n) with
 * A_0 = pi and A_1 = 2*pi/3. Dividing through by pi leaves 1 and 2/3.
 *
 * Clamped at zero: an L1 fit to a one-sided distribution goes negative on the
 * far side of the lobe, and a negative irradiance is not a dimmer answer, it is
 * a hole.
 */
[[nodiscard]] inline glm::vec3 sh_l1_irradiance_over_pi(const glm::vec3 coeffs[4],
                                                        const glm::vec3& n) {
    const glm::vec4 y = sh_l1_basis(n);
    const glm::vec3 result = coeffs[0] * y[0] +
                             (coeffs[1] * y[1] + coeffs[2] * y[2] + coeffs[3] * y[3]) * (2.0f / 3.0f);
    return glm::max(result, glm::vec3(0.0f));
}

// ============================================================================
// Filtering
// ============================================================================

/// Temporal EMA weight given to THIS frame's sample. 0.1 is ~22 frames to 90%
/// of a step change; the probe atlas is re-jittered every frame, so this is
/// also the number of directions each texel has integrated.
inline constexpr float SCREEN_PROBE_TEMPORAL_ALPHA = 0.1f;

/**
 * Firefly clamp, as a MULTIPLE OF THE PROBE'S OWN MEAN luminance.
 *
 * Scale-free on purpose. A fixed radiance threshold would have to be re-tuned
 * for every exposure the day/night cycle passes through — at the playground's
 * ~10^3 luminance swing a threshold that tames a noon specular is an
 * indiscriminate ceiling at midnight. R5's exposure-aware clamp
 * (docs/rt_lighting_plan.md §5.6) is where this gets the adapted EV; until
 * then, dividing by the probe's own mean is the honest exposure-independent
 * approximation, and it costs one groupshared reduction the pass was already
 * doing.
 */
inline constexpr float SCREEN_PROBE_FIREFLY_MAX = 8.0f;

/// Relative camera-distance tolerance for accepting a probe (temporal history,
/// spatial neighbour, or forward interpolation). Relative because a 20 cm depth
/// difference is a different surface at 2 m and the same one at 60 m.
inline constexpr float SCREEN_PROBE_DEPTH_TOLERANCE = 0.05f;

/// Exponent on the normal agreement term. 8 in probe space (neighbours on the
/// same wall agree closely), 4 in the forward gather (a pixel's normal-mapped
/// normal legitimately differs from its tile anchor's geometric one).
inline constexpr float SCREEN_PROBE_NORMAL_POWER = 8.0f;
inline constexpr float SCREEN_PROBE_FORWARD_NORMAL_POWER = 4.0f;

/**
 * Live weight at which the screen probes are trusted completely in the forward
 * pass; below it the world field takes the remainder.
 *
 * The same shape and the same reason as `GI_SAMPLE_WEIGHT_SUM_FLOOR` one layer
 * down: a pixel whose four probes are all on other surfaces (a thin railing, a
 * silhouette against sky, geometry disoccluded this frame) must not have one
 * survivor's radiance stretched over it. It falls back to the probe field,
 * which is exactly the tier below.
 */
inline constexpr float SCREEN_PROBE_WEIGHT_FLOOR = 0.5f;

/// Bilateral weight between two probe anchors, shared by the spatial filter and
/// the forward gather (which differ only in the normal exponent).
[[nodiscard]] inline float screen_probe_bilateral_weight(float camera_distance_a,
                                                         float camera_distance_b,
                                                         const glm::vec3& normal_a,
                                                         const glm::vec3& normal_b,
                                                         float normal_power) {
    const float scale = std::max(SCREEN_PROBE_DEPTH_TOLERANCE * std::max(camera_distance_a, 1.0f), 1.0f);
    const float depth_term = std::exp(-std::abs(camera_distance_a - camera_distance_b) / scale);
    const float cosine = std::max(0.0f, glm::dot(normal_a, normal_b));
    return depth_term * std::pow(cosine, normal_power);
}

} // namespace bracken::render::gi
