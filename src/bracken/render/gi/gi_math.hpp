#pragma once

/**
 * Probe-field addressing math — the CPU source of truth for phase R1 of
 * `docs/rt_lighting_plan.md` (§5.2).
 *
 * Header-only, glm + `<std>` only, NO GPU headers: every function here is
 * mirrored in HLSL by the GI shaders, and the only way to keep the two honest
 * is to be able to test this half with no device, no window and no SDL. Each
 * function names its shader counterpart in its comment, and that counterpart is
 * a transliteration of the body below, not a re-derivation.
 *
 * THREE LAYERS, all of them here:
 *   - ADDRESSING (R1a) — the torus wrap, the banded atlas tiling, octahedral
 *     encode/decode and the border wrap, world <-> probe mapping, the scroll
 *     and its invalidation bookkeeping. Mirrored by `shaders/gi_probe.hlsli`.
 *   - UPDATE (R1b) — spherical-Fibonacci ray directions with their per-frame
 *     rotation, the change-adaptive blend curve, the relocation and
 *     classification thresholds. Mirrored by the `gi_probe_*.comp.hlsl` passes.
 *   - SAMPLING (R1c) — the bias, cascade selection, the eight-probe weights,
 *     the Chebyshev test and the cascade blend, in the `SAMPLE_*` block and
 *     `sample_field()` below. Mirrored by `shaders/gi_sample.hlsli`, and the
 *     one layer whose failure modes (seams at cascade and cell boundaries) are
 *     continuity properties, so they are asserted numerically in the tests
 *     rather than eyeballed in a frame.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <vector>

namespace bracken::render::gi {

// ---------------------------------------------------------------------------
// Grid dimensions
// ---------------------------------------------------------------------------

/// Cascades in the clipmap. Each is the same probe COUNT at twice the spacing,
/// so cascade n covers 8x the volume of cascade n-1 at 1/8 the density.
inline constexpr int CASCADE_COUNT = 3;

/// Probes per cascade, per axis. Y is deliberately half of X/Z: this is a
/// ground-plane engine (Y is height), and a village is far wider than it is
/// tall — spending the same probe count vertically would put most of the field
/// in empty sky.
inline constexpr int GRID_X = 24;
inline constexpr int GRID_Y = 12;
inline constexpr int GRID_Z = 24;
inline constexpr int PROBES_PER_CASCADE = GRID_X * GRID_Y * GRID_Z; // 6912
inline constexpr int PROBES_TOTAL = PROBES_PER_CASCADE * CASCADE_COUNT;

/**
 * Cascade 0's probe spacing, in WORLD UNITS.
 *
 * 150 is 1.5 metres at the playground's `VILLAGE_SCALE` of 100 units/metre
 * (`examples/playground/village_layout.h`), which is the scale every asset in
 * this repo is authored at. The reasoning, stated because "150" is otherwise a
 * magic number: the playground village is roughly 4000 units (~40 m) across its
 * plateau, so cascade 0 at 24 x 150 = 3600 units covers essentially the whole
 * settlement horizontally and 12 x 150 = 1800 units (18 m) vertically, which
 * clears the tallest roofline. Cascades 1 and 2 (300 / 600) then reach 7200 and
 * 14400 units — past the outlying hills and out to the D4000 default draw
 * distance. Below ~1 m spacing a probe field stops being amortizable at our ray
 * budgets; above ~3 m it leaks through single-room interiors.
 */
inline constexpr float BASE_PROBE_SPACING = 150.0f;

/// Octahedral payload edge, in texels, per probe.
/// Irradiance is low-frequency, so 6x6 directions is plenty; visibility carries
/// the Chebyshev depth moments and needs the resolution (DDGI's own numbers).
inline constexpr int IRRADIANCE_PAYLOAD_TEXELS = 6;
inline constexpr int VISIBILITY_PAYLOAD_TEXELS = 14;

/// One texel of border on every side, so a bilinear tap at the payload edge
/// wraps around the octahedron instead of bleeding into the neighbouring probe.
/// The border ring is a copy of the opposite payload edge — written by R1b.
inline constexpr int PROBE_BORDER_TEXELS = 1;
inline constexpr int IRRADIANCE_PROBE_TEXELS = IRRADIANCE_PAYLOAD_TEXELS + 2 * PROBE_BORDER_TEXELS; // 8
inline constexpr int VISIBILITY_PROBE_TEXELS = VISIBILITY_PAYLOAD_TEXELS + 2 * PROBE_BORDER_TEXELS; // 16

/// Per-probe SCHEDULING state (`probe_state` texture, R8_UINT). Owned by the
/// CPU: `ProbeCascade` writes it as the scroll invalidates probes and as the
/// updater schedules them, and `ProbeVolume` uploads the whole 6912-byte plane
/// whenever it changed.
///
/// The GPU's CLASSIFICATION verdict deliberately does NOT live here — see
/// PROBE_CLASS_* below for why a second writer would lose its writes.
inline constexpr std::uint8_t PROBE_UNINITIALIZED = 0;
inline constexpr std::uint8_t PROBE_ACTIVE = 1;

/**
 * Per-probe CLASSIFICATION verdict, stored in the W channel of the per-probe
 * offset texture (`ProbeVolume::probe_offset`, RGBA16F: xyz = the relocation
 * offset in world units, w = one of these).
 *
 * WHY NOT IN `probe_state`. That texture has a CPU writer — every scroll
 * re-uploads all of it from `ProbeCascade::state()`, which is the only way an
 * entering probe can be made to read as uninitialized on the GPU. A GPU-written
 * verdict in the same texels would be silently erased by the next scroll, and
 * the erasure would look like "classification is flaky" rather than like a
 * lost-update bug. One texture, one writer.
 *
 * R1c'S SAMPLER READS ONLY THIS TEXTURE, and that is a deliberate narrowing of
 * what R1a anticipated. The forward shaders have three sampler slots to spend
 * on the field (see ATLAS_ROWS_ALL_CASCADES below for why three and not four),
 * so the "has this probe ever been filled" question has to be answerable from
 * the same texel as "is it inside geometry". It is, because
 * `ProbeVolume::init()` seeds the WHOLE offset plane with
 * PROBE_CLASS_INSIDE_GEOMETRY: a probe reads as "skip me" from allocation until
 * the classify pass — which runs only on probes the blend passes have just
 * filled — overwrites it with PROBE_CLASS_OK. That seed is a one-time CPU
 * upload before any GPU write, so it is not a second writer; it is the initial
 * value. It also removes the last path that could read undefined texture
 * memory, which the atlases' never-cleared policy otherwise leaves open once
 * something outside the update passes starts sampling them.
 *
 * The residual, stated rather than hidden: a probe whose slot was taken over by
 * a scrolled-in probe that the frame's ray budget could not reach still reads
 * PROBE_CLASS_OK while holding its predecessor's radiance. `select_updates`
 * gives entering probes the whole budget first and a single-plane scroll enters
 * 288-576 probes against a budget of 864-1728, so this needs a jump of several
 * probe spacings in one frame to happen at all, and it is corrected on the
 * cascade's next turn.
 *
 * Anything sampling the field must also apply the offset — a probe nudged 0.45
 * spacings out of a wall is not at its lattice position any more.
 */
inline constexpr float PROBE_CLASS_OK = 0.0f;
inline constexpr float PROBE_CLASS_INSIDE_GEOMETRY = 2.0f;

// ---------------------------------------------------------------------------
// THE 2D TILING — defined once, here
// ---------------------------------------------------------------------------
//
// A cascade's probes are laid out in a single 2D atlas as
//
//     column = x + GRID_X * y        (GRID_X * GRID_Y = 288 columns)
//     row    = z                     (GRID_Z = 24 rows)
//
// which is exactly the linear probe index split at PROBES_PER_ATLAS_ROW, so
// `probe_index()` and `probe_atlas_origin()` are the same number read two ways
// and there is no second mapping to keep in sync. The (x,y) pairing rather than
// (x,z) is deliberate: a row of the atlas is then a constant-z slab of the
// world, which is the axis a ground-plane scroll moves least often, so a
// scrolled row stays a contiguous run.
//
// Why 2D at all, when the field is a 3D grid: SDL3 GPU 3D textures cannot be
// bound as compute storage-write targets on every backend, and the octahedral
// payload is 2D per probe regardless — a 3D texture would only move the tiling
// problem, not remove it.

inline constexpr int PROBES_PER_ATLAS_ROW = GRID_X * GRID_Y; // 288
inline constexpr int ATLAS_ROWS = GRID_Z;                    // 24

// ---------------------------------------------------------------------------
// THE CASCADE BANDS — one texture for the whole field, not one per cascade
// ---------------------------------------------------------------------------
//
// Every atlas holds ALL THREE cascades, stacked vertically: cascade c occupies
// rows [c * ATLAS_ROWS, (c+1) * ATLAS_ROWS), i.e. its tiles start at
// texel row c * ATLAS_ROWS * probe_texels. Nothing else changes — within a band
// the tiling is exactly the (column = x + GRID_X*y, row = z) mapping above.
//
// WHY, and this is a consumption decision rather than a storage one (R1c): the
// forward shaders sample the field, and a per-cascade texture would cost them
// THREE sampler slots per atlas — nine for irradiance + visibility + offsets,
// against three for the banded form. Scene fragment stages are already at 12
// (model) / 10 (terrain) / 8 (grass) samplers and DXC's practical ceiling for
// this stage was measured at 17, so nine new slots does not fit and three does.
// The volumetric fog pass and R4's screen probes reuse the same three.
//
// The cost is that a compute pass writing one cascade now writes into a shared
// texture. That is harmless here because the GPU passes serialise on one queue
// and each dispatch touches only its own band's texels — but it is why the
// read-write bindings use `cycle = false` (see probe_volume.cpp): cycling would
// hand the pass a fresh allocation and discard the other two cascades.
inline constexpr int ATLAS_ROWS_ALL_CASCADES = ATLAS_ROWS * CASCADE_COUNT; // 72

// ---------------------------------------------------------------------------
// Probe indexing
// ---------------------------------------------------------------------------

/// Linear slot for an in-range grid coordinate. Shader counterpart-to-be:
/// `giProbeIndex(int3)` in `gi_probe.hlsli`.
[[nodiscard]] inline int probe_index(const glm::ivec3& c) {
    return c.x + GRID_X * (c.y + GRID_Y * c.z);
}

/// Inverse of probe_index(). Shader counterpart-to-be: `giProbeCoord(int)`.
[[nodiscard]] inline glm::ivec3 probe_coord(int index) {
    const int x = index % GRID_X;
    const int y = (index / GRID_X) % GRID_Y;
    const int z = index / (GRID_X * GRID_Y);
    return {x, y, z};
}

/// Euclidean modulo — C++ `%` is truncating, and every wrap below crosses zero.
[[nodiscard]] inline int wrap_axis(int v, int n) {
    const int m = v % n;
    return m < 0 ? m + n : m;
}

/**
 * TORUS ADDRESSING. Maps an ABSOLUTE probe coordinate (world lattice, may be
 * negative, unbounded) to the storage slot that holds it.
 *
 * This is what makes scrolling free: the slot is a pure function of the
 * absolute coordinate, so a probe that is still inside the window after the
 * cascade moves keeps the exact texels it had. Only probes that ENTER the
 * window land on a slot whose previous occupant left, and those are precisely
 * the ones invalidated.
 *
 * Shader counterpart-to-be: `giWrapCoord(int3)`.
 */
[[nodiscard]] inline glm::ivec3 wrap_coord(const glm::ivec3& absolute) {
    return {wrap_axis(absolute.x, GRID_X), wrap_axis(absolute.y, GRID_Y), wrap_axis(absolute.z, GRID_Z)};
}

/// Storage slot for an absolute coordinate: wrap, then index.
[[nodiscard]] inline int wrapped_probe_index(const glm::ivec3& absolute) {
    return probe_index(wrap_coord(absolute));
}

// ---------------------------------------------------------------------------
// Atlas mapping
// ---------------------------------------------------------------------------

/// Atlas width in texels for a given per-probe tile size.
[[nodiscard]] inline int atlas_width(int probe_texels) { return PROBES_PER_ATLAS_ROW * probe_texels; }
/// Height of ONE cascade's band.
[[nodiscard]] inline int atlas_band_height(int probe_texels) { return ATLAS_ROWS * probe_texels; }
/// Height of the whole texture: every cascade's band, stacked.
[[nodiscard]] inline int atlas_height(int probe_texels) {
    return ATLAS_ROWS_ALL_CASCADES * probe_texels;
}

/**
 * Top-left texel of a probe's tile, BORDER INCLUDED, in the banded atlas. Add
 * PROBE_BORDER_TEXELS to reach the payload's first texel.
 *
 * Shader counterpart: `giProbeAtlasOrigin(int slot, int probeTexels, int cascade)`.
 */
[[nodiscard]] inline glm::ivec2 probe_atlas_origin(int slot_index, int probe_texels, int cascade) {
    const int col = slot_index % PROBES_PER_ATLAS_ROW;
    const int row = slot_index / PROBES_PER_ATLAS_ROW + cascade * ATLAS_ROWS;
    return {col * probe_texels, row * probe_texels};
}

/// Absolute atlas texel of a payload texel `local` in [0, payload)^2.
[[nodiscard]] inline glm::ivec2 probe_payload_texel(int slot_index, int probe_texels, int cascade,
                                                    const glm::ivec2& local) {
    return probe_atlas_origin(slot_index, probe_texels, cascade) + glm::ivec2(PROBE_BORDER_TEXELS) +
           local;
}

/// The one-texel-per-probe planes (probe state, probe offset) use the same
/// banding with a tile size of 1. Shader counterpart:
/// `giProbeStateTexel(int slot, int cascade)`.
[[nodiscard]] inline glm::ivec2 probe_state_texel(int slot_index, int cascade) {
    return probe_atlas_origin(slot_index, 1, cascade);
}

// ---------------------------------------------------------------------------
// Cascade placement — world <-> probe coordinates
// ---------------------------------------------------------------------------

/// Probe spacing of cascade `c`, in world units. Doubles per level, so every
/// coarser lattice is a strict SUBSET of the finer one — a cascade boundary
/// therefore never lands between two lattices, which is what lets R1b blend
/// across it without a resampling step.
[[nodiscard]] inline float cascade_spacing(int cascade) {
    return BASE_PROBE_SPACING * static_cast<float>(1 << cascade);
}

/// World position of an absolute probe coordinate. The lattice is anchored at
/// the world origin, not at the camera — that anchoring is what makes the snap
/// below stable and the whole field reproducible from the camera alone.
/// Shader counterpart-to-be: `giProbeWorldPosition(int3, int cascade)`.
[[nodiscard]] inline glm::vec3 probe_world_position(const glm::ivec3& absolute, int cascade) {
    return glm::vec3(absolute) * cascade_spacing(cascade);
}

/// Continuous absolute probe coordinate of a world position (the value whose
/// floor/round is the surrounding lattice). Inverse of probe_world_position().
[[nodiscard]] inline glm::vec3 world_to_probe_space(const glm::vec3& world, int cascade) {
    return world / cascade_spacing(cascade);
}

/**
 * Where cascade `c` sits for a camera at `camera_position`: the absolute
 * coordinate of its (0,0,0) corner probe.
 *
 * SNAPPED to whole probe steps — that is the entire point. A cascade that
 * followed the camera continuously would move every probe every frame and
 * invalidate the whole field; snapping means the field is stationary in world
 * space and the window slides over it in integer steps.
 */
[[nodiscard]] inline glm::ivec3 cascade_base(const glm::vec3& camera_position, int cascade) {
    const glm::vec3 p = world_to_probe_space(camera_position, cascade);
    const glm::ivec3 centre{static_cast<int>(std::floor(p.x + 0.5f)),
                            static_cast<int>(std::floor(p.y + 0.5f)),
                            static_cast<int>(std::floor(p.z + 0.5f))};
    return centre - glm::ivec3(GRID_X / 2, GRID_Y / 2, GRID_Z / 2);
}

/// Is absolute coordinate `a` inside the window whose corner probe is `base`?
[[nodiscard]] inline bool window_contains(const glm::ivec3& base, const glm::ivec3& a) {
    const glm::ivec3 d = a - base;
    return d.x >= 0 && d.x < GRID_X && d.y >= 0 && d.y < GRID_Y && d.z >= 0 && d.z < GRID_Z;
}

/**
 * Visit every probe that is in the NEW window and was not in the OLD one, and
 * return how many there were.
 *
 * Defined as set difference rather than as "the entering planes", which is the
 * same set and is the reason there is no de-duplication step here: a diagonal
 * scroll enters an L-shaped slab whose arms overlap in a corner, and any
 * per-axis formulation has to remember not to count that corner twice. Testing
 * containment in the old window cannot double-count by construction, and a
 * jump larger than the grid degenerates correctly into "all of them" with no
 * special case. The cost is one pass over the window (6912 probes) on frames
 * where the cascade actually moved, and nothing at all on frames where it did
 * not.
 *
 * Iteration order is z-major, then y, then x — fixed, so the invalidation
 * SEQUENCE is deterministic for a given camera path.
 */
template <typename Fn>
std::uint32_t for_each_entering_probe(const glm::ivec3& old_base, const glm::ivec3& new_base, Fn&& fn) {
    if (old_base == new_base) {
        return 0;
    }
    std::uint32_t count = 0;
    for (int z = 0; z < GRID_Z; ++z) {
        for (int y = 0; y < GRID_Y; ++y) {
            for (int x = 0; x < GRID_X; ++x) {
                const glm::ivec3 a = new_base + glm::ivec3(x, y, z);
                if (window_contains(old_base, a)) {
                    continue;
                }
                fn(a);
                ++count;
            }
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// Octahedral mapping
// ---------------------------------------------------------------------------
//
// MUST MATCH the convention already in the tree: `octEncode` in
// `src/bracken/shaders/prepass_gbuffer.frag.hlsl` and its inverse `octDecode`
// in `src/bracken/shaders/rt_trace.hlsli`. Those two are the engine's existing
// unit-vector <-> [0,1]^2 mapping (Cigolle et al. 2014) and the probe payload
// reuses them verbatim, so a probe texel and a G-buffer normal speak the same
// language and R1b's shaders can `#include "rt_trace.hlsli"` rather than
// carrying a second copy.

/// CPU mirror of `octEncode` (`shaders/prepass_gbuffer.frag.hlsl`).
[[nodiscard]] inline glm::vec2 oct_encode(const glm::vec3& n_in) {
    const float l1 = std::abs(n_in.x) + std::abs(n_in.y) + std::abs(n_in.z);
    const glm::vec3 n = n_in / l1;
    glm::vec2 e{n.x, n.y};
    if (n.z < 0.0f) {
        e = glm::vec2(1.0f - std::abs(n.y), 1.0f - std::abs(n.x)) *
            glm::vec2(n.x >= 0.0f ? 1.0f : -1.0f, n.y >= 0.0f ? 1.0f : -1.0f);
    }
    return e * 0.5f + 0.5f;
}

/// CPU mirror of `octDecode` (`shaders/rt_trace.hlsli`).
[[nodiscard]] inline glm::vec3 oct_decode(const glm::vec2& e_in) {
    const glm::vec2 e = e_in * 2.0f - 1.0f;
    glm::vec3 n{e.x, e.y, 1.0f - std::abs(e.x) - std::abs(e.y)};
    const float t = std::max(0.0f, std::min(1.0f, -n.z));
    n.x += n.x >= 0.0f ? -t : t;
    n.y += n.y >= 0.0f ? -t : t;
    return glm::normalize(n);
}

/**
 * Direction a payload texel represents: texel centres over the full [0,1]^2
 * octahedron. `local` is in [0, payload)^2.
 *
 * Shader counterpart-to-be: `giTexelDirection(int2 local, int payload)` — the
 * probe update writes irradiance per this direction, and the sampler inverts it
 * with `oct_encode`.
 */
[[nodiscard]] inline glm::vec3 probe_texel_direction(const glm::ivec2& local, int payload_texels) {
    const glm::vec2 uv = (glm::vec2(local) + 0.5f) / static_cast<float>(payload_texels);
    return oct_decode(uv);
}

/**
 * BORDER WRAP. Given a texel of a probe's FULL tile (border included, so
 * `tile_local` is in [0, payload + 2)^2), return the payload texel it must
 * replicate — in payload coordinates, [0, payload)^2. Passing an interior texel
 * is a programming error and returns it unchanged (clamped).
 *
 * The octahedron's outer square edge folds onto itself: walking off the top
 * edge re-enters the top edge from the opposite side, mirrored. So the border
 * ring is not "clamp to edge" (which is what a naive copy does, and which makes
 * a bilinear tap at the payload boundary blend two directions that are 90 apart
 * on the sphere) — each border texel takes the payload texel diagonally opposite
 * across the relevant axis, and the four corners take the opposite corner.
 *
 * Shader counterpart: `giBorderSource(int2, int)` in `shaders/gi_probe.hlsli`.
 */
[[nodiscard]] inline glm::ivec2 probe_border_source(const glm::ivec2& tile_local, int payload_texels) {
    const int last = payload_texels - 1;
    const int tile = payload_texels + 2 * PROBE_BORDER_TEXELS;
    const bool left = tile_local.x < PROBE_BORDER_TEXELS;
    const bool right = tile_local.x >= tile - PROBE_BORDER_TEXELS;
    const bool top = tile_local.y < PROBE_BORDER_TEXELS;
    const bool bottom = tile_local.y >= tile - PROBE_BORDER_TEXELS;
    // Payload coordinate of this texel, valid only on the axes that are inside.
    const int px = std::clamp(tile_local.x - PROBE_BORDER_TEXELS, 0, last);
    const int py = std::clamp(tile_local.y - PROBE_BORDER_TEXELS, 0, last);

    if ((left || right) && (top || bottom)) {
        // Corner: the antipode of the octahedron's corner is the opposite corner.
        return {left ? 0 : last, top ? 0 : last};
    }
    if (top || bottom) {
        return {last - px, top ? 0 : last};
    }
    if (left || right) {
        return {left ? 0 : last, last - py};
    }
    return {px, py};
}

// ---------------------------------------------------------------------------
// Probe ray directions
// ---------------------------------------------------------------------------

/**
 * Spherical Fibonacci: `count` directions spread over the whole sphere with no
 * clustering at the poles and no preferred axis. `index` is in [0, count).
 *
 * The whole sphere, not a hemisphere: a probe has no normal, it integrates
 * incoming radiance from every direction, and the octahedral payload it feeds
 * covers 4pi.
 *
 * Shader counterpart: `giSphericalFibonacci(uint, uint)` in
 * `shaders/gi_probe.hlsli` — a transliteration of this body, float for float.
 */
[[nodiscard]] inline glm::vec3 spherical_fibonacci(int index, int count) {
    // Golden-ratio conjugate. The azimuth advances by this fraction of a turn
    // per sample, which is the most irrational rotation there is and therefore
    // the one that takes longest to revisit a direction.
    constexpr float GOLDEN_CONJUGATE = 0.6180339887498949f;
    const float n = static_cast<float>(count);
    const float i = static_cast<float>(index);
    const float phi = 6.283185307179586f * (i * GOLDEN_CONJUGATE - std::floor(i * GOLDEN_CONJUGATE));
    // Uniform in cos(theta) — that is what makes the set area-uniform rather
    // than angle-uniform (which would bunch at the poles).
    const float cos_theta = 1.0f - (2.0f * i + 1.0f) / n;
    const float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));
    return {std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, cos_theta};
}

/**
 * The per-frame rotation applied to the whole Fibonacci set.
 *
 * WHY: a fixed direction set traced every frame samples the same `count` points
 * of the sphere forever, so the estimate converges to a biased answer — a
 * 64-direction quadrature of the scene, not the scene. Rotating the set makes
 * consecutive updates of the same probe sample DIFFERENT directions, so the EMA
 * blend below integrates rather than averaging one fixed quadrature.
 *
 * DETERMINISTIC: derived from the frame counter alone through the R3 sequence
 * (the 3D generalization of R2 in `shaders/rt_sampling.hlsli`; Roberts 2018,
 * with 1/g, 1/g^2, 1/g^3 for the root of x^4 = x + 1), then turned into a
 * uniformly-distributed rotation by Shoemake's quaternion construction. Same
 * frame index in, same rotation out, on the CPU and in the shader — which is
 * what keeps a `--smoke` run reproducible.
 *
 * Shader counterpart: `giRayRotation(uint)` in `shaders/gi_probe.hlsli`.
 */
[[nodiscard]] inline glm::mat3 probe_ray_rotation(std::uint32_t frame_index) {
    constexpr float R3_A = 0.8191725133961644f; // 1/g
    constexpr float R3_B = 0.6710436067037893f; // 1/g^2
    constexpr float R3_C = 0.5497004779019702f; // 1/g^3
    const float f = static_cast<float>(frame_index);
    const auto frac = [](float v) { return v - std::floor(v); };
    const float u1 = frac(R3_A * f);
    const float u2 = frac(R3_B * f);
    const float u3 = frac(R3_C * f);

    // Shoemake (Graphics Gems III): three uniforms -> a uniformly distributed
    // unit quaternion. Cheaper and better distributed than three Euler angles,
    // which bunch rotations near the poles of whichever axis is applied last.
    const float s1 = std::sqrt(1.0f - u1);
    const float s2 = std::sqrt(u1);
    const float t1 = 6.283185307179586f * u2;
    const float t2 = 6.283185307179586f * u3;
    const float x = s1 * std::sin(t1);
    const float y = s1 * std::cos(t1);
    const float z = s2 * std::sin(t2);
    const float w = s2 * std::cos(t2);

    return {// column 0
            glm::vec3{1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + z * w), 2.0f * (x * z - y * w)},
            // column 1
            glm::vec3{2.0f * (x * y - z * w), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + x * w)},
            // column 2
            glm::vec3{2.0f * (x * z + y * w), 2.0f * (y * z - x * w), 1.0f - 2.0f * (x * x + y * y)}};
}

/// Direction of probe ray `index` of `count` on frame `frame_index`.
[[nodiscard]] inline glm::vec3 probe_ray_direction(int index, int count, std::uint32_t frame_index) {
    return probe_ray_rotation(frame_index) * spherical_fibonacci(index, count);
}

// ---------------------------------------------------------------------------
// Blend: change-adaptive hysteresis
// ---------------------------------------------------------------------------
//
// Per `docs/rt_lighting_plan.md` §5.2, the blend is an EMA whose hysteresis is
// driven by how far the new estimate is from the stored one (GI-1.0's
// Algorithm 3 shape). Two constants and one curve:

/// Steady-state hysteresis. High, because at 32-96 rays a probe's per-update
/// estimate is extremely noisy and the accumulated value IS the signal. A pure
/// 0.97 EMA reaches 90% of a step in ~76 updates — invisible at real
/// time-of-day rates, unacceptable for a lightning flash, which is what the
/// adaptive floor below exists for.
inline constexpr float PROBE_HYSTERESIS_MAX = 0.97f;
/// Floor reached when the new estimate disagrees with history by 100% or more
/// (a light switched on, a storm front, a door opened). 0.70 reaches 90% of a
/// step in ~7 updates.
inline constexpr float PROBE_HYSTERESIS_MIN = 0.70f;
/// Relative change at which the hysteresis has fully collapsed to the floor.
inline constexpr float PROBE_CHANGE_SATURATION = 1.0f;

/// Hysteresis for a normalized new-vs-old delta. `delta` is
/// |new - old| / max(|old|, epsilon), so it is scale-free: a courtyard at 0.02
/// and a sunlit wall at 20.0 respond to a doubling at the same rate.
///
/// `reset` is R5b's global hysteresis floor — the analytic sun gradient
/// (`render/rt_denoise_math.hpp`'s `lighting_reset_floor`), which the renderer
/// computes on the CPU before any probe ray is cast. It enters as an upper
/// BOUND on hysteresis rather than as a second term, so `reset == 0` is exactly
/// the pre-R5b curve and a static sun leaves the field bit-identical. Its whole
/// purpose is the probe that has not yet re-traced under the new sun and so
/// cannot observe a delta of its own to react to.
[[nodiscard]] inline float probe_blend_hysteresis(float delta, float reset = 0.0f) {
    const float t = std::clamp(delta / PROBE_CHANGE_SATURATION, 0.0f, 1.0f);
    const float adaptive = PROBE_HYSTERESIS_MAX + (PROBE_HYSTERESIS_MIN - PROBE_HYSTERESIS_MAX) * t;
    const float r = std::clamp(reset, 0.0f, 1.0f);
    const float ceiling = PROBE_HYSTERESIS_MAX + (PROBE_HYSTERESIS_MIN - PROBE_HYSTERESIS_MAX) * r;
    return std::min(adaptive, ceiling);
}

/// One EMA step of the adaptive blend, per colour channel set. Mirrors
/// `giBlend()` in `shaders/gi_probe.hlsli`.
[[nodiscard]] inline glm::vec3 probe_blend(const glm::vec3& old_value, const glm::vec3& new_value,
                                           float reset = 0.0f) {
    const float old_mag = glm::length(old_value);
    const float delta = glm::length(new_value - old_value) / std::max(old_mag, 1e-4f);
    const float hysteresis = probe_blend_hysteresis(delta, reset);
    return new_value + (old_value - new_value) * hysteresis;
}

// ---------------------------------------------------------------------------
// Classification / relocation thresholds
// ---------------------------------------------------------------------------

/// Backface fraction above which a probe is assumed to be sitting in a wall and
/// is nudged. A quarter of the sphere blocked from the inside is far more than
/// any convex exterior corner produces.
inline constexpr float PROBE_RELOCATE_BACKFACE_FRACTION = 0.25f;
/// Backface fraction above which relocation has failed and the probe is written
/// off as interior (PROBE_CLASS_INSIDE_GEOMETRY).
inline constexpr float PROBE_INSIDE_BACKFACE_FRACTION = 0.75f;

/// LEAVE thresholds, forming a dwell band with the two ENTER thresholds above.
///
/// Both verdicts are derived from a fraction of a 64-RAY BINOMIAL, whose
/// standard deviation at the 0.75 knee is sqrt(0.75*0.25/64) ~ 0.054. A bare
/// threshold on a statistic that noisy flips a probe's classification back and
/// forth on consecutive updates for no change in the world at all — and a probe
/// entering or leaving PROBE_CLASS_INSIDE_GEOMETRY is a whole member appearing
/// or disappearing from every sampler's 8-probe trilinear cage, which
/// `sample_cascade` then renormalizes over the survivors. One probe's verdict
/// therefore redistributes weight across the entire cell, which is why this
/// reads as flicker rather than as a single wrong probe.
///
/// The bands are ~2.8 sigma wide, so a probe genuinely sitting at the knee
/// settles instead of oscillating, while a probe that is really inside
/// geometry (fractions near 1) still crosses in one update. Hysteresis costs a
/// read of the verdict the classify pass already writes.
inline constexpr float PROBE_RELOCATE_BACKFACE_LEAVE = 0.10f;
inline constexpr float PROBE_INSIDE_BACKFACE_LEAVE = 0.60f;
/// Relocation offsets are clamped to this fraction of the cascade spacing.
/// Below 0.5 by construction: a probe that could move a full spacing would
/// cross into its neighbour's cell and the trilinear weights at sample time
/// would be nonsense.
inline constexpr float PROBE_MAX_OFFSET_FRACTION = 0.45f;

/// Ray hit distances are clamped to this multiple of the cascade spacing before
/// they enter the visibility moments. Chebyshev only ever compares a sample's
/// distance-to-probe against the stored mean, and a sample is at most ~1
/// spacing away from the probe that answers for it; letting one ray that flew
/// 4000 units into open sky drag the mean up would make the test declare every
/// nearby surface visible. 1.5 leaves headroom for the diagonal.
inline constexpr float PROBE_VISIBILITY_DISTANCE_SCALE = 1.5f;
/// Directional sharpness of the visibility blend's weighting lobe. Visibility
/// is a HIGH-frequency signal (a wall is either there or not), so its texels
/// take a much narrower cone of rays than the irradiance texels do — the
/// DDGI-standard trade.
inline constexpr float PROBE_VISIBILITY_SHARPNESS = 50.0f;

// ---------------------------------------------------------------------------
// SAMPLING — the CPU mirror of shaders/gi_sample.hlsli
// ---------------------------------------------------------------------------
//
// R1c's consumption path. Everything below is a transliteration partner of
// `giSampleDiffuse()` and its helpers; the shader does the texture taps, this
// half does the identical arithmetic over readings a caller supplies, so the
// cascade selection, the weights, the renormalisation and the cascade blend can
// all be tested with no device (`gi_math.test.cpp`).
//
// The quantity sampled is COSINE-WEIGHTED MEAN INCOMING RADIANCE (irradiance/pi)
// — the same convention as the IBL irradiance cube, so it substitutes for
// `ambientIBL`'s cube tap with no scale factor anywhere.

/// Sample-point bias, in fractions of the cascade's probe spacing.
///
/// DDGI's two-part bias, and both halves earn their keep. Along the NORMAL,
/// because a shading point sits exactly on a surface and the probes that see
/// that surface are all on one side of it — without the push, half the cage
/// answers for the wrong side and self-shadowing artefacts crawl along every
/// wall. Toward the CAMERA, because the error that survives the normal bias is
/// worst at grazing angles, where a surface fills many pixels and the probe
/// cage barely changes; pushing along the view ray decorrelates the two.
///
/// Both scale with the SPACING rather than being absolute: a coarse cascade's
/// probes are farther apart, so its acceptable bias is proportionally larger.
/// 0.2/0.3 are the RTXGI defaults, which is where they come from and why they
/// are not tuned here.
inline constexpr float SAMPLE_NORMAL_BIAS_FRACTION = 0.20f;
inline constexpr float SAMPLE_VIEW_BIAS_FRACTION = 0.30f;

/// How far inside a cascade's window the biased sample point must lie, in
/// probes, for that cascade to answer at all. Half a spacing: at exactly 0.5 the
/// floor of the local coordinate is 0 and the 8-tap cage is [0,1] on every
/// axis — wholly inside the window, which is the requirement. Any less and a
/// tap wraps around the torus and reads the opposite side of the world.
inline constexpr float SAMPLE_CASCADE_MARGIN_PROBES = 0.5f;
/// Width of the fade band at that boundary, in probes. THIS IS THE CONTINUITY
/// MECHANISM: a cascade's confidence ramps to zero over the last probe of its
/// window, so at the exact point where it stops answering it was already
/// contributing nothing and the next-coarser cascade already had all the
/// weight. Without it a cascade boundary is a visible step in the ambient term.
inline constexpr float SAMPLE_CASCADE_FADE_PROBES = 1.0f;

/// Live-weight fraction at which a cascade is trusted completely. The eight
/// trilinear weights sum to 1 when every probe is usable, so this is "at least
/// 30% of the cage answered"; below it the confidence ramps down and the
/// next-coarser cascade fills the rest in. It is what stops a cage that is
/// mostly buried probes from stretching one survivor's radiance over a room.
inline constexpr float SAMPLE_WEIGHT_SUM_FLOOR = 0.30f;

/// Floor added to the squared backface term so a cage that faces entirely away
/// from the shading normal ends up dim rather than black. Same value and same
/// reason as the bounce sampler in `gi_probe_trace.comp.hlsl`.
inline constexpr float SAMPLE_BACKFACE_WEIGHT_FLOOR = 0.02f;

/// Chebyshev variance floor, in units of spacing^2. The moments are accumulated
/// from a handful of rays per update, so a texel that has seen one consistent
/// distance reports a variance of ~0 and the bound collapses to a hard binary
/// test that flickers as the ray set rotates. Flooring it keeps the test soft.
///
/// No longer multiplied by spacing^2 at the use site: the stored moments are
/// themselves normalized by spacing (they overflowed fp16 otherwise — see
/// gi_probe_blend_visibility.comp.hlsl), so the whole test is scale-invariant
/// and this constant is applied directly. Same number, same meaning.
inline constexpr float SAMPLE_CHEBYSHEV_MIN_VARIANCE = 1e-4f;
/// Slack, in spacings, before the Chebyshev test starts attenuating at all. The
/// stored mean is itself a filtered estimate; without slack, a shading point at
/// the same distance as the surface the probe measured (i.e. its own surface)
/// tests as occluded and every wall self-darkens.
inline constexpr float SAMPLE_CHEBYSHEV_BIAS_FRACTION = 0.02f;

/**
 * Chebyshev's inequality as a soft visibility test between a probe and a
 * shading point `dist` away, given that probe's mean and mean-squared ray
 * distance in that direction. 1 = the probe can see the point.
 *
 * This is what makes a sparse grid usable indoors: without it, a probe on the
 * far side of a wall is trilinearly nearby and leaks its radiance through.
 *
 * Mirrors `giChebyshev()` in `shaders/gi_sample.hlsli`.
 */
[[nodiscard]] inline float chebyshev_visibility(float dist, float mean, float mean_squared,
                                                float spacing) {
    const float dist_n = dist / std::max(spacing, 1e-6f);
    const float bias = SAMPLE_CHEBYSHEV_BIAS_FRACTION;
    if (dist_n <= mean + bias) {
        return 1.0f;
    }
    const float floor_variance = SAMPLE_CHEBYSHEV_MIN_VARIANCE;
    const float variance = std::max(mean_squared - mean * mean, floor_variance);
    const float d = dist_n - (mean + bias);
    const float v = variance / (variance + d * d);
    // Cubed, per DDGI: the raw bound is far too permissive at the one-sample-
    // per-texel-per-update rates a probe field actually runs at, and light that
    // leaks 30% through a wall reads as a rendering bug rather than as softness.
    return std::max(v * v * v, 0.0f);
}

/// What one probe answers when the sampler asks. The shader reads these out of
/// the three atlases; the CPU mirror takes them from a caller-supplied functor
/// so the weight math can be exercised with no device.
struct ProbeReading {
    /// Relocation offset in world units (`probe_offset.xyz`). The probe's
    /// irradiance describes the world at lattice position + this.
    glm::vec3 offset{0.0f};
    /// `probe_offset.w` — PROBE_CLASS_INSIDE_GEOMETRY means "do not use me",
    /// and covers "never filled" as well (see PROBE_CLASS_* above).
    float classification = PROBE_CLASS_OK;
    /// The octahedral irradiance tap along the shading normal.
    glm::vec3 irradiance{0.0f};
    /// Visibility moments (mean, mean^2) toward the shading point, NORMALIZED BY
    /// CASCADE SPACING — the atlas is R16G16_FLOAT and raw squared world
    /// distances overflow it on cascades 1 and 2. See
    /// `gi_probe_blend_visibility.comp.hlsl`, which is what writes them.
    glm::vec2 moments{0.0f, 0.0f};
};

/// One cascade's answer: the renormalised irradiance and how much of the final
/// result it is entitled to, in [0, 1].
struct CascadeSample {
    glm::vec3 irradiance{0.0f};
    float confidence = 0.0f;
};

/// The biased sample position for a cascade. Cascade-dependent on purpose — the
/// bias scales with that cascade's spacing — which is why the containment test
/// below uses this and not the raw shading point.
[[nodiscard]] inline glm::vec3 sample_bias_position(const glm::vec3& world, const glm::vec3& normal,
                                                    const glm::vec3& to_camera, int cascade) {
    const float sp = cascade_spacing(cascade);
    return world + normal * (SAMPLE_NORMAL_BIAS_FRACTION * sp) +
           to_camera * (SAMPLE_VIEW_BIAS_FRACTION * sp);
}

/**
 * Trilinear-plus-weights gather over the eight probes surrounding `world` in
 * cascade `cascade`, whose window corner is at `base`.
 *
 * `read(cascade, absolute_coord, slot, irradiance_dir)` supplies the probe's
 * reading; the caller resolves the visibility direction itself from the offset
 * it returns, so the functor is called once per probe exactly as the shader
 * taps once per probe.
 *
 * Mirrors `giSampleCascade()` in `shaders/gi_sample.hlsli`.
 */
template <typename ReadFn>
[[nodiscard]] CascadeSample sample_cascade(int cascade, const glm::ivec3& base, const glm::vec3& world,
                                           const glm::vec3& normal, const glm::vec3& to_camera,
                                           ReadFn&& read) {
    const float sp = cascade_spacing(cascade);
    const glm::vec3 biased = sample_bias_position(world, normal, to_camera, cascade);
    const glm::vec3 local = biased / sp - glm::vec3(base);

    const glm::vec3 lo(SAMPLE_CASCADE_MARGIN_PROBES);
    const glm::vec3 hi(static_cast<float>(GRID_X) - 1.0f - SAMPLE_CASCADE_MARGIN_PROBES,
                       static_cast<float>(GRID_Y) - 1.0f - SAMPLE_CASCADE_MARGIN_PROBES,
                       static_cast<float>(GRID_Z) - 1.0f - SAMPLE_CASCADE_MARGIN_PROBES);
    float inset = std::numeric_limits<float>::max();
    for (int a = 0; a < 3; ++a) {
        inset = std::min(inset, std::min(local[a] - lo[a], hi[a] - local[a]));
    }
    if (inset <= 0.0f) {
        return {};
    }
    const float edge_fade = std::clamp(inset / SAMPLE_CASCADE_FADE_PROBES, 0.0f, 1.0f);

    const glm::vec3 base_f{std::floor(local.x), std::floor(local.y), std::floor(local.z)};
    const glm::vec3 frac = local - base_f;
    const glm::ivec3 b{static_cast<int>(base_f.x), static_cast<int>(base_f.y),
                       static_cast<int>(base_f.z)};

    glm::vec3 sum(0.0f);
    float weight_sum = 0.0f;
    for (int i = 0; i < 8; ++i) {
        const glm::ivec3 o{i & 1, (i >> 1) & 1, (i >> 2) & 1};
        const glm::ivec3 absolute = base + b + o;
        const int slot = wrapped_probe_index(absolute);
        const ProbeReading r = read(cascade, absolute, slot, normal);
        if (r.classification == PROBE_CLASS_INSIDE_GEOMETRY) {
            continue; // buried, or never filled — both leak if used
        }
        const glm::vec3 probe_pos = glm::vec3(absolute) * sp + r.offset;
        const glm::vec3 to_probe = probe_pos - biased;
        const float dist = glm::length(to_probe);
        const glm::vec3 dir = dist > 1e-4f ? to_probe / dist : normal;

        const glm::vec3 tri{o.x != 0 ? frac.x : 1.0f - frac.x, o.y != 0 ? frac.y : 1.0f - frac.y,
                            o.z != 0 ? frac.z : 1.0f - frac.z};
        const float wrapped = (glm::dot(dir, normal) + 1.0f) * 0.5f;
        const float w = tri.x * tri.y * tri.z * (wrapped * wrapped + SAMPLE_BACKFACE_WEIGHT_FLOOR) *
                        chebyshev_visibility(dist, r.moments.x, r.moments.y, sp);
        if (w <= 0.0f) {
            continue;
        }
        sum += w * r.irradiance;
        weight_sum += w;
    }
    if (weight_sum <= 1e-6f) {
        return {};
    }
    // RENORMALISE over the probes that actually answered. Dividing by the live
    // weight rather than by 8 (or by the full trilinear total) is what makes a
    // cage with two buried probes report the six survivors' average instead of
    // three quarters of it — the alternative reads as a dark halo around every
    // wall the relocation pass gave up on.
    CascadeSample out;
    out.irradiance = sum / weight_sum;
    out.confidence = edge_fade * std::clamp(weight_sum / SAMPLE_WEIGHT_SUM_FLOOR, 0.0f, 1.0f);
    return out;
}

/**
 * The whole sampler: finest cascade first, each taking as much of the remaining
 * weight as its confidence allows, with the hemisphere ambient absorbing
 * whatever is left over.
 *
 * The fallback is deliberately `EnvironmentState`'s hemisphere and not the IBL
 * irradiance cube even when IBL is on: it is the engine's GI-off tier, so a
 * pixel the field cannot answer for looks exactly as it did before this phase.
 *
 * Mirrors `giSampleDiffuse()` in `shaders/gi_sample.hlsli`.
 */
template <typename ReadFn>
[[nodiscard]] glm::vec3 sample_field(const std::array<glm::ivec3, CASCADE_COUNT>& bases,
                                     const glm::vec3& world, const glm::vec3& normal,
                                     const glm::vec3& to_camera, const glm::vec3& ambient_sky,
                                     const glm::vec3& ambient_ground, ReadFn&& read) {
    glm::vec3 accum(0.0f);
    float remaining = 1.0f;
    for (int c = 0; c < CASCADE_COUNT && remaining > 0.0f; ++c) {
        const CascadeSample s =
            sample_cascade(c, bases[static_cast<std::size_t>(c)], world, normal, to_camera, read);
        if (s.confidence <= 0.0f) {
            continue;
        }
        const float take = std::min(s.confidence, remaining);
        accum += s.irradiance * take;
        remaining -= take;
    }
    const glm::vec3 hemisphere =
        glm::mix(ambient_ground, ambient_sky, std::clamp(normal.y * 0.5f + 0.5f, 0.0f, 1.0f));
    return accum + hemisphere * remaining;
}

// ---------------------------------------------------------------------------
// One cascade's CPU-side state
// ---------------------------------------------------------------------------

/**
 * The scroll bookkeeping for a single cascade: where its window sits, which
 * slots hold a probe that has never been filled, and how many entered this
 * frame.
 *
 * Deliberately free of GPU types so the whole scroll can be tested headlessly
 * (`gi_math.test.cpp`) — `gi::ProbeVolume` owns one of these per cascade
 * alongside the atlases and does nothing with the geometry itself.
 */
/// One probe scheduled for update this frame. `was_uninitialized` is what tells
/// the blend pass to take the new estimate wholesale instead of blending it
/// into whatever the slot's previous occupant left behind.
struct ProbeUpdateSlot {
    int slot = 0;
    bool was_uninitialized = false;
};

class ProbeCascade {
public:
    /// (Re)bind to a cascade level. Leaves the cascade UNPLACED: the first
    /// update() both places it and reports every probe as invalidated, which is
    /// the honest answer — a freshly allocated atlas holds no probe data.
    void reset(int cascade_index) {
        cascade_ = cascade_index;
        placed_ = false;
        base_ = glm::ivec3(0);
        state_.assign(static_cast<std::size_t>(PROBES_PER_CASCADE), PROBE_UNINITIALIZED);
        state_dirty_ = true;
        last_invalidated_ = 0;
        cursor_ = 0;
    }

    /// Move the window to follow the camera. Returns the number of probes
    /// invalidated (0 on a frame where the snapped base did not move).
    std::uint32_t update(const glm::vec3& camera_position) {
        const glm::ivec3 new_base = cascade_base(camera_position, cascade_);
        if (!placed_) {
            placed_ = true;
            base_ = new_base;
            state_.assign(static_cast<std::size_t>(PROBES_PER_CASCADE), PROBE_UNINITIALIZED);
            state_dirty_ = true;
            last_invalidated_ = static_cast<std::uint32_t>(PROBES_PER_CASCADE);
            return last_invalidated_;
        }
        const std::uint32_t entered = for_each_entering_probe(
            base_, new_base, [&](const glm::ivec3& a) { state_[static_cast<std::size_t>(wrapped_probe_index(a))] = PROBE_UNINITIALIZED; });
        base_ = new_base;
        // Dirty on ANY entry, not on a byte diff. An entering probe must read
        // as uninitialized on the GPU regardless of what the previous occupant
        // of its slot left there, and once R1b writes PROBE_ACTIVE the two
        // stop agreeing — so "did probes enter" is the condition that stays
        // correct, and a byte-diff would be an optimization that silently
        // breaks then.
        if (entered > 0) {
            state_dirty_ = true;
        }
        last_invalidated_ = entered;
        return entered;
    }

    /**
     * Choose up to `budget` slots to trace this frame, appending them to `out`.
     * Returns how many were appended.
     *
     * TWO PHASES, and the order is the whole policy:
     *
     *  A. Every UNINITIALIZED slot, in slot order, jumps the queue. Those are
     *     the probes the scroll just brought into the window, holding the
     *     radiance of whatever used to occupy their storage — they are the
     *     visible-lag case, and nothing else in the field is wrong.
     *  B. The remainder of the budget comes from a persistent round-robin
     *     cursor over probes that are already live. The cursor advances past
     *     every slot EXAMINED, not every slot taken, so it cannot stall on a
     *     run of uninitialized slots.
     *
     * DETERMINISM: the output is a pure function of (state, cursor, budget), all
     * of which are main-thread state advanced once per frame. Two runs of the
     * same camera path update the same probes on the same frames — which is
     * what the smoke gate rests on.
     *
     * A slot never appears twice in one call: phase B only accepts slots that
     * are already PROBE_ACTIVE, and phase A's picks are promoted to ACTIVE only
     * after phase B has finished looking. Two threads writing one probe's texels
     * in the same dispatch would be a data race on the atlas, not a duplicated
     * cost.
     */
    std::uint32_t select_updates(std::uint32_t budget, std::vector<ProbeUpdateSlot>& out) {
        if (budget == 0 || state_.size() != static_cast<std::size_t>(PROBES_PER_CASCADE)) {
            return 0;
        }
        const std::size_t first = out.size();
        std::uint32_t taken = 0;
        for (int s = 0; s < PROBES_PER_CASCADE && taken < budget; ++s) {
            if (state_[static_cast<std::size_t>(s)] == PROBE_UNINITIALIZED) {
                out.push_back({s, true});
                ++taken;
            }
        }
        int examined = 0;
        while (taken < budget && examined < PROBES_PER_CASCADE) {
            const int s = (cursor_ + examined) % PROBES_PER_CASCADE;
            ++examined;
            if (state_[static_cast<std::size_t>(s)] == PROBE_ACTIVE) {
                out.push_back({s, false});
                ++taken;
            }
        }
        cursor_ = (cursor_ + examined) % PROBES_PER_CASCADE;
        // Now it is safe to promote: phase B has stopped reading the states.
        for (std::size_t i = first; i < out.size(); ++i) {
            if (out[i].was_uninitialized) {
                state_[static_cast<std::size_t>(out[i].slot)] = PROBE_ACTIVE;
                state_dirty_ = true;
            }
        }
        return taken;
    }

    [[nodiscard]] int cascade() const { return cascade_; }
    [[nodiscard]] int update_cursor() const { return cursor_; }
    [[nodiscard]] float spacing() const { return cascade_spacing(cascade_); }
    [[nodiscard]] bool placed() const { return placed_; }
    /// Absolute coordinate of the (0,0,0) corner probe.
    [[nodiscard]] const glm::ivec3& base() const { return base_; }
    /// Slot-indexed probe states, PROBES_PER_CASCADE bytes. Empty until reset().
    [[nodiscard]] const std::vector<std::uint8_t>& state() const { return state_; }
    [[nodiscard]] bool state_dirty() const { return state_dirty_; }
    void clear_state_dirty() { state_dirty_ = false; }
    [[nodiscard]] std::uint32_t last_invalidated() const { return last_invalidated_; }

    /// World position of the probe stored in `slot`, given the current base.
    /// Note this is NOT probe_world_position(probe_coord(slot)) — a slot holds
    /// whichever absolute coordinate currently wraps onto it.
    [[nodiscard]] glm::vec3 slot_world_position(int slot) const {
        const glm::ivec3 want = probe_coord(slot);
        const glm::ivec3 wrapped_base = wrap_coord(base_);
        glm::ivec3 a = base_;
        a.x += wrap_axis(want.x - wrapped_base.x, GRID_X);
        a.y += wrap_axis(want.y - wrapped_base.y, GRID_Y);
        a.z += wrap_axis(want.z - wrapped_base.z, GRID_Z);
        return probe_world_position(a, cascade_);
    }

private:
    int cascade_ = 0;
    bool placed_ = false;
    glm::ivec3 base_{0};
    std::vector<std::uint8_t> state_;
    bool state_dirty_ = false;
    std::uint32_t last_invalidated_ = 0;
    /// Round-robin position for select_updates(). Persistent across frames —
    /// that is what makes the rota cover the whole cascade instead of
    /// re-tracing the first `budget` probes forever.
    int cursor_ = 0;
};

} // namespace bracken::render::gi
