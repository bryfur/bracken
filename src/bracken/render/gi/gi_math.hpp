#pragma once

/**
 * Probe-field addressing math — the CPU source of truth for phase R1 of
 * `docs/rt_lighting_plan.md` (§5.2).
 *
 * Header-only, glm + `<std>` only, NO GPU headers: every function here is a
 * pure mapping that R1b's compute shaders will mirror in HLSL, and the only way
 * to keep the two honest is to be able to test this half with no device, no
 * window and no SDL. Each function names its future shader counterpart in its
 * comment; when that counterpart is written it must be a transliteration of the
 * body below, not a re-derivation.
 *
 * WHAT EXISTS AFTER R1a: the addressing, the scroll, and the invalidation
 * bookkeeping. Nothing traces a ray, nothing shades a probe, and every probe
 * therefore stays `PROBE_UNINITIALIZED` forever. `render/gi/probe_volume.hpp`
 * hangs GPU atlases off this and `scene::SceneRenderer` scrolls it; consumers
 * arrive in R1b/R1c.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
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

/// Per-probe state (`probe_state` texture, R8_UINT). Richer states — relocated,
/// classified-inactive, backface-heavy — arrive with R1b's classification pass;
/// these two are what the structure alone can say.
inline constexpr std::uint8_t PROBE_UNINITIALIZED = 0;
inline constexpr std::uint8_t PROBE_ACTIVE = 1;

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

/// Atlas width/height in texels for a given per-probe tile size.
[[nodiscard]] inline int atlas_width(int probe_texels) { return PROBES_PER_ATLAS_ROW * probe_texels; }
[[nodiscard]] inline int atlas_height(int probe_texels) { return ATLAS_ROWS * probe_texels; }

/**
 * Top-left texel of a probe's tile, BORDER INCLUDED. Add PROBE_BORDER_TEXELS to
 * reach the payload's first texel.
 *
 * Shader counterpart-to-be: `giProbeAtlasOrigin(int slot, int probeTexels)`.
 */
[[nodiscard]] inline glm::ivec2 probe_atlas_origin(int slot_index, int probe_texels) {
    const int col = slot_index % PROBES_PER_ATLAS_ROW;
    const int row = slot_index / PROBES_PER_ATLAS_ROW;
    return {col * probe_texels, row * probe_texels};
}

/// Absolute atlas texel of a payload texel `local` in [0, payload)^2.
[[nodiscard]] inline glm::ivec2 probe_payload_texel(int slot_index, int probe_texels,
                                                    const glm::ivec2& local) {
    return probe_atlas_origin(slot_index, probe_texels) + glm::ivec2(PROBE_BORDER_TEXELS) + local;
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

    [[nodiscard]] int cascade() const { return cascade_; }
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
};

} // namespace bracken::render::gi
