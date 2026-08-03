#pragma once

// The dynamic-light contract: what a host hands the engine, and what the GPU
// reads back.
//
// These are the light *structs* only. The cluster-grid resolution and the
// shadow-atlas budgets that the renderer sizes its buffers from live in
// `render/lighting/light.hpp`, which includes this header — that file mixed a
// host-facing data contract with renderer-internal budgets, and the frame packet
// only ever needed the former. Splitting them is what stops `scene/` from
// depending on `render/lighting/`.
//
// Namespace is unchanged (`bracken::render::lighting`): moving a file here is an
// include-line rewrite, never a rename. See contracts/README.md.

#include "bracken/core/angle.hpp"

#include <cstdint>
#include <glm/glm.hpp>

namespace bracken::render::lighting {

// ---------------------------------------------------------------------------
// What a host hands the engine
//
// These are the types FrameBuilder::add_point_light / add_spot_light take. They
// exist because the GPU-layout PODs below are NOT a host-facing API and never
// were: they carry `_pad0`, `_pad1` and `shadow_slot`, all engine-written, and
// hosts were aggregate-initializing straight across them positionally. Adding
// a field to the GPU struct — or reordering two — silently changed the meaning
// of every positional initializer in every game using the engine, with no
// diagnostic anywhere.
//
// The descriptors carry only what a host actually decides, in a fixed order,
// and spot cones are stated as ANGLES: the engine takes the cosine. That
// deletes the `std::cos(glm::radians(18.0f))` every call site used to write
// (and the class of bug where one of the two cones got the radians and the
// other got the degrees).
// ---------------------------------------------------------------------------

struct PointLightDesc {
    glm::vec3 position{0.0f};
    float radius = 0.0f;
    /// LINEAR RGB chromaticity, not a packed core::Color — lighting is
    /// linear-space throughout and 8 bits per channel is not enough here.
    glm::vec3 color{1.0f};
    /// Radiometric: falls off with inverse square in world units.
    float intensity = 0.0f;
    /// Opt in to the per-frame local-shadow budget. The engine picks the
    /// winners and writes the atlas slot; a host never sees one.
    bool casts_shadows = false;
    /// PERSISTENT IDENTITY across frames. 0 = "engine assigns" (the default,
    /// and what every existing call site gets). See the identity contract
    /// below the GPU structs.
    uint32_t stable_id = 0;
    /// EMITTER SIZE in world units — the radius of the sphere the ray-traced
    /// local-shadow pass samples a point on, and therefore this light's
    /// penumbra-width dial. See the area-light contract below the GPU structs.
    /// 0 = use the engine default (`RtLocalShadowSettings::source_radius`).
    float source_radius = 0.0f;
};

struct SpotLightDesc {
    glm::vec3 position{0.0f};
    float radius = 0.0f;
    glm::vec3 direction{0.0f, -1.0f, 0.0f};
    /// Cone half-angles from the axis; inner is the full-brightness core,
    /// outer the cutoff. inner <= outer (an inverted pair is not rejected —
    /// it simply produces no falloff band).
    core::Degrees inner_angle{};
    core::Degrees outer_angle{};
    glm::vec3 color{1.0f};
    float intensity = 0.0f;
    bool casts_shadows = false;
    /// PERSISTENT IDENTITY across frames. 0 = "engine assigns". See below.
    uint32_t stable_id = 0;
    /// EMITTER SIZE in world units. 0 = engine default. See the area-light
    /// contract below the GPU structs.
    float source_radius = 0.0f;
};

// ---------------------------------------------------------------------------
// What the GPU reads back
// ---------------------------------------------------------------------------

// Std140-compatible POD layouts mirrored exactly in clusters.hlsli.
// Sizes are static_asserted in light_cluster.cpp.
//
// Local shadows: hosts opt a light into the per-frame shadow budget with
// `casts_shadows = true`; the engine's LocalShadowMap (local_shadows.hpp)
// selects up to the budgeted count each frame and writes `shadow_slot` (the
// light's base tile in the local-shadow atlas, -1 = unshadowed) into the
// GPU copy before upload. Hosts never set shadow_slot themselves.

struct PointLight {
    glm::vec3 position;
    float radius;
    glm::vec3 color;
    float intensity;
    // Host opt-in for the local-shadow budget. Not read by the GPU.
    bool casts_shadows = false;
    uint8_t _pad0[3] = {0, 0, 0};
    // Engine-written before upload: base atlas tile of the light's 6 cube
    // faces (tiles [slot, slot+6)), -1 = unshadowed.
    int32_t shadow_slot = -1;
    // Persistent identity, carried over from PointLightDesc::stable_id.
    // 0 on arrival = "engine assigns"; the engine resolves it during binning
    // and it is non-zero from then on. Occupies what used to be declared
    // padding — sizeof/offsets are unchanged and no shader reads these bytes.
    uint32_t stable_id = 0;
    // Emitter radius in world units, carried over from
    // PointLightDesc::source_radius (R6). 0 = "use the engine default", which
    // the traced pass resolves; the raster path never reads it. Occupies what
    // used to be declared padding — sizeof/offsets are unchanged.
    float source_radius = 0.0f;
};

struct SpotLight {
    glm::vec3 position;
    float radius;
    glm::vec3 direction;
    float inner_cos;
    glm::vec3 color;
    float outer_cos;
    float intensity;
    // Host opt-in for the local-shadow budget. Not read by the GPU.
    bool casts_shadows = false;
    uint8_t _pad0[3] = {0, 0, 0};
    // Engine-written before upload: the light's atlas tile, -1 = unshadowed.
    int32_t shadow_slot = -1;
    // Persistent identity — see PointLight::stable_id. Occupies what used to
    // be declared padding; sizeof/offsets unchanged.
    uint32_t stable_id = 0;
    // Emitter radius (R6) — see PointLight::source_radius.
    //
    // THIS ONE COST A STRIDE. PointLight had a declared `_pad1` left over and
    // took the field for free; SpotLight was exactly 64 bytes with nothing
    // spare, so it GREW to 80. Two shader literals had to move with it
    // (`loadSpotLight` in cluster_light_load.hlsli and the flat walk in
    // rt_hit_shade.hlsli); the CPU side is all sizeof-driven. The alternative
    // considered and rejected was packing a half float into the three bytes
    // after `casts_shadows`: it keeps the stride, and it makes the one field a
    // host authors in world units the one field stored at 11 bits of mantissa.
    float source_radius = 0.0f;
    float _pad1[3] = {0.0f, 0.0f, 0.0f};
};

// ---------------------------------------------------------------------------
// Persistent light identity (`stable_id`)
//
// The frame packet rebuilds its light arrays every frame, so a light's array
// index is NOT stable frame-to-frame: index 3 can be a different lantern
// tomorrow. Anything that reuses per-light state across frames — R2's
// froxel reservoirs above all — needs identity instead of position-in-array.
//
// Contract:
//
//   * A host may set `stable_id` to any non-zero value it likes. Non-zero IDs
//     pass through the engine UNTOUCHED and win over hashing. A host that
//     already has entity ids should use them; that is the exact case this
//     field exists for.
//   * `stable_id == 0` (the default, and what every pre-existing call site
//     gets) means "engine assigns". The engine derives one during binning
//     from a QUANTIZED CONTENT HASH of the light — type, position snapped to
//     `LIGHT_ID_POSITION_QUANTUM` world units, color snapped to
//     `LIGHT_ID_COLOR_QUANTUM`, intensity snapped to the nearest octave — so:
//       - a static torch keeps its id for as long as it exists;
//       - a light jittering/flickering below the quanta keeps its id, EXCEPT
//         when the jitter happens to straddle a quantum boundary (rounding is
//         to nearest, so the boundary is a real, if rare, edge — one frame of
//         reset history there, no worse than the frame a light appears);
//       - a light that TELEPORTS further than the quantum gets a NEW id, and
//         that is correct: temporal history for the old position would be
//         wrong, so it resets.
//   * Two lights whose quantized content collides inside one frame are
//     disambiguated deterministically, in EMISSION order, by re-hashing with
//     an occurrence counter. Two identical lanterns therefore get distinct
//     ids. Their two identities swap only if the host reorders the emission
//     of the identical lights — harmless by construction: the lights are
//     identical, so any state reused across the swap is the state that light
//     would have had anyway.
//   * Auto-assigned ids are never 0, and are bit-identical run to run for the
//     same emission sequence (no pointer-order iteration anywhere in the
//     assignment).
//
// The renderer keeps the prev-frame-index -> current-frame-index remap that
// consumers actually read (render/lighting/light_cluster.hpp). Identity
// history is renderer-side, main-thread tier: it never rides the packet.
// ---------------------------------------------------------------------------

/// Quanta for the auto-assigned identity hash. Position is world units;
/// color is per linear-RGB channel; intensity is snapped in log2 (octaves),
/// because a flickering torch animates intensity by a *ratio*, and an
/// absolute quantum there would re-id it every frame.
inline constexpr float LIGHT_ID_POSITION_QUANTUM = 5.0f;
inline constexpr float LIGHT_ID_COLOR_QUANTUM = 0.25f;
inline constexpr float LIGHT_ID_INTENSITY_OCTAVE = 1.0f;

// ---------------------------------------------------------------------------
// Area lights (`source_radius`)
//
// A point light is a mathematical fiction: nothing in the world is a point, and
// the reason shadows have soft edges is that emitters have SIZE. R2b's traced
// local shadows already sampled a sphere cap around each light, which is what
// gave them penumbras at all — but the cap's radius was one global constant for
// every light in the scene, so a candle and a shop window cast identically soft
// shadows.
//
// `source_radius` is that number, per light, in WORLD UNITS, and it is the ONLY
// authoring surface an area light needs here: the solid-angle sampling was
// already correct, this just tells it how wide the solid angle is.
//
//   * `0` (the default, and what every existing call site gets) means "use the
//     engine default", `RtLocalShadowSettings::source_radius`. It is not
//     "a point light": a true point light is a hard shadow, which no host has
//     ever been able to ask for here and which this field deliberately does not
//     add — a zero-radius emitter also makes the sampled cone degenerate.
//   * The value is the emitter's RADIUS, not its range. `radius` above is how
//     far the light reaches; this is how big the thing glowing is. At the
//     playground's 100 units/metre, 10 is a candle flame, 40 a lantern globe,
//     150 a shop window.
//   * It affects TRACED shadows only (`LocalRtShadowMode` Ratio or Full). The
//     rasterized atlas has no notion of emitter size and never will; a host
//     that authors radii and then turns tracing off sees hard tile shadows, as
//     it did before.
//   * Wider is not free in noise: a larger cap spreads the one traced sample
//     over more of the emitter, so a very large radius under a fast-moving
//     camera converges more slowly. That is the honest cost of a soft shadow at
//     one ray per pixel.
// ---------------------------------------------------------------------------

// 0 = point, 1 = spot. payload_index is the index into the corresponding
// per-type array uploaded to the GPU.
struct LightHeader {
    uint32_t type;
    uint32_t payload_index;
};

inline constexpr uint32_t LIGHT_TYPE_POINT = 0;
inline constexpr uint32_t LIGHT_TYPE_SPOT = 1;

} // namespace bracken::render::lighting
