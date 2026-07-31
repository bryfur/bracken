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
    float _pad1[2] = {0.0f, 0.0f};
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
    float _pad1 = 0.0f;
};

// 0 = point, 1 = spot. payload_index is the index into the corresponding
// per-type array uploaded to the GPU.
struct LightHeader {
    uint32_t type;
    uint32_t payload_index;
};

inline constexpr uint32_t LIGHT_TYPE_POINT = 0;
inline constexpr uint32_t LIGHT_TYPE_SPOT = 1;

} // namespace bracken::render::lighting
