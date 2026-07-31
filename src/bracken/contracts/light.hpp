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

#include <cstdint>
#include <glm/glm.hpp>

namespace bracken::render::lighting {

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
