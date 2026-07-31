#pragma once

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

// Cluster grid configuration. Must match clusters.hlsli.
inline constexpr uint32_t CLUSTER_DIM_X = 16;
inline constexpr uint32_t CLUSTER_DIM_Y = 9;
inline constexpr uint32_t CLUSTER_DIM_Z = 24;
inline constexpr uint32_t CLUSTER_COUNT = CLUSTER_DIM_X * CLUSTER_DIM_Y * CLUSTER_DIM_Z;

// v1 limits.
inline constexpr uint32_t MAX_LIGHTS = 256;
inline constexpr uint32_t MAX_LIGHTS_PER_CLUSTER = 128;

// Local (point/spot) shadow atlas layout. One D32 atlas tiled 8x8; a point
// light owns 6 consecutive tiles (one per cube face), a spot light owns 1.
// Point groups occupy tiles [g*6, g*6+6) for g < 8; spot slots occupy tiles
// [48, 64). Must match cluster_lighting.hlsli (LOCAL_SHADOW_MAX_SLOTS there).
inline constexpr int LOCAL_SHADOW_TILES_PER_ROW = 8;
inline constexpr int LOCAL_SHADOW_MAX_SLOTS = LOCAL_SHADOW_TILES_PER_ROW * LOCAL_SHADOW_TILES_PER_ROW;
inline constexpr int LOCAL_SHADOW_MAX_POINT_BUDGET = 8;  // 8 * 6 = 48 tiles
inline constexpr int LOCAL_SHADOW_MAX_SPOT_BUDGET = 16;  // tiles 48..63
inline constexpr int LOCAL_SHADOW_SPOT_TILE_BASE = LOCAL_SHADOW_MAX_POINT_BUDGET * 6;

} // namespace bracken::render::lighting
