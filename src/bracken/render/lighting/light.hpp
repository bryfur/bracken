#pragma once

// Renderer-internal lighting budgets: the cluster grid the binner walks and the
// local-shadow atlas the selector allocates from. These size GPU buffers and
// must match the HLSL side; they are not part of the host-facing contract.
//
// The light structs themselves (PointLight / SpotLight / LightHeader /
// LIGHT_TYPE_*) are contracts/light.hpp, re-exported here so every existing
// `#include "bracken/render/lighting/light.hpp"` still sees them.

#include "bracken/contracts/light.hpp"

#include <cstdint>

namespace bracken::render::lighting {

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
