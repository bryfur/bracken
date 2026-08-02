// cluster_compute.hlsli - froxel light-list access for COMPUTE shaders.
//
// The fragment path reaches the cluster grid through cluster_lighting.hlsli,
// which also drags in the local-shadow atlas and a PBR accumulator. A compute
// pass that just wants "which lights touch this pixel's froxel, and who are
// they" needs neither, and cannot use those bindings anyway: SDL3 GPU lays
// compute resources out on entirely different sets (README.md, "SDL3 GPU
// resource binding order") — set 0 for read-only storage buffers, set 2 for
// uniforms, versus set 2/set 3 for fragment.
//
// So the BINDINGS differ and the ARITHMETIC does not: the froxel index math
// comes from cluster_index.hlsli and the light-record decode from
// cluster_light_load.hlsli, both shared with the fragment path, both included
// below. Nothing in this file recomputes anything either of them already does.
//
// Contract for the including COMPUTE shader — define BOTH before including:
//
//   CLUSTER_COMPUTE_SSBO_BINDING_BASE
//       set-0 slot of the first cluster storage buffer. Per the SDL3 GPU
//       layout that is (sampled texture count + read-only storage texture
//       count + preceding read-only storage buffer count). This file claims
//       FIVE consecutive slots from there:
//         +0 lightDataBuf        ByteAddressBuffer  packed light data
//         +1 clusterOffsetsBuf   uint2 per froxel   (offset, count)
//         +2 lightIndicesBuf     uint               concatenated header indices
//         +3 lightStableIdsBuf   uint               stable id, PACKED order
//         +4 lightIndexRemapBuf  uint               prev packed idx -> current
//       Bind them with one SDL_BindGPUComputeStorageBuffers call starting at
//       that base, in that order (ClusterGrid::light_data_buffer(),
//       cluster_offsets_buffer(), light_indices_buffer(),
//       light_stable_ids_buffer(), light_remap_buffer()).
//
//   CLUSTER_COMPUTE_PARAMS_BINDING
//       set-2 uniform slot for the ClusterParams cbuffer
//       (SDL_PushGPUComputeUniformData at that index, with
//       ClusterGrid::params()).

#ifndef CLUSTER_COMPUTE_HLSLI
#define CLUSTER_COMPUTE_HLSLI

#ifndef CLUSTER_COMPUTE_SSBO_BINDING_BASE
#error "Define CLUSTER_COMPUTE_SSBO_BINDING_BASE (= set-0 slot of the first cluster SSBO) before including cluster_compute.hlsli"
#endif
#ifndef CLUSTER_COMPUTE_PARAMS_BINDING
#error "Define CLUSTER_COMPUTE_PARAMS_BINDING (= set-2 uniform slot for ClusterParams) before including cluster_compute.hlsli"
#endif

#include "cluster_index.hlsli"

[[vk::binding(CLUSTER_COMPUTE_PARAMS_BINDING, 2)]]
cbuffer ClusterParamsCB { ClusterParams clusterParams; };

[[vk::binding(CLUSTER_COMPUTE_SSBO_BINDING_BASE + 0, 0)]]
ByteAddressBuffer lightDataBuf;

[[vk::binding(CLUSTER_COMPUTE_SSBO_BINDING_BASE + 1, 0)]]
StructuredBuffer<uint2> clusterOffsetsBuf;

[[vk::binding(CLUSTER_COMPUTE_SSBO_BINDING_BASE + 2, 0)]]
StructuredBuffer<uint> lightIndicesBuf;

[[vk::binding(CLUSTER_COMPUTE_SSBO_BINDING_BASE + 3, 0)]]
StructuredBuffer<uint> lightStableIdsBuf;

[[vk::binding(CLUSTER_COMPUTE_SSBO_BINDING_BASE + 4, 0)]]
StructuredBuffer<uint> lightIndexRemapBuf;

// The decode. Reads lightDataBuf + clusterParams by name, so it goes after the
// declarations above — same ordering rule as cluster_lighting.hlsli.
#include "cluster_light_load.hlsli"

// Mirrors render::lighting::LIGHT_REMAP_INVALID.
#define LIGHT_REMAP_INVALID 0xFFFFFFFFu

// ---------------------------------------------------------------------------
// Froxel access
// ---------------------------------------------------------------------------

// Total lights the frame binned. Zero means the CPU skipped binning entirely
// and every buffer below holds last frame's contents — check this first.
uint clusterTotalLights() { return clusterParams.gridDim.w; }

// Froxel of a pixel. `uv` is SV_Position-style (uv.y = 0 at the TOP of the
// screen); a compute shader with an integer pixel coordinate gets there with
// (pixel + 0.5) * clusterParams.screenSize.zw. `view_depth` is POSITIVE
// distance forward of the camera.
uint clusterFroxelIndex(float2 uv, float view_depth) {
    return clusterIndexFromScreen(uv, view_depth, clusterParams);
}

// How many lights the froxel holds, clamped to the CPU-enforced per-cluster
// cap for the same reason the forward loop clamps: a stale or partially
// uploaded offsets buffer must not be able to spin a loop 2^32 times.
uint clusterFroxelLightCount(uint froxel) {
    return min(clusterOffsetsBuf[froxel].y, clusterParams.maxPerCluster.x);
}

// The froxel's i-th light, as a PACKED (header) index — the index that
// loadLightHeader(), clusterLightStableId() and the remap all speak.
uint clusterFroxelLight(uint froxel, uint i) {
    return lightIndicesBuf[clusterOffsetsBuf[froxel].x + i];
}

// Persistent identity of a packed light (contracts/light.hpp). Never 0 for a
// light that made it into the frame.
uint clusterLightStableId(uint packed_index) {
    return lightStableIdsBuf[packed_index];
}

// PREVIOUS frame's packed index -> this frame's packed index, or
// LIGHT_REMAP_INVALID when that light is no longer in the frame. This is what
// makes a per-pixel reservoir that remembers "I sampled light 7 last frame"
// meaningful: light 7 may be light 3 today, or gone.
uint clusterRemapPrevLight(uint prev_packed_index) {
    if (prev_packed_index >= clusterParams.maxPerCluster.y) return LIGHT_REMAP_INVALID;
    return lightIndexRemapBuf[prev_packed_index];
}

#endif // CLUSTER_COMPUTE_HLSLI
