/**
 * Grass Vertex Shader - SDL3 GPU API (chunked, splat-aware)
 *
 * Design:
 *  - CPU frustum-culls grass chunks and uploads visible chunk origins
 *    into a storage buffer. Each chunk contains BLADES_PER_CHUNK^2 blades.
 *  - Vertex shader derives per-blade position from (chunk_origin, blade_idx)
 *    and applies a SMOOTH LOD fade over the last 30% of view distance.
 *  - LOD fade and the splatmap grass mask are evaluated BEFORE the heightmap
 *    fetch; fully collapsed blades early-out with a degenerate position and
 *    skip the texture reads entirely.
 *  - Blade orientation is a stable per-blade hash yaw (full 2π) — no
 *    camera-facing billboarding, so the meadow doesn't re-orient/re-light as
 *    the camera orbits. Blades are widened ~1.3x to stay readable edge-on.
 *
 * All blade placement/animation math lives in grass_blade.hlsli, shared with
 * the depth prepass (prepass_grass.vert.hlsl) so both passes compute
 * bit-identical world positions — SV_Position here is `precise` and the
 * forward pipeline depth-tests EQUAL against the prepass depth.
 */

struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 texCoord : TEXCOORD0;
    [[vk::location(3)]] float4 color : COLOR0;
    uint instanceID : SV_InstanceID;
};

struct VSOutput {
    precise float4 position : SV_Position;
    float3 normal : TEXCOORD0;
    float2 texCoord : TEXCOORD1;
    float3 worldPos : TEXCOORD2;
    float4 color : TEXCOORD3;
    float viewDepth : TEXCOORD4;
};

// Per-frame uniforms — must stay byte-identical to prepass_grass.vert.hlsl
// and render::GrassVertexUniforms (grass_renderer.hpp). The trailing
// no-jitter view-projection pair is only consumed by the prepass shader.
[[vk::binding(0, 1)]]
cbuffer GrassUniforms {
    float4x4 viewProjection;
    float3 cameraPos;          // world camera position (xz used)
    float time;
    float _pad2;
    float splatEnabled;        // 1 when a splatmap is bound at sampler slot 1
    float _pad0;
    float grassViewDistance;
    float2 windDirection;
    float windStrength;
    float chunkSize;           // world units per chunk side
    float bladeSpacing;        // world units between blades within a chunk
    uint bladesPerChunkSide;   // sqrt of blades per chunk
    uint bladesPerChunkSq;     // bladesPerChunkSide * bladesPerChunkSide
    float _pad1;
    float heightmapOriginX;
    float heightmapOriginZ;
    float heightmapWorldSize;
    float heightmapMinHeight;
    float heightmapMaxHeight;
    float3 cameraForward;      // world-space camera forward (view depth projection)
    float4 splatParams;        // xy = splat region origin (world XZ), zw = region size
    float4x4 currViewProjectionNoJitter; // prepass velocity only (unused here)
    float4x4 prevViewProjectionNoJitter; // prepass velocity only (unused here)
};

// Heightmap texture + sampler (vertex stage set 0 = samplers first).
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D heightmapTexture;
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState heightmapSampler;

// Terrain splatmap (R = grass weight), CLAMP sampler. Bound to the heightmap
// texture as a placeholder when splatEnabled == 0.
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D splatmapTexture;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState splatmapSampler;

// Per-visible-chunk origins (xz = world origin). Filled by CPU each frame.
// Within vertex set 0, storage buffers come after samplers (2 samplers -> binding 2).
[[vk::binding(2, 0)]]
StructuredBuffer<float4> ChunkOrigins;

// Shared blade placement/animation math (references the globals above).
#include "grass_blade.hlsli"

// Fully collapsed blade: emit a clipped degenerate without touching textures.
VSOutput deadBlade() {
    VSOutput output;
    output.position = float4(0.0, 0.0, 2.0, 1.0); // z/w > 1 -> clipped
    output.normal = float3(0.0, 1.0, 0.0);
    output.texCoord = float2(0.0, 0.0);
    output.worldPos = float3(0.0, 0.0, 0.0);
    output.color = float4(0.0, 0.0, 0.0, 0.0);
    output.viewDepth = 0.0;
    return output;
}

VSOutput VSMain(VSInput input) {
    BladeResult blade = computeBlade(input.position, input.instanceID);
    if (!blade.alive) {
        return deadBlade();
    }

    VSOutput output;
    float3 worldPos = blade.worldPos;

    // Base AO: darken blade base, lighten tip.
    float ao = lerp(0.45, 1.0, blade.heightT);
    float4 vertColor = input.color;
    vertColor.rgb *= ao;

    output.worldPos = worldPos;
    // Shading normal leans heavily toward up so lighting reads as a grounded
    // meadow surface instead of per-blade facets flipping with yaw.
    output.normal = normalize(lerp(blade.bladeFwd, float3(0.0, 1.0, 0.0), 0.7));
    output.texCoord = input.texCoord;
    output.color = vertColor;
    // View-space depth (-viewPos.z), matching the other mesh shaders — radial
    // distance would skew cluster Z-slice + cascade selection at screen edges.
    output.viewDepth = dot(worldPos - cameraPos, cameraForward);
    output.position = mul(viewProjection, float4(worldPos, 1.0));

    return output;
}
