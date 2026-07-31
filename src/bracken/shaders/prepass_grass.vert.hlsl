/**
 * Depth-Prepass Vertex Shader (grass blades) - SDL3 GPU API
 *
 * Same blade placement/animation math as grass.vert.hlsl (shared via
 * grass_blade.hlsli, identical GrassUniforms cbuffer) so the blade world
 * positions are bit-identical and the forward grass pipeline can depth-test
 * EQUAL against the prepass depth. Adds motion-vector clip positions.
 *
 * Grass is treated as static for velocity (wind displacement is sub-pixel
 * per frame and handled by the TAA history clamp): both clip positions use
 * the CURRENT animated world position, so velocity is camera-only.
 */

struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
    uint instanceID : SV_InstanceID;
};

struct VSOutput {
    precise float4 position : SV_Position;
    float4 currClip : TEXCOORD0;
    float4 prevClip : TEXCOORD1;
};

// Must stay byte-identical to grass.vert.hlsl GrassUniforms and
// render::GrassVertexUniforms (grass_renderer.hpp).
[[vk::binding(0, 1)]]
cbuffer GrassUniforms {
    float4x4 viewProjection;   // jittered when TAA jitter is enabled
    float3 cameraPos;
    float time;
    float _pad2;
    float splatEnabled;
    float _pad0;
    float grassViewDistance;
    float2 windDirection;
    float windStrength;
    float chunkSize;
    float bladeSpacing;
    uint bladesPerChunkSide;
    uint bladesPerChunkSq;
    float _pad1;
    float heightmapOriginX;
    float heightmapOriginZ;
    float heightmapWorldSize;
    float heightmapMinHeight;
    float heightmapMaxHeight;
    float3 cameraForward;
    float4 splatParams;
    float4x4 currViewProjectionNoJitter;
    float4x4 prevViewProjectionNoJitter;
};

// Heightmap texture + sampler (vertex stage set 0 = samplers first).
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D heightmapTexture;
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState heightmapSampler;

// Terrain splatmap (R = grass weight), CLAMP sampler.
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D splatmapTexture;
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState splatmapSampler;

// Per-visible-chunk origins (xz = world origin). Filled by CPU each frame.
[[vk::binding(2, 0)]]
StructuredBuffer<float4> ChunkOrigins;

// Shared blade placement/animation math (references the globals above).
#include "grass_blade.hlsli"

VSOutput VSMain(VSInput input) {
    VSOutput output;

    BladeResult blade = computeBlade(input.position, input.instanceID);
    if (!blade.alive) {
        // Clipped degenerate — must match grass.vert.hlsl deadBlade().
        output.position = float4(0.0, 0.0, 2.0, 1.0);
        output.currClip = float4(0.0, 0.0, 0.0, 1.0);
        output.prevClip = float4(0.0, 0.0, 0.0, 1.0);
        return output;
    }

    float4 worldPos4 = float4(blade.worldPos, 1.0);
    output.position = mul(viewProjection, worldPos4);
    output.currClip = mul(currViewProjectionNoJitter, worldPos4);
    output.prevClip = mul(prevViewProjectionNoJitter, worldPos4);

    return output;
}
