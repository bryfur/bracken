/**
 * Terrain Vertex Shader - SDL3 GPU API
 *
 * Transforms terrain vertices; fog is computed analytically in the fragment
 * shader from fragPos/cameraPos, so no per-vertex fog distance is exported.
 */

// Vertex input - locations match create_terrain_pipeline() in pipeline_registry.cpp.
// texCoord is part of the pipeline vertex layout but unused (world-space
// sampling in the fragment shader); leaving it unconsumed is valid.
struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float2 texCoord : TEXCOORD0;
    [[vk::location(2)]] float4 color : COLOR0;
    [[vk::location(3)]] float3 normal : NORMAL;
};

// Vertex output / Fragment input.
// position is `precise`: the depth prepass (prepass_terrain.vert.hlsl)
// computes the same view->projection chain and the forward pipeline
// depth-tests EQUAL against the prepass depth — both sides must be
// bit-identical, so contraction on the position path is disabled in both.
struct VSOutput {
    precise float4 position : SV_Position;
    float4 vertexColor : TEXCOORD0;
    float3 fragPos : TEXCOORD1;
    float3 normal : TEXCOORD2;
    float viewDepth : TEXCOORD3;
};

// Uniform buffer - SDL3 GPU SPIR-V requires vertex uniforms in set 1
[[vk::binding(0, 1)]]
cbuffer TransformUniforms {
    float4x4 view;
    float4x4 projection;
    float3 cameraPos;
    float _padding0;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;

    output.fragPos = input.position;
    output.vertexColor = input.color;

    // Use vertex normal calculated from heightmap
    output.normal = normalize(input.normal);

    float4 viewPos = mul(view, float4(input.position, 1.0));
    output.viewDepth = -viewPos.z;
    output.position = mul(projection, viewPos);

    return output;
}
