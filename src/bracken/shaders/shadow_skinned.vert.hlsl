/**
 * Shadow Depth Vertex Shader - Skinned Models
 *
 * Transforms skinned vertices into light space for shadow map generation.
 * Reads the same tightly packed bone-palette storage buffer as
 * skinned_model.vert.hlsl (uploaded once per frame, shared by all passes).
 */

struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 texCoord : TEXCOORD0;
    [[vk::location(3)]] float4 color : COLOR0;
    [[vk::location(4)]] float4 tangent : TANGENT;
    [[vk::location(5)]] uint4 joints : BLENDINDICES;  // uint4 matches UBYTE4 vertex format
    [[vk::location(6)]] float4 weights : BLENDWEIGHT;
};

struct VSOutput {
    float4 position : SV_Position;
};

[[vk::binding(0, 1)]]
cbuffer ShadowTransformUniforms {
    float4x4 lightViewProjection;
    float4x4 model;
    uint paletteOffset;     // start of this draw's palette in bonePalettes, in matrices
    float3 _padding0;
};

// All skinned draws' bone palettes this frame, tightly packed (set 0 = vertex storage buffers)
[[vk::binding(0, 0)]]
StructuredBuffer<float4x4> bonePalettes;

VSOutput VSMain(VSInput input) {
    VSOutput output;

    // Apply skeletal animation
    float4x4 skinMatrix =
        input.weights.x * bonePalettes[paletteOffset + input.joints.x] +
        input.weights.y * bonePalettes[paletteOffset + input.joints.y] +
        input.weights.z * bonePalettes[paletteOffset + input.joints.z] +
        input.weights.w * bonePalettes[paletteOffset + input.joints.w];

    float4 localPos = mul(skinMatrix, float4(input.position, 1.0));
    float4 worldPos = mul(model, localPos);
    output.position = mul(lightViewProjection, worldPos);
    return output;
}
