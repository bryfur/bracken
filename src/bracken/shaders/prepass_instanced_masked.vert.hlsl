/**
 * Depth-Prepass Vertex Shader (instanced models, alpha-MASK variant).
 *
 * prepass_instanced.vert.hlsl plus the texcoord passthrough the masked
 * fragment's alpha test needs. The SV_Position math stays operation-for-
 * operation identical to the forward pass (both `precise`) so forward MASK
 * draws land on exactly this depth.
 */

struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float2 texcoord : TEXCOORD0;
};

struct VSOutput {
    precise float4 position : SV_Position;
    float4 currClip : TEXCOORD0;
    float4 prevClip : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

// Matches gpu::PrepassCameraUniforms (set 1, b0).
[[vk::binding(0, 1)]]
cbuffer PrepassCameraUniforms {
    float4x4 view;
    float4x4 projection;
    float4x4 currViewProjectionNoJitter;
    float4x4 prevViewProjectionNoJitter;
};

// Per-instance data — must match model_instanced.vert.hlsl + gpu::InstanceData.
struct InstanceData {
    float4x4 model;
    float4x4 normalMatrix;
    float4 tint;
    float noFog;
    float _pad0;
    float _pad1;
    float _pad2;
    float4x4 prevModel;
};

[[vk::binding(0, 0)]]
StructuredBuffer<InstanceData> instances;

VSOutput VSMain(VSInput input, uint instanceID : SV_InstanceID) {
    VSOutput output;

    InstanceData inst = instances[instanceID];

    // EXACT forward operation order (model_instanced.vert.hlsl).
    float4 worldPos = mul(inst.model, float4(input.position, 1.0));
    float4 viewPos = mul(view, worldPos);
    output.position = mul(projection, viewPos);

    output.currClip = mul(currViewProjectionNoJitter, worldPos);
    float4 prevWorldPos = mul(inst.prevModel, float4(input.position, 1.0));
    output.prevClip = mul(prevViewProjectionNoJitter, prevWorldPos);

    output.uv = input.texcoord;
    return output;
}
