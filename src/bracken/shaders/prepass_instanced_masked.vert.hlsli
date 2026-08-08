// Compiled twice: once as prepass_instanced_masked.vert.hlsl (velocity only) and
// once as prepass_instanced_masked_gbuffer.vert.hlsl with BRACKEN_PREPASS_GBUFFER
// defined, which adds the world normal. ONE body so the SV_Position math -- which
// forward MASK draws depth-test EQUAL against -- cannot drift between the two.
/**
 * Depth-Prepass Vertex Shader (instanced models, alpha-MASK variant).
 *
 * prepass_instanced.vert.hlsl plus the texcoord passthrough the masked
 * fragment's alpha test needs. The SV_Position math stays operation-for-
 * operation identical to the forward pass (both `precise`) so forward MASK
 * draws land on exactly this depth.
 *
 * See prepass_model_masked.vert.hlsli for why the G-buffer variant carries a
 * real normal rather than the constant oct(+Y) this path wrote until 2026-08-08.
 * It matters more here than anywhere: the instanced MASK path is the flora, and
 * the flora is most of this scene's silhouette.
 */

struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
#ifdef BRACKEN_PREPASS_GBUFFER
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 texcoord : TEXCOORD0;
#else
    [[vk::location(1)]] float2 texcoord : TEXCOORD0;
#endif
};

struct VSOutput {
    precise float4 position : SV_Position;
    float4 currClip : TEXCOORD0;
    float4 prevClip : TEXCOORD1;
    float2 uv : TEXCOORD2;
#ifdef BRACKEN_PREPASS_GBUFFER
    float3 worldNormal : TEXCOORD3;
#endif
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
#ifdef BRACKEN_PREPASS_GBUFFER
    // Same transform as the forward pass (model_instanced.vert.hlsl).
    output.worldNormal = normalize(mul((float3x3)inst.normalMatrix, input.normal));
#endif
    return output;
}
