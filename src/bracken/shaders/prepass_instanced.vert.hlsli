// Compiled twice: once as prepass_instanced.vert.hlsl (velocity only) and once as
// prepass_instanced_gbuffer.vert.hlsl with BRACKEN_PREPASS_GBUFFER defined, which
// adds the world normal the reflection G-buffer needs. ONE body so the
// SV_Position math -- which the forward pass depth-tests EQUAL against and
// which must stay bit-identical -- cannot drift between the two variants.
/**
 * Depth-Prepass Vertex Shader (instanced models) - SDL3 GPU API
 *
 * Position-only instanced transform + motion-vector clip positions. Reads the
 * SAME per-frame InstanceData storage buffer as the forward pass
 * (model_instanced.vert.hlsl) — keep the struct in sync with both that shader
 * and gpu::InstanceData. The SV_Position math replicates the forward shader
 * operation-for-operation (both sides are `precise`) so the forward pass can
 * depth-test EQUAL.
 */

// Same vertex buffer as the forward pass (Vertex3D stride); position only.
struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
#ifdef BRACKEN_PREPASS_GBUFFER
    [[vk::location(1)]] float3 normal : NORMAL;
#endif
};

struct VSOutput {
    precise float4 position : SV_Position;
    float4 currClip : TEXCOORD0;
    float4 prevClip : TEXCOORD1;
#ifdef BRACKEN_PREPASS_GBUFFER
    float3 worldNormal : TEXCOORD2;
#endif
};

// Matches gpu::PrepassCameraUniforms (set 1, b0); layout shared with
// prepass_terrain.vert.hlsl.
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
    float4x4 prevModel; // previous-frame model matrix (== model for statics)
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

#ifdef BRACKEN_PREPASS_GBUFFER
    // Per-instance inverse-transpose, exactly as model_instanced.vert.hlsl.
    output.worldNormal = normalize(mul((float3x3)inst.normalMatrix, input.normal));
#endif
    return output;
}
