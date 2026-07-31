/**
 * Depth-Prepass Vertex Shader (terrain) - SDL3 GPU API
 *
 * Terrain vertices are already in world space; position-only transform plus
 * motion-vector clip positions (terrain is static, so velocity is camera-only:
 * both clip positions use the same world position). The SV_Position math
 * replicates terrain.vert.hlsl operation-for-operation (both sides are
 * `precise`) so the forward pass can depth-test EQUAL.
 */

// Same vertex buffer as the forward pass (12-float terrain stride); position only.
struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
};

struct VSOutput {
    precise float4 position : SV_Position;
    float4 currClip : TEXCOORD0;
    float4 prevClip : TEXCOORD1;
};

// Matches gpu::PrepassCameraUniforms (set 1, b0); layout shared with
// prepass_instanced.vert.hlsl.
[[vk::binding(0, 1)]]
cbuffer PrepassCameraUniforms {
    float4x4 view;
    float4x4 projection;
    float4x4 currViewProjectionNoJitter;
    float4x4 prevViewProjectionNoJitter;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;

    // EXACT forward operation order (terrain.vert.hlsl).
    float4 viewPos = mul(view, float4(input.position, 1.0));
    output.position = mul(projection, viewPos);

    output.currClip = mul(currViewProjectionNoJitter, float4(input.position, 1.0));
    output.prevClip = mul(prevViewProjectionNoJitter, float4(input.position, 1.0));

    return output;
}
