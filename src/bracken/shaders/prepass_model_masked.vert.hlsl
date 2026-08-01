/**
 * Depth-Prepass Vertex Shader (static models, alpha-MASK variant).
 *
 * prepass_model.vert.hlsl plus the texcoord passthrough the masked
 * fragment's alpha test needs. SV_Position math stays operation-for-
 * operation identical to model.vert.hlsl (both `precise`).
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

// Matches gpu::PrepassModelUniforms (set 1, b0).
[[vk::binding(0, 1)]]
cbuffer PrepassTransformUniforms {
    float4x4 model;
    float4x4 view;
    float4x4 projection;
    float4x4 prevModel;
    float4x4 currViewProjectionNoJitter;
    float4x4 prevViewProjectionNoJitter;
    uint paletteOffset;
    uint prevPaletteOffset;
    float2 _padding0;
    // Inverse-transpose of `model`, read only by the G-buffer variant. Kept in
    // this shared cbuffer (rather than a second binding) so all prepass model
    // pipelines push one identical uniform block.
    float4x4 normalMatrix;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;

    // EXACT forward operation order (model.vert.hlsl): model -> view -> projection.
    float4 worldPos = mul(model, float4(input.position, 1.0));
    float4 viewPos = mul(view, worldPos);
    output.position = mul(projection, viewPos);

    output.currClip = mul(currViewProjectionNoJitter, worldPos);
    float4 prevWorldPos = mul(prevModel, float4(input.position, 1.0));
    output.prevClip = mul(prevViewProjectionNoJitter, prevWorldPos);

    output.uv = input.texcoord;
    return output;
}
