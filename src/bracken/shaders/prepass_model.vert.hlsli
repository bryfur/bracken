// Compiled twice: once as prepass_model.vert.hlsl (velocity only) and once as
// prepass_model_gbuffer.vert.hlsl with BRACKEN_PREPASS_GBUFFER defined, which
// adds the world normal the reflection G-buffer needs. ONE body so the
// SV_Position math -- which the forward pass depth-tests EQUAL against and
// which must stay bit-identical -- cannot drift between the two variants.
/**
 * Depth-Prepass Vertex Shader (static models) - SDL3 GPU API
 *
 * Position-only transform + motion-vector clip positions. The SV_Position
 * math replicates model.vert.hlsl operation-for-operation (both sides are
 * `precise`) so the forward pass can depth-test EQUAL against this depth.
 *
 * `projection` is the jittered projection when TAA jitter is on; the
 * *ViewProjectionNoJitter matrices feed velocity and never carry jitter.
 * Statics submit prevModel == model, so their velocity is camera-only.
 */

// Same vertex buffer as the forward pass (Vertex3D stride); only POSITION is
// declared/fetched.
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

// Matches gpu::PrepassModelUniforms (set 1, b0); layout shared with
// prepass_skinned.vert.hlsl (palette offsets unused here).
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

#ifdef BRACKEN_PREPASS_GBUFFER
    // Same transform as the forward pass (model.vert.hlsl).
    output.worldNormal = normalize(mul((float3x3)normalMatrix, input.normal));
#endif
    return output;
}
