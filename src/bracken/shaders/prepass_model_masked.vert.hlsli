// Compiled twice: once as prepass_model_masked.vert.hlsl (velocity only) and
// once as prepass_model_masked_gbuffer.vert.hlsl with BRACKEN_PREPASS_GBUFFER
// defined, which adds the world normal. ONE body so the SV_Position math --
// which the forward pass depth-tests EQUAL against and which must stay
// bit-identical -- cannot drift between the two variants. Same split, and the
// same reason for it, as prepass_model.vert.hlsli.
/**
 * Depth-Prepass Vertex Shader (static models, alpha-MASK variant).
 *
 * prepass_model.vert.hlsli plus the texcoord passthrough the masked fragment's
 * alpha test needs (which the G-buffer variant of the opaque path also carries,
 * for a different reason -- it samples metallicRoughness there).
 *
 * WHY THE G-BUFFER VARIANT EXISTS AT ALL. It did not until 2026-08-08:
 * prepass_gbuffer_masked.frag.hlsl wrote a hardcoded oct(+Y) and said so, on the
 * premise that "the reflection pass rejects these on roughness before the normal
 * is ever used". That premise was true when the G-buffer served only the
 * reflection pass and is false now -- the normal is read by rt_shadow,
 * rt_shadow_gradient, rt_local_shadow's `surfaceAt`, rt_local_shadow_gradient,
 * the DI spatial filter's normal bilateral weight, and screen-probe placement.
 * 27 of the playground's 52 .glb assets are alphaMode:MASK, i.e. all the
 * foliage, so a constant +Y was being fed to the reservoir's target function and
 * to local_di.hlsli's relief factor across most of the frame's silhouette area.
 */

// Same vertex buffer as the forward pass (Vertex3D stride).
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
#ifdef BRACKEN_PREPASS_GBUFFER
    // Same transform as the forward pass (model.vert.hlsl).
    output.worldNormal = normalize(mul((float3x3)normalMatrix, input.normal));
#endif
    return output;
}
