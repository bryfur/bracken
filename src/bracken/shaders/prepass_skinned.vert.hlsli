// Compiled twice: once as prepass_skinned.vert.hlsl (velocity only) and once as
// prepass_skinned_gbuffer.vert.hlsl with BRACKEN_PREPASS_GBUFFER defined, which
// adds the world normal the reflection G-buffer needs. ONE body so the
// SV_Position math -- which the forward pass depth-tests EQUAL against and
// which must stay bit-identical -- cannot drift between the two variants.
/**
 * Depth-Prepass Vertex Shader (skinned models) - SDL3 GPU API
 *
 * Skinned position-only transform + motion-vector clip positions. The
 * SV_Position math replicates skinned_model.vert.hlsl operation-for-operation
 * (both sides are `precise`) so the forward pass can depth-test EQUAL.
 *
 * Two palette buffers are bound: this frame's packed palettes (binding 0)
 * and the previous frame's (binding 1, double-buffered by SceneRenderer).
 * prevPaletteOffset == 0xFFFFFFFF means "no palette history": the previous
 * position reuses the current skin matrix (zero skin motion; object/camera
 * motion still applies via prevModel / prev view-projection).
 */

// Same vertex buffer as the forward pass (SkinnedVertex stride); only the
// position/joints/weights attributes are declared/fetched.
struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] uint4 joints : BLENDINDICES;
    [[vk::location(2)]] float4 weights : BLENDWEIGHT;
#ifdef BRACKEN_PREPASS_GBUFFER
    [[vk::location(3)]] float3 normal : NORMAL;
    // The G-buffer samples the metallicRoughness map, because in real content
    // the material's roughness factor carries almost no information (see
    // prepass_gbuffer.frag.hlsl).
    [[vk::location(4)]] float2 texcoord : TEXCOORD0;
#endif
};

struct VSOutput {
    precise float4 position : SV_Position;
    float4 currClip : TEXCOORD0;
    float4 prevClip : TEXCOORD1;
#ifdef BRACKEN_PREPASS_GBUFFER
    float3 worldNormal : TEXCOORD2;
    float2 uv : TEXCOORD3;
#endif
};

// Matches gpu::PrepassModelUniforms (set 1, b0); layout shared with
// prepass_model.vert.hlsl.
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

// This frame's packed bone palettes (identical to skinned_model.vert.hlsl).
[[vk::binding(0, 0)]]
StructuredBuffer<float4x4> bonePalettes;

// Previous frame's packed bone palettes (double-buffered).
[[vk::binding(1, 0)]]
StructuredBuffer<float4x4> prevBonePalettes;

VSOutput VSMain(VSInput input) {
    VSOutput output;

    // EXACT forward skinning + transform order (skinned_model.vert.hlsl).
    float4x4 skinMatrix =
        input.weights.x * bonePalettes[paletteOffset + input.joints.x] +
        input.weights.y * bonePalettes[paletteOffset + input.joints.y] +
        input.weights.z * bonePalettes[paletteOffset + input.joints.z] +
        input.weights.w * bonePalettes[paletteOffset + input.joints.w];

    float4 localPos = mul(skinMatrix, float4(input.position, 1.0));
    float4 worldPos = mul(model, localPos);
    float4 viewPos = mul(view, worldPos);
    output.position = mul(projection, viewPos);

    output.currClip = mul(currViewProjectionNoJitter, worldPos);

    // Previous position: previous palettes + previous model matrix. Without
    // palette history, reuse the current skin matrix (zero skin motion).
    float4x4 prevSkinMatrix = skinMatrix;
    if (prevPaletteOffset != 0xFFFFFFFFu) {
        prevSkinMatrix =
            input.weights.x * prevBonePalettes[prevPaletteOffset + input.joints.x] +
            input.weights.y * prevBonePalettes[prevPaletteOffset + input.joints.y] +
            input.weights.z * prevBonePalettes[prevPaletteOffset + input.joints.z] +
            input.weights.w * prevBonePalettes[prevPaletteOffset + input.joints.w];
    }
    float4 prevLocalPos = mul(prevSkinMatrix, float4(input.position, 1.0));
    float4 prevWorldPos = mul(prevModel, prevLocalPos);
    output.prevClip = mul(prevViewProjectionNoJitter, prevWorldPos);

#ifdef BRACKEN_PREPASS_GBUFFER
    // Skin the normal in local space, then take it to world with the same
    // inverse-transpose the forward pass uses.
    float3 localNormal = mul((float3x3)skinMatrix, input.normal);
    output.worldNormal = normalize(mul((float3x3)normalMatrix, localNormal));
    output.uv = input.texcoord;
#endif
    return output;
}
