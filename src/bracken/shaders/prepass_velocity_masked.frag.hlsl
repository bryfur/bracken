/**
 * Depth-Prepass Velocity Fragment Shader, alpha-MASK variant - SDL3 GPU API
 *
 * prepass_velocity.frag.hlsl plus an alpha test: MASK meshes (foliage,
 * vines) clip against their base-color alpha so their depth footprint
 * matches the forward draw texel-for-texel. Including them in the prepass
 * is what lets GTAO shade leaves (they both receive and cast screen AO —
 * excluded, a canopy reads as a flat blob) and gives them real velocity.
 *
 * The clip must reproduce the forward shaders' alpha-test inputs exactly
 * (same texture, same UVs, same cutoff) so the discard pattern is identical
 * and the forward GREATER_OR_EQUAL test never finds holes.
 */

[[vk::binding(0, 3)]]
cbuffer PrepassMaskedUniforms {
    float alphaCutoff;
    float3 _padding0;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]] Texture2D baseColorTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]] SamplerState baseColorSampler : register(s0, space2);

struct PSInput {
    float4 position : SV_Position;
    float4 currClip : TEXCOORD0; // unjittered current clip position
    float4 prevClip : TEXCOORD1; // unjittered previous clip position
    float2 uv : TEXCOORD2;
};

float2 PSMain(PSInput input) : SV_Target0 {
    // EXACT forward alpha test (model.frag.hlsl / model_instanced.frag.hlsl),
    // including the mip-scaled coverage boost — the two passes must discard
    // the same texels or the prepass depth under-covers minified foliage.
    float alpha = baseColorTexture.Sample(baseColorSampler, input.uv).a;
    float mip = baseColorTexture.CalculateLevelOfDetailUnclamped(baseColorSampler, input.uv);
    alpha *= 1.0 + max(mip, 0.0) * 0.35;
    clip(alpha - alphaCutoff);

    if (input.currClip.w <= 0.0 || input.prevClip.w <= 0.0) {
        return float2(0.0, 0.0);
    }
    float2 currNdc = input.currClip.xy / input.currClip.w;
    float2 prevNdc = input.prevClip.xy / input.prevClip.w;
    return (currNdc - prevNdc) * float2(0.5, -0.5);
}
