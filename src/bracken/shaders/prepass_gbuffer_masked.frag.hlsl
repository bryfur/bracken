/**
 * Depth-Prepass G-Buffer Fragment Shader, alpha-MASK variant.
 *
 * prepass_velocity_masked.frag.hlsl (identical alpha test and velocity write
 * -- keep them in sync) plus the second G-buffer attachment, which every
 * pipeline in the pass must write once the target is attached.
 *
 * Roughness is a constant 1: MASK geometry is foliage, which is never a mirror
 * surface, so the reflection pass skips these pixels above any sane
 * RtReflectionSettings::max_roughness rather than tracing a ray to find out.
 * That is also why no normal is plumbed through the vertex stage here.
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
    float4 currClip : TEXCOORD0;
    float4 prevClip : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

struct PSOutput {
    float2 velocity : SV_Target0;
    float4 normalRough : SV_Target1;
};

PSOutput PSMain(PSInput input) {
    // EXACT forward alpha test — see prepass_velocity_masked.frag.hlsl.
    float alpha = baseColorTexture.Sample(baseColorSampler, input.uv).a;
    float mip = baseColorTexture.CalculateLevelOfDetailUnclamped(baseColorSampler, input.uv);
    alpha *= 1.0 + max(mip, 0.0) * 0.35;
    clip(alpha - alphaCutoff);

    PSOutput output;
    if (input.currClip.w <= 0.0 || input.prevClip.w <= 0.0) {
        output.velocity = float2(0.0, 0.0);
    } else {
        float2 currNdc = input.currClip.xy / input.currClip.w;
        float2 prevNdc = input.prevClip.xy / input.prevClip.w;
        output.velocity = (currNdc - prevNdc) * float2(0.5, -0.5);
    }
    // oct(+Y) = (0.75, 0.75); fully rough, non-metal — the reflection pass
    // rejects these on roughness before the normal is ever used.
    output.normalRough = float4(0.75, 0.75, 1.0, 0.0);
    return output;
}
