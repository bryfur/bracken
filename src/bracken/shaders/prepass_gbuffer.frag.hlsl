/**
 * Depth-Prepass G-Buffer Fragment Shader (velocity + normal/roughness).
 *
 * The G-buffer variant of prepass_velocity.frag.hlsl, used only when
 * ray-traced reflections are on (render/rt_renderer.hpp). SV_Target0 is
 * byte-for-byte the same velocity write as the base shader -- keep the two in
 * sync -- and SV_Target1 adds what the reflection pass needs to decide whether
 * a pixel is worth tracing:
 *
 *   RGB = world normal, encoded to [0,1] for the UNORM target
 *   A   = linear roughness
 *
 * Roughness comes from a uniform rather than a texture: the gate only needs to
 * separate "smooth enough to mirror" from "use the IBL probe", and the
 * material factor decides that. Sampling the MR map here would cost a texture
 * fetch per pixel in the prepass to refine a threshold test.
 */

struct PSInput {
    float4 position : SV_Position;
    float4 currClip : TEXCOORD0;
    float4 prevClip : TEXCOORD1;
    float3 worldNormal : TEXCOORD2;
};

struct PSOutput {
    float2 velocity : SV_Target0;
    float4 normalRough : SV_Target1;
};

[[vk::binding(0, 3)]]
cbuffer PrepassGBufferUniforms {
    float roughness;
    float _pad0;
    float _pad1;
    float _pad2;
};

PSOutput PSMain(PSInput input) {
    PSOutput output;

    // Identical to prepass_velocity.frag.hlsl.
    if (input.currClip.w <= 0.0 || input.prevClip.w <= 0.0) {
        output.velocity = float2(0.0, 0.0);
    } else {
        float2 currNdc = input.currClip.xy / input.currClip.w;
        float2 prevNdc = input.prevClip.xy / input.prevClip.w;
        output.velocity = (currNdc - prevNdc) * float2(0.5, -0.5);
    }

    // Interpolation shortens the normal; renormalize before encoding.
    float3 n = normalize(input.worldNormal);
    output.normalRough = float4(n * 0.5 + 0.5, saturate(roughness));
    return output;
}
