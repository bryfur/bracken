// Particle Fragment Shader.
// Paired with particle.vert (PSInput must byte-match its VSOutput). Renders
// into the linear HDR scene target with ADDITIVE (ONE, ONE) blending.
// Per-particle tint and opacity arrive pre-folded into the interpolated
// color (the instanced vertex stage scales rgb); this stage just samples the
// base color and applies the emissive boost. Alpha is written as 1.0 so
// SRC_ALPHA-based factors degenerate to ONE.

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
    float4 color : TEXCOORD1; // vertex color * per-particle tint * opacity
};

[[vk::binding(0, 3)]]
cbuffer ParticleFragmentUniforms : register(b0, space3) {
    float emissiveBoost;
    float _pad0;
    float _pad1;
    float _pad2;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D baseColorTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState baseColorSampler : register(s0, space2);

float4 PSMain(PSInput input) : SV_Target {
    // Bind a 1x1 white texture for untextured particles; the formula then
    // reduces to input.color.rgb * emissiveBoost.
    float4 baseColor = baseColorTexture.Sample(baseColorSampler, input.texCoord);
    float3 rgb = baseColor.rgb * input.color.rgb * emissiveBoost;
    return float4(rgb, 1.0);
}
