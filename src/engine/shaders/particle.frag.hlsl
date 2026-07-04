// Particle Fragment Shader.
// Paired with model.vert (PSInput must byte-match its VSOutput — copied from
// model.frag.hlsl). Renders into the linear HDR scene target with ADDITIVE
// (ONE, ONE) blending: fade is encoded by scaling rgb (tint * emissiveBoost *
// opacity); the alpha output is unused by the blend. Alpha is written as 1.0
// so SRC_ALPHA-based factors degenerate to ONE.

struct PSInput {
    float4 position : SV_Position;
    float3 fragPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 texCoord : TEXCOORD2;
    float4 vertexColor : TEXCOORD3;
    float fogDistance : TEXCOORD4;
    float viewDepth : TEXCOORD5;
    float4 tangent : TEXCOORD6;  // xyz = world-space tangent, w = bitangent sign
};

[[vk::binding(0, 3)]]
cbuffer ParticleFragmentUniforms : register(b0, space3) {
    float4 tintColor;
    float emissiveBoost;
    float opacity;
    float _pad0;
    float _pad1;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D baseColorTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState baseColorSampler : register(s0, space2);

float4 PSMain(PSInput input) : SV_Target {
    // Bind a 1x1 white texture for untextured particles; the formula then
    // reduces to vertexColor.rgb * tintColor.rgb * emissiveBoost * opacity.
    float4 baseColor = baseColorTexture.Sample(baseColorSampler, input.texCoord);
    float3 rgb = baseColor.rgb * input.vertexColor.rgb * tintColor.rgb * emissiveBoost * opacity;
    return float4(rgb, 1.0);
}
