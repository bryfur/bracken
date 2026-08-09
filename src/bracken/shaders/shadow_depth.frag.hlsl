/**
 * Shadow Depth Fragment Shader
 *
 * Supports alpha-tested shadows for transparent textures (leaves, etc.).
 * When hasTexture is set and texture alpha < 0.5, the pixel is discarded
 * so light passes through transparent parts.
 *
 * Known divergence from the forward/prepass alpha test: the cutoff here is a
 * literal 0.5, not the material's `alphaCutoff` (the pack foliage authors
 * 0.35), because ShadowFragUniforms has no channel to carry it. The cascade is
 * therefore slightly LESS permissive than the surfaces it shadows.
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer ShadowFragUniforms {
    int hasTexture;
    // Scalars (not a float array) so the declared block is 16 bytes and
    // matches gpu::ShadowFragUniforms — HLSL cbuffer arrays stride at 16 B.
    float _pad0;
    float _pad1;
    float _pad2;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D baseColorTexture;
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState baseColorSampler;

void PSMain(PSInput input) {
    if (hasTexture == 1) {
        float alpha = baseColorTexture.Sample(baseColorSampler, input.texCoord).a;
        // Same mip-coverage boost as model.frag.hlsl / prepass_gbuffer_masked:
        // box-filtered mips AVERAGE alpha, so a minified leaf card erodes out
        // of the far cascades while it still draws solid on screen — the
        // canopy's shadow perforates at exactly the distance its silhouette
        // does not.
        float mip = baseColorTexture.CalculateLevelOfDetailUnclamped(baseColorSampler, input.texCoord);
        alpha *= 1.0 + max(mip, 0.0) * 0.35;
        if (alpha < 0.5) discard;
    }
}
