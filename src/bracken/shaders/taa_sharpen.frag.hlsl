/**
 * TAA Sharpen Fragment Shader.
 *
 * Mild unsharp mask over the freshly resolved TAA history, written back to
 * the HDR offscreen scene color so bloom / volumetric fog / composite keep
 * reading their usual input. Always runs when TAA is active (strength 0 is a
 * plain copy) — the HISTORY stays unsharpened, so sharpening never compounds
 * frame over frame. Mirrors gpu::TaaSharpenUniforms (gpu_uniforms.hpp).
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer TaaSharpenUniforms : register(b0, space3) {
    float2 invScreenSize;
    float sharpenStrength; // 0 = pass-through; ~0.25 = subtle default
    float _pad0;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D resolvedTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState resolvedSampler : register(s0, space2); // nearest clamp

float4 PSMain(PSInput input) : SV_Target0 {
    float2 uv = input.texCoord;
    float3 center = resolvedTexture.SampleLevel(resolvedSampler, uv, 0).rgb;
    if (sharpenStrength <= 0.0) {
        return float4(center, 1.0);
    }

    float3 up = resolvedTexture.SampleLevel(resolvedSampler, uv - float2(0.0, invScreenSize.y), 0).rgb;
    float3 down = resolvedTexture.SampleLevel(resolvedSampler, uv + float2(0.0, invScreenSize.y), 0).rgb;
    float3 left = resolvedTexture.SampleLevel(resolvedSampler, uv - float2(invScreenSize.x, 0.0), 0).rgb;
    float3 right = resolvedTexture.SampleLevel(resolvedSampler, uv + float2(invScreenSize.x, 0.0), 0).rgb;

    // Unsharp mask against the 4-neighbor average; clamp at 0 so negative
    // undershoot never feeds the bloom threshold.
    float3 blurred = (up + down + left + right) * 0.25;
    return float4(max(center + (center - blurred) * sharpenStrength, 0.0), 1.0);
}
