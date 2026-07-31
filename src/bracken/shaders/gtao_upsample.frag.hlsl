/**
 * GTAO Upsample/Finalize Fragment Shader.
 *
 * Joint-bilateral (depth-aware) upsample of the working-res blurred AO to
 * the full-res R8 texture the forward pass samples in its ambient term.
 * Plain bilinear would mix foreground and background AO across depth
 * discontinuities (dark fringes around silhouettes); each of the 4 nearest
 * working-res taps is weighted by bilinear weight x linear-depth similarity
 * (same metric and sharpness as the bilateral AO blur / composite fog
 * upsample). When the working res equals the target res (AOMode GTAO_Full)
 * the filter degenerates to an exact passthrough.
 *
 * Also applies GraphicsSettings::ao_strength as pow(ao, strength): 0 = off
 * (AO forced to 1), 1 = stock, >1 = stronger darkening.
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer GtaoUpsampleUniforms : register(b0, space3) {
    float2 lowResTexelSize; // 1 / AO working resolution
    float depthA;           // viewZ = |depthB / (depth + depthA)| (glm RH_ZO)
    float depthB;
    float strength;
    float _pad0;
    float _pad1;
    float _pad2;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D<float> aoTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState aoSampler : register(s0, space2);

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D<float> depthTexture : register(t1, space2);
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerState depthSampler : register(s1, space2);

// Positive linear view depth from non-linear [0,1] depth.
float linearizeDepth(float depth) {
    return abs(depthB / (depth + depthA));
}

float PSMain(PSInput input) : SV_Target {
    float2 uv = input.texCoord;

    // The 4 nearest working-res texel centers around this full-res pixel.
    float2 pos = uv / lowResTexelSize - 0.5;
    float2 base = floor(pos);
    float2 f = pos - base;
    float2 uv00 = (base + 0.5) * lowResTexelSize;
    float2 tapUV[4];
    tapUV[0] = uv00;
    tapUV[1] = uv00 + float2(lowResTexelSize.x, 0.0);
    tapUV[2] = uv00 + float2(0.0, lowResTexelSize.y);
    tapUV[3] = uv00 + lowResTexelSize;

    float4 bilinear = float4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y),
                             (1.0 - f.x) * f.y, f.x * f.y);

    // Depth similarity on RELATIVE linear view-Z, same metric (and sharpness)
    // as the bilateral AO blur: a fixed percentage of the center depth counts
    // as "same surface" at any range.
    static const float DEPTH_SHARPNESS = 40.0;
    float centerZ = linearizeDepth(depthTexture.SampleLevel(depthSampler, uv, 0));
    float4 w;
    [unroll]
    for (int i = 0; i < 4; ++i) {
        float tapZ = linearizeDepth(depthTexture.SampleLevel(depthSampler, tapUV[i], 0));
        float relDiff = abs(tapZ - centerZ) / max(centerZ, 1e-3);
        w[i] = bilinear[i] * exp(-relDiff * DEPTH_SHARPNESS);
    }

    // Degenerate case (every tap rejected, e.g. subpixel silhouettes): fall
    // back to plain bilinear rather than dividing by ~0.
    float sum = w.x + w.y + w.z + w.w;
    w = (sum > 1e-4) ? (w / sum) : bilinear;

    float ao = 0.0;
    [unroll]
    for (int tap = 0; tap < 4; ++tap) {
        ao += aoTexture.SampleLevel(aoSampler, tapUV[tap], 0) * w[tap];
    }

    return saturate(pow(max(ao, 1e-4), strength));
}
