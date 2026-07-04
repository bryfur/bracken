// Composite Fragment Shader.
// Input: linear HDR scene color + (optional) AO / bloom / volumetric fog.
// AO and fog are half resolution and upsampled here with a depth-aware
// (joint bilateral) 4-tap filter to avoid silhouette halos.
// Output: gamma-encoded LDR to the swapchain (UNORM).
// All math runs in linear light; tonemapping (ACES fitted) is applied once,
// then gamma 2.2 is encoded explicitly because the swapchain is plain UNORM.

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer CompositeUniforms : register(b0, space3) {
    float aoStrength;
    float bloomStrength;
    float volumetricFogEnabled;
    float _padding;
    float exposure;
    int tonemapMode;
    float contrast;
    float saturation;
    // Joint bilateral upsample of the half-res AO/fog textures:
    // lowResTexelSize = 1 / half-res target size; depthA/depthB linearize the
    // depth buffer (viewZ = depthB / (depth + depthA), glm RH_ZO).
    float2 lowResTexelSize;
    float depthA;
    float depthB;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D colorTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState colorSampler : register(s0, space2);

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D<float> aoTexture : register(t1, space2);
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerState aoSampler : register(s1, space2);

[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
Texture2D bloomTexture : register(t2, space2);
[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
SamplerState bloomSampler : register(s2, space2);

[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
Texture2D fogTexture : register(t3, space2);
[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
SamplerState fogSampler : register(s3, space2);

[[vk::combinedImageSampler]][[vk::binding(4, 2)]]
Texture2D<float> depthTexture : register(t4, space2);
[[vk::combinedImageSampler]][[vk::binding(4, 2)]]
SamplerState depthSampler : register(s4, space2);

#include "lighting.hlsli"

float3 reinhard(float3 c) {
    return c / (c + 1.0);
}

// Positive linear view depth from non-linear [0,1] depth (glm RH_ZO).
float linearizeDepth(float depth) {
    return abs(depthB / (depth + depthA));
}

// Joint bilateral upsample setup for the half-res AO/fog textures: the 4
// nearest half-res texel centers, weighted by bilinear weight x depth
// similarity to the full-res pixel. Plain bilinear mixes foreground and
// background across depth discontinuities, producing dark AO fringes and
// god-ray/fog bleeding around silhouettes. Weights are computed once and
// shared by the AO and fog fetches (both are the same half-res grid).
void bilateralTaps(float2 uv, out float2 tapUV[4], out float4 weights) {
    float2 pos = uv / lowResTexelSize - 0.5;
    float2 base = floor(pos);
    float2 f = pos - base;
    float2 uv00 = (base + 0.5) * lowResTexelSize;
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
    weights = (sum > 1e-4) ? (w / sum) : bilinear;
}

float4 PSMain(PSInput input) : SV_Target {
    float3 color = colorTexture.Sample(colorSampler, input.texCoord).rgb;

    // Depth-aware upsample weights, shared by the half-res AO and fog fetches.
    float2 tapUV[4];
    float4 tapW;
    bilateralTaps(input.texCoord, tapUV, tapW);

    // AO with sunlight protection. Physically, AO should only attenuate
    // ambient/indirect light, but there is no depth prepass yet so the forward
    // pass can't split ambient from direct — AO arrives here as a flat multiply
    // on the FINAL lit color, which visibly darkens sun-lit + specular pixels
    // (the classic SSAO-on-everything artifact). Approximate "ambient only" by
    // protecting bright pixels: dim/ambient-lit regions receive full AO, while
    // sun-lit pixels keep at least ~65% of their light.
    float ao = 0.0;
    [unroll]
    for (int aoTap = 0; aoTap < 4; ++aoTap) {
        ao += aoTexture.SampleLevel(aoSampler, tapUV[aoTap], 0) * tapW[aoTap];
    }
    float protect = saturate(luminance(color) * 0.6);
    float aoFinal = lerp(ao, 1.0, protect * 0.65);
    color *= lerp(1.0, aoFinal, aoStrength);

    // Premultiplied fog blend (scene * transmittance + inscattered) stays linear.
    if (volumetricFogEnabled > 0.5) {
        float4 vfog = float4(0.0, 0.0, 0.0, 0.0);
        [unroll]
        for (int fogTap = 0; fogTap < 4; ++fogTap) {
            vfog += fogTexture.SampleLevel(fogSampler, tapUV[fogTap], 0) * tapW[fogTap];
        }
        color = color * (1.0 - vfog.a) + vfog.rgb;
    }

    // Bloom is additive in linear HDR.
    float3 bloom = bloomTexture.Sample(bloomSampler, input.texCoord).rgb;
    color += bloom * bloomStrength;

    // Exposure scales linear HDR before tonemap.
    color *= exposure;

    // Tonemap linear HDR -> [0,1] LDR.
    if (tonemapMode == 0) {
        color = acesFitted(color);
    } else if (tonemapMode == 1) {
        color = acesNarkowicz(color);
    } else if (tonemapMode == 2) {
        color = reinhard(color);
    } else {
        color = saturate(color);
    }

    // Post-tonemap contrast around mid-grey (0.5).
    color = (color - 0.5) * contrast + 0.5;

    // Saturation: blend between luminance and color.
    float lum = luminance(color);
    color = lerp(float3(lum, lum, lum), color, saturation);

    color = saturate(color);

    // Gamma-encode for UNORM swapchain.
    color = linearToSrgb(color);

    // IGN dither (+-0.5 LSB of an 8-bit channel) AFTER the sRGB encode, so it
    // is a true half-LSB everywhere before quantization. Added in linear space
    // the encode slope (~12.92 near black, ~0.4 near white) would over-dither
    // darks ~13x and leave bright gradients still banded.
    color += (interleavedGradientNoise(input.position.xy) - 0.5) / 255.0;

    return float4(color, 1.0);
}
