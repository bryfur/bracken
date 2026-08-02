// Composite Fragment Shader.
// Input: linear HDR scene color + (optional) bloom / volumetric fog.
// Fog is half resolution and upsampled here with a depth-aware (joint
// bilateral) 4-tap filter to avoid silhouette halos.
// Screen-space AO is NOT applied here: it is consumed by the forward pass
// (ambient term only) — see gtao.frag.hlsl.
// Exposure: manual bias (`exposure`) times, when autoExposureEnabled, the
// exp2 of the adapted EV read from the exposure-state SSBO written by
// exposure_adapt.comp.
// Color grading: white balance (LMS gains) PRE-tonemap in linear light —
// it models the illuminant, so it must see scene-referred values; the
// ASC-CDL block (pow(c * slope + offset, power)) POST-tonemap on the
// display-referred [0,1] image, before the contrast/saturation trims. Both
// blocks are flag-gated so all-neutral parameters are pixel-identical to the
// ungraded path (matrix/pow round-trips are not bit-exact even at neutral).
// Output: gamma-encoded LDR to the swapchain (UNORM).
// All math runs in linear light; tonemapping (ACES fitted) is applied once,
// then gamma 2.2 is encoded explicitly because the swapchain is plain UNORM.

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer CompositeUniforms : register(b0, space3) {
    float bloomStrength;
    float volumetricFogEnabled;
    float exposure;
    int tonemapMode;
    float contrast;
    float saturation;
    // Joint bilateral upsample of the half-res fog texture:
    // lowResTexelSize = 1 / half-res target size; depthA/depthB linearize the
    // depth buffer (viewZ = |depthB / (depth + depthA)|, glm RH_ZO — holds for
    // standard and reversed-Z projections alike).
    float2 lowResTexelSize;
    float depthA;
    float depthB;
    float autoExposureEnabled;
    float whiteBalanceEnabled;
    float3 whiteBalance; // linear-RGB gains via LMS (render/color_grading.hpp)
    float cdlEnabled;
    float3 cdlSlope;
    float _pad0;
    float3 cdlOffset;
    float _pad1;
    float3 cdlPower;
    float _pad2;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D colorTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState colorSampler : register(s0, space2);

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D bloomTexture : register(t1, space2);
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerState bloomSampler : register(s1, space2);

[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
Texture2D fogTexture : register(t2, space2);
[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
SamplerState fogSampler : register(s2, space2);

[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
Texture2D<float> depthTexture : register(t3, space2);
[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
SamplerState depthSampler : register(s3, space2);

// Auto-exposure state written by exposure_adapt.comp:
// x = adapted EV, y = target EV, z = avg log2 luminance, w = valid.
// Only read when autoExposureEnabled — the slot binds a zeroed dummy buffer
// when the auto-exposure system is off/unavailable.
[[vk::binding(4, 2)]]
StructuredBuffer<float4> exposureState : register(t4, space2);

#include "lighting.hlsli"

float3 reinhard(float3 c) {
    return c / (c + 1.0);
}

// Linear sRGB <-> LMS (CIECAM02 transform pair), used by the white-balance
// block. The gains themselves come from render::white_balance_coefficients.
static const float3x3 LIN_TO_LMS = {
    3.90405e-1, 5.49941e-1, 8.92632e-3,
    7.08416e-2, 9.63172e-1, 1.35775e-3,
    2.31082e-2, 1.28021e-1, 9.36245e-1
};
static const float3x3 LMS_TO_LIN = {
    2.85847e+0, -1.62879e+0, -2.48910e-2,
    -2.10182e-1, 1.15820e+0, 3.24281e-4,
    -4.18120e-2, -1.18169e-1, 1.06867e+0
};

// Positive linear view depth from non-linear [0,1] depth (glm RH_ZO,
// standard or reversed-Z — depthA/depthB come from the live projection).
float linearizeDepth(float depth) {
    return abs(depthB / (depth + depthA));
}

// Joint bilateral upsample setup for the half-res fog texture: the 4 nearest
// half-res texel centers, weighted by bilinear weight x depth similarity to
// the full-res pixel. Plain bilinear mixes foreground and background across
// depth discontinuities, producing god-ray/fog bleeding around silhouettes.
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

    // Premultiplied fog blend (scene * transmittance + inscattered) stays linear.
    if (volumetricFogEnabled > 0.5) {
        float2 tapUV[4];
        float4 tapW;
        bilateralTaps(input.texCoord, tapUV, tapW);
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

    // Exposure scales linear HDR before tonemap: manual bias x adapted
    // auto-exposure (exp2 of the EV the adapt compute pass converged on).
    float exposureMul = exposure;
    if (autoExposureEnabled > 0.5) {
        exposureMul *= exp2(exposureState[0].x);
    }
    color *= exposureMul;

    // White balance in linear light (scene-referred, pre-tonemap).
    if (whiteBalanceEnabled > 0.5) {
        float3 lms = mul(LIN_TO_LMS, color) * whiteBalance;
        color = max(mul(LMS_TO_LIN, lms), 0.0);
    }

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

    // ASC-CDL grade on the display-referred image (slope/offset/power;
    // neutral = 1/0/1). Base clamped >= 0 for the pow.
    if (cdlEnabled > 0.5) {
        color = pow(max(color * cdlSlope + cdlOffset, 0.0), cdlPower);
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
