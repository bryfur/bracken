/**
 * FXAA 3.11 Fragment Shader (quality preset, ~12-15 taps)
 *
 * Operates on the post-tonemap, gamma-encoded LDR image (composite output in
 * the swapchain format) and writes the anti-aliased result to the swapchain.
 * Luma is computed in-shader from a green-weighted dot product — no alpha
 * pre-pass needed. Fullscreen triangle via fullscreen.vert.
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer FxaaUniforms : register(b0, space3) {
    float2 invScreenSize;
    float _pad0;
    float _pad1;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D ldrTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState ldrSampler : register(s0, space2);

// Green-weighted luma on the gamma-encoded image (perceptual, cheap).
float fxaaLuma(float3 rgb) {
    return dot(rgb, float3(0.299, 0.587, 0.114));
}

float sampleLuma(float2 uv) {
    return fxaaLuma(ldrTexture.SampleLevel(ldrSampler, uv, 0).rgb);
}

// Quality preset: contrast thresholds + edge-walk step ramp (FXAA 3.11 PC).
static const float EDGE_THRESHOLD = 0.166;     // min local contrast to bother
static const float EDGE_THRESHOLD_MIN = 0.0833; // skip in true darks
static const float SUBPIXEL_QUALITY = 0.75;
static const int ITERATIONS = 12;
static const float QUALITY_STEPS[12] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0};

float4 PSMain(PSInput input) : SV_Target {
    float2 uv = input.texCoord;
    float3 colorCenter = ldrTexture.SampleLevel(ldrSampler, uv, 0).rgb;

    // 4 direct neighbours + center.
    float lumaC = fxaaLuma(colorCenter);
    float lumaD = sampleLuma(uv + float2(0.0, invScreenSize.y));
    float lumaU = sampleLuma(uv - float2(0.0, invScreenSize.y));
    float lumaL = sampleLuma(uv - float2(invScreenSize.x, 0.0));
    float lumaR = sampleLuma(uv + float2(invScreenSize.x, 0.0));

    float lumaMin = min(lumaC, min(min(lumaD, lumaU), min(lumaL, lumaR)));
    float lumaMax = max(lumaC, max(max(lumaD, lumaU), max(lumaL, lumaR)));
    float lumaRange = lumaMax - lumaMin;

    // Early-out: not on an edge (low local contrast).
    if (lumaRange < max(EDGE_THRESHOLD_MIN, lumaMax * EDGE_THRESHOLD)) {
        return float4(colorCenter, 1.0);
    }

    // Corners.
    float lumaDL = sampleLuma(uv + float2(-invScreenSize.x, invScreenSize.y));
    float lumaUR = sampleLuma(uv + float2(invScreenSize.x, -invScreenSize.y));
    float lumaUL = sampleLuma(uv - invScreenSize);
    float lumaDR = sampleLuma(uv + invScreenSize);

    float lumaDU = lumaD + lumaU;
    float lumaLR = lumaL + lumaR;
    float lumaLeftCorners = lumaDL + lumaUL;
    float lumaDownCorners = lumaDL + lumaDR;
    float lumaRightCorners = lumaDR + lumaUR;
    float lumaUpCorners = lumaUR + lumaUL;

    // Horizontal vs vertical edge classification.
    float edgeHorizontal = abs(-2.0 * lumaL + lumaLeftCorners) + abs(-2.0 * lumaC + lumaDU) * 2.0 +
                           abs(-2.0 * lumaR + lumaRightCorners);
    float edgeVertical = abs(-2.0 * lumaU + lumaUpCorners) + abs(-2.0 * lumaC + lumaLR) * 2.0 +
                         abs(-2.0 * lumaD + lumaDownCorners);
    bool isHorizontal = (edgeHorizontal >= edgeVertical);

    // Pick the edge orientation (which neighbour pair straddles the edge).
    float luma1 = isHorizontal ? lumaU : lumaL;
    float luma2 = isHorizontal ? lumaD : lumaR;
    float gradient1 = luma1 - lumaC;
    float gradient2 = luma2 - lumaC;
    bool is1Steepest = abs(gradient1) >= abs(gradient2);
    float gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));

    float stepLength = isHorizontal ? invScreenSize.y : invScreenSize.x;
    float lumaLocalAverage = 0.0;
    if (is1Steepest) {
        stepLength = -stepLength;
        lumaLocalAverage = 0.5 * (luma1 + lumaC);
    } else {
        lumaLocalAverage = 0.5 * (luma2 + lumaC);
    }

    // Shift half a pixel to the edge.
    float2 currentUv = uv;
    if (isHorizontal) {
        currentUv.y += stepLength * 0.5;
    } else {
        currentUv.x += stepLength * 0.5;
    }

    // Walk along the edge in both directions until the luma average breaks.
    float2 offset = isHorizontal ? float2(invScreenSize.x, 0.0) : float2(0.0, invScreenSize.y);
    float2 uv1 = currentUv - offset;
    float2 uv2 = currentUv + offset;

    float lumaEnd1 = sampleLuma(uv1) - lumaLocalAverage;
    float lumaEnd2 = sampleLuma(uv2) - lumaLocalAverage;
    bool reached1 = abs(lumaEnd1) >= gradientScaled;
    bool reached2 = abs(lumaEnd2) >= gradientScaled;
    bool reachedBoth = reached1 && reached2;

    if (!reached1) uv1 -= offset;
    if (!reached2) uv2 += offset;

    if (!reachedBoth) {
        for (int i = 2; i < ITERATIONS; ++i) {
            if (!reached1) {
                lumaEnd1 = sampleLuma(uv1) - lumaLocalAverage;
            }
            if (!reached2) {
                lumaEnd2 = sampleLuma(uv2) - lumaLocalAverage;
            }
            reached1 = abs(lumaEnd1) >= gradientScaled;
            reached2 = abs(lumaEnd2) >= gradientScaled;
            reachedBoth = reached1 && reached2;

            if (!reached1) uv1 -= offset * QUALITY_STEPS[i];
            if (!reached2) uv2 += offset * QUALITY_STEPS[i];
            if (reachedBoth) break;
        }
    }

    // Distance to each end of the edge.
    float distance1 = isHorizontal ? (uv.x - uv1.x) : (uv.y - uv1.y);
    float distance2 = isHorizontal ? (uv2.x - uv.x) : (uv2.y - uv.y);
    bool isDirection1 = distance1 < distance2;
    float distanceFinal = min(distance1, distance2);
    float edgeThickness = distance1 + distance2;

    // Offset toward the closer edge end, only when the luma variation there is
    // consistent with the local edge (avoids overshooting past the silhouette).
    bool isLumaCenterSmaller = lumaC < lumaLocalAverage;
    bool correctVariation =
        ((isDirection1 ? lumaEnd1 : lumaEnd2) < 0.0) != isLumaCenterSmaller;
    float pixelOffset = correctVariation ? (-distanceFinal / edgeThickness + 0.5) : 0.0;

    // Subpixel anti-aliasing for high-frequency detail.
    float lumaAverage = (1.0 / 12.0) * (2.0 * (lumaDU + lumaLR) + lumaLeftCorners + lumaRightCorners);
    float subPixelOffset1 = saturate(abs(lumaAverage - lumaC) / lumaRange);
    float subPixelOffset2 = (-2.0 * subPixelOffset1 + 3.0) * subPixelOffset1 * subPixelOffset1;
    float subPixelOffsetFinal = subPixelOffset2 * subPixelOffset2 * SUBPIXEL_QUALITY;

    pixelOffset = max(pixelOffset, subPixelOffsetFinal);

    float2 finalUv = uv;
    if (isHorizontal) {
        finalUv.y += pixelOffset * stepLength;
    } else {
        finalUv.x += pixelOffset * stepLength;
    }

    float3 finalColor = ldrTexture.SampleLevel(ldrSampler, finalUv, 0).rgb;
    return float4(finalColor, 1.0);
}
