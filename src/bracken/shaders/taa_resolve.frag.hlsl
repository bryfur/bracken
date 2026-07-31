/**
 * TAA Resolve Fragment Shader.
 *
 * Temporally accumulates the jittered linear-HDR scene color (RGBA16F
 * offscreen target) into a same-format history buffer. Runs BEFORE bloom /
 * composite so the whole post chain (bloom flicker included) is stabilized.
 * Mirrors gpu::TaaResolveUniforms (gpu_uniforms.hpp) byte-for-byte.
 *
 * Per pixel:
 *  1. Velocity dilation — the 3x3 depth neighborhood's CLOSEST texel
 *     (reversed-Z: max depth) supplies the motion vector, so silhouettes of
 *     moving objects reproject with the object, not the background.
 *  2. Reprojection — prev_uv = uv - velocity (velocity convention:
 *     scene/velocity_math.hpp). Pixels with zero velocity (sky, and
 *     MASK/BLEND draws that skip the velocity prepass) reproject through
 *     depth + the unjittered current/previous view-projection pair; device
 *     depth 0 (reversed-Z far plane) reprojects as a direction with w = 0,
 *     which is translation-independent — exact for points at infinity.
 *  3. History rectification — variance clipping in YCoCg space: the history
 *     sample is clipped toward the 3x3 neighborhood mean within mean +/-
 *     varianceGamma * stddev (clip toward center, not per-channel clamp).
 *  4. History sampling — 5-tap optimized Catmull-Rom (bicubic) so repeated
 *     reprojection doesn't accumulate bilinear blur.
 *  5. Blend — history feedback up to feedbackMax (~0.9), Karis
 *     luminance-weighted (1/(1+luma)) to suppress HDR firefly flicker;
 *     feedback is reduced with velocity magnitude and with the amount the
 *     history had to be clipped (disocclusion heuristics).
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer TaaResolveUniforms : register(b0, space3) {
    float4x4 currInvViewProjUnjittered; // inverse of this frame's UNJITTERED view-proj
    float4x4 prevViewProjUnjittered;    // last frame's UNJITTERED view-proj
    float2 screenSize;
    float2 invScreenSize;
    float3 cameraPos;
    float historyValid;   // 0 = first frame / history invalidated: output current
    float feedbackMax;    // history weight ceiling (~0.9)
    float varianceGamma;  // stddev multiplier for the YCoCg clip box
    float _pad0;
    float _pad1;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D sceneTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState sceneSampler : register(s0, space2); // nearest clamp

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D historyTexture : register(t1, space2);
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerState historySampler : register(s1, space2); // linear clamp (Catmull-Rom taps)

[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
Texture2D<float2> velocityTexture : register(t2, space2);
[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
SamplerState velocitySampler : register(s2, space2); // nearest clamp

[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
Texture2D<float> depthTexture : register(t3, space2);
[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
SamplerState depthSampler : register(s3, space2); // nearest clamp

// YCoCg keeps chroma bounded while luma carries the HDR range — variance
// clipping in it rejects ghosting with far less hue shift than RGB clipping.
float3 rgbToYCoCg(float3 c) {
    return float3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
                  0.5 * c.r - 0.5 * c.b,
                  -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

float3 yCoCgToRgb(float3 c) {
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Clip `p` toward the box center (not a per-channel clamp — that shifts hue).
// Returns the clipped point; `clipAmount` in [0,1) is how far outside the box
// the point was (0 = inside), used as a disocclusion signal.
float3 clipTowardCenter(float3 boxMin, float3 boxMax, float3 p, out float clipAmount) {
    float3 center = 0.5 * (boxMax + boxMin);
    float3 extents = 0.5 * (boxMax - boxMin) + 1e-5;
    float3 a = abs((p - center) / extents);
    float maxComp = max(a.x, max(a.y, a.z));
    clipAmount = maxComp > 1.0 ? 1.0 - 1.0 / maxComp : 0.0;
    return maxComp > 1.0 ? center + (p - center) / maxComp : p;
}

// 5-tap optimized Catmull-Rom (Jimenez): the 9-tap bicubic collapsed to 5
// bilinear fetches by folding the center-adjacent weights, corners dropped
// and the kernel renormalized. Sharp history resampling without the 1-pixel
// blur a plain bilinear fetch accumulates every frame.
float3 sampleHistoryCatmullRom(float2 uv) {
    float2 samplePos = uv * screenSize;
    float2 texPos1 = floor(samplePos - 0.5) + 0.5;
    float2 f = samplePos - texPos1;

    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;

    float2 texPos0 = (texPos1 - 1.0) * invScreenSize;
    float2 texPos3 = (texPos1 + 2.0) * invScreenSize;
    float2 texPos12 = (texPos1 + w2 / w12) * invScreenSize;

    float3 result = 0.0;
    float totalWeight = 0.0;

    float w;
    w = w12.x * w0.y;
    result += historyTexture.SampleLevel(historySampler, float2(texPos12.x, texPos0.y), 0).rgb * w;
    totalWeight += w;
    w = w0.x * w12.y;
    result += historyTexture.SampleLevel(historySampler, float2(texPos0.x, texPos12.y), 0).rgb * w;
    totalWeight += w;
    w = w12.x * w12.y;
    result += historyTexture.SampleLevel(historySampler, float2(texPos12.x, texPos12.y), 0).rgb * w;
    totalWeight += w;
    w = w3.x * w12.y;
    result += historyTexture.SampleLevel(historySampler, float2(texPos3.x, texPos12.y), 0).rgb * w;
    totalWeight += w;
    w = w12.x * w3.y;
    result += historyTexture.SampleLevel(historySampler, float2(texPos12.x, texPos3.y), 0).rgb * w;
    totalWeight += w;

    // Negative Catmull-Rom lobes can undershoot: clamp to valid HDR.
    return max(result / totalWeight, 0.0);
}

float4 PSMain(PSInput input) : SV_Target0 {
    float2 uv = input.texCoord;

    // ---- 3x3 neighborhood: color moments (YCoCg) + closest-depth texel ----
    float3 m1 = 0.0;
    float3 m2 = 0.0;
    float3 currYCoCg = 0.0;
    float centerDepth = 0.0;
    float closestDepth = -1.0;
    float2 closestOffset = 0.0;

    [unroll]
    for (int dy = -1; dy <= 1; ++dy) {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx) {
            float2 offset = float2(dx, dy);
            float2 suv = uv + offset * invScreenSize;
            float3 c = rgbToYCoCg(sceneTexture.SampleLevel(sceneSampler, suv, 0).rgb);
            m1 += c;
            m2 += c * c;
            float d = depthTexture.SampleLevel(depthSampler, suv, 0);
            if (d > closestDepth) { // reversed-Z: larger = closer
                closestDepth = d;
                closestOffset = offset;
            }
            if (dx == 0 && dy == 0) {
                currYCoCg = c;
                centerDepth = d;
            }
        }
    }

    float3 currRgb = yCoCgToRgb(currYCoCg);
    if (historyValid < 0.5) {
        return float4(currRgb, 1.0);
    }

    // ---- Velocity dilation: motion of the closest 3x3 texel ----
    float2 velocity = velocityTexture.SampleLevel(velocitySampler, uv + closestOffset * invScreenSize, 0);

    // ---- Reproject ----
    float2 prevUv;
    if (any(velocity != 0.0)) {
        prevUv = uv - velocity; // velocity = curr - prev (velocity_math.hpp)
    } else {
        // No prepass velocity written (sky, MASK foliage, blended draws):
        // reproject through depth + the unjittered view-projection pair.
        float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);
        float4 prevClip;
        if (centerDepth <= 0.0) {
            // Far plane (sky): reproject the view ray direction with w = 0 —
            // translation drops out, exact for points at infinity.
            float4 world = mul(currInvViewProjUnjittered, float4(ndc, 0.5, 1.0));
            float3 dir = world.xyz / world.w - cameraPos;
            prevClip = mul(prevViewProjUnjittered, float4(dir, 0.0));
        } else {
            float4 world = mul(currInvViewProjUnjittered, float4(ndc, centerDepth, 1.0));
            world /= world.w;
            prevClip = mul(prevViewProjUnjittered, world);
        }
        if (prevClip.w <= 0.0) {
            return float4(currRgb, 1.0); // behind the previous camera: no history
        }
        float2 prevNdc = prevClip.xy / prevClip.w;
        prevUv = float2(prevNdc.x * 0.5 + 0.5, 0.5 - prevNdc.y * 0.5);
    }

    if (any(prevUv < 0.0) || any(prevUv > 1.0)) {
        return float4(currRgb, 1.0); // reprojected off-screen: no history
    }

    // ---- Rectify history: variance clip in YCoCg ----
    float3 mu = m1 / 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - mu * mu, 0.0));
    float3 boxMin = mu - varianceGamma * sigma;
    float3 boxMax = mu + varianceGamma * sigma;

    float3 histYCoCg = rgbToYCoCg(sampleHistoryCatmullRom(prevUv));
    float clipAmount;
    histYCoCg = clipTowardCenter(boxMin, boxMax, histYCoCg, clipAmount);

    // ---- Blend: feedback reduced on disocclusion, Karis luma-weighted ----
    float velocityPixels = length(velocity * screenSize);
    float feedback = feedbackMax;
    feedback *= 1.0 - 0.35 * saturate(velocityPixels / 32.0);
    feedback *= 1.0 - 0.6 * clipAmount;

    // Tonemap-weighting (Karis): down-weight HDR fireflies so a single hot
    // sample can't flicker the accumulated result.
    float wCurr = (1.0 - feedback) * rcp(1.0 + currYCoCg.x);
    float wHist = feedback * rcp(1.0 + histYCoCg.x);
    float3 resolved = (currYCoCg * wCurr + histYCoCg * wHist) / max(wCurr + wHist, 1e-6);

    return float4(max(yCoCgToRgb(resolved), 0.0), 1.0);
}
