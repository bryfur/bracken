/**
 * Skybox Fragment Shader - SDL3 GPU API
 *
 * Procedural sky, all HDR linear (ACES tonemap happens at composite):
 * - 3-stop gradient (ground / horizon / zenith) from EnvironmentState colors
 * - analytic sun disc (~5.0 HDR, feeds bloom) + corona + glare
 * - two scrolling FBM value-noise cloud layers with sun-side silver lining
 * - aerial haze converging on the (sun-tinted) fog color so terrain fog and
 *   the sky meet invisibly at the horizon (matches applyAtmosphericFog)
 * - IGN dither to hide gradient banding; dormant hash starfield when the
 *   sun is below the horizon
 */

#include "lighting.hlsli"

// Fragment input (from vertex shader)
struct PSInput {
    float4 position : SV_Position;
    float2 clipPos : TEXCOORD0;
};

// Uniform buffer - SDL3 GPU SPIR-V requires fragment uniforms in set 3.
// Mirrors gpu::SkyboxFragmentUniforms (gpu_uniforms.hpp).
[[vk::binding(0, 3)]]
cbuffer SkyUniforms {
    float4x4 invViewProjection;
    float3 sunDirection;  // Direction TO sun (normalized)
    float time;
    float3 sunColor;
    float _pad0;
    float3 zenithColor;
    float _pad1;
    float3 horizonColor;
    float _pad2;
    float3 groundColor;
    float _pad3;
    float3 fogHorizonColor;
    float _pad4;
    float3 fogSunColor;
    float _pad5;
};

static const float CLOUD_COVERAGE = 0.35;

float hash21(float2 p) {
    p = frac(p * float2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return frac(p.x * p.y);
}

float valueNoise(float2 p) {
    float2 i = floor(p);
    float2 f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + float2(1.0, 0.0));
    float c = hash21(i + float2(0.0, 1.0));
    float d = hash21(i + float2(1.0, 1.0));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

float fbm4(float2 p) {
    float v = 0.0;
    float amp = 0.5;
    [unroll]
    for (int i = 0; i < 4; ++i) {
        v += amp * valueNoise(p);
        p = p * 2.13 + float2(17.3, 9.1);
        amp *= 0.5;
    }
    return v * (1.0 / 0.9375); // normalize amplitude sum to ~[0,1]
}

float4 PSMain(PSInput input) : SV_Target {
    // Compute world-space ray direction per-pixel
    float4 worldPos = mul(invViewProjection, float4(input.clipPos, 1.0, 1.0));
    float3 dir = normalize(worldPos.xyz / worldPos.w);

    float3 sunDir = normalize(sunDirection);
    float up = dir.y;
    // 1.0 at midday; ramps the whole sky down through dusk (night path).
    float dayFactor = saturate(sunDir.y * 2.0 + 0.2);

    // --- 3-stop gradient: ground below, horizon band, zenith above ---
    float3 zenith = zenithColor * lerp(0.03, 1.0, dayFactor);
    float3 horizonC = horizonColor * lerp(0.04, 1.0, dayFactor);
    float3 groundC = groundColor * lerp(0.05, 1.0, dayFactor);

    // pow compresses the gradient toward the horizon for a natural falloff.
    float3 sky = lerp(horizonC, zenith, pow(saturate(up), 0.42));
    sky = lerp(sky, groundC, 1.0 - smoothstep(-0.08, 0.0, up));

    // --- Clouds: two scrolling FBM layers projected onto a virtual plane ---
    float sunGlow = pow(saturate(dot(dir, sunDir)), 6.0);
    float cloudAlpha = 0.0;
    float3 cloudColor = float3(0.0, 0.0, 0.0);
    if (up > 0.005) {
        float2 planeUV = dir.xz / (up + 0.18);
        float d1 = fbm4(planeUV * 0.55 + float2(0.012, 0.005) * time);
        float d2 = fbm4(planeUV * 1.55 + float2(-0.021, 0.013) * time);
        float density = d1 * 0.65 + d2 * 0.35;

        float threshold = 1.0 - CLOUD_COVERAGE;
        float clouds = smoothstep(threshold, threshold + 0.22, density);
        float core = smoothstep(threshold + 0.10, threshold + 0.45, density);

        // Bright tops, shaded cores; silver lining where thin on the sun side.
        cloudColor = lerp(float3(1.15, 1.12, 1.08), float3(0.52, 0.55, 0.62), core) * dayFactor;
        cloudColor += sunColor * sunGlow * (1.0 - core) * 1.5 * dayFactor;

        // Clouds compress and fade into the horizon band.
        float horizonFade = smoothstep(0.02, 0.22, up);
        cloudAlpha = clouds * horizonFade * 0.85;
    }
    sky = lerp(sky, cloudColor, cloudAlpha);

    // --- Sun disc + corona + glare (occluded by cloud cover) ---
    float sunAngle = acos(clamp(dot(dir, sunDir), -1.0, 1.0));
    float sunVis = 1.0 - cloudAlpha * 0.9;
    float sunDisc = smoothstep(0.0175, 0.0115, sunAngle);
    float corona = exp(-sunAngle * sunAngle / 0.015) * 0.5;
    float glare = exp(-sunAngle / 0.35) * 0.15;
    sky += sunDisc * sunColor * 5.0 * sunVis;
    sky += corona * sunColor * float3(1.0, 0.88, 0.65) * sunVis;
    sky += glare * sunColor * 0.8 * dayFactor;

    // --- Aerial haze: converge on the fog color at the horizon with the same
    // directional sun lerp as applyAtmosphericFog() in fog.hlsli.
    float sunAmount = pow(saturate(dot(dir, sunDir)), 8.0);
    float3 hazeColor = lerp(fogHorizonColor, fogSunColor, sunAmount) * lerp(0.04, 1.0, dayFactor);
    float haze = exp(-abs(up) * 8.0);
    sky = lerp(sky, hazeColor, haze * 0.85);

    // --- Stars: dormant at midday; cheap hash field once the sun has set ---
    if (sunDir.y < 0.0) {
        float night = saturate(-sunDir.y * 4.0);
        float3 cell = floor(dir * 220.0);
        float h = frac(sin(dot(cell, float3(12.9898, 78.233, 37.719))) * 43758.5453);
        float star = step(0.998, h);
        sky += star * night * saturate(up) * float3(0.8, 0.85, 1.0);
    }

    // IGN dither (+/- 0.5/255 in linear) hides gradient banding after tonemap.
    // Offset decorrelates from the composite pass's post-encode IGN dither so
    // the two patterns average out on sky pixels instead of stacking.
    sky += (interleavedGradientNoise(input.position.xy + 71.0) - 0.5) * (1.0 / 255.0);

    return float4(max(sky, 0.0), 1.0);
}
