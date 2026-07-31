// sky_common.hlsli - the procedural sky radiance function.
//
// Shared by the skybox pass (skybox.frag.hlsl) and the IBL sky-capture pass
// (sky_capture.frag.hlsl) so both evaluate the IDENTICAL sky — edit here,
// never fork the math. All HDR linear (ACES tonemap happens at composite):
// - 3-stop gradient (ground / horizon / zenith) from EnvironmentState colors
// - analytic sun disc (~5.0 HDR, feeds bloom) + corona + glare
// - two scrolling domain-warped FBM cloud layers, shaped by the
//   EnvironmentState cloud params (coverage / density / scale; scroll speed
//   is folded into `time` on the CPU), with sun-proximity silver lining,
//   warm low-sun scattering, and cheap directional self-shadowing
// - aerial haze converging on the (sun-tinted) fog color so terrain fog and
//   the sky meet invisibly at the horizon (hazeSunColor is SHARED with
//   applyAtmosphericFog in fog.hlsli)
// - dormant hash starfield + full moon (opposite the sun) once the sun sets
// The skybox pass adds its screen-space IGN dither AFTER this function.

#ifndef SKY_COMMON_HLSLI
#define SKY_COMMON_HLSLI

#include "fog.hlsli" // hazeSunColor — the sky/fog horizon agreement contract

// Integer lattice hash (PCG2D, Jarzynski & Olano). valueNoise only ever
// hashes WHOLE lattice coordinates, so an integer hash is exact — the old
// frac(bigFloat * bigFloat) hash quantized under float32 at the coordinate
// magnitudes the upper FBM octaves reach (|p| ~ 100 -> products ~ 1e6, ULP
// 0.125), which rasterized as straight sawtooth creases across the clouds.
float hash21(float2 p) {
    uint2 v = uint2(int2(p));
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    return float(v.x) * (1.0 / 4294967296.0);
}

float valueNoise(float2 p) {
    float2 i = floor(p);
    float2 f = frac(p);
    // Quintic fade (C2): the cubic fade's derivative seams at cell borders
    // read as straight creases once the field is domain-warped and its
    // gradient drives the cloud self-shadowing.
    float2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float a = hash21(i);
    float b = hash21(i + float2(1.0, 0.0));
    float c = hash21(i + float2(0.0, 1.0));
    float d = hash21(i + float2(1.0, 1.0));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

// Rotating between octaves breaks the value-noise grid alignment that
// otherwise reads as blocky, axis-aligned cloud edges.
static const float2x2 FBM_ROT = float2x2(0.80, 0.60, -0.60, 0.80);

// Cloud density stack: 6 octaves with a FLAT spectrum (persistence 0.62).
// The classic 0.5 persistence lets the first octave dominate; thresholding
// such a field draws that octave's near-bilinear iso-contours as long
// straight sheet edges (the sky-plane projection squeezes the whole upper
// sky into ~one cell of it). The flatter spectrum keeps every scale
// contributing, so cloud outlines curve at every scale.
float cloudFbm(float2 p) {
    float v = 0.0;
    float amp = 1.0;
    float norm = 0.0;
    [unroll]
    for (int i = 0; i < 6; ++i) {
        v += amp * valueNoise(p);
        norm += amp;
        p = mul(FBM_ROT, p) * 2.13 + float2(17.3, 9.1);
        amp *= 0.62;
    }
    return v / norm; // ~[0,1]
}

// 3-octave FBM — the cheaper field used for the cloud domain warp.
float fbm3(float2 p) {
    float v = 0.0;
    float amp = 0.5;
    [unroll]
    for (int i = 0; i < 3; ++i) {
        v += amp * valueNoise(p);
        p = mul(FBM_ROT, p) * 2.13 + float2(17.3, 9.1);
        amp *= 0.5;
    }
    return v * (1.0 / 0.875); // normalize amplitude sum to ~[0,1]
}

// Evaluate the sky's linear HDR radiance along world direction `dir`
// (normalized). `sunDirection` points TO the sun (normalized inside, matching
// the historical skybox behavior); colors come from EnvironmentState.
// `time` is the pre-scaled cloud scroll phase (sky time already multiplied by
// EnvironmentState::cloud_speed on the CPU); cloudCoverage / cloudDensity /
// cloudScale / hazeStrength mirror the EnvironmentState fields.
float3 evaluateSkyRadiance(float3 dir, float3 sunDirection, float time, float3 sunColor,
                           float3 zenithColor, float3 horizonColor, float3 groundColor,
                           float3 fogHorizonColor, float3 fogSunColor,
                           float cloudCoverage, float cloudDensity, float cloudScale,
                           float hazeStrength) {
    float3 sunDir = normalize(sunDirection);
    float up = dir.y;
    // 1.0 at midday; ramps the whole sky down through dusk (night path).
    float dayFactor = saturate(sunDir.y * 2.0 + 0.2);
    // 1.0 with the sun at the horizon, 0 above ~19 deg — drives the warm
    // low-sun response (matches hazeSunColor in fog.hlsli).
    float lowSun = 1.0 - saturate(sunDir.y * 3.0);
    // Direct sun on clouds survives a few degrees past sunset (underlit
    // bases), gone by ~ -7 deg (matches hazeSunColor's afterglow window).
    float duskFade = saturate(sunDir.y * 8.0 + 1.0);
    float cosSun = dot(dir, sunDir);

    // --- 3-stop gradient: ground below, horizon band, zenith above ---
    float3 zenith = zenithColor * lerp(0.03, 1.0, dayFactor);
    float3 horizonC = horizonColor * lerp(0.04, 1.0, dayFactor);
    float3 groundC = groundColor * lerp(0.05, 1.0, dayFactor);

    // pow compresses the gradient toward the horizon for a natural falloff.
    float3 sky = lerp(horizonC, zenith, pow(saturate(up), 0.42));
    sky = lerp(sky, groundC, 1.0 - smoothstep(-0.08, 0.0, up));

    // --- Clouds: domain-warped FBM on a virtual plane, two frequencies ---
    float cloudAlpha = 0.0;
    float3 cloudColor = float3(0.0, 0.0, 0.0);
    if (up > 0.003 && cloudCoverage > 0.001 && cloudDensity > 0.001) {
        float2 planeUV = dir.xz / (up + 0.18) * cloudScale;
        float2 wind = float2(0.012, 0.005) * time;

        // Domain warp: a vector field between the base and detail frequencies
        // bends the straight FBM striations into billowing, curling fronts.
        float2 warpUV = planeUV * 0.7 + wind * 0.8;
        float2 warp = float2(fbm3(warpUV + float2(3.7, 1.2)),
                             fbm3(warpUV + float2(-2.4, 5.3))) - 0.5;

        float2 baseUV = planeUV * 0.7 + wind + warp * 0.35;
        float density = cloudFbm(baseUV);

        // Perspective compression: effective coverage thins toward the
        // horizon so the deck reads as flat sheets receding into the haze.
        float horizonFade = smoothstep(0.005, 0.16, up);
        float coverage = cloudCoverage * lerp(0.55, 1.0, horizonFade);

        float threshold = 1.0 - coverage;
        float clouds = smoothstep(threshold, threshold + 0.24, density);
        float core = smoothstep(threshold + 0.08, threshold + 0.42, density);

        // Cheap directional self-shadowing: the density gradient toward the
        // sun. Where the deck thickens sunward this texel sits in the cloud's
        // own shadow (dark base); where it thins sunward it is a lit face.
        float2 toSunUV = normalize(sunDir.xz + float2(1e-4, 0.0));
        float dSun = cloudFbm(baseUV + toSunUV * 0.15);
        float litFace = saturate((density - dSun) * 3.0 + 0.5);

        // Ambient (sky-lit) body: bright tops to blue-gray shadowed bases;
        // lit faces keep more of the top color. The night floor tracks the
        // gradient's own floor so clouds darken with the sky, never glow.
        float3 topColor = float3(1.16, 1.13, 1.09);
        float3 baseColor = float3(0.46, 0.50, 0.58);
        float baseShade = core * lerp(1.0, 0.45, litFace);
        cloudColor = lerp(topColor, baseColor, baseShade) * lerp(0.035, 1.0, dayFactor);

        // Sun-proximity scattering: a tight silver lining on thin edges plus
        // a broad warm bleed across lit faces; both strengthen and warm as
        // the sun drops (long Mie path), and follow duskFade below the
        // horizon so sunset bases stay underlit while night clouds go dark.
        float rim = clouds * (1.0 - core);
        float silverLobe = pow(saturate(cosSun), 12.0);
        float broadLobe = pow(saturate(cosSun * 0.5 + 0.5), 4.0);
        float3 scatterTint = lerp(float3(1.0, 0.96, 0.90), float3(1.2, 0.62, 0.32), lowSun);
        cloudColor += sunColor * scatterTint *
                      (rim * silverLobe * (1.2 + 1.3 * lowSun) + litFace * broadLobe * 0.35) *
                      duskFade;

        // Opacity: soft translucent edges, near-solid cores; cloudDensity
        // re-weights the whole mask (wispy < 1 < heavy), and sheets thin
        // toward the horizon (flat perspective).
        float alphaFade = smoothstep(0.005, 0.14, up);
        cloudAlpha = saturate(clouds * (0.72 + 0.28 * core) * cloudDensity) * alphaFade;
    }
    sky = lerp(sky, cloudColor, cloudAlpha);

    // --- Sun disc + corona + glare (occluded by cloud cover) ---
    float sunAngle = acos(clamp(cosSun, -1.0, 1.0));
    float sunVis = 1.0 - cloudAlpha * 0.9;
    float sunDisc = smoothstep(0.0175, 0.0115, sunAngle);
    float corona = exp(-sunAngle * sunAngle / 0.015) * 0.5;
    float glare = exp(-sunAngle / 0.35) * 0.15;
    sky += sunDisc * sunColor * 5.0 * sunVis;
    sky += corona * sunColor * float3(1.0, 0.88, 0.65) * sunVis;
    sky += glare * sunColor * 0.8 * dayFactor;

    // --- Aerial haze: converge on the fog color at the horizon with the same
    // sun-scatter response as applyAtmosphericFog() — hazeSunColor is shared
    // (including its day dimming), which is what makes fog and sky meet
    // invisibly at the horizon at every sun elevation.
    float3 hazeColor = hazeSunColor(dir, sunDir, fogHorizonColor, fogSunColor);
    float haze = exp(-abs(up) * 8.0);
    sky = lerp(sky, hazeColor, saturate(haze * 0.85 * hazeStrength));

    // --- Stars: dormant at midday; cheap hash field once the sun has set.
    // Clouds occlude them (dark silhouettes drifting across the field).
    if (sunDir.y < 0.0) {
        float night = saturate(-sunDir.y * 4.0);

        // Night skyglow: a real night sky is never black (moonlight scatter
        // + airglow), and this term is load-bearing — the IBL capture runs
        // through this same function and REPLACES the hemisphere ambient, so
        // whatever radiance the night dome carries IS the scene's night
        // ambient. Horizon-weighted faint blue, occluded by nothing.
        float glowDome = lerp(0.5, 1.0, exp(-saturate(up) * 2.5));
        sky += night * glowDome * float3(0.015, 0.021, 0.038);

        float3 cell = floor(dir * 220.0);
        float h = frac(sin(dot(cell, float3(12.9898, 78.233, 37.719))) * 43758.5453);
        float star = step(0.998, h);
        sky += star * night * saturate(up) * (1.0 - cloudAlpha) * float3(0.8, 0.85, 1.0);

        // --- Moon: a full moon rides exactly opposite the sun (astronomically
        // true for the full phase), so it rises as the sun sets with no extra
        // uniforms — day/night hosts get it for free. Cool disc with limb
        // darkening + a soft glow; clouds occlude it like the stars. ~1.4 HDR:
        // bright against the ~0.03 night gradient, a gentle bloom feed, far
        // below the sun's 5.0.
        float3 moonDir = -sunDir;
        float cosMoon = dot(dir, moonDir);
        float moonAngle = acos(clamp(cosMoon, -1.0, 1.0));
        float moonDisc = smoothstep(0.016, 0.0125, moonAngle);
        // Limb darkening + a fixed hash mottle so the disc reads as a body,
        // not a flat sticker.
        float limb = 1.0 - 0.45 * smoothstep(0.006, 0.014, moonAngle);
        float mottle = 0.85 + 0.15 * valueNoise(dir.xz / max(abs(dir.y), 0.2) * 60.0);
        float moonGlow = exp(-moonAngle * moonAngle / 0.006) * 0.30;
        float moonVis = night * (1.0 - cloudAlpha * 0.9);
        float3 moonColor = float3(0.92, 0.95, 1.05);
        sky += moonDisc * limb * mottle * moonColor * 1.4 * moonVis;
        sky += moonGlow * moonColor * moonVis;
    }

    return sky;
}

#endif // SKY_COMMON_HLSLI
