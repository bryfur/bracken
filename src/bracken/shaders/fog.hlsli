// fog.hlsli - shared fog helpers.
// All values in linear light; caller blends in linear space.

#ifndef FOG_HLSLI
#define FOG_HLSLI

float linearFog(float distance, float start, float end) {
    return saturate((distance - start) / max(end - start, 1e-5));
}

float exponentialFog(float distance, float density) {
    return 1.0 - exp(-distance * density);
}

float exponentialSquaredFog(float distance, float density) {
    float f = distance * density;
    return 1.0 - exp(-f * f);
}

// Legacy-compatible: linear factor with exp-shaped falloff; matches
// the previous inline formula used by model/terrain shaders.
float distanceFogFactor(float distance, float start, float end) {
    float t = linearFog(distance, start, end);
    return 1.0 - exp(-t * 2.0);
}

// Sun-facing haze/fog color: a Mie-like warm forward-scatter lobe around the
// sun that WIDENS and BRIGHTENS as the sun drops toward the horizon (long
// grazing path through the atmosphere) and stays subtle/neutral at midday.
// The afterglow survives a few degrees past sunset, then fades so night fog
// is not warm-tinted toward a sun that has set. toSun = normalized direction
// TO the sun. Used by BOTH the sky's aerial haze (sky_common.hlsli) and the
// surface atmospheric fog below — the shared function is what makes ground
// fog and sky agree exactly at the horizon line.
float3 hazeSunColor(float3 viewDir, float3 toSun, float3 fogHorizonColor, float3 fogSunColor) {
    float cosSun = saturate(dot(viewDir, toSun));
    // 0 with the sun >= ~19 deg elevation, 1 with the sun at the horizon.
    float lowSun = 1.0 - saturate(toSun.y * 3.0);
    // 1 above the horizon, fading out by ~ -7 deg elevation (afterglow window).
    float duskFade = saturate(toSun.y * 8.0 + 1.0);
    // Midday keeps the historical tight pow-8 lobe; grazing sun flattens it.
    float lobe = pow(cosSun, lerp(8.0, 2.5, lowSun));
    float3 c = lerp(fogHorizonColor, fogSunColor, lobe * duskFade);
    // Forward-scatter brightening hugging the sun at low elevations.
    c *= 1.0 + pow(cosSun, 6.0) * (1.4 * lowSun) * duskFade;
    // Day dimming: the same ramp the sky gradient uses (dayFactor =
    // saturate(sin(elevation) * 2 + 0.2)), applied HERE so surface fog and
    // the sky's haze darken together through dusk — without it, fog stays a
    // glowing daylight band under a night sky. No-op at midday.
    return c * lerp(0.04, 1.0, saturate(toSun.y * 2.0 + 0.2));
}

// Atmospheric fog: distanceFogFactor-shaped distance term, attenuated by the
// analytic average of exp(-y * heightFalloff) along the view ray (fog thins
// with altitude), with the fog color blending toward the sun on the sun side
// (hazeSunColor above). Matches the skybox aerial haze so terrain fog
// disappears into the horizon.
// sunDir is the direction the light TRAVELS (EnvironmentState::sun_dir).
float3 applyAtmosphericFog(float3 color, float3 fragPos, float3 cameraPos,
                           float3 sunDir, float3 fogHorizonColor, float3 fogSunColor,
                           float fogStart, float fogEnd, float heightFalloff) {
    float3 toFrag = fragPos - cameraPos;
    float dist = max(length(toFrag), 1e-4);
    float3 viewDir = toFrag / dist;

    float factor = distanceFogFactor(dist, fogStart, fogEnd);

    // Average density along the ray: integral of exp(-y*k) for y moving
    // linearly cameraPos.y -> fragPos.y is h0 * (1 - exp(-k*dy)) / (k*dy).
    float kdy = (fragPos.y - cameraPos.y) * heightFalloff;
    float h0 = exp(-cameraPos.y * heightFalloff);
    float heightFactor = (abs(kdy) > 1e-4) ? h0 * (1.0 - exp(-kdy)) / kdy : h0;
    factor *= saturate(heightFactor);

    float3 fogColor = hazeSunColor(viewDir, -sunDir, fogHorizonColor, fogSunColor);
    return lerp(color, fogColor, factor);
}

#endif // FOG_HLSLI
