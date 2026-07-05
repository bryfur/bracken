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

// Atmospheric fog: distanceFogFactor-shaped distance term, attenuated by the
// analytic average of exp(-y * heightFalloff) along the view ray (fog thins
// with altitude), with the fog color blending toward the sun on the sun side.
// Matches the skybox aerial haze so terrain fog disappears into the horizon.
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

    float sunAmount = pow(saturate(dot(viewDir, -sunDir)), 8.0);
    float3 fogColor = lerp(fogHorizonColor, fogSunColor, sunAmount);
    return lerp(color, fogColor, factor);
}

#endif // FOG_HLSLI
