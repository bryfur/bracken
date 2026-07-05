/**
 * Volumetric Fog Fragment Shader (God Rays / Light Shafts)
 *
 * Screen-space ray marching through height-based fog volume.
 * For each pixel, marches from camera toward the world position,
 * sampling the shadow map to determine lit vs shadowed fog. Sky pixels
 * (reversed-Z depth == 0) are marched to maxDist so shafts silhouette against
 * the sky instead of cutting off at the horizon. God rays sample the whole
 * CSM atlas: the cascade is selected per march sample from its camera
 * distance. Produces inscattered light (god rays) and fog extinction.
 *
 * Cost knob: numSteps below — every half-res pixel (sky included) pays
 * numSteps density samples, plus one shadow tap per step with god rays on.
 *
 * Output: RGB = inscattered light, A = 1 - transmittance (opacity)
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer VolumetricFogUniforms : register(b0, space3) {
    float4x4 invViewProjection;
    float4x4 shadowLightViewProjection[4]; // per-cascade light-space matrices
    float4 cascadeSplits;   // view-space far distance per cascade
    float4 shadowBias;      // per-cascade light-space depth compare bias
    float3 lightDir;
    float fogDensity;
    float3 lightColor;
    float scatterStrength;
    float3 fogColor;
    float fogHeight;
    float3 cameraPos;
    float fogFalloff;
    float startDist;        // near dead zone: no volumetric fog closer than this
    float maxDist;          // ray-march distance cap
    float godRaysEnabled;
    float densityMultiplier;
    float shadowAtlasTexel; // 1 / atlas width
    float lastActiveSplit;  // far distance of the last active cascade (lit beyond it)
    float _padding0;
    float _padding1;
};

// Depth texture
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D<float> depthTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState depthSampler : register(s0, space2);

// Shadow atlas, raw sampler (2x2 quadrant grid: cascade i at (i%2, i/2))
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D shadowMap : register(t1, space2);
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerState shadowSampler : register(s1, space2);

float3 reconstructWorldPos(float2 uv, float depth) {
    float4 clipPos = float4(uv * 2.0 - 1.0, depth, 1.0);
    clipPos.y = -clipPos.y;  // Flip Y for Vulkan
    float4 worldPos = mul(invViewProjection, clipPos);
    return worldPos.xyz / worldPos.w;
}

// dist = camera distance of the march sample. It stands in for view depth
// when selecting a cascade: distance >= view depth, so at worst a farther
// (coarser) cascade is picked slightly early - never a too-near one.
float sampleShadow(float3 worldPos, float dist) {
    // Past the last cascade: fully lit (matches the main pass fade-out).
    if (dist >= lastActiveSplit)
        return 1.0;

    int cascade = 0;
    if (dist >= cascadeSplits.x) cascade = 1;
    if (dist >= cascadeSplits.y) cascade = 2;
    if (dist >= cascadeSplits.z) cascade = 3;

    float4 lightSpace = mul(shadowLightViewProjection[cascade], float4(worldPos, 1.0));
    float3 projCoords = lightSpace.xyz / lightSpace.w;
    projCoords.x = projCoords.x * 0.5 + 0.5;
    projCoords.y = 1.0 - (projCoords.y * 0.5 + 0.5);

    if (projCoords.x < 0.0 || projCoords.x > 1.0 ||
        projCoords.y < 0.0 || projCoords.y > 1.0)
        return 1.0;

    // Cascade i occupies atlas quadrant (i%2, i/2); clamp one texel inside
    // it so the lookup never bleeds into a neighboring cascade. A single raw
    // compare per step is enough - the ray march is already dithered.
    float2 quadOrigin = float2(cascade & 1, cascade >> 1) * 0.5;
    float2 uv = quadOrigin + clamp(projCoords.xy * 0.5, shadowAtlasTexel, 0.5 - shadowAtlasTexel);
    float shadowDepth = shadowMap.SampleLevel(shadowSampler, uv, 0).r;
    return (shadowDepth < projCoords.z - shadowBias[cascade]) ? 0.0 : 1.0;
}

// Height-based fog density - smooth exponential falloff from ground level
float getFogDensity(float3 pos) {
    // Smooth falloff: densest at y=0, fades exponentially with height
    // fogHeight controls the scale (higher = fog extends higher)
    // No hard cutoff - just a continuous gradient
    float normalizedHeight = pos.y / max(fogHeight, 1.0);
    float heightFactor = exp(-normalizedHeight * normalizedHeight * fogFalloff * 10.0);
    return fogDensity * densityMultiplier * heightFactor;
}

float4 PSMain(PSInput input) : SV_Target {
    float depth = depthTexture.SampleLevel(depthSampler, input.texCoord, 0).r;

    // Sky pixels have no surface hit: march the full fog range instead of
    // early-outing so god rays and haze cross the horizon silhouette.
    // Reversed-Z: the cleared/skybox far-plane depth is exactly 0.0.
    bool isSky = depth <= 0.0;
    float3 worldPos = reconstructWorldPos(input.texCoord, depth);
    float3 rayDir = worldPos - cameraPos;
    float rayLength = length(rayDir);
    rayDir /= rayLength;

    // Mie scattering phase function (simplified forward scattering)
    float3 lightDirection = normalize(-lightDir);
    float cosAngle = dot(rayDir, lightDirection);
    float phase = 0.25 + 0.75 * pow(max(0.0, cosAngle), 8.0);

    // Ray march through the volume (numSteps = the pass's cost knob)
    int numSteps = 24;
    float endDist = isSky ? maxDist : min(rayLength, maxDist);
    if (endDist <= startDist) return float4(0, 0, 0, 0);
    float stepSize = (endDist - startDist) / float(numSteps);

    float3 fog = float3(0, 0, 0);
    float transmittance = 1.0;

    // Interleaved gradient noise (less visible pattern than white noise hash)
    float2 screenPos = input.position.xy;
    float dither = frac(52.9829189 * frac(dot(screenPos, float2(0.06711056, 0.00583715))));
    float dist = startDist + stepSize * dither;
    float3 pos = cameraPos + rayDir * dist;

    for (int i = 0; i < numSteps; i++) {
        float density = getFogDensity(pos);

        if (density > 0.001) {
            // Ambient fog contribution (always active)
            float3 ambientFog = fogColor * 0.2;

            // God rays: inscattered light through unshadowed fog
            float3 inScatter = float3(0, 0, 0);
            if (godRaysEnabled > 0.5) {
                float shadow = sampleShadow(pos, dist);
                inScatter = shadow * lightColor * phase * scatterStrength;
            }

            float extinction = density * stepSize;
            float stepTransmittance = exp(-extinction);

            fog += transmittance * (inScatter + ambientFog) * (1.0 - stepTransmittance);
            transmittance *= stepTransmittance;
        }

        pos += rayDir * stepSize;
        dist += stepSize;
    }

    return float4(fog, 1.0 - transmittance);
}
