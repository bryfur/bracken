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
    // The hemisphere ambient (EnvironmentState). Needed here only since R1c:
    // it is both the probe field's fallback and the REFERENCE the GI gain
    // below is measured against, so the two must be the same numbers.
    float3 ambientSky;
    float _ambientPad0;
    float3 ambientGround;
    float _ambientPad1;
    // Probe-field GI (render/gi/), R1c — see gi_sample.hlsli.
    int4 giCascadeBase[3];
    float giEnabled;
    float _giPad0;
    float _giPad1;
    float _giPad2;
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

// Probe-field GI (render/gi/), at this pass's slots 2-3. Bound to dummies when
// GI is off; giEnabled gates them.
//
// TWO atlases, not the three the forward shaders declare: this pass calls
// giSampleVolume and nothing else, so `giVisibilityAtlas` would be a resource
// no executed code reads. That is not free — an unpaired Texture2D/SamplerState
// does not survive DXC's combined-image-sampler legalization, and shadercross
// then reflects the orphaned image as a storage texture, which puts a storage
// texture BEFORE a sampler in the space2 run and breaks the packed order SDL3
// GPU assumes. GI_SAMPLE_VOLUME_ONLY compiles out everything that reads it.
#define GI_SAMPLE_VOLUME_ONLY

[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
Texture2D<float4> giIrradianceAtlas : register(t2, space2);
[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
SamplerState giIrradianceAtlasSampler : register(s2, space2);

[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
Texture2D<float4> giProbeMeta : register(t3, space2);
[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
SamplerState giProbeMetaSampler : register(s3, space2);

// worldPosFromDepth + interleavedGradientNoise
#include "lighting.hlsli"
#include "gi_sample.hlsli"

// The share of the horizon fog colour this pass in-scatters as ambient. It has
// been 0.2 since the pass was written and R1c does not change it — see the
// ambient block in the march below for why that is the whole point.
static const float VOLUMETRIC_AMBIENT_SCALE = 0.2;

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
    float3 worldPos = worldPosFromDepth(input.texCoord, depth, invViewProjection);
    float3 rayDir = worldPos - cameraPos;
    float rayLength = length(rayDir);
    rayDir /= rayLength;

    // Mie scattering phase function (simplified forward scattering)
    float3 lightDirection = normalize(-lightDir);
    float cosAngle = dot(rayDir, lightDirection);
    float phase = 0.25 + 0.75 * pow(max(0.0, cosAngle), 8.0);

    // The day-dimming ramp `hazeSunColor` (fog.hlsli) ends with, transliterated
    // rather than called: calling it would need a second uniform (fogSunColor)
    // and would bring its warm forward-scatter lobe with it, which this flat
    // ambient fraction deliberately does not have. `lightDirection` is toSun,
    // exactly the argument hazeSunColor takes. 1.0 with the sun above ~24 deg,
    // 0.04 once it is down — loop-invariant, so it is hoisted here.
    float ambientDayDim = lerp(0.04, 1.0, saturate(lightDirection.y * 2.0 + 0.2));

    // Ray march through the volume (numSteps = the pass's cost knob)
    int numSteps = 24;
    float endDist = isSky ? maxDist : min(rayLength, maxDist);
    if (endDist <= startDist) return float4(0, 0, 0, 0);
    float stepSize = (endDist - startDist) / float(numSteps);

    float3 fog = float3(0, 0, 0);
    float transmittance = 1.0;

    // Interleaved gradient noise (less visible pattern than white noise hash)
    float dither = interleavedGradientNoise(input.position.xy);
    float dist = startDist + stepSize * dither;
    float3 pos = cameraPos + rayDir * dist;

    for (int i = 0; i < numSteps; i++) {
        float density = getFogDensity(pos);

        if (density > 0.001) {
            // Ambient fog contribution (always active).
            //
            // THE SKY AGREEMENT CONTRACT, and how R1c keeps it. This pass is an
            // additive in-scatter layer over a scene that has already had
            // surface fog applied by `applyAtmosphericFog` (fog.hlsli), and
            // that function shares `hazeSunColor` with the skybox's aerial haze
            // — which is what makes ground fog and sky meet exactly at the
            // horizon. The volumetric layer stays inside that agreement by
            // being a fixed FRACTION of the same `fogColor` AFTER the same
            // day-dimming ramp (`ambientDayDim` above). The ramp is the
            // load-bearing half: `hazeSunColor`'s last line is where surface
            // fog and the sky's haze darken together through dusk, so a layer
            // that took `fogColor` raw was 5x the band it is supposed to sit
            // inside once the sun was down — a flat grey wash over a night
            // scene whose extinction had already saturated. Above ~24 deg sun
            // elevation the ramp is exactly 1.0, so daylight output is
            // bit-identical to the undimmed form.
            //
            // So GI does NOT replace the colour: it becomes a per-channel GAIN
            // around 1, measured against the very hemisphere ambient the probe
            // sampler falls back to. Where the field has no answer — which
            // includes everything past cascade 2, i.e. the horizon itself —
            // giSampleVolume returns that hemisphere mean exactly, the ratio is
            // exactly 1, and this term is bit-identical to the pre-R1c
            // `fogColor * 0.2 * ambientDayDim`. The contract therefore holds by
            // construction rather than by tuning, and what GI buys is what it
            // should: fog that darkens under an eave and warms in a lamp-lit
            // street. The dimming is applied to the BASE colour so the GI gain
            // (a ratio around 1) still multiplies it unchanged.
            float3 ambientFog = fogColor * VOLUMETRIC_AMBIENT_SCALE * ambientDayDim;
            if (giEnabled > 0.5) {
                float3 reference = max((ambientSky + ambientGround) * 0.5, 1e-4);
                float3 gain = giSampleVolume(pos, ambientSky, ambientGround) / reference;
                ambientFog *= clamp(gain, 0.0, GI_VOLUME_MAX_GAIN);
            }

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
