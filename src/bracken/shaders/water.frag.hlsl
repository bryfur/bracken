// water.frag.hlsl - river/lake water surface.
//
// Runs in its own pass after the opaque scene + skybox: the HDR scene color
// and depth were copied out first, so this shader composites refraction
// (distorted scene-behind-water), depth-based absorption, sky reflection
// from the IBL prefiltered specular cube, a sun glint, shoreline + ripple
// foam, and the shared atmospheric fog — all in-shader (the pipeline runs
// opaque, no fixed-function blending).
//
// Normals come from two layers: finite differences of the simulation height
// field (interactive ripples) and small analytic waves scrolling with the
// flow (the river's constant motion).

[[vk::binding(0, 3)]]
cbuffer WaterFragmentUniforms : register(b0, space3) {
    float4 cameraPosTime;   // xyz = camera world pos, w = time (s)
    float4 sunDirIntensity; // xyz = direction light travels, w = sun intensity
    float4 sunColorIbl;     // rgb = sun color (linear), w = iblEnabled (0/1)
    float4 deepColor;       // rgb = deep water tint, w = absorption (1/units)
    float4 shallowColor;    // rgb = shallow tint, w = shore foam depth (units)
    float4 fogHorizon;      // rgb = fog horizon color, w = fogStart
    float4 fogSun;          // rgb = fog sun color, w = fogEnd
    float4 screenDepthAB;   // xy = 1/screen size, z = depthA, w = depthB
    float4 flowDetail;      // xy = flow dir (world XZ), z = flow speed, w = detail wave strength
    float4 simParams;       // x = sim texel world size, y = sim normal strength,
                            // z = displacement scale, w = fogHeightFalloff
    float4 misc;            // x = refraction strength (uv), y = reflection roughness,
                            // z = ripple foam gain, w = reflection strength
    float4x4 viewProjSSR;   // world -> clip for the SSR march (this frame's draw VP)
    float4 ssrParams;       // x = strength (0 = off), y = max distance, z = thickness
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]] Texture2D sceneColor : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]] SamplerState sceneColorSampler : register(s0, space2);
[[vk::combinedImageSampler]][[vk::binding(1, 2)]] Texture2D<float> sceneDepth : register(t1, space2);
[[vk::combinedImageSampler]][[vk::binding(1, 2)]] SamplerState sceneDepthSampler : register(s1, space2);
[[vk::combinedImageSampler]][[vk::binding(2, 2)]] Texture2D<float2> simState : register(t2, space2);
[[vk::combinedImageSampler]][[vk::binding(2, 2)]] SamplerState simSampler : register(s2, space2);
[[vk::combinedImageSampler]][[vk::binding(3, 2)]] TextureCube iblSpecular : register(t3, space2);
[[vk::combinedImageSampler]][[vk::binding(3, 2)]] SamplerState iblSpecularSampler : register(s3, space2);

#include "fog.hlsli"

struct PSInput {
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 simUVFade : TEXCOORD1;
};

// Matches the composite/GTAO convention (glm RH_ZO reversed-Z).
float linearizeDepth(float depth) {
    return abs(screenDepthAB.w / (depth + screenDepthAB.z));
}

// Analytic detail waves: a few directional gradients scrolling with the
// flow. Returns the accumulated d(height)/d(x,z), normalized to ~unit
// amplitude — the caller's strength parameter IS the max surface slope, so
// keep it small (real water micro-slope is a few degrees, not tens).
float2 detailWaveGradient(float2 p, float2 flow, float t) {
    float2 grad = 0.0;
    // Divergent directions + irrational-ish wavelength/speed ratios so the
    // octaves never phase-lock into a visible lattice.
    float2 d0 = normalize(flow + float2(0.0, 1e-3));
    float2 d0b = normalize(d0 + float2(0.22, 0.13));
    float2 d1 = normalize(d0 + float2(0.83, -0.41));
    float2 d2 = normalize(d0 + float2(-0.71, 0.52));
    // The primary swell is TWO nearby wavelengths a few degrees apart: they
    // beat against each other, so no single crest spacing ever reads as a
    // repeating band (one frequency alone looks like corduroy).
    float ph0a = dot(p, d0) * (6.2831 / 131.0) + t * 1.31;
    float ph0b = dot(p, d0b) * (6.2831 / 89.0) + t * 1.03;
    float ph1 = dot(p, d1) * (6.2831 / 57.0) + t * 2.17;
    float ph2 = dot(p, d2) * (6.2831 / 26.0) + t * 3.71;
    // A slow cross-flow envelope modulates the smaller octaves so the
    // pattern drifts and breaks up instead of tiling.
    float envelope = 0.6 + 0.4 * sin(dot(p, float2(-d0.y, d0.x)) * (6.2831 / 217.0) + t * 0.53);
    grad += d0 * cos(ph0a) * 0.34;
    grad += d0b * cos(ph0b) * 0.27;
    grad += d1 * cos(ph1) * 0.25 * envelope;
    grad += d2 * cos(ph2) * 0.14 * envelope;
    return grad;
}

// Screen-space reflections: march the reflected ray against the scene depth
// copy (snapshotted BEFORE the water drew, so the surface can never self-hit)
// and sample the scene color copy on a hit. Geometric step growth keeps near
// reflections crisp and far ones cheap; `confidence` fades at screen edges
// and toward the ray's end so misses dissolve into the sky fallback.
float3 marchSSR(float3 origin, float3 R, out float confidence) {
    confidence = 0.0;
    const int STEPS = 24;
    const float T0 = 14.0;
    float maxDist = max(ssrParams.y, T0 * 2.0);
    float growth = pow(maxDist / T0, 1.0 / (float)STEPS);
    float t = T0;
    float prevT = 0.0;
    [loop]
    for (int i = 0; i < STEPS; ++i) {
        float3 p = origin + R * t;
        float4 clip = mul(viewProjSSR, float4(p, 1.0));
        if (clip.w < 1.0) {
            break; // marched behind the camera
        }
        float2 uv = clip.xy / clip.w * float2(0.5, -0.5) + 0.5;
        if (any(uv < 0.0) || any(uv > 1.0)) {
            break; // left the screen — nothing to sample
        }
        float sceneLin = linearizeDepth(sceneDepth.SampleLevel(sceneDepthSampler, uv, 0));
        float thickness = ssrParams.z + t * 0.05;
        if (clip.w > sceneLin + 1.0 && clip.w - sceneLin < thickness) {
            // One midpoint refinement between the last miss and this hit.
            float3 pm = origin + R * (0.5 * (prevT + t));
            float4 clipm = mul(viewProjSSR, float4(pm, 1.0));
            float2 uvm = clipm.xy / clipm.w * float2(0.5, -0.5) + 0.5;
            if (all(uvm >= 0.0) && all(uvm <= 1.0)) {
                float sceneLinM = linearizeDepth(sceneDepth.SampleLevel(sceneDepthSampler, uvm, 0));
                if (clipm.w > sceneLinM + 1.0) {
                    uv = uvm;
                }
            }
            float2 e = min(uv, 1.0 - uv);
            float edgeFade = saturate(min(e.x, e.y) * 8.0);
            float distFade = saturate(2.0 * (1.0 - t / maxDist));
            confidence = edgeFade * distFade * saturate(ssrParams.x);
            return sceneColor.SampleLevel(sceneColorSampler, uv, 0).rgb;
        }
        prevT = t;
        t *= growth;
    }
    return 0.0;
}

float4 PSMain(PSInput input) : SV_Target0 {
    float3 camPos = cameraPosTime.xyz;
    float time = cameraPosTime.w;
    float2 screenUV = input.position.xy * screenDepthAB.xy;

    // ---- water normal --------------------------------------------------
    float simFade = input.simUVFade.z;
    float2 simUV = input.simUVFade.xy;
    // Central differences of the sim height field (world-units slope). The
    // sim state is square, so one uv texel = 1/resolution in both axes.
    float simTexelWorld = simParams.x;
    float uvEps = 1.0 / 512.0;
    float hl = simState.SampleLevel(simSampler, saturate(simUV - float2(uvEps, 0)), 0).x;
    float hr = simState.SampleLevel(simSampler, saturate(simUV + float2(uvEps, 0)), 0).x;
    float hd = simState.SampleLevel(simSampler, saturate(simUV - float2(0, uvEps)), 0).x;
    float hu = simState.SampleLevel(simSampler, saturate(simUV + float2(0, uvEps)), 0).x;
    float hC = simState.SampleLevel(simSampler, saturate(simUV), 0).x;
    float2 simGrad = float2(hr - hl, hu - hd) * simParams.z / max(2.0 * simTexelWorld, 1e-3);

    float2 flow = flowDetail.xy;
    float2 detail = detailWaveGradient(input.worldPos.xz - flow * flowDetail.z * time, flow, time)
                    * flowDetail.w;

    float3 N = normalize(float3(-(simGrad.x * simParams.y * simFade + detail.x), 1.0,
                                -(simGrad.y * simParams.y * simFade + detail.y)));

    // ---- depths ---------------------------------------------------------
    float pixelLin = linearizeDepth(input.position.z);
    float sceneLin = linearizeDepth(sceneDepth.SampleLevel(sceneDepthSampler, screenUV, 0));
    float waterDepth = max(sceneLin - pixelLin, 0.0); // view-space thickness behind the surface

    // ---- refraction -----------------------------------------------------
    // Distort less in the shallows so the bed doesn't smear at the shoreline,
    // and reject samples that land on geometry IN FRONT of the surface.
    float refrScale = misc.x * saturate(waterDepth / 60.0);
    float2 refrUV = screenUV + N.xz * refrScale;
    float refrSceneLin = linearizeDepth(sceneDepth.SampleLevel(sceneDepthSampler, refrUV, 0));
    if (refrSceneLin < pixelLin) {
        refrUV = screenUV;
        refrSceneLin = sceneLin;
    }
    float3 behind = sceneColor.SampleLevel(sceneColorSampler, refrUV, 0).rgb;
    float refrDepth = max(refrSceneLin - pixelLin, 0.0);

    // Beer-Lambert absorption toward the deep tint (kept gentle: the bed
    // should stay readable through a metre of water).
    float absorb = 1.0 - exp(-refrDepth * deepColor.w);
    float3 refraction = lerp(behind * lerp(float3(1, 1, 1), shallowColor.rgb, absorb * 0.35),
                             deepColor.rgb, absorb);

    // ---- reflection -----------------------------------------------------
    float3 V = normalize(camPos - input.worldPos);
    float3 R = reflect(-V, N);
    R.y = abs(R.y); // the prefiltered sky has no below-horizon content
    // 7 prefiltered mips; roughness in misc.y.
    float3 skyRefl = iblSpecular.SampleLevel(iblSpecularSampler, R, misc.y * 6.0).rgb;
    // Hemisphere-ish fallback when IBL is off: fog horizon color.
    skyRefl = lerp(fogHorizon.rgb * 1.2, skyRefl, sunColorIbl.w);
    // Artistic dimmer: the HDR sky is far brighter than the refracted bed,
    // so even a few percent fresnel washes the water milky without it.
    skyRefl *= misc.w;
    // Screen-space reflections of actual geometry (banks, trees, buildings,
    // the player) layered over the sky by hit confidence. The wave normal
    // already perturbs R, so reflections shimmer with the surface for free.
    // Lift the origin slightly so grazing rays clear the bank silhouettes.
    float ssrConfidence;
    float3 ssrColor = marchSSR(input.worldPos + float3(0.0, 2.0, 0.0), R, ssrConfidence);
    float3 reflection = lerp(skyRefl, ssrColor, ssrConfidence);

    // Sun glint: tight Blinn lobe over the perturbed normal, Fresnel-scaled
    // at the half-vector so a top-down view doesn't sparkle.
    float3 L = -normalize(sunDirIntensity.xyz);
    float3 H = normalize(V + L);
    float fresnelH = 0.02 + 0.98 * pow(1.0 - saturate(dot(V, H)), 5.0);
    float glint = pow(saturate(dot(N, H)), 280.0) * sunDirIntensity.w * 40.0 * fresnelH;
    float3 sunSpec = sunColorIbl.rgb * glint;

    // Fresnel (Schlick, F0 = 0.02).
    float fresnel = 0.02 + 0.98 * pow(1.0 - saturate(dot(N, V)), 5.0);

    float3 color = lerp(refraction, reflection, fresnel) + sunSpec;

    // ---- foam -----------------------------------------------------------
    // Shoreline band, softly modulated so the line isn't a hard contour, plus
    // crest foam where the sim field is energetic (wakes, splashes).
    float shore = 1.0 - saturate(waterDepth / max(shallowColor.w, 1e-3));
    float foamWave = 0.65 + 0.35 * sin(waterDepth * 0.35 - time * 1.8 + (detail.x + detail.y) * 24.0);
    float foam = smoothstep(0.55, 0.95, shore * foamWave);
    foam += saturate(abs(hC) * simParams.z * misc.z) * simFade;
    float3 foamColor = float3(0.9, 0.93, 0.95) * (0.35 + 0.65 * sunDirIntensity.w * 0.33);
    color = lerp(color, foamColor, saturate(foam) * 0.8);

    // ---- waterline blend + fog -------------------------------------------
    // Feather the first few units of depth so the surface meets the banks
    // without a hard aliased edge.
    float edgeBlend = saturate(waterDepth / 7.0);
    float3 sceneAtPixel = sceneColor.SampleLevel(sceneColorSampler, screenUV, 0).rgb;
    color = lerp(sceneAtPixel, color, edgeBlend);

    color = applyAtmosphericFog(color, input.worldPos, camPos, sunDirIntensity.xyz,
                                fogHorizon.rgb, fogSun.rgb, fogHorizon.w, fogSun.w, simParams.w);

    return float4(color, 1.0);
}
