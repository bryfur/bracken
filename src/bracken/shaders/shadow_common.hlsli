/**
 * Common Shadow Functions - Cascaded Shadow Map atlas (hardware PCF + PCSS)
 *
 * All cascades live in one depth atlas laid out as a 2x2 quadrant grid;
 * cascade i occupies quadrant (i%2, i/2), each shadowMapResolution texels.
 *
 * Shared by all fragment shaders that receive shadows. Include AFTER declaring:
 *  - ShadowUniforms cbuffer: shadowLightViewProjection[4], cascadeSplits,
 *    cascadeTexelWorld, cascadeDepthRange, shadowMapResolution, lightSize,
 *    shadowEnabled, lastActiveSplit, fadeStart, fadeEnd.
 *  - shadowAtlas + shadowCmpSampler (SamplerComparisonState, LESS_EQUAL,
 *    linear: each SampleCmpLevelZero tap is a free 2x2 hardware PCF).
 *  - shadowAtlasRaw + shadowRawSampler (SamplerState, nearest) for the PCSS
 *    blocker search.
 */

#include "lighting.hlsli" // interleavedGradientNoise (per-pixel filter rotation/dither)

static const float2 shadowPoissonDisk[12] = {
    float2(-0.326212, -0.405805), float2(-0.840144, -0.073580),
    float2(-0.695914,  0.457137), float2(-0.203345,  0.620716),
    float2( 0.962340, -0.194983), float2( 0.473434, -0.480026),
    float2( 0.519456,  0.767022), float2( 0.185461, -0.893124),
    float2( 0.507431,  0.064425), float2( 0.896420,  0.412458),
    float2(-0.321940, -0.932615), float2(-0.791559, -0.597705)
};

// |shadowPoissonDisk[j]| for the first 8 entries — the ones the PCSS blocker
// search taps. It needs each tap's radius to size that tap's receiver-plane
// rejection; the 12-tap compare loop below does not.
static const float shadowPoissonRadius[8] = {
    0.520665, 0.843360, 0.832628, 0.653175,
    0.981894, 0.674214, 0.926368, 0.912177
};

// Widest penumbra the PCSS filter may produce, in shadow-map texels. The
// blocker search uses the SAME radius by construction: a blocker the search
// cannot see is a penumbra the filter must not be asked to draw.
static const float PCSS_MAX_PENUMBRA_TEXELS = 16.0;

int selectShadowCascade(float depth) {
    if (depth < cascadeSplits.x) return 0;
    if (depth < cascadeSplits.y) return 1;
    if (depth < cascadeSplits.z) return 2;
    return 3;
}

// Visibility [0,1] for one cascade. screenPos drives the per-pixel poisson rotation.
// filterMode: 0 = single hardware-PCF tap (cheap path for fragments whose sun
// contribution is already known to be zero), 1 = 3x3 PCF tent, 2 = PCSS.
float sampleShadowCascade(int cascade, float3 worldPos, float3 N, float NdotL, float2 screenPos,
                          float filterMode) {
    float texelWorld = cascadeTexelWorld[cascade];
    float depthRange = cascadeDepthRange[cascade];

    // Normal-offset bias: push the receiver one texel along the surface
    // normal so the projected sample lands off its own depth ramp.
    float4 lightSpacePos = mul(shadowLightViewProjection[cascade], float4(worldPos + N * texelWorld, 1.0));
    float3 sc = lightSpacePos.xyz / lightSpacePos.w;
    sc.x = sc.x * 0.5 + 0.5;
    sc.y = 1.0 - (sc.y * 0.5 + 0.5);

    if (sc.x < 0.0 || sc.x > 1.0 ||
        sc.y < 0.0 || sc.y > 1.0 ||
        sc.z < 0.0 || sc.z > 1.0)
        return 1.0;

    // Slope-scaled receiver bias in light-depth units. texelWorld/depthRange
    // converts one world texel to this cascade's [0,1] depth range, so the
    // bias is exactly as large as the geometry it must hide - no peter-panning.
    float tanTheta = min(sqrt(1.0 - NdotL * NdotL) / max(NdotL, 1e-3), 3.0);
    float refZ = sc.z - (0.5 + 1.5 * tanTheta) * texelWorld / depthRange;

    // Atlas addressing: quadrant origin + half-scale UV. Every tap clamps to
    // one texel inside the quadrant so filtering never bleeds across cascades.
    float atlasTexel = 0.5 / shadowMapResolution;
    float2 quadOrigin = float2((cascade & 1) * 0.5, (cascade >> 1) * 0.5);
    float2 uv = quadOrigin + sc.xy * 0.5;
    float2 uvMin = quadOrigin + atlasTexel;
    float2 uvMax = quadOrigin + 0.5 - atlasTexel;

    // Single tap (mode 0): one hardware-PCF compare (free 2x2 filter).
    if (filterMode < 0.5) {
        return shadowAtlas.SampleCmpLevelZero(shadowCmpSampler, clamp(uv, uvMin, uvMax), refZ);
    }

    // PCF (mode 1): 3x3 tent of hardware-PCF compares
    if (filterMode < 1.5) {
        static const float tentWeights[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
        float shadow = 0.0;
        [unroll]
        for (int i = 0; i < 9; i++) {
            float2 offset = float2(i % 3 - 1, i / 3 - 1) * atlasTexel;
            shadow += tentWeights[i] * shadowAtlas.SampleCmpLevelZero(shadowCmpSampler, clamp(uv + offset, uvMin, uvMax), refZ);
        }
        return shadow / 16.0;
    }

    // PCSS (mode 2): poisson disk rotated per pixel to trade banding for noise
    float angle = interleavedGradientNoise(screenPos) * 6.2831853;
    float sinA = sin(angle);
    float cosA = cos(angle);
    float2x2 rot = float2x2(cosA, -sinA, sinA, cosA);

    // Blocker search (raw taps) at the filter's ceiling. The textbook width -
    // tan(sun angle) * light-space distance to the near plane - is unusable
    // here: this cascade's near plane is a CASTER-REACH plane sitting
    // radius + CASTER_HEIGHT_BUDGET / max(|light_dir.y|, MIN_SUN_ELEVATION_SIN)
    // up-sun (render/shadow_map.cpp), i.e. never less than 1500 world units
    // above any receiver, so the bound comes out at 40-200 texels in cascade 0
    // and saturates any clamp it could be given.
    float searchTexels = PCSS_MAX_PENUMBRA_TEXELS;
    float blockerSum = 0.0;
    int blockerCount = 0;
    [unroll]
    for (int j = 0; j < 8; j++) {
        float2 tapUV = clamp(uv + mul(rot, shadowPoissonDisk[j]) * searchTexels * atlasTexel, uvMin, uvMax);
        float d = shadowAtlasRaw.SampleLevel(shadowRawSampler, tapUV, 0).r;
        // A tap r texels out reads the RECEIVER'S OWN plane r*tanTheta texels
        // shallower, and refZ's 1.5-texel slope term is sized for the 3x3 tent
        // above, not for a 16-texel search. Without a per-tap threshold every
        // sloped lit pixel counts itself as its own blocker, the early-out
        // below never fires, and the self-taps (the deepest samples in the
        // set) drag avgBlocker toward the receiver and narrow every penumbra.
        // Every radius here is >= 8.3 texels, so this is always looser than
        // refZ - it can only reject the receiver's own plane, never a blocker
        // refZ would have caught.
        float tapRef = sc.z - (0.5 + shadowPoissonRadius[j] * searchTexels * tanTheta)
                              * texelWorld / depthRange;
        if (d < tapRef) {
            blockerSum += d;
            blockerCount++;
        }
    }
    if (blockerCount == 0) {
        // Nothing outside the receiver's own plane band: at most a sub-texel
        // penumbra, which the 12-tap loop would clamp to a 1-texel filter
        // anyway. One hardware-PCF compare still keeps the contact shadows the
        // plane rejection cannot tell apart from the receiver; returning 1.0
        // outright would peter-pan them.
        return shadowAtlas.SampleCmpLevelZero(shadowCmpSampler, clamp(uv, uvMin, uvMax), refZ);
    }

    // Directional-light penumbra: width grows with the world-space gap
    // between receiver and average blocker times tan(sun angular radius).
    float avgBlocker = blockerSum / float(blockerCount);
    float penumbraTexels = (sc.z - avgBlocker) * depthRange * lightSize / texelWorld;
    float filterTexels = clamp(penumbraTexels, 1.0, PCSS_MAX_PENUMBRA_TEXELS);

    float shadow = 0.0;
    [unroll]
    for (int k = 0; k < 12; k++) {
        float2 tapUV = clamp(uv + mul(rot, shadowPoissonDisk[k]) * filterTexels * atlasTexel, uvMin, uvMax);
        shadow += shadowAtlas.SampleCmpLevelZero(shadowCmpSampler, tapUV, refZ);
    }
    return shadow / 12.0;
}

// returns raw [0,1] visibility, ambient applied by caller.
// N = geometric surface normal, L = surface-to-light direction,
// screenPos = SV_Position.xy (per-pixel filter rotation).
float calcShadow(float3 worldPos, float3 N, float3 L, float viewDepth, float2 screenPos) {
    if (shadowEnabled < 0.5) return 1.0;
    if (viewDepth >= lastActiveSplit) return 1.0;

    int cascade = selectShadowCascade(viewDepth);
    float rawNdotL = dot(N, L);
    float NdotL = saturate(rawNdotL);

    // Distance fade: shadows dissolve over the tail of the shadow range
    // instead of popping off at lastActiveSplit.
    float fade = saturate((viewDepth - fadeStart) / max(fadeEnd - fadeStart, 1e-4));

    // Back-facing fragments get no direct sun anyway (evaluatePBR rejects
    // NdotL <= 0); the result only nudges the caller's ambientLift (and the
    // faint wrap-lit tail in grass.frag), so a single tap replaces the full
    // filter (saves up to 20 atlas taps on roughly half of all geometry).
    if (rawNdotL <= 0.0) {
        float shadow = sampleShadowCascade(cascade, worldPos, N, NdotL, screenPos, 0.0);
        return lerp(shadow, 1.0, fade);
    }

    // Blend into the next cascade over the last 15% of this cascade's range
    // so the resolution step never shows as a hard seam.
    float splitFar = cascadeSplits[cascade];
    float splitNear = (cascade == 0) ? 0.0 : cascadeSplits[cascade - 1];
    float blendStart = splitFar - 0.15 * (splitFar - splitNear);
    float blend = 0.0;
    if (viewDepth > blendStart && splitFar < lastActiveSplit) {
        blend = saturate((viewDepth - blendStart) / max(splitFar - blendStart, 1e-4));
    }

    float shadow;
    if (shadowEnabled >= 1.5) {
        // PCSS: dither the blend with IGN and filter one cascade per pixel
        // instead of two — the band would otherwise double the PCSS cost
        // (2 x (8 blocker + 12 filter) taps). The offset decorrelates the
        // dither from the poisson rotation angle above.
        if (blend > 0.0 && interleavedGradientNoise(screenPos + float2(17.0, 59.0)) < blend) {
            cascade += 1;
        }
        shadow = sampleShadowCascade(cascade, worldPos, N, NdotL, screenPos, shadowEnabled);
    } else {
        // PCF is cheap enough to sample both cascades and lerp.
        shadow = sampleShadowCascade(cascade, worldPos, N, NdotL, screenPos, shadowEnabled);
        if (blend > 0.0) {
            float next = sampleShadowCascade(cascade + 1, worldPos, N, NdotL, screenPos, shadowEnabled);
            shadow = lerp(shadow, next, blend);
        }
    }

    return lerp(shadow, 1.0, fade);
}
