// Instanced Model Fragment Shader.
// Linear HDR output, sRGB sampler on base color. Per-instance tint is linear.

struct PSInput {
    float4 position : SV_Position;
    float3 fragPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 texCoord : TEXCOORD2;
    float4 vertexColor : TEXCOORD3;
    float viewDepth : TEXCOORD4;
    float4 instanceTint : TEXCOORD5;
    float noFog : TEXCOORD6;
    float4 tangent : TEXCOORD7;  // xyz = world-space tangent, w = bitangent sign
};

[[vk::binding(0, 3)]]
cbuffer LightingUniforms {
    float3 lightDir;
    float _padding0;
    float3 lightColor;
    float _padding1;
    float3 ambientSky;
    float _padding2;
    float3 ambientGround;
    float _padding3;
    float4 baseColorFactor;
    float3 fogColor;        // horizon fog color
    float fogStart;
    float3 fogSunColor;
    float fogEnd;
    float3 emissiveFactor;
    float fogHeightFalloff;
    float3 cameraPos;
    float normalScale;
    int hasTexture;
    int fogEnabled;
    int hasNormalMap;
    float ambientStrength;
    float sunIntensity;
    float metallicFactor;
    float roughnessFactor;
    float occlusionStrength;
    float hasMRMap;
    float hasEmissiveMap;
    float hasPackedOcclusion;
    float alphaCutoff;      // glTF MASK cutoff; 0 disables alpha test (OPAQUE/BLEND)
    float screenAOTexelX;   // 1 / render-target size: SV_Position.xy * texel = AO UV
    float screenAOTexelY;
    float iblEnabled;       // 1 = sky IBL ambient (samplers 7-9), 0 = hemisphere fallback
    float screenAOEnabled;  // 0 for MASK/BLEND: not in the depth prepass, so the AO
                            // texel under them belongs to the geometry behind them
};

[[vk::binding(1, 3)]]
cbuffer ShadowUniforms {
    float4x4 shadowLightViewProjection[4];
    float4 cascadeSplits;
    float4 cascadeTexelWorld;
    float4 cascadeDepthRange;
    float shadowMapResolution;
    float lightSize;
    float shadowEnabled;
    float lastActiveSplit;
    float fadeStart;
    float fadeEnd;
    float _shadowPad0;
    float _shadowPad1;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D baseColorTexture;
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState baseColorSampler;

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D shadowAtlas;
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerComparisonState shadowCmpSampler;

[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
Texture2D shadowAtlasRaw;
[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
SamplerState shadowRawSampler;

[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
Texture2D normalMapTexture;
[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
SamplerState normalMapSampler;

[[vk::combinedImageSampler]][[vk::binding(4, 2)]]
Texture2D metallicRoughnessTexture;
[[vk::combinedImageSampler]][[vk::binding(4, 2)]]
SamplerState metallicRoughnessSampler;

[[vk::combinedImageSampler]][[vk::binding(5, 2)]]
Texture2D emissiveTexture;
[[vk::combinedImageSampler]][[vk::binding(5, 2)]]
SamplerState emissiveSampler;

// Full-res screen-space GTAO (R8, 1 = unoccluded), computed before this pass
// from the prepass depth. Bound to a 1x1 white dummy when AO is off.
[[vk::combinedImageSampler]][[vk::binding(6, 2)]]
Texture2D<float> screenAOTexture;
[[vk::combinedImageSampler]][[vk::binding(6, 2)]]
SamplerState screenAOSampler;

// Sky IBL chain (render/ibl.hpp): irradiance cube + GGX-prefiltered specular
// cube + split-sum BRDF LUT. Bound to dummies when iblEnabled == 0.
[[vk::combinedImageSampler]][[vk::binding(7, 2)]]
TextureCube<float4> iblIrradianceMap;
[[vk::combinedImageSampler]][[vk::binding(7, 2)]]
SamplerState iblIrradianceSampler;

[[vk::combinedImageSampler]][[vk::binding(8, 2)]]
TextureCube<float4> iblSpecularMap;
[[vk::combinedImageSampler]][[vk::binding(8, 2)]]
SamplerState iblSpecularSampler;

[[vk::combinedImageSampler]][[vk::binding(9, 2)]]
Texture2D<float4> iblBrdfLut;
[[vk::combinedImageSampler]][[vk::binding(9, 2)]]
SamplerState iblBrdfLutSampler;

#include "shadow_common.hlsli"
#include "fog.hlsli"
#include "lighting.hlsli"
#include "pbr.hlsli"
// Cluster SSBOs follow this shader's 11 fragment samplers in set 2 (slots
// 0-9 above + the local-shadow atlas at 10, declared by the include).
#define CLUSTER_SSBO_BINDING_BASE 11
#include "cluster_lighting.hlsli"

float4 PSMain(PSInput input) : SV_Target {
    float3 N_geo = normalize(input.normal);
    float3 V = normalize(cameraPos - input.fragPos);
    float3 L = normalize(-lightDir);

    float3 N = N_geo;
    if (hasNormalMap == 1) {
        float3 Nts = normalMapTexture.Sample(normalMapSampler, input.texCoord).xyz * 2.0 - 1.0;
        Nts.xy *= normalScale;
        float3 T_ws = normalize(input.tangent.xyz);
        float3 B_ws = cross(N_geo, T_ws) * input.tangent.w;
        N = normalize(T_ws * Nts.x + B_ws * Nts.y + N_geo * Nts.z);
    }

    float4 baseColor;
    if (hasTexture == 1) {
        baseColor = baseColorTexture.Sample(baseColorSampler, input.texCoord) * baseColorFactor * input.instanceTint;
    } else {
        // glTF: baseColorFactor multiplies COLOR_0 (white-baked when absent).
        baseColor = input.vertexColor * baseColorFactor * input.instanceTint;
    }
    // glTF alphaMode: the CPU sends alphaCutoff > 0 only for MASK materials,
    // so OPAQUE/BLEND draws never take the discard. Alpha is scaled up with
    // the sampled mip level first: box-filtered mips AVERAGE alpha, so
    // minified alpha-tested foliage otherwise erodes into sub-pixel holes
    // that AA resolves into a see-through wash (the classic alpha-coverage
    // fix, e.g. Castano 2010).
    if (alphaCutoff > 0.0) {
        float mip = baseColorTexture.CalculateLevelOfDetailUnclamped(baseColorSampler, input.texCoord);
        baseColor.a *= 1.0 + max(mip, 0.0) * 0.35;
        if (baseColor.a < alphaCutoff) discard;
    }

    // Metallic-roughness: scalar factors modulated by the MR texture when present
    // (glTF convention: G = roughness, B = metallic, R = packed AO for ORM maps).
    float metallic = metallicFactor;
    float roughness = roughnessFactor;
    float matAO = 1.0;
    if (hasMRMap > 0.5) {
        float3 mrSample = metallicRoughnessTexture.Sample(metallicRoughnessSampler, input.texCoord).rgb;
        roughness *= mrSample.g;
        metallic *= mrSample.b;
        if (hasPackedOcclusion > 0.5) {
            matAO = lerp(1.0, mrSample.r, occlusionStrength);
        }
    }
    roughness = specularAARoughness(N, roughness);

    float shadow = calcShadow(input.fragPos, N_geo, L, input.viewDepth, input.position.xy);
    float3 directLight = lightColor * sunIntensity * shadow;

    float3 direct = evaluatePBR(N, V, L, directLight, baseColor.rgb, metallic, roughness, 1.0);

    // Screen-space GTAO, ambient only: min-combined with the material (ORM)
    // AO to avoid double-darkening. Direct light/speculars stay untouched.
    float screenAO = 1.0;
    if (screenAOEnabled > 0.5) {
        screenAO = screenAOTexture.Sample(
            screenAOSampler, input.position.xy * float2(screenAOTexelX, screenAOTexelY));
    }
    float aoCombined = min(matAO, screenAO);

    float ambientLift = lerp(ambientStrength, 1.0, shadow);
    float3 ambient;
    if (iblEnabled > 0.5) {
        ambient = ambientIBL(N, N_geo, V, baseColor.rgb, metallic, roughness, aoCombined, ambientLift,
                             iblIrradianceMap, iblIrradianceSampler, iblSpecularMap, iblSpecularSampler,
                             iblBrdfLut, iblBrdfLutSampler);
    } else {
        ambient = ambientPBR(N, V, baseColor.rgb, metallic, roughness, aoCombined,
                             ambientSky * ambientLift, ambientGround * ambientLift);
    }

    float3 result = direct + ambient;

    if (clusterParams.gridDim.w > 0u) {
        float2 screen_uv = input.position.xy * clusterParams.screenSize.zw;
        result += accumulateClusterLights(input.fragPos, N, V,
                                          baseColor.rgb, metallic, roughness, 1.0,
                                          input.viewDepth, screen_uv);
    }

    float3 emissive = emissiveFactor;
    if (hasEmissiveMap > 0.5) {
        emissive *= emissiveTexture.Sample(emissiveSampler, input.texCoord).rgb;
    }
    result += emissive;

    if (fogEnabled == 1 && input.noFog < 0.5) {
        result = applyAtmosphericFog(result, input.fragPos, cameraPos, lightDir,
                                     fogColor, fogSunColor, fogStart, fogEnd, fogHeightFalloff);
    }

    return float4(result, baseColor.a);
}
