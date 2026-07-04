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

#include "shadow_common.hlsli"
#include "fog.hlsli"
#include "lighting.hlsli"
// Cluster SSBOs follow this shader's 6 fragment samplers in set 2.
#define CLUSTER_SSBO_BINDING_BASE 6
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
        baseColor = input.vertexColor * input.instanceTint;
    }
    // glTF alphaMode: the CPU sends alphaCutoff > 0 only for MASK materials,
    // so OPAQUE/BLEND draws never take the discard.
    if (baseColor.a < alphaCutoff) discard;

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

    float ambientLift = lerp(ambientStrength, 1.0, shadow);
    float3 ambient = ambientPBR(N, V, baseColor.rgb, metallic, roughness, matAO,
                                ambientSky * ambientLift, ambientGround * ambientLift);

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
