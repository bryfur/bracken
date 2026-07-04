// Grass Fragment Shader.
// Linear HDR output. Grass color comes from vertex color (authored linear).
// Uses a simplified PBR call (high roughness, non-metallic) plus a cheap tip
// specular, hemisphere ambient via ambientPBR, and sky-coupled atmospheric fog
// so distant blades dissolve into the same fog as the terrain behind them.

struct PSInput {
    float4 position : SV_Position;
    float3 normal : TEXCOORD0;
    float2 texCoord : TEXCOORD1;
    float3 worldPos : TEXCOORD2;
    float4 color : TEXCOORD3;
    float viewDepth : TEXCOORD4;
};

[[vk::binding(0, 3)]]
cbuffer GrassLightingUniforms {
    float3 cameraPos;
    float fogStart;
    float3 fogColorHorizon;
    float fogEnd;
    float3 fogColorSun;
    float fogHeightFalloff;
    float3 lightDir;
    float ambientStrength;
    float3 sunColor;
    float sunIntensity;
    float3 ambientSky;
    float _pad0;
    float3 ambientGround;
    float _pad1;
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
Texture2D shadowAtlas;
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerComparisonState shadowCmpSampler;

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D shadowAtlasRaw;
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerState shadowRawSampler;

#include "shadow_common.hlsli"
#include "fog.hlsli"
#include "lighting.hlsli"
// Cluster SSBOs follow this shader's 2 fragment samplers in set 2.
#define CLUSTER_SSBO_BINDING_BASE 2
#include "cluster_lighting.hlsli"

static const float GRASS_METALLIC  = 0.0;
static const float GRASS_ROUGHNESS = 0.9;

float4 PSMain(PSInput input) : SV_Target {
    float3 baseColor = input.color.rgb;
    float3 N = normalize(input.normal);
    float3 V = normalize(cameraPos - input.worldPos);
    float3 L = normalize(-lightDir);

    float shadow = calcShadow(input.worldPos, N, L, input.viewDepth, input.position.xy);

    // Half-Lambert wraps soft shading around thin grass surfaces.
    float halfLambert = dot(N, L) * 0.5 + 0.5;
    float wrap = halfLambert * halfLambert;

    float3 lightColor = sunColor * sunIntensity * shadow;

    float3 direct = evaluatePBR(N, V, L, lightColor,
                                 baseColor, GRASS_METALLIC, GRASS_ROUGHNESS, 1.0);
    // Blend wrap-lit diffuse with the PBR result for softness near grazing,
    // which PBR's NdotL clamp otherwise kills.
    float3 softDiffuse = baseColor * lightColor * wrap * (1.0 / PI_PBR);
    direct = max(direct, softDiffuse);

    // Tip specular highlight preserved from previous art direction.
    float tipFactor = input.texCoord.y * input.texCoord.y;
    float3 halfVec = normalize(V + L);
    float spec = pow(max(dot(N, halfVec), 0.0), 32.0) * 0.3 * tipFactor;

    float ambientLift = lerp(ambientStrength, 1.0, shadow);
    float3 ambient = ambientPBR(N, V, baseColor, GRASS_METALLIC, GRASS_ROUGHNESS, 1.0,
                                ambientSky * ambientLift, ambientGround * ambientLift);

    float3 color = direct + spec * lightColor + ambient;

    if (clusterParams.gridDim.w > 0u) {
        float2 screen_uv = input.position.xy * clusterParams.screenSize.zw;
        color += accumulateClusterLights(input.worldPos, N, V,
                                         baseColor, GRASS_METALLIC, GRASS_ROUGHNESS, 1.0,
                                         input.viewDepth, screen_uv);
    }

    // Same atmospheric fog as terrain/models: blades dissolve into the scene
    // fog instead of a hardcoded color band. The VS LOD collapse handles the
    // view-distance boundary itself.
    color = applyAtmosphericFog(color, input.worldPos, cameraPos, lightDir,
                                fogColorHorizon, fogColorSun, fogStart, fogEnd, fogHeightFalloff);

    return float4(color, input.color.a);
}
