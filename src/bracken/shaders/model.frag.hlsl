// Model Fragment Shader (shared: static + skinned).
// Linear HDR output. baseColor texture is sampled sRGB-aware (UNORM_SRGB format),
// so values are already linearized by the sampler. Vertex colors are assumed linear.

struct PSInput {
    float4 position : SV_Position;
    float3 fragPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 texCoord : TEXCOORD2;
    float4 vertexColor : TEXCOORD3;
    float viewDepth : TEXCOORD4;
    float4 tangent : TEXCOORD5;  // xyz = world-space tangent, w = bitangent sign
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
    float4 tintColor;
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
    // Ray-traced sun shadow mask (rtShadowTexture, fragment sampler 10).
    // 0 = off, and the CSM result is used unchanged. It reuses screenAOTexel*
    // for its UV: SV_Position.xy * (1 / render-target size) is a NORMALIZED
    // full-screen coordinate, so a half- or quarter-resolution mask samples
    // correctly and the linear sampler magnifies it for free.
    float rtShadowEnabled;
    // The traced-reflection handover window. The IBL specular lobe is scaled
    // by 1 - rtReflectionWeight(roughness, fadeStart, maxRoughness) and the
    // compute pass writes that same weight into its alpha, so the two sum to
    // one. 0 = reflections off, probe used as normal.
    float rtReflectionMaxRoughness;
    // Roughness at which the traced contribution starts fading out; it reaches
    // zero at rtReflectionMaxRoughness. See reflection_gate.hlsli.
    float rtReflectionFadeStart;
    float _rtPad2;
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
// from the prepass depth. Bound to a 1x1 white dummy when AO is off, which is
// why the white dummy has to be white: this is sampled unconditionally
// wherever screenAOEnabled is set.
//
// This slot USED to carry the traced sun shadow in G as well, because the
// stage was believed to be at an 11-sampler DXC ceiling. It is not (12
// compiles; 17 compiles), and the packing was actively harmful: it forced the
// RT pass to copy AO through with a texel coordinate derived from a different
// texture's dimensions, which read out of bounds -- returning 0 -- and zeroed
// the ambient term scene-wide whenever AO was off. See rtShadowTexture below.
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


// Ray-traced sun visibility (R8, 1 = lit), at the traced pass's own
// resolution -- half res by default, so ALWAYS sample it with a normalized UV
// and never with a texel coordinate borrowed from another texture. Bound to
// the 1x1 white dummy when ray tracing is off, and read only when
// rtShadowEnabled: a white dummy then reads 1.0 = "fully lit" from any UV,
// which is the correct no-op.
[[vk::combinedImageSampler]][[vk::binding(10, 2)]]
Texture2D<float> rtShadowTexture;
[[vk::combinedImageSampler]][[vk::binding(10, 2)]]
SamplerState rtShadowSampler;

// Local (point/spot) shadow atlas + comparison sampler: declared by
// cluster_lighting.hlsli at this shader's LAST sampler slot (11).

#include "shadow_common.hlsli"
#include "fog.hlsli"
#include "lighting.hlsli"
#include "pbr.hlsli"
#include "reflection_gate.hlsli"
// Cluster SSBOs follow this shader's 12 fragment samplers in set 2 (slots
// 0-10 above + the local-shadow atlas at 11, declared by the include).
#define CLUSTER_SSBO_BINDING_BASE 12
#include "cluster_lighting.hlsli"

float4 PSMain(PSInput input) : SV_Target {
    float3 N_geo = normalize(input.normal);
    float3 V = normalize(cameraPos - input.fragPos);
    // lightDir is the direction the light travels; L is surface->light.
    float3 L = normalize(-lightDir);

    // Tangent-space normal mapping. Without a normal map the geometric normal
    // is used directly; the 1x1 flat texture bound in that case only keeps the
    // sampler slot valid for SDL3 GPU and is never sampled here.
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
        baseColor = baseColorTexture.Sample(baseColorSampler, input.texCoord) * baseColorFactor * tintColor;
    } else {
        // glTF: baseColorFactor multiplies COLOR_0 (white-baked when absent).
        baseColor = input.vertexColor * baseColorFactor * tintColor;
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
    // Widen roughness where the surface normal varies fast to kill specular shimmer.
    roughness = specularAARoughness(N, roughness);

    float shadow = calcShadow(input.fragPos, N_geo, L, input.viewDepth, input.position.xy);

    // Ray-traced shadows REPLACE the cascade result rather than multiplying
    // with it: multiplying would double-darken everywhere the two agree, and
    // tracing exists because it is the more accurate answer. The cascades stay
    // as the fallback when it is off.
    if (rtShadowEnabled > 0.5) {
        // Normalized UV into the mask's OWN resolution: screenAOTexel* is
        // 1/screen size, and the mask's sampler does the rest. This is what
        // makes half- or quarter-res masks a free parameter.
        float2 rtUV = input.position.xy * float2(screenAOTexelX, screenAOTexelY);
        shadow = rtShadowTexture.SampleLevel(rtShadowSampler, rtUV, 0).r;
    }


    // Direct lighting: full PBR BRDF (material AO only occludes ambient).
    float3 directLight = lightColor * sunIntensity * shadow;
    float3 direct = evaluatePBR(N, V, L, directLight, baseColor.rgb, metallic, roughness, 1.0);

    // Screen-space GTAO, ambient only: min-combined with the material (ORM)
    // AO — both estimate the same occlusion at overlapping scales, min avoids
    // double-darkening. Direct sun light, speculars, and cluster lights are
    // NEVER attenuated by screen-space AO.
    float screenAO = 1.0;
    if (screenAOEnabled > 0.5) {
        screenAO = screenAOTexture.Sample(
            screenAOSampler, input.position.xy * float2(screenAOTexelX, screenAOTexelY)).r;
    }
    float aoCombined = min(matAO, screenAO);

    // See rtReflectionMaxRoughness: hand the specular lobe over to the traced
    // reflection rather than adding both.
    // ONE function, shared with rt_reflection.comp.hlsl, so traced + probe sum
    // to exactly 1 at every pixel: the trace writes this weight into its alpha
    // and the composite adds `rtSpec.rgb * rtSpec.a`. `roughness` here is the
    // MR-modulated, specular-AA-widened value, which is exactly what the
    // prepass G-buffer now writes — the two used to disagree because the
    // G-buffer wrote the raw material factor.
    float iblSpecularScale =
        1.0 - rtReflectionWeight(roughness, rtReflectionFadeStart, rtReflectionMaxRoughness);

    // Ambient. Shadow dims ambient via ambientLift to avoid a flat fill-light
    // look, matching the previous art direction. With IBL on, the hemisphere
    // terms are replaced by the sky prefilter chain; the same combined AO and
    // lift multiply either path.
    float ambientLift = lerp(ambientStrength, 1.0, shadow);
    float3 ambient;
    if (iblEnabled > 0.5) {
        ambient = ambientIBL(N, N_geo, V, baseColor.rgb, metallic, roughness, aoCombined, ambientLift,
                             iblIrradianceMap, iblIrradianceSampler, iblSpecularMap, iblSpecularSampler,
                             iblBrdfLut, iblBrdfLutSampler, iblSpecularScale);
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

    if (fogEnabled == 1) {
        result = applyAtmosphericFog(result, input.fragPos, cameraPos, lightDir,
                                     fogColor, fogSunColor, fogStart, fogEnd, fogHeightFalloff);
    }

    return float4(result, baseColor.a);
}
