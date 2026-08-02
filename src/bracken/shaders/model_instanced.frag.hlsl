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
    // Ray-traced sun shadow mask (rtShadowTexture, its own sampler slot) —
    // see model.frag.hlsl. screenAOTexel* doubles as its UV scale.
    float rtShadowEnabled;
    // The traced-reflection handover window. The IBL specular lobe is scaled
    // by 1 - rtReflectionWeight(roughness, fadeStart, maxRoughness) and the
    // compute pass writes that same weight into its alpha, so the two sum to
    // one. 0 = reflections off, probe used as normal.
    float rtReflectionMaxRoughness;
    // Roughness at which the traced contribution starts fading out; it reaches
    // zero at rtReflectionMaxRoughness. See reflection_gate.hlsli.
    float rtReflectionFadeStart;
    // 1 = multiply the LOCAL (point/spot) subtotal by the ray-traced
    // local-shadow mask at sampler 14 (R2b Ratio tier). Was _rtPad2, so
    // the cbuffer neither grew nor moved a field.
    float rtLocalShadowEnabled;
    // Probe-field GI (render/gi/), R1c. giCascadeBase is
    // `render::gi::cascade_base()` of the position ProbeVolume was scrolled
    // with — passed rather than re-derived from cameraPos so a shader and the
    // CPU can never disagree about which probe answers for which point in the
    // world. giEnabled gates every tap; 0 restores the pre-R1c ambient exactly.
    int4 giCascadeBase[3];
    float giEnabled;
    // Screen-probe gather (R4b, render/gi/screen_probes.hpp). These three were
    // _giPad0.._giPad2, so the cbuffer neither grew nor moved a field.
    // giScreenEnabled gates every tap; 0 restores the pure probe-field ambient.
    // The grid arrives as floats because the pads were floats — the SH atlas's
    // addressing casts them back to ints in gi_screen_sample.hlsli.
    float giScreenEnabled;
    float giScreenProbesX;
    float giScreenProbesY;
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

// Full-res screen-space GTAO (R8, 1 = unoccluded); 1x1 white dummy when off.
// It used to carry the traced sun shadow in G too — see model.frag.hlsl for
// why that packing is gone.
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

// Ray-traced sun visibility (R8, 1 = lit) at the traced pass's own resolution
// — sample with a normalized UV, never a borrowed texel coordinate. White
// dummy when ray tracing is off; read only when rtShadowEnabled.
[[vk::combinedImageSampler]][[vk::binding(10, 2)]]
Texture2D<float> rtShadowTexture;
[[vk::combinedImageSampler]][[vk::binding(10, 2)]]
SamplerState rtShadowSampler;

// ---------------------------------------------------------------------------
// Probe-field GI (render/gi/) — the three atlases gi_sample.hlsli reads.
// ---------------------------------------------------------------------------
// Every cascade is BANDED into one texture each, which is why the field costs
// three sampler slots here and not nine (gi_math.hpp explains the trade). The
// first two are octahedral payloads and want a LINEAR clamp sampler — their
// one-texel border rings exist so a bilinear tap at a payload edge wraps the
// octahedron. The third is one texel per probe and is read with Load(), never
// filtered: a linear tap would blend two unrelated probes' relocation offsets
// into a position that is neither.
//
// Always bound, like the IBL and AO slots and for the same reason: SDL3 GPU
// leaves an unbound fragment sampler descriptor undefined, so dummies stand in
// when GI is off and `giEnabled` is what decides whether anything is read.
[[vk::combinedImageSampler]][[vk::binding(11, 2)]]
Texture2D<float4> giIrradianceAtlas;
[[vk::combinedImageSampler]][[vk::binding(11, 2)]]
SamplerState giIrradianceAtlasSampler;

[[vk::combinedImageSampler]][[vk::binding(12, 2)]]
Texture2D<float2> giVisibilityAtlas;
[[vk::combinedImageSampler]][[vk::binding(12, 2)]]
SamplerState giVisibilityAtlasSampler;

[[vk::combinedImageSampler]][[vk::binding(13, 2)]]
Texture2D<float4> giProbeMeta;
[[vk::combinedImageSampler]][[vk::binding(13, 2)]]
SamplerState giProbeMetaSampler;

// Ray-traced LOCAL (point/spot) shadow ratio (R8, 1 = lit), at the local pass's
// own traced resolution — so, like rtShadowTexture, ALWAYS sample it with a
// normalized UV. Bound to the 1x1 white dummy and read only when
// rtLocalShadowEnabled, which is the same no-op the sun mask uses.
[[vk::combinedImageSampler]][[vk::binding(14, 2)]]
Texture2D<float> rtLocalShadowTexture;
[[vk::combinedImageSampler]][[vk::binding(14, 2)]]
SamplerState rtLocalShadowSampler;


#include "shadow_common.hlsli"
#include "fog.hlsli"
#include "lighting.hlsli"
#include "pbr.hlsli"
#include "reflection_gate.hlsli"
// Cluster SSBOs follow this shader's 16 fragment samplers in set 2 (slots
// 0-14 above + the local-shadow ATLAS at 15, declared by the include). R4b's
// screen-probe SH buffer is a FIFTH storage buffer at base + 4, declared by
// gi_screen_sample.hlsli.
#define CLUSTER_SSBO_BINDING_BASE 16
#include "gi_sample.hlsli"
#include "cluster_lighting.hlsli"
#include "gi_screen_sample.hlsli"

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

    // Traced sun shadows REPLACE the cascade result (see model.frag.hlsl).
    if (rtShadowEnabled > 0.5) {
        float2 rtUV = input.position.xy * float2(screenAOTexelX, screenAOTexelY);
        shadow = rtShadowTexture.SampleLevel(rtShadowSampler, rtUV, 0).r;
    }
    float3 directLight = lightColor * sunIntensity * shadow;

    float3 direct = evaluatePBR(N, V, L, directLight, baseColor.rgb, metallic, roughness, 1.0);

    // Screen-space GTAO, ambient only: min-combined with the material (ORM)
    // AO to avoid double-darkening. Direct light/speculars stay untouched.
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

    float ambientLift = lerp(ambientStrength, 1.0, shadow);
    // Probe-field GI (R1c, render/gi/ + gi_sample.hlsli). It replaces the
    // DIFFUSE half of whichever ambient branch runs below — ambientIBL's
    // irradiance-cube tap or ambientPBR's hemisphere gradient — and leaves the
    // SPECULAR half exactly as it was; R4 is what changes that, not this. The
    // same combined AO and the same ambient lift multiply it as multiply the
    // term it replaces, which is the lighting-quantity ledger in
    // docs/rt_lighting_plan.md §4. giEnabled == 0 skips every tap and the
    // ambient term is bit-identical to the pre-R1c image.
    float3 giIrradiance = float3(0, 0, 0);
    if (giEnabled > 0.5) {
        giIrradiance = giSampleDiffuse(input.fragPos, N, V, ambientSky, ambientGround);
        // Screen probes (R4b) on TOP of the field, never instead of it. Same
        // unit — irradiance/pi — so this is a straight lerp, and the weight is
        // what makes the hand-off safe: a pixel whose four probes sit on other
        // surfaces, or whose tile was sky, gets weight 0 and keeps the field's
        // answer exactly.
        float giScreenWeight;
        float3 giScreen =
            giScreenSampleDiffuse(input.position.xy, input.fragPos, N, cameraPos, giScreenWeight);
        giIrradiance = lerp(giIrradiance, giScreen, giScreenWeight);
    }
    float3 ambient;
    if (iblEnabled > 0.5) {
        ambient = ambientIBL(N, N_geo, V, baseColor.rgb, metallic, roughness, aoCombined, ambientLift,
                             iblIrradianceMap, iblIrradianceSampler, iblSpecularMap, iblSpecularSampler,
                             iblBrdfLut, iblBrdfLutSampler, iblSpecularScale, giIrradiance, giEnabled);
    } else {
        ambient = ambientPBR(N, V, baseColor.rgb, metallic, roughness, aoCombined,
                             ambientSky * ambientLift, ambientGround * ambientLift,
                             // Pre-scaled by the lift, because the hemisphere
                             // colours above are. ambientIBL's override is not:
                             // that function applies ao * lift at the end.
                             giIrradiance * ambientLift, giEnabled);
    }

    float3 result = direct + ambient;

    if (clusterParams.gridDim.w > 0u) {
        float2 screen_uv = input.position.xy * clusterParams.screenSize.zw;
        // Ratio tier: one traced visibility multiplies the whole local
        // subtotal. -1 is "no traced mask", which skips the multiply outright.
        float localVis = -1.0;
        if (rtLocalShadowEnabled > 0.5) {
            localVis = rtLocalShadowTexture.SampleLevel(rtLocalShadowSampler, screen_uv, 0).r;
        }
        result += accumulateClusterLights(input.fragPos, N, V,
                                          baseColor.rgb, metallic, roughness, 1.0,
                                          input.viewDepth, screen_uv, localVis);
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
