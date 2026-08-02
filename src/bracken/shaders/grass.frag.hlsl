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
    float screenAOTexelX;  // 1 / render-target size: SV_Position.xy * texel = AO UV
    float3 ambientGround;
    float screenAOTexelY;
    float iblEnabled;      // 1 = sky IBL ambient (samplers 3-5), 0 = hemisphere fallback
    float rtShadowEnabled; // 1 = traced sun visibility replaces the cascades
    // GraphicsSettings::rt_local_shadows as a float (a cbuffer has no enums):
    // 0 = Off, 1 = Ratio (multiply the LOCAL subtotal by the traced mask),
    // 2 = Full-DI (R5c: do not run the analytic local loop at all and read the
    // two demodulated DI storage textures instead). Was the first _padIbl lane.
    float rtLocalShadowMode;
    float _padIbl;
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
Texture2D shadowAtlas;
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerComparisonState shadowCmpSampler;

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D shadowAtlasRaw;
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerState shadowRawSampler;

// Full-res screen-space GTAO (RGBA8 since R5a: R = AO with 1 = unoccluded,
// GB = octahedral world bent normal, A = cone aperture — bent_cone.hlsli), computed before this pass
// from the prepass depth. Bound to a 1x1 white dummy when AO is off.
[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
Texture2D<float4> screenAOTexture;
[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
SamplerState screenAOSampler;

// Sky IBL chain (render/ibl.hpp): irradiance cube + GGX-prefiltered specular
// cube + split-sum BRDF LUT. Bound to dummies when iblEnabled == 0.
[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
TextureCube<float4> iblIrradianceMap;
[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
SamplerState iblIrradianceSampler;

[[vk::combinedImageSampler]][[vk::binding(4, 2)]]
TextureCube<float4> iblSpecularMap;
[[vk::combinedImageSampler]][[vk::binding(4, 2)]]
SamplerState iblSpecularSampler;

[[vk::combinedImageSampler]][[vk::binding(5, 2)]]
Texture2D<float4> iblBrdfLut;
[[vk::combinedImageSampler]][[vk::binding(5, 2)]]
SamplerState iblBrdfLutSampler;

[[vk::combinedImageSampler]][[vk::binding(6, 2)]]
Texture2D<float> rtShadowTexture;
[[vk::combinedImageSampler]][[vk::binding(6, 2)]]
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
[[vk::combinedImageSampler]][[vk::binding(7, 2)]]
Texture2D<float4> giIrradianceAtlas;
[[vk::combinedImageSampler]][[vk::binding(7, 2)]]
SamplerState giIrradianceAtlasSampler;

[[vk::combinedImageSampler]][[vk::binding(8, 2)]]
Texture2D<float2> giVisibilityAtlas;
[[vk::combinedImageSampler]][[vk::binding(8, 2)]]
SamplerState giVisibilityAtlasSampler;

[[vk::combinedImageSampler]][[vk::binding(9, 2)]]
Texture2D<float4> giProbeMeta;
[[vk::combinedImageSampler]][[vk::binding(9, 2)]]
SamplerState giProbeMetaSampler;

// Ray-traced LOCAL (point/spot) shadow ratio (R8, 1 = lit), at the local pass's
// own traced resolution — so, like rtShadowTexture, ALWAYS sample it with a
// normalized UV. Bound to the 1x1 white dummy and read only when
// rtLocalShadowMode, which is the same no-op the sun mask uses.
[[vk::combinedImageSampler]][[vk::binding(10, 2)]]
Texture2D<float> rtLocalShadowTexture;
[[vk::combinedImageSampler]][[vk::binding(10, 2)]]
SamplerState rtLocalShadowSampler;


#include "shadow_common.hlsli"
#include "fog.hlsli"
#include "lighting.hlsli"
#include "pbr.hlsli"
#include "bent_cone.hlsli"
// Cluster SSBOs follow this shader's 12 fragment samplers in set 2 (slots
// 0-10 above + the local-shadow ATLAS at 11, declared by the include). R4b's
// screen-probe SH buffer is a FIFTH storage buffer at base + 4, declared by
// gi_screen_sample.hlsli.
// R5c: the Full-DI tier's two demodulated targets are READ-ONLY STORAGE
// TEXTURES, which SDL3 GPU places between the sampled textures and the storage
// buffers in fragment set 2. So the sampler count and the SSBO base are no
// longer the same number, and both are declared: the local-shadow atlas
// cluster_lighting.hlsli owns follows CLUSTER_SAMPLER_COUNT, the cluster
// buffers follow CLUSTER_SSBO_BINDING_BASE. local_di.hlsli explains why the DI
// pair cannot be samplers here.
#define CLUSTER_SAMPLER_COUNT 12
#define LOCAL_DI_BINDING_BASE CLUSTER_SAMPLER_COUNT
#define CLUSTER_SSBO_BINDING_BASE (CLUSTER_SAMPLER_COUNT + 2)
#include "local_di.hlsli"
#include "gi_sample.hlsli"
#include "cluster_lighting.hlsli"
#include "gi_screen_sample.hlsli"

static const float GRASS_METALLIC  = 0.0;
static const float GRASS_ROUGHNESS = 0.9;

float4 PSMain(PSInput input) : SV_Target {
    float3 baseColor = input.color.rgb;
    float3 N = normalize(input.normal);
    float3 V = normalize(cameraPos - input.worldPos);
    float3 L = normalize(-lightDir);

    float shadow = calcShadow(input.worldPos, N, L, input.viewDepth, input.position.xy);
    if (rtShadowEnabled > 0.5) {
        float2 rtUV = input.position.xy * float2(screenAOTexelX, screenAOTexelY);
        shadow = rtShadowTexture.SampleLevel(rtShadowSampler, rtUV, 0).r;
    }

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

    // Screen-space GTAO, ambient only (grass has no material AO map).
    // Direct light and the tip specular stay untouched. Since R5a the texel is
    // a whole BENT CONE — shaders/bent_cone.hlsli.
    BentCone cone = decodeBentCone(
        screenAOTexture.Sample(screenAOSampler,
                               input.position.xy * float2(screenAOTexelX, screenAOTexelY)),
        N);
    float screenAO = cone.ao;
    // The DIFFUSE GI term is gated by the cone instead of that scalar — never
    // by both (docs/rt_lighting_plan.md §5.6); without a cone it IS the scalar.
    float giGate = bentConeGiGate(cone, 1.0, screenAO);

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
        giIrradiance = giSampleDiffuse(input.worldPos, N, cone.direction, V, ambientSky, ambientGround);
        // Screen probes (R4b) on TOP of the field, never instead of it. Same
        // unit — irradiance/pi — so this is a straight lerp, and the weight is
        // what makes the hand-off safe: a pixel whose four probes sit on other
        // surfaces, or whose tile was sky, gets weight 0 and keeps the field's
        // answer exactly.
        float giScreenWeight;
        float3 giScreen =
            giScreenSampleDiffuse(input.position.xy, input.worldPos, N, cone.direction, cameraPos,
                                  giScreenWeight);
        giIrradiance = lerp(giIrradiance, giScreen, giScreenWeight);
    }
    float3 ambient;
    if (iblEnabled > 0.5) {
        ambient = ambientIBL(N, N, V, baseColor, GRASS_METALLIC, GRASS_ROUGHNESS, screenAO,
                             ambientLift, iblIrradianceMap, iblIrradianceSampler,
                             iblSpecularMap, iblSpecularSampler, iblBrdfLut, iblBrdfLutSampler,
                             1.0, giIrradiance, giEnabled, giGate);
    } else {
        ambient = ambientPBR(N, V, baseColor, GRASS_METALLIC, GRASS_ROUGHNESS, screenAO,
                             ambientSky * ambientLift, ambientGround * ambientLift,
                             // Pre-scaled by the lift, because the hemisphere
                             // colours above are. ambientIBL's override is not:
                             // that function applies the lift at the end (and
                             // the occlusion per ambient source, since R5a).
                             giIrradiance * ambientLift, giEnabled, giGate);
    }

    float3 color = direct + spec * lightColor + ambient;

    if (clusterParams.gridDim.w > 0u) {
        float2 screen_uv = input.position.xy * clusterParams.screenSize.zw;
        // Which tier owns the local lights this frame (rtLocalShadowMode):
        //   2 = Full-DI  — they are ALL traced and this loop does not run.
        //   1 = Ratio    — the analytic loop runs and one traced visibility
        //                  multiplies its whole point+spot subtotal.
        //   0 = Off      — the rasterized atlas, untouched.
        // -1 as the localVis argument is "no traced mask", which skips the
        // multiply outright rather than multiplying by 1.0.
        if (rtLocalShadowMode > 1.5) {
            color += localDiLighting(screen_uv, input.viewDepth, baseColor, GRASS_METALLIC);
        } else {
            float localVis = -1.0;
            if (rtLocalShadowMode > 0.5) {
                localVis = rtLocalShadowTexture.SampleLevel(rtLocalShadowSampler, screen_uv, 0).r;
            }
            color += accumulateClusterLights(input.worldPos, N, V,
                                             baseColor, GRASS_METALLIC, GRASS_ROUGHNESS, 1.0,
                                             input.viewDepth, screen_uv, localVis);
        }
    }

    // Same atmospheric fog as terrain/models: blades dissolve into the scene
    // fog instead of a hardcoded color band. The VS LOD collapse handles the
    // view-distance boundary itself.
    color = applyAtmosphericFog(color, input.worldPos, cameraPos, lightDir,
                                fogColorHorizon, fogColorSun, fogStart, fogEnd, fogHeightFalloff);

    return float4(color, input.color.a);
}
