// Terrain Fragment Shader.
// Linear HDR output. Material array is sRGB (sampler linearizes); the splatmap
// is a linear control mask with its own CLAMP sampler. Anti-tiling: each layer
// blends two sample scales by a low-frequency value noise of world XZ. Flat
// terrain (N.y > 0.8) takes a single-plane fast path instead of full triplanar,
// and layers below a small splat weight are skipped. All samples use explicit
// gradients (SampleGrad) so the divergent layer skips stay well-defined.

struct PSInput {
    float4 position : SV_Position;
    float4 vertexColor : TEXCOORD0;
    float3 fragPos : TEXCOORD1;
    float3 normal : TEXCOORD2;
    float viewDepth : TEXCOORD3;
};

[[vk::binding(0, 3)]]
cbuffer LightingUniforms {
    float3 fogColorHorizon;
    float fogStart;
    float3 fogColorSun;
    float fogEnd;
    float3 lightDir;
    float fogHeightFalloff;
    float3 cameraPos;
    float screenAOTexelX;  // 1 / render-target size: SV_Position.xy * texel = AO UV
    float3 ambientSky;
    float ambientStrength;
    float3 ambientGround;
    float sunIntensity;
    float3 sunColor;
    float screenAOTexelY;
    float4 splatParams;    // xy = splat origin (world XZ), zw = splat X/Z size
    float iblEnabled;      // 1 = sky IBL ambient (samplers 5-7), 0 = hemisphere fallback
    float rtShadowEnabled; // 1 = traced sun visibility replaces the cascades
    // 1 = multiply the LOCAL (point/spot) subtotal by the ray-traced
    // local-shadow mask (R2b Ratio tier). Was the first _padIbl lane.
    float rtLocalShadowEnabled;
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
Texture2DArray materialTextures;
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState materialSampler;

[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
Texture2D splatmapTexture;
[[vk::combinedImageSampler]][[vk::binding(1, 2)]]
SamplerState splatmapSampler;

[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
Texture2D shadowAtlas;
[[vk::combinedImageSampler]][[vk::binding(2, 2)]]
SamplerComparisonState shadowCmpSampler;

[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
Texture2D shadowAtlasRaw;
[[vk::combinedImageSampler]][[vk::binding(3, 2)]]
SamplerState shadowRawSampler;

// Full-res screen-space GTAO (RGBA8 since R5a: R = AO with 1 = unoccluded,
// GB = octahedral world bent normal, A = cone aperture — bent_cone.hlsli), computed before this pass
// from the prepass depth. Bound to a 1x1 white dummy when AO is off.
[[vk::combinedImageSampler]][[vk::binding(4, 2)]]
Texture2D<float4> screenAOTexture;
[[vk::combinedImageSampler]][[vk::binding(4, 2)]]
SamplerState screenAOSampler;

// Sky IBL chain (render/ibl.hpp): irradiance cube + GGX-prefiltered specular
// cube + split-sum BRDF LUT. Bound to dummies when iblEnabled == 0.
[[vk::combinedImageSampler]][[vk::binding(5, 2)]]
TextureCube<float4> iblIrradianceMap;
[[vk::combinedImageSampler]][[vk::binding(5, 2)]]
SamplerState iblIrradianceSampler;

[[vk::combinedImageSampler]][[vk::binding(6, 2)]]
TextureCube<float4> iblSpecularMap;
[[vk::combinedImageSampler]][[vk::binding(6, 2)]]
SamplerState iblSpecularSampler;

[[vk::combinedImageSampler]][[vk::binding(7, 2)]]
Texture2D<float4> iblBrdfLut;
[[vk::combinedImageSampler]][[vk::binding(7, 2)]]
SamplerState iblBrdfLutSampler;

[[vk::combinedImageSampler]][[vk::binding(8, 2)]]
Texture2D<float> rtShadowTexture;
[[vk::combinedImageSampler]][[vk::binding(8, 2)]]
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
[[vk::combinedImageSampler]][[vk::binding(9, 2)]]
Texture2D<float4> giIrradianceAtlas;
[[vk::combinedImageSampler]][[vk::binding(9, 2)]]
SamplerState giIrradianceAtlasSampler;

[[vk::combinedImageSampler]][[vk::binding(10, 2)]]
Texture2D<float2> giVisibilityAtlas;
[[vk::combinedImageSampler]][[vk::binding(10, 2)]]
SamplerState giVisibilityAtlasSampler;

[[vk::combinedImageSampler]][[vk::binding(11, 2)]]
Texture2D<float4> giProbeMeta;
[[vk::combinedImageSampler]][[vk::binding(11, 2)]]
SamplerState giProbeMetaSampler;

// Ray-traced LOCAL (point/spot) shadow ratio (R8, 1 = lit), at the local pass's
// own traced resolution — so, like rtShadowTexture, ALWAYS sample it with a
// normalized UV. Bound to the 1x1 white dummy and read only when
// rtLocalShadowEnabled, which is the same no-op the sun mask uses.
[[vk::combinedImageSampler]][[vk::binding(12, 2)]]
Texture2D<float> rtLocalShadowTexture;
[[vk::combinedImageSampler]][[vk::binding(12, 2)]]
SamplerState rtLocalShadowSampler;


#include "shadow_common.hlsli"
#include "fog.hlsli"
#include "lighting.hlsli"
#include "pbr.hlsli"
#include "bent_cone.hlsli"
// Cluster SSBOs follow this shader's 14 fragment samplers in set 2 (slots
// 0-12 above + the local-shadow ATLAS at 13, declared by the include). R4b's
// screen-probe SH buffer is a FIFTH storage buffer at base + 4, declared by
// gi_screen_sample.hlsli.
#define CLUSTER_SSBO_BINDING_BASE 14
#include "gi_sample.hlsli"
#include "cluster_lighting.hlsli"
#include "gi_screen_sample.hlsli"

static const float TERRAIN_METALLIC = 0.0;
static const float TERRAIN_AO = 1.0;
static const float TEX_SCALE = 0.01;           // base material scale (tiles every 100 units)
static const float DETAIL_SCALE_RATIO = 0.23;  // second anti-tiling scale (non-integer ratio)
static const float SPLAT_EPSILON = 0.005;      // skip layers below this splat weight
static const float FLAT_NORMAL_Y = 0.8;        // above this, single-plane fast path

// Integer (murmur-finalizer) hash, same as grass.vert.hlsl — stable across
// the whole world. A sin-based hash loses its mantissa to fp32 argument
// reduction ~10^4 units from the origin and bands/plateaus at world scale.
uint murmurHash2D(int2 p) {
    uint seed = asuint(p.x) ^ (asuint(p.y) * 0x9e3779b9u);
    seed ^= seed >> 16;
    seed *= 0x85ebca6bu;
    seed ^= seed >> 13;
    seed *= 0xc2b2ae35u;
    seed ^= seed >> 16;
    return seed;
}

// p arrives integer-valued (a floor'ed lattice coordinate from valueNoise).
float hash2D(float2 p) {
    uint h = murmurHash2D(int2(p));
    return float(h & 0x00FFFFFFu) / float(0x01000000u);
}

// Cheap low-frequency value noise; drives the anti-tiling scale blend and
// subtle roughness variation.
float valueNoise(float2 p) {
    float2 i = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash2D(i);
    float b = hash2D(i + float2(1.0, 0.0));
    float c = hash2D(i + float2(0.0, 1.0));
    float d = hash2D(i + float2(1.0, 1.0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

// One layer, one projection plane, two scales blended by noise (anti-tiling).
float3 samplePlaneLayer(float2 uv, float2 ddxUV, float2 ddyUV, int layer, float scaleBlend) {
    float3 c1 = materialTextures.SampleGrad(materialSampler, float3(uv * TEX_SCALE, layer),
                                            ddxUV * TEX_SCALE, ddyUV * TEX_SCALE).rgb;
    float detailScale = TEX_SCALE * DETAIL_SCALE_RATIO;
    float3 c2 = materialTextures.SampleGrad(materialSampler, float3(uv * detailScale, layer),
                                            ddxUV * detailScale, ddyUV * detailScale).rgb;
    return lerp(c1, c2, scaleBlend);
}

float3 sampleTriplanarLayer(float3 worldPos, float3 dpdx, float3 dpdy, float3 blendWeights, int layer,
                            float scaleBlend) {
    float3 xProj = samplePlaneLayer(worldPos.yz, dpdx.yz, dpdy.yz, layer, scaleBlend);
    float3 yProj = samplePlaneLayer(worldPos.xz, dpdx.xz, dpdy.xz, layer, scaleBlend);
    float3 zProj = samplePlaneLayer(worldPos.xy, dpdx.xy, dpdy.xy, layer, scaleBlend);
    return xProj * blendWeights.x + yProj * blendWeights.y + zProj * blendWeights.z;
}

float4 PSMain(PSInput input) : SV_Target {
    float3 worldPos = input.fragPos;
    float3 N = normalize(input.normal);

    // Splat UV from the world-rect splat region (CLAMP sampler handles the
    // margin ring outside the world bounds).
    float2 splatmapUV = (worldPos.xz - splatParams.xy) / splatParams.zw;
    float4 splatWeights = splatmapTexture.Sample(splatmapSampler, splatmapUV);

    float noise = valueNoise(worldPos.xz * 0.004);

    // Explicit world-position gradients so layer skipping / flat-path branches
    // can't corrupt implicit derivatives.
    float3 dpdx = ddx(worldPos);
    float3 dpdy = ddy(worldPos);

    float3 triBlend = abs(N);
    triBlend = pow(triBlend, 4.0);
    triBlend /= (triBlend.x + triBlend.y + triBlend.z);

    bool flatPath = N.y > FLAT_NORMAL_Y;

    float3 terrainColor = float3(0.0, 0.0, 0.0);
    float weightSum = 0.0;
    [unroll]
    for (int layer = 0; layer < 4; ++layer) {
        float w = splatWeights[layer];
        if (w < SPLAT_EPSILON) {
            continue;
        }
        float3 layerColor;
        if (flatPath) {
            layerColor = samplePlaneLayer(worldPos.xz, dpdx.xz, dpdy.xz, layer, noise);
        } else {
            layerColor = sampleTriplanarLayer(worldPos, dpdx, dpdy, triBlend, layer, noise);
        }
        terrainColor += layerColor * w;
        weightSum += w;
    }
    terrainColor /= max(weightSum, 1e-3);

    float3 baseColor = terrainColor * input.vertexColor.rgb;

    // Subtle roughness variation rides the same noise field.
    float roughness = 0.9 + (noise - 0.5) * 0.1;

    float3 V = normalize(cameraPos - worldPos);
    float3 L = normalize(-lightDir);

    float shadow = calcShadow(worldPos, N, L, input.viewDepth, input.position.xy);
    if (rtShadowEnabled > 0.5) {
        float2 rtUV = input.position.xy * float2(screenAOTexelX, screenAOTexelY);
        shadow = rtShadowTexture.SampleLevel(rtShadowSampler, rtUV, 0).r;
    }
    float3 lightColor = sunColor * sunIntensity * shadow;

    float3 direct = evaluatePBR(N, V, L, lightColor,
                                 baseColor, TERRAIN_METALLIC, roughness, TERRAIN_AO);

    // Screen-space GTAO, ambient only (terrain has no material AO map, so
    // min(TERRAIN_AO, screenAO) == screenAO). Direct light stays untouched.
    // Since R5a the texel is a whole BENT CONE — shaders/bent_cone.hlsli.
    BentCone cone = decodeBentCone(
        screenAOTexture.Sample(screenAOSampler,
                               input.position.xy * float2(screenAOTexelX, screenAOTexelY)),
        N);
    float screenAO = cone.ao;
    // The DIFFUSE GI term is gated by the cone instead of that scalar — never
    // by both (docs/rt_lighting_plan.md §5.6); without a cone it IS the scalar.
    float giGate = bentConeGiGate(cone, TERRAIN_AO, min(TERRAIN_AO, screenAO));

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
        giIrradiance = giSampleDiffuse(worldPos, N, cone.direction, V, ambientSky, ambientGround);
        // Screen probes (R4b) on TOP of the field, never instead of it. Same
        // unit — irradiance/pi — so this is a straight lerp, and the weight is
        // what makes the hand-off safe: a pixel whose four probes sit on other
        // surfaces, or whose tile was sky, gets weight 0 and keeps the field's
        // answer exactly.
        float giScreenWeight;
        float3 giScreen =
            giScreenSampleDiffuse(input.position.xy, worldPos, N, cone.direction, cameraPos,
                                  giScreenWeight);
        giIrradiance = lerp(giIrradiance, giScreen, giScreenWeight);
    }
    float3 ambient;
    if (iblEnabled > 0.5) {
        ambient = ambientIBL(N, N, V, baseColor, TERRAIN_METALLIC, roughness, min(TERRAIN_AO, screenAO),
                             ambientLift, iblIrradianceMap, iblIrradianceSampler,
                             iblSpecularMap, iblSpecularSampler, iblBrdfLut, iblBrdfLutSampler,
                             1.0, giIrradiance, giEnabled, giGate);
    } else {
        ambient = ambientPBR(N, V, baseColor, TERRAIN_METALLIC, roughness, min(TERRAIN_AO, screenAO),
                             ambientSky * ambientLift, ambientGround * ambientLift,
                             // Pre-scaled by the lift, because the hemisphere
                             // colours above are. ambientIBL's override is not:
                             // that function applies the lift at the end (and
                             // the occlusion per ambient source, since R5a).
                             giIrradiance * ambientLift, giEnabled, giGate);
    }

    float3 color = direct + ambient;

    if (clusterParams.gridDim.w > 0u) {
        float2 screen_uv = input.position.xy * clusterParams.screenSize.zw;
        // Ratio tier: one traced visibility multiplies the whole local
        // subtotal. -1 is "no traced mask", which skips the multiply outright.
        float localVis = -1.0;
        if (rtLocalShadowEnabled > 0.5) {
            localVis = rtLocalShadowTexture.SampleLevel(rtLocalShadowSampler, screen_uv, 0).r;
        }
        color += accumulateClusterLights(worldPos, N, V,
                                         baseColor, TERRAIN_METALLIC, roughness, TERRAIN_AO,
                                         input.viewDepth, screen_uv, localVis);
    }

    color = applyAtmosphericFog(color, worldPos, cameraPos, lightDir,
                                fogColorHorizon, fogColorSun, fogStart, fogEnd, fogHeightFalloff);

    return float4(color, 1.0);
}
