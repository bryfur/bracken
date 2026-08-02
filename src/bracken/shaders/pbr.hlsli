// pbr.hlsli - Cook-Torrance microfacet BRDF (GGX + Smith + Schlick)
// with Burley diffuse. All inputs are linear; output is linear radiance
// contribution from a single directional light.

#ifndef PBR_HLSLI
#define PBR_HLSLI

#ifndef PI_PBR
#define PI_PBR 3.14159265358979323846
#endif

// GGX/Trowbridge-Reitz normal distribution.
float ndfGGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / max(PI_PBR * d * d, 1e-7);
}

// Smith correlated visibility: V = G2 / (4 * NdotV * NdotL).
// Returns the already-divided visibility term; multiply by F and D for full spec.
float visibilitySmithGGX(float NdotV, float NdotL, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float ggxV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float ggxL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(ggxV + ggxL, 1e-5);
}

// Schlick Fresnel with roughness-aware grazing term (used by IBL later).
float3 fresnelSchlick(float cosTheta, float3 F0) {
    float f = pow(1.0 - cosTheta, 5.0);
    return F0 + (1.0 - F0) * f;
}

float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness) {
    float f = pow(1.0 - cosTheta, 5.0);
    return F0 + (max((1.0 - roughness).xxx, F0) - F0) * f;
}

// Burley (Disney) diffuse. Energy-conserving wrap that darkens grazing.
float3 diffuseBurley(float3 baseColor, float roughness, float NdotV, float NdotL, float LdotH) {
    float fd90 = 0.5 + 2.0 * LdotH * LdotH * roughness;
    float lightScatter = 1.0 + (fd90 - 1.0) * pow(1.0 - NdotL, 5.0);
    float viewScatter  = 1.0 + (fd90 - 1.0) * pow(1.0 - NdotV, 5.0);
    return baseColor * (lightScatter * viewScatter / PI_PBR);
}

// Single directional-light contribution.
// normal, view, light: world-space unit vectors (view points from surface to camera,
// light points from surface to light).
// lightColor: linear radiance at the surface (already shadow-modulated by caller if wanted).
// baseColor: linear albedo.
// metallic, roughness: glTF convention (metallic in [0,1], perceptual roughness in [0,1]).
// ao: baked occlusion multiplier applied to diffuse only.
float3 evaluatePBR(float3 normal, float3 view, float3 light, float3 lightColor,
                    float3 baseColor, float metallic, float roughness, float ao) {
    float3 N = normal;
    float3 V = view;
    float3 L = light;

    float NdotV = max(dot(N, V), 1e-4);
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) {
        return float3(0, 0, 0);
    }
    // Keep H after the NdotL early-out: V == -L is the only normalize(0)
    // input, and the rejected path must never form it. Guard the length for
    // the (measure-zero) exact V == -L case that survives the early-out.
    float3 sum = V + L;
    float3 H = sum / max(length(sum), 1e-5);
    float NdotH = max(dot(N, H), 0.0);
    float LdotH = max(dot(L, H), 0.0);

    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);

    float  D = ndfGGX(NdotH, roughness);
    float  Vv = visibilitySmithGGX(NdotV, NdotL, roughness);
    float3 F = fresnelSchlick(LdotH, F0);

    float3 spec = D * Vv * F;

    // Energy conservation: metals have no diffuse.
    float3 kd = (1.0 - F) * (1.0 - metallic);
    float3 diff = kd * diffuseBurley(baseColor, roughness, NdotV, NdotL, LdotH) * ao;

    return (diff + spec) * lightColor * NdotL;
}

// Hemisphere "IBL" ambient: two-color sky/ground hemisphere for diffuse
// irradiance, plus a Karis analytic env-BRDF specular term sampled along the
// reflection vector against the same hemisphere. AO multiplies the whole
// ambient. All colors linear.
float3 ambientPBR(float3 N, float3 V, float3 baseColor, float metallic, float roughness,
                  float ao, float3 skyColor, float3 groundColor,
                  // Probe-field GI (R1c, render/gi/): when giWeight > 0 this
                  // REPLACES the hemisphere diffuse irradiance below. It must
                  // arrive already scaled by the caller's ambient lift, because
                  // skyColor/groundColor are — see model.frag.hlsl's call.
                  // The SPECULAR half keeps reading the hemisphere; R4 is what
                  // changes that, not R1c. giWeight == 0 leaves the branch
                  // untaken, so the GI-off path is bit-identical.
                  float3 giIrradiance = float3(0, 0, 0), float giWeight = 0.0,
                  // BENT-CONE GATE (R5a). The GI term is the only ambient
                  // quantity with a direction of its own, so it is the only one
                  // the AO pass's bent cone can gate directionally: pass the
                  // cone's aperture here and the caller's giIrradiance sampled
                  // along the BENT normal. Negative means "no cone" and the
                  // scalar `ao` gates the GI term like everything else, which
                  // is what keeps the AO-off and pre-R5a paths bit-identical.
                  // The two are never both applied — that is the double-count
                  // rule of docs/rt_lighting_plan.md §5.6.
                  float giGate = -1.0) {
    float NdotV = max(dot(N, V), 1e-4);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);

    // Diffuse: hemisphere gradient on the normal, energy split via roughness-
    // aware Fresnel (no Burley — irradiance is already cosine-integrated).
    float3 F = fresnelSchlickRoughness(NdotV, F0, roughness);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    float3 irradiance = lerp(groundColor, skyColor, N.y * 0.5 + 0.5);
    // Each half of the diffuse blend carries its OWN occlusion — the
    // hemisphere's is the scalar AO, the probe's is the bent cone's aperture
    // (or the scalar AO again when there is no cone). Folding them in here
    // rather than multiplying `ao` over the sum at the end is what makes the
    // bent-cone gate REPLACE the scalar one instead of stacking with it;
    // giGate == ao reproduces the pre-R5a arithmetic exactly.
    float gate = (giGate >= 0.0) ? giGate : ao;
    float3 diffuseIrradiance = irradiance * ao;
    if (giWeight > 0.0) {
        diffuseIrradiance = lerp(diffuseIrradiance, giIrradiance * gate, giWeight);
    }
    float3 diffuse = kd * baseColor * diffuseIrradiance;

    // Specular: Karis mobile split-sum fit (UE4 EnvBRDFApprox) against the
    // hemisphere color sampled toward the reflection vector.
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    float2 AB = float2(-1.04, 1.04) * a004 + r.zw;

    float3 R = reflect(-V, N);
    float3 envColor = lerp(groundColor, skyColor, R.y * 0.5 + 0.5);
    float3 specular = envColor * (F0 * AB.x + AB.y);

    // The diffuse half is already gated above (per-source); specular keeps the
    // plain scalar AO it always had.
    return diffuse + specular * ao;
}

// Image-based ambient from the sky prefilter chain (render/ibl.hpp):
// diffuse = irradiance-cubemap sample x albedo; specular = GGX-prefiltered
// cubemap sample (roughness -> mip) x split-sum BRDF LUT, with a Frostbite
// horizon-occlusion guard (normal mapping can push R below the geometric
// surface). `ao` (combined material + screen AO) and `lift` (shadow-driven
// ambient lift) multiply the WHOLE ambient — exactly like ambientPBR, whose
// hemisphere terms this replaces when the ibl uniform flag is on. All colors
// linear.
static const float IBL_SPECULAR_MIPS = 7.0; // render::IBL::SPECULAR_MIPS — keep in sync

// How much of the ambient SPECULAR lobe the probe field supplies instead of the
// sky-only prefilter chain, as a function of roughness. TRANSLITERATION of
// `rough_specular_gi_weight` in render/rt_reflection_math.hpp, which is the
// source of truth, carries the reasoning for both edges, and is tested with no
// GPU. Change one, change both.
#define GI_ROUGH_SPECULAR_START 0.6
#define GI_ROUGH_SPECULAR_END 1.0

float giRoughSpecularWeight(float roughness) {
    return smoothstep(GI_ROUGH_SPECULAR_START, GI_ROUGH_SPECULAR_END, roughness);
}

float3 ambientIBL(float3 N, float3 Ngeo, float3 V, float3 baseColor, float metallic,
                  float roughness, float ao, float lift,
                  TextureCube<float4> irradianceMap, SamplerState irradianceSampler,
                  TextureCube<float4> specularMap, SamplerState specularSampler,
                  Texture2D<float4> brdfLut, SamplerState brdfLutSampler,
                  // 0 drops the probe's specular lobe entirely, for pixels a
                  // ray-traced reflection is about to supply instead. Adding
                  // both would count the same specular energy twice, which
                  // reads as blown-out metal.
                  float specularScale = 1.0,
                  // Probe-field GI (R1c, render/gi/): when giWeight > 0 this
                  // REPLACES the irradiance-cube diffuse below. NOT pre-scaled
                  // by the ambient lift — unlike ambientPBR's, this function
                  // applies `ao * lift` to everything at the end. Since R3 the
                  // ROUGH end of the specular half uses it too — see
                  // giRoughSpecularWeight.
                  float3 giIrradiance = float3(0, 0, 0), float giWeight = 0.0,
                  // BENT-CONE GATE (R5a) — see ambientPBR's giGate. Applies to
                  // the DIFFUSE probe term only: the rough-specular handover
                  // below reuses giIrradiance as a wide GGX lobe, which the
                  // cone (a cosine-lobe fit) does not describe, so that one
                  // keeps the scalar AO. Negative = no cone.
                  float giGate = -1.0) {
    float NdotV = max(dot(N, V), 1e-4);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);

    // Diffuse: cosine-convolved sky irradiance along the normal, energy split
    // via roughness-aware Fresnel (irradiance is already cosine-integrated).
    float3 F = fresnelSchlickRoughness(NdotV, F0, roughness);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    float3 irradiance = irradianceMap.SampleLevel(irradianceSampler, N, 0).rgb;
    // Per-source occlusion — see ambientPBR's identical block. `ao` is NOT
    // applied over the sum at the end any more; each term carries its own.
    float gate = (giGate >= 0.0) ? giGate : ao;
    float3 diffuseIrradiance = irradiance * ao;
    if (giWeight > 0.0) {
        diffuseIrradiance = lerp(diffuseIrradiance, giIrradiance * gate, giWeight);
    }
    float3 diffuse = kd * baseColor * diffuseIrradiance;

    // Specular: prefiltered radiance along R at the roughness-matched mip,
    // scaled/biased by the split-sum LUT.
    float3 R = reflect(-V, N);
    float mip = roughness * (IBL_SPECULAR_MIPS - 1.0);
    float3 prefiltered = specularMap.SampleLevel(specularSampler, R, mip).rgb;
    // ROUGH-TIER FALLBACK GOES LOCAL (R3, docs/rt_lighting_plan.md §5.4).
    //
    // The prefilter chain is built over the procedural SKY alone, so at the
    // rough end — where the reflection trace has already handed over and the
    // lobe is wide enough to be an average of everything the surface can see —
    // it returns an average of open sky the surface may be nowhere near. Rough
    // metal in a courtyard mirrors a sky it cannot see.
    //
    // The probe field's irradiance is the occluded version of exactly that
    // average (at alpha near 1 the GGX and cosine lobes integrate to nearly the
    // same thing), so above the band it takes over. Scaled by giWeight, so with
    // GI off the branch is not taken and this path is bit-identical.
    float roughSpecWeight = giRoughSpecularWeight(roughness) * giWeight;
    if (roughSpecWeight > 0.0) {
        prefiltered = lerp(prefiltered, giIrradiance, roughSpecWeight);
    }
    float2 envBRDF = brdfLut.SampleLevel(brdfLutSampler, float2(NdotV, roughness), 0).rg;
    // Horizon occlusion: fade lobes that dip below the geometric surface.
    // No-op when N == Ngeo (dot(R, N) == NdotV >= 0).
    float horizon = saturate(1.0 + dot(R, Ngeo));
    horizon *= horizon;
    float3 specular = prefiltered * (F0 * envBRDF.x + envBRDF.y) * horizon;

    // Diffuse is already occluded per-source above; specular keeps the plain
    // scalar AO. Both still take the shadow-driven ambient lift.
    return (diffuse + specular * specularScale * ao) * lift;
}

// Kaplanyan-style specular anti-aliasing: widen perceptual roughness by the
// screen-space normal variance so curved/thin geometry doesn't shimmer.
//
// FRAGMENT-SHADER ONLY: ddx/ddy exist in no other stage, so a compute module
// that wants the BRDF halves above defines PBR_NO_DERIVATIVES and this one
// function goes away. It is the only thing in this header that is not stage
// agnostic, which is why the guard is around it rather than around the file.
#ifndef PBR_NO_DERIVATIVES
float specularAARoughness(float3 N, float roughness) {
    float3 dndx = ddx(N);
    float3 dndy = ddy(N);
    float variance = 0.25 * (dot(dndx, dndx) + dot(dndy, dndy));
    float kernel = min(variance, 0.18);
    return clamp(sqrt(roughness * roughness + kernel), roughness, 1.0);
}
#endif // PBR_NO_DERIVATIVES

#endif // PBR_HLSLI
