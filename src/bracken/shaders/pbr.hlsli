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
                  float3 giIrradiance = float3(0, 0, 0), float giWeight = 0.0) {
    float NdotV = max(dot(N, V), 1e-4);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);

    // Diffuse: hemisphere gradient on the normal, energy split via roughness-
    // aware Fresnel (no Burley — irradiance is already cosine-integrated).
    float3 F = fresnelSchlickRoughness(NdotV, F0, roughness);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    float3 irradiance = lerp(groundColor, skyColor, N.y * 0.5 + 0.5);
    if (giWeight > 0.0) {
        irradiance = lerp(irradiance, giIrradiance, giWeight);
    }
    float3 diffuse = kd * baseColor * irradiance;

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

    return (diffuse + specular) * ao;
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
                  // applies `ao * lift` to everything at the end. The specular
                  // half is untouched until R4.
                  float3 giIrradiance = float3(0, 0, 0), float giWeight = 0.0) {
    float NdotV = max(dot(N, V), 1e-4);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);

    // Diffuse: cosine-convolved sky irradiance along the normal, energy split
    // via roughness-aware Fresnel (irradiance is already cosine-integrated).
    float3 F = fresnelSchlickRoughness(NdotV, F0, roughness);
    float3 kd = (1.0 - F) * (1.0 - metallic);
    float3 irradiance = irradianceMap.SampleLevel(irradianceSampler, N, 0).rgb;
    if (giWeight > 0.0) {
        irradiance = lerp(irradiance, giIrradiance, giWeight);
    }
    float3 diffuse = kd * baseColor * irradiance;

    // Specular: prefiltered radiance along R at the roughness-matched mip,
    // scaled/biased by the split-sum LUT.
    float3 R = reflect(-V, N);
    float mip = roughness * (IBL_SPECULAR_MIPS - 1.0);
    float3 prefiltered = specularMap.SampleLevel(specularSampler, R, mip).rgb;
    float2 envBRDF = brdfLut.SampleLevel(brdfLutSampler, float2(NdotV, roughness), 0).rg;
    // Horizon occlusion: fade lobes that dip below the geometric surface.
    // No-op when N == Ngeo (dot(R, N) == NdotV >= 0).
    float horizon = saturate(1.0 + dot(R, Ngeo));
    horizon *= horizon;
    float3 specular = prefiltered * (F0 * envBRDF.x + envBRDF.y) * horizon;

    return (diffuse + specular * specularScale) * (ao * lift);
}

// Kaplanyan-style specular anti-aliasing: widen perceptual roughness by the
// screen-space normal variance so curved/thin geometry doesn't shimmer.
// Fragment-shader only (uses derivatives).
float specularAARoughness(float3 N, float roughness) {
    float3 dndx = ddx(N);
    float3 dndy = ddy(N);
    float variance = 0.25 * (dot(dndx, dndx) + dot(dndy, dndy));
    float kernel = min(variance, 0.18);
    return clamp(sqrt(roughness * roughness + kernel), roughness, 1.0);
}

#endif // PBR_HLSLI
