#ifndef LOCAL_DI_HLSLI
#define LOCAL_DI_HLSLI

// local_di.hlsli — the forward half of the Full-DI tier of local-light shadows
// (`LocalRtShadowMode::Full`, phase R5c of docs/rt_lighting_plan.md §5.3).
//
// TRANSLITERATION of render/lighting/local_di.hpp, which is the source of truth,
// carries the derivation of every line below, and is tested with no GPU. Change
// one, change both — `local_di.test.cpp` pins the C++ half against an
// independent transliteration of pbr.hlsli's own evaluatePBR, which is the test
// that keeps the Ratio and Full tiers honest against each other.
//
// WHAT THE TWO BUFFERS HOLD. Neither is a radiance anyone would want to look
// at: both have the per-pixel material factors the traced pass does not know
// DIVIDED OUT, so that the denoiser sees lighting alone and the albedo is
// multiplied back at full resolution, per texel, after the blur.
//
//   diffuse.rgb = (1 - F_d) * (1 - metallic) * (ls*vs/PI) * E     [no albedo]
//   diffuse.a   = linear view depth of the texel  (the spatial pass wrote it)
//   spec.rgb    = D * Vis * F_d * E                               [dielectric F]
//   spec.a      = the Schlick fraction (1 - LdotH)^5 of the winning light
//
// with E = lightColor * atten * cone * NdotL * V * W — the reservoir estimate
// of the WHOLE point+spot sum, not of one light, which is what makes one ray
// enough. F_d is Schlick against F0 = 0.04: the G-buffer carries roughness and
// metallic but not albedo, so a metal's real F0 is not knowable there.
//
// BOTH halves therefore need a Fresnel correction on the way back, and both
// corrections are exact rather than approximate. The diffuse one needs only the
// albedo (Schlick's (1 - f) cancels out of the ratio) — localDiDiffuseTint. The
// specular one does not, which is what the alpha lane is for —
// localDiSpecularTint. render/lighting/local_di.hpp derives both.
//
// RESOURCE DECLARATIONS are conditional on LOCAL_DI_BINDING_BASE, so the
// compute passes that need only the maths can include this file too. In the
// four scene fragment shaders the two textures are READ-ONLY STORAGE TEXTURES,
// not samplers, and that is a hard constraint rather than a preference: those
// shaders sit at SDL's 16-sampler ceiling exactly (see pipeline_catalog.cpp's
// k_fs_lit_model). SDL3 GPU lays fragment set 2 out as sampled textures, then
// STORAGE textures, then storage buffers, so the DI pair goes in the gap
// between the samplers and the cluster SSBOs — which is why
// CLUSTER_SSBO_BINDING_BASE stopped being "the sampler count" and
// CLUSTER_SAMPLER_COUNT was split out of it.

// ---------------------------------------------------------------------------
// The demodulation constants and the multiply-back
// ---------------------------------------------------------------------------

#define LOCAL_DI_DIELECTRIC_F0 0.04

/// The Schlick fraction of a half-vector angle. CPU mirror:
/// `di_schlick_fraction`.
float localDiSchlickFraction(float ldoth) {
    float x = saturate(1.0 - ldoth);
    float x2 = x * x;
    return x2 * x2 * x;
}

/// Scalar dielectric Fresnel — what the trace multiplied the specular lobe by.
/// CPU mirror: `di_dielectric_fresnel`.
float localDiDielectricFresnel(float schlickFraction) {
    return LOCAL_DI_DIELECTRIC_F0 + (1.0 - LOCAL_DI_DIELECTRIC_F0) * schlickFraction;
}

/**
 * The DIFFUSE remodulation hook: `(1 - F_true) / (1 - F_d)`, per channel.
 *
 * Needs no half-vector angle — Schlick's `(1 - f)` factors out of the ratio —
 * so unlike the specular hook this one reads nothing the denoiser touched.
 * 1 for a dielectric, 0 for a white metal, which has no diffuse lobe at all.
 *
 * CPU mirror: `di_diffuse_tint`.
 */
float3 localDiDiffuseTint(float3 albedo, float metallic) {
    return max(1.0 - saturate(metallic) * ((albedo - LOCAL_DI_DIELECTRIC_F0) /
                                           (1.0 - LOCAL_DI_DIELECTRIC_F0)),
               float3(0.0, 0.0, 0.0));
}

/**
 * The specular remodulation hook: `F_true / F_dielectric`, per channel.
 *
 * This is `lerp(1.0, albedoTintFactor, metallic)` with the ALGEBRAICALLY EXACT
 * tint factor rather than a chosen one — see render/lighting/local_di.hpp for
 * the derivation. Exactly 1 for a dielectric, `albedo / 0.04` for a metal at
 * normal incidence, and relaxing to 1 at grazing angles because every
 * material's Fresnel goes to white at the horizon. The denominator is bounded
 * below by 0.04.
 *
 * CPU mirror: `di_specular_tint`.
 */
float3 localDiSpecularTint(float3 albedo, float metallic, float schlickFraction) {
    float f = saturate(schlickFraction);
    float denom = localDiDielectricFresnel(f);
    float3 tint = 1.0 + (albedo - LOCAL_DI_DIELECTRIC_F0) * ((1.0 - f) / denom);
    return lerp(float3(1.0, 1.0, 1.0), tint, saturate(metallic));
}

// ---------------------------------------------------------------------------
// The spatial filter's radius curve (shared with rt_di_denoise_spatial.comp)
// ---------------------------------------------------------------------------

#define LOCAL_DI_FILTER_CONVERGED_COUNT 8.0
#define LOCAL_DI_FILTER_MIN_RADIUS 1.0

/// CPU mirror: `di_filter_radius`.
float localDiFilterRadius(float count, float maxRadius) {
    float t = saturate((max(count, 1.0) - 1.0) / (LOCAL_DI_FILTER_CONVERGED_COUNT - 1.0));
    float wide = max(maxRadius, LOCAL_DI_FILTER_MIN_RADIUS);
    return LOCAL_DI_FILTER_MIN_RADIUS + (wide - LOCAL_DI_FILTER_MIN_RADIUS) * (1.0 - t);
}

/// Relative view-depth difference at which a reconstruction tap is worth half
/// its bilinear weight. The denoiser's own tolerance, not the gradient's tight
/// one: this asks "does this traced texel describe the surface I am shading?",
/// which a neighbouring point on the same wall answers yes to.
#define LOCAL_DI_DEPTH_TOLERANCE 0.05

#ifdef LOCAL_DI_BINDING_BASE

// The two READ-ONLY STORAGE textures. Declared without a sampler pair on
// purpose: DXC only merges a Texture2D with its [[vk::combinedImageSampler]]
// partner where executed code pairs them, so an unpaired, Load()-only texture
// reflects as a storage texture — which is exactly the slot class wanted here
// (shaders/README.md, "Declaring a texture the shader never reads changes the
// reflected layout").
[[vk::binding(LOCAL_DI_BINDING_BASE + 0, 2)]]
Texture2D<float4> localDiDiffuseTexture;
[[vk::binding(LOCAL_DI_BINDING_BASE + 1, 2)]]
Texture2D<float4> localDiSpecularTexture;

/**
 * The reconstruction: a manual 4-tap bilinear from the traced resolution with
 * DEPTH-AWARE weights, in the traced buffers' own texel space.
 *
 * Manual rather than a hardware bilinear tap for the reason the visibility
 * denoiser fuses its upsample into its filter: hardware bilinear weights have
 * no idea a silhouette is there, so a half-resolution signal bleeds a
 * foreground surface's lighting onto the background behind it and crawls along
 * every edge as the camera moves. The alpha lane of the diffuse buffer carries
 * each traced texel's own linear view depth precisely so this test can be made.
 *
 * A tap at depth 0 is sky, or a pixel the trace answered nothing for; it is
 * rejected rather than blended, and when every tap is rejected the result is
 * zero — which is the correct local-light contribution for a pixel no local
 * light reaches.
 */
void localDiSample(float2 uv, float viewDepth, out float3 diffuse, out float3 spec,
                   out float schlickFraction) {
    diffuse = float3(0.0, 0.0, 0.0);
    spec = float3(0.0, 0.0, 0.0);
    schlickFraction = 0.0;

    uint2 size;
    localDiDiffuseTexture.GetDimensions(size.x, size.y);
    if (size.x == 0u || size.y == 0u) {
        return;
    }

    float2 c = uv * float2(size) - 0.5;
    int2 base = int2(floor(c));
    float2 fr = saturate(c - float2(base));

    float4 diffuseSum = float4(0.0, 0.0, 0.0, 0.0);
    float4 specSum = float4(0.0, 0.0, 0.0, 0.0);
    float weightSum = 0.0;
    float tolerance = LOCAL_DI_DEPTH_TOLERANCE * max(viewDepth, 1.0);

    [unroll]
    for (int dy = 0; dy < 2; ++dy) {
        [unroll]
        for (int dx = 0; dx < 2; ++dx) {
            int2 s = clamp(base + int2(dx, dy), int2(0, 0), int2(size) - 1);
            float4 d = localDiDiffuseTexture.Load(int3(s, 0));
            if (d.a <= 0.0) {
                continue; // sky, or a texel the trace answered nothing for
            }
            float wBilinear = (dx == 0 ? 1.0 - fr.x : fr.x) * (dy == 0 ? 1.0 - fr.y : fr.y);
            // Reciprocal rather than an exponential: it falls off fast enough
            // to keep a silhouette and never reaches exactly zero, so a pixel
            // whose four taps are all slightly off still gets an answer instead
            // of falling off the edge of the buffer into black.
            float wDepth = 1.0 / (1.0 + abs(d.a - viewDepth) / max(tolerance, 1e-4));
            float w = wBilinear * wDepth;
            if (w <= 0.0) {
                continue;
            }
            diffuseSum += float4(d.rgb, 0.0) * w;
            specSum += localDiSpecularTexture.Load(int3(s, 0)) * w;
            weightSum += w;
        }
    }

    if (weightSum <= 1e-6) {
        return;
    }
    diffuse = diffuseSum.rgb / weightSum;
    spec = specSum.rgb / weightSum;
    schlickFraction = specSum.a / weightSum;
}

/**
 * The whole Full-tier local-light term for one pixel: fetch, reconstruct,
 * remodulate.
 *
 * Replaces `accumulateClusterLights` outright in Full mode — the point and spot
 * lights are not summed here and are not summed there either, which is what
 * makes this per-light correct where the Ratio tier's one scalar is only
 * aggregate correct.
 */
float3 localDiLighting(float2 uv, float viewDepth, float3 albedo, float metallic) {
    float3 diffuse;
    float3 spec;
    float schlick;
    localDiSample(uv, viewDepth, diffuse, spec, schlick);
    return diffuse * albedo * localDiDiffuseTint(albedo, metallic) +
           spec * localDiSpecularTint(albedo, metallic, schlick);
}

#endif // LOCAL_DI_BINDING_BASE

#endif // LOCAL_DI_HLSLI
