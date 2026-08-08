/**
 * Depth-Prepass G-Buffer Fragment Shader, alpha-MASK variant.
 *
 * prepass_velocity_masked.frag.hlsl (identical alpha test and velocity write
 * -- keep them in sync) plus the second G-buffer attachment, which every
 * pipeline in the pass must write once the target is attached.
 *
 * Roughness is a constant 1 and metallic 0: MASK geometry is foliage, which is
 * never a mirror surface, so the reflection pass skips these pixels above any
 * sane RtReflectionSettings::max_roughness rather than tracing a ray to find
 * out. That part is unchanged.
 *
 * THE NORMAL IS REAL AS OF 2026-08-08, and it used to be a hardcoded oct(+Y)
 * with a comment saying "the reflection pass rejects these on roughness before
 * the normal is ever used". That was true when the G-buffer served only the
 * reflection pass. It stopped being true and the comment did not: the normal is
 * now read by rt_shadow.comp.hlsl (ray-origin normal bias, the N.L early-out,
 * the cone rejection), rt_shadow_gradient, rt_local_shadow's `surfaceAt` --
 * which feeds the RESERVOIR'S TARGET FUNCTION and its GGX terms --
 * rt_local_shadow_gradient, rt_di_denoise_spatial's a-trous normal weight, and
 * gi_screen_probe_place's probe hemisphere.
 *
 * 27 of the playground's 52 .glb assets are alphaMode:MASK -- all the foliage --
 * so a constant +Y was reaching all of that. Three consequences, worst first:
 *
 *  * local_di.hlsli's relief factor divides the shading cosine by the GEOMETRIC
 *    one, on the documented invariant that the geometric normal is "the same
 *    quantity the prepass wrote into the G-buffer and therefore the same cosine
 *    the trace already applied". model.frag.hlsl passes the real interpolated
 *    vertex normal, so on a vertical leaf card facing the light the trace had
 *    applied cos(+Y, L) ~ 0.2 while the remodulation divided by cos(N_geo, L)
 *    ~ 0.9 -- off by ~4.5x, straight into the LOCAL_DI_RELIEF_MAX clamp, at
 *    full resolution, after every denoiser, varying per card. That is the most
 *    direct account of why the Full tier is noisy where Ratio (which has no
 *    relief term and no per-pixel remodulation) is not.
 *  * The DI spatial filter's normal bilateral weight was dead over all of it:
 *    every tap read +Y, so pow(dot(n, nc), power) was 1 and the a-trous
 *    degenerated to a depth-only blur that smeared across every leaf silhouette
 *    -- disabling the one filter that could have suppressed the variance above.
 *  * RIS candidate selection was importance-sampled for a horizontal surface.
 *
 * NOT FLIPPED for back faces, deliberately. MASK draws are two-sided
 * (cull_mode NONE) so a leaf card's interpolated normal often points away from
 * the viewer, and flipping it here would be defensible in isolation -- but the
 * forward pass does not flip either, and the relief invariant above is an
 * equality between THIS value and the `N_geo` model.frag.hlsl passes. Writing
 * exactly what the forward pass reads is what makes that invariant true; a
 * flip on one side only would re-break it in the other direction.
 */

#include "oct_encode.hlsli" // octEncode

[[vk::binding(0, 3)]]
cbuffer PrepassMaskedUniforms {
    float alphaCutoff;
    float3 _padding0;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]] Texture2D baseColorTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]] SamplerState baseColorSampler : register(s0, space2);

struct PSInput {
    float4 position : SV_Position;
    float4 currClip : TEXCOORD0;
    float4 prevClip : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float3 worldNormal : TEXCOORD3;
};

struct PSOutput {
    float2 velocity : SV_Target0;
    float4 normalRough : SV_Target1;
};

PSOutput PSMain(PSInput input) {
    // EXACT forward alpha test — see prepass_velocity_masked.frag.hlsl.
    float alpha = baseColorTexture.Sample(baseColorSampler, input.uv).a;
    float mip = baseColorTexture.CalculateLevelOfDetailUnclamped(baseColorSampler, input.uv);
    alpha *= 1.0 + max(mip, 0.0) * 0.35;
    clip(alpha - alphaCutoff);

    PSOutput output;
    if (input.currClip.w <= 0.0 || input.prevClip.w <= 0.0) {
        output.velocity = float2(0.0, 0.0);
    } else {
        float2 currNdc = input.currClip.xy / input.currClip.w;
        float2 prevNdc = input.prevClip.xy / input.prevClip.w;
        output.velocity = (currNdc - prevNdc) * float2(0.5, -0.5);
    }

    // Interpolation shortens the normal; renormalize before encoding.
    float3 n = normalize(input.worldNormal);
    output.normalRough = float4(octEncode(n), 1.0, 0.0);
    return output;
}
