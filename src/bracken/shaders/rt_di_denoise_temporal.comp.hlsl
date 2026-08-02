// rt_di_denoise_temporal.comp.hlsl — temporal accumulation for the Full-DI
// tier's two demodulated local-light radiance estimates (R5c,
// docs/rt_lighting_plan.md §5.3 + §5.6).
//
// THIS IS THE REFLECTION TEMPORAL PASS'S SHAPE, INSTANTIATED FOR A SECOND
// SIGNAL — sigma-clamped EMA against the neighbourhood, R5b's exposure-aware
// sigma floor and firefly ceiling, R5b's gradient as the reset driver. It is
// not the reflection pipeline itself, and the two things that stop it being so
// are worth naming, because "reuse the pipeline" was the first thing tried
// (R2b's precedent, where the local visibility signal reuses all three of the
// sun's denoiser pipelines verbatim):
//
//  1. THE ALPHA LANE MEANS SOMETHING HERE. rt_reflection_denoise_temporal owns
//     alpha as its accumulated sample count and passes `a <= 0` through as "not
//     reflective". This signal needs alpha for data — the Schlick fraction the
//     forward shader rebuilds a metal's Fresnel from, which is the entire
//     reason the metallic remodulation is exact rather than a fudge — so the
//     count cannot live there for BOTH targets.
//  2. TWO TARGETS, ONE REPROJECTION. Diffuse and specular are estimated from
//     the same reservoir at the same texel and reproject through the same
//     velocity. Two dispatches of a one-target pipeline would do that work
//     twice and, worse, could clamp them against neighbourhoods computed from
//     different validity masks.
//
// What is deliberately NOT taken from the reflection pass: virtual-hit-point
// reprojection. A reflection is an image of a point that is not on the surface;
// direct lighting is not. The surface's own motion vector is the correct one
// here, and it is also the quiet one.
//
// History layout, both targets RGBA16F:
//   diffuse: rgb = accumulated demodulated diffuse, a = accumulated frame count
//   spec:    rgb = accumulated demodulated specular, a = accumulated Schlick f
//
// The spatial pass downstream replaces the diffuse alpha with the texel's
// linear view depth, which is what the forward reconstruction reads. The count
// is an internal quantity and never leaves this pair.

#include "rt_sampling.hlsli"
#include "rt_denoise.hlsli"

[[vk::binding(0, 2)]]
cbuffer RtDiTemporalUniforms : register(b0, space2) {
    uint2 tracedSize;
    /// Gradient texture dimensions (traced / RT_GRADIENT_STRATUM).
    uint2 stratumCount;
    /// Upper bound on accumulated frames. The local signal's, not the
    /// reflection signal's: direct lighting does not move at twice the parallax
    /// rate of the surface carrying it, so a longer tail is free here.
    float maxAccumFrames;
    /// Neighbourhood standard deviations the reprojected history may sit
    /// outside before being clamped back.
    float clampSigma;
    /// R5b's analytic sun gradient, already reduced to a global hysteresis
    /// floor. 0 is the identity.
    float lightingReset;
    uint historyValid;
    /// 0 when no traced gradient was reconstructed this frame, in which case
    /// the bound texture is a valid stand-in that must not be read.
    uint gradientValid;
    uint exposureEnabled;
    float manualExposureEv;
    uint _pad0;
};

// Set 0: sampled textures first, then read-only storage textures, then
// read-only storage buffers. Not a style choice — see shaders/README.md.
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float4> diffuseIn : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState diffuseInSampler : register(s0, space0);

[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<float4> specularIn : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState specularInSampler : register(s1, space0);

[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
Texture2D<float2> velocityTexture : register(t2, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
SamplerState velocitySampler : register(s2, space0);

[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
Texture2D<float4> diffuseHistory : register(t3, space0);
[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
SamplerState diffuseHistorySampler : register(s3, space0);

[[vk::combinedImageSampler]][[vk::binding(4, 0)]]
Texture2D<float4> specularHistory : register(t4, space0);
[[vk::combinedImageSampler]][[vk::binding(4, 0)]]
SamplerState specularHistorySampler : register(s4, space0);

/// R5b's reconstructed A-SVGF gradient at STRATUM resolution, for the LOCAL
/// signal — the same texture the visibility temporal pass reads, dispatched
/// against the same VisibilityDenoiser. A reservoir that changed which light it
/// picked is a lighting change here in exactly the way it is there.
[[vk::binding(5, 0)]] Texture2D<float> gradientTexture : register(t5, space0);

/// render::AutoExposure::state_buffer(): float4 {current EV, target EV, avg
/// log2 luma, valid}.
[[vk::binding(6, 0)]] StructuredBuffer<float4> exposureState : register(t6, space0);

[[vk::image_format("rgba16f")]]
[[vk::binding(0, 1)]] RWTexture2D<float4> diffuseOut : register(u0, space1);
[[vk::image_format("rgba16f")]]
[[vk::binding(1, 1)]] RWTexture2D<float4> specularOut : register(u1, space1);

/// Neighbourhood mean and standard deviation of one 3x3 window, over texels the
/// trace answered for. `validMask` comes from the DIFFUSE alpha for both
/// targets, because it is one validity statement about the texel and not two.
void neighbourhood(int2 coord, out float3 diffuseMean, out float3 diffuseSigma,
                   out float3 specMean, out float3 specSigma, out float valid) {
    float3 d1 = 0.0;
    float3 d2 = 0.0;
    float3 s1 = 0.0;
    float3 s2 = 0.0;
    valid = 0.0;
    [unroll]
    for (int dy = -1; dy <= 1; ++dy) {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx) {
            const int2 p = clamp(coord + int2(dx, dy), int2(0, 0), int2(tracedSize) - 1);
            const float4 d = diffuseIn.Load(int3(p, 0));
            if (d.a <= 0.0) {
                continue;
            }
            const float3 s = specularIn.Load(int3(p, 0)).rgb;
            d1 += d.rgb;
            d2 += d.rgb * d.rgb;
            s1 += s;
            s2 += s * s;
            valid += 1.0;
        }
    }
    diffuseMean = 0.0;
    diffuseSigma = 0.0;
    specMean = 0.0;
    specSigma = 0.0;
    if (valid > 0.0) {
        diffuseMean = d1 / valid;
        diffuseSigma = sqrt(max(d2 / valid - diffuseMean * diffuseMean, 0.0));
        specMean = s1 / valid;
        specSigma = sqrt(max(s2 / valid - specMean * specMean, 0.0));
    }
}

/// The R5b firefly clamp, rescaling the whole colour so a clamped sample keeps
/// its hue. Shared by both targets — a demodulated diffuse estimate can spike
/// exactly as a specular one can, because the spike comes from the reservoir's
/// unbiasing weight W and not from the lobe.
float3 clampFirefly(float3 value, float3 mean, float3 sigma, float adaptedEv) {
    const float luma = luminance(value);
    const float ceiling = rtReflectionFireflyCeiling(luminance(mean), luminance(sigma), adaptedEv);
    if (luma > ceiling && luma > 1e-6) {
        return value * (ceiling / luma);
    }
    return value;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= tracedSize.x || tid.y >= tracedSize.y) {
        return;
    }
    const int2 coord = int2(tid.xy);

    float4 current = diffuseIn.Load(int3(coord, 0));
    if (current.a <= 0.0) {
        // Sky, or a texel the trace answered nothing for. Zero rather than
        // history: an alpha of 0 is what the forward reconstruction rejects.
        diffuseOut[coord] = float4(0.0, 0.0, 0.0, 0.0);
        specularOut[coord] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    float4 currentSpec = specularIn.Load(int3(coord, 0));

    const float adaptedEv = rtAdaptedEv(exposureState[0], exposureEnabled, manualExposureEv);

    float3 diffuseMean;
    float3 diffuseSigma;
    float3 specMean;
    float3 specSigma;
    float valid;
    neighbourhood(coord, diffuseMean, diffuseSigma, specMean, specSigma, valid);

    if (valid > 0.0) {
        // The one absolute term in an otherwise scale-free clamp, and therefore
        // the one that has to know the exposure (R5b). Without it a flat region
        // collapses sigma onto this frame's single sample and the accumulation
        // is discarded at exactly the pixels that would have converged best.
        const float sigmaFloor = rtClampSigmaFloor(adaptedEv);
        diffuseSigma = max(diffuseSigma, sigmaFloor.xxx);
        specSigma = max(specSigma, sigmaFloor.xxx);

        current.rgb = clampFirefly(current.rgb, diffuseMean, diffuseSigma, adaptedEv);
        currentSpec.rgb = clampFirefly(currentSpec.rgb, specMean, specSigma, adaptedEv);
    }

    // How much history this texel is allowed to keep. Same two-source max the
    // visibility temporal pass takes: the global analytic sun signal, and the
    // local traced gradient the reconstruction dilated over this stratum.
    float gradient = 0.0;
    if (gradientValid != 0u) {
        const int2 g = clamp(coord / RT_GRADIENT_STRATUM, int2(0, 0), int2(stratumCount) - 1);
        gradient = gradientTexture.Load(int3(g, 0));
    }
    const float reset = saturate(max(lightingReset, gradient));

    float3 accumulated = current.rgb;
    float3 accumulatedSpec = currentSpec.rgb;
    float schlick = currentSpec.a;
    float count = 1.0;

    if (historyValid != 0u) {
        const float2 uv = (float2(coord) + 0.5) / float2(tracedSize);
        const float2 velocity = velocityTexture.SampleLevel(velocitySampler, uv, 0);
        const float2 prevUv = uv - velocity; // velocity = curr - prev (velocity_math.hpp)
        if (all(prevUv >= 0.0) && all(prevUv <= 1.0)) {
            const float4 hist = diffuseHistory.SampleLevel(diffuseHistorySampler, prevUv, 0);
            if (hist.a > 0.0 && valid > 0.0) {
                const float4 histSpec =
                    specularHistory.SampleLevel(specularHistorySampler, prevUv, 0);

                count = rtApplyReset(min(hist.a + 1.0, maxAccumFrames), reset);
                const float alpha = 1.0 / count;

                accumulated = lerp(clamp(hist.rgb, diffuseMean - diffuseSigma * clampSigma,
                                         diffuseMean + diffuseSigma * clampSigma),
                                   current.rgb, alpha);
                accumulatedSpec = lerp(clamp(histSpec.rgb, specMean - specSigma * clampSigma,
                                             specMean + specSigma * clampSigma),
                                       currentSpec.rgb, alpha);
                // The Schlick fraction is geometry, not radiance: it is bounded
                // in [0, 1] by construction and has no fireflies, so it takes
                // the EMA and none of the clamps.
                schlick = lerp(histSpec.a, currentSpec.a, alpha);
            }
        }
    }

    diffuseOut[coord] = float4(accumulated, count);
    specularOut[coord] = float4(accumulatedSpec, schlick);
}
