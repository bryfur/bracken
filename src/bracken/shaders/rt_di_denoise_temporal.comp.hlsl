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
// History layout:
//   diffuse (RGBA16F): rgb = accumulated demodulated diffuse, a = frame count
//   spec    (RGBA16F): rgb = accumulated demodulated specular, a = Schlick f
//   dir     (RG16F):   octahedral accumulated light direction        [R7b]
//
// The spatial pass downstream replaces the diffuse alpha with the texel's
// linear view depth, which is what the forward reconstruction reads. The count
// is an internal quantity and never leaves this pair.
//
// R7b'S THIRD TARGET IS ACCUMULATED DIFFERENTLY FROM THE OTHER TWO, and has to
// be. A direction is not a radiance: it has no fireflies to clamp, no
// neighbourhood variance worth computing, and — the part that actually bites —
// it cannot be averaged in its stored form. Octahedral coordinates are
// discontinuous across the fold seam, so an EMA taken on the encoded pair walks
// through directions neither input names. So the history is DECODED, mixed as a
// vector with the same alpha the radiance halves use, and RENORMALIZED. That
// makes the mix a (non-uniform-speed) slerp toward the current sample, which is
// the right shape: a reservoir that switched light should swing the direction
// over exactly as many frames as it takes the radiance to follow, and both are
// driven by the same `count`.

#include "rt_sampling.hlsli"
#include "rt_denoise.hlsli"
// For the octahedral direction codec and its cancellation test. The resource
// half of that header is conditional on LOCAL_DI_BINDING_BASE, which this pass
// does not define, so it costs nothing but the maths.
#include "local_di.hlsli"

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
Texture2D<float2> directionIn : register(t2, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
SamplerState directionInSampler : register(s2, space0);

[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
Texture2D<float2> velocityTexture : register(t3, space0);
[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
SamplerState velocitySampler : register(s3, space0);

[[vk::combinedImageSampler]][[vk::binding(4, 0)]]
Texture2D<float4> diffuseHistory : register(t4, space0);
[[vk::combinedImageSampler]][[vk::binding(4, 0)]]
SamplerState diffuseHistorySampler : register(s4, space0);

[[vk::combinedImageSampler]][[vk::binding(5, 0)]]
Texture2D<float4> specularHistory : register(t5, space0);
[[vk::combinedImageSampler]][[vk::binding(5, 0)]]
SamplerState specularHistorySampler : register(s5, space0);

/// The direction history. Sampled with a NEAREST sampler by the caller, unlike
/// the two radiance histories: a bilinear tap on octahedral coordinates crosses
/// the fold seam exactly where a silhouette does, and the reprojected direction
/// is renormalized after decode anyway, so the filtering would buy nothing and
/// cost a wrong direction at every edge.
[[vk::combinedImageSampler]][[vk::binding(6, 0)]]
Texture2D<float2> directionHistory : register(t6, space0);
[[vk::combinedImageSampler]][[vk::binding(6, 0)]]
SamplerState directionHistorySampler : register(s6, space0);

/// R5b's reconstructed A-SVGF gradient at STRATUM resolution, for the LOCAL
/// signal — the same texture the visibility temporal pass reads, dispatched
/// against the same VisibilityDenoiser. A reservoir that changed which light it
/// picked is a lighting change here in exactly the way it is there.
[[vk::binding(7, 0)]] Texture2D<float> gradientTexture : register(t7, space0);

/// render::AutoExposure::state_buffer(): float4 {current EV, target EV, avg
/// log2 luma, valid}.
[[vk::binding(8, 0)]] StructuredBuffer<float4> exposureState : register(t8, space0);

[[vk::image_format("rgba16f")]]
[[vk::binding(0, 1)]] RWTexture2D<float4> diffuseOut : register(u0, space1);
[[vk::image_format("rgba16f")]]
[[vk::binding(1, 1)]] RWTexture2D<float4> specularOut : register(u1, space1);
[[vk::image_format("rg16f")]]
[[vk::binding(2, 1)]] RWTexture2D<float2> directionOut : register(u2, space1);

/// One 3x3 window's statistics, in two forms — and which form you may use is
/// not a matter of taste.
///
/// `mean`/`sigma` are over the WHOLE window, centre included, and drive the
/// HISTORY clamp box. Bounding last frame's estimate by this frame's
/// neighbourhood is the standard TAA construction and the centre belongs in it:
/// history and the window are independent quantities.
///
/// `looMean`/`looSigma` LEAVE THE CENTRE OUT, and only these may judge whether
/// the centre is a firefly. Testing a sample against statistics it contributed
/// to is not a weak test, it is an impossible one: Samuelson's inequality bounds
/// every member of an n-sample set by `mean + sigma*sqrt(n-1)`, so across these
/// 9 taps no value can exceed `mean + 2.83*sigma` — while
/// RT_REFLECTION_FIREFLY_SIGMA is 4.0 and `rtReflectionFireflyCeiling` then
/// takes a `max` with an absolute floor that can only raise it further. The
/// relative half of the R5b clamp could not fire on ANY input, at any exposure,
/// in any scene, which is why the fireflies it was built for outlived it.
struct RtDiWindow {
    float3 diffuseMean;
    float3 diffuseSigma;
    float3 specMean;
    float3 specSigma;
    float valid;
    float3 diffuseLooMean;
    float3 diffuseLooSigma;
    float3 specLooMean;
    float3 specLooSigma;
    float looValid;
};

/// Statistics of one 3x3 window, over texels the trace answered for. Validity
/// comes from the DIFFUSE alpha for both targets, because it is one validity
/// statement about the texel and not two.
RtDiWindow neighbourhood(int2 coord) {
    float3 d1 = 0.0;
    float3 d2 = 0.0;
    float3 s1 = 0.0;
    float3 s2 = 0.0;
    float valid = 0.0;
    // The centre's own contribution, kept aside so it can be subtracted back
    // out exactly rather than re-gathered in a second pass.
    float3 centreD = 0.0;
    float3 centreS = 0.0;
    float centreValid = 0.0;
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
            if (dx == 0 && dy == 0) {
                centreD = d.rgb;
                centreS = s;
                centreValid = 1.0;
            }
        }
    }

    RtDiWindow w;
    w.diffuseMean = 0.0;
    w.diffuseSigma = 0.0;
    w.specMean = 0.0;
    w.specSigma = 0.0;
    w.valid = valid;
    w.diffuseLooMean = 0.0;
    w.diffuseLooSigma = 0.0;
    w.specLooMean = 0.0;
    w.specLooSigma = 0.0;
    w.looValid = max(valid - centreValid, 0.0);

    if (valid > 0.0) {
        w.diffuseMean = d1 / valid;
        w.diffuseSigma = sqrt(max(d2 / valid - w.diffuseMean * w.diffuseMean, 0.0));
        w.specMean = s1 / valid;
        w.specSigma = sqrt(max(s2 / valid - w.specMean * w.specMean, 0.0));
    }
    if (w.looValid > 0.0) {
        const float3 ld1 = d1 - centreD;
        const float3 ld2 = d2 - centreD * centreD;
        const float3 ls1 = s1 - centreS;
        const float3 ls2 = s2 - centreS * centreS;
        w.diffuseLooMean = ld1 / w.looValid;
        w.diffuseLooSigma =
            sqrt(max(ld2 / w.looValid - w.diffuseLooMean * w.diffuseLooMean, 0.0));
        w.specLooMean = ls1 / w.looValid;
        w.specLooSigma = sqrt(max(ls2 / w.looValid - w.specLooMean * w.specLooMean, 0.0));
    }
    return w;
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
        directionOut[coord] = localDiDirEncode(float3(0.0, 1.0, 0.0));
        return;
    }
    float4 currentSpec = specularIn.Load(int3(coord, 0));
    const float3 currentDir = localDiDirDecode(directionIn.Load(int3(coord, 0)).xy);

    const float adaptedEv = rtAdaptedEv(exposureState[0], exposureEnabled, manualExposureEv);

    const RtDiWindow win = neighbourhood(coord);
    const float valid = win.valid;
    float3 diffuseMean = win.diffuseMean;
    float3 diffuseSigma = win.diffuseSigma;
    float3 specMean = win.specMean;
    float3 specSigma = win.specSigma;

    if (valid > 0.0) {
        // The one absolute term in an otherwise scale-free clamp, and therefore
        // the one that has to know the exposure (R5b). Without it a flat region
        // collapses sigma onto this frame's single sample and the accumulation
        // is discarded at exactly the pixels that would have converged best.
        const float sigmaFloor = rtClampSigmaFloor(adaptedEv);
        diffuseSigma = max(diffuseSigma, sigmaFloor.xxx);
        specSigma = max(specSigma, sigmaFloor.xxx);

        // The firefly test takes the LEAVE-ONE-OUT statistics — see RtDiWindow.
        // With `looValid == 0` (an isolated traced texel) both are zero and the
        // ceiling falls back to `4*sigmaFloor` against the absolute perceptual
        // floor, which is the right answer when there is no neighbouring
        // evidence that a bright centre is real.
        current.rgb = clampFirefly(current.rgb, win.diffuseLooMean,
                                   max(win.diffuseLooSigma, sigmaFloor.xxx), adaptedEv);
        currentSpec.rgb = clampFirefly(currentSpec.rgb, win.specLooMean,
                                       max(win.specLooSigma, sigmaFloor.xxx), adaptedEv);
    }

    // How much history this texel is allowed to keep. Same two-source max the
    // visibility temporal pass takes: the global analytic sun signal, and the
    // local traced gradient the reconstruction dilated over this stratum.
    float gradient = 0.0;
    if (gradientValid != 0u) {
        gradient = rtGradientSample(gradientTexture, coord, stratumCount);
    }
    const float reset = saturate(max(lightingReset, gradient));

    float3 accumulated = current.rgb;
    float3 accumulatedSpec = currentSpec.rgb;
    float3 accumulatedDir = currentDir;
    float schlick = currentSpec.a;
    float count = 1.0;

    if (historyValid != 0u) {
        const float2 uv = (float2(coord) + 0.5) / float2(tracedSize);
        const float2 velocity = velocityTexture.SampleLevel(velocitySampler, uv, 0);
        const float2 prevUv = uv - velocity; // velocity = curr - prev (velocity_math.hpp)
        if (all(prevUv >= 0.0) && all(prevUv <= 1.0)) {
            const float4 hist = diffuseHistory.SampleLevel(diffuseHistorySampler, prevUv, 0);

            // NO GEOMETRIC HISTORY VALIDATION HAPPENS HERE, and that is a known
            // gap rather than an oversight (docs/rt_lighting_plan.md §5.9).
            // History is accepted on `prevUv` being on screen and `hist.a > 0`,
            // nothing else, so `count` does not reset on disocclusion — and
            // `count` is what gates R7c's 3->1 filter-radius ramp and its second
            // a-trous iteration, both of which therefore cannot fire on the
            // pixels they were built for.
            //
            // A depth gate was BUILT AND REVERTED on 2026-08-08, and the reasons
            // are worth recording because the next attempt should not repeat
            // either of them.
            //
            // It worked, mechanically: the previous frame's `RtReservoir::depth`
            // lane is last frame's true view depth at that texel, `current.a` is
            // this frame's (`diDiffuse.a = surf.viewDepth`), and comparing them
            // at the standard 0.05 needs no linearization at all. It read the
            // previous reservoir buffer as a second read-only storage buffer.
            //
            // It was reverted for ONE sound reason and one unsound one:
            //
            //  * SOUND: no quality improvement was demonstrable. `di noise` is a
            //    TEMPORAL statistic, and a denoiser that correctly discards
            //    history necessarily has more temporal variation than one that
            //    wrongly keeps it — so the probe moved the wrong way for
            //    structural reasons (0.00014 -> 0.00016 pinned, 0.00384 ->
            //    0.00435 orbiting) while the compensating SPATIAL win, which is
            //    the entire point of letting `count` reset, has no probe at all.
            //    Until there is one, this cannot be evaluated on evidence.
            //  * UNSOUND: it looked like it cost 10-15% of frame time. That
            //    figure is NOT TRUSTWORTHY and should not be quoted. It was
            //    taken with `--rt-bench 120`, which on this machine completes in
            //    ~0.2 s — far too short for the GPU to leave its 255 MHz idle
            //    clock — and against a baseline measured hours earlier. The
            //    RT-OFF bench, whose output is bit-identical and whose code path
            //    none of this touches, drifted from 954 to 853 fps over the same
            //    session, which is the size of the effect being attributed.
            //    Any re-measurement needs long runs (`--rt-bench 3000`) taken
            //    back to back on a quiet machine.
            //
            // The next attempt needs NO new texture, which is worth knowing
            // because "a dedicated R16F depth history" was the cost this note
            // used to quote. `local_di_.filtered[0].a` ALREADY IS a linear view
            // depth: rt_di_denoise_spatial writes `isFinal ? centerDepth :
            // count`, and its final iteration runs strictly AFTER this dispatch
            // inside the same denoise_local_di — so at this point the target
            // still holds LAST frame's depth, in the same units and from the
            // same linearization as `current.a`. Two things it must respect:
            //
            //  * `filtered[]` carries no SAMPLER usage (ensure_di_targets asks
            //    for storage read/write only), so it binds as a second READ-ONLY
            //    STORAGE TEXTURE and is point-`Load`ed at `prevUv`. Filtering
            //    depth across a silhouette manufactures an intermediate value
            //    that passes a test neither side would — the same argument the
            //    direction history makes for its NEAREST sampler.
            //  * The comparison must be made under the PREVIOUS camera. A fixed
            //    relative window between last frame's depth and this frame's is
            //    the error rt_local_shadow_gradient.comp.hlsl documents at
            //    length: at a 420-unit boom the camera's own motion is several
            //    world units per frame, which is under 1% at 500 units but 25%
            //    against a 20-unit foreground prop — so the gate would reject
            //    GOOD history exactly in the near field, pin `count` at 1, and
            //    put the foreground permanently on the widest radius at
            //    `alpha = 1`. Louder, not quieter. Reconstruct the view-space
            //    point from `current.a` and the pixel's NDC, push it through a
            //    `prevViewFromView` uniform, and test THAT against the stored
            //    depth at the standard 0.05 relative window.
            //
            // And it must ship with an escape hatch whose 0 reproduces this
            // image bit for bit, so the cost A/B is one build rather than two
            // sessions — which is the half of the 2026-08-08 measurement that
            // made it worthless.
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
                // The direction: decoded, mixed as a VECTOR, renormalized. The
                // fallback when the mix cancels is this frame's own direction,
                // which is the one input that is certainly a real direction —
                // not the surface normal, because unlike the forward pass this
                // one has a better answer than "flat" available.
                const float3 histDir =
                    localDiDirDecode(directionHistory.SampleLevel(directionHistorySampler,
                                                                  prevUv, 0).xy);
                accumulatedDir =
                    localDiDirNormalize(lerp(histDir, currentDir, alpha), currentDir);
            }
        }
    }

    diffuseOut[coord] = float4(accumulated, count);
    specularOut[coord] = float4(accumulatedSpec, schlick);
    directionOut[coord] = localDiDirEncode(accumulatedDir);
}
