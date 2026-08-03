// rt_shadow_denoise_temporal.comp.hlsl — temporal accumulation, pass 2 of 3.
//
// One thread per TRACED pixel. Unpacks this frame's single stochastic sample,
// reprojects last frame's estimate through the velocity buffer, and merges
// them into a running mean plus the second moment the spatial filter needs to
// know how much it still has to blur.
//
// THIS PASS IS WHERE THE NOISE ACTUALLY GOES. One sun-disk sample per pixel is
// ~50% error in the penumbra; sixteen accumulated frames is ~12%; the spatial
// filter only has to clean up what temporal cannot reach — disocclusions, and
// the first frames after the camera cuts.
//
// History layout (RGBA16F): R = mean visibility, G = second moment,
// B = accumulated sample count, A = linear view depth at the time of writing.
// Depth rides in the history rather than in a separate prev-depth target
// because that is one screen-sized texture the engine then does not allocate,
// and the value is only ever needed at exactly the texel the history is read
// from.
//
// ---------------------------------------------------------------------------
// R5b: HISTORY IS NOW INVALIDATED BY LIGHTING CHANGE, NOT ONLY BY MOTION
// (docs/rt_lighting_plan.md §5.6)
// ---------------------------------------------------------------------------
//
// Everything above answers "is this history describing the same SURFACE?".
// Nothing in it answers "is it describing the same LIGHTING?" — and a velocity
// buffer cannot, because a cloud crossing the sun moves no pixel. Three
// mechanisms now do, in increasing order of locality:
//
//  1. `lightingReset` — the ANALYTIC sun gradient. The renderer computed, on
//     the CPU and before any ray was cast, how far the sun moved and how much
//     its radiance changed since the last frame, and mapped that through
//     `render/rt_denoise_math.hpp`'s documented curve. Global, free, and
//     exactly zero for a static sun — which is what makes the pre-R5b path
//     bit-identical rather than merely close.
//  2. `gradientTexture` — the TRACED sparse gradient (A-SVGF). One re-traced
//     ray per 3x3 stratum said whether THIS surface's visibility answer changed
//     with motion and sampling noise factored out; the reconstruction pass
//     dilated that verdict over its neighbourhood. Local, and the only one of
//     the three that can tell a moving occluder's shadow from its surroundings.
//  3. The DUAL FAST/SLOW history below, which catches what the other two miss:
//     a change inside a stratum that this frame's ray did not land on.
//
// The first two combine with MAX into one reset in [0, 1] and collapse the
// accumulated sample count toward 1 — not by throwing history away outright,
// which would put a frame of raw single-sample noise on screen at every
// flicker, but by re-weighting it so this frame's sample dominates. A reset of
// 0 leaves `count` untouched, bit for bit.
//
// ---------------------------------------------------------------------------
// DUAL FAST/SLOW HISTORY
// ---------------------------------------------------------------------------
//
// Beside the main accumulator runs a four-frame mean of the same samples
// (RG16F: mean, count). It is far too noisy to display — that is the point. It
// is a CHANGE DETECTOR, not an estimator: when the main mean drifts outside
// what the fast mean's own sampling error can explain, the main mean is clamped
// back toward it. `rtFastHistoryMargin` derives that tolerance from the fast
// mean itself, so a region that has been unanimously shadowed for four frames
// pins a stale bright history immediately while a genuine half-penumbra — where
// four binary samples resolve nothing — is left alone.
//
// This is the FidelityFX/NRD "fast history clamp" idea, and it is what covers
// the reservoir switching to a different light on the local signal, where a
// gradient replay deliberately reports nothing (rt_local_shadow_gradient).

#include "rt_sampling.hlsli"
#include "rt_denoise.hlsli"
#include "depth_linear.hlsli"

[[vk::binding(0, 2)]]
cbuffer RtShadowTemporalUniforms : register(b0, space2) {
    uint2 tracedSize;     // resolution the rays were traced at
    uint2 tileCount;      // packed-bit texture dimensions
    uint2 stratumCount;   // gradient texture dimensions
    float depthA;         // projection[2][2], for linearizing reversed-Z depth
    float depthB;         // projection[3][2]
    /// Upper bound on accumulated samples. Higher converges further but lags
    /// longer behind moving occluders; 32 is about where a moving shadow stops
    /// looking sharp and starts looking smeared.
    float maxAccumFrames;
    /// Relative depth difference above which history is rejected. Relative,
    /// not absolute, because a fixed epsilon that works at 2 m is meaningless
    /// at 400 m.
    float depthRejectRelative;
    /// The analytic sun gradient's global hysteresis floor, [0, 1]. 0 is the
    /// identity and the default for every frame whose sun did not move.
    float lightingReset;
    uint historyValid;    // 0 on the first frame and after any resize
    /// 0 when no traced gradient was produced this frame (first frame, or the
    /// gradient pipeline is unavailable), in which case the bound texture is a
    /// valid stand-in that must not be read.
    uint gradientValid;
    /// First of this signal's four uint slots in the shared statistics buffer.
    /// The sun's are 0..3 and the local signal's 4..7 — see RtRenderer.
    uint counterBase;
    uint _pad0;
    uint _pad1;
};

// BINDING ORDER IS NOT FREE. Within set 0 SDL3 GPU assigns sampled textures
// first, then read-only storage textures, then read-only storage buffers, all
// numbered contiguously (gpu_compute.hpp). Declaring the storage textures
// first compiles and reflects fine and then silently binds the samplers into
// the slots this shader reads as storage — which returns garbage visibility
// bits for every pixel, i.e. exactly the full-screen speckle that looks like
// a broken denoiser rather than a broken binding.
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float> sceneDepth : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState sceneDepthSampler : register(s0, space0);

[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<float2> velocityTexture : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState velocitySampler : register(s1, space0);

[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
Texture2D<float4> historyIn : register(t2, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
SamplerState historySampler : register(s2, space0);

/// The fast history's own reprojected read (RG16F: mean, count). A separate
/// texture rather than two more channels of the main one: RGBA16F is full, and
/// a five-channel history would mean a second target either way.
[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
Texture2D<float2> fastHistoryIn : register(t3, space0);
[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
SamplerState fastHistorySampler : register(s3, space0);

[[vk::binding(4, 0)]] Texture2D<uint> shadowBits : register(t4, space0);
[[vk::binding(5, 0)]] Texture2D<uint> tileClass : register(t5, space0);
/// Reconstructed A-SVGF gradient at STRATUM resolution (traced / 3).
[[vk::binding(6, 0)]] Texture2D<float> gradientTexture : register(t6, space0);

[[vk::binding(0, 1)]] RWTexture2D<float4> historyOut : register(u0, space1);
[[vk::binding(1, 1)]] RWTexture2D<float2> fastHistoryOut : register(u1, space1);
/// Monotonically increasing diagnostics, differenced by the CPU between two
/// lagged readbacks. Never reset — unsigned wraparound differences exactly, and
/// a reset pass would cost a copy pass every frame to serve a statistic.
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> denoiseCounters : register(u2, space1);

/// Reversed-Z depth -> positive linear view depth, `|depthB / (d + depthA)|`.
/// depth_linear.hlsli is the one home for that expression; depthA/depthB are
/// the projection terms so this does not need the whole matrix.
///
/// This pass carried `depthB / max(d - depthA, 1e-6)` until 2026-08-02 — the
/// sign of the depthA term inverted, which reads as t / (1 - 2t/far). It is
/// self-consistent (`historyOut.a` below is written with the same expression it
/// is later compared against), so the visible effect was not a wrong picture
/// but a wrong SCALE: the reject threshold's effective value was
/// depthRejectRelative * (1 - 2t/far) instead of depthRejectRelative, and past
/// far/2 the clamp pinned every pixel to one constant, disabling rejection
/// entirely out there. See contracts/depth_linear.hpp.
float linearizeDepth(float d) {
    return brackenLinearViewDepth(d, depthA, depthB);
}

// One partial per group, so the statistics cost three global atomics per 64
// pixels rather than three per pixel.
groupshared uint g_accumSum;
groupshared uint g_accumPixels;
groupshared uint g_gradientFired;
groupshared uint g_pixels;

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    if (gi == 0) {
        g_accumSum = 0;
        g_accumPixels = 0;
        g_gradientFired = 0;
        g_pixels = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // NO EARLY RETURNS past this point: the group-wide barriers below must be
    // reached by every thread, including the out-of-bounds padding lanes and
    // the uniform-tile fast path.
    const bool inBounds = tid.x < tracedSize.x && tid.y < tracedSize.y;
    const int2 coord = int2(tid.xy);

    if (inBounds) {
        const uint2 tile = uint2(coord) / RT_TILE_SIZE;
        const uint2 pixelInTile = uint2(coord) % RT_TILE_SIZE;

        const uint bits = shadowBits.Load(int3(int2(tile), 0));
        const float sample = rtUnpackVisibility(bits, pixelInTile);

        const float2 uv = (float2(coord) + 0.5) / float2(tracedSize);

        uint2 depthSize;
        sceneDepth.GetDimensions(depthSize.x, depthSize.y);
        const int2 depthCoord = int2(uv * float2(depthSize));
        const float rawDepth = sceneDepth.Load(int3(depthCoord, 0));
        const float linearDepth = rawDepth > 0.0 ? linearizeDepth(rawDepth) : 0.0;

        // ---- how much history this pixel is allowed to keep ---------------
        float gradient = 0.0;
        if (gradientValid != 0u) {
            const int2 g = clamp(coord / RT_GRADIENT_STRATUM, int2(0, 0), int2(stratumCount) - 1);
            gradient = gradientTexture.Load(int3(g, 0));
        }
        const float reset = saturate(max(lightingReset, gradient));

        InterlockedAdd(g_pixels, 1u);
        if (gradient >= RT_GRADIENT_FIRE_THRESHOLD) {
            InterlockedAdd(g_gradientFired, 1u);
        }

        const uint verdict = tileClass.Load(int3(int2(tile), 0));
        if (verdict != 2u) {
            // A uniform neighbourhood carries no variance and needs no history:
            // write the converged answer directly. This also stops the
            // accumulator from dragging a stale penumbra across a region that
            // has become fully lit. The fast history is written to match, so it
            // does not go stale under a tile that flips back to mixed.
            historyOut[coord] = float4(sample, sample * sample, maxAccumFrames, linearDepth);
            fastHistoryOut[coord] = float2(sample, RT_FAST_HISTORY_WINDOW);
        } else {
            float mean = sample;
            float moment2 = sample * sample;
            float count = 1.0;
            float fastMean = sample;
            float fastCount = 1.0;

            if (historyValid != 0u) {
                const float2 velocity = velocityTexture.SampleLevel(velocitySampler, uv, 0);
                const float2 prevUv = uv - velocity; // velocity = curr - prev (velocity_math.hpp)

                if (all(prevUv >= 0.0) && all(prevUv <= 1.0)) {
                    const float4 hist = historyIn.SampleLevel(historySampler, prevUv, 0);

                    // Reject history across a depth discontinuity. Without this,
                    // a disocclusion drags the occluder's shadow along behind it
                    // — the classic ghost trailing anything that moves.
                    const float depthDelta = abs(hist.a - linearDepth);
                    const bool depthOk =
                        linearDepth > 0.0 && depthDelta <= depthRejectRelative * max(linearDepth, 1.0);

                    if (depthOk) {
                        count = rtApplyReset(min(hist.b + 1.0, maxAccumFrames), reset);
                        const float alpha = 1.0 / count;
                        mean = lerp(hist.r, sample, alpha);
                        moment2 = lerp(hist.g, sample * sample, alpha);

                        const float2 fastHist =
                            fastHistoryIn.SampleLevel(fastHistorySampler, prevUv, 0);
                        fastCount = rtApplyReset(
                            min(fastHist.y + 1.0, RT_FAST_HISTORY_WINDOW), reset);
                        fastMean = lerp(fastHist.x, sample, 1.0 / fastCount);

                        // The fast clamp. Shifting the second moment by the same
                        // amount keeps variance = moment2 - mean^2 exactly what
                        // it was, so the spatial filter downstream does not read
                        // a negative variance out of a clamped mean.
                        const float margin = rtFastHistoryMargin(fastMean, fastCount);
                        const float clamped = clamp(mean, fastMean - margin, fastMean + margin);
                        moment2 += clamped * clamped - mean * mean;
                        mean = clamped;
                    }
                }
            }

            historyOut[coord] = float4(mean, moment2, count, linearDepth);
            fastHistoryOut[coord] = float2(fastMean, fastCount);

            InterlockedAdd(g_accumSum, uint(round(count)));
            InterlockedAdd(g_accumPixels, 1u);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (gi == 0) {
        // Four monotone totals. The CPU differences two lagged readbacks, so
        // wraparound is not merely tolerated but exact.
        InterlockedAdd(denoiseCounters[counterBase + 0u], g_accumSum);
        InterlockedAdd(denoiseCounters[counterBase + 1u], g_accumPixels);
        InterlockedAdd(denoiseCounters[counterBase + 2u], g_gradientFired);
        InterlockedAdd(denoiseCounters[counterBase + 3u], g_pixels);
    }
}
