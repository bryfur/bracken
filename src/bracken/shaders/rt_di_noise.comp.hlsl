// rt_di_noise.comp.hlsl — the Full-DI tier's TEMPORAL-NOISE PROBE.
//
// WHY THIS EXISTS. The complaint that drove the R7 package was "the Full-DI tier
// is very noisy", and the smoke test structurally cannot see it: a screenshot's
// mean luma and standard deviation are statistics of ONE frame, and temporal
// noise is a statistic ACROSS frames. A denoiser change can halve the crawl and
// move mean_luma by 0.0. So the claim needed an instrument before it could have
// a number, and this is it.
//
// WHAT IT MEASURES. The mean absolute frame-to-frame delta of the FINAL DI
// buffers — the ones the forward shaders actually read — reduced to one scalar
// per texel:
//
//     L      = luminance(diffuse.rgb) + luminance(spec.rgb)
//     sample = |L - L_previous_frame|
//
// summed over every texel the trace answered for, alongside the mean level, so
// the figure can be read absolutely and relative to the signal it is measuring.
// Under a pinned camera and a static sun the delta is TEMPORAL NOISE and nothing
// else; under `PLAYGROUND_SUN_SWEEP` it also contains the legitimate change,
// which is why the harness reports both conditions and compares like with like.
//
// WHY THE DEMODULATED BUFFERS AND NOT THE FRAME. Because that is where the
// signal under test lives. The composited frame carries the sun, the sky, TAA,
// bloom, the tonemap and auto-exposure, every one of which would move the number
// for reasons that have nothing to do with the DI denoiser. The demodulated
// pair has the albedo divided out and the local lights alone in it.
//
// COST AND WHEN IT RUNS. One dispatch over the traced resolution, three loads
// and one store per texel, and it is OFF unless
// `RtLocalShadowSettings::di_noise_probe` is set — which only `--rt-bench` does.
// A diagnostic that a shipping frame pays for is a diagnostic that will be
// turned off and then rot.
//
// THE ACCUMULATION IS MONOTONE AND FIXED POINT, for the same two reasons the
// R5b denoiser counters next to it are: an atomic float add does not exist here,
// and never resetting means the CPU can DIFFERENCE two lagged readbacks instead
// of paying a clear pass every frame.
//
// TWO CLAMPS, ONE SCALE. The delta lane needs a TIGHT ceiling — one firefly the
// denoiser let through must not become the frame's average — and the level lane
// needs a WIDE one, because a clamp low enough to bound the delta truncates the
// demodulated specular tail, and that is not harmless conservatism: it makes the
// reported level DROP whenever a change reduces the tail, so a denoiser
// improvement reads as a signal loss. The first version of this probe shared a
// clamp of 4 and reported exactly that, while the frame it came from got
// brighter rather than dimmer.
//
// ROUNDED, not truncated. Truncation is a one-sided error, and one-sided errors
// on a quantity 5e-4 wide against a quantization of 1e-3 do not average out —
// they bias the whole metric toward zero, which is the direction that would
// flatter the change under test. Rounding leaves a symmetric error that averages
// to nothing over ~360 k texels.
//
// OVERFLOW. The differencing is exact under unsigned wraparound as long as fewer
// than 2^32 counts accumulate between two readbacks. At half res on a 1600x900
// frame and a ten-frame readback interval that bounds the DELTA lane's mean at
// ~1.2 — against a measured delta of 5e-4, three orders of headroom, and the
// level lane sits under the same bound at a measured 4e-3. A frame in which
// every texel saturated its clamp would exceed it; such a frame is not a noise
// measurement anyway.

#include "rt_sampling.hlsli" // luminance, via lighting.hlsli

[[vk::binding(0, 2)]]
cbuffer RtDiNoiseUniforms : register(b0, space2) {
    uint2 tracedSize;
    /// First of this probe's uints in the shared denoiser counter buffer.
    uint counterBase;
    /// 0 on the first probed frame, after a resize, and after any frame the
    /// tier was off — in which case `prevLuma` holds either nothing or a
    /// measurement of a different image, and the delta would be meaningless.
    /// The texel is still WRITTEN, so the next frame has a baseline.
    uint historyValid;
};

[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float4> diffuseIn : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState diffuseInSampler : register(s0, space0);

[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<float4> specularIn : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState specularInSampler : register(s1, space0);

/// Last frame's scalar. Read AND written, which is why it is a storage image
/// with an explicit format rather than a sampled texture.
[[vk::image_format("r16f")]]
[[vk::binding(0, 1)]] RWTexture2D<float> prevLuma : register(u0, space1);
[[vk::binding(1, 1)]] RWStructuredBuffer<uint> denoiseCounters : register(u1, space1);

/// ONE SCALE, TWO CLAMPS. 1024 counts per unit puts a typical 5e-4 delta at half
/// a count and a typical 4e-3 level at four, which rounding turns into unbiased
/// estimates of both over ~360 k texels. The clamps differ because the two lanes
/// need different things from their tails.
#define RT_DI_NOISE_SCALE 1024.0
/// Delta: a firefly the denoiser let through must not dominate a frame's mean.
#define RT_DI_NOISE_DELTA_CLAMP 8.0
/// Level: WIDE, so the demodulated specular tail is measured rather than
/// truncated — see the header for why that distinction decides whether the
/// metric is honest.
#define RT_DI_NOISE_LEVEL_CLAMP 64.0

groupshared uint g_delta;
groupshared uint g_level;
groupshared uint g_pixels;

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    if (gi == 0) {
        g_delta = 0;
        g_level = 0;
        g_pixels = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    const int2 coord = int2(tid.xy);
    if (tid.x < tracedSize.x && tid.y < tracedSize.y) {
        const float4 d = diffuseIn.Load(int3(coord, 0));
        // The diffuse alpha is the linear view depth by this point in the
        // chain: > 0 is "the trace answered for this texel", 0 is sky.
        if (d.a > 0.0) {
            const float3 s = specularIn.Load(int3(coord, 0)).rgb;
            const float level = luminance(d.rgb) + luminance(s);
            if (historyValid != 0u) {
                const float delta = abs(level - prevLuma[coord]);
                InterlockedAdd(g_delta,
                               uint(round(min(delta, RT_DI_NOISE_DELTA_CLAMP) * RT_DI_NOISE_SCALE)));
                InterlockedAdd(g_level,
                               uint(round(min(level, RT_DI_NOISE_LEVEL_CLAMP) * RT_DI_NOISE_SCALE)));
                InterlockedAdd(g_pixels, 1u);
            }
            prevLuma[coord] = level;
        } else {
            // Sky keeps a zero baseline, so a texel that becomes geometry next
            // frame reports its first real value as a delta rather than as
            // noise against uninitialized memory.
            prevLuma[coord] = 0.0;
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (gi == 0) {
        InterlockedAdd(denoiseCounters[counterBase + 0u], g_delta);
        InterlockedAdd(denoiseCounters[counterBase + 1u], g_level);
        InterlockedAdd(denoiseCounters[counterBase + 2u], g_pixels);
    }
}
