// rt_reflection_denoise_temporal.comp.hlsl — temporal accumulation for the
// GGX-sampled reflection signal.
//
// One thread per TRACED (quarter-res by default) texel. The trace now sends
// one importance-sampled ray per pixel into the specular lobe instead of one
// mirror ray, which is the correct integrand and is far too noisy to show
// directly. This is what makes it viewable.
//
// NEIGHBOURHOOD CLAMPING, NOT A DEPTH TEST. The shadow denoiser rejects stale
// history by comparing linear depth, because it has a spare channel to keep
// the previous depth in. Here the signal is RGB and the fourth channel is the
// sample count, so instead the reprojected history is clamped to the colour
// range its own neighbourhood currently spans — the standard TAA constraint.
// It is strictly better for reflections anyway: what invalidates a reflection
// is the REFLECTED content changing, which a depth test on the receiving
// surface cannot see at all (a mirror can be perfectly static while everything
// in it moves).
//
// Output (RGBA16F): rgb = accumulated radiance, a = accumulated sample count.
// The count sits in alpha deliberately: rt_reflection_resolve.frag tests
// `traced.a <= 0.0` purely as a validity check and never uses it as a weight,
// so "0 samples" and "invalid" are the same statement and the resolve needed
// no change.

#include "rt_sampling.hlsli"

[[vk::binding(0, 2)]]
cbuffer RtReflectionTemporalUniforms : register(b0, space2) {
    uint2 tracedSize;
    /// Upper bound on accumulated frames. Lower than the shadow denoiser's:
    /// reflected content moves with the camera at twice the parallax rate, so
    /// a long tail smears more than it cleans.
    float maxAccumFrames;
    /// How far the clamp box is allowed to expand beyond the neighbourhood's
    /// standard deviation. Larger keeps more history (cleaner, more ghosting).
    float clampSigma;
    uint historyValid;
    uint3 _pad0;
};

[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float4> tracedIn : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState tracedSampler : register(s0, space0);

[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<float2> velocityTexture : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState velocitySampler : register(s1, space0);

[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
Texture2D<float4> historyIn : register(t2, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
SamplerState historySampler : register(s2, space0);

[[vk::binding(0, 1)]] RWTexture2D<float4> historyOut : register(u0, space1);

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= tracedSize.x || tid.y >= tracedSize.y) {
        return;
    }
    const int2 coord = int2(tid.xy);
    const float4 current = tracedIn.Load(int3(coord, 0));

    // alpha <= 0 is the trace's "this pixel is not reflective" encoding. Pass
    // it straight through so the resolve keeps skipping it.
    if (current.a <= 0.0) {
        historyOut[coord] = float4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float3 accumulated = current.rgb;
    float count = 1.0;

    if (historyValid != 0u) {
        const float2 uv = (float2(coord) + 0.5) / float2(tracedSize);
        const float2 velocity = velocityTexture.SampleLevel(velocitySampler, uv, 0);
        const float2 prevUv = uv - velocity; // velocity = curr - prev (velocity_math.hpp)

        if (all(prevUv >= 0.0) && all(prevUv <= 1.0)) {
            // Colour statistics of the 3x3 neighbourhood, used to bound how far
            // the history is allowed to be from what this frame actually saw.
            float3 m1 = 0.0;
            float3 m2 = 0.0;
            float valid = 0.0;
            [unroll]
            for (int dy = -1; dy <= 1; ++dy) {
                [unroll]
                for (int dx = -1; dx <= 1; ++dx) {
                    const int2 s = clamp(coord + int2(dx, dy), int2(0, 0), int2(tracedSize) - 1);
                    const float4 n = tracedIn.Load(int3(s, 0));
                    if (n.a <= 0.0) {
                        continue; // a non-reflective neighbour says nothing about this lobe
                    }
                    m1 += n.rgb;
                    m2 += n.rgb * n.rgb;
                    valid += 1.0;
                }
            }

            const float4 hist = historyIn.SampleLevel(historySampler, prevUv, 0);
            if (hist.a > 0.0 && valid > 0.0) {
                const float3 mean = m1 / valid;
                const float3 sigma = sqrt(max(m2 / valid - mean * mean, 0.0));
                const float3 lo = mean - sigma * clampSigma;
                const float3 hi = mean + sigma * clampSigma;
                const float3 clamped = clamp(hist.rgb, lo, hi);

                count = min(hist.a + 1.0, maxAccumFrames);
                accumulated = lerp(clamped, current.rgb, 1.0 / count);
            }
        }
    }

    historyOut[coord] = float4(accumulated, count);
}
