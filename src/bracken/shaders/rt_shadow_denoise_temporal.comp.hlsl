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

#include "rt_sampling.hlsli"

[[vk::binding(0, 2)]]
cbuffer RtShadowTemporalUniforms : register(b0, space2) {
    uint2 tracedSize;     // resolution the rays were traced at
    uint2 tileCount;      // packed-bit texture dimensions
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
    uint historyValid;    // 0 on the first frame and after any resize
    uint3 _pad0;
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

[[vk::binding(3, 0)]] Texture2D<uint> shadowBits : register(t3, space0);
[[vk::binding(4, 0)]] Texture2D<uint> tileClass : register(t4, space0);

[[vk::binding(0, 1)]] RWTexture2D<float4> historyOut : register(u0, space1);

/// Reversed-Z depth -> positive linear view depth. Same algebra the GTAO pass
/// uses; depthA/depthB are the projection terms so this does not need the
/// whole matrix.
float linearizeDepth(float d) {
    return depthB / max(d - depthA, 1e-6);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= tracedSize.x || tid.y >= tracedSize.y) {
        return;
    }
    const int2 coord = int2(tid.xy);
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

    // A uniform neighbourhood carries no variance and needs no history: write
    // the converged answer directly. This also stops the accumulator from
    // dragging a stale penumbra across a region that has become fully lit.
    const uint verdict = tileClass.Load(int3(int2(tile), 0));
    if (verdict != 2u) {
        historyOut[coord] = float4(sample, sample * sample, maxAccumFrames, linearDepth);
        return;
    }

    float mean = sample;
    float moment2 = sample * sample;
    float count = 1.0;

    if (historyValid != 0u) {
        const float2 velocity = velocityTexture.SampleLevel(velocitySampler, uv, 0);
        const float2 prevUv = uv - velocity; // velocity = curr - prev (velocity_math.hpp)

        if (all(prevUv >= 0.0) && all(prevUv <= 1.0)) {
            const float4 hist = historyIn.SampleLevel(historySampler, prevUv, 0);

            // Reject history across a depth discontinuity. Without this, a
            // disocclusion drags the occluder's shadow along behind it — the
            // classic ghost trailing anything that moves.
            const float depthDelta = abs(hist.a - linearDepth);
            const bool depthOk =
                linearDepth > 0.0 && depthDelta <= depthRejectRelative * max(linearDepth, 1.0);

            if (depthOk) {
                count = min(hist.b + 1.0, maxAccumFrames);
                const float alpha = 1.0 / count;
                mean = lerp(hist.r, sample, alpha);
                moment2 = lerp(hist.g, sample * sample, alpha);
            }
        }
    }

    historyOut[coord] = float4(mean, moment2, count, linearDepth);
}
