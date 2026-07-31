// Auto-exposure pass 1: 256-bin log2-luminance histogram of the HDR scene.
//
// Runs over the TAA-resolved (pre-bloom) linear-HDR scene color. Each 16x16
// workgroup accumulates its tile into an LDS histogram, then merges the
// non-zero bins into the global histogram buffer with one atomic add per bin
// (most of a tile's 256 bins are empty, so the global traffic is tiny).
//
// Bin mapping (CPU mirror + tests: render/auto_exposure.hpp):
//   bin 0     = luma below exp2(minLog2Luma) ("black" — counted toward the
//               percentile total but contributes no luminance)
//   bin 1-255 = log2(luma) linearly mapped over [minLog2Luma, maxLog2Luma]
// The default range is EV -10..+16 (ExposureHistogramUniforms).
//
// The histogram buffer is NOT cleared here: exposure_adapt.comp zeroes every
// bin after reading it, so the buffer arrives empty each frame (and is
// created zero-filled).

#include "lighting.hlsli"

[[vk::binding(0, 2)]]
cbuffer ExposureHistogramUniforms : register(b0, space2) {
    uint width;         // scene color extent in pixels
    uint height;
    float minLog2Luma;  // log2 luminance mapped to bin 1
    float invLog2Range; // 1 / (maxLog2Luma - minLog2Luma)
};

[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D sceneColor : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState sceneSampler : register(s0, space0);

[[vk::binding(0, 1)]]
RWStructuredBuffer<uint> histogram : register(u0, space1);

groupshared uint ldsBins[256];

// CPU mirror: render/auto_exposure.hpp exposure_bin_from_luma().
uint binFromLuma(float luma) {
    if (luma < exp2(minLog2Luma)) {
        return 0;
    }
    float t = saturate((log2(luma) - minLog2Luma) * invLog2Range);
    return 1u + (uint)(t * 254.0);
}

[numthreads(16, 16, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex) {
    ldsBins[groupIndex] = 0u;
    GroupMemoryBarrierWithGroupSync();

    if (dtid.x < width && dtid.y < height) {
        float2 uv = (float2(dtid.xy) + 0.5) / float2(width, height);
        float3 color = sceneColor.SampleLevel(sceneSampler, uv, 0).rgb;
        uint bin = binFromLuma(luminance(color));
        InterlockedAdd(ldsBins[bin], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    uint count = ldsBins[groupIndex];
    if (count > 0u) {
        InterlockedAdd(histogram[groupIndex], count);
    }
}
