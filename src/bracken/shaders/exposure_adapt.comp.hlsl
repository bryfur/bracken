// Auto-exposure pass 2: histogram reduce -> target EV -> temporal adaptation.
//
// One 256-thread workgroup, dispatched (1,1,1) right after
// exposure_histogram.comp each frame. Every thread loads (and zeroes — the
// reset for next frame's accumulation) one bin into LDS; thread 0 then runs
// the percentile-clipped reduce and the adaptation step serially, which keeps
// it deterministic and bit-identical to the CPU reference
// (render/auto_exposure.hpp — exposure_target_ev() / exposure_adapt()).
//
// Reduce: pixels below the lowPercentile and above the (1 - highPercentile)
// cumulative-count marks are ignored (bin counts straddling a mark contribute
// their inside portion); the remaining counts average the bins' log2
// luminance. target_ev = log2TargetGray - avgLog2Luma, clamped to
// [minEv, maxEv].
//
// Adaptation: current EV moves toward the target at most speedUp EV/s upward
// (scene got darker -> brighten) / speedDown EV/s downward, landing exactly
// on the target when within one step. firstFrame != 0 snaps straight to the
// target (deterministic first frame / re-enable).
//
// State buffer (float4, persistent across frames, read by composite.frag as
// a read-only SSBO): x = current EV, y = target EV, z = avg log2 luminance,
// w = 1 (valid marker for debug readback).

[[vk::binding(0, 2)]]
cbuffer ExposureAdaptUniforms : register(b0, space2) {
    float minLog2Luma;    // log2 luminance of bin 1 (matches the histogram pass)
    float log2Range;      // maxLog2Luma - minLog2Luma
    float deltaTime;      // frame dt in seconds
    float firstFrame;     // != 0 -> snap current EV to the target
    float speedUp;        // EV/s toward brighter (positive EV direction)
    float speedDown;      // EV/s toward darker
    float minEv;          // target/current EV clamp
    float maxEv;
    float lowPercentile;  // fraction of pixels ignored from the dark end
    float highPercentile; // fraction of pixels ignored from the bright end
    float log2TargetGray; // scene log2 luminance the exposure converges to
    float _pad0;
};

[[vk::binding(0, 1)]]
RWStructuredBuffer<uint> histogram : register(u0, space1);
[[vk::binding(1, 1)]]
RWStructuredBuffer<float4> exposureState : register(u1, space1);

groupshared uint ldsBins[256];

// CPU mirror: render/auto_exposure.hpp exposure_bin_log2_luma().
float binLog2Luma(uint bin) {
    return minLog2Luma + (float(bin) - 1.0) / 254.0 * log2Range;
}

[numthreads(256, 1, 1)]
void CSMain(uint groupIndex : SV_GroupIndex) {
    ldsBins[groupIndex] = histogram[groupIndex];
    histogram[groupIndex] = 0u; // reset for next frame's accumulation
    GroupMemoryBarrierWithGroupSync();

    if (groupIndex != 0u) {
        return;
    }

    // ---- Percentile-clipped weighted mean (CPU mirror: exposure_target_ev)
    float total = 0.0;
    for (uint i = 0u; i < 256u; ++i) {
        total += float(ldsBins[i]);
    }

    float avgLog2Luma = log2TargetGray; // no-signal fallback -> target EV 0
    if (total > 0.0) {
        float clipLo = total * lowPercentile;
        float clipHi = total * (1.0 - highPercentile);
        float cumulative = 0.0;
        float weightSum = 0.0;
        float weightedLog2 = 0.0;
        for (uint bin = 0u; bin < 256u; ++bin) {
            float count = float(ldsBins[bin]);
            float inside = max(0.0, min(cumulative + count, clipHi) - max(cumulative, clipLo));
            cumulative += count;
            if (bin == 0u) {
                continue; // "black" bin: eats percentile budget, adds no luma
            }
            weightSum += inside;
            weightedLog2 += inside * binLog2Luma(bin);
        }
        if (weightSum > 0.0) {
            avgLog2Luma = weightedLog2 / weightSum;
        }
    }

    float targetEv = clamp(log2TargetGray - avgLog2Luma, minEv, maxEv);

    // ---- Temporal adaptation (CPU mirror: exposure_adapt)
    float currentEv;
    if (firstFrame != 0.0) {
        currentEv = targetEv;
    } else {
        currentEv = exposureState[0].x;
        float delta = targetEv - currentEv;
        currentEv += clamp(delta, -speedDown * deltaTime, speedUp * deltaTime);
        currentEv = clamp(currentEv, minEv, maxEv); // settings may have narrowed
    }

    exposureState[0] = float4(currentEv, targetEv, avgLog2Luma, 1.0);
}
