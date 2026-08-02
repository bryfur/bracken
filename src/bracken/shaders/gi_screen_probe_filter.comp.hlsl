// gi_screen_probe_filter.comp.hlsl - probe-space TEMPORAL filtering: firefly
// clamp, then an EMA against the previous frame's probe whose anchor this one
// reprojects onto.
//
// Phase R4b of docs/rt_lighting_plan.md §5.5. One threadgroup per probe, one
// thread per octahedral texel.
//
// FILTERING IN PROBE SPACE IS THE WHOLE ECONOMIC ARGUMENT of the screen-probe
// tier (§3's GI-1.0 row): a pixel-space GI denoiser costs 2.5-3.3 ms at 1080p
// because it runs over 2 M pixels; this runs over 5700 probes x 64 texels =
// 365 k texels, and the per-pixel reconstruction downstream is four SH taps
// with no history of its own. There is deliberately NO per-pixel GI denoiser
// anywhere in this plan.
//
// THE SPATIAL HALF IS NOT HERE. A 3x3 probe filter needs its neighbours'
// TEMPORALLY filtered values, and SDL3 GPU has no intra-pass synchronization —
// reading them in this dispatch would read whatever half of the atlas had been
// written. It is folded into the projection pass instead, which runs later and
// therefore sees a complete atlas, and which was already visiting every texel
// of every probe to project it. A separate spatial pass would have re-read the
// same atlas to write a third one.

#include "gi_screen_probe.hlsli"

// ---------------------------------------------------------------------------
// Uniforms (set 2, b0) — mirrors gpu::GiScreenProbeFilterUniforms.
// ---------------------------------------------------------------------------
[[vk::binding(0, 2)]]
cbuffer GiScreenProbeFilterUniforms : register(b0, space2) {
    int2 probeGrid;
    int2 screenSize;
    float alpha;
    float historyValid;
    float fireflyMax;
    uint frameIndex;
};

// --- sampled (3) ---
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float4> rawAtlas : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState rawAtlasSampler : register(s0, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<float4> historyAtlas : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState historyAtlasSampler : register(s1, space0);
// UV-space motion vectors, current minus previous (contracts/velocity_math.hpp).
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
Texture2D<float2> velocity : register(t2, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
SamplerState velocitySampler : register(s2, space0);

// --- read-only storage buffers (2) ---
[[vk::binding(3, 0)]] StructuredBuffer<GiScreenProbe> probes : register(t3, space0);
[[vk::binding(4, 0)]] StructuredBuffer<GiScreenProbe> prevProbes : register(t4, space0);

// --- read-write (set 1) ---
[[vk::binding(0, 1)]] RWTexture2D<float4> filteredAtlas : register(u0, space1);

groupshared float gsLuma[GI_SP_RAYS];

float lumaOf(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

[numthreads(GI_SP_RAYS, 1, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex) {
    const int probeCount = probeGrid.x * probeGrid.y;
    const int probeIndex = int(groupId.x);
    if (probeIndex >= probeCount) {
        return;
    }
    const int texel = int(groupIndex);
    const int2 probeCoord = giSpCoord(probeIndex, probeGrid);
    const int2 atlasTexel = giSpRadianceTexel(probeCoord, texel);

    GiScreenProbe probe = probes[probeIndex];
    float4 raw = rawAtlas.Load(int3(atlasTexel, 0));
    bool resolved = raw.a >= 0.0;

    // ---- firefly clamp, against THIS probe's own mean --------------------
    //
    // Scale-free on purpose: the day/night cycle moves scene luminance by ~10^3
    // and a fixed radiance threshold cannot serve both ends. R5's
    // exposure-aware clamp (§5.6) is where this gets the adapted EV; dividing
    // by the probe's own mean is the honest approximation until then, and the
    // reduction is one barrier the pass can afford.
    gsLuma[groupIndex] = resolved ? lumaOf(raw.rgb) : 0.0;
    GroupMemoryBarrierWithGroupSync();
    float mean = 0.0;
    {
        // Log2 tree, so the reduction is 6 steps rather than 64 in one thread.
        [unroll]
        for (uint stride = GI_SP_RAYS / 2; stride > 0; stride >>= 1) {
            if (groupIndex < stride) {
                gsLuma[groupIndex] += gsLuma[groupIndex + stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        mean = gsLuma[0] / float(GI_SP_RAYS);
    }
    if (resolved && mean > 1e-6) {
        float luma = lumaOf(raw.rgb);
        float ceiling = fireflyMax * mean;
        if (luma > ceiling) {
            raw.rgb *= ceiling / luma;
        }
    }

    // ---- temporal reprojection -------------------------------------------
    float3 history = float3(0, 0, 0);
    float historyDistance = 0.0;
    bool haveHistory = false;

    if (historyValid > 0.5 && (probe.flags & GI_SP_FLAG_VALID) != 0u) {
        // Through the ANCHOR's pixel, not the tile centre. For a probe anchored
        // on a foreground silhouette those are different objects moving at
        // different speeds, and the tile centre would reproject the background.
        int2 anchorPixel = probeCoord * GI_SP_TILE + giSpUnpackAnchor(probe.flags);
        float2 uv = (float2(anchorPixel) + 0.5) / float2(screenSize);
        float2 prevUv = uv - velocity.SampleLevel(velocitySampler, uv, 0).xy;
        if (all(prevUv >= 0.0) && all(prevUv <= 1.0)) {
            int2 prevPixel = int2(prevUv * float2(screenSize));
            int2 prevCoord = clamp(prevPixel / GI_SP_TILE, int2(0, 0), probeGrid - 1);
            int prevIndex = giSpIndex(prevCoord, probeGrid);
            GiScreenProbe old = prevProbes[prevIndex];
            if ((old.flags & GI_SP_FLAG_VALID) != 0u) {
                // Same acceptance test the spatial filter and the forward
                // gather use, so "compatible probe" means one thing in this
                // subsystem and not three.
                float w = giSpBilateral(probe.cameraDistance, old.cameraDistance, probe.normal,
                                        old.normal, GI_SP_NORMAL_POWER);
                if (w > 0.5) {
                    float4 h = historyAtlas.Load(int3(giSpRadianceTexel(prevCoord, texel), 0));
                    if (h.a >= 0.0) {
                        history = h.rgb;
                        historyDistance = h.a;
                        haveHistory = true;
                    }
                }
            }
        }
    }

    float4 result;
    if (!resolved) {
        // The BVH pass never wrote this texel (a dropped miss record). Keep
        // history rather than folding a black sample in; with no history, stay
        // unresolved so the projection pass can ignore it.
        result = haveHistory ? float4(history, historyDistance) : float4(0, 0, 0, GI_SP_UNRESOLVED);
    } else if (!haveHistory) {
        result = raw;
    } else {
        result = float4(lerp(history, raw.rgb, alpha), lerp(historyDistance, raw.a, alpha));
    }
    filteredAtlas[atlasTexel] = result;
}
