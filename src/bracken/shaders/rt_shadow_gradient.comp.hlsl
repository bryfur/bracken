// rt_shadow_gradient.comp.hlsl — the A-SVGF temporal gradient for the SUN
// visibility signal. Phase R5b of docs/rt_lighting_plan.md §5.6.
//
// WHAT THIS ANSWERS. A temporal accumulator invalidates history on MOTION,
// because motion is the only thing a velocity buffer can report. It is blind to
// the other reason history goes stale: the LIGHTING changed while nothing
// moved. A cloud crosses the sun, a time-of-day scrub runs, a lightning flash
// fires — the geometry is identical, the reprojection is perfect, and a
// 32-frame mean of the OLD lighting is dutifully carried forward.
//
// Schied et al. 2018 ("Gradient Estimation for Real-Time Adaptive Temporal
// Filtering", A-SVGF) answer it by RE-RUNNING the estimator: take a sparse set
// of pixels, re-shade them with the sample the PREVIOUS frame used, and compare
// against what the previous frame actually got. The random sequence is held
// fixed, so sampling noise cancels in the difference and what is left is the
// change in the integrand itself. That difference is the gradient; where it is
// large, history is wrong and must be dropped.
//
// WHY THIS ENGINE CAN DO IT AT ALL. Replay needs a sampler that answers "the
// sample pixel P took on frame N" for an arbitrary N. `rtSample2D`
// (rt_sampling.hlsli) is exactly that — a per-pixel hash offset advanced by the
// R2 sequence at `frameIndex` — so the previous frame's sample is one call with
// a decremented counter, and no per-pixel seed has to be stored anywhere.
// §2 of the plan lists that replayability as a standing requirement on every
// stochastic pass here for this reason.
//
// ONE RAY PER 3x3 STRATUM. That is the 1-in-9 budget: this dispatch is one
// thread per stratum of the TRACED resolution, and each thread casts at most
// one ray. At half-res traced resolution on a 1600x900 frame that is 80000
// gradient rays against 720000 shadow rays — 11% — and the reconstruction pass
// spreads each verdict over its neighbourhood so the result is dense.
//
// The chosen pixel within the stratum is `rtGradientPixelInStratum`, a hash of
// the stratum coordinate and the frame counter. Deterministic from a counter,
// like every other stochastic decision in this engine, and rotating so a
// stratum does not keep re-testing the same corner forever.
//
// Output: R16F at stratum resolution, 0 = nothing changed, 1 = this surface's
// sun visibility flipped with motion factored out. Nothing consumes it
// directly; rt_gradient_reconstruct.comp.hlsl dilates it and the temporal pass
// reads that.

#include "rt_sampling.hlsli"
#include "rt_denoise.hlsli"

#define RT_TRACE_TYPES
#include "rt_trace.hlsli"

[[vk::binding(0, 2)]]
cbuffer RtShadowGradientUniforms : register(b0, space2) {
    float4x4 invViewProj;   // clip -> world, for depth reconstruction
    float3 sunDirection;    // direction light TRAVELS (engine convention)
    float rayMaxDistance;
    uint2 tracedSize;       // resolution the shadow rays were traced at
    uint2 stratumCount;     // ceil(tracedSize / RT_GRADIENT_STRATUM)
    float normalBias;
    float depthBias;
    float sunCosThetaMax;
    /// THIS frame's counter. The replay ray uses frameIndex - 1, which is the
    /// value the previous frame's trace ran with.
    uint frameIndex;
    uint tlasNodeCount;     // 0 = no acceleration structure this frame
    uint hasGBuffer;
    /// 0 whenever the previous frame's raw bits or history describe a different
    /// image (first frame, resize, resolution change, a gap with the signal
    /// off). The whole pass then writes zeros: no comparison is possible, and
    /// claiming a gradient of 0 is correct — the temporal pass is about to
    /// discard its history for the same reason anyway.
    uint historyValid;
    float depthA;           // projection[2][2], for linearizing reversed-Z
    float depthB;           // projection[3][2]
    float depthRejectRelative;
    /// §5.8 — the capsule set the trace used. The replay MUST see the same one
    /// or every capsule-shadowed stratum reports a permanent gradient.
    uint shadowCapsuleCount;
    uint _pad1;
};

// --- set 0: sampled textures, then read-only storage textures, then buffers --
// The order is not a style choice — see shaders/README.md.

[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float> sceneDepth : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState sceneDepthSampler : register(s0, space0);

[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<float4> sceneNormal : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState sceneNormalSampler : register(s1, space0);

[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
Texture2D<float2> velocityTexture : register(t2, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
SamplerState velocitySampler : register(s2, space0);

// The denoiser's accumulation history, for its ALPHA alone: the linear view
// depth the previous frame recorded at that texel. Comparing it against this
// frame's depth is what separates "the lighting changed" from "a different
// surface is here now" — without it a disocclusion would read as a full
// gradient and reset history that was about to be reset anyway, for the wrong
// reason and over a wider region.
[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
Texture2D<float4> historyIn : register(t3, space0);
[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
SamplerState historySampler : register(s3, space0);

/// The PREVIOUS frame's raw traced bits, 8x4 per uint — the thing being
/// compared against. Retained one extra frame by a ping-pong on the bits
/// target; that ping-pong is the only new memory this mechanism costs
/// (22 KB at half res).
[[vk::binding(4, 0)]] Texture2D<uint> prevShadowBits : register(t4, space0);

[[vk::binding(5, 0)]] StructuredBuffer<BvhNodeGpu> tlasNodes : register(t5, space0);
[[vk::binding(6, 0)]] StructuredBuffer<TlasInstance> tlasInstances : register(t6, space0);
[[vk::binding(7, 0)]] StructuredBuffer<BvhNodeGpu> blasNodes : register(t7, space0);
[[vk::binding(8, 0)]] StructuredBuffer<BvhTriangle> blasTriangles : register(t8, space0);
/// The frame's capsule occluders — the SAME buffer the trace bound, and this is
/// load-bearing rather than tidy. This pass RE-TRACES last frame's sample to
/// ask whether the lighting changed under it; if it could not see the occluders
/// the trace saw, every pixel a capsule shadows would report a full gradient
/// forever and the denoiser's accumulation would be permanently reset exactly
/// on the character. A fifth read-only storage buffer, which this pass has room
/// for (five of the eight SDL3 allows).
[[vk::binding(9, 0)]] StructuredBuffer<ShadowCapsule> shadowCapsules : register(t9, space0);

[[vk::binding(0, 1)]] RWTexture2D<float> gradientOut : register(u0, space1);

#define RT_TRACE_CAPSULES
#define RT_TRACE_TRAVERSAL
#include "rt_trace.hlsli"

float3 worldAt(int2 c, uint2 size) {
    float d = sceneDepth.Load(int3(c, 0));
    float2 ndc = (float2(c) + 0.5) / float2(size) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 w = mul(invViewProj, float4(ndc, d, 1.0));
    return w.xyz / w.w;
}

// Identical to rt_shadow.comp.hlsl's, deliberately: a replay that reconstructed
// the surface differently from the pass it is replaying would report a gradient
// for the reconstruction rather than for the lighting.
float3 reconstructNormal(int2 c, uint2 size) {
    float3 p = worldAt(c, size);
    float3 px = worldAt(int2(min(c.x + 1, int(size.x) - 1), c.y), size);
    float3 nx = worldAt(int2(max(c.x - 1, 0), c.y), size);
    float3 py = worldAt(int2(c.x, min(c.y + 1, int(size.y) - 1)), size);
    float3 ny = worldAt(int2(c.x, max(c.y - 1, 0)), size);

    float3 dx = length(px - p) < length(p - nx) ? (px - p) : (p - nx);
    float3 dy = length(py - p) < length(p - ny) ? (py - p) : (p - ny);
    float3 n = cross(dy, dx);
    float len = length(n);
    return len > 1e-6 ? n / len : float3(0.0, 1.0, 0.0);
}

float linearizeDepth(float d) {
    return depthB / max(d - depthA, 1e-6);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= stratumCount.x || tid.y >= stratumCount.y) {
        return;
    }
    const int2 stratum = int2(tid.xy);

    if (historyValid == 0u) {
        gradientOut[stratum] = 0.0;
        return;
    }

    // ---- 1. which pixel of the stratum carries this frame's ray -----------
    const uint2 pick = rtGradientPixelInStratum(uint2(stratum), frameIndex);
    const int2 coord = stratum * RT_GRADIENT_STRATUM + int2(pick);
    if (coord.x >= int(tracedSize.x) || coord.y >= int(tracedSize.y)) {
        // A stratum on the right/bottom edge can pick a pixel past the end.
        // Reporting 0 rather than clamping into a neighbour keeps the sample
        // set a partition of the traced image.
        gradientOut[stratum] = 0.0;
        return;
    }

    const float2 uv = (float2(coord) + 0.5) / float2(tracedSize);

    // ---- 2. reproject to the previous frame -------------------------------
    const float2 velocity = velocityTexture.SampleLevel(velocitySampler, uv, 0);
    const float2 prevUv = uv - velocity; // velocity = curr - prev (velocity_math.hpp)
    if (any(prevUv < 0.0) || any(prevUv > 1.0)) {
        gradientOut[stratum] = 0.0;
        return;
    }
    const int2 prevCoord =
        clamp(int2(prevUv * float2(tracedSize)), int2(0, 0), int2(tracedSize) - 1);

    // ---- 3. the current surface ------------------------------------------
    uint2 depthSize;
    sceneDepth.GetDimensions(depthSize.x, depthSize.y);
    const int2 depthCoord = int2((float2(coord) + 0.5) / float2(tracedSize) * float2(depthSize));
    const float depth = sceneDepth.Load(int3(depthCoord, 0));
    const float linearDepth = depth > 0.0 ? linearizeDepth(depth) : 0.0;

    // Is it the SAME surface the previous frame saw there? The history's alpha
    // is the linear view depth it was written with, and the test is the exact
    // one the temporal pass applies to decide whether to keep that history —
    // so a gradient can only fire where history would in fact have been used.
    const float4 hist = historyIn.SampleLevel(historySampler, prevUv, 0);
    const bool depthOk = linearDepth > 0.0 &&
                         abs(hist.a - linearDepth) <= depthRejectRelative * max(linearDepth, 1.0);
    if (!depthOk) {
        gradientOut[stratum] = 0.0;
        return;
    }

    // ---- 4. the previous frame's RAW verdict at that pixel ----------------
    const uint2 prevTile = uint2(prevCoord) / RT_TILE_SIZE;
    const uint2 prevInTile = uint2(prevCoord) % RT_TILE_SIZE;
    const float prevLit = rtUnpackVisibility(prevShadowBits.Load(int3(int2(prevTile), 0)), prevInTile);

    // ---- 5. replay: THIS frame's scene, the PREVIOUS frame's sample -------
    //
    // Both halves matter. The scene is this frame's, so a moved occluder or a
    // moved sun changes the answer. The sample is the previous frame's — drawn
    // for the previous PIXEL, since that is whose sequence produced `prevLit` —
    // so the stochastic jitter is identical on both sides of the comparison and
    // subtracts out. Without that, every stratum would report the sun disk's
    // own sampling variance as a lighting change.
    bool replayLit = true;
    if (depth > 0.0) {
        float3 normal;
        if (hasGBuffer != 0u) {
            normal = octDecode(sceneNormal.Load(int3(depthCoord, 0)).xy);
        } else {
            normal = reconstructNormal(depthCoord, depthSize);
        }

        const float3 toLight = -sunDirection;
        const float ndotl = dot(normal, toLight);
        if (ndotl <= 0.0) {
            replayLit = false;
        } else {
            float2 ndc = (float2(coord) + 0.5) / float2(tracedSize) * 2.0 - 1.0;
            ndc.y = -ndc.y;
            const float4 world = mul(invViewProj, float4(ndc, depth, 1.0));
            const float3 worldPos = world.xyz / world.w;

            const float slope = saturate(1.0 - ndotl);
            const float3 rayOrigin = worldPos + normal * (normalBias * (1.0 + slope * 4.0)) +
                                     toLight * depthBias;

            // frameIndex - 1 at the PREVIOUS pixel. Unsigned wraparound on
            // frame 0 is unreachable: historyValid is 0 there.
            const float2 xi = rtSample2D(uint2(prevCoord), frameIndex - 1u);
            const float3 rayDir = sampleCone(toLight, sunCosThetaMax, xi);

            if (dot(normal, rayDir) <= 0.0) {
                replayLit = false;
            } else {
                replayLit = !sceneOccluded(rayOrigin, rayDir, rayMaxDistance);
            }
        }
    }

    gradientOut[stratum] = abs((replayLit ? 1.0 : 0.0) - prevLit);
}
