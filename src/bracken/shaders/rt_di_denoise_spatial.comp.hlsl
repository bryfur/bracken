// rt_di_denoise_spatial.comp.hlsl — the Full-DI tier's edge-aware spatial pass
// (R5c, docs/rt_lighting_plan.md §5.3; widened to two a-trous iterations and a
// third target in R7).
//
// One thread per TRACED texel. Gathers the previous stage's output over a window
// whose RADIUS IS DRIVEN BY THE ACCUMULATION COUNT — young history filters
// wide, converged history filters at the floor — weighted by depth and normal
// similarity so the blur stops at a silhouette.
//
// It is driven by the count rather than by a variance because, unlike the two
// visibility signals, this one has no second moment to read: those accumulate a
// Bernoulli sample whose variance is p(1-p) and comes free, and this one
// accumulates an unbounded radiance whose second moment would cost a third
// RGBA16F target to carry. The count is the honest proxy — see
// `di_filter_radius` in render/lighting/local_di.hpp for why the curve is
// linear in it and why the floor is not zero.
//
// ---------------------------------------------------------------------------
// TWO ITERATIONS, A-TROUS (R7c)
// ---------------------------------------------------------------------------
//
// This pipeline is dispatched TWICE — `stepSize` 1 then 2 — which is the
// standard a-trous trick: the same tap count covers twice the footprint by
// spacing the taps out, so a second pass buys a 2x wider filter for the price of
// a first one instead of the price of a 4x kernel. The alternative (one pass
// with double the radius) costs quadratically and, worse, blurs the near taps it
// already had.
//
// THE SECOND ITERATION IS SKIPPABLE, TWO WAYS, and both matter:
//
//  * PER PIXEL, here: a texel whose accumulation has reached
//    LOCAL_DI_FILTER_CONVERGED_COUNT passes THROUGH the wide iteration
//    untouched. A converged 1-spp radiance estimate is already as good as a
//    wide blur can make it, and widening it further is how a steady-state
//    image goes soft for no reason. The wide pass therefore does its work
//    exactly where the noise is — disocclusions, a reservoir that just changed
//    light, the frames after a camera cut — and nowhere else.
//  * PER FRAME, by the caller: `RtLocalShadowSettings::di_spatial_iterations`
//    of 1 dispatches this once with `isFinal`, which reproduces the R5c chain.
//
// `isFinal` is what decides the diffuse alpha's meaning on the way out: an
// intermediate iteration must pass the COUNT along (the next iteration reads it
// to size its own radius and to make the skip decision), and the last one
// replaces it with the texel's LINEAR VIEW DEPTH, which is what the forward
// reconstruction reads. Getting that backwards is a silent failure — the forward
// pass would read a sample count as a depth and reject every tap — so it is a
// uniform rather than an inference.
//
// NO JOINT-BILATERAL UPSAMPLE. This pass ends at the traced resolution and the
// forward shaders reconstruct with their own depth-aware 4-tap bilinear
// (`localDiSample`). The visibility denoiser fuses filter and upsample into one
// full-resolution pass because its output is a single R8 channel that four
// fragment shaders then sample with a hardware bilinear; here the output is
// three targets that four fragment shaders read as STORAGE textures, where
// there is no hardware filter to be had at all. Doing the upsample here would
// mean writing full-resolution RGBA16F targets — 4x the bandwidth and 4x the
// memory — to save a 4-tap gather the forward pass has to do anyway.
//
// Output layout:
//   diffuse: rgb = filtered demodulated diffuse, a = COUNT, or on the final
//            iteration the LINEAR VIEW DEPTH
//   spec:    rgb = filtered demodulated specular, a = Schlick fraction
//   dir:     rg  = filtered octahedral light direction (R7b)
//
// The direction takes the SAME weights as the radiance halves and is then
// renormalized, because it is an estimate of the same thing at the same texels
// and a second weight function would let the two disagree about where the
// silhouette is. What it does NOT take is the averaging: vectors are summed and
// re-normalized, never lerped in octahedral space. local_di.hlsli's
// `localDiDirNormalize` is the shared shape.

#include "rt_sampling.hlsli"
#include "local_di.hlsli"

#define RT_TRACE_TYPES
#include "rt_trace.hlsli" // octDecode

[[vk::binding(0, 2)]]
cbuffer RtDiSpatialUniforms : register(b0, space2) {
    uint2 tracedSize;
    float depthA;  ///< projection[2][2], for linearizing reversed-Z depth
    float depthB;  ///< projection[3][2]
    /// Filter radius in TAPS for a one-sample pixel. The curve from there down
    /// to LOCAL_DI_FILTER_MIN_RADIUS is local_di.hlsli's.
    float maxRadius;
    /// Relative depth difference at which a neighbour is worth 1/e of its
    /// spatial weight. Larger = sharper edges, more residual noise along them.
    float depthSigma;
    /// Exponent on the normal similarity. Higher keeps creases; a plane is
    /// unaffected either way (dot == 1).
    float normalPower;
    /// A-trous tap spacing in traced texels: 1 on the first iteration, 2 on the
    /// second. Any value > 1 also enables the converged-pixel pass-through.
    uint stepSize;
    /// 1 on the last iteration of the frame, which is the one that converts the
    /// diffuse alpha from the accumulation count to the linear view depth.
    uint isFinal;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

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
Texture2D<float> sceneDepth : register(t3, space0);
[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
SamplerState sceneDepthSampler : register(s3, space0);

[[vk::combinedImageSampler]][[vk::binding(4, 0)]]
Texture2D<float4> sceneNormal : register(t4, space0);
[[vk::combinedImageSampler]][[vk::binding(4, 0)]]
SamplerState sceneNormalSampler : register(s4, space0);

[[vk::image_format("rgba16f")]]
[[vk::binding(0, 1)]] RWTexture2D<float4> diffuseOut : register(u0, space1);
[[vk::image_format("rgba16f")]]
[[vk::binding(1, 1)]] RWTexture2D<float4> specularOut : register(u1, space1);
[[vk::image_format("rg16f")]]
[[vk::binding(2, 1)]] RWTexture2D<float2> directionOut : register(u2, space1);

float linearizeDepth(float d) {
    return depthB / max(d - depthA, 1e-6);
}

/// The full-resolution depth/G-buffer texel a traced texel stands for. Same
/// mapping rt_local_shadow.comp.hlsl used to produce it, so the surface this
/// pass filters against is the surface that was traced.
int2 depthCoordFor(int2 traced, uint2 depthSize) {
    return int2((float2(traced) + 0.5) / float2(tracedSize) * float2(depthSize));
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= tracedSize.x || tid.y >= tracedSize.y) {
        return;
    }
    const int2 coord = int2(tid.xy);

    uint2 depthSize;
    sceneDepth.GetDimensions(depthSize.x, depthSize.y);
    const float centerRaw = sceneDepth.Load(int3(depthCoordFor(coord, depthSize), 0));
    if (centerRaw <= 0.0) {
        // Sky. Alpha 0 is the "no answer" the forward reconstruction rejects,
        // and it means the same thing to the next iteration.
        diffuseOut[coord] = float4(0.0, 0.0, 0.0, 0.0);
        specularOut[coord] = float4(0.0, 0.0, 0.0, 0.0);
        directionOut[coord] = localDiDirEncode(float3(0.0, 1.0, 0.0));
        return;
    }
    const float centerDepth = linearizeDepth(centerRaw);
    const float3 centerNormal = octDecode(sceneNormal.Load(int3(depthCoordFor(coord, depthSize), 0)).xy);

    const float4 center = diffuseIn.Load(int3(coord, 0));
    // `center.a` is the accumulation count on every iteration but the one that
    // consumed the final output, which is why `isFinal` writes the depth LAST.
    const float count = center.a;
    const float outAlpha = isFinal != 0u ? centerDepth : count;

    // The converged pass-through. Only on a widened iteration: the first one is
    // never skipped, because LOCAL_DI_FILTER_MIN_RADIUS exists precisely to keep
    // a 1-spp estimate from shimmering once it has stopped moving.
    if (stepSize > 1u && count >= LOCAL_DI_FILTER_CONVERGED_COUNT) {
        diffuseOut[coord] = float4(center.rgb, outAlpha);
        specularOut[coord] = specularIn.Load(int3(coord, 0));
        directionOut[coord] = directionIn.Load(int3(coord, 0)).xy;
        return;
    }

    const float radius = localDiFilterRadius(count, maxRadius);
    const int r = int(ceil(radius));
    const int step = int(max(stepSize, 1u));

    float3 diffuseSum = 0.0;
    float4 specSum = 0.0;
    float3 dirSum = 0.0;
    float weightSum = 0.0;

    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            const int2 s = clamp(coord + int2(dx, dy) * step, int2(0, 0), int2(tracedSize) - 1);
            const float4 d = diffuseIn.Load(int3(s, 0));
            if (d.a <= 0.0) {
                continue; // the previous stage wrote nothing here
            }
            const int2 dc = depthCoordFor(s, depthSize);
            const float nRaw = sceneDepth.Load(int3(dc, 0));
            if (nRaw <= 0.0) {
                continue;
            }
            const float nDepth = linearizeDepth(nRaw);
            const float3 nNormal = octDecode(sceneNormal.Load(int3(dc, 0)).xy);

            // Distance in TAPS, not texels: the a-trous kernel keeps its shape
            // and the stride is what widens the footprint.
            const float d2 = float(dx * dx + dy * dy);
            const float wSpatial = exp(-d2 / max(2.0 * radius * radius, 1e-4));
            const float wDepth =
                exp(-abs(nDepth - centerDepth) / max(depthSigma * max(centerDepth, 1.0), 1e-4));
            const float wNormal = pow(saturate(dot(centerNormal, nNormal)), normalPower);

            const float w = wSpatial * wDepth * wNormal;
            if (w <= 0.0) {
                continue;
            }
            diffuseSum += d.rgb * w;
            specSum += specularIn.Load(int3(s, 0)) * w;
            dirSum += localDiDirDecode(directionIn.Load(int3(s, 0)).xy) * w;
            weightSum += w;
        }
    }

    if (weightSum <= 1e-6) {
        // Every neighbour rejected, including this texel's own contribution
        // (which can only happen when its normal is degenerate). Pass the
        // unfiltered estimate through rather than writing black.
        diffuseOut[coord] = float4(center.rgb, outAlpha);
        specularOut[coord] = specularIn.Load(int3(coord, 0));
        directionOut[coord] = directionIn.Load(int3(coord, 0)).xy;
        return;
    }

    diffuseOut[coord] = float4(diffuseSum / weightSum, outAlpha);
    specularOut[coord] = float4(specSum.rgb / weightSum, specSum.a / weightSum);
    // Renormalized, with this texel's own direction as the fallback when the
    // taps cancelled — the one input that is certainly a real direction.
    directionOut[coord] = localDiDirEncode(
        localDiDirNormalize(dirSum, localDiDirDecode(directionIn.Load(int3(coord, 0)).xy)));
}
