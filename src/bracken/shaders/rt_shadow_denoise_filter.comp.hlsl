// rt_shadow_denoise_filter.comp.hlsl — variance-guided filter + upsample,
// pass 3 of 3. Produces the full-resolution R8 mask the forward shaders read.
//
// FILTERING AND UPSAMPLING ARE THE SAME OPERATION, so this does them once
// instead of twice. One thread per FULL-RESOLUTION pixel gathers a
// neighbourhood of TRACED-resolution texels, weighting each by how much it
// looks like it belongs to the same surface. Splitting it would mean a filter
// pass writing a traced-res target and an upsample pass immediately re-reading
// it — an entire round trip through memory to accomplish nothing the combined
// weighting does not.
//
// It also removes the artifact that started this whole exercise: the forward
// shaders used to take a plain bilinear tap of a half-res binary mask, which
// crawls along every silhouette as the camera moves because bilinear weights
// have no idea a depth edge is there.
//
// THE RADIUS IS DRIVEN BY THE ESTIMATE'S STANDARD ERROR, not by a constant. A
// pixel whose temporal estimate has already converged (many samples, so a small
// standard error) gets almost no blur, so contact shadows stay sharp; a pixel
// that just disoccluded gets the full kernel. That is the entire reason the
// temporal pass carries a second moment and a sample count around — and it is
// the COUNT that does the work here, because for a one-bit signal the second
// moment alone is a constant (`rtVisibilityFilterVariance`).
//
// EVERY TAP IS EDGE-STOPPED ON THREE THINGS: distance, the neighbour's own
// recorded depth, and its prepass G-buffer NORMAL. The normal term is what stops
// a wall and the floor it meets from averaging into each other — the depth term
// cannot see that junction at all, because the two surfaces are at the same
// depth there. Same shape, same exponent, as the sibling gather in
// rt_di_denoise_spatial.comp.hlsl.
//
// This shader uses wave intrinsics and therefore compiles against
// vulkan1.1 — see SHADER_WAVE_OPS in shaders/CMakeLists.txt.

#include "rt_sampling.hlsli"
#include "rt_denoise.hlsli"
#include "depth_linear.hlsli"

#define RT_TRACE_TYPES
#include "rt_trace.hlsli" // octDecode

[[vk::binding(0, 2)]]
cbuffer RtShadowFilterUniforms : register(b0, space2) {
    uint2 outputSize;   // full resolution
    uint2 tracedSize;   // resolution the history is stored at
    uint2 tileCount;
    float depthA;
    float depthB;
    /// Filter radius in TRACED texels for an estimate with NO accumulated
    /// history. Beyond ~3 the penumbra starts bleeding through thin geometry.
    float maxRadius;
    /// How hard the depth term rejects a neighbouring texel. Larger = sharper
    /// edges, more residual noise along them.
    float depthSigma;
    /// Exponent on the normal similarity. Higher keeps creases; a plane is
    /// unaffected either way (dot == 1).
    float normalPower;
    /// 0 when no prepass G-buffer exists this frame, in which case the bound
    /// normal texture is a valid stand-in that must not be decoded.
    uint hasGBuffer;
};

// BINDING ORDER IS NOT FREE — see the same warning in
// rt_shadow_denoise_temporal.comp.hlsl. Within set 0 SDL3 GPU numbers sampled
// textures first and read-only storage textures after them, contiguously, so
// the G-buffer sampler takes slot 2 and `tileClass` moves to 3.
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float4> historyIn : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState historySampler : register(s0, space0);

[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<float> sceneDepth : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState sceneDepthSampler : register(s1, space0);

/// The prepass G-buffer: octahedral world normal in .xy. Bound with a NEAREST
/// sampler — a filtered octahedral pair across a silhouette decodes to a
/// direction neither surface has.
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
Texture2D<float4> sceneNormal : register(t2, space0);
[[vk::combinedImageSampler]][[vk::binding(2, 0)]]
SamplerState sceneNormalSampler : register(s2, space0);

[[vk::binding(3, 0)]] Texture2D<uint> tileClass : register(t3, space0);

[[vk::binding(0, 1)]] RWTexture2D<float> shadowMask : register(u0, space1);

/// See depth_linear.hlsli. Self-consistent here — `h.a` is the neighbour's own
/// depth written by the temporal pass with this same expression — so the fix
/// (the depthA sign, 2026-08-02) changes what `depthSigma` means in real units
/// rather than what is being compared: the effective sigma was
/// depthSigma * (1 - 2t/far), and every surface past far/2 shared one saturated
/// depth, which made wDepth uniformly 1 out there and the bilateral filter a
/// plain box.
float linearizeDepth(float d) {
    return brackenLinearViewDepth(d, depthA, depthB);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    const bool inBounds = tid.x < outputSize.x && tid.y < outputSize.y;
    const int2 coord = int2(tid.xy);

    // Map this full-res pixel onto the traced grid.
    const float2 uv = (float2(coord) + 0.5) / float2(max(outputSize, uint2(1, 1)));
    const float2 tracedCoordF = uv * float2(tracedSize) - 0.5;
    const int2 tracedCoord = int2(round(tracedCoordF));
    const uint2 tile = uint2(clamp(tracedCoord, int2(0, 0), int2(tracedSize) - 1)) / RT_TILE_SIZE;

    const uint verdict = inBounds ? tileClass.Load(int3(int2(tile), 0)) : 0u;


    // A uniform tile still has to be UPSAMPLED, not stamped. Writing the tile's
    // constant here is what produced 16x8 rectangles all over the frame: the
    // mixed tiles came out smoothly filtered and their uniform neighbours came
    // out flat, so every boundary between the two classes was a visible tile
    // edge. A bilinear tap of the history costs one fetch and is continuous
    // across that boundary, because the history of a uniform tile already holds
    // exactly the constant in question.
    //
    // Wave-wide early-out on top of that. When no lane in the wave landed on a
    // mixed tile — which, over a sunlit exterior, is the overwhelming majority
    // of waves — the whole gather below is skipped rather than executed and
    // thrown away lane by lane. This is what the vulkan1.1 requirement buys.
    if (WaveActiveAllTrue(verdict != 2u)) {
        if (inBounds) {
            shadowMask[coord] = saturate(historyIn.SampleLevel(historySampler, uv, 0).r);
        }
        return;
    }
    if (!inBounds) {
        return;
    }
    if (verdict != 2u) {
        shadowMask[coord] = saturate(historyIn.SampleLevel(historySampler, uv, 0).r);
        return;
    }

    const float centerRaw = sceneDepth.SampleLevel(sceneDepthSampler, uv, 0);
    if (centerRaw <= 0.0) {
        shadowMask[coord] = 1.0; // sky
        return;
    }
    const float centerDepth = linearizeDepth(centerRaw);

    const float4 center = historyIn.Load(int3(clamp(tracedCoord, int2(0, 0), int2(tracedSize) - 1), 0));

    // This thread IS a full-res pixel, so the centre normal is a straight fetch
    // at its own UV. Sampled rather than Loaded for the same reason the depth
    // above is: the bound texture's resolution is not assumed to equal
    // `outputSize`, and the sampler is NEAREST, so this is still one texel.
    const float3 nCenter =
        hasGBuffer != 0u ? octDecode(sceneNormal.SampleLevel(sceneNormalSampler, uv, 0).xy)
                         : float3(0.0, 1.0, 0.0);

    // Standard error of the ACCUMULATED estimate, not of one sample. The traced
    // sample is one BIT, so `sample*sample == sample` and the stored second
    // moment is bit-identical to the stored mean: `g - r*r` is p(1-p), the
    // variance of a SINGLE Bernoulli draw, which does not shrink as history
    // accumulates. Dividing by the count is what makes it the estimator's — see
    // rt_denoise_math.hpp's `visibility_filter_variance`, which is also where
    // the ~2x over-estimate at full convergence is argued for.
    const float variance = rtVisibilityFilterVariance(center.r, center.g, center.b);

    // Radius from that AND from how little history there is. A pixel with one
    // sample is filtered hard whatever its moments say, because a single
    // Bernoulli sample's moments carry no information about the distribution —
    // and for a binary signal they are literally zero variance, so this term is
    // the only thing covering count == 1.
    const float confidence = saturate(center.b / 8.0);
    const float noisiness = saturate(sqrt(variance) * 4.0);
    const float radius = max(1.0, maxRadius * max(noisiness, 1.0 - confidence));
    const int r = int(ceil(radius));

    float sum = 0.0;
    float weightSum = 0.0;

    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            const int2 s = clamp(tracedCoord + int2(dx, dy), int2(0, 0), int2(tracedSize) - 1);
            const float4 h = historyIn.Load(int3(s, 0));

            // Spatial falloff measured from the FRACTIONAL traced position,
            // not from the rounded texel. Using the rounded one makes every
            // full-res pixel inside the same traced texel produce an identical
            // kernel, which is a nearest-neighbour upsample wearing a filter's
            // clothing — and looks like one.
            const float2 delta = float2(s) - tracedCoordF;
            const float d2 = dot(delta, delta);
            const float wSpatial = exp(-d2 / max(2.0 * radius * radius, 1e-4));

            // Depth similarity, against the neighbour's own stored depth. A
            // neighbour on the far side of a silhouette is rejected here, which
            // is precisely what plain bilinear could not do.
            const float dz = abs(h.a - centerDepth);
            const float wDepth = exp(-dz / max(depthSigma * max(centerDepth, 1.0), 1e-4));

            // Orientation similarity, which is the edge the depth term is blind
            // to: at a wall/floor junction, an eave or a beam the two surfaces
            // meet AT THE SAME DEPTH, so `dz` is ~0 and the gather happily
            // averaged a lit facade into a shadowed soffit. The tap is mapped
            // back to full res the way rt_di_denoise_spatial's `depthCoordFor`
            // does, so the surface weighted here is the surface that was traced.
            float wNormal = 1.0;
            if (hasGBuffer != 0u) {
                const float2 nUv = (float2(s) + 0.5) / float2(max(tracedSize, uint2(1, 1)));
                const float3 nSample =
                    octDecode(sceneNormal.SampleLevel(sceneNormalSampler, nUv, 0).xy);
                wNormal = pow(saturate(dot(nCenter, nSample)), normalPower);
            }

            // Trust neighbours that have accumulated more history.
            const float wCount = saturate(h.b / 4.0) * 0.75 + 0.25;

            const float w = wSpatial * wDepth * wNormal * wCount;
            sum += h.r * w;
            weightSum += w;
        }
    }

    // The fallback is no longer theoretical: before the normal term the weights
    // were a product of three strictly positive exponentials and this branch
    // could not be taken, but `pow(0, normalPower)` is a real zero and a texel
    // whose whole neighbourhood faces away from it now falls through to its own
    // unfiltered estimate rather than to a divide by zero.
    shadowMask[coord] = weightSum > 0.0 ? saturate(sum / weightSum) : center.r;
}
