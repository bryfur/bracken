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
// THE RADIUS IS DRIVEN BY VARIANCE, not by a constant. A pixel whose temporal
// estimate has already converged (many samples, low variance) gets almost no
// blur, so contact shadows stay sharp; a pixel that just disoccluded gets the
// full kernel. That is the entire reason the temporal pass carries a second
// moment around.
//
// This shader uses wave intrinsics and therefore compiles against
// vulkan1.1 — see SHADER_WAVE_OPS in shaders/CMakeLists.txt.

#include "rt_sampling.hlsli"
#include "depth_linear.hlsli"

[[vk::binding(0, 2)]]
cbuffer RtShadowFilterUniforms : register(b0, space2) {
    uint2 outputSize;   // full resolution
    uint2 tracedSize;   // resolution the history is stored at
    uint2 tileCount;
    float depthA;
    float depthB;
    /// Filter radius in TRACED texels at maximum variance. Beyond ~3 the
    /// penumbra starts bleeding through thin geometry.
    float maxRadius;
    /// How hard the depth term rejects a neighbouring texel. Larger = sharper
    /// edges, more residual noise along them.
    float depthSigma;
    uint2 _pad0;
};

[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float4> historyIn : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState historySampler : register(s0, space0);

[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<float> sceneDepth : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState sceneDepthSampler : register(s1, space0);

[[vk::binding(2, 0)]] Texture2D<uint> tileClass : register(t2, space0);

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

    // Variance of the accumulated estimate. Clamped at zero because the two
    // moments are filtered independently in the temporal pass and can disagree
    // by a hair in the last bit of a half float.
    const float variance = max(center.g - center.r * center.r, 0.0);

    // Radius from variance AND from how little history there is. A pixel with
    // one sample is filtered hard whatever its moments say, because a single
    // Bernoulli sample's moments carry no information about the distribution.
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

            // Trust neighbours that have accumulated more history.
            const float wCount = saturate(h.b / 4.0) * 0.75 + 0.25;

            const float w = wSpatial * wDepth * wCount;
            sum += h.r * w;
            weightSum += w;
        }
    }

    shadowMask[coord] = weightSum > 0.0 ? saturate(sum / weightSum) : center.r;
}
