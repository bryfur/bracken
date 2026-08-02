// rt_reflection_denoise_temporal.comp.hlsl — temporal accumulation for the
// GGX-sampled reflection signal.
//
// One thread per TRACED (quarter-res by default) texel. The trace sends one
// importance-sampled ray per pixel into the specular lobe instead of one mirror
// ray, which is the correct integrand and is far too noisy to show directly.
// This is what makes it viewable.
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
// ---------------------------------------------------------------------------
// VIRTUAL-HIT-POINT REPROJECTION (R3, docs/rt_lighting_plan.md §5.4)
// ---------------------------------------------------------------------------
//
// The clamp above fixes what history is ALLOWED to be. This fixes where it is
// FETCHED FROM, which is the older and larger error.
//
// A reflection is an image of a point that is not on the surface. On a mirror
// it is the "virtual" point: the eye ray reflected about the normal and
// extended by the hit distance, sitting as far behind the surface as the
// reflected object is in front of it. Under camera motion that point sweeps the
// screen at roughly TWICE the parallax rate of the surface carrying it — so
// reprojecting it through the surface's own motion vector fetches the wrong
// texel, by a margin that grows with hit distance. That is the smear that
// forced `max_accum_frames` down to 16, and no amount of clamping fixes it: the
// clamp can only throw the bad history away, which costs the accumulation.
//
// So this pass computes a SECOND motion vector — the virtual point projected
// through the previous frame's camera — and blends between the two by
// roughness. Both are expressed as UV DELTAS in the velocity buffer's own
// convention (current minus previous, unjittered), which is why the jitter
// never has to be removed from anything: it cancels inside each difference.
//
// The blend, not a switch: `rtVirtualReprojectionWeight` (CPU source of truth
// and the reasoning for the band: render/rt_reflection_math.hpp). Mirror-ish
// pixels take the virtual vector; by the time the lobe is wide enough that the
// reflection is effectively painted on the surface, the hit distance is a
// one-sample estimate of a wide lobe and the surface vector is both correct and
// quieter.
//
// A STATIC CAMERA MAKES THIS EXACTLY NEUTRAL: prev == current, so both vectors
// are identically zero and every texel reads its own history, bit for bit.
//
// Output (RGBA16F): rgb = accumulated radiance, a = accumulated sample count.
// The count sits in alpha deliberately: rt_reflection_resolve.frag tests
// `traced.a <= 0.0` purely as a validity check and never uses it as a weight,
// so "0 samples" and "invalid" are the same statement and the resolve needed
// no change.

#include "rt_sampling.hlsli"

// For `octDecode` alone — the G-buffer normal is octahedrally packed and
// rt_trace.hlsli's type phase is where that decode lives for the RT passes.
// Phase 1 declares no resources, so it costs nothing here.
#define RT_TRACE_TYPES
#include "rt_trace.hlsli"

[[vk::binding(0, 2)]]
cbuffer RtReflectionTemporalUniforms : register(b0, space2) {
    /// Reconstructs the RECEIVER's world position from the depth buffer, so it
    /// must be the matrix the DEPTH was rasterized with — jitter and all — and
    /// is the same one the trace uses.
    float4x4 invViewProj;
    /// UNJITTERED, both. Their difference for one world point is a motion
    /// vector in exactly the velocity buffer's units (velocity_math.hpp).
    float4x4 viewProjUnjittered;
    float4x4 prevViewProjUnjittered;
    float3 cameraPosition;
    /// What the R16F hitT target is normalized by. Multiplying it back is the
    /// only way to recover world units here.
    float rayMaxDistance;
    uint2 tracedSize;
    /// Upper bound on accumulated frames. Lower than the shadow denoiser's:
    /// reflected content moves with the camera at twice the parallax rate, so
    /// a long tail smears more than it cleans.
    float maxAccumFrames;
    /// How far the clamp box is allowed to expand beyond the neighbourhood's
    /// standard deviation. Larger keeps more history (cleaner, more ghosting).
    float clampSigma;
    uint historyValid;
    /// 1 when the hit-distance, depth and G-buffer textures are all real. 0
    /// binds valid-but-meaningless stand-ins and takes the pure surface-motion
    /// path — exactly the pre-R3 behaviour.
    uint virtualValid;
    uint _pad0;
    uint _pad1;
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

// R16F, this pass's own resolution: hit.t / rayMaxDistance, 1.0 on a miss.
[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
Texture2D<float> hitDistanceTexture : register(t3, space0);
[[vk::combinedImageSampler]][[vk::binding(3, 0)]]
SamplerState hitDistanceSampler : register(s3, space0);

// The receiver, full-res: reversed-Z depth for its world position, and the
// prepass G-buffer for the normal that builds the mirror ray and the roughness
// that decides how much to trust the virtual point.
[[vk::combinedImageSampler]][[vk::binding(4, 0)]]
Texture2D<float> sceneDepth : register(t4, space0);
[[vk::combinedImageSampler]][[vk::binding(4, 0)]]
SamplerState sceneDepthSampler : register(s4, space0);
[[vk::combinedImageSampler]][[vk::binding(5, 0)]]
Texture2D<float4> sceneNormal : register(t5, space0);
[[vk::combinedImageSampler]][[vk::binding(5, 0)]]
SamplerState sceneNormalSampler : register(s5, space0);

[[vk::binding(0, 1)]] RWTexture2D<float4> historyOut : register(u0, space1);

// ---------------------------------------------------------------------------
// The roughness -> reprojection blend.
//
// TRANSLITERATION of `virtual_reprojection_weight` in
// render/rt_reflection_math.hpp, which is the source of truth, carries the full
// reasoning for both edges, and is tested with no GPU. Change one, change both.
// ---------------------------------------------------------------------------
#define RT_VIRTUAL_MIRROR_ROUGHNESS 0.05
#define RT_VIRTUAL_SURFACE_ROUGHNESS 0.40

/// 1 = reproject the virtual hit point, 0 = reproject the surface.
float rtVirtualReprojectionWeight(float roughness) {
    return 1.0 - smoothstep(RT_VIRTUAL_MIRROR_ROUGHNESS, RT_VIRTUAL_SURFACE_ROUGHNESS, roughness);
}

/// World point -> UV, in the render-target convention the velocity buffer uses
/// (origin top-left). Returns false behind the camera, where the divide is
/// meaningless — the same guard prepass_velocity.frag applies.
bool rtProjectToUv(float4x4 viewProj, float3 world, out float2 uv) {
    uv = float2(0.0, 0.0);
    float4 clip = mul(viewProj, float4(world, 1.0));
    if (clip.w <= 0.0) {
        return false;
    }
    float2 ndc = clip.xy / clip.w;
    uv = ndc * float2(0.5, -0.5) + 0.5;
    return true;
}

/**
 * The motion vector for this texel's reflection, in the velocity buffer's units
 * (current UV minus previous UV).
 *
 * Falls back to the surface's own velocity whenever the virtual point cannot be
 * formed or cannot be projected: sky pixels (reversed-Z depth 0), a hitT of 1
 * (a miss, whose virtual point is at infinity and whose reflection is the sky —
 * which does not move with parallax at all), and anything behind either
 * camera.
 */
float2 rtReflectionMotion(float2 uv, float2 surfaceVelocity) {
    if (virtualValid == 0u) {
        return surfaceVelocity;
    }
    float hitT = hitDistanceTexture.SampleLevel(hitDistanceSampler, uv, 0);
    if (hitT >= 1.0) {
        return surfaceVelocity; // the reflection is sky; it has no parallax
    }

    float depth = sceneDepth.SampleLevel(sceneDepthSampler, uv, 0);
    if (depth <= 0.0) {
        return surfaceVelocity; // reversed-Z far plane
    }
    float4 gbuf = sceneNormal.SampleLevel(sceneNormalSampler, uv, 0);
    float roughness = gbuf.z;
    float virtualWeight = rtVirtualReprojectionWeight(roughness);
    if (virtualWeight <= 0.0) {
        return surfaceVelocity; // rough enough that the surface vector is the answer
    }

    float2 ndc = uv * float2(2.0, -2.0) + float2(-1.0, 1.0);
    float4 world = mul(invViewProj, float4(ndc, depth, 1.0));
    if (abs(world.w) < 1e-6) {
        return surfaceVelocity;
    }
    float3 surfacePos = world.xyz / world.w;

    // The MIRROR direction, not the traced GGX sample: the virtual point is
    // defined off the lobe's axis, and the axis is the only part of it that is
    // the same from one frame to the next. Reconstructing the stochastic sample
    // here would make the reprojection as noisy as the signal it is cleaning.
    float3 N = octDecode(gbuf.xy);
    float3 viewDir = normalize(surfacePos - cameraPosition);
    float3 virtualPos = surfacePos + reflect(viewDir, N) * (hitT * rayMaxDistance);

    float2 currUv;
    float2 prevUv;
    if (!rtProjectToUv(viewProjUnjittered, virtualPos, currUv) ||
        !rtProjectToUv(prevViewProjUnjittered, virtualPos, prevUv)) {
        return surfaceVelocity;
    }
    return lerp(surfaceVelocity, currUv - prevUv, virtualWeight);
}

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
        // velocity = curr - prev (velocity_math.hpp), and so is the virtual
        // motion this blends it with.
        const float2 motion = rtReflectionMotion(uv, velocity);
        const float2 prevUv = uv - motion;

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
