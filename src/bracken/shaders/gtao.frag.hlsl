/**
 * GTAO Fragment Shader (horizon-based ground-truth ambient occlusion,
 * Jimenez et al. 2016, XeGTAO-style formulation).
 *
 * Runs at the AO working resolution (half or full res per AOMode) from the
 * depth-prepass depth alone, BEFORE the forward pass — the forward shaders
 * multiply the final AO texture into their AMBIENT term only.
 *
 * Per pixel: reconstruct the view-space position from depth and an
 * edge-aware normal from 4 extra depth taps, then march SLICE_COUNT
 * screen-space slices; per slice track the two horizon angles over
 * STEPS_PER_SLICE samples each side (distance falloff + thin-occluder
 * decay) and integrate cosine-weighted visibility analytically over the
 * arc. Slice rotation / step offset are jittered per pixel (IGN) and per
 * frame (golden-ratio sequence) — the temporal pass integrates the rotating
 * pattern over time.
 *
 * Outputs a single-channel AO value (1.0 = no occlusion, 0.0 = fully
 * occluded).
 */

struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer GTAOUniforms : register(b0, space3) {
    float4x4 projection;
    float4x4 invProjection;
    float2 screenSize;
    float2 invScreenSize;
    float radius;      // world/view-unit sample radius
    float radiusScale; // GraphicsSettings::ao_radius multiplier
    float falloffRange; // fraction of radius over which occlusion fades out
    float thinOccluderCompensation; // horizon decay for samples below the horizon
    float temporalRotation; // per-frame slice-angle offset in [0,1)
    float temporalOffset;   // per-frame step offset in [0,1)
    float maxDistance; // view-space far cutoff for AO (fades over the last 10%)
    float _padding0;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D<float> depthTexture : register(t0, space2);
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState depthSampler : register(s0, space2);

// viewPosFromDepth + interleavedGradientNoise
#include "lighting.hlsli"

#define SLICE_COUNT 3
#define STEPS_PER_SLICE 4

float3 viewPosAt(float2 uv) {
    float d = depthTexture.SampleLevel(depthSampler, uv, 0);
    return viewPosFromDepth(uv, d, invProjection);
}

// Edge-aware normal reconstruction: per axis pick the neighbor whose depth is
// closer to the center so the derivative never crosses a silhouette (plain
// ddx/ddy smears normals across depth discontinuities -> AO halos on edges).
// Chosen over a prepass normal MRT: 4 extra half-res depth taps are cheaper
// than a full-res normal write+sample, and AO only needs geometric normals.
float3 reconstructNormal(float2 uv, float3 centerPos) {
    float3 l = viewPosAt(uv - float2(invScreenSize.x, 0.0));
    float3 r = viewPosAt(uv + float2(invScreenSize.x, 0.0));
    float3 u = viewPosAt(uv - float2(0.0, invScreenSize.y));
    float3 d = viewPosAt(uv + float2(0.0, invScreenSize.y));
    // Derivatives along +screen-x and +screen-y (screen y is down).
    float3 dx = (abs(l.z - centerPos.z) < abs(r.z - centerPos.z)) ? (centerPos - l) : (r - centerPos);
    float3 dy = (abs(u.z - centerPos.z) < abs(d.z - centerPos.z)) ? (centerPos - u) : (d - centerPos);
    // Same orientation as cross(ddy, ddx) with implicit derivatives.
    return normalize(cross(dy, dx));
}

float PSMain(PSInput input) : SV_Target {
    float2 uv = input.texCoord;
    float depth = depthTexture.SampleLevel(depthSampler, uv, 0);

    // Sky never gets AO: reversed-Z cleared/skybox depth is exactly 0.0 (same
    // test as volumetric_fog.frag). The explicit maxDistance cutoff below is
    // the real far-field fade; this only skips true sky pixels.
    if (depth <= 0.0) {
        return 1.0;
    }

    float3 viewPos = viewPosFromDepth(uv, depth, invProjection);

    // Explicit far-field cutoff in linear view distance: past maxDistance the
    // AO footprint is subpixel at half resolution anyway. Fade over the last
    // 10% so the boundary doesn't pop.
    float distanceFade = saturate((maxDistance - (-viewPos.z)) / max(0.1 * maxDistance, 1e-3));
    if (distanceFade <= 0.0) {
        return 1.0;
    }

    float3 V = normalize(-viewPos);
    float3 N = reconstructNormal(uv, viewPos);
    // The reconstruction can flip on grazing silhouettes; AO needs a
    // camera-facing normal.
    if (dot(N, V) < 0.0) {
        N = -N;
    }

    float effRadius = radius * radiusScale;

    // Screen-space (UV-unit) radius of the world-space sampling sphere at
    // this depth. Clamped so the footprint is never subpixel nor a whole
    // screen march.
    float screenRadiusUV = effRadius * projection[0][0] * 0.5 / max(-viewPos.z, 0.1);
    screenRadiusUV = clamp(screenRadiusUV, 2.0 * invScreenSize.x, 0.2);

    // Spatial (IGN) + temporal (golden ratio) jitter for slice rotation and
    // step offset. Two decorrelated IGN lattices via the transposed pixel.
    float noiseSlice = frac(interleavedGradientNoise(input.position.xy) + temporalRotation);
    float noiseStep = frac(interleavedGradientNoise(input.position.yx + 17.317) + temporalOffset);

    // Linear falloff over the outer falloffRange fraction of the radius:
    // weight = saturate(dist * falloffMul + falloffAdd) is 1 inside, 0 at the
    // radius edge.
    float falloffDist = max(effRadius * falloffRange, 1e-4);
    float falloffMul = -1.0 / falloffDist;
    float falloffAdd = effRadius / falloffDist;

    float visibility = 0.0;

    [unroll]
    for (int slice = 0; slice < SLICE_COUNT; ++slice) {
        // Slices span pi (each covers both directions).
        float phi = PI_ * frac((float(slice) + noiseSlice) / float(SLICE_COUNT));
        float cosPhi = cos(phi);
        float sinPhi = sin(phi);

        // Screen y is down, view y is up.
        float2 omegaUV = float2(cosPhi, -sinPhi) * screenRadiusUV;
        float3 directionVec = float3(cosPhi, sinPhi, 0.0);

        // Project the normal into the slice plane (plane containing V and
        // the slice direction) — GTAO integrates visibility around it.
        float3 orthoDirectionVec = directionVec - dot(directionVec, V) * V;
        float3 axisVec = normalize(cross(orthoDirectionVec, V));
        float3 projectedNormalVec = N - axisVec * dot(N, axisVec);
        float projectedNormalVecLength = length(projectedNormalVec);
        float signNorm = sign(dot(orthoDirectionVec, projectedNormalVec));
        float cosNorm = saturate(dot(projectedNormalVec, V) / max(projectedNormalVecLength, 1e-5));
        float n = signNorm * acos(cosNorm);

        // Horizon cosines start at the "no occlusion" hemisphere boundary
        // around the projected normal. Side 0 marches -omega (final angle
        // h0 = -acos, hemisphere edge at n - pi/2); side 1 marches +omega
        // (h1 = +acos, edge at n + pi/2).
        float lowHorizonCos0 = cos(n - PI_ * 0.5);
        float lowHorizonCos1 = cos(n + PI_ * 0.5);
        float horizonCos0 = lowHorizonCos0;
        float horizonCos1 = lowHorizonCos1;

        [unroll]
        for (int step = 0; step < STEPS_PER_SLICE; ++step) {
            float s = (float(step) + noiseStep) / float(STEPS_PER_SLICE);
            s *= s; // denser sampling near the shading point
            float2 offset = s * omegaUV;

            // Side 0: negative slice direction.
            {
                float3 sPos = viewPosAt(uv - offset);
                float3 delta = sPos - viewPos;
                float dist = length(delta);
                float sampleCos = dot(delta, V) / max(dist, 1e-5);
                float weight = saturate(dist * falloffMul + falloffAdd);
                float shc = lerp(lowHorizonCos0, sampleCos, weight);
                horizonCos0 = (shc > horizonCos0) ? shc : lerp(horizonCos0, shc, thinOccluderCompensation);
            }
            // Side 1: positive slice direction.
            {
                float3 sPos = viewPosAt(uv + offset);
                float3 delta = sPos - viewPos;
                float dist = length(delta);
                float sampleCos = dot(delta, V) / max(dist, 1e-5);
                float weight = saturate(dist * falloffMul + falloffAdd);
                float shc = lerp(lowHorizonCos1, sampleCos, weight);
                horizonCos1 = (shc > horizonCos1) ? shc : lerp(horizonCos1, shc, thinOccluderCompensation);
            }
        }

        // Cosine-weighted arc integration (Jimenez et al. eq. 9 / XeGTAO).
        // CPU reference: render/gtao.hpp detail::gtao_slice_visibility.
        float h0 = -acos(clamp(horizonCos0, -1.0, 1.0));
        float h1 = acos(clamp(horizonCos1, -1.0, 1.0));
        h0 = n + clamp(h0 - n, -PI_ * 0.5, PI_ * 0.5);
        h1 = n + clamp(h1 - n, -PI_ * 0.5, PI_ * 0.5);
        float sinN = sin(n);
        float iarc0 = (cosNorm + 2.0 * h0 * sinN - cos(2.0 * h0 - n)) * 0.25;
        float iarc1 = (cosNorm + 2.0 * h1 * sinN - cos(2.0 * h1 - n)) * 0.25;
        visibility += projectedNormalVecLength * (iarc0 + iarc1);
    }

    visibility = saturate(visibility / float(SLICE_COUNT));
    return lerp(1.0, visibility, distanceFade);
}
