// gi_screen_probe_march.comp.hlsl - stage A of the screen-probe gather: one
// thread per probe texel, marching the Hi-Z pyramid, sampling last frame's lit
// scene where it hits, and COMPACTING everything it could not resolve into a
// miss buffer for the indirect-dispatched BVH pass.
//
// Phase R4b of docs/rt_lighting_plan.md §5.5. One threadgroup per probe, 64
// threads = the probe's whole 8x8 octahedral payload.
//
// WHY A SCREEN MARCH AT ALL, given the engine already has a BVH: a screen hit
// costs log-n texture fetches instead of triangle tests, AND — the reason §3
// of the plan lists this as its own adopted technique — the depth buffer
// contains the SKINNED and dynamic geometry whose BLAS is a bind pose. The
// villager's own bounce light exists in this pass and cannot exist in the BVH
// one. The march is not an optimization of the world trace; it answers a
// question the world trace answers wrong.
//
// WHAT IT COSTS: the radiance it reads is the PREVIOUS frame's resolved scene
// colour (the TAA history), so a screen hit carries one frame of latency and
// everything the previous frame's shading contained, including that frame's own
// indirect. That is the standard screen-space feedback loop — infinite bounce
// for free, with the standard caveat that it also feeds back its own error. The
// probe-space temporal filter's alpha is the damping on it.

#include "gi_screen_probe.hlsli"

// ---------------------------------------------------------------------------
// Uniforms (set 2, b0) — mirrors gpu::GiScreenProbeMarchUniforms.
// ---------------------------------------------------------------------------
[[vk::binding(0, 2)]]
cbuffer GiScreenProbeMarchUniforms : register(b0, space2) {
    float4x4 viewProj;               // JITTERED — this is the depth buffer's matrix
    float4x4 prevViewProjUnjittered; // the pair the velocity buffer / TAA use
    float3 cameraPosition;
    float depthA;
    int2 probeGrid;
    int2 screenSize;
    float depthB;
    float rayMaxDistance;
    float normalBias;
    float historyValid;
    uint frameIndex;
    uint mipCount;
    uint missCapacity;
    uint _marchPad0;
};

// ---------------------------------------------------------------------------
// Bindings (set 0: sampled, then RO storage textures, then RO storage buffers).
// ---------------------------------------------------------------------------

// --- sampled (2) ---
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float> hzb : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState hzbSampler : register(s0, space0); // POINT clamp: a mip is a MAX, not an average

// Last frame's resolved linear-HDR scene colour (render::TAA's history). Read
// with a LINEAR sampler — the reprojected hit lands between texels.
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<float4> sceneHistory : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState sceneHistorySampler : register(s1, space0);

// --- read-only storage buffers (1) ---
[[vk::binding(2, 0)]] StructuredBuffer<GiScreenProbe> probes : register(t2, space0);

// --- read-write (set 1): storage textures first, then storage buffers ---
[[vk::binding(0, 1)]] RWTexture2D<float4> radianceAtlas : register(u0, space1);
[[vk::binding(1, 1)]] RWByteAddressBuffer missBuf : register(u1, space1);
[[vk::binding(2, 1)]] RWByteAddressBuffer counters : register(u2, space1);

// One InterlockedAdd on device memory per THREADGROUP, not per thread: the
// group reduces into groupshared first and thread 0 reserves the whole run.
// Same shape as exposure_histogram.comp.hlsl's LDS merge, and for the same
// reason — 364800 atomics on one address would serialize the pass.
groupshared uint gsMissCount;
groupshared uint gsMissBase;
groupshared uint gsHitCount;

int2 mipSizeAt(int level) {
    return int2(max(1, screenSize.x >> level), max(1, screenSize.y >> level));
}

/// The ray parameter t (in [0,1] of the WORLD ray) at screen parameter s.
///
/// Screen position and device depth are affine in s along a projected line —
/// that is why the march can step linearly at all — but the WORLD distance is
/// not: 1/w is what interpolates linearly in screen space. Recovering w at s
/// and inverting the affine w(t) is the exact conversion, and it is what makes
/// the handoff's `resumeDistance` a real distance rather than a plausible one.
float worldTAtScreenS(float s, float w0, float w1) {
    if (abs(w1 - w0) < 1e-6) {
        return s; // the projection is effectively affine along this ray
    }
    float invW = lerp(1.0 / w0, 1.0 / w1, s);
    float w = 1.0 / max(abs(invW), 1e-9) * sign(invW);
    return (w - w0) / (w1 - w0);
}

[numthreads(64, 1, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex) {
    if (groupIndex == 0) {
        gsMissCount = 0;
        gsHitCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    const int probeCount = probeGrid.x * probeGrid.y;
    const int probeIndex = int(groupId.x);
    const int texel = int(groupIndex);
    const int2 probeCoord = giSpCoord(probeIndex, probeGrid);
    const int2 atlasTexel = giSpRadianceTexel(probeCoord, texel);

    GiScreenProbe probe = (GiScreenProbe)0;
    bool live = probeIndex < probeCount;
    if (live) {
        probe = probes[probeIndex];
        live = (probe.flags & GI_SP_FLAG_VALID) != 0u;
    }

    float3 dir = giSpTexelDirection(texel, frameIndex);
    // Only the hemisphere above the anchor is traced. Zero below is not an
    // approximation for the anchor's own normal — see screen_probe_math.hpp's
    // `screen_probe_traces_texel`, and the test that proves the L1 convolution
    // reproduces the hemisphere integral exactly.
    bool traced = live && dot(dir, probe.normal) > 0.0;

    bool isHit = false;
    bool isMiss = false;
    float3 radiance = float3(0, 0, 0);
    float hitDistance = 0.0;
    float resumeDistance = 0.0;

    if (traced) {
        const float3 origin = probe.worldPosition + probe.normal * normalBias;

        // Clip the world ray so its far end stays in FRONT of the camera. w is
        // affine in t, so the crossing is exact and costs one dot product —
        // versus projecting a point behind the eye, which folds the ray back
        // across the screen and makes the march walk a line that does not exist.
        const float w0 = mul(viewProj, float4(origin, 1.0)).w;
        const float wd = mul(viewProj, float4(dir, 0.0)).w;
        float tMax = rayMaxDistance;
        const float W_EPS = 1e-3;
        if (w0 + tMax * wd < W_EPS && abs(wd) > 1e-9) {
            tMax = max(0.0, (W_EPS - w0) / wd);
        }

        float4 h0 = mul(viewProj, float4(origin, 1.0));
        float4 h1 = mul(viewProj, float4(origin + dir * tMax, 1.0));
        if (h0.w > W_EPS && h1.w > W_EPS && tMax > 1e-4) {
            float3 n0 = h0.xyz / h0.w;
            float3 n1 = h1.xyz / h1.w;
            float3 P0 = float3(n0.x * 0.5 + 0.5, 0.5 - n0.y * 0.5, n0.z);
            float3 P1 = float3(n1.x * 0.5 + 0.5, 0.5 - n1.y * 0.5, n1.z);
            float3 D = P1 - P0;

            // The ray's length in mip-0 texels. A ray pointing almost straight
            // down the view axis has none, and no amount of marching a 2D
            // pyramid will resolve it — hand it to the BVH immediately rather
            // than burning 48 steps inside one texel.
            float texelLength = length(D.xy * float2(screenSize));
            if (texelLength > 1.0 && historyValid > 0.5) {
                float sEps = 0.25 / texelLength;
                float s = min(2.0 / texelLength, 1.0); // start clear of the anchor's own surface
                int level = GI_SP_START_MIP;
                bool escaped = false;

                [loop]
                for (int iter = 0; iter < GI_SP_MAX_STEPS && level >= 0; ++iter) {
                    float2 uv = P0.xy + D.xy * s;
                    if (s >= 1.0 || any(uv < 0.0) || any(uv > 1.0)) {
                        escaped = true;
                        break;
                    }
                    // The cell grid is screenSize / 2^level and the LOOKUP is a
                    // Load at the clamped integer texel — not a uv sample. The
                    // pyramid's last texel on each axis has an OVERSIZED
                    // footprint (render/hzb.hpp), so a uv-addressed sample asks
                    // the wrong texel near the right/bottom edge and the march
                    // walks straight through whatever it was reporting.
                    int2 mipSize = mipSizeAt(level);
                    float2 cells = giSpCellGrid(screenSize, level);
                    float2 cellBefore = floor(uv * cells);
                    float hz = hzb.Load(int3(giSpCell(uv, cells, mipSize), level)).r;

                    float sNext = s;
                    // REVERSED-Z, and the pyramid is a MAX (render/hzb.hpp): a
                    // larger stored value is a CLOSER surface, so `hz <= rayZ`
                    // means the ray is in front of everything in the footprint
                    // and may be advanced all the way to that depth plane.
                    if (hz <= P0.z + D.z * s && abs(D.z) > 1e-9) {
                        sNext = max(s, (hz - P0.z) / D.z);
                    }

                    float2 cellAfter = floor((P0.xy + D.xy * sNext) * cells);
                    if (any(cellAfter != cellBefore)) {
                        // The depth-plane advance left this cell, so it is not
                        // evidence about the next one: clamp to the boundary and
                        // coarsen. +2 here against the -1 below is a NET +1 —
                        // the standard Hi-Z ladder.
                        float dt = giSpCellStep(uv, D.xy, cells);
                        sNext = s + dt + sEps;
                        level = min(level + 2, int(mipCount) - 1);
                    }
                    s = sNext;
                    --level;
                }

                if (!escaped && level < 0 && s < 1.0) {
                    float2 uv = P0.xy + D.xy * s;
                    if (all(uv >= 0.0) && all(uv <= 1.0)) {
                        float hz = hzb.Load(int3(int2(uv * float2(screenSize)), 0)).r;
                        float rayZ = giSpLinearDepth(P0.z + D.z * s, depthA, depthB);
                        float surfaceZ = giSpLinearDepth(hz, depthA, depthB);
                        // THE THICKNESS HEURISTIC. The depth buffer says where a
                        // surface starts and nothing about where it ends, so a
                        // ray that arrives behind it is a hit only within an
                        // assumed slab. Beyond the slab the ray passed BEHIND
                        // something the screen cannot see around, and the honest
                        // answer is the BVH's — see the handoff below.
                        if (hz > 0.0 && rayZ >= surfaceZ - 1.0 &&
                            rayZ <= surfaceZ + giSpThickness(surfaceZ)) {
                            float t = worldTAtScreenS(s, h0.w, h1.w);
                            float worldDistance = saturate(t) * tMax;
                            float3 hitPos = origin + dir * worldDistance;

                            // Reproject into LAST frame's screen. Matrices only,
                            // no velocity: the hit is an arbitrary world point,
                            // not this pixel's surface, so the velocity buffer
                            // has nothing to say about it.
                            float4 ph = mul(prevViewProjUnjittered, float4(hitPos, 1.0));
                            if (ph.w > 1e-4) {
                                float2 pn = ph.xy / ph.w;
                                float2 puv = float2(pn.x * 0.5 + 0.5, 0.5 - pn.y * 0.5);
                                if (all(puv >= 0.0) && all(puv <= 1.0)) {
                                    // Disocclusion guard, and an honestly partial
                                    // one: the engine keeps no PREVIOUS-frame
                                    // depth target (docs/rt_lighting_plan.md §1,
                                    // limit 7), so what is compared is this
                                    // frame's depth at the reprojected texel.
                                    // That is exact for a static camera, good for
                                    // a slow one, and wrong for a fast one — in
                                    // which case it rejects and the ray becomes a
                                    // world ray, which is the safe direction.
                                    float prevSurfaceZ =
                                        giSpLinearDepth(
                                            hzb.Load(int3(clamp(int2(puv * float2(screenSize)),
                                                                int2(0, 0), screenSize - 1),
                                                          0))
                                                .r,
                                            depthA, depthB);
                                    float reference = giSpLinearDepth(hz, depthA, depthB);
                                    if (abs(prevSurfaceZ - reference) <=
                                        max(giSpThickness(reference) * 4.0, 0.1 * reference)) {
                                        radiance =
                                            sceneHistory.SampleLevel(sceneHistorySampler, puv, 0).rgb;
                                        hitDistance = worldDistance;
                                        isHit = true;
                                    }
                                }
                            }
                            if (!isHit) {
                                resumeDistance = worldDistance;
                            }
                        } else {
                            resumeDistance = saturate(worldTAtScreenS(s, h0.w, h1.w)) * tMax;
                        }
                    }
                } else {
                    resumeDistance = saturate(worldTAtScreenS(min(s, 1.0), h0.w, h1.w)) * tMax;
                }
            }
        }
        isMiss = !isHit;
    }

    // ---- group compaction -------------------------------------------------
    uint localSlot = 0;
    if (isMiss) {
        InterlockedAdd(gsMissCount, 1u, localSlot);
    }
    if (isHit) {
        InterlockedAdd(gsHitCount, 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    if (groupIndex == 0) {
        if (gsMissCount > 0u) {
            missBuf.InterlockedAdd(0, gsMissCount, gsMissBase);
        } else {
            gsMissBase = 0u;
        }
        if (gsHitCount > 0u) {
            counters.InterlockedAdd(0, gsHitCount);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (isMiss) {
        uint index = gsMissBase + localSlot;
        if (index < missCapacity) {
            missBuf.Store4(giSpMissRecordOffset(index),
                           uint4(uint(probeIndex), uint(texel), asuint(resumeDistance), 0u));
        } else {
            // Should never fire: the buffer is sized for every traced ray
            // missing. A nonzero `dropped` means the sizing rule in
            // screen_probe_math.hpp and this shader disagree.
            counters.InterlockedAdd(12, 1u);
        }
    }

    if (!live) {
        // A tile with no probe still owns its atlas texels; leaving them alone
        // would let the filter accumulate whatever the last probe that lived
        // here saw.
        radianceAtlas[atlasTexel] = float4(0, 0, 0, 0);
    } else if (isHit) {
        radianceAtlas[atlasTexel] = float4(radiance, hitDistance);
    } else if (!traced) {
        radianceAtlas[atlasTexel] = float4(0, 0, 0, 0); // below the horizon: a resolved zero
    } else {
        // Handed off. Alpha < 0 says "unresolved" so the filter keeps history
        // for this texel if the BVH pass never writes it (a dropped record).
        radianceAtlas[atlasTexel] = float4(0, 0, 0, GI_SP_UNRESOLVED);
    }
}
