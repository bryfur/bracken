// gi_screen_probe_project.comp.hlsl - probe-space SPATIAL filtering fused with
// the L1 spherical-harmonic projection: the pass that turns 64 radiance texels
// per probe into the one 64-byte SH record the forward shaders read.
//
// Phase R4b of docs/rt_lighting_plan.md §5.5. One threadgroup per probe, one
// thread per octahedral texel.
//
// THE SPATIAL FILTER IS PER DIRECTION, not per probe. Thread i owns octahedral
// texel i, and gathers texel i from the 3x3 neighbourhood weighted by how much
// each neighbour's anchor agrees with this one's plane. That is the whole
// difference between a probe filter and a blur: averaging probes would smear a
// wall's radiance onto the floor beside it, while averaging what nine probes
// saw IN THE SAME DIRECTION is a variance reduction on one estimate.
//
// It lives here rather than in the filter pass because SDL3 GPU has no
// intra-pass synchronization — a 3x3 gather needs its neighbours' temporal
// results complete, which only a later pass can guarantee — and because this
// pass was already reading every texel of every probe. Fusing them costs one
// extra atlas read per thread and saves a whole atlas.

#include "gi_screen_probe.hlsli"

// ---------------------------------------------------------------------------
// Uniforms (set 2, b0) — mirrors gpu::GiScreenProbeProjectUniforms.
// ---------------------------------------------------------------------------
[[vk::binding(0, 2)]]
cbuffer GiScreenProbeProjectUniforms : register(b0, space2) {
    int2 probeGrid;
    float normalPower;
    uint frameIndex;
};

// --- sampled (1) ---
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float4> filteredAtlas : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState filteredAtlasSampler : register(s0, space0);

// --- read-only storage buffers (1) ---
[[vk::binding(1, 0)]] StructuredBuffer<GiScreenProbe> probes : register(t1, space0);

// --- read-write (set 1): storage buffers only ---
// A BUFFER, not a texture. SDL3 GPU caps a shader stage at 16 samplers
// (SDL_CreateGPUShader asserts past it) and the scene shaders are already at
// 16, so the published SH cannot be a sampled texture there — while they have
// storage-buffer headroom. See gpu::GiScreenProbeSh for the full argument.
[[vk::binding(0, 1)]] RWStructuredBuffer<GiScreenProbeSh> shBuffer : register(u0, space1);

// Four L1 coefficients x RGB, per thread, folded by thread 0. 3 KB of LDS.
groupshared float3 gsSh[4][GI_SP_RAYS];

[numthreads(GI_SP_RAYS, 1, 1)]
void CSMain(uint3 groupId : SV_GroupID, uint groupIndex : SV_GroupIndex) {
    const int probeCount = probeGrid.x * probeGrid.y;
    const int probeIndex = int(groupId.x);
    if (probeIndex >= probeCount) {
        return;
    }
    const int texel = int(groupIndex);
    const int2 probeCoord = giSpCoord(probeIndex, probeGrid);
    GiScreenProbe probe = probes[probeIndex];
    const bool valid = (probe.flags & GI_SP_FLAG_VALID) != 0u;

    float3 radiance = float3(0, 0, 0);
    if (valid) {
        float wsum = 0.0;
        float3 acc = float3(0, 0, 0);
        [unroll]
        for (int oy = -1; oy <= 1; ++oy) {
            [unroll]
            for (int ox = -1; ox <= 1; ++ox) {
                int2 nc = probeCoord + int2(ox, oy);
                if (any(nc < 0) || any(nc >= probeGrid)) {
                    continue;
                }
                GiScreenProbe n = probes[giSpIndex(nc, probeGrid)];
                if ((n.flags & GI_SP_FLAG_VALID) == 0u) {
                    continue;
                }
                float4 sample = filteredAtlas.Load(int3(giSpRadianceTexel(nc, texel), 0));
                if (sample.a < 0.0) {
                    continue; // unresolved: no sample, not a black one
                }
                float w = (ox == 0 && oy == 0)
                              ? 1.0
                              : giSpBilateral(probe.cameraDistance, n.cameraDistance, probe.normal,
                                              n.normal, normalPower);
                if (w <= 1e-4) {
                    continue;
                }
                acc += sample.rgb * w;
                wsum += w;
            }
        }
        radiance = wsum > 1e-6 ? acc / wsum : float3(0, 0, 0);
    }

    // ---- L1 projection ----------------------------------------------------
    float3 dir = giSpTexelDirection(texel, frameIndex);
    float4 basis = giSpShBasis(dir);
    [unroll]
    for (int k = 0; k < 4; ++k) {
        gsSh[k][groupIndex] = radiance * (basis[k] * GI_SP_TEXEL_SOLID_ANGLE);
    }
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction over the 64 texels, four coefficients at a time.
    [unroll]
    for (uint stride = GI_SP_RAYS / 2; stride > 0; stride >>= 1) {
        if (groupIndex < stride) {
            [unroll]
            for (int k = 0; k < 4; ++k) {
                gsSh[k][groupIndex] += gsSh[k][groupIndex + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (groupIndex == 0) {
        GiScreenProbeSh out_;
        // ONE QUAD PER COLOUR CHANNEL — (c0, c1, c2, c3) of R, then G, then B.
        out_.shR = float4(gsSh[0][0].r, gsSh[1][0].r, gsSh[2][0].r, gsSh[3][0].r);
        out_.shG = float4(gsSh[0][0].g, gsSh[1][0].g, gsSh[2][0].g, gsSh[3][0].g);
        out_.shB = float4(gsSh[0][0].b, gsSh[1][0].b, gsSh[2][0].b, gsSh[3][0].b);
        // Metadata: the anchor's normal and camera distance, which is what the
        // forward gather weights its four probes with. A ZERO camera distance is
        // the "no probe here" signal — a tile that was all sky — and the only
        // one the forward pass gets, so it must be written even for an invalid
        // probe rather than left holding the last frame's occupant.
        out_.meta = valid ? float4(probe.normal, probe.cameraDistance) : float4(0, 0, 0, 0);
        shBuffer[probeIndex] = out_;
    }
}
