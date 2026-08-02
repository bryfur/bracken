// gi_screen_probe_place.comp.hlsl - one thread per screen probe: pick the
// tile's anchor pixel, unproject it, and publish the probe record every later
// pass reads.
//
// Phase R4b of docs/rt_lighting_plan.md §5.5.
//
// WHY THIS IS A PASS OF ITS OWN and not the first sixty-fourth of the march.
// Five things need the anchor: the march (ray origin), the BVH handoff (ray
// origin again, in a dispatch that has no idea which pixel it came from), the
// temporal filter (the pixel to reproject through the velocity buffer), the
// spatial filter (the plane to weight neighbours against) and the SH atlas
// (the metadata the forward gather weights with). Recomputing a nine-tap search
// in each of them is both slower and — far worse — five places for the search
// order to drift. 5700 threads once a frame is the cheapest pass in the block.
//
// It also OWNS THE FRAME'S ZEROING: thread 0 clears the miss-buffer header and
// the diagnostic counters. That is not a convenience either. Both must be zero
// before the march's first InterlockedAdd, and this pass is the only one that
// runs before it, so a separate clear pass would exist purely to be forgotten.

#include "gi_screen_probe.hlsli"

// ---------------------------------------------------------------------------
// Uniforms (set 2, b0) — mirrors gpu::GiScreenProbePlaceUniforms.
// ---------------------------------------------------------------------------
[[vk::binding(0, 2)]]
cbuffer GiScreenProbePlaceUniforms : register(b0, space2) {
    float4x4 invViewProj;
    float3 cameraPosition;
    uint frameIndex;
    int2 probeGrid;
    int2 screenSize;
    uint missCapacity;
    uint _placePad0;
    uint _placePad1;
    uint _placePad2;
};

// ---------------------------------------------------------------------------
// Bindings. SET 0 ORDER IS THE CONTRACT: sampled, then read-only storage
// textures, then read-only storage buffers (shaders/README.md).
// ---------------------------------------------------------------------------

// --- sampled (2) ---
// Mip 0 of the Hi-Z pyramid IS this frame's prepass depth, copied into R32F
// because a D32 texture cannot be a compute storage image (render/hzb.hpp). The
// probe reads it here rather than the depth target so the whole gather has one
// depth source and cannot disagree with the march about which frame it is
// looking at.
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float> hzb : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState hzbSampler : register(s0, space0);

// The prepass G-buffer: octahedral world normal in RGB, roughness in A. Read
// rather than reconstructed from depth taps for the reason scene_renderer.cpp
// records at the allocation site — a four-tap reconstruction flips between
// neighbouring pixels on foliage, and here that would move the probe's PLANE,
// which every bilateral weight downstream is measured against.
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<float4> gbufferNormal : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState gbufferNormalSampler : register(s1, space0);

// --- read-write (set 1): storage buffers only ---
[[vk::binding(0, 1)]] RWStructuredBuffer<GiScreenProbe> probes : register(u0, space1);
[[vk::binding(1, 1)]] RWByteAddressBuffer missBuf : register(u1, space1);
[[vk::binding(2, 1)]] RWByteAddressBuffer counters : register(u2, space1);

// Inverse of octEncode in prepass_gbuffer.frag.hlsl.
float3 placeOctDecode(float2 e) {
    e = e * 2.0 - 1.0;
    float3 n = float3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0, float2(-t, -t), float2(t, t));
    return normalize(n);
}

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x == 0) {
        // Header word 0 is the miss count; the other three are reserved so the
        // records stay 16-byte aligned for a ByteAddressBuffer read.
        missBuf.Store4(0, uint4(0, 0, 0, 0));
        counters.Store4(0, uint4(0, 0, 0, 0));
    }

    int probeCount = probeGrid.x * probeGrid.y;
    if (int(tid.x) >= probeCount) {
        return;
    }
    int2 probeCoord = giSpCoord(int(tid.x), probeGrid);
    int2 tileOrigin = probeCoord * GI_SP_TILE;

    GiScreenProbe probe;
    probe.worldPosition = float3(0, 0, 0);
    probe.flags = 0;
    probe.normal = float3(0, 1, 0);
    probe.cameraDistance = 0.0;

    // THE FIXED SEARCH ORDER. Centre first, then the quarter-points, then the
    // edge midpoints; first candidate with a non-sky depth wins. Deterministic
    // in the depth buffer alone — a probe that hopped between two surfaces
    // inside its tile would invalidate its own history every frame.
    [loop]
    for (int attempt = 0; attempt < GI_SP_ANCHOR_ATTEMPTS; ++attempt) {
        int2 pixel = tileOrigin + giSpAnchorOffset(attempt);
        if (pixel.x >= screenSize.x || pixel.y >= screenSize.y) {
            continue; // partial edge tile
        }
        float2 uv = (float2(pixel) + 0.5) / float2(screenSize);
        float depth = hzb.SampleLevel(hzbSampler, uv, 0).r;
        // Reversed-Z: 0 is the cleared far plane, i.e. sky. A sky tile gets no
        // probe at all and the forward gather falls back to the world field
        // there, which is the correct answer: there is no surface to gather for.
        if (depth <= 0.0) {
            continue;
        }
        float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);
        float4 world = mul(invViewProj, float4(ndc, depth, 1.0));
        if (abs(world.w) < 1e-8) {
            continue;
        }
        float3 P = world.xyz / world.w;

        float3 N = placeOctDecode(gbufferNormal.SampleLevel(gbufferNormalSampler, uv, 0).rgb);

        probe.worldPosition = P;
        probe.normal = N;
        probe.cameraDistance = distance(P, cameraPosition);
        probe.flags = giSpPackFlags(giSpAnchorOffset(attempt), true);
        break;
    }

    probes[tid.x] = probe;
}
