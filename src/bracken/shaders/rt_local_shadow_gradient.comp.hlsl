// rt_local_shadow_gradient.comp.hlsl — the A-SVGF temporal gradient for the
// LOCAL (point/spot) visibility signal. Phase R5b of
// docs/rt_lighting_plan.md §5.6.
//
// Same mechanism as rt_shadow_gradient.comp.hlsl — one re-traced ray per 3x3
// stratum, this frame's scene against the previous frame's sample, compared
// with the previous frame's raw bit — with one addition the sun does not need.
//
// THE SUN IS ONE LIGHT AND THE LOCAL SIGNAL IS N. A replay of the sun ray only
// has to reproduce a direction. A replay of the local ray has to reproduce
// WHICH LIGHT the previous frame's reservoir picked, or it is not a replay at
// all: aiming at a different lantern would report a full gradient on every
// pixel whose reservoir happened to move, which is most of them, every frame.
//
// The previous frame's reservoir already stores that choice — as a PACKED INDEX
// into the previous frame's light array, which this frame's array need not
// agree with. R2a's prev->current remap buffer is exactly the translation, and
// it is what the pass's own temporal merge uses one dispatch earlier. Two
// outcomes:
//
//  * the light survived: re-trace at it, with the previous frame's sample.
//  * the light is GONE (remap invalid): that is an unambiguous lighting change
//    and needs no ray at all — the gradient is 1 by inspection. A torch being
//    extinguished is precisely the event fixed-alpha accumulation smears.
//
// A pixel whose previous reservoir held no light at all reports 0: there is
// nothing to replay, and inventing a comparison would report the reservoir's
// own churn as a lighting change. That gap is deliberate and is what the dual
// fast/slow history (§5.6, rt_shadow_denoise_temporal.comp.hlsl) is for.

#include "rt_sampling.hlsli"
#include "rt_denoise.hlsli"

#define RT_TRACE_TYPES
#include "rt_trace.hlsli"

[[vk::binding(0, 2)]]
cbuffer RtLocalShadowGradientUniforms : register(b0, space2) {
    float4x4 invViewProj;
    uint2 tracedSize;
    uint2 stratumCount;
    float normalBias;
    float depthBias;
    float lightSourceRadius;
    /// THIS frame's counter; the replay ray uses frameIndex - 1.
    uint frameIndex;
    uint tlasNodeCount;
    uint hasGBuffer;
    uint historyValid;
    /// R6 — the emissive record set the PREVIOUS frame's reservoirs were
    /// written against. Zero either when there are no records or when the
    /// record set was renumbered since, which is the epoch test.
    uint emissiveCount;
    uint emissiveHistoryValid;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

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

/// The PREVIOUS frame's raw traced bits, 8x4 per uint.
[[vk::binding(3, 0)]] Texture2D<uint> prevShadowBits : register(t3, space0);

[[vk::binding(4, 0)]] StructuredBuffer<BvhNodeGpu> tlasNodes : register(t4, space0);
[[vk::binding(5, 0)]] StructuredBuffer<TlasInstance> tlasInstances : register(t5, space0);
[[vk::binding(6, 0)]] StructuredBuffer<BvhNodeGpu> blasNodes : register(t6, space0);
[[vk::binding(7, 0)]] StructuredBuffer<BvhTriangle> blasTriangles : register(t7, space0);

// Four cluster buffers (light data, froxel offsets, froxel indices, remap) —
// eight read-only storage buffers in total, which is SDL3 GPU's hard ceiling
// for a compute pipeline. The froxel accessors go unused here (a replay does
// not draw new candidates), but the include is taken whole rather than opted
// out of piecemeal: it also carries `clusterParams`, whose view matrix is the
// only source of a view depth that agrees with the CPU binner's.
#define CLUSTER_COMPUTE_SSBO_BINDING_BASE 8
#define CLUSTER_COMPUTE_PARAMS_BINDING 1
#include "cluster_compute.hlsli"

// R6's emissive record type, so the binding block below can declare its buffer.
#define RT_EMISSIVE_TYPES
#include "rt_emissive.hlsli"

/// Mirrors render::lighting::Reservoir (16 B) and rt_local_shadow.comp.hlsl's
/// RtReservoir field for field.
struct RtReservoir {
    uint lightIndex;
    float weight;
    float sampleCount;
    float depth;
};

#define RT_RESERVOIR_NO_LIGHT 0xFFFFFFFFu
// Source tag — see rt_local_shadow.comp.hlsl, which owns the convention.
#define RT_RESERVOIR_EMISSIVE_FLAG 0x80000000u
#define RT_RESERVOIR_EMISSIVE_MASK 0x7FFFFFFFu

[[vk::binding(0, 1)]] RWTexture2D<float> gradientOut : register(u0, space1);
/// LAST FRAME'S reservoirs — READ ONLY, despite the read-write declaration, for
/// the reason rt_local_shadow.comp.hlsl spells out: the eight read-only
/// storage-buffer slots are spent above.
[[vk::binding(1, 1)]] RWStructuredBuffer<RtReservoir> prevReservoirs : register(u1, space1);
/// R6's emissive light records, read-only in a read-write slot for the same
/// reason. A reservoir may now name one of these instead of an analytic light,
/// and the replay has to be able to aim at it — WITHOUT this the flagged index
/// falls through clusterRemapPrevLight as out-of-range, every emissive-lit
/// stratum reports a full gradient every frame, and the denoiser's history is
/// reset forever exactly where the new lights are.
[[vk::binding(2, 1)]] RWStructuredBuffer<EmissiveRecord> emissiveRecords : register(u2, space1);

#define RT_EMISSIVE_FUNCTIONS
#include "rt_emissive.hlsli"

#define RT_TRACE_TRAVERSAL
#include "rt_trace.hlsli"

float3 worldAt(int2 c, uint2 size) {
    float d = sceneDepth.Load(int3(c, 0));
    float2 ndc = (float2(c) + 0.5) / float2(size) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 w = mul(invViewProj, float4(ndc, d, 1.0));
    return w.xyz / w.w;
}

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

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= stratumCount.x || tid.y >= stratumCount.y) {
        return;
    }
    const int2 stratum = int2(tid.xy);

    if (historyValid == 0u || clusterTotalLights() == 0u) {
        gradientOut[stratum] = 0.0;
        return;
    }

    const uint2 pick = rtGradientPixelInStratum(uint2(stratum), frameIndex);
    const int2 coord = stratum * RT_GRADIENT_STRATUM + int2(pick);
    if (coord.x >= int(tracedSize.x) || coord.y >= int(tracedSize.y)) {
        gradientOut[stratum] = 0.0;
        return;
    }

    const float2 uv = (float2(coord) + 0.5) / float2(tracedSize);
    const float2 velocity = velocityTexture.SampleLevel(velocitySampler, uv, 0);
    const float2 prevUv = uv - velocity;
    if (any(prevUv < 0.0) || any(prevUv > 1.0)) {
        gradientOut[stratum] = 0.0;
        return;
    }
    const int2 prevCoord =
        clamp(int2(prevUv * float2(tracedSize)), int2(0, 0), int2(tracedSize) - 1);

    uint2 depthSize;
    sceneDepth.GetDimensions(depthSize.x, depthSize.y);
    const int2 depthCoord = int2((float2(coord) + 0.5) / float2(tracedSize) * float2(depthSize));
    const float depth = sceneDepth.Load(int3(depthCoord, 0));
    if (depth <= 0.0) {
        gradientOut[stratum] = 0.0; // sky: no local light reaches it, ever
        return;
    }

    float3 normal;
    if (hasGBuffer != 0u) {
        normal = octDecode(sceneNormal.Load(int3(depthCoord, 0)).xy);
    } else {
        normal = reconstructNormal(depthCoord, depthSize);
    }

    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;
    const float4 world = mul(invViewProj, float4(ndc, depth, 1.0));
    const float3 worldPos = world.xyz / world.w;
    const float viewDepth = -mul(clusterParams.view, float4(worldPos, 1.0)).z;

    const RtReservoir hist =
        prevReservoirs[uint(prevCoord.y) * tracedSize.x + uint(prevCoord.x)];

    // The reservoir merge's disocclusion test, TIGHTENED by two orders of
    // magnitude — see RT_GRADIENT_DEPTH_REJECT_RELATIVE. The merge asks whether
    // the stored reservoir is reusable; this asks whether the stored BIT was
    // measured at the same point, which is a strictly stronger question and is
    // the difference between reporting a lighting change and reporting the TAA
    // jitter moving the sample half a pixel along a leaf.
    if (!(hist.depth > 0.0 &&
          abs(hist.depth - viewDepth) <= RT_GRADIENT_DEPTH_REJECT_RELATIVE * max(viewDepth, 1.0))) {
        gradientOut[stratum] = 0.0;
        return;
    }
    if (hist.lightIndex == RT_RESERVOIR_NO_LIGHT || hist.sampleCount <= 0.0) {
        gradientOut[stratum] = 0.0; // nothing to replay — see the file comment
        return;
    }

    // R6: the stored choice may be an EMISSIVE RECORD rather than an analytic
    // light. Same two outcomes, one translation each — the remap for analytic
    // lights, the record epoch for emissive ones. An emissive index whose epoch
    // has turned over names a different triangle now, and "the light I was
    // shadowed by is not there any more" is exactly the gradient's 1.0 case.
    const bool histEmissive = (hist.lightIndex & RT_RESERVOIR_EMISSIVE_FLAG) != 0u;
    uint mapped;
    if (histEmissive) {
        const uint rec = hist.lightIndex & RT_RESERVOIR_EMISSIVE_MASK;
        mapped = (emissiveHistoryValid != 0u && rec < emissiveCount) ? rec : LIGHT_REMAP_INVALID;
    } else {
        mapped = clusterRemapPrevLight(hist.lightIndex);
    }
    if (mapped == LIGHT_REMAP_INVALID) {
        // The light this pixel was shadowed by no longer exists. No ray needed:
        // the answer changed by construction.
        gradientOut[stratum] = 1.0;
        return;
    }

    const uint2 prevTile = uint2(prevCoord) / RT_TILE_SIZE;
    const uint2 prevInTile = uint2(prevCoord) % RT_TILE_SIZE;
    const float prevLit = rtUnpackVisibility(prevShadowBits.Load(int3(int2(prevTile), 0)), prevInTile);

    // ---- replay: an EMISSIVE record, the previous frame's sample ----------
    //
    // Transliterated from the trace's own emissive branch, down to the relative
    // ray-length margin that stops the emitter occluding itself. `rtSample2D`
    // at (prevCoord, frameIndex - 1) returns EXACTLY the pair the previous
    // frame drew — the replayability requirement §2 of the plan states — so
    // this is a replay of the ray that ran, not a fresh sample of the same
    // light.
    if (histEmissive) {
        const EmissiveRecord er = emissiveRecords[mapped];
        const float2 xi = rtSample2D(uint2(prevCoord), frameIndex - 1u);
        const float3 pt = rtEmissiveSamplePoint(er, xi);
        const float3 toPt = pt - worldPos;
        const float dist = length(toPt);
        const float3 L = toPt / max(dist, 1e-4);
        const float ndotl = dot(normal, L);
        bool emissiveLit = true;
        if (ndotl > 0.0 && dist > 1e-4) {
            const float slope = saturate(1.0 - ndotl);
            const float3 rayOrigin =
                worldPos + normal * (normalBias * (1.0 + slope * 4.0)) + L * depthBias;
            const float rayLen = min(dist * 0.998, dist - depthBias) - normalBias;
            emissiveLit = rayLen <= 0.0 ? true : !sceneOccluded(rayOrigin, L, rayLen);
        }
        gradientOut[stratum] = abs((emissiveLit ? 1.0 : 0.0) - prevLit);
        return;
    }

    // ---- replay: the same analytic light, the previous frame's sample ------
    bool replayLit = true;
    const ClusterLightSample geom = clusterLightGeometry(mapped);
    const float3 toLight = geom.position - worldPos;
    const float dist = length(toLight);
    const float3 L = toLight / max(dist, 1e-4);

    // Per-light emitter radius since R6, with the pass default as the fallback
    // — the same resolution the trace makes, or the replay would sample a
    // different cone from the ray it is meant to be reproducing.
    const float emitterRadius =
        geom.source_radius > 0.0 ? geom.source_radius : lightSourceRadius;
    const float sinThetaMax = min(emitterRadius / max(dist, 1e-4), 1.0);
    const float cosThetaMax = sqrt(saturate(1.0 - sinThetaMax * sinThetaMax));

    const float ndotl = dot(normal, L);
    if (ndotl > 0.0) {
        const float slope = saturate(1.0 - ndotl);
        const float3 rayOrigin =
            worldPos + normal * (normalBias * (1.0 + slope * 4.0)) + L * depthBias;
        const float2 xi = rtSample2D(uint2(prevCoord), frameIndex - 1u);
        const float3 rayDir = sampleCone(L, cosThetaMax, xi);

        if (dot(normal, rayDir) <= 0.0) {
            replayLit = false;
        } else {
            const float rayLen = max(dist - depthBias - normalBias, 0.0);
            replayLit = rayLen <= 0.0 ? true : !sceneOccluded(rayOrigin, rayDir, rayLen);
        }
    }

    gradientOut[stratum] = abs((replayLit ? 1.0 : 0.0) - prevLit);
}
