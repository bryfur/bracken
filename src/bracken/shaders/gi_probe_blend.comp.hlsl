// gi_probe_blend.comp.hlsl - fold one update's rays into the irradiance atlas.
//
// One THREADGROUP per updated probe, one thread per texel of its 8x8 tile
// (6x6 payload + a one-texel border ring). Each payload thread integrates the
// probe's whole ray set against its own direction; the border threads then
// replicate the octahedral wrap out of groupshared memory.
//
// WHY THE BORDER IS IN THIS PASS and not a fourth dispatch: the border is a
// pure function of the payload this group just computed. A separate pass would
// have to re-read the atlas it had only just written (a full round trip through
// memory plus a second dispatch's launch) to copy 28 of 64 texels. In-group it
// costs one barrier and no extra bandwidth. The coupling it buys is that
// [numthreads] MUST equal the tile size — groupshared spans a group and nothing
// else — which is why both are written as GI_IRRADIANCE_TILE and the dispatch
// side covers exactly tile x (tile * probeCount) elements.
//
// WHAT IS STORED: cosine-weighted MEAN INCOMING RADIANCE, i.e. irradiance/pi.
// Same convention as the IBL irradiance cube, so gi_sample.hlsli substitutes one
// for the other with no scale factor. See gi_probe_trace.comp.hlsl's units note.

#include "gi_probe.hlsli"

struct GiProbeUpdate {
    float3 worldPosition;
    uint slot;
    uint flags;
    float3 _pad;
};
#define GI_UPDATE_FLAG_UNINITIALIZED 1u

[[vk::binding(0, 2)]]
cbuffer GiProbeBlendUniforms : register(b0, space2) {
    uint giProbeCount;
    uint giRayCount;
    float giDistanceClamp; // visibility blend only; shared struct
    float giRayMaxDistance;
    uint giFrameIndex;
    uint giCascadeIndex; // which BAND of the shared atlas this dispatch writes
    /// R5b: the analytic sun gradient's global hysteresis floor, [0, 1]. The
    /// renderer knows on the CPU how far the sun moved this frame, so a
    /// time-of-day scrub or a lightning spike collapses the field's
    /// accumulation window BEFORE any probe ray reports the change. Used by the
    /// IRRADIANCE blend only — the visibility moments describe geometry, which
    /// the sun moving does not alter — but the struct is shared by both passes,
    /// so the field is declared in both. 0 is the identity.
    float giLightingReset;
    uint _giPad0;
};

// SET 0 ORDER: (no sampled textures) READ-ONLY storage textures, then
// READ-ONLY storage buffers — shaders/README.md. Getting this backwards binds
// the wrong resource to every slot and validates cleanly while doing it.
[[vk::binding(0, 0)]] Texture2D<float4> giProbeOffset : register(t0, space0);
[[vk::binding(1, 0)]] StructuredBuffer<float4> giRayResults : register(t1, space0);
[[vk::binding(2, 0)]] StructuredBuffer<GiProbeUpdate> giUpdates : register(t2, space0);

// The EMA reads this target as well as writing it. An explicit image format is
// what makes that legal without the optional `shaderStorageImageReadWithoutFormat`
// device feature — a storage image declared Unknown may only be written on
// hardware that does not advertise it.
[[vk::image_format("rgba16f")]]
[[vk::binding(0, 1)]] RWTexture2D<float4> giIrradiance : register(u0, space1);

/// The payload the border ring replicates.
groupshared float3 gPayload[GI_IRRADIANCE_PAYLOAD][GI_IRRADIANCE_PAYLOAD];

[numthreads(GI_IRRADIANCE_TILE, GI_IRRADIANCE_TILE, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID) {
    uint probeIndex = tid.y / uint(GI_IRRADIANCE_TILE);
    if (probeIndex >= giProbeCount) {
        return;
    }
    GiProbeUpdate probe = giUpdates[probeIndex];
    bool fresh = (probe.flags & GI_UPDATE_FLAG_UNINITIALIZED) != 0u;

    // A probe classified as sitting inside geometry contributes nothing worth
    // blending. Its rays were still cast — the classify pass needs them to
    // notice when it stops being buried — but its tile is left alone. The test
    // is uniform across the group, so this early return cannot split the group
    // across the barrier below.
    float4 offset = giProbeOffset.Load(int3(giProbeStateTexel(int(probe.slot), int(giCascadeIndex)), 0));
    if (!fresh && offset.w == GI_CLASS_INSIDE_GEOMETRY) {
        return;
    }

    int2 tileLocal = int2(lid.xy);
    int2 tileOrigin = giProbeAtlasOrigin(int(probe.slot), GI_IRRADIANCE_TILE, int(giCascadeIndex));
    bool isPayload = tileLocal.x >= GI_BORDER && tileLocal.x < GI_IRRADIANCE_TILE - GI_BORDER &&
                     tileLocal.y >= GI_BORDER && tileLocal.y < GI_IRRADIANCE_TILE - GI_BORDER;

    if (isPayload) {
        int2 local = tileLocal - GI_BORDER;
        float3 texelDir = giTexelDirection(local, GI_IRRADIANCE_PAYLOAD);

        // The per-frame rotation is constant across the probe's rays, so it is
        // hoisted out of the loop; only the Fibonacci direction varies. It MUST
        // be the rotation the trace used, which is why the frame counter is a
        // uniform of both passes and not re-derived from anything local.
        float3 c0, c1, c2;
        giRayRotation(giFrameIndex, c0, c1, c2);

        float3 sum = float3(0, 0, 0);
        float wsum = 0.0;
        for (uint r = 0; r < giRayCount; ++r) {
            float3 d = giSphericalFibonacci(r, giRayCount);
            float3 rayDir = normalize(c0 * d.x + c1 * d.y + c2 * d.z);
            float w = max(0.0, dot(texelDir, rayDir));
            if (w <= 0.0) {
                continue;
            }
            sum += w * giRayResults[probeIndex * giRayCount + r].rgb;
            wsum += w;
        }

        // Cosine-weighted MEAN, not sum: dividing by the weight total is what
        // makes the stored quantity irradiance/pi independently of how many
        // rays this quality tier casts, so changing GiQuality mid-session does
        // not rescale the field.
        float3 estimate = wsum > 1e-6 ? sum / wsum : float3(0, 0, 0);
        float3 previous = giIrradiance[tileOrigin + tileLocal].rgb;
        float3 result;
        if (fresh) {
            // Nothing usable in this slot — its previous occupant described a
            // different point in the world. Take the estimate wholesale; that
            // is what makes a scrolled-in probe correct in one update instead
            // of fading in from a stranger's radiance over 76.
            result = estimate;
        } else if (wsum <= 1e-6) {
            result = previous;
        } else {
            result = giBlend(previous, estimate, giLightingReset);
        }

        giIrradiance[tileOrigin + tileLocal] = float4(result, 1.0);
        gPayload[local.y][local.x] = result;
    }

    GroupMemoryBarrierWithGroupSync();

    if (!isPayload) {
        int2 src = giBorderSource(tileLocal, GI_IRRADIANCE_PAYLOAD);
        giIrradiance[tileOrigin + tileLocal] = float4(gPayload[src.y][src.x], 1.0);
    }
}
