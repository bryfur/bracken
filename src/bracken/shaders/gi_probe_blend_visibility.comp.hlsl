// gi_probe_blend_visibility.comp.hlsl - fold one update's ray DISTANCES into
// the visibility atlas: mean and mean-squared, for the Chebyshev leak test
// R1c's sampler performs.
//
// Same shape as gi_probe_blend.comp.hlsl (one threadgroup per probe, one thread
// per texel of the 16x16 tile, border replicated out of groupshared) and the
// same reasoning behind it. Two things differ, both deliberate:
//
//  * THE DISTANCE IS CLAMPED to GI_VISIBILITY_DISTANCE_SCALE * spacing. The
//    Chebyshev test only ever compares a shading point's distance-to-probe
//    against the stored mean, and a point is at most about one spacing away
//    from the probe that answers for it. Letting one ray that flew 4000 units
//    into open sky drag the mean up would make the test declare every nearby
//    wall visible, which is exactly the leak the moments exist to prevent.
//    Clamping also folds misses in for free: a miss recorded rayMaxDistance,
//    which clamps to "as far as anything matters", i.e. no occluder.
//
//  * THE WEIGHTING LOBE IS SHARP (cos^50 rather than cos^1). Visibility is a
//    high-frequency signal — a wall is there or it is not — so a texel must
//    take only the rays that went essentially its way. The price is that at
//    32-96 rays most texels get NO ray inside their lobe on any given update
//    and simply keep their previous value; the per-frame rotation of the ray
//    set is what makes every texel get its turn over successive updates. That
//    is the DDGI trade, and it is why the ray set is rotated rather than fixed.

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
    float giDistanceClamp;
    float giRayMaxDistance;
    uint giFrameIndex;
    uint _giPad0;
    uint _giPad1;
    uint _giPad2;
};

// SET 0 ORDER: READ-ONLY storage textures, then READ-ONLY storage buffers.
[[vk::binding(0, 0)]] Texture2D<float4> giProbeOffset : register(t0, space0);
[[vk::binding(1, 0)]] StructuredBuffer<float4> giRayResults : register(t1, space0);
[[vk::binding(2, 0)]] StructuredBuffer<GiProbeUpdate> giUpdates : register(t2, space0);

// The EMA reads this target as well as writing it. An explicit image format is
// what makes that legal without the optional `shaderStorageImageReadWithoutFormat`
// device feature — a storage image declared Unknown may only be written on
// hardware that does not advertise it.
[[vk::image_format("rg16f")]]
[[vk::binding(0, 1)]] RWTexture2D<float2> giVisibility : register(u0, space1);

groupshared float2 gPayload[GI_VISIBILITY_PAYLOAD][GI_VISIBILITY_PAYLOAD];

[numthreads(GI_VISIBILITY_TILE, GI_VISIBILITY_TILE, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 lid : SV_GroupThreadID) {
    uint probeIndex = tid.y / uint(GI_VISIBILITY_TILE);
    if (probeIndex >= giProbeCount) {
        return;
    }
    GiProbeUpdate probe = giUpdates[probeIndex];
    bool fresh = (probe.flags & GI_UPDATE_FLAG_UNINITIALIZED) != 0u;

    float4 offset = giProbeOffset.Load(int3(giProbeStateTexel(int(probe.slot)), 0));
    if (!fresh && offset.w == GI_CLASS_INSIDE_GEOMETRY) {
        return;
    }

    int2 tileLocal = int2(lid.xy);
    int2 tileOrigin = giProbeAtlasOrigin(int(probe.slot), GI_VISIBILITY_TILE);
    bool isPayload = tileLocal.x >= GI_BORDER && tileLocal.x < GI_VISIBILITY_TILE - GI_BORDER &&
                     tileLocal.y >= GI_BORDER && tileLocal.y < GI_VISIBILITY_TILE - GI_BORDER;

    if (isPayload) {
        int2 local = tileLocal - GI_BORDER;
        float3 texelDir = giTexelDirection(local, GI_VISIBILITY_PAYLOAD);

        float3 c0, c1, c2;
        giRayRotation(giFrameIndex, c0, c1, c2);

        float2 sum = float2(0, 0);
        float wsum = 0.0;
        for (uint r = 0; r < giRayCount; ++r) {
            float3 d = giSphericalFibonacci(r, giRayCount);
            float3 rayDir = normalize(c0 * d.x + c1 * d.y + c2 * d.z);
            float w = pow(max(0.0, dot(texelDir, rayDir)), GI_VISIBILITY_SHARPNESS);
            if (w <= 1e-8) {
                continue;
            }
            // abs(): a backface hit is still an occluder at that distance. The
            // sign carries the classification signal, not a distance.
            float dist = min(abs(giRayResults[probeIndex * giRayCount + r].w), giDistanceClamp);
            sum += w * float2(dist, dist * dist);
            wsum += w;
        }

        float2 previous = giVisibility[tileOrigin + tileLocal];
        float2 result;
        if (wsum <= 1e-8) {
            // No ray went this way on this update. An unfilled probe has no
            // previous value worth keeping, so seed it with "nothing in the
            // way" rather than with whatever its slot's last occupant saw.
            result = fresh ? float2(giDistanceClamp, giDistanceClamp * giDistanceClamp) : previous;
        } else {
            float2 estimate = sum / wsum;
            // The same change-adaptive EMA as the irradiance blend, so the two
            // halves of a probe never describe different moments in time.
            result = fresh ? estimate : giBlend(float3(previous, 0.0), float3(estimate, 0.0)).xy;
        }

        giVisibility[tileOrigin + tileLocal] = result;
        gPayload[local.y][local.x] = result;
    }

    GroupMemoryBarrierWithGroupSync();

    if (!isPayload) {
        int2 src = giBorderSource(tileLocal, GI_VISIBILITY_PAYLOAD);
        giVisibility[tileOrigin + tileLocal] = gPayload[src.y][src.x];
    }
}
