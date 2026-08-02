// rt_reflection.comp.hlsl - ray-traced specular reflections over the same
// software two-level BVH as rt_shadow.comp.hlsl.
//
// Reflections are the expensive half of ray tracing, so the gating matters
// more than the tracing:
//
//  * QUARTER RESOLUTION by default. A reflection is a low-frequency signal on
//    everything except a mirror, so it survives being traced sparsely and
//    upsampled far better than a shadow does.
//  * ROUGHNESS GATE. Above `maxRoughness` the lobe is wide enough that the
//    prefiltered IBL probe the engine already has is indistinguishable from a
//    traced result, and vastly cheaper. Only smooth surfaces trace.
//  * CLOSEST hit, not any hit — unlike shadows, we have to shade what we find,
//    so traversal cannot stop early. This is precisely why reflections cost
//    several times what shadows do, and why they are a separate toggle.
//  * ONE BOUNCE. The hit is shaded with sun lighting plus a flat ambient, not
//    by recursing. A second bounce roughly squares the cost to change pixels
//    almost nobody looks at.
//  * MISSES FALL BACK TO THE SKY, matching the skybox the raster path draws,
//    so a ray that escapes the scene is not a black hole in the reflection.

#include "reflection_gate.hlsli"
#include "rt_sampling.hlsli"

// The BVH structs and the traversal walks are shared with rt_shadow.comp.hlsl
// — see rt_trace.hlsli for why it is included in two phases.
#define RT_TRACE_TYPES
#include "rt_trace.hlsli"

// Stays here: only reflections bind the attribute buffer, and hit SHADING is
// this pass's business, not the traversal's.
struct TriangleAttributes {
    float3 n0;
    float pad0;
    float3 n1;
    float pad1;
    float3 n2;
    float pad2;
    float4 baseColor;
};

[[vk::binding(0, 2)]]
cbuffer RtReflectionUniforms : register(b0, space2) {
    float4x4 invViewProj;
    float3 cameraPosition;
    float rayMaxDistance;
    float3 sunDirection;    // direction light TRAVELS
    float maxRoughness;     // the traced contribution reaches zero here
    float3 sunColor;
    float normalBias;
    float3 skyColor;
    float _pad0;
    uint2 outputSize;
    uint tlasNodeCount;
    float fadeStart;        // ...and starts fading out here (< maxRoughness)
    float4 waterBounds;     // minX, minZ, maxX, maxZ
    float4 waterSurface;    // height, roughness, displacement, normal strength
    float4 waterSimWindow;  // min XZ, inverse extent, edge fade width
    float4 waterFlowTime;   // flow XZ, speed, absolute time
    float4 waterDetail;     // detail strength, receiver mode, visibility bias, unused
    uint frameIndex;        // advances the GGX sample sequence
    float3 _pad1;
};

[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D<float> sceneDepth : register(t0, space0);
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState sceneDepthSampler : register(s0, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
Texture2D<float4> sceneNormal : register(t1, space0);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]]
SamplerState sceneNormalSampler : register(s1, space0);
[[vk::binding(2, 0)]] StructuredBuffer<BvhNodeGpu> tlasNodes : register(t2, space0);
[[vk::binding(3, 0)]] StructuredBuffer<TlasInstance> tlasInstances : register(t3, space0);
[[vk::binding(4, 0)]] StructuredBuffer<BvhNodeGpu> blasNodes : register(t4, space0);
[[vk::binding(5, 0)]] StructuredBuffer<BvhTriangle> blasTriangles : register(t5, space0);
[[vk::binding(6, 0)]] StructuredBuffer<TriangleAttributes> blasAttributes : register(t6, space0);

[[vk::binding(0, 1)]] RWTexture2D<float4> reflectionOut : register(u0, space1);
// Normalized hit distance (hit.t / rayMaxDistance), 1.0 on a miss or any
// early-out. R16F, traced resolution, written by every in-bounds thread so the
// target never carries a stale frame's geometry. NOTHING CONSUMES IT YET: it
// exists because R3's virtual-hit-point reprojection needs the reflected
// point's distance to reproject the *virtual* position rather than the surface
// carrying it, and the trace is the only place that number is free. Writing it
// now keeps that from being a change to the traversal later.
[[vk::binding(1, 1)]] RWTexture2D<float> reflectionHitDistance : register(u1, space1);

// The buffers and `tlasNodeCount` above are what the walks bind to.
#define RT_TRACE_TRAVERSAL
#include "rt_trace.hlsli"

// Must stay operation-identical to water.frag.hlsl. The analytic receiver
// uses this normal for the ray; the water forward pass uses it for Fresnel,
// so a mismatch would make reflected features slide over the waves.
float2 waterDetailWaveGradient(float2 p, float2 flow, float t) {
    float2 grad = 0.0;
    float2 d0 = normalize(flow + float2(0.0, 1e-3));
    float2 d0b = normalize(d0 + float2(0.22, 0.13));
    float2 d1 = normalize(d0 + float2(0.83, -0.41));
    float2 d2 = normalize(d0 + float2(-0.71, 0.52));
    float ph0a = dot(p, d0) * (6.2831 / 131.0) + t * 1.31;
    float ph0b = dot(p, d0b) * (6.2831 / 89.0) + t * 1.03;
    float ph1 = dot(p, d1) * (6.2831 / 57.0) + t * 2.17;
    float ph2 = dot(p, d2) * (6.2831 / 26.0) + t * 3.71;
    float envelope =
        0.6 + 0.4 * sin(dot(p, float2(-d0.y, d0.x)) * (6.2831 / 217.0) + t * 0.53);
    grad += d0 * cos(ph0a) * 0.34;
    grad += d0b * cos(ph0b) * 0.27;
    grad += d1 * cos(ph1) * 0.25 * envelope;
    grad += d2 * cos(ph2) * 0.14 * envelope;
    return grad;
}

bool waterReceiver(float2 ndc, float opaqueDepth, out float3 worldPos,
                   out float3 normal, out float roughness) {
    worldPos = 0.0;
    normal = float3(0.0, 1.0, 0.0);
    roughness = waterSurface.y;

    // Reversed-Z near-plane point gives the camera ray without requiring a
    // second inverse matrix. Water viewed from below is intentionally rejected.
    float4 nearH = mul(invViewProj, float4(ndc, 1.0, 1.0));
    float3 rayDir = normalize(nearH.xyz / nearH.w - cameraPosition);
    if (rayDir.y >= -1e-5) {
        return false;
    }

    float t = (waterSurface.x - cameraPosition.y) / rayDir.y;
    if (t <= 0.0) {
        return false;
    }

    // Two fixed refinements converge the flat-plane intersection onto the
    // simulated displacement without a variable loop or a water depth pass.
    float2 simUV = 0.0;
    float simFade = 0.0;
    [unroll]
    for (int iteration = 0; iteration < 2; ++iteration) {
        float2 xz = cameraPosition.xz + rayDir.xz * t;
        simUV = (xz - waterSimWindow.xy) * waterSimWindow.z;
        float fadeW = max(waterSimWindow.w, 1e-4);
        float2 edge = min(simUV, 1.0 - simUV) / fadeW;
        simFade = saturate(min(edge.x, edge.y));
        float h = sceneNormal.SampleLevel(sceneNormalSampler, saturate(simUV), 0).x;
        float displacedY = waterSurface.x + h * waterSurface.z * simFade;
        t = (displacedY - cameraPosition.y) / rayDir.y;
    }

    worldPos = cameraPosition + rayDir * t;
    if (worldPos.x < waterBounds.x || worldPos.z < waterBounds.y ||
        worldPos.x > waterBounds.z || worldPos.z > waterBounds.w) {
        return false;
    }

    // Reject water hidden by opaque geometry. The opaque prepass deliberately
    // remains untouched so the later water pass can still refract what lies
    // behind the surface.
    if (opaqueDepth > 0.0) {
        float4 opaqueH = mul(invViewProj, float4(ndc, opaqueDepth, 1.0));
        float3 opaquePos = opaqueH.xyz / opaqueH.w;
        float opaqueT = dot(opaquePos - cameraPosition, rayDir);
        if (t >= opaqueT - max(waterDetail.z, 0.0)) {
            return false;
        }
    }

    uint2 simSize;
    sceneNormal.GetDimensions(simSize.x, simSize.y);
    float2 texel = 1.0 / float2(max(simSize, uint2(1, 1)));
    float hl = sceneNormal.SampleLevel(sceneNormalSampler, saturate(simUV - float2(texel.x, 0)), 0).x;
    float hr = sceneNormal.SampleLevel(sceneNormalSampler, saturate(simUV + float2(texel.x, 0)), 0).x;
    float hd = sceneNormal.SampleLevel(sceneNormalSampler, saturate(simUV - float2(0, texel.y)), 0).x;
    float hu = sceneNormal.SampleLevel(sceneNormalSampler, saturate(simUV + float2(0, texel.y)), 0).x;
    float simExtent = 1.0 / max(waterSimWindow.z, 1e-6);
    float simTexelWorld = simExtent / max(float(simSize.x), 1.0);
    float2 simGrad = float2(hr - hl, hu - hd) * waterSurface.z /
                     max(2.0 * simTexelWorld, 1e-3);
    float2 flow = waterFlowTime.xy;
    float2 detail = waterDetailWaveGradient(
                        worldPos.xz - flow * waterFlowTime.z * waterFlowTime.w,
                        flow, waterFlowTime.w) * waterDetail.x;
    normal = normalize(float3(-(simGrad.x * waterSurface.w * simFade + detail.x), 1.0,
                              -(simGrad.y * waterSurface.w * simFade + detail.y)));
    return true;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= outputSize.x || tid.y >= outputSize.y) {
        return;
    }
    int2 coord = int2(tid.xy);

    float2 ndc = (float2(coord) + 0.5) / float2(outputSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    uint2 depthSize;
    sceneDepth.GetDimensions(depthSize.x, depthSize.y);
    int2 depthCoord = int2((float2(coord) + 0.5) / float2(outputSize) * float2(depthSize));
    float depth = sceneDepth.Load(int3(depthCoord, 0));

    float3 worldPos;
    float3 normal;
    float roughness;
    if (waterDetail.y > 0.5) {
        if (!waterReceiver(ndc, depth, worldPos, normal, roughness)) {
            reflectionOut[coord] = float4(0, 0, 0, 0);
            reflectionHitDistance[coord] = 1.0;
            return;
        }
    } else {
        // Reversed-Z: 0 is the far plane. Sky reflects nothing.
        if (depth <= 0.0) {
            reflectionOut[coord] = float4(0, 0, 0, 0);
            reflectionHitDistance[coord] = 1.0;
            return;
        }
        float4 gbuf = sceneNormal.Load(int3(depthCoord, 0));
        normal = octDecode(gbuf.xy);
        roughness = gbuf.z;
        float4 clip = float4(ndc, depth, 1.0);
        float4 world = mul(invViewProj, clip);
        worldPos = world.xyz / world.w;
    }

    // The gate that makes reflections affordable: a rough surface's lobe is so
    // wide that the prefiltered IBL probe is indistinguishable from a traced
    // result. `gate` is how much of this pixel's specular the trace supplies;
    // the forward pass scales its IBL lobe by exactly 1 - gate, so the two
    // hand over continuously instead of stepping (shaders/reflection_gate.hlsli).
    // alpha = 0 tells the full-resolution resolve to keep using the probe alone.
    float gate = rtReflectionWeight(roughness, fadeStart, maxRoughness);
    if (gate <= 0.0) {
        reflectionOut[coord] = float4(0, 0, 0, 0);
        reflectionHitDistance[coord] = 1.0;
        return;
    }

    float3 viewDir = normalize(worldPos - cameraPosition);

    // GGX IMPORTANCE SAMPLING, not a mirror ray. A single reflect() about the
    // surface normal is only correct for a perfect mirror; every rougher
    // surface has a LOBE, and collapsing it to its axis is why the roughness
    // gate had to be held down at 0.55 — anything blurrier than that looked
    // wrong, so it was handed to the IBL probe instead.
    //
    // Sampling the VISIBLE normal distribution (Heitz 2018) rather than the
    // full NDF matters at grazing angles, where half the NDF's samples point
    // into the surface and would have to be discarded.
    float3 rayDir;
    {
        const float alpha = max(roughness * roughness, 1e-4);
        float3 tangent, bitangent;
        buildBasis(normal, tangent, bitangent);
        // View vector pointing FROM the surface toward the camera.
        const float3 v = -viewDir;
        const float3 vTangent =
            float3(dot(v, tangent), dot(v, bitangent), dot(v, normal));
        const float2 xi = rtSample2D(uint2(coord), frameIndex);
        const float3 hTangent = sampleGGXVndf(vTangent, alpha, xi);
        const float3 h = hTangent.x * tangent + hTangent.y * bitangent + hTangent.z * normal;
        rayDir = reflect(viewDir, h);
        // A sampled half-vector can still put the ray under the horizon on a
        // grazing pixel. Fall back to the mirror direction rather than tracing
        // a ray that starts by entering the surface it came from.
        if (dot(rayDir, normal) <= 0.0) {
            rayDir = reflect(viewDir, normal);
        }
    }

    float3 rayOrigin = worldPos + normal * normalBias;

    Hit hit = sceneClosest(rayOrigin, rayDir, rayMaxDistance);

    // Normalized so the target stays format-independent and the consumer only
    // ever needs `rayMaxDistance` to recover metres. A miss is 1.0 rather than
    // 0.0 because "as far as the ray could see" is the useful answer for a
    // reprojection: it reads as the sky, which is at infinity.
    reflectionHitDistance[coord] = hit.valid ? (hit.t / rayMaxDistance) : 1.0;

    if (!hit.valid) {
        // Water already owns a high-quality SSR/IBL fallback. Leave alpha at
        // zero on a miss so that path remains authoritative instead of
        // replacing it with the RT pass's deliberately cheap flat sky.
        if (waterDetail.y > 0.5) {
            reflectionOut[coord] = float4(0, 0, 0, 0);
            return;
        }
        // Escaped the scene: return sky, matching what the raster path draws.
        // Returning black here is the classic ray-traced-reflection artifact —
        // dark voids wherever geometry is thin.
        reflectionOut[coord] = float4(skyColor, gate);
        return;
    }

    // Shade the hit: interpolated normal, base colour, one sun term plus a
    // flat ambient. Deliberately not a second bounce.
    TriangleAttributes attr = blasAttributes[hit.triIndex];
    float w = 1.0 - hit.bary.x - hit.bary.y;
    float3 hitNormalModel = normalize(attr.n0 * w + attr.n1 * hit.bary.x + attr.n2 * hit.bary.y);
    // Vertex attributes live in BLAS/model space. Lighting is world-space, so
    // transform by inverse-transpose(model) — transpose(worldToModel) — just
    // like the raster path. Using the model-space normal here made rotated
    // instances light from the wrong direction inside reflections.
    TlasInstance hitInstance = tlasInstances[hit.instanceIndex];
    float3 hitNormal = normalize(mul((float3x3)transpose(hitInstance.worldToModel), hitNormalModel));

    float3 toLight = -sunDirection;
    float ndotl = saturate(dot(hitNormal, toLight));
    float3 albedo = attr.baseColor.rgb;
    float3 shaded = albedo * (sunColor * ndotl + skyColor * 0.25);

    // Fade with distance so a reflection that runs out of ray fades into the
    // sky rather than ending at a hard edge.
    float fade = saturate(1.0 - hit.t / rayMaxDistance);
    if (waterDetail.y > 0.5) {
        // For water alpha is the hand-off to SSR/IBL. Fade the traced result's
        // authority, not its radiance, so the existing fallback fills misses
        // and the end of the trace range continuously.
        reflectionOut[coord] = float4(shaded, gate * fade);
        return;
    }
    shaded = lerp(skyColor, shaded, fade);

    // Store incoming radiance, not a material-weighted contribution. The
    // full-resolution resolve evaluates the RECEIVING pixel's BRDF after its
    // material/depth/normal-aware upsample; doing it here was what let one
    // sparse stone texel carry stone reflectance onto neighbouring grass.
    reflectionOut[coord] = float4(shaded, gate);
}
