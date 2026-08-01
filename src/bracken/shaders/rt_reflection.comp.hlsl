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

static const uint TLAS_STACK = 32;
static const uint BLAS_STACK = 32;
static const float RAY_EPSILON = 0.02;

struct BvhNode {
    float3 boundsMin;
    uint leftFirst;
    float3 boundsMax;
    uint count;
};

struct BvhTriangle {
    float3 v0;
    float pad0;
    float3 e1;
    uint primId;
    float3 e2;
    float pad2;
};

struct TriangleAttributes {
    float3 n0;
    float pad0;
    float3 n1;
    float pad1;
    float3 n2;
    float pad2;
    float4 baseColor;
};

struct TlasInstance {
    float4x4 worldToModel;
    float3 boundsMin;
    uint blasNodeOffset;
    float3 boundsMax;
    uint blasTriangleOffset;
};

[[vk::binding(0, 2)]]
cbuffer RtReflectionUniforms : register(b0, space2) {
    float4x4 invViewProj;
    float3 cameraPosition;
    float rayMaxDistance;
    float3 sunDirection;    // direction light TRAVELS
    float maxRoughness;     // above this, the IBL probe is used instead
    float3 sunColor;
    float normalBias;
    float3 skyColor;
    float intensity;
    uint2 outputSize;
    uint tlasNodeCount;
    float pad;
};

[[vk::binding(0, 0)]] Texture2D<float> sceneDepth : register(t0, space0);
[[vk::binding(1, 0)]] Texture2D<float4> sceneNormal : register(t1, space0);
[[vk::binding(2, 0)]] StructuredBuffer<BvhNode> tlasNodes : register(t2, space0);
[[vk::binding(3, 0)]] StructuredBuffer<TlasInstance> tlasInstances : register(t3, space0);
[[vk::binding(4, 0)]] StructuredBuffer<BvhNode> blasNodes : register(t4, space0);
[[vk::binding(5, 0)]] StructuredBuffer<BvhTriangle> blasTriangles : register(t5, space0);
[[vk::binding(6, 0)]] StructuredBuffer<TriangleAttributes> blasAttributes : register(t6, space0);

[[vk::binding(0, 1)]] RWTexture2D<float4> reflectionOut : register(u0, space1);

struct Hit {
    float t;
    // NOTE: not named `triangle` — that is a reserved HLSL keyword (a
    // geometry-shader primitive modifier) and the error it produces
    // ("modifiers must appear before type") does not mention the name.
    uint triIndex;   // absolute index into blasTriangles / blasAttributes
    float2 bary;     // u, v
    bool valid;
};

bool slabTest(float3 boundsMin, float3 boundsMax, float3 origin, float3 invDir, float tMax,
              out float tNear) {
    float3 t0 = (boundsMin - origin) * invDir;
    float3 t1 = (boundsMax - origin) * invDir;
    float3 tsmall = min(t0, t1);
    float3 tbig = max(t0, t1);
    tNear = max(max(tsmall.x, tsmall.y), max(tsmall.z, 0.0));
    float tFar = min(min(tbig.x, tbig.y), min(tbig.z, tMax));
    return tNear <= tFar;
}

// Closest hit within one BLAS. `hit` carries the running best so a later
// instance cannot overwrite a nearer one.
void blasClosest(uint nodeOffset, uint triangleOffset, float3 origin, float3 dir, float scale,
                 inout Hit hit) {
    float3 invDir = 1.0 / dir;

    uint stack[BLAS_STACK];
    uint sp = 0;
    stack[sp++] = nodeOffset;

    while (sp > 0) {
        BvhNode node = blasNodes[stack[--sp]];

        float tNear;
        // Cull against the CURRENT best distance, so as the search tightens it
        // rejects more of the tree. This is the closest-hit equivalent of the
        // shadow ray's early-out and the main reason it is not far worse.
        if (!slabTest(node.boundsMin, node.boundsMax, origin, invDir, hit.t * scale, tNear)) {
            continue;
        }

        if (node.count == 0) {
            if (sp + 2 <= BLAS_STACK) {
                stack[sp++] = nodeOffset + node.leftFirst;
                stack[sp++] = nodeOffset + node.leftFirst + 1;
            }
            continue;
        }

        for (uint i = 0; i < node.count; ++i) {
            uint index = triangleOffset + node.leftFirst + i;
            BvhTriangle tri = blasTriangles[index];

            float3 pv = cross(dir, tri.e2);
            float det = dot(tri.e1, pv);
            if (abs(det) < 1e-8) {
                continue;
            }
            float invDet = 1.0 / det;
            float3 tv = origin - tri.v0;
            float u = dot(tv, pv) * invDet;
            if (u < 0.0 || u > 1.0) {
                continue;
            }
            float3 qv = cross(tv, tri.e1);
            float v = dot(dir, qv) * invDet;
            if (v < 0.0 || u + v > 1.0) {
                continue;
            }
            float t = dot(tri.e2, qv) * invDet;
            // Model-space t converted back to world scale before comparing.
            float worldT = t / scale;
            if (t > RAY_EPSILON && worldT < hit.t) {
                hit.t = worldT;
                hit.triIndex = index;
                hit.bary = float2(u, v);
                hit.valid = true;
            }
        }
    }
}

Hit sceneClosest(float3 origin, float3 dir, float tMax) {
    Hit hit;
    hit.t = tMax;
    hit.triIndex = 0;
    hit.bary = float2(0, 0);
    hit.valid = false;

    if (tlasNodeCount == 0) {
        return hit;
    }
    float3 invDir = 1.0 / dir;

    uint stack[TLAS_STACK];
    uint sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        BvhNode node = tlasNodes[stack[--sp]];

        float tNear;
        if (!slabTest(node.boundsMin, node.boundsMax, origin, invDir, hit.t, tNear)) {
            continue;
        }

        if (node.count == 0) {
            if (sp + 2 <= TLAS_STACK) {
                stack[sp++] = node.leftFirst;
                stack[sp++] = node.leftFirst + 1;
            }
            continue;
        }

        for (uint i = 0; i < node.count; ++i) {
            TlasInstance inst = tlasInstances[node.leftFirst + i];
            float3 mOrigin = mul(inst.worldToModel, float4(origin, 1.0)).xyz;
            float3 mDir = mul(inst.worldToModel, float4(dir, 0.0)).xyz;
            float scale = length(mDir);
            if (scale < 1e-8) {
                continue;
            }
            blasClosest(inst.blasNodeOffset, inst.blasTriangleOffset, mOrigin, mDir / scale, scale,
                        hit);
        }
    }
    return hit;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= outputSize.x || tid.y >= outputSize.y) {
        return;
    }
    int2 coord = int2(tid.xy);

    uint2 depthSize;
    sceneDepth.GetDimensions(depthSize.x, depthSize.y);
    int2 depthCoord = int2((float2(coord) + 0.5) / float2(outputSize) * float2(depthSize));
    float depth = sceneDepth.Load(int3(depthCoord, 0));

    // Reversed-Z: 0 is the far plane. Sky reflects nothing.
    if (depth <= 0.0) {
        reflectionOut[coord] = float4(0, 0, 0, 0);
        return;
    }

    float4 normalSample = sceneNormal.Load(int3(depthCoord, 0));
    float3 normal = normalize(normalSample.xyz * 2.0 - 1.0);
    float roughness = normalSample.w;

    // The gate that makes reflections affordable: a rough surface's lobe is so
    // wide that the prefiltered IBL probe is indistinguishable from a traced
    // result. alpha=0 tells the composite to keep using the probe.
    if (roughness > maxRoughness) {
        reflectionOut[coord] = float4(0, 0, 0, 0);
        return;
    }

    float2 ndc = (float2(coord) + 0.5) / float2(outputSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 clip = float4(ndc, depth, 1.0);
    float4 world = mul(invViewProj, clip);
    float3 worldPos = world.xyz / world.w;

    float3 viewDir = normalize(worldPos - cameraPosition);
    float3 rayDir = reflect(viewDir, normal);
    float3 rayOrigin = worldPos + normal * normalBias;

    Hit hit = sceneClosest(rayOrigin, rayDir, rayMaxDistance);

    if (!hit.valid) {
        // Escaped the scene: return sky, matching what the raster path draws.
        // Returning black here is the classic ray-traced-reflection artifact —
        // dark voids wherever geometry is thin.
        reflectionOut[coord] = float4(skyColor * intensity, 1.0);
        return;
    }

    // Shade the hit: interpolated normal, base colour, one sun term plus a
    // flat ambient. Deliberately not a second bounce.
    TriangleAttributes attr = blasAttributes[hit.triIndex];
    float w = 1.0 - hit.bary.x - hit.bary.y;
    float3 hitNormal = normalize(attr.n0 * w + attr.n1 * hit.bary.x + attr.n2 * hit.bary.y);

    float3 toLight = -sunDirection;
    float ndotl = saturate(dot(hitNormal, toLight));
    float3 albedo = attr.baseColor.rgb;
    float3 shaded = albedo * (sunColor * ndotl + skyColor * 0.25);

    // Fade with distance so a reflection that runs out of ray fades into the
    // sky rather than ending at a hard edge.
    float fade = saturate(1.0 - hit.t / rayMaxDistance);
    shaded = lerp(skyColor, shaded, fade);

    reflectionOut[coord] = float4(shaded * intensity, 1.0);
}
