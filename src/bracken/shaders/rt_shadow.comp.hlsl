// rt_shadow.comp.hlsl - ray-traced sun shadows over a software two-level BVH.
//
// There is no hardware ray tracing in SDL3 GPU, so this traverses a BVH the
// engine builds on the CPU (render/raytracing/bvh.hpp). Every design choice
// here exists to keep that affordable:
//
//  * NO PRIMARY RAYS. World position is reconstructed from the depth prepass,
//    so the only ray cast is the shadow ray itself. That is 1.44M rays saved
//    per frame at 1600x900 before anything else.
//  * ANY-HIT, NOT CLOSEST-HIT. A shadow ray only asks "is anything in the
//    way", so traversal returns on the first triangle it crosses instead of
//    sorting for the nearest. On occluded pixels this typically ends after a
//    handful of nodes.
//  * RAYS THAT CANNOT MATTER ARE NEVER CAST. A surface facing away from the
//    sun is already unlit, and sky pixels have no surface at all. Both are
//    rejected before traversal, which removes roughly half the dispatch.
//  * TWO LEVELS. The TLAS is walked in world space; only when an instance is
//    entered is the ray transformed into model space and its BLAS walked. The
//    scene's 6.2M drawn triangles are ~600k unique ones behind instancing.
//  * SHORT STACKS. Traversal stacks are sized to the deepest tree the builder
//    can produce, not to a comfortable round number: stack space is per-lane
//    registers, and register pressure caps occupancy, which caps latency
//    hiding, which is the whole game in a memory-bound traversal.
//
// Output is a single-channel visibility mask (1 = lit, 0 = shadowed), which
// the lighting pass multiplies in. Half-resolution by default; the composite
// upsamples with a depth-aware filter.

static const uint TLAS_STACK = 32;
static const uint BLAS_STACK = 32;
// Matches BvhBuildOptions::max_depth. A deeper tree cannot be built, so a
// deeper stack would only cost registers.
static const float RAY_EPSILON = 0.02;

struct BvhNode {
    float3 boundsMin;
    uint leftFirst; // interior: left child index (right is +1); leaf: first prim
    float3 boundsMax;
    uint count;     // 0 = interior
};

struct BvhTriangle {
    float3 v0;
    float pad0;
    float3 e1;
    uint primId;
    float3 e2;
    float pad2;
};

struct TlasInstance {
    float4x4 worldToModel;
    float3 boundsMin;
    uint blasNodeOffset;
    float3 boundsMax;
    uint blasTriangleOffset;
};

[[vk::binding(0, 2)]]
cbuffer RtShadowUniforms : register(b0, space2) {
    float4x4 invViewProj;   // clip -> world, for depth reconstruction
    float3 sunDirection;    // direction light TRAVELS (engine convention)
    float rayMaxDistance;   // ray length; shorter = cheaper and misses less
    float3 cameraPosition;
    float normalBias;       // push along the normal to stop self-shadowing
    uint2 outputSize;       // shadow mask resolution (may be half the screen)
    uint tlasNodeCount;     // 0 = no acceleration structure this frame
    float depthBias;
};

[[vk::binding(0, 0)]] Texture2D<float> sceneDepth : register(t0, space0);
[[vk::binding(1, 0)]] Texture2D<float> aoInput : register(t1, space0);
[[vk::binding(2, 0)]] StructuredBuffer<BvhNode> tlasNodes : register(t2, space0);
[[vk::binding(3, 0)]] StructuredBuffer<TlasInstance> tlasInstances : register(t3, space0);
[[vk::binding(4, 0)]] StructuredBuffer<BvhNode> blasNodes : register(t4, space0);
[[vk::binding(5, 0)]] StructuredBuffer<BvhTriangle> blasTriangles : register(t5, space0);

// RG8: R = the ambient occlusion read straight through from GTAO, G = sun
// visibility. Two terms in one texture because the lighting pass's fragment
// stage is at its 10-sampler ceiling and cannot take another binding.
[[vk::binding(0, 1)]] RWTexture2D<float2> shadowMask : register(u0, space1);

// Slab test. Returns the near hit distance, or a miss as tNear > tFar.
// Branchless min/max: a per-lane branch here would serialize the warp on the
// most-executed instruction in the whole shader.
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

// Möller-Trumbore against a triangle already stored as origin + two edges, so
// the two subtractions are not repeated per ray.
bool triangleHit(BvhTriangle tri, float3 origin, float3 dir, float tMax) {
    float3 pv = cross(dir, tri.e2);
    float det = dot(tri.e1, pv);
    // Two-sided: a shadow caster occludes regardless of winding, and culling
    // backfaces here would put holes in the shadows of single-sided art.
    if (abs(det) < 1e-8) {
        return false;
    }
    float invDet = 1.0 / det;
    float3 tv = origin - tri.v0;
    float u = dot(tv, pv) * invDet;
    if (u < 0.0 || u > 1.0) {
        return false;
    }
    float3 qv = cross(tv, tri.e1);
    float v = dot(dir, qv) * invDet;
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }
    float t = dot(tri.e2, qv) * invDet;
    return t > RAY_EPSILON && t < tMax;
}

// Walk one BLAS in model space. Returns on the first hit.
bool blasOccluded(uint nodeOffset, uint triangleOffset, float3 origin, float3 dir, float tMax) {
    float3 invDir = 1.0 / dir;

    uint stack[BLAS_STACK];
    uint sp = 0;
    stack[sp++] = nodeOffset;

    while (sp > 0) {
        BvhNode node = blasNodes[stack[--sp]];

        float tNear;
        if (!slabTest(node.boundsMin, node.boundsMax, origin, invDir, tMax, tNear)) {
            continue;
        }

        if (node.count == 0) {
            // Interior: push both children. No near-first ordering here — for
            // an ANY-hit query the first hit found is as good as the nearest,
            // so the sort would be pure overhead.
            if (sp + 2 <= BLAS_STACK) {
                stack[sp++] = nodeOffset + node.leftFirst;
                stack[sp++] = nodeOffset + node.leftFirst + 1;
            }
            continue;
        }

        // Leaf: primitives are contiguous (the builder writes them in leaf
        // order), so this is a linear scan of adjacent memory.
        for (uint i = 0; i < node.count; ++i) {
            if (triangleHit(blasTriangles[triangleOffset + node.leftFirst + i], origin, dir,
                            tMax)) {
                return true;
            }
        }
    }
    return false;
}

// Walk the TLAS in world space, descending into a BLAS only for instances the
// ray actually enters.
bool sceneOccluded(float3 origin, float3 dir, float tMax) {
    if (tlasNodeCount == 0) {
        return false;
    }
    float3 invDir = 1.0 / dir;

    uint stack[TLAS_STACK];
    uint sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        BvhNode node = tlasNodes[stack[--sp]];

        float tNear;
        if (!slabTest(node.boundsMin, node.boundsMax, origin, invDir, tMax, tNear)) {
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

            // Transform the RAY into model space rather than the model into
            // world space. One matrix-vector product per instance versus one
            // per vertex is the entire reason a two-level structure exists.
            float3 mOrigin = mul(inst.worldToModel, float4(origin, 1.0)).xyz;
            float3 mDir = mul(inst.worldToModel, float4(dir, 0.0)).xyz;

            // The transform may scale, which changes what tMax means in model
            // space. Normalizing and rescaling keeps the cutoff honest.
            float scale = length(mDir);
            if (scale < 1e-8) {
                continue;
            }
            if (blasOccluded(inst.blasNodeOffset, inst.blasTriangleOffset, mOrigin, mDir / scale,
                             tMax * scale)) {
                return true;
            }
        }
    }
    return false;
}

// World-space position at a depth texel, for normal reconstruction.
float3 worldAt(int2 c, uint2 size) {
    float d = sceneDepth.Load(int3(c, 0));
    float2 ndc = (float2(c) + 0.5) / float2(size) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 w = mul(invViewProj, float4(ndc, d, 1.0));
    return w.xyz / w.w;
}

// Geometric normal from neighbouring depth taps. Picks the closer neighbour on
// each axis so a depth discontinuity does not smear the normal across it,
// which would put shadow halos on every silhouette.
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
    if (tid.x >= outputSize.x || tid.y >= outputSize.y) {
        return;
    }
    int2 coord = int2(tid.xy);

    // Depth is sampled at the mask's resolution; when the mask is half-res
    // this is a 2x2 corner tap, which is what the depth-aware upsample in the
    // composite expects.
    uint2 depthSize;
    sceneDepth.GetDimensions(depthSize.x, depthSize.y);
    int2 depthCoord = int2((float2(coord) + 0.5) / float2(outputSize) * float2(depthSize));
    float depth = sceneDepth.Load(int3(depthCoord, 0));

    // Reversed-Z: 0 is the far plane, so depth == 0 is sky. Nothing to shade,
    // and casting a ray from the far plane is the single most expensive way to
    // learn that.
    if (depth <= 0.0) {
        shadowMask[coord] = float2(aoInput.Load(int3(depthCoord, 0)), 1.0);
        return;
    }

    // Normal reconstructed from depth rather than read from a G-buffer, for
    // the same reason GTAO does it: the engine has no normal target, and four
    // extra depth taps are cheaper than adding a full-res normal write.
    float3 normal = reconstructNormal(depthCoord, depthSize);
    float3 toLight = -sunDirection; // sunDirection is where light TRAVELS

    // A surface facing away from the sun is already unlit by N.L in the
    // lighting pass. Tracing it would spend a full traversal to confirm what
    // the dot product already said.
    float ndotl = dot(normal, toLight);
    if (ndotl <= 0.0) {
        shadowMask[coord] = float2(aoInput.Load(int3(depthCoord, 0)), 0.0);
        return;
    }

    // Reconstruct world position from depth.
    float2 ndc = (float2(coord) + 0.5) / float2(outputSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    float4 clip = float4(ndc, depth, 1.0);
    float4 world = mul(invViewProj, clip);
    float3 worldPos = world.xyz / world.w;

    // Offset along the normal, scaled by grazing angle. A constant bias either
    // acne-stripes flat surfaces or detaches contact shadows on steep ones;
    // scaling by 1/N.L spends the bias where the geometry actually needs it.
    float slope = clamp(1.0 - ndotl, 0.0, 1.0);
    float3 rayOrigin = worldPos + normal * (normalBias * (1.0 + slope * 4.0)) +
                       toLight * depthBias;

    float visibility = sceneOccluded(rayOrigin, toLight, rayMaxDistance) ? 0.0 : 1.0;
    shadowMask[coord] = float2(aoInput.Load(int3(depthCoord, 0)), visibility);
}
