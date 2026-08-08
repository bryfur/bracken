// rt_trace.hlsli - the software two-level BVH: structs, slab test, and the
// any-hit / closest-hit walks shared by every ray-consuming compute pass.
//
// SDL3 GPU has no hardware ray tracing, so this is the engine's own traversal
// over buffers built on the CPU (render/raytracing/bvh.hpp). It used to exist
// twice, copy-pasted into rt_shadow.comp.hlsl and rt_reflection.comp.hlsl, and
// the copies were already drifting: only one of them had the near-child-first
// ordering. One copy, two entry points.
//
// NODE LAYOUT. The GPU node is 64 bytes and holds BOTH children (Aila-Laine):
// two AABBs, two links, one fetch per binary decision. The CPU BUILD format is
// still the 32-byte one-AABB `BvhNode` in bvh.hpp; the conversion happens on
// the upload side (render/raytracing/bvh_gpu.hpp, `convert_to_gpu_nodes`) and
// nothing here ever sees a 32-byte node. Only interior nodes become nodes: a
// leaf lives in its parent's side (`count > 0`, `child` = first primitive) and
// so is intersected inline and never pushed. Everything below indexes the
// CONVERTED array — `TlasInstance::blasNodeOffset` included.
//
// TWO-PHASE INCLUDE, because the dependency runs both ways. The including
// shader needs `BvhNode` and friends to DECLARE its storage buffers, and the
// traversal functions need those buffers (and the `tlasNodeCount` uniform) to
// already exist. So the file is included twice, selecting a phase each time:
//
//     #define RT_TRACE_TYPES
//     #include "rt_trace.hlsli"      // structs + pure helpers
//     ... the shader's own cbuffer and binding block ...
//     #define RT_TRACE_TRAVERSAL
//     #include "rt_trace.hlsli"      // the walks
//
// Phase 2 REQUIRES the including shader to have declared, at whatever bindings
// its own layout dictates:
//
//     StructuredBuffer<BvhNodeGpu>   tlasNodes;
//     StructuredBuffer<TlasInstance> tlasInstances;
//     StructuredBuffer<BvhNodeGpu>   blasNodes;
//     StructuredBuffer<BvhTriangle>  blasTriangles;
//     uint                           tlasNodeCount;   // cbuffer member
//
// And, ONLY for a shader that additionally defines RT_TRACE_CAPSULES before the
// traversal phase (the four any-hit passes — see capsulesOccluded):
//
//     StructuredBuffer<ShadowCapsule> shadowCapsules;      // or RWStructuredBuffer
//     uint                            shadowCapsuleCount;  // cbuffer member
//
// Bindings deliberately stay in the entry shaders (unlike
// cluster_lighting.hlsli, which owns its slots): the two RT passes have
// different sampled-texture counts and reflections bind a sixth buffer of
// per-triangle attributes that shadows must not pay for.

#if !defined(RT_TRACE_TYPES) && !defined(RT_TRACE_TRAVERSAL)
#error "Define RT_TRACE_TYPES or RT_TRACE_TRAVERSAL before including rt_trace.hlsli"
#endif

// ---------------------------------------------------------------------------
// Phase 1 — types and binding-free helpers.
// ---------------------------------------------------------------------------
#if defined(RT_TRACE_TYPES) && !defined(BRACKEN_RT_TRACE_TYPES_INCLUDED)
#define BRACKEN_RT_TRACE_TYPES_INCLUDED

// STACK DEPTH, derived rather than guessed.
//
// BvhBuildOptions::max_depth is 64 and the builder forces a leaf at
// `depth >= max_depth` with the root at depth 0, so the deepest INTERIOR node
// of a built 32-byte tree sits at depth 63. A converted node exists per
// interior node and keeps its depth, so a root-to-deepest path visits at most
// 64 converted nodes. Each visit defers at most ONE side (the walk descends
// into the other immediately, and leaves are intersected inline and never
// pushed), and the node at depth 63 has only leaves under it, so at most 63
// entries can be live: 64 is a proof, not a hope.
//
// The old walk's 32 was neither — it pushed BOTH children per interior visit
// and, when that did not fit, silently pushed neither, dropping whole subtrees
// from the answer. Halving the pushes is what makes a provable bound
// affordable at all.
static const uint TLAS_STACK = 64;
static const uint BLAS_STACK = 64;
static const float RAY_EPSILON = 0.02;

// 64-byte traversal node: both children's boxes and links, so one fetch
// resolves a binary decision. Mirrors BvhNodeGpu in
// render/raytracing/bvh_gpu.hpp byte for byte. Per side, count == 0 means the
// side is an interior node and `child` indexes its own 64-byte node; count > 0
// means the side is a leaf and `child` is its first primitive.
struct BvhNodeGpu {
    float3 lmin;
    uint lchild;
    float3 lmax;
    uint lcount;
    float3 rmin;
    uint rchild;
    float3 rmax;
    uint rcount;
};

struct BvhTriangle {
    float3 v0;
    /// Alpha-cutout coverage in [0, 1]; 1 = solid. Was `pad0`. Mirrors
    /// `BvhTriangle::opacity` in render/raytracing/bvh.hpp — this struct is
    /// uploaded by raw memcpy, so the two must agree bit for bit.
    float opacity;
    float3 e1;
    uint primId;
    float3 e2;
    float pad2;
};

// SKINNED FLAG: the high bit of blasNodeOffset. Mirrors k_tlas_skinned_flag in
// render/raytracing/bvh.hpp — the CPU struct below is uploaded by raw memcpy,
// so the two must agree bit for bit.
//
// A skinned instance's BLAS holds the model's BIND POSE. The closest-hit walks
// keep it (a reflection needs something shadeable where the character is, and a
// probe ray cannot see the pose error); the ANY-HIT walks skip it, because
// their result replaces the cascade shadow and a bind-pose statue dragged
// around the world is exactly the artefact this exists to remove. The animated
// occluder those walks use instead is the frame's ShadowCapsule set — see
// RT_TRACE_CAPSULES below and docs/rt_lighting_plan.md §5.8.
//
// EVERY READ MASKS. tlasNodeOffset() is the only legal way to get the index.
static const uint TLAS_SKINNED_FLAG = 0x80000000u;
static const uint TLAS_NODE_OFFSET_MASK = 0x7FFFFFFFu;

struct TlasInstance {
    float4x4 worldToModel;
    float3 boundsMin;
    // Where this instance's BLAS starts in the CONVERTED (64-byte) node array,
    // OR'd with TLAS_SKINNED_FLAG for a skinned instance. Its root is that
    // offset + 0; node links inside a BLAS are relative to it.
    uint blasNodeOffset;
    float3 boundsMax;
    uint blasTriangleOffset;
};

uint tlasNodeOffset(TlasInstance inst) {
    return inst.blasNodeOffset & TLAS_NODE_OFFSET_MASK;
}

bool tlasInstanceIsSkinned(TlasInstance inst) {
    return (inst.blasNodeOffset & TLAS_SKINNED_FLAG) != 0u;
}

// ---------------------------------------------------------------------------
// Analytic capsule occluders (docs/rt_lighting_plan.md §5.8).
//
// The CPU SOURCE OF TRUTH for everything below is
// render/raytracing/capsule_occluder.hpp, which is unit-tested over the graze,
// cap, parallel, inside, degenerate and t-window cases this file can only be
// eyeballed for. The two are written statement for statement; edit one, edit
// the other.
// ---------------------------------------------------------------------------

// Mirrors bracken::render::rt::ShadowCapsule (contracts/shadow_capsule.hpp),
// 32 bytes / two float4 registers. The radius rides in p0's w because two
// consecutive float3s do not pack.
struct ShadowCapsule {
    float3 p0;
    float radius;
    float3 p1;
    float pad0;
};

// See k_capsule_parallel_epsilon.
static const float CAPSULE_PARALLEL_EPSILON = 1e-8;

struct Hit {
    float t;
    // NOTE: not named `triangle` — that is a reserved HLSL keyword (a
    // geometry-shader primitive modifier) and the error it produces
    // ("modifiers must appear before type") does not mention the name.
    uint triIndex;   // absolute index into blasTriangles / blasAttributes
    uint instanceIndex; // TLAS instance whose transform owns triIndex
    float2 bary;     // u, v
    bool valid;
};

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

/// STOCHASTIC ALPHA TEST for a cutout triangle, the any-hit half of the opacity
/// bake in `internal/model_material_proxy.hpp`.
///
/// `tri.opacity` is the fraction of the triangle that survives the material's
/// alpha test, measured at BLAS build over stratified barycentric taps. Here it
/// is a probability: the triangle occludes with probability `opacity`, so a leaf
/// card that is 40% leaf blocks 40% of the rays that hit it and the shadow it
/// casts has the right DENSITY without the trace ever sampling a texture.
///
/// Why stochastic rather than a threshold. A hard `opacity > 0.5` makes every
/// leaf on a card identical — a canopy either casts a solid shadow or none —
/// and no amount of denoising recovers the dappling, because there is no
/// variance to average. Sampling it puts the dappling in the estimator, where
/// the shadow denoiser that is already running turns it into soft light.
///
/// `seed` is hashed with the triangle's own id so the decision is per triangle
/// per ray, and it is passed BY VALUE all the way down rather than carried as
/// `inout` state. That is what keeps the A-SVGF gradient replay exact: a replay
/// that re-derives the same seed makes bit-identical alpha decisions, and one
/// that did not would report a full gradient on every cutout pixel forever.
bool cutoutPasses(float opacity, uint primId, uint seed) {
    if (opacity >= 1.0) {
        return true;
    }
    if (opacity <= 0.0) {
        return false;
    }
    uint h = seed ^ (primId * 2654435761u);
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return (float(h) * (1.0 / 4294967296.0)) < opacity;
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

// Does the ray hit this sphere anywhere in [tMin, tMax]? `dd` is dot(dir, dir),
// carried in so a non-unit model-space direction stays correct.
bool sphereOccluded(float3 origin, float3 dir, float dd, float3 center, float radiusSq,
                    float tMin, float tMax) {
    float3 oc = origin - center;
    float b = dot(dir, oc);
    float c = dot(oc, oc) - radiusSq;
    float disc = b * b - dd * c;
    if (disc < 0.0 || dd <= 0.0) {
        return false;
    }
    float sq = sqrt(disc);
    // Both roots: an origin between them is a ray starting inside the sphere,
    // and the far root is where it leaves.
    float t0 = (-b - sq) / dd;
    float t1 = (-b + sq) / dd;
    return (t0 >= tMin && t0 <= tMax) || (t1 >= tMin && t1 <= tMax);
}

// Does the ray hit this capsule anywhere in [tMin, tMax]? Three parts, because
// a capsule is three surfaces welded together:
//
//  1. INSIDE-REJECTION. An origin within the capsule is not occluded by it.
//     That is the contract, not a numerical dodge: a capsule is a PROXY for a
//     body, and the points inside it are that body's own surface. Occluding
//     them paints every character's own skin black — a cylinder cannot fit a
//     box cross section, so a chest pixel genuinely does sit inside its own
//     torso proxy. Those pixels are fine anyway: a surface facing away from the
//     light is already unlit by N.L before any ray is cast, and one facing the
//     light is not occluded by the body behind it. The rejection is PER
//     CAPSULE, so an arm still shadows a chest that lies inside the torso.
//  2. THE INFINITE CYLINDER as a quadratic, with BOTH roots tested and each
//     accepted only when its axial coordinate lands inside the segment. The far
//     root matters for a ray that starts inside the cylinder but past a cap —
//     straight above a head.
//  3. THE TWO CAP SPHERES, always, not selected by which side a root fell on.
//     That buys every degenerate case: p0 == p1 (the axis vanishes and the
//     answer is a pure sphere test) and a ray parallel to the axis (the
//     quadratic degenerates, and a parallel ray can only enter through a cap).
//
// Radius <= 0 occludes nothing, which is the useful reading of a degenerate
// emit inside a per-item hot loop.
bool capsuleOccluded(float3 origin, float3 dir, float tMin, float tMax, ShadowCapsule capsule) {
    float radiusSq = capsule.radius * capsule.radius;
    if (radiusSq <= 0.0) {
        return false;
    }
    float3 ba = capsule.p1 - capsule.p0;
    float3 oa = origin - capsule.p0;
    float baba = dot(ba, ba);
    float dd = dot(dir, dir);

    // 1. inside-rejection, via the clamped axial parameter.
    float h = baba > 0.0 ? saturate(dot(oa, ba) / baba) : 0.0;
    float3 radial = oa - ba * h;
    if (dot(radial, radial) <= radiusSq) {
        return false;
    }

    // 2. infinite cylinder, clamped to the segment. a t^2 + 2 b t + c = 0, all
    // three scaled by baba so the axial coordinate comes out in the same units
    // and no division by |ba| is needed.
    float bard = dot(ba, dir);
    float baoa = dot(ba, oa);
    float rdoa = dot(dir, oa);
    float oaoa = dot(oa, oa);
    float a = baba * dd - bard * bard;
    float b = baba * rdoa - baoa * bard;
    float c = baba * oaoa - baoa * baoa - radiusSq * baba;
    if (a > CAPSULE_PARALLEL_EPSILON) {
        float disc = b * b - a * c;
        if (disc >= 0.0) {
            float sq = sqrt(disc);
            float t0 = (-b - sq) / a;
            float y0 = baoa + t0 * bard;
            if (y0 >= 0.0 && y0 <= baba && t0 >= tMin && t0 <= tMax) {
                return true;
            }
            float t1 = (-b + sq) / a;
            float y1 = baoa + t1 * bard;
            if (y1 >= 0.0 && y1 <= baba && t1 >= tMin && t1 <= tMax) {
                return true;
            }
        }
    }

    // 3. the caps.
    return sphereOccluded(origin, dir, dd, capsule.p0, radiusSq, tMin, tMax) ||
           sphereOccluded(origin, dir, dd, capsule.p1, radiusSq, tMin, tMax);
}

// Inverse of octEncode in prepass_gbuffer.frag.hlsl.
float3 octDecode(float2 e) {
    e = e * 2.0 - 1.0;
    float3 n = float3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0, float2(-t, -t), float2(t, t));
    return normalize(n);
}

#endif // RT_TRACE_TYPES

// ---------------------------------------------------------------------------
// Phase 2 — the walks, over the including shader's buffers.
// ---------------------------------------------------------------------------
#if defined(RT_TRACE_TRAVERSAL) && !defined(BRACKEN_RT_TRACE_TRAVERSAL_INCLUDED)
#define BRACKEN_RT_TRACE_TRAVERSAL_INCLUDED

#ifndef BRACKEN_RT_TRACE_TYPES_INCLUDED
#error "rt_trace.hlsli: include the RT_TRACE_TYPES phase before RT_TRACE_TRAVERSAL"
#endif

// Walk one BLAS in model space. Returns on the first hit.
//
// UNORDERED, deliberately: for an ANY-hit query the first hit found is as good
// as the nearest, so sorting the two children by distance would be pure
// overhead. Both boxes are in hand either way now — the layout gives the
// ordering away for free — but taking it would still cost the compare and the
// swap, and buy nothing a boolean answer can see.
bool blasOccluded(uint nodeOffset, uint triangleOffset, float3 origin, float3 dir, float tMax,
                  uint seed) {
    float3 invDir = 1.0 / dir;

    uint stack[BLAS_STACK];
    uint sp = 0;
    uint nodeIndex = 0; // relative to nodeOffset; the root is always 0

    [loop] while (true) {
        BvhNodeGpu node = blasNodes[nodeOffset + nodeIndex];

        // One fetch, both decisions.
        float tL;
        float tR;
        bool goLeft = slabTest(node.lmin, node.lmax, origin, invDir, tMax, tL);
        bool goRight = slabTest(node.rmin, node.rmax, origin, invDir, tMax, tR);

        // Leaves live in the parent, so they are intersected here and never
        // pushed. Primitives are contiguous (the builder writes them in leaf
        // order), so this is a linear scan of adjacent memory.
        if (goLeft && node.lcount != 0) {
            for (uint i = 0; i < node.lcount; ++i) {
                BvhTriangle t = blasTriangles[triangleOffset + node.lchild + i];
                if (triangleHit(t, origin, dir, tMax) &&
                    cutoutPasses(t.opacity, t.primId, seed)) {
                    return true;
                }
            }
            goLeft = false;
        }
        if (goRight && node.rcount != 0) {
            for (uint i = 0; i < node.rcount; ++i) {
                BvhTriangle t = blasTriangles[triangleOffset + node.rchild + i];
                if (triangleHit(t, origin, dir, tMax) &&
                    cutoutPasses(t.opacity, t.primId, seed)) {
                    return true;
                }
            }
            goRight = false;
        }

        // What is left is interior descents.
        if (goLeft) {
            if (goRight && sp < BLAS_STACK) {
                stack[sp++] = node.rchild;
            }
            nodeIndex = node.lchild;
        } else if (goRight) {
            nodeIndex = node.rchild;
        } else if (sp > 0) {
            nodeIndex = stack[--sp];
        } else {
            break;
        }
    }
    return false;
}

// One TLAS leaf's instance window, as an any-hit query. Factored out because
// with both children in the node a leaf can now appear on either side, and
// this loop is the one thing that would otherwise be written twice.
bool instancesOccluded(uint firstInstance, uint count, float3 origin, float3 dir, float tMax,
                       uint seed) {
    for (uint i = 0; i < count; ++i) {
        TlasInstance inst = tlasInstances[firstInstance + i];

        // SKINNED INSTANCES DO NOT OCCLUDE HERE. Their BLAS is the bind pose,
        // and this walk's answer REPLACES the cascade shadow, so including them
        // means an animated character casts a rigid statue's silhouette that
        // never matches its pose. The animated stand-in is the ShadowCapsule
        // set, tested once per ray in sceneOccluded before this walk begins.
        // The closest-hit walk below is deliberately NOT changed: a reflection
        // or probe ray still finds the character, bind pose and all.
        if (tlasInstanceIsSkinned(inst)) {
            continue;
        }

        // Transform the RAY into model space rather than the model into world
        // space. One matrix-vector product per instance versus one per vertex
        // is the entire reason a two-level structure exists.
        float3 mOrigin = mul(inst.worldToModel, float4(origin, 1.0)).xyz;
        float3 mDir = mul(inst.worldToModel, float4(dir, 0.0)).xyz;

        // The transform may scale, which changes what tMax means in model
        // space. Normalizing and rescaling keeps the cutoff honest.
        float scale = length(mDir);
        if (scale < 1e-8) {
            continue;
        }
        // The instance ordinal is mixed in so two copies of the same tree do
        // not make identical cutout decisions for the same ray — `primId` is a
        // per-MODEL ordinal, so without this a ray through two instances of one
        // asset would see perfectly correlated leaves.
        if (blasOccluded(tlasNodeOffset(inst), inst.blasTriangleOffset, mOrigin, mDir / scale,
                         tMax * scale, seed ^ ((firstInstance + i) * 0x9E3779B9u))) {
            return true;
        }
    }
    return false;
}

// The frame's analytic capsule occluders, as one any-hit query.
//
// GATED, not unconditional: only the four passes that ask a VISIBILITY question
// bind the capsule buffer (rt_shadow, rt_local_shadow and their two R5b
// gradient replays). The closest-hit consumers of this file — rt_reflection,
// gi_probe_trace, gi_screen_probe_trace — never call sceneOccluded and must not
// be made to declare a buffer they would never read, because an unread
// declaration changes the reflected resource layout (see the README's note on
// DXC's combinedImageSampler merging). A shader defines RT_TRACE_CAPSULES
// before the traversal phase to opt in; without it this compiles to a constant
// false and the traversal is byte-identical to its pre-capsule form.
//
// Phase 2 additionally REQUIRES, from an opting-in shader:
//     StructuredBuffer<ShadowCapsule> shadowCapsules;   (or RWStructuredBuffer)
//     uint                            shadowCapsuleCount;  // cbuffer member
//
// A LINEAR loop with an early-out, deliberately. The set is ~a dozen capsules
// per character and 256 at the frame budget, which is far below where any
// acceleration structure pays back the pointer chasing it costs — and it runs
// BEFORE the TLAS walk precisely because a hit here skips that walk entirely,
// which is the common case for the pixels in a character's own shadow.
#ifdef RT_TRACE_CAPSULES
bool capsulesOccluded(float3 origin, float3 dir, float tMin, float tMax) {
    [loop] for (uint i = 0; i < shadowCapsuleCount; ++i) {
        if (capsuleOccluded(origin, dir, tMin, tMax, shadowCapsules[i])) {
            return true;
        }
    }
    return false;
}
#else
bool capsulesOccluded(float3 origin, float3 dir, float tMin, float tMax) {
    return false;
}
#endif

// Walk the TLAS in world space, descending into a BLAS only for instances the
// ray actually enters.
bool sceneOccluded(float3 origin, float3 dir, float tMax, uint seed) {
    // The analytic occluders first: they answer for the animated characters the
    // TLAS deliberately no longer holds for this query, and a hit here costs a
    // dozen quadratics instead of a tree descent.
    if (capsulesOccluded(origin, dir, RAY_EPSILON, tMax)) {
        return true;
    }
    if (tlasNodeCount == 0) {
        return false;
    }
    float3 invDir = 1.0 / dir;

    uint stack[TLAS_STACK];
    uint sp = 0;
    uint nodeIndex = 0;

    [loop] while (true) {
        BvhNodeGpu node = tlasNodes[nodeIndex];

        float tL;
        float tR;
        bool goLeft = slabTest(node.lmin, node.lmax, origin, invDir, tMax, tL);
        bool goRight = slabTest(node.rmin, node.rmax, origin, invDir, tMax, tR);

        // Still unordered, same reason as blasOccluded.
        if (goLeft && node.lcount != 0) {
            if (instancesOccluded(node.lchild, node.lcount, origin, dir, tMax, seed)) {
                return true;
            }
            goLeft = false;
        }
        if (goRight && node.rcount != 0) {
            if (instancesOccluded(node.rchild, node.rcount, origin, dir, tMax, seed)) {
                return true;
            }
            goRight = false;
        }

        if (goLeft) {
            if (goRight && sp < TLAS_STACK) {
                stack[sp++] = node.rchild;
            }
            nodeIndex = node.lchild;
        } else if (goRight) {
            nodeIndex = node.rchild;
        } else if (sp > 0) {
            nodeIndex = stack[--sp];
        } else {
            break;
        }
    }
    return false;
}

// One BLAS leaf's triangle window against the running best.
void blasLeafClosest(uint firstTriangle, uint count, float3 origin, float3 dir, float scale,
                     inout Hit hit) {
    for (uint i = 0; i < count; ++i) {
        uint index = firstTriangle + i;
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

// Closest hit within one BLAS. `hit` carries the running best so a later
// instance cannot overwrite a nearer one.
void blasClosest(uint nodeOffset, uint triangleOffset, float3 origin, float3 dir, float scale,
                 inout Hit hit) {
    float3 invDir = 1.0 / dir;

    uint stack[BLAS_STACK];
    uint sp = 0;
    uint nodeIndex = 0;

    [loop] while (true) {
        BvhNodeGpu node = blasNodes[nodeOffset + nodeIndex];

        // Cull against the CURRENT best distance, so as the search tightens it
        // rejects more of the tree. This is the closest-hit equivalent of the
        // shadow ray's early-out and the main reason it is not far worse.
        //
        // NEAR CHILD FIRST, and only children the ray actually enters.
        //
        // A closest-hit walk only gets to cull anything because hit.t
        // tightens; descending the FAR subtree first means it tightens late
        // and the cull rejects almost nothing. Ordering is therefore not a
        // micro-optimization here, it is what makes the early-out real. (The
        // shadow walk is any-hit, where the first hit found is as good as the
        // nearest, so it deliberately does not pay for this.)
        //
        // The 64-byte node is what makes it nearly free: both boxes arrive in
        // the same fetch, so the two extra child fetches this used to issue
        // just to decide an order are gone. Nothing is pushed before it is
        // tested, and a node popped later is not re-tested against its own box
        // — its CHILDREN are, against a hit.t that can only have tightened
        // meanwhile, which is the stricter test anyway.
        //
        // HONESTLY: the ordering measured ZERO on the playground village. The
        // reflection pass there is bound by wave divergence, not traversal
        // work — scattered reflective texels mean nearly every wave has a
        // tracing lane and pays full latency whatever the other 31 do, so
        // making traversal cheaper changes nothing until the rays are
        // compacted. It is kept because it is strictly less work per ray and
        // is what the pass becomes bound by once that is fixed. The same
        // caveat applies to this layout change: it removes fetches from a path
        // that is not currently fetch-bound.
        float tL;
        float tR;
        bool goLeft = slabTest(node.lmin, node.lmax, origin, invDir, hit.t * scale, tL);
        bool goRight = slabTest(node.rmin, node.rmax, origin, invDir, hit.t * scale, tR);

        // Leaves are intersected inline, which also tightens hit.t before the
        // interior descent below chooses where to go.
        if (goLeft && node.lcount != 0) {
            blasLeafClosest(triangleOffset + node.lchild, node.lcount, origin, dir, scale, hit);
            goLeft = false;
        }
        if (goRight && node.rcount != 0) {
            blasLeafClosest(triangleOffset + node.rchild, node.rcount, origin, dir, scale, hit);
            goRight = false;
        }

        if (goLeft && goRight) {
            // Defer the farther child, descend into the nearer one.
            uint near = node.lchild;
            uint far = node.rchild;
            if (tR < tL) {
                near = node.rchild;
                far = node.lchild;
            }
            if (sp < BLAS_STACK) {
                stack[sp++] = far;
            }
            nodeIndex = near;
        } else if (goLeft) {
            nodeIndex = node.lchild;
        } else if (goRight) {
            nodeIndex = node.rchild;
        } else if (sp > 0) {
            nodeIndex = stack[--sp];
        } else {
            break;
        }
    }
}

// One TLAS leaf's instance window against the running best.
void instancesClosest(uint firstInstance, uint count, float3 origin, float3 dir, inout Hit hit) {
    for (uint i = 0; i < count; ++i) {
        uint instanceIndex = firstInstance + i;
        TlasInstance inst = tlasInstances[instanceIndex];
        float3 mOrigin = mul(inst.worldToModel, float4(origin, 1.0)).xyz;
        float3 mDir = mul(inst.worldToModel, float4(dir, 0.0)).xyz;
        float scale = length(mDir);
        if (scale < 1e-8) {
            continue;
        }
        // NO skinned skip here, deliberately: a closest-hit ray has to shade
        // what it finds, and dropping skinned instances would delete every
        // character from reflections and from the GI probe field. The bind-pose
        // approximation is accepted there — see TLAS_SKINNED_FLAG.
        float previousT = hit.t;
        blasClosest(tlasNodeOffset(inst), inst.blasTriangleOffset, mOrigin, mDir / scale, scale,
                    hit);
        if (hit.t < previousT) {
            hit.instanceIndex = instanceIndex;
        }
    }
}

Hit sceneClosest(float3 origin, float3 dir, float tMax) {
    Hit hit;
    hit.t = tMax;
    hit.triIndex = 0;
    hit.instanceIndex = 0;
    hit.bary = float2(0, 0);
    hit.valid = false;

    if (tlasNodeCount == 0) {
        return hit;
    }
    float3 invDir = 1.0 / dir;

    uint stack[TLAS_STACK];
    uint sp = 0;
    uint nodeIndex = 0;

    [loop] while (true) {
        BvhNodeGpu node = tlasNodes[nodeIndex];

        // Near child first, same reasoning as blasClosest — and it matters
        // more here: entering a TLAS leaf costs a whole BLAS descent, so
        // ordering decides how many of those get skipped outright.
        float tL;
        float tR;
        bool goLeft = slabTest(node.lmin, node.lmax, origin, invDir, hit.t, tL);
        bool goRight = slabTest(node.rmin, node.rmax, origin, invDir, hit.t, tR);

        if (goLeft && node.lcount != 0) {
            instancesClosest(node.lchild, node.lcount, origin, dir, hit);
            goLeft = false;
        }
        if (goRight && node.rcount != 0) {
            instancesClosest(node.rchild, node.rcount, origin, dir, hit);
            goRight = false;
        }

        if (goLeft && goRight) {
            uint near = node.lchild;
            uint far = node.rchild;
            if (tR < tL) {
                near = node.rchild;
                far = node.lchild;
            }
            if (sp < TLAS_STACK) {
                stack[sp++] = far;
            }
            nodeIndex = near;
        } else if (goLeft) {
            nodeIndex = node.lchild;
        } else if (goRight) {
            nodeIndex = node.rchild;
        } else if (sp > 0) {
            nodeIndex = stack[--sp];
        } else {
            break;
        }
    }
    return hit;
}

#endif // RT_TRACE_TRAVERSAL
