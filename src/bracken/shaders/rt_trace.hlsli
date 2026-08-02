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
    float pad0;
    float3 e1;
    uint primId;
    float3 e2;
    float pad2;
};

struct TlasInstance {
    float4x4 worldToModel;
    float3 boundsMin;
    // Where this instance's BLAS starts in the CONVERTED (64-byte) node array.
    // Its root is that offset + 0; node links inside a BLAS are relative to it.
    uint blasNodeOffset;
    float3 boundsMax;
    uint blasTriangleOffset;
};

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
bool blasOccluded(uint nodeOffset, uint triangleOffset, float3 origin, float3 dir, float tMax) {
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
                if (triangleHit(blasTriangles[triangleOffset + node.lchild + i], origin, dir,
                                tMax)) {
                    return true;
                }
            }
            goLeft = false;
        }
        if (goRight && node.rcount != 0) {
            for (uint i = 0; i < node.rcount; ++i) {
                if (triangleHit(blasTriangles[triangleOffset + node.rchild + i], origin, dir,
                                tMax)) {
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
bool instancesOccluded(uint firstInstance, uint count, float3 origin, float3 dir, float tMax) {
    for (uint i = 0; i < count; ++i) {
        TlasInstance inst = tlasInstances[firstInstance + i];

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
        if (blasOccluded(inst.blasNodeOffset, inst.blasTriangleOffset, mOrigin, mDir / scale,
                         tMax * scale)) {
            return true;
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
    uint nodeIndex = 0;

    [loop] while (true) {
        BvhNodeGpu node = tlasNodes[nodeIndex];

        float tL;
        float tR;
        bool goLeft = slabTest(node.lmin, node.lmax, origin, invDir, tMax, tL);
        bool goRight = slabTest(node.rmin, node.rmax, origin, invDir, tMax, tR);

        // Still unordered, same reason as blasOccluded.
        if (goLeft && node.lcount != 0) {
            if (instancesOccluded(node.lchild, node.lcount, origin, dir, tMax)) {
                return true;
            }
            goLeft = false;
        }
        if (goRight && node.rcount != 0) {
            if (instancesOccluded(node.rchild, node.rcount, origin, dir, tMax)) {
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
        float previousT = hit.t;
        blasClosest(inst.blasNodeOffset, inst.blasTriangleOffset, mOrigin, mDir / scale, scale,
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
