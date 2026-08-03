// rt_emissive.hlsli — emissive triangles as SAMPLEABLE LIGHTS.
// R6, docs/rt_lighting_plan.md §5.7.
//
// This file NAMES a resource it does not own: `emissiveRecords`, a
// StructuredBuffer<EmissiveRecord> the including shader must declare at
// whatever binding its own layout dictates — the same contract
// cluster_light_load.hlsli has with `lightDataBuf`, and for the same reason:
// two passes bind it in different slots and must decode it with one copy of the
// arithmetic. It also names `clusterTargetLuminance`, so it must be included
// AFTER cluster_light_load.hlsli.
//
// THE RECORD IS THE ALIAS BUCKET. Vose's table has exactly as many buckets as
// entries, so the alias row rides inside the record (`aliasProb`/`aliasIndex`)
// and the pass spends ONE storage-buffer slot rather than two — which is not a
// nicety here: rt_local_shadow.comp.hlsl had already spent all EIGHT of SDL3
// GPU's read-only storage buffers before this phase existed, so both of these
// arrive as read-write slots that are only ever read.
//
// ---------------------------------------------------------------------------
// THE APPROXIMATION, STATED UP FRONT
// ---------------------------------------------------------------------------
//
// The reservoir's domain is the set of RECORDS, not the set of points on them.
// A record's target function is evaluated at the triangle's CENTROID with a
// solid-angle-clamped 1/d^2, and only the SHADOW RAY samples a point on the
// triangle's surface.
//
// That is exactly the treatment the analytic path already gives a sphere: the
// reservoir picks a light, `clusterLightTarget` and `clusterLightShading`
// evaluate at the light's CENTRE, and `sampleCone` scatters the ray across the
// emitter to produce a penumbra. Doing the same thing for a triangle keeps the
// two candidate streams commensurable (their p̂ are the same physical quantity,
// which the multi-source RIS weights below require), keeps the 16-byte
// reservoir intact (a stored surface point would need 12 more bytes and there
// are none), and lets a reservoir carried across a frame boundary be re-scored
// without needing to remember where last frame's ray went.
//
// What it costs: a receiver very close to a large emitter sees a slightly wrong
// target function, because a triangle subtending most of the hemisphere is not
// well described by its centroid. The `max(d^2, area)` clamp below is what
// keeps that bounded rather than singular; beyond it, the error is in the
// SAMPLING pdf, and RIS is unbiased under any positive pdf — a bad p̂ costs
// noise, not energy.
//
// ---------------------------------------------------------------------------
// TWO-SIDED, DELIBERATELY
// ---------------------------------------------------------------------------
//
// Every forward shader in this tree adds `emissiveFactor * emissiveTexture` to
// the fragment regardless of which way the triangle faces, so an emissive
// surface in this engine emits from both sides. The cosine below is therefore
// `abs(dot(n, -L))` and not `max(...)`. Making it one-sided here would create a
// class of light that glows in the raster path and casts nothing — which is
// precisely the kind of silent disagreement the shared-arithmetic rule in this
// directory exists to prevent.
//
// CPU mirror: render/raytracing/emissive_lights.hpp, pinned by
// emissive_lights.test.cpp and alias_table.test.cpp.
//
// ---------------------------------------------------------------------------
// TWO-PHASE INCLUDE, the same contract as rt_trace.hlsli and rt_hit_shade.hlsli
// and for the same reason: the including shader needs `EmissiveRecord` to
// DECLARE its record buffer, and the functions need that buffer to already
// exist.
//
//     #define RT_EMISSIVE_TYPES
//     #include "rt_emissive.hlsli"      // EmissiveRecord
//     ... the shader's own binding block declaring `emissiveRecords` ...
//     #define RT_EMISSIVE_FUNCTIONS
//     #include "rt_emissive.hlsli"      // the sampling

#if !defined(RT_EMISSIVE_TYPES) && !defined(RT_EMISSIVE_FUNCTIONS)
#error "Define RT_EMISSIVE_TYPES or RT_EMISSIVE_FUNCTIONS before including rt_emissive.hlsli"
#endif

#if defined(RT_EMISSIVE_TYPES) && !defined(BRACKEN_RT_EMISSIVE_TYPES_INCLUDED)
#define BRACKEN_RT_EMISSIVE_TYPES_INCLUDED

/// Mirrors render::rt::EmissiveTriangle (64 B) field for field.
struct EmissiveRecord {
    float3 v0;
    float area;
    float3 e1;
    float aliasProb;
    float3 e2;
    uint aliasIndex;
    float3 radiance;
    float power;
};

#endif // RT_EMISSIVE_TYPES

#if defined(RT_EMISSIVE_FUNCTIONS) && !defined(BRACKEN_RT_EMISSIVE_FUNCTIONS_INCLUDED)
#define BRACKEN_RT_EMISSIVE_FUNCTIONS_INCLUDED

/**
 * Draw one record index from the power-weighted distribution.
 *
 * Two loads and no loop — the whole reason the alias method is here rather than
 * a CDF search, in a shader whose cost model forbids adding dependent memory
 * traffic next to a BVH walk.
 */
uint rtEmissiveSampleIndex(uint count, float u1, float u2) {
    uint bucket = min((uint)(u1 * float(count)), count - 1u);
    EmissiveRecord r = emissiveRecords[bucket];
    return u2 < r.aliasProb ? bucket : min(r.aliasIndex, count - 1u);
}

/**
 * A point distributed UNIFORMLY over the triangle.
 *
 * The square-root parametrization: (u, v) -> barycentrics (1 - sqrt(u),
 * v*sqrt(u)) on the two edges. The naive (u, v) with u+v<=1 rejection or a fold
 * would both work; this one is branch-free and is the form every reference
 * (PBRT 13.6.5) states, so it is the one worth mirroring exactly on the CPU.
 */
float3 rtEmissiveSamplePoint(EmissiveRecord r, float2 xi) {
    float su = sqrt(saturate(xi.x));
    float b1 = 1.0 - su;
    float b2 = saturate(xi.y) * su;
    return r.v0 + r.e1 * b1 + r.e2 * b2;
}

/// Geometry + incident radiance of one record at surface point `P`, in the same
/// shape (and the same units) `clusterLightShading` returns for an analytic
/// light: `radiance` is everything up to but NOT including N.L and the BRDF.
struct EmissiveSample {
    float3 L;        ///< unit surface -> emitter centroid
    float3 radiance; ///< Le * cos(theta_emitter) * area / max(d^2, area)
    float dist;      ///< to the centroid
    bool valid;
};

EmissiveSample rtEmissiveShading(EmissiveRecord r, float3 P) {
    EmissiveSample s;
    s.L = float3(0, 1, 0);
    s.radiance = float3(0, 0, 0);
    s.dist = 0.0;
    s.valid = false;

    // Centroid = v0 + (e1 + e2)/3, since e1/e2 are edges FROM v0.
    float3 c = r.v0 + (r.e1 + r.e2) * (1.0 / 3.0);
    float3 toLight = c - P;
    float d2 = dot(toLight, toLight);
    if (d2 < 1e-8 || r.area <= 0.0) {
        return s;
    }
    float d = sqrt(d2);
    s.L = toLight / d;
    s.dist = d;

    float3 nRaw = cross(r.e1, r.e2);
    float nLen = length(nRaw);
    if (nLen < 1e-8) {
        return s;
    }
    // Two-sided, per the note at the top of this file.
    float cosEmitter = abs(dot(nRaw / nLen, -s.L));
    if (cosEmitter <= 0.0) {
        return s;
    }

    // SOLID-ANGLE CLAMP. A receiver at distance d from a triangle of area A
    // sees a solid angle of roughly A*cos/d^2, which is only a solid angle
    // while it is small — as d -> 0 the expression diverges and one unlucky
    // sample becomes a firefly the denoiser then has to spend frames eating.
    // Flooring the denominator at the emitter's own area caps the term at
    // `cos`, i.e. at "the emitter fills the hemisphere", which is the physical
    // limit rather than an arbitrary ceiling.
    s.radiance = r.radiance * (cosEmitter * r.area / max(d2, r.area));
    s.valid = true;
    return s;
}

/**
 * p̂ of a record at surface (`P`, `N`) — the SAME physical quantity
 * `clusterLightTarget` returns for an analytic light: the luminance of its
 * unshadowed diffuse contribution, albedo omitted.
 *
 * They have to be the same quantity, not merely similar, because the two
 * candidate streams are resampled against each other in one reservoir. If one
 * source's p̂ were on a different scale, the multi-source RIS weights would
 * still be individually unbiased and the SELECTION would systematically prefer
 * whichever source was measured in bigger numbers — which is a noise
 * catastrophe rather than an energy error, and correspondingly hard to see.
 */
float rtEmissiveTarget(EmissiveRecord r, float3 P, float3 N) {
    EmissiveSample s = rtEmissiveShading(r, P);
    if (!s.valid) {
        return 0.0;
    }
    float ndotl = max(dot(N, s.L), 0.0);
    if (ndotl <= 0.0) {
        return 0.0;
    }
    return clusterTargetLuminance(s.radiance) * ndotl;
}

#endif // RT_EMISSIVE_FUNCTIONS
