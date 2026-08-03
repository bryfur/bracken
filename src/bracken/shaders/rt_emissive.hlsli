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
// ---------------------------------------------------------------------------
// SINCE R8 THE DRAW IS FROXEL-LOCAL (docs/rt_lighting_plan.md §5.7 follow-up 2)
// ---------------------------------------------------------------------------
//
// The alias table above is still here and still normalizes the same way, but the
// pass that samples records now draws them from a MIXTURE of it with the pixel's
// PER-FROXEL emissive list — the second half of this file, opt-in with
// RT_EMISSIVE_LISTS because the gradient replay includes this file and binds no
// such table. `render/raytracing/emissive_clusters.hpp` owns the derivation, the
// CPU binner and the measurements that chose the shipped share.
//
// CPU mirror: render/raytracing/emissive_lights.hpp +
// render/raytracing/emissive_clusters.hpp, pinned by emissive_lights.test.cpp,
// emissive_clusters.test.cpp and alias_table.test.cpp.
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

/**
 * Length of the shadow ray toward a sampled point on an emissive triangle, in
 * ONE place because two shaders must agree on it exactly: the trace casts it and
 * R5b's gradient pass replays it.
 *
 * Two clamps, for two different reasons.
 *
 * THE EMITTER MUST NOT OCCLUDE ITS OWN LIGHT. Unlike an analytic light this one
 * IS geometry in the TLAS and the ray is aimed exactly at a point on it, so an
 * any-hit walk to the full distance hits the emitter every single time and every
 * emissive light comes out fully shadowed. Stopping short by a RELATIVE margin
 * (rather than the absolute bias the analytic path uses) is what keeps that true
 * across four orders of magnitude of scene scale: 0.2% of the distance is under
 * a millimetre at arm's length and still clears the triangle at 500 units.
 * Anything genuinely between the surface and the emitter is longer than that
 * margin by construction, so nothing real is missed.
 *
 * AND THE RAY IS NOT OTHERWISE TRUNCATED, which R8 established by measurement
 * rather than by taste. Bounding this walk at the localization horizon
 * (`k_emissive_froxel_max_radius`) is the obvious companion to bounding the
 * BINNING there, it is cheap, and it is wrong: a distant emitter behind a
 * building comes back visible, and the night village measured **37.0 -> 43.8
 * mean luma** under it — light through walls, not a speedup. The near occlusion
 * a truncated ray does capture is the occlusion that gives a shadow its shape,
 * which is exactly why the failure is invisible in the shadow and glaring in the
 * fill. R8 localizes the DRAW and leaves the ray alone.
 */
float rtEmissiveRayLength(float dist, float depthBias, float normalBias) {
    return min(dist * 0.998, dist - depthBias) - normalBias;
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

// ---------------------------------------------------------------------------
// PER-FROXEL emissive candidate lists (R8, docs/rt_lighting_plan.md §5.7's
// follow-up 2). Opt-in with RT_EMISSIVE_LISTS, because the gradient replay
// includes this file and does NOT bind the table — it replays a record index
// somebody else already chose, and never draws a candidate of its own.
//
// The including shader must declare, before the FUNCTIONS include:
//     RWStructuredBuffer<uint> emissiveFroxels;   (read-only in practice)
//
// CPU mirror + the derivation: render/raytracing/emissive_clusters.hpp.
// ---------------------------------------------------------------------------
#ifdef RT_EMISSIVE_LISTS

/// Mirrors k_emissive_froxel_capacity / k_emissive_froxel_stride. The block is
/// [count][record x capacity], the records ASCENDING by index.
#define RT_EMISSIVE_FROXEL_CAPACITY 16u
#define RT_EMISSIVE_FROXEL_STRIDE 17u

/// Records this froxel holds, clamped for the same reason
/// clusterFroxelLightCount clamps: a stale or partially uploaded table must not
/// be able to spin a loop past the block it owns.
uint rtEmissiveFroxelCount(uint froxel) {
    return min(emissiveFroxels[froxel * RT_EMISSIVE_FROXEL_STRIDE], RT_EMISSIVE_FROXEL_CAPACITY);
}

uint rtEmissiveFroxelRecord(uint froxel, uint i) {
    return emissiveFroxels[froxel * RT_EMISSIVE_FROXEL_STRIDE + 1u + i];
}

/// Is `rec` in this froxel's list? The `q_g` half of the mixture pdf needs it,
/// and the CPU packs each block ASCENDING BY RECORD INDEX so this early-outs
/// rather than always costing the full capacity.
bool rtEmissiveFroxelHolds(uint froxel, uint count, uint rec) {
    for (uint i = 0; i < count; ++i) {
        const uint r = rtEmissiveFroxelRecord(froxel, i);
        if (r == rec) {
            return true;
        }
        if (r > rec) {
            return false; // ascending — past where it would have been
        }
    }
    return false;
}

/// q_f — the share of a draw that actually reaches the froxel list. An EMPTY
/// list takes none of it, which is what makes an unbinned froxel behave exactly
/// as R6 did instead of dividing by zero.
float rtEmissiveFroxelShare(float share, uint count) { return count > 0u ? share : 0.0; }

/// One mixture draw: the record, its index, and the SOURCE PDF it was drawn
/// with — which is the number the RIS weight divides by, so it is returned
/// rather than recomputed at the call site.
struct EmissiveDraw {
    EmissiveRecord rec;
    uint index;
    float pdf;
};

/**
 * Draw one emissive candidate from the MIXTURE
 *
 *     p(x) = q_f * [x in L_f] / |L_f|  +  q_g * power(x) / totalPower
 *
 * over the froxel list and the scene-wide power alias table. The two halves
 * OVERLAP by design and a record reachable both ways simply carries both terms —
 * which is what "dedupe" means for a mixture, and why every record keeps a
 * positive pdf and the estimator stays unbiased over the full record set.
 *
 * THE LIST HALF IS UNIFORM, and that is a measured choice rather than the lazy
 * one. Drawing it proportional to the same relevance score that put a record in
 * it is the textbook improvement, it was implemented (a per-froxel CDF in the
 * second half of each block), and it cost 0.4 ms of device time for no
 * measurable change in either mean luma or the `di noise` probe: the list is
 * short, the reservoir re-weights by p̂ anyway, and a 16-step dependent-load
 * search is expensive next to what it saves. One index per entry it is.
 *
 * `u1` selects the component and, rescaled onto [0, 1), the entry within it —
 * exact, because `(u1 - a) / (b - a)` is uniform on [0, 1) conditioned on the
 * branch it chose. `u2` is the alias table's second uniform; the froxel branch
 * does not need it and takes it anyway, so BOTH branches consume the same two
 * draws and the sample stream keeps the shape R6 gave it.
 *
 * WITH `share == 0` THIS IS R6 BIT FOR BIT: q_f is 0, `t` is `u1` unchanged, the
 * membership test is skipped, and the pdf is `power / totalPower` times an exact
 * 1.0. That is what makes the A/B control reproduce the old figures rather than
 * merely approach them.
 */
EmissiveDraw rtEmissiveDrawMixture(uint froxel, uint listCount, float share, uint recordCount,
                                   float totalPower, float u1, float u2) {
    EmissiveDraw d;
    d.index = 0u;
    d.pdf = 0.0;

    const float qF = rtEmissiveFroxelShare(share, listCount);
    const float qG = 1.0 - qF;

    // `qG <= 0` takes the froxel branch unconditionally rather than relying on
    // `u1 < 1.0`: an RNG that ever returned exactly 1.0 would otherwise fall into
    // the global branch with zero probability mass, draw a record with pdf 0 and
    // contribute a candidate the reservoir must count and cannot weight.
    bool inList = false;
    if (qG <= 0.0 || u1 < qF) {
        const float t = saturate(u1 / max(qF, 1e-6));
        const uint slot = min((uint)(t * float(listCount)), listCount - 1u);
        d.index = rtEmissiveFroxelRecord(froxel, slot);
        inList = true;
    } else {
        const float t = saturate((u1 - qF) / max(qG, 1e-6));
        d.index = rtEmissiveSampleIndex(recordCount, t, u2);
        inList = qF > 0.0 ? rtEmissiveFroxelHolds(froxel, listCount, d.index) : false;
    }
    // Defensive only: the CPU contract is that every listed index is in range,
    // and `share` is sent as 0 on any frame the table is not this frame's.
    d.index = min(d.index, recordCount - 1u);
    d.rec = emissiveRecords[d.index];

    d.pdf = qG * (d.rec.power / max(totalPower, 1e-6)) +
            (inList ? qF / float(listCount) : 0.0);
    return d;
}

#endif // RT_EMISSIVE_LISTS

#endif // RT_EMISSIVE_FUNCTIONS
