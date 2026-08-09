// gi_probe.hlsli - the probe field's addressing, octahedral mapping, ray
// directions and blend curve, shared by every GI compute pass.
//
// EVERY function here is a transliteration of its counterpart in
// `src/bracken/render/gi/gi_math.hpp`, which is the source of truth and is
// tested with no GPU, no window and no SDL. When one side changes, the other
// changes in the same commit — a probe field whose CPU and GPU disagree about
// which texel belongs to which direction produces a slightly wrong ambient term
// that looks like a shading bug for a week.
//
// Binding-free by construction: no resource is named below, so this can be
// included before or after a shader's binding block, unlike rt_trace.hlsli.

#ifndef BRACKEN_GI_PROBE_HLSLI
#define BRACKEN_GI_PROBE_HLSLI

// ---------------------------------------------------------------------------
// Grid dimensions — mirrors gi_math.hpp
// ---------------------------------------------------------------------------

#define GI_CASCADE_COUNT 3
#define GI_GRID_X 24
#define GI_GRID_Y 12
#define GI_GRID_Z 24
#define GI_PROBES_PER_CASCADE (GI_GRID_X * GI_GRID_Y * GI_GRID_Z)
#define GI_PROBES_PER_ATLAS_ROW (GI_GRID_X * GI_GRID_Y)
#define GI_ATLAS_ROWS GI_GRID_Z
// Every atlas holds all three cascades, stacked: cascade c occupies rows
// [c * GI_ATLAS_ROWS, (c+1) * GI_ATLAS_ROWS). See gi_math.hpp's
// ATLAS_ROWS_ALL_CASCADES for why (three sampler slots in the forward pass
// instead of nine).
#define GI_ATLAS_ROWS_ALL (GI_ATLAS_ROWS * GI_CASCADE_COUNT)

#define GI_BASE_PROBE_SPACING 150.0

#define GI_IRRADIANCE_PAYLOAD 6
#define GI_VISIBILITY_PAYLOAD 14
#define GI_BORDER 1
#define GI_IRRADIANCE_TILE (GI_IRRADIANCE_PAYLOAD + 2 * GI_BORDER)
#define GI_VISIBILITY_TILE (GI_VISIBILITY_PAYLOAD + 2 * GI_BORDER)

// probe_state (R8_UINT), CPU-owned. See gi_math.hpp for why the classification
// verdict lives in the offset texture instead.
#define GI_PROBE_UNINITIALIZED 0
#define GI_PROBE_ACTIVE 1

// probe_offset.w, GPU-owned.
#define GI_CLASS_OK 0.0
#define GI_CLASS_INSIDE_GEOMETRY 2.0

#define GI_PI 3.141592653589793
#define GI_TWO_PI 6.283185307179586

// ---------------------------------------------------------------------------
// Probe indexing and the 2D tiling
// ---------------------------------------------------------------------------

int giProbeIndex(int3 c) { return c.x + GI_GRID_X * (c.y + GI_GRID_Y * c.z); }

int3 giProbeCoord(int index) {
    return int3(index % GI_GRID_X, (index / GI_GRID_X) % GI_GRID_Y,
                index / (GI_GRID_X * GI_GRID_Y));
}

// Euclidean modulo — HLSL `%` truncates toward zero like C++, and every wrap
// below crosses the origin.
int giWrapAxis(int v, int n) {
    int m = v % n;
    return m < 0 ? m + n : m;
}

/// TORUS ADDRESSING: absolute lattice coordinate -> storage slot. A probe still
/// inside the window after a scroll keeps the exact texels it had.
int3 giWrapCoord(int3 absolute) {
    return int3(giWrapAxis(absolute.x, GI_GRID_X), giWrapAxis(absolute.y, GI_GRID_Y),
                giWrapAxis(absolute.z, GI_GRID_Z));
}

int giWrappedProbeIndex(int3 absolute) { return giProbeIndex(giWrapCoord(absolute)); }

/// Top-left texel of a slot's tile, BORDER INCLUDED, in cascade `cascade`'s band.
int2 giProbeAtlasOrigin(int slot, int probeTexels, int cascade) {
    return int2((slot % GI_PROBES_PER_ATLAS_ROW) * probeTexels,
                (slot / GI_PROBES_PER_ATLAS_ROW + cascade * GI_ATLAS_ROWS) * probeTexels);
}

/// One texel of the per-probe planes (probe state, probe offset): the same
/// banding at a tile size of 1.
int2 giProbeStateTexel(int slot, int cascade) { return giProbeAtlasOrigin(slot, 1, cascade); }

/// Texel dimensions of a banded atlas with `probeTexels` per probe.
float2 giAtlasSize(int probeTexels) {
    return float2(GI_PROBES_PER_ATLAS_ROW * probeTexels, GI_ATLAS_ROWS_ALL * probeTexels);
}

float giCascadeSpacing(int cascade) { return GI_BASE_PROBE_SPACING * float(1 << cascade); }

// ---------------------------------------------------------------------------
// Octahedral mapping
// ---------------------------------------------------------------------------
//
// The same convention as `octEncode` in prepass_gbuffer.frag.hlsl and
// `octDecode` in rt_trace.hlsli (Cigolle et al. 2014). Copied rather than
// included because the blend passes want the mapping without the BVH, and
// rt_trace.hlsli's type phase drags a traversal stack constant with it.

float2 giOctEncode(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 e = n.xy;
    if (n.z < 0.0) {
        e = (1.0 - abs(n.yx)) * select(n.xy >= 0.0, float2(1.0, 1.0), float2(-1.0, -1.0));
    }
    return e * 0.5 + 0.5;
}

float3 giOctDecode(float2 e) {
    e = e * 2.0 - 1.0;
    float3 n = float3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0, float2(-t, -t), float2(t, t));
    return normalize(n);
}

/// Direction a payload texel represents. `local` is in [0, payload)^2.
float3 giTexelDirection(int2 local, int payload) {
    return giOctDecode((float2(local) + 0.5) / float(payload));
}

/**
 * BORDER WRAP. `tileLocal` is a texel of the FULL tile, [0, payload + 2)^2;
 * returns the PAYLOAD texel it must replicate, [0, payload)^2.
 *
 * The octahedron's outer square edge folds onto itself, so the border is not
 * clamp-to-edge: each edge texel takes the payload texel mirrored along that
 * edge, and each corner takes the opposite corner. Get this wrong and a
 * bilinear tap at a payload boundary blends two directions 90 degrees apart.
 *
 * CPU mirror: `probe_border_source` in render/gi/gi_math.hpp.
 */
int2 giBorderSource(int2 tileLocal, int payload) {
    const int last = payload - 1;
    const int tile = payload + 2 * GI_BORDER;
    const bool left = tileLocal.x < GI_BORDER;
    const bool right = tileLocal.x >= tile - GI_BORDER;
    const bool top = tileLocal.y < GI_BORDER;
    const bool bottom = tileLocal.y >= tile - GI_BORDER;
    const int px = clamp(tileLocal.x - GI_BORDER, 0, last);
    const int py = clamp(tileLocal.y - GI_BORDER, 0, last);

    if ((left || right) && (top || bottom)) {
        // The edge fold applied on both axes: the diagonal past a square corner
        // passes through the -Z pole into the opposite azimuth quadrant.
        return int2(left ? last : 0, top ? last : 0);
    }
    if (top || bottom) {
        return int2(last - px, top ? 0 : last);
    }
    if (left || right) {
        return int2(left ? 0 : last, last - py);
    }
    return int2(px, py);
}

// ---------------------------------------------------------------------------
// Probe ray directions
// ---------------------------------------------------------------------------

/// `count` directions spread over the WHOLE sphere (a probe has no normal).
/// CPU mirror: `spherical_fibonacci`.
float3 giSphericalFibonacci(uint index, uint count) {
    const float GOLDEN_CONJUGATE = 0.6180339887498949;
    float i = float(index);
    float phi = GI_TWO_PI * frac(i * GOLDEN_CONJUGATE);
    float cosTheta = 1.0 - (2.0 * i + 1.0) / float(count);
    float sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

/**
 * The per-frame rotation of the whole direction set, as three basis columns.
 *
 * A fixed set traced forever converges to a `count`-direction quadrature of the
 * scene rather than to the scene. Rotating it makes consecutive updates of one
 * probe sample different directions, so the EMA integrates.
 *
 * R3 sequence (the 3D generalization of r2Sequence in rt_sampling.hlsli) into
 * Shoemake's uniform-random-quaternion construction. Deterministic in the frame
 * counter alone, which is what keeps `--smoke` reproducible.
 *
 * CPU mirror: `probe_ray_rotation`.
 */
void giRayRotation(uint frameIndex, out float3 c0, out float3 c1, out float3 c2) {
    const float R3_A = 0.8191725133961644; // 1/g
    const float R3_B = 0.6710436067037893; // 1/g^2
    const float R3_C = 0.5497004779019702; // 1/g^3
    float f = float(frameIndex);
    float u1 = frac(R3_A * f);
    float u2 = frac(R3_B * f);
    float u3 = frac(R3_C * f);

    float s1 = sqrt(1.0 - u1);
    float s2 = sqrt(u1);
    float t1 = GI_TWO_PI * u2;
    float t2 = GI_TWO_PI * u3;
    float x = s1 * sin(t1);
    float y = s1 * cos(t1);
    float z = s2 * sin(t2);
    float w = s2 * cos(t2);

    c0 = float3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w));
    c1 = float3(2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w));
    c2 = float3(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y));
}

/// Direction of probe ray `index` of `count` on frame `frameIndex`.
float3 giRayDirection(uint index, uint count, uint frameIndex) {
    float3 c0, c1, c2;
    giRayRotation(frameIndex, c0, c1, c2);
    float3 d = giSphericalFibonacci(index, count);
    return normalize(c0 * d.x + c1 * d.y + c2 * d.z);
}

// ---------------------------------------------------------------------------
// Blend: change-adaptive hysteresis
// ---------------------------------------------------------------------------

#define GI_HYSTERESIS_MAX 0.97
#define GI_HYSTERESIS_MIN 0.70
#define GI_CHANGE_SATURATION 1.0
/// Standard errors of the update's OWN estimate subtracted from the observed
/// delta before the curve reads it as a lighting change. Without it the curve
/// is driven by the Monte Carlo noise of a 32-96 ray estimate whose directions
/// rotate every frame, and the window never reaches GI_HYSTERESIS_MAX at all —
/// the same positive feedback gi_probe_blend_visibility.comp.hlsl argues
/// against for the moments, with a larger n_eff. CPU mirror:
/// `PROBE_BLEND_NOISE_K`, which carries the derivation of the value.
#define GI_BLEND_NOISE_K 1.0

/// Hysteresis for a normalized (scale-free) new-vs-old delta, and R5b's global
/// reset floor from the analytic sun gradient.
///
/// The floor enters as an upper BOUND on hysteresis, never as a second term:
/// `reset` of 0 leaves the adaptive value exactly as it was, which is what
/// makes a static sun bit-identical to the pre-R5b field rather than merely
/// close. A reset of 1 pins hysteresis at GI_HYSTERESIS_MIN however small the
/// probe's own observed delta is — which is the case the floor exists for, a
/// probe whose rays have not yet been re-cast under the new sun and therefore
/// cannot report the change themselves.
///
/// CPU mirror: `probe_blend_hysteresis`.
float giBlendHysteresis(float delta, float reset) {
    float adaptive = lerp(GI_HYSTERESIS_MAX, GI_HYSTERESIS_MIN, saturate(delta / GI_CHANGE_SATURATION));
    float ceiling = lerp(GI_HYSTERESIS_MAX, GI_HYSTERESIS_MIN, saturate(reset));
    return min(adaptive, ceiling);
}

/// One EMA step. CPU mirror: `probe_blend`.
///
/// `sigma` is the standard error of `newValue` itself, computed by the caller
/// from the second moment of the same weighted ray sum. It enters as a SOFT
/// THRESHOLD on the disagreement, so a delta the estimator's own noise fully
/// explains reads as no change; `sigma == 0` is exactly the pre-gate curve.
float3 giBlend(float3 oldValue, float3 newValue, float reset, float sigma) {
    float signal = max(0.0, length(newValue - oldValue) - GI_BLEND_NOISE_K * max(sigma, 0.0));
    float delta = signal / max(length(oldValue), 1e-4);
    return lerp(newValue, oldValue, giBlendHysteresis(delta, reset));
}

// ---------------------------------------------------------------------------
// Classification / relocation thresholds — mirrors gi_math.hpp
// ---------------------------------------------------------------------------

#define GI_RELOCATE_BACKFACE_FRACTION 0.25
#define GI_INSIDE_BACKFACE_FRACTION 0.75
/// The LEAVE half of each dwell band — see gi_math.hpp for why a bare threshold
/// on a 64-ray binomial oscillates and what that does to the trilinear cage.
#define GI_RELOCATE_BACKFACE_LEAVE 0.10
#define GI_INSIDE_BACKFACE_LEAVE 0.60
#define GI_MAX_OFFSET_FRACTION 0.45
#define GI_VISIBILITY_DISTANCE_SCALE 1.5
#define GI_VISIBILITY_SHARPNESS 50.0
/// Minimum mean resultant length, per backface ray, for the relocation
/// direction to mean anything. CPU mirror: `PROBE_RELOCATE_MIN_COHERENCE`.
#define GI_RELOCATE_MIN_COHERENCE 0.10

/// pow(c, GI_VISIBILITY_SHARPNESS) <= 1e-8  <=>  c <= this. Lets the visibility
/// blend reject the ~85% of rays outside a texel's 46-degree lobe with a compare
/// instead of a log2/exp2 pair. Recompute as pow(1e-8, 1/sharpness) if
/// GI_VISIBILITY_SHARPNESS is ever retuned.
#define GI_VISIBILITY_WEIGHT_CUTOFF 0.6918309709

/// Ceiling on giRayCount, and the size of the blend passes' groupshared
/// direction cache. CPU mirror: `MAX_PROBE_RAYS`, pinned against
/// gi_budget(GiQuality::High) by a static_assert in probe_schedule.hpp.
#define GI_MAX_PROBE_RAYS 96

#endif // BRACKEN_GI_PROBE_HLSLI
