// gi_sample.hlsli - read the probe field: world position + shading normal ->
// diffuse irradiance. THE consumption path of phase R1c
// (docs/rt_lighting_plan.md §5.2).
//
// EVERY function here is a transliteration of its counterpart in
// `src/bracken/render/gi/gi_math.hpp`, which is the source of truth and is
// tested with no GPU, no window and no SDL — including the cascade blend's
// continuity, which is asserted numerically there rather than eyeballed here.
//
// THE UNIT is cosine-weighted MEAN INCOMING RADIANCE (irradiance/pi), the same
// convention as the IBL irradiance cube (`ibl_irradiance.frag.hlsl`). That is
// deliberate and load-bearing: the value below substitutes directly for
// `ambientIBL`'s cube tap and for `ambientPBR`'s hemisphere gradient, with no
// scale factor anywhere. Both of those keep their SPECULAR halves untouched —
// R4 is what changes those, not this.
//
// ---------------------------------------------------------------------------
// WHAT THE INCLUDING SHADER MUST DECLARE FIRST
// ---------------------------------------------------------------------------
// Like cluster_lighting.hlsli, this file names resources it does not own, so
// the including shader owns their slot numbers. Declare, by these exact names,
// three consecutive fragment samplers (each `[[vk::combinedImageSampler]]`):
//
//   giIrradianceAtlas / giIrradianceAtlasSampler   RGBA16F, LINEAR clamp
//   giVisibilityAtlas / giVisibilityAtlasSampler   RG16F,   LINEAR clamp
//   giProbeMeta       / giProbeMetaSampler         RGBA16F, POINT  clamp
//
// The first two are octahedral payloads whose one-texel border ring exists
// precisely so a bilinear tap at a payload edge wraps the octahedron — sampling
// them with a point filter throws that away and bands every ambient gradient.
// The third is one texel per probe and must NOT be filtered: a linear tap
// blends two unrelated probes' relocation offsets into a position that is
// neither, so it is read with Load() below and its sampler exists only because
// SDL3 GPU binds textures and samplers as pairs.
//
// And in a cbuffer, by these names:
//
//   int4  giCascadeBase[GI_CASCADE_COUNT];  // xyz = window corner probe, w unused
//   float giEnabled;                        // 0 = never call giSampleDiffuse
//
// giCascadeBase is passed rather than re-derived from the camera position on
// purpose. It is `render::gi::cascade_base()` of whatever position ProbeVolume
// was scrolled with, and a shader that re-snapped its own copy would agree with
// the CPU right up until the two were handed different camera positions — at
// which point every probe in the field would answer for the wrong point in the
// world and the failure would look like a shading bug.
//
// ---------------------------------------------------------------------------
// GI_SAMPLE_VOLUME_ONLY — for shaders that want giSampleVolume and nothing else
// ---------------------------------------------------------------------------
// `#define GI_SAMPLE_VOLUME_ONLY` before including this file to compile out
// everything that reads `giVisibilityAtlas` (giTapVisibility, giChebyshev,
// giSampleCascade, giSampleDiffuse). Such a shader then declares only TWO GI
// samplers — giIrradianceAtlas and giProbeMeta — and must not declare the
// visibility pair.
//
// This is not a convenience. A `Texture2D`/`SamplerState` pair that no
// executed code ever pairs up does NOT survive DXC's combined-image-sampler
// legalization: the two stay separate SPIR-V variables, SDL_shadercross
// reflects the orphaned image as a STORAGE texture rather than as a sampler,
// and the resulting descriptor layout no longer matches the packed
// samplers-then-storage-textures order SDL3 GPU assumes (shaders/README.md).
// volumetric_fog.frag is the one shader in that position — it needs the media
// variant only — and it declares the opt-out for exactly this reason.

#ifndef BRACKEN_GI_SAMPLE_HLSLI
#define BRACKEN_GI_SAMPLE_HLSLI

#include "gi_probe.hlsli"

// ---------------------------------------------------------------------------
// Constants — mirrors the SAMPLE_* block in render/gi/gi_math.hpp, which
// carries the full reasoning for each.
// ---------------------------------------------------------------------------

// DDGI's two-part sample bias, in fractions of the cascade's probe spacing:
// along the normal (a shading point sits ON a surface, and the probes that see
// that surface are all on one side of it) and toward the camera (what survives
// the normal bias is worst at grazing angles). RTXGI's defaults.
#define GI_SAMPLE_NORMAL_BIAS_FRACTION 0.20
#define GI_SAMPLE_VIEW_BIAS_FRACTION 0.30

// How far inside a cascade's window the biased point must lie, in probes, for
// that cascade to answer: at exactly 0.5 the 8-tap cage is [0,1] on every axis,
// wholly inside the window. Any less and a tap wraps the torus and reads the
// opposite side of the world.
#define GI_SAMPLE_CASCADE_MARGIN 0.5
// Width of the fade band at that boundary. THIS IS THE CONTINUITY MECHANISM: a
// cascade's confidence ramps to zero over the last probe of its window, so
// where it stops answering it was already contributing nothing and the
// next-coarser cascade already held all the weight. Without it a cascade
// boundary is a visible step in the ambient term.
#define GI_SAMPLE_CASCADE_FADE 1.0

// Live trilinear weight at which a cascade is trusted completely. The eight
// weights sum to 1 when every probe is usable, so this is "at least 30% of the
// cage answered"; below it the confidence ramps down and the next-coarser
// cascade fills in. It stops a cage of mostly-buried probes stretching one
// survivor's radiance over a room.
#define GI_SAMPLE_WEIGHT_SUM_FLOOR 0.30
// Floor under the squared backface term, so a cage facing entirely away from
// the shading normal ends up dim rather than black. Same value and reason as
// the bounce sampler in gi_probe_trace.comp.hlsl.
#define GI_SAMPLE_BACKFACE_FLOOR 0.02

// Chebyshev variance floor as a fraction of spacing^2 (the moments are world
// distances, so a coarse cascade's are four times larger), and the slack before
// the test attenuates at all. Without the floor a texel that has seen one
// consistent distance reports ~0 variance and the bound collapses into a hard
// binary test that flickers as the ray set rotates; without the slack, a
// shading point at the same distance as the surface its own probe measured
// tests as occluded and every wall self-darkens.
#define GI_SAMPLE_CHEBYSHEV_MIN_VARIANCE 1e-4
#define GI_SAMPLE_CHEBYSHEV_BIAS_FRACTION 0.02

// ---------------------------------------------------------------------------
// Atlas taps
// ---------------------------------------------------------------------------

/// The probe's irradiance along `dir`. Bilinear: payload texel centres live at
/// origin + border + oct * payload, so oct = 0 lands exactly on the seam
/// between the border texel and payload texel 0 — which is the whole reason the
/// border ring is written.
float3 giTapIrradiance(int cascade, int slot, float3 dir) {
    float2 texel = float2(giProbeAtlasOrigin(slot, GI_IRRADIANCE_TILE, cascade)) + float(GI_BORDER) +
                   giOctEncode(dir) * float(GI_IRRADIANCE_PAYLOAD);
    return giIrradianceAtlas
        .SampleLevel(giIrradianceAtlasSampler, texel / giAtlasSize(GI_IRRADIANCE_TILE), 0)
        .rgb;
}

/// xyz = relocation offset in world units, w = GI_CLASS_OK or
/// GI_CLASS_INSIDE_GEOMETRY. Load, never Sample — see the header note.
float4 giTapMeta(int cascade, int slot) {
    return giProbeMeta.Load(int3(giProbeStateTexel(slot, cascade), 0));
}

// Everything below to the matching #endif reads `giVisibilityAtlas`, directly
// or through giSampleCascade. A GI_SAMPLE_VOLUME_ONLY shader does not declare
// that pair and must not compile these — see the header note.
#ifndef GI_SAMPLE_VOLUME_ONLY

/// The probe's depth moments (mean, mean^2) along `dir`.
float2 giTapVisibility(int cascade, int slot, float3 dir) {
    float2 texel = float2(giProbeAtlasOrigin(slot, GI_VISIBILITY_TILE, cascade)) + float(GI_BORDER) +
                   giOctEncode(dir) * float(GI_VISIBILITY_PAYLOAD);
    return giVisibilityAtlas
        .SampleLevel(giVisibilityAtlasSampler, texel / giAtlasSize(GI_VISIBILITY_TILE), 0)
        .rg;
}

// ---------------------------------------------------------------------------
// Chebyshev
// ---------------------------------------------------------------------------

/// Soft visibility between a probe and a shading point `dist` away, from that
/// probe's mean and mean-squared ray distance in that direction. 1 = visible.
///
/// This is what makes a sparse grid usable indoors: without it, a probe on the
/// far side of a wall is trilinearly nearby and leaks its radiance through.
/// CPU mirror: `chebyshev_visibility`.
float giChebyshev(float dist, float2 moments, float spacing) {
    float bias = GI_SAMPLE_CHEBYSHEV_BIAS_FRACTION * spacing;
    if (dist <= moments.x + bias) {
        return 1.0;
    }
    float floorVariance = GI_SAMPLE_CHEBYSHEV_MIN_VARIANCE * spacing * spacing;
    float variance = max(moments.y - moments.x * moments.x, floorVariance);
    float d = dist - (moments.x + bias);
    float v = variance / (variance + d * d);
    // Cubed, per DDGI: the raw bound is far too permissive at the
    // one-sample-per-texel-per-update rates a probe field actually runs at, and
    // light that leaks 30% through a wall reads as a bug rather than softness.
    return max(v * v * v, 0.0);
}

// ---------------------------------------------------------------------------
// One cascade
// ---------------------------------------------------------------------------

/// The biased sample position for a cascade. Cascade-DEPENDENT on purpose (the
/// bias scales with that cascade's spacing), which is why the containment test
/// below uses it and not the raw shading point.
float3 giBiasPosition(float3 P, float3 N, float3 V, int cascade) {
    float sp = giCascadeSpacing(cascade);
    return P + N * (GI_SAMPLE_NORMAL_BIAS_FRACTION * sp) + V * (GI_SAMPLE_VIEW_BIAS_FRACTION * sp);
}

/// Trilinear-plus-weights gather over the eight probes surrounding `P` in
/// cascade `cascade`. Returns the renormalised irradiance; `confidence` is how
/// much of the final result this cascade is entitled to, in [0,1].
/// CPU mirror: `sample_cascade`.
float3 giSampleCascade(int cascade, float3 P, float3 N, float3 V, out float confidence) {
    confidence = 0.0;
    float sp = giCascadeSpacing(cascade);
    float3 biased = giBiasPosition(P, N, V, cascade);
    float3 local = biased / sp - float3(giCascadeBase[cascade].xyz);

    float3 lo = float3(GI_SAMPLE_CASCADE_MARGIN, GI_SAMPLE_CASCADE_MARGIN, GI_SAMPLE_CASCADE_MARGIN);
    float3 hi = float3(GI_GRID_X - 1.0, GI_GRID_Y - 1.0, GI_GRID_Z - 1.0) - lo;
    float3 inset3 = min(local - lo, hi - local);
    float inset = min(inset3.x, min(inset3.y, inset3.z));
    if (inset <= 0.0) {
        return float3(0, 0, 0);
    }
    float edgeFade = saturate(inset / GI_SAMPLE_CASCADE_FADE);

    float3 baseF = floor(local);
    float3 f = local - baseF;
    int3 b = int3(baseF);

    float3 sum = float3(0, 0, 0);
    float wsum = 0.0;
    [unroll]
    for (int i = 0; i < 8; ++i) {
        int3 o = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        int3 absolute = giCascadeBase[cascade].xyz + b + o;
        int slot = giWrappedProbeIndex(absolute);

        float4 meta = giTapMeta(cascade, slot);
        if (meta.w == GI_CLASS_INSIDE_GEOMETRY) {
            continue; // buried, or never filled — both leak if used
        }
        // The probe's irradiance describes the world at its RELOCATED position.
        // Ignoring the offset reads a probe nudged out of a wall as if it were
        // still in it, which is exactly the leak the nudge existed to prevent.
        float3 probePos = float3(absolute) * sp + meta.xyz;
        float3 toProbe = probePos - biased;
        float dist = length(toProbe);
        float3 dir = dist > 1e-4 ? toProbe / dist : N;

        float3 tri = lerp(1.0 - f, f, float3(o));
        float wrapped = (dot(dir, N) + 1.0) * 0.5;
        float w = tri.x * tri.y * tri.z * (wrapped * wrapped + GI_SAMPLE_BACKFACE_FLOOR) *
                  giChebyshev(dist, giTapVisibility(cascade, slot, -dir), sp);
        if (w <= 0.0) {
            continue;
        }
        sum += w * giTapIrradiance(cascade, slot, N);
        wsum += w;
    }
    if (wsum <= 1e-6) {
        return float3(0, 0, 0);
    }
    // RENORMALISE over the probes that actually answered: dividing by the live
    // weight rather than by the full trilinear total is what makes a cage with
    // two buried probes report the six survivors' average instead of three
    // quarters of it, which would read as a dark halo around every wall the
    // relocation pass gave up on.
    confidence = edgeFade * saturate(wsum / GI_SAMPLE_WEIGHT_SUM_FLOOR);
    return sum / wsum;
}

// ---------------------------------------------------------------------------
// The field
// ---------------------------------------------------------------------------

/**
 * Diffuse irradiance/pi at `P` for a surface facing `N`, seen from a camera in
 * direction `V` (surface -> camera, unit).
 *
 * Finest cascade first, each taking as much of the remaining weight as its
 * confidence allows, with the hemisphere ambient absorbing what is left. The
 * fallback is deliberately EnvironmentState's hemisphere and not the IBL
 * irradiance cube even when IBL is on: it is the engine's GI-off tier, so a
 * pixel the field cannot answer for looks exactly as it did before this phase.
 *
 * CPU mirror: `sample_field`.
 */
float3 giSampleDiffuse(float3 P, float3 N, float3 V, float3 ambientSkyColor,
                       float3 ambientGroundColor) {
    float3 accum = float3(0, 0, 0);
    float remaining = 1.0;
    [unroll]
    for (int c = 0; c < GI_CASCADE_COUNT; ++c) {
        if (remaining <= 0.0) {
            continue;
        }
        float confidence;
        float3 irradiance = giSampleCascade(c, P, N, V, confidence);
        if (confidence <= 0.0) {
            continue;
        }
        float take = min(confidence, remaining);
        accum += irradiance * take;
        remaining -= take;
    }
    float3 hemisphere = lerp(ambientGroundColor, ambientSkyColor, saturate(N.y * 0.5 + 0.5));
    return accum + hemisphere * remaining;
}

#endif // GI_SAMPLE_VOLUME_ONLY

// ---------------------------------------------------------------------------
// The media variant
// ---------------------------------------------------------------------------

/// Cheap gain over the hemisphere mean, clamped so one runaway probe cannot
/// turn the fog into a light source.
#define GI_VOLUME_MAX_GAIN 4.0

/**
 * PARTICIPATING MEDIA at `P`: one probe, no cage, no visibility test.
 *
 * Deliberately not `giSampleDiffuse`. The volumetric fog pass marches 24 steps
 * per (half-res) pixel, and the full sampler costs 24 texture taps per call —
 * three orders of magnitude more work than this pass has ever spent on its
 * ambient term, for a signal that is integrated over the whole march and then
 * upsampled and temporally resolved. What media actually needs is the probe's
 * mean radiance, low-frequency and cheap, and that is what this returns:
 * the NEAREST usable probe, averaged over its +Y and -Y octahedral taps
 * (a medium has no normal, so one hemisphere would systematically over- or
 * under-report). Three texture fetches per step.
 *
 * The price is that the estimate is piecewise constant over 150-unit cells.
 * Three things hide it: the march is dithered per pixel (interleaved gradient
 * noise), consecutive steps land in different cells, and the whole result is a
 * transmittance-weighted integral over the march rather than a point sample.
 *
 * Falls back to the hemisphere MEAN — exactly what averaging +Y and -Y taps of
 * `lerp(ground, sky, N.y*0.5+0.5)` gives — when no cascade covers `P` or its
 * nearest probe is unusable. That exactness is what preserves the fog pass's
 * sky agreement; see volumetric_fog.frag.hlsl.
 */
float3 giSampleVolume(float3 P, float3 ambientSkyColor, float3 ambientGroundColor) {
    float3 hemisphereMean = (ambientSkyColor + ambientGroundColor) * 0.5;
    for (int c = 0; c < GI_CASCADE_COUNT; ++c) {
        float sp = giCascadeSpacing(c);
        float3 local = P / sp - float3(giCascadeBase[c].xyz);
        // No bias and no half-probe margin: a single nearest tap needs the
        // ROUNDED coordinate to be in range, not a whole 8-tap cage.
        if (any(local < -0.5) || any(local > float3(GI_GRID_X, GI_GRID_Y, GI_GRID_Z) - 0.5)) {
            continue;
        }
        int3 absolute = giCascadeBase[c].xyz + int3(round(local));
        int slot = giWrappedProbeIndex(absolute);
        if (giTapMeta(c, slot).w == GI_CLASS_INSIDE_GEOMETRY) {
            continue; // buried or never filled — let a coarser cascade answer
        }
        return 0.5 * (giTapIrradiance(c, slot, float3(0.0, 1.0, 0.0)) +
                      giTapIrradiance(c, slot, float3(0.0, -1.0, 0.0)));
    }
    return hemisphereMean;
}

#endif // BRACKEN_GI_SAMPLE_HLSLI
