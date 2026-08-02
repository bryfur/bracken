#pragma once

/**
 * The two roughness curves phase R3 of `docs/rt_lighting_plan.md` §5.4 added,
 * as the CPU source of truth for shader transliterations that cannot otherwise
 * be tested.
 *
 * Both are pure functions of roughness, both decide how much of one signal is
 * traded for another, and both are the kind of thing that silently drifts when
 * a constant is edited on one side of a shader boundary. `gi_math.hpp` sets the
 * precedent: the mapping lives here, is tested with no GPU and no window, and
 * the shader is a transliteration.
 *
 * Mirrors:
 *   virtual_reprojection_weight -> rtVirtualReprojectionWeight
 *                                  (shaders/rt_reflection_denoise_temporal.comp.hlsl)
 *   rough_specular_gi_weight    -> giRoughSpecularWeight (shaders/pbr.hlsli)
 *
 * Tier: pure (no state, no device, callable from anywhere).
 */

#include <algorithm>

namespace bracken::render {

/// HLSL `smoothstep`, exactly: clamped Hermite, and undefined for e0 >= e1 —
/// every caller below passes literal constants that satisfy it.
[[nodiscard]] constexpr float rt_smoothstep(float edge0, float edge1, float x) noexcept {
    const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// ---------------------------------------------------------------------------
// Virtual-hit-point reprojection
// ---------------------------------------------------------------------------

/**
 * Where a reflection LIVES, as a function of the receiver's roughness.
 *
 * A mirror's reflection is an image of a point BEHIND the surface — the
 * "virtual" hit point, the eye ray reflected and extended by the hit distance —
 * and it moves under camera motion at roughly twice the parallax rate of the
 * surface carrying it. Reprojecting it through the surface's own motion vector
 * is therefore wrong by that whole factor, which is the smear that forced
 * `RtReflectionSettings::max_accum_frames` down to 16.
 *
 * As roughness grows the specular lobe widens until the reflection is
 * effectively painted ON the surface, at which point the surface's motion
 * vector is the correct one and the virtual point — whose position depends on a
 * hit distance that is itself a one-sample estimate of a wide lobe — is noise.
 *
 * The band is deliberately NARROW and low. 0.05 is "still a mirror"; by 0.40
 * the lobe covers enough solid angle that the virtual point stops being a point
 * at all. Both sit below `RtReflectionSettings::fade_start` (0.35) and
 * `max_roughness` (0.75), so the curve spans the part of the range where the
 * trace is doing most of the work.
 */
inline constexpr float VIRTUAL_REPROJECTION_MIRROR_ROUGHNESS = 0.05f;
inline constexpr float VIRTUAL_REPROJECTION_SURFACE_ROUGHNESS = 0.40f;

/// 1 = reproject the virtual hit point, 0 = reproject the surface. The
/// denoiser lerps the two motion vectors by this, rather than picking one, so
/// there is no roughness at which a pixel's history jumps.
[[nodiscard]] constexpr float virtual_reprojection_weight(float roughness) noexcept {
    return 1.0f - rt_smoothstep(VIRTUAL_REPROJECTION_MIRROR_ROUGHNESS,
                                VIRTUAL_REPROJECTION_SURFACE_ROUGHNESS, roughness);
}

// ---------------------------------------------------------------------------
// Rough-tier specular fallback
// ---------------------------------------------------------------------------

/**
 * How much of the ambient SPECULAR lobe comes from the probe field instead of
 * the prefiltered IBL cube, as a function of roughness.
 *
 * Above the reflection trace's gate the engine's only specular answer is the
 * IBL prefilter chain, which is built over the procedural SKY alone. At high
 * roughness that lobe is wide enough to be an irradiance-like average of
 * everything the surface can see — and what it actually returns is an average
 * of open sky the surface may be nowhere near. Rough metal in a courtyard
 * mirrors a sky it cannot see.
 *
 * The probe field's irradiance is a good proxy for exactly that average (this
 * is standard practice: at alpha near 1 the GGX lobe and the cosine lobe
 * integrate to nearly the same thing), and it is occluded, so the swap is a
 * strict improvement where it applies. Below 0.6 the prefiltered lobe is still
 * directional enough to matter and the substitution would flatten it, so the
 * band starts there and completes at fully rough.
 *
 * Scaled by the caller's existing `giWeight`, so this is zero whenever GI is
 * off — the branch is not taken and the no-GI path stays bit-identical.
 */
inline constexpr float ROUGH_SPECULAR_GI_START = 0.6f;
inline constexpr float ROUGH_SPECULAR_GI_END = 1.0f;

[[nodiscard]] constexpr float rough_specular_gi_weight(float roughness) noexcept {
    return rt_smoothstep(ROUGH_SPECULAR_GI_START, ROUGH_SPECULAR_GI_END, roughness);
}

} // namespace bracken::render
