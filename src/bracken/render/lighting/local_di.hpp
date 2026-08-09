#pragma once

/**
 * The DEMODULATION CONTRACT of the Full-DI tier of local-light shadows
 * (`LocalRtShadowMode::Full`, phase R5c of `docs/rt_lighting_plan.md` §5.3).
 *
 * In the Ratio tier the forward cluster loop still evaluates every point and
 * spot light analytically and one traced scalar multiplies the subtotal. In
 * Full the traced lights leave that loop entirely: the reservoir pass evaluates
 * the WINNER'S OWN BRDF, splits it into a diffuse and a specular half with the
 * per-pixel material factors it does NOT know DIVIDED OUT, and the forward
 * shader multiplies them back in. This header is the arithmetic of that split
 * and of the multiply-back, as the CPU source of truth for
 * `shaders/local_di.hlsli` (the forward half) and the DI block of
 * `shaders/rt_local_shadow.comp.hlsl` (the trace half).
 *
 * WHY DEMODULATE AT ALL. The traced pass runs at half resolution and its
 * estimate is stochastic. Albedo is neither: it is a full-resolution, noise-free
 * texture fetch. Denoising `albedo * lighting` blurs the albedo across material
 * edges and reconstructs it at half resolution; denoising `lighting` alone and
 * multiplying afterwards keeps every texel of the albedo. That is the whole
 * technique, and it is why the buffers below carry a quantity that is not a
 * radiance anyone would want to look at.
 *
 * ---------------------------------------------------------------------------
 * THE SPLIT
 * ---------------------------------------------------------------------------
 *
 * `shaders/pbr.hlsli`'s `evaluatePBR`, for one light, is
 *
 *     F0    = lerp(0.04, albedo, metallic)
 *     f     = (1 - LdotH)^5                       // the Schlick fraction
 *     F     = F0 + (1 - F0) * f
 *     spec  = D * Vis * F
 *     diff  = (1 - F) * (1 - metallic) * albedo * (ls * vs / PI)
 *     out   = (diff + spec) * lightColor * NdotL
 *
 * with `ao` fixed at 1 in the cluster loop. Write `E = lightColor * atten *
 * cone * NdotL * V * W` — everything the trace knows and the forward does not,
 * including the traced visibility bit `V` and the reservoir's unbiasing weight
 * `W`. Then the trace writes:
 *
 *     diffuseDemod = (1 - F_d) * (1 - metallic) * (ls * vs / PI) * E      (rgb)
 *     specDemod    = D * Vis * F_d * E                                    (rgb)
 *     schlick      = f                                                    (alpha)
 *
 * where `F_d` is the DIELECTRIC Fresnel — Schlick against F0 = 0.04 — because
 * the G-buffer carries roughness and metallic but NOT albedo, and F0 needs
 * albedo for a metal.
 *
 * R7b LEAVES ALL OF THIS ALONE. It adds a third target carrying the winner's
 * DIRECTION and a per-pixel RELIEF FACTOR built from it — see "THE DIRECTIONAL
 * BASIS" below — but the two buffers above hold exactly what they held in R5c,
 * bit for bit, and every statement in this section is still true of them.
 *
 * ---------------------------------------------------------------------------
 * THE MULTIPLY-BACK, AND WHY BOTH METAL HOOKS ARE EXACT RATHER THAN FUDGES
 * ---------------------------------------------------------------------------
 *
 * Both halves need a `F_true / F_d` correction, and in both cases the ratio
 * turns out to be COMPUTABLE from what the forward pass has rather than merely
 * approximable. That is the difference between this tier being a second
 * estimate of the same lighting and it being a different, dimmer picture.
 *
 * DIFFUSE. The energy split is `kd = (1 - F)(1 - metallic)`, and Schlick
 * factors as `1 - F = (1 - F0)(1 - f)` — so the `f` cancels out of the ratio
 * entirely and what is left needs only the albedo:
 *
 *     kd_true / kd_dielectric = (1 - F0) / 0.96
 *                             = 1 - metallic * (albedo - 0.04) / 0.96
 *
 * which is `di_diffuse_tint`. It is 1 for a dielectric and 0 for a white metal
 * — a pure metal has no diffuse lobe, and the hook says so rather than leaving
 * `(1 - 0.04)` standing where `(1 - F0)` belonged. Skipping it (on the grounds
 * that `(1 - metallic)` is already there) costs ~9% at metallic 0.5, which is
 * the error the energy test caught.
 *
 * SPECULAR. Here `f` does NOT cancel, which is what the alpha lane is for:
 *
 *     F_true / F_d = 1 + metallic * (albedo - 0.04) * (1 - f) / (0.04 + 0.96 f)
 *                  = lerp(1, albedoTintFactor, metallic)
 *
 * which is the `lerp(1.0, albedo_tint_factor, metallic)` hook the plan calls
 * for, with the honest tint factor rather than a chosen one. At normal
 * incidence (f = 0) it is `albedo / 0.04`; at grazing (f -> 1) it relaxes to 1,
 * which is correct — every material's Fresnel goes to white at the horizon, so
 * a metal is NOT 25x brighter there and a fixed `albedo / 0.04` would say it
 * was. The denominator is bounded below by 0.04, so there is no division
 * hazard anywhere in the domain.
 *
 * The one place the specular half stops being exact is a pixel whose reservoir
 * alternates between lights with different `LdotH`: the denoiser averages
 * `specDemod` and `f` with the same weights, so the product of the two means
 * differs from the mean of the products by their covariance. Both factors are
 * smooth functions of geometry and the reservoir is dominated by one light at
 * almost every pixel, so this is a second-order term on a signal that is
 * stochastic to begin with — but it is an approximation and it is named here
 * rather than buried. The diffuse half has no such term: its hook does not read
 * the alpha lane at all.
 *
 * Tier: pure (no state, no device, callable from anywhere).
 */

#include <algorithm>
#include <cmath>

namespace bracken::render::lighting {

/// Dielectric normal-incidence reflectance, the F0 the trace assumes because
/// the G-buffer has no albedo. Same constant `pbr.hlsli` lerps from.
inline constexpr float DI_DIELECTRIC_F0 = 0.04f;

/// The Schlick fraction of a half-vector angle: `(1 - LdotH)^5`. Split out of
/// the Fresnel so the trace can store it and the forward can rebuild F exactly.
[[nodiscard]] inline float di_schlick_fraction(float ldoth) noexcept {
    const float x = std::clamp(1.0f - ldoth, 0.0f, 1.0f);
    const float x2 = x * x;
    return x2 * x2 * x;
}

/// Scalar dielectric Fresnel — what the trace multiplied the specular lobe by.
[[nodiscard]] inline float di_dielectric_fresnel(float schlick_fraction) noexcept {
    return DI_DIELECTRIC_F0 + (1.0f - DI_DIELECTRIC_F0) * schlick_fraction;
}

/**
 * One channel of the DIFFUSE remodulation factor: `(1 - F_true) / (1 - F_d)`.
 *
 * Needs no half-vector angle — Schlick's `(1 - f)` factors out of the ratio —
 * so unlike the specular hook this one reads nothing the denoiser touched.
 * 1 for a dielectric, 0 for a white metal (which has no diffuse lobe at all).
 * Clamped at zero: an albedo above 1 is not physical but a host can supply one
 * through `baseColorFactor`, and negative diffuse is worse than clipped
 * diffuse.
 */
[[nodiscard]] inline float di_diffuse_tint(float albedo, float metallic) noexcept {
    const float m = std::clamp(metallic, 0.0f, 1.0f);
    return std::max(1.0f - m * (albedo - DI_DIELECTRIC_F0) / (1.0f - DI_DIELECTRIC_F0), 0.0f);
}

/**
 * One channel of the specular remodulation factor: `F_true / F_dielectric`.
 *
 * `albedo` and `metallic` are the forward pixel's own; `schlick_fraction` is
 * the value the trace stored in the specular buffer's alpha. Returns 1 exactly
 * for a dielectric (metallic = 0), which is what makes the Full tier's
 * non-metal pixels agree with the analytic loop to the last bit of the trace's
 * own precision.
 */
[[nodiscard]] inline float di_specular_tint(float albedo, float metallic,
                                            float schlick_fraction) noexcept {
    const float f = std::clamp(schlick_fraction, 0.0f, 1.0f);
    const float denom = di_dielectric_fresnel(f);
    const float tint = 1.0f + (albedo - DI_DIELECTRIC_F0) * (1.0f - f) / denom;
    return 1.0f + std::clamp(metallic, 0.0f, 1.0f) * (tint - 1.0f);
}

// ---------------------------------------------------------------------------
// THE DIRECTIONAL BASIS (R7b, docs/rt_lighting_plan.md §5.7's first follow-up)
// ---------------------------------------------------------------------------
//
// WHAT R5c COST, restated as arithmetic. Everything above bakes `NdotL` into
// `E`, and the `N` in it is the PREPASS G-BUFFER'S — the geometric world
// normal, because that is the only normal a compute pass reading a G-buffer
// has. The forward shader has a better one: the normal-mapped `N` it computes
// per texel from a tangent-space map. Under a flashlight at night this is the
// whole difference between cobblestones and a photograph of cobblestones, and
// R5c's own A/B named it as the tier's principal cost.
//
// THE FIX. The trace publishes the DIRECTION it shaded with, and the forward
// pass applies a RELIEF FACTOR — the ratio of the cosine it can compute to the
// cosine the trace already applied:
//
//     direction = L, the winning light's unit direction at the traced texel
//     relief    = clamp( max(dot(N_shading, L), 0)
//                        / max(dot(N_geometric, L), DI_RELIEF_MIN_COSINE),
//                        0, DI_RELIEF_MAX )
//     out       = relief * (diffuse * albedo * tint_d + spec * tint_s)
//
// `N_geometric` is the interpolated vertex normal the fragment shader already
// has — the SAME quantity the prepass wrote into the G-buffer, so the
// denominator reconstructs the factor the trace applied — and `N_shading` is the
// normal-mapped one. The two demodulated buffers are UNCHANGED from R5c, down to
// the bit: `E` still carries its cosine, the denoiser still sees the same
// distribution, and every statistic ever measured about them still holds.
//
// A RATIO RATHER THAN MOVING THE COSINE, and this is the design decision worth
// defending. The obvious formulation is to stop multiplying `NdotL` in at the
// trace and multiply it in at the forward pass instead. It is equivalent on
// paper, it was implemented and measured, and it was rejected: dividing the
// cosine out inflates the stored estimate by `1/cos`, which is UNBOUNDED at a
// grazing light. Every texel whose geometric normal barely sees the emitter then
// carries a huge demodulated value into a denoiser whose entire job is to
// average it with its neighbours — so the tier gets noisier at exactly the
// angles it was supposed to be fixed at. The playground's night village measured
// **38.2 -> 43.7 mean luma** under that formulation: a systematic brightening
// rather than a wash, because the clamp at zero truncates only the darkening
// half of the ratio.
//
// The ratio form keeps the inflation OUT of the denoised signal and applies it
// per pixel, at full resolution, where it can be bounded.
//
// THE ENERGY IDENTITY IS THEREFORE TRIVIAL and exact ABOVE THE FLOOR: an
// unbumped texel has `N_shading == N_geometric`, and wherever
// `dot(N_geometric, L) >= DI_RELIEF_MIN_COSINE` the numerator and denominator
// are the same number, `relief == 1`, and the whole expression is R5c's term for
// term. That is what `di_relief_factor` is tested against, and it is why terrain
// and grass — which carry no normal map — come out of this phase bit-identical
// in both arithmetic and pixels.
//
// BELOW the floor the identity does not hold, and the reason is structural
// rather than incidental: `DI_RELIEF_MIN_COSINE` sits in the DENOMINATOR ONLY,
// so once the geometric cosine drops under it the denominator stops tracking the
// numerator. An equal-normals texel then fades linearly, `c / 0.05`, over
// `[0, 0.05]`, and is gated to zero at `c <= 0`. That is approximation 5 below —
// bounded, one-sided (it can only darken) and pinned by
// `local_di.test.cpp`'s `EqualNormalsGiveExactlyUnity`, which loops down to the
// floor inclusive and then asserts the fade explicitly. A flat-normal material
// at a grazing light therefore loses the tail of its Full-tier local term rather
// than keeping R5c's; it is the same direction of error as approximation 3, at
// the same angles, for the same reason.
//
// WHAT THE TWO COSINES MEAN, precisely, because they are not the same quantity:
//
//   * the TRACED cosine is `dot(N_geometric, L)` at the traced texel. It is the
//     one the reservoir's target function scored candidates with, so it decides
//     WHICH LIGHT the pixel gets and whether it gets one at all.
//   * the FULL-RES cosine is `dot(N_shading, L)` at the shading pixel. It
//     decides HOW BRIGHT that light's contribution is.
//
// Splitting the two is the entire technique, and it is exactly what a
// directional lightmap does. It also fixes the responsibility boundary: the
// trace answers "which light, and is it visible", the forward answers "how does
// this texel face it".
//
// THE APPROXIMATIONS, all five, named rather than buried:
//
//  1. ONLY THE COSINE IS CORRECTED. `D`, `Vis`, the Burley shape and the
//     Schlick fraction `f` stay evaluated against the geometric normal at
//     traced resolution. Re-deriving them at full res would make the specular
//     lobe's SHAPE track the normal map through a half-resolution direction
//     estimate — a sharper signal reconstructed from a blurrier one. The cosine
//     is the term with the first-order per-texel variation under a bump map; the
//     lobe shape is second order. A deliberate stopping point, not an oversight.
//  2. THE RATIO IS CLAMPED. `DI_RELIEF_MIN_COSINE` in the denominator and
//     `DI_RELIEF_MAX` on the result cap the brightening a normal map may claim.
//     Physically a facet turned from grazing to head-on IS enormously brighter,
//     and the cap says so only up to a point: beyond it the thing being scaled
//     is a half-resolution stochastic estimate, and amplifying it further
//     amplifies its error rather than its detail.
//  3. A LIGHT THE GEOMETRIC NORMAL CANNOT SEE IS STILL DARK. The reservoir
//     scores candidates by the geometric `NdotL`, so a facet whose geometric
//     normal faces away from every light selects nothing and the estimate is
//     zero however the normal map tilts. Relief is restored WITHIN the lit set,
//     never conjured outside it. The failure looks like a bump that should catch
//     a grazing light and does not — the opposite of the leak the alternative
//     would produce, and the safe direction to be wrong in.
//  4. DIRECTION DISCONTINUITIES. Directions do not average: two lights 90
//     degrees apart average to a direction neither occupies, and two at 180
//     degrees average to nothing at all. Every filter in this chain therefore
//     averages DECODED vectors and RENORMALIZES, and a degenerate sum falls back
//     to the surface normal — which makes both cosines equal, hence `relief`
//     exactly 1, and degrades the pixel to R5c's flat behaviour rather than to
//     black. The visible artefact is a softened, slightly wrong bump direction on
//     the few texels where the reservoir genuinely disagrees across the filter
//     footprint; it is bounded, local, and strictly better than no direction.
//  5. THE SUB-FLOOR BAND IS NOT THE IDENTITY. Because the floor is in the
//     denominator only, a texel whose GEOMETRIC cosine is under
//     `DI_RELIEF_MIN_COSINE` gets `relief < 1` even where the normal map is
//     flat, fading to 0 at 0. The lost energy is the tail of a term the trace
//     itself barely applied — `E` already carries that same small cosine — so
//     the absolute error is small, and it darkens rather than brightens, which
//     is the safe half. It is written down here because "relief is exactly 1 for
//     equal normals" is the sentence a future change to this file will be
//     checked against, and unqualified it is false.

/// Floor under the geometric cosine in the relief denominator. Not an epsilon
/// for numerical safety — 0.05 is orders of magnitude above any float hazard —
/// but the point at which "the trace barely lit this texel" stops being a number
/// worth dividing by. With DI_RELIEF_MAX it bounds the amplification at a
/// grazing light, which is the failure mode that killed the divide-the-cosine-out
/// form.
///
/// It has a SECOND effect, and it is not free: living in the denominator alone,
/// it also stops the equal-normals case cancelling, so an un-normal-mapped texel
/// under this cosine fades as `c / 0.05` instead of holding at 1 (approximation
/// 5 above). And it is what keeps `0/0` out of the expression at a fully grazing
/// texel — `clamp` of a NaN is not reliably 0 in HLSL, so the floor is
/// load-bearing numerically as well as perceptually.
inline constexpr float DI_RELIEF_MIN_COSINE = 0.05f;


/// Ceiling on the relief factor. 4x is two stops: enough that a cobble edge
/// turned toward a flashlight reads as a lit edge, little enough that a
/// half-resolution 1-spp estimate is never multiplied into a firefly. The
/// darkening side is deliberately unbounded below — scaling an estimate toward
/// zero cannot amplify its error.
inline constexpr float DI_RELIEF_MAX = 4.0f;

/// Below this squared length a filtered direction sum has cancelled itself out
/// and names no direction. The consumer substitutes the surface normal, which
/// makes the re-applied cosine 1 — R5c's behaviour, which is the right thing to
/// degrade to. Mirrors `bentNormalize`'s test in bent_cone.hlsli, and for the
/// same reason: both average unit vectors and both must survive them cancelling.
inline constexpr float DI_DIRECTION_MIN_LENGTH_SQ = 1e-8f;

/**
 * The relief factor: how much brighter (or darker) this texel's own normal makes
 * the traced estimate.
 *
 * `ndotl_geometric` is `dot(N_geometric, L)` — the cosine the trace already
 * applied, recomputed at full resolution from the interpolated vertex normal and
 * the denoised direction. `ndotl_shading` is the normal-mapped one. Exactly 1
 * when they agree AND `ndotl_geometric >= DI_RELIEF_MIN_COSINE`, which is the
 * energy identity and the reason terrain, grass and every un-normal-mapped
 * material are bit-identical to R5c wherever the light is not grazing. Below the
 * floor equal normals no longer cancel — it is in the denominator only — so the
 * factor fades to 0 across `[0, DI_RELIEF_MIN_COSINE]` and is 0 at or below
 * zero; `local_di.test.cpp` pins both the identity and the fade.
 */
[[nodiscard]] inline float di_relief_factor(float ndotl_geometric,
                                            float ndotl_shading) noexcept {
    const float denom = std::max(ndotl_geometric, DI_RELIEF_MIN_COSINE);
    return std::clamp(std::max(ndotl_shading, 0.0f) / denom, 0.0f, DI_RELIEF_MAX);
}

/**
 * One channel of `localDiLighting`: the whole Full-tier local-light term, given
 * a reconstructed estimate.
 *
 * `diffuse_demod` and `spec_demod` are the demodulated values the trace wrote,
 * unchanged from R5c; `relief` is `di_relief_factor`. Passing 1 reproduces R5c
 * exactly, which is the identity `local_di.test.cpp` pins.
 *
 * CPU mirror of `localDiRemodulate` in shaders/local_di.hlsli.
 */
[[nodiscard]] inline float di_remodulate_channel(float diffuse_demod, float spec_demod,
                                                 float albedo, float metallic, float schlick,
                                                 float relief) noexcept {
    return relief * (diffuse_demod * albedo * di_diffuse_tint(albedo, metallic) +
                     spec_demod * di_specular_tint(albedo, metallic, schlick));
}

// ---------------------------------------------------------------------------
// The denoiser's spatial radius curve
// ---------------------------------------------------------------------------

/**
 * Accumulated sample count at which the DI spatial pass stops widening.
 *
 * Eight rather than the visibility filter's `count / 8` confidence ramp read
 * against a variance — this signal has no second moment to read. A radiance
 * estimate from eight temporally accumulated 1-spp samples is roughly a third
 * as noisy as one, which is where the residual stops being the dominant error
 * next to the half-resolution reconstruction the forward pass does anyway.
 */
inline constexpr float DI_FILTER_CONVERGED_COUNT = 8.0f;

/**
 * Floor under the radius, in TRACED texels, however converged the pixel is.
 *
 * Not zero, and deliberately: unlike the visibility signals, which converge to
 * a binary answer the tile classifier can certify as uniform, a 1-spp radiance
 * estimate keeps a residual forever. One texel is a 3x3 edge-aware gather at
 * traced resolution, which the forward's depth-aware bilinear was going to
 * blend across anyway — it costs no sharpness a viewer can point at and it is
 * what stops converged surfaces shimmering.
 */
inline constexpr float DI_FILTER_MIN_RADIUS = 1.0f;

/**
 * Spatial filter radius for a pixel with `count` accumulated samples, in traced
 * texels. Young history filters wide; converged history filters at the floor.
 *
 * Linear in the count rather than in its square root: the point is not to track
 * the standard error (which the temporal pass is already doing) but to hide the
 * first frames after a disocclusion or a history reset, and those are exactly
 * the frames where `count` is small.
 */
[[nodiscard]] inline float di_filter_radius(float count, float max_radius) noexcept {
    const float span = DI_FILTER_CONVERGED_COUNT - 1.0f;
    const float t = std::clamp((std::max(count, 1.0f) - 1.0f) / span, 0.0f, 1.0f);
    const float wide = std::max(max_radius, DI_FILTER_MIN_RADIUS);
    return DI_FILTER_MIN_RADIUS + (wide - DI_FILTER_MIN_RADIUS) * (1.0f - t);
}

} // namespace bracken::render::lighting
