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
