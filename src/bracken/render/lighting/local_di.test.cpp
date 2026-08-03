// The energy identity that keeps the Ratio and Full tiers honest against each
// other, plus the two smaller curves R5c introduced.
//
// The claim under test is the one that makes the Full-DI tier a different way of
// spending the same ray rather than a different picture: for one light, one
// pixel, full visibility and a reservoir weight of 1, the DEMODULATED SPLIT
// REMODULATED must reproduce `pbr.hlsli`'s own `evaluatePBR` — which is exactly
// what the analytic cluster loop the Full tier deletes was computing.
//
// The reference below is an INDEPENDENT transliteration of that shader
// function, written out longhand rather than factored through
// render/lighting/local_di.hpp, so the test cannot pass by sharing the bug.

#include "bracken/render/lighting/local_di.hpp"

#include <gtest/gtest.h>

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <cmath>

namespace {

using bracken::render::lighting::di_dielectric_fresnel;
using bracken::render::lighting::di_diffuse_tint;
using bracken::render::lighting::di_filter_radius;
using bracken::render::lighting::di_schlick_fraction;
using bracken::render::lighting::di_specular_tint;
using bracken::render::lighting::DI_DIELECTRIC_F0;
using bracken::render::lighting::DI_FILTER_CONVERGED_COUNT;
using bracken::render::lighting::DI_FILTER_MIN_RADIUS;

constexpr float PI_PBR = 3.14159265358979323846f;

// ---------------------------------------------------------------------------
// The reference: shaders/pbr.hlsli's evaluatePBR, transliterated longhand.
// ---------------------------------------------------------------------------

float ndf_ggx(float ndoth, float roughness) {
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float d = (ndoth * ndoth) * (a2 - 1.0f) + 1.0f;
    return a2 / std::max(PI_PBR * d * d, 1e-7f);
}

float visibility_smith_ggx(float ndotv, float ndotl, float roughness) {
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float ggx_v = ndotl * std::sqrt(ndotv * ndotv * (1.0f - a2) + a2);
    const float ggx_l = ndotv * std::sqrt(ndotl * ndotl * (1.0f - a2) + a2);
    return 0.5f / std::max(ggx_v + ggx_l, 1e-5f);
}

glm::vec3 fresnel_schlick(float cos_theta, const glm::vec3& f0) {
    const float f = std::pow(1.0f - cos_theta, 5.0f);
    return f0 + (glm::vec3(1.0f) - f0) * f;
}

glm::vec3 diffuse_burley(const glm::vec3& base_color, float roughness, float ndotv, float ndotl,
                         float ldoth) {
    const float fd90 = 0.5f + 2.0f * ldoth * ldoth * roughness;
    const float light_scatter = 1.0f + (fd90 - 1.0f) * std::pow(1.0f - ndotl, 5.0f);
    const float view_scatter = 1.0f + (fd90 - 1.0f) * std::pow(1.0f - ndotv, 5.0f);
    return base_color * (light_scatter * view_scatter / PI_PBR);
}

/// `evaluatePBR(N, V, L, lightColor, baseColor, metallic, roughness, ao = 1)`.
glm::vec3 evaluate_pbr(const glm::vec3& n, const glm::vec3& v, const glm::vec3& l,
                       const glm::vec3& light_color, const glm::vec3& base_color, float metallic,
                       float roughness) {
    const float ndotv = std::max(glm::dot(n, v), 1e-4f);
    const float ndotl = std::max(glm::dot(n, l), 0.0f);
    if (ndotl <= 0.0f) {
        return glm::vec3(0.0f);
    }
    const glm::vec3 h = glm::normalize(v + l);
    const float ndoth = std::max(glm::dot(n, h), 0.0f);
    const float ldoth = std::max(glm::dot(l, h), 0.0f);

    const glm::vec3 f0 = glm::mix(glm::vec3(DI_DIELECTRIC_F0), base_color, metallic);
    const float d = ndf_ggx(ndoth, roughness);
    const float vis = visibility_smith_ggx(ndotv, ndotl, roughness);
    const glm::vec3 f = fresnel_schlick(ldoth, f0);

    const glm::vec3 spec = glm::vec3(d * vis) * f;
    const glm::vec3 kd = (glm::vec3(1.0f) - f) * (1.0f - metallic);
    const glm::vec3 diff = kd * diffuse_burley(base_color, roughness, ndotv, ndotl, ldoth);
    return (diff + spec) * light_color * ndotl;
}

// ---------------------------------------------------------------------------
// The thing under test: the trace's split, and the forward pass's rebuild.
// ---------------------------------------------------------------------------

struct DemodulatedEstimate {
    glm::vec3 diffuse{0.0f}; ///< what rt_local_shadow.comp writes to diDiffuseOut.rgb
    glm::vec3 specular{0.0f};///< ...and to diSpecularOut.rgb
    float schlick = 0.0f;    ///< ...and to diSpecularOut.a
};

/// The DI block of `shaders/rt_local_shadow.comp.hlsl`, which knows roughness
/// and metallic from the G-buffer and does NOT know the albedo.
DemodulatedEstimate demodulate(const glm::vec3& n, const glm::vec3& v, const glm::vec3& l,
                               const glm::vec3& light_radiance, float metallic, float roughness,
                               float visibility, float reservoir_w) {
    DemodulatedEstimate out;
    const float ndotv = std::max(glm::dot(n, v), 1e-4f);
    const float ndotl = std::max(glm::dot(n, l), 0.0f);
    if (ndotl <= 0.0f) {
        return out;
    }
    const glm::vec3 h = glm::normalize(v + l);
    const float ndoth = std::max(glm::dot(n, h), 0.0f);
    const float ldoth = std::max(glm::dot(l, h), 0.0f);

    const float schlick = di_schlick_fraction(ldoth);
    const float fd = di_dielectric_fresnel(schlick);
    const float d = ndf_ggx(ndoth, roughness);
    const float vis = visibility_smith_ggx(ndotv, ndotl, roughness);
    const float burley = diffuse_burley(glm::vec3(1.0f), roughness, ndotv, ndotl, ldoth).x;

    const glm::vec3 e = light_radiance * (ndotl * visibility * reservoir_w);
    out.diffuse = ((1.0f - fd) * (1.0f - metallic) * burley) * e;
    out.specular = (d * vis * fd) * e;
    out.schlick = schlick;
    return out;
}

/// `localDiLighting`'s remodulation, given a perfectly reconstructed estimate.
glm::vec3 remodulate(const DemodulatedEstimate& est, const glm::vec3& albedo, float metallic) {
    const glm::vec3 spec_tint{di_specular_tint(albedo.x, metallic, est.schlick),
                              di_specular_tint(albedo.y, metallic, est.schlick),
                              di_specular_tint(albedo.z, metallic, est.schlick)};
    const glm::vec3 diffuse_tint{di_diffuse_tint(albedo.x, metallic),
                                 di_diffuse_tint(albedo.y, metallic),
                                 di_diffuse_tint(albedo.z, metallic)};
    return est.diffuse * albedo * diffuse_tint + est.specular * spec_tint;
}

struct Sample {
    glm::vec3 n;
    glm::vec3 v;
    glm::vec3 l;
    glm::vec3 albedo;
    float metallic;
    float roughness;
    const char* name;
};

// A spread of surfaces rather than one: the whole point of the split is that it
// holds for metals AND dielectrics, at normal incidence AND at grazing, where a
// naive `albedo / 0.04` tint holds only at the first of each.
const Sample SAMPLES[] = {
    {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0.8f, 0.7f, 0.6f}, 0.0f, 0.5f, "dielectric, head on"},
    {{0, 1, 0}, glm::normalize(glm::vec3(1, 1, 0)), glm::normalize(glm::vec3(-1, 1, 0)),
     {0.8f, 0.2f, 0.1f}, 0.0f, 0.25f, "dielectric, oblique"},
    {{0, 1, 0}, glm::normalize(glm::vec3(4, 1, 0)), glm::normalize(glm::vec3(-4, 1, 0)),
     {0.9f, 0.9f, 0.9f}, 0.0f, 0.1f, "dielectric, grazing"},
    {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0.95f, 0.75f, 0.35f}, 1.0f, 0.2f, "gold, head on"},
    {{0, 1, 0}, glm::normalize(glm::vec3(4, 1, 0)), glm::normalize(glm::vec3(-4, 1, 0)),
     {0.95f, 0.75f, 0.35f}, 1.0f, 0.3f, "gold, grazing"},
    {{0, 1, 0}, glm::normalize(glm::vec3(1, 2, 0)), glm::normalize(glm::vec3(-1, 3, 1)),
     {0.5f, 0.5f, 0.55f}, 0.5f, 0.6f, "half metal, asymmetric"},
    {{0, 1, 0}, glm::normalize(glm::vec3(0, 1, 1)), glm::normalize(glm::vec3(0, 1, -1)),
     {0.05f, 0.05f, 0.05f}, 1.0f, 0.9f, "dark rough metal"},
};

} // namespace

// THE test. Demodulate, then remodulate, and land on the analytic per-light
// evaluation the Full tier deleted from the forward loop.
//
// The tolerance is RELATIVE and it is 1e-5, i.e. float round-off, not a fudge
// band: with `visibility = 1` and `W = 1` — the noise-free case — the identity
// is exact algebra, including for metals, because the specular tint rebuilds
// the true Fresnel from the stored Schlick fraction instead of approximating it
// at normal incidence. A tier that had to approximate here would need a
// percent-level band and would show it as a colour shift on every metal.
TEST(LocalDi, DemodulatedSplitReproducesTheAnalyticEvaluation) {
    const glm::vec3 radiance{3.0f, 2.5f, 1.5f}; // color * intensity * atten * cone
    for (const Sample& s : SAMPLES) {
        const glm::vec3 reference =
            evaluate_pbr(s.n, s.v, s.l, radiance, s.albedo, s.metallic, s.roughness);
        const DemodulatedEstimate est =
            demodulate(s.n, s.v, s.l, radiance, s.metallic, s.roughness, 1.0f, 1.0f);
        const glm::vec3 rebuilt = remodulate(est, s.albedo, s.metallic);

        for (int c = 0; c < 3; ++c) {
            const float scale = std::max(std::abs(reference[c]), 1e-4f);
            EXPECT_NEAR(rebuilt[c], reference[c], 1e-5f * scale)
                << s.name << ", channel " << c;
        }
    }
}

// The estimator is linear in the traced visibility bit and in the reservoir
// weight, which is what makes "the ratio tier's V" and "the full tier's W"
// interchangeable statements about the same integral. A pixel whose ray was
// occluded contributes exactly nothing, not "nearly nothing".
TEST(LocalDi, EstimateIsLinearInVisibilityAndReservoirWeight) {
    const Sample& s = SAMPLES[1];
    const glm::vec3 radiance{2.0f, 2.0f, 2.0f};

    const DemodulatedEstimate lit = demodulate(s.n, s.v, s.l, radiance, s.metallic, s.roughness, 1.0f, 1.0f);
    const DemodulatedEstimate occluded =
        demodulate(s.n, s.v, s.l, radiance, s.metallic, s.roughness, 0.0f, 1.0f);
    const DemodulatedEstimate weighted =
        demodulate(s.n, s.v, s.l, radiance, s.metallic, s.roughness, 1.0f, 2.5f);

    EXPECT_EQ(occluded.diffuse, glm::vec3(0.0f));
    EXPECT_EQ(occluded.specular, glm::vec3(0.0f));
    // The Schlick fraction is geometry and survives an occluded ray — which is
    // what lets the denoiser keep a usable alpha through a shadowed frame.
    EXPECT_FLOAT_EQ(occluded.schlick, lit.schlick);

    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(weighted.diffuse[c], lit.diffuse[c] * 2.5f, 1e-5f);
        EXPECT_NEAR(weighted.specular[c], lit.specular[c] * 2.5f, 1e-5f);
    }
}

// The diffuse half carries no albedo at all. Stated as a test because it is the
// property the whole technique rests on: it is what lets the denoiser blur
// lighting without blurring material.
TEST(LocalDi, DemodulatedDiffuseIsIndependentOfAlbedo) {
    const Sample& s = SAMPLES[0];
    const glm::vec3 radiance{1.0f, 1.0f, 1.0f};
    const DemodulatedEstimate a =
        demodulate(s.n, s.v, s.l, radiance, /*metallic=*/0.0f, s.roughness, 1.0f, 1.0f);
    // Same geometry, same material factors the G-buffer carries; the albedo is
    // not an input to `demodulate` at all, so there is nothing to vary — the
    // test is that remodulating with two albedos scales linearly.
    const glm::vec3 dark = remodulate(a, glm::vec3(0.1f), 0.0f);
    const glm::vec3 bright = remodulate(a, glm::vec3(0.8f), 0.0f);
    // Only the diffuse half scales with albedo; the specular half of a
    // dielectric does not (its tint is exactly 1), so the ratio is not 8.
    EXPECT_GT(bright.x, dark.x);
    const glm::vec3 diffuse_only_dark = a.diffuse * 0.1f;
    const glm::vec3 diffuse_only_bright = a.diffuse * 0.8f;
    EXPECT_NEAR(bright.x - dark.x, diffuse_only_bright.x - diffuse_only_dark.x, 1e-6f);
}

// ---------------------------------------------------------------------------
// The specular remodulation hook
// ---------------------------------------------------------------------------

TEST(LocalDi, DiffuseTintIsOneForADielectricAndZeroForAWhiteMetal) {
    EXPECT_FLOAT_EQ(di_diffuse_tint(0.9f, 0.0f), 1.0f);
    EXPECT_FLOAT_EQ(di_diffuse_tint(0.05f, 0.0f), 1.0f);
    // A pure metal has no diffuse lobe. Leaving `(1 - 0.04)` standing where
    // `(1 - F0)` belonged would say it had 4% of one.
    EXPECT_FLOAT_EQ(di_diffuse_tint(1.0f, 1.0f), 0.0f);
    // ...and an out-of-range albedo clips rather than going negative.
    EXPECT_FLOAT_EQ(di_diffuse_tint(1.5f, 1.0f), 0.0f);
}

TEST(LocalDi, DiffuseTintIsMonotoneDecreasingInMetallic) {
    float previous = di_diffuse_tint(0.8f, 0.0f);
    EXPECT_FLOAT_EQ(previous, 1.0f);
    for (float m = 0.1f; m <= 1.0f; m += 0.1f) {
        const float tint = di_diffuse_tint(0.8f, m);
        EXPECT_LE(tint, previous + 1e-6f);
        previous = tint;
    }
}

TEST(LocalDi, SpecularTintIsExactlyOneForADielectric) {
    for (float f : {0.0f, 0.1f, 0.5f, 1.0f}) {
        EXPECT_FLOAT_EQ(di_specular_tint(0.9f, 0.0f, f), 1.0f);
        EXPECT_FLOAT_EQ(di_specular_tint(0.02f, 0.0f, f), 1.0f);
    }
}

TEST(LocalDi, SpecularTintAtNormalIncidenceIsTheF0Ratio) {
    // f = 0 is normal incidence, where F reduces to F0 and the tint is exactly
    // the metal's F0 over the dielectric one — the `albedo / 0.04` figure.
    EXPECT_NEAR(di_specular_tint(0.9f, 1.0f, 0.0f), 0.9f / DI_DIELECTRIC_F0, 1e-4f);
    EXPECT_NEAR(di_specular_tint(0.04f, 1.0f, 0.0f), 1.0f, 1e-6f);
}

TEST(LocalDi, SpecularTintRelaxesToOneAtGrazing) {
    // Every material's Fresnel goes to white at the horizon, so a metal is NOT
    // 22x brighter there. A fixed `albedo / 0.04` hook would say it was, and
    // this is the test that would fail if someone replaced the exact form with
    // that one.
    EXPECT_NEAR(di_specular_tint(0.9f, 1.0f, 1.0f), 1.0f, 1e-6f);
    EXPECT_LT(di_specular_tint(0.9f, 1.0f, 0.9f), 0.25f * (0.9f / DI_DIELECTRIC_F0));
}

TEST(LocalDi, SpecularTintIsMonotoneInMetallic) {
    float previous = 0.0f;
    for (float m = 0.0f; m <= 1.0f; m += 0.1f) {
        const float tint = di_specular_tint(0.9f, m, 0.2f);
        EXPECT_GE(tint, previous);
        previous = tint;
    }
}

TEST(LocalDi, SchlickFractionSpansTheUnitInterval) {
    EXPECT_FLOAT_EQ(di_schlick_fraction(1.0f), 0.0f);
    EXPECT_FLOAT_EQ(di_schlick_fraction(0.0f), 1.0f);
    // Clamped rather than extrapolated: a reconstructed LdotH can leave [0, 1]
    // by a float hair, and a negative fifth power would flip the Fresnel.
    EXPECT_FLOAT_EQ(di_schlick_fraction(1.5f), 0.0f);
    EXPECT_FLOAT_EQ(di_schlick_fraction(-0.5f), 1.0f);
    EXPECT_GT(di_schlick_fraction(0.5f), 0.0f);
    EXPECT_LT(di_schlick_fraction(0.5f), 1.0f);
}

TEST(LocalDi, DielectricFresnelBracketsTheSchlickRange) {
    EXPECT_FLOAT_EQ(di_dielectric_fresnel(0.0f), DI_DIELECTRIC_F0);
    EXPECT_FLOAT_EQ(di_dielectric_fresnel(1.0f), 1.0f);
}

// ---------------------------------------------------------------------------
// The spatial filter's radius curve
// ---------------------------------------------------------------------------

TEST(LocalDi, FilterRadiusIsWidestForAOneSamplePixel) {
    EXPECT_FLOAT_EQ(di_filter_radius(1.0f, 3.0f), 3.0f);
    // ...and a count below 1 cannot widen it further; the curve is clamped, not
    // extrapolated.
    EXPECT_FLOAT_EQ(di_filter_radius(0.0f, 3.0f), 3.0f);
    EXPECT_FLOAT_EQ(di_filter_radius(-5.0f, 3.0f), 3.0f);
}

TEST(LocalDi, FilterRadiusReachesTheFloorAtConvergence) {
    EXPECT_FLOAT_EQ(di_filter_radius(DI_FILTER_CONVERGED_COUNT, 3.0f), DI_FILTER_MIN_RADIUS);
    // ...and stays there. A pixel with 32 accumulated samples must not start
    // widening again.
    EXPECT_FLOAT_EQ(di_filter_radius(32.0f, 3.0f), DI_FILTER_MIN_RADIUS);
}

TEST(LocalDi, FilterRadiusIsMonotoneDecreasingInCount) {
    float previous = di_filter_radius(1.0f, 4.0f);
    for (float count = 2.0f; count <= 16.0f; count += 1.0f) {
        const float radius = di_filter_radius(count, 4.0f);
        EXPECT_LE(radius, previous);
        EXPECT_GE(radius, DI_FILTER_MIN_RADIUS);
        previous = radius;
    }
}

TEST(LocalDi, FilterRadiusNeverGoesBelowTheFloorEvenWithASmallerMax) {
    // A host that dials `di_filter_radius` down to zero still gets the floor:
    // a 1-spp radiance estimate keeps a residual forever, unlike the binary
    // visibility signals whose tile classifier can certify convergence.
    EXPECT_FLOAT_EQ(di_filter_radius(1.0f, 0.0f), DI_FILTER_MIN_RADIUS);
    EXPECT_FLOAT_EQ(di_filter_radius(100.0f, 0.0f), DI_FILTER_MIN_RADIUS);
}

// ---------------------------------------------------------------------------
// The directional basis (R7b, docs/rt_lighting_plan.md §5.7)
// ---------------------------------------------------------------------------
//
// The trace publishes the winning light's DIRECTION, and the forward pass scales
// its estimate by a RELIEF FACTOR — the ratio of the cosine it can compute from
// its normal-mapped normal to the one the trace already applied from the
// geometric one. What these tests pin is that the correction is exactly 1 where
// the normal map is flat (so R5c's whole energy argument survives untouched),
// that it moves in the right direction and by the right amount where it is not,
// and that it is BOUNDED — the bound being the thing that makes this formulation
// preferable to dividing the cosine out of the trace.

namespace {

using bracken::render::lighting::di_relief_factor;
using bracken::render::lighting::di_remodulate_channel;
using bracken::render::lighting::DI_DIRECTION_MIN_LENGTH_SQ;
using bracken::render::lighting::DI_RELIEF_MAX;
using bracken::render::lighting::DI_RELIEF_MIN_COSINE;

/// `localDiLighting`'s remodulation, given a perfectly reconstructed estimate,
/// the denoised direction, and the two normals.
glm::vec3 remodulate_directional(const DemodulatedEstimate& est, const glm::vec3& albedo,
                                 float metallic, const glm::vec3& light_dir,
                                 const glm::vec3& geometric_normal,
                                 const glm::vec3& shading_normal) {
    const float relief =
        di_relief_factor(glm::dot(geometric_normal, light_dir), glm::dot(shading_normal, light_dir));
    glm::vec3 out(0.0f);
    for (int c = 0; c < 3; ++c) {
        out[c] = di_remodulate_channel(est.diffuse[c], est.specular[c], albedo[c], metallic,
                                       est.schlick, relief);
    }
    return out;
}

} // namespace

// THE ENERGY IDENTITY, restated for the directional basis and now trivial: an
// unbumped texel has the two normals equal, the relief is exactly 1, and the
// expression is R5c's term for term. Same 1e-5 relative band, because it is the
// same arithmetic with a factor of one in it.
TEST(LocalDiDirectional, FlatNormalsReproduceTheAnalyticEvaluation) {
    const glm::vec3 radiance{3.0f, 2.5f, 1.5f};
    for (const Sample& s : SAMPLES) {
        const glm::vec3 reference =
            evaluate_pbr(s.n, s.v, s.l, radiance, s.albedo, s.metallic, s.roughness);
        const DemodulatedEstimate est =
            demodulate(s.n, s.v, s.l, radiance, s.metallic, s.roughness, 1.0f, 1.0f);
        const glm::vec3 rebuilt =
            remodulate_directional(est, s.albedo, s.metallic, s.l, s.n, /*shading=*/s.n);
        for (int c = 0; c < 3; ++c) {
            const float scale = std::max(std::abs(reference[c]), 1e-4f);
            EXPECT_NEAR(rebuilt[c], reference[c], 1e-5f * scale) << s.name << ", channel " << c;
        }
    }
}

// The relief factor is exactly 1 for equal normals — checked directly, across
// the whole range of incidence, because "exactly" is what terrain, grass and
// every un-normal-mapped material depend on to stay bit-identical to R5c.
TEST(LocalDiDirectional, EqualNormalsGiveExactlyUnity) {
    for (float c : {1.0f, 0.75f, 0.5f, 0.2f, DI_RELIEF_MIN_COSINE}) {
        EXPECT_FLOAT_EQ(di_relief_factor(c, c), 1.0f) << "cos " << c;
    }
    // Below the floor the ratio is no longer 1 — by construction, because the
    // denominator has stopped tracking the numerator. That is the clamp doing
    // its job and it must be visible in a test rather than discovered.
    EXPECT_LT(di_relief_factor(0.01f, 0.01f), 1.0f);
}

// THE POINT OF THE WHOLE PACKAGE, as a number: a texel whose normal map tilts it
// toward the light gets brighter, one tilted away gets darker, in proportion to
// the cosine ratio. Before R7b both read the geometric value and the surface was
// flat.
TEST(LocalDiDirectional, BumpedNormalsScaleByTheCosineRatio) {
    const Sample& s = SAMPLES[1]; // dielectric, oblique
    const glm::vec3 radiance{2.0f, 2.0f, 2.0f};
    const DemodulatedEstimate est =
        demodulate(s.n, s.v, s.l, radiance, s.metallic, s.roughness, 1.0f, 1.0f);

    const glm::vec3 flat = remodulate_directional(est, s.albedo, s.metallic, s.l, s.n, s.n);
    const glm::vec3 toward = glm::normalize(s.n + s.l * 0.35f);
    const glm::vec3 away = glm::normalize(s.n - s.l * 0.35f);
    const glm::vec3 bright = remodulate_directional(est, s.albedo, s.metallic, s.l, s.n, toward);
    const glm::vec3 dark = remodulate_directional(est, s.albedo, s.metallic, s.l, s.n, away);

    const float base = glm::dot(s.n, s.l);
    ASSERT_GT(base, DI_RELIEF_MIN_COSINE); // the ratio is unclamped here
    for (int c = 0; c < 3; ++c) {
        EXPECT_GT(bright[c], flat[c]);
        EXPECT_LT(dark[c], flat[c]);
        EXPECT_NEAR(bright[c], flat[c] * (glm::dot(toward, s.l) / base), 1e-5f);
        EXPECT_NEAR(dark[c], flat[c] * (glm::dot(away, s.l) / base), 1e-5f);
    }
}

// THE BOUND, which is the reason this formulation was chosen over dividing the
// cosine out of the trace. A geometric normal that barely sees the light and a
// normal map that faces it head-on would ask for an arbitrarily large
// amplification of a half-resolution 1-spp estimate; the clamp says 4x and no
// more. Both halves of the clamp are pinned, because a missing floor and a
// missing ceiling fail in different ways.
TEST(LocalDiDirectional, ReliefIsBoundedAboveAndTheDenominatorHasAFloor) {
    // Grazing geometry, head-on bump: unbounded without the clamp.
    EXPECT_FLOAT_EQ(di_relief_factor(1e-6f, 1.0f), DI_RELIEF_MAX);
    EXPECT_FLOAT_EQ(di_relief_factor(0.0f, 1.0f), DI_RELIEF_MAX);
    // Exactly at the floor, the ratio is 1 / DI_RELIEF_MIN_COSINE = 20, capped.
    EXPECT_FLOAT_EQ(di_relief_factor(DI_RELIEF_MIN_COSINE, 1.0f), DI_RELIEF_MAX);
    // Just inside the cap, it is the true ratio and not the cap.
    EXPECT_NEAR(di_relief_factor(0.25f, 0.5f), 2.0f, 1e-6f);
    EXPECT_LT(di_relief_factor(0.25f, 0.5f), DI_RELIEF_MAX);
    // The darkening half is deliberately unbounded below.
    EXPECT_NEAR(di_relief_factor(1.0f, 0.01f), 0.01f, 1e-6f);
}

// A texel the normal map turns fully away from the light contributes nothing,
// and contributes it as a CLAMP rather than as a negative. Negative direct
// lighting would subtract from the sun and the ambient at the same pixel, which
// is the one way this basis could make a frame worse than having no direction.
TEST(LocalDiDirectional, BackFacingTexelsClampToZeroRatherThanGoNegative) {
    EXPECT_FLOAT_EQ(di_relief_factor(0.8f, -0.75f), 0.0f);
    EXPECT_FLOAT_EQ(di_relief_factor(0.8f, 0.0f), 0.0f);
    EXPECT_FLOAT_EQ(di_remodulate_channel(5.0f, 5.0f, 0.5f, 0.0f, 0.1f, 0.0f), 0.0f);
}

// Approximation 3 of the four named in local_di.hpp: relief is restored WITHIN
// the lit set and never conjured outside it. A geometric normal facing away
// scores zero in the reservoir's target function, so the trace writes nothing
// and no amount of normal-map tilt brings the light back.
TEST(LocalDiDirectional, ALightTheGeometricNormalCannotSeeStaysDark) {
    const glm::vec3 n{0.0f, 1.0f, 0.0f};
    const glm::vec3 l = glm::normalize(glm::vec3(0.3f, -1.0f, 0.0f)); // below the horizon
    const glm::vec3 v = glm::normalize(glm::vec3(0.0f, 1.0f, 1.0f));
    const DemodulatedEstimate est = demodulate(n, v, l, glm::vec3(4.0f), 0.0f, 0.4f, 1.0f, 1.0f);
    EXPECT_EQ(est.diffuse, glm::vec3(0.0f));
    EXPECT_EQ(est.specular, glm::vec3(0.0f));
    // Even a normal map pointing straight at the light cannot resurrect it: the
    // estimate being scaled is zero.
    EXPECT_EQ(remodulate_directional(est, glm::vec3(0.8f), 0.0f, l, n, -l), glm::vec3(0.0f));
}

// Approximation 4: filtered directions cancel. The threshold has to be low
// enough that a genuine unit vector never trips it and high enough that two
// opposed taps do — this pins both ends, because the consumer's fallback
// (substitute the geometric normal, relief becomes exactly 1) is a visible
// behaviour change on any texel that reaches it.
TEST(LocalDiDirectional, CancelledDirectionSumsAreDetected) {
    const glm::vec3 a{0.0f, 1.0f, 0.0f};
    const glm::vec3 opposed = a + glm::vec3(0.0f, -1.0f, 0.0f);
    EXPECT_LT(glm::dot(opposed, opposed), DI_DIRECTION_MIN_LENGTH_SQ);

    // Two taps 179 degrees apart still name a direction, barely, and must not
    // fall back: the fallback is flatter than the answer it replaces.
    const glm::vec3 nearly = a + glm::normalize(glm::vec3(0.0175f, -1.0f, 0.0f));
    EXPECT_GT(glm::dot(nearly, nearly), DI_DIRECTION_MIN_LENGTH_SQ);

    // A single unit tap, the common case, is nowhere near the threshold.
    EXPECT_GT(glm::dot(a, a), DI_DIRECTION_MIN_LENGTH_SQ * 1e6f);
}
