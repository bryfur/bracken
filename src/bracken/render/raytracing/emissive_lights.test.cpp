#include "bracken/render/raytracing/emissive_lights.hpp"

#include "bracken/render/lighting/alias_table.hpp"
#include "bracken/render/lighting/light_target.hpp"
#include "bracken/render/lighting/reservoir.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using bracken::render::lighting::build_alias_table;
using bracken::render::lighting::point_light_target;
using bracken::render::lighting::ris_source_scale;
using bracken::render::rt::emissive_sample_point;
using bracken::render::rt::emissive_shading;
using bracken::render::rt::emissive_target;
using bracken::render::rt::EmissiveTriangle;

namespace {

/// A unit-ish quad's worth of triangle: v0 at the origin, edges along +X and
/// +Z, so its normal is -Y or +Y depending on winding and its area is trivial
/// to state by hand.
EmissiveTriangle make_tri(glm::vec3 v0, glm::vec3 e1, glm::vec3 e2, glm::vec3 radiance) {
    EmissiveTriangle t;
    t.v0 = v0;
    t.e1 = e1;
    t.e2 = e2;
    t.radiance = radiance;
    t.area = 0.5f * glm::length(glm::cross(e1, e2));
    t.power = bracken::render::lighting::target_luminance(radiance) * t.area;
    return t;
}

/// Barycentrics of `p` against the triangle, for the containment assertions.
glm::vec3 barycentric(const EmissiveTriangle& t, const glm::vec3& p) {
    const glm::vec3 d = p - t.v0;
    const float d11 = glm::dot(t.e1, t.e1);
    const float d12 = glm::dot(t.e1, t.e2);
    const float d22 = glm::dot(t.e2, t.e2);
    const float dp1 = glm::dot(d, t.e1);
    const float dp2 = glm::dot(d, t.e2);
    const float denom = d11 * d22 - d12 * d12;
    const float b1 = (d22 * dp1 - d12 * dp2) / denom;
    const float b2 = (d11 * dp2 - d12 * dp1) / denom;
    return {1.0f - b1 - b2, b1, b2};
}

} // namespace

TEST(EmissiveLights, RecordIsTheGpuLayout) {
    // The shader declares this struct field for field
    // (shaders/rt_emissive.hlsli). A silent layout change here is the class of
    // bug CLAUDE.md names as the most common way to break rendering.
    EXPECT_EQ(sizeof(EmissiveTriangle), 64u);
    EXPECT_EQ(offsetof(EmissiveTriangle, area), 12u);
    EXPECT_EQ(offsetof(EmissiveTriangle, alias_prob), 28u);
    EXPECT_EQ(offsetof(EmissiveTriangle, alias_index), 44u);
    EXPECT_EQ(offsetof(EmissiveTriangle, power), 60u);
}

TEST(EmissiveLights, SamplePointStaysInsideTheTriangle) {
    const EmissiveTriangle t =
        make_tri({10.0f, 5.0f, -2.0f}, {4.0f, 0.0f, 1.0f}, {-1.0f, 0.0f, 3.0f}, {1.0f, 1.0f, 1.0f});
    for (int i = 0; i <= 32; ++i) {
        for (int j = 0; j <= 32; ++j) {
            const glm::vec2 xi{static_cast<float>(i) / 32.0f, static_cast<float>(j) / 32.0f};
            const glm::vec3 b = barycentric(t, emissive_sample_point(t, xi));
            EXPECT_GE(b.x, -1e-4f);
            EXPECT_GE(b.y, -1e-4f);
            EXPECT_GE(b.z, -1e-4f);
            EXPECT_NEAR(b.x + b.y + b.z, 1.0f, 1e-4f);
        }
    }
}

TEST(EmissiveLights, SamplePointIsUniformOverArea) {
    // Uniformity, tested where it is observable rather than by eye: the mean of
    // a uniform sample over a triangle is its CENTROID, and the mean of a
    // NON-uniform one (the naive un-square-rooted map, which biases toward v0)
    // is not. Integrating over a grid of the two uniforms rather than drawing
    // randoms keeps this exact and deterministic.
    const EmissiveTriangle t =
        make_tri({0.0f, 0.0f, 0.0f}, {6.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 9.0f}, {1.0f, 1.0f, 1.0f});
    constexpr int N = 200;
    glm::dvec3 sum(0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const glm::vec2 xi{(static_cast<float>(i) + 0.5f) / N, (static_cast<float>(j) + 0.5f) / N};
            sum += glm::dvec3(emissive_sample_point(t, xi));
        }
    }
    const glm::dvec3 mean = sum / static_cast<double>(N * N);
    const glm::vec3 centroid = t.v0 + (t.e1 + t.e2) / 3.0f;
    EXPECT_NEAR(mean.x, centroid.x, 2e-2);
    EXPECT_NEAR(mean.y, centroid.y, 2e-2);
    EXPECT_NEAR(mean.z, centroid.z, 2e-2);

    // And the second moment, which is what actually distinguishes uniform from
    // merely centroid-preserving: for a uniform triangle sample the variance of
    // a coordinate is (a^2 + b^2 + c^2 - ab - bc - ca)/18 over the vertices.
    double var = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const glm::vec2 xi{(static_cast<float>(i) + 0.5f) / N, (static_cast<float>(j) + 0.5f) / N};
            const double dx = emissive_sample_point(t, xi).x - mean.x;
            var += dx * dx;
        }
    }
    var /= static_cast<double>(N * N);
    const double a = 0.0;
    const double b = 6.0;
    const double c = 0.0;
    const double expected = (a * a + b * b + c * c - a * b - b * c - c * a) / 18.0;
    EXPECT_NEAR(var, expected, 5e-2);
}

TEST(EmissiveLights, EmissionIsTwoSided) {
    // Every forward shader in this tree adds emission regardless of facing, so
    // a record must light a receiver on either side. Swapping the two edges
    // flips the geometric normal and must change nothing.
    const glm::vec3 p{0.0f, 100.0f, 0.0f};
    const glm::vec3 n{0.0f, 1.0f, 0.0f}; // receiver faces the emitter above it
    const EmissiveTriangle front =
        make_tri({-10.0f, 200.0f, -10.0f}, {20.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 20.0f}, {5.0f, 5.0f, 5.0f});
    const EmissiveTriangle back =
        make_tri({-10.0f, 200.0f, -10.0f}, {0.0f, 0.0f, 20.0f}, {20.0f, 0.0f, 0.0f}, {5.0f, 5.0f, 5.0f});
    EXPECT_NEAR(emissive_target(front, p, n), emissive_target(back, p, n), 1e-6f);
    EXPECT_GT(emissive_target(front, p, n), 0.0f);
}

TEST(EmissiveLights, FallsOffAsInverseSquareInTheFarField) {
    const EmissiveTriangle t =
        make_tri({-5.0f, 0.0f, -5.0f}, {10.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 10.0f}, {3.0f, 3.0f, 3.0f});
    const glm::vec3 n{0.0f, 1.0f, 0.0f};
    // Far enough that the centroid approximation is the whole story.
    const float near_v = emissive_shading(t, glm::vec3(0.0f, -200.0f, 0.0f)).radiance.r;
    const float far_v = emissive_shading(t, glm::vec3(0.0f, -400.0f, 0.0f)).radiance.r;
    EXPECT_NEAR(near_v / far_v, 4.0f, 1e-3f);
    // Sanity: a receiver facing away gets nothing from the target function.
    EXPECT_EQ(emissive_target(t, glm::vec3(0.0f, -200.0f, 0.0f), -n), 0.0f);
}

TEST(EmissiveLights, SolidAngleClampBoundsTheNearField) {
    // Without the max(d^2, area) floor this diverges as the receiver
    // approaches, and one unlucky candidate becomes a firefly the denoiser then
    // spends frames eating. The clamped value can never exceed the emitter's
    // own radiance, which is the physical ceiling ("the emitter fills the
    // hemisphere").
    const EmissiveTriangle t =
        make_tri({-50.0f, 0.0f, -50.0f}, {100.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 100.0f}, {2.0f, 2.0f, 2.0f});
    for (float d = 0.01f; d < 40.0f; d *= 1.7f) {
        const auto s = emissive_shading(t, glm::vec3(0.0f, -d, 0.0f));
        ASSERT_TRUE(s.valid);
        EXPECT_LE(s.radiance.r, t.radiance.r + 1e-4f) << "at distance " << d;
        EXPECT_TRUE(std::isfinite(s.radiance.r));
    }
}

TEST(EmissiveLights, DegenerateRecordsContributeNothing) {
    EmissiveTriangle zero_area =
        make_tri({0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {5.0f, 5.0f, 5.0f});
    EXPECT_FALSE(emissive_shading(zero_area, glm::vec3(0.0f, 10.0f, 0.0f)).valid);
    EXPECT_EQ(emissive_target(zero_area, glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(0, -1, 0)), 0.0f);

    // A receiver sitting exactly on the centroid: no direction, no answer.
    const EmissiveTriangle t =
        make_tri({0.0f, 0.0f, 0.0f}, {3.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 3.0f}, {5.0f, 5.0f, 5.0f});
    const glm::vec3 centroid = t.v0 + (t.e1 + t.e2) / 3.0f;
    EXPECT_FALSE(emissive_shading(t, centroid).valid);
}

// ---------------------------------------------------------------------------
// The multi-source RIS weighting — the load-bearing claim of R6's DI half.
// ---------------------------------------------------------------------------

TEST(MultiSourceRis, SingleSourceScaleIsExactlyOne) {
    // Every pre-R6 configuration takes this path, and "exactly 1.0f" (not
    // "about 1") is what makes multiplying by it bit-identical — which is in
    // turn what lets the Ratio baselines be reproduced byte for byte.
    for (uint32_t m = 1; m <= 16; ++m) {
        EXPECT_EQ(ris_source_scale(m, m), 1.0f);
    }
    EXPECT_EQ(ris_source_scale(0, 4), 0.0f); // an empty source draws nothing
    EXPECT_FLOAT_EQ(ris_source_scale(2, 8), 4.0f);
}

namespace {

/// A deterministic LCG, so the estimator tests are reproducible run to run and
/// a failure is a real failure rather than an unlucky seed.
struct Lcg {
    uint32_t s = 0x9E3779B9u;
    float next() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
    }
};

struct AnalyticLight {
    glm::vec3 position;
    float radius;
    glm::vec3 color;
    float intensity;
};

} // namespace

TEST(MultiSourceRis, CombinedEstimatorConvergesToTheSumOfBothIntegrals) {
    // THE TEST THAT MATTERS. Two disjoint candidate sources feed one reservoir;
    // the estimate must converge to I_analytic + I_emissive, not to a blend of
    // them. Ground truth is available in closed form here because the target
    // function IS the integrand: summing every light's p̂ is the exact answer.
    const glm::vec3 P{0.0f, 0.0f, 0.0f};
    const glm::vec3 N{0.0f, 1.0f, 0.0f};

    const std::vector<AnalyticLight> lights{
        {{20.0f, 30.0f, 5.0f}, 400.0f, {1.0f, 0.9f, 0.7f}, 900.0f},
        {{-40.0f, 15.0f, 20.0f}, 300.0f, {0.4f, 0.6f, 1.0f}, 500.0f},
        {{5.0f, 60.0f, -30.0f}, 500.0f, {1.0f, 1.0f, 1.0f}, 1200.0f},
        {{200.0f, 10.0f, 0.0f}, 100.0f, {1.0f, 1.0f, 1.0f}, 800.0f}, // out of range: p̂ = 0
    };
    std::vector<EmissiveTriangle> records{
        make_tri({-30.0f, 80.0f, -30.0f}, {60.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 60.0f}, {2.0f, 1.4f, 0.5f}),
        make_tri({10.0f, 20.0f, 40.0f}, {8.0f, 0.0f, 0.0f}, {0.0f, 6.0f, 0.0f}, {9.0f, 9.0f, 9.0f}),
        make_tri({-5.0f, 200.0f, 5.0f}, {30.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 30.0f}, {0.3f, 0.3f, 0.6f}),
    };
    std::vector<float> weights(records.size());
    for (std::size_t i = 0; i < records.size(); ++i) {
        weights[i] = records[i].power;
    }
    std::vector<bracken::render::lighting::AliasEntry> table;
    const float total_power = build_alias_table(weights, table);
    for (std::size_t i = 0; i < records.size(); ++i) {
        records[i].alias_prob = table[i].prob;
        records[i].alias_index = table[i].alias;
    }

    double truth_a = 0.0;
    for (const AnalyticLight& l : lights) {
        truth_a += point_light_target(l.position, l.radius, l.color, l.intensity, P, N);
    }
    double truth_e = 0.0;
    for (const EmissiveTriangle& r : records) {
        truth_e += emissive_target(r, P, N);
    }
    ASSERT_GT(truth_a, 0.0);
    ASSERT_GT(truth_e, 0.0);

    // The shader's exact expressions, streamed through the shipped reservoir.
    const uint32_t m_a = 4;
    const uint32_t m_e = 2;
    const uint32_t m_cur = m_a + m_e;
    const float scale_a = ris_source_scale(m_a, m_cur);
    const float scale_e = ris_source_scale(m_e, m_cur);
    const auto list_count = static_cast<float>(lights.size());

    Lcg rng;
    constexpr int TRIALS = 400000;
    double acc = 0.0;
    for (int t = 0; t < TRIALS; ++t) {
        bracken::render::lighting::ReservoirBuilder res;
        for (uint32_t i = 0; i < m_a; ++i) {
            const auto slot = std::min(static_cast<std::size_t>(rng.next() * list_count),
                                       lights.size() - 1);
            const AnalyticLight& l = lights[slot];
            const float p = point_light_target(l.position, l.radius, l.color, l.intensity, P, N);
            res.update(static_cast<uint32_t>(slot), p * list_count * scale_a, p, rng.next());
        }
        for (uint32_t i = 0; i < m_e; ++i) {
            const float u1 = rng.next();
            const float u2 = rng.next();
            const uint32_t rec = sample_alias(table, u1, u2);
            const float p = emissive_target(records[rec], P, N);
            const float pdf = records[rec].power / total_power;
            res.update(rec, pdf > 0.0f ? p * scale_e / pdf : 0.0f, p, rng.next());
        }
        res.finalize();
        // The ReSTIR estimate of the WHOLE candidate sum: p̂(y) * W.
        acc += static_cast<double>(res.target_pdf) * static_cast<double>(res.r.weight);
    }
    const double estimate = acc / TRIALS;
    const double truth = truth_a + truth_e;
    EXPECT_NEAR(estimate / truth, 1.0, 0.02)
        << "estimate " << estimate << " vs truth " << truth << " (analytic " << truth_a
        << " + emissive " << truth_e << ")";

    // And the counter-example that shows the scales are load-bearing rather
    // than decorative: with a shared 1/M normalization (both scales 1) the same
    // estimator converges to the convex COMBINATION instead of the sum, i.e. it
    // dims the frame. Asserting on the specific wrong value — rather than just
    // "not equal" — is what would catch someone reintroducing it.
    Lcg rng2;
    double acc2 = 0.0;
    for (int t = 0; t < TRIALS; ++t) {
        bracken::render::lighting::ReservoirBuilder res;
        for (uint32_t i = 0; i < m_a; ++i) {
            const auto slot = std::min(static_cast<std::size_t>(rng2.next() * list_count),
                                       lights.size() - 1);
            const AnalyticLight& l = lights[slot];
            const float p = point_light_target(l.position, l.radius, l.color, l.intensity, P, N);
            res.update(static_cast<uint32_t>(slot), p * list_count, p, rng2.next());
        }
        for (uint32_t i = 0; i < m_e; ++i) {
            const float u1 = rng2.next();
            const float u2 = rng2.next();
            const uint32_t rec = sample_alias(table, u1, u2);
            const float p = emissive_target(records[rec], P, N);
            const float pdf = records[rec].power / total_power;
            res.update(rec, pdf > 0.0f ? p / pdf : 0.0f, p, rng2.next());
        }
        res.finalize();
        acc2 += static_cast<double>(res.target_pdf) * static_cast<double>(res.r.weight);
    }
    const double naive = acc2 / TRIALS;
    const double convex = (m_a * truth_a + m_e * truth_e) / static_cast<double>(m_cur);
    EXPECT_NEAR(naive / convex, 1.0, 0.02);
}
