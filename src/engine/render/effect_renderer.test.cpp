#include "engine/render/effect_renderer.hpp"
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <gtest/gtest.h>

using mmo::engine::render::build_particle_transform;
using mmo::engine::systems::Particle;

namespace {

// The naive construction build_particle_transform replaced (chained
// glm::translate/rotate/scale calls, rotation order Y then X then Z, then
// recentering). Kept as the reference so the direct Euler composition stays
// pinned to identical output.
glm::mat4 reference_transform(const Particle& particle, const glm::vec3& center) {
    glm::mat4 m(1.0f);
    m = glm::translate(m, particle.position);
    m = glm::rotate(m, particle.rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
    m = glm::rotate(m, particle.rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
    m = glm::rotate(m, particle.rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
    m = glm::scale(m, glm::vec3(particle.scale));
    return m * glm::translate(glm::mat4(1.0f), -center);
}

void expect_mat4_near(const glm::mat4& actual, const glm::mat4& expected, float eps) {
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            EXPECT_NEAR(actual[col][row], expected[col][row], eps) << "mismatch at [" << col << "][" << row << "]";
        }
    }
}

} // namespace

TEST(BuildParticleTransform, DefaultParticleWithNoCenterIsIdentity) {
    Particle p;
    expect_mat4_near(build_particle_transform(p, glm::vec3(0.0f)), glm::mat4(1.0f), 1e-6f);
}

TEST(BuildParticleTransform, TranslationOnly) {
    Particle p;
    p.position = {10.0f, -3.5f, 42.0f};
    expect_mat4_near(build_particle_transform(p, glm::vec3(0.0f)),
                     glm::translate(glm::mat4(1.0f), p.position), 1e-6f);
}

TEST(BuildParticleTransform, MatchesNaiveGlmChain) {
    // Sweep positions, all three Euler angles (incl. negatives and > pi),
    // scales, and center offsets against the reference construction.
    const glm::vec3 positions[] = {{0, 0, 0}, {5, 10, -3}, {-100.5f, 0.25f, 77.0f}};
    const glm::vec3 rotations[] = {
        {0, 0, 0},          {0.5f, 0, 0},       {0, 1.2f, 0},          {0, 0, -0.7f},
        {0.3f, 1.1f, -2.2f}, {-1.5f, 3.9f, 0.4f}, {6.5f, -4.25f, 2.75f},
    };
    const float scales[] = {1.0f, 0.35f, 7.5f};
    const glm::vec3 centers[] = {{0, 0, 0}, {1.5f, -0.75f, 2.25f}};

    for (const auto& pos : positions) {
        for (const auto& rot : rotations) {
            for (float s : scales) {
                for (const auto& c : centers) {
                    Particle p;
                    p.position = pos;
                    p.rotation = rot;
                    p.scale = s;
                    expect_mat4_near(build_particle_transform(p, c), reference_transform(p, c), 1e-3f);
                }
            }
        }
    }
}

TEST(BuildParticleTransform, CenterOffsetMapsModelCenterToParticlePosition) {
    // The recentering translate runs innermost, so the model-space center
    // must land exactly on the particle position for any rotation/scale.
    Particle p;
    p.position = {12.0f, 34.0f, -56.0f};
    p.rotation = {0.4f, -1.3f, 2.1f};
    p.scale = 2.5f;
    const glm::vec3 center(3.0f, -1.0f, 0.5f);

    const glm::vec4 mapped = build_particle_transform(p, center) * glm::vec4(center, 1.0f);
    EXPECT_NEAR(mapped.x, p.position.x, 1e-3f);
    EXPECT_NEAR(mapped.y, p.position.y, 1e-3f);
    EXPECT_NEAR(mapped.z, p.position.z, 1e-3f);
}
