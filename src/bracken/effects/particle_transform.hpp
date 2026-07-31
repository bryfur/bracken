#pragma once

// The particle world-transform composition, as a template over "anything with
// .position / .rotation (euler radians) / .scale".
//
// It lives in its own header so the effect renderer does not have to include
// `effect_system.hpp` just to declare an overload taking a live
// `systems::Particle`. The production render path never sees a live particle —
// particles reach the renderer only as the sealed packet's
// `scene::EffectParticleSnapshot`. Only the test instantiates the live-particle
// form, and it does so directly against this template, which is why the
// `render -> systems` edge (the only one in the tree) could go away without
// weakening what is tested.

#include <cmath>
#include <glm/glm.hpp>

namespace bracken::render {

/// Compose a particle's world transform: translate(position) * rotY * rotX *
/// rotZ * scale * translate(-center_offset).
///
/// Equivalent to translate(position) * rotate(Y) * rotate(X) * rotate(Z) *
/// scale(s) * translate(-center_offset), composed directly: one sin/cos pair
/// per axis instead of three chained 4x4 rotate multiplies
/// (effect_renderer.test.cpp pins equivalence with the glm chain).
template <typename ParticleT>
glm::mat4 build_particle_transform_of(const ParticleT& particle, const glm::vec3& center_offset) {
    const float sx = std::sin(particle.rotation.x), cx = std::cos(particle.rotation.x);
    const float sy = std::sin(particle.rotation.y), cy = std::cos(particle.rotation.y);
    const float sz = std::sin(particle.rotation.z), cz = std::cos(particle.rotation.z);

    // R = Ry * Rx * Rz, column-major (glm), column vectors.
    const glm::mat3 rotation(
        // column 0
        cy * cz + sy * sx * sz, cx * sz, -sy * cz + cy * sx * sz,
        // column 1
        -cy * sz + sy * sx * cz, cx * cz, sy * sz + cy * sx * cz,
        // column 2
        sy * cx, -sx, cy * cx);

    const float s = particle.scale;
    glm::mat4 model_mat(1.0f);
    model_mat[0] = glm::vec4(rotation[0] * s, 0.0f);
    model_mat[1] = glm::vec4(rotation[1] * s, 0.0f);
    model_mat[2] = glm::vec4(rotation[2] * s, 0.0f);
    // T * R * S * T(-center): translation = position - R * (center * s).
    model_mat[3] = glm::vec4(particle.position - rotation * (center_offset * s), 1.0f);
    return model_mat;
}

} // namespace bracken::render
