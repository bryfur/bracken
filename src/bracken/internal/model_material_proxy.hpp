#pragma once

// Internal CPU material proxy used by software ray tracing. This stays out of
// the supported surface: hosts author ordinary Mesh/PBRMaterial data, and the
// model upload path decides whether a retained-geometry model needs the proxy.

#include "bracken/model_loader.hpp"

#include <cmath>

namespace bracken::internal {

[[nodiscard]] inline float srgb_channel_to_linear(float c) {
    return c <= 0.04045f ? c / 12.92f
                         : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

[[nodiscard]] inline glm::vec4 sample_srgb_repeat(const std::vector<uint8_t>& pixels,
                                                   int width, int height, glm::vec2 uv) {
    if (width <= 0 || height <= 0 ||
        pixels.size() < static_cast<std::size_t>(width * height * 4)) {
        return glm::vec4(1.0f);
    }

    // Match the model sampler's repeat + linear filtering. Texel centres are
    // at n + 0.5, hence the -0.5 before the bilinear footprint.
    uv -= glm::floor(uv);
    const glm::vec2 p = uv * glm::vec2(width, height) - 0.5f;
    const glm::ivec2 lo = glm::ivec2(glm::floor(p));
    const glm::vec2 f = glm::fract(p);
    auto wrap = [](int v, int extent) {
        const int m = v % extent;
        return m < 0 ? m + extent : m;
    };
    auto load = [&](int x, int y) {
        x = wrap(x, width);
        y = wrap(y, height);
        const std::size_t i = static_cast<std::size_t>((y * width + x) * 4);
        constexpr float inv255 = 1.0f / 255.0f;
        return glm::vec4(srgb_channel_to_linear(pixels[i + 0] * inv255),
                         srgb_channel_to_linear(pixels[i + 1] * inv255),
                         srgb_channel_to_linear(pixels[i + 2] * inv255),
                         pixels[i + 3] * inv255);
    };

    const glm::vec4 a = glm::mix(load(lo.x, lo.y), load(lo.x + 1, lo.y), f.x);
    const glm::vec4 b = glm::mix(load(lo.x, lo.y + 1), load(lo.x + 1, lo.y + 1), f.x);
    return glm::mix(a, b, f.y);
}

inline void bake_rt_triangle_base_colors(Mesh& mesh) {
    mesh.rt_triangle_base_colors.clear();
    if (!mesh.has_texture || mesh.texture_pixels.empty() || mesh.indices.size() < 3 ||
        mesh.vertices.empty()) {
        return;
    }

    const std::size_t triangle_count = mesh.indices.size() / 3;
    mesh.rt_triangle_base_colors.reserve(triangle_count);
    for (std::size_t tri = 0; tri < triangle_count; ++tri) {
        const uint32_t i0 = mesh.indices[tri * 3 + 0];
        const uint32_t i1 = mesh.indices[tri * 3 + 1];
        const uint32_t i2 = mesh.indices[tri * 3 + 2];
        if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() ||
            i2 >= mesh.vertices.size()) {
            mesh.rt_triangle_base_colors.push_back(mesh.material.base_color_factor);
            continue;
        }
        const Vertex3D& v0 = mesh.vertices[i0];
        const Vertex3D& v1 = mesh.vertices[i1];
        const Vertex3D& v2 = mesh.vertices[i2];
        const glm::vec2 centroid_uv = (v0.texcoord + v1.texcoord + v2.texcoord) / 3.0f;
        const glm::vec4 vertex_color = (v0.color + v1.color + v2.color) / 3.0f;
        mesh.rt_triangle_base_colors.push_back(
            sample_srgb_repeat(mesh.texture_pixels, mesh.texture_width,
                               mesh.texture_height, centroid_uv) *
            vertex_color * mesh.material.base_color_factor);
    }
}

/**
 * Bake the per-triangle EMISSIVE proxy (R6, docs/rt_lighting_plan.md §5.7).
 *
 * `emissive_factor` (already carrying KHR_materials_emissive_strength) times
 * the emissive texture's centroid-UV sample where one exists, else times 1.
 *
 * THREE DIFFERENCES FROM THE BASE-COLOUR BAKE ABOVE, all of them consequences
 * of what emission is rather than style choices:
 *
 *  1. It runs with NO TEXTURE. A mesh whose only emission is a factor is the
 *     normal case for procedural content (the playground's orbs), and it is
 *     exactly the case that must produce records — refusing to bake without a
 *     texture would make "emissive mesh lights" a glTF-only feature.
 *  2. Vertex colour is NOT applied. glTF's COLOR_0 modulates base colour; the
 *     emissive term in every forward shader in this tree is
 *     `emissiveFactor * emissiveTexture`, and this proxy exists to agree with
 *     that shader, not to be a second interpretation of the material.
 *  3. It bakes NOTHING when the factor is black, leaving the vector empty. Both
 *     downstream consumers (the attribute buffer and the emissive-light gather)
 *     read "empty" as "this mesh does not emit", so a village of 52 models pays
 *     one branch per mesh rather than 600 k zeroed vec3s.
 */
inline void bake_rt_triangle_emissive(Mesh& mesh) {
    mesh.rt_triangle_emissive.clear();
    const glm::vec3 factor = mesh.material.emissive_factor;
    if (factor.r <= 0.0f && factor.g <= 0.0f && factor.b <= 0.0f) {
        return;
    }
    if (mesh.indices.size() < 3) {
        return;
    }

    const bool has_map = !mesh.material.emissive_pixels.empty() &&
                         mesh.material.emissive_width > 0 && mesh.material.emissive_height > 0;
    const std::size_t triangle_count = mesh.indices.size() / 3;
    mesh.rt_triangle_emissive.assign(triangle_count, factor);
    if (!has_map || mesh.vertices.empty()) {
        return;
    }

    for (std::size_t tri = 0; tri < triangle_count; ++tri) {
        const uint32_t i0 = mesh.indices[tri * 3 + 0];
        const uint32_t i1 = mesh.indices[tri * 3 + 1];
        const uint32_t i2 = mesh.indices[tri * 3 + 2];
        if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() ||
            i2 >= mesh.vertices.size()) {
            continue; // keeps the untextured factor, as the base-colour bake does
        }
        const glm::vec2 centroid_uv = (mesh.vertices[i0].texcoord + mesh.vertices[i1].texcoord +
                                       mesh.vertices[i2].texcoord) /
                                      3.0f;
        const glm::vec4 s = sample_srgb_repeat(mesh.material.emissive_pixels,
                                               mesh.material.emissive_width,
                                               mesh.material.emissive_height, centroid_uv);
        mesh.rt_triangle_emissive[tri] = factor * glm::vec3(s);
    }
}

} // namespace bracken::internal
