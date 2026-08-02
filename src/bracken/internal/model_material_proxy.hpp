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

} // namespace bracken::internal
