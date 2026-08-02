#include "model_material_proxy.hpp"

#include <gtest/gtest.h>

namespace bracken::internal {
namespace {

TEST(RtMaterialProxy, SrgbDecodeMatchesTransferFunctionEndpoints) {
    EXPECT_FLOAT_EQ(srgb_channel_to_linear(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(srgb_channel_to_linear(1.0f), 1.0f);
    EXPECT_NEAR(srgb_channel_to_linear(0.5f), 0.214041f, 1e-5f);
}

TEST(RtMaterialProxy, BilinearSampleRepeatsAcrossUvBoundary) {
    // Two horizontal texels: red, then green. At u=0 the repeat footprint is
    // exactly half the last and first texel.
    const std::vector<uint8_t> pixels{255, 0, 0, 255, 0, 255, 0, 255};
    const glm::vec4 sample = sample_srgb_repeat(pixels, 2, 1, {0.0f, 0.5f});
    EXPECT_NEAR(sample.r, 0.5f, 1e-6f);
    EXPECT_NEAR(sample.g, 0.5f, 1e-6f);
    EXPECT_NEAR(sample.b, 0.0f, 1e-6f);
    EXPECT_NEAR(sample.a, 1.0f, 1e-6f);
}

TEST(RtMaterialProxy, BakeUsesCentroidTextureVertexAndMaterialFactors) {
    Mesh mesh;
    mesh.has_texture = true;
    mesh.texture_width = 1;
    mesh.texture_height = 1;
    mesh.texture_pixels = {128, 64, 255, 128};
    mesh.material.base_color_factor = {0.5f, 0.25f, 0.75f, 0.5f};
    mesh.vertices.resize(3);
    mesh.vertices[0].texcoord = {0.0f, 0.0f};
    mesh.vertices[1].texcoord = {1.0f, 0.0f};
    mesh.vertices[2].texcoord = {0.0f, 1.0f};
    for (Vertex3D& v : mesh.vertices) {
        v.color = {0.8f, 0.6f, 0.4f, 0.5f};
    }
    mesh.indices = {0, 1, 2};

    bake_rt_triangle_base_colors(mesh);

    ASSERT_EQ(mesh.rt_triangle_base_colors.size(), 1u);
    const glm::vec4 c = mesh.rt_triangle_base_colors[0];
    EXPECT_NEAR(c.r, srgb_channel_to_linear(128.0f / 255.0f) * 0.8f * 0.5f, 1e-6f);
    EXPECT_NEAR(c.g, srgb_channel_to_linear(64.0f / 255.0f) * 0.6f * 0.25f, 1e-6f);
    EXPECT_NEAR(c.b, 1.0f * 0.4f * 0.75f, 1e-6f);
    EXPECT_NEAR(c.a, (128.0f / 255.0f) * 0.5f * 0.5f, 1e-6f);
}

TEST(RtMaterialProxy, MissingTextureKeepsProxyEmpty) {
    Mesh mesh;
    mesh.vertices.resize(3);
    mesh.indices = {0, 1, 2};
    bake_rt_triangle_base_colors(mesh);
    EXPECT_TRUE(mesh.rt_triangle_base_colors.empty());
}

} // namespace
} // namespace bracken::internal
