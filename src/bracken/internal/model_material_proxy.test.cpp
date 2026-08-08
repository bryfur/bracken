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

// ---------------------------------------------------------------------------
// Alpha-cutout opacity (docs/rt_lighting_plan.md §5.9)
// ---------------------------------------------------------------------------

namespace {

/// A mesh covering the whole of a `w x h` alpha texture with one triangle pair's
/// worth of UV span, so the bake's stratified taps land across the texture.
Mesh cutout_mesh(int w, int h, std::vector<uint8_t> pixels, float cutoff) {
    Mesh mesh;
    mesh.has_texture = true;
    mesh.texture_width = w;
    mesh.texture_height = h;
    mesh.texture_pixels = std::move(pixels);
    mesh.material.alpha_mode = AlphaMode::Mask;
    mesh.material.alpha_cutoff = cutoff;
    mesh.vertices.resize(3);
    mesh.vertices[0].texcoord = {0.0f, 0.0f};
    mesh.vertices[1].texcoord = {1.0f, 0.0f};
    mesh.vertices[2].texcoord = {0.0f, 1.0f};
    mesh.indices = {0, 1, 2};
    return mesh;
}

/// RGBA8 pixels whose ALPHA alternates fully-opaque / fully-transparent in a
/// checker, so the true coverage of any fair sampling is ~0.5.
std::vector<uint8_t> checker_alpha(int w, int h) {
    std::vector<uint8_t> px(static_cast<std::size_t>(w) * h * 4, 255);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            px[(static_cast<std::size_t>(y) * w + x) * 4 + 3] =
                ((x + y) % 2 == 0) ? 255 : 0;
        }
    }
    return px;
}

} // namespace

TEST(RtOpacityProxy, OpaqueMaterialBakesNothing) {
    // Empty means fully opaque — the same convention the emissive bake uses for
    // "does not emit", and what keeps an opaque model at zero cost.
    Mesh mesh = cutout_mesh(2, 2, checker_alpha(2, 2), 0.5f);
    mesh.material.alpha_mode = AlphaMode::Opaque;
    bake_rt_triangle_opacity(mesh);
    EXPECT_TRUE(mesh.rt_triangle_opacity.empty());
}

TEST(RtOpacityProxy, FullyOpaqueTextureReportsFullCoverage) {
    std::vector<uint8_t> px(4 * 4 * 4, 255);
    Mesh mesh = cutout_mesh(4, 4, std::move(px), 0.5f);
    bake_rt_triangle_opacity(mesh);
    ASSERT_EQ(mesh.rt_triangle_opacity.size(), 1u);
    EXPECT_FLOAT_EQ(mesh.rt_triangle_opacity[0], 1.0f);
}

TEST(RtOpacityProxy, FullyTransparentTextureReportsZeroCoverage) {
    std::vector<uint8_t> px(4 * 4 * 4, 255);
    for (std::size_t i = 3; i < px.size(); i += 4) {
        px[i] = 0;
    }
    Mesh mesh = cutout_mesh(4, 4, std::move(px), 0.5f);
    bake_rt_triangle_opacity(mesh);
    ASSERT_EQ(mesh.rt_triangle_opacity.size(), 1u);
    EXPECT_FLOAT_EQ(mesh.rt_triangle_opacity[0], 0.0f);
}

TEST(RtOpacityProxy, PartialCoverageIsMeasuredRatherThanRoundedToOne) {
    // THE POINT OF THE WHOLE BAKE. A single centroid tap returns 0 or 1 for an
    // alpha TEST and nothing between, so a half-gap leaf card would report 1.0
    // and the traced shadow would stay exactly as solid as it was. Stratified
    // taps put a real fraction in.
    Mesh mesh = cutout_mesh(16, 16, checker_alpha(16, 16), 0.5f);
    bake_rt_triangle_opacity(mesh);
    ASSERT_EQ(mesh.rt_triangle_opacity.size(), 1u);
    const float coverage = mesh.rt_triangle_opacity[0];
    EXPECT_GT(coverage, 0.0f) << "a half-transparent card must not be fully clear";
    EXPECT_LT(coverage, 1.0f) << "a half-transparent card must not be fully solid";
}

TEST(RtOpacityProxy, CoverageIsDeterministic) {
    // The smoke gate and the traced pins both require a model to bake
    // identically every run, which is why the sample points are a fixed
    // low-discrepancy sequence rather than a seeded draw.
    Mesh a = cutout_mesh(16, 16, checker_alpha(16, 16), 0.5f);
    Mesh b = cutout_mesh(16, 16, checker_alpha(16, 16), 0.5f);
    bake_rt_triangle_opacity(a);
    bake_rt_triangle_opacity(b);
    ASSERT_EQ(a.rt_triangle_opacity.size(), b.rt_triangle_opacity.size());
    EXPECT_FLOAT_EQ(a.rt_triangle_opacity[0], b.rt_triangle_opacity[0]);
}

TEST(RtOpacityProxy, CutoffIsHonoured) {
    // Alpha 0.5 everywhere: opaque under a 0.25 cutoff, clear under 0.75.
    std::vector<uint8_t> px(4 * 4 * 4, 255);
    for (std::size_t i = 3; i < px.size(); i += 4) {
        px[i] = 128;
    }
    Mesh low = cutout_mesh(4, 4, px, 0.25f);
    Mesh high = cutout_mesh(4, 4, px, 0.75f);
    bake_rt_triangle_opacity(low);
    bake_rt_triangle_opacity(high);
    EXPECT_FLOAT_EQ(low.rt_triangle_opacity[0], 1.0f);
    EXPECT_FLOAT_EQ(high.rt_triangle_opacity[0], 0.0f);
}

TEST(RtOpacityProxy, MaskWithoutATextureIsOpaque) {
    // Nothing to cut out: the mesh is solid wherever it draws, and inventing
    // holes would be worse than the status quo.
    Mesh mesh = cutout_mesh(4, 4, {}, 0.5f);
    mesh.has_texture = false;
    bake_rt_triangle_opacity(mesh);
    EXPECT_TRUE(mesh.rt_triangle_opacity.empty());
}

TEST(RtOpacityProxy, BarycentricSamplesStayInsideTheTriangle) {
    for (int i = 0; i < RT_OPACITY_SAMPLES; ++i) {
        const glm::vec3 b = rt_opacity_barycentric(i);
        EXPECT_GE(b.x, 0.0f);
        EXPECT_GE(b.y, 0.0f);
        EXPECT_GE(b.z, 0.0f);
        EXPECT_NEAR(b.x + b.y + b.z, 1.0f, 1e-5f) << "sample " << i;
    }
}

} // namespace
} // namespace bracken::internal
