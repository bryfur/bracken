#include "bracken/procedural/tree_generator.hpp"
#include "bracken/model_loader.hpp"
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <gtest/gtest.h>
#include <SDL3/SDL.h>
#include <string>

using namespace bracken::procedural;
using namespace bracken;

TEST(TreeGenerator, GenerateProducesNonEmptyModel) {
    TreeParams p;
    p.seed = 1;
    p.levels = 2;
    auto model = TreeGenerator::generate(p);
    ASSERT_NE(model, nullptr);
    EXPECT_FALSE(model->meshes.empty());
    EXPECT_TRUE(model->loaded);
}

TEST(TreeGenerator, BranchMeshHasGeometry) {
    TreeParams p;
    p.seed = 1;
    p.levels = 1;
    auto model = TreeGenerator::generate(p);
    ASSERT_GE(model->meshes.size(), 1u);
    const auto& branch_mesh = model->meshes[0];
    EXPECT_GT(branch_mesh.vertices.size(), 0u);
    EXPECT_GT(branch_mesh.indices.size(), 0u);
    // Indices should be multiples of 3 (triangles)
    EXPECT_EQ(branch_mesh.indices.size() % 3, 0u);
}

TEST(TreeGenerator, LeavesMeshGenerated) {
    TreeParams p;
    p.seed = 1;
    p.levels = 1;
    p.leaf_count = 5;
    auto model = TreeGenerator::generate(p);
    // Should have 2 meshes: branches + leaves
    ASSERT_GE(model->meshes.size(), 2u);
    const auto& leaf_mesh = model->meshes[1];
    EXPECT_GT(leaf_mesh.vertices.size(), 0u);
    EXPECT_GT(leaf_mesh.indices.size(), 0u);
}

TEST(TreeGenerator, NoLeavesMeshWhenZeroCount) {
    TreeParams p;
    p.seed = 1;
    p.levels = 1;
    p.leaf_count = 0;
    auto model = TreeGenerator::generate(p);
    // Only branch mesh, no leaf mesh
    EXPECT_EQ(model->meshes.size(), 1u);
}

TEST(TreeGenerator, SeededRNGIsDeterministic) {
    TreeParams p;
    p.seed = 42;
    p.levels = 2;
    auto model1 = TreeGenerator::generate(p);
    auto model2 = TreeGenerator::generate(p);
    ASSERT_EQ(model1->meshes[0].vertices.size(), model2->meshes[0].vertices.size());
    // Verify same geometry
    for (size_t i = 0; i < model1->meshes[0].vertices.size(); i++) {
        EXPECT_FLOAT_EQ(model1->meshes[0].vertices[i].position.x, model2->meshes[0].vertices[i].position.x);
    }
}

TEST(TreeGenerator, DifferentSeedsProduceDifferentTrees) {
    TreeParams p;
    p.levels = 2;
    p.seed = 1;
    auto model1 = TreeGenerator::generate(p);
    p.seed = 99;
    auto model2 = TreeGenerator::generate(p);
    // Different seeds should produce different vertex positions
    bool any_different = false;
    size_t n = std::min(model1->meshes[0].vertices.size(), model2->meshes[0].vertices.size());
    for (size_t i = 0; i < n; i++) {
        if (model1->meshes[0].vertices[i].position != model2->meshes[0].vertices[i].position) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different);
}

TEST(TreeGenerator, AABBIsComputed) {
    auto model = TreeGenerator::generate_oak(1);
    // Tree should extend upward from origin
    EXPECT_GT(model->max_y, 0.0f);
    EXPECT_NEAR(model->min_y, 0.0f, 1.0f);
    // Should have some width
    EXPECT_GT(model->max_x - model->min_x, 0.1f);
}

TEST(TreeGenerator, BoundingSphereIsComputed) {
    auto model = TreeGenerator::generate_oak(1);
    EXPECT_GT(model->bounding_half_diag, 0.0f);
}

TEST(TreeGenerator, NormalsAreUnitLength) {
    auto model = TreeGenerator::generate_oak(1);
    for (const auto& mesh : model->meshes) {
        for (const auto& v : mesh.vertices) {
            float len = glm::length(v.normal);
            EXPECT_NEAR(len, 1.0f, 0.01f);
        }
    }
}

TEST(TreeGenerator, OakPresetProducesTree) {
    auto model = TreeGenerator::generate_oak(42);
    ASSERT_NE(model, nullptr);
    EXPECT_GE(model->meshes.size(), 2u); // branches + leaves
}

TEST(TreeGenerator, PinePresetProducesTree) {
    auto model = TreeGenerator::generate_pine(42);
    ASSERT_NE(model, nullptr);
    EXPECT_GE(model->meshes.size(), 2u);
}

TEST(TreeGenerator, DeadPresetHasNoLeaves) {
    auto model = TreeGenerator::generate_dead(42);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->meshes.size(), 1u); // branches only
}

TEST(TreeGenerator, PresetsProduceDifferentShapes) {
    auto oak = TreeGenerator::generate_oak(42);
    auto pine = TreeGenerator::generate_pine(42);
    auto dead = TreeGenerator::generate_dead(42);
    // Different presets should produce different vertex counts
    EXPECT_NE(oak->meshes[0].vertices.size(), pine->meshes[0].vertices.size());
    EXPECT_NE(oak->meshes[0].vertices.size(), dead->meshes[0].vertices.size());
}

TEST(TreeGenerator, MoreLevelsProducesMoreGeometry) {
    TreeParams p;
    p.seed = 1;
    p.levels = 1;
    p.leaf_count = 0;
    auto simple = TreeGenerator::generate(p);
    p.levels = 3;
    auto complex = TreeGenerator::generate(p);
    EXPECT_GT(complex->meshes[0].vertices.size(), simple->meshes[0].vertices.size());
}

TEST(TreeGenerator, DoubleBillboardDoubleLeafQuads) {
    TreeParams p;
    p.seed = 1;
    p.levels = 1;
    p.leaf_count = 1;

    p.double_billboard = false;
    auto single = TreeGenerator::generate(p);

    p.double_billboard = true;
    auto doubled = TreeGenerator::generate(p);

    // Double billboard should have twice the leaf vertices
    ASSERT_GE(single->meshes.size(), 2u);
    ASSERT_GE(doubled->meshes.size(), 2u);
    EXPECT_EQ(doubled->meshes[1].vertices.size(), single->meshes[1].vertices.size() * 2);
}

TEST(TreeGenerator, TrunkOnlyWithZeroLevels) {
    TreeParams p;
    p.seed = 1;
    p.levels = 0;
    p.leaf_count = 1;
    auto model = TreeGenerator::generate(p);
    ASSERT_NE(model, nullptr);
    // Should still have geometry (trunk + leaves at level 0)
    EXPECT_GT(model->meshes[0].vertices.size(), 0u);
}

// ============================================================================
// Parameter validation (M20)
// ============================================================================

TEST(TreeGenerator, LevelsAboveMaxAreClampedWithoutOutOfBoundsReads) {
    TreeParams p;
    p.seed = 3;
    p.levels = TreeParams::MAX_LEVELS + 2;      // out of range
    p.children[TreeParams::MAX_LEVELS - 1] = 2; // would read past the arrays without clamping
    auto clamped = TreeGenerator::generate(p);
    ASSERT_NE(clamped, nullptr);

    // Clamped generation must match a run at the maximum valid level exactly.
    p.levels = TreeParams::MAX_LEVELS - 1;
    auto max_valid = TreeGenerator::generate(p);
    ASSERT_NE(max_valid, nullptr);
    ASSERT_EQ(clamped->meshes.size(), max_valid->meshes.size());
    for (size_t m = 0; m < clamped->meshes.size(); m++) {
        EXPECT_EQ(clamped->meshes[m].vertices.size(), max_valid->meshes[m].vertices.size());
        EXPECT_EQ(clamped->meshes[m].indices.size(), max_valid->meshes[m].indices.size());
    }
}

TEST(TreeGenerator, NegativeLevelsClampToTrunkOnly) {
    TreeParams p;
    p.seed = 3;
    p.levels = -5;
    auto negative = TreeGenerator::generate(p);
    ASSERT_NE(negative, nullptr);
    p.levels = 0;
    auto zero = TreeGenerator::generate(p);
    ASSERT_NE(zero, nullptr);
    EXPECT_EQ(negative->meshes[0].vertices.size(), zero->meshes[0].vertices.size());
}

TEST(TreeGenerator, ZeroSectionsAndSegmentsAreClampedToFiniteGeometry) {
    TreeParams p;
    p.seed = 3;
    p.levels = 1;
    for (int i = 0; i < TreeParams::MAX_LEVELS; i++) {
        p.sections[i] = 0;
        p.segments[i] = 0;
        p.children[i] = -3;
    }
    auto model = TreeGenerator::generate(p);
    ASSERT_NE(model, nullptr);
    for (const auto& mesh : model->meshes) {
        for (const auto& v : mesh.vertices) {
            ASSERT_TRUE(std::isfinite(v.position.x));
            ASSERT_TRUE(std::isfinite(v.position.y));
            ASSERT_TRUE(std::isfinite(v.position.z));
        }
    }
}

// ============================================================================
// Leaf material flags + texture fallback (M18/M19)
// ============================================================================

namespace {

// Writes a small solid-color BMP that IMG_Load can read back. Returns ""
// on failure.
std::string write_test_texture() {
    std::string path = ::testing::TempDir();
    if (!path.empty() && path.back() != '/') {
        path += '/';
    }
    path += "bracken_tree_generator_test_leaf.bmp";
    SDL_Surface* surface = SDL_CreateSurface(4, 4, SDL_PIXELFORMAT_RGBA32);
    if (!surface) {
        return "";
    }
    SDL_FillSurfaceRect(surface, nullptr, SDL_MapSurfaceRGBA(surface, 60, 160, 40, 255));
    const bool ok = SDL_SaveBMP(surface, path.c_str());
    SDL_DestroySurface(surface);
    return ok ? path : "";
}

} // namespace

TEST(TreeGenerator, UntexturedLeafMeshIsDoubleSidedAndOpaque) {
    TreeParams p;
    p.seed = 1;
    p.levels = 1;
    p.leaf_count = 2;
    auto model = TreeGenerator::generate(p);
    ASSERT_GE(model->meshes.size(), 2u);
    const auto& leaf = model->meshes[1];
    EXPECT_FALSE(leaf.has_texture);
    // Leaf cards must render from both sides even without a texture...
    EXPECT_TRUE(leaf.material.double_sided);
    // ...but alpha-testing only makes sense with a texture.
    EXPECT_EQ(leaf.material.alpha_mode, AlphaMode::Opaque);
}

TEST(TreeGenerator, TexturedLeafMeshUsesAlphaMaskAndDoubleSided) {
    const std::string texture = write_test_texture();
    if (texture.empty()) {
        GTEST_SKIP() << "could not write test texture";
    }
    TreeParams p;
    p.seed = 1;
    p.levels = 1;
    p.leaf_count = 2;
    p.leaf_texture_path = texture;
    auto model = TreeGenerator::generate(p);
    ASSERT_GE(model->meshes.size(), 2u);
    const auto& leaf = model->meshes[1];
    ASSERT_TRUE(leaf.has_texture);
    EXPECT_EQ(leaf.material.alpha_mode, AlphaMode::Mask);
    EXPECT_FLOAT_EQ(leaf.material.alpha_cutoff, 0.5f);
    EXPECT_TRUE(leaf.material.double_sided);
    // Vertex colors are whitened so the texture isn't tinted.
    for (const auto& v : leaf.vertices) {
        ASSERT_EQ(v.color, glm::vec4(1.0f));
    }
}

TEST(TreeGenerator, MissingTexturePreservesVertexColorFallback) {
    TreeParams p;
    p.seed = 1;
    p.levels = 1;
    p.leaf_count = 2;
    p.bark_texture_path = "/nonexistent/bracken_test/bark.png";
    p.leaf_texture_path = "/nonexistent/bracken_test/leaf.png";
    auto model = TreeGenerator::generate(p);
    ASSERT_GE(model->meshes.size(), 2u);
    const auto& branch = model->meshes[0];
    const auto& leaf = model->meshes[1];
    EXPECT_FALSE(branch.has_texture);
    EXPECT_FALSE(leaf.has_texture);
    // Failed loads must not whiten the vertex colors (all-white tree bug).
    for (const auto& v : branch.vertices) {
        ASSERT_EQ(v.color, p.bark_color);
    }
    for (const auto& v : leaf.vertices) {
        ASSERT_EQ(v.color, p.leaf_color);
    }
    // No texture -> no alpha test.
    EXPECT_EQ(leaf.material.alpha_mode, AlphaMode::Opaque);
}

// ============================================================================
// Canopy shadow proxy geometry (L72)
// ============================================================================

TEST(TreeGenerator, CanopyProxyEmitsBothWindingsPerQuad) {
    TreeParams p;
    p.seed = 1;
    p.levels = 1;
    p.leaf_count = 2;
    auto model = TreeGenerator::generate(p);
    ASSERT_GE(model->meshes.size(), 3u);
    const auto& proxy = model->meshes[2];
    EXPECT_TRUE(proxy.shadow_only);
    EXPECT_TRUE(proxy.cast_shadows);
    // Per clump: 12 vertices (3 quads) and 36 indices (12 triangles — each
    // quad is emitted with both winding orders for double-sided shadows).
    ASSERT_GT(proxy.vertices.size(), 0u);
    EXPECT_EQ(proxy.vertices.size() % 12, 0u);
    EXPECT_EQ(proxy.indices.size(), proxy.vertices.size() * 3);
    for (uint32_t index : proxy.indices) {
        ASSERT_LT(index, proxy.vertices.size());
    }
}
