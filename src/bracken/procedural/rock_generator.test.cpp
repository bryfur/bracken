#include "bracken/procedural/rock_generator.hpp"
#include "bracken/model_loader.hpp"
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <gtest/gtest.h>

using namespace bracken::procedural;
using namespace bracken;

namespace {

// Low resolution keeps the SDF grid evaluation fast; the marching-cubes path
// is identical at every resolution.
RockParams test_params(uint32_t seed = 1) {
    RockParams p;
    p.seed = seed;
    p.resolution = 24;
    return p;
}

// True if any vertex position differs between the two models' first meshes.
bool geometry_differs(const Model& a, const Model& b) {
    const auto& va = a.meshes[0].vertices;
    const auto& vb = b.meshes[0].vertices;
    if (va.size() != vb.size()) {
        return true;
    }
    for (size_t i = 0; i < va.size(); i++) {
        if (va[i].position != vb[i].position) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST(RockGenerator, GenerateProducesNonEmptyModel) {
    auto model = RockGenerator::generate(test_params());
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(model->loaded);
    ASSERT_EQ(model->meshes.size(), 1u);
    EXPECT_GT(model->meshes[0].vertices.size(), 0u);
    EXPECT_GT(model->meshes[0].indices.size(), 0u);
}

TEST(RockGenerator, IndicesFormValidTriangleList) {
    auto model = RockGenerator::generate(test_params());
    ASSERT_NE(model, nullptr);
    const auto& mesh = model->meshes[0];
    EXPECT_EQ(mesh.indices.size() % 3, 0u);
    const auto vert_count = static_cast<uint32_t>(mesh.vertices.size());
    for (uint32_t index : mesh.indices) {
        ASSERT_LT(index, vert_count);
    }
}

TEST(RockGenerator, SameSeedIsDeterministic) {
    auto a = RockGenerator::generate(test_params(42));
    auto b = RockGenerator::generate(test_params(42));
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    const auto& ma = a->meshes[0];
    const auto& mb = b->meshes[0];
    ASSERT_EQ(ma.vertices.size(), mb.vertices.size());
    ASSERT_EQ(ma.indices.size(), mb.indices.size());
    for (size_t i = 0; i < ma.vertices.size(); i++) {
        ASSERT_EQ(ma.vertices[i].position, mb.vertices[i].position);
        ASSERT_EQ(ma.vertices[i].normal, mb.vertices[i].normal);
        ASSERT_EQ(ma.vertices[i].texcoord, mb.vertices[i].texcoord);
    }
    for (size_t i = 0; i < ma.indices.size(); i++) {
        ASSERT_EQ(ma.indices[i], mb.indices[i]);
    }
}

TEST(RockGenerator, DifferentSeedsProduceDifferentRocks) {
    auto a = RockGenerator::generate(test_params(1));
    auto b = RockGenerator::generate(test_params(2));
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_TRUE(geometry_differs(*a, *b));
}

// Regression test: the seed used to enter the SDF hash as a raw float cast,
// so adjacent seeds above 2^24 collided on identical rocks. The seed is now
// integer-hashed first.
TEST(RockGenerator, AdjacentLargeSeedsBeyondFloatPrecisionDiffer) {
    const uint32_t big = (1u << 24) + 1u; // not exactly representable as float
    auto a = RockGenerator::generate(test_params(big));
    auto b = RockGenerator::generate(test_params(big + 1u));
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_TRUE(geometry_differs(*a, *b));
}

TEST(RockGenerator, FlatBaseSitsAtYZero) {
    auto model = RockGenerator::generate(test_params());
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->bounds.min.y, 0.0f);
    float lowest = 1e9f;
    for (const auto& v : model->meshes[0].vertices) {
        EXPECT_GE(v.position.y, 0.0f);
        lowest = std::min(lowest, v.position.y);
    }
    // The vertices are shifted so the lowest one lands exactly on Y=0.
    EXPECT_FLOAT_EQ(lowest, 0.0f);
}

TEST(RockGenerator, AABBIsSymmetricInXZ) {
    auto model = RockGenerator::generate(test_params());
    ASSERT_NE(model, nullptr);
    EXPECT_FLOAT_EQ(model->bounds.min.x, -model->bounds.max.x);
    EXPECT_FLOAT_EQ(model->bounds.min.z, -model->bounds.max.z);
    EXPECT_GT(model->bounds.max.x, 0.0f);
    EXPECT_GT(model->bounds.max.z, 0.0f);
    EXPECT_GT(model->bounds.max.y, 0.0f);
}

TEST(RockGenerator, VerticesStayWithinGridBounds) {
    RockParams p = test_params();
    p.scale = 2.0f;
    auto model = RockGenerator::generate(p);
    ASSERT_NE(model, nullptr);
    // The SDF grid covers [-0.7, 0.7]^3 (times scale); Y is shifted up by at
    // most the full grid height.
    const float extent = 0.7f * p.scale + 1e-4f;
    for (const auto& v : model->meshes[0].vertices) {
        EXPECT_LE(std::abs(v.position.x), extent);
        EXPECT_LE(std::abs(v.position.z), extent);
        EXPECT_GE(v.position.y, 0.0f);
        EXPECT_LE(v.position.y, 2.0f * extent);
    }
}

TEST(RockGenerator, NormalsAreUnitLength) {
    auto model = RockGenerator::generate(test_params());
    ASSERT_NE(model, nullptr);
    for (const auto& v : model->meshes[0].vertices) {
        EXPECT_NEAR(glm::length(v.normal), 1.0f, 0.01f);
    }
}

TEST(RockGenerator, TotalSurfaceAreaIsPositive) {
    auto model = RockGenerator::generate(test_params());
    ASSERT_NE(model, nullptr);
    const auto& mesh = model->meshes[0];
    float total_area = 0.0f;
    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const glm::vec3& a = mesh.vertices[mesh.indices[i]].position;
        const glm::vec3& b = mesh.vertices[mesh.indices[i + 1]].position;
        const glm::vec3& c = mesh.vertices[mesh.indices[i + 2]].position;
        total_area += 0.5f * glm::length(glm::cross(b - a, c - a));
    }
    EXPECT_GT(total_area, 0.01f);
}

TEST(RockGenerator, ScaleScalesTheMesh) {
    RockParams small = test_params(7);
    small.scale = 1.0f;
    RockParams large = test_params(7);
    large.scale = 2.0f;
    auto a = RockGenerator::generate(small);
    auto b = RockGenerator::generate(large);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_NEAR(b->max_dimension(), 2.0f * a->max_dimension(), 0.01f * a->max_dimension());
}

TEST(RockGenerator, VerticesUseParamColorAndNoTexture) {
    RockParams p = test_params();
    p.color = {0.1f, 0.2f, 0.3f, 1.0f};
    auto model = RockGenerator::generate(p);
    ASSERT_NE(model, nullptr);
    const auto& mesh = model->meshes[0];
    EXPECT_FALSE(mesh.has_texture);
    for (const auto& v : mesh.vertices) {
        ASSERT_EQ(v.color, p.color);
    }
}

TEST(RockGenerator, BoundingSphereIsComputed) {
    auto model = RockGenerator::generate(test_params());
    ASSERT_NE(model, nullptr);
    EXPECT_GT(model->bounding_half_diag, 0.0f);
}

TEST(RockGenerator, LowResolutionStillProducesValidMesh) {
    RockParams p = test_params();
    p.resolution = 12;
    auto model = RockGenerator::generate(p);
    ASSERT_NE(model, nullptr);
    const auto& mesh = model->meshes[0];
    EXPECT_GT(mesh.vertices.size(), 0u);
    EXPECT_EQ(mesh.indices.size() % 3, 0u);
    for (uint32_t index : mesh.indices) {
        ASSERT_LT(index, mesh.vertices.size());
    }
}

TEST(RockGenerator, PresetsReturnValidModels) {
    const std::unique_ptr<Model> models[] = {
        RockGenerator::generate_boulder(5), RockGenerator::generate_slate(5), RockGenerator::generate_spire(5),
        RockGenerator::generate_cluster(5), RockGenerator::generate_mossy(5),
    };
    for (const auto& model : models) {
        ASSERT_NE(model, nullptr);
        EXPECT_TRUE(model->loaded);
        ASSERT_EQ(model->meshes.size(), 1u);
        EXPECT_GT(model->meshes[0].vertices.size(), 0u);
        EXPECT_GT(model->meshes[0].indices.size(), 0u);
    }
}
