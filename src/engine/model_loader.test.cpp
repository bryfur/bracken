#include "engine/model_loader.hpp"
#include "engine/gpu/gpu_types.hpp"
#include <gtest/gtest.h>

using namespace mmo::engine;

TEST(ModelManager, InvalidHandleIsZero) {
    EXPECT_EQ(INVALID_MODEL_HANDLE, 0u);
}

TEST(ModelManager, GetModelInvalidHandleReturnsNull) {
    ModelManager mgr;
    EXPECT_EQ(mgr.get_model(INVALID_MODEL_HANDLE), nullptr);
}

TEST(ModelManager, GetModelUnknownHandleReturnsNull) {
    ModelManager mgr;
    EXPECT_EQ(mgr.get_model(static_cast<ModelHandle>(999)), nullptr);
}

TEST(ModelManager, GetHandleUnknownNameReturnsInvalid) {
    ModelManager mgr;
    EXPECT_EQ(mgr.get_handle("nonexistent"), INVALID_MODEL_HANDLE);
}

TEST(ModelManager, GetModelByUnknownNameReturnsNull) {
    ModelManager mgr;
    EXPECT_EQ(mgr.get_model(std::string("nonexistent")), nullptr);
}

// Defaults the renderer relies on when a glb carries no material data: the
// glTF-mandated alpha defaults, no emission, and no ORM-packed occlusion.
TEST(PBRMaterial, DefaultsMatchGltfSpec) {
    PBRMaterial mat;
    EXPECT_EQ(mat.alpha_mode, AlphaMode::Opaque);
    EXPECT_FLOAT_EQ(mat.alpha_cutoff, 0.5f);
    EXPECT_FALSE(mat.double_sided);
    EXPECT_FALSE(mat.has_packed_occlusion);
    EXPECT_EQ(mat.emissive_factor, glm::vec3(0.0f));
    EXPECT_EQ(mat.metallic_roughness_texture, nullptr);
    EXPECT_EQ(mat.emissive_texture, nullptr);
}

// Vertex3D must expose a tangent member so the GLTF loader can stash both the
// tangent direction and the bitangent sign as a single vec4 (glTF convention).
TEST(Vertex3DTangent, VertexExposesTangentVec4) {
    gpu::Vertex3D v;
    v.tangent = glm::vec4(0.0f, 1.0f, 0.0f, -1.0f);
    EXPECT_FLOAT_EQ(v.tangent.x, 0.0f);
    EXPECT_FLOAT_EQ(v.tangent.y, 1.0f);
    EXPECT_FLOAT_EQ(v.tangent.z, 0.0f);
    EXPECT_FLOAT_EQ(v.tangent.w, -1.0f);
}

TEST(Vertex3DTangent, DefaultTangentIsValid) {
    gpu::Vertex3D v;
    // Default-constructed vertex must have a usable tangent (non-zero, sign +1)
    // so meshes that bypass the loader still render without NaN normal maps.
    glm::vec3 t(v.tangent.x, v.tangent.y, v.tangent.z);
    EXPECT_GT(glm::length(t), 0.5f);
    EXPECT_TRUE(v.tangent.w == 1.0f || v.tangent.w == -1.0f);
}
