#include "bracken/model_loader.hpp"
#include "bracken/gpu/gpu_types.hpp"
#include <gtest/gtest.h>

// Declarations only — the tinygltf implementation is compiled into
// model_loader.cpp (TINYGLTF_IMPLEMENTATION); this TU just builds test GLBs.
// The NO_STB defines must match model_loader.cpp or the inline TinyGLTF
// constructor references stb-based image callbacks that were never compiled.
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include <cstdint>
#include <cstring>
#include <string>

using namespace bracken;

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

namespace {

// Writes a single-mesh GLB with three primitives sharing one triangle, each
// bound to a different alpha/cull material configuration:
//   [0] MASK, cutoff 0.35, doubleSided   [1] BLEND   [2] glTF defaults (OPAQUE)
std::string write_alpha_flags_glb() {
    tinygltf::Model m;
    m.asset.version = "2.0";

    // One buffer: 3 float3 positions followed by 3 uint16 indices.
    const float positions[9] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    const uint16_t indices[3] = {0, 1, 2};
    tinygltf::Buffer buffer;
    buffer.data.resize(sizeof(positions) + sizeof(indices));
    std::memcpy(buffer.data.data(), positions, sizeof(positions));
    std::memcpy(buffer.data.data() + sizeof(positions), indices, sizeof(indices));
    m.buffers.push_back(buffer);

    tinygltf::BufferView pos_view;
    pos_view.buffer = 0;
    pos_view.byteOffset = 0;
    pos_view.byteLength = sizeof(positions);
    pos_view.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    m.bufferViews.push_back(pos_view);

    tinygltf::BufferView idx_view;
    idx_view.buffer = 0;
    idx_view.byteOffset = sizeof(positions);
    idx_view.byteLength = sizeof(indices);
    idx_view.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
    m.bufferViews.push_back(idx_view);

    tinygltf::Accessor pos_accessor;
    pos_accessor.bufferView = 0;
    pos_accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    pos_accessor.count = 3;
    pos_accessor.type = TINYGLTF_TYPE_VEC3;
    pos_accessor.minValues = {0.0, 0.0, 0.0};
    pos_accessor.maxValues = {1.0, 1.0, 0.0};
    m.accessors.push_back(pos_accessor);

    tinygltf::Accessor idx_accessor;
    idx_accessor.bufferView = 1;
    idx_accessor.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
    idx_accessor.count = 3;
    idx_accessor.type = TINYGLTF_TYPE_SCALAR;
    m.accessors.push_back(idx_accessor);

    tinygltf::Material mask_mat;
    mask_mat.alphaMode = "MASK";
    mask_mat.alphaCutoff = 0.35;
    mask_mat.doubleSided = true;
    m.materials.push_back(mask_mat);

    tinygltf::Material blend_mat;
    blend_mat.alphaMode = "BLEND";
    m.materials.push_back(blend_mat);

    m.materials.push_back(tinygltf::Material{}); // glTF defaults: OPAQUE / 0.5 / single-sided

    tinygltf::Mesh mesh;
    for (int mat = 0; mat < 3; ++mat) {
        tinygltf::Primitive prim;
        prim.attributes["POSITION"] = 0;
        prim.indices = 1;
        prim.material = mat;
        prim.mode = TINYGLTF_MODE_TRIANGLES;
        mesh.primitives.push_back(prim);
    }
    m.meshes.push_back(mesh);

    tinygltf::Node node;
    node.mesh = 0;
    m.nodes.push_back(node);

    tinygltf::Scene scene;
    scene.nodes.push_back(0);
    m.scenes.push_back(scene);
    m.defaultScene = 0;

    const std::string path = ::testing::TempDir() + "bracken_alpha_flags.glb";
    tinygltf::TinyGLTF writer;
    if (!writer.WriteGltfSceneToFile(&m, path, /*embedImages=*/true, /*embedBuffers=*/true,
                                     /*prettyPrint=*/false, /*writeBinary=*/true)) {
        return {};
    }
    return path;
}

} // namespace

// The loader must propagate glTF alphaMode / alphaCutoff / doubleSided per
// primitive: the renderer derives the MASK discard cutoff, the sorted BLEND
// path, and the no-cull pipeline variants from exactly these fields.
TEST(ModelLoaderAlphaFlags, PropagatesAlphaModeCutoffAndDoubleSided) {
    const std::string path = write_alpha_flags_glb();
    ASSERT_FALSE(path.empty()) << "failed to write test glb";

    Model model;
    ASSERT_TRUE(ModelLoader::load_glb(path, model));
    ASSERT_EQ(model.meshes.size(), 3u);

    const PBRMaterial& mask = model.meshes[0].material;
    EXPECT_EQ(mask.alpha_mode, AlphaMode::Mask);
    EXPECT_FLOAT_EQ(mask.alpha_cutoff, 0.35f);
    EXPECT_TRUE(mask.double_sided);

    const PBRMaterial& blend = model.meshes[1].material;
    EXPECT_EQ(blend.alpha_mode, AlphaMode::Blend);
    EXPECT_FALSE(blend.double_sided);

    const PBRMaterial& opaque = model.meshes[2].material;
    EXPECT_EQ(opaque.alpha_mode, AlphaMode::Opaque);
    EXPECT_FLOAT_EQ(opaque.alpha_cutoff, 0.5f);
    EXPECT_FALSE(opaque.double_sided);
}
