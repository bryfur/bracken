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
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

// ============================================================================
// In-memory GLB builders (shared scaffolding for the loader tests below)
// ============================================================================

namespace {

size_t append_data(tinygltf::Buffer& buffer, const void* data, size_t size) {
    // Keep every block 4-byte aligned so float/u32 accessors stay legal.
    while (buffer.data.size() % 4 != 0) {
        buffer.data.push_back(0);
    }
    size_t offset = buffer.data.size();
    const auto* bytes = static_cast<const uint8_t*>(data);
    buffer.data.insert(buffer.data.end(), bytes, bytes + size);
    return offset;
}

int add_view(tinygltf::Model& m, size_t offset, size_t length) {
    tinygltf::BufferView v;
    v.buffer = 0;
    v.byteOffset = offset;
    v.byteLength = length;
    m.bufferViews.push_back(v);
    return static_cast<int>(m.bufferViews.size()) - 1;
}

int add_accessor(tinygltf::Model& m, int view, int component_type, int type, size_t count) {
    tinygltf::Accessor a;
    a.bufferView = view;
    a.componentType = component_type;
    a.type = type;
    a.count = count;
    m.accessors.push_back(a);
    return static_cast<int>(m.accessors.size()) - 1;
}

template<typename T>
int add_data_accessor(tinygltf::Model& m, const std::vector<T>& data, int component_type, int type, size_t count) {
    size_t offset = append_data(m.buffers[0], data.data(), data.size() * sizeof(T));
    return add_accessor(m, add_view(m, offset, data.size() * sizeof(T)), component_type, type, count);
}

std::string write_glb(tinygltf::Model& m, const std::string& name) {
    m.asset.version = "2.0";
    const std::string path = ::testing::TempDir() + name;
    tinygltf::TinyGLTF writer;
    if (!writer.WriteGltfSceneToFile(&m, path, /*embedImages=*/true, /*embedBuffers=*/true,
                                     /*prettyPrint=*/false, /*writeBinary=*/true)) {
        return {};
    }
    return path;
}

// Minimal solid-color 24-bit BMP (SDL_image decodes it; no PNG encoder needed
// in this TU since stb_image_write is compiled out).
std::vector<uint8_t> make_bmp(int w, int h, uint8_t r, uint8_t g, uint8_t b) {
    auto put_u32 = [](std::vector<uint8_t>& v, size_t at, uint32_t x) {
        v[at] = x & 0xFF;
        v[at + 1] = (x >> 8) & 0xFF;
        v[at + 2] = (x >> 16) & 0xFF;
        v[at + 3] = (x >> 24) & 0xFF;
    };
    int row = ((w * 3 + 3) / 4) * 4;
    std::vector<uint8_t> bmp(54 + static_cast<size_t>(row) * h, 0);
    bmp[0] = 'B';
    bmp[1] = 'M';
    put_u32(bmp, 2, static_cast<uint32_t>(bmp.size()));
    put_u32(bmp, 10, 54);                            // pixel data offset
    put_u32(bmp, 14, 40);                            // BITMAPINFOHEADER size
    put_u32(bmp, 18, static_cast<uint32_t>(w));
    put_u32(bmp, 22, static_cast<uint32_t>(h));
    bmp[26] = 1;                                     // planes
    bmp[28] = 24;                                    // bits per pixel
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t at = 54 + static_cast<size_t>(y) * row + static_cast<size_t>(x) * 3;
            bmp[at] = b;
            bmp[at + 1] = g;
            bmp[at + 2] = r;
        }
    }
    return bmp;
}

int add_bmp_image(tinygltf::Model& m, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
    auto bmp = make_bmp(w, h, r, g, b);
    size_t offset = append_data(m.buffers[0], bmp.data(), bmp.size());
    tinygltf::Image image;
    image.bufferView = add_view(m, offset, bmp.size());
    image.mimeType = "image/bmp";
    m.images.push_back(image);
    return static_cast<int>(m.images.size()) - 1;
}

// One-triangle position accessor; returns the accessor index.
int add_triangle_positions(tinygltf::Model& m) {
    std::vector<float> positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    int acc = add_data_accessor(m, positions, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, 3);
    m.accessors[acc].minValues = {0.0, 0.0, 0.0};
    m.accessors[acc].maxValues = {1.0, 1.0, 0.0};
    return acc;
}

void add_default_scene(tinygltf::Model& m, const std::vector<int>& root_nodes) {
    tinygltf::Scene scene;
    scene.nodes = root_nodes;
    m.scenes.push_back(scene);
    m.defaultScene = 0;
}

// Two-joint rig: node 0 "Root" -> node 1 "Child", plus node 2 carrying a
// skinned triangle. skin.joints order and IBM componentType are configurable
// so the joint-order and accessor-validation paths can both be exercised.
void build_two_joint_rig(tinygltf::Model& m, bool child_first, bool ibm_as_u16) {
    m.buffers.emplace_back();

    tinygltf::Node root;
    root.name = "Root";
    root.children = {1};
    m.nodes.push_back(root);

    tinygltf::Node child;
    child.name = "Child";
    child.translation = {0.0, 1.0, 0.0};
    m.nodes.push_back(child);

    // Skinned triangle: JOINTS_0/WEIGHTS_0 index into skin.joints order.
    int pos_acc = add_triangle_positions(m);
    std::vector<uint8_t> joints(12, 0);
    joints[0] = 0;
    joints[1] = 1; // vertex 0 weighted across both joints
    std::vector<float> weights(12, 0.0f);
    weights[0] = 0.5f;
    weights[1] = 0.5f;
    weights[4] = 1.0f; // vertices 1/2 fully on joint 0
    weights[8] = 1.0f;
    int joints_acc = add_data_accessor(m, joints, TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE, TINYGLTF_TYPE_VEC4, 3);
    int weights_acc = add_data_accessor(m, weights, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC4, 3);

    tinygltf::Primitive prim;
    prim.attributes["POSITION"] = pos_acc;
    prim.attributes["JOINTS_0"] = joints_acc;
    prim.attributes["WEIGHTS_0"] = weights_acc;
    prim.mode = TINYGLTF_MODE_TRIANGLES;
    tinygltf::Mesh mesh;
    mesh.primitives.push_back(prim);
    m.meshes.push_back(mesh);

    // IBMs in skin.joints order: Root = identity, Child = translate(0,-1,0).
    glm::mat4 root_ibm(1.0f);
    glm::mat4 child_ibm(1.0f);
    child_ibm[3][1] = -1.0f;
    std::vector<glm::mat4> ibms = child_first ? std::vector<glm::mat4>{child_ibm, root_ibm}
                                              : std::vector<glm::mat4>{root_ibm, child_ibm};
    int ibm_acc;
    if (ibm_as_u16) {
        std::vector<uint16_t> bogus(32, 0);
        ibm_acc = add_data_accessor(m, bogus, TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, TINYGLTF_TYPE_MAT4, 2);
    } else {
        std::vector<float> ibm_floats(32);
        std::memcpy(ibm_floats.data(), ibms.data(), sizeof(glm::mat4) * 2);
        ibm_acc = add_data_accessor(m, ibm_floats, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_MAT4, 2);
    }

    tinygltf::Skin skin;
    skin.joints = child_first ? std::vector<int>{1, 0} : std::vector<int>{0, 1};
    skin.inverseBindMatrices = ibm_acc;
    m.skins.push_back(skin);

    tinygltf::Node mesh_node;
    mesh_node.mesh = 0;
    mesh_node.skin = 0;
    m.nodes.push_back(mesh_node);

    add_default_scene(m, {0, 2});
}

// Appends a channel + sampler to m.animations[0] (created on demand).
void add_channel(tinygltf::Model& m, int node, const std::string& target_path, const std::string& interpolation,
                 int input_acc, int output_acc) {
    if (m.animations.empty()) {
        tinygltf::Animation anim;
        anim.name = "clip";
        m.animations.push_back(anim);
    }
    tinygltf::AnimationSampler sampler;
    sampler.input = input_acc;
    sampler.output = output_acc;
    sampler.interpolation = interpolation;
    m.animations[0].samplers.push_back(sampler);
    tinygltf::AnimationChannel channel;
    channel.sampler = static_cast<int>(m.animations[0].samplers.size()) - 1;
    channel.target_node = node;
    channel.target_path = target_path;
    m.animations[0].channels.push_back(channel);
}

} // namespace

// ============================================================================
// Animation sampler interpolation (STEP honored, CUBICSPLINE values extracted)
// ============================================================================

TEST(ModelLoaderInterpolation, StepAndCubicSplineSamplers) {
    tinygltf::Model m;
    build_two_joint_rig(m, /*child_first=*/false, /*ibm_as_u16=*/false);

    std::vector<float> times = {0.0f, 1.0f};
    int times_acc = add_data_accessor(m, times, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_SCALAR, 2);

    // STEP translation: plain 2-key output.
    std::vector<float> step_positions = {0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f};
    int step_acc = add_data_accessor(m, step_positions, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, 2);
    add_channel(m, /*node=*/1, "translation", "STEP", times_acc, step_acc);

    // CUBICSPLINE rotation: 3 vec4 elements per key (in-tangent, value,
    // out-tangent); tangents are distinctive garbage so a misread shows up.
    std::vector<float> cubic = {
        9.0f, 9.0f, 9.0f, 9.0f, /* value */ 0.0f, 0.0f,        0.0f, 1.0f,        8.0f, 8.0f, 8.0f, 8.0f,
        7.0f, 7.0f, 7.0f, 7.0f, /* value */ 0.0f, 0.70710678f, 0.0f, 0.70710678f, 6.0f, 6.0f, 6.0f, 6.0f,
    };
    int cubic_acc = add_data_accessor(m, cubic, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC4, 6);
    add_channel(m, /*node=*/1, "rotation", "CUBICSPLINE", times_acc, cubic_acc);

    const std::string path = write_glb(m, "bracken_interp.glb");
    ASSERT_FALSE(path.empty()) << "failed to write test glb";

    Model model;
    ASSERT_TRUE(ModelLoader::load_glb(path, model));
    ASSERT_EQ(model.animations.size(), 1u);
    const auto& clip = model.animations[0];
    ASSERT_EQ(clip.channels.size(), 1u); // both paths target the Child joint
    const auto& ch = clip.channels[0];
    EXPECT_EQ(ch.bone_index, 1);
    EXPECT_FLOAT_EQ(clip.duration, 1.0f);

    // STEP survives into the runtime channel and the values are untouched.
    EXPECT_EQ(ch.position_interp, bracken::animation::Interpolation::Step);
    ASSERT_EQ(ch.positions.size(), 2u);
    EXPECT_FLOAT_EQ(ch.positions[1].x, 5.0f);

    // CUBICSPLINE: the middle (in-place value) element of each triplet is
    // extracted — not the tangents — and the track downgrades to Linear.
    EXPECT_EQ(ch.rotation_interp, bracken::animation::Interpolation::Linear);
    ASSERT_EQ(ch.rotations.size(), 2u);
    EXPECT_NEAR(ch.rotations[0].w, 1.0f, 1e-6f);
    EXPECT_NEAR(ch.rotations[0].x, 0.0f, 1e-6f);
    EXPECT_NEAR(ch.rotations[1].w, 0.70710678f, 1e-6f);
    EXPECT_NEAR(ch.rotations[1].y, 0.70710678f, 1e-6f);
}

TEST(ModelLoaderInterpolation, LinearIsTheDefault) {
    tinygltf::Model m;
    build_two_joint_rig(m, /*child_first=*/false, /*ibm_as_u16=*/false);

    std::vector<float> times = {0.0f, 1.0f};
    int times_acc = add_data_accessor(m, times, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_SCALAR, 2);
    std::vector<float> positions = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    int pos_acc = add_data_accessor(m, positions, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, 2);
    add_channel(m, 1, "translation", "LINEAR", times_acc, pos_acc);

    const std::string path = write_glb(m, "bracken_interp_linear.glb");
    ASSERT_FALSE(path.empty());

    Model model;
    ASSERT_TRUE(ModelLoader::load_glb(path, model));
    ASSERT_EQ(model.animations.size(), 1u);
    ASSERT_EQ(model.animations[0].channels.size(), 1u);
    EXPECT_EQ(model.animations[0].channels[0].position_interp, bracken::animation::Interpolation::Linear);
}

// ============================================================================
// Accessor validation: quantized/non-float animation + IBM data must be
// dropped with a warning, never reinterpreted as floats
// ============================================================================

TEST(ModelLoaderAccessorValidation, QuantizedRotationOutputDropsChannel) {
    tinygltf::Model m;
    build_two_joint_rig(m, /*child_first=*/false, /*ibm_as_u16=*/false);

    std::vector<float> times = {0.0f, 1.0f};
    int times_acc = add_data_accessor(m, times, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_SCALAR, 2);
    // Normalized u16 quaternion output (gltfpack-style quantization): valid
    // core glTF, unsupported here — must not be memcpy'd as float.
    std::vector<uint16_t> quantized(8, 0x7FFF);
    int rot_acc = add_data_accessor(m, quantized, TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, TINYGLTF_TYPE_VEC4, 2);
    m.accessors[rot_acc].normalized = true;
    add_channel(m, 1, "rotation", "LINEAR", times_acc, rot_acc);

    const std::string path = write_glb(m, "bracken_quantized_rot.glb");
    ASSERT_FALSE(path.empty());

    Model model;
    ASSERT_TRUE(ModelLoader::load_glb(path, model)); // load itself succeeds
    // The only channel was dropped, so no clip should have been added.
    EXPECT_TRUE(model.animations.empty());
}

TEST(ModelLoaderAccessorValidation, NonFloatInverseBindMatricesFallBackToIdentity) {
    tinygltf::Model m;
    build_two_joint_rig(m, /*child_first=*/false, /*ibm_as_u16=*/true);

    const std::string path = write_glb(m, "bracken_bad_ibm.glb");
    ASSERT_FALSE(path.empty());

    Model model;
    ASSERT_TRUE(ModelLoader::load_glb(path, model));
    ASSERT_TRUE(model.has_skeleton);
    ASSERT_EQ(model.skeleton.joints.size(), 2u);
    for (const auto& joint : model.skeleton.joints) {
        EXPECT_EQ(joint.inverse_bind_matrix, glm::mat4(1.0f)) << "joint " << joint.name;
    }
}

// ============================================================================
// Joint ordering: child-before-parent skins are topologically normalized
// ============================================================================

TEST(ModelLoaderJointOrder, ChildFirstSkinIsReorderedAndRemapped) {
    tinygltf::Model m;
    build_two_joint_rig(m, /*child_first=*/true, /*ibm_as_u16=*/false);

    // Animation channel on the Child node: after reordering it must bind to
    // the Child's new joint index.
    std::vector<float> times = {0.0f, 1.0f};
    int times_acc = add_data_accessor(m, times, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_SCALAR, 2);
    std::vector<float> positions = {0.0f, 1.0f, 0.0f, 0.0f, 2.0f, 0.0f};
    int pos_acc = add_data_accessor(m, positions, TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, 2);
    add_channel(m, 1, "translation", "LINEAR", times_acc, pos_acc);

    const std::string path = write_glb(m, "bracken_child_first.glb");
    ASSERT_FALSE(path.empty());

    Model model;
    ASSERT_TRUE(ModelLoader::load_glb(path, model));
    ASSERT_TRUE(model.has_skeleton);
    ASSERT_EQ(model.skeleton.joints.size(), 2u);

    // Parents-before-children after normalization.
    EXPECT_EQ(model.skeleton.joints[0].name, "Root");
    EXPECT_EQ(model.skeleton.joints[0].parent_index, -1);
    EXPECT_EQ(model.skeleton.joints[1].name, "Child");
    EXPECT_EQ(model.skeleton.joints[1].parent_index, 0);

    // Inverse bind matrices travel with their joints.
    EXPECT_FLOAT_EQ(model.skeleton.joints[0].inverse_bind_matrix[3][1], 0.0f);
    EXPECT_FLOAT_EQ(model.skeleton.joints[1].inverse_bind_matrix[3][1], -1.0f);

    // Vertex joint indices referenced skin order {Child, Root} = {0, 1};
    // after the reorder they must point at {1, 0}.
    ASSERT_EQ(model.meshes.size(), 1u);
    ASSERT_TRUE(model.meshes[0].is_skinned);
    const auto& sv = model.meshes[0].skinned_vertices[0];
    EXPECT_EQ(sv.joints[0], 1); // was 0 = Child
    EXPECT_EQ(sv.joints[1], 0); // was 1 = Root
    EXPECT_FLOAT_EQ(sv.weights[0], 0.5f);
    const auto& sv1 = model.meshes[0].skinned_vertices[1];
    EXPECT_EQ(sv1.joints[0], 1); // fully weighted to old joint 0 = Child

    // The Child-targeting channel binds to the Child's post-reorder index.
    ASSERT_EQ(model.animations.size(), 1u);
    ASSERT_EQ(model.animations[0].channels.size(), 1u);
    EXPECT_EQ(model.animations[0].channels[0].bone_index, 1);
}

TEST(ModelLoaderJointOrder, ParentsFirstSkinIsUntouched) {
    tinygltf::Model m;
    build_two_joint_rig(m, /*child_first=*/false, /*ibm_as_u16=*/false);

    const std::string path = write_glb(m, "bracken_parents_first.glb");
    ASSERT_FALSE(path.empty());

    Model model;
    ASSERT_TRUE(ModelLoader::load_glb(path, model));
    ASSERT_EQ(model.skeleton.joints.size(), 2u);
    EXPECT_EQ(model.skeleton.joints[0].name, "Root");
    EXPECT_EQ(model.skeleton.joints[1].name, "Child");
    EXPECT_EQ(model.skeleton.joints[1].parent_index, 0);
    const auto& sv = model.meshes[0].skinned_vertices[0];
    EXPECT_EQ(sv.joints[0], 0);
    EXPECT_EQ(sv.joints[1], 1);
}

// ============================================================================
// Base-color image lookup uses raw glTF image indices (no compaction skew)
// ============================================================================

TEST(ModelLoaderImages, EmptyImageDoesNotShiftLaterBaseColorTextures) {
    tinygltf::Model m;
    m.buffers.emplace_back();

    // Image 0: external URI that cannot resolve — tinygltf keeps it as an
    // EMPTY image and only warns. Image 1: an embedded solid-color BMP.
    tinygltf::Image missing;
    missing.uri = "bracken_no_such_image_c1a7.png";
    m.images.push_back(missing);
    add_bmp_image(m, 4, 2, 10, 20, 30);

    for (int i = 0; i < 2; i++) {
        tinygltf::Texture tex;
        tex.source = i;
        m.textures.push_back(tex);
        tinygltf::Material mat;
        mat.pbrMetallicRoughness.baseColorTexture.index = i;
        m.materials.push_back(mat);
    }

    int pos_acc = add_triangle_positions(m);
    tinygltf::Mesh mesh;
    for (int i = 0; i < 2; i++) {
        tinygltf::Primitive prim;
        prim.attributes["POSITION"] = pos_acc;
        prim.material = i;
        prim.mode = TINYGLTF_MODE_TRIANGLES;
        mesh.primitives.push_back(prim);
    }
    m.meshes.push_back(mesh);
    tinygltf::Node node;
    node.mesh = 0;
    m.nodes.push_back(node);
    add_default_scene(m, {0});

    const std::string path = write_glb(m, "bracken_image_skew.glb");
    ASSERT_FALSE(path.empty());

    Model model;
    ASSERT_TRUE(ModelLoader::load_glb(path, model));
    ASSERT_EQ(model.meshes.size(), 2u);

    // Mesh 0 referenced the unresolvable image: untextured, not image 1's pixels.
    EXPECT_FALSE(model.meshes[0].has_texture);
    EXPECT_TRUE(model.meshes[0].texture_pixels.empty());

    // Mesh 1 must get image 1 (4x2 solid 10/20/30), not be dropped/shifted.
    ASSERT_TRUE(model.meshes[1].has_texture);
    EXPECT_EQ(model.meshes[1].texture_width, 4);
    EXPECT_EQ(model.meshes[1].texture_height, 2);
    ASSERT_GE(model.meshes[1].texture_pixels.size(), 4u);
    EXPECT_EQ(model.meshes[1].texture_pixels[0], 10); // R
    EXPECT_EQ(model.meshes[1].texture_pixels[1], 20); // G
    EXPECT_EQ(model.meshes[1].texture_pixels[2], 30); // B
}

// ============================================================================
// Occlusion: ORM-packed is honored, standalone AO maps are dropped
// ============================================================================

TEST(ModelLoaderOcclusion, PackedIsFlaggedStandaloneIsDropped) {
    tinygltf::Model m;
    m.buffers.emplace_back();

    int orm_img = add_bmp_image(m, 2, 2, 255, 128, 0); // shared MR+AO image
    int ao_img = add_bmp_image(m, 2, 2, 200, 200, 200); // standalone AO image
    for (int i = 0; i < 2; i++) {
        tinygltf::Texture tex;
        tex.source = i;
        m.textures.push_back(tex);
    }

    tinygltf::Material packed;
    packed.pbrMetallicRoughness.metallicRoughnessTexture.index = orm_img;
    packed.occlusionTexture.index = orm_img; // same image => ORM layout
    m.materials.push_back(packed);

    tinygltf::Material standalone;
    standalone.pbrMetallicRoughness.metallicRoughnessTexture.index = orm_img;
    standalone.occlusionTexture.index = ao_img; // different image => dropped
    standalone.occlusionTexture.strength = 0.6;
    m.materials.push_back(standalone);

    int pos_acc = add_triangle_positions(m);
    tinygltf::Mesh mesh;
    for (int i = 0; i < 2; i++) {
        tinygltf::Primitive prim;
        prim.attributes["POSITION"] = pos_acc;
        prim.material = i;
        prim.mode = TINYGLTF_MODE_TRIANGLES;
        mesh.primitives.push_back(prim);
    }
    m.meshes.push_back(mesh);
    tinygltf::Node node;
    node.mesh = 0;
    m.nodes.push_back(node);
    add_default_scene(m, {0});

    const std::string path = write_glb(m, "bracken_occlusion.glb");
    ASSERT_FALSE(path.empty());

    Model model;
    ASSERT_TRUE(ModelLoader::load_glb(path, model));
    ASSERT_EQ(model.meshes.size(), 2u);

    EXPECT_TRUE(model.meshes[0].material.has_packed_occlusion);
    EXPECT_FALSE(model.meshes[0].material.metallic_roughness_pixels.empty());

    // Standalone AO: no packed flag; the scalar strength still parses (it
    // only applies if the asset is later fixed to ORM packing).
    EXPECT_FALSE(model.meshes[1].material.has_packed_occlusion);
    EXPECT_FLOAT_EQ(model.meshes[1].material.occlusion_strength, 0.6f);
}

// ============================================================================
// reload_model keeps raw Model* stable (contents replaced in place)
// ============================================================================

namespace {

// Writes a GLB with `primitive_count` copies of one triangle to `path`.
bool write_triangle_glb_at(const std::string& path, int primitive_count) {
    tinygltf::Model m;
    m.buffers.emplace_back();
    int pos_acc = add_triangle_positions(m);
    tinygltf::Mesh mesh;
    for (int i = 0; i < primitive_count; i++) {
        tinygltf::Primitive prim;
        prim.attributes["POSITION"] = pos_acc;
        prim.mode = TINYGLTF_MODE_TRIANGLES;
        mesh.primitives.push_back(prim);
    }
    m.meshes.push_back(mesh);
    tinygltf::Node node;
    node.mesh = 0;
    m.nodes.push_back(node);
    add_default_scene(m, {0});
    m.asset.version = "2.0";
    tinygltf::TinyGLTF writer;
    return writer.WriteGltfSceneToFile(&m, path, true, true, false, true);
}

} // namespace

TEST(ModelManagerReload, ReloadKeepsRawModelPointerStable) {
    const std::string path = ::testing::TempDir() + "bracken_reload.glb";
    ASSERT_TRUE(write_triangle_glb_at(path, 1));

    ModelManager mgr;
    ModelHandle handle = mgr.load_model("reloadable", path);
    ASSERT_NE(handle, INVALID_MODEL_HANDLE);
    Model* before = mgr.get_model(handle);
    ASSERT_NE(before, nullptr);
    EXPECT_EQ(before->meshes.size(), 1u);

    // Overwrite the source and reload: the handle AND the raw pointer must
    // stay valid, with the contents replaced.
    ASSERT_TRUE(write_triangle_glb_at(path, 2));
    ASSERT_TRUE(mgr.reload_model(handle));
    EXPECT_EQ(mgr.get_model(handle), before);
    EXPECT_EQ(before->meshes.size(), 2u);
    EXPECT_TRUE(before->loaded);
}

TEST(ModelManagerReload, FailedReloadLeavesModelUntouched) {
    const std::string path = ::testing::TempDir() + "bracken_reload_fail.glb";
    ASSERT_TRUE(write_triangle_glb_at(path, 1));

    ModelManager mgr;
    ModelHandle handle = mgr.load_model("reloadable_fail", path);
    ASSERT_NE(handle, INVALID_MODEL_HANDLE);
    Model* before = mgr.get_model(handle);
    ASSERT_NE(before, nullptr);

    // Corrupt the file: reload must fail and leave the old contents alone.
    {
        std::string garbage = "not a glb";
        FILE* f = fopen(path.c_str(), "wb");
        ASSERT_NE(f, nullptr);
        fwrite(garbage.data(), 1, garbage.size(), f);
        fclose(f);
    }
    EXPECT_FALSE(mgr.reload_model(handle));
    EXPECT_EQ(mgr.get_model(handle), before);
    EXPECT_EQ(before->meshes.size(), 1u);
}

// ===========================================================================
// RESIDENCY: unload_model / reupload_model / gpu_bytes
// (docs/gap_streaming.md piece b)
//
// Device-free, as everything else in this file is: with no device set,
// register_model skips the upload, so there are no GPU buffers to free and
// gpu_bytes is 0 throughout. What that leaves testable is exactly what the
// residency contract is ABOUT — the slot, the handle, the name mapping and the
// `loaded` flag, which is the flag every consumer (the Prepare stage, the BLAS
// walk) tests.
// ===========================================================================

namespace {

/// register_model refuses a model with no geometry, so give it one triangle.
std::unique_ptr<bracken::Model> make_residency_model(float half_extent) {
    auto m = std::make_unique<bracken::Model>();
    bracken::Mesh mesh;
    mesh.vertices.resize(3);
    mesh.indices = {0, 1, 2};
    m->meshes.push_back(std::move(mesh));
    m->bounds.min = glm::vec3(-half_extent);
    m->bounds.max = glm::vec3(half_extent);
    m->compute_bounding_sphere();
    return m;
}

} // namespace

TEST(ModelManagerResidency, UnloadKeepsTheSlotTheNameAndTheHandle) {
    // THE INVARIANT rt_scene.cpp's all-handles walk depends on: model_count()
    // is unchanged by an eviction. If a slot were removed or compacted, every
    // handle after it would silently mean a different model.
    ModelManager mgr;
    const auto a = mgr.register_model("a", make_residency_model(1.0f));
    const auto b = mgr.register_model("b", make_residency_model(2.0f));
    ASSERT_TRUE(a.is_ok());
    ASSERT_TRUE(b.is_ok());
    const std::size_t count_before = mgr.model_count();
    Model* const b_ptr = mgr.get_model(b.value());
    ASSERT_NE(b_ptr, nullptr);

    (void)mgr.unload_model(a.value());

    EXPECT_EQ(mgr.model_count(), count_before);
    EXPECT_EQ(mgr.get_handle("a"), a.value());
    EXPECT_EQ(mgr.get_handle("b"), b.value());
    ASSERT_NE(mgr.get_model(a.value()), nullptr);
    EXPECT_FALSE(mgr.get_model(a.value())->loaded);
    // The untouched neighbour keeps its handle, its pointer and its payload.
    EXPECT_EQ(mgr.get_model(b.value()), b_ptr);
    EXPECT_TRUE(b_ptr->loaded);
}

TEST(ModelManagerResidency, UnloadDropsRetainedCpuGeometry) {
    // The other half of an evicted model's footprint, and for a ray-traced
    // scene the larger half. It is also precisely why the acceleration
    // structure has to be rebuilt after an eviction: its BLAS triangles were
    // copied out of these arrays.
    ModelManager mgr;
    mgr.set_retain_cpu_geometry(true);
    const auto h = mgr.register_model("prop", make_residency_model(1.0f));
    ASSERT_TRUE(h.is_ok());
    ASSERT_FALSE(mgr.get_model(h.value())->meshes.empty());
    ASSERT_EQ(mgr.get_model(h.value())->meshes[0].vertices.size(), 3u);

    (void)mgr.unload_model(h.value());

    EXPECT_TRUE(mgr.get_model(h.value())->meshes[0].vertices.empty());
    EXPECT_TRUE(mgr.get_model(h.value())->meshes[0].indices.empty());
}

TEST(ModelManagerResidency, UnloadingTwiceIsANoOp) {
    ModelManager mgr;
    const auto h = mgr.register_model("prop", make_residency_model(1.0f));
    ASSERT_TRUE(h.is_ok());
    (void)mgr.unload_model(h.value());
    EXPECT_EQ(mgr.unload_model(h.value()), 0u);
    EXPECT_FALSE(mgr.get_model(h.value())->loaded);
}

TEST(ModelManagerResidency, UnloadingAnUnknownHandleIsANoOp) {
    ModelManager mgr;
    EXPECT_EQ(mgr.unload_model(INVALID_MODEL_HANDLE), 0u);
    EXPECT_EQ(mgr.unload_model(static_cast<ModelHandle>(999)), 0u);
}

TEST(ModelManagerResidency, ReuploadRestoresTheSameSlot) {
    ModelManager mgr;
    const auto h = mgr.register_model("prop", make_residency_model(1.0f));
    ASSERT_TRUE(h.is_ok());
    Model* const before = mgr.get_model(h.value());
    (void)mgr.unload_model(h.value());
    ASSERT_FALSE(before->loaded);

    ASSERT_TRUE(mgr.reupload_model(h.value(), make_residency_model(3.0f)));

    EXPECT_TRUE(mgr.get_model(h.value())->loaded);
    // Same handle, same Model* — the contract every draw command still holding
    // the old handle depends on.
    EXPECT_EQ(mgr.get_model(h.value()), before);
    EXPECT_EQ(mgr.get_handle("prop"), h.value());
    // The freshly decoded bounds took effect, and the bounding sphere was
    // recomputed from them (forgetting that culls a model into invisibility
    // with no message anywhere).
    EXPECT_FLOAT_EQ(mgr.get_model(h.value())->bounds.max.x, 3.0f);
    EXPECT_GT(mgr.get_model(h.value())->bounding_half_diag, 0.0f);
}

TEST(ModelManagerResidency, ReuploadPreservesTheSlotsLodChain) {
    // A decode knows nothing about levels generated for this model. Letting the
    // fresh contents overwrite them would leave the level models registered,
    // unreachable, and select_lod silently returning the base forever.
    ModelManager mgr;
    const auto base = mgr.register_model("base", make_residency_model(1.0f));
    const auto level = mgr.register_model("base_lod1", make_residency_model(1.0f));
    ASSERT_TRUE(base.is_ok());
    ASSERT_TRUE(level.is_ok());
    mgr.get_model(base.value())->lod_handles = {level.value()};
    mgr.get_model(base.value())->lod_thresholds = {0.05f};

    (void)mgr.unload_model(base.value());
    ASSERT_TRUE(mgr.reupload_model(base.value(), make_residency_model(1.0f)));

    ASSERT_EQ(mgr.get_model(base.value())->lod_handles.size(), 1u);
    EXPECT_EQ(mgr.get_model(base.value())->lod_handles[0], level.value());
    ASSERT_EQ(mgr.get_model(base.value())->lod_thresholds.size(), 1u);
}

TEST(ModelManagerResidency, ReuploadRefusesAnUnknownHandleOrANullModel) {
    ModelManager mgr;
    const auto h = mgr.register_model("prop", make_residency_model(1.0f));
    ASSERT_TRUE(h.is_ok());
    EXPECT_FALSE(mgr.reupload_model(static_cast<ModelHandle>(999), make_residency_model(1.0f)));
    EXPECT_FALSE(mgr.reupload_model(INVALID_MODEL_HANDLE, make_residency_model(1.0f)));
    EXPECT_FALSE(mgr.reupload_model(h.value(), nullptr));
    EXPECT_TRUE(mgr.get_model(h.value())->loaded) << "a refused reupload must not have unloaded anything";
}

TEST(ModelManagerResidency, GpuBytesIsZeroWithoutADeviceAndForUnknownHandles) {
    // Honest about what this fixture can say: with no device nothing is
    // uploaded, so the estimate is 0. The number's SHAPE is tested in
    // texture_cache.test.cpp (texture_bytes_estimate); what is asserted here is
    // that a missing model answers 0 rather than reading a null slot.
    ModelManager mgr;
    const auto h = mgr.register_model("prop", make_residency_model(1.0f));
    ASSERT_TRUE(h.is_ok());
    EXPECT_EQ(mgr.gpu_bytes(h.value()), 0u);
    EXPECT_EQ(mgr.gpu_bytes(INVALID_MODEL_HANDLE), 0u);
    EXPECT_EQ(mgr.gpu_bytes(static_cast<ModelHandle>(999)), 0u);
}
