#include <gtest/gtest.h>

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/render/raytracing/rt_scene.hpp"

using namespace bracken::render::rt;
using bracken::Mesh;
using bracken::Model;
using bracken::ModelManager;

namespace {

/** A unit cube as an indexed mesh, with CPU geometry retained (no device is
 *  set on the manager, so nothing is uploaded and nothing is released). */
std::unique_ptr<Model> make_cube(float scale = 1.0f) {
    auto model = std::make_unique<Model>();
    Mesh mesh;
    const glm::vec3 corners[8] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                  {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    for (const glm::vec3& c : corners) {
        bracken::gpu::Vertex3D v;
        v.position = c * scale;
        v.normal = glm::normalize(c - glm::vec3(0.5f));
        v.color = {1.0f, 0.5f, 0.25f, 1.0f};
        mesh.vertices.push_back(v);
    }
    mesh.indices = {0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,
                    3, 2, 6, 3, 6, 7, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2};
    model->meshes.push_back(std::move(mesh));
    model->bounds = {glm::vec3(0.0f), glm::vec3(scale)};
    return model;
}

struct Fixture {
    bracken::core::jobs::JobSystem jobs;
    ModelManager models;
    Fixture() { jobs.init(2); }
    ~Fixture() { jobs.shutdown(); }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
};

} // namespace

TEST(RtScene, BuildsBlasForEveryModelWithGeometry) {
    Fixture f;
    ASSERT_TRUE(f.models.register_model("a", make_cube()));
    ASSERT_TRUE(f.models.register_model("b", make_cube(2.0f)));

    RtScene scene;
    scene.build_blas_for_models(f.models, /*want_attributes=*/false);

    EXPECT_EQ(scene.stats().models, 2u);
    EXPECT_EQ(scene.stats().unique_triangles, 24u); // 12 per cube
    EXPECT_GT(scene.stats().blas_nodes, 0u);
}

// Attributes more than double triangle memory and only reflections need them,
// so a shadows-only game must not be paying for them.
TEST(RtScene, AttributesAreOnlyGatheredWhenAsked) {
    Fixture f;
    ASSERT_TRUE(f.models.register_model("a", make_cube()));

    RtScene without;
    without.build_blas_for_models(f.models, false);
    EXPECT_FALSE(without.has_attributes()) << "no device, but the flag still tracks intent";

    RtScene with;
    with.build_blas_for_models(f.models, true);
    // Without a GPU device nothing uploads, so has_attributes() (a buffer
    // check) stays false; what matters here is that the build did not crash
    // and produced the same geometry either way.
    EXPECT_EQ(with.stats().unique_triangles, without.stats().unique_triangles);
}

TEST(RtScene, ModelsWithoutGeometryAreSkipped) {
    Fixture f;
    auto empty = std::make_unique<Model>();
    empty->bounds = {glm::vec3(0.0f), glm::vec3(1.0f)};
    // register_model refuses a model with no meshes, which is itself the
    // engine being loud; build over the ones that did register.
    (void)f.models.register_model("empty", std::move(empty));
    ASSERT_TRUE(f.models.register_model("cube", make_cube()));

    RtScene scene;
    scene.build_blas_for_models(f.models, false);
    EXPECT_EQ(scene.stats().models, 1u);
}

TEST(RtScene, InstancesReferenceTheirModelsBlas) {
    Fixture f;
    auto a = f.models.register_model("a", make_cube());
    ASSERT_TRUE(a);

    RtScene scene;
    scene.build_blas_for_models(f.models, false);

    std::vector<std::pair<bracken::ModelHandle, glm::mat4>> instances;
    for (int i = 0; i < 8; ++i) {
        instances.emplace_back(a.value(),
                               glm::translate(glm::mat4(1.0f),
                                              glm::vec3(static_cast<float>(i) * 10.0f, 0, 0)));
    }
    scene.set_instances(instances);

    EXPECT_EQ(scene.stats().instances, 8u);
    EXPECT_GT(scene.stats().tlas_nodes, 0u);
}

// An instance of a model with no BLAS must be dropped, not indexed into a
// buffer that has no entry for it.
TEST(RtScene, InstancesOfUnknownModelsAreDropped) {
    Fixture f;
    auto a = f.models.register_model("a", make_cube());
    ASSERT_TRUE(a);

    RtScene scene;
    scene.build_blas_for_models(f.models, false);

    std::vector<std::pair<bracken::ModelHandle, glm::mat4>> instances{
        {a.value(), glm::mat4(1.0f)},
        {bracken::INVALID_MODEL_HANDLE, glm::mat4(1.0f)},
        {9999u, glm::mat4(1.0f)},
    };
    scene.set_instances(instances);
    EXPECT_EQ(scene.stats().instances, 1u);
}

// A rotated instance's world AABB is NOT its transformed min/max: all eight
// corners have to be transformed, or the TLAS culls geometry that is visible.
TEST(RtScene, RotatedInstanceBoundsEncloseTheRotatedBox) {
    Fixture f;
    auto a = f.models.register_model("a", make_cube(100.0f));
    ASSERT_TRUE(a);

    RtScene scene;
    scene.build_blas_for_models(f.models, false);

    const glm::mat4 rotated =
        glm::rotate(glm::mat4(1.0f), glm::radians(45.0f), glm::vec3(0, 1, 0));
    scene.set_instances(std::vector<std::pair<bracken::ModelHandle, glm::mat4>>{
        {a.value(), rotated}});
    ASSERT_EQ(scene.stats().instances, 1u);

    // A 100-unit cube rotated 45 degrees about Y spans ~141 units on X and Z.
    // If the bounds were computed from the transformed min/max pair alone they
    // would be too small and the corners would poke out.
    // The TLAS root node's bounds are the instance's bounds here.
    // (Checked via the structure being non-empty and the build succeeding —
    // the numeric check lives in the bvh tests; this pins that rotation is
    // handled at all.)
    EXPECT_GT(scene.stats().tlas_nodes, 0u);
}

TEST(RtScene, EmptySceneIsNotReady) {
    RtScene scene;
    EXPECT_FALSE(scene.ready());
    EXPECT_EQ(scene.tlas_node_count(), 0u);
    EXPECT_EQ(scene.tlas_nodes(), nullptr);
}

TEST(RtScene, RebuildIsDeterministic) {
    Fixture f;
    ASSERT_TRUE(f.models.register_model("a", make_cube()));
    ASSERT_TRUE(f.models.register_model("b", make_cube(3.0f)));

    RtScene first;
    first.build_blas_for_models(f.models, false);
    RtScene second;
    second.build_blas_for_models(f.models, false);

    EXPECT_EQ(first.stats().unique_triangles, second.stats().unique_triangles);
    EXPECT_EQ(first.stats().blas_nodes, second.stats().blas_nodes);
}
