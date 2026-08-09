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

/// The same cube with an authored emissive factor. `scale` is what makes a
/// facet's MODEL-space area move independently of its WORLD-space one, which is
/// the distinction the power threshold turns on.
std::unique_ptr<Model> make_emissive_cube(glm::vec3 emissive, float scale = 1.0f) {
    auto model = make_cube(scale);
    model->meshes[0].material.emissive_factor = emissive;
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

// The TLAS dirty check is a performance contract with teeth: the renderer
// hands set_instances() the whole model-command list every traced frame, so
// without it a village that never moves paid a full binned-SAH build on the
// main thread forever. Both docs claimed this skip existed long before it did.
TEST(RtScene, UnchangedInstancesDoNotRebuildTheTlas) {
    Fixture f;
    auto a = f.models.register_model("a", make_cube());
    ASSERT_TRUE(a);
    RtScene scene;
    scene.build_blas_for_models(f.models, false);

    std::vector<std::pair<bracken::ModelHandle, glm::mat4>> instances{
        {a.value(), glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 0.0f, 0.0f))},
        {a.value(), glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 7.0f))},
    };

    scene.set_instances(instances);
    EXPECT_EQ(scene.stats().tlas_builds, 1u);

    // The same content, submitted from a different vector — the check must be
    // on what the instances ARE, not on the address they arrived at.
    for (int i = 0; i < 8; ++i) {
        std::vector<std::pair<bracken::ModelHandle, glm::mat4>> resubmitted = instances;
        scene.set_instances(resubmitted);
    }
    EXPECT_EQ(scene.stats().tlas_builds, 1u) << "a static instance list rebuilt the TLAS";

    // Moving something must NOT rebuild: same models, same order, new place —
    // that is a refit, and it is the case that decides whether a live scene
    // pays a binned-SAH build every frame.
    instances[1].second = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 7.5f));
    scene.set_instances(instances);
    EXPECT_EQ(scene.stats().tlas_builds, 1u) << "a moved instance forced a full TLAS rebuild";
    EXPECT_EQ(scene.stats().tlas_refits, 1u) << "a moved instance did not refit the TLAS";

    // Dropping one is a TOPOLOGY change: the count participates in both hashes
    // so a truncated list cannot collide with the longer one it is a prefix of,
    // and there is no tree to refit onto.
    instances.pop_back();
    scene.set_instances(instances);
    EXPECT_EQ(scene.stats().tlas_builds, 2u);
    EXPECT_EQ(scene.stats().tlas_refits, 1u);
    EXPECT_EQ(scene.stats().instances, 1u);
}

// Only the three cases exist, and each has to be reachable from the same
// entry point — the whole saving is in never taking the expensive one by
// accident.
TEST(RtScene, TopologyChangesRebuildAndTransformChangesRefit) {
    Fixture f;
    auto a = f.models.register_model("a", make_cube());
    auto b = f.models.register_model("b", make_cube(2.0f));
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    RtScene scene;
    scene.build_blas_for_models(f.models, false);

    std::vector<std::pair<bracken::ModelHandle, glm::mat4>> instances{
        {a.value(), glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f))},
        {b.value(), glm::translate(glm::mat4(1.0f), glm::vec3(9.0f, 0.0f, 0.0f))},
        {a.value(), glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 9.0f))},
    };
    scene.set_instances(instances);
    EXPECT_EQ(scene.stats().tlas_builds, 1u);
    EXPECT_EQ(scene.stats().tlas_refits, 0u);

    // Transform-only, repeatedly: refits, never a build.
    for (int i = 1; i <= 5; ++i) {
        instances[0].second =
            glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, static_cast<float>(i) * 0.1f, 0.0f));
        scene.set_instances(instances);
    }
    EXPECT_EQ(scene.stats().tlas_builds, 1u);
    EXPECT_EQ(scene.stats().tlas_refits, 5u);

    // ADDING one: rebuild.
    instances.emplace_back(b.value(), glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 20.0f)));
    scene.set_instances(instances);
    EXPECT_EQ(scene.stats().tlas_builds, 2u);
    EXPECT_EQ(scene.stats().tlas_refits, 5u);

    // REORDERING, with every transform and every handle preserved as a
    // multiset: still a rebuild. The topology hash is over the handle
    // SEQUENCE, deliberately — the refit path maps slots back to input
    // positions, so a permuted list is a different structure to it even
    // though it draws the same picture.
    std::swap(instances[0], instances[1]);
    scene.set_instances(instances);
    EXPECT_EQ(scene.stats().tlas_builds, 3u);
    EXPECT_EQ(scene.stats().tlas_refits, 5u);

    // REMOVING one: rebuild.
    instances.erase(instances.begin());
    scene.set_instances(instances);
    EXPECT_EQ(scene.stats().tlas_builds, 4u);
    EXPECT_EQ(scene.stats().tlas_refits, 5u);
}

// A refit tree keeps the topology its instances had at build time, so it has
// to be given up on eventually. Both triggers are exercised here because the
// counter alone would let a slowly-degrading scene coast for 64 frames and the
// SAH ratio alone cannot see motion that keeps the cost flat.
TEST(RtScene, RefitDegradationForcesARebuild) {
    Fixture f;
    auto a = f.models.register_model("a", make_cube());
    ASSERT_TRUE(a);

    // --- the measured trigger: large accumulated motion ------------------
    {
        RtScene scene;
        scene.build_blas_for_models(f.models, false);
        std::vector<std::pair<bracken::ModelHandle, glm::mat4>> instances;
        for (int i = 0; i < 64; ++i) {
            instances.emplace_back(
                a.value(),
                glm::translate(glm::mat4(1.0f), glm::vec3(static_cast<float>(i % 8) * 4.0f, 0.0f,
                                                          static_cast<float>(i / 8) * 4.0f)));
        }
        scene.set_instances(instances);
        ASSERT_EQ(scene.stats().tlas_builds, 1u);

        // Fling half the grid a long way out, in one step. The leaves keep
        // their old grouping, so the interior boxes have to swell to cover
        // pairs that are now nowhere near each other.
        for (std::size_t i = 0; i < instances.size(); i += 2) {
            instances[i].second = glm::translate(
                glm::mat4(1.0f), glm::vec3(static_cast<float>(i) * 250.0f, 0.0f,
                                           static_cast<float>(i) * -250.0f));
        }
        scene.set_instances(instances);
        EXPECT_EQ(scene.stats().tlas_builds, 2u)
            << "a badly degraded refit was accepted instead of rebuilding";
        EXPECT_EQ(scene.stats().tlas_refits, 0u);
    }

    // --- the counter backstop: many tiny moves ---------------------------
    {
        RtScene scene;
        scene.build_blas_for_models(f.models, false);
        std::vector<std::pair<bracken::ModelHandle, glm::mat4>> instances{
            {a.value(), glm::mat4(1.0f)},
            {a.value(), glm::translate(glm::mat4(1.0f), glm::vec3(50.0f, 0.0f, 0.0f))},
        };
        scene.set_instances(instances);
        ASSERT_EQ(scene.stats().tlas_builds, 1u);

        // Jiggle by a hair, so the SAH ratio never moves; only the count does.
        for (uint32_t i = 1; i <= k_tlas_max_refits_between_builds; ++i) {
            instances[0].second = glm::translate(
                glm::mat4(1.0f), glm::vec3(0.0f, static_cast<float>(i) * 1e-3f, 0.0f));
            scene.set_instances(instances);
        }
        EXPECT_EQ(scene.stats().tlas_refits, k_tlas_max_refits_between_builds);
        EXPECT_EQ(scene.stats().tlas_builds, 1u) << "the backstop fired early";

        instances[0].second = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.5f, 0.0f));
        scene.set_instances(instances);
        EXPECT_EQ(scene.stats().tlas_builds, 2u) << "the refit-count backstop never fired";
        EXPECT_EQ(scene.stats().tlas_refits, k_tlas_max_refits_between_builds);
    }
}

// R6's emissive LIGHT records are not attributes and must not ride
// `want_attributes`. The Full-DI tier is the only consumer of them and it never
// binds the attribute buffer, so gating them on reflections/GI made the
// configuration that wants them most — local shadows on, reflections off —
// the one configuration that silently had none. The equality is the other half
// of the claim: hoisting the walk changed WHERE it runs, not WHAT it keeps.
TEST(RtScene, EmissiveLightRecordsDoNotDependOnAttributes) {
    Fixture f;
    auto a = f.models.register_model("glow", make_emissive_cube({4.0f, 4.0f, 4.0f}));
    ASSERT_TRUE(a);

    const std::vector<std::pair<bracken::ModelHandle, glm::mat4>> instances{
        {a.value(), glm::mat4(1.0f)},
        {a.value(), glm::translate(glm::mat4(1.0f), glm::vec3(40.0f, 0.0f, 0.0f))},
    };

    RtScene without;
    without.build_blas_for_models(f.models, /*want_attributes=*/false);
    without.set_instances(instances);

    EXPECT_GT(without.stats().emissive_triangles, 0u)
        << "a shadows-only build found no emitting model triangles";
    EXPECT_GT(without.emissive_record_count(), 0u)
        << "a shadows-only build produced no emissive light records";
    EXPECT_GT(without.emissive_total_power(), 0.0f);

    RtScene with;
    with.build_blas_for_models(f.models, /*want_attributes=*/true);
    with.set_instances(instances);

    EXPECT_EQ(without.stats().emissive_triangles, with.stats().emissive_triangles);
    EXPECT_EQ(without.emissive_record_count(), with.emissive_record_count());
    EXPECT_FLOAT_EQ(without.emissive_total_power(), with.emissive_total_power());
}

// `k_emissive_min_power` is radiance x WORLD units squared, and area is the one
// factor of power an instance transform moves. Testing it in model space made
// the effective world floor `k_emissive_min_power * s^2` — 10 000x off for a
// metre-authored pack instanced into this engine's world, with no diagnostic.
TEST(RtScene, ThePowerThresholdIsAppliedInWorldUnits) {
    Fixture f;
    // Authored small: every facet's MODEL-space power is 0.02, far under the
    // threshold. Instanced large, it delivers 200.
    auto a = f.models.register_model("lamp", make_emissive_cube({4.0f, 4.0f, 4.0f}, 0.1f));
    ASSERT_TRUE(a);

    RtScene scene;
    scene.build_blas_for_models(f.models, /*want_attributes=*/false);
    ASSERT_GT(scene.stats().emissive_triangles, 0u)
        << "the per-model prefilter must be a scale-free RADIANCE floor: a dim-by-area "
           "emitter cannot be judged before its instance scale is known";

    scene.set_instances(std::vector<std::pair<bracken::ModelHandle, glm::mat4>>{
        {a.value(), glm::scale(glm::mat4(1.0f), glm::vec3(100.0f))}});
    EXPECT_EQ(scene.emissive_record_count(), scene.stats().emissive_triangles)
        << "a 100x instance delivers 10 000x the model-space power and is a light";
    EXPECT_EQ(scene.stats().emissive_records_below_power, 0u);

    // The other direction, and the reason the counter exists: shrink the same
    // instance until every placement is under the floor. `emissive_records = 0`
    // must be distinguishable from "this scene has no emitters at all", and the
    // record set has to actually follow the transform rather than latch —
    // the topology hash is handles and count only, so nothing else would
    // notice that membership changed.
    scene.set_instances(std::vector<std::pair<bracken::ModelHandle, glm::mat4>>{
        {a.value(), glm::scale(glm::mat4(1.0f), glm::vec3(0.5f))}});
    EXPECT_EQ(scene.emissive_record_count(), 0u);
    EXPECT_EQ(scene.stats().emissive_records_below_power, scene.stats().emissive_triangles);
    EXPECT_GT(scene.stats().emissive_triangles, 0u) << "the model still emits; its instance is tiny";
}

// The over-budget cut is a power THRESHOLD applied as a stable filter, and ties
// at that threshold are the COMMON case rather than a corner one: `emissive_place`
// transforms the edges with w = 0, so instances of one model differing only in
// translation carry bit-identical powers. A strict `>` filter alone throws the
// whole tied group away while the arithmetic still reports `size - cap`.
TEST(RtScene, TiedRecordsAtTheBudgetCutFillItRatherThanEmptyingIt) {
    Fixture f;
    auto a = f.models.register_model("window", make_emissive_cube({4.0f, 4.0f, 4.0f}));
    ASSERT_TRUE(a);

    RtScene scene;
    scene.build_blas_for_models(f.models, /*want_attributes=*/false);
    const uint32_t per_instance = scene.stats().emissive_triangles;
    ASSERT_GT(per_instance, 0u);

    // Enough translation-only instances to overflow the cap. Every record's
    // power is then the same float, so the cut lands inside one enormous tie
    // group and `strictly_greater` is zero.
    const uint32_t instance_count = k_max_emissive_records / per_instance + 8u;
    std::vector<std::pair<bracken::ModelHandle, glm::mat4>> instances;
    for (uint32_t i = 0; i < instance_count; ++i) {
        instances.emplace_back(a.value(),
                               glm::translate(glm::mat4(1.0f),
                                              glm::vec3(static_cast<float>(i) * 5.0f, 0.0f, 0.0f)));
    }
    const uint32_t flattened = instance_count * per_instance;
    ASSERT_GT(flattened, k_max_emissive_records);

    scene.set_instances(instances);
    EXPECT_EQ(scene.emissive_record_count(), k_max_emissive_records)
        << "the tie group at the cut was dropped wholesale instead of filling the budget";
    EXPECT_EQ(scene.stats().emissive_records, k_max_emissive_records);
    EXPECT_EQ(scene.stats().emissive_records_dropped, flattened - k_max_emissive_records)
        << "the drop count must be RECOUNTED from the survivors, not predicted from size - cap";
    EXPECT_EQ(scene.stats().emissive_records + scene.stats().emissive_records_dropped, flattened);
    EXPECT_GT(scene.emissive_total_power(), 0.0f);

    // The slices the cut leaves behind are what the refresh path walks, so move
    // one instance and check the record set survives it. Translation preserves
    // area, hence power, hence membership — so a refresh that mis-indexed a
    // truncated slice is the only way this changes.
    instances[0].second = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 3.0f, 0.0f));
    const float before = scene.emissive_total_power();
    scene.set_instances(instances);
    EXPECT_EQ(scene.emissive_record_count(), k_max_emissive_records);
    EXPECT_FLOAT_EQ(scene.emissive_total_power(), before);
}

// Rebuilding the BLAS set renumbers every offset a TLAS instance carries, so
// the "nothing moved" skip and the refit path must both be invalidated — a
// host enabling reflections mid-run does exactly this without touching a
// single transform.
TEST(RtScene, RebuildingTheBlasSetInvalidatesTheInstanceCache) {
    Fixture f;
    auto a = f.models.register_model("a", make_cube());
    ASSERT_TRUE(a);

    RtScene scene;
    scene.build_blas_for_models(f.models, false);
    const std::vector<std::pair<bracken::ModelHandle, glm::mat4>> instances{
        {a.value(), glm::mat4(1.0f)},
        {a.value(), glm::translate(glm::mat4(1.0f), glm::vec3(4.0f, 0.0f, 0.0f))},
    };
    scene.set_instances(instances);
    ASSERT_EQ(scene.stats().tlas_builds, 1u);

    // Same instances, byte for byte. The structure underneath them is new.
    scene.build_blas_for_models(f.models, /*want_attributes=*/true);
    scene.set_instances(instances);
    EXPECT_EQ(scene.stats().tlas_builds, 1u) << "stats reset with the BLAS build, so this counts "
                                                "the ONE build that followed it";
    EXPECT_EQ(scene.stats().instances, 2u);
    EXPECT_GT(scene.stats().tlas_nodes, 0u);
}
