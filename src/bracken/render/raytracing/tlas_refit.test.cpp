// The correctness gate for TLAS refit.
//
// `refit_tlas` reuses a tree that was built for where the instances USED to be.
// That is only sound because a refitted node bound is a conservative superset
// of the tight one: a ray may walk through more boxes than it strictly had to,
// but it cannot miss anything and it cannot find anything extra. The whole
// argument therefore rests on one claim — that a refit and a from-scratch
// rebuild of the same instance list answer every ray identically — and this
// file is where that claim is checked rather than asserted.
//
// The traversal below is deliberately two-level in shape but not in depth:
// leaves resolve to an instance, the ray is transformed into that instance's
// model space by the stored `world_to_model`, and the model-space box stands in
// for a BLAS descent. That is exactly the part of the two-level structure a
// TLAS refit can break; substituting a real BLAS would add thousands of
// triangle tests that the refit provably cannot touch (BLASes are in model
// space and never move) and cover nothing more.
//
// TIES. Two instances whose boxes are entered at exactly the same distance are
// a legitimate disagreement — the two trees visit them in different orders and
// both answers are "the closest hit". Only EXACT float equality is excused, it
// is counted, and the count is asserted to stay tiny.

#include <gtest/gtest.h>

#include "bracken/render/raytracing/bvh.hpp"
// For k_tlas_refit_sah_rebuild_ratio: the degradation test below is what
// licenses that threshold, so it reads the real one rather than a copy.
#include "bracken/render/raytracing/rt_scene.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

using namespace bracken::render::rt;

namespace {

/// The model-space box every test instance carries. Unit-ish and centered, so
/// a rotation actually changes the world AABB (a cube centered on its origin
/// still does — the corners move even when the extents do not).
constexpr glm::vec3 k_model_min{-0.5f, -0.5f, -0.5f};
constexpr glm::vec3 k_model_max{0.5f, 0.5f, 0.5f};

/** Build one instance record: world AABB from all eight transformed corners,
 *  plus the inverse transform traversal needs. `blas_node_offset` carries a
 *  stable identity through the builder's permutation, which is how the two
 *  trees' answers are compared at all. */
TlasInstance make_instance(const glm::mat4& transform, uint32_t id) {
    TlasInstance inst;
    glm::vec3 lo(std::numeric_limits<float>::max());
    glm::vec3 hi(std::numeric_limits<float>::lowest());
    for (int c = 0; c < 8; ++c) {
        const glm::vec3 corner{(c & 1) ? k_model_max.x : k_model_min.x,
                               (c & 2) ? k_model_max.y : k_model_min.y,
                               (c & 4) ? k_model_max.z : k_model_min.z};
        const glm::vec3 w = glm::vec3(transform * glm::vec4(corner, 1.0f));
        lo = glm::min(lo, w);
        hi = glm::max(hi, w);
    }
    inst.world_to_model = glm::inverse(transform);
    inst.bounds_min = lo;
    inst.bounds_max = hi;
    inst.blas_node_offset = id;
    inst.blas_triangle_offset = id;
    return inst;
}

struct Ray {
    glm::vec3 origin{0.0f};
    glm::vec3 dir{0.0f, 0.0f, 1.0f};
    float t_max = 1e30f;
};

/// Slab test, matching the arithmetic the shader walks use.
bool slab_test(const glm::vec3& lo, const glm::vec3& hi, const glm::vec3& origin,
               const glm::vec3& inv_dir, float t_max, float& t_near) {
    const glm::vec3 t0 = (lo - origin) * inv_dir;
    const glm::vec3 t1 = (hi - origin) * inv_dir;
    const glm::vec3 tsmall = glm::min(t0, t1);
    const glm::vec3 tbig = glm::max(t0, t1);
    t_near = std::max(std::max(tsmall.x, tsmall.y), std::max(tsmall.z, 0.0f));
    const float t_far = std::min(std::min(tbig.x, tbig.y), std::min(tbig.z, t_max));
    return t_near <= t_far;
}

/// Stand-in for a BLAS descent: the ray in model space against the model box.
/// Returns the entry distance, which is a world distance too because the test
/// transforms are rigid + uniform-scale-free (see `random_transform`).
bool instance_hit(const TlasInstance& inst, const Ray& ray, float t_max, float& out_t) {
    const glm::vec3 o = glm::vec3(inst.world_to_model * glm::vec4(ray.origin, 1.0f));
    const glm::vec3 d = glm::vec3(inst.world_to_model * glm::vec4(ray.dir, 0.0f));
    const glm::vec3 inv = 1.0f / d;
    return slab_test(k_model_min, k_model_max, o, inv, t_max, out_t) && out_t < t_max;
}

struct Closest {
    float t = 0.0f;
    uint32_t id = 0;
    bool valid = false;
};

Closest closest_hit(const Tlas& tlas, const Ray& ray) {
    Closest best;
    best.t = ray.t_max;
    if (tlas.nodes.empty()) {
        return best;
    }
    const glm::vec3 inv_dir = 1.0f / ray.dir;
    std::vector<uint32_t> stack{0};
    while (!stack.empty()) {
        const BvhNode& node = tlas.nodes[stack.back()];
        stack.pop_back();
        float t_near = 0.0f;
        if (!slab_test(node.bounds_min, node.bounds_max, ray.origin, inv_dir, best.t, t_near)) {
            continue;
        }
        if (node.count == 0) {
            stack.push_back(node.left_first);
            stack.push_back(node.left_first + 1);
            continue;
        }
        for (uint32_t i = 0; i < node.count; ++i) {
            const TlasInstance& inst = tlas.instances[node.left_first + i];
            float t = 0.0f;
            if (instance_hit(inst, ray, best.t, t) && t < best.t) {
                best.t = t;
                best.id = inst.blas_node_offset;
                best.valid = true;
            }
        }
    }
    return best;
}

bool any_hit(const Tlas& tlas, const Ray& ray) {
    if (tlas.nodes.empty()) {
        return false;
    }
    const glm::vec3 inv_dir = 1.0f / ray.dir;
    std::vector<uint32_t> stack{0};
    while (!stack.empty()) {
        const BvhNode& node = tlas.nodes[stack.back()];
        stack.pop_back();
        float t_near = 0.0f;
        if (!slab_test(node.bounds_min, node.bounds_max, ray.origin, inv_dir, ray.t_max, t_near)) {
            continue;
        }
        if (node.count == 0) {
            stack.push_back(node.left_first);
            stack.push_back(node.left_first + 1);
            continue;
        }
        for (uint32_t i = 0; i < node.count; ++i) {
            float t = 0.0f;
            if (instance_hit(tlas.instances[node.left_first + i], ray, ray.t_max, t)) {
                return true;
            }
        }
    }
    return false;
}

/// Rotation + translation only. No scale, so a model-space entry distance is a
/// world-space one and the two trees' `t` values are directly comparable.
glm::mat4 random_transform(std::mt19937& rng, float spread) {
    std::uniform_real_distribution<float> pos(-spread, spread);
    std::uniform_real_distribution<float> angle(0.0f, 6.2831853f);
    glm::mat4 m = glm::translate(glm::mat4(1.0f), glm::vec3(pos(rng), pos(rng) * 0.25f, pos(rng)));
    m = glm::rotate(m, angle(rng), glm::normalize(glm::vec3(0.3f, 1.0f, 0.7f)));
    return m;
}

std::vector<TlasInstance> random_instances(std::mt19937& rng, std::size_t count, float spread) {
    std::vector<TlasInstance> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(make_instance(random_transform(rng, spread), static_cast<uint32_t>(i)));
    }
    return out;
}

/// Apply a fresh instance list to an existing tree's slots, exactly the way
/// `RtScene::refit_tlas_in_place` does: `source_index` maps slot -> input.
void apply_and_refit(Tlas& tlas, const std::vector<TlasInstance>& updated) {
    for (std::size_t slot = 0; slot < tlas.instances.size(); ++slot) {
        tlas.instances[slot] = updated[tlas.source_index[slot]];
    }
    refit_tlas(tlas);
}

/**
 * Rays AIMED at the instances, not fired at random into a mostly empty volume.
 *
 * Unit-sized boxes scattered over a hundred units means a uniformly random ray
 * hits roughly nothing (measured: 13 hits in 4000), and a test that compares
 * two ways of answering "no" is not comparing much. Each ray here starts at a
 * random point and points at a random instance's centre with a few units of
 * jitter, so hits and misses both stay plentiful and the closest-hit
 * comparison has real work to do.
 */
std::vector<Ray> seeded_rays(std::mt19937& rng, std::size_t count,
                             const std::vector<TlasInstance>& targets, float origin_spread,
                             float jitter) {
    std::uniform_real_distribution<float> pos(-origin_spread, origin_spread);
    std::uniform_real_distribution<float> wobble(-jitter, jitter);
    std::uniform_int_distribution<std::size_t> pick(0, targets.size() - 1);
    std::vector<Ray> rays;
    rays.reserve(count);
    while (rays.size() < count) {
        const TlasInstance& target = targets[pick(rng)];
        const glm::vec3 aim = (target.bounds_min + target.bounds_max) * 0.5f +
                              glm::vec3(wobble(rng), wobble(rng), wobble(rng));
        const glm::vec3 origin(pos(rng), pos(rng) * 0.3f, pos(rng));
        const glm::vec3 d = aim - origin;
        // Reject near-degenerate directions: a component at exactly zero makes
        // the slab test's inverse an infinity, which is well defined but not
        // what this test is about.
        if (glm::length(d) < 1.0f || std::abs(d.x) < 1e-3f || std::abs(d.y) < 1e-3f ||
            std::abs(d.z) < 1e-3f) {
            continue;
        }
        Ray r;
        r.origin = origin;
        r.dir = glm::normalize(d);
        r.t_max = 1e30f;
        rays.push_back(r);
    }
    return rays;
}

} // namespace

// The builder's bottom-up union and the refit's bottom-up union have to be the
// same computation, or every comparison below is measuring two bugs against
// each other. Refitting a tree with the bounds it was built from must be a
// no-op, bit for bit.
TEST(TlasRefit, RefittingUnchangedInstancesReproducesTheBuiltBounds) {
    std::mt19937 rng(20260802u);
    const std::vector<TlasInstance> instances = random_instances(rng, 300, 60.0f);

    const Tlas built = build_tlas(instances);
    ASSERT_FALSE(built.nodes.empty());
    ASSERT_EQ(built.source_index.size(), built.instances.size());

    Tlas refitted = built;
    refit_tlas(refitted);

    ASSERT_EQ(refitted.nodes.size(), built.nodes.size());
    for (std::size_t i = 0; i < built.nodes.size(); ++i) {
        EXPECT_EQ(refitted.nodes[i].bounds_min, built.nodes[i].bounds_min) << "node " << i;
        EXPECT_EQ(refitted.nodes[i].bounds_max, built.nodes[i].bounds_max) << "node " << i;
        EXPECT_EQ(refitted.nodes[i].left_first, built.nodes[i].left_first) << "node " << i;
        EXPECT_EQ(refitted.nodes[i].count, built.nodes[i].count) << "node " << i;
    }
}

// The claim the whole optimization rests on: refit and rebuild answer every
// ray identically. Bounds may differ — hits may not.
TEST(TlasRefit, RefitAnswersEveryRayLikeARebuild) {
    std::mt19937 rng(99117u);
    const std::size_t count = 400;
    std::vector<TlasInstance> instances = random_instances(rng, count, 80.0f);

    Tlas refitted = build_tlas(instances);
    ASSERT_FALSE(refitted.nodes.empty());

    // Move a subset by small transforms — the animated-character case, scaled
    // up so the comparison has something to bite on.
    std::uniform_real_distribution<float> nudge(-3.0f, 3.0f);
    for (std::size_t i = 0; i < count; i += 3) {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), glm::vec3(nudge(rng), nudge(rng), nudge(rng)));
        t = glm::rotate(t, 0.4f, glm::vec3(0.0f, 1.0f, 0.0f));
        instances[i] = make_instance(t * glm::inverse(instances[i].world_to_model),
                                     static_cast<uint32_t>(i));
    }

    apply_and_refit(refitted, instances);
    const Tlas rebuilt = build_tlas(instances);

    const std::vector<Ray> rays = seeded_rays(rng, 4000, instances, 120.0f, 1.0f);
    std::size_t ties = 0;
    std::size_t hits = 0;
    for (const Ray& ray : rays) {
        EXPECT_EQ(any_hit(refitted, ray), any_hit(rebuilt, ray))
            << "any-hit disagreed for a ray from " << ray.origin.x << "," << ray.origin.y << ","
            << ray.origin.z;

        const Closest a = closest_hit(refitted, ray);
        const Closest b = closest_hit(rebuilt, ray);
        ASSERT_EQ(a.valid, b.valid);
        if (!a.valid) {
            continue;
        }
        ++hits;
        EXPECT_FLOAT_EQ(a.t, b.t);
        if (a.id != b.id) {
            // Only an exact distance tie excuses a different instance.
            EXPECT_EQ(a.t, b.t) << "refit and rebuild found different instances at different t";
            ++ties;
        }
    }

    EXPECT_GT(hits, 500u) << "the ray set barely hit anything; it is not testing much";
    EXPECT_LT(ties, hits / 200 + 2) << "suspiciously many exact-distance ties: " << ties;
}

// Same claim under motion large enough that the refit tree is genuinely a bad
// tree — the answers still have to be right, only slower to reach.
TEST(TlasRefit, RefitStaysCorrectUnderLargeMotion) {
    std::mt19937 rng(4242u);
    const std::size_t count = 256;
    std::vector<TlasInstance> instances = random_instances(rng, count, 40.0f);

    Tlas refitted = build_tlas(instances);
    const float built_cost = sah_cost(refitted.nodes);

    // Scatter every other instance across a volume 10x wider. Leaves keep
    // their old pairings, so interior boxes swell enormously.
    for (std::size_t i = 0; i < count; i += 2) {
        instances[i] = make_instance(random_transform(rng, 400.0f), static_cast<uint32_t>(i));
    }
    apply_and_refit(refitted, instances);
    const Tlas rebuilt = build_tlas(instances);

    // The degradation this test exists to provoke, stated numerically so the
    // trigger in RtScene has a documented reason to exist.
    const float refit_cost = sah_cost(refitted.nodes);
    const float rebuilt_cost = sah_cost(rebuilt.nodes);
    EXPECT_GT(refit_cost, rebuilt_cost * k_tlas_refit_sah_rebuild_ratio)
        << "large motion did not degrade the refit tree (built " << built_cost << ", refit "
        << refit_cost << ", rebuilt " << rebuilt_cost << ")";

    // Root bounds are the union of every instance either way, so these are the
    // one pair that must agree exactly.
    EXPECT_EQ(refitted.nodes[0].bounds_min, rebuilt.nodes[0].bounds_min);
    EXPECT_EQ(refitted.nodes[0].bounds_max, rebuilt.nodes[0].bounds_max);

    const std::vector<Ray> rays = seeded_rays(rng, 3000, instances, 500.0f, 1.5f);
    for (const Ray& ray : rays) {
        EXPECT_EQ(any_hit(refitted, ray), any_hit(rebuilt, ray));
        const Closest a = closest_hit(refitted, ray);
        const Closest b = closest_hit(rebuilt, ray);
        ASSERT_EQ(a.valid, b.valid);
        if (a.valid) {
            EXPECT_FLOAT_EQ(a.t, b.t);
        }
    }
}

// A single-instance TLAS is the degenerate root case the GPU conversion has to
// special-case; refit must not trip over it either.
TEST(TlasRefit, SingleInstanceTreeRefits) {
    std::vector<TlasInstance> instances{make_instance(glm::mat4(1.0f), 0u)};
    Tlas tlas = build_tlas(instances);
    ASSERT_EQ(tlas.nodes.size(), 1u);

    instances[0] = make_instance(glm::translate(glm::mat4(1.0f), glm::vec3(100.0f, 0.0f, 0.0f)), 0u);
    apply_and_refit(tlas, instances);

    EXPECT_FLOAT_EQ(tlas.nodes[0].bounds_min.x, 99.5f);
    EXPECT_FLOAT_EQ(tlas.nodes[0].bounds_max.x, 100.5f);
}

// An empty tree has no nodes and no instances; refitting it is a no-op rather
// than an out-of-range walk.
TEST(TlasRefit, EmptyTreeRefitsToNothing) {
    Tlas tlas;
    refit_tlas(tlas);
    EXPECT_TRUE(tlas.nodes.empty());
}
