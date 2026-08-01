#include <gtest/gtest.h>

#include "bracken/render/raytracing/bvh.hpp"

#include "bracken/core/random.hpp"

#include <cmath>

using namespace bracken::render::rt;

namespace {

/** A grid-tessellated plane in XZ: predictable triangle count and bounds. */
void make_grid(int n, std::vector<glm::vec3>& pos, std::vector<uint32_t>& idx) {
    for (int z = 0; z <= n; ++z) {
        for (int x = 0; x <= n; ++x) {
            pos.emplace_back(static_cast<float>(x), 0.0f, static_cast<float>(z));
        }
    }
    const int stride = n + 1;
    for (int z = 0; z < n; ++z) {
        for (int x = 0; x < n; ++x) {
            const auto a = static_cast<uint32_t>(z * stride + x);
            const auto b = a + 1;
            const auto c = a + static_cast<uint32_t>(stride);
            const auto d = c + 1;
            idx.insert(idx.end(), {a, c, b, b, c, d});
        }
    }
}

/** Scattered independent triangles — the case a median split handles badly and
 *  SAH handles well. */
void make_scatter(int count, std::vector<glm::vec3>& pos, std::vector<uint32_t>& idx) {
    bracken::core::Rng rng(1234);
    for (int i = 0; i < count; ++i) {
        const glm::vec3 base{rng.next_range(-500.0f, 500.0f), rng.next_range(-50.0f, 50.0f),
                             rng.next_range(-500.0f, 500.0f)};
        const auto first = static_cast<uint32_t>(pos.size());
        pos.push_back(base);
        pos.push_back(base + glm::vec3(rng.next_range(1.0f, 8.0f), 0.0f, 0.0f));
        pos.push_back(base + glm::vec3(0.0f, 0.0f, rng.next_range(1.0f, 8.0f)));
        idx.insert(idx.end(), {first, first + 1, first + 2});
    }
}

/**
 * CPU reference traversal — the same algorithm the compute shader runs.
 *
 * It exists to validate the tree independently of any GPU, and it is what the
 * HLSL traversal is checked against: a GPU result that disagrees with this is
 * a shader bug, and without a reference there is no way to tell that from a
 * BVH bug.
 */
bool occluded(const Blas& blas, const glm::vec3& origin, const glm::vec3& dir, float t_max) {
    if (blas.empty()) {
        return false;
    }
    const glm::vec3 inv_dir = 1.0f / dir;

    uint32_t stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const BvhNode& node = blas.nodes[stack[--sp]];

        // Slab test.
        const glm::vec3 t0 = (node.bounds_min - origin) * inv_dir;
        const glm::vec3 t1 = (node.bounds_max - origin) * inv_dir;
        const glm::vec3 tsmall = glm::min(t0, t1);
        const glm::vec3 tbig = glm::max(t0, t1);
        const float tnear = std::max(std::max(tsmall.x, tsmall.y), std::max(tsmall.z, 0.0f));
        const float tfar = std::min(std::min(tbig.x, tbig.y), std::min(tbig.z, t_max));
        if (tnear > tfar) {
            continue;
        }

        if (node.count == 0) {
            stack[sp++] = node.left_first;
            stack[sp++] = node.left_first + 1;
            continue;
        }

        for (uint32_t i = 0; i < node.count; ++i) {
            const BvhTriangle& tri = blas.triangles[node.left_first + i];
            // Möller–Trumbore.
            const glm::vec3 pv = glm::cross(dir, tri.e2);
            const float det = glm::dot(tri.e1, pv);
            if (std::fabs(det) < 1e-8f) {
                continue;
            }
            const float inv_det = 1.0f / det;
            const glm::vec3 tv = origin - tri.v0;
            const float u = glm::dot(tv, pv) * inv_det;
            if (u < 0.0f || u > 1.0f) {
                continue;
            }
            const glm::vec3 qv = glm::cross(tv, tri.e1);
            const float v = glm::dot(dir, qv) * inv_det;
            if (v < 0.0f || u + v > 1.0f) {
                continue;
            }
            const float t = glm::dot(tri.e2, qv) * inv_det;
            if (t > 1e-4f && t < t_max) {
                return true; // ANY hit: shadow rays stop here
            }
        }
    }
    return false;
}

} // namespace

TEST(Bvh, EmptyInputProducesEmptyTree) {
    BvhBuildStats stats;
    const Blas blas = build_blas({}, {}, {}, &stats);
    EXPECT_TRUE(blas.empty());
    EXPECT_EQ(stats.triangle_count, 0u);
}

TEST(Bvh, BuildsOverAGrid) {
    std::vector<glm::vec3> pos;
    std::vector<uint32_t> idx;
    make_grid(16, pos, idx);

    BvhBuildStats stats;
    const Blas blas = build_blas(pos, idx, {}, &stats);

    EXPECT_EQ(stats.triangle_count, idx.size() / 3);
    EXPECT_EQ(blas.triangles.size(), idx.size() / 3);
    EXPECT_GT(stats.node_count, 1u);
    EXPECT_GT(stats.leaf_count, 0u);
}

TEST(Bvh, RootBoundsEncloseTheMesh) {
    std::vector<glm::vec3> pos;
    std::vector<uint32_t> idx;
    make_grid(8, pos, idx);
    const Blas blas = build_blas(pos, idx);

    for (const glm::vec3& p : pos) {
        EXPECT_GE(p.x, blas.bounds_min.x - 1e-3f);
        EXPECT_LE(p.x, blas.bounds_max.x + 1e-3f);
        EXPECT_GE(p.z, blas.bounds_min.z - 1e-3f);
        EXPECT_LE(p.z, blas.bounds_max.z + 1e-3f);
    }
}

// The invariant traversal depends on: a child's bounds must never exceed its
// parent's, or a ray culled at the parent would have hit something inside.
TEST(Bvh, ChildBoundsAreContainedByTheirParent) {
    std::vector<glm::vec3> pos;
    std::vector<uint32_t> idx;
    make_scatter(400, pos, idx);
    const Blas blas = build_blas(pos, idx);

    for (const BvhNode& n : blas.nodes) {
        if (n.count != 0) {
            continue;
        }
        ASSERT_LT(n.left_first + 1, blas.nodes.size());
        for (uint32_t c : {n.left_first, n.left_first + 1}) {
            const BvhNode& child = blas.nodes[c];
            EXPECT_GE(child.bounds_min.x, n.bounds_min.x - 1e-3f);
            EXPECT_GE(child.bounds_min.y, n.bounds_min.y - 1e-3f);
            EXPECT_GE(child.bounds_min.z, n.bounds_min.z - 1e-3f);
            EXPECT_LE(child.bounds_max.x, n.bounds_max.x + 1e-3f);
            EXPECT_LE(child.bounds_max.y, n.bounds_max.y + 1e-3f);
            EXPECT_LE(child.bounds_max.z, n.bounds_max.z + 1e-3f);
        }
    }
}

// Every triangle must appear in exactly one leaf. A triangle in none is a hole
// in the shadow; a triangle in two is wasted work.
TEST(Bvh, EveryTriangleIsInExactlyOneLeaf) {
    std::vector<glm::vec3> pos;
    std::vector<uint32_t> idx;
    make_scatter(300, pos, idx);
    const Blas blas = build_blas(pos, idx);

    std::vector<int> seen(blas.triangles.size(), 0);
    for (const BvhNode& n : blas.nodes) {
        if (n.count == 0) {
            continue;
        }
        for (uint32_t i = 0; i < n.count; ++i) {
            ASSERT_LT(n.left_first + i, blas.triangles.size());
            seen[n.left_first + i]++;
        }
    }
    for (std::size_t i = 0; i < seen.size(); ++i) {
        EXPECT_EQ(seen[i], 1) << "triangle " << i << " appears in " << seen[i] << " leaves";
    }
}

TEST(Bvh, LeavesRespectTheSizeLimit) {
    std::vector<glm::vec3> pos;
    std::vector<uint32_t> idx;
    make_scatter(500, pos, idx);
    BvhBuildOptions opt;
    opt.max_leaf_triangles = 4;
    const Blas blas = build_blas(pos, idx, opt);

    for (const BvhNode& n : blas.nodes) {
        if (n.count == 0) {
            continue;
        }
        // SAH may stop early when a split does not pay, so a leaf can exceed
        // the target — but not wildly, or traversal degenerates to brute force.
        EXPECT_LE(n.count, 64u);
    }
    EXPECT_LT(blas.nodes[0].count, 1u) << "the root of a 500-triangle mesh must not be a leaf";
}

// The thing the whole structure exists for: a ray must find geometry it hits
// and miss geometry it does not.
TEST(Bvh, TraversalAgreesWithGeometry) {
    std::vector<glm::vec3> pos;
    std::vector<uint32_t> idx;
    make_grid(16, pos, idx); // a plane spanning x,z in [0,16] at y = 0
    const Blas blas = build_blas(pos, idx);

    // Straight down through the middle: must hit.
    EXPECT_TRUE(occluded(blas, {8.0f, 10.0f, 8.0f}, {0.0f, -1.0f, 0.0f}, 100.0f));
    // Straight up, away from the plane: must miss.
    EXPECT_FALSE(occluded(blas, {8.0f, 10.0f, 8.0f}, {0.0f, 1.0f, 0.0f}, 100.0f));
    // Down, but outside the plane's extent: must miss.
    EXPECT_FALSE(occluded(blas, {100.0f, 10.0f, 100.0f}, {0.0f, -1.0f, 0.0f}, 100.0f));
    // Down, but the ray ends before reaching the plane: must miss.
    EXPECT_FALSE(occluded(blas, {8.0f, 10.0f, 8.0f}, {0.0f, -1.0f, 0.0f}, 5.0f));
}

// Brute force over every triangle is slow but obviously correct; the BVH must
// agree with it on every ray, or it is culling something it should not.
TEST(Bvh, TraversalMatchesBruteForceOnRandomRays) {
    std::vector<glm::vec3> pos;
    std::vector<uint32_t> idx;
    make_scatter(200, pos, idx);
    const Blas blas = build_blas(pos, idx);

    auto brute_force = [&](const glm::vec3& o, const glm::vec3& d, float t_max) {
        for (const BvhTriangle& tri : blas.triangles) {
            const glm::vec3 pv = glm::cross(d, tri.e2);
            const float det = glm::dot(tri.e1, pv);
            if (std::fabs(det) < 1e-8f) {
                continue;
            }
            const float inv_det = 1.0f / det;
            const glm::vec3 tv = o - tri.v0;
            const float u = glm::dot(tv, pv) * inv_det;
            if (u < 0.0f || u > 1.0f) {
                continue;
            }
            const glm::vec3 qv = glm::cross(tv, tri.e1);
            const float v = glm::dot(d, qv) * inv_det;
            if (v < 0.0f || u + v > 1.0f) {
                continue;
            }
            const float t = glm::dot(tri.e2, qv) * inv_det;
            if (t > 1e-4f && t < t_max) {
                return true;
            }
        }
        return false;
    };

    bracken::core::Rng rng(99);
    int mismatches = 0;
    for (int i = 0; i < 2000; ++i) {
        const glm::vec3 o{rng.next_range(-600.0f, 600.0f), rng.next_range(20.0f, 200.0f),
                          rng.next_range(-600.0f, 600.0f)};
        glm::vec3 d{rng.next_range(-1.0f, 1.0f), rng.next_range(-1.0f, -0.1f),
                    rng.next_range(-1.0f, 1.0f)};
        d = glm::normalize(d);
        if (occluded(blas, o, d, 1000.0f) != brute_force(o, d, 1000.0f)) {
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "the BVH culled geometry a brute-force test found";
}

// SAH exists to produce a cheaper tree than a naive split. If it does not beat
// a 2-bin (effectively median) build, the heuristic is not doing its job.
TEST(Bvh, SahBeatsANaiveSplitOnScatteredGeometry) {
    std::vector<glm::vec3> pos;
    std::vector<uint32_t> idx;
    make_scatter(2000, pos, idx);

    BvhBuildStats sah_stats;
    BvhBuildOptions sah_opt;
    sah_opt.sah_bins = 16;
    (void)build_blas(pos, idx, sah_opt, &sah_stats);

    BvhBuildStats naive_stats;
    BvhBuildOptions naive_opt;
    naive_opt.sah_bins = 2;
    (void)build_blas(pos, idx, naive_opt, &naive_stats);

    EXPECT_LT(sah_stats.sah_cost, naive_stats.sah_cost)
        << "16-bin SAH cost " << sah_stats.sah_cost << " vs 2-bin " << naive_stats.sah_cost;
}

TEST(Bvh, DegenerateGeometryDoesNotHangOrOverflow) {
    // Every triangle at the same place: no split can separate the centroids.
    std::vector<glm::vec3> pos;
    std::vector<uint32_t> idx;
    for (int i = 0; i < 200; ++i) {
        const auto f = static_cast<uint32_t>(pos.size());
        pos.emplace_back(0.0f, 0.0f, 0.0f);
        pos.emplace_back(1.0f, 0.0f, 0.0f);
        pos.emplace_back(0.0f, 0.0f, 1.0f);
        idx.insert(idx.end(), {f, f + 1, f + 2});
    }
    BvhBuildStats stats;
    const Blas blas = build_blas(pos, idx, {}, &stats);
    EXPECT_FALSE(blas.empty());
    EXPECT_LE(stats.max_depth, 64u);
}

TEST(Bvh, MalformedIndicesAreDropped) {
    std::vector<glm::vec3> pos{{0, 0, 0}, {1, 0, 0}, {0, 0, 1}};
    std::vector<uint32_t> idx{0, 1, 2, 0, 1, 99}; // second triangle is bogus
    BvhBuildStats stats;
    const Blas blas = build_blas(pos, idx, {}, &stats);
    EXPECT_EQ(stats.triangle_count, 1u);
}

// A rebuilt tree must be identical, so a traced frame is reproducible.
TEST(Bvh, BuildIsDeterministic) {
    std::vector<glm::vec3> pos;
    std::vector<uint32_t> idx;
    make_scatter(300, pos, idx);
    const Blas a = build_blas(pos, idx);
    const Blas b = build_blas(pos, idx);

    ASSERT_EQ(a.nodes.size(), b.nodes.size());
    for (std::size_t i = 0; i < a.nodes.size(); ++i) {
        EXPECT_EQ(a.nodes[i].left_first, b.nodes[i].left_first);
        EXPECT_EQ(a.nodes[i].count, b.nodes[i].count);
        EXPECT_EQ(a.nodes[i].bounds_min, b.nodes[i].bounds_min);
    }
}

TEST(Tlas, BuildsOverInstances) {
    std::vector<TlasInstance> instances;
    for (int i = 0; i < 64; ++i) {
        TlasInstance inst;
        const auto f = static_cast<float>(i) * 100.0f;
        inst.bounds_min = {f, 0.0f, 0.0f};
        inst.bounds_max = {f + 50.0f, 50.0f, 50.0f};
        inst.blas_node_offset = static_cast<uint32_t>(i);
        instances.push_back(inst);
    }

    BvhBuildStats stats;
    const Tlas tlas = build_tlas(instances, {}, &stats);
    EXPECT_EQ(tlas.instances.size(), 64u);
    EXPECT_GT(stats.node_count, 1u);
    // One instance per leaf: an instance test is a whole BLAS descent.
    for (const BvhNode& n : tlas.nodes) {
        if (n.count != 0) {
            EXPECT_LE(n.count, 1u);
        }
    }
}

TEST(Tlas, EmptyInstanceListIsHandled) {
    const Tlas tlas = build_tlas({});
    EXPECT_TRUE(tlas.empty());
}

// The reason instances are reordered during the build: leaves index the
// reordered array, so the mapping has to survive.
TEST(Tlas, InstancePayloadSurvivesReordering) {
    std::vector<TlasInstance> instances;
    for (int i = 0; i < 32; ++i) {
        TlasInstance inst;
        const auto f = static_cast<float>((i * 37) % 32) * 100.0f; // deliberately shuffled
        inst.bounds_min = {f, 0.0f, 0.0f};
        inst.bounds_max = {f + 10.0f, 10.0f, 10.0f};
        inst.blas_node_offset = static_cast<uint32_t>(i);
        instances.push_back(inst);
    }
    const Tlas tlas = build_tlas(instances);

    // Every original offset must still be present exactly once.
    std::vector<int> seen(32, 0);
    for (const TlasInstance& inst : tlas.instances) {
        ASSERT_LT(inst.blas_node_offset, 32u);
        seen[inst.blas_node_offset]++;
    }
    for (int s : seen) {
        EXPECT_EQ(s, 1);
    }
}
