// The correctness gate for the 64-byte upload layout.
//
// The conversion in bvh_gpu.cpp and the rewritten walks in
// shaders/rt_trace.hlsli have to agree with the 32-byte tree they replace for
// every ray, and there is no way to assert that on the GPU: a traced frame is
// a screenshot statistic, which moves for a dozen reasons that are not this
// one. So the walk is mirrored here in C++ — the SAME control flow, over the
// SAME converted data the production path uploads (`convert_to_gpu_nodes`, not
// a copy of it) — and checked against a reference traversal of the built tree
// over thousands of seeded rays.
//
// What the mirror is allowed to differ in: nothing structural. It is written
// against BvhNodeGpu with the same near-first ordering, the same inline leaf
// handling and the same stack discipline, so a mistake in the encoding shows
// up here rather than as a dark pixel in a village screenshot.
//
// TIES. Two triangles at exactly the same distance are a legitimate
// disagreement — the two walks visit them in different orders and both answers
// are "the closest hit". Only EXACT float equality is skipped, it is counted,
// and the count is asserted to stay tiny; anything else must match to 1e-4.

#include <gtest/gtest.h>

#include "bracken/render/raytracing/bvh_gpu.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace bracken::render::rt;

namespace {

// --------------------------------------------------------------------------
// Reference traversal of the 32-byte BUILD tree. Deliberately naive: an
// unordered stack walk, no cleverness, so it cannot share a bug with the thing
// it is checking.
// --------------------------------------------------------------------------

struct Ray {
    glm::vec3 origin{0.0f};
    glm::vec3 dir{0.0f, 0.0f, 1.0f};
    float t_max = 1e30f;
};

/// The exact arithmetic of `slabTest` in rt_trace.hlsli.
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

constexpr float k_ray_epsilon = 0.02f;

struct TriHit {
    bool hit = false;
    float t = 0.0f;
};

/// Möller-Trumbore, matching the shader's `triangleHit` / inline copy.
TriHit intersect(const BvhTriangle& tri, const glm::vec3& origin, const glm::vec3& dir,
                 float t_max) {
    TriHit out;
    const glm::vec3 pv = glm::cross(dir, tri.e2);
    const float det = glm::dot(tri.e1, pv);
    if (std::abs(det) < 1e-8f) {
        return out;
    }
    const float inv_det = 1.0f / det;
    const glm::vec3 tv = origin - tri.v0;
    const float u = glm::dot(tv, pv) * inv_det;
    if (u < 0.0f || u > 1.0f) {
        return out;
    }
    const glm::vec3 qv = glm::cross(tv, tri.e1);
    const float v = glm::dot(dir, qv) * inv_det;
    if (v < 0.0f || u + v > 1.0f) {
        return out;
    }
    const float t = glm::dot(tri.e2, qv) * inv_det;
    out.hit = t > k_ray_epsilon && t < t_max;
    out.t = t;
    return out;
}

bool reference_occluded(const Blas& blas, const Ray& ray) {
    const glm::vec3 inv_dir = 1.0f / ray.dir;
    std::vector<uint32_t> stack{0};
    while (!stack.empty()) {
        const BvhNode& node = blas.nodes[stack.back()];
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
            if (intersect(blas.triangles[node.left_first + i], ray.origin, ray.dir, ray.t_max)
                    .hit) {
                return true;
            }
        }
    }
    return false;
}

struct Closest {
    float t = 0.0f;
    uint32_t tri_index = 0;
    bool valid = false;
};

Closest reference_closest(const Blas& blas, const Ray& ray) {
    Closest best;
    best.t = ray.t_max;
    const glm::vec3 inv_dir = 1.0f / ray.dir;
    std::vector<uint32_t> stack{0};
    while (!stack.empty()) {
        const BvhNode& node = blas.nodes[stack.back()];
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
            const uint32_t index = node.left_first + i;
            const TriHit h = intersect(blas.triangles[index], ray.origin, ray.dir, best.t);
            if (h.hit && h.t < best.t) {
                best.t = h.t;
                best.tri_index = index;
                best.valid = true;
            }
        }
    }
    return best;
}

// --------------------------------------------------------------------------
// CPU mirror of the rewritten HLSL walks over the CONVERTED nodes. Kept
// structurally identical to rt_trace.hlsli: leaves inline, one deferred side
// per visit, near-first in the closest-hit walk.
// --------------------------------------------------------------------------

constexpr uint32_t k_stack_size = 64;

struct WalkStats {
    uint32_t max_stack = 0;
};

bool mirror_occluded(const std::vector<BvhNodeGpu>& nodes, const std::vector<BvhTriangle>& tris,
                     const Ray& ray, WalkStats& stats) {
    const glm::vec3 inv_dir = 1.0f / ray.dir;
    uint32_t stack[k_stack_size];
    uint32_t sp = 0;
    uint32_t node_index = 0;

    while (true) {
        const BvhNodeGpu& node = nodes[node_index];
        float t_l = 0.0f;
        float t_r = 0.0f;
        bool go_left = slab_test(node.lmin, node.lmax, ray.origin, inv_dir, ray.t_max, t_l);
        bool go_right = slab_test(node.rmin, node.rmax, ray.origin, inv_dir, ray.t_max, t_r);

        if (go_left && node.lcount != 0) {
            for (uint32_t i = 0; i < node.lcount; ++i) {
                if (intersect(tris[node.lchild + i], ray.origin, ray.dir, ray.t_max).hit) {
                    return true;
                }
            }
            go_left = false;
        }
        if (go_right && node.rcount != 0) {
            for (uint32_t i = 0; i < node.rcount; ++i) {
                if (intersect(tris[node.rchild + i], ray.origin, ray.dir, ray.t_max).hit) {
                    return true;
                }
            }
            go_right = false;
        }

        if (go_left) {
            if (go_right) {
                stack[sp++] = node.rchild;
                stats.max_stack = std::max(stats.max_stack, sp);
            }
            node_index = node.lchild;
        } else if (go_right) {
            node_index = node.rchild;
        } else if (sp > 0) {
            node_index = stack[--sp];
        } else {
            break;
        }
    }
    return false;
}

Closest mirror_closest(const std::vector<BvhNodeGpu>& nodes, const std::vector<BvhTriangle>& tris,
                       const Ray& ray, WalkStats& stats) {
    Closest best;
    best.t = ray.t_max;
    const glm::vec3 inv_dir = 1.0f / ray.dir;
    uint32_t stack[k_stack_size];
    uint32_t sp = 0;
    uint32_t node_index = 0;

    const auto leaf = [&](uint32_t first, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t index = first + i;
            const TriHit h = intersect(tris[index], ray.origin, ray.dir, best.t);
            if (h.hit && h.t < best.t) {
                best.t = h.t;
                best.tri_index = index;
                best.valid = true;
            }
        }
    };

    while (true) {
        const BvhNodeGpu& node = nodes[node_index];
        float t_l = 0.0f;
        float t_r = 0.0f;
        bool go_left = slab_test(node.lmin, node.lmax, ray.origin, inv_dir, best.t, t_l);
        bool go_right = slab_test(node.rmin, node.rmax, ray.origin, inv_dir, best.t, t_r);

        if (go_left && node.lcount != 0) {
            leaf(node.lchild, node.lcount);
            go_left = false;
        }
        if (go_right && node.rcount != 0) {
            leaf(node.rchild, node.rcount);
            go_right = false;
        }

        if (go_left && go_right) {
            uint32_t near = node.lchild;
            uint32_t far = node.rchild;
            if (t_r < t_l) {
                near = node.rchild;
                far = node.lchild;
            }
            stack[sp++] = far;
            stats.max_stack = std::max(stats.max_stack, sp);
            node_index = near;
        } else if (go_left) {
            node_index = node.lchild;
        } else if (go_right) {
            node_index = node.rchild;
        } else if (sp > 0) {
            node_index = stack[--sp];
        } else {
            break;
        }
    }
    return best;
}

// --------------------------------------------------------------------------
// Deterministic meshes.
// --------------------------------------------------------------------------

struct Mesh {
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
};

Mesh one_triangle() {
    return {{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, {0, 1, 2}};
}

/// Exactly `max_leaf_triangles` (4), i.e. the largest mesh whose root stays a
/// leaf: the degenerate-root path in the conversion.
Mesh four_triangles() {
    Mesh m;
    for (int i = 0; i < 4; ++i) {
        const float x = static_cast<float>(i);
        m.positions.push_back({x, 0.0f, 0.0f});
        m.positions.push_back({x + 0.8f, 0.0f, 0.0f});
        m.positions.push_back({x, 1.0f, 0.3f});
        const auto base = static_cast<uint32_t>(i * 3);
        m.indices.insert(m.indices.end(), {base, base + 1, base + 2});
    }
    return m;
}

Mesh random_soup(uint32_t triangles, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos(-20.0f, 20.0f);
    std::uniform_real_distribution<float> size(0.2f, 3.0f);
    Mesh m;
    for (uint32_t t = 0; t < triangles; ++t) {
        const glm::vec3 a{pos(rng), pos(rng), pos(rng)};
        const glm::vec3 b = a + glm::vec3{size(rng), size(rng), size(rng)};
        const glm::vec3 c = a + glm::vec3{-size(rng), size(rng), size(rng)};
        const auto base = static_cast<uint32_t>(m.positions.size());
        m.positions.insert(m.positions.end(), {a, b, c});
        m.indices.insert(m.indices.end(), {base, base + 1, base + 2});
    }
    return m;
}

/// Degenerate + duplicated triangles: zero-area slivers, exact duplicates
/// (which produce exact-t ties), and coincident centroids, which is the case
/// that drives the builder's median-split fallback.
Mesh degenerate_soup() {
    Mesh m;
    const auto push = [&m](glm::vec3 a, glm::vec3 b, glm::vec3 c) {
        const auto base = static_cast<uint32_t>(m.positions.size());
        m.positions.insert(m.positions.end(), {a, b, c});
        m.indices.insert(m.indices.end(), {base, base + 1, base + 2});
    };
    for (int i = 0; i < 12; ++i) {
        // Same triangle, twelve times: every centroid coincides.
        push({0, 0, 0}, {2, 0, 0}, {0, 2, 0});
    }
    for (int i = 0; i < 8; ++i) {
        const float x = static_cast<float>(i);
        push({x, 0, 5}, {x, 0, 5}, {x, 0, 5});       // a point
        push({x, 0, 6}, {x + 1, 0, 6}, {x + 2, 0, 6}); // collinear
        push({x, 1, 7}, {x + 1, 2, 7}, {x, 3, 7});
        push({x, 1, 7}, {x + 1, 2, 7}, {x, 3, 7});   // exact duplicate
    }
    return m;
}

struct RayBattery {
    std::vector<Ray> rays;
};

/// Rays aimed at the mesh's bounds from outside, plus a share of rays whose
/// origin is inside it, so both the "enters the root" and "starts inside a
/// leaf" paths are exercised.
RayBattery make_rays(const Blas& blas, uint32_t count, uint32_t seed) {
    const glm::vec3 lo = blas.bounds_min - glm::vec3(2.0f);
    const glm::vec3 hi = blas.bounds_max + glm::vec3(2.0f);
    const glm::vec3 center = (lo + hi) * 0.5f;
    const float radius = glm::length(hi - lo) + 1.0f;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_real_distribution<float> zero_one(0.0f, 1.0f);

    RayBattery out;
    out.rays.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        glm::vec3 dir{unit(rng), unit(rng), unit(rng)};
        if (glm::length(dir) < 1e-4f) {
            dir = glm::vec3(0.0f, 0.0f, 1.0f);
        }
        dir = glm::normalize(dir);

        Ray r;
        if (zero_one(rng) < 0.25f) {
            r.origin = glm::mix(lo, hi, glm::vec3(zero_one(rng), zero_one(rng), zero_one(rng)));
        } else {
            glm::vec3 offset{unit(rng), unit(rng), unit(rng)};
            if (glm::length(offset) < 1e-4f) {
                offset = glm::vec3(1.0f, 0.0f, 0.0f);
            }
            r.origin = center + glm::normalize(offset) * radius;
            // Aim at a jittered point in the box so most rays actually enter.
            const glm::vec3 target =
                glm::mix(lo, hi, glm::vec3(zero_one(rng), zero_one(rng), zero_one(rng)));
            dir = glm::normalize(target - r.origin);
        }
        r.dir = dir;
        r.t_max = radius * 4.0f;
        out.rays.push_back(r);
    }
    return out;
}

struct Comparison {
    uint32_t exact_ties = 0;
    uint32_t hits = 0;
    uint32_t max_stack = 0;
};

Comparison compare(const Mesh& mesh, uint32_t rays, uint32_t seed) {
    const Blas blas = build_blas(mesh.positions, mesh.indices);
    EXPECT_FALSE(blas.empty());
    const std::vector<BvhNodeGpu> gpu = convert_to_gpu_nodes(blas.nodes);
    EXPECT_FALSE(gpu.empty());

    const RayBattery battery = make_rays(blas, rays, seed);
    Comparison c;
    WalkStats stats;
    for (const Ray& ray : battery.rays) {
        EXPECT_EQ(reference_occluded(blas, ray), mirror_occluded(gpu, blas.triangles, ray, stats))
            << "any-hit disagreed for origin " << ray.origin.x << "," << ray.origin.y << ","
            << ray.origin.z;

        const Closest ref = reference_closest(blas, ray);
        const Closest mir = mirror_closest(gpu, blas.triangles, ray, stats);
        EXPECT_EQ(ref.valid, mir.valid) << "closest-hit validity disagreed";
        if (!ref.valid || !mir.valid) {
            continue;
        }
        ++c.hits;
        EXPECT_NEAR(ref.t, mir.t, 1e-4f);
        if (ref.tri_index != mir.tri_index) {
            // Only an EXACT tie may legitimately resolve differently: with any
            // gap at all, both walks take the strict minimum over the same hit
            // set and must agree.
            EXPECT_EQ(ref.t, mir.t)
                << "closest-hit primitive disagreed at distinguishable distances";
            ++c.exact_ties;
        }
    }
    c.max_stack = stats.max_stack;
    return c;
}

} // namespace

TEST(BvhGpuLayout, NodeIsSixtyFourBytes) {
    // Mirrors the HLSL struct in rt_trace.hlsli; the static_assert in the
    // header is the real gate, this pins the field offsets the shader reads.
    EXPECT_EQ(sizeof(BvhNodeGpu), 64u);
    EXPECT_EQ(offsetof(BvhNodeGpu, lchild), 12u);
    EXPECT_EQ(offsetof(BvhNodeGpu, lmax), 16u);
    EXPECT_EQ(offsetof(BvhNodeGpu, lcount), 28u);
    EXPECT_EQ(offsetof(BvhNodeGpu, rmin), 32u);
    EXPECT_EQ(offsetof(BvhNodeGpu, rchild), 44u);
    EXPECT_EQ(offsetof(BvhNodeGpu, rmax), 48u);
    EXPECT_EQ(offsetof(BvhNodeGpu, rcount), 60u);
}

TEST(BvhGpuLayout, OnlyInteriorNodesBecomeNodes) {
    const Mesh mesh = random_soup(300, 7u);
    const Blas blas = build_blas(mesh.positions, mesh.indices);
    uint32_t interior = 0;
    for (const BvhNode& n : blas.nodes) {
        interior += n.count == 0 ? 1u : 0u;
    }
    ASSERT_GT(interior, 0u);
    const std::vector<BvhNodeGpu> gpu = convert_to_gpu_nodes(blas.nodes);
    EXPECT_EQ(gpu.size(), interior);
    // The point of the format: half as many nodes at twice the size is the
    // same number of bytes, so nothing paid for the second AABB. Every node
    // has two children, so interior = (N-1)/2 and the converted array is
    // exactly one 32-byte node smaller than the tree it came from.
    EXPECT_EQ(gpu.size() * sizeof(BvhNodeGpu),
              (blas.nodes.size() - 1) * sizeof(BvhNode));
}

TEST(BvhGpuLayout, EveryLinkAndPrimitiveWindowIsInRange) {
    const Mesh mesh = random_soup(500, 11u);
    const Blas blas = build_blas(mesh.positions, mesh.indices);
    const std::vector<BvhNodeGpu> gpu = convert_to_gpu_nodes(blas.nodes);
    ASSERT_FALSE(gpu.empty());

    for (const BvhNodeGpu& n : gpu) {
        if (n.lcount == 0) {
            EXPECT_LT(n.lchild, gpu.size());
        } else {
            EXPECT_LE(static_cast<std::size_t>(n.lchild) + n.lcount, blas.triangles.size());
        }
        if (n.rcount == 0) {
            EXPECT_LT(n.rchild, gpu.size());
        } else {
            EXPECT_LE(static_cast<std::size_t>(n.rchild) + n.rcount, blas.triangles.size());
        }
    }
}

// Every triangle must appear in exactly one leaf window, or the walk quietly
// stops seeing geometry the builder placed.
TEST(BvhGpuLayout, LeafWindowsPartitionThePrimitives) {
    const Mesh mesh = random_soup(400, 23u);
    const Blas blas = build_blas(mesh.positions, mesh.indices);
    const std::vector<BvhNodeGpu> gpu = convert_to_gpu_nodes(blas.nodes);

    std::vector<uint32_t> seen(blas.triangles.size(), 0u);
    for (const BvhNodeGpu& n : gpu) {
        for (uint32_t i = 0; i < n.lcount; ++i) {
            seen[n.lchild + i]++;
        }
        for (uint32_t i = 0; i < n.rcount; ++i) {
            seen[n.rchild + i]++;
        }
    }
    for (std::size_t i = 0; i < seen.size(); ++i) {
        EXPECT_EQ(seen[i], 1u) << "triangle " << i << " is in " << seen[i] << " leaves";
    }
}

// --------------------------------------------------------------------------
// The degenerate roots, tested directly rather than only through a walk.
// --------------------------------------------------------------------------

TEST(BvhGpuLayout, SingleLeafBlasRootBecomesOneNodeWithAnUnreachableRightSide) {
    for (const Mesh& mesh : {one_triangle(), four_triangles()}) {
        const Blas blas = build_blas(mesh.positions, mesh.indices);
        ASSERT_EQ(blas.nodes.size(), 1u) << "expected a root that is itself a leaf";
        const std::vector<BvhNodeGpu> gpu = convert_to_gpu_nodes(blas.nodes);

        ASSERT_EQ(gpu.size(), 1u);
        EXPECT_EQ(gpu[0].lcount, blas.nodes[0].count);
        EXPECT_EQ(gpu[0].lchild, blas.nodes[0].left_first);
        EXPECT_EQ(gpu[0].lmin, blas.nodes[0].bounds_min);
        EXPECT_EQ(gpu[0].lmax, blas.nodes[0].bounds_max);

        // A LEAF, not an interior link: count > 0 is what stops any walk from
        // treating `rchild` as a node index into an array that has one entry.
        EXPECT_GT(gpu[0].rcount, 0u);
        EXPECT_EQ(gpu[0].rmin, glm::vec3(k_empty_side_coord));
        EXPECT_EQ(gpu[0].rmax, glm::vec3(k_empty_side_coord));

        // And no ray reaches it. A degenerate point box that far out fails the
        // slab test from every direction; an INVERTED box would not, which is
        // why the sentinel is a point and not min > max.
        std::mt19937 rng(99u);
        std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
        for (int i = 0; i < 2000; ++i) {
            glm::vec3 dir{unit(rng), unit(rng), unit(rng)};
            if (glm::length(dir) < 1e-4f) {
                continue;
            }
            dir = glm::normalize(dir);
            const glm::vec3 origin{unit(rng) * 50.0f, unit(rng) * 50.0f, unit(rng) * 50.0f};
            float t_near = 0.0f;
            EXPECT_FALSE(slab_test(gpu[0].rmin, gpu[0].rmax, origin, 1.0f / dir, 1e4f, t_near))
                << "the sentinel side is reachable";
        }
        // Including axis-aligned rays, where a component of invDir is +-inf.
        for (const glm::vec3& dir : {glm::vec3(1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(0, 1, 0),
                                     glm::vec3(0, -1, 0), glm::vec3(0, 0, 1), glm::vec3(0, 0, -1)}) {
            float t_near = 0.0f;
            EXPECT_FALSE(slab_test(gpu[0].rmin, gpu[0].rmax, glm::vec3(0.0f), 1.0f / dir, 1e4f,
                                   t_near));
        }
    }
}

TEST(BvhGpuLayout, SingleInstanceTlasRootBecomesOneNode) {
    TlasInstance inst;
    inst.bounds_min = {-1.0f, -1.0f, -1.0f};
    inst.bounds_max = {1.0f, 1.0f, 1.0f};
    inst.blas_node_offset = 17;
    inst.blas_triangle_offset = 42;

    const Tlas tlas = build_tlas(std::vector<TlasInstance>{inst});
    ASSERT_EQ(tlas.nodes.size(), 1u);
    const std::vector<BvhNodeGpu> gpu = convert_to_gpu_nodes(tlas.nodes);

    ASSERT_EQ(gpu.size(), 1u);
    EXPECT_EQ(gpu[0].lcount, 1u); // one instance in the real leaf
    EXPECT_EQ(gpu[0].lchild, 0u);
    EXPECT_EQ(gpu[0].lmin, inst.bounds_min);
    EXPECT_EQ(gpu[0].lmax, inst.bounds_max);
    EXPECT_GT(gpu[0].rcount, 0u);
    EXPECT_EQ(gpu[0].rmin, glm::vec3(k_empty_side_coord));
}

TEST(BvhGpuLayout, EmptyTreeConvertsToNothing) {
    EXPECT_TRUE(convert_to_gpu_nodes(std::span<const BvhNode>{}).empty());
}

// A multi-instance TLAS converts by the same rule, and every instance stays
// reachable through exactly one leaf window.
TEST(BvhGpuLayout, TlasLeafWindowsCoverEveryInstance) {
    std::vector<TlasInstance> instances;
    std::mt19937 rng(5u);
    std::uniform_real_distribution<float> pos(-100.0f, 100.0f);
    for (int i = 0; i < 200; ++i) {
        TlasInstance inst;
        const glm::vec3 p{pos(rng), pos(rng), pos(rng)};
        inst.bounds_min = p;
        inst.bounds_max = p + glm::vec3(2.0f);
        inst.blas_node_offset = static_cast<uint32_t>(i);
        instances.push_back(inst);
    }
    const Tlas tlas = build_tlas(instances);
    const std::vector<BvhNodeGpu> gpu = convert_to_gpu_nodes(tlas.nodes);
    ASSERT_FALSE(gpu.empty());

    std::vector<uint32_t> seen(tlas.instances.size(), 0u);
    for (const BvhNodeGpu& n : gpu) {
        for (uint32_t i = 0; i < n.lcount; ++i) {
            ASSERT_LT(n.lchild + i, seen.size());
            seen[n.lchild + i]++;
        }
        for (uint32_t i = 0; i < n.rcount; ++i) {
            ASSERT_LT(n.rchild + i, seen.size());
            seen[n.rchild + i]++;
        }
        if (n.lcount == 0) {
            EXPECT_LT(n.lchild, gpu.size());
        }
        if (n.rcount == 0) {
            EXPECT_LT(n.rchild, gpu.size());
        }
    }
    for (uint32_t s : seen) {
        EXPECT_EQ(s, 1u);
    }
}

// --------------------------------------------------------------------------
// The equivalence gate itself.
// --------------------------------------------------------------------------

TEST(BvhGpuLayout, WalkMatchesTheReferenceOnOneTriangle) {
    const Comparison c = compare(one_triangle(), 2000, 1u);
    EXPECT_EQ(c.exact_ties, 0u);
    EXPECT_GT(c.hits, 0u) << "the battery never hit anything; it proves nothing";
}

TEST(BvhGpuLayout, WalkMatchesTheReferenceAtExactlyMaxLeafSize) {
    const Comparison c = compare(four_triangles(), 2000, 2u);
    EXPECT_EQ(c.exact_ties, 0u);
    EXPECT_GT(c.hits, 0u);
}

TEST(BvhGpuLayout, WalkMatchesTheReferenceOnRandomSoup) {
    for (uint32_t seed : {3u, 4u, 5u}) {
        const Comparison c = compare(random_soup(400, seed), 4000, seed * 31u);
        EXPECT_GT(c.hits, 100u);
        // Random floats tie essentially never; a nonzero count here would mean
        // the "tie" escape hatch is doing work it was not opened for.
        EXPECT_LE(c.exact_ties, c.hits / 100u) << "exact-t ties are not supposed to be common";
        // The bound the stacks are sized from: no walk over a tree the builder
        // can produce should come anywhere near 64.
        EXPECT_LT(c.max_stack, k_stack_size);
    }
}

TEST(BvhGpuLayout, WalkMatchesTheReferenceOnDegenerateAndDuplicatedTriangles) {
    const Comparison c = compare(degenerate_soup(), 4000, 12u);
    // Exact duplicates DO tie, by construction — that is what this mesh is
    // for. The assertion is that the walks still agree on distance and
    // occlusion, which `compare` has already checked.
    EXPECT_GT(c.hits, 0u);
    EXPECT_LT(c.max_stack, k_stack_size);
}

// The TLAS path converts on every rebuild, i.e. whenever anything in the scene
// moves. This is not a benchmark (the test tree is unoptimized) — it is an
// upper bound on a cost that must stay a small fraction of the build it
// follows, printed so a regression is visible.
TEST(BvhGpuLayout, TlasConversionIsCheapRelativeToTheBuild) {
    std::vector<TlasInstance> instances;
    std::mt19937 rng(77u);
    std::uniform_real_distribution<float> pos(-500.0f, 500.0f);
    for (int i = 0; i < 2569; ++i) { // the playground's village
        TlasInstance inst;
        const glm::vec3 p{pos(rng), pos(rng), pos(rng)};
        inst.bounds_min = p;
        inst.bounds_max = p + glm::vec3(3.0f);
        instances.push_back(inst);
    }

    const auto build_start = std::chrono::steady_clock::now();
    const Tlas tlas = build_tlas(instances);
    const auto build_end = std::chrono::steady_clock::now();

    std::vector<BvhNodeGpu> out;
    std::vector<uint32_t> scratch;
    convert_to_gpu_nodes(tlas.nodes, out, scratch); // warm the vectors
    const auto convert_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) {
        convert_to_gpu_nodes(tlas.nodes, out, scratch);
    }
    const auto convert_end = std::chrono::steady_clock::now();

    const double build_ms =
        std::chrono::duration<double, std::milli>(build_end - build_start).count();
    const double convert_ms =
        std::chrono::duration<double, std::milli>(convert_end - convert_start).count() / 10.0;
    std::printf("[bvh_gpu] TLAS %zu instances: build %.3f ms, convert %.3f ms (%.1f%%)\n",
                instances.size(), build_ms, convert_ms, 100.0 * convert_ms / build_ms);

    EXPECT_FALSE(out.empty());
    EXPECT_LT(convert_ms, build_ms * 0.25)
        << "conversion is supposed to be a small fraction of the build it follows";
}
