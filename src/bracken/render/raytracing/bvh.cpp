#include "bracken/render/raytracing/bvh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace bracken::render::rt {
namespace {

constexpr float k_inf = std::numeric_limits<float>::infinity();

struct Aabb {
    glm::vec3 lo{k_inf};
    glm::vec3 hi{-k_inf};

    void grow(const glm::vec3& p) {
        lo = glm::min(lo, p);
        hi = glm::max(hi, p);
    }
    void grow(const Aabb& b) {
        lo = glm::min(lo, b.lo);
        hi = glm::max(hi, b.hi);
    }
    [[nodiscard]] glm::vec3 extent() const { return glm::max(hi - lo, glm::vec3(0.0f)); }
    [[nodiscard]] float area() const {
        const glm::vec3 e = extent();
        // Half the surface area; the constant factor cancels in every SAH
        // comparison, so it is left out.
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }
    [[nodiscard]] bool valid() const { return hi.x >= lo.x; }
};

/** Per-primitive data the builder sorts, kept separate from the output so the
 *  final triangle array can be written once in leaf order. */
struct BuildPrim {
    Aabb bounds;
    glm::vec3 centroid{0.0f};
    uint32_t source = 0;
};

struct Bin {
    Aabb bounds;
    uint32_t count = 0;
};

/** Recursive builder shared by BLAS and TLAS: both are BVHs, differing only in
 *  what a leaf holds. */
struct Builder {
    std::vector<BuildPrim>& prims;
    std::vector<BvhNode>& nodes;
    const BvhBuildOptions& options;
    uint32_t max_depth_seen = 0;
    uint32_t leaf_count = 0;

    void build_range(uint32_t node_index, uint32_t first, uint32_t count, uint32_t depth) {
        max_depth_seen = std::max(max_depth_seen, depth);

        Aabb bounds;
        Aabb centroid_bounds;
        for (uint32_t i = 0; i < count; ++i) {
            bounds.grow(prims[first + i].bounds);
            centroid_bounds.grow(prims[first + i].centroid);
        }
        nodes[node_index].bounds_min = bounds.lo;
        nodes[node_index].bounds_max = bounds.hi;

        const bool must_leaf = count <= options.max_leaf_triangles || depth >= options.max_depth;
        if (must_leaf) {
            nodes[node_index].left_first = first;
            nodes[node_index].count = count;
            ++leaf_count;
            return;
        }

        // --- binned SAH -------------------------------------------------
        const glm::vec3 cextent = centroid_bounds.extent();
        int axis = 0;
        if (cextent.y > cextent.x) {
            axis = 1;
        }
        if (cextent[2] > cextent[axis]) {
            axis = 2;
        }
        if (cextent[axis] <= 1e-8f) {
            // Every centroid coincides — no split can separate them, and
            // trying produces an infinite recursion on one side.
            nodes[node_index].left_first = first;
            nodes[node_index].count = count;
            ++leaf_count;
            return;
        }

        const uint32_t bin_count = std::max(2u, options.sah_bins);
        std::vector<Bin> bins(bin_count);
        const float scale = static_cast<float>(bin_count) / cextent[axis];
        const float axis_lo = centroid_bounds.lo[axis];

        for (uint32_t i = 0; i < count; ++i) {
            const BuildPrim& p = prims[first + i];
            auto b = static_cast<uint32_t>((p.centroid[axis] - axis_lo) * scale);
            b = std::min(b, bin_count - 1);
            bins[b].count++;
            bins[b].bounds.grow(p.bounds);
        }

        // Sweep both directions once, so each candidate split's left and right
        // costs are available without an O(bins^2) pass.
        std::vector<float> left_area(bin_count, 0.0f);
        std::vector<uint32_t> left_count(bin_count, 0);
        {
            Aabb acc;
            uint32_t n = 0;
            for (uint32_t i = 0; i < bin_count; ++i) {
                acc.grow(bins[i].bounds);
                n += bins[i].count;
                left_area[i] = acc.valid() ? acc.area() : 0.0f;
                left_count[i] = n;
            }
        }

        float best_cost = k_inf;
        uint32_t best_split = 0;
        {
            Aabb acc;
            uint32_t n = 0;
            for (uint32_t i = bin_count - 1; i > 0; --i) {
                acc.grow(bins[i].bounds);
                n += bins[i].count;
                const float right_area = acc.valid() ? acc.area() : 0.0f;
                const uint32_t lc = left_count[i - 1];
                if (lc == 0 || n == 0) {
                    continue;
                }
                const float cost = left_area[i - 1] * static_cast<float>(lc) +
                                   right_area * static_cast<float>(n);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_split = i;
                }
            }
        }

        // Making a leaf costs count intersections; splitting costs a traversal
        // step plus the weighted children. If the split does not pay, stop —
        // this is the check that keeps SAH trees from over-splitting.
        const float leaf_cost = static_cast<float>(count) * bounds.area();
        if (best_cost >= leaf_cost || best_split == 0) {
            nodes[node_index].left_first = first;
            nodes[node_index].count = count;
            ++leaf_count;
            return;
        }

        const float split_at = axis_lo + static_cast<float>(best_split) / scale;
        auto mid = std::partition(prims.begin() + first, prims.begin() + first + count,
                                  [axis, split_at](const BuildPrim& p) {
                                      return p.centroid[axis] < split_at;
                                  });
        auto left_n = static_cast<uint32_t>(std::distance(prims.begin() + first, mid));
        if (left_n == 0 || left_n == count) {
            // The partition put everything on one side despite the bins
            // suggesting otherwise (coincident centroids at the boundary).
            // Fall back to a median split rather than recursing forever.
            left_n = count / 2;
            std::nth_element(prims.begin() + first, prims.begin() + first + left_n,
                             prims.begin() + first + count,
                             [axis](const BuildPrim& a, const BuildPrim& b) {
                                 return a.centroid[axis] < b.centroid[axis];
                             });
        }

        // Siblings are emitted adjacently so traversal can derive the right
        // child as left+1 and the node can stay 32 bytes.
        const auto left_index = static_cast<uint32_t>(nodes.size());
        nodes.emplace_back();
        nodes.emplace_back();
        nodes[node_index].left_first = left_index;
        nodes[node_index].count = 0;

        build_range(left_index, first, left_n, depth + 1);
        build_range(left_index + 1, first + left_n, count - left_n, depth + 1);
    }
};

void fill_stats(BvhBuildStats* out, const std::vector<BvhNode>& nodes, uint32_t prim_count,
                uint32_t leaves, uint32_t depth) {
    if (out == nullptr) {
        return;
    }
    out->node_count = static_cast<uint32_t>(nodes.size());
    out->leaf_count = leaves;
    out->max_depth = depth;
    out->triangle_count = prim_count;
    out->average_leaf_size =
        leaves > 0 ? static_cast<float>(prim_count) / static_cast<float>(leaves) : 0.0f;
    out->sah_cost = sah_cost(nodes);
}

} // namespace

float sah_cost(std::span<const BvhNode> nodes) {
    if (nodes.empty()) {
        return 0.0f;
    }
    const Aabb root{nodes[0].bounds_min, nodes[0].bounds_max};
    const float root_area = root.area();
    if (root_area <= 0.0f) {
        return 0.0f;
    }

    float cost = 0.0f;
    for (const BvhNode& n : nodes) {
        const Aabb b{n.bounds_min, n.bounds_max};
        const float p = b.area() / root_area;
        // Interior nodes cost a traversal step; leaves cost one test per
        // primitive. Both weighted by the probability a ray reaches them.
        cost += n.count == 0 ? p : p * static_cast<float>(n.count);
    }
    return cost;
}

Blas build_blas(std::span<const glm::vec3> positions, std::span<const uint32_t> indices,
                const BvhBuildOptions& options, BvhBuildStats* out_stats) {
    Blas blas;
    const std::size_t tri_count = indices.size() / 3;
    if (tri_count == 0 || positions.empty()) {
        fill_stats(out_stats, blas.nodes, 0, 0, 0);
        return blas;
    }

    std::vector<BuildPrim> prims;
    prims.reserve(tri_count);
    std::vector<std::array<glm::vec3, 3>> verts;
    verts.reserve(tri_count);

    for (std::size_t t = 0; t < tri_count; ++t) {
        const uint32_t i0 = indices[t * 3 + 0];
        const uint32_t i1 = indices[t * 3 + 1];
        const uint32_t i2 = indices[t * 3 + 2];
        if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size()) {
            continue; // malformed input: drop rather than trust
        }
        const glm::vec3 a = positions[i0];
        const glm::vec3 b = positions[i1];
        const glm::vec3 c = positions[i2];

        BuildPrim p;
        p.bounds.grow(a);
        p.bounds.grow(b);
        p.bounds.grow(c);
        p.centroid = (a + b + c) * (1.0f / 3.0f);
        p.source = static_cast<uint32_t>(verts.size());
        prims.push_back(p);
        verts.push_back({a, b, c});
    }

    if (prims.empty()) {
        fill_stats(out_stats, blas.nodes, 0, 0, 0);
        return blas;
    }

    blas.nodes.reserve(prims.size() * 2);
    blas.nodes.emplace_back();
    Builder builder{prims, blas.nodes, options};
    builder.build_range(0, 0, static_cast<uint32_t>(prims.size()), 0);

    // Write triangles in LEAF ORDER. A leaf's primitives then occupy a
    // contiguous run, so intersecting a leaf is a linear scan of adjacent
    // memory instead of a gather — which on a GPU is the difference between
    // one coalesced fetch and four scattered ones.
    blas.triangles.resize(prims.size());
    for (std::size_t i = 0; i < prims.size(); ++i) {
        const auto& v = verts[prims[i].source];
        BvhTriangle& t = blas.triangles[i];
        t.v0 = v[0];
        t.e1 = v[1] - v[0];
        t.e2 = v[2] - v[0];
        t.prim_id = prims[i].source;
    }

    blas.bounds_min = blas.nodes[0].bounds_min;
    blas.bounds_max = blas.nodes[0].bounds_max;
    fill_stats(out_stats, blas.nodes, static_cast<uint32_t>(prims.size()), builder.leaf_count,
               builder.max_depth_seen);
    return blas;
}

Tlas build_tlas(std::span<const TlasInstance> instances, const BvhBuildOptions& options,
                BvhBuildStats* out_stats) {
    Tlas tlas;
    if (instances.empty()) {
        fill_stats(out_stats, tlas.nodes, 0, 0, 0);
        return tlas;
    }

    std::vector<BuildPrim> prims;
    prims.reserve(instances.size());
    for (std::size_t i = 0; i < instances.size(); ++i) {
        BuildPrim p;
        p.bounds.grow(instances[i].bounds_min);
        p.bounds.grow(instances[i].bounds_max);
        p.centroid = (instances[i].bounds_min + instances[i].bounds_max) * 0.5f;
        p.source = static_cast<uint32_t>(i);
        prims.push_back(p);
    }

    BvhBuildOptions tlas_options = options;
    // One instance per leaf: an instance test is a full BLAS descent, so
    // grouping several behind one AABB just means doing more of them.
    tlas_options.max_leaf_triangles = 1;

    tlas.nodes.reserve(prims.size() * 2);
    tlas.nodes.emplace_back();
    Builder builder{prims, tlas.nodes, tlas_options};
    builder.build_range(0, 0, static_cast<uint32_t>(prims.size()), 0);

    tlas.instances.resize(prims.size());
    for (std::size_t i = 0; i < prims.size(); ++i) {
        tlas.instances[i] = instances[prims[i].source];
    }

    fill_stats(out_stats, tlas.nodes, static_cast<uint32_t>(prims.size()), builder.leaf_count,
               builder.max_depth_seen);
    return tlas;
}

} // namespace bracken::render::rt
