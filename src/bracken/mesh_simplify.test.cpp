#include <gtest/gtest.h>

#include "bracken/mesh_simplify.hpp"

#include <cmath>

using namespace bracken;

namespace {

/** A grid-tessellated plane: `n` x `n` quads in XZ. Predictable triangle count
 *  and a known bounding box, so expectations are exact rather than vibes. */
void make_grid(int n, std::vector<gpu::Vertex3D>& verts, std::vector<uint32_t>& idx) {
    for (int z = 0; z <= n; ++z) {
        for (int x = 0; x <= n; ++x) {
            gpu::Vertex3D v;
            v.position = {static_cast<float>(x), 0.0f, static_cast<float>(z)};
            v.normal = {0.0f, 1.0f, 0.0f};
            v.texcoord = {static_cast<float>(x) / static_cast<float>(n),
                          static_cast<float>(z) / static_cast<float>(n)};
            verts.push_back(v);
        }
    }
    const int stride = n + 1;
    for (int z = 0; z < n; ++z) {
        for (int x = 0; x < n; ++x) {
            const uint32_t a = static_cast<uint32_t>(z * stride + x);
            const uint32_t b = a + 1;
            const uint32_t c = a + static_cast<uint32_t>(stride);
            const uint32_t d = c + 1;
            idx.insert(idx.end(), {a, c, b, b, c, d});
        }
    }
}

/** A cube, to exercise a mesh with real extent on all three axes. */
void make_cube(std::vector<gpu::Vertex3D>& verts, std::vector<uint32_t>& idx) {
    const glm::vec3 corners[8] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                  {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    for (const glm::vec3& c : corners) {
        gpu::Vertex3D v;
        v.position = c;
        v.normal = glm::normalize(c - glm::vec3(0.5f));
        verts.push_back(v);
    }
    const uint32_t faces[36] = {0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,
                                3, 2, 6, 3, 6, 7, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2};
    idx.assign(faces, faces + 36);
}

std::size_t tri_count(const SimplifiedMesh& m) { return m.indices.size() / 3; }

} // namespace

TEST(MeshSimplify, ReducesTriangleCount) {
    std::vector<gpu::Vertex3D> v;
    std::vector<uint32_t> i;
    make_grid(32, v, i);
    const std::size_t before = i.size() / 3;

    const SimplifiedMesh out = simplify_by_clustering(v, i, 8);
    EXPECT_LT(tri_count(out), before);
    EXPECT_GT(tri_count(out), 0u) << "simplification must not delete the mesh";
}

TEST(MeshSimplify, CoarserGridMeansFewerTriangles) {
    std::vector<gpu::Vertex3D> v;
    std::vector<uint32_t> i;
    make_grid(32, v, i);
    EXPECT_LT(tri_count(simplify_by_clustering(v, i, 4)),
              tri_count(simplify_by_clustering(v, i, 16)));
}

TEST(MeshSimplify, IndicesStayInRange) {
    std::vector<gpu::Vertex3D> v;
    std::vector<uint32_t> i;
    make_grid(20, v, i);
    for (int res : {1, 2, 5, 13, 64}) {
        const SimplifiedMesh out = simplify_by_clustering(v, i, res);
        for (uint32_t index : out.indices) {
            ASSERT_LT(index, out.vertices.size()) << "res=" << res;
        }
        EXPECT_EQ(out.indices.size() % 3, 0u) << "res=" << res;
    }
}

// A collapsed triangle renders as a zero-area sliver: it costs rasterization
// and shows up as shimmer. They must be dropped, not emitted.
TEST(MeshSimplify, DropsDegenerateTriangles) {
    std::vector<gpu::Vertex3D> v;
    std::vector<uint32_t> i;
    make_grid(16, v, i);
    const SimplifiedMesh out = simplify_by_clustering(v, i, 3);
    for (std::size_t t = 0; t + 2 < out.indices.size(); t += 3) {
        EXPECT_NE(out.indices[t], out.indices[t + 1]);
        EXPECT_NE(out.indices[t + 1], out.indices[t + 2]);
        EXPECT_NE(out.indices[t], out.indices[t + 2]);
    }
}

TEST(MeshSimplify, StaysWithinTheOriginalBounds) {
    std::vector<gpu::Vertex3D> v;
    std::vector<uint32_t> i;
    make_cube(v, i);
    const SimplifiedMesh out = simplify_by_clustering(v, i, 4);
    for (const auto& vert : out.vertices) {
        EXPECT_GE(vert.position.x, -1e-4f);
        EXPECT_LE(vert.position.x, 1.0f + 1e-4f);
        EXPECT_GE(vert.position.y, -1e-4f);
        EXPECT_LE(vert.position.y, 1.0f + 1e-4f);
    }
}

// Averaging keeps the simplified surface centred on the original. Picking an
// arbitrary member of each cell would let the mesh drift or shrink.
TEST(MeshSimplify, RepresentativeIsTheCellAverage) {
    std::vector<gpu::Vertex3D> v;
    std::vector<uint32_t> i;
    make_grid(8, v, i);
    const SimplifiedMesh out = simplify_by_clustering(v, i, 1); // one cell

    ASSERT_EQ(out.vertices.size(), 1u);
    // The grid spans [0,8] on x and z, so its centroid is (4, 0, 4).
    EXPECT_NEAR(out.vertices[0].position.x, 4.0f, 1e-3f);
    EXPECT_NEAR(out.vertices[0].position.z, 4.0f, 1e-3f);
}

TEST(MeshSimplify, NormalsStayUnitLength) {
    std::vector<gpu::Vertex3D> v;
    std::vector<uint32_t> i;
    make_cube(v, i);
    const SimplifiedMesh out = simplify_by_clustering(v, i, 2);
    for (const auto& vert : out.vertices) {
        EXPECT_NEAR(glm::length(vert.normal), 1.0f, 1e-3f);
        EXPECT_FALSE(std::isnan(vert.normal.x));
    }
}

// A billboard quad has zero extent on one axis; the cell size must not become
// zero and produce NaN.
TEST(MeshSimplify, FlatMeshDoesNotProduceNaN) {
    std::vector<gpu::Vertex3D> v;
    std::vector<uint32_t> i;
    make_grid(4, v, i); // entirely in the y=0 plane
    const SimplifiedMesh out = simplify_by_clustering(v, i, 8);
    for (const auto& vert : out.vertices) {
        EXPECT_FALSE(std::isnan(vert.position.x));
        EXPECT_FALSE(std::isnan(vert.position.y));
        EXPECT_FALSE(std::isnan(vert.position.z));
    }
}

TEST(MeshSimplify, EmptyInputIsHandled) {
    EXPECT_TRUE(simplify_by_clustering({}, {}, 8).vertices.empty());
    std::vector<gpu::Vertex3D> v(3);
    EXPECT_TRUE(simplify_by_clustering(v, {}, 8).indices.empty());
}

TEST(MeshSimplify, MalformedIndicesAreDroppedNotTrusted) {
    std::vector<gpu::Vertex3D> v(4);
    // Last triangle references a vertex that does not exist.
    std::vector<uint32_t> i{0, 1, 2, 0, 1, 99};
    const SimplifiedMesh out = simplify_by_clustering(v, i, 4);
    for (uint32_t index : out.indices) {
        EXPECT_LT(index, out.vertices.size());
    }
}

// Generated LODs must be identical every run, or a level would differ between
// machines and the smoke run would stop being reproducible.
TEST(MeshSimplify, IsDeterministic) {
    std::vector<gpu::Vertex3D> v;
    std::vector<uint32_t> i;
    make_grid(24, v, i);
    const SimplifiedMesh a = simplify_by_clustering(v, i, 7);
    const SimplifiedMesh b = simplify_by_clustering(v, i, 7);
    ASSERT_EQ(a.vertices.size(), b.vertices.size());
    ASSERT_EQ(a.indices, b.indices);
    for (std::size_t k = 0; k < a.vertices.size(); ++k) {
        EXPECT_EQ(a.vertices[k].position, b.vertices[k].position);
    }
}

TEST(MeshSimplifyRatio, LandsNearTheRequestedRatio) {
    std::vector<gpu::Vertex3D> v;
    std::vector<uint32_t> i;
    make_grid(40, v, i);
    const std::size_t before = i.size() / 3;

    for (float ratio : {0.5f, 0.25f, 0.1f}) {
        const SimplifiedMesh out = simplify_to_ratio(v, i, ratio);
        const float actual = static_cast<float>(tri_count(out)) / static_cast<float>(before);
        EXPECT_LE(actual, ratio + 0.02f) << "ratio " << ratio << " overshot";
        EXPECT_GT(tri_count(out), 0u) << "ratio " << ratio << " produced nothing";
    }
}

TEST(MeshSimplifyRatio, RatioOfOneReturnsTheOriginal) {
    std::vector<gpu::Vertex3D> v;
    std::vector<uint32_t> i;
    make_grid(8, v, i);
    const SimplifiedMesh out = simplify_to_ratio(v, i, 1.0f);
    EXPECT_EQ(out.indices, i);
    EXPECT_EQ(out.vertices.size(), v.size());
}
