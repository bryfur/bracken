#include "bracken/mesh_simplify.hpp"

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <unordered_map>

namespace bracken {
namespace {

struct Accumulator {
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f};
    glm::vec2 texcoord{0.0f};
    glm::vec4 color{0.0f};
    glm::vec4 tangent{0.0f};
    uint32_t count = 0;
    uint32_t out_index = 0;
};

} // namespace

SimplifiedMesh simplify_by_clustering(const std::vector<gpu::Vertex3D>& vertices,
                                      const std::vector<uint32_t>& indices, int grid_resolution) {
    SimplifiedMesh out;
    if (vertices.empty() || indices.size() < 3 || grid_resolution < 1) {
        return out;
    }

    glm::vec3 lo(std::numeric_limits<float>::max());
    glm::vec3 hi(std::numeric_limits<float>::lowest());
    for (const auto& v : vertices) {
        lo = glm::min(lo, v.position);
        hi = glm::max(hi, v.position);
    }

    const glm::vec3 extent = hi - lo;
    // A flat mesh (a plane, a billboard quad) has a zero extent on one axis;
    // dividing by it would put every vertex in cell 0 of that axis, which is
    // correct, or produce NaN, which is not.
    const glm::vec3 cell = glm::max(extent / static_cast<float>(grid_resolution), glm::vec3(1e-6f));

    // Cell index -> accumulator. A hash map rather than a dense grid: a
    // resolution of 64 is 262144 cells, of which a typical prop touches a few
    // hundred, and the dense version would dominate the runtime.
    std::unordered_map<uint64_t, Accumulator> cells;
    cells.reserve(vertices.size());

    std::vector<uint64_t> vertex_cell(vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const glm::vec3 rel = (vertices[i].position - lo) / cell;
        const auto gx = static_cast<uint64_t>(
            std::clamp(static_cast<int>(rel.x), 0, grid_resolution - 1));
        const auto gy = static_cast<uint64_t>(
            std::clamp(static_cast<int>(rel.y), 0, grid_resolution - 1));
        const auto gz = static_cast<uint64_t>(
            std::clamp(static_cast<int>(rel.z), 0, grid_resolution - 1));
        const uint64_t key = (gx << 42) | (gy << 21) | gz;
        vertex_cell[i] = key;

        Accumulator& a = cells[key];
        a.position += vertices[i].position;
        a.normal += vertices[i].normal;
        a.texcoord += vertices[i].texcoord;
        a.color += vertices[i].color;
        a.tangent += vertices[i].tangent;
        ++a.count;
    }

    // Emit representatives in a deterministic order. Iterating the hash map
    // directly would order output by bucket layout, which is stable within a
    // build but not something to rely on — sorting the keys makes the result
    // reproducible by construction.
    std::vector<uint64_t> keys;
    keys.reserve(cells.size());
    for (const auto& [key, _] : cells) {
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());

    out.vertices.reserve(keys.size());
    for (uint64_t key : keys) {
        Accumulator& a = cells[key];
        const auto n = static_cast<float>(a.count);
        gpu::Vertex3D v;
        v.position = a.position / n;
        const glm::vec3 avg_normal = a.normal / n;
        const float len = glm::length(avg_normal);
        // Opposing normals in one cell can average to ~zero; fall back to up
        // rather than emitting a NaN normal that blackens the whole mesh.
        v.normal = len > 1e-4f ? avg_normal / len : glm::vec3(0.0f, 1.0f, 0.0f);
        v.texcoord = a.texcoord / n;
        v.color = a.color / n;
        const glm::vec4 avg_tangent = a.tangent / n;
        const float tlen = glm::length(glm::vec3(avg_tangent));
        v.tangent = tlen > 1e-4f
                        ? glm::vec4(glm::vec3(avg_tangent) / tlen, avg_tangent.w >= 0.0f ? 1.0f : -1.0f)
                        : glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        a.out_index = static_cast<uint32_t>(out.vertices.size());
        out.vertices.push_back(v);
    }

    out.indices.reserve(indices.size());
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
        const uint32_t i0 = indices[i], i1 = indices[i + 1], i2 = indices[i + 2];
        if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size()) {
            continue; // malformed input: drop the triangle rather than trust it
        }
        const uint64_t c0 = vertex_cell[i0], c1 = vertex_cell[i1], c2 = vertex_cell[i2];
        // Two corners in one cell means the triangle collapsed to a line or a
        // point. Keeping it would render a zero-area sliver, which costs
        // rasterization and shows up as shimmer.
        if (c0 == c1 || c1 == c2 || c0 == c2) {
            continue;
        }
        out.indices.push_back(cells[c0].out_index);
        out.indices.push_back(cells[c1].out_index);
        out.indices.push_back(cells[c2].out_index);
    }

    return out;
}

SimplifiedMesh simplify_to_ratio(const std::vector<gpu::Vertex3D>& vertices,
                                 const std::vector<uint32_t>& indices, float target_ratio) {
    const std::size_t source_tris = indices.size() / 3;
    if (source_tris == 0 || target_ratio >= 1.0f) {
        return SimplifiedMesh{vertices, indices};
    }

    const auto target_tris =
        static_cast<std::size_t>(static_cast<float>(source_tris) * std::max(target_ratio, 0.0f));

    // Bisect on grid resolution. Triangle count rises monotonically with
    // resolution, so this converges; 12 steps is plenty for the 1..256 range
    // and keeps the whole thing well under a millisecond for typical props.
    // `best` tracks the coarsest result that still HAS geometry, not simply the
    // coarsest result. Resolution 1 puts every vertex in one cell, so it always
    // yields zero triangles and always "satisfies" the target — and on a small
    // mesh (a pebble, a flower) no other resolution meets the target either, so
    // seeding with it made the whole LOD level silently vanish. Ask for 10% of
    // a 12-triangle prop and the honest answer is "here are 6", not "nothing".
    int lo = 1;
    int hi = 256;
    SimplifiedMesh best;
    SimplifiedMesh smallest_nonempty;
    for (int step = 0; step < 12 && lo <= hi; ++step) {
        const int mid = (lo + hi) / 2;
        SimplifiedMesh candidate = simplify_by_clustering(vertices, indices, mid);
        const std::size_t tris = candidate.indices.size() / 3;

        if (tris > 0 && (smallest_nonempty.indices.empty() ||
                         tris < smallest_nonempty.indices.size() / 3)) {
            smallest_nonempty = candidate;
        }

        if (tris > target_tris) {
            hi = mid - 1;
        } else {
            if (tris > 0) {
                best = std::move(candidate);
            }
            lo = mid + 1;
        }
    }

    if (best.indices.size() >= 3) {
        return best;
    }
    // Nothing hit the target without vanishing: return the coarsest level that
    // still has triangles, else the original untouched.
    if (smallest_nonempty.indices.size() >= 3) {
        return smallest_nonempty;
    }
    return SimplifiedMesh{vertices, indices};
}

} // namespace bracken
