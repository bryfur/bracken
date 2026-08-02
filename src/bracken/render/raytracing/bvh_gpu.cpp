#include "bracken/render/raytracing/bvh_gpu.hpp"

#include "bracken/core/assert.hpp"

namespace bracken::render::rt {
namespace {

/// One side of a 64-byte node, in the order the fields sit in it.
struct Side {
    glm::vec3 lo{0.0f};
    uint32_t child = 0;
    glm::vec3 hi{0.0f};
    uint32_t count = 0;
};

/** The side describing source node `index`, whose new home the map already
 *  knows if it is interior. */
Side side_for(const BvhNode& node, uint32_t index, const std::vector<uint32_t>& map) {
    Side s;
    s.lo = node.bounds_min;
    s.hi = node.bounds_max;
    s.count = node.count;
    // Interior: point at the child's own 64-byte node. Leaf: the primitive
    // window is carried verbatim, since `left_first` already means "first
    // primitive" there.
    s.child = node.count == 0 ? map[index] : node.left_first;
    return s;
}

/// The unused side of a degenerate root: a leaf no ray can reach.
Side empty_side() {
    Side s;
    s.lo = glm::vec3(k_empty_side_coord);
    s.hi = glm::vec3(k_empty_side_coord);
    s.child = 0;
    s.count = 1;
    return s;
}

BvhNodeGpu make_node(const Side& left, const Side& right) {
    BvhNodeGpu n;
    n.lmin = left.lo;
    n.lchild = left.child;
    n.lmax = left.hi;
    n.lcount = left.count;
    n.rmin = right.lo;
    n.rchild = right.child;
    n.rmax = right.hi;
    n.rcount = right.count;
    return n;
}

} // namespace

void convert_to_gpu_nodes(std::span<const BvhNode> nodes, std::vector<BvhNodeGpu>& out,
                          std::vector<uint32_t>& scratch) {
    out.clear();
    if (nodes.empty()) {
        return;
    }

    // Pass 1 — number the interior nodes. They keep their relative order, so
    // the root (source index 0, when it is interior) stays new index 0.
    scratch.assign(nodes.size(), 0u);
    uint32_t interior = 0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].count == 0) {
            scratch[i] = interior++;
        }
    }

    if (interior == 0) {
        // Degenerate root: the whole tree is one leaf. See the header for why
        // this is encoded rather than special-cased in the walks.
        Side real;
        real.lo = nodes[0].bounds_min;
        real.hi = nodes[0].bounds_max;
        real.child = nodes[0].left_first;
        real.count = nodes[0].count;
        out.push_back(make_node(real, empty_side()));
        return;
    }

    // Pass 2 — write. Children are adjacent (`left_first`, `left_first + 1`),
    // which is exactly the invariant the 32-byte format existed to exploit and
    // the last thing this conversion needs from it.
    out.resize(interior);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const BvhNode& node = nodes[i];
        if (node.count != 0) {
            continue;
        }
        const uint32_t left = node.left_first;
        BRACKEN_ASSERT(static_cast<std::size_t>(left) + 1 < nodes.size(),
                       "BVH interior node's children are out of range");
        out[scratch[i]] =
            make_node(side_for(nodes[left], left, scratch), side_for(nodes[left + 1], left + 1, scratch));
    }
}

std::vector<BvhNodeGpu> convert_to_gpu_nodes(std::span<const BvhNode> nodes) {
    std::vector<BvhNodeGpu> out;
    std::vector<uint32_t> scratch;
    convert_to_gpu_nodes(nodes, out, scratch);
    return out;
}

} // namespace bracken::render::rt
