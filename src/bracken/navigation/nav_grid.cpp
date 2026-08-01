#include "bracken/navigation/nav_grid.hpp"

#include "bracken/core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <queue>

namespace bracken::nav {
namespace {

/** Eight-way movement. Diagonals cost sqrt(2) so a diagonal is not a cheap
 *  shortcut — with uniform cost, A* produces visibly wrong zig-zag routes. */
constexpr int k_dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
constexpr int k_dz[8] = {0, 0, 1, -1, 1, -1, 1, -1};
constexpr float k_cost[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.41421356f, 1.41421356f, 1.41421356f,
                             1.41421356f};

struct OpenNode {
    float f = 0.0f;
    int index = 0;
    // Greater-than so std::priority_queue (a max-heap) pops the SMALLEST f.
    bool operator<(const OpenNode& o) const { return f > o.f; }
};

} // namespace

void NavGrid::build(const NavGridConfig& config, const HeightFn& height, const BlockedFn& blocked) {
    config_ = config;
    flags_.clear();
    heights_.clear();
    walkable_count_ = 0;

    if (config.width <= 0 || config.depth <= 0 || config.cell_size <= 0.0f || !height) {
        BRACKEN_LOG_ERROR("nav", "NavGrid::build: invalid config ({}x{} cells, size {})",
                          config.width, config.depth, config.cell_size);
        return;
    }

    const auto cells = static_cast<std::size_t>(config.width) * static_cast<std::size_t>(config.depth);
    flags_.assign(cells, CellFlag::Blocked);
    heights_.assign(cells, 0.0f);

    // Pass 1: sample heights.
    for (int z = 0; z < config.depth; ++z) {
        for (int x = 0; x < config.width; ++x) {
            const glm::vec3 world = cell_to_world(x, z);
            heights_[index(x, z)] = height(world.x, world.z);
        }
    }

    // Pass 2: slope and obstruction. Slope is measured from the height
    // gradient across neighbours rather than from a normal, because the height
    // function is all we are given.
    const float cos_limit = std::cos(glm::radians(config.max_slope_degrees));
    for (int z = 0; z < config.depth; ++z) {
        for (int x = 0; x < config.width; ++x) {
            const std::size_t i = index(x, z);
            const float h = heights_[i];

            const float hx = in_bounds(x + 1, z) ? heights_[index(x + 1, z)] : h;
            const float hz = in_bounds(x, z + 1) ? heights_[index(x, z + 1)] : h;
            const glm::vec3 normal =
                glm::normalize(glm::vec3(-(hx - h), config.cell_size, -(hz - h)));

            if (normal.y < cos_limit) {
                continue; // too steep
            }
            if (blocked) {
                glm::vec3 world = cell_to_world(x, z);
                world.y = h;
                if (blocked(world)) {
                    continue;
                }
            }
            flags_[i] = CellFlag::Walkable;
            ++walkable_count_;
        }
    }

    BRACKEN_LOG_INFO("nav", "NavGrid: {}x{} cells ({} units), {} walkable ({:.0f}%)", config.width,
                     config.depth, config.cell_size, walkable_count_,
                     100.0 * static_cast<double>(walkable_count_) / static_cast<double>(cells));
}

bool NavGrid::walkable(int x, int z) const {
    return in_bounds(x, z) && flags_[index(x, z)] == CellFlag::Walkable;
}

float NavGrid::height_at(int x, int z) const {
    return in_bounds(x, z) ? heights_[index(x, z)] : 0.0f;
}

glm::ivec2 NavGrid::world_to_cell(const glm::vec3& world) const {
    const glm::vec3 local = world - config_.origin;
    return {static_cast<int>(std::floor(local.x / config_.cell_size)),
            static_cast<int>(std::floor(local.z / config_.cell_size))};
}

glm::vec3 NavGrid::cell_to_world(int x, int z) const {
    // Cell CENTRE, not corner: an agent walking to a corner clips the
    // neighbouring cell, which may be a wall.
    return config_.origin + glm::vec3((static_cast<float>(x) + 0.5f) * config_.cell_size, 0.0f,
                                      (static_cast<float>(z) + 0.5f) * config_.cell_size);
}

bool NavGrid::nearest_walkable(const glm::vec3& world, glm::ivec2& out_cell,
                               int max_radius_cells) const {
    const glm::ivec2 start = world_to_cell(world);
    if (walkable(start.x, start.y)) {
        out_cell = start;
        return true;
    }
    // Expanding square rings. Not a true nearest-by-distance, but it is
    // deterministic, allocation-free and close enough for "snap me onto the
    // mesh".
    for (int r = 1; r <= max_radius_cells; ++r) {
        for (int dz = -r; dz <= r; ++dz) {
            for (int dx = -r; dx <= r; ++dx) {
                if (std::abs(dx) != r && std::abs(dz) != r) {
                    continue; // interior of the ring: already checked
                }
                const int x = start.x + dx;
                const int z = start.y + dz;
                if (walkable(x, z)) {
                    out_cell = {x, z};
                    return true;
                }
            }
        }
    }
    return false;
}

bool NavGrid::line_of_walk(const glm::vec3& from, const glm::vec3& to) const {
    // Sample along the segment at half-cell steps: fine enough not to tunnel
    // through a one-cell gap, coarse enough to stay cheap inside the smoother.
    const glm::vec3 delta = to - from;
    const float distance = glm::length(glm::vec2(delta.x, delta.z));
    const int steps = std::max(1, static_cast<int>(distance / (config_.cell_size * 0.5f)));
    for (int i = 0; i <= steps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(steps);
        const glm::ivec2 cell = world_to_cell(from + delta * t);
        if (!walkable(cell.x, cell.y)) {
            return false;
        }
    }
    return true;
}

bool NavGrid::find_path(const glm::vec3& start, const glm::vec3& goal,
                        std::vector<glm::vec3>& out_path) const {
    out_path.clear();
    if (!built()) {
        return false;
    }

    glm::ivec2 start_cell{};
    glm::ivec2 goal_cell{};
    if (!nearest_walkable(start, start_cell) || !nearest_walkable(goal, goal_cell)) {
        return false;
    }
    if (start_cell == goal_cell) {
        out_path.push_back(start);
        out_path.push_back(goal);
        return true;
    }

    const auto cells = flags_.size();
    // Parallel arrays rather than a node struct: A* touches g and f far more
    // than it touches came_from, and this keeps the hot arrays dense.
    std::vector<float> g(cells, std::numeric_limits<float>::infinity());
    std::vector<int> came_from(cells, -1);
    std::vector<uint8_t> closed(cells, 0);

    const int start_index = static_cast<int>(index(start_cell.x, start_cell.y));
    const int goal_index = static_cast<int>(index(goal_cell.x, goal_cell.y));

    auto heuristic = [&](int idx) {
        const int x = idx % config_.width;
        const int z = idx / config_.width;
        const float dx = static_cast<float>(std::abs(x - goal_cell.x));
        const float dz = static_cast<float>(std::abs(z - goal_cell.y));
        // Octile distance: the exact cost of the cheapest 8-way route on flat
        // ground, so it is admissible AND tight. Euclidean would be admissible
        // but looser, and expand many more nodes for the same answer.
        return (dx + dz) + (1.41421356f - 2.0f) * std::min(dx, dz);
    };

    std::priority_queue<OpenNode> open;
    g[static_cast<std::size_t>(start_index)] = 0.0f;
    open.push({heuristic(start_index), start_index});

    bool found = false;
    while (!open.empty()) {
        const OpenNode current = open.top();
        open.pop();
        const auto ci = static_cast<std::size_t>(current.index);
        if (closed[ci]) {
            continue; // a cheaper route already expanded this cell
        }
        closed[ci] = 1;

        if (current.index == goal_index) {
            found = true;
            break;
        }

        const int cx = current.index % config_.width;
        const int cz = current.index / config_.width;
        const float ch = heights_[ci];

        for (int d = 0; d < 8; ++d) {
            const int nx = cx + k_dx[d];
            const int nz = cz + k_dz[d];
            if (!walkable(nx, nz)) {
                continue;
            }
            const std::size_t ni = index(nx, nz);
            if (closed[ni]) {
                continue;
            }
            // A step too tall is not traversable even between two walkable
            // cells: this is what keeps paths off cliff edges.
            if (std::fabs(heights_[ni] - ch) > config_.max_step) {
                continue;
            }
            // Diagonals may not cut a corner between two blocked cells, or
            // agents clip through the diagonal gap between two walls.
            if (k_dx[d] != 0 && k_dz[d] != 0) {
                if (!walkable(cx + k_dx[d], cz) || !walkable(cx, cz + k_dz[d])) {
                    continue;
                }
            }

            const float tentative = g[ci] + k_cost[d];
            if (tentative < g[ni]) {
                g[ni] = tentative;
                came_from[ni] = current.index;
                open.push({tentative + heuristic(static_cast<int>(ni)), static_cast<int>(ni)});
            }
        }
    }

    if (!found) {
        return false;
    }

    // Walk the parents back, then reverse.
    std::vector<glm::vec3> raw;
    for (int at = goal_index; at != -1; at = came_from[static_cast<std::size_t>(at)]) {
        const int x = at % config_.width;
        const int z = at / config_.width;
        glm::vec3 p = cell_to_world(x, z);
        p.y = heights_[static_cast<std::size_t>(at)];
        raw.push_back(p);
        if (at == start_index) {
            break;
        }
    }
    std::reverse(raw.begin(), raw.end());

    // String-pull: keep a waypoint only when the straight line from the last
    // kept point to the NEXT one is blocked. Raw A* output follows the grid and
    // an agent walking it reads as drunk.
    out_path.push_back(start);
    std::size_t anchor = 0;
    for (std::size_t i = 1; i < raw.size(); ++i) {
        if (!line_of_walk(raw[anchor], raw[i])) {
            out_path.push_back(raw[i - 1]);
            anchor = i - 1;
        }
    }
    out_path.push_back(goal);
    return true;
}

} // namespace bracken::nav
