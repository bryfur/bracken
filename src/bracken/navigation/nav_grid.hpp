#pragma once

#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <span>
#include <vector>

namespace bracken::nav {

/** How a cell may be used. */
enum class CellFlag : uint8_t {
    Blocked = 0, ///< off-mesh: too steep, obstructed, or outside the world
    Walkable = 1,
};

struct NavGridConfig {
    glm::vec3 origin{0.0f};   ///< world position of cell (0,0)'s corner
    float cell_size = 100.0f; ///< world units per cell
    int width = 0;            ///< cells along +X
    int depth = 0;            ///< cells along +Z

    /** Slopes steeper than this are not walkable. Should match (or slightly
     *  undercut) the character controller's max_slope_degrees — a path that
     *  routes over ground the controller refuses to climb is worse than no
     *  path, because the agent walks into it and sticks. */
    float max_slope_degrees = 45.0f;

    /** Height difference between neighbouring cells an agent can step. Cells
     *  further apart vertically than this are not connected even when both are
     *  individually walkable — that is what stops paths crossing a cliff edge. */
    float max_step = 40.0f;
};

/**
 * A walkability grid built from a height function, with A* pathfinding.
 *
 * WHY A GRID AND NOT A POLYGON NAVMESH. A polygon mesh is what a shipping game
 * eventually wants: it is smaller, its paths need less smoothing, and it
 * handles overhangs. It is also a substantial piece of computational geometry
 * (voxelize, region-grow, contour, triangulate) that is easy to get subtly
 * wrong and hard to debug. A grid is built from the height function this
 * engine already has, is trivially inspectable, and is enough for outdoor
 * terrain — which is what this engine renders. The A* and the path-smoothing
 * are the same either way, so the grid can be replaced later without changing
 * the caller.
 *
 * It is height-per-cell, so it cannot represent a bridge over a road. That
 * limitation is stated rather than hidden.
 *
 * Tier: build() on any thread (it only calls the supplied height function).
 * find_path() is const and worker-safe — several agents can path at once.
 */
class NavGrid {
public:
    /** Height of the walkable surface at a world XZ, or nullopt for "no ground
     *  here". The playground passes its terrain sampler. */
    using HeightFn = std::function<float(float x, float z)>;

    /** True when a world position is obstructed by something the grid should
     *  route around (a building, a rock). Optional. */
    using BlockedFn = std::function<bool(const glm::vec3& world_position)>;

    void build(const NavGridConfig& config, const HeightFn& height,
               const BlockedFn& blocked = nullptr);

    [[nodiscard]] bool built() const noexcept { return !flags_.empty(); }
    [[nodiscard]] const NavGridConfig& config() const noexcept { return config_; }

    [[nodiscard]] bool walkable(int x, int z) const;
    [[nodiscard]] float height_at(int x, int z) const;

    /** Cell containing a world position. May be out of range; check with
     *  in_bounds(). */
    [[nodiscard]] glm::ivec2 world_to_cell(const glm::vec3& world) const;
    [[nodiscard]] glm::vec3 cell_to_world(int x, int z) const;
    [[nodiscard]] bool in_bounds(int x, int z) const noexcept {
        return x >= 0 && z >= 0 && x < config_.width && z < config_.depth;
    }

    /** Nearest walkable cell to `world`, searched outward in rings. Agents and
     *  destinations routinely sit just inside a wall or just off the mesh, and
     *  failing the whole query for that is useless behavior. Returns false if
     *  nothing walkable exists within `max_radius_cells`. */
    [[nodiscard]] bool nearest_walkable(const glm::vec3& world, glm::ivec2& out_cell,
                                        int max_radius_cells = 16) const;

    /**
     * A* from `start` to `goal`, returning world-space waypoints.
     *
     * The path is string-pulled before returning: raw A* output follows the
     * grid and looks like a staircase, which reads as drunk when an agent
     * walks it. Waypoints that a straight line can skip are removed.
     *
     * Returns false when no route exists. `out_path` is cleared first, and the
     * first waypoint is the start position so callers can walk the list
     * uniformly.
     */
    [[nodiscard]] bool find_path(const glm::vec3& start, const glm::vec3& goal,
                                 std::vector<glm::vec3>& out_path) const;

    /** Is the straight segment between two world points walkable end to end?
     *  Exposed because it is what path smoothing uses, and because "can I just
     *  walk at him" is the cheapest useful AI query there is. */
    [[nodiscard]] bool line_of_walk(const glm::vec3& from, const glm::vec3& to) const;

    /** Walkable cell count, for diagnostics — a nav grid that silently came
     *  out empty is otherwise indistinguishable from one that works. */
    [[nodiscard]] std::size_t walkable_count() const noexcept { return walkable_count_; }

private:
    [[nodiscard]] std::size_t index(int x, int z) const noexcept {
        return static_cast<std::size_t>(z) * static_cast<std::size_t>(config_.width) +
               static_cast<std::size_t>(x);
    }

    NavGridConfig config_{};
    std::vector<CellFlag> flags_;
    std::vector<float> heights_;
    std::size_t walkable_count_ = 0;
};

} // namespace bracken::nav
