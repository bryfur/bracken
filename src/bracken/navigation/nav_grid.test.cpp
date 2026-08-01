#include <gtest/gtest.h>

#include "bracken/navigation/nav_grid.hpp"

#include <cmath>

using namespace bracken::nav;

namespace {

NavGridConfig flat_config(int n = 32) {
    NavGridConfig c;
    c.origin = glm::vec3(0.0f);
    c.cell_size = 10.0f;
    c.width = n;
    c.depth = n;
    c.max_slope_degrees = 45.0f;
    c.max_step = 40.0f;
    return c;
}

NavGrid build_flat(int n = 32) {
    NavGrid g;
    g.build(flat_config(n), [](float, float) { return 0.0f; });
    return g;
}

/** Total XZ length of a path. */
float path_length(const std::vector<glm::vec3>& p) {
    float total = 0.0f;
    for (std::size_t i = 1; i < p.size(); ++i) {
        total += glm::length(glm::vec2(p[i].x - p[i - 1].x, p[i].z - p[i - 1].z));
    }
    return total;
}

} // namespace

TEST(NavGrid, FlatGroundIsFullyWalkable) {
    const NavGrid g = build_flat();
    EXPECT_TRUE(g.built());
    EXPECT_EQ(g.walkable_count(), 32u * 32u);
}

TEST(NavGrid, SteepGroundIsNotWalkable) {
    NavGrid g;
    // A wall of height rising 1000 units per cell: far past 45 degrees.
    g.build(flat_config(16), [](float x, float) { return x * 100.0f; });
    EXPECT_LT(g.walkable_count(), 16u * 16u);
}

TEST(NavGrid, InvalidConfigIsRejectedNotCrashed) {
    NavGrid g;
    NavGridConfig c = flat_config();
    c.width = 0;
    g.build(c, [](float, float) { return 0.0f; });
    EXPECT_FALSE(g.built());

    std::vector<glm::vec3> path;
    EXPECT_FALSE(g.find_path({0, 0, 0}, {10, 0, 10}, path));
}

TEST(NavGrid, FindsAStraightPathOnOpenGround) {
    const NavGrid g = build_flat();
    std::vector<glm::vec3> path;
    ASSERT_TRUE(g.find_path({15.0f, 0.0f, 15.0f}, {295.0f, 0.0f, 295.0f}, path));
    ASSERT_GE(path.size(), 2u);

    // On open ground the smoother should collapse it to nearly a straight
    // line — that is the whole point of string-pulling.
    const float direct = glm::length(glm::vec2(295.0f - 15.0f, 295.0f - 15.0f));
    EXPECT_LT(path_length(path), direct * 1.05f);
    EXPECT_LE(path.size(), 4u) << "open ground should not need many waypoints";
}

TEST(NavGrid, PathStartsAtTheStartAndEndsAtTheGoal) {
    const NavGrid g = build_flat();
    std::vector<glm::vec3> path;
    const glm::vec3 start{25.0f, 0.0f, 25.0f};
    const glm::vec3 goal{200.0f, 0.0f, 120.0f};
    ASSERT_TRUE(g.find_path(start, goal, path));
    EXPECT_FLOAT_EQ(path.front().x, start.x);
    EXPECT_FLOAT_EQ(path.front().z, start.z);
    EXPECT_FLOAT_EQ(path.back().x, goal.x);
    EXPECT_FLOAT_EQ(path.back().z, goal.z);
}

// The real test of a pathfinder: route around an obstacle rather than through.
TEST(NavGrid, RoutesAroundAWall) {
    NavGrid g;
    NavGridConfig c = flat_config(32);
    // A wall across the middle with a gap at high Z.
    g.build(c, [](float, float) { return 0.0f; }, [](const glm::vec3& p) {
        const bool in_wall_row = p.x > 150.0f && p.x < 170.0f;
        const bool in_gap = p.z > 280.0f;
        return in_wall_row && !in_gap;
    });

    std::vector<glm::vec3> path;
    ASSERT_TRUE(g.find_path({50.0f, 0.0f, 50.0f}, {250.0f, 0.0f, 50.0f}, path));

    // Every waypoint-to-waypoint segment must stay on walkable ground, and the
    // detour must be longer than the blocked straight line.
    for (std::size_t i = 1; i < path.size(); ++i) {
        EXPECT_TRUE(g.line_of_walk(path[i - 1], path[i])) << "segment " << i << " crosses the wall";
    }
    EXPECT_GT(path_length(path), 200.0f * 1.5f) << "it must actually detour via the gap";
}

TEST(NavGrid, ReportsFailureWhenTheGoalIsWalledOff) {
    NavGrid g;
    // A wall with NO gap: the goal is unreachable.
    g.build(flat_config(32), [](float, float) { return 0.0f; },
            [](const glm::vec3& p) { return p.x > 150.0f && p.x < 170.0f; });

    std::vector<glm::vec3> path;
    EXPECT_FALSE(g.find_path({50.0f, 0.0f, 50.0f}, {250.0f, 0.0f, 50.0f}, path));
    EXPECT_TRUE(path.empty());
}

// Agents and destinations routinely sit just off the mesh; failing outright is
// useless behavior.
TEST(NavGrid, SnapsToTheNearestWalkableCell) {
    NavGrid g;
    g.build(flat_config(32), [](float, float) { return 0.0f; },
            [](const glm::vec3& p) { return p.x < 100.0f; }); // left strip blocked

    glm::ivec2 cell{};
    ASSERT_TRUE(g.nearest_walkable({20.0f, 0.0f, 150.0f}, cell));
    EXPECT_TRUE(g.walkable(cell.x, cell.y));

    // ...and a path from inside the blocked strip still works.
    std::vector<glm::vec3> path;
    EXPECT_TRUE(g.find_path({20.0f, 0.0f, 150.0f}, {250.0f, 0.0f, 150.0f}, path));
}

TEST(NavGrid, NearestWalkableFailsWhenNothingIsNear) {
    NavGrid g;
    g.build(flat_config(32), [](float, float) { return 0.0f; },
            [](const glm::vec3&) { return true; }); // everything blocked
    glm::ivec2 cell{};
    EXPECT_FALSE(g.nearest_walkable({150.0f, 0.0f, 150.0f}, cell));
}

// A tall step between two individually-walkable cells is a cliff, not a path.
TEST(NavGrid, WillNotStepOffACliff) {
    NavGrid g;
    NavGridConfig c = flat_config(32);
    c.max_step = 20.0f;
    // Two flat plateaus with a 500-unit drop between them. Each plateau is
    // walkable; the join is not traversable.
    g.build(c, [](float x, float) { return x < 160.0f ? 0.0f : 500.0f; });

    std::vector<glm::vec3> path;
    EXPECT_FALSE(g.find_path({50.0f, 0.0f, 150.0f}, {250.0f, 500.0f, 150.0f}, path))
        << "a 500-unit drop must not be traversable with max_step 20";
}

TEST(NavGrid, WorldCellRoundTrip) {
    const NavGrid g = build_flat();
    for (int x : {0, 7, 31}) {
        for (int z : {0, 13, 31}) {
            const glm::vec3 world = g.cell_to_world(x, z);
            const glm::ivec2 cell = g.world_to_cell(world);
            EXPECT_EQ(cell.x, x);
            EXPECT_EQ(cell.y, z);
        }
    }
}

TEST(NavGrid, OutOfBoundsQueriesAreSafe) {
    const NavGrid g = build_flat(8);
    EXPECT_FALSE(g.walkable(-1, 0));
    EXPECT_FALSE(g.walkable(0, -1));
    EXPECT_FALSE(g.walkable(8, 0));
    EXPECT_FALSE(g.walkable(0, 8));
    EXPECT_FLOAT_EQ(g.height_at(-5, -5), 0.0f);
}

// Several agents path concurrently; find_path must not mutate the grid.
TEST(NavGrid, FindPathIsConstAndRepeatable) {
    const NavGrid g = build_flat();
    std::vector<glm::vec3> a;
    std::vector<glm::vec3> b;
    ASSERT_TRUE(g.find_path({15, 0, 15}, {285, 0, 205}, a));
    ASSERT_TRUE(g.find_path({15, 0, 15}, {285, 0, 205}, b));
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i], b[i]) << "waypoint " << i;
    }
}

TEST(NavGrid, PathFollowsTerrainHeight) {
    NavGrid g;
    // Gentle ramp, walkable throughout.
    g.build(flat_config(32), [](float x, float) { return x * 0.05f; });
    std::vector<glm::vec3> path;
    ASSERT_TRUE(g.find_path({15.0f, 0.0f, 150.0f}, {295.0f, 0.0f, 150.0f}, path));
    // Interior waypoints carry the sampled ground height, so an agent walking
    // them does not sink or float.
    for (std::size_t i = 1; i + 1 < path.size(); ++i) {
        EXPECT_NEAR(path[i].y, path[i].x * 0.05f, 10.0f);
    }
}
