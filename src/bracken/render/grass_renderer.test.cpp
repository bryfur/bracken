// Unit tests for the CPU side of GrassRenderer: chunk culling (exclusion
// rects, edge margin, world bounds, view distance, frustum), the view-distance
// API, and wind-direction normalization (audit findings M1 / L19 / L20 in
// docs/audits/documentation_sweep.md).

#include "bracken/render/grass_renderer.hpp"

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/scene/frustum.hpp"
#include <algorithm>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <span>
#include <vector>

using bracken::render::GrassExclusionRect;
using bracken::render::GrassRenderer;

namespace {

constexpr float WORLD = 8000.0f;
constexpr float CHUNK = GrassRenderer::CHUNK_SIZE;

GrassRenderer::ChunkCullParams make_params(const glm::vec3& camera_pos, float view_distance = 600.0f) {
    GrassRenderer::ChunkCullParams p;
    p.camera_pos = camera_pos;
    p.view_distance = view_distance;
    p.world_width = WORLD;
    p.world_height = WORLD;
    p.edge_margin = 0.0f;
    p.y_min = -100.0f;
    p.y_max = 170.0f;
    return p;
}

// Uninitialized caller-owned pool: cull_chunks runs its fan-out inline (one
// chunk) — the semantic tests below don't exercise parallelism.
std::vector<glm::vec4> cull(const GrassRenderer::ChunkCullParams& params,
                            std::span<const GrassExclusionRect> exclusions = {},
                            const bracken::scene::Frustum* frustum = nullptr) {
    bracken::core::jobs::JobSystem js;
    std::vector<glm::vec4> chunks;
    GrassRenderer::cull_chunks(js, params, exclusions, frustum, chunks);
    return chunks;
}

bool any_chunk_center_in_rect(const std::vector<glm::vec4>& chunks, float min_x, float min_z, float max_x,
                              float max_z) {
    return std::any_of(chunks.begin(), chunks.end(), [&](const glm::vec4& c) {
        float cx = c.x + CHUNK * 0.5f;
        float cz = c.z + CHUNK * 0.5f;
        return cx >= min_x && cx <= max_x && cz >= min_z && cz <= max_z;
    });
}

} // namespace

// ============================================================================
// M1: no hardcoded clearing / margin
// ============================================================================

// The old code unconditionally carved a 400x400 "town" clearing at the world
// center. By default grass must now grow everywhere, including dead center.
TEST(GrassChunkCulling, NoDefaultClearingAtWorldCenter) {
    const glm::vec3 center(WORLD * 0.5f, 0.0f, WORLD * 0.5f);
    auto chunks = cull(make_params(center));

    ASSERT_FALSE(chunks.empty());
    // Chunks exist inside the old ±200-unit town box, including at the camera.
    EXPECT_TRUE(any_chunk_center_in_rect(chunks, center.x - 200.0f, center.z - 200.0f, center.x + 200.0f,
                                         center.z + 200.0f));
}

// The old code hardcoded a 50-unit bare ring inside the world edges. With the
// default edge_margin of 0, grass reaches the world bounds; only chunks with
// no overlap of the world rect are culled.
TEST(GrassChunkCulling, DefaultMarginZeroKeepsEdgeChunks) {
    auto chunks = cull(make_params(glm::vec3(0.0f, 0.0f, 0.0f), 300.0f));

    ASSERT_FALSE(chunks.empty());
    for (const glm::vec4& c : chunks) {
        // Every kept chunk overlaps the world rect.
        EXPECT_GE(c.x + CHUNK, 0.0f);
        EXPECT_LE(c.x, WORLD);
        EXPECT_GE(c.z + CHUNK, 0.0f);
        EXPECT_LE(c.z, WORLD);
    }
    // The corner chunk at the world origin survives (old code culled it).
    EXPECT_TRUE(std::any_of(chunks.begin(), chunks.end(),
                            [](const glm::vec4& c) { return c.x == 0.0f && c.z == 0.0f; }));
}

TEST(GrassChunkCulling, EdgeMarginExcludesBorderRing) {
    auto params = make_params(glm::vec3(0.0f, 0.0f, 0.0f), 300.0f);
    params.edge_margin = 50.0f;
    auto chunks = cull(params);

    ASSERT_FALSE(chunks.empty());
    for (const glm::vec4& c : chunks) {
        // No chunk lies fully inside the 50-unit border ring.
        EXPECT_GE(c.x + CHUNK, 50.0f);
        EXPECT_LE(c.x, WORLD - 50.0f);
        EXPECT_GE(c.z + CHUNK, 50.0f);
        EXPECT_LE(c.z, WORLD - 50.0f);
    }
    // The origin corner chunk [0,32]x[0,32] is fully inside the ring -> culled.
    EXPECT_FALSE(std::any_of(chunks.begin(), chunks.end(),
                             [](const glm::vec4& c) { return c.x == 0.0f && c.z == 0.0f; }));
}

TEST(GrassChunkCulling, ExclusionRectsCullChunkCenters) {
    const glm::vec3 center(WORLD * 0.5f, 0.0f, WORLD * 0.5f);
    const GrassExclusionRect town{center.x - 200.0f, center.z - 200.0f, center.x + 200.0f, center.z + 200.0f};
    const std::vector<GrassExclusionRect> exclusions{town};

    auto excluded = cull(make_params(center), exclusions);
    auto unrestricted = cull(make_params(center));

    ASSERT_FALSE(excluded.empty());
    EXPECT_FALSE(any_chunk_center_in_rect(excluded, town.min_x, town.min_z, town.max_x, town.max_z));
    // Grass still grows outside the rect, and the rect removed something.
    EXPECT_LT(excluded.size(), unrestricted.size());
}

TEST(GrassChunkCulling, MultipleExclusionRects) {
    const glm::vec3 cam(1000.0f, 0.0f, 1000.0f);
    const std::vector<GrassExclusionRect> exclusions{
        {900.0f, 900.0f, 1100.0f, 1100.0f},
        {700.0f, 700.0f, 800.0f, 800.0f},
    };
    auto chunks = cull(make_params(cam), exclusions);

    ASSERT_FALSE(chunks.empty());
    for (const auto& rect : exclusions) {
        EXPECT_FALSE(any_chunk_center_in_rect(chunks, rect.min_x, rect.min_z, rect.max_x, rect.max_z));
    }
}

TEST(GrassChunkCulling, RespectsViewDistance) {
    const glm::vec3 cam(4000.0f, 0.0f, 4000.0f);
    const float view_dist = 320.0f;
    auto chunks = cull(make_params(cam, view_dist));

    ASSERT_FALSE(chunks.empty());
    for (const glm::vec4& c : chunks) {
        float dx = (c.x + CHUNK * 0.5f) - cam.x;
        float dz = (c.z + CHUNK * 0.5f) - cam.z;
        EXPECT_LE(dx * dx + dz * dz, view_dist * view_dist);
    }
}

TEST(GrassChunkCulling, FrustumCullsBehindCamera) {
    const glm::vec3 cam(4000.0f, 50.0f, 4000.0f);
    auto params = make_params(cam);

    // Engine-style reversed-Z projection (near/far swapped) looking down -Z.
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 2000.0f, 0.1f);
    glm::mat4 view = glm::lookAt(cam, cam + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    bracken::scene::Frustum frustum;
    frustum.extract_from_matrix(proj * view);

    auto with_frustum = cull(params, {}, &frustum);
    auto without_frustum = cull(params);

    ASSERT_FALSE(with_frustum.empty());
    EXPECT_LT(with_frustum.size(), without_frustum.size());
    // Nothing survives more than one chunk behind the camera plane.
    for (const glm::vec4& c : with_frustum) {
        EXPECT_LE(c.z, cam.z + CHUNK);
    }
}

// ============================================================================
// Determinism: chunked fan-out parity (multithreading_plan §4.3/§6.2)
// ============================================================================

// The chunked grid walk must produce a BIT-identical visible list at any
// worker count: per-chunk partials merged in chunk-index order. Order matters
// downstream — upload_chunks' FNV hash of the visible set keys on it.
TEST(GrassChunkCullingParallel, BitIdenticalAcrossWorkerCounts) {
    bracken::core::jobs::JobSystem js1;
    js1.init(1);
    bracken::core::jobs::JobSystem jsN;
    jsN.init(4);

    // Max view distance -> 79 grid rows -> multiple fan-out chunks at
    // CULL_ROW_GRAIN = 48, so jsN genuinely exercises the parallel merge.
    const glm::vec3 cam(WORLD * 0.5f, 40.0f, WORLD * 0.5f);
    auto params = make_params(cam, GrassRenderer::MAX_VIEW_DISTANCE);
    const std::vector<GrassExclusionRect> exclusions{
        {cam.x - 200.0f, cam.z - 200.0f, cam.x + 200.0f, cam.z + 200.0f},
        {cam.x + 400.0f, cam.z - 600.0f, cam.x + 700.0f, cam.z - 300.0f},
    };

    // Fixed engine-style reversed-Z frustum looking down -Z.
    glm::mat4 proj = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 2000.0f, 0.1f);
    glm::mat4 view = glm::lookAt(cam, cam + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    bracken::scene::Frustum frustum;
    frustum.extract_from_matrix(proj * view);

    std::vector<glm::vec4> serial;
    GrassRenderer::cull_chunks(js1, params, exclusions, &frustum, serial);
    ASSERT_FALSE(serial.empty());
    ASSERT_GT(serial.size(), GrassRenderer::CULL_ROW_GRAIN); // sanity: non-trivial workload

    for (int iteration = 0; iteration < 8; ++iteration) {
        std::vector<glm::vec4> parallel;
        GrassRenderer::cull_chunks(jsN, params, exclusions, &frustum, parallel);
        ASSERT_EQ(parallel.size(), serial.size());
        EXPECT_EQ(0, std::memcmp(parallel.data(), serial.data(), serial.size() * sizeof(glm::vec4)));
    }

    // No-frustum path parity too (different survivor pattern).
    std::vector<glm::vec4> serial_nf;
    GrassRenderer::cull_chunks(js1, params, exclusions, nullptr, serial_nf);
    std::vector<glm::vec4> parallel_nf;
    GrassRenderer::cull_chunks(jsN, params, exclusions, nullptr, parallel_nf);
    ASSERT_EQ(parallel_nf.size(), serial_nf.size());
    EXPECT_EQ(0, std::memcmp(parallel_nf.data(), serial_nf.data(), serial_nf.size() * sizeof(glm::vec4)));
}

// ============================================================================
// L19: view-distance API
// ============================================================================

TEST(GrassViewDistance, DefaultIsMaxViewDistance) {
    GrassRenderer grass;
    EXPECT_FLOAT_EQ(grass.effective_view_distance(), GrassRenderer::MAX_VIEW_DISTANCE);
}

TEST(GrassViewDistance, HostValueIsRespected) {
    GrassRenderer grass;
    grass.grass_view_distance = 300.0f;
    EXPECT_FLOAT_EQ(grass.effective_view_distance(), 300.0f);
}

TEST(GrassViewDistance, HostValueClampedToMax) {
    GrassRenderer grass;
    grass.grass_view_distance = 5000.0f;
    EXPECT_FLOAT_EQ(grass.effective_view_distance(), GrassRenderer::MAX_VIEW_DISTANCE);
}

TEST(GrassViewDistance, CapBoundsHostValue) {
    GrassRenderer grass;
    grass.set_view_distance_cap(500.0f);
    EXPECT_FLOAT_EQ(grass.effective_view_distance(), 500.0f);

    // A host value below the cap wins.
    grass.grass_view_distance = 300.0f;
    EXPECT_FLOAT_EQ(grass.effective_view_distance(), 300.0f);

    // Raising the cap back releases the bound (no ratchet).
    grass.grass_view_distance = 900.0f;
    grass.set_view_distance_cap(4000.0f);
    EXPECT_FLOAT_EQ(grass.effective_view_distance(), 900.0f);
}

TEST(GrassViewDistance, NonPositiveCapDisables) {
    GrassRenderer grass;
    grass.set_view_distance_cap(0.0f);
    EXPECT_FLOAT_EQ(grass.effective_view_distance(), GrassRenderer::MAX_VIEW_DISTANCE);
}

// ============================================================================
// L20: wind direction
// ============================================================================

TEST(GrassWind, DefaultDirectionMatchesLegacyConstant) {
    GrassRenderer grass;
    EXPECT_FLOAT_EQ(grass.wind_direction.x, 1.0f);
    EXPECT_FLOAT_EQ(grass.wind_direction.y, 0.3f);
}

TEST(GrassWind, EffectiveDirectionIsNormalized) {
    GrassRenderer grass;
    grass.wind_direction = glm::vec2(3.0f, 4.0f);
    glm::vec2 dir = grass.effective_wind_direction();
    EXPECT_NEAR(dir.x, 0.6f, 1e-6f);
    EXPECT_NEAR(dir.y, 0.8f, 1e-6f);
    EXPECT_NEAR(glm::length(dir), 1.0f, 1e-6f);
}

TEST(GrassWind, ZeroDirectionIsSafe) {
    GrassRenderer grass;
    grass.wind_direction = glm::vec2(0.0f, 0.0f);
    glm::vec2 dir = grass.effective_wind_direction();
    EXPECT_FLOAT_EQ(dir.x, 0.0f);
    EXPECT_FLOAT_EQ(dir.y, 0.0f);
}
