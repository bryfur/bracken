// Unit tests for HeightmapGrid — the paged terrain heightfield.
//
// Three things are being pinned here and they are not equally interesting.
//
//  (1) THE ADDITIVITY CLAIM. `from_single` must reproduce `Heightmap`'s own
//      sampling BIT FOR BIT, because SceneRenderer::set_heightmap is now the
//      1x1 adapter over the grid path and every existing host goes through it.
//      Anything less than bit equality would move the pinned smoke baselines.
//  (2) SEAM BEHAVIOUR. Two pages that share a boundary line must agree on it
//      exactly, and a sweep across the seam must not step.
//  (3) THE REFUSAL. heightmap.hpp has warned since before there was a grid
//      that independently packed tiles quantize differently; validate() turns
//      that warning into a hard error, and these tests are what make the
//      refusal a contract rather than a comment.
//
// Everything here is GPU-free.

#include "bracken/heightmap_grid.hpp"

#include "bracken/heightmap.hpp"
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using bracken::HeightRange;
using bracken::Heightmap;
using bracken::HeightmapGrid;
using bracken::WorldBounds;

namespace {

constexpr float PAGE = 1000.0f;
constexpr uint32_t RES = 17; // 16 cells per page side
constexpr HeightRange RANGE{-60.0f, 60.0f};

/// Gentle, seam-agnostic analytic terrain: smooth enough that bilinear
/// reconstruction error over a 62.5-unit cell stays far below the assertions
/// below, and aperiodic enough that a page-index bug shows up.
float analytic(float x, float z) {
    return 20.0f * std::sin(x / 700.0f) + 15.0f * std::cos(z / 900.0f) + 4.0f * std::sin((x + z) / 310.0f);
}

/// One page's worth of world-unit samples, INCLUDING both boundary lines —
/// which is what makes neighbouring pages agree on the line they share.
std::vector<float> page_samples(float origin_x, float origin_z) {
    std::vector<float> heights(static_cast<std::size_t>(RES) * RES);
    for (uint32_t iz = 0; iz < RES; ++iz) {
        const float z = origin_z + PAGE * static_cast<float>(iz) / static_cast<float>(RES - 1);
        for (uint32_t ix = 0; ix < RES; ++ix) {
            const float x = origin_x + PAGE * static_cast<float>(ix) / static_cast<float>(RES - 1);
            heights[static_cast<std::size_t>(iz) * RES + ix] = analytic(x, z);
        }
    }
    return heights;
}

HeightmapGrid make_2x2() {
    HeightmapGrid grid = HeightmapGrid::create(2, 2, PAGE, 0.0f, 0.0f, RANGE);
    for (uint32_t pz = 0; pz < 2; ++pz) {
        for (uint32_t px = 0; px < 2; ++px) {
            const std::vector<float> h =
                page_samples(static_cast<float>(px) * PAGE, static_cast<float>(pz) * PAGE);
            EXPECT_TRUE(grid.set_page(px, pz, h, RES));
        }
    }
    return grid;
}

/// A deterministic pseudo-random probe sequence — no <random>, so the corpus
/// is identical on every platform and a failure is reproducible from the seed.
struct Probe {
    uint32_t state = 0x9E3779B9u;
    float next(float lo, float hi) {
        state = state * 1664525u + 1013904223u;
        const float t = static_cast<float>(state >> 8) / static_cast<float>(1u << 24);
        return lo + t * (hi - lo);
    }
};

} // namespace

// ============================================================================
// (1) The 1x1 adapter — the additivity gate
// ============================================================================

TEST(HeightmapGrid, FromSingleReproducesTheMapsHeightBitForBit) {
    const std::vector<float> h = page_samples(0.0f, 0.0f);
    const Heightmap map = Heightmap::from_floats(h, RES, WorldBounds{0.0f, 0.0f, PAGE}, RANGE);
    ASSERT_TRUE(map.valid());
    const HeightmapGrid grid = HeightmapGrid::from_single(map);
    ASSERT_TRUE(grid.valid()) << (grid.validate().is_err() ? grid.validate().error() : std::string{});

    // Well outside the map on both axes too: the single map clamps to its own
    // edge and the grid must clamp to the same edge, not to something else.
    Probe probe;
    for (int i = 0; i < 10000; ++i) {
        const float x = probe.next(-2.0f * PAGE, 3.0f * PAGE);
        const float z = probe.next(-2.0f * PAGE, 3.0f * PAGE);
        EXPECT_EQ(grid.get_height_world(x, z), map.get_height_world(x, z)) << "at (" << x << ", " << z << ")";
    }
}

TEST(HeightmapGrid, FromSingleReproducesTheMapsNormalBitForBit) {
    const std::vector<float> h = page_samples(0.0f, 0.0f);
    const Heightmap map = Heightmap::from_floats(h, RES, WorldBounds{0.0f, 0.0f, PAGE}, RANGE);
    const HeightmapGrid grid = HeightmapGrid::from_single(map);

    Probe probe;
    for (int i = 0; i < 2000; ++i) {
        const float x = probe.next(-PAGE, 2.0f * PAGE);
        const float z = probe.next(-PAGE, 2.0f * PAGE);
        const glm::vec3 a = grid.get_normal_world(x, z);
        const glm::vec3 b = map.get_normal_world(x, z);
        EXPECT_EQ(a.x, b.x) << "at (" << x << ", " << z << ")";
        EXPECT_EQ(a.y, b.y) << "at (" << x << ", " << z << ")";
        EXPECT_EQ(a.z, b.z) << "at (" << x << ", " << z << ")";
    }
}

TEST(HeightmapGrid, FromSingleCarriesTheMapsGeometryAndRange) {
    Heightmap map = Heightmap::from_floats(page_samples(120.0f, -80.0f), RES,
                                           WorldBounds{120.0f, -80.0f, PAGE}, RANGE);
    const HeightmapGrid grid = HeightmapGrid::from_single(map);
    EXPECT_EQ(grid.pages_x, 1u);
    EXPECT_EQ(grid.pages_z, 1u);
    EXPECT_EQ(grid.page_size, PAGE);
    EXPECT_EQ(grid.origin_x, 120.0f);
    EXPECT_EQ(grid.origin_z, -80.0f);
    EXPECT_EQ(grid.range.min, RANGE.min);
    EXPECT_EQ(grid.range.max, RANGE.max);
    EXPECT_EQ(grid.resident_pages(), 1u);
    EXPECT_EQ(grid.page_resolution(), RES);
    EXPECT_TRUE(grid.valid());
}

TEST(HeightmapGrid, ANonResidentSingleGridIsValidAndAnswersTheFallback) {
    // A streamed world before anything has arrived: legal, empty, and it must
    // hand back the caller's fallback rather than an invented edge height.
    const HeightmapGrid grid = HeightmapGrid::create(1, 1, PAGE, 0.0f, 0.0f, RANGE);
    EXPECT_TRUE(grid.valid());
    EXPECT_EQ(grid.resident_pages(), 0u);
    EXPECT_EQ(grid.page_resolution(), 0u);
    EXPECT_EQ(grid.get_height_world(500.0f, 500.0f, -777.0f), -777.0f);
    EXPECT_FALSE(grid.resident_at(500.0f, 500.0f));
    // No resolution anywhere -> no stencil; up is the documented answer.
    EXPECT_EQ(grid.get_normal_world(500.0f, 500.0f), glm::vec3(0.0f, 1.0f, 0.0f));
}

// ============================================================================
// (2) Addressing and page lookup
// ============================================================================

TEST(HeightmapGrid, PageAtAddressesTheGridAndRejectsTheOutside) {
    const HeightmapGrid grid = make_2x2();
    EXPECT_EQ(grid.page_at(1.0f, 1.0f), glm::ivec2(0, 0));
    EXPECT_EQ(grid.page_at(999.0f, 1.0f), glm::ivec2(0, 0));
    EXPECT_EQ(grid.page_at(1001.0f, 1.0f), glm::ivec2(1, 0));
    EXPECT_EQ(grid.page_at(1.0f, 1500.0f), glm::ivec2(0, 1));
    EXPECT_EQ(grid.page_at(1500.0f, 1500.0f), glm::ivec2(1, 1));
    // The interior seam line belongs to the page it starts.
    EXPECT_EQ(grid.page_at(PAGE, 0.0f), glm::ivec2(1, 0));
    // The grid's OUTER boundary line belongs to the last page.
    EXPECT_EQ(grid.page_at(2.0f * PAGE, 2.0f * PAGE), glm::ivec2(1, 1));
    // Outside is outside.
    EXPECT_EQ(grid.page_at(-0.5f, 500.0f), glm::ivec2(-1, -1));
    EXPECT_EQ(grid.page_at(500.0f, 2.0f * PAGE + 0.5f), glm::ivec2(-1, -1));
}

TEST(HeightmapGrid, PageLookupReturnsResidentPagesAndNullOtherwise) {
    HeightmapGrid grid = make_2x2();
    ASSERT_EQ(grid.resident_pages(), 4u);
    for (int pz = 0; pz < 2; ++pz) {
        for (int px = 0; px < 2; ++px) {
            const Heightmap* page = grid.page(px, pz);
            ASSERT_NE(page, nullptr) << px << "," << pz;
            EXPECT_EQ(page->world_origin_x, static_cast<float>(px) * PAGE);
            EXPECT_EQ(page->world_origin_z, static_cast<float>(pz) * PAGE);
            EXPECT_EQ(page->world_size, PAGE);
        }
    }
    EXPECT_EQ(grid.page(-1, 0), nullptr);
    EXPECT_EQ(grid.page(2, 0), nullptr);
    EXPECT_EQ(grid.page(0, 2), nullptr);

    grid.clear_page(1, 0);
    EXPECT_EQ(grid.page(1, 0), nullptr);
    EXPECT_EQ(grid.resident_pages(), 3u);
    EXPECT_TRUE(grid.valid()) << "a hole is a streaming state, not an error";
    EXPECT_FALSE(grid.resident_at(1500.0f, 100.0f));
    EXPECT_TRUE(grid.resident_at(500.0f, 100.0f));
    EXPECT_EQ(grid.get_height_world(1500.0f, 100.0f, -12.5f), -12.5f);
}

TEST(HeightmapGrid, PageBoundsCoverTheGridWithoutGaps) {
    const HeightmapGrid grid = make_2x2();
    EXPECT_EQ(grid.page_min_xz(0, 0), glm::vec2(0.0f, 0.0f));
    EXPECT_EQ(grid.page_max_xz(0, 0), glm::vec2(PAGE, PAGE));
    EXPECT_EQ(grid.page_min_xz(1, 1), glm::vec2(PAGE, PAGE));
    EXPECT_EQ(grid.page_max_xz(1, 1), glm::vec2(2.0f * PAGE, 2.0f * PAGE));
    // A page's max is its neighbour's min — no gap, no overlap.
    EXPECT_EQ(grid.page_max_xz(0, 0).x, grid.page_min_xz(1, 0).x);
    EXPECT_EQ(grid.world_size_x(), 2.0f * PAGE);
    EXPECT_EQ(grid.world_size_z(), 2.0f * PAGE);
}

// ============================================================================
// (2) Cross-page sampling — the seam
// ============================================================================

TEST(HeightmapGrid, BothPagesAgreeExactlyOnTheLineTheyShare) {
    const HeightmapGrid grid = make_2x2();
    const Heightmap* left = grid.page(0, 0);
    const Heightmap* right = grid.page(1, 0);
    ASSERT_NE(left, nullptr);
    ASSERT_NE(right, nullptr);

    // Both pages store the seam line; with one shared quantization range they
    // store the same 16-bit values, so the two answers are EQUAL, not close.
    for (int i = 0; i <= 40; ++i) {
        const float z = static_cast<float>(i) * (PAGE / 40.0f);
        EXPECT_EQ(left->get_height_world(PAGE, z), right->get_height_world(PAGE, z)) << "z=" << z;
        // And the grid's own answer is one of them (it addresses the right).
        EXPECT_EQ(grid.get_height_world(PAGE, z), right->get_height_world(PAGE, z)) << "z=" << z;
    }
}

TEST(HeightmapGrid, SamplingSweepsAcrossASeamWithoutAStep) {
    const HeightmapGrid grid = make_2x2();
    // Quantization step of the shared range, the floor on any disagreement.
    const float quant = (RANGE.max - RANGE.min) / 65535.0f;

    float worst_step = 0.0f;
    float worst_error = 0.0f;
    for (int i = 0; i <= 200; ++i) {
        const float z = 40.0f + static_cast<float>(i) * 4.5f; // both sides of the x seam
        const float below = grid.get_height_world(PAGE - 0.5f, z);
        const float above = grid.get_height_world(PAGE + 0.5f, z);
        worst_step = std::max(worst_step, std::abs(above - below));
        worst_error = std::max(worst_error, std::abs(grid.get_height_world(PAGE, z) - analytic(PAGE, z)));

        // Same test across the z seam.
        const float x = 40.0f + static_cast<float>(i) * 4.5f;
        const float zb = grid.get_height_world(x, PAGE - 0.5f);
        const float za = grid.get_height_world(x, PAGE + 0.5f);
        worst_step = std::max(worst_step, std::abs(za - zb));
    }
    // One world unit of travel over a surface whose gradient is ~0.03 —
    // anything approaching the seam-step failure mode (a whole re-quantization
    // of a 120-unit range) would be orders of magnitude larger than this.
    EXPECT_LT(worst_step, 0.2f) << "the seam steps";
    EXPECT_LT(worst_error, 0.2f) << "seam samples do not reconstruct the authored surface";
    EXPECT_GT(quant, 0.0f);
}

TEST(HeightmapGrid, NormalsAtASeamUseBothPagesAndStayUnitLength) {
    const HeightmapGrid grid = make_2x2();
    // The stencil is one cell wide, so exactly on the seam two of its four
    // taps land in the neighbour. A per-page normal would clamp them against
    // an interior edge and tilt; this must not.
    for (int i = 1; i < 20; ++i) {
        const float z = static_cast<float>(i) * (PAGE / 20.0f);
        const glm::vec3 n = grid.get_normal_world(PAGE, z);
        EXPECT_NEAR(glm::length(n), 1.0f, 1e-4f) << "z=" << z;
        EXPECT_GT(n.y, 0.9f) << "z=" << z;
        // Against the neighbourhood's own normal a few units away: a clamped
        // stencil shows up as a discontinuity here.
        const glm::vec3 near_left = grid.get_normal_world(PAGE - 4.0f, z);
        const glm::vec3 near_right = grid.get_normal_world(PAGE + 4.0f, z);
        EXPECT_GT(glm::dot(near_left, near_right), 0.999f) << "z=" << z;
    }
}

TEST(HeightmapGrid, OutsideTheGridClampsToTheGridEdgeNotThePageEdge) {
    const HeightmapGrid grid = make_2x2();
    // Far off the +x side at a z inside page row 1: the answer must come from
    // page (1,1)'s own edge column, i.e. the GRID edge.
    const float edge = grid.get_height_world(2.0f * PAGE, 1500.0f);
    EXPECT_EQ(grid.get_height_world(10.0f * PAGE, 1500.0f), edge);
    const Heightmap* corner_page = grid.page(1, 1);
    ASSERT_NE(corner_page, nullptr);
    EXPECT_EQ(edge, corner_page->get_height_world(2.0f * PAGE, 1500.0f));
}

// ============================================================================
// (3) Validation — the refusals
// ============================================================================

TEST(HeightmapGrid, RejectsPagesPackedAgainstADifferentRange) {
    HeightmapGrid grid = HeightmapGrid::create(2, 1, PAGE, 0.0f, 0.0f, RANGE);
    ASSERT_TRUE(grid.set_page(0, 0, page_samples(0.0f, 0.0f), RES));

    // The trap, exactly as a host walks into it: the SCANNING from_floats
    // overload derives the range from this page's own extremes, so its 16-bit
    // encoding disagrees with its neighbour's on the line they share.
    const std::vector<float> h = page_samples(PAGE, 0.0f);
    grid.pages[1] = Heightmap::from_floats(h, RES, WorldBounds{PAGE, 0.0f, PAGE});
    ASSERT_TRUE(grid.pages[1].valid());
    ASSERT_NE(grid.pages[1].min_height, RANGE.min) << "the fixture no longer reproduces the trap";

    const auto check = grid.validate();
    ASSERT_TRUE(check.is_err());
    EXPECT_NE(check.error().find("range"), std::string::npos) << check.error();
    EXPECT_NE(check.error().find("(1,0)"), std::string::npos) << check.error();
    EXPECT_FALSE(grid.valid());

    // And the same data through set_page — which packs against the grid's
    // range — is accepted.
    ASSERT_TRUE(grid.set_page(1, 0, h, RES));
    EXPECT_TRUE(grid.valid()) << (grid.validate().is_err() ? grid.validate().error() : std::string{});
}

TEST(HeightmapGrid, RejectsAMismatchedRangeEvenByOneUlp) {
    // No tolerance, deliberately: a range is COPIED between pages, never
    // recomputed, so any difference at all is an authoring mistake — and a
    // one-ulp difference still re-quantizes every sample on the seam.
    HeightmapGrid grid = HeightmapGrid::create(2, 1, PAGE, 0.0f, 0.0f, RANGE);
    ASSERT_TRUE(grid.set_page(0, 0, page_samples(0.0f, 0.0f), RES));
    ASSERT_TRUE(grid.set_page(1, 0, page_samples(PAGE, 0.0f), RES));
    ASSERT_TRUE(grid.valid());

    grid.pages[1].max_height = std::nextafter(RANGE.max, 1000.0f);
    EXPECT_FALSE(grid.valid());
}

TEST(HeightmapGrid, RejectsAResolutionMismatchBetweenSiblings) {
    HeightmapGrid grid = HeightmapGrid::create(2, 1, PAGE, 0.0f, 0.0f, RANGE);
    ASSERT_TRUE(grid.set_page(0, 0, page_samples(0.0f, 0.0f), RES));

    constexpr uint32_t OTHER_RES = 9;
    std::vector<float> coarse(static_cast<std::size_t>(OTHER_RES) * OTHER_RES, 3.0f);
    ASSERT_TRUE(grid.set_page(1, 0, coarse, OTHER_RES));

    const auto check = grid.validate();
    ASSERT_TRUE(check.is_err());
    EXPECT_NE(check.error().find("resolution"), std::string::npos) << check.error();
}

TEST(HeightmapGrid, RejectsAPageThatDoesNotSitWhereTheGridAddressesIt) {
    HeightmapGrid grid = HeightmapGrid::create(2, 1, PAGE, 0.0f, 0.0f, RANGE);
    ASSERT_TRUE(grid.set_page(0, 0, page_samples(0.0f, 0.0f), RES));
    ASSERT_TRUE(grid.set_page(1, 0, page_samples(PAGE, 0.0f), RES));
    ASSERT_TRUE(grid.valid());

    // Half a page off — the classic off-by-one in a page loop.
    grid.pages[1].world_origin_x = PAGE * 1.5f;
    const auto check = grid.validate();
    ASSERT_TRUE(check.is_err());
    EXPECT_NE(check.error().find("addresses"), std::string::npos) << check.error();

    // A float ulp is NOT an error: geometry gets slack where ranges do not.
    grid.pages[1].world_origin_x = std::nextafter(PAGE, 2000.0f);
    EXPECT_TRUE(grid.valid());
}

TEST(HeightmapGrid, RejectsAPageWhoseWorldSizeIsNotThePageSize) {
    HeightmapGrid grid = HeightmapGrid::create(1, 1, PAGE, 0.0f, 0.0f, RANGE);
    ASSERT_TRUE(grid.set_page(0, 0, page_samples(0.0f, 0.0f), RES));
    ASSERT_TRUE(grid.valid());
    grid.pages[0].world_size = PAGE * 0.5f;
    EXPECT_FALSE(grid.valid());
}

TEST(HeightmapGrid, RejectsDegenerateShapes) {
    EXPECT_FALSE(HeightmapGrid{}.valid()); // 0x0
    EXPECT_FALSE(HeightmapGrid::create(0, 2, PAGE, 0.0f, 0.0f, RANGE).valid());
    EXPECT_FALSE(HeightmapGrid::create(2, 2, 0.0f, 0.0f, 0.0f, RANGE).valid());

    HeightmapGrid short_vector = HeightmapGrid::create(2, 2, PAGE, 0.0f, 0.0f, RANGE);
    short_vector.pages.pop_back();
    const auto check = short_vector.validate();
    ASSERT_TRUE(check.is_err());
    EXPECT_NE(check.error().find("3 pages"), std::string::npos) << check.error();
}

TEST(HeightmapGrid, SetPageRefusesBadIndicesAndBadSampleCounts) {
    HeightmapGrid grid = HeightmapGrid::create(2, 2, PAGE, 0.0f, 0.0f, RANGE);
    const std::vector<float> h = page_samples(0.0f, 0.0f);
    EXPECT_FALSE(grid.set_page(2, 0, h, RES));
    EXPECT_FALSE(grid.set_page(0, 2, h, RES));
    EXPECT_EQ(grid.resident_pages(), 0u);

    const std::vector<float> too_few(static_cast<std::size_t>(RES) * RES - 1, 0.0f);
    EXPECT_FALSE(grid.set_page(0, 0, too_few, RES));
    EXPECT_FALSE(grid.set_page(0, 0, h, 1)); // below the 2x2 minimum
    EXPECT_EQ(grid.resident_pages(), 0u);
}

TEST(HeightmapGrid, SetPageDerivesPageGeometryFromTheGrid) {
    // The whole point of set_page: a host never computes a page origin, so it
    // cannot get one wrong, and it never chooses a range, so it cannot walk
    // into the seam trap.
    HeightmapGrid grid = HeightmapGrid::create(3, 2, PAGE, -500.0f, 250.0f, RANGE);
    ASSERT_TRUE(grid.set_page(2, 1, page_samples(-500.0f + 2.0f * PAGE, 250.0f + PAGE), RES));
    const Heightmap* page = grid.page(2, 1);
    ASSERT_NE(page, nullptr);
    EXPECT_EQ(page->world_origin_x, -500.0f + 2.0f * PAGE);
    EXPECT_EQ(page->world_origin_z, 250.0f + PAGE);
    EXPECT_EQ(page->world_size, PAGE);
    EXPECT_EQ(page->min_height, RANGE.min);
    EXPECT_EQ(page->max_height, RANGE.max);
    EXPECT_TRUE(grid.valid());
}
