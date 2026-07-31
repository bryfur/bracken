#include "bracken/render_stats.hpp"

#include <array>
#include <gtest/gtest.h>

namespace bracken {
namespace {

FrameStatsPartial make_partial(uint32_t base) {
    FrameStatsPartial p;
    p.draw_calls = base + 1;
    p.triangle_count = base + 2;
    p.entities_rendered = base + 3;
    p.entities_distance_culled = base + 4;
    p.entities_frustum_culled = base + 5;
    return p;
}

TEST(FrameStatsPartial, DefaultIsZero) {
    const FrameStatsPartial p;
    EXPECT_EQ(p.draw_calls, 0u);
    EXPECT_EQ(p.triangle_count, 0u);
    EXPECT_EQ(p.entities_rendered, 0u);
    EXPECT_EQ(p.entities_distance_culled, 0u);
    EXPECT_EQ(p.entities_frustum_culled, 0u);
}

TEST(FrameStatsPartial, MergeIntoAccumulates) {
    RenderStats stats;
    stats.draw_calls = 10;
    stats.triangle_count = 20;
    stats.entities_rendered = 30;
    stats.entities_distance_culled = 40;
    stats.entities_frustum_culled = 50;
    stats.jobs_pending = 7;
    stats.pass_times_ms.emplace_back("ShadowPass", 1.5f);

    make_partial(100).merge_into(stats);

    EXPECT_EQ(stats.draw_calls, 111u);
    EXPECT_EQ(stats.triangle_count, 122u);
    EXPECT_EQ(stats.entities_rendered, 133u);
    EXPECT_EQ(stats.entities_distance_culled, 144u);
    EXPECT_EQ(stats.entities_frustum_culled, 155u);
    // Frame-level fields are not part of the partial and must be untouched.
    EXPECT_EQ(stats.jobs_pending, 7u);
    ASSERT_EQ(stats.pass_times_ms.size(), 1u);
    EXPECT_EQ(stats.pass_times_ms[0].first, "ShadowPass");
}

TEST(FrameStatsPartial, MergeOfZeroPartialIsNoOp) {
    RenderStats stats;
    stats.draw_calls = 3;
    stats.entities_frustum_culled = 9;

    FrameStatsPartial{}.merge_into(stats);

    EXPECT_EQ(stats.draw_calls, 3u);
    EXPECT_EQ(stats.triangle_count, 0u);
    EXPECT_EQ(stats.entities_frustum_culled, 9u);
}

TEST(FrameStatsPartial, PlusEqualsFoldsChunks) {
    FrameStatsPartial total;
    total += make_partial(0);
    total += make_partial(1000);

    EXPECT_EQ(total.draw_calls, 1002u);              // (0+1) + (1000+1)
    EXPECT_EQ(total.triangle_count, 1004u);          // (0+2) + (1000+2)
    EXPECT_EQ(total.entities_rendered, 1006u);       // (0+3) + (1000+3)
    EXPECT_EQ(total.entities_distance_culled, 1008u);// (0+4) + (1000+4)
    EXPECT_EQ(total.entities_frustum_culled, 1010u); // (0+5) + (1000+5)
}

// The Phase 3 discipline: per-chunk partials merged in chunk order at the
// join must equal one serial partial over the whole range.
TEST(FrameStatsPartial, ChunkedMergeMatchesSerialAccumulation) {
    constexpr uint32_t k_items = 97; // deliberately not a multiple of chunks
    constexpr uint32_t k_chunks = 4;

    // Serial reference: one partial over all items.
    FrameStatsPartial serial;
    for (uint32_t i = 0; i < k_items; ++i) {
        serial.draw_calls++;
        serial.triangle_count += i;
        if (i % 3 == 0) {
            serial.entities_rendered++;
        } else if (i % 3 == 1) {
            serial.entities_distance_culled++;
        } else {
            serial.entities_frustum_culled++;
        }
    }

    // Chunked: one partial per chunk, merged in chunk-index order.
    std::array<FrameStatsPartial, k_chunks> chunks{};
    for (uint32_t i = 0; i < k_items; ++i) {
        FrameStatsPartial& p = chunks[i * k_chunks / k_items];
        p.draw_calls++;
        p.triangle_count += i;
        if (i % 3 == 0) {
            p.entities_rendered++;
        } else if (i % 3 == 1) {
            p.entities_distance_culled++;
        } else {
            p.entities_frustum_culled++;
        }
    }

    RenderStats from_serial;
    serial.merge_into(from_serial);
    RenderStats from_chunks;
    for (const FrameStatsPartial& p : chunks) {
        p.merge_into(from_chunks);
    }

    EXPECT_EQ(from_chunks.draw_calls, from_serial.draw_calls);
    EXPECT_EQ(from_chunks.triangle_count, from_serial.triangle_count);
    EXPECT_EQ(from_chunks.entities_rendered, from_serial.entities_rendered);
    EXPECT_EQ(from_chunks.entities_distance_culled, from_serial.entities_distance_culled);
    EXPECT_EQ(from_chunks.entities_frustum_culled, from_serial.entities_frustum_culled);
}

} // namespace
} // namespace bracken
