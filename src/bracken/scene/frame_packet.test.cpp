#include <gtest/gtest.h>

#include "bracken/scene/frame_builder.hpp"
#include "bracken/scene/frame_packet.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

using namespace bracken::scene;

namespace {

struct Range {
    std::uintptr_t begin;
    std::uintptr_t end;
};

// Sort by start address and require strictly non-overlapping ranges.
void expect_disjoint(std::vector<Range>& ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) { return a.begin < b.begin; });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        EXPECT_LE(ranges[i - 1].end, ranges[i].begin) << "block " << i << " overlaps its predecessor";
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Lifecycle state machine
// ---------------------------------------------------------------------------

TEST(FramePacket, FreshPacketIsRetired) {
    FramePacket p;
    EXPECT_EQ(p.state(), FramePacket::State::Retired);
}

TEST(FramePacket, BeginFillTransitionsToFilling) {
    FramePacket p;
    p.begin_fill();
    EXPECT_EQ(p.state(), FramePacket::State::Filling);
}

TEST(FramePacket, ResetFromFillingReturnsToRetired) {
    FramePacket p;
    p.begin_fill();
    (void)p.reserve_block(64, 16);
    p.reset();
    EXPECT_EQ(p.state(), FramePacket::State::Retired);
    EXPECT_EQ(p.bytes_used(), 0u);
}

TEST(FramePacket, FullLifecycleFillSealConsumeRetire) {
    FramePacket p;
    FrameBuilder fb(p); // Retired -> Filling
    EXPECT_EQ(p.state(), FramePacket::State::Filling);
    fb.draw_model(1, glm::mat4(1.0f));
    fb.seal(); // Filling -> Sealed
    EXPECT_EQ(p.state(), FramePacket::State::Sealed);
    EXPECT_EQ(p.model_commands().size(), 1u); // reads legal in Sealed
    p.begin_consume(); // Sealed -> Consuming
    EXPECT_EQ(p.state(), FramePacket::State::Consuming);
    EXPECT_EQ(p.model_commands().size(), 1u); // and in Consuming
    p.reset(); // Consuming -> Retired
    EXPECT_EQ(p.state(), FramePacket::State::Retired);
}

// ---------------------------------------------------------------------------
// reserve_block: alignment + disjointness
// ---------------------------------------------------------------------------

TEST(FramePacket, ReserveBlockHonorsAlignment) {
    FramePacket p;
    p.begin_fill();
    const std::size_t aligns[] = {1, 2, 4, 8, 16, 32, 64, 128};
    std::vector<Range> ranges;
    for (int pass = 0; pass < 8; ++pass) {
        for (std::size_t align : aligns) {
            const std::size_t size = 24 + static_cast<std::size_t>(pass) * 7;
            void* mem = p.reserve_block(size, align);
            ASSERT_NE(mem, nullptr);
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(mem) % align, 0u) << "align " << align;
            ranges.push_back({reinterpret_cast<std::uintptr_t>(mem), reinterpret_cast<std::uintptr_t>(mem) + size});
        }
    }
    expect_disjoint(ranges);
    p.reset();
}

TEST(FramePacket, ReserveBlockSpansSlabRefills) {
    // Small initial arena; reservations well past the 64 KB slab size force
    // multiple refills. All blocks must stay disjoint and writable.
    FramePacket p(4096);
    p.begin_fill();
    std::vector<Range> ranges;
    std::vector<uint32_t*> stamps;
    for (uint32_t i = 0; i < 300; ++i) {
        const std::size_t size = 1024;
        auto* mem = static_cast<uint32_t*>(p.reserve_block(size, alignof(uint32_t)));
        ASSERT_NE(mem, nullptr);
        *mem = i; // stamp; verified after all reservations
        stamps.push_back(mem);
        ranges.push_back({reinterpret_cast<std::uintptr_t>(mem), reinterpret_cast<std::uintptr_t>(mem) + size});
    }
    expect_disjoint(ranges);
    for (uint32_t i = 0; i < 300; ++i) {
        EXPECT_EQ(*stamps[i], i);
    }
    p.reset();
}

// ---------------------------------------------------------------------------
// reserve_block under contention: every reservation conserved
// ---------------------------------------------------------------------------

TEST(FramePacket, ReserveBlockContentionConservesAllBlocks) {
    constexpr unsigned THREADS = 8;
    constexpr unsigned PER_THREAD = 2000;

    FramePacket p;
    p.begin_fill();

    std::vector<std::vector<Range>> per_thread(THREADS);
    std::vector<std::vector<uint32_t*>> stamps(THREADS);
    std::vector<std::thread> workers;
    workers.reserve(THREADS);
    for (unsigned t = 0; t < THREADS; ++t) {
        workers.emplace_back([&p, &per_thread, &stamps, t] {
            per_thread[t].reserve(PER_THREAD);
            stamps[t].reserve(PER_THREAD);
            for (unsigned i = 0; i < PER_THREAD; ++i) {
                const std::size_t size = 4 + ((t * 31 + i) % 253); // 4..256 bytes
                auto* mem = static_cast<uint32_t*>(p.reserve_block(size, 4));
                if (mem == nullptr) {
                    continue; // recorded as missing by the count check below
                }
                *mem = t * PER_THREAD + i;
                per_thread[t].push_back(
                    {reinterpret_cast<std::uintptr_t>(mem), reinterpret_cast<std::uintptr_t>(mem) + size});
                stamps[t].push_back(mem);
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }

    // Conservation: every reservation succeeded...
    std::vector<Range> all;
    for (unsigned t = 0; t < THREADS; ++t) {
        EXPECT_EQ(per_thread[t].size(), PER_THREAD) << "thread " << t << " lost reservations";
        all.insert(all.end(), per_thread[t].begin(), per_thread[t].end());
    }
    // ...no two blocks overlap...
    expect_disjoint(all);
    // ...and no thread's writes were clobbered by another's.
    for (unsigned t = 0; t < THREADS; ++t) {
        for (unsigned i = 0; i < stamps[t].size(); ++i) {
            ASSERT_EQ(*stamps[t][i], t * PER_THREAD + i) << "thread " << t << " block " << i << " corrupted";
        }
    }
    p.reset();
}

// ---------------------------------------------------------------------------
// Reset / reuse
// ---------------------------------------------------------------------------

TEST(FramePacket, ResetReclaimsMemoryAndCapacityStabilizes) {
    FramePacket p;
    std::size_t cap_after_cycle2 = 0;
    for (int cycle = 0; cycle < 4; ++cycle) {
        p.begin_fill();
        for (int i = 0; i < 100; ++i) {
            (void)p.reserve_block(1024, 16);
        }
        const std::size_t cap = p.bytes_capacity();
        if (cycle == 2) {
            cap_after_cycle2 = cap;
        } else if (cycle == 3) {
            // Identical load after warm-up must not grow the arena further.
            EXPECT_EQ(cap, cap_after_cycle2);
        }
        p.reset();
        EXPECT_EQ(p.bytes_used(), 0u);
        EXPECT_EQ(p.state(), FramePacket::State::Retired);
    }
}

TEST(FramePacket, ResetClearsSealedContentAndSingularState) {
    FramePacket p;
    {
        FrameBuilder fb(p);
        fb.draw_model(7, glm::mat4(1.0f));
        fb.add_debug_line({0, 0, 0}, {1, 1, 1}, 0xFFFFFFFF);
        fb.set_draw_skybox(false);
        fb.set_draw_grass(false);
        fb.set_camera(CameraState{glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::vec3(1, 2, 3)});
        fb.set_environment(bracken::render::EnvironmentState{});
        fb.set_frame_timing(0.5f, 12.0, 99);
        fb.seal();
    }
    p.reset();

    // Refill with nothing: the sealed view must show pristine defaults.
    FrameBuilder fb2(p);
    fb2.seal();
    EXPECT_TRUE(p.model_commands().empty());
    EXPECT_TRUE(p.debug_lines().empty());
    EXPECT_TRUE(p.palettes().empty());
    EXPECT_EQ(p.palette_matrix_total(), 0u);
    EXPECT_TRUE(p.should_draw_skybox());
    EXPECT_TRUE(p.should_draw_ground());
    EXPECT_TRUE(p.should_draw_grass());
    EXPECT_FALSE(p.has_camera());
    EXPECT_FALSE(p.has_environment());
    EXPECT_EQ(p.dt(), 0.0f);
    EXPECT_EQ(p.time(), 0.0);
    EXPECT_EQ(p.frame_index(), 0u);
    EXPECT_TRUE(p.has_3d_content()); // toggles default true
}

// ---------------------------------------------------------------------------
// Debug contract asserts (state gates). Suite name ends in "DeathTest" so
// gtest schedules these before threaded tests.
// ---------------------------------------------------------------------------

#ifndef NDEBUG

TEST(FramePacketDeathTest, ReadBeforeSealAsserts) {
    EXPECT_DEATH(
        {
            FramePacket p;
            p.begin_fill();
            (void)p.model_commands();
        },
        "read before seal");
}

TEST(FramePacketDeathTest, ReadWhileRetiredAsserts) {
    EXPECT_DEATH(
        {
            FramePacket p;
            (void)p.has_3d_content();
        },
        "read before seal");
}

TEST(FramePacketDeathTest, ReserveBlockOutsideFillingAsserts) {
    EXPECT_DEATH(
        {
            FramePacket p; // Retired
            (void)p.reserve_block(16, 16);
        },
        "outside the Filling state");
}

TEST(FramePacketDeathTest, ReserveBlockAfterSealAsserts) {
    EXPECT_DEATH(
        {
            FramePacket p;
            FrameBuilder fb(p);
            fb.seal();
            (void)p.reserve_block(16, 16);
        },
        "outside the Filling state");
}

TEST(FramePacketDeathTest, BeginFillTwiceAsserts) {
    EXPECT_DEATH(
        {
            FramePacket p;
            p.begin_fill();
            p.begin_fill();
        },
        "not Retired");
}

TEST(FramePacketDeathTest, BeginConsumeUnsealedAsserts) {
    EXPECT_DEATH(
        {
            FramePacket p;
            p.begin_consume();
        },
        "not Sealed");
}

#endif // NDEBUG
