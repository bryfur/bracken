#include <gtest/gtest.h>

#include "bracken/memory/frame_allocator.hpp"

#include <cstdint>
#include <cstring>
#include <iterator>
#include <thread>

using namespace bracken::memory;

TEST(FrameAllocator, SwapFlipsThisAndLast) {
    FrameAllocator fa(1024);
    void* a = fa.this_frame().allocate(32, 8);
    ASSERT_NE(a, nullptr);
    EXPECT_GE(fa.this_frame().bytes_used(), 32u);

    fa.swap();
    EXPECT_GE(fa.last_frame().bytes_used(), 32u);
    EXPECT_EQ(fa.this_frame().bytes_used(), 0u);

    void* b = fa.this_frame().allocate(64, 8);
    ASSERT_NE(b, nullptr);
    EXPECT_GE(fa.this_frame().bytes_used(), 64u);
}

TEST(FrameAllocator, SwapResetsNewFrame) {
    FrameAllocator fa(1024);
    ASSERT_NE(fa.this_frame().allocate(100, 8), nullptr);
    fa.swap();
    ASSERT_NE(fa.this_frame().allocate(50, 8), nullptr);
    fa.swap();
    EXPECT_EQ(fa.this_frame().bytes_used(), 0u);
}

TEST(FrameAllocator, ContentSurvivesOneFrame) {
    FrameAllocator fa(1024);
    int* p = fa.this_frame().create<int>(123);
    ASSERT_NE(p, nullptr);
    *p = 555;
    fa.swap();
    EXPECT_EQ(*p, 555);
    fa.swap();
    // After two swaps, the original arena is reset; pointer is invalid by contract.
    // We only verified one-frame survival above.
}

TEST(FrameAllocator, PreviousFrameDataIntactUntilSecondSwap) {
    FrameAllocator fa(1024);
    constexpr std::size_t count = 64;
    auto* data = static_cast<std::uint8_t*>(fa.this_frame().allocate(count, 8));
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < count; ++i) {
        data[i] = static_cast<std::uint8_t>(i * 3 + 1);
    }

    fa.swap();

    // Churn the new frame's arena; it must not disturb last frame's data.
    auto* churn = static_cast<std::uint8_t*>(fa.this_frame().allocate(count, 8));
    ASSERT_NE(churn, nullptr);
    std::memset(churn, 0xFF, count);

    for (std::size_t i = 0; i < count; ++i) {
        EXPECT_EQ(data[i], static_cast<std::uint8_t>(i * 3 + 1)) << "byte " << i;
    }
    EXPECT_GE(fa.last_frame().bytes_used(), count);

    // Second swap recycles the original arena; its storage is reclaimed.
    fa.swap();
    EXPECT_EQ(fa.this_frame().bytes_used(), 0u);
}

TEST(FrameAllocator, EveryFrameStartsEmptyAcrossManySwaps) {
    FrameAllocator fa(256);
    for (int frame = 0; frame < 8; ++frame) {
        EXPECT_EQ(fa.this_frame().bytes_used(), 0u) << "frame " << frame;
        void* p = fa.this_frame().allocate(32, 8);
        ASSERT_NE(p, nullptr) << "frame " << frame;
        EXPECT_GE(fa.this_frame().bytes_used(), 32u) << "frame " << frame;
        fa.swap();
    }
}

TEST(FrameAllocator, MixedSizeAllocationsRespectAlignment) {
    FrameAllocator fa(4096);

    struct Request {
        std::size_t size;
        std::size_t align;
    };
    const Request requests[] = {
        {1, 1}, {3, 2}, {8, 4}, {13, 8}, {32, 16}, {7, 4}, {64, 32}, {128, 64}, {5, 8},
    };

    std::uint8_t* blocks[std::size(requests)] = {};
    for (std::size_t i = 0; i < std::size(requests); ++i) {
        void* p = fa.this_frame().allocate(requests[i].size, requests[i].align);
        ASSERT_NE(p, nullptr) << "allocation " << i;
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % requests[i].align, 0u) << "allocation " << i;
        blocks[i] = static_cast<std::uint8_t*>(p);
        std::memset(blocks[i], static_cast<int>(i + 1), requests[i].size);
    }

    // No allocation may overlap another: every block keeps its own fill byte.
    for (std::size_t i = 0; i < std::size(requests); ++i) {
        for (std::size_t b = 0; b < requests[i].size; ++b) {
            ASSERT_EQ(blocks[i][b], static_cast<std::uint8_t>(i + 1)) << "allocation " << i << " byte " << b;
        }
    }
}

TEST(FrameAllocator, AllocationLargerThanChunkGrowsArena) {
    FrameAllocator fa(64);
    void* small = fa.this_frame().allocate(16, 8);
    ASSERT_NE(small, nullptr);

    constexpr std::size_t big_size = 4096;
    auto* big = static_cast<std::uint8_t*>(fa.this_frame().allocate(big_size, 16));
    ASSERT_NE(big, nullptr);
    EXPECT_GE(fa.this_frame().bytes_capacity(), big_size);
    EXPECT_GE(fa.this_frame().bytes_used(), big_size + 16);

    // The whole oversized block must be usable.
    std::memset(big, 0x5A, big_size);
    EXPECT_EQ(big[0], 0x5A);
    EXPECT_EQ(big[big_size - 1], 0x5A);
}

TEST(FrameAllocator, SwapCycleReleasesGrowthChunks) {
    FrameAllocator fa(64);
    ASSERT_NE(fa.this_frame().allocate(16, 8), nullptr);
    ASSERT_NE(fa.this_frame().allocate(4096, 8), nullptr);
    const std::size_t grown_capacity = fa.this_frame().bytes_capacity();
    EXPECT_GT(grown_capacity, 64u);

    fa.swap(); // grown arena becomes last frame
    fa.swap(); // grown arena is reset and active again
    EXPECT_EQ(fa.this_frame().bytes_used(), 0u);
    // reset() keeps only the initial chunk; the growth chunks are freed.
    EXPECT_EQ(fa.this_frame().bytes_capacity(), 64u);
}

// --- Owner-thread enforcement ---
//
// The owner check is BRACKEN_CONTRACT (arena.cpp), which is live in release
// too, so this is a plain EXPECT_DEATH exactly like the equivalent cases in
// arena.test.cpp. It was EXPECT_DEBUG_DEATH — which under NDEBUG expects the
// statement NOT to die — and so failed the moment the tree was first built
// with optimization flags that actually define NDEBUG.

TEST(FrameAllocatorOwnerDeathTest, ForeignThreadAllocateAsserts) {
    FrameAllocator fa(1024);
    ASSERT_NE(fa.this_frame().allocate(8, 8), nullptr); // binds to this thread
    EXPECT_DEATH(
        {
            std::thread t([&] { (void)fa.this_frame().allocate(8, 8); });
            t.join();
        },
        "non-owner thread");
}

TEST(FrameAllocator, SwapReclaimsArenaFilledByAnotherThread) {
    FrameAllocator fa(1024);
    // A worker fills this frame's arena (becomes its owner).
    std::thread t([&] { (void)fa.this_frame().allocate(64, 8); });
    t.join();
    // swap() force-rebinds the newly-active arena before resetting it, so the
    // frame fence never trips the owner assert even when workers filled frames.
    fa.swap();
    fa.swap(); // resets the worker-owned arena — must not assert
    EXPECT_NE(fa.this_frame().allocate(16, 8), nullptr);
}

TEST(FrameAllocator, SwapReopensBindingForNextFillerThread) {
    FrameAllocator fa(1024);
    ASSERT_NE(fa.this_frame().allocate(8, 8), nullptr);
    fa.swap();
    // The freshly reset arena's binding is open: a worker may claim it.
    bool worker_alloc_ok = false;
    std::thread t([&] { worker_alloc_ok = fa.this_frame().allocate(16, 8) != nullptr; });
    t.join();
    EXPECT_TRUE(worker_alloc_ok);
}

TEST(FrameAllocator, AllocateAfterSwapReturnsFreshMemory) {
    FrameAllocator fa(1024);
    auto* first = static_cast<std::uint8_t*>(fa.this_frame().allocate(128, 16));
    ASSERT_NE(first, nullptr);
    std::memset(first, 0x11, 128);

    fa.swap();
    auto* second = static_cast<std::uint8_t*>(fa.this_frame().allocate(128, 16));
    ASSERT_NE(second, nullptr);
    // Distinct arenas: the fresh allocation cannot alias the live last-frame block.
    EXPECT_NE(static_cast<void*>(first), static_cast<void*>(second));
    std::memset(second, 0x22, 128);

    // Writing the fresh block must not touch the previous frame's block.
    EXPECT_EQ(first[0], 0x11);
    EXPECT_EQ(first[127], 0x11);
    EXPECT_GE(fa.this_frame().bytes_used(), 128u);
}
