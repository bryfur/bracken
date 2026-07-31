#include <gtest/gtest.h>

#include "bracken/memory/arena.hpp"

#include <cstdint>
#include <thread>
#include <vector>

using namespace bracken::memory;

TEST(Arena, AllocateRespectsAlignment) {
    Arena a(4096);
    void* p1 = a.allocate(1, 1);
    void* p16 = a.allocate(8, 16);
    void* p64 = a.allocate(4, 64);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p16, nullptr);
    ASSERT_NE(p64, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p16) % 16u, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p64) % 64u, 0u);
}

TEST(Arena, ResetRewindsUsage) {
    Arena a(1024);
    ASSERT_NE(a.allocate(256, 8), nullptr);
    EXPECT_GE(a.bytes_used(), 256u);
    a.reset();
    EXPECT_EQ(a.bytes_used(), 0u);
}

TEST(Arena, GrowsAcrossMultipleChunks) {
    Arena a(64);
    std::vector<void*> ptrs;
    for (int i = 0; i < 32; ++i) {
        void* p = a.allocate(16, 8);
        ASSERT_NE(p, nullptr);
        ptrs.push_back(p);
    }
    EXPECT_GE(a.bytes_capacity(), 64u);
    EXPECT_GE(a.bytes_used(), 32u * 16u);
}

TEST(Arena, FixedBufferDoesNotOverrun) {
    alignas(16) std::byte buf[128];
    Arena a(buf, sizeof(buf));
    void* p = a.allocate(64, 8);
    ASSERT_NE(p, nullptr);
    void* q = a.allocate(64, 8);
    ASSERT_NE(q, nullptr);
    void* r = a.allocate(1, 1);
    EXPECT_EQ(r, nullptr); // out of fixed budget
}

TEST(Arena, CreateInvokesConstructorAndDestructor) {
    static int alive = 0;
    struct Tracked {
        Tracked() { ++alive; }
        ~Tracked() { --alive; }
        int v = 7;
    };
    {
        Arena a(1024);
        auto* t = a.create<Tracked>();
        ASSERT_NE(t, nullptr);
        EXPECT_EQ(t->v, 7);
        EXPECT_EQ(alive, 1);
    }
    EXPECT_EQ(alive, 0);
}

TEST(Arena, ResetRunsDestructors) {
    static int alive = 0;
    struct Tracked {
        Tracked() { ++alive; }
        ~Tracked() { --alive; }
    };
    Arena a(1024);
    ASSERT_NE(a.create<Tracked>(), nullptr);
    ASSERT_NE(a.create<Tracked>(), nullptr);
    EXPECT_EQ(alive, 2);
    a.reset();
    EXPECT_EQ(alive, 0);
}

TEST(Arena, ZeroSizeAllocationSucceedsWithNonNullPointer) {
    Arena a(1024);
    void* p = a.allocate(0, 8);
    EXPECT_NE(p, nullptr); // nullptr uniquely signals failure
    EXPECT_EQ(a.bytes_used(), 0u);

    // Also succeeds on an arena that has never grown a chunk...
    Arena fresh;
    EXPECT_NE(fresh.allocate(0, 16), nullptr);
    EXPECT_EQ(fresh.bytes_capacity(), 0u);

    // ...and on an exhausted fixed-buffer arena, where size>0 fails.
    alignas(16) std::byte buf[32];
    Arena fixed(buf, sizeof(buf));
    ASSERT_NE(fixed.allocate(32, 1), nullptr);
    EXPECT_EQ(fixed.allocate(1, 1), nullptr);  // genuine exhaustion
    EXPECT_NE(fixed.allocate(0, 1), nullptr);  // zero-size still succeeds
}

TEST(ArenaDeathTest, NonPowerOfTwoAlignmentAsserts) {
    Arena a(1024);
    // Debug builds must reject a non-power-of-two alignment (e.g. a
    // stride-derived 12); release builds skip the check.
    EXPECT_DEBUG_DEATH((void)a.allocate(16, 12), "alignment must be a power of two");
}

TEST(Arena, ZeroAlignmentIsRemappedNotRejected) {
    Arena a(1024);
    void* p = a.allocate(8, 0);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % alignof(std::max_align_t), 0u);
}

// --- Owner-thread enforcement (debug builds; no-op in release) ---
// EXPECT_DEBUG_DEATH runs the statement without expecting death under NDEBUG,
// so these also verify the release-build no-op path compiles and passes.

TEST(ArenaOwnerDeathTest, AllocateFromForeignThreadAsserts) {
    Arena a(1024);
    ASSERT_NE(a.allocate(8, 8), nullptr); // binds the arena to this thread
    EXPECT_DEBUG_DEATH(
        {
            std::thread t([&] { (void)a.allocate(8, 8); });
            t.join();
        },
        "non-owner thread");
}

TEST(ArenaOwnerDeathTest, ResetFromForeignThreadAsserts) {
    Arena a(1024);
    ASSERT_NE(a.allocate(8, 8), nullptr); // binds the arena to this thread
    EXPECT_DEBUG_DEATH(
        {
            std::thread t([&] { a.reset(); });
            t.join();
        },
        "non-owner thread");
}

TEST(ArenaOwnerDeathTest, FixedBufferArenaEnforcesOwnerToo) {
    alignas(16) std::byte buf[128];
    Arena a(buf, sizeof(buf));
    ASSERT_NE(a.allocate(8, 8), nullptr); // binds the arena to this thread
    EXPECT_DEBUG_DEATH(
        {
            std::thread t([&] { (void)a.allocate(8, 8); });
            t.join();
        },
        "non-owner thread");
}

TEST(ArenaOwnerDeathTest, BindOwnerBlocksOtherThreads) {
    Arena a(1024);
    // Explicit bind without allocating: a worker claims the arena up front,
    // then an allocation from this (non-owner) thread must assert. The claim
    // thread is spawned *inside* the death statement so gtest's fast-style
    // fork() happens while the parent process is still single-threaded
    // (forking after spawning a thread crashes in the objc runtime on macOS).
    EXPECT_DEBUG_DEATH(
        {
            std::thread claim([&] { a.bind_owner(); });
            claim.join();
            (void)a.allocate(8, 8);
        },
        "non-owner thread");
}

TEST(Arena, ResetReopensOwnerBinding) {
    Arena a(1024);
    ASSERT_NE(a.allocate(8, 8), nullptr); // owner: this thread
    a.reset();                            // re-opens the binding
    // A different thread may now become the new owner — must not assert.
    bool worker_alloc_ok = false;
    std::thread t([&] { worker_alloc_ok = a.allocate(16, 8) != nullptr; });
    t.join();
    EXPECT_TRUE(worker_alloc_ok);
}

TEST(Arena, RebindOwnerTransfersOwnershipExplicitly) {
    Arena a(1024);
    // A worker binds by allocating first.
    std::thread t([&] { (void)a.allocate(8, 8); });
    t.join();
    // Explicit handoff to this thread; allocate/reset must not assert.
    a.rebind_owner();
    EXPECT_NE(a.allocate(8, 8), nullptr);
    a.reset();
}

TEST(Arena, BindOwnerIsIdempotentOnOwnerThread) {
    Arena a(1024);
    a.bind_owner();
    a.bind_owner(); // same thread: must not assert
    EXPECT_NE(a.allocate(8, 8), nullptr);
}

TEST(Arena, MoveTransfersOwnership) {
    Arena a(1024);
    ASSERT_NE(a.allocate(64, 8), nullptr);
    Arena b = std::move(a);
    EXPECT_GE(b.bytes_used(), 64u);
    // Intentional use of moved-from `a` to verify the post-move contract:
    // a moved-from Arena must still observe as empty (valid but empty state).
    // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
    EXPECT_EQ(a.bytes_used(), 0u);
}
