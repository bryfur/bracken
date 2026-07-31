#include <gtest/gtest.h>

#include "bracken/core/handle.hpp"

#include <string>
#include <unordered_set>

using namespace bracken::core;

namespace {
struct MeshTag {};
struct TexTag {};
} // namespace

TEST(Handle, InvalidIsNotValid) {
    auto h = Handle<MeshTag>::invalid();
    EXPECT_FALSE(h.is_valid());
}

TEST(Handle, DefaultConstructedIsInvalid) {
    Handle<MeshTag> h{};
    EXPECT_FALSE(h.is_valid());
    EXPECT_EQ(h, Handle<MeshTag>::invalid());

    Handle<MeshTag> plain; // default-init, not value-init
    EXPECT_FALSE(plain.is_valid());
    EXPECT_EQ(plain, Handle<MeshTag>::invalid());
}

TEST(Handle, EqualityCompares) {
    Handle<MeshTag> a{1, 2};
    Handle<MeshTag> b{1, 2};
    Handle<MeshTag> c{1, 3};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(Handle, HashSpecializationCompiles) {
    std::unordered_set<Handle<MeshTag>> set;
    set.insert(Handle<MeshTag>{1, 1});
    set.insert(Handle<MeshTag>{2, 1});
    set.insert(Handle<MeshTag>{1, 1});
    EXPECT_EQ(set.size(), 2u);
}

TEST(Handle, HashDistinguishesIndexAndGeneration) {
    std::hash<Handle<MeshTag>> hasher;
    // Computed via uint64_t: (generation << 32) | index — no size_t-width shift.
    EXPECT_NE(hasher(Handle<MeshTag>{1, 1}), hasher(Handle<MeshTag>{1, 2}));
    EXPECT_NE(hasher(Handle<MeshTag>{1, 1}), hasher(Handle<MeshTag>{2, 1}));
    if constexpr (sizeof(size_t) >= 8) {
        EXPECT_EQ(hasher(Handle<MeshTag>{3, 5}), (uint64_t{5} << 32) | 3u);
    }
}

TEST(HandlePool, AcquireReturnsLiveHandle) {
    HandlePool<MeshTag, std::string> pool;
    auto h = pool.acquire("hello");
    EXPECT_TRUE(h.is_valid());
    auto* ptr = pool.get(h);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(*ptr, "hello");
}

TEST(HandlePool, ReleaseInvalidatesOldHandle) {
    HandlePool<MeshTag, std::string> pool;
    auto h = pool.acquire("a");
    EXPECT_TRUE(pool.release(h));
    EXPECT_EQ(pool.get(h), nullptr);
}

TEST(HandlePool, GenerationalReuseDetectsStale) {
    HandlePool<MeshTag, int> pool;
    auto h1 = pool.acquire(10);
    auto idx1 = h1.index;
    EXPECT_TRUE(pool.release(h1));

    auto h2 = pool.acquire(20);
    EXPECT_EQ(h2.index, idx1);               // index recycled
    EXPECT_NE(h2.generation, h1.generation); // generation bumped

    EXPECT_EQ(pool.get(h1), nullptr); // stale handle returns null
    auto* live = pool.get(h2);
    ASSERT_NE(live, nullptr);
    EXPECT_EQ(*live, 20);
}

TEST(HandlePool, MultipleAcquireAndContains) {
    HandlePool<MeshTag, int> pool;
    auto a = pool.acquire(1);
    auto b = pool.acquire(2);
    auto c = pool.acquire(3);
    EXPECT_TRUE(pool.contains(a));
    EXPECT_TRUE(pool.contains(b));
    EXPECT_TRUE(pool.contains(c));
    EXPECT_EQ(pool.size(), 3u);
    EXPECT_TRUE(pool.release(b));
    EXPECT_FALSE(pool.contains(b));
    EXPECT_EQ(pool.size(), 2u);
}

TEST(HandlePool, DoubleReleaseFails) {
    HandlePool<MeshTag, int> pool;
    auto h = pool.acquire(7);
    EXPECT_TRUE(pool.release(h));
    EXPECT_FALSE(pool.release(h));
}

TEST(HandlePool, ClearInvalidatesRetainedHandles) {
    HandlePool<MeshTag, int> pool;
    auto h1 = pool.acquire(10);
    auto h2 = pool.acquire(20);
    pool.clear();

    EXPECT_EQ(pool.size(), 0u);
    EXPECT_EQ(pool.get(h1), nullptr);
    EXPECT_EQ(pool.get(h2), nullptr);
    EXPECT_FALSE(pool.release(h1));

    // Post-clear acquisitions reuse slots, but stale pre-clear handles must
    // not alias them (generations survive the clear).
    auto h3 = pool.acquire(30);
    auto h4 = pool.acquire(40);
    EXPECT_EQ(pool.get(h1), nullptr);
    EXPECT_EQ(pool.get(h2), nullptr);
    ASSERT_NE(pool.get(h3), nullptr);
    ASSERT_NE(pool.get(h4), nullptr);
    EXPECT_EQ(*pool.get(h3), 30);
    EXPECT_EQ(*pool.get(h4), 40);
    EXPECT_EQ(pool.size(), 2u);
}

TEST(HandlePool, ClearIsIdempotentAndSkipsFreeSlots) {
    HandlePool<MeshTag, int> pool;
    auto a = pool.acquire(1);
    auto b = pool.acquire(2);
    EXPECT_TRUE(pool.release(a)); // one slot already free before clear
    pool.clear();
    pool.clear(); // second clear must not bump generations of free slots again

    EXPECT_EQ(pool.size(), 0u);
    auto c = pool.acquire(3);
    EXPECT_EQ(pool.get(a), nullptr);
    EXPECT_EQ(pool.get(b), nullptr);
    ASSERT_NE(pool.get(c), nullptr);
    EXPECT_EQ(*pool.get(c), 3);
}

TEST(HandlePool, TagsAreTypeSafe) {
    Handle<MeshTag> mh{0, 1};
    Handle<TexTag> th{0, 1};
    static_assert(!std::is_same_v<decltype(mh), decltype(th)>);
}
