#include "bracken/scene/bone_palette_table.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

using bracken::scene::BonePaletteTable;

TEST(BonePaletteTable, FindOnEmptyReturnsNull) {
    BonePaletteTable table(16);
    int dummy = 0;
    EXPECT_EQ(table.find(&dummy), nullptr);
    EXPECT_EQ(table.size(), 0u);
}

TEST(BonePaletteTable, InsertThenFindRoundTrips) {
    BonePaletteTable table(16);
    int a = 0;
    int b = 0;

    ASSERT_NE(table.insert(&a, 0, 60), nullptr);
    ASSERT_NE(table.insert(&b, 60, 30), nullptr);
    EXPECT_EQ(table.size(), 2u);

    auto* ea = table.find(&a);
    ASSERT_NE(ea, nullptr);
    EXPECT_EQ(ea->offset, 0u);
    EXPECT_EQ(ea->joint_count, 60u);

    auto* eb = table.find(&b);
    ASSERT_NE(eb, nullptr);
    EXPECT_EQ(eb->offset, 60u);
    EXPECT_EQ(eb->joint_count, 30u);
}

TEST(BonePaletteTable, FoundEntryIsMutable) {
    // pack_bone_palettes repoints an entry when a palette is repacked with
    // more joints; the mutation must persist across a later find().
    BonePaletteTable table(16);
    int a = 0;
    table.insert(&a, 0, 10);

    auto* entry = table.find(&a);
    ASSERT_NE(entry, nullptr);
    entry->offset = 128;
    entry->joint_count = 64;

    auto* again = table.find(&a);
    ASSERT_NE(again, nullptr);
    EXPECT_EQ(again->offset, 128u);
    EXPECT_EQ(again->joint_count, 64u);
}

TEST(BonePaletteTable, ClearInvalidatesAllEntriesInO1) {
    BonePaletteTable table(16);
    int a = 0;
    int b = 0;
    table.insert(&a, 0, 1);
    table.insert(&b, 1, 1);

    table.clear();
    EXPECT_EQ(table.size(), 0u);
    EXPECT_EQ(table.find(&a), nullptr);
    EXPECT_EQ(table.find(&b), nullptr);

    // Re-inserting after clear works and finds the new values.
    ASSERT_NE(table.insert(&a, 7, 3), nullptr);
    auto* entry = table.find(&a);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->offset, 7u);
    EXPECT_EQ(entry->joint_count, 3u);
}

TEST(BonePaletteTable, InsertBeyondCapReturnsNull) {
    BonePaletteTable table(4);
    std::vector<int> keys(5);
    for (int i = 0; i < 4; ++i) {
        ASSERT_NE(table.insert(&keys[i], static_cast<uint32_t>(i), 1), nullptr);
    }
    EXPECT_EQ(table.size(), 4u);
    EXPECT_EQ(table.insert(&keys[4], 4, 1), nullptr);
    EXPECT_EQ(table.size(), 4u);
}

TEST(BonePaletteTable, NullKeyIsRejected) {
    BonePaletteTable table(4);
    EXPECT_EQ(table.insert(nullptr, 0, 1), nullptr);
    EXPECT_EQ(table.size(), 0u);
}

TEST(BonePaletteTable, ManyCollidingKeysAllRetrievable) {
    // Fill to the cap with adjacent addresses (worst case for pointer hashes)
    // across many generations; every key must resolve to its own entry.
    constexpr uint32_t kMax = 1024;
    BonePaletteTable table(kMax);
    std::vector<uint64_t> storage(kMax);

    for (int gen = 0; gen < 3; ++gen) {
        table.clear();
        for (uint32_t i = 0; i < kMax; ++i) {
            EXPECT_EQ(table.find(&storage[i]), nullptr);
            ASSERT_NE(table.insert(&storage[i], i * 64, 64), nullptr);
        }
        EXPECT_EQ(table.size(), kMax);
        for (uint32_t i = 0; i < kMax; ++i) {
            auto* entry = table.find(&storage[i]);
            ASSERT_NE(entry, nullptr) << "key " << i << " missing in generation " << gen;
            EXPECT_EQ(entry->offset, i * 64);
        }
    }
}
