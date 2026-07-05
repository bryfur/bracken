#include <gtest/gtest.h>

#include "bracken/render/text_renderer.hpp"

#include <string_view>

using namespace bracken::render;

// --- ShelfPacker (glyph atlas packing) ---------------------------------------

TEST(ShelfPacker, PacksLeftToRightOnOneShelf) {
    ShelfPacker packer(64, 64);
    int x = -1;
    int y = -1;
    ASSERT_TRUE(packer.pack(10, 20, x, y));
    EXPECT_EQ(x, 0);
    EXPECT_EQ(y, 0);
    ASSERT_TRUE(packer.pack(12, 18, x, y));
    EXPECT_EQ(x, 10);
    EXPECT_EQ(y, 0);
}

TEST(ShelfPacker, OpensNewShelfWhenRowIsFull) {
    ShelfPacker packer(32, 64);
    int x = 0;
    int y = 0;
    ASSERT_TRUE(packer.pack(20, 10, x, y));
    // 20 + 20 > 32 -> wraps to a new shelf below the tallest rect so far
    ASSERT_TRUE(packer.pack(20, 8, x, y));
    EXPECT_EQ(x, 0);
    EXPECT_EQ(y, 10);
}

TEST(ShelfPacker, ShelfHeightTracksTallestRect) {
    ShelfPacker packer(32, 64);
    int x = 0;
    int y = 0;
    ASSERT_TRUE(packer.pack(10, 5, x, y));
    ASSERT_TRUE(packer.pack(10, 12, x, y)); // grows current shelf to 12
    ASSERT_TRUE(packer.pack(20, 4, x, y));  // doesn't fit (10+10+20 > 32) -> new shelf
    EXPECT_EQ(x, 0);
    EXPECT_EQ(y, 12);
}

TEST(ShelfPacker, ReportsOverflow) {
    ShelfPacker packer(16, 16);
    int x = 0;
    int y = 0;
    ASSERT_TRUE(packer.pack(16, 8, x, y));
    ASSERT_TRUE(packer.pack(16, 8, x, y));
    EXPECT_FALSE(packer.pack(1, 1, x, y)); // both shelves exhausted
}

TEST(ShelfPacker, RejectsRectsLargerThanAtlas) {
    ShelfPacker packer(16, 16);
    int x = 0;
    int y = 0;
    EXPECT_FALSE(packer.pack(17, 1, x, y));
    EXPECT_FALSE(packer.pack(1, 17, x, y));
    EXPECT_FALSE(packer.pack(0, 5, x, y));
    EXPECT_FALSE(packer.pack(5, -1, x, y));
}

TEST(ShelfPacker, ResetReclaimsAllSpace) {
    ShelfPacker packer(16, 16);
    int x = 0;
    int y = 0;
    ASSERT_TRUE(packer.pack(16, 16, x, y));
    EXPECT_FALSE(packer.pack(1, 1, x, y));
    packer.reset();
    ASSERT_TRUE(packer.pack(16, 16, x, y));
    EXPECT_EQ(x, 0);
    EXPECT_EQ(y, 0);
}

// --- decode_utf8 --------------------------------------------------------------

TEST(DecodeUtf8, Ascii) {
    std::string_view text = "Ab";
    size_t i = 0;
    EXPECT_EQ(decode_utf8(text, i), static_cast<uint32_t>('A'));
    EXPECT_EQ(i, 1u);
    EXPECT_EQ(decode_utf8(text, i), static_cast<uint32_t>('b'));
    EXPECT_EQ(i, 2u);
}

TEST(DecodeUtf8, TwoByteSequence) {
    std::string_view text = "\xC3\xA9"; // U+00E9 é
    size_t i = 0;
    EXPECT_EQ(decode_utf8(text, i), 0xE9u);
    EXPECT_EQ(i, 2u);
}

TEST(DecodeUtf8, ThreeByteSequence) {
    std::string_view text = "\xE2\x82\xAC"; // U+20AC €
    size_t i = 0;
    EXPECT_EQ(decode_utf8(text, i), 0x20ACu);
    EXPECT_EQ(i, 3u);
}

TEST(DecodeUtf8, FourByteSequence) {
    std::string_view text = "\xF0\x9F\x98\x80"; // U+1F600 😀
    size_t i = 0;
    EXPECT_EQ(decode_utf8(text, i), 0x1F600u);
    EXPECT_EQ(i, 4u);
}

TEST(DecodeUtf8, StrayContinuationByteYieldsReplacement) {
    std::string_view text = "\x80x";
    size_t i = 0;
    EXPECT_EQ(decode_utf8(text, i), 0xFFFDu);
    EXPECT_EQ(i, 1u); // resync: advances one byte
    EXPECT_EQ(decode_utf8(text, i), static_cast<uint32_t>('x'));
}

TEST(DecodeUtf8, TruncatedSequenceYieldsReplacement) {
    std::string_view text = "\xE2\x82"; // 3-byte lead, only 2 bytes present
    size_t i = 0;
    EXPECT_EQ(decode_utf8(text, i), 0xFFFDu);
    EXPECT_EQ(i, text.size()); // consumed to end of string
}

TEST(DecodeUtf8, InvalidContinuationYieldsReplacement) {
    std::string_view text = "\xC3Zx"; // 2-byte lead followed by ASCII
    size_t i = 0;
    EXPECT_EQ(decode_utf8(text, i), 0xFFFDu);
    EXPECT_EQ(i, 1u);
    EXPECT_EQ(decode_utf8(text, i), static_cast<uint32_t>('Z'));
}

TEST(DecodeUtf8, MixedStringDecodesFully) {
    std::string_view text = "a\xC3\xA9!";
    size_t i = 0;
    uint32_t cps[3] = {};
    int n = 0;
    while (i < text.size() && n < 3) {
        cps[n++] = decode_utf8(text, i);
    }
    EXPECT_EQ(n, 3);
    EXPECT_EQ(cps[0], static_cast<uint32_t>('a'));
    EXPECT_EQ(cps[1], 0xE9u);
    EXPECT_EQ(cps[2], static_cast<uint32_t>('!'));
}
