#include <gtest/gtest.h>

#include "bracken/core/color.hpp"

#include <cstdint>
#include <ios>
#include <type_traits>

using bracken::core::pack_color;
using bracken::core::unpack_color;

// The engine-wide packed-color convention is ARGB: 0xAARRGGBB. These tests
// pin the absolute byte positions — a byte-order regression in either helper
// silently red/blue-swaps every UI, text, and debug-line color.

TEST(PackColor, ByteOrderIsARGB) {
    EXPECT_EQ(pack_color(0x12, 0x34, 0x56, 0x78), 0x78123456u);
}

TEST(PackColor, EachChannelLandsInItsByte) {
    EXPECT_EQ(pack_color(0xFF, 0x00, 0x00, 0x00), 0x00FF0000u); // red -> bits 23-16
    EXPECT_EQ(pack_color(0x00, 0xFF, 0x00, 0x00), 0x0000FF00u); // green -> bits 15-8
    EXPECT_EQ(pack_color(0x00, 0x00, 0xFF, 0x00), 0x000000FFu); // blue -> bits 7-0
    EXPECT_EQ(pack_color(0x00, 0x00, 0x00, 0xFF), 0xFF000000u); // alpha -> bits 31-24
}

TEST(PackColor, AlphaDefaultsToOpaque) {
    EXPECT_EQ(pack_color(0x10, 0x20, 0x30), 0xFF102030u);
}

TEST(UnpackColor, EachByteLandsInItsChannel) {
    const glm::vec4 red = unpack_color(0x00FF0000u);
    EXPECT_FLOAT_EQ(red.r, 1.0f);
    EXPECT_FLOAT_EQ(red.g, 0.0f);
    EXPECT_FLOAT_EQ(red.b, 0.0f);
    EXPECT_FLOAT_EQ(red.a, 0.0f);

    const glm::vec4 green = unpack_color(0x0000FF00u);
    EXPECT_FLOAT_EQ(green.g, 1.0f);
    EXPECT_FLOAT_EQ(green.r + green.b + green.a, 0.0f);

    const glm::vec4 blue = unpack_color(0x000000FFu);
    EXPECT_FLOAT_EQ(blue.b, 1.0f);
    EXPECT_FLOAT_EQ(blue.r + blue.g + blue.a, 0.0f);

    const glm::vec4 alpha = unpack_color(0xFF000000u);
    EXPECT_FLOAT_EQ(alpha.a, 1.0f);
    EXPECT_FLOAT_EQ(alpha.r + alpha.g + alpha.b, 0.0f);
}

TEST(UnpackColor, NormalizesToUnitRange) {
    const glm::vec4 c = unpack_color(0x78123456u);
    EXPECT_FLOAT_EQ(c.r, 0x12 / 255.0f);
    EXPECT_FLOAT_EQ(c.g, 0x34 / 255.0f);
    EXPECT_FLOAT_EQ(c.b, 0x56 / 255.0f);
    EXPECT_FLOAT_EQ(c.a, 0x78 / 255.0f);
}

TEST(ColorRoundTrip, PackThenUnpackRecoversChannels) {
    const glm::vec4 c = unpack_color(pack_color(0xAB, 0xCD, 0xEF, 0x01));
    EXPECT_FLOAT_EQ(c.r, 0xAB / 255.0f);
    EXPECT_FLOAT_EQ(c.g, 0xCD / 255.0f);
    EXPECT_FLOAT_EQ(c.b, 0xEF / 255.0f);
    EXPECT_FLOAT_EQ(c.a, 0x01 / 255.0f);
}

TEST(ColorRoundTrip, UnpackThenPackIsIdentity) {
    for (uint32_t word : {0x00000000u, 0xFFFFFFFFu, 0x12345678u, 0xFF30FF60u, 0xB0101418u}) {
        const glm::vec4 c = unpack_color(word);
        const uint32_t repacked =
            pack_color(static_cast<uint8_t>(c.r * 255.0f + 0.5f), static_cast<uint8_t>(c.g * 255.0f + 0.5f),
                       static_cast<uint8_t>(c.b * 255.0f + 0.5f), static_cast<uint8_t>(c.a * 255.0f + 0.5f));
        EXPECT_EQ(repacked, word);
    }
}

// ---------------------------------------------------------------------------
// core::Color — the packed word as a type. It exists so a signature can say
// WHICH of the engine's three color conventions a parameter uses; these pin
// that it is byte-for-byte the word it replaced and loses nothing round-tripping.
// ---------------------------------------------------------------------------

using bracken::core::Color;

TEST(ColorType, IsTheWordItReplaced) {
    static_assert(sizeof(Color) == sizeof(uint32_t));
    static_assert(std::is_trivially_copyable_v<Color>);
    static_assert(std::is_standard_layout_v<Color>);
    // Implicit from the packed literal: existing hex call sites keep working.
    const Color c = 0xFF30FF60u;
    EXPECT_EQ(c.argb, 0xFF30FF60u);
}

TEST(ColorType, DecodeMatchesTheFreeFunction) {
    // to_linear() must be exactly unpack_color — every renderer decodes one
    // way, and a second decode with a different byte order is the bug this
    // whole convention exists to prevent.
    for (uint32_t word : {0x00000000u, 0xFFFFFFFFu, 0x12345678u, 0xFF30FF60u, 0xB0101418u}) {
        EXPECT_EQ(Color{word}.to_linear(), unpack_color(word)) << std::hex << word;
    }
}

TEST(ColorType, FromLinearRoundTripsEveryChannelValue) {
    // Rounding, not truncation: a truncating quantizer loses a step on most
    // channel values, which shows up as a UI color drifting one bit darker
    // every time it passes through the type.
    for (int v = 0; v <= 255; ++v) {
        const auto b = static_cast<uint8_t>(v);
        const Color c = Color::rgba8(b, b, b, b);
        EXPECT_EQ(Color::from_linear(c.to_linear()), c) << v;
    }
}

TEST(ColorType, FromLinearClampsOutOfRange) {
    // HDR values do not belong in a packed color; saturating is the honest
    // answer, and it is why draw tints and light colors are NOT this type.
    EXPECT_EQ(Color::from_linear({4.0f, -2.0f, 0.0f, 1.0f}).argb, 0xFFFF0000u);
}

TEST(ColorType, NamedConstantsAreConstexpr) {
    static_assert(Color::white().argb == 0xFFFFFFFFu);
    static_assert(Color::black().argb == 0xFF000000u);
    static_assert(Color::transparent().argb == 0x00000000u);
    static_assert(Color::rgba8(0x12, 0x34, 0x56, 0x78).argb == pack_color(0x12, 0x34, 0x56, 0x78));
    static_assert(Color::rgb(1.0f, 0.0f, 0.0f).argb == 0xFFFF0000u);
    static_assert(Color::white() == Color{0xFFFFFFFFu});
    SUCCEED();
}

TEST(ColorHelpers, AreConstexpr) {
    static_assert(pack_color(1, 2, 3, 4) == 0x04010203u);
    // glm's constexpr ctor initializes the x/y/z/w union members; only those
    // are readable in a constant expression (.r/.g/.b/.a are fine at runtime).
    static_assert(unpack_color(0x00FF0000u).x == 1.0f); // red channel
    static_assert(unpack_color(0xFF000000u).w == 1.0f); // alpha channel
    SUCCEED();
}
