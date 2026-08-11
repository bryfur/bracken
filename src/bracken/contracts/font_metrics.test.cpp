#include "bracken/contracts/font_metrics.hpp"

#include <gtest/gtest.h>

#include <string>

namespace {

using bracken::FontMetrics;

/// A hand-built table standing in for an exported font: 'a'..'e' with distinct
/// advances, one 2-byte, one 3-byte and one 4-byte codepoint in the sparse
/// table, and one nonzero kerning pair. Everything the tests assert is
/// derivable from these numbers by hand, which is the point — this file pins
/// the pen walk, not SDL_ttf.
FontMetrics make_table() {
    FontMetrics m;
    m.line_height = 20;
    m.line_skip = 24;
    m.ascent = 16;
    m.descent = -4;
    m.pixel_size = 18;
    m.fallback_advance = 7;
    m.advance.fill(FontMetrics::k_absent);
    m.advance['a' - FontMetrics::k_first_cp] = 10;
    m.advance['b' - FontMetrics::k_first_cp] = 11;
    m.advance['c' - FontMetrics::k_first_cp] = 12;
    m.advance['d' - FontMetrics::k_first_cp] = 13;
    m.advance[' ' - FontMetrics::k_first_cp] = 5;
    // U+00E9 (2-byte) is inside the dense range; U+2019 (3-byte) and
    // U+1F600 (4-byte) are not, so they exercise the sparse path.
    m.advance[0xE9 - FontMetrics::k_first_cp] = 14;
    m.sparse_advance = {{0x2019u, 6}, {0x1F600u, 20}};
    m.kerning = {{(static_cast<uint64_t>('a') << 32) | 'b', -3}};
    return m;
}

TEST(FontMetrics, InvalidTableMeasuresZeroAndNeverDividesByLineHeight) {
    const FontMetrics empty;
    EXPECT_FALSE(empty.valid());
    EXPECT_EQ(empty.measure_width("hello"), 0.0f);
    EXPECT_EQ(empty.measure_height(2.0f), 0.0f);
    EXPECT_EQ(empty.advance_of('a'), 0);
    EXPECT_EQ(empty.kerning_of('a', 'b'), 0);
    EXPECT_EQ(empty.fit_prefix("hello", 1000.0f), 0u);
    EXPECT_EQ(empty.count_fallbacks("hello"), 0u);
    // With no metrics, wrapping refuses to guess: one line, everything on it.
    const auto wp = empty.wrap_point("hello world", 10.0f);
    EXPECT_EQ(wp.consumed, 11u);
    EXPECT_EQ(wp.emit, 11u);
}

TEST(FontMetrics, MeasureWidthReproducesTheHandWrittenPenWalk) {
    const FontMetrics m = make_table();
    // "abc": a(10) + kern(a,b)=-3 + b(11) + kern(b,c)=0 + c(12) = 30.
    EXPECT_FLOAT_EQ(m.measure_width("abc"), 30.0f);
    // Scale multiplies the accumulated integer pen exactly once.
    EXPECT_FLOAT_EQ(m.measure_width("abc", 0.5f), 15.0f);
    // Order matters: "bac" has no kerning pair, so 11 + 10 + 12 = 33.
    EXPECT_FLOAT_EQ(m.measure_width("bac"), 33.0f);
    EXPECT_FLOAT_EQ(m.measure_width(""), 0.0f);
}

TEST(FontMetrics, MeasureWidthWalksMultiByteUtf8) {
    const FontMetrics m = make_table();
    // "a" + U+00E9 + U+2019 + U+1F600 = 10 + 14 + 6 + 20 = 50, and no kerning
    // pair among them.
    const std::string s = "aé’\U0001F600";
    EXPECT_EQ(s.size(), 1u + 2u + 3u + 4u);
    EXPECT_FLOAT_EQ(m.measure_width(s), 50.0f);
    EXPECT_EQ(m.count_fallbacks(s), 0u);
}

TEST(FontMetrics, AdvanceFallsBackExactlyOncePerUnknownCodepoint) {
    const FontMetrics m = make_table();
    EXPECT_TRUE(m.has_advance('a'));
    EXPECT_FALSE(m.has_advance('z'));   // in range, absent from the font
    EXPECT_FALSE(m.has_advance(0x4E00)); // out of range entirely
    EXPECT_EQ(m.advance_of('z'), m.fallback_advance);
    EXPECT_EQ(m.advance_of(0x4E00), m.fallback_advance);
    // Two unknowns, one known: exactly two fallbacks.
    EXPECT_EQ(m.count_fallbacks("zaz"), 2u);
    EXPECT_FLOAT_EQ(m.measure_width("zaz"), 7.0f + 10.0f + 7.0f);
}

TEST(FontMetrics, KerningIsZeroForAnAbsentPair) {
    const FontMetrics m = make_table();
    EXPECT_EQ(m.kerning_of('a', 'b'), -3);
    EXPECT_EQ(m.kerning_of('b', 'a'), 0); // the reverse pair is not stored
    EXPECT_EQ(m.kerning_of('c', 'd'), 0);
}

TEST(FontMetrics, FitPrefixIsMonotoneAndNeverSplitsAUtf8Sequence) {
    const FontMetrics m = make_table();
    const std::string s = "aé’\U0001F600"; // widths 10, 14, 6, 20

    // Monotone: a wider box never fits fewer bytes.
    std::size_t last = 0;
    for (float w = 0.0f; w <= 60.0f; w += 1.0f) {
        const std::size_t got = m.fit_prefix(s, w);
        EXPECT_GE(got, last) << "fit_prefix regressed at max_width " << w;
        last = got;
    }

    // Every achievable answer is a sequence boundary: 0, 1, 3, 6 or 10.
    for (float w = 0.0f; w <= 60.0f; w += 0.5f) {
        const std::size_t got = m.fit_prefix(s, w);
        EXPECT_TRUE(got == 0 || got == 1 || got == 3 || got == 6 || got == 10)
            << "fit_prefix returned mid-sequence byte " << got << " at width " << w;
    }

    // The exact string width fits the whole string.
    float achieved = 0.0f;
    EXPECT_EQ(m.fit_prefix(s, m.measure_width(s), 1.0f, &achieved), s.size());
    EXPECT_FLOAT_EQ(achieved, 50.0f);
    // One pixel short of the last glyph drops exactly that glyph.
    EXPECT_EQ(m.fit_prefix(s, 49.0f), 6u);
}

TEST(FontMetrics, WrapPointBreaksOnSpacesAndAlwaysMakesProgress) {
    const FontMetrics m = make_table();
    // "ab cd": a(10) kern(-3) b(11) space(5) c(12) d(13) = 48 total.
    const std::string s = "ab cd";

    // Wide enough for everything: one line.
    auto wp = m.wrap_point(s, 100.0f);
    EXPECT_EQ(wp.consumed, s.size());
    EXPECT_EQ(wp.emit, s.size());

    // Room for "ab " but not "ab cd": break at the space, swallow it.
    wp = m.wrap_point(s, 30.0f);
    EXPECT_EQ(wp.emit, 2u);      // "ab"
    EXPECT_EQ(wp.consumed, 3u);  // "ab "

    // Narrower than one glyph: emit one codepoint anyway, so a wrap loop
    // driven by `consumed` cannot spin.
    wp = m.wrap_point(s, 1.0f);
    EXPECT_EQ(wp.consumed, 1u);
    EXPECT_EQ(wp.emit, 1u);

    // Driving the whole string terminates and covers every byte.
    std::size_t pos = 0;
    int iterations = 0;
    while (pos < s.size() && iterations < 100) {
        const auto step = m.wrap_point(std::string_view(s).substr(pos), 30.0f);
        ASSERT_GT(step.consumed, 0u) << "wrap_point must always consume";
        pos += step.consumed;
        ++iterations;
    }
    EXPECT_EQ(pos, s.size());
    EXPECT_LT(iterations, 100);
}

} // namespace
