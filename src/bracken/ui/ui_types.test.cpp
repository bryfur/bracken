// The value types every other file in the module is written in terms of.
// Small, but load-bearing: a Rect whose intersect is wrong silently un-clips a
// scroll view, and an id whose scoping collapses makes two widgets share hot
// state, which reads as "the wrong button fired".

#include "bracken/ui/ui_types.hpp"

#include <gtest/gtest.h>

#include <set>

namespace {

using namespace bracken::ui;

TEST(UiRect, ContainsIsHalfOpenSoAdjacentRectsNeverBothClaimAPixel) {
    constexpr Rect a{0.0f, 0.0f, 10.0f, 10.0f};
    constexpr Rect b{10.0f, 0.0f, 10.0f, 10.0f};
    // The shared edge belongs to exactly one of them — which is what makes hit
    // testing over a laid-out row unambiguous.
    EXPECT_FALSE(a.contains({10.0f, 5.0f}));
    EXPECT_TRUE(b.contains({10.0f, 5.0f}));
    EXPECT_TRUE(a.contains({0.0f, 0.0f}));
    EXPECT_FALSE(a.contains({9.999f, 10.0f}));
}

TEST(UiRect, IntersectClipsAndReportsEmptyWithoutCollapsingToTheOrigin) {
    constexpr Rect a{0.0f, 0.0f, 100.0f, 100.0f};
    constexpr Rect b{50.0f, 50.0f, 100.0f, 100.0f};
    const Rect ab = a.intersect(b);
    EXPECT_FLOAT_EQ(ab.x, 50.0f);
    EXPECT_FLOAT_EQ(ab.y, 50.0f);
    EXPECT_FLOAT_EQ(ab.w, 50.0f);
    EXPECT_FLOAT_EQ(ab.h, 50.0f);
    EXPECT_FALSE(ab.empty());

    // Disjoint: empty, and it stays WHERE it went empty rather than jumping to
    // (0,0) — a clip chain that collapses at the origin would then start
    // intersecting things near the origin back in.
    constexpr Rect far_away{500.0f, 500.0f, 10.0f, 10.0f};
    const Rect none = a.intersect(far_away);
    EXPECT_TRUE(none.empty());
    EXPECT_FLOAT_EQ(none.x, 500.0f);
    EXPECT_FLOAT_EQ(none.y, 500.0f);

    // Intersecting with an already-empty rect stays empty.
    EXPECT_TRUE(a.intersect(none).empty());
}

TEST(UiRect, IntersectIsCommutativeAndIdempotent) {
    constexpr Rect a{10.0f, 20.0f, 100.0f, 40.0f};
    constexpr Rect b{40.0f, 0.0f, 100.0f, 45.0f};
    const Rect ab = a.intersect(b);
    const Rect ba = b.intersect(a);
    EXPECT_FLOAT_EQ(ab.x, ba.x);
    EXPECT_FLOAT_EQ(ab.y, ba.y);
    EXPECT_FLOAT_EQ(ab.w, ba.w);
    EXPECT_FLOAT_EQ(ab.h, ba.h);
    const Rect again = ab.intersect(a);
    EXPECT_FLOAT_EQ(again.w, ab.w);
    EXPECT_FLOAT_EQ(again.h, ab.h);
}

TEST(UiRect, EmptyIsWidthOrHeightNonPositive) {
    EXPECT_TRUE((Rect{0.0f, 0.0f, 0.0f, 10.0f}).empty());
    EXPECT_TRUE((Rect{0.0f, 0.0f, 10.0f, 0.0f}).empty());
    EXPECT_TRUE((Rect{0.0f, 0.0f, -1.0f, 10.0f}).empty());
    EXPECT_FALSE((Rect{0.0f, 0.0f, 1.0f, 1.0f}).empty());
    // An empty rect contains nothing, including its own corner.
    EXPECT_FALSE((Rect{5.0f, 5.0f, 0.0f, 0.0f}).contains({5.0f, 5.0f}));
}

TEST(UiEdges, HelpersMatchTheirSums) {
    constexpr Edges all = Edges::all(6.0f);
    EXPECT_FLOAT_EQ(all.horizontal(), 12.0f);
    EXPECT_FLOAT_EQ(all.vertical(), 12.0f);

    constexpr Edges xy = Edges::xy(8.0f, 4.0f);
    EXPECT_FLOAT_EQ(xy.l, 8.0f);
    EXPECT_FLOAT_EQ(xy.r, 8.0f);
    EXPECT_FLOAT_EQ(xy.t, 4.0f);
    EXPECT_FLOAT_EQ(xy.b, 4.0f);
    EXPECT_FLOAT_EQ(xy.horizontal(), 16.0f);
    EXPECT_FLOAT_EQ(xy.vertical(), 8.0f);

    constexpr Edges none{};
    EXPECT_FLOAT_EQ(none.horizontal(), 0.0f);
    EXPECT_FLOAT_EQ(none.vertical(), 0.0f);
}

TEST(UiSize, FactoriesSelectTheMode) {
    constexpr Size fixed = Size::px(24.0f);
    EXPECT_EQ(fixed.mode, SizeMode::Fixed);
    EXPECT_FLOAT_EQ(fixed.value, 24.0f);

    constexpr Size flex = Size::flex(2.0f);
    EXPECT_EQ(flex.mode, SizeMode::Flex);
    EXPECT_FLOAT_EQ(flex.value, 2.0f);
    EXPECT_FLOAT_EQ(Size::flex().value, 1.0f) << "the default weight is 1";

    constexpr Size content = Size::content();
    EXPECT_EQ(content.mode, SizeMode::Content);
    // The default-constructed Size must BE Content: every BoxStyle field
    // relies on it, so a change here silently resizes every UI in the tree.
    EXPECT_EQ(Size{}.mode, SizeMode::Content);
}

TEST(UiWidgetId, AnEmptyLabelIsTheInvalidId) {
    EXPECT_FALSE(id("").valid());
    EXPECT_TRUE(id("x").valid());
    EXPECT_FALSE(WidgetId{}.valid());
}

TEST(UiWidgetId, DistinctLabelsAndIndicesDoNotCollide) {
    // Not a hash-quality claim — a claim about the ids this module actually
    // produces. A collision here means two widgets share hot/active state.
    std::set<uint64_t> seen;
    const WidgetId parent = id("list");
    for (uint64_t i = 0; i < 512; ++i) {
        EXPECT_TRUE(seen.insert(id(parent, i).value).second) << "row " << i << " collided";
    }
    for (const char* label : {"resume", "quit", "volume", "fullscreen", "opt.a", "opt.b", "list", "name"}) {
        EXPECT_TRUE(seen.insert(id(label).value).second) << "label '" << label << "' collided";
    }
}

TEST(UiWidgetId, ScopingIsOrderSensitiveAndSeparatesEqualLabels) {
    const WidgetId panel_a = id("panel.a");
    const WidgetId panel_b = id("panel.b");
    const WidgetId ok = id("ok");

    // The same label under two panels is two widgets — the property push_id
    // exists to provide.
    EXPECT_NE(combine(panel_a, ok), combine(panel_b, ok));
    // ...and combining is not commutative, so a path is a path and not a set.
    EXPECT_NE(combine(panel_a, ok), combine(ok, panel_a));
    // Deterministic: the same inputs give the same id every time, which is the
    // entire cross-frame contract.
    EXPECT_EQ(combine(panel_a, ok), combine(panel_a, ok));
}

TEST(UiStyle, DefaultsAreOpaqueWhereTheyHaveToBe) {
    const Style s{};
    // begin_panel skips a FULLY TRANSPARENT border rather than emitting four
    // invisible quads, so the default border must have alpha or the default
    // panel silently loses its outline.
    EXPECT_NE(s.panel_border.argb >> 24, 0u);
    EXPECT_NE(s.text.argb >> 24, 0u);
    EXPECT_NE(s.focus_border.argb >> 24, 0u);
    EXPECT_GT(s.border_width, 0.0f);
    EXPECT_GT(s.control_height, 0.0f);
    EXPECT_GT(s.text_scale, 0.0f);
}

} // namespace
