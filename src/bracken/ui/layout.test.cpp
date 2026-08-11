#include "bracken/ui/layout.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <vector>

namespace {

using namespace bracken::ui;

constexpr Rect k_viewport{0.0f, 0.0f, 400.0f, 300.0f};

TEST(UiLayout, RowOfFixedChildrenHonoursPaddingAndSpacing) {
    LayoutTree t;
    BoxStyle s{};
    s.padding = Edges{10.0f, 5.0f, 10.0f, 5.0f};
    s.spacing = 4.0f;
    s.width = Size::flex(1.0f);
    s.height = Size::flex(1.0f);
    const uint32_t row = t.begin_box(Axis::Horizontal, s);
    const uint32_t a = t.leaf(Size::px(30.0f), Size::px(20.0f), {30.0f, 20.0f});
    const uint32_t b = t.leaf(Size::px(50.0f), Size::px(20.0f), {50.0f, 20.0f});
    const uint32_t c = t.leaf(Size::px(70.0f), Size::px(20.0f), {70.0f, 20.0f});
    t.end_box();
    t.solve(k_viewport);

    EXPECT_FLOAT_EQ(t.rect_of(row).x, 0.0f);
    EXPECT_FLOAT_EQ(t.rect_of(a).x, 10.0f);
    EXPECT_FLOAT_EQ(t.rect_of(b).x, 10.0f + 30.0f + 4.0f);
    EXPECT_FLOAT_EQ(t.rect_of(c).x, 10.0f + 30.0f + 4.0f + 50.0f + 4.0f);
    EXPECT_FLOAT_EQ(t.rect_of(a).y, 5.0f);
    EXPECT_FLOAT_EQ(t.rect_of(a).w, 30.0f);
    EXPECT_FLOAT_EQ(t.rect_of(c).w, 70.0f);
}

TEST(UiLayout, FlexWeightsSplitTheLeftoverWithNoDrift) {
    LayoutTree t;
    BoxStyle s{};
    s.width = Size::flex(1.0f);
    s.height = Size::flex(1.0f);
    const uint32_t row = t.begin_box(Axis::Horizontal, s);
    const uint32_t a = t.leaf(Size::flex(1.0f), Size::px(10.0f), {0.0f, 10.0f});
    const uint32_t b = t.leaf(Size::flex(2.0f), Size::px(10.0f), {0.0f, 10.0f});
    const uint32_t c = t.leaf(Size::flex(1.0f), Size::px(10.0f), {0.0f, 10.0f});
    t.end_box();
    t.solve(k_viewport);

    EXPECT_FLOAT_EQ(t.rect_of(a).w, 100.0f);
    EXPECT_FLOAT_EQ(t.rect_of(b).w, 200.0f);
    EXPECT_FLOAT_EQ(t.rect_of(c).w, 100.0f);
    // The gate: the last child's right edge lands exactly on the content edge.
    EXPECT_FLOAT_EQ(t.rect_of(c).right(), t.rect_of(row).right());
}

TEST(UiLayout, FlexWithAwkwardWeightsStillEndsExactlyOnTheContentEdge) {
    // Three sevenths do not sum to one in floating point when added one at a
    // time. The cumulative-fraction distribution is what makes this exact.
    LayoutTree t;
    BoxStyle s{};
    s.width = Size::flex(1.0f);
    s.height = Size::flex(1.0f);
    const uint32_t row = t.begin_box(Axis::Horizontal, s);
    uint32_t last = LayoutTree::k_invalid;
    for (int i = 0; i < 7; ++i) {
        last = t.leaf(Size::flex(1.0f), Size::px(10.0f), {0.0f, 10.0f});
    }
    t.end_box();
    t.solve(k_viewport);
    EXPECT_FLOAT_EQ(t.rect_of(last).right(), t.rect_of(row).right());
}

TEST(UiLayout, ContentSizingPropagatesThroughTwoNestingLevels) {
    LayoutTree t;
    BoxStyle outer{};
    outer.padding = Edges::all(6.0f);
    BoxStyle inner{};
    inner.padding = Edges::all(3.0f);
    const uint32_t o = t.begin_box(Axis::Vertical, outer);
    const uint32_t i = t.begin_box(Axis::Vertical, inner);
    (void)t.leaf(Size::px(40.0f), Size::px(12.0f), {40.0f, 12.0f});
    (void)t.leaf(Size::px(25.0f), Size::px(12.0f), {25.0f, 12.0f});
    t.end_box();
    t.end_box();
    t.solve(k_viewport);

    // inner: widest child 40 + 2*3 padding; heights 12+12 + 2*3.
    EXPECT_FLOAT_EQ(t.content_of(i).x, 46.0f);
    EXPECT_FLOAT_EQ(t.content_of(i).y, 30.0f);
    // outer: inner + 2*6 padding, bottom-up.
    EXPECT_FLOAT_EQ(t.content_of(o).x, 58.0f);
    EXPECT_FLOAT_EQ(t.content_of(o).y, 42.0f);
    EXPECT_FLOAT_EQ(t.rect_of(o).w, 58.0f);
    EXPECT_FLOAT_EQ(t.rect_of(o).h, 42.0f);
}

TEST(UiLayout, CrossAxisAlignmentPlacesChildren) {
    // The child has an EXPLICIT cross size (40 px wide in a vertical box), so
    // the alignment only ever places it.
    const auto solve_with = [](Align cross) {
        LayoutTree t;
        BoxStyle s{};
        s.cross = cross;
        s.width = Size::px(200.0f);
        s.height = Size::px(100.0f);
        t.begin_box(Axis::Vertical, s);
        const uint32_t child = t.leaf(Size::px(40.0f), Size::px(10.0f), {40.0f, 10.0f});
        t.end_box();
        t.solve(k_viewport);
        return t.rect_of(child);
    };

    EXPECT_FLOAT_EQ(solve_with(Align::Start).x, 0.0f);
    EXPECT_FLOAT_EQ(solve_with(Align::Center).x, 80.0f);
    EXPECT_FLOAT_EQ(solve_with(Align::End).x, 160.0f);
    // Stretch does NOT override an explicit Fixed cross size. The child asked
    // for exactly 40 px and the parent's alignment is the less specific of the
    // two — otherwise a 100 px-wide toggle placed in a Stretch column would be
    // silently widened to the column, and there would be no way to say "this
    // one is narrow" at all. This is also the flexbox rule (align-items only
    // stretches an `auto` cross size).
    const Rect stretched = solve_with(Align::Stretch);
    EXPECT_FLOAT_EQ(stretched.x, 0.0f);
    EXPECT_FLOAT_EQ(stretched.w, 40.0f) << "an explicit Fixed cross size wins over the parent's Stretch";
}

TEST(UiLayout, StretchWidensAChildThatDidNotAskForACrossSize) {
    // The other half of the rule above, and the one that makes Stretch worth
    // having: a CONTENT-sized child takes the container's cross extent.
    LayoutTree t;
    BoxStyle s{};
    s.cross = Align::Stretch;
    s.width = Size::px(200.0f);
    s.height = Size::px(100.0f);
    t.begin_box(Axis::Vertical, s);
    const uint32_t child = t.leaf(Size::content(), Size::px(10.0f), {40.0f, 10.0f});
    t.end_box();
    t.solve(k_viewport);

    const Rect r = t.rect_of(child);
    EXPECT_FLOAT_EQ(r.x, 0.0f);
    EXPECT_FLOAT_EQ(r.w, 200.0f);
}

TEST(UiLayout, MainAxisAlignmentPlacesTheLeftover) {
    const auto solve_with = [](Align main) {
        LayoutTree t;
        BoxStyle s{};
        s.main = main;
        s.width = Size::px(200.0f);
        s.height = Size::px(100.0f);
        t.begin_box(Axis::Horizontal, s);
        const uint32_t child = t.leaf(Size::px(40.0f), Size::px(10.0f), {40.0f, 10.0f});
        t.end_box();
        t.solve(k_viewport);
        return t.rect_of(child).x;
    };
    EXPECT_FLOAT_EQ(solve_with(Align::Start), 0.0f);
    EXPECT_FLOAT_EQ(solve_with(Align::Center), 80.0f);
    EXPECT_FLOAT_EQ(solve_with(Align::End), 160.0f);
    // Stretch has no meaning on the main axis and documents itself as Start.
    EXPECT_FLOAT_EQ(solve_with(Align::Stretch), 0.0f);
}

TEST(UiLayout, GridOfSevenInThreeColumnsMakesThreeRowsWithAPartialLast) {
    LayoutTree t;
    BoxStyle s{};
    s.spacing = 5.0f;
    s.width = Size::px(200.0f);
    s.height = Size::px(200.0f);
    t.begin_grid(3, s);
    std::vector<uint32_t> cells;
    for (int i = 0; i < 7; ++i) {
        cells.push_back(t.leaf(Size::px(20.0f), Size::px(10.0f), {20.0f, 10.0f}));
    }
    t.end_box();
    t.solve(k_viewport);

    // Row-major: 0,1,2 / 3,4,5 / 6.
    EXPECT_FLOAT_EQ(t.rect_of(cells[0]).y, t.rect_of(cells[1]).y);
    EXPECT_FLOAT_EQ(t.rect_of(cells[0]).y, t.rect_of(cells[2]).y);
    EXPECT_GT(t.rect_of(cells[3]).y, t.rect_of(cells[0]).y);
    EXPECT_GT(t.rect_of(cells[6]).y, t.rect_of(cells[3]).y);
    // Three distinct rows and the last one holds exactly one cell.
    EXPECT_FLOAT_EQ(t.rect_of(cells[6]).x, t.rect_of(cells[0]).x);
    // Columns are evenly divided across the inner width.
    const float cell_w = (200.0f - 2.0f * 5.0f) / 3.0f;
    EXPECT_FLOAT_EQ(t.rect_of(cells[0]).w, cell_w);
    EXPECT_FLOAT_EQ(t.rect_of(cells[1]).x, t.rect_of(cells[0]).x + cell_w + 5.0f);
}

TEST(UiLayout, SolvingTheSameTreeTwiceIsBitIdentical) {
    LayoutTree t;
    BoxStyle s{};
    s.padding = Edges::all(7.0f);
    s.spacing = 3.0f;
    s.width = Size::flex(1.0f);
    t.begin_box(Axis::Vertical, s);
    std::vector<uint32_t> nodes;
    for (int i = 0; i < 12; ++i) {
        nodes.push_back(t.leaf(Size::flex(static_cast<float>(i % 3 + 1)), Size::px(11.0f), {0.0f, 11.0f}));
    }
    t.end_box();

    t.solve(k_viewport);
    std::vector<Rect> first;
    for (uint32_t n : nodes) {
        first.push_back(t.rect_of(n));
    }
    t.solve(k_viewport);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const Rect r = t.rect_of(nodes[i]);
        // Bit-identical, not merely close: layout feeds the packet, and the
        // merged frame must be byte-identical at any worker count.
        EXPECT_EQ(std::bit_cast<uint32_t>(r.x), std::bit_cast<uint32_t>(first[i].x));
        EXPECT_EQ(std::bit_cast<uint32_t>(r.y), std::bit_cast<uint32_t>(first[i].y));
        EXPECT_EQ(std::bit_cast<uint32_t>(r.w), std::bit_cast<uint32_t>(first[i].w));
        EXPECT_EQ(std::bit_cast<uint32_t>(r.h), std::bit_cast<uint32_t>(first[i].h));
    }
}

TEST(UiLayout, DepthOverflowIsCountedAndLeavesTheTreeWellFormed) {
    LayoutTree t;
    // Open twice the limit, then close them all. The overflow must be
    // absorbed symmetrically or the extra end_box calls would close real
    // containers and every rect after them would be wrong.
    const uint32_t over = LayoutTree::k_max_depth * 2;
    for (uint32_t i = 0; i < over; ++i) {
        t.begin_box(Axis::Vertical, {});
    }
    const uint32_t marker = t.leaf(Size::px(5.0f), Size::px(5.0f), {5.0f, 5.0f});
    for (uint32_t i = 0; i < over; ++i) {
        t.end_box();
    }
    EXPECT_GT(t.depth_overflow(), 0u);
    EXPECT_EQ(t.unbalanced_end_box(), 0u);
    EXPECT_EQ(t.depth(), 1u); // back at the root, nothing left open
    t.solve(k_viewport);
    EXPECT_GT(t.rect_of(marker).w, 0.0f); // still placed, not corrupted
}

TEST(UiLayout, UnbalancedEndBoxIsCountedAndDoesNotCorruptTheTree) {
    LayoutTree t;
    t.begin_box(Axis::Vertical, {});
    t.end_box();
    t.end_box(); // one too many
    t.end_box(); // two too many
    EXPECT_EQ(t.unbalanced_end_box(), 2u);
    EXPECT_EQ(t.depth(), 1u);
    const uint32_t after = t.leaf(Size::px(9.0f), Size::px(9.0f), {9.0f, 9.0f});
    t.solve(k_viewport);
    EXPECT_FLOAT_EQ(t.rect_of(after).w, 9.0f);
}

TEST(UiLayout, ScrollFreeAxisLaysChildrenOutBeyondTheViewport) {
    LayoutTree t;
    BoxStyle s{};
    s.width = Size::px(100.0f);
    s.height = Size::px(50.0f);
    const uint32_t scroll = t.begin_box(Axis::Vertical, s);
    t.set_scroll_free(scroll, Axis::Vertical);
    std::vector<uint32_t> rows;
    for (int i = 0; i < 10; ++i) {
        rows.push_back(t.leaf(Size::px(80.0f), Size::px(20.0f), {80.0f, 20.0f}));
    }
    t.end_box();
    t.solve(k_viewport);

    // The viewport stays 50 tall but the children occupy the full 200.
    EXPECT_FLOAT_EQ(t.rect_of(scroll).h, 50.0f);
    EXPECT_FLOAT_EQ(t.rect_of(rows.back()).y, t.rect_of(scroll).y + 9.0f * 20.0f);
    // content_of reports the Fixed viewport size; content_extent_of reports
    // what the children actually wanted. A scroll offset computed from the
    // former is always zero, which is exactly the bug the split exists for.
    EXPECT_FLOAT_EQ(t.content_of(scroll).y, 50.0f);
    EXPECT_FLOAT_EQ(t.content_extent_of(scroll).y, 200.0f);

    // translate_children moves the rows and leaves the viewport alone.
    const float before = t.rect_of(rows.front()).y;
    t.translate_children(scroll, {0.0f, -30.0f});
    EXPECT_FLOAT_EQ(t.rect_of(rows.front()).y, before - 30.0f);
    EXPECT_FLOAT_EQ(t.rect_of(scroll).h, 50.0f);
    EXPECT_FLOAT_EQ(t.rect_of(scroll).y, 0.0f);
}

TEST(UiLayout, DescendantQueryFindsAncestorsAndRejectsSiblings) {
    LayoutTree t;
    const uint32_t outer = t.begin_box(Axis::Vertical, {});
    const uint32_t inner = t.begin_box(Axis::Vertical, {});
    const uint32_t deep = t.leaf(Size::px(1.0f), Size::px(1.0f), {1.0f, 1.0f});
    t.end_box();
    const uint32_t sibling = t.leaf(Size::px(1.0f), Size::px(1.0f), {1.0f, 1.0f});
    t.end_box();

    EXPECT_TRUE(t.is_descendant_of(deep, inner));
    EXPECT_TRUE(t.is_descendant_of(deep, outer));
    EXPECT_TRUE(t.is_descendant_of(inner, inner));
    EXPECT_FALSE(t.is_descendant_of(sibling, inner));
}

TEST(UiRect, IntersectAndContainsAreHalfOpen) {
    const Rect a{0.0f, 0.0f, 10.0f, 10.0f};
    const Rect b{5.0f, 5.0f, 10.0f, 10.0f};
    const Rect i = a.intersect(b);
    EXPECT_FLOAT_EQ(i.x, 5.0f);
    EXPECT_FLOAT_EQ(i.w, 5.0f);
    EXPECT_FALSE(i.empty());

    EXPECT_TRUE(a.contains({0.0f, 0.0f}));
    EXPECT_FALSE(a.contains({10.0f, 5.0f})); // far edge is exclusive
    EXPECT_TRUE(a.intersect(Rect{20.0f, 20.0f, 5.0f, 5.0f}).empty());
}

TEST(UiWidgetId, IdsAreStableAndScopingSeparatesEqualLabels) {
    EXPECT_EQ(id("play").value, id("play").value);
    EXPECT_NE(id("play").value, id("quit").value);
    EXPECT_TRUE(id("play").valid());
    EXPECT_FALSE(id("").valid());

    const WidgetId panel_a = id("settings");
    const WidgetId panel_b = id("audio");
    EXPECT_NE(combine(panel_a, id("volume")).value, combine(panel_b, id("volume")).value);
    EXPECT_EQ(combine(panel_a, id("volume")).value, combine(panel_a, id("volume")).value);

    // List rows are distinct and stable.
    EXPECT_NE(id(panel_a, 0).value, id(panel_a, 1).value);
    EXPECT_EQ(id(panel_a, 7).value, id(panel_a, 7).value);
}

} // namespace
