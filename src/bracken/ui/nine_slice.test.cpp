#include "bracken/ui/nine_slice.hpp"

#include <gtest/gtest.h>

#include <span>

namespace {

using namespace bracken::ui;

NineSlice make_slice(bool center = true) {
    NineSlice s;
    s.texture_size = {32.0f, 32.0f};
    s.border = Edges::all(8.0f);
    s.draw_center = center;
    return s;
}

TEST(UiNineSlice, NineQuadsExactlyTileTheDestination) {
    NineSliceQuad q[9]{};
    const Rect dst{10.0f, 20.0f, 100.0f, 60.0f};
    const uint32_t n = build_nine_slice(make_slice(), dst, std::span<NineSliceQuad, 9>(q));
    ASSERT_EQ(n, 9u);

    float area = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_GE(q[i].dst.w, 0.0f);
        EXPECT_GE(q[i].dst.h, 0.0f);
        area += q[i].dst.w * q[i].dst.h;
    }
    EXPECT_FLOAT_EQ(area, dst.w * dst.h);

    // Row-major, and the outer edges are the destination's own edges.
    EXPECT_FLOAT_EQ(q[0].dst.x, dst.x);
    EXPECT_FLOAT_EQ(q[0].dst.y, dst.y);
    EXPECT_FLOAT_EQ(q[2].dst.right(), dst.right());
    EXPECT_FLOAT_EQ(q[6].dst.bottom(), dst.bottom());
    EXPECT_FLOAT_EQ(q[8].dst.right(), dst.right());
    EXPECT_FLOAT_EQ(q[8].dst.bottom(), dst.bottom());

    // Interior seams are SHARED edges, so there is no gap and no overlap.
    EXPECT_FLOAT_EQ(q[0].dst.right(), q[1].dst.x);
    EXPECT_FLOAT_EQ(q[1].dst.right(), q[2].dst.x);
    EXPECT_FLOAT_EQ(q[0].dst.bottom(), q[3].dst.y);
    EXPECT_FLOAT_EQ(q[3].dst.bottom(), q[6].dst.y);

    // Corners keep their authored size; the middle takes the rest.
    EXPECT_FLOAT_EQ(q[0].dst.w, 8.0f);
    EXPECT_FLOAT_EQ(q[0].dst.h, 8.0f);
    EXPECT_FLOAT_EQ(q[4].dst.w, 100.0f - 16.0f);
    EXPECT_FLOAT_EQ(q[4].dst.h, 60.0f - 16.0f);
}

TEST(UiNineSlice, UvsAreMonotoneAndInsideTheUnitSquare) {
    NineSliceQuad q[9]{};
    const uint32_t n = build_nine_slice(make_slice(), Rect{0.0f, 0.0f, 100.0f, 100.0f}, std::span<NineSliceQuad, 9>(q));
    ASSERT_EQ(n, 9u);
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_GE(q[i].u0, 0.0f);
        EXPECT_LE(q[i].u1, 1.0f);
        EXPECT_GE(q[i].v0, 0.0f);
        EXPECT_LE(q[i].v1, 1.0f);
        EXPECT_LE(q[i].u0, q[i].u1);
        EXPECT_LE(q[i].v0, q[i].v1);
    }
    // 8 / 32 authored border.
    EXPECT_FLOAT_EQ(q[0].u1, 0.25f);
    EXPECT_FLOAT_EQ(q[2].u0, 0.75f);
    EXPECT_FLOAT_EQ(q[4].u0, 0.25f);
    EXPECT_FLOAT_EQ(q[4].u1, 0.75f);
}

TEST(UiNineSlice, BordersWiderThanTheBoxScaleDownInsteadOfInverting) {
    NineSliceQuad q[9]{};
    // 8 + 8 borders in a 10 px-wide box: each must become 5, and the middle 0.
    const Rect dst{0.0f, 0.0f, 10.0f, 100.0f};
    const uint32_t n = build_nine_slice(make_slice(), dst, std::span<NineSliceQuad, 9>(q));
    ASSERT_EQ(n, 9u);
    EXPECT_FLOAT_EQ(q[0].dst.w, 5.0f);
    EXPECT_FLOAT_EQ(q[2].dst.w, 5.0f);
    EXPECT_FLOAT_EQ(q[1].dst.w, 0.0f);
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_GE(q[i].dst.w, 0.0f) << "quad " << i << " inverted";
    }
    // The UVs are unchanged — they come from the SOURCE border, so the corner
    // is still the corner even though it drew half size.
    EXPECT_FLOAT_EQ(q[0].u1, 0.25f);
}

TEST(UiNineSlice, NoCenterYieldsEightQuadsAndLeavesTheHole) {
    NineSliceQuad q[9]{};
    const Rect dst{0.0f, 0.0f, 100.0f, 100.0f};
    const uint32_t n = build_nine_slice(make_slice(false), dst, std::span<NineSliceQuad, 9>(q));
    ASSERT_EQ(n, 8u);
    float area = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        area += q[i].dst.w * q[i].dst.h;
    }
    const float middle = (100.0f - 16.0f) * (100.0f - 16.0f);
    EXPECT_FLOAT_EQ(area, dst.w * dst.h - middle);
}

TEST(UiNineSlice, BordersWiderThanTheSOURCEDoNotInvertTheUvs) {
    // The destination bands were guarded against inversion from the start; the
    // UV columns were not. Authored borders whose sum exceeds the SOURCE width
    // (or a texture_size the caller got wrong) put the third UV column below
    // the second, and a quad with u1 < u0 samples its band MIRRORED — corner
    // art drawn inside out, which reads as a texture bug rather than as bad
    // authoring.
    NineSlice slice{};
    slice.texture_size = {32.0f, 32.0f};
    slice.border = Edges::all(24.0f); // 24 + 24 against a 32 px source

    NineSliceQuad q[9]{};
    const uint32_t n = build_nine_slice(slice, Rect{0.0f, 0.0f, 200.0f, 200.0f}, std::span<NineSliceQuad, 9>(q));
    ASSERT_EQ(n, 9u);
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_LE(q[i].u0, q[i].u1) << "quad " << i << " samples u mirrored";
        EXPECT_LE(q[i].v0, q[i].v1) << "quad " << i << " samples v mirrored";
    }
    // The degenerate-but-correct answer: the middle band collapses to zero
    // width rather than running backwards.
    EXPECT_FLOAT_EQ(q[4].u0, q[4].u1);
    EXPECT_FLOAT_EQ(q[4].v0, q[4].v1);
}

TEST(UiNineSlice, EmptyDestinationYieldsNoQuads) {
    NineSliceQuad q[9]{};
    EXPECT_EQ(build_nine_slice(make_slice(), Rect{0.0f, 0.0f, 0.0f, 50.0f}, std::span<NineSliceQuad, 9>(q)), 0u);
    EXPECT_EQ(build_nine_slice(make_slice(), Rect{0.0f, 0.0f, 50.0f, -1.0f}, std::span<NineSliceQuad, 9>(q)), 0u);
}

} // namespace
