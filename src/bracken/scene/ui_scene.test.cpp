#include "bracken/scene/ui_scene.hpp"
#include <gtest/gtest.h>
#include <string>

using namespace bracken::scene;

TEST(UIScene, InitiallyEmpty) {
    UIScene scene;
    EXPECT_TRUE(scene.commands().empty());
}

TEST(UIScene, FilledRectStoresParameters) {
    UIScene scene;
    scene.add_filled_rect(10.0f, 20.0f, 100.0f, 50.0f, 0xFF0000FF);

    ASSERT_EQ(scene.commands().size(), 1u);
    const auto& cmd = scene.commands()[0];
    ASSERT_TRUE(cmd.is<FilledRectCommand>());
    const auto& rect = cmd.get<FilledRectCommand>();
    EXPECT_FLOAT_EQ(rect.x, 10.0f);
    EXPECT_FLOAT_EQ(rect.y, 20.0f);
    EXPECT_FLOAT_EQ(rect.w, 100.0f);
    EXPECT_FLOAT_EQ(rect.h, 50.0f);
    EXPECT_EQ(rect.color, 0xFF0000FFu);
}

TEST(UIScene, RectOutlineStoresParameters) {
    UIScene scene;
    scene.add_rect_outline(1.0f, 2.0f, 3.0f, 4.0f, 0x00FF00FF, 5.0f);

    ASSERT_EQ(scene.commands().size(), 1u);
    ASSERT_TRUE(scene.commands()[0].is<RectOutlineCommand>());
    const auto& outline = scene.commands()[0].get<RectOutlineCommand>();
    EXPECT_FLOAT_EQ(outline.x, 1.0f);
    EXPECT_FLOAT_EQ(outline.y, 2.0f);
    EXPECT_FLOAT_EQ(outline.w, 3.0f);
    EXPECT_FLOAT_EQ(outline.h, 4.0f);
    EXPECT_EQ(outline.color, 0x00FF00FFu);
    EXPECT_FLOAT_EQ(outline.line_width, 5.0f);
}

TEST(UIScene, RectOutlineDefaultLineWidth) {
    UIScene scene;
    scene.add_rect_outline(0.0f, 0.0f, 10.0f, 10.0f, 0xFFFFFFFF);

    ASSERT_EQ(scene.commands().size(), 1u);
    EXPECT_FLOAT_EQ(scene.commands()[0].get<RectOutlineCommand>().line_width, 2.0f);
}

TEST(UIScene, CircleStoresParameters) {
    UIScene scene;
    scene.add_circle(5.0f, 6.0f, 7.5f, 0x0000FFFF, 32);

    ASSERT_EQ(scene.commands().size(), 1u);
    ASSERT_TRUE(scene.commands()[0].is<CircleCommand>());
    const auto& circle = scene.commands()[0].get<CircleCommand>();
    EXPECT_FLOAT_EQ(circle.x, 5.0f);
    EXPECT_FLOAT_EQ(circle.y, 6.0f);
    EXPECT_FLOAT_EQ(circle.radius, 7.5f);
    EXPECT_EQ(circle.color, 0x0000FFFFu);
    EXPECT_EQ(circle.segments, 32);
}

TEST(UIScene, CircleDefaultSegments) {
    UIScene scene;
    scene.add_circle(0.0f, 0.0f, 1.0f, 0xFFFFFFFF);

    ASSERT_EQ(scene.commands().size(), 1u);
    EXPECT_EQ(scene.commands()[0].get<CircleCommand>().segments, 24);
}

TEST(UIScene, CircleOutlineStoresParameters) {
    UIScene scene;
    scene.add_circle_outline(3.0f, 4.0f, 5.0f, 0x12345678, 1.5f, 12);

    ASSERT_EQ(scene.commands().size(), 1u);
    ASSERT_TRUE(scene.commands()[0].is<CircleOutlineCommand>());
    const auto& outline = scene.commands()[0].get<CircleOutlineCommand>();
    EXPECT_FLOAT_EQ(outline.x, 3.0f);
    EXPECT_FLOAT_EQ(outline.y, 4.0f);
    EXPECT_FLOAT_EQ(outline.radius, 5.0f);
    EXPECT_EQ(outline.color, 0x12345678u);
    EXPECT_FLOAT_EQ(outline.line_width, 1.5f);
    EXPECT_EQ(outline.segments, 12);
}

TEST(UIScene, LineStoresParameters) {
    UIScene scene;
    scene.add_line(1.0f, 2.0f, 3.0f, 4.0f, 0xAABBCCDD, 3.0f);

    ASSERT_EQ(scene.commands().size(), 1u);
    ASSERT_TRUE(scene.commands()[0].is<LineCommand>());
    const auto& line = scene.commands()[0].get<LineCommand>();
    EXPECT_FLOAT_EQ(line.x1, 1.0f);
    EXPECT_FLOAT_EQ(line.y1, 2.0f);
    EXPECT_FLOAT_EQ(line.x2, 3.0f);
    EXPECT_FLOAT_EQ(line.y2, 4.0f);
    EXPECT_EQ(line.color, 0xAABBCCDDu);
    EXPECT_FLOAT_EQ(line.line_width, 3.0f);
}

TEST(UIScene, TextStoresParameters) {
    UIScene scene;
    scene.add_text("Hello", 15.0f, 25.0f, 2.0f, 0xFF00FF00);

    ASSERT_EQ(scene.commands().size(), 1u);
    ASSERT_TRUE(scene.commands()[0].is<TextCommand>());
    const auto& text = scene.commands()[0].get<TextCommand>();
    EXPECT_EQ(text.text, "Hello");
    EXPECT_FLOAT_EQ(text.x, 15.0f);
    EXPECT_FLOAT_EQ(text.y, 25.0f);
    EXPECT_FLOAT_EQ(text.scale, 2.0f);
    EXPECT_EQ(text.color, 0xFF00FF00u);
}

TEST(UIScene, TextDefaultScaleAndColor) {
    UIScene scene;
    scene.add_text("Default", 0.0f, 0.0f);

    ASSERT_EQ(scene.commands().size(), 1u);
    const auto& text = scene.commands()[0].get<TextCommand>();
    EXPECT_FLOAT_EQ(text.scale, 1.0f);
    EXPECT_EQ(text.color, 0xFFFFFFFFu);
}

TEST(UIScene, ButtonStoresParameters) {
    UIScene scene;
    scene.add_button(10.0f, 20.0f, 120.0f, 30.0f, "OK", 0x336699FF, true);

    ASSERT_EQ(scene.commands().size(), 1u);
    ASSERT_TRUE(scene.commands()[0].is<ButtonCommand>());
    const auto& btn = scene.commands()[0].get<ButtonCommand>();
    EXPECT_FLOAT_EQ(btn.x, 10.0f);
    EXPECT_FLOAT_EQ(btn.y, 20.0f);
    EXPECT_FLOAT_EQ(btn.w, 120.0f);
    EXPECT_FLOAT_EQ(btn.h, 30.0f);
    EXPECT_EQ(btn.label, "OK");
    EXPECT_EQ(btn.color, 0x336699FFu);
    EXPECT_TRUE(btn.selected);
}

TEST(UIScene, ButtonDefaultNotSelected) {
    UIScene scene;
    scene.add_button(0.0f, 0.0f, 50.0f, 20.0f, "Cancel", 0xFFFFFFFF);

    ASSERT_EQ(scene.commands().size(), 1u);
    EXPECT_FALSE(scene.commands()[0].get<ButtonCommand>().selected);
}

TEST(UIScene, CommandsAccumulateInSubmissionOrder) {
    UIScene scene;
    scene.add_filled_rect(1.0f, 0.0f, 1.0f, 1.0f, 0x1);
    scene.add_filled_rect(2.0f, 0.0f, 1.0f, 1.0f, 0x2);
    scene.add_filled_rect(3.0f, 0.0f, 1.0f, 1.0f, 0x3);

    ASSERT_EQ(scene.commands().size(), 3u);
    EXPECT_FLOAT_EQ(scene.commands()[0].get<FilledRectCommand>().x, 1.0f);
    EXPECT_FLOAT_EQ(scene.commands()[1].get<FilledRectCommand>().x, 2.0f);
    EXPECT_FLOAT_EQ(scene.commands()[2].get<FilledRectCommand>().x, 3.0f);
}

TEST(UIScene, InterleavedTypesPreserveRelativeOrder) {
    UIScene scene;
    // Single flat list: draw order across types must match submission order
    // (e.g. text drawn on top of a rect submitted before it).
    scene.add_filled_rect(0.0f, 0.0f, 10.0f, 10.0f, 0x1);
    scene.add_text("A", 1.0f, 1.0f);
    scene.add_circle(5.0f, 5.0f, 2.0f, 0x2);
    scene.add_line(0.0f, 0.0f, 1.0f, 1.0f, 0x3);
    scene.add_button(0.0f, 0.0f, 5.0f, 5.0f, "B", 0x4);
    scene.add_rect_outline(0.0f, 0.0f, 10.0f, 10.0f, 0x5);
    scene.add_circle_outline(5.0f, 5.0f, 2.0f, 0x6);

    ASSERT_EQ(scene.commands().size(), 7u);
    EXPECT_TRUE(scene.commands()[0].is<FilledRectCommand>());
    EXPECT_TRUE(scene.commands()[1].is<TextCommand>());
    EXPECT_TRUE(scene.commands()[2].is<CircleCommand>());
    EXPECT_TRUE(scene.commands()[3].is<LineCommand>());
    EXPECT_TRUE(scene.commands()[4].is<ButtonCommand>());
    EXPECT_TRUE(scene.commands()[5].is<RectOutlineCommand>());
    EXPECT_TRUE(scene.commands()[6].is<CircleOutlineCommand>());
    EXPECT_EQ(scene.commands()[1].get<TextCommand>().text, "A");
    EXPECT_EQ(scene.commands()[4].get<ButtonCommand>().label, "B");
}

TEST(UIScene, ClearEmptiesEverything) {
    UIScene scene;
    scene.add_filled_rect(0.0f, 0.0f, 1.0f, 1.0f, 0xFFFFFFFF);
    scene.add_text("gone", 0.0f, 0.0f);
    scene.add_button(0.0f, 0.0f, 1.0f, 1.0f, "gone", 0xFFFFFFFF);
    ASSERT_EQ(scene.commands().size(), 3u);

    scene.clear();
    EXPECT_TRUE(scene.commands().empty());
}

TEST(UIScene, ClearIsReusableAcrossFrames) {
    UIScene scene;
    for (int i = 0; i < 5; ++i) {
        scene.add_filled_rect(0.0f, 0.0f, 1.0f, 1.0f, 0xFFFFFFFF);
    }
    EXPECT_EQ(scene.commands().size(), 5u);

    scene.clear();
    EXPECT_TRUE(scene.commands().empty());

    // Scene stays usable after clear, like a per-frame reset
    scene.add_text("next frame", 1.0f, 2.0f);
    scene.add_circle(3.0f, 4.0f, 5.0f, 0xFF00FF00);
    ASSERT_EQ(scene.commands().size(), 2u);
    EXPECT_TRUE(scene.commands()[0].is<TextCommand>());
    EXPECT_TRUE(scene.commands()[1].is<CircleCommand>());
}

// ============================================================================
// Wrapped text
// ============================================================================

TEST(UIScene, WrappedTextSingleLineFits) {
    UIScene scene;
    // 5 chars * 8px = 40px, well under max_width 100 -> single line
    float height = scene.add_text_wrapped("hello", 10.0f, 20.0f, 100.0f);

    ASSERT_EQ(scene.commands().size(), 1u);
    const auto& text = scene.commands()[0].get<TextCommand>();
    EXPECT_EQ(text.text, "hello");
    EXPECT_FLOAT_EQ(text.x, 10.0f);
    EXPECT_FLOAT_EQ(text.y, 20.0f);
    // Default line height is 18 * scale
    EXPECT_FLOAT_EQ(height, 18.0f);
}

TEST(UIScene, WrappedTextBreaksAtWordBoundary) {
    UIScene scene;
    // char width 8px, max_width 80 -> 10 chars per line.
    // "hello world foo" breaks at the space after "hello".
    float height = scene.add_text_wrapped("hello world foo", 0.0f, 0.0f, 80.0f);

    ASSERT_EQ(scene.commands().size(), 2u);
    const auto& line0 = scene.commands()[0].get<TextCommand>();
    const auto& line1 = scene.commands()[1].get<TextCommand>();
    EXPECT_EQ(line0.text, "hello");
    EXPECT_FLOAT_EQ(line0.y, 0.0f);
    EXPECT_EQ(line1.text, "world foo");
    EXPECT_FLOAT_EQ(line1.y, 18.0f);
    // Both lines share the caller's x
    EXPECT_FLOAT_EQ(line0.x, 0.0f);
    EXPECT_FLOAT_EQ(line1.x, 0.0f);
    EXPECT_FLOAT_EQ(height, 36.0f);
}

TEST(UIScene, WrappedTextHardBreaksWithoutSpaces) {
    UIScene scene;
    // max_width 32 / 8px -> 4 chars per line, no spaces to break at
    float height = scene.add_text_wrapped("abcdefghij", 0.0f, 0.0f, 32.0f);

    ASSERT_EQ(scene.commands().size(), 3u);
    EXPECT_EQ(scene.commands()[0].get<TextCommand>().text, "abcd");
    EXPECT_EQ(scene.commands()[1].get<TextCommand>().text, "efgh");
    EXPECT_EQ(scene.commands()[2].get<TextCommand>().text, "ij");
    EXPECT_FLOAT_EQ(height, 54.0f);
}

TEST(UIScene, WrappedTextEmptyStringProducesNothing) {
    UIScene scene;
    float height = scene.add_text_wrapped("", 0.0f, 0.0f, 100.0f);

    EXPECT_TRUE(scene.commands().empty());
    EXPECT_FLOAT_EQ(height, 0.0f);
}

TEST(UIScene, WrappedTextCustomLineHeight) {
    UIScene scene;
    // 10-char lines, wraps once; explicit line_height overrides the default
    float height = scene.add_text_wrapped("hello world foo", 0.0f, 100.0f, 80.0f, 1.0f, 0xFF0000FF, 25.0f);

    ASSERT_EQ(scene.commands().size(), 2u);
    EXPECT_FLOAT_EQ(scene.commands()[0].get<TextCommand>().y, 100.0f);
    EXPECT_FLOAT_EQ(scene.commands()[1].get<TextCommand>().y, 125.0f);
    EXPECT_FLOAT_EQ(height, 50.0f);
}

TEST(UIScene, WrappedTextPropagatesScaleAndColor) {
    UIScene scene;
    // scale 2 -> char width 16px, default line height 36px.
    // max_width 160 -> 10 chars per line, so it wraps at the space.
    float height = scene.add_text_wrapped("hello world foo", 0.0f, 0.0f, 160.0f, 2.0f, 0x11223344);

    ASSERT_EQ(scene.commands().size(), 2u);
    for (const auto& cmd : scene.commands()) {
        const auto& text = cmd.get<TextCommand>();
        EXPECT_FLOAT_EQ(text.scale, 2.0f);
        EXPECT_EQ(text.color, 0x11223344u);
    }
    EXPECT_FLOAT_EQ(scene.commands()[1].get<TextCommand>().y, 36.0f);
    EXPECT_FLOAT_EQ(height, 72.0f);
}

TEST(UIScene, WrappedTextTinyMaxWidthClampsToOneCharPerLine) {
    UIScene scene;
    // max_width 4 / 8px -> 0 chars, clamped to a minimum of 1 per line
    float height = scene.add_text_wrapped("ab", 0.0f, 0.0f, 4.0f);

    ASSERT_EQ(scene.commands().size(), 2u);
    EXPECT_EQ(scene.commands()[0].get<TextCommand>().text, "a");
    EXPECT_EQ(scene.commands()[1].get<TextCommand>().text, "b");
    EXPECT_FLOAT_EQ(height, 36.0f);
}
