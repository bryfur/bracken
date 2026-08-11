#include <gtest/gtest.h>

#include "bracken/core/color.hpp"
#include "bracken/render/ui_renderer.hpp"

#include <cstdint>

using bracken::render::UIRenderer;

// Recording is pure CPU work — no GPU device is touched until execute() — so
// batch accounting (vertex counts, draw-range splitting, text queueing) is
// testable headlessly on an uninitialized renderer.

namespace {

SDL_GPUTexture* fake_texture(uintptr_t id) {
    // Opaque handle used only as a range key; never dereferenced during
    // recording.
    return reinterpret_cast<SDL_GPUTexture*>(id);
}

} // namespace

TEST(UIRendererBatch, StartsEmpty) {
    UIRenderer ui;
    EXPECT_EQ(ui.batch_vertex_count(), 0u);
    EXPECT_EQ(ui.batch_draw_count(), 0u);
    EXPECT_EQ(ui.queued_text_count(), 0u);
}

TEST(UIRendererBatch, FilledRectRecordsSixVerticesInOneRange) {
    UIRenderer ui;
    ui.draw_filled_rect(0.0f, 0.0f, 10.0f, 10.0f, 0xFFFF0000);
    EXPECT_EQ(ui.batch_vertex_count(), 6u);
    EXPECT_EQ(ui.batch_draw_count(), 1u);
}

TEST(UIRendererBatch, SolidPrimitivesShareOneDrawRange) {
    UIRenderer ui;
    ui.draw_filled_rect(0.0f, 0.0f, 10.0f, 10.0f, 0xFFFF0000);
    ui.draw_line(0.0f, 0.0f, 5.0f, 5.0f, 0xFF00FF00);
    ui.draw_circle(20.0f, 20.0f, 5.0f, 0xFF0000FF, 8);
    // rect (6) + line (6) + circle (8 * 3)
    EXPECT_EQ(ui.batch_vertex_count(), 6u + 6u + 24u);
    EXPECT_EQ(ui.batch_draw_count(), 1u);
}

TEST(UIRendererBatch, RectOutlineRecordsFourQuads) {
    UIRenderer ui;
    ui.draw_rect_outline(0.0f, 0.0f, 20.0f, 20.0f, 0xFFFFFFFF);
    EXPECT_EQ(ui.batch_vertex_count(), 24u);
    EXPECT_EQ(ui.batch_draw_count(), 1u);
}

TEST(UIRendererBatch, CircleOutlineRecordsRing) {
    UIRenderer ui;
    ui.draw_circle_outline(0.0f, 0.0f, 10.0f, 0xFFFFFFFF, 2.0f, 12);
    // Two triangles per segment
    EXPECT_EQ(ui.batch_vertex_count(), 12u * 6u);
}

TEST(UIRendererBatch, ThickCircleOutlineFallsBackToFilledCircle) {
    UIRenderer ui;
    // line_width >= radius -> filled circle (one triangle per segment)
    ui.draw_circle_outline(0.0f, 0.0f, 5.0f, 0xFFFFFFFF, 6.0f, 12);
    EXPECT_EQ(ui.batch_vertex_count(), 12u * 3u);
}

TEST(UIRendererBatch, DegenerateLineRecordsNothing) {
    UIRenderer ui;
    ui.draw_line(3.0f, 3.0f, 3.0f, 3.0f, 0xFFFFFFFF);
    EXPECT_EQ(ui.batch_vertex_count(), 0u);
    EXPECT_EQ(ui.batch_draw_count(), 0u);
}

TEST(UIRendererBatch, TexturedQuadSplitsDrawRanges) {
    UIRenderer ui;
    ui.draw_filled_rect(0.0f, 0.0f, 10.0f, 10.0f, 0xFFFF0000);
    ui.draw_textured_quad(fake_texture(0x1), 10.0f, 10.0f, 32.0f, 32.0f);
    ui.draw_filled_rect(50.0f, 50.0f, 10.0f, 10.0f, 0xFF00FF00);
    // solid | textured | solid: three ranges preserving submission order
    EXPECT_EQ(ui.batch_vertex_count(), 18u);
    EXPECT_EQ(ui.batch_draw_count(), 3u);
}

TEST(UIRendererBatch, ConsecutiveQuadsWithSameTextureShareOneRange) {
    UIRenderer ui;
    ui.draw_textured_quad(fake_texture(0x1), 0.0f, 0.0f, 16.0f, 16.0f);
    ui.draw_textured_quad(fake_texture(0x1), 16.0f, 0.0f, 16.0f, 16.0f);
    EXPECT_EQ(ui.batch_vertex_count(), 12u);
    EXPECT_EQ(ui.batch_draw_count(), 1u);
}

TEST(UIRendererBatch, DifferentTexturesSplitRanges) {
    UIRenderer ui;
    ui.draw_textured_quad(fake_texture(0x1), 0.0f, 0.0f, 16.0f, 16.0f);
    ui.draw_textured_quad(fake_texture(0x2), 16.0f, 0.0f, 16.0f, 16.0f);
    EXPECT_EQ(ui.batch_vertex_count(), 12u);
    EXPECT_EQ(ui.batch_draw_count(), 2u);
}

TEST(UIRendererBatch, NullTextureQuadIsSolidAndMerges) {
    UIRenderer ui;
    ui.draw_filled_rect(0.0f, 0.0f, 10.0f, 10.0f, 0xFFFF0000);
    ui.draw_textured_quad(nullptr, 10.0f, 10.0f, 32.0f, 32.0f, 0xFF00FF00);
    EXPECT_EQ(ui.batch_vertex_count(), 12u);
    EXPECT_EQ(ui.batch_draw_count(), 1u);
}

TEST(UIRendererBatch, BeginResetsRecordedState) {
    UIRenderer ui;
    ui.draw_filled_rect(0.0f, 0.0f, 10.0f, 10.0f, 0xFFFF0000);
    ui.draw_textured_quad(fake_texture(0x1), 0.0f, 0.0f, 16.0f, 16.0f);
    ui.draw_text("hello", 0.0f, 0.0f, 0xFFFFFFFF);
    ASSERT_GT(ui.batch_vertex_count(), 0u);

    ui.begin(nullptr);
    EXPECT_EQ(ui.batch_vertex_count(), 0u);
    EXPECT_EQ(ui.batch_draw_count(), 0u);
    EXPECT_EQ(ui.queued_text_count(), 0u);
}

TEST(UIRendererBatch, TextQueuesWithoutVertices) {
    UIRenderer ui;
    ui.draw_text("hello", 0.0f, 0.0f, 0xFFFFFFFF);
    EXPECT_EQ(ui.queued_text_count(), 1u);
    EXPECT_EQ(ui.batch_vertex_count(), 0u);
}

TEST(UIRendererBatch, EmptyTextIsNotQueued) {
    UIRenderer ui;
    ui.draw_text("", 0.0f, 0.0f, 0xFFFFFFFF);
    EXPECT_EQ(ui.queued_text_count(), 0u);
}

TEST(UIRendererBatch, ButtonWithoutFontRecordsShapesOnly) {
    UIRenderer ui;
    // Not initialized: no text renderer, so only the fill + outline quads
    ui.draw_button(0.0f, 0.0f, 100.0f, 30.0f, "OK", 0xFF336699, false);
    EXPECT_EQ(ui.batch_vertex_count(), 30u); // fill (6) + outline (24)
    EXPECT_EQ(ui.batch_draw_count(), 1u);
    EXPECT_EQ(ui.queued_text_count(), 0u);
}

TEST(UIRendererBatch, RecordingIsUnboundedPastLegacyGpuCapacity) {
    UIRenderer ui;
    // 3000 rects = 18000 vertices, past the initial 16384-vertex GPU buffer.
    // Recording must retain everything; execute() grows the GPU buffer to
    // match instead of silently drawing stale contents.
    for (int i = 0; i < 3000; ++i) {
        ui.draw_filled_rect(static_cast<float>(i), 0.0f, 1.0f, 1.0f, 0xFFFFFFFF);
    }
    EXPECT_EQ(ui.batch_vertex_count(), 18000u);
    EXPECT_EQ(ui.batch_draw_count(), 1u);
}

// Packed colors are engine-wide ARGB (0xAARRGGBB), decoded through
// core::unpack_color — these tests pin that the recorded vertex colors match
// the canonical decode exactly. The text and debug-line paths route through
// the same helper, so a byte-order regression here means one there too.

TEST(UIRendererColor, FilledRectVerticesDecodeCanonicalARGB) {
    UIRenderer ui;
    constexpr uint32_t packed = 0x80112233; // a=0x80, r=0x11, g=0x22, b=0x33
    ui.draw_filled_rect(0.0f, 0.0f, 10.0f, 10.0f, packed);

    const glm::vec4 expected = bracken::core::unpack_color(packed);
    EXPECT_FLOAT_EQ(expected.r, 0x11 / 255.0f);
    EXPECT_FLOAT_EQ(expected.g, 0x22 / 255.0f);
    EXPECT_FLOAT_EQ(expected.b, 0x33 / 255.0f);
    EXPECT_FLOAT_EQ(expected.a, 0x80 / 255.0f);

    ASSERT_EQ(ui.batch_vertices().size(), 6u);
    for (const auto& v : ui.batch_vertices()) {
        EXPECT_FLOAT_EQ(v.r, expected.r);
        EXPECT_FLOAT_EQ(v.g, expected.g);
        EXPECT_FLOAT_EQ(v.b, expected.b);
        EXPECT_FLOAT_EQ(v.a, expected.a);
    }
}

TEST(UIRendererColor, TexturedQuadTintDecodesSameAsPrimitives) {
    // L40/M6 pin: the textured-quad tint and the shape primitives must decode
    // the same literal to the same hue.
    UIRenderer ui;
    constexpr uint32_t packed = 0xC0FF8040; // a=0xC0, r=0xFF, g=0x80, b=0x40
    ui.draw_filled_rect(0.0f, 0.0f, 10.0f, 10.0f, packed);
    ui.draw_textured_quad(fake_texture(0x1), 10.0f, 10.0f, 32.0f, 32.0f, packed);

    ASSERT_EQ(ui.batch_vertices().size(), 12u);
    const auto& rect_v = ui.batch_vertices()[0];
    const auto& quad_v = ui.batch_vertices()[6];
    EXPECT_FLOAT_EQ(rect_v.r, quad_v.r);
    EXPECT_FLOAT_EQ(rect_v.g, quad_v.g);
    EXPECT_FLOAT_EQ(rect_v.b, quad_v.b);
    EXPECT_FLOAT_EQ(rect_v.a, quad_v.a);

    const glm::vec4 expected = bracken::core::unpack_color(packed);
    EXPECT_FLOAT_EQ(quad_v.r, expected.r);
    EXPECT_FLOAT_EQ(quad_v.g, expected.g);
    EXPECT_FLOAT_EQ(quad_v.b, expected.b);
    EXPECT_FLOAT_EQ(quad_v.a, expected.a);
}

TEST(UIRendererColor, LineAndCircleDecodeCanonicalARGB) {
    UIRenderer ui;
    constexpr uint32_t packed = 0xFF00FF00; // opaque green in ARGB
    ui.draw_line(0.0f, 0.0f, 5.0f, 5.0f, packed);
    ui.draw_circle(20.0f, 20.0f, 5.0f, packed, 4);

    const glm::vec4 expected = bracken::core::unpack_color(packed);
    EXPECT_FLOAT_EQ(expected.g, 1.0f); // green, not red/blue: byte-order pin
    EXPECT_FLOAT_EQ(expected.r, 0.0f);
    EXPECT_FLOAT_EQ(expected.b, 0.0f);
    for (const auto& v : ui.batch_vertices()) {
        EXPECT_FLOAT_EQ(v.r, expected.r);
        EXPECT_FLOAT_EQ(v.g, expected.g);
        EXPECT_FLOAT_EQ(v.b, expected.b);
        EXPECT_FLOAT_EQ(v.a, expected.a);
    }
}

TEST(UIRendererMeasurement, ZeroBeforeInit) {
    const UIRenderer ui;
    EXPECT_EQ(ui.text_width("hello"), 0);
    EXPECT_EQ(ui.text_width("hello", 2.0f), 0);
    EXPECT_EQ(ui.text_height(), 0);
    EXPECT_EQ(ui.text_renderer(), nullptr);
}

// ---------------------------------------------------------------------------
// Submission-order ops — the z-order fix.
//
// The UI pass used to run two hard phases (every primitive, then every queued
// text), so a panel recorded AFTER a label could never cover it. Recording now
// appends submission-ordered spans and execute() walks them. These tests pin
// the RECORDING half; the GPU half is one loop over the same list.
// ---------------------------------------------------------------------------

TEST(UIRendererOps, RectThenTextThenRectProducesThreeOpsInOrder) {
    UIRenderer ui;
    ui.draw_filled_rect(0, 0, 10, 10, 0xFFFFFFFF);
    ui.draw_text("label", 1, 1, 0xFFFFFFFF);
    ui.draw_filled_rect(0, 0, 10, 10, 0xFF000000);

    ASSERT_EQ(ui.ops().size(), 3u);
    EXPECT_EQ(ui.ops()[0].kind, UIRenderer::Op::Kind::Geometry);
    EXPECT_EQ(ui.ops()[1].kind, UIRenderer::Op::Kind::Text);
    EXPECT_EQ(ui.ops()[2].kind, UIRenderer::Op::Kind::Geometry);
    // The second geometry op opens a NEW range even though the texture did not
    // change — otherwise it would be merged into the range that precedes the
    // text and would paint underneath it.
    EXPECT_EQ(ui.batch_draw_count(), 2u);
    EXPECT_EQ(ui.ops()[2].first, 1u);
    EXPECT_EQ(ui.ops()[2].count, 1u);
}

TEST(UIRendererOps, TheCommonHudShapeIsStillExactlyTwoOps) {
    // A panel followed by its labels — what every HUD does, and what the smoke
    // screenshot draws. It must cost exactly the command stream it always did.
    UIRenderer ui;
    ui.draw_filled_rect(10, 10, 280, 122, 0xB0101418);
    for (int i = 0; i < 6; ++i) {
        ui.draw_text("row", 22, static_cast<float>(20 + i * 20), 0xFFFFFFFF);
    }
    ASSERT_EQ(ui.ops().size(), 2u);
    EXPECT_EQ(ui.ops()[0].kind, UIRenderer::Op::Kind::Geometry);
    EXPECT_EQ(ui.ops()[0].count, 1u);
    EXPECT_EQ(ui.ops()[1].kind, UIRenderer::Op::Kind::Text);
    EXPECT_EQ(ui.ops()[1].count, 6u);
    EXPECT_EQ(ui.batch_draw_count(), 1u);
}

TEST(UIRendererOps, ConsecutiveTextsShareOneOp) {
    UIRenderer ui;
    ui.draw_text("a", 0, 0, 0xFFFFFFFF);
    ui.draw_text("b", 0, 10, 0xFFFFFFFF);
    ui.draw_text("c", 0, 20, 0xFFFFFFFF);
    ASSERT_EQ(ui.ops().size(), 1u);
    EXPECT_EQ(ui.ops()[0].kind, UIRenderer::Op::Kind::Text);
    EXPECT_EQ(ui.ops()[0].count, 3u);
}

TEST(UIRendererOps, BeginClearsTheOpList) {
    UIRenderer ui;
    ui.draw_filled_rect(0, 0, 1, 1, 0xFFFFFFFF);
    ui.draw_text("x", 0, 0, 0xFFFFFFFF);
    ASSERT_FALSE(ui.ops().empty());
    ui.begin(nullptr);
    EXPECT_TRUE(ui.ops().empty());
    EXPECT_EQ(ui.batch_draw_count(), 0u);
    EXPECT_EQ(ui.queued_text_count(), 0u);
}

// ---------------------------------------------------------------------------
// Clipping
// ---------------------------------------------------------------------------

TEST(UIRendererClip, SettingAClipOpensANewDrawRange) {
    UIRenderer ui;
    ui.draw_filled_rect(0, 0, 10, 10, 0xFFFFFFFF);
    EXPECT_EQ(ui.batch_draw_count(), 1u);
    ui.set_clip_rect(5, 5, 50, 50);
    ui.draw_filled_rect(0, 0, 10, 10, 0xFFFFFFFF);
    EXPECT_EQ(ui.batch_draw_count(), 2u);
}

TEST(UIRendererClip, TwoConsecutiveIdenticalClipsDoNotSplit) {
    UIRenderer ui;
    ui.set_clip_rect(5, 5, 50, 50);
    ui.draw_filled_rect(0, 0, 10, 10, 0xFFFFFFFF);
    ui.set_clip_rect(5, 5, 50, 50); // same rect
    ui.draw_filled_rect(0, 0, 10, 10, 0xFFFFFFFF);
    EXPECT_EQ(ui.batch_draw_count(), 1u);
}

TEST(UIRendererClip, ClearingTheClipOpensANewRange) {
    UIRenderer ui;
    ui.set_clip_rect(5, 5, 50, 50);
    ui.draw_filled_rect(0, 0, 10, 10, 0xFFFFFFFF);
    ui.clear_clip_rect();
    ui.draw_filled_rect(0, 0, 10, 10, 0xFFFFFFFF);
    EXPECT_EQ(ui.batch_draw_count(), 2u);
}

TEST(UIRendererClip, NoClipCostsNoExtraRanges) {
    // The standing gate: an unclipped HUD must batch exactly as it did before
    // clipping existed.
    UIRenderer ui;
    for (int i = 0; i < 20; ++i) {
        ui.draw_filled_rect(0, 0, 10, 10, 0xFFFFFFFF);
    }
    EXPECT_EQ(ui.batch_draw_count(), 1u);
}

// ---------------------------------------------------------------------------
// UI texture registry (the sim stage's only way to name a texture)
// ---------------------------------------------------------------------------

TEST(UIRendererTextures, RegistrationRoundTripsAndIdsAreNeverRecycled) {
    UIRenderer ui;
    const bracken::UiTextureId a = ui.register_texture(fake_texture(0x100));
    const bracken::UiTextureId b = ui.register_texture(fake_texture(0x200));
    ASSERT_TRUE(a.valid());
    ASSERT_TRUE(b.valid());
    EXPECT_NE(a.value, b.value);
    EXPECT_EQ(ui.texture_for(a), fake_texture(0x100));
    EXPECT_EQ(ui.texture_for(b), fake_texture(0x200));

    ui.unregister_texture(a);
    EXPECT_EQ(ui.texture_for(a), nullptr);
    // The next registration must NOT reuse a's slot: a stale id has to resolve
    // to nothing (diagnosable) rather than to somebody else's texture.
    const bracken::UiTextureId c = ui.register_texture(fake_texture(0x300));
    EXPECT_NE(c.value, a.value);
    EXPECT_EQ(ui.texture_for(a), nullptr);
    EXPECT_EQ(ui.texture_for(c), fake_texture(0x300));
}

TEST(UIRendererTextures, InvalidAndUnknownIdsResolveToNull) {
    UIRenderer ui;
    EXPECT_EQ(ui.texture_for(bracken::UiTextureId{}), nullptr);
    EXPECT_EQ(ui.texture_for(bracken::UiTextureId{999}), nullptr);
    EXPECT_FALSE(ui.register_texture(nullptr).valid());
}

TEST(UIRendererTextures, AnUnresolvedTextureDegradesToASolidTintQuad) {
    // The consume side counts this in FrameDiagnostics::ui_textures_unresolved
    // and hands the null through; the renderer must then draw a solid quad
    // rather than skipping the draw.
    UIRenderer ui;
    ui.draw_textured_quad(nullptr, 0, 0, 10, 10, 0xFF804020);
    EXPECT_EQ(ui.batch_vertex_count(), 6u);
    ASSERT_EQ(ui.batch_draw_count(), 1u);
}
