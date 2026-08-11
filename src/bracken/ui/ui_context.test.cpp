#include "bracken/ui/ui_context.hpp"

#include "bracken/testing/frame_test.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace {

using namespace bracken::ui;
namespace scene = bracken::scene;

constexpr Rect k_viewport{0.0f, 0.0f, 800.0f, 600.0f};

/// A synthetic font: every printable ASCII codepoint advances 10 px, lines are
/// 20 px. Layout arithmetic is then exact and readable in the assertions —
/// this file tests the UI, not SDL_ttf.
bracken::FontMetrics test_font() {
    bracken::FontMetrics m;
    m.line_height = 20;
    m.line_skip = 22;
    m.pixel_size = 18;
    m.fallback_advance = 10;
    m.advance.fill(bracken::FontMetrics::k_absent);
    for (uint32_t cp = 0x20; cp <= 0x7E; ++cp) {
        m.advance[cp - bracken::FontMetrics::k_first_cp] = 10;
    }
    return m;
}

/// One headless UI frame: build a TestFrame, run `body` against the Context,
/// seal, and hand back the packet for inspection. No GPU, no window, no SDL.
struct Harness {
    bracken::FontMetrics font = test_font();
    Context ui;

    Harness() {
        ContextConfig cfg;
        cfg.font = &font;
        ui.configure(cfg);
    }

    template<typename Fn> std::unique_ptr<bracken::testing::TestFrame> run(const InputFrame& in, Fn&& body) {
        auto frame = std::make_unique<bracken::testing::TestFrame>(1u, scene::FramePacket::k_default_arena_bytes,
                                                                   &font);
        ui.begin(frame->sink().ui(), in, k_viewport);
        body(ui);
        ui.end();
        frame->seal();
        return frame;
    }
};

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

TEST(UiContext, LabelTextReachesThePacketArenaInDeclarationOrder) {
    Harness h;
    const auto frame = h.run(InputFrame{}, [](Context& ui) {
        ui.begin_column({});
        ui.label("first");
        ui.label("second");
        ui.label("third");
        ui.end_box();
    });

    const std::vector<std::string_view> text = frame->ui_text();
    ASSERT_EQ(text.size(), 3u);
    EXPECT_EQ(text[0], "first");
    EXPECT_EQ(text[1], "second");
    EXPECT_EQ(text[2], "third");
}

TEST(UiContext, TemporaryStringsAreCopiedAtDeclaration) {
    // `label(std::format(...))` hands in a view over a temporary that dies at
    // the end of the full expression, long before end() emits. The Context
    // copies into its own arena at declaration time, which is what makes the
    // idiomatic call site safe.
    Harness h;
    const auto frame = h.run(InputFrame{}, [](Context& ui) {
        for (int i = 0; i < 3; ++i) {
            ui.label(std::string("row ") + std::to_string(i));
        }
    });
    const std::vector<std::string_view> text = frame->ui_text();
    ASSERT_EQ(text.size(), 3u);
    EXPECT_EQ(text[0], "row 0");
    EXPECT_EQ(text[2], "row 2");
}

TEST(UiContext, PanelBackgroundIsEmittedBeforeItsChildren) {
    // The z-order contract, from the emitting side: the panel's fill and
    // border precede every child command, so the renderer's submission-order
    // walk paints them behind. (Before the UI pass stopped drawing all
    // geometry before all text, no emission order could achieve this.)
    Harness h;
    const auto frame = h.run(InputFrame{}, [](Context& ui) {
        ui.begin_panel({});
        ui.label("inside");
        ui.end_panel();
    });

    const std::span<const scene::UiCommand> cmds = frame->ui_commands();
    ASSERT_GE(cmds.size(), 3u);
    EXPECT_TRUE(cmds[0].is<scene::UiFilledRectCommand>());
    EXPECT_TRUE(cmds[1].is<scene::UiRectOutlineCommand>());
    EXPECT_TRUE(cmds[2].is<scene::UiTextCommand>());
}

TEST(UiContext, APanelDeclaredAfterALabelIsEmittedAfterIt) {
    // The other half of the z-order contract, and the half that was
    // STRUCTURALLY IMPOSSIBLE before B3: a panel declared after a label must
    // reach the packet after it, so the renderer's submission-order walk
    // paints it OVER the label. The old UI pass drew every primitive and then
    // every queued text, so this ordering could be emitted and then ignored —
    // which is why this pairs with
    // UIRendererOps.GeometryRecordedAfterTextLandsInALaterRange, the consuming
    // half of the same claim.
    Harness h;
    const auto frame = h.run(InputFrame{}, [](Context& ui) {
        ui.label("underneath");
        ui.begin_panel({});
        ui.label("on top");
        ui.end_panel();
    });

    const std::span<const scene::UiCommand> cmds = frame->ui_commands();
    ASSERT_GE(cmds.size(), 4u);
    EXPECT_TRUE(cmds[0].is<scene::UiTextCommand>());        // the label
    EXPECT_TRUE(cmds[1].is<scene::UiFilledRectCommand>());  // the panel fill, ON TOP of it
    EXPECT_TRUE(cmds[2].is<scene::UiRectOutlineCommand>()); // its border
    EXPECT_TRUE(cmds[3].is<scene::UiTextCommand>());        // the panel's own label
}

TEST(UiContext, TwoIdenticalFramesEmitAnIdenticalCommandStream) {
    Harness h;
    const InputFrame in{};
    const auto build = [](Context& ui) {
        ui.begin_panel({});
        ui.label("hud");
        (void)ui.button(id("go"), "Go");
        ui.end_panel();
    };

    const auto a = h.run(in, build);
    const auto b = h.run(in, build);

    const auto ca = a->ui_commands();
    const auto cb = b->ui_commands();
    ASSERT_EQ(ca.size(), cb.size());
    for (std::size_t i = 0; i < ca.size(); ++i) {
        EXPECT_EQ(ca[i].data.index(), cb[i].data.index()) << "command " << i << " changed kind between frames";
    }
    // Rectangles are identical too — the solve is a pure function of the tree.
    ASSERT_GE(ca.size(), 1u);
    const auto& ra = ca[0].get<scene::UiFilledRectCommand>();
    const auto& rb = cb[0].get<scene::UiFilledRectCommand>();
    EXPECT_FLOAT_EQ(ra.x, rb.x);
    EXPECT_FLOAT_EQ(ra.y, rb.y);
    EXPECT_FLOAT_EQ(ra.w, rb.w);
    EXPECT_FLOAT_EQ(ra.h, rb.h);
}

TEST(UiContext, TextWithoutAFontIsCountedAndStillEmits) {
    Context ui; // configured with no FontMetrics at all
    bracken::testing::TestFrame frame;
    ui.begin(frame.sink().ui(), InputFrame{}, k_viewport);
    ui.label("no font here");
    ui.end();
    frame.seal();

    EXPECT_GT(ui.diagnostics().text_without_font, 0u);
    // The text still reaches the packet: measuring is what failed, not
    // drawing, and a silently missing label is the harder bug.
    ASSERT_EQ(frame.ui_text().size(), 1u);
    EXPECT_EQ(frame.ui_text()[0], "no font here");
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

TEST(UiContext, ButtonFiresExactlyOnceOnAReleaseInsideItsRect) {
    Harness h;
    bool clicked = false;
    const auto build = [&clicked](Context& ui) {
        ui.begin_column({});
        clicked = ui.button(id("ok"), "OK", Size::px(100.0f), Size::px(30.0f));
        ui.end_box();
    };

    // Frame 1: the widget has no recorded rectangle yet, so it cannot be
    // clicked. This is the documented first-frame rule.
    InputFrame in{};
    in.mouse = {50.0f, 15.0f};
    in.mouse_pressed = true;
    in.mouse_down = true;
    (void)h.run(in, build);
    EXPECT_FALSE(clicked);
    EXPECT_FLOAT_EQ(h.ui.rect_of(id("ok")).w, 100.0f);

    // Frame 2: press inside. Not a click yet — the release is what fires.
    (void)h.run(in, build);
    EXPECT_FALSE(clicked);
    EXPECT_EQ(h.ui.active(), id("ok"));

    // Frame 3: release inside → fires exactly once.
    InputFrame up{};
    up.mouse = {50.0f, 15.0f};
    up.mouse_released = true;
    (void)h.run(up, build);
    EXPECT_TRUE(clicked);
    EXPECT_FALSE(h.ui.active().valid());

    // Frame 4: nothing happening → does not fire again.
    (void)h.run(InputFrame{}, build);
    EXPECT_FALSE(clicked);
}

TEST(UiContext, ButtonDoesNotFireWhenTheReleaseLandsOutside) {
    Harness h;
    bool clicked = false;
    const auto build = [&clicked](Context& ui) {
        clicked = ui.button(id("ok"), "OK", Size::px(100.0f), Size::px(30.0f));
    };

    InputFrame press{};
    press.mouse = {50.0f, 15.0f};
    press.mouse_pressed = true;
    (void)h.run(press, build); // frame 1: records the rect
    (void)h.run(press, build); // frame 2: press lands, active latches
    EXPECT_EQ(h.ui.active(), id("ok"));

    InputFrame release_away{};
    release_away.mouse = {600.0f, 400.0f}; // dragged off the button
    release_away.mouse_released = true;
    (void)h.run(release_away, build);
    EXPECT_FALSE(clicked);
    EXPECT_FALSE(h.ui.active().valid()); // the drag still ends the press
}

TEST(UiContext, HotAndActiveSurviveAcrossFramesAndActiveClearsOnRelease) {
    Harness h;
    const auto build = [](Context& ui) { (void)ui.button(id("btn"), "Btn", Size::px(80.0f), Size::px(24.0f)); };

    InputFrame hover{};
    hover.mouse = {40.0f, 12.0f};
    (void)h.run(hover, build); // frame 1: no rect yet
    EXPECT_FALSE(h.ui.hot().valid());

    (void)h.run(hover, build); // frame 2: hovering
    EXPECT_EQ(h.ui.hot(), id("btn"));
    EXPECT_FALSE(h.ui.active().valid());

    InputFrame press = hover;
    press.mouse_pressed = true;
    (void)h.run(press, build);
    EXPECT_EQ(h.ui.active(), id("btn"));

    // Held across a frame with no new edges.
    InputFrame held = hover;
    held.mouse_down = true;
    (void)h.run(held, build);
    EXPECT_EQ(h.ui.active(), id("btn"));

    InputFrame up = hover;
    up.mouse_released = true;
    (void)h.run(up, build);
    EXPECT_FALSE(h.ui.active().valid());
}

TEST(UiContext, DuplicateWidgetIdsAreCounted) {
    Harness h;
    (void)h.run(InputFrame{}, [](Context& ui) {
        (void)ui.button(id("same"), "A");
        (void)ui.button(id("same"), "B");
    });
    EXPECT_EQ(h.ui.diagnostics().duplicate_widget_id, 1u);

    // Scoping them apart makes them distinct again.
    (void)h.run(InputFrame{}, [](Context& ui) {
        ui.push_id(id("left"));
        (void)ui.button(id("same"), "A");
        ui.pop_id();
        ui.push_id(id("right"));
        (void)ui.button(id("same"), "B");
        ui.pop_id();
    });
    EXPECT_EQ(h.ui.diagnostics().duplicate_widget_id, 0u);
}

TEST(UiContext, ToggleFlipsOnceAndSliderClampsAndReports) {
    Harness h;
    bool value = false;
    float volume = 0.5f;
    bool toggled = false;
    bool moved = false;
    const auto build = [&](Context& ui) {
        ui.begin_column({});
        toggled = ui.toggle(id("mute"), "Mute", value, Size::px(200.0f));
        moved = ui.slider(id("vol"), volume, 0.0f, 1.0f, Size::px(200.0f));
        ui.end_box();
    };

    (void)h.run(InputFrame{}, build); // frame 1: record rects
    EXPECT_FALSE(toggled);
    EXPECT_FALSE(value);

    const Rect toggle_rect = h.ui.rect_of(id("mute"));
    ASSERT_GT(toggle_rect.w, 0.0f);

    InputFrame press{};
    press.mouse = toggle_rect.center();
    press.mouse_pressed = true;
    (void)h.run(press, build);
    EXPECT_FALSE(toggled); // press alone does not flip

    InputFrame release{};
    release.mouse = toggle_rect.center();
    release.mouse_released = true;
    (void)h.run(release, build);
    EXPECT_TRUE(toggled);
    EXPECT_TRUE(value);

    // Slider: drag to the far right of its recorded rect.
    const Rect slider_rect = h.ui.rect_of(id("vol"));
    ASSERT_GT(slider_rect.w, 0.0f);
    InputFrame grab{};
    grab.mouse = slider_rect.center();
    grab.mouse_pressed = true;
    (void)h.run(grab, build);
    InputFrame drag{};
    drag.mouse = {slider_rect.right() + 500.0f, slider_rect.center().y};
    drag.mouse_down = true;
    (void)h.run(drag, build);
    EXPECT_TRUE(moved);
    EXPECT_FLOAT_EQ(volume, 1.0f); // clamped, not overshooting past max
}

// ---------------------------------------------------------------------------
// Focus
// ---------------------------------------------------------------------------

TEST(UiContext, FocusOrderIsDeclarationOrderAndTabWrapsAround) {
    Harness h;
    const auto build = [](Context& ui) {
        ui.begin_column({});
        (void)ui.button(id("a"), "A", Size::px(80.0f), Size::px(20.0f));
        (void)ui.button(id("b"), "B", Size::px(80.0f), Size::px(20.0f));
        (void)ui.button(id("c"), "C", Size::px(80.0f), Size::px(20.0f));
        ui.end_box();
    };

    InputFrame tab{};
    tab.nav_next = true;

    (void)h.run(tab, build); // no focus yet -> lands on the first
    EXPECT_EQ(h.ui.focused(), id("a"));
    const std::vector<WidgetId> order = h.ui.focus_order();
    ASSERT_EQ(order.size(), 3u);
    EXPECT_EQ(order[0], id("a"));
    EXPECT_EQ(order[1], id("b"));
    EXPECT_EQ(order[2], id("c"));

    (void)h.run(tab, build);
    EXPECT_EQ(h.ui.focused(), id("b"));
    (void)h.run(tab, build);
    EXPECT_EQ(h.ui.focused(), id("c"));
    (void)h.run(tab, build);
    EXPECT_EQ(h.ui.focused(), id("a")); // wrapped

    // Shift+Tab wraps backwards.
    InputFrame back{};
    back.nav_prev = true;
    (void)h.run(back, build);
    EXPECT_EQ(h.ui.focused(), id("c"));
}

TEST(UiContext, FocusDoesNotWrapWhenWrapFocusIsOff) {
    Harness h;
    ContextConfig cfg;
    cfg.font = &h.font;
    cfg.wrap_focus = false;
    h.ui.configure(cfg);

    const auto build = [](Context& ui) {
        (void)ui.button(id("a"), "A", Size::px(80.0f), Size::px(20.0f));
        (void)ui.button(id("b"), "B", Size::px(80.0f), Size::px(20.0f));
    };
    InputFrame tab{};
    tab.nav_next = true;
    (void)h.run(tab, build);
    (void)h.run(tab, build);
    EXPECT_EQ(h.ui.focused(), id("b"));
    (void)h.run(tab, build);
    EXPECT_EQ(h.ui.focused(), id("b")); // stuck at the end, as configured
}

TEST(UiContext, DirectionalNavMovesDownAColumnAndAcrossARow) {
    Harness h;
    const auto build = [](Context& ui) {
        ui.begin_column({});
        ui.begin_row({});
        (void)ui.button(id("tl"), "TL", Size::px(80.0f), Size::px(20.0f));
        (void)ui.button(id("tr"), "TR", Size::px(80.0f), Size::px(20.0f));
        ui.end_box();
        ui.begin_row({});
        (void)ui.button(id("bl"), "BL", Size::px(80.0f), Size::px(20.0f));
        (void)ui.button(id("br"), "BR", Size::px(80.0f), Size::px(20.0f));
        ui.end_box();
        ui.end_box();
    };

    (void)h.run(InputFrame{}, build);
    h.ui.set_focus(id("tl"));

    InputFrame right{};
    right.nav_right = true;
    (void)h.run(right, build);
    EXPECT_EQ(h.ui.focused(), id("tr"));

    InputFrame down{};
    down.nav_down = true;
    (void)h.run(down, build);
    EXPECT_EQ(h.ui.focused(), id("br"));

    InputFrame left{};
    left.nav_left = true;
    (void)h.run(left, build);
    EXPECT_EQ(h.ui.focused(), id("bl"));
}

TEST(UiContext, GamepadActivateFiresTheFocusedButtonNotTheHoveredOne) {
    Harness h;
    bool a_fired = false;
    bool b_fired = false;
    const auto build = [&](Context& ui) {
        ui.begin_column({});
        a_fired = ui.button(id("a"), "A", Size::px(80.0f), Size::px(20.0f));
        b_fired = ui.button(id("b"), "B", Size::px(80.0f), Size::px(20.0f));
        ui.end_box();
    };

    (void)h.run(InputFrame{}, build);
    h.ui.set_focus(id("b"));

    InputFrame activate{};
    activate.nav_activate = true;
    // The pointer sits over A, but B has focus.
    activate.mouse = h.ui.rect_of(id("a")).center();
    (void)h.run(activate, build);
    EXPECT_FALSE(a_fired);
    EXPECT_TRUE(b_fired);
}

// ---------------------------------------------------------------------------
// Scrolling and clipping
// ---------------------------------------------------------------------------

TEST(UiContext, ScrollViewEmitsAClipAroundItsBodyAndRestoresAfter) {
    Harness h;
    const auto frame = h.run(InputFrame{}, [](Context& ui) {
        BoxStyle s{};
        s.width = Size::px(200.0f);
        s.height = Size::px(100.0f);
        ui.begin_scroll(id("list"), s);
        for (int i = 0; i < 10; ++i) {
            ui.label("row");
        }
        ui.end_scroll();
    });

    const std::vector<scene::UiClipCommand> clips = frame->ui_clips();
    ASSERT_EQ(clips.size(), 2u);
    EXPECT_EQ(clips[0].enabled, 1u);
    EXPECT_FLOAT_EQ(clips[0].w, 200.0f);
    EXPECT_FLOAT_EQ(clips[0].h, 100.0f);
    EXPECT_EQ(clips[1].enabled, 0u); // no enclosing clip -> full screen again

    // The clip brackets the body: the first clip precedes every text command
    // and the second follows them.
    const auto cmds = frame->ui_commands();
    std::size_t first_clip = cmds.size();
    std::size_t last_clip = 0;
    std::size_t first_text = cmds.size();
    std::size_t last_text = 0;
    for (std::size_t i = 0; i < cmds.size(); ++i) {
        if (cmds[i].is<scene::UiClipCommand>()) {
            first_clip = std::min(first_clip, i);
            last_clip = std::max(last_clip, i);
        } else if (cmds[i].is<scene::UiTextCommand>()) {
            first_text = std::min(first_text, i);
            last_text = std::max(last_text, i);
        }
    }
    EXPECT_LT(first_clip, first_text);
    EXPECT_GT(last_clip, last_text);
}

TEST(UiContext, NestedScrollRestoresTheEnclosingClipNotTheFullScreen) {
    Harness h;
    const auto frame = h.run(InputFrame{}, [](Context& ui) {
        BoxStyle outer{};
        outer.width = Size::px(300.0f);
        outer.height = Size::px(200.0f);
        BoxStyle inner{};
        inner.width = Size::px(100.0f);
        inner.height = Size::px(50.0f);
        ui.begin_scroll(id("outer"), outer);
        ui.begin_scroll(id("inner"), inner);
        ui.label("deep");
        ui.end_scroll();
        ui.label("shallow");
        ui.end_scroll();
    });

    const std::vector<scene::UiClipCommand> clips = frame->ui_clips();
    ASSERT_EQ(clips.size(), 4u);
    EXPECT_EQ(clips[0].enabled, 1u); // outer push
    EXPECT_EQ(clips[1].enabled, 1u); // inner push
    EXPECT_EQ(clips[2].enabled, 1u); // inner pop -> back to the OUTER rect
    EXPECT_FLOAT_EQ(clips[2].w, clips[0].w);
    EXPECT_FLOAT_EQ(clips[2].h, clips[0].h);
    EXPECT_EQ(clips[3].enabled, 0u); // outer pop -> full screen
    // The inner clip is intersected with the outer one, never larger.
    EXPECT_LE(clips[1].w, clips[0].w);
    EXPECT_LE(clips[1].h, clips[0].h);
}

TEST(UiContext, ScrollOffsetClampsAndDoesNotMoveWhenTheContentFits) {
    Harness h;
    const auto tall = [](Context& ui) {
        BoxStyle s{};
        s.width = Size::px(200.0f);
        s.height = Size::px(100.0f);
        ui.begin_scroll(id("list"), s);
        for (int i = 0; i < 20; ++i) {
            ui.label("row"); // 20 px each -> 400 px of content
        }
        ui.end_scroll();
    };

    InputFrame wheel_down{};
    wheel_down.mouse = {50.0f, 50.0f}; // inside the viewport
    wheel_down.wheel = -1.0f;          // scroll down

    (void)h.run(InputFrame{}, tall); // establish the viewport
    (void)h.run(wheel_down, tall);
    const auto& scroll = h.ui.state().scroll;
    ASSERT_EQ(scroll.size(), 1u);
    EXPECT_GT(scroll[0].offset.y, 0.0f);

    // Scroll far past the end: clamped to content - viewport = 300.
    for (int i = 0; i < 50; ++i) {
        (void)h.run(wheel_down, tall);
    }
    EXPECT_FLOAT_EQ(h.ui.state().scroll[0].offset.y, 400.0f - 100.0f);

    // And back past the start: clamped to 0.
    InputFrame wheel_up = wheel_down;
    wheel_up.wheel = 1.0f;
    for (int i = 0; i < 50; ++i) {
        (void)h.run(wheel_up, tall);
    }
    EXPECT_FLOAT_EQ(h.ui.state().scroll[0].offset.y, 0.0f);
}

TEST(UiContext, ContentThatFitsNeverScrolls) {
    Harness h;
    const auto shortlist = [](Context& ui) {
        BoxStyle s{};
        s.width = Size::px(200.0f);
        s.height = Size::px(200.0f);
        ui.begin_scroll(id("list"), s);
        ui.label("only row");
        ui.end_scroll();
    };
    InputFrame wheel{};
    wheel.mouse = {50.0f, 50.0f};
    wheel.wheel = -5.0f;
    (void)h.run(InputFrame{}, shortlist);
    (void)h.run(wheel, shortlist);
    ASSERT_EQ(h.ui.state().scroll.size(), 1u);
    EXPECT_FLOAT_EQ(h.ui.state().scroll[0].offset.y, 0.0f);
}

TEST(UiContext, FocusIsScrolledIntoView) {
    Harness h;
    const auto build = [](Context& ui) {
        BoxStyle s{};
        s.width = Size::px(200.0f);
        s.height = Size::px(60.0f); // shows about two rows
        ui.begin_scroll(id("list"), s);
        for (int i = 0; i < 10; ++i) {
            (void)ui.button(id(id("row"), static_cast<uint64_t>(i)), "row", Size::px(150.0f), Size::px(20.0f));
        }
        ui.end_scroll();
    };

    (void)h.run(InputFrame{}, build);
    ASSERT_EQ(h.ui.state().scroll.size(), 1u);
    EXPECT_FLOAT_EQ(h.ui.state().scroll[0].offset.y, 0.0f);

    // Tab down to row 9; the viewport must follow.
    InputFrame tab{};
    tab.nav_next = true;
    for (int i = 0; i < 10; ++i) {
        (void)h.run(tab, build);
    }
    EXPECT_GT(h.ui.state().scroll[0].offset.y, 0.0f);
    // Bounded by the content extent.
    EXPECT_LE(h.ui.state().scroll[0].offset.y, 200.0f - 60.0f);
}

TEST(UiContext, EveryEnclosingScrollContainerBringsTheFocusIntoView) {
    // A widget inside a NESTED scroll view has to be brought into view by BOTH
    // containers. Scrolling only the innermost one leaves the inner window
    // itself off the outer window's screen, and the focused button is invisible
    // while claiming to be focused — the bug an exact "which container is my
    // parent" match produces and an ancestor test does not.
    Harness h;
    const auto build = [](Context& ui) {
        BoxStyle outer{};
        outer.width = Size::px(220.0f);
        outer.height = Size::px(60.0f);
        ui.begin_scroll(id("outer"), outer);

        // Filler so the inner list starts well below the outer window.
        for (int i = 0; i < 6; ++i) {
            ui.label("filler", 0xFFFFFFFF, 1.0f, Size::px(20.0f));
        }

        BoxStyle inner{};
        inner.width = Size::px(200.0f);
        inner.height = Size::px(40.0f);
        ui.begin_scroll(id("inner"), inner);
        for (int i = 0; i < 8; ++i) {
            (void)ui.button(id(id("row"), static_cast<uint64_t>(i)), "row", Size::px(150.0f), Size::px(20.0f));
        }
        ui.end_scroll();
        ui.end_scroll();
    };

    (void)h.run(InputFrame{}, build);
    ASSERT_EQ(h.ui.state().scroll.size(), 2u);

    InputFrame tab{};
    tab.nav_next = true;
    for (int i = 0; i < 8; ++i) {
        (void)h.run(tab, build);
    }
    // The scroll store is a SORTED vector keyed by id, so look the entries up
    // by id rather than assuming an order.
    float outer_off = -1.0f;
    float inner_off = -1.0f;
    for (const auto& e : h.ui.state().scroll) {
        if (e.id == id("outer").value) {
            outer_off = e.offset.y;
        }
        if (e.id == id("inner").value) {
            inner_off = e.offset.y;
        }
    }
    ASSERT_GE(outer_off, 0.0f);
    ASSERT_GE(inner_off, 0.0f);
    EXPECT_GT(inner_off, 0.0f) << "the inner list did not follow the focus";
    EXPECT_GT(outer_off, 0.0f) << "the outer container left the inner list off screen";
}

TEST(UiContext, HitTestingRespectsTheScrollClip) {
    Harness h;
    bool row9_fired = false;
    const auto build = [&](Context& ui) {
        BoxStyle s{};
        s.width = Size::px(200.0f);
        s.height = Size::px(40.0f); // two rows visible
        ui.begin_scroll(id("list"), s);
        for (int i = 0; i < 10; ++i) {
            const bool fired =
                ui.button(id(id("row"), static_cast<uint64_t>(i)), "row", Size::px(150.0f), Size::px(20.0f));
            if (i == 9) {
                row9_fired = fired;
            }
        }
        ui.end_scroll();
    };

    (void)h.run(InputFrame{}, build); // record rects and the scroll viewport
    (void)h.run(InputFrame{}, build); // the clip is now recorded against the entries

    // begin_scroll pushes its own id scope, so a row's real id is the scroll's
    // id folded with the row's. Asserting the composed spelling here is also
    // the test that scoping is what makes two "row 3"s in two lists distinct.
    const WidgetId row9 = combine(id("list"), id(id("row"), 9u));
    // Row 9 is laid out at y = 180, far below the 40 px viewport. Clicking
    // where it WOULD be must not activate it — the clip rejects the hit.
    const Rect r9 = h.ui.rect_of(row9);
    ASSERT_GT(r9.w, 0.0f) << "row 9 must exist in the layout even though it is scrolled out of view";
    ASSERT_GT(r9.y, 40.0f) << "row 9 must be outside the 40px viewport for this test to mean anything";
    InputFrame press{};
    press.mouse = {r9.x + 5.0f, r9.y + 5.0f};
    press.mouse_pressed = true;
    (void)h.run(press, build);
    EXPECT_FALSE(h.ui.active().valid()) << "a click outside the scroll viewport must not latch a clipped row";

    InputFrame release = press;
    release.mouse_pressed = false;
    release.mouse_released = true;
    (void)h.run(release, build);
    EXPECT_FALSE(row9_fired);
}

// ---------------------------------------------------------------------------
// Input capture
// ---------------------------------------------------------------------------

TEST(UiContext, CaptureFlagsMouseOnlyWhenThePointerIsOverAWidget) {
    Harness h;
    const auto build = [](Context& ui) { (void)ui.button(id("b"), "B", Size::px(100.0f), Size::px(30.0f)); };

    InputFrame outside{};
    outside.mouse = {700.0f, 500.0f};
    (void)h.run(outside, build);
    (void)h.run(outside, build);
    EXPECT_FALSE(h.ui.capture().mouse);
    EXPECT_FALSE(h.ui.capture().keyboard);

    InputFrame inside{};
    inside.mouse = {50.0f, 15.0f};
    (void)h.run(inside, build);
    EXPECT_TRUE(h.ui.capture().mouse);
}

TEST(UiContext, CaptureFlagsKeyboardOnlyWhileSomethingHasFocus) {
    Harness h;
    const auto build = [](Context& ui) { (void)ui.button(id("b"), "B", Size::px(100.0f), Size::px(30.0f)); };

    (void)h.run(InputFrame{}, build);
    EXPECT_FALSE(h.ui.capture().keyboard);

    InputFrame tab{};
    tab.nav_next = true;
    (void)h.run(tab, build);
    EXPECT_TRUE(h.ui.capture().keyboard);
}

TEST(UiContext, CapturedInputIsMaskedOutOfTheSnapshotHandedToTheGame) {
    // The worked example, as an assertion: run the UI, then gate the game on
    // capture().mask(...). This is the whole "the UI ate the event" contract.
    Harness h;
    const auto build = [](Context& ui) { (void)ui.button(id("b"), "B", Size::px(100.0f), Size::px(30.0f)); };

    bracken::InputSnapshot raw;
    raw.mouse_x = 50.0f;
    raw.mouse_y = 15.0f;
    raw.mouse_left_pressed = true;

    (void)h.run(InputFrame::from_snapshot(raw), build); // frame 1: record rect
    (void)h.run(InputFrame::from_snapshot(raw), build); // frame 2: pointer is over the button

    ASSERT_TRUE(h.ui.capture().mouse);
    const bracken::InputSnapshot for_game = h.ui.capture().mask(raw);
    EXPECT_FALSE(for_game.mouse_left_pressed) << "the game must not also see the click the UI took";
    EXPECT_FLOAT_EQ(for_game.mouse_x, 50.0f) << "the pointer position is not an event and must survive";
}

// ---------------------------------------------------------------------------
// Retention
// ---------------------------------------------------------------------------

TEST(UiContext, UnstableIdsDoNotGrowTheRetainedStoreWithoutBound) {
    // The named trap on WidgetId: an id built from a frame counter resets the
    // widget every frame and leaks an entry. Retention is bounded so the leak
    // has a ceiling rather than being unbounded.
    Harness h;
    ContextConfig cfg;
    cfg.font = &h.font;
    cfg.retain_frames = 4;
    h.ui.configure(cfg);

    for (int frame = 0; frame < 40; ++frame) {
        (void)h.run(InputFrame{}, [frame](Context& ui) {
            (void)ui.button(id(id("unstable"), static_cast<uint64_t>(frame)), "x", Size::px(10.0f), Size::px(10.0f));
        });
    }
    // Bounded by retain_frames, not by the 40 frames that ran.
    EXPECT_LE(h.ui.diagnostics().retained_entries, 8u);
}

TEST(UiContext, StableIdsKeepTheirStateAcrossManyFrames) {
    Harness h;
    float value = 0.25f;
    const auto build = [&value](Context& ui) {
        BoxStyle s{};
        s.width = Size::px(200.0f);
        s.height = Size::px(100.0f);
        ui.begin_scroll(id("list"), s);
        for (int i = 0; i < 20; ++i) {
            ui.label("row");
        }
        ui.end_scroll();
        (void)ui.slider(id("vol"), value, 0.0f, 1.0f, Size::px(200.0f));
    };

    InputFrame wheel{};
    wheel.mouse = {50.0f, 50.0f};
    wheel.wheel = -1.0f;
    (void)h.run(InputFrame{}, build);
    (void)h.run(wheel, build);
    const float scrolled = h.ui.state().scroll[0].offset.y;
    EXPECT_GT(scrolled, 0.0f);

    for (int i = 0; i < 30; ++i) {
        (void)h.run(InputFrame{}, build);
    }
    ASSERT_EQ(h.ui.state().scroll.size(), 1u);
    EXPECT_FLOAT_EQ(h.ui.state().scroll[0].offset.y, scrolled) << "a stable id must not lose its scroll offset";
}

} // namespace
