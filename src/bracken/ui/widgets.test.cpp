// A MENU, built entirely out of the shipped widgets, driven headlessly.
//
// This file is the "does the vocabulary actually compose" test: panel +
// column + label + button + toggle + slider + scroll view, navigated by
// gamepad, with the game's input gated on the capture. If a widget cannot be
// used this way it is not finished, whatever its own unit test says.

#include "bracken/testing/frame_test.hpp"
#include "bracken/ui/ui_context.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace {

using namespace bracken::ui;
namespace scene = bracken::scene;

constexpr Rect k_viewport{0.0f, 0.0f, 800.0f, 600.0f};

bracken::FontMetrics test_font() {
    bracken::FontMetrics m;
    m.line_height = 20;
    m.pixel_size = 18;
    m.fallback_advance = 10;
    m.advance.fill(bracken::FontMetrics::k_absent);
    for (uint32_t cp = 0x20; cp <= 0x7E; ++cp) {
        m.advance[cp - bracken::FontMetrics::k_first_cp] = 10;
    }
    return m;
}

/// The menu under test. A real one: it owns its settings, reports what the
/// player pressed, and is declared exactly the way a game would declare it.
struct PauseMenu {
    bool fullscreen = false;
    float volume = 0.5f;
    bool resume_pressed = false;
    bool quit_pressed = false;

    void declare(Context& ui) {
        resume_pressed = false;
        quit_pressed = false;

        BoxStyle panel{};
        panel.padding = Edges::all(12.0f);
        panel.spacing = 6.0f;
        panel.width = Size::px(320.0f);

        ui.begin_panel(panel);
        ui.label("Paused");
        resume_pressed = ui.button(id("resume"), "Resume", Size::flex(1.0f), Size::px(28.0f));
        (void)ui.toggle(id("fullscreen"), "Fullscreen", fullscreen, Size::flex(1.0f));
        (void)ui.slider(id("volume"), volume, 0.0f, 1.0f, Size::flex(1.0f));
        quit_pressed = ui.button(id("quit"), "Quit", Size::flex(1.0f), Size::px(28.0f));
        ui.end_panel();
    }
};

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

TEST(UiWidgets, MenuLaysOutAsAColumnInsideItsPanel) {
    Harness h;
    PauseMenu menu;
    const auto frame = h.run(InputFrame{}, [&menu](Context& ui) { menu.declare(ui); });

    const Rect resume = h.ui.rect_of(id("resume"));
    const Rect fullscreen = h.ui.rect_of(id("fullscreen"));
    const Rect volume = h.ui.rect_of(id("volume"));
    const Rect quit = h.ui.rect_of(id("quit"));

    // Stacked, in declaration order, none overlapping.
    EXPECT_LT(resume.bottom(), fullscreen.y + 0.001f);
    EXPECT_LT(fullscreen.bottom(), volume.y + 0.001f);
    EXPECT_LT(volume.bottom(), quit.y + 0.001f);
    // Flex width fills the panel's content box: 320 minus 12 px of padding
    // each side.
    EXPECT_FLOAT_EQ(resume.w, 320.0f - 24.0f);
    EXPECT_FLOAT_EQ(quit.w, 320.0f - 24.0f);
    // The panel background is the first thing emitted (it must paint behind).
    const auto cmds = frame->ui_commands();
    ASSERT_GE(cmds.size(), 1u);
    EXPECT_TRUE(cmds[0].is<scene::UiFilledRectCommand>());
    EXPECT_FLOAT_EQ(cmds[0].get<scene::UiFilledRectCommand>().w, 320.0f);
}

TEST(UiWidgets, MenuIsFullyNavigableByGamepad) {
    Harness h;
    PauseMenu menu;
    const auto build = [&menu](Context& ui) { menu.declare(ui); };

    (void)h.run(InputFrame{}, build); // establish rects

    InputFrame down{};
    down.nav_down = true;
    (void)h.run(down, build); // no focus yet -> first focusable = Resume
    EXPECT_EQ(h.ui.focused(), id("resume"));

    (void)h.run(down, build);
    EXPECT_EQ(h.ui.focused(), id("fullscreen"));

    // Activate the toggle from the pad — no pointer anywhere near it.
    InputFrame activate{};
    activate.nav_activate = true;
    activate.mouse = {-100.0f, -100.0f};
    (void)h.run(activate, build);
    EXPECT_TRUE(menu.fullscreen);

    // Down to the slider, then Right to raise the volume.
    (void)h.run(down, build);
    EXPECT_EQ(h.ui.focused(), id("volume"));
    const float before = menu.volume;
    InputFrame right{};
    right.nav_right = true;
    (void)h.run(right, build);
    EXPECT_GT(menu.volume, before);
    EXPECT_LE(menu.volume, 1.0f);

    // Down to Quit and fire it.
    (void)h.run(down, build);
    EXPECT_EQ(h.ui.focused(), id("quit"));
    (void)h.run(activate, build);
    EXPECT_TRUE(menu.quit_pressed);
    EXPECT_FALSE(menu.resume_pressed);
}

TEST(UiWidgets, MenuMouseAndPadAgreeOnWhatFired) {
    Harness h;
    PauseMenu menu;
    const auto build = [&menu](Context& ui) { menu.declare(ui); };

    (void)h.run(InputFrame{}, build);
    const Rect resume = h.ui.rect_of(id("resume"));
    ASSERT_GT(resume.w, 0.0f);

    InputFrame press{};
    press.mouse = resume.center();
    press.mouse_pressed = true;
    (void)h.run(press, build);
    EXPECT_FALSE(menu.resume_pressed);

    InputFrame release{};
    release.mouse = resume.center();
    release.mouse_released = true;
    (void)h.run(release, build);
    EXPECT_TRUE(menu.resume_pressed);
    EXPECT_FALSE(menu.quit_pressed);
}

TEST(UiWidgets, MenuCapturesInputSoTheGameDoesNotAlsoAct) {
    // The worked example from docs/gap_ui.md, asserted.
    Harness h;
    PauseMenu menu;
    const auto build = [&menu](Context& ui) { menu.declare(ui); };

    (void)h.run(InputFrame{}, build);
    const Rect resume = h.ui.rect_of(id("resume"));

    // HOVER only — no click yet. A click would (correctly) also take focus,
    // and this half of the test is about the mouse channel alone.
    bracken::InputSnapshot raw;
    raw.mouse_x = resume.center().x;
    raw.mouse_y = resume.center().y;
    raw.mouse_left_held = true;
    raw.key_just_pressed[SDL_SCANCODE_W] = true;
    raw.keys_held[SDL_SCANCODE_W] = true;
    raw.move_forward = true;

    (void)h.run(InputFrame::from_snapshot(raw), build);
    ASSERT_TRUE(h.ui.capture().mouse);

    const bracken::InputSnapshot for_game = h.ui.capture().mask(raw);
    EXPECT_FALSE(for_game.mouse_left_held);
    // Nothing has focus yet, so the keyboard is NOT captured and the player
    // keeps walking. That is the intended granularity: hovering a button must
    // not freeze movement.
    EXPECT_FALSE(h.ui.capture().keyboard);
    EXPECT_TRUE(for_game.move_forward);

    // Once a widget takes focus the keyboard goes to the UI too.
    InputFrame tab{};
    tab.nav_next = true;
    (void)h.run(tab, build);
    ASSERT_TRUE(h.ui.capture().keyboard);
    const bracken::InputSnapshot gated = h.ui.capture().mask(raw);
    EXPECT_FALSE(gated.move_forward);
    EXPECT_FALSE(gated.keys_held[SDL_SCANCODE_W]);
}

TEST(UiWidgets, NineSlicePanelEmitsOneTexturedQuadPerPatch) {
    Harness h;
    const bracken::UiTextureId tex{7};
    NineSlice slice;
    slice.texture_size = {32.0f, 32.0f};
    slice.border = Edges::all(8.0f);

    const auto frame = h.run(InputFrame{}, [tex, &slice](Context& ui) {
        BoxStyle s{};
        s.padding = Edges::all(10.0f);
        s.width = Size::px(200.0f);
        s.height = Size::px(120.0f);
        ui.begin_panel_sprite(tex, slice, s);
        ui.label("framed");
        ui.end_panel();
    });

    const std::vector<scene::UiTexturedQuadCommand> quads = frame->ui_textured_quads();
    ASSERT_EQ(quads.size(), 9u);
    for (const auto& q : quads) {
        EXPECT_EQ(q.texture.value, 7u);
        EXPECT_GE(q.u0, 0.0f);
        EXPECT_LE(q.u1, 1.0f);
    }
    // The patches tile the panel exactly.
    float area = 0.0f;
    for (const auto& q : quads) {
        area += q.w * q.h;
    }
    EXPECT_FLOAT_EQ(area, 200.0f * 120.0f);
    // And the label still lands inside, after the sprite.
    ASSERT_EQ(frame->ui_text().size(), 1u);
    EXPECT_EQ(frame->ui_text()[0], "framed");
}

TEST(UiWidgets, ImageAndRectEmitAtTheirDeclaredSize) {
    Harness h;
    const bracken::UiTextureId tex{3};
    const auto frame = h.run(InputFrame{}, [tex](Context& ui) {
        ui.begin_row({});
        ui.image(tex, {64.0f, 48.0f});
        ui.spacer(10.0f);
        ui.rect({20.0f, 20.0f}, bracken::core::Color{0xFF00FF00});
        ui.end_box();
    });

    const auto quads = frame->ui_textured_quads();
    ASSERT_EQ(quads.size(), 1u);
    EXPECT_FLOAT_EQ(quads[0].w, 64.0f);
    EXPECT_FLOAT_EQ(quads[0].h, 48.0f);
    EXPECT_EQ(quads[0].texture.value, 3u);

    // The spacer occupies 10 px between the image and the rect.
    const auto cmds = frame->ui_commands();
    const scene::UiFilledRectCommand* fill = nullptr;
    for (const auto& c : cmds) {
        if (c.is<scene::UiFilledRectCommand>()) {
            fill = &c.get<scene::UiFilledRectCommand>();
        }
    }
    ASSERT_NE(fill, nullptr);
    EXPECT_FLOAT_EQ(fill->x, 64.0f + 10.0f);
    EXPECT_FLOAT_EQ(fill->w, 20.0f);
}

TEST(UiWidgets, WrappedLabelBreaksOnTheRealMetrics) {
    // Ten characters at 10 px each; a 55 px box fits five per line, and the
    // break lands on a space rather than mid-word.
    Harness h;
    const auto frame = h.run(InputFrame{}, [](Context& ui) { ui.label_wrapped("aaa bbb ccc", 55.0f); });

    const std::vector<std::string_view> lines = frame->ui_text();
    ASSERT_GE(lines.size(), 2u);
    for (const std::string_view& line : lines) {
        EXPECT_LE(line.size(), 5u) << "line '" << line << "' is wider than the box";
        EXPECT_NE(line.front(), ' ') << "a wrapped line must not start with the break space";
    }
    std::string joined;
    for (const std::string_view& line : lines) {
        if (!joined.empty()) {
            joined += ' ';
        }
        joined.append(line);
    }
    EXPECT_EQ(joined, "aaa bbb ccc") << "wrapping must not lose or duplicate text";
}

// ---------------------------------------------------------------------------
// The second-pass widgets: radio, text field, progress bar, separator.
// ---------------------------------------------------------------------------

/// Click at `p`: a press frame followed by a release frame. Two frames,
/// because hit testing uses LAST frame's rectangle — a widget declared for the
/// first time is not clickable until the frame after.
InputFrame press_at(glm::vec2 p) {
    InputFrame in{};
    in.mouse = p;
    in.mouse_down = true;
    in.mouse_pressed = true;
    return in;
}
InputFrame release_at(glm::vec2 p) {
    InputFrame in{};
    in.mouse = p;
    in.mouse_released = true;
    return in;
}

/// A two-option radio group, declared identically on every frame.
struct RadioGroup {
    int choice = 0;
    bool changed = false;

    void declare(Context& ui) {
        BoxStyle col{};
        col.spacing = 4.0f;
        ui.begin_column(col);
        const bool a = ui.radio(id("opt.a"), "Low", choice, 0, Size::px(200.0f));
        const bool b = ui.radio(id("opt.b"), "High", choice, 1, Size::px(200.0f));
        changed = a || b;
        ui.end_box();
    }
};

TEST(UiWidgets, RadioSelectsExactlyOneOptionAndReportsOnlyTheChange) {
    Harness h;
    RadioGroup g;
    const auto declare = [&g](Context& ui) { g.declare(ui); };

    h.run(InputFrame{}, declare); // frame 1 establishes the rectangles
    const Rect high = h.ui.rect_of(id("opt.b"));
    ASSERT_GT(high.w, 0.0f);
    const glm::vec2 p = high.center();

    h.run(press_at(p), declare);
    EXPECT_EQ(g.choice, 0) << "a press alone must not select; the release does";

    h.run(release_at(p), declare);
    EXPECT_EQ(g.choice, 1);
    EXPECT_TRUE(g.changed);

    // Clicking the SAME option again writes the same value and reports NO
    // change — that is precisely what separates a radio from a toggle.
    h.run(press_at(p), declare);
    h.run(release_at(p), declare);
    EXPECT_EQ(g.choice, 1);
    EXPECT_FALSE(g.changed) << "re-selecting the current option is not a change";
}

TEST(UiWidgets, TextFieldAppendsBackspacesAndCommitsOnEnter) {
    Harness h;
    std::string value = "ab";
    bool committed = false;
    const auto declare = [&value, &committed](Context& ui) {
        committed = ui.text_field(id("name"), value, 8, Size::px(200.0f));
    };

    h.run(InputFrame{}, declare);

    // Nothing has focus yet, so typing goes nowhere.
    InputFrame typing{};
    typing.text = "c";
    h.run(typing, declare);
    EXPECT_EQ(value, "ab") << "an unfocused field must ignore typing";

    InputFrame tab{};
    tab.nav_next = true;
    h.run(tab, declare);
    ASSERT_EQ(h.ui.focused(), id("name"));

    h.run(typing, declare);
    EXPECT_EQ(value, "abc");
    EXPECT_FALSE(committed) << "a keystroke is not a commit";

    InputFrame back{};
    back.backspace = true;
    h.run(back, declare);
    EXPECT_EQ(value, "ab");

    InputFrame enter{};
    enter.enter = true;
    h.run(enter, declare);
    EXPECT_TRUE(committed) << "ENTER commits";
    EXPECT_EQ(value, "ab");
}

TEST(UiWidgets, TextFieldBackspaceRemovesAWholeUtf8Sequence) {
    // "é" is two bytes. Removing one of them leaves an undecodable string,
    // which the next measure charges the fallback advance for and which
    // reaches the packet as broken UTF-8.
    Harness h;
    std::string value = "a\xC3\xA9"; // "aé"
    ASSERT_EQ(value.size(), 3u);
    const auto declare = [&value](Context& ui) { (void)ui.text_field(id("f"), value, 32, Size::px(200.0f)); };

    h.run(InputFrame{}, declare);
    InputFrame tab{};
    tab.nav_next = true;
    h.run(tab, declare);
    ASSERT_EQ(h.ui.focused(), id("f"));

    InputFrame back{};
    back.backspace = true;
    h.run(back, declare);
    EXPECT_EQ(value, "a") << "backspace must consume the whole two-byte sequence";
}

TEST(UiWidgets, TextFieldRefusesARunThatWouldOverflowRatherThanSplittingIt) {
    Harness h;
    std::string value = "12345";
    const auto declare = [&value](Context& ui) { (void)ui.text_field(id("f"), value, 6, Size::px(200.0f)); };
    h.run(InputFrame{}, declare);
    InputFrame tab{};
    tab.nav_next = true;
    h.run(tab, declare);

    InputFrame typing{};
    typing.text = "ab"; // would reach 7 > 6
    h.run(typing, declare);
    EXPECT_EQ(value, "12345") << "an over-budget run is refused whole, not truncated mid-character";

    typing.text = "z"; // reaches exactly 6
    h.run(typing, declare);
    EXPECT_EQ(value, "12345z");
}

TEST(UiWidgets, TextFieldDrawsACaretOnlyWhileFocused) {
    Harness h;
    std::string value = "ab";
    const auto declare = [&value](Context& ui) { (void)ui.text_field(id("f"), value, 32, Size::px(200.0f)); };

    const auto unfocused = h.run(InputFrame{}, declare);
    EXPECT_EQ(unfocused->ui_commands().size(), 3u) << "background, border, text — and nothing else";

    InputFrame tab{};
    tab.nav_next = true;
    const auto focused = h.run(tab, declare);
    ASSERT_EQ(h.ui.focused(), id("f"));
    ASSERT_EQ(focused->ui_commands().size(), 4u) << "...plus the caret quad";
    const auto& caret = focused->ui_commands()[3].get<scene::UiFilledRectCommand>();
    const Rect box = h.ui.rect_of(id("f"));
    // Two characters at 10 px each, after the 8 px horizontal control padding.
    EXPECT_FLOAT_EQ(caret.x, box.x + 8.0f + 20.0f);
    EXPECT_FLOAT_EQ(caret.h, 20.0f);
}

TEST(UiWidgets, ProgressBarFillsTheClampedFraction) {
    Harness h;
    const auto frame = h.run(InputFrame{}, [](Context& ui) {
        ui.begin_column({});
        ui.progress_bar(0.25f, Size::px(200.0f), 10.0f);
        ui.progress_bar(2.0f, Size::px(200.0f), 10.0f);  // over 1 -> full
        ui.progress_bar(-1.0f, Size::px(200.0f), 10.0f); // under 0 -> no fill at all
        ui.end_box();
    });

    const auto cmds = frame->ui_commands();
    // track+fill, track+fill, track (a zero fill emits nothing).
    ASSERT_EQ(cmds.size(), 5u);
    const auto& quarter = cmds[1].get<scene::UiFilledRectCommand>();
    EXPECT_FLOAT_EQ(quarter.w, 50.0f);
    EXPECT_FLOAT_EQ(quarter.h, 10.0f);
    EXPECT_FLOAT_EQ(cmds[3].get<scene::UiFilledRectCommand>().w, 200.0f);
    EXPECT_FLOAT_EQ(cmds[4].get<scene::UiFilledRectCommand>().w, 200.0f);
}

TEST(UiWidgets, SeparatorSpansTheContainerAndTakesNoFocus) {
    Harness h;
    const auto frame = h.run(InputFrame{}, [](Context& ui) {
        BoxStyle panel{};
        panel.width = Size::px(300.0f);
        panel.padding = Edges::all(10.0f);
        ui.begin_column(panel);
        ui.label("a");
        ui.separator(4.0f);
        ui.label("b");
        ui.end_box();
    });

    const auto cmds = frame->ui_commands();
    ASSERT_EQ(cmds.size(), 3u);
    const auto& rule = cmds[1].get<scene::UiFilledRectCommand>();
    EXPECT_FLOAT_EQ(rule.w, 300.0f - 20.0f) << "the rule spans the container's content box";
    EXPECT_FLOAT_EQ(rule.h, 1.0f);
    EXPECT_TRUE(h.ui.focus_order().empty()) << "decoration is not focusable";
}

TEST(UiWidgets, NewWidgetsJoinTheTabOrderInDeclarationOrder) {
    Harness h;
    int choice = 0;
    std::string name = "x";
    h.run(InputFrame{}, [&](Context& ui) {
        ui.begin_column({});
        (void)ui.button(id("b"), "B");
        (void)ui.radio(id("r"), "R", choice, 0);
        (void)ui.text_field(id("t"), name);
        ui.progress_bar(0.5f); // not focusable
        ui.separator();        // not focusable
        ui.end_box();
    });

    const std::vector<WidgetId> order = h.ui.focus_order();
    ASSERT_EQ(order.size(), 3u);
    EXPECT_EQ(order[0], id("b"));
    EXPECT_EQ(order[1], id("r"));
    EXPECT_EQ(order[2], id("t"));
}

} // namespace
