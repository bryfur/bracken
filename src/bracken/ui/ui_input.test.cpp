#include "bracken/ui/ui_input.hpp"

#include <gtest/gtest.h>

namespace {

using namespace bracken::ui;
using bracken::InputSnapshot;

InputSnapshot busy_snapshot() {
    InputSnapshot in;
    in.mouse_x = 100.0f;
    in.mouse_y = 200.0f;
    in.mouse_left_held = true;
    in.mouse_left_pressed = true;
    in.mouse_left_released = true;
    in.mouse_right_pressed = true;
    in.wheel_x = 1.0f;
    in.wheel_y = -3.0f;
    in.camera_zoom_delta = 150.0f;
    in.menu_up_pressed = true;
    in.menu_select_pressed = true;
    in.text_input = "hello";
    in.text_enter_pressed = true;
    in.text_escape_pressed = true;
    in.controller_primary = true;
    in.controller_secondary = true;
    in.move_forward = true;
    in.state.move_dir_x = 0.5f;
    in.state.move_dir_y = -0.5f;
    in.key_just_pressed[SDL_SCANCODE_TAB] = true;
    in.keys_held[SDL_SCANCODE_LSHIFT] = true;
    in.keys_held[SDL_SCANCODE_W] = true;
    return in;
}

TEST(UiInputFrame, FromSnapshotMapsMouseWheelAndNav) {
    const InputSnapshot in = busy_snapshot();
    const InputFrame f = InputFrame::from_snapshot(in);

    EXPECT_FLOAT_EQ(f.mouse.x, 100.0f);
    EXPECT_FLOAT_EQ(f.mouse.y, 200.0f);
    EXPECT_TRUE(f.mouse_down);
    EXPECT_TRUE(f.mouse_pressed);
    EXPECT_TRUE(f.mouse_released);
    // The RAW wheel, not the camera's cooked zoom: -3 notches, not -150 units.
    EXPECT_FLOAT_EQ(f.wheel, -3.0f);

    EXPECT_TRUE(f.nav_up);
    EXPECT_FALSE(f.nav_down);
    EXPECT_TRUE(f.nav_activate); // menu_select
    EXPECT_TRUE(f.nav_cancel);   // escape
    // Tab with shift held is PREV, not NEXT.
    EXPECT_FALSE(f.nav_next);
    EXPECT_TRUE(f.nav_prev);
}

TEST(UiInputFrame, ArrowScancodesAndPadFeedTheSameNavFields) {
    InputSnapshot in;
    in.key_just_pressed[SDL_SCANCODE_DOWN] = true;
    in.key_just_pressed[SDL_SCANCODE_RIGHT] = true;
    in.controller_primary = true;
    in.controller_secondary = true;
    const InputFrame f = InputFrame::from_snapshot(in);
    EXPECT_TRUE(f.nav_down);
    EXPECT_TRUE(f.nav_right);
    EXPECT_FALSE(f.nav_up);
    EXPECT_TRUE(f.nav_activate); // pad primary
    EXPECT_TRUE(f.nav_cancel);   // pad secondary

    InputSnapshot tab_only;
    tab_only.key_just_pressed[SDL_SCANCODE_TAB] = true;
    const InputFrame g = InputFrame::from_snapshot(tab_only);
    EXPECT_TRUE(g.nav_next);
    EXPECT_FALSE(g.nav_prev);
}

TEST(UiInputCapture, MouseCaptureLeavesTheKeyboardAlone) {
    const InputSnapshot in = busy_snapshot();
    InputCapture cap;
    cap.mouse = true;
    const InputSnapshot out = cap.mask(in);

    EXPECT_FALSE(out.mouse_left_held);
    EXPECT_FALSE(out.mouse_left_pressed);
    EXPECT_FALSE(out.mouse_left_released);
    EXPECT_FALSE(out.mouse_right_pressed);
    // The POSITION survives: a position is not an event, and zeroing it would
    // teleport a host's cursor logic to the origin.
    EXPECT_FLOAT_EQ(out.mouse_x, 100.0f);
    EXPECT_FLOAT_EQ(out.mouse_y, 200.0f);
    // Nothing keyboard-side moved.
    EXPECT_TRUE(out.keys_held[SDL_SCANCODE_W]);
    EXPECT_TRUE(out.was_key_just_pressed(SDL_SCANCODE_TAB));
    EXPECT_EQ(out.text_input, "hello");
    EXPECT_TRUE(out.move_forward);
    // And the wheel is a separate channel.
    EXPECT_FLOAT_EQ(out.wheel_y, -3.0f);
}

TEST(UiInputCapture, KeyboardCaptureLeavesTheMouseAlone) {
    const InputSnapshot in = busy_snapshot();
    InputCapture cap;
    cap.keyboard = true;
    const InputSnapshot out = cap.mask(in);

    EXPECT_FALSE(out.was_key_just_pressed(SDL_SCANCODE_TAB));
    EXPECT_FALSE(out.keys_held[SDL_SCANCODE_W]);
    EXPECT_FALSE(out.menu_up_pressed);
    EXPECT_FALSE(out.menu_select_pressed);
    EXPECT_TRUE(out.text_input.empty());
    EXPECT_FALSE(out.text_enter_pressed);
    EXPECT_FALSE(out.controller_primary);
    EXPECT_FALSE(out.move_forward);
    EXPECT_FLOAT_EQ(out.state.move_dir_x, 0.0f);
    EXPECT_FLOAT_EQ(out.state.move_dir_y, 0.0f);

    // Mouse untouched.
    EXPECT_TRUE(out.mouse_left_pressed);
    EXPECT_FLOAT_EQ(out.mouse_x, 100.0f);
}

TEST(UiInputCapture, WheelCaptureAlsoSilencesTheCookedCameraZoom) {
    const InputSnapshot in = busy_snapshot();
    InputCapture cap;
    cap.wheel = true;
    const InputSnapshot out = cap.mask(in);

    EXPECT_FLOAT_EQ(out.wheel_x, 0.0f);
    EXPECT_FLOAT_EQ(out.wheel_y, 0.0f);
    // Same physical event: leaving the cooked delta alone would dolly the
    // camera every time the player scrolled a list.
    EXPECT_FLOAT_EQ(out.camera_zoom_delta, 0.0f);
    EXPECT_TRUE(out.mouse_left_pressed); // other channels untouched
}

TEST(UiInputCapture, NothingCapturedIsAnExactPassthrough) {
    const InputSnapshot in = busy_snapshot();
    const InputSnapshot out = InputCapture{}.mask(in);
    EXPECT_TRUE(out.mouse_left_pressed);
    EXPECT_FLOAT_EQ(out.wheel_y, -3.0f);
    EXPECT_TRUE(out.was_key_just_pressed(SDL_SCANCODE_TAB));
    EXPECT_EQ(out.text_input, "hello");
}

TEST(UiInputCapture, AFullyMaskedSnapshotReportsNothingHeldOrPressed) {
    const InputSnapshot in = busy_snapshot();
    InputCapture cap;
    cap.mouse = true;
    cap.keyboard = true;
    cap.wheel = true;
    const InputSnapshot out = cap.mask(in);

    // This is the state an ActionMap sees when the UI ate the frame: every
    // key-bound and pad-bound action must read as idle.
    for (int i = 0; i < SDL_SCANCODE_COUNT; ++i) {
        EXPECT_FALSE(out.key_just_pressed[static_cast<std::size_t>(i)]);
        EXPECT_FALSE(out.key_held(static_cast<SDL_Scancode>(i)));
    }
    EXPECT_FALSE(out.mouse_left_held);
    EXPECT_FALSE(out.mouse_left_pressed);
    EXPECT_FALSE(out.mouse_left_released);
    EXPECT_FALSE(out.controller_primary);
    EXPECT_FALSE(out.controller_secondary);
    EXPECT_FLOAT_EQ(out.wheel_y, 0.0f);
}

} // namespace
