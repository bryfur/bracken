#include "bracken/ui/ui_input.hpp"

namespace bracken::ui {

InputFrame InputFrame::from_snapshot(const InputSnapshot& in) {
    InputFrame f{};
    f.mouse = {in.mouse_x, in.mouse_y};
    f.mouse_down = in.mouse_left_held;
    f.mouse_pressed = in.mouse_left_pressed;
    f.mouse_released = in.mouse_left_released;
    // wheel_y, not camera_zoom_delta: the latter is negated and multiplied by
    // 50 for the camera controller, so a scroll view driven by it would run
    // backwards at fifty times the speed (see InputSnapshot::wheel_y).
    f.wheel = in.wheel_y;

    f.nav_up = in.menu_up_pressed || in.pad_nav_up || in.was_key_just_pressed(SDL_SCANCODE_UP);
    f.nav_down = in.menu_down_pressed || in.pad_nav_down || in.was_key_just_pressed(SDL_SCANCODE_DOWN);
    f.nav_left = in.menu_left_pressed || in.pad_nav_left || in.was_key_just_pressed(SDL_SCANCODE_LEFT);
    f.nav_right = in.menu_right_pressed || in.pad_nav_right || in.was_key_just_pressed(SDL_SCANCODE_RIGHT);
    // pad_accept_pressed, NOT controller_primary: the latter is a LEVEL, so a
    // held A button would re-fire the focused widget on every frame.
    f.nav_activate = in.menu_select_pressed || in.text_enter_pressed || in.pad_accept_pressed ||
                     in.was_key_just_pressed(SDL_SCANCODE_RETURN) || in.was_key_just_pressed(SDL_SCANCODE_SPACE);
    f.nav_cancel = in.text_escape_pressed || in.menu_toggle_pressed || in.pad_cancel_pressed ||
                   in.was_key_just_pressed(SDL_SCANCODE_ESCAPE);

    const bool shift = in.key_held(SDL_SCANCODE_LSHIFT) || in.key_held(SDL_SCANCODE_RSHIFT);
    const bool tab = in.was_key_just_pressed(SDL_SCANCODE_TAB);
    f.nav_next = (tab && !shift) || in.pad_next_pressed;
    f.nav_prev = (tab && shift) || in.pad_prev_pressed;

    // Text entry. A VIEW into the snapshot — see the header's lifetime note.
    f.text = in.text_input;
    f.backspace = in.text_backspace_pressed;
    f.enter = in.text_enter_pressed || in.was_key_just_pressed(SDL_SCANCODE_RETURN);
    return f;
}

InputSnapshot InputCapture::mask(const InputSnapshot& in) const {
    InputSnapshot out = in;
    if (mouse) {
        out.mouse_left_held = false;
        out.mouse_left_pressed = false;
        out.mouse_left_released = false;
        out.mouse_right_pressed = false;
        // mouse_x/mouse_y are deliberately preserved — see the header.
    }
    if (wheel) {
        out.wheel_x = 0.0f;
        out.wheel_y = 0.0f;
        // The cooked camera zoom is the same physical event, so a captured
        // wheel must silence it too — otherwise scrolling a list also dollies
        // the camera.
        out.camera_zoom_delta = 0.0f;
    }
    if (keyboard) {
        out.key_just_pressed.fill(false);
        out.keys_held.reset();
        out.menu_toggle_pressed = false;
        out.menu_up_pressed = false;
        out.menu_down_pressed = false;
        out.menu_left_pressed = false;
        out.menu_right_pressed = false;
        out.menu_select_pressed = false;
        out.text_input.clear();
        out.text_backspace_pressed = false;
        out.text_enter_pressed = false;
        out.text_escape_pressed = false;
        out.controller_primary = false;
        out.controller_secondary = false;
        out.pad_nav_up = false;
        out.pad_nav_down = false;
        out.pad_nav_left = false;
        out.pad_nav_right = false;
        out.pad_accept_pressed = false;
        out.pad_cancel_pressed = false;
        out.pad_next_pressed = false;
        out.pad_prev_pressed = false;
        // Movement is keyboard/pad derived, so a keyboard-capturing UI must
        // stop the character walking while a menu is open. Both the raw flags
        // and the derived camera-relative axes, or a host reading `state`
        // would still move.
        out.move_forward = false;
        out.move_backward = false;
        out.move_left = false;
        out.move_right = false;
        out.state.move_up = false;
        out.state.move_down = false;
        out.state.move_left = false;
        out.state.move_right = false;
        out.state.move_dir_x = 0.0f;
        out.state.move_dir_y = 0.0f;
    }
    return out;
}

} // namespace bracken::ui
