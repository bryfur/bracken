#include "bracken/input_handler.hpp"

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_scancode.h>
#include <gtest/gtest.h>

using bracken::InputHandler;

// All tests here are headless: no window, no video init. InputHandler's
// constructor touches the SDL gamepad subsystem, which works (or fails
// gracefully) without a display; process_event on synthesized keyboard
// events makes no SDL calls at all.

namespace {

// Build a synthetic key-down event. `scancode` is the physical key,
// `keycode` the layout-dependent label — on AZERTY the physical W key
// reports SDLK_Z, which is exactly the mismatch these tests exercise.
SDL_Event key_down(SDL_Scancode scancode, SDL_Keycode keycode, bool repeat = false) {
    SDL_Event e{};
    e.type = SDL_EVENT_KEY_DOWN;
    e.key.scancode = scancode;
    e.key.key = keycode;
    e.key.down = true;
    e.key.repeat = repeat;
    return e;
}

} // namespace

// --- Menu navigation is matched by physical scancode (M4) ---

TEST(InputHandlerMenu, WasdScancodesNavigateRegardlessOfKeycode) {
    InputHandler h;
    h.set_game_input_enabled(false); // menu mode

    // AZERTY: physical WASD positions produce keycodes Z/Q/S/D. Navigation
    // must follow the physical key, not the label.
    h.process_event(key_down(SDL_SCANCODE_W, SDLK_Z));
    EXPECT_TRUE(h.menu_up_pressed());
    h.process_event(key_down(SDL_SCANCODE_S, SDLK_S));
    EXPECT_TRUE(h.menu_down_pressed());
    h.process_event(key_down(SDL_SCANCODE_A, SDLK_Q));
    EXPECT_TRUE(h.menu_left_pressed());
    h.process_event(key_down(SDL_SCANCODE_D, SDLK_D));
    EXPECT_TRUE(h.menu_right_pressed());
}

TEST(InputHandlerMenu, KeycodeAloneDoesNotNavigate) {
    InputHandler h;
    h.set_game_input_enabled(false);

    // Keycode says W but the physical key is elsewhere (AZERTY user pressing
    // the key labeled W, physically at the QWERTY-Z position): no menu-up.
    h.process_event(key_down(SDL_SCANCODE_Z, SDLK_W));
    EXPECT_FALSE(h.menu_up_pressed());
    h.process_event(key_down(SDL_SCANCODE_Q, SDLK_A));
    EXPECT_FALSE(h.menu_left_pressed());
}

TEST(InputHandlerMenu, ArrowsReturnSpaceEscapeStillWork) {
    InputHandler h;
    h.set_game_input_enabled(false);

    h.process_event(key_down(SDL_SCANCODE_UP, SDLK_UP));
    EXPECT_TRUE(h.menu_up_pressed());
    h.process_event(key_down(SDL_SCANCODE_DOWN, SDLK_DOWN));
    EXPECT_TRUE(h.menu_down_pressed());
    h.process_event(key_down(SDL_SCANCODE_LEFT, SDLK_LEFT));
    EXPECT_TRUE(h.menu_left_pressed());
    h.process_event(key_down(SDL_SCANCODE_RIGHT, SDLK_RIGHT));
    EXPECT_TRUE(h.menu_right_pressed());

    EXPECT_FALSE(h.menu_select_pressed());
    h.process_event(key_down(SDL_SCANCODE_RETURN, SDLK_RETURN));
    EXPECT_TRUE(h.menu_select_pressed());
    h.clear_menu_inputs();
    h.process_event(key_down(SDL_SCANCODE_SPACE, SDLK_SPACE));
    EXPECT_TRUE(h.menu_select_pressed());

    EXPECT_FALSE(h.menu_toggle_pressed());
    h.process_event(key_down(SDL_SCANCODE_ESCAPE, SDLK_ESCAPE));
    EXPECT_TRUE(h.menu_toggle_pressed());
}

TEST(InputHandlerMenu, NavigationGatedOnGameInputDisabled) {
    InputHandler h;
    ASSERT_TRUE(h.is_game_input_enabled()); // default: gameplay mode

    h.process_event(key_down(SDL_SCANCODE_W, SDLK_W));
    h.process_event(key_down(SDL_SCANCODE_RETURN, SDLK_RETURN));
    EXPECT_FALSE(h.menu_up_pressed());
    EXPECT_FALSE(h.menu_select_pressed());

    // Escape toggles the menu in either mode.
    h.process_event(key_down(SDL_SCANCODE_ESCAPE, SDLK_ESCAPE));
    EXPECT_TRUE(h.menu_toggle_pressed());
}

TEST(InputHandlerMenu, AutoRepeatIgnored) {
    InputHandler h;
    h.set_game_input_enabled(false);

    h.process_event(key_down(SDL_SCANCODE_W, SDLK_W, /*repeat=*/true));
    EXPECT_FALSE(h.menu_up_pressed());
}

TEST(InputHandlerMenu, ClearMenuInputsResetsAllFlags) {
    InputHandler h;
    h.set_game_input_enabled(false);

    h.process_event(key_down(SDL_SCANCODE_ESCAPE, SDLK_ESCAPE));
    h.process_event(key_down(SDL_SCANCODE_W, SDLK_W));
    h.process_event(key_down(SDL_SCANCODE_S, SDLK_S));
    h.process_event(key_down(SDL_SCANCODE_A, SDLK_A));
    h.process_event(key_down(SDL_SCANCODE_D, SDLK_D));
    h.process_event(key_down(SDL_SCANCODE_RETURN, SDLK_RETURN));
    h.clear_menu_inputs();
    EXPECT_FALSE(h.menu_toggle_pressed());
    EXPECT_FALSE(h.menu_up_pressed());
    EXPECT_FALSE(h.menu_down_pressed());
    EXPECT_FALSE(h.menu_left_pressed());
    EXPECT_FALSE(h.menu_right_pressed());
    EXPECT_FALSE(h.menu_select_pressed());
}

// --- Scancode-indexed key edges (consistency companion to the menu path) ---

TEST(InputHandlerKeyEdges, EdgeIsScancodeIndexedAndRepeatFree) {
    InputHandler h;

    EXPECT_FALSE(h.was_key_just_pressed(SDL_SCANCODE_W));
    h.process_event(key_down(SDL_SCANCODE_W, SDLK_Z)); // AZERTY-style mismatch
    EXPECT_TRUE(h.was_key_just_pressed(SDL_SCANCODE_W));
    EXPECT_FALSE(h.was_key_just_pressed(SDL_SCANCODE_Z));

    h.clear_key_edges();
    EXPECT_FALSE(h.was_key_just_pressed(SDL_SCANCODE_W));

    h.process_event(key_down(SDL_SCANCODE_W, SDLK_W, /*repeat=*/true));
    EXPECT_FALSE(h.was_key_just_pressed(SDL_SCANCODE_W));
}
