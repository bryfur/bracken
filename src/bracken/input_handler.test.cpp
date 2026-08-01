#include "bracken/input_handler.hpp"

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_scancode.h>
#include <cstdint>
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

// --- Held-key level state (P9: the snapshot carries state, not just edges) ---

TEST(InputHandlerHeldKeys, SnapshotKeyHeldIsScancodeBoundsChecked) {
    bracken::InputSnapshot s;
    EXPECT_FALSE(s.key_held(SDL_SCANCODE_LSHIFT));
    s.keys_held[SDL_SCANCODE_LSHIFT] = true;
    EXPECT_TRUE(s.key_held(SDL_SCANCODE_LSHIFT));
    EXPECT_FALSE(s.key_held(SDL_SCANCODE_RSHIFT));
    // Out-of-range scancodes read false instead of walking off the bitset.
    EXPECT_FALSE(s.key_held(static_cast<SDL_Scancode>(-1)));
    EXPECT_FALSE(s.key_held(static_cast<SDL_Scancode>(SDL_SCANCODE_COUNT)));
}

TEST(InputHandlerHeldKeys, CapturedKeyboardReportsNothingHeld) {
    InputHandler h;
    // Whatever the real keyboard is doing while the suite runs, an overlay
    // that owns the keyboard must mute the polled held state — the same rule
    // the polled movement axes follow.
    h.set_keyboard_captured(true);
    const bracken::InputSnapshot s = h.snapshot();
    EXPECT_TRUE(s.keys_held.none());
}

// --- snapshot() consumes the mouse edges it copies (P9) ---

namespace {

SDL_Event mouse_button(uint32_t type, uint8_t button) {
    SDL_Event e{};
    e.type = type;
    e.button.button = button;
    e.button.x = 4.0f;
    e.button.y = 8.0f;
    e.button.down = (type == SDL_EVENT_MOUSE_BUTTON_DOWN);
    return e;
}

} // namespace

TEST(InputHandlerMouseEdges, SnapshotConsumesTheEdgesItCopies) {
    InputHandler h;
    h.process_event(mouse_button(SDL_EVENT_MOUSE_BUTTON_DOWN, SDL_BUTTON_LEFT));
    h.process_event(mouse_button(SDL_EVENT_MOUSE_BUTTON_DOWN, SDL_BUTTON_RIGHT));
    EXPECT_TRUE(h.mouse_left_pressed());

    const bracken::InputSnapshot first = h.snapshot();
    EXPECT_TRUE(first.mouse_left_pressed);
    EXPECT_TRUE(first.mouse_right_pressed);

    // One click is exactly one snapshot's edge: the handler's edges are gone...
    EXPECT_FALSE(h.mouse_left_pressed());
    EXPECT_FALSE(h.mouse_right_pressed());
    const bracken::InputSnapshot second = h.snapshot();
    EXPECT_FALSE(second.mouse_left_pressed);
    EXPECT_FALSE(second.mouse_right_pressed);
    // ...but the LEVEL state is not an edge and survives.
    EXPECT_TRUE(second.mouse_left_held);
}

TEST(InputHandlerMouseEdges, RetainMouseEdgesSpansExactlyOneSnapshot) {
    InputHandler h;
    h.process_event(mouse_button(SDL_EVENT_MOUSE_BUTTON_DOWN, SDL_BUTTON_LEFT));

    h.retain_mouse_edges();
    const bracken::InputSnapshot kept = h.snapshot();
    EXPECT_TRUE(kept.mouse_left_pressed);
    EXPECT_TRUE(h.mouse_left_pressed()); // still there for the UI layer

    // One-shot: the next snapshot consumes as usual.
    const bracken::InputSnapshot consumed = h.snapshot();
    EXPECT_TRUE(consumed.mouse_left_pressed);
    EXPECT_FALSE(h.mouse_left_pressed());
}

TEST(InputHandlerMouseEdges, ReleaseEdgeIsAlsoConsumed) {
    InputHandler h;
    h.process_event(mouse_button(SDL_EVENT_MOUSE_BUTTON_DOWN, SDL_BUTTON_LEFT));
    (void)h.snapshot();
    h.process_event(mouse_button(SDL_EVENT_MOUSE_BUTTON_UP, SDL_BUTTON_LEFT));

    const bracken::InputSnapshot s = h.snapshot();
    EXPECT_TRUE(s.mouse_left_released);
    EXPECT_FALSE(s.mouse_left_held);
    EXPECT_FALSE(h.mouse_left_released());
}
