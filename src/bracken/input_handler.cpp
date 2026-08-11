#include "input_handler.hpp"
#include "bracken/core/logger.hpp"
#include "SDL3/SDL_events.h"
#include "SDL3/SDL_gamepad.h"
#include "SDL3/SDL_init.h"
#include "SDL3/SDL_joystick.h"
#include "SDL3/SDL_keyboard.h"
#include "SDL3/SDL_keycode.h"
#include "SDL3/SDL_mouse.h"
#include "SDL3/SDL_scancode.h"
#include "SDL3/SDL_stdinc.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace bracken {

InputHandler::InputHandler() {
    // Get initial mouse position
    SDL_GetMouseState(&mouse_x_, &mouse_y_);
    last_mouse_x_ = mouse_x_;
    last_mouse_y_ = mouse_y_;

    // Initialize controller subsystem
    if (!SDL_WasInit(SDL_INIT_GAMEPAD)) {
        SDL_InitSubSystem(SDL_INIT_GAMEPAD);
    }

    // Check for any already-connected controllers
    int num_joysticks = 0;
    SDL_JoystickID* joysticks = SDL_GetJoysticks(&num_joysticks);
    if (joysticks) {
        for (int i = 0; i < num_joysticks && !gamepad_; i++) {
            if (SDL_IsGamepad(joysticks[i])) {
                handle_controller_added(joysticks[i]);
            }
        }
        SDL_free(joysticks);
    }
}

InputHandler::~InputHandler() {
    if (gamepad_) {
        SDL_CloseGamepad(gamepad_);
        gamepad_ = nullptr;
    }
}

const char* InputHandler::get_controller_name() const {
    if (gamepad_) {
        return SDL_GetGamepadName(gamepad_);
    }
    return "No Controller";
}

void InputHandler::set_text_input_enabled(bool enabled) {
    if (text_input_enabled_ == enabled) {
        return;
    }
    text_input_enabled_ = enabled;
    if (enabled) {
        SDL_StartTextInput(SDL_GetKeyboardFocus());
    } else {
        SDL_Window* win = SDL_GetKeyboardFocus();
        if (win) {
            SDL_StopTextInput(win);
        }
        text_input_buffer_.clear();
    }
    text_backspace_pressed_ = false;
    text_enter_pressed_ = false;
    text_escape_pressed_ = false;
}

void InputHandler::process_event(const SDL_Event& event) {
    // Text input mode consumes key + text events first.
    if (text_input_enabled_) {
        if (event.type == SDL_EVENT_TEXT_INPUT) {
            if (event.text.text) {
                text_input_buffer_ += event.text.text;
            }
            return;
        }
        if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
            switch (event.key.key) {
                case SDLK_BACKSPACE:
                    text_backspace_pressed_ = true;
                    return;
                case SDLK_RETURN:
                    text_enter_pressed_ = true;
                    return;
                case SDLK_ESCAPE:
                    text_escape_pressed_ = true;
                    return;
                default:
                    break;
            }
        }
        // Swallow other key events so gameplay doesn't receive them.
        if (event.type == SDL_EVENT_KEY_DOWN || event.type == SDL_EVENT_KEY_UP) {
            return;
        }
    }

    // Track per-frame key-press edges from real KEY_DOWN events (excluding
    // auto-repeat). Done before menu-key handling so all consumers can read
    // the same source of truth via was_key_just_pressed().
    if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
        SDL_Scancode sc = event.key.scancode;
        if (sc >= 0 && sc < SDL_SCANCODE_COUNT) {
            key_just_pressed_[sc] = true;
        }
    }

    // Handle controller connection/disconnection
    if (event.type == SDL_EVENT_GAMEPAD_ADDED) {
        handle_controller_added(event.gdevice.which);
    }
    if (event.type == SDL_EVENT_GAMEPAD_REMOVED) {
        handle_controller_removed(event.gdevice.which);
    }

    // Handle controller button events for menu. The `menu_*` half stays gated
    // on !game_input_enabled_ (that is what it has always meant); the `pad_*`
    // half is UNGATED, because a HUD is navigable while the game runs and a UI
    // that only answers the pad once the host has paused everything is not a
    // pad-navigable UI. See InputSnapshot's pad_* block.
    if (event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && gamepad_) {
        switch (event.gbutton.button) {
            case SDL_GAMEPAD_BUTTON_START:
                menu_toggle_pressed_ = true;
                break;
            case SDL_GAMEPAD_BUTTON_DPAD_UP:
                pad_nav_up_ = true;
                if (!game_input_enabled_) {
                    menu_up_pressed_ = true;
                }
                break;
            case SDL_GAMEPAD_BUTTON_DPAD_DOWN:
                pad_nav_down_ = true;
                if (!game_input_enabled_) {
                    menu_down_pressed_ = true;
                }
                break;
            case SDL_GAMEPAD_BUTTON_DPAD_LEFT:
                pad_nav_left_ = true;
                if (!game_input_enabled_) {
                    menu_left_pressed_ = true;
                }
                break;
            case SDL_GAMEPAD_BUTTON_DPAD_RIGHT:
                pad_nav_right_ = true;
                if (!game_input_enabled_) {
                    menu_right_pressed_ = true;
                }
                break;
            case SDL_GAMEPAD_BUTTON_SOUTH: // A button
                pad_accept_pressed_ = true;
                if (!game_input_enabled_) {
                    menu_select_pressed_ = true;
                }
                break;
            case SDL_GAMEPAD_BUTTON_EAST: // B button
                pad_cancel_pressed_ = true;
                break;
            case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER:
                pad_next_pressed_ = true;
                break;
            case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER:
                pad_prev_pressed_ = true;
                break;
            default:
                // Other gamepad buttons are not bound to menu actions.
                break;
        }
    }

    // Handle key events for menu. Matched by physical scancode — the same
    // convention as movement polling and the key-edge array — so the physical
    // WASD keys that move the character also navigate menus on non-QWERTY
    // layouts (AZERTY etc.).
    if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
        switch (event.key.scancode) {
            case SDL_SCANCODE_ESCAPE:
                menu_toggle_pressed_ = true;
                break;
            case SDL_SCANCODE_UP:
            case SDL_SCANCODE_W:
                if (!game_input_enabled_) {
                    menu_up_pressed_ = true;
                }
                break;
            case SDL_SCANCODE_DOWN:
            case SDL_SCANCODE_S:
                if (!game_input_enabled_) {
                    menu_down_pressed_ = true;
                }
                break;
            case SDL_SCANCODE_LEFT:
            case SDL_SCANCODE_A:
                if (!game_input_enabled_) {
                    menu_left_pressed_ = true;
                }
                break;
            case SDL_SCANCODE_RIGHT:
            case SDL_SCANCODE_D:
                if (!game_input_enabled_) {
                    menu_right_pressed_ = true;
                }
                break;
            case SDL_SCANCODE_RETURN:
            case SDL_SCANCODE_SPACE:
                if (!game_input_enabled_) {
                    menu_select_pressed_ = true;
                }
                break;
            default:
                // Other keys are not bound to menu actions.
                break;
        }
    }

    // Track mouse position on every motion event regardless of mode — UI
    // drag/hover logic needs a live cursor location even when game input is
    // disabled (menus open) or when no buttons are held.
    if (event.type == SDL_EVENT_MOUSE_MOTION) {
        mouse_x_ = event.motion.x;
        mouse_y_ = event.motion.y;
    }

    // Mouse button edges (tracked regardless of game/menu mode so UI can consume clicks).
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
        if (event.button.button == SDL_BUTTON_LEFT) {
            left_mouse_down_ = true;
            left_mouse_pressed_edge_ = true;
            left_mouse_press_x_ = event.button.x;
            left_mouse_press_y_ = event.button.y;
        }
        if (event.button.button == SDL_BUTTON_RIGHT) {
            right_mouse_pressed_edge_ = true;
        }
    }
    if (event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
        if (event.button.button == SDL_BUTTON_LEFT) {
            left_mouse_down_ = false;
            left_mouse_released_edge_ = true;
        }
    }

    // Only process game input if enabled
    if (game_input_enabled_) {
        // Track mouse button state for camera orbit
        if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
            if (event.button.button == SDL_BUTTON_RIGHT) {
                right_mouse_down_ = true;
                SDL_SetWindowRelativeMouseMode(SDL_GetWindowFromEvent(&event), true);
            }
        }
        if (event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
            if (event.button.button == SDL_BUTTON_RIGHT) {
                right_mouse_down_ = false;
                SDL_SetWindowRelativeMouseMode(SDL_GetWindowFromEvent(&event), false);
            }
        }

        // Mouse wheel: the cooked camera zoom AND the raw delta. Two
        // accumulators, not one, because they answer different questions —
        // the camera wants world units per notch (negated, times 50) and a UI
        // scroll view wants notches. Do not "simplify" by deriving one from
        // the other: the cook's sign and scale are camera policy.
        if (event.type == SDL_EVENT_MOUSE_WHEEL) {
            camera_zoom_delta_ -= event.wheel.y * 50.0f;
            wheel_x_ += event.wheel.x;
            wheel_y_ += event.wheel.y;
        }

        // Mouse motion for camera orbit (only when right button held)
        if (event.type == SDL_EVENT_MOUSE_MOTION && right_mouse_down_) {
            float x_mult = invert_camera_x_ ? 1.0f : -1.0f;
            float y_mult = invert_camera_y_ ? -1.0f : 1.0f;

            camera_yaw_ += event.motion.xrel * mouse_sensitivity_ * x_mult;
            camera_pitch_ += event.motion.yrel * mouse_sensitivity_ * y_mult;

            camera_pitch_ = std::clamp(camera_pitch_, -70.0f, 70.0f);

            while (camera_yaw_ < 0.0f) camera_yaw_ += 360.0f;
            while (camera_yaw_ >= 360.0f) camera_yaw_ -= 360.0f;
        }
    }
}

bool InputHandler::was_key_just_pressed(SDL_Scancode scancode) const {
    if (scancode < 0 || scancode >= SDL_SCANCODE_COUNT) {
        return false;
    }
    return key_just_pressed_[scancode];
}

void InputHandler::clear_key_edges() {
    key_just_pressed_.fill(false);
}

void InputHandler::post_process_events() {
    // Save previous input for change detection
    last_input_ = current_input_;

    // Update controller state
    if (gamepad_ && game_input_enabled_) {
        update_input_from_controller();
    }
    // The UI's pad navigation is polled whatever the game-input mode — a HUD
    // is navigable during play. Cheap (two axis reads) and edge-latched, so
    // running it every frame costs nothing when the stick is at rest.
    update_ui_pad_nav();

    // Only update game input if enabled. While chat is capturing text, treat
    // game input as suspended so W/A/S/D in the chat box don't drive movement.
    const bool gameplay_active = game_input_enabled_ && !text_input_enabled_;
    if (gameplay_active) {
        update_input_from_keyboard();
        update_camera_from_mouse();

        // Apply controller camera input (smooth analog)
        if (gamepad_) {
            float x_mult = invert_camera_x_ ? 1.0f : -1.0f;
            float y_mult = invert_camera_y_ ? -1.0f : 1.0f;

            camera_yaw_ += controller_camera_x_ * controller_sensitivity_ * x_mult;
            camera_pitch_ += controller_camera_y_ * controller_sensitivity_ * y_mult;

            camera_pitch_ = std::clamp(camera_pitch_, -70.0f, 70.0f);

            while (camera_yaw_ < 0.0f) camera_yaw_ += 360.0f;
            while (camera_yaw_ >= 360.0f) camera_yaw_ -= 360.0f;
        }
    } else {
        move_forward_ = false;
        move_backward_ = false;
        move_left_ = false;
        move_right_ = false;
        current_input_.move_dir_x = 0.0f;
        current_input_.move_dir_y = 0.0f;
    }

    // Check if input changed
    input_changed_ =
        (current_input_.move_up != last_input_.move_up || current_input_.move_down != last_input_.move_down ||
         current_input_.move_left != last_input_.move_left || current_input_.move_right != last_input_.move_right ||
         std::abs(current_input_.move_dir_x - last_input_.move_dir_x) > 0.01f ||
         std::abs(current_input_.move_dir_y - last_input_.move_dir_y) > 0.01f ||
         std::abs(current_input_.attack_dir_x - last_input_.attack_dir_x) > 0.01f ||
         std::abs(current_input_.attack_dir_y - last_input_.attack_dir_y) > 0.01f);
}

bool InputHandler::process_events() {
    // Reset per-frame deltas
    camera_zoom_delta_ = 0.0f;
    menu_toggle_pressed_ = false;
    menu_up_pressed_ = false;
    menu_down_pressed_ = false;
    menu_left_pressed_ = false;
    menu_right_pressed_ = false;
    menu_select_pressed_ = false;
    clear_pad_nav_edges();
    clear_key_edges();

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_EVENT_QUIT) {
            return false;
        }
        process_event(event);
    }

    post_process_events();
    return true;
}

void InputHandler::update_input_from_keyboard() {
    const bool* keys = SDL_GetKeyboardState(nullptr);

    // Keyboard movement axes. Polled state bypasses event consumption, so an
    // overlay capturing the keyboard mutes these here instead.
    const bool kb = !keyboard_captured_;
    bool kb_forward = kb && (keys[SDL_SCANCODE_W] || keys[SDL_SCANCODE_UP]);
    bool kb_backward = kb && (keys[SDL_SCANCODE_S] || keys[SDL_SCANCODE_DOWN]);
    bool kb_left = kb && (keys[SDL_SCANCODE_A] || keys[SDL_SCANCODE_LEFT]);
    bool kb_right = kb && (keys[SDL_SCANCODE_D] || keys[SDL_SCANCODE_RIGHT]);

    // Merge with controller digital input (left stick beyond deadzone)
    move_forward_ = kb_forward || (controller_move_y_ < -0.5f);
    move_backward_ = kb_backward || (controller_move_y_ > 0.5f);
    move_left_ = kb_left || (controller_move_x_ < -0.5f);
    move_right_ = kb_right || (controller_move_x_ > 0.5f);
}

void InputHandler::update_camera_from_mouse() {
    // Use actual camera forward direction (set by renderer, accounts for shoulder offset)
    // This ensures W moves exactly "into the screen" regardless of camera offset
    float forward_x = camera_forward_x_;
    float forward_z = camera_forward_z_;

    // Normalize forward (should already be normalized, but be safe)
    float forward_len = std::sqrt(forward_x * forward_x + forward_z * forward_z);
    if (forward_len > 0.001f) {
        forward_x /= forward_len;
        forward_z /= forward_len;
    }

    // Right vector (perpendicular to forward)
    float right_x = -forward_z;
    float right_z = forward_x;

    // Transform WASD input to world-space movement direction
    float move_x = 0.0f;
    float move_z = 0.0f;

    // Check if controller has analog movement input
    bool has_controller_analog = (std::abs(controller_move_x_) > 0.01f || std::abs(controller_move_y_) > 0.01f);

    if (has_controller_analog) {
        // Use smooth analog controller input
        // Controller Y is inverted (up is negative), map to forward
        move_x = controller_move_x_ * right_x + (-controller_move_y_) * forward_x;
        move_z = controller_move_x_ * right_z + (-controller_move_y_) * forward_z;
    } else {
        // Use digital keyboard input
        if (move_forward_) {
            move_x += forward_x;
            move_z += forward_z;
        }
        if (move_backward_) {
            move_x -= forward_x;
            move_z -= forward_z;
        }
        if (move_left_) {
            move_x -= right_x;
            move_z -= right_z;
        }
        if (move_right_) {
            move_x += right_x;
            move_z += right_z;
        }
    }

    // Normalize movement direction (clamp magnitude to 1, preserve partial analog tilt)
    float move_len = normalize_move_dir(move_x, move_z);

    // Send continuous movement direction for perfectly smooth movement
    // In the 2D game world: X stays X, Z becomes Y
    current_input_.move_dir_x = move_x;
    current_input_.move_dir_y = move_z;

    // Legacy boolean flags (kept for compatibility, not used for movement)
    current_input_.move_up = move_len > 0.1f && move_z < -0.3f;
    current_input_.move_down = move_len > 0.1f && move_z > 0.3f;
    current_input_.move_left = move_len > 0.1f && move_x < -0.3f;
    current_input_.move_right = move_len > 0.1f && move_x > 0.3f;

    // Attack direction matches movement forward direction
    current_input_.attack_dir_x = forward_x;
    current_input_.attack_dir_y = forward_z;
}

void InputHandler::update_input_from_controller() {
    if (!gamepad_) {
        return;
    }

    // Read left stick for movement (raw axis values are -32768 to 32767)
    float raw_lx = SDL_GetGamepadAxis(gamepad_, SDL_GAMEPAD_AXIS_LEFTX) / 32767.0f;
    float raw_ly = SDL_GetGamepadAxis(gamepad_, SDL_GAMEPAD_AXIS_LEFTY) / 32767.0f;

    // Apply deadzone to left stick
    float left_magnitude = std::sqrt(raw_lx * raw_lx + raw_ly * raw_ly);
    if (left_magnitude > CONTROLLER_STICK_DEADZONE) {
        // Rescale to 0-1 range outside deadzone
        float scale = (left_magnitude - CONTROLLER_STICK_DEADZONE) / (1.0f - CONTROLLER_STICK_DEADZONE);
        scale = std::min(scale, 1.0f);
        controller_move_x_ = (raw_lx / left_magnitude) * scale;
        controller_move_y_ = (raw_ly / left_magnitude) * scale;
    } else {
        controller_move_x_ = 0.0f;
        controller_move_y_ = 0.0f;
    }

    // Read right stick for camera (raw axis values are -32768 to 32767)
    float raw_rx = SDL_GetGamepadAxis(gamepad_, SDL_GAMEPAD_AXIS_RIGHTX) / 32767.0f;
    float raw_ry = SDL_GetGamepadAxis(gamepad_, SDL_GAMEPAD_AXIS_RIGHTY) / 32767.0f;

    // Apply deadzone to right stick
    float right_magnitude = std::sqrt(raw_rx * raw_rx + raw_ry * raw_ry);
    if (right_magnitude > CONTROLLER_STICK_DEADZONE) {
        float scale = (right_magnitude - CONTROLLER_STICK_DEADZONE) / (1.0f - CONTROLLER_STICK_DEADZONE);
        scale = std::min(scale, 1.0f);
        controller_camera_x_ = (raw_rx / right_magnitude) * scale;
        controller_camera_y_ = (raw_ry / right_magnitude) * scale;
    } else {
        controller_camera_x_ = 0.0f;
        controller_camera_y_ = 0.0f;
    }

    // Read triggers; expose raw "primary" / "secondary" action state — clients
    // map these to game actions (attack, sprint).
    float right_trigger = SDL_GetGamepadAxis(gamepad_, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) / 32767.0f;
    float left_trigger = SDL_GetGamepadAxis(gamepad_, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) / 32767.0f;

    controller_primary_ = right_trigger > CONTROLLER_TRIGGER_DEADZONE ||
                          SDL_GetGamepadButton(gamepad_, SDL_GAMEPAD_BUTTON_SOUTH); // A button
    controller_secondary_ = left_trigger > CONTROLLER_TRIGGER_DEADZONE ||
                            SDL_GetGamepadButton(gamepad_, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER); // LB button

    // Bumpers for zoom
    if (SDL_GetGamepadButton(gamepad_, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER)) {
        camera_zoom_delta_ += 2.0f; // Zoom out
    }
}

void InputHandler::update_ui_pad_nav() {
    if (!gamepad_) {
        pad_stick_dir_x_ = 0;
        pad_stick_dir_y_ = 0;
        return;
    }

    const float lx = SDL_GetGamepadAxis(gamepad_, SDL_GAMEPAD_AXIS_LEFTX) / 32767.0f;
    const float ly = SDL_GetGamepadAxis(gamepad_, SDL_GAMEPAD_AXIS_LEFTY) / 32767.0f;

    // Per axis: latch a direction when the stick crosses k_pad_nav_enter, and
    // re-arm only once it falls back under k_pad_nav_leave. The EDGE is the
    // transition into a nonzero latch, so one flick is one navigation step no
    // matter how long the stick is held — matching the D-pad's button-down.
    const auto latch = [](float v, int8_t& dir) -> int8_t {
        const float a = v < 0.0f ? -v : v;
        if (dir != 0) {
            if (a < k_pad_nav_leave) {
                dir = 0;
            }
            return 0;
        }
        if (a < k_pad_nav_enter) {
            return 0;
        }
        dir = v > 0.0f ? int8_t{1} : int8_t{-1};
        return dir;
    };

    const int8_t ex = latch(lx, pad_stick_dir_x_);
    const int8_t ey = latch(ly, pad_stick_dir_y_);
    if (ex > 0) {
        pad_nav_right_ = true;
    } else if (ex < 0) {
        pad_nav_left_ = true;
    }
    // SDL's Y axis is positive DOWN, which is also the UI's convention.
    if (ey > 0) {
        pad_nav_down_ = true;
    } else if (ey < 0) {
        pad_nav_up_ = true;
    }
}

void InputHandler::handle_controller_added(SDL_JoystickID id) {
    // Only connect one controller at a time
    if (gamepad_) {
        return;
    }

    if (SDL_IsGamepad(id)) {
        gamepad_ = SDL_OpenGamepad(id);
        if (gamepad_) {
            gamepad_id_ = id;
            BRACKEN_LOG_INFO("input", "Controller connected: {}", SDL_GetGamepadName(gamepad_));
        }
    }
}

void InputHandler::handle_controller_removed(SDL_JoystickID id) {
    if (gamepad_ && gamepad_id_ == id) {
        BRACKEN_LOG_INFO("input", "Controller disconnected: {}", SDL_GetGamepadName(gamepad_));
        SDL_CloseGamepad(gamepad_);
        gamepad_ = nullptr;
        gamepad_id_ = 0;

        // Reset controller input state
        controller_move_x_ = 0.0f;
        controller_move_y_ = 0.0f;
        controller_camera_x_ = 0.0f;
        controller_camera_y_ = 0.0f;
        controller_primary_ = false;
        controller_secondary_ = false;
        // Drop the stick latch too, or a pad unplugged while pushed would arm
        // a phantom edge the instant the next one is plugged in.
        pad_stick_dir_x_ = 0;
        pad_stick_dir_y_ = 0;

        // Try to find another controller
        int num_joysticks = 0;
        SDL_JoystickID* joysticks = SDL_GetJoysticks(&num_joysticks);
        if (joysticks) {
            for (int i = 0; i < num_joysticks && !gamepad_; i++) {
                if (SDL_IsGamepad(joysticks[i])) {
                    handle_controller_added(joysticks[i]);
                }
            }
            SDL_free(joysticks);
        }
    }
}

InputSnapshot InputHandler::snapshot() {
    InputSnapshot s;
    s.state = current_input_;

    s.mouse_x = mouse_x_;
    s.mouse_y = mouse_y_;
    s.mouse_left_held = left_mouse_down_;
    s.mouse_left_pressed = left_mouse_pressed_edge_;
    s.mouse_left_released = left_mouse_released_edge_;
    s.mouse_right_pressed = right_mouse_pressed_edge_;

    s.camera_yaw = camera_yaw_;
    s.camera_pitch = camera_pitch_;
    s.camera_zoom_delta = camera_zoom_delta_;
    s.wheel_x = wheel_x_;
    s.wheel_y = wheel_y_;

    s.move_forward = move_forward_;
    s.move_backward = move_backward_;
    s.move_left = move_left_;
    s.move_right = move_right_;

    s.menu_toggle_pressed = menu_toggle_pressed_;
    s.menu_up_pressed = menu_up_pressed_;
    s.menu_down_pressed = menu_down_pressed_;
    s.menu_left_pressed = menu_left_pressed_;
    s.menu_right_pressed = menu_right_pressed_;
    s.menu_select_pressed = menu_select_pressed_;

    // Drain the text buffer (take_text_input semantics): the sim stage is
    // the text consumer, and each snapshot carries only the delta.
    s.text_input = std::move(text_input_buffer_);
    text_input_buffer_.clear();
    s.text_backspace_pressed = text_backspace_pressed_;
    s.text_enter_pressed = text_enter_pressed_;
    s.text_escape_pressed = text_escape_pressed_;

    s.has_controller = gamepad_ != nullptr;
    s.controller_primary = controller_primary_;
    s.controller_secondary = controller_secondary_;

    s.pad_nav_up = pad_nav_up_;
    s.pad_nav_down = pad_nav_down_;
    s.pad_nav_left = pad_nav_left_;
    s.pad_nav_right = pad_nav_right_;
    s.pad_accept_pressed = pad_accept_pressed_;
    s.pad_cancel_pressed = pad_cancel_pressed_;
    s.pad_next_pressed = pad_next_pressed_;
    s.pad_prev_pressed = pad_prev_pressed_;

    s.key_just_pressed = key_just_pressed_;

    // Held keys, POLLED (not event-tracked): an overlay that consumes a
    // key-up event cannot latch a modifier on, which is exactly the bug hosts
    // hit when they tracked "shift is down" from on_event themselves. Muted
    // while an overlay owns the keyboard, for the same reason the polled
    // movement axes are (update_input_from_keyboard).
    if (const bool* keys = SDL_GetKeyboardState(nullptr); keys != nullptr && !keyboard_captured_) {
        for (std::size_t i = 0; i < static_cast<std::size_t>(SDL_SCANCODE_COUNT); ++i) {
            s.keys_held[i] = keys[i];
        }
    }

    // Consume the mouse edges this snapshot just copied: they belong to
    // exactly one frame, and the sim stage cannot clear them (the snapshot is
    // immutable and the live handler is main-thread tier). retain_mouse_edges()
    // is the one-shot opt-out for a UI layer that reads the handler directly.
    if (retain_mouse_edges_) {
        retain_mouse_edges_ = false;
    } else {
        clear_mouse_edges();
    }
    return s;
}

} // namespace bracken
