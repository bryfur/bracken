#pragma once

#include "bracken/input_state.hpp"
#include "SDL3/SDL_gamepad.h"
#include "SDL3/SDL_joystick.h"
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_scancode.h>
#include <array>
#include <bitset>
#include <cstddef>
#include <string>

namespace bracken {

/**
 * InputSnapshot — one frame's immutable, by-value copy of the InputHandler
 * state, taken on the main thread after event pumping and carried into the
 * sim stage (SimFrame::input()). Everything here is plain data: the sim job
 * reads it freely from any pool worker while the main thread renders — no
 * reference back into the live InputHandler (tier: `main-thread`) ever
 * crosses the pipeline.
 *
 * Field names mirror the InputHandler getters they were copied from, so a
 * port is mechanical (in.move_forward() -> in.move_forward,
 * in.get_camera_yaw() -> in.camera_yaw); was_key_just_pressed keeps its
 * function form for the scancode bounds check.
 *
 * Thread-safety tier: `worker-safe` (immutable value; copy freely).
 */
struct InputSnapshot {
    /// Derived camera-relative movement/aim state (InputHandler::get_input).
    InputState state{};

    // Raw mouse position + button edges (mirror of the mouse getters). The
    // edges are CONSUMED by the snapshot that copies them (see
    // InputHandler::snapshot), so one click is exactly one snapshot's edge.
    float mouse_x = 0.0f;
    float mouse_y = 0.0f;
    bool mouse_left_held = false;
    bool mouse_left_pressed = false;
    bool mouse_left_released = false;
    bool mouse_right_pressed = false;

    /// RAW mouse-wheel delta accumulated since the previous snapshot, in SDL
    /// wheel units (one notch = 1.0), sign as SDL reports it. Distinct from
    /// `camera_zoom_delta` on purpose: that one is already COOKED for the
    /// camera (negated and multiplied by 50), so a scroll view reading it
    /// would scroll backwards at fifty times the speed. Consumed by whoever
    /// reads the snapshot; `InputHandler::reset_wheel()` clears the
    /// accumulator at the top of each event pump.
    float wheel_x = 0.0f;
    float wheel_y = 0.0f;

    // Camera orbit controls (sensitivity/inversion already applied).
    float camera_yaw = 0.0f;
    float camera_pitch = 20.0f;
    float camera_zoom_delta = 0.0f;

    // Raw movement keys (camera-relative axes live in `state`).
    bool move_forward = false;
    bool move_backward = false;
    bool move_left = false;
    bool move_right = false;

    // Menu navigation edges.
    bool menu_toggle_pressed = false;
    bool menu_up_pressed = false;
    bool menu_down_pressed = false;
    bool menu_left_pressed = false;
    bool menu_right_pressed = false;
    bool menu_select_pressed = false;

    // Text input: characters typed since the previous snapshot (the snapshot
    // drains the handler's buffer — take_text_input semantics) + edge keys.
    std::string text_input;
    bool text_backspace_pressed = false;
    bool text_enter_pressed = false;
    bool text_escape_pressed = false;

    // Controller. NOTE that `controller_primary`/`controller_secondary` are
    // LEVELS — polled from the trigger axes and the face/shoulder buttons, so
    // they stay true for as long as the input is held. That is what a game
    // action wants and exactly what a MENU does not: a UI driven off them
    // re-activates the focused widget every frame the pad is held down. Use
    // the `pad_*` edges below for anything menu-shaped.
    bool has_controller = false;
    bool controller_primary = false;
    bool controller_secondary = false;

    /**
     * Gamepad navigation EDGES for a UI. Two properties separate them from the
     * `menu_*_pressed` fields above, and both are why a pad-navigable HUD
     * could not be built on those:
     *
     * (1) They are EDGES — one press produces exactly one `true`, and holding
     *     the stick or the D-pad produces nothing further. There is
     *     deliberately NO hold-to-repeat: repeat rate is a policy decision and
     *     a menu that wants one owns it.
     * (2) They are UNGATED. `menu_*_pressed` is only ever set while the host
     *     has called `set_game_input_enabled(false)`, i.e. while the whole game
     *     is paused. A HUD is navigable during play.
     *
     * Sources: the D-pad buttons and the LEFT STICK (flicked past magnitude
     * 0.5, re-armed below 0.35 — hysteresis, so a stick resting on the
     * threshold cannot chatter) for the four directions; SOUTH for accept,
     * EAST for cancel; the shoulder buttons for tab order.
     */
    bool pad_nav_up = false;
    bool pad_nav_down = false;
    bool pad_nav_left = false;
    bool pad_nav_right = false;
    bool pad_accept_pressed = false;
    bool pad_cancel_pressed = false;
    bool pad_next_pressed = false;
    bool pad_prev_pressed = false;

    /// Per-frame key-press edges by physical scancode
    /// (InputHandler::was_key_just_pressed).
    std::array<bool, SDL_SCANCODE_COUNT> key_just_pressed{};

    /// LEVEL (held-down) state by physical scancode, polled at snapshot time.
    /// Read it through key_held(); the bitset is public only because the whole
    /// snapshot is plain data. A bitset, not another array<bool>, because the
    /// snapshot is copied into every sim frame — 64 bytes instead of 512.
    std::bitset<SDL_SCANCODE_COUNT> keys_held{};

    [[nodiscard]] bool was_key_just_pressed(SDL_Scancode scancode) const {
        return scancode >= 0 && scancode < SDL_SCANCODE_COUNT && key_just_pressed[scancode];
    }

    /// Is the key held down right now? Edges (was_key_just_pressed) fire once;
    /// this is the state a run modifier or a charge-up needs. Polled from the
    /// keyboard, so a key-up swallowed by an overlay cannot latch it on.
    [[nodiscard]] bool key_held(SDL_Scancode scancode) const {
        return scancode >= 0 && scancode < SDL_SCANCODE_COUNT && keys_held[static_cast<std::size_t>(scancode)];
    }
};

// Engine-level raw input. Tracks mouse, keyboard, controller, and
// camera-relative movement axes. Game-specific actions (attack, sprint,
// skills, panel toggles) live in a client-side InputBindings layer.
//
// Thread-safety tier: `main-thread` — owned and pumped by Application::run's
// event step (which runs at the frame fence, with no sim job in flight).
// Sim-stage code never touches the handler; it reads the frame's
// InputSnapshot through SimFrame::input().
class InputHandler {
public:
    InputHandler();
    ~InputHandler();

    // Process SDL events, returns false if quit requested
    bool process_events();

    // Process a single SDL event (for when Application polls events itself)
    void process_event(const SDL_Event& event);

    // Per-frame finalization (keyboard/controller state, change detection).
    // Call after all events have been processed via process_event().
    void post_process_events();

    // Get current input state
    const bracken::InputState& get_input() const { return current_input_; }

    // Build the frame's by-value InputSnapshot for the sim stage. Call on the
    // main thread after post_process_events(); non-const because it CONSUMES
    // what it copies:
    //   - the pending text-input buffer (take_text_input semantics), and
    //   - the mouse-button edges, so one click is exactly one snapshot's edge
    //     (retain_mouse_edges() opts the next snapshot out — see there).
    // Also polls the keyboard for the snapshot's held-key state (keys_held).
    [[nodiscard]] InputSnapshot snapshot();

    // Ask the NEXT snapshot() not to consume the mouse-button edges, so a UI
    // layer that reads them off the live handler after the snapshot still
    // sees the click. One-shot: call it every frame you need it. Without it
    // the edges belong to the snapshot that copied them.
    void retain_mouse_edges() { retain_mouse_edges_ = true; }

    // Check if input changed since last frame
    bool input_changed() const { return input_changed_; }

    // Reset input changed flag
    void reset_changed() { input_changed_ = false; }

    // Get raw mouse position
    float mouse_x() const { return mouse_x_; }
    float mouse_y() const { return mouse_y_; }

    // Left-click edges (consumed by UI before gameplay).
    bool mouse_left_pressed() const { return left_mouse_pressed_edge_; }
    bool mouse_left_released() const { return left_mouse_released_edge_; }
    bool mouse_left_held() const { return left_mouse_down_; }
    bool mouse_right_pressed() const { return right_mouse_pressed_edge_; }
    void consume_mouse_left_pressed() { left_mouse_pressed_edge_ = false; }
    // Clear per-frame mouse button edge flags. Called after all UI has had
    // a chance to consume them.
    void clear_mouse_edges() {
        left_mouse_pressed_edge_ = false;
        left_mouse_released_edge_ = false;
        right_mouse_pressed_edge_ = false;
    }

    // Camera control - third person action cam style
    float get_camera_yaw() const { return camera_yaw_; }
    float get_camera_pitch() const { return camera_pitch_; }
    float get_camera_zoom_delta() const { return camera_zoom_delta_; }
    void reset_camera_deltas() { camera_zoom_delta_ = 0.0f; }

    // Raw wheel accumulator, in SDL wheel units. Kept ALONGSIDE the cooked
    // camera_zoom_delta_ rather than replacing it: the camera controller
    // depends on the cook, and a scroll view needs the raw value. Cleared by
    // reset_wheel(), which Application::pump_events calls beside
    // reset_camera_deltas() before polling.
    float wheel_x() const { return wheel_x_; }
    float wheel_y() const { return wheel_y_; }
    void reset_wheel() {
        wheel_x_ = 0.0f;
        wheel_y_ = 0.0f;
    }

    // Set camera yaw externally (for syncing with renderer)
    void set_camera_yaw(float yaw) { camera_yaw_ = yaw; }

    // Raw movement input (camera-relative; engine derives from WASD or stick)
    bool move_forward() const { return move_forward_; }
    bool move_backward() const { return move_backward_; }
    bool move_left() const { return move_left_; }
    bool move_right() const { return move_right_; }

    // Set actual camera forward direction (accounting for shoulder offset)
    void set_camera_forward(float x, float z) {
        camera_forward_x_ = x;
        camera_forward_z_ = z;
    }

    // Menu controls
    bool menu_toggle_pressed() const { return menu_toggle_pressed_; }
    bool menu_up_pressed() const { return menu_up_pressed_; }
    bool menu_down_pressed() const { return menu_down_pressed_; }
    bool menu_left_pressed() const { return menu_left_pressed_; }
    bool menu_right_pressed() const { return menu_right_pressed_; }
    bool menu_select_pressed() const { return menu_select_pressed_; }
    void clear_menu_inputs() {
        menu_toggle_pressed_ = false;
        menu_up_pressed_ = false;
        menu_down_pressed_ = false;
        menu_left_pressed_ = false;
        menu_right_pressed_ = false;
        menu_select_pressed_ = false;
        clear_pad_nav_edges();
    }

    // ---- Gamepad UI navigation (ungated edges; see InputSnapshot) ---------
    // Kept separate from the menu_* block above because those only fire while
    // game input is DISABLED, and a HUD has to be navigable during play.
    bool pad_nav_up() const { return pad_nav_up_; }
    bool pad_nav_down() const { return pad_nav_down_; }
    bool pad_nav_left() const { return pad_nav_left_; }
    bool pad_nav_right() const { return pad_nav_right_; }
    bool pad_accept_pressed() const { return pad_accept_pressed_; }
    bool pad_cancel_pressed() const { return pad_cancel_pressed_; }
    bool pad_next_pressed() const { return pad_next_pressed_; }
    bool pad_prev_pressed() const { return pad_prev_pressed_; }

    /// Clear the pad edges. Folded into clear_menu_inputs() — which
    /// Application::pump_events already calls before polling — so no caller
    /// has to learn about it.
    void clear_pad_nav_edges() {
        pad_nav_up_ = false;
        pad_nav_down_ = false;
        pad_nav_left_ = false;
        pad_nav_right_ = false;
        pad_accept_pressed_ = false;
        pad_cancel_pressed_ = false;
        pad_next_pressed_ = false;
        pad_prev_pressed_ = false;
    }

    // Generic one-shot key press detection (true only on the frame the key was first pressed)
    bool was_key_just_pressed(SDL_Scancode scancode) const;

    // Clear per-frame key-press edge flags. Called from the application loop
    // before the next round of event polling.
    void clear_key_edges();

    // Enable/disable game input (for menu mode)
    void set_game_input_enabled(bool enabled) { game_input_enabled_ = enabled; }
    bool is_game_input_enabled() const { return game_input_enabled_; }

    // While an overlay (e.g. the dev UI) captures the keyboard, polled
    // movement keys read as released; controller input is unaffected.
    // Separate from set_game_input_enabled so the two owners can't stomp
    // each other. Set every frame by Application::run.
    void set_keyboard_captured(bool captured) { keyboard_captured_ = captured; }

    // Chat text input: when enabled, typed characters accumulate in text_input_buffer_.
    void set_text_input_enabled(bool enabled);
    bool is_text_input_enabled() const { return text_input_enabled_; }
    // Returns pending text since last call, clearing the buffer.
    std::string take_text_input() {
        std::string s = std::move(text_input_buffer_);
        text_input_buffer_.clear();
        return s;
    }
    bool text_backspace_pressed() const { return text_backspace_pressed_; }
    bool text_enter_pressed() const { return text_enter_pressed_; }
    bool text_escape_pressed() const { return text_escape_pressed_; }
    void clear_text_events() {
        text_backspace_pressed_ = false;
        text_enter_pressed_ = false;
        text_escape_pressed_ = false;
    }

    // Controller support
    bool has_controller() const { return gamepad_ != nullptr; }
    const char* get_controller_name() const;
    // Raw latched pad controls; clients map them to game actions.
    // Primary  = right trigger or south (A/cross) button.
    // Secondary = left trigger or left shoulder (LB) button.
    bool controller_primary() const { return controller_primary_; }
    bool controller_secondary() const { return controller_secondary_; }

    // Camera inversion settings
    bool is_camera_x_inverted() const { return invert_camera_x_; }
    bool is_camera_y_inverted() const { return invert_camera_y_; }
    void set_camera_x_inverted(bool inverted) { invert_camera_x_ = inverted; }
    void set_camera_y_inverted(bool inverted) { invert_camera_y_ = inverted; }
    void toggle_camera_x_invert() { invert_camera_x_ = !invert_camera_x_; }
    void toggle_camera_y_invert() { invert_camera_y_ = !invert_camera_y_; }

    // Sensitivity settings
    float get_mouse_sensitivity() const { return mouse_sensitivity_; }
    float get_controller_sensitivity() const { return controller_sensitivity_; }
    void set_mouse_sensitivity(float sens) { mouse_sensitivity_ = sens; }
    void set_controller_sensitivity(float sens) { controller_sensitivity_ = sens; }

private:
    void update_input_from_keyboard();
    void update_camera_from_mouse();
    void update_input_from_controller();
    /// Poll the left stick and turn it into the four directional EDGES a UI
    /// navigates on. Runs whatever the game-input mode (unlike
    /// update_input_from_controller), because a HUD is navigable during play.
    void update_ui_pad_nav();
    void handle_controller_added(SDL_JoystickID id);
    void handle_controller_removed(SDL_JoystickID id);

    bracken::InputState current_input_;
    bracken::InputState last_input_;
    bool input_changed_ = false;

    float mouse_x_ = 0.0f;
    float mouse_y_ = 0.0f;
    float last_mouse_x_ = 0.0f;
    float last_mouse_y_ = 0.0f;

    // Raw movement keys
    bool move_forward_ = false;
    bool move_backward_ = false;
    bool move_left_ = false;
    bool move_right_ = false;

    // Mouse button edges (independent of attack-latch so UI can consume)
    bool left_mouse_down_ = false;
    bool left_mouse_pressed_edge_ = false;
    bool left_mouse_released_edge_ = false;
    bool right_mouse_pressed_edge_ = false;
    // One-shot opt-out of snapshot()'s mouse-edge consumption (see
    // retain_mouse_edges()); cleared by the snapshot it applies to.
    bool retain_mouse_edges_ = false;
    float left_mouse_press_x_ = 0.0f;
    float left_mouse_press_y_ = 0.0f;

    // Camera orbit controls
    float camera_yaw_ = 0.0f;    // Horizontal rotation (degrees)
    float camera_pitch_ = 20.0f; // Vertical angle - lower for over-the-shoulder action view
    float camera_zoom_delta_ = 0.0f;
    float wheel_x_ = 0.0f;
    float wheel_y_ = 0.0f;
    bool right_mouse_down_ = false;

    // Actual camera forward direction (set by renderer, accounts for shoulder offset)
    float camera_forward_x_ = 0.0f;
    float camera_forward_z_ = -1.0f;

    // Menu input state
    bool menu_toggle_pressed_ = false;
    bool menu_up_pressed_ = false;
    bool menu_down_pressed_ = false;
    bool menu_left_pressed_ = false;
    bool menu_right_pressed_ = false;
    bool menu_select_pressed_ = false;
    bool game_input_enabled_ = true;
    bool keyboard_captured_ = false;

    // Gamepad UI navigation edges (see InputSnapshot's pad_* block). Cleared
    // by clear_pad_nav_edges() at the top of each poll cycle; set from the
    // D-pad button-down events and from update_ui_pad_nav()'s stick latch.
    bool pad_nav_up_ = false;
    bool pad_nav_down_ = false;
    bool pad_nav_left_ = false;
    bool pad_nav_right_ = false;
    bool pad_accept_pressed_ = false;
    bool pad_cancel_pressed_ = false;
    bool pad_next_pressed_ = false;
    bool pad_prev_pressed_ = false;
    // Latched left-stick direction, -1/0/+1 per axis. The EDGE is the
    // transition into a nonzero value; the hysteresis band between
    // k_pad_nav_leave and k_pad_nav_enter is what stops a stick resting on the
    // threshold from emitting one edge per frame.
    int8_t pad_stick_dir_x_ = 0;
    int8_t pad_stick_dir_y_ = 0;
    static constexpr float k_pad_nav_enter = 0.5f;
    static constexpr float k_pad_nav_leave = 0.35f;

    // Text input (chat)
    bool text_input_enabled_ = false;
    std::string text_input_buffer_;
    bool text_backspace_pressed_ = false;
    bool text_enter_pressed_ = false;
    bool text_escape_pressed_ = false;

    // Per-frame edge flags set on SDL_EVENT_KEY_DOWN (!repeat). Cleared at
    // the top of each event-poll cycle in process_events()/Application.
    std::array<bool, SDL_SCANCODE_COUNT> key_just_pressed_{};

    // Sensitivity settings (configurable)
    float mouse_sensitivity_ = 0.35f;
    float controller_sensitivity_ = 2.5f;
    static constexpr float CONTROLLER_STICK_DEADZONE = 0.15f;
    static constexpr float CONTROLLER_TRIGGER_DEADZONE = 0.1f;

    // Camera inversion settings (default: not inverted)
    bool invert_camera_x_ = false;
    bool invert_camera_y_ = false;

    // Controller state
    SDL_Gamepad* gamepad_ = nullptr;
    SDL_JoystickID gamepad_id_ = 0;

    // Controller input state (for smooth analog input)
    float controller_move_x_ = 0.0f;
    float controller_move_y_ = 0.0f;
    float controller_camera_x_ = 0.0f;
    float controller_camera_y_ = 0.0f;
    bool controller_primary_ = false;
    bool controller_secondary_ = false;
};

} // namespace bracken
