#pragma once

#include "bracken/input_handler.hpp" // InputSnapshot

#include <glm/glm.hpp>

namespace bracken::ui {

/**
 * InputFrame — everything the UI reacts to, in UI terms.
 *
 * A deliberate narrowing of `InputSnapshot`: the UI has no opinion about
 * movement axes or camera orbit, and a host that wants a different mapping
 * (WASD navigating a menu, a bound pad face button activating) fills this
 * struct by hand. That is why `from_snapshot` is a free factory and not a
 * constructor.
 */
struct InputFrame {
    glm::vec2 mouse{0.0f, 0.0f};
    bool mouse_down = false;
    bool mouse_pressed = false;
    bool mouse_released = false;
    /// Wheel notches this frame, sign as SDL reports (positive = scroll up).
    float wheel = 0.0f;

    // Directional navigation + the two verbs. Keyboard AND gamepad feed the
    // same fields, which is what makes every widget pad-navigable for free.
    bool nav_up = false;
    bool nav_down = false;
    bool nav_left = false;
    bool nav_right = false;
    bool nav_activate = false;
    bool nav_cancel = false;
    bool nav_next = false; // tab order forward
    bool nav_prev = false; // tab order backward

    /**
     * The default mapping.
     *
     *   mouse/wheel  -> straight through (wheel_y, NOT camera_zoom_delta,
     *                   which is already cooked for the camera)
     *   nav_up/down/ -> menu_{up,down,left,right}_pressed OR the arrow
     *   left/right      scancodes
     *   nav_activate -> menu_select_pressed OR Return OR Space OR the pad's
     *                   primary
     *   nav_cancel   -> Escape OR menu_toggle_pressed OR the pad's secondary
     *   nav_next/prev-> Tab, Shift+Tab
     */
    [[nodiscard]] static InputFrame from_snapshot(const InputSnapshot& in);
};

/**
 * InputCapture — what the UI took this frame, so the game does not act on it
 * too.
 *
 * WHY IT IS A GATE AND NOT A CONSUMPTION. Sim-stage input is an IMMUTABLE
 * by-value snapshot (`InputSnapshot`), and the live `InputHandler` is
 * `main-thread` tier, so nothing in `on_simulate` can reach back and clear an
 * edge. `InputHandler::retain_mouse_edges()` exists for a MAIN-THREAD overlay
 * and a sim-stage UI must never call it — the edge would then be delivered to
 * two consecutive snapshots and a button would fire twice.
 *
 * So consumption is a software gate: run the UI FIRST, then feed the game the
 * masked snapshot.
 *
 *     ui_.end();
 *     const InputSnapshot game_in = ui_.capture().mask(sim.input());
 *     actions_.update(game_in);        // the whole action layer goes quiet
 *     move_player(game_in);
 */
struct InputCapture {
    /// The pointer is over a UI surface, or a widget is held.
    bool mouse = false;
    /// A widget has focus, so navigation/activation keys belong to the UI.
    bool keyboard = false;
    /// The pointer is over a scrollable container.
    bool wheel = false;

    /**
     * A copy of `in` with EXACTLY the captured channels zeroed and nothing
     * else touched: mouse capture leaves `keys_held` alone, keyboard capture
     * leaves `mouse_x/mouse_y` alone (a position is not an event — clearing it
     * would teleport a host's cursor logic to the origin).
     */
    [[nodiscard]] InputSnapshot mask(const InputSnapshot& in) const;
};

} // namespace bracken::ui
