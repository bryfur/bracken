#pragma once

#include "bracken/input_handler.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace bracken {

/** Where an action's input comes from. */
enum class BindingSource : uint8_t {
    Key,              ///< physical scancode (layout-independent)
    MouseButton,      ///< see MouseButton
    ControllerButton, ///< see ControllerButton
};

enum class MouseButton : uint8_t { Left, Right };

/** The controller inputs `InputSnapshot` currently exposes. Deliberately not a
 *  full gamepad map: binding to something the snapshot does not carry would be
 *  a silent dead binding, which is worse than not offering it. */
enum class ControllerButton : uint8_t { Primary, Secondary };

/**
 * One way to trigger an action. An action may have several — that is the
 * point: "jump" is Space AND the controller's primary button, and both should
 * work without the host writing `||` at every call site.
 */
struct Binding {
    BindingSource source = BindingSource::Key;
    /// Scancode / MouseButton / ControllerButton, depending on `source`.
    int code = 0;

    [[nodiscard]] static Binding key(SDL_Scancode sc) {
        return {BindingSource::Key, static_cast<int>(sc)};
    }
    [[nodiscard]] static Binding mouse(MouseButton b) {
        return {BindingSource::MouseButton, static_cast<int>(b)};
    }
    [[nodiscard]] static Binding controller(ControllerButton b) {
        return {BindingSource::ControllerButton, static_cast<int>(b)};
    }

    friend bool operator==(Binding, Binding) noexcept = default;
};

/** Index into an ActionMap's action list. Stable for the map's lifetime. */
struct ActionId {
    static constexpr uint16_t k_invalid = 0xFFFFu;
    uint16_t value = k_invalid;
    [[nodiscard]] bool valid() const noexcept { return value != k_invalid; }
    friend bool operator==(ActionId, ActionId) noexcept = default;
};

/**
 * Named game actions, bound to physical inputs.
 *
 * WHY THIS IS IN THE ENGINE. `InputHandler` is raw input by design, and its
 * own header used to tell hosts that "game-specific actions live in a
 * client-side bindings layer". Every game then writes the same layer: a
 * scancode table, an is-this-held query, edge detection, and — the part
 * everyone postpones and then retrofits painfully — rebinding and its file
 * format. None of that is game-specific. What IS game-specific is the *names*,
 * and those stay with the host: the engine never learns what "attack" means,
 * it just reports that the thing the host called "attack" went down this frame.
 *
 * USAGE. Register actions once, bind them, then call update() with each
 * frame's snapshot:
 *
 *     ActionId jump = actions.add("jump");
 *     actions.bind(jump, Binding::key(SDL_SCANCODE_SPACE));
 *     actions.bind(jump, Binding::controller(ControllerButton::Primary));
 *     ...
 *     actions.update(sim.input());          // once per sim frame
 *     if (actions.pressed(jump)) { ... }
 *
 * Tier: owned. An ActionMap is host state — build it on the main thread,
 * update and query it from the sim stage. It touches no engine globals.
 */
class ActionMap {
public:
    /** Register an action, or return the existing id if the name is already
     *  registered. Idempotent so a host can call it from several places
     *  without bookkeeping. */
    ActionId add(std::string_view name);

    /** Look up a previously registered action. Returns an invalid id if the
     *  name is unknown — querying with it is safe and always false, so a typo
     *  degrades to "never triggers" rather than to undefined behavior. */
    [[nodiscard]] ActionId find(std::string_view name) const;

    [[nodiscard]] std::string_view name_of(ActionId id) const;
    [[nodiscard]] std::size_t action_count() const noexcept { return actions_.size(); }

    /** Add a trigger. Binding the same thing twice is a no-op rather than an
     *  error — re-running setup code should not accumulate duplicates. */
    void bind(ActionId id, Binding binding);
    void unbind(ActionId id, Binding binding);
    void clear_bindings(ActionId id);
    [[nodiscard]] std::span<const Binding> bindings(ActionId id) const;

    /**
     * Rebind for a settings screen: point `binding` at `id` and remove it from
     * whatever else had it.
     *
     * The steal is the whole reason this is not just bind(): a rebinding UI
     * that lets two actions share one key produces a bug report that reads
     * "pressing E does two things", and the fix belongs here rather than in
     * every game's options menu.
     */
    void rebind_exclusive(ActionId id, Binding binding);

    /** Evaluate every action against this frame's input. Call once per frame,
     *  before any query — edges are computed by comparing against the previous
     *  call. */
    void update(const InputSnapshot& input);

    [[nodiscard]] bool held(ActionId id) const;
    /** Went down between the previous update() and this one. */
    [[nodiscard]] bool pressed(ActionId id) const;
    /** Came up between the previous update() and this one. */
    [[nodiscard]] bool released(ActionId id) const;

    /** Digital axis in [-1, 1]: +1 when only `positive` is held, -1 when only
     *  `negative` is, 0 when both or neither are. Opposed keys cancelling is
     *  what players expect from a keyboard. */
    [[nodiscard]] float axis(ActionId negative, ActionId positive) const;

    /**
     * Two axes as a vector, normalized so diagonal movement is not faster than
     * straight movement — the oldest bug in keyboard movement code.
     */
    [[nodiscard]] glm::vec2 vector(ActionId left, ActionId right, ActionId down,
                                   ActionId up) const;

    // ---- Rebinding persistence ------------------------------------------

    /** Write bindings as `action=source:code,...` lines. Actions are written
     *  by NAME, so the file survives a host reordering its add() calls. */
    [[nodiscard]] bool save(const std::string& path) const;

    /** Read bindings back. Unknown action names are skipped with a warning
     *  rather than failing the load: a settings file from an older build that
     *  had an action you have since removed should still restore the rest.
     *  Actions present in the map but absent from the file keep their current
     *  bindings, so a newly added action keeps its default. */
    [[nodiscard]] bool load(const std::string& path);

private:
    struct Action {
        std::string name;
        std::vector<Binding> bindings;
    };

    [[nodiscard]] bool evaluate(const Binding& b, const InputSnapshot& input) const;

    std::vector<Action> actions_;
    // Parallel to actions_: current and previous held state. Two vectors of
    // bytes rather than a bitset because the count is host-defined and small.
    std::vector<uint8_t> held_;
    std::vector<uint8_t> prev_held_;
};

} // namespace bracken
