#pragma once

#include "bracken/contracts/font_metrics.hpp"
#include "bracken/contracts/ui_texture.hpp"
#include "bracken/scene/frame_builder.hpp"
#include "bracken/ui/layout.hpp"
#include "bracken/ui/nine_slice.hpp"
#include "bracken/ui/ui_input.hpp"
#include "bracken/ui/ui_types.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bracken::ui {

/** What a Context needs to know that is not per-frame. */
struct ContextConfig {
    /**
     * The frame's glyph metrics — normally `&sim.font_metrics()`. Null (or an
     * invalid table) means the UI cannot measure text: labels still emit, but
     * every text extent is 0, so content-sized boxes collapse. Counted in
     * `Diagnostics::text_without_font` so this is visible rather than
     * mysterious.
     */
    const FontMetrics* font = nullptr;
    /// Multiplies every style metric and text scale. 1.0 is the authored size.
    float ui_scale = 1.0f;
    /// Focus wraps from the last widget to the first (and back).
    bool wrap_focus = true;
    /**
     * Frames a retained entry (scroll offset, last rect, focus) survives after
     * its widget stops being declared. Bounded so an id that changes every
     * frame leaks memory at a fixed ceiling instead of forever — and the
     * leak itself shows up as `Diagnostics::retained_entries` growing.
     */
    uint32_t retain_frames = 120;
};

/**
 * UiState — everything the UI remembers BETWEEN frames.
 *
 * Host-owned (a `Context` holds one, and the host owns the Context), so a
 * game that wants two independent UI surfaces gets two independent sets of
 * hot/active/focus/scroll state without a global anywhere.
 *
 * Every container is a SORTED VECTOR, never a hash map. Emission order must
 * be a pure function of declaration order for the frame to stay byte-
 * identical at any worker count, and an unordered container's iteration order
 * is neither pinned nor stable across standard-library versions.
 */
struct UiState {
    struct ScrollEntry {
        uint64_t id = 0;
        glm::vec2 offset{0.0f, 0.0f};
        /// Last frame's solved viewport. Retained for the SAME reason widget
        /// rectangles are: a widget declared inside this container is hit
        /// tested during declaration, before the solve, so the clip it is
        /// tested against has to be last frame's too. Seeding the current
        /// frame's clip slot from it is what makes a button inside a scroll
        /// view clickable at all.
        Rect viewport{};
        uint64_t last_frame = 0;
    };
    /// Last frame's solved rectangle for a widget, plus the clip it was
    /// declared under. Hit testing uses THESE — see Context's class comment.
    struct RectEntry {
        uint64_t id = 0;
        Rect rect{};
        Rect clip{};
        bool clipped = false;
        uint64_t last_frame = 0;
    };

    WidgetId hot{};    ///< hovered this frame
    WidgetId active{}; ///< held since the press landed on it
    WidgetId focused{};
    std::vector<ScrollEntry> scroll;
    std::vector<RectEntry> rects;
    uint64_t frame = 0;

    /// Drop entries not touched for `retain_frames`. Called by Context::end().
    void retire(uint32_t retain_frames);
};

/**
 * Context — immediate-mode declarations, retained layout, retained
 * interaction.
 *
 * THE SHAPE, and why it is not pure immediate mode. An immediate-mode call
 * cannot place a widget when it is declared: the rectangle depends on
 * siblings that do not exist yet. So a declaration appends to a `LayoutTree`
 * and a deferred draw list, and `end()` solves, resolves focus and scrolling,
 * and only THEN emits into the frame's `UiLane`. Nothing reaches the packet
 * before `end()` returns.
 *
 * HIT TESTING USES LAST FRAME'S RECTANGLE, and this is the one thing to
 * understand before using the API. `button()` has to return its result at the
 * call site, which is before the solve; the rectangle it tests against is
 * therefore the one that widget occupied last frame (stored in `UiState`).
 * Consequences, all of them intentional:
 *   - a widget cannot be clicked on the first frame it exists;
 *   - clicking a widget that MOVED between frames uses where it was, not
 *     where it will be;
 *   - the effect is one frame of latency, on top of the one frame the render
 *     pipeline already has (`Application`, depth 2).
 * The alternative — returning results from `end()` — makes the call site
 * unable to branch on a click, which is the entire ergonomic point.
 *
 * FOCUS resolves at `end()` (it needs solved rectangles), so a navigation
 * key pressed during frame N moves the focus that frame and the newly focused
 * widget can be ACTIVATED from frame N+1. Same one-frame rule, same reason.
 *
 * Thread-safety tier: `owned`. One Context per UI surface, driven from ONE
 * sim job: it emits into ONE `FrameBuilder` lane, and a lane aborts if two
 * threads are inside it at once. Never fan a UI build out with
 * `SimFrame::parallel_for` into a shared lane.
 */
class Context {
public:
    struct Diagnostics {
        /// `end_box()` calls with no matching `begin_*`.
        uint32_t unbalanced_end_box = 0;
        /// Containers opened past `LayoutTree::k_max_depth`.
        uint32_t depth_overflow = 0;
        /// `end_scroll()` calls with no matching `begin_scroll()`.
        uint32_t clip_stack_underflow = 0;
        /// Two widgets claiming one id in a frame. They then SHARE hot/active
        /// state, which reads as one button firing when you click another.
        uint32_t duplicate_widget_id = 0;
        /// Text declared with no usable FontMetrics: it emits, but measures 0.
        uint32_t text_without_font = 0;
        /// Codepoints measured with `FontMetrics::fallback_advance` — text
        /// outside the exported range, which will overrun its box.
        uint32_t text_fallback_advances = 0;
        /// Retained entries currently held (rects + scroll). Watch it for the
        /// unstable-id leak described on WidgetId.
        uint32_t retained_entries = 0;
    };

    Context() = default;

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    void configure(const ContextConfig& cfg);
    [[nodiscard]] const ContextConfig& config() const noexcept { return cfg_; }

    void set_style(const Style& s) { style_ = s; }
    [[nodiscard]] const Style& style() const noexcept { return style_; }
    [[nodiscard]] Style& style() noexcept { return style_; }

    [[nodiscard]] UiState& state() noexcept { return state_; }
    [[nodiscard]] const UiState& state() const noexcept { return state_; }

    /**
     * Start the frame. Emits nothing — every declaration between here and
     * `end()` builds the layout tree and the deferred draw list.
     */
    void begin(scene::FrameBuilder::UiLane lane, const InputFrame& input, const Rect& viewport);

    /** Solve, resolve scrolling and focus, then emit every deferred draw in
     *  declaration order. After this the packet has the frame's UI. */
    void end();

    /** Valid after `end()`. Feed `capture().mask(sim.input())` to the game. */
    [[nodiscard]] const InputCapture& capture() const noexcept { return capture_; }

    // ---- containers ----------------------------------------------------

    void begin_column(const BoxStyle& style = {});
    void begin_row(const BoxStyle& style = {});
    void begin_grid(uint32_t columns, const BoxStyle& style = {});
    void end_box();

    /**
     * A container with a background, drawn BEHIND its children. The
     * background is recorded before the children are declared, which — since
     * the renderer now paints UI in strict submission order — is exactly what
     * puts it behind them. Before the ordering fix a panel could not cover a
     * label no matter when it was recorded.
     */
    void begin_panel(const BoxStyle& style = {}, Axis axis = Axis::Vertical);
    /** Nine-slice variant: the background is a sprite instead of a fill. */
    void begin_panel_sprite(UiTextureId texture, const NineSlice& slice, const BoxStyle& style = {},
                            core::Color tint = core::Color::white(), Axis axis = Axis::Vertical);
    void end_panel();

    /**
     * A clipped, scrollable container. `id` must be STABLE — the scroll
     * offset is retained under it. Emits a clip command (intersected with any
     * enclosing clip) around its body and restores the previous clip after.
     * The wheel scrolls it while the pointer is inside; focus navigation
     * scrolls the focused child into view.
     */
    void begin_scroll(WidgetId id, const BoxStyle& style = {}, Axis axis = Axis::Vertical);
    void end_scroll();

    /** Push/pop an id SCOPE, so the same label under two panels is two
     *  widgets. `begin_scroll` pushes its own id automatically. */
    void push_id(WidgetId scope);
    void pop_id();

    // ---- widgets --------------------------------------------------------

    /**
     * A single line of text. `height` overrides the measured line height —
     * a fixed-height label is how you reproduce an existing hand-positioned
     * HUD row for row without giving up the layout engine.
     */
    void label(std::string_view text, core::Color color = core::Color{0xFFE8ECF0}, float scale = 0.0f,
               Size height = Size::content());

    /** Word-wrapped against the real metrics. `max_width` <= 0 means "the
     *  width this box ends up with", resolved at solve time. */
    void label_wrapped(std::string_view text, float max_width = 0.0f, core::Color color = core::Color{0xFFE8ECF0},
                       float scale = 0.0f);

    [[nodiscard]] bool button(WidgetId id, std::string_view label, Size width = Size::content(),
                              Size height = Size::content());

    /** Checkbox/toggle. Writes `value` and returns true on the frame it
     *  changed. */
    [[nodiscard]] bool toggle(WidgetId id, std::string_view label, bool& value, Size width = Size::content());

    /** Horizontal slider. Writes `value` (clamped to [min, max]) and returns
     *  true on any frame it changed. Left/Right nav steps it by 5% when it
     *  has focus. */
    [[nodiscard]] bool slider(WidgetId id, float& value, float min, float max, Size width = Size::flex(1.0f));

    void image(UiTextureId texture, glm::vec2 size, core::Color tint = core::Color::white());
    void spacer(float pixels);
    void rect(glm::vec2 size, core::Color color);

    // ---- focus ----------------------------------------------------------

    void set_focus(WidgetId id);
    [[nodiscard]] WidgetId focused() const noexcept { return state_.focused; }
    [[nodiscard]] WidgetId hot() const noexcept { return state_.hot; }
    [[nodiscard]] WidgetId active() const noexcept { return state_.active; }

    // ---- introspection (tests + dev overlay; headless) ------------------

    /** This frame's solved rectangle, valid after `end()`. */
    [[nodiscard]] Rect rect_of(WidgetId id) const;
    [[nodiscard]] std::size_t emitted_command_count() const noexcept { return emitted_; }
    [[nodiscard]] Diagnostics diagnostics() const noexcept { return diag_; }
    /// Focusable widgets of the last frame, in DECLARATION order — the tab
    /// order. Exposed so a test can assert it without a screen.
    [[nodiscard]] std::vector<WidgetId> focus_order() const;

private:
    /// A draw whose geometry is a sub-rectangle of a layout node, resolved
    /// after the solve. `frac` is a fraction of the node rect and `pad` an
    /// absolute pixel adjustment, which is enough to express a slider knob or
    /// an inset border without a widget-specific draw kind per widget.
    struct DeferredDraw {
        enum class Kind : uint8_t {
            FilledRect,
            RectOutline,
            Text,
            TextCentered,
            WrappedText,
            TexturedQuad,
            NineSlicePanel,
            ClipPush,
            ClipPop
        };
        Kind kind = Kind::FilledRect;
        uint32_t node = LayoutTree::k_invalid;
        glm::vec4 frac{0.0f, 0.0f, 1.0f, 1.0f};
        glm::vec4 pad{0.0f, 0.0f, 0.0f, 0.0f};
        core::Color color{};
        float scale = 1.0f; // text scale, or outline line width
        uint32_t text_offset = 0;
        uint32_t text_length = 0;
        UiTextureId texture{};
        glm::vec4 uv{0.0f, 0.0f, 1.0f, 1.0f};
        NineSlice slice{};
        /// ClipPush/ClipPop: index into clip_rects_ resolved at emit.
        uint32_t clip_slot = 0;
    };

    struct FocusEntry {
        WidgetId id{};
        uint32_t node = LayoutTree::k_invalid;
        uint32_t scroll_index = 0xFFFFFFFFu; // enclosing scroll container, if any
    };

    struct ScrollNode {
        WidgetId id{};
        uint32_t node = LayoutTree::k_invalid;
        uint32_t clip_slot = 0;
        Axis axis = Axis::Vertical;
        Rect viewport{};
    };

    // ---- helpers --------------------------------------------------------

    [[nodiscard]] WidgetId scoped(WidgetId local) const;
    [[nodiscard]] float measure_text(std::string_view text, float scale);
    [[nodiscard]] float line_height(float scale) const;
    [[nodiscard]] float resolved_scale(float scale) const;
    [[nodiscard]] uint32_t store_text(std::string_view text);
    [[nodiscard]] std::string_view text_at(uint32_t offset, uint32_t length) const;
    [[nodiscard]] const UiState::RectEntry* previous_rect(WidgetId id) const;
    void record_rect(WidgetId id, uint32_t node);
    [[nodiscard]] Rect current_clip() const;
    [[nodiscard]] Rect resolve(const DeferredDraw& d) const;
    /// hot/active bookkeeping shared by every clickable widget. Returns true
    /// when the widget fired this frame.
    [[nodiscard]] bool interact(WidgetId id, bool& out_hovered, bool& out_held);
    void resolve_scrolling();
    void resolve_focus();
    void emit();

    ContextConfig cfg_{};
    Style style_{};
    UiState state_{};

    LayoutTree tree_;
    std::vector<DeferredDraw> draws_;
    std::vector<Rect> clip_rects_;
    std::vector<uint32_t> clip_stack_; // indices into clip_rects_
    std::vector<ScrollNode> scroll_nodes_;
    std::vector<uint32_t> scroll_stack_; // indices into scroll_nodes_
    std::vector<FocusEntry> focus_order_;
    std::vector<WidgetId> id_stack_;
    std::vector<uint64_t> seen_ids_; // per-frame duplicate detection (sorted)
    /// Declaration-order character store, so a `label(std::format(...))`
    /// temporary is copied immediately instead of dangling until `end()`.
    std::string text_arena_;

    /// Bound by begin(), cleared by end(). Optional rather than a pointer
    /// because UiLane's constructor is private to FrameBuilder — a Context
    /// can only ever hold one it was handed.
    std::optional<scene::FrameBuilder::UiLane> lane_;
    bool in_frame_ = false;
    InputFrame input_{};
    Rect viewport_{};
    InputCapture capture_{};
    /// A focused slider owns the horizontal navigation axis for the frame it
    /// uses it, so Left/Right tune the value instead of also jumping the focus
    /// to another widget. Set by slider(), consumed by resolve_focus().
    bool nav_horizontal_consumed_ = false;
    Diagnostics diag_{};
    std::size_t emitted_ = 0;
    /// Widgets declared this frame, for rect_of() and the retained store.
    std::vector<std::pair<uint64_t, uint32_t>> frame_rects_;
};

} // namespace bracken::ui
