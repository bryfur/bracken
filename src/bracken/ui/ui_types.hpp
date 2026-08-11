#pragma once

#include "bracken/core/color.hpp"
#include "bracken/core/hash.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <string_view>

namespace bracken::ui {

/** Axis-aligned rectangle in SCREEN PIXELS, origin top-left, +y down — the
 *  same space every `scene::Ui*Command` uses. */
struct Rect {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;

    [[nodiscard]] constexpr float right() const noexcept { return x + w; }
    [[nodiscard]] constexpr float bottom() const noexcept { return y + h; }
    [[nodiscard]] constexpr glm::vec2 center() const noexcept { return {x + w * 0.5f, y + h * 0.5f}; }
    [[nodiscard]] constexpr bool empty() const noexcept { return w <= 0.0f || h <= 0.0f; }

    /// Half-open on the far edges, so two rects that share an edge never both
    /// claim the same pixel — the property hit testing over a laid-out row
    /// depends on.
    [[nodiscard]] constexpr bool contains(glm::vec2 p) const noexcept {
        return p.x >= x && p.x < right() && p.y >= y && p.y < bottom();
    }

    /// Overlap, or an EMPTY rect (w or h <= 0) when there is none. Empty
    /// results keep the intersection point rather than collapsing to the
    /// origin, so a clip chain that goes empty stays where it went empty.
    [[nodiscard]] constexpr Rect intersect(const Rect& o) const noexcept {
        const float nx = x > o.x ? x : o.x;
        const float ny = y > o.y ? y : o.y;
        const float nr = right() < o.right() ? right() : o.right();
        const float nb = bottom() < o.bottom() ? bottom() : o.bottom();
        return Rect{nx, ny, nr - nx, nb - ny};
    }
};

/** Per-side insets (padding, margins, nine-slice borders). */
struct Edges {
    float l = 0.0f;
    float t = 0.0f;
    float r = 0.0f;
    float b = 0.0f;

    [[nodiscard]] static constexpr Edges all(float v) noexcept { return Edges{v, v, v, v}; }
    [[nodiscard]] static constexpr Edges xy(float horizontal, float vertical) noexcept {
        return Edges{horizontal, vertical, horizontal, vertical};
    }
    [[nodiscard]] constexpr float horizontal() const noexcept { return l + r; }
    [[nodiscard]] constexpr float vertical() const noexcept { return t + b; }
};

/** Main axis of a container. Grids ignore it (they are row-major 2D). */
enum class Axis : uint8_t { Horizontal, Vertical };

/**
 * Cross-axis (and leftover-space main-axis) placement.
 *
 * `Stretch` gives a child the container's whole cross extent — but only a
 * child that did not ask for one itself. An explicit `Size::px` on the cross
 * axis WINS over the parent's Stretch, the same way flexbox only stretches an
 * `auto` cross size: otherwise a 100 px-wide toggle dropped into a Stretch
 * column would be silently widened, and there would be no way to say "this one
 * is narrow" at all. `Size::flex` on the cross axis stretches (there is no
 * leftover to weight there), and so does `Size::content` under a Stretch
 * parent.
 *
 * On the MAIN axis `Stretch` has no meaning and reads as `Start`.
 */
enum class Align : uint8_t { Start, Center, End, Stretch };

/**
 * How a box or leaf gets its size along one axis.
 *
 * - `Content`: whatever it measured (a label's text extent, a container's
 *   children).
 * - `Fixed`: exactly `value` pixels.
 * - `Flex`: `value` is a WEIGHT; the leftover space of the parent's main axis
 *   is divided among flex siblings in proportion. On the CROSS axis a Flex
 *   size behaves as `Stretch` — there is no leftover to weight there.
 */
enum class SizeMode : uint8_t { Content, Fixed, Flex };

struct Size {
    SizeMode mode = SizeMode::Content;
    float value = 0.0f;

    [[nodiscard]] static constexpr Size px(float v) noexcept { return Size{SizeMode::Fixed, v}; }
    [[nodiscard]] static constexpr Size flex(float weight = 1.0f) noexcept { return Size{SizeMode::Flex, weight}; }
    [[nodiscard]] static constexpr Size content() noexcept { return Size{SizeMode::Content, 0.0f}; }
};

/** Everything a container decides about its children. */
struct BoxStyle {
    Edges padding{};
    float spacing = 0.0f;
    /// Placement of the leftover main-axis space when NO child is Flex (a
    /// flex child eats the leftover, so this then has nothing to place).
    /// `Stretch` on the main axis behaves as `Start`.
    Align main = Align::Start;
    Align cross = Align::Start;
    Size width{};
    Size height{};
};

/**
 * A widget's identity, stable across frames — that is the entire contract,
 * and everything retained (hot, active, focus, scroll offset, last frame's
 * rect for hit testing) is keyed by it.
 *
 * THE TRAP, stated once: an id derived from something that changes every
 * frame (a formatted string containing the frame index or a live value)
 * silently resets that widget's state every frame AND leaks one retained
 * entry per frame. Build ids from a stable label, and index list rows with
 * `id(parent, i)`.
 */
struct WidgetId {
    uint64_t value = 0;

    [[nodiscard]] constexpr bool valid() const noexcept { return value != 0; }
    friend constexpr bool operator==(WidgetId a, WidgetId b) noexcept = default;
};

/// Id from a call-site label (FNV-1a 64). `0` can never be produced for a
/// non-empty label in practice; an empty label yields the invalid id.
[[nodiscard]] WidgetId id(std::string_view label) noexcept;

/// Id of the i-th item under `parent` — the list-row form.
[[nodiscard]] WidgetId id(WidgetId parent, uint64_t index) noexcept;

/// Fold a child id into a parent PATH. This is what makes the same label
/// under two different panels two different widgets.
[[nodiscard]] WidgetId combine(WidgetId parent, WidgetId child) noexcept;

/**
 * The whole theme. Deliberately one flat struct of colors and four metrics:
 * a real theming system (cascading, per-widget overrides, named tokens) is a
 * separate feature and explicitly out of scope here. Copy it, change the
 * fields you want, hand it to `Context::set_style`.
 */
struct Style {
    core::Color panel_bg{0xB0101418};
    core::Color panel_border{0xFF303840};
    core::Color text{0xFFE8ECF0};
    core::Color text_dim{0xFF8894A0};
    core::Color control_bg{0xFF303840};
    core::Color control_bg_hot{0xFF3E4A55};
    core::Color control_bg_active{0xFF556577};
    core::Color control_border{0xFF888888};
    core::Color focus_border{0xFFFFD080};
    core::Color slider_track{0xFF202830};
    core::Color slider_fill{0xFF5A8CC0};
    core::Color toggle_on{0xFF5AC08C};
    core::Color toggle_off{0xFF404850};
    /// Filled portion of a progress bar (the empty portion uses slider_track).
    core::Color progress_fill{0xFF5AC08C};
    /// Background of an editable text field, and its caret.
    core::Color field_bg{0xFF14181C};
    core::Color caret{0xFFE8ECF0};
    /// A `separator()` rule.
    core::Color separator{0xFF303840};

    float border_width = 2.0f;
    /// Padding inside a button/toggle/slider, before its label.
    Edges control_padding = Edges::xy(8.0f, 4.0f);
    /// Height of a control when its height Size is Content.
    float control_height = 26.0f;
    /// Text scale for widget labels.
    float text_scale = 1.0f;
};

} // namespace bracken::ui
