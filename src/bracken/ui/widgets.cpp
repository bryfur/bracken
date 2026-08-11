// Widget bodies for ui::Context. Split from ui_context.cpp — same class,
// same translation-unit-local invariants — because the frame machinery
// (layout, scrolling, focus, emission) and the widget vocabulary are read for
// completely different reasons and neither wants to be scrolled past.
//
// EVERY widget in this file follows the same four steps, in this order:
//   1. resolve the scoped id;
//   2. interact() against LAST frame's rectangle (see Context's class
//      comment for why it cannot be this frame's);
//   3. declare a layout leaf and record the id -> node mapping;
//   4. append deferred draws bound to that node, background first.
// A widget that skips step 3 is invisible; one that skips step 2 is inert.

#include "bracken/ui/ui_context.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

namespace bracken::ui {

void Context::label(std::string_view text, core::Color color, float scale, Size height) {
    const float s = resolved_scale(scale);
    const float w = measure_text(text, scale);
    const float h = height.mode == SizeMode::Fixed ? height.value : line_height(scale);

    const uint32_t node = tree_.leaf(Size::px(w), height.mode == SizeMode::Fixed ? height : Size::px(h), {w, h});

    DeferredDraw d{};
    d.kind = DeferredDraw::Kind::Text;
    d.node = node;
    d.color = color;
    d.scale = s;
    d.text_offset = store_text(text);
    d.text_length = static_cast<uint32_t>(text.size());
    d.frac = {0.0f, 0.0f, 0.0f, 0.0f}; // positioned at the node origin
    draws_.push_back(d);
}

void Context::label_wrapped(std::string_view text, float max_width, core::Color color, float scale) {
    const float s = resolved_scale(scale);
    const float lh = line_height(scale);

    // Height needs the line COUNT, which needs the wrap width. When the caller
    // did not give one we cannot know it before the solve, so the leaf claims
    // one line and flexes its width; the emitted text then wraps into whatever
    // the box became. That under-reports the height by design — a
    // content-sized parent cannot size to text whose width it is itself
    // deciding, and guessing would be worse than a documented limitation.
    float height = lh;
    if (max_width > 0.0f && cfg_.font != nullptr && cfg_.font->valid()) {
        uint32_t lines = 0;
        std::size_t pos = 0;
        while (pos < text.size()) {
            const auto wp = cfg_.font->wrap_point(text.substr(pos), max_width, s);
            pos += wp.consumed;
            ++lines;
        }
        height = lh * static_cast<float>(std::max(lines, 1u));
        diag_.text_fallback_advances += cfg_.font->count_fallbacks(text);
    } else if (cfg_.font == nullptr || !cfg_.font->valid()) {
        ++diag_.text_without_font;
    }

    const Size width = max_width > 0.0f ? Size::px(max_width) : Size::flex(1.0f);
    const uint32_t node = tree_.leaf(width, Size::px(height), {max_width > 0.0f ? max_width : 0.0f, height});

    DeferredDraw d{};
    d.kind = DeferredDraw::Kind::WrappedText;
    d.node = node;
    d.color = color;
    d.scale = s;
    d.text_offset = store_text(text);
    d.text_length = static_cast<uint32_t>(text.size());
    d.frac = {0.0f, 0.0f, 0.0f, 0.0f};
    d.pad = {0.0f, 0.0f, max_width, 0.0f}; // pad.z carries the explicit wrap width (0 = use the solved box)
    draws_.push_back(d);
}

bool Context::button(WidgetId local, std::string_view text, Size width, Size height) {
    const WidgetId id = scoped(local);
    bool hovered = false;
    bool held = false;
    const bool fired = interact(id, hovered, held);

    const float s = resolved_scale(0.0f);
    const float tw = measure_text(text, 0.0f);
    const float th = line_height(0.0f);
    const Edges pad{style_.control_padding.l * cfg_.ui_scale, style_.control_padding.t * cfg_.ui_scale,
                    style_.control_padding.r * cfg_.ui_scale, style_.control_padding.b * cfg_.ui_scale};
    const glm::vec2 content{tw + pad.horizontal(),
                            std::max(th + pad.vertical(), style_.control_height * cfg_.ui_scale)};

    const uint32_t node = tree_.leaf(width, height, content);
    record_rect(id, node);
    focus_order_.push_back({id, node});

    DeferredDraw bg{};
    bg.kind = DeferredDraw::Kind::FilledRect;
    bg.node = node;
    bg.color = held ? style_.control_bg_active : (hovered ? style_.control_bg_hot : style_.control_bg);
    draws_.push_back(bg);

    DeferredDraw border{};
    border.kind = DeferredDraw::Kind::RectOutline;
    border.node = node;
    border.color = state_.focused == id ? style_.focus_border : style_.control_border;
    border.scale = style_.border_width * cfg_.ui_scale;
    draws_.push_back(border);

    DeferredDraw lbl{};
    lbl.kind = DeferredDraw::Kind::TextCentered;
    lbl.node = node;
    lbl.color = style_.text;
    lbl.scale = s;
    lbl.text_offset = store_text(text);
    lbl.text_length = static_cast<uint32_t>(text.size());
    draws_.push_back(lbl);

    return fired;
}

bool Context::toggle(WidgetId local, std::string_view text, bool& value, Size width) {
    const WidgetId id = scoped(local);
    bool hovered = false;
    bool held = false;
    const bool fired = interact(id, hovered, held);
    if (fired) {
        value = !value;
    }

    const float s = resolved_scale(0.0f);
    const float tw = measure_text(text, 0.0f);
    const float th = line_height(0.0f);
    const float box = std::max(th, 12.0f * cfg_.ui_scale);
    const Edges pad{style_.control_padding.l * cfg_.ui_scale, style_.control_padding.t * cfg_.ui_scale,
                    style_.control_padding.r * cfg_.ui_scale, style_.control_padding.b * cfg_.ui_scale};
    const glm::vec2 content{box + 6.0f * cfg_.ui_scale + tw + pad.horizontal(),
                            std::max(th + pad.vertical(), style_.control_height * cfg_.ui_scale)};

    const uint32_t node = tree_.leaf(width, Size::content(), content);
    record_rect(id, node);
    focus_order_.push_back({id, node});

    DeferredDraw bg{};
    bg.kind = DeferredDraw::Kind::FilledRect;
    bg.node = node;
    bg.color = hovered || held ? style_.control_bg_hot : style_.control_bg;
    draws_.push_back(bg);

    // The tick box: a fixed-size square inset from the left edge, expressed
    // as a zero-fraction rect plus absolute pads so it does not stretch with
    // the widget.
    DeferredDraw tick{};
    tick.kind = DeferredDraw::Kind::FilledRect;
    tick.node = node;
    tick.frac = {0.0f, 0.5f, 0.0f, 0.0f};
    tick.pad = {pad.l, -box * 0.5f, box, box};
    tick.color = value ? style_.toggle_on : style_.toggle_off;
    draws_.push_back(tick);

    DeferredDraw border{};
    border.kind = DeferredDraw::Kind::RectOutline;
    border.node = node;
    border.color = state_.focused == id ? style_.focus_border : style_.control_border;
    border.scale = style_.border_width * cfg_.ui_scale;
    draws_.push_back(border);

    DeferredDraw lbl{};
    lbl.kind = DeferredDraw::Kind::Text;
    lbl.node = node;
    lbl.frac = {0.0f, 0.5f, 0.0f, 0.0f};
    lbl.pad = {pad.l + box + 6.0f * cfg_.ui_scale, -th * 0.5f, 0.0f, 0.0f};
    lbl.color = style_.text;
    lbl.scale = s;
    lbl.text_offset = store_text(text);
    lbl.text_length = static_cast<uint32_t>(text.size());
    draws_.push_back(lbl);

    return fired;
}

bool Context::slider(WidgetId local, float& value, float min, float max, Size width) {
    const WidgetId id = scoped(local);
    bool hovered = false;
    bool held = false;
    (void)interact(id, hovered, held);

    const float span = max - min;
    const float before = value;

    // Dragging uses LAST frame's rect, like every other hit test here. It is
    // also the only widget where that latency is visible as a lag between the
    // pointer and the knob, which is the honest cost of returning a result at
    // the call site.
    if (state_.active == id && span > 0.0f) {
        if (const UiState::RectEntry* prev = previous_rect(id); prev != nullptr && prev->rect.w > 0.0f) {
            const float t = std::clamp((input_.mouse.x - prev->rect.x) / prev->rect.w, 0.0f, 1.0f);
            value = min + t * span;
        }
    }
    if (state_.focused == id && span > 0.0f && (input_.nav_left || input_.nav_right)) {
        // 5% per press: enough to cross the range in twenty taps, small enough
        // to tune. A held key repeats through SDL's key-repeat, not here.
        if (input_.nav_right) {
            value = std::min(max, value + span * 0.05f);
        }
        if (input_.nav_left) {
            value = std::max(min, value - span * 0.05f);
        }
        // Claim the horizontal axis so resolve_focus does not ALSO move the
        // focus off this slider on the same press, which would make it
        // impossible to tune with a pad.
        nav_horizontal_consumed_ = true;
    }
    // NOT std::clamp(value, min, max): that is UNDEFINED BEHAVIOUR when
    // min > max, which a data-driven UI hands in the moment two settings rows
    // swap. min wins on an inverted range, and NaN falls through to `min`
    // rather than propagating into the layout as a NaN rectangle.
    value = value > max ? max : value;
    value = value < min || !(value == value) ? min : value;
    const float t = span > 0.0f ? (value - min) / span : 0.0f;

    const float h = style_.control_height * cfg_.ui_scale;
    const uint32_t node = tree_.leaf(width, Size::px(h), {120.0f * cfg_.ui_scale, h});
    record_rect(id, node);
    focus_order_.push_back({id, node});

    DeferredDraw track{};
    track.kind = DeferredDraw::Kind::FilledRect;
    track.node = node;
    track.frac = {0.0f, 0.5f, 1.0f, 0.0f};
    track.pad = {0.0f, -3.0f * cfg_.ui_scale, 0.0f, 6.0f * cfg_.ui_scale};
    track.color = style_.slider_track;
    draws_.push_back(track);

    DeferredDraw fill{};
    fill.kind = DeferredDraw::Kind::FilledRect;
    fill.node = node;
    fill.frac = {0.0f, 0.5f, t, 0.0f};
    fill.pad = {0.0f, -3.0f * cfg_.ui_scale, 0.0f, 6.0f * cfg_.ui_scale};
    fill.color = style_.slider_fill;
    draws_.push_back(fill);

    DeferredDraw knob{};
    knob.kind = DeferredDraw::Kind::FilledRect;
    knob.node = node;
    knob.frac = {t, 0.0f, 0.0f, 1.0f};
    knob.pad = {-5.0f * cfg_.ui_scale, 0.0f, 10.0f * cfg_.ui_scale, 0.0f};
    knob.color = hovered || held ? style_.control_bg_active : style_.text;
    draws_.push_back(knob);

    if (state_.focused == id) {
        DeferredDraw border{};
        border.kind = DeferredDraw::Kind::RectOutline;
        border.node = node;
        border.color = style_.focus_border;
        border.scale = style_.border_width * cfg_.ui_scale;
        draws_.push_back(border);
    }

    return value != before;
}

bool Context::radio(WidgetId local, std::string_view text, int& value, int option, Size width) {
    const WidgetId id = scoped(local);
    bool hovered = false;
    bool held = false;
    const bool fired = interact(id, hovered, held);
    const bool changed = fired && value != option;
    if (fired) {
        value = option;
    }
    const bool on = value == option;

    const float s = resolved_scale(0.0f);
    const float tw = measure_text(text, 0.0f);
    const float th = line_height(0.0f);
    const float dot = std::max(th, 12.0f * cfg_.ui_scale);
    const Edges pad{style_.control_padding.l * cfg_.ui_scale, style_.control_padding.t * cfg_.ui_scale,
                    style_.control_padding.r * cfg_.ui_scale, style_.control_padding.b * cfg_.ui_scale};
    const glm::vec2 content{dot + 6.0f * cfg_.ui_scale + tw + pad.horizontal(),
                            std::max(th + pad.vertical(), style_.control_height * cfg_.ui_scale)};

    const uint32_t node = tree_.leaf(width, Size::content(), content);
    record_rect(id, node);
    focus_order_.push_back({id, node});

    DeferredDraw bg{};
    bg.kind = DeferredDraw::Kind::FilledRect;
    bg.node = node;
    bg.color = hovered || held ? style_.control_bg_hot : style_.control_bg;
    draws_.push_back(bg);

    // The marker: the same inset square the toggle uses, but drawn only when
    // this option is the selected one. A radio's OFF state is an empty well —
    // drawing a grey square would read as a disabled toggle.
    DeferredDraw well{};
    well.kind = DeferredDraw::Kind::FilledRect;
    well.node = node;
    well.frac = {0.0f, 0.5f, 0.0f, 0.0f};
    well.pad = {pad.l, -dot * 0.5f, dot, dot};
    well.color = style_.toggle_off;
    draws_.push_back(well);
    if (on) {
        const float inset = dot * 0.25f;
        DeferredDraw mark{};
        mark.kind = DeferredDraw::Kind::FilledRect;
        mark.node = node;
        mark.frac = {0.0f, 0.5f, 0.0f, 0.0f};
        mark.pad = {pad.l + inset, -dot * 0.5f + inset, dot - inset * 2.0f, dot - inset * 2.0f};
        mark.color = style_.toggle_on;
        draws_.push_back(mark);
    }

    DeferredDraw border{};
    border.kind = DeferredDraw::Kind::RectOutline;
    border.node = node;
    border.color = state_.focused == id ? style_.focus_border : style_.control_border;
    border.scale = style_.border_width * cfg_.ui_scale;
    draws_.push_back(border);

    DeferredDraw lbl{};
    lbl.kind = DeferredDraw::Kind::Text;
    lbl.node = node;
    lbl.frac = {0.0f, 0.5f, 0.0f, 0.0f};
    lbl.pad = {pad.l + dot + 6.0f * cfg_.ui_scale, -th * 0.5f, 0.0f, 0.0f};
    lbl.color = style_.text;
    lbl.scale = s;
    lbl.text_offset = store_text(text);
    lbl.text_length = static_cast<uint32_t>(text.size());
    draws_.push_back(lbl);

    return changed;
}

bool Context::text_field(WidgetId local, std::string& value, std::size_t max_length, Size width) {
    const WidgetId id = scoped(local);
    bool hovered = false;
    bool held = false;
    (void)interact(id, hovered, held); // click-to-focus; the fire itself is not a commit

    bool committed = false;
    if (state_.focused == id) {
        if (input_.backspace && !value.empty()) {
            // Back off a whole UTF-8 sequence, never a continuation byte —
            // deleting one byte of a 3-byte character leaves the string
            // undecodable and the next measure charges the fallback advance
            // for the wreckage.
            std::size_t n = value.size() - 1;
            while (n > 0 && (static_cast<unsigned char>(value[n]) & 0xC0u) == 0x80u) {
                --n;
            }
            value.resize(n);
        }
        if (!input_.text.empty() && value.size() + input_.text.size() <= max_length) {
            // All-or-nothing on the whole typed run: the run is already a
            // sequence of complete characters, and truncating it to fit would
            // be the byte-splitting bug backspace goes out of its way to avoid.
            value.append(input_.text);
        }
        committed = input_.enter;
    }

    const float s = resolved_scale(0.0f);
    const float tw = measure_text(value, 0.0f);
    const float th = line_height(0.0f);
    const Edges pad{style_.control_padding.l * cfg_.ui_scale, style_.control_padding.t * cfg_.ui_scale,
                    style_.control_padding.r * cfg_.ui_scale, style_.control_padding.b * cfg_.ui_scale};
    const glm::vec2 content{std::max(tw + pad.horizontal(), 120.0f * cfg_.ui_scale),
                            std::max(th + pad.vertical(), style_.control_height * cfg_.ui_scale)};

    const uint32_t node = tree_.leaf(width, Size::content(), content);
    record_rect(id, node);
    focus_order_.push_back({id, node});

    DeferredDraw bg{};
    bg.kind = DeferredDraw::Kind::FilledRect;
    bg.node = node;
    bg.color = style_.field_bg;
    draws_.push_back(bg);

    DeferredDraw border{};
    border.kind = DeferredDraw::Kind::RectOutline;
    border.node = node;
    border.color = state_.focused == id ? style_.focus_border : style_.control_border;
    border.scale = style_.border_width * cfg_.ui_scale;
    draws_.push_back(border);

    DeferredDraw txt{};
    txt.kind = DeferredDraw::Kind::Text;
    txt.node = node;
    txt.frac = {0.0f, 0.5f, 0.0f, 0.0f};
    txt.pad = {pad.l, -th * 0.5f, 0.0f, 0.0f};
    txt.color = style_.text;
    txt.scale = s;
    txt.text_offset = store_text(value);
    txt.text_length = static_cast<uint32_t>(value.size());
    draws_.push_back(txt);

    if (state_.focused == id) {
        // Solid, not blinking: a blink needs wall-clock time and begin() is
        // deliberately not given any (a frame's UI must be a pure function of
        // its declarations). Documented on the method.
        DeferredDraw car{};
        car.kind = DeferredDraw::Kind::FilledRect;
        car.node = node;
        car.frac = {0.0f, 0.5f, 0.0f, 0.0f};
        car.pad = {pad.l + tw, -th * 0.5f, std::max(1.0f, cfg_.ui_scale), th};
        car.color = style_.caret;
        draws_.push_back(car);
    }

    return committed;
}

void Context::progress_bar(float t, Size width, float height) {
    const float clamped = std::clamp(t, 0.0f, 1.0f);
    const float h = height > 0.0f ? height * cfg_.ui_scale : 8.0f * cfg_.ui_scale;
    const uint32_t node = tree_.leaf(width, Size::px(h), {120.0f * cfg_.ui_scale, h});

    DeferredDraw track{};
    track.kind = DeferredDraw::Kind::FilledRect;
    track.node = node;
    track.color = style_.slider_track;
    draws_.push_back(track);

    if (clamped > 0.0f) {
        DeferredDraw fill{};
        fill.kind = DeferredDraw::Kind::FilledRect;
        fill.node = node;
        fill.frac = {0.0f, 0.0f, clamped, 1.0f};
        fill.color = style_.progress_fill;
        draws_.push_back(fill);
    }
}

void Context::separator(float pad) {
    const float p = pad * cfg_.ui_scale;
    const float thickness = std::max(1.0f, cfg_.ui_scale);
    // A flex-width leaf so it spans whatever the container gives it, with the
    // rule drawn as a thin band centred in a taller leaf — the clearance is
    // part of the widget rather than something the caller has to add.
    const uint32_t node = tree_.leaf(Size::flex(1.0f), Size::px(thickness + p * 2.0f), {0.0f, thickness + p * 2.0f});
    DeferredDraw d{};
    d.kind = DeferredDraw::Kind::FilledRect;
    d.node = node;
    d.frac = {0.0f, 0.5f, 1.0f, 0.0f};
    d.pad = {0.0f, -thickness * 0.5f, 0.0f, thickness};
    d.color = style_.separator;
    draws_.push_back(d);
}

void Context::image(UiTextureId texture, glm::vec2 size, core::Color tint) {
    const uint32_t node = tree_.leaf(Size::px(size.x), Size::px(size.y), size);
    DeferredDraw d{};
    d.kind = DeferredDraw::Kind::TexturedQuad;
    d.node = node;
    d.texture = texture;
    d.color = tint;
    draws_.push_back(d);
}

void Context::spacer(float pixels) {
    // Square, so it works as a gap on either axis without the caller having to
    // know which way the enclosing container runs.
    const float p = pixels * cfg_.ui_scale;
    (void)tree_.leaf(Size::px(p), Size::px(p), {p, p});
}

void Context::rect(glm::vec2 size, core::Color color) {
    const uint32_t node = tree_.leaf(Size::px(size.x), Size::px(size.y), size);
    DeferredDraw d{};
    d.kind = DeferredDraw::Kind::FilledRect;
    d.node = node;
    d.color = color;
    draws_.push_back(d);
}

} // namespace bracken::ui
