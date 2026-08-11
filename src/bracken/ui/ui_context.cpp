#include "bracken/ui/ui_context.hpp"

#include "bracken/core/assert.hpp"

#include <algorithm>
#include <cmath>

namespace bracken::ui {

// ============================================================================
// UiState
// ============================================================================

void UiState::retire(uint32_t retain_frames) {
    const uint64_t cutoff = frame > retain_frames ? frame - retain_frames : 0;
    const auto stale_scroll = [cutoff](const ScrollEntry& e) { return e.last_frame < cutoff; };
    scroll.erase(std::remove_if(scroll.begin(), scroll.end(), stale_scroll), scroll.end());
    const auto stale_rect = [cutoff](const RectEntry& e) { return e.last_frame < cutoff; };
    rects.erase(std::remove_if(rects.begin(), rects.end(), stale_rect), rects.end());
}

// ============================================================================
// Context — frame lifecycle
// ============================================================================

void Context::configure(const ContextConfig& cfg) {
    cfg_ = cfg;
}

void Context::begin(scene::FrameBuilder::UiLane lane, const InputFrame& input, const Rect& viewport) {
    // A HOST mistake, so a CONTRACT (live in release) rather than an assert:
    // an unbalanced begin/end silently discards a frame's declarations and the
    // symptom — a UI that intermittently vanishes — points nowhere near here.
    BRACKEN_CONTRACT(!in_frame_, "ui::Context::begin called twice without an end()");
    // TIER. There is deliberately NO stage assertion on this path. The Context
    // touches nothing tier-bound: it reaches SDL and the GPU through exactly
    // nothing (pinned by ui/ui_purity.test.cpp, which scans the module's
    // includes), and its one engine dependency — the UiLane it emits into —
    // carries its own release-mode contracts (assert_filling, and EmitGuard's
    // two-threads-in-one-lane abort). Asserting Stage::Sim here would be a lie
    // in the other direction: a host is free to drive a Context from a
    // main-thread tool as long as it owns the lane, and every test in this
    // module does exactly that.
    lane_.emplace(lane);
    input_ = input;
    viewport_ = viewport;
    in_frame_ = true;

    tree_.clear();
    draws_.clear();
    clip_rects_.clear();
    clip_stack_.clear();
    scroll_nodes_.clear();
    scroll_stack_.clear();
    focus_order_.clear();
    id_stack_.clear();
    seen_ids_.clear();
    text_arena_.clear();
    frame_rects_.clear();
    diag_ = {};
    capture_ = {};
    nav_horizontal_consumed_ = false;
    emitted_ = 0;

    ++state_.frame;
    state_.hot = {};
}

void Context::end() {
    BRACKEN_CONTRACT(in_frame_, "ui::Context::end called without a begin()");

    tree_.solve(viewport_);
    // Focus BEFORE scrolling: directional navigation compares solved rect
    // centres, and a scroll translation applied first would make "the widget
    // below this one" mean "below where it currently scrolls to". Scrolling
    // then gets to bring the newly focused widget into view in the same frame.
    resolve_focus();
    resolve_scrolling();
    emit();

    // Record this frame's rectangles for NEXT frame's hit testing (see the
    // class comment: results have to be available at the call site, which is
    // before the solve).
    for (const auto& [raw_id, node] : frame_rects_) {
        const Rect r = tree_.rect_of(node);
        const auto it = std::lower_bound(state_.rects.begin(), state_.rects.end(), raw_id,
                                         [](const UiState::RectEntry& e, uint64_t k) { return e.id < k; });
        if (it != state_.rects.end() && it->id == raw_id) {
            it->rect = r;
            it->last_frame = state_.frame;
        } else {
            state_.rects.insert(it, UiState::RectEntry{raw_id, r, Rect{}, false, state_.frame});
        }
    }
    state_.retire(cfg_.retain_frames);
    diag_.retained_entries = static_cast<uint32_t>(state_.rects.size() + state_.scroll.size());
    diag_.unbalanced_end_box += tree_.unbalanced_end_box();
    diag_.depth_overflow += tree_.depth_overflow();

    capture_.keyboard = state_.focused.valid();

    lane_.reset();
    in_frame_ = false;
}

// ============================================================================
// Helpers
// ============================================================================

WidgetId Context::scoped(WidgetId local) const {
    return id_stack_.empty() ? local : combine(id_stack_.back(), local);
}

float Context::resolved_scale(float scale) const {
    const float s = scale > 0.0f ? scale : style_.text_scale;
    return s * cfg_.ui_scale;
}

float Context::line_height(float scale) const {
    if (cfg_.font == nullptr || !cfg_.font->valid()) {
        return 18.0f * resolved_scale(scale);
    }
    return cfg_.font->measure_height(resolved_scale(scale));
}

float Context::measure_text(std::string_view text, float scale) {
    if (cfg_.font == nullptr || !cfg_.font->valid()) {
        ++diag_.text_without_font;
        return 0.0f;
    }
    diag_.text_fallback_advances += cfg_.font->count_fallbacks(text);
    return cfg_.font->measure_width(text, resolved_scale(scale));
}

uint32_t Context::store_text(std::string_view text) {
    const auto offset = static_cast<uint32_t>(text_arena_.size());
    text_arena_.append(text);
    return offset;
}

std::string_view Context::text_at(uint32_t offset, uint32_t length) const {
    if (offset + length > text_arena_.size()) {
        return {};
    }
    return std::string_view(text_arena_).substr(offset, length);
}

const UiState::RectEntry* Context::previous_rect(WidgetId id) const {
    const auto it = std::lower_bound(state_.rects.begin(), state_.rects.end(), id.value,
                                     [](const UiState::RectEntry& e, uint64_t k) { return e.id < k; });
    if (it == state_.rects.end() || it->id != id.value) {
        return nullptr;
    }
    return &*it;
}

void Context::record_rect(WidgetId id, uint32_t node) {
    if (!id.valid() || node == LayoutTree::k_invalid) {
        return;
    }
    // Duplicate detection: two widgets under one id SHARE hot/active, which
    // reads as the wrong button firing. Counted, never silently tolerated.
    const auto it = std::lower_bound(seen_ids_.begin(), seen_ids_.end(), id.value);
    if (it != seen_ids_.end() && *it == id.value) {
        ++diag_.duplicate_widget_id;
    } else {
        seen_ids_.insert(it, id.value);
    }
    frame_rects_.emplace_back(id.value, node);
    // Remember the clip a widget was declared under so next frame's hit test
    // can reject a click that landed on a scrolled-away row.
    const auto rit = std::lower_bound(state_.rects.begin(), state_.rects.end(), id.value,
                                      [](const UiState::RectEntry& e, uint64_t k) { return e.id < k; });
    const Rect clip = current_clip();
    const bool clipped = !clip_stack_.empty();
    if (rit != state_.rects.end() && rit->id == id.value) {
        rit->clip = clip;
        rit->clipped = clipped;
    }
}

Rect Context::current_clip() const {
    if (clip_stack_.empty()) {
        return viewport_;
    }
    return clip_rects_[clip_stack_.back()];
}

Rect Context::resolve(const DeferredDraw& d) const {
    const Rect base = tree_.rect_of(d.node);
    return Rect{base.x + d.frac.x * base.w + d.pad.x, base.y + d.frac.y * base.h + d.pad.y,
                d.frac.z * base.w + d.pad.z, d.frac.w * base.h + d.pad.w};
}

bool Context::interact(WidgetId id, bool& out_hovered, bool& out_held) {
    out_hovered = false;
    out_held = false;
    if (!id.valid()) {
        return false;
    }

    const UiState::RectEntry* prev = previous_rect(id);
    const bool inside = prev != nullptr && prev->rect.contains(input_.mouse) &&
                        (!prev->clipped || prev->clip.contains(input_.mouse));
    if (inside) {
        state_.hot = id;
        capture_.mouse = true;
        out_hovered = true;
    }

    bool fired = false;
    if (state_.active == id) {
        out_held = true;
        capture_.mouse = true;
        if (input_.mouse_released) {
            // Fire only when the release lands inside the widget — the
            // press-and-drag-away escape every desktop UI has.
            fired = inside;
            state_.active = {};
        }
    } else if (inside && input_.mouse_pressed) {
        state_.active = id;
        state_.focused = id;
        out_held = true;
    }

    // Focused widgets activate from the keyboard/pad, and that path must NOT
    // require the pointer to be anywhere.
    if (state_.focused == id && input_.nav_activate) {
        fired = true;
    }
    return fired;
}

// ============================================================================
// Containers
// ============================================================================

void Context::begin_column(const BoxStyle& style) {
    tree_.begin_box(Axis::Vertical, style);
}

void Context::begin_row(const BoxStyle& style) {
    tree_.begin_box(Axis::Horizontal, style);
}

void Context::begin_grid(uint32_t columns, const BoxStyle& style) {
    tree_.begin_grid(columns, style);
}

void Context::end_box() {
    tree_.end_box();
}

void Context::begin_panel(const BoxStyle& style, Axis axis) {
    const uint32_t node = tree_.begin_box(axis, style);
    if (node == LayoutTree::k_invalid) {
        return;
    }
    // Recorded BEFORE the children are declared, so submission order puts it
    // behind them. That only became true once the UI pass stopped drawing all
    // geometry before all text.
    DeferredDraw bg{};
    bg.kind = DeferredDraw::Kind::FilledRect;
    bg.node = node;
    bg.color = style_.panel_bg;
    draws_.push_back(bg);
    // A FULLY TRANSPARENT border is not drawn at all, rather than drawn with
    // alpha 0. Four invisible quads still cost four quads of vertex batch, and
    // a host porting an existing HUD onto this needs "panel with no border" to
    // be free — otherwise the ported frame is not the frame it replaced.
    if ((style_.panel_border.argb >> 24) != 0u) {
        DeferredDraw border{};
        border.kind = DeferredDraw::Kind::RectOutline;
        border.node = node;
        border.color = style_.panel_border;
        border.scale = style_.border_width * cfg_.ui_scale;
        draws_.push_back(border);
    }
}

void Context::begin_panel_sprite(UiTextureId texture, const NineSlice& slice, const BoxStyle& style,
                                 core::Color tint, Axis axis) {
    const uint32_t node = tree_.begin_box(axis, style);
    if (node == LayoutTree::k_invalid) {
        return;
    }
    DeferredDraw bg{};
    bg.kind = DeferredDraw::Kind::NineSlicePanel;
    bg.node = node;
    bg.texture = texture;
    bg.slice = slice;
    bg.color = tint;
    draws_.push_back(bg);
}

void Context::end_panel() {
    tree_.end_box();
}

void Context::begin_scroll(WidgetId id, const BoxStyle& style, Axis axis) {
    const WidgetId sid = scoped(id);
    const uint32_t node = tree_.begin_box(axis, style);
    if (node == LayoutTree::k_invalid) {
        // Depth overflow already counted; push an id scope anyway so the
        // matching end_scroll stays balanced.
        id_stack_.push_back(sid);
        scroll_stack_.push_back(0xFFFFFFFFu);
        return;
    }
    tree_.set_scroll_free(node, axis);

    // Seed the slot with LAST frame's viewport, intersected with the clip
    // that is active here. emit() overwrites it with this frame's solved rect
    // — but declaration-time hit testing reads it through current_clip(), and
    // at declaration time this frame's rect does not exist yet. Without the
    // seed the clip reads as an empty rectangle and nothing inside a scroll
    // view can ever be clicked.
    Rect seeded = current_clip();
    const auto prev = std::lower_bound(state_.scroll.begin(), state_.scroll.end(), sid.value,
                                       [](const UiState::ScrollEntry& e, uint64_t k) { return e.id < k; });
    if (prev != state_.scroll.end() && prev->id == sid.value) {
        seeded = prev->viewport.intersect(seeded);
    } else {
        seeded = Rect{}; // never declared before: nothing was visible last frame
    }

    const auto clip_slot = static_cast<uint32_t>(clip_rects_.size());
    clip_rects_.push_back(seeded); // overwritten at emit from the solved node rect
    clip_stack_.push_back(clip_slot);

    DeferredDraw push{};
    push.kind = DeferredDraw::Kind::ClipPush;
    push.node = node;
    push.clip_slot = clip_slot;
    draws_.push_back(push);

    ScrollNode sn{};
    sn.id = sid;
    sn.node = node;
    sn.clip_slot = clip_slot;
    sn.axis = axis;
    scroll_stack_.push_back(static_cast<uint32_t>(scroll_nodes_.size()));
    scroll_nodes_.push_back(sn);
    id_stack_.push_back(sid);
}

void Context::end_scroll() {
    if (scroll_stack_.empty()) {
        ++diag_.clip_stack_underflow;
        return;
    }
    const uint32_t sn_index = scroll_stack_.back();
    scroll_stack_.pop_back();
    if (!id_stack_.empty()) {
        id_stack_.pop_back();
    }
    if (sn_index == 0xFFFFFFFFu) {
        return; // the depth-overflow case; nothing was opened
    }
    if (!clip_stack_.empty()) {
        clip_stack_.pop_back();
    }
    tree_.end_box();

    DeferredDraw pop{};
    pop.kind = DeferredDraw::Kind::ClipPop;
    // Restore the ENCLOSING clip, not "no clip": a scroll view nested in
    // another must stay inside its parent's window when the inner one ends.
    pop.clip_slot = clip_stack_.empty() ? 0xFFFFFFFFu : clip_stack_.back();
    draws_.push_back(pop);
}

void Context::push_id(WidgetId scope) {
    id_stack_.push_back(scoped(scope));
}

void Context::pop_id() {
    if (!id_stack_.empty()) {
        id_stack_.pop_back();
    }
}

// ============================================================================
// Scrolling + focus
// ============================================================================

void Context::resolve_scrolling() {
    for (uint32_t i = 0; i < scroll_nodes_.size(); ++i) {
        ScrollNode& sn = scroll_nodes_[i];
        if (sn.node == LayoutTree::k_invalid) {
            continue; // depth overflow swallowed this container
        }
        sn.viewport = tree_.rect_of(sn.node);

        // Look the retained offset up (sorted vector — deterministic).
        const auto it = std::lower_bound(state_.scroll.begin(), state_.scroll.end(), sn.id.value,
                                         [](const UiState::ScrollEntry& e, uint64_t k) { return e.id < k; });
        UiState::ScrollEntry* entry = nullptr;
        if (it != state_.scroll.end() && it->id == sn.id.value) {
            entry = &*it;
        } else {
            entry = &*state_.scroll.insert(it, UiState::ScrollEntry{sn.id.value, {0.0f, 0.0f}, Rect{}, state_.frame});
        }
        entry->last_frame = state_.frame;
        entry->viewport = sn.viewport; // next frame's declaration-time clip

        // content_extent_of, not content_of: a scroll viewport is Fixed by
        // definition, so content_of reports the viewport's own size and the
        // max offset would always come out zero.
        const glm::vec2 content = tree_.content_extent_of(sn.node);
        const float extent = sn.axis == Axis::Horizontal ? content.x : content.y;
        const float window = sn.axis == Axis::Horizontal ? sn.viewport.w : sn.viewport.h;
        const float max_offset = std::max(0.0f, extent - window);

        float& off = sn.axis == Axis::Horizontal ? entry->offset.x : entry->offset.y;
        if (sn.viewport.contains(input_.mouse) && input_.wheel != 0.0f && max_offset > 0.0f) {
            // One notch = one line of text, the convention every list uses.
            off -= input_.wheel * line_height(0.0f) * 3.0f;
            capture_.wheel = true;
            capture_.mouse = true;
        }

        // Scroll the focused widget into view. Done BEFORE this container's
        // translation so the comparison is against its own untranslated
        // children — and note the ANCESTOR test rather than an exact
        // `fe.scroll_index == i`: a widget inside a nested scroll view has to
        // be brought into view by every container that encloses it, not only
        // the innermost, or tabbing into a nested list scrolls the inner
        // window while the outer one still has it off screen. Outer containers
        // are declared first and are therefore processed first, so by the time
        // an inner one runs its viewport already reflects the outer scroll;
        // the offsets it computes are relative and translation-invariant.
        if (state_.focused.valid()) {
            for (const FocusEntry& fe : focus_order_) {
                if (fe.id != state_.focused || fe.node == LayoutTree::k_invalid ||
                    !tree_.is_descendant_of(fe.node, sn.node)) {
                    continue;
                }
                const Rect r = tree_.rect_of(fe.node);
                const float lo = sn.axis == Axis::Horizontal ? r.x - sn.viewport.x : r.y - sn.viewport.y;
                const float hi = sn.axis == Axis::Horizontal ? lo + r.w : lo + r.h;
                if (lo < off) {
                    off = lo;
                } else if (hi > off + window) {
                    off = hi - window;
                }
                break;
            }
        }

        off = std::clamp(off, 0.0f, max_offset);
        const glm::vec2 delta = sn.axis == Axis::Horizontal ? glm::vec2{-off, 0.0f} : glm::vec2{0.0f, -off};
        tree_.translate_children(sn.node, delta);
    }
}

void Context::resolve_focus() {
    if (focus_order_.empty()) {
        state_.focused = {};
        return;
    }

    // Drop a focus that no longer names a declared widget rather than keeping
    // a ghost that nothing can move off.
    auto index_of = [this](WidgetId id) -> std::size_t {
        for (std::size_t i = 0; i < focus_order_.size(); ++i) {
            if (focus_order_[i].id == id) {
                return i;
            }
        }
        return focus_order_.size();
    };
    std::size_t cur = index_of(state_.focused);
    const bool had_focus = cur < focus_order_.size();

    // A focused slider ate Left/Right to tune its value; moving the focus on
    // the same press would make the control impossible to use.
    const bool nav_left = input_.nav_left && !nav_horizontal_consumed_;
    const bool nav_right = input_.nav_right && !nav_horizontal_consumed_;

    const bool wants_move =
        input_.nav_next || input_.nav_prev || input_.nav_up || input_.nav_down || nav_left || nav_right;
    if (!had_focus) {
        if (wants_move) {
            state_.focused = focus_order_.front().id;
        }
        return;
    }

    if (input_.nav_next || input_.nav_prev) {
        const std::size_t n = focus_order_.size();
        if (input_.nav_next) {
            if (cur + 1 < n) {
                cur += 1;
            } else if (cfg_.wrap_focus) {
                cur = 0;
            }
        } else {
            if (cur > 0) {
                cur -= 1;
            } else if (cfg_.wrap_focus) {
                cur = n - 1;
            }
        }
        state_.focused = focus_order_[cur].id;
        return;
    }

    if (!(input_.nav_up || input_.nav_down || nav_left || nav_right)) {
        return;
    }

    // Directional move over the SOLVED rectangles: among the candidates whose
    // centre lies in the requested direction, take the nearest by a distance
    // that weights the perpendicular offset, so "down" from a wide row prefers
    // the row beneath it rather than a far column that happens to be lower.
    const glm::vec2 from = tree_.rect_of(focus_order_[cur].node).center();
    const bool vertical = input_.nav_up || input_.nav_down;
    const float sign = (input_.nav_down || nav_right) ? 1.0f : -1.0f;

    std::size_t best = focus_order_.size();
    float best_cost = 0.0f;
    for (std::size_t i = 0; i < focus_order_.size(); ++i) {
        if (i == cur) {
            continue;
        }
        const glm::vec2 to = tree_.rect_of(focus_order_[i].node).center();
        const float along = vertical ? (to.y - from.y) * sign : (to.x - from.x) * sign;
        if (along <= 0.0f) {
            continue;
        }
        const float across = vertical ? std::abs(to.x - from.x) : std::abs(to.y - from.y);
        const float cost = along + across * 2.0f;
        if (best == focus_order_.size() || cost < best_cost) {
            best = i;
            best_cost = cost;
        }
    }

    if (best < focus_order_.size()) {
        state_.focused = focus_order_[best].id;
        return;
    }
    if (cfg_.wrap_focus) {
        // Nothing in that direction: wrap to the extreme on the far side, which
        // is the behaviour a pad user expects at the end of a list.
        std::size_t extreme = cur;
        float extreme_v = 0.0f;
        for (std::size_t i = 0; i < focus_order_.size(); ++i) {
            const glm::vec2 to = tree_.rect_of(focus_order_[i].node).center();
            const float v = (vertical ? to.y : to.x) * sign;
            if (i == 0 || v < extreme_v) {
                extreme = i;
                extreme_v = v;
            }
        }
        state_.focused = focus_order_[extreme].id;
    }
}

// ============================================================================
// Emission
// ============================================================================

void Context::emit() {
    if (!lane_.has_value()) {
        return;
    }
    scene::FrameBuilder::UiLane lane = *lane_;

    // Resolve every scroll clip rect now that the nodes are solved, and
    // intersect each with the clip that was active where it was pushed.
    for (const ScrollNode& sn : scroll_nodes_) {
        if (sn.node != LayoutTree::k_invalid) {
            clip_rects_[sn.clip_slot] = tree_.rect_of(sn.node);
        }
    }
    // Nested clips: a slot's rect must already be intersected with its
    // enclosing slot's. Slots are created in declaration order, so a single
    // forward pass over the scroll nodes in order is enough.
    for (const ScrollNode& sn : scroll_nodes_) {
        uint32_t parent_slot = LayoutTree::k_invalid;
        for (const ScrollNode& outer : scroll_nodes_) {
            if (outer.node == sn.node) {
                continue;
            }
            if (tree_.is_descendant_of(sn.node, outer.node)) {
                // Deepest enclosing scroll wins; declaration order means the
                // last match encountered is the innermost.
                parent_slot = outer.clip_slot;
            }
        }
        if (parent_slot != LayoutTree::k_invalid) {
            clip_rects_[sn.clip_slot] = clip_rects_[sn.clip_slot].intersect(clip_rects_[parent_slot]);
        }
    }

    for (const DeferredDraw& d : draws_) {
        switch (d.kind) {
        case DeferredDraw::Kind::ClipPush: {
            const Rect c = clip_rects_[d.clip_slot];
            lane.set_clip(c.x, c.y, c.w, c.h);
            ++emitted_;
            break;
        }
        case DeferredDraw::Kind::ClipPop: {
            if (d.clip_slot == 0xFFFFFFFFu) {
                lane.clear_clip();
            } else {
                const Rect c = clip_rects_[d.clip_slot];
                lane.set_clip(c.x, c.y, c.w, c.h);
            }
            ++emitted_;
            break;
        }
        case DeferredDraw::Kind::FilledRect: {
            const Rect r = resolve(d);
            if (r.empty()) {
                break;
            }
            lane.add_filled_rect(r.x, r.y, r.w, r.h, d.color);
            ++emitted_;
            break;
        }
        case DeferredDraw::Kind::RectOutline: {
            const Rect r = resolve(d);
            if (r.empty()) {
                break;
            }
            lane.add_rect_outline(r.x, r.y, r.w, r.h, d.color, d.scale);
            ++emitted_;
            break;
        }
        case DeferredDraw::Kind::Text: {
            const Rect r = resolve(d);
            lane.add_text(text_at(d.text_offset, d.text_length), r.x, r.y, d.scale, d.color);
            ++emitted_;
            break;
        }
        case DeferredDraw::Kind::TextCentered: {
            const Rect r = resolve(d);
            const std::string_view t = text_at(d.text_offset, d.text_length);
            float tw = 0.0f;
            float th = 0.0f;
            if (cfg_.font != nullptr && cfg_.font->valid()) {
                tw = cfg_.font->measure_width(t, d.scale);
                th = cfg_.font->measure_height(d.scale);
            }
            lane.add_text(t, r.x + (r.w - tw) * 0.5f, r.y + (r.h - th) * 0.5f, d.scale, d.color);
            ++emitted_;
            break;
        }
        case DeferredDraw::Kind::WrappedText: {
            const Rect r = resolve(d);
            const float width = d.pad.z > 0.0f ? d.pad.z : r.w;
            (void)lane.add_text_wrapped(text_at(d.text_offset, d.text_length), r.x, r.y, width, d.scale, d.color);
            ++emitted_;
            break;
        }
        case DeferredDraw::Kind::TexturedQuad: {
            const Rect r = resolve(d);
            if (r.empty()) {
                break;
            }
            lane.add_textured_quad(d.texture, r.x, r.y, r.w, r.h, d.color, d.uv.x, d.uv.y, d.uv.z, d.uv.w);
            ++emitted_;
            break;
        }
        case DeferredDraw::Kind::NineSlicePanel: {
            const Rect r = resolve(d);
            NineSliceQuad quads[9]{};
            const uint32_t n = build_nine_slice(d.slice, r, std::span<NineSliceQuad, 9>(quads));
            for (uint32_t i = 0; i < n; ++i) {
                const NineSliceQuad& q = quads[i];
                if (q.dst.empty()) {
                    continue;
                }
                lane.add_textured_quad(d.texture, q.dst.x, q.dst.y, q.dst.w, q.dst.h, d.color, q.u0, q.v0, q.u1,
                                       q.v1);
                ++emitted_;
            }
            break;
        }
        }
    }
}

Rect Context::rect_of(WidgetId id) const {
    for (const auto& [raw, node] : frame_rects_) {
        if (raw == id.value) {
            return tree_.rect_of(node);
        }
    }
    return {};
}

std::vector<WidgetId> Context::focus_order() const {
    std::vector<WidgetId> out;
    out.reserve(focus_order_.size());
    for (const FocusEntry& e : focus_order_) {
        out.push_back(e.id);
    }
    return out;
}

void Context::set_focus(WidgetId id) {
    state_.focused = scoped(id);
}

} // namespace bracken::ui
