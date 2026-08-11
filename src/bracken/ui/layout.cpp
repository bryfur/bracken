#include "bracken/ui/layout.hpp"

#include "bracken/core/assert.hpp"

#include <algorithm>

namespace bracken::ui {

namespace {

/// Main/cross accessors so the solver is written once for both axes rather
/// than twice with the x and y swapped (which is how axis bugs happen).
[[nodiscard]] float& axis_ref(glm::vec2& v, Axis a) noexcept {
    return a == Axis::Horizontal ? v.x : v.y;
}
[[nodiscard]] float axis_get(glm::vec2 v, Axis a) noexcept {
    return a == Axis::Horizontal ? v.x : v.y;
}
[[nodiscard]] constexpr Axis other(Axis a) noexcept {
    return a == Axis::Horizontal ? Axis::Vertical : Axis::Horizontal;
}

} // namespace

LayoutTree::LayoutTree() {
    clear();
}

void LayoutTree::clear() {
    nodes_.clear();
    stack_.clear();
    unbalanced_end_box_ = 0;
    depth_overflow_ = 0;
    suppressed_depth_ = 0;

    Node root{};
    root.axis = Axis::Vertical;
    root.style.width = Size::flex(1.0f);
    root.style.height = Size::flex(1.0f);
    nodes_.push_back(root);
    stack_.push_back(0);
}

uint32_t LayoutTree::add_child(Node&& node) {
    BRACKEN_ASSERT(!stack_.empty(), "LayoutTree: no open container");
    const uint32_t parent = stack_.back();
    const uint32_t index = static_cast<uint32_t>(nodes_.size());
    node.parent = parent;
    nodes_.push_back(std::move(node));
    Node& p = nodes_[parent];
    if (p.first_child == k_invalid) {
        p.first_child = index;
    } else {
        nodes_[p.last_child].next_sibling = index;
    }
    p.last_child = index;
    ++p.child_count;
    return index;
}

uint32_t LayoutTree::begin_box(Axis axis, const BoxStyle& style) {
    if (stack_.size() >= k_max_depth) {
        // Discard, but remember that an end_box is owed — otherwise the very
        // next end_box would close a container the caller still thinks is
        // open, and every rectangle after it would be wrong.
        ++depth_overflow_;
        ++suppressed_depth_;
        return k_invalid;
    }
    Node n{};
    n.axis = axis;
    n.style = style;
    const uint32_t index = add_child(std::move(n));
    stack_.push_back(index);
    return index;
}

uint32_t LayoutTree::begin_grid(uint32_t columns, const BoxStyle& style) {
    const uint32_t index = begin_box(Axis::Vertical, style);
    if (index != k_invalid) {
        nodes_[index].columns = std::max(columns, 1u);
    }
    return index;
}

void LayoutTree::end_box() {
    if (suppressed_depth_ > 0) {
        --suppressed_depth_;
        return;
    }
    if (stack_.size() <= 1) {
        // Closing the root is the "one end_box too many" mistake. Counted, not
        // asserted: a UI built from data can get this wrong at runtime and the
        // useful outcome is a slightly wrong frame plus a number, not an abort.
        ++unbalanced_end_box_;
        return;
    }
    stack_.pop_back();
}

uint32_t LayoutTree::leaf(const Size& width, const Size& height, glm::vec2 content) {
    Node n{};
    n.is_leaf = true;
    n.style.width = width;
    n.style.height = height;
    n.content = content;
    n.measured = content;
    return add_child(std::move(n));
}

void LayoutTree::set_scroll_free(uint32_t node, Axis axis) {
    if (node >= nodes_.size()) {
        return;
    }
    Node& n = nodes_[node];
    if (axis == n.axis) {
        n.scroll_free_main = true;
    } else {
        n.scroll_free_cross = true;
    }
}

Rect LayoutTree::rect_of(uint32_t node) const {
    return node < nodes_.size() ? nodes_[node].rect : Rect{};
}

glm::vec2 LayoutTree::content_of(uint32_t node) const {
    return node < nodes_.size() ? nodes_[node].content : glm::vec2{0.0f, 0.0f};
}

glm::vec2 LayoutTree::content_extent_of(uint32_t node) const {
    return node < nodes_.size() ? nodes_[node].measured : glm::vec2{0.0f, 0.0f};
}

bool LayoutTree::is_descendant_of(uint32_t node, uint32_t ancestor) const {
    while (node != k_invalid && node < nodes_.size()) {
        if (node == ancestor) {
            return true;
        }
        node = nodes_[node].parent;
    }
    return false;
}

void LayoutTree::measure(uint32_t index) {
    Node& n = nodes_[index];
    if (n.is_leaf) {
        // `measured` is the requested content, set once by leaf() and never
        // written again — which is what makes solve() idempotent.
        glm::vec2 want = n.measured;
        if (n.style.width.mode == SizeMode::Fixed) {
            want.x = n.style.width.value;
        }
        if (n.style.height.mode == SizeMode::Fixed) {
            want.y = n.style.height.value;
        }
        n.content = want;
        return;
    }

    for (uint32_t c = n.first_child; c != k_invalid; c = nodes_[c].next_sibling) {
        measure(c);
    }

    glm::vec2 want{0.0f, 0.0f};
    if (n.columns > 0) {
        // Grid: uniform cells sized by the largest child, row-major.
        glm::vec2 cell{0.0f, 0.0f};
        for (uint32_t c = n.first_child; c != k_invalid; c = nodes_[c].next_sibling) {
            cell.x = std::max(cell.x, nodes_[c].content.x);
            cell.y = std::max(cell.y, nodes_[c].content.y);
        }
        const uint32_t cols = n.columns;
        const uint32_t rows = n.child_count == 0 ? 0 : (n.child_count + cols - 1) / cols;
        const uint32_t used_cols = std::min(cols, std::max(n.child_count, 1u));
        want.x = cell.x * static_cast<float>(used_cols) +
                 n.style.spacing * static_cast<float>(used_cols > 0 ? used_cols - 1 : 0);
        want.y = cell.y * static_cast<float>(rows) + n.style.spacing * static_cast<float>(rows > 0 ? rows - 1 : 0);
    } else {
        const Axis main = n.axis;
        const Axis cross = other(main);
        float main_total = 0.0f;
        float cross_max = 0.0f;
        for (uint32_t c = n.first_child; c != k_invalid; c = nodes_[c].next_sibling) {
            main_total += axis_get(nodes_[c].content, main);
            cross_max = std::max(cross_max, axis_get(nodes_[c].content, cross));
        }
        if (n.child_count > 1) {
            main_total += n.style.spacing * static_cast<float>(n.child_count - 1);
        }
        axis_ref(want, main) = main_total;
        axis_ref(want, cross) = cross_max;
    }

    want.x += n.style.padding.horizontal();
    want.y += n.style.padding.vertical();
    n.measured = want; // what the CHILDREN wanted, before any Fixed override

    // An explicit Fixed size overrides the measurement outright; a Flex size
    // does not — it still reports its content as its MINIMUM, which is what
    // lets a flex child inside a content-sized parent still push the parent
    // open.
    if (n.style.width.mode == SizeMode::Fixed) {
        want.x = n.style.width.value;
    }
    if (n.style.height.mode == SizeMode::Fixed) {
        want.y = n.style.height.value;
    }
    n.content = want;
}

void LayoutTree::arrange(uint32_t index) {
    Node& n = nodes_[index];
    if (n.is_leaf || n.child_count == 0) {
        return;
    }

    const Rect inner{n.rect.x + n.style.padding.l, n.rect.y + n.style.padding.t,
                     n.rect.w - n.style.padding.horizontal(), n.rect.h - n.style.padding.vertical()};

    if (n.columns > 0) {
        const uint32_t cols = n.columns;
        const uint32_t rows = (n.child_count + cols - 1) / cols;
        const float span_x = n.style.spacing * static_cast<float>(cols - 1);
        const float cell_w = cols > 0 ? (inner.w - span_x) / static_cast<float>(cols) : 0.0f;
        float cell_h = 0.0f;
        for (uint32_t c = n.first_child; c != k_invalid; c = nodes_[c].next_sibling) {
            cell_h = std::max(cell_h, nodes_[c].content.y);
        }
        if (n.style.cross == Align::Stretch && rows > 0) {
            const float span_y = n.style.spacing * static_cast<float>(rows - 1);
            cell_h = std::max(cell_h, (inner.h - span_y) / static_cast<float>(rows));
        }
        uint32_t i = 0;
        for (uint32_t c = n.first_child; c != k_invalid; c = nodes_[c].next_sibling, ++i) {
            const uint32_t col = i % cols;
            const uint32_t row = i / cols;
            nodes_[c].rect = Rect{inner.x + static_cast<float>(col) * (cell_w + n.style.spacing),
                                  inner.y + static_cast<float>(row) * (cell_h + n.style.spacing), cell_w, cell_h};
            arrange(c);
        }
        return;
    }

    const Axis main = n.axis;
    const Axis cross = other(main);
    const glm::vec2 inner_size{inner.w, inner.h};
    float avail_main = axis_get(inner_size, main);
    const float avail_cross = axis_get(inner_size, cross);
    if (n.scroll_free_main) {
        // A scroll viewport lays its children out at FULL length and shows a
        // window onto them; the offset is applied afterwards by
        // translate_children. `measured`, not `content`: a scroll viewport has
        // an explicit Fixed size by definition, and `content` is that size.
        const float wanted = axis_get(n.measured, main) -
                             (main == Axis::Horizontal ? n.style.padding.horizontal() : n.style.padding.vertical());
        avail_main = std::max(avail_main, wanted);
    }

    const float spacing_total = n.style.spacing * static_cast<float>(n.child_count - 1);

    float fixed_total = 0.0f;
    float flex_total = 0.0f;
    for (uint32_t c = n.first_child; c != k_invalid; c = nodes_[c].next_sibling) {
        const Size& s = main == Axis::Horizontal ? nodes_[c].style.width : nodes_[c].style.height;
        if (s.mode == SizeMode::Flex) {
            flex_total += std::max(s.value, 0.0f);
        } else {
            fixed_total += axis_get(nodes_[c].content, main);
        }
    }
    const float leftover = std::max(0.0f, avail_main - fixed_total - spacing_total);

    const float main_origin = axis_get(glm::vec2{inner.x, inner.y}, main);
    float align_shift = 0.0f;
    if (flex_total <= 0.0f) {
        // No flex child to eat the slack, so the main alignment places it.
        // Stretch has no meaning on the main axis; it reads as Start.
        switch (n.style.main) {
        case Align::Center:
            align_shift = leftover * 0.5f;
            break;
        case Align::End:
            align_shift = leftover;
            break;
        case Align::Start:
        case Align::Stretch:
            break;
        }
    }

    // Flex distribution WITHOUT DRIFT, in two parts.
    //
    // (1) Sizes come from the DIFFERENCE OF TWO CUMULATIVE fractions rather
    //     than from `leftover * w / total` per child. The last child's
    //     cumulative weight is exactly `flex_total`, so the last cumulative
    //     size is exactly `leftover`.
    // (2) Positions are computed as an OFFSET FROM THE CONTENT ORIGIN
    //     (fixed-so-far + spacing * index + flex-so-far) rather than by
    //     accumulating a running cursor. A running cursor rounds once per
    //     child, so its error grows with the child count and a twelve-item row
    //     ends visibly short of its container; an offset from the origin does
    //     not accumulate at all, and for the common all-flex row the final
    //     right edge lands on the content edge EXACTLY (the last size is
    //     `leftover - placed`, which Sterbenz makes an exact subtraction).
    float cum_weight = 0.0f;
    float placed_flex = 0.0f;
    float placed_fixed = 0.0f;
    uint32_t child_index = 0;
    for (uint32_t c = n.first_child; c != k_invalid; c = nodes_[c].next_sibling, ++child_index) {
        Node& child = nodes_[c];
        const Size& s_main = main == Axis::Horizontal ? child.style.width : child.style.height;
        const Size& s_cross = cross == Axis::Horizontal ? child.style.width : child.style.height;

        const float cursor =
            main_origin + align_shift + placed_fixed + placed_flex + n.style.spacing * static_cast<float>(child_index);

        float main_size = 0.0f;
        if (s_main.mode == SizeMode::Flex && flex_total > 0.0f) {
            cum_weight += std::max(s_main.value, 0.0f);
            const float cum_size = leftover * (cum_weight / flex_total);
            main_size = cum_size - placed_flex;
            placed_flex = cum_size;
        } else {
            main_size = axis_get(child.content, main);
            placed_fixed += main_size;
        }

        float cross_size = 0.0f;
        if (s_cross.mode == SizeMode::Fixed) {
            cross_size = s_cross.value;
        } else if (s_cross.mode == SizeMode::Flex || n.style.cross == Align::Stretch) {
            cross_size = avail_cross;
        } else {
            cross_size = axis_get(child.content, cross);
        }

        float cross_pos = axis_get(glm::vec2{inner.x, inner.y}, cross);
        switch (n.style.cross) {
        case Align::Center:
            cross_pos += (avail_cross - cross_size) * 0.5f;
            break;
        case Align::End:
            cross_pos += avail_cross - cross_size;
            break;
        case Align::Start:
        case Align::Stretch:
            break;
        }

        glm::vec2 pos{0.0f, 0.0f};
        glm::vec2 size{0.0f, 0.0f};
        axis_ref(pos, main) = cursor;
        axis_ref(pos, cross) = cross_pos;
        axis_ref(size, main) = main_size;
        axis_ref(size, cross) = cross_size;
        child.rect = Rect{pos.x, pos.y, size.x, size.y};
        arrange(c);
    }
}

void LayoutTree::solve(const Rect& viewport) {
    if (nodes_.empty()) {
        return;
    }
    measure(0);
    nodes_[0].rect = viewport;
    arrange(0);
}

void LayoutTree::translate_subtree(uint32_t index, glm::vec2 delta) {
    Node& n = nodes_[index];
    n.rect.x += delta.x;
    n.rect.y += delta.y;
    for (uint32_t c = n.first_child; c != k_invalid; c = nodes_[c].next_sibling) {
        translate_subtree(c, delta);
    }
}

void LayoutTree::translate_children(uint32_t node, glm::vec2 delta) {
    if (node >= nodes_.size() || (delta.x == 0.0f && delta.y == 0.0f)) {
        return;
    }
    for (uint32_t c = nodes_[node].first_child; c != k_invalid; c = nodes_[c].next_sibling) {
        translate_subtree(c, delta);
    }
}

} // namespace bracken::ui
