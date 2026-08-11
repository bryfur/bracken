#pragma once

#include "bracken/ui/ui_types.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

namespace bracken::ui {

/**
 * LayoutTree — the two-pass box solver, and nothing else.
 *
 * It knows about rectangles, sizes and alignment. It does NOT know about the
 * frame packet, input, widgets, fonts or the engine, which is exactly why it
 * is a separate file: `layout.test.cpp` builds trees and asserts rectangles
 * with no GPU, no window and no `FramePacket`.
 *
 * WHY A TREE AT ALL, given the module is immediate-mode. An immediate-mode
 * call cannot place a widget at the moment it is declared: its rectangle
 * depends on siblings that have not been declared yet. So declaration builds
 * this tree plus a deferred draw list, and `Context::end()` solves and THEN
 * emits. That is the "retained layout" half of the design.
 *
 * DETERMINISM. `solve()` is a pure function of the tree and the viewport:
 * same tree, same rectangles, bit for bit. It contains no map iteration, no
 * floating-point accumulation across siblings (flex sizes are computed from
 * CUMULATIVE weights, see the note in solve), and no allocation once the
 * node vector has warmed up.
 *
 * Thread-safety tier: `owned`.
 */
class LayoutTree {
public:
    static constexpr uint32_t k_invalid = 0xFFFFFFFFu;

    /**
     * Container nesting limit. A guard, not a budget: a runaway `begin_box`
     * (a container opened inside a loop with no matching `end_box`) would
     * otherwise grow the stack until the allocator gives up. Boxes opened past
     * the limit are DISCARDED and counted in `depth_overflow()`, and their
     * `end_box` is absorbed, so the tree stays well-formed instead of
     * corrupting.
     */
    static constexpr uint32_t k_max_depth = 64;

    LayoutTree();

    /** Drop every node and reopen an empty root. Keeps the node storage. */
    void clear();

    /** The implicit root box — a vertical container filling the viewport. */
    [[nodiscard]] uint32_t root() const noexcept { return 0; }

    /** Open a container. Returns its node index (k_invalid on depth overflow). */
    uint32_t begin_box(Axis axis, const BoxStyle& style);

    /**
     * Open a row-major grid of `columns` columns. Children fill left to
     * right, wrapping to a new row; the last row may be partial. `columns`
     * is clamped to at least 1.
     */
    uint32_t begin_grid(uint32_t columns, const BoxStyle& style);

    /** Close the innermost open container. An unbalanced call is counted in
     *  `unbalanced_end_box()` and otherwise ignored. */
    void end_box();

    /**
     * A measured leaf. `content` is what it WANTS — a measured string extent,
     * an image's pixel size — and is what `SizeMode::Content` resolves to.
     */
    uint32_t leaf(const Size& width, const Size& height, glm::vec2 content);

    /**
     * Let `node` size itself to its CONTENT along `axis` during arrange, even
     * when its own rectangle is smaller. This is what makes a scroll viewport
     * work: the children are laid out at full length and the viewport shows a
     * window onto them.
     */
    void set_scroll_free(uint32_t node, Axis axis);

    /** Measure (bottom-up), then arrange (top-down). */
    void solve(const Rect& viewport);

    [[nodiscard]] Rect rect_of(uint32_t node) const;

    /** The size the node ASKED for during measure, padding included and any
     *  explicit Fixed size APPLIED. This is what a parent sizes against. */
    [[nodiscard]] glm::vec2 content_of(uint32_t node) const;

    /**
     * The measured extent of the node's CHILDREN plus padding, BEFORE an
     * explicit Fixed size overrode it.
     *
     * This is the number a scroll container needs and `content_of` is not: a
     * scroll viewport is Fixed by definition, so `content_of` reports the
     * viewport's own height and a max-offset computed from it is always zero —
     * i.e. nothing ever scrolls. Keeping both is the fix.
     */
    [[nodiscard]] glm::vec2 content_extent_of(uint32_t node) const;

    /** Offset every DESCENDANT of `node` (not the node itself) — how a scroll
     *  offset is applied after solving. */
    void translate_children(uint32_t node, glm::vec2 delta);

    /** True when `ancestor` is `node` or one of its ancestors. */
    [[nodiscard]] bool is_descendant_of(uint32_t node, uint32_t ancestor) const;

    [[nodiscard]] uint32_t node_count() const noexcept { return static_cast<uint32_t>(nodes_.size()); }
    [[nodiscard]] uint32_t depth() const noexcept { return static_cast<uint32_t>(stack_.size()); }
    [[nodiscard]] uint32_t unbalanced_end_box() const noexcept { return unbalanced_end_box_; }
    [[nodiscard]] uint32_t depth_overflow() const noexcept { return depth_overflow_; }

private:
    struct Node {
        Axis axis = Axis::Vertical;
        BoxStyle style{};
        uint32_t parent = k_invalid;
        uint32_t first_child = k_invalid;
        uint32_t last_child = k_invalid;
        uint32_t next_sibling = k_invalid;
        uint32_t columns = 0; // 0 = not a grid
        uint32_t child_count = 0;
        glm::vec2 content{0.0f, 0.0f};  // leaf: requested; box: measured (incl. padding), Fixed applied
        glm::vec2 measured{0.0f, 0.0f}; // same, but BEFORE the Fixed override — see content_extent_of
        Rect rect{};
        bool is_leaf = false;
        bool scroll_free_main = false;
        bool scroll_free_cross = false;
    };

    uint32_t add_child(Node&& node);
    void measure(uint32_t node);
    void arrange(uint32_t node);
    void translate_subtree(uint32_t node, glm::vec2 delta);

    std::vector<Node> nodes_;
    std::vector<uint32_t> stack_;
    uint32_t unbalanced_end_box_ = 0;
    uint32_t depth_overflow_ = 0;
    /// Boxes opened past k_max_depth whose end_box must still be absorbed.
    uint32_t suppressed_depth_ = 0;
};

} // namespace bracken::ui
