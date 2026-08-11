// The one thing in `bracken::ui` that could plausibly cost sim time.
//
// Everything else the module does per frame is a push_back; the solve is two
// recursive passes over the whole tree and is the only part whose cost scales
// with how much UI a game declares. A number from here answers "can a HUD
// afford to be laid out every frame" — which is the design's central bet.
//
// Numbers from a default configure are meaningless (-O0 with live asserts);
// reconfigure with -DCMAKE_BUILD_TYPE=RelWithDebInfo before quoting one.

#include "bracken/ui/layout.hpp"

#include <benchmark/benchmark.h>

#include <cstdint>

using namespace bracken::ui;

namespace {

/// A tree shaped like a real screen: `rows` stat rows in a padded panel, each
/// row a horizontal box holding a label leaf and a flex-width value leaf.
/// Node count is roughly 3 * rows + 2.
void build_hud(LayoutTree& tree, uint32_t rows) {
    BoxStyle panel{};
    panel.padding = Edges::all(12.0f);
    panel.spacing = 6.0f;
    panel.width = Size::px(320.0f);
    tree.begin_box(Axis::Vertical, panel);

    BoxStyle row{};
    row.spacing = 8.0f;
    for (uint32_t i = 0; i < rows; ++i) {
        tree.begin_box(Axis::Horizontal, row);
        tree.leaf(Size::px(120.0f), Size::px(20.0f), {120.0f, 20.0f});
        tree.leaf(Size::flex(1.0f), Size::content(), {60.0f, 20.0f});
        tree.end_box();
    }
    tree.end_box();
}

/// A deep chain, to separate the recursion's cost from the per-node cost. The
/// solver recurses once per level in measure and once in arrange, so this is
/// the shape that would expose a bad constant factor in the descent.
void build_deep(LayoutTree& tree, uint32_t depth) {
    BoxStyle s{};
    s.padding = Edges::all(2.0f);
    for (uint32_t i = 0; i < depth; ++i) {
        tree.begin_box(Axis::Vertical, s);
    }
    tree.leaf(Size::px(10.0f), Size::px(10.0f), {10.0f, 10.0f});
    for (uint32_t i = 0; i < depth; ++i) {
        tree.end_box();
    }
}

constexpr Rect k_viewport{0.0f, 0.0f, 1600.0f, 900.0f};

// DECLARE + SOLVE, which is what a frame actually pays: the tree is rebuilt
// from scratch every frame (that is what "immediate mode" means here), so
// benchmarking solve() alone would flatter the design.
void BM_UiLayoutDeclareAndSolve(benchmark::State& state) {
    const auto rows = static_cast<uint32_t>(state.range(0));
    LayoutTree tree; // reused, so the node vector's capacity warms up — as it
                     // does in a host holding one Context for the process's life
    for (auto _ : state) {
        tree.clear();
        build_hud(tree, rows);
        tree.solve(k_viewport);
        benchmark::DoNotOptimize(tree.rect_of(tree.node_count() - 1));
    }
    state.SetItemsProcessed(state.iterations() * tree.node_count());
}
BENCHMARK(BM_UiLayoutDeclareAndSolve)->Arg(8)->Arg(64)->Arg(166); // ~500 nodes at the top end

// SOLVE ALONE on an already-built tree, for the same shapes. The gap between
// this and the pair above is the declaration cost.
void BM_UiLayoutSolveOnly(benchmark::State& state) {
    const auto rows = static_cast<uint32_t>(state.range(0));
    LayoutTree tree;
    build_hud(tree, rows);
    for (auto _ : state) {
        tree.solve(k_viewport);
        benchmark::DoNotOptimize(tree.rect_of(tree.node_count() - 1));
    }
    state.SetItemsProcessed(state.iterations() * tree.node_count());
}
BENCHMARK(BM_UiLayoutSolveOnly)->Arg(8)->Arg(64)->Arg(166);

// The recursion itself, at the k_max_depth guard's limit.
void BM_UiLayoutDeepChain(benchmark::State& state) {
    const auto depth = static_cast<uint32_t>(state.range(0));
    LayoutTree tree;
    for (auto _ : state) {
        tree.clear();
        build_deep(tree, depth);
        tree.solve(k_viewport);
        benchmark::DoNotOptimize(tree.rect_of(tree.node_count() - 1));
    }
}
BENCHMARK(BM_UiLayoutDeepChain)->Arg(8)->Arg(63); // 63: one under k_max_depth, so nothing is discarded

} // namespace
