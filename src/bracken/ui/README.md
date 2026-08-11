# `bracken/ui/` — the sim-stage UI module

An immediate-mode **declaration** surface over a retained **layout** and
retained **interaction**, running entirely inside `on_simulate`. It emits into
the frame packet's `UiLane` and touches nothing else the engine owns.

## The rule, enforced by tests

> Nothing under `src/bracken/ui/` may include `bracken/gpu/`,
> `bracken/render/`, `SDL3/SDL_gpu.h`, `SDL3/SDL_render.h` or SDL_ttf — and
> nothing under `src/bracken/render/` may include `bracken/ui/`.

Both halves are pinned by `src/bracken/ui/ui_purity.test.cpp`. The first is the
threading claim: the UI is described from a pool worker, so it must be unable
to *name* a device resource, not merely careful never to touch one. The second
is the layering claim: `scene/` already includes `render/` and `ui/` includes
`scene/`, so a `render/` → `ui/` edge would close
`render → ui → scene → render` and fail
`src/bracken/core/module_layering.test.cpp` somewhere that does not name the
file that caused it.

`ui` is also listed in that test's upper-layer set, so `core/`, `memory/` and
`gpu/` cannot reach it either.

## Why it is not pure immediate mode

An immediate-mode call cannot place a widget at the moment it is declared: the
rectangle depends on siblings that have not been declared yet. So a declaration
appends to a `LayoutTree` plus a deferred draw list, and `Context::end()`
solves, resolves focus and scrolling, and only then emits.

The consequence, and the one thing to understand before using the API:
**hit testing uses LAST frame's rectangle.** `button()` has to return its result
at the call site, which is before the solve. A widget therefore cannot be
clicked on the first frame it exists, and clicking one that moved between frames
uses where it was. The alternative — returning results from `end()` — makes the
call site unable to branch on a click, which is the entire ergonomic point.

## The files

| File | What is in it |
|---|---|
| `ui_types.hpp` | `Rect`, `Edges`, `Axis`, `Align`, `Size`, `BoxStyle`, `WidgetId`, `Style` — the value vocabulary |
| `layout.hpp` | `LayoutTree`: the two-pass box solver, and nothing else. No packet, no input, no font |
| `nine_slice.hpp` | `build_nine_slice` — pure math, nine quads that exactly tile a destination |
| `ui_input.hpp` | `InputFrame` (what the UI reacts to) and `InputCapture` (what it took) |
| `ui_context.hpp` | `Context` + `UiState` — identity, focus, scrolling, emission |
| `widgets.cpp` | The widget vocabulary. Same class as `Context`, split for readability |

## Determinism

The merged frame must be byte-identical at any `BRACKEN_WORKERS` count, so
every container here is a **sorted vector** and never a hash map: emission order
is a pure function of declaration order, and an unordered container's iteration
order is neither pinned nor stable across standard-library versions.

`LayoutTree::solve` is likewise a pure function of the tree and the viewport —
same tree, same rectangles, bit for bit — which is what
`SolvingTheSameTreeTwiceIsBitIdentical` in `layout.test.cpp` asserts.

## Threading

Tier `owned`. One `Context` per UI surface, driven from ONE sim job, because it
emits into ONE `FrameBuilder` lane and a lane aborts if two threads are inside
it at once. Never fan a UI build out into a shared lane.

There is deliberately no `BRACKEN_ASSERT_SIM_STAGE` on `Context::begin`: a host
is free to drive a `Context` from a main-thread tool as long as it owns the
lane, and every test in this module does exactly that. The purity scan is the
stronger statement, and it holds for lines nobody has a test for yet.
