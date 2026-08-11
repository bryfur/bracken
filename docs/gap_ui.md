# The UI layer (`src/bracken/ui/`)

An immediate-mode UI that runs **entirely in the sim stage**: it touches no
GPU, no SDL and no main-thread API, measures text against a frozen CPU glyph
table, and emits into the frame packet through `scene::FrameBuilder::UiLane`.

This file is the integrator's folding sheet — a user guide, a spec, an
architecture note, and the exact prose to fold into `CLAUDE.md`.

---

## 1. User guide — building a UI

### 1.1 The shape of it

One host-owned `ui::Context` per UI surface. It is long-lived: hot/active/
focus state, scroll offsets and last frame's widget rectangles all live in it
between frames. Declare widgets inside `on_simulate`, then gate the game's
input on what the UI took.

```cpp
#include "bracken/bracken.hpp"

class Game : public bracken::Application {
    bracken::ui::Context ui_;
    bracken::ui::InputCapture ui_capture_{};
    bracken::ActionMap actions_;
    int score_ = 0;
    float volume_ = 0.8f;
    bool fullscreen_ = false;

    void on_simulate(bracken::SimFrame& sim) override {
        namespace ui = bracken::ui;

        // 1. Configure. The font pointer is the frozen metrics table the
        //    engine exported at renderer init — see §1.2.
        ui::ContextConfig cfg;
        cfg.font = &sim.font_metrics();
        ui_.configure(cfg);

        // 2. Declare. Nothing reaches the packet until end().
        ui_.begin(sim.frame().ui(), ui::InputFrame::from_snapshot(sim.input()),
                  ui::Rect{0.0f, 0.0f, static_cast<float>(sim.screen_width()),
                           static_cast<float>(sim.screen_height())});

        ui::BoxStyle panel;
        panel.padding = ui::Edges::all(12.0f);
        panel.spacing = 6.0f;
        panel.width = ui::Size::px(300.0f);

        ui_.begin_panel(panel);
        ui_.label(std::format("Score: {}", score_), 0xFFFFD080);
        if (ui_.button(ui::id("reset"), "Reset", ui::Size::flex(1.0f))) {
            score_ = 0;
        }
        (void)ui_.toggle(ui::id("fullscreen"), "Fullscreen", fullscreen_, ui::Size::flex(1.0f));
        (void)ui_.slider(ui::id("volume"), volume_, 0.0f, 1.0f, ui::Size::flex(1.0f));
        ui_.end_panel();

        // 3. Solve, resolve focus and scrolling, emit.
        ui_.end();
        ui_capture_ = ui_.capture();

        // 4. The game sees a snapshot with exactly the channels the UI took
        //    zeroed out. The whole action layer goes quiet for free.
        const bracken::InputSnapshot game_in = ui_capture_.mask(sim.input());
        actions_.update(game_in);
        // ... game simulation using `actions_` / `game_in` ...
    }
};
```

Run the UI **before** the game logic if you want the gate to apply in the same
frame. The playground runs it after (its HUD is built at the end of the frame),
which costs one frame of gate latency — documented at its `build_ui`.

### 1.2 Text metrics

`sim.font_metrics()` returns a frozen `FontMetrics` (`contracts/font_metrics.hpp`):
advances, kerning pairs and vertical metrics for one font at one size, exported
once on the main thread at `init_renderer()` and immutable for the process's
life. It reproduces the pen walk the text renderer performs, so a measured
width and a drawn string agree to the pixel.

```cpp
const float w = sim.font_metrics().measure_width("Continue", 1.0f);
const float h = sim.font_metrics().measure_height(1.0f);
```

Its `valid()` is false on a machine with no installed font; every measure then
returns 0 and `ui::Context::Diagnostics::text_without_font` counts the
declarations that could not be measured.

`FrameSink::ui().add_text_wrapped(...)` measures against the same table now, so
existing wrapped text starts wrapping where the glyphs actually end instead of
at a fixed 8 px per character.

### 1.3 Widget identity

```cpp
ui::id("save")                       // from a stable call-site label
ui::id(ui::id("inventory"), row)     // list rows
ui_.push_id(ui::id("left_panel"));   // scope: the same label under two panels
ui_.pop_id();                        //        is two different widgets
```

**The trap.** An id derived from something that changes every frame (a
formatted string containing the frame index, or a live value) resets that
widget's state every frame *and* leaks one retained entry per frame. Retention
is bounded by `ContextConfig::retain_frames` (default 120), so the leak has a
ceiling, and `Diagnostics::retained_entries` is the instrument.

### 1.4 Focus, keyboard and gamepad

Tab order is declaration order. `nav_next`/`nav_prev` walk it (Tab / Shift+Tab
by default, wrapping unless `ContextConfig::wrap_focus` is false); the
directional verbs move focus geometrically over the solved rectangles, so a
D-pad works on a grid. `nav_activate` fires the **focused** widget regardless of
where the pointer is. A focused slider claims Left/Right for its own value and
suppresses the focus move on that press.

Focus resolves at `end()` — it needs solved rectangles — so a navigation key
pressed on frame N moves the focus that frame and the newly focused widget can
be activated from frame N+1. Anything a widget draws *because* it is focused
(a focus border, a text field's caret) therefore also appears from frame N+1,
because the draw is appended at declaration time.

A focus that stops naming a declared widget is **dropped**, not kept. That
matters more than it sounds: `end()` sets `capture().keyboard` from the focused
id, so a ghost focus — a menu closed with a row focused, a list row that
scrolled out of existence — would silence the host's entire keyboard channel
for the rest of the process, with nothing on screen to explain why the
character stopped walking.

**The gamepad path, and why it needed engine work.** `InputSnapshot` already
carried `menu_up_pressed` and friends, and it looked like the answer. It was
not: `InputHandler` only sets those while `set_game_input_enabled(false)`, so
they are dead during play, and a HUD is navigable during play. It also carried
`controller_primary`/`controller_secondary`, which are polled LEVELS — a
focused button mapped onto them fires once per frame for as long as A is held.

So the second pass added a separate, ungated, EDGE-only channel:
`pad_nav_up/down/left/right`, `pad_accept_pressed`, `pad_cancel_pressed`,
`pad_next_pressed`, `pad_prev_pressed`. Sources are the D-pad buttons, the
SOUTH/EAST face buttons, the shoulder buttons, and the LEFT STICK — the stick
latches a direction when it crosses magnitude 0.5 and re-arms only below 0.35,
so a stick resting on the threshold cannot chatter out one edge per frame.
`InputHandler::update_ui_pad_nav()` polls it every frame regardless of the
game-input mode; `clear_menu_inputs()` (already called from
`Application::pump_events`) clears the edges, so no host has to learn about it.

There is deliberately **no hold-to-repeat**: one flick is one step. Repeat rate
is a policy decision and a menu that wants one owns it.

`InputCapture::mask` treats the pad edges as a keyboard-class channel — a UI
that took the keyboard took the pad too, or a menu and the game both act on
one press.

### 1.5 Scrolling and clipping

```cpp
ui::BoxStyle list;
list.width = ui::Size::flex(1.0f);
list.height = ui::Size::px(180.0f);
ui_.begin_scroll(ui::id("inventory"), list);
for (uint64_t i = 0; i < items.size(); ++i) {
    if (ui_.button(ui::id(ui::id("item"), i), items[i].name, ui::Size::flex(1.0f))) {
        use(items[i]);
    }
}
ui_.end_scroll();
```

`begin_scroll` pushes a real scissor into the packet (a `scene::UiClipCommand`,
intersected with any enclosing clip) and `end_scroll` restores the enclosing
one. The wheel scrolls it while the pointer is inside; focus navigation scrolls
the focused child into view; the offset clamps to `[0, content - viewport]` and
does not move at all when the content fits. Hit testing rejects a click that
lands outside the clip, so a row scrolled out of view cannot be pressed.

`begin_scroll` also pushes its own id scope, so a row's real id is
`combine(id("inventory"), id(id("item"), i))`.

### 1.6 Sprites and nine-slice

The sim stage cannot hold an `SDL_GPUTexture*`. Register once on the main
thread, reference by handle forever after:

```cpp
// main thread, e.g. in on_init after init_renderer()
panel_tex_ = scene_renderer().register_ui_texture(my_texture);

// sim stage
ui::NineSlice slice;
slice.texture_size = {48.0f, 48.0f};
slice.border = ui::Edges::all(12.0f);
ui_.begin_panel_sprite(panel_tex_, slice, panel_style);
ui_.label("Framed");
ui_.end_panel();

ui_.image(icon_tex_, {32.0f, 32.0f});
```

An id the renderer cannot resolve draws a solid tint quad and is counted in
`FrameDiagnostics::ui_textures_unresolved` — never a silently missing panel.

### 1.7 The one-frame rule

`button()` returns its result at the call site, which is **before** the layout
solve. It therefore hit-tests against the rectangle that widget occupied last
frame. Consequences, all intentional:

* a widget cannot be clicked on the first frame it exists;
* clicking a widget that moved between frames uses where it was;
* this is one frame on top of the one the render pipeline already has.

The alternative — returning results from `end()` — makes the call site unable
to branch on a click, which is the entire ergonomic point of immediate mode.

### 1.8 Testing a UI headlessly

Everything above works with no GPU and no window:

```cpp
bracken::testing::TestFrame frame(1u, bracken::scene::FramePacket::k_default_arena_bytes, &font);
ui_.begin(frame.sink().ui(), input, viewport);
my_menu.declare(ui_);
ui_.end();
frame.seal();
EXPECT_EQ(frame.ui_text().size(), 4u);
EXPECT_EQ(frame.ui_clips().size(), 2u);
```

`src/bracken/ui/widgets.test.cpp` is a full worked example: a pause menu built
from panel + column + label + button + toggle + slider + scroll view, driven by
mouse and by gamepad, with the input-capture gate asserted.

---

## 2. Spec

### 2.1 `contracts/font_metrics.hpp` — `bracken::FontMetrics`

| Member | Meaning |
|---|---|
| `line_height`, `line_skip`, `ascent`, `descent` | vertical metrics in font pixels |
| `pixel_size` | the size the atlas was rasterized at (18 today) |
| `fallback_advance` | charged to a codepoint outside the table (the `'?'` advance) |
| `advance` | dense `int16_t[560]` over U+0020..U+024F; `k_absent` where the font has no glyph |
| `sparse_advance` | sorted `(codepoint, advance)` outside the dense range |
| `kerning` | sorted `((prev<<32)\|cur, kerning)`, NONZERO pairs only |

| Method | Contract |
|---|---|
| `valid()` | false until a font was exported; every measure then returns 0 |
| `has_advance(cp)` | false when `advance_of` would fall back |
| `advance_of(cp)` / `kerning_of(prev, cur)` | pen primitives, in unscaled font pixels |
| `measure_width(utf8, scale)` | the authoritative pen walk: kerning before the glyph, then the advance, accumulated as integers and scaled exactly once |
| `measure_height(scale)` | `line_height * scale` — the glyph CELL, matching the emitted quads |
| `count_fallbacks(utf8)` | how many codepoints would measure wrong |
| `fit_prefix(utf8, max_width, scale, out_width)` | longest prefix that fits, on a UTF-8 boundary; monotone in `max_width` |
| `wrap_point(utf8, max_width, scale)` | `{consumed, emit}`; `consumed > 0` always, so a wrap loop terminates |

Constants: `k_first_cp = 0x20`, `k_last_cp = 0x24F`, `k_dense_count = 560`,
`k_absent = INT16_MIN`.

**Two stated limits.** (a) Kerning is exported over the printable-ASCII cross
product only — the dense range's cross product is 313 600 FreeType lookups at
init against 9 025, so Latin-1 and Latin Extended-A kern as zero. (b) A
codepoint outside the exported range measures with `fallback_advance`. Neither
is a divergence between measuring and drawing: `TextRenderer::build_string_quads`
consumes the same table, so whatever it covers, both walks agree.

### 2.2 `contracts/ui_texture.hpp` — `bracken::UiTextureId`

A bare `uint32_t` handle, `0` = invalid. Registered on the main thread
(`SceneRenderer::register_ui_texture`), resolved on the main thread at consume.
Ids are **never recycled**: `unregister_ui_texture` clears the slot so a stale
id resolves to null rather than to somebody else's texture.

### 2.3 `ui/ui_types.hpp`

`Rect` (half-open `contains`, `intersect`), `Edges`, `Axis`, `Align`
(`Start`/`Center`/`End`/`Stretch`), `SizeMode` (`Content`/`Fixed`/`Flex`),
`Size`, `BoxStyle`, `WidgetId`, `Style` (one flat struct of colors + four
metrics; real theming is out of scope), and the id functions `id(label)`,
`id(parent, index)`, `combine(parent, child)` — all FNV-1a 64, deliberately not
`std::hash` (which is not specified to be stable across library versions, and a
widget id that moves between compiler releases loses every player's scroll
position).

### 2.4 `ui/layout.hpp` — `ui::LayoutTree`

Two passes. **Measure** (bottom-up) computes each node's desired size: a leaf
wants its `content` unless a `Fixed` size overrides it; a box wants the sum of
its children along the main axis plus spacing, the max along the cross axis,
plus padding. **Arrange** (top-down) assigns rectangles: non-flex children take
their measured main size, flex children split the leftover in proportion to
their weights, and the cross axis is placed by `BoxStyle::cross`.
`Align` on the main axis places the leftover when no child is flex; `Stretch`
there reads as `Start`.

`Stretch` on the CROSS axis fills — but only for a child that did not ask for a
cross size itself. An explicit `Size::px` **wins over** the parent's `Stretch`,
the same rule flexbox has (`align-items` only stretches an `auto` cross size);
a `Size::flex` or `Size::content` child stretches. Without that, a 100 px-wide
toggle dropped into a `Stretch` column would be silently widened to the column
and there would be no way to say "this one is narrow".

Two properties the tests pin:

* **No drift.** Flex sizes are the difference of two *cumulative* fractions,
  and positions are an offset from the content origin rather than a running
  cursor. A twelve-item flex row's last right edge lands on the content edge to
  the float; a running cursor rounds once per child and ends visibly short.
* **`content_of` vs `content_extent_of`.** The first is the size after an
  explicit `Fixed` override; the second is what the children wanted before it.
  A scroll viewport is `Fixed` by definition, so a max-offset computed from
  `content_of` is always zero — that is what the split exists for.

Constants: `k_invalid = 0xFFFFFFFF`, `k_max_depth = 64` (containers opened past
it are discarded and counted; their `end_box` is absorbed so the tree stays
well-formed). An unbalanced `end_box` is counted and ignored.

### 2.5 `ui/nine_slice.hpp`

`build_nine_slice(slice, dst, out)` writes up to 9 `NineSliceQuad`s that
exactly tile `dst` (8 with `draw_center` false, 0 for an empty `dst`).
Destination band widths come from differencing absolute edge positions, so the
patches share edges and cannot open a hairline seam. Borders wider than `dst`
scale down proportionally rather than inverting a quad, and borders wider than
the **source** collapse the middle UV band to zero rather than running it
backwards (a quad with `u1 < u0` samples its band mirrored — corner art drawn
inside out). UVs come from the **source** border, so the corner is still the
corner at any box size.

### 2.6 `ui/ui_input.hpp`

`InputFrame` — mouse position and edges, raw wheel, eight nav verbs, and a
text-entry channel (`text` / `backspace` / `enter`) for `text_field`. `text` is
a **view into the snapshot**, not a copy — the same rule the rest of the frame
path follows; `InputFrame::from_snapshot(sim.input())` is safe because that
snapshot outlives the frame.

`InputFrame::from_snapshot` is the default mapping:

| Field | Sources |
|---|---|
| `nav_up/down/left/right` | `menu_*_pressed`, the arrow scancodes, `pad_nav_*` |
| `nav_activate` | `menu_select_pressed`, Return, Space, `pad_accept_pressed` |
| `nav_cancel` | Escape, `menu_toggle_pressed`, `pad_cancel_pressed` |
| `nav_next` / `nav_prev` | Tab / Shift+Tab, `pad_next_pressed` / `pad_prev_pressed` |
| `text` / `backspace` / `enter` | the snapshot's text-input channel |

**The pad mapping uses `pad_*`, not `controller_primary`/`controller_secondary`,
and the difference is load-bearing.** Those two are LEVELS polled from the
trigger axes and the face buttons, so a UI driven off them re-activates the
focused widget on every frame the button is held. They are also only ever
meaningful to a menu while `set_game_input_enabled(false)` — i.e. while the
host has paused the whole game — which a navigable HUD is not. The `pad_*`
channel added in the second pass is a set of one-per-press EDGES that fire
whatever the game-input mode.

`InputCapture{mouse, keyboard, wheel}` and `mask(snapshot)` returns a copy with
exactly the captured channels zeroed: mouse capture leaves `keys_held` alone,
keyboard capture leaves `mouse_x`/`mouse_y` alone (a position is not an event),
and wheel capture also zeroes `camera_zoom_delta` because it is the same
physical event.

### 2.7 `ui/ui_context.hpp` — `ui::Context`

Frame: `configure` / `begin(lane, input, viewport)` / declarations / `end()` /
`capture()`.
Containers: `begin_column` / `begin_row` / `begin_grid` / `end_box`,
`begin_panel` / `begin_panel_sprite` / `end_panel`,
`begin_scroll` / `end_scroll`, `push_id` / `pop_id`.
Widgets: `label`, `label_wrapped`, `button`, `toggle`, `radio`, `slider`,
`text_field`, `progress_bar`, `separator`, `image`, `spacer`, `rect`.

Return conventions, which differ per widget deliberately:

| Widget | Returns true when |
|---|---|
| `button` | it fired this frame (release inside, or activate while focused) |
| `toggle` | it fired — `value` has just been flipped |
| `radio` | the SELECTION CHANGED; re-picking the current option returns false |
| `slider` | `value` differs from what it was on entry (drag or Left/Right) |
| `text_field` | ENTER committed it — **not** on every keystroke |

`text_field` is deliberately minimal and its limits ARE the API: append and
backspace only, no caret movement, no selection, no clipboard, and no blink (a
blink needs wall-clock time and `begin()` is deliberately not given any — a
frame's UI must be a pure function of its declarations). Backspace removes a
whole UTF-8 sequence and an over-budget typed run is refused WHOLE rather than
truncated mid-character; both exist so a broken byte sequence can never reach
the packet, where the next `measure_width` would charge `fallback_advance` for
the wreckage.

`progress_bar` and `separator` take no id and never enter the focus order —
there is nothing to activate.

Focus: `set_focus`, `focused`, `hot`, `active`, `focus_order`.
Introspection: `rect_of`, `emitted_command_count`, `diagnostics`, `state`.

`Diagnostics`: `unbalanced_end_box`, `depth_overflow`, `clip_stack_underflow`,
`duplicate_widget_id`, `text_without_font`, `text_fallback_advances`,
`retained_entries`.

`UiState` holds `hot`/`active`/`focused`, a sorted scroll table and a sorted
last-rect table. **Sorted vectors, never hash maps** — emission order must be a
pure function of declaration order for the merged frame to stay byte-identical
at any worker count.

A fully transparent `Style::panel_border` is not drawn at all rather than drawn
with alpha 0: four invisible quads still cost four quads of vertex batch, and a
host porting an existing HUD needs "panel with no border" to be free.

### 2.8 New packet commands

```cpp
struct UiTexturedQuadCommand {   // 40 B — exactly UiButtonCommand's width
    float x, y, w, h;
    float u0, v0, u1, v1;
    bracken::UiTextureId texture;
    bracken::core::Color tint;
};
struct UiClipCommand {           // 20 B
    float x, y, w, h;
    uint32_t enabled;            // uint32, not bool: padding-free by construction
};
```

Emitted by `UiLane::add_textured_quad` / `set_clip` / `clear_clip`, forwarded
through `FrameSink::ui()`, merged by the existing `ui_commands_` chain (no seal
change), read back by `TestFrame::ui_textured_quads()` / `ui_clips()`.
`static_assert`s pin both as padding-free and pin the textured quad at
`<= sizeof(UiButtonCommand)` — a wider payload would silently halve the
commands per 2048-byte UI chunk for every host.

The clip stack is owned by the **caller**: a lane that remembered clip depth
would make the seal merge order-dependent, and the merge is
(lane-index, emission-order) precisely so a frame is byte-identical at any
worker count.

---

## 3. Architecture

### 3.1 Where it sits

`ui/` is **above** `scene/` — it emits into a `FrameBuilder` lane. Therefore
`render/` must never include `ui/`: `render → ui → scene → render` would close
a cycle and `module_layering.test.cpp` HeaderGraphIsAcyclic would fail. That is
why `FontMetrics` lives in `contracts/` (so `render/` can produce one) and why
`decode_utf8` moved down to `core/utf8.hpp` (so `contracts/` can use it —
`contracts/` may include only `core/`, `contracts/`, glm and `<std>`).
`render::decode_utf8` is now a using-declaration onto `core::decode_utf8`, so
every existing call site is untouched.

### 3.2 Threading tiers

| Thing | Tier | Notes |
|---|---|---|
| `ui::Context` | `owned` | one per surface, driven from ONE sim job; it emits into one lane, and a lane aborts if two threads are inside it |
| `FontMetrics` (after export) | `worker-safe` | immutable value; read by const reference from any job |
| `TextRenderer::export_metrics` | `main-thread` | called once, at renderer init |
| `UIRenderer::register_texture` / `unregister_texture` | `main-thread` | asserted with `BRACKEN_ASSERT_MAIN_THREAD` |
| `SceneRenderer::register_ui_texture` | main-thread | asserted with `BRACKEN_ASSERT_NOT_SIM_STAGE` |
| `SceneRenderer::font_metrics` | main-thread | asserted with `BRACKEN_ASSERT_NOT_SIM_STAGE` — sim code reads its own frozen copy through `SimFrame` |
| `UiLane::add_textured_quad` / `set_clip` | `sim-stage` | ordinary lane emits, `assert_filling()` contract |
| `SimFrame::font_metrics` | `worker-safe` | returns the frozen table, or a frozen empty singleton |

**The tier audit is a SOURCE SCAN, not only assertions.** A tier macro is a
debug-only runtime check on the callee, and it only fires if some test happens
to execute that call on a bad thread. "The UI module cannot touch the GPU" is
a stronger claim, and `src/bracken/ui/ui_purity.test.cpp` makes it directly:
no file under `src/bracken/ui/` may include `bracken/gpu/`, `bracken/render/`,
`bracken/scene/scene_renderer*`, `SDL3/SDL_gpu.h`, `SDL3/SDL_render.h` or
SDL_ttf — so the call cannot be written, for every line of the module at once.
The same file pins the reverse edge (`render/` never includes `ui/`, which
would close `render → ui → scene → render`) and that `contracts/font_metrics.*`
never reaches back into the renderer that produces it.

`ui_input.hpp` including `input_handler.hpp` — and through it SDL's scancode
and event headers — is deliberately allowed. `InputSnapshot` is plain data
whose key edges are indexed by scancode; naming a scancode is not touching a
device.

`ui::Context::begin` deliberately carries no `BRACKEN_ASSERT_SIM_STAGE`. A host
may legitimately drive a Context from a main-thread tool as long as it owns the
lane, and every test in the module does exactly that. What it does carry is a
`BRACKEN_CONTRACT` on begin/end pairing — a host mistake, live in release,
because an unbalanced pair silently discards a frame's declarations and the
symptom (a UI that intermittently vanishes) points nowhere near the cause.

**How the metrics are published, exactly.** `TextRenderer::init` exports into
its own frozen table. `Application::init_renderer()` copies that into
`Application::font_metrics_` — on the main thread, before `run()` has ever
kicked a sim job — and never writes it again. `kick_sim` hands
`&font_metrics_` to the frame's `FrameBuilder` and to the `SimFrame`. The
happens-before edge that makes a pool worker's read safe is the sim job's own
`JobSystem::submit`/execute edge, the same edge that publishes the input
snapshot. There is no atomic and none is needed: after init this is an
immutable value. A `FrameBuilder` constructed without one (a hand-built test
frame) falls back visibly rather than crashing.

### 3.3 Where it sits in the frame

Unchanged in shape. `on_simulate` declares → `Context::end()` emits into the
lane → `seal()` publishes → `SceneRenderer::render_ui_commands` replays into
`UIRenderer` → `end_ui()` runs `execute()`. Billboards are still recorded after
packet UI and still paint over it; the single `post_ui_callback` (DevUI) still
draws after everything.

### 3.4 The z-order fix

`UIRenderer::execute` used to run two hard phases — every primitive, then every
queued text — so a panel recorded *after* a label could never cover it, no
matter what order the host recorded in. Recording now appends to a
submission-ordered `ops_` list of `{Geometry|Text, first, count}` spans and
`execute` walks it, alternating the UI and Text pipeline binds; `TextRenderer`
grew `queued_run_count()` / `draw_run_range()` / `clear_queued_text()` so a
span of text draws can be drawn without retiring the whole queue.

**The common case is unchanged.** A HUD that records its panel and then its
labels produces exactly two ops — one geometry range, one text span — which is
byte for byte the command stream it always produced. The extra pipeline binds
are paid only by content that actually interleaves.

#### The invariant `execute()` is built on, and how it is pinned

`execute()` walks `ops_` and, for a Geometry op, draws
`draw_ranges_[first .. first+count)`; for a Text op it maps
`queued_text_draws_[first .. first+count)` onto the `TextRenderer` runs those
draws produced. That is correct **only if the ops exactly partition both
vectors, contiguously and in order**. A gap silently drops a draw, an overlap
draws it twice, and a non-monotone `first` reorders the frame — precisely the
class of bug the two-phase form used to have, and none of it is visible from
eyeballing the op list.

`UIRendererOps.OpsExactlyPartitionRangesAndTextDraws` asserts the partition
directly, over a recording containing every transition that can open an op or a
range: geometry/text alternation, a texture switch, a clip change, and a clip
change that happens to be a no-op. Four more tests pin the individual rules —
a later geometry lands in a later range, an empty string opens no op (an op with
`count == 0` would bind a pipeline for nothing and break the guard), a clip
change splits a RANGE and not an OP (a clip is scissor state, not pipeline
state), and `draw_button`'s own internal interleaving does not break it.

The emitting half is pinned separately, in
`UiContext.APanelDeclaredAfterALabelIsEmittedAfterIt` — the packet order that
the renderer walk then honours. The two together are the whole claim.

The parts of the walk that a headless test **cannot** reach are the GPU state
transitions: that `bind_ui_state()` re-pushes the screen uniform and re-binds
the vertex buffer after a text span has stolen the pipeline, and that
`draw_run_range` hands the scissor back unchanged so a clip cannot leak out of
a text span. Those were verified by reading, not by running. See the report.

### 3.5 Clipping

`DrawRange` and `TextRun` each carry a clip rectangle. `range_for` opens a new
range when the texture **or** the clip changes; the single full-screen
`SDL_SetGPUScissor` became a per-range set inside the walk, issued only when
the wanted scissor differs from the bound one — so an unclipped frame issues
exactly the one call it always did. `draw_run_range` restores the caller's
scissor before returning, so a clip can never leak out of a text span, and
`render_ui_commands` clears the clip at the end of the packet replay so a host
that pushes without popping cannot crop the billboards.

### 3.6 Lifetimes

`ui::Context` copies every declared string into its own arena at declaration
time, so `label(std::format(...))` — whose temporary dies at the end of the
full expression — is safe. `UiLane::add_text` then copies again into the packet
arena. No new lifetime exception is added to the frame packet.

---

## 4. CLAUDE.md delta

**Module table** — new row after `dev/`:

> | `ui/` | `bracken::ui` | The immediate-mode UI layer, run entirely in the SIM STAGE: `LayoutTree` (row/column/grid solver, no drift), `ui::Context` (widget identity, hot/active/focus, keyboard AND gamepad navigation, scrolling with real clipping and scroll-into-view, input capture), `widgets.cpp` (label/wrapped label/button/toggle/radio/slider/text field/progress bar/separator/image/panel/scroll view), plus `nine_slice.hpp` and `ui_input.hpp`. Touches no GPU and no SDL — it measures text through `contracts/font_metrics.hpp` and emits through `scene::FrameBuilder::UiLane`, and `ui/ui_purity.test.cpp` proves it by scanning the module's includes rather than trusting a runtime assert. Sits ABOVE `scene/`, so `render/` must never include it. `src/bracken/ui/README.md` is the module's contract page; `docs/gap_ui.md` is the full write-up |

**`contracts/` row** — append to the list of pure-data types:

> ...`GraphicsSettings`, `RenderStats`, **`FontMetrics`** (the frozen CPU glyph
> table sim-stage text layout measures against — exported once at renderer init
> and immutable thereafter) and **`UiTextureId`** (the packet-safe name for a UI
> texture, since the sim stage may never hold an `SDL_GPUTexture*`).

**Load-bearing constants** — append:

> - `FontMetrics::k_first_cp = 0x20` / `k_last_cp = 0x24F` (`contracts/font_metrics.hpp`)
>   — the exported glyph range, 560 dense advances. A codepoint outside it
>   measures with `fallback_advance` and DRAWS at the same advance (the text
>   renderer consumes the same table), so measuring and drawing agree; what is
>   wrong is that both are wrong. Kerning is exported over printable ASCII
>   only — the dense cross product is 313 600 FreeType lookups at init against
>   9 025 — so accented Latin kerns as zero.
> - `LayoutTree::k_max_depth = 64` (`src/bracken/ui/layout.hpp`) — container
>   nesting guard. Boxes opened past it are discarded and counted; their
>   `end_box` is absorbed so the tree stays well-formed.
> - `ContextConfig::retain_frames = 120` (`src/bracken/ui/ui_context.hpp`) —
>   how long a retained widget entry survives after it stops being declared.
>   It is what bounds the unstable-id leak; see the trap on `WidgetId`.
> - `TextRenderer::INITIAL_TEXT_QUADS = 4096` — no longer a cap. The text
>   vertex buffer GROWS (a scrolling list is a large frame every frame);
>   `MAX_TEXT_QUADS = 1<<20` is the runaway ceiling.

**Gotchas** — append:

> - **The UI hit-tests against LAST frame's rectangle.** `ui::Context::button`
>   returns its result at the call site, which is before the layout solve, so
>   a widget cannot be clicked on the first frame it exists and a widget that
>   moved is clicked where it was. One frame, on top of the one the render
>   pipeline already has. Returning results from `end()` instead would make the
>   call site unable to branch on a click, which is the point of immediate mode.
> - **`UiLane::add_text_wrapped` now measures the real font.** It used to wrap
>   proportional text on a hardcoded 8 px per character, which under-measures
>   wide text — the playground's keybind strip ran off the right edge of the
>   screen instead of wrapping. It falls back to the old guess only when the
>   frame carries no `FontMetrics` (a hand-built test frame, or a machine with
>   no font). Treat the fallback as "no answer available", not as a mode.
> - **A widget id that changes every frame leaks.** Retention is bounded by
>   `ContextConfig::retain_frames`, so the ceiling is fixed, but the widget's
>   scroll offset and focus reset every frame. `Diagnostics::retained_entries`
>   is the instrument.
> - **The UI pass paints in strict submission order now, text included.** It
>   used to draw all geometry then all text, so a panel could not cover a
>   label. A HUD that draws its panel first is unaffected (two ops, the same
>   command stream); anything that interleaves pays one extra pipeline bind per
>   alternation.

**Screenshot baselines** — the playground HUD was ported to `ui::Context` with
explicit fixed row heights that reproduce the previous y offsets 20/44/64/84/
104/124 exactly, and a transparent panel border (not drawn). The one intended
visual change is the bottom keybind strip, which now WRAPS to a second line
instead of running off the right edge. **The RT-off rows were not re-measured —
no build tree exists in this worktree by design.**

The expected size of that change, worked out from the actual font rather than
guessed, so the integrator knows what "unchanged" should look like:

* At 1600×900 the strip's box is `screen_width - 20 = 1580` px and the string
  is 228 characters. Against DejaVuSans at 18 px, scale 0.9, its pen width is
  **1832 px**.
* The OLD 8-px-per-character guess allowed `1580 / (8 * 0.9) = 219` characters
  per line, so the string did **not** wrap — it drew as one 1832 px line
  starting at x = 10, i.e. the last ~135 px ran past the 1600 px swapchain and
  were never rasterized.
* The NEW wrap breaks at 1580 px. Line 1 is therefore very nearly the same
  pixels as before (the part that overflowed was already invisible); line 2 is
  **new** — roughly 250 × 21 px of box at the bottom of the frame.
* Glyph coverage inside a text line is ~20%, and the delta per lit pixel is
  ~120/255, so the expected `mean_luma` shift is
  `250 * 21 * 0.2 * 120 / (1600 * 900) ≈ 0.09` — under the tenth the pins are
  quoted to. `86.3 / 45.6` should therefore still read `86.3 / 45.6`.

If it does move, the one-line revert that restores the previous rendering
exactly is to draw the strip with `UiLane::add_text` (one line, clipped at the
screen edge) instead of `add_text_wrapped`. Do not "fix" it by restoring the
8 px guess: that guess is the bug, and it is engine-wide.

**Second pass, additionally:**

* The widget set is complete: `radio`, `text_field`, `progress_bar` and
  `separator` joined `label`/`label_wrapped`/`button`/`toggle`/`slider`/
  `image`/`spacer`/`rect`, and the playground's F9 demo menu declares every one
  of them. Nothing in that menu is declared during a smoke run.
* `InputSnapshot` gained the ungated gamepad EDGE channel (`pad_nav_*`,
  `pad_accept_pressed`, `pad_cancel_pressed`, `pad_next_pressed`,
  `pad_prev_pressed`) and `InputFrame::from_snapshot` maps it. It **no longer
  maps `controller_primary`/`controller_secondary`**, which are levels — a
  gotcha worth carrying into CLAUDE.md:

> - **Do not map a menu onto `InputSnapshot::controller_primary`.** It and
>   `controller_secondary` are polled LEVELS (trigger axes plus face buttons),
>   so anything treating them as "the player pressed A" fires once per frame for
>   as long as the button is held. The `pad_*_pressed` fields beside them are
>   one-per-press edges, and unlike `menu_*_pressed` they are not gated on
>   `set_game_input_enabled(false)` — which is what makes a HUD navigable while
>   the game is running.

* Scroll-into-view now runs for **every enclosing** scroll container, not only
  the innermost: the search asks `LayoutTree::is_descendant_of` rather than
  matching a cached parent index. Without it, tabbing into a nested list
  scrolled the inner window while the outer one still had it off screen.
