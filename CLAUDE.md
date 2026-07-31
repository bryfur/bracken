# Project: Bracken

A standalone **C++23** real-time rendering engine (library/target name
`bracken`, formerly `mmo_engine`), extracted from the `mmo-4`
game project. Built on **SDL3 GPU** (Vulkan/Metal/D3D12 backends) with
HLSL→SPIR-V shader compilation, glTF model loading, skeletal animation, and a
clustered-forward renderer.

This repo contains **only the engine** — no game, server, networking, or ECS.
Rigid-body physics *is* engine infrastructure here: `src/bracken/physics/`
wraps Jolt Physics behind a glm-only pimpl API (owner decision 2026-07-21 —
see engine_design_notes.md). It builds as a static library (`bracken`) plus a GoogleTest suite
(`bracken_tests`), and one example app: `examples/playground`, a medieval
village orb-collector demo that doubles as the integration test (`--smoke N`
self-drives for N frames and exits — run it after changes that could affect
rendering). The village is assembled from a modular building pack, and the
flora (trees/bushes/rocks/grass tufts/flowers) converted 1:1 from a stylized
nature pack, by `examples/playground/tools/build_medieval_assets.py` (raw
packs in `assets/glTF/` and `assets/nature/` — untracked; outputs `.glb`s +
terrain textures + `village_layout.h` into `assets/generated/`, staged next
to the exe by CMake).
Without the generated assets the playground still builds and the smoke test
still passes — it falls back to a procedural-only scene (generator trees +
rocks, engine blade grass). In the full scene everything except the
character/orbs/effect meshes is pack art, and the engine's blade grass is
replaced by instanced nature-pack grass tufts. All scene dressing (props,
trees, rocks, flowers — everything with a silhouette) is hand-placed area by
area in `examples/playground/scene_layout.cpp`; only the grass-tuft ground
cover is a procedural fill. A simulated river (engine `WaterRenderer` +
compute wave sim) runs along the west map edge; wading, ambient impulses and
rain push ripples into it. The playground also runs a dynamic time-of-day +
weather system (`examples/playground/day_weather.*`: animated sun, night with
stars/moon/skyglow, auto-drifting clear/overcast/rain with volume-emitter
rain streaks + ray-cast impact splashes), controllable from the "Playground"
dev panel. Plain `--smoke` runs keep the legacy fixed environment
(deterministic); `PLAYGROUND_TOD_HOUR=<h>` / `PLAYGROUND_WEATHER=<w>` pin it
for art shots. See [README.md](README.md) for the user-facing overview.

## House rules (general)

@.claude/RULES.md

In short: **no full clean rebuilds.** Always `cmake --build build` (or
`ninja -C build`). The first configure fetches and builds SDL_shadercross
(bundling DXC + SPIRV-Tools) and also fetches Jolt Physics (v5.2.0, built as
part of the tree) — SDL_shadercross is the slow part of a cold build; never
throw it away with a clean rebuild. New `.cpp`/`.hpp`/`.test.cpp` files under
`src/bracken/` are picked up automatically by the CMake `GLOB_RECURSE`, so an
incremental build is all that's needed after adding a file.

---

## Documentation map

Read the doc that matches the task before diving into code:

- [docs/user_guide.md](docs/user_guide.md) — consumer-side guide: how a game
  developer uses every engine feature (build/integration, per-subsystem usage,
  defaults, gotchas).
- [docs/technical_spec.md](docs/technical_spec.md) — per-feature specification:
  API surfaces, data structures, algorithms, contracts, constants, known
  limitations/stubs.
- [docs/architecture.md](docs/architecture.md) — **the deep reference**:
  module map, `Application` lifecycle, exact frame/pass order, GPU layer,
  threading & lifetime contracts, constants, known stubs/silent behaviors.
- [docs/engine_design_notes.md](docs/engine_design_notes.md) — rationale +
  decision principles. **Required reading before adding any cross-cutting
  infrastructure** (allocators, watchers, render-graph wiring). Note the
  2026-07-18 decision record: the "default to serial" guidance is superseded
  for threading.
- [docs/multithreading_plan.md](docs/multithreading_plan.md) — **the plan of
  record for the native multithreading migration** (Phases 1–4 landed:
  always-on pool, lock-free job internals, thread-safety tiers, FramePacket,
  Prepare-stage fan-out, and the pipelined frame — `on_simulate` overlapping
  the previous frame's rendering). Read §7
  (synchronization design) before touching any concurrency code.
- [docs/skinned_mesh_format.md](docs/skinned_mesh_format.md) — the `.glb`
  authoring contract (bones, channels, clip naming, IK bone names).
- [docs/effects_system.md](docs/effects_system.md) — particle effects:
  definitions / runtime / rendering split, host responsibilities.
- [src/bracken/shaders/README.md](src/bracken/shaders/README.md) — shader
  naming, the pipeline→shader map, SDL3 GPU binding order, uniform contract.

Keep these accurate: if a change invalidates a documented fact, update the doc
in the same change.

## Architecture

The include root is `src/`; engine headers are included with a `bracken/`-rooted
path (`#include "bracken/render/world_renderer.hpp"`). Everything lives under the
`bracken` namespace, with sub-namespaces per module.

| Module | Namespace | What's there |
|---|---|---|
| `core/` | `bracken::core` | logger, asserts, `Result`, `Handle`, `hash`/`StringId`, Tracy profiler hooks |
| `core/jobs/` | `core::jobs` | `JobSystem` (**always-on** lock-free work-stealing pool — Chase–Lev deques, `atomic::wait` parking), `TaskGroup`, `parallel_for`, task handles. Worker count via `BRACKEN_WORKERS` (=1 is single-worker debug mode, same code path) |
| `core/asset/` | `core::asset` | `FileWatcher` (hot-reload; opt-in via `BRACKEN_HOT_RELOAD=1`) + `AssetRegistry` (host-facing utility — engine systems register on the watcher directly) |
| `memory/` | `bracken::memory` | arena allocator, frame allocator |
| `gpu/` | `bracken::gpu` | SDL3 GPU abstraction — buffers, textures, shaders, graphics + compute pipelines (`gpu_compute.hpp`), `PipelineRegistry`, `gpu_uniforms.hpp` (CPU↔GPU contract) |
| `render/` | `bracken::render` | sub-renderers: terrain, grass, world/skybox, shadow map (CSM), GTAO, TAA, histogram auto exposure (compute), IBL (sky prefilter chain), bloom, volumetric fog, post-process stack (composite incl. color grading), water (surface + compute wave sim), effects, text, UI; `lighting/` = clustered light binning + budgeted local (point/spot) shadow atlas |
| `render_graph/` | `render_graph` | pass scheduling (topo-sort, transient pool). **In-tree + tested but not wired into the renderer** — see design notes |
| `scene/` | `bracken::scene` | cameras, frustum culling, transforms, `FramePacket` (immutable per-frame command snapshot) + `FrameBuilder` (the only write interface) + `PacketRing` (SPSC packet ring), `SceneRenderer` (frame orchestrator) |
| `animation/` | `bracken::animation` | GLB skeleton, `AnimationPlayer` (crossfade), data-driven state machine, two-bone IK |
| `systems/` | `bracken::systems` | camera controller/system, particle `EffectSystem` (sim-stage owned: spawned/updated/snapshotted inside the sim job; the renderer draws the packet snapshot) |
| `procedural/` | `bracken::procedural` | tree + rock generators |
| `physics/` | `bracken::physics` | `PhysicsWorld` — rigid-body physics over vendored **Jolt** (pimpl; Jolt never appears in public headers). Fixed-step accumulator `update()` (tier owned/sim-stage), sphere/box/capsule/heightfield bodies, ray casts; Jolt's step jobs run on the engine `JobSystem` via an internal `JoltJobSystem` bridge — one pool owns the cores |
| `dev/` | `bracken::dev` | `DevUI` Dear ImGui developer overlay (F10 in the playground; no-op stub when `BRACKEN_ENABLE_DEV_UI=OFF`) |
| `shaders/` | — | HLSL sources, compiled to SPIR-V at build time |

`Application` (`src/bracken/application.hpp`) is the base class that owns the
SDL lifecycle, the **pipelined** main loop (depth 2 — Simulate(i+1) overlaps
Render(i)), frame timing, and input. Consumers subclass it and override
`on_init`/`on_simulate(SimFrame&)`/`on_shutdown` (plus optional
`on_main_frame(MainFrame&)` and `on_event`). There is no
`on_update`/`on_render`: the host *describes* a frame through
`SimFrame::frame()`'s `FrameBuilder` — from a pool job — and the engine
draws it.

### Key patterns
- **Y-up** coordinate system (x,z = ground plane, y = height) and
  **reversed-Z zero-to-one depth** (`GLM_FORCE_DEPTH_ZERO_TO_ONE` is PUBLIC on
  the engine target — all projection math is glm `RH_ZO`, with near/far
  swapped in the scene `glm::perspective` call so near→depth 1, far→depth 0;
  scene pipelines compare `GREATER_OR_EQUAL` and clear depth to 0.0. The
  shadow-cascade passes deliberately stay standard-Z — see architecture.md).
- Frame data flows one way, by value: in `on_simulate` the host fills a
  `scene::FramePacket` through `sim.frame()`'s `scene::FrameBuilder` (3D
  draws, lights, UI, effect spawns, camera/environment), the engine's
  sim-stage tail appends the effect-instance snapshot + scene/sky clocks and
  `seal()`s, and the renderer consumes `const FramePacket&`. Everything is
  copied or packet-allocated (bone palettes via `alloc_bone_palette`, UI text
  copied into the packet arena) — **no host-lifetime contracts** cross the
  boundary. A packet is consumed exactly once, then retired to its
  `PacketRing`. `BonePaletteId`s held by the host across `seal()` are stale;
  only re-read them from sealed commands. Do not call `seal()` or
  `set_frame_timing` from `on_simulate` — the engine does both.
- All lighting is **linear-space**; `render::EnvironmentState` colors are
  linear, the composite pass tonemaps to the swapchain.
- Animation is per-instance: an `AnimationPlayer` + `AnimationStateMachine`.
  The host feeds parameters ("speed", "attacking"); the state machine drives
  transitions. Foot IK / body lean are applied by the host *after* the
  animation update, before render submission.
- CPU uniform structs in `gpu/gpu_uniforms.hpp` must match their HLSL cbuffer
  counterparts **byte-for-byte** (`alignas(16)`, HLSL packing rules). This is
  the most common way to silently break rendering.
- Engine code is **game-agnostic**: no gameplay concepts (combat, inventory,
  skills) belong in engine headers. `InputHandler` is raw input only;
  `CameraController` exposes generic shake/focus/fov primitives.
- **Threading** (Phase 4 of [docs/multithreading_plan.md](docs/multithreading_plan.md)):
  the frame **is pipelined** — `Application::run()` kicks `on_simulate(i+1)`
  as a pool job and renders frame i on the main thread while it runs
  (depth 2; one frame of input latency, documented on `Application`). The
  job-system pool is always on — `init_engine` starts it before SDL; parked
  workers cost ~zero. `on_simulate` is tier **sim-stage**: nothing reachable
  from it may touch GPU/SDL/main-thread-tier APIs (asserted in debug); the
  frame **fence** (join → hot-reload poll → deferred settings/screenshot/
  camera-forward → FPS → `on_main_frame`) is the only point where the main
  thread may read live game state. The renderer's **Prepare stage fans out**:
  culling/batching/binning loops run as `parallel_for` chunks over the
  sealed packet, per-chunk partials merged single-threaded in chunk-index
  order — bit-identical at any worker count; no atomics, mutexes, or shared
  cache-line writes inside per-item frame loops; grain guards keep small
  scenes effectively serial. Engine APIs carry thread-safety tiers
  (`main-thread` / `sim-stage` / `worker-safe` / `owned`, asserted in debug —
  see architecture.md "Threading & lifetime contracts"). Every `std::atomic`
  documents its protocol (what it publishes, ordering, release/acquire
  pairing partner) at its declaration; bare `seq_cst` in a hot path needs a
  justifying comment. Run the TSan lane after touching concurrency code.

### Load-bearing constants
- `MAX_BONES = 128` — defined **only** in `animation/animation_types.hpp`. It
  caps the CPU-side per-instance bone-matrix arrays; the skinned shaders read
  a per-frame `StructuredBuffer<float4x4>` of packed palettes indexed by a
  `paletteOffset` uniform, so the GPU side has no compile-time bone count.
  Don't raise it without auditing every `std::array<glm::mat4, MAX_BONES>`.
- `MAX_BONE_PALETTES = 1024` (`scene/frame_packet.hpp`) — bone palettes per
  frame, enforced at **build time** in `FrameBuilder::alloc_bone_palette`
  (over-budget allocations log an error and return an invalid id; draws
  using an invalid id are dropped with a warning).
- `CSM_MAX_CASCADES = 4` (`render/shadow_map.hpp`) — 2×2-quadrant D32 atlas.
- Cluster lighting (`render/lighting/light.hpp`): 16×9×24 grid,
  `MAX_LIGHTS = 256`, `MAX_LIGHTS_PER_CLUSTER = 128`. Local light shadows:
  one 8×8-tile D32 atlas — point budget ≤ 8 (6 face tiles each), spot ≤ 16,
  ≤ 6 tile re-renders per frame (`render/lighting/local_shadows.hpp`).
- Formats (`gpu/pipeline_registry.hpp`): swapchain `RGBA8_UNORM`, HDR scene
  `RGBA16F`, depth `D32_FLOAT`.

### Gotchas (full list in [docs/architecture.md](docs/architecture.md))
- **GPU per-pass timings are stubs** — `GPUTimestampPool` returns zeros (SDL3
  limitation). Use Tracy (CPU) or RenderDoc/Nsight (GPU) for real numbers.
- glTF animation channels targeting non-skin nodes are **silently dropped**
  at load; `Mesh::near_only` exists but is **not enforced**.
- `render_graph/` is not in the frame path — don't route the renderer through
  it without a measured reason (design notes explain why).

---

## Conventions

- **C++23**, CMake ≥ 3.20. Python 3 is only needed to build the vendored DXC
  on the first configure — there is no Python tooling in this repo.
- glTF binary (`.glb`), **Y-up**, `+Y up`/`-Z forward`. The engine's tinygltf
  has **no Draco** support — never feed it Draco-compressed glb. See
  [docs/skinned_mesh_format.md](docs/skinned_mesh_format.md).
- Tests live next to their unit as `*.test.cpp` and are auto-globbed into
  `bracken_tests`. Add a test for new logic.
- **Error handling**: new fallible engine APIs return `core::Result<T, E>`
  (`Result<void, E>` for no-payload operations) or `bool`, marked
  `[[nodiscard]]` either way. Prefer `Result` when the error payload is
  actionable data (parse/compile errors at the asset/IO boundary — hot-reload
  wants to display them); `bool` + `BRACKEN_LOG_*` is fine for boot-time
  `init()` chains where failure is fatal and the log is the diagnostic.
  Existing `bool`+log APIs stay as they are until touched — do not
  mass-convert.
- Shaders are `{name}.vert.hlsl` / `{name}.frag.hlsl` / `{name}.comp.hlsl`
  (entry points `VSMain` / `PSMain` / `CSMain`), `.hlsli` for shared
  includes. See [src/bracken/shaders/README.md](src/bracken/shaders/README.md).

## Building & testing

```bash
cmake -S . -B build -G Ninja   # first configure only (fetches deps, incl. Jolt Physics v5.2.0)
cmake --build build            # incremental — use this every time
ctest --test-dir build --output-on-failure
```

Options: `BRACKEN_BUILD_TESTS` (default ON), `BRACKEN_BUILD_BENCHMARKS` (default ON —
Google Benchmark, `*.bench.cpp` next to units, auto-globbed), `BRACKEN_BUILD_EXAMPLES`
(default ON), `BRACKEN_PLAYGROUND_SMOKE_TEST` (default OFF — registers the
playground smoke run as a ctest; needs GPU + display), `BRACKEN_ENABLE_DEV_UI`
(default ON — Dear ImGui developer overlay; OFF compiles `bracken::dev::DevUI`
to a no-op stub and skips fetching ImGui), `BRACKEN_ENABLE_TRACY`
(default OFF), `BRACKEN_WERROR` (default ON at top level, OFF when consumed via
`add_subdirectory` — warnings are applied PRIVATE so a consuming game never
inherits them; see `cmake/BrackenWarnings.cmake`).
Env: `BRACKEN_HOT_RELOAD=1` enables the file watcher + model/shader hot-reload.
`BRACKEN_WORKERS=<n>` sets the job-system worker count (unset/0 =
`hardware_concurrency()-1`, min 1; `=1` is the single-worker debug mode —
same code path, one worker).

A `CMakePresets.json` (version 6) exists at the repo root. Four trees, all
long-lived and coexisting — the no-clean-rebuild rule applies to each:

| Preset | Tree | What it is for |
|---|---|---|
| `default` | `build/` | The primary incremental tree. No `CMAKE_BUILD_TYPE` → `-O0`. |
| `tsan` | `build-tsan/` | ThreadSanitizer lane (below). Tests only. |
| `rel` | `build-rel/` | RelWithDebInfo — the **only** lane that compiles the `NDEBUG` path, where `BRACKEN_ASSERT` and the debug-only thread-tier assertions vanish. Run it before relying on an assert for correctness. |
| `bench` | `build-bench/` | Optimized benchmark lane. `build/` is `-O0` with arena poisoning and live tier asserts, so **numbers measured there are meaningless** — configure warns about this. |

### TSan lane

A **separate** build directory (`build-tsan/`) configured with
`-fsanitize=thread` — run it whenever you touch concurrency code (job system,
atomics, anything in the multithreading plan). Its *first* configure
refetches and rebuilds the deps (SDL_shadercross/DXC) from scratch — slow,
expected, one-time; after that it is incremental like any other tree. It
builds tests only (examples/benchmarks OFF), `RelWithDebInfo`.

```bash
cmake --preset tsan                          # first time only (slow: refetches deps)
cmake --build build-tsan                     # incremental
ctest --test-dir build-tsan --output-on-failure
```

Never delete or reconfigure the primary `build/` to "switch" to TSan — the
two trees coexist; the no-clean-rebuild rule applies to `build-tsan/` too
once it exists.

Integration test: `build/examples/playground/playground --smoke` (self-drives
240 frames, prints `[smoke] OK ...` with render stats, exit code 0). It also
saves + self-verifies a screenshot (`playground_smoke.png` in CWD) — **you can
Read that PNG to visually inspect what the renderer actually produced.** The
API is `Application::request_screenshot(path)` (deferred to the frame fence —
a request from sim frame i captures frame i's final swapchain image, UI
included; the capturing frame blocks on the GPU).

Benchmarks: `build/src/bracken/bracken_benchmarks` (add
`--benchmark_filter=<regex>` for a subset). Audit reports live in
`docs/audits/`.

## Consuming the engine

```cmake
add_subdirectory(path/to/engine)
target_link_libraries(my_game PRIVATE bracken)
bracken_copy_shaders(my_game)   # stages compiled SPIR-V next to the exe
```

## What you must not do
- No full clean rebuild — see [.claude/RULES.md](.claude/RULES.md).
- Don't reintroduce gameplay concepts (combat, inventory, skills, ECS) or
  server/networking code into engine headers — this repo is deliberately
  engine-only. Physics is the exception: it lives in `physics/` as engine
  infrastructure (Jolt-backed), added by owner decision 2026-07-21. Jolt
  types must stay out of public bracken headers (pimpl — link stays PRIVATE).
- Don't change `MAX_BONES` casually — it sizes CPU-side palette arrays; audit
  every `std::array<glm::mat4, MAX_BONES>` user before raising it.
- Don't edit one side of a CPU↔GPU uniform struct pair without the other.
