# Bracken Engine — User Guide

Bracken is a standalone **C++23** real-time rendering engine built on
**SDL3 GPU** (Vulkan/Metal/D3D12 backends), with build-time HLSL→SPIR-V shader
compilation, glTF model loading, skeletal animation, and a clustered-forward
renderer. It is deliberately **engine-only**: there is no game, server,
networking, or ECS layer. Rigid-body physics, however, is included as engine
infrastructure — `bracken::physics::PhysicsWorld`, a glm-typed API over
vendored Jolt Physics (see the [Physics](#physics-jolt-backed) chapter). The
engine builds as a static library
(CMake target `bracken`; all code lives under the `bracken` namespace) that
your game links against, plus a GoogleTest suite, a benchmark binary, and one
example app — `examples/playground`, a medieval-village orb-collector demo
that doubles as the integration test.

This guide is for **game developers consuming the engine**. It covers every
subsystem from the consumer's side: how to build and integrate, what to call
each frame, every public knob with its default, and the silent behaviors and
gotchas that bite in practice. For engine internals — module maps, the exact
frame/pass order, threading and lifetime contracts, and design rationale —
see `docs/technical_spec.md` (plus `docs/architecture.md` and
`docs/engine_design_notes.md`).

## Table of contents

1. [Getting Started: Building & Integrating Bracken](#getting-started-building--integrating-bracken)
2. [Application Lifecycle, Main Loop & Input](#application-lifecycle-main-loop--input)
3. [Core Utilities: Logging, Errors, Handles, Hashing, Profiling](#core-utilities-logging-errors-handles-hashing-profiling)
4. [Job System](#job-system)
5. [Asset Registry & Hot Reload](#asset-registry--hot-reload)
6. [Memory: Arena & Frame Allocators](#memory-arena--frame-allocators)
7. [GPU Abstraction Layer](#gpu-abstraction-layer)
8. [Shader Authoring & Compilation](#shader-authoring--compilation)
9. [Scene: Cameras, Culling, FramePacket & SceneRenderer](#scene-cameras-culling-framepacket--scenerenderer)
10. [Model Loading, Heightmaps & the glTF Contract](#model-loading-heightmaps--the-gltf-contract)
11. [Skeletal Animation, State Machines & IK](#skeletal-animation-state-machines--ik)
12. [Terrain & Grass Rendering](#terrain--grass-rendering)
13. [Lighting, Shadows & Ambient Occlusion](#lighting-shadows--ambient-occlusion)
14. [Model PBR Surface Shading](#model-pbr-surface-shading)
15. [Environment, Sky, Fog, Bloom & Post-Processing](#environment-sky-fog-bloom--post-processing)
16. [Particle Effects System](#particle-effects-system)
17. [Text & UI Rendering](#text--ui-rendering)
18. [Camera Controller & Camera System](#camera-controller--camera-system)
19. [Procedural Generation: Trees & Rocks](#procedural-generation-trees--rocks)
20. [Render Graph (Available, Not Yet Wired)](#render-graph-available-not-yet-wired)
21. [DevUI Developer Overlay](#devui-developer-overlay)
22. [Physics (Jolt-backed)](#physics-jolt-backed)

---

## Getting Started: Building & Integrating Bracken

Bracken is a standalone C++23 real-time rendering engine built on SDL3 GPU (Vulkan/Metal/D3D12 backends), with HLSL→SPIR-V shader compilation at build time, glTF model loading, skeletal animation, and a clustered-forward renderer. This repository contains only the engine: it builds as a static library (`bracken`), a GoogleTest suite (`bracken_tests`), a Google Benchmark binary (`bracken_benchmarks`), and one example app (`examples/playground`) that doubles as the integration test. See `README.md` for the feature overview.

### Prerequisites

- A **C++23** compiler and **CMake ≥ 3.20** (`CMakeLists.txt` line 1 requires 3.20; `CMAKE_CXX_STANDARD 23` is set with `CMAKE_CXX_STANDARD_REQUIRED ON`).
- **Ninja** is the generator used throughout this project's docs and tooling.
- **Python 3** — only needed on the *first* configure, to build the vendored DXC inside SDL_shadercross. There is no Python tooling in the repo otherwise.
- **SDL3 dev packages (optional but recommended)**: `src/bracken/CMakeLists.txt` first tries `find_package` for system **SDL3 ≥ 3.4.0**, **SDL3_ttf ≥ 3.2.2**, and **SDL3_image ≥ 3.4.0**. If all three are found at those versions, they are used; otherwise all three are fetched and built from source as static libraries. Installing the system packages avoids a from-source SDL build.

Everything else is pulled automatically by CMake `FetchContent`: glm 1.0.3, tinygltf v2.9.7 (header-only, **no Draco**), SDL_shadercross (pinned commit, bundling DXC + SPIRV-Tools), GoogleTest v1.15.2 (when tests are on), Google Benchmark v1.9.1 (when benchmarks are on), Dear ImGui v1.92.2 (when the dev UI is on), and Tracy v0.11.1 (only when Tracy is on).

### Building

```bash
cmake -S . -B build -G Ninja   # first configure only (fetches + builds deps)
cmake --build build            # incremental — use this every time after
```

The first configure fetches and builds SDL_shadercross, which bundles DXC and SPIRV-Tools — **that is the slow part of a cold build**. This leads to the project's one hard house rule (`.claude/RULES.md`):

- **Never do a full clean rebuild.** No `rm -rf build`, no `cmake --build build --clean-first`, no reconfiguring over an existing `build/`. Incremental builds only: `cmake --build build` (or `ninja -C build`).
- New `.cpp`/`.hpp`/`.test.cpp`/`.bench.cpp` files under `src/bracken/` are auto-discovered by CMake `GLOB_RECURSE ... CONFIGURE_DEPENDS` (`src/bracken/CMakeLists.txt` lines 78–80, 150, 160). Adding a file triggers a CMake re-run on the next incremental build — you never need to reconfigure by hand, and never need to rebuild from scratch.

Shaders compile as part of the normal build: every `src/bracken/shaders/*.{vert,frag}.hlsl` is cross-compiled to SPIR-V by the `shadercross` CLI into `build/src/bracken/shaders/spv/*.spv` (the `compile_shaders` target, `src/bracken/shaders/CMakeLists.txt`). Editing an `.hlsl` or `.hlsli` file and rebuilding recompiles the affected shaders — `.hlsli` includes are dependencies of every shader, so touching one recompiles all of them.

#### CMake options

All options with their defaults (defined in the root `CMakeLists.txt` except the smoke test, which lives in `examples/playground/CMakeLists.txt`):

| Option | Default | Effect |
|---|---|---|
| `BRACKEN_BUILD_TESTS` | `ON` | Build `bracken_tests` (GoogleTest v1.15.2) and call `enable_testing()`. |
| `BRACKEN_BUILD_BENCHMARKS` | `ON` | Build `bracken_benchmarks` (Google Benchmark v1.9.1). |
| `BRACKEN_BUILD_EXAMPLES` | `ON` | Build `examples/` (currently just `playground`). |
| `BRACKEN_PLAYGROUND_SMOKE_TEST` | `OFF` | Register `playground --smoke 240` as a ctest named `playground_smoke` (label `integration`, 180 s timeout). Needs a GPU + display, so off by default. |
| `BRACKEN_ENABLE_DEV_UI` | `ON` | Build the Dear ImGui developer overlay (`bracken::dev::DevUI`). `OFF` compiles `DevUI` to a no-op stub and skips fetching/linking ImGui entirely. When `ON`, the engine defines `BRACKEN_DEV_UI=1` publicly (use it to guard panel bodies that call `ImGui::` directly). |
| `BRACKEN_ENABLE_TRACY` | `OFF` | Fetch Tracy v0.11.1, link `TracyClient`, and define `BRACKEN_ENABLE_TRACY=1` publicly. OFF means zero overhead. |
| `BRACKEN_WERROR` | top-level: `ON`<br>as subproject: `OFF` | Treat warnings as errors on the four first-party targets (`bracken`, `bracken_tests`, `bracken_benchmarks`, `playground`). Warning flags are applied **PRIVATE**, so consuming a game via `add_subdirectory` never inherits them — and the default flips to `OFF` off-top-level so a new compiler release can't break your build. |

Example — a lean "shipping-style" configure:

```bash
cmake -S . -B build -G Ninja \
  -DBRACKEN_BUILD_TESTS=OFF \
  -DBRACKEN_BUILD_BENCHMARKS=OFF \
  -DBRACKEN_ENABLE_DEV_UI=OFF
```

(Do this in a *new* build directory if you already have one configured — remember the no-reconfigure rule for the existing `build/`.)

Note: `BRACKEN_PLAYGROUND_SMOKE_TEST` is a root-level option, and the root `CMakeLists.txt` calls `enable_testing()` when *either* `BRACKEN_BUILD_TESTS` or `BRACKEN_PLAYGROUND_SMOKE_TEST` is `ON` — so the smoke ctest works even in a tests-off configure (turning it on without `BRACKEN_BUILD_EXAMPLES` gets a configure-time warning instead of a silently missing test).

#### CMake presets and the TSan lane

A `CMakePresets.json` (version 6) at the repo root defines two configure presets:

- **`default`** — the primary tree: Ninja, `build/`, project-default options. It exists to reproduce the standard configuration; it is *not* an invitation to reconfigure an existing `build/` from scratch.
- **`tsan`** — the ThreadSanitizer lane: a **separate** build directory `build-tsan/` with `-fsanitize=thread` on compile and link, `RelWithDebInfo`, examples and benchmarks OFF (the lane runs `bracken_tests`). Run it whenever you touch concurrency code (the job system, atomics, anything from `docs/multithreading_plan.md`).

```bash
cmake --preset tsan                          # first configure only — slow: refetches/builds deps into build-tsan/
cmake --build build-tsan                     # incremental thereafter
ctest --test-dir build-tsan --output-on-failure
```

The TSan tree's first configure refetches SDL_shadercross/DXC just like any cold configure — expected, one-time. Keep `build-tsan/` alive afterwards; the no-clean-rebuild rule applies to it too. Never delete or reconfigure the primary `build/` to "switch" sanitizers — the two trees coexist.

### Environment variables

Three runtime environment variables, all read once at boot (before any worker thread exists):

- **`BRACKEN_WORKERS=<n>`** — checked in `Application::init_engine()` (`src/bracken/application.cpp`). Sets the job-system worker count; the pool is **always initialized** (see [Job System](#job-system)). Unset or `0`: `hardware_concurrency() - 1`, clamped to at least 1. An explicit positive value is used as-is; `BRACKEN_WORKERS=1` is the single-worker debug mode — the *same* code runs on one worker (there is no serial mode). Negative or unparsable values log a warning and fall back to the default.
- **`BRACKEN_HOT_RELOAD=1`** — checked in `Application::init_engine()` (`src/bracken/application.cpp` line 42; the check is "value starts with `1`"). Starts the global `core::asset::FileWatcher`, and `init_renderer()` then wires shader-SPIR-V hot-reload (via `gpu::PipelineRegistry::enable_hot_reload`) and model hot-reload (via `ModelManager::enable_hot_reload`) to it. Default: off — no watcher thread at all. Watcher callbacks fire only from `poll_main_thread()` on the main loop, so they may safely touch GPU state.
- **`BRACKEN_GPU_DEBUG`** — checked in `gpu::GPUDevice::init` (`src/bracken/gpu/gpu_device.cpp` line 30). Controls SDL GPU validation. Unset: validation is **on in Debug builds, off in Release** (keyed off `NDEBUG`). Set: any value enables it *except* the empty string, `0`, `false`, or `off` (case-insensitive), which disable it. GPU validation costs real CPU time per submit — leave it off for performance measurements.

There is also `--benchmark_filter` and friends for the benchmark binary (standard Google Benchmark flags), and Tracy's own configuration when `BRACKEN_ENABLE_TRACY=ON` — but those three variables above are the only engine-defined environment switches.

### Running the tests

```bash
ctest --test-dir build --output-on-failure
```

Tests live next to their unit as `*.test.cpp` anywhere under `src/bracken/` and are auto-globbed into the `bracken_tests` executable, which is registered per-test via `gtest_discover_tests` (`src/bracken/CMakeLists.txt` lines 159–169). You can also run the binary directly with GoogleTest flags:

```bash
./build/src/bracken/bracken_tests --gtest_filter='Animation*'
```

When adding new engine logic, add a `*.test.cpp` next to the unit — it is picked up automatically by an incremental build.

Every test in `bracken_tests` is headless: no GPU device, no window, no display. Two of them read files the build points them at via compile definitions — `BRACKEN_REPO_ROOT` (source-scanning tests such as the logger-discipline check) and `BRACKEN_SHADER_SPV_DIR` (`gpu/uniform_layout.test.cpp`, which reflects the compiled `.spv` files). If the path is missing, e.g. for a relocated test binary, those tests `GTEST_SKIP` rather than fail.

If you configured with `-DBRACKEN_PLAYGROUND_SMOKE_TEST=ON` (and have a GPU + display), the integration test runs under ctest too:

```bash
ctest --test-dir build -L integration --output-on-failure
```

### Running the benchmarks

```bash
./build/src/bracken/bracken_benchmarks                          # all
./build/src/bracken/bracken_benchmarks --benchmark_filter=Anim  # subset (regex)
```

Micro-benchmarks live next to their unit as `*.bench.cpp` (auto-globbed, linked against `benchmark::benchmark_main`) and cover the CPU hot paths: animation sampling/crossfade, IK, state machines, frustum culling, clustered light binning, particle updates, allocators, hashing, and procedural mesh generation. Audit reports referencing benchmark results live in `docs/audits/`.

### Profiling (Tracy)

Configure with `-DBRACKEN_ENABLE_TRACY=ON`, run any engine app, and connect the Tracy profiler GUI. The engine is instrumented end to end: frame marks, per-pass CPU zones, draw-call/triangle/particle plots, allocator memory tracking, named threads. **GPU zones are not available** — SDL3 GPU does not expose backend queues, and its per-pass timings are CPU-record times, not GPU execution times. Use RenderDoc or Nsight for real GPU timing.

### Consuming the engine from a game project

The intended integration (from `README.md`) is `add_subdirectory` + linking the `bracken` target + staging shaders:

```cmake
add_subdirectory(path/to/engine)   # or use it as a submodule
target_link_libraries(my_game PRIVATE bracken)
bracken_copy_shaders(my_game)      # stages compiled SPIR-V next to the exe
```

What each line does:

- `add_subdirectory(path/to/engine)` runs the engine's root `CMakeLists.txt`: it fetches dependencies, builds the `bracken` static library, and compiles all shaders. All engine CMake options above can be set from your project before the `add_subdirectory` call.
- Linking `bracken` is `PUBLIC`-transitive: you automatically get SDL3/SDL3_ttf/SDL3_image, SDL3_shadercross, glm, the engine include root, and the `GLM_FORCE_DEPTH_ZERO_TO_ONE` compile definition (all projection math in the engine is reversed-Z zero-to-one; this definition **propagates to your code**, which is required for your glm math to agree with the engine's).
- `bracken_copy_shaders(my_game)` (function defined in `src/bracken/CMakeLists.txt` lines 141–152) adds a custom target that removes and recopies the compiled SPIR-V directory to `$<TARGET_FILE_DIR:my_game>/shaders` after every build (deleted shaders don't linger). The engine loads shaders at runtime from `SDL_GetBasePath() + "shaders/"` — i.e. a `shaders/` directory **next to your executable** (`src/bracken/gpu/pipeline_registry.cpp` lines 36–41), so this step is mandatory; without it every pipeline fails to build at startup.

Conventions your project inherits: **Y-up** coordinates (x,z ground plane), **reversed-Z zero-to-one depth**, linear-space lighting, `.glb` models only (Y-up, `+Y up`/`-Z forward`, no Draco — the vendored tinygltf can't decode it).

The engine's public include directory is declared as `${PROJECT_SOURCE_DIR}/src` (`src/bracken/CMakeLists.txt`), which resolves to the engine's own root regardless of who the top-level project is — the three-line snippet above works verbatim from a consuming super-project, with no extra `target_include_directories` needed.

The engine's root `CMakeLists.txt` sets `BUILD_SHARED_LIBS OFF` as a plain (non-cached) variable scoped to the engine's own tree — it does not leak into a consuming super-project's cache or affect your libraries' shared/static default.

#### Subclassing `bracken::Application`

`bracken::Application` (`src/bracken/application.hpp`) owns the SDL lifecycle, the **pipelined** main loop, frame timing (nanosecond `dt`, clamped to 0.1 s), and input. You subclass it and override the hooks:

- Pure virtual: `on_init()`, `on_simulate(bracken::SimFrame&)`.
- Optional: `on_shutdown()`, `on_main_frame(bracken::MainFrame&)`, and `on_event(const SDL_Event&) -> bool` (return `true` to consume an event before `InputHandler` sees it).

There is **no `on_update`/`on_render`**: you don't render — you *describe* a frame through `sim.frame()`'s `FrameBuilder` and the engine draws it. `on_simulate` **runs as a job on the engine's worker pool, overlapped with the previous frame's rendering** — write it as straight-line single-threaded code; the engine handles the packet ring, timing, sealing, and rendering around it.

**Important:** `Application` never calls `on_init`/`on_shutdown` itself — `run()` only drives `on_simulate`/`on_main_frame`/`on_event`. Your `main()` drives the full boot sequence, and since the hooks are `protected`, the playground exposes tiny public wrappers. Minimal skeleton, adapted from `examples/playground/`:

```cpp
#include "bracken/application.hpp"   // includes sim_frame.hpp (SimFrame/MainFrame)
#include "bracken/systems/camera_controller.hpp"

class MyGame : public bracken::Application {
public:
    // Public wrappers so main() can drive the protected lifecycle hooks.
    bool start() { return on_init(); }
    void stop() { on_shutdown(); }

protected:
    bool on_init() override {
        // window w/h, title, then world_width/world_height (default 32000 x 32000)
        return init_renderer(1600, 900, "my game");
    }
    void on_shutdown() override { shutdown_renderer(); }

    // The frame. Runs as a pool job while the previous frame renders.
    void on_simulate(bracken::SimFrame& sim) override {
        const bracken::InputSnapshot& in = sim.input();  // by-value snapshot, taken at kick
        // ... game logic with sim.dt(), in.move_forward, in.was_key_just_pressed(...) ...

        auto& fb = sim.frame();                          // the ONLY way render state leaves the game
        fb.set_camera(camera().get_camera_state());
        // ... fb.draw_model / draw_skinned / add_point_light / spawn_effect ...
        // ... fb.ui().add_text / add_filled_rect ...
        // NO set_frame_timing, NO seal(), NO render_frame — the engine does all three.
    }

    // Optional: runs at the frame fence (main thread, no sim in flight) —
    // the one safe place to read live game state from the main thread.
    void on_main_frame(bracken::MainFrame& mf) override {}
};

int main() {
    MyGame game;
    if (!game.init_engine()) return 1;                       // job pool + SDL up
    if (!game.start()) { game.shutdown_engine(); return 1; } // your on_init
    game.run();                                              // pipelined loop until quit()
    game.stop();                                             // your on_shutdown
    game.shutdown_engine();                                  // SDL down
    return 0;
}
```

Boot sequence: `init_engine()` → `init_renderer(width, height, title, world_width = 32000, world_height = 32000)` (inside your `on_init`) → `run()` → `shutdown_renderer()` (inside your `on_shutdown`) → `shutdown_engine()`. `quit()` stops the loop (callable from anywhere, including inside `on_simulate`). `init_engine()` and `init_renderer()` are `[[nodiscard]] bool` — check them.

**One frame of input latency, stated plainly:** the input snapshot your `on_simulate` sees was taken when this frame was kicked, and what you build presents one pipeline slot after that — one frame between input and photons, inherent to every pipelined engine. `on_main_frame` exists for the rare truly-latency-critical main-thread need.

Side effects worth knowing (all in `src/bracken/application.cpp`):

- `init_renderer` **auto-loads** `GraphicsSettings` from `settings_file_path()` (SDL pref path + `settings.cfg`; missing file is fine — defaults apply) and applies them; `shutdown_renderer` **auto-saves** them.
- Nothing you hand the builder carries a lifetime obligation: everything is copied or packet-allocated (skinned bone matrices go into packet memory via `fb.alloc_bone_palette`; UI strings are copied at emit time). The one exception is asset data — `EffectDefinition*` passed to `spawn_effect` must outlive the spawned instances.
- Change settings at runtime by mutating `graphics_settings()` and calling **`request_graphics_settings(gs)`** — the sim-safe deferred apply; the actual GPU/SDL work (sampler rebuilds, swapchain reconfigure, fullscreen transitions) happens at the next frame fence, diffed so expensive setters only run when their inputs changed. (`apply_all_graphics_settings` still exists for main-thread-only paths: init and `on_main_frame`.)
- `request_screenshot(path)` is deferred to the frame fence: a request from sim frame i captures frame i's final swapchain image (UI included) to a PNG. The capturing frame blocks on the GPU — integration tests and debugging only, not a gameplay feature.

#### The developer overlay (DevUI)

One call after `init_renderer()` brings up the Dear ImGui overlay with built-in panels (Stats / Environment / Graphics / Camera):

```cpp
const bool dev_ui_ok = init_dev_ui();          // false in stub builds — still safe
dev_ui().set_toggle_key(SDL_SCANCODE_F10);     // show/hide key
#ifdef BRACKEN_DEV_UI
dev_ui().add_panel("My Panel", [this] {
    ImGui::Text("hello");                       // panel bodies call ImGui:: directly
});
#endif
```

With `BRACKEN_ENABLE_DEV_UI=OFF`, `init_dev_ui()` returns `false` and every `DevUI` call is a harmless no-op — host code needs **no** `#ifdef`s except around panel bodies, which include `imgui.h` directly and therefore need the `BRACKEN_DEV_UI` guard (see `examples/playground/playground.cpp`). Gotcha: `init_dev_ui()` claims the `SceneRenderer`'s **single** post-UI callback slot — a host that uses the dev UI cannot also install its own `set_post_ui_callback`.

**Panel data safety in the pipelined loop** (documented on `dev::DevUI`): panel bodies draw *during rendering, while the next frame's `on_simulate` runs on the pool*. Never read live game state from a panel — instead, **snapshot it in `on_main_frame`** (which runs with no sim in flight) into plain members the panel reads:

```cpp
void on_main_frame(bracken::MainFrame&) override {
    hud_snapshot_.player_pos = player_.pos;   // safe: no sim in flight here
    hud_snapshot_.orbs = orbs_collected_;
}
// panel body (drawn mid-render): reads hud_snapshot_ only.
```

Render-side data (`SceneRenderer` stats, `graphics_settings()` via the built-in Graphics tab — whose apply is routed through the deferred `request_graphics_settings`) is safe by construction.

### The playground: canonical integration walkthrough

`examples/playground/` is a complete medieval-village orb-collector demo and the engine's integration test. The village — a cobbled plaza ringed by six modular buildings, lanes, fences, and props — is assembled offline by `examples/playground/tools/build_medieval_assets.py` from a modular glTF asset pack (raw pack untracked under `assets/`): it merges each building/prop set into per-material `.glb`s so the engine uploads every texture set exactly once (the loader does not dedupe textures across meshes), bakes metres→world-units ×100, generates seamless terrain-layer textures, and emits `village_layout.h` (constexpr plaza/building/lane rects, chimney and door anchors) that the playground consumes for splatmap painting, grass exclusion, prop placement, push-out collision, and orb spots. Only the character, orbs, and effect meshes are **built in code**; the buildings/props come from the building pack and all flora — forest trees, bushes, ferns, rocks, flowers, and the grass tufts that replace the engine's blade grass in this scene — from a stylized nature pack (`assets/nature/`), converted 1:1 with their own materials (base color + normal + MASK alpha) as `nature_*.glb`. Placement is hand-authored, area by area, in `examples/playground/scene_layout.cpp` (`layout::build_layout()` returns village-metre placements the playground resolves to world transforms): plaza market corner, tavern/workshop/farm yards, cottage garden, campsite fire ring, lane-shoulder pebbles, meadow rock outcrops and flower drifts, and a forest ring composed from designed grove stamps with an outer sentinel band. Only the grass-tuft ground cover stays procedural (a jittered grid of tufts + clover over the meadow and forest floor). Without `assets/generated/` the playground compiles and runs a procedural-only fallback scene (generator trees/rocks + engine blade grass; the smoke test still passes). It deliberately touches every major subsystem: terrain + splatmap + grass exclusion + skybox, `.glb` loading with full PBR materials (base color / normal / metallic-roughness / ORM occlusion / emissive window glass / alpha-tested vines), skinned animation with a data-driven state machine plus foot IK and body lean, dynamic point/spot lights, particle effects (burst and continuous chimney smoke), 3D billboards, debug lines, UI text, the dev overlay, and runtime graphics-settings toggles. It also runs a **dynamic time-of-day + weather system** (`examples/playground/day_weather.*`): the sun animates around the clock (24-minute day by default) with elevation-keyed sun/ambient/fog colors — the engine sky supplies dusk scattering, stars, moon and night skyglow on its own — and an Auto weather machine drifts between clear/overcast/rain. Rain is a camera-following `spawn_volume` streak effect plus ray-cast impact splashes on terrain/props/roofs (via `physics::cast_ray`) and ripple impulses where drops land on the river. Everything is testable from the "Playground" dev panel (hour/speed/animate sliders, dawn/noon/dusk/midnight presets, weather mode + rain intensity), and two art hooks pin it for screenshots: `PLAYGROUND_TOD_HOUR=<0..24>` and `PLAYGROUND_WEATHER=clear|overcast|rain` (plain `--smoke` runs keep the legacy fixed environment for determinism).

```bash
./build/examples/playground/playground            # play it
./build/examples/playground/playground --smoke    # integration test: 240 frames, stats, exit
./build/examples/playground/playground --smoke 600  # custom frame count
```

Interactive controls (from the in-game help text): WASD move, Shift run, mouse-drag orbit, wheel zoom, LMB attack, F flashlight, F3 debug draw, F12 screenshot, Esc quit, F10 dev overlay, and graphics toggles F1 (AO cycles Off/GTAO Half/GTAO Full), F2 (bloom), F4 (volumetric fog + god rays), F5 (shadows cycles Off/PCF/PCSS), F6 (AA cycles Off/FXAA/TAA), F7 (image-based lighting on/off), F8 (local point/spot light shadows on/off).

File-by-file map:

- `examples/playground/main.cpp` — argument parsing (`--smoke [frames]`, default 240; unknown args and invalid/non-positive frame counts exit 2), the exact `init_engine → start → run → stop → shutdown_engine` sequence shown above, and post-run screenshot verification.
- `examples/playground/playground.hpp` / `.cpp` — the `PlaygroundGame : public bracken::Application` subclass; everything below.
- `examples/playground/character.hpp` / `.cpp` — code-built assets: a skinned blocky biped (11 joints named to match the engine's default foot-IK bone names, four clips authored by sampling closures), an emissive orb, a shard mesh, and a smoke-puff mesh for particles.
- `examples/playground/tools/build_medieval_assets.py` — the offline asset pipeline (see its header comment); re-run it after changing recipes/layout, then rebuild (CMake restages the outputs).

What `on_init` does, in order (`examples/playground/playground.cpp`): `init_renderer(1600, 900, "bracken playground", WORLD, WORLD)` with an 8000×8000 world → `init_dev_ui()` + F10 toggle + a game panel → build a 257×257 procedural `Heightmap` and `set_heightmap(...)` → register models — code-built ones via `models().register_model(name, build_...())` (player, orb, shard, smoke puff) and the pack `.glb`s via `load_model` (village + props + `nature_*`; `TreeGenerator`/`RockGenerator` stand-ins register only in the fallback) → build the animation state machine (Idle/Walk/Run/Attack with `speed`/`attacking` parameters, clip indices bound once via `anim_sm_.bind_clips(...)`, parameter names resolved to indices once via `anim_sm_.param_index(...)` so nothing hashes per frame) → define two `EffectDefinition`s in code (a radial pickup burst and an arc attack slash — note the definitions are *members* because they **must outlive all spawned instances**) → apply the hand-authored scene layout + ground-cover carpet (`scatter_wilds`, backed by `scene_layout.cpp`) → configure the third-person `camera()` (terrain-height callback, distance, target).

What `on_simulate` does — first the simulation: reads `sim.input()` (or the smoke self-driver), moves the player with terrain clamping, feeds `speed`/`attacking` into the state machine, runs `anim_sm_.update(anim_player_)` then `anim_player_.update(skeleton, animations, sim.dt())`, and then — as the engine intends — applies the procedural layers on top: `anim::apply_foot_ik(...)` pinning feet to terrain height and `anim::apply_body_lean(...)` from speed/turn rate. Camera follows via `set_target`/`set_target_velocity`/`update(dt)`, and `sim.set_camera_forward(...)` keeps WASD camera-relative.

Then the frame description, into `auto& fb = sim.frame()`: `set_camera`, `set_draw_skybox/ground/grass`, the player as `alloc_bone_palette` + copy of `anim_player_.bone_matrices()` into packet memory + `draw_skinned(..., pal.id)`, `draw_model` per prop and orb, an `add_point_light` per orb, `add_billboard_3d` proximity bars, optional `add_spot_light` flashlight, `spawn_effect` for queued bursts, optional `add_debug_sphere`/`add_debug_line` overlays (including a post-IK skeleton overlay), UI text via `fb.ui().add_text`/`add_filled_rect`/`add_text_wrapped`. No timing, sealing, rendering, or ring management — the engine wraps all of that around the hook.

#### Smoke mode and screenshot self-verification

`--smoke [N]` (default N = 240) enables a self-driver in `update_smoke_driver` (`examples/playground/playground.cpp`): the camera orbits at 35°/s over the plaza (pitch re-asserted at 14° down each frame; distance/height and a start yaw framing the sunlit side are set at frame 1), the player auto-runs to the nearest orb **on the plaza** (orb 0 spawns beside the player so every run proves the pickup path; lane/door orbs are ignored because the orbit camera has no building collision), attacks fire periodically (every 120 frames), the shadow-casting flashlight is forced on from frame 1 (environment A/B hooks, all applied at frame 1: `PLAYGROUND_LOCAL_SHADOWS=0` forces local shadows off for the whole run, `PLAYGROUND_AUTO_EXPOSURE=0` forces histogram auto exposure off, `PLAYGROUND_SUN_SHADOWS=0` forces the sun CSM off, `PLAYGROUND_GRADE=1` applies a deliberately non-neutral color grade, `PLAYGROUND_EXPOSURE_PROBE=1` prints a per-frame `[probe]` line with the adapted/target EV; frame 1 also resets persisted end-of-run state — `sun_intensity` back to the canonical 3.0, volumetric fog/god rays off), and heavyweight settings flip mid-run to exercise the whole post-processing matrix — TAA forced at frame 1 (whatever the persisted file stored), volumetric fog + god rays ON at frame 80 and OFF again at frame 185 (both toggle paths; the screenshot stays free of the dusk wash), local shadows off at 100 / back on at 115 (cached tiles revive), FXAA at frame 110 then back to TAA at frame 160 (history restarts cleanly), full-res GTAO at frame 140, dev overlay made visible at frame 170 (so the screenshot proves it renders), a sun-direction swing at frame 180 (restored to the scene's late-afternoon sun at 195 — each swing forces an amortized IBL prefilter refresh), PCF shadows at frame 200, a dimmed sun at frame 205 (`sun_intensity` 2.4 = 80%, so the local-light shadows are unmistakable in the screenshot). At frame `max(1, N-20)` it calls `request_screenshot("playground_smoke.png")`, and at frame N prints a summary (now incl. `village=0/1` — whether the generated assets loaded) and quits.

Two more hooks turn a smoke run into a reproducible **sky art shot**: `PLAYGROUND_SUN_ELEVATION=<degrees>` pins the sun at that elevation for the whole run (default azimuth kept) and plays day/night host — sun intensity and tint follow the elevation — while pointing a steady camera at the sun/horizon so the screenshot frames the sky; `PLAYGROUND_CLOUDS="coverage,density,scale"` additionally overrides the cloud shaping. Sun-pinned runs skip the mid-run flips that would repaint the sky (camera orbit, volumetric fog at 80, AA flips at 110/160, dev overlay at 170, sun swing at 180/195, sun dim at 205) and reset the persisted volumetric/sun-intensity knobs, so e.g. `PLAYGROUND_SUN_ELEVATION=8 playground --smoke` always produces the same golden-hour shot:

```text
[smoke] OK frames=240 fps=60 draw_calls=... triangles=... entities=... orbs_collected=... ibl_refreshes=... local_tiles=... local_lights=...p+...s auto_ev=... driver=vulkan
[smoke] screenshot OK path=playground_smoke.png 1600x900 mean_luma=... stddev=...
```

After the run, `main.cpp::verify_smoke_screenshot` loads the PNG and rejects a missing/unreadable file or a degenerate image: it requires mean luma in (2.0, 250.0) and luma standard deviation > 5.0 — so a black screen, a white screen, or any solid color fails. **A silently broken renderer that still hits 60 fps can't pass.** Exit codes: 0 success, 1 init/run/screenshot failure, 2 unknown argument.

Run it after any change that could affect rendering. You can also open `playground_smoke.png` (written to the working directory) to visually inspect exactly what the renderer produced.

### Gotchas checklist

- **Never clean-rebuild** — the SDL_shadercross/DXC build is the expensive part and lives in `build/`.
- **Missing `bracken_copy_shaders(<target>)`** — your app will fail to create every pipeline because there is no `shaders/` directory next to the executable.
- **`on_init`/`on_shutdown` are not called by `run()`** — your `main()` must drive them (see the playground's `start()`/`stop()` wrappers).
- `EffectDefinition`s referenced by `spawn_effect` must **outlive all spawned instances** — make them long-lived members, not locals.
- `BonePaletteId`s from `alloc_bone_palette` are only valid within the frame they were allocated in — allocate, write the matrices, and `draw_skinned` within the same `on_simulate`; never cache an id across frames.
- Never call `fb.seal()` or `fb.set_frame_timing(...)` from `on_simulate` — the engine stamps timing at kick and seals after its own sim-stage tail.
- Never read live game state from a DevUI panel body — panels draw while the next `on_simulate` runs; snapshot in `on_main_frame` instead.
- The smoke screenshot and interactive F12 screenshots stall the GPU for that frame — debug/test only.
- `--smoke` frame counts must be positive integers; an invalid or non-positive value after `--smoke` is rejected with an error and exit code 2.
- Don't feed Draco-compressed `.glb` to the engine — the vendored tinygltf has no Draco support.

---

## Application Lifecycle, Main Loop & Input

This chapter covers everything you touch when bootstrapping a game on Bracken: the `bracken::Application` base class (`src/bracken/application.hpp`), the main loop and frame timing, window/display management, the raw input layer (`bracken::InputHandler` / `bracken::InputState`), persisted graphics settings (`bracken::GraphicsSettings`), per-frame render statistics (`bracken::RenderStats`), and screenshot capture.

### Subclassing Application

`bracken::Application` owns the SDL lifecycle, the **pipelined** main loop, frame timing, input handling, and all core engine subsystems (render context, scene renderer, camera system, effect system, dev overlay). You subclass it and override the hooks:

```cpp
// adapted from examples/playground/playground.hpp
#include "bracken/application.hpp"   // brings in SimFrame/MainFrame (sim_frame.hpp)

class MyGame : public bracken::Application {
public:
    // Application never calls on_init/on_shutdown itself — expose them so
    // main() can drive the full boot sequence (see below).
    bool start() { return on_init(); }
    void stop()  { on_shutdown(); }

protected:
    bool on_init() override;                       // pure virtual: build your world here
    void on_shutdown() override;                   // optional (default is empty)
    void on_simulate(bracken::SimFrame& sim) override; // pure virtual: THE frame — runs as a pool job
    void on_main_frame(bracken::MainFrame& mf) override {} // optional: the frame fence, main thread
    bool on_event(const SDL_Event& event) override { return false; } // optional
};
```

The hooks:

| Hook | Required | When/where it runs | Notes |
|---|---|---|---|
| `on_init()` | yes (pure virtual) | **you** call it, after `init_engine()`; main thread | Call `init_renderer(...)` here. Return `false` to abort boot. |
| `on_simulate(SimFrame&)` | yes | once per frame, **as a job on the worker pool**, overlapped with the previous frame's rendering | Simulate with `sim.dt()`/`sim.input()`, describe the frame via `sim.frame()`. Tier sim-stage: nothing reachable from here may touch GPU/SDL/main-thread APIs (debug asserts enforce). Do not `seal()` or `set_frame_timing`. |
| `on_main_frame(MainFrame&)` | no | once per frame at the **frame fence** (main thread, no sim in flight) | The only safe place to read live game state from the main thread — snapshot values here for DevUI panels; window/screenshot work. Most games never override it. |
| `on_shutdown()` | no | **you** call it, after `run()` returns; main thread | Call `shutdown_renderer()` here. |
| `on_event(const SDL_Event&)` | no | per SDL event, during the pump (main thread, no sim in flight), before `InputHandler` sees it | Return `true` to consume the event. Touching host state here is safe. |

**Gotcha — `on_init`/`on_shutdown` are not called automatically.** `Application::run()` calls only `on_simulate`/`on_main_frame`/`on_event`. The canonical boot sequence (from `examples/playground/main.cpp`) drives the rest explicitly:

```cpp
// adapted from examples/playground/main.cpp
int main() {
    MyGame game;
    if (!game.init_engine()) {   // job-system pool, SDL_Init(VIDEO | EVENTS), main-thread binding
        return 1;
    }
    if (!game.start()) {         // your on_init(): init_renderer(...), load assets
        game.shutdown_engine();
        return 1;
    }
    game.run();                  // main loop until quit()
    game.stop();                 // your on_shutdown(): shutdown_renderer()
    game.shutdown_engine();      // FileWatcher/JobSystem shutdown, SDL_Quit()
    return 0;
}
```

`quit()` (public) requests the loop to stop; closing the window (`SDL_EVENT_QUIT`) does the same. Both cause `run()` to return after the current event-poll phase.

Side effects of `init_engine()` worth knowing (`src/bracken/application.cpp`):

- It records the calling thread as the engine main thread (`core::bind_main_thread()`); debug builds assert that swapchain acquire, `render_frame`, and `FileWatcher` entry points run on this thread.
- It initializes the always-on job-system worker pool (before SDL, so every subsystem may submit jobs), sized by `BRACKEN_WORKERS` (unset/0 = `hardware_concurrency() - 1`, min 1; `=1` is the single-worker debug mode). See [Job System](#job-system). Teardown is symmetric: the pool is joined at the top of `shutdown_renderer()`, before GPU teardown.
- If the environment variable `BRACKEN_HOT_RELOAD=1` is set, it starts the global `FileWatcher`. `init_renderer()` later wires shader-SPIR-V and model hot-reload to it. Default: off, no watcher thread.

### The pipelined main loop and frame timing

`Application::run()` (`src/bracken/application.cpp`) is the pipeline coordinator (depth 2 — Simulate(i+1) overlaps Render(i)). After a bootstrap (one event pump, then sim 0 is kicked with no overlap), each iteration i:

1. **Joins** sim job i — the main thread helps the pool while waiting. If your `on_simulate` threw, the exception is fatal-logged and **rethrown out of `run()`** on the main thread.
2. Runs the **frame fence** (no sim in flight): hot-reload callbacks (`FileWatcher`), pending `request_graphics_settings` application, camera-forward feedback, the FPS counter (`fps()` is the number of frames completed in the last whole second — updated once per second, so it reads `0.0f` for the first second, and never written while a sim is in flight, so sim code may read it), then your `on_main_frame(MainFrame&)`, then any deferred screenshot request is forwarded.
3. **Pumps events**: clears per-frame input edges (`reset_camera_deltas()`, `clear_menu_inputs()`, `clear_key_edges()`), then drains SDL events. For each event: quit stops the loop; otherwise the dev overlay sees it first (and may consume it), then your `on_event()` (return `true` to consume), then `InputHandler::process_event()`. Marks the keyboard as captured when the dev overlay wants it, then `input().post_process_events()`.
4. **Kicks sim i+1**: computes `dt` in seconds from `SDL_GetTicksNS()` (nanosecond ticks — whole-millisecond timing would quantize `dt` at high frame rates and judder animation), clamps it (`dt = std::min(dt, 0.1f)`; there is **no fixed timestep** — your `on_simulate` must handle a variable `dt`), takes the by-value `InputSnapshot` your sim will read, and submits the sim job (your `on_simulate` plus the engine's particle/clock tail, ending in the packet seal).
5. **Renders frame i** on the main thread while sim i+1 runs, then retires the packet.

Because the loop clears key/menu/zoom edges itself each frame, **do not call `InputHandler::process_events()` from inside an `Application` subclass** — that standalone helper exists for hosts that own their own loop and want the handler to do the SDL polling. Under `Application`, read `sim.input()` from `on_simulate` (the live `input()` handler is main-thread tier: `on_event`/`on_main_frame`/init only).

### Initializing the renderer and managing the window

Call from `on_init()`:

```cpp
// adapted from examples/playground/playground.cpp
bool MyGame::on_init() {
    if (!init_renderer(1600, 900, "my game", /*world_width=*/8000.0f, /*world_height=*/8000.0f)) {
        return false;
    }
    // ...
    return true;
}
```

`init_renderer(width, height, title, world_width = 32000.0f, world_height = 32000.0f)` creates the SDL window + GPU device (`render::RenderContext`), the `scene::SceneRenderer` (sized to the world extents — they bound terrain/grass), and the `systems::CameraSystem`. It also **auto-loads `GraphicsSettings`** from `settings_file_path()` and applies them (see below). It is `[[nodiscard]]` — check the return.

`shutdown_renderer()` (call from `on_shutdown()`) **auto-saves the current `GraphicsSettings`** to the same path, then tears down the dev overlay, camera, scene renderer, and GPU context in that order.

All the facade methods below dereference subsystems created by `init_renderer` — calling any of them before a successful `init_renderer()` is invalid.

#### Window mode, resolution, vsync, filtering

```cpp
set_window_mode(WindowMode::BorderlessFullscreen);
auto modes = available_resolutions(); // std::vector<Application::DisplayMode>{ int w, h; }
set_window_mode(2, /*resolution_index=*/0); // exclusive fullscreen at modes[0]
set_vsync_mode(VSyncMode::Mailbox); // triple buffer
int best = max_vsync_mode();        // highest mode the swapchain supports
set_anisotropic_filter(AnisotropicFilter::X16);
int w = screen_width();             // current drawable size
int h = screen_height();
```

- `set_vsync_mode` is clamped to `max_vsync_mode()` internally — requesting mailbox on a driver without it silently falls back (see `src/bracken/render/render_context.cpp`).
- `resolution_index` indexes `available_resolutions()`; it only matters for exclusive fullscreen (mode 2). If the index does not match a native display mode (stale settings file, different monitor), the engine logs a warning, clears any previously applied exclusive mode, and falls back to borderless fullscreen deterministically.
- `set_window_mode` is a no-op when both the mode and resolution index are unchanged (fullscreen transitions are expensive).

Prefer not to call these directly for user-facing settings — go through `apply_all_graphics_settings()` so the persisted `GraphicsSettings` stay in sync (next section).

#### Describing a frame

From `on_simulate`, describe the frame into `sim.frame()` — the engine owns the packet ring, stamps the timing at kick, seals after its sim-stage tail, and renders:

```cpp
// adapted from examples/playground/playground.cpp
void MyGame::on_simulate(bracken::SimFrame& sim) {
    // ... simulation (movement, animation, IK) using sim.dt()/sim.input() ...

    auto& fb = sim.frame();
    fb.set_camera(camera().get_camera_state());

    auto pal = fb.alloc_bone_palette(joint_count);           // packet-owned palette memory
    std::ranges::copy(anim_player_.bone_matrices().first(joint_count), pal.matrices.begin());
    fb.draw_skinned(player_model_, player_transform_, pal.id);

    fb.ui().add_text(std::format("FPS {:.0f}", fps()), 22, 44, 0.9f, 0xFFB8C4D0);
    // No seal(), no render_frame(), no ring management here.
}
```

The headline contract (details in [Scene: Cameras, Culling, FramePacket & SceneRenderer](#scene-cameras-culling-framepacket--scenerenderer), but you hit it here): **nothing you hand the builder needs to outlive the call.** Bone matrices are written into packet memory, UI strings are copied at emit time, and the renderer consumes the sealed packet by `const&` and never mutates your state. The only lifetime rule left is that `EffectDefinition*` passed to `spawn_effect` are long-lived assets.

Need parallelism? `sim.parallel_for(begin, end, body)` / `sim.parallel_for(range, body)` fan a loop out on the engine pool and join before returning (small counts run inline — write it whenever you have thousands of anything). `sim.run_async(group, fn)` submits against a host-owned `core::jobs::TaskGroup` you must join before touching results. One rule: `parallel_for` bodies must **not** emit into the builder — emit from your frame's root code, or from `run_async` jobs that each own an explicit `fb.lane(i)`.

Hosts that bypass `run()` entirely (tools, tests) can still drive the sequential Phase-2 flow themselves: `frame_packets().acquire_for_fill()` → `FrameBuilder fb(packet)` → emit + `set_frame_timing` → `fb.seal()` → `render_frame(packet)` → `frame_packets().retire(packet)`.

Other facade pass-throughs on `Application` you will use from these hooks: `models()` (the `ModelManager`; const lookups are worker-safe, loading is main-thread), `set_heightmap(...)` / `get_terrain_height(x, z)` (worker-safe), `camera()` (the `systems::CameraController` — sim-stage under `run()`), `effect_system()` (sim-stage owned; you normally just `fb.spawn_effect`), `scene_renderer()`, and `gpu_driver_name()`. They are documented in their own chapters.

### GraphicsSettings

`bracken::GraphicsSettings` (`src/bracken/contracts/graphics_settings.hpp`) is a plain struct of user-facing quality knobs, persisted as a `key=value` text file. `Application` owns one instance:

```cpp
auto& gs = graphics_settings();     // the persisted instance (loaded at init_renderer)
gs.bloom_enabled = !gs.bloom_enabled;
gs.shadow_mode = bracken::ShadowMode::PCF;
request_graphics_settings(gs);      // from on_simulate: applied at the next frame fence
// apply_all_graphics_settings(gs); // main-thread only (init / on_main_frame): applies immediately
```

`apply_all_graphics_settings()` copies the struct into `graphics_settings()`, always pushes the cheap renderer settings, and only invokes the expensive setters (sampler rebuild for anisotropy, swapchain reconfigure for vsync, fullscreen transition for window mode/resolution) when those specific fields actually changed since the last apply. It does GPU/SDL work, so it is **main-thread tier with no sim in flight** — from `on_simulate` (where the playground's F1–F8 hotkeys live, `examples/playground/playground.cpp::handle_input`) use **`request_graphics_settings(gs)`** instead: the request is stored and applied through the same diffing path at the next frame fence (last request before a fence wins; the DevUI Graphics tab's main-thread request wins over a simultaneous sim request). Own the struct from one phase at a time — mutate it in `on_simulate` or in `on_main_frame`, not both concurrently.

Settings are auto-loaded by `init_renderer()` and auto-saved by `shutdown_renderer()`. The file lives at `settings_file_path()` — the SDL pref path for org/app `"bracken"/"bracken"` plus `settings.cfg` (e.g. `~/.local/share/bracken/bracken/settings.cfg` on Linux). A missing file on first run is normal and logged at INFO level; defaults apply.

#### Fields and defaults

Enum modes are `enum class : int` with **stable underlying values** — they are the integers written to the settings file, so never renumber them.

| Field | Default | Meaning |
|---|---|---|
| `fog_enabled` | `true` | distance fog |
| `grass_enabled` / `skybox_enabled` / `trees_enabled` / `rocks_enabled` | `true` | feature toggles |
| `show_fps` / `show_debug_hud` | `false` | HUD toggles (host-interpreted) |
| `draw_distance` | `DrawDistance::D4000` | `DrawDistance` enum; world units via `get_draw_distance()` / `draw_distance_units()` |
| `frustum_culling` | `true` | CPU frustum culling toggle |
| `anisotropic_filter` | `AnisotropicFilter::X16` | `AnisotropicFilter` enum; sample count via `anisotropy_samples()` |
| `vsync_mode` | `VSyncMode::Mailbox` | `VSyncMode` enum; clamped to `max_vsync_mode()` at apply |
| `shadow_mode` | `ShadowMode::PCSS` | `Off`/`PCF`/`PCSS` (0/1/2) |
| `shadow_cascades` | `ShadowCascades::Four` | `ShadowCascades` enum; actual count via `cascade_count()` (enumerator values are indices) |
| `shadow_resolution` | `ShadowResolution::R2048` | `ShadowResolution` enum; texels via `shadow_map_texels()` |
| `shadow_distance` | `4000.0f` | CSM range in world units (also clamped by draw distance); fades out over the last 15% |
| `local_shadows` | `true` | point/spot light shadows for lights submitted with `casts_shadows = true` |
| `local_shadow_point_budget` | `4` | shadowed point lights per frame (0–8; 6 atlas tiles each) |
| `local_shadow_spot_budget` | `2` | shadowed spot lights per frame (0–16; 1 atlas tile each) |
| `local_shadow_resolution` | `LocalShadowResolution::R2048` | `LocalShadowResolution` enum; texels via `local_shadow_atlas_texels()` (8×8 tiles → 128/256/512 px per tile) |
| `ao_mode` | `AOMode::GTAO_Half` | `Off`/`GTAO_Half`/`GTAO_Full` (0/1/2) |
| `aa_mode` | `AAMode::TAA` | `Off`/`FXAA`/`TAA` (0/1/2); TAA is the fresh-settings default, persisted files keep their stored mode |
| `ibl_enabled` | `true` | image-based lighting: ambient diffuse + specular come from the prefiltered procedural sky; `false` falls back to the two-color hemisphere ambient |
| `bloom_enabled` / `bloom_strength` | `true` / `0.25f` | bloom toggle + intensity (0.0–1.0) |
| `bloom_threshold` | `1.1f` | post-exposure HDR luminance above which bloom contributes |
| `volumetric_fog` | `false` | expensive; off by default |
| `god_rays` | `false` | requires `volumetric_fog` |
| `window_mode` / `resolution_index` | `0` / `0` | see window management above |
| `exposure` | `1.0f` | linear scene multiplier before tonemap; with `auto_exposure` on it is the manual BIAS multiplied on top of the adapted exp2(EV) |
| `auto_exposure` | `true` | histogram auto exposure (compute; needs the HDR path): 256-bin log2-luminance metering of the pre-bloom scene with percentile clipping (darkest 60% / brightest 5% ignored), temporally adapted |
| `auto_exposure_min_ev` / `auto_exposure_max_ev` | `-4.0f` / `4.0f` | clamp window on the adapted EV (load clamps to \[−12, 12\] and keeps max ≥ min) |
| `auto_exposure_speed_up` / `auto_exposure_speed_down` | `1.5f` / `2.5f` | adaptation rates in EV/s (brightening / darkening; load clamps to \[0, 20\]) |
| `tonemap_mode` | `TonemapMode::ACESFitted` | `ACESFitted`/`ACESNarkowicz`/`Reinhard`/`None` (0–3) |
| `contrast` / `saturation` | `1.0f` / `1.0f` | post-tonemap grading; 1.0 = neutral. Look-style grading (white balance, ASC-CDL) lives in `EnvironmentState`, not here |
| `ambient_strength` | `0.45f` | shadow-region lift (0 = black shadows) |
| `sun_intensity` | `3.0f` | directional sun multiplier (HDR headroom) |
| `ao_strength` / `ao_radius` | `1.0f` / `1.0f` | AO darkening intensity / sample-radius multiplier |
| `fog_density` | `0.05f` | volumetric fog density multiplier |

#### save() / load() behavior

Both return `bool` and are `[[nodiscard]]`:

```cpp
bracken::GraphicsSettings s;
if (!s.load(path)) { /* no file or unreadable — defaults are untouched */ }
s.bloom_strength = 0.5f;
if (!s.save(path)) { /* directory missing / not writable */ }
```

Format and parser facts you can rely on (all covered by `src/bracken/contracts/graphics_settings.test.cpp`):

- Lines are `key=value`; empty lines and lines starting with `#` are skipped; unknown keys are silently ignored.
- Booleans: anything other than `"0"` reads as `true` (so `true`, `yes`, `42` are all true).
- Malformed numeric values leave the field at its previous value (defaults, on a fresh struct).
- `load()` clamps every enum-like/index field to its valid range at the end (`clamp_to_valid_ranges()`), so a corrupt or hand-edited file can never leave e.g. `shadow_mode=99` in memory. **Float fields are not clamped**, with one exception: the `auto_exposure_*` EV window/speeds are (they feed a GPU pass every frame). If you mutate index/enum fields from untrusted input yourself, call `clamp_to_valid_ranges()` afterwards.
- `get_draw_distance()` internally clamps the index, so even a directly-assigned out-of-range `draw_distance` cannot index out of bounds.
- **Version migration:** files saved without a `settings_version` key (or with a lower version than `GraphicsSettings::CURRENT_SETTINGS_VERSION`, currently `2`) predate the current lighting model. On load, the lighting-related knobs (`sun_intensity`, `ambient_strength`, `exposure`, `bloom_strength`, `bloom_threshold`, `shadow_mode`, `shadow_cascades`, `ao_radius`, `vsync_mode`) are reset to the new defaults; everything else (toggles, draw distance, window mode, …) is preserved.

### Raw input: InputHandler and InputState

`Application` owns one `bracken::InputHandler` (`src/bracken/input_handler.hpp`), reachable via the protected `input()` accessor. It is deliberately **raw**: mouse, keyboard, controller, and camera-relative movement axes only. Game actions (attack, sprint, skills, panel toggles) belong in a client-side bindings layer, not the engine.

**How you read it in the pipelined loop:** the live handler is main-thread tier (`on_event`/`on_main_frame`/init only). Your frame code reads **`sim.input()`** — a by-value `bracken::InputSnapshot` taken at the frame's kick. The snapshot mirrors the handler's getters as **public fields** (the port from the old getter API is mostly "delete the parens": `in.move_forward()` → `in.move_forward`, `in.get_camera_yaw()` → `in.camera_yaw`, `in.mouse_left_pressed()` → `in.mouse_left_pressed`), embeds the derived `InputState` as `in.state`, and keeps one function: `in.was_key_just_pressed(scancode)`.

#### One-shot key presses

`was_key_just_pressed(SDL_Scancode)` is true only on the frame the key first went down (auto-repeat is excluded). The main loop clears these edges every frame before polling — you never clear them yourself:

```cpp
// adapted from examples/playground/playground.cpp::handle_input (in on_simulate)
const bracken::InputSnapshot& in = sim.input();
if (in.was_key_just_pressed(SDL_SCANCODE_ESCAPE)) quit();
if (in.was_key_just_pressed(SDL_SCANCODE_F12))
    request_screenshot(std::format("screenshot_{:03d}.png", screenshot_counter_++));
```

For *held* movement keys, the snapshot carries the level state (`in.move_forward` etc.). Polling SDL directly (`SDL_GetKeyboardState`) is main-thread-only — do it in `on_event`/`on_main_frame` and mirror what you need into host state, never from `on_simulate`.

#### Mouse buttons and position

```cpp
float mx = in.mouse_x, my = in.mouse_y;      // window coordinates at kick time
if (in.mouse_left_pressed)  { /* press edge this frame */ }
if (in.mouse_left_released) { /* release edge this frame */ }
if (in.mouse_left_held)     { /* level state */ }
if (in.mouse_right_pressed) { /* press edge */ }
```

**Gotcha:** unlike key edges, mouse-button edges are **not** cleared by the main loop — the live handler's `clear_mouse_edges()` contract survives, and it is a main-thread call: clear them from `on_main_frame` (or `on_event`) once your UI/game logic has had its look. Forgetting it makes `mouse_left_pressed` stick to `true` in every snapshot.

Mouse position updates on every motion event regardless of mode, so UI hover/drag logic works even while menus are open.

#### Camera orbit (third-person action style)

Holding the **right mouse button** enters SDL relative-mouse mode and accumulates yaw/pitch; the **mouse wheel** accumulates a zoom delta; on a controller the **right stick** orbits and the **right shoulder** zooms out. Read the snapshot each frame and feed your camera:

```cpp
// adapted from examples/playground/playground.cpp::update_camera (in on_simulate)
auto& cam = camera();
cam.set_yaw(in.camera_yaw);              // degrees, wrapped to [0, 360)
cam.set_pitch(in.camera_pitch);          // degrees, clamped to [-70, 70]
cam.adjust_zoom(in.camera_zoom_delta);   // wheel: -y * 50 per notch
cam.update(sim.dt());
// REQUIRED feedback: tell the input layer which way the camera actually faces
// so WASD is camera-relative (accounts for shoulder offset). Deferred:
// applied to the InputHandler at the next frame fence.
sim.set_camera_forward(cam.get_forward().x, cam.get_forward().z);
```

- The zoom delta is reset by the main loop each frame (`reset_camera_deltas()`); yaw/pitch persist and accumulate.
- `set_camera_yaw(yaw)` on the live handler lets you force the stored yaw (e.g. after a scripted camera cut) — main-thread hooks only.
- Sensitivity and inversion are runtime-configurable on the live handler (main-thread hooks): `set_mouse_sensitivity()` (default `0.35f`), `set_controller_sensitivity()` (default `2.5f`), `set_camera_x_inverted()` / `set_camera_y_inverted()` / the `toggle_*` variants (default: not inverted). The snapshot's `camera_yaw/camera_pitch/camera_zoom_delta` already have sensitivity/inversion applied.

**Gotcha:** if you never call `sim.set_camera_forward`, movement directions are derived from the default forward `(0, -1)` (-Z), not your camera.

#### Movement

Two ways to read movement, both camera-relative and merged keyboard+controller:

```cpp
// Digital axes (WASD / arrows / left stick beyond half tilt):
if (in.move_forward)  { /* W, Up, or stick forward */ }
if (in.move_backward) { /* ... */ }
if (in.move_left)     { /* ... */ }
if (in.move_right)    { /* ... */ }

// Continuous direction (preserves analog stick partial tilt, magnitude <= 1):
const bracken::InputState& s = in.state;
float dx = s.move_dir_x;   // world X
float dy = s.move_dir_y;   // world Z (2D convention: Z becomes Y)
```

`bracken::InputState` (`src/bracken/input_state.hpp`) also carries legacy boolean movement flags (derived, kept for compatibility) and `attack_dir_x/y` (set to the camera forward each frame). Game action state (attacking, sprinting, abilities) is deliberately not in the struct — own it in your bindings layer. `input_changed()` / `reset_changed()` report whether movement/attack direction changed beyond a 0.01 threshold since the previous frame (useful for network send gating).

The free helper `bracken::normalize_move_dir(float& x, float& y)` clamps a movement vector's magnitude to 1.0 while preserving sub-unit analog magnitudes, returning the resulting length (tested in `src/bracken/input_state.test.cpp`).

#### Menus, text input, and input modes

Three orthogonal gates control what the handler produces:

- `set_game_input_enabled(bool)` — your menu-mode switch. When disabled: movement reads as zero, RMB orbit / wheel zoom stop, and the **menu navigation edges** activate: `menu_up/down/left/right_pressed()` (arrows, WASD, d-pad), `menu_select_pressed()` (Return, Space, gamepad South), and `menu_toggle_pressed()` (Escape or Start — these two fire **regardless** of the gate). Menu edges are cleared by the main loop each frame.
- `set_keyboard_captured(bool)` — mutes only the *polled* movement keys (controller unaffected). **Do not call this yourself under `Application`**: the main loop sets it every frame from whether the dev overlay wants the keyboard, so your value would be overwritten.
- `set_text_input_enabled(bool)` — chat-style text entry. While enabled, typed text accumulates and each frame's **snapshot drains it** (`in.text_input` is the delta typed since the previous snapshot — `take_text_input` semantics); `in.text_backspace_pressed` / `in.text_enter_pressed` / `in.text_escape_pressed` report edit-key edges. All other key events are swallowed so gameplay never sees them, and keyboard movement is suspended. Enabling starts SDL text input on the focused window (needed for IME); disabling clears the buffer. Toggle it from a main-thread hook.

```cpp
input().set_text_input_enabled(true);          // main-thread hook (on_event/on_main_frame)
// ... in on_simulate, each frame:
chat_line += in.text_input;                    // the drained delta
if (in.text_enter_pressed) { submit(chat_line); /* disable from a main-thread hook */ }
```

#### Controllers

One gamepad at a time: the first recognized gamepad is opened at startup or on hot-plug, and on disconnect the handler automatically falls over to the next connected one. `has_controller()` / `get_controller_name()` report status (`"No Controller"` when absent).

- Left stick = movement (radial deadzone `0.15`, rescaled so full tilt maps to 1.0), merged with keyboard.
- Right stick = camera orbit, applied every frame with `controller_sensitivity`.
- `in.controller_primary` = right trigger (deadzone `0.1`) **or** South (A/cross); `in.controller_secondary` = left trigger **or** left shoulder (LB). These are raw latched states — map them to actions yourself. `in.has_controller` reports connection.
- Right shoulder (RB) adds `+2.0` per frame to the camera zoom delta (zoom out).
- Start = menu toggle; d-pad + South navigate menus while game input is disabled.

### RenderStats

`bracken::RenderStats` (`src/bracken/contracts/render_stats.hpp`) is a plain struct of per-frame counters. Collection is **off by default** — enable it, render at least one frame, then read:

```cpp
// adapted from examples/playground/playground.cpp (smoke driver)
set_collect_render_stats(true);
// ... after rendering frames:
const auto& stats = render_stats();
std::cout << std::format("draw_calls={} triangles={} entities={}\n",
                         stats.draw_calls, stats.triangle_count, stats.entities_rendered);
```

Counters: `draw_calls`, `triangle_count`, `entities_rendered`, `entities_distance_culled`, `entities_frustum_culled`, `jobs_pending`, `instances_drawn`, `ui_vertices`, `ui_draws`, `local_shadow_tiles_rendered` (all `uint32_t`). Budgets, so the caps that actually get hit stop failing silently: `packet_bytes_used`/`packet_bytes_capacity`, `bone_palettes_used`/`bone_palettes_budget`, `lights_submitted`/`lights_binned`. Per-pass timing: `pass_times_ms` is `std::vector<PassTiming>` where `PassTiming` is `{ std::string_view name; std::optional<float> cpu_ms; std::optional<float> gpu_ms; }`.

`render_stats()` returns `std::optional<RenderStats>` — `nullopt` when collection is off, so zeros can no longer be mistaken for "the renderer drew nothing".

**Gotcha:** `cpu_ms` is the cost of **recording** the pass on the CPU, not of executing it on the GPU; a pass can record in 40 us and cost 4 ms on the device. `gpu_ms` is `nullopt` always — SDL3 exposes no timestamp query API. Use RenderDoc/Nsight for GPU numbers, and read `Application::present_wait_ms()` first to find out whether the frame is GPU-bound at all.

### What the frame threw away: frame_diagnostics

`Application::frame_diagnostics()` returns a `FrameDiagnostics` covering every silent drop path: `draws_dropped_unknown_model`, `skinned_dropped_no_skeleton`, `palettes_over_budget`, `effect_models_unresolved`, `lights_dropped`, `cluster_lights_dropped`, and `camera_missing`. Unlike `render_stats()` these are accumulated **unconditionally** — a dropped draw is not opt-in instrumentation. `diag.any()` is the one-line "is anything wrong" predicate. If you built a frame and see nothing, read this before reading anything else.

### Where the frame's time went

`Application::sim_ms()`, `render_ms()`, `fence_ms()` and `present_wait_ms()` are EMA-smoothed CPU milliseconds for the pipelined loop's phases. `sim_ms` overlaps the previous frame's `render_ms` (pipeline depth 2), so they deliberately do not sum to frame time. `present_wait_ms` is a subset of `render_ms`: the time the CPU spent blocked in swapchain acquire + submit, i.e. waiting for the GPU — the only in-engine signal separating a GPU-bound frame from a CPU-bound one.

### Screenshots: request_screenshot

```cpp
request_screenshot("screenshot_001.png");
```

Captures the **next rendered frame's final swapchain image — UI and dev overlay included** — and saves it as a PNG (alpha dropped). One pending request at a time; calling again before the next frame replaces the path.

**The capturing frame blocks on the GPU** (submit + fence wait), so this is for integration tests and debugging, not something to leave in a shipping hot path. The only other GPU→CPU readback is the auto-exposure system's tiny fenced diagnostic download (never a stall).

The playground uses it two ways: F12 saves numbered screenshots interactively, and the `--smoke N` self-driving mode saves `playground_smoke.png` near the end of the run and then verifies the image isn't black/solid (`examples/playground/main.cpp::verify_smoke_screenshot`) — a pattern worth copying for your own integration tests.

### Dev overlay hookup (from Application)

Full DevUI documentation lives in [DevUI Developer Overlay](#devui-developer-overlay); the `Application`-side contract is:

```cpp
// adapted from examples/playground/playground.cpp::on_init
const bool dev_ui_ok = init_dev_ui();   // AFTER init_renderer()
dev_ui().set_toggle_key(SDL_SCANCODE_F10); // safe even when dev_ui_ok == false
```

- `init_dev_ui()` returns `false` when the overlay is compiled out (`BRACKEN_ENABLE_DEV_UI=OFF`) or backend init failed — but `dev_ui()` is valid either way (every call becomes a harmless no-op on the stub). You only need `#ifdef BRACKEN_DEV_UI` around panel bodies that call `ImGui::` directly.
- **It claims the `SceneRenderer`'s single post-UI callback slot** — a host that calls `init_dev_ui()` must not install its own `set_post_ui_callback`.
- Calling `dev_ui()` without ever calling `init_dev_ui()` trips a debug assert.

### Common mistakes checklist

- Forgetting that `on_init`/`on_shutdown` are host-driven — nothing renders because `init_renderer` never ran.
- Ignoring the `[[nodiscard]]` results of `init_engine()`, `init_renderer()`, `GraphicsSettings::save()/load()`.
- Not calling `clear_mouse_edges()` (from a main-thread hook) — click edges latch forever in every snapshot.
- Calling `set_keyboard_captured()` yourself — the main loop overwrites it every frame.
- Never calling `sim.set_camera_forward(...)` — WASD moves relative to -Z instead of the camera.
- Mutating `graphics_settings()` fields without calling `request_graphics_settings()` (or `apply_all_graphics_settings()` from a main-thread hook) — the renderer never sees the change (until something else applies).
- Touching GPU/SDL/main-thread-tier APIs from `on_simulate` — it runs on a pool thread; debug builds assert.
- Setting `god_rays = true` without `volumetric_fog = true` — god rays require volumetric fog.
- Reading `PassTiming::cpu_ms` as GPU cost — it is the CPU's cost of recording the pass. `gpu_ms` is `nullopt`; SDL3 has no timestamp queries.
- Calling `request_screenshot` every frame — each capture stalls the GPU pipeline.

---

## Core Utilities: Logging, Errors, Handles, Hashing, Profiling

Everything in this chapter lives in `src/bracken/core/` (top-level files; the job system and asset registry under `core/jobs/` and `core/asset/` are covered elsewhere). All types are in the `bracken::core` namespace, and headers are included with a `bracken/`-rooted path:

```cpp
#include "bracken/core/logger.hpp"    // Logger, LogSink, BRACKEN_LOG_* macros
#include "bracken/core/assert.hpp"    // BRACKEN_VERIFY / BRACKEN_ASSERT / main-thread checks
#include "bracken/core/result.hpp"    // Result<T,E>, Err, Ok(), make_err()
#include "bracken/core/handle.hpp"    // Handle<Tag>, HandlePool<Tag,T>
#include "bracken/core/hash.hpp"      // fnv1a, fnv1a64, StringId, ""_sid
#include "bracken/core/profiler.hpp"  // BRACKEN_PROFILE_* Tracy macros
```

These utilities are header-mostly and have no GPU or SDL-window dependencies (the logger's default sink does write through SDL's logging API), so they are safe to use in tools, tests, and host-game code alike.

### Logging

The engine logs through a process-wide singleton `bracken::core::Logger` (`src/bracken/core/logger.hpp`). **This is the engine's only log channel** — every diagnostic the engine emits, including GPU/pipeline/shader/renderer boot failures, goes through it, so a sink you install sees all of them and `set_min_level` genuinely silences all of them. (Engine sources may not call `SDL_Log`, `std::cerr` or `std::cout`; `src/bracken/core/logger_discipline.test.cpp` enforces that.) You almost never talk to the logger directly — use the macros:

```cpp
#include "bracken/core/logger.hpp"

BRACKEN_LOG_INFO("playground", "Dev UI {}", dev_ui_ok ? "initialized" : "unavailable");
BRACKEN_LOG_WARN("render", "texture {} exceeds budget by {} bytes", name, overflow);
BRACKEN_LOG_ERROR("settings", "Failed to save settings to {}", path);
```

There is one macro per severity: `BRACKEN_LOG_TRACE`, `BRACKEN_LOG_DEBUG`, `BRACKEN_LOG_INFO`, `BRACKEN_LOG_WARN`, `BRACKEN_LOG_ERROR`, `BRACKEN_LOG_FATAL`. The first argument is a free-form **category** string (engine subsystems use one short tag per file, held in a file-local `constexpr const char* LOG_TAG`: `"app"`, `"gpu"`, `"render"`, `"scene"`, `"terrain"`, `"shadows"`, `"grass"`, `"taa"`, `"gtao"`, `"bloom"`, `"post"`, `"fog"`, `"exposure"`, `"lights"`, `"water"`, `"ui"`, `"text"`, `"effect"`, `"model"`, `"jobs"`, `"hot_reload"`, `"assert"`, ...); the rest is a `std::format` format string plus arguments — formatting errors are compile-time errors because the macros take `std::format_string`. Passing `nullptr` as the category is tolerated; sinks then receive `"general"`.

**Logging from a per-frame path.** A plain `BRACKEN_LOG_WARN` inside `render()` or an emit method is a 60 Hz denial-of-service on the log: the first line — the one that says what broke — scrolls away before anyone reads it. `bracken/core/log_throttle.hpp` provides `BRACKEN_LOG_WARN_ONCE(category, ...)`, `BRACKEN_LOG_ERROR_ONCE`, and `BRACKEN_LOG_WARN_THROTTLED(seconds, category, ...)` / `BRACKEN_LOG_ERROR_THROTTLED`. Each expands to a function-local `static core::LogThrottle` (constexpr-constructible, so it is constant-initialized — no thread-safe-static guard on a hot path) and gates the whole call, formatting included. Choose ONCE for a standing fact ("UI pipeline not available", "no camera was ever set") and THROTTLED for a condition that can come and go and whose recurrence is the information (budget exhaustion, atlas eviction) — silence after one line would read as "fixed".

`BRACKEN_LOG_FATAL` only logs — it does **not** abort. If you want "log and die", use `BRACKEN_VERIFY` (below).

#### Controlling verbosity

The default minimum level is `LogLevel::Info` — `Trace` and `Debug` messages are dropped by default. Change it at runtime:

```cpp
using bracken::core::Logger;
using bracken::core::LogLevel;

Logger::instance().set_min_level(LogLevel::Debug);  // show Debug and up
Logger::instance().set_min_level(LogLevel::Off);    // silence everything
```

Levels are ordered `Trace < Debug < Info < Warn < Error < Fatal < Off` (`bracken::core::LogLevel` in `src/bracken/core/logger.hpp`). A message passes when its level is `>=` the minimum and the minimum is not `Off`. When a message is filtered out, the `std::format` call is skipped entirely — but the **argument expressions are still evaluated**, so don't put expensive calls in log arguments on hot paths; guard with `Logger::instance().should_log(level)` if needed.

There is no environment variable or config-file knob for the log level — it is code-only.

#### Custom sinks

By default the logger has one sink, `bracken::core::StdoutSink`, which forwards to SDL's logging (`SDL_LogMessage` on `SDL_LOG_CATEGORY_APPLICATION`, category rendered as a `[category] ` prefix). You can add your own sink — e.g. to capture logs into an in-game console — by deriving from `bracken::core::LogSink`. This is exactly what the engine's own test does (`src/bracken/core/logger.test.cpp`):

```cpp
#include "bracken/core/logger.hpp"
#include <mutex>
#include <vector>

class CapturingSink : public bracken::core::LogSink {
public:
    struct Entry {
        bracken::core::LogLevel level;
        std::string category;
        std::string msg;
    };

    void write(bracken::core::LogLevel level, const char* category,
               std::string_view msg) override {
        std::lock_guard<std::mutex> lock(mutex_);
        entries.push_back(Entry{level, std::string(category ? category : ""), std::string(msg)});
    }

    std::mutex mutex_;
    std::vector<Entry> entries;
};

auto sink = std::make_shared<CapturingSink>();
bracken::core::Logger::SinkHandle handle = bracken::core::Logger::instance().add_sink(sink);
// ... later:
bracken::core::Logger::instance().remove_sink(handle);
```

Notes on sinks:

- `add_sink` takes a `std::shared_ptr<LogSink>` and returns a `Logger::SinkHandle` (a `uint32_t`) for later `remove_sink`. `clear_sinks()` removes everything **including the default stdout sink** — re-add a `StdoutSink` if you still want console output (the test fixture in `src/bracken/core/logger.test.cpp` does this in `TearDown`).
- `LogSink` is non-copyable and non-movable; always hold it by `shared_ptr`.
- The logger is thread-safe, and it deliberately calls `write()` **outside** its internal lock (so a sink may itself log without deadlocking). Consequence: your `write()` can be called from multiple threads concurrently — make it internally thread-safe, like the mutex in `CapturingSink` above.
- The internal mutex guards the **sink list only**. The level check (`should_log`) is a lock-free relaxed load of an atomic, so a filtered-out message on a worker thread costs one atomic load and never contends with the other workers.

#### Gotchas

- **Trace/Debug reach stdout.** `StdoutSink` forwards through SDL, and the `Logger` constructor opens SDL's own per-category filter fully (`SDL_SetLogPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_TRACE)`), so `set_min_level()` is the single level knob — lowering the engine level to `Trace`/`Debug` is enough to see those messages on the console.
- A sink removed with `remove_sink` may still receive messages that were already in flight on another thread at the moment of removal (the dispatch snapshot keeps it alive via `shared_ptr`).
- Log-level state is global to the process — tests that change it should restore it (see the `LoggerFixture` pattern in `src/bracken/core/logger.test.cpp`).

### Assertions

`src/bracken/core/assert.hpp` provides four macros. All failures log a `Fatal`-level message under the `"assert"` category (`file:line: assertion \`expr\` failed: message`) and then call `std::abort()`.

```cpp
#include "bracken/core/assert.hpp"

// Always active, in release builds too. Use for contract violations that
// must never ship broken.
BRACKEN_VERIFY(ptr != nullptr);
BRACKEN_VERIFY(is_ok(), "Result::value() called on error");
BRACKEN_VERIFY(count <= max, "count {} exceeds max {}", count, max);  // std::format message

// Debug-only: compiled to ((void)0) under NDEBUG.
BRACKEN_ASSERT(index < size);
BRACKEN_ASSERT(index < size, "index {} out of range", index);

// Marks impossible code paths; always active.
switch (mode) {
    case Mode::A: return handle_a();
    case Mode::B: return handle_b();
}
BRACKEN_UNREACHABLE("unhandled mode {}", static_cast<int>(mode));
```

The message arguments are optional on `BRACKEN_VERIFY`/`BRACKEN_ASSERT` and use `std::format` syntax.

**Gotcha:** under `NDEBUG`, `BRACKEN_ASSERT(cond, ...)` does **not evaluate** `cond` — never put side effects in an assert condition. `BRACKEN_VERIFY` always evaluates its condition (exactly once) and is the right choice when the check itself must run in release.

The failure message bypasses the min-level filter (a fatal contract violation is never silenced by `set_min_level(Off)`): it is dispatched to every sink **and** printed to `stderr` as a last-resort channel before the abort. With the default stdout sink attached, that means the message can appear twice on the console — by design.

#### Main-thread checks

Several engine boundaries are main-thread-only (FileWatcher entry points, swapchain acquire, `SceneRenderer::render_frame` — see `docs/architecture.md`, "Threading & lifetime contracts"). The check macro is:

```cpp
BRACKEN_ASSERT_MAIN_THREAD();  // debug builds: aborts if called off the main thread
                               // release builds: ((void)0), zero cost
```

Which thread counts as "main" is recorded by `bracken::core::bind_main_thread()`, which the engine calls for you in `Application::init_engine` (`src/bracken/application.cpp:34`) and in `core::asset::FileWatcher::init`. If nothing ever binds, the **first thread that calls** `bracken::core::is_main_thread()` captures itself as main. You only need to call `bind_main_thread()` yourself if you don't use `Application` (e.g. a custom tool) and want the checks to mean something. Rebinding is allowed — calling `bind_main_thread()` from another thread moves "main" there (exercised in `src/bracken/core/assert.test.cpp`).

You can also query directly:

```cpp
if (!bracken::core::is_main_thread()) { /* defer work to the main loop */ }
```

### Error handling: core::Result<T, E>

`bracken::core::Result<T, E>` (`src/bracken/core/result.hpp`) is a lightweight, exception-free ok-or-error type. `E` defaults to `std::string`. The whole class is `[[nodiscard]]` — ignoring a returned `Result` is a compiler warning.

#### Producing results

```cpp
#include "bracken/core/result.hpp"
using namespace bracken::core;

Result<int> parse_count(std::string_view s) {
    if (s.empty()) {
        return Err<std::string>{"empty input"};       // explicit error
    }
    if (s == "?") {
        return make_err("unknown count");             // deduces Err<std::string>
    }
    return 42;                                        // T converts implicitly to ok
}

auto r = Ok(123);   // factory; produces Result<int, std::string>
```

`Ok(value)` always produces the **default** error type (`std::string`); for a custom `E`, rely on the implicit `T` → `Result<T, E>` conversion or name the type.

For fallible operations with no payload there is a `Result<void, E>` specialization:

```cpp
Result<void> save(const std::filesystem::path& p) {
    if (!write_file(p)) {
        return make_err("write failed");
    }
    return Ok();       // or: return {};  — default-constructed Result<void> is ok
}
```

#### Consuming results

```cpp
auto r = parse_count(input);
if (!r) {                                  // explicit operator bool == is_ok()
    BRACKEN_LOG_ERROR("app", "parse failed: {}", r.error());
    return;
}
int n = r.value();

int with_default = parse_count(input).value_or(0);
```

`value()` on an error (or `error()` on an ok) is a hard fault: it trips `BRACKEN_VERIFY` and aborts — in release builds too. Always check first, or use `value_or`.

Chaining (all short-circuit on error, as tested in `src/bracken/core/result.test.cpp`):

```cpp
// map: transform the ok value, keep the error type
auto doubled = parse_count(input).map([](int v) { return v * 2; });

// and_then: chain another fallible step
auto chained = parse_count(input).and_then([](int v) -> Result<int> {
    if (v > 100) return Err<std::string>{"too large"};
    return v + 10;
});

// Result<void>::and_then takes a zero-argument callable
Result<void> r2 = save(path);
auto next = r2.and_then([]() -> Result<int> { return 7; });
```

Move the error out of an rvalue when you want to keep it: `std::string e = std::move(r).error();`.

#### When to use Result vs bool (project convention)

From `CLAUDE.md` ("Conventions"): new fallible engine APIs return `core::Result<T, E>` (`Result<void, E>` for no-payload operations) **or** `bool`, and are marked `[[nodiscard]]` either way.

- Prefer `Result` when the error payload is **actionable data** — parse/compile errors at the asset/IO boundary, where hot-reload wants to display the message.
- `bool` + `BRACKEN_LOG_*` is fine for boot-time `init()` chains where failure is fatal and the log is the diagnostic (this is the dominant existing pattern, e.g. `Application::init_engine`).
- Existing `bool`+log APIs stay as they are until touched — **do not mass-convert**.

Note that as of this writing no engine module outside `core/` returns `Result` yet — it is the convention for *new* code, exercised by `src/bracken/core/result.test.cpp`.

### Handles: Handle\<Tag\> and HandlePool

`src/bracken/core/handle.hpp` provides generational handles — the safe alternative to raw pointers/indices for resources that can be destroyed and their slots reused. `core::asset::AssetRegistry` builds on `Handle<T>`; use `HandlePool` for your own managers.

A `Handle<Tag>` is 8 bytes: `uint32_t index` + `uint32_t generation`. The `Tag` is a phantom type that makes handles of different resource kinds mutually incompatible at compile time:

```cpp
#include "bracken/core/handle.hpp"
using namespace bracken::core;

struct MeshTag {};
struct TexTag {};
using MeshHandle = Handle<MeshTag>;
using TexHandle  = Handle<TexTag>;
// MeshHandle and TexHandle are distinct types — mixing them is a compile error.
```

`HandlePool<Tag, T>` owns the values and validates handles:

```cpp
HandlePool<MeshTag, std::string> pool;

MeshHandle h = pool.acquire("hello");   // [[nodiscard]]
std::string* p = pool.get(h);           // nullptr if handle is stale/invalid
bool ok = pool.release(h);              // [[nodiscard]]; false if already released

// The stale-handle guarantee (from src/bracken/core/handle.test.cpp):
auto h1 = pool.acquire("a");
(void)pool.release(h1);
auto h2 = pool.acquire("b");            // reuses h1's slot index...
assert(pool.get(h1) == nullptr);        // ...but h1 is now stale: generation mismatch
assert(pool.get(h2) != nullptr);
```

Other members: `contains(h)`, `size()` (live count), `capacity()` (total slots ever created), `clear()`. `Handle<Tag>` has `==`/`!=`, a `std::hash` specialization (usable as an `unordered_map`/`unordered_set` key), `is_valid()`, and `Handle<Tag>::invalid()`.

Gotchas and contracts:

- **A default-constructed `Handle{}` is invalid**: `Handle{} == Handle<Tag>::invalid()` (index `0xFFFFFFFF`), and `is_valid()` returns `false` for it — safe to use as a plain "was this assigned" sentinel.
- `release()` immediately resets the stored value to `T{}` — destructors of the held resource run at release time, not at some later sweep. `T` must be default-constructible and movable.
- `clear()` destroys all values but **bumps every slot's generation** — handles retained across a `clear()` safely fail validation (`get` returns `nullptr`) instead of aliasing post-clear acquisitions. Slot storage is kept, so `capacity()` is preserved across `clear()`.
- `HandlePool` is **not thread-safe**; external synchronization is on you.
- `pool.size()` walks the free list — it is O(number of free slots), not O(1). Fine for stats, avoid per-item hot loops.

### String hashing: fnv1a and StringId

`src/bracken/core/hash.hpp` provides constexpr FNV-1a hashing and a hashed-string identifier type.

```cpp
#include "bracken/core/hash.hpp"
using namespace bracken::core;

uint32_t h32 = fnv1a("foobar");        // 0xbf9cf968 — canonical FNV-1a 32-bit
uint64_t h64 = fnv1a64("foobar");      // 0x85944171f73967e8

constexpr uint32_t ch = fnv1a("constexpr");   // fully constexpr — zero runtime cost
```

`StringId` wraps the 32-bit hash as a cheap, comparable, 4-byte identifier — ideal for name lookups where you don't want to store or compare strings:

```cpp
StringId a("render");        // implicit from string_view, constexpr
StringId b = "render"_sid;   // user-defined literal (operator""_sid in bracken::core)
assert(a == b);

std::unordered_map<StringId, int> m;   // std::hash<StringId> is provided
m[StringId("alpha")] = 1;
m["beta"_sid] = 2;
```

`StringId` supports `==`, `!=`, `<` (so it also works in ordered containers), default-constructs to `value == 0`, and can be built from a raw hash with the explicit `StringId(uint32_t)` constructor. The `""_sid` literal lives in `namespace bracken::core` — bring it in with `using namespace bracken::core;` or `using bracken::core::operator""_sid;`.

**Gotcha:** `StringId` equality is hash equality. FNV-1a-32 has no collision detection here — two different strings that collide compare equal. With typical asset/state-name vocabularies this is a non-issue, but don't use `StringId` as a security or dedup primitive. The original string is not stored; keep your own reverse map if you need to print names.

### Profiling: Tracy hooks

`src/bracken/core/profiler.hpp` wraps the Tracy client behind `BRACKEN_PROFILE_*` macros. Tracy is **off by default and zero-cost** — every macro compiles to `((void)0)`.

#### Enabling

```bash
cmake -S . -B build -G Ninja -DBRACKEN_ENABLE_TRACY=ON
cmake --build build
```

The top-level `CMakeLists.txt` fetches Tracy v0.11.1 with `TRACY_ON_DEMAND=ON` (data is only collected while a profiler is connected) and, in `src/bracken/CMakeLists.txt`, links `TracyClient` and defines `BRACKEN_ENABLE_TRACY=1` **PUBLIC** on the `bracken` target — so your game's own `BRACKEN_PROFILE_*` zones light up too, with no extra setup. Then: run your app, launch the `tracy-profiler` GUI, and click Connect (loopback by default).

#### The macros

```cpp
#include "bracken/core/profiler.hpp"

void MyGame::update(float dt) {
    BRACKEN_PROFILE_ZONE("MyGame::update");                 // scoped zone, literal name
    // ...
}

void MySystem::simulate() {
    BRACKEN_PROFILE_ZONE_COLOR("MySystem::simulate", 0x3D6E9E);   // + 0xRRGGBB color
    BRACKEN_PROFILE_PLOT("mysystem.items", static_cast<int64_t>(items_.size()));
}
```

Full set (from `src/bracken/core/profiler.hpp`):

- `BRACKEN_PROFILE_FRAME()` — frame boundary; **`Application::run` already calls this once per frame** (`src/bracken/application.cpp:118`), so hosts subclassing `Application` must not call it again.
- `BRACKEN_PROFILE_FRAME_NAMED(n)` — named frame boundary (string literal).
- `BRACKEN_PROFILE_ZONE(name)` / `BRACKEN_PROFILE_ZONE_COLOR(name, color)` — scoped zones; the name must be a **string literal**; the zone covers the enclosing scope from the macro to the closing brace.
- `BRACKEN_PROFILE_ZONE_TAG(t)` — scoped zone with a dynamic (runtime) string attached.
- `BRACKEN_PROFILE_PLOT(n, v)` — numeric sample on plot `n` (engine usage casts to `int64_t`).
- `BRACKEN_PROFILE_MESSAGE(m)` — one-shot timeline message.
- `BRACKEN_PROFILE_ALLOC(ptr, sz)` / `BRACKEN_PROFILE_FREE(ptr)` — memory tracking pairs (used by `src/bracken/memory/arena.cpp`).
- `BRACKEN_PROFILE_THREAD(name)` — name the current thread in captures (used by the job-system workers and the file watcher).

Suggested zone colors, used consistently across the engine's own passes (keep them for visual consistency in captures):

```cpp
// 0x3D6E9E update   | 0x8E44AD animation | 0x27AE60 culling/scene
// 0xE67E22 shadows  | 0xC0392B main pass | 0x16A085 post-processing
```

#### Gotchas

- **No GPU zones.** SDL3 GPU does not expose backend queues/devices, so Tracy's GPU zones cannot be wired up — Tracy shows CPU time only. For GPU timing use RenderDoc or Nsight. (Related: `RenderStats` per-pass GPU timings are stubbed zeros — see `docs/architecture.md`.)
- `BRACKEN_PROFILE_ZONE_TAG` expands to **two statements** (`ZoneScoped; ZoneText(...)`) when Tracy is enabled — never use it as the un-braced body of an `if`/`for`.
- Zone macros declare a local scoped object: placing one mid-scope profiles from that line to the end of the scope, not the whole function.

---

## Job System

Bracken ships a lock-free work-stealing thread pool (`bracken::core::jobs::JobSystem`), a fork-join counter (`TaskGroup`), a data-parallel loop helper (`parallel_for`), and a completion handle (`TaskHandle`). All of it lives in `src/bracken/core/jobs/` under the namespace `bracken::core::jobs`.

The single most important thing to know: **the pool is always on.** Bracken is a natively multithreaded engine (`docs/multithreading_plan.md` is the plan of record; Phases 1–4 have landed), and the job system is engine infrastructure, not an opt-in extra — your own `on_simulate` runs on it.

### The always-on contract

`Application::init_engine()` (`src/bracken/application.cpp`) initializes the pool unconditionally, before SDL and every other subsystem. Concretely, for you as a user:

- **You never call `init()` yourself** when subclassing `Application` — the pool is up before `on_init` runs and every engine subsystem may assume it exists.
- **Worker count** comes from the `BRACKEN_WORKERS` environment variable: unset or `0` → `hardware_concurrency() - 1` (one core left for the main thread), clamped to at least 1; an explicit positive value is used as-is. **`BRACKEN_WORKERS=1` is the debugging/bisection mode** — the same code runs on one worker. There is no serial mode and no second code path.
- **Idle costs ~zero.** Parked workers spin briefly then sleep on `std::atomic::wait` (futex-style) until a job is submitted — no periodic wakeup, no idle mutex contention. Running the pool in a scene with no parallel work is free by design.
- `Application` shuts the pool down for you, *first* in teardown (top of `shutdown_renderer()`, before any GPU teardown) so no job can outlive the subsystems it might touch.
- **Your frame code runs on the pool** (Phase 4): `run()` kicks `on_simulate(i+1)` as a job while the main thread renders frame i. Prefer `sim.parallel_for`/`sim.run_async` from inside `on_simulate` over raw `JobSystem` calls — same machinery, frame-scoped ergonomics. Nested fork-join from inside the sim job is fully supported.
- **The engine itself uses the pool every frame**: the sim-job tail updates particles in parallel, and inside `render_frame` the Prepare-stage loops — culling + instance batching, skinned resolve + palette layout, cluster light binning, grass/terrain culls — fan out as `parallel_for` chunks over your sealed packet. This is invisible to you and deterministic by contract: per-chunk partials merge in chunk-index order, so results are bit-identical at any worker count (including `BRACKEN_WORKERS=1`), and grain guards keep small scenes effectively serial.
- A convention worth knowing if you read engine code or write tests: engine-internal stage helpers take a **`JobSystem&`** parameter (production call sites pass `JobSystem::instance()`), so tests can run the same stage on caller-owned pools of different sizes and assert bit-identical outputs.

Guidance for your own code is unchanged in one respect: **parallelize loops with real counts** (thousands of anything — skinning, culling, particles). The grain guards mean small counts run inline on your thread automatically, so writing `parallel_for` is safe either way, but a plain loop over a handful of entities is still simpler and just as fast.

### Standalone (non-`Application`) use

`JobSystem` is a Meyers singleton reached via `JobSystem::instance()` (`src/bracken/core/jobs/job_system.hpp`). If you don't use `Application` (a tool, a test harness), initialize it yourself before submitting work:

```cpp
#include "bracken/core/jobs/job_system.hpp"

using bracken::core::jobs::JobSystem;

// 0 (the default) = std::thread::hardware_concurrency() - 1, minimum 1.
// Pass an explicit count to override.
JobSystem::instance().init();

// Queries:
JobSystem::instance().is_initialized(); // bool
JobSystem::instance().worker_count();   // unsigned, >= 1 once initialized
JobSystem::instance().pending_count();  // unsigned, jobs submitted but not yet finished
```

- `init()` is **idempotent**: calling it again while initialized does nothing (see `InitIsIdempotent` in `src/bracken/core/jobs/job_system.test.cpp`).
- `init()` does not read `BRACKEN_WORKERS` itself — that translation happens in `Application::init_engine()`; standalone callers pass a count (or 0) directly.

`JobSystem` is also publicly constructible for caller-owned pools (tests use this), and `parallel_for` has an overload taking an explicit pool (`parallel_for(JobSystem&, ...)`); the plain `parallel_for(begin, end, ...)` form services `JobSystem::instance()`.

### Submitting individual tasks

`submit` takes a `std::function<void()>` and returns a `TaskHandle` (`src/bracken/core/jobs/task_handle.hpp`):

```cpp
#include "bracken/core/jobs/job_system.hpp"

using namespace bracken::core::jobs;

auto& js = JobSystem::instance();

std::atomic<int> counter{0};
TaskHandle h = js.submit([&]() { counter.fetch_add(1); });
js.wait(h);                     // blocks (helping-style) until this task is done
// counter == 1, h.is_done() == true
```

`TaskHandle` is a cheap, copyable handle (a generation-checked reference to a pooled job node — no heap allocation, no `shared_ptr`):

- `valid()` — the handle refers to a task (a default-constructed handle is invalid; `wait()` on it returns immediately).
- `is_done()` — the task has finished (successfully or with an exception).
- `has_error()` — the task finished **and** threw.
- `error()` — the stored `std::exception_ptr` (null if none). Returns `nullptr` until `is_done()` is true — never a torn read of an in-flight task.

There is no return-value channel — tasks are `void()`. Communicate results through captured state (make it thread-safe; the tests use `std::atomic` or per-chunk locals).

Waiting comes in two forms:

```cpp
js.wait(h);      // wait for one task; RETHROWS the task's exception if it threw
js.wait_all();   // wait until pending_count() == 0; does NOT rethrow anything
```

Both are *helping* waits: while waiting, the calling thread drains its own local queue first, then the global queue, then steals from other workers, and finally parks on the task's done flag instead of spinning. That makes nested submit+wait (fork-join) safe **on any pool size, including a 1-worker pool** — see `WaitByHelpingDrainsNestedSubmits` and the fork-join tests in `src/bracken/core/jobs/job_system.test.cpp`.

Fire-and-forget is fine — discard the handle and later call `wait_all()`:

```cpp
for (int i = 0; i < 1024; ++i) {
    (void)js.submit([&]() { counter.fetch_add(1); });
}
js.wait_all();
```

### Exceptions

A job that throws does **not** kill the pool. The exception is captured into the task and:

- `wait(h)` rethrows it on the waiting thread.
- `wait_all()` swallows it — if you care about failures, keep the handles and check `has_error()` / `wait()` per handle.

```cpp
auto h = js.submit([]() { throw std::runtime_error("boom"); });
try {
    js.wait(h);            // throws std::runtime_error
} catch (const std::runtime_error&) {}
// h.is_done() == true, h.has_error() == true; the pool keeps working.
```

(Adapted from `ExceptionPropagatesViaWait` / `ExceptionDoesNotKillPool` in `src/bracken/core/jobs/job_system.test.cpp`.)

### TaskGroup

`TaskGroup` is the fork-join primitive for batches of related jobs: submit any number of jobs against a group, then wait for the whole group with one call — no per-handle bookkeeping.

```cpp
using namespace bracken::core::jobs;

auto& js = JobSystem::instance();
TaskGroup group;
for (auto& chunk : chunks) {
    js.submit(group, [&chunk]() { process(chunk); });
}
js.wait(group);   // helping wait: blocks until every job in the group is done
```

- Internally a `TaskGroup` is a single cache-line-padded atomic counter — cheap to create per batch; waiters park on `std::atomic::wait` rather than spinning.
- A group may be carried across frames for long-running host async work (e.g. pathfinding); you must `wait(group)` before touching the results.
- `wait(group)` is a helping wait like `wait(handle)` — the calling thread executes queued jobs while it waits, so fork-join nesting is safe on any pool size.
- If a member job throws, `wait(group)` rethrows the **first** captured exception — only after every job in the group has finished, and (with concurrent waiters) in exactly one waiter. Consuming the error clears it, so the group stays reusable.

### parallel_for

`src/bracken/core/jobs/parallel_for.hpp` provides the loop helper:

```cpp
template<typename Fn>
void parallel_for(std::size_t begin, std::size_t end, Fn&& fn, std::size_t grain = 0);
```

Your callable receives a **chunk**, not an index: `void(std::size_t chunk_begin, std::size_t chunk_end)`. You write the inner loop yourself so tight loops avoid per-element call overhead:

```cpp
#include "bracken/core/jobs/parallel_for.hpp"

using namespace bracken::core::jobs;

std::vector<int> data(10'000, 0);
parallel_for(0, data.size(), [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) data[i] = static_cast<int>(i);
});
// parallel_for blocks until every chunk has run — data is fully written here.
```

A realistic engine-side use, ticking independent `animation::AnimationPlayer` instances (from `src/bracken/animation/animation_parallel.test.cpp`):

```cpp
std::vector<bracken::animation::AnimationPlayer> players(N);
parallel_for(0, N, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
        players[i].update(skel, clips, dt);
    }
});
```

How the range is split:

- **Serial fallbacks** (chunk runs inline on your thread, no dispatch): empty range; job system not initialized or `worker_count() <= 1`; in auto-grain mode, total work below `PARALLEL_MIN_TOTAL` (**256** — below this, dispatch overhead exceeds the work); or total `<=` the chosen grain.
- **Auto grain** (`grain == 0`, the default): `max(64, total / (workers * 4))` — roughly four chunks per worker.
- **Explicit grain** bypasses the 256-element small-workload guard entirely — you are opting in to dispatch overhead because you know each element is heavy. The benchmark `BM_ParallelForSum` in `src/bracken/core/core.bench.cpp` runs with grains 1024 and 16384.

Reduction pattern — accumulate per-chunk locals, merge once per chunk (from `src/bracken/core/jobs/parallel_for.test.cpp` and the benchmark):

```cpp
std::atomic<long long> sum{0};
parallel_for(0, N, [&](std::size_t b, std::size_t e) {
    long long local = 0;
    for (std::size_t i = b; i < e; ++i) local += static_cast<long long>(i);
    sum.fetch_add(local, std::memory_order_relaxed);
});
```

`parallel_for` **always blocks until all chunks are done** before returning, so references captured into the body only need to outlive the call itself.

### Observing load

- `JobSystem::pending_count()` is the live number of submitted-but-unfinished jobs. The renderer snapshots it into `RenderStats::jobs_pending` each frame when stats collection is enabled (`src/bracken/scene/scene_renderer.cpp`).
- With Tracy enabled (`BRACKEN_ENABLE_TRACY=ON`), worker threads are named `JobWorker` and worker 0 plots `jobs.pending` about once per second while it is executing jobs (`src/bracken/core/jobs/job_system.cpp`).

### Gotchas and common mistakes

- **`parallel_for` is not free even on its serial path** — dispatch + wait + the snapshot patterns it encourages add fixed overhead. For a handful of entities, just write a plain loop.
- **Your callable is copied once per chunk** (`parallel_for` captures `fn` by value into each submitted lambda). Capture big state by reference/pointer inside your lambda, not by value.
- **Chunks must be independent.** `parallel_for` gives no ordering between chunks; shared accumulators must be atomic or per-chunk-merged as shown above.
- **Prefer non-throwing `parallel_for` bodies.** If a chunk throws, `parallel_for` **joins all chunks first** and then rethrows the first exception (in chunk order; later exceptions are dropped) — your captured references are safe, but you only see one failure.
- **`wait_all()` hides errors.** It never rethrows; use per-handle `wait()`/`has_error()` when failure matters.
- **Jobs still queued at `shutdown()` do not run** — but they are not lost silently: each discarded task completes with a `"job discarded by JobSystem::shutdown() before it ran"` error (`is_done()` true, `has_error()` true, `wait()` rethrows) and a WARN is logged with the count. Waiters wake instead of spinning forever. Still, call `wait_all()` first if you need the work to actually happen. (When you subclass `Application`, `shutdown_engine()` drives shutdown for you.)
- **Caller-owned `JobSystem` objects don't receive plain `parallel_for` work** — the `parallel_for(begin, end, ...)` form services the singleton. Use the explicit-pool overload `parallel_for(JobSystem&, begin, end, ...)` for a pool you own.
- **Don't submit before `init()`** in standalone use — initialize first. (Under `Application` this cannot arise: the pool is up before `on_init`.)

---

## Asset Registry & Hot Reload

Bracken's asset layer lives in `src/bracken/core/asset/` (namespace `bracken::core::asset`) and consists of two pieces:

- **`FileWatcher`** (`src/bracken/core/asset/file_watcher.hpp`) — a polling, cross-platform file/directory mtime watcher that runs on a background thread but fires your callbacks on the **main thread**, so callbacks may safely touch GPU state.
- **`AssetRegistry<T>`** (`src/bracken/core/asset/asset_registry.hpp`) — a generic, per-type registry with stable handles, name lookup, and opt-in hot reload built on top of `FileWatcher`.

On top of these, the engine wires two concrete hot-reload paths for you when you use `Application`: **compiled shaders** (`gpu::PipelineRegistry`) and **glTF models** (`ModelManager`). Everything is gated behind one environment variable: `BRACKEN_HOT_RELOAD=1`. Default is off — no watcher thread exists at all.

### Enabling hot reload in an Application-based app

If your app subclasses `bracken::Application`, hot reload is zero-code. Set the env var and run:

```bash
BRACKEN_HOT_RELOAD=1 ./my_game
```

What happens under the hood (`src/bracken/application.cpp`):

1. `Application::init_engine()` reads `BRACKEN_HOT_RELOAD`; if it is set and its first character is `'1'`, it calls `core::asset::FileWatcher::instance().init()` (global singleton, default poll interval **1000 ms**) and logs `Hot-reload enabled (BRACKEN_HOT_RELOAD=1)`.
2. `Application::init_renderer(...)` wires the two built-in reload targets to the singleton watcher:
   - `scene_renderer_->pipeline_registry()->enable_hot_reload(...)` — shaders.
   - `scene_renderer_->models().enable_hot_reload(...)` — models.
3. Every frame, `Application::run()` calls `FileWatcher::instance().poll_main_thread()` (only when the watcher is initialized), which drains detected changes and fires reload callbacks on the main thread.
4. `Application::shutdown_engine()` calls `FileWatcher::instance().shutdown()`.

If the env var is *not* set, steps 2–4 are harmless no-ops: `enable_hot_reload` is still called, but `watch_file`/`watch_directory` on an uninitialized watcher return `FileWatcher::k_invalid_handle` and nothing is ever watched.

Gotcha: `PipelineRegistry::enable_hot_reload` logs `PipelineRegistry hot-reload enabled at shaders/` even when the watcher was never initialized (the underlying `watch_directory` silently returned an invalid handle). Don't take that log line as proof hot reload is live — look for `FileWatcher initialized (poll 1000 ms)` earlier in the log.

### What hot-reloads (and what doesn't)

| Asset | Mechanism | Granularity |
|---|---|---|
| Shaders (compiled `.spv`) | `gpu::PipelineRegistry::enable_hot_reload` — recursive directory watch on the registry's shader path (default `"shaders/"`, relative to the working directory), filtered to `.spv` | **Coarse**: any single `.spv` change invalidates *all* cached pipelines and the shader cache, then rebuilds everything via `preload_all_pipelines()` |
| Models (`.glb` loaded via `ModelManager::load_model` — the loader is GLB-binary-only) | `ModelManager::enable_hot_reload` — one file watch per loaded model source path | Per file: re-parses the glTF, uploads fresh GPU buffers, frees the old ones, replaces the model's *contents* in place (pointer-stable) |
| Procedural models (`ModelManager::register_model`) | none — they have no source path (`model_paths_` entry is `""`); `reload_model` returns `false` for them | — |
| Textures | **not auto-wired.** `gpu::GPUTexture::reload_from_file(path, generate_mipmaps = true)` (`src/bracken/gpu/gpu_texture.hpp`) exists as a building block, but no engine system installs a watch for standalone textures. Textures embedded in a `.glb` are re-uploaded as part of a model reload | — |

#### The shader edit loop

Shaders on disk are compiled SPIR-V, not HLSL — the watcher watches `*.spv`. Your loop is:

```bash
# terminal 1: run with hot reload
BRACKEN_HOT_RELOAD=1 ./build/examples/playground/playground

# terminal 2: edit src/bracken/shaders/model.frag.hlsl, then
cmake --build build
```

The build recompiles HLSL→SPIR-V and `bracken_copy_shaders(<target>)` (defined at `src/bracken/CMakeLists.txt:141`) re-stages the fresh `.spv` files into `shaders/` next to the executable. The watcher notices, logs `Shader changed: <file> -- rebuilding pipelines`, and rebuilds every pipeline. If any pipeline fails to rebuild you get `PipelineRegistry: one or more pipelines failed to rebuild` — fix the shader and save again; the previous frame's live pipelines are released by SDL once in-flight commands finish.

#### Model reload — pointer-stable

`ModelManager::reload_model(handle)` re-parses the source file into a fresh `Model`, uploads it to the GPU, frees the old model's GPU resources, and **move-assigns the new contents into the existing `Model` object** (`src/bracken/model_loader.cpp`). Both the `ModelHandle` **and any raw `Model*` you cached from `get_model(...)` stay valid** across a reload — only the model's contents change. (Pointers *into* the model — mesh vectors, skeleton joints — are still replaced.)

If the file on disk is broken mid-edit, `reload_model` logs `ModelManager: reload failed for '<path>'` and **leaves the existing model untouched** — you never lose the working version.

`enable_hot_reload` also retrofits watches onto models that were loaded *before* it was called, so call order doesn't matter. Model watches are installed for the lifetime of the app and never unwatched.

### Using FileWatcher directly

You need `FileWatcher` directly when you want hot reload for your own file types (config files, data tables, etc.). The engine's own singleton is `FileWatcher::instance()`; you can also own a private instance (see `src/bracken/core/asset/file_watcher.test.cpp` for many working examples).

```cpp
#include "bracken/core/asset/file_watcher.hpp"

using bracken::core::asset::FileWatcher;

FileWatcher watcher;
if (!watcher.init(std::chrono::milliseconds{250})) {   // default is 1000 ms
    // false also means "already initialized"
}

// Watch a single file:
auto h = watcher.watch_file("data/tuning.json", [&](const std::filesystem::path& p) {
    // Runs on the MAIN thread (from poll_main_thread) — GPU work is safe here.
    reload_tuning(p);
});
// h == FileWatcher::k_invalid_handle (0) if init() wasn't called or callback is empty.

// Watch a directory tree, filtered by extension (".ext", exact match; "" = any file):
auto h2 = watcher.watch_directory("data/tables", ".csv",
    [&](const std::filesystem::path& p) { reload_table(p); });

// Once per frame, on the main thread:
watcher.poll_main_thread();

// Stop receiving events for a watch:
watcher.unwatch(h);

// On shutdown (also done automatically by the destructor):
watcher.shutdown();
```

Rules and behaviors you must know:

- **Main-thread contract**: `init`, `shutdown`, `watch_file`, `watch_directory`, `unwatch`, and `poll_main_thread` are main-thread-only. This is *asserted* in debug builds via `BRACKEN_ASSERT_MAIN_THREAD()` (`src/bracken/core/assert.hpp`), not just documented.
- **Callbacks fire from `poll_main_thread()`**, never from the background thread. If you never call `poll_main_thread()`, events queue up forever and nothing fires.
- **Detection is mtime-based polling.** A change is detected when the file's `last_write_time` differs from the cached value — content is never read. Touching a file (mtime-only change) fires the callback.
- **100 ms settle window**: a file whose mtime is less than 100 ms old is skipped and picked up on a later poll tick — this avoids reacting to half-written files mid-save.
- **First scan primes, doesn't fire.** Files that already exist when the watch is installed do not generate an initial event. A file *created after* the first scan completes does fire (this is how "watch a path that doesn't exist yet" works — see the `FileCreatedAfterWatchIsReported` test).
- **Deletions are never reported.** Missing files/directories are silently skipped; if the file reappears with a different mtime you get a change event.
- **Multiple watches on one file all fire** — each watch keeps its own mtime cache.
- **Extension filter is an exact, case-sensitive string match** against `path.extension()` — pass `".spv"`, not `"spv"` or `"*.spv"`.
- Watching before `init()` returns `k_invalid_handle`; a null callback returns `k_invalid_handle`; `unwatch(k_invalid_handle)` and `poll_main_thread()` on an uninitialized watcher are safe no-ops.
- `init()` returns `false` if already initialized; `shutdown()` is idempotent and re-`init()` afterwards works.

Common mistake: writing a file and expecting the callback within the same frame. Worst-case latency is one poll interval (1000 ms by default under `Application`) *plus* the 100 ms settle window *plus* however long until the next `poll_main_thread()`.

### Using AssetRegistry\<T\> for your own asset types

`AssetRegistry<T>` (`src/bracken/core/asset/asset_registry.hpp`) gives any type `T` load-by-path, name lookup, generation-checked handles, manual reload, and automatic hot reload. Note: the engine's built-in shader and model reload paths do **not** go through `AssetRegistry` — they talk to `FileWatcher` directly. `AssetRegistry` is the general-purpose tool intended for host-app asset types.

`T` must be default-constructible, and either:

- expose `bool T::reload_from(const std::filesystem::path&)` (the default load function calls it), or
- have a load function installed via `set_load_fn`.

Adapted from `src/bracken/core/asset/asset_registry.test.cpp`:

```cpp
#include "bracken/core/asset/asset_registry.hpp"

using bracken::core::asset::AssetRegistry;

struct TuningAsset {
    int value = 0;
    bool reload_from(const std::filesystem::path& p) {
        std::ifstream in(p);
        if (!in) return false;
        in >> value;
        return true;
    }
};

AssetRegistry<TuningAsset> reg;

// Load: returns Handle::invalid() (and logs an error) if the load fn fails.
auto h = reg.load("tuning", "data/tuning.txt");
if (!h.is_valid()) { /* file missing or parse failed */ }

// Access (nullptr if handle is stale/invalid):
TuningAsset* a = reg.get(h);

// Name lookup:
auto h2 = reg.find("tuning");        // Handle::invalid() if not found

// Manual reload (loads into a fresh T first; on failure the old value survives):
if (!reg.reload(h)) { /* file broken; previous value still intact */ }

// Invalidate: after unload, get(h) returns nullptr and the slot is recycled.
reg.unload(h);
```

Custom load function (e.g. when `T` can't own its parsing, or for tests):

```cpp
reg.set_load_fn([](const std::filesystem::path& p, TuningAsset& out) -> bool {
    return parse_tuning_file(p, out);   // your code; return false on failure
});
```

Hot reload — hand the registry a live watcher and it installs a file watch per loaded entry (including entries loaded *before* the call), reloading automatically when the source file changes:

```cpp
#include "bracken/core/asset/file_watcher.hpp"

// Inside an Application subclass, the engine singleton is already init'd
// when BRACKEN_HOT_RELOAD=1:
auto& fw = bracken::core::asset::FileWatcher::instance();
if (fw.is_initialized()) {
    reg.enable_hot_reload(fw);
}
```

Behaviors worth knowing:

- **Handles are stable across reload** — the slot is reused, only its `T value` is replaced (by move-assignment). The `T*` from `get()` also stays valid across a reload, because each entry lives in its own heap allocation; it becomes invalid only on `unload`.
- **Failed reloads keep the old value.** `reload` builds a fresh `T` first and only move-assigns on success; failures log a `hot_reload` warning and return `false` (`[[nodiscard]]`).
- **Failed initial loads return `Handle::invalid()`** and log an error — always check `is_valid()`.
- **Stale handles are rejected by generation**: after `unload`, an old handle's generation no longer matches and `get`/`reload` fail cleanly (nullptr / false).
- **Loading an existing name replaces it.** `load("same_name", ...)` a second time warns (`already loaded; replacing previous entry`) and unloads the old entry (including its watch) — but only after the new load succeeded, so a failed load keeps the old asset intact. Handles to the replaced entry fail generation checks; a stale `unload` on them is a safe no-op.
- The registry itself is not thread-safe; use it from the main thread, matching the `FileWatcher` contract (hot-reload callbacks arrive on the main thread via `poll_main_thread()`).
- If your app doesn't use `Application`, you must call `poll_main_thread()` yourself each frame or registry hot-reload callbacks never run.

### Configuration reference

| Knob | Default | Where |
|---|---|---|
| `BRACKEN_HOT_RELOAD` env var | unset (off) | read in `Application::init_engine` (`src/bracken/application.cpp`); only the value `1` (first char) enables |
| `FileWatcher::init(poll_interval)` | `1000 ms` | `src/bracken/core/asset/file_watcher.hpp`; `Application` uses the default |
| Settle window | `100 ms` (fixed, not configurable) | `k_settle_window_ns` in `src/bracken/core/asset/file_watcher.cpp` |
| Shader watch path | `"shaders/"` relative to the working directory | `PipelineRegistry::shader_path_` (`src/bracken/gpu/pipeline_registry.hpp`) |
| Shader watch filter | `".spv"` | `PipelineRegistry::enable_hot_reload` (`src/bracken/gpu/pipeline_registry.cpp`) |

All hot-reload activity logs under the `hot_reload` logger category — filter on that to watch the machinery work.

---

## Memory: Arena & Frame Allocators

Bracken ships two small, allocation-churn-fighting utilities in
`src/bracken/memory/`, both in namespace `bracken::memory`:

- **`Arena`** (`src/bracken/memory/arena.hpp`) — a growable bump allocator
  with optional destructor registration. Use it for burst-shaped work: build a
  big blob of temporary data, use it, throw it all away at once.
- **`FrameAllocator`** (`src/bracken/memory/frame_allocator.hpp`) — two
  `Arena`s double-buffered behind a `swap()` call. Use it when data produced
  during frame *N* must stay readable while frame *N+1* is being built
  (e.g. handing spans of transient data to the renderer).

Neither is wired into `Application` or `SceneRenderer` — the engine does not
own a global frame allocator. You create these yourself where you need them.
The project's design notes (`docs/engine_design_notes.md`) are explicit about
this: replacing capacity-stable `std::vector`s with `FrameAllocator` "yielded
the same performance and more bookkeeping. Use it where allocation churn is
actually measured, not preemptively."

### When to use which

| Situation | Tool |
|---|---|
| One-shot scratch for a build step (mesh generation, parsing) | `Arena`, local variable, dies at end of scope |
| Fixed memory budget, no heap allowed (stack buffer, embedded pool) | `Arena(void* buffer, size)` |
| Per-frame transient data that the *next* frame may still read | `FrameAllocator` + one `swap()` per frame |
| A container that reaches a stable capacity and stays there | Plain `std::vector` — measured as just as fast (`docs/audits/performance.md`) |

### Arena

#### Creating an arena

Three constructors (`src/bracken/memory/arena.hpp`):

```cpp
#include "bracken/memory/arena.hpp"

using bracken::memory::Arena;

// 1. Default: growable, chunks of Arena::k_default_chunk_size (64 KiB).
Arena a;

// 2. Growable with a custom chunk size (passing 0 falls back to the default).
//    Nothing is allocated until the first allocate() call.
Arena b(4096);

// 3. Fixed buffer: bumps inside caller-supplied memory, NEVER grows and
//    NEVER frees the buffer. allocate() returns nullptr when the budget
//    is exhausted.
alignas(16) std::byte buf[128];
Arena c(buf, sizeof(buf));
```

Arenas are move-only (copy is deleted). A moved-from arena is valid and
empty — `bytes_used()` reads 0 and it can be reused.

#### Raw allocation

```cpp
void* Arena::allocate(std::size_t size, std::size_t align);
```

- Returns a pointer aligned to `align`, or `nullptr` on failure. **No
  exceptions are thrown** — always check the result (`allocate` and
  `create<T>` are `[[nodiscard]]`).
- `align` must be a **power of two** (debug-asserted). Passing `align == 0`
  falls back to `alignof(std::max_align_t)`.
- `size == 0` succeeds and returns a valid **non-null** sentinel pointer
  that must not be dereferenced (and carries no alignment guarantee) — so
  `nullptr` uniquely signals allocation failure. No arena bytes are consumed.
- A request larger than the chunk size is fine: the arena mallocs a
  dedicated chunk big enough for it (growable arenas only).

Real usage from the engine — `src/bracken/render/terrain_renderer.cpp` sizes
one arena exactly for a one-shot mesh build, replacing two heap-grown
`std::vector`s:

```cpp
const size_t arena_bytes = vertex_count_max * (sizeof(TerrainVertex) + sizeof(float)) +
                           index_count_max * sizeof(uint32_t) + 4096;
memory::Arena scratch(arena_bytes);

auto* vertices = static_cast<TerrainVertex*>(
    scratch.allocate(vertex_count_max * sizeof(TerrainVertex), alignof(TerrainVertex)));
auto* indices = static_cast<uint32_t*>(
    scratch.allocate(index_count_max * sizeof(uint32_t), alignof(uint32_t)));
// ... fill and upload; everything freed when `scratch` goes out of scope.
```

#### Typed allocation with `create<T>()`

`create<T>(args...)` allocates, placement-news a `T`, and — only if `T` is
not trivially destructible — registers its destructor to run at `reset()` or
arena destruction (in reverse creation order):

```cpp
Arena a(1024);
auto* s = a.create<std::string>("hello");  // destructor registered
auto* i = a.create<int>(123);              // trivially destructible: no registration
if (!s) { /* out of memory — create returns nullptr, no exception */ }
a.reset();  // ~std::string() runs here; the int needs nothing
```

From `src/bracken/memory/arena.test.cpp`:

```cpp
static int alive = 0;
struct Tracked {
    Tracked() { ++alive; }
    ~Tracked() { --alive; }
};
Arena a(1024);
a.create<Tracked>();
a.create<Tracked>();
EXPECT_EQ(alive, 2);
a.reset();
EXPECT_EQ(alive, 0);   // reset() ran the destructors
```

#### `reset()` — the whole point of an arena

`reset()`:

1. Runs all registered destructors, newest first.
2. Growable arena: frees every chunk **except the first** (kept so repeated
   fill/reset cycles stop hitting `malloc`), rewinds it to empty.
3. Fixed-buffer arena: just rewinds the buffer; nothing is freed.

**Every pointer previously handed out is invalid after `reset()`.** There is
no per-object free — that is the trade you make for bump-speed allocation.

#### Introspection

```cpp
std::size_t used = a.bytes_used();      // bumped bytes (includes align padding)
std::size_t cap  = a.bytes_capacity();  // sum of all chunk sizes
bool owns        = a.owns_memory();     // false for fixed-buffer arenas
```

#### Gotchas

- **Single-owner, owner-asserted.** No internal synchronization anywhere in
  `src/bracken/memory/` — an arena belongs to one thread (the `owned` tier in
  `docs/architecture.md`), and as of the Phase-1 threading work debug builds
  assert the owning thread on use. One arena per thread (workers get
  per-worker scratch arenas), or synchronize externally.
- **Fixed-buffer exhaustion is silent** — `allocate` returns `nullptr`
  instead of growing. From `arena.test.cpp`: a 128-byte buffer serves
  two 64-byte allocations, then the next 1-byte request returns `nullptr`.
- Leftover space in earlier chunks is never revisited after the arena grows;
  only the newest chunk is bumped. Size your chunk to your workload if that
  matters.
- Destructor registration itself uses the normal heap (a `std::vector` of
  entries) — `create<T>()` on non-trivially-destructible types is not
  allocation-free bookkeeping.
- The chunk kept by `reset()` is the *first* chunk ever allocated. If your
  very first allocation was huge, that huge chunk is what gets retained.

### FrameAllocator

#### Setup and the per-frame contract

```cpp
#include "bracken/memory/frame_allocator.hpp"

using bracken::memory::FrameAllocator;

FrameAllocator fa(1 << 20);   // both internal arenas use 1 MiB chunks
                               // (default ctor: 64 KiB chunks)
```

Three members matter:

- `fa.this_frame()` → `Arena&` to allocate from for the current frame.
- `fa.last_frame()` → `Arena&` holding the previous frame's allocations
  (read-only by convention; the type doesn't stop you).
- `fa.swap()` → flip the two arenas and `reset()` the new "this frame" one.
  Call it **exactly once per frame**.

The lifetime rule, straight from `src/bracken/memory/frame_allocator.test.cpp`:
**data allocated in frame N survives through frame N+1 and dies at the start
of frame N+2** (the second `swap()` resets that arena — destructors run,
pointers invalid):

```cpp
FrameAllocator fa(1024);
int* p = fa.this_frame().create<int>(123);
*p = 555;
fa.swap();
EXPECT_EQ(*p, 555);   // still valid: it lives in last_frame() now
fa.swap();
// p is now dangling by contract — the arena was reset.
```

#### Integration with the frame loop

The engine does not call `swap()` for you. In an `Application` subclass
(`src/bracken/application.hpp` — override `on_init`/`on_simulate`), own the
allocator and flip it once at the top of the frame (the allocator is tier
`owned` — under the pipelined loop that owner is your sim code, and only
your sim code):

```cpp
class MyGame : public bracken::Application {
    bracken::memory::FrameAllocator frame_mem_{1 << 20};

    void on_simulate(bracken::SimFrame& sim) override {
        frame_mem_.swap();  // frame N-2's data is recycled here
        auto* scratch = static_cast<Particle*>(
            frame_mem_.this_frame().allocate(count * sizeof(Particle), alignof(Particle)));
        // ... scratch stays valid through the whole of this sim frame AND the next.
    }
    // on_init / on_shutdown as usual
};
```

This mirrors the benchmark harness in `src/bracken/core/core.bench.cpp`
(`BM_FrameAllocatorAllocSwap`): a batch of `this_frame().allocate(...)` calls
followed by one `swap()`.

#### Growth is reclaimed automatically

If a frame spikes and the arena grows extra chunks, the next time that arena
comes back around (two `swap()`s later) its `reset()` frees the growth chunks
and keeps only the initial one — verified by
`FrameAllocator.SwapCycleReleasesGrowthChunks` in
`src/bracken/memory/frame_allocator.test.cpp`. Memory use does not ratchet up
permanently after one bad frame.

#### Gotchas

- **Forgetting `swap()`** turns the frame allocator into a leak: the active
  arena grows forever and nothing recycles.
- **Calling `swap()` more than once per frame** halves the lifetime guarantee
  and will invalidate "last frame" data mid-frame.
- Single-owner, owner-asserted in debug — same rule as `Arena`.
- `FrameAllocator` is movable but not copyable.

### Configuration knobs

| Knob | Where | Default | Notes |
|---|---|---|---|
| `Arena::k_default_chunk_size` | `src/bracken/memory/arena.hpp` | `64 * 1024` (64 KiB) | Used by the default ctor and when `initial_chunk_size == 0` |
| `Arena(initial_chunk_size)` | ctor arg | — | Chunk granularity for growth; first chunk allocated lazily |
| `FrameAllocator(initial_chunk_size)` | ctor arg | 64 KiB via default `Arena`s | Applied to both internal arenas |
| `BRACKEN_ENABLE_TRACY` | CMake option, default `OFF` | — | When ON, arena chunk mallocs/frees appear in Tracy's memory view via `BRACKEN_PROFILE_ALLOC`/`FREE` (`src/bracken/core/profiler.hpp`) |

There are no environment variables and no runtime toggles for this module.

---

## GPU Abstraction Layer

Bracken's GPU layer (`src/bracken/gpu/`, namespace `bracken::gpu`) is a thin, ownership-safe wrapper over the SDL3 GPU API (Vulkan/Metal/D3D12 backends). It gives you:

- `GPUDevice` — device/context lifetime, the per-frame command-buffer cycle, swapchain access, and raw resource factories (`src/bracken/gpu/gpu_device.hpp`).
- `GPUBuffer` — vertex/index/storage buffers with automatic transfer-buffer handling for static and per-frame-dynamic data (`src/bracken/gpu/gpu_buffer.hpp`).
- `GPUTexture` / `GPUSampler` — file loading, raw-pixel textures, render targets, depth buffers, array textures, samplers with presets (`src/bracken/gpu/gpu_texture.hpp`).
- `GPUShader` / `ShaderManager` — pre-compiled SPIR-V loading with runtime cross-compilation via SDL_shadercross (`src/bracken/gpu/gpu_shader.hpp`).
- `PipelineConfig` / `GPUPipeline` — fluent graphics-pipeline configuration and creation (`src/bracken/gpu/gpu_pipeline.hpp`).
- `PipelineRegistry` — the engine's cache of all built-in pipelines (pre-created at init as of the Phase-1 threading work; `get_*` are read-only lookups), with shader hot-reload (`src/bracken/gpu/pipeline_registry.hpp`).
- `gpu_uniforms.hpp` — the byte-for-byte CPU mirrors of every HLSL cbuffer (`src/bracken/gpu/gpu_uniforms.hpp`).
- `gpu_types.hpp` — shared vertex formats, texture-format and blend-mode enums (`src/bracken/gpu/gpu_types.hpp`).
- `gpu_debug.hpp` — GPU debug groups/labels and `GPUTimestampPool` (real CPU record timings; GPU timings unavailable) (`src/bracken/gpu/gpu_debug.hpp`).

Most games never create a `GPUDevice` themselves: subclass `bracken::Application`, and the engine's `render::RenderContext` (`src/bracken/render/render_context.hpp`) owns the window + device. You reach the GPU layer through `context.device()` and `scene_renderer.pipeline_registry()`. The sections below still show standalone usage, because every renderer and custom pass ultimately talks to these classes.

### The device: `GPUDevice`

`GPUDevice` (`src/bracken/gpu/gpu_device.hpp`) wraps `SDL_GPUDevice`. It is **non-copyable and non-movable**: buffers, textures, and pipelines hold raw pointers back to it, so it must outlive every resource created from it. One device per application.

#### Bring-up and shutdown

```cpp
#include "bracken/gpu/gpu_device.hpp"

bracken::gpu::GPUDevice device;
if (!device.init(window)) {          // SDL_Window* you created
    // fatal - details were SDL_Log'd
}
// ... run ...
device.shutdown();                    // waits for GPU idle, releases window, destroys device
```

- `init()` creates the device requesting `SPIRV | METALLIB | DXIL` shader formats and lets SDL pick the backend, then claims the window. It returns `false` (and logs) on any failure, including double-init.
- The `prefer_low_power` parameter is honored: it is forwarded to SDL via the properties-based device creation (`SDL_PROP_GPU_DEVICE_CREATE_PREFERLOWPOWER_BOOLEAN`), preferring the energy-efficient adapter when set.
- `shutdown()` calls `SDL_WaitForGPUIdle` first, so it is safe to destroy the device after all frames are submitted — but **destroy all `GPUBuffer`/`GPUTexture`/`GPUPipeline`/`PipelineRegistry` objects before the device**, since their destructors call back into it.
- `driver_name()` returns e.g. `"vulkan"`, `"metal"`, `"d3d12"` (`"none"` before init).

#### GPU validation: `BRACKEN_GPU_DEBUG`

Validation layers cost real CPU time per submit. `init()` enables them by default in Debug builds (`#ifndef NDEBUG`) and disables them in Release. The env var `BRACKEN_GPU_DEBUG` overrides either way:

```bash
BRACKEN_GPU_DEBUG=1 ./playground     # force validation on in a release build
BRACKEN_GPU_DEBUG=0 ./playground     # force it off in a debug build ("0", "false", "off" all disable)
```

#### The frame cycle

```cpp
SDL_GPUCommandBuffer* cmd = device.begin_frame();   // acquire, nullptr on failure

uint32_t w = 0, h = 0;
SDL_GPUTexture* swapchain = device.acquire_swapchain_texture(cmd, &w, &h);
if (swapchain) {
    // begin render passes targeting `swapchain` ...
}

bracken::gpu::GPUDevice::end_frame(cmd);            // submit (fire-and-forget)
```

Gotchas:

- `acquire_swapchain_texture` **must be called from the thread that created the window** (asserted via `BRACKEN_ASSERT_MAIN_THREAD` in debug builds). It uses `SDL_WaitAndAcquireGPUSwapchainTexture`, and returns `nullptr` without error when the window is minimized — you must still submit the command buffer in that case.
- `end_frame_and_wait(cmd)` submits with a fence and **blocks until the GPU finishes**. It exists for readback paths only (the engine uses it for screenshots via `Application::request_screenshot`); never call it in the normal frame loop.

#### Swapchain parameters (vsync)

```cpp
if (device.supports_present_mode(SDL_GPU_PRESENTMODE_MAILBOX)) {
    (void)device.set_swapchain_parameters(SDL_GPU_PRESENTMODE_MAILBOX);
}
```

`set_swapchain_parameters` always uses `SDL_GPU_SWAPCHAINCOMPOSITION_SDR` and returns `false` if the mode is unsupported. In an `Application`-based game, prefer `render::RenderContext::set_vsync_mode(0|1|2)` (off / vsync / mailbox), which wraps this.

#### Raw factories

For anything the wrappers don't cover, `GPUDevice` exposes pass-through factories that null-check the device: `create_buffer`, `create_transfer_buffer`, `create_texture`, `create_sampler`, `create_graphics_pipeline`, `create_shader`, matching `release_*` calls, and `map_transfer_buffer`/`unmap_transfer_buffer`. `handle()` gives you the raw `SDL_GPUDevice*` (e.g. `SDL_CreateGPUSampler(device.handle(), &info)` as `src/bracken/render/text_renderer.cpp` does for a custom sampler).

Note: `device.width()`/`height()` return the window size in **window coordinates** (`SDL_GetWindowSize`), while `render::RenderContext::width()`/`height()` return the drawable size in **pixels** — on HiDPI displays those differ; use the context's values for anything sized to match the swapchain.

### Buffers: `GPUBuffer`

`GPUBuffer` (`src/bracken/gpu/gpu_buffer.hpp`) handles the SDL transfer-buffer dance for you. Two creation patterns:

#### Static buffers — upload once

For geometry that never changes. Data is uploaded immediately on its own internal command buffer; the temporary transfer buffer is created and released inside the call.

```cpp
#include "bracken/gpu/gpu_buffer.hpp"
using bracken::gpu::GPUBuffer;

// e.g. from model_loader.cpp:
auto vbo = GPUBuffer::create_static(device, GPUBuffer::Type::Vertex,
                                    mesh.vertices.data(),
                                    mesh.vertices.size() * sizeof(bracken::gpu::Vertex3D));
auto ibo = GPUBuffer::create_static(device, GPUBuffer::Type::Index,
                                    mesh.indices.data(),
                                    mesh.indices.size() * sizeof(uint32_t));
```

Returns `nullptr` on any failure (null data, zero size, allocation failure) — always check. `[[nodiscard]]`.

#### Dynamic buffers — update every frame

Dynamic buffers keep a persistent upload transfer buffer of the same size and update with `cycle=true` (SDL's double-buffering), so a full per-frame rewrite is safe:

```cpp
// creation (from ui_renderer.cpp):
vertex_buffer_ = GPUBuffer::create_dynamic(device, GPUBuffer::Type::Vertex,
                                           INITIAL_VERTEX_CAPACITY * sizeof(UIVertex));

// per frame — simple form, opens its own copy pass:
vertex_buffer_->update(cmd, vertex_batch_.data(), vertex_batch_.size() * sizeof(UIVertex));
```

When you update several buffers in one frame, share a single copy pass (each `update(cmd, ...)` otherwise opens and closes its own):

```cpp
SDL_GPUCopyPass* copy_pass = SDL_BeginGPUCopyPass(cmd);
chunk_storage_buffer_->update(copy_pass, visible_chunks_scratch_.data(), needed_bytes);
other_buffer_->update(copy_pass, data, bytes);
SDL_EndGPUCopyPass(copy_pass);
```

Silent-drop behaviors to know:

- Calling `update()` on a **static** buffer logs "Not a dynamic buffer" and does nothing.
- If `offset + size > buffer size`, the update is logged and **dropped** — the buffer keeps stale contents. Grow the buffer instead (creating a new `GPUBuffer` is legal even inside an open copy pass; see `src/bracken/render/grass_renderer.cpp`).

#### Zero-copy writes: `GPUBuffer::Mapping`

For large per-frame data, skip the intermediate CPU staging vector and write straight into the mapped transfer memory. This is how `SceneRenderer` packs instance data (`src/bracken/scene/scene_renderer.cpp`):

```cpp
if (bracken::gpu::GPUBuffer::Mapping mapping = instance_storage_buffer_->map()) {
    auto* dst = static_cast<bracken::gpu::InstanceData*>(mapping.data());
    for (const auto& instances : instance_batches_) {
        std::memcpy(dst, instances.data(), instances.size() * sizeof(bracken::gpu::InstanceData));
        dst += instances.size();
    }
    mapping.upload(copy_pass, required_size);   // unmaps, then records the GPU copy
}
```

Rules: `map()` works on dynamic buffers only (empty `Mapping` otherwise, also on map failure — hence the `if`). `upload()` unmaps before recording; if you never call `upload()` (early error exit), the destructor unmaps without recording a copy. The `Mapping` must not outlive the buffer.

#### Buffer types — and the `Uniform` gotcha

`GPUBuffer::Type` maps to SDL usage flags (`src/bracken/gpu/gpu_buffer.cpp`):

| Type | SDL usage |
|---|---|
| `Vertex` | `SDL_GPU_BUFFERUSAGE_VERTEX` |
| `Index` | `SDL_GPU_BUFFERUSAGE_INDEX` |
| `Uniform` | `SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ` |
| `Storage` | `SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ` |

**SDL3 GPU has no bindable uniform buffers** — small per-draw uniforms go through push-style `SDL_PushGPUVertexUniformData`/`SDL_PushGPUFragmentUniformData` (see the uniforms section below). `Type::Uniform` therefore creates a read-only *storage* buffer, identical to `Type::Storage`. The engine itself only ever uses `Vertex`, `Index`, and `Storage`.

### Textures: `GPUTexture`

`GPUTexture` (`src/bracken/gpu/gpu_texture.hpp`) covers sampled textures, render targets, depth buffers, and 2D arrays. All factories return `nullptr` on failure and are `[[nodiscard]]`.

#### Loading from files

```cpp
#include "bracken/gpu/gpu_texture.hpp"
using bracken::gpu::GPUTexture;
using bracken::gpu::TextureFormat;

// Color/albedo art: use RGBA8_SRGB so the sampler linearizes on read.
auto albedo = GPUTexture::load_from_file(device, "assets/textures/grass.png",
                                         /*generate_mipmaps=*/true,
                                         TextureFormat::RGBA8_SRGB);
// Data masks (splatmaps, heightfields exported as PNG, ...): plain RGBA8.
auto mask = GPUTexture::load_from_file(device, "assets/textures/splat.png",
                                       true, TextureFormat::RGBA8);
```

Loads any SDL_image-supported format and converts to RGBA32 before upload. Mipmaps default **on** here. The whole upload (transfer buffer, copy pass, `SDL_GenerateMipmapsForGPUTexture`, submit) happens inside the call.

Because lighting is linear-space engine-wide, picking `RGBA8` vs `RGBA8_SRGB` wrongly is a classic "everything looks washed out / too dark" bug.

#### From raw pixels

```cpp
// 1x1 dummy textures, exactly as SceneRenderer creates them:
uint32_t white_pixel = 0xFFFFFFFF;
auto white = GPUTexture::create_2d(device, 1, 1, TextureFormat::RGBA8, &white_pixel);

uint8_t flat_normal[4] = {128, 128, 255, 255};   // tangent-space (0,0,1)
auto default_normal = GPUTexture::create_2d(device, 1, 1, TextureFormat::RGBA8, flat_normal);
```

`pixels` may be `nullptr` for an uninitialized texture; `generate_mipmaps` defaults **off** for this overload. Mipmap generation forces `COLOR_TARGET` usage onto the texture (an SDL requirement).

#### Render targets and depth buffers

```cpp
// Simple color target (enum-format):
auto rt = GPUTexture::create_render_target(device, 1920, 1080, TextureFormat::RGBA8);

// Full control (this is how the HDR scene target is made, post_process_stack.cpp):
auto hdr = GPUTexture::create_2d(device, width, height,
                                 SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                                 SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);

auto depth = GPUTexture::create_depth(device, width, height);            // D32_FLOAT
auto ds    = GPUTexture::create_depth_stencil(device, width, height);    // D24_UNORM_S8_UINT
auto csm   = GPUTexture::create_depth_array(device, res, res, 4);        // D32 2D array
```

Query state with `width()`, `height()`, `format()`, `layers()`, `is_render_target()`, `is_depth()`; get the raw handle with `handle()`.

#### 2D array textures (terrain material layers)

```cpp
const void* layer_pixels[3] = {grass_rgba, rock_rgba, dirt_rgba};
auto materials = GPUTexture::create_2d_array(device, 512, 512, 3,
                                             TextureFormat::RGBA8_SRGB,
                                             layer_pixels, /*generate_mipmaps=*/true);
```

All layers upload through one transfer buffer and one command buffer; null entries in `layer_data` are skipped (that layer stays uninitialized). Any upload failure (transfer buffer, command buffer, or an unsupported-texel-size format) logs and returns `nullptr` — you never get a texture with silently missing pixel data.

#### Updating and hot-reloading

```cpp
// Re-upload pixels into an existing texture (dimensions must match exactly):
tex->upload(cmd, pixels, tex->width(), tex->height());

// Replace contents from disk (used by the model hot-reload path):
if (!tex->reload_from_file("assets/textures/grass.png")) {
    // file failed to load; the existing GPU texture is left untouched
}
```

`upload()` allocates a fresh transfer buffer per call — fine for occasional updates (font atlases), wasteful per-frame — and rejects formats with an unknown texel size (logged, returns without uploading) rather than mis-sizing the transfer buffer. `reload_from_file` preserves the texture's creation format (notably `RGBA8_SRGB` albedo art keeps its sRGB-ness) and recreates the underlying `SDL_GPUTexture`, releasing the old one — so any cached raw `handle()` from before the reload is stale.

### Samplers: `SamplerConfig` and `GPUSampler`

```cpp
#include "bracken/gpu/gpu_texture.hpp"
using bracken::gpu::GPUSampler;
using bracken::gpu::SamplerConfig;

auto linear = GPUSampler::create(device, SamplerConfig::linear_clamp());
auto aniso  = GPUSampler::create(device, SamplerConfig::anisotropic(16.0f));

SDL_GPUTextureSamplerBinding binding{ .texture = tex->handle(), .sampler = aniso->handle() };
SDL_BindGPUFragmentSamplers(render_pass, 0, &binding, 1);
```

Presets: `linear_repeat()`, `linear_clamp()`, `nearest_repeat()`, `nearest_clamp()`, `anisotropic(max_aniso = 16.0f)`. Defaults in `SamplerConfig`: linear min/mag/mip filtering, repeat addressing, `mip_lod_bias = 0`, `min_lod = 0`, `max_lod = 1000` (VK_LOD_CLAMP_NONE — leaving max_lod at 0 would **silently disable mipmaps** on Vulkan, which is why the default is 1000), anisotropy off with `max_anisotropy = 1`.

Limitation: `GPUSampler::create` always sets `enable_compare = false` — you cannot build a shadow compare sampler through this wrapper; use `SDL_CreateGPUSampler` directly (as the shadow-map code does).

### Shaders: `GPUShader` and `ShaderManager`

Shaders are authored in HLSL (`{name}.vert.hlsl` / `{name}.frag.hlsl`, entry points `VSMain`/`PSMain` — see `src/bracken/shaders/README.md`), compiled to SPIR-V at build time by CMake, and loaded at runtime as `.spv`. SDL_shadercross converts SPIR-V to the backend format (Vulkan uses it directly; Metal/D3D12 get transpiled).

The recommended path is `ShaderManager`, which caches by `(path, stage, entry_point)`:

```cpp
#include "bracken/gpu/gpu_shader.hpp"
using namespace bracken::gpu;

ShaderManager shaders(device);   // initializes SDL_shadercross

ShaderResources vs_res;
vs_res.num_uniform_buffers = 1;

ShaderResources fs_res;
fs_res.num_uniform_buffers = 3;  // lighting + shadow data + cluster params
fs_res.num_samplers = 6;
fs_res.num_storage_buffers = 3;

GPUShader* vs = shaders.get_vertex("shaders/model.vert.spv", "VSMain", vs_res);
GPUShader* fs = shaders.get_fragment("shaders/model.frag.spv", "PSMain", fs_res);
```

`ShaderResources` counts (`num_samplers`, `num_storage_textures`, `num_storage_buffers`, `num_uniform_buffers`) tell SDL how many of each resource the stage binds. They are **validated against SPIR-V reflection at load**: a mismatch logs an error (`declared resource counts do not match SPIR-V`) and the reflected counts win, so a stale hand-maintained count can't silently break rendering. Copy the counts from the shader's catalogue row in `src/bracken/gpu/pipeline_catalog.cpp` when reusing engine shaders (and fix the caller if the error fires).

Other notes:

- There is deliberately **no default `entry_point`** on the `GPUShader` load APIs (a wrong entry point fails late and confusingly on Vulkan); `get_vertex`/`get_fragment` default to `VSMain`/`PSMain`.
- The `ShaderManager` cache has no internal locking; loading/mutation is **main-thread** tier (asserted in debug as of the Phase-1 threading work) — lookups of already-loaded shaders are safe from jobs because the cache is immutable between init/hot-reload fences.
- On a cache hit, `get` verifies the requested `ShaderResources` against the counts baked at first load and logs an error if they differ (the cached shader is returned either way).
- `reload(path)` evicts one path (all its entry-point variants) so the next `get` reloads from disk; `reload_all()` clears everything. `clear_cache()` destroys all cached shaders — don't call it while pipelines built from them are being created this frame.
- One-off loading without the cache: `GPUShader::load_spirv(device, path, stage, entry, resources)` or `GPUShader::create_from_spirv(device, bytecode, ...)` for in-memory SPIR-V.
- `GPUShader::init_compiler()` is called lazily; `ShaderManager`'s constructor also calls it. Call `GPUShader::shutdown_compiler()` at application shutdown if you initialized it.

In CMake, stage the engine's compiled SPIR-V next to your executable:

```cmake
add_subdirectory(path/to/engine)
target_link_libraries(my_game PRIVATE bracken)
bracken_copy_shaders(my_game)   # copies compiled .spv into <exe dir>/shaders/
```

### Pipelines: `PipelineConfig` and `GPUPipeline`

`PipelineConfig` (`src/bracken/gpu/gpu_pipeline.hpp`) is a plain struct with fluent helpers. This is a real engine pipeline, adapted from the `Model` row of the catalogue (`src/bracken/gpu/pipeline_catalog.cpp`):

```cpp
#include "bracken/gpu/gpu_pipeline.hpp"
using namespace bracken::gpu;

PipelineConfig config;
config.vertex_shader = vs->handle();
config.fragment_shader = fs->handle();
config.with_vertex3d().opaque();                              // Vertex3D layout, no blend, depth on
config.color_format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT; // HDR scene target
config.depth_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;

auto pipeline = GPUPipeline::create(device, config);           // nullptr on failure

// in a render pass:
pipeline->bind(render_pass);
```

Builder methods (all return `*this` for chaining; later calls override earlier ones):

- Vertex layouts: `with_vertex3d()` (5 attributes), `with_skinned_vertex()` (7, incl. `UBYTE4` joints), `with_vertex2d()` (3, for UI). For custom layouts, fill `vertex_buffers`/`vertex_attributes` directly (the engine's own set is enumerated by `gpu::VertexLayout` and applied in `pipeline_catalog.cpp` — see the terrain/grid/text cases).
- Blend/depth: `opaque()`, `alpha_blended()` (depth test on, **depth write off**), `additive()` (same depth behavior; blend factors are `SRC_ALPHA, ONE` — shaders that want pure `ONE, ONE` write alpha 1.0).
- `no_depth()` disables depth test, write, **and the depth target itself**.
- Culling: `no_cull()`, `cull_front()` (default is back-face culling, CCW front).
- `with_depth_bias(constant, slope, clamp = 0.0f)`.

Defaults that matter (verified in `src/bracken/gpu/gpu_pipeline.test.cpp`):

- **Reversed-Z**: the default `depth_compare_op` is `SDL_GPU_COMPAREOP_GREATER_OR_EQUAL`, because the engine's scene projection maps near→depth 1, far→depth 0 and clears depth to 0.0. If you build a scene pipeline, keep this default. Shadow-cascade pipelines are the deliberate exception — they override back to `LESS_OR_EQUAL` (standard-Z orthographic light projections).
- `primitive_type = TRIANGLELIST`, `color_format = R8G8B8A8_UNORM`, `depth_format = D32_FLOAT`, `has_depth_target = true`, `sample_count = 1`, `enable_depth_clip = true`.

Depth-only pipelines (shadow passes): set `config.color_format = SDL_GPU_TEXTUREFORMAT_INVALID` — `GPUPipeline::create` then emits zero color targets.

### The engine's pipeline cache: `PipelineRegistry`

`PipelineRegistry` (`src/bracken/gpu/pipeline_registry.hpp`) creates and caches one `GPUPipeline` per `PipelineType` — 56 types covering scene, UI, shadow (including the local-shadow tile clear), water, post-processing (including the GTAO/GTAOTemporal/GTAOUpsample/BlurAO chain, the TAA resolve/sharpen pair, and the IBL prefilter passes), glTF material variants, and the depth-prepass/velocity + post-prepass depth-EQUAL variants — plus one `GPUComputePipeline` per `ComputePipelineType` (3: the auto-exposure histogram/adapt pair and the water wave sim) with the same lazy-create/preload/hot-reload lifecycle, via `get_compute_pipeline()`. What each of those pipelines *is* comes from its catalogue row (`gpu/pipeline_catalog.hpp`); the registry is the cache and lifecycle only. `SceneRenderer` owns one; get it via `scene_renderer.pipeline_registry()`.

```cpp
#include "bracken/gpu/pipeline_registry.hpp"
using namespace bracken::gpu;

PipelineRegistry registry;
if (!registry.init(device)) { /* fatal */ }

// Lazy: created on first access, cached after.
GPUPipeline* model = registry.get_pipeline(PipelineType::Model);
GPUPipeline* ui    = registry.get_pipeline(PipelineType::UI);

// During loading, avoid first-use hitches:
if (!registry.preload_all_pipelines()) { /* at least one failed - logged */ }
```

Things to know:

- `init()` resolves the shader directory as `SDL_GetBasePath() + "shaders/"` (i.e. next to the executable — which is what `bracken_copy_shaders` populates), falling back to relative `"shaders/"`.
- `get_pipeline` returns `nullptr` if creation fails (missing `.spv`, bad shader); every renderer must handle that.
- **Formats** (`src/bracken/gpu/pipeline_registry.hpp` members): scene pipelines render to the HDR offscreen target `R16G16B16A16_FLOAT` with `D32_FLOAT` depth; AO pipelines to `R8_UNORM`; only `UI`, `Text`, `Composite`, and `Fxaa` target the swapchain format (`R8G8B8A8_UNORM` default). `set_swapchain_format()` invalidates only those four if pipelines already exist.
- `invalidate_all()` clears every pipeline **and** the shader cache; pipelines rebuild lazily.
- glTF material variants are their own types: `ModelDoubleSided`, `ModelBlend`, `ModelBlendDoubleSided` (and skinned/instanced equivalents). `Blend` = alpha blending with depth write off (drawn back-to-front by `SceneRenderer`); `DoubleSided` = culling off.

#### Shader hot-reload

```cpp
// Application does this when BRACKEN_HOT_RELOAD=1 (application.cpp):
registry.enable_hot_reload(bracken::core::asset::FileWatcher::instance());
```

Any `.spv` change under the shader directory invalidates **all** pipelines and preloads them again (coarse but correct; SDL releases the old pipelines once in-flight frames finish). Rebuild the shader (e.g. `cmake --build build`) and the running app picks it up. `reload_all_pipelines()` forces the same pass manually.

### Uniforms: the byte-for-byte CPU-to-GPU contract

SDL3 GPU pushes uniform data per draw/pass rather than binding uniform buffers:

```cpp
#include "bracken/gpu/gpu_uniforms.hpp"

bracken::gpu::ModelTransformUniforms transform_uniforms;
transform_uniforms.model = model_matrix;
transform_uniforms.view = view;
transform_uniforms.projection = projection;
// slot 0 here = cbuffer b0 in the vertex stage's uniform set:
SDL_PushGPUVertexUniformData(gpu_cmd, 0, &transform_uniforms, sizeof(transform_uniforms));

bracken::gpu::ModelLightingUniforms lighting_uniforms;
// ... fill ...
SDL_PushGPUFragmentUniformData(gpu_cmd, 0, &lighting_uniforms, sizeof(lighting_uniforms));
```

Every HLSL cbuffer in `src/bracken/shaders/` has exactly one `alignas(16)` mirror struct in `src/bracken/gpu/gpu_uniforms.hpp`, annotated with its `(set, binding)`. The two sides must match **byte-for-byte** — this is the single most common way to silently break rendering in this engine. The binding order is fixed by SDL3 GPU (see `src/bracken/shaders/README.md`): vertex uniforms live in set 1, fragment uniforms in set 3; sampled textures / storage buffers occupy sets 0 (vertex) and 2 (fragment).

Rules when editing or adding a uniform struct:

- `struct alignas(16) MyUniforms { ... };` and total size a multiple of 16.
- Follow HLSL cbuffer packing: 16-byte registers; **a `float3` followed by a `float` packs into one register, two consecutive `float3`s do not** — always pair every `glm::vec3` with a scalar or explicit `float _padding`.
- Pad with *named scalars*, not arrays: an HLSL `float pad[3]` strides at 16 bytes per element and desynchronizes the layout (see the comment on `UIFragmentUniforms` / `ShadowFragUniforms`).
- Update the HLSL cbuffer and the C++ struct **in the same change**, and add a `BRACKEN_HLSL_CBUFFER(T, size)` + one `BRACKEN_HLSL_FIELD(T, field, offset)` line per field directly after the struct (`gpu/hlsl_packing.hpp`). Those are `static_assert`s: a layout change becomes a compile error in the same translation unit, and the straddle check catches a `glm::vec3` that crossed a 16-byte register. Then add a row to `UNIFORM_TABLE` in `src/bracken/gpu/uniform_layout.test.cpp`, which reflects the **compiled SPIR-V** and fails when the shader and the mirror disagree — an unlisted cbuffer fails it too.

Storage-buffer record structs live in the same header but pack tightly (std430-style, no cbuffer padding): `InstanceData`, `ShadowInstanceData`, `ParticleInstanceData`. Useful constants: `ShadowDataUniforms::shadowEnabled` is `0` off / `1` PCF / `2` PCSS.

The clustered-lighting cbuffer layout is defined once by `render::lighting::ClusterParams` (`light_cluster.hpp`) — `SceneRenderer` pushes that struct directly, and `gpu_uniforms.hpp` deliberately carries no mirror; keep it in sync with `clusters.hlsli`.

### Shared vertex formats and enums: `gpu_types.hpp`

`src/bracken/gpu/gpu_types.hpp` defines the vertex structs the engine's pipelines expect, plus helpers that keep the SDL attribute descriptions in sync with the structs:

```cpp
using bracken::gpu::Vertex3D;      // pos, normal, uv, color, tangent  (64 bytes)
using bracken::gpu::SkinnedVertex; // Vertex3D + uint8 joints[4] + float weights[4] (84 bytes)
using bracken::gpu::Vertex2D;      // pos, uv, color (32 bytes, UI)

Vertex3D v;
v.position = {0.0f, 1.0f, 0.0f};
v.tangent  = {1.0f, 0.0f, 0.0f, 1.0f};  // w = bitangent sign per glTF
```

- `get_vertex3d_attributes()` / `get_vertex3d_buffer_desc()` (and the `skinned`/`2d` variants) are what `with_vertex3d()` etc. use — if you add a field to a vertex struct, these helpers and the corresponding HLSL inputs must change together (`src/bracken/gpu/gpu_types.test.cpp` pins the layouts).
- `TextureFormat` (`RGBA8`, `RGBA8_SRGB`, `BGRA8`, `R8`, `R16`, `D32F`, `D24S8`) with `to_sdl_format()` — unknown values fall back to `R8G8B8A8_UNORM`.
- `BlendMode` (`None`, `Alpha`, `Additive`, `Multiply`) with `get_blend_state()`; all modes write RGBA.
- The `CameraUniforms`/`ModelUniforms`/`LightUniforms`/`TimeUniforms` structs and `GrassInstance` at the top of this header are **not referenced anywhere else in the engine** — treat them as legacy; the live uniform contract is `gpu_uniforms.hpp`.

### Debug labels and GPU timers: `gpu_debug.hpp`

Label your passes so RenderDoc/Nsight captures are navigable:

```cpp
#include "bracken/gpu/gpu_debug.hpp"

{
    BRACKEN_GPU_SCOPE(cmd, "MainScene");        // RAII SDL debug group push/pop
    BRACKEN_GPU_LABEL(cmd, "opaque draws");     // single label
    // ... record the pass ...
}
```

`GPUTimestampPool` + `BRACKEN_GPU_TIMER(pool, cmd, "PassName")` are wired through `SceneRenderer` (64 pairs) and report a **real CPU record time** per scope through `core::PassTimeline` (`PassTiming::cpu_ms`, exponentially smoothed). The GPU half remains unavailable: SDL3 GPU (as of SDL 3.2) exposes no per-pass timestamp queries, so `is_supported()` is always `false` and `PassTiming::gpu_ms` is `std::nullopt` — deliberately not `0.0f`, which would read as a free pass. For GPU-side numbers use RenderDoc/Nsight. The API is kept so call sites stay identical when SDL adds support.

One contract if you use the pool directly: the `pass_name` passed to `reserve_pair`/`BRACKEN_GPU_TIMER` is stored as a raw `const char*` — it must be a string literal (or otherwise outlive the frame).

### Common mistakes checklist

- Editing an HLSL cbuffer without its `gpu_uniforms.hpp` mirror (or vice versa) — silently corrupt rendering. Add the layout test.
- Two consecutive `glm::vec3`s in a uniform struct, or array padding instead of named scalars.
- Using `TextureFormat::RGBA8` for albedo art (washed-out lighting) or `RGBA8_SRGB` for data masks (wrongly linearized values).
- Wrong `ShaderResources` counts for a shader — the load logs an error and uses the SPIR-V-reflected counts; fix the caller when you see it.
- Building a scene pipeline with `LESS_OR_EQUAL` depth compare (that's for standard-Z shadow cascades only; the scene is reversed-Z).
- Calling `GPUBuffer::update()` with data larger than the buffer — the update is silently dropped (logged), not resized.
- Expecting `GPUBuffer::Type::Uniform` to be a bindable uniform buffer — it's a storage buffer; per-draw uniforms use `SDL_PushGPU*UniformData`.
- Acquiring the swapchain off the main thread.
- Destroying the `GPUDevice` before the resources created from it.
- Reading `GPUTimestampPool`'s `cpu_ms` as GPU time — it measures command recording. `gpu_ms` is `nullopt` by design.

---

## Shader Authoring & Compilation

Bracken shaders are written in **HLSL**, compiled to **SPIR-V at build time** by the SDL_shadercross CLI, staged next to your executable by CMake, and loaded at runtime through `bracken::gpu::ShaderManager` / `bracken::gpu::PipelineRegistry`. This chapter covers the authoring rules, the build integration, the SDL3 GPU binding-order contract, the CPU↔GPU uniform contract, and the end-to-end recipe for adding a brand-new shader + pipeline.

Sources: `src/bracken/shaders/` (HLSL + `CMakeLists.txt` + `README.md`), `src/bracken/gpu/gpu_shader.hpp`, `src/bracken/gpu/gpu_pipeline.hpp`, `src/bracken/gpu/pipeline_registry.hpp`, `src/bracken/gpu/gpu_uniforms.hpp`, `src/bracken/gpu/gpu_types.hpp`.

### File naming and entry points

Shader sources live flat in `src/bracken/shaders/`. The filename decides how the build compiles the file (`src/bracken/shaders/CMakeLists.txt` derives stage and entry point from the name):

| Name pattern | Stage | Entry point | Output |
|---|---|---|---|
| `{name}.vert.hlsl` | vertex | `VSMain` | `{name}.vert.spv` |
| `{name}.frag.hlsl` | fragment | `PSMain` | `{name}.frag.spv` |
| `{name}.comp.hlsl` | compute | `CSMain` | `{name}.comp.spv` |
| `{name}.hlsli` | shared include | — | not compiled standalone |

Anything else with an `.hlsl` extension is skipped with a CMake **warning** (“Unknown shader type”), not an error — a misnamed shader silently never compiles. Compute shaders do NOT load through `GPUShader`/`ShaderManager` (SDL3 GPU has no `SDL_GPUShader` for compute — the module compiles directly into a pipeline): use `gpu::GPUComputePipeline` (`src/bracken/gpu/gpu_compute.hpp`), or `PipelineRegistry::get_compute_pipeline` for the engine's own compute passes. Compute bindings follow a different set layout than graphics (read-only vs read-write storage are split; see `src/bracken/shaders/README.md`), and read-write resources are bound at `SDL_BeginGPUComputePass` time — `gpu::ComputePass` wraps the whole begin/bind/dispatch/end sequence.

All vertex input/output interpolators use `TEXCOORD0..N` semantics (an SDL3 GPU D3D12-backend assumption); vertex inputs additionally carry `[[vk::location(n)]]` attributes matching the pipeline’s vertex-attribute locations.

### How the build compiles shaders

`src/bracken/shaders/CMakeLists.txt` does all of the work on the first configure:

- Fetches **SDL_shadercross** (pinned commit, vendored DXC + SPIRV-Tools) via `FetchContent`. This is the slow part of a cold build — per `.claude/RULES.md`, **never** delete `build/` or run a clean rebuild.
- Always compiles with the freshly built, pinned CLI (`$<TARGET_FILE:shadercross>`) — a `shadercross` binary on your `PATH` is deliberately ignored, so the build-time compiler and the runtime SDL3_shadercross library can never drift apart.
- Globs `*.hlsl` with `CONFIGURE_DEPENDS` and emits one `add_custom_command` per shader:

```bash
shadercross model.vert.hlsl -s HLSL -d SPIRV -t vertex -e VSMain \
    -I src/bracken/shaders -o <build>/src/bracken/shaders/spv/model.vert.spv
```

- **Every `.hlsli` is a declared dependency of every shader**, so editing any include recompiles all 49 shaders. An incremental `cmake --build build` is always enough.
- The `compile_shaders` target (built as part of `ALL`, and a dependency of the `bracken` library target) drives all of the compilations.

Because the glob uses `CONFIGURE_DEPENDS`, adding or deleting a shader file only needs an incremental build — CMake re-runs itself automatically:

```bash
# after adding src/bracken/shaders/myfx.frag.hlsl
cmake --build build     # re-glob + compile; NEVER a clean rebuild
```

Compiled `.spv` files land in a dedicated `spv/` output directory inside the build tree (exported as the internal cache variable `BRACKEN_SHADER_OUTPUT_DIR`; concretely `build/src/bracken/shaders/spv/*.spv` in this repo). At configure time, `.spv` files in that directory with no matching `.hlsl` source are removed, so deleted/renamed shaders can't keep shipping.

### Staging shaders next to your executable

At runtime `PipelineRegistry` loads SPIR-V from the `shaders/` directory **next to the executable** (`SDL_GetBasePath() + "shaders/"`; falls back to the relative path `"shaders/"` if `SDL_GetBasePath` fails). CMake stages them there with `bracken_copy_shaders`, defined in `src/bracken/CMakeLists.txt`:

```cmake
# examples/playground/CMakeLists.txt — the canonical consumer
add_executable(playground main.cpp playground.cpp character.cpp)
target_link_libraries(playground PRIVATE bracken)
bracken_copy_shaders(playground)   # copies $BRACKEN_SHADER_OUTPUT_DIR -> $<TARGET_FILE_DIR:playground>/shaders
```

The copy target runs on **every** build (`add_custom_target(... ALL)`): it removes the staging `shaders/` directory and recopies it from `BRACKEN_SHADER_OUTPUT_DIR`, so rebuilt shaders are re-staged automatically and deleted/renamed shaders don't linger as stale `.spv` files next to the executable.

### SDL3 GPU resource binding order — the rules you must follow

SDL3 GPU dictates which Vulkan descriptor set each resource class lives in. Every resource in a Bracken shader is annotated explicitly with `[[vk::binding(slot, set)]]`:

| Stage | Set | Contents (in slot order) |
|---|---|---|
| Vertex | 0 | sampled textures, then storage textures, then storage buffers |
| Vertex | 1 | uniform buffers (cbuffers) |
| Fragment | 2 | sampled textures, then storage textures, then storage buffers |
| Fragment | 3 | uniform buffers (cbuffers) |

Within a set, **storage buffers start at the slot right after your sampled textures**. Example: `src/bracken/shaders/grass.vert.hlsl` binds two vertex samplers at `(0,0)` and `(1,0)`, so its chunk-origin storage buffer is `(2,0)`. A vertex shader with no textures (e.g. `skinned_model.vert.hlsl`) puts its storage buffer at `(0,0)`.

Textures use SDL3 GPU’s **combined image sampler** convention — texture and sampler share one binding:

```hlsl
// fragment sampled texture: set 2
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
Texture2D baseColorTexture;
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState baseColorSampler;

// fragment cbuffer: set 3
[[vk::binding(0, 3)]]
cbuffer LightingUniforms { /* ... */ };
```

The CPU-side bind calls map by slot, not by set:

- `SDL_PushGPUVertexUniformData(cmd, slot, ...)` → vertex `cbuffer` at `[[vk::binding(slot, 1)]]`
- `SDL_PushGPUFragmentUniformData(cmd, slot, ...)` → fragment `cbuffer` at `[[vk::binding(slot, 3)]]`
- `SDL_BindGPUFragmentSamplers(pass, first, bindings, n)` → `[[vk::binding(first.., 2)]]`
- `SDL_BindGPUVertexStorageBuffers(pass, first, bufs, n)` → set-0 storage-buffer slots (after samplers)
- `SDL_BindGPUFragmentStorageBuffers(pass, first, bufs, n)` → set-2 storage-buffer slots (after samplers)

**Cluster-lighting include:** any scene fragment shader that wants clustered lights must define its sampler count before including `src/bracken/shaders/cluster_lighting.hlsli`, because the cluster SSBOs bind right after the samplers (the include `#error`s if you forget):

```hlsl
// model.frag.hlsl has 11 fragment samplers (the last one is the local-shadow
// atlas the include itself declares), so cluster SSBOs start at slot 11
#define CLUSTER_SSBO_BINDING_BASE 11
#include "cluster_lighting.hlsli"
```

Current values: `model.frag`/`model_instanced.frag` 11, `terrain.frag` 9,
`grass.frag` 7. The include declares the local-shadow atlas + comparison
sampler at `CLUSTER_SSBO_BINDING_BASE - 1` (your shader's LAST sampler slot)
and a 4th storage buffer for the per-tile shadow slots, so
`CLUSTER_SSBO_BINDING_BASE` = your in-file sampler count **plus one**.

The include also reserves fragment cbuffer `(2, 3)` for `ClusterParams` — don’t declare your own `b2` alongside it.

### The CPU↔GPU uniform contract

Every HLSL `cbuffer` has an `alignas(16)` CPU mirror struct in `src/bracken/gpu/gpu_uniforms.hpp`, annotated in its doc comment with the `(set, binding)` it is pushed to. The two sides must match **byte-for-byte** — a mismatch doesn’t crash, it silently mis-renders. This is the most common way to break rendering in this engine.

Rules of thumb (all enforced by example in the existing structs):

- Mind HLSL 16-byte register packing: `float3` followed by `float` packs into one register; two consecutive `float3`s do **not**. Every `glm::vec3` in the mirrors is followed by a scalar or explicit `float _paddingN`.
- Pad with **named scalars, not arrays**: an HLSL `cbuffer` array like `float _pad[3]` strides at 16 bytes per element and desynchronizes the layout. See `gpu::UIFragmentUniforms` / `gpu::ShadowFragUniforms` (three named `int`/`float` pads each).
- Total size must be a multiple of 16 (asserted for every struct by the `BRACKEN_HLSL_CBUFFER` block that follows it).
- **When you add or reorder a field, change both files in the same commit.** The CPU side is pinned by `static_assert`: `BRACKEN_HLSL_CBUFFER` (size + alignment) and one `BRACKEN_HLSL_FIELD` per field (offset + no 16-byte-register straddle), both from `gpu/hlsl_packing.hpp`. Forgetting the shader side is caught at test time by `src/bracken/gpu/uniform_layout.test.cpp`, which reads the compiled `.spv` and compares the cbuffer's declared byte extent to `sizeof(T)`. `gpu_uniforms.test.cpp` no longer carries offset assertions — it keeps only the cross-module default-value checks a `static_assert` cannot express.

Storage-buffer records are tightly packed (std430-style), no cbuffer padding rules — e.g. `gpu::ParticleInstanceData` is exactly `mat4 + vec4` = 80 bytes, mirrored by `struct ParticleInstance` in `particle.vert.hlsl`.

Pushing uniforms at draw time (real code from `src/bracken/effects/effect_renderer.cpp`):

```cpp
gpu::InstancedCameraUniforms camera_uniforms = {};
camera_uniforms.view = view;
camera_uniforms.projection = projection;
camera_uniforms.cameraPos = camera_pos;
SDL_PushGPUVertexUniformData(cmd, 0, &camera_uniforms, sizeof(camera_uniforms)); // set 1, b0

gpu::ParticleFragmentUniforms frag_uniforms = {};
SDL_PushGPUFragmentUniformData(cmd, 0, &frag_uniforms, sizeof(frag_uniforms));   // set 3, b0

SDL_GPUBuffer* instance_buf = instance_buffer_->handle();
SDL_BindGPUVertexStorageBuffers(pass, 0, &instance_buf, 1);                      // set 0, binding 0
```

### Loading shaders at runtime: `GPUShader` and `ShaderManager`

You normally never load shaders yourself — `PipelineRegistry` does it — but the API is public (`src/bracken/gpu/gpu_shader.hpp`):

```cpp
#include "bracken/gpu/gpu_shader.hpp"
using namespace bracken;

gpu::ShaderManager shaders(device);

gpu::ShaderResources fs_resources;
fs_resources.num_uniform_buffers = 1;
fs_resources.num_samplers = 1;

gpu::GPUShader* fs = shaders.get_fragment("shaders/ui.frag.spv", "PSMain", fs_resources);
```

Things to know:

- **`ShaderResources` counts should exactly describe what the shader declares** (`num_samplers`, `num_storage_textures`, `num_storage_buffers`, `num_uniform_buffers`). They are **validated against SPIR-V reflection at load** — a mismatch logs an error and the reflected counts win, so wrong counts are loud, not silent. Count `[[vk::combinedImageSampler]]` pairs as one sampler each.
- **There is no default `entry_point`** on the `GPUShader` loaders — the build compiles entry points `VSMain` / `PSMain`; `get_vertex`/`get_fragment` default to those.
- `ShaderManager` caches by `(path, stage, entry_point)`; repeat `get()` calls are free. The cache is **not thread-safe** — do all shader loading on one thread (the render/main thread).
- All failures return `nullptr` and log via `SDL_Log` (missing file, empty file, SPIR-V→backend compile failure). There is no exception path.
- `GPUShader::init_compiler()` (SDL_shadercross runtime init) is called automatically by the `ShaderManager` constructor and lazily by `create_from_spirv`; you only need it if you drive `GPUShader` directly.

At runtime SDL_shadercross converts the SPIR-V to whatever the backend wants: Vulkan consumes SPIR-V directly, Metal/D3D12 get transpiled via SPIRV-Cross.

### Pipelines: `PipelineRegistry` and `PipelineConfig`

`gpu::PipelineRegistry` (`src/bracken/gpu/pipeline_registry.hpp`) is the central owner of every graphics pipeline (`PipelineType::Count`). It is the cache and lifecycle only — what each pipeline *is* lives one row per type in the catalogue, `src/bracken/gpu/pipeline_catalog.hpp`, so **adding a pipeline is adding a row** (plus the enum value; a missing row is a compile error). `SceneRenderer` creates and preloads the registry during `init`; hosts usually only interact with it for hot-reload or when adding a new pipeline type.

```cpp
gpu::PipelineRegistry registry;
if (!registry.init(device)) { /* fatal */ }

gpu::GPUPipeline* model = registry.get_pipeline(gpu::PipelineType::Model); // lazy-created

if (!registry.preload_all_pipelines()) { /* one or more failed; each is logged */ }

// in a render pass:
model->bind(render_pass);
```

- `get_pipeline` lazily creates and caches; returns `nullptr` on failure (missing `.spv`, pipeline-creation error) and logs which pipeline failed.
- `preload_all_pipelines()` iterates every `PipelineType` — call at load time to avoid first-use hitches. A new enum value is picked up automatically.
- `invalidate_all()` drops all pipelines *and* the shader cache; everything is rebuilt on next access.
- `set_swapchain_format(fmt)` only invalidates the four pipelines that actually target the swapchain (`UI`, `Text`, `Composite`, `Fxaa`); scene/HDR/post pipelines render to fixed-format offscreen targets and stay cached.
- Formats: swapchain `RGBA8_UNORM` (queried from the device at `init`), HDR scene color `RGBA16F` (`get_scene_color_format()`), depth `D32_FLOAT`.

The full **pipeline → shader map** lives in `src/bracken/shaders/README.md` — notable sharing: the skinned pipeline reuses `model.frag`, every post-process pass reuses `fullscreen.vert` (a bufferless fullscreen triangle generated from `SV_VertexID`), and the model-family variants (`ModelDoubleSided`, `ModelBlend`, `SkinnedModel*`, `InstancedModelDoubleSided`) load the *same* shader pair as their base type and differ only in fixed-function state.

`gpu::PipelineConfig` (`src/bracken/gpu/gpu_pipeline.hpp`) is a fluent builder over `SDL_GPUGraphicsPipelineCreateInfo`:

```cpp
gpu::PipelineConfig config;
config.vertex_shader = vs->handle();
config.fragment_shader = fs->handle();
config.with_vertex3d().alpha_blended().no_cull();   // chainable
config.color_format = registry.get_scene_color_format();
config.depth_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
auto pipeline = gpu::GPUPipeline::create(device, config);   // nullptr on failure
```

Builder cheat-sheet (see `src/bracken/gpu/gpu_pipeline.cpp` and `gpu_pipeline.test.cpp`):

| Helper | Effect |
|---|---|
| `with_vertex3d()` | `gpu::Vertex3D` layout (5 attributes: position/normal/texcoord/color/tangent) |
| `with_skinned_vertex()` | `gpu::SkinnedVertex` layout (7 attributes: + joints `UBYTE4`, weights `FLOAT4`) |
| `with_vertex2d()` | `gpu::Vertex2D` layout (3 attributes, UI) |
| `opaque()` | no blending, depth test + write on |
| `alpha_blended()` | `BlendMode::Alpha`, depth test on, **depth write off** |
| `additive()` | `BlendMode::Additive` (SRC_ALPHA, ONE), depth test on, depth write off |
| `no_depth()` | disables depth test, write, *and* the depth target |
| `no_cull()` / `cull_front()` | cull mode NONE / FRONT (default is BACK) |
| `with_depth_bias(c, s, clamp=0)` | enables rasterizer depth bias |

Post-process pipelines set no vertex layout at all (fullscreen triangle, no vertex input). Depth-only pipelines set `config.color_format = SDL_GPU_TEXTUREFORMAT_INVALID`.

### The depth convention — get this right or your pass draws nothing

The scene renders **reversed-Z** (near = device depth 1.0, far/clear = 0.0), so `PipelineConfig`’s **default** `depth_compare_op` is `SDL_GPU_COMPAREOP_GREATER_OR_EQUAL`. Consequences for shader/pipeline authors:

- Depth-reading shaders treat depth `<= 0.0` as sky/far (`gtao`, `gtao_temporal`, `blur_ao`, `volumetric_fog` all do).
- The skybox vertex shader emits `z = 0.0` so it only passes where nothing was drawn.
- Depth linearization `viewZ = |B / (depth + A)|` with `A = proj[2][2]`, `B = proj[3][2]` works for both conventions — reuse it rather than writing convention-specific math.
- **Exception:** the four shadow-cascade pipelines deliberately stay standard-Z (`LESS_OR_EQUAL`, atlas cleared to 1.0, front-face culling, depth bias `(1.25, 1.75)`, depth clip off for caster pancaking). If you copy a shadow pipeline as a template for a scene pass, remember to drop the `LESS_OR_EQUAL` override.

### Shader hot-reload

Set the environment variable and run — no code changes needed if you use `Application` (`src/bracken/application.cpp` wires it in `init_renderer`):

```bash
BRACKEN_HOT_RELOAD=1 ./build/examples/playground/playground
# ... in another terminal, edit any .hlsl, then:
cmake --build build      # recompiles the .spv and re-stages it next to the exe
```

The `PipelineRegistry` watches the runtime `shaders/` directory for `.spv` changes (`enable_hot_reload(FileWatcher&)`). On **any** change it invalidates *all* pipelines and shader caches and preloads everything again — coarse but correct; SDL releases the previous pipelines once in-flight frames finish. If a rebuilt shader fails to load, the failure is logged (`hot_reload` category) and that pipeline stays missing (`get_pipeline` returns `nullptr`) until the next successful reload.

Direct users of `PipelineRegistry` outside `Application` call `enable_hot_reload` themselves after `init`; `reload_all_pipelines()` forces a pass immediately (used by tests/diagnostics).

### Recipe: adding a brand-new shader + pipeline end-to-end

Say you want a new fullscreen post-process pass `vignette`.

**1. Write the shader.** Create `src/bracken/shaders/vignette.frag.hlsl` (reuse `fullscreen.vert` — don’t write a new vertex shader for fullscreen passes):

```hlsl
struct PSInput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]                    // fragment cbuffer -> set 3
cbuffer VignetteUniforms {
    float strength;
    float _pad0;                          // named scalar pads, never float[3]
    float _pad1;
    float _pad2;
};

[[vk::combinedImageSampler]][[vk::binding(0, 2)]]   // fragment texture -> set 2
Texture2D sceneTexture;
[[vk::combinedImageSampler]][[vk::binding(0, 2)]]
SamplerState sceneSampler;

float4 PSMain(PSInput input) : SV_Target {
    float4 c = sceneTexture.Sample(sceneSampler, input.texCoord);
    float d = distance(input.texCoord, float2(0.5, 0.5));
    return float4(c.rgb * (1.0 - strength * d * d), c.a);
}
```

**2. Build.** `cmake --build build` — the `CONFIGURE_DEPENDS` glob picks the file up, compiles `vignette.frag.spv`, and the copy target stages it.

**3. Add the CPU uniform mirror** in `src/bracken/gpu/gpu_uniforms.hpp`, with its `BRACKEN_HLSL_CBUFFER`/`BRACKEN_HLSL_FIELD` static-assert block, and a row in `UNIFORM_TABLE` in `src/bracken/gpu/uniform_layout.test.cpp`:

```cpp
/** Vignette pass fragment uniforms (set 3, b0) */
struct alignas(16) VignetteUniforms {
    float strength = 0.5f;
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
    float _pad2 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(VignetteUniforms, 16);
BRACKEN_HLSL_FIELD(VignetteUniforms, strength, 0);
BRACKEN_HLSL_FIELD(VignetteUniforms, _pad0, 4);
BRACKEN_HLSL_FIELD(VignetteUniforms, _pad1, 8);
BRACKEN_HLSL_FIELD(VignetteUniforms, _pad2, 12);
```

```cpp
// src/bracken/gpu/uniform_layout.test.cpp, in UNIFORM_TABLE:
ROW("vignette.frag.spv", 3, 0, gpu::VignetteUniforms),
```

**4. Register the pipeline** in `src/bracken/gpu/pipeline_registry.hpp/.cpp` — four edits, all following existing patterns:
   - add `Vignette` to `enum class PipelineType` (before `Count`);
   - add a case to `pipeline_type_to_string`;
   - declare + implement `create_vignette_pipeline()` — copy `create_fxaa_pipeline` and adjust the `ShaderResources` counts to match your declarations exactly (here: `fs_resources.num_uniform_buffers = 1; fs_resources.num_samplers = 1;`):

```cpp
std::unique_ptr<GPUPipeline> PipelineRegistry::create_vignette_pipeline() {
    ShaderResources vs_resources;                       // fullscreen.vert: none
    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1;               // VignetteUniforms
    fs_resources.num_samplers = 1;                      // scene texture

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "vignette.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);
    if (!vs || !fs) return nullptr;

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();               // no vertex input: fullscreen triangle
    config.color_format = swapchain_format_;            // or scene_color_format_ for HDR passes
    return GPUPipeline::create(*device_, config);
}
```

   - add the `case PipelineType::Vignette:` to the `get_pipeline` switch. `preload_all_pipelines()` and hot-reload then cover it automatically. If it targets the swapchain, also add it to the `swapchain_dependent` list in `set_swapchain_format`.

**5. Draw with it** (inside a render pass whose color target format matches `config.color_format`):

```cpp
auto* pipeline = registry.get_pipeline(gpu::PipelineType::Vignette);
if (pipeline) {
    pipeline->bind(pass);
    gpu::VignetteUniforms u;
    SDL_PushGPUFragmentUniformData(cmd, 0, &u, sizeof(u));
    SDL_GPUTextureSamplerBinding binding = {scene_tex, sampler};
    SDL_BindGPUFragmentSamplers(pass, 0, &binding, 1);
    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);   // fullscreen triangle
}
```

**6. Verify.** Run the smoke test and inspect the screenshot it self-verifies:

```bash
cmake --build build && ctest --test-dir build --output-on-failure
build/examples/playground/playground --smoke    # writes playground_smoke.png
```

Also update the pipeline → shader table in `src/bracken/shaders/README.md` — the docs are part of the contract.

### Common mistakes checklist

- **Wrong `ShaderResources` counts** — nothing validates them; symptoms range from silent misbinding to crashes.
- **Editing one side of a cbuffer pair** — always change the HLSL cbuffer and the `gpu_uniforms.hpp` mirror together, and update the layout tests.
- **`float3` packing / array padding** — two adjacent `float3`s, or `float _pad[3]` instead of three named scalars, silently shifts every following field.
- **Forgetting the entry point** — `GPUShader::load_spirv`’s default entry is `"main"`; the engine’s shaders use `VSMain`/`PSMain`.
- **Wrong depth compare** — new scene pipelines want the default `GREATER_OR_EQUAL` (reversed-Z); only shadow-cascade pipelines use `LESS_OR_EQUAL`.
- **Wrong color format** — the pipeline’s `color_format` must match the render-target texture format at draw time (`scene_color_format_` = RGBA16F for scene/post, `swapchain_format_` for UI/Composite/FXAA, `R8_UNORM` for AO, `INVALID` for depth-only).
- **Misnamed file** — `myshader.hlsl` (without `.vert`/`.frag`) is skipped with only a configure-time warning.
- **Expecting hot-reload to compile HLSL** — the watcher watches the staged `.spv` files; you still need `cmake --build build` after editing HLSL.

---

## Scene: Cameras, Culling, FramePacket & SceneRenderer

The `scene/` module (`bracken::scene`, `src/bracken/scene/`) is the boundary between your game code and the renderer. Each frame your game describes everything it wants drawn — 3D world, lights, particles, 2D overlay, camera, environment — into a **`FramePacket`** through a **`FrameBuilder`**, and the engine seals it and hands the immutable packet to `SceneRenderer::render_frame(const FramePacket&)` — overlapped with your next frame's `on_simulate`. The renderer owns every GPU resource and sub-renderer; your code never touches a render pass directly, and nothing you emit carries a lifetime obligation — everything is copied or allocated inside the packet.

The canonical consumer is the playground example (`examples/playground/playground.cpp`, `on_simulate()`), and most snippets below are lifted from it.

### The per-frame pattern

Under `Application::run()` the engine owns the packet ring and the acquire → build → seal → render → retire cycle; your `on_simulate` only emits:

```cpp
// examples/playground/playground.cpp (abridged)
void PlaygroundGame::on_simulate(bracken::SimFrame& sim) {
    // ... simulation first (movement, animation, IK) ...

    scene::FrameBuilder& fb = sim.frame();          // engine-acquired, engine-sealed

    fb.set_camera(camera().get_camera_state());
    fb.set_draw_skybox(graphics_settings().skybox_enabled);
    fb.set_draw_ground(true);
    fb.set_draw_grass(graphics_settings().grass_enabled);

    auto pal = fb.alloc_bone_palette(player_joint_count_);   // packet memory
    std::ranges::copy(anim_player_.bone_matrices().first(player_joint_count_),
                      pal.matrices.begin());
    fb.draw_skinned(player_model_, player_transform_, pal.id);

    for (const auto& prop : props_) {
        fb.draw_model(prop.model, prop.transform);
    }

    build_ui(fb.ui());
    // The engine's sim-stage tail (particles, clocks) runs after this
    // returns, then seals; render + retire happen on the main thread.
}
```

(Hosts that bypass `run()` — tools, tests — drive the same cycle by hand: `frame_packets().acquire_for_fill()`, `FrameBuilder fb(packet)`, emit + `set_frame_timing`, `fb.seal()`, `render_frame(packet)`, `frame_packets().retire(packet)`.)

The packet's storage is a per-frame arena that is reset (memory retained) at retire, so the steady-state frame allocates nothing. A `FrameBuilder` is **spent after `seal()`** — never stash the reference (or the old packet's spans) beyond `on_simulate`.

Contracts to know up front:

- **`render_frame` takes the packet by `const&`** and consumes it exactly once. There is no mutation of your state and no queue draining — effect spawns, palettes, UI text all live inside the packet.
- **`render_frame` runs on the main thread** (guarded by `BRACKEN_ASSERT_MAIN_THREAD()` in debug builds); **builder emission is sim-stage** — `on_simulate` and jobs it fans out to (one `fb.lane(i)` per job; the plain `fb.` conveniences are lane 0, for the frame's root code).
- **Write-after-seal is a bug and asserts in debug** — every builder emit checks the packet is still Filling; every packet read checks it is sealed. Do not call `seal()`/`set_frame_timing` yourself under `run()`.

### Cameras and projection (`CameraState`)

`scene::CameraState` (`src/bracken/contracts/camera_state.hpp`) is a plain struct — the game computes it, the renderer consumes it:

```cpp
struct CameraState {
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 view_projection;   // must equal projection * view
    glm::vec3 position;
};
```

If you use the built-in camera (`systems::CameraController`, accessed via `Application::camera()`), you never build one by hand:

```cpp
fb.set_camera(camera().get_camera_state());  // camera_controller.hpp; snapshotted by value
```

(If you never call `set_camera` in a frame, `packet.has_camera()` stays false and the renderer skips the 3D pass rather than draw with a stale view.)

If you drive your own camera, you must follow the engine's projection conventions or depth testing breaks silently:

- **Y-up, right-handed**: x/z is the ground plane, y is height. The view matrix is a standard `glm::lookAt` with `(0,1,0)` up.
- **Zero-to-one clip depth**: `GLM_FORCE_DEPTH_ZERO_TO_ONE` is a PUBLIC compile definition on the `bracken` target, so it propagates to your game target automatically — glm produces `RH_ZO` matrices.
- **Reversed-Z**: the engine swaps near and far in the `glm::perspective` call, so the *near* plane maps to device depth **1** and the *far* plane to **0**. Scene pipelines compare `GREATER_OR_EQUAL` and clear depth to 0.0. This is what the engine's own camera does (`src/bracken/systems/camera_system.cpp`):

```cpp
// Reversed-Z: near/far deliberately swapped (far=15000, near=5).
projection_matrix_ = glm::perspective(glm::radians(current_fov_), aspect, 15000.0f, 5.0f);
```

The test `src/bracken/contracts/camera_state.test.cpp` (`ProjectionIsRightHandedReversedZeroToOneDepth`, `ReversedZKeepsDistantDepthsDistinct`) pins this behavior down and is the best executable reference.

Common mistakes:

- Building a projection with near/far in the *conventional* order. The frustum culling will still be correct (see below), but every depth-tested scene draw will resolve wrong (near geometry loses to far geometry).
- Forgetting `view_projection = projection * view`. The renderer extracts the culling frustum from `camera.view_projection` and uses `view`/`projection` separately for drawing — if they disagree, visible objects get culled.
- Using `-1..1`-depth helper functions from other engines/code. Never mix them in.

### Frustum culling (`Frustum`, `Plane`)

`scene::Frustum` (`src/bracken/contracts/frustum.hpp`) extracts six planes from any view-projection matrix (Gribb-Hartmann method) and offers two intersection tests:

```cpp
#include "bracken/contracts/frustum.hpp"

scene::Frustum frustum;
frustum.extract_from_matrix(cam.view_projection);

bool visible_sphere = frustum.intersects_sphere(center, radius);
bool visible_box    = frustum.intersects_aabb(min_corner, max_corner);
```

It works unchanged for perspective, orthographic (shadow-map style), standard-Z, and reversed-Z matrices — under reversed-Z the two depth planes swap near/far roles, but the extracted plane *set* is identical, so culling results match a standard-Z frustum exactly (proven in `src/bracken/contracts/frustum.test.cpp`, `ReversedZFrustumTest.MatchesStandardZFrustumOnSamplePoints`).

You normally don't call this yourself for world content: `SceneRenderer` extracts the main-camera frustum once per frame and culls every packet draw command with it (plus an independent frustum per shadow cascade). Culling is controlled by `GraphicsSettings::frustum_culling` (default `true`, `src/bracken/contracts/graphics_settings.hpp`). Distance culling is always on and uses the XZ-plane distance to `GraphicsSettings::get_draw_distance()` — note it ignores height, so an object directly above the camera is never distance-culled.

`scene::Plane` is public too (normal + distance, `distance_to_point`, `normalize`) if you need half-space tests for gameplay.

### Transform helpers (`transform_math.hpp`)

`src/bracken/scene/transform_math.hpp` has three free inline functions the renderer uses on every command; they're available to game code as well:

```cpp
#include "bracken/scene/transform_math.hpp"

bool uniform   = scene::has_uniform_scale(model_mat);      // ~0.1% tolerance on squared scales
glm::mat4 nrm  = scene::compute_normal_matrix(model_mat);  // returns model_mat unchanged for uniform scale,
                                                           // transpose(inverse(model_mat)) otherwise
float max_s    = scene::max_scale_factor(model_mat);       // for expanding bounding spheres
```

Gotcha: `compute_normal_matrix` relies on the shader re-normalizing normals, which is why the uniform-scale fast path can skip the inverse-transpose. If you scale non-uniformly, the slow path (a full 4×4 inverse) runs per command per frame — prefer uniform scale on things you submit in bulk.

### Building the 3D frame (`FrameBuilder`)

`scene::FrameBuilder` (`src/bracken/scene/frame_builder.hpp`) is the only write interface to a packet. Its emit methods mirror the retired `RenderScene`/`UIScene` API one-for-one, so porting is mechanical.

#### The draw flags

```cpp
fb.set_draw_skybox(bool);  // also: set_draw_ground, set_draw_grass
```

Every packet starts each frame with all three flags `true` (a freshly retired packet behaves like a freshly constructed one). `packet.has_3d_content()` returns true if skybox/ground/grass is enabled or the frame has any models, skinned models, billboards, debug lines, or lights; when it's false the whole 3D section of the frame is skipped (UI still renders).

#### Static models

```cpp
fb.draw_model(handle, transform);                          // ModelHandle from ModelManager
fb.draw_model(handle, transform, /*tint=*/{1,0.5f,0.5f,1},
              /*force_non_instanced=*/false, /*no_fog=*/false, /*motion_id=*/0);
```

Handles are resolved once at load/spawn time (`models().load_model(...)` / `models().register_model(...)`); commands are handle-only so the per-frame path does zero string work. All commands for the same `ModelHandle` are automatically batched into one instanced draw per mesh. `force_non_instanced = true` opts a command out of instancing (it renders through the individual-draw path — use for one-offs that need it). `no_fog = true` disables distance fog for that command. `tint` multiplies the material's base color. A nonzero `motion_id` (host-unique per moving object) lets the velocity prepass pair this frame's transform with last frame's for correct motion vectors; 0 means static.

#### Skinned models and packet bone palettes

```cpp
auto pal = fb.alloc_bone_palette(joint_count);   // 1..=MAX_BONES (128); pal.matrices is packet memory
std::ranges::copy(anim_player_.bone_matrices().first(joint_count), pal.matrices.begin());
fb.draw_skinned(player_model_, player_transform_, pal.id);
```

The palette lives **inside the packet**: `alloc_bone_palette(joint_count)` returns uninitialized, 16-byte-aligned packet memory (`std::span<glm::mat4>`) plus a `BonePaletteId`. Write every matrix before `seal()` (copy from your `AnimationPlayer`, or compute straight into the span), then reference the palette from any number of `draw_skinned` calls — sharing one palette across many draws is done by reusing the id. What this buys you:

- **No lifetime contract.** The old non-owning bone-matrix span over host storage is gone; re-`update()`ing the player after submission is harmless.
- Palettes are packed tightly (`joint_count` matrices, not `MAX_BONES`); the renderer clamps drawn joints to `min(model joint count, joint_count)`.
- Commands whose model is missing or has no skeleton are silently skipped; a `draw_skinned` with an invalid id drops that draw with a warning.

**Per-frame budget**: `MAX_BONE_PALETTES = 1024` palettes (`src/bracken/scene/frame_packet.hpp`), enforced **when you allocate**: an over-budget `alloc_bone_palette` logs an error and returns an invalid allocation (`pal.valid() == false`, empty span) — check it if you run fleets of characters. This replaces the old silent render-time drop.

**Gotcha**: `BonePaletteId`s are only meaningful within the frame that allocated them (at `seal()` they are remapped to dense packet indices). Never cache an id across frames.

#### Dynamic lights

```cpp
// examples/playground/playground.cpp
fb.add_point_light({pos, /*radius=*/320.0f, /*color=*/{1.0f, 0.72f, 0.30f},
                    /*intensity=*/12000.0f, /*casts_shadows=*/true});

fb.add_spot_light({origin, /*radius=*/1400.0f, dir,
                   std::cos(glm::radians(18.0f)), {0.9f, 0.95f, 1.0f},
                   std::cos(glm::radians(32.0f)), /*intensity=*/200000.0f,
                   /*casts_shadows=*/true});
```

Types are `render::lighting::PointLight` / `SpotLight` (`src/bracken/contracts/light.hpp`), copied by value into the packet; spot cone angles are passed as **cosines** (inner, outer). Intensity is radiometric — falloff is inverse-square in world units, so a light meant to read against the default sun (`sun_intensity` 3.0) needs `intensity ≈ 3 × distance²`. Lights feed the clustered-forward binning; the frame cap is `MAX_LIGHTS = 256` (point + spot combined).

**Local light shadows** (`casts_shadows = true`): opt a light into the per-frame shadow budget (`GraphicsSettings::local_shadow_point_budget`/`_spot_budget`, defaults 4 + 2). Each frame the engine picks the highest-priority casters (score = intensity × radius / camera distance, with eviction hysteresis so lights don't pop at the boundary) and renders them into a shared D32 atlas — 6 perspective cube-face tiles per point light, 1 per spot. Tiles are **cached**: they re-render only when the light moves/changes or a dynamic caster (skinned, or a model submitted with nonzero `motion_id`) enters the light radius, capped at 6 tiles/frame (round-robin) so many moving lights degrade to slightly stale shadows, never a hitch. Non-selected casters and lights whose tiles haven't rendered yet simply fall back to unshadowed. Casters are model + skinned commands (`mesh.cast_shadows` respected); terrain and grass never cast local shadows. **Gotcha**: emissive "bulb" geometry that encloses its own light (the playground's orbs) must set `mesh.cast_shadows = false`, or the shadow map — rendered from inside the bulb — occludes everything. Leave `shadow_slot` alone; the engine overwrites it every frame.

#### 3D billboards

```cpp
fb.add_billboard_3d(pos.x, pos.y + 28.0f, pos.z,
                    /*width=*/46.0f, /*fill_ratio=*/fill,
                    /*fill_color=*/0xFFFFC040, /*bg_color=*/0x60000000,
                    /*frame_color=*/0xFFFFD080);
```

A billboard is a screen-space progress bar (fill + background + 1px frame) anchored at a world position — health bars, capture progress. The renderer projects it, scales with distance (clamped ×0.3–×1.5), and drops it when behind the camera or well outside the viewport. Colors use the UI renderer's `0xAARRGGBB` packing (same values as every 2D UI command).

#### Particle effect spawns

```cpp
fb.spawn_effect(&pickup_burst_, pos, /*direction=*/{0, 1, 0});
fb.spawn_effect(&attack_arc_, origin, fwd);
```

The `bracken::EffectDefinition*` must outlive all spawned instances (definitions are typically long-lived assets). Spawns ride the packet and are consumed exactly once by the engine's **sim-stage tail** (applied to the sim-owned `EffectSystem` before its update, so a spawn produces particles the same frame; the renderer draws the packet's effect-instance snapshot). There is no persistent spawn queue anymore — emit a spawn in the frame you want it, exactly once per event.

#### Debug drawing

```cpp
fb.add_debug_line(a, b, 0xFFFF5050);
fb.add_debug_sphere(player_pos_ + glm::vec3(0, 95, 0), 65.0f, 0xFF30FF60); // 3 circles × 16 segments
fb.add_debug_box(aabb_min, aabb_max, 0xFFFFFFFF);                          // 12 edges
```

Colors are packed **ARGB** `0xAARRGGBB` — the single engine-wide convention (`bracken/core/color.hpp`), identical to every UI/text/billboard color parameter. Lines render in-world (depth-tested via the grid pipeline) after all other 3D content. Sphere = 3 axis-aligned circle outlines (`segments` line segments each, default 16); box = 12 AABB edges.

### 2D UI (`fb.ui()`)

`FrameBuilder::ui()` returns a small `UiLane` value handle recording screen-space commands in pixels (origin top-left, y down) into the packet:

```cpp
auto ui = fb.ui();
ui.add_filled_rect(10, 10, 280, 122, 0xB0101418);
ui.add_rect_outline(x, y, w, h, color, /*line_width=*/2.0f);
ui.add_circle(x, y, radius, color, /*segments=*/24);
ui.add_circle_outline(x, y, radius, color, /*line_width=*/2.0f, /*segments=*/24);
ui.add_line(x1, y1, x2, y2, color, /*line_width=*/2.0f);
ui.add_text("bracken playground", 22, 20, /*scale=*/1.0f, /*color=*/0xFFFFD080);
ui.add_button(x, y, w, h, "label", color, /*selected=*/false);

float used_height = ui.add_text_wrapped(
    "WASD move | Shift run | ...", 10, y, /*max_width=*/w, /*scale=*/0.9f, 0xFFC8D0DC);
```

Text parameters are `std::string_view` and are **copied into the packet at emit time**, so `std::format(...)` temporaries are safe. `add_text_wrapped` word-wraps on spaces using an approximate glyph width of 8 px × scale (line height default 18 × scale, overridable via the trailing `line_height` parameter) and returns the total height consumed so you can stack elements below it. Commands are stored as a `std::variant` (`UiCommandData`) and replayed against `render::UIRenderer` inside `render_frame` — UI always draws *after* the 3D scene and composite, directly to the swapchain, so it is not tonemapped.

### SceneRenderer — the frame orchestrator

`scene::SceneRenderer` (`src/bracken/scene/scene_renderer.hpp`) is owned by `Application` (accessor: `Application::scene_renderer()`); if you use `Application::init_renderer` you never construct or `init()` it yourself. It is non-copyable and non-movable.

#### What a frame does

`render_frame(const FramePacket& frame)` runs, in order (full detail in `docs/technical_spec.md` and `docs/architecture.md`):

1. CPU prep: propagate `environment()` to sub-renderers (adopting the packet's snapshot when the host called `set_environment` that frame), adopt the packet's sim-advanced scene/sky clocks, pre-upload models referenced by the packet's effect-instance snapshot, cull + batch instances (main view and per shadow cascade), resolve skinned draws against the packet's pre-packed bone palettes, select the budgeted local-shadow lights + dirty tiles, build the cluster light grid. (Effect spawning and particle simulation happened on the sim side, before the packet sealed.)
2. One consolidated copy pass uploading everything (instances, shadow instances, bone palettes from packet memory, snapshot particles, debug lines, grass, cluster buffers, local-shadow slot data).
3. IBL upkeep (when `ibl_enabled` and the sky-relevant environment fields changed): a few amortized cube-face prefilter passes.
4. Shadow cascade passes (when `shadow_mode != ShadowMode::Off`), then this frame's dirty local-shadow tiles (when `local_shadows` is on).
5. Depth prepass: opaque geometry writes scene depth + RG16F motion vectors, then the GTAO chain (raw → temporal → blur → upsample; when `ao_mode != Off`) — both BEFORE the forward pass, which samples the AO texture in its ambient term.
6. The 3D scene into a linear-HDR offscreen target (opaques depth-test `EQUAL` against the prepass): terrain → grass → instanced models → non-instanced models → skinned models → skybox → sorted transparent (glTF `BLEND`) meshes → particles → debug lines.
7. Post chain: TAA resolve (when `aa_mode == TAA`; stabilizes the HDR image before bloom reads it) → auto exposure (when `auto_exposure`; two compute dispatches metering the resolved image) → bloom → volumetric fog → composite (exposure/white balance/tonemap/CDL grade) → FXAA only when `aa_mode == FXAA`.
8. UI + 3D billboards to the swapchain, then the optional post-UI callback.

If the window is minimized the frame is skipped but still submitted (no stall); if a frame has no 3D content (`has_3d_content()` false), only UI renders.

#### Configuration

```cpp
auto& sr = scene_renderer();
sr.set_heightmap(heightmap);            // terrain + grass heightfield
sr.set_graphics_settings(settings);     // GraphicsSettings snapshot; also resizes/reinits shadows
sr.set_anisotropic_filter(3);           // level N => 2^N x, clamped to 16x; 0 disables
sr.set_vsync_mode(mode);                // clamped by max_vsync_mode()
sr.set_screen_size(w, h);               // resizes UI projection + AO/fog targets
```

`Application` mirrors most of these (`set_heightmap`, `set_graphics_settings`, …) so you rarely call the renderer directly. Settings changes take effect on the next frame.

#### Live environment editing

```cpp
render::EnvironmentState& env = scene_renderer().environment();
env.sun_dir = glm::normalize(glm::vec3(-0.3f, -0.9f, -0.2f)); // direction light TRAVELS
env.fog_color_horizon = {0.45f, 0.56f, 0.74f};                 // linear color
```

`environment()` is the single source of truth for sun/sky/fog when the host doesn't snapshot one into the frame. All colors are **linear** (the composite pass tonemaps). Mutations apply next frame — `render_frame` re-propagates `env_` to every sub-renderer at its top. Alternatively, drive it per frame from your game state with `fb.set_environment(env)`: the renderer then adopts that snapshot for the frame (note this overwrites live edits made through `environment()`, including the DevUI Environment tab — hosts that want DevUI live-editing should *not* call `set_environment` every frame).

#### Render stats

```cpp
scene_renderer().set_collect_stats(true);
const bracken::RenderStats& stats = scene_renderer().render_stats();
// stats.draw_calls, stats.triangle_count, stats.entities_rendered,
// stats.entities_distance_culled, stats.entities_frustum_culled, stats.pass_times_ms
```

Stats are only accumulated while `set_collect_stats(true)` (`Application::render_stats()` returns `nullopt` otherwise). Gotcha: `pass_times_ms` carries CPU **record** times; `PassTiming::gpu_ms` is `nullopt` because SDL3 has no per-pass timestamp queries. Use RenderDoc/Nsight for GPU timing.

#### Screenshots

```cpp
request_screenshot("shot.png");   // Application forwards to SceneRenderer::request_screenshot
```

One-shot: captures the *next* frame's final swapchain image (UI included) to a PNG. The capturing frame submits with a blocking GPU fence wait — expect a visible hitch; it's meant for integration tests and debugging (the playground's `--smoke` mode uses it, then re-reads the PNG to self-verify).

#### Post-UI callback

```cpp
scene_renderer().set_post_ui_callback(
    [](SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchain) { /* extra pass */ });
```

Runs after UI, before submit — this is how Dear ImGui overlays hook in. **Single slot**: `Application::init_dev_ui()` claims it for the dev overlay, so you cannot use both the built-in dev UI and your own callback.

#### Accessors

`context()`, `terrain()`, `models()` (the `ModelManager` — load/register models here), `grass()`, `pipeline_registry()`, `get_terrain_height(x, z)`, and `effect_system()` (read-only view of particles; null until `init()` succeeds).

### Limits and silent behaviors (quick reference)

- Bone palettes beyond **`MAX_BONE_PALETTES = 1024`** per frame: `alloc_bone_palette` fails visibly at build time (error log + invalid allocation); draws submitted with an invalid id are dropped with a warning.
- `BonePaletteId`s are frame-local — never cache them across frames (they are remapped at `seal()`).
- Dynamic lights beyond `MAX_LIGHTS = 256`: excess ignored by the cluster grid.
- Distance culling is XZ-only (height ignored); frustum culling needs model bounds and is skipped for commands whose model isn't resolvable.
- A retired packet resets its draw flags to `true` and drops everything, including unconsumed effect spawns — spawns emitted into a packet that never reaches `render_frame` are lost.
- Per-pass GPU times in `RenderStats` are zeros (timestamp stub).
- Transparent (`BLEND`) meshes: sorted back-to-front per mesh per command — no OIT, intersecting transparents can blend wrong; `BLEND` meshes in instanced batches fall back to one draw per instance.

---

## Model Loading, Heightmaps & the glTF Contract

This chapter covers getting geometry into Bracken: loading `.glb` models (via
`ModelLoader` / `ModelManager` in `src/bracken/model_loader.hpp`), what meshes,
materials, and textures the loader produces, the transform/tangent helpers in
`src/bracken/model_utils.hpp`, terrain heightmaps (`src/bracken/heightmap.hpp`),
and the authoring rules your `.glb` files must follow
(`docs/skinned_mesh_format.md`).

### Loading a model: the 60-second version

Models are glTF **binary** files (`.glb`) loaded through the `ModelManager`
that every `Application` owns (accessor: `Application::models()` in
`src/bracken/application.hpp`). You load once, keep the returned integer
handle, and submit that handle to the frame builder every frame:

```cpp
#include "bracken/model_loader.hpp"

// In your Application subclass, inside on_init():
bracken::ModelHandle hero = models().load_model("hero", "assets/models/hero.glb");
if (hero == bracken::INVALID_MODEL_HANDLE) {
    return false; // file missing, unparsable, or Draco-compressed
}

// Later, every frame in on_simulate() (fb = sim.frame()):
fb.draw_model(hero, transform);                          // static mesh
auto pal = fb.alloc_bone_palette(joint_count);           // skinned mesh
std::ranges::copy(anim_player_.bone_matrices().first(joint_count), pal.matrices.begin());
fb.draw_skinned(hero, transform, pal.id);
```

`load_model` parses the file, uploads vertex/index buffers and all textures to
the GPU (the `Application` has already given the manager its `gpu::GPUDevice`),
and returns a `ModelHandle` — a plain `uint32_t` where `0` is
`INVALID_MODEL_HANDLE`. Handle lookup (`get_model(handle)`) is O(1); name
lookup (`get_model("hero")`) goes through a hash map and exists for
convenience. **Resolve names to handles once at spawn/placement time, not per
frame** — that is the documented intent of the handle API
(`src/bracken/scene/frame_packet.hpp`).

### The ModelManager

Declared in `src/bracken/model_loader.hpp`. Key facts:

- **Non-copyable, non-movable.** It owns GPU-backed models; handles must stay
  stable.
- Slot 0 is reserved for `INVALID_MODEL_HANDLE`; real models get handles
  1, 2, 3, … in registration order. Handles are never recycled (there is no
  per-model unload — only `unload_all()`).
- `load_model(name, path)` — parse a `.glb`, upload to GPU if a device is set,
  return the handle. Returns `INVALID_MODEL_HANDLE` on failure (error already
  logged under the `model` log category).
- `get_model(ModelHandle)` / `get_model(const std::string&)` — return
  `Model*`, or `nullptr` for invalid/unknown handles and names. Never crashes
  on a bad handle (verified by `src/bracken/model_loader.test.cpp`).
- `get_handle(name)` — name → handle, `INVALID_MODEL_HANDLE` if not found.
- `register_model(name, std::unique_ptr<Model>)` — hand the manager a model
  you built in code (procedural geometry). Takes ownership, uploads to GPU if
  a device is set, returns a handle.
- `reload_model(handle)` — re-parse the source file and swap the model in
  place; the handle stays valid. Returns `false` (and leaves the old model
  untouched) if the file fails to parse.
- `enable_hot_reload(watcher)` — watch every file-backed model for on-disk
  changes and reload automatically. `Application` wires this up for you when
  the `BRACKEN_HOT_RELOAD=1` environment variable is set.
- `set_device(gpu::GPUDevice*)` — normally called by the engine during
  `init_renderer`; you only call this yourself when using a `ModelManager`
  standalone.

Registering the same name twice is not an error: the name map is simply
re-pointed at the new handle. The old model stays alive and reachable through
its old handle, but is no longer reachable by name.

#### Registering procedural models

The playground (`examples/playground/playground.cpp`,
`PlaygroundGame::register_models`) is the canonical example — it builds *all*
of its assets in code:

```cpp
void PlaygroundGame::register_models() {
    auto& mm = models();
    player_model_ = mm.register_model("player", build_character_model());
    orb_model_ = mm.register_model("orb", build_orb_model());
    mm.register_model("orb_shard", build_shard_model()); // referenced by name from effects

    mm.register_model("tree_oak", procedural::TreeGenerator::generate_oak(7));
    mm.register_model("tree_pine", procedural::TreeGenerator::generate_pine(21));
    mm.register_model("rock_boulder", procedural::RockGenerator::generate_boulder(11));
}
```

A hand-built `Model` just needs `meshes` filled with vertices/indices, the
`min_*`/`max_*` bounds set, and `compute_bounding_sphere()` called (the
procedural generators in `src/bracken/procedural/` do all this). Procedural
models have no source path, so `reload_model` returns `false` for them and hot
reload skips them.

#### Hot reload

```bash
BRACKEN_HOT_RELOAD=1 ./build/examples/playground/playground
```

With the env var set, `Application` initializes the
`core::asset::FileWatcher` and calls
`ModelManager::enable_hot_reload(FileWatcher::instance())` during
`init_renderer` (`src/bracken/application.cpp`). Every `.glb` loaded before
*or after* that point is watched; saving the file re-parses it, uploads fresh
GPU buffers, frees the old ones, and swaps the model into the same handle
slot.

**Gotcha — cached `Model*` and hot reload:** `reload_model` replaces the
`Model` object behind the handle. A raw `Model*` you fetched earlier dangles
after a reload. Re-fetch via `get_model(handle)` each frame (cheap — a vector
index) rather than caching the pointer across frames when hot reload may be
active.

#### Using ModelLoader directly (no manager)

`ModelLoader` is three static functions; the manager is a convenience on top:

```cpp
bracken::Model model;
if (!bracken::ModelLoader::load_glb("assets/models/rock.glb", model)) {
    // parse failure — details already logged
}
bracken::ModelLoader::upload_to_gpu(device, model);   // needs a gpu::GPUDevice&
// ... draw ...
bracken::ModelLoader::free_gpu_resources(model);      // before device teardown
```

`load_glb` is `[[nodiscard]] static bool` — pure CPU parsing, safe without a
GPU (the tests call it headless). `upload_to_gpu` creates the vertex/index
buffers and textures and then **frees the CPU-side vertex, index, and pixel
data** to save memory. Consequences you must know:

- After upload, `Mesh::vertices` / `skinned_vertices` / `indices` are empty.
  `Mesh::index_count()` still works (it returns a cached count), but
  `Mesh::vertex_count()` returns 0.
- If you need CPU-side geometry (e.g. for collision), read it *before* calling
  `upload_to_gpu`, or keep your own copy.
- `Model::skeleton`, `Model::animations`, bounds, and the bounding sphere
  survive upload — animation and culling keep working.

### What the loader produces

`Model` (in `src/bracken/model_loader.hpp`) contains:

- `std::vector<Mesh> meshes` — one `Mesh` per glTF **primitive** (a glTF mesh
  with 3 primitives yields 3 engine meshes).
- Axis-aligned bounds `min_x/min_y/min_z` / `max_x/max_y/max_z` accumulated
  from every vertex position, plus helpers `width()`, `height()`, `depth()`,
  `max_dimension()`.
- A precomputed bounding sphere (`bounding_center`, `bounding_half_diag`) for
  frustum culling — `compute_bounding_sphere()` is called at the end of
  `load_glb`; call it yourself for hand-built models.
- Skeleton + animation clips when the glb has a skin: `skeleton`,
  `animations`, `has_skeleton`, `foot_ik`
  (types in `src/bracken/animation/animation_types.hpp`).
- `find_animation(name)` — linear search over clips, returns the clip index or
  `-1`:

```cpp
Model* m = models().get_model(player_model_);
int idle = m->find_animation("Idle");   // -1 if the glb has no such clip
```

#### Meshes

Each `Mesh` holds either `std::vector<Vertex3D> vertices` (static) or
`std::vector<SkinnedVertex> skinned_vertices` (when the primitive has
`JOINTS_0` + `WEIGHTS_0`; `is_skinned` is set). Vertex layouts are the
canonical ones from `src/bracken/gpu/gpu_types.hpp`: position, normal,
texcoord, color, tangent (`vec4`, glTF convention: xyz direction + bitangent
sign in `.w`), and for skinned vertices `uint8_t joints[4]` +
`float weights[4]`.

Per-mesh perf flags (set them on procedural meshes; the loader never sets
them from glTF):

- `cast_shadows` (default `true`) — `false` makes all shadow passes skip the
  mesh. The tree generator uses this for leaf canopies.
- `shadow_only` (default `false`) — `true` makes the *main* pass skip the mesh;
  only shadow passes render it. Used for low-poly canopy shadow proxies
  (`src/bracken/procedural/tree_generator.cpp`).
- `near_only` (default `false`) — declared as a "render only near the camera"
  hint but **not enforced by the renderer**. Setting it currently does
  nothing.

#### Materials

Each mesh gets a `PBRMaterial` mirroring glTF `pbrMetallicRoughness`:

- `base_color_factor` (vec4), `metallic_factor`, `roughness_factor`,
  `normal_scale`, `occlusion_strength`, `emissive_factor` (already multiplied
  by `KHR_materials_emissive_strength` when the extension is present).
- `alpha_mode` (`AlphaMode::Opaque` / `Mask` / `Blend`), `alpha_cutoff`
  (default 0.5), `double_sided`. These are honored by the renderer: `Mask`
  alpha-tests against the cutoff, `Blend` meshes go through the sorted
  transparent pass, `double_sided` selects a no-cull pipeline variant (see
  `docs/architecture.md`, "Assets & models"). Per glTF spec, `Opaque`
  materials never alpha-test.
- `has_packed_occlusion` — set when the glb's occlusion texture is the *same
  image* as the metallic-roughness texture (ORM packing); AO is then read from
  the R channel of the metallic-roughness texture.

Texture-wise, the loader picks up base color, metallic-roughness, normal, and
emissive maps. **A separate (non-ORM-packed) occlusion texture is unsupported:
it is warned about and dropped at load** — pack AO into the ORM texture (the
occlusion texture must share the metallic-roughness image) if you want it to
render. Base-color and emissive textures upload as sRGB
(`RGBA8_SRGB` — sampler linearizes on read); normal and metallic-roughness
upload as linear `RGBA8`. Mipmaps are generated for all of them. Images are
decoded via SDL3_image and converted to RGBA8, so any format SDL3_image
understands works — but there is a soft budget: images larger than
**2048×2048** log a warning at load (resize offline instead).

glTF texture *sampler* settings (wrap mode, filter) are ignored — the renderer
uses its own samplers.

How each material field behaves at shading time — and the silent behaviors
attached to each — is covered in depth in
[Model PBR Surface Shading](#model-pbr-surface-shading).

### The .glb authoring contract

Full contract: `docs/skinned_mesh_format.md`. The essentials:

- **glTF binary (`.glb`) only**, **Y-up**, `+Y up` / `-Z forward` (standard
  glTF). A mesh exported with this convention loads with no rotation applied.
- **No Draco.** The bundled tinygltf has no Draco decoder; a Draco-compressed
  glb fails to load.
- **Node transforms are not applied to mesh geometry.** `load_glb` iterates
  `gltf.meshes` directly and never walks the node hierarchy for static
  geometry — vertex positions are used exactly as authored in the accessors.
  Bake all object transforms into the geometry at export.
- Accessor data must be **tightly packed**. A bufferView with an interleaved
  `byteStride` gets the attribute skipped with a warning (and if it's
  `POSITION`, the whole primitive is skipped).
- Skinning: joint indices must be **u8** (`JOINTS_0` componentType
  UNSIGNED_BYTE) and weights **float**. Anything else logs a warning and the
  mesh ships *unskinned*. This pairs with the `MAX_BONES = 128` cap
  (`src/bracken/animation/animation_types.hpp`) — ≤128 joints per skin, which
  also keeps joint indices in 8 bits.
- Only the **first skin** (`skins[0]`) in the file becomes the model's
  skeleton.
- **Animation channels that target nodes outside the skin's joint list are
  silently dropped.** If a clip "doesn't play," check that its channels drive
  the skinned joints. A clip whose channels are *all* dropped disappears from
  `Model::animations` entirely.
- Clip names are matched by string by the host's `AnimationStateMachine`
  config — name clips (`Idle`, `Walk`, …) to match whatever the host loads.
- Foot IK resolves bones by name via `FootIKBoneNames` (defaults: `Hips`,
  `Spine`, `LeftUpperLeg`/`LeftLowerLeg`/`LeftFoot` + `Right*` mirrors). Other
  rigs work if the host supplies matching names.
- Author characters with **feet at the origin plane**; the engine places
  models on the ground using `min_y`.
- Missing attributes are tolerated: no `NORMAL` → normals default to
  `(0,1,0)`; no `TEXCOORD_0` → UVs `(0,0)`; no `COLOR_0` → vertex color is
  baked **white** (`baseColorFactor` is applied in the shader, so untextured
  meshes still shade as `vertexColor × baseColorFactor × tint`); no `TANGENT`
  → tangents are generated from UVs (see below). `COLOR_0` must be VEC4 (vec3
  colors are skipped with a warning).

### model_utils helpers

Declared in `src/bracken/model_utils.hpp`.

#### build_model_transform

Builds the standard model-to-world matrix the engine's conventions expect:

```cpp
glm::mat4 build_model_transform(const Model& model, const glm::vec3& position,
                                float yaw_radians, float target_size,
                                float attack_tilt = 0.0f);
```

It composes `translate(position) * rotate_Y(yaw) * [rotate_X(attack_tilt)] *
scale(uniform) * translate(-pivot)` where the pivot is the bounding-box center
in x/z and `min_y` in y. Two properties follow (verified in
`src/bracken/model_utils.test.cpp`):

- The model's **bottom-center lands exactly at `position`** — drop a model on
  terrain by passing the terrain height as `position.y`.
- The uniform scale is `(target_size * 1.5f) / model.max_dimension()`, i.e.
  `target_size` is a rough world-space size knob, not an exact extent (the
  ×1.5 fudge makes models read slightly larger than their box).

`attack_tilt` rotates around X *after* yaw — a lean-forward primitive for
attack animations. `0.0f` (the default) is a strict no-op.

You don't have to use it — the playground builds prop transforms manually with
`glm::translate/rotate/scale` — but it's the quickest way to get "put this
model at that spot on the ground, facing this way, about this big."

Watch out: `max_dimension()` on a default-constructed `Model` is 0, so the
scale divides by zero. Only call it on models with real bounds.

#### compute_tangents_from_uvs

```cpp
void compute_tangents_from_uvs(std::vector<gpu::Vertex3D>& vertices,
                               const std::vector<uint32_t>& indices);
```

Per-vertex tangent generation from UV gradients, packed into
`Vertex3D::tangent` as `(Tx, Ty, Tz, bitangent_sign)` per the glTF convention.
`load_glb` calls this automatically for any primitive missing `TANGENT`; call
it yourself when hand-building normal-mapped meshes. It is *not*
MikkTSpace-correct on UV seams (documented as fine for v1). Degenerate input
is safe: empty vertices/indices is a no-op, degenerate UV triangles are
skipped, and vertices that end up without a usable tangent get a deterministic
perpendicular fallback. Mirrored UVs correctly produce `tangent.w == -1`
(tests in `src/bracken/model_utils.test.cpp`).

### Heightmaps

`bracken::Heightmap` (`src/bracken/heightmap.hpp`) is a plain struct — there
is **no file format and no loader**; the host fills the fields itself
("game code converts from its own heightmap format to this", per the header).
It stores a square grid of 16-bit normalized heights:

- `resolution` — grid points per side (`height_data.size()` must be
  `resolution * resolution`, row-major, indexed `z * resolution + x`).
- `world_origin_x`, `world_origin_z`, `world_size` — where the grid sits on
  the XZ plane and how many world units it spans.
- `min_height`, `max_height` (defaults −500/+500) — the range the `uint16_t`
  values decode into: `height = raw / 65535.0 * (max - min) + min`.

The playground builds one procedurally
(`PlaygroundGame::build_heightmap`, `examples/playground/playground.cpp`):

```cpp
constexpr uint32_t RES = 257;
heightmap_.resolution = RES;
heightmap_.world_origin_x = 0.0f;
heightmap_.world_origin_z = 0.0f;
heightmap_.world_size = WORLD;
heightmap_.min_height = -100.0f;
heightmap_.max_height = 150.0f;
heightmap_.height_data.resize(RES * RES);
for (uint32_t z = 0; z < RES; ++z) {
    for (uint32_t x = 0; x < RES; ++x) {
        float h = /* any height function, in world units */;
        float normalized = (h - heightmap_.min_height) / (heightmap_.max_height - heightmap_.min_height);
        heightmap_.height_data[z * RES + x] = static_cast<uint16_t>(normalized * 65535.0f);
    }
}
```

#### Feeding the renderer

```cpp
set_heightmap(heightmap_);   // Application::set_heightmap(const Heightmap&)
```

`Application::set_heightmap` forwards to the `SceneRenderer`, which **deep
copies** the heightmap into the terrain renderer, uploads it as an `R16`
(16-bit UNORM) GPU texture, regenerates the terrain meshes (main + shadow
LOD), and hands the heightmap parameters to the grass renderer
(`src/bracken/scene/scene_renderer.cpp`, `SceneRenderer::set_heightmap`). You
may discard or mutate your copy afterwards — but keep it around if you want
CPU-side height queries, as the playground does.

#### Sampling

All sampling is on your CPU-side struct, header-only:

- `get_height_local(x, z)` — decode one grid point. **Out-of-range indices
  return `0.0f`** (not clamped, not an error — beware if your terrain's real
  height at the edge isn't 0).
- `get_height_world(world_x, world_z)` — bilinear interpolation between the
  four surrounding grid points; world coordinates outside the heightmap
  **clamp to the edge**. This is what you use for placing things on the
  ground:

```cpp
player_pos_.y = heightmap_.get_height_world(player_pos_.x, player_pos_.z);
```

- `get_normal_world(x, z, nx, ny, nz)` — unit surface normal from
  central-difference height taps one grid-cell apart; degenerate cases return
  straight up `(0,1,0)`.

There is also `Application::get_terrain_height(x, z)`, which queries the
renderer's own copy — the playground uses its local `Heightmap` for gameplay
and passes `get_terrain_height` to the camera controller for terrain
collision.

Precision note: 16-bit normalization gives you `(max_height − min_height) /
65535` vertical resolution — about 4 mm for a 250 m range. Don't set the range
wider than the terrain actually needs.

### Common mistakes checklist

- Feeding a Draco-compressed `.glb` — fails to load; there is no decoder.
- Expecting node/scene-graph transforms to position static meshes — they are
  ignored; bake transforms at export.
- Checking a bool return: both `ModelLoader::load_glb` and
  `ModelManager::reload_model` are `[[nodiscard]]` for a reason.
- Reading `mesh.vertices` after `upload_to_gpu` — the CPU copy is freed.
- Relying on `Mesh::near_only` — declared but not enforced.
- Shipping a separate AO texture — warned about and dropped at load; use ORM
  packing.
- Exporting non-u8 joints / non-float weights — mesh silently ships unskinned
  (with a log warning).
- Animation channels targeting non-skin nodes — silently dropped; a clip can
  vanish entirely.
- Naming clips differently from the host's state-machine config — the clip
  loads but nothing ever plays it.
- `Heightmap::get_height_local` out of range returns `0.0f`, not an edge
  clamp — use `get_height_world` for queries near/past the border.

---

## Skeletal Animation, State Machines & IK

Bracken's animation stack lives in `src/bracken/animation/` (namespace `bracken::animation`). It is **per-instance and host-driven**: the engine gives you data types and update functions; your game owns one `AnimationPlayer` (and usually one `AnimationStateMachine`) per animated entity, ticks them every frame, optionally layers procedural IK/lean on top, and submits the resulting bone matrices to the renderer.

The pieces, top to bottom:

| Piece | Header | What it does |
|---|---|---|
| `Skeleton`, `AnimationClip`, constants | `src/bracken/animation/animation_types.hpp` | shared, read-only animation data (usually loaded from a `.glb`) |
| `AnimationPlayer` | `src/bracken/animation/animation_player.hpp` | samples one clip (or a crossfade of two) into bone matrices |
| `AnimationStateMachine` | `src/bracken/animation/animation_state_machine.hpp` | data-driven state graph that drives the player from game parameters |
| `solve_two_bone_ik`, `apply_foot_ik`, `apply_body_lean` | `src/bracken/animation/ik_solver.hpp` | procedural post-passes on the player's output |
| `FootIKSmoother`, `RotationSmoother`, `FootIKBoneNames`, `ProceduralConfig` | `src/bracken/animation/animation_types.hpp` | small host-side helpers for the procedural layer |

The required per-frame order (from `docs/architecture.md`, "Frame data flow"):

```cpp
// host on_simulate — simulation first:
anim_sm_.update(anim_player_);                                  // 1. state machine picks the clip
anim_player_.update(model->skeleton, model->animations, sim.dt()); // 2. player samples the pose
// 3. optional procedural layers, in place:
//    apply_foot_ik(...); apply_body_lean(...);

// ...then the frame description (fb = sim.frame()):
auto pal = fb.alloc_bone_palette(joint_count);   // packet-owned palette memory
std::ranges::copy(anim_player_.bone_matrices().first(joint_count), pal.matrices.begin());
fb.draw_skinned(handle, transform, pal.id);
```

(The whole sequence runs inside the sim job — `AnimationPlayer`/IK are tier sim-stage, per-instance, and safe to fan out with `sim.parallel_for` when you have many characters, one player per entity.)

The canonical consumer is `examples/playground/` (`playground.cpp::update_player`, `setup_state_machine`); every snippet below is adapted from it or from the unit tests.

### Getting a skeleton and clips

#### From a `.glb` file

`ModelLoader` (`src/bracken/model_loader.cpp`) fills these fields on `bracken::Model` (`src/bracken/model_loader.hpp`):

```cpp
struct Model {
    // ...
    animation::Skeleton skeleton;                    // joints from gltf.skins[0]
    std::vector<animation::AnimationClip> animations;
    bool has_skeleton = false;
    animation::FootIKData foot_ik;                   // pre-resolved with default bone names

    int find_animation(const std::string& name) const;  // -1 if not found
};
```

The authoring contract is `docs/skinned_mesh_format.md`. The load-bearing points:

- **glTF binary (`.glb`)**, Y-up, `+Y up`/`-Z forward`, **no Draco** (the bundled tinygltf cannot decode it).
- **Joint count ≤ 128** (`MAX_BONES`, see below). `JOINTS_0` must be `u8` and `WEIGHTS_0` must be `float` — anything else logs a warning and the mesh ships **unskinned** rather than decoding garbage.
- **Clip names matter**: `AnimationStateMachine::bind_clips` matches clips to states **by string name**. Name your clips to match your state-machine config (the playground uses `Idle`, `Walk`, `Run`, `Attack`).
- Animation channels that target nodes **outside the skin's joint list are silently dropped** at load. If a clip "doesn't play", check that its channels drive the skinned joints. A clip whose channels are all dropped is not added to `Model::animations` at all.
- Only `translation` / `rotation` / `scale` channel paths are read; clip duration is the max keyframe time seen across its channels.
- Sampler interpolation: `LINEAR` (default) and `STEP` are honored. `CUBICSPLINE` is unsupported — its keys are degraded to linear by reading the middle output element per key, with a warning (once per model).
- Out-of-order skins (a joint's parent appearing after it in `skin.joints`) are **normalized at load** with a stable topological reorder; vertex joint indices are remapped to match. Parents-before-children ordering is therefore guaranteed for any skeleton from `load_glb`.

#### Building animation data in code

Nothing forces you through the loader. `Skeleton`, `Joint`, `AnimationClip`, and `AnimationChannel` are plain structs; the playground builds its entire character — skeleton, skinned meshes, and four clips — in `examples/playground/character.cpp`. A minimal clip (from `animation_player.test.cpp`):

```cpp
using namespace bracken::animation;

AnimationClip clip;
clip.name = "test_clip";
clip.duration = 1.0f;

AnimationChannel ch;
ch.bone_index = 0;                              // index into Skeleton::joints
ch.position_times = {0.0f, 1.0f};
ch.positions = {glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f)};
ch.rotation_times = {0.0f};
ch.rotations = {glm::quat(1.0f, 0.0f, 0.0f, 0.0f)};
ch.scale_times = {0.0f};
ch.scales = {glm::vec3(1.0f)};
clip.channels.push_back(ch);
```

A joint with no channel in the current clip keeps its bind-pose local transform (`Joint::local_translation/rotation/scale`).

**Gotcha — joint ordering:** `AnimationPlayer::compute_bone_matrices` propagates parent transforms in a single index-order pass, so every joint's `parent_index` must be **less than its own index** (parents before children). Skeletons from `ModelLoader::load_glb` are guaranteed to satisfy this (out-of-order skins are topologically normalized at load); hand-built skeletons must satisfy it themselves.

#### `MAX_BONES` — what the 128 cap actually caps

```cpp
// src/bracken/animation/animation_types.hpp
constexpr int MAX_BONES = 128;
```

`MAX_BONES` sizes the **CPU-side** per-instance arrays: `AnimationPlayer` holds `std::array<glm::mat4, MAX_BONES>` for bone matrices and world transforms (2 × 8 KB per player). Joints beyond index 127 are **silently ignored** at playback — no warning at load or at update. The skinned shaders read a per-frame `StructuredBuffer<float4x4>` of tightly packed palettes indexed by a `paletteOffset` uniform, so there is **no GPU-side bone-count constant** to keep in sync. Don't raise `MAX_BONES` without auditing every `std::array<glm::mat4, MAX_BONES>` user (see `CLAUDE.md`).

Skinned vertices carry exactly 4 influences — the layout is `uint8_t joints[4]; float weights[4];` in `gpu::SkinnedVertex` (`src/bracken/gpu/gpu_types.hpp`); there is no separate influence-count constant.

### Playing clips: `AnimationPlayer`

One player per animated instance. The player is **move-only** — copying a player used to share keyframe cursors with the original and caused animation glitches, so the copy constructor is deleted (`src/bracken/animation/animation_player.hpp`). Clips and skeletons are shared, read-only inputs; all mutable playback state (time, blend, per-channel cursors) lives on the player.

```cpp
#include "bracken/animation/animation_player.hpp"

bracken::animation::AnimationPlayer player;   // defaults: clip 0, t=0, playing,
                                              // looping, speed 1.0, no blend

// every frame:
player.update(model->skeleton, model->animations, dt);

// results:
std::span<const glm::mat4> palette = player.bone_matrices();   // always MAX_BONES long
size_t used = player.used_joint_count();                       // slots >= used are identity
```

Controls (all in `animation_player.hpp`):

| Call | Default | Effect |
|---|---|---|
| `set_speed(float)` / `speed()` | `1.0f` | time scale for both the current and (mid-blend) previous clip |
| `set_playing(bool)` / `is_playing()` | `true` | when false, `update()` returns immediately — the pose freezes |
| `set_loop(bool)` / `loops()` | `true` | looping wraps time with `fmod`; non-looping clamps at the clip end and sets `playing = false` |
| `seek(float)` / `current_time()` | `0.0f` | raw time write, no clamping |
| `crossfade_to(int clip_index, float duration = 0.2f)` | — | start a blend to another clip |
| `reset()` | — | back to defaults; all bone matrices become identity |

Behavior worth knowing (all covered by `animation_player.test.cpp`):

- `update()` does nothing when `clips` is empty or the player is paused.
- An out-of-range current clip index is reset to `0` on the next `update()`.
- A **zero-duration clip** (single-keyframe pose) sets `time = 0` and `playing = false` — it renders one static pose and stops.
- When a **non-looping** clip reaches its end, time clamps to the duration and `is_playing()` becomes `false`. The state machine uses exactly this as its "clip ended" signal.

#### Crossfading

```cpp
int attack = model->find_animation("Attack");
if (attack >= 0) {
    player.set_loop(false);
    player.crossfade_to(attack, 0.08f);   // 80 ms linear blend
}
```

- The new clip starts at `t = 0`; the previous clip keeps advancing (and wraps at its end) so the outgoing motion doesn't freeze mid-blend.
- The blend factor advances linearly by `dt / duration` — note it uses **raw `dt`**, not `dt * speed`.
- Both clips are sampled and blended per joint (`glm::mix` for translation/scale, `glm::slerp` for rotation) until the factor reaches 1, then the previous clip is dropped.
- **`crossfade_to` with the current clip index is a no-op** — time is not reset. If you want to restart the same clip from the top, `seek(0.0f)` instead.
- Starting a new crossfade while one is in progress discards the old previous pose (there is no three-way blend) — the current in-blend pose is not captured, so a fast double-transition can pop.

`crossfade_to` only switches clips; it does **not** touch loop/speed/playing. The state machine sets those per state (see below); do the same when driving the player manually.

#### Output spans and their lifetime

```cpp
std::span<const glm::mat4> bone_matrices() const;      // world * inverse_bind, palette for skinning
std::span<const glm::mat4> world_transforms() const;   // model-space joint transforms
std::span<glm::mat4>       mutable_bone_matrices();    // for IK passes
std::span<glm::mat4>       mutable_world_transforms();
size_t used_joint_count() const;                       // written prefix; the tail is identity
```

All spans view storage **owned by the player** and are always `MAX_BONES` (128) entries long; only the first `used_joint_count()` entries are live, the rest are identity. They stay valid as long as the player object lives and isn't moved-from — which is exactly what makes the render submission below work.

### Driving the player: `AnimationStateMachine`

The state machine is data-driven: you describe states (name → clip name, loop flag, playback speed) and transitions (target state, conditions, crossfade duration, priority), bind it to a clip list once, then feed it parameters every frame.

#### Building the graph

Adapted from `examples/playground/playground.cpp::setup_state_machine`:

```cpp
#include "bracken/animation/animation_state_machine.hpp"
namespace anim = bracken::animation;
using TC = anim::TransitionCondition;

auto cond = [](const char* name, TC::Op op, float threshold = 0.0f) {
    TC c; c.param_name = name; c.op = op; c.threshold = threshold; return c;
};
auto to = [](const char* target, TC c, float xfade, int priority = 0) {
    anim::StateTransition t;
    t.target_state = target;
    t.conditions.push_back(std::move(c));
    t.crossfade_duration = xfade;
    t.priority = priority;
    return t;
};
const auto attack_now = [&] { return to("Attack", cond("attacking", TC::Op::IS_TRUE), 0.08f, 10); };

anim::AnimState idle;
idle.name = "Idle";
idle.clip_name = "Idle";            // matched by string against AnimationClip::name
idle.transitions.push_back(attack_now());
idle.transitions.push_back(to("Walk", cond("speed", TC::Op::GT, 20.0f), 0.15f));

anim::AnimState walk;
walk.name = "Walk"; walk.clip_name = "Walk";
walk.transitions.push_back(attack_now());
walk.transitions.push_back(to("Run",  cond("speed", TC::Op::GT, 240.0f), 0.15f));
walk.transitions.push_back(to("Idle", cond("speed", TC::Op::LT, 20.0f),  0.2f));

anim::AnimState attack;
attack.name = "Attack"; attack.clip_name = "Attack";
attack.loop = false;                // one-shot
attack.transitions.push_back(to("Idle", cond("attacking", TC::Op::IS_FALSE), 0.2f));

anim_sm_.add_state(std::move(idle));
anim_sm_.add_state(std::move(walk));
anim_sm_.add_state(std::move(attack));
anim_sm_.set_default_state("Idle");
if (!anim_sm_.bind_clips(model->animations)) {
    // some state's clip_name wasn't found in the clip list
}
```

State knobs (`AnimState` in `animation_state_machine.hpp`): `loop` (default `true`) and `speed` (default `1.0f`) are pushed onto the player on state entry, along with `set_playing(true)` and a `crossfade_to(clip, transition.crossfade_duration)`.

Condition operators (`TransitionCondition::Op`): `GT`, `LT`, `EQ`, `NE` (floats; `EQ`/`NE` use a fixed `0.001` epsilon), `IS_TRUE`, `IS_FALSE` (bools). **Types are strict**: `GT` on a parameter currently holding a bool evaluates `false`, and `IS_TRUE` on a float evaluates `false` — no coercion.

Transition logic per transition: all entries in `conditions` must pass (AND), and — only if `any_of` is non-empty — at least one entry in `any_of` must also pass (OR). Transitions are evaluated in **descending `priority`** (sorted once in `add_state`); the first passing transition fires and the update returns, so **at most one transition fires per `update()` call**.

`bind_clips(clips)` resolves clip names → indices and state names → indices, registers every condition's parameter, and enters the default state. It returns `false` if any state's clip was not found (`[[nodiscard]]`) — but the machine still becomes bound and states with missing clips simply can never be entered (`enter_state` bails when `clip_index < 0`). `add_state` with an existing name replaces that state, so you can rebuild and re-`bind_clips` (hot-reload friendly).

#### Feeding parameters and updating

Parameters are name-keyed values you set from gameplay. The string API hashes the name on every call; on per-frame paths resolve the index once and use the index overloads:

```cpp
// setup (after bind_clips — conditions already registered "speed"/"attacking"):
param_speed_    = anim_sm_.param_index("speed");      // creates if absent
param_attacking_ = anim_sm_.param_index("attacking");

// every frame, before updating the player:
anim_sm_.set_float(param_speed_, player_speed_);
anim_sm_.set_bool(param_attacking_, attack_timer_ > 0.0f);
anim_sm_.update(anim_player_);                        // evaluates transitions, drives the player
```

API summary (`animation_state_machine.hpp`):

- `param_index(name)` — resolve-or-create, returns a stable index.
- `find_param_index(name)` — lookup only, `-1` if absent.
- `set_float/set_bool/get_float/get_bool` — string overloads (string setters **create** the parameter; getters don't) and index overloads (out-of-range indices are safely ignored on set, return `0.0f`/`false` on get — they never create).
- `current_state()` — current state name, `""` when unbound.
- `is_bound()`.

Setting a float on a parameter that previously held a bool re-types it (and vice versa) — the old typed value is gone.

#### The clip-ended rule (one-shot states)

When the current state is non-looping and its clip has finished (`!state.loop && !player.is_playing()`), **every transition passes unconditionally** — the highest-priority transition out of the state fires regardless of its conditions. That's how one-shots like `Attack` return to locomotion even if the trigger parameter is still set. Consequence: a non-looping state should list its intended "exit" transition first (highest priority), and a non-looping state with **no** transitions freezes on the clip's last frame.

`update()` before `bind_clips` is a safe no-op, and `current_state()` returns `""`.

### Foot IK, two-bone IK, and body lean

The IK functions in `src/bracken/animation/ik_solver.hpp` are **procedural post-passes**: call them after `AnimationPlayer::update` and before render submission. They modify the player's bone matrices and world transforms **in place** through the mutable spans, and they keep the invariant `bone_matrix = world_transform * inverse_bind_matrix` for every joint they touch. The spans must point to contiguous storage of at least `MAX_BONES` matrices — `mutable_bone_matrices()` / `mutable_world_transforms()` satisfy this by construction.

#### Bone-name conventions: `FootIKBoneNames` / `FootIKData`

Foot IK resolves joints **by exact name** at setup time:

```cpp
// src/bracken/animation/animation_types.hpp — the defaults:
struct FootIKBoneNames {
    std::string hips = "Hips";
    std::string spine = "Spine";
    std::string left_hip = "LeftUpperLeg";
    std::string left_knee = "LeftLowerLeg";
    std::string left_foot = "LeftFoot";
    std::string right_hip = "RightUpperLeg";
    std::string right_knee = "RightLowerLeg";
    std::string right_foot = "RightFoot";
};
```

`ModelLoader` already calls `model.foot_ik.init(model.skeleton)` with the defaults after loading a skeleton. `FootIKData::valid` is `true` only if **all eight** names resolved. For rigs with different naming (e.g. raw Mixamo `mixamorig:`-prefixed bones — the header comment says the defaults "match Mixamo conventions", but raw Mixamo exports are prefixed, so per `docs/skinned_mesh_format.md` you must override), re-init with your own names:

```cpp
bracken::animation::FootIKBoneNames names;
names.hips = "mixamorig:Hips";
names.spine = "mixamorig:Spine";
names.left_hip = "mixamorig:LeftUpLeg";
// ... all eight ...
model->foot_ik.init(model->skeleton, names);
```

Passing an invalid `FootIKData` to `apply_foot_ik` is a guaranteed no-op (the indices are `-1`; indexing with them would be UB, so the function hard-bails). Check `foot_ik.valid` yourself if you want to log it.

#### Terrain foot placement: `apply_foot_ik` + `FootIKSmoother`

The full playground pattern (`examples/playground/playground.cpp::update_player`):

```cpp
Model* model = models().get_model(player_model_);
anim_player_.update(model->skeleton, model->animations, dt);

const auto& ik = model->foot_ik;
if (ik.valid) {
    auto world = anim_player_.world_transforms();
    auto foot_offset = [&](int joint) {
        glm::vec3 foot = glm::vec3(player_transform_ * world[joint][3]);
        return get_terrain_height(foot.x, foot.z) - player_pos_.y;   // world-space y error
    };
    foot_smoother_.update(foot_offset(ik.left_foot), foot_offset(ik.right_foot), dt);
    anim::apply_foot_ik(anim_player_.mutable_bone_matrices(), anim_player_.mutable_world_transforms(),
                        model->skeleton, ik, /*scale=*/1.0f,
                        foot_smoother_.smoothed_left_offset, foot_smoother_.smoothed_right_offset);
}
```

Semantics of `apply_foot_ik(bones, world, skeleton, ik, scale, left_offset, right_offset)`:

- The offsets are **world-space vertical corrections** per foot (positive = foot should be higher). `scale` (the instance's uniform world scale) converts them into model space (`offset / scale`) — the instance transform itself never participates.
- **Deadzone:** if both offsets are within ±`0.1` (world units), the whole pass is skipped. Offsets are clamped to ±`8.0`. Both constants are absolute world-space values baked into `ik_solver.cpp` (`IK_THRESHOLD`, `IK_MAX_CORRECTION`) — if your world uses very small units, this deadzone may swallow all corrections.
- If the lower correction is negative (a foot needs to go **down**), the whole skeleton is dropped by that amount first (pelvis drop), then the other leg is IK'd back up. Positive-only offsets (stepping up) solve each leg individually without moving the pelvis.

`FootIKSmoother` (`animation_types.hpp`) is a simple exponential smoother (rate `15.0` by default, tweak `smooth_rate`) that prevents foot targets from snapping when the terrain height under a foot changes discontinuously. Keep one per character, call `update(left_target, right_target, dt)` each frame, read the `smoothed_*_offset` fields.

#### Body lean: `apply_body_lean`

Rotates the spine joint and **all of its descendants** around the spine's position — forward lean about +X, lateral lean about +Z (radians). Joints outside the spine subtree (hips, legs) are untouched. Values inside ±`0.001` are a no-op.

```cpp
float forward_lean = std::min(player_speed_ * 0.0004f, 0.15f);
float lateral_lean = std::clamp(-facing_.turn_rate * 0.04f, -0.12f, 0.12f);
anim::apply_body_lean(anim_player_.mutable_bone_matrices(), anim_player_.mutable_world_transforms(),
                      model->skeleton, ik.spine, forward_lean, lateral_lean);
```

`ProceduralConfig` (`animation_types.hpp`) is a convenience struct of host-side tuning knobs for this layer (`foot_ik`/`lean` toggles, `forward_lean_factor 0.015`, `forward_lean_max 0.18`, `lateral_lean_factor 0.06`, `lateral_lean_max 0.15`, `attack_tilt_max 0.4`, `attack_tilt_cooldown 0.5`). Note the engine itself never reads it — it exists for your code to consume; the playground uses its own inline constants instead.

`RotationSmoother` (`animation_types.hpp`) is the companion facing-yaw helper: `smooth_toward(target_yaw, dt, speed = 12.0f)` takes the wrap-aware shortest path and tracks `turn_rate` (rad/s), which the playground feeds into lateral lean; call `decay_turn_rate()` on frames where you don't steer.

#### Custom chains: `solve_two_bone_ik`

For anything that isn't a leg (arms reaching, look-at chains), call the low-level solver directly:

```cpp
using namespace bracken::animation;
solve_two_bone_ik(player.mutable_bone_matrices(), player.mutable_world_transforms(),
                  model->skeleton,
                  upper_idx, lower_idx, end_idx,   // e.g. shoulder, elbow, hand joint indices
                  target,                          // model-space position for the end effector
                  pole_hint);                      // model-space point the middle joint bends toward
```

It is an analytic two-bone solver: preserves both bone lengths, clamps unreachable targets onto the reach sphere (with a 0.01 margin at full extension and minimum compression), bends toward the pole hint (falling back to the current bend direction if the hint is collinear with the chain), and translates direct children of the end effector along with it. All degenerate inputs — zero-length bones, non-finite matrices/targets, no derivable bend plane — make it a **silent no-op** that leaves the pose untouched (see `ik_solver.test.cpp` for the exact cases). Targets and pole hints are in **model space** (the same space as `world_transforms()`), not world space.

### Submitting the pose to the renderer

```cpp
// on_simulate, after the animation update (fb = sim.frame()):
auto pal = fb.alloc_bone_palette(joint_count);   // 1..=MAX_BONES; uninitialized packet memory
std::ranges::copy(anim_player_.bone_matrices().first(joint_count), pal.matrices.begin());
fb.draw_skinned(player_model_, player_transform_, pal.id);
```

The palette is **copied into packet memory** (`FrameBuilder::alloc_bone_palette`, `src/bracken/scene/frame_builder.hpp`); the draw references it by `BonePaletteId`. The contract (also in `docs/architecture.md`):

- **No lifetime rule on your storage.** The old non-owning span over `AnimationPlayer::bone_matrices()` is gone — once the matrices are copied into `pal.matrices` you may destroy, move, or re-`update()` the player freely.
- Allocate exactly the joints you use (`joint_count = min(model joint count, MAX_BONES)`); palettes are packed tightly, and the renderer clamps drawn joints to `min(model joints, allocation size)`.
- Write **every** matrix of the allocation before `seal()` — the memory is uninitialized.
- Budget: **`MAX_BONE_PALETTES = 1024`** palettes per frame (`src/bracken/scene/frame_packet.hpp`), enforced in `alloc_bone_palette` — over-budget allocations log an error and return an invalid allocation (`valid() == false`); draws with an invalid id are dropped with a warning. One palette can back many draws — reuse the id instead of allocating per draw.
- A command whose model has no skeleton, or whose palette id is invalid, is skipped (not drawn as skinned).

### Per-frame checklist

1. Set state-machine parameters (`set_float`/`set_bool`, index overloads on hot paths).
2. `anim_sm_.update(anim_player_)` — picks/crossfades the clip.
3. `anim_player_.update(skeleton, clips, dt)` — samples the pose.
4. Optional, in this order: `apply_foot_ik(...)`, then `apply_body_lean(...)` (both via the `mutable_*` spans).
5. `auto pal = fb.alloc_bone_palette(n)`; copy the pose into `pal.matrices`; `fb.draw_skinned(handle, transform, pal.id)`.

### Gotchas and common mistakes

- **Skipping the palette copy** — `alloc_bone_palette` returns *uninitialized* memory; a palette you never wrote skins with garbage.
- **Sharing one `AnimationPlayer` across instances** makes them all play the same pose and fights over cursors. One player per instance; players are move-only for exactly this reason. Ticking *different* players from worker threads is safe (verified by `animation_parallel.test.cpp`), but the render path itself is main-thread.
- **Clip name mismatches** are the top "nothing plays" cause: `bind_clips` returns `false` but still binds; the affected states silently never activate. Check the return value.
- **A transition targeting a missing state** (or a state whose clip didn't bind) passes its conditions, consumes the update, and then does nothing — it also shadows lower-priority transitions that frame. Keep state/clip names consistent.
- **Parameter type strictness**: `IS_TRUE` on a float param (or `GT` on a bool) is always false. Parameters created implicitly by `bind_clips` conditions start as `Float 0.0`.
- **Non-looping states auto-exit through their first transition** when the clip ends, conditions ignored. Order transitions accordingly.
- **`crossfade_to` to the already-current clip is a no-op** — it will not restart the clip.
- **Joints beyond 128 silently don't animate**; keep rigs ≤ 128 joints (`docs/skinned_mesh_format.md` checklist).
- **IK thresholds are absolute**: the 0.1 deadzone / ±8 clamp in `apply_foot_ik` assume world-scale units like the playground's (~180-unit-tall character).
- Foot IK does nothing at all when `FootIKData::valid == false` — override `FootIKBoneNames` for non-default rigs, or you'll get a silent no-op.

---

## Terrain & Grass Rendering

Bracken renders the ground as a single heightmap-driven terrain mesh (`bracken::render::TerrainRenderer`, `src/bracken/render/terrain_renderer.hpp`) and a GPU-instanced procedural grass layer (`bracken::render::GrassRenderer`, `src/bracken/render/grass_renderer.hpp`). Both live in the `bracken::render` namespace and are **owned and driven by `scene::SceneRenderer`** — in normal use you never construct them yourself. You feed a `bracken::Heightmap`, control the per-frame draw flags on your `FrameBuilder`, and tune a handful of public knobs through the accessors `SceneRenderer::terrain()` and `SceneRenderer::grass()`.

### Quick start: getting ground and grass on screen

The minimal path, adapted from the canonical consumer `examples/playground/playground.cpp`:

```cpp
#include "bracken/application.hpp"
#include "bracken/heightmap.hpp"

bool MyGame::on_init() {
    // World extent in world units (defaults to 32000 x 32000 if omitted).
    if (!init_renderer(1600, 900, "my game", 8000.0f, 8000.0f)) {
        return false;
    }

    // Build a heightmap and hand it to the renderer.
    bracken::Heightmap hm;
    hm.resolution = 257;                 // grid points per side
    hm.world_origin_x = 0.0f;
    hm.world_origin_z = 0.0f;
    hm.world_size = 8000.0f;             // world units covered by the grid
    hm.min_height = -100.0f;             // range used for 16-bit encoding
    hm.max_height = 150.0f;
    hm.height_data.resize(257 * 257);    // uint16_t, row-major, z-major rows
    // ... fill height_data (see "Authoring heightmap data" below) ...

    set_heightmap(hm);                   // Application forwards to SceneRenderer
    return true;
}

void MyGame::on_simulate(bracken::SimFrame& sim) {
    auto& fb = sim.frame();
    fb.set_draw_ground(true);            // default true each frame; set false to hide terrain
    fb.set_draw_grass(true);             // ditto for grass
}
```

`Application::set_heightmap` (`src/bracken/application.hpp`) forwards to `SceneRenderer::set_heightmap`, which:

1. deep-copies the heightmap into the terrain renderer (CPU-side, for height queries),
2. uploads it as an `R16` GPU texture,
3. regenerates the terrain mesh **and** the low-LOD shadow mesh,
4. wires the same GPU heightmap texture plus a `render::HeightmapParams` (origin/size/height range) into the grass renderer, so grass blades sit on the terrain surface.

Both draw flags reset to `true` at the start of every packet (`scene::FramePacket`), so terrain and grass draw unless you turn them off. Grass additionally requires `GraphicsSettings::grass_enabled` (default `true`).

### Authoring heightmap data

`bracken::Heightmap` (`src/bracken/heightmap.hpp`) is a plain struct — no loader, no file format. Game code converts from its own representation (the full `Heightmap` reference — fields, sampling helpers, precision notes — is in [Model Loading, Heightmaps & the glTF Contract](#model-loading-heightmaps--the-gltf-contract)). Heights are stored as **16-bit normalized** values over `[min_height, max_height]`:

```cpp
// Real height = (uint16 / 65535.0) * (max_height - min_height) + min_height
float norm = (height - hm.min_height) / (hm.max_height - hm.min_height);
hm.height_data[z * hm.resolution + x] = static_cast<uint16_t>(norm * 65535.0f);
```

The playground fills it procedurally (`PlaygroundGame::build_heightmap` in `examples/playground/playground.cpp`) with layered sines — a good template for test terrain. `src/bracken/heightmap.test.cpp` shows encode/decode round-trips and every query edge case.

CPU-side queries (useful for placement, character grounding, camera collision):

```cpp
float y = hm.get_height_world(x, z);      // bilinear; clamps outside bounds
float nx, ny, nz;
hm.get_normal_world(x, z, nx, ny, nz);    // central-difference, unit length
```

After the heightmap is installed, the same queries are available through the renderer without keeping your own copy:

```cpp
float y = get_terrain_height(x, z);                    // Application helper
float y2 = scene_renderer().terrain().get_height(x, z);
glm::vec3 n = scene_renderer().terrain().get_normal(x, z);
```

Before a heightmap is set, `TerrainRenderer::get_height` returns `0.0f` and `get_normal` returns `(0,1,0)` — terrain renders as a flat plane. Note `Heightmap::get_height_local` returns `0.0f` for out-of-range indices, while `get_height_world` clamps to the edge; don't confuse the two.

The playground also feeds terrain height to the camera so it can't clip through hills:

```cpp
camera().set_terrain_height_func([this](float x, float z) {
    return get_terrain_height(x, z);
});
```

### Terrain materials and the splatmap

The terrain fragment shader blends **4 material layers — grass, dirt, rock, sand — in that layer order**, weighted by an RGBA *splatmap* (R=grass, G=dirt, B=rock, A=sand). At `TerrainRenderer::init` the renderer tries to load, from the process working directory:

- `assets/textures/grass_seamless.png`, `dirt_seamless.png`, `rock_seamless.png`, `sand_seamless.png` — must all have identical dimensions; they become one sRGB, mipmapped `Texture2DArray`. Paths are anchored to the executable's directory (`SDL_GetBasePath`, same policy as the compiled shaders), not the process CWD.
- `assets/textures/terrain_splatmap.png` — linear RGBA control mask, no mipmaps.

The engine **ships no assets**, so both have procedural fallbacks: a 64×64 noise-shaded color per layer, and a 4×4 all-grass splatmap (`R=255` everywhere). You get a plausible green world out of the box; failures log via `SDL_Log` and never abort.

To update the splatmap at runtime (e.g. an editor painting terrain):

```cpp
// data = resolution*resolution RGBA8 pixels (4 bytes each)
scene_renderer().terrain().update_splatmap(pixels.data(), resolution);
```

`update_splatmap` **recreates** the splatmap texture. It silently returns if `data` is null or the renderer isn't initialized; a failed texture creation only logs. Grass splat masking is unaffected by the recreation — `GrassRenderer::set_splatmap` takes a *provider callable* fetched each frame, so it can never dangle (see below).

Splatmap UVs are derived from a `splat_params` uniform (`xy` = region origin in world XZ, `zw` = region X/Z size), which the engine populates as origin (0,0) with the full world width/height — non-square worlds map correctly.

### Terrain configuration knobs

All per-frame environment and lighting plumbing is done by `SceneRenderer` — you normally configure it via `GraphicsSettings` and `EnvironmentState`, not on the terrain renderer directly:

| Knob | Where | Default | Effect |
|---|---|---|---|
| `ambient_strength`, `sun_intensity` | `GraphicsSettings` (`src/bracken/contracts/graphics_settings.hpp`), forwarded via `TerrainRenderer::set_lighting_tunables` | 0.45 / 3.0 (settings); renderer-side defaults 0.5 / 1.0 | shadow-region lift, sun multiplier |
| fog + sun/ambient colors | `render::EnvironmentState` (`src/bracken/contracts/environment.hpp`), forwarded via `set_environment` once per frame | see header | atmospheric fog, hemisphere ambient |
| `anisotropic_filter` | `GraphicsSettings` int 0–4 → `SceneRenderer::set_anisotropic_filter` maps to 1/2/4/8/16× → `TerrainRenderer::set_anisotropic_filter(float)` | 4 (=16×); the terrain's init-time sampler is 8× | material texture sharpness at grazing angles |
| `draw_distance` | `GraphicsSettings` (index 0–6 → 500…32000) | 3 (=4000) | also caps grass view distance |

`TerrainRenderer::set_anisotropic_filter` clamps to `[1, 16]` and recreates the terrain material sampler.

If you drive `TerrainRenderer` directly (custom render loop), call `set_environment(env)` once per frame; the struct is copied by value. The `set_cluster_lighting(...)` pointers, however, are **non-owning and the `params` pointer is read at `render` time** — the pointed-to memory must stay valid through `render_frame` (the `SceneRenderer` keeps it in a frame-lifetime member).

### Terrain LOD, tiling and culling (what you get for free)

You don't configure any of this, but it explains the performance profile:

- The mesh is a uniform grid, cell size **25 world units**, extended by a **500-unit margin ring** past the world bounds for visual continuity.
- It is chunked into **32×32-cell tiles (800 world units)**, each with an index range and world-space AABB. Every pass frustum-culls tiles — first whole *rows* of tiles via a merged row AABB, then per-tile — and merges adjacent visible tiles into single draw calls. Typically 80–95% of the world is culled per pass.
- Shadow passes use a separate **low-LOD mesh** (100-unit cells, ~16× fewer triangles) sampled conservatively (min over a 5×5 probe window, sunk 5 units) so the simplified mesh never floats above real terrain and casts phantom shadows.

There is no distance-based LOD on the main terrain mesh — only culling.

### Grass: how it works and what you can tune

Grass is fully procedural: no placement data, no per-blade memory. The world is divided into **32×32-unit chunks of 12×12 = 144 blades**; each frame the CPU culls chunks around the camera (distance + frustum) and uploads visible chunk origins to a storage buffer; the vertex shader (`src/bracken/shaders/grass.vert.hlsl`) derives every blade's position, size, yaw, curve and wind sway from a stable world-space hash, and reads the blade's ground height from the terrain heightmap texture. Distribution inside a chunk is a jittered grid; a low-frequency "clump" hash varies density continuously so there is no pop-in.

Public knobs, on `render::GrassRenderer` via `scene_renderer().grass()` (may be null if grass failed to init):

```cpp
if (auto* grass = scene_renderer().grass()) {
    grass->wind_magnitude = 1.2f;              // default 0.8; strength of the sway
    grass->wind_direction = {0.2f, 1.0f};      // world-XZ; normalized at upload
    grass->grass_view_distance = 800.0f;       // default 1200 (MAX_VIEW_DISTANCE)
    grass->edge_margin = 50.0f;                // grass-free ring inside the world edges; default 0
    grass->set_exclusion_rects({{min_x, min_z, max_x, max_z}}); // no-grass zones (towns, roads); default none
    // ambient_strength (0.5) / sun_intensity (1.0) are overwritten from
    // GraphicsSettings every apply_graphics_settings call.
}
```

Notes on those fields:

- **`grass_view_distance` is respected** (default 1200). The effective distance each frame is `min(grass_view_distance, MAX_VIEW_DISTANCE = 1200, view-distance cap)`; in the normal pipeline `SceneRenderer` sets the cap to `GraphicsSettings::get_draw_distance()` via `set_view_distance_cap` (a cap ≤ 0 disables it).
- Wind **direction** is a world-XZ vector, normalized at upload (a zero vector disables directional sway; default `{1.0, 0.3}`). `wind_magnitude` scales the shader's layered sine waves (frequencies 2.0 and 3.7) with spatial phase so the meadow ripples rather than swaying uniformly.
- Blades fade smoothly to zero over the **last 30%** of the view distance (no density bands, no pop).

Grass **receives** cascaded shadows, clustered lights, and local light shadows but **does not cast shadows** (neither in the CSM nor the local-light passes) — there is no grass shadow pass.

#### Splat-aware grass placement (opt-in)

The grass shader can suppress blades where the terrain splatmap's grass channel (R) is weak — dirt paths, rock, sand stay bare. **This is not wired automatically**: `SceneRenderer` never calls `GrassRenderer::set_splatmap`, and neither does the playground. To enable it:

```cpp
auto& sr = scene_renderer();
if (auto* grass = sr.grass()) {
    grass->set_splatmap([&sr] { return sr.terrain().splatmap_texture(); });
}
```

`set_splatmap` takes a **provider callable**, fetched each frame at draw time — so the terrain recreating its splatmap texture (`update_splatmap`) can never leave grass with a dangling pointer. Pass an empty/null-returning provider to disable masking. Blades appear where the R weight passes `smoothstep(0.15, 0.5, weight)`.

#### Excluding areas from grass (opt-in)

There are no hardcoded clearings — by default grass grows everywhere inside the world bounds. Two host knobs control exclusions, both chunk-granular (`CHUNK_SIZE = 32`):

- `grass->edge_margin` (default 0) — a grass-free ring inset from the world edges.
- `grass->set_exclusion_rects(...)` — world-space XZ rectangles (towns, roads, arenas) where chunks whose center falls inside are culled; `exclusion_rects()` reads them back.

### Driving the renderers directly (custom render loops)

If you bypass `SceneRenderer`, the contracts are:

```cpp
render::TerrainRenderer terrain;
render::GrassRenderer grass;
bool ok = terrain.init(device, pipeline_registry, world_w, world_h);   // [[nodiscard]]
ok = grass.init(device, pipeline_registry, world_w, world_h);          // [[nodiscard]]

terrain.set_heightmap(heightmap);
grass.set_heightmap(terrain.heightmap_texture(), hm_params);           // not owned

// Per frame, in this order:
grass.update(total_time);          // absolute-time wind clock (no dt param)
// inside an SDL_GPUCopyPass, BEFORE any render pass begins:
grass.upload_chunks(copy_pass, camera_pos, &frustum);
// inside the main render pass:
terrain.render(pass, cmd, view, proj, cam_pos, light_dir,
               shadow_bindings, shadow_binding_count, &frustum);
grass.render(pass, cmd, view, proj, cam_pos, light_dir,
             shadow_bindings, shadow_binding_count);
// inside each shadow cascade's depth-only pass:
terrain.render_shadow(shadow_pass, cmd, light_view_proj, &cascade_frustum);
```

Rules that bite:

- `grass.render()` **requires `upload_chunks()` earlier in the same frame**; copy passes cannot be recorded inside render passes, so uploads must happen first (the engine batches them into one `FrameUploads` copy pass).
- Both fragment shaders expect the **shadow uniform block pushed at fragment uniform slot 1 by the caller** before `render()` — `SceneRenderer::render_3d_scene` does `SDL_PushGPUFragmentUniformData(cmd, 1, &frame_shadow_uniforms_, ...)`. If you skip shadows entirely, push a zeroed block (with `shadowEnabled = 0`).
- Passing `frustum = nullptr` to `terrain.render`/`render_shadow` is legal and draws the whole mesh (no culling). For grass, `frustum = nullptr` skips only the frustum test, not distance culling.
- Everything must run on the render thread; neither class is thread-safe, and both are non-copyable/non-movable.
- Failure behavior is "skip and log": missing pipelines, textures, or buffers cause `render()` to return without drawing, never to crash.

### Common mistakes checklist

- Forgot `scene.set_draw_ground(true)` / `set_draw_grass(true)` — both default off.
- No grass at all: grass silently no-ops until a heightmap is set (`set_heightmap`), because blades need the height texture.
- Grass floating/underground after regenerating terrain: re-call `SceneRenderer::set_heightmap` (it re-wires the grass heightmap binding); if driving directly, re-call `grass.set_heightmap(...)` since the terrain recreates its heightmap texture.
- Called `terrain().update_splatmap` but grass still grows on the new dirt: grass splat masking is opt-in — wire it once with `set_splatmap(provider)` (the provider is re-fetched every frame, so no re-pointing is needed).
- Tried to raise grass distance past 1200: clamped by `GrassRenderer::MAX_VIEW_DISTANCE` (and by the `set_view_distance_cap` value, which `SceneRenderer` sets from `GraphicsSettings::draw_distance`).
- Feeding Draco-compressed or non-square textures: the four material PNGs must share dimensions or the loader falls back to procedural colors.

---

## Lighting, Shadows & Ambient Occlusion

This chapter covers the three systems that light a Bracken frame: **clustered
dynamic lights** (point + spot), the **directional sun with cascaded shadow
maps** (CSM), and **screen-space ambient occlusion** (horizon-based GTAO).
All three are orchestrated by `scene::SceneRenderer` — as a game developer
you mostly feed data (lights into the frame via `FrameBuilder`, settings into
`GraphicsSettings`, sun/sky into `EnvironmentState`) and the engine does the
rest. The lower-level classes (`ClusterGrid`, `ShadowMap`, `GTAO`,
`PostProcessStack`) are documented too for hosts that drive rendering
manually.

All light colors are **linear** — the scene renders into an RGBA16F HDR
target and tonemapping happens at composite (see
`src/bracken/contracts/environment.hpp`). Intensities are HDR: values well above
1.0 are normal and give bloom/tonemapping headroom.

### Dynamic lights (point & spot)

#### Light types and parameters

The light structs live in `src/bracken/contracts/light.hpp`
(namespace `bracken::render::lighting`). They are plain PODs whose layout is
mirrored byte-for-byte by the shaders — do not reorder or extend them.

```cpp
#include "bracken/contracts/light.hpp"

struct PointLight {
    glm::vec3 position;         // world space
    float radius;               // falloff cutoff in world units; no light past it
    glm::vec3 color;            // linear RGB
    float intensity;            // HDR multiplier on color
    bool casts_shadows = false; // opt into the local-shadow budget (see below)
    int32_t shadow_slot = -1;   // engine-written atlas tile; never set it yourself
    // (+ internal padding)
};

struct SpotLight {
    glm::vec3 position;
    float radius;               // same falloff cutoff as point lights
    glm::vec3 direction;        // direction the cone points (normalized)
    float inner_cos;            // cos(inner cone half-angle) — full intensity inside
    glm::vec3 color;
    float outer_cos;            // cos(outer cone half-angle) — zero outside
    float intensity;
    bool casts_shadows = false; // opt into the local-shadow budget (see below)
    int32_t shadow_slot = -1;   // engine-written; never set it yourself
    // (+ internal padding)
};
```

Falloff (see `pointFalloff` in `src/bracken/shaders/clusters.hlsli`) is
inverse-square multiplied by a smooth window `(1 - (d/R)^4)^2`, so intensity
fades to exactly zero at `radius` — pick a radius generous enough that the
window doesn't visibly clip the inverse-square tail. Spot lights use the same
distance falloff times `smoothstep(outer_cos, inner_cos, ...)` on the cone
angle. Both are shaded with the full PBR BRDF.

Setting `casts_shadows = true` opts a light into the **local-shadow budget**
(`GraphicsSettings::local_shadows` master toggle;
`local_shadow_point_budget`/`_spot_budget`, defaults 4 point + 2 spot per
frame): the engine scores candidates, renders the winners into a cached
8×8-tile D32 atlas (6 cube-face tiles per point light, 1 per spot, ≤ 6 tile
re-renders per frame), and everything over budget falls back to unshadowed.
The full behavior — tile caching, dynamic-caster invalidation, the emissive
"bulb" gotcha — is documented with the frame-building API in
[Dynamic lights](#dynamic-lights) under the Scene chapter. `shadow_slot` is
engine-written each frame; never set it.

#### Adding lights per frame

Lights are **retained-mode nothing** — you re-emit every light every frame
into the `FrameBuilder` (each packet starts empty; lights are copied by
value into the packet).

From `examples/playground/playground.cpp` (the canonical consumer) — a warm
glow on each collectible orb, and a flashlight cone while toggled on:

```cpp
void MyGame::on_simulate(bracken::SimFrame& sim) {
    auto& fb = sim.frame();
    fb.set_camera(camera().get_camera_state());

    // Point light: {position, radius, color, intensity, casts_shadows}
    fb.add_point_light({orb_pos, 320.0f, {1.0f, 0.72f, 0.30f}, 12000.0f, true});

    // Spot light: {position, radius, direction, inner_cos, color,
    //              outer_cos, intensity, casts_shadows}
    glm::vec3 dir = camera().get_forward();
    fb.add_spot_light({player_pos_ + glm::vec3(0, 150, 0), 1400.0f, dir,
                       std::cos(glm::radians(18.0f)), {0.9f, 0.95f, 1.0f},
                       std::cos(glm::radians(32.0f)), 200000.0f, true});
}
```

#### Limits and silent behaviors

- **`MAX_LIGHTS = 256`** point + spot lights combined per frame
  (`src/bracken/render/lighting/light.hpp` — the cap; the structs are `src/bracken/contracts/light.hpp`). Lights added past the cap are
  **silently ignored** — no log, no error
  (`ClusterGrid::add_point_light`/`add_spot_light` just return).
- **`MAX_LIGHTS_PER_CLUSTER = 128`** lights affecting a single screen cluster.
  Overflow is truncated (keeping the earliest-added lights) with **one
  `SDL_Log` warning per frame** (`ClusterGrid::build`,
  `src/bracken/render/lighting/light_cluster.cpp`).
- A light with `radius <= 0` is uploaded but never binned into any cluster —
  it simply does nothing.
- On frames with **zero** dynamic lights the engine skips the cluster
  build/upload entirely and publishes zeroed cluster params; the GPU buffers
  keep last frame's contents but the shader early-outs on the zero light
  count, so this is invisible (documented in `docs/architecture.md` under
  "Known stubs & silent behaviors").

#### How it works underneath (what you should know)

`SceneRenderer::render_frame` bins your lights on the CPU into a 16×9×24
view-frustum cluster grid (`CLUSTER_DIM_X/Y/Z`, 3456 clusters, exponential
depth slices) and uploads three storage buffers; opaque scene shaders
(`model`, `model_instanced`, `terrain`, `grass`) then loop only over the
lights binned into their fragment's cluster. Cost scales with *lights ×
clusters actually touched*, so hundreds of small lights are fine; a handful
of scene-spanning radii are the expensive case. The cluster near plane is
fixed at 0.1 and the far plane tracks `GraphicsSettings::get_draw_distance()`
(`src/bracken/scene/scene_renderer.cpp`).

### The sun (directional light)

The sun is *not* a clustered light — it is part of
`render::EnvironmentState` (`src/bracken/contracts/environment.hpp`), applied to
every fragment, and it is the only light with *cascaded* shadow maps (point
and spot lights get budgeted atlas shadows via `casts_shadows` — see the
local-shadow notes under [Dynamic lights](#dynamic-lights-point--spot) and
the Scene chapter). Mutate the
renderer's environment (changes take effect next frame):

```cpp
auto& env = scene_renderer().environment();
env.sun_dir   = glm::normalize(glm::vec3(-0.5f, -0.8f, -0.3f)); // direction light TRAVELS
env.sun_color = {1.0f, 0.95f, 0.9f};                            // linear tint
```

Two `GraphicsSettings` fields scale it (`src/bracken/contracts/graphics_settings.hpp`):

- `sun_intensity` (default `3.0f`) — HDR multiplier on the sun.
- `ambient_strength` (default `0.45f`) — how much hemisphere ambient survives
  in shadowed regions (0 = pitch black shadows, 1 = fully lifted).

Note the convention: `sun_dir` is the direction the light *travels* (points
from sun toward the ground), not the vector toward the sun. The full
`EnvironmentState` field table lives in
[Environment, Sky, Fog, Bloom & Post-Processing](#environment-sky-fog-bloom--post-processing).

### Cascaded shadow maps

#### Settings-level control (the normal path)

Everything a game usually needs is in `GraphicsSettings`
(`src/bracken/contracts/graphics_settings.hpp`), applied via
`Application::apply_all_graphics_settings()`:

```cpp
#include "bracken/contracts/graphics_settings.hpp"

auto& gs = graphics_settings();
gs.shadow_mode       = bracken::ShadowMode::PCSS; // Off / PCF / PCSS (default PCSS)
gs.shadow_cascades   = 3;                         // 0=1, 1=2, 2=3, 3=4 cascades (default 3)
gs.shadow_resolution = ShadowResolution::R2048;   // default
request_graphics_settings(gs);   // sim-safe; applied at the next frame fence
```

The playground cycles `shadow_mode` on F5 exactly this way. Changing
resolution recreates the atlas; changing cascade count does not.

Modes (`ShadowMode`):

- `Off` — no shadow passes run at all; the atlas is never rendered.
- `PCF` — 3×3 hardware-PCF tent filter; cheap, fixed-width soft edges.
- `PCSS` — contact-hardening soft shadows: penumbra width grows with
  blocker-receiver distance (8 blocker-search + 12 filter taps per pixel).

Shadow reach is a quality setting: the cascades cover
`min(get_draw_distance(), shadow_distance)` world units
(`GraphicsSettings::shadow_distance`, default 4000, clamped [200, 20000] —
also on the dev overlay's Graphics tab) and the shader fades shadows out over
the last 15% of that range instead of popping
(`src/bracken/scene/scene_renderer.cpp`, `shadow_common.hlsli`). Raising it
trades texel density for reach at a given `shadow_resolution`, and more
casters land in the cascades, so it costs GPU time.

#### Per-mesh opt-out

Meshes carry a `cast_shadows` flag; the shadow passes skip meshes with it
cleared (see the caster loops in `src/bracken/scene/scene_renderer.cpp`).
Alpha-`BLEND` materials are alpha-tested at 0.5 in the shadow pass rather
than casting translucent shadows.

#### Direct `ShadowMap` use (custom renderers)

`render::ShadowMap` (`src/bracken/render/shadow_map.hpp`) manages a single
D32 depth **atlas** holding up to `CSM_MAX_CASCADES = 4` cascades as 2×2
quadrants — cascade *i* occupies quadrant `((i%2)*res, (i/2)*res)`, so a
2048 per-cascade resolution means a 4096×4096 atlas. One atlas (rather than a
texture array) exists because SDL3 GPU can't render to depth-array layers,
and it lets all cascades share one render pass and one pair of sampler
bindings in receivers.

```cpp
#include "bracken/render/shadow_map.hpp"

render::ShadowMap shadow_map;
if (!shadow_map.init(device, /*resolution=*/2048)) { /* shadows disabled */ }
shadow_map.set_active_cascades(4);       // clamped to 1..4, no reinit needed
shadow_map.light_size   = 0.02f;         // tan(sun angular radius); bigger = softer PCSS
shadow_map.split_lambda = 0.85f;         // log/uniform cascade split mix

// Per frame, before rendering casters:
shadow_map.update(camera.view, camera.projection, light_dir, cam_near, shadow_far);

// Render all cascades in ONE pass, one viewport per quadrant:
SDL_GPURenderPass* pass = shadow_map.begin_shadow_pass(cmd); // clears whole atlas
for (int c = 0; c < shadow_map.active_cascades(); ++c) {
    shadow_map.set_cascade_viewport(c);  // viewport + scissor to quadrant c
    const auto& data = shadow_map.cascades()[c];
    // draw casters with data.light_view_projection ...
}
shadow_map.end_shadow_pass();

// For receivers:
gpu::ShadowDataUniforms u = shadow_map.get_shadow_uniforms(/*shadow_mode=*/2);
```

Receiver binding contract (what `SceneRenderer::bind_shadow_data` does): the
atlas is bound **twice**, back to back — first with
`shadow_compare_sampler()` (LINEAR + `LESS_OR_EQUAL` compare, giving each tap
free 2×2 hardware PCF) and then with `shadow_sampler()` (raw NEAREST, used by
the PCSS blocker search and volumetric fog) — and the
`gpu::ShadowDataUniforms` struct is pushed as a fragment uniform. Include
`src/bracken/shaders/shadow_common.hlsli` in your fragment shader after
declaring the matching cbuffer and the two atlas bindings, then call
`calcShadow(worldPos, N, L, viewDepth, screenPos)`.

#### Why the cascades are standard-Z while the scene is reversed-Z

The whole scene uses reversed-Z (near → depth 1, far → depth 0,
`GREATER_OR_EQUAL`), but the shadow cascade passes deliberately stay
**standard-Z** — atlas cleared to `1.0`, pipelines and the compare sampler use
`LESS_OR_EQUAL` (`src/bracken/render/shadow_map.cpp`,
`src/bracken/gpu/pipeline_catalog.cpp`). The cascades render through
**orthographic** projections whose depth is already distributed linearly, so
reversed-Z would fix nothing; and all the receiver bias / PCF / PCSS math in
`shadow_common.hlsli` is calibrated for near=0/far=1. Don't "fix" this.

#### Gotchas

- `shadow_mode = Off` means the atlas is never rendered and contains garbage.
  `SceneRenderer` force-disables god rays in that case for exactly this
  reason — if you consume the atlas yourself (e.g. custom fog), gate on
  shadows being on.
- `ShadowMap::update()` fits each cascade to the bounding **sphere** of its
  frustum slice with texel snapping, so shadow edges are pixel-stable under
  camera rotation *and* translation. If you write your own caster pass, don't
  re-derive matrices per draw — use `cascades()[i].light_view_projection`.
- Shadow-caster pipelines front-face cull with a depth bias of
  `(1.25, 1.75)` — one named constant, `k_shadow_bias` in
  `src/bracken/gpu/pipeline_catalog.cpp`; single-sided
  geometry that is open toward the light can leak.
- Tall casters outside a cascade slice (towers, mountains) still shadow it:
  each cascade's light-space depth range is padded by 300 world units
  (`Z_CASTER_PAD` in `src/bracken/render/shadow_map.cpp`).

### Ambient occlusion (GTAO)

#### Settings-level control

```cpp
auto& gs = graphics_settings();
gs.ao_mode     = bracken::AOMode::GTAO_Half; // Off / GTAO_Half / GTAO_Full (default GTAO_Half)
gs.ao_strength = 1.0f;                       // darkening power (0 = off, 1 = stock, >1 darker)
gs.ao_radius   = 1.0f;                       // multiplier on the world-space sample radius
request_graphics_settings(gs);   // sim-safe; applied at the next frame fence
```

The playground cycles `ao_mode` on F1. `GTAO_Half` computes AO at half
resolution (4× less AO fragment work), `GTAO_Full` at full resolution; both
produce a full-resolution AO texture. On-disk numeric values were redefined
in the AO rearchitecture (1 was "SSAO", 2 the old fake "GTAO"); old settings
files map 1 → `GTAO_Half` and 2 → `GTAO_Full`.

How it behaves:

- This is **true ground-truth AO** (Jimenez et al. 2016): per-pixel horizon
  search along rotating screen-space slices with cosine-weighted arc
  integration, computed **from the depth-prepass depth only** (normals
  reconstructed edge-aware from depth) **before the forward pass**.
- AO attenuates the **ambient term only**: the forward shaders min-combine
  the screen AO with the material's ORM occlusion and feed it to the ambient
  BRDF (sky-IBL or hemisphere). Direct sun light, speculars, and dynamic
  lights are never darkened by screen-space AO — open sunlit ground stays
  clean.
- The raw AO is **temporally accumulated** (reprojected with the prepass
  motion vectors, depth-rejected on disocclusion — same invalidation policy
  as TAA), bilaterally blurred, and joint-bilaterally upsampled, so it is
  stable under camera motion with no silhouette halos or trails.
- The base radius is 4.0 world units with a 12000-unit distance cutoff
  (defaults in `gpu::GTAOUniforms`, `src/bracken/gpu/gpu_uniforms.hpp`) —
  `ao_radius` scales the radius, `ao_strength` is applied as a power curve;
  the other tunables aren't exposed through `GraphicsSettings`.
- `AOMode::Off` skips the whole AO chain and the forward pass samples a 1×1
  white dummy (no occlusion). The frame still renders through the offscreen
  HDR path — AO-off is not a different render path.
- OPAQUE and alpha-tested (MASK) geometry are both in the depth prepass —
  foliage occludes and receives screen AO (that inter-leaf occlusion is what
  keeps a canopy from reading as a flat blob). BLEND transparents (and the
  rare skinned MASK mesh) neither occlude nor receive their own AO.

#### Direct `PostProcessStack` / `GTAO` use (custom renderers)

`render::PostProcessStack` (`src/bracken/render/post_process_stack.hpp`,
formerly named `AmbientOcclusion`) owns the **offscreen HDR scene target**,
the **composite** (tonemap) pass, and the **FXAA** pass. `render::GTAO`
(`src/bracken/render/gtao.hpp`) owns the AO chain. The per-frame sequence
`SceneRenderer` runs is:

```cpp
render::PostProcessStack post;
render::GTAO gtao;
if (!post.init(device, width, height)) { /* HDR path disabled */ }
if (!gtao.init(device, width, height, /*half_res=*/true)) { /* AO disabled */ }
post.set_clear_color(env.fog_color_horizon); // linear; uncovered pixels melt into fog

// 1. Depth prepass into post.offscreen_depth() + the velocity target
//    (SceneRenderer does this; GTAO and TAA both consume it).

// 2. GTAO from the prepass depth — BEFORE the forward pass. The forward
//    shaders sample gtao.ao_texture() (full-res R8) in their ambient term.
render::GTAO::FrameParams ao_params;
ao_params.projection     = camera.projection;   // the DRAW (jittered) projection
ao_params.inv_projection = inv_projection;
ao_params.radius_scale   = gs.ao_radius;
ao_params.strength       = gs.ao_strength;
ao_params.frame_index    = frame_index;
gtao.render(cmd, pipelines, *post.offscreen_depth(), *velocity_texture, ao_params);
// If AO didn't run this frame (mode off / no prepass): gtao.invalidate_history();

// 3. Render the whole 3D scene into the offscreen HDR target
//    (load_depth = true after a prepass).
SDL_GPURenderPass* pass = post.begin_offscreen_pass(cmd, /*load_depth=*/true);
// ... draw scene (bind gtao.ao_texture() on the shaders' AO slot) ...
post.end_offscreen_pass();

// 4. Composite HDR scene (+bloom +fog) with tonemapping. AO is NOT
//    composited here — it was applied in the forward ambient term.
render::PostProcessStack::CompositeParams p;
p.exposure     = gs.exposure;          // manual bias
p.tonemap_mode = static_cast<int>(gs.tonemap_mode);
p.depth_a = camera.projection[2][2];   // for the depth-aware fog upsample
p.depth_b = camera.projection[3][2];
// Optional: histogram auto exposure (render::AutoExposure — record its two
// compute dispatches after the TAA block, before bloom) + color grading.
// All default CompositeParams grading values are neutral == pixel-identical.
p.auto_exposure   = auto_exposure_active;
p.exposure_buffer = auto_exposure_active ? auto_exposure.state_buffer() : nullptr;
p.white_balance   = render::white_balance_coefficients(env.grade_temperature, env.grade_tint);
p.cdl_slope = env.grade_slope; p.cdl_offset = env.grade_offset; p.cdl_power = env.grade_power;
SDL_GPUTexture* target = (gs.aa_mode == AAMode::FXAA) ? post.ldr_target()
                                                      : swapchain_texture;
post.render_composite_pass(cmd, pipelines, target, bloom_tex, fog_tex, p);

// 5. Optional FXAA resolve to the swapchain (LDR path only).
if (gs.aa_mode == AAMode::FXAA) {
    post.render_fxaa_pass(cmd, swapchain_texture);
}
```

With `aa_mode == AAMode::TAA` the anti-aliasing instead happens pre-bloom in
HDR: `render::TAA` (`src/bracken/render/taa.hpp`) resolves the offscreen
color between step 3 and the bloom pass (`taa.render(cmd, pipelines,
scene_color, velocity, depth, params)`) and composite writes the swapchain
directly — no LDR intermediate, no FXAA. TAA requires the depth-prepass
velocity target (`SceneRenderer` provides it on the HDR path) and enables the
sub-pixel camera jitter; a custom renderer must invalidate the TAA history
whenever a frame is rendered without the resolve.

Contracts you must respect:

- **`render_fxaa_pass` must run after `render_composite_pass` in the same
  frame** — the composite pass caches the pipeline registry FXAA uses.
- `resize(width, height)` on window resize (it's a no-op when unchanged) on
  both objects; `GTAO::set_half_res` switches the working resolution (both
  invalidate the AO history); `ldr_target()` lazily recreates itself at the
  current size/swapchain format.
- `GTAO::render` needs the frame's prepass depth + velocity (this frame's
  contents, not last frame's) and the **draw** projection — jittered under
  TAA, because it must match the depth buffer. Call `invalidate_history()`
  for any frame where the chain didn't run, exactly like TAA.
- `depth_a`/`depth_b` must come from the *actual* projection matrix
  (`proj[2][2]`, `proj[3][2]`, glm `RH_ZO` — works for standard and
  reversed-Z). Zeros degrade to plain bilinear upsampling.
- `offscreen_depth()` clears to `0.0` (reversed-Z far) and the AO shaders
  treat depth `<= 0.0` as sky — an AO chain fed a standard-Z scene depth
  buffer will misclassify the near plane as sky.
- Pass `nullptr` for `bloom_texture`/`fog_texture` when those features are
  off; the composite pass zeroes their contribution.
- `AutoExposure::render` needs no per-frame textures beyond the scene color
  it meters and has no resize; call `invalidate()` for any frame it didn't
  run (same policy as the TAA/GTAO histories) so re-enabling snaps to the
  metered target instead of easing from a stale EV. Leave
  `CompositeParams::exposure_buffer` null and the composite binds a zeroed
  dummy with the flag off.

### Quick reference: which knob lives where

| What | Where | Default |
|---|---|---|
| Add point/spot light | `scene::FrameBuilder::add_point_light` / `add_spot_light` | per frame |
| Sun direction/color | `SceneRenderer::environment()` (`EnvironmentState`) | midday preset |
| Sun intensity / ambient lift | `GraphicsSettings::sun_intensity` / `ambient_strength` | 3.0 / 0.45 |
| Shadow filter mode | `GraphicsSettings::shadow_mode` | `PCSS` |
| Cascade count / resolution | `GraphicsSettings::shadow_cascades` / `shadow_resolution` | 4 cascades / 2048 |
| Sun softness (PCSS) | `render::ShadowMap::light_size` | 0.02 |
| Local (point/spot) shadows | `PointLight`/`SpotLight::casts_shadows` + `GraphicsSettings::local_shadows` / `local_shadow_point_budget` / `_spot_budget` / `local_shadow_resolution` | opt-in / on / 4 / 2 / 2048 atlas |
| Ambient source (sky IBL vs hemisphere) | `GraphicsSettings::ibl_enabled` | `true` (IBL) |
| AO mode / strength / radius | `GraphicsSettings::ao_mode` / `ao_strength` / `ao_radius` | `GTAO_Half` / 1.0 / 1.0 |

---

## Model PBR Surface Shading

This chapter explains what happens to every glTF or procedural model you submit through the frame builder — static (`draw_model`), instanced, skinned (`draw_skinned`), and their DoubleSided/Blend variants. All of these paths converge on the same physically-based shading algorithm: a Cook-Torrance GGX BRDF for the sun, a two-color hemisphere ambient, clustered point/spot lights, emissive, and atmospheric fog. The shader sources are `src/bracken/shaders/pbr.hlsli` (BRDF library), `src/bracken/shaders/model.frag.hlsl` / `src/bracken/shaders/model_instanced.frag.hlsl` (material evaluation), and the CPU side that feeds them is `src/bracken/scene/scene_renderer.cpp` with the uniform mirror structs in `src/bracken/gpu/gpu_uniforms.hpp`.

You normally never call any of this directly — you author a `PBRMaterial` on each `Mesh`, submit draws through a `FrameBuilder`, and set scene-wide knobs on `SceneRenderer`. This chapter tells you which knob does what, exactly how the shader interprets it, and the silent behaviors that will bite you if you don't know about them.

### What you get on every model draw

For each visible mesh fragment, in linear HDR space (the composite pass tonemaps later — all colors you provide must be linear):

1. **Base color** — either `baseColorTexture × base_color_factor × tint` (when the mesh has a texture) or `vertexColor × tint` (when it doesn't). Base-color textures are uploaded as `RGBA8_SRGB` so the sampler linearizes them; vertex colors are assumed already linear (`src/bracken/shaders/model.frag.hlsl` header comment).
2. **Alpha test** — only for glTF `MASK` materials (see below). `OPAQUE` and `BLEND` never `discard`.
3. **Normal mapping** — tangent-space, using the mesh tangents and the material's `normal_scale`.
4. **Metallic/roughness** — scalar factors, optionally modulated by a metallic-roughness texture (glTF convention: G = roughness, B = metallic, R = packed occlusion for ORM maps).
5. **Direct sun light** — full Cook-Torrance GGX evaluation (`evaluatePBR` in `src/bracken/shaders/pbr.hlsli`), shadow-modulated by the cascaded shadow maps.
6. **Ambient** — image-based lighting from the prefiltered procedural sky (`ambientIBL`: irradiance-cubemap diffuse + GGX-prefiltered specular with a split-sum BRDF LUT), dimmed inside shadows; with `ibl_enabled = false` a sky/ground two-color hemisphere gradient plus an analytic environment-specular term (`ambientPBR`).
7. **Clustered point/spot lights** — added when any dynamic lights were submitted this frame.
8. **Emissive** — `emissive_factor`, optionally × an emissive texture.
9. **Atmospheric fog** — distance + height fog blending toward the sky, unless disabled per-draw or globally.

The image-based ambient reflects the **procedural sky only** (the engine prefilters its own sky into small cubemaps — see the Environment chapter): metals genuinely mirror the sky, sun disc and clouds included, but never scene geometry (no SSR or reflection probes), and interiors still see the open sky.

### Authoring materials: PBRMaterial on each Mesh

The material lives on the mesh (`Mesh::material`, `struct PBRMaterial` in `src/bracken/model_loader.hpp`). glTF loading fills it from `pbrMetallicRoughness`; for procedural models you set the fields yourself. Fields and defaults:

```cpp
// src/bracken/model_loader.hpp
struct PBRMaterial {
    glm::vec4 base_color_factor{1.0f, 1.0f, 1.0f, 1.0f};
    float metallic_factor = 0.0f;
    float roughness_factor = 0.85f;
    float normal_scale = 1.0f;
    float occlusion_strength = 1.0f;
    glm::vec3 emissive_factor{0.0f, 0.0f, 0.0f}; // includes KHR_materials_emissive_strength
    AlphaMode alpha_mode = AlphaMode::Opaque;    // Opaque | Mask | Blend
    float alpha_cutoff = 0.5f;
    bool double_sided = false;
    bool has_packed_occlusion = false;           // AO in R of the MR texture (glTF ORM)
    // ... texture pixel buffers + GPUTexture slots (normal / MR / occlusion / emissive)
};
```

A minimal procedural material, adapted from the playground (`examples/playground/character.cpp`, `build_orb_model`):

```cpp
#include "bracken/model_loader.hpp"

auto model = std::make_unique<bracken::Model>();
bracken::Mesh mesh;
add_sphere(mesh, 8.0f, 10, 14, {1.0f, 0.85f, 0.45f, 1.0f}); // vertex colors = base color
mesh.material.roughness_factor = 0.35f;                      // shinier than the 0.85 default
// HDR emissive so the orbs feed the bloom pass (bloom threshold defaults to ~1):
mesh.material.emissive_factor = {2.4f, 1.7f, 0.55f};
model->meshes.push_back(std::move(mesh));
```

How each field behaves in the shader:

#### Base color

- If the mesh has a base-color texture (`mesh.has_texture && mesh.texture`), the fragment shader computes `texture × baseColorFactor × tint`. Otherwise it computes `vertexColor × baseColorFactor × tint` — **`base_color_factor` is honored on both paths**, and the loader bakes white vertex color when `COLOR_0` is absent, so an untextured mesh with no vertex colors shades as `baseColorFactor × tint`.
- The base-color texture is sampled through an sRGB view; author it sRGB-encoded like any glTF texture. Vertex colors must be linear.

#### Metallic / roughness

- `metallic_factor` (default 0 = dielectric) and `roughness_factor` (default 0.85) are the whole story when there is no metallic-roughness texture.
- When `metallic_roughness_texture` is present, the factors are *multiplied* by the texture's G (roughness) and B (metallic) channels — glTF semantics. If you want the texture to fully drive the response, set the factors to 1.
- Roughness is automatically widened by screen-space specular anti-aliasing (`specularAARoughness` in `src/bracken/shaders/pbr.hlsli`) to stop shimmer on curved or thin geometry. You cannot turn this off; it only ever *increases* roughness (clamped to [authored roughness, 1]).
- Fully metallic materials have no diffuse term (energy conservation): a metal's ambient is entirely the sky reflection (a polished `metallic = 1, roughness ≈ 0.05` sphere is the classic chrome-ball look — the playground places one by the spawn as a living example).

#### Normal mapping

- Requires the mesh to carry tangents (`Vertex3D::tangent` — `xyz` tangent, `w` bitangent sign; the glTF loader fills these). `normal_scale` scales the sampled normal's `xy` before renormalization — 0 flattens the map, >1 exaggerates it.
- With no normal map, the engine binds a 1×1 flat-normal fallback texture `(128,128,255)` and sets `hasNormalMap = 0`, so shading uses the interpolated geometric normal. You never need to provide a placeholder.
- Normal maps must be authored linear (UNORM); the loader uploads them as `RGBA8` (not sRGB) per the glTF spec (`src/bracken/model_loader.cpp`).

#### Occlusion

- Only **ORM-packed** occlusion works: when the glTF `occlusionTexture` shares the image with `metallicRoughnessTexture`, the loader sets `has_packed_occlusion = true` and the shader reads AO from the R channel, faded by `occlusion_strength` (`matAO = lerp(1, R, occlusion_strength)`).
- **A separate (non-packed) occlusion texture is warned about and dropped at load** (`src/bracken/model_loader.cpp` — standalone AO is unsupported). If your DCC exports a standalone AO map, repack it into the ORM texture.
- Material AO darkens **ambient only** (hemisphere diffuse + env specular); direct sun light is never occluded by the AO map. This is intentional — baked AO on directly lit surfaces looks like dirt.

#### Emissive

- `emissive_factor` is added after all lighting; if `emissive_texture` exists it multiplies the factor. Emissive textures are sRGB-encoded (uploaded `RGBA8_SRGB`).
- Values above 1.0 are fine and expected — the scene target is HDR (`RGBA16F`) and bright emissive is how you feed bloom (see the playground orbs above).

#### Alpha modes (`alpha_mode`, `alpha_cutoff`)

- `AlphaMode::Opaque` (default): alpha is ignored; the CPU forces the shader's `alphaCutoff` uniform to 0 so no fragment is ever discarded.
- `AlphaMode::Mask`: fragments with `baseColor.a < alpha_cutoff` are discarded. Only MASK materials get a non-zero cutoff — this is decided per-mesh on the CPU in `apply_mesh_material` (`src/bracken/scene/scene_renderer.cpp`), so an OPAQUE texture with a garbage alpha channel cannot accidentally alpha-test.
- `AlphaMode::Blend`: the mesh is silently **re-routed out of the opaque pass** into a sorted back-to-front transparent pass drawn after the skybox, with depth writes off. Sorting is per queued draw using the *model's* bounding-sphere center — there is no OIT and no intra-mesh sorting, so large or interpenetrating transparent meshes can sort wrong. Instanced BLEND meshes lose their instancing: each instance becomes an individual sorted draw.

#### Double-sided

- `double_sided = true` switches the draw to a no-backface-cull pipeline variant (`ModelDoubleSided`, `SkinnedModelDoubleSided`, `InstancedModelDoubleSided`, plus the Blend combinations — `src/bracken/gpu/pipeline_registry.hpp`). Lighting is unchanged; back faces are shaded with their (now backwards-facing) interpolated normal. Pipeline switches happen automatically per mesh; interleaving single- and double-sided meshes in one model costs extra pipeline binds.

### Submitting draws

```cpp
// examples/playground/playground.cpp — the canonical consumer
auto pal = fb.alloc_bone_palette(joint_count);
std::ranges::copy(anim_player_.bone_matrices().first(joint_count), pal.matrices.begin());
fb.draw_skinned(player_model_, player_transform_, pal.id);

for (const auto& prop : props_) {
    fb.draw_model(prop.model, prop.transform);
}
```

The relevant signatures (`src/bracken/scene/frame_builder.hpp`):

```cpp
void draw_model(bracken::ModelHandle handle, const glm::mat4& transform,
                const glm::vec4& tint = {1.0f, 1.0f, 1.0f, 1.0f},
                bool force_non_instanced = false, bool no_fog = false,
                uint32_t motion_id = 0);

void draw_skinned(bracken::ModelHandle handle, const glm::mat4& transform,
                  BonePaletteId palette,
                  const glm::vec4& tint = {1.0f, 1.0f, 1.0f, 1.0f},
                  bool no_fog = false, uint32_t motion_id = 0);
```

- **`tint`** multiplies the resolved base color (texture path *and* vertex-color path), including alpha. It is a linear-space multiplier. For instanced draws the tint rides per-instance in the instance storage buffer, so instances of one model can be tinted individually at no extra draw cost.
- **`no_fog`** exempts the draw from atmospheric fog (per-instance for instanced draws) — available on both `draw_model` and `draw_skinned` (`ModelDrawCommand::no_fog` / `SkinnedDrawCommand::no_fog`).
- **`palette`** comes from `fb.alloc_bone_palette(joint_count)` — packet-owned memory you fill with skinning matrices before `seal()`; no host storage is referenced after the emit (see the Scene and Animation chapters).
- The palette budget is `MAX_BONE_PALETTES = 1024` per frame (`src/bracken/scene/frame_packet.hpp`), **enforced at allocation** (error log + invalid id; invalid-id draws are dropped with a warning).

### Scene-wide lighting knobs

Two objects feed every model draw's `LightingUniforms` (built per draw in `SceneRenderer::make_model_lighting`, `src/bracken/scene/scene_renderer.cpp`):

#### EnvironmentState — sun, sky, fog colors

Live-editable through `SceneRenderer::environment()` (`src/bracken/scene/scene_renderer.hpp`); all colors linear:

```cpp
#include "bracken/contracts/environment.hpp"

auto& env = scene_renderer().environment();
env.sun_dir       = glm::normalize(glm::vec3(-0.5f, -0.8f, -0.3f)); // direction light TRAVELS
env.sun_color     = {1.0f, 0.95f, 0.9f};   // linear tint; sun_intensity multiplies at use site
env.ambient_sky   = {0.35f, 0.42f, 0.55f}; // hemisphere upper color
env.ambient_ground= {0.22f, 0.20f, 0.17f}; // hemisphere lower color (terrain bounce)
env.fog_start     = 1000.0f;               // world units
env.fog_end       = 8000.0f;
env.fog_height_falloff = 0.002f;           // fog thins with altitude
```

Note the sun convention: `sun_dir` is the direction light *travels*; the shader negates it internally to get the surface-to-light vector.

#### GraphicsSettings — intensity and shadow-fill

From `src/bracken/contracts/graphics_settings.hpp`, applied via `SceneRenderer::set_graphics_settings`:

- `sun_intensity` (default **3.0**) — HDR multiplier on `sun_color` for the direct term. Deliberately >1 so bloom and the tonemapper have headroom.
- `ambient_strength` (default **0.45**) — how much hemisphere ambient survives inside shadow. In fully lit areas ambient is at full strength; in full shadow it is multiplied by `ambient_strength` (`ambientLift = lerp(ambientStrength, 1, shadow)` in the fragment shaders). 0 = pitch-black shadows, 1 = shadows only remove the direct term. This "ambient dims with shadow" behavior is an art-direction choice to avoid a flat fill-light look — if your shadows look too dark or too flat, this is the knob.
- `fog_enabled` (default **true**) — global gate for the per-draw atmospheric fog.

#### Dynamic lights

Point and spot lights submitted through the frame builder (e.g. `fb.add_point_light(...)` in the playground) are accumulated by the same BRDF via the clustered-lighting include; they respect the mesh's metallic/roughness but not the AO map. Details are in [Lighting, Shadows & Ambient Occlusion](#lighting-shadows--ambient-occlusion) — from a material author's perspective they "just work" on every model draw.

### Using the BRDF library in your own shaders

If you add a custom surface shader, reuse `src/bracken/shaders/pbr.hlsli` rather than re-deriving the math (terrain and grass already do). The two entry points:

```hlsl
#include "pbr.hlsli"

// Direct light: all vectors world-space unit; view = surface->camera,
// light = surface->light; lightColor = linear radiance (pre-shadowed if wanted).
float3 direct = evaluatePBR(N, V, L, lightColor, baseColor, metallic, roughness, /*ao*/1.0);

// Hemisphere ambient (two-color "IBL"): AO multiplies the whole ambient.
float3 ambient = ambientPBR(N, V, baseColor, metallic, roughness, ao, skyColor, groundColor);

// Fragment-shader only (uses ddx/ddy): widen roughness against specular shimmer.
roughness = specularAARoughness(N, roughness);
```

Contracts worth knowing: `evaluatePBR` returns exactly zero when `NdotL <= 0` (no wrap lighting), applies `ao` to the diffuse term only, and derives F0 as `lerp(0.04, baseColor, metallic)`. `specularAARoughness` must only be called from a fragment shader (it uses screen-space derivatives). The shared helper header `src/bracken/shaders/lighting.hlsli` additionally provides `luminance`, exact `linearToSrgb`, `interleavedGradientNoise` (dither/rotation), the `acesNarkowicz`/`acesFitted` tonemap curves, and `viewPosFromDepth`/`worldPosFromDepth` reconstruction — include it instead of copy-pasting constants.

### Gotchas checklist

- **Untextured meshes ignore `base_color_factor`** — color them via vertex colors (× tint).
- **Standalone occlusion textures are silently not uploaded**; only ORM-packed (R-channel) occlusion reaches the shader, and it darkens ambient only.
- **`alpha_cutoff` does nothing unless `alpha_mode == AlphaMode::Mask`** — the CPU zeroes the uniform for Opaque/Blend.
- **BLEND meshes leave the opaque path**: sorted per-draw by model bounding-sphere center, depth-write off, no OIT; instanced BLEND falls back to one draw per instance.
- **Bone palettes live in packet memory** (`alloc_bone_palette`); the per-frame budget is 1024 palettes, enforced at build time (over-budget allocations return an invalid id and the draw is dropped with a warning).
- **All colors are linear** — tints, vertex colors, `emissive_factor`, `EnvironmentState`. Only image files are sRGB (and the sampler handles that).
- **Emissive is your bloom input** — values >1 are the intended way to make things glow.
- **Metals have no diffuse** — a `metallic_factor = 1` material lives entirely off its specular response: the sun highlight plus the prefiltered-sky reflection (scene geometry is never reflected).
- **Shadows also dim ambient** (`ambient_strength`); if shadowed areas read wrong, tune that setting, not your material.
- **Specular AA is always on** and can only increase roughness — mirror-sharp highlights on high-curvature geometry will be slightly blurred by design.

---

## Water

`render::WaterRenderer` draws flat water surfaces (rivers, ponds, lakes) with
screen-space refraction, depth-based color absorption, prefiltered-sky
reflections, a sun glint, shoreline foam — and an interactive compute-shader
wave simulation you can push ripples into.

### Setting up a water region

```cpp
#include "bracken/render/water_renderer.hpp"

if (auto* water = scene_renderer().water()) {
    render::WaterRegion river;
    river.min_x = 150.0f;  river.max_x = 1400.0f;   // world rect the plane covers
    river.min_z = 0.0f;    river.max_z = 8000.0f;
    river.height = -25.0f;                          // water level (world Y)
    river.flow = {0.0f, -1.0f};                     // normalized flow direction
    river.flow_speed = 55.0f;                       // world units/s
    river.sim_center = {900.0f, 4000.0f};           // interactive-sim window centre
    river.sim_extent = 2048.0f;                     // window edge length
    water->set_region(river);
}
```

Nothing renders until a region is set. The rect can generously cover a carved
riverbed: terrain above the water level hides the plane through the depth
test, so only the channel shows water. **The terrain must stay above
`height` everywhere else inside the rect** or the plane pools in the dips —
clamp your heightmap in a margin band around the channel (the playground's
`build_heightmap` shows the pattern).

Look tuning lives on the region struct (linear-space colors):
`deep_color`/`shallow_color`/`absorption` (Beer-Lambert), `foam_depth`
(shoreline band), `reflection_roughness` (IBL prefilter mip),
`refraction_strength`, `detail_wave_strength` (analytic micro-wave slope —
keep small, it is literally the max surface slope), `sim_displacement`,
`sim_normal_strength`, `ripple_foam_gain`, `wave_speed`, `wave_damping`.

### Ripples

```cpp
// world position; radius/strength in world units. Cheap — call freely.
scene_renderer().water()->add_ripple(player_pos, 26.0f, 0.6f);
```

Impulses splat into the wave-equation field, propagate, reflect off nothing
(the window border absorbs), and drift downstream with `flow`. Impulses
outside the sim window are silently ignored. Use it for wading, splashes,
rain, or ambient life (the playground pushes both wade ripples at the
player's feet and occasional random rings).

### What it costs / when it runs

The sim is a 512² RG16F ping-pong compute pass at fixed 60 Hz substeps (≤4
per frame). The surface pass snapshots the HDR color + depth once per frame
(two texture copies) and draws one indexed grid. Both only run while a
region is configured. There is no graphics-settings toggle — an unset region
is the off switch.

### Gotchas

- The water pass runs after transparents/particles: where a particle or
  BLEND mesh overlaps water ON SCREEN and is closer to the camera, the water
  still overdraws it. Keep effects away from water, or accept the artifact.
- Reflections are screen-space (SSR) over a sky-IBL fallback: geometry that
  is on screen — banks, trees, buildings, the player — mirrors in the water;
  rays that leave the screen or start from occluded content fall back to the
  sky. Tune with `ssr_strength` (0 disables), `ssr_max_distance`,
  `ssr_thickness`.
- The interactive window (`sim_extent` around `sim_center`) should cover the
  stretch the player can actually reach; outside it the surface falls back to
  the analytic detail waves (the sim contribution fades at the window edge).

---

## Environment, Sky, Fog, Bloom & Post-Processing

This chapter covers Bracken's atmosphere and post-processing stack: the
`EnvironmentState` struct that drives sun, sky, ambient and fog everywhere;
the procedural skybox in `render::WorldRenderer`; analytic distance fog;
volumetric fog with god rays; the bloom chain; the linear-HDR render target
and the composite/tonemap pass that produces the final swapchain image; and
`render::RenderContext`, the window/GPU-device foundation everything sits on.

Relevant source:

- `src/bracken/contracts/environment.hpp` — `EnvironmentState`
- `src/bracken/render/world_renderer.hpp/.cpp` — skybox + debug grid
- `src/bracken/render/volumetric_fog.hpp/.cpp` — volumetric fog / god rays
- `src/bracken/render/bloom.hpp/.cpp` — bloom mip chain
- `src/bracken/render/render_context.hpp/.cpp` — window, device, swapchain
- `src/bracken/shaders/skybox.frag.hlsl`, `volumetric_fog.frag.hlsl`,
  `bloom_downsample.frag.hlsl`, `bloom_upsample.frag.hlsl`,
  `composite.frag.hlsl`, `fog.hlsli`

### The HDR pipeline at a glance

Everything in the 3D scene renders in **linear light** into an offscreen
**RGBA16F** color target (with a D32 depth buffer). Post passes (AO, bloom,
volumetric fog) read that target. Only at the very end does the **composite
pass** apply exposure (manual bias × the auto-adapted EV), white balance,
tonemap (ACES by default), CDL grade, contrast/saturation,
gamma-encode, and write the **RGBA8 UNORM** swapchain:

```text
IBL upkeep (only when the sky changed: ~9 cube-face prefilter passes/frame)
Depth prepass ──▶ GTAO chain (raw AO ▸ temporal ▸ blur ▸ upsample; full-res R8)
                        │ (AO + the IBL cubemaps feed the forward ambient term)
3D scene ──▶ RGBA16F offscreen (clear = env.fog_color_horizon)
                │
                ├─▶ Auto exposure (2 compute dispatches: histogram + adapt)
                ├─▶ Bloom down/up chain        (half-res RGBA16F mips)
                └─▶ Volumetric fog ray march   (half-res RGBA16F)
                            │
             Composite: scene + fog + bloom, exposure (bias × exp2(EV)),
             white balance, tonemap, CDL grade, contrast/saturation,
             sRGB encode, dither
                            │
              swapchain (RGBA8)  [FXAA mode: LDR target ──▶ FXAA ──▶ swapchain]
   (TAA mode: the HDR scene is TAA-resolved in place before the exposure tap)
```

**Auto exposure** (`auto_exposure`, default on) meters the pre-bloom HDR
image with a 256-bin log2-luminance histogram (compute shaders — the darkest
60% and brightest 5% of pixels are ignored) and eases the exposure toward
the metered target at `auto_exposure_speed_up`/`_down` EV/s, clamped to
`auto_exposure_min/max_ev`. The old `exposure` setting stays as a manual
bias on top. It converges so the default scene looks the same as with auto
exposure off; walk into shadow or dim the sky and the image re-brightens
over ~a second instead of crushing to black. The first frames after
enabling snap immediately (no fade-in from a wrong exposure), and the
adapted EV is visible in the dev overlay's Graphics tab.

Consequences you must internalize:

- **Every color you feed the engine is linear**, and may exceed 1.0. That
  includes all `EnvironmentState` colors, model tints, light colors, and
  emissive factors. Do not pass sRGB-picked values without converting.
- Values above 1.0 are meaningful: they survive into the bloom threshold and
  tonemapper. The default `GraphicsSettings::sun_intensity` is `3.0f`
  precisely so bloom/tonemap have headroom.
- The composite pass is where all the "look" knobs live
  (exposure/tonemap/contrast/saturation from `GraphicsSettings`; white
  balance + CDL grading from `EnvironmentState`) — see below.

You get this pipeline automatically by using `bracken::Application` +
`scene::SceneRenderer`; there is nothing to opt into. (If the post-process
stack failed to initialize, the renderer falls back to a single
direct-to-swapchain pass with no prepass or post-processing — see
`docs/technical_spec.md`.)

### EnvironmentState — the atmosphere's single source of truth

`bracken::render::EnvironmentState` (`src/bracken/contracts/environment.hpp`) is
one plain struct that describes the sun, sky gradient, ambient,
distance fog, and volumetric fog. The scene layer pushes it into every
sub-renderer (world/skybox, terrain, grass, volumetric fog, model lighting)
once per frame, so **editing this one struct changes the whole frame
consistently**.

#### Getting and editing it

The host path is to keep your own `EnvironmentState` and **stage it into the
frame** with `fb.set_environment(env)` from `on_simulate` — the sim stage
also reads the staged environment's `cloud_speed` to advance the sky-cloud
scroll clock, so a host that animates cloud speed must stage the
environment for the change to affect the scroll rate.
(`scene::SceneRenderer::environment()` still returns a live mutable
reference — that is the DevUI Environment tab's main-thread live-edit path,
used when a frame stages nothing; mutations take effect next frame.)

```cpp
#include "bracken/contracts/environment.hpp"

// Inside on_simulate (env_ is a host member):
bracken::render::EnvironmentState& env = env_;

// A warm sunset:
env.sun_dir = -glm::normalize(glm::vec3(0.9f, 0.15f, 0.2f)); // light TRAVELS this way
env.sun_color = glm::vec3(1.0f, 0.55f, 0.30f);
env.sky_zenith = glm::vec3(0.10f, 0.15f, 0.35f);
env.sky_horizon = glm::vec3(0.85f, 0.45f, 0.30f);
env.fog_color_horizon = env.sky_horizon;  // fog must melt into the sky
env.fog_color_sun = glm::vec3(1.0f, 0.60f, 0.35f);
env.fog_volumetric_sun_color = env.sun_color; // see gotcha below

sim.frame().set_environment(env);         // stage it into this frame's packet
```

This is exactly what the dev overlay's **Environment tab** does
(`src/bracken/dev/dev_panels.cpp`, `draw_environment_tab`) — press **F10** in
the playground and edit every field live with HDR color pickers. That panel is
the fastest way to find values before hard-coding them.

#### Fields and defaults

All colors linear. Defaults model a clear summer midday.

| Field | Default | Meaning |
|---|---|---|
| `sun_dir` | `normalize(-0.5, -0.8, -0.3)` | direction the light **TRAVELS** (points down/away from the sun) |
| `sun_color` | `(1.0, 0.95, 0.9)` | linear sun tint; `GraphicsSettings::sun_intensity` multiplies at use site |
| `sky_zenith` | `(0.20, 0.37, 0.66)` | sky color overhead |
| `sky_horizon` | `(0.45, 0.56, 0.74)` | pale band at the horizon |
| `ground_color` | `(0.16, 0.14, 0.11)` | sky color below the horizon line |
| `ambient_sky` | `(0.35, 0.42, 0.55)` | hemisphere-ambient upper color (pre `ambient_strength` scale) — **fallback path only**: with `ibl_enabled` (the default) ambient comes from the prefiltered sky instead |
| `ambient_ground` | `(0.22, 0.20, 0.17)` | hemisphere-ambient lower color (terrain bounce) — fallback path only, as above |
| `fog_color_horizon` | `(0.45, 0.56, 0.74)` | distance-fog color; **should track `sky_horizon`** |
| `fog_color_sun` | `(0.95, 0.75, 0.50)` | fog tint when looking toward the sun |
| `fog_start` | `1000.0f` | world units; fog factor 0 at/below this view distance |
| `fog_end` | `8000.0f` | world units; fog factor saturates approaching this |
| `fog_height_falloff` | `0.002f` | exponential falloff on world Y (thinner fog at altitude) |
| `cloud_coverage` | `0.35f` | sky fraction reading as cloud; 0 = clear, ~1 = overcast |
| `cloud_density` | `1.0f` | cloud opacity/edge weight; < 1 wispy sheets, > 1 heavy solid decks |
| `cloud_scale` | `1.0f` | cloud-noise footprint multiplier; < 1 = larger formations |
| `cloud_speed` | `1.0f` | cloud scroll-rate multiplier (phase accumulates — live edits never make the deck jump) |
| `haze_strength` | `1.0f` | horizon haze amount; the haze color always comes from the fog colors |
| `fog_volumetric_sun_color` | `(1.0, 0.9, 0.7)` | god-ray inscatter color |
| `fog_volumetric_scatter` | `0.25f` | inscatter strength along the forward-scatter lobe |
| `fog_volumetric_density` | `0.0015f` | base extinction per world unit at y = 0 |
| `fog_volumetric_height` | `100.0f` | height scale of the fog volume (world Y) |
| `fog_volumetric_falloff` | `0.005f` | exponential shaping with height |
| `fog_volumetric_start` | `100.0f` | near dead zone — ray march starts here |
| `fog_volumetric_end` | `3000.0f` | ray-march distance cap (world units) |
| `grade_temperature` / `grade_tint` | `0.0f` / `0.0f` | white balance in \[−1, 1\]: temperature warm(+)/cool(−), tint magenta(+)/green(−); applied pre-tonemap in linear light |
| `grade_slope` / `grade_offset` / `grade_power` | `(1,1,1)` / `(0,0,0)` / `(1,1,1)` | ASC-CDL grade `pow(c·slope + offset, power)` on the tonemapped image, per channel |

Grading lives here (not in `GraphicsSettings`) because it is a **look**,
live-editable like the sky colors — the dev overlay's Environment tab has
sliders for all of it. All-neutral values (the defaults) are guaranteed
pixel-identical to no grading at all: the composite shader skips the gated
blocks entirely.

#### Gotchas

- **`sun_dir` is the direction light travels**, not the direction to the sun.
  `WorldRenderer::sun_direction()` returns `-env_.sun_dir` (direction TO the
  sun) if you need the other convention. Getting this backwards puts the sun
  disc opposite the lighting.
- **`fog_color_horizon` should track `sky_horizon`.** Distance fog and the
  skybox's aerial haze both converge on the fog colors; if they diverge,
  terrain fog and the sky no longer meet invisibly at the horizon. The
  offscreen scene target is also **cleared to `fog_color_horizon`** every
  frame, so any uncovered pixels melt into the atmosphere.
- **Distance fog day-dims itself.** `hazeSunColor` (shared by sky haze and
  `applyAtmosphericFog`) scales the fog color by the same
  `saturate(sin(elevation)·2 + 0.2)` ramp the sky gradient uses, so ground
  fog darkens through dusk together with the sky instead of glowing daylight
  white under a night sky. Your fog *colors* stay time-of-day-agnostic; a
  day/night host only retints them for artistic shifts (e.g. redder dusk).
- **Cloud shaping changes relight the world.** `cloud_coverage` /
  `cloud_density` / `cloud_scale` / `haze_strength` are part of the IBL sky
  snapshot: editing them starts the same amortized prefilter refresh a
  `sun_dir` change does. `cloud_speed` (and the scroll itself) never does —
  reflections keep the frozen cloud state of the last refresh.
- **`fog_volumetric_sun_color` does NOT get multiplied by
  `sun_intensity`** (unlike `sun_color` for surface lighting). A day/night
  cycle host must track its sun tint into this field itself, or god rays stay
  daytime-colored at night.
- The struct is copied by value into each sub-renderer per frame. There is no
  eventing; just mutate and the next `render_frame` picks it up.

### Distance fog

The cheap per-pixel analytic fog every opaque surface shader applies. Shaped
by `fog_start`, `fog_end`, `fog_height_falloff`, `fog_color_horizon`, and
`fog_color_sun` from `EnvironmentState`. Implemented in
`src/bracken/shaders/fog.hlsli` (`applyAtmosphericFog`): a distance factor
with exponential-shaped falloff, attenuated by an analytic average of fog
density along the view ray so fog thins with altitude. The fog color comes
from `hazeSunColor` — a Mie-like forward-scatter lobe toward the sun that is
tight and subtle at midday, widens/brightens toward `fog_color_sun` as the
sun drops (golden hour), keeps a fading afterglow a few degrees past sunset,
and day-dims with the sky. The skybox's aerial haze calls the **same
function**, which is what makes ground fog and sky meet invisibly at the
horizon at every sun elevation.

Controls:

- **Global toggle**: `GraphicsSettings::fog_enabled` (default `true`).
- **Per-draw opt-out**: `ModelDrawCommand::no_fog` /
  `FrameBuilder::draw_model(...)`'s no-fog path — useful for UI-ish 3D elements.
- Everything else via `EnvironmentState` fields above.

Distance fog is not a pass; it costs nothing extra to leave on.

### Skybox and WorldRenderer

`render::WorldRenderer` (`src/bracken/render/world_renderer.hpp`) renders the
procedural sky. Under `SceneRenderer` it is created,
initialized, and driven for you — your interaction is limited to:

```cpp
// each frame, when building the frame (playground.cpp on_simulate):
fb.set_draw_skybox(gs.skybox_enabled);  // packet flag, defaults true each frame
```

The skybox draws only when **both** the packet's skybox flag is `true`
(the default; every packet resets it) **and** `GraphicsSettings::skybox_enabled`
is `true` (default `true`).

What you get (all linear HDR, from `skybox.frag.hlsl` /
`sky_common.hlsli`):

- a 3-stop gradient (ground / horizon / zenith) from the `EnvironmentState`
  colors;
- an analytic sun disc at ~5.0 HDR intensity — **bright enough to feed
  bloom** — plus corona and glare;
- a scrolling, domain-warped FBM cloud deck shaped by the
  **`EnvironmentState` cloud knobs**: `cloud_coverage` (0 clear … ~1
  overcast, default 0.35), `cloud_density` (wispy < 1 < heavy, default 1),
  `cloud_scale` (noise footprint, default 1) and `cloud_speed` (scroll rate,
  default 1). Clouds thin toward the horizon (flat perspective), carry a
  sun-proximity silver lining that strengthens and warms as the sun drops,
  and self-shadow their bases via the density gradient toward the sun;
- aerial haze converging on the (sun-tinted, day-dimmed) fog color —
  `hazeSunColor` in `fog.hlsli` is shared with `applyAtmosphericFog`, so
  terrain fog disappears into the sky at any sun elevation. `haze_strength`
  (default 1) scales the haze band;
- a dormant hash starfield that appears when `sun_dir.y` goes above 0 (sun
  below the horizon), occluded by cloud silhouettes, with the whole sky
  ramping down through dusk;
- a full moon that rides exactly opposite the sun (no extra parameters — it
  rises as the sun sets), with a soft glow, occluded by clouds like the
  stars;
- a faint blue night skyglow once the sun is down. This is what keeps night
  scenes navigable: the IBL capture shares the sky function and replaces the
  hemisphere ambient, so the glow becomes the scene's moonlit ambient.

The whole day/night response is driven purely by `sun_dir` — lower the sun
and the sky warms at the horizon, dims, and stars, moon and skyglow appear.
At golden hour the
warm forward-scatter lobe around the sun widens and brightens automatically;
you still own the *colors* (a day/night host should dim/redden `sun_color`
and the fog colors as it likes). There is no separate "time of day"
parameter; the only clock is `sky_time()` (the cloud scroll phase —
`cloud_speed` scales its advance, so speed edits never make the deck
jump). The phase is advanced by the **sim stage** from the environment the
host *stages* via `fb.set_environment` (last staged wins; defaults if a
host never stages one) — so a DevUI live edit of `cloud_speed` changes the
scroll rate only for hosts that stage their environment each frame. All
five cloud/haze knobs are live-editable in the DevUI Environment tab;
coverage/density/scale/haze changes kick off an amortized IBL refresh,
`cloud_speed` and the scroll itself never do.

#### Direct (advanced) use

If you bypass `SceneRenderer`, the API is:

```cpp
bracken::render::WorldRenderer world;
if (!world.init(device, pipeline_registry)) { /* fatal — includes skybox
                                                 mesh-buffer creation */ }

world.set_environment(env);   // once per frame
world.set_sky_time(t);        // adopt your accumulated cloud-scroll phase
                              // (under the engine this comes from the packet's
                              // sim-advanced sky clock; accumulate t += dt *
                              // env.cloud_speed yourself in standalone use)

// inside an active render pass, after opaque geometry:
world.render_skybox(pass, cmd, view, projection);
```

Notes for direct users:

- `init(device, registry)` returns `false` if the skybox mesh buffers fail to
  create — check it.
- `render_skybox` must run **after opaque geometry**: it draws a fullscreen
  triangle at depth 0.0 (reversed-Z far plane) with a `GREATER_OR_EQUAL`
  test, so it only fills pixels nothing else touched.
- Rendering is a silent no-op when the pass/cmd/pipeline/buffers are
  missing — nothing crashes, you just get no sky.
- `sky_time()` is the accumulated cloud scroll **phase** (`dt` integrated
  scaled by `EnvironmentState::cloud_speed`) and drives **only** the skybox's
  cloud scroll (the `time` uniform in `skybox.frag`); grass wind runs on
  `SceneRenderer`'s separately accumulated scene clock.
- `sun_direction()` returns the direction **to** the sun; `light_dir()`
  returns the travel direction (same as `env().sun_dir`).

### Image-based lighting from the sky (`render::IBL`)

The ambient light on every surface — terrain, grass, and all model draws —
comes from the sky itself: `render::IBL` (`src/bracken/render/ibl.hpp`)
prefilters the procedural sky into a small diffuse-irradiance cubemap, a
GGX-prefiltered specular cubemap (7 mips, roughness-indexed), and a split-sum
BRDF LUT, and the scene shaders sample them in their ambient term. In
practice this means:

- **sky-facing surfaces pick up the cool sky color, ground-facing surfaces
  the warm ground bounce** — ambient has directional character instead of a
  flat two-color blend;
- **metals mirror the actual sky** — sun disc, clouds and horizon included
  (drop a `metallic = 1, roughness ≈ 0.05` sphere in your scene, or look at
  the chrome ball near the playground spawn);
- **the maps follow your `EnvironmentState` live**: change `sun_dir` or any
  sky color (day/night cycle, DevUI edits) and the chain re-renders
  automatically, spread over ~7 frames (9 tiny cube-face passes per frame) —
  no hitch, sub-second convergence. Cloud scrolling alone deliberately does
  NOT re-render the chain; the reflected clouds are frozen at the last
  refresh.

You normally interact with exactly one knob:

```cpp
graphics_settings().ibl_enabled = true;   // default; false = legacy hemisphere ambient
request_graphics_settings(graphics_settings());   // sim-safe deferred apply
```

With `ibl_enabled = false` (or if IBL failed to initialize) the shaders fall
back to the old two-color hemisphere ambient built from
`EnvironmentState::ambient_sky` / `ambient_ground` — those two fields are
ONLY used by the fallback path. The `ambient_strength` shadow-lift knob and
both AO sources apply identically on either path. There is nothing to
resize, rebuild, or bind yourself; `SceneRenderer` owns the whole chain
(read-only stats via `scene_renderer().ibl()`, e.g. `refresh_count()`).

Scope to keep in mind: the cubemaps contain the **sky only** — no scene
geometry (no reflection probes / SSR), so a mirror indoors still reflects
open sky.

### Volumetric fog and god rays

`render::VolumetricFog` (`src/bracken/render/volumetric_fog.hpp`) ray-marches
a height-based fog volume at **half resolution**, sampling the CSM shadow
atlas to produce visible light shafts. Its output (RGBA16F: RGB = inscattered
light, A = fog opacity) is blended in the composite pass with premultiplied
alpha.

#### Turning it on

It is **off by default** (it's the most expensive post pass):

```cpp
auto& gs = graphics_settings();
gs.volumetric_fog = true;   // enable the fog volume
gs.god_rays = true;         // light shafts; needs shadow_mode != ShadowMode::Off
request_graphics_settings(gs);   // sim-safe; applied at the next frame fence
```

(the playground binds this to **F4**, `examples/playground/playground.cpp`).

Key behaviors:

- **God rays require shadows.** When `shadow_mode == ShadowMode::Off` the
  shadow atlas is never rendered, so `SceneRenderer` force-disables god rays
  that frame regardless of `gs.god_rays`.
- `GraphicsSettings::fog_density` (default `0.05f`) is a **multiplier**, not
  an absolute density: the renderer computes
  `density_multiplier = gfx.fog_density / 0.05f` and applies it on top of
  `env.fog_volumetric_density`. Leave it at 0.05 for a 1× multiplier and
  shape the fog with the `fog_volumetric_*` fields instead.
- All shape/color parameters come from `EnvironmentState`
  (`fog_volumetric_*` fields, plus `sun_dir` and `fog_color_horizon` for the
  ambient fog tint).
- Sky pixels are marched too, so shafts silhouette against the sky instead of
  cutting off at the horizon.
- `fog_volumetric_start` (default 100) is a near **dead zone** — no
  volumetric fog closer than this to the camera. If fog seems to "start late",
  this is why.

#### Cost

The fixed step count in `volumetric_fog.frag.hlsl` (`numSteps = 24`) is the
cost knob: every half-res pixel — sky included — pays 24 density samples,
plus one shadow tap per step when god rays are on. There is no C++ setting
for it; change the shader if you need a different quality/cost point.

#### Direct (advanced) use

```cpp
bracken::render::VolumetricFog fog;
if (!fog.init(device, width, height)) { /* fatal */ }   // allocates half-res RGBA16F

// per frame, after the offscreen scene + shadow passes, before composite,
// with NO render pass active:
fog.render(cmd, registry, depth_texture, shadow_map, camera, env,
           /*god_rays_enabled=*/true, /*fog_enabled=*/true, /*density_multiplier=*/1.0f);
SDL_GPUTexture* result = fog.fog_texture();  // bind in your composite
```

`build_uniforms(...)` is public and pure-CPU so the environment plumbing is
unit-testable — see `src/bracken/render/volumetric_fog.test.cpp` for
working examples (it verifies every `fog_volumetric_*` field lands in
`gpu::VolumetricFogUniforms`, and that disabling fog only zeroes density).
Call `resize(w, h)` on window resizes (no-op when unchanged).

### Bloom

`render::Bloom` (`src/bracken/render/bloom.hpp`) extracts pixels above a
luminance threshold, downsamples through a mip chain (13-tap Jimenez 2014
filter with Karis averaging on the first pass to kill fireflies), then
progressively tent-upsamples back, each mip **accumulating** additively — a
tight glow core plus wide falloff. The composite pass adds the result scaled
by `bloom_strength`.

#### Configuration

All through `GraphicsSettings`:

| Setting | Default | Meaning |
|---|---|---|
| `bloom_enabled` | `true` | run the chain at all |
| `bloom_strength` | `0.25f` | multiplier on the bloom texture at composite (0.0–1.0 intended) |
| `bloom_threshold` | `1.1f` | **post-exposure** HDR luminance above which pixels bloom |

```cpp
auto& gs = graphics_settings();
gs.bloom_enabled = !gs.bloom_enabled;   // playground F2 toggle
request_graphics_settings(gs);   // sim-safe; applied at the next frame fence
```

The threshold is compared against `scene_luminance * exposure`, so **what
blooms stays stable when you change exposure** (identical to naive behavior
at `exposure == 1.0`). The extracted color itself stays in pre-exposure scene
units; composite applies exposure exactly once.

#### Making something glow

Give it HDR output above the threshold. The playground's collectible orbs do
it with an emissive material (`examples/playground/character.cpp`):

```cpp
Mesh mesh;
add_sphere(mesh, 8.0f, 10, 14, {1.0f, 0.85f, 0.45f, 1.0f});
// HDR emissive so the orbs feed the bloom pass (bloom_threshold defaults to 1.1).
mesh.material.emissive_factor = {2.4f, 1.7f, 0.55f};
```

The skybox's sun disc (~5.0 HDR) and anything lit past ~1.1 luminance by the
default `sun_intensity = 3.0` sun also bloom naturally.

#### Gotchas

- The mip count scales with resolution — `clamp(log2(min(w,h)) − 7, 4, 6)`:
  4 mips at 1080p–4K, more only at 8K+. Wider glow at high res without extra
  cost at low res. Not user-configurable.
- Bloom reads the **offscreen HDR color**, so it sees the scene before
  fog is applied (screen AO is already in the forward-lit ambient).
- Setting `bloom_enabled = false` skips the chain entirely and composite
  receives `bloomStrength = 0`.

### Composite, tonemapping, and color grading

The composite pass (`composite.frag.hlsl`; invoked as
`PostProcessStack::render_composite_pass` in
`src/bracken/render/post_process_stack.cpp`) combines everything and
produces the final image. Screen-space AO is **not** part of composite —
it is applied to the ambient term in the forward shaders. In order:

1. Depth-aware (joint bilateral) upsample of the half-res fog texture
   (prevents bleed at silhouettes).
2. Volumetric fog: `color * (1 − fog.a) + fog.rgb` (premultiplied, linear).
3. Bloom: `color += bloom * bloomStrength`.
4. Exposure: `color *= exposure` — the manual bias, multiplied by
   `exp2(adapted EV)` when `auto_exposure` is on (the adapted EV is read
   from the auto-exposure state SSBO).
5. White balance (flag-gated; linear LMS gains from
   `EnvironmentState::grade_temperature`/`grade_tint`).
6. Tonemap.
7. ASC-CDL grade (flag-gated; `pow(c·slope + offset, power)` from
   `EnvironmentState::grade_slope/offset/power`).
8. Post-tonemap contrast (around mid-grey 0.5), then saturation.
9. sRGB gamma encode + a half-LSB dither so gradients don't band on the
   RGBA8 swapchain.

Both grading blocks are gated on non-neutral values, so the all-neutral
defaults are pixel-identical to the ungraded path.

Your knobs — display trims on `GraphicsSettings`, look grading on
`EnvironmentState`:

| Setting | Default | Notes |
|---|---|---|
| `exposure` | `1.0f` | manual bias; multiplies the adapted exp2(EV) when `auto_exposure` is on |
| `auto_exposure` | `true` | histogram-metered adaptive EV (see the HDR pipeline overview) |
| `tonemap_mode` | `TonemapMode::ACESFitted` | also `ACESNarkowicz`, `Reinhard`, `None` (clamp) |
| `contrast` | `1.0f` | post-tonemap; 1.2 = punchy |
| `saturation` | `1.0f` | post-tonemap; 0.0 = grayscale |
| `bloom_strength` | `0.25f` | see Bloom |
| `ao_strength` | `1.0f` | AO darkening intensity |
| `aa_mode` | `AAMode::TAA` | TAA resolves pre-bloom in HDR; FXAA runs *after* composite on the LDR image |
| `EnvironmentState::grade_*` | neutral | white balance + ASC-CDL (see the EnvironmentState field table) |

```cpp
auto& gs = graphics_settings();
gs.exposure = 1.2f;
gs.tonemap_mode = bracken::TonemapMode::ACESFitted;
gs.contrast = 1.05f;
gs.saturation = 1.1f;
request_graphics_settings(gs);   // sim-safe; applied at the next frame fence
```

With `aa_mode == AAMode::FXAA`, composite writes an intermediate LDR target
and an FXAA pass resolves onto the swapchain; with `AAMode::Off` **or
`AAMode::TAA`** (the default — TAA already resolved the HDR scene before
bloom) composite writes the swapchain directly. Toggle at runtime freely
(playground F6 cycles Off/FXAA/TAA).

Common mistake: trying to do color grading in linear space by scaling
`EnvironmentState` colors when what you want is `contrast`/`saturation` —
those operate post-tonemap where they behave like a grading LUT would.

### RenderContext

`render::RenderContext` (`src/bracken/render/render_context.hpp`) owns the
SDL window and `gpu::GPUDevice`, and manages the frame lifecycle
(command-buffer acquire/submit), the swapchain, vsync, and window modes.

**If you subclass `bracken::Application` you never construct one** — the
application owns it and exposes what you need
(`Application::init_renderer(width, height, title)`, `set_vsync_mode`,
`set_window_mode`, `screen_width()/screen_height()`). Access it via
`scene_renderer().context()` if you must.

#### Standalone use

```cpp
#include "bracken/render/render_context.hpp"

bracken::render::RenderContext ctx;
if (!ctx.init(1280, 720, "my app")) { return 1; }   // window + GPU device

// per frame:
SDL_GPUCommandBuffer* cmd = ctx.begin_frame_cmd();  // may be nullptr on failure
SDL_GPUTexture* backbuffer = ctx.acquire_swapchain_texture(cmd);
// ... record render passes ...
ctx.end_frame(cmd);                                  // submit + present
```

There is also a legacy void `begin_frame()`/`end_frame()` pair that stores
the command buffer internally (`current_command_buffer()` between them);
`SceneRenderer` uses that form. Calling `begin_frame()` twice without ending
logs a warning and submits the in-flight frame rather than leaking it.

#### Sizing, vsync, window modes

- `width()`/`height()` are the drawable size **in pixels**
  (`SDL_GetWindowSizeInPixels`) — HiDPI-correct and what the swapchain and
  every swapchain-sized offscreen target (AO, bloom, fog, LDR) use. Call
  `update_viewport()` after resize events (the internal `begin_frame_cmd()`
  already does).
- `set_vsync_mode(mode)`: `0` = off (immediate), `1` = vsync, `2` = triple
  buffer (mailbox). The request is **clamped** to `max_vsync_mode()` — mailbox
  isn't available on all platforms, and asking for more than supported is
  silently downgraded.
- `set_window_mode(mode, resolution_index)`: `0` = windowed, `1` = borderless
  fullscreen, `2` = exclusive fullscreen using an entry from
  `available_resolutions()` (populate with `query_display_modes()`, deduped
  by w×h). Repeat calls with unchanged values are no-ops (transitions are
  expensive).
- `end_frame_and_wait()` submits with a fence and **blocks until the GPU
  finishes** — readback paths only (the screenshot feature uses it). Expect a
  hitch.

#### Gotchas

- `RenderContext` is non-copyable/non-movable and everything runs on the SDL
  main thread (swapchain acquisition asserts this).
- There is no `RenderContext` clear-color API — the offscreen scene clear
  tracks `env.fog_color_horizon` (via `PostProcessStack::set_clear_color`),
  and the direct-to-swapchain fallback uses a hardcoded color.

### Quick checklist / common mistakes

- Colors are **linear** everywhere before composite; use HDR-capable pickers
  (the DevUI environment tab does this correctly).
- `sun_dir` points the way light travels — negate for "toward the sun".
- Keep `fog_color_horizon` equal to `sky_horizon` unless you want a visible
  seam at the horizon.
- Day/night hosts: also update `fog_volumetric_sun_color`; `sun_intensity`
  is not applied to god rays.
- Every packet resets the skybox flag to `true` — call `fb.set_draw_skybox(false)` each frame you want it hidden.
- God rays silently turn off when shadows are off.
- `GraphicsSettings::fog_density` is a multiplier around 0.05, not an
  absolute density.
- `bloom_threshold` is post-exposure; retune it only if the old pre-exposure
  intuition (at exposure 1.0 they're identical) doesn't apply.
- Route settings changes through `request_graphics_settings(...)` (or
  `apply_all_graphics_settings` from main-thread hooks) so vsync, window
  mode and renderer state all stay in sync.

---

## Particle Effects System

Bracken ships a data-driven **mesh-particle** effects system: particles are small 3D meshes (shards, blades, motes) simulated on the CPU and drawn GPU-instanced with unlit additive HDR shading, so bright effects feed the bloom chain and glow "for free". The system is split into three layers, and as a host you normally only touch the first one:

| Layer | Files | Namespace | You touch it? |
|---|---|---|---|
| Definitions (pure data) | `src/bracken/effects/effect_definition.hpp` | `bracken` | **Yes** — author these |
| Runtime simulation | `src/bracken/effects/effect_system.hpp/.cpp` | `bracken::systems` | Only if you bypass `SceneRenderer` |
| Rendering | `src/bracken/effects/effect_renderer.hpp/.cpp` | `bracken::render` | No — owned by `SceneRenderer` |

Two important scoping facts up front (both from `docs/effects_system.md`):

- **The engine never does file I/O for effects.** Parsing effect files (JSON or otherwise) is your job; the engine only ever sees populated `EffectDefinition` structs. Build them in code, load them from JSON, or emit them from an editor — all equivalent.
- **Only mesh particles exist.** Every particle renders as its emitter's mesh model (`EmitterDefinition::model`, a `ModelManager` name); there is no sprite/billboard particle type.

### Quick start: spawn a burst effect

Three steps: register the particle model, build a definition, emit spawns through the frame builder. This is lifted from `examples/playground/playground.cpp` (the `pickup_burst_` effect):

```cpp
#include "bracken/effects/effect_definition.hpp"

using namespace bracken;

// 1. Register the mesh the particles will use — BEFORE the first spawn.
//    (In on_init(); Application::models() returns the ModelManager.)
auto& mm = models();
mm.register_model("orb_shard", build_shard_model());

// 2. Build the definition. It MUST outlive every effect spawned from it —
//    the runtime stores raw pointers into it. Keep it as a member / in a
//    long-lived registry, never on the stack of a per-frame function.
EffectDefinition pickup_burst;
pickup_burst.name = "pickup_burst";
pickup_burst.duration = 1.0f;

EmitterDefinition e;
e.name = "shards";
e.model = "orb_shard";               // ModelManager name
e.spawn_mode = SpawnMode::BURST;
e.spawn_count = 26;
e.particle_lifetime = 0.8f;
e.velocity.type = VelocityType::RADIAL;
e.velocity.speed = 260.0f;
e.velocity.gravity = {0.0f, -500.0f, 0.0f};
e.velocity.drag = 0.4f;
e.rotation.rotation_rate = {180.0f, 240.0f, 90.0f};     // degrees/sec
e.appearance.scale_over_lifetime = {CurveType::LINEAR, 1.0f, 0.25f, 0.8f};
e.appearance.opacity_over_lifetime = {CurveType::FADE_OUT_LATE, 1.0f, 0.0f, 0.55f};
e.appearance.color_tint = {1.0f, 0.85f, 0.4f, 1.0f};
pickup_burst.emitters.push_back(std::move(e));
```

```cpp
// 3. When the event happens (in on_simulate(), while building the frame):
fb.spawn_effect(&pickup_burst_, pos, {0, 1, 0});
```

The engine's sim-stage tail applies the sealed packet's spawns to the (engine-owned, sim-stage) `EffectSystem` exactly once, before the same frame's particle update. From then on the effect simulates, terrain-clamps, and renders itself to completion; you do not track it. The renderer never touches the live system — it draws the effect-instance snapshot the sim stage writes into each packet.

The full signature (`src/bracken/scene/frame_builder.hpp`):

```cpp
void spawn_effect(const bracken::EffectDefinition* definition, const glm::vec3& position,
                  const glm::vec3& direction = {1, 0, 0});
```

`direction` matters for `DIRECTIONAL` and `ARC` emitters (spread/arc are oriented along it).

#### Timing

The sim-job tail applies spawns **before** the `EffectSystem` update (`src/bracken/application.cpp::run_sim_job`), so a spawn emitted in `on_simulate` produces its first particles in the same frame's snapshot — which presents when that frame renders. The effect's model is GPU-uploaded in the render pre-pass of the same frame (`ensure_models_uploaded` over the packet's emitter snapshot), so there is no upload hitch.

### Authoring `EffectDefinition`s

Everything lives in `src/bracken/effects/effect_definition.hpp`. Defaults below are the struct initializers.

#### `EffectDefinition`

| Field | Default | Meaning |
|---|---|---|
| `name` | `""` | Identifier (used in logs) |
| `emitters` | empty | `std::vector<EmitterDefinition>` — one or more emitters fired together |
| `duration` | `-1.0f` | Effect-level duration override. `-1` = emitter-driven completion. When `>= 0`, all emitters stop spawning once the effect has been alive this long, and the effect completes as soon as its remaining particles die |

With the default `duration = -1`, an effect is *complete* (and removed) when every emitter has passed its own duration **and** all its particles have died. There is no looping — if you need a looping effect, re-spawn it from the host.

#### `EmitterDefinition`

| Field | Default | Meaning |
|---|---|---|
| `name` | `""` | Identifier (used in the "unknown model" warning) |
| `model` | `""` | `ModelManager` name of the particle mesh — every particle renders as this mesh |
| `spawn_mode` | `SpawnMode::BURST` | `BURST` = all `spawn_count` at once (after `delay`); `CONTINUOUS` = `spawn_rate`/sec |
| `spawn_count` | `1` | Particles per burst (BURST only) |
| `spawn_rate` | `10.0f` | Particles per second (CONTINUOUS only). **Must be > 0** — see gotchas |
| `spawn_volume` | `{0,0,0}` | Box half-extents around the origin; each particle spawns at a uniform random point inside. `{0,0,0}` = point emitter. Use for area effects (rain, snow) instead of many point effects |
| `particle_lifetime` | `1.0f` | Seconds each particle lives |
| `velocity` | — | `VelocityDefinition` (below) |
| `rotation` | — | `RotationDefinition` (below) |
| `appearance` | — | `AppearanceDefinition` (below) |
| `delay` | `0.0f` | Seconds before the emitter starts spawning |
| `duration` | `-1.0f` | Seconds the emitter spawns for; `-1` = use `particle_lifetime` |

**Gotcha — `delay` vs `duration`:** the emitter's active window is `age < duration`, measured from effect spawn, *not* from the end of the delay. A `BURST` with `delay = 1.0` and `particle_lifetime = 0.5` (so effective duration 0.5) goes inactive before the delay elapses and **never fires**. Make sure `duration > delay`.

**Gotcha — CONTINUOUS spawn rate:** `spawn_rate` must be **> 0 and finite**. Invalid rates are rejected at `spawn_effect` with a warning (once per definition) and the emitter stays inert; the update loop guards against them independently, so a bad rate can no longer hang a frame.

#### `VelocityDefinition` — how particles move

```cpp
struct VelocityDefinition {
    VelocityType type = VelocityType::DIRECTIONAL;
    float speed = 100.0f;            // Units per second
    glm::vec3 direction = {1, 0, 0}; // For DIRECTIONAL/CUSTOM
    float spread_angle = 0.0f;       // Random spread in degrees
    glm::vec3 gravity = {0, 0, 0};   // Gravity acceleration
    float drag = 0.0f;               // Velocity damping (0-1)
    // ORBITAL:
    float orbit_radius = 50.0f;
    float orbit_speed = 2.0f;        // Rotations per second
    float orbit_height_base = 0.0f;
    float height_variation = 0.0f;   // Vertical oscillation amplitude
    // ARC (melee slash):
    float arc_radius = 36.0f;
    float arc_height_base = 25.0f;
    float arc_height_amplitude = 15.0f;
    float arc_tilt_amplitude = 0.8f; // radians
};
```

| `VelocityType` | Behavior |
|---|---|
| `DIRECTIONAL` | Move along the **spawn** direction at `speed`, randomized ±`spread_angle` degrees. The spread rotates in the **XZ (ground) plane only** — vertical directions get no useful cone |
| `RADIAL` | Each particle picks a random horizontal (XZ) outward direction at `speed`. Ignores spawn direction — good for pickups/explosions |
| `ORBITAL` | Position computed each frame: circle of `orbit_radius` around the spawn point at `orbit_speed` rev/sec, height `orbit_height_base` (+ `sin`-wobble of `height_variation`). `speed`/gravity/drag ignored. Multiple particles are distributed evenly around the circle |
| `CUSTOM` | Fixed world-space velocity `direction * speed`, regardless of spawn direction |
| `ARC` | Kinematic arc swing around the spawn point (melee slash): sweeps 2 radians over the particle lifetime along `arc_radius`, with a sine height bump (`arc_height_base` + `arc_height_amplitude`) and tilt animation. `speed`/gravity/drag ignored; rotation is fully driven by the arc |

`gravity` and `drag` apply only to the integrating types (`DIRECTIONAL`, `RADIAL`, `CUSTOM`). Drag is a simple per-frame damping `velocity *= (1 - drag * dt)`.

The playground's melee slash (`attack_arc_` in `examples/playground/playground.cpp`) is the canonical ARC example:

```cpp
EmitterDefinition e;
e.name = "slash";
e.model = "orb_shard";
e.spawn_mode = SpawnMode::BURST;
e.spawn_count = 12;
e.particle_lifetime = 0.35f;
e.velocity.type = VelocityType::ARC;
e.velocity.arc_radius = 70.0f;
e.velocity.arc_height_base = 95.0f;
e.velocity.arc_height_amplitude = 30.0f;
e.rotation.face_velocity = true;
e.appearance.opacity_over_lifetime = {CurveType::LINEAR, 1.0f, 0.0f, 0.8f};
e.appearance.color_tint = {0.85f, 0.92f, 1.0f, 1.0f};
// spawn it in front of the player, oriented along the facing direction:
fb.spawn_effect(&attack_arc_, player_pos_ + fwd * 30.0f, fwd);
```

#### `RotationDefinition` — how particles spin

```cpp
struct RotationDefinition {
    glm::vec3 initial_rotation = {0, 0, 0}; // Euler angles, DEGREES
    glm::vec3 rotation_rate = {0, 0, 0};    // DEGREES per second
    bool face_velocity = false;             // Yaw toward movement direction
};
```

Angles are authored in degrees and converted to radians at spawn. With `face_velocity = true`, a particle moving faster than 0.01 units/sec yaws (`rotation.y`) toward its velocity **instead of** integrating `rotation_rate`; below that speed it falls back to `rotation_rate` integration (pinned by `FaceVelocityYawsTowardMovementDirection` / `FaceVelocityIgnoresNearZeroVelocity` in `src/bracken/effects/effect_system.test.cpp`).

#### `AppearanceDefinition` — how particles look

```cpp
struct AppearanceDefinition {
    Curve scale_over_lifetime;      // default CONSTANT 1.0
    Curve opacity_over_lifetime;    // default CONSTANT 1.0
    glm::vec4 color_tint = {1, 1, 1, 1};
    glm::vec4 color_end = {1, 1, 1, 1};
    bool use_color_gradient = false; // lerp color_tint -> color_end over lifetime
};
```

**Gotcha — the alpha channels of `color_tint`/`color_end` are ignored.** Opacity comes exclusively from `opacity_over_lifetime`; the renderer packs `{color.rgb, opacity}` per particle and additive blending fades by scaling RGB. If your particles won't fade, it's the curve, not the tint alpha.

Because rendering is additive into the linear HDR target, `color_tint` is a linear-space emissive color: values near 1.0 with the built-in ×2 emissive boost push past the default bloom threshold (`bloom_threshold`, 1.1 post-exposure), so saturated tints bloom.

#### Curves

```cpp
struct Curve {
    CurveType type = CurveType::CONSTANT;
    float start_value = 1.0f;
    float end_value = 1.0f;
    float fade_start = 0.8f; // FADE_OUT_LATE only
    float evaluate(float t) const; // t = normalized particle age, clamped to [0,1]
};
```

| `CurveType` | Shape |
|---|---|
| `CONSTANT` | Always `start_value` |
| `LINEAR` | Lerp start→end |
| `EASE_IN` | Lerp with `t*t` (slow start) |
| `EASE_OUT` | Lerp with `1-(1-t)²` (fast start) |
| `EASE_IN_OUT` | Smooth-stepped lerp |
| `FADE_OUT_LATE` | Hold `start_value` until `t = fade_start` (default 0.8), then lerp to `end_value` |

Aggregate initialization order is `{type, start_value, end_value, fade_start}` — e.g. `{CurveType::FADE_OUT_LATE, 1.0f, 0.0f, 0.55f}` holds full opacity until 55% of the lifetime, then fades out.

### Particle models

- Emitters reference models **by name** in the `ModelManager` — `ModelManager::get_handle(name)` resolves them. Load or `register_model` them **before the first spawn**.
- **Gotcha — the lookup result is cached permanently.** `EffectRenderer` caches the name→handle resolution per `EmitterDefinition` address (`resolve_emitter_model` in `src/bracken/effects/effect_renderer.cpp`), *including failures*. If the model wasn't registered when the effect first became active, you get one `Effect emitter '...' references unknown model '...'` warning and that emitter **never renders for the rest of the run**, even if you register the model later.
- Unknown/missing models are a **silent visual drop**, not an error: the particles still simulate (and count in `particle_count()`); they just don't draw.
- Meshes are drawn **recentered on their bounding-box center**, so the particle's `position` is the visual center of the mesh regardless of how the model was authored.
- Mesh textures work: if a mesh has a base-color texture it is sampled; untextured meshes use a built-in 1×1 white fallback so the particle is exactly `color_tint`-colored.

### How particles are rendered (what you should expect on screen)

You don't call the `EffectRenderer` yourself, but its behavior shapes how you author effects (`src/bracken/effects/effect_renderer.cpp`, `src/bracken/shaders/particle.vert.hlsl` / `particle.frag.hlsl`):

- **Unlit, unfogged, additive HDR.** Particles ignore all scene lighting and fog. They render into the HDR scene target with effectively (ONE, ONE) additive blending, after the skybox and sorted transparent meshes, with depth *test* on (occluded by opaques) and depth *write* off (they never occlude each other or anything else, and draw order between particles doesn't matter).
- **Emissive boost ×2.** Fragment color is `baseColor.rgb * tint.rgb * 2.0` (`PARTICLE_EMISSIVE_BOOST` in `effect_renderer.cpp`), deliberately above the default bloom threshold (1.1) so effects glow.
- **One instanced draw per (emitter, mesh)** — thousands of particles from one emitter cost one draw call. There is no per-emitter particle cap and no silent draw drop (unlike skinned models' `MAX_BONE_PALETTES`); the instance buffer grows geometrically as needed.

### Terrain clamping

When driven by the engine (the sim-job tail), particles are clamped to the terrain height every update (`particle.position.y = max(y, terrain height)`) — no bounce, they slide on the ground. This is why gravity-heavy bursts pool on the terrain instead of sinking through it. If you drive `EffectSystem` yourself, you choose the sampler (below).

### Driving `EffectSystem` directly (no `SceneRenderer`)

For headless simulation, tests, or a custom renderer, use `bracken::systems::EffectSystem` (`src/bracken/effects/effect_system.hpp`) directly:

```cpp
#include "bracken/effects/effect_system.hpp"

bracken::systems::EffectSystem system;

// Returns the effect's index, or -1 if definition == nullptr.
int idx = system.spawn_effect(&pickup_burst, /*position=*/{0, 0, 0},
                              /*direction=*/{1, 0, 0});

// Per frame — pick ONE of:
system.update(dt);                       // no terrain clamping
system.update(dt, nullptr);              // same (nullptr overload)
system.update(dt, [](float x, float z) { // templated terrain sampler:
    return 0.0f;                         // any callable float(float, float)
});

// Introspection / rendering source:
const std::vector<bracken::systems::EffectInstance>& fx = system.get_effects();
size_t n_effects   = system.effect_count();
size_t n_particles = system.particle_count();
system.clear(); // drop everything
```

The terrain-sampler overload is a **template** — any callable `float(float x, float z)` works with zero `std::function` overhead in the inner particle loop. The engine's sim-job tail passes a lambda over `SceneRenderer::get_terrain_height` (worker-safe const reads).

Iteration structure for custom rendering: `EffectInstance` → `emitters` (`EmitterInstance`) → `particles` (`Particle` with `position`, `rotation` (radians), `scale`, `opacity`, `color`, `age`, `lifetime`). `bracken::render::build_particle_transform_of(particle, center_offset)` (`bracken/render/particle_transform.hpp`) is a free function template over anything with `.position`/`.rotation`/`.scale` if you want the exact world matrix the engine renderer uses — it works on a live `Particle` and on the packet's `EffectParticleSnapshot` alike.

`EffectSystem` is **owned** — drive one instance from one thread (the RNG used at spawn is `thread_local`, so separate systems on separate threads are fine). `update()` internally parallelizes particle integration and compaction on the job pool and joins before returning — this is invisible and deterministic (spawning stays serial, so results are bit-identical at any worker count), with one contract to know: the terrain-sampler callable is invoked **from job workers**, so it must be safe to call concurrently — a pure function of `(x, z)` (the engine's `TerrainRenderer::get_height` lambda qualifies). There are also explicit-pool forms `update(JobSystem&, dt[, fn])` if you own a pool (tests use this); the plain forms service `JobSystem::instance()`.

### Monitoring

- The DevUI Stats panel (F10 in the playground) shows `Particles: N (in M effects)` via `SceneRenderer::frame_effect_count()`/`frame_particle_count()` — counts cached from the rendered packet's snapshot (the live `EffectSystem` is sim-stage owned and is never read from the overlay).
- With Tracy enabled, the live particle count is plotted as `effects.particles` once per update.
- Benchmarks: `build/src/bracken/bracken_benchmarks --benchmark_filter=BM_Effect` (steady-state update at 1k/10k particles, burst spawn, per-curve-type evaluation — see `src/bracken/effects/effect_system.bench.cpp`).

### Common mistakes checklist

- **Definition lifetime**: `EffectInstance`/`EmitterInstance` hold raw `const` pointers into your `EffectDefinition` *and its `emitters` vector elements*. The definition must stay alive and **unmodified** (no push_back into `emitters`, which can reallocate) while any spawned instance is live. Stack-local or reassigned definitions are use-after-free bugs.
- **Registering the particle model after the first spawn** — recoverable: unresolved model lookups are retried every frame (never negatively cached; the miss is warned once per emitter definition), so a late `register_model` is picked up automatically. Registering before the first spawn just avoids invisible first frames.
- **Expecting effects to loop** — there is no loop flag; completion is emitter-driven (or gated by an effect-level `duration >= 0`), and looping must be host-driven.
- **Fading via `color_tint.a`** — ignored; use `opacity_over_lifetime`.
- **`spawn_rate <= 0` (or non-finite)** on a CONTINUOUS emitter — rejected with a warning at spawn; the emitter stays inert.
- **`delay >= duration`** on a BURST emitter — the burst never fires.
- **Emitting the same spawn every frame until it "takes"** — a packet's spawns are consumed exactly once by the engine's sim-stage tail, so emit exactly once per event. (There is no persistent spawn queue anymore.)

---

## Text & UI Rendering

Bracken's 2D layer draws screen-space UI — filled/outlined rectangles, circles, lines, text, and simple buttons — plus world-anchored 3D billboards, directly to the swapchain as the last render pass of every frame. As a host you never touch the renderers directly: you record 2D commands into the frame through `fb.ui()` (a `scene::FrameBuilder::UiLane`) and they ride the packet into `render_frame`. The engine-internal machinery (`render::UIRenderer` for primitives, `render::TextRenderer` for the glyph atlas) is documented at the end for advanced/standalone use.

Coordinates are **pixels**, origin at the **top-left**, Y increasing downward. UI renders with alpha blending, no depth test, on top of the 3D scene (or over a dark clear color when no 3D scene rendered that frame — e.g. a menu state). The Dear ImGui developer overlay (`bracken::dev::DevUI`) is a separate system layered *after* this UI pass via the post-UI callback; it does not go through the packet's UI commands.

### Submitting UI: `fb.ui()`

`FrameBuilder::ui()` returns a small `UiLane` value handle (`src/bracken/scene/frame_builder.hpp`) — no GPU resources, no init, copy it freely or chain calls off `fb.ui()` directly. Text is copied into the packet at emit time, so you can build strings from temporaries freely.

This is the exact pattern the playground uses (`examples/playground/playground.cpp`):

```cpp
#include "bracken/scene/frame_builder.hpp"

class PlaygroundGame : public bracken::Application {
    void on_simulate(bracken::SimFrame& sim) override {
        auto& fb = sim.frame();
        // ... 3D emits, set_camera ...
        build_ui(fb.ui());
    }

    void build_ui(scene::FrameBuilder::UiLane ui) {
        ui.add_filled_rect(10, 10, 280, 122, 0xB0101418);
        ui.add_text("bracken playground", 22, 20, 1.0f, 0xFFFFD080);
        ui.add_text(std::format("FPS {:.0f}", fps()), 22, 44, 0.9f, 0xFFB8C4D0);
    }
};
```

Every packet starts with an empty command stream. Commands render in submission order within their layer (see the layering rules below).

#### Shape commands

All shapes take pixel coordinates and a packed `uint32_t` color (format caveat below).

```cpp
auto ui = fb.ui();
ui.add_filled_rect(x, y, w, h, 0xFF3060A0);
ui.add_rect_outline(x, y, w, h, 0xFFFFFFFF, /*line_width=*/2.0f);
ui.add_circle(cx, cy, radius, 0xFF40A0FF, /*segments=*/24);
ui.add_circle_outline(cx, cy, radius, 0xFF40A0FF, /*line_width=*/2.0f, /*segments=*/24);
ui.add_line(x1, y1, x2, y2, 0xFFFF5050, /*line_width=*/2.0f);
```

Defaults (from `src/bracken/scene/frame_builder.hpp`): `line_width = 2.0f`, `segments = 24`. Notes on how these actually rasterize (`src/bracken/render/ui_renderer.cpp`):

- Rect outlines are four filled strips drawn *inside* the rect bounds.
- Circles are triangle fans (`segments` triangles); outlines are rings (`segments * 6` vertices). If a circle outline's `line_width >= radius`, it silently falls back to a filled circle. Non-positive `line_width`, `radius`, or `segments` draws nothing.
- Lines are screen-space quads extruded by `line_width / 2` perpendicular to the segment; lines shorter than 0.001 px are silently dropped.

#### Text

```cpp
ui.add_text("Orbs collected: 3", 22, 64, /*scale=*/0.9f, /*color=*/0xFFFFC040);
```

`add_text(std::string_view text, float x, float y, float scale = 1.0f, uint32_t color = 0xFFFFFFFF)` — the text is copied into the packet at emit time. `(x, y)` is the **top-left of the line box** (not the baseline); the built-in font is loaded at 18 px and `scale` multiplies the rendered quad size (it is quad scaling of 18 px glyphs, so large scales look soft under the linear filter — there is only one rasterized size).

Text is UTF-8. Kerning is applied between consecutive codepoints; malformed byte sequences render as U+FFFD (replacement character). Glyphs are rasterized on demand into a shared 512×512 atlas, so new strings render the same frame with no per-string textures.

Word-wrapped text returns the total height it consumed so you can stack elements:

```cpp
float used = ui.add_text_wrapped(
    "WASD move | Shift run | mouse-drag orbit | wheel zoom",
    10.0f, static_cast<float>(screen_height()) - 56.0f,
    /*max_width=*/static_cast<float>(screen_width()) - 20.0f,
    /*scale=*/0.9f, /*color=*/0xFFC8D0DC /*, line_height = 0.0f -> 18*scale */);
```

Wrapping is **approximate**: it assumes a fixed 8 px character width (× scale) rather than measuring real glyph advances (`src/bracken/scene/frame_builder.cpp`, `CHAR_WIDTH_BASE = 8.0f`). It breaks at the last space that fits, hard-breaks when a "word" exceeds a full line, and clamps to at least 1 character per line. `line_height <= 0` means "use `18 * scale`". Long proportional text can therefore wrap earlier or later than a true measurement would — leave margin.

#### Buttons

```cpp
ui.add_button(100, 200, 160, 40, "Start", 0xFF203040, /*selected=*/true);
```

A button is a composite: filled rect in `color`, a border (white `0xFFFFFFFF`, 3 px when `selected`; grey `0xFF888888`, 2 px otherwise), and the `label` centered in white at scale 1.0 using real text measurement (`TextRenderer::get_text_width/get_text_height`). There is no hit-testing or input handling — the UI command stream is draw-only; do your own mouse math.

### Colors: one packed format everywhere

Every packed `uint32_t` color in the engine — shapes, outlines, lines, buttons, billboards, text, textured-quad tints, and 3D debug lines — is **ARGB `0xAARRGGBB`** (`0xFFFF0000` opaque red, `0x80000000` half-transparent black). The single pack/unpack pair lives in `bracken/core/color.hpp` (`core::pack_color` / `core::unpack_color`); every renderer decodes through it, so a color literal means the same thing in every API.

One parameter-order trap remains: `UiLane::add_text(text, x, y, scale, color)` but the internal `UIRenderer::draw_text(text, x, y, color, scale)`. If you ever call the renderer directly, don't copy argument lists from frame-building code.

### Layering and draw order

Within a frame the UI pass draws in this fixed order (`UIRenderer::execute`, `SceneRenderer::render_frame` in `src/bracken/scene/scene_renderer.cpp`):

1. All **shape/line/button-body** primitives, in submission order, as one batch.
2. All **3D billboards** from the packet (they are appended to the same primitive batch *after* every 2D UI primitive).
3. All **text**, in submission order.

Consequences:

- Text **always renders on top of** every primitive, even a rect submitted after the text. You cannot draw a shape over text.
- Billboards render over your 2D UI shapes but under all text.
- Among primitives (and among text runs) submission order is painter's order, so overlapping rects layer as expected.

### World-anchored billboards

Billboards are the "health bar over a unit's head" primitive. They are submitted through the **3D emit API** (`fb.add_billboard_3d`, not `fb.ui()`) because they carry a world position, but they draw in the UI pass as screen-space rects — there is no billboard GPU pipeline:

```cpp
// examples/playground/playground.cpp — proximity bar above each orb
fb.add_billboard_3d(pos.x, pos.y + 28.0f, pos.z,
                    /*width=*/46.0f, /*fill_ratio=*/fill,
                    /*fill_color=*/0xFFFFC040, /*bg_color=*/0x60000000,
                    /*frame_color=*/0xFFFFD080);
```

`FrameBuilder::add_billboard_3d(world_x, world_y, world_z, width, fill_ratio, fill_color, bg_color, frame_color)` (`src/bracken/scene/frame_builder.hpp`). Each frame the renderer projects the world point through the camera, and draws three filled rects: a 1 px-inflated frame, a background bar, and a fill bar `fill_ratio` (0–1) of the width. The bar is `width * 2` px wide and `width * 0.4` px tall at the reference distance, scaled by distance and clamped to 0.3×–1.5× so bars never balloon up close or vanish far away. Billboards behind the camera or well outside the view are silently culled. Colors are ARGB. There is no way to attach text or textures to a billboard — for labels, project the position yourself and use `fb.ui().add_text`.

### Fonts: what you get and what you don't

There is **no font-loading API**. At init the text renderer probes a hardcoded list of system font paths (`TextRenderer::init`, `src/bracken/render/text_renderer.cpp`): DejaVu Sans (three common Linux locations), Ubuntu-R, FreeSans, macOS `Helvetica.ttc`, Windows `arial.ttf` — first hit wins, loaded via SDL_ttf at a fixed **18 px** (`font_size_ = 18`, no setter). If none of those exist, the engine logs `Could not load any font, text rendering disabled` and continues: init still succeeds, and every text command is **silently skipped** while shapes keep working. If your text isn't appearing on a minimal container/CI image, check the log for that warning and install DejaVu.

Because there is a single font at a single size, `add_text`'s `scale` is bitmap scaling — crisp at 1.0, progressively blurrier above it.

### Budgets and silent-drop behaviors

These are per-frame limits; exceeding them does not crash, but content disappears:

- **Text: 4096 glyph quads per frame** (`TextRenderer::MAX_TEXT_QUADS`, `src/bracken/render/text_renderer.hpp`), shared across all strings. Overflow drops the remaining glyphs and logs `Text quad budget (4096) exhausted - dropping glyphs this frame` once per frame. Spaces cost no quads (advance only).
- **Primitives: unbounded.** The CPU vertex batch has no cap, and the GPU vertex buffer (initial capacity 16384 vertices — ~2,730 rects at 6 vertices each; a circle is `segments * 3`, a circle outline `segments * 6`) **grows geometrically on demand** inside `execute()` when a frame outgrows it. Only if the growth allocation itself fails is the batch truncated, with a warning.
- **Glyph atlas: 512×512.** If enough distinct codepoints accumulate to fill it, all glyphs are evicted and re-rasterized on demand — logged, and text may glitch for one frame. Steady-state UIs never hit this; rotating through huge CJK/emoji sets can.

### Advanced: the renderers underneath

You normally never construct these — `scene::SceneRenderer` owns the `UIRenderer` privately with no public accessor. Text measurement **is** reachable from a host going through `Application`: `SceneRenderer::text_width(text, scale)` / `text_height(scale)` are passthroughs to the UI renderer's glyph metrics (both return 0 until the renderer is initialized and a font loaded). What follows matters only if you embed the renderers in your own frame loop.

`render::UIRenderer` (`src/bracken/render/ui_renderer.hpp`) is a record-then-execute batcher:

```cpp
render::UIRenderer ui;
if (!ui.init(device, pipeline_registry, width, height)) { /* fatal */ }

// per frame, outside any render pass:
ui.begin(cmd);                       // resets batches; frame housekeeping
ui.draw_filled_rect(10, 10, 200, 40, 0xFF203040);
ui.draw_text(label, 20, 20, 0xFFFFFFFF, 1.0f);   // note: color before scale
ui.end();                            // recording done (no GPU work yet)
ui.execute(cmd, swapchain_texture, /*clear_background=*/false);
```

Contracts you must honor:

- `execute()` performs the copy passes (vertex + atlas uploads) and then its own render pass on the swapchain, so **call it outside any other pass**, after `end()`, before submission. `clear_background = true` clears to a dark blue-grey (`{0.1, 0.1, 0.15, 1.0}`) instead of loading existing content — used when no 3D scene rendered.
- `UIRenderer::draw_text` (and `draw_button`'s label) stores a **`std::string_view` — no copy**. The referenced characters must stay alive until `execute()` returns. Packet-arena text satisfies this automatically (the packet outlives the frame); a `std::format(...)` temporary passed directly to `UIRenderer::draw_text` is a use-after-free.
- Call `set_screen_size(w, h)` on resize; it rebuilds the ortho projection. (Under `SceneRenderer` this happens automatically every frame from the swapchain size in `begin_frame`.)
- Everything is main-thread-only, like the rest of the renderer.

For measurement, `UIRenderer::text_width(text, scale)` / `text_height(scale)` wrap the glyph metrics directly (0 when no font loaded); the underlying `render::TextRenderer` (`src/bracken/render/text_renderer.hpp`) also exposes `get_text_width`/`get_text_height` via `ui.text_renderer()`.

`UIRenderer::draw_textured_quad(texture, x, y, w, h, tint, ...)` draws an arbitrary `SDL_GPUTexture*` in the UI pass (tint is packed ARGB via `core/color.hpp`) — useful for minimaps, portraits, or render-target previews.

Text drawing is exclusively the batched flow the engine uses (`queue_text_draw` → `upload_queued_text` before the pass → `draw_queued_text` inside it); the legacy immediate paths were removed.

---

## Camera Controller & Camera System

Bracken ships one camera: a smooth third-person follow/orbit camera. It comes in two layers:

- `bracken::systems::CameraController` (`src/bracken/systems/camera_controller.hpp`) — the abstract interface your game code talks to. It deliberately exposes only *generic* framing primitives: target tracking, yaw/pitch/zoom, soft-focus, shake/kick, and an FOV bias. Game feel (combat kicks, hit reactions, sprint FOV ramps) is composed **on top** of these primitives in your own code — no combat or gameplay concepts exist in the engine header. This is the engine's game-agnostic rule (see `docs/engine_design_notes.md`, "Architecture: the engine is game-agnostic").
- `bracken::systems::CameraSystem` (`src/bracken/systems/camera_system.hpp` / `.cpp`) — the concrete implementation: a spring-smoothed orbit camera with look-ahead, auto-centering, terrain/obstacle avoidance, shake accumulation, and speed-based dynamic FOV.

`Application` (`src/bracken/application.hpp`) creates a `CameraSystem` for you in `init_renderer()` and exposes it to subclasses through the protected facade `systems::CameraController& camera()`.

### Getting a camera and driving the frame

If you subclass `Application`, the camera already exists after `init_renderer()`. You are responsible for three things every frame, in `on_simulate` (the camera is tier sim-stage under `run()` — update it from sim code, nowhere else):

1. feed it the follow target and target velocity,
2. call `update(dt)` — **the engine does not call it for you**,
3. hand its output to the frame via `get_camera_state()`.

This is exactly what the playground does (`examples/playground/playground.cpp`, `PlaygroundGame::update_camera`):

```cpp
void PlaygroundGame::update_camera(bracken::SimFrame& sim) {
    const bracken::InputSnapshot& in = sim.input();
    auto& cam = camera();                       // systems::CameraController&
    cam.set_screen_size(screen_width(), screen_height());
    cam.set_yaw(in.camera_yaw);                 // mouse-driven orbit (snapshot fields)
    cam.set_pitch(in.camera_pitch);
    cam.adjust_zoom(in.camera_zoom_delta);
    cam.set_target(player_pos_ + glm::vec3(0, 110, 0));
    cam.set_target_velocity(player_vel_);
    cam.update(sim.dt());
    // Camera-relative movement axes for the input system (deferred —
    // applied to the InputHandler at the next frame fence):
    sim.set_camera_forward(cam.get_forward().x, cam.get_forward().z);
}
```

And while building the frame:

```cpp
sim.frame().set_camera(camera().get_camera_state());
```

`get_camera_state()` builds a `scene::CameraState` (`src/bracken/contracts/camera_state.hpp`) — `view`, `projection`, `view_projection = projection * view`, and `position` — which is the entire contract between the camera and `SceneRenderer`. Nothing forces you to use `CameraSystem`; any code that can fill a `CameraState` can drive the renderer.

You can also construct a `CameraSystem` directly (no GPU or window needed) — the unit tests do exactly that (`src/bracken/systems/camera_system.test.cpp`):

```cpp
#include "bracken/systems/camera_system.hpp"

bracken::systems::CameraSystem camera;
camera.set_target({100.0f, 0.0f, 200.0f});
for (int i = 0; i < 600; ++i) camera.update(1.0f / 60.0f);   // converge
glm::vec3 pos = camera.get_position();
```

### Following a target

```cpp
cam.set_target(player_pos_ + glm::vec3(0, 110, 0)); // world-space point to frame
cam.set_target_velocity(player_vel_);               // world units/second
```

- The target position is smoothed (critically-damped spring) before use, so teleporting the target produces a smooth catch-up rather than a snap.
- **Gotcha — the target's own Y is ignored for framing height.** The look-at height is computed as `terrain_height(x, z) + config.height_offset` (and terrain height is 0 if you never set a terrain callback). If your target flies or stands on elevation without a terrain callback, the camera keeps looking at `height_offset` above y = 0. This is intentional (the test file pins it: "the look-at height is terrain_y + height_offset; the target's own y is intentionally ignored").
- **Look-ahead**: the camera shifts its framing along the target's velocity direction, up to `config.look_ahead_dist` at full speed (speed is normalized against `config.look_ahead_speed_ref`, default 300 units/s). Feed a zero velocity and the offset smoothly decays. Set `config.look_ahead_dist = 0` to disable.

### Orbit control: yaw, pitch, zoom

Two flavors, with an important behavioral difference:

```cpp
// Deltas — always mark "the player gave input this frame" (resets the auto-center timer):
cam.rotate_yaw(delta_degrees);
cam.rotate_pitch(delta_degrees);
cam.adjust_zoom(delta);          // negative = zoom in

// Direct sets — mark input only when the value actually changes:
cam.set_yaw(degrees);
cam.set_pitch(degrees);
```

- Yaw is normalized into `[0, 360)`. Pitch is clamped to `[config.pitch_min, config.pitch_max]` (defaults −70…+70; positive pitch puts the camera above the target looking down).
- Zoom distance is clamped to `[config.zoom_min, config.zoom_max]` (defaults 80–600).
- All three are spring-smoothed toward the input values inside `update()`, so a direct `set_yaw` still animates rather than snapping (with the default near-zero `rotation_lag` the catch-up is essentially instant).
- `set_yaw`/`set_pitch` register as input **on change** — a host that pipes absolute mouse angles through `set_yaw` every frame no longer fights auto-centering while the value is static, and auto-center engages normally once the angles stop changing.

The playground's smoke mode self-drives an orbit with the delta API:

```cpp
camera().rotate_yaw(35.0f * dt);   // slow continuous orbit
```

### Configuration: `CameraModeConfig`

All per-mode tuning lives in one aggregate (`src/bracken/systems/camera_controller.hpp`). Defaults:

```cpp
struct CameraModeConfig {
    float distance = 280.0f;         // base orbit distance from target
    float height_offset = 90.0f;     // look-at height above terrain
    float shoulder_offset = 40.0f;   // horizontal over-the-shoulder offset
    float fov = 55.0f;               // base field of view (degrees)
    float position_lag = 0.001f;     // position spring smooth-time (seconds)
    float rotation_lag = 0.001f;     // rotation spring smooth-time (seconds)
    float look_ahead_dist = 60.0f;   // max velocity look-ahead (world units)
    float pitch_min = -70.0f;        // min pitch (camera below target, looking up)
    float pitch_max = 70.0f;         // max pitch (camera above, looking down)
    float auto_return_speed = 1.5f;  // auto-center blend rate (1/s)
    bool auto_center_enabled = true;
    float auto_center_delay = 2.0f;  // seconds without input before auto-centering
    float zoom_min = 80.0f;          // minimum camera distance (zoom clamp)
    float zoom_max = 600.0f;         // maximum camera distance (zoom clamp)
    float near_plane = 5.0f;         // projection near clip plane
    float far_plane = 15000.0f;      // projection far clip plane
    float ground_clearance = 25.0f;  // minimum camera height above terrain
    float sprint_fov_bonus = 8.0f;   // extra FOV (degrees) at/above sprint_speed_ref
    float look_ahead_speed_ref = 300.0f; // target speed at which look-ahead saturates
    float sprint_speed_ref = 400.0f;     // target speed at which the sprint FOV bonus saturates
};
```

Defaults are tuned for a world scale of roughly 1 unit = 1 cm (the playground's scale); differently-scaled games should adjust distances, planes, and speed references accordingly.

Read-modify-write it; `set_config` replaces the whole struct:

```cpp
auto cfg = cam.get_config();
cfg.distance = 380.0f;
cfg.height_offset = 120.0f;
cam.set_config(cfg);          // also resets the zoom distance to cfg.distance
```

(That's verbatim the playground's `on_init` setup.)

Gotchas:

- **`set_config` resets zoom.** It sets the internal zoom target back to `config.distance` (clamped to `[zoom_min, zoom_max]`), discarding any user zoom the player applied via `adjust_zoom`. Swap configs sparingly or re-apply the zoom delta after.
- **`position_lag`/`rotation_lag` are spring smooth-times in seconds, and *smaller* means *snappier*.** The value is `smoothTime` in a critically-damped spring (see `docs/technical_spec.md`): the defaults of 0.001 are near-instant; raise them (e.g. 0.1–0.3) to add floaty lag.

### Auto-centering

With `config.auto_center_enabled` (default on), after `config.auto_center_delay` seconds (default 2) without input (input = `rotate_yaw`/`rotate_pitch`/`adjust_zoom` calls, or a `set_yaw`/`set_pitch` that changes the value), the camera gradually swings its yaw to sit behind the target's movement direction, at rate `config.auto_return_speed`. It only engages while the target is actually moving (speed ≥ 10 units/s). Any input cancels it immediately.

### Soft-focus target (action-cam soft-lock)

Bias the camera yaw toward a point of interest without hard-locking:

```cpp
glm::vec3 boss_pos = ...;            // storage YOU own, must outlive the focus
cam.set_focus_target(&boss_pos);     // raw pointer — see lifetime warning
cam.set_focus_strength(0.6f);        // 0..1; 0 disables
// later:
cam.set_focus_target(nullptr);       // clear
```

While active, the camera continuously pulls its yaw toward the direction from the follow target to the focus point, scaled by a smoothed strength. Strength ramps in/out smoothly, so toggling doesn't snap.

**Lifetime warning**: `set_focus_target` stores the raw pointer and dereferences it inside every `update()`. The pointed-to `glm::vec3` must stay valid until you pass `nullptr` (or set another pointer). Pointing it at a temporary or an element of a reallocating `std::vector` is a use-after-free.

`set_focus_strength` clamps its argument to [0, 1] (values above 1 would over-rotate and oscillate); `get_focus_strength()` reads it back.

### Camera shake

Two generic primitives; compose game feel on top of them:

```cpp
// Typed shakes (defaults: duration = 0.3f):
cam.add_shake(systems::ShakeType::Impact, 4.0f, 0.25f);   // playground: orb pickup
cam.add_shake(systems::ShakeType::Heavy,  2.0f, 1.0f);    // sustained rumble
cam.add_shake(systems::ShakeType::Subtle, 1.0f, 2.0f);    // idle micro-movement

// Directional kick along a world direction (default duration = 0.2f):
cam.add_directional_shake(attack_dir, 6.0f, 0.15f);       // playground: attack kick
```

- `ShakeType` (`src/bracken/systems/camera_controller.hpp`): `Impact` (quick punch), `Heavy` (low-frequency rumble), `Directional` (along a direction — used implicitly by `add_directional_shake`), `Subtle` (idle micro-movements).
- **Intensity is in world units** of positional offset (pre-falloff, pre per-axis scaling). The playground world uses intensities around 4–6; the unit tests use 20 to make displacement obvious.
- Shakes stack additively; each decays with a quadratic falloff over its duration and is removed automatically.
- At most **8** shakes are active at once (`CameraSystem::MAX_ACTIVE_SHAKES`); adding a 9th silently evicts the *oldest*.
- A zero-length direction passed to `add_directional_shake` falls back to +X.
- Shake is applied **before** collision avoidance, so terrain/obstacle clamps contain the shaken position — a large shake can no longer push the camera into terrain.

The dev overlay's Camera tab (`src/bracken/dev/dev_panels.cpp`, F10 in the playground) has live buttons for all four shake types plus intensity/duration sliders — the quickest way to tune feel.

### Field of view

The effective FOV each frame is:

```text
config.fov  +  sprint bonus (up to +config.sprint_fov_bonus as target speed approaches config.sprint_speed_ref, quadratic)  +  fov_bias
```

smoothed over ~0.2 s. Your only direct hook is the bias:

```cpp
cam.set_fov_bias(+10.0f);   // e.g. sprint or aim-down-sights widening
cam.set_fov_bias(0.0f);     // clear
```

The bias is a persistent additive degree offset — the camera does *not* animate it over time for you; if you want a ramp, vary the value yourself each frame (this is by design: "Caller manages over-time variation", `src/bracken/systems/camera_system.hpp`). The dev overlay's Camera tab has an "FOV bias" slider wired to the same call.

The sprint bonus is always active but fully configurable — set `config.sprint_fov_bonus = 0` to disable it, or tune `config.sprint_speed_ref` for your game's speed scale.

### Terrain and obstacle avoidance

Both are optional host callbacks — the engine has no world knowledge:

```cpp
// Height of the ground at world (x, z) — keeps the camera out of terrain:
cam.set_terrain_height_func([this](float x, float z) {
    return get_terrain_height(x, z);   // playground: samples its heightmap
});

// Line-of-sight blocker (CameraSystem only, not on the CameraController
// interface): return true and fill hit_point if [from → to] is obstructed.
camera_system.set_collision_func(
    [&](const glm::vec3& from, const glm::vec3& to, glm::vec3& hit_point) {
        return world_raycast(from, to, hit_point);
    });
```

- Terrain: the camera samples the height under itself plus 6 points along the boom arm and lifts itself to keep at least `config.ground_clearance` (default 25 units) of clearance. Without the callback, nothing stops the camera from sinking below ground.
- Obstacles: on a blocked ray the camera is pulled to ~20 units in front of the hit point and eases back out over ~a fifth of a second once clear.
- The callbacks are `std::function`s invoked (possibly several times) inside every `update()` — keep them cheap, and make sure anything they capture outlives the camera or is cleared before destruction.
- **`set_collision_func` is not reachable through `Application::camera()`** — the facade returns the abstract `CameraController&`, and `Application` owns the concrete `CameraSystem` privately. If you need wall avoidance with the `Application` facade today, you don't have a hook; hosts that own their own `CameraSystem` do.

### Screen size and projection

```cpp
cam.set_screen_size(screen_width(), screen_height());   // call on resize (or every frame)
```

Defaults to 1280×720; degenerate sizes are clamped to 1×1 so the projection never NaNs. The projection is right-handed, **reversed-Z zero-to-one depth** with `config.near_plane` (default 5) and `config.far_plane` (default 15000) *swapped* in the `glm::perspective` call (`src/bracken/systems/camera_system.cpp`, `compute_matrices`). Near maps to depth 1, far to depth 0 — this matches the engine-wide convention (`GLM_FORCE_DEPTH_ZERO_TO_ONE`, `GREATER_OR_EQUAL` depth tests, clear-to-0). Don't hand-build a standard-Z projection into `CameraState` unless you also understand what breaks (skybox, AO, fog all assume depth 0 = far).

### Reading camera state

From the `CameraController` interface: `get_view_matrix()`, `get_projection_matrix()`, `get_position()`, `get_forward()` (unit length), and `get_camera_state()`.

`CameraSystem` adds concrete-only getters used by tests and tools: `get_right()`, `get_yaw()` / `get_pitch()` (smoothed), `get_input_yaw()` / `get_input_pitch()` (raw), `get_current_fov()`, `get_current_distance()`, `get_target_position()` (the *smoothed* follow point). The dev overlay works around the interface's lack of yaw/pitch getters by deriving them from `get_forward()`.

A common pattern from the playground — camera-relative aiming:

```cpp
glm::vec3 dir = camera().get_forward();   // where the camera looks
```

### Dev overlay integration

`Application::init_dev_ui()` binds `&camera()` into the built-in Bracken Dev window (`src/bracken/dev/dev_panels.cpp`). The **Camera** tab shows position/forward/derived yaw+pitch/config readouts, shake test buttons, an FOV-bias slider, and a focus-strength slider (which only does something if your game has set a focus *target*).

### Quick gotcha checklist

- Call `cam.update(dt)` yourself every frame; nothing else will.
- `dt` is clamped to 0.1 s internally — long hitches slow the camera instead of exploding it.
- Look-at height ignores the target's Y; it's `terrain_y + height_offset`.
- `set_config` resets player zoom to `config.distance` (clamped to `config.zoom_min`/`zoom_max`).
- `set_yaw`/`set_pitch` reset the auto-center timer only when the value changes; `rotate_*`/`adjust_zoom` always do.
- `set_focus_target` keeps a raw pointer — your storage, your lifetime.
- Max 8 concurrent shakes; oldest is silently dropped.
- Fast targets widen the FOV automatically unless you set `config.sprint_fov_bonus = 0`.
- Single-threaded: drive the camera from the main loop only; there is no internal locking.

---

## Procedural Generation: Trees & Rocks

### Overview

`src/bracken/procedural/` contains two CPU-side mesh generators:

- **`bracken::procedural::TreeGenerator`** (`src/bracken/procedural/tree_generator.hpp`) — builds trees with a breadth-first recursive branching algorithm (based on the ez-tree algorithm by dgreenheck). Output: a branch mesh, an optional leaf mesh, and an optional shadow-only "canopy proxy" mesh.
- **`bracken::procedural::RockGenerator`** (`src/bracken/procedural/rock_generator.hpp`) — builds rocks by evaluating a signed-distance field and extracting a mesh with marching cubes (based on przemyslawzaworski/Unity-Procedural-Rock-Generation, MIT). Output: a single flat-bottomed rock mesh.

Both are stateless static classes returning `std::unique_ptr<bracken::Model>` — the same `Model` type produced by `ModelLoader::load_glb`, so generated assets plug into the normal model pipeline: register with `ModelManager`, draw with `FrameBuilder::draw_model`. Generation is deterministic per parameter set (same params + seed → identical mesh), runs entirely on the CPU, and touches no GPU or global engine state.

The playground example (`examples/playground/playground.cpp`) is the canonical consumer: it builds all its trees and rocks in code at startup.

### Quick start: generate, register, draw

Inside an `Application` subclass (after `init_renderer` succeeded, so `models()` has a GPU device):

```cpp
#include "bracken/procedural/rock_generator.hpp"
#include "bracken/procedural/tree_generator.hpp"

// At init — generate once, register once (from examples/playground/playground.cpp):
auto& mm = models();
mm.register_model("tree_oak", procedural::TreeGenerator::generate_oak(7));
mm.register_model("tree_pine", procedural::TreeGenerator::generate_pine(21));
mm.register_model("tree_birch", procedural::TreeGenerator::generate_birch(33));
mm.register_model("rock_boulder", procedural::RockGenerator::generate_boulder(11));
mm.register_model("rock_slate", procedural::RockGenerator::generate_slate(29));
mm.register_model("rock_cluster", procedural::RockGenerator::generate_cluster(43));

// Resolve handles once at placement time, not per frame:
ModelHandle oak = mm.get_handle("tree_oak");
```

Each frame, submit the placed instances:

```cpp
// on_simulate(), per placed prop (fb = sim.frame()):
fb.draw_model(prop.model, prop.transform);   // ModelHandle + glm::mat4
```

Static models submitted through `draw_model` are automatically batched into one instanced draw per mesh per model type, frustum-culled against the model's precomputed bounding sphere, and included in the cascaded shadow passes. Nothing extra is needed for trees/rocks specifically.

`ModelManager::register_model` (`src/bracken/model_loader.hpp`) takes ownership, uploads all meshes to the GPU if the device is set, and returns a `ModelHandle`. **Gotcha:** the upload frees the CPU-side `vertices`/`indices`/`texture_pixels` to save memory — if you want to post-process the generated geometry or tweak per-mesh material flags, do it *before* calling `register_model`. Procedural models have no source path, so hot-reload (`reload_model`) is a no-op for them (returns `false`).

### What a generated tree Model contains

`TreeGenerator::generate` returns a `Model` with **up to three meshes**, in this order (a mesh is omitted if it would be empty, so indices shift — e.g. the `dead` preset has only the branch mesh):

1. **Branches** — a closed generalized-cylinder mesh for trunk + all branch levels. Casts shadows and renders in the main pass (default `Mesh` flags).
2. **Leaves** (present when leaf geometry was generated) — camera-independent quad "cards" attached to terminal branches. `cast_shadows = false`: the raw leaf quads are skipped by every shadow pass because alpha-tested leaf rasterization in shadow depth is expensive; the proxy mesh below replaces them.
3. **Canopy shadow proxy** (present when any leaf clumps exist) — `shadow_only = true`, `cast_shadows = true`: skipped in all main/color passes, rendered only into shadow maps. Per leaf cluster it emits three mutually perpendicular quads (double-winding, so they cast from any light angle despite the shadow pipelines' front-face culling). If a leaf texture was supplied, the shadow depth shader (`src/bracken/shaders/shadow_depth.frag.hlsl`) alpha-tests it at a hardcoded 0.5 cutoff, punching leaf-shaped holes into the shadow. Without a texture the proxy casts a solid cluster-sized shadow.

The model's AABB is computed from all vertices, with `min_y` clamped to `0` (the trunk base is always at the local origin), and `compute_bounding_sphere()` is called — so frustum culling works out of the box.

A generated rock `Model` contains exactly **one mesh** with default flags (casts shadows, renders in main pass). The rock's flat base sits at local `Y = 0` and it is centered on the XZ origin (the AABB is forced symmetric in X/Z so `build_model_transform`-style pivot centering is a no-op).

### TreeGenerator

#### Presets

All presets are `[[nodiscard]] static std::unique_ptr<Model>` with signature `(uint32_t seed = 0, const std::string& texture_base_path = "")`:

| Preset | Character | Textures requested under `texture_base_path` |
|---|---|---|
| `generate_oak` | thick trunk, wide crown, 3 levels | `bark/oak_color_1k.jpg`, `leaves/oak_color.png` |
| `generate_pine` | tall conical, `evergreen = true` | `bark/pine_color_1k.jpg`, `leaves/pine_color.png` |
| `generate_dead` | sparse, gnarled, 2 levels, **no leaves** | `bark/birch_color_1k.jpg` only |
| `generate_willow` | drooping (downward `force_direction = {0,-0.3,0}`) | `bark/willow_color_1k.jpg`, `leaves/ash_color.png` |
| `generate_birch` | tall, thin, white-ish bark | `bark/birch_color_1k.jpg`, `leaves/aspen_color.png` |
| `generate_maple` | round dense canopy, autumn orange-red leaves | `bark/oak_color_1k.jpg`, `leaves/oak_color.png` |
| `generate_aspen` | slender columnar, small yellow-green leaves | `bark/birch_color_1k.jpg`, `leaves/aspen_color.png` |

`texture_base_path` is a directory expected to contain `bark/` and `leaves/` subdirectories with those exact filenames. **The engine repo ships no such textures** — pass `""` (the default, as the playground does) to get vertex-colored trees, or point it at your own asset directory.

```cpp
auto oak = procedural::TreeGenerator::generate_oak(42);                    // vertex colors
auto tex_oak = procedural::TreeGenerator::generate_oak(42, "assets/trees"); // textured
```

#### TreeParams reference

For full control, fill a `TreeParams` and call `TreeGenerator::generate(params)`. Per-level arrays are indexed by branch level: **index 0 = trunk, 1+ = successive child generations.**

```cpp
procedural::TreeParams p;
p.seed = 1;
p.levels = 2;
p.leaf_count = 5;
auto model = procedural::TreeGenerator::generate(p);   // never returns nullptr
```

| Field | Default | Meaning |
|---|---|---|
| `seed` | `0` | RNG seed. Internally `0` is remapped to `1`, so seeds 0 and 1 produce identical trees. |
| `levels` | `3` | Branch recursion depth; `0` = trunk only. Out-of-range values (valid range `[0, 3]` = `MAX_LEVELS - 1`) are **clamped with a warning** — no out-of-bounds reads. |
| `length[4]` | `{20, 20, 10, 1}` | Branch length per level (world units). |
| `radius[4]` | `{1.5, 0.7, 0.7, 0.7}` | Level 0: trunk base radius. Levels 1+: a *multiplier* on the parent's radius at the attachment point. |
| `taper[4]` | `{0.7, 0.7, 0.7, 0.7}` | Linear radius reduction along the branch (fraction of base radius lost by the tip). Ignored in `evergreen` mode (always tapers fully to 0). Terminal branches' last ring is pinched to radius 0.001 regardless. |
| `children[4]` | `{7, 7, 5, 0}` | Number of child branches spawned per branch of that level. |
| `angle[4]` | `{0, 70, 60, 60}` | Child branch tilt from the parent axis, **degrees**. |
| `gnarliness[4]` | `{0.15, 0.2, 0.3, 0.02}` | Per-section random bend amplitude; internally scaled up for thin branches (`max(1, 1/sqrt(radius))`). |
| `twist[4]` | `{0, 0, 0, 0}` | Per-section rotation about the branch axis, **radians** (unlike `angle`, which is degrees). |
| `sections[4]` | `{12, 10, 8, 6}` | Rings along each branch (length subdivision). |
| `segments[4]` | `{8, 6, 4, 3}` | Vertices around each ring (radial subdivision; +1 duplicate for the UV seam). |
| `start[4]` | `{0, 0.4, 0.3, 0.3}` | Children of that level attach at a random parametric position in `[start[level], 1]` along the parent. |
| `force_direction` | `{0, 1, 0}` | Global growth direction; branches are slerped toward it each section. |
| `force_strength` | `0.01` | Force amount, divided by section radius (thin branches bend more), clamped to 1. Negative-Y directions droop (see the willow preset). |
| `leaf_count` | `1` | Leaf cards per terminal branch. `0` disables the leaf mesh (and the shadow proxy). |
| `leaf_size` | `2.5` | Card width/height (a square quad). |
| `leaf_size_variance` | `0.7` | Random size factor: `size = leaf_size * (1 + U(-variance, +variance))`. |
| `leaf_angle` | `10` | Card tilt from the branch axis, degrees. |
| `leaf_start` | `0` | Leaves attach in `[leaf_start, 1]` along the branch. |
| `double_billboard` | `true` | Emit a second quad rotated 90° about the card's local Y (cross shape). Doubles leaf vertex count. |
| `bark_color` | `{0.45, 0.32, 0.22, 1}` | Linear-space vertex color for branches when untextured. |
| `leaf_color` | `{0.2, 0.5, 0.15, 1}` | Linear-space vertex color for leaves when untextured. |
| `bark_texture_path` / `leaf_texture_path` | `""` | Image files loaded via SDL_image. Empty = vertex colors. |
| `evergreen` | `false` | Conifer mode: branch radius tapers linearly to 0 (ignoring `taper[]`), child branches shorten toward the top (`length *= 1 - attach_position`). |

#### Textures vs vertex colors

When a texture path is set and the image loads successfully, all vertex colors of that mesh are set to white (so the texture isn't tinted) and the image — converted to RGBA32 — is stored in `Mesh::texture_pixels` for deferred GPU upload (uploaded as sRGB with mipmaps by `ModelLoader::upload_to_gpu`). When no texture is set, or the load **fails** (warned via `BRACKEN_LOG_WARN`, "falling back to vertex colors"), the mesh keeps its per-vertex colors — a failed load no longer produces a white tree.

Textured **leaves** additionally get `material.alpha_mode = AlphaMode::Mask` with `alpha_cutoff = 0.5` (transparent texels discard instead of rendering as opaque rectangles), and leaf meshes are always `material.double_sided = true` (flat cards viewed from either side would otherwise vanish from behind). No manual patching is needed for cutout leaves.

### RockGenerator

#### Presets

All presets: `(uint32_t seed = 0)`. Unlike the tree presets, **`seed == 0` is remapped to a per-preset default seed** (the "signature look" of that preset). Rocks are always vertex-colored — there is no texture parameter.

| Preset | Default seed | `steps` | `smoothness` | displacement (scale / spread) | Color |
|---|---|---|---|---|---|
| `generate_boulder` | 880 | 20 | 0.05 | 0.15 / 10 | mid gray `{0.5, 0.48, 0.45}` |
| `generate_slate` | 123 | 15 | 0.03 (sharper edges) | 0.08 / 15 | darker gray `{0.42, 0.4, 0.38}` |
| `generate_spire` | 456 | 25 | 0.08 (smoother blends) | 0.1 / 8 | lighter `{0.55, 0.52, 0.48}` |
| `generate_cluster` | 789 | 30 | 0.12 (merged blobs) | 0.2 / 12 | `{0.48, 0.45, 0.42}` |
| `generate_mossy` | 321 | 18 | 0.06 | 0.18 / 7 | greenish `{0.35, 0.42, 0.32}` |

All presets use `resolution = 40` and `scale = 1`.

#### RockParams reference

```cpp
procedural::RockParams p;
p.seed = 880;
p.resolution = 80;         // more detail, ~8x the cost of 40
auto rock = procedural::RockGenerator::generate(p);
if (!rock) { /* degenerate params produced no surface */ }
```

| Field | Default | Meaning |
|---|---|---|
| `seed` | `880` | Drives the SDF's pseudo-random sphere placement. Deterministic. Hashed internally (avalanche hash masked to `[0, 2^20)`), so the full `uint32` range yields distinct rocks. |
| `resolution` | `40` | Marching-cubes grid resolution (`40` = fast, `80` = detailed). Cost scales with the cube of this. |
| `scale` | `1.0` | Uniform scale applied to output positions. The unscaled rock spans roughly the `[-0.7, 0.7]` sampling cube. Prefer scaling at draw time via the instance transform (as the playground does) so one mesh serves many sizes. |
| `steps` | `20` | SDF iteration count — each step smooth-blends another pseudo-random sphere into the shape; more steps = more complex silhouette (and more CPU). |
| `smoothness` | `0.05` | Smooth-min blend width between the SDF spheres; small = sharp creases, large = merged blobs. |
| `displacement_scale` | `0.15` | Surface noise amplitude (triplanar 6-octave value-noise FBM). `0` disables displacement entirely (skips the extra SDF evaluations). |
| `displacement_spread` | `10.0` | Surface noise frequency. |
| `color` | `{0.5, 0.48, 0.45, 1}` | Linear-space vertex color of every vertex. |

The scalar field clips everything below `y = -0.05` (unit space), producing a flat underside; after extraction all vertices are shifted so that flat base sits exactly at `Y = 0`. Normals are smooth (central-difference gradient of the field), and UVs are triplanar-projected per triangle by dominant axis — usable if you later assign a texture yourself, but untested by any in-tree content.

Unlike trees, `RockGenerator::generate` **can return `nullptr`** if the field produces no surface; the presets use known-good parameters.

### Placing generated props on terrain

Both generators put the "rests on the ground" plane at local `Y = 0`, so a plain translate-to-terrain-height works. The playground's procedural fallback scene sinks props slightly and randomizes yaw/scale (`examples/playground/playground.cpp`, the fallback branch of `scatter_wilds`):

```cpp
glm::mat4 t = glm::translate(glm::mat4(1.0f), pos - glm::vec3(0, sink, 0));
t = glm::rotate(t, rand01() * 6.28f, glm::vec3(0, 1, 0));
t = glm::scale(t, glm::vec3(scale));
props_.push_back({handle, t});
// playground values: trees scale 4–7 with sink 2; rocks scale 8–22 with sink 0.4*scale
```

`bracken::build_model_transform` (`src/bracken/model_utils.hpp`) also works: its pivot uses the XZ bounding-box center and `min_y`, and both generators arrange their bounds (trees clamp `min_y = 0`; rocks force a symmetric X/Z AABB and `min_y = 0`) so the centering does not shift the mesh.

### Determinism and seeding

- **Trees:** a private Park–Miller LCG (`TreeRNG`, `state = state * 16807 % 2147483647`) seeded from `TreeParams::seed`; seed `0` maps to `1`. Two calls with identical `TreeParams` produce bit-identical geometry (verified by `TreeGenerator.SeededRNGIsDeterministic` in `src/bracken/procedural/tree_generator.test.cpp`). The shadow proxy uses a separate avalanche hash also derived only from `seed`.
- **Rocks:** no RNG object at all — output is a pure function of `RockParams`. The integer seed is reduced by an avalanche hash masked to `[0, 2^20)` **before** entering the `sin`-based SDF hash, keeping every value exactly float-representable — distinct large seeds no longer collide. (Note: this hashing changed rock shapes for a given seed relative to older builds.)
- As with any float-heavy code, "deterministic" means within one compiler/platform/build; don't rely on cross-platform bit-equality.

### Performance guidance

- **Generate at load time, never per frame.** Tree generation is fast (~1–2 ms per preset tree, Debug build, `BM_TreeOak`/`BM_TreePine` in `src/bracken/procedural/procedural.bench.cpp`). Rock generation is *much* slower — seconds per rock at `resolution = 40` in a Debug build (`BM_RockBoulder`) because every one of the `(res+1)³` grid samples (plus per-vertex normal evaluation) runs the full `steps`-iteration SDF with FBM displacement.
- Reuse one generated model for many instances (vary yaw/scale per instance) instead of generating per placement — the instanced renderer makes N placements of one model nearly free.
- The tree presets deliberately use low `sections`/`segments` counts ("saves ~40% of branch triangles" per the source comments); raise them only if trees are seen close-up.
- The leaf mesh dominates a tree's triangle count (~60–70% per the source comments); `leaf_count`, `double_billboard`, and `children[]` are your main triangle knobs.

### Gotchas and common mistakes

1. **Mesh indices shift.** `meshes[1]` is leaves *only if* leaf geometry exists (`leaf_count > 0` and terminal branches were reached). The dead preset has `meshes.size() == 1`.
2. **Rock preset seed 0 ≠ "random":** it selects the preset's built-in default seed. Tree presets pass 0 through, where it aliases seed 1.
3. **Modify geometry before `register_model`** — GPU upload frees the CPU-side vertex/index/texture data.
4. **Hot reload does not apply** to registered procedural models (no source path).
5. `TreeGenerator::generate` never returns `nullptr`; `RockGenerator::generate` can — check it if you use custom params.
6. `twist[]` is radians while `angle[]`/`leaf_angle` are degrees.

---

## Render Graph (Available, Not Yet Wired)

Bracken ships a complete frame-graph style pass scheduler in
`src/bracken/render_graph/` (namespace `bracken::render_graph`). You declare
passes with the resources they read and write; the graph topologically sorts
them, allocates transient render targets from a pooled allocator (with
within-frame memory aliasing), optionally begins/ends the SDL GPU render pass
for you, and executes the passes in dependency order.

**Status — read this first:** the render graph is in-tree, unit-tested, and
correct, but it is **deliberately not wired into the engine's frame path**.
`scene::SceneRenderer` issues its passes directly, and the playground example
does not use the graph at all. Per `docs/engine_design_notes.md` ("The
'infrastructure without load' trap"), the graph's value only materializes when
the pass count grows past hand-tractable size, or for TAA history buffers,
transient aliasing, or multi-camera rendering — TAA + motion vectors is called
out as "the first concrete reason to wire up `render_graph/`". Until then,
don't route the renderer through it without a measured reason. It is,
however, fully usable standalone today, and this chapter shows how.

### Quick start: the per-frame lifecycle

The graph is an *immediate-mode* graph over a *retained* texture pool: you
re-declare all passes every frame; the underlying `SDL_GPUTexture`s persist
across frames inside the `TransientAllocator` pool.

```cpp
#include "bracken/render_graph/render_graph.hpp"

using namespace bracken::render_graph;

// Once, at startup (device is a bracken::gpu::GPUDevice&):
RenderGraph graph;
graph.init(device);   // void — binds the device to the TransientAllocator

// Every frame:
SDL_GPUCommandBuffer* cb = SDL_AcquireGPUCommandBuffer(device.handle());
graph.begin_frame(cb);          // resets passes/resources, keeps the pool

// ... graph.add_pass(...) calls, see below ...

if (graph.compile()) {          // topo-sort + transient allocation
    graph.execute();            // runs passes in compiled order
}
graph.end_frame();              // returns transient textures to the pool
```

Key facts about the lifecycle:

- `begin_frame()` calls `reset()` internally — all passes and resource
  declarations from the previous frame are gone. Only the texture pool
  survives.
- `compile()` returns `false` if the pass dependencies form a **cycle** or a
  transient texture cannot be allocated; the error is logged
  (`BRACKEN_LOG_ERROR` under tag `"rg"`) and a subsequent `execute()` is a
  **silent no-op**. Always check the `[[nodiscard]]` result.
- `end_frame()` releases every transient texture back to the pool and clears
  the command buffer pointer. Call it even if `compile()` failed.
- `shutdown()` (also run by the destructor) destroys all pooled textures.
- The graph has **no internal synchronization** — construct, build, compile,
  and execute it on one thread (the thread that owns the command buffer).

### Declaring passes

#### Untyped `add_pass` — capture locals

The simplest form takes a name, a setup lambda that declares
resources via `PassBuilder`, and an execute lambda that records GPU work
(signature `void(RenderPassContext&)`). This is the form the unit tests in
`src/bracken/render_graph/render_graph.test.cpp` use throughout:

```cpp
TextureDesc hdr;
hdr.width = 1920;
hdr.height = 1080;
hdr.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
hdr.usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;   // added automatically anyway

graph.add_pass(
    "scene",
    [&](PassBuilder& b) {
        b.create_color_target("scene_hdr", hdr);
    },
    [&](RenderPassContext& ctx) {
        SDL_GPURenderPass* rp = ctx.render_pass();  // begun by the graph
        // bind pipelines / draw into rp ...
    });

graph.add_pass(
    "post",
    [&](PassBuilder& b) {
        b.read("scene_hdr");                        // creates the edge scene -> post
        b.create_color_target("post_out", hdr);
    },
    [&](RenderPassContext& ctx) {
        // sample scene_hdr, write post_out ...
    });
```

`add_pass` is uniformly **three arguments** — name, setup, execute; a pass's
character (graphics vs. copy/upload) is expressed through what its setup
declares (attachments enable the managed render pass), not through a type
enum.

Execution order is derived purely from the read/write declarations —
`"post"` runs after `"scene"` because it reads `"scene_hdr"`. Passes with no
dependency between them run in the order they were added (the scheduler is
Kahn's algorithm with a FIFO queue).

#### Typed `add_pass<Data>` — carry state from setup to execute

When setup needs to hand handles (or anything else) to execute, use the
template overload. The graph heap-allocates one `Data` per pass
(`std::make_shared<Data>`) and keeps it alive inside the execute closure:

```cpp
struct PassData {
    ResourceHandle color;
    int marker = 0;
};

graph.add_pass<PassData>(
    "Typed",
    [](PassBuilder& b, PassData& d) {
        d.color = b.create_color_target("typed_color", desc);
        d.marker = 42;
    },
    [&](RenderPassContext& ctx, const PassData& d) {
        SDL_GPUTexture* tex = ctx.get_texture(d.color);
        // ...
    });
```

**Gotcha:** do not call `graph.add_pass(...)` from inside a setup lambda. The
pass list is a `std::vector<PassNode>`; a nested add can reallocate it and
dangle the `PassNode` reference the outer `add_pass` is still writing to.

### The `PassBuilder` API

`PassBuilder` (`src/bracken/render_graph/pass_builder.hpp`) is only valid
inside the setup lambda. Everything it does mutates the pass node and the
graph's resource table.

#### Creating transient render targets

```cpp
// Color target: adds SDL_GPU_TEXTUREUSAGE_COLOR_TARGET to desc.usage if
// missing, declares the resource as a write, records a color attachment,
// and enables the managed render pass for this pass.
ResourceHandle create_color_target(const std::string& name, const TextureDesc& desc,
                                   SDL_GPULoadOp loadop = SDL_GPU_LOADOP_CLEAR,
                                   glm::vec4 clear_color = glm::vec4(0.0f));

// Depth target: same, with SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET.
ResourceHandle create_depth_target(const std::string& name, const TextureDesc& desc,
                                   SDL_GPULoadOp loadop = SDL_GPU_LOADOP_CLEAR,
                                   float clear_depth = 0.0f);
```

`TextureDesc` (`src/bracken/render_graph/render_graph_resource.hpp`) fields
and defaults: `width = 0`, `height = 0`,
`format = SDL_GPU_TEXTUREFORMAT_INVALID`, `usage = 0`, `mip_levels = 1`,
`array_layers = 1`, `sample_count = 1`. You must set width/height/format
yourself — there are no validity checks.

**Reversed-Z gotcha:** `create_depth_target` / `attach_depth` default
`clear_depth` to **0.0f**, matching Bracken's reversed-Z scene convention
(far plane = depth 0). If you are building a standard-Z pass (e.g. a
shadow-style pass), pass `1.0f` explicitly — the default will silently give
you an always-failing depth test otherwise.

**Name-collision gotcha:** transient names are graph-global for the frame.
Declaring a transient with a name that already exists returns the *existing*
handle and **silently ignores your new `TextureDesc`**.

#### Reading, writing, and importing

```cpp
b.read("scene_hdr");            // by name; returns the handle (invalid if unknown)
b.write("gbuffer_normals");     // by name
b.read(handle);                 // by handle
b.write(handle);                // by handle
ResourceHandle h = b.lookup("scene_hdr");  // find without declaring anything

// Bring an externally owned texture (e.g. the CSM shadow atlas or the
// swapchain image) into the graph. The graph never owns or frees it.
ResourceHandle atlas = b.import_texture("shadow_atlas", sdl_texture,
                                        4096, 4096, SDL_GPU_TEXTUREFORMAT_D32_FLOAT);
```

**Silent no-op gotcha:** `read(name)` / `write(name)` with a name nobody has
declared return `ResourceHandle::invalid()` and record **nothing** — no log,
no error. If you typo a resource name, the dependency edge simply doesn't
exist and your pass may run too early. Check `h.valid()` when in doubt, or
declare by handle.

Duplicate `read`/`write` declarations of the same handle within one pass are
deduplicated. A pass may both read and write the same resource
(read-modify-write); this does not create a cycle.

Importing a name that already exists **overwrites** that resource's imported
pointer/dimensions and forces it to non-transient — useful for re-pointing a
name at this frame's swapchain image, but it can also silently convert a
transient you declared earlier.

#### Attaching existing resources as render targets

When the target was created by an earlier pass (or imported) and you want
this pass to render into it:

```cpp
b.attach_color(handle, SDL_GPU_LOADOP_LOAD);              // default load op: LOAD
b.attach_depth(handle, SDL_GPU_LOADOP_CLEAR, 0.0f);       // default clear: 0.0 (reversed-Z)
```

**Gotcha:** `attach_color` defaults to `SDL_GPU_LOADOP_LOAD` (preserve
contents), while `create_color_target` defaults to `SDL_GPU_LOADOP_CLEAR`.
Store ops are always `SDL_GPU_STOREOP_STORE`, and stencil load/store are
always `DONT_CARE` — none of these are configurable.

Each pass has a list of color attachments (calling `attach_color` /
`create_color_target` repeatedly appends) but only **one** depth attachment —
a second `attach_depth`/`create_depth_target` replaces the first.

#### Managed vs. unmanaged render passes

Any of `create_color_target`, `create_depth_target`, `attach_color`,
`attach_depth` sets `managed_render_pass = true`: at execute time the graph
calls `SDL_BeginGPURenderPass` with your attachments before your execute
lambda and `SDL_EndGPURenderPass` after it. Inside the lambda,
`ctx.render_pass()` is the live `SDL_GPURenderPass*`.

Call `b.set_managed_render_pass(false)` to opt out (you begin/end passes
yourself, or the pass is a copy/compute/upload pass) — then
`ctx.render_pass()` is `nullptr` and you work with `ctx.command_buffer()`
directly.

The managed pass is only begun when the frame actually has a command buffer
(`begin_frame(nullptr)` skips it — see "Headless testing") and the pass has
at least one attachment.

### Inside execute: `RenderPassContext`

`RenderPassContext` (`src/bracken/render_graph/render_graph_context.hpp`)
gives the execute lambda:

```cpp
SDL_GPUCommandBuffer* command_buffer() const noexcept;  // the frame's cb
SDL_GPURenderPass*    render_pass()    const noexcept;  // non-null only for managed graphics passes
SDL_GPUTexture*       get_texture(ResourceHandle h) const noexcept;
```

`get_texture` resolves a transient handle to the pooled `SDL_GPUTexture`
chosen at `compile()`, or an imported handle to the pointer you imported. It
returns `nullptr` for invalid handles.

The graph is **textures-only**: resources are transient or imported
textures; there is no buffer-resource API.

### Transient pooling, aliasing, and resize

`compile()` computes each transient's lifetime (first writing pass → last
reading pass in the compiled order) and walks the schedule acquiring and
releasing textures from the `TransientAllocator`
(`src/bracken/render_graph/transient_allocator.hpp`). Two transients with
**identical** `TextureDesc`s and non-overlapping lifetimes share one
underlying `SDL_GPUTexture` in the same frame — this is the aliasing the test
`TransientReuseSharesUnderlyingTexture` verifies. Pool matching is exact
`TextureDesc` equality (all seven fields), so a 1920×1080 target never
aliases a 1921×1080 one.

Pooled textures persist across frames until you evict them:

```cpp
// On swapchain / resolution resize — otherwise old-sized textures leak
// in the pool forever (they will never match the new descs):
graph.evict_pool();
```

Telemetry after `compile()`:

```cpp
graph.last_compiled_passes();   // passes in the compiled order
graph.last_transient_count();   // transient textures acquired this compile
graph.last_reuse_count();       // how many of those came from the pool
graph.pass_count();             // declared passes (pre/post compile)
graph.resource_count();         // declared resources
```

**Failure behavior:** if `SDL_CreateGPUTexture` fails while allocating a
transient, the allocator logs an error and `compile()` **fails** (returns
`false`, releasing whatever transients were already acquired) — a managed
render pass can never be handed a null attachment texture.

### Debugging: dump the graph as Graphviz DOT

```cpp
std::ostringstream os;
graph.dump_dot(os);
// os.str() -> "digraph RenderGraph { ... }"
```

Passes render as boxes, resources as ellipses labeled `(T)` transient or
`(I)` imported, with write edges `pass -> res` and read edges `res -> pass`.
Names are DOT-escaped, so quotes/backslashes in pass or resource names can't
corrupt the output. After a successful `compile()`, each pass box is
additionally annotated with its compiled execution order (`[step N]`); before
compilation only the declared graph is dumped. Pipe it through `dot -Tsvg`
to inspect the dependency structure.

### Headless testing (no GPU needed)

The whole graph runs without a device or command buffer — this is how its
unit tests work and how you can test your own pass wiring:

- Skip `init()`: the `TransientAllocator` then hands out stable fake sentinel
  pointers instead of real textures, so aliasing/reuse logic is still
  observable.
- `begin_frame(nullptr)`: execute callbacks still run in compiled order, but
  the graph never calls `SDL_BeginGPURenderPass` (so `ctx.render_pass()` and
  `ctx.command_buffer()` are `nullptr`).

```cpp
RenderGraph g;                 // no init()
g.begin_frame(nullptr);
g.add_pass("A",
    [](PassBuilder& b) { b.create_color_target("X", desc); },
    [&](RenderPassContext&) { order.emplace_back("A"); });
ASSERT_TRUE(g.compile());
g.execute();
g.end_frame();
```

Run the module's tests with:

```bash
cmake --build build
ctest --test-dir build --output-on-failure -R RenderGraph
# or: build/src/bracken/bracken_tests --gtest_filter='RenderGraph.*'
```

### Lifetime and threading contract (summary)

- `RenderGraph` and `TransientAllocator` are non-copyable and have no locks;
  single-threaded use on the command-buffer-owning thread only.
- The `gpu::GPUDevice` passed to `init()` must outlive the graph (raw pointer
  held; needed at `shutdown()`/`evict_pool()` to release pooled textures).
- Imported textures are never owned — you must keep them alive through
  `execute()` and release them yourself.
- Execute lambdas run inside `execute()`; anything they capture by reference
  must be alive at that point (they are stored as `std::function`s, and the
  typed overload's `Data` is kept alive by a `shared_ptr` in the closure).
- `ResourceHandle`s are indices valid only for the current frame's
  declarations — `reset()`/`begin_frame()` invalidates them all.

### When (and why not) to wire it into the frame

The documented rationale (`docs/engine_design_notes.md`, "The 'infrastructure
without load' trap"; `docs/architecture.md`, "Known stubs & silent
behaviors") is:

- **Why not now:** infrastructure has constant per-frame cost; the graph only
  pays off once the pass count grows past hand-tractable size.
  `SceneRenderer` issues its fixed pass sequence directly and that is
  currently simpler and cheaper. "Keep it in tree (the tests are valuable);
  don't route the renderer through it without a reason."
- **What would justify wiring it:** pass count outgrowing hand-scheduling,
  TAA history buffers ("the first concrete reason to wire up
  `render_graph/`"), broader transient aliasing wins, or multi-camera
  rendering.

Note: the graph does not push GPU debug groups around passes (the engine's
`gpu::GPUDebugGroup` RAII helper exists in `src/bracken/gpu/gpu_debug.hpp`
but is not used here).

---

## DevUI Developer Overlay

Bracken ships an immediate-mode developer overlay — `bracken::dev::DevUI`
(`src/bracken/dev/dev_ui.hpp`) — built on Dear ImGui with the SDL3 platform
backend and the SDLGPU3 renderer backend. It gives you a tabbed "Bracken Dev"
window with live render stats, environment/fog editing, graphics-settings
editing, and camera debugging, plus an API for registering your own game
panels. In the playground example, **F10** toggles it.

The overlay is designed so host code never needs `#ifdef`s: when it is
compiled out, every `DevUI` method is a safe no-op and `init()` simply returns
`false`. Only panel *bodies* — code that calls `ImGui::` functions directly —
need a guard (see "Registering custom panels" below).

### Enabling and disabling the overlay (`BRACKEN_ENABLE_DEV_UI`)

The CMake option `BRACKEN_ENABLE_DEV_UI` (root `CMakeLists.txt`, default
**ON**) controls whether the overlay is real or a stub:

```bash
# Default: overlay compiled in, Dear ImGui v1.92.2 fetched at configure time
cmake -S . -B build -G Ninja

# Shipping build: overlay compiles to a no-op stub, ImGui is neither
# fetched nor linked
cmake -S . -B build -G Ninja -DBRACKEN_ENABLE_DEV_UI=OFF
```

When ON (`src/bracken/CMakeLists.txt`):

- ImGui core + `imgui_impl_sdl3.cpp` + `imgui_impl_sdlgpu3.cpp` (and
  `imgui_demo.cpp`, so `ImGui::ShowDemoWindow()` works from your panels) are
  compiled into a companion static library `bracken_imgui`, linked `PUBLIC`
  into `bracken`.
- The preprocessor define `BRACKEN_DEV_UI=1` is set `PUBLIC` on the `bracken`
  target, so your game code can test `#ifdef BRACKEN_DEV_UI` and can
  `#include <imgui.h>` directly (the ImGui include dirs propagate).

When OFF, `src/bracken/dev/dev_ui.cpp` compiles its stub half: `init()`
returns `false`, `process_event()` returns `false`, `render()` does nothing,
`wants_keyboard()`/`wants_mouse()` return `false`. State-only methods
(`add_panel`, `set_visible`, `toggle`, `set_toggle_key`, `visible()`,
`initialized()`) are header-inline and still work, but nothing ever draws.

**Gotcha:** in a stub build the toggle key is inert — the stub
`process_event()` never looks at it. Don't rely on "press F10 and see
nothing" to conclude your build is broken; check the log line your host
prints after `init_dev_ui()` (the playground logs
`Dev UI initialized (F10 toggles it)` vs `unavailable (stub build or init failed)`).

### Quick start: `Application::init_dev_ui()` and the F10 toggle

If your game subclasses `bracken::Application`, everything is one call. From
the playground (`examples/playground/playground.cpp`):

```cpp
bool PlaygroundGame::on_init() {
    if (!init_renderer(1600, 900, "bracken playground", WORLD, WORLD)) {
        return false;
    }

    // Dear ImGui developer overlay (F10). In stub builds
    // (BRACKEN_ENABLE_DEV_UI=OFF) init_dev_ui() returns false and every DevUI
    // call is a harmless no-op, so nothing here needs guarding except the
    // panel body, which calls ImGui:: directly.
    const bool dev_ui_ok = init_dev_ui();
    BRACKEN_LOG_INFO("playground", "Dev UI {}",
                     dev_ui_ok ? "initialized (F10 toggles it)"
                               : "unavailable (stub build or init failed)");
    dev_ui().set_toggle_key(SDL_SCANCODE_F10);
    // ...
}
```

`Application::init_dev_ui()` (`src/bracken/application.cpp`) does all the
wiring for you:

1. Constructs the `DevUI` (always — even when the renderer is down or the
   overlay is compiled out, so `dev_ui()` stays safe to call).
2. Binds the built-in panels to the engine subsystems: the `SceneRenderer`,
   the `CameraController`, your persisted `GraphicsSettings` (edits are
   pushed through `request_graphics_settings` — the overlay draws during
   rendering, so the GPU/SDL apply work lands at the next frame fence), and
   the app's FPS counter.
3. Initializes the ImGui context and both backends against the render
   context's window/device/swapchain format.
4. **Claims the `SceneRenderer`'s single post-UI callback slot**
   (`scene::SceneRenderer::set_post_ui_callback`) to draw the overlay on top
   of the finished frame.

Call it **after** `init_renderer()`. It returns `false` when the overlay is
compiled out or backend init fails — but `dev_ui()` remains valid and safe to
use either way, so you don't need to branch on the result (log it, at most).

The **toggle key** deserves a note: it is *not* something you handle in your
own input code. `dev_ui().set_toggle_key(SDL_SCANCODE_F10)` registers it with
the overlay, and `DevUI::process_event` handles it internally — deliberately,
so the toggle works even while an ImGui widget (say, a focused `InputText`)
has keyboard capture and your `InputHandler` never sees the key-down. The
default is `SDL_SCANCODE_UNKNOWN`, i.e. no toggle key; the F10 binding is a
playground convention, not an engine default.

You can also drive visibility programmatically:

```cpp
dev_ui().set_visible(true);   // show
dev_ui().toggle();            // flip
bool open = dev_ui().visible();
```

The playground's smoke run does exactly this — at frame 170 it calls
`dev_ui().set_visible(true)` so the end-of-run screenshot proves the overlay
renders (harmless no-op in stub builds).

**Gotcha — one post-UI slot:** `SceneRenderer` has exactly one post-UI
callback slot. If you call `init_dev_ui()`, the overlay owns it; do not call
`scene_renderer().set_post_ui_callback(...)` yourself afterwards or you will
silently replace the overlay's draw (and vice versa). You can't have both.

### The built-in panels

While visible, the overlay draws one tabbed window titled **"Bracken Dev"**
(default size 420×540, first use only). Tabs whose data source wasn't bound
are hidden entirely — Stats needs `fps` or `scene_renderer`, Environment
needs `scene_renderer`, Graphics needs **both** `settings` and the
`apply_settings` callback, Camera needs `camera`. With
`Application::init_dev_ui()` all four are present.
Implementation: `src/bracken/dev/dev_panels.cpp` (the panel bodies are a
testable seam: `draw_builtin_panels(const DevUI::Bindings&, DevUI::PanelState&)` —
per-instance `PanelState` holds the FPS plot history and shake sliders, so
two `DevUI` instances never share widget state).

#### Stats

- FPS with a 120-sample scrolling history plot and derived frame time (ms).
- Render counters from `RenderStats` (`src/bracken/contracts/render_stats.hpp`): draw
  calls, triangles, entities rendered, entities distance/frustum-culled.
- Particle/effect counts from the rendered packet's effect-instance snapshot
  (`SceneRenderer::frame_particle_count()`/`frame_effect_count()`).

**Behavior to know:** drawing this tab calls
`SceneRenderer::set_collect_stats(true)` every frame so the readouts are
live, and it deliberately **never turns collection back off** — hosts (e.g.
the playground's smoke driver, which enables collection itself) would
otherwise have their setting stomped when the overlay closes. The cost is a
handful of CPU-side increments per frame.

The tab also shows value/budget bars (bone palettes, packet arena, lights
binned vs submitted), a "Dropped this frame" section that appears only when
`FrameDiagnostics::any()` is true, and the frame-phase breakdown
(simulate/render/fence/blocked-on-GPU).

**Gotcha:** the per-pass table's "CPU (record)" column is real; the "GPU"
column reads `n/a` because SDL3 exposes no timestamp queries, and the tab says
so in disabled text. Use RenderDoc/Nsight for GPU numbers.

#### Environment

Live-edits `render::EnvironmentState` **in place** via
`SceneRenderer::environment()` — the intended path, since `render_frame`
re-propagates the environment to every sub-renderer each frame. Sections:

- **Sun** — direction edited as azimuth/elevation of the direction *to* the
  sun (the engine stores `sun_dir` as the direction light travels). The
  angles are re-decomposed from `env.sun_dir` every frame, so an external
  day/night cycle in your game keeps the sliders in sync. Plus sun color.
- **Sky & ambient** — sky zenith/horizon, ground color, ambient sky/ground.
- **Clouds & haze** — cloud coverage/density/scale/speed and haze strength.
  Coverage/density/scale/haze edits kick off the amortized IBL refresh;
  speed only scales the scroll phase (never refreshes).
- **Distance fog** — horizon/sun-tint colors, start/end, height falloff.
- **Volumetric fog / god rays** — sun color, scatter, density, height,
  falloff, march start/end.
- **Color grading** — white-balance temperature/tint and the ASC-CDL
  slope/offset/power triplets (`grade_*` fields; all-neutral defaults are
  pixel-identical to no grading).

All color pickers use HDR float mode (`ImGuiColorEditFlags_Float |
ImGuiColorEditFlags_HDR`) because `EnvironmentState` colors are linear HDR —
values above 1.0 are legal and expected; the composite pass tonemaps.

**Gotcha:** the volumetric parameters only have a visible effect when
*Volumetric fog* is enabled on the Graphics tab — the panel reminds you of
this in disabled text.

**Gotcha:** environment edits are *not* persisted. They live in the
`SceneRenderer`'s `EnvironmentState` and are lost on exit (unlike Graphics
tab edits — see below).

#### Graphics

Edits your `GraphicsSettings` struct (`src/bracken/contracts/graphics_settings.hpp`)
directly, then — only on frames where a widget actually changed something —
pushes the whole struct through the host's apply path
(`Application::request_graphics_settings`, applied with internal diffing at
the next frame fence — panel bodies run mid-render, so the GPU/SDL work
must not happen inline). Exposed:
shadow mode/cascades/resolution, the local-shadow block (master toggle,
shadowed point/spot budgets, atlas resolution), AO mode
(Off/GTAO Half/GTAO Full), anti-aliasing (Off/FXAA/TAA), tonemap operator,
anisotropic filtering, vsync, window mode;
feature checkboxes (image-based lighting, bloom, volumetric fog, god rays,
distance fog, grass, skybox, trees, rocks, frustum culling, FPS/debug HUD);
draw distance; the auto-exposure block (toggle, adapt speeds, and a live
adapted-EV readout); and the tone/grading and bloom/AO/fog sliders (exposure
bias, contrast, saturation, ambient strength, sun intensity, bloom
strength/threshold, AO strength/radius, fog density).

**Gotcha — edits persist:** because the panel edits the `Application`'s own
`graphics_settings_` and `shutdown_renderer()` auto-saves them to
`settings_file_path()` (the SDL pref path, `settings.cfg`), anything you
change on this tab survives across runs. Handy, but surprising if you forget
you flipped something.

#### Camera

Read-only pose readout (position, forward, yaw/pitch derived from the forward
vector, orbit distance, FOV) plus interactive testers for the
`systems::CameraController` primitives:

- **Shake test** — intensity/duration sliders and buttons firing
  `add_shake(ShakeType::Impact / Heavy / Subtle, ...)` and
  `add_directional_shake(forward, ...)`.
- **Framing** — an FOV-bias slider (−30…+30 degrees, `set_fov_bias`) with a
  Reset button, and a focus-strength slider (`set_focus_strength`).

Pipelining note: this tab reads (and its testers poke) the live
`CameraController` from the main thread while the next frame's sim may be
updating it — a documented tear-only exception (float reads/writes for
display and debug feel; never consumed by logic).

### Registering custom panels from your game

`DevUI::add_panel(std::string name, std::function<void()> draw)` registers a
host panel, drawn as its own collapsible window whenever the overlay is
visible. The `draw` callback runs inside the ImGui frame between
`Begin(name)`/`End()` — write **body-only** `ImGui::` calls (no
`Begin`/`End` of your own for the panel window itself; child windows are
fine). It is safe to call `add_panel` before `init()` — panels are just
stored until the overlay actually draws.

From the playground:

```cpp
#ifdef BRACKEN_DEV_UI
    dev_ui().add_panel("Playground", [this] {
        ImGui::Text("Orbs collected: %d", orbs_collected_);
        ImGui::Text("Player pos: %.1f, %.1f, %.1f", player_pos_.x, player_pos_.y, player_pos_.z);
        ImGui::Text("Anim state: %s", anim_sm_.current_state().c_str());
        if (ImGui::Button("Spawn pickup burst")) {
            pending_pickup_bursts_.push_back(player_pos_);
        }
        ImGui::Checkbox("Flashlight", &flashlight_);
        ImGui::Checkbox("Debug draw", &debug_draw_);
    });
#endif
```

Notes:

- The `#ifdef BRACKEN_DEV_UI` is needed **only because the body includes and
  calls `imgui.h` symbols**, which don't exist in stub builds. The
  `add_panel` call itself would compile and no-op fine either way.
- **Panel bodies run mid-render, while the next `on_simulate` runs on the
  pool.** Read only values snapshotted in `on_main_frame` (or render-side
  data); a panel that reads live sim state (as the abridged example above
  appears to) must actually be reading a fence-time snapshot of it. Widgets
  that *write* host flags the sim reads (the checkboxes) are tear-only for
  `bool`s but should also prefer fence-snapshotted mirrors.
- When the user collapses your window, the body is skipped (the overlay
  honors `ImGui::Begin`'s return value), so don't put must-run-per-frame
  logic in a panel body.
- Callbacks run on the main thread, during command-buffer recording. Capture
  `this` freely if your game object outlives the `DevUI` (it does under
  `Application`, which destroys the overlay in `shutdown_renderer()`).

### Input capture and routing

When the overlay is open you want ImGui to own the mouse and keyboard, and
your game to *not* react — without wedging any input state when it closes.
Under `Application` this all works automatically; the rules are worth knowing
regardless (`src/bracken/dev/dev_ui.cpp`, `src/bracken/application.cpp`):

1. **Events reach the overlay first.** `Application::run`'s poll loop feeds
   every SDL event to `dev_ui_->process_event(event)` before `on_event()` and
   before the `InputHandler`. If the overlay consumes the event, nothing else
   sees it. (`SDL_EVENT_QUIT` is handled even before the overlay.)
2. **The backend is always fed**, even while hidden, so ImGui's
   mouse/keyboard/window state is coherent the moment the overlay opens.
   Hidden-state events are *not consumed* though — visibility alone never
   steals input.
3. **Consumption rules while visible:** mouse motion / button-down / wheel
   events are consumed iff ImGui `WantCaptureMouse`; key-down / text-input
   events iff `WantCaptureKeyboard`.
4. **Release events (`KEY_UP`, `MOUSE_BUTTON_UP`) are never consumed.**
   `WantCapture*` lags one frame behind ImGui's click ownership; a swallowed
   release would wedge `InputHandler` edge state (classic symptom:
   right-mouse relative mode stuck on). Your game input code must therefore
   tolerate seeing a release for a press it never saw.
5. **Polled input needs muting separately.** Event consumption can't stop
   `SDL_GetKeyboardState` polling, so `Application::run` calls
   `input_.set_keyboard_captured(dev_ui_ && dev_ui_->wants_keyboard())` every
   frame — while the overlay owns the keyboard, `InputHandler`'s polled
   movement keys (WASD/arrows) read as released (and therefore so do the
   `InputSnapshot` fields your sim reads); controller input is unaffected.
   If you poll SDL state yourself (main-thread only — from `on_event` or
   `on_main_frame`, never from `on_simulate`), do the same:

```cpp
if (!dev_ui().wants_keyboard()) {
    // read SDL_GetKeyboardState-driven game keys
}
if (!dev_ui().wants_mouse()) {
    // custom mouse picking, etc.
}
```

`wants_keyboard()`/`wants_mouse()` return `false` whenever the overlay is
hidden or uninitialized, so they are always safe to call unconditionally.

### Using DevUI without `Application` (manual wiring)

If you own the frame loop yourself, the pieces are:

```cpp
#include "bracken/dev/dev_ui.hpp"

bracken::dev::DevUI dev_ui;
dev_ui.set_toggle_key(SDL_SCANCODE_F10);

// After the renderer is up:
bracken::dev::DevUI::Bindings bindings{
    &scene_renderer,             // environment, render stats, effects
    &camera_controller,          // readouts + shake/fov primitives
    &graphics_settings,          // edited in place...
    [&](const bracken::GraphicsSettings& s) { apply_settings(s); }, // ...then pushed here
    &fps,                        // your FPS counter
};
if (!dev_ui.init(render_context, std::move(bindings))) {
    // stub build or backend init failed — everything below is still safe
}

// Draw hook: DevUI::render must run from the SceneRenderer post-UI callback
// (command-buffer recording, outside any render pass).
scene_renderer.set_post_ui_callback(
    [&](SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchain) {
        dev_ui.render(cmd, swapchain);
    });
```

Contracts you must honor:

- **Bindings are non-owning and must outlive the `DevUI`.** Null members are
  fine — they simply hide the corresponding built-in tab/section.
- **`init()` after the renderer is up** (it needs the window, device, and
  swapchain format from `render::RenderContext`). Calling `init()` on an
  already-initialized overlay **warns and returns `false`** — re-init is an
  error, not a rebind (the ImGui context is tied to the original
  window/device); `shutdown()` first to re-point.
- **Feed every SDL event** through `process_event` *before* your game input,
  and honor its `true` return by skipping the event.
- **`render()` must be called every frame** from the post-UI callback — even
  while hidden it does real work (it drains ImGui's input queue so hours of
  hidden-overlay input can't replay as phantom clicks when it opens).
  `SceneRenderer` calls the post-UI callback each frame the swapchain is
  available, so this comes for free once wired.
- **Destroy/`shutdown()` the overlay before the GPU device.** `shutdown()`
  calls `SDL_WaitForGPUIdle` first (the last frame's draw data may still be
  in flight), then tears down the backends and ImGui context. It is
  idempotent and the destructor calls it too.
- **Main thread only**, like the rest of the frame path.

### Testing DevUI code headlessly

`src/bracken/dev/dev_ui.test.cpp` and `src/bracken/dev/dev_panels.test.cpp`
show the approach: all DevUI unit tests are **headless** — no `init()`, no
GPU, no window, nothing that touches the ImGui backend. The six tests in
`dev_ui.test.cpp` pin the no-op-safety contract identically in both build
configurations (`BRACKEN_ENABLE_DEV_UI=ON` and the stub); the sixteen
`DevPanels.*` tests (ON-only) exercise the panel bodies through the
`draw_builtin_panels(bindings, state)` seam with a real ImGui context but no
GPU backend — pinning tab visibility (e.g. the Graphics tab hides without an
apply callback), per-instance `PanelState` (FPS history doesn't leak between
instances), and the "read from the owner, don't write back" rule that keeps
the Environment/Graphics/Camera tabs from stomping host state on every frame
the overlay is open. A tab other than the first only draws its body when
selected, so the harness forces the tab under test selected and asserts that
it was — otherwise a "changes nothing" test would pass by drawing nothing.

```cpp
#include <gtest/gtest.h>
#include "bracken/dev/dev_ui.hpp"

using bracken::dev::DevUI;

TEST(DevUI, AddPanelBeforeInitDoesNotCrashOrDraw) {
    DevUI ui;
    int draws = 0;
    ui.add_panel("Test", [&draws] { ++draws; });
    EXPECT_EQ(draws, 0);          // panels only stored; nothing draws without init()
    EXPECT_FALSE(ui.initialized());
}

TEST(DevUI, ProcessEventReturnsFalseWhenUninitialized) {
    DevUI ui;
    SDL_Event event{};
    event.type = SDL_EVENT_KEY_DOWN;
    EXPECT_FALSE(ui.process_event(event));
    ui.set_visible(true);         // visibility alone must not consume events
    EXPECT_FALSE(ui.process_event(event));
}
```

Follow the same rule for your own tests around DevUI-adjacent code: if a test
would need `init()`, it needs a GPU and a window — that's integration-test
territory (the playground `--smoke` run, which flips the overlay visible at
frame 170 and screenshots it), not a unit test. Tests live next to the unit
as `*.test.cpp` and are auto-globbed into `bracken_tests`.

### Common mistakes

- **Setting your own post-UI callback after `init_dev_ui()`** — silently
  replaces the overlay's draw hook. One slot; the overlay claims it.
- **Handling the toggle key in game input instead of `set_toggle_key`** —
  works until an ImGui text field has focus, then your binding is swallowed
  and the overlay can't be closed.
- **Calling `ImGui::Begin`/`End` inside an `add_panel` body** — the overlay
  already wraps your callback in `Begin(name)`/`End()`; body-only calls.
- **Forgetting `#ifdef BRACKEN_DEV_UI` around panel bodies** — stub builds
  don't link ImGui; the `DevUI` calls themselves need no guard, the
  `ImGui::` calls do.
- **Assuming stats collection turns off when the overlay closes** — the Stats
  tab turns `set_collect_stats(true)` on and leaves it on by design.
- **Expecting the F10 toggle in a `BRACKEN_ENABLE_DEV_UI=OFF` build** — the
  stub's `process_event` ignores the toggle key entirely.
- **Treating consumed presses and released keys asymmetrically** — you may
  receive a `KEY_UP`/`MOUSE_BUTTON_UP` whose matching down-event the overlay
  consumed; releases are always passed through on purpose.

---

## Physics (Jolt-backed)

`#include "bracken/physics/physics_world.hpp"` — rigid-body simulation with
a glm-only API (Jolt Physics runs underneath but never appears in the
header; you need no Jolt includes, defines, or link flags beyond linking
`bracken`).

### Setting up a world

```cpp
#include "bracken/physics/physics_world.hpp"
#include "bracken/core/jobs/job_system.hpp"

bracken::physics::PhysicsWorld physics;

bool MyGame::on_init() {
    bracken::physics::PhysicsConfig cfg;      // defaults shown:
    // cfg.gravity                = {0, -9.81f, 0};
    // cfg.max_bodies             = 4096;
    // cfg.max_body_pairs         = 4096;
    // cfg.max_contact_constraints = 2048;
    // cfg.fixed_dt               = 1.0f / 60.0f;
    // cfg.max_steps_per_update   = 4;
    if (!physics.init(cfg, bracken::core::jobs::JobSystem::instance()))
        return false;
    // ...
}
```

Pass the engine's always-on pool (`JobSystem::instance()`): Jolt's internal
step parallelism runs as jobs on *that* pool — the engine never spins up a
second set of threads for physics.

### Creating bodies

```cpp
using namespace bracken::physics;

// Static terrain from your heightmap (row-major world heights, square grid)
BodyId terrain = physics.add_heightfield({-half, 0, -half}, world_size,
                                         heights /* res*res floats */, res);

// A dynamic crate
BodyDesc crate;
crate.type        = BodyType::Dynamic;
crate.position    = {10.0f, 5.0f, -3.0f};
crate.mass        = 20.0f;
crate.friction    = 0.6f;
crate.restitution = 0.1f;
BodyId box = physics.add_box(crate, {0.5f, 0.5f, 0.5f});

// A kinematic character proxy (moved by you, pushes dynamic bodies)
BodyDesc kd;
kd.type     = BodyType::Kinematic;
kd.position = spawn_pos;
BodyId character = physics.add_capsule(kd, /*cylinder half-height*/ 0.6f,
                                       /*radius*/ 0.35f);
```

All `add_*` return `INVALID_BODY_ID` (and log) on failure. `BodyDesc::mass`
only applies to dynamic bodies; `start_active = false` spawns a body asleep.

### Stepping and reading back — from `on_simulate`

`PhysicsWorld` is tier **owned / sim-stage**: drive it from your sim code
(one owner; don't call into one world from two threads at once). `update()`
is a plain CPU job internally — safe from the sim stage by design:

```cpp
void MyGame::on_simulate(bracken::SimFrame& sim) {
    // Kinematic bodies: command a target for this slice, then step.
    physics.move_kinematic(character, want_pos, want_rot, sim.dt());
    physics.update(sim.dt());   // fixed-step accumulator inside

    // Read back and draw. transform() is translation*rotation — scale your
    // model matrix separately.
    sim.frame().draw_model(crate_model, physics.transform(box) * crate_scale);
}
```

`update(dt)` consumes time in `fixed_dt` slices (at most
`max_steps_per_update` per call — beyond that the debt is dropped with a
one-time warning, so a hitch slows the sim instead of spiraling). With the
defaults, a 60 Hz frame steps exactly once.

### Interacting

```cpp
physics.apply_impulse(box, {0, 200.0f, 0});             // N·s, wakes the body
physics.apply_impulse_at(box, imp, hit_point);           // torque-inducing
physics.set_linear_velocity(box, v);                     // wakes if nonzero
if (!physics.is_active(box)) { /* asleep — skip its VFX */ }

auto hit = physics.cast_ray(cam_pos, cam_dir * 100.0f);  // direction*length
if (hit.hit) { /* hit.position, hit.normal, hit.body, hit.fraction */ }
```

### Character collision (capsule overlap)

For a heightmap-grounded character that should slide along walls, trees, and
props without being a dynamic body, overlap a capsule against the static
world and push out of whatever it reports (deepest contact first):

```cpp
bracken::physics::PhysicsWorld::ShapeContact contacts[8];
// Static bodies only; exclude the terrain heightfield — height snapping
// owns the ground, and slope contacts would otherwise shove the capsule.
size_t n = physics.collide_capsule(pos + glm::vec3(0, capsule_center_y, 0),
                                   half_cylinder, radius, contacts,
                                   /*include_moving=*/false, terrain_body);
for (size_t i = 0; i < n; ++i) {
    glm::vec3 push = contacts[i].normal * contacts[i].penetration;
    push.y = 0.0f;                                   // walls only
    pos += push;
}
```

Re-query once or twice after pushing (escaping one body can enter a
neighbour). The playground's `update_player` is the reference
implementation.

### Gotchas

- **Determinism:** identical world setup + identical `update()` calls give
  bit-identical trajectories (any worker count, same binary/platform). Don't
  rely on it across machines or compiler upgrades.
- **Heightfield grids** should be even-sized (a Jolt block-size rule);
  odd-resolution grids are padded internally by duplicating the last
  row/column, which extends the surface slightly past `world_size` on +x/+z.
- **`remove_body` invalidates the id immediately**; queries on stale ids
  return identity values (position `{0,0,0}`), never UB.
- **Sleeping:** resting bodies deactivate automatically; impulses,
  `set_linear_velocity`, contacts from moving bodies, and
  `set_position(..., activate=true)` wake them.
- One `PhysicsWorld` per independent simulation; worlds share Jolt's global
  type registry (initialized once, thread-safe) but nothing else.
