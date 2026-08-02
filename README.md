# Bracken

**Bracken** is a standalone C++23 real-time rendering engine for big outdoor
worlds — terrain, instanced grass, procedural trees, volumetric atmosphere —
extracted from the `mmo-4` game project. Built on **SDL3 GPU**
(Vulkan/Metal/D3D12 backends) with HLSL→SPIR-V shader compilation, glTF model
loading, skeletal animation, and a clustered-forward renderer.

> *bracken (n.) — the coarse fern that carpets a forest floor.*

This repository contains *only* the engine — no game, server, or networking
code (rigid-body physics, however, ships in-engine: `bracken::physics`, a
glm-typed API over vendored Jolt Physics). It builds as a static library
(`bracken`, aliased `bracken::bracken`) plus its unit-test suite
(`bracken_tests`, GoogleTest).

## Features

- **GPU abstraction** over the SDL3 GPU API (`src/bracken/gpu/`) — buffers,
  textures, shaders, compute pipelines, a lazy `PipelineRegistry`,
  uniform/SSBO management.
- **Clustered-forward renderer** (`src/bracken/render/`) — linear-space PBR
  with an HDR pipeline: clustered point/spot lighting (up to 256 lights),
  cascaded shadow maps (PCF/PCSS), budgeted point/spot light shadows
  (cached-tile atlas, host opt-in per light — or, opt-in, one ray-traced
  reservoir shadow ray per pixel that gives EVERY local light a soft shadow
  with no budget at all, in either a visibility-ratio or a full demodulated-DI
  tier), horizon-based GTAO (temporally
  accumulated, ambient-only), image-based lighting prefiltered live from the
  procedural sky (diffuse irradiance + GGX specular cubemaps + split-sum
  BRDF LUT, amortized regeneration on sky changes), histogram auto exposure
  (compute-based, percentile-clipped, temporally adapted), bloom, volumetric
  fog with god rays, TAA (motion-vector based, the default) or FXAA, color
  grading (white balance + ASC-CDL) at composite, a procedural sky
  (domain-warped cloud deck with sun-aware scattering, live cloud/haze
  controls, golden-hour/dusk/night response, fog-sky horizon agreement),
  terrain, instanced grass, skinned meshes, and water — a surface renderer
  (screen-space refraction with Beer-Lambert absorption, screen-space
  reflections with a prefiltered-sky fallback, fresnel, sun glint, shoreline
  foam) driven by a compute-shader
  wave simulation (ping-pong wave equation with ripple impulses and
  downstream advection).
- **Skeletal animation** (`src/bracken/animation/`) — playback with crossfade,
  data-driven state machines, two-bone IK (foot placement, body lean).
- **Scene layer** (`src/bracken/scene/`) — cameras, frustum culling, the
  immutable per-frame `FramePacket` snapshot (described through a
  `FrameSink`, built by `FrameBuilder`, recycled through a `PacketRing`)
  consumed by `SceneRenderer`, incl. 2D UI commands. Games can unit-test
  their frame code headlessly with `bracken/testing/frame_test.hpp`.
- **Particle effects** (`src/bracken/effects/`) —
  data-driven mesh-particle emitters (burst/continuous, orbital/arc paths,
  lifetime curves).
- **Procedural geometry** (`src/bracken/procedural/`) — tree and rock
  generators.
- **Physics** (`src/bracken/physics/`) — rigid-body simulation backed by
  **Jolt Physics** behind a glm-only `PhysicsWorld` API (Jolt never leaks
  into public headers): fixed-timestep stepping, sphere/box/capsule bodies,
  static heightfield terrain, kinematic movers, impulses, ray casts —
  deterministic, with Jolt's internal parallelism running on the engine's
  own job-system pool (no second thread pool).
- **Core utilities** (`src/bracken/core/`) — logging, asserts, `Result`,
  handles, hashing/`StringId`, a job system, asset registry + file watcher,
  Tracy profiler hooks.
- **Memory** (`src/bracken/memory/`) — arena + frame allocators.
- **Render graph** (`src/bracken/render_graph/`) — pass scheduling with
  transient-resource pooling (in-tree and tested; not yet driving the frame).
- **Shaders** (`src/bracken/shaders/`) — HLSL, compiled to SPIR-V at build time
  via SDL_shadercross; hot-reloadable at runtime with `BRACKEN_HOT_RELOAD=1`.

## Dependencies

| Dependency | How it's obtained |
|---|---|
| **SDL3**, **SDL3_ttf**, **SDL3_image** | System install preferred; vendored from source automatically if not found (≥ 3.4.0 / 3.2.2 / 3.4.0). |
| **SDL_shadercross** | Fetched + built (build-time shader compiler CLI + runtime cross-compile library; bundles DXC and SPIRV-Tools). |
| **glm** | Fetched (header-only math). |
| **tinygltf** | Fetched (header-only glTF loader — no Draco support). |
| **Jolt Physics** | Fetched + built (v5.2.0, static; linked PRIVATE — Jolt types never appear in public bracken headers). |
| **GoogleTest** | Fetched when `BRACKEN_BUILD_TESTS=ON` (default). |
| **Dear ImGui** | Fetched when `BRACKEN_ENABLE_DEV_UI=ON` (default). |
| **Tracy** | Fetched only when `BRACKEN_ENABLE_TRACY=ON` (default OFF). |

All non-system dependencies are pulled by CMake `FetchContent` — no manual
setup beyond a C++23 compiler, CMake ≥ 3.20, and Python 3 (only needed to
build the vendored DXC inside SDL_shadercross on the first configure).
Installing system SDL3 dev packages avoids a from-source SDL build.

## Building

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

> The first configure fetches and builds SDL_shadercross (which bundles DXC
> and SPIRV-Tools) — this is the slow part of a cold build. Subsequent
> incremental builds are fast; never delete `build/` to "fix" something.

### Useful options

| Option | Default | Effect |
|---|---|---|
| `BRACKEN_BUILD_TESTS` | `ON` | Build `bracken_tests` (GoogleTest). |
| `BRACKEN_BUILD_BENCHMARKS` | `ON` | Build `bracken_benchmarks` (Google Benchmark). |
| `BRACKEN_BUILD_EXAMPLES` | `ON` | Build the example apps (`examples/playground`). |
| `BRACKEN_PLAYGROUND_SMOKE_TEST` | `OFF` | Register `playground --smoke` as a ctest (needs a GPU + display). |
| `BRACKEN_ENABLE_DEV_UI` | `ON` | Build the Dear ImGui developer overlay (`bracken::dev::DevUI`). `OFF` compiles it to a no-op stub and skips fetching ImGui. |
| `BRACKEN_ENABLE_TRACY` | `OFF` | Compile in Tracy profiler instrumentation. |

### Tests

```bash
ctest --test-dir build --output-on-failure
```

Tests live next to their unit as `*.test.cpp` and are auto-discovered.

### One build tree

There is exactly one tree, `build/`, and nothing ever deletes or
reconfigures it from scratch. Sanitizers, build types and option overrides are
`-D` flags you pass when you want them — not parallel trees that drift out of
sync with each other and with the docs describing them.

```bash
# a sanitizer run, into the same tree, when you actually need one
cmake -S . -B build -DCMAKE_C_FLAGS=-fsanitize=thread -DCMAKE_CXX_FLAGS=-fsanitize=thread \
                    -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread
```

### Benchmarks

```bash
./build/src/bracken/bracken_benchmarks                          # all
./build/src/bracken/bracken_benchmarks --benchmark_filter=Anim  # subset
```

`build/` has no `CMAKE_BUILD_TYPE` by default, so it compiles at `-O0` with
live thread-tier assertions — any timing measured that way says nothing about
the shipped engine. Reconfigure with `-DCMAKE_BUILD_TYPE=RelWithDebInfo`
before quoting a number (CMake warns if you forget). Micro-benchmarks live next to their unit as
`*.bench.cpp` (Google Benchmark, auto-discovered) and cover the CPU hot paths: animation sampling/crossfade,
IK, state machines, frustum culling, clustered light binning, particle
updates, allocators, hashing, and procedural mesh generation.

### Profiling (Tracy)

Configure with `-DBRACKEN_ENABLE_TRACY=ON`, run any engine app, and connect the
Tracy profiler GUI. The engine is instrumented end to end: frame marks,
per-pass CPU zones (colored by pass type), draw-call/triangle/particle plots,
allocator memory tracking, and named threads. GPU zones are not available —
SDL3 GPU does not expose backend queues; use RenderDoc/Nsight for GPU timing.

## Example: playground

[`examples/playground`](examples/playground) is a small medieval-village
orb-collector game that doubles as the engine's integration test. A simulated
river runs along the west edge of the map (wade in to push ripples around). The village
— a cobbled plaza ringed by six modular timber/stone buildings with glowing
windows, chimney smoke, lanes, fences, and props — is assembled offline from a
modular glTF asset pack into merged per-material `.glb`s (so the engine
uploads each texture set exactly once), and those converted `.glb`s are what
`examples/playground/assets/` carries. Only the character, orbs, and effect meshes are built in code —
the buildings and props come from the building pack, and the forest ring,
bushes, rocks, flowers, and grass tufts (which replace the engine's blade
grass in this scene) come from a stylized nature pack, converted 1:1 with
their own materials. The scene dressing is hand-placed, area by area, in
[`scene_layout.cpp`](examples/playground/scene_layout.cpp) — market corner,
farm wheat field, cottage garden, campsite, rock outcrops, flower drifts, and
a forest ring composed from designed grove stamps; only the grass-tuft ground
cover is a procedural fill. On a sparse checkout without those assets the
playground falls back to a procedural-only scene (generator trees/rocks +
engine blade grass). It deliberately touches every major subsystem: terrain
(splatmap-painted dirt lanes) + grass (exclusion zones under the paving) +
skybox, `.glb` loading with full PBR materials, skinned animation driven by a
state machine with foot IK and body lean, dynamic point/spot lights, particle
effects (incl. continuous chimney smoke), 3D billboards, debug lines, UI text,
and runtime graphics-settings toggles (AO / bloom / fog / shadows / AA mode /
IBL / local shadows on F1–F8).

```bash
./build/examples/playground/playground            # play it (WASD, mouse)
./build/examples/playground/playground --smoke    # self-drive 240 frames, print stats, exit
./build/examples/playground/playground --rt-bench 240  # same run as a timing jig, prints a table
```

`--smoke [N]` is the integration-test mode: the player auto-runs to orbs,
attacks periodically, and flips heavyweight settings mid-run, then prints a
`[smoke] OK ...` summary with frame/draw-call/triangle counts. It also
captures a screenshot (`playground_smoke.png`) near the end of the run and
verifies it isn't black or solid-color — a broken renderer that still hits
60 fps can't pass. In the interactive game, F12 saves a screenshot; the
underlying API is `Application::request_screenshot(path)` (final swapchain
image, UI included, saved as PNG).

`--rt-bench N` is the same self-driven run turned into a measurement jig:
pinned camera, no screenshot, none of the mid-run settings flips, and a
machine-readable `[rtbench] ...` table of per-pass CPU record times (mean and
p95 over the run) plus the ray-tracing stats. It is the gate for any traversal
performance claim. `PLAYGROUND_RT=shadows|reflections|both` picks which traced
effects run and `PLAYGROUND_GI=probes|probes_screen|off` pins the probe-field
GI mode (unset means off for both, because those settings persist); numbers are
only worth quoting from a `-DCMAKE_BUILD_TYPE=RelWithDebInfo` tree.

### Developer overlay

The engine ships an immediate-mode developer overlay (`bracken::dev::DevUI`,
Dear ImGui on the SDL3 GPU backend) with built-in panels for render stats,
environment/fog, graphics settings, and the camera. In the playground, **F10**
toggles it. Consumer usage is one call — `init_dev_ui()` after
`init_renderer()` — then `dev_ui().add_panel(name, draw)` for game panels;
note it claims the `SceneRenderer` post-UI callback slot for itself. With
`BRACKEN_ENABLE_DEV_UI=OFF` everything compiles to a no-op stub, so host code
needs no `#ifdef`s (only panel bodies, which include `imgui.h` directly, need
a `BRACKEN_DEV_UI` guard).

## Layout & include convention

The include root is `src/`, and engine headers are included with an
`bracken/`-rooted path:

```cpp
#include "bracken/render/world_renderer.hpp"
#include "bracken/gpu/gpu_device.hpp"
```

```
src/bracken/
├── CMakeLists.txt      # builds the bracken static lib + tests
├── animation/  core/  gpu/  memory/  procedural/
├── render/  render_graph/  scene/  systems/
└── shaders/            # HLSL sources + their CMake compile rules
```

## Consuming the engine

Subclass `bracken::Application` and override `on_init` / `on_simulate` /
`on_shutdown`, link against the `bracken::bracken` target, and call
`bracken_copy_shaders(<your_target>)` to stage the compiled SPIR-V next to
your executable:

```cmake
add_subdirectory(path/to/engine)   # or use it as a submodule
target_link_libraries(my_game PRIVATE bracken::bracken)
bracken_copy_shaders(my_game)
```

```cpp
#include "bracken/bracken.hpp"   // the supported surface, in one header

class MyGame final : public bracken::Application {
protected:
    bool on_init() override { /* load models, configure renderers */ return true; }
    void on_simulate(bracken::SimFrame& sim) override {
        // Runs as a pool job, overlapped with the previous frame's render.
        // Describe the frame; the engine draws it.
        sim.frame().draw_model(hero_, transform);
    }
};
```

There is no `on_update`/`on_render`: the host *describes* a frame through the
`FrameSink` that `SimFrame::frame()` returns, and the engine renders it.
`bracken.hpp` is an umbrella over the supported surface — the headers the
example app actually uses. It says what to reach for; it is **not** a stability
promise (the engine is pre-1.0 and breaks are compiler-caught). Headers under
`<module>/internal/` are engine-private and may include vendored dependencies:
that is where Jolt lives, and it is why `bracken::physics` stays Jolt-free.

For your own tests, `#include "bracken/testing/frame_test.hpp"` — build a
frame, seal it, and assert on exactly what your game described, with no GPU
and no window.

Conventions consumers inherit: **Y-up** coordinates, **reversed-Z zero-to-one
depth** (`GLM_FORCE_DEPTH_ZERO_TO_ONE` propagates from the engine target;
scene projections swap near/far so near→1, far→0 — see
[docs/architecture.md](docs/architecture.md)), linear-space lighting, `.glb`
models (no Draco).

## Documentation

| Doc | Contents |
|---|---|
| [docs/user_guide.md](docs/user_guide.md) | The definitive consumer guide: how to build, integrate, and use every engine feature, with code examples, defaults, and gotchas. |
| [docs/technical_spec.md](docs/technical_spec.md) | Detailed specification of every engine feature: data structures, algorithms, contracts, constants, and known limitations. |
| [docs/architecture.md](docs/architecture.md) | Runtime architecture: module map, `Application` lifecycle, frame/pass flow, GPU layer, threading & lifetime contracts, load-bearing constants, known stubs. |
| [docs/engine_design_notes.md](docs/engine_design_notes.md) | Design rationale and lessons — read before adding cross-cutting infrastructure. |
| [docs/skinned_mesh_format.md](docs/skinned_mesh_format.md) | The `.glb` authoring contract for skinned meshes (bones, channels, naming, checklist). |
| [docs/effects_system.md](docs/effects_system.md) | Data-driven particle effects: definitions, runtime, rendering. |
| [docs/audits/](docs/audits/) | Point-in-time engine audits (2026-07): performance, graphics features, C++23 modernization, shaders, and a full-source documentation sweep ([documentation_sweep.md](docs/audits/documentation_sweep.md), all findings addressed 2026-07-12). Findings are adversarially verified; [RESOLUTIONS.md](docs/audits/RESOLUTIONS.md) records the outcome of each finding (fixed / partial / deferred). |
| [src/bracken/shaders/README.md](src/bracken/shaders/README.md) | Shader naming, pipeline→shader map, SDL3 GPU binding order, CPU↔GPU uniform contract. |
| [CLAUDE.md](CLAUDE.md) | Orientation + house rules for AI coding agents working in this repo. |
