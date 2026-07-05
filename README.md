# mmo_engine

A standalone C++23 real-time rendering engine, extracted from the `mmo-4`
game project. Built on **SDL3 GPU** (Vulkan/Metal/D3D12 backends) with
HLSL→SPIR-V shader compilation, glTF model loading, skeletal animation, and a
clustered-forward renderer.

This repository contains *only* the engine — no game, server, or networking
code. It builds as a static library (`mmo_engine`) plus its unit-test suite
(`mmo_engine_tests`, GoogleTest).

## Features

- **GPU abstraction** over the SDL3 GPU API (`src/engine/gpu/`) — buffers,
  textures, shaders, a lazy `PipelineRegistry`, uniform/SSBO management.
- **Clustered-forward renderer** (`src/engine/render/`) — linear-space PBR
  with an HDR pipeline: clustered point/spot lighting (up to 256 lights),
  cascaded shadow maps (PCF/PCSS), SSAO/GTAO, bloom, volumetric fog with god
  rays, FXAA, procedural skybox, terrain, instanced grass, skinned meshes.
- **Skeletal animation** (`src/engine/animation/`) — playback with crossfade,
  data-driven state machines, two-bone IK (foot placement, body lean).
- **Scene layer** (`src/engine/scene/`) — cameras, frustum culling, a
  command-buffer `RenderScene` consumed by `SceneRenderer`, 2D `UIScene`.
- **Particle effects** (`src/engine/systems/`, `effect_definition.hpp`) —
  data-driven mesh-particle emitters (burst/continuous, orbital/arc paths,
  lifetime curves).
- **Procedural geometry** (`src/engine/procedural/`) — tree and rock
  generators.
- **Core utilities** (`src/engine/core/`) — logging, asserts, `Result`,
  handles, hashing/`StringId`, a job system, asset registry + file watcher,
  Tracy profiler hooks.
- **Memory** (`src/engine/memory/`) — arena + frame allocators.
- **Render graph** (`src/engine/render_graph/`) — pass scheduling with
  transient-resource pooling (in-tree and tested; not yet driving the frame).
- **Shaders** (`src/engine/shaders/`) — HLSL, compiled to SPIR-V at build time
  via SDL_shadercross; hot-reloadable at runtime with `MMO_HOT_RELOAD=1`.

## Dependencies

| Dependency | How it's obtained |
|---|---|
| **SDL3**, **SDL3_ttf**, **SDL3_image** | System install preferred; vendored from source automatically if not found (≥ 3.4.0 / 3.2.2 / 3.4.0). |
| **SDL_shadercross** | Fetched + built (build-time shader compiler CLI + runtime cross-compile library; bundles DXC and SPIRV-Tools). |
| **glm** | Fetched (header-only math). |
| **tinygltf** | Fetched (header-only glTF loader — no Draco support). |
| **GoogleTest** | Fetched when `MMO_BUILD_TESTS=ON` (default). |
| **Tracy** | Fetched only when `MMO_ENABLE_TRACY=ON` (default OFF). |

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
| `MMO_BUILD_TESTS` | `ON` | Build `mmo_engine_tests` (GoogleTest). |
| `MMO_BUILD_BENCHMARKS` | `ON` | Build `mmo_engine_benchmarks` (Google Benchmark). |
| `MMO_BUILD_EXAMPLES` | `ON` | Build the example apps (`examples/playground`). |
| `MMO_PLAYGROUND_SMOKE_TEST` | `OFF` | Register `playground --smoke` as a ctest (needs a GPU + display). |
| `MMO_ENABLE_TRACY` | `OFF` | Compile in Tracy profiler instrumentation. |

### Tests

```bash
ctest --test-dir build --output-on-failure
```

Tests live next to their unit as `*.test.cpp` and are auto-discovered.

### Benchmarks

```bash
./build/src/engine/mmo_engine_benchmarks                          # all
./build/src/engine/mmo_engine_benchmarks --benchmark_filter=Anim  # subset
```

Micro-benchmarks live next to their unit as `*.bench.cpp` (Google Benchmark,
auto-discovered) and cover the CPU hot paths: animation sampling/crossfade,
IK, state machines, frustum culling, clustered light binning, particle
updates, allocators, hashing, and procedural mesh generation.

### Profiling (Tracy)

Configure with `-DMMO_ENABLE_TRACY=ON`, run any engine app, and connect the
Tracy profiler GUI. The engine is instrumented end to end: frame marks,
per-pass CPU zones (colored by pass type), draw-call/triangle/particle plots,
allocator memory tracking, and named threads. GPU zones are not available —
SDL3 GPU does not expose backend queues; use RenderDoc/Nsight for GPU timing.

## Example: playground

[`examples/playground`](examples/playground) is a small orb-collector game
that doubles as the engine's integration test. All assets are built in code
(procedural terrain, trees, rocks, a skinned biped with authored Idle / Walk /
Run / Attack clips) — the repo ships no binary assets. It deliberately touches
every major subsystem: terrain + grass + skybox, skinned animation driven by a
state machine with foot IK and body lean, dynamic point/spot lights, particle
effects, 3D billboards, debug lines, UI text, and runtime graphics-settings
toggles (AO / bloom / fog / shadows / FXAA on F1–F6).

```bash
./build/examples/playground/playground            # play it (WASD, mouse)
./build/examples/playground/playground --smoke    # self-drive 240 frames, print stats, exit
```

`--smoke [N]` is the integration-test mode: the player auto-runs to orbs,
attacks periodically, and flips heavyweight settings mid-run, then prints a
`[smoke] OK ...` summary with frame/draw-call/triangle counts. It also
captures a screenshot (`playground_smoke.png`) near the end of the run and
verifies it isn't black or solid-color — a broken renderer that still hits
60 fps can't pass. In the interactive game, F12 saves a screenshot; the
underlying API is `Application::request_screenshot(path)` (final swapchain
image, UI included, saved as PNG).

## Layout & include convention

The include root is `src/`, and engine headers are included with an
`engine/`-rooted path:

```cpp
#include "engine/render/world_renderer.hpp"
#include "engine/gpu/gpu_device.hpp"
```

```
src/engine/
├── CMakeLists.txt      # builds the mmo_engine static lib + tests
├── animation/  core/  gpu/  memory/  procedural/
├── render/  render_graph/  scene/  systems/
└── shaders/            # HLSL sources + their CMake compile rules
```

## Consuming the engine

Subclass `mmo::engine::Application` (override `on_init` / `on_update` /
`on_render`), link against the `mmo_engine` target, and call
`mmo_engine_copy_shaders(<your_target>)` to stage the compiled SPIR-V next to
your executable:

```cmake
add_subdirectory(path/to/engine)   # or use it as a submodule
target_link_libraries(my_game PRIVATE mmo_engine)
mmo_engine_copy_shaders(my_game)
```

Conventions consumers inherit: **Y-up** coordinates, **reversed-Z zero-to-one
depth** (`GLM_FORCE_DEPTH_ZERO_TO_ONE` propagates from the engine target;
scene projections swap near/far so near→1, far→0 — see
[docs/architecture.md](docs/architecture.md)), linear-space lighting, `.glb`
models (no Draco).

## Documentation

| Doc | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Runtime architecture: module map, `Application` lifecycle, frame/pass flow, GPU layer, threading & lifetime contracts, load-bearing constants, known stubs. |
| [docs/engine_design_notes.md](docs/engine_design_notes.md) | Design rationale and lessons — read before adding cross-cutting infrastructure. |
| [docs/skinned_mesh_format.md](docs/skinned_mesh_format.md) | The `.glb` authoring contract for skinned meshes (bones, channels, naming, checklist). |
| [docs/effects_system.md](docs/effects_system.md) | Data-driven particle effects: definitions, runtime, rendering. |
| [docs/audits/](docs/audits/) | Point-in-time engine audits (2026-07): performance, graphics features, C++23 modernization, shaders. Findings are adversarially verified; [RESOLUTIONS.md](docs/audits/RESOLUTIONS.md) records the outcome of each finding (fixed / partial / deferred). |
| [src/engine/shaders/README.md](src/engine/shaders/README.md) | Shader naming, pipeline→shader map, SDL3 GPU binding order, CPU↔GPU uniform contract. |
| [CLAUDE.md](CLAUDE.md) | Orientation + house rules for AI coding agents working in this repo. |
