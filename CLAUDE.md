# Project: mmo_engine

A standalone **C++23** real-time rendering engine, extracted from the `mmo-4`
game project. Built on **SDL3 GPU** (Vulkan/Metal/D3D12 backends) with
HLSL→SPIR-V shader compilation, glTF model loading, skeletal animation, and a
clustered-forward renderer.

This repo contains **only the engine** — no game, server, networking, ECS, or
physics. It builds as a static library (`mmo_engine`) plus a GoogleTest suite
(`mmo_engine_tests`), and one example app: `examples/playground`, an
orb-collector demo that doubles as the integration test (all assets built in
code; `--smoke N` self-drives for N frames and exits — run it after changes
that could affect rendering). See [README.md](README.md) for the user-facing
overview.

## House rules (general)

@.claude/RULES.md

In short: **no full clean rebuilds.** Always `cmake --build build` (or
`ninja -C build`). The first configure fetches and builds SDL_shadercross
(bundling DXC + SPIRV-Tools) — that's the slow part of a cold build; never
throw it away with a clean rebuild. New `.cpp`/`.hpp`/`.test.cpp` files under
`src/engine/` are picked up automatically by the CMake `GLOB_RECURSE`, so an
incremental build is all that's needed after adding a file.

---

## Documentation map

Read the doc that matches the task before diving into code:

- [docs/architecture.md](docs/architecture.md) — **the deep reference**:
  module map, `Application` lifecycle, exact frame/pass order, GPU layer,
  threading & lifetime contracts, constants, known stubs/silent behaviors.
- [docs/engine_design_notes.md](docs/engine_design_notes.md) — rationale +
  decision principles. **Required reading before adding any cross-cutting
  infrastructure** (jobs, allocators, watchers, render-graph wiring).
- [docs/skinned_mesh_format.md](docs/skinned_mesh_format.md) — the `.glb`
  authoring contract (bones, channels, clip naming, IK bone names).
- [docs/effects_system.md](docs/effects_system.md) — particle effects:
  definitions / runtime / rendering split, host responsibilities.
- [src/engine/shaders/README.md](src/engine/shaders/README.md) — shader
  naming, the pipeline→shader map, SDL3 GPU binding order, uniform contract.

Keep these accurate: if a change invalidates a documented fact, update the doc
in the same change.

## Architecture

The include root is `src/`; engine headers are included with an `engine/`-rooted
path (`#include "engine/render/world_renderer.hpp"`). Everything lives under the
`mmo::engine` namespace, with sub-namespaces per module.

| Module | Namespace | What's there |
|---|---|---|
| `core/` | `mmo::engine::core` | logger, asserts, `Result`, `Handle`, `hash`/`StringId`, Tracy profiler hooks |
| `core/jobs/` | `core::jobs` | `JobSystem`, `parallel_for`, task handles — **kept serial/idle by design** |
| `core/asset/` | `core::asset` | asset registry + `FileWatcher` (hot-reload; opt-in via `MMO_HOT_RELOAD=1`) |
| `memory/` | `mmo::engine::memory` | arena allocator, frame allocator |
| `gpu/` | `mmo::engine::gpu` | SDL3 GPU abstraction — buffers, textures, shaders, pipelines, `PipelineRegistry`, `gpu_uniforms.hpp` (CPU↔GPU contract) |
| `render/` | `mmo::engine::render` | sub-renderers: terrain, grass, world/skybox, shadow map (CSM), SSAO/GTAO, bloom, volumetric fog, effects, text, UI; `lighting/` = clustered light binning |
| `render_graph/` | `render_graph` | pass scheduling (topo-sort, transient pool). **In-tree + tested but not wired into the renderer** — see design notes |
| `scene/` | `mmo::engine::scene` | cameras, frustum culling, transforms, `RenderScene` command buffer, `SceneRenderer` (frame orchestrator), `UIScene` |
| `animation/` | `mmo::engine::animation` | GLB skeleton, `AnimationPlayer` (crossfade), data-driven state machine, two-bone IK |
| `systems/` | `mmo::engine::systems` | camera controller/system, particle `EffectSystem` |
| `procedural/` | `mmo::engine::procedural` | tree + rock generators |
| `shaders/` | — | HLSL sources, compiled to SPIR-V at build time |

`Application` (`src/engine/application.hpp`) is the base class that owns the SDL
lifecycle, main loop, frame timing, and input. Consumers subclass it and
override `on_init`/`on_update`/`on_render`/`on_shutdown`.

### Key patterns
- **Y-up** coordinate system (x,z = ground plane, y = height) and
  **reversed-Z zero-to-one depth** (`GLM_FORCE_DEPTH_ZERO_TO_ONE` is PUBLIC on
  the engine target — all projection math is glm `RH_ZO`, with near/far
  swapped in the scene `glm::perspective` call so near→depth 1, far→depth 0;
  scene pipelines compare `GREATER_OR_EQUAL` and clear depth to 0.0. The
  shadow-cascade passes deliberately stay standard-Z — see architecture.md).
- `RenderScene` accumulates draw commands; `SceneRenderer` consumes them.
  It's passed by **non-const ref** (the renderer drains particle spawns), and
  `SkinnedModelCommand::bone_matrices` is a non-owning `std::span` over host
  storage that must stay valid through `render_frame`.
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

### Load-bearing constants
- `MAX_BONES = 128` — defined **only** in `animation/animation_types.hpp`. It
  caps the CPU-side per-instance bone-matrix arrays; the skinned shaders read
  a per-frame `StructuredBuffer<float4x4>` of packed palettes indexed by a
  `paletteOffset` uniform, so the GPU side has no compile-time bone count.
  Don't raise it without auditing every `std::array<glm::mat4, MAX_BONES>`.
- `MAX_BONE_PALETTES = 1024` (`scene/scene_renderer.hpp`) — skinned draws per
  frame; **excess draws are silently dropped + logged**.
- `CSM_MAX_CASCADES = 4` (`render/shadow_map.hpp`) — 2×2-quadrant D32 atlas.
- Cluster lighting (`render/lighting/light.hpp`): 16×9×24 grid,
  `MAX_LIGHTS = 256`, `MAX_LIGHTS_PER_CLUSTER = 128`.
- Formats (`gpu/pipeline_registry.hpp`): swapchain `RGBA8_UNORM`, HDR scene
  `RGBA16F`, depth `D32_FLOAT`.

### Gotchas (full list in [docs/architecture.md](docs/architecture.md))
- **GPU per-pass timings are stubs** — `GPUTimestampPool` returns zeros (SDL3
  limitation). Use Tracy (CPU) or RenderDoc/Nsight (GPU) for real numbers.
- glTF animation channels targeting non-skin nodes are **silently dropped**
  at load; `Mesh::near_only` exists but is **not enforced**; particle
  `particle_type: "sprite"` is declared but **not implemented**.
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
  `mmo_engine_tests`. Add a test for new logic.
- Shaders are `{name}.vert.hlsl` / `{name}.frag.hlsl` (entry points `VSMain` /
  `PSMain`), `.hlsli` for shared includes. See
  [src/engine/shaders/README.md](src/engine/shaders/README.md).

## Building & testing

```bash
cmake -S . -B build -G Ninja   # first configure only (fetches deps)
cmake --build build            # incremental — use this every time
ctest --test-dir build --output-on-failure
```

Options: `MMO_BUILD_TESTS` (default ON), `MMO_BUILD_BENCHMARKS` (default ON —
Google Benchmark, `*.bench.cpp` next to units, auto-globbed), `MMO_BUILD_EXAMPLES`
(default ON), `MMO_PLAYGROUND_SMOKE_TEST` (default OFF — registers the
playground smoke run as a ctest; needs GPU + display), `MMO_ENABLE_TRACY`
(default OFF).
Env: `MMO_HOT_RELOAD=1` enables the file watcher + model/shader hot-reload.

Integration test: `build/examples/playground/playground --smoke` (self-drives
240 frames, prints `[smoke] OK ...` with render stats, exit code 0). It also
saves + self-verifies a screenshot (`playground_smoke.png` in CWD) — **you can
Read that PNG to visually inspect what the renderer actually produced.** The
API is `Application::request_screenshot(path)` (captures the next frame's
final swapchain image, UI included; the capturing frame blocks on the GPU).

Benchmarks: `build/src/engine/mmo_engine_benchmarks` (add
`--benchmark_filter=<regex>` for a subset). Audit reports live in
`docs/audits/`.

## Consuming the engine

```cmake
add_subdirectory(path/to/engine)
target_link_libraries(my_game PRIVATE mmo_engine)
mmo_engine_copy_shaders(my_game)   # stages compiled SPIR-V next to the exe
```

## What you must not do
- No full clean rebuild — see [.claude/RULES.md](.claude/RULES.md).
- Don't reintroduce game/server/ECS/physics concepts into engine headers — this
  repo is deliberately engine-only.
- Don't change `MAX_BONES` casually — it sizes CPU-side palette arrays; audit
  every `std::array<glm::mat4, MAX_BONES>` user before raising it.
- Don't edit one side of a CPU↔GPU uniform struct pair without the other.
