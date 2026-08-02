# Project: Bracken

A standalone **C++23** real-time rendering engine (library/target name
`bracken`, formerly `mmo_engine`), extracted from the `mmo-4`
game project. Built on **SDL3 GPU** (Vulkan/Metal/D3D12 backends) with
HLSL→SPIR-V shader compilation, glTF model loading, skeletal animation, and a
clustered-forward renderer.

This repo contains **only the engine** — no game, server, networking, or ECS.
Rigid-body physics *is* engine infrastructure here: `src/bracken/physics/`
wraps Jolt Physics behind a glm-only pimpl API (owner decision 2026-07-21 —
see engine_design_notes.md). It builds as a static library (`bracken`, aliased
`bracken::bracken`; `src/bracken/bracken.hpp` is the umbrella over its public
surface) plus a GoogleTest suite
(`bracken_tests`), and one example app: `examples/playground`, a medieval
village orb-collector demo that doubles as the integration test (`--smoke N`
self-drives for N frames and exits — run it after changes that could affect
rendering). The village is assembled from a modular building pack, and the
flora (trees/bushes/rocks/grass tufts/flowers) converted 1:1 from a stylized
nature pack. The converted result is what the repo carries and all it carries:
`examples/playground/assets/` holds 52 `.glb`s, 4 terrain textures and
`village_layout.h` (50 MB, tracked in git, every file referenced by name from
`playground.cpp`), staged next to the exe by CMake. The ~2 GB of raw packs and
the `tools/build_medieval_assets.py` pipeline that converted them were removed
on 2026-08-01 — the outputs are now hand-maintained, not reproducible from
inputs in this tree.
On a sparse checkout without those assets the playground still builds and the
smoke test still passes — it falls back to a procedural-only scene (generator
trees + rocks, engine blade grass). In the full scene everything except the
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
incremental build is all that's needed after adding a file. Editing
`src/bracken/bracken.hpp` also triggers a reconfigure — the umbrella's header
list is read back out of it to generate the standalone-compile TUs.

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
  module map, module layering + the public/`internal/` surface,
  `Application` lifecycle, exact frame/pass order, GPU layer, threading &
  lifetime contracts, constants, known stubs/silent behaviors.
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
- [docs/rt_lighting_plan.md](docs/rt_lighting_plan.md) — **the plan of record
  for the ray-traced lighting system** (R0 foundations, R1 probe-field GI,
  R2 froxel-reservoir local-light shadows, R3 honest hit lighting, R4 HZB +
  screen probes and R5a SSILVB/bent-cone have landed; the rest of R5 onward is
  proposal): probe-field GI, froxel-reservoir local-light shadows,
  hit-lighting upgrades, screen-probe gather, denoiser evolution, scalability
  tiers. Read it before touching
  anything under `render/raytracing/` or the `rt_*` shaders — and before touching the GTAO
chain, whose visibility bitmask and bent cone are R5a of that plan.
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
| `contracts/` | *(each type keeps its own)* | the **leaf layer** — pure-data types two layers must agree on: `ModelHandle`, `EnvironmentState`, `CameraState`, `Frustum`, `velocity_math`, the host light descriptors `PointLightDesc`/`SpotLightDesc` **and** the GPU PODs `PointLight`/`SpotLight`/`LightHeader` they pack into, `GraphicsSettings` (every mode field is a scoped `enum class` — `ShadowMode`, `AOMode`, `AAMode`, `TonemapMode`, `DrawDistance`, `AnisotropicFilter`, `VSyncMode`, `ShadowCascades`, `ShadowResolution`, `LocalShadowResolution`, `WindowMode` — never a magic int), `RenderStats`. May include `core/`, glm and `<std>` and nothing else; a header's directory and its namespace are independent here. This is what makes the header graph a DAG — see `contracts/README.md` and architecture.md "Module layering" |
| `core/` | `bracken::core` | logger (**the engine's only log channel**), asserts, `Result`, `Handle`, `hash`/`StringId`, Tracy profiler hooks |
| `core/jobs/` | `core::jobs` | `JobSystem` (**always-on** lock-free work-stealing pool — Chase–Lev deques, `atomic::wait` parking), `TaskGroup`, `parallel_for`, task handles. Worker count via `BRACKEN_WORKERS` (=1 is single-worker debug mode, same code path) |
| `core/asset/` | `core::asset` | `FileWatcher` (hot-reload; opt-in via `BRACKEN_HOT_RELOAD=1`) + `AssetRegistry` (host-facing utility — engine systems register on the watcher directly) |
| `memory/` | `bracken::memory` | arena allocator, frame allocator |
| `gpu/` | `bracken::gpu` | SDL3 GPU abstraction — buffers, textures, shaders, graphics + compute pipelines (`gpu_compute.hpp`), `PipelineRegistry` (cache/lifecycle) over `pipeline_catalog.hpp` (one `constexpr` row per `PipelineType`: shaders, resources, vertex layout, target, depth/blend state), `gpu_uniforms.hpp` (CPU↔GPU contract) |
| `render/` | `bracken::render` | sub-renderers: terrain, grass, world/skybox, shadow map (CSM), GTAO (SSILVB visibility bitmask + bent cone), TAA, histogram auto exposure (compute), IBL (sky prefilter chain), bloom, volumetric fog, post-process stack (composite incl. color grading), water (surface + compute wave sim), text, UI, hierarchical-Z pyramid (`hzb.hpp`); `lighting/` = clustered light binning + budgeted local (point/spot) shadow atlas; `gi/` = the probe-field clipmap and the screen-probe gather |
| `render/raytracing/` | `render::rt` | software BVH (32-byte nodes, binned SAH, two-level BLAS/TLAS) + `RtScene`; `render/rt_renderer.hpp` drives the traced passes and their denoisers. Reflection and probe rays shade their hits through ONE shared include, `shaders/rt_hit_shade.hlsli` (CSM-shadowed sun + the frame's local lights + probe-field ambient, hemisphere where the field has no answer). **SDL3 GPU has no hardware ray tracing**, so the acceleration structure is the engine's own. Both signals are **stochastic** — the sun ray is jittered across the solar disk (soft shadows), the reflection ray importance-samples the GGX lobe — at one sample per pixel per frame, so the denoisers (`shaders/rt_shadow_denoise_*.comp.hlsl`, `rt_reflection_denoise_temporal.comp.hlsl`) are load-bearing, not polish. The TLAS is rebuilt only when the instance SET changes (count/handle-sequence hash); an instance that merely moved costs a bottom-up **refit** of the existing tree instead (`refit_tlas`, quality-gated on `sah_cost`), which is what keeps a live scene off the main thread. Both features off by default and independently switchable (`rt_shadows`, `rt_reflections`) |
| `render/gi/` | `render::gi` | probe-field global illumination (phase R1 of `docs/rt_lighting_plan.md`). `ProbeVolume` (`render/gi/probe_volume.hpp`) owns three camera-scrolled cascades of octahedral irradiance + visibility atlases (24x12x24 probes, spacing 150/300/600 world units, ~30 MB); `gi_math.hpp` is the header-only CPU source of truth for everything the GI shaders mirror — addressing, ray directions, the blend curve, and the sampler's cascade/weight math. **Complete as of R1c**: the field traces and accumulates itself, and `shaders/gi_sample.hlsli` replaces the DIFFUSE half of the ambient term in every scene fragment shader (three banded atlases, three sampler slots) plus the volumetric fog's in-scatter. Off by default (`gi_mode`); GI on builds the ray-tracing acceleration structure with attributes |
| `render_graph/` | `render_graph` | pass scheduling (topo-sort, transient pool). **In-tree + tested but not wired into the renderer** — see design notes |
| `scene/` | `bracken::scene` | transforms, motion history, `FramePacket` (immutable per-frame command snapshot) + `FrameBuilder` (the only write interface) + `FrameSink` (its host-facing half, what `SimFrame::frame()` returns) + `DrawFlags` + `PacketRing` (SPSC packet ring), `RenderPlan` (which passes run this frame — pure, tested headless), `ScenePrep` (the Prepare-stage tier boundary as a type: device-free, stateless, cannot reach a main-thread API), `SceneRenderer` (frame orchestrator) |
| `animation/` | `bracken::animation` | GLB skeleton, `AnimationPlayer` (crossfade), data-driven state machine, two-bone IK |
| `systems/` | `bracken::systems` | camera controller/system |
| `effects/` | *(namespaces unchanged)* | the particle-effect triad in one place: `effect_definition.hpp` (authoring data), `EffectSystem` (sim-stage owned: spawned/updated/snapshotted inside the sim job), `EffectRenderer` (draws the packet snapshot) |
| `procedural/` | `bracken::procedural` | tree + rock generators |
| `audio/` | `bracken::audio` (+ `::dsp`, `::synth`, `::music`) | SDL3 playback device, lock-free mixer, 3D positional voices, WAV clips — **plus a software synthesizer**: band-limited oscillators / ADSR / SVF (`dsp/`), patches + polyphonic voices (`synth/`), and a sample-accurate song sequencer (`music/`). The one subsystem with a thread the engine does not own |
| `save/` | `bracken::save` | `SaveDocument` — path-addressed typed records, so schema changes never invalidate old saves; unknown keys round-trip; **atomic** write (temp + rename) + checksum, so an interrupted save cannot corrupt the previous one |
| `navigation/` | `bracken::nav` | `NavGrid` — walkability from a height function + A* with string-pulled paths. A grid, not a polygon navmesh: built from the height function the engine already has, and honest about its limit (one height per cell, so no bridges over roads) |
| `physics/` | `bracken::physics` | `PhysicsWorld` — rigid-body physics over vendored **Jolt** (pimpl; Jolt never appears in a public header). Fixed-step accumulator `update()` (tier owned/sim-stage), sphere/box/capsule/heightfield bodies, ray casts; Jolt's step jobs run on the engine `JobSystem` via the `JoltJobSystem` bridge in **`physics/internal/`** — one pool owns the cores |
| `dev/` | `bracken::dev` | `DevUI` Dear ImGui developer overlay (F10 in the playground; no-op stub when `BRACKEN_ENABLE_DEV_UI=OFF`) + `dev_panels` (built-in Stats/Environment/Graphics/Camera panels). ImGui appears only in `dev/*.cpp` |
| `testing/` | `bracken::testing` | `frame_test.hpp` — `TestFrame` + `FrameAccess`, published **for host games** to unit-test their frame code with no GPU/window/SDL. Header-only; nothing in the engine's production path includes it |
| `shaders/` | — | HLSL sources, compiled to SPIR-V at build time |

**Public vs internal.** `src/bracken/bracken.hpp` is the umbrella over the
supported surface — the 24 headers the playground actually includes, grouped
and annotated. It is a statement of *what to reach for*, explicitly **not** a
stability promise (pre-1.0; this refactor broke six of those signatures). A
header under `<module>/internal/` is outside that surface and may include a
vendored dependency freely — `physics/internal/jolt_job_system.hpp` is the one
header in the tree that includes Jolt. `public_headers.test.cpp` enforces both
halves: nothing outside an `internal/` directory may transitively reach Jolt,
ImGui, tinygltf or stb_, and `application.hpp` must not drag the `gpu/` layer
into every host TU. The umbrella's 24 headers are also each compiled
**standalone** by the generated `bracken_public_headers` target. Adopt
`<module>/internal/` for any future header that has to include a vendored
dependency.

`Application` (`src/bracken/application.hpp`) is the base class that owns the
SDL lifecycle, the **pipelined** main loop (depth 2 — Simulate(i+1) overlaps
Render(i)), frame timing, and input. Consumers subclass it and override
`on_init`/`on_simulate(SimFrame&)`/`on_shutdown` (plus optional
`on_main_frame(MainFrame&)` and `on_event`). There is no
`on_update`/`on_render`: the host *describes* a frame through
`SimFrame::frame()`'s `FrameSink` — from a pool job — and the engine
draws it.

### Key patterns
- **Level of detail** is opt-in per model and lives entirely outside the hot
  path: `ModelManager::load_model_with_lods` generates simplified levels
  (`mesh_simplify.hpp`, vertex clustering) and registers each as its own
  MODEL, so the Prepare stage only swaps a handle before binning and every
  stage after it is unchanged. Selection (`select_lod`) is a pure function of
  apparent size (bounding radius / distance), so it stays bit-identical at any
  worker count. It must be `load_model_with_lods`, not `load_model` then
  `build_lods`: uploading frees the CPU geometry simplification needs, and
  build_lods logs an error rather than silently doing nothing in that case.
- **Y-up** coordinate system (x,z = ground plane, y = height) and
  **reversed-Z zero-to-one depth** (`GLM_FORCE_DEPTH_ZERO_TO_ONE` is PUBLIC on
  the engine target — all projection math is glm `RH_ZO`, with near/far
  swapped in the scene `glm::perspective` call so near→depth 1, far→depth 0;
  scene pipelines compare `GREATER_OR_EQUAL` and clear depth to 0.0. The
  shadow-cascade passes deliberately stay standard-Z — see architecture.md).
- Frame data flows one way, by value: in `on_simulate` the host fills a
  `scene::FramePacket` through `sim.frame()`'s **`scene::FrameSink`** (3D
  draws, lights, UI, effect spawns, environment), the engine's
  sim-stage tail drives the camera into the packet (unless the host took the
  rig back with `set_camera_driven_by_engine(false)`), appends the
  effect-instance snapshot + scene/sky clocks and `seal()`s, and the renderer consumes `const FramePacket&`. `FrameSink` is
  the host half of `FrameBuilder`; the sim-stage tail (`seal`,
  `set_frame_timing`, `alloc_effect_snapshot`, `set_engine_frame`, the
  staged-environment reads, `packet()`) and `FramePacket`'s lifecycle
  (`begin_fill`/`begin_consume`/`reset`/`reserve_block`) are **private** —
  friends are `Application`, `systems::EffectSystem` and
  `testing::FrameAccess`. Everything is copied or packet-allocated (bone
  palettes via `alloc_bone_palette`, UI text copied into the packet arena),
  with **two named lifetime exceptions**: `EffectSpawnCommand::definition`
  and `EffectEmitterSnapshot::definition` are host-owned pointers that must
  outlive every instance spawned from them (documented at the top of
  `scene/frame_packet.hpp`; null is a `BRACKEN_CONTRACT` at emit and at
  consume). A packet is consumed exactly once, then retired to its
  `PacketRing`. `BonePaletteId`s held by the host across `seal()` are stale;
  only re-read them from sealed commands.
- **Diagnostics: the engine does not drop work silently.** Every budget or
  lookup failure that used to be a `continue` is counted in
  `RenderStats::diagnostics` (`contracts/render_stats.hpp`) —
  `draws_dropped_unknown_model`, `skinned_dropped_no_skeleton`,
  `palettes_over_budget`, `effect_models_unresolved`, `lights_dropped`,
  `cluster_lights_dropped`, `camera_missing`. The per-chunk accumulators fold
  in chunk-index order, so the counts are bit-identical at any worker count.
  Per-pass CPU record time is real (`core::PassTimeline` →
  `RenderStats::pass_times_ms`), the GPU half is honestly `nullopt`, and
  `ModelLoadReport` summarizes what a `.glb` load quietly discarded. Check
  these before believing "nothing is drawing" is a shader bug.
- Hosts can unit-test their own frame code headlessly:
  `bracken/testing/frame_test.hpp` ships `testing::TestFrame` (build a frame
  through a `FrameSink`, `seal()`, read back typed views + `digest()`) and
  `testing::FrameAccess` (the friend escape hatch for the engine tail). No
  GPU, no window.
- All lighting is **linear-space**; `render::EnvironmentState` colors are
  linear, the composite pass tonemaps to the swapchain.
- Animation is per-instance: an `AnimationPlayer` + `AnimationStateMachine`.
  The host feeds parameters ("speed", "attacking"); the state machine drives
  transitions. Foot IK / body lean are applied by the host *after* the
  animation update, before render submission.
- CPU uniform structs in `gpu/gpu_uniforms.hpp` must match their HLSL cbuffer
  counterparts **byte-for-byte** (`alignas(16)`, HLSL packing rules). This is
  the most common way to silently break rendering. Both halves are now
  checked: a `BRACKEN_HLSL_CBUFFER` / `BRACKEN_HLSL_FIELD` block
  (`gpu/hlsl_packing.hpp`) follows every struct, so a CPU-side layout edit is
  a **compile error in the same TU**; and `gpu/uniform_layout.test.cpp`
  reflects the **compiled SPIR-V** and fails when a cbuffer and its mirror
  disagree. Adding a cbuffer to a shader without a row in that test's table
  also fails.
- Engine code is **game-agnostic**: no gameplay concepts (combat, inventory,
  skills) belong in engine headers. `InputHandler` is raw input only (`ActionMap` in `action_map.hpp` is the host-facing layer above it: named actions bound to keys/mouse/pad, with edges, normalized axes and rebinding persistence — the engine never learns what an action *means*);
  `CameraController` exposes generic shake/focus/fov primitives.
- **Threading** (Phase 4 of [docs/multithreading_plan.md](docs/multithreading_plan.md)):
  the frame **is pipelined** — `Application::run()` kicks `on_simulate(i+1)`
  as a pool job and renders frame i on the main thread while it runs
  (depth 2; one frame of input latency, documented on `Application`). The
  job-system pool is always on — `init_engine` starts it before SDL; parked
  workers cost ~zero. `on_simulate` is tier **sim-stage**: nothing reachable
  from it may touch GPU/SDL/main-thread-tier APIs — asserted in debug by
  `BRACKEN_ASSERT_NOT_SIM_STAGE()`, which works because the stage rides the
  **job node** (`core/jobs/job_stage.hpp`) rather than a thread-local, so it
  is still correct when a job is stolen or when the main thread executes a
  sim job inside a helping `wait`; the
  frame **fence** (join → hot-reload poll → deferred settings/screenshot/
  camera-forward → FPS → `on_main_frame`) is the only point where the main
  thread may read live game state. The renderer's **Prepare stage fans out**:
  culling/batching/binning loops run as `parallel_for` chunks over the
  sealed packet, per-chunk partials merged single-threaded in chunk-index
  order — bit-identical at any worker count; no atomics, mutexes, or shared
  cache-line writes inside per-item frame loops; grain guards keep small
  scenes effectively serial. Engine APIs carry thread-safety tiers
  (`main-thread` / `sim-stage` / `worker-safe` / `owned`, asserted in debug —
  architecture.md "Threading & lifetime contracts" lists the exact mechanism
  behind each). Host-boundary preconditions are separate and live **in
  release too**: `BRACKEN_CONTRACT` (packet-before-seal, seal-once, lane
  index, every `Application` facade member that needs `init_renderer()`).
  There is no switch that compiles them out. Engine-internal invariants
  and per-item hot-loop checks stay on `BRACKEN_ASSERT`. Every `std::atomic`
  documents its protocol (what it publishes, ordering, release/acquire
  pairing partner) at its declaration; bare `seq_cst` in a hot path needs a
  justifying comment. Rebuild `build/` with `-fsanitize=thread` and run the
  suite after touching concurrency code.

- **Audio threading**: SDL calls back on *its* thread whenever the device
  wants samples. That thread never blocks, allocates or logs, so everything
  crossing into it goes through a bounded lock-free ring
  (`audio/command_ring.hpp`) and pre-sized buffers; a new song is adopted
  under a `try_lock` and the *outgoing* song is handed back for the game
  thread to free (`Sequencer::swap_song`). The payoff for hosts is that
  `play_sound` / `play_music` / `set_listener` are **worker-safe** — call them
  straight from `on_simulate`. Audio does not touch the `FramePacket` at all.

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
  The *budgets* live there; the light **structs** `PointLight`/`SpotLight`/
  `LightHeader` are `contracts/light.hpp` (host-facing data contract — the
  frame packet carries them, so they must not drag the renderer in).
  `render/lighting/light.hpp` includes it, so both spellings keep working.
- Formats: swapchain `RGBA8_UNORM`, HDR scene `RGBA16F`, depth `D32_FLOAT` —
  the live values are `PipelineRegistry`'s members (`gpu/pipeline_registry.hpp`,
  the swapchain one updated at runtime by `set_swapchain_format`), and the
  catalogue's `TargetFormats` defaults (`gpu/pipeline_catalog.hpp`) must agree
  with them.

### Gotchas (full list in [docs/architecture.md](docs/architecture.md))
- **Per-pass timings are CPU-record only** — `RenderStats::pass_times_ms` is
  `PassTiming{name, cpu_ms, gpu_ms}`: `cpu_ms` is real (wall time spent
  BUILDING the pass's command stream, via `core::PassTimeline`), `gpu_ms` is
  `nullopt` because SDL3 exposes no timestamp query API. A pass can record in
  40 us and cost 4 ms on the device — for GPU-side cost use RenderDoc/Nsight.
  `Application::present_wait_ms()` is the CPU-bound-vs-GPU-bound signal.
- glTF animation channels targeting non-skin nodes are dropped (counted and
  summarized in one `ModelLoadReport` line per model since phase 4) at load.
  (`Mesh::near_only` was deleted in phase 9 — it was a hint the draw walk
  never read.)
- `render_graph/` is not in the frame path — don't route the renderer through
  it without a measured reason (design notes explain why).
- A number measured in a default `build/` is not a performance number: no
  `CMAKE_BUILD_TYPE` means `-O0` with live tier assertions. Reconfigure with
  `-DCMAKE_BUILD_TYPE=RelWithDebInfo` before quoting one.
- `bracken.hpp` is not a stability contract, and `<module>/internal/` is not a
  suggestion — `public_headers.test.cpp` fails the suite if a public header
  reaches Jolt/ImGui/tinygltf/stb_, or if `application.hpp` starts exporting
  the `gpu/` layer again.

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
- **Logging**: `BRACKEN_LOG_{TRACE,DEBUG,INFO,WARN,ERROR,FATAL}(LOG_TAG, ...)`
  only. `SDL_Log`, `std::cerr` and `std::cout` are **banned** in engine
  sources — they bypass `core::Logger` and are therefore invisible to a host's
  `LogSink` and to `set_min_level`. `core/logger_discipline.test.cpp` fails the
  suite on any reintroduction (exemptions: `core/logger.cpp`, the SDL bridge;
  `core/assert.hpp`, the abort path). Each `.cpp` declares its category once as
  a file-local `constexpr const char* LOG_TAG` in an anonymous namespace.
  Pick the severity honestly: a failed resource creation is `ERROR`, a
  fallback/degradation is `WARN`, per-item boot chatter is `DEBUG`. When
  editing log calls in bulk, `tools/logdiff.sh [<rev>]` proves no message body
  was lost.
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

There is **one** build tree, `build/`, configured by the
`cmake -S . -B build -G Ninja` line above. There is no preset file, no preset
zoo and no second directory: five parallel trees meant five things that could
disagree with each other and with the documentation describing them, and four
of them were stale most of the time.

Everything the old lanes did is a `-D` flag on the same tree, used when you
actually need it:

| Want | Flags |
|---|---|
| The `NDEBUG` path (`BRACKEN_ASSERT` and the debug-only tier checks compile out; `BRACKEN_CONTRACT` stays) | `-DCMAKE_BUILD_TYPE=RelWithDebInfo` |
| A benchmark number worth quoting | `-DCMAKE_BUILD_TYPE=RelWithDebInfo` (CMake warns if you forget) |
| ThreadSanitizer, after touching the job system, an atomic, or anything in the multithreading plan | `-DCMAKE_{C,CXX}_FLAGS=-fsanitize=thread -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread` |
| Address + UB sanitizer — use-after-free / out-of-bounds across the arena, frame allocator, packet ring and job-system node pools | `-DCMAKE_{C,CXX}_FLAGS="-fsanitize=address,undefined -fno-sanitize=enum -fno-omit-frame-pointer"` |

`-fno-sanitize=enum` is deliberate: the engine bounds-checks enum parameters at
the host boundary, and the only way to test such a guard is to hand it an
out-of-range enumerator.

Reconfiguring `build/` with different flags is a rebuild, not a *clean*
rebuild — the fetched dependency trees (SDL_shadercross/DXC, Jolt) survive,
which is the thing the no-clean-rebuild rule exists to protect.

Integration test: `build/examples/playground/playground --smoke` (self-drives
240 frames, prints `[smoke] OK ...` with render stats, exit code 0). It also
saves + self-verifies a screenshot (`playground_smoke.png` in CWD) — **you can
Read that PNG to visually inspect what the renderer actually produced.** The
API is `Application::request_screenshot(path)` (deferred to the frame fence —
a request from sim frame i captures frame i's final swapchain image, UI
included; the capturing frame blocks on the GPU).

Known limitation of that shot, so you do not misread it as a rendering bug:
the capture frame (`smoke_frames - 20`, i.e. 220 of 240) currently lands with
the player at a building doorway, and the camera boom has no collision, so it
sits **inside the wall** and the frame is mostly one flat facade. Materials,
normal maps, AO, shadows, UI and the dev overlay are all visible and correct
in it, and the self-verify (not black, not solid) still does its job — but it
is not the village establishing shot it was set up to be
(`playground.cpp`, the `cam_cfg`/`set_pitch`/`set_yaw` block in the smoke
init). Pre-existing and unrelated to the refactor: `mean_luma=80.5
stddev=38.8` is unchanged all the way back to the Phase 8 baseline. Judge
rendering health from the diagnostics line and the screenshot statistics;
treat the PNG's framing as a "did anything draw at all" check until it is
fixed.

**The smoke run is hermetic w.r.t. `settings.cfg`, and only became so on
2026-08-01.** `Application::init_renderer()` auto-loads `GraphicsSettings` from
`settings_file_path()` and `shutdown_renderer()` auto-saves them, so every
`--smoke` run used to *start* from whatever the previous run (or a concurrent
one, or a developer's Graphics-panel tick) had left in that one global file —
and then wrote its own end state back. That is why plain `--smoke` was observed
building a 27 MB acceleration structure and tracing with `rt_shadows` /
`rt_reflections` both defaulting to `false` and no env var set, and why the
recorded `mean_luma` figures drifted by several points with no code change at
all. The playground now calls `Application::set_settings_persistence(false)`
as soon as `--smoke` is parsed, resets `GraphicsSettings{}` wholesale in
`setup_smoke_run()`, and pins `rt_shadows`/`rt_reflections` from
`PLAYGROUND_RT` (unset = both off) and `gi_mode` from `PLAYGROUND_GI`
(`probes`|`probes_screen`|`off`; unset = off, same rule and same reason).
`[smoke] settings persist=0 shadow_mode=…`
prints what the run actually rendered with — check it before believing any
luma number, yours or a commit message's.

**Screenshot baselines** (`mean_luma` / `stddev`, 1600×900, Metal). These were
re-measured hermetically after that fix; the previously circulated
`87.6 / 45.4` is *not* reproducible and was itself taken against a drifted
settings file. `--smoke 30` at commit `8f21a3d` — which predates the entire
G-buffer/reflection series — produces the identical figure below, which is the
evidence that no RT-series commit moved the RT-off path.

| Invocation | mean_luma / stddev (pre-R5a) | at R5a |
|---|---|---|
| `--smoke 30` (RT off) | `86.6 / 44.9` | `86.3 / 45.5` |
| `--smoke 30 PLAYGROUND_AO=0` | `87.0 / 44.5` | `87.0 / 44.5` (unchanged — AO off) |
| `--smoke 30 PLAYGROUND_AO=2` | `86.4 / 45.0` | `86.0 / 45.7` |

**R5a moved every AO-ON row on purpose** (SSILVB visibility bitmask + bent
cone, docs/rt_lighting_plan.md §5.6): mean essentially unmoved, spread up.
Thin geometry — fences, crop rows, grass tufts, distant canopies — lost the
grey wash the old thin-occluder heuristic produced, and solid crevices, eaves
and beam junctions gained real contact depth. `PLAYGROUND_AO=0` reproducing
its old figure to the digit is the evidence that the AO-off path is untouched;
the "R5a" figures here were measured on Linux/Vulkan and the pre-R5a column is
the Metal history, so compare within a column, not across.

Re-measured on the Linux/GCC/Vulkan checkout at R4b (this machine reads `86.5`
where the Metal table above reads `86.6` — see the memory note on Metal-only
baselines). The three the ray-tracing phases gate on:

| Invocation | pre-R5a | at R5a |
|---|---|---|
| `--smoke 30` (RT off, GI off) | `86.5 / 44.9` | `86.3 / 45.5` |
| `--smoke 30 PLAYGROUND_RT=both PLAYGROUND_GI=probes` | `81.7 / 49.8` | `81.8 / 50.0` |
| `--smoke 30 PLAYGROUND_RT=both PLAYGROUND_GI=probes_screen` | `82.3 / 50.4` | `82.2 / 50.5` |

The GI rows moved for a second reason on top of the AO change: since R5a the
diffuse GI term is gated by the AO pass's BENT CONE — sampled along the bent
normal, scaled by the cone aperture — instead of by the scalar screen AO,
never by both. Every row above is 3-run identical.

The third stopped being a copy of the second at R4b, when `GiMode::ProbesScreen`
gained the screen-probe gather (`render/gi/screen_probes.hpp`,
docs/rt_lighting_plan.md §5.5) instead of behaving as `Probes`.

| Invocation | pre-R5a | at R5a |
|---|---|---|
| `--smoke 30 PLAYGROUND_GI=probes` | `84.3 / 46.9` | `84.6 / 46.8` |
| `PLAYGROUND_RT=both --smoke 30` | `83.9 / 47.9` | `83.7 / 48.4` |
| `PLAYGROUND_RT=both PLAYGROUND_GI=probes --smoke 30` | `81.7 / 49.8` | `81.8 / 50.0` |
| `PLAYGROUND_RT_LOCAL=ratio PLAYGROUND_RT=both --smoke 30` | `85.3 / 47.5` | `85.0 / 47.9` |

The GI row moved from `84.7 / 46.6` at R3 (`pbr.hlsli`'s rough-tier ambient
specular now blends toward the probe field above roughness 0.6). The
`PLAYGROUND_RT=both` row did **not** move at R3 and that is a coincidence of
the aggregate, not a null result: 94% of the screenshot's bytes differ, to a
maximum of 172/255, because reflections of sky-facing foliage brightened by as
much as reflections of shadowed facades darkened. If you are checking the
reflection path, read the chrome orb over the plaza, not the mean.

The last two rows were measured on a **Linux/Vulkan** box, not the Metal
machine the three above came from, so compare them with each other and not
across the table. The R2b row reads *brighter* than plain `RT=both` because
`LocalRtShadowMode::Ratio` retires the rasterized local-shadow atlas whose
hard-edged tile shadow over-darkened the plaza; the difference is concentrated
in the orb light's pool around the character, where the traced pass puts a
soft, distance-widening shadow in place of a hard blob. `local_tiles` reads 0
under that pin — the atlas retirement showing up in the smoke line itself.

**The GI row is the only one that is not RT-off, and it is deliberately
different.** Phase R1c made the probe field light the frame, so a GI-pinned run
*must* diverge — lower mean and wider histogram, because surfaces that cannot
see the sky stop being lit as though they could. The RT-off rows are unchanged
by R1c and are the regression gate; the GI row is the GI gate, and it reproduced
identically over three consecutive runs.

Backend caveat: the three RT-off rows above were measured on **Metal**; the GI
row was measured on **Vulkan**, where the plain `--smoke 30` row reproduces as
`86.5 / 44.9` — 0.1 off the Metal figure. Compare like with like, and judge
smoke health from the diagnostics line first.

Benchmarks: `build/src/bracken/bracken_benchmarks` (add
`--benchmark_filter=<regex>` for a subset). Numbers from a default configure are
**meaningless** — no `CMAKE_BUILD_TYPE` means `-O0` with live asserts.
Reconfigure with `-DCMAKE_BUILD_TYPE=RelWithDebInfo` for anything you intend to
quote. Audit reports live in `docs/audits/`.

## Consuming the engine

```cmake
add_subdirectory(path/to/engine)
target_link_libraries(my_game PRIVATE bracken::bracken)
bracken_copy_shaders(my_game)   # stages compiled SPIR-V next to the exe
```

```cpp
#include "bracken/bracken.hpp"   // the supported surface, in one header
```

`bracken::bracken` is an alias for `bracken`; prefer the namespaced spelling —
a typo in it fails at configure time, whereas a bare `bracken` silently
degrades into a `-lbracken` link flag. Linking is `PUBLIC`-transitive: you get
SDL3/SDL3_ttf/SDL3_image, glm, the `src/` include root and
`GLM_FORCE_DEPTH_ZERO_TO_ONE` (load-bearing — your glm math must agree with the
engine's reversed-Z). Jolt is linked **PRIVATE**: you get its symbols, never
its headers.

For a game's own tests, `#include "bracken/testing/frame_test.hpp"` — build a
frame through a `FrameSink`, `seal()`, assert on the typed views, with no GPU
and no window.

## What you must not do
- No full clean rebuild — see [.claude/RULES.md](.claude/RULES.md).
- Don't reintroduce gameplay concepts (combat, inventory, skills, ECS) or
  server/networking code into engine headers — this repo is deliberately
  engine-only. `physics/character_controller.hpp` (capsule controller: slopes, step-up, crouch, jump — kinematic, never a rigid body) and `animation/attachment.hpp` (bone-space attachment: reads the joint WORLD transform, not the skinning matrix) are engine infrastructure too. Physics is the exception: it lives in `physics/` as engine
  infrastructure (Jolt-backed), added by owner decision 2026-07-21. Jolt
  types must stay out of public bracken headers (pimpl — link stays PRIVATE).
- Don't change `MAX_BONES` casually — it sizes CPU-side palette arrays; audit
  every `std::array<glm::mat4, MAX_BONES>` user before raising it.
- Don't edit one side of a CPU↔GPU uniform struct pair without the other.
