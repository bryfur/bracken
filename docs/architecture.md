# Engine Architecture Reference

The detailed map of how the engine fits together at runtime. For the short
orientation see [CLAUDE.md](../CLAUDE.md) / [README.md](../README.md); for
design rationale ("why is X off by default") see
[engine_design_notes.md](engine_design_notes.md).

Everything in this file is verified against the code it cites. If you change
one of the cited facts, update this file in the same change.

---

## Module map

Include root is `src/`; headers are included as `bracken/<module>/<file>.hpp`.
Root namespace `bracken`, one sub-namespace per module.

| Path | Namespace | Contents |
|---|---|---|
| `src/bracken/` (root) | `bracken` | `Application` + `SimFrame`/`MainFrame` (the two per-frame host views it hands the hooks), `ModelLoader`/`ModelManager`, `model_utils`, `InputHandler`/`InputState`, `Heightmap`, and `bracken.hpp` — the umbrella over the public surface |
| `contracts/` | *(unchanged per type)* | the **leaf layer**: pure-data types two or more layers must agree on — `ModelHandle`, `EnvironmentState`, `CameraState`, `Frustum`, `velocity_math`, the host descriptors `PointLightDesc`/`SpotLightDesc` and the GPU PODs `PointLight`/`SpotLight`/`LightHeader`, `GraphicsSettings` (every mode field a scoped `enum class`), `RenderStats` (+ `FrameDiagnostics`, `PassTiming`). May include `core/`, glm and `<std>` and nothing else; namespaces are those of the owning module, not the directory. See [`contracts/README.md`](../src/bracken/contracts/README.md) |
| `core/` | `core` | `Logger` (+`BRACKEN_LOG_*`), `assert.hpp`, `Result<T,E>`, `Handle` pools, `hash.hpp` (`StringId`, FNV-1a), `profiler.hpp` (Tracy macros) |
| `core/jobs/` | `core::jobs` | `JobSystem` (always-on lock-free work-stealing pool — Chase–Lev deques, MPSC injection, pooled job nodes), `TaskGroup`, `parallel_for`, `TaskHandle` |
| `core/asset/` | `core::asset` | `FileWatcher` (hot-reload poller); `AssetRegistry` (typed, stable handles — a host-facing utility: no engine system instantiates it) |
| `memory/` | `memory` | `Arena` (bump + destructor registration), `FrameAllocator` |
| `gpu/` | `gpu` | `GPUDevice`, `GPUBuffer`, `GPUTexture`, `ShaderManager`, `GPUPipeline`, `GPUComputePipeline`/`ComputePass` (`gpu_compute.hpp`), `PipelineRegistry` (cache/lifecycle) + `pipeline_catalog.hpp` (what each pipeline *is*), `gpu_uniforms.hpp` (CPU↔GPU contract), `gpu_debug.hpp` |
| `render/` | `render` | `RenderContext`, sub-renderers: world/skybox, terrain, grass, shadow map (CSM), GTAO, TAA, auto exposure (compute histogram), IBL (sky prefilter chain), bloom, volumetric fog, post-process stack (composite incl. color grading), water (surface + compute wave sim), text, UI; `color_grading.hpp` |
| `render/lighting/` | `render::lighting` | `ClusterGrid` (CPU light binning), `LocalShadowMap`/`LocalShadowSelector` (budgeted point/spot shadow atlas + tile cache), and the cluster/atlas **budget constants** in `light.hpp` (the light *structs* are `contracts/light.hpp`) |
| `render_graph/` | `render_graph` | pass scheduler (topo-sort, transient pool) — **in-tree, tested, not wired into `SceneRenderer`** |
| `scene/` | `scene` | `FramePacket` (immutable per-frame snapshot) + `FrameBuilder` (lane-based write interface) + `PacketRing` (SPSC handoff ring), `SceneRenderer` (frame orchestrator), `ScenePrep`, `RenderPlan`, `MotionHistory`, `transform_math.hpp` |
| `animation/` | `animation` | `animation_types.hpp` (skeleton/clips/constants), `AnimationPlayer`, `AnimationStateMachine`, `ik_solver` |
| `systems/` | `systems` | `CameraController` / `CameraSystem` |
| `effects/` | `bracken` + `systems` + `render` | the particle-effect triad, together at last: `effect_definition.hpp` (authoring data), `EffectSystem` (sim-stage simulation), `EffectRenderer` + `particle_transform.hpp` (drawing). Namespaces are unchanged — the directory groups the feature, not the namespace |
| `procedural/` | `procedural` | `tree_generator` (ez-tree algorithm), `rock_generator` (SDF + marching cubes) |
| `physics/` | `physics` | `PhysicsWorld` (glm-only pimpl over vendored **Jolt Physics** — linked PRIVATE, Jolt never in public headers). `physics/internal/` holds the `JoltJobSystem` bridge onto `core::jobs::JobSystem` — the one header in the tree that includes Jolt (see "Physics" below) |
| `dev/` | `dev` | `DevUI` — the Dear ImGui developer overlay (F10 in the playground) — and `dev_panels` (the built-in Stats / Environment / Graphics / Camera panels). Compiles to a no-op stub when `BRACKEN_ENABLE_DEV_UI=OFF`; ImGui appears only in `dev/*.cpp`, never in a header |
| `testing/` | `testing` | `frame_test.hpp` — `TestFrame` + `FrameAccess`, the headless frame fixture published **for host games**: build a frame, seal it, read back typed views and the packet digest with no GPU, window or SDL. Header-only; nothing in the engine's production path includes it |
| `shaders/` | — | HLSL sources → SPIR-V at build time; see [shaders/README.md](../src/bracken/shaders/README.md) |

### Module layering

The engine's **header** graph is a DAG, and
`core/module_layering.test.cpp` keeps it one. Reading downward, every edge
points from a layer to one below it:

```
core                                          sink: fan-out 0, includes nothing in-tree
  core/jobs, core/asset, memory, contracts    contracts: core + glm + <std> only
    gpu, animation
      model_loader.{hpp,cpp}, heightmap.{hpp,cpp}   asset-layer leaves, at the root
        render, render/lighting, procedural, physics
          scene                               FramePacket / FrameBuilder / PacketRing
            effects, systems
              @root                           application.hpp, sim_frame.hpp, input
                dev
                testing                       frame_test.hpp: host-facing, nothing
                                              in-tree includes it
```

Two things this diagram does not hide:

- **The root directory is not a layer.** It holds the top of the stack
  (`application.hpp`, `sim_frame.hpp`, the input pair) *and* two dependency
  leaves that `render/` and `scene/` reach up for by path — `model_loader.hpp`
  and `heightmap.hpp`. That is untidy, and a `model/` directory would tidy it,
  but no header in the root's top half is included by any header in `render/`,
  `scene/` or `systems/`, so it closes no cycle. Honest summary: acyclic, with
  two files filed one directory too high.
- **`.cpp` files are excluded**, and only from this claim. `application.cpp`,
  `dev/dev_panels.cpp` and `scene/scene_renderer*.cpp` all reach downward into
  the modules they orchestrate, which produces residual module cycles at *link*
  level. A `.cpp` cannot participate in a compile cycle, so those cost nothing
  and are deliberately not chased.

Why it matters: until this was made a DAG the production module graph had a
six-module strongly-connected component — `{@root, dev, render, render/lighting,
scene, systems}` — with 21 elementary cycles of length ≤ 4, and **zero**
file-level include cycles, which is why nothing in the build ever complained.
The cause was not wrong names, it was pure-data types filed under whichever
module happened to author them: `EnvironmentState` under `render/` although the
frame packet carries one, `CameraState`/`Frustum` under `scene/` although every
sub-renderer takes them, `GraphicsSettings`/`RenderStats` at the root although
`scene/` reads and writes them. Each misfiling reversed the direction of a real
dependency edge. Moving them to `contracts/` reverses those edges back;
namespaces were deliberately left alone, so it cost include lines and nothing
else.

The three checked properties are: the header module graph has no cycle; nothing
under `contracts/` includes anything but `bracken/core/*`, `bracken/contracts/*`,
glm and `<std>`; and `core/`, `memory/` and `gpu/` never include upward (tests
and benchmarks exempt — cross-layer contract tests like
`gpu/uniform_layout.test.cpp` exist precisely to check that the layers agree).

### Public surface and `<module>/internal/`

Layering says which header may include which. A second, independent axis says
which headers a **host** is meant to include at all, and it is written down in
two places:

- **`src/bracken/bracken.hpp`** — the umbrella. It includes exactly the set
  `examples/playground` uses, grouped by section with one comment per header.
  Read it as a table of contents for the engine. It is explicitly **not** a
  stability promise: the engine is pre-1.0, and the refactor that introduced
  the umbrella changed six of those signatures in the same release. Publishing
  an API contract on top of that would be signalling.
- **`src/bracken/<module>/internal/`** — the convention for a header that is
  *not* part of that surface. An `internal/` header may include a vendored
  dependency freely; a public one may not. Today there is exactly one:
  `physics/internal/jolt_job_system.hpp`, which includes `Jolt/Jolt.h` and
  derives from `JPH::JobSystemWithBarrier`.

The Jolt invariant is the reason the convention exists. "Jolt types never
appear in a public bracken header" was asserted in four documents and enforced
by nothing: it held only because Jolt is linked `PRIVATE`, which is a *link*-time
property standing in for an *include*-time rule. A consumer with Jolt already on
its include path would have compiled that header straight out of the public tree.
`public_headers.test.cpp` now checks the include graph directly — no header
outside an `internal/` directory may transitively reach Jolt, ImGui, tinygltf or
stb_ — plus two more facts worth keeping true: the umbrella covers every bracken
header the reference consumer includes, and **`application.hpp` pulls in no
`gpu/` header** (before the `ModelHandle` hoist into `contracts/`, the chain
`application.hpp → sim_frame.hpp → frame_builder.hpp → frame_packet.hpp →
model_loader.hpp` exported `gpu_buffer.hpp`, `gpu_texture.hpp` and
`gpu_types.hpp` into every host translation unit, purely so the packet could
name one handle type).

Compilability is a separate question a source scan cannot answer, so
`src/bracken/CMakeLists.txt` reads the header list back out of `bracken.hpp` and
generates one translation unit per header into the `bracken_public_headers`
object library. Each includes its header twice — first line of the file, nothing
before it — which proves both self-sufficiency and the include guard.

---

## Coordinate, depth, and color conventions

- **Y-up**, right-handed: x/z is the ground plane, y is height. glTF assets are
  `+Y up` / `-Z forward` (standard glTF), loaded with no rotation.
- **Reversed-Z, zero-to-one depth**: `GLM_FORCE_DEPTH_ZERO_TO_ONE` is a PUBLIC
  compile definition on the `bracken` target (`src/bracken/CMakeLists.txt`) —
  it propagates to consumers. All projection math assumes glm `RH_ZO`, and the
  scene projection is **reversed-Z**: near/far are swapped in the
  `glm::perspective` call (`systems/camera_system.cpp`), so the near plane maps
  to device depth **1** and the far plane to **0**. This spreads float32 depth
  precision near-uniformly over MMO-scale view distances instead of burning it
  all within a few meters of the camera. Consequences, wired through the code:
  - Scene depth targets clear to **0.0** (`scene_renderer.cpp`,
    `post_process_stack.cpp`); the `PipelineConfig` default depth compare is
    `GREATER_OR_EQUAL` (`gpu/gpu_pipeline.hpp`).
  - The skybox draws at z = 0.0 with `GREATER_OR_EQUAL` (`skybox.vert.hlsl` +
    the skybox pipeline), passing only where nothing was drawn.
  - Depth-reading shaders treat device depth **0.0 as sky/far**
    (`gtao`/`gtao_temporal`/`blur_ao`/`volumetric_fog`). The linearization
    `viewZ = |B / (depth + A)|` with `A = proj[2][2]`, `B = proj[3][2]` holds
    for standard and reversed matrices alike, so `blur_ao`/`composite` needed
    no math change.
  - `SceneRenderer` recovers the near plane as `m[3][2]/(m[2][2] + 1)`
    (under standard-Z it was `m[3][2]/m[2][2]`).
  - `Frustum::extract_from_matrix` is convention-agnostic: the two depth-plane
    rows swap near/far roles under reversed-Z but the extracted plane set (and
    thus culling) is identical (`contracts/frustum.cpp`).
  - **Exception: shadow cascades stay standard-Z** (clear 1.0,
    `LESS_OR_EQUAL` pipelines + compare sampler, `orthoRH_ZO` light
    projections). Their orthographic depth is already linear, so reversed-Z
    buys nothing, and all receiver bias/PCF/PCSS math assumes near=0/far=1.
  Never mix in `-1..1`-depth projection helpers, and never build a
  depth-tested scene pass with a standard (non-reversed) projection.
- **Linear-space lighting**: all colors in `render::EnvironmentState`
  (`contracts/environment.hpp`) are linear. The scene renders into a linear HDR
  target; the composite pass tonemaps (ACES et al.) + gamma-corrects to the
  swapchain. sRGB textures are converted on sample.
- **Texture formats** (`PipelineRegistry`'s members, surfaced to the catalogue
  as `gpu::PipelineFormats`): swapchain `R8G8B8A8_UNORM`, HDR scene color
  `R16G16B16A16_FLOAT`, depth `D32_FLOAT`. Only the swapchain entry changes at
  runtime.

---

## `Application` lifecycle

`Application` (`src/bracken/application.hpp`) owns SDL, the **pipelined**
main loop, frame timing, input, and the renderer facade. Consumers subclass
it:

- Pure virtual: `on_init()`, `on_simulate(SimFrame&)`.
- Optional: `on_shutdown()`, `on_main_frame(MainFrame&)`,
  `on_event(const SDL_Event&) -> bool` (return `true` to consume an event
  before `InputHandler` sees it).
- There is **no `on_update`/`on_render`**: the host does not "render" — it
  *describes* a frame through the `FrameSink` that `SimFrame::frame()` returns, and the
  engine draws it. `on_simulate` **runs as a job on the worker pool**,
  overlapped with the previous frame's rendering (tier `sim-stage` — see
  Threading & lifetime contracts); `on_main_frame` runs at the frame fence
  on the main thread with no sim in flight.
- Boot sequence — **host-driven, engine-owned entry points**: `run()` drives
  only `on_simulate`/`on_main_frame`/`on_event`; the surrounding lifecycle is
  `init_engine()` → `start()` → `run()` → `stop()` → `shutdown_engine()`.
  `Application::start()`/`stop()` are public and call the protected
  `on_init()` (which typically calls `init_renderer(width, height, title,
  world_width = 32000, world_height = 32000)`) / `on_shutdown()` (typically
  `shutdown_renderer()`) hooks; `stop()` is a no-op unless `start()` returned
  true, and runs `on_shutdown()` at most once.
  `on_init`/`on_shutdown` run on the main thread with the GPU available and
  no sim in flight. `quit()` is worker-safe (`running_` is atomic) and stops
  the loop at the next iteration.

**The frame schedule** (`application.cpp::run()`, plan Phase 4; pipeline
depth 2 — Simulate(i+1) overlaps Render(i)). Bootstrap: pump events once,
then kick sim 0 (no overlap for frame 0). Each steady-state iteration i:

1. **Join** sim job i (`JobSystem::wait`; the main thread helps the pool).
   An exception escaping `on_simulate` is fatal-logged, the partial packet
   abandoned (`reset()`), and the exception **rethrown out of `run()`** on
   the main thread — host bugs must not vanish into the pool.
2. **Frame fence** — no sim in flight, the pipeline's quiescent point:
   `FileWatcher::poll_main_thread()` (hot-reload applies here, never while
   a packet referencing the old resource is in flight); pending
   graphics-settings application (`request_graphics_settings`; two
   single-writer slots — sim-phase and main-thread/DevUI — main wins when
   both are set); camera-forward feedback applied to the `InputHandler`;
   FPS accounting (so `fps()` is never written while a sim that may read it
   is in flight); `on_main_frame(MainFrame&)` — the **only** point where
   the main thread may safely read live game state (snapshot here for DevUI
   panels); then the deferred screenshot request is forwarded to the
   renderer (a request from sim frame i captures frame i's rendering).
3. **Pump events** (`on_event`/DevUI first, then
   `InputHandler::process_event`). On quit: retire the sealed frame, break.
4. **Kick sim i+1**: dt computed at kick (`SDL_GetTicksNS()`, clamped
   0.1 s) and stamped into the packet; by-value `InputSnapshot` taken
   (`InputHandler::snapshot()` — drains the text buffer); packet acquired
   from the ring (`FrameBuilder` with 16 fixed lanes); the viewport
   snapshotted onto `SimFrame::screen_width()/height()`; sim job submitted.
   The job body: `SimFrame` → `on_simulate` → the engine's **sim-stage
   tail**, in this order —
   (a) **camera** (unless `set_camera_driven_by_engine(false)`):
   `set_screen_size` from the kick-time viewport, `update(dt)`,
   `fb.set_camera(...)`, and the camera-forward feedback unless the host
   set it itself. This runs *after* `on_simulate` so the frame's target
   move and its shakes are both already in;
   (b) apply the builder's effect spawns to the sim-owned
   `EffectSystem` in deterministic (lane, emission) order, then
   `EffectSystem::update` on the pool, then `snapshot_into(fb)` (the
   effect-instance snapshot);
   (c) advance the scene/sky clocks (sky = dt ×
   the staged environment's `cloud_speed`) and `set_engine_frame(...)`
   — → `seal()`. (`Application::run_sim_job`, `application.cpp`.)
5. **Render frame i** on the main thread (`render_frame(packet)` — Prepare
   fan-out + Record + present, unchanged from Phase 3) while sim i+1 runs.
6. **Retire** packet i.

The packet handoff needs no extra synchronization: seal happens inside the
sim job and the TaskHandle kick/join edges publish it (the `PacketRing`
publish/acquire pair is not needed under `run()`). After the loop, any
in-flight sim is joined before `run()` returns, so teardown never races the
pipeline.

**Input latency**: the snapshot taken at kick(i+1) is simulated in frame
i+1 and presents at frame i+1 — one pipeline slot between input and
photons, inherent to pipelining and stated on `Application` deliberately;
`on_main_frame` exists for the rare truly-latency-critical main-thread
need.

Side effects worth knowing:
- `init_engine` initializes the **always-on job-system pool** before SDL and
  every other subsystem (worker count from `BRACKEN_WORKERS`; see Threading &
  lifetime contracts below); `shutdown_renderer` joins the pool at its top,
  before any GPU teardown.
- `init_renderer` auto-loads `GraphicsSettings` from `settings_file_path()`
  (platform pref path) and `shutdown_renderer` auto-saves them.
- If the env var `BRACKEN_HOT_RELOAD=1` is set, `init_engine` starts the global
  `FileWatcher` and `init_renderer` registers model + shader-SPIR-V hot-reload
  callbacks directly on `FileWatcher::instance()` via
  `PipelineRegistry::enable_hot_reload` / `ModelManager::enable_hot_reload`
  (`application.cpp`); `AssetRegistry` plays no part in this path. Default:
  off, no watcher thread.
- `init_dev_ui()` (protected; optional, call after `init_renderer`) brings up
  the Dear ImGui developer overlay (`dev::DevUI`, `dev_ui()` accessor): it
  binds the built-in panels, **claims the `SceneRenderer` post-UI callback
  slot**, and feeds SDL events to the overlay before game input. When
  `BRACKEN_ENABLE_DEV_UI=OFF` it returns `false` and every `DevUI` call is a
  safe no-op.

---

## Frame data flow

Frame data flows one way, by value (Phases 2 + 4 of
[multithreading_plan.md](multithreading_plan.md)): the host describes a frame
into a `scene::FramePacket` through a `scene::FrameBuilder`, the engine seals
it inside the sim job, and the renderer consumes the sealed packet `const` —
one pipeline slot later, overlapped with the next frame's sim:

```
sim job i+1 (pool):   SimFrame sim(builder, input snapshot, dt, ...)
                      → host on_simulate(sim):
                          AnimationStateMachine::update → AnimationPlayer::update
                          → (optional) apply_foot_ik / apply_body_lean
                          → fb = sim.frame(): draw_model/draw_skinned/lights/UI/
                            spawn_effect/set_environment (camera: AIM only)
                      → engine sim-stage tail: camera set_screen_size/update(dt)
                        → fb.set_camera + camera-forward feedback
                        → effect spawns → EffectSystem::update
                        → effect snapshot + scene/sky clocks into the packet
                      → fb.seal()          (the packet's one release-publish edge)
main thread, frame i: → scene::SceneRenderer::render_frame(const FramePacket&)
                      → ring.retire(p)     (arena reset, memory retained)
```

The builder is engine-created with 16 fixed lanes; the host emits from
`on_simulate`'s root code (the `FrameBuilder` conveniences = lane 0) or fans
emission out across `fb.lane(i)` with its own `sim.run_async` jobs, one lane
per job. `sim.parallel_for` bodies must **not** emit into the builder (no
automatic lane assignment). The host never calls `seal()` or
`set_frame_timing` — the engine does both.

The camera is engine-driven by default
(`Application::set_camera_driven_by_engine`): the host aims `camera()` during
`on_simulate` and the tail does the viewport (from the kick-time snapshot —
`RenderContext` is main-thread tier, so neither host nor engine may read
`screen_width()` from the sim stage; `SimFrame::screen_width()` carries it),
the `update(dt)`, the packet's `set_camera`, and the camera-forward feedback
for the next input snapshot's movement basis. Because that runs at the END of
the frame, camera *output* read during `on_simulate` is the previous frame's,
while shakes and target moves staged anywhere in the frame land in this
frame's camera. A host with its own rig turns the drive off and owes the
frame all four calls (`SimFrame::set_camera_forward` overrides the feedback
either way).

Inside `render_frame`, the CPU **Prepare** work (culling, batching, binning
— see "Prepare-stage fan-out" under Threading & lifetime contracts) fans out
as jobs over the sealed packet (Phase 3 of the plan): packet reads after
seal are plain loads from any worker, which is exactly what makes the
fan-out safe.

`FramePacket` (`scene/frame_packet.hpp`) carries the frame's complete,
immutable description: static (`ModelDrawCommand`) and skinned
(`SkinnedDrawCommand`) model draws, bone palettes, point/spot lights,
billboards, debug lines, particle-effect spawns, 2D UI commands, camera +
environment snapshots (each with a `has_*()` set-this-frame flag), the
skybox/ground/grass draw toggles, and dt/time/frame-index. It also carries a
sim-stage-written **engine frame section** (never host-written): the
effect-instance snapshot (`effect_emitters()` — `EffectEmitterSnapshot`
records referencing ranges of `effect_particles()` —
`EffectParticleSnapshot` position/scale/rotation/opacity/color),
`effect_instance_count()` (DevUI stats), and the sim-advanced
`scene_time()`/`sky_time()` clocks, gated by `has_engine_frame()` (always
true under `Application::run`; false for hand-built test packets, whose
consumers keep their previous clocks). All storage is
carved from the packet's own arena — steady-state frames allocate no heap —
and the packet walks a `Retired → Filling → Sealed → Consuming → Retired`
state machine with debug asserts on both directions (writes only while
Filling, reads only after seal).

`FrameBuilder` (`scene/frame_builder.hpp`) is the **only** write interface,
and hosts see exactly half of it: `SimFrame::frame()` returns a
`scene::FrameSink` (`scene/frame_sink.hpp`), a pointer-sized non-owning view
that forwards the host surface — draws, lights, billboards, debug shapes,
effect spawns, `ui()`, `lane()`, camera/environment/toggles — and nothing
else. The engine's **sim-stage tail** (`seal`, `set_frame_timing`,
`alloc_effect_snapshot`, `set_engine_frame`, `for_each_effect_spawn`, the
staged-environment reads, `packet()`) is *private* on `FrameBuilder`, as are
`FramePacket`'s lifecycle and raw allocator (`begin_fill`, `begin_consume`,
`reset`, `reserve_block`, `take_palette_slot`, the drop counters). The friend
lists are the complete set of code that may run them: `Application` (the sim
tail and the abandon-on-throw path), `systems::EffectSystem`
(`snapshot_into` calls `alloc_effect_snapshot`, and is *not* reached through
`Application`), `scene::PacketRing` (stage handoff), and
`testing::FrameAccess` (the published escape hatch,
`bracken/testing/frame_test.hpp`). Before this the rule was a doc comment
plus a `BRACKEN_ASSERT`, so a shipped game that sealed early republished a
packet the renderer was already consuming, silently.

Its emit methods mirror the retired `RenderScene`/`UIScene` API one-for-one
(`draw_model`, `draw_skinned`, `add_point_light`, `add_billboard_3d`,
`add_debug_line/sphere/box`, `spawn_effect`, `ui().add_text(...)`, …).
Emission is multi-lane for the later jobified phases: each `Lane` is
cache-line-padded, thread-owned chunked storage in the packet arena (one
atomic `fetch_add` per block reservation, private bumps per item); `seal()`
merges lanes in deterministic (lane-index, emission-order) order —
byte-identical at any worker count when lanes are keyed by job/chunk index.
Single-threaded hosts use the `FrameBuilder` forwards (= lane 0) and never
see a lane.

Model and skinned commands carry an optional `motion_id` (default 0): a
host-unique nonzero id lets the velocity prepass pair this frame's transform
(and skinned palette) with last frame's via `scene::MotionHistory`; id 0 is
treated as static (camera-only velocity). `MotionHistory` is two open-addressed
tables (linear probing, Fibonacci hash, per-slot frame stamp) swapped at
`end_frame()`, so the per-frame prune is a stamp bump rather than a node-by-node
clear and steady-state frames allocate nothing.

Lights are submitted as **host descriptors**, not as the GPU-layout PODs:
`add_point_light` takes `render::lighting::PointLightDesc`, `add_spot_light`
takes `SpotLightDesc` (both in `contracts/light.hpp`), and the builder
converts. The PODs carry `_pad0`, `_pad1` and the engine-written
`shadow_slot`, which hosts were aggregate-initializing positionally straight
across; the descriptors carry only what a host decides, and spot cones are
`core::Degrees` half-angles with the cosine taken inside the builder.

Per-draw options are `scene::DrawFlags` (`scene/draw_flags.hpp`) — a named
`uint32_t` bit set (`None`, `ForceNonInstanced`, `NoFog`) with `|`, `&` and
`has(set, flag)`, stored as one word in `ModelDrawCommand::flags` /
`SkinnedDrawCommand::flags`. They replace two adjacent anonymous `bool`
parameters that were silently transposable at every call site.

**Everything crossing the boundary is by value or packet-allocated, with two
named exceptions.** Bone palettes live in packet memory
(`fb.alloc_bone_palette(joint_count)` returns a writable `span<glm::mat4>` +
a `BonePaletteId` for `draw_skinned` — the old non-owning
`bone_matrices` span over host storage is gone), and UI text/labels are
copied into the packet arena at emit time. The exceptions are long-lived
asset pointers, not per-frame state: `EffectSpawnCommand::definition` and
`EffectEmitterSnapshot::definition`, both raw pointers into host storage and
the second retained across frames by the `EffectSystem`. **An
`EffectDefinition` handed to `spawn_effect` must outlive every instance
spawned from it.** Null is a `BRACKEN_CONTRACT` at emit and at consume; a
*dangling* one is only covered by the host's ownership discipline, which is
why the rule is written at the top of `scene/frame_packet.hpp` (where that
comment used to claim, falsely, that a sealed packet carried no lifetime
contract at all). Handle-ising the two pointers behind an `EffectLibrary` is
the real fix and is deferred — see `docs/refactor_plan.md` §6. `ModelHandle`s
are values and carry nothing. `BonePaletteId`s are opaque while filling and remapped to
dense `FramePacket::palettes()` indices at seal — ids held by the host across
`seal()` are stale; only re-read them from sealed commands. Each
`BonePalette` record carries a prefix-summed `matrix_offset` that is exactly
the GPU palette offset once the renderer uploads palettes in `palettes()`
order.

`PacketRing` (`scene/frame_packet_ring.hpp`, header-only) is the SPSC
fill↔consume handoff: `acquire_for_fill()` (round-robin, asserts the slot is
Retired), `publish()` (one release store), `try_acquire_sealed()` (one
acquire exchange), `retire()` (arena reset, memory retained).
`Application::run()` owns a depth-2 ring (one packet rendering while the
next is being simulated) and drives the acquire/retire cycle itself; under
`run()` the publish/acquire pair is not needed — the sim job's TaskHandle
kick/join edges order the handoff. Hosts that bypass `run()` may still
drive the Phase-2 sequential flow directly (acquire → build → seal →
`render_frame` → retire).

### Which passes run this frame — `scene/render_plan.hpp`

`plan_frame(const GraphicsSettings&, const FramePacket&, RendererCaps)` is
the authoritative "which passes run this frame" decision point, evaluated
once at the top of `render_frame` and read by everything downstream:
`has_content`, `use_hdr_path`, `taa_active`, `ibl_active`, `fog_active`,
`do_frustum_cull`, `draw_dist_sq` and `active_cascades` (which **is** the
"shadows on" predicate — `ShadowMap::active_cascades()` is clamped to
[1, `CSM_MAX_CASCADES`], so a zero means the atlas is unavailable or the
host set `ShadowMode::Off`). It is **pure**: `RendererCaps` carries the four
sub-renderer readiness bits plus the sticky "a camera was ever set" flag,
which is why the whole tree is testable headless
(`scene/render_plan.test.cpp`).

Two decisions deliberately stay in `render_frame`: the minimized-window
early-out (it must sit between `begin_frame()` and `end_frame()` so the
command-buffer lifecycle stays balanced) and the mid-frame TAA history drop
for a jittered frame whose prepass/velocity target turned out unavailable.
The plan also *reports* rather than performs its two invalidations
(`invalidate_taa_history`, `invalidate_auto_exposure`) — `render_frame`
makes those calls, after the minimized early-out, so a minimized frame drops
nothing.

`RenderPlan` is the **immutable half only**. Three frame-scoped renderer
members are written mid-frame, after the camera/frustum state is final —
`frame_prepass_active_` (the prepass reports whether it recorded),
`frame_ao_texture_` (GTAO publishes its output) and
`frame_shadow_uniforms_` — and therefore cannot live in a const struct built
before those passes run.

### Pass order inside `SceneRenderer::render_frame`

(`scene/scene_renderer.cpp`; each GPU section has a matching debug group +
timer label, so RenderDoc captures read in this order.)

`SceneRenderer`'s implementation is **one class across six translation
units** — no interface change, and the pass *sequence* below still lives as
one linear script in `render_frame`:

| File | Holds |
|---|---|
| `scene/scene_renderer.cpp` | init/shutdown/lifecycle/setters + the `render_frame` orchestration script |
| `scene/scene_renderer_upload.cpp` | the per-frame GPU uploads (instances, bone palettes) — the **main-thread side** of the Prepare tier boundary; everything here takes an `SDL_GPUCopyPass*` |
| `scene/scene_renderer_prepass.cpp` | the depth prepass + its instanced/static/skinned draw walks |
| `scene/scene_renderer_draw.cpp` | the main pass: `render_3d_scene`, model/skinned/instanced/transparent draws, the model-family pipeline binds, UI, billboards, debug lines |
| `scene/scene_renderer_shadow.cpp` | CSM cascade passes, the instanced shadow walk, local (point/spot) tile passes, `bind_shadow_data` / `bind_cluster_lights` |
| `scene/scene_renderer_capture.cpp` | the screenshot download + PNG encode (the only GPU download + fence block in the renderer) |

`scene/scene_renderer_internal.hpp` carries what more than one of them needs
and no host should see (`LOG_TAG`, the `apply_mesh_material` template). It
does **not** carry the class's private member declarations — C++ requires
those inside the class definition, so `scene_renderer.hpp` keeps them.

1. **CPU prep** — capture the frame's unjittered view-projection (and apply
   the TAA sub-pixel Halton(2,3) jitter to the draw projection — applied
   exactly when `aa_mode == AAMode::TAA` and the TAA resolve can run;
   `SceneRenderer::taa_jitter_enabled()` reports it), propagate
   `EnvironmentState` to
   sub-renderers, **adopt** the packet's sim-advanced scene/sky clocks
   (`scene_time()`/`sky_time()` — the sky clock is a cloud scroll PHASE,
   `dt` scaled by the staged environment's `cloud_speed`, integrated on the
   sim side; the renderer no longer integrates dt, and hand-built packets
   without an engine section keep the previous clocks), pre-upload every
   model referenced by the packet's effect-instance snapshot
   (`ensure_models_uploaded(frame.effect_emitters())` — uploads can't run
   inside a render pass; the snapshot is the render side's ONLY view of the
   particle system, since spawns were applied and the live `EffectSystem`
   updated on the sim side before the packet sealed),
   update CSM cascade matrices, cull + build instance batches
   per cascade and main view (each instance carries its previous-frame model
   matrix from the `MotionHistory`), resolve skinned commands (model, bounds,
   main-view visibility, previous-frame transform/palette offset — the
   palette GPU offset is read straight from the packet's prefix-summed
   `BonePalette::matrix_offset`), run the
   local-shadow selection (`render::lighting::LocalShadowSelector`: pick the
   budgeted shadow-casting point/spot lights by score with hysteresis, assign
   atlas tiles, decide which tiles re-render this frame, and write each
   selected light's `shadow_slot` into the frame's light copies), build the
   cluster light grid on the CPU from those copies (both skipped when the
   scene has zero dynamic lights). The heavy loops in this step — per-view
   culling + instance batching (main view, cascades, local-shadow tiles),
   skinned resolve + palette layout, cluster binning, grass/terrain tile
   culls — are the Phase-3 **Prepare fan-out**: each
   runs as `parallel_for` chunks over the sealed packet with per-chunk
   partials merged in chunk-index order after the join (see "Prepare-stage
   fan-out" below); grain guards keep small scenes to one inline chunk.
2. **`FrameUploads`** — one `SDL_GPUCopyPass` for every per-frame upload:
   instances, shadow instances, bone palettes, particle instances, debug
   lines, grass chunks, cluster light buffers, local-shadow slot data (only
   when a tile re-renders this frame).
3. **`IBL`** (only when `ibl_enabled` and `render::IBL` initialized) — sky
   image-based-lighting upkeep: `render::IBL` snapshots the sky-relevant
   `EnvironmentState` fields (sun/sky/fog colors + the cloud
   coverage/density/scale and haze-strength shaping — NOT `sky_time` or
   `cloud_speed`, which are time-like) and, when they changed, re-renders its
   prefilter
   chain (64px sky-capture cube → capture mips → 7-mip GGX-prefiltered
   specular cube → 16px irradiance cube; the split-sum BRDF LUT renders once
   at init) **amortized at 9 tiny cube-face passes per frame** — a full
   refresh spans 7 frames, an unchanged sky records nothing. The forward
   shaders sample the chain in their ambient term (see step 8). The capture
   pass evaluates the exact `sky_common.hlsli` radiance the skybox draws, so
   the reflected sky can never drift from the visible one.
4. **`ShadowPass`** — cascades rendered into the 2×2-quadrant D32 atlas
   (`ShadowPass:Cascade0..3`), only when `shadow_mode != ShadowMode::Off`
   (settings modes are `enum class`es in `graphics_settings.hpp`; the on-disk
   settings format stays numeric).
5. **`LocalShadowPass`** (only when `local_shadows` is on and the scene has
   dynamic lights) — re-renders this frame's dirty local-shadow tiles into
   the 8×8-tile D32 atlas (`render::lighting::LocalShadowMap`; independent of
   the CSM toggle). Point lights own 6 consecutive face tiles (~90° FOV,
   slightly widened so seam PCF taps stay valid), spots one tile at their
   cone angle; all local maps are **standard-Z perspective** (near→0,
   far = light radius→1, `LESS_OR_EQUAL`, clear 1.0), matching the CSM
   cascades' standard-Z convention so the caster pipelines are shared
   verbatim. Cached tiles persist (the pass LOADs; a dirty tile clears itself
   via the `LocalShadowClear` fullscreen draw with the viewport depth range
   pinned to [1,1]); a tile re-renders only when its light moved/changed, a
   dynamic caster (nonzero `motion_id` or skinned) is inside the light
   sphere, or the assignment is new — capped at 6 tiles/frame with
   round-robin fairness (worst case = stale tiles, never a hitch). Casters =
   model + skinned commands within the light radius, drawn individually
   through the CSM `ShadowModel`/`ShadowSkinnedModel` pipelines
   (`mesh.cast_shadows` respected; terrain and grass do NOT cast local
   shadows).
6. **`DepthPrepass`** (HDR path — runs regardless of `ao_mode`; it feeds
   GTAO, the TAA velocity, and the `EQUAL` forward draws) — every glTF
   alphaMode `OPAQUE` draw
   (terrain, static/instanced/skinned models, grass blades) renders
   position-only into the scene depth target plus an RG16F **velocity**
   target (screen-sized, cleared to zero; `SceneRenderer::velocity_texture()`).
   Velocity = UV-space delta, **current minus previous**, computed from
   UNJITTERED matrices (`contracts/velocity_math.hpp` is the CPU reference;
   consumers reproject with `prev_uv = uv - velocity`). Statics use the
   previous view-projection with their current model matrix; objects
   submitted with a nonzero `motion_id` get their previous model matrix (and,
   for skinned, previous bone palette via the double-buffered palette SSBO)
   from `scene::MotionHistory`. BLEND draws are excluded; MASK draws go
   through alpha-tested prepass variants (`DepthPrepass*Masked`, no-cull:
   texcoord passthrough + the EXACT forward alpha test incl. the mip-scaled
   coverage boost, so both passes discard the same texels). Skinned MASK
   meshes are the one exception (no masked skinned variant — they keep the
   old forward-only depth behavior).
7. **`GTAO`** (only when `ao_mode != Off` and the prepass ran this frame) —
   `render::GTAO`, computed from the prepass depth alone BEFORE the forward
   pass; the forward shaders sample the result in their ambient term (screen
   AO never touches direct sun light or speculars, and is min-combined with
   material ORM AO to avoid double-darkening). Chain of 5 raster fullscreen
   passes at the AO working resolution (half res for `GTAO_Half`, full res
   for `GTAO_Full`): horizon-based GTAO (Jimenez et al.; 3 slices × 4 steps
   per side, per-pixel IGN + per-frame golden-ratio jitter, edge-aware
   depth-reconstructed normals, thin-occluder heuristic, world-unit radius ×
   `ao_radius`) → temporal accumulation (velocity reprojection into a
   double-buffered RG16F history storing AO + linear depth; relative-depth
   disocclusion rejection, feedback ≤ 0.9; history invalidates on
   resize/quality change/AO-off/prepass gaps — same policy as TAA) →
   `BlurAO` ×2 (depth-aware separable blur of the accumulated AO; the
   history itself stays unfiltered) → joint-bilateral upsample to the
   full-res R8 AO texture (+ `ao_strength` as a power curve; exact
   passthrough at full working res). MASK geometry is IN the prepass (the
   alpha-tested variants above), so foliage both receives and casts screen
   AO — leaf-on-leaf occlusion is what visually separates a canopy from a
   flat blob. BLEND geometry stays absent: it neither receives its own AO
   nor occludes (enforced by `screenAOEnabled = 0` in its per-mesh uniforms —
   without the gate the forward pass would sample the AO of whatever is
   BEHIND the surface).
8. **`HDROffscreen`** (the normal path whenever the post stack initialized) —
   the whole 3D scene renders into the linear-HDR offscreen target: terrain,
   grass, static/instanced/skinned models, skybox, sorted transparent
   (alphaMode `BLEND`) meshes, particles, debug lines.
   The offscreen clear color tracks `env.fog_color_horizon`.
   After a depth prepass the pass LOADs the prepass depth and every OPAQUE
   draw binds its `*DepthEqual` pipeline variant (compare `EQUAL`, depth
   writes off; forward + prepass vertex shaders are `precise` so positions
   are bit-identical); MASK/BLEND/skybox/particles keep `GREATER_OR_EQUAL`.
   Every model-family/terrain/grass draw binds the frame's AO texture (the
   1×1 white dummy when AO is off; BLEND draws always get the dummy — the
   screen AO under a transparent belongs to the surface behind it) plus the
   three sky-IBL samplers (irradiance cube, prefiltered specular cube, BRDF
   LUT — typed dummies when IBL is off) plus the local-shadow atlas +
   comparison sampler at the shader's last sampler slot (the CSM atlas is a
   typed filler when the local system is unavailable — it is only sampled for
   lights with `shadow_slot >= 0`). Ambient = `ambientIBL` when the
   `iblEnabled` uniform flag is on, the hemisphere `ambientPBR` otherwise;
   the combined (material ∧ screen) AO and the shadow-driven ambient lift
   multiply either path identically (`pbr.hlsli`). Cluster lights with a
   shadow slot sample the local atlas inside `accumulateClusterLights`
   (point lights select a face by the dominant axis of light→fragment; 4
   hardware-PCF taps + world-space normal/slope bias + smooth fade at the
   light radius — `cluster_lighting.hlsli`).
9. **`Water`** (only when a `render::WaterRenderer` region is configured via
   `SceneRenderer::water()->set_region(...)`) — two parts. **`WaterSim`**
   (recorded earlier, right after `IBL` upkeep, outside every render pass):
   the engine's first compute-storage-TEXTURE user — a 512² RG16F ping-pong
   wave-equation field (height + previous height) stepped at fixed 60 Hz
   substeps (≤4/frame), with Gaussian ripple impulses queued via
   `water()->add_ripple(pos, radius, strength)`, semi-Lagrangian advection
   along the region's flow, and an absorbing border band; the field covers a
   square `sim_extent` window around `sim_center`. **`Water` pass** (right
   after `HDROffscreen`, before TAA): the HDR color + depth are snapshotted
   via `SDL_CopyGPUTextureToTexture` into screen-sized SAMPLER copies, then
   one render pass re-binds the live HDR color + depth with `LOAD` and draws
   the water surface grid (world-XZ float2 vertices, lifted to the water
   level and displaced by the sim field in the VS). The fragment shader
   composites in-shader (the pipeline is opaque, depth test + write on, so
   TAA reprojection and the fog's depth reads treat water like any opaque
   surface): refraction = distorted sample of the color copy with
   depth-proportional offset (samples landing on geometry in front of the
   surface fall back to the undistorted UV), Beer-Lambert absorption toward
   the region's deep color, screen-space reflections ray-marched against the
   depth copy (on-screen geometry mirrors in the water; confidence-faded into
   the IBL prefiltered specular cube as fallback — mip by
   `reflection_roughness`, horizon-fallback when IBL is off),
   Schlick fresnel, a fresnel-scaled sun glint, shoreline + sim-crest foam,
   a depth-feathered waterline, and the shared `fog.hlsli` atmospheric fog.
   Normals = finite-differenced sim ripples + analytic beat-frequency detail
   waves scrolling with the flow. Known limitation: transparents and
   particles rendered in step 8, so the water overdraws them where they
   overlap on screen. Terrain above the water level hides the plane by the
   depth test, so a region rect can generously cover a carved riverbed.
10. **Post chain** (all offscreen, swapchain not yet acquired):
   **`TAA`** (only when `aa_mode == AAMode::TAA`;
   `render::TAA`: resolves the jittered HDR offscreen color against a
   double-buffered RGBA16F history using the prepass velocity + scene depth —
   3×3 closest-depth velocity dilation, depth reprojection for velocity-less
   pixels (sky/MASK; far-plane pixels reproject as w=0 directions), YCoCg
   variance clipping, 5-tap Catmull-Rom history sampling, Karis
   luminance-weighted blend with feedback ≤ 0.9 — then writes the mildly
   sharpened result back over the offscreen color so bloom/fog/composite read
   the stabilized image; history invalidates on resize/mode change/prepass
   gaps) → **`AutoExposure`** (only when `auto_exposure` is on;
   `render::AutoExposure`, the engine's first COMPUTE passes: a 256-bin
   log2-luminance histogram of the TAA-resolved HDR color — 16×16 workgroups
   accumulate in LDS then atomically merge non-zero bins — then a
   single-workgroup reduce with percentile clipping (darkest 60% / brightest
   5% ignored by default) and rate-limited temporal adaptation
   (`auto_exposure_speed_up/down` EV/s, clamped to
   `auto_exposure_min/max_ev`) into a tiny persistent state buffer; the
   first `SNAP_FRAMES` (3) metered frames after init/re-enable snap straight
   to the target; any frame the passes cannot record stales the state like
   TAA history) → `Bloom` (13-tap downsample / 9-tap tent upsample;
   the prefilter thresholds post-MANUAL-exposure luminance — the adapted
   auto-EV is deliberately not included, see stubs) → `VolumetricFog` (ray
   march against depth + shadow atlas; all parameters come from
   `EnvironmentState` — sun_dir, fog_color_horizon, and the
   `fog_volumetric_*` fields; god rays select a cascade per march sample
   and sky pixels are marched too, so shafts silhouette against the sky;
   god rays force-disabled when shadows are off because the atlas would
   contain garbage).
11. **Late swapchain acquire** → **`Composite`** (depth-aware joint-bilateral
   upsample of the half-res fog, scene + bloom + fog, then exposure =
   manual bias × exp2(adapted EV) — the adapted EV is read from the
   auto-exposure state buffer bound as the pass's only fragment SSBO (a
   zeroed dummy + flag-off when auto exposure is inactive) — then white
   balance (pre-tonemap, linear LMS gains from
   `EnvironmentState::grade_temperature/tint`), tonemap, ASC-CDL
   slope/offset/power (`grade_slope/offset/power`), contrast/saturation.
   Both grading blocks are flag-gated on non-neutral values so the
   all-neutral default is pixel-identical to the ungraded path. AO is no
   longer composited here; it was applied in the forward ambient term.
   With `aa_mode == AAMode::FXAA`
   composite goes to an LDR intermediate and **`FXAA`** resolves to the
   swapchain; otherwise (Off and TAA — TAA already resolved pre-bloom)
   composite writes the swapchain directly.
12. **`UI`** — the packet's `ui_commands()` + 3D billboards, straight to
   swapchain.
   **`PostUICallback`** — optional host hook that renders into the swapchain
   after UI (`SceneRenderer::set_post_ui_callback`). There is a single slot:
   `Application::init_dev_ui()` claims it for the dev overlay's ImGui draw,
   so a host can't use both the overlay and its own post-UI callback.
13. Fallback: if the post stack isn't ready, steps 6–11 collapse to a single
   **`MainScene`** pass direct to swapchain (no depth prepass, no AO — every
   draw keeps its `GREATER_OR_EQUAL` write-on pipeline and samples the white
   dummy AO texture). Minimized windows skip all recording but still submit,
   keeping the frame lifecycle balanced.
14. **Screenshot readback** (only when `request_screenshot(path)` was called
   since the last frame; `Application` defers the request and forwards it to
   `SceneRenderer::request_screenshot` at the frame fence, so a request from
   sim frame i captures frame i's rendering): after UI/PostUICallback, a copy pass
   downloads the swapchain texture into a `DOWNLOAD` transfer buffer, the
   frame submits via `RenderContext::end_frame_and_wait()`
   (`SDL_SubmitGPUCommandBufferAndAcquireFence` + fence wait — a deliberate
   pipeline stall, so tests/debug only), and the mapped pixels are saved as
   PNG (`IMG_SavePNG`, alpha dropped via RGB24 conversion; BGRA/RGBA
   swapchain formats both handled). The only other GPU→CPU readback is
   `AutoExposure`'s diagnostic 16-byte fenced download of its state buffer
   (own command buffer, polled non-blocking, ~once per second — never a
   stall).

### Frustum culling

`contracts/frustum.hpp` extracts planes from `view_projection`. Culling runs on
the CPU for the main view and independently per shadow cascade during
instance-batch building. Toggle via `GraphicsSettings::frustum_culling`.

---

## GPU layer

- **Registry vs catalogue.** These are two files on purpose.
  **`gpu/pipeline_catalog.hpp`** is the catalogue: one `constexpr`
  `PipelineDesc` row per `PipelineType` (56) and one `ComputePipelineDesc` per
  `ComputePipelineType` (3), carrying the shader pair, the resource counts
  those shaders declare, the vertex layout (`VertexLayout`), the render target
  (`TargetFormat`) and the fixed-function state — cull, blend, the depth
  triple, depth clip, depth bias. `build_pipeline_config(desc, formats)` turns
  a row plus the frame's texture formats into a `PipelineConfig`; it is pure,
  so the whole catalogue is testable headless. A missing row is a **compile
  error** (`static_assert` on the table size), and the table is where the
  reversed-Z story is reviewable as one column: cascades stay standard-Z
  (`LESS_OR_EQUAL`), the `*DepthEqual` variants are `EQUAL` with writes off,
  `LocalShadowClear` is `ALWAYS`, everything else is `GREATER_OR_EQUAL`.
  `gpu/pipeline_catalog.test.cpp` diffs every row field-by-field against
  `gpu/pipeline_config_golden.inc` — a text golden captured from the
  hand-written creators that preceded the table — and additionally pins those
  depth rules. It is the only mechanical check in the tree that sees GPU
  *state*; the packet digest hashes what the host described, not what the GPU
  was told to do.
- **`PipelineRegistry`** (`gpu/pipeline_registry.hpp`) is the cache and
  lifecycle around that catalogue, and names no pipeline itself. It lazily
  creates and caches one `GPUPipeline` per `PipelineType` (including the
  model-family `DoubleSided`/`Blend` material variants, the 8 `DepthPrepass*`
  pipelines that write the RG16F velocity target, the 8 `*DepthEqual`
  forward variants used after the prepass, the
  `GTAO`/`GTAOTemporal`/`GTAOUpsample`/`BlurAO` AO pipelines, the
  `TaaResolve`/`TaaSharpen` post pipelines, the 4 IBL prefilter
  pipelines (`SkyCapture`/`IblSpecularPrefilter`/`IblIrradiance`/
  `IblBrdfLut`) that render into cubemap faces, and `LocalShadowClear` —
  the per-tile depth clear of the local-shadow atlas). Scene pipelines
  target the HDR format; UI/Text/Composite/Fxaa target the swapchain
  format and are the only ones invalidated when the swapchain format changes
  — which `set_swapchain_format` now *asks the catalogue*
  (`pipeline_targets_swapchain`) rather than reading from a hand-kept list.
  The registry also caches one `GPUComputePipeline` per `ComputePipelineType`
  (3: `ExposureHistogram`/`ExposureAdapt`/`WaterSim`) with the same
  lazy-create / preload / invalidate / hot-reload lifecycle.
  `preload_all_pipelines()` avoids first-use hitches. Lookups are by
  `get_pipeline(PipelineType)` / `get_compute_pipeline(ComputePipelineType)`;
  there are no per-type convenience accessors.
- **Compute** (`gpu/gpu_compute.hpp`): `GPUComputePipeline` compiles a
  `.comp.spv` module straight into an `SDL_GPUComputePipeline` via
  SDL_shadercross (there is no `SDL_GPUShader` for compute) and exposes the
  reflected threadgroup dimensions (`threadcount_x/y/z`, `group_count_*`
  ceil-div helpers). Creation REQUIRES SPIR-V reflection (the `[numthreads]`
  values only exist in the module); declared `ComputeResources` are
  cross-checked against reflection exactly like the graphics stages —
  read-only and read-write storage counts are distinct because SDL3 GPU puts
  them in different binding sets. `ComputePass` is the RAII pass helper:
  read-write storage bindings are declared at `begin()` (SDL3 GPU's model),
  read-only resources bind after the pipeline, `dispatch_for()` sizes the
  grid from the pipeline's threadgroup. Binding-order conventions are
  documented in [shaders/README.md](../src/bracken/shaders/README.md).
- **Uniform contract**: every HLSL cbuffer has an `alignas(16)` CPU mirror in
  `gpu/gpu_uniforms.hpp`, annotated with its `(set, binding)`. They must match
  **byte-for-byte** — this is the most common way to break rendering silently.
  Two nets enforce it. (1) Each struct is followed by a
  `BRACKEN_HLSL_CBUFFER` + `BRACKEN_HLSL_FIELD` block (`gpu/hlsl_packing.hpp`)
  pinning `sizeof`, `alignof`, every field offset, and the "no member ≤ 16 B
  straddles a 16-byte register" rule — so a CPU-side layout change is a
  compile error in the same translation unit. (2) `gpu/uniform_layout.test.cpp`
  parses the **compiled SPIR-V** in the staged shader directory (explicit
  Offset/ArrayStride/MatrixStride decorations — no HLSL packing rules are
  reimplemented) and compares each block's declared byte extent against
  `sizeof(T)`, plus each `StructuredBuffer` element stride against its CPU
  record. Its table is required to be exhaustive: a uniform block in any
  staged `.spv` that no row covers fails the test.
  `ClusterParams` is intentionally duplicated between
  `render/lighting/light_cluster.hpp` and `gpu_uniforms.hpp` to avoid a header
  dependency; keep both in sync.
- **Skinned palettes**: bone palettes are packed in the `FramePacket` at
  build time (`FrameBuilder::alloc_bone_palette` + prefix-summed
  `BonePalette::matrix_offset`); the renderer uploads `frame.palettes()` in
  order into a single per-frame `StructuredBuffer<float4x4>` (vertex set 0,
  binding 0) of `frame.palette_matrix_total()` matrices; the skinned vertex
  shader indexes it with a `paletteOffset` uniform. There is **no GPU-side
  bone-count constant**.
  The palette buffer is **double-buffered**: the previous frame's buffer
  stays bindable (vertex set 0, binding 1 in the skinned prepass shader) so
  the depth prepass can re-skin last frame's pose for per-bone motion
  vectors; `scene::MotionHistory` maps a command's `motion_id` to last
  frame's palette offset.
- **Debug tooling**: `BRACKEN_GPU_SCOPE` (SDL debug groups) labels every pass.
  `BRACKEN_GPU_TIMER` scopes report a REAL **CPU record** time (the cost of
  building the pass's command stream), measured by `core::PassTimeline` inside
  `GPUTimestampPool` — all 19 existing call sites, unchanged. The GPU half is
  still unavailable: SDL3 has no per-pass timestamp queries, so
  `is_supported()` is false and `PassTiming::gpu_ms` is `nullopt` rather than
  a zero that reads as "free" (`gpu/gpu_debug.hpp`). Use RenderDoc/Nsight for
  real GPU timing, and `Application::present_wait_ms()` to decide whether it
  is worth looking.

---

## Assets & models

- **`ModelLoader`** (`src/bracken/model_loader.hpp`): static
  `load_glb` / `upload_to_gpu` / `free_gpu_resources`. glTF binary only, no
  Draco (bundled tinygltf has no decoder). Model bounds are accumulated from
  every parsed vertex position — the POSITION accessor's declared min/max is
  not used. See [skinned_mesh_format.md](skinned_mesh_format.md) for
  the authoring contract.
- **`ModelManager`**: name → `ModelHandle` (`uint32_t`, `0` =
  `INVALID_MODEL_HANDLE`) registry. `load_model(name, path)`,
  `get_model(handle|name)`, `register_model(name, unique_ptr<Model>)` for
  procedural meshes — returns `core::Result<ModelHandle, ModelRegisterError>`
  and finishes the model (bounds from mesh vertices when unset, bounding
  sphere, `loaded`) — `reload_model(handle)`, `enable_hot_reload(watcher)`.
- **Mesh flags** (`model_loader.hpp`): `cast_shadows`, `shadow_only`,
  `has_packed_occlusion` (ORM-packed occlusion in the metallic-roughness
  texture). Standalone (non-ORM) occlusion textures are unsupported: the
  loader warns and drops them at load.
- **Material alpha/cull flags** (`PBRMaterial`): glTF `alphaMode` /
  `alphaCutoff` / `doubleSided` are honored by the renderer. `OPAQUE`
  materials never alpha-test (base-color alpha is ignored, per spec); `MASK`
  discards below the material's cutoff (`alphaCutoff` uniform); `BLEND`
  meshes are drawn in a sorted transparent pass after all opaque content and
  the skybox — back-to-front by camera distance, depth test on, depth writes
  off. `doubleSided` selects a no-cull pipeline variant. Limitations: no OIT,
  and the sort granularity is one mesh per submitted command/instance (using
  the model's bounding-sphere center), so intersecting or mutually enclosing
  transparent geometry can still sort wrong. `BLEND` meshes inside instanced
  batches fall back to one non-instanced draw per instance.
- **`model_utils`**: `build_model_transform` (translation/rotation/scale →
  mat4 with engine conventions), `compute_tangents_from_uvs` (for meshes
  missing tangents).

---

## Animation

Per-instance, host-driven. Types and constants in
`animation/animation_types.hpp`:

- `MAX_BONES = 128` (line 15) — CPU-side cap on joints per skinned mesh; sizes
  `std::array<glm::mat4, MAX_BONES>` in `AnimationPlayer` (the scene layer
  passes bone matrices as a length-carrying `std::span`).
- `AnimationPlayer` (`animation_player.hpp`): `update(skeleton, clips, dt)`,
  `crossfade_to(clip_index, duration)`, speed/loop/seek. Keyframe cursors are
  per-player (clips are shared, players are not — thread-hazard lesson, see
  design notes).
- `AnimationStateMachine` (`animation_state_machine.hpp`): data-driven states
  + transitions; the host feeds parameters (`set_float("speed", …)`,
  `set_bool("attacking", …)` — or, on per-frame paths, resolve the name once
  with `param_index()` and use the int-indexed `set_float`/`set_bool`
  overloads to skip the per-call string hash). Conditions are a tagged union
  supporting AND and `any_of` (OR). Clip binding is by **string name** → see
  the format doc's clip-naming section.
- **IK is applied by the host after the state-machine/player update**, before
  render submission: `solve_two_bone_ik`, `apply_foot_ik`, `apply_body_lean`
  (`ik_solver.hpp`). `FootIKBoneNames` defaults to the standard rig names
  `Hips`, `Spine`, `Left/RightUpperLeg`, `Left/RightLowerLeg`,
  `Left/RightFoot` — override per rig.
- glTF channels that target nodes outside the skin's joint list are
  dropped at load — counted per model and reported in one `ModelLoadReport`
  warning line; a clip that "doesn't play" usually means its
  channels don't drive skinned joints.

---

## Physics

`src/bracken/physics/` — rigid-body simulation over vendored **Jolt Physics
v5.2.0** (owner decision 2026-07-21, see the design-notes decision record).
Engine infrastructure, not a gameplay concept: the API is bodies, shapes,
impulses, and queries — nothing game-specific.

**API surface** (`physics_world.hpp`, the only public header — glm + POD
types over a pimpl; Jolt is linked `PRIVATE` and must never appear in a
public bracken header):

- `PhysicsWorld::init(PhysicsConfig, core::jobs::JobSystem&)` /
  `shutdown()` (RAII dtor). Config: gravity, `max_bodies` (4096),
  `max_body_pairs` (4096), `max_contact_constraints` (2048),
  `fixed_dt` (1/60), `max_steps_per_update` (4).
- Bodies: `add_sphere` / `add_box` / `add_capsule` (each takes a `BodyDesc`:
  Static/Dynamic/Kinematic, pose, mass, friction, restitution,
  `start_active`) and `add_heightfield` (static terrain from a row-major
  square height grid; Jolt's sample-count-multiple-of-block-size constraint
  is satisfied internally by duplicating the last row/column when needed).
  `BodyId` is the raw `JPH::BodyID` u32 (index + sequence number — stale ids
  never alias new bodies); `INVALID_BODY_ID` on failure.
- `update(dt)` — fixed-step accumulator: steps `fixed_dt` slices, at most
  `max_steps_per_update` per call; overflow debt is clamped away (sim slows
  down rather than spiraling) with a once-per-world warning.
- State/queries: `position`/`rotation`/`transform`, `set_position`,
  `linear_velocity`/`set_linear_velocity`, `apply_impulse[_at]`,
  `move_kinematic`, `is_active`, `cast_ray` (closest hit + surface normal),
  `collide_capsule` (upright-capsule overlap → per-body `ShapeContact`
  push-out contacts, deepest first; static-only by default with an
  `include_moving` widening and an `ignore_body` exclusion — the
  character-controller depenetration primitive).

**Threading tiers.** `PhysicsWorld` is **owned / sim-stage**: one logical
owner, `update()` designed to run inside the sim stage (it is a plain CPU
job — no GPU/SDL/main-thread coupling anywhere in the module). It is not
internally synchronized against concurrent host calls. Jolt's own global
registration (allocator, factory, type registry) is `std::call_once`-guarded
and process-wide.

**The job-system bridge** (`physics/internal/jolt_job_system.hpp` + `.cpp` — the one header in the tree that includes Jolt, which is why it lives under `internal/`; see "Public surface and `<module>/internal/`"):
`JoltJobSystem` subclasses `JPH::JobSystemWithBarrier` and mirrors
`JPH::JobSystemThreadPool`'s structure — pooled `Job` storage in a
`FixedSizeFreeList`, `QueueJob(s)` submitting a
`job->Execute(); job->Release();` thunk — but onto the engine's
`core::jobs::JobSystem` instead of its own threads: **one pool owns the
cores** (multithreading_plan.md §7 — no second pool, no oversubscription).
Thunks ride the zero-handle `TaskGroup` submit path; Jolt's
`WaitForJobs` executes ready barrier jobs on the waiting thread itself and
bracken waits help-execute, so `update()` makes progress from a sim job even
on a 1-worker pool. Two worlds stepping identical inputs produce
bit-identical results at any worker count (Jolt guarantee, pinned by
`physics_world.test.cpp`).

---

## Threading & lifetime contracts

Bracken is a **natively multithreaded engine** —
[docs/multithreading_plan.md](multithreading_plan.md) is the plan of record.
As of **Phase 4** the worker pool is always-on engine infrastructure, the
concurrency substrate (lock-free queues, `TaskGroup`, owner asserts,
pre-created GPU lookup tables, the tier table below) is live, all frame
data flows through the immutable `FramePacket` snapshot (see "Frame data
flow" above) — the old `RenderScene`/`UIScene` mutable-shared-state contract
is deleted — the renderer's Prepare stage is internally parallel (see
"Prepare-stage fan-out" below), and the **frame is pipelined**:
`Application::run()` kicks the host's `on_simulate` as a pool job for frame
i+1 while the main thread renders frame i (schedule under "`Application`
lifecycle"). There is deliberately
no "serial mode": debugging/bisection is `BRACKEN_WORKERS=1`, which runs the
same code on one worker.

**The frame fence** (schedule step 2) is the pipeline's quiescent point —
the sim job for the next frame has been joined and the next one is not yet
kicked, so *nothing* runs concurrently with the main thread. Everything
that must observe a quiescent engine happens exactly there: hot-reload
apply (`FileWatcher::poll_main_thread` — epoch reclamation by construction,
no reader locks), deferred graphics-settings application
(`request_graphics_settings` — two single-writer slots, sim-phase and
main-thread/DevUI, main wins), camera-forward feedback into the
`InputHandler`, FPS accounting, the `on_main_frame` host hook (the only
safe place for the main thread to read live game state — e.g. snapshots
for DevUI panels, which draw later during rendering while the next sim
runs), and deferred screenshot forwarding.

Main-thread contracts are asserted, not just documented:
`core::bind_main_thread()` runs in `Application::init_engine`, and
`BRACKEN_ASSERT_MAIN_THREAD()` (`core/assert.hpp`) guards the main-thread
boundaries in debug builds.

### Thread-safety tiers

Every engine API belongs to a tier; violating a tier trips a debug assert,
not a heisenbug in ship. Current (Phase 4) assignments:

**How each tier is actually enforced** — the table below claims enforcement,
so here is the mechanism for each claim:

- `main-thread`: `BRACKEN_ASSERT_MAIN_THREAD()` against the thread recorded by
  `core::bind_main_thread()` at boot. Debug builds only.
- `sim-stage` / *not* `sim-stage`: `BRACKEN_ASSERT_SIM_STAGE()` and
  `BRACKEN_ASSERT_NOT_SIM_STAGE()` (`core/jobs/job_stage.hpp`). The stage is
  **carried on the job node**, not in a thread-local depth counter, and the
  executor installs it around the invoke: `JobNode::stage` is inherited from
  the submitting context in `JobSystem::acquire_node_`, and
  `Application::run_sim_job` sets the root. This is what makes the check
  correct under work stealing (`parallel_for` chunk bodies run on other
  threads), under nested fork-join, and — the case a main-thread assert cannot
  cover at all — when `JobSystem::wait`'s helping executes a **sim-stage job
  on the main thread** during `join_sim`. Debug builds only.
  `BRACKEN_ASSERT_NOT_SIM_STAGE()` is applied today to `GPUDevice`
  create/destroy + swapchain acquire, `PipelineRegistry` creation/reload,
  `ModelManager` mutation, and every `DevUI` entry point.
- `worker-safe`: nothing to assert — the tier is the absence of a restriction.
- `owned`: `Arena`/`FrameAllocator` bind a first-toucher owner thread
  (`Arena::assert_owner`, re-openable via `reset()`/`rebind_owner()`).
  `FrameBuilder::Lane` deliberately does **not** bind an owner: a lane
  legitimately changes threads whenever a fan-out is joined and the root
  continues emitting (Application's own sim tail does exactly that, and the
  join is what makes it safe). What it detects instead is genuine **overlap**
  — two threads inside the lane's chain mutation at once, which is the case
  that corrupts the chunk chain. Debug builds only.
- **Host-boundary preconditions** are a separate mechanism and are **live in
  release**: `BRACKEN_CONTRACT` (`core/assert.hpp`,
  `-DBRACKEN_ENABLE_CONTRACTS=OFF` to remove). It guards what a *game* does to
  the engine — reading a `FramePacket` before seal, sealing twice, emitting
  after seal, an out-of-range lane index, and every `Application` rendering-
  facade member called before `init_renderer()` (those dereferenced
  `scene_renderer_`/`context_` unconditionally and segfaulted). Engine-internal
  invariants and per-item hot-loop checks stay on `BRACKEN_ASSERT` and remain
  debug-only.

| Tier | Meaning | Members today |
|---|---|---|
| `main-thread` | asserted via `BRACKEN_ASSERT_MAIN_THREAD()`; `on_main_frame`/`on_event` and the fence run here with no sim in flight | swapchain acquire, `render_frame`, command submission, ImGui/DevUI, GPU resource creation/destruction (pipeline/shader/model *mutation* paths), `FileWatcher` entry points, the live `InputHandler`, graphics-settings *application* (`apply_all_graphics_settings`/`set_*`) |
| `sim-stage` | any job within the Simulate stage — `on_simulate` and everything it fans out to; **live**: this tier now runs concurrently with frame i's rendering | `SimFrame` (its `parallel_for`/`run_async`/`set_camera_forward`), `FrameBuilder` emission (multi-lane: `FramePacket::reserve_block` / `take_palette_slot` are safe from any fill-stage thread; singular setters like `set_camera` are single-writer), `AnimationPlayer`/IK as host sim work, `EffectSystem` (sim-stage **owned**: engine-driven from the sim-job tail), `Application::camera()` under `run()`, `request_graphics_settings` / `edit_graphics_settings` (single writer per phase; do not hold an edit handle across a frame boundary), `request_screenshot` |
| `worker-safe` | callable from any thread, any time | `Logger`, `JobSystem` submit/wait, `TaskGroup`, `parallel_for`, `Application::quit()`, pipeline/shader/model **lookups** (read-only after init-time pre-creation), `get_terrain_height` (const reads of the immutable heightmap), **`FramePacket` reads after seal** (plain loads, no atomics on the read path) |
| `owned` | single owner at a time; debug-asserted (owner binding for arenas, overlap detection for lanes — see above) | `Arena`, `FrameAllocator`, per-worker scratch arenas, a `FrameBuilder::Lane` (one thread at a time), a `FramePacket` during fill (per-lane) |

Facade members safe to *read* from sim code because they are only written
at the fence / event pump (never while a sim is in flight): `fps()`,
`screen_width()`/`screen_height()`. Publish protocol: the packet's
`Filling → Sealed` release store in `FrameBuilder::seal()` is THE payload
edge (non-atomic content + one atomic publish, house style); under `run()`
the cross-stage handoff rides the sim TaskHandle kick/join edges (the
`PacketRing` release-publish / acquire-take slot word serves ring-driving
hosts that bypass `run()`). Every atomic involved documents its ordering
and pairing partner at its declaration.

Known deliberate tears, documented at their declarations: the DevUI Camera
tab reads the live `CameraController` during render while the next sim may
be updating it (tear-only reads of floats — display-quality, never
consumed by logic); the DevUI Graphics tab edits the host-owned
`graphics_settings()` struct on the main thread (its *apply* is routed
through `request_graphics_settings` to the fence, but the struct itself is
a documented single-writer-per-phase host contract, not a fenced one).

`get_pipeline` and its siblings are read-only lookups over tables
**pre-created at init** (the registry knows the full pipeline set); creation
and hot-reload paths are main-thread-asserted. This deletes the old
first-touch lazy-creation race instead of locking it. Hot-reload swaps apply
only at the main-loop fence via `poll_main_thread()`, when no worker is
reading the tables — epoch reclamation by construction, no reader locks.

### Prepare-stage fan-out (Phase 3)

The Prepare work inside `SceneRenderer::render_frame` — everything
view-dependent computed on the CPU from the sealed packet — is jobified
(plan §4.3).

**The tier boundary is a type: `scene::ScenePrep`** (`scene/scene_prep.hpp`).
The renderer's three prep cores — `prep_instance_batches`,
`prep_resolve_skinned`, `prep_copy_palettes` — plus their params, scratch
pools and outputs live on it, and `ScenePrep` has **no `GPUDevice`, no
`RenderContext`, no sub-renderer pointer and no state at all** (it is not
even instantiable). That is the enforcement: a prep core structurally cannot
reach a main-thread-tier API, so the contract is checked by the compiler
instead of by a reviewer reading a comment. Its inputs are exactly what is
worker-safe on a sealed frame — const `ModelManager` lookups, const
`MotionHistory` lookups, spans over packet memory, by-value frame state.
**`ScenePrep` must never gain a `RenderContext*`, a `GPUDevice*` or a
sub-renderer pointer "for convenience"**: that silently re-opens the tier.
Whatever needs a device is main-thread work and belongs on the other side of
the boundary, in `scene/scene_renderer_upload.cpp` — which is also why the
palette copy takes a plain `glm::mat4*` and the mapping/upload around it
stays on `SceneRenderer`. The renderer keeps three thin adapters
(`build_instance_batches`, `pack_bone_palettes`, `upload_bone_palettes`)
that gather its own members and call across.

The loops that fan out:

- **per-view culling + instance batching**: main view and the active CSM
  cascades — disjoint outputs (per-cascade shadow bins), one job per cascade
  with the main view on the caller, chunked within each view;
- **skinned-command resolve + palette layout**;
- **cluster light binning** (per-cluster form — each cluster writes only its
  own bin; the memoized cluster-bounds rebuild chunks trivially);
- **grass chunk cull** and **terrain tile cull** (the run-merge consumes
  chunk results in order).

Particle work moved to the **sim stage** in Phase 4: spawns are applied
serially in deterministic (lane, emission) order in the sim-job tail
(order-defining for the RNG stream), `EffectSystem::update` integrates as a
per-particle `parallel_for` (compaction per-emitter) on the pool from
inside the sim job, and the renderer only uploads/draws the packet's
effect-instance snapshot.

Explicitly *not* jobbed: CSM cascade fit (≤ 4 tiny fits), the local-shadow
tile-caster cull (serial — interleaved with GPU tile recording in
`render_local_shadow_passes`; it moves with parallel recording in plan
Phase 5), water (GPU compute), the
GPU-side post chain (IBL/TAA/GTAO/bloom/fog/exposure), DevUI.

The discipline, uniform across every fan-out (this is the determinism
contract, §6.2 of the plan):

- Chunked dispatch via `parallel_for(JobSystem&, begin, end, fn, grain)` or
  `TaskGroup` jobs; **each chunk accumulates into its own partial** (instance
  bins, stats, scratch); partials merge **single-threaded in chunk-index
  order** after the join — bit-identical to the serial order at any worker
  count.
- **No atomics and no shared cache-line writes inside per-item loops**; no
  mutexes anywhere in the frame path. `render_stats_` follows suit: per-chunk
  `FrameStatsPartial`s, merged at join.
- **Grain guards** keep small scenes inline — same code path, one chunk; the
  1-character playground scene runs effectively serial.
- Engine-internal stage helpers take a **`JobSystem&`** parameter (call
  sites pass `JobSystem::instance()`), so parity tests can construct
  caller-owned pools (`init(1)` vs `init(4)`) and assert bit-identical
  outputs per stage.

The renderer's prep cores are covered by two test files with deliberately
different jobs, and both are needed:

- `scene/scene_prep_parity.test.cpp` — **determinism**: a large synthetic
  packet prepared at 0/1/4 workers, compared byte for byte. It proves the
  outputs do not depend on the worker count. It does not, and cannot, say
  what the outputs should be.
- `scene/scene_renderer_prep.test.cpp` — **correctness**: a tiny hand-built
  scene (camera at the origin, ortho frustums, axis-aligned bounds,
  pure-translation transforms) whose every expectation is a number derived
  from the rules in the source. It pins the cull counts, bin membership and
  order, the `InstanceData` bytes, the shadow cast rules (including the two
  that look like redundancy: a command with **no bounds** casts into every
  cascade, and a main-frustum-culled command still casts if the light
  frustum sees it), the skinned drop counters and palette offsets, the packed
  palette stream, and the per-batch `base_instance` that the instance-buffer
  packing loop and each of the three instanced draw walks must independently
  agree on. That last one is asserted with **two model types and an
  unresolvable handle present at once** — the only configuration in which a
  packing/draw-walk desync changes a number rather than nothing.
- `scene/instance_batch_walk.test.cpp` — the **draw side** of that same
  agreement: `SceneRenderer::for_each_instance_batch` is the one definition
  of the packing order (ascending `ModelHandle`, running `base_instance`),
  used by the packer and all three instanced draw walks now that they live in
  different translation units. The test pins the visited (handle,
  `base_instance`) pairs by value, including a middle bin that does not draw
  — the case where a bin was packed but its handle no longer resolves, so the
  walk must skip the draw and still advance past its range.

### JobSystem internals (Phase 1 rewrite)

`core/jobs/` (`job_system.hpp`, `task_handle.hpp`, `parallel_for.hpp`) is the
always-on frame engine, rewritten from the original mutex-deque pool to the
lock-free design of plan §7:

- **Per-worker Chase–Lev work-stealing deques** (the C++11-memory-model
  formulation, Lê/Pop/Cohen/Nardelli 2013): the owning worker pushes/pops
  the bottom with relaxed/acquire-release ops; thieves CAS the top. The only
  contended operation is a steal — by design the rare path of a well-grained
  workload.
- **MPSC injection queue** for external (non-worker) submits, replacing the
  mutexed global queue.
- **Pooled job nodes**: fixed-size nodes in per-worker pools recycled via
  freelist; handles carry a generation counter for done/error queries — no
  per-job heap allocation, no `shared_ptr` refcount churn, ABA-safe.
- **Two-phase parking**: idle workers spin briefly, then park on C++20
  `std::atomic::wait`/`notify_one` (futex/ulock). No CV+mutex pairing, no
  periodic idle poll — a parked pool costs ~zero. `notify_one` per submitted
  job; `notify_all` at shutdown **and once per batch dispatch**
  (`JobSystem::submit_batch` — see below).
- **`TaskGroup`**: one cache-line-padded atomic counter (padded to the
  destructive interference size — 128 B on Apple Silicon). Release
  `fetch_sub` on job completion; the zero-observer acquires and notifies;
  waiters park on `atomic::wait`. Submit-against-group + `wait(group)` is the
  stage-orchestration primitive; there is no dependency-graph scheduler.
- **`parallel_for(JobSystem&, begin, end, fn, grain)`** overload alongside
  the singleton-servicing form; grain guards stay (small counts run inline on
  the caller — granularity control, not a second code path).
- **Memory-model discipline**: every atomic documents its protocol (what it
  publishes, its ordering, its release/acquire pairing partner) at its
  declaration; bare `seq_cst` in a hot path requires a justifying comment.
  Race-shaped stress tests per primitive plus the TSan lane (`cmake --preset
  tsan`) guard the protocols.

**Wait semantics are unchanged**: `wait(handle)` is a helping wait (drains
own deque, injection queue, then steals; parks when there is nothing to
help with), fork-join safe on any pool size including one worker; task
exceptions are captured and rethrown by `wait`.

### Always-on lifecycle

`Application::init_engine` initializes the pool unconditionally, **before**
SDL and every other subsystem (so any of them may submit jobs). The worker
count comes from the **`BRACKEN_WORKERS`** env var, read once at boot:
unset or `0` → `hardware_concurrency() − 1`, clamped ≥ 1; an explicit
positive value is used as-is (`BRACKEN_WORKERS=1` is the single-worker debug
mode). Negative/garbage values warn and fall back to the default. Shutdown is
first-in-teardown: the pool is joined at the top of `shutdown_renderer()`,
before any GPU teardown, so no job outlives the subsystems it might touch
(`shutdown_engine()` repeats the call as an idempotent safety net for hosts
that never brought up a renderer).

### Lifetime contracts

- **`FileWatcher`** (`core/asset/file_watcher.hpp`): polls on a background
  `std::jthread`, but callbacks fire only from `poll_main_thread()` on the
  main loop — watcher callbacks may therefore safely touch GPU state.
- **Frame data carries host-lifetime contracts only for the two named effect
  definition pointers** (Phase 2, corrected in Phase 8): everything else a
  `FramePacket` holds is by value or packet-allocated, the packet outlives
  its own consumption by construction, and `render_frame` takes it `const`.
  The old bone-matrix-span and spawn-drain contracts are deleted. What
  remains: a `FrameBuilder` is spent after `seal()` (do not use it after the
  packet resets), and `PacketRing::retire` must happen-before the fill side's
  next acquire of that slot (under `run()` this is the loop order — retire(i)
  precedes kick(i+2)'s acquire on the main thread; ring-driving hosts
  provide their own fence).
- **`EffectDefinition`s** referenced by `spawn_effect`, and the
  `EmitterDefinition`s reached from `EffectEmitterSnapshot`, are long-lived
  host asset data and must outlive their spawned instances. These are the
  packet's only host-lifetime exceptions and are stated as such at the top of
  `scene/frame_packet.hpp`; null is a `BRACKEN_CONTRACT` at emit and consume.
- **`Logger`** sinks are thread-safe. It is also the engine's *only* log
  channel — no engine source calls `SDL_Log`/`std::cerr`/`std::cout`
  (enforced by `core/logger_discipline.test.cpp`; the exemptions are
  `core/logger.cpp`, which is the SDL bridge, and `core/assert.hpp`, which
  must still print on the abort path). Each `.cpp` carries a file-local
  `LOG_TAG` category. The mutex guards the sink list; `should_log` is a
  lock-free relaxed load of an atomic level.
- GPU resource owners (`PipelineRegistry`, buffers, textures) are
  non-copyable/non-movable and must be destroyed before the `GPUDevice`.

### Determinism

One code path at every worker count: `BRACKEN_WORKERS=1` and `=N` must be
bit-identical. Merges of parallel partials are ordered by chunk index, never
thread id; float-reassociating reductions are forbidden in merges unless a
test explicitly accepts an epsilon (plan §6.2). Render-stats accumulation
follows the same pattern: per-chunk `FrameStatsPartial`s merged
single-threaded at join — an atomic counter increment inside a per-item loop
is a review defect, as is a mutex anywhere in the frame hot path (plan §7).

---

## Load-bearing constants

| Constant | Value | Where | Caps |
|---|---|---|---|
| `MAX_BONES` | 128 | `animation/animation_types.hpp:15` | joints per skinned mesh; CPU palette arrays |
| `MAX_BONE_PALETTES` | 1024 | `scene/frame_packet.hpp` | bone palettes per frame — enforced at **build time** in `FrameBuilder::alloc_bone_palette` (over-budget: error log + invalid id; draws with an invalid id are dropped with a warning) |
| `CSM_MAX_CASCADES` | 4 | `render/shadow_map.hpp` | cascades in the 2×2 D32 shadow atlas |
| `CLUSTER_DIM_X/Y/Z` | 16×9×24 | `render/lighting/light.hpp` | cluster grid resolution |
| `MAX_LIGHTS` | 256 | `render/lighting/light.hpp` | dynamic point+spot lights per frame |
| `MAX_LIGHTS_PER_CLUSTER` | 128 | `render/lighting/light.hpp` | lights binned into one cluster |
| `LOCAL_SHADOW_TILES_PER_ROW` | 8 (→ 64 tiles) | `render/lighting/light.hpp` | local-shadow atlas grid; point budget ≤ 8 (6 tiles each), spot budget ≤ 16 |
| local-shadow tile update cap | 6/frame | `LocalShadowConfig` (`render/lighting/local_shadows.hpp`) | tiles re-rendered per frame; excess degrades to stale tiles |
| `EXPOSURE_HISTOGRAM_BINS` | 256 | `render/auto_exposure.hpp` | auto-exposure histogram bins (fixed by the shaders' LDS/`[numthreads]`) |
| `EXPOSURE_MIN/MAX_LOG2_LUMA` | −10 / +16 | `render/auto_exposure.hpp` | metered luminance range (EV); bin 0 collects everything below |
| default world size | 32000 × 32000 | `application.hpp::init_renderer` | terrain/grass extent defaults |
| swapchain / scene / depth formats | RGBA8 / RGBA16F / D32 | `gpu/pipeline_registry.hpp` | see GPU layer above |

`shadowEnabled` in `ShadowDataUniforms` is a mode, not a bool: `0` off,
`1` PCF, `2` PCSS (`gpu/gpu_uniforms.hpp`).

---

## Known stubs & silent behaviors

Things that look functional but aren't, or fail quietly — check here before
debugging "why is this value zero / where did my draw go":

- **GPU-side per-pass timings do not exist** — SDL3 has no timestamp query
  API, so `PassTiming::gpu_ms` is `nullopt` (not zero). `PassTiming::cpu_ms`
  IS real: the CPU cost of recording that pass. Do not read it as GPU cost.
- **`Application::render_stats()` returns `nullopt` while collection is off**
  (`set_collect_render_stats`, default off) rather than an all-zero struct
  that is indistinguishable from an empty frame.
- **Bone-palette allocations beyond `MAX_BONE_PALETTES` (1024) fail at build
  time** — `FrameBuilder::alloc_bone_palette` counts the refusal in
  `FramePacket::build_diagnostics()` (summarized once per frame at `seal()`,
  surfaced as `FrameDiagnostics::palettes_over_budget`) and returns an
  invalid allocation (empty span); `draw_skinned` with an invalid handle or
  id drops the draw and counts it. Not asserted.
- **Animation channels targeting non-skin nodes are dropped** (counted and
  logged once per model as a `ModelLoadReport` summary) at glb
  load.
- **`BRACKEN_ASSERT` conditions are never evaluated in release builds** —
  under `NDEBUG` the condition is type-checked via `sizeof` but not executed
  (`core/assert.hpp`), so asserted expressions must be side-effect-free. Use
  `BRACKEN_VERIFY` when the check must always run.
- **Transparency has no OIT** — glTF `BLEND` meshes are sorted back-to-front
  per queued draw (one entry per mesh per command/instance, keyed on the
  model's bounding-sphere center); overlapping transparent geometry can still
  blend in the wrong order. Shadow passes still alpha-test `BLEND` casters at
  0.5 rather than rendering translucent shadows.
- **The IBL cubemaps contain the procedural sky only** — no scene geometry
  (no reflection probes/SSR): metals mirror the open sky even indoors, and
  the reflected clouds are frozen at the last chain refresh (cloud scroll
  deliberately doesn't re-render the chain; see `render/ibl.hpp`).
- **Screen-space AO ignores MASK/BLEND geometry** — GTAO is computed from the
  opaque-only prepass depth, so alpha-tested foliage neither occludes nor
  receives its own AO (MASK pixels sample the AO of whatever opaque surface
  is behind them; BLEND draws bind the white dummy and get none).
- **Cluster light SSBOs keep stale contents** when a frame has zero dynamic
  lights — correctness is preserved by zeroed `ClusterParams`
  (`gridDim.w == 0` short-circuits the shader path), not by clearing buffers.
- **The bloom threshold excludes the adapted auto-exposure EV** — the bloom
  prefilter thresholds luminance × the MANUAL exposure bias only (the
  adapted EV lives in a GPU buffer the bloom pass doesn't read), so during
  adaptation the set of blooming pixels tracks the raw scene, not the
  displayed brightness. Stable, mildly "wrong"; wire the exposure SSBO into
  `bloom_downsample.frag` if it ever matters.
- **Auto-exposure metering is percentile-band, not center-weighted** — in
  open outdoor scenes the surviving 60–95% band is dominated by sky pixels,
  so the adapted EV tracks sky brightness far more than ground light
  (dimming the sun barely moves it; darkening the whole sky does). That is
  the intended behavior for day/night cycles, but hosts wanting
  subject-locked exposure should tune `set_percentiles` /
  `set_log2_target_gray` (`render/auto_exposure.hpp`).
- **Local light shadows can be one-or-more frames stale**: the per-frame tile
  cap (6) means moving lights beyond the budget re-render round-robin, and a
  cached tile is only invalidated by light changes or *dynamic* casters
  (nonzero `motion_id` / skinned) — a **static caster appearing or vanishing
  inside a cached light does not dirty the tile** until the light moves.
  Terrain and grass never cast local shadows. Emissive "bulb" geometry
  enclosing its own light must set `mesh.cast_shadows = false` or it
  occludes the light entirely (the shadow map renders from inside it). A
  light whose tiles haven't all rendered yet (fresh assignment over the cap)
  falls back to **unshadowed**, never to garbage.
- **`render_graph/` is not in the frame path** — `SceneRenderer` issues passes
  directly. The graph is kept (with its tests) for when pass count or
  transient aliasing justifies it.
