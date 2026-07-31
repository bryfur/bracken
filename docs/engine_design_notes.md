# Engine Design Notes & Rationale

Distilled from the `mmo-4` engine development sessions that produced this code,
trimmed to what's relevant to the **standalone engine**. The server, networking,
and gameplay material from the original retrospective has been dropped — only
the engine lessons remain. Read this before adding any cross-cutting
infrastructure to the engine.

---

## Rendering / correctness foundations (kept)

These are net wins, validated visually or by passing tests, and define the
engine's current rendering character:

- **Linear-space PBR pipeline**: GGX/Smith/Schlick/Burley BRDF, sRGB→linear
  sample, ACES tonemap output. See `shaders/pbr.hlsli`.
- **HDR scene buffer** (`R16G16B16A16_FLOAT`) so bloom/fog have headroom before
  tonemap.
- **Tangent buffers + tangent-space normal mapping** end-to-end (vertex format →
  vertex shader → fragment shader → material slot).
- **Tunable graphics settings**: exposure, tonemap mode, contrast, saturation,
  ambient strength, sun intensity, bloom threshold, AO strength/radius, fog
  density (`graphics_settings.hpp`).
- **Clustered-forward lighting** (16×9×24 grid, point + spot up to 256,
  Frostbite falloff) in `render/lighting/`. Cluster bounds are memoized
  (recomputed only when the projection changes).
- **GPU debug markers** (`SDL_PushGPUDebugGroup`) on every render pass, so
  RenderDoc/Nsight captures are readable.
- **Cascaded shadow maps** (`CSM_MAX_CASCADES = 4`) with per-cascade bias.

### Animation correctness
- Per-clip cursors (`pos_cursor`/`rot_cursor`/`scale_cursor`) live on the
  `AnimationPlayer`, not on the shared `AnimationChannel` — shared clips were a
  thread hazard otherwise.
- `MAX_BONES` is 128, a CPU-side cap in `animation_types.hpp`. The skinned
  shaders read a dynamic per-frame `StructuredBuffer` palette (indexed by
  `paletteOffset`), so they carry no bone-count constant to keep in sync.
- FootIK bone names are parameterized (no hardcoded Mixamo strings).
- Transition conditions use a tagged union (not `std::variant`) and support OR
  (`any_of`) in addition to AND.

### Architecture: the engine is game-agnostic
Gameplay concepts were deliberately hoisted out of the engine:
- `InputHandler` is raw input only (mouse, keyboard, gamepad axes, key edges).
  Game-specific actions belong in the host.
- `CameraController` exposes only generic shake/focus/fov-bias primitives.
- Core utilities live in `core/`: `logger` (default sink writes through SDL's
  logging, thread-safe sinks, `std::format`), `assert`, `result`, `handle`,
  `hash` (`StringId`, fnv1a). The logger is the engine's *single* log channel:
  engine code never calls `SDL_Log`/`std::cerr` directly, because anything
  that does is invisible to a host `LogSink` — which is how the entire GPU and
  renderer boot-failure stream used to escape hosts unnoticed.

Keep it this way. No combat/inventory/skill/ECS types in engine headers.
(Physics is no longer on that list — it became engine infrastructure by the
2026-07-21 decision record below; the boundary it must respect is the same
one everything else respects: no *gameplay* concepts, and no backend types
in public headers.)

---

## Decision record — 2026-07-21: rigid-body physics as engine infrastructure

**Decision.** Bracken ships a physics module: `src/bracken/physics/` —
`bracken::physics::PhysicsWorld`, a glm-only pimpl over vendored **Jolt
Physics v5.2.0**. This revises the original extraction-era scope ("no
physics"), by owner decision. The scope line moves, the principle doesn't:
physics is *engine* infrastructure (bodies, shapes, impulses, queries — the
same altitude as rendering and animation), while gameplay concepts (combat,
inventory, ECS) stay out. Jolt is linked `PRIVATE`; no `JPH::` type appears
in a public bracken header, so the backend remains a swappable
implementation detail.

**Why a job bridge instead of Jolt's thread pool.** Jolt ships
`JobSystemThreadPool`, which would spawn its own N workers — exactly the
oversubscription the multithreading plan forbids (§7: **one pool owns the
cores**; the risk table lists "pool + FileWatcher + driver threads" as the
trap). So `physics/jolt_job_system.{hpp,cpp}` implements
`JPH::JobSystemWithBarrier` over `core::jobs::JobSystem`: Jolt's step jobs
become bracken jobs (pooled `FixedSizeFreeList` records,
`Execute()`/`Release()` thunks matching the thread pool's reference
discipline). This composes with the pipelined frame for free — a
`PhysicsWorld::update()` inside the sim stage fans its Jolt jobs onto the
same workers the sim already runs on, bracken's helping waits keep nested
fork-join deadlock-free, and Jolt's own barrier wait executes ready jobs on
the calling thread, so even a 1-worker pool makes progress. No second pool,
no new parking/wake protocol, no new lock-free structure to get subtly
wrong.

**Boundaries that hold.** `PhysicsWorld` is tier owned/sim-stage (single
owner, no GPU/main-thread coupling); determinism is pinned by test
(bit-identical twin worlds after 120 steps, any worker count); the
fixed-step accumulator clamps overload debt instead of spiraling. The
module follows the boot-time `bool init()` + log error-handling convention.

---

## Decision record — 2026-07-31: no per-pass objects in the renderer

**Decision.** `SceneRenderer`'s implementation is split across six
translation units (see architecture.md) and the Prepare stage is extracted as
a type (`scene::ScenePrep`), but the passes themselves are **not** turned into
objects. There is no `DepthPrepassPass`, `ShadowCascadePass`, `LocalShadowPass`
or `ForwardScenePass`, and there will not be one without a new argument.

**Why.** The pass *sequence* is load-bearing, not incidental: `render_frame`
carries ~20 explicit, commented ordering decisions (uploads before any render
pass because copy passes cannot nest; IBL refresh before the forward pass so
same-frame draws sample the newest faces; water after the opaque scene but
before TAA; the late swapchain acquire; GTAO between the prepass and the
forward pass). That is the same reason `render_graph/` is deliberately not in
the frame path. The sequence therefore has to stay one linear script whatever
else changes, and pass objects buy nothing but 6–10 back-reference members
each while turning `main_render_pass_`, the prepass flag and the pipeline-
variant flags into cross-type invariants — strictly worse than the
entry-order dependency they would be replacing.

**What was done instead.** The two extractions that change a real property:
`RenderPlan` (the pass-selection tree becomes pure and testable) and
`ScenePrep` (the thread-tier boundary becomes a type that structurally cannot
violate itself). The TU split is navigability and parallel compilation, and
is honest about being nothing more.

**Named, deferred, not re-derivable from scratch each time.**
`FrameView`/`FramePassState` — collapsing the ~22 frame-scoped header members
into an immutable half and a mutable half — is the prerequisite for anything
further and is worth doing, but only after the above has held. The naive
"build one immutable view and pass it everywhere" version is *wrong*:
`frame_prepass_active_`, `frame_ao_texture_` and `frame_shadow_uniforms_` are
written mid-frame, after the camera state is final. `FrameGeometry`,
`ModelBindState` (four verbatim copies of the slot-1/2/6/7/10 bind sequence),
`PostChain` and `ScreenshotCapture`-as-a-type are worth doing only if a
measured need appears.

---

## Decision record — 2026-07-18: natively multithreaded engine

**Decision.** Bracken is rebuilt as a natively multithreaded engine — the
Frostbite model: an always-on worker pool, a frame pipelined through immutable
per-frame snapshots, one code path at every worker count.
[docs/multithreading_plan.md](multithreading_plan.md) is the plan of record;
Phase 1 (always-on pool with lock-free internals, `TaskGroup`,
owner-asserted arenas, pre-created pipeline/shader/model tables, the
thread-safety tier table, the TSan lane) has landed.

**What this supersedes.** The "default to serial / infrastructure without
load" guidance below, *as it applies to threading only*. That guidance was
written for an engine with no parallel workload and no mandate; the product
direction now is a modern-engine architecture, and "optionally parallel"
would mean maintaining two code paths forever — the one thing the plan
forbids. `Application::init_engine` now initializes the pool
unconditionally (`BRACKEN_WORKERS` env; `=1` is the single-worker debug
mode, same code, one worker), and the old objections were fixed rather than
argued with: idle workers no longer wake on a 50 ms poll to contend on a
shared mutex — they park on `std::atomic::wait` and cost ~zero when quiet.

**What still stands from the superseded lesson** (these sub-lessons govern
every phase of the migration):

- **Measure.** Each phase must win at scale and not regress the 1-character
  smoke scene; grains are sized by measurement, not vibes.
- **Grain guards.** `parallel_for` still runs tiny counts inline on the
  caller — that is granularity control inside the one code path, not a
  serial mode.
- **The lambda-capture container bug.** The historical never-diagnosed
  libstdc++ bounds assert (an entity container snapshotted by reference into
  worker lambdas) is why the end-state API takes caller-stable ranges and
  engine-owned builder lanes — the pattern that caused it becomes unwritable,
  not merely discouraged.
- The non-threading clauses — FileWatcher opt-in, FrameAllocator
  by-measurement, keep the engine game-agnostic, diagnose-don't-patch — are
  untouched. `render_graph/` stays unwired until Phase 5 wires it as the
  frame scheduler with a measured recording workload.

---

## The "infrastructure without load" trap

> **Superseded for threading (2026-07-18)** — see the decision record above.
> The job-system bullet below is retained as history; the pool is now
> always-on. The lesson still binds every *other* kind of infrastructure.

The single most important lesson from building this engine. Several subsystems
were added with the right *shape* for a production engine but no *workload* to
justify their per-frame cost. **Infrastructure has constant cost; workload has
variable cost. Add infrastructure when you have measured workload it absorbs,
not the other way around.**

Subsystems in this tree that this lesson applies to — they exist and are correct,
but should stay **off / serial / unwired** until a measured workload justifies
them:

- **`core/jobs/` JobSystem + `parallel_for`** *(superseded 2026-07-18 — see
  the decision record)*: the historical objection was that idle workers woke
  every 50 ms to look for steals and contend on a shared mutex — pure
  overhead with near-zero parallel work. The Phase-1 rewrite removed the
  overhead instead of the pool: lock-free deques, `atomic::wait` parking
  (~zero cost when quiet), always-on by product direction.
- **`core/asset/` FileWatcher**: a background thread doing recursive directory
  iteration + `stat()` on a poll interval is real DX value *only while actively
  iterating on shaders/models*. Gate it behind an opt-in flag; default OFF.
- **`render_graph/`**: a correct pass scheduler (topo-sort, transient texture
  pool with lifetime-based reuse), but **not wired into `scene_renderer.cpp`** — passes
  are issued directly. The graph's value materializes only when the pass count
  grows past hand-tractable size, or for transient aliasing / multi-camera.
  (TAA and GTAO both landed with hand-managed double-buffered histories and
  did not need it — history buffers are persistent, which is exactly what a
  transient pool doesn't manage.) Keep it in tree (the tests are valuable);
  don't route the renderer through it without a reason.
- **`memory/` FrameAllocator**: replacing capacity-stable `std::vector`s with it
  yielded the same performance and more bookkeeping. Use it where allocation
  churn is actually measured, not preemptively.

`parallel_for` is **not free** even with a serial fallback below threshold —
dispatch + wait + snapshot patterns add fixed overhead. For a handful of
entities, just iterate — the grain guards exist so that this stays true with
an always-on pool. (A historical bug: snapshotting an entity container by
reference into worker lambdas tripped a libstdc++ vector bounds assertion that
was never cleanly diagnosed. The multithreading plan's builder-lane API is
designed so this pattern cannot be written.)

---

## Decision principles

Re-read before adding any cross-cutting infrastructure:

1. **Infrastructure is added when there's a workload it absorbs.**
   "Future-proof" is a justification trap.
2. **Default to serial** *(superseded for threading, 2026-07-18 — see the
   decision record; the engine is natively multithreaded with one code path)*.
   The surviving kernel: parallelize a given loop only when measured
   wall-clock improves — grain guards keep tiny counts inline.
3. **Default to off.** Watchers, profilers, hot-reloaders — opt-in.
4. **Diagnose, don't patch.** If a fix introduces a regression, revert and
   understand. Don't stack patches.
5. **Visual + correctness first, perf-on-quiet-work second.** PBR is felt every
   frame; a job system saving 200µs on a 300-entity scene is felt nowhere.
6. **Defaults should match real workloads.** A grain size or poll interval that
   sounds reasonable in a vacuum can be 5× off for the actual scene.

---

## Forward-looking rendering work (when budget permits)

Real engine features that push toward "envy of engine developers" once the
foundation is stable. Items 1–3 **landed in the 2026-07 graphics feature
campaign** (see `docs/audits/RESOLUTIONS.md`, 2026-07-12 entry) and are kept
here for the record:

1. **IBL** — ✅ landed: `render::IBL` prefilters the procedural sky into
   irradiance + GGX-specular cubemaps and a split-sum BRDF LUT, amortized
   over frames on sky changes; `ambientIBL` in `shaders/pbr.hlsli` replaces
   the hemisphere ambient when `ibl_enabled` (which remains the fallback).
2. **Per-light shadow atlas** for point/spot lights — ✅ landed:
   `render::lighting::LocalShadowMap`/`LocalShadowSelector`, a budgeted
   cached-tile 8×8 D32 atlas, host opt-in per light via `casts_shadows`.
3. **TAA + motion vectors** — ✅ landed: depth prepass writes RG16F velocity;
   `render::TAA` resolves in HDR pre-bloom. Notably it did NOT need
   `render_graph/` — the history buffers are persistent and hand-managed.
4. **Compute-shader-driven cluster light culling** — still future: the
   CPU-side bin is fine for v1, but for >>100 lights, GPU-driven is the way
   (the compute foundation — `gpu_compute.hpp`, used by the histogram auto
   exposure — is now in place to build on).

### Tooling
- **Tracy-by-default in dev builds** — make `BRACKEN_ENABLE_TRACY=ON` the default for
  a `BRACKEN_DEV` build so there's a measurement surface before adding any infra.
- **Per-pass GPU timestamp telemetry** for a frame budget baseline.
- **CI**: run `bracken_tests` on every push.

---

*Adapted from the mmo-4 engine retrospective. Original last updated 2026-04-25.*
