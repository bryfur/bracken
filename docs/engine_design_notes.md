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
- Core utilities live in `core/`: `logger` (over `SDL_Log`, thread-safe sinks,
  `std::format`), `assert`, `result`, `handle`, `hash` (`StringId`, fnv1a).

Keep it this way. No combat/inventory/skill/ECS/physics types in engine headers.

---

## The "infrastructure without load" trap

The single most important lesson from building this engine. Several subsystems
were added with the right *shape* for a production engine but no *workload* to
justify their per-frame cost. **Infrastructure has constant cost; workload has
variable cost. Add infrastructure when you have measured workload it absorbs,
not the other way around.**

Subsystems in this tree that this lesson applies to — they exist and are correct,
but should stay **off / serial / unwired** until a measured workload justifies
them:

- **`core/jobs/` JobSystem + `parallel_for`**: idle workers still wake to look
  for steals and contend on a shared mutex. With near-zero parallel work this is
  pure overhead. Prefer lazy init (spawn workers only when a caller actually
  submits work above the parallel threshold). Default to serial; parallelize
  only skinning/culling/particles in the thousands.
- **`core/asset/` FileWatcher**: a background thread doing recursive directory
  iteration + `stat()` on a poll interval is real DX value *only while actively
  iterating on shaders/models*. Gate it behind an opt-in flag; default OFF.
- **`render_graph/`**: a correct pass scheduler (topo-sort, transient pool,
  debug-group integration), but **not wired into `scene_renderer.cpp`** — passes
  are issued directly. The graph's value materializes only when the pass count
  grows past hand-tractable size, or for TAA history / transient aliasing /
  multi-camera. Keep it in tree (the tests are valuable); don't route the
  renderer through it without a reason.
- **`memory/` FrameAllocator**: replacing capacity-stable `std::vector`s with it
  yielded the same performance and more bookkeeping. Use it where allocation
  churn is actually measured, not preemptively.

`parallel_for` is **not free** even with a serial fallback below threshold —
dispatch + wait + snapshot patterns add fixed overhead. For a handful of
entities, just iterate. (A historical bug: snapshotting an entity container by
reference into worker lambdas tripped a libstdc++ vector bounds assertion that
was never cleanly diagnosed. Default to serial.)

---

## Decision principles

Re-read before adding any cross-cutting infrastructure:

1. **Infrastructure is added when there's a workload it absorbs.**
   "Future-proof" is a justification trap.
2. **Default to serial.** Parallelize only when measured wall-clock improves.
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
foundation is stable:

1. **IBL** — pre-filtered cubemap + split-sum BRDF LUT. The PBR path already has
   the analytic stand-in to upgrade: `ambientPBR` in `shaders/pbr.hlsli`
   (hemisphere irradiance for diffuse, Karis `EnvBRDFApprox` for specular).
   Replace the hemisphere lerp with an irradiance-map sample and the Karis fit
   with prefiltered-cubemap + BRDF-LUT lookups.
2. **Per-light shadow atlas** for point/spot lights (only the directional sun
   casts today). Clustered forward+ is already in place, so this slots in.
3. **TAA + motion vectors** — the biggest remaining AA-quality gap. This is also
   the first concrete reason to wire up `render_graph/` (history buffers).
4. **Compute-shader-driven cluster light culling** — the CPU-side bin is fine for
   v1, but for >>100 lights, GPU-driven is the way.

### Tooling
- **Tracy-by-default in dev builds** — make `MMO_ENABLE_TRACY=ON` the default for
  a `MMO_DEV` build so there's a measurement surface before adding any infra.
- **Per-pass GPU timestamp telemetry** for a frame budget baseline.
- **CI**: run `mmo_engine_tests` on every push.

---

*Adapted from the mmo-4 engine retrospective. Original last updated 2026-04-25.*
