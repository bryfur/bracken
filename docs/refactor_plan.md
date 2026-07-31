# Bracken — Full-Scale Refactoring Plan of Record

Status: proposal, awaiting owner approval of scope.
Baseline verified 2026-07-30 on branch `refactor/full-scale` at `1b3fcc6`.

---

## 1. Thesis

Bracken is not structurally broken. `scene_renderer.cpp` is 3,696 lines of ~60 named
methods, the Prepare-stage cores are already device-free statics factored for testing
(`scene_renderer.hpp:345,:358,:372`), and the threading design — lock-free pool,
chunk-index-ordered merges, immutable packet — is genuinely good. Its real defect is
**silence and prose**: 180 `SDL_Log` calls and 37 `std::cerr` writes that no host `LogSink`
ever sees, lights dropped without a counter at `light_cluster.cpp:208-211`, draws dropped
on an unknown handle at `scene_renderer.cpp:1634-1637`, and every load-bearing contract —
settings encodings, draw flags, host-vs-engine API surface, packet lifetime, threading
tiers — living in a trailing comment or in a `BRACKEN_ASSERT` that compiles to
`((void)sizeof(!(cond)))` under `NDEBUG` (`core/assert.hpp:66-71`).

Two surveys arrived after the first draft of this document — a whole-tree dependency-graph
scan and a decomposition survey of the two largest files — and they change the picture on
exactly one axis, "logical components", which is the owner's first stated goal. Their net
effect is recorded here rather than hidden: (a) the module *graph* has a six-module
dependency cycle that nobody could see because there is no file-level include cycle, and it
is breakable by **three surgical edge cuts and one leaf directory**, not by a module
reshuffle — Phase 6b; (b) Phase 6's `scene_renderer.cpp` split as originally written is a
pure translation-unit move whose coupling reduction is *exactly zero*, so it has been
amended to carry one real type extraction (`ScenePrep`) that the surveys show is the
highest value-to-risk item in the file; (c) there is a **second** god object the first draft
missed entirely, `gpu/pipeline_registry.cpp`, 72 % of which is a hand-written catalogue that
duplicates ground truth already available from SPIR-V reflection — Phase 6c. Nothing else in
the surveys changed a conclusion; in particular they *strengthened* the rejection of path
moves, see §6.

So this plan is: **make the frame reproducible, make failures loud, turn prose contracts
into types, build the test net, and only then move code or optimize.** It is explicitly
*not* a rewrite, not a module reshuffle, not a rename pass, and not a performance project —
the honest CPU ceiling of every perf item here is a few hundred microseconds per frame, and
§7 asks the owner directly whether that is worth doing at all before real levers (LOD,
occlusion, GPU-side measurement) exist. The single most important thing in this document is
Phase 2: today the integration test is nondeterministic, so **no phase of any refactor
currently has a working correctness gate**, and everything downstream is sequenced behind
fixing that.

---

## 2. Preconditions

**Working tree.** The survey brief records the tree as carrying 261 modified files,
`+16268 -5519`, uncommitted. As of this writing that work appears to have been captured:
`git status --porcelain` returns empty on branch `refactor/full-scale`, HEAD is
`1b3fcc6 "WIP checkpoint: physics, water, TAA/GTAO/IBL, FramePacket pipeline"`. **Confirm
this before starting.** If any of that work is still uncommitted when you begin, commit or
branch it first — every phase gate below is a `git revert`/`git reset` story, and a dirty
tree destroys revert granularity for 7,000 lines of change.

Recommended: `git tag refactor-baseline 1b3fcc6` so every phase can diff against a fixed
point, and land each phase as its own commit (several phases specify sub-commits).

**Build trees.** `.claude/RULES.md` forbids reconfiguring over `build/`. This plan **never
touches `build/`**, and specifically does not give it a `CMAKE_BUILD_TYPE`: `build/` has no
build type today, which means `NDEBUG` is undefined, which means every `BRACKEN_ASSERT` and
`BRACKEN_ASSERT_MAIN_THREAD` is live. That is correct for the primary dev tree and must stay
that way. New coexisting trees (`build-rel/`, `build-bench/`) are added in Phase 1.

**Verified baseline facts** (re-measured, not taken on faith):

| Fact | Verification |
|---|---|
| `ctest --test-dir build` green | 1000/1000 |
| Smoke exits 0 | yes, driver=metal |
| **Smoke is nondeterministic** | run A: `draw_calls=542 triangles=6238831 mean_luma=89.1 stddev=44.2`; run B: `draw_calls=451 triangles=6172060 mean_luma=89.5 stddev=44.6` |
| Cause 1 | `application.cpp:113-118` derives `dt` from `SDL_GetTicksNS` |
| Cause 2 | `effect_system.cpp:21` seeds `thread_local std::mt19937` from `std::random_device` |
| Cause 3 | physics debt: both runs logged `update overflow: dropping 0.028s` |
| `SDL_Log` non-test sites | 180 |
| `std::cerr` non-test sites | 37 |
| `BRACKEN_LOG_*` non-test sites | 113 |
| `Result<` users outside `core/result.*` | **0** |
| `sizeof(gpu::InstanceData)` | 224 (64+64+16+4+12+64), **not** reducible to 144 — see §6 |
| Intra-tree includes | 554 total; 284 cross-module in production files |
| Production module cycles | **one SCC of six modules** `{@root, dev, render, render/lighting, scene, systems}`, 21 elementary cycles ≤ 4; headers-only SCC is 5 modules |
| File-level include cycles | **0** — which is why it compiles and why this was invisible |
| Layering violations (production) | **0**. `core/` fan-out is 0, a perfect sink. The only inversion is `gpu/gpu_uniforms.test.cpp:2 → render/environment.hpp` (a test) |
| `scene/frame_packet.hpp:5` includes `model_loader.hpp` | for `ModelHandle` + `INVALID_MODEL_HANDLE` and **nothing else** (`:66`, `:85`); `model_loader.hpp` is 249 lines and pulls `gpu/gpu_buffer.hpp`, `gpu/gpu_texture.hpp`, `gpu/gpu_types.hpp`, `animation/animation_types.hpp` |
| Pure-data leaf headers (zero `bracken/` includes) | `render/environment.hpp`, `render/lighting/light.hpp`, `scene/camera_state.hpp`, `scene/frustum.hpp`, `scene/velocity_math.hpp`, `graphics_settings.hpp`, `render_stats.hpp`, `heightmap.hpp`, `effect_definition.hpp` — glm/`<std>` only |
| `gpu/pipeline_registry.cpp` | 1,686 LOC; 38 creator functions, 76 hand-written `ShaderResources` counts, two hand-maintained 56-case switches, 4 dead accessors (verified 0 call sites) |

**Corrections to the survey that this plan bakes in** (each verified in source):

- `ModelManager` SoA bounds **cannot** use a `radius = -1` sentinel. A missing model today
  means *no bounds*, and no-bounds is not "cull it": `scene_renderer.cpp:1756-1764` leaves
  `main_visible = true`, and `scene_renderer.cpp:1843-1849` computes
  `casts = !have_bounds || light_frustum.intersects_sphere(...)`. The table must carry a
  validity bit and the loops must keep the three-way branch.
- `FrameBuilder::alloc_effect_snapshot` is called from `effect_system.cpp:331`, not from
  `Application`. Privatising it with only `friend class Application` does not compile.
- A `thread_local` sim-stage depth counter does not work: `parallel_for` dispatches chunk
  bodies onto worker threads (`parallel_for.hpp:65-67`), and `JobSystem::wait` runs stolen
  jobs on whatever thread is waiting. The stage must travel **with the job node**.
- `ClusterGrid::compute_cluster_bounds` is guarded at `light_cluster.cpp:200-206` by
  `!bounds_valid_ || projection_changed || near/far changed`. It is *not* a per-frame cost.
- `TerrainRenderer::draw_culled_tiles` culls into the shared member `tile_runs_scratch_`
  (`terrain_renderer.cpp:733`) and is called by the forward pass (`:581`), the prepass
  (`:789`) **and every shadow cascade** (`:835`). Caching into that member is a bug.

---

## 3. Phase table

| # | Phase | Goal | Est LOC | Risk | Breaks API? | Gate |
|---|---|---|---|---|---|---|
| 1 | Guard rails | `-Werror`, canonical includes, extra build trees | ~130 | low | no | build + ctest |
| 2 | **Determinism & gates** | Reproducible smoke; packet digest; config sweep | ~450 | med | no (adds flags) | 5× identical smoke |
| 3 | One log channel | 217 sites → `core::Logger`; lock-free `should_log` | ~500 | low | no | sink-capture test + digest |
| 4 | Loud failures | `FrameDiagnostics`, `BRACKEN_CONTRACT`, real stage timing | ~900 | med | yes (stats types) | new drop tests + digest |
| 5 | Test net | Goldens, GPU lifetime, uniform static_asserts + SPIR‑V reflection | ~1600 | low | no | net proven by injected breakage |
| 6 | `RenderPlan` + `ScenePrep` + TU split | Decompose `scene_renderer.cpp` | ~1150 | med | no | digest byte-identical, prep goldens, `nm -g` diff reviewed |
| **6b** | **Module graph is a DAG** | Cut 3 edges + one leaf `contracts/` layer; effects triad | ~350 | low | yes (paths) | header-cycle test + digest + `nm -g` identical |
| **6c** | **`PipelineRegistry` catalogue** | 38 hand-written creators → one descriptor table | **−1100 net** | low‑med | no | preload + sweep + per-pipeline config diff |
| 7 | Settings become types | 7 magic ints → scoped enums | ~280 | low | yes | on-disk round-trip + sweep |
| 8 | Draw API & `FrameSink` | Kill positional bools; hide engine tail; host test fixture | ~750 | med | yes | digest + sweep |
| 9 | Host lifecycle | boot, input, camera, settings-edit, heightmap, model bounds | ~600 | med | yes | digest + sweep |
| 10 | Perf, benchmark-gated | batched dispatch, SoA bounds, terrain double-cull | ~450 | med | no | per-item bench + goldens + TSan |
| 11 | Public surface & docs | umbrella, `internal/`, doc coherence (effects triad → 6b) | ~620 | low | yes (paths) | header test + docs test |

**Aggregate: ~8,000 LOC touched, of which Phase 6c is a net deletion of ~1,100.** Roughly
3,700 of the total is mechanical and compiler-verified (Phase 3's 217 log sites, Phase 7's
~104 enum call sites, Phase 6's TU moves, Phase 6b's ~55 include-line rewrites, Phase 11's
path rewrites). The remaining ~4,300 is semantic and needs review. Phases 1, 3, 5, 6b, 11
are low-risk; 6, 6c are low-to-medium; 2, 4, 8, 9, 10 are the ones that need care; nothing
here is high-risk *because* the two highest-risk items in the surveyed proposals (the
CPU↔HLSL uniform repacks) are rejected — see §6.

**Where "logical components" is served.** The goal is met by Phases 6, 6b and 6c and by
nothing else, and it is met through *shape*, not through paths: 6 turns the renderer's only
thread-tier boundary into a type that cannot violate it; 6b makes the module dependency
graph acyclic so the layering can be stated at all; 6c separates a registry from the
catalogue that is currently welded into it. Path changes appear only where they are the
mechanism for a graph change (6b's leaf layer, the effects triad) and are priced in the
phase. §6 records, with measurements, why the remaining path moves are still rejected.

Every phase leaves the tree building and `ctest` green. Every phase is one or more
self-contained commits and is revertible without touching its neighbours.

---

## 4. Phase detail

### Phase 1 — Guard rails

**Serves:** modernization, performance (measurability). Prerequisite for everything else:
a refactor that moves types across boundaries needs the compiler talking.

**Changes**

- New `cmake/BrackenWarnings.cmake` → `bracken_target_warnings(<t>)` applying
  `-Wall -Wextra -Wshadow -Wnon-virtual-dtor -Woverloaded-virtual -Wimplicit-fallthrough -Wnull-dereference`,
  plus `-Werror` behind `BRACKEN_WERROR` (default ON **only** when `PROJECT_IS_TOP_LEVEL`, so
  a consuming game's compiler cannot break their build). Applied PRIVATE to `bracken`
  (`src/bracken/CMakeLists.txt:106`), `bracken_tests` (`:201`), `bracken_benchmarks` (`:191`),
  `playground` (`examples/playground/CMakeLists.txt:11`).
- Fix the 11 measured sites: `application.hpp:164` (`[[maybe_unused]] event`),
  `procedural/rock_generator.cpp:438-440` (dead `dx/dy/dz`),
  `render/lighting/local_shadows.cpp:64,:76` (unused `const FACES`),
  `scene/scene_renderer.cpp:1410` (shadowed local), `systems/effect_system.cpp:173`
  (`particle_index`), `animation/animation_types.test.cpp:326-327`,
  `examples/playground/playground.cpp:376`.
- Delete `src/bracken/CMakeLists.txt:114` — `${SDL_shadercross_SOURCE_DIR}` is empty in that
  scope (FetchContent populates the lowercased name), so all 134 TUs carry a literal
  `-I/include`. The real headers already arrive via
  `SDL3_shadercross::SDL3_shadercross-static`'s INTERFACE dirs (`:123`).
- Normalize the 14 header-to-header relative includes to `bracken/`-rooted form:
  `gpu/gpu_compute.hpp:3`, `gpu/pipeline_registry.hpp:3-6`, `gpu/gpu_pipeline.hpp:3-4`,
  `gpu/gpu_shader.hpp:3`, `gpu/gpu_texture.hpp:3-4`, `gpu/gpu_buffer.hpp:3`,
  `animation/animation_state_machine.hpp:3`, `animation/animation_player.hpp:3`,
  `animation/ik_solver.hpp:3`. Five point at `gpu_device.hpp`, the highest-fanout header
  (71 of 139 TUs). Canonical spellings make every later move a path substitution.
- `CMakePresets.json`: add `rel` (`build-rel/`, RelWithDebInfo — the *only* place `NDEBUG`
  is exercised) and `bench` (`build-bench/`, RelWithDebInfo, benchmarks ON, examples OFF).
  Both coexist with `build/` and `build-tsan/`. `build/` is not touched.
- Root `CMakeLists.txt` (~:56): `message(WARNING)` when `BRACKEN_BUILD_BENCHMARKS=ON` and
  `CMAKE_BUILD_TYPE` is empty — the current state, in which every benchmark number in this
  repo was measured at `-O0` with arena poisoning and live tier asserts.
- Docs: `BRACKEN_WERROR` in the options tables (`docs/user_guide.md` ~:109,
  `docs/technical_spec.md:100-110`); the two new lanes next to the TSan lane in
  `CLAUDE.md`; drop the third entry from the quoted include list at
  `docs/technical_spec.md:93`.

**Deliberately not enabled:** `-Wsign-conversion` / `-Wconversion` / `-Wdouble-promotion`.
Measured: they take the count from 11 to 328, with 307 being numeric noise in five files
(`local_shadows.cpp` 74, `terrain_renderer.cpp` 49, `model_loader.cpp` 40,
`scene_renderer.cpp` 40, `ik_solver.cpp` 28). Bundling them turns an hour into a multi-day
audit and buries the 11 real findings.

**Gate**

```bash
cmake --build build                                  # clean with -Werror on all 4 targets
ctest --test-dir build --output-on-failure           # 1000/1000
grep -c -- ' -I/include ' build/compile_commands.json  # 0
cmake --preset rel && cmake --build build-rel && ctest --test-dir build-rel --output-on-failure
```

The `build-rel` run is load-bearing: it is the first time the `NDEBUG` path has ever been
compiled or tested in this repo.

**Rollback:** single commit, `git revert`. No behavior changes.

---

### Phase 2 — Determinism and gates (**highest-value phase**)

**Serves:** all four goals, indirectly — this is the phase that makes the other ten
verifiable. Without it every "compare the screenshot" gate in every proposal is a coin flip:
two consecutive runs of the current binary differ by 91 draw calls and 67k triangles.

**Changes**

1. **Fixed timestep.** `Application::set_fixed_timestep(std::optional<float>)`; `kick_sim`
   (`application.cpp:112-121`) uses it in place of the `SDL_GetTicksNS` delta when set.
   Smoke sets `1.0f/60.0f`.
2. **Deterministic effect RNG.** `effect_system.cpp:18-22`'s `thread_local mt19937` seeded
   from `random_device` is replaced by a counter-based hash RNG keyed on
   `(emitter_id, particle_index, frame_index)` — independent of wall clock *and* of which
   worker runs the sim job. This mirrors the bit-identical-at-any-worker-count discipline
   already documented for INTEGRATE at `effect_system.hpp:297-346`.
3. **Deterministic physics stepping** under fixed timestep: the accumulator consumes exactly
   `round(dt / fixed_step)` steps and the overflow-drop path (`max_steps_per_update = 4`,
   observed firing in both baseline runs) becomes unreachable at 1/60 s.
4. **Packet digest — the device-independent gate.** New
   `scene::packet_digest(const FramePacket&) -> uint64_t`: an FNV-1a over every sealed
   command in canonical order (model draws: handle + transform + tint + flags + motion id;
   skinned: + palette bytes; lights; billboards; debug prims; camera; environment; UI text).
   `Application` accumulates a rolling per-frame digest; `--smoke` prints
   `[smoke] packet_digest=0x...`. **This is the primary gate for Phases 3–10**: it is exact,
   has no GPU or driver dependency, and covers everything the host describes and everything
   the engine's sim tail appends. A swapped `DrawFlag`, a transposed light field, a dropped
   draw, or a changed camera all move it.
5. **Screenshot fingerprint.** `main.cpp:14-45`'s liveness check
   (`mean>2 && mean<250 && stddev>5`) is replaced by comparison against a checked-in
   `examples/playground/tools/smoke_golden.txt`: exact `draw_calls / triangles / entities /
   local_tiles / ibl_refreshes / packet_digest`, plus screenshot SHA-256 recorded as
   *advisory* (reported, non-fatal) with `mean_luma`/`stddev` banded ±0.5. `--smoke-write-golden`
   regenerates. Rationale for advisory SHA: driver/GPU nondeterminism in floating-point
   rasterization is outside our control; the digest is the hard gate and the SHA is the
   canary that tells a human to look at the PNG.
6. **Config sweep.** `--smoke-sweep` runs 60-frame headless-ish passes across the
   configurations the single smoke run never touches, printing a digest per config:
   `aa_mode` ∈ {Off, FXAA, TAA}, `shadow_cascades` ∈ {1..4}, `shadow_resolution` ∈
   {512, 4096}, `ao_mode` ∈ {Off, GTAO}, `local_shadows` ∈ {on, off}, `ibl` ∈ {on, off},
   plus a `--no-camera` run exercising `scene_renderer.cpp:643`'s
   has-content-without-camera path. Goldens per config. This closes the survey's biggest
   blind spot: the current smoke exercises exactly one settings configuration, so any
   config-dependent regression in Phases 6–9 lands green.
7. Docs: `CLAUDE.md`'s smoke paragraph and `docs/architecture.md` state that `--smoke` is
   deterministic, that goldens are regenerated deliberately and never silently, and that the
   digest — not the image — is the correctness gate.

**Gate**

```bash
cmake --build build
for i in 1 2 3 4 5; do build/examples/playground/playground --smoke | tail -3; done
# draw_calls / triangles / entities / packet_digest MUST be identical across all five
build/examples/playground/playground --smoke-sweep     # writes the per-config golden set
BRACKEN_WORKERS=1 build/examples/playground/playground --smoke   # digest identical
BRACKEN_WORKERS=8 build/examples/playground/playground --smoke   # digest identical
ctest --test-dir build --output-on-failure
```

The phase is **not done** until five consecutive runs and the 1/8-worker runs all agree on
the digest. Expect a one-time visual delta from the RNG change (a different particle
stream): review `playground_smoke.png` by eye once, then baseline.

**Rollback:** three commits (timestep, RNG, gate harness). Reverting the gate harness alone
restores the old liveness check; reverting the RNG change restores nondeterminism. If
digest stability proves unreachable for a specific sub-system, quarantine it out of the
digest with a named exclusion in `docs/architecture.md` rather than weakening the gate for
everything.

---

### Phase 3 — One log channel

**Serves:** transparency, ergonomics. A game that installs a `LogSink` today captures
barely half the engine's diagnostics and essentially **zero** GPU/pipeline/shader/renderer
boot errors — the exact failures that produce a black screen. `Logger::set_min_level`
(`logger.cpp:62-66`) is advertised as the level knob and silences none of them.

**Changes**

- Convert 180 `SDL_Log` sites across 22 non-test files to `BRACKEN_LOG_{ERROR,WARN,INFO}`
  with a per-file `constexpr const char* LOG_TAG` (pattern already at
  `render/water_renderer.cpp:15`). Distribution: `gpu_texture` 29, `pipeline_registry` 21,
  `gpu_buffer` 20, `terrain_renderer` 16, `gpu_shader` 16, `gpu_device` 13, `gpu_compute` 9,
  `grass_renderer` 8, `shadow_map` 7, `render_context` 7, `local_shadows` 7, `gtao` 5, plus
  bloom/taa/volumetric_fog/post_process_stack/light_cluster/ibl/auto_exposure/gpu_pipeline.
- Convert the 37 `std::cerr` sites in `scene/scene_renderer.cpp` (`:149`, `:189`, `:487`,
  `:1268`, `:2074`, `:2115`, `:2298`, …).
- Assign real severities during conversion: every `SDL_Log` is currently emitted at SDL's
  default INFO priority, so `pipeline_registry.cpp:99` ("Failed to create %s pipeline") and
  `:103` ("Created %s pipeline") are indistinguishable to a host.
- `core/logger.hpp:73` → `std::atomic<LogLevel> min_level_` with its protocol documented at
  the declaration as `CLAUDE.md` requires. `should_log` (`logger.cpp:83-86`) becomes a
  lock-free relaxed load: today it takes a `std::lock_guard` on the single global mutex
  **inside a `noexcept` function** — a per-call serialization point for every worker thread
  *and* a latent `std::terminate`. The mutex then guards only `sinks_`.
- New `core/logger_discipline.test.cpp`: fails on any `SDL_Log(` or `std::cerr` outside
  `core/logger.cpp` and `core/assert.hpp`. Locates the tree via a `BRACKEN_REPO_ROOT`
  compile definition; `GTEST_SKIP` when unavailable.

**Gate** — the survey's proposed "diff the smoke log stream" gate is broken, because this
phase intentionally changes severities and prefixes on all 217 sites, so a genuinely dropped
message is indistinguishable from an intended change. Replace it with two mechanical checks:

```bash
# 1. message-body preservation: extract the format strings before and after and diff the
#    normalized set (strip level/tag/category, collapse %s->{}), must be empty
tools/logdiff.sh                       # added by this phase, ~40 LOC of shell
cmake --build build && ctest --test-dir build --output-on-failure
grep -rn "SDL_Log(" src/bracken --include="*.cpp" | grep -v "\.test\." | wc -l   # 0
build/examples/playground/playground --smoke     # packet_digest unchanged vs Phase 2 golden
```

Plus a new positive test that installs a capturing `LogSink`, forces a GPU resource creation
failure, and asserts the sink received it at ERROR — the actual user-visible win, currently
untestable.

**Rollback:** land per directory (`gpu/`, then `render/`, then `scene/`, then the logger
atomics) as four commits, each independently revertible and attributable. The logger atomic
change requires the TSan lane.

---

### Phase 4 — Loud failures, honest numbers, contracts that survive release

**Serves:** transparency (the user's stated goal, literally), ergonomics. The day-1
experience of "I built a frame and see nothing" has no answer inside the engine today:
every drop path is a bare early-out.

**Changes**

*Diagnostics*

- `render_stats.hpp`: `struct FrameDiagnostics { uint32_t draws_dropped_unknown_model,
  skinned_dropped_no_skeleton, palettes_over_budget, effect_models_unresolved, lights_dropped,
  clusters_truncated; bool camera_missing; }`, accumulated through the existing
  `FrameStatsPartial` chunk-index-ordered merge (no new atomics, no shared cache lines),
  exposed as `Application::frame_diagnostics()`. Added to the smoke stat line and to the
  packet digest's sibling output so a regression in any drop path shows up in the gate.
- Wire the silent sites: `scene_renderer.cpp:1634-1637` (unknown/stale `ModelHandle`),
  `:2169-2179` (skinned draw with no skeleton / zero joints), `:643`
  (`has_3d_content && !have_frame_camera_` → `camera_missing`, with a warn-once naming the
  fix: *"frame has 3D content but no camera was ever set — call `frame().set_camera()`"*).
- Validate the handle at emit time in `FrameBuilder::Lane::draw_model`, mirroring the
  treatment `draw_skinned` already gives palettes (`frame_builder.cpp:274-278`), so the
  diagnostic names the emitting frame rather than the render pass.
- `light_cluster.cpp:208-220`: `add_point_light`/`add_spot_light` become `[[nodiscard]] bool`
  with a `lights_dropped_` counter and a warn-once latch in the style of `warned_truncation_`
  (`:478`), naming the count and the `MAX_LIGHTS=256` cap. `scene_renderer.cpp:799-800`
  checks the results. Today a host submitting 300 lights loses 44, chosen by lane-merge
  order, in total silence.
- New `core/log_throttle.hpp`: `BRACKEN_LOG_WARN_ONCE` / `_THROTTLED(interval_s)`. Applied
  at `text_renderer.cpp:203,:349`, `ui_renderer.cpp:238`, `effect_renderer.cpp:152,:210`.
  For `frame_builder.cpp:277,:291` prefer **aggregation**: count on the packet, emit one
  summary at `seal()` — quieter and more useful than N identical lines at 60 Hz.
- `model_loader.cpp:741-751`: the silently `continue`d glTF channels (non-skin nodes,
  non-TRS paths) become a counted `ModelLoadReport` logged once per model.

*Contracts*

- `core/assert.hpp`: add `BRACKEN_CONTRACT(cond, ...)` — always compiled (a compare and a
  branch), aborting via the existing `BRACKEN_VERIFY` path, with `BRACKEN_CONTRACTS=OFF` as
  a measured opt-out. Convert the host-boundary checks that vanish under `NDEBUG`:
  `frame_packet.hpp:514-517` (`assert_readable`), `frame_builder.hpp:184-187`
  (`assert_filling`), `frame_builder.cpp:185` (seal-once), `frame_builder.hpp:254-256`
  (lane index), `application.cpp:558` (`dev_ui()` before init). Hot per-item asserts stay on
  `BRACKEN_ASSERT`.
- Facade guards: `application.cpp:421-423, :453-455, :457-459, :473-475, :477-479, :574-576`
  dereference `scene_renderer_` unconditionally and segfault before `init_renderer`, while
  `screen_width()` (`:461-467`) null-checks — the rule is not inferable from the header. All
  get `BRACKEN_CONTRACT(scene_renderer_, "models() requires init_renderer() first")`.
- **Sim-stage tier, done correctly.** A `thread_local` depth counter cannot work: chunk
  bodies run on other threads (`parallel_for.hpp:65-67`) and `JobSystem::wait` executes
  stolen jobs on the waiting thread. Instead the stage travels **with the job node**: add a
  one-byte `stage` to `JobNode`, inherited from the submitting context at
  `job_system.cpp:368-384`; the worker sets a thread-local `current_stage` around the invoke
  and restores it after. Then `BRACKEN_ASSERT_SIM_STAGE()` / `BRACKEN_ASSERT_NOT_SIM_STAGE()`
  are correct under stealing, under nested waits, and on the main thread inside `join_sim`.
  `Application::run_sim_job` (`application.cpp:139-187`) sets the root stage.
  `NOT_SIM_STAGE` is applied to `GPUDevice` create/destroy, `PipelineRegistry` creation,
  `ModelManager` mutation and `DevUI` — **not** to `Application::screen_width()`, because
  Phase 9's engine-driven camera legitimately needs the viewport from sim and gets it via a
  value snapshotted at kick (see Phase 9).
- `FrameBuilder::Lane` gets an owner-thread member with the first-toucher CAS from
  `memory/arena.hpp:52-68`, checked in `assert_filling()`. Two host jobs emitting into
  `lane(0)` currently race on `chain.tail->count++` (`frame_builder.cpp:248-249`) and corrupt
  the chunk chain; the abort names the lane index.
- `docs/architecture.md:739-747` and `CLAUDE.md` are amended to state exactly which tiers
  are enforced and by which mechanism. Either the claim becomes true or the table says so.

*Honest numbers*

- New `render/pass_timeline.{hpp,cpp}`: `steady_clock` deltas into a fixed
  `std::array<Sample, 64>`, EMA-smoothed, driven from `GPUTimerScope`'s ctor/dtor
  (`gpu_debug.cpp:69-82`) so all 19 existing `BRACKEN_GPU_TIMER` scopes start reporting real
  numbers with no call-site change. Column labelled **"CPU (record)"** — this measures the
  cost of *recording* commands, not GPU execution, and the overlay must say so.
- `RenderStats::pass_times_ms` → `std::vector<PassTiming>` with
  `{ std::string_view name; std::optional<float> cpu_ms; std::optional<float> gpu_ms; }`;
  `gpu_ms` stays `nullopt` while `GPUTimestampPool::is_supported()` is false
  (`gpu_debug.hpp:52`, never surfaced today). Names are static literals owned by the
  `reserve_pair(const char*)` call sites (`gpu_debug.cpp:33-43`), so a view is lifetime-correct
  and removes ~15 `std::string` constructions per frame.
- `Application::render_stats()` → `std::optional<RenderStats>`, `nullopt` while
  `collect_stats_` is false (`scene_renderer.hpp:718`), so an all-zero struct stops being
  indistinguishable from a real empty frame.
- `Application::sim_ms() / render_ms() / fence_ms() / present_wait_ms()` next to `fps()`
  (`application.hpp:127`). **`present_wait_ms` is the important one**: with no GPU timestamp
  API, time blocked in swapchain acquire/submit is the only in-engine signal for
  "GPU-bound vs CPU-bound", and it is currently the unanswerable first question about a
  depth-2 pipelined frame.
- `RenderStats` grows the budgets that actually fail, each rendered as a value/budget bar in
  the Stats tab (`dev/dev_panels.cpp:63-77`): `packet_bytes_used/capacity`
  (`FramePacket::bytes_used()` at `:460` is read by nothing), `bone_palettes_used/budget`
  (no getter exists despite `MAX_BONE_PALETTES=1024` being a hard cap),
  `lights_submitted/binned/dropped`, `clusters_truncated`, `local_shadow_tiles_rendered`,
  `instances_drawn`, `ui_vertices/ui_draws`.
- Docs: delete `docs/user_guide.md:713`; amend the "GPU per-pass timings are stubs" gotcha in
  `CLAUDE.md` and `docs/architecture.md` to "CPU-record timings are live; GPU-side cost still
  needs RenderDoc/Nsight".

**Gate**

```bash
cmake --build build && ctest --test-dir build --output-on-failure
# new: frame_diagnostics.test.cpp drives a synthetic sealed packet containing
#   (a) a draw with an unregistered ModelHandle, (b) a skinned draw against a
#   skeleton-less model, (c) 3D content with no camera, (d) 300 lights
#   -> asserts each counter increments exactly once
# new: contracts.test.cpp EXPECT_DEATH for lane double-owner, seal-twice,
#   emit-after-seal, dev_ui-before-init, facade-before-init
build/examples/playground/playground --smoke        # packet_digest unchanged
build/examples/playground/playground --smoke-sweep  # all config digests unchanged
cmake --build build-tsan && ctest --test-dir build-tsan --output-on-failure   # job stage field
cmake --build build-rel && ctest --test-dir build-rel --output-on-failure     # BRACKEN_CONTRACT live
```

**Rollback:** four commits — diagnostics counters; contracts + facade guards; job-node stage
tier; pass timeline + stats. The tier commit is the one most likely to abort a legitimate
path; land it separately and smoke between.

---

### Phase 5 — The test net (no code moves in this phase)

**Serves:** transparency, and it is the precondition for Phase 6. `scene_renderer.cpp` is
3,696 LOC with **no behavioral test anywhere**; `scene_prep_parity.test.cpp` proves only
"same output at 1 and 4 workers" for three statics and never checks correctness. Goldens
authored *after* an extraction cannot detect that the extraction changed something — so they
land first.

**Changes**

- New `scene/scene_renderer_prep.test.cpp` — **golden-value**, not parity: a fixed synthetic
  packet with hand-computed expected bin membership, cull decisions, instance-batch grouping
  and order, palette byte layout, motion-id prev lookup, reusing the device-free
  `ModelManager` + `Lcg` harness at `scene_prep_parity.test.cpp:40-60`. **Mandatory
  coverage, two cases, both required by later phases:** (a) at least one command whose
  `ModelHandle` is unregistered, asserted in both the main-view and shadow-view arms — the
  exact case Phase 10's SoA bounds table can break and that the parity test structurally
  cannot catch; (b) a packet with **≥ 2 distinct model types *and* an unregistered handle
  simultaneously**, with the expected `base_instance` of every batch pinned — the only
  configuration that can detect a desync between the instance-batch pack loop
  (`scene_renderer.cpp:2086`) and the three draw walks (`:2427`, `:2595`, `:2844`), which
  Phase 6 separates into three different translation units. The smoke scene does not contain
  that combination today either; add it there in this phase, so the digest gate covers it
  before Phase 6 moves anything.
- New `gpu/gpu_resource_lifetime.test.cpp` — `create_*` on a null/failed device returns
  nullptr *and logs* (observable thanks to Phase 3), double-shutdown is safe,
  `GPUBuffer::update` rejects size overruns. Covers ~2,100 untested LOC of
  buffer/texture/device lifetime under every renderer.
- New `dev/dev_panels.test.cpp` — the four builtin tabs against a stub binding set,
  following the headless pattern at `dev_ui.test.cpp:194-260` (379 LOC, zero tests today).
- `gpu/gpu_uniforms.hpp`: a `static_assert(sizeof/alignof/offsetof)` block after each of the
  37 structs, **including the nine with no field-offset coverage at all**:
  `BloomUpsampleUniforms:638`, `GridVertexUniforms:161`, `InstancedCameraUniforms:190`,
  `InstancedShadowUniforms:267`, `PrepassMaskedUniforms:289`, `ShadowInstanceData:260`,
  `ShadowTerrainUniforms:345`, `ShadowTransformUniforms:324`, `UIScreenUniforms:168`.
  Delete the corresponding 225 `offsetof` `EXPECT_EQ`s from `gpu_uniforms.test.cpp`, leaving
  ~10 cross-module default-value tests. Net: CPU-side drift becomes a compile error in the
  same TU as the edit.
- New `gpu/hlsl_packing.hpp` with a `consteval` 16-byte-register-straddle predicate,
  expressing once the rule restated in prose six times (`gpu_uniforms.hpp:29-30, :176-178,
  :352-353`), used inside those blocks.
- **CPU↔HLSL validation via SPIR-V reflection, not a text scanner.** The survey proposed
  parsing 66 `cbuffer` blocks out of 55 `.hlsl` files with a homemade scanner that would have
  to reimplement HLSL packing, array stride and matrix majorness as DXC emits it — new
  untested code gating the failure `CLAUDE.md` calls the most common silent breaker. The
  ground truth is already in-tree: `GPUShader::reflect_spirv_resources`
  (`gpu_shader.cpp:125-145`) calls `SDL_ShaderCross_ReflectGraphicsSPIRV`, and
  `create_from_spirv` (`:183-192`) already reports declared-vs-reflected *count* mismatches.
  New `gpu/uniform_layout.test.cpp`: a table mapping each `gpu_uniforms.hpp` struct to
  (shader file, cbuffer slot), reading the **compiled SPIR-V** in the staged shader
  directory, extracting each uniform block's declared byte size, and `EXPECT_EQ`-ing against
  `sizeof(T)`. Plus a debug-only `push_uniform<T>(cmd, slot, const T&)` wrapper asserting
  `sizeof(T)` against the reflected size, applied to the 59 raw `SDL_PushGPU*UniformData`
  sites over time (not required to be complete in this phase).

**Gate — the net must be proven to work, not merely to pass:**

```bash
cmake --build build && ctest --test-dir build --output-on-failure
# then, locally and reverted afterwards:
#  * perturb one offset in gpu_uniforms.hpp  -> must be a COMPILE error
#  * add a float to one HLSL cbuffer         -> uniform_layout.test.cpp must FAIL
#  * flip one expected bin index in the prep golden -> that test must FAIL
```

A net that has never caught anything is not known to work. Record the three injected-breakage
results in the commit message.

**Rollback:** additive only — every file is new or a test deletion offset by a
`static_assert`. Revert is trivial and loses only coverage.

---

### Phase 6 — `RenderPlan` + `ScenePrep` extraction and the `scene_renderer.cpp` split

**Serves:** "logical components", partly. **Honest framing, corrected against the
decomposition survey:** a translation-unit split of one class into five files moves ~97 % of
`scene_renderer.cpp`'s 3,696 lines and reduces coupling by **exactly zero** — the header
stays 729 lines with 61 members, and all 23 shared-state relationships survive untouched.
Its payoff is navigability, parallel compilation, and — the real reason to do it first — it
is a *dry run* of the type boundaries: a function that refuses to sit cleanly in one TU is
evidence its state is genuinely shared rather than merely badly placed. So this phase does
the TU split **and** one true type extraction, `ScenePrep`, which is where the structural
value actually is. The remaining candidate types are named and deliberately deferred below.

**Changes**

- New `scene/render_plan.{hpp,cpp}`:
  ```cpp
  struct RendererCaps { bool post_ready, taa_ready, ibl_ready, camera_seen; };
  struct RenderPlan {
      bool has_content, use_hdr_path, taa_active, ibl_active,
           fog_active, do_frustum_cull;
      float draw_dist_sq;
      int   active_cascades;
      bool  invalidate_taa_history, invalidate_auto_exposure;
  };
  static RenderPlan plan_frame(const GraphicsSettings&, const FramePacket&,
                               ViewportState, RendererCaps);
  ```
  Note the shape corrections against the survey: three of the decisions are **not** derivable
  from settings + packet. `has_content` is `frame.has_3d_content() && have_frame_camera_`
  (`scene_renderer.cpp:643`) where `have_frame_camera_` is a *sticky* renderer member
  (`scene_renderer.hpp:617`, set at `:580`, never reset); `use_hdr_path` is `post_->is_ready()`
  (`:653`); `taa_active` needs `taa_->is_ready()` (`:660`); `ibl_active` needs
  `ibl_->is_ready()` (`:651`). Those four come in through `RendererCaps`.
  Equally: the prologue **interleaves state mutation with decisions** —
  `taa_->invalidate_history()` (`:662`) and `auto_exposure_->invalidate()` (`:673`) must fire
  on exactly the frames their decision flips false. The plan therefore *reports* those two as
  flags and `render_frame` performs the calls; it does not move them.
  **The minimized early-out is not hoisted.** `scene_renderer.cpp:630-638` sits deliberately
  between `begin_frame()` and `end_frame()` "so the frame lifecycle stays balanced"; moving
  it ahead of `begin_frame` unbalances the command-buffer lifecycle. It stays.
  **`RenderPlan` is the immutable half only, and that is deliberate.** The survey found three
  frame-scoped members that are set *mid-frame*, after the camera/frustum state is final, and
  therefore cannot live in a single const struct: `frame_prepass_active_`
  (`scene_renderer.hpp:703`, written at `:644` and again by `render_depth_prepass` at `:928`,
  read from 11 sites across four passes), `frame_ao_texture_` (`:708`, written `:648`/`:949`),
  and `frame_shadow_uniforms_` (`:670`, written `:874`). The naive "build one immutable
  `FrameView` at line 731 and pass it everywhere" refactor is *wrong* for exactly these three.
  Their eventual home is a separate mutable `FramePassState` — named here so the next pass
  does not re-derive it, and **not** scheduled: see "deliberately deferred" below.
- New `scene/scene_prep.{hpp,cpp}` — `class ScenePrep`, the one true type extraction in this
  phase. It owns `PrepParams`, `PrepChunkScratch`, `PrepViewScratch`, `PrepScratch`,
  `InstanceBinOutputs`, `ResolvedSkinnedEntry`, `PREP_COMMAND_GRAIN`, `PREP_PALETTE_GRAIN`
  (`scene_renderer.hpp:216-372`, ~157 header lines) and the bodies at `scene_renderer.cpp`
  `:1735-2021`, `:2154-2245`, `:2319-2337` (~483 LOC). It borrows `const ModelManager*`,
  `const MotionHistory*`, `const Frustum*` ×(1+N), `JobSystem&`, spans over the sealed
  packet, and the output containers by reference.
  **Why this one and not the pass objects:** it is already a device-free static API with an
  explicit params-in / outputs-out / scratch contract, it already has a direct bit-identity
  acceptance test (`scene_prep_parity.test.cpp`, 442 LOC, driving the statics at 0/1/4
  workers), and it is **the only thread-tier boundary inside the renderer**. Extracting it
  converts that boundary from a comment plus a hand-checked `PrepParams` field list into a
  compile-time fact: a `ScenePrep` with no `GPUDevice`, no `RenderContext` and no
  sub-renderer member *cannot* reach a main-thread-tier API. That is this plan's own thesis —
  prose contract becomes a type — applied to the one place in `scene/` where it pays.
  **Hard rule, stated in the header:** `ScenePrep` never gains a `RenderContext*`, a
  `GPUDevice*` or a sub-renderer pointer "for convenience". Doing so silently re-opens the
  tier, and Phase 4's `BRACKEN_ASSERT_NOT_SIM_STAGE` would then start firing from a worker.
  Three thin adapters stay on `SceneRenderer` (`build_instance_batches :2023-2053`,
  `pack_bone_palettes :2247-2280`, the `prep_copy_palettes` call at `:2314`, ~55 LOC total),
  because they touch renderer members.
- New `scene/render_plan.test.cpp`: table test over every settings × packet-content ×
  caps combination, including TAA-requested-but-post-not-ready (must be inactive — jitter
  without a resolve just shimmers), no-camera, IBL-enabled-but-not-ready, and every cascade
  count. This is the first coverage the pass-selection tree has ever had.
- Split the remainder of `scene/scene_renderer.cpp` into six TUs of the **same class**, no
  interface change. Six, not five: the original list left `render_instanced_shadow_models`
  (`:2811-2902`) unassigned, and the survey's line audit shows it must travel with the shadow
  passes, not the draw TU, because it is the third of the three `instance_batches_` draw
  walks (see the invariant note below).
  - `scene_renderer.cpp` — init/shutdown/lifecycle/setters + `render_frame` orchestration
    (`:116-565`, `:1255-1339`)
  - `scene_renderer_upload.cpp` — `upload_instance_buffers :2055-2143`, bone-palette double
    buffer `:2282-2317`. **Split out from the prep TU on purpose**: what is left in
    `ScenePrep` is device-free and worker-safe, what is here takes a `SDL_GPUCopyPass*` and
    is main-thread. Making the tier a *file* boundary as well as a type boundary means a
    reviewer sees a `#include "bracken/gpu/..."` appear in the wrong TU.
  - `scene_renderer_shadow.cpp` — `render_instanced_shadow_models :2811`,
    `render_shadow_passes :3264`, per-cascade, `bind_local_shadow_atlas :3466`,
    `render_local_shadow_passes :3487-3652`, `bind_shadow_data :3654`,
    `bind_cluster_lights :3671`
  - `scene_renderer_prepass.cpp` — `render_depth_prepass :2502`, `:2562`, `:2648`, `:2732`
  - `scene_renderer_draw.cpp` — `render_3d_scene :1339`, model commands `:1470/:1633`,
    binds `:1484-1631`, instanced `:2339`, skinned `:2904`, transparent `:2992/:3005`,
    UI `:3094`, billboards `:3126`, debug lines `:3163/:3224`
  - `scene_renderer_capture.cpp` — screenshot download + PNG encode (`:1255-1333`). The
    frame-lifecycle branch at `:1248-1252` (`end_frame_and_wait` vs `end_frame`) **stays in
    the orchestrator**; only the capture body moves.
  Private implementation declarations move out of `scene_renderer.hpp` (729 LOC) into
  `scene/scene_renderer_internal.hpp`, included only by those TUs, leaving the public
  header as the host-facing surface. That internal header must also carry the file-static
  template `apply_mesh_material` (`:61-89`) — it is called from `:1698`, `:2469` and `:3070`,
  i.e. from three of the six new TUs, so it cannot stay TU-local. Record that as a known,
  accepted weakening of its internal linkage. **No renames**: `SceneRenderer` appears on 206
  doc lines. Files are auto-globbed (`GLOB_RECURSE`), so no CMake edit.
- **Two invariants the split scatters. Both must be handled inside this phase, not after.**
  - *`base_instance` agreement (survey S8).* The instance-batch pack loop
    (`:2086`, packs every non-empty bin in ascending dense index) and the three draw walks
    (`:2427` forward, `:2595` prepass, `:2844` shadow) must agree exactly, and the walks skip
    bins whose `get_model()` is null while still advancing `base_instance` (`:2435/:2603/
    :2852`). It is correct today and enforced by nothing. The split puts those four sites in
    four *different files*. Mitigation, required: hoist the walk into one
    `for_each_instance_batch(fn)` on `SceneRenderer` so the ordering has exactly one
    definition, and have all four sites call it. Failure mode if skipped: instances render
    with another model's transforms, visible only in a scene with both a missing-model handle
    and ≥ 2 model types — which the smoke scene does not currently contain, so **add one to
    the Phase 5 prep golden and to the smoke scene before splitting.**
  - *Never-reset pipeline-variant flags (survey S16, a latent defect).*
    `model_pipeline_double_sided_`, `skinned_pipeline_double_sided_`,
    `model_pipeline_depth_equal_`, `skinned_pipeline_depth_equal_`
    (`scene_renderer.hpp:570-573`) are written only at `:1502-1503`/`:1531-1532` and read at
    `:1692-1693`/`:2962-2963`, and are **never reset at a pass or frame boundary**. This is
    safe today purely because the first draw of every frame enters through
    `bind_model_pipeline`/`bind_skinned_pipeline` from `render_3d_scene` (`:1409`, `:1431`).
    A TU split does not change entry order, so it does not break this — but it moves the
    writer and the readers into different files, where the coupling stops being visible.
    Required: a `BRACKEN_ASSERT` at each read that a pipeline bind has happened this pass,
    plus a comment at the declaration naming the entry-order dependency. Do **not** silently
    "fix" it by resetting the flags in this phase: that is a behaviour change (extra pipeline
    binds) and would move the digest.
- **Deliberately deferred, named so they are not re-derived.** The survey enumerates further
  extractions; this phase takes none of them. `FrameView`/`FramePassState` (collapses 22
  header members into 2 structs, ~120 read sites) is the prerequisite for everything below it
  and is worth doing — but only after `ScenePrep` and the TU split have landed and the digest
  has held. `FrameGeometry` (the instance/shadow/palette buffers), `ModelBindState` (dedupes
  four verbatim copies of the slot-1/2/6/7/10 bind sequence at `:1497`, `:1524`, `:2411`,
  `:3041`), `PostChain`, and `DebugLineRenderer`/`ScreenshotCapture` as real types: worth
  doing only if a measured need appears. **The per-pass objects (`DepthPrepassPass`,
  `ShadowCascadePass`, `LocalShadowPass`, `ForwardScenePass`, ~1,290 LOC) are rejected
  outright** — the pass *sequence* is load-bearing (20 explicit, commented ordering decisions
  in `render_frame`, the same reason `render_graph/` is not in the frame path), so it must
  stay one linear script regardless; pass objects buy nothing but 6–10 back-reference members
  each and turn `main_render_pass_`, the prepass flag and the pipeline-variant flags into
  cross-type invariants. Recorded in `docs/engine_design_notes.md` as a dated decision.
- `docs/architecture.md`: `RenderPlan` named as the authoritative "which passes run this
  frame" decision point; `ScenePrep` named as the Prepare-stage tier boundary in the
  "Threading & lifetime contracts" section, replacing the prose there; the `scene/` module
  row gains the file split. `CLAUDE.md`'s `scene/` row gains `ScenePrep`.

**Gate**

```bash
cmake --build build && ctest --test-dir build --output-on-failure   # +~60 render_plan cases
# scene_prep_parity.test.cpp must pass UNCHANGED except for the type it drives —
#   it compares bins, bin order, touched lists, non-instanced list, stats, resolved
#   skinned entries and palette bytes with memcmp at 0/1/4 workers. It is the
#   acceptance criterion for the ScenePrep extraction.
# Phase 5's prep goldens must be byte-identical, including the added
#   unregistered-handle + two-model-type case that pins the base_instance walk.
build/examples/playground/playground --smoke        # packet_digest EXACTLY unchanged
build/examples/playground/playground --smoke-sweep  # all config digests unchanged
BRACKEN_WORKERS=1 build/examples/playground/playground --smoke   # and =8: digests identical
nm -g build/src/bracken/libbracken.a | sort > /tmp/after.syms
# diff against the pre-phase capture
```

The `nm -g` diff mechanically proves nothing was added, dropped, or given different linkage
during the *TU move* — a check neither the compiler nor the tests provide. **It is an
identity check for the split commits only.** The `ScenePrep` commit legitimately changes the
symbol set (the three statics move from `SceneRenderer`'s scope into `ScenePrep`'s), so for
that commit the diff is *reviewed*, not required to be empty: the expected delta is exactly
those symbols and nothing else. Capture the baseline before the phase, and diff after each
commit rather than once at the end. The sweep is what catches a config-dependent regression
the single smoke run cannot see.

**Rollback:** eight commits — `RenderPlan` + its test first (pure addition); then
`for_each_instance_batch` + the S16 asserts (behaviour-neutral, digest must not move); then
`ScenePrep` (the one commit with a legitimate symbol-set delta); then one TU extraction per
commit. Any digest movement means the move was not pure; revert that single commit rather
than debugging forward. `ScenePrep` is ordered *before* the TU splits so that if it has to be
reverted, the splits do not have to be.

---

### Phase 6b — Make the module dependency graph a DAG

**Serves:** "logical components" — this is the structural half of that goal, and it is the
one axis the surveys behind the first draft of this plan did not cover.

**The finding.** Attributing all 554 intra-tree includes to modules (longest prefix;
`core/jobs`, `core/asset` and `render/lighting` counted separately) gives a production module
graph with **one non-trivial strongly-connected component of six modules** — `{@root, dev,
render, render/lighting, scene, systems}` — containing 21 elementary cycles of length ≤ 4.
Restricted to headers only, where the coupling is a *compile* dependency rather than a link
one, it is still five modules and six cycles. There are **zero file-level include cycles**,
which is why it compiles and why this has been invisible.

Two things bound the scope before anything is proposed:

- **There is nothing to rescue.** Zero production includes run from `core/`, `core/jobs/`,
  `core/asset/`, `memory/` or `gpu/` into anything above them; `core/` has fan-out 0, a
  perfect sink. The only layering inversion in the tree is
  `gpu/gpu_uniforms.test.cpp:2 → render/environment.hpp` (a test), which this phase fixes
  incidentally. The engine's layering is already correct in the direction that matters.
- **The cycles are not caused by wrong names.** They are caused by pure-data types filed
  under whichever module happened to author them. Verified: `render/environment.hpp`,
  `render/lighting/light.hpp`, `scene/camera_state.hpp`, `scene/frustum.hpp`,
  `scene/velocity_math.hpp`, `graphics_settings.hpp` and `render_stats.hpp` each include
  **nothing** from `bracken/` — glm and `<std>` only. They are a leaf layer that has no
  directory, so each one is currently attributed to a module *above* the layer that needs it.

**Why this is not the module reshuffle §6 rejects.** The distinction is measurable, and it is
the whole argument: moving `input_handler.hpp` to `input/` changes a path and nothing else
(fan-in from `@root` only, fan-out 0 into any module). Moving `EnvironmentState` out of
`render/` **reverses the direction of a dependency edge** — it is the reason `scene → render`
exists at the header level, which is the reason the six-module SCC exists. §6 is amended to
say this rather than lumping the two together.

**The cut set.** Four changes, each verified against source, chosen to be the minimum that
makes the header graph acyclic:

1. **Hoist `ModelHandle` out of `model_loader.hpp`.** New `contracts/model_handle.hpp`
   (~12 lines): `using ModelHandle = uint32_t;` and
   `inline constexpr ModelHandle INVALID_MODEL_HANDLE = 0;` (today `model_loader.hpp:31-32`,
   which keeps them visible by including the new header). `scene/frame_packet.hpp:5`
   includes that instead. Its comment already says `// ModelHandle`, and that is literally
   all it uses (`:66`, `:85`). Today that one line drags `gpu/gpu_buffer.hpp`,
   `gpu/gpu_texture.hpp`, `gpu/gpu_types.hpp` and `animation/animation_types.hpp` into every
   TU that touches the packet — **including `application.hpp` and `sim_frame.hpp`**, via
   `application.hpp:5 → sim_frame.hpp:7 → scene/frame_builder.hpp → frame_packet.hpp:5`.
   This is the single highest-value line in the phase: it is a three-line change that cuts
   the GPU resource layer out of the host-facing header chain, which is exactly what
   Phase 11's umbrella needs and cannot otherwise get.
2. **New `src/bracken/contracts/` — a leaf module** whose rule is stated in a `README`-style
   header comment and enforced by a test: *may include `bracken/core/*`, glm and `<std>`,
   nothing else.* Namespaces are **unchanged** — a header's directory and its namespace are
   independent, and keeping `bracken::render::EnvironmentState` spelled exactly as it is
   today is what makes this an include-line rewrite instead of a rename pass. Moves:
   `render/environment.hpp` (14 include lines, 12 doc citations),
   `scene/camera_state.hpp` (10 / 5), `scene/frustum.{hpp,cpp}` (9 / 3),
   `scene/velocity_math.hpp` (2 / 2), `graphics_settings.{hpp,cpp}` (8 / 14),
   `render_stats.hpp` (5 / 8).
   `render/lighting/light.hpp` is **split, not moved**: `PointLight`, `SpotLight`,
   `LightHeader` and `LIGHT_TYPE_*` (`:17-55`) go to `contracts/light.hpp` (6 include lines,
   6 doc citations); the cluster and atlas budget constants (`:58-75` — `CLUSTER_DIM_*`,
   `MAX_LIGHTS`, `MAX_LIGHTS_PER_CLUSTER`, `LOCAL_SHADOW_*`) **stay** at
   `render/lighting/light.hpp`, which includes the contract header. That preserves the
   documented constant home (`CLAUDE.md`'s "Load-bearing constants", `architecture.md`) and
   is the more accurate split anyway: the file today mixes a host-facing data contract with
   renderer-internal budgets, which is this phase's whole thesis in miniature.
   **Total churn: 54 include-line rewrites and ~50 doc path citations** — an order of
   magnitude below the `bracken/gpu/` (125 doc lines) / `bracken/core/` (101) figures §6 used
   to price and reject module moves. Measured, not estimated.
3. **Cut `render → systems` entirely.** `render/effect_renderer.hpp:10` is the *only*
   production `render → systems` include in the tree, and it exists for one declaration:
   `build_particle_transform(const bracken::systems::Particle&, const glm::vec3&)` at `:29`.
   Its only caller anywhere is `render/effect_renderer.test.cpp`; the production path uses the
   `EffectParticleSnapshot` overload at `:34`, and both forward to the same template
   `build_particle_transform_impl` (`effect_renderer.cpp:39`). Move that template into a small
   `render/particle_transform.hpp` and delete the `systems::Particle` overload from the header
   and from `effect_renderer.cpp:69-71`; the test instantiates the template against
   `systems::Particle` directly and keeps its exact assertion. ~12 lines.
4. **Pull the effects-triad consolidation forward from Phase 11 into this phase.**
   `effect_definition.hpp` (top level) + `systems/effect_system.*` + `render/effect_renderer.*`
   → `src/bracken/effects/`. Phase 11 already scheduled this as "the one genuine mis-filing";
   it belongs *here* because it is a cycle fix, not tidiness — it removes the
   `systems → @root` edge (`effect_system.hpp:5 → effect_definition.hpp`) and makes
   `effect_renderer → effect_system` intra-module. Phase 11 keeps the umbrella and the docs
   and cites this phase.

**What this deliberately does not do**, with the reason:

- **`model_loader.hpp` / `model_utils.hpp` stay at the root.** After change 1 the only
  remaining `render → @root` and `scene → @root` header edges are
  `render/effect_renderer.hpp:8 → model_loader.hpp` (it needs the real `ModelManager` and
  `Model`) and `render/terrain_renderer.hpp:8 → heightmap.hpp`, and there is **no `@root →
  render` header edge at all** (`application.hpp` includes only `graphics_settings`,
  `input_handler`, `sim_frame`, `render_stats`, `core/jobs/task_handle`). So those edges do
  not close a compile cycle. The residual `@root ↔ render` and `@root ↔ scene` module cycles
  after this phase exist **only through `.cpp` files** (`application.cpp:9 → render/
  render_context.hpp`, `:10-14 → scene/`, `dev/`, `systems/`), and a `.cpp` cannot participate
  in a compile cycle — it is a statement about where `Application` lives, not about coupling.
  Moving `model_loader` (20 doc path citations, 78 doc mentions, 18 source references) and
  `heightmap` (8 / 104 / 167) buys nothing this phase needs. §6 records the amended verdict.
- **`SceneRenderer` is not moved out of `scene/`.** It is the one arguably-wrong boundary
  left — `scene/` holds both the bottom-of-stack frame contract (`FramePacket`,
  `FrameBuilder`, `PacketRing`) and the top-of-stack frame orchestrator, which is why `scene`
  uniquely has out-degree 8 *and* in-degree 5. But `scene_renderer.hpp` includes no `render/`
  header after change 2, and `scene_renderer.cpp`'s 16 sub-renderer includes are `.cpp`-only,
  so the boundary costs conceptual clarity and zero compile coupling. Deferred to §7 Q7 as an
  owner decision, not scheduled.

**Result.** The production **header** graph becomes a DAG. Written as layers:

```
core                                              (sink, fan-out 0)
  core/jobs, core/asset, memory, contracts        (contracts: core + glm + <std> only)
    gpu, animation
      model_loader.hpp, heightmap.hpp             (see note)
        render, render/lighting, procedural
          scene                                   (FramePacket / FrameBuilder / PacketRing)
            effects, systems
              @root  (application, sim_frame, input)
                dev
```

`render → scene` survives only as `effects/effect_renderer.hpp → scene/frame_packet.hpp`,
which points *down* this ordering. The SCC is gone.

**Note, stated rather than glossed:** `@root` is not one layer after this phase — it holds
both the top of the stack (`application.hpp`, `sim_frame.hpp`, the input pair) and two
dependency *leaves* that modules below reach up for by path,
`model_loader.hpp` (16 include lines) and `heightmap.hpp` (6). That is untidy, and a `model/`
directory would tidy it. It is still rejected (§6) because those two edges do not close a
cycle — nothing in `@root`'s top half is included by any header in `render/`, `scene/`,
`effects/` or `systems/`, so the graph is acyclic despite the layer diagram needing a
footnote. The honest summary is: **acyclic, with two files filed one directory too high.**
If Q7(b) is ever answered "yes, split `scene/`", fold the `model/` move into the same pass —
that is the moment it becomes free.

**Changes to docs**

- `docs/architecture.md`: the module map gains a `contracts/` row and an `effects/` row,
  loses the `systems/effect_system` and `render/effect_renderer` entries, and gains a short
  **"Module layering"** subsection stating the DAG above and the leaf-module rule. This is
  the first time the engine's layering has been written down as a checkable fact rather than
  implied by a table.
- `CLAUDE.md`: same two module-table rows; the cluster/atlas constants entry is amended to
  say `PointLight`/`SpotLight` live in `contracts/light.hpp` and the budgets stay in
  `render/lighting/light.hpp`.
- Phase 11's "the one genuine mis-filing" bullet now points here.

**Gate**

```bash
cmake --build build && ctest --test-dir build --output-on-failure
# new: module_layering.test.cpp — builds the header-only include graph over src/bracken
#   (same BRACKEN_REPO_ROOT mechanism as Phase 3's logger_discipline test, GTEST_SKIP when
#   unavailable) and asserts:
#     (a) the module graph over .hpp files has NO cycle  -- the phase's whole point
#     (b) nothing under contracts/ includes anything but bracken/core/*, glm, <std>
#     (c) nothing under core/, memory/, gpu/ includes render/, scene/, systems/, effects/,
#         dev/ or a root header      -- pins the zero-layering-violations baseline
#   (b) and (c) are the regression guards; (a) is the assertion this phase makes true.
grep -rn '#include "bracken/render/environment.hpp"' src examples | wc -l   # 0
build/examples/playground/playground --smoke        # packet_digest EXACTLY unchanged
build/examples/playground/playground --smoke-sweep  # all config digests unchanged
nm -g build/src/bracken/libbracken.a | sort | diff - /tmp/before.syms       # EMPTY
```

`nm -g` must be **byte-identical** here, unlike Phase 6: this phase moves files and rewrites
include lines but changes no definition, no namespace and no signature. The single exception
is change 3, which deletes one overload — land it as its own commit so the symbol-set diff
for that commit is exactly one removed symbol and the other commits are empty.

Note that `-Werror` from Phase 1 plus the compiler is the real gate for changes 1–2 and 4:
every moved header is a path substitution, so a missed site is a hard compile error, not a
silent behaviour change. The digest is a formality here and is included only because a
formality that has never been run is not a formality.

**Rollback:** five commits — `ModelHandle` hoist; `contracts/` creation + moves;
`light.hpp` split; `render → systems` cut; effects triad. Each is independently revertible
and each is a pure path/include change except the fourth. If `contracts/` proves to be the
wrong name (see §7 Q7), renaming the directory afterwards is one `git mv` plus the same 54
include lines — do not treat the name as load-bearing.

---

### Phase 6c — `PipelineRegistry`: separate the registry from the catalogue

**Serves:** "logical components", and it is the largest single deletion available anywhere in
this plan. The first draft missed this file entirely.

**The finding.** `gpu/pipeline_registry.cpp` is 1,686 LOC and is two things welded together,
at a ratio of about 1:6 against the part that justifies the name:

- **A genuine registry, ~200 LOC (`:21-464`)**: cache keyed by a closed enum, lookup-or-create
  (`:69-109`, `:291-328`), eager preload (`:344-376`), invalidation (`:378-388`), selective
  swapchain-format invalidation with eager re-creation (`:390-423`), file-watcher hot reload
  (`:425-464`), and a creation counter plus a `preloaded_` latch that logs any lazy creation
  after preload (`:85-94`, `:306-313`). This part is good — the "worker-safe read-only after
  preload" contract (`pipeline_registry.hpp:282-292`) is exactly right, and the latch is a
  real enforcement mechanism. **It is kept as-is.**
- **A hardcoded catalogue, ~1,215 LOC (`:470-1684`, 72 % of the file)**: 38 creator functions,
  a 172-line enum→function switch (`:114-286`), and a 118-line enum→string switch in the
  header (`pipeline_registry.hpp:137-254`).

The tell that it is a catalogue and not a registry is that **the enum is a flattened Cartesian
product**: 20 of the 56 `PipelineType` values are `(base, doubleSided, alphaBlend, depthEqual)`
tuples already served by 5 parameterized creators — and the flat names spell the axes in three
different orders (`ModelDoubleSided`, `ModelBlendDoubleSided`, `ModelDoubleSidedDepthEqual`,
`DepthPrepassModelDoubleSided`). Measured repetition: 38 verbatim
`if (!vs || !fs) return nullptr;` blocks, 76 `shader_manager_->get(...)` calls (exactly 2 per
creator), 14 copies of `config.opaque().no_depth().no_cull();`, 4 copies of
`with_depth_bias(1.25f, 1.75f)` (`:573, :1138, :1166, :1209` — a tuning constant with no
single definition), 3 copies of the terrain 12-float vertex stride (`:650, :1070, :1196`, the
full 4-attribute layout written out twice), 3 copies of `num_samplers = 11` **with its
hand-maintained inventory comment** (`:477, :520, :593`), 5 copies of
`num_storage_buffers = 4`. The 15 fullscreen post creators (`:1246-1610`, 365 lines) differ in
exactly six values: fragment shader path, uniform-buffer count, sampler count, storage-buffer
count, color format, and blend mode (additive for `BloomUpsample` only, `:1412`).

**Changes**

- New `gpu/pipeline_catalog.{hpp,cpp}`: a `constexpr` `std::array<PipelineDesc,
  PipelineType::Count>` plus `PipelineConfig build(const PipelineDesc&, const FormatSet&)`.
  `PipelineDesc` carries shader paths, resource counts, `VertexLayout`, `TargetFormat`, blend
  mode, cull/front-face, the depth triple (test/write/compare), depth bias, depth clip, and a
  `name` string. Every field of `PipelineConfig` (`gpu_pipeline.hpp:19-57`) is already POD, so
  nothing here is a new abstraction — it is the same data, tabulated.
  `static_assert(std::size(k_pipeline_descs) == size_t(PipelineType::Count))` turns two
  hand-maintained 56-case switches that can silently drift from the enum into one table where
  a missing row is a compile error.
- Two resolvers, ~120 LOC total, because two things in the table are not constants:
  - `enum class TargetFormat { Swapchain, SceneColor, Velocity, DepthOnly, R8, RG16F, RGBA16F }`
    resolved at build time against `scene_color_format_`, `swapchain_format_`,
    `velocity_format_`, `depth_format_` (`pipeline_registry.hpp:566-571`), which
    `set_swapchain_format` mutates at runtime (`:390-423`). Bonus: the hand-maintained
    4-entry `swapchain_dependent` list at `:400-402` becomes a query over the table
    (`desc.color == TargetFormat::Swapchain`) — today it is 4 entries maintained by hand
    against 56 rows, which is a drift bug waiting to happen.
  - `enum class VertexLayout` with ~9 cases (`Vertex3D`, `SkinnedVertex`, `Vertex2D`, terrain
    12-float, grid 7-float, text 4-float, water 2-float, position-only, position+texcoord,
    none). Removes the 3 duplicated terrain strides and the 2 duplicated terrain layouts.
- `pipeline_registry.{hpp,cpp}` keeps `:21-464` verbatim, takes the catalog by reference, and
  **no longer names a single pipeline**. `create_pipeline_uncached`'s 172-line switch becomes
  `build(k_pipeline_descs[size_t(type)], formats())`; `pipeline_type_to_string`'s 118 header
  lines become `k_pipeline_descs[size_t(type)].name`.
- Delete the 4 dead convenience accessors — `get_grass_pipeline`, `get_particle_pipeline`,
  `get_terrain_pipeline`, `get_volumetric_fog_pipeline` (`pipeline_registry.hpp:355-383`),
  verified 0 call sites outside the header. Inline the 23 that have exactly one call site.
- **Correctness win worth calling out.** The reversed-Z overrides are currently scattered
  across 12 sites (`:504, :540, :578, :619, :663, :702, :835, :865, :1140, :1168, :1211,
  :1239`) — the post-prepass `EQUAL` + `depth_write_enable=false` variants, the shadow
  cascades' deliberate standard-Z `LESS_OR_EQUAL`, and the skybox's z=0 `GREATER_OR_EQUAL`.
  A table makes all 12 one reviewable column. Given `CLAUDE.md`'s reversed-Z contract, this is
  the strongest single argument for the change and also its main risk — see the gate.
- `src/bracken/shaders/README.md`'s pipeline→shader map becomes a pointer at the table
  instead of a second hand-maintained copy of it. `docs/architecture.md`'s GPU-layer section
  gains the registry/catalogue split.

**Explicitly out of scope for this phase, with its trigger.** *Reflecting* `ShaderResources`
from the compiled SPIR-V instead of declaring them by hand — which would delete 76 numbers
including all three copies of `num_samplers = 11` and its inventory comment — is the *right*
end state and is what the compute path already does (`GPUComputePipeline::load_spirv`
validates declared `ComputeResources` against reflection). It is **not** done here: tabulating
the catalogue and changing where resource counts come from are two independent risks, and
stacking them makes a failure unattributable. Trigger: land this phase, then do the reflection
switch as a follow-on once Phase 5's `gpu/uniform_layout.test.cpp` has been *proven* to catch
an injected one-sided edit. Until then the table centralizes the three copies into one row
reference, which is a real improvement and stops short of the risky part.

**Gate**

Pipeline creation is boot-time and fully covered: `preload_all_pipelines` (`:344-376`) returns
false on any failure, so a broken row cannot reach a frame. That makes the mechanical gate
easy and the semantic gate the interesting one.

```bash
cmake --build build && ctest --test-dir build --output-on-failure
# new: pipeline_catalog.test.cpp — for every PipelineType, build() the descriptor and
#   compare the resulting PipelineConfig FIELD BY FIELD against a checked-in expected
#   table captured from the CURRENT creators before the migration. This is the gate.
#   Capture it with a temporary dump added to the old code path in the first commit,
#   checked in as gpu/pipeline_config_golden.inc, and deleted in the last.
build/examples/playground/playground --smoke        # packet_digest unchanged; screenshot
                                                    # SHA advisory — REVIEW THE PNG BY EYE
build/examples/playground/playground --smoke-sweep  # varies aa_mode/ao_mode/cascades/
                                                    # shadow_resolution: exercises the
                                                    # depth-EQUAL and shadow rows
cmake --build build-rel && ctest --test-dir build-rel --output-on-failure
```

The field-by-field config diff is non-negotiable and is what makes this low-medium rather than
high risk: the failure mode of a wrong row is not a crash but a wrong image (a cascade
compared with the wrong depth op, a prepass variant that writes depth), and the digest is
device-side-blind by construction — it hashes what the host described, not what was drawn. The
golden config table is the only mechanical check that sees GPU state.

**Rollback:** four commits — golden capture (additive, no behaviour); catalog + resolvers,
old creators still present and still used; switch the registry over to the table and delete
the creators; delete dead accessors + docs. Commit 3 is the only one that can move an image;
revert it alone.

---

### Phase 7 — Settings become types

**Serves:** ergonomics, self-description. `graphics_settings.hpp:13-40` already establishes
the pattern with four scoped enums; seven fields were left behind, encoding their meaning in
a trailing comment and silently `std::clamp`ed (`graphics_settings.cpp:23-33`). The failure
mode is invisible: `shadow_resolution = 2048` looks correct and becomes `3` = 4096;
`shadow_cascades = 4` clamps to `3`, which also means four cascades.

**Changes**

- Add `DrawDistance`, `AnisotropicFilter`, `VSyncMode{Immediate,Vsync,Mailbox}`,
  `ShadowCascades{One=0..Four=3}`, `ShadowResolution{R512..R4096}`, `LocalShadowResolution`,
  `WindowMode{Windowed,BorderlessFullscreen,ExclusiveFullscreen}` — all `: int` with today's
  underlying values, carrying the existing "values are the on-disk format, keep them stable"
  comment from `:8-11`.
- Retype the seven fields (`:60, :64, :65, :67, :68, :77, :88`). `resolution_index` stays an
  int index.
- `scene_renderer.cpp:198-205`: `resolution_table[std::clamp(gfx.shadow_resolution,0,3)]`
  becomes a total switch; `set_active_cascades(gfx.shadow_cascades + 1)` becomes
  `set_active_cascades(cascade_count(gfx.shadow_cascades))`.
  `graphics_settings.hpp:130-136` `get_draw_distance()` loses its out-of-range path.
- `Application` setters (`application.hpp:206-209`) take the enum types.
- `static_cast<int>` survives only at the two boundaries the file's own header comment
  prescribes: `save()`/`load()` (`graphics_settings.cpp:68, :82, :165`) and the GPU-uniform
  boundary. `clamp_to_valid_ranges()` still runs at the load trust boundary but validates an
  integer read from disk before casting into the enum, instead of silently clamping a field
  a host just set.
- ~104 references across 12 engine files. **Zero in `examples/`** — the playground never
  touches these fields, so the reference consumer needs no edits despite the API break.

**Gate** — `-Werror` proves the *unconverted* sites, but the dangerous ones are already
written as explicit casts and therefore silent: `graphics_settings.cpp:68/:82/:165` and
`dev_panels.cpp`'s combo widgets (`dev_panels.cpp:165` takes `GraphicsSettings& settings`
and indexes via `static_cast<int>`). So:

```bash
cmake --build build && ctest --test-dir build --output-on-failure
# graphics_settings.test.cpp gains: load a checked-in pre-change settings file fixture,
#   assert every field decodes to the expected ENUMERATOR (not just that bytes round-trip),
#   re-save and byte-compare against the fixture
build/examples/playground/playground --smoke-sweep
#   the sweep varies shadow_cascades 1..4 and shadow_resolution 512/4096 explicitly —
#   this is the only mechanical check that the DECODE mapping did not move
```

The file-format round-trip alone proves the bytes, not the decode. The sweep is the decode
test, and it only exists because of Phase 2.

**Rollback:** one commit, mechanical, fully compiler-verified. On-disk format unchanged, so
no user settings file migrates.

---

### Phase 8 — Draw API, `FrameSink`, and a public test fixture for hosts

**Serves:** ergonomics — the user's top priority. This is the flagship defect in the
flagship API.

**Changes**

- New `scene/draw_flags.hpp`: `enum class DrawFlags : uint32_t { None=0, ForceNonInstanced=1<<0,
  NoFog=1<<1 }` with `constexpr` `|`, `&`, `has()`. Change the four declarations
  (`frame_builder.hpp:126, :136, :261, :266`) and their definitions. Fold the flags into
  `ModelDrawCommand`/`SkinnedDrawCommand` as one packed `uint32` replacing the separate
  bools — the structs are documented padding-free and byte-deterministic for lane merges, so
  the layout comment and any `sizeof` assumptions get verified in the same edit.
- `core/color.hpp` grows `struct Color { uint32_t argb; static constexpr Color srgb8(...);
  static constexpr Color from_linear(glm::vec4); glm::vec4 to_linear() const; }`, used for
  every packed-color parameter (`frame_packet.hpp:114-121, :185-230`). Today the API mixes
  three unrelated conventions — packed ARGB `uint32_t`, linear `glm::vec4` tint, linear
  `glm::vec3` light color — and nothing at a call site says which.
- Host-facing light descriptors split from the GPU-layout POD.
  `light.hpp:17-29`/`:31-46` contain `_pad0`, `shadow_slot` (engine-written) and `_pad1`,
  and hosts aggregate-initialize positionally straight across them.
  `PointLightDesc { glm::vec3 position; float radius; Color color; float intensity; bool casts_shadows; }`,
  `SpotLightDesc { …; Degrees inner_angle; Degrees outer_angle; }` with the cosine taken
  inside `add_spot_light`.
- New `scene/frame_sink.hpp`: `class FrameSink` holds a `FrameBuilder*` and exposes **only**
  the host surface (draw/draw_skinned/alloc_bone_palette/add_*_light/add_billboard_3d/
  add_debug_*/spawn_effect/ui()/lane()/set_camera/set_environment/set_draw_*).
  `SimFrame::frame()` returns `FrameSink&`. The engine tail — `seal`, `set_frame_timing`,
  `alloc_effect_snapshot`, `set_engine_frame`, `for_each_effect_spawn`,
  `has_staged_environment`, `staged_environment`, `packet()` (`frame_builder.hpp:296-370`) —
  goes private. **Friends are `bracken::Application` *and* `bracken::systems::EffectSystem`**:
  `alloc_effect_snapshot` is called from `effect_system.cpp:331`, not from `Application`, and
  every surveyed proposal got this wrong and would not have compiled. Same treatment for
  `FramePacket::begin_fill/begin_consume/reset/reserve_block/take_palette_slot`
  (`frame_packet.hpp:463-495`). This deletes the prose contract at `sim_frame.hpp:85-88` —
  today a released game calling `seal()` early corrupts the frame with no diagnostic,
  because the only guard is a `BRACKEN_ASSERT`.
- **New `bracken/testing/frame_test.hpp` — a public test fixture for game developers.**
  `FrameBuilder`/`FramePacket` are the host's entire output surface and are perfectly
  testable headlessly (`scene_prep_parity.test.cpp` builds synthetic sealed packets with no
  GPU), yet nothing ships that capability, and this phase makes *more* of the surface
  private. The fixture gives a host: a `TestFrame` that constructs a packet + builder,
  `sink()` to hand to their own frame code, `seal()`, and typed read-back
  (`draws()`, `lights()`, `ui_text()`, `digest()`). For a goal stated as "ergonomic interface
  for game developers", being able to unit-test your own frame code is worth more than any
  single enum retype.
- `frame_packet.hpp:30-35` claims "no field references host storage, so a sealed FramePacket
  carries no lifetime contract whatsoever". That is **false**: `EffectSpawnCommand::definition`
  (`:136-142`) and `EffectEmitterSnapshot::definition` (`:155-163`) hold host pointers
  retained across frames (`effect_system.hpp:56, :84`). This phase does the honest 20 lines —
  rewrite the comment to state the exception prominently, add `BRACKEN_CONTRACT` null checks
  at consume, reference the deferred `EffectLibrary` work — and does *not* attempt the ~350
  LOC handle migration (see §6).
- Docs: `docs/architecture.md`'s FrameBuilder API table, `docs/user_guide.md`'s frame section.

**Gate**

```bash
cmake --build build && ctest --test-dir build --output-on-failure
# frame_builder.test.cpp drives the builder through the friend fixture (proves the escape hatch)
# a compile-only TU constructs a FrameSink and confirms the engine tail is unreachable
build/examples/playground/playground --smoke        # packet_digest EXACTLY unchanged
build/examples/playground/playground --smoke-sweep
```

`DrawFlags` is a pure re-encoding of the same two booleans, so **any digest movement means a
flag was transposed during the mechanical conversion** — precisely the bug class this phase
exists to make impossible. Before Phase 2 this was undetectable.

**Rollback:** four commits — DrawFlags; Color; light descriptors; FrameSink + test fixture.

---

### Phase 9 — Host lifecycle: delete the ceremony

**Serves:** ergonomics. Five of the eleven entries in the engine's own "Common mistakes"
list (`docs/user_guide.md:587, :705, :707, :709, :710`) are ceremony the engine could own.

**Changes**

- **Boot.** `Application::start()` (public, calls `on_init()`) and `Application::stop()`
  (public, calls `on_shutdown()`), or equivalently have `run()` drive them. Deletes the
  hand-written wrappers at `playground.hpp:47-49` whose comment reads *"Public wrappers so
  main() can drive the protected lifecycle hooks"*, and deletes
  `docs/user_guide.md:705` (mistake #1). **Not** a `run_game(Application&, AppConfig)`
  free function: that hides the boot order and duplicates parameters `init_engine` already
  takes, to save three lines in `main.cpp`.
- **Input.** `InputSnapshot` gains `std::bitset<SDL_SCANCODE_COUNT> key_held` +
  `bool key_held(SDL_Scancode) const`, filled in `InputHandler::snapshot()` from the polled
  keyboard state. `snapshot()` also consumes the mouse edges it copies — it is already
  non-const and already drains the text buffer (`input_handler.hpp:106-110`) — with an
  explicit `retain_mouse_edges()` escape hatch for UI layers. Deletes
  `input().clear_mouse_edges()` from the host contract (`playground.cpp:2079`), the
  hand-tracked shift bool and its documented latching bug (`playground.cpp:1110-1123`,
  `playground.hpp:238-242`), and `docs/user_guide.md:587, :707`.
- **Camera, tier-correctly.** `Application` already owns the camera
  (`application.hpp:388`, created at `application.cpp:363`) yet the host must call
  `set_screen_size` (`playground.cpp:1693`), `update(dt)` (`:1723`), `fb.set_camera(...)`
  (`:1899`) and `set_camera_forward(...)` (`:1727`) every frame. The engine now does all four
  in the sim-stage tail. **The viewport must not be read from sim**: `screen_width()` is
  `context_ ? context_->width() : 0` (`application.cpp:461-463`), and `RenderContext` is
  main-thread-tier state that may be reconfiguring the swapchain while frame i+1's sim runs
  under the depth-2 pipeline. Instead `kick_sim` — which already computes frame timing on the
  main thread (`application.cpp:112-121`) — snapshots the viewport into the `SimFrame`, and
  the tail uses that value. `set_camera_forward(float,float)` becomes
  `set_camera_forward(glm::vec2)` and an *override* rather than a requirement.
  `Application::set_camera_driven_by_engine(bool)` (default true) for custom rigs.
- **Settings.** `Application::edit_graphics_settings()` returns a move-only
  `GraphicsSettingsEdit` (`operator->` for mutation; destructor calls
  `request_graphics_settings` with the edited copy, no-op if unchanged, reusing the existing
  diff at `application.cpp:507-524`); `graphics_settings()` becomes const-only. Deletes the
  `gs_dirty` dance (`playground.cpp:1145-1178`) and the eight `apply_all_graphics_settings(gs)`
  resends (`:1454-1520`) and `docs/user_guide.md:710`.
- **Heightmap.** `Heightmap::from_floats(std::span<const float>, uint32_t resolution,
  WorldBounds)` owning the min/max scan and the uint16 packing the host currently
  reimplements (`playground.cpp:400-401`); `get_normal_world` returns `glm::vec3` instead of
  three out-params (`heightmap.hpp:63-83`) in a codebase that is glm everywhere;
  `bool valid() const` (resolution ≥ 2, data size matches). `Application::set_heightmap`
  (`application.cpp:421-423`) rejects an invalid map with a logged error instead of
  installing a world where `resolution - 1` underflows and every query silently returns 0.
- **Model bounds.** Replace `Model`'s six loose floats `min_x..max_z`
  (`model_loader.hpp:129-152`) with `struct AABB { glm::vec3 min, max; }`.
  `ModelManager::register_model` (`model_loader.cpp:1024-1035` — validates nothing today, and
  silently orphans a previous handle on name collision) derives the AABB and bounding sphere
  from mesh vertices when unset, sets `loaded` itself, and returns
  `core::Result<ModelHandle, ModelRegisterError>{ EmptyModel, NameAlreadyRegistered,
  UploadFailed }` — the first real user of `Result` in the engine. Deletes
  `character.cpp:88-97`'s `finalize_bounds` and its seven call sites (`:295, :313, :325, :335,
  :371, :381, :393`); forgetting `compute_bounding_sphere()` today yields
  `bounding_half_diag == 0` and silent frustum-culling into invisibility.
- Delete `Mesh::near_only` (`model_loader.hpp:101-103`, documented "Not yet enforced") rather
  than ship a hint the draw walk ignores. Note the removal in `CLAUDE.md`'s gotchas.

**Gate**

```bash
cmake --build build && ctest --test-dir build --output-on-failure
# heightmap.test.cpp (438 lines, well covered) gains a from_floats round-trip asserting
#   bit-identical uint16 output vs the playground's old hand-rolled quantization
# input_handler.test.cpp gains key_held + snapshot-consumes-edges coverage
build/examples/playground/playground --smoke        # packet_digest
build/examples/playground/playground --smoke-sweep
build/examples/playground/playground --smoke --engine-camera=off   # escape hatch still works
```

The camera change is the only item here that can move the digest: the smoke run force-settles
the camera springs with 120 fixed steps (`playground.cpp:1705-1727`), so staging order is
directly exercised. If the digest moves, inspect the camera state field-by-field before
re-baselining — do not re-baseline first.

**Rollback:** five commits — boot; input; camera; settings-edit; heightmap + model bounds.

---

### Phase 10 — Performance, each item gated on its own benchmark

**Serves:** performance. **Read the honest framing first:** the total CPU saving available
here is on the order of 100–300 µs per frame. The smoke run reports 105 fps with 542 draw
calls and 6.2M triangles, and until Phase 4's `present_wait_ms` says otherwise, nobody knows
whether this engine is CPU-bound at all. Every measurement in the survey was taken against a
`-O0` tree. **Run the benchmarks in `build-bench/` first and drop any item whose gate fails.**

**Changes, each with its gate stated up front**

- **10a — Batched job dispatch.** `job_system.cpp:386-401` does
  `epoch_.fetch_add(1, release); epoch_.notify_one();` per submitted node, and `parallel_for`
  submits one node per chunk (`parallel_for.hpp:61-67`). Measured 175 ns per wake with
  workers parked. Add `JobSystem::submit_batch(TaskGroup&, N, make_body)` enqueueing all N
  then doing a single `fetch_add(N, release)` + notify — which satisfies the documented
  publish-before-bump protocol (`job_system.cpp:394-398`) trivially, since all publishes
  precede the bump. Rewrite `parallel_for` onto it.
  *Gate:* `BM_ParallelForDispatch` (added in Phase 5's benchmark set) must improve ≥20 % µs/chunk
  at ≥32 chunks, at 4 and 8 workers. Otherwise revert. TSan lane mandatory.
- **10b — SoA model bounds, with a validity bit.** Add
  `std::vector<glm::vec4> bounds_` to `ModelManager` (`center.xyz`, `half_diag` in `.w`),
  written alongside `compute_bounding_sphere()`, plus `model_bounds(ModelHandle)`.
  **`.w < 0` means "no bounds", and the call sites keep their existing three-way branch** —
  `scene_renderer.cpp:1756-1764` leaves a bounds-less command *visible*, and `:1843-1849`
  computes `casts = !have_bounds || intersects(...)`, so a sentinel radius would silently
  invert both. `PrepParams` (`scene_renderer.hpp:269-280`) carries
  `std::span<const glm::vec4>`; `resolve_skinned_chunk` keeps its `Model*` lookup for the
  skeleton. Rationale: `sizeof(Model)=184` with `bounding_center` at offset 168, so the 16
  bytes the cull wants sit in the last cache line of a separately heap-allocated object,
  chased once for the main view and once per cascade.
  *Gate:* `BM_PrepInstanceBatches` at 2048/16384 commands must improve ≥15 %, **and** the
  Phase-5 prep goldens (which include an unregistered-handle command in both the main and
  shadow arms) must be byte-identical. `scene_prep_parity.test.cpp` alone is **not** a valid
  gate here: it compares across worker counts and this change moves all arms identically.
- **10c — Terrain double-cull, with a separate scratch.** `draw_culled_tiles`
  (`terrain_renderer.cpp:730-737`) culls into the **shared member** `tile_runs_scratch_` and
  is called from the forward pass (`:581`), the prepass (`:789`) **and every shadow cascade**
  (`:835`) — and cascades record *before* the prepass, so caching the main-view list into
  that member would be clobbered. Add a distinct `main_view_runs_` member populated once from
  the Prepare stage, consumed by `render()` and `render_prepass()` only when the frustum
  pointer matches; cascades keep using `tile_runs_scratch_` against their own light frusta.
  The forward pass depth-tests EQUAL against the prepass (`terrain_renderer.cpp:780-781`), so
  a stale list drops tiles and breaks that invariant.
  Also promote the per-call `std::vector<RunPartial>` / `RowPartial` pools
  (`terrain_renderer.cpp:686-689`, `grass_renderer.cpp:213-216`) to grow-only members
  mirroring `PrepViewScratch`'s documented contract (`scene_renderer.hpp:303-320`), keeping
  the `alignas(k_cache_line)` padding.
  *Gate:* `--smoke` digest **and** `draw_calls` unchanged; `--smoke-sweep` unchanged (cascade
  counts vary there, which is exactly what a shared-scratch bug breaks).
- **10d — `MotionHistory` open addressing.** Replace the two `std::unordered_map`s
  (`motion_history.hpp:75-76`) with `keys_`/`values_` + a per-slot generation stamp so
  `end_frame()` is a bump instead of a node-destroying clear. Public API byte-identical.
  *Gate:* `BM_MotionHistoryRecordFind` improves; `motion_history.test.cpp` unchanged; digest
  unchanged.

**Deferred to a later, separately-approved pass, with the gate they'd need:** `ClusterGrid`
bin flattening + parallel pack tail, `JobSystem` inline callable storage (survey measured it
within noise: 0.301 vs 0.375 µs/chunk), particle instance-pack fan-out, single-copy light
upload, the cluster-bounds corner-ray hoist (guarded at `light_cluster.cpp:200-206`, so it is
*not* a per-frame cost — ~50 µs on FOV/draw-distance changes only). Each needs its own
benchmark gate; none is worth a scheduled phase today.

**Gate (whole phase)**

```bash
cmake --preset bench && cmake --build build-bench
build-bench/src/bracken/bracken_benchmarks --benchmark_filter='Prep|ParallelFor|Motion'
# before/after at BRACKEN_WORKERS=0,4,8; an item that does not move its benchmark is reverted
cmake --build build && ctest --test-dir build --output-on-failure
build/examples/playground/playground --smoke && --smoke-sweep    # digests unchanged
cmake --build build-tsan && ctest --test-dir build-tsan --output-on-failure   # required for 10a
```

**Rollback:** one commit per item, each independently revertible. Items are ordered
10a→10d and none depends on another.

---

### Phase 11 — Public surface and documentation

**Serves:** ergonomics, self-description. Lands **last**, after every API break, so the
surface it publishes is the final one.

**Changes**

- New `src/bracken/bracken.hpp`: an umbrella including exactly the public subset — the 24
  headers the reference consumer actually uses — grouped by section, one comment line per
  header. The preamble says *"this is the supported surface"*, **not** a stability promise:
  publishing a stability contract in the same release that changed six of these signatures
  would be signalling, not ergonomics.
- Move `physics/jolt_job_system.hpp` → `physics/internal/jolt_job_system.hpp`. It includes
  `Jolt/Jolt.h` (`:16`) and declares `class JoltJobSystem final : public JPH::JobSystemWithBarrier`
  (`:52`) inside the PUBLIC include tree; the documented invariant
  (`architecture.md:33, :654`, `CLAUDE.md:106`, `README.md:73`) holds only because Jolt is
  linked PRIVATE. Update `physics/jolt_job_system.cpp:1`, `physics/physics_world.cpp:7`,
  `physics_world.hpp:18`. Adopt `<module>/internal/` as the convention.
- New `public_headers.test.cpp`: no header outside an `internal/` directory may transitively
  include `Jolt/`, `imgui`, `tiny_gltf` or `stb_`; every header in `bracken.hpp` compiles
  standalone. Converts four hand-maintained doc invariants into a build check. (Honest note:
  this test passes on day one except for the header being moved — its value is preventing
  regression, not finding a bug.)
- `add_library(bracken::bracken ALIAS bracken)` after `src/bracken/CMakeLists.txt:106`;
  update the three consumption snippets (`CLAUDE.md:276`, `README.md:215`,
  `docs/user_guide.md:167`).
- The effects triad (`effect_definition.hpp` + `systems/effect_system.*` +
  `render/effect_renderer.*` → `src/bracken/effects/`) **has moved to Phase 6b**, where it
  has a structural reason rather than a tidiness one: it removes the `systems → @root` header
  edge and makes `effect_renderer → effect_system` intra-module. Nothing to do here beyond
  citing it in the docs. Together with 6b's `contracts/` layer these are the plan's only
  module moves; `input/` and `model/` remain rejected (see §6).
- **Phase 6b is a hard prerequisite for the umbrella, not a nicety.** Before 6b,
  `bracken.hpp` cannot honestly claim a public subset: `application.hpp` transitively exports
  `gpu/gpu_buffer.hpp`, `gpu/gpu_texture.hpp`, `gpu/gpu_types.hpp` and
  `animation/animation_types.hpp` through the chain
  `application.hpp:5 → sim_frame.hpp:7 → scene/frame_builder.hpp → scene/frame_packet.hpp:5 →
  model_loader.hpp`, for the sole purpose of naming `ModelHandle`. 6b's `ModelHandle` hoist
  cuts that chain; `public_headers.test.cpp` below should additionally assert that no header
  reachable from `bracken.hpp` pulls `gpu/gpu_buffer.hpp` or `gpu/gpu_texture.hpp` unless it
  is itself a `gpu/` header.
- New `docs_coherence.test.cpp` — **anchored, not line-numbered.** The survey's version
  asserts every `path:NNN` citation resolves and identifiers match within ±5 lines, which
  institutionalizes line numbers at exactly the moment they should be abandoned and turns
  every ordinary edit into a doc chore enforced by a red suite. This version asserts:
  (a) every cited **path** exists; (b) every cited **identifier** appears somewhere in the
  cited file (grep, not line); (c) every directory under `src/bracken` containing a `.hpp`
  has a row in `architecture.md`'s module map; (d) every `.hpp` uses only `bracken/`-rooted
  or third-party quoted includes. Line numbers are printed as an advisory drift report, not a
  failure. Implemented in C++ (`CLAUDE.md:200-201`: no Python tooling in this repo).
- Repair the 20 verified-stale citations (`technical_spec.md:77,78,79,80,81,83,85,93,107,108,
  109,115,131,139,165,239,249,1040,1047,1048`); add the missing `dev/` row and
  `SimFrame`/`MainFrame` to `architecture.md:16-36, :20`; fix "as of SDL 3.2" → 3.4 at
  `docs/user_guide.md:2156` and `gpu/gpu_debug.hpp:32` (`CMakeLists.txt:2` requires ≥ 3.4.0).
- Rewrite `docs/user_guide.md`'s getting-started and per-subsystem sections against the new
  API, and **delete** the "Common mistakes" entries the API now makes impossible: `:587`,
  `:705`, `:707`, `:709`, `:710`, `:713`.
- `CLAUDE.md`: four build lanes, the public/internal convention, the enum'd settings, the
  diagnostics story, and the statement that benchmark numbers from a no-build-type tree are
  meaningless.

**Gate**

```bash
cmake --build build && ctest --test-dir build --output-on-failure
# public_headers.test.cpp + docs_coherence.test.cpp green
# a compile-only TU that includes ONLY <bracken/bracken.hpp> and subclasses Application
build/examples/playground/playground --smoke && --smoke-sweep    # digests unchanged
cmake --preset asan && cmake --build build-asan && ctest --test-dir build-asan --output-on-failure
```

The ASan/UBSan lane (a ~30-line copy of the `tsan` preset, added here) is the first pass over
the arena poisoning, the packet ring and the job system with a sanitizer that finds
use-after-free — which TSan does not.

**Rollback:** three commits — umbrella + alias; `internal/` move + header test; docs. The
effects move that used to live here is now Phase 6b's, with its own rollback.

---

## 5. Before / after — real host code

All "before" snippets are current code with file:line. This is the deliverable the user
cares most about: the diff in `examples/playground` **is** the ergonomics result.

### 5.1 Draw with two anonymous booleans → named flags
`examples/playground/playground.cpp:1949-1950`
```cpp
// BEFORE — which false is instancing and which is fog?
fb.draw_model(orb_model_, glm::translate(glm::mat4(1.0f), pos), {1.0f, 1.0f, 1.0f, 1.0f}, false, false,
              MOTION_ID_ORB_BASE + static_cast<uint32_t>(orb_index));

// AFTER
fb.draw_model(orb_model_, glm::translate(glm::mat4(1.0f), pos), Color::white(),
              DrawFlags::None, MotionId{MOTION_ID_ORB_BASE + orb_index});
```
Swapping the two `false`s today compiles silently and changes the image. Nothing catches it.
After Phase 2 + Phase 8, the packet digest does.

### 5.2 Spot light: positional init across engine padding, hand-computed cosines
`examples/playground/playground.cpp:1981-1982` — initializing a struct that also contains
`_pad0[3]`, `shadow_slot` and `_pad1` (`light.hpp:31-46`), all engine-written.
```cpp
// BEFORE
fb.add_spot_light({origin, 1400.0f, dir, std::cos(glm::radians(18.0f)),
                   {0.9f, 0.95f, 1.0f}, std::cos(glm::radians(32.0f)), 200000.0f, true});

// AFTER
fb.add_spot_light({ .position      = origin,
                    .radius        = 1400.0f,
                    .direction     = dir,
                    .inner_angle   = Degrees{18.0f},
                    .outer_angle   = Degrees{32.0f},
                    .color         = Color::rgb(0.9f, 0.95f, 1.0f),
                    .intensity     = 200000.0f,
                    .casts_shadows = true });
```
Reordering a field in `SpotLight` today silently changes the meaning of every positional
initializer in every game using the engine.

### 5.3 Boot: hand-written wrappers because the hooks are protected
`examples/playground/main.cpp:80-90` + `examples/playground/playground.hpp:47-49`
```cpp
// BEFORE — playground.hpp:
//   // Public wrappers so main() can drive the protected lifecycle hooks.
//   bool start() { return on_init(); }
//   void stop()  { on_shutdown(); }
// main.cpp:
playground::PlaygroundGame game(smoke_frames);
if (!game.init_engine()) { return 1; }
if (!game.start()) { game.shutdown_engine(); return 1; }
game.run();
game.stop();
game.shutdown_engine();

// AFTER — the wrappers in playground.hpp are deleted; start()/stop() are Application's.
playground::PlaygroundGame game(smoke_frames);
if (!game.init_engine()) { return 1; }
if (!game.start())      { game.shutdown_engine(); return 1; }
game.run();
game.stop();
game.shutdown_engine();
```
The win is the *deletion* of two host methods and of `docs/user_guide.md:705`
("Forgetting that `on_init`/`on_shutdown` are host-driven — nothing renders"), not a shorter
`main`. This plan deliberately does not hide the boot order behind a `run_game()` façade.

### 5.4 Settings: mutate the engine's struct, then hand it back
`examples/playground/playground.cpp:1145-1178` — eight toggles, all with the same tell.
```cpp
// BEFORE
auto& gs = graphics_settings();
bool gs_dirty = false;
if (in.was_key_just_pressed(SDL_SCANCODE_F2)) { gs.bloom_enabled = !gs.bloom_enabled; gs_dirty = true; }
if (in.was_key_just_pressed(SDL_SCANCODE_F4)) { gs.volumetric_fog = !gs.volumetric_fog;
                                                gs.god_rays = gs.volumetric_fog; gs_dirty = true; }
/* …six more… */
if (gs_dirty) { request_graphics_settings(gs); }

// AFTER — forgetting to commit is unrepresentable; the RAII handle commits on scope exit.
auto gs = edit_graphics_settings();
if (in.was_key_just_pressed(SDL_SCANCODE_F2)) { gs->bloom_enabled = !gs->bloom_enabled; }
if (in.was_key_just_pressed(SDL_SCANCODE_F4)) { gs->volumetric_fog = !gs->volumetric_fog;
                                                gs->god_rays = gs->volumetric_fog; }
/* …six more… */
```
Plus Phase 7 makes `gs->shadow_cascades = ShadowCascades::Four` mean four cascades, where
today `gs.shadow_cascades = 4` is silently clamped to `3` — which *also* means four, so the
mistake is invisible.

### 5.5 Derived model state the host must remember to compute
`examples/playground/character.cpp:88-97`, called at `:295, :313, :325, :335, :371, :381, :393`
```cpp
// BEFORE — host-side helper the engine has every input to do itself
void finalize_bounds(Model& model, const glm::vec3& min, const glm::vec3& max) {
    model.min_x = min.x; model.min_y = min.y; model.min_z = min.z;
    model.max_x = max.x; model.max_y = max.y; model.max_z = max.z;
    model.compute_bounding_sphere();
    model.loaded = true;
}

// AFTER — deleted. register_model derives the AABB and the sphere and sets `loaded`.
auto handle = models().register_model("character_body", std::move(model));
if (!handle) { BRACKEN_LOG_ERROR("game", "character body: {}", to_string(handle.error())); }
```
Forget `compute_bounding_sphere()` today and `bounding_half_diag == 0`: the model is
frustum-culled into invisibility with no message anywhere.

### 5.6 Terrain: the host implements 16-bit quantization by hand
`examples/playground/playground.cpp:400-401`
```cpp
// BEFORE
float normalized = (h - heightmap_.min_height) / (heightmap_.max_height - heightmap_.min_height);
heightmap_.height_data[z * RES + x] = static_cast<uint16_t>(normalized * 65535.0f);

// AFTER — the host produces floats; the engine owns the scan and the packing.
heights[z * RES + x] = h;                       // …loop as before, in world units
set_heightmap(Heightmap::from_floats(heights, RES, world_bounds));
```
And `Application::set_heightmap` (`application.cpp:421-423`, which validates nothing today)
rejects an invalid map instead of installing a world where `resolution - 1` underflows and
every terrain query silently returns 0.

### 5.7 Mouse edges the host must clear every fence
`examples/playground/playground.cpp:2073-2079`
```cpp
// BEFORE — with a nine-line comment explaining why the host has to do this
input().clear_mouse_edges();

// AFTER — deleted. InputHandler::snapshot() consumes the edges it copies,
//          exactly as it already drains the text buffer (input_handler.hpp:106-110).
```
Along with the hand-tracked Shift bool at `playground.cpp:1110-1123` whose own comment
(`:1116`) documents a latching bug — replaced by `in.key_held(SDL_SCANCODE_LSHIFT)`.

---

## 6. Explicitly rejected

Direct, with reasons. Several of these were in every surveyed proposal.

**The rule the structural rejections below apply, stated once.** "Logical components" is a
property of the dependency graph and of what owns which state — it is not a property of
directory names. So a proposed move is accepted when it changes an **edge** and rejected when
it changes only a **path**. Under that rule this plan does serve the goal, through four
changes of shape: Phase 6 turns the renderer's only thread-tier boundary into a type that
structurally cannot violate it (`ScenePrep`) and gives the pass-selection tree a name and a
test (`RenderPlan`); Phase 6b makes the module header graph acyclic and writes the layering
down as a checkable fact for the first time; Phase 6c separates a registry from the 1,215-line
catalogue welded into it; Phase 8 splits the host-facing draw/light surface from the GPU-layout
PODs it is currently aliased onto. Two of those four do involve moving files — because moving
the file *is* the mechanism for the edge cut, and each phase names the edge. Everything below
that does not name an edge is a rename, and a rename is churn.

**Rejected outright**

- **C++20 modules.** 57 engine TUs against SDL3/glm/Jolt/tinygltf/ImGui/GoogleTest, none of
  which ships a module interface, and BMI cache invalidation collides head-on with
  `.claude/RULES.md`. The measured compile-time lever is a PCH, not modules. Recorded as a
  closed question in `docs/engine_design_notes.md`.
- **`std::ranges` / `views::` inside the Prepare fan-out or `EffectSystem::update_t`.**
  The bit-identical-at-any-worker-count guarantee is enforced by chunk-index-ordered merges
  and protected by `scene_prep_parity.test.cpp`; a `transform | filter` pipeline produces the
  same values but is not obviously order- and accumulation-stable *to a reviewer*, and the
  reviewer is the safety mechanism. Engine-wide there are 9 std-algorithm sites outside
  tests. Permitted in setup, tooling, serialization and dev-UI code. Written as a dated
  decision so the next pass does not re-derive it.
- **glm include narrowing and pimpl-ing SDL out of `gpu_device.hpp` for compile time.**
  Measured: glm vec2+3+4+mat4x4 is 52,431 preprocessed lines vs 54,725 for the monolith — 3 %
  for days of work; `SDL3/SDL_gpu.h` is 5,155 lines, cheaper than `<mutex>`. The cost is
  `<format>` (44,374 lines) and glm's core machinery, neither removable by include hygiene.
- **Shrinking `gpu::InstanceData` and splitting `PrepassModelUniforms`.** The survey's
  arithmetic is wrong: 64+64+16+16+64 = 224, and replacing `normalMatrix` with three `vec4`
  rows plus folding `noFog` gives **192, not 144**, so the stated bandwidth win is ~33 %
  overstated (4.5 → 3.8 MB/frame at 20k instances, not 4.5 → 2.9). These are the two highest-
  risk edits in any proposal — they must touch `gpu_uniforms.hpp` and three-to-four `.hlsl`
  files atomically, the failure mode is wrong lighting or wrong motion vectors rather than a
  crash, and every image gate we have is Metal-only. Revisit only after Phase 5's SPIR-V
  reflection test has been proven to catch an injected one-sided edit, and then only with a
  benchmark isolating upload cost.
- **A homemade HLSL `cbuffer` text scanner.** It would have to reimplement HLSL packing,
  array stride and matrix majorness as DXC emits them — brand-new untested parsing gating the
  failure `CLAUDE.md` names as the most common silent breaker, plus a third hand-maintained
  copy of the layout table. Phase 5 uses SPIR-V reflection over the already-compiled outputs,
  which is ground truth and reuses `gpu_shader.cpp:125-145`.
- **`PanelMirror<T>` + `Edited<T>` + `DevUI::add_panel`.** ~200 LOC of new template machinery
  plus a cross-thread exchange at the frame fence (requiring its own TSan pass) to delete 64
  lines of playground code, generalized from a single client. `dev_panels.hpp` already binds
  `settings` + an `apply_settings` callback.
- **`run_game(Application&, AppConfig)`.** Hides the boot order and duplicates parameters
  `init_engine` already takes, to save three lines in `main.cpp`. Phase 9 makes
  `start()`/`stop()` public instead — same deletion in the host, no façade.
- **The ~55 `const std::string&` → `std::string_view` sweep + 8 transparent-hasher retypes.**
  No measured benefit is claimed for either, and it re-touches headers that three other phases
  are already editing. The `std::span` half (`animation_types.hpp:188,:229`,
  `animation_player.hpp:25,:85`, and the container-leaking getters at `light_cluster.hpp:97-101`,
  `local_shadows.hpp:126`, `ui_renderer.hpp:145`) has real decoupling value and should be
  picked up opportunistically when those files are open — vectors convert implicitly, so no
  call site changes.
- **Deducing-`this` sweeps.** Eight of the twelve pairs are one-line forwarders where the
  transform costs readability. `HandlePool::get` has **zero users** outside its own test
  (`grep -rn HandlePool src examples` excluding `handle.hpp`/`.test.cpp` returns 0), so
  "modernizing" it is work on nearly-dead code. `AssetRegistry::entry_at`'s
  `const_cast<AssetRegistry*>(this)` (`asset_registry.hpp:69`) *is* a genuine
  const-correctness hole and is a ~10-line fix the next time that file is opened — not a phase.
- **Moving the loose top-level files into `input/` and `model/` directories.** *Reaffirmed,
  and now with the dependency-graph evidence that was missing when this was first written —
  which strengthens it rather than reversing it.* The test a path move has to pass is not
  "does this directory read better", it is **"does this change an edge in the module graph".**
  Applying that test file by file:
  - `input_handler.hpp` + `input_state.hpp`: fan-in from `@root` only
    (`application.hpp:4`, `sim_frame.hpp:6`, `input_handler.hpp:3`), fan-out 0 into any
    module. They are already a leaf group sitting next to the only thing that uses them.
    An `input/` directory changes 9 source references and 27 doc mentions and **zero
    edges**. Rejected, more firmly than before.
  - `model_loader.hpp` + `model_utils.hpp`: the survey is right that this one is *different*
    in kind — 8 production includers across **4 modules** (render, scene, procedural, @root),
    transitively reached by 32 TUs across 6 modules, and its position at the root is why
    `@root` has fan-in 5 and appears in the SCC at all. The original wording here ("two-file
    groups that gain nothing but a path change") was **factually wrong about this file** and
    is corrected. But the conclusion still stands, for a better reason: Phase 6b establishes
    that after the `ModelHandle` hoist there is **no `@root → render` or `@root → scene`
    header edge**, so `render/effect_renderer.hpp:8 → model_loader.hpp` and
    `render/terrain_renderer.hpp:8 → heightmap.hpp` do not close a compile cycle. A `model/`
    directory would therefore change 20 doc path citations, 78 doc mentions and 18 source
    references to fix a `.cpp`-only, link-level module cycle that costs nothing. **Rejected
    on cost/benefit, not on principle** — if `Application` is ever split up, revisit.
  - `heightmap.{hpp,cpp}` (8 doc paths / 104 doc mentions / 167 source references) and
    `render_stats.hpp` / `graphics_settings.hpp`: the last two **do** move, in Phase 6b, and
    the reason is precisely that they cut `scene → @root`. `heightmap` does not, because
    after that cut it no longer closes anything and it is the most expensive file in the tree
    to move.
  The general measured churn stands: `bracken/gpu/` appears on 125 doc lines,
  `bracken/core/` 101, `bracken/render/` 89, `bracken/scene/` 72 — no wholesale module
  renaming is contemplated. What moves in Phase 6b is 7 pure-data leaf headers at a measured
  54 include lines and ~50 doc citations, plus the effects triad; every one of those is the
  mechanism for a specific edge cut, and each edge is named in the phase.

- **Reorganizing `scene/` — splitting the frame contract from the frame orchestrator.**
  This is the one boundary in the tree that is arguably *wrong* rather than merely
  differently-named: `scene/` holds both `FramePacket`/`FrameBuilder`/`PacketRing` (bottom of
  the stack, produced by the host, consumed by every renderer) and `SceneRenderer` (top of the
  stack, drives all 16 sub-renderers), which is why `scene` is the only module with out-degree
  8 *and* in-degree 5. Not scheduled: after Phase 6b, `scene_renderer.hpp` includes no
  `render/` header and the 16 sub-renderer includes are `.cpp`-only, so the defect costs
  conceptual clarity and **zero compile coupling**. Splitting it means a third new directory,
  a namespace decision on `bracken::scene::SceneRenderer` (206 doc lines mention the type),
  and it lands on top of Phase 6, which is already rewriting all 53 doc references to
  `scene/scene_renderer.cpp`. Raised as §7 Q7 so the owner decides rather than the plan
  deciding by omission.

- **A whole-tree "make every module acyclic including `.cpp` edges" pass.** Phase 6b makes
  the *header* graph a DAG, which is the graph that governs compile coupling and rebuild
  amplification. Chasing the residual `.cpp`-only module cycles (all of which run through
  `application.cpp`, `dev/dev_panels.cpp` and `scene/scene_renderer.cpp` reaching downward
  into the modules they orchestrate) would mean relocating `Application` itself. A `.cpp`
  cannot participate in a compile cycle; "acyclic at link level" is not a property worth
  buying at that price.
- **Renaming public types.** `SceneRenderer` appears on 206 doc lines, `Application` 198,
  `JobSystem` 81, `FrameBuilder` 81, `FramePacket` 66. The names are accurate; the ergonomics
  problem is the *shape*.
- **Mass `bool` → `Result` conversion.** Forbidden by `CLAUDE.md` and settled at
  `docs/audits/RESOLUTIONS.md:70`. Phase 9 introduces `Result` for exactly one genuinely new
  fallible API (`register_model`). The boot chain (`RenderContext::init`, `SceneRenderer::init`,
  `Application::init_renderer`) keeps `bool` + log — Phase 3 fixes the actual complaint, which
  is that those failures never reach a host sink.
- **Wiring `render_graph/` into the frame path.** Deliberately unwired per the design notes;
  routing a working clustered-forward renderer through an unproven scheduler mid-refactor
  destroys the digest gate. See §7 Q4 for what to do with it instead.
- **Raising `MAX_BONES`, `MAX_BONE_PALETTES`, or the CSM/cluster constants.** Nothing here
  needs it; `MAX_BONES=128` sizes `std::array<glm::mat4, MAX_BONES>` members across the
  animation path. Phase 4 makes the budgets *observable*, which is the actual complaint.
- **`std::print` for the log sinks.** `logger.cpp:58` routes through `SDL_LogMessage`;
  `std::print` would bypass SDL's plumbing and re-split the output stream Phase 3 just
  unified.
- **A GPU-timestamp implementation.** SDL3 has no timestamp query API (`gpu_debug.hpp:31-33`).
  No amount of engine work produces real GPU-side numbers today. Phase 4 makes CPU-record
  timings real, labels them honestly, adds `present_wait_ms` as the CPU/GPU-bound signal, and
  keeps pointing at RenderDoc/Nsight.
- **Line-number-keyed doc coherence tests.** Phase 11 checks paths and identifiers, not lines.

**Deferred with a named trigger, not rejected**

- `ModelHandle` → generation-checked `core::Handle` (~260 LOC, touches every stored handle in
  the playground). Phase 4's `draws_dropped_unknown_model` + throttled warn captures most of
  the diagnostic value at a fraction of the risk. Trigger: owner approval as a standalone
  phase — see §7 Q3.
- `EffectLibrary` / handle-ising the packet's effect pointers (~350 LOC on the packet
  contract plus the playground's effect authoring). Phase 8 does the honest 20 lines instead.
- `core::Result` → `std::expected` — see §7 Q2; genuinely zero-benefit-today, and the
  "source-compatible shims" the survey promised do not exist (`core::Err` is a struct with a
  public `.value`; `std::unexpected` exposes `.error()`; `is_ok`, `is_err`, `map` and
  `Result<void>::and_then` have no direct equivalents).
- `clang-tidy` as a blocking gate. `.clang-tidy` (105 lines) and `.clang-format` (87) exist
  and are referenced by no build file and no doc line; enabling `modernize-*`/`bugprone-*`
  across 43k LOC mid-refactor generates a backlog that competes with the phases. Add
  `BRACKEN_ENABLE_CLANG_TIDY` (default OFF) and a `bracken_format` target as a standalone
  chore after Phase 11.
- A precompiled header (measured 31 % off full compile time). Real, but a PCH containing
  `<format>`, `<span>`, glm and `SDL_gpu.h` **masks missing includes**, and Phases 1, 6 and 11
  all move includes and files. Land it *after* Phase 11, and gate it on
  `BRACKEN_ENABLE_PCH=OFF` also compiling clean — the configuration every consuming game
  uses, which no proposal verified.

---

## 7. Open questions for the owner

**Q1 — How much API breakage is acceptable in one pass?**
Phases 4, 7, 8, 9 and 11 all break the host API: `RenderStats` types, seven settings fields,
`draw_model`/light signatures, `SimFrame::frame()`'s return type, `Model` bounds, header
paths. The playground is the only consumer in this repo and is fixed in each phase.
**Recommendation: take all of it, in this order.** Every break is compiler-caught, and
staggering them across releases means paying the migration cost twice. If you have external
consumers, tell me and I will add a deprecation layer to Phases 7–9 (roughly +200 LOC of
inline forwarders).

**Q2 — `core::Result` → `std::expected`?**
`__cpp_lib_expected == 202211` on the configured toolchain, and there are **zero** call sites
outside `core/result.test.cpp`, so migration cost is zero today and rises permanently with
every `Result`-returning API. The benefit today is also zero, and `value()` changes from
`BRACKEN_VERIFY`-abort to `throw bad_expected_access`.
**Recommendation: do it, in Phase 9, right before `register_model` becomes the first real
user — and delete `Ok`/`Err`/`is_ok`/`map` rather than pretending to shim them.** ~120 LOC
net deletion. If you prefer keeping the abort-on-`value()` semantics, say so and we keep the
hand-rolled type; that is a legitimate engine-style call.

**Q3 — Generation-checked `ModelHandle`?**
`model_loader.hpp:31` is `using ModelHandle = uint32_t;` — a bare index. After `unload_all()`
a stale handle either vanishes (7 silent `if (!model) continue;` sites) or resolves to a
*different* model, and both are multi-hour bisects. `core::Handle`/`HandlePool`
(`handle.hpp:65-77`, 153 lines of tests) already implements exactly this and has one user.
**Recommendation: yes, but as a standalone phase after Phase 11**, not inside this plan — it
touches every stored handle in the playground and collides with Phases 8 and 10 on the same
files. ~260 LOC.

**Q4 — What happens to `render_graph/`?**
1,095 LOC, tested, maintained, deliberately not in the frame path, and it costs build time
and doc coherence on every phase. Carrying a dead subsystem through a full-scale refactor is
the worst of the three options.
**Recommendation: delete it.** Git history is the archive; `docs/engine_design_notes.md`
already records *why* it is not wired, and that rationale is what has value. If you intend to
wire it within a year, the alternative is moving it to a separate CMake target excluded from
`bracken` so it stops being a dependency of the shipped library.

**Q5 — Is performance actually a goal, and against what target?**
Every CPU item in Phase 10 totals ~100–300 µs/frame. The smoke run does 105 fps with 542 draw
calls and 6.2M triangles on Metal, and until Phase 4's `present_wait_ms` lands, nobody knows
whether this engine is CPU-bound at all. The real levers — LOD, occlusion culling,
material-batched draw submission — are feature work, not refactoring, and none of the three
surveyed proposals contained one.
**Recommendation: land Phase 4's `present_wait_ms` + stage timing, measure for a week, and
then decide whether Phase 10 is worth doing at all.** If the answer is "GPU-bound", cut
Phase 10 entirely and open a separate LOD/culling project. I would rather delete a phase than
ship a bit-identical frame that is 200 µs cheaper on a thread that was already idle.

**Q6 — Metal-only validation. Acceptable?**
Every image and digest gate in this plan runs on one driver on one machine (`driver=metal`).
The packet digest is device-independent and therefore fully portable, but anything downstream
of pipeline state or uniform packing — which is exactly the class Phase 5's reflection test,
Phase 6c's pipeline table and the rejected uniform repacks live in — is validated for Metal
only.
**Recommendation: accept it for this plan** (the digest carries most of the weight), and
treat a Vulkan smoke run on a Linux box as a separate, worthwhile follow-on. If you have a
Windows/Vulkan machine available, add `--smoke` + `--smoke-sweep` there as a manual
pre-merge step for Phases 6, 6c, 8 and 10, and say so now so I can make the goldens
per-backend rather than global. Note that Phase 6c raises the value of this: the swapchain-
format resolver it introduces is the one piece of the pipeline table whose behaviour is
plausibly backend-dependent.

**Q7 — Phase 6b's leaf module: what is it called, and does `scene/` get split too?**
Two decisions, both taste, both cheap to make now and annoying to change later.
*(a) The name.* Phase 6b proposes `src/bracken/contracts/` for the 7 pure-data headers
(`EnvironmentState`, `PointLight`/`SpotLight`, `CameraState`, `Frustum`, velocity math,
`GraphicsSettings`, `RenderStats`). `contracts/` matches this plan's thesis language but reads
oddly for `Frustum`. Alternatives: `types/` (bland, unarguable, describes exactly what is
there) or `frame/` (accurate for 5 of 7, wrong for `Frustum` and `GraphicsSettings`).
**Recommendation: `types/`.** It makes the leaf-module rule — *"anything in here includes
nothing but `core/`, glm and `<std>`"* — self-evident from the name, which is the property the
test enforces. Say the word before 6b lands; afterwards it is one `git mv` plus 54 include
lines, so this is a cheap decision, not a permanent one.
*(b) `scene/`.* It holds both the bottom-of-stack frame contract and the top-of-stack
orchestrator (§6). Splitting it is the last genuinely-wrong boundary in the tree, and Phase 6
is already rewriting every file and doc reference involved, so the marginal cost is near zero
*if done in the same pass* and meaningful if done later. **Recommendation: don't** — after 6b
it costs zero compile coupling, and a third new directory plus a namespace change on a type
named in 206 doc lines is a lot of motion for a diagram that is already correct on paper. But
if you want the module map to be literally readable top-to-bottom, this is the move, and the
only cheap moment to make it is inside Phase 6.
