# Native Multithreading Plan

The plan for rebuilding Bracken as a **natively multithreaded engine** — the
Frostbite model, not the "optionally parallel" model. One code path. The job
system is always on, the frame is pipelined by design, all frame data flows
through immutable per-frame snapshots, and the host-facing API is *simpler*
than today's, not more complex: a game developer writes straight-line frame
code and never sees a thread unless they ask for one.

This supersedes the earlier opt-in/serial-default posture and, deliberately,
the "default to serial" clauses of
[engine_design_notes.md](engine_design_notes.md) as they apply to threading —
that guidance was written for an engine with no parallel workload and no
mandate; the mandate now is a modern-engine architecture. When implementation
starts, amend the design notes to record this decision (the *other* lessons
there — measure, don't stack patches, keep the engine game-agnostic — still
govern every phase). Grounded in the 2026-07-18 code audit of the job system,
frame path, and CPU subsystems; file:line references are anchors from that
audit.

---

## 1. The model in one page

Frostbite's essential moves, mapped to Bracken:

1. **Everything is a job.** A fixed worker pool owns the cores (N = cores − 1,
   main thread helps during joins). Engine stages and game simulation code all
   run as jobs. There is no "serial mode" — debugging/bisection is
   `BRACKEN_WORKERS=1`, which runs the *same* code on one worker. No dual
   paths, ever: a parallel loop that gets one chunk at small counts is
   granularity control, not a second code path.
2. **The frame is a pipeline of stages** connected by immutable snapshots:

   ```
   frame k+1:  [ Simulate ]──────────────▶ FramePacket k+1
   frame k:              [ Prepare ]──▶ [ Record/Submit ]──▶ present k
   ```

   - **Simulate** — game logic, animation, IK, particles, effect spawning.
     Runs the host's frame code as a job (fanning out into more jobs). Output:
     a sealed **FramePacket**.
   - **Prepare** — view-dependent engine work: culling (main + cascades +
     local-shadow tiles), instance batching, cluster light binning, palette
     layout. Pure fan-out over the sealed packet.
   - **Record/Submit** — GPU command recording (eventually parallel across
     command buffers), submission, present. Swapchain acquire, ImGui, and
     submit live here, on the main thread.

   Simulate for frame k+1 overlaps Prepare/Record for frame k. Pipeline depth
   2 (one sim frame in flight); the packet ring makes deeper pipelining a
   constant, not a rewrite.
3. **Data flows one way, by value.** The host writes into a FramePacket
   through a builder; once sealed, the packet is immutable and the renderer
   consumes `const`. Today's lifetime traps — the non-owning
   `bone_matrices` span over live host storage, the renderer draining
   `RenderScene` through a non-const ref — are deleted, not worked around.
4. **The render graph earns its keep.** `render_graph/` (in-tree, tested,
   unwired) becomes the actual frame scheduler in the final render phase: sub-
   renderers declare passes, the graph topo-sorts, manages transients, and
   partitions recording across per-thread command buffers. Game developers
   never see it.
5. **Main thread = OS + GPU front-end only.** Event pump, input snapshot
   publish, swapchain, command submission, ImGui, FileWatcher callback fence.
   Everything else is pool work.

---

## 2. What the audit says we have and lack

**Have (keep, promote):**
- A complete work-stealing `JobSystem` with fork-join-safe `wait()`,
  exception propagation, and a strong test suite — never `init()`ed in
  production. It becomes always-on engine infrastructure.
- A proven data-parallel workload shape: `animation_parallel.test.cpp` shows
  256 `AnimationPlayer`s tick bit-identically under `parallel_for`
  (per-player cursors — the "clips shared, players not" lesson — is exactly
  the per-instance ownership the sim stage needs).
- Frame-N/N−1 double buffering precedents: bone-palette buffer flip,
  `MotionHistory`, `prev_view_proj_unjittered_`. The FramePacket ring
  generalizes this.
- The FileWatcher's worker-detect → main-thread-apply handoff — the template
  for streaming and hot-reload in the pipelined world.
- Thread-safe `Logger`; asserted main-thread contracts
  (`core::bind_main_thread` + `BRACKEN_ASSERT_MAIN_THREAD`); `thread_local`
  PRNG in `EffectSystem`; tested-but-unwired `render_graph/` and
  `FrameAllocator`.

**Lack (the work):**
- One `SDL_GPUCommandBuffer` per frame, recorded serially (SDL3 command
  buffers are single-thread objects; parallel recording = one buffer per
  thread, ordered submit; swapchain stays main-thread).
- `on_update`/`on_render` strictly sequential against one shared, mutated-in-
  place `RenderScene`; bone spans alias live host storage; particle spawns
  survive `clear()` and are drained by the renderer.
- Unsynchronized shared state in the frame path: `PipelineRegistry` lazy
  creation (unlocked map insert on first touch), `render_stats_` plain
  accumulator, every `SceneRenderer` per-frame scratch member, the
  `gpu_shader` cache (documented not-thread-safe), `Arena`/`FrameAllocator`
  (lockless bump by contract).
- No job counters/groups — only submit + wait(handle) + wait_all. Stage
  orchestration needs a `TaskGroup`.
- Hard main-thread residents to design around, not fight: SDL pump,
  swapchain acquire, ImGui's single global context.

---

## 3. The host-facing API (design constraint #1: elegance)

The measure of success: the playground gets *simpler*. A game developer
writes frame code that looks serial, owns their own data, and hands the
engine value-typed descriptions. Threads are ambient, opt-in, and safe.

### 3.1 The application shape

```cpp
class MyGame : public bracken::Application {
    // Runs as a job on the pool, overlapped with the previous frame's
    // rendering. Everything reachable through `sim` is safe from any job.
    void on_simulate(bracken::SimFrame& sim) override {
        // sim.dt(), sim.input() — immutable input snapshot, by value
        // sim.time(), sim.frame_index()

        // Straight-line code is fine — this is already off the critical
        // path of frame k's rendering:
        player_.update(sim.input(), sim.dt());

        // Fan out when you have real counts — engine-scheduled, joined
        // before on_simulate returns unless you keep the group:
        sim.parallel_for(npcs_, [&](Npc& n) { n.animate(sim.dt()); });

        // Build the frame. FrameBuilder is the ONLY way render state
        // leaves the game. Everything is copied or packet-allocated.
        auto& fb = sim.frame();
        fb.set_camera(camera_.state());
        fb.set_environment(env_);
        fb.draw_model(rock_, xform);
        fb.spawn_effect(sparkle_def_, pos);       // queued, engine-owned
        fb.ui().text(...);
    }

    // Rare escape hatch: runs on the main thread between frames, after the
    // previous frame's packet is retired — window calls, screenshots,
    // graphics-settings application. Most games never override it.
    void on_main_frame(bracken::MainFrame& mf) override {}
};
```

`run()` owns the loop and the pipeline. There is no `on_render` — "rendering"
is not a thing the host does; the host *describes a frame* and the engine
draws it. `on_init`/`on_shutdown` keep their contracts (main thread, GPU
available).

### 3.2 Skinned characters — the lifetime trap, deleted

Today: the host keeps `AnimationPlayer` storage alive and unmodified through
`render_frame` because the command holds a non-owning span. Natively:

```cpp
// Inside on_simulate — palette memory belongs to the packet, not the host:
auto palette = fb.alloc_bone_palette(skeleton_.joint_count());
anim_player_.update(sim.dt());              // per-instance, job-safe
foot_ik_.apply(anim_player_, terrain_fn);   // ditto
anim_player_.write_palette(palette);        // writes into packet memory
fb.draw_skinned(model_, xform, palette.id);
```

Zero-copy (the palette is computed straight into frame memory), zero
lifetime contract (the packet outlives its own consumption by construction),
and the double-buffered GPU palette upload the renderer already does keeps
working — it just reads packet memory instead of host memory.

### 3.3 Rules the API enforces (not documents — enforces)

- **Everything crossing the host→engine boundary is by value or
  packet-allocated.** No spans over host storage, no non-const refs into the
  renderer, no "must outlive render_frame."
- **`SimFrame`/`FrameBuilder` are job-safe.** `FrameBuilder` appends through
  per-thread lanes merged in deterministic (lane-chunk) order at seal — host
  jobs from `sim.parallel_for` can emit draws directly, and the frame is
  bit-identical regardless of worker count.
- **Engine services callable from jobs are marked and asserted.** The
  thread-safety tier table (§6.1) is public API documentation; violating it
  trips an assert in debug, not a heisenbug in ship.
- **Latency is explicit:** what you build in `on_simulate` k appears on
  screen at present k, one pipeline slot after the input snapshot it was
  built from. One frame of input latency versus a non-pipelined engine —
  stated plainly in the user guide, as every pipelined engine must.
- **`sim.parallel_for` / `sim.run_async(group, fn)`** are the whole threading
  surface a game team needs. `run_async` returns a `TaskGroup` the host may
  carry across frames (e.g. pathfinding) and must join before touching the
  results — the one sharp edge, fenced by an assert-on-unjoined-group at
  packet seal.

---

## 4. Engine architecture changes

### 4.1 Job system: from pool to frame engine

- `JobSystem` initializes in `init_engine`, unconditionally. Workers park on
  the CV indefinitely when idle (replace the 50 ms poll — an always-on pool
  must cost zero when quiet).
- Add **`TaskGroup`** (atomic counter + CV): submit-against-group,
  `wait(group)`, group-carrying handles for host async work. Stages are
  fork-join over groups; no full dependency-graph scheduler until a workload
  demands one (start simple, keep the door open).
- `parallel_for(JobSystem&, …)` overload (already prescribed in
  `job_system.hpp`'s header note); grain guards stay — at count 1 the loop
  inlines on the caller. Same code path, right granularity.
- Per-worker scratch `Arena`s, owner-asserted. The shared `Arena`/
  `FrameAllocator` stay single-owner by contract, now with a debug
  owner-thread check.

### 4.2 FramePacket (replaces the RenderScene contract)

- A ring of `FramePacket`s (depth = pipeline depth + 1), each backed by its
  own arena, reset on reuse. `RenderScene`/`UIScene` content becomes packet
  content: model/skinned/instanced commands, lights, billboards, debug lines,
  particle spawn queue, camera, environment, settings snapshot, UI commands,
  input snapshot, dt/time.
- **Seal semantics:** Simulate fills (multi-lane), seals, publishes; Prepare
  and Record consume `const FramePacket&`. The renderer's particle-spawn
  drain moves into the sim stage — `EffectSystem` becomes sim-stage-owned,
  and the renderer consumes a particle *instance snapshot* from the packet.
  `render_frame`'s non-const `RenderScene&` disappears from the API.
- Bone palettes live in the packet (§3.2). `MAX_BONE_PALETTES` accounting
  moves to `alloc_bone_palette` — over-budget fails visibly at build time,
  replacing today's silent drop-and-log at render time.
- Motion history keys (`motion_id`) unchanged — the packet carries them; the
  N/N−1 pairing logic already matches this model.

### 4.3 Stage orchestration

- `Application::run()` becomes the pipeline coordinator: pump events →
  publish input snapshot → kick Simulate(k+1) as a job → run Prepare(k) +
  Record(k) → submit/present → retire packet k. The main thread *helps* the
  pool inside Prepare joins rather than idling.
- **Prepare fan-out** (all fork-join over the sealed packet, per-thread
  partials merged in deterministic chunk order — this is also where
  `render_stats_` becomes merged partials):
  - per-view culling + instance batching: main view, 4 CSM cascades,
    local-shadow tile casters — disjoint outputs, one job each, chunked
    within;
  - skinned-command resolve + palette layout;
  - cluster light binning (per-cluster form — each cluster writes only its
    own bin; the memoized cluster-bounds rebuild chunks trivially);
  - grass chunk cull, terrain tile cull (run-merge consumes chunk results in
    order);
  - particle integration for effects (spawn applied at sim; integrate is
    per-particle `parallel_for`; compaction per-emitter).
  - Explicitly *not* jobbed: CSM cascade fit (≤4 tiny fits), water (GPU
    compute), IBL/TAA/GTAO/bloom/fog/exposure (GPU), DevUI.
- **Hot-reload fence:** FileWatcher callbacks apply at top-of-loop, before
  the next packet is acquired — never while a packet referencing the old
  resource is in flight. Same fence hosts graphics-settings application and
  `on_main_frame`.
- **DevUI** stays main-thread (ImGui global context), fed from a stats/frame
  snapshot; panels read published data, never live renderer members.

### 4.4 GPU front-end

- **Pipeline pre-creation at init** (the registry knows the full set);
  `get_pipeline` becomes read-only lookup, creation paths main-thread-
  asserted. This deletes the first-touch race instead of locking it.
- Same treatment for the shader cache and `ModelManager` mutation paths:
  loading/creation main-thread (until streaming, §5 Phase 6), lookups
  const, asserts everywhere.
- **Parallel recording** (final render phase): wire `render_graph/` as the
  frame scheduler — sub-renderers declare passes (reads/writes/attachments),
  the graph compiles topo order + transient lifetimes, then partitions the
  pass list into ordered segments recorded on pool workers, **one SDL3
  command buffer per recording thread**, submitted in sequence from the main
  thread. The swapchain-touching segment (late-acquire composite + UI)
  records on the main thread. The graph's transient pool finally earns its
  keep (post-chain and shadow targets); persistent histories (TAA, GTAO,
  IBL) stay imported resources, which the graph already models.

---

## 5. Migration roadmap

Each phase lands green (build + tests + smoke) and replaces the old
mechanism outright — no phase leaves two ways to do the same thing.

**Phase 1 — Core substrate.** Always-on pool with indefinite park;
`TaskGroup`; `parallel_for(JobSystem&, …)`; per-worker arenas + owner
asserts; pipeline/shader/model pre-creation + main-thread asserts;
`FrameStatsPartial`; thread-safety tier table in `architecture.md`. TSan CI
lane from day one. *Engine behavior unchanged; the floor is now safe.*

**Phase 2 — FramePacket + FrameBuilder.** Packet ring, builder lanes,
palette-in-packet, spawn queue moved to sim side, `render_frame(const
FramePacket&)`. Playground ported; `RenderScene` and its lifetime contracts
deleted. *Still sequential sim → render — the data architecture lands before
the concurrency does, so this phase is bisectable.*
**Status: LANDED** — `scene/frame_packet.{hpp,cpp}`, `scene/frame_builder.{hpp,cpp}`,
`scene/frame_packet_ring.hpp` + tests; renderer/host ported off
`RenderScene`/`UIScene`; palette budget enforced at build time in
`alloc_bone_palette`.

**Phase 3 — Jobify the engine.** Prepare fan-out (§4.3) converted to jobs
with deterministic merges; particle split (spawn/integrate/compact);
animation helpers (`write_palette`, job-safe update paths — the parity test
already exists). Worker-count-1 vs N bit-equality tests per stage. *Main
thread still runs the stage sequence; stages are internally parallel.*
**Status: LANDED** — the §4.3 Prepare loops fan out via
`parallel_for(JobSystem&, …)`/`TaskGroup` chunks over the sealed packet:
per-view culling + instance batching (main view + CSM cascades; the
local-shadow tile-caster cull stays serial for now — it is interleaved with
GPU tile recording in `render_local_shadow_passes` and moves with Record in
Phase 5), skinned-command resolve + palette layout, cluster light binning,
grass chunk cull, terrain tile cull, and particle integration (spawn stays
serial on the calling thread — it moves to sim in Phase 4; compaction
per-emitter). Every loop follows the
determinism contract (§6.2): per-chunk partials (bins, stats, scratch)
merged single-threaded in chunk-index order after the join — no atomics or
shared cache-line writes inside per-item loops; `render_stats_` accumulates
via merged `FrameStatsPartial`s. Engine-internal stage helpers take a
`JobSystem&` (call sites pass `JobSystem::instance()`); per-stage parity
tests construct caller-owned pools (`init(1)` vs `init(4)`) and assert
bit-identical outputs. Grain guards keep the 1-character playground scene
effectively serial (one chunk, same code path).

**Phase 4 — Pipeline the frame.** `run()` becomes the coordinator;
`on_simulate`/`on_main_frame` replace `on_update`/`on_render`; input
snapshots; sim-stage ownership of `EffectSystem`, scene/sky time; hot-reload
fence; DevUI snapshot feed. Playground ported to the final API. *This is the
flip: Simulate(k+1) ∥ Render(k). The Application API change is the
headline; hosts migrate once.*
**Status: LANDED** — `Application::run()` is the pipeline coordinator
(depth 2: join sim i → frame fence → event pump → kick sim i+1 → render i →
retire i; `application.hpp` documents the exact schedule).
`on_update`/`on_render` are deleted; hosts implement
`on_simulate(SimFrame&)` (runs as a pool job, overlapped with the previous
frame's rendering) and optionally `on_main_frame(MainFrame&)` (the fence
hook — the only safe place to read live game state from the main thread).
`SimFrame`/`MainFrame` live in `sim_frame.hpp`; input reaches the sim as a
by-value `InputSnapshot` taken at kick (`InputHandler::snapshot()`); the
§3.1 sketch's `sim.input()` returns the richer `InputSnapshot` (embedding
`InputState` as `.state` plus key edges / mouse / camera-orbit mirrors of
the old getters) rather than the bare `InputState` — real hosts need the
edges. dt is computed at kick and carried in the packet. The engine's
sim-stage tail (inside the sim job, after `on_simulate`): builder effect
spawns applied to the now sim-stage-owned `EffectSystem` in deterministic
lane order → `EffectSystem::update` on the pool → effect-instance snapshot
+ sim-advanced scene/sky clocks written into the packet's engine section
(`FramePacket::effect_emitters()/effect_particles()/scene_time()/sky_time()`)
→ seal. The renderer consumes only the sealed snapshot and adopts the
packet clocks (no renderer-side dt integration; `WorldRenderer::update(dt)`
became `set_sky_time`). Hot-reload, deferred graphics settings
(`request_graphics_settings`), camera-forward feedback, FPS accounting, and
deferred screenshots all apply at the frame fence. Handoff synchronization
is the sim TaskHandle kick/join edges (the ring publish/acquire pair is not
needed under `run()`); a sim exception is fatal-logged, the partial packet
abandoned, and the exception rethrown out of `run()` on the main thread.
One frame of input latency (snapshot at kick(i+1) presents at i+1) is
documented on `Application`.

**Phase 5 — Graph-scheduled parallel recording.** Sub-renderers declare
passes; graph compiles and partitions; per-thread command buffers; ordered
submit. Transient pool live. *Gated honestly: at Bracken's current pass and
draw count, recording may not dominate — the phase still lands the graph
wiring (correctness + multi-view + transient memory), and fans out recording
only where a measured segment justifies a buffer split. Native architecture,
sized to the actual GPU front-end cost.*

**Phase 6 — Native streaming.** Asset parse/decode/generate as jobs;
main-thread publish queue (the FileWatcher pattern) creating GPU resources at
the frame fence; `ModelManager` loading→resident states formalizing the
existing "skip if not uploaded" walk into the streaming contract.

Sequencing note: Phases 1→4 are strictly ordered; 5 and 6 are independent
after 4 and 1 respectively. The stress knob and Tracy dev preset from the
measurement groundwork land alongside Phase 1 — not to gate whether we go
multithreaded (that decision is made) but to size grains, verify each phase
wins at scale, and catch regressions at the 1-character scene.

---

## 6. Cross-cutting contracts

### 6.1 Thread-safety tiers (published in architecture.md, asserted in debug)

| Tier | Meaning | Examples (end state) |
|---|---|---|
| `main-thread` | asserted | swapchain, submit, ImGui/DevUI, GPU create/destroy, FileWatcher entry points, `on_main_frame` |
| `sim-stage` | any job within Simulate | `SimFrame`, `FrameBuilder`, `AnimationPlayer`, `EffectSystem` spawn, IK |
| `worker-safe` | any thread, any time | `Logger`, `parallel_for`, `TaskGroup`, packet reads after seal, pipeline/shader/model lookups |
| `owned` | single owner, owner-asserted | `Arena`, per-worker scratch, packet during fill (per-lane) |

### 6.2 Determinism

Same code path at every worker count; builder lanes and all Prepare merges
ordered by chunk index, never thread id; parity tests assert bit-equality
between `BRACKEN_WORKERS=1` and `=N` runs of tests and the smoke scene
(draw/instance/palette counts exact; screenshot self-verify for gross visual
breaks). Float-reassociating reductions are forbidden in merges unless a test
explicitly accepts an epsilon.

### 6.3 Risks

| Risk | Mitigation |
|---|---|
| The historical lambda-capture bug (container snapshotted by reference into workers; libstdc++ bounds assert, never diagnosed) | `sim.parallel_for` takes caller-stable ranges; builder lanes are engine-owned; the pattern that caused it is no longer writable through the API |
| Pipelined-frame heisenbugs | `BRACKEN_WORKERS=1` (same path), TSan lane, deterministic merges, packet immutability after seal (debug write-protect the arena where the platform allows) |
| One frame of input latency | Inherent to pipelining; documented; `on_main_frame` exists for the rare truly-latency-critical main-thread need |
| Oversubscription (pool + FileWatcher + driver threads) | N = cores − 1, main thread helps at joins, sim runs *on* the pool (no dedicated sim thread) |
| Host async jobs outliving their data | `TaskGroup` join asserted at packet seal for frame-scoped groups; cross-frame groups are explicitly owned by the host |
| Small-scene regression (always-on infra tax) | Parked workers cost ~zero; grain guards keep tiny counts inline; the 1-character smoke scene is a perf regression gate |
| API migration churn for hosts | One migration (Phase 4), mechanical (`on_update`+`on_render` → `on_simulate`); the playground port is the worked example |
| Subtle lock-free bugs (deque, wake protocol, publish edges) | Only three bespoke primitives exist (deque, injection queue, TaskGroup); each follows a published reference algorithm, documents its ordering protocol at the declaration, and lands with race-shaped stress tests + the TSan lane (§7) |
| Doc drift | Each phase updates `architecture.md`, `user_guide.md`, `engine_design_notes.md` (decision record), and this file in the same change — house rule |

---

## 7. Synchronization design

The contention question is settled by architecture first and primitives
second. The rule hierarchy, in order — each level is only reached when the
one above genuinely can't apply:

1. **Don't share.** Per-worker / per-lane / per-chunk data everywhere the hot
   path iterates. Zero shared cache lines touched per item is the design
   target, not an optimization.
2. **Share immutably.** Sealed packets, pre-created pipeline/shader/model
   tables — plain loads, no atomics at all.
3. **Transfer ownership, don't guard it.** Single release-store /
   acquire-load publish points hand whole data structures between stages.
4. **Lock-free structures** only where genuine multi-producer/multi-consumer
   traffic exists (the job queues — and almost nowhere else).
5. **Mutexes on cold paths only** (init, shutdown, hot-reload apply,
   FileWatcher bookkeeping). A mutex inside the frame hot path is a review
   defect.

Concretely, per component:

- **Job queues — Chase–Lev work-stealing deques.** The current per-worker
  `std::deque` + mutex (stealers `try_lock`) is correct but is exactly the
  "simple mutexes with contention" trap at high submit rates. Phase 1
  upgrades to the bounded lock-free Chase–Lev deque with the C++11-memory-
  model formulation (Lê/Pop/Cohen/Nardelli 2013): owner pushes/pops the
  bottom with relaxed/acquire-release ops, thieves CAS the top — the only
  contended op is a steal, which is by definition the rare path of a
  well-grained workload. External (non-worker) submits go through an MPSC
  injection queue instead of a mutexed global queue.
- **Job lifetimes — pooled, not heap-allocated.** `submit()` currently
  allocates a `shared_ptr<Task>` per job (atomic refcount churn + allocator
  traffic on the hottest path). Replace with fixed-size job nodes in
  per-worker pools, recycled via freelist, handles carrying a generation
  counter for the done/error query — no allocation, no refcounts, ABA-safe.
- **Parking — futex-style, two-phase.** Idle workers spin briefly then park
  on C++20 `std::atomic::wait`/`notify_one` (futex/ulock under the hood) —
  no condition-variable + mutex pairing, no 50 ms poll, no thundering herd
  (targeted `notify_one` per submitted job, `notify_all` at shutdown and
  once per `submit_batch` dispatch).
  The existing lost-wakeup regression tests (gated-job discard, external
  waiter) carry over as the safety net for the wake protocol.
- **`TaskGroup` — one padded atomic.** A cache-line-aligned
  (`std::hardware_destructive_interference_size`) counter: release
  `fetch_sub` on job completion, the zero-observer does an acquire fence and
  notifies; waiters park on `atomic::wait`. Nothing else in it.
- **Frame pipeline handoff — two atomic ops per frame.** The packet ring is
  a state machine (filling → sealed → consuming → retired) advanced by
  single release-publishes of a slot index, acquired by the consuming stage.
  No mutex, no CV, no queue: the *entire* cross-stage synchronization cost
  of a frame is one publish and one acquire (plus the TaskGroup join inside
  each stage).
- **Builder lanes and allocators — reserve in blocks, bump privately.**
  Each sim job writes into its own lane (header padded to the destructive
  interference size — false sharing is treated as a bug class, not a
  curiosity). Packet-arena and palette allocation grab block-sized ranges
  with one `fetch_add` per *block*, then bump privately per item — amortized
  to ~zero shared traffic. Stats identically: per-chunk partials, merged
  single-threaded at join; an atomic counter increment inside a per-item
  loop is, again, a review defect.
- **Read-mostly data + hot-reload — epoch reclamation via the frame fence.**
  Pipeline/shader/model tables are immutable after init, so readers pay
  nothing. Hot-reload swaps happen only at the top-of-loop fence where no
  packet is in flight — the pipeline's natural quiescent state gives us
  epoch-based reclamation for free, which is why no hazard pointers, RCU
  library, or reader locks are needed anywhere.
- **Logger** keeps its mutex for the **sink list** (cold path, correct,
  snapshot-then-write-unlocked). The level check no longer takes it:
  `should_log` is on every worker's fast path and is `noexcept`, so a
  `lock_guard` there was both a global serialization point and a latent
  `std::terminate`; `min_level_` is now a relaxed `std::atomic<LogLevel>`
  with its protocol at the declaration. If Tracy ever shows contention from
  chatty workers the *dispatch* side upgrades to an MPSC ring drained on the
  main thread — decided by measurement, not preemptively.

**Memory-model discipline**, because clever atomics without rigor are worse
than mutexes:

- Every atomic in the codebase documents its protocol at the declaration:
  what it publishes, its ordering, and its release/acquire pairing partner.
  Bare `seq_cst` in a hot path needs a justifying comment or gets rewritten
  with explicit orderings.
- The existing `Task::done` release-store-gates-non-atomic-`error` pattern is
  the house style: non-atomic payload + one atomic publish edge, stated in a
  comment.
- TSan lane in CI (Phase 1 onward) plus targeted stress tests for each
  protocol (the job-system suite's gated-job/discard/waiter races are the
  template — every new primitive lands with its own race-shaped tests, run
  at high iteration counts).
- What we deliberately do **not** build: general-purpose lock-free MPMC
  containers, hazard pointers, or a dependency-graph scheduler with atomic
  edge counting. Not because they're too hard — because the architecture
  above removes their use cases, and every lock-free structure we don't
  write is one we can't get subtly wrong.

**Field notes (2026-07-19)** — lessons from the landed primitives:

- **Tagged single-word lifetime state.** Separate generation + refcount
  atomics on job nodes allowed a resurrection race (stale handle re-inflates
  a recycled node's count). `JobNode::state` now packs {generation, refcount}
  into one 64-bit word mutated by a single generation-checked CAS
  (`job_system.hpp`).
- **Waiters park on an immortal word.** Parking on a `TaskGroup`'s own
  counter is a UAF once the group destructs between the zero-decrement and
  the notify. All group waiters park on the static
  `detail::g_group_join_epoch` word instead (`task_group.hpp`).
- **TSan does not model `atomic_thread_fence`.** The Chase–Lev push()
  publication (release fence + relaxed bottom store) is invisible to TSan;
  under TSan it compiles as an equivalent release store
  (`BRACKEN_TSAN_FENCE_WORKAROUND`, `chase_lev_deque.hpp`) — production keeps
  the paper's fence formulation.
- **Workers drain their own deque on exit.** A nested submit landing after
  shutdown's first discard pass would otherwise strand a job reachable only
  by random-probe steal, deadlocking a parked nested waiter; the exiting
  worker completes each stranded job with the shutdown error
  (`job_system.cpp`, worker-loop exit).

## 8. End-state check

**Where reality stands after Phase 4:** a game developer subclasses
`Application`, writes `on_simulate` as if single-threaded, calls
`sim.parallel_for` where they have thousands of anything, and ships a game
whose simulation overlaps rendering — never once managing a thread, a
lifetime-across-the-renderer, or a command buffer. That much is landed and
live: one code path, deterministic at any worker count, TSan-laned, with
the job system and packet ring as load-bearing pieces, and the Prepare
stage fanned out across the cores.

Not yet reality: the render graph is still unwired (recording is one
serial command buffer on the main thread — Phase 5), and asset streaming
is still boot-time/hot-reload only (Phase 6). Known residual sharp edges,
documented at their declarations: the DevUI Camera tab reads the live
`CameraController` during render (tear-only reads); `graphics_settings()`
is a host-owned struct with a documented single-writer-per-phase contract,
not a fenced one; `sim.parallel_for` bodies must not emit into the builder
(no automatic lane assignment — emit from root code or from `run_async`
jobs with explicit `fb.lane(i)`); and sky-cloud scroll speed follows the
environment the host last *staged* via `set_environment`, not live DevUI
edits, for hosts that never stage one.
