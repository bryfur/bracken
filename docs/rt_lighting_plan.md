# Ray-Traced Lighting — Research & Plan of Record

Status: **proposed** (research complete, no phase started). Written 2026-08-02
against the working tree at the RT-denoiser series (post "RT shadows: stop
multiplexing ambient occlusion into the shadow mask").

This is the plan of record for growing the current two-effect ray tracer
(stochastic sun shadows + GGX reflections) into a **full ray-traced lighting
system**: diffuse global illumination with infinite bounce, ray-traced
point/spot shadows with real penumbras, emissive-mesh lights, GI-aware
reflections and fog — scaling from integrated GPUs to high-end discrete,
with nothing baked (the dynamic time-of-day/weather system forbids it).

Like `docs/multithreading_plan.md`, this document is the *why* and the *what*
in dependency order, not a sprint schedule. Phases are independently
shippable; each leaves the tree working, tested, and documented.

**Note on doc tooling:** this file deliberately stays out of
`docs_coherence.test.cpp`'s `DOCS[]` list while phases are unlanded — it cites
files that do not exist yet, which check (a) of that test would fail. Add it to
the list once R1 lands (the same reasoning that keeps `docs/refactor_plan.md`
excluded, run in reverse). Everything cited from the *current* tree below was
verified by hand on 2026-08-02.

---

## 1. Where we are

What exists today, because the plan builds on every piece of it:

**Acceleration structure** (`render/raytracing/`): two-level software BVH.
`BvhNode` (`bvh.hpp`) is 32 bytes — one AABB, `left_first`, `count`; siblings
adjacent so the right child is `left+1`. That is the BUILD format: since WP2 it
is converted at upload to the 64-byte both-children `BvhNodeGpu`
(`bvh_gpu.hpp`) that the shaders actually walk — see §5.0.
`BvhTriangle` is 48 B in
Möller–Trumbore form (v0/e1/e2 + `prim_id`). CPU binned-SAH build (16 bins,
leaf ≤ 4 tris, max depth 64), one BLAS per model fanned out over the job pool
(~80 ms single-threaded across the playground's 52 models), everything
concatenated into one node buffer + one triangle buffer so traversal never
rebinds. The TLAS is rebuilt only when the instance SET changes — a second
FNV-1a hash over the count and handle sequence alone — while a list whose
`(handle, transform)` content hash moved but whose topology did not costs a
bottom-up **refit** of the existing tree (`RtScene::set_instances`,
`rt_scene.cpp`; a full build is 2569 instances ≈ 0.49 ms at -O2, the refit
~4.5× less end to end). Either way it is written into the frame's command
buffer through persistent grow-only buffers. Playground scale:
6.2 M drawn triangles = ~600 k unique behind instancing, ~29 MB of BLAS.

**Traced effects** (`render/rt_renderer.hpp/.cpp`): both 1 sample/pixel,
stochastic, denoised, independently optional, off by default.

- *Sun shadows* (`rt_shadow.comp.hlsl`): no primary rays (world position from
  the depth prepass), **any-hit** traversal with no child ordering, N·L and
  sky rejection before traversal, slope-scaled normal bias, one ray jittered
  across the solar disk (`sampleCone`, R2 sequence + interleaved gradient
  noise from `rt_sampling.hlsli`). Output is 1 bit/pixel packed 8×4 per uint.
  Denoiser: tile classify → temporal (mean/moment/count/depth history,
  velocity-reprojected, RGBA16F) → variance-guided spatial filter **fused with
  the upsample** to a full-res R8 mask. The forward shaders sample that mask
  and it **replaces** the CSM term (slots: model/instanced 10, terrain 8,
  grass 6).
- *Reflections* (`rt_reflection.comp.hlsl`): GGX **visible-NDF** sampling
  (Heitz 2018), closest-hit traversal with near-child-first ordering and
  `hit.t`-tightened culling, roughness-gated hand-over to the prefiltered IBL
  probe (`reflection_gate.hlsli` — trace weight + IBL weight always sum to 1).
  Hit shading is `TriangleAttributes` (64 B/tri: three normals + one flat
  base color baked at load from the albedo texture's centroid-UV sample —
  `internal/model_material_proxy.hpp`) lit by sun N·L **with no visibility
  test** plus a flat `skyColor * 0.25` ambient. Temporal-only denoise
  (σ-clamped neighborhood, accumulation in alpha), then a full-res
  material/depth/normal-guided resolve evaluates the *receiver* BRDF. A
  separate analytic water-receiver path reuses the same target, undenoised.

**Around them**: RGBA8 G-buffer (octahedral normal / roughness / metallic)
from the depth prepass, RG16F velocity, reversed-Z D32 depth, TAA (Halton-8,
RT runs before it, deliberately), GTAO multiplying the ambient term only,
CPU-binned clustered lighting (16×9×24 froxels, ≤256 lights, ≤128/cluster), a
rasterized local-shadow atlas (8×8 D32 tiles; point ≤8 lights × 6 faces, spot
≤16; ≤6 tile re-renders/frame), IBL chain (irradiance cube + 7-mip prefiltered
specular + BRDF LUT over the procedural sky only), volumetric fog whose only
ambient term is a hard-coded `fogColor * 0.2`, and histogram auto-exposure.

**Known limits the plan addresses directly:**

1. A ray hit cannot sample textures — no UVs, no material index, no bindless.
   Hit shading is interpolated normal + flat color, full stop.
2. Reflection hit lighting is wrong twice: the sun term is unshadowed and the
   ambient is a flat constant. Reflections show no GI and no shadows.
3. Skinned meshes trace their **bind pose** (translated/rotated into place).
   Because the traced sun mask *replaces* CSM, an animated character's RT
   shadow does not animate.
4. Local (point/spot) lights get 8×8-tile rasterized atlas shadows under hard
   budgets (8 point / 16 spot / 6 tile updates a frame) — no penumbras, and
   light #9 gets nothing.
5. There is no diffuse GI at all: ambient is IBL-sky or a hemisphere constant.
   Nothing bounces. Fog scatters a constant.
6. The reflection pass is **wave-divergence bound**, not traversal bound
   (measured; see the comment in `rt_reflection.comp.hlsl`'s `blasClosest`) —
   scattered reflective texels leave nearly every wave with one active lane.
7. No HZB, no previous-frame depth/normal, no indirect dispatch wrapper, no
   GPU timestamps (SDL3 has none — CPU record times only).

---

## 2. Ground rules the platform imposes

Established by measurement or by reading SDL3/SPIRV-Cross, not negotiable by
enthusiasm:

- **No hardware RT, ever, on any backend.** SDL3 GPU has no acceleration
  structure API. Every ray is compute-shader arithmetic against our buffers.
- **Ray throughput is the budget.** Portable-compute anchors (sources in §12):
  incoherent rays at 1080p ≈ **15–60 M/s** on Deck/base-M-class iGPUs,
  **300–700 M/s** on RTX 4060 / RX 7600 class, **1–2.5 G/s** on 4090 class.
  One full-screen 1080p effect is 2.07 M rays; inside 2 ms it needs ≥1 G
  rays/s — *only high-end compute has that*. Mid-range wants half-res;
  iGPUs want quarter-res or amortized (probe) workloads. This single table
  shapes every tier decision below.
- **No bindless** → rays cannot shade arbitrary materials at hit points.
  Everything downstream either (a) shades hits with a *restricted* material
  model (our flat-color table — already built), or (b) looks up **cached
  radiance** instead of shading. The plan uses (a) for cheap hits and (b) for
  everything expensive. This is the same fork Lumen answered with its surface
  cache (a giant subsystem we reject) and GI-1.0/Brixelizer answered with
  radiance caches (the shape we adopt).
- **Wide BVHs are not the win here.** The 2017 CWBVH numbers (2.2–2.7× over
  BVH2) lean on NVIDIA-only instructions; in portable compute (tinybvh's
  cross-vendor OpenCL kernels, 2026) **BVH2 beats CWBVH outright**, especially
  on Apple. The portable wins are: a 64-byte "both children in the parent"
  node layout (halves dependent fetches), higher-quality *offline* builds
  (SBVH + reinsertion: up to ~90 % GPU traversal gain on complex scenes),
  any-hit specialization (we have it), and resolution tiers.
- **Wave intrinsics survive DXC→SPIR-V→MSL** for the mainstream subset
  (ballot+countbits, prefix, reductions, readFirstLane) but wave *width* is
  not knowable at compile time (this machine: reports 32, driver advertises
  64 — already documented in `rt_shadow.comp.hlsl`). Rule: traversal loops
  stay scalar; wave ops live only in denoiser/reduction passes, each listed in
  `SHADER_WAVE_OPS` (raises that shader's floor to Vulkan 1.1).
- **Indirect dispatch exists in SDL3** (`SDL_DispatchGPUComputeIndirect`,
  `SDL_GPU_BUFFERUSAGE_INDIRECT`) and the engine has never wrapped it. Two
  planned passes want it (adaptive screen-probe placement, variable-rate
  reflection tiles); it enters the `gpu/` layer in R4, not before.
- **No async compute.** Passes serialize on one queue. The mitigation is not
  overlap tricks — it is honest per-pass budgets and per-tier resolution.
- **Determinism is a feature.** The smoke gate depends on it. Every stochastic
  pass seeds from a frame counter (`shadow_frame_index_`,
  `reflection_frame_index_`); every new pass must do the same, and the
  A-SVGF-style gradient work in §5.7 additionally requires the sampler be
  *replayable* (same pixel, same frame index → same sample) — which R2+IGN
  and STBN both are.

---

## 3. What the research says (condensed verdicts)

Five parallel research sweeps (many-light ReSTIR, radiance caches, radiance
cascades + SSGI, software traversal, denoising/sampling) — full source list in
§12. The one-line verdicts that drive the architecture:

| Technique | Verdict | Why |
|---|---|---|
| ReSTIR DI (RIS + temporal reservoirs, 1 shadow ray/px) | **Adopt**, phased | The proven answer to "shadow N lights with ~1 ray"; every stage is plain compute. No shipped title runs it over a *software* BVH yet — we'd be early, but nothing in it needs HW RT |
| Froxel grid as the ReSTIR candidate structure | **Adopt** (our own synthesis) | The 16×9×24 cluster grid already built per frame *is* a screen-space light-culling structure; candidates drawn from the pixel's froxel beat ReGIR/world grids at ≤256 lights and cost zero new infrastructure |
| Light BVH / ReGIR / Area ReSTIR | Reject at our scale | Built for 10⁴–10⁶ lights / lens-area integration; froxel-RIS dominates below ~1 k lights |
| DDGI-style probe volumes (octahedral irradiance + Chebyshev visibility, scrolled cascades) | **Adopt** as the GI base | The only GI shape whose cost is *amortized* (rays/frame budget, not rays/pixel) — the only thing that fits an iGPU. ~1.3 KB/probe. Our flat-color hit table + CSM-at-hit makes probe rays cheap to shade |
| Lumen-style screen-probe gather (1 probe / 8×8 px, probe-space filtering) | **Adopt** as the mid/high GI tier | GI-1.0 (AMD, MIT) measured 1.9–3.4 ms @1080p on 6900 XT-class doing exactly this over its own tracer; probe-space filtering costs ~0.6 ms where pixel-space GI denoisers cost 2.5–3.3 ms |
| Lumen surface cache / mesh SDFs / Brixelizer SDF bricks | Reject | Second scene representation with its own build pipeline; Godot's SDFGI→HDDAGI history is years of bleeding on the *structure*, not the lighting. We already maintain a triangle BVH — trace it |
| SHaRC-style world hash radiance cache | Watch (stretch) | Valuable only once screen feedback + probe recursion prove insufficient; NVIDIA license forces a reimplementation |
| Radiance cascades (2D as shipped in PoE2) | Reject | The 3 ms/GTX 1050 figure is load-bearingly tied to a fixed isometric camera and flatland angular basis |
| Radiance cascades (dense 3D) | Reject | Memory scales with scene volume³ (0.25–4 GB for our village); the inventor's own dense-3D prototype runs 30–50 ms on an RTX 3060 |
| Split Radiance Cascades (Freeman & Sannikov, arXiv, **July 2026**) | **Watch closely** | Sparse world probes + ray splitting; eleven days old at time of writing; re-evaluate in 12–18 months |
| Visibility-bitmask SSGI (SSILVB) + bent cones | **Adopt** as near-field supplement | ~1 ms half-res on mid GPUs, shares GTAO's horizon loop and inputs; recovers the contact-scale GI probes cannot carry |
| Screen-trace-first, world-trace fallback (HZB march) | **Adopt** | Screen hits cost log-n texture fetches instead of BVH triangle tests, and **skinned/dynamic meshes appear in GI/reflections via the depth buffer even though their BLAS is bind-pose** — the Lumen/Tiny Glade trick that solves limit #3 for free where it matters most |
| CWBVH / GPU BVH builds / persistent threads / ray sorting at 1 rpp / DGF | Reject (watch H-PLOC, DOBB-BVH, DGF's Vulkan ext) | Portable-compute evidence says no (§2); our BLASes are static and TLAS is microseconds on CPU |
| 64-B two-child node layout + offline HQ BLAS (SBVH + reinsertion) | **Adopt** | The two traversal wins that survive portability; tinybvh (MIT) vendorable as the offline builder |
| NRD (ReBLUR/ReLAX/SIGMA) as a library | Reject (license + NRI integration); **adopt the ideas** | Hit-distance-driven accumulation, fast-history clamp, penumbra-from-hit-distance are all reimplementable; FidelityFX (MIT) is the code quarry — our shadow denoiser already mirrors its architecture |
| A-SVGF temporal gradients | **Adopt** | Sparse (1-in-9) sample replay detects *lighting* change per region and resets history surgically — the correct answer to fast time-of-day/weather/lightning, which fixed-α accumulation smears |
| Virtual-hit-point reflection reprojection | **Adopt** | Highest-value single upgrade to the existing reflection denoiser; needs the trace to output hit distance |
| STBN blue noise (via EA FastNoise, BSD-3) | **Adopt** | Better 1-spp error shaping than R2+IGN for the denoisers; NVIDIA's own STBN SDK is license-restricted — generate with FastNoise instead |
| DLSS-RR / FSR4 Ray Regeneration / NRC / cooperative vectors | Reject / watch | ML- and API-locked; SDL3 GPU exposes none of it. SHaRC-class caches are NVIDIA's own stated non-neural equivalent |
| MegaLights (UE 5.5+) | Watch the *design* | Validates fixed-budget stochastic DI as the industry direction; its tile-importance scheme is the simpler fallback if reservoir complexity balloons |

---

## 4. Destination architecture — "trace few, reuse everything"

Every signal follows one law: **cast the fewest rays that carry new
information, then squeeze them through every reuse axis available** —
temporal (reservoirs, accumulation), spatial (probes, filters), and
structural (caches feeding caches). Rays per pixel is the only currency;
budgets are set per tier and enforced by settings, not hope.

The four signals and their planes:

```
                     ┌──────────────────────────────────────────────┐
                     │        world radiance state (persistent)     │
                     │  probe clipmap: octa irradiance + visibility │
                     │  (R1)          ▲ feeds & samples itself      │
                     └───────┬────────┴─────────────┬───────────────┘
             sampled at hit  │                      │ ambient/fog/rough-spec
                             │                      ▼
  sun ray (have) ────────────┤            ┌──────────────────────┐
  local-light ray (R2) ──────┼──────────► │  per-pixel composite │
  probe rays (R1) ───────────┤            │  forward shaders +   │
  reflection ray (have) ─────┤            │  resolve passes      │
  screen-probe rays (R4) ────┘            └──────────────────────┘
        every ray: screen-trace first (R4+), BVH second, probe/sky fallback
```

- **Sun** stays a dedicated 1-spp cone-sampled ray + the existing denoiser
  (visibility is too high-frequency to share a pipeline with anything).
- **Local lights** become *sampled*, not enumerated: RIS over the pixel's
  froxel light list picks one light, one sphere-cap shadow ray answers it,
  temporal reservoirs make ~1 ray/frame behave like dozens (§5.3).
- **Diffuse GI** lives in a scrolled probe clipmap updated by a fixed
  ray budget per frame — never per pixel (§5.2). A screen-probe gather
  sharpens it on mid/high tiers (§5.5); bitmask SSGI supplies contact
  detail (§5.6).
- **Specular** keeps its 1-spp trace but its *hits* start telling the truth:
  sun-shadowed via CSM sample, ambient from the probe field — so reflections
  show GI and shadows without one extra ray (§5.4).
- **Fog** stops scattering a constant and samples the probe field per march
  step (§5.2).

One shading contract change underneath all of it: today `ambientIBL`
(`pbr.hlsli`) supplies "ambient" from the sky alone. After R1, **indirect
diffuse = probe field irradiance** (which itself contains sky via ray misses
and bounce via recursion), IBL keeps only the specular half until R4, and
GTAO/SSILVB modulate the probe term exactly as they modulate ambient today.
`EnvironmentState::ambient_sky`/`ambient_ground` become the no-GI fallback
tier rather than the look.

---

## 5. Component specifications

### 5.0 R0 — Foundations (traversal, sampling, plumbing)

Small, zero-visual-change groundwork everything else stands on:

- **Shared traversal include.** `rt_shadow.comp.hlsl` and
  `rt_reflection.comp.hlsl` each carry a private copy of the BVH structs +
  slab/Möller–Trumbore/traversal code. Extract `rt_trace.hlsli` (structs,
  `slabTest`, any-hit and closest-hit walks, octahedral decode) before three
  more consumers appear. The two entry shaders keep their distinct binding
  blocks; only the code moves.
- **64-byte GPU node layout.** *Landed (WP2, 2026-08-02.)* `BvhNodeGpu`
  (`render/raytracing/bvh_gpu.hpp`) holds both children's AABBs and links in
  64 bytes: `{lmin, lchild, lmax, lcount, rmin, rchild, rmax, rcount}`, where
  `count == 0` on a side means "interior, `child` is that child's own 64-B
  node" and `count > 0` means "leaf, `child` is its first primitive". One
  fetch resolves a binary decision; leaves never occupy a node or a stack
  slot; the closest-hit walk's near-first ordering now comes from data already
  in hand, so its two extra child fetches are gone.

  The CPU build format is untouched — `convert_to_gpu_nodes` maps one new node
  per old *interior* node in two linear passes, called per BLAS in
  `build_blas_for_models` and per TLAS rebuild **or refit** in `set_instances`
  (measured at 4.6 % of the binned-SAH build it follows for 2569 instances —
  which is also why a refit re-derives the 64-byte form rather than editing it
  in place: the conversion stays the single producer of every uploaded byte,
  and `bvh_gpu_layout.test.cpp` therefore still covers the refit path). Only interior
  nodes survive, so node count halves as the node doubles and the buffers do
  not grow: the playground's 140736 BLAS nodes became 70339, at the same
  26.7 MB. A degenerate root (single-leaf BLAS, single-instance TLAS) is
  encoded as one node with the real leaf on the left and an unreachable
  sentinel leaf on the right, rather than a case in every walk. Stacks went
  32 → **64** entries, which is a proof rather than a guess now that a visit
  defers at most one side against the old two: `max_depth` 64 with the root at
  depth 0 bounds a root-to-leaf path at 64 converted nodes, hence ≤ 63 live
  entries. (The old 32 was neither — on overflow it pushed *neither* child and
  silently dropped subtrees.)

  Correctness is gated by `bvh_gpu_layout.test.cpp`: the new walk mirrored in
  C++ over the production conversion, checked ray-for-ray against a reference
  traversal of the 32-byte tree (occlusion identical; closest-hit `t` within
  1e-4 and the same primitive except on exactly-equal ties, which are counted
  and asserted rare) across one-triangle, exactly-max-leaf, random-soup and
  degenerate/duplicated meshes. `--smoke 30 PLAYGROUND_RT=both` reproduces
  `mean_luma=83.9 stddev=47.9` bit for bit. **No formal perf A/B yet** — the
  smoke run's render time moved from 3.74 ms to 2.90 ms, which is one sample
  of a number that is not a benchmark; the honest measurement waits for the
  `--rt-bench` harness (WP3).
- **Offline HQ BLAS.** Vendor tinybvh (MIT, single header) under
  `render/raytracing/internal/` as an *optional at-load builder* for static
  BLASes (SBVH + reinsertion), behind a build flag, falling back to our
  binned SAH. Never for anything refittable (SBVH cannot refit). Cache the
  result next to the model file keyed by content hash so load cost is paid
  once per asset change.
- **Hit distance out of the reflection trace.** The temporal denoiser and R3's
  virtual-position reprojection need it. *Decided (R0):* a **second read-write
  target**, R16F at the traced resolution, rather than packing into the raw
  target's alpha. Alpha already carries the roughness gate weight, and every
  packing scheme considered (moving the gate to a side target, sign/range
  encoding) traded a 2-byte-per-texel target for a subtler contract on a
  channel three passes already read. The trace writes it on every in-bounds
  thread — `hit.t / rayMaxDistance` on a hit, 1.0 on a miss or any early-out
  — and `RtRenderer::reflection_hit_distance()` exposes it. Nothing consumes
  it until R3.
- **STBN sampling.** Generate scalar + vec2 spatiotemporal blue-noise tiles
  (128×128×16) with EA FastNoise (BSD-3) offline; embed as a generated `.inc`
  (≈0.5 MB) the way `pipeline_config_golden.inc` is embedded. `rtSample2D`
  gains an STBN path selected by a uniform; R2+IGN stays as the fallback and
  the deterministic reference. Different effects offset into different slices
  (decorrelation).
- **Ray-budget accounting.** *Landed (WP3, 2026-08-02.)* `RtStats`
  (`rt_renderer.hpp`) now carries the acceleration bytes split three ways —
  `blas_bytes` / `attribute_bytes` / `tlas_bytes`, summing to the retained
  `acceleration_bytes` — plus the live `instances` count (post range filter)
  and `tlas_builds` forwarded from `RtSceneStats`. The split is computed once
  in `RtScene::upload` and copied out by a single `publish_scene_stats`, so
  the parts and the total can never disagree about which upload they came from.
  The playground prints it as a second `[smoke] rt accel ...` line. Per-signal
  dispatched-thread counts were already there and stay the honest proxy —
  GPU-side ray counters would need per-wave atomics.

  Timers: three entries now reach `RenderStats::pass_times_ms` —
  **`RTInstances`** (call site: the range filter, content hash and any TLAS
  rebuild), **`RTShadows`** (trace + the three denoiser dispatches) and
  **`RTReflectionTrace`** (trace + temporal denoise), alongside the
  pre-existing `RTReflectionResolve`. The latter two are timed *inside*
  `RtRenderer::render` rather than at the call site, because the call site
  makes one `render()` call for both signals and cannot separate them;
  `RtRenderer::set_timer_pool` lends it `SceneRenderer`'s timeline and
  `GPUTimerScope` grew a nullable-pointer overload so a pool-less RtRenderer
  times nothing.
- **`rt_max_instance_range` setting.** *Landed (WP3, 2026-08-02.)*
  `RtRange { Near = 0, Medium, Far, Unlimited }` in
  `contracts/graphics_settings.hpp`, defaulting to **Unlimited** so the filter
  is inert until dialed. `rt_range_units` maps it to 1000 / 2000 / 4000 world
  units and infinity — deliberately `DrawDistance` table entries, so the traced
  range reads as a subset of the drawn one and `Far` is exactly the D4000
  default. Infinity rather than a large sentinel, so the use site tests
  `std::isfinite` and skips the filter entirely.

  `SceneRenderer::refresh_raytracing_instances` drops an instance whose bounds
  lie **entirely** beyond the cutoff, measured to the world-space bounds (the
  same `Model::bounding_center`/`bounding_half_diag` pair the main-view cull
  uses) rather than to the origin, so a large building never pops; an
  unresolvable handle has no bounds and is kept. The filter runs before
  `set_instances`, so the content hash sees the filtered list — which is also
  the knob's price: a finite range makes the list a function of camera
  position, so crossing the threshold re-triggers TLAS rebuilds a static scene
  never pays. Measured on the playground village (`--smoke 30
  PLAYGROUND_RT=both`, 2570 instances at Unlimited): Far 2257, Medium 483,
  Near 34, with `mean_luma`/`stddev` **identical** to Unlimited at Far
  (83.9 / 47.9) — i.e. Far drops 313 instances that contribute nothing visible.
- **A/B harness.** *Landed (WP3, 2026-08-02.)* `--rt-bench N`
  (`examples/playground/main.cpp`) reuses the smoke driver wholesale — fixed
  timestep, deterministic environment, settings persistence off — with a pinned
  camera over the plaza, no screenshot, and **none** of the scripted mid-run
  settings flips, each of which would change what is being timed halfway
  through. It prints one `[rtbench] ...` record per line, fixed field order:
  a `run` header, one `pass` row per `pass_times_ms` entry with mean and p95
  CPU record time over the run (first 30 frames discarded as warm-up), four
  `frame` rows (sim / render / fence / `present_wait`), and the `rt` stats
  line. `PLAYGROUND_RT` picks the effects; `PLAYGROUND_RT_RANGE=0..3` pins the
  new range enum, which is what makes the knob measurable.

  First use immediately found something the smoke line could not: at
  `--rt-bench 120 PLAYGROUND_RT=both`, `RTInstances` is the **most expensive
  CPU pass in the frame** — roughly 4× the next one, with `tlas_builds` equal
  to the frame count, because the moving skinned character changes the content
  hash every frame and forces a full binned-SAH TLAS rebuild. That is the
  first concrete argument for TLAS refit over rebuild, and it is exactly the
  kind of claim this harness exists to license. (Ratios only — those runs were
  in an `-O0` default tree.) **The refit landed on the strength of it** — see
  the finding note at the end of §10 for the before/after table.

  Still outstanding for this bullet: the WP2 64-byte-node A/B has **not** been
  re-measured through the harness, and there is no Apple-device number. No
  traversal-layout claim in this plan is settled until both exist.

*Tests:* golden-inc + catalog rows for any new pipeline; `uniform_layout`
rows for new cbuffers; BVH conversion equivalence test (same ray set, CPU
reference traversal over both layouts → identical hit sets); STBN
determinism (fixed pixel/frame → fixed sample); docs update (`shaders/README.md`
include list).

### 5.1 Sun (existing signal, two upgrades)

Already in good shape. Two changes, both late (R5): dual fast/slow history and
A-SVGF gradient reset (§5.7) so sunrise/sunset and cloud-shadow sweeps stop
relying on the σ-clamp alone; and exposure-aware clamp thresholds (the
day↔night luminance swing is ~10³ — fixed radiance-space thresholds cannot
serve both ends).

### 5.2 R1 — Probe-field GI (the base layer, every tier)

The single highest-value addition in this plan: bounce light and correct sky
occlusion, everywhere, at a cost that *amortizes*.

**Structure.** DDGI-shaped, 3 scrolled cascades centered on the camera
(spacings ~1.5 m / 3 m / 6 m — wide enough for the village, tunable), each
~24×12×24 probes ⇒ ~21 k probes total. Per probe: 6×6 octahedral irradiance
(RGBA16F, 1-texel border) + 14×14 visibility (RG16F mean / mean² of ray hit
distance, Chebyshev leak test at sample time). ≈35 MB total. Infinite
scrolling: probe indices wrap; only the entering plane re-traces on scroll.
Probe **relocation** (nudge out of geometry by backface ratio) and
**classification** (inactive probes trace a handful of fixed rays only) come
with the structure — they are what make sparse grids survive interiors.

**Update.** Fixed ray budget per frame, round-robin over active probes,
deterministic order (probe index + frame counter — bit-identical at any
worker count, same rule as the Prepare-stage fan-out). Tier examples: High =
¼ of probes × 96 rays ≈ 500 k rays/frame (~1 ms mid-GPU class); Low = 1/16 ×
64 ≈ 85 k rays (iGPU-viable). Rays are cosine-ish distributed spherical
Fibonacci with per-frame rotation (STBN-seeded).

**Hit shading v1 — the part our engine is uniquely ready for.** A probe ray
hit shades as:

```
radiance = albedo_flat × ( sunColor · N·L · sunVisibility
                         + Σ local lights (range-windowed loop)
                         + probeIrradiance(hitPos, hitNormal) )   // bounce n-1
         | on miss: sky radiance (sample the existing sky function)
```

- `albedo_flat` is `TriangleAttributes::base_color` — already baked from the
  real albedo texture at load. The no-bindless problem is *already solved*
  for stylized content; attributes just become mandatory when GI is on
  (today they are reflections-only).
- `sunVisibility` at the hit comes from **sampling the CSM atlas** (the same
  `shadow_common.hlsli` math the forward pass uses) — a texture fetch instead
  of a second ray. Hits outside cascade coverage fall back to
  visibility 1 with the distance fade CSM itself uses. This is the
  "re-light the cache for free" trick: when the sun moves, next frame's probe
  rays see the new CSM — response lag is bounded by the accumulation
  hysteresis alone, not by ray budget.
- Local lights: brute loop over the ≤256 packet lights with early range
  reject is fine *at probe-ray counts* (hundreds of k rays, not millions);
  the froxel grid cannot serve here (world-space hits, screen-space grid).
- `probeIrradiance` at the hit is the previous frame's field → infinite
  bounce, one extra texture fetch, the standard DDGI recursion.
- Miss: the sky radiance function — probes carry the sky, which is what
  finally couples ambient to time-of-day *with occlusion* (blue sky in
  courtyards, dark under eaves — the thing `ambient_sky` could never do).

**Blend.** Per-texel EMA with change-adaptive hysteresis (GI-1.0's
Algorithm-3 shape: blend α driven by normalized new-vs-old delta, clamped, so
a weather flip or lightning drops history *locally*). Typical hysteresis
0.9–0.97; a full-field response to a step change lands in tens of frames,
which at real time-of-day rates is invisible — the adaptive α handles the
step cases.

**Consumption.**

- Forward shaders: third ambient branch — sample 8 surrounding probes with
  trilinear × normal × Chebyshev weights, multiply by the same
  `min(screenAO, ORM AO)` and ambient-lift the IBL branch uses today.
  `ambientIBL`'s diffuse irradiance-cube term is replaced; its specular term
  survives until R4. The hemisphere `ambientPBR` remains the GI-off tier.
- Volumetric fog: replace the flat `fogColor * 0.2` in-scatter ambient with a
  per-step probe fetch at the march position (SH0/mean irradiance is enough
  for media). God rays already come from CSM; this adds *colored bounce* to
  fog at ~one texture fetch per step.
- The reflection hit shader and R2's DI pass sample the same field (§5.4).

**Settings.** `GiMode { Off = 0, Probes, ProbesScreen }` +
`GiQuality { Low = 0, Medium, High }` (rays/probe × cadence × cascade count),
seven-touch-point pattern from `graphics_settings.cpp`, dev-panel combos, and
smoke line printing `gi=` state.

*Tests:* octahedral encode/decode round-trip; scroll indexing (probe →
texel → probe identity under wraps); Chebyshev weight CPU mirror; CSM-sample
parity against `shadow_common.hlsli` convention (standard-Z! — the cascades
deliberately differ from the scene's reversed-Z, keep the compute copy
honest); determinism (two runs, same probe texel digests at frame N);
leak scene (probe inside a closed hut stays dark); luma re-baseline with
`PLAYGROUND_RT=gi` added to the pin vocabulary.

### 5.3 R2 — Local-light ray-traced shadows (froxel-RIS reservoirs)

Replaces the atlas's hard budgets and hard edges with sampled soft shadows
for *every* light, at ~1 ray/pixel.

**Sampling.** Per pixel: read the pixel's froxel light list (the CPU-binned
cluster grid — same indexing math as `cluster_lighting.hlsli`, recomputed in
compute), draw M = 4–8 candidates uniformly from it, weight by the engine's
own unshadowed contribution (diffuse+spec × range window — reuse the exact
attenuation the forward loop applies so "culled ⇒ zero" stays bias-free),
keep one via weighted reservoir sampling. Merge with the motion-reprojected
previous-frame reservoir (clamp its M to ~20×). One **sphere-cap** shadow ray
to the winner (`sampleCone` generalized: light radius / distance → cosθmax —
the sun code path with a per-light aperture; spots sample the same cap, the
cone falloff lives in the target function, not the sampling domain). Optional
spatial pass (3–5 neighbors, pairwise MIS) on High only.

**Reservoir state.** 16 B/px (light id, W, targetPdf, M+flags), ping-ponged;
half-res default ⇒ ~8 MB per buffer at 1080p-class.

**Persistent light IDs — the one engine-contract change.** Temporal reuse
stores light *identity* across frames, and the packet rebuilds light arrays
every frame. `PointLightDesc`/`SpotLightDesc` (`contracts/light.hpp`) gain an
optional host-supplied `stable_id` (0 = auto), with the frame builder
assigning a quantized position/color hash when absent. The renderer keeps a
prev→current index remap table (≤256 entries, built during binning). Hosts
that animate lights per frame get correct reuse for free via the hash's
quantization; hosts that care supply IDs.

**Fold-back: two tiers, one trace.**

- *Ratio tier (default, and the iGPU answer):* output a single R8
  visibility **ratio** = Σ(picked-light contribution × V) / Σ(picked-light
  contribution), denoised by a scalar clone of the existing shadow denoiser
  (moments temporal + variance filter — the classify pass generalizes: tiles
  uniformly-1 skip filtering exactly as sun tiles do). The forward cluster
  loop multiplies its *traced-light subtotal* by the mask (one extra sampler,
  the `rtShadowTexture` wiring pattern repeated). Lights with
  `shadow_slot >= 0` drop their atlas taps when the traced mask owns them.
  Unbiased before denoise (Heitz 2018 ratio estimator), and shadow-only
  noise — the analytic lighting stays sharp.
- *Full-DI tier (High, later — R5):* demodulated diffuse + specular traced-DI
  buffers (2× RGBA16F half-res), denoised separately, remodulated in the
  forward pass, and the traced lights leave the analytic loop entirely. This
  is the ReSTIR-proper endgame; it ships only after the ratio tier has
  soaked, because it doubles denoiser surface.

**Atlas coexistence.** `LocalRtShadowMode { Off = 0, Ratio, Full }`. Off keeps
today's atlas path untouched. Ratio/Full retire atlas *tiles* for traced
lights (freeing the 6-updates/frame budget for any lights the tier leaves
untraced — e.g. a per-light host opt-out for gameplay-critical hard shadows).
The atlas code does not die; it is the fallback tier and the
`BRACKEN_ENABLE_DEV_UI`-style insurance.

*Costs (1080p-class, half-res):* ~0.52 M rays + reservoir passes ≈ 1–1.7 ms
mid-GPU; the pass never runs on froxels with 0 traced lights (tile-classified
like the sun denoiser). *Tests:* WRS math CPU mirror (streaming merge ==
batch); froxel-index parity with `cluster_lighting.hlsli`; light-ID remap
under add/remove churn; energy A/B vs. atlas scene (mean luma within band);
penumbra widens with light radius (analytic disc occluder test scene);
determinism.

### 5.4 R3 — Honest hit lighting (reflections meet GI)

One shared include, `rt_hit_shade.hlsli`, used by probe rays and reflection
rays (and later screen-probe rays):

```
hitRadiance = albedo_flat × ( sun N·L × CSM(hitPos)
                            + local lights (range loop, no shadow)
                            + probeIrradiance(hitPos, hitNormal) )
            + emissive_flat                      // R6
```

For the reflection pass this deletes the flat `skyColor * 0.25` ambient and
the unshadowed sun — reflections start showing bounce-lit interiors, shadowed
facades, and time-of-day-correct ambience, for the price of a CSM fetch and a
probe fetch per hit. Misses keep returning the real sky.

Plus the two reflection-denoiser upgrades from the research:

- **Virtual-hit-point reprojection** using the R0 hitT output: reproject the
  reflected *virtual* position (eye ray extended through the surface by hitT)
  for low-roughness pixels, surface motion for rough ones, blended by
  roughness — kills the smearing that today forces `max_accum_frames` down
  to 16.
- **Rough-tier fallback goes local.** Above the trace gate, today's IBL probe
  is sky-only. Post-R1, high-roughness specular samples the *probe field*
  (irradiance is a fine specular proxy at high α) blended with IBL by the
  existing `reflection_gate.hlsli` weights — rough metal in a courtyard stops
  mirroring open sky it cannot see. (`ambientIBL`'s specular keeps serving
  mid-roughness until R4's screen probes.)

*Tests:* hit-shade include compiled into both entry shaders stays
binding-compatible (catalog counts); reflection luma A/B (interiors darken,
lit walls brighten — banded assertions per pinned ToD); virtual-reproj
ghosting scene (orbiting camera over water/mirror floor, stddev of frame
delta must drop vs. baseline).

### 5.5 R4 — Screen-probe gather (the quality tier)

GI-1.0's shape over our tracer, `GiMode::ProbesScreen`, mid/high GPUs only:

- **HZB.** New min-depth pyramid pass after the prepass (the engine's first —
  also unlocks future occlusion culling). Enters the `gpu/` layer with the
  indirect-dispatch wrapper (`ComputePass::dispatch_indirect`,
  `GPUBuffer::Type::Indirect`).
- **Probes.** One 8×8 octahedral probe per 8×8 screen tile (adaptive extras
  where depth/normal interpolation fails — the indirect dispatch consumer),
  placed on G-buffer depth, jittered. ~32 k probes at 1080p ⇒ ~2 M ray
  *slots*, most resolved cheaply:
- **Trace ladder per ray:** HZB screen march first (hits cost texture
  fetches; dynamic/skinned objects included via the depth buffer — the
  bind-pose problem sidestepped where it is most visible); on miss/handoff
  a *short* BVH ray (few-meter clamp); beyond that, sample the R1 probe
  field. The world field bounds the per-ray cost — screen probes buy
  *sharpness near geometry*, not range.
- **Filtering in probe space** (temporal reproject + 3×3 probe spatial with
  plane/depth weights, then per-pixel 4-probe integrate against the BRDF) —
  ~0.6 ms-class where pixel-space GI denoising costs 2.5+ ms. No per-pixel GI
  denoiser exists in this plan at any tier.
- Screen probes also serve **mid-roughness specular** (importance-sample the
  probe's radiance by the GGX lobe) — this is when `ambientIBL` retires to
  sky-visible-surfaces duty and the reflection trace can hand off *earlier*
  (gate `fade_start` drops), cutting reflection rays on rough scenes.

*Costs:* GI-1.0 measured 1.9–3.4 ms @1080p on 6900 XT/3080-class for the full
stack over *its* tracer; budget 2.5–4 ms on our mid tier, gated by
`GiQuality`. *Tests:* probe placement determinism; HZB correctness (mip N min
== reference); handoff continuity scene (wall half on/half off screen — no
seam in indirect); tier A/B screenshots.

### 5.6 R5 — Near field, denoiser core, full-DI

- **SSILVB upgrade of GTAO**: same horizon loop, sectors become a 32-bit
  visibility bitmask (thin-occluder correctness), outputs AO + bent cone +
  optional half-res near-field indirect from last frame's lit color. Bent
  cone gates the probe/screen-probe irradiance (directional occlusion instead
  of scalar AO) — and the double-count rule is explicit: SS indirect *replaces*
  probe irradiance within its sampled radius, never adds.
- **Denoiser core evolution** (shared by sun/local/reflection paths):
  dual fast/slow history with fast-clamp; exposure-aware clamp and firefly
  thresholds (feed the adapted EV from the exposure SSBO); hit-distance
  moments for penumbra-width-driven filter radius (SIGMA's idea) on both
  shadow signals; **A-SVGF-style gradient pass** — re-trace 1-in-9 pixels
  with the *previous* frame's sample (replayable sampler from R0), dense
  small à-trous reconstruction of the gradient, gradient drives per-region
  history reset for shadow + reflection + probe temporal passes. This is the
  mechanism that makes lightning, cloud shadows, and light toggles clean at
  every tier.
- **Full-DI tier** of local lights lands here (§5.3's second fold), reusing
  the matured denoiser core.

### 5.7 R6 — Emissive meshes, area lights, stretch caches

- **Emissive extraction** at load (the loader already walks materials for
  `rt_triangle_base_colors`; emissive factor joins it): per-instance emissive
  triangle lists → power-weighted alias table (Vose, CPU) → these join the
  froxel candidates as a per-froxel "emissive present" flag + global alias
  fallback, and `rt_hit_shade.hlsli` gains the `emissive_flat` term (windows
  glow in reflections and feed probes — lit windows at dusk bouncing onto
  the street is the money shot of this whole plan).
- Sphere/disk **area-light** polish for the DI pass (solid-angle sampling is
  already correct; this is authoring surface: radius on `PointLightDesc`).
- **Stretch, only if screen feedback + probe recursion measurably starve
  secondary lighting**: SHaRC-shaped world hash radiance cache (reimplemented
  from the papers — the NVIDIA repo's license forbids lifting), 2²⁰ entries ≈
  40 MB, updated by the rays we already cast. Enters only with a measured
  deficiency in hand.
- **Watch list re-evaluation**: Split Radiance Cascades (the July 2026
  preprint) as a potential probe-field replacement; MegaLights-style
  fixed-budget tile importance as a reservoir simplification; DGF's Vulkan
  extension.

### 5.8 Dynamic & skinned geometry policy (cross-cutting)

Explicit, per signal — today's silent bind-pose limitation becomes a stated
contract:

| Signal | Dynamic-mesh answer |
|---|---|
| Sun shadow | Bind-pose BLAS (status quo) + **capsule occluders** (per-bone capsules from the skeleton — analytic-cheap, soft, and *animated*); revisit CPU re-skin + refit (plain-SAH BLASes only) if fidelity demands |
| Local-light DI | Same as sun (the capsule set answers both) |
| GI probes | Bind-pose is fine (probe-scale error invisible); characters *receive* GI via normal probe sampling |
| Screen-probe GI / reflections | **Screen traces carry them** (in the depth buffer + last frame's lit color) — the Lumen answer, free once R4 lands |

Capsule occluders are a small analytic intersection in the shadow kernels
(`animation/attachment.hpp` already reads joint world transforms), traced
before the BVH walk; ≤ a dozen capsules per character.

---

## 6. Scalability ladder

Per-effect toggles stay orthogonal (house style); the table is the *intended*
preset mapping, each row honest about its floor. "RT total" is the target
budget at 1080p-class internal resolution.

| Tier (example HW) | Sun | Local lights | Diffuse GI | Reflections | RT total |
|---|---|---|---|---|---|
| Off (any) | CSM | atlas | hemisphere/IBL ambient | IBL/SSR-water | 0 ms |
| Floor (Deck / base M) | RT Quarter *or* CSM | atlas | `Probes` Low (85 k rays/f) | off | ~1.5–2.5 ms |
| Bronze (780M / M3 / consoles-min) | RT Half | atlas | `Probes` Medium | Quarter, rough→probes | ~3–4 ms |
| Silver (4060 / 7600) | RT Half–Full | `Ratio` Half | `Probes` High | Quarter–Half + virtual reproj | ~4–5.5 ms |
| Gold (4070 Ti+ / M-Max) | RT Full | `Ratio`→`Full` Half–Full | `ProbesScreen` | Half + screen-first | ~5–7 ms |
| Ultra (4090-class) | Full | `Full` Full-res | `ProbesScreen` High | Full/Half + earlier handoff | ~3–5 ms (headroom) |

Ladder knobs, in impact order (all scoped enums in `GraphicsSettings`):
per-effect resolution (`RtResolution`, exists) → `GiMode`/`GiQuality` →
`LocalRtShadowMode` → `rt_max_instance_range` → roughness handoff
(`rt_reflection_max_roughness`, exists) → BLAS LOD source (stretch: build
BLASes from `load_model_with_lods` level 1 on low tiers — the Nixxes
"geometry detail" knob, nearly free since LODs already exist).

Memory envelope (Gold, 1080p): probes ~35 MB + reservoirs ~16 MB + screen
probes ~10 MB + STBN 0.5 MB + hitT/history deltas ~10 MB ≈ **~70 MB** on top
of today's ~29 MB acceleration + targets. Floor tier: ~12 MB (small probe
volume only).

---

## 7. Phase order & exit criteria (summary)

Dependency-ordered; each phase ships alone.

| Phase | Delivers | Exit criteria (beyond "suite green, docs updated") |
|---|---|---|
| **R0** | shared traversal include, 64-B node A/B, HQ-BLAS option, hitT out, STBN, stats/timers, `rt_max_instance_range`, `--rt-bench` | zero visual change (luma bands identical); measured traversal verdict recorded in this doc; bench harness in CI-runnable form |
| **R1** | probe-field GI + fog coupling + settings + smoke pins | bounce visible in pinned art shots (`PLAYGROUND_TOD_HOUR` sunset: west walls warm); closed-hut leak test; Floor-tier budget met on the weakest local box; luma bands re-baselined for `PLAYGROUND_RT=gi` |
| **R2** | froxel-RIS local shadows (Ratio), light IDs, scalar denoise clone | 32-light night scene fully shadowed (atlas caps exceeded) within budget; penumbra-vs-radius test; atlas A/B luma parity on atlas-sized scenes |
| **R3** | `rt_hit_shade.hlsli`, CSM-at-hit, probe-at-hit, virtual reproj, rough→probe fallback | reflection interiors/shadows correct in pinned shots; orbit-ghosting metric improves; no reflection cost regression >10 % |
| **R4** | HZB + indirect dispatch wrapper, screen probes, probe-space filtering, mid-rough specular from probes | `ProbesScreen` vs `Probes` A/B sharpness; handoff-seam scene clean; Silver budget held |
| **R5** | SSILVB/bent-cone, denoiser core (gradients, dual history, exposure-aware), Full-DI tier | lightning/toggle scene: history resets locally within 2 frames, no global flash-smear; contact-GI A/B |
| **R6** | emissive mesh lights, area-light authoring, stretch cache decision | dusk-windows shot; emissive in reflections + probes; watch-list re-evaluation recorded here |

Standing rules for every phase: TSAN suite run after any job-system-adjacent
work (HQ builder, probe update scheduling); every new cbuffer gets its
`BRACKEN_HLSL_CBUFFER` mirror + `uniform_layout` row; every new pipeline gets
its catalog row + golden-inc update; every new setting follows the seven
touch points (enum → field → clamp → save → load → test → dev panel);
`docs/architecture.md` finally gets the **"Ray tracing" section its module
map already cites** (fix the dangling reference in R0), and
`docs/technical_spec.md`'s settings table gains the `rt_*`/`gi_*` rows it
currently lacks.

---

## 8. Risks & mitigations

- **iGPU throughput falls short of even the Floor tier.** The math says
  15–60 M incoherent rays/s. Mitigation is structural, not hopeful: the only
  per-pixel ray on Floor is the (quarter-res, any-hit, clamped) sun ray —
  everything else is amortized (probes) or off. If even that misses, Floor
  becomes probes-only + CSM, which is still a generational look upgrade over
  today's flat ambient.
- **Wave divergence (limit #6) caps reflection gains before traversal does.**
  Half-res tiles double effective coherence; the R4 rough-handoff removes the
  scattered-glossy texels that cause the divergence; revisit binning only
  after those land (research: +7 % at best at 1 rpp — not worth it earlier).
- **Temporal reuse vs. the pipelined frame.** Reservoirs and gradients read
  previous-frame state; the frame is pipelined (`on_simulate(i+1)` overlaps
  render i). All RT history lives renderer-side (main-thread tier, like the
  existing denoiser histories) — no packet crossing, no new tier exceptions.
  The light-ID remap builds during binning (already main-thread).
- **CSM-at-hit convention trap.** The cascade atlas is deliberately
  standard-Z while the scene is reversed-Z; the compute-side sampler must
  reuse `shadow_common.hlsli` logic, not reimplement it. (Called out because
  it is exactly the kind of silent-wrongness the uniform tests exist for —
  add a parity test.)
- **Settings/League-of-knobs sprawl.** Every knob above maps to one enum with
  one dev-panel row and appears in the smoke `[smoke] settings` line;
  presets live host-side. The ladder table in §6 is the contract.
- **License hygiene.** NRD, NVIDIA STBN, SHaRC: ideas yes, code no. FFX
  (MIT), tinybvh (MIT), FastNoise (BSD-3), Capsaicin (MIT): code with
  attribution. Record provenance in headers when porting kernels.
- **Scope creep toward a second scene representation.** The single most
  repeated lesson across the research (Lumen SDF+cards, Brixelizer bricks,
  Godot's four-year SDFGI→HDDAGI grind): the structure eats the project.
  Standing rule: **this plan adds no world structure other than the probe
  clipmap** (fixed-size, engine-owned, ~35 MB). Any proposal for
  voxels/SDF/surfels/hash-caches must arrive with a measured deficiency the
  probe field + screen feedback cannot cover (the R6 stretch gate).

---

## 9. What we are deliberately not doing

CWBVH/8-wide traversal (portable-compute evidence says BVH2 wins without
vendor intrinsics — recorded verdict, revisit only with new measurements);
GPU BVH builds (our BLASes are static, TLAS is 0.49 ms on CPU to build and
refits for a fraction of that); persistent
threads and ray sorting at ≤1 rpp; dense 3D radiance cascades (memory ∝
volume³); 2D radiance cascades (camera-constrained); Lumen surface cache,
mesh SDFs, Brixelizer bricks, surfel GI (second-structure rule); NRD/DLSS-RR/
FSR-RR/NRC/cooperative vectors (licenses, ML hardware, APIs we do not have);
light BVHs and ReGIR below ~1 k lights; Area ReSTIR (rasterized primaries);
checkerboard rendering (half-res + probes already cover the budget).

---

## 10. The one-frame picture, after R6

```
CPU: cluster binning (+light-ID remap) → uploads → CSM → local-atlas (fallback lights)
GPU: depth/G-buffer prepass → HZB → GTAO/SSILVB
     → probe update (budgeted rays: screen-first→BVH→probe/sky; CSM+lights at hit)
     → sun shadow trace+denoise            (unchanged shape)
     → local-DI RIS + reservoir + 1 ray → ratio/full denoise
     → gradient pass (1/9 replay, drives all history resets)
     → reflection trace (hitT, hit-shade v2) + virtual-reproj temporal
     → screen-probe gather (tier) → forward lighting (probe ambient, traced masks)
     → reflection resolve → water → TAA → exposure → bloom → fog (probe-lit) → composite
```

Six ray-consuming passes, one acceleration structure, one probe field, no
per-pixel GI denoiser, every signal with a raster fallback tier, and every
budget a named setting.

---

**First `--rt-bench` finding (2026-08-02):** `RTInstances` (TLAS hash + rebuild
+ upload) is the most expensive CPU pass in the playground frame — ~0.54 ms
mean with `tlas_builds=119` over a 120-frame run, because the animated
character changes the instance content hash every frame; the "static scene
builds once" contract never engages in a live scene. A TLAS **refit** path
(rebuild topology only on instance add/remove; refit node bounds when only
transforms changed) and/or a static/dynamic TLAS split is now a tracked R0
follow-up — it should land before R2, whose reservoir work assumes TLAS cost
stays flat. (Numbers from a default `-O0` tree: ratios meaningful, absolutes
not.)

**Landed (R0 follow-up, 2026-08-02): TLAS refit.** `RtScene::set_instances` is
now a three-case decision taken from two hashes computed in one pass — content
unchanged (do nothing), topology unchanged (**refit**), topology changed (full
rebuild) — where topology is the instance count plus the handle *sequence*.
The refit recomputes an inverse and an eight-corner world AABB only for the
instances whose transform actually differs from last frame (the playground: one
skinned character out of 2570), then `refit_tlas` (`bvh.hpp`) re-unions every
node bottom-up over the existing topology as a plain reverse index walk — the
builder guarantees children sit at higher indices than their parent, which is
the same invariant the 64-byte conversion already relies on. The 32-byte tree
stays authoritative and the 64-byte upload form is re-derived, so
`convert_to_gpu_nodes` remains the single producer of every uploaded byte.
Refit quality is gated by `sah_cost` against the last full build (rebuild past
1.5×) with a 64-refit counter backstop; `RtSceneStats`/`RtStats` gained
`tlas_refits` and both playground lines print the pair.

Measured, same `-O0` default tree, `--rt-bench 120 PLAYGROUND_RT=both`, before
→ after:

| | mean_ms | p95_ms | tlas_builds | tlas_refits |
|---|---|---|---|---|
| `RTInstances` before | 0.5439 | 0.5871 | 119 | — |
| `RTInstances` after | 0.1200 | 0.1485 | 2 | 117 |

4.5× on the mean, 4.0× on p95, and `RTInstances` is no longer 4× the next
pass — it is now ~2× `ShadowPass` and the residual is the content hash over
2570 instances plus the upload, not a build. `--smoke 30 PLAYGROUND_RT=both`
reproduces `mean_luma=83.9 stddev=47.9` bit for bit, which is the evidence the
refitted tree answers rays identically in the real scene; the seeded
refit-vs-rebuild ray equivalence proof is `tlas_refit.test.cpp`. (Ratios
meaningful, absolutes not.)

## 11. Immediate next actions

1. R0 in full (it is small, zero-risk, and the bench harness de-risks every
   later claim).
2. R1 probe field behind `GiMode::Off` default — the single biggest visual
   payoff and the dependency of R3/R4/R6.
3. Record the R0 traversal A/B verdict in §5.0 and prune this plan's watch
   list accordingly.

---

## 12. References

Direct lighting / ReSTIR: Bitterli et al., *Spatiotemporal Reservoir
Resampling* (SIGGRAPH 2020); Wyman et al., *A Gentle Introduction to ReSTIR*
(SIGGRAPH 2023 course); Kozlowski & De Francesco, *Cyberpunk 2077 ReSTIR
Integration* (2023); Boksansky et al., *ReGIR* (Ray Tracing Gems II, 2021);
Lin et al., *GRIS* (SIGGRAPH 2022); Estévez & Kulla, *Adaptive Tree
Splitting* (2018); Heitz, Hill, McGuire, *Analytic Direct Illumination +
Stochastic Shadows* (i3D 2018); Epic, *MegaLights* (UE 5.5 docs; SIGGRAPH
2025 Advances); NVIDIA RTXDI 2.x SDK; Zhang et al., *Area ReSTIR* (SIGGRAPH
2024).

Caches / probes: Majercik et al., *DDGI* (JCGT 2019) + RTXGI-DDGI SDK docs
(relocation, classification, scrolling); Boissé et al., *GI-1.0* (AMD
2022/23) + Capsaicin (MIT); AMD *Brixelizer GI* (GDC 2024, FidelityFX SDK
1.1, MIT); NVIDIA *SHaRC* (RTXGI 2.0, 2024 — license-restricted, docs only);
Apers et al., *Surfel GI* (SIGGRAPH 2021) + EA College Football 25 (2024);
Wright et al., *Lumen* (SIGGRAPH 2022 Advances; UE 5.x docs); Godot
SDFGI/HDDAGI (PRs #86007/#119869).

Radiance cascades / screen space: Sannikov, *Radiance Cascades* (2023–25,
PoE2); Freeman, Sannikov, Margel, *Holographic RC* (arXiv 2505.02041);
Freeman & Sannikov, *Split Radiance Cascades* (arXiv 2607.20384, Jul 2026);
Therrien et al., *SSILVB* (2023); Jimenez et al., *GTAO/GTSO* (2016);
Stachowiak, *Stochastic SSR* (2015) and *Tiny Glade* rendering (GPC 2024).

Traversal / builds: Ylitie, Karras, Laine, *Compressed Wide BVHs* (HPG 2017);
Meister et al., *BVH survey* (CGF 2021); Bikker, *tinybvh* (MIT, v1.8.0,
2026) + *BVH quality: beyond SBVH* (2025); Benthin et al., *PLOC++* (HPG
2022), *H-PLOC* (HPG 2024); AMD *DGF* (HPG 2024), *DOBB-BVH* (2025); Usher,
*Ray Tracing on the Apple M1* (2020); Anagnostou, *Ray binning* (2022);
*Megakernel vs Wavefront* (arXiv 2605.27323, 2026); AMD *HIPRT* (arXiv 2026).

Denoising / sampling: Schied et al., *SVGF* (2017), *A-SVGF* (2018); NVIDIA
NRD (ReBLUR/ReLAX/SIGMA — license-restricted, papers/README only); AMD
FidelityFX Denoiser / SSSR / Hybrid Shadows (MIT); Wolfe et al., *STBN*
(EGSR 2022) + EA *FastNoise* (BSD-3); Burley, *Hash-based Owen Scrambling*
(JCGT 2020); Bitsquid, *Reprojecting Reflections* (2017); Salvi, variance
clipping (TAA lineage); Epic, *Lumen Performance Guide*.

URLs for all of the above were collected during the 2026-08-02 research
sweep; ask the repo history for the full annotated list if a link is needed.

---

## Appendix A — the API-floor question (2026-08)

*"Would higher Metal / Vulkan / D3D12 versions get better results?"* Yes,
dramatically — but the lever is not the version number. The versions already
deployed on target hardware contain everything this plan could want; what
stands between the engine and them is **SDL3 GPU's feature envelope**. The
engine compiles SPIR-V at `vulkan1.0` (opt-in 1.1 per shader via
`SHADER_WAVE_OPS`), requests MSL 2.2, and SDL's DXIL contract is pinned at
SM 6.0 (`SDL_GPU_SHADERFORMAT_DXIL`, `SDL_gpu.h`). SDL's own in-header FAQ on
ray tracing / mesh shaders: *"not in the 'never' category… definitely not
'near future'"*. Raising a `-fspv-target-env` flag gains nothing unless the
backend also enables the device feature and the abstraction exposes the API —
so "use higher versions" really means "outgrow SDL3 GPU."

What the locked-out features would be worth, against this plan:

| Unlock | Where it lives | Effect on this plan |
|---|---|---|
| **Hardware ray query** (VK `ray_query`, DXR 1.1/SM 6.5, Metal 3 on M3+/A17+) | all three APIs since ~2020–23 | Roughly an order of magnitude on incoherent rays. The §2 throughput table stops binding: full-res effects on mid GPUs, the R5 Full-DI and per-pixel tiers everywhere, Deck's RDNA2 included. **The architecture is invariant** — only `sceneOccluded`/`sceneClosest` and the AS build/upload swap out (our BLAS/TLAS maps 1:1 to driver acceleration structures); reservoirs, probes, denoisers, caches are API-agnostic compute. R0's `rt_trace.hlsli` extraction is deliberately that seam |
| **Bindless** (VK 1.2 descriptor indexing, SM 6.6 dynamic resources, Metal 3 argument buffers) | core in all three | Real materials at hit points: textured/emissive hit shading, retiring the flat-color proxy. Moderate win for stylized content (flat colors already earn their keep), large for engine generality |
| **fp16 arithmetic** (SM 6.2 + `shaderFloat16`, native half in MSL) | broadly available | 1.3–2× ALU/bandwidth in denoisers, probe math, quantized nodes — biggest on the iGPU floor |
| **SER + opacity micromaps** (DXR 1.2, 2025; next DXR preview summer 2026) | newest HW | Shader execution reordering attacks exactly the measured wave-divergence bound in the reflection pass (limit #6) |
| **Cooperative vectors/matrix** (SM 6.9 preview, VK extensions) | 2025–26 wave | Would un-block the NRC-class neural caches §3 marks API-locked |

Also worth naming: several current pains are SDL-envelope pains at *any*
version — GPU timestamps (a Vulkan 1.0 feature SDL doesn't surface, why
`gpu_ms` is forever `nullopt`), async compute, indirect *count* draws.
SDL 3.4 did grow opt-in device-create feature booleans (clip distance, depth
clamping, anisotropy…), which is the plausible upstream mechanism if RT ever
lands there (community request: SDL issue #11147).

**`SDL_GPUVulkanOptions` (SDL 3.4.0) — the sanctioned crack in the
envelope.** Passed via `SDL_PROP_GPU_DEVICE_CREATE_VULKAN_OPTIONS_POINTER` at
device creation, it lets the app request a higher Vulkan API version (default
without it: 1.0, minimal features), enable arbitrary device-feature chains
(`VkPhysicalDeviceVulkan12Features` etc.), and require extra instance/device
extensions. The dividing line for what it can unlock: **features that are
purely shader-visible flow through SDL's existing binding model; features
that need new API objects, descriptor types, or command types do not.**

- *Reachable with it* (Vulkan backend): fp16 + 16-bit storage (this
  appendix's row 3 — no fork needed; pair with a `SHADER_FP16` compile list
  next to `SHADER_WAVE_OPS`, `-enable-16bit-types
  -fspv-target-env=vulkan1.2`), the full extended-subgroup-op set, and
  subgroup size control (which retires the "wave width unknowable" caveat on
  that backend). The engine already creates its device with
  `SDL_CreateGPUDeviceWithProperties` (`gpu_device.cpp`), so wiring the
  struct in is a small change — device creation is try-with/retry-without,
  since SDL exposes no feature *query*, only enablement. Linux + Deck,
  exactly where the fp16 win lands, are the Vulkan targets.
- *fp16 per backend, precisely*: **Metal needs no options struct at all** —
  SPIRV-Cross translates 16-bit SPIR-V types to native MSL `half`, every
  Apple GPU runs fp16 at double rate, and the requested MSL 2.2 floor
  already covers it. **D3D12 is a hard no through the shipped stack**: the
  vendored SDL_shadercross compiles its DXIL at `cs_6_0` (hard-coded profile
  strings in `SDL_shadercross.c`) and DXC rejects `float16_t` below SM 6.2 —
  plus SDL surfaces no `Native16BitShaderOps` capability query. So the fp16
  tier ships as a dual-compiled variant set (fp32 `.spv` + fp16 `.spv` per
  listed shader): Metal always takes fp16, Vulkan takes it when the device
  came up with the features, D3D12 stays fp32 until shadercross grows a
  shader-model knob (worth an upstream patch — it is a small one).
  `min16float` (relaxed precision, legal at SM 6.0/Vulkan 1.0) is the only
  D3D12-reachable hedge today; it is a hint drivers may ignore, and
  SPIRV-Cross's MSL path does not reliably demote it to `half` — use real
  16-bit types where they're supported instead. The D3D12 gap is also
  sidesteppable wholesale: **SDL's Vulkan backend is fully supported on
  Windows** — D3D12 is merely first in the platform's default probe order —
  and a host can force it per device with `SDL_HINT_GPU_DRIVER`/the
  device-create name property. A game that ships Vulkan-on-Windows gets one
  PC backend (Windows/Linux/Deck) with `SDL_GPUVulkanOptions` everywhere,
  keeping D3D12 as the fp32-tier fallback for old or driver-broken hardware —
  free, since the same `.spv` set ships regardless. Scope note: fp16 belongs in
  the bandwidth/register-bound passes (denoisers, probe blend/filter,
  screen-probe filtering) and **never in traversal** — intersection math
  needs fp32 for watertightness.
- *Not reachable*: hardware ray query — you can *enable*
  `VK_KHR_ray_query`/`VK_KHR_acceleration_structure` with it, but SDL's
  pipeline layouts have no acceleration-structure descriptor type and its
  command buffers no AS-build call, so there is nothing to bind or build
  against; likewise true bindless (no update-after-bind layouts), timestamps,
  async compute. Enabling an extension is not the same as having API surface
  to call it.
- It is Vulkan-only (no Metal/D3D12 counterpart), so anything adopted through
  it keeps the fp32/scalar fallback path — the same per-shader opt-in shape
  the wave-ops list already models.

Net effect on the table above: **fp16 moves from "outgrow SDL3 GPU" to
"available on the Vulkan backend today, optional tier"**; the ray-query and
bindless verdicts are unchanged.

**Options, in ascending commitment:** (a) stay pure SDL3 GPU — this plan is
designed for that floor and loses none of its features, only its ceiling;
(b) patch/fork SDL's backends to expose ray query behind a feature boolean —
medium effort, permanent three-backend maintenance; (c) grow the engine's own
`gpu/` backend beside SDL — the layer is already the engine's own abstraction
(`GPUDevice`/`ComputePass`/`PipelineRegistry`), so the blast radius is
contained to `gpu/`, but it is months of swapchain/sync/allocator work times
three platforms, i.e. exactly what adopting SDL avoided.

**Decision of record: proceed with the plan unchanged on option (a), and keep
the two seams clean** — the `rt_trace.hlsli` traversal boundary and the
`gpu/` abstraction boundary — so hardware RT arrives later as a traversal
backend swap plus a tier-table update, not a redesign. Revisit after R2/R3
with measured software-tracing costs in hand; re-check SDL upstream and the
summer-2026 DXR preview then.
