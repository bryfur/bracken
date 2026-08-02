# Ray-Traced Lighting — Research & Plan of Record

Status: **in progress — R0 (foundations), R1 (probe-field GI), R2
(froxel-RIS local-light shadows), R3 (honest hit lighting), R4 (the HZB +
indirect dispatch in R4a, the screen-probe gather in R4b) and R5a (the SSILVB
visibility bitmask + bent-cone GI gating) have landed**; the rest of R5 and R6
are still proposal. Written 2026-08-02 against the working tree at the
RT-denoiser series (post "RT shadows: stop multiplexing ambient occlusion into
the shadow mask"); R0's landed work packages are marked in §5.0 and R1's three
packages in §5.2. As of R1c the probe field lights the frame — the first
intentional image change of the plan.

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
files that do not exist yet, which check (a) of that test would fail. The
earlier plan was to add it once R1 landed; R1 has landed and that is **still
wrong**, for a reason worth recording rather than silently deferring: §5.0 cites
an STBN `.inc` that does not exist, so check (a) would fail on a document that is
accurate about everything it claims. (§5.4's `rt_hit_shade.hlsli` was the other
half of that argument until R3 created it.)
The right trigger is "no unlanded phase cites a file that does not exist" —
i.e. after R6, or an earlier split of this document into a landed-history half
and a proposal half. Everything cited from the *current* tree below was
verified by hand.

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
2. ~~Reflection hit lighting is wrong twice: the sun term is unshadowed and the
   ambient is a flat constant. Reflections show no GI and no shadows.~~
   **Fixed by R3** — reflection and probe rays now share
   `shaders/rt_hit_shade.hlsli`. See §5.4.
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
7. ~~No HZB, no previous-frame depth/normal, no indirect dispatch wrapper~~,
   no GPU timestamps (SDL3 has none — CPU record times only). **R4a added the
   HZB and `ComputePass::dispatch_indirect`** (§5.5); no previous-frame
   depth/normal target still stands.

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

**Landed (R1a, 2026-08-02) — the structure, and only the structure.**
`render/gi/` (namespace `render::gi`) is in the tree: `ProbeVolume`
(`render/gi/probe_volume.hpp`) owns the three cascades' atlases, and
`gi_math.hpp` is the header-only CPU source of truth for every mapping R1b's
shaders will mirror — torus wrap, the atlas tiling, octahedral encode/decode
(the `octDecode` convention already in `shaders/rt_trace.hlsli`, not a second
one), and the world↔probe mapping per cascade. Shipped numbers: 3 cascades ×
24×12×24 probes, spacing 150/300/600 world units (1.5/3/6 m at the playground's
100 units/metre), irradiance 8×8 texels/probe RGBA16F (2304×192, 3.375 MB),
visibility 16×16 RG16F (4608×384, 6.75 MB), probe state 1 texel R8_UINT
(288×24) — **10.13 MB per cascade, 30.4 MB for the field**, comfortably inside
the ~35 MB this section estimated. Tiling is column = `x + GRID_X*y`, row = `z`,
i.e. the linear probe index split at `PROBES_PER_ATLAS_ROW`, so there is exactly
one mapping and not two.

Scrolling works and is tested: cascades snap to whole probe steps, torus
addressing keeps a surviving probe's texels, and `for_each_entering_probe`
defines "entered" as *in the new window, not in the old* rather than as a
per-axis plane list — the same set, but it cannot double-count the corner a
diagonal scroll's two arms share, and a jump past one grid width degenerates to
a full invalidation with no special case. A scrolled cascade re-uploads its
whole 6912-byte state texture (worst case 20.25 KB/frame across three cascades)
instead of carrying dirty rects.

`GiMode { Off = 0, Probes, ProbesScreen }` and `GiQuality { Low, Medium, High }`
landed with all seven touch points and a dev-panel combo. Two are deliberately
ahead of their implementation, both documented as such in the enum, the tooltip
and the docs: `ProbesScreen` behaves as `Probes` until R4, and `gi_quality` is
persisted but read by nothing until R1b. `SceneRenderer` owns the volume
lazily (created on the first non-Off frame, destroyed when it returns to Off)
and scrolls it in a `GIProbes` timer scope; the playground pins it with
`PLAYGROUND_GI` and prints a `[smoke] gi ...` line.

**Nothing consumes the field.** No ray is cast into it, no shader samples it,
every probe stays uninitialized, and the default (`Off`) path is byte-identical
— `--smoke 30` reproduces its baseline luma exactly with GI on *or* off, which
is the correctness claim R1a is allowed to make. Probe relocation and
classification are R1b's, with the update pass. What follows in this section is
still the plan, not the tree.

**Landed (R1b, 2026-08-02) — the update: probes trace, accumulate and
classify.** The field now fills itself. Four compute passes per updating cascade
(`gpu::ComputePipelineType::GiProbe*`, all inside the one `GIProbes` timer):

- `gi_probe_trace.comp.hlsl` — one thread per (scheduled probe, ray).
  Spherical-Fibonacci directions over the whole sphere, rotated per frame by a
  uniform-random quaternion derived from the frame counter through the R3
  sequence and Shoemake's construction — deterministic, mirrored on the CPU in
  `gi_math.hpp` and tested for determinism, orthonormality, coverage
  (worst-case best dot ≥ 0.85/0.90/0.93 at 32/64/96 rays), separation and
  balance. Closest-hit through `rt_trace.hlsli`'s shared walks. Hit shading v1
  exactly as specified above, with one thing better than planned: rather than
  transliterating the cascade math, the shader's cbuffer embeds
  `ShadowDataUniforms` field for field under the names `shadow_common.hlsli`
  requires and **includes that file**, calling the same `calcShadow()` the
  forward pass calls. §8's "CSM-at-hit convention trap" is therefore not a
  parity test but a structural impossibility. Misses sample `IBL::capture_map()`
  — the raw sky-radiance capture cube, newly exposed for this — with a uniform
  flag falling back to the `EnvironmentState` hemisphere when IBL is
  unavailable. Output per ray: RGB radiance + a **signed** distance, negative on
  a backface.
- `gi_probe_blend.comp.hlsl` / `gi_probe_blend_visibility.comp.hlsl` — one
  threadgroup per probe, one thread per tile texel, `[numthreads]` equal to the
  tile. Cosine weighting for irradiance, a `cos^50` lobe and a 1.5×spacing
  distance clamp for the visibility moments. **The border ring is folded into
  the same dispatch** via `groupshared` + one barrier rather than being a fourth
  pass: it is a pure function of the payload the group just computed, so a
  separate pass would re-read the atlas it had only just written to copy 28 of
  64 texels. The blend is the planned change-adaptive EMA, 0.97 → 0.70 on a
  scale-free delta; measured response is 16 updates for a full dark→lit step and
  65 for a 5% nudge, both pinned by test. Uninitialized probes take the estimate
  wholesale.
- `gi_probe_classify.comp.hlsl` — backface fraction per probe; >25% steps the
  probe 0.25 spacings away from the resultant backface direction, clamped to
  0.45; >75% while already at the clamp marks it inside geometry. The verdict
  and the offset live in a new per-probe RGBA16F texture rather than in
  `probe_state`, because that texture has a CPU writer (the scroll re-uploads
  all of it) that would erase a GPU-written verdict. Relocation lags one update
  on purpose — the rays that motivated it were cast from the old position.

**Budget and rota**, `render/gi/probe_schedule.hpp`: rays/probe {32, 64, 96} and
{1/8, 1/4, 1/4} of a cascade per turn; cascade 0 every frame, cascade 1 on odd
frames, cascade 2 on frames ≡ 2 (mod 4). The coarse cascades are **phased** so
at most two trace in any one frame, which puts the peak at 55 k / 221 k / 332 k
rays — under this section's own {85 k, …, 500 k} tier examples. Which probes:
scrolled-in uninitialized probes jump the queue, the rest come from a persistent
round-robin cursor; the whole selection is a pure function of main-thread state
and is tested for exact sets, determinism and the absence of duplicates within a
turn (a duplicate would be two threadgroups racing on one probe's texels).

**GI is a ray-traced feature, loudly.** `gi_mode != Off` now routes through the
same `ensure_raytracing_scene(want_attributes=true)` the traced effects use, so
turning it on builds the BLAS + attribute buffers even with `rt_shadows` and
`rt_reflections` off, and inherits the retained-CPU-geometry requirement. Said
in the code, in `docs/architecture.md`, and in `docs/user_guide.md`.

Measured on the playground village (`--smoke 30`, Medium): 110592 rays and 1728
probes on a cascade-0-only frame, 5.53 M rays over the run, 66 of a frame's 1728
scheduled probes relocated and 1 written off as interior. `--rt-bench 120` puts
`GIProbes` at **0.054 ms mean / 0.065 p95** CPU record time with `RTInstances`
unchanged at 0.126 vs 0.122 (`-O0` tree — ratios only). And the claim that
matters: **`--smoke 30` reproduces `mean_luma=86.5 stddev=44.9` bit for bit with
GI on and with GI off.** Consumption is R1c.

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

**Landed (R1c, 2026-08-02) — consumption: the field lights the frame.** The
first intentional image change of this plan, still behind `GiMode`, still
`Off` by default.

*The repack, done first because it sets the budget for everything after it.*
The three cascades' atlases became **one texture each**, cascades stacked as
row bands (`ATLAS_ROWS_ALL_CASCADES`, `probe_atlas_origin(slot, texels,
cascade)`). Same texels, same 30.6 MB — `probe_volume.test.cpp` derives the
total both ways to prove it — but the forward shaders now spend **three**
sampler slots on the field instead of nine, which is the difference between
fitting and not: they were at 12/10/8 samplers against a DXC ceiling measured
at 17. The fog march and R4's screen probes reuse the same three. It also
collapsed the trace pass's own bindings (6 sampled + 4 storage textures → 4 +
2) and made the probe-state upload one contiguous 20.25 KB plane. R1b's
behaviour is bit-identical across it: same `rays=110592 probes=1728 c0=1728
relocated=66 inside=1`, same `86.5 / 44.9`.

*`shaders/gi_sample.hlsli`*, the shared sampler, binding-declared by the
including shader the way `cluster_lighting.hlsli` is. DDGI bias (0.2×spacing
along the normal, 0.3 toward the camera — both scaled by the cascade's
spacing); cascade select = finest cascade containing the biased point with a
**half-spacing margin**; 8-probe trilinear × backface × **Chebyshev** (cubed
bound, variance floored at 1e-4×spacing², compared past mean + 0.02×spacing) ×
a skip on `probe_offset.w == INSIDE_GEOMETRY`, sampled at the probes'
**relocated** positions; renormalised over the live weights; a weight-sum floor
of 0.30 and a one-probe **edge fade** hand what is left to the next-coarser
cascade, and the hemisphere ambient absorbs the remainder. `giCascadeBase[]` is
sent, not re-derived from `cameraPos` — the two would agree until they were
handed different camera positions. Everything above is mirrored in
`render/gi/gi_math.hpp`'s `SAMPLE_*` block and `sample_field()`, and **C0
continuity is asserted numerically** rather than eyeballed: one test walks a
line through a cascade boundary with each cascade answering a different primary
colour, another walks through cell boundaries with hashed per-probe values, and
both bound the per-step delta.

*The one narrowing.* Three slots, not four, means the CPU-owned `probe_state`
texture does not reach the forward shaders, so `probe_offset.w` had to answer
"never filled" as well as "buried". It does, because `ProbeVolume::init()`
seeds the whole offset plane with `INSIDE_GEOMETRY` — a one-time upload before
any GPU write, not a second writer — which also closes the last path to reading
undefined texture memory now that something outside the update passes samples
the atlases. Cost: GI fades in over the first ~16 frames rather than popping.

*Forward integration.* `model` (which skinned reuses), `model_instanced`,
`terrain` and `grass` each gained the three slots **below** their existing ones
so the local-shadow atlas stays LAST per `cluster_lighting.hlsli`'s contract:
`CLUSTER_SSBO_BINDING_BASE` 12→15, 12→15, 10→13, 8→11. `ambientIBL` and
`ambientPBR` (`pbr.hlsli`) each took a trailing `giIrradiance`/`giWeight` pair
whose branch is not taken at weight 0, so the GI-off path is bit-identical, not
merely equal. The ledger of §4 holds exactly: the DIFFUSE ambient becomes
`gi × kd × albedo × (min(screenAO, ORM AO) × ambientLift)` and the SPECULAR
half is untouched.

*Fog.* `giSampleVolume` — nearest usable probe, ±Y taps averaged, three fetches
per march step, because the full sampler's 24 taps × 24 steps is three orders of
magnitude more than this pass has ever spent on ambient. It does **not** replace
`fogColor * 0.2`; it becomes a per-channel gain around 1 measured against the
same hemisphere ambient the sampler falls back to. Where the field has no answer
— which includes the horizon — the ratio is exactly 1 and the term is
bit-identical, so the sky↔fog agreement `hazeSunColor` encodes holds by
construction rather than by tuning.

*Measured*, playground village, Medium, 1600×900: `--smoke 30` unchanged at
`86.5 / 44.9` with `gi_mode` Off; `PLAYGROUND_GI=probes --smoke 30` moved to
**`84.7 / 46.6`**, reproduced identically over three runs. Lower mean and wider
histogram is the right direction — the diff is red (darker) on facades, crate
sides, eave undersides and mortar recesses, green (brighter) on open ground and
in canopy gaps, and exactly zero on the sky. `--rt-bench 120` puts `GIProbes`
at 0.056 ms / 0.072 p95 CPU record (0.054 / 0.065 before the repack) with frame
render 0.86 → 3.14 ms and `present_wait` 0.59 → 2.58 ms: the cost is GPU-side —
acceleration structure, probe rays, and an 8-probe gather per forward pixel —
not CPU record time. (`-O0` tree; ratios only.) 1358 tests pass.

*Still to come here.* The reflection hit shader and R2's DI pass sample the same
field (§5.4).

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

**Landed (R2a, 2026-08-02) — persistent identity + compute-side froxel
access, with zero image change.** The foundation half of R2, and nothing but:
every default is untouched, both smoke baselines are bit-identical
(`--smoke 30` 86.5/44.9, `PLAYGROUND_RT=both` 83.9/47.9), and no shader in the
frame path reads any of it.

- **`stable_id` on `PointLightDesc`/`SpotLightDesc`** (`contracts/light.hpp`),
  appended with a `0` default so every existing positional initializer keeps
  compiling and meaning what it meant (pinned by a `TestFrame` test). `0` =
  engine assigns; a host-supplied non-zero id passes through verbatim and
  wins. Auto-assignment happens at the top of `ClusterGrid::build()` —
  main-thread tier, serial, emission order — as an FNV-1a hash of
  `{type, round(pos / 5.0), round(color / 0.25), round(log2(intensity))}`.
  Intensity is quantized by RATIO because that is how a torch flickers; an
  absolute step would have re-identified every torch in the village every
  frame. In-frame collisions are broken by re-hashing with an occurrence
  counter, so two identical lanterns get distinct, deterministic ids.
- **The remap**, built in the same pass: `remap[prev_packed_index] =
  current_packed_index`, `0xFFFFFFFF` where the light vanished. Two
  `<= MAX_LIGHTS` passes, no hashing container, no pointer-order iteration.
  History is renderer-side per §8 — `reset_light_history()` exists and
  `SceneRenderer` calls it on frames it skips binning, so a lights → no
  lights → lights sequence cannot remap across the gap.
- **Where it rides to the GPU:** two parallel `uint[MAX_LIGHTS]` storage
  buffers (stable ids in packed order; the remap), uploaded with the cluster
  buffers in the same copy pass, ~1 KB each. `LightHeader` is exactly 8 bytes
  on both sides with the stride hardcoded in the shader, so it had no room —
  and packed-order arrays are what a reservoir wants anyway (one load, no
  header hop). The ids ALSO land in the light records, in bytes that were
  declared padding: no size, offset or shader load changed.
- **`shaders/cluster_compute.hlsli`** — the compute-side froxel accessor R2b
  dispatches against. The bindings differ (set 0 / set 2 vs. fragment's
  set 2 / set 3, per SDL3 GPU's layout) and the arithmetic cannot: the froxel
  index math moved to `cluster_index.hlsli` and the light-record decode to
  `cluster_light_load.hlsli`, both included by the fragment path and the
  compute path alike — the `shadow_common`/`gi_probe` move again.
  `render/lighting/cluster_index.hpp` is the CPU mirror, and
  `light_cluster.cpp`'s binner calls its `cluster_z_slice` directly rather
  than keeping a private lambda that could drift.
- **Tests** (+14): froxel-index parity across a full 3456-cluster sweep,
  checked in both directions (the shared math returns the intended cluster,
  AND the binner's independent AABB path binned a light there); Z-slice
  monotonicity and clamping; id stability across identical frames and across
  a fresh grid; reorder/add/remove with the remap naming exactly who moved and
  who vanished; sub-quantum jitter and intensity flicker keeping the id while
  a teleport does not; identical-light collision determinism; host ids winning
  regardless of emission order; the remap at the frame cap. The old
  hand-copied shader-index transliteration in `light_cluster.test.cpp` is
  gone — the tests now call the shipped function.
- **`cluster_compute_validate.comp`** exists only so the new include is a
  compiled artifact rather than a text file. Nothing dispatches it; it is
  registered in no pipeline; delete it when R2b's real pass includes the same
  header. *(Deleted in R2b, as planned.)*

**Landed (R2b, 2026-08-02) — the Ratio tier: one ray per pixel, and every
local light casts a soft shadow. R2 is complete.** `LocalRtShadowMode
{ Off = 0, Ratio = 1 }` + `rt_local_shadow_resolution` (Half), both `Off`/
default, seven touch points, dev-panel row, `[smoke] settings` field and a
`PLAYGROUND_RT_LOCAL=ratio|off` pin.

- **`shaders/rt_local_shadow.comp.hlsl`** is the pass: surface from depth +
  G-buffer, candidates from the pixel's froxel, RIS over M = 6 uniform draws
  weighted by `clusterLightTarget`, a temporal merge with the reprojected
  previous reservoir, and ONE sphere-cap any-hit ray at the winner. Output is
  the same 8×4-packed bit tile the sun pass writes, which is the whole point.
- **The target function moved to the shared level**, `cluster_light_load.hlsli`,
  next to the decode — the same anti-drift move as R2a's froxel index. It is
  built from the `pointFalloff`/`spotConeFactor` the FORWARD accumulator calls,
  so "the forward loop skips this light" and "the sampler weights it zero" are
  one statement rather than two that agree today. `render/lighting/
  light_target.hpp` is the CPU third, and `light_target.test.cpp` pins its
  support against the binner's own radius cull across a light row.
- **Reservoir: 16 B/px**, `{light index, W, M, linear depth}`, ping-ponged at
  the traced resolution. p̂ is deliberately NOT stored: the only consumer of a
  stored reservoir is next frame's merge, which must re-evaluate p̂ at the
  CURRENT surface — the stored value is the one number it may not use — so the
  freed lane carries the depth the merge cannot reconstruct instead. That
  re-evaluation is also why no normal test is needed: a light the new surface
  cannot see evaluates to zero and contributes nothing but M. History M is
  clamped to 20× the current candidate count. `render/lighting/reservoir.hpp` +
  `reservoir.test.cpp` pin the maths that a screenshot cannot: selection
  probability proportional to weight, streaming-merge == batch weights,
  zero-weight candidates still advancing M, and the remap dropping a vanished
  light rather than re-pointing it.
- **The denoiser was NOT cloned.** §5.3 above proposed "a scalar clone of the
  existing shadow denoiser"; there is no clone. The sun path's targets and flags
  became `RtRenderer::VisibilityDenoiser`, and the local signal is a second
  instance of it dispatched through the SAME three pipelines. Nothing in those
  passes was sun-specific. The proof that the refactor is behaviour-neutral is
  that `PLAYGROUND_RT=both --smoke 30` still reproduces `83.9 / 47.9` bit for
  bit.
- **Atlas retirement.** In Ratio mode `SceneRenderer` skips the selector and the
  tile renders entirely, so no `shadow_slot` is assigned and the atlas branch is
  dead through its existing `shadow_slot < 0` path rather than through a new
  one. `[smoke] ... local_tiles=0` under the pin. `Off` is byte-identical to
  before.
- **Two ceilings found the hard way**, both now documented where they bite.
  (1) SDL3 GPU refuses a compute pipeline declaring more than EIGHT read-only
  storage buffers; the four acceleration buffers plus four cluster buffers spend
  the budget exactly, so `cluster_compute.hlsli`'s packed stable-id buffer became
  **opt-in** (`CLUSTER_COMPUTE_WANT_STABLE_IDS`, which this pass declines — a
  temporal reservoir wants the remap, not the ids) and the reservoir ping-pong
  moved to two read-write slots of which one is only ever read. (2) The froxel
  lookup needs an ABSOLUTE view depth agreeing with the CPU binner's, and the
  `depthB / (d - depthA)` linearization every other depth-reading pass uses is
  wrong by ~33 % at typical distances — invisible everywhere it is only compared
  against itself, and enough here to land every pixel in a Z slice with an empty
  light list. The pass takes its view depth from `clusterParams.view` instead,
  the matrix `ClusterGrid` binned against, and carries no depthA/depthB at all.
- **Measured**, playground village, 1600×900, Linux/Vulkan, `--smoke 30`:
  `PLAYGROUND_RT=both` unchanged at `83.9 / 47.9`; adding
  `PLAYGROUND_RT_LOCAL=ratio` gives **`85.3 / 47.5`**, reproduced identically
  over three runs. Brighter, because the atlas it replaced was over-darkening
  the plaza with a hard-edged tile shadow. `--rt-bench 120` (`-O0` tree, ratios
  only): `RTLocalShadows` records **0.0087 ms mean / 0.0097 p95** CPU — and
  `LocalShadowPass`, the rasterized atlas it replaced, **disappears from the
  table**, having cost 0.0384 / 0.0423. Frame render 2.069 → 2.177 ms and
  `present_wait` 1.631 → 1.762 ms: the real cost is on the device, as it should
  be for 360 k more rays. 1398 tests pass.
- **Still open here.** No spatial reuse pass (§5.3's "High only" pairwise-MIS
  neighbours) — temporal alone is enough at this light count. The emitter
  radius is one global constant (10 world units); per-light authoring is R6's.
  And the Full-DI tier remains R5's, deliberately not declared as an
  enumerator so an on-disk `2` clamps down rather than opting into a tier that
  does not exist.

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

**Landed (R3, 2026-08-02) — reflections stop lying about their hits, and the
denoiser stops reprojecting the wrong point. R3 is complete.** No new pass, no
new setting, no new default: everything below is a change to what two existing
dispatches compute.

- **`shaders/rt_hit_shade.hlsli`** is the extraction, and it went out of
  `gi_probe_trace.comp.hlsl` verbatim — the recipe, the units argument and the
  reasoning for every term moved with the code. Two-phase include, the
  `rt_trace.hlsli` contract (`TriangleAttributes` in phase 1, so the including
  shader can declare its attribute buffer; the functions in phase 2, because
  they read `shadowAtlas`/`giIrradiance`/`lightDataBuf` by name). It includes
  `shadow_common.hlsli` ITSELF rather than requiring the entry shader to, so a
  consumer cannot forget it and silently get the unshadowed sun back. The proof
  the extraction is behaviour-neutral is that `PLAYGROUND_GI=probes --smoke 30`
  reproduces `84.7 / 46.6` bit for bit with the rough-specular change below
  disabled — the two were measured separately for exactly this reason.
- **Two flags, not knobs**, are what let one function serve both callers.
  `giEnabled` is 0 when `gi_mode` is Off, where `RtRenderer` binds its own 1×1
  stand-ins; the probe-state plane is a read-only STORAGE texture and the
  engine's shared dummies carry sampler usage only, which is why they are owned
  here rather than borrowed. `giAmbientFallback` is 1 for reflection rays and
  **0 for probe rays**: a probe already collects sky through its ray misses, so
  a hemisphere added where the field has no answer would double-count it AND
  break the bounce series' convergence argument (the series sums albedo^k only
  because each term is the previous field, not the previous field plus a
  constant). A reflected rooftop outside the field has no such source and must
  fall back.
- **The π that was hiding.** The old reflection shading was wrong by π twice, in
  opposite directions: no Lambertian 1/π, and `sunColor` pushed without
  `GraphicsSettings::sun_intensity`. At the playground's `sun_intensity` of 3
  those very nearly cancel (3/π ≈ 0.955), which is why the flat shading looked
  plausible and would not have at any other value. Both are fixed together;
  reflected surfaces are correspondingly darker and now match the same surface
  drawn directly.
- **Budget.** The reflection pass went to 6 samplers + 1 read-only storage
  texture + **6** read-only storage buffers. Six, not ten: the light list is
  read FLAT, out of the same byte layout `cluster_lighting.hlsli` decodes,
  rather than through `cluster_compute.hlsli`'s froxel grid — which would have
  cost four more read-only storage buffers against a ceiling of eight, and
  which cannot answer for a world-space hit anyway. R2b's ceiling held without
  a fight.
- **Virtual-hit-point reprojection** in `rt_reflection_denoise_temporal.comp.hlsl`
  finally consumes R0's `hitT`. The virtual point is built off the MIRROR
  direction, not the traced GGX sample: the lobe axis is the only part of it
  stable frame to frame, and reconstructing the stochastic sample would make the
  reprojection as noisy as the signal it cleans. Both motion vectors are UV
  deltas in the velocity buffer's own convention (current minus previous,
  unjittered), so the TAA jitter cancels inside each difference rather than
  having to be removed — which also means the pass never needs to know whether
  jitter is on. Blended, not switched, by
  `render/rt_reflection_math.hpp`'s `virtual_reprojection_weight` (1 at
  roughness 0.05, 0 by 0.40, both below `fade_start`), so no roughness makes a
  pixel's history jump texels. `max_accum_frames` was NOT raised off 16 in this
  phase: the smoke camera is static, which makes the mechanism exactly neutral
  and therefore proves nothing about how far the window can now be pushed.
  Raising it wants a moving-camera A/B that this phase did not build.
- **Rough-tier ambient specular goes local.** `pbr.hlsli`'s `ambientIBL` blends
  the sky-only prefiltered lobe toward the probe field over a 0.6→1.0 roughness
  band, scaled by the existing `giWeight`. This is the one change that moves the
  GI-on image without any RT setting: `PLAYGROUND_GI=probes --smoke 30` goes
  `84.7 / 46.6` → **`84.3 / 46.9`**. GI off is bit-identical, branch untaken.
- **One incidental fix.** `rt_sampling.hlsli` carried a byte-identical private
  copy of `interleavedGradientNoise`, which collided the moment the reflection
  pass included both it and `shadow_common.hlsli`. It now includes
  `lighting.hlsli` like everything else. One definition in the tree.
- **Measured**, playground village, 1600×900, Linux/Vulkan, `--smoke 30`, three
  identical runs each. `--smoke 30` unchanged at `86.5 / 44.9`;
  `PLAYGROUND_RT=both` unchanged at `83.9 / 47.9` — which is a COINCIDENCE of
  the aggregate, not a null result: 94% of the screenshot's bytes differ, to a
  maximum of 172/255, because reflections of sky-facing foliage brightened by
  exactly as much as reflections of shadowed facades darkened.
  `PLAYGROUND_RT=both PLAYGROUND_GI=probes` moves `82.1 / 49.6` → **`81.7 /
  49.8`**. The chrome orb over the plaza is the readable evidence: before, a
  featureless cream blob with brown smears (every reflected surface lit as if in
  full sun and washed toward one horizon colour); after, a proper mirror ball
  with a blue sky hemisphere, visible cloud and sun, a clean horizon line, and a
  dark lower hemisphere — warm and bounce-varied with GI on, neutral grey with
  it off. `--rt-bench 120 PLAYGROUND_RT=both` (`-O0` tree, ratios only):
  `RTReflectionTrace` CPU record 0.0043 → 0.0052 ms mean (four more sampler
  binds and a 240 → 672-byte uniform push), everything else flat, and
  `present_wait` 1.408 → 1.372 ms — i.e. **no measurable GPU cost**, which is
  what the pass being wave-divergence bound rather than ALU bound predicts.
  1406 tests pass (+8).
- **Still open here.** `max_accum_frames` (above). The rough-specular band's
  edges are eyeballed constants pinned by test, not derived. And the reflection
  hit still has no shadowed LOCAL lights — R2b's froxel-RIS mask is screen-space
  and cannot answer for a world-space hit, so the loop stays unshadowed, exactly
  as the probe rays' does.

### 5.5 R4 — Screen-probe gather (the quality tier)

**Landed (R4a, 2026-08-02) — the HZB and the `gpu/` layer's indirect dispatch,
and nothing that changes a pixel.** The foundation half: the pyramid is built,
nothing marches it; the wrapper exists, nothing dispatches through it. Both
smoke baselines are bit-identical (`--smoke 30` 86.5/44.9,
`PLAYGROUND_RT=both PLAYGROUND_GI=probes` 81.7/49.8), and
`PLAYGROUND_GI=probes_screen` reproduces the `probes` figure exactly while
adding an `HZB` timer row and an `hzb=1600x900 mips=11` field to the
`[smoke] gi` line.

- **Indirect dispatch** (`gpu/gpu_compute.hpp`, `gpu/gpu_buffer.hpp`).
  `ComputePass::dispatch_indirect(const GPUBuffer&, offset)` wraps
  `SDL_DispatchGPUComputeIndirect`; `GPUBuffer::Type::Indirect` is
  `INDIRECT | COMPUTE_STORAGE_WRITE` — the write bit is not decoration, since a
  workgroup count the CPU could compute would not need indirection at all. The
  doc comment records what SDL actually requires, read off the header rather
  than assumed: the 12 bytes at `offset` are one
  `SDL_GPUIndirectDispatchCommand` (three `Uint32` counts, x/y/z; no count or
  stride field — exactly one dispatch), a pipeline must already be bound, 4-byte
  offset alignment, and the producing pass must have ENDED because SDL provides
  no intra-pass ordering. It also records, in the same place, that **no test
  drives the SDL call** — the type→usage mapping and the command layout are
  pinned CPU-side and the rest waits for R4b, which is the honest statement.
- **HZB** (`render/hzb.hpp/.cpp`, `shaders/hzb_copy.comp.hlsl` +
  `hzb_reduce.comp.hlsl`). One R32_FLOAT mip chain at depth resolution
  (1600×900 ⇒ 11 levels, ~7.7 MB), built right after the depth prepass.
  `HzbCopy` writes mip 0 from the D32 depth through a sampler (a depth format
  cannot be a compute storage texture); `HzbReduce` runs once per further mip,
  each in its own compute pass — required, because SDL3 has no intra-pass
  barrier. A single-pass groupshared ladder (SPD) is the known optimization and
  is deliberately not here: it would land before anything measured the naive
  chain. Timed as `HZB`.

  *The reduction is a MAX.* This section says "min-depth pyramid" above; that is
  the standard-Z spelling. The engine's depth is reversed-Z, so a larger value
  is a CLOSER surface, and the conservative rule — a march may skip a footprint
  only when the ray is in front of everything in it — means a coarse texel must
  keep the footprint's CLOSEST surface, i.e. the maximum. A min would keep the
  farthest and screen rays would pass through near geometry.

  *The footprint is not always 2×2.* Mip N is `max(1, size >> N)`, so an odd
  source loses its last row/column to integer division and a plain 2×2 gather
  never reads it; clamping re-reads a covered texel rather than fixing it. The
  last destination texel therefore extends to the end of the source (at most one
  extra texel, since a source extent is `2·dst` or `2·dst + 1`), keeping the
  footprints a partition and making "mip N texel == max over its whole mip-0
  region" true by induction. `render/hzb.test.cpp` brute-forces both claims
  against even, odd, 3×3 and 1600×900 grids, plus mip-count formula, partition
  and determinism.
- **Gating.** The pyramid builds only when `gi_mode == GiMode::ProbesScreen` —
  the mode R4b will consume it from, and which still behaves exactly as `Probes`
  otherwise. That keeps every existing configuration byte-for-byte untouched
  *and* gives the pyramid a live path to run in, which an unreachable pass would
  not have.

**Landed (R4b, 2026-08-02) — the screen-probe gather. R4 is complete.**
`render/gi/screen_probes.hpp` + six compute passes, `GiMode::ProbesScreen` only,
and the mode finally means something: `--smoke 30 PLAYGROUND_RT=both
PLAYGROUND_GI=probes_screen` moved from `81.7 / 49.8` (identical to `probes`) to
**`82.3 / 50.4`**, reproduced over three runs, while `--smoke 30` and
`PLAYGROUND_GI=probes` stay bit-identical at `86.5 / 44.9` and `81.7 / 49.8`.

**Shape.** One 8x8 octahedral probe per **16x16** screen tile (1600x900 =>
100x57 = 5700 probes), not §5.5's 8x8 tile: only the hemisphere above the anchor
is traced, so a tile is 32 rays, and at an 8x8 pitch that is one ray per two
pixels — a per-pixel budget wearing a probe's clothes, which §2 says does not
fit this engine's hardware floor. 16x16 puts it at one ray per eight pixels.
The price is paid in the interpolation, which is why there is a weight floor and
a field fallback at all. Radiance atlas 800x456 RGBA16F x3 (raw + a ping-ponged
filtered pair), probe headers ping-ponged, ~11.8 MB with the miss buffer.

**Six passes, one `GIScreenProbes` scope**, each its own compute pass because
SDL3 has no intra-pass synchronization: *place* (fixed nine-candidate anchor
search on the depth buffer; also zeroes the frame's counters), *march* (the Hi-Z
ladder, the TAA-history radiance tap, and a groupshared compaction of every
unresolved ray into a miss buffer with one atomic per threadgroup), *args* (the
miss count -> `SDL_GPUIndirectDispatchCommand` — **the first runtime use of the
`ComputePass::dispatch_indirect` R4a landed unexercised**), *trace* (indirect
dispatch over the compacted misses; short 600-unit BVH rays through
`rt_hit_shade.hlsli`, sky/field blend beyond the clamp), *filter* (probe-space
temporal: firefly clamp against the probe's own mean, velocity-reprojected EMA),
*project* (3x3 probe-space spatial fused with the L1 SH projection).

**Two things the design assumed that the platform refused.**

1. **The seventeenth sampler does not exist.** §5.5's consumption plan was one
   new fragment sampler per scene shader (model/instanced 16 -> 17). DXC
   legalizes 17 — that measurement is real and is what retired the old
   11-sampler belief — but `SDL_CreateGPUShader` asserts *"Shader sampler count
   cannot be higher than 16!"* and returns null, so it is a missing pipeline and
   a black scene, not a slow frame. The SH is a **fifth fragment storage
   buffer** instead (`gpu::GiScreenProbeSh`, 64 B/probe), which is also the
   better fit: probe-indexed records read by index and never filtered.
2. **The pyramid cannot be addressed by `uv * mipSize`.** R4a's own odd-edge
   rule — the last destination texel of each axis extends to the end of the
   source — makes mip N's footprints non-uniform, so a uv-addressed sample asks
   the wrong texel near the right and bottom edges and the march walks through
   whatever it was reporting. The correct grid is `screenSize / 2^level` with a
   `Load`. The CPU mirror's brute-force march test found this before a pixel was
   rendered; fixing it took the playground's screen-hit count from 20900 to
   32204 on the same frame.

**The economics claim, measured.** On the smoke frame: 32204 rays resolved on
screen, 148734 handed to the BVH, of which 36589 hit and 112145 found nothing
inside the clamp. So **47 % of the rays that found any geometry resolved on
screen**, and 62 % of all traced rays hit nothing — an outdoor village under
open sky. "Most rays resolve on screen" holds for interiors and dense geometry;
it does not hold here, and this is the honest number.

**Cost.** `--rt-bench 120`, RelWithDebInfo, 1600x900, `PLAYGROUND_RT=both`:
frame render 4.478 -> 5.032 ms and `present_wait` 3.893 -> 4.427 ms against
`PLAYGROUND_GI=probes`, i.e. **~0.55 ms** for the tier — inside this section's
2.5-4 ms budget. `HZB` records 0.017 ms and `GIScreenProbes` 0.016 ms of CPU
time; the work is on the device.

**Visually**, against the `probes` shot: warmer bounce on the open cobbles,
cooler and darker on facades and eave undersides where local geometry occludes
what 1.5 m probes blur through, and the animated villager changed most of all —
he is in the depth buffer and only bind-pose in the BVH. No tile structure: the
mean 16x16 intra-tile pattern of the difference image spans 0.086 against a
difference standard deviation of 2.33, i.e. under 4 %.

**Deliberately not in v1**, both documented rather than deferred silently: no
adaptive extra probes (the depth/normal gather and the field fallback cover
discontinuities without a measured seam to motivate them), and no mid-roughness
specular hand-off (L1 SH is a diffuse basis; that proposal needs the octahedral
atlas exposed to the forward pass). Alpha-MASK foliage is not in the depth
prepass, so canopy-tile anchors describe what is behind the leaves — the same
limitation the screen AO term has.

*Still to come in this section:* nothing. R4 is complete; the mid-roughness
specular hand-off moves to R5's scope alongside the near-field work.

*Follow-up (2026-08-02): a validated run is clean again.* Two defects the R1b/R1c
and R4a work left behind are fixed with no pixel moving: the integer-format
textures (GI probe state, RT visibility bits + tile classes) no longer request
`TEXTUREUSAGE_SAMPLER`, so `BRACKEN_GPU_DEBUG=1` no longer trips SDL's assertion
and poisons the device; and `volumetric_fog.frag` no longer declares the
visibility atlas it never reads, so its reflected layout matches the catalogue
row. `SDL_ASSERT=always_ignore BRACKEN_GPU_DEBUG=1 PLAYGROUND_RT=both
PLAYGROUND_GI=probes_screen ./playground --smoke 30` now runs with zero SDL
assertions and zero `declared resource counts` warnings, at the same
81.7/49.8 the un-validated run produces.

GI-1.0's shape over our tracer, `GiMode::ProbesScreen`, mid/high GPUs only:

- ~~**HZB.** New min-depth pyramid pass after the prepass (the engine's first —
  also unlocks future occlusion culling). Enters the `gpu/` layer with the
  indirect-dispatch wrapper (`ComputePass::dispatch_indirect`,
  `GPUBuffer::Type::Indirect`).~~ **Landed (R4a)** — see above; the pyramid is a
  reversed-Z *closest-surface* (MAX) pyramid, which is this bullet's "min-depth"
  under the engine's depth convention.
- ~~**Probes.** One 8×8 octahedral probe per 8×8 screen tile (adaptive extras
  where depth/normal interpolation fails — the indirect dispatch consumer),
  placed on G-buffer depth, jittered.~~ **Landed (R4b)** at a 16×16 tile and
  without the adaptive extras — see the note above for both arguments. The
  indirect dispatch found a better consumer: the compacted MISS list of the
  screen march, whose length only the GPU knows.
- ~~**Trace ladder per ray:**~~ **Landed (R4b)**, exactly as written: HZB march,
  then a 600-unit BVH ray, then a sky/field blend.
- ~~**Filtering in probe space**~~ **Landed (R4b)**, with the spatial half fused
  into the projection pass rather than standing alone — it needs its
  neighbours' temporal results complete, which only a later pass can guarantee,
  and that pass was already visiting every texel. No per-pixel GI denoiser
  exists in this plan at any tier.
- Screen probes also serve **mid-roughness specular** (importance-sample the
  probe's radiance by the GGX lobe) — this is when `ambientIBL` retires to
  sky-visible-surfaces duty and the reflection trace can hand off *earlier*
  (gate `fade_start` drops), cutting reflection rays on rough scenes. **NOT in
  R4b**: L1 SH is a diffuse basis, and this needs the octahedral radiance atlas
  itself exposed to the forward pass. Carried into R5.

*Costs:* GI-1.0 measured 1.9–3.4 ms @1080p on 6900 XT/3080-class for the full
stack over *its* tracer; budget 2.5–4 ms on our mid tier, gated by
`GiQuality`. *Tests:* probe placement determinism; HZB correctness (mip N min
== reference); handoff continuity scene (wall half on/half off screen — no
seam in indirect); tier A/B screenshots.

### 5.6 R5 — Near field, denoiser core, full-DI

- ~~**SSILVB upgrade of GTAO**~~ (**R5a, landed**): same horizon loop, sectors
  became a 32-bit visibility bitmask (thin-occluder correctness), outputting
  AO + bent cone. Bent cone gates the probe/screen-probe irradiance
  (directional occlusion instead of scalar AO), and the double-count rule is
  honoured: the cone REPLACES the scalar-AO multiply for the GI term, never
  joins it.

  **What landed.** `gtao.frag.hlsl` keeps its 3 slices × 4 steps × 2 sides and
  its jitter; per slice it now marks a 32-sector `uint` mask, each depth sample
  occluding the sectors subtended by a slab `GTAOUniforms::occluderThickness`
  = **1.0 world units** deep along the view ray (scaled by `ao_radius`). That
  constant replaces the unitless `thinOccluderCompensation = 0.2` horizon
  decay; it was chosen by A/B at 0.5 / 1.0 / 2.0 / 4.0 against the village
  smoke shot — below 1.0 fence posts and crates lose their ground contact,
  above it the crop foliage starts greying again, and 1.0 is also ¼ of the
  sampling radius and roughly a plank's thickness at this project's scale
  (~6 cm per unit). The surviving sectors are integrated with the SAME cosine
  weight as the arc integral they replace, so an unoccluded pixel is unchanged
  by construction (`GtaoSectorWeights.SumMatchesArcIntegral`). No wave
  intrinsics: `countbits`-class ALU only.

  The whole chain went R8 → RGBA8 — R = AO, GB = octahedral WORLD bent normal,
  A = cone aperture as sin²(half-angle) — with `shaders/bent_cone.hlsli` as the
  single owner of the layout, the `A >= 0.998` "fully open / no directional
  data" escape value, and the gate. `GTAOTemporal` became MRT (RGBA16F history
  + RG16F depth history) because five linearly filtered quantities do not fit
  in four channels; every filter round-trips the bent normal through decode /
  average-as-direction / renormalise / re-encode. The escape value is what
  makes the 1×1 white dummy decode to exactly the pre-R5a behaviour, so **no
  new uniform flag was needed in any scene cbuffer**.

  `giSampleDiffuse` / `giScreenSampleDiffuse` gained a `sampleDir` argument
  distinct from `N`: probe SELECTION stays on the surface normal (bias,
  backface wrap, screen-probe forward-normal test — geometry), the octahedral
  and SH TAPS take the bent normal (a directional query). `pbr.hlsli` now
  occludes each ambient source separately instead of multiplying `ao` over the
  sum, which is what makes "replaces" mean replaces.

  **Deliberately deferred to a later package**: SSILVB's near-field screen-
  space indirect DIFFUSE half (gathering last frame's lit colour into the
  unoccluded sectors). It needs the previous frame's lit colour bound into the
  AO pass and its own energy rules against the probe field — "SS indirect
  *replaces* probe irradiance within its sampled radius, never adds" — which is
  a package's worth of ledger on its own, not a rider on this one.

  **Measured** (Linux/Vulkan, RelWithDebInfo, `--smoke 30`, 3-run identical):
  default `86.5 / 44.9` → `86.3 / 45.5`; `PLAYGROUND_GI=probes` `81.7 / 49.8` →
  `81.8 / 50.0`; `probes_screen` `82.3 / 50.4` → `82.2 / 50.5`. Mean unmoved,
  spread up: thin geometry (fences, crop rows, grass tufts, distant canopies)
  lost the grey wash, solid crevices and eaves gained contact depth. The GTAO
  chain's CPU RECORD time went 0.0118 → ~0.0132 ms mean (`--rt-bench 120`,
  `probes_screen`); the GPU-side ALU cost of the sector loop is not measurable
  here — SDL3 exposes no timestamp queries — and the observable proxies (frame
  render 4.95 ms, present_wait ~4.3 ms, fps 179 → 182) did not move.
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
| **R4** | ~~HZB + indirect dispatch wrapper~~ (**R4a, landed**), screen probes, probe-space filtering, mid-rough specular from probes (R4b) | R4a: every configuration bit-identical, `probes_screen` == `probes` luma with the `HZB` row present. R4b: `ProbesScreen` vs `Probes` A/B sharpness; handoff-seam scene clean; Silver budget held |
| **R5** | ~~SSILVB/bent-cone~~ (**R5a, landed**), denoiser core (gradients, dual history, exposure-aware), Full-DI tier | lightning/toggle scene: history resets locally within 2 frames, no global flash-smear; contact-GI A/B |
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
