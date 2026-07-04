# CPU Performance Audit — Per-Frame Path

**Date:** 2026-07-03
**Scope:** CPU-side cost of the per-frame path: `Application::run` →
`SceneRenderer::render_frame` → render passes, plus `AnimationPlayer`,
`AnimationStateMachine`, `EffectSystem`, `ClusterGrid`, and the UI/text stack.
**Method:** static read of the cited code (no builds run, no profiling
performed). Every cost estimate below is an analytic cost-class estimate
(ns / µs / ms per frame) under stated scene assumptions — verify with Tracy
(`MMO_ENABLE_TRACY=ON`) before acting on the ms-class items.

---

## Executive summary

The frame path is in strong shape for its current workload. The expensive
lessons documented in `docs/engine_design_notes.md` are visibly applied:
per-frame GPU uploads are consolidated into a single `FrameUploads` copy pass
(`scene_renderer.cpp:602-620`), cluster AABBs are memoized against projection
changes (`light_cluster.cpp:166-174`), the camera inverse-projection is cached
(`scene_renderer.cpp:541-544`), instance bins clear in O(touched)
(`scene_renderer.cpp:1166-1171`), grass chunk uploads are hash-skipped when the
visible set is unchanged (`grass_renderer.cpp:197-204`), redundant uniform
pushes are memcmp-guarded (`scene_renderer.cpp:1140-1145`), terrain draws are
tile-culled with contiguous-run merging (`terrain_renderer.cpp:535-554`), and
the normal matrix has a uniform-scale fast path (`transform_math.hpp:27-32`).
Deliberate decisions (serial `JobSystem`, unwired `render_graph/`) were
verified as intentional and are **not** flagged.

The two findings that can reach **ms-class** as content scales are (1) the
particle renderer, which issues one draw call plus a 288-byte uniform push per
particle instead of instancing, and (2) the cluster light binner, which tests
all 3,456 clusters against every light (serial). Everything else is µs-class
polish: a triple copy on the instance-upload path, redundant identity fills in
`AnimationPlayer`, an O(N²) palette-dedupe scan, per-frame texture creation
for dynamic UI text, and string-keyed animation parameters. None of these are
defects at today's scene sizes; they are the items that will surface first in
a Tracy capture as entity/particle/light counts grow.

---

## Deliberate designs verified (not flagged)

- **`JobSystem` serial by default** — `init()` is never called from engine
  code (only `core.bench.cpp:36`), so `parallel_for` takes the serial path
  (`parallel_for.hpp:30-33`). Matches "default to serial" in the design notes.
- **`render_graph/` unwired** — `SceneRenderer` issues passes directly, per
  design notes.
- **Cluster SSBOs keep stale contents with zero lights** — guarded by zeroed
  `frame_cluster_params_` (`scene_renderer.cpp:592-596`); the empty-scene
  binning walk is fully skipped (`light_cluster.cpp:248-254`). Correct and cheap.
- **Post-target `resize()` calls every frame** (`scene_renderer.cpp:650-652`)
  all early-out on unchanged size (`ambient_occlusion.cpp:122`,
  `bloom.cpp:48`, `volumetric_fog.cpp:48`).
- **`FrameAllocator` not used for capacity-stable vectors** — design notes say
  this was measured as a wash; the per-frame vectors below all preserve
  capacity across frames.

---

## Findings — HIGH

### H1. Particle rendering: one draw call + 288 B uniform re-push per particle

**Evidence:** `src/engine/render/effect_renderer.cpp:214-225` — inner loop per
particle:

```cpp
for (const auto& particle : emitter.particles) {
    vertex_uniforms.model = build_particle_transform(particle, *model);
    SDL_PushGPUVertexUniformData(cmd, 0, &vertex_uniforms, sizeof(vertex_uniforms)); // 288 B
    ...
    SDL_PushGPUFragmentUniformData(cmd, 0, &frag_uniforms, sizeof(frag_uniforms));
    SDL_DrawGPUIndexedPrimitives(pass, mesh.index_count(), 1, 0, 0, 0);
}
```

`ModelTransformUniforms` (`gpu/gpu_uniforms.hpp:15-25`) is 288 bytes; `view`,
`projection`, `normalMatrix`, `cameraPos` are frame-invariant, yet all 288
bytes are re-pushed per particle because `model` lives in the same cbuffer.
Each particle also pays `build_particle_transform`
(`effect_renderer.cpp:28-47`): `glm::translate` + three `glm::rotate` calls
(each a full 4×4 multiply) + `glm::scale` + a final 4×4 multiply for
centering — and the model-center recentering translation is recomputed per
particle although it is constant per model.

**Why it matters:** per-draw CPU cost through SDL_GPU/Vulkan is roughly 1–2 µs
(command recording + uniform copy). At 200 particles this is ~0.3–0.5 ms; at
1,000+ (continuous emitters at `spawn_rate`, several effects) it is multi-ms —
the single largest CPU scaling hazard in the frame. The engine already has the
instancing pattern this needs (static models use a per-frame instance SSBO +
one draw per mesh, `scene_renderer.cpp:1496-1560`).

**Recommendation:** per (emitter, mesh) group, pack per-particle
`{model, color, opacity}` into a per-frame storage buffer uploaded in the
existing `FrameUploads` copy pass, and issue one
`SDL_DrawGPUIndexedPrimitives(..., instance_count, ...)` per mesh. Split
frame-invariant camera data into its own cbuffer pushed once per pass. Compose
the rotation matrix directly from the three Euler angles (9 sin/cos products)
instead of chained `glm::rotate`, and hoist the center-offset per model.
**Cost class: ms at high particle counts; µs at tens of particles.**

### H2. Cluster light binning is O(clusters × lights), serial

**Evidence:** `src/engine/render/lighting/light_cluster.cpp:276-295` — for
every one of `CLUSTER_COUNT = 16×9×24 = 3,456` clusters, every light is
sphere/AABB-tested:

```cpp
core::jobs::parallel_for(0u, CLUSTER_COUNT, [&](size_t begin, size_t end) {
    for (std::size_t c = begin; c < end; ++c) {
        ...
        for (std::size_t i = 0; i < light_count; ++i) {
            ...
            if (sphere_intersects_aabb(vl.view_pos, vl.radius, cb.view_min, cb.view_max)) {
```

Since the `JobSystem` is (correctly) never initialized, this runs serial
(`parallel_for.hpp:30-33`). At 50 lights that is ~173k sphere/AABB tests per
frame; at `MAX_LIGHTS = 256` it is ~885k tests — roughly 1–3 ms serial,
every frame there is at least one dynamic light.

**Why it matters:** the design notes already earmark GPU compute for ">>100
lights", but the CPU-side algorithm can be improved by ~two orders of
magnitude without any new infrastructure, keeping the CPU path viable far
longer.

**Recommendation:** invert the loops. For each light, compute its view-space
AABB, project it to a cluster index range (x/y from the linear NDC tiling, z
from the inverse of `exp_z_slice`, `light_cluster.cpp:67-69`), and push the
light index only into the clusters in that range (refine with the existing
sphere/AABB test inside the range). Complexity becomes O(lights ×
touched-clusters) — a typical point light touches 10–100 clusters, not 3,456.
This also removes the need for the 3,456-entry `bins_` vector-of-vectors walk
in the empty regions. **Cost class: ms → tens of µs at 100+ lights.**

---

## Findings — MEDIUM

### M1. Dynamic UI text rasterizes a whole texture per unique string

**Evidence:** `src/engine/render/text_renderer.cpp:192` —
`TTF_RenderText_Blended(font_, text.c_str(), 0, sdl_color)` inside
`get_or_create_text_texture`, which also creates a GPU texture
(`text_renderer.cpp:213`) and a fresh transfer buffer
(`text_renderer.cpp:225`) per new string. `queue_text_draw`
(`text_renderer.cpp:399-403`) misses the cache for any string not seen
before and defers to next frame (`pending_text_creates_.insert(text)`).

**Why it matters:** any text that changes every frame — FPS counters, timers,
damage numbers, coordinates — pays a full TTF rasterize + surface convert +
texture create + transfer-buffer create + copy-pass upload *every frame*
(~100–500 µs per changed string), renders one frame late, and leaves a trail
of dead textures that linger for `CACHE_EXPIRY_FRAMES = 300`
(`text_renderer.hpp:159`) while the eviction sweep walks the whole cache each
frame (`text_renderer.cpp:159-168`).

**Recommendation:** bake a glyph atlas once (one texture, one sampler) and
emit per-glyph quads into the existing `batch_vertices_` batching path; string
cost becomes pure vertex generation (ns/char) with zero per-frame GPU resource
creation and no one-frame latency. **Cost class: 100 µs–ms with dynamic text;
zero after the fix.**

### M2. Instance upload makes three full passes over instance data

**Evidence:** instance data is written into per-model bins
(`scene_renderer.cpp:1284-1289`), copied bin-by-bin into `packed_instances_`
(`scene_renderer.cpp:1328-1334`), then memcpy'd again into the mapped transfer
buffer (`gpu_buffer.cpp:232-238`). The same triple-copy applies to shadow
instances (`scene_renderer.cpp:1362-1371`) and, to a lesser degree, bone
palettes (packed into `bone_palette_scratch_` then memcpy'd,
`scene_renderer.cpp:1428-1430`).

**Why it matters:** `InstanceData` is 160 B (`gpu_uniforms.hpp:161-167`). At
10k visible instances that is 1.6 MB × 3 passes ≈ 4.8 MB of memory traffic per
frame just for the main view (~0.3–0.5 ms of pure copy on top of the
build-time write), plus up to 4 cascades of 64 B `ShadowInstanceData`.

**Recommendation:** add a `map()/unmap()` (or `update_from(fn)`) API to
`GPUBuffer` so `upload_instance_buffers` packs bins **directly into the mapped
transfer buffer**, deleting `packed_instances_`/`packed_shadow_instances_` and
one full copy each. (The map-with-cycle + single `SDL_UploadToGPUBuffer` at
the end preserves current semantics.) **Cost class: ~0.1–0.5 ms at 5–10k
instances; negligible below ~1k.**

### M3. `AnimationPlayer` rewrites identity tails of two 8 KB arrays every update

**Evidence:** `src/engine/animation/animation_player.cpp:210-213`:

```cpp
for (size_t i = count; i < MAX_BONES; i++) {
    bone_matrices_[i] = glm::mat4(1.0f);
    world_transforms_[i] = glm::mat4(1.0f);
}
```

runs on **every** `compute_bone_matrices` call. Nothing ever writes those tail
slots elsewhere (`reset()` initializes them, `animation_player.cpp:18-19`; the
per-frame loops stop at `count`), and consumers read only `joint_count`
matrices (`scene_renderer.cpp:1429-1430` packs `joint_count`, not
`MAX_BONES`).

**Why it matters:** a typical 60-joint humanoid wastes (128−60)×2×64 B ≈
8.7 KB of writes per instance per frame. At 100 animated characters that is
~0.87 MB/frame of dead stores (~0.1–0.2 ms of memory bandwidth) plus cache
pollution right before the palettes are read back for packing.

**Recommendation:** fill the tail once in `reset()` (already done) and when
the bound skeleton's joint count changes; delete the per-update loop.
**Cost class: ~µs per character, ~0.1–0.2 ms at 100 characters.**

### M4. Bone-palette dedupe is an O(N²) linear scan

**Evidence:** `src/engine/scene/scene_renderer.cpp:1410-1416` — for each
skinned command, `palette_dedupe_` is scanned linearly for a matching
bone-array pointer. The in-code comment documents the linear scan as a
deliberate choice ("skinned counts are small and this avoids per-frame
map-node allocations"), which is right — but the result is N²/2 pointer
compares when palettes are mostly unique (the common case: every character has
its own `AnimationPlayer`).

**Why it matters:** at 300 skinned draws → ~45k compares (~20–40 µs); at the
`MAX_BONE_PALETTES = 1024` cap → ~0.5M compares (~0.2–0.5 ms), all in the
pre-pass CPU section.

**Recommendation:** keep the allocation-free property with a small
open-addressing hash table over the pointer key (power-of-two array of
`{key, offset, joint_count}` slots, cleared by generation counter, sized 2×
cap) — O(N) expected, zero per-frame allocations. **Cost class: tens of µs →
sub-ms near the cap.**

### M5. String-keyed animation parameters hash per entity per frame

**Evidence:** `src/engine/animation/animation_state_machine.cpp:113-121` —
`set_float(const std::string&, float)` / `set_bool` call `ensure_param`, an
`unordered_map<std::string, int>` lookup (`animation_state_machine.cpp:53-62`).
The documented host pattern is `set_float("speed", …)` per entity per frame
(CLAUDE.md, architecture.md §Animation). Each call constructs a `std::string`
from a literal (SSO, no alloc, but hash + compare every time).

**Why it matters:** 300 entities × 3 params × 60 Hz ≈ 54k hash lookups/s in
`on_update`; ~20–50 µs/frame at that scale. The machinery for index-based
access already exists (`TransitionCondition::param_index`,
`animation_state_machine.cpp:11-15`).

**Recommendation:** make `ensure_param`/`find_param` public and add
`set_float(int idx, float)` / `set_bool(int idx, bool)` overloads; hosts
resolve names once at setup. (Alternative: key by the existing
`core::hash::StringId`.) **Cost class: tens of µs at hundreds of entities.**

### M6. UI text takes three string copies per label per frame

**Evidence:** per label per frame: host string → `TextCommand`
(`ui_scene.cpp:46`, copy 1), `UIRenderer::draw_text` fills a local
`QueuedTextDraw` (`ui_renderer.cpp:541`, copy 2) and `push_back(td)` copies it
again (`ui_renderer.cpp:546`, copy 3 — `td` is not moved), then
`queue_text_draw` hashes the string against `text_cache_`
(`text_renderer.cpp:399`; a second hash into `pending_text_creates_` on cache
miss only) after `UIScene::clear()` freed all of last frame's strings
(`ui_scene.cpp:9-11`). `add_text_wrapped` adds a `substr` temporary per line
on top (`ui_scene.cpp:71`, `82`).

**Why it matters:** every label longer than the SSO limit (~15 chars) is a
heap alloc+free × 3 per frame. A 50-label HUD ≈ 150–300 allocations/frame
(~5–15 µs plus allocator contention/fragmentation noise).

**Recommendation:** cheapest first: `push_back(std::move(td))` and
`commands_.emplace_back(...)` with moved strings (removes 2 of 3 copies);
then switch the queue path to `std::string_view` into a per-frame char arena
owned by `UIScene` (cleared, capacity-stable). **Cost class: µs-class; grows
linearly with label count.**

---

## Findings — LOW

### L1. Per-particle `sqrt` zero-tests and loop-invariant checks

**Evidence:** `src/engine/systems/effect_system.cpp:217` —
`if (glm::length(vel_def.gravity) > 0)` executes a `sqrt` per particle per
frame to test a per-*emitter* constant; `effect_system.cpp:228` —
`glm::length(particle.velocity) > 0.01f` is another per-particle `sqrt`
followed by `glm::normalize` (a third) when facing velocity
(`effect_system.cpp:230`).
**Fix:** hoist gravity/drag flags out of the particle loop (they are
`EmitterDefinition` constants) and compare squared magnitudes
(`glm::dot(v,v) > 1e-4f`). **Cost class: ~ns/particle; 20–50 µs at 5k
particles.**

### L2. Cluster-offsets hash is byte-wise FNV over 27.6 KB

**Evidence:** `light_cluster.cpp:71-79` (`fnv1a64` byte loop) called on
`cluster_offsets_` (3,456 × 2 × 4 B) every frame the upload runs
(`light_cluster.cpp:375`). **Fix:** hash 8 bytes at a time (or set a dirty
flag while packing offsets in `build()` — the pack loop already touches every
entry, `light_cluster.cpp:304-312`). **Cost class: ~10–30 µs per lit frame.**

### L3. `ClusterGrid::begin_frame` recomputes `glm::inverse(projection)` every frame

**Evidence:** `light_cluster.cpp:155` — `params_.invProjection =
glm::inverse(camera.projection)` runs unconditionally, although the renderer
already caches exactly this inverse on projection change
(`scene_renderer.cpp:541-544`), and the grid itself memoizes *bounds* on
change (`light_cluster.cpp:166-174`) — the inverse is the only uncached step.
**Fix:** pass `cached_inv_projection_` into `begin_frame`, or cache the
inverse behind the same `projection_changed` check. **Cost class: sub-µs
(~80-flop 4×4 inverse); consistency cleanup.**

### L4. Terrain tile culling is a linear scan per pass

**Evidence:** `terrain_renderer.cpp:538-541` — every visible-pass render walks
all `tiles_` (~2,700 for a 42k-unit world at 800-unit tiles, per the in-code
comment at `terrain_renderer.cpp:313-315`) doing frustum-AABB tests; the
shadow path repeats it per cascade, but over the 4×-coarser `shadow_tiles_`
(~190 tiles at `kShadowCellSize = 100`, `terrain_renderer.cpp:656`;
loop at `terrain_renderer.cpp:621-637`). Main + 4 cascades ≈ 3–4k AABB
tests/frame. **Fix (only if Tracy shows it):** derive a conservative
tile-index rectangle from the frustum's XZ AABB before testing, or a 2-level
grid. **Cost class: ~30–100 µs/frame total.**

### L5. `GPUTimestampPool` builds `std::string` pass names every frame

**Evidence:** `gpu_debug.cpp:41` — `pass_names_.emplace_back(pass_name)` per
`ENGINE_GPU_TIMER` per frame (~12–16 passes); names like
`"ShadowPass:Cascade0"` (19 chars) exceed SSO and heap-allocate.
`begin_frame()` destroys them all (`gpu_debug.cpp:61-63`). Since the pool is a
stub that reports zeros anyway (SDL3 limitation, per architecture.md), this is
pure overhead. **Fix:** store `const char*` (all call sites pass string
literals). Same pattern applies to `read_results` re-allocating
`pass_times_ms` strings when stats are on (`gpu_debug.cpp:53-59`,
`scene_renderer.cpp:473` re-constructs `RenderStats` including the vector).
**Cost class: ~1–2 µs/frame.**

### L6. Skinned main-view loop re-resolves `Model*` already resolved during palette packing

**Evidence:** `scene_renderer.cpp:979` — `model_manager_->get_model(...)` per
skinned command for frustum culling, after `pack_bone_palettes` resolved the
same handle at `scene_renderer.cpp:1395` (and the shadow pass builds its own
resolved scratch at `scene_renderer.cpp:1920-1938`). The lookup is a cheap
vector index, so this is a minor coherence cleanup: extend
`skinned_palette_offsets_` into a small `{offset, Model*, world_center,
radius}` scratch shared by main and shadow paths, removing the duplicate
transform×bounds math (`scene_renderer.cpp:981-983` vs `1936-1937`).
**Cost class: ~ns/entity; single-digit µs at hundreds of skinned draws.**

### L7. `Particle` is a ~136-byte AoS record with type-specific cold fields

**Evidence:** `systems/effect_system.hpp:20-47` — every particle carries
orbit fields, arc fields, and a `const std::string* model` regardless of
velocity type (132 B of fields, 136 B with tail padding);
`update_particle` (`effect_system.cpp:158-245`) branches per
particle on `vel_def.type`, which is constant per emitter.
**Fix (opportunity, only at thousands of particles):** hoist the type switch
to the emitter loop (three tight loops) and/or split hot fields
(position/velocity/age) from cold. **Cost class: cache-miss noise today;
matters ≥5k particles.**

---

## Parallelization opportunities (explicitly *not* defects)

Per the design notes ("default to serial"), these are flagged as future
opportunities gated on measured workloads, in descending value:

1. **Cluster light binning** (`light_cluster.cpp:276`) — already written
   against `parallel_for`; simply initializing the JobSystem would parallelize
   it, but do H2's algorithmic fix first (it likely removes the need).
2. **`build_instance_batches`** (`scene_renderer.cpp:1210-1286`) — the
   cull+bin loop is independent per command; parallel chunks with per-chunk
   bins merged at the end. Only worth it ≥ several thousand model commands.
3. **Host-side animation updates** — `AnimationPlayer::update` instances are
   fully independent (per-player cursors were designed for exactly this);
   a host-side `parallel_for` over characters is the natural first parallel
   workload once character counts reach the hundreds.
4. **Particle updates** (`effect_system.hpp:182-190`) — independent per
   emitter; thousands-of-particles territory.

---

## Prioritized top-10 action list

| # | Action | Finding | Est. payoff |
|---|--------|---------|-------------|
| 1 | Instance the particle renderer (per-frame particle SSBO + one draw per emitter/mesh) | H1 | ms-class at high particle counts |
| 2 | Invert cluster binning to per-light cluster-range insertion | H2 | ms-class with ≥100 lights |
| 3 | Glyph-atlas the text renderer; kill per-string texture creation | M1 | 100 µs–ms with dynamic HUD text |
| 4 | Pack instance/shadow/palette data directly into mapped transfer buffers (add `GPUBuffer::map`) | M2 | 0.1–0.5 ms at 5–10k instances |
| 5 | Delete the per-update identity tail-fill in `AnimationPlayer` | M3 | 0.1–0.2 ms at 100 characters |
| 6 | Replace palette-dedupe linear scan with allocation-free open-addressing hash | M4 | up to ~0.5 ms near palette cap |
| 7 | Add index-based `set_float/set_bool` param overloads to `AnimationStateMachine` | M5 | tens of µs at hundreds of entities |
| 8 | Move (don't copy) UI text through `UIScene`/`UIRenderer`; arena-back label storage | M6 | µs-class + allocator churn removal |
| 9 | Hoist per-emitter constants + squared-length tests in `update_particle`; cheapen `build_particle_transform` | L1, H1 | tens of µs at thousands of particles |
| 10 | `const char*` pass names in `GPUTimestampPool`; word-wise (or dirty-flag) cluster-offsets hash | L5, L2 | ~10–30 µs housekeeping |

---

*Audit performed by static inspection on 2026-07-03. All file:line references
verified against the working tree at time of writing. No builds were run; no
code outside this report was modified.*

---

## Verification

Independent adversarial verification pass (2026-07-03): every finding was
re-checked against the cited source and the design docs, with intent to
refute. No finding was refuted; several line references and detail
figures were corrected in place (noted below).

| Finding | Verdict | Notes |
|---|---|---|
| H1. Particle rendering: one draw + 288 B uniform push per particle | **CONFIRMED** | 288 B size of `ModelTransformUniforms` verified (`gpu_uniforms.hpp:15-25`); per-particle push+draw at `effect_renderer.cpp:214-225` (line refs corrected). No doc claims this is intentional. |
| H2. Cluster light binning O(3,456 × N) serial | **CONFIRMED** | `JobSystem` never initialized → `parallel_for` serial verified; design notes call the CPU bin "fine for v1" with GPU compute for >>100 lights, which does not cover the CPU-side algorithmic improvement. Line refs corrected to 276-295. |
| M1. Per-string text texture creation each frame for changing text | **CONFIRMED** | Rasterize/texture/transfer-buffer per cache miss, one-frame latency, 300-frame expiry, full-cache sweep all verified at the cited lines. |
| M2. Triple pass over instance data | **CONFIRMED** | Bins → `packed_instances_` → mapped transfer buffer verified; `InstanceData` = 160 B verified. Line refs corrected (1284-1289, 1328-1334, 1362-1371). |
| M3. `AnimationPlayer` identity tail rewrite every update | **CONFIRMED** | Tail fill at `animation_player.cpp:210-213` verified; `reset()` initializes tails, per-frame loops stop at `count`, consumers (palette packing, IK) never read past `joint_count`. Consumer line ref corrected to 1429-1430. |
| M4. O(N²) bone-palette dedupe scan | **CONFIRMED** | Linear scan verified at 1410-1416 (ref corrected). The in-code comment documents it as deliberate for small counts — report now says so; scaling concern near the 1024 cap stands. |
| M5. String-keyed state-machine parameters | **CONFIRMED** | `set_float`/`set_bool` → `ensure_param` `unordered_map<string,int>` lookup verified; per-entity-per-frame host pattern is documented in CLAUDE.md. |
| M6. UI text label copies | **CONFIRMED** (detail corrected) | Three copies verified (`ui_scene.cpp:46`, `ui_renderer.cpp:541`, `546`); the original "hashed twice per frame" claim held only on cache miss — steady-state cached labels hash once. Title and text corrected. |
| L1. Per-particle sqrt zero-tests | **CONFIRMED** | Gravity `glm::length` at `effect_system.cpp:217`, velocity length/normalize at 228-230 (line refs corrected from 210/221). |
| L2. Byte-wise FNV over 27.6 KB | **CONFIRMED** | Byte loop at `light_cluster.cpp:71-79`, called at 375 (refs corrected). |
| L3. Uncached `glm::inverse(projection)` in `begin_frame` | **CONFIRMED** | Unconditional at `light_cluster.cpp:155` (ref corrected); renderer-side cache at `scene_renderer.cpp:541-544` verified. Sub-µs as stated. |
| L4. Terrain tile linear scan per pass | **CONFIRMED** (estimate corrected) | ~2,700 main tiles matches the in-code comment; but shadow cascades scan the 4×-coarser `shadow_tiles_` (~190), not the main list — total ≈ 3-4k tests, not 5-10k. Report corrected. |
| L5. `GPUTimestampPool` per-frame `std::string` churn | **CONFIRMED** | `emplace_back(pass_name)` into `vector<std::string>` cleared by `begin_frame()` verified; pool is a zeros-returning stub per architecture.md. |
| L6. Skinned main-view re-resolves `Model*`/bounds | **CONFIRMED** | Three per-command resolutions per frame verified (palette pack 1395, main view 979-983, shadow scratch 1920-1938); line refs corrected. |
| L7. `Particle` AoS with cold fields | **CONFIRMED** (size corrected) | Field sum is 132 B; `sizeof` is 136 B with tail padding (8-byte pointer alignment). Per-particle branch on per-emitter `vel_def.type` verified. Framed as opportunity, not defect. |
