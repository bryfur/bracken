# Shader Audit — HLSL Sources & CPU↔GPU Contract

**Date:** 2026-07-03
**Scope:** every `*.hlsl` / `*.hlsli` under `src/engine/shaders/` (42 files, all read in full),
cross-checked against `gpu/gpu_uniforms.hpp`, `render/terrain_renderer.hpp`,
`render/grass_renderer.hpp`, `render/lighting/light.hpp`, `render/lighting/light_cluster.{hpp,cpp}`,
`render/shadow_map.cpp`, `scene/scene_renderer.cpp`, and `gpu/gpu_uniforms.test.cpp`.
Deliberate design decisions from `docs/engine_design_notes.md` (serial job system, unwired
render_graph, analytic `ambientPBR` as IBL stand-in, CPU cluster binning) were treated as
intentional and are not flagged.

## Executive summary

The shader tree is in good shape overall: binding-space usage is uniformly correct against the
documented SDL3 GPU convention (vertex uniforms in space1, fragment samplers in space2, fragment
uniforms in space3, vertex storage after vertex samplers in space0), the big cbuffers
(`LightingUniforms`, `ShadowUniforms`, `SkyUniforms`, `GTAOUniforms`, `VolumetricFogUniforms`,
terrain/grass uniforms) match their CPU mirrors byte-for-byte, sky pixels are early-outed in every
post pass, AO and volumetric fog run at half resolution, and the expensive loops (PCF/PCSS taps,
GTAO/SSAO kernels, bloom taps, 24-step fog march) are all statically bounded.

Three problems stand out. **First and most serious: the clustered-light screen lookup is
vertically flipped relative to the CPU binning** — the CPU maps cluster row `cy=0` to NDC
`y=-1` (bottom of screen) while the shader maps `cy=0` to `SV_Position.y=0` (top of screen), so
fragments read the mirror-image cluster row; the bug is masked by conservative binning and large
light radii but mis-lights any small point/spot light away from the vertical screen center.
**Second**, two cbuffers (`shadow_depth.frag`, `ui.frag`) pad with `float _pad[3]` / `int
_padding[3]` arrays, which under HLSL cbuffer rules take a 16-byte stride — the declared GPU block
(52 B) does not match the 16-byte CPU struct, exactly the trap `ssao.frag`'s comment warns about.
**Third**, the cluster light-payload offsets are hardcoded to 256 lights in
`cluster_lighting.hlsli` with no compile-time link to `MAX_LIGHTS` or the runtime `max_lights_`
parameter. The rest of the findings are efficiency trims (a dead `fogDistance` varying computed
with a per-vertex `sqrt` in three vertex shaders, an AO "sky" test that is actually a
near-plane-dependent distance cutoff — ~11.5 km with the built-in camera) and robustness hardening (unclamped cluster light count, sin-based hash precision
at 32 km world scale, a couple of theoretical `normalize(0)`s).

## Files read

**Includes:** `pbr.hlsli`, `lighting.hlsli`, `fog.hlsli`, `shadow_common.hlsli`, `clusters.hlsli`,
`cluster_lighting.hlsli`.
**Scene:** `model.vert.hlsl`, `model.frag.hlsl`, `skinned_model.vert.hlsl`,
`model_instanced.vert.hlsl`, `model_instanced.frag.hlsl`, `terrain.vert.hlsl`, `terrain.frag.hlsl`,
`grass.vert.hlsl`, `grass.frag.hlsl`, `skybox.vert.hlsl`, `skybox.frag.hlsl`, `particle.frag.hlsl`.
**Shadows:** `shadow_depth.vert.hlsl`, `shadow_depth.frag.hlsl`, `shadow_depth_instanced.vert.hlsl`,
`shadow_depth_opaque.frag.hlsl`, `shadow_skinned.vert.hlsl`, `shadow_terrain.vert.hlsl`.
**Post:** `fullscreen.vert.hlsl`, `ssao.frag.hlsl`, `gtao.frag.hlsl`, `blur_ao.frag.hlsl`,
`bloom_downsample.frag.hlsl`, `bloom_upsample.frag.hlsl`, `volumetric_fog.frag.hlsl`,
`composite.frag.hlsl`, `fxaa.frag.hlsl`.
**UI/debug:** `ui.vert.hlsl`, `ui.frag.hlsl`, `text.vert.hlsl`, `text.frag.hlsl`,
`billboard.vert.hlsl`, `billboard.frag.hlsl`, `grid.vert.hlsl`, `grid.frag.hlsl`.

## Contract verification (what checked out)

- `ModelTransformUniforms` / `TransformUniforms` (`gpu_uniforms.hpp:15` ↔ `model.vert.hlsl:30`,
  `skinned_model.vert.hlsl:34`): offsets 0/64/128/192(+12 pad)/208/272/276/280 — exact match;
  covered by `gpu_uniforms.test.cpp:79`.
- `ModelLightingUniforms` ↔ `model.frag.hlsl:17` and `InstancedLightingUniforms` ↔
  `model_instanced.frag.hlsl:17`: every `float3` is followed by a scalar; the 12-scalar tail
  (`hasTexture`…`_padding4`) never straddles a register — exact match (208 B / 192 B).
- `ShadowDataUniforms` ↔ the `ShadowUniforms` cbuffer declared in all five shadow-receiving
  fragment shaders: `mat4[4]` (natural 64-B array stride) + 3×`float4` + 8 scalars = 336 B — match.
- `SkyboxFragmentUniforms` ↔ `skybox.frag.hlsl:24` (176 B), `VolumetricFogUniforms` ↔
  `volumetric_fog.frag.hlsl:17` (224 B), `CompositeUniforms`, `BlurUniforms`,
  `Bloom{Down,Up}sampleUniforms`, `FxaaUniforms`, `ParticleFragmentUniforms`,
  `GridVertexUniforms`, `UIScreenUniforms` — all match.
- `GTAOUniforms` ↔ `ssao.frag.hlsl:19` / `gtao.frag.hlsl:17`: the three scalar paddings (instead
  of `float[3]`) put `sampleKernel` at byte 176 on both sides — correct, and the shader comment
  (`ssao.frag.hlsl:30-31`) documents exactly why.
- `TerrainTransformUniforms`/`TerrainLightingUniforms` (`terrain_renderer.hpp:39,50`) ↔
  `terrain.vert.hlsl:28` / `terrain.frag.hlsl:17` — match (112 B fragment block).
- `GrassVertexUniforms` (`grass_renderer.hpp:36`) ↔ `grass.vert.hlsl:35`: 160 B, including the
  tricky trailing `float3 cameraForward` that packs at offset 148 (fits the last 12 bytes of
  register 9 without straddling) — match, but fragile; see M-5.
- `InstanceData` / `ShadowInstanceData` structured-buffer layouts (`gpu_uniforms.hpp:161,172`) ↔
  `model_instanced.vert.hlsl:40` / `shadow_depth_instanced.vert.hlsl:26` — 160 B / 64 B, match.
- `PointLight`/`SpotLight`/`LightHeader` (`light.hpp:11-34`, static_asserts at
  `light_cluster.cpp:19-20`) ↔ `clusters.hlsli:24-39` and the `ByteAddressBuffer` decode in
  `cluster_lighting.hlsli:44-72` — field order and strides (32/64/8 B) match.
- Binding spaces: all vertex cbuffers `[[vk::binding(0,1)]]`; all fragment cbuffers in set 3
  (b0 lighting, b1 shadow, b2 cluster params); fragment samplers set 2 with cluster SSBOs correctly
  offset past the sampler count (`model*.frag` base 6, `terrain.frag` base 4, `grass.frag` base 2);
  vertex storage buffers in set 0 after vertex samplers (`grass.vert.hlsl:74` binding 2 after 2
  samplers). All consistent with `shaders/README.md`.
- `ShadowMap::get_shadow_uniforms` (`shadow_map.cpp:254-276`) clones inactive cascades so
  `lastActiveSplit == cascadeSplits[last]`; combined with the `splitFar < lastActiveSplit` guard in
  `shadow_common.hlsli:134`, `sampleShadowCascade(cascade + 1, …)` can never index cascade 4 —
  no out-of-bounds read.
- Half-rate targets: AO at half res (`ambient_occlusion.cpp:79-80`), volumetric fog at half res
  (`volumetric_fog.cpp:21-22`), bloom on a 4-6 level mip chain (`bloom.cpp:62-68`) — all correct.

---

## High severity

### H-1. Clustered-light screen lookup is vertically flipped vs. CPU binning

**Evidence:**
- CPU binning, `render/lighting/light_cluster.cpp:52`:
  `float ndc_y = (cy / CLUSTER_DIM_Y) * 2.0f - 1.0f;` fed directly to `invProj` (no Y flip), so
  cluster row `cy = 0` covers glm-NDC `y ∈ [-1, -7/9]` → view-space **−Y** → **bottom** of screen.
- Shader lookup, `clusters.hlsli:51`:
  `uint cy = (uint)clamp(floor(uv.y * (float)cp.gridDim.y), …)` where `uv` is
  `input.position.xy * clusterParams.screenSize.zw` (`model.frag.hlsl:160`,
  `model_instanced.frag.hlsl:152`, `terrain.frag.hlsl:180`, `grass.frag.hlsl:103`).
  `SV_Position.y = 0` is the **top** of the screen, so `cy = 0` is the **top** row.
- The engine's own convention proves clip +Y = screen top: every UV→clip reconstruction negates Y
  (`lighting.hlsli:66`, `ssao.frag.hlsl:47`, `gtao.frag.hlsl:46`, `volumetric_fog.frag.hlsl:53`).
  The cluster path is the only screen↔NDC conversion in the tree without that flip.

**Why it matters:** fragments read the mirror-image cluster row: a pixel in the bottom third of the
screen fetches the light list binned for the top third. Row 4 of 9 (screen center) self-maps, and
conservative sphere-vs-AABB binning plus multi-tile light radii pad the error, which is why it can
pass casual visual inspection — but any point/spot light whose screen footprint is small and off
vertical center is dimmed or dropped, and lights pop as the camera pitches. X is not affected
(`cx=0` ↔ NDC −1 ↔ left ↔ `uv.x=0` — consistent).

**Recommendation:** flip Y on one side, e.g. in `clusterIndexFromScreen`
(`clusters.hlsli:49-54`): `uint cy = (uint)clamp(floor((1.0 - uv.y) * (float)cp.gridDim.y), …)`.
Then add a regression test in `light_cluster.test.cpp` that bins a light known to sit at NDC
y≈−0.9 (bottom) and asserts the shader-side index math (replicated in C++) lands in the same
cluster row.

### H-2. `shadow_depth.frag` / `ui.frag` pad with cbuffer arrays — declared GPU block ≠ CPU struct

**Evidence:** `shadow_depth.frag.hlsl:14-18`:

```hlsl
cbuffer ShadowFragUniforms {
    int hasTexture;
    float _pad[3];
};
```

and `ui.frag.hlsl:19-23` (`int has_texture; int _padding[3];`). In HLSL cbuffer layout, array
elements have a 16-byte stride: `float _pad[3]` starts at offset 16 and occupies 36 bytes, so the
declared block is 52 bytes. The CPU pushes 16 bytes (`scene_renderer.cpp:1634-1639` — anonymous
`{int; float _pad[3];}` struct — and `gpu_uniforms.hpp:107` `UIFragmentUniforms`, pushed at
`ui_renderer.cpp:270`).

**Why it matters:** today it is benign — only offset 0 is read, and SDL's uniform ring allocation
absorbs the size gap. But it silently violates the byte-for-byte contract that
`shaders/README.md:101-109` declares, and it is a loaded trap: appending any real field after the
pad array puts it at HLSL offset 52+ while the CPU appends at 16+ — an invisible corruption of the
new field. `ssao.frag.hlsl:30-31` documents this exact rule ("Scalars (not a float array) so the
kernel lands at byte 176") — these two shaders just don't follow it.

**Recommendation:** replace the arrays with three named scalars in both cbuffers
(`float _pad0; float _pad1; float _pad2;`), matching the ssao/gtao pattern, and give
`ShadowFragUniforms` a named CPU mirror in `gpu_uniforms.hpp` (the inline anonymous struct at
`scene_renderer.cpp:1634` is a second, un-audited copy of the layout).

---

## Medium severity

### M-1. Cluster light payload offsets hardcode 256 with no link to `MAX_LIGHTS`

**Evidence:** `cluster_lighting.hlsli:37,45,57` — `uint base = 16 + i * 8;`,
`16 + 256 * 8 + i * 32`, `16 + 256 * 8 + 256 * 32 + i * 64`. The CPU sizes and packs the buffer
from a **runtime** `max_lights_` (`light_cluster.cpp:35-37,345-347`), settable via
`ClusterGrid::init(device, max_lights = MAX_LIGHTS)` (`light_cluster.hpp:36`).
`scene_renderer.cpp:180` happens to pass `MAX_LIGHTS` (= 256, `light.hpp:46`).

**Why it matters:** raising `MAX_LIGHTS`, or any caller passing a different `max_lights`, shifts
the CPU-side `points_off`/`spots_off` while the shader keeps reading at the 256-light offsets —
every light payload decodes as garbage, with no validation layer error (it's all in-bounds
`ByteAddressBuffer` reads).

**Recommendation:** either (a) put the light count into the unused `maxPerCluster.y` (or
`gridDim`) and compute the section offsets in the shader from it, or (b) drop the runtime
parameter: hard-assert `max_lights_ == MAX_LIGHTS` in `ClusterGrid::init` and add
`// must equal cluster_lighting.hlsli section stride` comments on both sides.

### M-2. Cluster loop count is trusted from the SSBO; `maxPerCluster` uniform is dead

**Evidence:** `cluster_lighting.hlsli:79-83` — `count` comes straight from
`clusterOffsetsBuf[cidx].y` and drives `for (uint i = 0; i < count; ++i)`. `maxPerCluster`
(`clusters.hlsli:21`, mirrored in two CPU headers) is never referenced by any shader
(verified by grep over `src/engine/shaders/`).

**Why it matters:** a stale or partially-uploaded offsets buffer (e.g. the
`last_offsets_hash_` skip in `light_cluster.cpp` combined with a resize edge case) can make a
fragment iterate up to 2³² times — a device hang, not a visual glitch. The engine already ships a
zeroed-`ClusterParams` safety story for the no-lights frame (`docs/architecture.md:289-291`);
this is the matching in-shader guard.

**Recommendation:** `count = min(count, clusterParams.maxPerCluster.x);` — one instruction, and it
makes the uniform earn its slot. Similarly cheap: `offset + i` could be clamped, but count alone
removes the hang risk.

### M-3. Dead `fogDistance` varying — a `sqrt` per vertex and an interpolant on every mesh draw

**Evidence:** written in `model.vert.hlsl:60`, `skinned_model.vert.hlsl:80`,
`model_instanced.vert.hlsl:71` (`output.fogDistance = length(worldPos.xyz - cameraPos);`).
No consuming fragment shader reads it: `model.frag.hlsl`, `model_instanced.frag.hlsl`, and
`particle.frag.hlsl` mention `fogDistance` only in their `PSInput` declaration (verified by grep);
fog is computed analytically from `fragPos`/`cameraPos` inside `applyAtmosphericFog`
(`fog.hlsli:32-51`).

**Why it matters:** one `length()` (sqrt) per vertex across every static, skinned, instanced, and
particle draw, plus a wasted scalar interpolant (the instanced path is already at 9 varying
slots + SV_Position).

**Recommendation:** delete `fogDistance` from the three `VSOutput`s and the three paired
`PSInput`s in the same change (they must stay byte-matched — `particle.frag.hlsl:2-3` documents
that coupling).

### M-4. AO "skybox skip" at `depth >= 0.9999` is really a near-plane-dependent distance cutoff

**Evidence:** `ssao.frag.hlsl:59`, `gtao.frag.hlsl:60`, `blur_ao.frag.hlsl:49` — all early-out
with `depth >= 0.9999` under a "Skip skybox" comment. With glm RH_ZO (non-reversed) depth,
`depth = far·(z−near)/(z·(far−near))`; the near/far that matter are those of the camera
projection that produced the depth buffer — the engine's built-in `CameraSystem` uses
near = 5, far = 15 000 (`camera_system.cpp:361`), for which depth 0.9999 is crossed at
z ≈ 11 540. AO (and its bilateral blur) silently turns off for the farthest ~23 % of the view
range, and the cutoff moves with the near plane (near = 1 at the same far pulls it to ~6 km;
a host camera with near = 0.1, far = 5000 puts it at ~830 units).
`volumetric_fog.frag.hlsl:90` uses `depth >= 1.0` — exact sky only — so the two passes disagree
about what "sky" means.

**Why it matters:** mostly it is an accidental (and arguably beneficial) LOD: at an AO radius of
4 world units, distant AO is subpixel anyway. But it is undocumented, it makes the AO range
depend on the host's near plane, and the returned `1.0` skips the blur's edge handling at the
cutoff boundary.

**Recommendation:** keep the optimization but make it intentional: compare against an explicit
linearized max-AO-distance (the blur pass already has `depthA`/`depthB` for exactly this), or at
minimum change the comment to "sky + far-field cutoff (near-plane dependent)" and align
`volumetric_fog.frag` if the same LOD is wanted there.

### M-5. Renderer-local uniform structs have no layout tests; `GrassVertexUniforms` sits on a packing knife-edge

**Evidence:** `gpu_uniforms.test.cpp` covers only `gpu_uniforms.hpp` structs.
`TerrainLightingUniforms` (`terrain_renderer.hpp:50`), `GrassVertexUniforms`
(`grass_renderer.hpp:36`), and `GrassLightingUniforms` (`grass_renderer.hpp:63`) rely on
hand-maintained offset comments. `GrassVertexUniforms`'s trailing
`glm::vec3 camera_forward` at offset 148 matches HLSL only because a `float3` exactly fits the
last 12 bytes of the register starting at 144 (`grass.vert.hlsl:55-56`); inserting any scalar
before it desynchronizes the two sides without a compile error on either.

**Why it matters:** this is the project's self-declared "most common way to silently break
rendering" (`CLAUDE.md`), and the grass struct is the most fragile layout in the tree.

**Recommendation:** add `static_assert(offsetof(...))` blocks (or extend the gtest) for the three
renderer-local structs — at minimum `offsetof(GrassVertexUniforms, camera_forward) == 148` and
`sizeof(GrassVertexUniforms) == 160`.

---

## Low severity

### L-1. Dead uniforms
- `useSkinning` (`model.vert.hlsl:38`, `skinned_model.vert.hlsl:42`) is declared in both vertex
  shaders and set by the CPU (`scene_renderer.cpp:1125,1688`) but read by neither — the skinned
  pipeline "always applies skinning" (`skinned_model.vert.hlsl:55`). Remove field from both HLSL
  cbuffers + `ModelTransformUniforms` (renumber pads), or keep and document as reserved.
- `nearPlane` / `farPlane` (`volumetric_fog.frag.hlsl:29-30`) are never referenced in the shader
  body; the CPU still fills them (`gpu_uniforms.hpp:345-346`).
- `maxPerCluster` — see M-2.

### L-2. Billboard pipeline and shaders have no draw path and no CPU uniform mirror
`billboard.vert.hlsl` declares a `BillboardUniforms` cbuffer (view/projection/worldPos/size/offset)
with **no** mirror in `gpu_uniforms.hpp`; nothing in the engine binds
`PipelineType::Billboard` for a draw — `SceneRenderer::draw_billboard_3d`
(`scene_renderer.cpp:1763-1791`) renders billboards as screen-space UI rects instead. The pipeline
is still created, preloaded, and invalidated on swapchain format changes
(`pipeline_registry.cpp:198`). Either delete the pipeline+shaders or document them as host-facing
API and add the CPU uniform struct so the contract is auditable.

### L-3. Helper duplication across post shaders
`interleavedGradientNoise` exists four times (`lighting.hlsli:27`, `shadow_common.hlsli:27` —
acknowledged, `ssao.frag.hlsl:52`, `gtao.frag.hlsl:52`, plus inlined at
`volumetric_fog.frag.hlsl:114`); depth/world reconstruction is re-implemented in `ssao.frag:45`,
`gtao.frag:44`, `volumetric_fog.frag:51` while `lighting.hlsli:64-76` provides unused
`viewPosFromDepth`/`worldPosFromDepth`. Meanwhile `model.frag.hlsl:99`,
`model_instanced.frag.hlsl:98`, `grass.frag.hlsl:62`, `terrain.frag.hlsl:73` include
`lighting.hlsli` and use nothing from it. Pure maintenance risk (constants drifting apart);
consolidate when convenient.

### L-4. `terrain.frag` sin-based hash degrades at world scale
`terrain.frag.hlsl:85-87`: `hash2D` = `frac(sin(dot(p, (127.1, 311.7))) * 43758.5453)`. With the
default 32 000-unit world, `worldPos.xz * 0.004` (line 127) makes the dot product reach ~4×10⁴,
where fp32 `sin` argument reduction is GPU-vendor-dependent and loses most of its mantissa — the
anti-tiling noise (and the roughness variation riding it, line 162) can band or plateau far from
the origin. `grass.vert.hlsl:77-102` already uses an integer murmur hash immune to this; port it.

### L-5. `normalize(0)` / degenerate-input edges
- `grass.vert.hlsl:214`: `normalize(hashFloat2(seed + int2(5,7)) - 0.5)` — NaN blade if the hash
  lands exactly on (0.5, 0.5); probability ≈ 2⁻⁴⁸ per blade but free to guard
  (`+ float2(1e-4, 0)` or `max(len, 1e-6)` divide).
- `ssao.frag.hlsl:95-98`: samples projected with `clipPos.w <= 0` (geometry within the effective
  sample radius of the camera plane) aren't rejected — negative-w projection mirrors the UV, which
  can pass the bounds check and fetch an unrelated depth. Not reachable with the built-in camera
  (near = 5, `camera_system.cpp:361`) at the default `ao_radius` multiplier (effective radius
  4.0 < near), but a host-supplied camera with a smaller near plane, or `ao_radius > 1.25`,
  exposes it. Add `if (clipPos.w <= 0.0) continue;`.
- `pbr.hlsli:61` computes `H = normalize(V + L)` before the `NdotL <= 0` early-out, but the
  NaN `H` is provably unused on the early-out path (NdotH is computed after the return) — OK as
  written; worth a comment so a refactor doesn't reorder it.

### L-6. PCSS cost doubles in the cascade blend band
`shadow_common.hlsli:130-138`: in the last 15 % of each cascade, `sampleShadowCascade` runs twice —
in PCSS mode that is 2 × (8 blocker taps + 12 PCF-compare taps) = 40 atlas taps per fragment, on
top of the per-cascade normal-offset matrix multiply (line 45). Options if it ever shows up in a
capture: dither the blend with IGN and pick one cascade per pixel, or drop the second cascade's
blocker search and reuse the first's penumbra estimate.

### L-7. Full shadow filter runs even where the sun contribution is zero
`model.frag.hlsl:145-155` (same in instanced/terrain/grass): `calcShadow` executes at full PCSS
rate before `evaluatePBR`'s `NdotL <= 0` rejection; for back-facing fragments the only consumer is
the `ambientLift` scalar (line 153). A cheap `NdotL <= 0` path (skip the blocker search, single
tap) would save the worst-case 20 taps on roughly half of all geometry — art-affecting, so
measure/eyeball before and after.

### L-8. Cosmetic
- `shadow_common.hlsli:26` note ("lighting.hlsli is included after this file") is accurate today
  for all five includers — keep it true or extract the constant.

---

## Top-10 prioritized actions

1. **Fix the cluster-lookup Y flip** (`clusters.hlsli:51` — use `1.0 - uv.y`), and add a
   CPU-side regression test for screen→cluster orientation. (H-1)
2. **Replace `float _pad[3]` / `int _padding[3]` with three scalars** in
   `shadow_depth.frag.hlsl:17` and `ui.frag.hlsl:22`; add a named `ShadowFragUniforms` CPU struct.
   (H-2)
3. **Clamp the cluster light loop**: `count = min(count, clusterParams.maxPerCluster.x)` in
   `cluster_lighting.hlsli:83`. (M-2)
4. **Tie the 256-light payload offsets to one constant**: assert `max_lights_ == MAX_LIGHTS` in
   `ClusterGrid::init` or derive offsets from a value in `ClusterParams`.
   (M-1)
5. **Delete the dead `fogDistance` varying** from `model.vert` / `skinned_model.vert` /
   `model_instanced.vert` and their paired `PSInput`s. (M-3)
6. **Add layout static_asserts/tests** for `GrassVertexUniforms`, `GrassLightingUniforms`,
   `TerrainLightingUniforms`. (M-5)
7. **Make the AO far cutoff explicit** (linearized distance threshold or a documented constant)
   and reconcile with `volumetric_fog.frag`'s `>= 1.0` sky test. (M-4)
8. **Remove or wire the Billboard pipeline**; if kept, add its CPU uniform mirror. (L-2)
9. **Swap `terrain.frag`'s sin-hash for the integer murmur hash** already in `grass.vert`. (L-4)
10. **Prune dead uniforms** (`useSkinning`, volumetric `nearPlane`/`farPlane`) and consolidate the
    duplicated IGN/depth-reconstruction helpers into `lighting.hlsli`. (L-1, L-3)

---

## Verification

Each finding was independently re-checked against the cited sources by an adversarial pass
(code re-read at the cited lines; docs checked for intentional-behavior notes; none found).

| Finding | Verdict | Notes |
|---|---|---|
| H-1 Clustered-light screen lookup vertically flipped vs CPU binning | **CONFIRMED** | CPU `cy=0` → NDC y=−1 (bottom, `light_cluster.cpp:52`, no Y flip); shader `cy=0` = `SV_Position.y=0` (top); no VS-side Y flip exists; all other screen↔clip conversions negate Y. |
| H-2a `shadow_depth.frag` cbuffer pads with `float _pad[3]` (52 B GPU block vs 16 B CPU push) | **CONFIRMED** | Array stride rule verified; CPU pushes an inline 16-B anonymous struct (`scene_renderer.cpp:1634-1639`). Benign today (only offset 0 read). |
| H-2b `ui.frag` cbuffer pads with `int _padding[3]` | **CONFIRMED** | Same pattern; `UIFragmentUniforms` is 16 B (`gpu_uniforms.hpp`), pushed at `ui_renderer.cpp:268`. |
| M-1 Cluster payload offsets hardcode 256 | **CONFIRMED** | Literal `16 + 256*8` / `+ 256*32` offsets at `cluster_lighting.hlsli:36,45,57`; CPU packs from runtime `max_lights_`; only `scene_renderer.cpp:180` passing `MAX_LIGHTS=256` keeps them aligned. |
| M-2 Cluster light loop count unclamped; `maxPerCluster` dead | **CONFIRMED** | `maxPerCluster` appears only in its declaration (`clusters.hlsli:21`); loop trusts `clusterOffsetsBuf` count. Hang scenario is speculative (CPU always caps counts at 128 and uploads before first use) but the guard is free. |
| M-3 Dead `fogDistance` varying (3 vertex shaders) | **CONFIRMED** | Written in all three VS; every consuming PSInput declares it but no fragment body reads it. |
| M-4 AO `depth >= 0.9999` skip is a distance cutoff | **CONFIRMED (details corrected)** | Mechanism verified, but the original numbers assumed near 0.1 / far 5000 (the `VolumetricFogUniforms` struct defaults, not the camera). The built-in camera is near 5 / far 15 000 (`camera_system.cpp:361`) → cutoff ≈ 11.5 km, i.e. the farthest ~23 % of the view range, not ~83 %. Rewritten above. |
| M-5 Renderer-local uniform structs lack layout tests; `GrassVertexUniforms` knife-edge | **CONFIRMED** | `camera_forward` at offset 148 / sizeof 160 verified by manual packing on both sides; no test file references these structs. |
| L-1 Dead uniforms (`useSkinning`, fog `nearPlane`/`farPlane`) | **CONFIRMED** | `useSkinning` set at `scene_renderer.cpp:1125,1688`, read by neither VS (skinned VS skins unconditionally); fog near/far appear only in the cbuffer declaration. |
| L-2 Billboard pipeline has no draw path, no CPU uniform mirror | **CONFIRMED** | No callers of `get_billboard_pipeline`; no `BillboardUniforms` in `gpu_uniforms.hpp`; `draw_billboard_3d` renders via UI rects; pipeline is in the preload list (`pipeline_registry.cpp:198`). |
| L-4 `terrain.frag` sin-based hash at world scale | **CONFIRMED** | Hash/scale/roughness usage verified; the 32 000-unit default world is real (`application.hpp:82`), giving sin arguments ~4×10⁴; grass already uses a murmur hash. |
| L-5 SSAO `clipPos.w <= 0` not rejected; grass `normalize(0)` | **CONFIRMED (impact qualified)** | Both code facts verified (`ssao.frag.hlsl:95-98`, `grass.vert.hlsl:213`). SSAO case cannot trigger with the built-in camera (near 5 > effective radius 4 at default `ao_radius=1.0`); latent for host cameras. Noted above. |
| L-6/L-7 PCSS blend-band cost doubles; full filter where NdotL ≤ 0 | **CONFIRMED** | 8 blocker + 12 filter taps per cascade verified in `shadow_common.hlsli`; both cascades sampled in the 15 % band; `calcShadow` runs before any NdotL rejection in `model.frag.hlsl:145`. |
| L-3 Duplicated IGN/depth helpers; `lighting.hlsli` included unused by 4 shaders | **CONFIRMED** | Grep verified: 4 IGN copies + 1 inline; local `viewPosFromDepth` in ssao/gtao/fog; none of the four frag shaders use any `lighting.hlsli` symbol. |
| fxaa luma comment "green-weighted" vs Rec.601 code | **REFUTED (removed)** | `dot(rgb, float3(0.299, 0.587, 0.114))` *is* a green-dominant weighted dot product; the comment accurately describes the code unless "green-weighted" is read as FXAA's pure-green luma, which the comment does not claim. No doc/code mismatch. |
