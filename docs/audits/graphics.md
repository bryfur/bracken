# Graphics Audit — mmo_engine

**Date:** 2026-07-03
**Scope:** feature completeness and correctness of the rendering stack versus 2026 expectations for an indie-scale real-time engine targeting outdoor MMO-style scenes.
**Method:** read-only review of `src/engine/render/`, `src/engine/scene/scene_renderer.cpp`, `src/engine/gpu/`, and every shader in `src/engine/shaders/`, cross-checked against `docs/architecture.md` and `docs/engine_design_notes.md`. Deliberate design decisions documented there (serial job system, unwired render_graph, CPU cluster binning) are not flagged as defects.

---

## Executive summary

The engine's rendering foundation is in unusually good shape for its size. Color management is genuinely correct end to end: base-color/emissive textures upload as `RGBA8_SRGB` while normal/MR maps stay linear (`model_loader.cpp:738,756,768,779`), all lighting runs in an RGBA16F linear target, and a single composite pass tonemaps (fitted ACES) and hand-encodes sRGB with a post-encode half-LSB IGN dither (`composite.frag.hlsl:98-105`). The BRDF is a physically plausible Cook-Torrance: GGX NDF with correct perceptual-roughness squaring, height-correlated Smith visibility, Schlick Fresnel at LdotH, Burley diffuse with `(1-F)(1-metallic)` energy split, plus Kaplanyan specular AA (`pbr.hlsli:13-126`). The CSM implementation is above average — bounding-sphere cascade fitting with texel snapping, depth-clamp pancaking, front-face caster culling, normal-offset plus slope-scaled receiver bias, hardware-PCF 3×3 tent and a real PCSS mode, cascade blending, and distance fade (`shadow_map.cpp`, `shadow_common.hlsli`).

The most significant correctness problems are: (1) standard (non-reversed) Z with a 0.1 near plane against draw distances up to 32 km, which wastes most of the D32 float depth precision exactly where an outdoor MMO needs it; (2) glTF alpha semantics parsed by the loader but ignored by the renderer — every textured mesh, including `OPAQUE` ones, gets a hardcoded 0.5 alpha-test, `BLEND` and `doubleSided` are unimplemented, and there is no transparent draw path at all; (3) SSAO applied as a multiplier on the final lit color at composite, patched with a luminance "protection" hack the code itself annotates as wrong; and (4) the volumetric fog pass hardcoding its own sun/fog colors and density, ignoring the `EnvironmentState` that the rest of the frame treats as the single source of truth.

On the gap side, the highest value-for-effort items for this engine (outdoor, foliage-heavy, long view distances, many animated characters) are reversed-Z, TAA with motion vectors (FXAA cannot fix alpha-tested grass/tree shimmer, the dominant artifact class in these scenes), and IBL (the design notes already identify `ambientPBR` as the drop-in upgrade point). SSR, motion blur/DoF, and terrain virtual texturing are poor fits for the current content and correctly absent.

---

## Feature inventory (what exists, verified in code)

| Area | Status | Evidence |
|---|---|---|
| Clustered forward+ | CPU-binned 16×9×24 grid, 256 point+spot lights, log-Z slicing, windowed inverse-square falloff, memoized cluster bounds | `render/lighting/light_cluster.cpp:132-206`, `shaders/clusters.hlsli:42-62` |
| CSM shadows | 1–4 cascades in a 2×2 D32 atlas (512–4096/cascade); modes: 0=off, 1=PCF (3×3 tent of hardware 2×2 compares), 2=PCSS (8-tap blocker search + 12-tap poisson, IGN-rotated); sphere-fit + texel snap; pancaking; cascade blend + distance fade | `shadow_map.cpp:107-177`, `shadow_common.hlsli:39-144`, `pipeline_registry.cpp:316-317` |
| AO | SSAO (24-sample Halton hemisphere kernel) and "GTAO" (spiral, depth-derived normals), both half-res + separable bilateral blur on relative linear view-Z | `ambient_occlusion.cpp:32-63,73-96`, `ssao.frag.hlsl`, `gtao.frag.hlsl`, `blur_ao.frag.hlsl:63-78` |
| Bloom | Jimenez 13-tap downsample, Karis average + soft-knee threshold on first mip, 9-tap tent progressive additive upsample, 4–6 RGBA16F mips | `bloom_downsample.frag.hlsl:37-99`, `bloom.cpp:62-209` |
| Volumetric fog / god rays | Half-res, 24-step dithered ray march against depth, single raw shadow tap per step from cascade 0, height-exponential density, premultiplied composite | `volumetric_fog.frag.hlsl:86-142`, `volumetric_fog.cpp:17-33,94-161` |
| Analytic fog | Distance+height atmospheric fog with sun-side tint, matched between surface shaders and skybox haze | `fog.hlsli:32-51`, `skybox.frag.hlsl:127-132` |
| AA | FXAA 3.11 quality preset on the tonemapped LDR image; no TAA, no MSAA (pipelines fixed at `SDL_GPU_SAMPLECOUNT_1`, `gpu_pipeline.hpp:56`) | `fxaa.frag.hlsl`, `scene_renderer.cpp:735-754` |
| Tonemap/grading | ACES fitted / ACES Narkowicz / Reinhard / clamp; exposure, post-tonemap contrast + saturation; post-encode IGN dither | `composite.frag.hlsl:75-105`, `lighting.hlsli:31-61` |
| Procedural sky | 3-stop gradient, HDR sun disc (5.0) + corona + glare, 2-layer FBM clouds w/ silver lining, night starfield, horizon haze | `skybox.frag.hlsl:74-148` |
| Terrain | 4-layer splat, triplanar with flat-path fast path, dual-scale anti-tiling noise, explicit-gradient sampling, PBR + clustered lights + CSM | `terrain.frag.hlsl:102-189` |
| Grass | Instanced chunked blades off heightmap, wind, half-Lambert wrap blended with PBR, tip specular, receives shadows + cluster lights | `grass.frag.hlsl:70-116` |
| Skinned meshes | Per-frame packed palette `StructuredBuffer` (no GPU bone cap), tangent-space normal mapping end to end | `skinned_model.vert.hlsl:47-87` |
| Particles | Additive unlit HDR mesh particles; `"sprite"` type declared but **not implemented** (documented stub) | `particle.frag.hlsl`, `docs/architecture.md:288` |
| Culling | CPU frustum culling (main + per-cascade), distance culling, per-model instanced batching | `scene_renderer.cpp:1173-1298` |

Known documented stubs not re-flagged as findings: GPU timestamp pool returns zeros; cluster SSBOs keep stale contents when light count is zero (guarded by zeroed params); `render_graph/` deliberately unwired.

---

## Findings — HIGH

### H1. No reversed-Z; standard [0,1] depth over MMO-scale view distances
**Evidence:** default compare op `SDL_GPU_COMPAREOP_LESS_OR_EQUAL` (`gpu/gpu_pipeline.hpp:44`), depth cleared to 1.0 (`scene/scene_renderer.cpp:417`, `render/ambient_occlusion.cpp:245`), D32_FLOAT target (`gpu/pipeline_registry.hpp:303`). Near plane is recovered and clamped to 0.1 (`scene/scene_renderer.cpp:553`) while `GraphicsSettings::draw_distance` reaches 4000 by default and 32000 at max (`graphics_settings.hpp:26,64-67`).
**Why it matters:** with conventional ZO mapping, hyperbolic depth distribution and float precision both concentrate near z=0/near-plane; at a 0.1 near and multi-kilometer far, nearly all of D32's precision is spent in the first few meters. Distant terrain/props z-fight and depth-derived passes (SSAO/GTAO position reconstruction, volumetric fog world-pos reconstruction, bilateral blur linearization at `blur_ao.frag.hlsl:40-42`) degrade at range. Reversed-Z with a float depth buffer makes precision near-uniform and is essentially free on modern hardware.
**Recommendation:** flip to reversed-Z: GREATER_OR_EQUAL compare, clear depth to 0.0, reversed projection (swap near/far in the RH_ZO projection or post-multiply), flip the skybox z==1 trick to z==0, and audit every reader of raw depth: `ssao.frag.hlsl:59` / `gtao.frag.hlsl:60` / `blur_ao.frag.hlsl:49` (`depth >= 0.9999` sky checks become `<=` epsilon), `volumetric_fog.frag.hlsl:90` (`depth >= 1.0`), the depth-linearization constants in `BlurUniforms` (`gpu_uniforms.hpp:263-274`), and the shadow pipelines (can stay standard-Z; they are orthographic and independently biased).

### H2. glTF alpha modes and double-sided flags are parsed but never honored; no transparent draw path exists
**Evidence:** the loader fills `PBRMaterial::alpha_mode`, `alpha_cutoff`, and `double_sided` (`model_loader.cpp:275-281`, `model_loader.hpp:39-57`), but no renderer file reads them (grep over `scene_renderer.cpp` and `render/*.cpp` finds zero consumers). Instead `model.frag.hlsl:124` and `model_instanced.frag.hlsl:120` hardcode `if (baseColor.a < 0.5) discard;` for every textured mesh, all model pipelines are built `.opaque()` with `CULLMODE_BACK` (`pipeline_registry.cpp:256,285,345`; `gpu_pipeline.hpp:33`), and there is no sorted alpha-blend scene pass anywhere in `render_3d_scene` (`scene_renderer.cpp:900-1031`).
**Why it matters:** three distinct wrong behaviors: (a) `OPAQUE` materials whose base-color texture carries incidental alpha get holes punched in them — the glTF spec requires alpha to be ignored for OPAQUE; (b) `MASK` materials with a non-default cutoff clip at the wrong threshold; (c) `BLEND` materials (glass, ghosts, fading corpses, spell effects — MMO staples) render as hard-edged cutouts. `double_sided` foliage cards lose their back faces.
**Recommendation:** short term, gate the discard on `alpha_mode == Mask` and pass `alpha_cutoff` through `ModelLightingUniforms` (a float slot is spare at `gpu_uniforms.hpp:63`), and add a `no_cull` pipeline variant selected by `double_sided`. Medium term, add a back-to-front-sorted alpha-blended queue drawn after opaques + skybox with depth-write off (the additive particle pipeline at `pipeline_registry.cpp:548-579` is the template).

### H3. SSAO/GTAO darkens final lit color (sun + specular included), patched by a luminance heuristic
**Evidence:** `composite.frag.hlsl:53-63` — the code comment itself says "Physically, AO should only attenuate ambient/indirect light … AO arrives here as a flat multiply on the FINAL lit color," then blends AO toward 1 by `saturate(luminance(color) * 0.6) * 0.65`.
**Why it matters:** the classic SSAO-on-everything artifact: sunlit ground under a character darkens as if shadowed, specular highlights dim in crevices, and the luminance-protection heuristic re-introduces AO leaks on bright-but-ambient-lit pixels (e.g. bright sky-facing surfaces at dusk) while under-occluding dark corners. It also double-counts with the material AO already applied to ambient in `model.frag.hlsl:138-155`.
**Recommendation:** render a depth prepass (also unlocks H1-adjacent wins and reduces overdraw), run AO before the forward pass, bind the blurred AO texture into the forward shaders, and multiply it only into the `ambientPBR` term (`model.frag.hlsl:154`, `terrain.frag.hlsl:174`). Then delete the composite hack and the `aoStrength` multiply on final color.

### H4. Volumetric fog ignores `EnvironmentState` — hardcoded sun color, fog color, density, height
**Evidence:** `volumetric_fog.cpp:122-130` hardcodes `fogDensity = 0.0015`, `lightColor = (1.0, 0.9, 0.7)`, `fogColor = (0.5, 0.6, 0.7)`, `fogHeight = 100`, `fogFalloff = 0.005`, and `nearPlane/farPlane = 0.1/5000`, while `scene_renderer.cpp:484-491` establishes `env_` as "the single source of truth for sun/sky/fog this frame" and every other renderer consumes it (`environment.hpp:16-29`).
**Why it matters:** the moment a host implements a day/night cycle or weather (the whole point of `EnvironmentState`), god rays stay warm-white and the fog stays powder-blue at sunset/night, visibly detaching from the analytic fog (`fog.hlsli:32-51`) and sky haze that do track the environment. The `nearPlane`/`farPlane` uniforms are also dead weight — the shader never reads them.
**Recommendation:** plumb `env_.sun_color * gfx.sun_intensity`, `env_.fog_color_horizon`, and height-falloff-derived parameters into `VolumetricFog::render` (it already receives `light_dir`); delete or use the dead uniforms.

---

## Findings — MEDIUM

### M1. God rays only exist within ~50–60 units of the camera (cascade 0 only) and never over sky pixels
**Evidence:** the fog pass receives only `cascades()[0]` (`volumetric_fog.cpp:114-119`) and `sampleShadow` returns 1.0 (fully lit) outside that cascade's ortho window (`volumetric_fog.frag.hlsl:58-74`). With defaults (near 0.1, shadow far 1200, λ=0.85, 3 cascades — `shadow_map.hpp:86`, `scene_renderer.cpp:553-555`, `graphics_settings.hpp:33`), cascade 0's split is ≈62 units. Sky pixels early-out entirely (`volumetric_fog.frag.hlsl:90`).
**Why it matters:** light shafts through distant tree lines / mountains — the marquee use of the feature in outdoor scenes — silently degrade to plain unshadowed inscatter beyond ~60 m, and shafts cannot cross the horizon silhouette. Users tuning `god_rays` will see it "not work."
**Recommendation:** pass the full 4-cascade matrix array + splits (the `ShadowDataUniforms` layout already exists, `gpu_uniforms.hpp:217-230`) and select cascade per march step from the accumulated ray distance; march sky pixels to `maxDist` instead of returning early.

### M2. No TAA / motion vectors; FXAA is the only AA over alpha-tested foliage
**Evidence:** `aa_mode` is 0/1 = off/FXAA (`graphics_settings.hpp:36`); FXAA runs on the LDR composite (`fxaa.frag.hlsl`, wired at `scene_renderer.cpp:735-754`). Grass is opaque geometry with a VS LOD collapse, trees use alpha-tested cutouts (H2's discard path), and specular AA exists but only for shading (`pbr.hlsli:120-126`). No velocity buffer exists anywhere; MSAA is structurally present but fixed at 1× (`gpu_pipeline.hpp:56`).
**Why it matters:** for exactly this content class — sub-pixel grass blades, cutout canopies, high-frequency terrain detail in motion — FXAA does nearly nothing (it is a static-edge filter with no temporal accumulation). This is the single largest remaining image-quality gap, and the design notes already rank it first among AA work (`docs/engine_design_notes.md:122-124`).
**Recommendation:** TAA with camera+object motion vectors (skinned palettes of the previous frame are already CPU-resident per instance, so skinned velocity is cheap). This is also the stated first justification for wiring `render_graph/` (history buffers). Ship with a neighborhood-clamp + luminance-weighted resolve; keep FXAA as fallback.

### M3. No IBL — ambient specular is a two-color hemisphere lerp through a mobile analytic fit
**Evidence:** `ambientPBR` uses `lerp(groundColor, skyColor, N.y)` for irradiance and Karis `EnvBRDFApprox` against the same hemisphere lerp along R for specular (`pbr.hlsli:90-115`). No cubemap, no BRDF LUT, no reflection probes exist in the tree.
**Why it matters:** metals and low-roughness surfaces (armor, weapons, water, wet stone — MMO material staples) have nothing coherent to reflect; they read as flat tinted gradients. Diffuse ambient is acceptable outdoors but has no sky-occlusion or directionality beyond up/down.
**Recommendation:** the design notes already spec the exact upgrade (`docs/engine_design_notes.md:112-119`): prefilter the procedural sky into a small cubemap (it is analytic — re-render on environment change, not per frame), split-sum BRDF LUT, swap into `ambientPBR`'s two terms. Moderate effort, large material payoff.

### M4. Point/spot lights cast no shadows
**Evidence:** `accumulateClusterLights` applies distance/cone falloff only — no shadow term of any kind (`cluster_lighting.hlsli:74-108`); the only shadow-caster is the directional sun.
**Why it matters:** night/dungeon/interior scenes lit by the cluster path leak light through walls and floors; with up to 256 lights this becomes the dominant realism limiter the moment content moves indoors. Acknowledged as future work in `docs/engine_design_notes.md:120-121` — recorded here for rank ordering, not as an oversight.
**Recommendation:** per-light shadow atlas (single big D32 atlas + per-light viewport, same pattern as the CSM atlas) for the N highest-priority lights; cheap PCF only.

### M5. Half-res AO and fog are upsampled with plain bilinear sampling — silhouette halos
**Evidence:** AO runs at half resolution and "Upsample happens for free in the composite pass via bilinear sampling" (`ambient_occlusion.cpp:77-80`); composite samples both AO and fog textures with a linear clamp sampler (`ambient_occlusion.cpp:457-466`, `composite.frag.hlsl:60-68`). Volumetric fog is also half-res (`volumetric_fog.cpp:21-22`).
**Why it matters:** bilinear upsampling mixes foreground and background AO/fog values across depth discontinuities: dark AO fringes around character silhouettes against sky, and god-ray/fog bleeding across geometry edges. The bilateral blur helps within the half-res image but cannot fix the 2× upsample itself.
**Recommendation:** depth-aware (joint bilateral) upsample in the composite pass — 4 taps of half-res AO/fog weighted by full-vs-half-res depth similarity; the linearization constants already exist in `BlurUniforms`.

### M6. Opaque pass has no depth prepass and no front-to-back ordering
**Evidence:** draw order is terrain → grass → instanced models → non-instanced models → skinned → skybox → particles (`scene_renderer.cpp:915-1026`); instanced batches iterate in model-handle order (`scene_renderer.cpp:1326-1334`), not depth order.
**Why it matters:** the fragment shaders here are expensive (triplanar terrain, PCSS, cluster loops), and in dense scenes overdraw multiplies that cost. Grass drawn second, in particular, shades blades later occluded by props/characters. Not a correctness issue; a scalability one.
**Recommendation:** cheapest first: sort instanced bins and non-instanced command lists by camera distance (coarse bucket sort is enough). The full fix is the depth prepass shared with H3, after which opaque shading is nearly overdraw-free via early-Z.

---

## Findings — LOW

### L1. "GTAO" is not GTAO
**Evidence:** `gtao.frag.hlsl:87-121` accumulates cosine-weighted, distance-attenuated point samples along rotated screen directions — there is no per-direction horizon search or arc integration (the defining feature of GTAO). It is closer to an HBAO-lite variant.
**Why it matters:** expectation mismatch for anyone selecting "GTAO" for ground-truth-matched occlusion; quality difference vs. the SSAO mode is modest for the cost.
**Recommendation:** either implement true horizon-based arc integration (2 directions × horizon walk, per Jimenez et al.) or rename the mode; the uniform/kernel plumbing (`ambient_occlusion.cpp:52-63`) already supports both.

### L2. Bloom threshold operates pre-exposure
**Evidence:** threshold is applied to raw scene HDR in the first downsample (`bloom_downsample.frag.hlsl:37-45,90`), while exposure multiplies the summed result later (`composite.frag.hlsl:72-76`); default threshold 1.1 (`graphics_settings.hpp:55`).
**Why it matters:** changing `exposure` re-brightens bloom but does not change *what* blooms, so an exposure-dropped night scene still blooms surfaces that will tonemap to mid-grey, and a high-exposure scene misses highlights that end up clipping.
**Recommendation:** pass `exposure` into the bloom prefilter and threshold `color * exposure` (then don't double-apply at composite), or document the threshold as pre-exposure scene units.

### L3. Volumetric fog has a hardcoded 100-unit near dead zone and 3000-unit cap
**Evidence:** `startDist = 100.0`, `maxDist = min(rayLength, 3000.0)` (`volumetric_fog.frag.hlsl:104-105`); any pixel closer than 100 units gets zero volumetric fog (`:106`).
**Why it matters:** ground fog visibly cuts out in a radius around the camera and pops as geometry crosses 100 units; the values are invisible to hosts (not uniforms).
**Recommendation:** move both to `VolumetricFogUniforms` (two padding slots are free, `gpu_uniforms.hpp:351-352`) and default the near skip much lower now that the march is dithered.

### L4. No HDR display output path
**Evidence:** swapchain composition hardcoded to `SDL_GPU_SWAPCHAINCOMPOSITION_SDR` (`gpu_device.cpp:315`); composite always encodes gamma-2.2 sRGB (`composite.frag.hlsl:99`).
**Why it matters:** the pipeline is already scene-referred HDR (RGBA16F end to end), so an HDR10/scRGB output path is mostly a swapchain-composition query plus an alternate OETF in composite. Low urgency, cheap differentiation later.
**Recommendation:** when desired: query `SDL_WindowSupportsGPUSwapchainComposition` for `HDR_EXTENDED_LINEAR`/`HDR10_ST2084`, branch the composite encode (PQ or pass-through linear), and skip FXAA's gamma assumption check.

### L5. UI/text/billboard blend in gamma space
**Evidence:** UI pipelines alpha-blend directly into the sRGB-encoded swapchain (`pipeline_registry.cpp:481,518,541`; composite has already gamma-encoded by then).
**Why it matters:** antialiased font edges and translucent panels blend slightly dark/fringed — the standard, near-universally-accepted UI compromise. Recorded for completeness only.
**Recommendation:** accept (industry norm), or composite UI in linear before tonemap at real cost. No action suggested.

### L6. `gtao_direction_kernel` latches its first `num_directions` forever
**Evidence:** `ambient_occlusion.cpp:52-63` — the kernel is a function-local `static` initialized from the first call's `num_directions`; `GTAOUniforms::numDirections` (`gpu_uniforms.hpp:253`) is currently a constant 6, so this is latent, not live.
**Why it matters:** the day `numDirections` becomes a quality setting, changing it silently keeps the old kernel spacing.
**Recommendation:** drop the `static` (24-entry build is trivial) or key it on the requested count.

---

## Missing-feature ranking (value ÷ effort, for this engine's content)

Ranked for outdoor, long-view-distance, foliage- and character-heavy MMO scenes:

1. **Reversed-Z** — small, localized effort; eliminates the depth-precision class of artifacts at exactly this engine's draw distances (H1).
2. **TAA + motion vectors** — largest visible quality jump given alpha-tested foliage; also the designated reason to wire `render_graph/` (M2).
3. **IBL (prefiltered sky cubemap + BRDF LUT)** — biggest material-quality win; upgrade point already isolated in `ambientPBR` (M3).
4. **Depth prepass + ambient-only AO** — fixes H3 correctly, cuts opaque overdraw (M6), and is prerequisite plumbing for SSR/particle-soft-depth later.
5. **Transparent (sorted alpha-blend) pass + glTF alpha-mode compliance** — correctness debt blocking common asset types (H2).
6. **Per-light shadow atlas for point/spot** — the realism limiter for night/interior content; clustered plumbing already in place (M4).
7. **GPU-driven culling / indirect draws** — valuable at true MMO instance counts, but current CPU cull + per-model instancing is not yet the measured bottleneck; per the project's own "infrastructure without load" principle, defer until profiling shows it.
8. **Decals** — spell telegraphs, footprints, scorch marks are MMO staples; moderate effort once a depth prepass exists (projected decals on depth).
9. **HDR display output** — cheap because the pipeline is already HDR-internal, but presentation polish only (L4).
10. **SSR / motion blur / DoF / terrain virtual texturing** — poor value here: mostly-rough outdoor materials (SSR), MMO camera style (MB/DoF), and a 4-layer anti-tiled splat that already covers the terrain need (VT). Correctly absent; revisit SSR only alongside a water system.

---

## Top-10 prioritized actions

1. Switch to reversed-Z (flip compare/clear/projection; audit `ssao/gtao/blur_ao/volumetric_fog` depth reads and skybox z-trick) — H1.
2. Honor glTF `alphaMode`/`alphaCutoff`/`doubleSided`: gate the hardcoded discard, add cutoff uniform, add no-cull variant — H2 (first half).
3. Plumb `EnvironmentState` into `VolumetricFog::render`; delete dead near/far uniforms — H4.
4. Feed all cascades to the god-ray march (or at minimum cascade selection by ray distance) and march sky pixels — M1.
5. Add a depth prepass; move AO application into the forward ambient term; delete the composite luminance hack — H3 + M6.
6. Depth-aware upsample for half-res AO and fog in composite — M5.
7. Implement TAA + motion vectors (wire `render_graph/` for history) — M2.
8. Prefilter the procedural sky into an IBL cubemap + BRDF LUT; upgrade `ambientPBR` — M3.
9. Add a sorted transparent queue for `AlphaMode::Blend` materials — H2 (second half).
10. Per-light shadow atlas for the top-N cluster lights — M4.

*Every file:line above was read directly during this audit; line numbers reference the working tree as of 2026-07-03.*

---

## Verification

Adversarial re-check (2026-07-03): every cited file/line was re-opened and each claim actively tested against the code and against `docs/architecture.md` / `docs/engine_design_notes.md` for intentional-design explanations. No finding was refuted. One line-range correction applied (H2: loader alpha parsing is `model_loader.cpp:275-281`).

| Finding | Verdict | Note |
|---|---|---|
| H1. No reversed-Z over MMO-scale view distances | **CONFIRMED** | `LESS_OR_EQUAL` default (`gpu_pipeline.hpp:44`), clear-to-1.0 (`scene_renderer.cpp:417`, `ambient_occlusion.cpp:245`), near clamp 0.1 (`scene_renderer.cpp:553`), draw distance to 32000 (`graphics_settings.hpp:26,64-67`) all verified. |
| H2. glTF alphaMode/alphaCutoff/doubleSided parsed but ignored; no transparent pass | **CONFIRMED** | Grep over `src/engine` finds zero consumers outside the loader; hardcoded `a < 0.5` discard at `model.frag.hlsl:124` / `model_instanced.frag.hlsl:120`; all model pipelines `.opaque()` with default back-face cull; no blend queue in `render_3d_scene`. Loader comment even says "pipeline selection is the renderer's job" — the renderer never does it. |
| H3. AO multiplies final lit color, patched with luminance hack | **CONFIRMED** | `composite.frag.hlsl:53-63` matches verbatim, including the self-acknowledging comment; material AO separately occludes ambient in `model.frag.hlsl:139-155`, so screen-space AO double-darkens ambient. In-code comment says "no depth prepass **yet**" — a known gap, not a documented decision. |
| H4. Volumetric fog ignores `EnvironmentState` | **CONFIRMED** | Hardcoded density/lightColor/fogColor/height/falloff at `volumetric_fog.cpp:122-130`; `nearPlane`/`farPlane` pushed but never read in `volumetric_fog.frag.hlsl`; `env_` single-source-of-truth comment at `scene_renderer.cpp:484-486`. |
| M1. God rays cascade-0-only (~60 units) and skip sky pixels | **CONFIRMED** | `cascades()[0]` only (`volumetric_fog.cpp:114-119`); out-of-window returns 1.0 (`volumetric_fog.frag.hlsl:64-66`); `depth >= 1.0` early-out (`:90`). Split math independently re-derived: λ=0.85 (`shadow_map.hpp:86`), near 0.1, far 1200, 3 cascades → cascade-0 split ≈ 62 units. |
| M2. No TAA/motion vectors; FXAA-only | **CONFIRMED** | `aa_mode` 0/1 only; `SDL_GPU_SAMPLECOUNT_1` fixed; grep confirms no velocity/TAA code anywhere; design notes call TAA "the biggest remaining AA-quality gap" (future work, not a decision to omit). |
| M3. No IBL; two-color hemisphere + Karis EnvBRDFApprox | **CONFIRMED** | `pbr.hlsli:90-115` matches exactly; design notes list IBL as the #1 forward-looking item with `ambientPBR` as the upgrade point. |
| M4. Point/spot lights cast no shadows | **CONFIRMED** | `accumulateClusterLights` (`cluster_lighting.hlsli:74-109`) has distance/cone falloff only, no shadow term. Acknowledged future work in design notes, as the finding itself states. |
| M5. Half-res AO/fog upsampled with plain bilinear | **CONFIRMED** | Half-res + "free" bilinear upsample comment (`ambient_occlusion.cpp:77-80`); composite binds AO/bloom/fog all with `linear_clamp_sampler_` (`ambient_occlusion.cpp:457-466`); fog is half-res (`volumetric_fog.cpp:21-22`). |
| M6. No depth prepass, no front-to-back opaque sorting | **CONFIRMED** | Draw order terrain→grass→instanced→non-instanced→skinned→skybox→particles verified at `scene_renderer.cpp:915-1026`; instances packed "in ascending handle order" (`:1326-1332`); no sorting anywhere. |
| L1. "GTAO" is not GTAO | **CONFIRMED** | `gtao.frag.hlsl:87-121` is cosine-weighted, distance-attenuated point sampling along rotated directions — no horizon search, no arc integration. |
| L2. Bloom threshold pre-exposure | **CONFIRMED** | Threshold in prefilter on raw scene HDR (`bloom_downsample.frag.hlsl:37-45,90`); exposure applied after bloom add at `composite.frag.hlsl:73,76`. |
| L3. Hardcoded 100-unit fog dead zone / 3000-unit cap | **CONFIRMED** | `startDist = 100.0` / `maxDist = min(rayLength, 3000.0)` at `volumetric_fog.frag.hlsl:104-106`; `_padding0`/`_padding1` free in `VolumetricFogUniforms` verified. |
| L4. No HDR display output path | **CONFIRMED** | `SDL_GPU_SWAPCHAINCOMPOSITION_SDR` hardcoded in `GPUDevice::set_swapchain_parameters` (`gpu_device.cpp:315`); composite always sRGB-encodes. |
| L6. `gtao_direction_kernel` latches first `num_directions` | **CONFIRMED** | Function-local `static` capturing the first call's `num_directions` (`ambient_occlusion.cpp:52-63`); latent because `numDirections` is currently a constant 6, exactly as stated. |

(L5, UI gamma-space blending, was recorded for completeness in the original audit and was not part of this verification pass.)
