# HLSL Shader Sources

HLSL shaders compiled to SPIR-V at build time. `CMakeLists.txt` in this
directory fetches SDL_shadercross (bundling DXC + SPIRV-Tools), globs
`*.hlsl`, and emits one `shadercross` CLI invocation per shader; stage and
entry point are derived from the filename. Compiled `.spv` files land in the
`spv/` subdirectory of this directory's build dir —
`build/src/bracken/shaders/spv/` in a standalone build; the authoritative
path is the `BRACKEN_SHADER_OUTPUT_DIR` INTERNAL cache variable. Deleting or
renaming a `.hlsl` prunes its orphaned `.spv` at the next configure. At
runtime `ShaderManager` / `PipelineRegistry` load from the `shaders/`
directory next to the executable; `bracken_copy_shaders(<target>)` stages it
there, removing and re-copying the whole directory each build so stale files
never ship.

Every `.hlsli` is a declared dependency of every shader, so editing an
include recompiles everything — an incremental `cmake --build build` is
always enough. With `BRACKEN_HOT_RELOAD=1`, the `PipelineRegistry` watches the
runtime `shaders/` directory and rebuilds all pipelines when any `.spv`
changes.

## Naming convention

- `{name}.vert.hlsl` — vertex shader, entry point `VSMain`
- `{name}.frag.hlsl` — fragment shader, entry point `PSMain`
- `{name}.comp.hlsl` — compute shader, entry point `CSMain`
- `{name}.hlsli` — shared include (not compiled standalone)

## Pipeline → shader map

Which shaders each `PipelineType` (`gpu/pipeline_registry.cpp`) actually
loads. Note the shared shaders: the skinned pipeline reuses `model.frag`,
and every post-process pass reuses `fullscreen.vert`.

| Pipeline | Vertex | Fragment |
|---|---|---|
| Model (+ DoubleSided/Blend variants) | `model.vert` | `model.frag` |
| SkinnedModel (+ DoubleSided/Blend variants) | `skinned_model.vert` | `model.frag` |
| InstancedModel (+ DoubleSided variant) | `model_instanced.vert` | `model_instanced.frag` |
| Terrain | `terrain.vert` | `terrain.frag` |
| Grass | `grass.vert` | `grass.frag` |
| Skybox | `skybox.vert` | `skybox.frag` |
| Water | `water.vert` | `water.frag` |
| Particle | `particle.vert` | `particle.frag` |
| Grid | `grid.vert` | `grid.frag` |
| UI | `ui.vert` | `ui.frag` |
| Text | `text.vert` | `text.frag` |
| ShadowModel | `shadow_depth.vert` | `shadow_depth.frag` |
| InstancedShadowModel | `shadow_depth_instanced.vert` | `shadow_depth.frag` |
| ShadowSkinnedModel | `shadow_skinned.vert` | `shadow_depth_opaque.frag` |
| ShadowTerrain | `shadow_terrain.vert` | `shadow_depth_opaque.frag` |
| LocalShadowClear | `fullscreen.vert` | `shadow_depth_opaque.frag` |
| DepthPrepassModel (+ DoubleSided) | `prepass_model.vert` | `prepass_velocity.frag` |
| DepthPrepassSkinnedModel (+ DoubleSided) | `prepass_skinned.vert` | `prepass_velocity.frag` |
| DepthPrepassInstancedModel (+ DoubleSided) | `prepass_instanced.vert` | `prepass_velocity.frag` |
| DepthPrepassTerrain | `prepass_terrain.vert` | `prepass_velocity.frag` |
| DepthPrepassGrass | `prepass_grass.vert` | `prepass_velocity.frag` |
| DepthPrepassModelMasked | `prepass_model_masked.vert` | `prepass_velocity_masked.frag` |
| DepthPrepassInstancedModelMasked | `prepass_instanced_masked.vert` | `prepass_velocity_masked.frag` |
| GTAO | `fullscreen.vert` | `gtao.frag` |
| GTAOTemporal | `fullscreen.vert` | `gtao_temporal.frag` |
| GTAOUpsample | `fullscreen.vert` | `gtao_upsample.frag` |
| BlurAO | `fullscreen.vert` | `blur_ao.frag` |
| BloomDownsample | `fullscreen.vert` | `bloom_downsample.frag` |
| BloomUpsample | `fullscreen.vert` | `bloom_upsample.frag` |
| VolumetricFog | `fullscreen.vert` | `volumetric_fog.frag` |
| Composite | `fullscreen.vert` | `composite.frag` |
| Fxaa | `fullscreen.vert` | `fxaa.frag` |
| TaaResolve | `fullscreen.vert` | `taa_resolve.frag` |
| TaaSharpen | `fullscreen.vert` | `taa_sharpen.frag` |
| SkyCapture | `fullscreen.vert` | `sky_capture.frag` |
| IblSpecularPrefilter | `fullscreen.vert` | `ibl_prefilter.frag` |
| IblIrradiance | `fullscreen.vert` | `ibl_irradiance.frag` |
| IblBrdfLut | `fullscreen.vert` | `ibl_brdf_lut.frag` |

Compute pipelines (`ComputePipelineType`, same registry/hot-reload
lifecycle; one `.comp.hlsl` is the whole pipeline — there is no separate
vertex/fragment pair):

| Compute pipeline | Shader | Threadgroup |
|---|---|---|
| ExposureHistogram | `exposure_histogram.comp` | 16×16×1 |
| ExposureAdapt | `exposure_adapt.comp` | 256×1×1 |
| WaterSim (compute) | — | `water_sim.comp` |

The model-family material variants (`ModelDoubleSided`, `ModelBlend`,
`ModelBlendDoubleSided`, `SkinnedModel*`, `InstancedModelDoubleSided`) load
the same shader pair as their base pipeline and differ only in fixed-function
state: `doubleSided` disables back-face culling, `BLEND` enables alpha
blending with depth writes off. The MASK alpha test is driven by the
`alphaCutoff` field of the lighting cbuffer (0 = no discard, so OPAQUE/BLEND
draws skip it); MASK draws scale alpha by the sampled mip level first
(`1 + mip * 0.35`) so minified foliage keeps its coverage instead of eroding
into a translucent wash. The `*DepthEqual` variants (Model/SkinnedModel/
InstancedModel × cull, Terrain, Grass) are likewise fixed-function-only:
depth compare `EQUAL` with depth writes off, bound for OPAQUE draws after
the depth prepass laid down this frame's depth.

## Depth prepass + velocity

The `DepthPrepass*` pipelines render every glTF alphaMode OPAQUE draw
(terrain, static/instanced/skinned models, grass) position-only into the
scene depth target plus an RG16F velocity target before the forward pass.
The shared `prepass_velocity.frag` writes **UV-space motion vectors,
current minus previous** (`prev_uv = uv - velocity` reprojects; CPU
reference: `src/bracken/scene/velocity_math.hpp`), computed from UNJITTERED
view-projection matrices carried alongside the (possibly TAA-jittered) draw
projection. Skinned prepass draws bind TWO palette buffers (vertex set 0,
bindings 0/1: current + previous frame); `prevPaletteOffset == 0xFFFFFFFF`
means "no palette history — reuse the current skin matrix".

Because the forward pass depth-tests `EQUAL` against the prepass depth, the
position math in each prepass vertex shader replicates its forward
counterpart operation-for-operation and **both sides declare
`precise float4 position : SV_Position`** (disables FMA
contraction/reassociation on the position dependency chain). Grass goes
further: the entire blade placement/wind math is shared via
`grass_blade.hlsli`, and `grass.vert` + `prepass_grass.vert` must keep their
`GrassUniforms` cbuffers byte-identical (pinned by static_asserts in
`render/grass_renderer.hpp`). When editing any position math, edit both
shaders in the same change or opaque geometry will drop pixels.

## Depth convention

The scene renders **reversed-Z** (glm `RH_ZO` projection with near/far
swapped): the near plane is device depth **1.0**, the far plane / cleared
background is **0.0**, and scene pipelines depth-test `GREATER_OR_EQUAL`. The
skybox vertex shader emits z = 0.0 so it passes only where nothing was drawn.
Depth-reading shaders (`gtao`, `gtao_temporal`, `blur_ao`, `volumetric_fog`)
test `depth <= 0.0` for sky; linearization/reconstruction goes through the (inverse)
projection matrix and needs no convention-specific math.
**Exception:** the shadow passes stay standard-Z — the CSM atlas is cleared
to 1.0 with `LESS_OR_EQUAL` compares (pipeline + `SampleCmpLevelZero`
sampler) and `orthoRH_ZO` light projections (orthographic depth is linear
and gains nothing from reversal), and the local point/spot shadow tiles use
standard-Z `perspectiveRH_ZO` (near→0, far = light radius→1) so they share
the same caster pipelines, clear value, and comparison-sampler convention.

## Shader notes

**Scene** — `model.*`: static models, PBR + clustered lights + CSM + fog.
Every scene fragment shader (model/instanced/terrain/grass; skinned reuses
`model.frag`) samples the full-res screen-space GTAO texture (UV from
`SV_Position` × a texel-size uniform) and multiplies it into the AMBIENT term
only, min-combined with the material ORM AO — direct sun light and speculars
are never attenuated by screen AO. The slot is bound to a 1×1 white dummy
when AO is off (and always for BLEND draws). The LAST THREE sampler slots of
every scene fragment shader are the sky IBL chain (irradiance cube,
GGX-prefiltered specular cube, split-sum BRDF LUT — `render/ibl.hpp`);
when the `iblEnabled` uniform flag is on, `ambientIBL()` (pbr.hlsli) replaces
the hemisphere ambient with sky-derived diffuse irradiance + prefiltered
specular (roughness → mip, `IBL_SPECULAR_MIPS` must match
`render::IBL::SPECULAR_MIPS`), multiplied by the same combined AO and
shadow-driven ambient lift. When off, cube-typed dummies keep the slots
valid and the shaders branch to the hemisphere `ambientPBR` fallback.
The very LAST sampler slot of every scene fragment shader is the LOCAL
(point/spot) shadow atlas + `LESS_OR_EQUAL` comparison sampler, declared by
`cluster_lighting.hlsli` at `CLUSTER_SSBO_BINDING_BASE - 1` (fragment sampler
counts: model/instanced 11, terrain 9, grass 7). Cluster lights whose
`shadow_slot >= 0` sample it inside `accumulateClusterLights` — point lights
pick one of their 6 face tiles by the dominant axis of light→fragment
(`localShadowCubeFace`; CPU mirror `lighting::local_shadow_cube_face`), then
4 comparison taps (each a hardware 2×2 PCF) with world-space normal-offset +
slope bias and a smooth fade at the light radius. Per-tile matrices/rects
ride a 4th fragment storage buffer (`StructuredBuffer<LocalShadowSlot>`,
96 B/entry — CPU mirror `lighting::LocalShadowSlotGPU`), so
`SDL_BindGPUFragmentStorageBuffers(pass, 0, bufs, 4)` now binds cluster
light data / offsets / indices / local-shadow slots. When local shadows are
unavailable the slot binds any D32 texture (the CSM atlas) and the SSBO slot
binds the light-data buffer — never sampled while every `shadow_slot` is -1.
`skinned_model.vert`: skeletal skinning; bone matrices come from a per-frame
`StructuredBuffer<float4x4>` of tightly-packed palettes (vertex set 0,
binding 0) indexed by a `paletteOffset` uniform — no compile-time bone count
(the CPU caps palettes at `MAX_BONES = 128`, `animation/animation_types.hpp`).
`model_instanced.*`: per-instance transforms from a storage buffer (rocks,
trees). `terrain.*`: splatmap texturing, height-based lighting. `grass.*`:
instanced blades, wind, alpha test. `skybox.*`: procedural sky on a fullscreen
triangle; the entire radiance function (gradient / domain-warped FBM cloud
deck / sun disc / haze / stars) lives in `sky_common.hlsli` and is SHARED
with `sky_capture.frag` so the IBL prefilter chain can never drift from the
visible sky — edit the include, never fork the math. Cloud shaping
(coverage/density/scale + haze strength) comes from `EnvironmentState` via
the sky uniforms; the scroll `time` arrives pre-scaled by `cloud_speed`.
The sky's horizon haze calls `hazeSunColor` from `fog.hlsli` — the SAME
function `applyAtmosphericFog` uses for surface fog (warm Mie forward lobe
that widens at low sun + day dimming), which is the sky↔fog horizon
agreement contract. Cloud noise uses a PCG2D integer lattice hash — do not
swap in a float `frac`-product hash (it quantizes at upper-octave
coordinates and draws straight creases across the clouds).

**Shadows** — depth-only cascade passes into the 2×2 shadow atlas.
`shadow_depth_opaque.frag` is the no-alpha-test variant used by skinned +
terrain casters. The LOCAL (point/spot) shadow passes reuse the exact same
caster shaders/pipelines (`shadow_depth.*`, `shadow_skinned.vert`) with
per-tile perspective view-projections — **standard-Z** like the cascades
(near→0, far = light radius→1, `LESS_OR_EQUAL`). `LocalShadowClear` is a
fixed-function trick, not a new shader: the fullscreen triangle drawn with
the viewport depth range pinned to [1,1] and an `ALWAYS` depth test clears
one atlas tile to far-plane depth (cached tiles persist, so the atlas pass
LOADs instead of clearing).

**IBL prefilter** — all use `fullscreen.vert` into one cubemap (face, mip)
per pass, driven by `render::IBL` (amortized over frames when the sky
changes; see render/ibl.hpp): `sky_capture.frag` (the shared sky radiance
into the 64px RGBA16F capture cube; mips via
`SDL_GenerateMipmapsForGPUTexture`), `ibl_prefilter.frag` (GGX importance
sampling of the capture cube, one pass per specular mip, roughness =
mip/(mips−1), mip-biased source sampling), `ibl_irradiance.frag`
(cosine-hemisphere convolution into the 16px irradiance cube; stores
(1/π)·∫L·cosθ so a uniform sky convolves to itself), `ibl_brdf_lut.frag`
(Karis split-sum scale/bias LUT, RG16F, rendered once at init). The face
basis (direction ↔ face texel) is supplied by the CPU —
`render::ibl_face_direction` is the tested mirror of the cube-lookup
convention.

**Post-processing** — all use `fullscreen.vert` (fullscreen triangle, no
vertex input). The AO chain runs BEFORE the forward pass, from the prepass
depth: `gtao` (true horizon-based GTAO, Jimenez et al. 2016 — per-slice
horizon search with cosine-weighted arc integration, 3 slices × 4 steps per
side, per-pixel IGN + per-frame golden-ratio temporal jitter, edge-aware
depth-reconstructed normals, thin-occluder heuristic; CPU reference for the
arc integral: `render/gtao.hpp`), `gtao_temporal` (velocity-reprojected
accumulation into an RG16F history of AO + linear depth, relative-depth
disocclusion rejection), `blur_ao` (bilateral, ×2 separable),
`gtao_upsample` (joint-bilateral upsample to the full-res R8 texture the
forward pass samples; applies `ao_strength`). The rest sample the HDR
offscreen scene target: `bloom_downsample` (13-tap, threshold in
post-exposure luminance) / `bloom_upsample` (9-tap tent, additive),
`volumetric_fog` (ray march with god rays from the shadow atlas — cascade
picked per sample by camera distance; sky pixels march too), `composite`
(depth-aware joint-bilateral upsample of half-res fog, scene + bloom + fog,
exposure — the manual bias times, when the `autoExposureEnabled` flag is
set, exp2 of the adapted EV read from the exposure-state SSBO at fragment
storage-buffer slot 0 — then flag-gated white balance (linear LMS gains,
pre-tonemap; CPU side: `render/color_grading.hpp`), tonemap, flag-gated
ASC-CDL `pow(c*slope+offset, power)` (post-tonemap), contrast/saturation +
gamma → LDR; the grading gates keep all-neutral parameters pixel-identical
to the ungraded path; AO is NOT composited here), `fxaa`
(FXAA 3.11 on the tonemapped LDR image), `taa_resolve` (`aa_mode == TAA`, pre-bloom: temporal accumulation of
the jittered HDR scene into an RGBA16F history — 3×3 closest-depth velocity
dilation on the prepass velocity target, depth reprojection for
velocity-less pixels with far-plane pixels reprojected as w=0 directions,
YCoCg variance clipping, 5-tap Catmull-Rom history sampling, Karis
luminance-weighted blend; CPU reference for the depth-reprojection fallback:
`render/taa.hpp`) + `taa_sharpen` (mild unsharp mask writing the resolved
history back over the offscreen scene color — history itself stays
unsharpened so the kernel never compounds).

**Auto exposure (compute)** — `exposure_histogram.comp` builds a 256-bin
log2-luminance histogram of the TAA-resolved (pre-bloom) HDR scene color:
each 16×16 workgroup accumulates its tile into `groupshared` bins and merges
the non-zero ones into the global histogram SSBO with one `InterlockedAdd`
per bin. `exposure_adapt.comp` (single 256-thread workgroup) loads and
ZEROES the bins (that is the histogram's per-frame reset), then thread 0
serially computes the percentile-clipped average log2 luminance, the target
EV, and one rate-limited adaptation step into the persistent
`exposureState` float4 — serial on purpose: it stays bit-identical to the
CPU reference in `render/auto_exposure.hpp`, which the tests pin. Bin 0
holds "black" pixels (below EV −10): they count toward the percentile total
but contribute no luminance.

**Effects/UI** — `particle.*`: unlit additive HDR particles, instanced —
per-particle transform + color come from a per-frame
`StructuredBuffer<ParticleInstance>` (vertex set 0, binding 0) indexed by an
`instanceOffset` uniform, one draw per (emitter, mesh) batch. `ui`/`text`:
2D pipelines targeting the swapchain (3D billboards are projected on the CPU
and drawn as UI rects — there is no billboard pipeline). `grid.*`: debug grid
overlay.

**Includes** — `pbr.hlsli` (GGX/Smith/Schlick/Burley BRDF, `evaluatePBR`,
hemisphere `ambientPBR`, sky-IBL `ambientIBL`, specular AA),
`sky_common.hlsli` (the procedural sky radiance function shared by the
skybox and sky-capture passes; includes `fog.hlsli` for the shared
`hazeSunColor`), `lighting.hlsli` (luminance,
linear→sRGB transfer, interleaved gradient noise, the ACES tonemap curves —
Narkowicz + Hill fit, consumed by `composite.frag` — and the shared
`viewPosFromDepth`/`worldPosFromDepth` depth-reconstruction helpers used by
`gtao`/`volumetric_fog`), `cluster_lighting.hlsli` + `clusters.hlsli`
(cluster grid sampling + the local point/spot shadow bindings and sampling —
define `CLUSTER_SSBO_BINDING_BASE` to the including shader's fragment sampler
count first), `shadow_common.hlsli` (cascade selection, PCF/PCSS),
`fog.hlsli` (distance-fog factors + `hazeSunColor`, the sun-scatter /
day-dimmed haze color shared by surface fog AND the sky's horizon haze —
they must stay one function so fog melts into the sky at the horizon).

## SDL3 GPU resource binding order

Bindings must follow SDL3 GPU's set layout, expressed with
`[[vk::binding(slot, set)]]`:

- **Vertex**: set 0 = sampled textures, then storage textures, then storage
  buffers; set 1 = uniform buffers.
- **Fragment**: set 2 = sampled textures, then storage textures, then storage
  buffers; set 3 = uniform buffers.
- **Compute** — note the read-only/read-write split, which graphics stages
  don't have:
  - set 0 / `(t[n]+s[n], space0)` = sampled textures, then READ-ONLY storage
    textures, then READ-ONLY storage buffers
    (`StructuredBuffer`/`Texture2D`);
  - set 1 / `(u[n], space1)` = READ-WRITE storage textures, then READ-WRITE
    storage buffers (`RWStructuredBuffer`/`RWTexture2D`) — these are bound
    up front by `SDL_BeginGPUComputePass` (see `gpu::ComputePass`), not
    after the pipeline;
  - set 2 / `(b[n], space2)` = uniform buffers
    (`SDL_PushGPUComputeUniformData`).

All vertex semantics are `TEXCOORD0..N` (SDL3 GPU's D3D12 backend
assumption).

At creation every shader's declared resource counts are cross-checked
against SPIR-V reflection (graphics: `SDL_ShaderCross_ReflectGraphicsSPIRV`;
compute: `SDL_ShaderCross_ReflectComputeSPIRV`, which also supplies the
`[numthreads]` threadgroup baked into the pipeline) — a mismatch logs
`declared resource counts do not match SPIR-V` and the reflected counts win.
Compute creation fails outright if reflection fails (there is no
caller-side fallback for the threadgroup size).

## CPU↔GPU uniform contract

Every cbuffer here has an `alignas(16)` CPU mirror in
`src/bracken/gpu/gpu_uniforms.hpp` annotated with its `(set, binding)` —
compute cbuffers included (their set is always 2). The
layouts must match **byte-for-byte** — when adding or reordering cbuffer
fields, update both sides in the same change and mind HLSL packing rules
(16-byte registers; a `float3` followed by `float` packs, two `float3`s do
not).
