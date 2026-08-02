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

**The map lives in the code, in one place:
[`gpu/pipeline_catalog.cpp`](../gpu/pipeline_catalog.cpp).** Each
`PipelineType` and `ComputePipelineType` has exactly one `PipelineDesc` /
`ComputePipelineDesc` row naming its shader(s), the resource counts they
declare, its vertex layout and its fixed-function state; a missing row is a
compile error. This file used to carry a second, hand-maintained copy of that
table, which is exactly the kind of thing that drifts — read the rows instead.

Worth knowing without opening the file: shaders are shared. The skinned
pipeline reuses `model.frag`, all eight depth-prepass pipelines reuse
`prepass_velocity.frag` (or `prepass_velocity_masked.frag` for the alpha-MASK
pair), the shadow casters share `shadow_depth.frag` /
`shadow_depth_opaque.frag`, and every post-process and IBL pass reuses
`fullscreen.vert`.

A compute pipeline is one `.comp.hlsl` — there is no vertex/fragment pair, and
its threadgroup size is never written down on the C++ side: it comes from the
module's `[numthreads]` via SPIR-V reflection, which is why compute pipeline
creation *requires* reflection to succeed.

**Binding order inside set 0 is not free.** SDL3 GPU assigns sampled textures
first, then read-only storage textures, then read-only storage buffers, all
numbered contiguously (see `gpu/gpu_compute.hpp`). Declaring them in any other
order compiles, reflects and validates cleanly, and then silently binds the
wrong resource to each slot — `rt_shadow_denoise_temporal` was written with its
storage textures first and produced full-screen noise that looked for all the
world like a broken denoiser.

**Wave (subgroup) intrinsics need an opt-in.** The shadercross CLI never passes
`-fspv-target-env`, so DXC defaults to `vulkan1.0`, under which every `Wave*`
intrinsic is a hard compile error. A shader that needs them must be listed in
`SHADER_WAVE_OPS` in `CMakeLists.txt`, which routes it through the vendored
DXC CLI at `vulkan1.1` with otherwise identical flags. That list is explicit
rather than a filename convention on purpose: adding to it raises the Vulkan
floor for the pipelines involved, which should be visible in a diff. The Metal
path needs nothing — SPIRV-Cross maps these to `simd_*` and the runtime already
requests MSL 2.2.0, above the 2.1 floor Metal requires for them.

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
reference: `src/bracken/contracts/velocity_math.hpp`), computed from UNJITTERED
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
counts: model/instanced 15, terrain 13, grass 11). The THREE slots below it
are the probe field's atlases (model/instanced 11–13, terrain 9–11, grass
7–9): irradiance and visibility with a LINEAR clamp sampler, the per-probe
metadata plane with a POINT one, read by `gi_sample.hlsli` and gated by the
`giEnabled` uniform. All three cascades live in one texture each (banded rows
— `render/gi/gi_math.hpp`), which is why the field costs three slots and not
nine. Below THOSE is
`rtShadowTexture` (model/instanced 10, terrain 8, grass 6), the R8 ray-traced
sun-visibility mask — sampled with a NORMALIZED UV because it lives at the
traced pass's own resolution, and bound to the 1x1 white dummy (= fully lit)
when ray tracing is off. It used to be multiplexed into `screenAOTexture`'s
G channel on the belief that a 12th sampler failed DXC legalization; it does
not (12 compiles, verified to 17 — which is the headroom the three GI slots
were spent out of), and the packing zeroed the ambient term scene-wide
whenever AO was off. Cluster lights whose
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
forward pass samples; applies `ao_strength`). Ray-traced reflections are
resolved after forward lighting by `rt_reflection_resolve`: it rejects
ineligible centre materials, performs a depth/normal/roughness/metallic-guided
4-tap gather from the reduced-resolution raw-radiance texture, evaluates the
full-resolution receiver BRDF, and adds RGB while preserving HDR alpha. This
placement is before water/TAA, so the signal cannot bleed from stone onto
grass or bypass temporal stabilization. The RT compute shaders bind D32 depth
and the G-buffer as sampled textures with a point-clamp sampler, never as
compute storage. `rt_reflection.comp` writes TWO read-write storage textures:
`u0` the RGBA16F raw radiance, and `u1` an R16F `hitT` — the hit distance
normalized by `rayMaxDistance`, 1.0 on a miss or any early-out (sky, roughness
gate, water-receiver rejection), written by every in-bounds thread. Nothing
reads `hitT` yet; it exists because the virtual-hit-point reprojection planned
for the reflection denoiser needs the reflected point's distance, and the trace
is the only place that number is free. When water is active, that resolved quarter-resolution target
is recycled for a second analytic water-receiver dispatch: camera-ray/plane
intersection is refined against the live wave field and the trace uses the
same finite-difference + analytic-detail normal as `water.frag`. The water
shader keeps confident full-detail SSR, uses RT for off-screen/occluded/edge
gaps, and leaves true misses to IBL. The remaining passes sample the HDR
offscreen scene target:
`bloom_downsample` (13-tap, threshold in
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

**Probe-field GI (compute)** — four passes over `render::gi::ProbeVolume`'s
atlases, recorded once per cascade whose turn the rota gives it and all timed as
one `GIProbes` scope. `gi_probe_trace.comp` casts one ray per (scheduled probe,
ray slot) over the SAME software BVH the other traced effects walk — it includes
`rt_trace.hlsli` in both phases and uses the **closest-hit** walk, because a
probe ray has to shade what it finds. Hit shading is flat `TriangleAttributes`
albedo times (sun N·L with a **CSM atlas fetch** for visibility + the frame's
point/spot lights, range-windowed, read straight out of the cluster light
buffer's byte layout + π × last frame's own probe irradiance at the hit, which
is what makes one bounce per update converge to infinite bounce); a miss samples
the IBL sky-CAPTURE cube (raw radiance — `IBL::capture_map()`), falling back to
the `EnvironmentState` hemisphere behind a uniform flag when the cube dummy is
bound. **That shader includes `shadow_common.hlsli` and calls `calcShadow()`
directly**: its cbuffer's first 336 bytes are `ShadowDataUniforms` field for
field under the names that file requires, which is the only formulation in which
the probe path and the raster path cannot drift apart on the deliberately
standard-Z cascade atlas. Output is RGB radiance + a **signed** hit distance per
ray, negative on a backface — the classification signal.
`gi_probe_blend.comp` and `gi_probe_blend_visibility.comp` run one threadgroup
per probe with `[numthreads]` equal to the probe's tile (8×8 / 16×16): payload
threads cosine-weight (or `cos^50`-weight, for the visibility moments) the whole
ray set against their own octahedral direction, then a
`GroupMemoryBarrierWithGroupSync` later the border threads replicate the
octahedral wrap out of `groupshared` — the border is a pure function of the
payload the group just computed, so a separate pass would re-read the atlas it
had only just written. Both read-modify-write their atlas, which is why their
`RWTexture2D` carries an explicit `[[vk::image_format]]`: a storage image
declared `Unknown` may only be WRITTEN without the optional
`shaderStorageImageReadWithoutFormat` device feature. `gi_probe_classify.comp`
is one thread per probe: backface fraction → a relocation offset (and, past
75% at the clamp, an inside-geometry verdict) into the per-probe RGBA16F offset
texture, plus two `InterlockedAdd` diagnostic counters. Ray directions are
spherical Fibonacci rotated per frame from the frame counter alone, and all four
passes derive them the same way — the blend and classify passes rebuild each
ray's direction rather than having it written out, which is why `frameIndex` is
a uniform of all of them.

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
they must stay one function so fog melts into the sky at the horizon),
`gi_probe.hlsli` (the probe field's grid constants, torus wrap, atlas tiling,
octahedral encode/decode + the octahedral **border wrap**, spherical-Fibonacci
ray directions and their R3/Shoemake per-frame rotation, and the
change-adaptive blend curve — every function a transliteration of
`render/gi/gi_math.hpp`, which is the source of truth and is tested with no GPU;
binding-free by construction, so unlike `rt_trace.hlsli` it can be included
anywhere), `gi_sample.hlsli` (the CONSUMPTION half of the probe field:
DDGI-style sample bias, cascade selection with a half-spacing margin and a
one-probe fade band, the 8-probe trilinear gather with backface, Chebyshev and
relocation weights, the coarser-cascade blend and the hemisphere fallback,
plus `giSampleVolume` for participating media — a transliteration of the
`SAMPLE_*` block and `sample_field()` in `render/gi/gi_math.hpp`. Unlike
`gi_probe.hlsli` it NAMES resources — `giIrradianceAtlas`, `giVisibilityAtlas`,
`giProbeMeta` and their samplers, plus the `giCascadeBase[]`/`giEnabled`
cbuffer members — so the including shader owns their slots and must declare
them first, exactly like `cluster_lighting.hlsli`), `rt_sampling.hlsli` (R2 + interleaved-gradient-noise sample sequence, cone and
GGX-VNDF sampling, the packed visibility-tile helpers), `rt_trace.hlsli` (the
software two-level BVH: `BvhNodeGpu`/`BvhTriangle`/`TlasInstance`, `slabTest`,
`octDecode`, the any-hit `blasOccluded`/`sceneOccluded` walk and the
closest-hit `Hit`/`blasClosest`/`sceneClosest` walk — shared by
`rt_shadow.comp`, `rt_reflection.comp` and `gi_probe_trace.comp`). The traversal node is **64 bytes
and holds BOTH children** (Aila–Laine): two AABBs and two links, so one fetch
resolves a binary decision and a leaf never needs a node of its own. The CPU
build format is still the 32-byte one-AABB `BvhNode` (`bvh.hpp`); the
conversion happens on the upload side (`bvh_gpu.hpp`), which is also why only
interior nodes become nodes and the buffers do not grow. Traversal stacks are
64 entries, the depth bound the builder's `max_depth` implies now that a visit
defers at most one side. `rt_trace.hlsli` is a **two-phase**
include: `#define RT_TRACE_TYPES` before the first include to get the structs
(the including shader needs them to declare its own `StructuredBuffer`s), then
`#define RT_TRACE_TRAVERSAL` and include it again *after* the binding block and
cbuffer, because the walks read `tlasNodes`/`tlasInstances`/`blasNodes`/
`blasTriangles` and the `tlasNodeCount` uniform by name. Missing both macros is
an `#error`, the same shape as `cluster_lighting.hlsli`'s required
`CLUSTER_SSBO_BINDING_BASE`. Bindings deliberately stay in the entry shaders —
the two passes have different sampled-texture counts and only reflections bind
the per-triangle attribute buffer.

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

Both directions are tested. The CPU side is pinned by the
`BRACKEN_HLSL_CBUFFER` / `BRACKEN_HLSL_FIELD` static-assert block after each
struct in `gpu_uniforms.hpp` (see `gpu/hlsl_packing.hpp`), so a layout edit
there fails to compile. Agreement with *this* directory is checked by
`src/bracken/gpu/uniform_layout.test.cpp`, which reads the **compiled**
`.spv` files and compares each cbuffer's declared byte extent to `sizeof(T)`.
Two consequences when editing shaders:

- Declare trailing padding explicitly (named scalars, never arrays) so the
  cbuffer's declared extent equals the CPU struct's `sizeof` — the test
  requires exact equality, not "rounds up to the same multiple of 16".
- A **new** `cbuffer` needs a row in that test's `UNIFORM_TABLE` naming the
  CPU struct behind it; an unlisted uniform block fails the test.
