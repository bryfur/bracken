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
Depth-reading shaders (`gtao`, `gtao_temporal`, `blur_ao`, `gtao_upsample`,
`volumetric_fog`)
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
are never attenuated by screen AO. Since R5a that texture is RGBA
(`bent_cone.hlsli`), and the DIFFUSE GI term is gated by the decoded bent
cone — evaluated along the bent normal, scaled by the cone aperture — INSTEAD
of the scalar screen AO, never by both. The slot is bound to a 1×1 white dummy
when AO is off (and always for BLEND draws), which decodes to the fully-open
escape and restores the scalar path exactly. The LAST THREE sampler slots of
every scene fragment shader are the sky IBL chain (irradiance cube,
GGX-prefiltered specular cube, split-sum BRDF LUT — `render/ibl.hpp`);
when the `iblEnabled` uniform flag is on, `ambientIBL()` (pbr.hlsli) replaces
the hemisphere ambient with sky-derived diffuse irradiance + prefiltered
specular (roughness → mip, `IBL_SPECULAR_MIPS` must match
`render::IBL::SPECULAR_MIPS`), multiplied by the same combined AO and
shadow-driven ambient lift. When off, cube-typed dummies keep the slots
valid and the shaders branch to the hemisphere `ambientPBR` fallback.
The very LAST sampler slot of every scene fragment shader is the LOCAL
(point/spot) shadow ATLAS + `LESS_OR_EQUAL` comparison sampler, declared by
`cluster_lighting.hlsli` at `CLUSTER_SSBO_BINDING_BASE - 1` (fragment sampler
counts: model/instanced 16, terrain 14, grass 12).

**SIXTEEN IS A HARD CEILING, and it is SDL's.** `SDL_CreateGPUShader` asserts
`"Shader sampler count cannot be higher than 16!"` and returns null, so a
17-sampler `model.frag` is a missing pipeline and an unrendered scene, not a
slow one. DXC legalizes 17 — that is the measurement that retired the old
belief in an 11-sampler limit — but DXC's ceiling and SDL's are different
numbers and SDL's binds. `model`/`model_instanced` are AT it. Anything further
these shaders need is a storage buffer: R4b's screen-probe SH is exactly that,
a FIFTH fragment storage buffer at `CLUSTER_SSBO_BINDING_BASE + 4`, declared by
`gi_screen_sample.hlsli`. Directly below it is
`rtLocalShadowTexture` (model/instanced 14, terrain 12, grass 10), the R8
ray-traced LOCAL-light visibility RATIO — sampled with a NORMALIZED UV, bound
to the 1×1 white dummy and read only when `rtLocalShadowEnabled`, and
multiplied into the whole point+spot subtotal by `accumulateClusterLights`
(the Heitz ratio estimator, `LocalRtShadowMode::Ratio`). Note that the two are
mutually exclusive by construction rather than by a branch: in Ratio mode the
engine never assigns a `shadow_slot`, so the atlas taps above are already dead.
The THREE slots below THAT
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
`SDL_BindGPUFragmentStorageBuffers(pass, 0, bufs, 5)` binds cluster
light data / offsets / indices / local-shadow slots / **R4b's screen-probe SH**.
When local shadows are unavailable the slot binds any D32 texture (the CSM
atlas) and the SSBO slot binds the light-data buffer — never sampled while
every `shadow_slot` is -1; the fifth slot takes the same substitution when the
screen-probe gather is not running, and `giScreenEnabled == 0` is what stops it
being read.
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
depth: `gtao` (horizon-based GTAO, Jimenez et al. 2016, with an SSILVB
**visibility bitmask**, Therrien et al. 2023 — per slice a 32-sector `uint`
occlusion mask marked by constant-thickness slabs instead of a horizon pair,
integrated with the same cosine weight, 3 slices × 4 steps per side, per-pixel
IGN + per-frame golden-ratio temporal jitter, edge-aware depth-reconstructed
normals; no wave intrinsics — `countbits`-class work only. CPU reference for
every piece of that math: `render/gtao.hpp`), `gtao_temporal` (velocity-
reprojected accumulation into an RGBA16F history plus a separate RG16F depth
history, relative-depth disocclusion rejection — MRT, because five linearly
filtered quantities do not fit in four channels), `blur_ao` (bilateral, ×2
separable), `gtao_upsample` (joint-bilateral upsample to the full-res RGBA8
texture the forward pass samples; applies `ao_strength` to both occlusion
channels).

Every texture in that chain carries FOUR channels, and `bent_cone.hlsli` is
the single owner of the layout: R = AO, GB = octahedral WORLD-space **bent
normal**, A = **cone aperture** as sin²(half-angle). Filters decode the bent
normal, average it as a direction, renormalise and re-encode — averaging
octahedral coordinates across the fold seam averages nothing. `A >= 0.998`
(`BENT_CONE_OPEN`) means "fully open, no directional data", which is what the
1×1 white dummy decodes to, and is why AO-off needs no uniform flag. Ray-traced reflections are
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
gate, water-receiver rejection), written by every in-bounds thread.

Since R3 that pass shades its hits through **`rt_hit_shade.hlsli`** (below), not
with the flat `albedo × (sunColor·N·L + skyColor × 0.25)` it used to: the sun is
CSM-shadowed, the frame's point/spot lights are looped, the ambient comes from
the probe field (or the hemisphere), and a miss samples the sky cube
directionally instead of returning one horizon colour. That widened its binding
block to 6 samplers + 1 read-only storage texture + **6** read-only storage
buffers — two under SDL3 GPU's eight, which is why the light list is read flat
rather than through `cluster_compute.hlsli`'s froxel grid (a ray hit is a
world-space point and has no froxel anyway).

`rt_reflection_denoise_temporal.comp` now CONSUMES `hitT`, plus the receiver's
depth and G-buffer: it extends the eye ray by the hit distance to place the
reflection's **virtual point**, projects that through the previous frame's
unjittered camera, and blends the resulting motion vector against the velocity
buffer's by roughness (mirror-ish → virtual, rough → surface;
`render/rt_reflection_math.hpp` is the CPU source of truth for the curve). Both
vectors are UV deltas in the velocity buffer's own convention, so the TAA jitter
cancels inside each difference. A static camera makes both identically zero and
the mechanism exactly neutral.

R5b gave it one read-only storage buffer: `render::AutoExposure`'s persistent
state block (a `float4` {current EV, target EV, avg log2 luma, valid}), which
was already a `ComputeStorage` buffer and so needed no change to be bound
read-only here. `rtAdaptedEv` adds the manual bias to it exactly as the
composite pass multiplies the two, so the clamps mean the same thing with auto
exposure on and off. It buys two things a scale-free σ-clamp could not: a FLOOR
under the neighbourhood σ, so a flat region's clamp box cannot collapse onto
this frame's single stochastic sample and throw away the accumulation at exactly
the pixels that would have converged best; and a FIREFLY clamp, new in R5b,
whose ceiling is the max of a relative term and an absolute post-exposure one so
that neither a bright neighbourhood nor a dark one can make the other stricter.
Reflection *gradients* are deliberately out of R5b — the replay would have to
interact with the virtual-point reprojection above, whose motion vector is
itself a function of a stochastic hit distance. `gi_screen_probe_filter.comp`
binds the same buffer for the same reason, giving its firefly ceiling the
absolute floor its own comment had been promising since R4b.

`rt_local_shadow.comp` (R2b) is the third traced signal and the only one that
reads the cluster grid: per pixel it draws six candidates from the pixel's
froxel through `cluster_compute.hlsli`, weights them by `clusterLightTarget`,
keeps one by weighted-reservoir sampling, merges the reprojected previous-frame
reservoir (its light index resolved through the prev→current remap; a vanished
light drops the history), and casts ONE sphere-cap ray at the winner. Its
output is the same 8×4-packed bit tile `rt_shadow.comp` produces — which is the
point, because it goes through the SAME `rt_shadow_denoise_*` chain, dispatched
a second time against a second set of targets. Its two set-1 read-write storage
buffers are the reservoir ping-pong, of which the second is only ever READ:
SDL3 GPU allows a compute pipeline eight read-only storage buffers and the four
acceleration-structure buffers plus four cluster buffers spend them exactly.

`rt_shadow_gradient.comp` and `rt_local_shadow_gradient.comp` (R5b) are the
**A-SVGF temporal gradient** for the two visibility signals — the only thing in
the frame that can tell a denoiser its history is stale because the LIGHTING
changed rather than because something moved. One thread per 3×3 stratum of the
traced resolution (the 1-in-9 ray budget), a stratum pixel chosen by
`rtGradientPixelInStratum` (a hash of the stratum and the frame counter, so it
rotates and stays deterministic), reprojected through velocity, then **re-traced
against THIS frame's scene with the PREVIOUS frame's sample** —
`rtSample2D(prevCoord, frameIndex - 1)`, which is literally the sample the
previous frame took there. Holding the random draw fixed is what cancels the
sun-disk jitter in the comparison; without it every stratum would report its own
sampling variance as a lighting change. The comparison target is the previous
frame's raw packed bits, retained by a ping-pong on the bits texture.

The local variant differs in one place that matters: it must also reproduce
WHICH LIGHT the previous frame's reservoir picked, resolved through R2a's
prev→current remap, or it would report a full gradient on every pixel whose
reservoir merely moved. A light the remap says is gone gives a gradient of 1
with no ray at all. It takes the same eight-read-only-storage-buffer split
`rt_local_shadow.comp` does, and for the same ceiling; `clusterParams` at
set 2 b1 is why it includes `cluster_compute.hlsli` whole despite drawing no
new candidates — that view matrix is the only source of a view depth that agrees
with the CPU binner's. Both use a depth-continuity tolerance **two orders of
magnitude tighter** than the denoiser's, because they ask a strictly stronger
question ("is this the same POINT?" rather than "is this history reusable?") and
because at the loose one the TAA jitter alone reports a 3.3% fire rate on a
completely static scene.

`rt_gradient_reconstruct.comp` turns the sparse verdicts dense: the **weighted
maximum** over the 5×5 stratum window, weights falling off by Chebyshev ring.
Not an à-trous average, deliberately — the verdict is binary, and a b3-spline
blur of one fired stratum among 24 quiet ones is 0.14, which resets nothing. One
pipeline, dispatched once per enabled signal against that signal's own pair of
targets, exactly as the `rt_shadow_denoise_*` three are.

`rt_shadow_denoise_temporal.comp` consumes all of it. It gained a fourth sampler
(the FAST history — a four-frame mean in `RG16F` that is never displayed and
exists only to clamp the main mean back when it drifts outside that mean's own
sampling error), a third read-only storage texture (the reconstructed gradient),
a second read-write target (the fast history out) and one read-write buffer: the
never-reset diagnostics counters whose differences the `--rt-bench` harness
reports as the mean accumulation window and the gradient fire fraction. The
`lightingReset` uniform is the analytic sun gradient, computed on the CPU before
any ray was cast and shared with the probe field's blend; at 0 — every frame
whose sun held still — the whole chain reduces to its pre-R5b arithmetic bit for
bit.

When water is active, that resolved quarter-resolution target
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
probe ray has to shade what it finds. Hit shading moved to `rt_hit_shade.hlsli`
in R3 and is shared with `rt_reflection.comp`: flat `TriangleAttributes`
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

**Hierarchical-Z pyramid (compute)** — two passes, both timed as one `HZB`
scope, and read by R4b's screen-probe march (`docs/rt_lighting_plan.md` §5.5).
They run only when
`gi_mode == GiMode::ProbesScreen`. `hzb_copy.comp` writes mip 0 of an R32_FLOAT
mip chain from the D32 prepass depth — the depth arrives through a **sampler**
because a depth format cannot be a compute storage texture — and stores the raw
reversed-Z device depth without linearizing (that would bake this frame's
projection into the pyramid). `hzb_reduce.comp` is dispatched once per further
mip, **each in its own compute pass**, reading mip N-1 and writing mip N of the
same texture: both arrive as set-1 read-write bindings, because SDL3 GPU only
permits reading a texture inside a compute pass when it was created with
`COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE`, and the read-write binding's
`mip_level` is the only way to name one level. Both `RWTexture2D` declarations
carry `[[vk::image_format("r32f")]]` for the same reason the GI blend passes do
— a storage image declared Unknown may only be written without the optional
`shaderStorageImageReadWithoutFormat` feature.

The reduction is a **MAX**, not a min: under reversed-Z a larger stored depth is
a *closer* surface, and a march may skip a footprint only when the ray is in
front of the footprint's closest surface, so that is what a coarse texel must
keep. The gather is 2×2 except at a trailing odd edge, where the last
destination texel extends to the end of the source (a plain `dst*2 + {0,1}`
gather drops the last row/column of an odd mip, and clamping does not fix it —
it re-reads a column already covered). `render/hzb.hpp` carries the CPU mirror
of both rules and `render/hzb.test.cpp` brute-forces the whole pyramid against
it.

**A CONSUMER MUST NOT ADDRESS THIS PYRAMID BY `uv * mipSize`.** Because the last
destination texel of each axis is oversized, mip N's footprints are 2^N, 2^N, …,
2^N + r — so a hardware uv sample (which maps uv linearly onto the mip's
dimensions) asks the wrong texel near the right and bottom edges. The correct
grid is `screenSize / 2^level`, with the index clamped to the mip's dimension
for the LOOKUP only, and the fetch is a `Load` rather than a `SampleLevel`.
`gi_screen_probe.hlsli`'s `giSpCellGrid`/`giSpCell` are that, and the CPU mirror
test caught the mistake before a pixel was rendered.

**Screen-probe gather (compute, six passes)** — R4b, all inside one
`GIScreenProbes` scope, in this order and each in its own compute pass:
`gi_screen_probe_place.comp` (one thread per probe: the fixed nine-candidate
anchor search on the depth buffer, the probe header, and the frame's counter
zeroing), `gi_screen_probe_march.comp` (one threadgroup per probe, 64 threads =
the whole octahedral payload; hemisphere-only rays, the Hi-Z ladder, the TAA
history tap, and a groupshared compaction of everything unresolved into a miss
buffer with ONE `InterlockedAdd` per group), `gi_screen_probe_args.comp` (one
thread; the miss count becomes an `SDL_GPUIndirectDispatchCommand` — the
engine's first runtime use of `ComputePass::dispatch_indirect`),
`gi_screen_probe_trace.comp` (**indirectly dispatched** over the compacted
misses; short BVH rays through `rt_trace.hlsli`, shaded by `rt_hit_shade.hlsli`,
with a sky/probe-field blend beyond the 600-unit clamp — and at EIGHT read-only
storage buffers, SDL3 GPU's ceiling exactly, which is why the miss count lives
in the miss buffer's own header rather than in a ninth binding),
`gi_screen_probe_filter.comp` (probe-space temporal: firefly clamp against the
probe's own mean, then a velocity-reprojected EMA), and
`gi_screen_probe_project.comp` (the 3x3 probe-space spatial filter fused with
the L1 SH projection, writing the `GiScreenProbeSh` record the scene shaders
read). The pass boundaries are not stylistic: SDL3 GPU has no intra-pass
synchronization and each reads the previous one's output, and the indirect
dispatch additionally requires its producing pass to have ENDED.

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
hemisphere `ambientPBR`, sky-IBL `ambientIBL`, specular AA; both ambient
functions occlude each ambient SOURCE separately since R5a — hemisphere/cube
diffuse and specular by the scalar `ao`, the GI diffuse by the bent-cone
`giGate` — which is what lets the cone REPLACE the scalar for that one term
rather than stack with it; `giGate < 0` means "no cone" and reproduces the
old arithmetic), `bent_cone.hlsli` (the AO texture's four channels in one
place: octahedral encode/decode for the bent normal, the `BENT_CONE_OPEN`
escape value, `decodeBentCone`, and `bentConeGiGate` — the single place the
GI double-count rule is decided. Written by the whole GTAO chain, read by
every scene fragment shader),
`sky_common.hlsli` (the procedural sky radiance function shared by the
skybox and sky-capture passes; includes `fog.hlsli` for the shared
`hazeSunColor`), `lighting.hlsli` (luminance,
linear→sRGB transfer, interleaved gradient noise, the ACES tonemap curves —
Narkowicz + Hill fit, consumed by `composite.frag` — and the shared
`viewPosFromDepth`/`worldPosFromDepth` depth-reconstruction helpers used by
`gtao`/`volumetric_fog`), `cluster_index.hlsli` (the froxel
index math — grid constants, `ClusterParams`, the light records,
`clusterZSlice`/`clusterIndexFromScreen`, `pointFalloff`/`spotConeFactor`;
binding-free AND pbr-free by construction, so the fragment and compute paths
share one copy of the arithmetic, and its CPU mirror
`render/lighting/cluster_index.hpp` is what the binner itself calls),
`cluster_light_load.hlsli` (the packed light-data decode — NAMES `lightDataBuf`
and `clusterParams`, so the including file declares them first),
`cluster_lighting.hlsli` + `clusters.hlsli`
(the FRAGMENT path: cluster grid sampling, the PBR accumulator, and the local
point/spot shadow bindings and sampling — define `CLUSTER_SSBO_BINDING_BASE`
to the including shader's fragment sampler count first),
`cluster_compute.hlsli` (the COMPUTE path to the same froxel grid: SDL3 GPU
puts compute storage buffers on set 0 and compute uniforms on set 2, so the
bindings must differ from the fragment path — but the two includes above mean
the index math and record decode cannot. Declares FOUR consecutive set-0
storage buffers from `CLUSTER_COMPUTE_SSBO_BINDING_BASE` (light data, cluster
offsets, light indices, prev→current index remap) plus `ClusterParams` at
`CLUSTER_COMPUTE_PARAMS_BINDING` on set 2, and exposes
`clusterFroxelIndex`/`clusterFroxelLightCount`/`clusterFroxelLight` plus
`clusterRemapPrevLight` and, since R2b, the RIS target function
`clusterLightTarget` — shared with the fragment accumulator through
`cluster_light_load.hlsli` so "the forward loop skips this light" and "the
sampler weights it zero" stay one statement. A FIFTH buffer, the packed stable
ids, and with it `clusterLightStableId`, are **opt-in** behind
`CLUSTER_COMPUTE_WANT_STABLE_IDS`: SDL3 GPU caps a compute pipeline at EIGHT
read-only storage buffers and any ray-tracing pass spends four on the
acceleration structure before it gets here. A pass reusing a sample ACROSS
FRAMES wants the remap, not the ids. Its consumer
is `rt_local_shadow.comp`; the placeholder `cluster_compute_validate.comp` that
kept the include a compiled artifact between R2a and R2b is gone),
`shadow_common.hlsli` (cascade selection, PCF/PCSS),
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
them first, exactly like `cluster_lighting.hlsli`. A shader that wants only
`giSampleVolume` `#define`s **`GI_SAMPLE_VOLUME_ONLY`** first: everything that
reads `giVisibilityAtlas` is compiled out and that pair must NOT be declared,
because an unread `Texture2D`/`SamplerState` pair does not survive DXC's
combined-image-sampler legalization and reflects as a storage texture instead
of a sampler — see "Declared resource counts" below. `volumetric_fog.frag` is
the one shader that takes the opt-out), `gi_screen_probe.hlsli` (R4b's screen-probe constants, tile/probe indexing, the
fixed anchor search order, the jittered octahedral directions, the Hi-Z march's
cell-step and mip-addressing functions, the L1 SH basis and evaluation, and the
`GiScreenProbe` / `GiScreenProbeSh` / miss-record layouts — every function a
transliteration of `render/gi/screen_probe_math.hpp`, which is the source of
truth and is tested headless against a brute-force marcher. Binding-free, like
`gi_probe.hlsli`), `gi_screen_sample.hlsli` (the CONSUMPTION half of the screen
probes: the 2x2 probe gather with per-tap depth/normal rejection, renormalised
and ramped by a weight floor, returning the SAME unit as `gi_sample.hlsli` so
the scene shaders blend one into the other with a plain `lerp`. Unlike
`gi_sample.hlsli` it OWNS its binding — the fifth fragment storage buffer at
`CLUSTER_SSBO_BINDING_BASE + 4` — so it must be included AFTER
`cluster_lighting.hlsli`; it still names three cbuffer members
(`giScreenEnabled`, `giScreenProbesX/Y`) the including shader must declare),
`rt_sampling.hlsli` (R2 + interleaved-gradient-noise sample sequence, cone and
GGX-VNDF sampling, the packed visibility-tile helpers, and `rtSeed`/`rtNextFloat`
— a per-pixel per-frame random STREAM for passes like the froxel-RIS reservoir
that make a dozen independent decisions per pixel, where feeding them all from
one low-discrepancy sequence would correlate the candidate choice with the
acceptance test — and, because both are seeded from `(pixel, frameIndex)` alone,
both are REPLAYABLE, which is the property R5b's A-SVGF gradient is built on),
`rt_denoise.hlsli` (the R5b denoiser core: the analytic sun gradient's ramp, the
gradient stratum picker and its dilation weights, the dual fast/slow history's
margin, and the exposure-aware clamp thresholds — a transliteration of
`render/rt_denoise_math.hpp`, which is the tested source of truth for every
constant in it. It declares no resource and reads no uniform, so the two
visibility temporal passes, the reflection temporal pass and the screen-probe
filter all include it; it takes `rt_sampling.hlsli` itself for `rtSeed` and
`luminance`), `rt_hit_shade.hlsli` (**the one
hit-shading model** every traced signal uses, R3: `TriangleAttributes` in phase
1, then `rtHitSkyRadiance` for a miss and `rtShadeHit` for a hit —
`albedo_flat × (CSM-shadowed sun N·L + range-windowed point/spot lights + π ×
probe-field irradiance) / π`. A **two-phase** include like `rt_trace.hlsli`, and
it names resources it does not own — `shadowAtlas`/`shadowCmpSampler`,
`shadowAtlasRaw`/`shadowRawSampler`, `giSkyCube`, `giIrradiance`, `giProbeState`,
`lightDataBuf` — so the including shader owns their slots, exactly like
`cluster_lighting.hlsli`. It includes `shadow_common.hlsli` itself, so a
consumer cannot forget it and silently get an unshadowed sun back. Two flags in
the required cbuffer separate its callers: `giEnabled` (0 binds 1×1 dummies and
the ambient falls to the hemisphere) and `giAmbientFallback` — 1 for reflection
rays, **0 for probe rays**, because a probe already collects sky through its ray
misses and adding a hemisphere on top would double-count it and break the bounce
series' convergence. Consumers: `rt_reflection.comp`, `gi_probe_trace.comp` and
`gi_screen_probe_trace.comp`),
`rt_trace.hlsli` (the
software two-level BVH: `BvhNodeGpu`/`BvhTriangle`/`TlasInstance`, `slabTest`,
`octDecode`, the any-hit `blasOccluded`/`sceneOccluded` walk and the
closest-hit `Hit`/`blasClosest`/`sceneClosest` walk — shared by
`rt_shadow.comp`, `rt_local_shadow.comp`, `rt_reflection.comp` and
`gi_probe_trace.comp`). The traversal node is **64 bytes
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
the passes have different sampled-texture counts, only reflections bind the
per-triangle attribute buffer, and only `rt_local_shadow.comp` binds the
cluster grid.

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

A mismatch is not always a stale count. **Declaring a texture the shader never
reads changes the reflected layout**: DXC merges a `Texture2D` and its
`[[vk::combinedImageSampler]]` partner into one SPIR-V sampled image only where
executed code pairs them, so an unread pair stays two separate variables and
shadercross counts the orphaned image as a *storage texture* — which lands a
storage texture in the middle of the `space2` sampler run and breaks the packed
order above. Delete the declaration (see `GI_SAMPLE_VOLUME_ONLY` in
`gi_sample.hlsli`); do not "fix" the catalogue row to match.

**Integer-format textures may not carry `SDL_GPU_TEXTUREUSAGE_SAMPLER`.** There
is no filtering defined for a `_UINT`/`_INT` texel, so `SDL_CreateGPUTexture`
asserts — and under `BRACKEN_GPU_DEBUG=1` that assertion poisons the device,
failing every texture creation after it. The engine's integer textures (the GI
probe-state plane, the RT denoisers' packed visibility bits and tile classes)
are storage-only for that reason: bind them with
`bind_storage_textures`/a read-write binding and `Load()` them.
`GPUTexture::is_integer_format()` is the predicate, and `GPUTexture::create_2d`
asserts on the combination.

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
