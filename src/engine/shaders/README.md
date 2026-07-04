# HLSL Shader Sources

HLSL shaders compiled to SPIR-V at build time. `CMakeLists.txt` in this
directory fetches SDL_shadercross (bundling DXC + SPIRV-Tools), globs
`*.hlsl`, and emits one `shadercross` CLI invocation per shader; stage and
entry point are derived from the filename. Compiled `.spv` files land in
`<build>/shaders/` and are loaded at runtime by `ShaderManager` /
`PipelineRegistry` from the `shaders/` directory next to the executable
(`mmo_engine_copy_shaders(<target>)` stages them there).

Every `.hlsli` is a declared dependency of every shader, so editing an
include recompiles everything — an incremental `cmake --build build` is
always enough. With `MMO_HOT_RELOAD=1`, the `PipelineRegistry` watches the
runtime `shaders/` directory and rebuilds all pipelines when any `.spv`
changes.

## Naming convention

- `{name}.vert.hlsl` — vertex shader, entry point `VSMain`
- `{name}.frag.hlsl` — fragment shader, entry point `PSMain`
- `{name}.hlsli` — shared include (not compiled standalone)

## Pipeline → shader map

Which shaders each `PipelineType` (`gpu/pipeline_registry.cpp`) actually
loads. Note the shared shaders: the skinned pipeline reuses `model.frag`,
particles reuse `model.vert`, and every post-process pass reuses
`fullscreen.vert`.

| Pipeline | Vertex | Fragment |
|---|---|---|
| Model | `model.vert` | `model.frag` |
| SkinnedModel | `skinned_model.vert` | `model.frag` |
| InstancedModel | `model_instanced.vert` | `model_instanced.frag` |
| Terrain | `terrain.vert` | `terrain.frag` |
| Grass | `grass.vert` | `grass.frag` |
| Skybox | `skybox.vert` | `skybox.frag` |
| Particle | `model.vert` | `particle.frag` |
| Grid | `grid.vert` | `grid.frag` |
| UI | `ui.vert` | `ui.frag` |
| Text | `text.vert` | `text.frag` |
| Billboard | `billboard.vert` | `billboard.frag` |
| ShadowModel | `shadow_depth.vert` | `shadow_depth.frag` |
| InstancedShadowModel | `shadow_depth_instanced.vert` | `shadow_depth.frag` |
| ShadowSkinnedModel | `shadow_skinned.vert` | `shadow_depth_opaque.frag` |
| ShadowTerrain | `shadow_terrain.vert` | `shadow_depth_opaque.frag` |
| SSAO | `fullscreen.vert` | `ssao.frag` |
| GTAO | `fullscreen.vert` | `gtao.frag` |
| BlurAO | `fullscreen.vert` | `blur_ao.frag` |
| BloomDownsample | `fullscreen.vert` | `bloom_downsample.frag` |
| BloomUpsample | `fullscreen.vert` | `bloom_upsample.frag` |
| VolumetricFog | `fullscreen.vert` | `volumetric_fog.frag` |
| Composite | `fullscreen.vert` | `composite.frag` |
| Fxaa | `fullscreen.vert` | `fxaa.frag` |

## Shader notes

**Scene** — `model.*`: static models, PBR + clustered lights + CSM + fog.
`skinned_model.vert`: skeletal skinning; bone matrices come from a per-frame
`StructuredBuffer<float4x4>` of tightly-packed palettes (vertex set 0,
binding 0) indexed by a `paletteOffset` uniform — no compile-time bone count
(the CPU caps palettes at `MAX_BONES = 128`, `animation/animation_types.hpp`).
`model_instanced.*`: per-instance transforms from a storage buffer (rocks,
trees). `terrain.*`: splatmap texturing, height-based lighting. `grass.*`:
instanced blades, wind, alpha test. `skybox.*`: procedural sky (sun disc /
corona / gradient / horizon glow) on a fullscreen triangle.

**Shadows** — depth-only cascade passes into the 2×2 shadow atlas.
`shadow_depth_opaque.frag` is the no-alpha-test variant used by skinned +
terrain casters.

**Post-processing** — all sample the HDR offscreen scene target via
`fullscreen.vert` (fullscreen triangle, no vertex input): `ssao` / `gtao` +
`blur_ao` (bilateral), `bloom_downsample` (13-tap) / `bloom_upsample` (9-tap
tent, additive), `volumetric_fog` (ray march with god rays from the shadow
atlas), `composite` (scene × AO + bloom + fog, exposure + tonemap + gamma →
LDR), `fxaa` (FXAA 3.11 on the tonemapped LDR image).

**Effects/UI** — `particle.frag`: unlit additive HDR particles. `ui`/`text`:
2D pipelines targeting the swapchain. `billboard.*`: camera-facing quads.
`grid.*`: debug grid overlay.

**Includes** — `pbr.hlsli` (GGX/Smith/Schlick/Burley BRDF, ACES tonemap,
`ambientPBR`), `lighting.hlsli`, `cluster_lighting.hlsli` + `clusters.hlsli`
(cluster grid sampling), `shadow_common.hlsli` (cascade selection, PCF/PCSS),
`fog.hlsli`.

## SDL3 GPU resource binding order

Bindings must follow SDL3 GPU's set layout, expressed with
`[[vk::binding(slot, set)]]`:

- **Vertex**: set 0 = sampled textures, then storage textures, then storage
  buffers; set 1 = uniform buffers.
- **Fragment**: set 2 = sampled textures, then storage textures, then storage
  buffers; set 3 = uniform buffers.

All vertex semantics are `TEXCOORD0..N` (SDL3 GPU's D3D12 backend
assumption).

## CPU↔GPU uniform contract

Every cbuffer here has an `alignas(16)` CPU mirror in
`src/engine/gpu/gpu_uniforms.hpp` annotated with its `(set, binding)`. The
layouts must match **byte-for-byte** — when adding or reordering cbuffer
fields, update both sides in the same change and mind HLSL packing rules
(16-byte registers; a `float3` followed by `float` packs, two `float3`s do
not).
