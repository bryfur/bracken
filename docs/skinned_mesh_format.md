# Skinned Mesh & glTF Format Contract

What the engine's model loader and animation system require from a `.glb` file.
This is the **engine-side contract** — how to *author/rig* a character (DCC tool
choices, auto-riggers, retargeting) is a host/asset-pipeline concern and lives
with whatever project consumes the engine, not here.

The reference implementation is `src/bracken/model_loader.cpp` (loading) and
`src/bracken/animation/` (playback).

## Format

- **glTF binary (`.glb`)**, **Y-up**, `+Y up` / `-Z forward` (standard glTF). A
  mesh exported with this convention loads with no rotation applied.
- Model bounds are **accumulated from every parsed vertex position** at load;
  the `POSITION` accessor's declared `min`/`max` are not read.
- **No Draco.** The engine's bundled tinygltf has no Draco decompression — never
  enable Draco mesh compression on the export, or the load fails.
- **No quantization / interleaving.** Everywhere the engine reads float data —
  vertex attributes, inverse bind matrices, animation sampler input/output —
  the accessor must be `FLOAT` componentType and tightly packed (no interleaved
  `byteStride`). Quantized-normalized exports (gltfpack/meshoptimizer u8/u16
  animation outputs) are valid core glTF but unsupported: the affected
  attribute / skin data / channel is **dropped with a warning**, never
  reinterpreted. Sparse accessors are likewise unsupported. `JOINTS_0` must be
  u8, `WEIGHTS_0` float.
- Materials and textures are loaded from the glb; PBR base-color + metallic-
  roughness + normal + emissive are honored. Keep `materials > 0` and
  `textures > 0` for a textured asset.
- Ambient occlusion must be **ORM-packed**: `occlusionTexture` must reference
  the *same image* as `metallicRoughnessTexture` (AO in the R channel). The
  model shaders have no separate AO sampler, so a standalone occlusion map is
  **dropped with a warning** at load.

## Skeleton & skinning

- **`MAX_BONES = 128`.** Joint count per skinned mesh must stay ≤ 128. This cap
  is a single CPU-side constant in
  `src/bracken/animation/animation_types.hpp:15` (`constexpr int MAX_BONES`); it
  sizes the per-instance bone-matrix arrays in `AnimationPlayer` and caps
  `FrameBuilder::alloc_bone_palette`'s joint count. At ≤ 128 joints the glb's joint indices
  fit in 8-bit integers. Don't raise it without auditing every
  `std::array<glm::mat4, MAX_BONES>` user.
- The bone palette is uploaded as a **per-frame StructuredBuffer SSBO** (vertex
  set 0, binding 0) holding tightly-packed per-entity palettes; the skinned
  shaders index it via a `paletteOffset` uniform rather than a fixed-size array.
  Because of this the **shaders carry no compile-time bone count** — there's no
  shader-side constant to keep in sync with `MAX_BONES`.
- **Joint order**: the animation/IK runtime requires parents-before-children
  joint order (every joint's `parent_index` less than its own index).
  `skin.joints` may be in any order in the glb — the loader topologically
  reorders out-of-order skins at load (a warning is logged and the vertex
  joint indices are remapped to match). A cyclic joint hierarchy fails the
  load with an error.
- A small amount of weight on a stray bone parked at the world origin (e.g. an
  exporter's `neutral_bone`) is harmless in practice (< ~0.2% of verts). Large
  leakage (> ~0.5%) shows as stretched triangles streaking from each instance
  toward a fixed point — fix it at export time by reweighting to the deform
  chain.
- Distinct from `MAX_BONES`: the renderer uploads at most
  `MAX_BONE_PALETTES = 1024` skinned instances per frame
  (`scene/scene_renderer.hpp`); draws beyond that are dropped and logged.

## Foot IK bone naming

Foot IK / body lean (`animation/ik_solver.hpp`) resolve bones by name via
`FootIKBoneNames` (`animation/animation_types.hpp`). The defaults are `Hips`,
`Spine`, `LeftUpperLeg`/`LeftLowerLeg`/`LeftFoot` and the `Right*` mirrors —
rigs using other names (e.g. raw Mixamo `mixamorig:` prefixes) work fine, but
the host must supply a matching `FootIKBoneNames` instead of the defaults.

## Animation channels

- Animations are matched by **glTF channel target node → joint index** lookup.
  Channels that target nodes **outside the skin's joint list are silently
  dropped** — if a clip seems to "not play," verify its channels drive the
  skinned joints.
- **Interpolation modes**: `LINEAR` and `STEP` samplers are honored
  (`STEP` holds each key's value until the next key — Blender "Constant"
  keyframes stay constant). `CUBICSPLINE` is **not implemented**: the loader
  keeps each key's in-place value, ignores the tangents, and plays the track
  back with linear interpolation (one warning per clip) — poses at the keys
  are exact, in-between frames lose the authored easing. Export with
  linear/step keys for exact playback.
- Each clip should have real frame-to-frame variation (not constant keyframes);
  a clip whose every frame is identical produces a static pose.
- **Clip names are matched by the state machine**, by string. The engine's
  `AnimationStateMachine` is data-driven — the host supplies the state/transition
  config and the clip names it expects (commonly `Idle`, `Walk`, `Run`,
  `Attack`). Name your clips to match whatever config the host loads.

## Ground placement

- The engine offsets a character so its lowest vertex sits on the ground using
  the model's `min_y`. Author the mesh with **feet at the origin plane** in the
  source scene so this foot offset lands the character correctly on terrain.

## Checklist before shipping a skinned glb

- [ ] `.glb`, Y-up, `+Y up`/`-Z forward`
- [ ] **No Draco**, no quantized (non-float) accessors, no interleaved strides
- [ ] Joint count ≤ 128
- [ ] Animation samplers use `LINEAR`/`STEP` (no `CUBICSPLINE` easing)
- [ ] AO (if any) packed into the metallic-roughness image's R channel (ORM)
- [ ] Negligible weight on any origin-parked `neutral_bone` (< ~0.2% of verts)
- [ ] Animation clips named to match the host's state-machine config
- [ ] Each clip has non-constant frames (`np.diff(quats, axis=0).max() > 0`)
- [ ] Materials and textures preserved (`materials > 0`, `textures > 0`)
- [ ] Feet at the origin plane in the source scene
- [ ] Loads with `bracken_tests` green and the model_loader logging the
      expected joint/animation counts
