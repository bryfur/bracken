# Engine Architecture Reference

The detailed map of how the engine fits together at runtime. For the short
orientation see [CLAUDE.md](../CLAUDE.md) / [README.md](../README.md); for
design rationale ("why is X off by default") see
[engine_design_notes.md](engine_design_notes.md).

Everything in this file is verified against the code it cites. If you change
one of the cited facts, update this file in the same change.

---

## Module map

Include root is `src/`; headers are included as `engine/<module>/<file>.hpp`.
Root namespace `mmo::engine`, one sub-namespace per module.

| Path | Namespace | Contents |
|---|---|---|
| `src/engine/` (root) | `mmo::engine` | `Application`, `ModelLoader`/`ModelManager`, `model_utils`, `InputHandler`/`InputState`, `GraphicsSettings`, `Heightmap`, `RenderStats`, `effect_definition.hpp` |
| `core/` | `core` | `Logger` (+`ENGINE_LOG_*`), `assert.hpp`, `Result<T,E>`, `Handle` pools, `hash.hpp` (`StringId`, FNV-1a), `profiler.hpp` (Tracy macros) |
| `core/jobs/` | `core::jobs` | `JobSystem` (work-stealing singleton), `parallel_for`, `TaskHandle` |
| `core/asset/` | `core::asset` | `AssetRegistry` (typed, stable handles, hot-reload), `FileWatcher` |
| `memory/` | `memory` | `Arena` (bump + destructor registration), `FrameAllocator` |
| `gpu/` | `gpu` | `GPUDevice`, `GPUBuffer`, `GPUTexture`, `ShaderManager`, `GPUPipeline`, `PipelineRegistry`, `gpu_uniforms.hpp` (CPU↔GPU contract), `gpu_debug.hpp` |
| `render/` | `render` | `RenderContext`, sub-renderers: world/skybox, terrain, grass, shadow map (CSM), SSAO/GTAO, bloom, volumetric fog, effects, text, UI; `environment.hpp` |
| `render/lighting/` | `render::lighting` | `PointLight`/`SpotLight`, `ClusterGrid` (CPU light binning) |
| `render_graph/` | `render_graph` | pass scheduler (topo-sort, transient pool) — **in-tree, tested, not wired into `SceneRenderer`** |
| `scene/` | `scene` | `RenderScene` (command buffer), `SceneRenderer` (frame orchestrator), `CameraState`, `Frustum`, `transform_math.hpp`, `UIScene` |
| `animation/` | `animation` | `animation_types.hpp` (skeleton/clips/constants), `AnimationPlayer`, `AnimationStateMachine`, `ik_solver` |
| `systems/` | `systems` | `CameraController` / `CameraSystem`, `EffectSystem` |
| `procedural/` | `procedural` | `tree_generator` (ez-tree algorithm), `rock_generator` (SDF + marching cubes) |
| `shaders/` | — | HLSL sources → SPIR-V at build time; see [shaders/README.md](../src/engine/shaders/README.md) |

---

## Coordinate, depth, and color conventions

- **Y-up**, right-handed: x/z is the ground plane, y is height. glTF assets are
  `+Y up` / `-Z forward` (standard glTF), loaded with no rotation.
- **Reversed-Z, zero-to-one depth**: `GLM_FORCE_DEPTH_ZERO_TO_ONE` is a PUBLIC
  compile definition on the `mmo_engine` target (`src/engine/CMakeLists.txt`) —
  it propagates to consumers. All projection math assumes glm `RH_ZO`, and the
  scene projection is **reversed-Z**: near/far are swapped in the
  `glm::perspective` call (`systems/camera_system.cpp`), so the near plane maps
  to device depth **1** and the far plane to **0**. This spreads float32 depth
  precision near-uniformly over MMO-scale view distances instead of burning it
  all within a few meters of the camera. Consequences, wired through the code:
  - Scene depth targets clear to **0.0** (`scene_renderer.cpp`,
    `ambient_occlusion.cpp`); the `PipelineConfig` default depth compare is
    `GREATER_OR_EQUAL` (`gpu/gpu_pipeline.hpp`).
  - The skybox draws at z = 0.0 with `GREATER_OR_EQUAL` (`skybox.vert.hlsl` +
    the skybox pipeline), passing only where nothing was drawn.
  - Depth-reading shaders treat device depth **0.0 as sky/far**
    (`ssao`/`gtao`/`blur_ao`/`volumetric_fog`). The linearization
    `viewZ = |B / (depth + A)|` with `A = proj[2][2]`, `B = proj[3][2]` holds
    for standard and reversed matrices alike, so `blur_ao`/`composite` needed
    no math change.
  - `SceneRenderer` recovers the near plane as `m[3][2]/(m[2][2] + 1)`
    (under standard-Z it was `m[3][2]/m[2][2]`).
  - `Frustum::extract_from_matrix` is convention-agnostic: the two depth-plane
    rows swap near/far roles under reversed-Z but the extracted plane set (and
    thus culling) is identical (`scene/frustum.cpp`).
  - **Exception: shadow cascades stay standard-Z** (clear 1.0,
    `LESS_OR_EQUAL` pipelines + compare sampler, `orthoRH_ZO` light
    projections). Their orthographic depth is already linear, so reversed-Z
    buys nothing, and all receiver bias/PCF/PCSS math assumes near=0/far=1.
  Never mix in `-1..1`-depth projection helpers, and never build a
  depth-tested scene pass with a standard (non-reversed) projection.
- **Linear-space lighting**: all colors in `render::EnvironmentState`
  (`render/environment.hpp`) are linear. The scene renders into a linear HDR
  target; the composite pass tonemaps (ACES et al.) + gamma-corrects to the
  swapchain. sRGB textures are converted on sample.
- **Texture formats** (`gpu/pipeline_registry.hpp`): swapchain
  `R8G8B8A8_UNORM`, HDR scene color `R16G16B16A16_FLOAT`, depth `D32_FLOAT`.

---

## `Application` lifecycle

`Application` (`src/engine/application.hpp`) owns SDL, the main loop, frame
timing, input, and the renderer facade. Consumers subclass it:

- Pure virtual: `on_init()`, `on_update(float dt)`, `on_render()`.
- Optional: `on_shutdown()`, `on_event(const SDL_Event&) -> bool` (return
  `true` to consume an event before `InputHandler` sees it).
- Boot sequence: `init_engine()` → `init_renderer(width, height, title,
  world_width = 32000, world_height = 32000)` → `run()` →
  `shutdown_renderer()` → `shutdown_engine()`. `quit()` stops the loop.

The main loop (`application.cpp::run()`): nanosecond `dt` from
`SDL_GetTicksNS()` clamped to 0.1 s; `FileWatcher::poll_main_thread()` when
hot-reload is on; input edge-clearing + `SDL_PollEvent` (quit → stop, else
`on_event` then `InputHandler::process_event`); `on_update(dt)`;
`on_render()`; `ENGINE_PROFILE_FRAME()`.

Side effects worth knowing:
- `init_renderer` auto-loads `GraphicsSettings` from `settings_file_path()`
  (platform pref path) and `shutdown_renderer` auto-saves them.
- If the env var `MMO_HOT_RELOAD=1` is set, `init_engine` starts the global
  `FileWatcher` and `init_renderer` wires model + shader-SPIR-V hot-reload to
  it (`application.cpp`). Default: off, no watcher thread.

---

## Frame data flow

The host builds descriptions; the engine consumes them:

```
host on_update:   AnimationStateMachine::update → AnimationPlayer::update
                  → (optional) apply_foot_ik / apply_body_lean
host on_render:   populate scene::RenderScene + scene::UIScene
                  → Application::render_frame(scene, ui, camera, dt)
                  → scene::SceneRenderer::render_frame(...)
```

`RenderScene` (`scene/render_scene.hpp`) accumulates: static model commands,
skinned model commands, instanced batches, point/spot lights, billboards,
debug lines, particle-effect spawn commands, and per-frame draw toggles
(skybox/ground/grass/…). It is passed to `render_frame` by **non-const
reference** — the renderer drains the particle-spawn queue.

**Lifetime contract**: `SkinnedModelCommand::bone_matrices` is a *non-owning*
`std::span<const glm::mat4>` over host storage (e.g. the host's
`AnimationPlayer::bone_matrices()`). That storage must stay valid and
unmodified until `render_frame` returns.

### Pass order inside `SceneRenderer::render_frame`

(`scene/scene_renderer.cpp`; each GPU section has a matching debug group +
timer label, so RenderDoc captures read in this order.)

1. **CPU prep** — propagate `EnvironmentState` to sub-renderers, advance sky
   and effect-system time, drain `RenderScene` particle spawns, pre-upload any
   effect models (uploads can't run inside a render pass), update CSM cascade
   matrices, cull + build instance batches per cascade and main view, resolve
   skinned commands (model, bounds, deduped palette offset) and plan the
   palette copy ranges, build the cluster light grid on the CPU (skipped when
   the scene has zero dynamic lights).
2. **`FrameUploads`** — one `SDL_GPUCopyPass` for every per-frame upload:
   instances, shadow instances, bone palettes, particle instances, debug
   lines, grass chunks, cluster light buffers.
3. **`ShadowPass`** — cascades rendered into the 2×2-quadrant D32 atlas
   (`ShadowPass:Cascade0..3`), only when `shadow_mode != ShadowMode::Off`
   (settings modes are `enum class`es in `graphics_settings.hpp`; the on-disk
   settings format stays numeric).
4. **`AOOffscreen`** (the normal path whenever the AO subsystem initialized) —
   the whole 3D scene renders into the linear-HDR offscreen target: terrain,
   grass, static/instanced/skinned models, skybox, sorted transparent
   (alphaMode `BLEND`) meshes, particles, debug lines.
   The offscreen clear color tracks `env.fog_color_horizon`.
5. **Post chain** (all offscreen, swapchain not yet acquired):
   `SSAO`/`GTAO` (per `ao_mode`; despite the setting name, the "GTAO" mode is
   an HBAO-style directional variant — no horizon search/arc integration) →
   `BlurAO` (bilateral) → `Bloom` (13-tap downsample / 9-tap tent upsample;
   the prefilter thresholds post-exposure luminance) → `VolumetricFog` (ray
   march against depth + shadow atlas; all parameters come from
   `EnvironmentState` — sun_dir, fog_color_horizon, and the
   `fog_volumetric_*` fields; god rays select a cascade per march sample
   and sky pixels are marched too, so shafts silhouette against the sky;
   god rays force-disabled when shadows are off because the atlas would
   contain garbage).
6. **Late swapchain acquire** → **`Composite`** (depth-aware joint-bilateral
   upsample of the half-res AO + fog, scene × AO + bloom + fog,
   exposure/tonemap/contrast/saturation). With `aa_mode == 1` composite goes
   to an LDR intermediate and **`FXAA`** resolves to the swapchain; otherwise
   composite writes the swapchain directly.
7. **`UI`** — `UIScene` commands + 3D billboards, straight to swapchain.
   **`PostUICallback`** — optional host hook that renders into the swapchain
   after UI (`SceneRenderer::set_post_ui_callback`).
8. Fallback: if the AO subsystem isn't ready, step 4–6 collapse to a single
   **`MainScene`** pass direct to swapchain. Minimized windows skip all
   recording but still submit, keeping the frame lifecycle balanced.
9. **Screenshot readback** (only when `request_screenshot(path)` was called
   since the last frame; `Application` forwards to
   `SceneRenderer::request_screenshot`): after UI/PostUICallback, a copy pass
   downloads the swapchain texture into a `DOWNLOAD` transfer buffer, the
   frame submits via `RenderContext::end_frame_and_wait()`
   (`SDL_SubmitGPUCommandBufferAndAcquireFence` + fence wait — a deliberate
   pipeline stall, so tests/debug only), and the mapped pixels are saved as
   PNG (`IMG_SavePNG`, alpha dropped via RGB24 conversion; BGRA/RGBA
   swapchain formats both handled). This is the engine's only GPU→CPU
   readback path.

### Frustum culling

`scene/frustum.hpp` extracts planes from `view_projection`. Culling runs on
the CPU for the main view and independently per shadow cascade during
instance-batch building. Toggle via `GraphicsSettings::frustum_culling`.

---

## GPU layer

- **`PipelineRegistry`** (`gpu/pipeline_registry.hpp`) lazily creates and
  caches one `GPUPipeline` per `PipelineType` (29 types, including the
  model-family `DoubleSided`/`Blend` material variants). Scene pipelines
  target the HDR format; UI/Text/Composite/Fxaa target the swapchain
  format and are the only ones invalidated when the swapchain format changes.
  `preload_all_pipelines()` avoids first-use hitches.
  The full pipeline → shader-pair table lives in
  [shaders/README.md](../src/engine/shaders/README.md).
- **Uniform contract**: every HLSL cbuffer has an `alignas(16)` CPU mirror in
  `gpu/gpu_uniforms.hpp`, annotated with its `(set, binding)`. They must match
  **byte-for-byte** — this is the most common way to break rendering silently.
  `ClusterParams` is intentionally duplicated between
  `render/lighting/light_cluster.hpp` and `gpu_uniforms.hpp` to avoid a header
  dependency; keep both in sync.
- **Skinned palettes**: `SceneRenderer` packs every visible skinned instance's
  bone matrices into a single per-frame `StructuredBuffer<float4x4>` (vertex
  set 0, binding 0); the skinned vertex shader indexes it with a
  `paletteOffset` uniform. There is **no GPU-side bone-count constant**.
- **Debug tooling**: `ENGINE_GPU_SCOPE` (SDL debug groups) labels every pass.
  `GPUTimestampPool` / `ENGINE_GPU_TIMER` exist but are **stubs** — SDL3 has
  no per-pass timestamp queries, so `read_results` returns zeroed durations
  and any per-pass GPU times in `RenderStats` are placeholders
  (`gpu/gpu_debug.hpp`). Use RenderDoc/Nsight for real GPU timing.

---

## Assets & models

- **`ModelLoader`** (`src/engine/model_loader.hpp`): static
  `load_glb` / `upload_to_gpu` / `free_gpu_resources`. glTF binary only, no
  Draco (bundled tinygltf has no decoder). Bounds come from the POSITION
  accessor min/max. See [skinned_mesh_format.md](skinned_mesh_format.md) for
  the authoring contract.
- **`ModelManager`**: name → `ModelHandle` (`uint32_t`, `0` =
  `INVALID_MODEL_HANDLE`) registry. `load_model(name, path)`,
  `get_model(handle|name)`, `register_model(name, unique_ptr<Model>)` for
  procedural meshes, `reload_model(handle)`, `enable_hot_reload(watcher)`.
- **Mesh flags** (`model_loader.hpp`): `cast_shadows`, `shadow_only`,
  `near_only` (declared but **not yet enforced** by the renderer),
  `has_packed_occlusion` (ORM-packed occlusion in the metallic-roughness
  texture).
- **Material alpha/cull flags** (`PBRMaterial`): glTF `alphaMode` /
  `alphaCutoff` / `doubleSided` are honored by the renderer. `OPAQUE`
  materials never alpha-test (base-color alpha is ignored, per spec); `MASK`
  discards below the material's cutoff (`alphaCutoff` uniform); `BLEND`
  meshes are drawn in a sorted transparent pass after all opaque content and
  the skybox — back-to-front by camera distance, depth test on, depth writes
  off. `doubleSided` selects a no-cull pipeline variant. Limitations: no OIT,
  and the sort granularity is one mesh per submitted command/instance (using
  the model's bounding-sphere center), so intersecting or mutually enclosing
  transparent geometry can still sort wrong. `BLEND` meshes inside instanced
  batches fall back to one non-instanced draw per instance.
- **`model_utils`**: `build_model_transform` (translation/rotation/scale →
  mat4 with engine conventions), `compute_tangents_from_uvs` (for meshes
  missing tangents).

---

## Animation

Per-instance, host-driven. Types and constants in
`animation/animation_types.hpp`:

- `MAX_BONES = 128` (line 15) — CPU-side cap on joints per skinned mesh; sizes
  `std::array<glm::mat4, MAX_BONES>` in `AnimationPlayer` (the scene layer
  passes bone matrices as a length-carrying `std::span`).
  `MAX_BONE_INFLUENCES = 4`.
- `AnimationPlayer` (`animation_player.hpp`): `update(skeleton, clips, dt)`,
  `crossfade_to(clip_index, duration)`, speed/loop/seek. Keyframe cursors are
  per-player (clips are shared, players are not — thread-hazard lesson, see
  design notes).
- `AnimationStateMachine` (`animation_state_machine.hpp`): data-driven states
  + transitions; the host feeds parameters (`set_float("speed", …)`,
  `set_bool("attacking", …)` — or, on per-frame paths, resolve the name once
  with `param_index()` and use the int-indexed `set_float`/`set_bool`
  overloads to skip the per-call string hash). Conditions are a tagged union
  supporting AND and `any_of` (OR). Clip binding is by **string name** → see
  the format doc's clip-naming section.
- **IK is applied by the host after the state-machine/player update**, before
  render submission: `solve_two_bone_ik`, `apply_foot_ik`, `apply_body_lean`
  (`ik_solver.hpp`). `FootIKBoneNames` defaults to the standard rig names
  `Hips`, `Spine`, `Left/RightUpperLeg`, `Left/RightLowerLeg`,
  `Left/RightFoot` — override per rig.
- glTF channels that target nodes outside the skin's joint list are
  **silently dropped** at load; a clip that "doesn't play" usually means its
  channels don't drive skinned joints.

---

## Threading & lifetime contracts

The engine is single-threaded by design; the exceptions are explicit. The
main-thread contracts are asserted, not just documented:
`core::bind_main_thread()` runs in `Application::init_engine`, and
`ENGINE_ASSERT_MAIN_THREAD()` (`core/assert.hpp`) guards the boundaries below
(FileWatcher entry points, swapchain acquire, `render_frame`) in debug builds.

- **`FileWatcher`** (`core/asset/file_watcher.hpp`): polls on a background
  `std::jthread`, but callbacks fire only from `poll_main_thread()` on the
  main loop — watcher callbacks may therefore safely touch GPU state.
- **`JobSystem`** (`core/jobs/job_system.hpp`): work-stealing pool, exists and
  is tested, but the engine issues no parallel work by default. Keep it that
  way unless a measured workload justifies it (see
  [engine_design_notes.md](engine_design_notes.md) — "infrastructure without
  load"). `parallel_for` has a serial path below its grain threshold but still
  costs dispatch overhead.
- **`Logger`** sinks are thread-safe.
- **`render_frame` mutates the `RenderScene`** (drains particle spawns), and
  skinned-command bone-matrix spans must outlive the call (see above).
- GPU resource owners (`PipelineRegistry`, buffers, textures) are
  non-copyable/non-movable and must be destroyed before the `GPUDevice`.

---

## Load-bearing constants

| Constant | Value | Where | Caps |
|---|---|---|---|
| `MAX_BONES` | 128 | `animation/animation_types.hpp:15` | joints per skinned mesh; CPU palette arrays |
| `MAX_BONE_PALETTES` | 1024 | `scene/scene_renderer.hpp` | skinned draws uploaded per frame — **excess draws are dropped and logged**, not an error |
| `CSM_MAX_CASCADES` | 4 | `render/shadow_map.hpp` | cascades in the 2×2 D32 shadow atlas |
| `CLUSTER_DIM_X/Y/Z` | 16×9×24 | `render/lighting/light.hpp` | cluster grid resolution |
| `MAX_LIGHTS` | 256 | `render/lighting/light.hpp` | dynamic point+spot lights per frame |
| `MAX_LIGHTS_PER_CLUSTER` | 128 | `render/lighting/light.hpp` | lights binned into one cluster |
| default world size | 32000 × 32000 | `application.hpp::init_renderer` | terrain/grass extent defaults |
| swapchain / scene / depth formats | RGBA8 / RGBA16F / D32 | `gpu/pipeline_registry.hpp` | see GPU layer above |

`shadowEnabled` in `ShadowDataUniforms` is a mode, not a bool: `0` off,
`1` PCF, `2` PCSS (`gpu/gpu_uniforms.hpp`).

---

## Known stubs & silent behaviors

Things that look functional but aren't, or fail quietly — check here before
debugging "why is this value zero / where did my draw go":

- **GPU per-pass timings are zeroed** — `GPUTimestampPool` is a stub (SDL3
  limitation). Wall-clock CPU timings and Tracy zones are real; GPU pass times
  in `RenderStats` are not.
- **Skinned draws beyond `MAX_BONE_PALETTES` (1024) are dropped** with a
  logged error, not asserted.
- **Animation channels targeting non-skin nodes are silently dropped** at glb
  load.
- **`Mesh::near_only` is not enforced** — the flag exists, the renderer
  ignores it.
- **`EmitterDefinition::particle_type` "sprite" is not implemented** — only
  `"mesh"` particles render (`render/effect_renderer.cpp`).
- **Transparency has no OIT** — glTF `BLEND` meshes are sorted back-to-front
  per queued draw (one entry per mesh per command/instance, keyed on the
  model's bounding-sphere center); overlapping transparent geometry can still
  blend in the wrong order. Shadow passes still alpha-test `BLEND` casters at
  0.5 rather than rendering translucent shadows.
- **Cluster light SSBOs keep stale contents** when a frame has zero dynamic
  lights — correctness is preserved by zeroed `ClusterParams`
  (`gridDim.w == 0` short-circuits the shader path), not by clearing buffers.
- **`render_graph/` is not in the frame path** — `SceneRenderer` issues passes
  directly. The graph is kept (with its tests) for when pass count or
  transient aliasing justifies it.
