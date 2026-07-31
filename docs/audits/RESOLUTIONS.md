# Audit Resolutions — 2026-07-05

> The engine was renamed **mmo_engine → Bracken** after this campaign; the
> four dated audit reports are unmodified historical records, so their file
> paths (`src/engine/...`), namespaces (`mmo::engine`), and `MMO_*`/`ENGINE_*`
> identifiers refer to the pre-rename layout.

Outcome of the fix campaign for the four 2026-07-03 audits. Executed as
eight parallel worktree branches (merged to master) plus two sequential
main-tree passes (reversed-Z, modernization). Every wave was verified with a
full build, the unit suite, and the playground smoke run + screenshot
self-check before merging/committing.

**Baseline → end state:** 599 → 625 tests · smoke 102 → 159 fps ·
cluster binning 75.3 ms → 2.77 ms @ 256 lights (unoptimized build, same
flags) · one draw per (emitter, mesh) instead of one per particle · glyph
atlas instead of one texture per string · reversed-Z depth · glTF
alphaMode/doubleSided honored.

Legend: ✅ fixed · 🟡 partially fixed (remainder listed under Follow-ups) ·
⏸️ deferred (reason given).

## performance.md

| Finding | Status | Notes |
|---|---|---|
| H1 per-particle draw + uniform re-push | ✅ | Instanced: per-frame `StructuredBuffer<ParticleInstance>` + one draw per (emitter, mesh); new `particle.vert.hlsl` |
| H2 O(clusters × lights) binning | ✅ | Inverted to per-light cluster-range insertion; 75.3 → 2.77 ms @ 256 lights, 18.7 → 0.96 ms @ 64 |
| M1 texture-per-string text | ✅ | 512² shelf-packed glyph atlas, dirty-band uploads, run-merged draws |
| M2 triple-pass instance upload | ✅ | `GPUBuffer::Mapping` RAII; packs straight into mapped transfer buffers |
| M3 AnimationPlayer identity tail-fill | ✅ | `used_joint_count_` prefix tracking |
| M4 O(N²) palette dedupe | ✅ | `BonePaletteTable` open-addressing hash, allocation-free |
| M5 per-frame param-name hashing | ✅ | `param_index()` resolution at bind time; string API kept as wrapper |
| M6 UI text string copies | 🟡 | Text stack copies eliminated; `ui_scene.cpp` host-string copy remains |
| L1 per-particle sqrt/invariant checks | 🟡 | sqrt zero-tests removed; full loop-invariant hoist deferred (restructure) |
| L2 27.6 KB FNV dirty-check | ✅ | Dirty flag computed during pack + all-zero fast path |
| L3 per-frame `glm::inverse(projection)` | ✅ | Cached, keyed on matrix change |
| L4 linear terrain-tile culling | ✅ | Row metadata early-out |
| L5 per-frame `std::string` pass names | 🟡 | Timestamp pool interned; `RenderStats::pass_times_ms` still `vector<pair<string,float>>` |
| L6 double `Model*` resolution | ✅ | `ResolvedSkinnedEntry` shared by main view + cascades |
| L7 136-byte AoS `Particle` | ⏸️ | SoA restructure; low severity — needs profiling evidence to justify the churn |

## graphics.md

| Finding | Status | Notes |
|---|---|---|
| H1 no reversed-Z | ✅ | near→1/far→0, `GREATER_OR_EQUAL`, clear 0, skybox z=0; shadow cascades deliberately standard-Z (orthographic depth is linear); depth-precision regression test added |
| H2 alpha modes/doubleSided ignored | 🟡 | MASK cutoff uniform, 7 cull/blend pipeline variants, sorted back-to-front BLEND pass; shadow passes still alpha-test at fixed 0.5 (BLEND casters get cutout shadows) |
| H3 AO darkens final lit color | ⏸️ | Correct fix needs a depth prepass + AO in the forward ambient term — architecture change, tracked as roadmap |
| H4 fog ignores EnvironmentState | ✅ | Sun/fog colors + `fog_volumetric_*` fields plumbed; dead near/far uniforms deleted |
| M1 god rays cascade-0-only, no sky | ✅ | Cascade per march sample; sky pixels marched |
| M2 no TAA / motion vectors | ⏸️ | Feature roadmap, not a defect |
| M3 no IBL | ⏸️ | Feature roadmap |
| M4 no point/spot shadows | ⏸️ | Feature roadmap |
| M5 bilinear upsample halos | ✅ | Joint-bilateral (depth-aware) upsample of AO + fog in composite |
| M6 no depth prepass | ⏸️ | With H3 |
| L1 "GTAO" is not GTAO | ✅ | Naming honesty in docs/comments (settings enum unchanged) |
| L2 bloom threshold pre-exposure | ✅ | Thresholds post-exposure luminance |
| L3 hardcoded fog dead zone/cap | ✅ | EnvironmentState parameters, defaults preserve visuals |
| L4 no HDR display output | ⏸️ | Feature roadmap |
| L5 UI blends in gamma space | ⏸️ | Needs sRGB swapchain rework; visually minor |
| L6 `gtao_direction_kernel` latch | ✅ | Regenerates when the setting changes |

## modernization.md

| Finding | Status | Notes |
|---|---|---|
| H1 stale `.clang-tidy` header filter | ✅ | Points at `src/bracken`; header diagnostics live again |
| H2 `get_draw_distance` OOB | ✅ | Clamped + `clamp_to_valid_ranges()` after `load()` + corrupt-file tests |
| H3 `Result` unused, two conventions | ✅ | `[[nodiscard]]` Result + `Result<void,E>`; policy written into CLAUDE.md (no mass conversion, by design) |
| M1 zero `[[nodiscard]]` | ✅ | Sweep over fallible bools + factories |
| M2 unasserted threading contracts | ✅ | `core::bind_main_thread()` + `BRACKEN_ASSERT_MAIN_THREAD()` at documented boundaries |
| M3 rule-of-five gaps | ✅ | Explicit deleted moves on resource owners |
| M4 raw bone-matrix pointer | ✅ | `std::span<const glm::mat4>` (done in the perf wave) |
| M5 `SDL.h` umbrella includes | ✅ | Narrowed to specific SDL3 headers |
| M6 raw-int settings modes | ✅ | `enum class` AOMode/ShadowMode/AAMode/TonemapMode; on-disk format still numeric |
| M7 inline settings file I/O | ✅ | Moved to `graphics_settings.cpp` |
| M8 hardcoded 4 cascades | ✅ | `CSM_MAX_CASCADES` everywhere |
| L1 string-keyed frame lookups | ✅ | State machine fixed in perf wave; remaining cited sites addressed |
| L2 singleton surface | ⏸️ | Deliberate; design note added instead of restructuring |
| L3 gameplay-named input accessors | ✅ | `controller_attack/sprint` → `controller_primary/secondary` |
| L4 copyable polymorphic `LogSink` | ✅ | Copy ops protected |
| L5 non-engine-rooted includes | ✅ | Fixed cited files |
| L6 C array in InputHandler | ✅ | `std::array` |
| L7 `unique_ptr(new T)` idiom | — | Observation only (audit recommended no action) |

## shaders.md

| Finding | Status | Notes |
|---|---|---|
| H-1 cluster lookup Y-flip | ✅ | `1.0 - uv.y` + orientation contract pinned by CPU-side test |
| H-2 cbuffer array padding ≠ CPU struct | ✅ | Scalars replace `[3]` arrays; `ShadowFragUniforms` mirror + layout tests |
| M-1 hardcoded 256 payload offsets | ✅ | Capacity flows through `ClusterParams::maxPerCluster.y` |
| M-2 unclamped cluster loop | ✅ | Clamped to `maxPerCluster.x` (uniform now live) |
| M-3 dead `fogDistance` varying | ✅ | Removed from model family + particle |
| M-4 near-plane-dependent sky skip | ✅ | Explicit far test (now `depth <= 0.0` under reversed-Z) |
| M-5 missing layout tests | ✅ | static_asserts incl. the `GrassVertexUniforms` offset-148 knife-edge |
| L-1 dead uniforms | 🟡 | fog near/far deleted, `maxPerCluster` made live; `useSkinning` remains (needs lockstep model/skinned vert + CPU struct edit) |
| L-2 orphaned billboard pipeline | ✅ | Deleted (enum, factory, both shaders); docs updated |
| L-3 helper duplication | 🟡 | `shadowIGN` deduped into `lighting.hlsli`; IGN/depth-reconstruction copies in ssao/gtao/fog remain |
| L-4 sin-hash degrades at world scale | ✅ | Integer (murmur-finalizer) hash |
| L-5 normalize(0) edges | 🟡 | grass curve-dir + pbr half-vector guarded; ssao `clipPos.w <= 0` guard remains |
| L-6 PCSS double cost in blend band | ✅ | IGN dither to a single filtered cascade |
| L-7 full filter where sun is zero | ✅ | Single-tap path when NdotL <= 0 |
| L-8 cosmetic | ✅ | |

## Follow-ups (small, non-blocking)

1. Remove the dead `useSkinning` uniform (lockstep: `model.vert` +
   `skinned_model.vert` + `ModelTransformUniforms` + layout test).
2. Dedupe `interleavedGradientNoise` + depth reconstruction across
   `ssao`/`gtao`/`volumetric_fog` into a shared `.hlsli`.
3. Guard `clipPos.w <= 0` sample rejection in `ssao.frag`.
4. Intern `RenderStats::pass_times_ms` names (drop per-frame strings).
5. Eliminate the remaining `ui_scene.cpp` text-command string copy.
6. Shadow-pass alpha: respect per-material cutoff for MASK casters instead
   of the fixed 0.5 (BLEND casters keep cutout shadows until then).
7. `GTAOUniforms::maxDistance` (12000) was tuned to match the old hidden
   0.9999 cutoff; reversed-Z far-field precision allows revisiting it.

Deferred **roadmap** items (from graphics.md, in the audit's value order):
depth prepass + AO-in-ambient (H3/M6), TAA + motion vectors (M2), IBL (M3),
point/spot shadows (M4), HDR display output (L4), sRGB-correct UI blending
(L5), SoA particles (perf L7).

# Audit Resolutions — 2026-07-12

Outcome of the fix campaign for the 2026-07-11 full-source documentation
sweep ([documentation_sweep.md](documentation_sweep.md)). Executed as four
parallel-agent waves plus a docs-reconciliation wave on the main tree; every
wave was verified with an incremental build, the full unit suite, and the
playground smoke run + screenshot self-check.

**Baseline → end state:** 631 → 774 tests · smoke stable ~155–162 fps ·
all 127 findings (3 high, 34 medium, 90 low) addressed. Pre-1.0 policy
applied throughout: dead/vestigial API was deleted rather than preserved.

Legend: ✅ fixed · 🟡 partially fixed (remainder noted) · ⏸️ deliberate
no-change (reason given).

## documentation_sweep.md

| Finding | Status | Notes |
|---|---|---|
| H1 `add_subdirectory` consumption path fails | ✅ | PUBLIC include dir is now `${PROJECT_SOURCE_DIR}/src`; the documented snippet works verbatim |
| H2 nested submit+wait livelock | ✅ | `wait()` drains its local queue, steals, then parks — fork-join safe on any pool size |
| H3 CONTINUOUS `spawn_rate` unvalidated | ✅ | Zero/negative spawn rates validated; no hang, no silent single particle |
| M1 grass "town" clearing hardcoded | ✅ | Game-specific clearing removed; host-configurable `set_exclusion_rects` + edge margin |
| M2 `BUILD_SHARED_LIBS`/PIC force-cached | ✅ | Plain `set()`s scoped to the engine tree; nothing leaks into super-project caches |
| M3 system `shadercross` preferred over pinned | ✅ | Always uses the pinned `$<TARGET_FILE:shadercross>` |
| M4 menu keys keycode-matched | ✅ | Scancode-matched, same convention as movement |
| M5 draw flags false at construction, true after `clear()` | ✅ | Construction and `clear()` agree — fresh-scene trap removed |
| M6 three packed-color conventions | ✅ | Single ARGB `0xAARRGGBB` via new `core/color.hpp` `pack_color`/`unpack_color` |
| M7 Trace/Debug suppressed by SDL priority filter | ✅ | SDL log priorities opened at `Logger` construction |
| M8 `enable_hot_reload` latches uninitialized watcher | ✅ | No-op with honest logging when the FileWatcher is uninitialized |
| M9 `ShaderResources` never validated | ✅ | Validated against SPIR-V reflection at load; mismatch logs an error, reflected counts win |
| M10 `ShaderManager::get` ignores resources on cache hit | ✅ | Cache hits verify declared counts |
| M11 separate occlusion textures vestigial | ✅ | Standalone (non-ORM) occlusion now warned + dropped at load; `occlusion_texture`/`occlusion_pixels` deleted |
| M12 sampler interpolation ignored | ✅ | STEP honored; CUBICSPLINE degrades to middle element with a warning |
| M13 IBM/sampler accessors read raw | ✅ | componentType/stride/sparse validation added |
| M14 joint order undocumented, unvalidated | ✅ | Joint order topologically normalized at load |
| M15 `apply_foot_ik` pelvis-drop wrong for non-translation IBMs | ✅ | Incremental bone-matrix update corrected |
| M16 base-color texture raw glTF image index | ✅ | Correct compacted-vector index mapping |
| M17 `Model*` stability broken by reload | ✅ | `reload_model` replaces contents in place; raw `Model*` stable across reload |
| M18 tree leaves opaque single-sided | ✅ | Textured leaves alpha-masked + double-sided |
| M19 failed tree texture → all-white tree | ✅ | Load failure warns and keeps vertex colors |
| M20 `TreeParams::levels` UB | ✅ | Clamped with warning |
| M21 splatmap UVs assume square origin-anchored world | ✅ | Splat UVs derived from actual world extents (`splat_params`; `TerrainLightingUniforms` 112 → 128 B) |
| M22 grass splatmap pointer dangles | ✅ | Provider-based splatmap access survives texture recreation |
| M23 fullscreen no-match silently mis-records | ✅ | Warns, clears stale mode, records the borderless fallback |
| M24 default `Handle{}` valid | ✅ | `Handle{}` now `== Handle::invalid()` (index `0xFFFFFFFF`), matching `render_graph::ResourceHandle` |
| M25 `AssetRegistry` duplicate-name orphan/erase | ✅ | `load()` auto-replaces with warning; stale unload is safe |
| M26 emitter→model cache stores failures forever | ✅ | Failed lookups no longer cached permanently; late-registered models render |
| M27 UIRenderer batch overflow draws stale buffers | ✅ | Batch capacity grows; no stale-GPU-buffer draws |
| M28 legacy `draw_text` illegal copy pass | ✅ | Legacy `draw_text*`/`prepare_text_textures` deleted |
| M29 `create_2d_array` returns empty texture on upload failure | ✅ | Returns nullptr as promised (also fixed a pre-existing double-release in the failure paths) |
| M30 `reload_from_file` drops sRGB format | ✅ | Original format preserved on reload |
| M31 `TaskHandle::error()` data race | ✅ | Returns nullptr until `is_done()` |
| M32 `parallel_for` rethrows before join | ✅ | Joins all chunks, then rethrows the first exception |
| M33 `shutdown()` silently discards queued jobs | ✅ | Handles complete with a "discarded at shutdown" error + WARN log; `wait()` rethrows |
| M34 transient failure but `compile()` succeeds | ✅ | `compile()` fails on transient allocation failure |
| L1 `EnvironmentState` namespace mismatch | ✅ | Moved to `bracken::render` — code now matches the docs |
| L2 host-driven `on_init`/`on_shutdown` undocumented | ✅ | Contract documented in `application.hpp` + architecture.md |
| L3 lag comments inverted | ✅ | Comments corrected |
| L4 shaders README `.spv` path wrong | ✅ | `.spv` land in `<shaders binary dir>/spv/`; README updated |
| L5 ACES tonemap attribution wrong | ✅ | Attributed to `lighting.hlsli` |
| L6 `FootIKBoneNames` "Mixamo" claim | ✅ | Comment corrected — standard rig names, override per rig |
| L7 ".glb/.gltf" doc comment | ✅ | ".glb only" |
| L8 `baseColorFactor` ignored with COLOR_0 | ✅ | Untextured path is vertexColor × baseColorFactor × tint; loader bakes white when COLOR_0 absent |
| L9 `RockParams::texture_path` ignored | ✅ | Field + preset texture params deleted |
| L10 skinned draws can't opt out of fog | ✅ | `SkinnedModelCommand::no_fog` added and plumbed (`add_skinned_model` grew `bool no_fog = false`) |
| L11 hardcoded camera constants | ✅ | `CameraModeConfig` gained 8 fields (zoom limits, planes, clearance, sprint FOV, look-ahead, auto-center delay) |
| L12 text measurement unreachable | ✅ | `SceneRenderer::text_width`/`text_height` passthroughs via UIRenderer |
| L13 default entry point "main" | ✅ | Defaults are `VSMain`/`PSMain` |
| L14 `gpu::ShaderProgram` dead | ✅ | Deleted — use `ShaderManager::get_vertex`/`get_fragment` |
| L15 `--smoke` atoi-to-0 runs interactive | ✅ | Invalid/non-positive counts error with exit code 2 |
| L16 smoke ctest never runs without `BRACKEN_BUILD_TESTS` | ✅ | Option moved to root CMakeLists; `enable_testing()` fires for either option |
| L17 non-recursive `.hlsli` GLOB | ✅ | Recursive include-dependency glob |
| L18 stale `.spv` never removed | ✅ | Staging removes stale files each build |
| L19 `grass_view_distance` silently overwritten | ✅ | Respected (default 1200, capped via `set_view_distance_cap`) |
| L20 wind direction hardcoded | ✅ | `wind_direction` knob exposed |
| L21 splat-aware grass dormant | ✅ | Kept as a documented host opt-in — dormant by design |
| L22 `GrassRenderer::update` ignores `delta_time` | ✅ | Dead parameter dropped; signature is `update(current_time)` |
| L23 `has_3d_content()` ignores billboards/debug lines/lights | ✅ | Now covers all three |
| L24 `set_draw_rocks`/`set_draw_trees` never read | ✅ | Dead flags deleted |
| L25 `clear_color` dead API | ✅ | Deleted |
| L26 `set_debug_draw_enabled` dead flag | ✅ | Deleted |
| L27 `set_focus_strength` unclamped | ✅ | Clamped to the documented 0–1 |
| L28 `set_yaw`/`set_pitch` don't mark input | ✅ | Mark input on change; auto-centering works for absolute-angle hosts |
| L29 shake applied after collision avoidance | ✅ | Shake moved before collision (stage 6/7) — shakes can no longer clip the camera |
| L30 bloom default 0.35 vs 0.25 | ✅ | 0.25 everywhere |
| L31 `available_resolutions` density claim | ✅ | Contract documented honestly in the header (no density filtering implied) |
| L32 `sky_time()` "shared clock" comment | ✅ | Contract documented: drives the skybox only |
| L33 Skybox "cubemap" comment | ✅ | Corrected — fully procedural |
| L34 particle blend documented ONE/ONE | ✅ | Docs/comments corrected to (SRC_ALPHA, ONE) with shader-forced alpha 1.0 |
| L35 effect depth-test comment LESS_OR_EQUAL | ✅ | Corrected to GREATER_OR_EQUAL (reversed-Z) |
| L36 `duration`/`loop` never read | ✅ | Split: `loop` removed; `duration` implemented (default -1 = emitter-driven) |
| L37 effect `range` plumbed but unconsumed | ✅ | Entire `range`/`default_range` chain removed (RenderScene → EffectSystem) |
| L38 `particle_type` "sprite" stub | ✅ | Field deleted — mesh particles are the contract |
| L39 `Particle::model` written, never read | ✅ | Dead field removed |
| L40 ABGR "same as rest of renderer" comment | ✅ | Comment site deleted with legacy `draw_text`; single ARGB convention (see M6) |
| L41 `ui.frag` textured path dead plumbing | ✅ | Live via new `draw_textured_quad` |
| L42 `InputState::attacking`/`sprinting` never written | ✅ | Fields removed — hosts own action state |
| L43 `set_player_screen_pos` dead | ✅ | API removed |
| L44 misaligned non-pow2 `Arena` alignment | ✅ | Debug-asserted |
| L45 `allocate(0)` nullptr ambiguity | ✅ | Returns a valid non-null sentinel; nullptr uniquely means failure |
| L46 `allocate`/`create` not `[[nodiscard]]` | ✅ | Marked |
| L47 dead `frame_arena` upload overload | ✅ | Deleted |
| L48 `ClusterParamsUniforms` unreferenced mirror | ✅ | Deleted — `ClusterParams` is the single definition |
| L49 GTAO tunables unreachable | ✅ | Contract documented: compile-time constants by design, only `radiusScale` is settings-driven |
| L50 `AmbientOcclusion` owns the whole post chain | 🟡 | Scope documented honestly; rename to `PostProcessStack` recommended as follow-up |
| L51 `prefer_low_power` never read | ✅ | Honored via properties-based device creation |
| L52 `get_bytes_per_pixel` defaults unknown to 4 | ✅ | `upload` rejects unknown-texel-size formats |
| L53 transfer buffer released before submit | ✅ | Verified safe; the SDL3 GPU deferred-release contract is now pinned in a comment |
| L54 gpu_types dead structs | ✅ | Deleted |
| L55 `useSkinning` written, never read | ✅ | Removed; `ModelTransformUniforms` repacked (`paletteOffset` at 272, size 288) with layout tests |
| L56 unused lighting.hlsli helpers + local duplicates | ✅ | ssao/gtao/fog use the shared helpers; dead `srgbToLinear` deleted |
| L57 "branch-free path" comment wrong | ✅ | Comment corrected |
| L58 pbr.hlsli only transitively included | ✅ | Model shaders include `pbr.hlsli` directly |
| L59 `HandlePool::clear()` handle aliasing | ✅ | `clear()` bumps generations — stale handles fail validation; capacity preserved |
| L60 assert messages dropped when logger Off | ✅ | Fatal asserts print via sinks AND stderr |
| L61 `BRACKEN_ASSERT` discards condition under NDEBUG | ✅ | Condition type-checked via `sizeof` (never evaluated); no-side-effects rule documented |
| L62 result.hpp "need to migrate" comment | ✅ | Comment aligned with the recorded no-mass-conversion policy |
| L63 `std::hash<Handle>` 32-bit UB | ✅ | Hash fixed for 32-bit `size_t` |
| L64 `Result`/`StringId`/`HandlePool` zero consumers | ⏸️ | Infrastructure-ahead-of-use kept deliberately; adopt at natural touch points |
| L65 `wait()` busy-yields | ✅ | Parks on a condition variable |
| L66 "no re-init path" bench comment | ✅ | Comment corrected — `shutdown()`+`init()` re-init is supported |
| L67 architecture.md `AssetRegistry` hot-reload claim | ✅ | Doc corrected: host-facing utility; engine registers directly on `FileWatcher::instance()` |
| L68 architecture.md POSITION-min/max bounds claim | ✅ | Doc corrected: bounds accumulate from parsed vertices |
| L69 `set_anisotropic_filter` "grass sampler" wording | ✅ | Doc/comment/log corrected (terrain material sampler) |
| L70 `Mesh::base_color` packing mismatch | ✅ | Write-only field deleted outright |
| L71 rock seed float-hash collisions | ✅ | Seed integer-hashed to [0, 2^20); same-seed shapes changed |
| L72 canopy 6-vs-12-triangle comment | ✅ | Comment + `reserve` corrected (36 indices) |
| L73 self-cancelling `section_length` code | ✅ | Dead code deleted |
| L74 RockGenerator untested | ✅ | New `rock_generator.test.cpp` (14 tests) |
| L75 dead combat/FOV camera members | ✅ | Deleted (plus unused `total_time`/`yaw_rad`) |
| L76 `CameraShake::frequency` vestigial | ✅ | Field removed |
| L77 `AnimationKeyframe`/`MAX_BONE_INFLUENCES` unreferenced | ✅ | Deleted |
| L78 `apply_foot_ik` ignored matrix param | ✅ | Parameter removed; call sites updated, deprecation shim deleted |
| L79 design-notes debug-group claim | ✅ | engine_design_notes.md corrected |
| L80 `ResourceNode::version` write-only | ✅ | Unimplemented versioning/cycle-detection claims removed |
| L81 render-graph buffer scaffolding dead | ✅ | `get_buffer`/`BufferDesc` removed |
| L82 `PassType` stored, never consumed | ✅ | `PassType` removed; `add_pass` is 3-arg |
| L83 `dump_dot` declared-vs-compiled + escaping | ✅ | Labels escaped; post-compile `[step N]` annotation |
| L84 `[[nodiscard]] bool init` with no failure path | ✅ | `init` now void; `compile()` gained a real failure path (see M34) |
| L85 DevUI double `init()` discards Bindings | ✅ | Returns false + warns |
| L86 Graphics tab shown without `apply_settings` | ✅ | Requires settings AND `apply_settings` |
| L87 Camera tab function-local statics | ✅ | Per-instance `PanelState`; re-reads the controller via new `get_fov_bias`/`get_focus_strength` |
| L88 Stats FPS history single-instance statics | ✅ | Per-instance `PanelState` (multi-instance ImGui-context handling noted under Follow-ups) |
| L89 `render()` "every frame" drain comment | ✅ | Comment corrected for null-swapchain frames |
| L90 WorldRenderer four vestigial pieces | ✅ | `render_grid`/`model_manager_`/height callback/sampler deleted; `init(device, registry)`; returns false on skybox buffer failure |

## Follow-ups (small, non-blocking)

1. Rename `AmbientOcclusion` → `PostProcessStack` (L50 remainder) — a quiet
   mechanical rename; the class's real scope is documented honestly until
   then.
2. Add a `UIScene` image-widget command if hosts want textured quads through
   the scene path — `draw_textured_quad` (L41) currently lives at the
   UIRenderer level.
3. Multi-instance `DevUI` ImGui-context handling (L88 residual) —
   per-instance panel state is fixed, but a single ImGui context still
   assumes one overlay per process.

# Graphics feature campaign — 2026-07-12

The 2026-07-05 graphics.md **deferred/roadmap** items are now implemented
(the historical table above is unchanged; this section records the closure):

| 07-05 item | Now | Where |
|---|---|---|
| H3 AO darkens final lit color | ✅ implemented | Depth prepass + ambient-only AO: GTAO runs before the forward pass and is applied to the ambient term only (`render/gtao.hpp`, `pbr.hlsli`); `AmbientOcclusion` renamed → `PostProcessStack` (closes 07-12 follow-up 1) |
| M2 no TAA / motion vectors | ✅ implemented | RG16F velocity from the prepass + `render::TAA` HDR resolve/sharpen; `AAMode::TAA` is the fresh-settings default (`render/taa.hpp`) |
| M3 no IBL | ✅ implemented | `render::IBL` sky prefilter chain (capture → GGX specular mips → irradiance → BRDF LUT), amortized on sky changes; `ibl_enabled` setting, hemisphere fallback (`render/ibl.hpp`) |
| M4 no point/spot shadows | ✅ implemented | Budgeted cached-tile 8×8 local-shadow atlas, host opt-in via `casts_shadows` (`render/lighting/local_shadows.hpp`) |
| M6 no depth prepass | ✅ implemented | With H3: opaque prepass (position-only, velocity MRT) + `*DepthEqual` forward variants |

The same campaign also landed (beyond the audit's list): histogram auto
exposure + color grading on the engine's first compute pipelines
(`render/auto_exposure.hpp`, `gpu/gpu_compute.hpp`), and the sky/cloud/fog
polish pass (domain-warped clouds, `cloud_*`/`haze_strength` environment
controls, shared `hazeSunColor` sky↔fog horizon agreement).

Consciously deferred from this campaign:

1. **IBL prefilter compute migration** — the prefilter chain runs as tiny
   raster passes (~0.1 ms amortized); migrating it to compute is a
   cost-neutral cleanup, not a win, until profiling says otherwise.
2. **Adapted EV in the bloom threshold** — the bloom prefilter thresholds
   post-MANUAL-exposure luminance only; the adapted auto-exposure EV lives in
   a GPU buffer the bloom pass doesn't read (documented as a known stub in
   `docs/architecture.md`). Wire the exposure SSBO into
   `bloom_downsample.frag` if it ever matters visually.

Still open from the 07-05 roadmap: HDR display output (L4), sRGB-correct UI
blending (L5), SoA particles (perf L7).
