# Audit Resolutions — 2026-07-05

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
| H1 stale `.clang-tidy` header filter | ✅ | Points at `src/engine`; header diagnostics live again |
| H2 `get_draw_distance` OOB | ✅ | Clamped + `clamp_to_valid_ranges()` after `load()` + corrupt-file tests |
| H3 `Result` unused, two conventions | ✅ | `[[nodiscard]]` Result + `Result<void,E>`; policy written into CLAUDE.md (no mass conversion, by design) |
| M1 zero `[[nodiscard]]` | ✅ | Sweep over fallible bools + factories |
| M2 unasserted threading contracts | ✅ | `core::bind_main_thread()` + `ENGINE_ASSERT_MAIN_THREAD()` at documented boundaries |
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
