# Documentation Sweep Audit — Source Findings

**Date:** 2026-07-11
**Scope:** the entire engine — every module under `src/bracken/` (headers,
implementations, tests), top-level headers, the build system (CMake + the
shader-compilation pipeline), the HLSL shader tree, and the playground example
where it exercises engine API.
**Method:** findings were surfaced by per-area documentation agents that read
every header, implementation, and test while authoring `docs/user_guide.md`
and `docs/technical_spec.md`. Each finding was then re-verified against the
current source with exact locations, and adversarially checked by an
independent skeptic instructed to refute it (checking, among other things,
whether the behavior is a recorded deliberate design). No builds or profiling
were run for this audit; all evidence is static inspection of the working
tree.

> **Status: a fix campaign ran 2026-07-12 and addressed all 127 findings.**
> Per-finding outcomes are recorded in [RESOLUTIONS.md](RESOLUTIONS.md).
> The findings below are preserved unmodified as a historical record and no
> longer describe the current source.

**Counts, honestly:** 136 candidate findings entered verification. 3 were
refuted by the skeptic pass and are excluded from the findings sections
(listed under "Refuted during verification" at the end). Of the 133 that
survived, 6 were duplicates reported independently by two areas and are
merged below — one entry per underlying defect, with every affected area and
location noted. That leaves **127 unique findings: 3 high, 34 medium,
90 low.**

---

## Executive summary

The engine's own frame path is in good health: this sweep found no in-tree
rendering-correctness breakage, no crash reachable from the playground, and
the prior audit campaign's fixes intact. What it found instead is the residue
of a fast extraction from `mmo-4` plus a library whose real consumers are out
of tree: **90 of the 127 findings are low severity** — stale or wrong
comments, dead/vestigial API surface (game leftovers like
`set_player_screen_pos`, combat camera fields, the grass "town" locals), and
infrastructure built ahead of any consumer (`Result`/`StringId`/`HandlePool`,
render-graph buffer scaffolding). The dominant categories across all tiers
are doc-code mismatch, dead-code, and latent bugs on host-facing surfaces
that no in-tree code exercises.

The three highs are all reachable by an ordinary consumer. **H1**: the
engine's PUBLIC include dir is `${CMAKE_SOURCE_DIR}/src`, so the documented
`add_subdirectory` consumption one-liner in README.md/CLAUDE.md does not
compile — the engine's own `bracken/`-rooted self-includes fail to resolve
inside any super-project. **H2**: `JobSystem::wait()` called from a worker
never drains its own local queue or steals, so the ordinary fork-join pattern
(submit sub-jobs, wait) livelocks deterministically on a 1-worker pool — and
default `init()` produces exactly one worker on ≤2-hardware-thread machines,
where the bundled test suite itself hangs. **H3**: a non-positive
`spawn_rate` in a CONTINUOUS emitter definition (host JSON data) either
silently emits one particle (0) or hangs `update()` with unbounded particle
growth (negative).

The mediums cluster into four groups. (a) **Library-consumer traps** the
in-tree code happens to sidestep: `RenderScene`'s draw flags default false at
construction but true after `clear()`; `Handle{}` reports valid;
`AssetRegistry` corrupts its name map on duplicate-name loads; GPU texture
APIs return success on partial failure or silently drop sRGB on reload; the
jobs module has an exception-unwind use-after-scope, a `TaskHandle::error()`
data race, and shutdown-discard semantics that make later `wait()` spin
forever. (b) **Silent asset-pipeline gaps in the glTF loader and animation
stack**: sampler interpolation modes ignored (CUBICSPLINE data misread),
inverse-bind/animation accessors read raw with no componentType checks,
standalone occlusion maps parsed then discarded, a compacted-vector index
skew for base-color textures, and an unenforced parents-before-children joint
order. (c) **Game leftovers violating the game-agnostic rule**: the grass
renderer's hardcoded 400×400 "town" clearing and edge margin (every consumer
world gets a bald patch), and menu keys matched by layout-dependent keycode
while movement uses scancodes. (d) **Build-system leaks**: force-cached
`BUILD_SHARED_LIBS`/PIC globals that override a consuming super-project, and
`find_program` preferring any system `shadercross` over the pinned CLI.

Fix-first: H1–H3, then M1 (town clearing), M2 (force-cached globals), M4
(keycode/scancode), M5 (RenderScene flag defaults), M6 (packed-color
conventions — the playground already renders "the same" literal as two
different hues), and M9/M10 (unvalidated + cache-ignored `ShaderResources`,
the documented most-common way to silently break rendering). A recurring
theme worth exploiting during fixes: where a header comment and the new docs
disagree, the docs written during this sweep are almost always the accurate
side — most comment fixes are one-liners at the cited sites, and many
findings already have gotcha entries in `docs/user_guide.md` /
`docs/technical_spec.md`; this report is the source-side ledger for actually
fixing them.

---

## Index

| ID | Area | Title | Category |
|---|---|---|---|
| H1 | build | PUBLIC include dir is `${CMAKE_SOURCE_DIR}/src` — documented `add_subdirectory` consumption fails to compile | build-system |
| H2 | jobs | `wait()` on a worker never drains its own queue or steals — nested submit+wait livelocks on a 1-worker pool | hang-risk |
| H3 | effects | CONTINUOUS `spawn_rate` unvalidated: 0 yields one particle, negative hangs `update()` with unbounded growth | hang-risk |
| M1 | terrain-grass | Hardcoded game-specific "town" clearing and world-edge margin in grass chunk culling | design-violation |
| M2 | build | `BUILD_SHARED_LIBS=OFF` / `CMAKE_POSITION_INDEPENDENT_CODE=ON` force-cached globally, leaking into consumers | build-system |
| M3 | build, shaders | Any system `shadercross` on PATH silently preferred over the pinned, freshly built CLI (no version check) | build-system |
| M4 | application | Menu keys match layout-dependent keycodes while movement/edge tracking use scancodes | latent-bug |
| M5 | scene | `RenderScene` draw flags default false at construction but `clear()` resets them to true | latent-bug |
| M6 | text-ui, scene | Three incompatible packed-color conventions: UI ARGB, text ABGR, debug lines RGBA | api-quirk |
| M7 | core | `StdoutSink` Trace/Debug output silently suppressed by SDL's own priority filter | api-quirk |
| M8 | assets | `PipelineRegistry::enable_hot_reload` logs "enabled" and latches `watcher_` even when nothing was watched | latent-bug |
| M9 | shaders | `ShaderResources` counts never validated against the SPIR-V or device limits | api-quirk |
| M10 | shaders | `ShaderManager::get` silently ignores the `ShaderResources` argument on cache hits | latent-bug |
| M11 | models, model-pbr-shading | Separate (non-ORM) occlusion textures parsed but never uploaded; `occlusion_texture` never assigned | latent-bug |
| M12 | animation | glTF sampler interpolation ignored — STEP smoothed, CUBICSPLINE output layout misread | latent-bug |
| M13 | models | Inverse-bind-matrix and animation-sampler accessors read raw — no componentType or stride checks | latent-bug |
| M14 | animation | Parents-before-children joint order neither documented nor validated at load | latent-bug |
| M15 | animation | `apply_foot_ik` pelvis-drop incremental update wrong for rotated/scaled inverse-bind matrices | latent-bug |
| M16 | models | Base-color texture lookup indexes a compacted vector by raw glTF image index | latent-bug |
| M17 | models, assets | Header promises raw `Model*` stability, but `reload_model` destroys and replaces the `Model` object | doc-code-mismatch |
| M18 | procedural | Textured tree leaves render as opaque single-sided rectangles in the color pass | bug |
| M19 | procedural | Failed tree texture load silently yields an all-white tree | latent-bug |
| M20 | procedural | `TreeParams::levels` unvalidated; `levels >= 4` with `children[3] > 0` reads past 4-entry arrays (UB) | latent-bug |
| M21 | terrain-grass | Terrain/grass splatmap UVs assume a square world anchored at origin | latent-bug |
| M22 | terrain-grass | Grass splatmap pointer dangles after `TerrainRenderer::update_splatmap` recreates the texture | latent-bug |
| M23 | application | Exclusive fullscreen silently falls back and mis-records state when no display mode matches | latent-bug |
| M24 | core | Default-constructed `Handle{}` reports `is_valid() == true` | api-quirk |
| M25 | assets | `AssetRegistry` duplicate-name load orphans the old entry; unloading it erases the newer entry's name mapping | latent-bug |
| M26 | effects | `EffectRenderer` emitter→model cache stores failed lookups permanently — late-registered models never render | latent-bug |
| M27 | text-ui | UI primitive-batch overflow unhandled: `flush_batch()` is a no-op, oversized frames draw stale GPU buffer contents | latent-bug |
| M28 | text-ui | Legacy `TextRenderer::draw_text` begins a copy pass inside the caller's active render pass (illegal in SDL3 GPU) | latent-bug |
| M29 | gpu | `create_2d_array` returns a valid-looking texture with no pixel data when the upload stage fails | latent-bug |
| M30 | gpu | `reload_from_file` always reloads as linear RGBA8, dropping the original sRGB format | latent-bug |
| M31 | jobs | `TaskHandle::error()` reads `Task::error` without checking `done` — data race against the worker's write | latent-bug |
| M32 | jobs | `parallel_for` rethrows the first chunk exception before joining remaining chunks — use-after-scope in stragglers | latent-bug |
| M33 | jobs | `shutdown()` silently discards queued jobs and zeroes `pending_`; a later `wait()` spins forever | hang-risk |
| M34 | render-graph | Transient texture creation failure logged but `compile()` still succeeds — null attachments reach `execute()` | latent-bug |
| L1 | atmosphere-post | `EnvironmentState` is in `namespace bracken`, but CLAUDE.md/architecture.md/user_guide.md call it `render::EnvironmentState` | doc-code-mismatch |
| L2 | application | `Application` header implies an `on_init`/`on_shutdown` lifecycle but never states that `run()` doesn't call them | api-quirk |
| L3 | camera-systems | `position_lag`/`rotation_lag` comments invert the parameter's actual meaning (spring smooth-time, lower = snappier) | doc-code-mismatch |
| L4 | shaders | Shaders README says `.spv` land in `<build>/shaders/`; actual dir is `build/src/bracken/shaders/` | doc-code-mismatch |
| L5 | model-pbr-shading | shaders/README.md attributes the ACES tonemap to `pbr.hlsli`; it lives in `lighting.hlsli` | doc-code-mismatch |
| L6 | animation | `FootIKBoneNames` comment claims defaults "match Mixamo conventions"; they don't match raw Mixamo rigs | doc-code-mismatch |
| L7 | models | `enable_hot_reload` comment says ".glb/.gltf" but the loader is binary-only (`LoadBinaryFromFile`) | doc-code-mismatch |
| L8 | model-pbr-shading | `baseColorFactor` ignored when a mesh has COLOR_0 but no base-color texture (spec deviation) | bug |
| L9 | procedural | `RockParams::texture_path` and all preset texture parameters declared but completely ignored | api-quirk |
| L10 | model-pbr-shading | `SkinnedModelCommand` has no `no_fog` field — skinned draws can never be exempted from fog | api-quirk |
| L11 | camera-systems | Near/far planes, zoom limits, auto-center delay, clearance, sprint-FOV constants hard-coded, no config API | api-quirk |
| L12 | text-ui | Text measurement (`get_text_width/height`) unreachable for hosts — no UIRenderer/TextRenderer accessor | api-quirk |
| L13 | shaders | GPUShader/ShaderManager default entry point "main" never matches engine-built SPIR-V (VSMain/PSMain) | api-quirk |
| L14 | shaders | `gpu::ShaderProgram` has no in-tree callers and hardwires empty `ShaderResources` | dead-code |
| L15 | build | `playground --smoke <non-numeric>` atoi's to 0 and silently runs interactive mode | bug |
| L16 | build | `BRACKEN_PLAYGROUND_SMOKE_TEST=ON` without `BRACKEN_BUILD_TESTS` registers a test ctest never runs | build-system |
| L17 | build | `.hlsli` dependency glob is non-recursive while shader sources use GLOB_RECURSE — stale SPIR-V if includes move | build-system |
| L18 | shaders | `bracken_copy_shaders` never removes stale `.spv` from staging (deleted shaders keep shipping) | build-system |
| L19 | terrain-grass | Public `grass_view_distance` field silently overwritten every frame by SceneRenderer | api-quirk |
| L20 | terrain-grass | Grass wind direction hardcoded to `vec2(1.0, 0.3)` despite full uniform plumbing | api-quirk |
| L21 | terrain-grass | Splat-aware grass placement dormant: `set_splatmap`/`update_splatmap` have zero in-tree callers | dead-code |
| L22 | terrain-grass | `GrassRenderer::update` takes `delta_time` but explicitly ignores it | api-quirk |
| L23 | scene | `has_3d_content()` ignores billboards, debug lines, and lights — such-only frames skip the 3D pass | latent-bug |
| L24 | scene | `set_draw_rocks`/`set_draw_trees` flags stored and reset but never read by any renderer | dead-code |
| L25 | atmosphere-post | `RenderContext::clear_color()`/`set_clear_color()` are dead API — no pass reads them | dead-code |
| L26 | camera-systems | `set_debug_draw_enabled` sets a flag nothing reads (no-op API) | dead-code |
| L27 | camera-systems | `set_focus_strength` documented as 0–1 but never clamped | api-quirk |
| L28 | camera-systems | `set_yaw`/`set_pitch` don't mark input — auto-centering silently inert for absolute-angle hosts | api-quirk |
| L29 | camera-systems | Camera shake applied after collision avoidance — large shakes can clip through geometry | latent-bug |
| L30 | atmosphere-post | `CompositeParams`/`CompositeUniforms` default bloom strength (0.35) disagrees with GraphicsSettings (0.25) | api-quirk |
| L31 | atmosphere-post | `available_resolutions` doc claims density==1.0 filter; `query_display_modes` never filters by density | doc-code-mismatch |
| L32 | atmosphere-post | `sky_time()` comment claims it is the shared clock for grass wind/cloud scroll, but nothing reads it | doc-code-mismatch |
| L33 | atmosphere-post | `PipelineType::Skybox` commented "cubemap rendering" but the skybox is fully procedural — no cubemap exists | doc-code-mismatch |
| L34 | gpu, effects | Docs/comments describe particle blend as ONE/ONE; configured factors are (SRC_ALPHA, ONE) with shader alpha 1.0 | doc-code-mismatch |
| L35 | effects | `EffectRenderer` class comment claims depth test LESS_OR_EQUAL; Particle pipeline uses GREATER_OR_EQUAL (reversed-Z) | doc-code-mismatch |
| L36 | effects | `EffectDefinition::duration`/`loop` never read by the runtime; docs list them as functional | dead-code |
| L37 | effects | Effect `range`/`default_range` plumbed into `EmitterInstance::range` but never consumed | dead-code |
| L38 | effects | `EmitterDefinition::particle_type` never read; "sprite" is a declared-but-unimplemented stub | dead-code |
| L39 | effects | `Particle::model` written at spawn but never read (renderer resolves per emitter) | dead-code |
| L40 | text-ui | `draw_text` comment claims ABGR is "same as rest of renderer"; UI primitives decode ARGB | doc-code-mismatch |
| L41 | text-ui | ui.frag `has_texture` textured-quad path is dead plumbing — every caller pushes 0 with a dummy texture | dead-code |
| L42 | application | `InputState::attacking`/`sprinting` never written by engine code; header doesn't state the host-owned contract | dead-code |
| L43 | application | `InputHandler::set_player_screen_pos` stores state nothing reads (vestigial from the extracted game) | dead-code |
| L44 | memory | `Arena::allocate` silently returns misaligned memory for non-power-of-two alignment (unasserted precondition) | api-quirk |
| L45 | memory | `Arena::allocate(0, ...)` returns nullptr, indistinguishable from allocation failure | api-quirk |
| L46 | memory | `Arena::allocate`/`create` are fallible but not `[[nodiscard]]` (missed by the M1 sweep) | design-violation |
| L47 | lighting-shadows, memory | `ClusterGrid::upload` command-buffer overload keeps a dead `frame_arena` parameter with zero callers | dead-code |
| L48 | lighting-shadows | `gpu::ClusterParamsUniforms` is an unreferenced mirror of `ClusterParams` with no layout check | dead-code |
| L49 | lighting-shadows | GTAOUniforms tunables other than radiusScale are compile-time constants unreachable via GraphicsSettings | api-quirk |
| L50 | lighting-shadows | `AmbientOcclusion` class owns the whole HDR offscreen/composite/tonemap/FXAA chain, far beyond its name | api-quirk |
| L51 | gpu | `GPUDevice::init`'s `prefer_low_power` parameter is documented but never read | doc-code-mismatch |
| L52 | gpu | `get_bytes_per_pixel` defaults unknown SDL formats to 4 bytes, mis-sizing `upload()` for wide formats | latent-bug |
| L53 | gpu | `GPUTexture::upload` releases its transfer buffer before the caller submits (relies on SDL deferred release) | api-quirk |
| L54 | gpu | gpu_types.hpp declares five dead uniform structs that shadow the live contract in gpu_uniforms.hpp | dead-code |
| L55 | model-pbr-shading | `useSkinning` uniform written every draw but never read by any shader | dead-code |
| L56 | model-pbr-shading | lighting.hlsli canonical depth/sRGB helpers unused while ssao/gtao/fog keep local duplicates | dead-code |
| L57 | model-pbr-shading | Comment claims flat-normal fallback serves a "branch-free path"; shaders branch on `hasNormalMap` | doc-code-mismatch |
| L58 | model-pbr-shading | Model fragment shaders use pbr.hlsli functions only via transitive include through cluster_lighting.hlsli | api-quirk |
| L59 | core | `HandlePool::clear()` resets generations — pre-clear handles alias post-clear slots | latent-bug |
| L60 | core | Assertion failure messages routed through the logger and dropped when min level is Off | latent-bug |
| L61 | core | `BRACKEN_ASSERT` discards its condition entirely under NDEBUG (not even type-checked) | api-quirk |
| L62 | core | result.hpp comment says bool init/load/save APIs "need to migrate", contradicting the recorded no-mass-conversion policy | doc-code-mismatch |
| L63 | core | `std::hash<Handle>` shifts by 32 bits — UB when `size_t` is 32-bit | latent-bug |
| L64 | core | `Result`, `StringId`/fnv1a, and `HandlePool` have zero consumers outside core/ — infrastructure ahead of use | dead-code |
| L65 | jobs | `wait()` busy-yields when it cannot help — no CV block, burns a core on long waits | api-quirk |
| L66 | jobs | core.bench.cpp comment says the singleton "has no re-init path"; shutdown()+init() fully supports it (untested) | doc-code-mismatch |
| L67 | assets | architecture.md presents `AssetRegistry` as the asset module's hot-reload mechanism; no engine system instantiates it | doc-code-mismatch |
| L68 | models | architecture.md says bounds come from the POSITION accessor min/max; code accumulates from every parsed vertex | doc-code-mismatch |
| L69 | terrain-grass | `set_anisotropic_filter` doc/comment/log say "grass sampler"; code recreates the terrain material sampler | doc-code-mismatch |
| L70 | procedural | Procedural generators pack `Mesh::base_color` as ARGB while the glTF loader packs/reads ABGR | latent-bug |
| L71 | procedural | Rock seed participates in the SDF hash as a float — distinct seeds ≥ 2^24 collide | latent-bug |
| L72 | procedural | Canopy-proxy comment and `indices.reserve` assume 18 indices per clump; 36 are emitted (2× tri budget) | doc-code-mismatch |
| L73 | procedural | Self-cancelling `section_length` adjustment in `generate_branch` is dead code | dead-code |
| L74 | procedural | `RockGenerator` has no unit tests (only the shared benchmark exercises it) | design-violation |
| L75 | camera-systems | Dead combat/FOV members plus unused `static total_time` and local `yaw_rad` in CameraSystem | dead-code |
| L76 | camera-systems | `add_shake` per-type frequency switch assigns 1.0f in every branch — `CameraShake::frequency` is vestigial | dead-code |
| L77 | animation | `AnimationKeyframe` struct and `MAX_BONE_INFLUENCES` constant declared but unreferenced anywhere | dead-code |
| L78 | animation | `apply_foot_ik` accepts a `model_to_world` matrix the implementation ignores | dead-code |
| L79 | render-graph | engine_design_notes.md credits render_graph with debug-group integration that does not exist | doc-code-mismatch |
| L80 | render-graph | `ResourceNode::version` is write-only; its comment claims cycle detection and versioned reads | doc-code-mismatch |
| L81 | render-graph | Buffer resource support is dead scaffolding — no API can declare or import a buffer | dead-code |
| L82 | render-graph | `PassType` stored but never consumed by the executor; typed `add_pass` overload cannot set it | api-quirk |
| L83 | render-graph | `dump_dot` claims the compiled graph but dumps the declared graph, and doesn't escape DOT labels | doc-code-mismatch |
| L84 | render-graph | `RenderGraph::init`/`TransientAllocator::init` are `[[nodiscard]] bool` with no failure path | api-quirk |
| L85 | dev-ui | `DevUI::init()` on an already-initialized instance returns true but silently discards the new Bindings | api-quirk |
| L86 | dev-ui | Graphics tab shown whenever `bindings.settings` is non-null, even with an empty `apply_settings` | latent-bug |
| L87 | dev-ui | Camera tab shake/FOV-bias/focus sliders are function-local statics never re-read from the controller | api-quirk |
| L88 | dev-ui | Stats tab FPS history uses function-local statics, assuming a single DevUI instance per process | latent-bug |
| L89 | dev-ui | Comment claims `render()` drains the ImGui input queue "every frame"; skipped on null-swapchain frames | doc-code-mismatch |
| L90 | atmosphere-post | WorldRenderer carries four vestigial pieces: uncalled `render_grid`, unused `model_manager_`, dead terrain-height callback, unused sampler | dead-code |

---
## Findings — Build system & getting started

### H1. PUBLIC include dir is `${CMAKE_SOURCE_DIR}/src` — the documented `add_subdirectory` consumption path fails to compile

**Category:** build-system
**Locations:** `src/bracken/CMakeLists.txt:84`, `:154`, `:164`; `README.md:171`; `docs/user_guide.md:160`; `docs/technical_spec.md:90`

`target_include_directories(bracken PUBLIC ${CMAKE_SOURCE_DIR}/src ...)`
resolves against the **top-level** project's source dir, which is only the
engine root when Bracken itself is top-level. Under
`add_subdirectory(path/to/engine)` — the exact pattern documented without
caveat in README.md ("Consuming the engine") and CLAUDE.md — it resolves to
the consumer's `src/`. This is the only mechanism supplying the engine's
`bracken/`-rooted include root, so the `bracken` target's own
`#include "bracken/..."` self-includes fail and the library itself does not
compile inside a super-project (or silently compiles against wrong headers if
the consumer coincidentally has a `src/bracken/` tree). The same pattern is
repeated for `bracken_benchmarks` (line 154) and `bracken_tests` (line 164),
whose options default ON. The limitation is acknowledged with a workaround in
user_guide.md/technical_spec.md, but the primary consumption docs carry no
caveat and the CMake is unfixed.

**Impact:** a consumer following the README verbatim gets a hard build
failure; the documented one-liner does not work.
**Fix:** replace `${CMAKE_SOURCE_DIR}/src` with
`${CMAKE_CURRENT_SOURCE_DIR}/..` (equivalently `${PROJECT_SOURCE_DIR}/src`)
at all three sites — the fix user_guide.md already anticipates.

### M2. `BUILD_SHARED_LIBS=OFF` and `CMAKE_POSITION_INDEPENDENT_CODE=ON` are force-cached globally, leaking into consuming super-projects

**Category:** build-system
**Locations:** `CMakeLists.txt:11`, `src/bracken/shaders/CMakeLists.txt:19`

Both are global CMake variables written to the shared cache with FORCE on
every configure (to make vendored deps static and to satisfy the
SDL_shadercross/DXC build). Under `add_subdirectory` consumption they
override the super-project's own settings: a consumer's
`-DBUILD_SHARED_LIBS=ON` is silently reverted each configure and all of its
libraries build static; explicit PIC-off choices are forced back on.

**Impact:** silent build-configuration changes outside the engine's own tree.
**Fix:** scope the intent to the vendored deps — per-project switches (as
already done with `SDL_SHARED`/`SDL_STATIC` at `src/bracken/CMakeLists.txt:24-25`),
non-FORCE sets, or save/restore plain variables around
`FetchContent_MakeAvailable`; set `POSITION_INDEPENDENT_CODE` as a target
property where required.

### M3. Shader compilation prefers any system `shadercross` on PATH over the pinned, freshly built CLI — no version or capability check

**Category:** build-system · **Reported by two areas:** build/getting-started and shaders (merged)
**Locations:** `src/bracken/shaders/CMakeLists.txt:29-37`; runtime link at `src/bracken/CMakeLists.txt:95`

After FetchContent pins SDL_shadercross to commit `7b7365a8` (with
`SDLSHADERCROSS_CLI=ON`), `find_program(SHADERCROSS_EXECUTABLE shadercross)`
lets any system binary win — with only a STATUS message, no `--version`
check, and (in that branch) an empty `SHADERCROSS_DEPENDS`, so shader custom
commands no longer depend on the CLI target. The engine still links the
pinned `SDL3_shadercross-static` runtime library either way, so build-time
compiler and runtime transpiler can be different versions. `find_program`
caches its result, so the skew persists across reconfigures; CI (no system
binary) never sees it.

**Impact:** machine-specific SPIR-V from an arbitrary stale/newer binary,
silent build-vs-runtime compiler skew, and the expensive pinned CLI built
then ignored.
**Fix:** always use `$<TARGET_FILE:shadercross>` (drop the `find_program`),
or gate the system binary behind an explicit opt-in option plus a version
check against the pinned commit.

### L15. `playground --smoke` with a non-numeric frame count atoi's to 0 and silently runs interactive mode

**Category:** bug
**Locations:** `examples/playground/main.cpp:61-64`, `:83`; `examples/playground/playground.hpp:109`; `examples/playground/playground.cpp:99`

The optional frame count after `--smoke` is parsed with `std::atoi` and no
validation; `--smoke abc` and `--smoke 0` both yield 0. All smoke behavior
gates on `smoke_frames_ > 0`, so 0 disables self-driving, the frame-limited
exit, the screenshot request, and the post-run `verify_smoke_screenshot`
check — the binary launches the normal interactive game with no warning.

**Impact:** a typo in a scripted/CI invocation turns the integration test
into a session that never exits (hangs headless CI) and skips the screenshot
self-check.
**Fix:** validate with `std::from_chars`; reject non-positive/unparseable
counts with an error and exit code 2, matching the unknown-argument handling.

### L16. `BRACKEN_PLAYGROUND_SMOKE_TEST=ON` without `BRACKEN_BUILD_TESTS` registers a test ctest never runs

**Category:** build-system
**Locations:** `examples/playground/CMakeLists.txt:13-18`, `CMakeLists.txt:28-29`

The project's only `enable_testing()` sits inside `if(BRACKEN_BUILD_TESTS)`;
the playground registers `playground_smoke` via `add_test` whenever its own
option is ON. With tests off, no `CTestTestfile.cmake` is generated — the
registration is a silent no-op, and (verified empirically) `ctest` prints
"No test configuration file found!" yet **exits 0**, so even exit-code-checking
CI passes green without running the smoke test.

**Impact:** an integration-only CI runner (`-DBRACKEN_BUILD_TESTS=OFF
-DBRACKEN_PLAYGROUND_SMOKE_TEST=ON`) never runs the smoke test it asked for.
**Fix:** call `enable_testing()` whenever either option is ON (or
unconditionally), or emit a warning/fatal on the mismatched combination.

### L17. `.hlsli` include dependencies collected with a non-recursive GLOB while shader sources use GLOB_RECURSE

**Category:** build-system
**Locations:** `src/bracken/shaders/CMakeLists.txt:49-51` (vs. `:44-46`; DEPENDS at `:86`)

Sources are gathered with `GLOB_RECURSE "*.hlsl"` but the shared includes
used as recompile dependencies with a flat `GLOB "*.hlsli"`. All six current
`.hlsli` files live flat, so behavior is correct today — but the compile
command passes `-I ${CMAKE_CURRENT_SOURCE_DIR}`, so a subdirectory `.hlsli`
would compile fine while edits to it trigger no recompilation.

**Impact:** silently stale SPIR-V — one of the hardest failure modes to
diagnose in this pipeline (shaders/README.md documents the opposite contract:
"editing an include recompiles everything").
**Fix:** `file(GLOB_RECURSE ... CONFIGURE_DEPENDS "*.hlsli")` to match the
source glob.

---

## Findings — Application & input

### M4. Menu key handling matches keycodes (`SDLK_W`) while movement and key-edge tracking use scancodes

**Category:** latent-bug
**Locations:** `src/bracken/input_handler.cpp:162-201`, `:355-358`, `:109-114`

`process_event` handles menu navigation by switching on `event.key.key`
(layout-dependent keycodes: SDLK_UP/SDLK_W, SDLK_DOWN/SDLK_S, etc.), while
movement polling uses physical scancodes (SDL_SCANCODE_W/S/A/D) and the
`was_key_just_pressed` edge array is scancode-indexed. The same handler uses
two key-identity conventions for the same logical WASD keys.

**Impact:** on AZERTY and other non-QWERTY layouts, the physical key that
moves the character forward does not navigate "up" in menus, and vice versa —
silent, undocumented misbehavior for a large user population (arrows/Return/
gamepad behave consistently, which keeps menus navigable).
**Fix:** switch menu handling to `event.key.scancode`; if letter-based menu
keys are deliberate, document the split in the header.

### M23. `RenderContext::set_window_mode` exclusive mode silently falls back (and mis-records state) when `resolution_index` matches no native display mode

**Category:** latent-bug
**Locations:** `src/bracken/render/render_context.cpp:193-218`, `:179-183`

For exclusive fullscreen, an out-of-range `resolution_index` or a
no-match against `SDL_GetFullscreenDisplayModes` skips
`SDL_SetWindowFullscreenMode` entirely yet still calls
`SDL_SetWindowFullscreen(window_, true)` — with no log — producing
borderless-like fullscreen or, if a previous exclusive call set a mode,
staying exclusive at the OLD resolution (the stale mode is never cleared).
Worse, `current_window_mode_`/`current_resolution_index_` are recorded
*before* the attempt, so a retry with the same arguments no-ops at the
early-out. `GraphicsSettings::clamp_to_valid_ranges` clamps the index only to
`>= 0` while its comment claims "RenderContext re-clamps at apply time" —
which it does not.

**Impact:** a stale persisted `settings.cfg` `resolution_index` (monitor
changed, mode list changed) silently yields the wrong fullscreen
mode/resolution, and re-applying the same settings can never correct it.
**Fix:** on no-match, log a warning, clear any prior mode via
`SDL_SetWindowFullscreenMode(window_, nullptr)` for a deterministic
borderless fallback, and don't record the failed request as satisfied.

### L2. `Application` header implies an `on_init`/`on_shutdown` lifecycle but never states that `Application` itself never calls them

**Category:** api-quirk
**Locations:** `src/bracken/application.hpp:48`, `:54`, `:63-67`; `docs/architecture.md:84-89`; `src/bracken/application.cpp:111-115`

`init_engine()` is documented "Call before on_init()" and `shutdown_engine()`
"Call after on_shutdown()", with `on_init()` pure virtual — but `run()`
drives only `on_update`/`on_render`, and no engine code ever calls
`on_init`/`on_shutdown`. The host-driven pattern IS documented (repeatedly)
in user_guide.md and recorded as deliberate in technical_spec.md; the gap is
that the header doc comments and architecture.md's boot sequence don't
restate it, so a reader of only those sources writes
`init_engine(); run();` and gets nothing rendered with no diagnostic.

**Impact:** black-screen trap for consumers following the common
framework pattern where `run()` drives all hooks.
**Fix:** one sentence on the `on_init`/`on_shutdown` doc comments and in
architecture.md's boot sequence stating the host must invoke them (as the
playground's `start()`/`stop()` wrappers do). Do not change `run()` — the
asymmetry is recorded as intentional.

### L42. `InputState::attacking`/`sprinting` are never written by any engine code — host-owned fields with no in-header contract

**Category:** dead-code
**Locations:** `src/bracken/input_state.hpp:23-24`

`InputHandler` fills only movement flags, `move_dir_x/y`, and
`attack_dir_x/y`; the "action state" booleans are never assigned (only
default-value assertions in the unit test reference them). The host-owned
contract is stated in the new docs, but the header's own struct doc
("Contains movement and action state") implies the engine fills them.

**Impact:** a consumer reading only the header polls
`get_input().attacking` and gets a permanently-false value with no warning.
**Fix:** remove the two fields (hosts can keep action state in their own
layer) or comment them as host-owned and never written by `InputHandler`.
**Cross-ref:** RESOLUTIONS.md modernization L3 renamed
`controller_attack/sprint` for game-agnosticism; these fields were not
covered by that fix.

### L43. `InputHandler::set_player_screen_pos` stores state that nothing ever reads

**Category:** dead-code
**Locations:** `src/bracken/input_handler.hpp:40-44`, `:174-175`

The setter (comment: "Set player position for mouse direction calculation")
writes `player_screen_x_/y_` (defaults 640/360); repo-wide grep confirms no
reader anywhere. The promised mouse-direction calculation does not exist in
this codebase, and the setter leaks a gameplay concept ("player") into an
engine header, contrary to the game-agnostic rule.

**Impact:** dead API surface that misleads consumers into calling a no-op.
**Fix:** delete the setter and members (also update the two
technical_spec.md mentions that document the vestigial state).

---

## Findings — Core (logger, asserts, Result, Handle)

### M7. `StdoutSink` Trace/Debug output silently suppressed by SDL's own priority filter; the engine never configures SDL log priorities

**Category:** api-quirk
**Locations:** `src/bracken/core/logger.cpp:58`, `:31`, `:70`

The default sink forwards every message through
`SDL_LogMessage(SDL_LOG_CATEGORY_APPLICATION, ...)`. SDL3 applies its own
per-category threshold on top of `Logger::should_log`, and the application
category defaults to Info; nothing in the repo calls `SDL_SetLogPriority`.
Verified at runtime: TRACE/DEBUG messages are dropped inside SDL regardless
of `Logger::set_min_level`.

**Impact:** a consumer sets `set_min_level(LogLevel::Trace)` to diagnose a
problem and sees nothing on stdout — the documented log-level knob appears
broken for two of seven levels on the default sink.
**Fix:** have the Logger constructor (or StdoutSink) call
`SDL_SetLogPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_TRACE)`
once, since `should_log` already does the engine-side filtering; or print
directly to stdout.

### M24. Default-constructed `Handle{}` reports `is_valid() == true`

**Category:** api-quirk
**Locations:** `src/bracken/core/handle.hpp:11`, `:16`, `:18`

`Handle<Tag>` default-initializes to {index 0, generation 0} and
`is_valid()` tests only `index != k_invalid_index` (0xFFFFFFFF), so a
default-constructed handle claims validity and compares unequal to
`Handle::invalid()`. It happens never to resolve in a `HandlePool` (live
generations start at 1), but any consumer using `is_valid()` as a
"has this been assigned?" check gets a false positive. The sibling
`render_graph::ResourceHandle` has the opposite semantics
(default-constructed IS invalid), sharpening the trap.

**Impact:** silent false positives in consumer sentinel logic; two distinct
"empty" values that compare unequal.
**Fix:** make the default member initializer `index = k_invalid_index` so
`Handle{} == Handle::invalid()`; audit the few in-tree brace-init sites; at
minimum document the trap in architecture.md's gotcha list.

### L59. `HandlePool::clear()` resets generations, letting pre-clear handles alias post-clear slots

**Category:** latent-bug
**Locations:** `src/bracken/core/handle.hpp:107`, `:41`; documented at `docs/user_guide.md:928`, `docs/technical_spec.md:930`

`clear()` does `slots_.clear()`, discarding generation counters; fresh slots
restart at generation 1, so a retained pre-clear handle {idx, 1} validates
against a post-clear slot and `get()` returns a pointer to an unrelated
resource (classic ABA). The hazard is a documented contract ("don't keep
handles across a clear()") and `HandlePool` currently has no non-test
consumers, but the code-level hole remains and `clear()` is untested.

**Impact:** use-after-clear silently resolves to a different object rather
than failing validation.
**Fix:** preserve generation monotonicity across `clear()` (release each
occupied slot, or a pool-wide generation floor); add a `clear()` test.

### L60. Assertion failure messages are routed through the logger and dropped when min level is Off

**Category:** latent-bug
**Locations:** `src/bracken/core/assert.hpp:17`, `src/bracken/core/logger.cpp:82`

`detail::abort_with` passes the "file:line: assertion failed" message to
`log_plain`, which early-returns when `should_log(Fatal)` is false —
which it is when `min_level_ == LogLevel::Off` (or after `clear_sinks()`).
A failing BRACKEN_VERIFY/ASSERT/UNREACHABLE then calls `std::abort()` having
printed nothing.

**Impact:** a process that silenced logging (benchmark/test harness) aborts
on a contract violation with zero diagnostic — file, line, and expression
lost.
**Fix:** in `abort_with`, bypass the level filter (call `Logger::dispatch`
directly) and add an `fprintf(stderr, ...)` last-resort channel before
`std::abort()`.

### L61. `BRACKEN_ASSERT` discards its condition entirely under NDEBUG (not even type-checked)

**Category:** api-quirk
**Locations:** `src/bracken/core/assert.hpp:60-61`

Under NDEBUG the macro expands to `((void)0)`: the condition is neither
evaluated (standard assert semantics, documented) nor compiled, so an
asserted expression can bit-rot against renamed symbols and only fail to
compile in Debug builds. All 8 current non-test call sites are
side-effect-free.

**Impact:** side-effecting conditions silently change behavior between
Debug and Release; Release-only CI won't catch stale asserted expressions.
**Fix:** expand to `((void)sizeof(!(cond)))` under NDEBUG so the expression
stays type-checked at zero cost; add the no-side-effects rule to
architecture.md's gotcha list.

### L62. result.hpp comment says bool `init()/load()/save()` APIs "need to migrate" to `Result<void,E>`, contradicting the recorded no-mass-conversion policy

**Category:** doc-code-mismatch
**Locations:** `src/bracken/core/result.hpp:116`

The doc comment on the `Result<void, E>` specialization reads "the shape
every `bool init()/load()/save()`-style API needs to migrate to" — the
opposite of CLAUDE.md's convention ("existing bool+log APIs stay as they are
until touched — do not mass-convert"), recorded as deliberate in
RESOLUTIONS.md modernization H3.

**Impact:** a contributor takes the comment as marching orders and churns
APIs against written policy.
**Fix:** reword: "the shape new no-payload fallible APIs should take
(existing bool APIs migrate only when touched — see CLAUDE.md)".
**Cross-ref:** RESOLUTIONS.md modernization H3.

### L63. `std::hash<Handle>` shifts by 32 bits — UB when `size_t` is 32-bit

**Category:** latent-bug
**Locations:** `src/bracken/core/handle.hpp:131`

`(static_cast<size_t>(h.generation) << 32) ^ static_cast<size_t>(h.index)`
is undefined when `size_t` is 32 bits (shift count equals type width). All
current targets are 64-bit; portability-latent only.

**Impact:** 32-bit/WASM-class builds get UB in every unordered container
keyed on `Handle` (likely a `-Wshift-count-overflow` warning, then degraded
or broken hashing).
**Fix:** compute in a fixed-width type:
`static_cast<size_t>((static_cast<uint64_t>(h.generation) << 32) | h.index)`.

### L64. `Result`, `StringId`/fnv1a, and `HandlePool` have zero consumers outside core/ — infrastructure ahead of use

**Category:** dead-code
**Locations:** `src/bracken/core/result.hpp:23`, `src/bracken/core/hash.hpp:33`, `src/bracken/core/handle.hpp:26`

Exhaustive grep: `Result` is used only by its own header/test; `StringId`/
`fnv1a` only by hash.hpp, its test, and core.bench.cpp; `HandlePool` only by
handle.hpp/its test. The only cross-module consumer of any of these types is
`Handle<T>` in `AssetRegistry` — which notably rolls its own slot/generation
bookkeeping instead of using `HandlePool`. Zero-runtime-cost header
utilities, exercised solely by their own tests.

**Impact:** the utilities can drift from the needs of their first real
consumer (see L59/M24) precisely because nothing exercises them.
**Fix:** no code change now; adopt at natural touch points per CLAUDE.md
(new fallible asset/IO APIs return `Result`; name-keyed lookups take
`StringId`), fixing the companion Handle/HandlePool findings before first
real adoption.
**Cross-ref:** RESOLUTIONS.md modernization H3 (Result convention, no mass
conversion — covers Result adoption policy only, not StringId/HandlePool).

---
## Findings — Jobs

### H2. `JobSystem::wait()` on a worker thread never drains its own local queue or steals — nested submit+wait livelocks on a 1-worker pool

**Category:** hang-risk
**Locations:** `src/bracken/core/jobs/job_system.cpp:198-220` (helping loop), `:92-96` (submit → local deque), `:32-35` (default worker count), `src/bracken/core/jobs/job_system.test.cpp:68-87`

`wait()`'s helping loop only pops the global queue; the steal fallback is
gated on `t_worker_index_ < 0` (non-workers only) and there is no
`try_pop_local_` call. `submit()` from a worker pushes onto that worker's own
local deque, so a job that submits sub-jobs and then waits depends entirely
on *other* workers stealing them. Default `init()` creates
`hardware_concurrency()-1` workers clamped to 1, and `try_steal_` returns
false outright with ≤1 worker queues — so on ≤2-hardware-thread machines the
livelock is total and deterministic: the waiting worker yield-spins forever
while its sub-jobs sit unreachable in its own queue. The bundled test
`WaitByHelpingDrainsNestedSubmits` exercises exactly this and hangs
`bracken_tests` on such hosts. Even on larger pools, all workers waiting
concurrently with runnable jobs stranded in local queues stalls the pool.
(`parallel_for` is unaffected on 1-worker pools — serial fallback.)

**Impact:** deterministic, silent, unrecoverable livelock (100% CPU yield
spin) for the ordinary fork-join pattern on small VMs/CI containers; the
test suite itself hangs there. user_guide.md documents the sharp edge, but
the code has no guard and the test violates it.
**Fix:** in `wait()`, when the caller is a worker, also
`try_pop_local_(t_worker_index_, job)` and allow `try_steal_` for workers;
alternatively enforce a minimum of 2 workers in `init()` and ban nested
submit+wait explicitly.

### M31. `TaskHandle::error()` reads `Task::error` without checking `done` — data race against the worker's write

**Category:** latent-bug
**Locations:** `src/bracken/core/jobs/task_handle.hpp:24`; `src/bracken/core/jobs/job_system.cpp:161`, `:165`

The worker writes `task->error` and only afterwards release-stores
`done = true`. `is_done()`/`has_error()` correctly acquire-load `done`
first, but `error()` returns `task_->error` unconditionally: polled on an
in-flight task it copies a non-atomic `std::exception_ptr` concurrently with
the worker's write — a data race/UB, and no happens-before edge even absent
tearing. No in-tree caller invokes `error()` at all; latent for library
consumers.

**Impact:** formal UB exactly in the failure case the caller is probing for.
**Fix:** gate on done:
`return (task_ && task_->done.load(std::memory_order_acquire)) ? task_->error : nullptr;`
and/or document that `error()` is valid only after `is_done()`.

### M32. `parallel_for` rethrows the first chunk exception before waiting on remaining handles — later chunks can run against dead stack frames

**Category:** latent-bug
**Locations:** `src/bracken/core/jobs/parallel_for.hpp:59-61`, `:57`; `src/bracken/core/jobs/job_system.cpp:217-219`

Chunk handles are waited on sequentially and `wait()` rethrows a stored
exception, so if chunk k throws, handles k+1..N are never joined while those
chunks may still be queued or executing. The typical `fn` is a
`[&]`-capturing lambda over caller locals, so once the caller unwinds,
in-flight chunks dereference destroyed stack data. Exception propagation
through `wait()` is an advertised, tested feature, so throwing bodies are
legitimate API use. Cannot fire in-tree today (JobSystem never initialized →
serial fallback), which keeps this medium.

**Impact:** one throwing chunk converts `parallel_for` into use-after-scope
UB in concurrently running chunks — timing-dependent memory corruption.
**Fix:** on exception, capture it, keep waiting on all remaining handles,
then rethrow the first captured exception; document that `fn` should not
throw otherwise.

### M33. `shutdown()` silently discards queued-but-unstarted jobs and zeroes `pending_`; a later `wait()` on their handles spins forever

**Category:** hang-risk
**Locations:** `src/bracken/core/jobs/job_system.cpp:56-85` (queue clears at `:72-82`), `:165` (only site setting `done`), `:198-220`

`shutdown()` joins workers (each finishes only its current job), clears all
queues, and force-resets `pending_` to 0. Tasks whose jobs were still queued
are dropped with `done` permanently false — no error, no log. A subsequent
`wait()` on such a handle yield-spins forever (nothing to help with;
`try_steal_` false once queues are cleared), while `wait_all()` returns
instantly, masking the loss. The singleton destructor calls `shutdown()`, so
this also applies at static destruction.

**Impact:** a consumer that shuts down (or lets the singleton destruct)
with jobs queued and later waits on a retained handle hangs a thread at 100%
CPU with no diagnostic; fire-and-forget work is silently lost.
**Fix:** in `shutdown()`, mark every discarded Job's task done (optionally
storing a "discarded at shutdown" error) before clearing queues so waiters
wake; or drain via `wait_all()` and document the drain requirement.

### L65. `wait()` busy-yields when it cannot help — no condition-variable block

**Category:** api-quirk
**Locations:** `src/bracken/core/jobs/job_system.cpp:198-220` (yield at `:215`; contrast `wait_all` at `:222-239`)

When the global queue is empty and there is nothing to steal, `wait()` falls
through to `std::this_thread::yield()` and re-checks — a pure spin, unlike
`wait_all()`, which parks on `wait_all_cv_` with a 50 ms timeout. There is
no per-task CV or atomic-wait.

**Impact:** an external thread waiting on one long-running job burns an
entire core for the duration, degrading the very workers it waits on.
**Fix:** use C++20 `std::atomic<bool>::wait`/`notify_one` on `Task::done`
(notify in `execute_`), or a completion CV mirroring `wait_all()`.

### L66. core.bench.cpp comment says the JobSystem singleton "has no re-init path", but shutdown()+init() fully supports re-initialization

**Category:** doc-code-mismatch
**Locations:** `src/bracken/core/core.bench.cpp:30-32`; `src/bracken/core/jobs/job_system.cpp:26-54`, `:56-85`

`shutdown()` restores the object to its pre-init state and `init()` guards
only on `initialized_`, so init-after-shutdown is supported by the
implementation — but no test exercises the sequence (only init idempotence),
so the capability is real yet unverified.

**Impact:** misleads maintainers into treating the pool as one-shot, or
leaves the supported re-init path free to rot.
**Fix:** correct the comment (the static guard avoids redundant init across
benchmark repetitions); add an `InitAfterShutdown` test to pin the behavior.

---

## Findings — Assets (registry, watcher, hot reload)

### M8. `PipelineRegistry::enable_hot_reload` logs "hot-reload enabled" and latches `watcher_` even when the FileWatcher is uninitialized and nothing was watched

**Category:** latent-bug
**Locations:** `src/bracken/gpu/pipeline_registry.cpp:225-239` (latch at `:233`, log at `:238`); `src/bracken/core/asset/file_watcher.cpp:96`; `src/bracken/application.cpp:144`

`watch_directory` silently returns `k_invalid_handle` when the watcher was
never `init()`ed (i.e. every run without `BRACKEN_HOT_RELOAD=1`);
`enable_hot_reload` never checks the returned handle, stores it,
unconditionally logs "PipelineRegistry hot-reload enabled at {}", and sets
`watcher_` *before* the watch attempt — after which the idempotency guard
early-outs forever, so a second call after the watcher IS initialized is a
no-op. `Application::init_renderer` calls this unconditionally.

**Impact:** (1) every default run emits a misleading "enabled" log, sending
shader-reload debugging down the wrong path; (2) a host that initializes its
FileWatcher after Application's unconditional first call can never actually
enable shader hot reload — the latch blocks retry.
**Fix:** check the `watch_directory` return; on failure log a warning, leave
`watcher_` null so a later call can retry, and skip the success log (or
early-return on `!watcher.is_initialized()`).

### M25. `AssetRegistry` duplicate-name load silently orphans the old entry, and unloading the old handle then erases the newer entry's name mapping

**Category:** latent-bug
**Locations:** `src/bracken/core/asset/asset_registry.hpp:49`, `:95`

`load` assigns `name_to_handle_[entry->name] = h` unconditionally: a second
load under an existing name overwrites the mapping while the previous entry
stays alive (holding memory and its file watch), reachable only by handle.
`unload` then erases the name-map entry keyed by the slot's stored name
without checking the mapping still points at this handle — so
`load("hero", a); load("hero", b); unload(a)` leaves b alive and valid but
`find("hero")` returning `Handle::invalid()`. Silent; user_guide.md even
advises unloading the old handle — the exact corrupting sequence.

**Impact:** live asset unfindable by name; the orphaned old entry keeps
firing hot-reloads on a value nothing can look up.
**Fix:** in `load`, detect an existing live mapping and reject or unload the
previous entry first; in `unload`, erase the name mapping only if
`name_to_handle_[slot->name] == h`.

### L67. architecture.md presents `AssetRegistry` as the asset module's hot-reload mechanism, but no engine system instantiates it

**Category:** doc-code-mismatch
**Locations:** `docs/architecture.md:23`; `src/bracken/core/asset/asset_registry.hpp:27`; `src/bracken/application.cpp:144`, `:146`

The module table leads with "`AssetRegistry` (typed, stable handles,
hot-reload)", but the engine's two real hot-reload paths bypass it entirely:
`init_renderer` wires `PipelineRegistry::enable_hot_reload` and
`ModelManager::enable_hot_reload` directly to `FileWatcher::instance()`.
`AssetRegistry<T>` is referenced only by its own header and unit test — a
host-facing utility, not the engine's hot-reload machinery.

**Impact:** misdirects debugging and code archaeology; implies AssetRegistry
is hardened by engine runtime paths when it is test-only.
**Fix:** amend architecture.md (module table + hot-reload paragraph) to say
built-in shader/model reload registers watches directly on FileWatcher, and
AssetRegistry is a general-purpose utility for host asset types.
**Cross-ref:** technical_spec.md:1264 and user_guide.md:1327 already note
the discrepancy; architecture.md is still uncorrected.

---

## Findings — Memory

### L44. `Arena::allocate` silently returns misaligned memory for non-power-of-two alignment

**Category:** api-quirk
**Locations:** `src/bracken/memory/arena.cpp:13`, `:77`, `:94`; `src/bracken/memory/arena.hpp:26`

`align_up` uses the pow2-only formula `(v + (a-1)) & ~(a-1)`; caller-supplied
`align` passes through unchecked (only `align == 0` is remapped). The
power-of-two precondition is documented in the new docs but not enforced by
a debug assert and not stated on the declaration in arena.hpp.

**Impact:** a stride-derived alignment (e.g. 12) silently yields a
misaligned pointer — UB on placement, invisible on x86 for most types.
**Fix:** `BRACKEN_ASSERT(std::has_single_bit(align))` in `allocate` and a
precondition comment on the declaration.

### L45. `Arena::allocate(0, ...)` returns nullptr, indistinguishable from allocation failure

**Category:** api-quirk
**Locations:** `src/bracken/memory/arena.cpp:67`, `:85`, `:88`, `:96`

Zero-size requests and every genuine failure path (fixed-buffer exhaustion,
malloc failure in grow, post-grow recheck) all return nullptr; the zero-size
behavior is undocumented in arena.hpp. `FrameAllocator` inherits the same
semantics.

**Impact:** nullptr-is-fatal callers spuriously report OOM on empty batches;
nullptr-tolerant callers can mask real exhaustion.
**Fix:** return a valid non-null pointer for size==0 (current cursor, no
bump) so nullptr uniquely means failure — or document the contract on the
declaration.

### L46. `Arena::allocate` and `Arena::create` are fallible (return nullptr) but not `[[nodiscard]]`

**Category:** design-violation
**Locations:** `src/bracken/memory/arena.hpp:26`, `:28`

No declaration in the memory module carries `[[nodiscard]]`, despite
CLAUDE.md's convention and the prior campaign's M1 sweep (79 occurrences
landed elsewhere; these factory-style allocation APIs were missed). Note:
several test files deliberately discard returns and will need `(void)`/
`std::ignore` when the attribute lands.

**Impact:** discarded `arena.create<T>(...)` compiles without warning,
hiding unchecked-failure patterns the convention exists to flag.
**Fix:** add `[[nodiscard]]` to both; adjust the handful of test call sites.
**Cross-ref:** RESOLUTIONS.md modernization M1 ("zero [[nodiscard]]",
marked fixed) — these sites were missed by that sweep.

---
## Findings — GPU layer

### M29. `GPUTexture::create_2d_array` returns a texture with no pixel data when the upload stage fails, despite the header promising nullptr on failure

**Category:** latent-bug
**Locations:** `src/bracken/gpu/gpu_texture.cpp:399`, `:406`, `:420`, `:428`; `src/bracken/gpu/gpu_texture.hpp:124`

After `SDL_CreateGPUTexture` succeeds, any failure in the upload stage
(transfer-buffer creation, mapping, command-buffer acquisition, copy-pass
begin) logs and executes `return texture;` — a valid-looking `GPUTexture`
whose contents were never written, while the header @return says "nullptr on
failure". The sibling `create_2d` correctly returns nullptr on all four
analogous paths. The terrain material array's null-check at
`terrain_renderer.cpp:189` can never fire for upload failures.

**Impact:** under VRAM exhaustion/device loss, consumers render garbage
sampled from an undefined texture with only a log line; callers cannot
detect the failure programmatically.
**Fix:** release the created texture and return nullptr on any
post-creation upload failure, matching the header contract and `create_2d`.

### M30. `GPUTexture::reload_from_file` always reloads with default RGBA8, dropping the original sRGB format

**Category:** latent-bug
**Locations:** `src/bracken/gpu/gpu_texture.cpp:91`, `:99-104`; `src/bracken/gpu/gpu_texture.hpp:162`, `:47-49`

`reload_from_file` calls `load_from_file` without forwarding a format, so
the `TextureFormat::RGBA8` default always applies; it then steals the fresh
texture's handle and overwrites `format_`. A texture created as RGBA8_SRGB
comes back linear after a successful reload. No in-engine callers — it is
the documented public hot-reload building block (the header itself tells
callers to use RGBA8_SRGB for albedo art, and user_guide.md demonstrates
this API as the texture hot-reload pattern).

**Impact:** hot-reloading a color/albedo texture silently disables sRGB
linearization — visibly washed-out rendering with no error or log.
**Fix:** store the creation `TextureFormat` (or derive it from `format_`)
and pass it through in `reload_from_file`.

### L34. Docs and comments describe the particle blend as ONE/ONE, but the configured factors are (SRC_ALPHA, ONE) with the shader forcing alpha to 1.0

**Category:** doc-code-mismatch · **Reported by two areas:** gpu and effects (merged)
**Locations:** `src/bracken/gpu/pipeline_registry.hpp:36`; `src/bracken/gpu/gpu_uniforms.hpp:339`; `src/bracken/render/effect_renderer.hpp:36`; `src/bracken/shaders/particle.vert.hlsl:67`; `docs/effects_system.md:102`; configured blend at `src/bracken/gpu/gpu_types.hpp:157-165`, contract at `src/bracken/shaders/particle.frag.hlsl:6`

Five sites state "ONE/ONE additive" without noting it is only the effective
behavior: `BlendMode::Additive` is configured as color (SRC_ALPHA, ONE), and
ONE/ONE holds only because particle.frag writes alpha 1.0 — a dependency
correctly explained in particle.frag.hlsl's own header,
`pipeline_registry.cpp:596-598`/`:873`, and the new docs, but not at the
stale sites.

**Impact:** changing particle.frag's output alpha (e.g. per-pixel fade)
silently changes the color blend to alpha-weighted additive, contradicting
the doc claims; auditors comparing blend state to docs find a phantom
mismatch and may "fix" the wrong side.
**Fix:** reword all stale sites to the configured-plus-effective phrasing:
"configured (SRC_ALPHA, ONE); the fragment shader writes alpha 1.0 so the
effective color blend is (ONE, ONE)" — or add a true ONE/ONE blend preset
for the particle pipeline.

### L51. `GPUDevice::init`'s `prefer_low_power` parameter is documented but never read

**Category:** doc-code-mismatch
**Locations:** `src/bracken/gpu/gpu_device.hpp:46`, `:49`; `src/bracken/gpu/gpu_device.cpp:18`, `:45-49`

The header promises "If true, prefer integrated GPU over discrete", but the
implementation never reads the parameter — `SDL_CreateGPUDevice` is called
with only shader-format flags, debug_mode, and a null driver name.

**Impact:** a consumer passing `prefer_low_power=true` silently gets default
device selection.
**Fix:** wire it via `SDL_CreateGPUDeviceWithProperties` +
`SDL_PROP_GPU_DEVICE_CREATE_PREFERLOWPOWER_BOOLEAN`, or remove the parameter
and doc claim.

### L52. `get_bytes_per_pixel` defaults unknown SDL formats to 4 bytes, mis-sizing `upload()` for wide formats

**Category:** latent-bug
**Locations:** `src/bracken/gpu/gpu_texture.cpp:82-83`, `:501`

The switch handles R8/R16 and five 4-byte formats, then `default: return 4;`.
`upload()` uses it to size the transfer buffer and memcpy, so a texture
created via the explicit-SDL-format overload with e.g. R16G16B16A16_FLOAT
(8 B/px) gets a half-sized, silently truncated upload. No current engine
caller hits the path (`upload()` itself has no in-engine callers).

**Impact:** external consumers pairing the advanced overload with `upload()`
get corrupted uploads with no error.
**Fix:** use `SDL_GPUTextureFormatTexelBlockSize`, or at minimum log/assert
in the default case.

### L53. `GPUTexture::upload` releases its transfer buffer immediately after recording, relying on SDL deferred-release semantics

**Category:** api-quirk
**Locations:** `src/bracken/gpu/gpu_texture.cpp:542-549`

`upload()` records the copy into the caller's command buffer and calls
`release_transfer_buffer` before the caller has submitted; the in-source
comment acknowledges this only works because SDL defers destruction, and the
method allocates a fresh transfer buffer per call. No in-engine callers (the
text renderer keeps its own persistent transfer buffer).

**Impact:** correct today under SDL3 GPU's reference-tracked release, but
fragile against SDL semantic changes; per-frame external use churns a
transfer-buffer allocation per call.
**Fix:** defer the release until after submission (per-frame staging pool or
pending-release list) and make the comment state the actual SDL guarantee.

### L54. gpu_types.hpp declares five dead structs that shadow the live uniform contract

**Category:** dead-code
**Locations:** `src/bracken/gpu/gpu_types.hpp:52-57`, `:64-69`, `:72-75`, `:78-84`, `:87-91`

`GrassInstance` and the alignas(16) `CameraUniforms`/`ModelUniforms`/
`LightUniforms`/`TimeUniforms` are referenced nowhere; the live CPU↔GPU
contract is gpu_uniforms.hpp. The section banner ("Uniform Buffer Structures
— Must match shader layouts") plus the fact that model_instanced.vert.hlsl
and particle.vert.hlsl have a cbuffer literally named `CameraUniforms`
(whose real CPU mirror is `gpu::InstancedCameraUniforms`) makes the dead
`CameraUniforms` actively misleading — it is even byte-identical to the live
struct, so edits to it silently do nothing.

**Impact:** grepping the HLSL cbuffer name lands on the dead struct;
maintenance noise.
**Fix:** delete the five structs and the banner, leaving vertex formats,
`TextureFormat`, `BlendMode`, and the attribute helpers.

---

## Findings — Shaders & shader tooling

### M9. `ShaderResources` counts are never validated against the SPIR-V or device limits

**Category:** api-quirk
**Locations:** `src/bracken/gpu/gpu_shader.hpp:22`, `:27`; `src/bracken/gpu/gpu_shader.cpp:141`, `:158`

The hand-maintained sampler/storage/uniform counts are copied verbatim into
`SDL_ShaderCross_GraphicsShaderResourceInfo` with no cross-check against
what the SPIR-V actually declares (the vendored library's reflection API
exists but is never called) and no check against device limits. The header
comment itself warns wrong counts "may cause silent failures or crashes";
every `create_*_pipeline` in pipeline_registry.cpp hand-maintains them
(~15 functions, comment-only bookkeeping).

**Impact:** a count drifting from the shader's declarations (add a sampler
to a .frag.hlsl, forget the matching create_*) produces silent misbinding,
garbage rendering, or a backend crash — the documented most-common way to
silently break rendering, and nothing catches it.
**Fix:** lightweight SPIR-V reflection at load (descriptor-binding scan, or
`SDL_ShaderCross_ReflectGraphicsSPIRV`) with a debug-build
BRACKEN_ASSERT/log on mismatch; optionally validate against
`SDL_GetGPUDeviceProperties()`.

### M10. `ShaderManager::get` silently ignores the `ShaderResources` argument on cache hits

**Category:** latent-bug
**Locations:** `src/bracken/gpu/gpu_shader.cpp:200`, `:205`, `:182`; `src/bracken/gpu/gpu_shader.hpp:160`

The cache key is (path, stage, entry_point) only; resource counts are baked
into the SDL_GPUShader at first load, and later `get()` calls with different
counts return the first-baked shader unconditionally — no compare, no log.
PipelineRegistry requests the same .spv from multiple factories with
independently hand-typed counts (model.frag ×2, shadow_depth.frag ×2,
fullscreen.vert ×8) — identical today, so latent; any future divergence is
first-caller-wins, with hot-reload rebuild order deciding the outcome.

**Impact:** silent misbinding that compounds M9, load-order dependent.
**Fix:** store the creation counts in `GPUShader` and assert/warn on
mismatch at cache hits, or fold the counts into the cache key.

### L4. Shaders README says compiled `.spv` land in `<build>/shaders/`; the actual output dir is `build/src/bracken/shaders/`

**Category:** doc-code-mismatch
**Locations:** `src/bracken/shaders/README.md:6`; `src/bracken/shaders/CMakeLists.txt:40`; `src/bracken/CMakeLists.txt:132`

`SHADER_OUTPUT_DIR` is `${CMAKE_CURRENT_BINARY_DIR}/../shaders`, which
resolves to `build/src/bracken/shaders/` (verified on disk; `build/shaders/`
does not exist). The path is exported as the INTERNAL cache variable
`BRACKEN_SHADER_OUTPUT_DIR`.

**Impact:** anyone following the README to inspect or script against
compiled shaders finds nothing.
**Fix:** correct the README and point readers at `BRACKEN_SHADER_OUTPUT_DIR`
as the authoritative path.

### L13. GPUShader/ShaderManager default entry point is "main", which never matches engine-built SPIR-V

**Category:** api-quirk
**Locations:** `src/bracken/gpu/gpu_shader.hpp:68`, `:75`, `:148`, `:154`; build entry points at `src/bracken/shaders/CMakeLists.txt:63/67`

The build compiles every shader with entry point VSMain/PSMain, so the
"main" defaults on `load_spirv`/`create_from_spirv`/`get_vertex`/
`get_fragment` can never load an engine-built .spv. Every in-tree call site
passes the entry explicitly; only `ShaderProgram::load` has correct
defaults. Failure mode is backend-dependent: Metal/D3D12 fail at load
(nullptr + SDL log), but on Vulkan the shader *loads successfully* (the
entry name is unvalidated until pipeline creation), surfacing later and more
confusingly.

**Impact:** the defaulted parameters are effectively unusable API surface
with a delayed, misleading failure mode.
**Fix:** per-stage "VSMain"/"PSMain" defaults (the callers know the stage),
or drop the defaults to force explicit entry points.

### L14. `gpu::ShaderProgram` has no in-tree callers (dead API, only usable with empty `ShaderResources`)

**Category:** dead-code
**Locations:** `src/bracken/gpu/gpu_shader.hpp:194`; `src/bracken/gpu/gpu_shader.cpp:265`

The vertex+fragment pair loader has zero callers and hardwires default
(all-zero) `ShaderResources`, so it could only ever correctly load shaders
declaring no resources at all — yet user_guide.md actively recommends it as
the loader with correct entry-point defaults.

**Impact:** a consumer reaching for the only convenient loader hits silent
resource misbinding on any shader with samplers/uniforms.
**Fix:** delete `ShaderProgram`, or give `load()` ShaderResources parameters
and add a test.

### L18. `bracken_copy_shaders` never removes stale `.spv` files from the staging directory

**Category:** build-system
**Locations:** `src/bracken/CMakeLists.txt:136`, `:138`

Staging uses `cmake -E copy_directory`, which only adds/overwrites. Verified
today: `build/examples/playground/shaders/` still contains
`billboard.{vert,frag}.spv` although the billboard pipeline and its HLSL were
deleted in the 2026-07 campaign (RESOLUTIONS.md shaders L-2). Note the
originally suggested `copy_directory_synchronized` does not exist in CMake;
the viable fix is remove_directory + copy_directory.

**Impact:** deleted/renamed shaders keep shipping next to the executable; a
pipeline still referencing a deleted shader works locally and breaks on
fresh checkouts.
**Fix:** clear the destination before copying in `bracken_copy_shaders`
(and consider the same for `SHADER_OUTPUT_DIR`).
**Cross-ref:** RESOLUTIONS.md shaders.md L-2 (billboard deletion — source of
the observed stale files); the copy semantics themselves were untracked.

---

## Findings — Scene & RenderScene

### M5. `RenderScene` draw flags default to false at construction but `clear()` resets them to true

**Category:** latent-bug
**Locations:** `src/bracken/scene/render_scene.hpp:200-204`, `:91-94`; `src/bracken/scene/render_scene.cpp:20-25`

The five world-element flags (skybox/rocks/trees/ground/grass) have member
initializers of `false`, but `clear()` — under a comment reading "Reset
world element flags to defaults" — sets them all `true`. A freshly
constructed scene therefore behaves differently from a cleared one, and
since `has_3d_content()` checks those same flags, a first frame submitted
without `clear()` and without model commands renders nothing at all, no log.
The clear-to-true behavior is the tested, documented per-frame-toggle design;
the false member initializers are the anomaly. The playground is unaffected
only because it happens to call `clear()` before setting flags.

**Impact:** out-of-tree hosts submitting a freshly constructed scene (or
clearing at end-of-frame) get a silent blank/incomplete first frame.
**Fix:** make the member initializers `true` so constructed and cleared
states match; extend the `clear()` doc comment to say it resets draw flags
to true and does not clear particle spawns.

### L23. `has_3d_content()` ignores billboards, debug lines, and lights — a frame with only those skips the entire 3D pass

**Category:** latent-bug
**Locations:** `src/bracken/scene/render_scene.hpp:173-175`; `src/bracken/scene/scene_renderer.cpp:539`, `:548`, `:594-608`, `:623`, `:1033`, `:794-805`

The gate considers only the skybox/ground/grass flags and model/skinned
commands, so a debug-lines-only or lights-only frame skips the cluster-light
build, debug-line upload, and debug-line render — while billboards still
draw in the ungated UI phase, a confusing asymmetry. Triggering requires the
host to disable skybox/ground/grass every frame (clear() resets them true).
This is a documented known limitation in the new docs, not a recorded
design decision.

**Impact:** gizmo-only debug viewers silently render nothing for debug
geometry while billboards in the same scene appear.
**Fix:** include `!debug_lines_.empty()` (and arguably billboards) in
`has_3d_content()`, or note the limitation on the method declaration itself.

### L24. `set_draw_rocks`/`set_draw_trees` flags are stored and reset but never read by any renderer

**Category:** dead-code
**Locations:** `src/bracken/scene/render_scene.hpp:163-164`, `:168-169`, `:201-202`; `src/bracken/scene/render_scene.cpp:22-23`

Setters, getters, members, and `clear()` resets all exist; the only reader
is the unit test. Only skybox/ground/grass gate any rendering behavior.

**Impact:** `set_draw_rocks(false)` has no effect — API implying a
capability that does not exist.
**Fix:** delete the rocks/trees setters/getters/members/resets (updating
the four doc references), or wire them into a draw path if one is intended.

---
## Findings — Models (glTF loader, ModelManager)

### M11. Separate (non-ORM) glTF occlusion textures are parsed but never uploaded or sampled; `occlusion_texture` is vestigial

**Category:** latent-bug · **Reported by two areas:** models and model-pbr-shading (merged)
**Locations:** `src/bracken/model_loader.cpp:237-248` (extraction), `:792-795` (pixels cleared, "not uploaded yet"), `:820` (only other reference); `src/bracken/model_loader.hpp:76`; `src/bracken/shaders/model.frag.hlsl:139-140`

An occlusionTexture whose image differs from the metallic-roughness source
has its pixels copied into `PBRMaterial::occlusion_pixels`, which
`upload_to_gpu` then clears without ever creating a GPU texture (unlike the
adjacent normal/MR/emissive uploads). `PBRMaterial::occlusion_texture` is
never assigned anywhere. The model fragment shaders apply AO exclusively via
the ORM-packed path (`hasPackedOcclusion` gating the R channel of the MR
texture); there is no separate AO sampler binding, so `occlusion_strength`
is applied to nothing for standalone maps.

**Impact:** assets authored with a standalone AO map — a common DCC export
layout — silently lose all baked ambient occlusion (flat ambient, no error
or log), and the loader wastes a full RGBA copy at load only to discard it.
**Fix:** either upload `occlusion_pixels` to `occlusion_texture` and bind it
as a fifth material texture (sampled when `hasPackedOcclusion` is false), or
warn at load and delete the never-populated fields; document the
ORM-packed-only constraint in docs/skinned_mesh_format.md either way.
**Cross-ref:** acknowledged only by the in-code comment at
`model_loader.cpp:792-793`; absent from architecture.md's stub list and
RESOLUTIONS.md.

### M13. Inverse-bind-matrix and animation-sampler accessors are read raw — bypassing the packed-stride guard, without componentType checks

**Category:** latent-bug
**Locations:** `src/bracken/model_loader.cpp:527-538` (IBMs), `:643-657` (sampler input/output); contrast `:298-314` (stride guard), `:478-487` (JOINTS/WEIGHTS validation)

Vertex attributes go through `get_packed_data` (warns and skips on non-tight
byteStride) and JOINTS_0/WEIGHTS_0 get componentType validation — but
skeleton IBMs and animation sampler data are computed via raw pointer
arithmetic and `reinterpret_cast` to float with no stride or componentType
check. Core glTF explicitly permits normalized u8/u16 animation outputs for
rotation (gltfpack/meshoptimizer quantized exports), which this code would
silently reinterpret as float.

**Impact:** a quantization-optimized .glb — a realistic asset-pipeline
output violating no documented constraint — loads "successfully" but
animates as visual noise; malformed strided files decode garbage IBMs/keys
silently.
**Fix:** validate componentType == FLOAT (and tight stride) on sampler
input/output and the IBM accessor, warning and dropping the channel/skin
data on mismatch, mirroring the JOINTS_0/WEIGHTS_0 pattern.

### M16. Base-color texture lookup indexes a compacted `loaded_textures` vector by raw glTF image index

**Category:** latent-bug
**Locations:** `src/bracken/model_loader.cpp:156-168`, `:205`, `:457-458`

`load_glb` builds `loaded_textures` by iterating `gltf.images` and skipping
any image with empty pixel data — a compacted vector — but
`texture_image_idx` is the raw glTF image index, and line 457 indexes the
compacted vector with it. All other texture types index `gltf.images[src]`
directly; only base color goes through the compacted vector. Trigger nuance:
an embedded-image decode failure aborts the whole load in tinygltf, so the
skew only occurs when an image is kept empty while the load succeeds — in
practice an external-URI image that fails to resolve (tinygltf downgrades to
a warning and keeps an empty image).

**Impact:** in such a multi-image .glb, every base-color texture after the
gap is assigned the wrong image's pixels (off-by-N shift) or silently
dropped — meshes render with another mesh's texture, no error logged.
**Fix:** index `gltf.images[texture_image_idx]` directly like the other
texture paths (checking `!image.image.empty()` at use) and delete the
compacted staging vector.

### M17. ModelManager header promises raw `Model*` stability, but `reload_model` destroys and replaces the Model object

**Category:** doc-code-mismatch · **Reported by two areas:** models and assets (merged; one verification pass argued Low given the docs coverage — kept Medium because the consequence is use-after-free)
**Locations:** `src/bracken/model_loader.hpp:180-181`, `:206-208`; `src/bracken/model_loader.cpp:898-909` (replacement at `:909`)

The deleted copy/move rationale comment says "handles and raw Model* handed
to callers must stay stable", and `reload_model`'s doc claims it "swap[s]
GPU buffers in place". The implementation parses into a fresh
`std::make_unique<Model>()`, frees the old model's GPU resources, and does
`models_[handle] = std::move(fresh)` — destroying the old object. Only the
handle→slot mapping is stable; every raw `Model*` previously returned by
`get_model` dangles after a reload. Engine-internal callers re-resolve per
frame and are safe; the user-facing docs state the real contract — the
header comments are the stale side.

**Impact:** with `BRACKEN_HOT_RELOAD=1`, a host that caches `Model*` (as the
header comment invites) gets a use-after-free the frame after the .glb is
touched on disk.
**Fix:** reword both comments — handles are stable; raw `Model*` are
invalidated by `reload_model`/hot reload; re-resolve via the handle each
frame. Alternatively make the pointer genuinely stable by move-assigning the
fresh contents into the existing object.

### L7. `enable_hot_reload` doc comment says ".glb/.gltf" but the loader is binary-only

**Category:** doc-code-mismatch
**Locations:** `src/bracken/model_loader.hpp:210-211`; `src/bracken/model_loader.cpp:137`

`ModelLoader::load_glb` only ever calls
`tinygltf::TinyGLTF::LoadBinaryFromFile` (GLB containers only); a JSON
.gltf fails at load, so no .gltf model can exist in the manager to be
hot-reloaded.

**Impact:** cosmetic — implies text .gltf support that does not exist.
**Fix:** say ".glb" only (or dispatch to `LoadASCIIFromFile` by extension if
.gltf support is wanted).

### L68. architecture.md says model bounds come from the POSITION accessor min/max; code accumulates them from every parsed vertex

**Category:** doc-code-mismatch
**Locations:** `docs/architecture.md:234-235`; `src/bracken/model_loader.cpp:374-379`

`load_glb` never reads `accessor.minValues/maxValues`; it initializes bounds
to ±1e10 and expands per vertex. Identical results for conformant files, but
the mechanism described is wrong (stale accessor min/max would still yield
correct bounds; stride-rejected primitives contribute nothing).

**Impact:** documentation inaccuracy only; could mislead a load-time
optimization or non-conformant-asset debugging.
**Fix:** reword to "bounds are accumulated from every parsed vertex position
(not read from the accessor's declared min/max)".

---

## Findings — Animation & IK

### M12. glTF animation sampler interpolation mode is ignored — STEP smoothed, CUBICSPLINE data misread

**Category:** latent-bug
**Locations:** `src/bracken/model_loader.cpp:630-679` (sampler fetched at `:641`, never reads `.interpolation`); `docs/skinned_mesh_format.md:54-60`

All keys are stored as plain time/value pairs and `interpolate_keyframes`
always lerps/slerps, so STEP samplers (Blender "Constant" keyframes) are
silently smoothed. Worse, CUBICSPLINE output stores three elements per key
(in-tangent, value, out-tangent); the loader indexes output as tightly
packed, reading a garbage mix of tangents and values covering the first
third of the data. The authoring contract
(docs/skinned_mesh_format.md, "Animation channels") says nothing about
interpolation modes.

**Impact:** spec-valid GLBs animate with unintended smoothing (STEP) or as
garbage (CUBICSPLINE), silently, with no warning to the asset author.
**Fix:** read `sampler.interpolation`: implement STEP (or warn), and reject
or warn-and-skip CUBICSPLINE rather than misreading it; add the constraint
to docs/skinned_mesh_format.md.

### M14. Parents-before-children joint-order requirement is neither documented in headers/docs nor validated at load

**Category:** latent-bug
**Locations:** `src/bracken/animation/animation_player.cpp:197-205`; `src/bracken/animation/ik_solver.cpp:217-225`; `src/bracken/model_loader.cpp:542-543`; precondition stated only at `src/bracken/animation/animation.bench.cpp:23-24`

`compute_bone_matrices` propagates the hierarchy in a single index-order
pass, and `apply_body_lean` marks spine descendants with the same
single-pass scan — both correct only if every joint's `parent_index` is less
than its own index. The loader copies `skin.joints` order verbatim with no
validation or topological reorder, and the glTF spec does not require
parents-first ordering.

**Impact:** a spec-valid GLB listing a child before its parent silently
animates wrong (child reads the parent's previous-frame transform; body lean
misses part of the spine subtree) — no assert, no log. Most exporters emit
parents-first, so uncommon but realistic.
**Fix:** document the precondition in animation_player.hpp/ik_solver.hpp and
docs/skinned_mesh_format.md; validate `parent_index < i` at load with a
clear error, or topologically reorder joints in the loader.

### M15. `apply_foot_ik` pelvis-drop incremental bone-matrix update is wrong for non-translation-only inverse-bind matrices

**Category:** latent-bug
**Locations:** `src/bracken/animation/ik_solver.cpp:167-181` (incremental update at `:174-177`)

After lowering every joint's world transform, the code adds
`drop_model * column 1 of inverse_bind` to the bone matrix's translation
column. The exact affine delta for `bone = world * inverse_bind` after a
world-Y bump is simply `bone[3].y += drop_model` for ANY affine IBM — the
code's delta matches only when IBM column 1 is (0,1,0,0), i.e.
translation-only bind poses (numerically verified; errors up to ~0.95 for
rotated IBMs). All in-tree rigs have translation-only IBMs, so tests pass;
the loader ingests IBMs verbatim from glTF, so rotated IBMs from real DCC
exports are fully reachable.

**Impact:** rigs with rotation/scale in their IBMs plus foot IK over
descending terrain get visibly wrong skinning, breaking the module invariant
`bone = world * inverse_bind` that the leg IK solve and tests rely on.
**Fix:** replace the four-component update with `bone_matrices[i][3].y +=
drop_model;` (the exact delta); add a rotated-IBM test asserting the
invariant after a pelvis drop.

### L6. `FootIKBoneNames` comment claims defaults "match Mixamo conventions" but they don't match raw Mixamo rigs

**Category:** doc-code-mismatch
**Locations:** `src/bracken/animation/animation_types.hpp:75-86`; `docs/skinned_mesh_format.md:47-52`

The defaults are unprefixed Hips/Spine/LeftUpperLeg/LeftLowerLeg/LeftFoot;
raw Mixamo exports use `mixamorig:`-prefixed names, and even prefix-stripped
Mixamo bones are LeftUpLeg/LeftLeg. Matching is exact-string with no
normalization; docs/skinned_mesh_format.md (correctly) says raw Mixamo rigs
must supply an override — the header contradicts the doc.

**Impact:** a consumer trusting the header loads a Mixamo character, gets
`FootIKData::valid == false`, and foot IK/body lean silently no-op — with
the comment pointing away from the cause.
**Fix:** correct the header comment to state the actual default names and
that Mixamo rigs require an explicit `FootIKBoneNames` override.

### L77. `AnimationKeyframe` struct and `MAX_BONE_INFLUENCES` constant are declared but unreferenced anywhere

**Category:** dead-code
**Locations:** `src/bracken/animation/animation_types.hpp:16`, `:18-23`

Zero references in the tree: channels store per-path SoA arrays, and the
real 4-influence vertex layout is hardcoded independently in
`gpu::SkinnedVertex` and the loader's decode loop — changing
`MAX_BONE_INFLUENCES` would silently do nothing. architecture.md's constants
list still presents it as authoritative.

**Impact:** vestigial API misleads readers about the authoritative keyframe
representation / influence-count knob.
**Fix:** delete `AnimationKeyframe`; either delete `MAX_BONE_INFLUENCES` or
make it authoritative (size `SkinnedVertex` arrays from it +
static_assert in the loader).

### L78. `apply_foot_ik` accepts a `model_to_world` matrix parameter the implementation ignores

**Category:** dead-code
**Locations:** `src/bracken/animation/ik_solver.hpp:16-18`; `src/bracken/animation/ik_solver.cpp:145-147`; callers at `examples/playground/playground.cpp:458-460`, `src/bracken/animation/ik.bench.cpp:163` (+7 test sites)

The definition comments the parameter name out and never uses it —
world-to-model conversion of foot offsets is done solely via the separate
`scale` float. Callers still construct and pass a matrix.

**Impact:** misleading API — consumers reasonably assume the instance
transform (rotation/scale) participates in the correction; call-site noise.
**Fix:** remove the parameter from declaration/definition and update the
callers (and the two doc mentions), or use it to derive the scale factor and
drop the separate argument.

---

## Findings — Terrain & grass

### M1. GrassRenderer hardcodes a game-specific "town" clearing and world-edge margin into chunk culling

**Category:** design-violation
**Locations:** `src/bracken/render/grass_renderer.cpp:173-177` (town box), `:168-171` (edge margin)

`upload_chunks` unconditionally rejects any grass chunk whose center lies
within a ±200-unit box around the world center — locals literally named
`town_cx`/`town_cz`, a leftover from the source game — and separately
rejects chunks within 50 units of the world edges. Neither exclusion is
configurable or exposed through any API.

**Impact:** every consumer gets a permanent 400×400-unit bald patch centered
in their world and a bare border ring, silently, regardless of splatmap or
game content. Directly contradicts CLAUDE.md's "engine code is
game-agnostic" rule.
**Fix:** delete the town box; replace both exclusions with host-configurable
placement controls (optional exclusion AABBs, edge margin defaulting to 0).
**Cross-ref:** recorded as a limitation in technical_spec.md:3899 /
user_guide.md:3750; not in RESOLUTIONS.md.

### M21. Terrain and grass splatmap UVs assume a square world anchored at origin (0,0)

**Category:** latent-bug
**Locations:** `src/bracken/render/terrain_renderer.cpp:495`; `src/bracken/shaders/terrain.frag.hlsl:139`; `src/bracken/render/grass_renderer.cpp:260`; `src/bracken/shaders/grass.vert.hlsl:175`; `src/bracken/render/grass_renderer.hpp:41`

`world_size = world_width_` ("Assumes square terrain") with
`splatmapUV = worldPos.xz / worldSize`; grass mirrors it with
`splat_world_size` and origin fixed at (0,0). Yet both `init()` signatures
accept independent width/height with no check, log, or assert — and grass
*heightmap* sampling does honor `HeightmapParams` origin/size, so only splat
sampling carries the assumption.

**Impact:** a non-square world (which the API accepts) gets terrain
material layers and grass splat masking stretched/mis-mapped along Z; a
non-origin world mis-samples entirely (grass path). All silent.
**Fix:** plumb splat origin + separate X/Z sizes (a vec4) through both
uniform blocks and shaders; as a stopgap, warn/assert in `init()` when
`world_width != world_height`.

### M22. GrassRenderer's splatmap pointer dangles after `TerrainRenderer::update_splatmap` recreates the texture

**Category:** latent-bug
**Locations:** `src/bracken/render/terrain_renderer.cpp:86`; `src/bracken/render/grass_renderer.hpp:152-155`; `src/bracken/render/grass_renderer.cpp:300-305`

`update_splatmap` destroys and recreates `splatmap_texture_`; GrassRenderer
caches a raw non-owning `GPUTexture*` set via `set_splatmap` and binds it as
vertex sampler slot 1 every frame. Nothing re-points it — the header comment
documents the re-set contract but nothing enforces it. The asymmetry is
telling: `SceneRenderer::set_heightmap` DOES auto-rewire the grass heightmap
pointer after terrain texture recreation; the identical hazard is automated
for heightmap but not splatmap.

**Impact:** a host that wires grass splat masking and then repaints the
splatmap (the editor flow the comment itself anticipates) binds a destroyed
`SDL_GPUTexture` next frame — use-after-free / GPU validation errors.
**Fix:** have SceneRenderer own the wiring (re-call `set_splatmap` after any
`update_splatmap`), or fetch the current texture from TerrainRenderer each
frame instead of caching a raw pointer.

### L19. Public `grass_view_distance` field is silently overwritten every frame by SceneRenderer

**Category:** api-quirk
**Locations:** `src/bracken/scene/scene_renderer.cpp:584-585`; `src/bracken/render/grass_renderer.hpp:172-175`

`render_frame` assigns the public field every frame grass is active
(`min(gfx.get_draw_distance(), MAX_VIEW_DISTANCE)`); the header comment
mentions the per-frame clamp but not that direct host writes are discarded —
user_guide.md even shows direct assignment as host-facing usage.

**Impact:** the knob only works when driving GrassRenderer standalone;
confusing public API surface.
**Fix:** pass view distance as a parameter to `upload_chunks`/`render`, or
respect a host-set override; at minimum say direct writes are overwritten.

### L20. Grass wind direction is hardcoded to `vec2(1.0, 0.3)` despite full uniform plumbing

**Category:** api-quirk
**Locations:** `src/bracken/render/grass_renderer.cpp:263`; `src/bracken/render/grass_renderer.hpp:169-170`

`render` unconditionally writes `vu.wind_direction = glm::vec2(1.0f, 0.3f)`
(unnormalized, ~1.04× effective magnitude); the uniform and shader support
arbitrary direction, but only `wind_magnitude` is public.

**Impact:** hosts cannot steer wind to match weather/environment state; the
uniform slot ships a constant every frame.
**Fix:** add a public `wind_direction` next to `wind_magnitude` (normalize
before upload), or source both from EnvironmentState.

### L21. Splat-aware grass placement is dormant: `set_splatmap` and `update_splatmap` have zero in-tree callers

**Category:** dead-code
**Locations:** `src/bracken/render/grass_renderer.hpp:155`; `src/bracken/render/terrain_renderer.cpp:80-93`

`splat_enabled` is always 0 in-tree, so the fully implemented shader masking
path (grass.vert.hlsl:172-180) never runs; the terrain splatmap only ever
holds its init-time PNG or the 4×4 all-grass fallback. SceneRenderer has
access to both objects but does not connect them.

**Impact:** grass grows uniformly over dirt/rock/sand regions unless the
host discovers and wires the API manually; the splat path is never exercised
by the smoke test, so regressions go unnoticed.
**Fix:** wire SceneRenderer to pass `terrain_->splatmap_texture()` to the
grass renderer at heightmap setup and after any splatmap update (fixing M22
at the same time), or clearly mark both APIs as host-opt-in.

### L22. `GrassRenderer::update` takes `delta_time` but explicitly ignores it

**Category:** api-quirk
**Locations:** `src/bracken/render/grass_renderer.cpp:72-75`; `src/bracken/render/grass_renderer.hpp:133`

`(void)delta_time;` — only `current_time_` is stored; all wind animation is
absolute-time driven.

**Impact:** misleading signature (dt=0 does not pause direct-driven wind);
vestigial parameter carried through the SceneRenderer call chain.
**Fix:** drop the parameter (e.g. `set_time(current_time)`) or document that
animation is absolute-time driven.

### L69. `set_anisotropic_filter` doc/comment/log say "grass sampler" but the code recreates the terrain material sampler

**Category:** doc-code-mismatch
**Locations:** `src/bracken/render/terrain_renderer.hpp:179`; `src/bracken/render/terrain_renderer.cpp:610`, `:613`

Header doc, inline comment, and failure log all name a "grass sampler"; the
code recreates `material_sampler_` — the sampler for the 4-layer terrain
material Texture2DArray. Copy-paste leftover from a pre-material-array era;
behavior itself is correct.

**Impact:** a sampler-creation failure logs a message pointing at the grass
renderer, misdirecting debugging.
**Fix:** s/grass sampler/material sampler/ in all three places.

---
## Findings — Lighting & shadows

### L47. `ClusterGrid::upload` command-buffer overload keeps a dead `frame_arena` parameter whose "call-site compatibility" rationale is stale — no caller exists

**Category:** dead-code · **Reported by two areas:** memory and lighting-shadows (merged)
**Locations:** `src/bracken/render/lighting/light_cluster.hpp:59-61` (comment + default), `:5` (arena include); `src/bracken/render/lighting/light_cluster.cpp:403` (parameter commented out); sole upload call site `src/bracken/scene/scene_renderer.cpp:628`

`upload(SDL_GPUCommandBuffer*, memory::Arena* frame_arena = nullptr)`
ignores `frame_arena` entirely (staging lives in the member scratch
`light_staging_`); the header says the parameter is "kept for call-site
compatibility", but the only in-tree call site uses the separate
`upload(SDL_GPUCopyPass*)` overload — the command-buffer overload has zero
callers. The dead parameter is also the sole reason light_cluster.hpp drags
`bracken/memory/arena.hpp` into every consumer.

**Impact:** misleads readers into believing cluster upload consumes a
per-frame arena (the memory docs had to document it as a gotcha); unused
overload+parameter in public API; unnecessary include.
**Fix:** delete the `frame_arena` parameter (or the whole zero-caller
overload) and the arena include; update the two docs that describe it.

### L48. `gpu::ClusterParamsUniforms` is a completely unreferenced mirror of `render::lighting::ClusterParams` with no layout check

**Category:** dead-code
**Locations:** `src/bracken/gpu/gpu_uniforms.hpp:425`, `:430`; live struct `src/bracken/render/lighting/light_cluster.hpp:19`; live push path `src/bracken/scene/scene_renderer.hpp:350`

Nothing references the mirror: SceneRenderer stores and pushes
`render::lighting::ClusterParams` directly. Unlike the other mirrored
uniform pairs that got layout tests in the audit campaign, no static_assert
ties the mirror's size/offsets to the real struct — it can drift with no
compile-time signal, while the docs' "keep both in sync" instruction invites
editing the dead one.

**Impact:** a maintainer edits the dead mirror believing it is the live GPU
contract; the edit silently has no effect.
**Fix:** delete `ClusterParamsUniforms`, or fulfill its stated purpose: use
it at the push site and add sizeof/offsetof static_asserts against
`ClusterParams`.

### L49. GTAOUniforms tunables other than `radiusScale` are compile-time constants unreachable through GraphicsSettings

**Category:** api-quirk
**Locations:** `src/bracken/gpu/gpu_uniforms.hpp:271`, `:280` (defaults); `src/bracken/render/ambient_occlusion.cpp:277`, `:320`; `src/bracken/scene/scene_renderer.cpp:694`; `src/bracken/graphics_settings.hpp:91`

Both AO passes build `gpu::GTAOUniforms uniforms = {}` and assign only
projection/screen/radiusScale/kernel — radius (4.0), bias (0.01),
numDirections (6), numSteps (3), maxDistance (12000) keep their NSDMI
defaults. The only host knob is `GraphicsSettings::ao_radius`; `maxDistance`
is additionally coupled by comment to `BlurUniforms::maxDistance`, which
must match.

**Impact:** AO tuning beyond the radius multiplier requires a header edit
and rebuild; a `maxDistance` edit that misses the BlurUniforms twin silently
breaks the AO blur pass-through at the fade boundary.
**Fix:** document the fields as engine compile-time constants, or plumb the
useful ones through GraphicsSettings, deriving `BlurUniforms::maxDistance`
from the same source.
**Cross-ref:** RESOLUTIONS.md Follow-ups #7 tracks revisiting maxDistance
under reversed-Z, but not the general non-exposure.

### L50. `AmbientOcclusion` owns the whole HDR offscreen/composite/tonemap/FXAA chain, far beyond its name

**Category:** api-quirk
**Locations:** `src/bracken/render/ambient_occlusion.hpp:39`, `:61`, `:90`, `:100`, `:112`

The class owns the full-resolution linear-HDR scene color+depth targets, the
composite pass that performs the frame's tonemapping/exposure/contrast/
saturation and the depth-aware AO+fog upsample, the FXAA resolve with its
lazily created LDR intermediate, and the shared samplers. This is the normal
frame path even when `ao_mode == Off`; bloom and fog fetch their scene
targets from it.

**Impact:** discoverability/maintenance — nobody looks for tonemapping,
FXAA, or HDR target ownership in a file named ambient_occlusion; other
systems couple to "AO" for their render targets.
**Fix:** rename/split (e.g. a PostProcessChain/SceneTargets unit with AO
passes as one client), or at minimum retitle the header comment to state the
real scope.

---

## Findings — Model shading & PBR (shader side)

### L5. shaders/README.md attributes the ACES tonemap to pbr.hlsli; it actually lives in lighting.hlsli

**Category:** doc-code-mismatch
**Locations:** `src/bracken/shaders/README.md:111`; `src/bracken/shaders/lighting.hlsli:34`, `:44`

pbr.hlsli contains no tonemap code; `acesNarkowicz`/`acesFitted` are defined
in lighting.hlsli and consumed by composite.frag.hlsl.

**Impact:** anyone following the README (which CLAUDE.md tells contributors
to trust) to modify tonemapping opens the wrong file.
**Fix:** remove "ACES tonemap" from the pbr.hlsli line; note lighting.hlsli
carries the tonemap curves plus IGN and color-transfer helpers.

### L8. `baseColorFactor` is ignored when a primitive has COLOR_0 vertex colors but no base-color texture

**Category:** bug
**Locations:** `src/bracken/shaders/model.frag.hlsl:124`; `src/bracken/shaders/model_instanced.frag.hlsl:120`; loader baking at `src/bracken/model_loader.cpp:407`

The untextured branch computes `vertexColor * tint`, omitting
baseColorFactor. The common case is masked CPU-side: the loader bakes
baseColorFactor into per-vertex color when COLOR_0 is absent, so untextured
factor-colored assets without COLOR_0 render correctly. The glTF spec
deviation manifests only for COLOR_0 + non-white factor + no texture: the
factor is silently ignored instead of multiplying COLOR_0. Note: fixing the
shader alone would double-apply the factor for no-COLOR_0 meshes — the
loader's vertex-color baking must switch to a white fallback in the same
change.

**Impact:** silent wrong colors for a narrower-than-usual but spec-valid
asset combination.
**Fix:** multiply baseColorFactor into the untextured branch of the model
fragment shaders AND change the loader fallback at
`model_loader.cpp:406-411` to white, in lockstep.

### L10. `SkinnedModelCommand` has no `no_fog` field, so skinned draws can never be exempted from fog

**Category:** api-quirk
**Locations:** `src/bracken/scene/render_scene.hpp:39` (vs. `:30`); `src/bracken/scene/scene_renderer.cpp:1802`, `:1827`

`ModelCommand` carries `bool no_fog` and instanced draws a per-instance
noFog float, but the skinned path uses the global `frame_fog_active_`
unconditionally and queues transparent skinned meshes with a hardcoded
`/*no_fog=*/false`. The transparent-draw machinery already honors a per-draw
flag, so this is pure plumbing.

**Impact:** the fog-exemption API is asymmetric across command types —
animated first-person/UI-adjacent meshes can't opt out without disabling fog
globally.
**Fix:** add `bool no_fog = false` to `SkinnedModelCommand` and the submit
overloads; gate `fog_active` at `:1802` and the queue call at `:1827`,
matching the static path at `:1161`.

### L55. `useSkinning` uniform is written by the CPU every draw but never read by any shader

**Category:** dead-code
**Locations:** `src/bracken/gpu/gpu_uniforms.hpp:22`; `src/bracken/shaders/model.vert.hlsl:37`; `src/bracken/shaders/skinned_model.vert.hlsl:41`; writes at `src/bracken/scene/scene_renderer.cpp:1159`, `:1799`, `:1923`

Declared in both vertex cbuffers, set to 0/1 at three CPU sites, read by no
shader body — skinning is selected entirely by pipeline choice. The field
precedes the live `paletteOffset` at offset 276 of the 288-byte cbuffer, so
a one-sided removal would silently break the byte-for-byte contract.

**Impact:** dead layout padding masquerading as a live switch.
**Fix:** execute RESOLUTIONS.md Follow-up #1 — delete the field from both
shaders, the CPU struct, and the three write sites in one change, updating
the layout static_asserts.
**Cross-ref:** RESOLUTIONS.md shaders.md L-1 (partially fixed) +
Follow-ups #1.

### L56. lighting.hlsli helpers `srgbToLinear`/`viewPosFromDepth`/`worldPosFromDepth` are unused while ssao/gtao/fog keep local duplicates

**Category:** dead-code
**Locations:** `src/bracken/shaders/lighting.hlsli:14`, `:64`, `:71`; local copies at `src/bracken/shaders/ssao.frag.hlsl:45`, `:52`, `src/bracken/shaders/gtao.frag.hlsl:50`, `:58`, `src/bracken/shaders/volumetric_fog.frag.hlsl:59`

The canonical helpers have zero call sites; ssao/gtao include nothing at all
and define their own `viewPosFromDepth` + `interleavedGradientNoise`, and
volumetric_fog defines its own `reconstructWorldPos`. Four independent
copies of the reversed-Z/Y-flip depth-reconstruction math.

**Impact:** a convention fix to one copy silently leaves the others stale;
the unused canonical versions give false confidence of a single source of
truth.
**Fix:** execute RESOLUTIONS.md Follow-up #2 — make ssao/gtao/fog include
lighting.hlsli and delete their local copies, or delete the never-called
canonical helpers.
**Cross-ref:** RESOLUTIONS.md shaders.md L-3 (partially fixed) +
Follow-ups #2.

### L57. scene_renderer.cpp comment claims the flat-normal fallback serves a "branch-free path", but the model shaders branch on `hasNormalMap`

**Category:** doc-code-mismatch
**Locations:** `src/bracken/scene/scene_renderer.cpp:260`; `src/bracken/shaders/model.frag.hlsl:112`; `src/bracken/shaders/model_instanced.frag.hlsl:108`

Both fragment shaders branch explicitly on `if (hasNormalMap == 1)` and take
the geometric normal otherwise; the 1×1 (128,128,255) fallback texture is
only a binding-validity filler, never sampled on the no-normal-map path.

**Impact:** a reader could conclude the fallback's contents are
load-bearing, or that removing the `hasNormalMap` uniform is safe.
**Fix:** reword: the fallback keeps the normal-map slot valid for SDL3 GPU;
the shader branches and the texture contents are inert.

### L58. Model fragment shaders use pbr.hlsli functions only via transitive include through cluster_lighting.hlsli → clusters.hlsli

**Category:** api-quirk
**Locations:** `src/bracken/shaders/model.frag.hlsl:101`, `:144`; `src/bracken/shaders/model_instanced.frag.hlsl:100`; chain at `src/bracken/shaders/cluster_lighting.hlsli:16`, `src/bracken/shaders/clusters.hlsli:8`

`specularAARoughness`/`evaluatePBR`/`ambientPBR` resolve only through the
clustered-lighting include chain; neither model shader includes pbr.hlsli
directly (same pattern in terrain.frag and grass.frag). pbr.hlsli has an
include guard, so a direct include is safe.

**Impact:** removing the cluster include (e.g. a no-cluster shader variant)
breaks the core PBR path with undefined-function errors two headers away
from the real dependency.
**Fix:** add explicit `#include "pbr.hlsli"` to the shaders that call BRDF
functions.

---

## Findings — Atmosphere, sky & post-processing

### L1. `EnvironmentState` lives in `namespace bracken`, but CLAUDE.md/architecture.md/user_guide.md call it `render::EnvironmentState`

**Category:** doc-code-mismatch
**Locations:** `src/bracken/render/environment.hpp:5`, `:16`; `CLAUDE.md:93`; `docs/architecture.md:70`; `docs/user_guide.md:2722`, `:3691`, `:3916`, `:6303`

The struct is declared directly in `namespace bracken` although the file
lives under render/; engine code uses it unqualified (enclosing-namespace
lookup), but `bracken::render::EnvironmentState` — the name six doc sites
use, including copy-pasteable example code — does not exist.

**Impact:** a consumer (or agent following CLAUDE.md) writing the documented
qualified name gets a compile error.
**Fix:** move the struct into `bracken::render` (optionally leaving a
`using` in `bracken` for compatibility), or correct all six doc sites to
`bracken::EnvironmentState`.

### L25. `RenderContext::clear_color()`/`set_clear_color()` are dead API — no pass in the frame path reads them

**Category:** dead-code
**Locations:** `src/bracken/render/render_context.hpp:137`, `:142`, `:159`; actual clears at `src/bracken/scene/scene_renderer.cpp:419`, `:670`, `src/bracken/render/ambient_occlusion.cpp:244`, `src/bracken/render/ui_renderer.cpp:229`

Nothing reads `clear_color_`. The HDR offscreen pass clears with
AmbientOcclusion's own color fed from `env_.fog_color_horizon`; the
direct-to-swapchain fallback and the UI menu pass hardcode their own colors.

**Impact:** `set_clear_color()` is a silent no-op despite the doc comment
"Set clear color for render passes".
**Fix:** delete the member and accessors (background color is legitimately
owned by EnvironmentState), or wire the fallback-pass clear to it and fix
the comment.

### L30. `CompositeParams`/`CompositeUniforms` default bloom strength (0.35) disagrees with GraphicsSettings default (0.25)

**Category:** api-quirk
**Locations:** `src/bracken/render/ambient_occlusion.hpp:72`; `src/bracken/gpu/gpu_uniforms.hpp:309`; `src/bracken/graphics_settings.hpp:66`; overwrite at `src/bracken/scene/scene_renderer.cpp:742`

The normal frame path always overwrites the param from settings, so the 0.35
defaults are live only when a host drives the composite pass directly — and
the user guide's direct-API example default-constructs `CompositeParams`
without setting it. (A layout test pins the uniforms-side 0.35; align it in
the same change.)

**Impact:** direct API users get 40% stronger bloom than the persisted
default; readers learning "the default" from one header pick the wrong
number.
**Fix:** set both defaults to 0.25 to match GraphicsSettings (updating the
pinning test), or comment each pointing at GraphicsSettings as the source of
truth.

### L31. `available_resolutions` doc claims "native display modes (pixel_density == 1.0)" but `query_display_modes` never filters by density

**Category:** doc-code-mismatch
**Locations:** `src/bracken/render/render_context.hpp:125`; `src/bracken/render/render_context.cpp:156-157`; `src/bracken/application.cpp:215`

Every SDL fullscreen mode is kept, deduplicated by (w, h, pixel_density) —
while the inline comment says "Deduplicate by w×h". Since
`Application::available_resolutions` strips density down to {w, h}, the
app-facing list can show duplicate rows on HiDPI displays. Caution when
fixing: exclusive-mode matching (`set_window_mode` mode 2) keys on the
density-inclusive triple, so correcting the *comments* is the safer branch.

**Impact:** documented invariant false; duplicate w×h rows in a
consumer-built resolution picker.
**Fix:** either filter to density == 1.0 (matching the doc) or fix both
comments to state all densities are kept and the dedupe key.

### L32. `WorldRenderer::sky_time()` comment claims it is the shared clock for grass wind/cloud scroll, but nothing reads it

**Category:** doc-code-mismatch
**Locations:** `src/bracken/render/world_renderer.hpp:67`, `:73`; `src/bracken/scene/scene_renderer.hpp:338`; `src/bracken/scene/scene_renderer.cpp:493`, `:586`

The comment declares WorldRenderer "the single owner of sky time"; in
reality no code calls `sky_time()` — grass wind uses SceneRenderer's
separately accumulated `scene_time_`, and `skybox_time_` is consumed only by
`render_skybox`. Two independent dt accumulators that happen to advance in
lockstep today.

**Impact:** pausing/rescaling one clock (e.g. freezing sky time from the dev
UI) would silently desynchronize grass wind from cloud scroll while the
comment promises a shared clock.
**Fix:** feed grass from `world_->sky_time()` (fulfilling the comment) or
rewrite the comment to say sky_time only drives cloud scroll.

### L33. `PipelineType::Skybox` is commented "Skybox cubemap rendering" but the skybox is fully procedural — no cubemap exists

**Category:** doc-code-mismatch
**Locations:** `src/bracken/gpu/pipeline_registry.hpp:28`; `src/bracken/render/world_renderer.cpp:90`; `src/bracken/shaders/skybox.frag.hlsl:1`

The skybox path draws a fullscreen triangle and skybox.frag.hlsl declares no
TextureCube or sampler at all — gradient, sun disc, clouds, and starfield
are computed analytically; the pipeline's fs resources are one uniform
buffer, zero samplers.

**Impact:** misleads readers into looking for cubemap bindings; extenders
may wire sampler slots the shader contract does not have.
**Fix:** change the comment to "Procedural sky (fullscreen triangle, no
textures)".

### L90. WorldRenderer carries four vestigial pieces: uncalled `render_grid`, unused `model_manager_`, never-invoked terrain-height callback, unused anisotropic sampler

**Category:** dead-code
**Locations:** `src/bracken/render/world_renderer.hpp:58`, `:88`, `:99`, `:106`, `:127`; `src/bracken/render/world_renderer.cpp:40`, `:42`, `:83`, `:109`, `:214`; wiring at `src/bracken/scene/scene_renderer.cpp:134`

(1) `render_grid` has no callers, yet `create_grid_mesh` unconditionally
builds and uploads the full-world grid vertex buffer at init; (2)
`model_manager_` is stored and never read; (3) SceneRenderer wires
`set_terrain_height_func`, but the only consumer (`get_terrain_height`) is
never called; (4) init() creates a 16×-anisotropic REPEAT sampler that
nothing binds — and returns false (fatal) if that unused sampler fails.
Leftovers from the pre-extraction rocks/trees path.

**Impact:** wasted GPU memory, a spurious init failure mode, and misleading
structure (readers think WorldRenderer places objects on terrain or loads
models).
**Fix:** delete `model_manager_`, the terrain-height wiring, and the
sampler; either delete `render_grid`+`create_grid_mesh` or keep them as
documented host-facing debug API with the grid mesh built lazily.

---
## Findings — Effects

### H3. CONTINUOUS spawn loop does not validate `spawn_rate`: 0 silently yields one particle, negative values hang the update

**Category:** hang-risk
**Locations:** `src/bracken/systems/effect_system.hpp:174-177` (loop; sole read of `spawn_rate` at `:175`); `src/bracken/effect_definition.hpp:92`

The CONTINUOUS spawn loop computes `spawn_interval = 1.0f / def.spawn_rate`
inside a `while (emitter.age >= emitter.next_spawn_time)` loop, with no
validation anywhere. `spawn_rate == 0` → interval +inf: one particle, then
the emitter silently never spawns again. `spawn_rate < 0` → the loop
condition stays true forever while each iteration push_backs a particle —
a non-terminating loop with unbounded memory growth inside `update()`, on
the first tick after the emitter's delay.

**Impact:** docs/effects_system.md says definitions are conventionally
loaded from host JSON; a typo'd negative rate freezes the main loop and
OOMs; a zero rate yields a confusing single-particle effect with no
diagnostic. (The docs say "must be > 0", but nothing enforces it.)
**Fix:** compute the interval once outside the loop; if `spawn_rate <= 0`
skip continuous spawning (warn once per definition). Add unit tests for 0
and negative rates.

### M26. EffectRenderer's emitter→model cache stores failed lookups permanently, so late-registered models never render for the rest of the run

**Category:** latent-bug
**Locations:** `src/bracken/render/effect_renderer.cpp:119`, `:131` (unconditional emplace), `:108` (only clear, in shutdown)

`resolve_emitter_model` caches `EmitterDefinition* → ModelHandle` and
emplaces the result even when the ModelManager lookup fails
(INVALID_MODEL_HANDLE, one cache-gated WARN). There is no invalidation path,
so if an effect becomes active before its model is registered, registering
it afterwards has no effect — both consumers skip on the cached INVALID
handle. (docs/effects_system.md does instruct hosts to register models
before the first spawn; the permanent, unrecoverable consequence of a slip
is the issue. The engine's built-in hot reload is unaffected —
`reload_model` swaps in place, keeping handles stable; only host-driven
re-registration of the same name leaves the cache serving the stale old
handle.)

**Impact:** a host that spawns an effect one frame early (or registers
models lazily/asynchronously) gets one warning and an invisible effect for
the whole session.
**Fix:** don't cache INVALID_MODEL_HANDLE (retry each call until it
succeeds, with a warned-set if log spam is a concern), or add an explicit
cache-invalidation API called on model registration.

### L35. EffectRenderer class comment claims depth test LESS_OR_EQUAL; the Particle pipeline uses GREATER_OR_EQUAL (reversed-Z)

**Category:** doc-code-mismatch
**Locations:** `src/bracken/render/effect_renderer.hpp:36`; `src/bracken/gpu/pipeline_registry.cpp:600`

The pipeline sets `SDL_GPU_COMPAREOP_GREATER_OR_EQUAL; // reversed-Z`,
consistent with the engine-wide convention; the registry's own comment is
correct and the header comment is stale.

**Impact:** dangerous specifically because reversed-Z compare-op mistakes
are called out in CLAUDE.md as a common way to silently break rendering when
someone "fixes" code to match a stale comment.
**Fix:** edit the class comment to "depth test GREATER_OR_EQUAL
(reversed-Z) against opaques, depth write off".

### L36. `EffectDefinition::duration` and `::loop` are never read by the runtime; docs list them as functional fields

**Category:** dead-code
**Locations:** `src/bracken/effect_definition.hpp:113-114`; completion logic at `src/bracken/systems/effect_system.hpp:71`, `:90`; `docs/effects_system.md:48`

Effect completion is purely emitter-driven (emitter-level duration +
particle death); `EffectInstance::age` is accumulated but never compared.
Setting `loop = true` does nothing; docs/effects_system.md lists both fields
with no inertness note (while it does flag the "sprite" stub).

**Impact:** a consumer tuning effect-level duration or setting loop in a
data file gets a silently emitter-timed, non-looping effect.
**Fix:** implement effect-level duration/loop in `EffectSystem::update`
(restart emitters when age wraps) or delete the fields; at minimum annotate
them as inert in the header and docs.

### L37. Effect `range`/`default_range` is plumbed all the way into `EmitterInstance::range` but never consumed

**Category:** dead-code
**Locations:** `src/bracken/systems/effect_system.hpp:64`; `src/bracken/systems/effect_system.cpp:35`, `:43`; `src/bracken/effect_definition.hpp:117`; `src/bracken/scene/render_scene.cpp:55`; `src/bracken/scene/scene_renderer.cpp:519`

The public API plumbs range through RenderScene → SceneRenderer →
spawn_effect → every EmitterInstance, but no code reads
`EmitterInstance::range` — neither spawning, `update_particle`, nor the
renderer. docs/effects_system.md documents it as live API.

**Impact:** passing range expecting a scaled/ranged effect does nothing,
with no diagnostic.
**Fix:** use range to scale spawn geometry/velocities (its apparent intent)
or remove the whole parameter chain; at minimum document it as
reserved/unused.

### L38. `EmitterDefinition::particle_type` is never read anywhere; "sprite" is a declared-but-unimplemented stub

**Category:** dead-code
**Locations:** `src/bracken/effect_definition.hpp:86`

Only the declaration and a default-value test reference it; every particle
renders as its emitter's mesh model. Already recorded in architecture.md's
known-stubs list, CLAUDE.md's gotchas, and effects_system.md — the residual
cost is carrying an inert string per emitter definition.

**Impact:** `particle_type = "sprite"` is silently identical to "mesh"
(documented).
**Fix:** keep as a documented stub (status quo) or remove the field until
sprite particles exist.
**Cross-ref:** `docs/architecture.md:350`, CLAUDE.md gotchas,
`docs/effects_system.md:50`.

### L39. `Particle::model` is written at spawn but never read; the renderer resolves models per emitter

**Category:** dead-code
**Locations:** `src/bracken/systems/effect_system.hpp:37`; `src/bracken/systems/effect_system.cpp:80`; `src/bracken/render/effect_renderer.cpp:175`

`spawn_particles` sets `particle.model = &def.model` on every particle, but
EffectRenderer resolves the model once per emitter and batches. The only
other reference is a test asserting the default.

**Impact:** 8 dead bytes per particle in the already-flagged 136-byte AoS
hot struct, plus a field falsely suggesting per-particle model variation.
**Fix:** delete the field and the spawn-time assignment (a free partial win
for the deferred SoA restructure); update the test.
**Cross-ref:** RESOLUTIONS.md performance L7 (136-byte AoS Particle,
deferred).

---

## Findings — Text & UI

### M6. Three incompatible packed-color conventions across UI primitives (ARGB), text (ABGR), and debug lines (RGBA)

**Category:** api-quirk · **Reported by two areas:** text-ui and scene (merged)
**Locations:** `src/bracken/render/ui_renderer.cpp:337` (ARGB); `src/bracken/render/text_renderer.cpp:537`, `:426`, `:596` (ABGR); `src/bracken/scene/scene_renderer.cpp:2049` (RGBA debug lines); undocumented parameters at `src/bracken/scene/ui_scene.hpp:114` and `src/bracken/scene/render_scene.hpp:50-57`, `:76`; in-tree collision at `examples/playground/playground.cpp:523`, `:570`

`UIRenderer::color_from_uint32` decodes 0xAARRGGBB for all shapes, buttons,
and 3D billboards; all three text paths decode the same `uint32_t` as
0xAABBGGRR; `upload_debug_lines` decodes 0xRRGGBBAA. All flow through the
identical undocumented `uint32_t color` parameters on UIScene/RenderScene,
and nothing warns or converts. The playground already hits it: it passes
`0xFFFFC040` to a billboard fill (warm orange) and to `ui_.add_text` (light
blue) — the "same" literal renders as two different hues.

**Impact:** the same literal silently renders different colors depending on
which draw call receives it; rects/billboards get R and B swapped relative
to text.
**Fix:** standardize on one packed convention (ARGB, matching
`color_from_uint32`) and convert the text and debug-line decodes in one
change (auditing in-tree literals), or introduce a strong color type; at
minimum document the format on every `uint32_t color` parameter.

### M27. UIRenderer primitive-batch overflow is unhandled: `flush_batch()` is a no-op, oversized frames draw stale GPU buffer contents

**Category:** latent-bug
**Locations:** `src/bracken/render/ui_renderer.cpp:329` (no-op flush), overflow branches at `:348`, `:398`, `:445`, `:507`, draw at `:287`, dead rebinds at `:209`; `src/bracken/gpu/gpu_buffer.cpp:225`; `src/bracken/render/ui_renderer.hpp:134`

Every primitive generator checks `size() + n > MAX_VERTICES` (16384), calls
`flush_batch()` — an explicit no-op ("the actual flush is now done in
execute()") — then appends anyway, growing past the 512 KiB GPU buffer. In
`execute()`, `GPUBuffer::update` rejects the oversized upload with only an
`SDL_Log`, but `SDL_DrawGPUPrimitives` is still issued for the full vertex
count, sourcing the previous frame's stale contents. The pipeline-rebind
blocks inside the overflow branches are vestigial (`current_pass_` is always
null during recording; the wrong-type uniform push is overwritten later).
Unlike the text path, which truncates at its quad budget with a warning,
there is no graceful degradation.

**Impact:** any frame exceeding 16384 UI vertices (~2,730 rects) renders
garbage/stale primitives, with a non-obvious raw SDL_Log line as the only
diagnostic.
**Fix:** truncate the batch at MAX_VERTICES with a once-per-frame
BRACKEN_LOG_WARN (mirroring `text_renderer.cpp:346-353`), or grow the
vertex buffer on demand; delete the dead rebind/uniform blocks either way.

### M28. Legacy `TextRenderer::draw_text` begins a copy pass while the caller's render pass is active (illegal in SDL3 GPU); zero in-tree callers

**Category:** latent-bug
**Locations:** `src/bracken/render/text_renderer.cpp:414` (buffer update inside pass), `:394`, `:445` (draw_text_centered inherits); `src/bracken/gpu/gpu_buffer.cpp:205` (begins copy pass); `src/bracken/render/text_renderer.hpp:166`

The public immediate path takes an active `SDL_GPURenderPass` parameter
("Active render pass to draw into") and calls
`vertex_buffer_->update(cmd, ...)`, which begins a copy pass on the same
command buffer — SDL3 GPU allows only one active pass. It also silently
skips the whole string for a frame when any glyph is newly rasterized.
Nothing in the tree calls these; the engine exclusively uses the batched
queue path (`draw_text_immediate` and `prepare_text_textures` are likewise
uncalled).

**Impact:** a host embedding TextRenderer directly and calling this public,
documented API as its signature instructs gets SDL3 GPU validation
failures/UB on first use — plus mysteriously missing strings on
glyph-rasterizing frames.
**Fix:** delete the legacy immediate paths, or rewrite `draw_text` to draw
only from already-uploaded state and document that uploads must precede the
pass.

### L12. Text measurement (`get_text_width`/`get_text_height`) is unreachable for hosts: SceneRenderer exposes no UIRenderer/TextRenderer accessor

**Category:** api-quirk
**Locations:** `src/bracken/scene/scene_renderer.hpp:210` (private `ui_`, no accessor); `src/bracken/application.hpp:139`; `src/bracken/render/ui_renderer.hpp:104`; `src/bracken/render/text_renderer.cpp:451`; `src/bracken/scene/ui_scene.hpp:126`

Real TTF glyph metrics exist but are effectively engine-internal;
`UIScene::add_text_wrapped` consequently wraps using a hardcoded
8 px-per-char × scale approximation. The engine's own `draw_button` centers
correctly by measuring internally — highlighting the capability gap.

**Impact:** Application-based hosts cannot center, right-align, or wrap text
against real metrics; wrapped text wraps early/late vs. rendered width.
**Fix:** add const measurement passthroughs on SceneRenderer
(`text_width(text, scale)` / `text_height(scale)`, 0 before init), surface
via Application, and optionally use them from `add_text_wrapped`.

### L40. Comment in `TextRenderer::draw_text` claims ABGR is "same as rest of renderer" but UI primitives decode ARGB

**Category:** doc-code-mismatch
**Locations:** `src/bracken/render/text_renderer.cpp:425`; `src/bracken/render/ui_renderer.cpp:338`

The header doc ("Text color in ABGR format") is accurate; only the .cpp
comment's cross-renderer claim is false — the 2D renderer is ARGB for
primitives and RGBA for debug lines (see M6).

**Impact:** a developer trusting the comment "fixes" the ARGB primitive path
(or writes new code) with swapped R/B channels.
**Fix:** change to "(ABGR — note: UI primitives use ARGB, see
UIRenderer::color_from_uint32)", or delete once M6 unifies the conventions.

### L41. ui.frag `has_texture` textured-quad path is dead plumbing: every caller pushes hasTexture=0 with a dummy white texture

**Category:** dead-code
**Locations:** `src/bracken/shaders/ui.frag.hlsl:21`, `:41`; `src/bracken/gpu/gpu_uniforms.hpp:110`; `src/bracken/render/ui_renderer.cpp:270`, `:308`

Both push sites hardcode `hasTexture = 0`; the only bound texture is the
1×1 white dummy required because the sampler slot is declared. There is no
API to bind a real texture or set the flag; UV coordinates are always
(0,0).

**Impact:** the shader branch, uniform field, layout test, and dummy-texture
machinery suggest a textured-UI-quad feature that does not exist.
**Fix:** remove the branch, sampler, and CPU field in lockstep (per the
uniform-pairing rule), or mark it as the intended extension point with an
actual API.

---

## Findings — Camera systems

### L3. `CameraModeConfig` `position_lag`/`rotation_lag` comments invert the parameter's actual meaning

**Category:** doc-code-mismatch
**Locations:** `src/bracken/systems/camera_controller.hpp:15-16`; `src/bracken/systems/camera_system.cpp:63`, `:69`, `:158`, `:175`, `:201`

The comments say "Position smoothing (0-1, lower = more lag)", but both
values are passed verbatim as the smoothTime (seconds) of critically-damped
springs, where LOWER means FASTER — less lag. The 0.001 defaults are
near-instant; the value is a time constant, not a 0-1 blend factor, and this
is the only in-header documentation of these public fields.

**Impact:** a consumer tuning from the header moves the dial exactly
backwards: 0.9 expecting snappy gets a ~0.9 s floaty spring.
**Fix:** rewrite both comments: "spring smooth-time in seconds; lower =
snappier (0.001 ≈ instant, 0.1–0.3 = floaty)", matching the new docs.

### L11. Near/far planes, zoom limits, auto-center delay, ground clearance, and sprint-FOV/speed constants are hard-coded with no public configuration API

**Category:** api-quirk
**Locations:** `src/bracken/systems/camera_system.cpp:366` (near 5 / far 15000 — the engine's only scene projection), `:84`, `:334`; `src/bracken/systems/camera_system.hpp:126`, `:133-134`, `:140`, `:169`

Zoom clamps (80/600), auto-center delay (2 s), ground clearance (25), sprint
FOV bonus (8°), and the 300/400 u/s speed normalizations are private members
or literals with no setter and no `CameraModeConfig` field — all tuned to
the playground's world scale.

**Impact:** a differently-scaled (e.g. meter-scale) game cannot adapt the
engine's only camera without editing engine source: min zoom 80 units,
25-unit hover, near plane 5.
**Fix:** promote these to `CameraModeConfig` fields (or setters) with
current values as defaults; near/far especially, since `compute_matrices`
builds the engine's only scene projection.

### L26. `CameraSystem::set_debug_draw_enabled` sets a flag nothing reads

**Category:** dead-code
**Locations:** `src/bracken/systems/camera_system.hpp:83`, `:195`

No reader anywhere in src/ or examples/ (the playground's `debug_draw_` is
an unrelated app-level flag); no debug visualization code exists in
CameraSystem.

**Impact:** a public method that silently does nothing.
**Fix:** remove it, or implement the intended debug draw (camera arm,
terrain samples, collision ray as debug lines).

### L27. `set_focus_strength` is documented as 0-1 but never clamped

**Category:** api-quirk
**Locations:** `src/bracken/systems/camera_controller.hpp:56` (parameter named `strength_0_1`); `src/bracken/systems/camera_system.hpp:50`; `src/bracken/systems/camera_system.cpp:139`, `:148`

Values ≤ 0 are safely treated as disabled, but values above 1 are used as-is
in `input_yaw_ += diff * soft_lock_strength_ * dt * 2.0f`. The dev-panel
slider happens to stay in range; the engine never self-protects.

**Impact:** a host scaling strength past 1 over-rotates per frame; large
values overshoot and oscillate around the focus target.
**Fix:** `glm::clamp(strength_0_1, 0.0f, 1.0f)` in the setter, or rename the
parameter if super-unity is meant to be allowed.

### L28. `set_yaw`/`set_pitch` do not mark input, so auto-centering is silently inert for absolute-angle hosts

**Category:** api-quirk
**Locations:** `src/bracken/systems/camera_system.cpp:438`, `:444`; auto-center at `:99`, `:125`; `:451`; `examples/playground/playground.cpp:474`

Only the delta APIs set `had_input_this_frame_`; a host driving the camera
exclusively via `set_yaw`/`set_pitch` never resets `time_since_input_`, so
auto-centering activates after 2 s — but its nudge lands after
`update_input_smoothing` consumed `input_yaw_` and is overwritten by the
host's next `set_yaw`, so it is discarded every frame: no jitter renders,
the feature is simply inert. The playground is unaffected (its per-frame
`adjust_zoom` marks input even for zero deltas). Documented as intentional
with workarounds in the new docs; the header carries no note.

**Impact:** absolute-angle hosts get neither working auto-centering nor a
documented reason why.
**Fix:** set `had_input_this_frame_` in `set_yaw`/`set_pitch` when the value
changes, or document that absolute-angle hosts must set
`auto_center_enabled = false`.

### L29. Camera shake is applied after collision avoidance, so large shakes can clip the camera into terrain/walls

**Category:** latent-bug
**Locations:** `src/bracken/systems/camera_system.cpp:41-42` (pass order), `:323` (shake added, no re-check)

`update_collision_avoidance` clamps `final_camera_pos_` (25-unit ground
clearance, ~20-unit obstacle pushback) before `update_camera_shake` adds
`shake_offset_` directly with no re-clamp. Shake displacement can reach
~intensity × 0.5 per axis stacking across up to 8 shakes; playground
intensities (4–6) are far below the safety margins, so latent host-tuning
hazard only.

**Impact:** a large/stacked shake (~40-50+) while pinned against geometry
momentarily pushes the near plane through the ground/obstacle.
**Fix:** shake the ideal position before collision avoidance, or re-run the
ground-clearance/occlusion clamp after adding `shake_offset_`.

### L75. Dead combat/FOV members plus unused static `total_time` and local `yaw_rad` in CameraSystem

**Category:** dead-code
**Locations:** `src/bracken/systems/camera_system.hpp:159-160`, `:166`, `:170`; `src/bracken/systems/camera_system.cpp:21`, `:32`, `:345`

`in_combat_`, `combat_target_`, and `combat_fov_reduction_` are never
referenced in any .cpp (combat concepts also violate the game-agnostic
rule); `base_fov_` is written once and never read; `update()` accumulates a
function-local `static float total_time` ("idle breathing effect") that is
never read — and being function-local static is shared across ALL
CameraSystem instances; `compute_matrices` computes an unused `yaw_rad`.

**Impact:** misleading remnants of pre-extraction game code; the shared
static becomes a cross-instance bug the moment breathing is implemented
naively.
**Fix:** delete all six; if idle breathing is ever implemented, use a
per-instance member.

### L76. `add_shake`'s per-type frequency switch assigns 1.0f in every branch — `CameraShake::frequency` is vestigial

**Category:** dead-code
**Locations:** `src/bracken/systems/camera_system.cpp:474`, `:503`; `src/bracken/systems/camera_system.hpp:20`; use at `src/bracken/systems/camera_system.cpp:289`

The "Set frequency based on type" switch assigns 1.0f in Impact, Heavy,
Subtle, and default branches alike; `elapsed * frequency` is always the
identity. All per-type character comes from hard-coded sine multipliers
inside `update_camera_shake`; there is no API for a custom frequency.

**Impact:** dead differentiation machinery implying per-type oscillation
tuning that does not exist.
**Fix:** give the branches real distinct frequencies (folding in the
per-type multipliers), or delete the field and switch.

---
## Findings — Procedural generation

### M18. Textured tree leaves render as opaque single-sided rectangles in the color pass (no alpha test, no double-sided)

**Category:** bug
**Locations:** `src/bracken/procedural/tree_generator.cpp:429`, `:434`; `src/bracken/scene/scene_renderer.cpp:70`; `src/bracken/shaders/model.frag.hlsl:128`; `src/bracken/shaders/shadow_depth.frag.hlsl:30`

`TreeGenerator::generate` builds the leaf mesh with a default-constructed
`PBRMaterial` (alpha_mode=Opaque, double_sided=false) even when
`leaf_texture_path` is set; `alphaCutoff` reaches the shader only for
`AlphaMode::Mask`, so transparent leaf texels never discard, and the
back-face-culling pipeline variant is used. Meanwhile the shadow pass
independently alpha-tests any textured mesh at a hardcoded 0.5 — so the
canopy proxy casts leaf-shaped shadows while the visible leaves render as
solid quads that vanish when viewed from behind. The playground never passes
`texture_base_path`, so the smoke test never exercises the textured path.

**Impact:** any consumer using a tree preset with textures gets colored
rectangles instead of leaves, disappearing from behind — while shadows look
correct, maximizing confusion.
**Fix:** when the leaf texture loads, set
`alpha_mode = AlphaMode::Mask` (cutoff 0.5) and `double_sided = true` on the
leaf mesh material; or document the host's responsibility on the returned
mesh.

### M19. Failed tree texture load silently yields an all-white tree (vertex colors overwritten before the load attempt)

**Category:** latent-bug
**Locations:** `src/bracken/procedural/tree_generator.cpp:413`, `:434`, `:372`

When a texture path is set, `generate()` first overwrites every vertex color
to `vec4(1.0)`, then calls `load_texture_from_file` and ignores its bool
return (the helper only raw-`SDL_Log`s failures, bypassing the engine
logger). Nothing restores the bark/leaf colors, and the `base_color`
fallback written afterward is never consumed by any render path — shaders
use vertex color when `hasTexture == 0`.

**Impact:** a missing/unreadable texture file (wrong `texture_base_path`,
deployment omission) produces a fully white tree with no engine-logger
diagnostic.
**Fix:** check the return value and only whiten vertex colors after a
successful load (or restore on failure); route the error through
BRACKEN_LOG_WARN.

### M20. `TreeParams::levels` is unvalidated; `levels >= 4` with `children[3] > 0` reads past the 4-entry per-level arrays (UB)

**Category:** latent-bug
**Locations:** `src/bracken/procedural/tree_generator.hpp:20`; `src/bracken/procedural/tree_generator.cpp:129`, `:177`, `:186`, `:216`, `:230`, `:245`, `:251`, `:255`, `:266-267`

Branch level reaches `params.levels` inclusive, and every per-level array
has `MAX_LEVELS = 4` entries; the taper/gnarliness/twist and child-branch
reads index with a value that can equal `levels`. With default
`children = {7,7,5,0}` a `levels = 4` tree silently generates without leaves
(no OOB); with `children[3] > 0` it is an out-of-bounds read — UB. The
header comment ("Branch recursion depth (0 = trunk only)") does not state
the 0..3 valid range and `generate()` enforces nothing. Negative/zero
`sections`/`segments`/`children` are likewise unvalidated (division by zero
→ inf/NaN geometry; negative sections underflow `sections.size()-1`).

**Impact:** plausible params (MAX_LEVELS = 4 is public) yield garbage trees
in practice and hard faults under ASan; presets/defaults unaffected.
**Fix:** clamp `levels` to [0, MAX_LEVELS-1] at the top of `generate()`,
sanity-clamp sections/segments/children, and document the valid range.

### L9. `RockParams::texture_path` and all preset texture parameters are declared but completely ignored

**Category:** api-quirk
**Locations:** `src/bracken/procedural/rock_generator.hpp:28`, `:39`; `src/bracken/procedural/rock_generator.cpp:603`, `:640`

`generate` never reads `texture_path` (every rock gets
`has_texture = false`); all five presets declare a texture parameter left
unnamed and unused in the definitions. Contrast TreeGenerator, whose
identical-looking parameters work.

**Impact:** a consumer passing a rock texture path gets a vertex-colored
rock with no warning.
**Fix:** implement texture loading for rocks (reusing the tree generator's
helper), or delete `texture_path` and the preset parameters and fix the
header comment.

### L70. Procedural generators pack `Mesh::base_color` as ARGB while the glTF loader packs/reads ABGR

**Category:** latent-bug
**Locations:** `src/bracken/procedural/tree_generator.cpp:418`, `:439`, `:482`; `src/bracken/procedural/rock_generator.cpp:604`; `src/bracken/model_loader.cpp:187`, `:407`

The generators pack `0xFF000000 | (r<<16) | (g<<8) | b`; the loader packs
`(a<<24) | (b<<16) | (g<<8) | r` and its only reader decodes r from bits
0-7. Currently inert — no render path reads `Mesh::base_color` for
procedural meshes — but the same field has two incompatible encodings
depending on producer.

**Impact:** any future generic consumer of the field (debug UI, material
export, fallback-tint uniform) red/blue-swaps one producer's meshes.
**Fix:** standardize the generators on the loader's packing (or a shared
pack helper), or drop the generators' base_color writes since nothing
consumes them.

### L71. Rock seed participates in the SDF hash as a float, so distinct seeds ≥ 2^24 collide

**Category:** latent-bug
**Locations:** `src/bracken/procedural/rock_generator.cpp:327`, `:331`, `:349`

`scalar_field` passes `static_cast<float>(params.seed)` (a uint32_t) into
the sphere-blend hash (`sin(j*727.1)` etc.). Above 2^24 a float cannot
represent every integer, so distinct large seeds produce identical rocks;
well below that, `sin()` of large arguments already degrades hash quality.

**Impact:** a host deriving seeds from a wide hash (world-position hashes
across the full uint32 range) gets duplicate rock shapes — silent loss of
variety.
**Fix:** reduce the seed to a small range before the cast (e.g.
`seed % 1024` or hash into [0, 1024)) and document the effective range.

### L72. Canopy-proxy comment and `indices.reserve` assume 6 triangles/18 indices per clump, but 12 triangles/36 indices are emitted

**Category:** doc-code-mismatch
**Locations:** `src/bracken/procedural/tree_generator.cpp:454`, `:487`, `:529`

The emission loop writes each quad with both winding orders for
double-sidedness (deliberate — shadow pipelines cull front), i.e. 36 indices
per clump and ~720 tris for the quoted 60-clump tree, double the comment's
"~360 tris" and the 18-index reserve. Vertex reserve is correct.

**Impact:** the documented shadow-triangle budget is 2× wrong; one extra
vector reallocation.
**Fix:** reserve `* 36` and update the comment to "12 triangles (3
double-sided quads) / ~720 tris per tree".

### L73. Self-cancelling `section_length` adjustment in `generate_branch` is dead code

**Category:** dead-code
**Locations:** `src/bracken/procedural/tree_generator.cpp:110`

The non-evergreen path divides `section_length` by `max(1, levels-1)`,
multiplies it back, then reassigns `section_length = branch.length /
section_count` anyway (comment: "Actually ez-tree divides by (levels-1) for
deciduous. Let's keep it simple:"). The whole if-block is a strict no-op on
the default (deciduous) path.

**Impact:** reads as if deciduous trees use a different section length,
inviting a wrong "fix" (deleting only the final reassignment would change
geometry).
**Fix:** delete lines 110-115, keeping the single computation; keep the
ez-tree divergence note as a plain comment if worth it.

### L74. RockGenerator has no unit tests (only the shared benchmark exercises it)

**Category:** design-violation
**Locations:** `src/bracken/procedural/rock_generator.cpp:393`; `src/bracken/procedural/tree_generator.test.cpp:1` (the module's only test file)

No `*.test.cpp` references RockGenerator; the marching-cubes `generate()`
path is exercised only by procedural.bench.cpp (a separate binary, not in
ctest). Violates the repo convention "tests live next to their unit".

**Impact:** regressions in the SDF/marching-cubes path (determinism,
nullptr-on-empty-mesh, flat base at Y=0, symmetric AABB) pass CI silently.
**Fix:** add `rock_generator.test.cpp` covering: presets return non-null
models, same-seed determinism, different-seed variation, `min_y == 0` and
the forced-symmetric X/Z AABB, low-resolution validity.

---

## Findings — Render graph

### M34. Transient texture creation failure is logged but `compile()` still succeeds, leaving null attachments for `execute()`

**Category:** latent-bug
**Locations:** `src/bracken/render_graph/transient_allocator.cpp:81`; `src/bracken/render_graph/render_graph.cpp:370`, `:392`, `:423`, `:437`

`acquire_texture` returns nullptr when `SDL_CreateGPUTexture` fails;
`allocate_transients` stores it into `ResourceNode::resolved_texture`
unchecked (the function is void, so it cannot propagate), and `compile()`
returns true regardless — only the cycle path fails. `execute()` then builds
target infos with `texture == nullptr` and calls `SDL_BeginGPURenderPass`
with them; the pass callback runs with `ctx.render_pass() == nullptr`
despite declaring managed attachments.

**Impact:** under VRAM exhaustion or a driver-rejected TextureDesc, an SDL
API-contract violation at best, and callbacks that reasonably assume a live
render pass null-deref. (The module is not in the frame path; this is a
public, tested API for consumers.)
**Fix:** make `allocate_transients` return false on any null acquisition and
have `compile()` propagate it (mirroring the cycle path); or skip beginning
a managed pass whose resolved attachment is null and log once.

### L79. engine_design_notes.md credits render_graph with debug-group integration that does not exist

**Category:** doc-code-mismatch
**Locations:** `docs/engine_design_notes.md:75-76`; `src/bracken/render_graph/render_graph.cpp:408`

`execute()` contains no `SDL_PushGPUDebugGroup`/`SDL_PopGPUDebugGroup`, and
nothing in the module references the engine's `gpu::GPUDebugGroup` helper —
managed passes are wrapped only in Begin/EndGPURenderPass.

**Impact:** anyone wiring the graph in expecting per-pass RenderDoc/Nsight
markers (a stated selling point) gets an unannotated capture.
**Fix:** drop "debug-group integration" from the bullet, or add push/pop
around each pass in `execute()` to make the claim true.

### L80. `ResourceNode::version` is write-only; its comment claims cycle detection and versioned reads that are not implemented

**Category:** doc-code-mismatch
**Locations:** `src/bracken/render_graph/render_graph_resource.hpp:77-79`; `src/bracken/render_graph/render_graph.cpp:120` (only write), `:271` (topo_sort, no version use)

The field is only ever incremented in `PassBuilder::write` and never read;
cycle detection works purely off reads/writes lists — a reader depends on
ALL writers of a name, so two passes that each read-modify-write the same
resource produce mutual edges and `compile()` reports a false cycle, a
limitation the comment implies is solved.

**Impact:** misleads maintainers into thinking versioned reads exist.
**Fix:** correct the comment to "unused scaffolding for future versioned
reads" (or delete the field); alternatively implement version-aware edges in
topo_sort.

### L81. Buffer resource support is dead scaffolding — no API can declare or import a buffer, so `get_buffer` always returns null

**Category:** dead-code
**Locations:** `src/bracken/render_graph/render_graph_resource.hpp:13`, `:32`, `:68`, `:75`; `src/bracken/render_graph/render_graph_context.hpp:24`; `src/bracken/render_graph/render_graph.cpp:25`

`ResourceType::Buffer`, `BufferDesc`, the buffer pointer fields, and
`RenderPassContext::get_buffer` all exist, but neither PassBuilder nor
RenderGraph exposes any declare/import-buffer API, and TransientAllocator
pools textures exclusively — `imported_buffer`/`resolved_buffer` are never
assigned non-null.

**Impact:** the API advertises buffer support that silently returns nullptr
with no error.
**Fix:** remove the buffer scaffolding until buffer support is built, or add
a clear "not implemented" comment on `get_buffer` and the fields.

### L82. `PassType` is stored but never consumed by the executor, and the typed `add_pass` overload cannot set it directly

**Category:** api-quirk
**Locations:** `src/bracken/render_graph/render_graph_pass.hpp:14`; `src/bracken/render_graph/render_graph.hpp:37`; `src/bracken/render_graph/render_graph.cpp:154`, `:408`; documented at `docs/user_guide.md:5899`, `docs/technical_spec.md:6986`

`execute()` drives pass management solely from `managed_render_pass` +
attachments; nothing reads `PassNode::type` (even dump_dot omits it), and
the typed `add_pass<Data>` template takes no PassType parameter, defaulting
to Generic unless setup calls `set_pass_type`. The inertness is documented
in the new docs; the header itself carries no note.

**Impact:** declaring `PassType::Compute` gets no compute-pass management
and no behavioral difference — the enum reads like a contract but is inert
metadata.
**Fix:** document PassType as declarative metadata in the header, or make
the executor honor it (begin a compute pass for Compute); add a PassType
parameter to the typed overload for symmetry.

### L83. `dump_dot` claims to emit the compiled graph but dumps the declared graph, and does not escape names in DOT labels

**Category:** doc-code-mismatch
**Locations:** `src/bracken/render_graph/render_graph.hpp:72`; `src/bracken/render_graph/render_graph.cpp:468`, `:472`, `:475`

The implementation iterates raw `passes_`/`resources_` and never consults
`compiled_order_`/`compiled_` — output is identical before and after
`compile()`. Pass/resource names are interpolated into `label="..."`
verbatim; a name containing a double quote produces invalid DOT.

**Impact:** a developer dumping after compile() sees no ordering information
and may assume declaration order is execution order; Graphviz rejects
quoted names.
**Fix:** fix the comment (declared dependency graph, usable pre-compile),
optionally annotate nodes with their compiled step, and escape `"`/`\` in
labels.

### L84. `RenderGraph::init` and `TransientAllocator::init` are `[[nodiscard]] bool` but have no failure path

**Category:** api-quirk
**Locations:** `src/bracken/render_graph/render_graph.hpp:30`; `src/bracken/render_graph/render_graph.cpp:164`; `src/bracken/render_graph/transient_allocator.hpp:27`; `src/bracken/render_graph/transient_allocator.cpp:26`

Both store the device pointer and unconditionally return true. Consistent
with the house bool-init convention as future-proofing, and no in-tree
caller exists (tests deliberately skip init for device-less mode) — the
burden is a reader hunting for a nonexistent failure mode.

**Impact:** cosmetic; headers imply a failure mode that does not exist.
**Fix:** add a brief comment on both declarations noting they currently
cannot fail and the bool exists for forward compatibility (or make them
void until a real failure mode exists).

---

## Findings — Dev UI

### L85. `DevUI::init()` on an already-initialized instance returns true but silently discards the new Bindings

**Category:** api-quirk
**Locations:** `src/bracken/dev/dev_ui.cpp:38`, `:41`; `src/bracken/dev/dev_ui.hpp:62`

`if (impl_) { return true; }` runs before `bindings_ = std::move(bindings)`,
so a repeat call reports success while the fresh Bindings (and
RenderContext) are never stored; conversely a first init that fails partway
HAS already stored `bindings_` with `impl_` null. The header ("Call once,
after the renderer is up") does not state repeat-call semantics; no test
pins double-init. (Application constructs a fresh DevUI each time, so only
host-owned instances hit this.)

**Impact:** a host re-initing to point the overlay at a new
SceneRenderer/CameraController gets `true` back while panels keep
reading/mutating the OLD objects — dangling pointers if those were
destroyed.
**Fix:** update `bindings_` (and log) before the early return, or make the
repeat call return false/assert; document the choice and add a headless
test.

### L86. Graphics tab is shown whenever `bindings.settings` is non-null, even with an empty `apply_settings`

**Category:** latent-bug
**Locations:** `src/bracken/dev/dev_panels.cpp:284`, `:199`; `src/bracken/dev/dev_ui.hpp:53`

Tab visibility checks only `bindings.settings != nullptr`;
`bindings.apply_settings` is consulted solely at push time. The header
promises "Null members hide their panel sections", but an empty
`std::function` does not hide the tab — its widgets mutate the host's
GraphicsSettings in place with no apply call. Only manually constructed
Bindings hit this (Application always binds both).

**Impact:** the UI silently does nothing renderer-side while the mutated
struct now disagrees with actual renderer state — the host's next legitimate
apply unexpectedly flushes stale dev-UI edits.
**Fix:** include `apply_settings` in the visibility condition (matching the
documented contract), or render the tab read-only when the callback is
empty.

### L87. Camera tab shake/FOV-bias/focus slider values are function-local statics never re-read from the controller

**Category:** api-quirk
**Locations:** `src/bracken/dev/dev_panels.cpp:227`, `:248`, `:257`

fov_bias/focus values are pushed to the controller only on slider change and
never re-derived from CameraController state (no getters exist) — the
opposite of the Environment tab's sun sliders, which re-decompose from
`env.sun_dir` every frame precisely to stay in sync with external changes.
The statics are also process-wide.

**Impact:** host code calling `set_fov_bias()`/`set_focus_strength()`
directly (the documented usage) sees a stale panel value, and the next
slider touch stomps the host's setting.
**Fix:** add `get_fov_bias`/`get_focus_strength` to CameraController and
re-derive slider values each frame like the sun sliders; keep statics only
for the fire-and-forget shake test parameters.

### L88. Stats tab FPS history uses function-local statics, assuming a single DevUI instance per process

**Category:** latent-bug
**Locations:** `src/bracken/dev/dev_panels.cpp:46-47`

The 120-sample scrolling history lives in function-local statics with a
comment justifying them only on main-thread/once-per-frame grounds; nothing
enforces single-instance (DevUI is a normal class). Multi-instance is broken
more fundamentally anyway: each `init()` calls `ImGui::CreateContext` and no
method calls `SetCurrentContext`, so a second instance's context would
hijack the first's calls.

**Impact:** latent only (Application wires exactly one), but two visible
instances would interleave writes into shared history — a symptom of the
broader unenforced single-instance assumption.
**Fix:** move the history into per-instance state, or explicitly document
single-instance-per-process in dev_ui.hpp and assert it (static instance
counter in init()).

### L89. Comment claims `render()` drains the ImGui input queue "every frame", but SceneRenderer skips the post-UI callback on null-swapchain frames

**Category:** doc-code-mismatch
**Locations:** `src/bracken/dev/dev_ui.cpp:100`, `:154`; `src/bracken/scene/scene_renderer.cpp:810`

The only caller in the standard wiring is SceneRenderer's post-UI callback,
invoked only when `post_ui_callback_ && current_swapchain_` — on
minimized/acquire-failure frames the drain does not run (DevUI's own
null-target drain branch is unreachable through this wiring). Accumulation
is bounded and consumed by the first NewFrame after recovery.

**Impact:** events queued while minimized replay into the next successfully
rendered frame; the "every frame" comment overstates the guarantee and could
mislead a refactor of the callback wiring.
**Fix:** soften the comment (drained every frame the post-UI callback runs;
skipped on swapchain-acquire failure — bounded), or invoke the callback with
a null swapchain and let DevUI's drain branch handle it.

---

## Refuted during verification

Three candidate findings were adversarially refuted and are recorded here so
the record shows they were considered; they are **not** open findings.

| Candidate | Area | Refutation |
|---|---|---|
| `BRACKEN_PROFILE_ZONE_TAG` expands to two statements when Tracy is enabled | core | The macro shape is real, but the claimed silent misuse cannot exist: Tracy's own `ZoneScoped` is itself two statements, and compiling the un-braced `if` misuse against the vendored Tracy **fails with a compile error** (out-of-scope source-location identifier). Enabling Tracy turns misuse into a build break, the opposite of the claimed silent divergence; the two-statement expansion is also documented in user_guide.md/technical_spec.md and inherited from upstream. |
| `PBRMaterial` engine defaults (metallic 0.0, roughness 0.85) deviate from glTF spec defaults (1.0/1.0) for material-less primitives | models | Recorded deliberate design: technical_spec.md documents the deviation in bold with exact scoping, user_guide.md records the rationale (metallic=1 renders near-black without IBL), and the default is load-bearing — procedural tree/rock generators rely on the default-constructed material. With a material present, tinygltf's spec defaults overwrite the engine defaults, so only material-less primitives are affected, as intended. |
| `ProceduralConfig` is read only by its own unit test — engine and playground never consume it | animation | Recorded deliberate design: user_guide.md documents it as an intentional host-facing convenience struct ("the engine itself never reads it — it exists for your code to consume"), and technical_spec.md lists it under host-side helper structs. A documented host-consumption struct with out-of-tree consumers is public API surface, not dead code. |

---

*Audit performed by static inspection on 2026-07-11. All file:line references
verified against the working tree at time of writing; every finding was
independently re-verified and adversarially checked. No builds were run; no
code outside this report was modified. Findings are OPEN — record fix
outcomes in RESOLUTIONS.md.*
