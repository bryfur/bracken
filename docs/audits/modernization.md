# C++23 Modernization & API-Design Audit

**Date:** 2026-07-03
**Scope:** `src/engine/` (excluding `examples/` and `*.test.cpp`) — ~24,800 lines across 66 headers.
**Method:** read of all public headers plus targeted `.cpp` bodies; grep sweeps for ownership, casts, singletons, and include hygiene; `clang-tidy-22` runs (checks: `modernize-use-nodiscard`, `readability-make-member-function-const`, `cppcoreguidelines-special-member-functions`) against `build/compile_commands.json` on four representative TUs.

Deliberate design decisions from `docs/engine_design_notes.md` (serial-by-default JobSystem, unwired `render_graph/`, stubbed `GPUTimestampPool`, opt-in FileWatcher) are **not** flagged as defects.

---

## Executive summary

The engine is in better modern-C++ shape than most codebases its size: **zero raw owning `new`/`delete`** (the only `new` calls are placement-new in `memory/arena.hpp:33` and the `unique_ptr<T>(new T())` private-ctor factory idiom in `gpu/`), **zero C-style casts** in engine code, one documented `const_cast` (`core/asset/asset_registry.hpp:60`), `std::span` already adopted where it matters most (`animation/animation_player.hpp:41-60`), and correct move semantics on `GPUBuffer`. Ownership is uniformly `unique_ptr` + non-owning raw pointer/reference with documented lifetime rules.

The three findings that matter most are systemic, not local:

1. **The static-analysis program is silently disabled.** `.clang-tidy` enables `modernize-*`, `cppcoreguidelines-special-member-functions`, etc., but `HeaderFilterRegex` still points at the old repo path (`.*/mmo-4/src/.*`), so no header diagnostic has fired since extraction. This single line explains most of the drift below.
2. **`core::Result<T,E>` has zero adoption.** It is built, tested, and used nowhere: the real error-handling convention is 28 `bool init()/load()/reload()`-style functions plus 29 `unique_ptr`-or-`nullptr` factories, all log-and-return, none `[[nodiscard]]`. The engine has two error-handling stories on paper and one in practice.
3. **There are zero `[[nodiscard]]` annotations and near-zero contract asserts** (6 runtime asserts total outside `core/`). Threading and lifetime contracts that the architecture doc calls out (FileWatcher main-thread-only, swapchain-acquire thread affinity, bone-matrix pointer lifetime) are enforced only by comments.

Everything else is incremental polish: a systemic rule-of-five gap (13+ resource-owning classes declare dtor + deleted copy but never mention move), `<SDL3/SDL.h>` umbrella includes in six headers, raw-int mode fields in `GraphicsSettings`, one real out-of-bounds-read bug in `get_draw_distance()`, and a handful of string-keyed hot-path lookups.

Recommended posture: fix the two genuine defects (tidy config, draw-distance clamp), make one **decision** about error handling and enforce it with `[[nodiscard]]`, and land the rest as small mechanical sweeps. No wholesale rewrites are warranted.

---

## High-severity findings

### H1. `.clang-tidy` header filter points at the old repo — all header diagnostics suppressed

**Evidence:** `.clang-tidy:57`

```
HeaderFilterRegex: '.*/mmo-4/src/.*'
```

This repo lives at `~/git/engine`; the regex requires a `/mmo-4/src/` path segment that no longer exists, so clang-tidy reports **nothing** for any project header. Verified by running `clang-tidy-22 -p build` with a corrected filter on four TUs (`model_loader.cpp`, `animation_state_machine.cpp`, `file_watcher.cpp`, `input_handler.cpp`): 79 `modernize-use-nodiscard` and 4 `cppcoreguidelines-special-member-functions` header warnings appear that are currently invisible (21 special-member warnings once `scene_renderer.cpp` is included).

**Why it matters:** the project's `.clang-tidy` is thoughtfully curated (naming rules, `cppcoreguidelines-use-enum-class`, special-member checks), yet none of it applies to the 66 headers where the API lives. This is why an engine with `modernize-use-nodiscard` enabled has zero `[[nodiscard]]`.

**Recommendation:** change to `HeaderFilterRegex: '.*/src/engine/.*'` (keep `ExcludeHeaderFilterRegex: '.*/_deps/.*'`). Expect a one-time wave of warnings; burn it down with the sweeps in M1/M3 below.

### H2. `GraphicsSettings::get_draw_distance()` — out-of-bounds read from an unclamped file-loaded index

**Evidence:** `src/engine/graphics_settings.hpp:64-67`

```cpp
float get_draw_distance() const {
    constexpr float distances[] = {500.0f, 1000.0f, 2000.0f, 4000.0f, 8000.0f, 16000.0f, 32000.0f};
    return distances[draw_distance];
}
```

`draw_distance` is read straight from the settings file with no range check (`read_int("draw_distance", draw_distance)` at `graphics_settings.hpp:165`; `read_int` at `:139-147` catches parse exceptions but never clamps). `Application::init_renderer` auto-loads this file from the platform pref path on every boot. A hand-edited or corrupted file with `draw_distance=9` is UB (OOB stack read) feeding `frame_draw_dist_sq_`, cascade planning, and the cluster grid (`scene/scene_renderer.cpp:480,554,587`).

The codebase already knows the right discipline — `shadow_resolution` is clamped at use (`scene_renderer.cpp:160`: `resolution_table[std::clamp(gfx.shadow_resolution, 0, 3)]`) and `ShadowMap::set_active_cascades` clamps internally (`render/shadow_map.cpp:104`) — `draw_distance` just missed it.

**Recommendation:** clamp in the accessor (`distances[std::clamp(draw_distance, 0, 6)]`), and add a `clamp_to_valid_ranges()` pass at the end of `load()` so every enum-like int (`vsync_mode`, `tonemap_mode`, `ao_mode`, `aa_mode`, `window_mode`, `anisotropic_filter`, …) is normalized once at the trust boundary instead of at each use site.

### H3. `core::Result<T,E>` exists, is tested, and is used by nothing — the engine has two error-handling conventions on paper, one in practice

**Evidence:** grep for `Result<` / `Ok(` / `make_err` across `src/engine/` matches only `core/result.hpp` and its test. The actual convention:

- **28** `bool`-returning fallible functions in public headers (`init`/`load`/`save`/`reload`/`compile` family), e.g. `scene/scene_renderer.hpp:78`, `render/render_context.hpp:33`, `gpu/gpu_device.hpp:49`, `gpu/pipeline_registry.hpp:161`, `model_loader.hpp:201`, `graphics_settings.hpp:69,113`, `application.hpp:46,82`.
- **29** static factories returning `std::unique_ptr<T>` (null on failure) with the reason only in the log: `gpu/gpu_buffer.hpp:54,67`, `gpu/gpu_texture.hpp:47-143`, `gpu/gpu_shader.hpp:67-74,204`, `gpu/gpu_pipeline.hpp:126`, `procedural/tree_generator.hpp:68-77`, `procedural/rock_generator.hpp:36-43`.
- The `ENGINE_LOG_ERROR` + `return false/nullptr` pairing appears throughout the `.cpp` bodies.

Additionally `Result` itself is weaker than it should be: no `[[nodiscard]]` on the class (`core/result.hpp:23`), and no `Result<void>` specialization — which is exactly what the 28 `bool` functions would need, so even a motivated caller can't migrate `init()` today.

**Why it matters:** callers cannot tell *why* an init/load failed without scraping logs; ignored failures compile silently; and every new API forces a convention decision.

**Recommendation:** decide, then enforce. Pragmatic split that fits this engine:
1. Keep `bool` for boot-time `init()` chains (failure is fatal, the log is the diagnostic) but make them `[[nodiscard]]`.
2. Adopt `Result` where the error payload is actionable data, starting at the asset/IO boundary: `ModelLoader::load_glb` (`model_loader.hpp:170`), `GraphicsSettings::load/save`, and the `GPUShader`/`ShaderProgram` factories (compile errors are exactly the case where the message belongs in the return channel — hot-reload wants to display it, not crash).
3. Add `[[nodiscard]]` + a `Result<void, E>` specialization to `core/result.hpp` regardless. If instead the team decides `Result` will never be adopted, delete it — a tested-but-unused vocabulary type is a standing invitation to fork conventions. (Given C++23, also consider making `Result` a thin alias layer over `std::expected` — the current `map`/`and_then` mirror `std::expected::transform`/`and_then` almost exactly.)

---

## Medium-severity findings

### M1. Zero `[[nodiscard]]` in the engine; ~80 high-value sites flagged in a 4-TU sample

**Evidence:** `grep -rn '\[\[nodiscard\]\]' src/engine` → 0 hits. `clang-tidy-22` with `modernize-use-nodiscard` on four TUs → 79 warnings, including exactly the accessors the API depends on: `gpu/gpu_buffer.hpp:98-113` (`handle`/`size`/`type`/`is_dynamic`), `gpu/gpu_device.hpp:59,164,173-174`, `core/asset/file_watcher.hpp:53`, `animation/animation_types.hpp:171-172`.

**Why it matters (prioritized, not dogma):** the *high-value* subset is small and behavioral: (a) fallible `bool` returns (`init`, `reload_model`, `save`, `compile`) where discarding is a real bug; (b) factory functions returning `unique_ptr` where discarding leaks nothing but silently drops a resource the caller believes exists; (c) `core::Handle::is_valid` / `HandlePool::release` (`core/handle.hpp:18,48`) where an ignored `false` is a use-after-free precursor. Pure getters (`size()`, `type()`) are nice-to-have noise by comparison.

**Recommendation:** annotate at the *type* level first — `class [[nodiscard]] Result`, `struct [[nodiscard]] Err` — then a one-pass sweep of the 28 `bool` fallible functions and 29 factories. Leave plain getters to an eventual `clang-tidy --fix` once H1 is fixed.

### M2. Threading and lifetime contracts are documented but nothing asserts them

**Evidence:**
- `core/asset/file_watcher.hpp:31-33`: "init, shutdown, watch_file, watch_directory, and unwatch are callable from the main thread only. poll_main_thread() must run on the main thread." No thread-id check anywhere (`grep this_thread|get_id` across `src/engine` finds only a `yield()` in `job_system.cpp:214`).
- `gpu/gpu_device.hpp:92-94`: `acquire_swapchain_texture` carries an SDL `@warning` about thread affinity — unenforced.
- `scene/render_scene.hpp:37,42`: bone-matrix pointer "must remain valid until the frame is rendered" — never validated (a null check is cheap; staleness is untestable, but null catches the most common host bug).
- Assert usage overall: 6 `ENGINE_ASSERT/VERIFY` in `scene/scene_renderer.cpp` and 6 inside `core/result.hpp`; nothing else engine-wide, despite `core/assert.hpp` providing a good zero-cost-in-release `ENGINE_ASSERT`.

**Why it matters:** these are exactly the bugs that surface as GPU corruption or heisen-crashes when a host grows a loading thread. The infrastructure to catch them at the callsite already exists and costs nothing in release.

**Recommendation:** add an `ENGINE_ASSERT_MAIN_THREAD()` macro (capture `std::this_thread::get_id()` in `Application::init_engine` / `FileWatcher::init`, compare in debug). Apply to the five FileWatcher entry points, `GPUDevice::acquire_swapchain_texture`, and `SceneRenderer::render_frame`. Add `ENGINE_ASSERT(cmd.bone_matrices != nullptr)` in `pack_bone_palettes`.

### M3. Rule-of-five gaps are systemic: 13+ resource-owning classes declare dtor + deleted copy but never mention move

**Evidence:** `cppcoreguidelines-special-member-functions` (5-TU run) flags, among others: `scene/scene_renderer.hpp:64-70` (SceneRenderer: user dtor + deleted copies, move unmentioned), `render/shadow_map.hpp:39`, `render/bloom.hpp:25`, `render/ambient_occlusion.hpp:21`, `render/terrain_renderer.hpp:71`, `render/world_renderer.hpp:31`, `render/ui_renderer.hpp:37`, `render/grass_renderer.hpp:91`, `effects/effect_renderer.hpp:26`, `render/volumetric_fog.hpp:29`, `render/lighting/light_cluster.hpp:28`, `gpu/gpu_shader.hpp:130` (ShaderManager), `core/jobs/job_system.hpp:18`. Inverse case: `animation/animation_player.hpp:14-15` declares move ops only (copy implicitly deleted — intended? unclear). `model_loader.hpp:175` (ModelManager) declares a dtor and nothing else — so ModelManager is *copyable by default* while owning `vector<unique_ptr<Model>>`… saved only because `unique_ptr` members make the implicit copy ill-formed when actually used.

**Why it matters:** today's behavior is safe (user-declared dtor/copy suppresses implicit moves; attempts fail to compile), but the intent is invisible and *fragile against edits* — e.g. someone adding `SceneRenderer(SceneRenderer&&) = default;` for a refactor would move 30+ members including raw `SDL_GPUSampler*` and `RenderContext*` back-pointers with no compiler pushback. The architecture doc states "GPU resource owners … are non-copyable/non-movable" (`docs/architecture.md:251`) — the code should say it too. (Note `GPUBuffer` at `gpu/gpu_buffer.hpp:37-40` *is* deliberately movable, so the doc sentence is also slightly over-broad.)

**Recommendation:** mechanical sweep — every class above gets explicit `X(X&&) = delete; X& operator=(X&&) = delete;` (or a two-line `NonCopyable` convention). For `AnimationPlayer`, add `= delete` copy ops with a comment, since a copied player sharing keyframe cursors was a documented historical hazard. Fixing H1 makes clang-tidy guard this class of gap permanently.

### M4. `SkinnedModelCommand` couples the scene layer to `MAX_BONES` via a raw fixed-array pointer — `std::span` decouples it

**Evidence:** `scene/render_scene.hpp:42`:

```cpp
const std::array<glm::mat4, mmo::engine::animation::MAX_BONES>* bone_matrices = nullptr;
```

and `add_skinned_model` (`render_scene.hpp:110-112`) takes the same fixed-size array type, which is why `render_scene.hpp:3` must include `animation/animation_types.hpp` for the constant, and why `animation/animation_player.hpp:47` must expose `bone_matrix_array()` alongside its otherwise-clean span accessors (`animation_player.hpp:41-60`). Meanwhile the consumer already ignores the fixed size: `SceneRenderer::pack_bone_palettes` packs only `joint_count` matrices per instance (`scene/scene_renderer.hpp:240-249`).

**Why it matters:** CLAUDE.md's own warning — "Don't raise `MAX_BONES` without auditing every `std::array<glm::mat4, MAX_BONES>`" — exists largely *because* the command type bakes the constant in. A `std::span<const glm::mat4>` carries (pointer, actual size), removes the animation-header dependency from the scene layer, lets hosts submit right-sized storage, and gives the packer the real length instead of trusting `joint_count` separately. The lifetime contract is unchanged (span is still non-owning — keep the comment), but the type stops lying about extent.

**Recommendation:** change the command field and `add_skinned_model` parameter to `std::span<const glm::mat4>`; delete `bone_matrix_array()`; keep `MAX_BONES` purely as the `AnimationPlayer` storage cap. This is the single most valuable span adoption left in the codebase. Do it before more hosts are written against the array-pointer signature.

### M5. `<SDL3/SDL.h>` umbrella included by six headers, reaching every consumer TU

**Evidence:** `application.hpp:8`, `gpu/gpu_device.hpp:4`, `input_handler.hpp:6`, `render/render_context.hpp:5`, `render/text_renderer.hpp:5`, `render/ui_renderer.hpp:5`. The propagation is total: `gpu_device.hpp` → `gpu_buffer.hpp:3` → `model_loader.hpp:13` → `render_scene.hpp:4` → `scene_renderer.hpp:13` → every host file; plus `application.hpp` → every subclass.

**Why it matters:** the umbrella pulls the entire SDL surface (audio, haptics, camera, dialogs…) into ~every TU in the engine and in consumers. The needed subset is small and explicit: `gpu_device.hpp` needs `<SDL3/SDL_gpu.h>` + `<SDL3/SDL_video.h>`; `application.hpp` needs only `SDL_Event` for the `on_event` virtual (`application.hpp:74`) → `<SDL3/SDL_events.h>`; `input_handler.hpp` already includes the specific gamepad/joystick headers (`input_handler.hpp:4-5`) and additionally needs `<SDL3/SDL_events.h>` + `<SDL3/SDL_scancode.h>`. Notably `model_loader.hpp:21-22` already demonstrates the house style with `struct SDL_GPUDevice;` forward declarations.

**Recommendation:** replace the six umbrella includes with the specific SDL3 module headers. Zero behavior risk, measurable rebuild-time win, and it stops SDL's macro/type surface leaking into every consumer. (A general IWYU pass is *not* recommended — `misc-include-cleaner` is deliberately disabled in `.clang-tidy:32` and the rest of the tree is reasonably tight.)

### M6. `GraphicsSettings` mode fields are raw ints with comment-encoded enumerations

**Evidence:** `graphics_settings.hpp:30-46` — `anisotropic_filter`, `vsync_mode`, `shadow_mode` ("0=off, 1=PCF, 2=PCSS"), `shadow_cascades`, `shadow_resolution`, `ao_mode` ("0=off, 1=SSAO, 2=GTAO"), `aa_mode`, `window_mode`, `tonemap_mode` (`:46`). Consumption is magic-number comparison: `gfx.ao_mode == 1 ? "SSAO" : "GTAO"` (`scene/scene_renderer.cpp:677-680`), `gfx.shadow_mode > 0` (`:548`), and the mode even crosses the GPU ABI as a float (`gpu/gpu_uniforms.hpp:224`: `float shadowEnabled; // 0=off, 1=PCF, 2=PCSS`). Public setters propagate the pattern: `Application::set_vsync_mode(int)`, `set_window_mode(int, int)` (`application.hpp:95-97`).

**Why it matters:** nothing stops `settings.ao_mode = 7` or transposing `shadow_mode`/`ao_mode` at a callsite; the meaning lives in comments duplicated across three files. `.clang-tidy` enables `cppcoreguidelines-use-enum-class` (`:25`) — this is its target case.

**Recommendation:** incremental, not a rewrite: introduce `enum class ShadowMode : int { Off, PCF, PCSS };` etc. in `graphics_settings.hpp`, keep the struct fields as enums with `static_cast<int>` only at the serialization (`save`/`load`) and GPU-uniform boundaries. Start with `shadow_mode`/`ao_mode`/`tonemap_mode` (most-branched); leave index-like ints (`shadow_resolution`, `draw_distance`) as clamped ints since they're genuine table indices.

### M7. `graphics_settings.hpp` carries ~150 lines of inline file I/O (`<fstream>`, `<unordered_map>`) into every TU

**Evidence:** `graphics_settings.hpp:3-5` includes `<fstream>`/`<unordered_map>`; `save()` (`:69-111`) and `load()` (`:113-214`) are fully inline. The header is included by `application.hpp:3` and `scene/scene_renderer.hpp:7` — i.e., everything.

**Why it matters:** a POD settings struct is exactly what hot include paths want; a key-value parser with exception handling is exactly what they don't. Every engine and consumer TU pays for `<fstream>` and re-instantiates the parsing lambdas.

**Recommendation:** move `save`/`load` definitions to a new `graphics_settings.cpp` (auto-globbed by CMake). Combine with the range-clamping pass from H2 — same function, same change.

### M8. Cascade arrays hardcode `4` while the constant `CSM_MAX_CASCADES` lives one module away

**Evidence:** `scene/scene_renderer.hpp:219-233` — `std::array<std::vector<...>, 4> shadow_instance_batches_;`, `std::array<std::vector<ModelHandle>, 4> shadow_batches_touched_;`, `std::array<uint32_t, 4> shadow_cascade_base_instance_{};`, `std::array<Frustum, 4> shadow_cascade_frustums_{};` vs `render/shadow_map.hpp:16`: `static constexpr int CSM_MAX_CASCADES = 4;`.

**Why it matters:** `docs/architecture.md` lists `CSM_MAX_CASCADES` as load-bearing. Raising it to 6 would compile cleanly and index four-element arrays out of bounds at runtime (cascade loops run to `shadow_map_->active_cascades()`). The literal `4`s aren't discoverable by grep-for-constant.

**Recommendation:** `scene_renderer.hpp` already avoids including `shadow_map.hpp` deliberately (forward-declared at `:32`). Either hoist `CSM_MAX_CASCADES` into a tiny shared header, or duplicate the constant locally with a `static_assert(kMaxCascades == render::CSM_MAX_CASCADES)` in `scene_renderer.cpp` (which includes both). Ten-minute fix, removes a documented foot-gun.

---

## Low-severity findings

### L1. String-keyed lookups on per-frame paths

- `animation/animation_state_machine.hpp:51-54`: `set_float(const std::string&, float)` / `set_bool` — the documented host pattern is calling these **every frame** per animated instance ("the host feeds parameters"). Each call hashes the string; small keys dodge heap via SSO but still pay hash+compare per call, and there's no index-based fast path even though `TransitionCondition::param_index` (`:16`) shows the machine already resolves names to indices internally.
- `model_loader.hpp:189-192`: `ModelManager::get_model(const std::string&)` / `get_handle(const std::string&)` — fine at spawn time (the handle system exists precisely to keep this off hot paths), but the signatures force `std::string` temporaries on callers holding literals/`string_view`s. Same for `Model::find_animation(const std::string&)` (`:156`).

**Recommendation:** expose `int param_index(std::string_view)` + `set_float(int, float)` on the state machine (resolve once at setup); switch `ModelManager`'s map to heterogeneous lookup (`std::unordered_map<std::string, ModelHandle, StringHash, std::equal_to<>>` — a transparent FNV-1a hash belongs next to `StringId` in `core/hash.hpp`) and take `std::string_view` in the lookup-only signatures. Skip anything fancier; measured workloads first, per the design notes.

### L2. Singleton surface: three `instance()` globals, one hardwired dependency

**Evidence:** `core/logger.hpp:41`, `core/jobs/job_system.hpp:20`, `core/asset/file_watcher.hpp:48`. All three are defensible (design notes: default-off/serial). Two wrinkles: `JobSystem` is dual-mode — public default ctor *and* Meyers singleton (`job_system.hpp:20-22`), so instances can exist while `parallel_for` hardwires `JobSystem::instance()` (`core/jobs/parallel_for.hpp:27`), meaning a caller-owned pool can never service `parallel_for`; and `FileWatcher::instance()` coexists with instance-based use in tests, so its `~FileWatcher` on a static at exit ordering is worth keeping in mind if sinks/logging ever run in shutdown.

**Recommendation:** low priority given the job system is intentionally idle. When it's ever put under load: add `parallel_for(JobSystem& js, ...)` overload (default to `instance()`), and document which mode is canonical. Don't build DI scaffolding for infrastructure that's off by design.

### L3. `InputHandler` exposes gameplay-named accessors, contradicting its own contract

**Evidence:** `input_handler.hpp:131-132`: `controller_attack()`, `controller_sprint()` — in a class whose header comment (`:11-13`) and `docs/engine_design_notes.md` both say "raw input only... Game-specific actions belong in the host."

**Recommendation:** rename to the raw controls they read (per `input_handler.cpp` these latch specific pad buttons — e.g. `controller_south_button()` / `controller_right_shoulder()` or similar), or move the mapping to the host layer. Cosmetic, but it's the documented architectural boundary — cheap to hold the line now, expensive after more hosts ship.

### L4. `LogSink` is a copyable polymorphic base

**Evidence:** `core/logger.hpp:26-30` — virtual dtor, no copy/move suppression; flagged by `cppcoreguidelines-special-member-functions`. Derived sinks passed by value would slice.

**Recommendation:** delete copy/move in `LogSink` (protected default ctor optional). Two lines.

### L5. Non-`engine/`-rooted include paths in `model_loader` and several `render/` files

**Evidence:** `model_loader.hpp:13-18` uses `"gpu/gpu_buffer.hpp"`, `"animation/animation_types.hpp"` etc. (same in `model_loader.cpp:10-12`). It is **not** the only deviant: `render/world_renderer.hpp:3-7`, `effects/effect_renderer.hpp:3-6`, and `render/grass_renderer.hpp:3-6` use relative `"../gpu/..."`/`"../model_loader.hpp"` paths, as do `render/text_renderer.cpp:2-5`, `render/world_renderer.cpp:2-3`, `render/ui_renderer.cpp:2-6`, and `effects/effect_renderer.cpp:2-3`. CLAUDE.md specifies `engine/`-rooted include paths (`#include "engine/render/world_renderer.hpp"`); roughly ten files deviate.

**Recommendation:** mechanical path fix across the ~10 deviating files, each next time it is touched (or one small sweep).

### L6. `key_just_pressed_` is a C array

**Evidence:** `input_handler.hpp:208`: `bool key_just_pressed_[SDL_SCANCODE_COUNT] = {};`.

**Recommendation:** `std::array<bool, SDL_SCANCODE_COUNT>` — enables `.fill(false)` in `clear_key_edges()` and bounds-checked access in debug via `.at()` if desired. Trivial.

### L7. `unique_ptr<T>(new T())` factory idiom (observation, no action)

**Evidence:** `gpu/gpu_buffer.cpp:86,168`, `gpu_texture.cpp:138-644`, `gpu_pipeline.cpp:116`, `gpu_shader.cpp:136,268`. This is the standard private-ctor workaround for `make_unique`; it is not a leak risk as used (single expression, no interleaved throws). Fine as-is; a private tag-struct + `std::make_unique` would silence future tidy noise but isn't worth a sweep.

---

## What was checked and found clean

- **No owning raw pointers, no manual `delete`** anywhere in engine code; non-owning raws (`GraphicsSettings* graphics_`, `RenderContext* context_`, `GPUDevice* device_`) are consistently paired with lifetime comments.
- **No C-style casts** (regex sweep over all fundamental-type cast forms: 0 hits); one `const_cast`, documented, in `asset_registry.hpp:60`.
- **`std::span` already used well** in `AnimationPlayer` (`animation_player.hpp:41-60`) and `std::string_view` in the logger/assert plumbing. Remaining span candidates: M4, plus `GPUBuffer::create_static/update` taking `(const void*, size_t)` (`gpu_buffer.hpp:54,80,93`) where `std::span<const std::byte>` would end size-mismatch bugs at zero cost — fold into any future GPU-layer touch.
- **Enum hygiene outside GraphicsSettings is good**: `enum class` with sized underlying types throughout (`AlphaMode : uint8_t` `model_loader.hpp:39`, `WatchKind : uint8_t` `file_watcher.hpp:63`, `LogLevel : uint8_t` `logger.hpp:14`, `GPUBuffer::Type` `gpu_buffer.hpp:27`).
- **Deliberate non-issues** (per design docs, not re-litigated here): serial JobSystem, unwired `render_graph/`, `GPUTimestampPool` stub, tagged-union transition conditions (explicitly chosen over `std::variant`), `FrameAllocator` non-adoption, `ClusterParams` intentional duplication.

---

## Prioritized top-10 actions

| # | Action | Files | Effort |
|---|--------|-------|--------|
| 1 | Fix `HeaderFilterRegex` to `'.*/src/engine/.*'` | `.clang-tidy:57` | 1 line |
| 2 | Clamp `draw_distance` in `get_draw_distance()`; add `clamp_to_valid_ranges()` after `load()` | `graphics_settings.hpp:64,113` | <1 h |
| 3 | `class [[nodiscard]] Result` + `Result<void,E>` specialization; then decide Result-vs-bool policy and write it into CLAUDE.md | `core/result.hpp:23` | ~half day incl. decision |
| 4 | `[[nodiscard]]` sweep over the 28 fallible `bool` functions and 29 `unique_ptr` factories | headers listed in H3 | ~1 h, mechanical |
| 5 | `ENGINE_ASSERT_MAIN_THREAD()` macro; apply to FileWatcher entry points, `acquire_swapchain_texture`, `render_frame`; null-assert `bone_matrices` in `pack_bone_palettes` | `core/assert.hpp`, `file_watcher.*`, `gpu_device.*`, `scene_renderer.cpp` | ~2 h |
| 6 | Rule-of-five sweep: explicit deleted moves on SceneRenderer + 12 renderer/GPU classes; deleted copies on AnimationPlayer, ModelManager | M3 list | ~1 h, mechanical |
| 7 | `SkinnedModelCommand::bone_matrices` → `std::span<const glm::mat4>`; drop `bone_matrix_array()` | `render_scene.hpp:42,110`, `animation_player.hpp:47`, `scene_renderer.cpp` | ~2 h |
| 8 | Replace six `<SDL3/SDL.h>` umbrella includes with specific SDL3 module headers | M5 list | ~1 h |
| 9 | Move `GraphicsSettings::save/load` to a `.cpp`; introduce `enum class` for `shadow_mode`/`ao_mode`/`tonemap_mode` | `graphics_settings.hpp` | ~half day |
| 10 | Kill the literal-`4` cascade arrays via shared constant + `static_assert`; heterogeneous string lookup for `ModelManager`; index-based `set_float` on the state machine | M8, L1 | ~half day |

Items 1-2 are defect fixes; 3-6 are the enforcement layer that keeps the rest true; 7-10 are API-quality improvements safe to land independently and incrementally. After item 1 lands, re-running clang-tidy over the tree (checks already curated in `.clang-tidy`) becomes the regression guard for essentially every finding above.

---

## Verification

Each finding was independently re-checked against the cited files (adversarial pass, 2026-07-03):

| Finding | Verdict |
|---|---|
| H1. `.clang-tidy` HeaderFilterRegex points at old repo path — header diagnostics suppressed | **CONFIRMED** — `.clang-tidy:57` reads `'.*/mmo-4/src/.*'`; repo lives at `~/git/engine`, so no project header matches. |
| H2. `get_draw_distance()` OOB read from unclamped file-loaded int | **CONFIRMED** — unclamped `distances[draw_distance]` (`graphics_settings.hpp:64-67`), `read_int` never clamps (`:139-147,165`); sibling clamps verified at `scene_renderer.cpp:160` and `shadow_map.cpp:104`. |
| H3. `core::Result<T,E>` zero adoption; bool/nullptr convention; no `[[nodiscard]]`/`Result<void>` | **CONFIRMED** — grep for `Result<`/`Ok(`/`make_err` hits only `result.hpp` + its test; `std::variant<T,E>` storage rules out `T=void`; class not `[[nodiscard]]`. |
| M1. Zero `[[nodiscard]]` engine-wide | **CONFIRMED** — grep: 0 hits; `handle.hpp:18` (`is_valid`) and `:48` (`release`) verified. |
| M2. Threading/lifetime contracts documented but never asserted | **CONFIRMED** — `file_watcher.hpp:31-34` main-thread-only doc, `gpu_device.hpp:92-93` `@warning`, no thread-id checks anywhere; exactly 6 `ENGINE_ASSERT` uses outside `core/`, all in `scene_renderer.cpp`. |
| M3. Rule-of-five gap in SceneRenderer + 12 classes | **CONFIRMED** — spot-checked SceneRenderer, ShadowMap, Bloom, JobSystem, ShaderManager, ClusterGrid (dtor + deleted copy, move unmentioned); AnimationPlayer move-only (`animation_player.hpp:14-15`); ModelManager declares ctor+dtor only while owning `unique_ptr` members. |
| M4. `SkinnedModelCommand` fixed-array pointer couples scene layer to `MAX_BONES` | **CONFIRMED** — `render_scene.hpp:42` field, `:110-112` parameter, `:3` include of `animation_types.hpp`, `bone_matrix_array()` at `animation_player.hpp:47`, `pack_bone_palettes` packs only `joint_count` (`scene_renderer.cpp:1396-1430`). |
| M5. Six `<SDL3/SDL.h>` umbrella includes | **CONFIRMED** — exactly the six cited headers; chain `gpu_device.hpp` → `gpu_buffer.hpp:3` → `model_loader.hpp:13` → `render_scene.hpp:4` verified. |
| M6. Raw-int mode fields with comment-encoded enumerations | **CONFIRMED** — `graphics_settings.hpp:30-46`; `ao_mode == 1 ? "SSAO" : "GTAO"` (`scene_renderer.cpp:677-680`), `shadow_mode > 0` (`:548`), `float shadowEnabled` (`gpu_uniforms.hpp:224`); `cppcoreguidelines-use-enum-class` enabled (`.clang-tidy:25`). |
| M7. ~150 lines of inline file I/O in `graphics_settings.hpp` | **CONFIRMED** — `save()` `:69-111`, `load()` `:113-214` fully inline; included by `application.hpp:3` and `scene_renderer.hpp:7`. |
| M8. Literal-`4` cascade arrays vs `CSM_MAX_CASCADES` | **CONFIRMED** — four `std::array<...,4>` members (`scene_renderer.hpp:219-233`), constant at `shadow_map.hpp:16`, ShadowMap forward-declared (`scene_renderer.hpp:32`). |
| L1. String-keyed per-frame lookups (state machine, ModelManager) | **CONFIRMED** — `set_float/set_bool(const std::string&)` (`animation_state_machine.hpp:51-52`), `param_index` (`:16`), `ModelManager` string signatures and `find_animation` verified. |
| L2. JobSystem dual-mode + `parallel_for` hardwires `instance()` | **CONFIRMED** — public default ctor (`job_system.hpp:22`), `instance()` (`:20`, Meyers singleton `job_system.cpp:17`), unconditional `JobSystem::instance()` in `parallel_for.hpp:27`. |
| L3. `InputHandler` gameplay-named accessors | **CONFIRMED** — `controller_attack()/controller_sprint()` (`input_handler.hpp:131-132`) directly contradict the class's own raw-input-only comment (`:11-13`). |
| L4. `LogSink` copyable polymorphic base | **CONFIRMED** — `logger.hpp:26-30`: virtual dtor, no copy/move suppression. |
| L5. `model_loader.hpp` "only file" breaking the include convention | **REFUTED as stated, rewritten** — the observation is real, but exclusivity is false: `render/world_renderer.hpp`, `effect_renderer.hpp`, `grass_renderer.hpp` and four `render/*.cpp` files use relative `"../..."` includes that also break the `engine/`-rooted convention. Finding L5 above now reflects the full set. |
| L7/clean-list note. `GPUBuffer` upload APIs take `(const void*, size_t)` | **CONFIRMED** — `create_static` (`gpu_buffer.hpp:54`) and both `update()` overloads (`:80,:93`) verified. |
