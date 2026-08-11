# Host-extensible rendering — runtime pipelines, host shader roots, host passes

Staging document for the integrator. Four sections: **user guide**, **spec**,
**architecture**, **CLAUDE.md delta**. Fold each into the real doc; this file
is not in `core/docs_coherence.test.cpp`'s `DOCS` list, the real ones are.

The feature is three things a game could not do before and one bug fix:

1. **register a pipeline the engine has never heard of**, graphics or compute,
   and get back an opaque handle that participates in every lifecycle path the
   `PipelineType` enum does;
2. **point the shader loader at its own directories**, with hot-reload watching
   them too;
3. **compile its own HLSL with the engine's pinned shadercross/DXC**
   (`bracken_compile_shaders`);
4. **run its own passes** at named points in the frame — ordered, many per
   slot. That last one replaces `SceneRenderer::set_post_ui_callback`, which was
   a single assignable slot that `Application::init_dev_ui` already owned, so a
   game that used it silently turned the developer overlay off.

The catalogue (`gpu/pipeline_catalog.hpp`) is unchanged. Not one enumerator,
not one row, not one field. That is deliberate and it is tested — see
"Why the golden still passes" below.

---

## 1. User guide — "Extending the renderer from a game"

### 1.1 Compiling your own shaders

Put your HLSL next to your game, using the engine's naming convention
(`{name}.vert.hlsl` → `VSMain`, `.frag.hlsl` → `PSMain`, `.comp.hlsl` →
`CSMain`), then in your `CMakeLists.txt`, **after** `add_subdirectory(<bracken>)`:

```cmake
add_executable(my_game main.cpp)
target_link_libraries(my_game PRIVATE bracken::bracken)
bracken_copy_shaders(my_game)                        # the engine's shaders
bracken_compile_shaders(my_game                      # …and yours
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
```

Full signature:

```cmake
bracken_compile_shaders(<target>
    [DIRECTORY   <dir>]           # GLOB_RECURSE *.hlsl + *.hlsli, CONFIGURE_DEPENDS
    [SOURCES     <a.vert.hlsl> ...]
    [STAGE_DIR   <name>]          # default "shaders_<target>"; MUST NOT be "shaders"
    [INCLUDE_DIRS <dir> ...]      # the engine's .hlsli root is always added
    [WAVE_OPS    <file.hlsl> ...])
```

It creates `<target>_compile_shaders` and `<target>_stage_shaders`, makes
`<target>` depend on both, emits `.spv` into
`${CMAKE_CURRENT_BINARY_DIR}/spv_<target>` and stages them into
`$<TARGET_FILE_DIR:<target>>/<STAGE_DIR>`.

**Include directories: shadercross accepts exactly ONE.** The CLI answers
`ERROR: '-I' can only be used once`, and it does not resolve a quoted
`#include` relative to the source file either — it hands the source to DXC as
an in-memory buffer named `hlsl.hlsl`, so an include outside the single `-I`
root is simply not found. The engine's own shader build never hit this because
it has one root. So:

- the roots are ordered `DIRECTORY`, then `INCLUDE_DIRS…`, then the engine's
  `.hlsli` root, deduplicated;
- the **shadercross** path passes only the first — by default your own
  `DIRECTORY`, which is where your `.hlsli` files are;
- the **`WAVE_OPS`/dxc** path passes all of them, because real DXC accepts
  repeats;
- a `message(STATUS)` at configure names which root shadercross got, whenever
  there is more than one.

If you need the engine's `.hlsli` from a non-wave shader, pass the engine root
as the *only* root (`SOURCES …` + `INCLUDE_DIRS ${BRACKEN_SHADER_INCLUDE_DIR}`,
no `DIRECTORY`) and keep your own includes in that same file.

Three things it will not let you do, each because of a real failure mode:

- **`STAGE_DIR` may not be `"shaders"`** (hard `FATAL_ERROR`).
  `bracken_copy_shaders` does `rm -rf` on that directory next to the exe on
  every build, so your `.spv` would disappear intermittently.
- **the `.spv` never lands in the engine's own output directory.**
  `gpu/uniform_layout.test.cpp`'s `EveryShaderBlockIsAccountedFor` enumerates
  every `.spv` in `BRACKEN_SHADER_OUTPUT_DIR` and fails on any uniform block it
  has no table row for — a host cbuffer there fails the *engine's* test suite.
- **`WAVE_OPS` is an explicit list, not a convention.** The shadercross CLI
  never passes `-fspv-target-env`, so DXC defaults to `vulkan1.0` where every
  `Wave*` intrinsic is a hard compile error. Listing a file routes it through
  the vendored `dxc` at `vulkan1.1` with byte-identical `-fspv-*` flags — and
  raises the Vulkan floor for the pipelines that use it. That is a hardware
  requirement your players will feel, so it should be visible in a diff.

`src/bracken/shaders/README.md`'s **set-0 binding-order** warning applies to
your shaders identically: SDL3 GPU numbers sampled textures, then read-only
storage textures, then read-only storage buffers, contiguously. Declaring them
in another order compiles, reflects and validates cleanly, and then binds the
wrong resource.

### 1.2 Registering a pipeline

```cpp
#include "bracken/bracken.hpp"

class MyGame : public bracken::Application {
    bracken::gpu::HostPipelineHandle tint_{};
    bracken::scene::RenderHookHandle  hook_{};

    bool on_init() override {
        if (!init_renderer()) { return false; }

        auto* pipelines = scene_renderer().pipeline_registry();

        // 1. Where your .spv lives, relative to the exe. Engine root stays first.
        (void)pipelines->add_shader_search_path(std::string(SDL_GetBasePath()) + "shaders_my_game/");

        // 2. The pipeline. Fixed-function defaults are the engine's: back-face
        //    culled, opaque, REVERSED-Z, writing the HDR scene target.
        bracken::gpu::HostGraphicsPipelineDesc desc;
        desc.name            = "my_tint";
        desc.vertex_shader   = "fullscreen.vert.spv";     // resolves in the ENGINE root
        desc.fragment_shader = "my_tint.frag.spv";        // resolves in YOUR root
        desc.vertex_layout   = bracken::gpu::VertexLayout::None;  // fullscreen triangle
        desc.blend_mode      = bracken::gpu::BlendMode::Alpha;
        desc.color           = bracken::gpu::TargetFormat::SceneColor;
        desc.depth_target    = false;
        tint_ = pipelines->register_graphics_pipeline(std::move(desc));
        if (!tint_.is_valid()) { return false; }          // the registry logged why

        // 3. When it runs.
        hook_ = scene_renderer().render_hooks().add(
            bracken::scene::RenderHookSlot::AfterOpaque, "my tint", /*order=*/0,
            [this](const bracken::scene::RenderHookContext& ctx) { draw_tint(ctx); });
        return hook_.is_valid();
    }
};
```

Rules, all of them enforced or logged:

- **main-thread tier.** `add_shader_search_path`, `register_*_pipeline`,
  `unregister_pipeline` and every `RenderHookRegistry` mutator assert the main
  thread and assert *not* sim-stage. Call them from `on_init` after
  `init_renderer()`, or from `on_main_frame` at the frame fence. **Never from
  `on_simulate`** — the sim job overlaps the previous frame's recording.
- **re-fetch by handle every frame.** `get_pipeline(handle)` is a cheap
  read-only lookup, and the `GPUPipeline*` it returns is destroyed by
  `invalidate_all()`, `reload_all_pipelines()` (i.e. any shader hot reload) and
  by a swapchain-format change. Caching the pointer at init is a
  use-after-free the first time somebody saves a `.spv`. The engine's own call
  sites re-fetch by enum for exactly this reason.
- **names must be non-empty and unique** among live host registrations.
  Duplicates are refused with an ERROR, so that a log line names one pipeline.
- **resource counts are reflected by default** (`reflect_resources = true`).
  Set it false and fill `vs`/`fs` only if you have a reason; leaving it true is
  what keeps the engine's declared/reflected mismatch ERROR meaningful instead
  of routine.
- **the engine shader root is always search root 0.** A game shipping its own
  `model.frag.spv` can never shadow `PipelineType::Model`'s.

Compute is the same shape:

```cpp
bracken::gpu::HostComputePipelineDesc blur;
blur.name   = "my_blur";
blur.shader = "my_blur.comp.spv";
auto handle = pipelines->register_compute_pipeline(std::move(blur));
```

with one extra constraint the engine cannot work around: **compute creation
requires SPIR-V reflection to succeed**, because the threadgroup size exists
only in the module's `[numthreads]`.

### 1.3 Running your own pass

```cpp
void draw_tint(const bracken::scene::RenderHookContext& ctx) {
    auto* pipeline = ctx.pipelines->get_pipeline(tint_);   // re-fetch, every frame
    if (pipeline == nullptr || ctx.scene_color == nullptr) { return; }

    SDL_GPUColorTargetInfo target{};
    target.texture     = ctx.scene_color;
    target.load_op     = SDL_GPU_LOADOP_LOAD;
    target.store_op    = SDL_GPU_STOREOP_STORE;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(ctx.cmd, &target, 1, nullptr);
    if (pass == nullptr) { return; }
    pipeline->bind(pass);
    const glm::vec4 tint{0.2f, 0.0f, 0.0f, 0.35f};
    SDL_PushGPUFragmentUniformData(ctx.cmd, 0, &tint, sizeof(tint));
    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);        // MANDATORY — see below
}
```

**Every slot fires with a live command buffer and no pass open, and your
callback must leave it that way.** SDL3 GPU offers no way to ask whether a pass
is open, so a leaked pass corrupts every later pass in the frame and fails
inside the driver, far from your code.

| Slot | Where | `scene_color` | `scene_depth` | `swapchain` | Notes |
|---|---|---|---|---|---|
| `PreForward` | after GTAO/GI/RT, before the forward pass | about to be **cleared** | **read-only** | null | for compute writing a target the forward pass will sample |
| `AfterOpaque` | after the forward pass, before the RT reflection resolve | this frame's HDR | this frame's depth | null | **the default slot** |
| `BeforeTaa` | after water, before the TAA resolve | HDR | depth | null | anything drawn here is temporally stabilized |
| `AfterComposite` | after composite (and FXAA), before UI | **pre-tonemap** HDR | depth | live, LDR | |
| `AfterUi` | after the engine UI pass | HDR | depth | live, LDR | runs **before** `set_post_ui_callback`, so the dev overlay stays on top |

`PreForward`'s depth is read-only for a specific reason: after a depth prepass
the opaque forward draws test `EQUAL` against exactly what the prepass wrote,
so writing depth there makes the forward pass drop geometry.

The first four slots are **HDR-path only**. On the fallback
direct-to-swapchain path they cannot run, and the frame counts them in
`FrameDiagnostics::render_hooks_skipped` (visible in the dev overlay's "Dropped
this frame" section) rather than dropping them silently.

Ordering: lower `order` first, ties broken by registration order. That makes
the hooks in a slot a total order determined by nothing but what the host
asked for — no hash, no pointer values.

### 1.4 What replaced what

`SceneRenderer::set_post_ui_callback` still exists and still works;
`Application::init_dev_ui` uses it. It is **deprecated for host use** and the
header says so. Use `render_hooks()`; `RenderHookSlot::AfterUi` is its
many-per-slot replacement and runs first, so a host overlay ends up *under* the
developer overlay instead of *instead of* it.

---

## 2. Spec

### 2.1 `gpu/host_pipeline.hpp`

| Symbol | What |
|---|---|
| `HostGraphicsPipelineTag` / `HostComputePipelineTag` | handle tag types; distinct so the wrong handle is a compile error |
| `HostPipelineHandle` = `core::Handle<HostGraphicsPipelineTag>` | opaque, generation-checked, default = invalid |
| `HostComputePipelineHandle` | ditto for compute |
| `PipelineCacheKey` | `{Origin origin, uint32_t index}`, `Origin ∈ {Catalog, Host}`; `catalog(PipelineType)`, `catalog(ComputePipelineType)`, `host(uint32_t)`; `std::hash` specialized |
| `HostGraphicsPipelineDesc` | owning-string mirror of `PipelineDesc` + overrides (below) |
| `HostComputePipelineDesc` | `{name, shader, entry_point="CSMain", reflect_resources=true, resources}` |
| `build_host_pipeline_config(desc, formats) -> PipelineConfig` | pure; no device, no shader loading |
| `HostPipelineTable<Tag, Desc>` | the slot table: `add`, `remove`, `find`, `has_live_name`, `slots`, `live_count`, `clear` |

`HostGraphicsPipelineDesc` fields beyond `PipelineDesc`'s:

| Field | Default | Meaning |
|---|---|---|
| `vertex_entry` / `fragment_entry` | `"VSMain"` / `"PSMain"` | per-stage entry point |
| `reflect_resources` | `true` | take the counts from SPIR-V reflection; `vs`/`fs` ignored |
| `color_format_override` | `INVALID` | wins over `color` when set |
| `color2_format_override` | `INVALID` | wins over `color2` |
| `depth_format_override` | `INVALID` | wins over the engine depth format |
| `buffers` / `attributes` | empty | when `buffers` is non-empty it **replaces** whatever `vertex_layout` produced, attributes and all |

Every shared fixed-function field defaults **identically** to `PipelineDesc`'s
(`gpu/host_pipeline.test.cpp` `DefaultsMatchPipelineDescDefaults`), so a
default-constructed host desc is a back-face-culled, opaque, reversed-Z
pipeline writing the HDR scene target.

`build_host_pipeline_config` is implemented by projecting onto a `PipelineDesc`
and calling `build_pipeline_config`, then applying the four overrides. Both
halves therefore cannot disagree about reversed-Z, the `depth_target == false`
force-off, or target-format resolution.

### 2.2 `PipelineRegistry` additions

```cpp
[[nodiscard]] bool add_shader_search_path(std::string dir);
[[nodiscard]] std::span<const std::string> shader_search_paths() const;

[[nodiscard]] HostPipelineHandle        register_graphics_pipeline(HostGraphicsPipelineDesc desc);
[[nodiscard]] HostComputePipelineHandle register_compute_pipeline(HostComputePipelineDesc desc);

[[nodiscard]] GPUPipeline*        get_pipeline(HostPipelineHandle);
[[nodiscard]] GPUComputePipeline* get_compute_pipeline(HostComputePipelineHandle);
[[nodiscard]] const HostGraphicsPipelineDesc* host_pipeline_desc(HostPipelineHandle) const;
[[nodiscard]] const HostComputePipelineDesc*  host_compute_pipeline_desc(HostComputePipelineHandle) const;

void unregister_pipeline(HostPipelineHandle);
void unregister_pipeline(HostComputePipelineHandle);
[[nodiscard]] size_t host_pipeline_count() const;
```

Refusal contract for both `register_*`: an **invalid handle**, an ERROR log,
and `creation_count()` untouched when the registry has no device, the name is
empty or duplicates a live registration, a shader name is empty, a shader
resolves against no search root, or creation fails.

**Keying.** The two caches were `unordered_map<PipelineType, …>` and
`unordered_map<ComputePipelineType, …>`; they are now keyed on
`PipelineCacheKey`. `PipelineCacheKey::catalog(t)` and
`PipelineCacheKey::host(i)` are never equal even when `t == i`, which
`gpu/host_pipeline.test.cpp` pins across the whole ordinal range.

**Lifecycle coverage** — the point of the shared key:

| Path | Host coverage | How |
|---|---|---|
| cache | shared map | one lookup, keyed by origin |
| `preload_all_pipelines` | yes | walks the host tables **after** `preloaded_ = true`, so a host rebuild is never logged as an enumeration gap |
| `invalidate_all` | yes, free | `pipelines_.clear()` drops both origins; the host **descs** survive, which is what lets preload rebuild them |
| `set_swapchain_format` | yes | a **second** loop over host entries with `color == TargetFormat::Swapchain`, deliberately not through `pipeline_targets_swapchain()` |
| hot reload | yes | `reload_all_pipelines` = `invalidate_all` + `preload_all_pipelines` |
| `shutdown` | yes | unwatches every root, clears both caches and both host tables |

A host desc that sets `color_format_override` to the presentable format has
opted **out** of the swapchain rebuild: the registry cannot tell that format
apart from any other one the host chose. Use `TargetFormat::Swapchain`.

### 2.3 Shader search roots

`shader_path_roots_` is a `std::vector<std::string>`; element 0 is the engine
root (`shader_path()` returns it) and is rewritten, never appended, by `init()`.
`add_shader_search_path` normalizes a trailing `/`, refuses empty and duplicate
roots (returning `false` for both), and watches the new root immediately when
hot-reload is already live.

Resolution: `resolve_shader_path(file)` returns the first
`root + file` that exists, and logs an ERROR naming the file and the root count
when none does. **Catalogue rows do not go through it** — they still
concatenate `shader_path()` directly, so engine shader resolution is
byte-identical to before.

`shader_watch_handles_` is kept the same length as `shader_path_roots_`, one
watch handle per root, `FileWatcher::k_invalid_handle` for a root that is not
watched yet. `enable_hot_reload` is now idempotent and per-root: an
already-watched root is skipped (double-watching means two rebuilds per save),
and a root whose watch failed is retried on the next call. If no root takes at
all, `watcher_` is cleared again, preserving the old single-root retry
behaviour.

### 2.4 `gpu_shader.hpp` / `gpu_compute.hpp`

Two additive statics:

```cpp
[[nodiscard]] static bool GPUShader::reflect_spirv_file(const std::string& path, ShaderResources& out);
[[nodiscard]] static bool GPUComputePipeline::reflect_spirv_file(const std::string& path, ComputeResources& out);
```

Read the module and reflect it. Used only by the host registration path, which
feeds the reflected counts back in as the *declared* ones so
`create_from_spirv`'s declared/reflected cross-check is a no-op rather than an
ERROR per load. Costs one extra read of the file, at registration time.
Nothing in `create_from_spirv` changed.

### 2.5 `scene/render_hook.hpp`

| Symbol | What |
|---|---|
| `RenderHookSlot` | `PreForward`, `AfterOpaque`, `BeforeTaa`, `AfterComposite`, `AfterUi` |
| `k_render_hook_slot_count` | `5`. A free constant, not an enumerator, so `switch` stays exhaustive under `-Wswitch` |
| `render_hook_slot_to_string(slot)` | `"Unknown"` out of range |
| `RenderHookContext` | see below |
| `RenderHookFn` | `std::function<void(const RenderHookContext&)>` |
| `RenderHookHandle` | `core::Handle<RenderHookTag>` |
| `RenderHookRegistry` | `add`, `remove`, `clear`, `empty()`, `empty(slot)`, `size`, `entries(slot)`, `is_dispatching`, `DispatchScope` |

`RenderHookContext` fields: `cmd`, `pipelines`, `slot`, `scene_color`,
`scene_depth`, `velocity`, `normal_roughness`, `ao`, `swapchain`,
`scene_color_format`, `depth_format`, `swapchain_format`, `width`, `height`,
`camera`, `view_projection_unjittered`, `prev_view_projection_unjittered`,
`depth_prepass_active`, `scene_time`, `dt`, `frame_index`.
`camera.projection` carries this frame's TAA jitter; use the two unjittered
matrices for any reprojection. `ao` is a 1×1 white dummy when AO is off, not
null. Every pointer is valid **only for the duration of the call**.

`RenderHookRegistry::Entry` is heap-allocated with a stable address for its
whole life, because `entry->name.c_str()` is handed to `BRACKEN_GPU_SCOPE` /
`BRACKEN_GPU_TIMER`, which store the raw pointer and read it later in the
frame. `EntryNameAddressesAreStableAcrossRegistration` is the direct test.

`DispatchScope` is an RAII marker `SceneRenderer` holds around each slot's
dispatch; while it is alive `add`/`remove`/`clear` trip a `BRACKEN_CONTRACT`
(live in release). A hook registering another hook would reallocate the list
being walked — a host mistake, so a named abort rather than undefined
behaviour three passes later.

### 2.6 Constants and limits

There is **no budget** on host pipelines or host hooks. Both are host-lifetime
registrations made at init, not per-frame data, and neither has a fixed-size
GPU-side table behind it — a cap would be a number with nothing to justify it.
The costs a host should know:

- one `unordered_map` entry per registered pipeline, shared with the ~100
  catalogue entries;
- one `empty(slot)` bool test per slot per frame with nothing registered
  (five tests total);
- one `RenderHookContext` construction per non-empty slot per frame.

### 2.7 Why the golden still passes

`gpu/pipeline_catalog.test.cpp`'s `GoldenParsesIntoOnePerTypeBlock` asserts
exactly `PipelineType::Count + ComputePipelineType::Count` blocks. Host
registrations are invisible to it because:

- `pipeline_catalog.hpp` and `.cpp` are **not modified at all** — no enumerator,
  no row, no field, no new `TargetFormat` case. (An earlier design added
  `TargetFormat::Custom`; the format-override fields replaced it precisely so
  that file could stay untouched.)
- the golden regenerator walks `[0, Count)` of the two enums and reads only
  `pipeline_desc(type)` + `k_canonical_formats`. Neither can reach a host
  registration; there is no registration API in `pipeline_catalog.hpp`.
- `pipeline_targets_swapchain()` keeps its enum-only meaning. The registry's
  host rebuild is a separate loop.

Two new tests state that mechanically rather than by argument:

- `PipelineCatalog.GoldenBlockNamesAreExactlyTheCatalogueNames` — the golden is
  a bijection with the two enums, so a named-but-unenumerated pipeline cannot
  be in it;
- `PipelineCatalog.AHostDescMirroringARowProducesTheSameGoldenText` — for all
  ~67 rows, a host desc mirroring the row renders to the same `describe()` text
  as the catalogue row. If the two builders ever diverge the diff is printed in
  golden syntax.

**Regeneration check for the integrator:**

```
BRACKEN_REGEN_PIPELINE_GOLDEN=src/bracken/gpu/pipeline_config_golden.inc \
  ./build/src/bracken/bracken_tests --gtest_filter='PipelineCatalog.*'
git diff --stat src/bracken/gpu/pipeline_config_golden.inc   # must be EMPTY
```

---

## 3. Architecture

### 3.1 Where it sits in the frame

`SceneRenderer::render_frame` gains five guarded call sites, each
`if (!render_hooks_.empty(slot)) { run_render_hooks(...); }`:

```
… prepass → GTAO → GI → ray tracing
   ├── [PreForward]                       cmd live, no pass open
   ├── HDROffscreen (forward pass)
   ├── [AfterOpaque]                      scene colour + depth readable
   ├── RT reflection resolve → water
   ├── [BeforeTaa]
   ├── TAA → auto exposure → bloom → fog
   ├── acquire swapchain → composite → FXAA
   ├── [AfterComposite]                   swapchain live
   ├── UI pass
   ├── [AfterUi]                          swapchain live
   └── post_ui_callback_ (DevUI)          ← unchanged, still last
```

With nothing registered the cost is five bool tests and the frame is otherwise
unchanged — the RT-off smoke baseline is the standing check.

`run_render_hooks` builds one `RenderHookContext` per slot from renderer state
(`post_->offscreen_color()/offscreen_depth()`, `velocity_texture_`,
`normal_rough_texture_`, `frame_ao_texture_`, `frame_cam_`,
`prev_view_proj_unjittered_`, `scene_time_`, `frame_index_`), then walks
`entries(slot)` under a `DispatchScope`, wrapping each entry in its own
`BRACKEN_PROFILE_ZONE_COLOR`, `BRACKEN_GPU_SCOPE` and `BRACKEN_GPU_TIMER`. Host
passes therefore appear in the dev overlay's per-pass CPU record table by name.

### 3.2 Threading tiers

| Operation | Tier | Enforcement |
|---|---|---|
| `add_shader_search_path`, `register_*_pipeline`, `unregister_pipeline` | main-thread | `BRACKEN_ASSERT_MAIN_THREAD` + `BRACKEN_ASSERT_NOT_SIM_STAGE` |
| `RenderHookRegistry::add/remove/clear` | main-thread | same, plus `BRACKEN_CONTRACT(!is_dispatching())` |
| `get_pipeline(HostPipelineHandle)` | worker-safe after creation | read-only: generation check + one map lookup, **never creates** |
| `run_render_hooks` | main-thread | `BRACKEN_ASSERT_MAIN_THREAD` |
| the host callback itself | main-thread | it runs inside `render_frame` |

`get_pipeline(handle)` deliberately does not fall back to creation the way the
enum overload does: the enum overload's fallback is guarded by a main-thread
assert and a WARN that means "the static enumeration is incomplete", neither of
which is meaningful for a registration that is outside the enumeration by
construction.

### 3.3 Lifetimes

- The registry keeps each host **desc** by value. That copy is what makes
  rebuild-after-invalidation possible, and it is why `HostGraphicsPipelineDesc`
  owns `std::string`s where `PipelineDesc` holds non-owning `const char*`.
- Host entries are `std::unique_ptr<Entry>` in a slot vector, never inline
  values, so `desc.name.c_str()` has a stable address for the registration's
  life. Same reasoning for `RenderHookRegistry::Entry`.
- A `GPUPipeline*` from `get_pipeline(handle)` dies on `invalidate_all()`,
  `reload_all_pipelines()` and a swapchain-format change. Re-fetch every frame.
- Releasing a registration bumps the slot's generation, so a retained handle
  validates false rather than aliasing whatever takes the slot next. Both the
  pipeline table and the hook registry do this; both have a test for it.
- `RenderHookContext` pointers are frame-scoped and must not be cached.

### 3.4 Determinism

Host tables and hook lists are walked in **slot order** and
**(order, sequence)** order respectively — never hash order — so preload,
rebuild and dispatch are reproducible. Nothing here is reachable from the
Prepare stage, so the bit-identical-at-any-worker-count rule is not at issue;
the ordering discipline is the same one for the same reason.

### 3.5 Additivity

Nothing here runs unless a host registers something:

- `pipeline_catalog.{hpp,cpp}`: untouched.
- catalogue pipeline creation: unchanged, including its shader-path
  concatenation and its enumeration-gap WARN.
- `render_frame`: five `empty(slot)` bool tests.
- `FrameDiagnostics`: one new counter, zero unless a hook is skipped.
- the playground registers **nothing** and installs **no** hook, so the
  `--smoke 30` baselines must be bit-identical. That has **not been verified**
  — see the report.

---

## 4. CLAUDE.md delta

### Module table

Replace the `gpu/` row's contents with (change in **bold**):

> `gpu/` | `bracken::gpu` | SDL3 GPU abstraction — buffers, textures, shaders, graphics + compute pipelines (`gpu_compute.hpp`), `PipelineRegistry` (cache/lifecycle) over `pipeline_catalog.hpp` (one `constexpr` row per `PipelineType`: shaders, resources, vertex layout, target, depth/blend state), **`host_pipeline.hpp` (the catalogue is closed for the engine and open at runtime for hosts: `register_graphics_pipeline`/`register_compute_pipeline` return an opaque handle, and one `PipelineCacheKey` puts host and catalogue pipelines in the same cache so invalidation / swapchain rebuild / hot reload cover both)**, `gpu_uniforms.hpp` (CPU↔GPU contract)

Add to the `scene/` row, after `SceneRenderer`:

> **`RenderHookRegistry` (`scene/render_hook.hpp`) — ordered, many-per-slot host passes at five named points in the frame, replacing the single `set_post_ui_callback` slot `init_dev_ui` already owned**

### Key patterns

Add a bullet:

> - **Extending the renderer from a game** is three calls and no engine edit:
>   `bracken_compile_shaders(<target> DIRECTORY <dir>)` in CMake compiles the
>   game's HLSL with the engine's pinned shadercross/DXC into
>   `shaders_<target>/` next to the exe (never into the engine's own `.spv`
>   directory — `gpu/uniform_layout.test.cpp` enumerates that one and would
>   fail on a host cbuffer — and never into `shaders/`, which
>   `bracken_copy_shaders` wipes every build);
>   `PipelineRegistry::add_shader_search_path` + `register_graphics_pipeline`
>   turn a `HostGraphicsPipelineDesc` into a handle; `SceneRenderer::render_hooks()`
>   says when it runs. All main-thread tier, all no-ops when unused. The
>   catalogue enums and their golden are untouched by design and
>   `PipelineCatalog.GoldenBlockNamesAreExactlyTheCatalogueNames` is the gate.

### Load-bearing constants

Add:

> - `k_render_hook_slot_count = 5` (`scene/render_hook.hpp`) — host render-pass
>   slots. Not a budget: it is the number of named points in `render_frame`
>   where a host pass can run, and adding one means adding a guarded call site
>   in `SceneRenderer::render_frame`.

### Gotchas

Add two:

> - **A host `GPUPipeline*` is invalidated by any shader hot reload.**
>   `invalidate_all()` / `reload_all_pipelines()` / a swapchain-format change
>   destroy every pipeline, host ones included. Re-fetch by handle every frame,
>   exactly as every engine call site re-fetches by enum. Caching the pointer at
>   init works perfectly until somebody saves a `.spv`.
> - **A render hook that leaves a pass open corrupts the rest of the frame.**
>   SDL3 GPU cannot be asked whether a pass is open, so the failure surfaces
>   inside the driver, in a later engine pass, with nothing pointing at the
>   host callback. Every `RenderHookSlot` fires with no pass open and must
>   return that way.

### Doc lines that need rewording (not done here, to avoid merge conflicts)

- `docs/user_guide.md:1525` — the table row citing
  `PipelineRegistry::shader_path_`. That member is now
  `shader_path_roots_`, a `std::vector<std::string>` whose element 0 is the
  engine root; `shader_path()` returns element 0. The name was chosen so
  `core/docs_coherence.test.cpp`'s identifier grep still resolves, but the row's
  wording should say "roots, engine root first".
- `docs/technical_spec.md:167` and `:1539` — same rename, and `:167` should
  gain "plus any roots a host added with `add_shader_search_path`, searched
  after it".
- `docs/user_guide.md:280`, `:758`, `:7029` — "`init_dev_ui()` claims the
  SceneRenderer's **single** post-UI callback slot; a host that uses the dev UI
  cannot also install its own `set_post_ui_callback`". Still true, but the
  conclusion changed: the host should use `render_hooks()` with
  `RenderHookSlot::AfterUi`, which composes.
- `docs/architecture.md:973` — "There is a single slot" now needs the
  `RenderHookRegistry` sentence next to it, and the pass-order list in the
  "Pass order inside `SceneRenderer::render_frame`" section needs the five slots
  inserted at their positions (see §3.1 above).
