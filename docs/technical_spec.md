# Bracken Engine — Technical Specification

This document specifies every Bracken engine feature in detail — data structures, algorithms, API contracts, constants, and known limitations — as implemented in the source tree. It is the engine's reference of record for *how things work*; it complements `docs/user_guide.md` (how to *use* the engine from a host application) and `docs/architecture.md` (module map and exact frame/pass anatomy). Everything here is sourced from the code; where the code and other documentation disagree, each chapter says so explicitly.

The engine is **Bracken**: a standalone C++23 real-time rendering engine built on SDL3 GPU (Vulkan/Metal/D3D12), namespace `bracken`, static library target `bracken`. Conventions that hold throughout this document: Y-up coordinates (x,z = ground plane), reversed-Z zero-to-one depth for the scene (shadow cascades deliberately stay standard-Z), all lighting math in linear space, and a **pipelined** main loop over an always-on job-system worker pool (Phase 4 of `docs/multithreading_plan.md`: all frame data flows through the immutable `scene::FramePacket` snapshot; the host's `on_simulate(SimFrame&)` runs as a pool job for frame i+1 while the main thread renders frame i; the renderer's Prepare stage additionally fans out internally as `parallel_for` chunks with deterministic chunk-index-ordered merges).

## Load-bearing constants

Consolidated from the chapters below. Changing any of these requires auditing every listed consumer.

| Constant | Value | Defining header | What it caps |
|---|---|---|---|
| `MAX_BONES` | 128 | `animation/animation_types.hpp` (sole definition) | Joints per skinned mesh on the CPU side; sizes the two `std::array<glm::mat4, MAX_BONES>` per `AnimationPlayer`; joints beyond it silently don't animate. No GPU-side counterpart (palettes are packed tightly). |
| `MAX_BONE_PALETTES` | 1024 | `scene/frame_packet.hpp` | Bone palettes per frame, enforced at **build time**: `FrameBuilder::alloc_bone_palette` past the budget logs an error and returns an invalid allocation; `draw_skinned` with an invalid id drops the draw with a warning |
| `CSM_MAX_CASCADES` / `MAX_SHADOW_CASCADES` | 4 | `render/shadow_map.hpp` / `scene/scene_renderer.hpp` (static-asserted equal) | Shadow cascades in the single 2×2-quadrant D32 atlas |
| `CLUSTER_DIM_X/Y/Z` | 16 × 9 × 24 (= `CLUSTER_COUNT` 3456) | `render/lighting/light.hpp` (duplicated as `#define`s in `clusters.hlsli`) | Clustered-lighting grid dimensions |
| `MAX_LIGHTS` | 256 | `render/lighting/light.hpp` | Dynamic point + spot lights per frame; excess REFUSED at `ClusterGrid::add_*` (`[[nodiscard]] bool` returns false), counted in `FrameDiagnostics::lights_dropped`, one `BRACKEN_LOG_WARN` per grid |
| `MAX_LIGHTS_PER_CLUSTER` | 128 | `render/lighting/light.hpp` | Per-cluster light-bin cap (the highest-contribution lights are kept, ranked per froxel; one log per frame) |
| Swapchain / scene / depth formats | `RGBA8_UNORM` / `RGBA16F` / `D32_FLOAT` | `gpu/pipeline_registry.hpp` | Swapchain, HDR scene color, and depth target formats |
| Default world extents | 32000 × 32000 | `application.hpp` (`init_renderer` default arguments) | Terrain/grass world size when the host omits it |
| Near / far planes | `CameraModeConfig::near_plane`/`far_plane` (defaults 5 / 15000), arguments swapped (reversed-Z) | `systems/camera_controller.hpp` (config), `systems/camera_system.cpp` (`compute_matrices`) | The engine's only scene projection; configurable via `CameraModeConfig` |
| Shadow resolution table | {512, 1024, 2048, 4096} | `contracts/graphics_settings.hpp` (`shadow_map_texels`) | Per-cascade quadrant resolution. A `constexpr` mapping off `GraphicsSettings::shadow_resolution`, which is a `ShadowResolution` enum — not an int index |
| Local-shadow atlas table | {1024, 2048, 4096} | `contracts/graphics_settings.hpp` (`local_shadow_atlas_texels`) | 8×8-tile point/spot shadow atlas, off the `LocalShadowResolution` enum |
| `LOCAL_SHADOW_TILES_PER_ROW` (→ `LOCAL_SHADOW_MAX_SLOTS`) | 8 (→ 64 tiles) | `render/lighting/light.hpp` (mirrored in `cluster_lighting.hlsli`) | Local-shadow atlas grid; caps `local_shadow_point_budget` ≤ 8 (6 face tiles each) and `local_shadow_spot_budget` ≤ 16 |
| `LocalShadowConfig::max_tile_updates_per_frame` | 6 | `render/lighting/local_shadows.hpp` | Local-shadow tiles re-rendered per frame; excess degrades to stale tiles (round-robin), never a hitch |
| `EXPOSURE_HISTOGRAM_BINS` | 256 | `render/auto_exposure.hpp` (fixed by the compute shaders' LDS/`[numthreads]`) | Auto-exposure histogram bins; `EXPOSURE_MIN/MAX_LOG2_LUMA` = −10/+16 bounds the metered EV range |
| `PARALLEL_MIN_TOTAL` | 256 | `core/jobs/parallel_for.hpp` | Below this element count, auto-grain `parallel_for` runs serially |
| `Arena::k_default_chunk_size` | 64 KiB | `memory/arena.hpp` | Default growable-arena chunk size |
| `Bloom::MAX_MIPS` | 6 | `render/bloom.hpp` | Bloom mip-chain slots (4–6 active depending on resolution) |
| `INITIAL_VERTEX_CAPACITY` (UI) | 16384 | `render/ui_renderer.hpp` | Initial GPU vertex-buffer capacity; the buffer grows geometrically when a frame outgrows it (see Text & UI Rendering) |
| `MAX_TEXT_QUADS` | 4096 | `render/text_renderer.hpp` | Glyph quads per frame across all strings; excess dropped with one warning per frame |
| `ATLAS_SIZE` | 512 | `render/text_renderer.hpp` | Glyph atlas dimensions (one atlas, one hardcoded 18 px font size) |
| `GrassRenderer::MAX_VIEW_DISTANCE` | 1200 | `render/grass_renderer.hpp` | Grass draw-distance clamp; chunks are 32 world units with 144 blades each |
| `MAX_ACTIVE_SHAKES` | 8 | `systems/camera_system.hpp` | Concurrent camera shakes; the oldest is evicted silently |
| `TreeParams::MAX_LEVELS` | 4 | `procedural/tree_generator.hpp` | Per-level tree parameter arrays; out-of-range `levels` is clamped with a warning (`sanitize_params`) |
| `INVALID_MODEL_HANDLE` | 0 | `model_loader.hpp` | Reserved null model handle (slot 0 permanently reserved) |

## Table of contents

1. [Getting Started: Building & Integrating Bracken](#getting-started-building--integrating-bracken)
2. [Application Lifecycle, Main Loop & Input](#application-lifecycle-main-loop--input)
3. [Core Utilities: Logging, Errors, Handles, Hashing, Profiling](#core-utilities-logging-errors-handles-hashing-profiling)
4. [Job System](#job-system)
5. [Asset Registry & Hot Reload](#asset-registry--hot-reload)
6. [Memory: Arena & Frame Allocators](#memory-arena--frame-allocators)
7. [GPU Abstraction Layer](#gpu-abstraction-layer)
8. [Shader Authoring & Compilation](#shader-authoring--compilation)
9. [Scene: Cameras, Culling, FramePacket & SceneRenderer](#scene-cameras-culling-framepacket--scenerenderer)
10. [Model Loading, Heightmaps & the glTF Contract](#model-loading-heightmaps--the-gltf-contract)
11. [Skeletal Animation, State Machines & IK](#skeletal-animation-state-machines--ik)
12. [Terrain & Grass Rendering](#terrain--grass-rendering)
13. [Lighting, Shadows & Ambient Occlusion](#lighting-shadows--ambient-occlusion)
14. [Model PBR Surface Shading](#model-pbr-surface-shading)
15. [Environment, Sky, Fog, Bloom & Post-Processing](#environment-sky-fog-bloom--post-processing)
16. [Particle Effects System](#particle-effects-system)
17. [Text & UI Rendering](#text--ui-rendering)
18. [Camera Controller & Camera System](#camera-controller--camera-system)
19. [Procedural Generation: Trees & Rocks](#procedural-generation-trees--rocks)
20. [Render Graph (Available, Not Yet Wired)](#render-graph-available-not-yet-wired)
21. [DevUI Developer Overlay](#devui-developer-overlay)
22. [Physics (Jolt)](#physics-jolt)

---

## Getting Started: Building & Integrating Bracken

This chapter specifies the build system, all build-time configuration, the consumption contract for host projects, the `bracken::Application` lifecycle facade, environment variables, the test/benchmark infrastructure, and the playground integration test. Sources: root `CMakeLists.txt`, `src/bracken/CMakeLists.txt`, `src/bracken/shaders/CMakeLists.txt`, `examples/CMakeLists.txt`, `examples/playground/*`, `src/bracken/application.{hpp,cpp}`, `README.md`.

### Build-system layout and targets

CMake ≥ 3.20, project `bracken` VERSION 0.1.0, languages C and CXX, `CMAKE_CXX_STANDARD 23` + `CMAKE_CXX_STANDARD_REQUIRED ON`, `CMAKE_EXPORT_COMPILE_COMMANDS ON` (root `CMakeLists.txt`, its opening block). `BUILD_SHARED_LIBS` is set `OFF` as a plain (non-cache) variable so every vendored dependency links statically; being directory-scoped, it does not leak into a super-project that consumes the engine via `add_subdirectory` or override the consumer's own `BUILD_SHARED_LIBS` choice.

Targets produced:

Line numbers are deliberately omitted from this table: cite the file and the
target name, which survive an edit. `docs_coherence.test.cpp` checks the paths
and identifiers; nothing checks the lines, on purpose.

| Target | Kind | Defined in | Condition |
|---|---|---|---|
| `bracken` | STATIC library | `src/bracken/CMakeLists.txt` | always |
| `bracken::bracken` | ALIAS for `bracken` | `src/bracken/CMakeLists.txt` | always — the spelling consumers should use |
| `bracken_imgui` | STATIC library (ImGui core + `imgui_impl_sdl3` + `imgui_impl_sdlgpu3` backends) | `src/bracken/CMakeLists.txt` | `BRACKEN_ENABLE_DEV_UI=ON` |
| `bracken_public_headers` | OBJECT library — one generated TU per header listed in `src/bracken/bracken.hpp`, each including it twice as the first thing in the file | `src/bracken/CMakeLists.txt` | always; proves every public header compiles standalone |
| `compile_shaders` | custom target (ALL) | `src/bracken/shaders/CMakeLists.txt` | always |
| `bracken_tests` | executable (GoogleTest) | `src/bracken/CMakeLists.txt` | `BRACKEN_BUILD_TESTS=ON` and at least one `*.test.cpp` exists |
| `bracken_benchmarks` | executable (Google Benchmark) | `src/bracken/CMakeLists.txt` | `BRACKEN_BUILD_BENCHMARKS=ON` and at least one `*.bench.cpp` exists |
| `playground` | executable | `examples/playground/CMakeLists.txt` | `BRACKEN_BUILD_EXAMPLES=ON` |
| `<target>_copy_shaders` | custom target (ALL), one per `bracken_copy_shaders()` call | `src/bracken/CMakeLists.txt` | per consumer |

Source discovery is glob-based and reconfigure-free: `file(GLOB_RECURSE BRACKEN_SOURCES CONFIGURE_DEPENDS "*.cpp")` with `list(FILTER ... EXCLUDE REGEX ".*\\.(test|bench)\\.cpp$")`. `CONFIGURE_DEPENDS` makes Ninja re-run CMake when the file set changes, so adding a source file requires only an incremental build. The same pattern collects `*.test.cpp` for `bracken_tests` and `*.bench.cpp` for `bracken_benchmarks`. `bracken.hpp` is additionally listed in the directory's `CMAKE_CONFIGURE_DEPENDS`, so editing the umbrella regenerates the `bracken_public_headers` TUs.

House rule (`.claude/RULES.md`): **incremental builds only** — `cmake --build build` or `ninja -C build`. Full clean rebuilds (`rm -rf build`, `--clean-first`, reconfiguring over an existing `build/`) are prohibited because the first configure fetches and compiles SDL_shadercross with its vendored DXC + SPIRV-Tools, the dominant cost of a cold build.

#### `bracken` target usage requirements

From `src/bracken/CMakeLists.txt`:

- `target_include_directories(bracken PUBLIC ${PROJECT_SOURCE_DIR}/src)` plus `target_include_directories(bracken SYSTEM PUBLIC ${tinygltf_SOURCE_DIR})` (SYSTEM: tinygltf's bundled `json.hpp` trips `-Wdeprecated-literal-operator`, and a vendored header must never be able to fail our `-Werror` build). There is deliberately **no** SDL_shadercross include directory here — `FetchContent` populates `SDL3_shadercross_SOURCE_DIR` (note the "3"), so the old entry expanded to a literal `/include` and put a bogus `-I` on every TU; the real headers arrive through the `SDL3_shadercross::SDL3_shadercross-static` interface include dirs. Headers are included `bracken/`-rooted, e.g. `#include "bracken/render/world_renderer.hpp"`. `${PROJECT_SOURCE_DIR}` resolves to the engine root (where `project(bracken)` is declared) regardless of the top-level project, so `add_subdirectory` consumption works verbatim with no extra include dirs. The same `${PROJECT_SOURCE_DIR}/src` is used for the `bracken_tests` and `bracken_benchmarks` include dirs.
- `add_library(bracken::bracken ALIAS bracken)` — the namespaced spelling consumers should link. A misspelled namespaced target is a configure-time error; a misspelled plain one silently becomes a `-lbracken` link flag.
- `target_compile_definitions(bracken PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)` — propagates reversed-Z zero-to-one clip-space to all consumers; host glm math must agree with engine projection math, so this being PUBLIC is load-bearing.
- `target_link_libraries(bracken PUBLIC ...)`: SDL3, SDL3_ttf, SDL3_image (each via generator expression `$<IF:$<BOOL:${VENDORED_SDL}>,SDL3::SDL3-static,SDL3::SDL3>` etc.), `SDL3_shadercross::SDL3_shadercross-static` (runtime cross-compile library used by shader hot-reload), `glm::glm`.
- `target_link_libraries(bracken PRIVATE Jolt)` — **PRIVATE by design**. Jolt's headers and its `JPH_*` definitions are an implementation detail of `src/bracken/physics/`; static archives still propagate the link requirement (`$<LINK_ONLY>`), so an executable linking `bracken` gets Jolt's symbols without its include paths. The include-side half of that invariant is enforced by `public_headers.test.cpp`, not by this line.
- `BRACKEN_ENABLE_TRACY=ON` additionally links `TracyClient` and defines `BRACKEN_ENABLE_TRACY=1` PUBLIC.
- `BRACKEN_ENABLE_DEV_UI=ON` links `bracken_imgui` PUBLIC and defines `BRACKEN_DEV_UI=1` PUBLIC. `bracken_imgui` compiles exactly: `imgui.cpp`, `imgui_demo.cpp`, `imgui_draw.cpp`, `imgui_tables.cpp`, `imgui_widgets.cpp`, `backends/imgui_impl_sdl3.cpp`, `backends/imgui_impl_sdlgpu3.cpp`, with `${imgui_SOURCE_DIR}` and `${imgui_SOURCE_DIR}/backends` as PUBLIC include dirs.
- `add_dependencies(bracken compile_shaders)` — shaders always compile before the engine library.

### CMake options — exact definitions and defaults

| Option | Default | Defined at | Behavior |
|---|---|---|---|
| `BRACKEN_ENABLE_TRACY` | `OFF` | root `CMakeLists.txt:18` | FetchContent tracy `v0.11.1`; forces `TRACY_ENABLE=ON`, `TRACY_ON_DEMAND=ON`; links `TracyClient`; defines `BRACKEN_ENABLE_TRACY=1` PUBLIC. OFF = the profiler macros compile to nothing (zero overhead). |
| `BRACKEN_BUILD_TESTS` | `ON` | root `CMakeLists.txt:31` | FetchContent googletest `v1.15.2` with `gtest_force_shared_crt=ON`, `INSTALL_GTEST=OFF`; builds `bracken_tests` from globbed `*.test.cpp` linked against `bracken` + `GTest::gtest_main`; `gtest_discover_tests(bracken_tests)` registers each test case as a ctest. `enable_testing()` is called when this **or** `BRACKEN_PLAYGROUND_SMOKE_TEST` is ON (root `CMakeLists.txt:40–42`). |
| `BRACKEN_BUILD_BENCHMARKS` | `ON` | root `CMakeLists.txt:56` | FetchContent benchmark `v1.9.1` with `BENCHMARK_ENABLE_TESTING=OFF`, `BENCHMARK_ENABLE_INSTALL=OFF`, `BENCHMARK_INSTALL_DOCS=OFF`; builds `bracken_benchmarks` from globbed `*.bench.cpp` linked against `bracken` + `benchmark::benchmark_main`. |
| `BRACKEN_ENABLE_DEV_UI` | `ON` | root `CMakeLists.txt:72` | FetchContent imgui `v1.92.2` (ImGui ships no CMakeLists; `MakeAvailable` only populates `imgui_SOURCE_DIR`); builds `bracken_imgui`; defines `BRACKEN_DEV_UI=1` PUBLIC. `OFF`: ImGui is neither fetched nor linked, and `src/bracken/dev/dev_ui.cpp` compiles `bracken::dev::DevUI` to a no-op stub (`init()` returns false, all other members safe to call). |
| `BRACKEN_BUILD_EXAMPLES` | `ON` | root `CMakeLists.txt:90` | `add_subdirectory(examples)` → `examples/CMakeLists.txt` (a single line: `add_subdirectory(playground)`). |
| `BRACKEN_WERROR` | `${PROJECT_IS_TOP_LEVEL}` | `cmake/BrackenWarnings.cmake:15` | Adds `-Werror` (MSVC `/WX`) to the warning set `bracken_target_warnings()` applies PRIVATE to `bracken`, `bracken_tests`, `bracken_benchmarks` and `playground`. Base set: `-Wall -Wextra -Wshadow -Wnon-virtual-dtor -Woverloaded-virtual -Wimplicit-fallthrough -Wnull-dereference`. Defaults OFF when Bracken is consumed via `add_subdirectory`, so a consuming project's toolchain can never fail on our warnings. `-Wsign-conversion`/`-Wconversion`/`-Wdouble-promotion` are deliberately excluded (they raise the count from 11 to 328, ~307 of it numeric noise in five files). |
| `BRACKEN_PLAYGROUND_SMOKE_TEST` | `OFF` | root `CMakeLists.txt:38` | Declared at the root so `enable_testing()` fires even with `BRACKEN_BUILD_TESTS=OFF`. When ON and examples are built: `add_test(NAME playground_smoke COMMAND playground --smoke 240 WORKING_DIRECTORY $<TARGET_FILE_DIR:playground>)` with properties `LABELS "integration"`, `TIMEOUT 180` (`examples/playground/CMakeLists.txt:16–19`). Requires a GPU + display. ON with `BRACKEN_BUILD_EXAMPLES=OFF` emits a configure-time warning (the playground is not built). |

### Dependency matrix — exact pins

| Dependency | Version / pin | Mechanism | Notes |
|---|---|---|---|
| SDL3 | ≥ 3.4.0 system, else `release-3.4.0` source | `find_package` QUIET, fallback FetchContent (`src/bracken/CMakeLists.txt:1–29`) | Vendored build: `SDL_SHARED=OFF`, `SDL_STATIC=ON`, test libs off, `SDL_X11_XTEST=OFF`. `VENDORED_SDL` (INTERNAL cache) selects the `-static` link targets. All three SDL libs must be found at version or all three are vendored. |
| SDL3_ttf | ≥ 3.2.2 system, else `release-3.2.2` | same | `SDLTTF_VENDORED=ON`. |
| SDL3_image | ≥ 3.4.0 system, else `release-3.4.0` | same | `SDLIMAGE_VENDORED=ON`, `SDLIMAGE_AVIF=OFF`, deps static. |
| SDL_shadercross | commit `7b7365a86611b2a7b6462e521cf1c43a037d0970`, submodules recursive | FetchContent (`src/bracken/shaders/CMakeLists.txt`) | Builds both the `shadercross` CLI (`SDLSHADERCROSS_CLI=ON`) and the static runtime library. `SDLSHADERCROSS_VENDORED=ON` + `SDLSHADERCROSS_DXC=ON` builds DXC from source (needs Python 3, the cold-build cost). Exception to all-static: `SDLSHADERCROSS_SPIRVCROSS_SHARED=ON`. Also sets `CMAKE_POSITION_INDEPENDENT_CODE=ON` as a plain directory-scoped variable (covers the FetchContent deps, does not leak into a consuming super-project) and disables SPIRV-Tools/LLVM tests. |
| tinygltf | `v2.9.7` | FetchContent | `TINYGLTF_HEADER_ONLY=ON`. **No Draco support.** |
| glm | `1.0.3` | FetchContent | header-only. |
| googletest | `v1.15.2` | FetchContent, tests only | |
| benchmark | `v1.9.1` | FetchContent, benchmarks only | |
| imgui | `v1.92.2` | FetchContent, dev UI only | |
| tracy | `v0.11.1` | FetchContent, Tracy only | |

### Shader compilation pipeline (`src/bracken/shaders/CMakeLists.txt`)

Build-time HLSL→SPIR-V via the `shadercross` CLI:

- Compiler resolution: always the freshly built, pinned CLI — `SHADERCROSS_COMMAND = "$<TARGET_FILE:shadercross>"`, with `shadercross` in every custom command's `DEPENDS`. A `shadercross` binary on `PATH` is deliberately ignored, so the build-time compiler and the runtime SDL3_shadercross library can never version-skew.
- Output directory: `SHADER_OUTPUT_DIR = "${CMAKE_CURRENT_BINARY_DIR}/spv"` — a dedicated `.spv`-only directory (`build/src/bracken/shaders/spv/` in this repo). Exported to parent scope and cached as `BRACKEN_SHADER_OUTPUT_DIR` (INTERNAL) in `src/bracken/CMakeLists.txt:135`.
- Sources: `file(GLOB_RECURSE ... CONFIGURE_DEPENDS "*.hlsl")`. Stage and entry point come purely from filename convention: `<name>.vert.hlsl` → `-t vertex -e VSMain` → `<name>.vert.spv`; `<name>.frag.hlsl` → `-t fragment -e PSMain` → `<name>.frag.spv`; anything else emits a CMake warning and is skipped.
- Per-shader `add_custom_command`: `shadercross <src> -s HLSL -d SPIRV -t <stage> -e <entry> -I <shaders src dir> -o <out>`, `DEPENDS` on the source, **all** `.hlsli` includes (`GLOB_RECURSE` of `*.hlsli`, matching the source glob), and the `shadercross` CLI target. Editing any `.hlsli` recompiles every shader.
- Stale-output pruning: at configure time, any `.spv` in `SHADER_OUTPUT_DIR` that no longer corresponds to a globbed `.hlsl` is deleted — deleted/renamed shaders can't keep shipping via `bracken_copy_shaders`.
- `compile_shaders` is an `ALL` custom target depending on every compiled `.spv`.

#### `bracken_copy_shaders(target)` — the consumer staging function

Defined at `src/bracken/CMakeLists.txt:141–152`:

```cmake
function(bracken_copy_shaders target)
    add_custom_target(${target}_copy_shaders ALL
        COMMAND ${CMAKE_COMMAND} -E rm -rf
            "$<TARGET_FILE_DIR:${target}>/shaders"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${BRACKEN_SHADER_OUTPUT_DIR}"
            "$<TARGET_FILE_DIR:${target}>/shaders"
        COMMENT "Staging shaders to ${target} output directory"
    )
    add_dependencies(${target}_copy_shaders compile_shaders)
    add_dependencies(${target} ${target}_copy_shaders)
endfunction()
```

Properties: ordered after `compile_shaders`; the consumer target depends on the copy, so shaders are staged before the executable is considered built; being a custom target with no stamp file, the remove-and-recopy re-runs on **every** build (cheap, but not a no-op) — the staging directory is deleted first, so deleted/renamed shaders never linger next to the executable.

Runtime lookup contract: `gpu::PipelineRegistry::init` sets `shader_path_ = SDL_GetBasePath() + "shaders/"`, falling back to the relative `"shaders/"` if `SDL_GetBasePath()` fails (`src/bracken/gpu/pipeline_registry.cpp:36–41`). Compiled SPIR-V must therefore sit in a `shaders/` directory next to the executable — exactly what `bracken_copy_shaders` produces. Without it, every pipeline build fails at startup.

### Environment variables — exact semantics

| Variable | Read at | Parsing | Effect |
|---|---|---|---|
| `BRACKEN_WORKERS` | `Application::init_engine()`, `src/bracken/application.cpp` (before `SDL_Init`) | `std::strtol`; `> 0` → used as-is; `< 0` → WARN `Ignoring invalid BRACKEN_WORKERS=...` + default; unset/empty/`0` → default (`hardware_concurrency() - 1`, min 1) | Worker count for the **always-on** job-system pool, passed to `JobSystem::instance().init(workers)`. `=1` is the single-worker debug mode — same code path, one worker. Logged at INFO: `Job system running with {n} worker(s)`. |
| `BRACKEN_HOT_RELOAD` | `Application::init_engine()`, `src/bracken/application.cpp:42` | `env && env[0] == '1'` (first char must be `1`) | Initializes the singleton `core::asset::FileWatcher` (background `std::jthread` poller; callbacks dispatched only from `poll_main_thread()` in the main loop). `init_renderer()` then wires `PipelineRegistry::enable_hot_reload(FileWatcher::instance())` (shader SPIR-V) and `ModelManager::enable_hot_reload(...)` (models) — `application.cpp:143–146`. Default: off, no watcher thread exists. |
| `BRACKEN_GPU_DEBUG` | `gpu::GPUDevice::init`, `src/bracken/gpu/gpu_device.cpp:30–41` | Unset → `debug_mode = !defined(NDEBUG)` (on in Debug, off in Release). Set → enabled unless the value is empty, `"0"`, `"false"`, or `"off"` (case-insensitive via `SDL_strcasecmp`). | Passes `debug_mode` to `SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV \| SDL_GPU_SHADERFORMAT_METALLIB \| SDL_GPU_SHADERFORMAT_DXIL, debug_mode, nullptr)` — SDL GPU validation layer. Costs CPU per submit. The chosen state is logged with its provenance. |

All three reads are `std::getenv` with an explicit single-threaded-at-boot justification (`BRACKEN_WORKERS` is read before the pool spawns its workers).

### `bracken::Application` — lifecycle and facade (`src/bracken/application.hpp`)

Public API:

```cpp
class Application {
public:
    Application();
    virtual ~Application();
    [[nodiscard]] bool init_engine();   // bind_main_thread, JobSystem init (always-on, BRACKEN_WORKERS), SDL_Init(VIDEO|EVENTS), hot-reload watcher
    void run();                         // main loop until quit()
    void shutdown_engine();             // FileWatcher::shutdown, JobSystem::shutdown (idempotent repeat), SDL_Quit
    void quit();                        // running_ = false
    float fps() const;
```

Protected hooks (the subclass contract): pure virtual `on_init()` and `on_simulate(SimFrame&)`; optional `on_shutdown()`, `on_main_frame(MainFrame&)`, and `on_event(const SDL_Event&) -> bool` (return `true` to consume before `InputHandler::process_event`). There is **no `on_update`/`on_render`** — `on_simulate` runs as a pool job overlapped with the previous frame's rendering and describes the frame through the `scene::FrameSink` that `SimFrame::frame()` returns; `on_main_frame` runs at the frame fence. **`run()` does not call `on_init`/`on_shutdown`** — the public `start()`/`stop()` do, and the host `main()` drives the sequence `init_engine() → start() → run() → stop() → shutdown_engine()`. `stop()` is a no-op unless `start()` returned `true`, and runs `on_shutdown()` at most once.

Protected rendering facade (all forwarding to owned subsystems, `src/bracken/application.cpp`):

```cpp
[[nodiscard]] bool init_renderer(int width, int height, const std::string& title,
                                 float world_width = 32000.0f, float world_height = 32000.0f);
void shutdown_renderer();
void render_frame(const scene::FramePacket& frame);   // consumes one sealed packet (main thread; ring-driving hosts only — run() drives it itself)
scene::PacketRing& frame_packets();                   // depth-2 ring; run() owns the acquire/retire cycle
void set_heightmap(const Heightmap& heightmap);
void set_graphics_settings(const GraphicsSettings& settings);       // main-thread tier, no sim in flight
void set_anisotropic_filter(int level);
void set_vsync_mode(int mode);
int  max_vsync_mode() const;
void set_window_mode(int window_mode, int resolution_index = 0);
void apply_all_graphics_settings(const GraphicsSettings& settings); // main-thread tier, no sim in flight
void request_graphics_settings(const GraphicsSettings& settings);   // sim-stage or main; applied at the next frame fence (last request wins; main-thread/DevUI slot wins over the sim slot)
bool init_dev_ui();
dev::DevUI& dev_ui();
GraphicsSettings& graphics_settings();                // host-owned; single writer per phase (on_simulate OR on_main_frame, not both)
std::string settings_file_path() const;                    // SDL_GetPrefPath("bracken","bracken") + "settings.cfg"
void set_settings_persistence(bool enabled);               // default true; false = never read or write that file (call BEFORE init_renderer)
bool settings_persistence() const;
std::vector<DisplayMode> available_resolutions() const;    // DisplayMode { int w, h; }
ModelManager& models();                               // mutation main-thread; const lookups worker-safe
float get_terrain_height(float x, float z);           // worker-safe (immutable heightmap reads)
int screen_width() const;  int screen_height() const; // safe from sim (written only at fence/pump)
scene::SceneRenderer& scene_renderer();
systems::EffectSystem& effect_system();               // sim-stage owned; engine-driven from the sim-job tail
void set_collect_render_stats(bool enabled);
const RenderStats& render_stats() const;
std::string gpu_driver_name() const;
void request_screenshot(const std::string& path);     // sim-stage or main; deferred to the fence — a request from sim frame i captures frame i
InputHandler& input();                                // main-thread tier (on_event/on_main_frame/init); sim code reads SimFrame::input()
systems::CameraController& camera();                  // sim-stage under run(); main-thread reads tear at worst (DevUI)
```

Owned subsystems (private, opaque to subclasses): `InputHandler input_`, `std::unique_ptr` to `render::RenderContext`, `scene::PacketRing`, `scene::SceneRenderer`, `systems::CameraSystem`, `systems::EffectSystem`, `dev::DevUI`, plus `GraphicsSettings graphics_settings_`, a `last_applied_settings_` copy for diffing, and the sim-pipeline state (builder/packet/task handle, input snapshot, dt/time/frame counters, scene/sky clocks, `SimFeedback`, pending fence requests).

#### The pipelined main loop (`Application::run`, `src/bracken/application.cpp`)

Pipeline depth 2 — Simulate(i+1) overlaps Render(i). Bootstrap: pump events once (so frame 0 sims real input), then `kick_sim()` for frame 0 (no overlap). Per steady-state iteration i:

1. **`join_sim()`** — `JobSystem::wait(sim_task_)`, a helping wait (the main thread executes pool jobs, including this frame's own sim fan-out). **Sim-job error policy:** an exception escaping `on_simulate` is caught here, `BRACKEN_LOG_FATAL`-logged, the partial packet abandoned (`FramePacket::reset()`), and the exception **rethrown out of `run()`** on the main thread.
The sim-stage tail, before the effect drain: unless `set_camera_driven_by_engine(false)`, `camera_->set_screen_size(kick-time viewport)`, `camera_->update(sim_dt_)`, `fb.set_camera(camera_->get_camera_state())`, and the ground forward into `SimFeedback` when the host did not set it. Consequence: camera output read during `on_simulate` is the previous frame's, while shakes/target moves staged during the frame land in this frame's camera.

2. **`frame_fence(i)`** — no sim in flight (asserted): `FileWatcher::poll_main_thread()` if hot-reload is initialized (callbacks run on the main thread while no packet references the old resource); pending `request_graphics_settings` application (two single-writer slots — sim-phase and main-thread/DevUI — main wins; both cleared); camera-forward feedback (`SimFrame::set_camera_forward`) applied to the `InputHandler` (shapes the NEXT snapshot's movement basis); FPS accounting (1-second window; `fps_` never written while a sim that may read `fps()` is in flight); `on_main_frame(MainFrame&)`; then the deferred screenshot path is forwarded to `SceneRenderer::request_screenshot` so the frame about to render is captured.
3. **`pump_events()`** — clears camera deltas / menu inputs / key edges; `SDL_PollEvent` loop where `SDL_EVENT_QUIT` clears `running_`; otherwise the DevUI sees the event first (`dev_ui_->process_event` — ImGui tracks state even while hidden, consumes only when visible and wanting input), then `on_event(event)` (subclass veto), then `InputHandler::process_event`. After the pump: `input_.set_keyboard_captured(dev_ui_ && dev_ui_->wants_keyboard())` and `input_.post_process_events()`. On quit: retire the sealed frame i and break.
4. **`kick_sim()`** — dt from `SDL_GetTicksNS()` (nanoseconds through double, clamped **0.1 s**), computed **at kick** and stamped into the packet via `set_frame_timing`; `sim_time_ += dt`; by-value `InputSnapshot` taken (`input_.snapshot()` — drains the text-input buffer); packet acquired from the ring, `FrameBuilder` constructed with **16 fixed lanes** (never derived from worker count — lane assignment must be deterministic); sim job submitted. Job body (`run_sim_job`, pool thread): construct `SimFrame` → `on_simulate(sim)` → **engine sim-stage tail**: apply the builder's effect spawns to the sim-owned `EffectSystem` in deterministic (lane, emission) order (`FrameBuilder::for_each_effect_spawn`; null definitions skipped) → `EffectSystem::update(JobSystem&, dt, terrain sampler)` (nested fork-join on the pool) → `EffectSystem::snapshot_into(fb)` (the effect-instance snapshot into packet memory) → advance clocks (`sim_scene_time_ += dt`; `sim_sky_time_ += dt * cloud_speed` of the staged environment — last `set_environment` wins, defaults otherwise) → `fb.set_engine_frame(scene_time, sky_time, effect_count)` → `fb.seal()` (Filling → Sealed; the TaskHandle join publishes the packet — no ring publish/acquire needed).
5. **`render_frame(packet i)`** on the main thread (Prepare fan-out + Record + present, unchanged from Phase 3) while sim i+1 runs.
6. **Retire packet i**; `BRACKEN_PROFILE_FRAME()`.

After the loop, any in-flight sim is joined (and its packet retired) before `run()` returns, so teardown never races the pipeline.

Threading contract: `init_engine()` calls `core::bind_main_thread()`; `BRACKEN_ASSERT_MAIN_THREAD()` guards FileWatcher entry points, swapchain acquire, and `render_frame` in debug builds; `on_simulate` is tier `sim-stage` — nothing reachable from it may touch GPU/SDL/main-thread-tier APIs (see `docs/architecture.md` "Threading & lifetime contracts"). **Input latency:** the snapshot taken at kick(i+1) is simulated in frame i+1 and presents at frame i+1 — one pipeline slot between input and photons, inherent to pipelining, documented on `Application`.

#### `init_renderer` / `shutdown_renderer` side effects (`application.cpp:130–185`)

`init_renderer`: creates + inits `RenderContext` (window, GPU device), queries display modes, creates + inits `SceneRenderer` with the world dimensions, wires shader and model hot-reload to the FileWatcher, creates the `CameraSystem`, **auto-loads `GraphicsSettings`** from `settings_file_path()` (missing file logs info and keeps defaults), and applies them via `apply_all_graphics_settings`. Returns `false` on either init failure.

`shutdown_renderer`: **joins the job-system pool first** (`JobSystem::instance().shutdown()` at the top — no in-flight or queued job may touch a subsystem about to be destroyed), then **auto-saves settings** (logs an error on failure), destroys the DevUI *before* the device (it owns GPU resources), then camera → `SceneRenderer::shutdown` → `RenderContext::shutdown` → resets pointers. Destruction order is load-bearing.

`apply_all_graphics_settings` (`src/bracken/application.cpp`): stores the settings, always calls `set_graphics_settings` (cheap), but gates the expensive setters on diffs against `last_applied_settings_`: `set_anisotropic_filter` (sampler rebuild) only when `anisotropic_filter` changed, `set_vsync_mode` (swapchain reconfigure) only when `vsync_mode` changed, `set_window_mode` (fullscreen transition) only when `window_mode`/`resolution_index` changed. First call applies everything.

`GraphicsSettings` (`src/bracken/contracts/graphics_settings.hpp`): plain struct with `[[nodiscard]] bool save(path)/load(path)`, `clamp_to_valid_ranges()` applied by `load()` at the trust boundary, `CURRENT_SETTINGS_VERSION = 2` with lighting-value migration for older files. Enums with stable serialized integer values: `ShadowMode {Off=0, PCF=1, PCSS=2}`, `AOMode {Off=0, GTAO_Half=1, GTAO_Full=2}`, `AAMode {Off=0, FXAA=1, TAA=2}`, `TonemapMode {ACESFitted=0, ACESNarkowicz=1, Reinhard=2, None=3}`. Notable defaults: `shadow_mode=PCSS`, `ao_mode=GTAO_Half`, `aa_mode=TAA` (fresh settings only — persisted files keep their stored mode), `bloom_enabled=true` (`bloom_threshold=1.1`, post-exposure HDR luminance), `volumetric_fog=false`, `god_rays=false`, `vsync_mode=VSyncMode::Mailbox` (clamped to `max_vsync_mode()` at apply), `anisotropic_filter=AnisotropicFilter::X16`, `draw_distance=DrawDistance::D4000` (`get_draw_distance()` maps the enum through the saturating table {500,1000,2000,4000,8000,16000,32000}), `window_mode=WindowMode::Windowed`. Every mode field is a scoped `enum class` with stable serialized integer values — `DrawDistance`, `AnisotropicFilter`, `VSyncMode`, `ShadowCascades`, `ShadowResolution`, `LocalShadowResolution` and `WindowMode` joined the four listed above in phase 7; none of them is a bare `int` any more, and the lookup helpers (`cascade_count`, `shadow_map_texels`, …) saturate rather than defaulting, because a corrupt file or a `static_cast` can still present an out-of-range value.

#### Dev UI wiring (`init_dev_ui`, `application.cpp:276–305`)

Always constructs the `DevUI` object first (so `dev_ui()` and event/render calls are safe regardless of outcome), then returns `false` if the renderer isn't up. Bindings passed to `DevUI::init` (`dev::DevUI::Bindings`, non-owning, must outlive the DevUI): `SceneRenderer*`, `CameraController*`, `GraphicsSettings*`, an `apply_settings` callback routed through **`request_graphics_settings`** (the overlay draws DURING rendering, while the next sim runs — the GPU/SDL settings work must land at the frame fence), and `const float* fps`. Panel-data safety in the pipelined loop is documented on `dev::DevUI`: panel bodies run mid-render, concurrent with the next sim — draw render-side data (packet-derived stats, `SceneRenderer` state) or values snapshotted in `on_main_frame`; never read live game state being mutated by `on_simulate` (the built-in Camera tab's live `CameraController` reads are a documented tear-only exception). On success it installs `scene_renderer_->set_post_ui_callback(...)` rendering the overlay — **there is a single post-UI callback slot**; a host using the dev UI cannot install its own. `dev_ui()` asserts `init_dev_ui()` was called. Stub builds (`BRACKEN_ENABLE_DEV_UI=OFF`): `DevUI::init` returns `false`, every other member is a safe no-op; only host code that includes `imgui.h` (panel bodies) needs a `#ifdef BRACKEN_DEV_UI` guard. Key API on `dev::DevUI` (`src/bracken/dev/dev_ui.hpp`): `process_event` (release events never consumed so game edge-state can't wedge), `set_toggle_key(SDL_Scancode)` (default `SDL_SCANCODE_UNKNOWN` = none; handled inside `process_event` so it works while ImGui has keyboard focus), `wants_keyboard()/wants_mouse()`, `render(SDL_GPUCommandBuffer*, SDL_GPUTexture*)`, `set_visible/visible/toggle`, `add_panel(std::string, std::function<void()>)` (safe before init; drawn as its own window between `Begin(name)`/`End`).

#### Screenshot path (`request_screenshot`)

`Application::request_screenshot(path)` is **deferred** (tier: sim-stage or main thread): the path is stored and forwarded to `SceneRenderer::request_screenshot` at the frame fence, so a request from sim frame i captures frame i's rendering. Per `docs/architecture.md`: on the capturing frame, after UI/PostUICallback, a copy pass downloads the swapchain texture into a `DOWNLOAD` transfer buffer; the frame submits via `RenderContext::end_frame_and_wait()` (`SDL_SubmitGPUCommandBufferAndAcquireFence` + fence wait — a **deliberate pipeline stall**, tests/debug only); pixels are saved as PNG via `IMG_SavePNG` with alpha dropped through RGB24 conversion (BGRA and RGBA swapchain formats both handled). The only other GPU→CPU readback is `render::AutoExposure`'s 16-byte fenced diagnostic download (non-blocking). The capture includes the UI and the dev overlay.

### Test and benchmark infrastructure

- Unit tests: `*.test.cpp` next to their unit, auto-globbed into `bracken_tests`, registered individually via `gtest_discover_tests`. Run: `ctest --test-dir build --output-on-failure`, or the binary directly with `--gtest_filter`. Binary path: `build/src/bracken/bracken_tests`.
- Benchmarks: `*.bench.cpp` next to their unit, auto-globbed into `bracken_benchmarks` (linked to `benchmark::benchmark_main`, so no `main` needed in bench files). Binary path: `build/src/bracken/bracken_benchmarks`; `--benchmark_filter=<regex>` selects a subset.
- Integration test: the playground smoke run (below); optionally a ctest named `playground_smoke` (label `integration`, timeout 180 s, working directory = the playground binary dir) when `BRACKEN_PLAYGROUND_SMOKE_TEST=ON`.
- `RenderStats` (`src/bracken/contracts/render_stats.hpp`), consumed by the smoke summary and dev-UI Stats panel: counters (`draw_calls`, `triangle_count`, `entities_rendered`, `entities_distance_culled`, `entities_frustum_culled`, `jobs_pending`, `instances_drawn`, `ui_vertices`, `ui_draws`, `local_shadow_tiles_rendered`), budgets (`packet_bytes_used/capacity`, `bone_palettes_used/budget`, `lights_submitted/binned`) and `std::vector<PassTiming> pass_times_ms`. Collection is opt-in via `set_collect_render_stats(bool)`; `Application::render_stats()` returns `std::optional<RenderStats>`, `nullopt` while off.
- `FrameDiagnostics` (same header), via `Application::frame_diagnostics()`: every drop path the frame took (`draws_dropped_unknown_model`, `skinned_dropped_no_skeleton`, `palettes_over_budget`, `shadow_capsules_over_budget`, `effect_models_unresolved`, `lights_dropped`, `cluster_lights_dropped`, `camera_missing`). Accumulated unconditionally — not gated on `set_collect_render_stats`.
- `PassTiming { std::string_view name; std::optional<float> cpu_ms, gpu_ms; }`: `cpu_ms` is the real CPU cost of RECORDING the pass (`core::PassTimeline`); `gpu_ms` is `nullopt` — SDL3 exposes no timestamp query API.

### Playground example — the canonical integration (`examples/playground/`)

Five files: `main.cpp` (entry point + screenshot verification), `playground.hpp/.cpp` (the `PlaygroundGame : public bracken::Application` subclass), `character.hpp/.cpp` (code-built assets). `examples/playground/CMakeLists.txt` is the reference consumer: `add_executable(playground main.cpp playground.cpp character.cpp)`, `target_link_libraries(playground PRIVATE bracken)`, `bracken_copy_shaders(playground)`. Binary: `build/examples/playground/playground`.

#### Entry point and argument contract (`main.cpp`)

`playground [--smoke [frames]]`. `--smoke` sets `smoke_frames = 240`; an immediately following argument not starting with `-` is parsed with `std::from_chars` — a non-numeric, partially numeric, or non-positive value prints `invalid --smoke frame count (expected a positive integer)` and **exits 2**. Unknown arguments → stderr + **exit 2**. Sequence: `PlaygroundGame game(smoke_frames); game.init_engine() → game.start() [= on_init] → game.run() → game.stop() [= on_shutdown] → game.shutdown_engine()`; failures exit 1. After a smoke run, `verify_smoke_screenshot(PlaygroundGame::SMOKE_SCREENSHOT_PATH)` must pass or exit 1.

`SMOKE_SCREENSHOT_PATH = "playground_smoke.png"` (`playground.hpp:34`, `static constexpr`), written to the process working directory.

#### Screenshot verification algorithm (`main.cpp::verify_smoke_screenshot`)

Loads the PNG via `IMG_Load`, converts to `SDL_PIXELFORMAT_RGB24`, computes per-pixel luma `0.2126*R + 0.7152*G + 0.0722*B`, then mean and standard deviation over all pixels. Pass criteria: `mean > 2.0 && mean < 250.0 && stddev > 5.0` — rejects missing/unreadable files, black frames, blown-out white frames, and any solid color. Prints `[smoke] screenshot OK|FAIL path=... WxH mean_luma=... stddev=...`. Rationale (comment at `main.cpp:12–14`): a silently broken renderer that still hits 60 fps cannot pass.

#### Game structure (`playground.cpp`)

Constants: `WORLD=8000`, `WORLD_MARGIN=250`, `ORB_COUNT=15`, `ORB_PICKUP_RADIUS=60`, `ORB_HOVER=26`, `WALK_SPEED=140`, `RUN_SPEED=300`, `ATTACK_DURATION=0.45`.

`on_init`: `init_renderer(1600, 900, "bracken playground", WORLD, WORLD)`; `init_dev_ui()` + `dev_ui().set_toggle_key(SDL_SCANCODE_F10)` + one host panel guarded by `#ifdef BRACKEN_DEV_UI`; builds a 257×257 `Heightmap` via `Heightmap::from_floats(heights, 257, {0, 0, 8000}, {-160, 150})` (world-unit samples from layered sines; the range is explicit so the encoding is stable under edits) and `set_heightmap`; registers models by name through `models().register_model(...)`, logging `to_string(error)` on a refusal — `"player"`, `"orb"`, `"orb_shard"` (referenced *by name* from effect definitions), three `procedural::TreeGenerator` species and three `procedural::RockGenerator` shapes with fixed seeds (fallback scene only; with village assets the pack `.glb`s load instead); builds the animation state machine; defines two member `EffectDefinition`s (`pickup_burst_`: BURST of 26 radial shards with gravity/drag and fade curves; `attack_arc_`: BURST of 12 ARC-path particles with `face_velocity`) — members because **definitions must outlive all spawned instances**; applies the hand-authored scene layout (`scene_layout.cpp`: per-area placements for props, flora, rock outcrops, and the grove-stamp forest ring) plus the procedural grass-tuft ground cover — or, in the fallback scene, scatters 70 procedural trees + 25 rocks outside the plateau clearing; places 15 orbs; configures the camera (terrain-height callback, screen size, `distance=380`, `height_offset=120`, target above the player); enables render-stats collection when in smoke mode.

State machine (`setup_state_machine`): states Idle/Walk/Run/Attack bound to same-named clips; transitions on parameters `speed` (float) and `attacking` (bool) — Idle→Walk at `speed GT 20`, Walk→Run at `GT 240`, Run→Walk at `LT 220`, Walk→Idle at `LT 20`, any→Attack on `attacking IS_TRUE` with crossfade 0.08 and priority 10; Attack is non-looping and exits to Idle on `IS_FALSE`. `bind_clips(model->animations)` resolves clip names to indices once (warns on unbound states); `param_index("speed"/"attacking")` resolves parameter names to indices once so per-frame updates (`set_float`/`set_bool` index overloads) hash nothing.

`on_simulate(SimFrame& sim)` — the whole frame, as a sim-stage pool job. Simulation half: smoke driver or input handling off `sim.input()` (the by-value `InputSnapshot`); player kinematics (speed smoothing, facing via `RotationSmoother::smooth_toward`, position clamped to the world margin, `y` from `get_terrain_height` — worker-safe); `anim_sm_.update(anim_player_)` then `anim_player_.update(model->skeleton, model->animations, sim.dt())`; then the host-side procedural layers **after** the animation update: `anim::apply_foot_ik` using a `FootIKSmoother` fed with per-foot terrain offsets from `anim_player_.world_transforms()`, and `anim::apply_body_lean` from speed (forward) and turn rate (lateral); camera AIMING only (`set_target`/`set_target_velocity`, or yaw/pitch from the snapshot) — the engine's sim-stage tail does `set_screen_size`/`update(dt)`/`set_camera` into the packet/camera-forward feedback, unless `--engine-camera=off` puts the demo back in charge of all four. Input handling shows edge-triggered keys (`in.was_key_just_pressed(...)` — the one snapshot member that keeps function form; everything else is a field, e.g. `in.move_forward`, `in.camera_yaw`, `in.mouse_left_pressed`), the graphics-toggle path (**`edit_graphics_settings()`**, a scoped handle that requests its edited copy on scope exit — the sim-safe deferred apply; F1 AO cycle / F2 bloom / F4 fog+god-rays / F5 shadow cycle / F6 AA cycle Off→FXAA→TAA / F7 IBL toggle / F8 local-shadow toggle), and F12 → `request_screenshot("screenshot_NNN.png")` (deferred to the fence).

Frame-description half of `on_simulate`: emits into `scene::FrameSink& fb = sim.frame()` — `set_camera(camera().get_camera_state())`, per-frame toggles (`set_draw_skybox(gs.skybox_enabled)`, `set_draw_ground(true)`, `set_draw_grass(gs.grass_enabled)`); for the player: `auto pal = fb.alloc_bone_palette(joint_count)`, copy `anim_player_.bone_matrices()` into `pal.matrices` (packet memory — no lifetime contract), then `fb.draw_skinned(player_model_, player_transform_, pal.id)`; `fb.draw_model` per prop/orb; `fb.add_point_light({pos, 320.0f, {1.0f,0.72f,0.30f}, 12000.0f, /*casts_shadows=*/true})` (a `PointLightDesc`) per orb (radiometric intensity; the orb mesh sets `cast_shadows = false` so the bulb doesn't occlude its own light); `fb.add_billboard_3d` proximity bars; optional `fb.add_spot_light` flashlight; `fb.spawn_effect(&definition, pos, dir)` for queued bursts; optional debug spheres/lines including a post-IK skeleton overlay from `world_transforms()`; UI via `fb.ui().add_filled_rect/add_text/add_text_wrapped`. No `set_frame_timing`, no `seal()`, no `render_frame`, no ring management — the engine does all of that around the hook.

#### Smoke driver schedule (`update_smoke_driver`)

Replaces input handling when `smoke_frames_ > 0`. Per frame: `camera().rotate_yaw(35.0f * dt)` + `set_pitch(14)` re-asserted (the controller eases pitch back to its free-play default); steer toward the nearest orb **on the plaza** (orb 0 spawns beside the player, so real pickups run every time; lane/door orbs are ignored — the orbit camera has no building collision), else idle back to the plaza centre; always running. Fixed schedule (default N=240):

| Frame | Action |
|---|---|
| every `frame % 120 == 60` | attack (`attack_timer_ = ATTACK_DURATION`, arc effect spawned) |
| 1 | force `aa_mode = TAA`, `local_shadows` on (`PLAYGROUND_LOCAL_SHADOWS=0` forces off), `auto_exposure` on (`PLAYGROUND_AUTO_EXPOSURE=0` forces off), sun CSM on (`PLAYGROUND_SUN_SHADOWS=0` forces off); reset persisted end-of-run state (`sun_intensity` → canonical 3.0, volumetric fog/god rays off); screenshot framing (camera distance 520 / height 150 / start yaw 310); flashlight on; auto-exposure readback interval → 0 (fresh `auto_ev=` at run end); `PLAYGROUND_GRADE=1` applies a non-neutral grade (temp +0.4, CDL slope (1.05,1,0.9) / offset (0.02,0,−0.02) / power 1.1); `PLAYGROUND_EXPOSURE_PROBE=1` starts per-frame `[probe] frame= ev= target= avg_log2=` lines; `PLAYGROUND_SUN_ELEVATION=<deg>` pins the sun at that elevation for the whole run (art-shot mode: default azimuth kept, host-style day/night sun intensity/tint, steady sun-facing camera at pitch −12° / yaw +10° off the sun azimuth, volumetric/sun-intensity knobs reset, and the sky-repainting flips below — orbit, 80, 110/160, 170, 180/195, 205 — are skipped); `PLAYGROUND_CLOUDS="coverage,density,scale"` (with the sun pin) overrides the cloud shaping |
| 80 | `volumetric_fog = true`, `god_rays = true` via the fence's `edit_graphics_settings()` handle — applied at the NEXT fence (skipped in sun-pinned runs) |
| 185 | volumetric fog + god rays back OFF (teardown path; keeps the dusk wash out of the screenshot; skipped in sun-pinned runs) |
| 100 / 115 | `local_shadows` off / back on (cached-tile revival) |
| 110 / 160 | `aa_mode = FXAA` / back to TAA (history restart) (skipped in sun-pinned runs) |
| 140 | `ao_mode = AOMode::GTAO_Full` → apply (fresh default is `GTAO_Half`) |
| 170 | `dev_ui().set_visible(true)` (screenshot proves the overlay renders; no-op in stub builds; skipped in sun-pinned runs) |
| 180 | swing `environment().sun_dir` off the scene's late-afternoon sun (a "day-cycle" step — the sky-relevant fields change, so the IBL prefilter chain refreshes amortized) (skipped in sun-pinned runs) |
| 195 | restore the scene's `sun_dir` (second amortized refresh; the screenshot keeps the canonical late-afternoon lighting) (skipped in sun-pinned runs) |
| 200 | `shadow_mode = ShadowMode::PCF` → apply |
| 205 | `sun_intensity = 2.4` (80% of the canonical 3.0 — "cloud passes over"; the local-light shadows read clearly in the screenshot; shading-only, no IBL refresh) (skipped in sun-pinned runs) |
| `max(1, N-20)` | `request_screenshot(SMOKE_SCREENSHOT_PATH)` — late enough that all flips are visible |
| `>= N` | print summary and `quit()` |

Summary format (exact): `[smoke] OK frames={} fps={:.0f} draw_calls={} triangles={} entities={} orbs_collected={} village={} ibl_refreshes={} local_tiles={} local_lights={}p+{}s auto_ev={} driver={}` sourced from `render_stats()` (requires the `set_collect_render_stats(true)` done in `on_init`), `scene_renderer().ibl()->refresh_count()` (4 on a default run: init build, the frame-1 environment apply, and the two sun swings), `village=` (1 when the generated medieval assets loaded, 0 in the procedural fallback), `scene_renderer().local_shadows()->selector()` (total tiles re-rendered over the run + point/spot lights shadowed on the final frame), `scene_renderer().auto_exposure()->latest_state()` (the adapted EV from the fenced debug readback, `{:.2f}`; `n/a` when auto exposure is off/never read back), and `gpu_driver_name()`.

#### Code-built assets (`character.cpp`)

Demonstrates the minimum viable in-code asset pipeline: `build_character_model()` constructs an 11-joint skeleton (parents strictly before children — an `AnimationPlayer` requirement; joint names `Hips/Spine/Head/LeftUpperLeg/.../RightArm` match the engine's default `FootIK` bone names so `model->foot_ik.init(model->skeleton)` binds), skinned box geometry (`gpu::SkinnedVertex` with single-joint weights, per-face normals), inverse bind matrices as negative translations of the pure-translation bind pose, and four clips — `Idle` (3.0 s), `Walk` (0.9 s), `Run` (0.55 s) via a parameterized locomotion-cycle sampler (16 keys, closures over phase 0..1, last key repeating phase 0 for clean loops), and a non-looping `Attack` (0.45 s) with explicit keys. `build_orb_model()`/`build_shard_model()` are UV spheres (`gpu::Vertex3D`) with HDR emissive factors (orb `{2.4,1.7,0.55}` — above the default `bloom_threshold` of 1.1 so orbs feed the bloom pass); `build_chrome_sphere_model()` is a polished-metal UV sphere (metallic 1.0, roughness 0.06) placed in the spawn clearing — the specular-IBL proof piece (it mirrors the prefiltered sky, sun disc included). Every model sets its AABB `bounds` explicitly (the sphere models to their construction radius, the skinned character to hand-authored bounds with animation headroom); `register_model` computes the bounding sphere and sets `loaded`.

### Known limitations and silent behaviors in this area

- `bracken_copy_shaders` removes and re-copies the shader staging directory on every build (custom `ALL` target, no stamp).
- `request_screenshot` stalls the GPU for the capturing frame by design.
- The default main-loop `dt` clamp is 0.1 s; `fps()` updates once per second.

---

## Application Lifecycle, Main Loop & Input

Scope: `src/bracken/application.hpp`/`.cpp`, `src/bracken/input_handler.hpp`/`.cpp`, `src/bracken/input_state.hpp`, `src/bracken/contracts/graphics_settings.hpp`/`.cpp`, `src/bracken/contracts/render_stats.hpp`. All types live in namespace `bracken` (no sub-namespace). This layer is the host-facing boundary: it owns SDL, drives the frame, and forwards everything rendering-related to opaque owned subsystems (`render::RenderContext`, `scene::SceneRenderer`, `systems::CameraSystem`, `dev::DevUI`).

### Application (`src/bracken/application.hpp`)

Abstract base class. Subclasses implement game logic; `Application` owns lifecycle, loop, timing, input, and the renderer/camera facade.

#### Public API

```cpp
class Application {
public:
    Application();                       // = default
    virtual ~Application();              // = default
    [[nodiscard]] bool init_engine();    // SDL init; call before start()
    void run();                          // main loop until quit()
    void shutdown_engine();              // SDL teardown; call after on_shutdown()
    void quit();                         // sets running_ = false
    float fps() const;                   // frames completed in the last whole second
};
```

Note the asymmetry: `run()` drives `on_simulate`/`on_main_frame`/`on_event` per frame, but **never** invokes `on_init`/`on_shutdown` — the public `start()`/`stop()` do, and the host driver calls those around `run()` (see `examples/playground/main.cpp`). `quit()` is worker-safe (`running_` is a relaxed atomic flag — it publishes nothing; all cross-thread data movement rides the sim kick/join edges).

#### Protected virtual hooks

```cpp
virtual bool on_init() = 0;                            // main thread, driven by start(), GPU ready, no sim in flight
virtual void on_shutdown() {}                          // ditto
virtual void on_simulate(SimFrame& sim) = 0;           // POOL JOB, overlapped with the previous frame's rendering (tier sim-stage)
virtual void on_main_frame(MainFrame& mf) {}           // main thread, at the frame fence (no sim in flight)
virtual bool on_event(const SDL_Event& event) { return false; } // main thread, during the pump (no sim in flight); true = consume
```

#### `SimFrame` / `MainFrame` surfaces (`src/bracken/sim_frame.hpp`)

`SimFrame` — the host's view of one simulation frame; tier `sim-stage`; valid only for the duration of `on_simulate` (never store a pointer to it or to `input()`). Non-copyable/movable. Engine-internal constructor `SimFrame(FrameBuilder&, const InputSnapshot&, float dt, double time, uint64_t frame_index, SimFeedback* = nullptr, int screen_width = 0, int screen_height = 0)` — tests may hand-build one (feedback null ⇒ `set_camera_forward` is a no-op).

- `float dt()` — computed at kick, clamped 0.1 s; identical to the packet's dt. `double time()` — accumulated dt. `uint64_t frame_index()` — monotonic, 0 on the first frame.
- `const InputSnapshot& input()` — the frame's immutable input snapshot, taken on the main thread at this frame's kick (`InputHandler::snapshot()`, which also drains the text-input buffer). `InputSnapshot` (`input_handler.hpp`) embeds the derived `InputState` as `.state` plus public fields mirroring the old `InputHandler` getters — `mouse_x/mouse_y`, `mouse_left_held/pressed/released`, `mouse_right_pressed`, `camera_yaw/camera_pitch/camera_zoom_delta`, `move_forward/backward/left/right`, `menu_*_pressed`, `text_input` (drained delta) + `text_backspace/enter/escape_pressed`, `has_controller/controller_primary/secondary` — plus `keys_held` (a `std::bitset<SDL_SCANCODE_COUNT>` of polled level state) — and two members that keep function form, `was_key_just_pressed(SDL_Scancode)` and `key_held(SDL_Scancode)`.
- `scene::FrameSink& frame()` — the frame's host write surface (a pointer-sized non-owning view of the frame's `FrameBuilder`, 16 engine-fixed lanes); the ONLY way render state leaves the game. It forwards draws/lights/billboards/debug/spawns/`ui()`/`lane()`/camera/environment/toggles. `seal()` and `set_frame_timing` are **not members of it** and are private on `FrameBuilder` — the engine does both.
- `parallel_for(begin, end, body(i), grain = 0)` and `parallel_for(range, body(elem&), grain = 0)` — delegate to `core::jobs::parallel_for(JobSystem::instance(), …)`, joined before returning; nested fork-join supported; small ranges run inline. Bodies must **not** emit into a builder lane another job is using (no automatic lane assignment — emit from root code or via `run_async` jobs with explicit `fb.lane(i)`).
- `run_async(TaskGroup& group, std::function<void()>)` — host-owned group; the host must join it (`JobSystem::instance().wait(group)`) before touching results; frame-scoped groups must be joined before `on_simulate` returns.
- `screen_width()` / `screen_height()` — the frame's window client size, snapshotted on the main thread at kick. Sim code reads these, never `Application::screen_width()`: `RenderContext` is main-thread tier and may be reconfiguring the swapchain while frame i+1 simulates. 0 without a renderer.
- `set_camera_forward(glm::vec2 ground_forward)` — **overrides** the movement basis the engine records from the camera it drives: written into `SimFeedback` (which the engine fills only if the host did not), applied to the `InputHandler` at the next frame fence, shaping the NEXT snapshot's `InputState::move_dir` (same one-frame latency the pre-pipelined loop had).

`MainFrame` — the host's view of the frame fence; tier `main-thread`, no sim in flight. `uint64_t frame_index()` (the frame about to render), `float fps()`, `int screen_width()/screen_height()` (0 without a renderer), `request_screenshot(std::string)` (passthrough to `Application::request_screenshot`). Deliberately exposes no GPU/renderer surface.

#### Protected facade surface

```cpp
InputHandler& input();  const InputHandler& input() const;

[[nodiscard]] bool init_renderer(int width, int height, const std::string& title,
                                 float world_width = 32000.0f, float world_height = 32000.0f);
void shutdown_renderer();
void render_frame(const scene::FramePacket& frame);   // consumes one sealed packet

void set_heightmap(const Heightmap& heightmap);
void set_graphics_settings(const GraphicsSettings& settings);
void set_anisotropic_filter(int level);
void set_vsync_mode(int mode);
int  max_vsync_mode() const;
void set_window_mode(int window_mode, int resolution_index = 0);
void apply_all_graphics_settings(const GraphicsSettings& settings);

bool init_dev_ui();
dev::DevUI& dev_ui();

GraphicsSettings& graphics_settings();  const GraphicsSettings& graphics_settings() const;
std::string settings_file_path() const;
void set_settings_persistence(bool enabled);  bool settings_persistence() const;

struct DisplayMode { int w, h; };
std::vector<DisplayMode> available_resolutions() const;

ModelManager& models();
float get_terrain_height(float x, float z);
int screen_width() const;  int screen_height() const;
scene::SceneRenderer& scene_renderer();

void set_collect_render_stats(bool enabled);
const RenderStats& render_stats() const;
std::string gpu_driver_name() const;
void request_screenshot(const std::string& path);

systems::CameraController& camera();  const systems::CameraController& camera() const;
```

#### Private state

```cpp
InputHandler input_;                 // constructed with Application (before SDL_Init — see InputHandler ctor)
std::atomic<bool> running_{false};   // relaxed flag: quit() is worker-safe; publishes nothing
uint64_t last_frame_time_ = 0;       // nanoseconds (SDL_GetTicksNS) at last kick
float fps_ = 0.0f;                   // written only at the frame fence
int frame_count_ = 0;
uint64_t fps_timer_ = 0;             // nanoseconds

// Sim pipeline state (owned by the main-thread coordinator; handed to the
// sim job at kick and back at join — ordering rides the kick/join edges,
// single writer per phase, no atomics):
std::unique_ptr<scene::FrameBuilder> sim_builder_;   // outlives the job; reset after join
scene::FramePacket* sim_packet_ = nullptr;
core::jobs::TaskHandle sim_task_;
bool sim_in_flight_ = false;                         // main-thread bookkeeping only
static constexpr uint32_t k_sim_lanes = 16;          // fixed lane budget (determinism: never worker-count-derived)
InputSnapshot sim_input_;            // written at kick, read by the job
float sim_dt_; double sim_time_; uint64_t sim_frame_index_, sim_frame_current_;
float sim_scene_time_, sim_sky_time_;                // written inside the sim job (tail)
render::EnvironmentState sim_env_;                   // last host-staged environment (cloud_speed scales the sky clock)
SimFeedback sim_feedback_;                           // camera-forward feedback, applied at the fence
GraphicsSettings pending_graphics_settings_;         // sim-phase request slot (+ has_ flag)
GraphicsSettings pending_graphics_settings_main_;    // main-thread/DevUI request slot (+ has_ flag; wins)
std::string pending_screenshot_path_;                // deferred, forwarded at the fence

std::unique_ptr<render::RenderContext>  context_;
std::unique_ptr<scene::PacketRing>      packet_ring_;  // lazily created; depth 2
std::unique_ptr<scene::SceneRenderer>   scene_renderer_;
std::unique_ptr<systems::CameraSystem>  camera_;
std::unique_ptr<systems::EffectSystem>  effect_system_; // sim-stage owned
std::unique_ptr<dev::DevUI>             dev_ui_;   // created by init_dev_ui(); destroyed before the renderer

GraphicsSettings graphics_settings_;
GraphicsSettings last_applied_settings_;   // diff base for apply_all_graphics_settings
bool has_applied_settings_ = false;
```

The subsystem pointers are opaque to subclasses; the facade dereferences them **unchecked** (no null asserts in `application.cpp` except in `dev_ui()`), so any facade call before a successful `init_renderer()` is undefined behavior (null deref).

#### init_engine() (`application.cpp`)

1. `core::bind_main_thread()` — records the engine main thread for `BRACKEN_ASSERT_MAIN_THREAD()` checks (FileWatcher entry points, swapchain acquire, `render_frame`; debug-build asserts, see `docs/architecture.md` "Threading & lifetime contracts").
2. **Job system** (always-on, before every other subsystem so any of them may submit jobs): parse `BRACKEN_WORKERS` (see the env-var table), `core::jobs::JobSystem::instance().init(workers)`, log `Job system running with {n} worker(s)` at INFO.
3. `SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)`; on failure logs `BRACKEN_LOG_FATAL("app", ...)` and returns `false`.
4. If env `BRACKEN_HOT_RELOAD` is set and its first char is `'1'`: `core::asset::FileWatcher::instance().init()` (logged at INFO on success). The `getenv` calls are annotated single-threaded-at-boot (`NOLINTNEXTLINE(concurrency-mt-unsafe)`).

#### run() — the pipeline coordinator

See "The pipelined main loop" earlier in this document for the full per-iteration schedule (join → fence → pump → kick → render → retire). Details that live at this level:

- **Timing** (in `kick_sim`): `now = SDL_GetTicksNS()`; `dt = (now - last_frame_time_) / SDL_NS_PER_SECOND` computed in `double`, narrowed to `float`. Nanosecond ticks are deliberate: whole-millisecond timing quantizes `dt` to 0/1 ms steps at high frame rates (comment in source). Then `dt = std::min(dt, 0.1f)` (clamp against huge jumps; no fixed timestep, no accumulator). dt is computed **at kick** and carried in the packet.
- **FPS** (at the fence): `frame_count_++`; once per second `fps_ = frame_count_`. So `fps()` is a once-per-second integer-valued frame count (0.0 during the first second), never written while a sim is in flight — safe to read from sim code.
- **Event phase** (`pump_events`, profiled zone `"App::poll_events"`, runs after the fence with no sim in flight):
  - `input_.reset_camera_deltas()` (zeroes zoom delta), `input_.clear_menu_inputs()`, `input_.clear_key_edges()`.
  - `SDL_PollEvent` loop. Per event: `SDL_EVENT_QUIT` → `running_ = false` (relaxed store) and return. Else if `dev_ui_ && dev_ui_->process_event(event)` → consumed (the overlay sees events **first**; ImGui tracks state even while hidden but consumes only when visible + wants input). Else if `on_event(event)` returns `true` → consumed by subclass. Else `input_.process_event(event)`.
  - `input_.set_keyboard_captured(dev_ui_ && dev_ui_->wants_keyboard())` — set **every frame** (host values are overwritten).
  - `input_.post_process_events()`.
- Profile zones: `App::join_sim`, `App::frame_fence`, `App::on_main_frame`, `App::poll_events`, `App::render` on the main thread; `App::simulate` / `App::on_simulate` inside the sim job; `BRACKEN_PROFILE_FRAME()` per loop iteration.

Mouse-button edge flags are consumed by `InputHandler::snapshot()` — the snapshot that copies an edge owns it, so one click is exactly one sim frame's edge. `InputHandler::retain_mouse_edges()` is a one-shot opt-out for a UI layer that reads the live handler after the kick; `clear_mouse_edges()` / `consume_mouse_left_pressed()` remain for hosts that drive their own loop.

#### shutdown_engine()

`FileWatcher::instance().shutdown()` → `core::jobs::JobSystem::instance().shutdown()` → `SDL_Quit()`. The `JobSystem` call here is normally an idempotent no-op — the pool is already joined at the top of `shutdown_renderer()` (before GPU teardown, so no job can outlive the renderer/device); this repeat is the safety net for hosts that never brought up a renderer.

#### init_renderer(width, height, title, world_width, world_height)

1. Construct `render::RenderContext`; `context_->init(width, height, title)`; on failure return `false` (context left constructed).
2. `context_->query_display_modes()`.
3. Construct `scene::SceneRenderer`; `scene_renderer_->init(*context_, world_width, world_height)`; on failure return `false`.
4. Hot-reload wiring: `scene_renderer_->pipeline_registry()->enable_hot_reload(FileWatcher::instance())` (if the registry exists) and `scene_renderer_->models().enable_hot_reload(...)`. These are wired unconditionally; they only do anything if the watcher was initialized.
5. Construct `systems::CameraSystem`.
6. Auto-load settings: `graphics_settings_.load(settings_file_path())`; a missing file is logged at INFO ("using defaults") — not an error. **Skipped entirely when `settings_persistence()` is false** (logged once at INFO), so the struct's defaults stand.
7. `apply_all_graphics_settings(graphics_settings_)`.

Default world extents 32000×32000 are a documented load-bearing constant (`docs/architecture.md` constants table).

#### shutdown_renderer() — teardown order

1. `core::jobs::JobSystem::instance().shutdown()` — **join the worker pool before any GPU teardown**: no in-flight or queued job can touch a subsystem (renderer, device, model manager) about to be destroyed (source comment).
2. Auto-save: `graphics_settings_.save(settings_file_path())`; failure logged `BRACKEN_LOG_ERROR("settings", ...)`, not fatal. **Skipped entirely when `settings_persistence()` is false** — a run that did not read the file must not write it either.
3. `dev_ui_->shutdown(); dev_ui_.reset()` — **the dev overlay owns GPU resources and must die before the device** (source comment).
4. `camera_.reset()`; `scene_renderer_->shutdown()`; `context_->shutdown()`; then reset both pointers.

#### apply_all_graphics_settings(settings) — diffing algorithm

- `graphics_settings_ = settings` (the persisted copy) and `set_graphics_settings(settings)` (cheap, always forwarded to `SceneRenderer`).
- Expensive setters run only when their inputs changed vs `last_applied_settings_` (or on the first apply, `has_applied_settings_ == false`):
  - `set_anisotropic_filter` — sampler rebuild — gated on `anisotropic_filter`.
  - `set_vsync_mode` — swapchain reconfigure — gated on `vsync_mode`.
  - `set_window_mode(window_mode, resolution_index)` — fullscreen transition — gated on `window_mode` **or** `resolution_index`.
- Finally `last_applied_settings_ = settings; has_applied_settings_ = true`.

#### settings_file_path()

`SDL_GetPrefPath("bracken", "bracken")` + `"settings.cfg"`; returns `""` if SDL cannot produce a pref path (both auto-load and auto-save skip an empty path). The `char*` from SDL is freed with `SDL_free`.

#### set_settings_persistence(bool) / settings_persistence()

Plain bool member, default `true`, main-thread tier; read at `init_renderer()` (step 6) and `shutdown_renderer()` (step 2), so it must be set **before** `init_renderer()` to suppress the load. `false` makes the engine neither read nor write `settings_file_path()`.

The file is a single process-global mutable input: with persistence on, a run's starting state is whatever the last run (or a concurrent one) left behind, and every field round-trips — including `rt_shadows` / `rt_reflections`, so one opt-in ray-tracing run arms every later run on that machine. Any run whose output is a *measurement* (regression/CI/benchmark/screenshot A/B) should turn persistence off; `examples/playground` does so as soon as `--smoke` is parsed.

#### Window / presentation facade semantics (implemented in `render/render_context.cpp`)

- `set_window_mode(window_mode, resolution_index)`: no-op if window absent or both values unchanged (transitions are expensive). Mode `0` = windowed (`SDL_SetWindowFullscreenMode(win, nullptr)` + `SDL_SetWindowFullscreen(win, false)`); `1` = borderless fullscreen (null mode + fullscreen true); `2` = exclusive: matches `available_resolutions_[resolution_index]` against `SDL_GetFullscreenDisplayModes` by `w`, `h`, and `pixel_density`; if no match is found it still calls `SDL_SetWindowFullscreen(win, true)` without setting a display mode (silent fallback to borderless-like behavior). Ends with `update_viewport()`.
- `max_vsync_mode()`: `2` if the device supports `SDL_GPU_PRESENTMODE_MAILBOX`, else `1` if `SDL_GPU_PRESENTMODE_VSYNC`, else `0`. (`SceneRenderer::max_vsync_mode()` returns `1` when it has no context.)
- `set_vsync_mode(mode)`: clamps `mode = std::min(mode, max_vsync_mode())`, no-op if unchanged; maps 0→`SDL_GPU_PRESENTMODE_IMMEDIATE`, 1→`VSYNC`, 2→`MAILBOX`; failure of `set_swapchain_parameters` is logged (`BRACKEN_LOG_WARN`, tag `"render"`), not returned.
- `available_resolutions()` (`Application` level) copies `RenderContext`'s display-mode list into `Application::DisplayMode{int w, h}` (pixel density is dropped at this boundary; `RenderContext` keeps it for exclusive-mode matching).
- `screen_width()`/`screen_height()` return `RenderContext::width()/height()` (current drawable size).

#### init_dev_ui() / dev_ui()

- Always constructs `dev::DevUI` first, even when the renderer is down or the overlay is compiled out — so `dev_ui()`, `process_event()`, and `render()` remain safe no-ops afterwards.
- Returns `false` if `context_`/`scene_renderer_`/`camera_` are missing, if the build is the stub (`BRACKEN_ENABLE_DEV_UI=OFF`), or if backend init fails.
- On success, builds `dev::DevUI::Bindings{ scene_renderer_.get(), &camera(), &graphics_settings_, apply-callback → apply_all_graphics_settings, &fps_ }` and **claims the `SceneRenderer`'s single post-UI callback slot** via `scene_renderer_->set_post_ui_callback(...)` to draw the overlay into the swapchain after UI. Hosts that call `init_dev_ui()` must not set their own post-UI callback (single slot; last writer wins).
- `dev_ui()` asserts `dev_ui_ != nullptr` (`BRACKEN_ASSERT`) — valid after `init_dev_ui()` *including* when it returned `false`.

#### request_screenshot(path)

Forwards to `SceneRenderer::request_screenshot(path)`, which just stores the path (`pending_screenshot_path_ = std::move(path)` — one pending slot, later calls overwrite). Consumed during the next `render_frame` (`scene/scene_renderer.cpp`): after UI/PostUICallback, a copy pass downloads the swapchain texture into a `DOWNLOAD` transfer buffer; the frame submits via `RenderContext::end_frame_and_wait()` (`SDL_SubmitGPUCommandBufferAndAcquireFence` + fence wait — a **deliberate full pipeline stall**); mapped pixels are saved via `IMG_SavePNG` after RGB24 conversion (alpha dropped; both BGRA and RGBA swapchain orders handled). Intended for integration tests and debugging only (the only other GPU→CPU readback, `render::AutoExposure`'s fenced 16-byte diagnostic download, never stalls). The capture includes UI and the dev overlay because it happens after the post-UI callback. Full pass-order context: `docs/architecture.md` §"Pass order", step 12.

#### Threading & lifetime contracts

- Everything in this chapter is main-thread-only. `core::bind_main_thread()` in `init_engine` + `BRACKEN_ASSERT_MAIN_THREAD()` guard the asserted boundaries (debug builds).
- `render_frame(const scene::FramePacket&)` consumes a sealed packet exactly once; the packet carries **no host-lifetime contracts** (everything in it is by value or packet-allocated — see the scene chapter). The one asset-lifetime rule that survives: `EffectDefinition*` in spawn commands must outlive the spawned instances.
- Destruction order is load-bearing: `dev_ui_` before `scene_renderer_`/`context_` (GPU resources vs device), enforced in `shutdown_renderer()`. Member declaration order in the header also places `dev_ui_` after the renderer members, so default destruction (if `shutdown_renderer` was skipped) destroys `dev_ui_` first as well.

### InputHandler (`src/bracken/input_handler.hpp`, `.cpp`)

Engine-level raw input: mouse, keyboard, one controller, camera-orbit accumulation, camera-relative movement axes. Explicit non-goal: game actions (attack/sprint/skills) belong in a client-side bindings layer.

#### Constants

| Constant | Value | Where |
|---|---|---|
| `CONTROLLER_STICK_DEADZONE` | `0.15f` | `input_handler.hpp` (private, `static constexpr`) |
| `CONTROLLER_TRIGGER_DEADZONE` | `0.1f` | `input_handler.hpp` |
| default `mouse_sensitivity_` | `0.35f` | member initializer |
| default `controller_sensitivity_` | `2.5f` | member initializer |
| default `camera_pitch_` | `20.0f`° | member initializer ("over-the-shoulder action view") |
| pitch clamp | `[-70.0f, 70.0f]`° | `process_event` / `post_process_events` |
| yaw wrap | `[0, 360)`° via ±360 loops | same |
| wheel → zoom | `camera_zoom_delta_ -= wheel.y * 50.0f` | `process_event` |
| RB → zoom | `camera_zoom_delta_ += 2.0f` per frame | `update_input_from_controller` |
| stick digital threshold | `±0.5` on post-deadzone axes | `update_input_from_keyboard` |
| change-detection epsilon | `0.01f` on move/attack dirs | `post_process_events` |
| key-edge storage | `std::array<bool, SDL_SCANCODE_COUNT>` | `key_just_pressed_` |
| held-key storage (snapshot only) | `std::bitset<SDL_SCANCODE_COUNT>` | `InputSnapshot::keys_held` |

#### Construction / destruction

The constructor runs when `Application` is constructed (i.e. **before** `init_engine()`'s `SDL_Init`): it reads the initial mouse position via `SDL_GetMouseState`, calls `SDL_InitSubSystem(SDL_INIT_GAMEPAD)` if the gamepad subsystem isn't up, and scans `SDL_GetJoysticks` for an already-connected gamepad (opens the first one that `SDL_IsGamepad`). The destructor closes the gamepad if open.

#### Event API — two usage modes

- Under `Application`: the app polls SDL itself and calls `process_event(const SDL_Event&)` per event, then `post_process_events()` once; the app clears edges at the top of the frame.
- Standalone: `bool process_events()` does everything — resets zoom delta, all six menu edges, and key edges; polls SDL; returns `false` on `SDL_EVENT_QUIT`; calls `post_process_events()`. Not used by `Application::run()`.

#### process_event(event) — processing order

1. **Text-input mode** (`text_input_enabled_`): `SDL_EVENT_TEXT_INPUT` appends `event.text.text` to `text_input_buffer_` and returns. Non-repeat `KEY_DOWN` of Backspace/Return/Escape latches the corresponding `text_*_pressed_` flag and returns. All other `KEY_DOWN`/`KEY_UP` events are swallowed (gameplay never sees them). Non-key events fall through.
2. **Key edges**: non-repeat `SDL_EVENT_KEY_DOWN` with in-range scancode sets `key_just_pressed_[scancode] = true`. Done before menu handling so all consumers share one source of truth (`was_key_just_pressed`).
3. **Gamepad hot-plug**: `SDL_EVENT_GAMEPAD_ADDED` / `REMOVED` → `handle_controller_added/removed`.
4. **Gamepad menu buttons** (`GAMEPAD_BUTTON_DOWN`, requires open gamepad): Start → `menu_toggle_pressed_` (unconditional); D-pad Up/Down/Left/Right and South → corresponding `menu_*_pressed_` **only while `!game_input_enabled_`**.
5. **Keyboard menu keys** (non-repeat `KEY_DOWN`, matched by physical scancode — the same convention as movement polling and the key-edge array, so the physical WASD keys that move the character also navigate menus on non-QWERTY layouts): Escape → `menu_toggle_pressed_` (unconditional); Up/W, Down/S, Left/A, Right/D, Return/Space → menu edges only while `!game_input_enabled_`.
6. **Mouse position** (`MOUSE_MOTION`, any mode): `mouse_x_/mouse_y_ = event.motion.x/y` — UI hover/drag needs a live cursor even with menus open.
7. **Mouse button edges** (any mode, so UI can consume clicks): LMB down → `left_mouse_down_ = true`, `left_mouse_pressed_edge_ = true`, press position recorded (`left_mouse_press_x_/y_`); RMB down → `right_mouse_pressed_edge_ = true`; LMB up → `left_mouse_down_ = false`, `left_mouse_released_edge_ = true`.
8. **Game-input-gated camera controls** (only when `game_input_enabled_`): RMB down/up toggles `right_mouse_down_` and SDL relative mouse mode on the event's window (`SDL_SetWindowRelativeMouseMode`); `MOUSE_WHEEL` accumulates zoom; `MOUSE_MOTION` while RMB held applies `xrel/yrel * mouse_sensitivity_` to yaw/pitch with inversion multipliers (`x_mult = invert_x ? +1 : -1`, `y_mult = invert_y ? -1 : +1`), clamps pitch to ±70°, wraps yaw to [0, 360).

#### post_process_events() — per-frame finalization

1. `last_input_ = current_input_` (change-detection base).
2. `update_input_from_controller()` if a gamepad is open **and** `game_input_enabled_` (stick/trigger/bumper polling; see below).
3. `gameplay_active = game_input_enabled_ && !text_input_enabled_`. If active:
   - `update_input_from_keyboard()`: polls `SDL_GetKeyboardState`. Keyboard contribution is muted when `keyboard_captured_` (overlay owns the keyboard; polling bypasses event consumption, hence this separate gate). Axes: forward = W/Up, backward = S/Down, left = A/Left, right = D/Right (scancodes), each OR-ed with the controller's post-deadzone left stick beyond ±0.5 to produce `move_forward_/backward_/left_/right_`.
   - `update_camera_from_mouse()`: derives world-space movement. Normalizes the host-provided camera forward (`camera_forward_x_/z_`, default `(0, -1)`), builds `right = (-fz, fx)`. If the controller stick has analog input (|axis| > 0.01), movement = `stick_x * right + (-stick_y) * forward` (stick Y is inverted); else digital WASD composes ±forward/±right. `normalize_move_dir` clamps magnitude to 1 (preserving sub-unit analog tilt). Writes `current_input_.move_dir_x = move_x`, `.move_dir_y = move_z` (2D convention: world Z → InputState Y). Legacy booleans: `move_up = len > 0.1 && move_z < -0.3` etc. (thresholded direction components). `attack_dir_x/y` = normalized camera forward.
   - Controller camera: right stick applied as `camera_yaw_ += stick_x * controller_sensitivity_ * x_mult` (and pitch analogously), same clamp/wrap as mouse.
4. If not active: the four digital movement flags and `move_dir_x/y` are zeroed (yaw/pitch/attack_dir keep their last values).
5. `input_changed_` = any legacy boolean changed, or any of `move_dir_x/y`, `attack_dir_x/y` moved by more than `0.01f`.

#### Controller subsystem

- **Single-gamepad policy**: `handle_controller_added` ignores new devices while one is open. `handle_controller_removed` (matching id) closes it, zeroes all controller state, and re-scans `SDL_GetJoysticks` for a replacement. Connect/disconnect logged at INFO (`"input"` category).
- Axis reads normalize by `/32767.0f`. Sticks use a **radial** deadzone: if magnitude > `0.15`, rescale `(mag - dz) / (1 - dz)` (clamped to 1) along the stick direction; else zero. Left stick → `controller_move_x_/y_`, right stick → `controller_camera_x_/y_`.
- `controller_primary_` = right trigger > `0.1` **or** `SDL_GAMEPAD_BUTTON_SOUTH` held; `controller_secondary_` = left trigger > `0.1` **or** `SDL_GAMEPAD_BUTTON_LEFT_SHOULDER` held. Raw latched levels, not edges; clients map to actions.
- `SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER` held adds `+2.0f`/frame to `camera_zoom_delta_` (zoom out). There is no zoom-in bumper.
- `has_controller()`, `get_controller_name()` (returns `"No Controller"` when absent).

#### Edge/state accessor contract (who clears what)

| Accessor group | Set by | Cleared by |
|---|---|---|
| `was_key_just_pressed(sc)` | non-repeat KEY_DOWN | `clear_key_edges()` — called by `Application::run` (or `process_events()`) each frame |
| `menu_*_pressed()` | events (see above) | `clear_menu_inputs()` — called by the app loop each frame |
| `get_camera_zoom_delta()` | wheel / RB | `reset_camera_deltas()` — called by the app loop each frame (yaw/pitch are *not* deltas and persist) |
| `mouse_left_pressed/released`, `mouse_right_pressed` | button events | `snapshot()` consumes them (opt out for one snapshot with `retain_mouse_edges()`); `clear_mouse_edges()` / `consume_mouse_left_pressed()` remain for manual loops. |
| `mouse_left_held()` | level state | button-up event |
| `text_backspace/enter/escape_pressed()` | key events in text mode | **host**: `clear_text_events()`; also reset by `set_text_input_enabled()` transitions |
| `take_text_input()` | TEXT_INPUT events | self-clearing (move + clear) |

#### Mode gates

- `set_game_input_enabled(bool)` / `is_game_input_enabled()` — menu mode. Gates: RMB orbit + relative mouse mode, wheel zoom, movement axes, controller polling; enables menu navigation edges.
- `set_keyboard_captured(bool)` — mutes only polled keyboard movement (controller unaffected). Deliberately separate from `set_game_input_enabled` so the two owners (dev overlay vs. game menus) can't stomp each other; written every frame by `Application::run` from `DevUI::wants_keyboard()`.
- `set_text_input_enabled(bool)` / `is_text_input_enabled()` — toggling on calls `SDL_StartTextInput(SDL_GetKeyboardFocus())`; off calls `SDL_StopTextInput` on the focused window (if any) and clears the buffer; both directions reset the three text edges; no-op if state unchanged. While on, `post_process_events` suspends gameplay movement even though `game_input_enabled_` may still be true.

#### Misc accessors

`mouse_x()/mouse_y()`; `get_camera_yaw()/get_camera_pitch()/get_camera_zoom_delta()`; `set_camera_yaw(float)` (external sync); `set_camera_forward(float x, float z)` (host feeds the *actual* camera forward each frame — accounts for shoulder offset — consumed by movement derivation); `move_forward()/move_backward()/move_left()/move_right()`; `get_input()` → `const InputState&`; `input_changed()` / `reset_changed()`; inversion (`is_camera_x/y_inverted`, `set_...`, `toggle_...`; default false); sensitivity get/set pairs.

### InputState (`src/bracken/input_state.hpp`)

```cpp
struct InputState {
    bool move_up = false, move_down = false, move_left = false, move_right = false; // legacy derived flags
    float move_dir_x = 0.0f, move_dir_y = 0.0f;  // continuous, camera-relative, |v| <= 1
    float attack_dir_x = 0.0f, attack_dir_y = 1.0f; // set to camera forward by InputHandler
};

inline float normalize_move_dir(float& x, float& y);
```

Action state (attacking, sprinting, abilities, ...) is the host's to own — `InputState` carries raw movement/aim data only.

`normalize_move_dir` clamps the vector's magnitude to 1.0 (dividing both components by the length when `len > 1.0f`) and returns the resulting magnitude; sub-unit magnitudes (analog partial tilt) pass through untouched, zero stays zero. Fully covered by `src/bracken/input_state.test.cpp`. No serialization or protocol knowledge in this struct by design.

### GraphicsSettings (`src/bracken/contracts/graphics_settings.hpp`, `.cpp`)

Plain aggregate + text-file persistence. Enum modes are `enum class : int` with **stable explicit underlying values** — they are the on-disk integers and the GPU-uniform integers; serialize with `static_cast<int>` only at those two boundaries.

```cpp
enum class ShadowMode  : int { Off = 0, PCF = 1, PCSS = 2 };
enum class AOMode      : int { Off = 0, GTAO_Half = 1, GTAO_Full = 2 };
enum class AAMode      : int { Off = 0, FXAA = 1, TAA = 2 };
enum class TonemapMode : int { ACESFitted = 0, ACESNarkowicz = 1, Reinhard = 2, None = 3 };
enum class RtResolution: int { Full = 0, Half = 1, Quarter = 2 };
enum class RtRange     : int { Near = 0, Medium = 1, Far = 2, Unlimited = 3 };
enum class GiMode      : int { Off = 0, Probes = 1, ProbesScreen = 2 };
enum class GiQuality   : int { Low = 0, Medium = 1, High = 2 };
```

#### Field table (name = file key; defaults verified by `graphics_settings.test.cpp::DefaultValues`/`LoadEmptyFileKeepsDefaults`)

| Field | Type | Default | Valid range (clamped by `clamp_to_valid_ranges`) |
|---|---|---|---|
| `settings_version` | int | `CURRENT_SETTINGS_VERSION` = **2** | — (always rewritten on load) |
| `fog_enabled` | bool | true | — |
| `grass_enabled` | bool | true | — |
| `skybox_enabled` | bool | true | — |
| `trees_enabled` | bool | true | — |
| `rocks_enabled` | bool | true | — |
| `show_fps` | bool | false | — |
| `show_debug_hud` | bool | false | — |
| `draw_distance` | int | 3 | [0, 6] → {500, 1000, 2000, 4000, 8000, 16000, 32000} |
| `frustum_culling` | bool | true | — |
| `anisotropic_filter` | int | 4 | [0, 4] (off/2x/4x/8x/16x) |
| `vsync_mode` | int | 2 | [0, 2]; re-clamped to `max_vsync_mode()` at apply |
| `shadow_mode` | ShadowMode | PCSS | [Off, PCSS] |
| `shadow_cascades` | int | 3 | [0, 3] (1–4 cascades) |
| `shadow_resolution` | int | 2 | [0, 3] (512/1024/2048/4096) |
| `shadow_distance` | float | 4000 | [200, 20000] (CSM range in world units; also clamped by draw distance at use) |
| `local_shadows` | bool | true | — (point/spot light shadows for `casts_shadows` lights; serialized 0/1, pinned by `LocalShadowsSerializeNumerically`) |
| `local_shadow_point_budget` | int | 4 | [0, 8] (shadowed point lights per frame; 6 atlas tiles each) |
| `local_shadow_spot_budget` | int | 2 | [0, 16] (shadowed spot lights per frame; 1 atlas tile each) |
| `local_shadow_resolution` | int | 1 | [0, 2] (1024/2048/4096 atlas → 128/256/512 px tiles) |
| `ao_mode` | AOMode | GTAO_Half | [Off, GTAO_Full]; numeric values redefined by the AO rearchitecture (1 was "SSAO", 2 the old fake "GTAO") — old files map 1 → GTAO_Half, 2 → GTAO_Full |
| `aa_mode` | AAMode | TAA | [Off, TAA]; TAA default applies to fresh settings only — files with a stored `aa_mode` keep it |
| `ibl_enabled` | bool | true | — (sky image-based lighting; OFF = hemisphere ambient fallback; serialized 0/1, pinned by `IblEnabledSerializesNumerically`) |
| `bloom_enabled` | bool | true | — |
| `bloom_strength` | float | 0.25 | **not clamped** (doc range 0.0–1.0) |
| `rt_shadows` | bool | **false** | — (compute-shader traced sun shadows; the mask REPLACES the CSM term where it applies. Off by default and pinned so by `DefaultValues`; round trip pinned by `RayTracingFlagsRoundTrip`) |
| `rt_reflections` | bool | **false** | — (traced specular; additionally needs per-triangle attributes, so enabling it after the structure was built for shadows alone forces a rebuild) |
| `rt_shadow_resolution` | RtResolution | Full | [Full, Quarter] (full/half/quarter screen res → the shadow ray count; the denoiser upsamples either way) |
| `rt_reflection_resolution` | RtResolution | Full | [Full, Quarter] (same dial for the reflection trace) |
| `rt_sun_softness_deg` | float | 0.35 | [0, 10] (angular RADIUS of the sun disk in degrees; the real sun is ~0.265. 0 collapses the cone to a hard shadow. Free to trace — same ray count, the denoiser works harder) |
| `rt_reflection_max_roughness` | float | 0.75 | [0, 1] (roughness at which traced specular hands over entirely to the prefiltered IBL probe; the two weights always sum to 1) |
| `rt_local_shadows` | LocalRtShadowMode | **Off** | [Off, Ratio, Full] (traced shadows for POINT and SPOT lights, `shaders/rt_local_shadow.comp.hlsl`. `Off` is the rasterized 8×8-tile atlas byte for byte. `Ratio` is one froxel-RIS reservoir shadow ray per pixel whose denoised R8 mask multiplies the whole clustered point+spot subtotal — the Heitz ratio estimator, so the analytic sum stays sharp. `Full` (R5c) spends the SAME one ray differently: the pass evaluates the winner's own BRDF into two demodulated RGBA16F radiance targets (diffuse without albedo, specular without Fresnel-F0), denoises them as radiance, and the forward shaders remodulate per pixel and skip the analytic point/spot loop ENTIRELY — per-light correct where Ratio is aggregate correct (coloured shadows from coloured lights; specular that pools under the light that made it). Both traced tiers **need the acceleration structure and the prepass G-buffer** and RETIRE the atlas for the frames they are on: no selector, no tile renders, every `shadow_slot` stays -1, and the smoke line's `local_tiles` reads 0. Since R7b `Full` also writes a THIRD target — the winner's octahedral light DIRECTION (RG16F) — from which the forward shaders build a bounded relief factor `clamp(max(dot(N_shading,L),0) / max(dot(N_geometric,L),0.05), 0, 4)` and recover the normal-map relief the tier used to lose (its named R5c cost). `Ratio` is still the recommendation below the Gold tier, but for COST now rather than quality. Clamping is pinned by `LocalRtShadowModeClampsOutOfRangeValues` + `LocalRtShadowFullTierRoundTrips`) |
| `rt_local_shadow_resolution` | RtResolution | **Half** | [Full, Quarter] (Half rather than Full because this signal picks a DIFFERENT light per pixel per frame, so the denoiser is working hard regardless and sharper input mostly buys detail it is about to filter away) |
| `rt_max_instance_range` | RtRange | **Unlimited** | [Near, Unlimited] → {1000, 2000, 4000, ∞} world units via `rt_range_units` (instances whose bounds lie entirely beyond it are dropped before the TLAS hashes; distant sun shadows fall back to the cascades. A finite value makes camera motion change the instance SET, which is a topology change and therefore a full TLAS rebuild rather than the refit a mere move costs — see architecture.md "Ray tracing") |
| `gi_mode` | GiMode | **Off** | [Off, ProbesScreen] (diffuse GI from the scrolled probe clipmap, `render/gi/probe_volume.hpp`. Off by default: the field costs ~30 MB of atlases the moment it exists. `ProbesScreen` adds the R4b **screen-probe gather** on top of the field — one 8x8 octahedral probe per 16x16 screen tile, screen-traced against the Hi-Z pyramid and BVH-traced for the rest, published as L1 SH the scene shaders blend over the field's answer. It also switches the depth prepass to its G-buffer variants (the probes' anchor normals) and allocates the Hi-Z pyramid; budget ~0.55 ms and ~20 MB at 1600x900 on top of `Probes`. After phase R1b the field traces, shades and accumulates probe rays but nothing samples it, so it still lights nothing; see architecture.md "Probe-field GI". **Turning it on builds the ray-tracing acceleration structure with per-triangle attributes even with `rt_shadows`/`rt_reflections` off** — probe rays shade what they hit) |
| `gi_quality` | GiQuality | Medium | [Low, High] (probe ray budget — `render/gi/probe_schedule.hpp::gi_budget`: rays/probe {32, 64, 96} and probes per cascade turn {864, 1728, 1728}, i.e. {1/8, 1/4, 1/4} of a cascade. With the rota putting at most two cascades in any one frame, the peak is {55296, 221184, 331776} rays/frame) |
| `volumetric_fog` | bool | false | — |
| `god_rays` | bool | false | requires `volumetric_fog` (not enforced here) |
| `window_mode` | int | 0 | [0, 2] |
| `resolution_index` | int | 0 | `>= 0` only (upper bound is runtime display-mode count; `RenderContext` re-clamps at apply) |
| `exposure` | float | 1.0 | not clamped; with `auto_exposure` on it is the manual bias multiplied on top of exp2(adapted EV) |
| `auto_exposure` | bool | true | — (histogram auto exposure; serialized 0/1, pinned by `AutoExposureSerializesNumerically`) |
| `auto_exposure_min_ev` | float | −4.0 | [−12, 12] |
| `auto_exposure_max_ev` | float | 4.0 | [`auto_exposure_min_ev`, 12] (kept ≥ min) |
| `auto_exposure_speed_up` | float | 1.5 | [0, 20] (EV/s toward brighter) |
| `auto_exposure_speed_down` | float | 2.5 | [0, 20] (EV/s toward darker) |
| `tonemap_mode` | TonemapMode | ACESFitted | [ACESFitted, None] |
| `contrast` | float | 1.0 | not clamped |
| `saturation` | float | 1.0 | not clamped |
| `ambient_strength` | float | 0.45 | not clamped |
| `sun_intensity` | float | 3.0 | not clamped |
| `bloom_threshold` | float | 1.1 | not clamped (thresholds **post-exposure** luminance) |
| `ao_strength` | float | 1.0 | not clamped |
| `ao_radius` | float | 1.0 | not clamped |
| `fog_density` | float | 0.05 | not clamped |

`get_draw_distance()` indexes a local `constexpr float distances[7]` with `std::clamp(draw_distance, 0, 6)` — safe against out-of-range direct assignment (public field).

#### save(path) → bool ([[nodiscard]])

Opens `std::ofstream`; `false` if it can't open. Writes every field as `key=value\n` (bools as 0/1, enums as int, floats via stream default formatting). Always writes `settings_version=2` first so `load()` can distinguish new files from pre-versioning ones. Returns `out.good()`.

#### load(path) → bool ([[nodiscard]])

- `false` only when the file can't be opened (fields untouched). An empty/garbage file that opens returns `true` with defaults intact.
- Parser: line-based; skips empty lines and `#`-prefixed comments; splits on the first `=`; keys are **not** whitespace-trimmed (a padded key won't match); values go through `std::stoi`/`std::stof` with catch-all → keep previous value on malformed input; bools are `value != "0"`; unknown keys ignored.
- **Migration**: reads `settings_version` (missing → 0). If `< CURRENT_SETTINGS_VERSION (2)`, the file predates the wave-3 lighting model; these fields are reset to fresh defaults: `sun_intensity`, `ambient_strength`, `exposure`, `bloom_strength`, `bloom_threshold`, `shadow_mode`, `shadow_cascades`, `ao_radius`, `vsync_mode`. All other loaded values survive. `settings_version` is then set to current regardless.
- Ends with `clamp_to_valid_ranges()` — the single trust-boundary normalization (int/enum fields, plus the four `auto_exposure_*` floats; all other floats pass through unclamped).

`clamp_to_valid_ranges()` is also public for hosts that mutate fields from untrusted input directly.

### RenderStats (`src/bracken/contracts/render_stats.hpp`)

```cpp
struct RenderStats {
    uint32_t draw_calls = 0;
    uint32_t triangle_count = 0;          // sum of index_count/3 (× instance count for instanced draws)
    uint32_t entities_rendered = 0;
    uint32_t entities_distance_culled = 0;
    uint32_t entities_frustum_culled = 0;
    uint32_t jobs_pending = 0;            // JobSystem::pending_count()
    uint32_t instances_drawn = 0;
    std::size_t packet_bytes_used = 0, packet_bytes_capacity = 0;
    uint32_t bone_palettes_used = 0, bone_palettes_budget = 0;
    uint32_t lights_submitted = 0, lights_binned = 0;
    uint32_t local_shadow_tiles_rendered = 0;
    uint32_t ui_vertices = 0, ui_draws = 0;
    std::vector<PassTiming> pass_times_ms;    // cpu_ms real, gpu_ms nullopt — see below
};
```

- Owned by `SceneRenderer` (`scene/scene_renderer.hpp`: `bracken::RenderStats render_stats_`); exposed via `Application::render_stats()` (`std::optional<RenderStats>`, `nullopt` while collection is off) and gated by `Application::set_collect_render_stats(bool)` → `SceneRenderer::set_collect_stats`. **Default off**; when off, none of the counters are incremented (every increment site across the `scene_renderer*.cpp` family checks `collect_stats_`).
- `FrameDiagnostics` is the exception to that gate: `SceneRenderer::frame_diagnostics()` is reset and accumulated every frame regardless, because a dropped draw is not opt-in instrumentation. It rides the same chunk-index-ordered `FrameStatsPartial` merge, so it needs no atomics and is bit-identical at any worker count.
- `pass_times_ms` is filled from `gpu_timer_pool_.read_results(...)`. `cpu_ms` is a real wall-clock measurement of the pass's command recording (`core::PassTimeline`, EMA alpha 0.2, 64 slots, slot averages discarded when a slot's pass name changes). `gpu_ms` stays `nullopt`: SDL3 GPU exposes no timestamp queries. Use RenderDoc/Nsight for GPU-side cost.
- When collecting, draw calls/triangles/entities are also emitted as Tracy plots (`render.draw_calls`, `render.triangles`, `render.entities`).

### Error handling summary

This layer predates the `core::Result` convention and uses `bool` + `BRACKEN_LOG_*` throughout (boot-time init chains where failure is fatal and the log is the diagnostic — sanctioned by `CLAUDE.md`): `init_engine`, `init_renderer`, `GraphicsSettings::save/load` are `[[nodiscard]] bool`; `init_dev_ui` returns plain `bool` (false = stub build or backend failure, both non-fatal); `set_vsync_mode` failures are log-only; settings save failure on shutdown is log-only.

### Performance characteristics

- The loop is single-threaded; per-frame overhead beyond the host hooks is event polling, a keyboard-state poll, and a handful of float ops. Profiling zones: `App::poll_events`, `App::update`, `App::render`, plus the Tracy frame mark.
- `apply_all_graphics_settings` makes repeated application cheap (diffed expensive setters); `set_window_mode` and `set_vsync_mode` additionally self-no-op when unchanged. `GraphicsSettingsEdit` adds an earlier gate: an edit whose copy still compares equal (`GraphicsSettings::operator==`, defaulted, exact float compare) requests nothing at all.
- `request_screenshot` stalls the pipeline for one frame (submit + fence wait) by design.
- `available_resolutions()` copies the mode list per call.

### Known limitations / stubs / silent behaviors

- **No fixed timestep**: variable `dt` clamped at 0.1 s; long stalls silently compress simulated time.
- `fps()` is a once-per-second integer frame count, `0.0f` for the first second.
- `on_init`/`on_shutdown` are reached only through `start()`/`stop()` — a host that forgets to call `start()` gets a null-deref in the first facade call, not a diagnostic.
- Facade methods are guarded by `BRACKEN_CONTRACT` (live in every build, no switch): calling one before `init_renderer()` aborts with the member's name instead of segfaulting.
- `PassTiming::gpu_ms` is always `nullopt` (SDL3 has no timestamp queries); `cpu_ms` measures command recording, not GPU execution.
- Exclusive fullscreen with an unmatched `resolution_index` warns and falls back to borderless fullscreen (stale exclusive modes are cleared).
- `set_vsync_mode` silently clamps to the supported maximum.
- Held-key state (`InputSnapshot::keys_held` / `key_held(sc)`) is polled from `SDL_GetKeyboardState` at snapshot time and reads all-false while `set_keyboard_captured(true)` is in effect.
- `set_keyboard_captured` is overwritten every frame by `Application::run`.
- `GraphicsSettings::load` does not trim key whitespace; float fields are never clamped; booleans treat any non-`"0"` string as true.
- `god_rays` requiring `volumetric_fog` is enforced downstream in the renderer (god rays are also force-disabled when shadows are off), not by `GraphicsSettings`.

---

## Core Utilities: Logging, Errors, Handles, Hashing, Profiling

Scope: the top-level files of `src/bracken/core/` — `logger.{hpp,cpp}`, `assert.{hpp,cpp}`, `result.hpp`, `handle.hpp`, `hash.hpp`, `profiler.hpp`, their tests (`*.test.cpp`) and the shared benchmark file `core.bench.cpp`. (`core/jobs/` and `core/asset/` are specified in their own chapters.) Namespace: `bracken::core`; implementation helpers in `bracken::core::detail`. No GPU state is touched by any of these facilities; the only external dependency is SDL's logging API (`SDL3/SDL_log.h`) inside `StdoutSink`, and optionally Tracy.

### Logging (`src/bracken/core/logger.hpp`, `logger.cpp`)

#### Purpose

Process-wide, thread-safe, level-filtered, multi-sink logging with `std::format`-checked message formatting, consumed via `BRACKEN_LOG_*` macros.

#### Public API surface

```cpp
namespace bracken::core {

enum class LogLevel : uint8_t { Trace=0, Debug=1, Info=2, Warn=3, Error=4, Fatal=5, Off=6 };
const char* to_string(LogLevel level) noexcept;   // "TRACE".."FATAL", "OFF", "?" fallback

class LogSink {                                   // abstract; non-copyable, non-movable
public:
    virtual ~LogSink() = default;
    virtual void write(LogLevel level, const char* category, std::string_view msg) = 0;
};

class StdoutSink final : public LogSink {         // default sink; forwards to SDL logging
    void write(LogLevel, const char*, std::string_view) override;
};

class Logger {
public:
    using SinkHandle = uint32_t;
    static Logger& instance();                    // Meyers singleton (function-local static)

    void set_min_level(LogLevel) noexcept;
    LogLevel min_level() const noexcept;
    SinkHandle add_sink(std::shared_ptr<LogSink>);
    void remove_sink(SinkHandle);
    void clear_sinks();
    bool should_log(LogLevel) const noexcept;
    void dispatch(LogLevel, const char* category, std::string_view msg);
};

namespace detail {
template<typename... Args>
void log_formatted(LogLevel, const char* category, std::format_string<Args...>, Args&&...);
void log_plain(LogLevel, const char* category, std::string_view msg);
}
} // namespace bracken::core
```

Macros (variadic, category first, then a `std::format` string + args): `BRACKEN_LOG_TRACE/DEBUG/INFO/WARN/ERROR/FATAL(category, ...)`. Each expands to `::bracken::core::detail::log_formatted(LogLevel::X, (category), __VA_ARGS__)`. Format strings are compile-time checked (`std::format_string<Args...>`). `BRACKEN_LOG_FATAL` does not abort.

Rate-limited variants (`bracken/core/log_throttle.hpp`, for call sites on a per-frame path): `BRACKEN_LOG_WARN_ONCE` / `BRACKEN_LOG_ERROR_ONCE` (first occurrence only) and `BRACKEN_LOG_WARN_THROTTLED(interval_s, ...)` / `BRACKEN_LOG_ERROR_THROTTLED` (at most one per interval). Each declares a function-local `static core::LogThrottle`; the type has a `constexpr` constructor so the static is constant-initialized (no guard variable), and every atomic inside is `relaxed` — the state is a self-contained bool/timestamp with no payload riding on it. `LogThrottle::suppressed()` reports how many occurrences the gate swallowed.

#### Data structures and state

`Logger` holds `mutable std::mutex mutex_`, `std::vector<Entry> sinks_` where `Entry { SinkHandle id; std::shared_ptr<LogSink> sink; }`, `SinkHandle next_id_ = 1`, and `std::atomic<LogLevel> min_level_{LogLevel::Info}`. The mutex guards `sinks_`/`next_id_` **only**; `min_level_` is deliberately outside it (protocol documented at the declaration): relaxed store in `set_min_level`, relaxed load in `min_level`/`should_log`. The value is self-contained — nothing else is published with it — so a racing level change costs at most one extra or one missing line around the switch, which is the semantics a level knob has anyway. The constructor (private) opens SDL's per-category filter fully — `SDL_SetLogPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_TRACE)` — so `set_min_level()` is the single level knob, then installs one `StdoutSink` with id 1; the first user `add_sink` therefore returns 2. Sink ids are never reused (monotonic counter; no overflow handling — `uint32_t`).

#### Behavior

- **Filtering**: `should_log(level)` returns `level >= min && min != LogLevel::Off` from a **single** relaxed load of `min_level_` (so the two comparisons cannot disagree about which level they are testing) — lock-free, and callable from every worker without serializing on the global mutex. Default minimum is `Info`. `detail::log_formatted` checks `should_log` *before* calling `std::format`, so formatting cost is skipped for filtered messages; macro argument expressions are still evaluated by the caller.
- **Dispatch**: `dispatch()` snapshots the sink `shared_ptr`s into a local vector under the lock, then calls `write()` on each **outside** the lock (explicit comment: avoids deadlock if a sink logs re-entrantly). A `nullptr` category is replaced with `"general"` at dispatch time.
- **`StdoutSink::write`**: builds `"[category] msg"` (no prefix when category is null/empty) and calls `SDL_LogMessage(SDL_LOG_CATEGORY_APPLICATION, prio, "%s", body)`. Level→SDL priority map: Trace→`SDL_LOG_PRIORITY_TRACE`, Debug→`DEBUG`, Info→`INFO`, Warn→`WARN`, Error→`ERROR`, Fatal→`CRITICAL`, default→`INFO`. SDL3's own per-category priority threshold (default Info for the application category) is opened to `TRACE` in the `Logger` constructor, so engine-level Trace/Debug that pass `should_log` reach stdout.

#### Threading and lifetime contracts

- Sink-list transitions (`add_sink`/`remove_sink`/`clear_sinks`/`dispatch`'s snapshot) are mutex-guarded; the level knob is atomic and lock-free. `docs/architecture.md` states "Logger sinks are thread-safe" (meaning the logger machinery — concurrent logging from 8 threads is tested in `logger.test.cpp` `ThreadSafeConcurrentLogging`). `should_log`/`set_min_level`/`min_level` are `noexcept` and now genuinely cannot throw: they no longer take a lock inside a `noexcept` function.
- Because sink writes happen outside the lock, a sink's `write()` may run concurrently on multiple threads; sink implementations must be internally synchronized (the test `CapturingSink` carries its own mutex).
- `remove_sink`/`clear_sinks` do not synchronize with in-flight dispatches: a concurrently running `dispatch` holds `shared_ptr` snapshots, so a removed sink can receive at most the in-flight batch after removal returns, and is destroyed only when the last snapshot drops.
- Singleton lifetime: function-local static, constructed on first use, destroyed at static destruction — logging from other static destructors after that point is undefined.

#### Performance characteristics

Every non-filtered log takes: one relaxed atomic load for `should_log`, one `std::format` heap allocation, one mutex lock + snapshot vector allocation in `dispatch`, then per-sink virtual calls. Filtered logs cost one relaxed atomic load and nothing else. There is no async/buffered path and no category-based filtering (level only, global).

#### Known limitations

- No timestamp, thread-id, or source-location decoration is added by the core logger (SDL may add its own prefix on the default sink).
- No file sink is provided in-tree; `StdoutSink` is the only concrete sink.
- Level is global — per-category verbosity is not supported.

#### Discipline: one log channel

`Logger` is the engine's **only** log channel. Engine sources under `src/bracken/` may not call `SDL_Log`, `std::cerr` or `std::cout`; the sole exemptions are `core/logger.cpp` (the bridge itself, via `SDL_LogMessage`) and `core/assert.hpp` (the abort path, which must print when the logger may itself be the failure). `src/bracken/core/logger_discipline.test.cpp` scans the tree and fails on any reintroduction; `tools/logdiff.sh [<rev>]` diffs the normalized set of message bodies against a git revision, which is what makes a log-severity/tag change reviewable without hiding an accidentally dropped message.

Each engine `.cpp` names its category once as a file-local `constexpr const char* LOG_TAG` in an anonymous namespace (pattern: `render/water_renderer.cpp`), so a subsystem's messages are filterable as a group by a host sink.

### Assertions and main-thread checks (`src/bracken/core/assert.hpp`, `assert.cpp`)

#### Purpose

Fatal contract checks that log through the `Logger` (category `"assert"`, level `Fatal`) and `std::abort()`, plus a debug-only main-thread ownership check backing the engine's documented threading contracts.

#### Public API surface

```cpp
namespace bracken::core {
void bind_main_thread() noexcept;   // record calling thread as the main thread
bool is_main_thread() noexcept;     // true if caller is the recorded main thread;
                                    // first caller self-captures if never bound
}
```

Macros:

- `BRACKEN_VERIFY(cond, ...)` — always active (debug **and** release). Evaluates `cond` exactly once; on false (`[[unlikely]]`) formats the optional `std::format` message via `detail::abort_format` (`__VA_OPT__` selects it) or falls through to `detail::abort_plain`, then aborts. Wrapped in `do { } while (0)`.
- `BRACKEN_ASSERT(cond, ...)` — `#ifdef NDEBUG` → `((void)sizeof(!(cond)))` (condition **type-checked but not evaluated** — a stale symbol still fails to compile, but side effects never run); otherwise identical to `BRACKEN_VERIFY`.
- `BRACKEN_ASSERT_MAIN_THREAD()` — `NDEBUG` → `((void)0)`; else `BRACKEN_VERIFY(is_main_thread(), "must be called from the main thread")`.
- `BRACKEN_UNREACHABLE(...)` — always active; unconditionally logs (expr string `"unreachable"`, optional formatted message) and aborts.

Failure path (`detail::abort_with`, header-inline, `[[noreturn]]`): message is `std::format("{}:{}: assertion `{}` failed: {}", file, line, expr ? expr : "<unreachable>", msg)`, emitted via `Logger::instance().dispatch(level, "assert", full)` — **bypassing the min-level filter**, so `set_min_level(Off)` can never silence an aborting contract violation — and additionally printed to `stderr` via `std::fprintf` as a last-resort channel (sinks may have been cleared). With the default stdout sink attached the message can therefore appear twice. Level passed is always `LogLevel::Fatal` from the macros. Then `std::abort()` (no stack unwinding, no exception).

#### Main-thread tracking algorithm (`assert.cpp`)

State: one function-local `static std::atomic<std::thread::id>`; a default-constructed `std::thread::id` means "not yet bound".

- `bind_main_thread()` — unconditional relaxed `store` of the caller's id. Called by `Application::init_engine` (`src/bracken/application.cpp:34`) and `core::asset::FileWatcher::init` (`src/bracken/core/asset/file_watcher.cpp:37`). Rebinding from another thread moves "main" there (tested: `assert.test.cpp` `RebindMovesMainThread`).
- `is_main_thread()` — `compare_exchange_strong(expected=unset, self, relaxed)`: the first-ever caller wins the capture and returns true; afterwards the CAS fails and `expected` carries the bound id for comparison. A failed query from a non-main thread does not rebind (tested).
- All operations use `std::memory_order_relaxed`; only identity of the id matters, no data is published through it.

Guarded boundaries in-tree: `FileWatcher` entry points (`file_watcher.cpp:51,76,95,114,124`), swapchain acquire (`gpu/gpu_device.cpp:125`), `SceneRenderer::render_frame` (`scene/scene_renderer.cpp:482`). Per `docs/architecture.md` these contracts are "asserted, not just documented" — in debug builds only.

#### Limitations

- `BRACKEN_ASSERT`'s condition disappears under `NDEBUG` — side-effecting conditions are a latent release-mode behavior change (the condition is still type-checked but unevaluated).
- Abort messages dispatch to sinks and print to `stderr`; sinks that buffer without flushing can still lose the sink-side copy (the default `StdoutSink` writes synchronously through SDL).

### `core::Result<T, E>` (`src/bracken/core/result.hpp`)

#### Purpose

Exception-free ok-or-error return type for fallible APIs, with a `void`-payload specialization. Depends only on `assert.hpp` (for the misuse guards) and the standard library.

#### Public API surface

```cpp
namespace bracken::core {

template<typename E> struct [[nodiscard]] Err { E value; };   // + deduction guide Err(E) -> Err<E>

template<typename T, typename E = std::string> class [[nodiscard]] Result {
public:
    using value_type = T; using error_type = E;
    Result(const T&); Result(T&&);                       // implicit ok construction
    Result(const Err<E>&); Result(Err<E>&&);             // error construction
    template<typename U> Result(Err<U>&&) requires(std::is_convertible_v<U, E>);
    bool is_ok() const noexcept; bool is_err() const noexcept;
    explicit operator bool() const noexcept;             // == is_ok()
    T& value() &; const T& value() const&; T&& value() &&;          // BRACKEN_VERIFY(is_ok())
    E& error() &; const E& error() const&; E&& error() &&;          // BRACKEN_VERIFY(is_err())
    template<typename U> T value_or(U&& fallback) const&;           // and && overload
    template<typename Fn> auto map(Fn&&) const& -> Result<std::invoke_result_t<Fn, const T&>, E>;  // and &&
    template<typename Fn> auto and_then(Fn&&) const& -> std::invoke_result_t<Fn, const T&>;        // and &&
};

template<typename E> class [[nodiscard]] Result<void, E> {   // no value()/value_or()/map()
    Result();                                            // default-constructed == ok
    // Err constructors, is_ok/is_err/operator bool, error() accessors,
    // and_then(Fn) with zero-argument Fn — same semantics as the primary template
};

template<typename T> Result<std::decay_t<T>> Ok(T&& v);  // note: E fixed to default std::string
Result<void> Ok();
template<typename E> Err<std::decay_t<E>> make_err(E&& e);
}
```

#### Storage and layout

Primary template: `std::variant<T, E>` with index 0 = ok, index 1 = error; all construction uses `std::in_place_index`, so `T == E` is well-formed. `Result<void, E>` stores `std::variant<std::monostate, E>`. Size/alignment are those of the variant (max of `T`/`E` plus discriminant). No heap allocation beyond what `T`/`E` themselves do. There is no empty/valueless state in normal use.

#### Behavior details

- **Misuse is fatal, not UB by default**: `value()` on error and `error()` on ok trip `BRACKEN_VERIFY` (message `"Result::value() called on error"` / `"Result::error() called on ok"`) — active in release builds; the process aborts.
- `[[nodiscard]]` is on the class templates and on `Err`, so discarding any returned `Result` warns.
- `map(fn)`: applies `fn` to the ok value, producing `Result<U, E>`; on error, copies (lvalue overload) or moves (rvalue overload) the error through.
- `and_then(fn)`: `fn` must itself return a `Result<..., E>`-shaped type (its return type is used verbatim); short-circuits on error by constructing `R(Err<E>{...})`. The `Result<void,E>` overload invokes `fn()` with no arguments.
- `value_or(fallback)`: `static_cast<T>`s the fallback; rvalue overload moves the stored value out.
- Error-type conversion: the constructor `Result(Err<U>&&)` requires `std::is_convertible_v<U, E>`. `make_err` returns `Err<std::decay_t<E>>`, so `make_err("literal")` yields `Err<const char*>`, which converts into a `Result<..., std::string>` through that constructor (tested in `result.test.cpp` `ErrConvertibleErrorType`).
- No `map_error`, no `operator*`/`operator->`, no reference payloads, no `expected`-style monadic `or_else`.

#### Project convention (from `CLAUDE.md`)

New fallible engine APIs return `core::Result<T, E>` (or `Result<void, E>`) **or** `bool`, `[[nodiscard]]` either way. `Result` when the error payload is actionable data (parse/compile errors at the asset/IO boundary — hot-reload wants to display them); `bool` + `BRACKEN_LOG_*` for boot-time `init()` chains where failure is fatal. Existing `bool`+log APIs are not to be mass-converted. Note a mild tension: the header comment on `Result<void, E>` (`result.hpp:116`) calls it "the shape every `bool init()/load()/save()`-style API **needs to migrate to**", while `CLAUDE.md` says existing `bool` APIs "stay as they are until touched". Current state: no engine module outside `core/` returns `Result` yet; coverage is `result.test.cpp` only.

### Generational handles (`src/bracken/core/handle.hpp`)

#### Purpose

Type-tagged, generation-checked resource handles plus a validating slot-pool container. In-tree consumer: `core::asset::AssetRegistry` (`src/bracken/core/asset/asset_registry.hpp`) uses `bracken::core::Handle<T>` directly (with its own slot bookkeeping, not `HandlePool`).

#### `Handle<Tag>`

```cpp
template<typename Tag> struct Handle {
    uint32_t index = k_invalid_index;
    uint32_t generation = 0;
    static constexpr uint32_t k_invalid_index = static_cast<uint32_t>(-1);  // 0xFFFFFFFF
    static constexpr Handle invalid();                    // Handle{}
    [[nodiscard]] bool is_valid() const noexcept;         // index != k_invalid_index
    // friend operator==, operator!= (compare index AND generation)
};
```

Layout: two `uint32_t`, 8 bytes, trivially copyable, aggregate-initializable (`Handle<MeshTag>{1, 2}`). `Tag` is a phantom type — `Handle<MeshTag>` and `Handle<TexTag>` are unrelated types (static_assert-tested in `handle.test.cpp`).

`std::hash<Handle<Tag>>` specialization: `static_cast<size_t>((uint64_t(generation) << 32) | index)` — the shift happens in `uint64_t`, so it is well-defined on any `size_t` width.

**Default-constructed `Handle{}` is invalid**: `Handle{} == Handle<Tag>::invalid()` (index `k_invalid_index`), matching `render_graph::ResourceHandle`'s convention, so `is_valid()` doubles as a "was this assigned" test.

#### `HandlePool<Tag, T>`

Slot-map with an intrusive singly linked free list:

```cpp
template<typename Tag, typename T> class HandlePool {
    using HandleT = Handle<Tag>;
    [[nodiscard]] HandleT acquire(T value);
    [[nodiscard]] bool release(HandleT h);
    T* get(HandleT h);  const T* get(HandleT h) const;   // nullptr on any validation failure
    bool contains(HandleT h) const;                       // get(h) != nullptr
    size_t size() const noexcept;                         // live count — O(free-list length)!
    size_t capacity() const noexcept;                     // slots_.size()
    void clear();                                         // releases all live slots, bumping generations
};
```

Internal layout: `std::vector<Slot> slots_` with `Slot { T value{}; uint32_t generation = 0; uint32_t next_free = k_occupied; }`, plus `uint32_t free_head_` (initially `k_invalid_index`). Sentinel `k_occupied = static_cast<uint32_t>(-2)` (0xFFFFFFFE) marks a live slot; any other `next_free` value is a free-list link (`k_invalid_index` terminates the list).

Algorithms:

- `acquire(T value)`: pop the free-list head if non-empty (O(1)), keeping the slot's current generation; otherwise append a new slot with `generation = 1` (amortized O(1), vector growth). Returns `{index, generation}`. Value is move-assigned into the slot.
- `release(h)`: validates (`is_valid()`, `index < slots_.size()`, slot occupied, generation match) and returns `false` on any failure — double-release is a `false`, not a fault (tested). On success: `slot.value = T{}` (resource destroyed immediately), `generation += 1`, slot pushed on the free list. O(1).
- `get(h)`: same four-way validation; returns `&slot.value` or `nullptr`. Stale handles (slot reused → generation bumped) return `nullptr` (tested: `GenerationalReuseDetectsStale`).
- `size()`: computed by walking the free list and subtracting — **O(number of free slots)**, not cached.
- `clear()`: releases every live slot in place — `slot.value = T{}`, `generation += 1`, slot pushed on the free list. Generations are preserved (and bumped), so handles retained across a `clear()` fail validation instead of aliasing post-clear acquisitions. Slot storage is kept: `capacity()` is unchanged.

Type requirements on `T`: movable (acquire) and default-constructible + assignable (release's `T{}` reset). Threading: no internal synchronization; single-threaded or externally locked use only. Generation counters can wrap after 2^32 release cycles of one slot (theoretical ABA).

### Hashing and `StringId` (`src/bracken/core/hash.hpp`)

#### Purpose

Constexpr FNV-1a string hashing (32- and 64-bit) and a 4-byte hashed-string identifier for name-keyed lookups.

#### Constants (exact values)

```cpp
constexpr uint32_t k_fnv1a_offset_basis_32 = 0x811c9dc5u;
constexpr uint32_t k_fnv1a_prime_32        = 0x01000193u;
constexpr uint64_t k_fnv1a_offset_basis_64 = 0xcbf29ce484222325ull;
constexpr uint64_t k_fnv1a_prime_64        = 0x100000001b3ull;
```

#### Functions and types

```cpp
constexpr uint32_t fnv1a(std::string_view s) noexcept;     // FNV-1a, byte-wise: h ^= byte; h *= prime
constexpr uint64_t fnv1a64(std::string_view s) noexcept;

struct StringId {
    uint32_t value = 0;
    constexpr StringId() = default;
    constexpr explicit StringId(uint32_t v) noexcept;
    constexpr StringId(std::string_view s) noexcept;       // implicit; value = fnv1a(s)
    // constexpr ==, !=, < (by value)
};
constexpr StringId operator""_sid(const char* s, std::size_t n) noexcept;  // in bracken::core
// std::hash<StringId> = identity on value
```

Algorithm: canonical FNV-1a — start from the offset basis, per byte XOR then multiply by the prime; bytes are read as `unsigned char` (sign-safe). Verified against the canonical test vectors in `hash.test.cpp`: `fnv1a("") == 0x811c9dc5`, `fnv1a("a") == 0xe40c292c`, `fnv1a("foobar") == 0xbf9cf968`, `fnv1a64("a") == 0xaf63dc4c8601ec8c`, `fnv1a64("foobar") == 0x85944171f73967e8`. Fully `constexpr` (static_assert-tested), O(n) per hash, no allocation.

Properties/limits: `StringId` equality is hash equality — 32 bits, no collision detection, no stored source string (no reverse lookup). The implicit `string_view` constructor makes any string literal convertible to `StringId` at call sites. `std::hash<StringId>` passes the FNV value through unchanged (`static_cast<size_t>(value)`).

Current consumers: none outside `core/` — `hash.hpp` is exercised by `hash.test.cpp` and benchmarked in `core.bench.cpp` (`BM_Fnv1a`/`BM_Fnv1a64`, 8/64/512-byte inputs, bytes-processed counters). `docs/architecture.md` lists it as core infrastructure; name→handle maps elsewhere in the engine currently use `std::unordered_map<std::string, ...>`.

### Tracy profiler hooks (`src/bracken/core/profiler.hpp`)

#### Purpose

Compile-time-switched wrappers over the Tracy client so instrumentation can stay in the code permanently at zero cost when disabled.

#### Configuration

- CMake option `BRACKEN_ENABLE_TRACY` (top-level `CMakeLists.txt:14`), **default OFF**. When ON: FetchContent of Tracy `v0.11.1` with cache-forced `TRACY_ENABLE=ON` and `TRACY_ON_DEMAND=ON` (capture only while a profiler GUI is connected).
- `src/bracken/CMakeLists.txt:99-101`: `target_link_libraries(bracken PUBLIC TracyClient)` and `target_compile_definitions(bracken PUBLIC BRACKEN_ENABLE_TRACY=1)` — both PUBLIC, so consumers of the `bracken` target inherit the define and their own `BRACKEN_PROFILE_*` macros are live without extra setup.
- The header gates on `#if defined(BRACKEN_ENABLE_TRACY) && BRACKEN_ENABLE_TRACY`; when disabled every macro is `((void)0)` and `<tracy/Tracy.hpp>` is not included.

#### Macro → Tracy mapping (enabled build)

| Macro | Expansion | Notes |
|---|---|---|
| `BRACKEN_PROFILE_FRAME()` | `FrameMark` | called once per frame by `Application::run` (`src/bracken/application.cpp:118`) |
| `BRACKEN_PROFILE_FRAME_NAMED(n)` | `FrameMarkNamed(n)` | string literal |
| `BRACKEN_PROFILE_ZONE(name)` | `ZoneScopedN(name)` | literal name; scoped to enclosing block |
| `BRACKEN_PROFILE_ZONE_COLOR(name, color)` | `ZoneScopedNC(name, color)` | color is `0xRRGGBB` |
| `BRACKEN_PROFILE_ZONE_TAG(t)` | `ZoneScoped; ZoneText((t), std::strlen(t))` | **two statements**, not do-while-wrapped — unsafe as an unbraced `if` body; tag is a runtime string |
| `BRACKEN_PROFILE_PLOT(n, v)` | `TracyPlot(n, v)` | engine call sites cast counts to `int64_t` |
| `BRACKEN_PROFILE_MESSAGE(m)` | `TracyMessage(m, std::strlen(m))` | |
| `BRACKEN_PROFILE_ALLOC(ptr, sz)` / `BRACKEN_PROFILE_FREE(ptr)` | `TracyAlloc` / `TracyFree` | paired; used by `src/bracken/memory/arena.cpp:109,129,165` |
| `BRACKEN_PROFILE_THREAD(name)` | `tracy::SetThreadName(name)` | used by `core/jobs/job_system.cpp:175` ("JobWorker") and `core/asset/file_watcher.cpp:226` ("FileWatcher") |

Suggested zone-color palette (header comment; used consistently at engine call sites): `0x3D6E9E` update, `0x8E44AD` animation, `0x27AE60` culling/scene, `0xE67E22` shadows, `0xC0392B` main pass, `0x16A085` post-processing. Representative call sites: `animation/animation_state_machine.cpp:194` (0x8E44AD), `render/shadow_map.cpp:184` (0xE67E22), `render/terrain_renderer.cpp:457` (0xC0392B), `render/post_process_stack.cpp` composite (0x16A085), `effects/effect_system.cpp:62` (`BRACKEN_PROFILE_PLOT("effects.particles", ...)`).

#### Known limitations (documented in the header and `docs/architecture.md`)

- **No GPU zones**: SDL3 GPU does not expose backend queues/devices, so Tracy's GPU instrumentation (`TracyVkZone` etc.) cannot be wired up. GPU timing must come from RenderDoc/Nsight. `GPUTimestampPool` reports real CPU record times per pass (`PassTiming::cpu_ms`) but no GPU times (`gpu_ms` is `nullopt`).
- Zone names must be string literals (Tracy `ZoneScopedN` requirement); dynamic names go through `BRACKEN_PROFILE_ZONE_TAG`'s `ZoneText` instead.
- There is no zero-overhead runtime toggle: enablement is a compile-time CMake decision (though `TRACY_ON_DEMAND` keeps the enabled build cheap while disconnected).

### Benchmarks (`src/bracken/core/core.bench.cpp`)

Google Benchmark file (built into `bracken_benchmarks` when `BRACKEN_BUILD_BENCHMARKS=ON`, the default) covering this chapter's hashing plus neighboring modules: `BM_Fnv1a` / `BM_Fnv1a64` (string lengths 8/64/512, `DoNotOptimize` on the input to defeat constexpr folding, reports bytes/s), `BM_ArenaAllocate`, `BM_FrameAllocatorAllocSwap`, and `BM_SerialSum` vs `BM_ParallelForSum`. Run: `build/src/bracken/bracken_benchmarks --benchmark_filter=Fnv1a`.

---

## Job System

### Overview and design posture

`src/bracken/core/jobs/` implements the engine's **always-on** lock-free work-stealing thread pool (Phase 1 of `docs/multithreading_plan.md` — the plan of record; the section below describes the Phase-1 specification), plus a fork-join group primitive and a blocking data-parallel loop helper. Namespace: `bracken::core::jobs`. Files:

- `src/bracken/core/jobs/job_system.hpp` / `job_system.cpp` — `JobSystem` and its internals (per-worker deques, injection queue, job-node pools), `TaskGroup`
- `src/bracken/core/jobs/task_handle.hpp` — `TaskHandle`
- `src/bracken/core/jobs/parallel_for.hpp` — `parallel_for` (singleton and explicit-pool overloads), `PARALLEL_MIN_TOTAL`
- Tests: `job_system.test.cpp`, `parallel_for.test.cpp` (same directory) — including the race-shaped stress tests each lock-free primitive lands with — plus `src/bracken/animation/animation_parallel.test.cpp` (serial-vs-parallel equivalence for `AnimationPlayer::update`)
- Benchmarks: `BM_SerialSum` / `BM_ParallelForSum` in `src/bracken/core/core.bench.cpp`

**Contract with the rest of the engine (always-on infrastructure):** `Application::init_engine()` (`src/bracken/application.cpp`) calls `JobSystem::instance().init(workers)` unconditionally, **before `SDL_Init`**, where `workers` comes from the `BRACKEN_WORKERS` env var (unset/`0` → pool default `hardware_concurrency() - 1` min 1; positive value used as-is — `=1` is the single-worker debug mode, same code path; negative/unparsable → WARN + default). Teardown is first-in: `Application::shutdown_renderer()` calls `JobSystem::instance().shutdown()` at its top, before any GPU teardown, so no in-flight or queued job can touch a subsystem being destroyed; `shutdown_engine()` repeats the (idempotent) call as a safety net for hosts that never brought up a renderer. `SceneRenderer::render_frame` snapshots `pending_count()` into `RenderStats::jobs_pending` (`src/bracken/contracts/render_stats.hpp`) when stats collection is on. Rationale and phase roadmap: `docs/multithreading_plan.md`; the superseding decision record is in `docs/engine_design_notes.md` (2026-07-18). `docs/architecture.md` (Threading & lifetime contracts) carries the thread-safety tier table.

**Determinism contract:** one code path at every worker count — `BRACKEN_WORKERS=1` and `=N` runs must be bit-identical; merges of parallel partials are ordered by chunk index, never thread id.

**Parallel stages & the `JobSystem&` convention (Phase 3):** the renderer's Prepare stage is internally parallel — per-view culling + instance batching (main view + active CSM cascades; the local-shadow tile-caster cull stays serial inside `render_local_shadow_passes`), skinned-command resolve + palette layout, cluster light binning, grass chunk cull, and terrain tile cull all fan out as `parallel_for`/`TaskGroup` chunks over the sealed `FramePacket`. (Particle work moved to the **sim stage** in Phase 4: spawns applied serially in deterministic lane order — order-defining for the RNG stream — then `EffectSystem::update` integrates as a per-particle `parallel_for` from inside the sim job.) Each chunk accumulates into its own partial (bins, stats via `FrameStatsPartial`, scratch); partials merge single-threaded in chunk-index order after the join. No atomics or shared cache-line writes inside per-item loops, no mutexes in the frame path; grain guards keep small scenes to a single inline chunk. **Engine-internal stage helpers take a `JobSystem&` parameter** — production call sites pass `JobSystem::instance()`; per-stage parity tests construct caller-owned pools (`init(1)` vs `init(4)`) and assert bit-identical outputs.

### Public API surface

#### `TaskHandle` (`src/bracken/core/jobs/task_handle.hpp`)

Phase 1 replaces the old heap-allocated `shared_ptr<Task>` handle with a **pooled-node + generation** design: jobs live in fixed-size nodes in per-worker pools, recycled via freelist; a `TaskHandle` refers to a node by (node, generation) — no allocation per job, no refcount churn, ABA-safe queries.

- `TaskHandle` is a cheap, copyable value. `valid()` — refers to a task; `is_done()` — the task finished (successfully or with an exception); `has_error()` — finished **and** threw; `error()` — the stored `std::exception_ptr` (null until done). A handle whose node has been recycled to a newer generation reports the *old* job as done — the generation check makes stale queries safe, never a read of someone else's job state.
- Publication contract (house style — kept from the original design): the executing worker writes the non-atomic error slot **before** the release-store of the done flag; every reader observes done with acquire before trusting the payload. `error()` returns `nullptr` while the task is in flight — no racy pre-done read of the exception slot.
- There is no result channel, no cancellation, no continuation/dependency API — a task is a callable plus a done flag and an exception slot.

#### `TaskGroup`

The fork-join counter for batches of jobs (plan §7): a single cache-line-padded atomic (padded to the destructive interference size — 128 B on Apple Silicon; guarded fallback constant where `std::hardware_destructive_interference_size` is unavailable). Protocol: `fetch_add` on submit-against-group; release `fetch_sub` on job completion; the decrement that observes zero performs an acquire fence and notifies; waiters park on `std::atomic::wait`. Beyond the counter it carries one CAS-guarded error slot: the **first** exception thrown by a member job is captured; `wait` rethrows it only after the whole group has joined (the join is unconditional), and when several threads wait concurrently exactly one rethrows — consuming the error clears it, keeping the group reusable. No aggregation of multiple errors, no nesting semantics beyond the count. `JobSystem::submit(TaskGroup&, fn)` and `wait(TaskGroup&)` are the group API; groups may be carried across frames by a host and must be joined before their results are read.

#### `JobSystem` (`src/bracken/core/jobs/job_system.hpp`)

```cpp
class JobSystem {
public:
    static JobSystem& instance();          // Meyers singleton (function-local static)

    JobSystem() = default;                 // publicly constructible (caller-owned pools, tests)
    ~JobSystem();                          // calls shutdown()
    // non-copyable, non-movable (all four special members deleted)

    void init(unsigned thread_count = 0);  // idempotent; 0 = hardware_concurrency()-1, min 1
    void shutdown();

    bool is_initialized() const noexcept;
    unsigned worker_count() const noexcept;
    unsigned pending_count() const noexcept;

    TaskHandle submit(std::function<void()> fn);
    void submit(TaskGroup& group, std::function<void()> fn);  // group-counted submit
    void wait(TaskHandle h);               // helping wait; rethrows task exception
    void wait(TaskGroup& group);           // helping wait until the group count is zero
    void wait_all();                       // helping wait until pending == 0; never rethrows
};
```

Dual-mode note: the singleton is canonical (the engine and `Application` use `JobSystem::instance()`; the plain `parallel_for(begin, end, ...)` form services it), and the class remains publicly constructible for caller-owned pools (tests). Phase 1 adds the `parallel_for(JobSystem&, ...)` overload the original header note prescribed, so caller-owned pools can receive `parallel_for` work explicitly.

#### `parallel_for` (`src/bracken/core/jobs/parallel_for.hpp`)

```cpp
inline constexpr std::size_t PARALLEL_MIN_TOTAL = 256;

template<typename Fn>
void parallel_for(std::size_t begin, std::size_t end, Fn&& fn, std::size_t grain = 0);
// services JobSystem::instance()

template<typename Fn>
void parallel_for(JobSystem& js, std::size_t begin, std::size_t end, Fn&& fn, std::size_t grain = 0);
// explicit-pool overload (Phase 1)

// Fn signature: void(std::size_t chunk_begin, std::size_t chunk_end)
```

Chunk form (not per-index) is deliberate: tight loops avoid per-element indirect-call overhead. `PARALLEL_MIN_TOTAL` is tuned so "animation/cull loops with a few dozen entities" never pay the dispatch tax (header comment).

Dispatch is a single **batch**: the `ceil(total / grain)` chunk nodes are queued and then announced with one epoch bump + wake (`JobSystem::submit_batch`), instead of a bump + notify per chunk. Chunk boundaries are unchanged — chunk `c` is `[begin + c*grain, min(end, begin + (c+1)*grain))` — so every caller's chunk-index arithmetic and determinism contract is untouched.

### Internals (Phase-1 lock-free design, plan §7)

The Phase-1 rewrite replaces the original mutex-deque scheduler. Load-bearing structures:

- **Per-worker Chase–Lev work-stealing deques** — the bounded lock-free deque in the C++11-memory-model formulation (Lê/Pop/Cohen/Nardelli 2013). The owning worker pushes and pops the **bottom** with relaxed/acquire-release operations; thieves CAS the **top**. The only contended operation is a steal — by design the rare path of a well-grained workload. LIFO owner pop / FIFO steal keeps the standard locality/contention split.
- **MPSC injection queue** — external (non-worker) submits go through a multi-producer/single-consumer queue instead of a mutexed global queue; workers drain it alongside their own deques.
- **Pooled job nodes** — fixed-size nodes in per-worker pools, recycled via freelist, with a generation counter per node backing `TaskHandle`'s done/error queries. No per-job heap allocation, no `shared_ptr` refcount traffic on the submit path (large `std::function` captures may still allocate inside the callable itself).
- **Two-phase parking** — an idle worker spins briefly, then parks on C++20 `std::atomic::wait`/`notify_one` (futex/ulock under the hood). No condition-variable + mutex pairing, no periodic idle poll: a parked pool costs ~zero. Submits `notify_one` a parked worker; `notify_all` happens at shutdown and once per `submit_batch` dispatch (which made N jobs runnable at once, so there is no herd to avoid). The pre-rewrite lost-wakeup regression tests (gated-job discard, external waiter) carry over as the safety net for the wake protocol.
- **Ordering discipline** — every atomic in the module documents its protocol at its declaration: what it publishes, its memory ordering, and its release/acquire pairing partner. Bare `seq_cst` in a hot path requires a justifying comment. Genuinely contended atomics are padded to the destructive interference size (128 B on Apple Silicon; guarded fallback constant where the standard trait is unavailable/warns).

`job_system.cpp` keeps a `thread_local` worker index (`-1` on non-worker threads) for submit routing and helping waits.

### Algorithms

#### Thread configuration — `init(unsigned thread_count = 0)`

1. If already initialized, return — **idempotent** (test `InitIsIdempotent`).
2. Resolve count: `thread_count == 0` → `hardware_concurrency() - 1` if `hardware_concurrency() > 1`, else `1`; then clamp to ≥ 1. The default leaves one hardware thread for the main thread and the pool always has ≥ 1 worker.
3. Build per-worker deques/pools; spawn the worker threads; publish initialized (release); log `Initialized {n} workers` via `BRACKEN_LOG_INFO("jobs", ...)`.

`init` itself has no env-var knowledge; the **`BRACKEN_WORKERS`** env var is read by `Application::init_engine()` and translated into the `thread_count` argument. There is no CMake option.

#### Submission

`pending` is incremented **before** enqueue (so `pending_count()` never undercounts). A job node is taken from the submitting worker's pool (or the injection path's pool for external threads); worker submits push onto the caller's own deque bottom, external submits go through the MPSC injection queue. One `notify_one` wakes a parked worker. The group-counted overload additionally increments the `TaskGroup` counter before enqueue.

`submit_batch(TaskGroup&, n, make_body)` is the n-jobs-one-wake form `parallel_for` dispatches through: it queues all `n` nodes (identical per-node work — same pool, same queue, same order) and then bumps the work epoch **once, by n**, followed by a single `notify_all`. The publish-before-bump protocol holds trivially because every push precedes the single bump. Nothing else observable changes; the win is the `n - 1` epoch RMWs and futex wakes that no longer happen (measured: 40–60 % off a parked-pool dispatch of ≥ 64 chunks, 10–15 % off the whole Prepare stage — `BM_ParallelForDispatch`, `BM_ParallelForDispatchParked`, `BM_PrepInstanceBatches`). If `make_body` throws mid-batch, the already-queued jobs are published before the exception propagates, so the group still joins.

#### Worker loop

Per iteration, until stopping: pop own deque bottom (LIFO, cache-warm) → drain the injection queue → steal from a random victim's deque top (FIFO end, CAS; never blocks) → spin briefly → park on `std::atomic::wait`. There is no periodic wake — a parked worker runs zero instructions until notified.

Profiler integration (`src/bracken/core/profiler.hpp` macros): workers are registered as thread `"JobWorker"`; a `jobs.pending` Tracy plot is emitted from the worker loop.

#### Execution

The house publication pattern: run the callable; on throw, store the exception into the node's non-atomic error slot; **then** release-store the done flag and notify parked waiters; release-`fetch_sub` the pending count and the job's `TaskGroup` counter (if any) — the decrement observing zero notifies group/`wait_all` waiters. Exceptions never escape a worker thread (test `ExceptionDoesNotKillPool`); the pool survives throwing jobs.

#### Waiting — `wait(TaskHandle)` / `wait(TaskGroup&)`

- Invalid handle → return immediately.
- Helping loop until done (acquire): (a) if the caller is a worker, pop+execute from its **own deque** first — nested `submit()` from a worker lands there, which is what makes fork-join safe on a 1-worker pool; (b) drain the injection queue; (c) steal from other workers (any caller, worker or not); (d) nothing to help with → **park** on the done flag / group counter (`std::atomic::wait`) instead of yield-spinning — completion and shutdown-discard both notify, and `atomic::wait` re-checks the value internally so a completion landing between the loop check and the park is not lost.
- `wait(TaskHandle)` rethrows the task's stored exception after completion (test `ExceptionPropagatesViaWait`). `wait(TaskGroup&)` returns when the group count reaches zero, then rethrows the first captured member exception in exactly one waiter (subsequent errors are dropped, not aggregated).

Nested submit-then-wait (fork-join) is safe on **any** pool size, including one worker — pinned by the single-worker and nested-wait tests.

#### Waiting — `wait_all()`

Helping loop until `pending == 0`: execute injected/stolen jobs; park on the pending counter when there is nothing to help with. Waits for *all* pending jobs, including ones submitted concurrently while it waits, and **never rethrows** — task errors are only observable via `TaskHandle::has_error()`/`wait(h)`.

#### Shutdown — `shutdown()`

1. No-op unless initialized.
2. Publish stopping, then a **first discard pass**: every queued-but-unstarted job is completed with a stored "discarded at shutdown" error — done published, waiters notified — so threads blocked in `wait()` on such tasks (including workers in nested waits) wake and unwind instead of deadlocking the joins.
3. `notify_all` parked workers; join all workers (each finishes its current job).
4. A **second discard pass** catches jobs submitted by in-flight work while workers wound down. If any jobs were discarded, a WARN is logged on the `jobs` channel with the count. Retained handles report `is_done() == true`, `has_error() == true`, and `wait()` rethrows the discard error (test `ShutdownCompletesDiscardedJobsWithError`).
5. State returns to pre-init; `shutdown()` + `init()` re-initialization is supported (test `InitAfterShutdownReinitializes`).

The destructor calls `shutdown()`; for the singleton this runs at static destruction. In `Application` apps the pool is shut down explicitly at the top of `shutdown_renderer()` (before GPU teardown), with an idempotent repeat in `shutdown_engine()`.

#### `parallel_for` decision tree (`parallel_for.hpp:24`)

Given `total = end - begin`, `workers = is_initialized() ? worker_count() : 1`:

1. `end <= begin` → return (no-op; test `EmptyRangeIsNoop`).
2. Not initialized or `workers <= 1` → `fn(begin, end)` inline, return (**inline fallback** — granularity control, not a second code path; this is also what `BRACKEN_WORKERS=1` exercises for whole-loop determinism checks).
3. Auto-grain (`grain == 0`): if `total < PARALLEL_MIN_TOTAL` (**256**) → serial inline; else `grain = max(64, total / (workers * 4))` (~4 chunks per worker, 64-element floor).
4. Explicit `grain != 0` **bypasses** the 256 guard (caller opts into dispatch overhead).
5. `total <= grain` → serial inline (test `SerialFallbackForLargeGrain`).
6. Otherwise: submit `ceil(total / grain)` jobs — each `js.submit([i, chunk_end, fn]() { fn(i, chunk_end); })`, i.e. **`fn` is copied by value into every chunk job** — then `js.wait(h)` on each handle in submission order, **joining all chunks before propagating any exception**: the first chunk exception (in chunk order) is captured and rethrown only after every chunk has completed; subsequent exceptions are dropped. Fully blocking on both the normal and the throwing path, so captured references need only outlive the call.

The plain form services `JobSystem::instance()`; the `parallel_for(JobSystem&, ...)` overload runs the same decision tree against an explicit pool. The grain guards are the granularity control inside the one code path — a chunk count of 1 at small totals is not a "serial mode".

### Constants

| Constant | Value | Where | Meaning |
|---|---|---|---|
| `PARALLEL_MIN_TOTAL` | `256` | `parallel_for.hpp` | below this total (auto-grain mode) run serially |
| auto-grain floor | `64` | `parallel_for.hpp` | `grain = max(64, total / (workers * 4))` |
| auto-grain divisor | `workers * 4` | `parallel_for.hpp` | targets ~4 chunks per worker |
| default worker count | `hardware_concurrency() - 1`, min 1 | `job_system.cpp` | `init(0)`; overridden per-app by `BRACKEN_WORKERS` via `Application::init_engine` |
| cache-line padding | 128 B | jobs headers | destructive-interference padding for contended atomics (Apple Silicon cache lines; guarded fallback constant when the standard trait is unavailable) |

### Threading and lifetime contracts

- All `JobSystem` public methods are `worker-safe` (callable from any thread — see the tier table in `docs/architecture.md`); the tests exercise concurrent submitters. Queue state is lock-free (Chase–Lev deques + MPSC injection); flags/counters are atomics with documented acquire/release pairing.
- `init()` must complete before concurrent use is meaningful; in practice `Application::init_engine()` calls it once at boot on the main thread before anything else runs.
- Jobs run on arbitrary worker threads (or inline on a waiting thread via helping). Jobs must **not** touch `main-thread`-tier engine state — swapchain acquire, `render_frame`, GPU resource creation/destruction, `FileWatcher` entry points are `BRACKEN_ASSERT_MAIN_THREAD()`-guarded per `docs/architecture.md`.
- Job-node lifetime is pool-owned; `TaskHandle`s are (node, generation) references and may outlive the pool safely — a recycled node's generation check makes stale handles report done rather than read a newer job's state.
- `wait()`/`wait_all()` may execute *other* queued jobs on the calling thread (helping) — reentrancy into arbitrary submitted code from the waiting thread is by design.
- Teardown order in `Application` apps: pool joined at the top of `shutdown_renderer()` (before GPU teardown); idempotent repeat in `shutdown_engine()` before `SDL_Quit()`; otherwise the static singleton's destructor shuts down at program exit.
- Concurrency changes to this module must be run under ThreadSanitizer: reconfigure `build/` with `-DCMAKE_CXX_FLAGS=-fsanitize=thread` (and the matching `_C_FLAGS`/`_EXE_LINKER_FLAGS`), rebuild, and run `ctest --test-dir build`.

### Error handling

- Job exceptions: captured to `Task::error` via `catch (...)`; rethrown only by `wait(h)`. `wait_all()` swallows errors. No logging on job failure; shutdown-discarded jobs are the exception — they get a WARN with the discard count.
- `init` failure modes: none surfaced (thread creation failure would throw `std::system_error` out of `init`; not handled).
- No `core::Result` usage anywhere in this module — predates/exempt from the newer `Result` convention, consistent with "existing APIs stay as they are until touched" (`CLAUDE.md`).

### Performance characteristics

- Per-`submit` cost: one pooled-node acquire (freelist), potential `std::function` allocation for large captures, the deque/injection push, one atomic RMW on `pending`, one `notify_one` if a worker is parked. No per-job heap allocation, no refcount churn.
- LIFO local pop / FIFO steal gives the standard locality/contention split; a steal is a CAS on the victim's top — the only contended operation in the scheduler, rare under proper grains.
- Idle cost when initialized: **~zero** — parked workers sleep on `std::atomic::wait` with no periodic wakeup (the old 50 ms poll is gone; this is what makes an always-on pool free in quiet scenes).
- `wait()` on work it cannot help with parks on the task's done flag (`std::atomic::wait`) — no busy-yield, no core burned while waiting.
- `parallel_for` fixed overhead exists even on the serial path (branching, `instance()` call); design notes: "for a handful of entities, just iterate", and "a job system saving 200µs on a 300-entity scene is felt nowhere".
- `parallel_for` copies `fn` per chunk; heavy captures multiply.
- Benchmarks: `build/src/bracken/bracken_benchmarks --benchmark_filter=ParallelFor` compares `BM_SerialSum` vs `BM_ParallelForSum` (1M-float reduction, grains 1024/16384).

### Known limitations, stubs, and silent behaviors

- **Jobs queued at shutdown do not run**: they complete with a "discarded at shutdown" error (handles wake, `wait()` rethrows) and a WARN log — drain with `wait_all()` first if the work must happen.
- **`wait_all()` swallows errors** — per-handle `wait()`/`has_error()` is the failure-aware path.
- **`parallel_for` reports only the first chunk exception** (in chunk order); later exceptions are dropped after all chunks join.
- **`wait(TaskGroup&)` does not aggregate errors** — a group is a bare counter; keep handles when per-job failure matters.
- No priorities, dependencies/continuations, cancellation, task return values, or affinity — `TaskGroup` fork-join is deliberately the only orchestration primitive (plan §4.1: no dependency-graph scheduler until a workload demands one).

---

## Asset Registry & Hot Reload

### Scope and module layout

Module `src/bracken/core/asset/`, namespace `bracken::core::asset`. Two units:

| File | Contents |
|---|---|
| `src/bracken/core/asset/file_watcher.hpp` / `.cpp` | `FileWatcher`, `WatchCallback` |
| `src/bracken/core/asset/asset_registry.hpp` (header-only) | `AssetRegistry<T>` class template |
| `src/bracken/core/asset/file_watcher.test.cpp`, `asset_registry.test.cpp` | GoogleTest coverage |

Consumers of the hot-reload flow outside the module (documented here because they define what actually hot-reloads):

- `bracken::Application` (`src/bracken/application.cpp`) — lifecycle owner of the singleton watcher.
- `bracken::gpu::PipelineRegistry` (`src/bracken/gpu/pipeline_registry.{hpp,cpp}`) — shader (`.spv`) hot reload.
- `bracken::ModelManager` (`src/bracken/model_loader.{hpp,cpp}`) — glTF model hot reload.
- `bracken::gpu::GPUTexture::reload_from_file` (`src/bracken/gpu/gpu_texture.hpp`) — texture reload primitive, **not wired to any watch**.

Note a doc discrepancy: `docs/architecture.md:23` describes `core/asset/` as "`AssetRegistry` (typed, stable handles, hot-reload), `FileWatcher`", but no engine system instantiates `AssetRegistry<T>` — `ModelManager` and `PipelineRegistry` register watches directly on `FileWatcher`. `AssetRegistry` is exercised only by its own tests; it is a host-app-facing utility.

### FileWatcher

#### Purpose

Polling-based, cross-platform file/directory mtime watcher. A background `std::jthread` scans registered paths every `poll_interval`; detected changes are pushed onto an internal queue; callbacks dispatch on the main thread from `poll_main_thread()`, so they may safely touch GPU state.

#### Public API

```cpp
using WatchCallback = std::function<void(const std::filesystem::path&)>;

class FileWatcher {
public:
    using WatchHandle = uint32_t;
    static constexpr WatchHandle k_invalid_handle = 0;

    FileWatcher() = default;
    ~FileWatcher();                       // calls shutdown()
    // non-copyable, non-movable

    static FileWatcher& instance();       // Meyers singleton (function-local static)

    [[nodiscard]] bool init(std::chrono::milliseconds poll_interval = std::chrono::milliseconds{1000});
    void shutdown();
    bool is_initialized() const noexcept;

    [[nodiscard]] WatchHandle watch_file(std::filesystem::path path, WatchCallback on_change);
    [[nodiscard]] WatchHandle watch_directory(std::filesystem::path dir, std::string extension_filter,
                                              WatchCallback on_change);
    void unwatch(WatchHandle h);

    void poll_main_thread();              // drain queue, fire callbacks on calling thread
};
```

#### Data structures

- `Watch` (private): `{ WatchHandle handle; WatchKind kind (File|Directory, uint8_t enum); std::filesystem::path target; std::string ext_filter; WatchCallback callback; std::unordered_map<std::string,int64_t> mtimes; bool first_scan = true; }`. The mtime cache is keyed by `path.lexically_normal().string()` and stores `std::filesystem::file_time_type` converted to nanoseconds since the filesystem-clock epoch (`to_nanos`).
- `PendingEvent` (private): `{ WatchHandle handle; std::filesystem::path path; }`.
- State: `std::vector<Watch> watches_` guarded by `watches_mutex_`; `std::vector<PendingEvent> pending_` guarded by `queue_mutex_` (unbounded); `next_handle_ = 1` monotonically increasing (never recycled, so `0` is permanently invalid); `wake_cv_`/`wake_mutex_` for prompt shutdown; `std::atomic<bool> initialized_`, `stop_`; `std::jthread worker_`.

#### Constants

| Constant | Value | Location |
|---|---|---|
| `k_invalid_handle` | `0` | `file_watcher.hpp` |
| default `poll_interval` | `1000 ms` | `FileWatcher::init` default argument |
| `k_settle_window_ns` | `100 * 1'000'000` ns (100 ms) | anonymous namespace, `file_watcher.cpp` |

#### Algorithm

`worker_loop()` (background thread, Tracy-tagged `"FileWatcher"` via `BRACKEN_PROFILE_THREAD`), per tick:

1. Take `watches_mutex_` for the **entire scan** and call `scan_watch` on every watch (profiled zone `"FileWatcher::scan"`).
2. Append produced events to `pending_` under `queue_mutex_`.
3. `wake_cv_.wait_for(poll_interval_)` — woken early only by `shutdown()`.

`scan_watch(Watch&, out, now_ns)` per path (`record` lambda):

- `fs::last_write_time(p, ec)`; on error, skip silently.
- **Settle window**: if `now_ns - mtime_ns < k_settle_window_ns` (file modified within the last 100 ms), skip — picked up on a later tick. Avoids reacting to half-written files.
- New key in `mtimes`: insert; emit an event **only if `!first_scan`** (the first pass primes the cache without firing; files appearing after the first pass do fire — verified by test `FileCreatedAfterWatchIsReported`).
- Existing key with different mtime: update and emit.

Kind dispatch: `File` watches check `fs::exists(target)` then record that one path (missing file = silent skip; deletion never emits). `Directory` watches require `exists && is_directory`, then walk `fs::recursive_directory_iterator` with `std::error_code` (no exceptions), skipping non-regular files and, when `ext_filter` is non-empty, any entry whose `path().extension().string()` is not **exactly equal** (case-sensitive) to the filter. Empty filter matches every regular file. `first_scan` is cleared at the end of the scan for file watches even when the file is missing; a directory watch whose target is missing **early-returns before clearing `first_scan`**, so its first successful scan still primes the cache without firing.

`poll_main_thread()`:

1. Swap `pending_` out under `queue_mutex_`; early-return if empty.
2. Under `watches_mutex_`, snapshot `(callback copy, path)` pairs for each drained event whose handle still matches a live watch (linear search per event); events for unwatched handles are dropped.
3. Invoke callbacks **outside** both locks — explicitly to avoid deadlock when a callback registers/unregisters watches (comment in `file_watcher.cpp`). Consequence: `WatchCallback` must be copyable, and one copy is made per fired event.

#### Threading and lifetime contract

- `init`, `shutdown`, `watch_file`, `watch_directory`, `unwatch`, `poll_main_thread` are **main-thread-only**; all but `init` are guarded with `BRACKEN_ASSERT_MAIN_THREAD()` (`src/bracken/core/assert.hpp`; compiled out under `NDEBUG`). `init` instead calls `core::bind_main_thread()` itself, so a watcher inited before `Application::init_engine` still records the correct main thread.
- `shutdown()` sets `stop_`, notifies `wake_cv_`, `request_stop()`s and joins the jthread, then clears `watches_` and `pending_`. Idempotent (`initialized_.exchange(false)` guard); re-`init()` after shutdown works (test `InitIsExclusiveAndShutdownIdempotent`). The destructor calls `shutdown()`; test `DestroyWhileWatchingJoinsPromptly` asserts join completes < 2 s at a 10 ms interval.
- Callbacks captured by reference must outlive the watch or be `unwatch`ed first; the engine's built-in users capture `this` of objects that live until shutdown.

#### Error handling

- `init()` returns `false` if already initialized (no re-configuration of the interval).
- `watch_file`/`watch_directory` return `k_invalid_handle` when the watcher is uninitialized **or** the callback is empty (tests `WatchBeforeInitReturnsInvalidHandle`, `NullCallbackReturnsInvalidHandle`). No validation of the path — nonexistent targets are legal and are watched gracefully (test `MissingPathsAreWatchedGracefully`).
- `unwatch(k_invalid_handle)` and `poll_main_thread()` on an uninitialized watcher are no-ops.
- All filesystem errors use the `std::error_code` overloads and are swallowed silently.

#### Performance characteristics

- Cost per tick: O(total files under all watches) `stat` calls, on the background thread, while holding `watches_mutex_` — main-thread `watch_*`/`unwatch`/`poll_main_thread` can block for up to a full scan of large directory trees.
- Event queue is unbounded; if `poll_main_thread()` is never called, events accumulate.
- Rapid successive writes within one poll interval coalesce into (at most) one event per watch per tick; per-event callback lookup in `poll_main_thread` is O(#watches).

#### Known limitations / silent behaviors

- Pure polling — no inotify/FSEvents/ReadDirectoryChangesW backend.
- Change detection is mtime-only; a rewrite that preserves mtime is invisible, and an mtime-only touch with identical content fires (test `MtimeChangeAloneIsDetected`).
- Deletions and renames-away are never reported; a file re-appearing with an unchanged mtime after deletion does not fire (its cache entry survives).
- Directory watches do not honor filters like `"*.spv"` or extension case variants — exact string match against `.extension()` only.
- Worst-case notification latency = poll interval + 100 ms settle window + time to next `poll_main_thread()`.

### AssetRegistry\<T\>

#### Purpose

Generic per-type asset registry (`src/bracken/core/asset/asset_registry.hpp`, header-only template) with: load-by-path, name→handle lookup, generation-checked stable handles, manual reload, and hot reload via a borrowed `FileWatcher`. Handles are stable across reload (slot reused) and invalidated on unload.

#### Public API

```cpp
template<typename T> class AssetRegistry {
public:
    using Handle = bracken::core::Handle<T>;                       // core/handle.hpp
    using LoadFn = std::function<bool(const std::filesystem::path&, T&)>;

    AssetRegistry();                                               // installs default_load_fn()
    void set_load_fn(LoadFn fn);

    Handle load(std::string name, std::filesystem::path path);    // Handle::invalid() on failure
    T* get(Handle h);                                              // nullptr on invalid/stale
    const T* get(Handle h) const;
    Handle find(const std::string& name) const;                    // Handle::invalid() if absent
    [[nodiscard]] bool reload(Handle h);
    void unload(Handle h);
    void enable_hot_reload(FileWatcher& watcher);
};
```

`core::Handle<Tag>` (`src/bracken/core/handle.hpp`) is `{ uint32_t index; uint32_t generation; }` with `k_invalid_index = 0xFFFFFFFF`, `invalid() = {k_invalid_index, 0}`, `is_valid()` = index check only, `==`/`!=`, and a `std::hash` specialization. (`AssetRegistry` uses `Handle<T>` directly, not `HandlePool`.)

#### Data structures and slot lifecycle

- `Entry` (private): `{ std::string name; std::filesystem::path path; T value{}; Handle handle{}; FileWatcher::WatchHandle watch = k_invalid_handle; }`.
- Storage: `std::vector<std::unique_ptr<Entry>> slots_` — each entry is heap-allocated, so `T*` from `get()` remains stable even when `slots_` reallocates, and across `reload` (value is move-assigned in place). It is invalidated only by `unload`.
- Free-list: parallel stacks `free_indices_` / `free_generations_`. `next_handle()` pops both (reusing the index with the bumped generation) or appends a new slot with generation **1**. `unload` pushes `generation + 1` and the index, then resets the slot; `entry_at` validates `is_valid() && index < slots_.size() && slot && slot->handle.generation == h.generation`, so stale handles fail cleanly.
- Name map: `std::unordered_map<std::string, Handle> name_to_handle_`. `load` assigns unconditionally — loading a duplicate name **overwrites the mapping without unloading the previous entry**; the old entry (and its file watch) stays live and reachable only by its handle. `unload` erases by the entry's stored name (which, after such an overwrite, removes the *newer* mapping — a silent behavior to be aware of).

#### Load / reload semantics

- Default `LoadFn` (installed by the constructor): `[](const fs::path& p, T& out) { return out.reload_from(p); }` — `T` must provide `bool reload_from(const std::filesystem::path&)` unless `set_load_fn` overrides it. `T` must be default-constructible in all cases (`T value{}` in `Entry`, `T fresh{}` in `reload`).
- `load`: constructs the entry, runs `load_fn_`; on failure logs `BRACKEN_LOG_ERROR("hot_reload", "AssetRegistry: failed to load '{}'")` and returns `Handle::invalid()` **without** consuming a slot. On success, installs a file watch if `enable_hot_reload` was already called, registers the name, and stores the slot.
- `reload` (`[[nodiscard]] bool`): loads into a fresh `T` first; on failure logs a `hot_reload` warning and returns `false`, leaving the current value untouched. On success move-assigns and logs `AssetRegistry: reloaded '{}'`.
- `unload`: validates the handle (index+generation), calls `watcher_->unwatch` on the entry's watch if any, erases the name mapping, recycles the slot.
- `enable_hot_reload(FileWatcher&)`: stores the (non-owning) watcher pointer and retrofits `watch_file` onto every already-loaded entry that lacks one; each watch callback is `[this, h](const fs::path&) { (void)reload(h); }`. Because the callback captures the handle, generation checking makes callbacks for since-unloaded entries harmless (though `unload` also unwatches). There is no API to disable hot reload once enabled.

#### Threading, error handling, limitations

- No internal locking; single-threaded (main-thread) use is implied by the `FileWatcher` contract, since hot-reload callbacks arrive via `poll_main_thread()`.
- All failures are `bool`/invalid-handle + `hot_reload`-category logs — consistent with the project error-handling convention for non-`Result` APIs.
- The registry never calls `FileWatcher::init`; the watcher passed to `enable_hot_reload` must already be initialized or `watch_file` silently returns `k_invalid_handle` and no reloads will occur (the stored `watch` stays `k_invalid_handle`, so a later `enable_hot_reload` call with a live watcher would retrofit it).
- Not used by any in-tree engine system (see the discrepancy note above).

### Hot-reload wiring in Application

`src/bracken/application.cpp`:

- `init_engine()` (line ~42): reads `BRACKEN_HOT_RELOAD` with `std::getenv`; enables iff `env && env[0] == '1'`. Calls `core::asset::FileWatcher::instance().init()` — the process-global singleton, default 1000 ms interval — and logs `Hot-reload enabled (BRACKEN_HOT_RELOAD=1)`. `core::bind_main_thread()` is called just before, anchoring the `BRACKEN_ASSERT_MAIN_THREAD` checks.
- `run()` main loop: `if (FileWatcher::instance().is_initialized()) FileWatcher::instance().poll_main_thread();` — executed once per frame at the **frame fence** (`Application::frame_fence`), with no sim job in flight, so hot-reload never applies while a packet referencing the old resource is being built or rendered.
- `init_renderer(...)` (lines ~143–146): unconditionally calls `pipeline_registry()->enable_hot_reload(FileWatcher::instance())` and `scene_renderer_->models().enable_hot_reload(FileWatcher::instance())`. When the env var is unset the watcher is uninitialized, all `watch_*` calls return `k_invalid_handle`, and these are effective no-ops.
- `shutdown_engine()` (line ~123): `FileWatcher::instance().shutdown()` before `JobSystem` shutdown and `SDL_Quit()`.

### Shader hot reload (gpu::PipelineRegistry)

`src/bracken/gpu/pipeline_registry.{hpp,cpp}`:

```cpp
void enable_hot_reload(core::asset::FileWatcher& watcher);   // watch shader dir
void reload_all_pipelines();                                  // force a reload pass (tests/diagnostics)
const std::string& shader_path() const;                       // default "shaders/"
```

- `enable_hot_reload`: warns and returns if called before `init` (`device_ == nullptr`); idempotent (`watcher_ != nullptr` early-out). Registers `watch_directory(shader_path_, ".spv", cb)` and stores the `uint32_t` handle in `shader_watch_handle_`. By this point `init` has set `shader_path_ = SDL_GetBasePath() + "shaders/"` — **next to the executable** (falling back to the relative `"shaders/"` only if `SDL_GetBasePath()` fails); that is where `bracken_copy_shaders` (`src/bracken/CMakeLists.txt:136`) stages `${BRACKEN_SHADER_OUTPUT_DIR}` as `$<TARGET_FILE_DIR:target>/shaders`.
- Silent behavior: if the `FileWatcher` is uninitialized, `watch_directory` returns `k_invalid_handle` but the method still sets `watcher_`, stores handle `0`, and logs `PipelineRegistry hot-reload enabled at {}` — the log can claim "enabled" with nothing watched.
- The callback fires on **any** `.spv` change under the tree and calls `reload_all_pipelines()`: `invalidate_all()` (clears the `unordered_map<PipelineType, unique_ptr<GPUPipeline>>` cache **and** `shader_manager_->clear_cache()`), then `preload_all_pipelines()` (iterates every `PipelineType` through `get_pipeline`, recreating all). Failure of any pipeline logs `PipelineRegistry: one or more pipelines failed to rebuild`; there is no rollback — failed pipelines are simply absent until recreated on next access/reload. Header comment: "Coarse but correct; live pipelines from the previous frame are released by SDL once the in-flight commands finish."
- N saved `.spv` files can produce up to N events in one tick → up to N full rebuilds drained in a single `poll_main_thread()` call (events are not deduplicated per watch across distinct files).

### Model hot reload (ModelManager)

`src/bracken/model_loader.{hpp,cpp}`. `ModelHandle` is `uint32_t`; `INVALID_MODEL_HANDLE = 0` (slot 0 reserved; models start at index 1). `models_` (vector of `unique_ptr<Model>`) is parallel to `model_paths_` (`""` for procedural models registered via `register_model`).

```cpp
[[nodiscard]] bool reload_model(ModelHandle handle);
void enable_hot_reload(core::asset::FileWatcher& watcher);
```

- `enable_hot_reload`: stores the watcher and installs `watch_file(path, [this,h]{ (void)reload_model(h); })` for every already-loaded entry with a non-empty path. `load_model` installs the same watch at load time when `watcher_` is set, so call order relative to loading doesn't matter. Watches are **never unwatched** (source comment: "Watched for the lifetime of the app").
- `reload_model`: rejects invalid/out-of-range handles and empty (procedural) paths. Parses into a fresh `Model` via `ModelLoader::load_glb`; on parse failure logs `ModelManager: reload failed for '{}'` (`hot_reload` category) and returns `false`, leaving the old model intact. On success (and when a `GPUDevice` is set): `upload_to_gpu(fresh)` → `free_gpu_resources(old)` → `*models_[handle] = std::move(*fresh)` — the new contents are **move-assigned into the existing `Model` object**, so both the `ModelHandle` and any raw `Model*` previously returned by `get_model` stay valid; only the model's contents (and pointers *into* it) change.
- Uses the model's original load path; embedded textures are re-created as part of `upload_to_gpu`.

### Texture reload primitive (unwired)

`gpu::GPUTexture::reload_from_file(const std::string& path, bool generate_mipmaps = true)` (`src/bracken/gpu/gpu_texture.hpp`, `[[nodiscard]] bool`): recreates the underlying `SDL_GPUTexture` from a file and releases the old one through the device; returns `false` (existing texture untouched) if the file cannot be loaded. No engine system installs a watch that calls it; it exists as a hot-reload building block only.

### Constants summary

| Constant | Value | Location |
|---|---|---|
| `FileWatcher::k_invalid_handle` | `0` | `src/bracken/core/asset/file_watcher.hpp` |
| `FileWatcher::init` default poll interval | 1000 ms | `src/bracken/core/asset/file_watcher.hpp` |
| `k_settle_window_ns` | 100 ms (`int64_t{100} * 1'000'000`) | `src/bracken/core/asset/file_watcher.cpp` |
| First slot generation in `AssetRegistry` | 1 | `src/bracken/core/asset/asset_registry.hpp` (`next_handle`) |
| `core::Handle<T>::k_invalid_index` | `0xFFFFFFFF` | `src/bracken/core/handle.hpp` |
| `INVALID_MODEL_HANDLE` | `0` | `src/bracken/model_loader.hpp` |
| `PipelineRegistry::shader_path_` default | `"shaders/"` | `src/bracken/gpu/pipeline_registry.hpp` |
| Shader watch extension filter | `".spv"` | `src/bracken/gpu/pipeline_registry.cpp` |
| `BRACKEN_HOT_RELOAD` accepted value | first char `'1'` | `src/bracken/application.cpp` |

---

## Memory: Arena & Frame Allocators

### Scope and files

Namespace `bracken::memory`. Four source files plus tests:

| File | Contents |
|---|---|
| `src/bracken/memory/arena.hpp` | `class Arena` — growable/fixed bump allocator with destructor registration |
| `src/bracken/memory/arena.cpp` | allocation, growth, reset, move implementation |
| `src/bracken/memory/frame_allocator.hpp` | `class FrameAllocator` — double-buffered `Arena` pair (header-only logic) |
| `src/bracken/memory/frame_allocator.cpp` | empty translation unit (namespace only) |
| `src/bracken/memory/arena.test.cpp`, `frame_allocator.test.cpp` | GoogleTest coverage |
| `src/bracken/core/core.bench.cpp` | `BM_ArenaAllocate`, `BM_FrameAllocatorAllocSwap` benchmarks |

No other headers are exported from `memory/`. There is no STL-allocator
adapter, no `std::pmr` integration, and no thread-safety layer.

### Arena

#### Purpose

Linear (bump-pointer) allocator over a list of malloc'd chunks, with an
optional caller-supplied fixed buffer mode and registration of non-trivial
destructors so `create<T>()`d objects are destroyed on `reset()`/destruction.
Described in `docs/architecture.md` module map as "`Arena` (bump + destructor
registration)".

#### Public API surface (`src/bracken/memory/arena.hpp`)

```cpp
class Arena {
public:
    static constexpr std::size_t k_default_chunk_size = 64 * 1024;   // 65536 B

    Arena() = default;                          // growable, default chunk size
    explicit Arena(std::size_t initial_chunk_size); // growable; 0 -> default size
    Arena(void* buffer, std::size_t size);      // fixed-buffer, non-owning

    ~Arena();                                   // destructors, then free chunks

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&& other) noexcept;              // moved-from: valid + empty
    Arena& operator=(Arena&& other) noexcept;

    void* allocate(std::size_t size, std::size_t align);

    template<typename T, typename... Args> T* create(Args&&... args);

    void reset();

    std::size_t bytes_used() const noexcept;
    std::size_t bytes_capacity() const noexcept;
    bool owns_memory() const noexcept;          // false for fixed-buffer arenas
};
```

Note: `allocate` and `create` are **not** marked `[[nodiscard]]` (the
project convention in `CLAUDE.md` mandates `[[nodiscard]]` for new fallible
APIs; these predate/omit it).

#### Data structures and memory layout

```cpp
struct Chunk      { std::byte* base; std::size_t size; std::size_t used; };
struct Destructor { void* obj; void (*fn)(void*); };

std::vector<Chunk>      chunks_;
std::vector<Destructor> destructors_;
std::size_t default_chunk_size_ = k_default_chunk_size;
bool owns_memory_  = true;   // false when constructed from caller buffer
bool fixed_buffer_ = false;  // true only for the (void*, size) ctor
```

- Chunk payloads come from `std::malloc`; base alignment is therefore
  `alignof(std::max_align_t)`.
- Bookkeeping (`chunks_`, `destructors_`) lives on the **regular heap** —
  `create<T>()` of a non-trivially-destructible `T` performs a
  `std::vector::push_back`, which may itself allocate.
- Constructors do **not** preallocate: `Arena(initial_chunk_size)` only
  stores the size; the first chunk is malloc'd lazily on first `allocate()`.
  The fixed-buffer ctor pushes one non-owned `Chunk` immediately.

#### Allocation algorithm (`arena.cpp:66-101`)

```cpp
inline std::size_t align_up(std::size_t v, std::size_t a) {
    return (v + (a - 1)) & ~(a - 1);   // REQUIRES a to be a power of two
}
```

0. `BRACKEN_ASSERT(align == 0 || std::has_single_bit(align))` — non-power-of-two
   alignments are debug-asserted (they would silently misalign).
1. `size == 0` → return a valid **non-null** sentinel (the address of a
   `static std::max_align_t`; never dereferenceable, no alignment guarantee),
   so `nullptr` uniquely signals failure. No arena bytes are consumed.
2. `align == 0` → substitute `alignof(std::max_align_t)`.
3. Try to bump **only** `chunks_.back()`: compute
   `aligned = align_up(base + used, align) - base`; if `aligned + size <=
   chunk.size`, set `used = aligned + size` and return `base + aligned`.
   Alignment padding is thus counted in `used`. Alignment is computed from
   the absolute address, so an arbitrarily aligned caller buffer works at the
   cost of padding bytes.
4. On miss: fixed-buffer arenas return `nullptr`; growable arenas call
   `grow(size + align)`, then retry against the fresh chunk. A defensive
   post-grow recheck can still return `nullptr`, though it is unreachable
   given the over-request by `align` bytes.

Earlier chunks' tail space is **never revisited** — allocation only ever
bumps the newest chunk.

#### Growth (`arena.cpp:103-116`)

`grow(min_bytes)` mallocs `max(default_chunk_size_, min_bytes)` bytes,
records it via `BRACKEN_PROFILE_ALLOC` (Tracy memory tracking, compiled out
unless `BRACKEN_ENABLE_TRACY=ON` — see `src/bracken/core/profiler.hpp`), and
appends the chunk. Returns `false` on malloc failure, which `allocate`
surfaces as `nullptr`. Oversized requests get a dedicated exact-fit
(`size + align`) chunk; there is no doubling policy.

#### `create<T>()` (header, `arena.hpp:28-38`)

- `allocate(sizeof(T), alignof(T))`; on `nullptr`, returns `nullptr` without
  constructing.
- Placement-new with perfectly forwarded args.
- `if constexpr (!std::is_trivially_destructible_v<T>)` registers
  `{obj, [](void* p){ static_cast<T*>(p)->~T(); }}` in `destructors_`.
  Trivial types cost nothing beyond the bump.
- If `T`'s constructor throws, the exception propagates; no destructor was
  registered and the bumped bytes are simply abandoned until reset (safe, but
  unstated in source).

#### `reset()` semantics (`arena.cpp:118-137`)

1. `destroy_all()`: run registered destructors in **reverse registration
   order** (rbegin→rend), then clear the list.
2. Fixed-buffer: rewind `chunks_[0].used = 0`; the caller's buffer is kept.
3. Growable: `std::free` every chunk except index 0 (each paired with
   `BRACKEN_PROFILE_FREE`), `chunks_.resize(1)`, rewind chunk 0. The retained
   chunk is whichever was allocated *first* — if the first-ever allocation
   forced an oversized chunk, that oversized chunk is what persists across
   resets. Comment in source: "Keep the first chunk to amortize allocations
   across resets."

All previously returned pointers are invalidated by contract.

#### Destruction and move semantics (`arena.cpp:32-64`)

- `~Arena()`: `destroy_all()` then `free_chunks()`; `free_chunks` frees chunk
  memory only when `owns_memory_` is true (never for fixed-buffer arenas).
- Move ctor/assign transfer both vectors and flags; the moved-from arena is
  left with cleared vectors, `owns_memory_ = true`, `fixed_buffer_ = false` —
  i.e. a valid, empty, growable default arena (`bytes_used() == 0`; verified
  by `Arena.MoveTransfersOwnership`). Move-assignment first destroys and
  frees the destination's current contents. Both are `noexcept` (the vector
  moves cannot throw; note the destination cleanup in move-assign runs
  registered destructors, which must therefore not throw).

#### Introspection

`bytes_used()` = Σ `chunk.used` (includes alignment padding);
`bytes_capacity()` = Σ `chunk.size`. Both O(#chunks), `noexcept`.

#### Error handling

No exceptions, no `core::Result`, no logging: every failure mode
(fixed-buffer exhaustion, malloc failure) is a **silent `nullptr`** —
uniquely, since zero-size requests return a non-null sentinel. `allocate`
and `create<T>` are `[[nodiscard]]`.

#### Threading contract

**Single owner, owner-asserted** (`owned` tier — `docs/architecture.md`).
No atomics or mutexes anywhere in the module; an arena is thread-confined by
contract, and Phase 1 of `docs/multithreading_plan.md` adds a debug
owner-thread assert on use (lockless bump stays lockless — the check catches
accidental cross-thread sharing instead of guarding it). Per-worker scratch
arenas follow the same rule, one per worker. All other use must be
externally synchronized.

#### Performance characteristics

Steady-state `allocate` is one address round-up, one compare, one add.
Growth path adds a `malloc` + `vector::push_back`. Benchmarked by
`BM_ArenaAllocate` (`src/bracken/core/core.bench.cpp`): mixed 16–256 B
allocations at align 16 from a 1 MiB-chunk arena, reset every 1024 allocs.

#### Known limitations / silent behaviors

- `align` must be a power of two — debug-asserted; in release (`NDEBUG`)
  builds a non-power-of-two value still produces a misaligned pointer.
- No per-allocation free, no realloc, no shrink other than `reset()`.
- Abandoned tail space in non-final chunks is unrecoverable until reset.
- Destructor registration allocates on the global heap.

### FrameAllocator

#### Purpose

Double-buffered per-frame scratch: producers allocate into the current
frame's arena while consumers may still read last frame's data. One `swap()`
per frame flips buffers and recycles the older one.

#### Public API surface (`src/bracken/memory/frame_allocator.hpp`, header-only)

```cpp
class FrameAllocator {
public:
    FrameAllocator() = default;                        // two default Arenas (64 KiB chunks)
    explicit FrameAllocator(std::size_t initial_chunk_size); // both arenas use this size

    FrameAllocator(const FrameAllocator&) = delete;
    FrameAllocator& operator=(const FrameAllocator&) = delete;
    FrameAllocator(FrameAllocator&&) = default;
    FrameAllocator& operator=(FrameAllocator&&) = default;

    Arena& this_frame() noexcept;        // active arena (also const overload)
    Arena& last_frame() noexcept;        // previous frame's arena (also const overload)
    void swap();                         // current_ = 1 - current_; arenas_[current_].reset();
private:
    Arena arenas_[2]{};
    int current_ = 0;
};
```

#### Algorithm and lifetime contract

`swap()` flips `current_` and immediately `reset()`s the newly active arena.
Consequences (all test-verified in
`src/bracken/memory/frame_allocator.test.cpp`):

- An allocation made in frame N is valid for the remainder of frame N and
  all of frame N+1 (reachable as `last_frame()`); it is destroyed/invalidated
  by the `swap()` that begins frame N+2 (`ContentSurvivesOneFrame`,
  `PreviousFrameDataIntactUntilSecondSwap`).
- `this_frame()` is empty (`bytes_used() == 0`) at the start of every frame
  (`EveryFrameStartsEmptyAcrossManySwaps`).
- Growth chunks acquired during a spike frame are released when that arena is
  reset two swaps later; capacity returns to the initial chunk size
  (`SwapCycleReleasesGrowthChunks`).
- Objects `create<T>()`d into a frame arena get their destructors run by that
  deferred `reset()` — i.e. destruction happens two `swap()`s after creation,
  not at frame end.
- The two arenas are distinct allocations, so fresh-frame writes cannot alias
  live last-frame blocks (`AllocateAfterSwapReturnsFreshMemory`).

#### Threading / error handling

Same as `Arena`: single-owner, owner-asserted in debug (no locks), silent
`nullptr` on failure. `swap()` must be externally serialized against all
`this_frame()`/`last_frame()` users.

#### Performance

`swap()` cost = one integer flip + `Arena::reset()` (destructor sweep + free
of any growth chunks). Benchmarked by `BM_FrameAllocatorAllocSwap`
(`src/bracken/core/core.bench.cpp`): 256 × 64 B/align-16 allocations then
`swap()`, on a 1 MiB-chunk allocator.

### Engine-internal consumers and frame-loop status

- **Not wired into the frame loop.** Neither `Application` nor
  `SceneRenderer` owns an `Arena` or `FrameAllocator`; the playground
  (`examples/playground/`) does not use them. Hosts that want per-frame
  scratch own a `FrameAllocator` and call `swap()` once per frame themselves.
- `src/bracken/render/terrain_renderer.cpp:330` — one-shot local
  `memory::Arena scratch(arena_bytes)` sized exactly for terrain mesh build
  (vertices + indices + height cache), replacing two heap-grown vectors;
  freed at scope exit.
- Design rationale (`docs/engine_design_notes.md`, "`memory/`
  FrameAllocator" bullet): adopting it for capacity-stable vectors measured
  as a performance wash with extra bookkeeping; use only where allocation
  churn is measured. `docs/audits/performance.md` records the non-adoption as
  a deliberate non-issue.

### Constants

| Constant | Value | Defined in |
|---|---|---|
| `Arena::k_default_chunk_size` | `64 * 1024` (65536) | `src/bracken/memory/arena.hpp` |
| `FrameAllocator` buffer count | 2 (fixed array `arenas_[2]`) | `src/bracken/memory/frame_allocator.hpp` |

### Profiler integration

Chunk mallocs/frees are bracketed with `BRACKEN_PROFILE_ALLOC(ptr, size)` /
`BRACKEN_PROFILE_FREE(ptr)` (`src/bracken/core/profiler.hpp`), which map to
Tracy's `TracyAlloc`/`TracyFree` when built with `BRACKEN_ENABLE_TRACY=ON`
(CMake option, default OFF) and compile to `((void)0)` otherwise. Individual
bump allocations are not instrumented — only chunk-level events appear in
Tracy's memory view.

---

## GPU Abstraction Layer

Module: `src/bracken/gpu/`, namespace `bracken::gpu`. A thin RAII layer over SDL3 GPU (`SDL_gpu.h`) plus the engine's pipeline cache and the CPU↔GPU uniform contract. No render-pass orchestration lives here (that is `scene::SceneRenderer` and the `render/` sub-renderers); this module owns device lifetime, resource creation/upload, shader cross-compilation, pipeline state objects, and layout contracts.

File map:

| File | Contents |
|---|---|
| `src/bracken/gpu/gpu_device.hpp/.cpp` | `GPUDevice` — device/context, frame cycle, swapchain, raw factories |
| `src/bracken/gpu/gpu_buffer.hpp/.cpp` | `GPUBuffer` (+ nested `Mapping`) — static/dynamic buffers |
| `src/bracken/gpu/gpu_texture.hpp/.cpp` | `GPUTexture`, `SamplerConfig`, `GPUSampler` |
| `src/bracken/gpu/gpu_shader.hpp/.cpp` | `ShaderStage`, `ShaderResources`, `GPUShader`, `ShaderManager` |
| `src/bracken/gpu/gpu_pipeline.hpp/.cpp` | `PipelineConfig`, `GPUPipeline` |
| `src/bracken/gpu/gpu_compute.hpp/.cpp` | `ComputeResources`, `GPUComputePipeline`, `ComputePass` |
| `src/bracken/gpu/pipeline_catalog.hpp/.cpp` | `PipelineType` (67 values), `ComputePipelineType` (16), `PipelineDesc`/`ComputePipelineDesc` + the catalogue table, `VertexLayout`, `TargetFormat`, `PipelineFormats`, `build_pipeline_config` |
| `src/bracken/gpu/pipeline_registry.hpp/.cpp` | `PipelineRegistry` — cache, pre-creation contract, invalidation, hot reload |
| `src/bracken/gpu/gpu_uniforms.hpp` | CPU mirrors of every HLSL cbuffer + storage-buffer records |
| `src/bracken/gpu/gpu_types.hpp` | Vertex formats, `TextureFormat`, `BlendMode`, vertex-attribute helpers |
| `src/bracken/gpu/gpu_debug.hpp/.cpp` | `GPUDebugGroup`, `GPUTimestampPool` (CPU record timings; GPU timings unavailable), `GPUTimerScope`, macros |
| `src/bracken/gpu/hlsl_packing.hpp` | `straddles_register` (consteval) + the `BRACKEN_HLSL_CBUFFER`/`BRACKEN_HLSL_STRUCT`/`BRACKEN_HLSL_FIELD` static-assert macros used after every struct in `gpu_uniforms.hpp` |
| Tests | `gpu_types.test.cpp`, `gpu_uniforms.test.cpp` (cross-module DEFAULT values only — layout is `static_assert`ed in the header), `uniform_layout.test.cpp` (CPU structs vs the compiled SPIR-V), `gpu_pipeline.test.cpp`, `gpu_compute.test.cpp`, `gpu_logging.test.cpp`, `gpu_resource_lifetime.test.cpp` (create/release failure paths; no GPU required) |

### GPUDevice (`src/bracken/gpu/gpu_device.hpp`)

Purpose: owns the `SDL_GPUDevice*` and claimed `SDL_Window*`; provides the command-buffer acquire/submit cycle and null-checked pass-through factories.

Ownership/lifetime: non-copyable, **non-movable** (other GPU objects hold raw `GPUDevice*` back-pointers; moving would invalidate them). One per application. Destructor calls `shutdown()`.

API surface:

```cpp
[[nodiscard]] bool init(SDL_Window* window, bool prefer_low_power = false);
void shutdown();
bool is_initialized() const;

SDL_GPUCommandBuffer* begin_frame();
static void end_frame(SDL_GPUCommandBuffer* cmd);
[[nodiscard]] bool end_frame_and_wait(SDL_GPUCommandBuffer* cmd);
SDL_GPUTexture* acquire_swapchain_texture(SDL_GPUCommandBuffer* cmd,
                                          uint32_t* out_width = nullptr,
                                          uint32_t* out_height = nullptr);

SDL_GPUBuffer*           create_buffer(const SDL_GPUBufferCreateInfo&);
SDL_GPUTransferBuffer*   create_transfer_buffer(const SDL_GPUTransferBufferCreateInfo&);
SDL_GPUTexture*          create_texture(const SDL_GPUTextureCreateInfo&);
SDL_GPUSampler*          create_sampler(const SDL_GPUSamplerCreateInfo&);
SDL_GPUGraphicsPipeline* create_graphics_pipeline(const SDL_GPUGraphicsPipelineCreateInfo&);
SDL_GPUShader*           create_shader(const SDL_GPUShaderCreateInfo&);
void release_buffer/release_transfer_buffer/release_texture/release_sampler/
     release_graphics_pipeline/release_shader(...);
void* map_transfer_buffer(SDL_GPUTransferBuffer*, bool cycle);
void unmap_transfer_buffer(SDL_GPUTransferBuffer*);

bool supports_present_mode(SDL_GPUPresentMode) const;
[[nodiscard]] bool set_swapchain_parameters(SDL_GPUPresentMode);
SDL_GPUDevice* handle() const;   SDL_Window* window() const;
int width() const;               int height() const;
SDL_GPUTextureFormat swapchain_format() const;
std::string driver_name() const;
bool supports_format(SDL_GPUTextureFormat, SDL_GPUTextureType, SDL_GPUTextureUsageFlags) const;
```

Behavior details (`src/bracken/gpu/gpu_device.cpp`):

- `init()` creates the device via the properties-based `SDL_CreateGPUDeviceWithProperties` path, requesting `SPIRV | METALLIB | DXIL` shader formats (no driver preference) with `debug_mode` and `SDL_PROP_GPU_DEVICE_CREATE_PREFERLOWPOWER_BOOLEAN = prefer_low_power`, then `SDL_ClaimWindowForGPUDevice`. Double-init returns `false`. **`prefer_low_power` is honored** — when set, SDL prefers the energy-efficient adapter (logged at init).
- Validation `debug_mode`: default `true` iff `NDEBUG` is not defined; env var `BRACKEN_GPU_DEBUG` overrides in both directions (truthy = non-empty and not `"0"`, `"false"`, `"off"` case-insensitive). Read via `std::getenv` before any worker thread exists.
- `shutdown()`: `SDL_WaitForGPUIdle` → release window → `SDL_DestroyGPUDevice`. Idempotent.
- `begin_frame()` = `SDL_AcquireGPUCommandBuffer`; returns `nullptr` (logged) on failure or if uninitialized.
- `end_frame(cmd)` = `SDL_SubmitGPUCommandBuffer` (fire-and-forget; null-tolerant; static).
- `end_frame_and_wait(cmd)` = `SDL_SubmitGPUCommandBufferAndAcquireFence` + `SDL_WaitForGPUFences` + `SDL_ReleaseGPUFence`. Full pipeline stall; used only by the screenshot readback path (`RenderContext::end_frame_and_wait`).
- `acquire_swapchain_texture` asserts main thread (`BRACKEN_ASSERT_MAIN_THREAD()`, SDL contract) and uses `SDL_WaitAndAcquireGPUSwapchainTexture` (the waiting variant, to avoid command-buffer accumulation). Returns `nullptr` both on error (logged) and on success-with-null-texture (window minimized/not ready) — callers cannot distinguish the two through this API.
- `set_swapchain_parameters(mode)` hardcodes `SDL_GPU_SWAPCHAINCOMPOSITION_SDR`; returns `false` + log when unsupported.
- `swapchain_format()` returns `SDL_GetGPUSwapchainTextureFormat`, or the fallback `SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM` when device/window is null.
- `width()/height()` use `SDL_GetWindowSize` (window coordinates, **not pixels**; contrast with `render::RenderContext::width()` which caches `SDL_GetWindowSizeInPixels`).
- All factory/release/map methods null-check `device_` (and the resource) and otherwise delegate 1:1 to SDL; factories return `nullptr` if uninitialized without logging.

### GPUBuffer (`src/bracken/gpu/gpu_buffer.hpp`)

Purpose: GPU buffer + transfer-buffer lifecycle for two patterns: upload-once (static) and per-frame update (dynamic).

Types: `enum class Type { Vertex, Index, Uniform, Storage, ComputeStorage }`. Usage mapping (`get_usage_flags`, `gpu_buffer.cpp`):

| Type | SDL usage flag |
|---|---|
| `Vertex` | `SDL_GPU_BUFFERUSAGE_VERTEX` |
| `Index` | `SDL_GPU_BUFFERUSAGE_INDEX` |
| `Uniform` | `SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ` |
| `Storage` | `SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ` |
| `ComputeStorage` | `SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ \| COMPUTE_STORAGE_WRITE \| GRAPHICS_STORAGE_READ` |

`Uniform` is intentionally a read-only storage buffer: SDL3 GPU delivers small uniform data via `SDL_PushGPU{Vertex,Fragment,Compute}UniformData`, not bindable UBOs. Engine code uses `Vertex`, `Index`, `Storage`, and `ComputeStorage` (the auto-exposure histogram/state buffers — compute read-write plus fragment-readable without a copy).

API:

```cpp
[[nodiscard]] static std::unique_ptr<GPUBuffer> create_static(GPUDevice&, Type, const void* data, size_t size);
[[nodiscard]] static std::unique_ptr<GPUBuffer> create_dynamic(GPUDevice&, Type, size_t size);
void update(SDL_GPUCommandBuffer* cmd, const void* data, size_t size, size_t offset = 0);
void update(SDL_GPUCopyPass* copy_pass, const void* data, size_t size, size_t offset = 0);
[[nodiscard]] Mapping map();
SDL_GPUBuffer* handle() const;  size_t size() const;  Type type() const;  bool is_dynamic() const;
```

Semantics:

- Non-copyable, movable (move releases the destination's prior resources first). Private constructor; factory-only.
- `create_static`: rejects null data / zero size (`nullptr`, logged). Creates the GPU buffer, a temporary `UPLOAD` transfer buffer, maps with `cycle=false`, memcpy, then acquires **its own command buffer**, records one copy pass (`SDL_UploadToGPUBuffer`, `cycle=false`), and submits immediately. The temporary transfer buffer is released right after submit (SDL defers actual destruction until GPU completion). Every intermediate failure is logged, cleans up, and returns `nullptr`.
- `create_dynamic`: allocates the GPU buffer and a **persistent** `UPLOAD` transfer buffer of the same size (`is_dynamic()` ⇔ `transfer_buffer_ != nullptr`). No initial contents.
- `update(SDL_GPUCommandBuffer*, ...)`: begins/ends its own copy pass and delegates to the copy-pass overload. Prefer the copy-pass overload to batch several updates into one pass.
- `update(SDL_GPUCopyPass*, ...)`: no-op with log if not dynamic, args invalid, or `offset + size > size_` (**silent drop**, buffer unchanged). Otherwise map transfer buffer with `cycle=true` (double-buffering; safe to rewrite while previous frame's copy is in flight), memcpy at `offset`, unmap, `SDL_UploadToGPUBuffer` with `cycle=true`. The same `offset` applies to both the transfer buffer and the GPU buffer region.
- `Mapping map()`: dynamic-only; maps the persistent transfer buffer with `cycle=true` and returns an RAII handle. Empty `Mapping` (falsy) on failure/non-dynamic. `Mapping::data()` is the write pointer; `Mapping::upload(copy_pass, size, offset = 0)` unmaps **before** recording the copy, bounds-checks (`offset + size > buffer->size_` drops with log), records `SDL_UploadToGPUBuffer` with `cycle=true`. If `upload` is never called, the destructor unmaps without recording. Move-only; must not outlive its `GPUBuffer`. Canonical consumer: `scene::SceneRenderer` instance/bone-palette packing (`src/bracken/scene/scene_renderer_upload.cpp`).
- Destructor releases transfer buffer then GPU buffer through the device.

Performance: dynamic `update`/`map` do zero heap allocation per call; `create_static` allocates and submits a private command buffer per call (do not call per frame). Growing a buffer = create a new `GPUBuffer` (legal inside an open copy pass — see `src/bracken/render/grass_renderer.cpp`); engine callers over-allocate ×2 on growth.

### GPUTexture, SamplerConfig, GPUSampler (`src/bracken/gpu/gpu_texture.hpp`)

`GPUTexture`: non-copyable, movable; private constructor; all factories `[[nodiscard]]` and return `nullptr` on failure (logged via `SDL_Log`). State: `width()`, `height()`, `format()` (SDL format), `layers()` (1 unless array; 6 for cubemaps), `mip_levels()`, `is_render_target()`, `is_depth()`, `handle()`.

Factories (`src/bracken/gpu/gpu_texture.cpp`):

| Factory | Type / format | Usage flags | Notes |
|---|---|---|---|
| `load_from_file(device, path, generate_mipmaps = true, format = TextureFormat::RGBA8)` | 2D, enum format | via `create_2d` | `IMG_Load` → `SDL_ConvertSurface(RGBA32)` → `create_2d`. Use `RGBA8_SRGB` for color art. |
| `create_2d(device, w, h, TextureFormat, pixels = nullptr, generate_mipmaps = false)` | 2D | `SAMPLER` (+`COLOR_TARGET` iff mipmaps) | Uploads iff `pixels`; own cmd buffer + copy pass + `SDL_GenerateMipmapsForGPUTexture` (outside the pass) + submit. |
| `create_2d(device, w, h, SDL_GPUTextureFormat, SDL_GPUTextureUsageFlags)` | 2D, raw SDL | caller-specified | 1 mip; sets `is_depth_` for D32F/D24S8/D16, `is_render_target_` from usage. No upload path. |
| `create_render_target(device, w, h, TextureFormat)` | 2D | `COLOR_TARGET \| SAMPLER` | 1 mip. |
| `create_depth(device, w, h)` | 2D `D32_FLOAT` | `DEPTH_STENCIL_TARGET \| SAMPLER` | Used for scene depth and the 2×2 CSM atlas (`shadow_map.cpp` creates `resolution*2` square). |
| `create_depth_stencil(device, w, h)` | 2D `D24_UNORM_S8_UINT` | `DEPTH_STENCIL_TARGET \| SAMPLER` | |
| `create_depth_array(device, w, h, layers)` | `2D_ARRAY` `D32_FLOAT` | `DEPTH_STENCIL_TARGET \| SAMPLER` | `is_render_target_ = true`. |
| `create_2d_array(device, w, h, layers, TextureFormat, layer_data = nullptr, generate_mipmaps = false)` | `2D_ARRAY` | `SAMPLER` (+`COLOR_TARGET` iff mipmaps) | One transfer buffer sized `layers * w * h * bpp`; per-layer null pointers skipped; one mipmap-gen call covers all layers; any upload failure returns `nullptr`. |
| `create_cube(device, size, SDL_GPUTextureFormat, SDL_GPUTextureUsageFlags, mipmaps = false)` | `CUBE`, raw SDL | caller-specified | 6 layers; with `COLOR_TARGET` every (face, mip) is individually renderable via `SDL_GPUColorTargetInfo::layer_or_depth_plane`/`mip_level`; sampled as `TextureCube`; `SDL_GenerateMipmapsForGPUTexture` fills the chain across all faces. Contents undefined until rendered/uploaded. Used by `render::IBL`. |

Methods:

- `upload(cmd, pixels, w, h)`: dimensions must equal the texture's (else logged drop), and formats with an unknown texel size are rejected with a log (no mis-sized transfer buffer). Allocates a **fresh transfer buffer per call**, records into a new copy pass on the caller's `cmd`, and releases the transfer buffer immediately after `SDL_EndGPUCopyPass` — an in-source comment flags that this relies on release semantics being deferred/synchronous; a per-frame staging pool is the suggested improvement for async use.
- `reload_from_file(path, generate_mipmaps = true)` (`[[nodiscard]] bool`): loads a fresh texture via `load_from_file` **preserving the creation format** (notably `RGBA8_SRGB` albedo art stays sRGB; a texture whose SDL format has no `TextureFormat` mapping fails with a log), steals its handle/metadata, releases the old `SDL_GPUTexture`. On load failure returns `false` and leaves the existing texture untouched. Raw handles cached by callers become stale on success.

Internals: `calculate_mip_levels(w, h) = floor(log2(max(w, h))) + 1`. `get_bytes_per_pixel` knows R8=1, R16=2, RGBA8/RGBA8_SRGB/BGRA8/D32F/D24S8=4, `R16G16B16A16_FLOAT`=8, and returns **0 for unknown formats** so upload paths fail loudly (logged rejection) instead of mis-sizing the transfer buffer and silently truncating.

In `create_2d`/`create_2d_array`, any upload failure after texture creation (unsupported texel size, transfer buffer, map, cmd, copy pass) logs and returns `nullptr` — never a texture with silently missing pixel data.

`SamplerConfig` (aggregate) defaults: `LINEAR` min/mag, `SAMPLERMIPMAPMODE_LINEAR`, `REPEAT` u/v/w, `mip_lod_bias = 0.0f`, `min_lod = 0.0f`, `max_lod = 1000.0f` (= `VK_LOD_CLAMP_NONE`; 0 would silently disable mipmapping on Vulkan), `max_anisotropy = 1.0f`, `enable_anisotropy = false`. Presets: `linear_repeat()`, `linear_clamp()`, `nearest_repeat()`, `nearest_clamp()`, `anisotropic(max_aniso = 16.0f)`.

`GPUSampler::create(device, config)` maps the config 1:1 onto `SDL_GPUSamplerCreateInfo` with `enable_compare = false` **always** — compare (shadow) samplers cannot be created through this wrapper. Non-copyable, movable, `handle()` accessor.

### Shaders (`src/bracken/gpu/gpu_shader.hpp`)

Compilation model: HLSL → SPIR-V at build time (CMake, DXC via SDL_shadercross tooling); at runtime `SDL_ShaderCross_CompileGraphicsShaderFromSPIRV` converts SPIR-V to the backend format (Vulkan: passthrough; Metal/D3D12: SPIRV-Cross transpile).

`enum class ShaderStage { Vertex, Fragment, Compute }` — `Compute` exists for API completeness only: SDL3 GPU has no `SDL_GPUShader` for compute (the module compiles directly into an `SDL_GPUComputePipeline`), so `GPUShader::create_from_spirv` rejects it with a log pointing at `GPUComputePipeline` (`gpu_compute.hpp`). No geometry/tessellation stages exist.

`struct ShaderResources { uint32_t num_samplers, num_storage_textures, num_storage_buffers, num_uniform_buffers; }` (all default 0, defaulted `operator==`). These feed `SDL_ShaderCross_GraphicsShaderResourceInfo` after being **validated against SPIR-V reflection** — a mismatch logs an error and the reflected counts win (see the Shader System chapter).

**`num_samplers` has a hard ceiling of 16 per stage**, and it is SDL's rather than DXC's: `SDL_CreateGPUShader` asserts `"Shader sampler count cannot be higher than 16!"` and returns null, so exceeding it is a missing pipeline and an unrendered scene, not a slow one. DXC legalizes 17 — that measurement is what retired the earlier belief in an 11-sampler limit — but SDL's number is the one that binds. The lit scene shaders are AT the ceiling (`model`/`model_instanced` 16, `terrain` 14, `grass` 12, all four with the local-shadow atlas last); anything else they need has to become a storage buffer, which is what R4b's screen-probe SH did.

`GPUShader`:

```cpp
[[nodiscard]] static std::unique_ptr<GPUShader> load_spirv(GPUDevice&, const std::string& path, ShaderStage,
                                                           const std::string& entry_point,
                                                           const ShaderResources& = {});
[[nodiscard]] static std::unique_ptr<GPUShader> create_from_spirv(GPUDevice&, const std::vector<uint8_t>& spirv,
                                                                  ShaderStage, const std::string& entry_point,
                                                                  const ShaderResources& = {});
[[nodiscard]] static bool reflect_spirv_resources(const std::vector<uint8_t>& spirv, ShaderResources& out);
SDL_GPUShader* handle() const;  ShaderStage stage() const;
const ShaderResources& resources() const;    // effective (baked) counts
[[nodiscard]] static bool init_compiler();   // SDL_ShaderCross_Init, idempotent
static void shutdown_compiler();             // SDL_ShaderCross_Quit
static bool is_compiler_available();
```

`create_from_spirv` lazily calls `init_compiler()`. Compiler state is a process-global `static bool s_compiler_initialized`. File reading is a private binary `ifstream` helper (empty vector + log on error). There is deliberately **no default entry point** — engine entry points are `VSMain`/`PSMain`, and a wrong entry point fails late and confusingly on Vulkan.

`ShaderManager` (recommended interface): per-device in-memory cache keyed by a collision-safe, length-prefixed string `"<len>:<path>|<stage-int>|<len>:<entry>"`.

```cpp
explicit ShaderManager(GPUDevice&);           // calls GPUShader::init_compiler(), failure logged
GPUShader* get(const std::string& path, ShaderStage, const std::string& entry_point, const ShaderResources& = {});
GPUShader* get_vertex(path, entry = "VSMain", resources = {});
GPUShader* get_fragment(path, entry = "PSMain", resources = {});
void clear_cache();
[[nodiscard]] bool reload(const std::string& path);   // evicts all entries for path; true if any removed
int reload_all();                                      // clears cache, returns prior count
```

Threading: the cache has no internal locking; loading/mutation is **main-thread tier** (Phase 1 of `docs/multithreading_plan.md` adds `BRACKEN_ASSERT_MAIN_THREAD` on the mutation paths). Lookups of already-loaded shaders are safe from jobs between init/hot-reload fences (the cache is immutable there). Returned `GPUShader*` are owned by the cache — invalidated by `clear_cache`/`reload*`/destruction. On a cache hit, `get` compares the requested `ShaderResources` against the cached shader's baked counts and logs an error on mismatch (the cached shader is returned).

### PipelineConfig / GPUPipeline (`src/bracken/gpu/gpu_pipeline.hpp`)

`PipelineConfig` — full pipeline state as a value type. Defaults (pinned by `gpu_pipeline.test.cpp`):

| Field | Default |
|---|---|
| `vertex_shader` / `fragment_shader` | `nullptr` (required) |
| `vertex_buffers` / `vertex_attributes` | empty (valid: fullscreen-triangle pipelines have no vertex input) |
| `primitive_type` | `SDL_GPU_PRIMITIVETYPE_TRIANGLELIST` |
| `fill_mode` / `cull_mode` / `front_face` | `FILL` / `CULLMODE_BACK` / `FRONTFACE_COUNTER_CLOCKWISE` |
| depth bias (`constant`/`clamp`/`slope`, `enable_depth_bias`) | 0 / 0 / 0, `false`; `enable_depth_clip = true` |
| `depth_test_enable` / `depth_write_enable` | `true` / `true` |
| `depth_compare_op` | **`SDL_GPU_COMPAREOP_GREATER_OR_EQUAL`** (reversed-Z scene convention: near→1, far→0, clear 0.0) |
| `stencil_test_enable` | `false` |
| `blend_mode` / `color_format` | `BlendMode::None` / `R8G8B8A8_UNORM` |
| `has_depth_target` / `depth_format` | `true` / `D32_FLOAT` |
| `sample_count` | `SDL_GPU_SAMPLECOUNT_1` |

Fluent helpers (all return `PipelineConfig&`): `with_vertex3d()`, `with_skinned_vertex()`, `with_vertex2d()` (install buffer desc + attributes from `gpu_types.hpp`); `opaque()` (None blend, depth test+write on); `alpha_blended()` / `additive()` (respective blend, depth test on, **write off**); `no_depth()` (test/write off **and** `has_depth_target = false`); `no_cull()`; `cull_front()`; `with_depth_bias(constant, slope, clamp = 0.0f)` (sets `enable_depth_bias = true`).

`GPUPipeline::create(device, config)` (`gpu_pipeline.cpp`):

- Requires both shaders (`nullptr` + log otherwise).
- `multisample_state.sample_mask = 0` (required by SDL3 GPU).
- `config.color_format == SDL_GPU_TEXTUREFORMAT_INVALID` ⇒ zero color targets (depth-only pipeline); otherwise exactly one color target with `get_blend_state(config.blend_mode)`.
- Depth-stencil target attached iff `has_depth_target`.
- Vertex buffer/attribute vectors are only read during the call (SDL copies); the config need not outlive it.

`GPUPipeline`: non-copyable, movable; `bind(SDL_GPURenderPass*)` = `SDL_BindGPUGraphicsPipeline` (null-tolerant no-op); `handle()`; destructor releases via device.

### GPUComputePipeline / ComputePass (`src/bracken/gpu/gpu_compute.hpp`)

`struct ComputeResources { uint32_t num_samplers, num_readonly_storage_textures, num_readonly_storage_buffers, num_readwrite_storage_textures, num_readwrite_storage_buffers, num_uniform_buffers; }` (all default 0, defaulted `operator==`) — read-only vs read-write are distinct because SDL3 GPU compute puts them in different binding sets.

`GPUComputePipeline` (non-copyable, movable; factory-only):

```cpp
[[nodiscard]] static std::unique_ptr<GPUComputePipeline> load_spirv(GPUDevice&, const std::string& path,
                                                                    const std::string& entry_point,
                                                                    const ComputeResources& = {});
[[nodiscard]] static std::unique_ptr<GPUComputePipeline> create_from_spirv(GPUDevice&, const std::vector<uint8_t>&,
                                                                           const std::string& entry_point,
                                                                           const ComputeResources& = {});
[[nodiscard]] static bool reflect_spirv_resources(const std::vector<uint8_t>&, ComputeResources& out,
                                                  uint32_t out_threadcount[3] = nullptr);
SDL_GPUComputePipeline* handle() const;  const ComputeResources& resources() const;
uint32_t threadcount_x/y/z() const;      uint32_t group_count_x/y/z(uint32_t items) const;  // ceil-div
```

Semantics: `create_from_spirv` REQUIRES reflection (`SDL_ShaderCross_ReflectComputeSPIRV`) to succeed — the `[numthreads]` threadgroup only exists in the module, so reflection failure fails creation (log + `nullptr`), unlike graphics shaders which degrade to declared counts. Declared-vs-reflected mismatch logs the same `declared resource counts do not match SPIR-V` error and the reflected counts win. The pipeline is built by `SDL_ShaderCross_CompileComputePipelineFromSPIRV` with the reflected metadata. Destructor releases via `GPUDevice::release_compute_pipeline`.

`ComputePass` (RAII, move-only; destructor ends a still-open pass): `static ComputePass begin(cmd, span<const SDL_GPUStorageTextureReadWriteBinding>, span<const SDL_GPUStorageBufferReadWriteBinding>)` — read-write storage bindings are declared at pass begin per SDL3 GPU's model; each binding's `cycle` flag matters (false = preserve contents; the auto-exposure buffers always pass false). Falsy on `SDL_BeginGPUComputePass` failure (logged); all members are null-tolerant no-ops then. `bind_pipeline(const GPUComputePipeline&)`, `bind_samplers(first_slot, span<const SDL_GPUTextureSamplerBinding>)` (set 0), `bind_storage_textures`/`bind_storage_buffers` (read-only, set 0 after the sampled textures), `push_uniforms(slot, data, size)` (= `SDL_PushGPUComputeUniformData` on the command buffer), `dispatch(gx, gy, gz)`, `dispatch_for(pipeline, items_x, items_y = 1, items_z = 1)` (ceil-div by the pipeline's threadgroup), `end()` (idempotent).

### PipelineRegistry (`src/bracken/gpu/pipeline_registry.hpp`)

Purpose: cached creation of every engine pipeline — **pre-created at init** as of Phase 1 of `docs/multithreading_plan.md` (the registry knows the full set; `get_pipeline` is a read-only lookup in the steady state, and creation/hot-reload paths are main-thread-asserted — this deletes the old first-touch lazy-creation race instead of locking it); single owner of the `ShaderManager`; shader hot-reload hook. Non-copyable/non-movable. Owned by `scene::SceneRenderer` (`pipeline_registry()` accessor).

`enum class PipelineType` — 56 values + `Count` (declared in `gpu/pipeline_catalog.hpp`):
`Model`, `SkinnedModel`, `Terrain`, `Skybox`, `Grid`, `UI`, `Text`, `Particle`, `Grass`, `InstancedModel`, `InstancedShadowModel`, `ShadowModel`, `ShadowSkinnedModel`, `ShadowTerrain`, `LocalShadowClear`, `GTAO`, `GTAOTemporal`, `GTAOUpsample`, `BlurAO`, `RtReflectionResolve`, `Composite`, `BloomDownsample`, `BloomUpsample`, `VolumetricFog`, `Fxaa`, `TaaResolve`, `TaaSharpen`, `SkyCapture`, `IblSpecularPrefilter`, `IblIrradiance`, `IblBrdfLut`, `ModelDoubleSided`, `ModelBlend`, `ModelBlendDoubleSided`, `SkinnedModelDoubleSided`, `SkinnedModelBlend`, `SkinnedModelBlendDoubleSided`, `InstancedModelDoubleSided`, the 8 `DepthPrepass*` types, the 8 `*DepthEqual` forward variants, the 2 `DepthPrepass*Masked` types, their 10 `*GBuffer` twins, and `Water`. `pipeline_type_to_string()` returns the type's catalogue row name.

`enum class ComputePipelineType` — 16 values + `Count`: `ExposureHistogram`, `ExposureAdapt`, `WaterSim`, the four `RtShadow*` passes, `RtReflection`/`RtReflectionTemporal`, `RtLocalShadow`, the four `GiProbe*` passes, and `HzbCopy`/`HzbReduce` (`compute_pipeline_type_to_string()` for debug names; like the graphics enum it and its `ComputePipelineDesc` rows live in `gpu/pipeline_catalog.hpp`). Compute pipelines live in a sibling cache (`get_compute_pipeline(type)` → `GPUComputePipeline*`, lazy + cached, uncached failures retried) and share the graphics lifecycle: `preload_all_pipelines()` iterates both enums, `invalidate_all()`/`reload_all_pipelines()`/hot-reload clear both caches, `cached_pipeline_count()` counts both. Declared `ComputeResources` are validated against `SDL_ShaderCross_ReflectComputeSPIRV` at creation (same "declared resource counts do not match SPIR-V" error contract as graphics; see `gpu_compute.hpp` below).

API:

```cpp
[[nodiscard]] bool init(GPUDevice&);      // false if already initialized
void shutdown();                           // clears pipelines, destroys ShaderManager
GPUPipeline* get_pipeline(PipelineType);   // read-only lookup after init pre-creation; nullptr on failure
[[nodiscard]] bool preload_all_pipelines();// iterates 0..Count; false if any failed
size_t cached_pipeline_count() const;
void invalidate_all();                     // clears pipeline cache AND shader cache
void set_swapchain_format(SDL_GPUTextureFormat);
SDL_GPUTextureFormat get_swapchain_format() const;
SDL_GPUTextureFormat get_scene_color_format() const;
SDL_GPUTextureFormat get_velocity_format() const;  // prepass motion-vector target
void enable_hot_reload(core::asset::FileWatcher&);
void reload_all_pipelines();               // invalidate_all + preload_all_pipelines
const std::string& shader_path() const;
```

Formats (member initializers, `pipeline_registry.hpp`):

| Constant | Value |
|---|---|
| `swapchain_format_` | `SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM` default; overwritten from `device.swapchain_format()` in `init()` |
| `scene_color_format_` | `SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT` (linear HDR scene target) |
| `depth_format_` | `SDL_GPU_TEXTUREFORMAT_D32_FLOAT` |
| `velocity_format_` | `SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT` (motion-vector target written by the `DepthPrepass*` pipelines; `get_velocity_format()`) |
| AO pipelines' color target | `SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM` (GTAO/BlurAO/GTAOUpsample creators — AO + octahedral bent normal + cone aperture; GTAOTemporal writes an `R16G16B16A16_FLOAT` history plus an `R16G16_FLOAT` depth history) |

Shader path resolution (`init()`): `SDL_GetBasePath() + "shaders/"`, fallback `"shaders/"` (logged). This matches `bracken_copy_shaders()` (defined in `src/bracken/CMakeLists.txt`), which stages compiled `.spv` next to the consumer executable.

`set_swapchain_format(format)`: no-op if unchanged; otherwise erases only the swapchain-dependent pipelines — a `static constexpr` list of `{UI, Text, Composite, Fxaa}`. All other pipelines target fixed offscreen formats and stay cached.

Hot-reload: `enable_hot_reload(watcher)` registers `watcher.watch_directory(shader_path_, ".spv", callback)` once (subsequent calls no-op); the callback runs `reload_all_pipelines()` — coarse (all 53 graphics + 2 compute pipelines rebuilt on any `.spv` change) but correct; previous pipelines are released by SDL after in-flight command buffers finish. Requires `init()` first (warn + no-op otherwise). The engine wires this in `Application` when `BRACKEN_HOT_RELOAD=1`.

Per-pipeline configuration summary (from the catalogue rows in `pipeline_catalog.cpp`; shader files relative to `shader_path()`; "resources" = `ShaderResources` as `{uniform_buffers, samplers, storage_buffers}`):

| Pipeline | Shaders (VS / FS) | VS res | FS res | Vertex input | Key state |
|---|---|---|---|---|---|
| Model (+DoubleSided/Blend variants) | `model.vert.spv` / `model.frag.spv` | {1,0,0} | {3,12,4} | Vertex3D | opaque or alpha-blend; blend ⇒ depth write off; double-sided ⇒ no cull; HDR color, D32 |
| SkinnedModel (+variants) | `skinned_model.vert.spv` / `model.frag.spv` | {1,0,1} | {3,12,4} | SkinnedVertex | as Model; VS storage buffer = packed bone palettes |
| InstancedModel (+DoubleSided) | `model_instanced.vert.spv` / `model_instanced.frag.spv` | {1,0,1} | {3,12,4} | Vertex3D | opaque; VS storage = `InstanceData` |
| Terrain | `terrain.vert.spv` / `terrain.frag.spv` | {1,0,0} | {3,10,4} | custom: float3 pos, float2 uv, float4 color, float3 normal, pitch 48 B | opaque |
| Skybox | `skybox.vert.spv` / `skybox.frag.spv` | {0,0,0} | {1,0,0} | float3 position only | no cull, depth write off, `GREATER_OR_EQUAL` (draws at z = 0.0 where nothing rendered) |
| Grid | `grid.vert.spv` / `grid.frag.spv` | {1,0,0} | {0,0,0} | float3 pos + float4 color, pitch 28 B | alpha blend, no cull, `LINELIST` |
| UI | `ui.vert.spv` / `ui.frag.spv` | {1,0,0} | {1,1,0} | Vertex2D | alpha blend, no depth, no cull, **swapchain format** |
| Text | `text.vert.spv` / `text.frag.spv` | {1,0,0} | {1,1,0} | float2 pos + float2 uv, pitch 16 B | alpha blend, no depth, no cull, **swapchain format** |
| Particle | `particle.vert.spv` / `particle.frag.spv` | {2,0,1} | {1,1,0} | Vertex3D | additive (`SRC_ALPHA, ONE`; shader writes alpha 1.0 ⇒ effective `ONE, ONE`), `GREATER_OR_EQUAL`, depth write off, HDR |
| Grass | `grass.vert.spv` / `grass.frag.spv` | {1,2,1} | {3,8,4} | Vertex3D | opaque, no cull |
| ShadowModel | `shadow_depth.vert.spv` / `shadow_depth.frag.spv` | {1,0,0} | {1,1,0} | Vertex3D | depth-only (`color_format = INVALID`), cull FRONT, bias (1.25, 1.75), `enable_depth_clip = false` (caster pancaking), **`LESS_OR_EQUAL`** (cascades stay standard-Z) |
| ShadowSkinnedModel | `shadow_skinned.vert.spv` / `shadow_depth_opaque.frag.spv` | {1,0,1} | {0,0,0} | SkinnedVertex | as ShadowModel |
| ShadowTerrain | `shadow_terrain.vert.spv` / `shadow_depth_opaque.frag.spv` | {1,0,0} | {0,0,0} | terrain layout | as ShadowModel but **no cull** |
| LocalShadowClear | `fullscreen.vert.spv` / `shadow_depth_opaque.frag.spv` | {0,0,0} | {0,0,0} | none (SV_VertexID triangle) | depth-only, `ALWAYS` compare + write; drawn per local-shadow tile with viewport depth range [1,1] |
| InstancedShadowModel | `shadow_depth_instanced.vert.spv` / `shadow_depth.frag.spv` | {1,0,1} | {1,1,0} | Vertex3D | as ShadowModel |
| GTAO | `fullscreen.vert.spv` / `gtao.frag.spv` | {0,0,0} | {1,1,0} | none (SV_VertexID triangle) | no depth, no cull, `R8G8B8A8_UNORM` target |
| GTAOTemporal | `fullscreen.vert.spv` / `gtao_temporal.frag.spv` | {0,0,0} | {1,5,0} | none | MRT: `R16G16B16A16_FLOAT` history (AO + bent oct + aperture) + `R16G16_FLOAT` depth history |
| GTAOUpsample | `fullscreen.vert.spv` / `gtao_upsample.frag.spv` | {0,0,0} | {1,2,0} | none | `R8G8B8A8_UNORM` (full-res AO + bent cone the forward pass samples) |
| BlurAO | `fullscreen.vert.spv` / `blur_ao.frag.spv` | {0,0,0} | {1,2,0} | none | `R8_UNORM` |
| RtReflectionResolve | `fullscreen.vert.spv` / `rt_reflection_resolve.frag.spv` | {0,0,0} | {1,3,0} | none | additive RGB with destination alpha preserved; `R16G16B16A16_FLOAT` |
| Water | `water.vert.spv` / `water.frag.spv` | {1,1,0} | {1,5,0} | float2 world XZ | opaque/no-cull; refraction + SSR/RT/IBL reflection; HDR color, D32 |
| Composite | `fullscreen.vert.spv` / `composite.frag.spv` | {0,0,0} | {1,4,1} | none | **swapchain format**; the FS storage buffer is the auto-exposure state SSBO (or `PostProcessStack`'s zeroed dummy) |
| BloomDownsample | `fullscreen.vert.spv` / `bloom_downsample.frag.spv` | {0,0,0} | {1,1,0} | none | `R16G16B16A16_FLOAT` |
| BloomUpsample | `fullscreen.vert.spv` / `bloom_upsample.frag.spv` | {0,0,0} | {1,1,0} | none | additive onto larger mip (`LOADOP_LOAD` at pass level), `R16G16B16A16_FLOAT` |
| VolumetricFog | `fullscreen.vert.spv` / `volumetric_fog.frag.spv` | {0,0,0} | {1,4,0} | none | `R16G16B16A16_FLOAT`; depth + shadow atlas + GI irradiance/probe-metadata (no visibility atlas — `GI_SAMPLE_VOLUME_ONLY`) |
| Fxaa | `fullscreen.vert.spv` / `fxaa.frag.spv` | {0,0,0} | {1,1,0} | none | **swapchain format** |
| SkyCapture | `fullscreen.vert.spv` / `sky_capture.frag.spv` | {0,0,0} | {1,0,0} | none | `R16G16B16A16_FLOAT`, targets one capture-cube face |
| IblSpecularPrefilter | `fullscreen.vert.spv` / `ibl_prefilter.frag.spv` | {0,0,0} | {1,1,0} | none | `R16G16B16A16_FLOAT`, targets one specular-cube (face, mip) |
| IblIrradiance | `fullscreen.vert.spv` / `ibl_irradiance.frag.spv` | {0,0,0} | {1,1,0} | none | `R16G16B16A16_FLOAT`, targets one irradiance-cube face |
| IblBrdfLut | `fullscreen.vert.spv` / `ibl_brdf_lut.frag.spv` | {0,0,0} | {0,0,0} | none | `R16G16_FLOAT`, rendered once at init |

Compute pipelines (`ComputeResources` as `{samplers, ro_storage_tex, ro_storage_buf, rw_storage_tex, rw_storage_buf, uniform_buffers}`; threadgroup reflected from `[numthreads]`):

| Compute pipeline | Shader | Resources | Threadgroup | Notes |
|---|---|---|---|---|
| ExposureHistogram | `exposure_histogram.comp.spv` | {1,0,0,0,1,1} | 16×16×1 | sampler = HDR scene color; RW buffer = 256-bin histogram |
| ExposureAdapt | `exposure_adapt.comp.spv` | {0,0,0,0,2,1} | 256×1×1 | RW buffers = histogram (read + zeroed) and exposure state |
| RtLocalShadow | `rt_local_shadow.comp.spv` | {3,0,8,4,5,2} | 8×4×1 | froxel-RIS local-light shadows. Samplers = depth, prepass G-buffer, velocity. **Eight** RO buffers is SDL3 GPU's hard ceiling for a compute pipeline, not a choice: four acceleration-structure buffers + four cluster buffers (light data, froxel offsets, froxel indices, prev→current remap). `cluster_compute.hlsli`'s fifth, the packed stable ids, is opt-in (`CLUSTER_COMPUTE_WANT_STABLE_IDS`) and this pass declines it. The reservoir ping-pong therefore rides two RW buffers of which one is only ever read, and for the same reason so do R6's emissive records, §5.8's **`ShadowCapsule` set** and R8's **per-froxel emissive candidate table** (`u8`; a flat `uint` block of `[count][record x 16]` per froxel, 235 KB, built by `render::rt::EmissiveClusterGrid` against the same froxel bounds the light binner uses; a 1-block stand-in and `emissiveFroxelShare = 0` on any frame it is not this frame's) — five RW buffers, four of them read-only in the shader. Two uniform buffers: b0 `RtLocalShadowUniforms`, b1 ClusterGrid's own `ClusterParams`. R7b's direction target is the FOURTH RW storage texture. **Dispatched twice** (R7a): `stage` 0 resamples, `stage` 1 does the pairwise-MIS spatial merge, the ray and the DI block — one pipeline, different reservoir buffers in the two RW slots. One group per 8×4 output tile |
| RtShadow | `rt_shadow.comp.spv` | {2,0,5,1,0,1} | 8×4×1 | traced sun shadows. Samplers = depth, prepass G-buffer. **Five** RO buffers: the four acceleration-structure buffers plus §5.8's `ShadowCapsule` set — this pass has headroom under SDL3's eight, unlike `RtLocalShadow`, so the capsules bind read-only here and to a read-write slot there. One group per 8×4 tile, packing 32 pixels of visibility into one uint |
| RtShadowGradient / RtLocalShadowGradient | `rt_shadow_gradient.comp.spv` / `rt_local_shadow_gradient.comp.spv` | {4,1,5,1,0,1} / {3,1,8,1,3,2} | 8×8×1 | the R5b A-SVGF replays. Each binds the **same** capsule buffer and count its trace bound: a replay that cannot see the trace's occluders reports a full gradient on every capsule-shadowed stratum, forever, which is a permanent history reset exactly on the character |

Error handling: `get_pipeline` before `init` logs and returns `nullptr`; unknown enum values log; per-pipeline creation failure (usually a missing/uncompiled `.spv`) returns `nullptr` and is not cached (retried on next access).

### Uniform contract (`src/bracken/gpu/gpu_uniforms.hpp`)

Every HLSL cbuffer has exactly one `alignas(16)` CPU mirror struct; layouts must match **byte-for-byte**. Data is delivered per draw/pass with `SDL_PushGPUVertexUniformData(cmd, slot, ptr, size)` / `SDL_PushGPUFragmentUniformData(...)` / `SDL_PushGPUComputeUniformData(...)`, where `slot` is the `b<N>` register within the stage's uniform set.

SDL3 GPU binding order (from `src/bracken/shaders/README.md`, expressed as `[[vk::binding(slot, set)]]` in HLSL): **vertex** — set 0 = sampled textures, storage textures, storage buffers; set 1 = uniform buffers. **Fragment** — set 2 = sampled textures, storage textures, storage buffers; set 3 = uniform buffers. **Compute** — set 0 = sampled textures, READ-ONLY storage textures, READ-ONLY storage buffers; set 1 = READ-WRITE storage textures, READ-WRITE storage buffers (bound at `SDL_BeginGPUComputePass`); set 2 = uniform buffers.

HLSL packing rules that the mirrors encode: 16-byte registers; a `float3` followed by a scalar packs into one register, two consecutive `float3`s do not; padding must be **named scalars**, never arrays (an HLSL `float pad[N]` strides at 16 B/element). The first two rules are machine-checked: a `BRACKEN_HLSL_CBUFFER(T, size)` + per-field `BRACKEN_HLSL_FIELD(T, field, offset)` block follows each struct in `gpu_uniforms.hpp` (macros in `gpu/hlsl_packing.hpp`), pinning `alignof == 16`, `sizeof % 16 == 0`, every offset, and `!straddles_register(offset, size)` — all as `static_assert`, so drift is a compile error. Agreement with the shaders is checked at test time by `src/bracken/gpu/uniform_layout.test.cpp`, which parses the compiled `.spv` files.

Cbuffer mirror inventory (binding as annotated in the header). Every size below is machine-checked twice: `static_assert` in `gpu_uniforms.hpp`, and against the compiled SPIR-V in `src/bracken/gpu/uniform_layout.test.cpp`.

| Struct | Binding | Size | Notable fields / defaults |
|---|---|---|---|
| `ModelTransformUniforms` | vertex set 1, b0 | 288 | `model/view/projection/normalMatrix`, `cameraPos`, `paletteOffset` at offset 272 + 12-byte pad (index into the frame's bone-palette SSBO; read only by `skinned_model.vert.hlsl`) |
| `ModelLightingUniforms` | fragment set 3, b0 | 304 | hemisphere ambient (`ambientSky`/`ambientGround`), fog params, PBR factors (`roughnessFactor = 0.85`), `alphaCutoff = 0` (0 disables MASK alpha test), `screenAOTexelX/Y` (SV_Position → screen-AO UV), `iblEnabled` (sky-IBL vs hemisphere ambient), `screenAOEnabled` (0 for MASK/BLEND — no background-AO bleed onto foliage), the RT-reflection handover window, and R1c's probe-field block (`giCascadeBase[3]` + `giEnabled`) |
| `SkyboxFragmentUniforms` | fragment set 3, b0 | 192 | `invViewProjection`, sun/zenith/horizon/ground/fog colors + cloud coverage/density/scale + haze strength (linear, from `bracken::render::EnvironmentState`) |
| `SkyCaptureUniforms` | fragment set 3, b0 | 176 | IBL sky-capture face basis (`faceForward/Right/Up`) + the same atmosphere/cloud fields as the skybox; `time` is the sky-time snapshot at refresh start so all six faces render one cloud state |
| `IblFilterUniforms` | fragment set 3, b0 | 48 | face basis + `roughness` (specular prefilter mip; ignored by irradiance) + `sourceSize` (mip-bias) |
| `PrepassModelUniforms` | vertex set 1, b0 | 400 | depth-prepass model/skinned uniforms: jittered `projection` + UNJITTERED curr/prev view-projections, `prevModel`, `paletteOffset`/`prevPaletteOffset` (`NO_PREV_PALETTE = 0xFFFFFFFF` ⇒ reuse current palette) |
| `PrepassCameraUniforms` | vertex set 1, b0 | 256 | shared by the instanced + terrain prepass shaders: jittered `projection` + unjittered curr/prev view-projections (grass keeps its own `GrassUniforms`, byte-identical to the forward one) |
| `TaaResolveUniforms` | fragment set 3, b0 | 176 | `feedbackMax = 0.9`, `varianceGamma = 1.0`, `historyValid` 0 on gaps; curr-inverse/prev unjittered view-projections for the depth-reprojection fallback |
| `TaaSharpenUniforms` | fragment set 3, b0 | 16 | `sharpenStrength = 0.25` (0 = pass-through); applied to the resolved output only, never the history |
| `GridVertexUniforms` | vertex set 1, b0 | 64 | `viewProjection` |
| `UIScreenUniforms` | vertex set 1, b0 | 16 | `width`, `height` |
| `UIFragmentUniforms` | fragment set 3, b0 | 16 | `hasTexture` + 3 named int pads |
| `InstancedCameraUniforms` | vertex set 1, b0 | 144 | shared by `model_instanced.vert.hlsl` and `particle.vert.hlsl` |
| `InstancedLightingUniforms` | fragment set 3, b0 | 288 | `ModelLightingUniforms` minus `tintColor` (tint is per-instance) |
| `InstancedShadowUniforms` | vertex set 1, b0 | 64 | `lightViewProjection` |
| `ShadowTransformUniforms` | vertex set 1, b0 | 128 | `lightViewProjection`, `model` |
| `ShadowSkinnedTransformUniforms` | vertex set 1, b0 | 144 | + `paletteOffset` |
| `ShadowTerrainUniforms` | vertex set 1, b0 | 64 | terrain verts already world-space |
| `ShadowFragUniforms` | fragment set 3, b0 | 16 | `hasTexture` (alpha-tested casters) |
| `ShadowDataUniforms` | fragment set 3, b1 | 336 | per-cascade `lightViewProjection[4]`, splits/texel/depth-range vec4s; `shadowEnabled`: 0 = off, 1 = PCF, 2 = PCSS; 2×2 atlas, cascade i at quadrant (i%2, i/2) |
| `GTAOUniforms` | fragment set 3, b0 | 240 | defaults `radius = 4.0` (world units), `falloffRange = 0.615`, `occluderThickness = 1.0` (SSILVB slab depth in world units, replaced `thinOccluderCompensation = 0.2`), `maxDistance = 12000`; `radiusScale` = `ao_radius`, `invView` = view→world for the world-space bent normal, `temporalRotation/Offset` = per-frame golden-ratio jitter |
| `GtaoTemporalUniforms` | fragment set 3, b0 | 32 | `feedbackMax = 0.9`, `depthRejectScale = 10`; `historyValid` 0 on first frame/resize/mode gaps |
| `GtaoUpsampleUniforms` | fragment set 3, b0 | 32 | joint-bilateral upsample constants + `strength` (`ao_strength`, applied as a power curve) |
| `BlurUniforms` | fragment set 3, b0 | 32 | `sharpness = 40`; `depthA = proj[2][2]`, `depthB = proj[3][2]` linearize depth (`viewZ = |B/(depth+A)|`, valid for standard and reversed-Z); `maxDistance` must equal `GTAOUniforms::maxDistance` (asserted in tests) |
| `RtReflectionResolveUniforms` | fragment set 3, b0 | 112 | inverse view-projection, camera position, full/trace extents, depth linearization constants, roughness gate and intensity for the full-resolution material-guided HDR resolve |
| `CompositeUniforms` | fragment set 3, b0 | 112 | `bloomStrength = 0.25` (tracks `GraphicsSettings::bloom_strength`, the source of truth), `exposure = 1` (manual bias), `tonemapMode = 0`, `contrast = 1`, `saturation = 1`; `lowResTexelSize`/`depthA`/`depthB` drive the joint-bilateral fog upsample (0 ⇒ plain bilinear); `autoExposureEnabled = 0` gates the exp2(EV) multiply from the exposure-state SSBO; `whiteBalanceEnabled = 0` + `whiteBalance = (1,1,1)` gate the pre-tonemap LMS white balance; `cdlEnabled = 0` + `cdlSlope/Offset/Power = (1,0,1)` gate the post-tonemap ASC-CDL — all-neutral defaults are pixel-identical to the ungraded path; AO is no longer composited here |
| `FxaaUniforms` | fragment set 3, b0 | 16 | `invScreenSize`; operates on post-tonemap LDR |
| `ExposureHistogramUniforms` | **compute** set 2, b0 | 16 | `width`/`height` (scene extent), `minLog2Luma = −10`, `invLog2Range = 1/26` — the 256-bin log2-luminance mapping (CPU mirror `render/auto_exposure.hpp::exposure_bin_from_luma`) |
| `ExposureAdaptUniforms` | **compute** set 2, b0 | 48 | `minLog2Luma = −10`, `log2Range = 26`, `deltaTime`, `firstFrame` (≠0 ⇒ snap), `speedUp = 1.5`/`speedDown = 2.5` EV/s, `minEv = −4`/`maxEv = 4`, `lowPercentile = 0.60`/`highPercentile = 0.05`, `log2TargetGray = −1.80` (`AutoExposure::DEFAULT_LOG2_TARGET_GRAY`) |
| `ParticleFragmentUniforms` | fragment set 3, b0 | 16 | `emissiveBoost = 1`; pushed once per pass |
| `ParticleBatchUniforms` | vertex set 1, b1 | 16 | `instanceOffset` into per-frame particle SSBO |
| `RtReflectionUniforms` | compute set 2, b0 | 672 | since R3 the **first 336 bytes are `ShadowDataUniforms` field for field** (same arrangement, and same argument, as `GiProbeTraceUniforms`: the shader declares them under the names `shadow_common.hlsli` requires and `rt_hit_shade.hlsli` calls the forward pass's own `calcShadow()` at the hit); then `invViewProj` at 336, camera position/forward at 400/416 (a hit needs a view depth to pick a cascade), the roughness gate as `maxRoughness`/`gateFadeStart` (**not** `fadeStart` — that name now belongs to the cascade distance fade), sun direction/colour at 432/448 with the colour ALREADY multiplied by `sun_intensity`, the `ambientSky`/`ambientGround`/`useSkyCube` miss fallback at 464–495, `bounceScale` at 492, `giEnabled`/`giAmbientFallback` at 496/500, all three cascades' `giCascadeBase[3]` at 512, the optional analytic-water region/simulation/flow/detail fields at 560–655, and `outputSize`/`tlasNodeCount`/`frameIndex`/light counts at 640–671 |
| `RtShadowUniforms` | compute set 2, b0 | 128 | trace fields through offset 111, then `sunCosThetaMax` (cos of the sun's angular radius; 1.0 = a point light, i.e. the old hard shadow), `frameIndex`, `hasGBuffer`, `shadowCapsuleCount` (§5.8; took the old `_pad0` slot at 124, so the extent is unchanged — 0 collapses the capsule loop and reproduces the pre-capsule trace) |
| `ShadowCapsule` | — (`contracts/shadow_capsule.hpp`) | **32** | not a cbuffer but a CPU↔GPU POD under the same rule, mirrored in `shaders/rt_trace.hlsli`. `{vec3 p0; float radius; vec3 p1; float _pad0;}` — two `float4` registers, `static_assert`ed at 32 B with every offset pinned. The radius rides in p0's `w` because two consecutive `float3`s do not pack. Semantics: the set of points within `radius` of the segment p0..p1; `p0 == p1` is a legal sphere |
| `RtLocalShadowUniforms` | compute set 2, b0 | 160 | `invViewProj`, `outputSize`, `tlasNodeCount`, `normalBias`/`depthBias`, `lightSourceRadius = 10` world units (the emitter sphere the shadow ray samples — the penumbra-width dial, one value for every light until R6), `historyMaxMRatio = 20`, `frameIndex`, `hasGBuffer`, `historyValid`, `candidateCount = 6`. **No `depthA`/`depthB` pair**, unlike every other depth-reading pass: the froxel lookup needs an ABSOLUTE view depth that agrees with the CPU binner's, and a reversed-Z linearization that is only ever used for comparisons elsewhere is off by enough to land the pixel in a different Z slice — so the depth comes from `clusterParams.view`, the matrix `ClusterGrid` binned against. The pass takes ClusterGrid's own `ClusterParams` at **set 2, b1**, pushed verbatim rather than copied into a second struct. R7a appended `stage` (0 = resample, 1 = spatial reuse + ray + DI), `spatialNeighbors = 3` (0 reproduces the pre-R7a sampling bit for bit) and `spatialRadius = 8` traced texels |
| `RtShadowClassifyUniforms` | compute set 2, b0 | 16 | `tileCount` of the packed-bit texture |
| `RtShadowTemporalUniforms` | compute set 2, b0 | 128 | `tracedSize`/`tileCount`, `depthA`/`depthB` (reversed-Z linearization), `maxAccumFrames = 32`, `depthRejectRelative = 0.05`, `historyValid`, `prevClipFromClip` (the disocclusion test is applied in the PREVIOUS frame's clip space, so it measures the surface and not the camera) |
| `RtShadowFilterUniforms` | compute set 2, b0 | 48 | `outputSize` (full res) / `tracedSize` / `tileCount`, `depthA`/`depthB`, `maxRadius = 3` traced texels at zero accumulated history, `depthSigma = 0.05`, `normalPower = 8`, `hasGBuffer` |
| `RtReflectionTemporalUniforms` | compute set 2, b0 | 240 | R3's virtual-hit-point reprojection needs three matrices: `invViewProj` at 0 (reconstructs the RECEIVER from the depth buffer, so it carries whatever jitter the depth was rasterized with) and the **unjittered** current/previous view-projections at 64/128, whose difference for one world point is a motion vector in exactly the velocity buffer's units; then `cameraPosition`/`rayMaxDistance` at 192/204, `tracedSize` at 208, `maxAccumFrames = 16` and `clampSigma = 1.5` (neighbourhood clamp width) at 216/220, `historyValid` at 224 and `virtualValid` at 228 (0 falls back to pure surface motion — the pre-R3 behaviour) |
| `GiProbeTraceUniforms` | **compute** set 2, b0 | 528 | the **first 336 bytes are `ShadowDataUniforms` field for field**, under exactly the names `shadow_common.hlsli` requires, so the probe trace `#include`s that file and calls the same `calcShadow()` the forward pass calls instead of reimplementing the standard-Z cascade math; then camera position/forward (a hit needs a view depth to pick a cascade), `rayMaxDistance`, `probeSpacing`, `maxOffset` (0.45 × spacing), sun direction/colour, `bounceScale = 1`, the `ambientSky`/`ambientGround` miss fallback with the `useSkyCube` flag, all three cascades' `cascadeBase[3]` (a ray fired from cascade 2 can land inside cascade 0 and takes its bounce from the finest field covering it), the ray/probe/light counts + `frameIndex`, and (R3) `rt_hit_shade.hlsli`'s two ambient flags at 512/516, pinned to `giEnabled = 1` / `giAmbientFallback = 0` — a probe already collects sky through its ray misses, so a hemisphere added where the field has no answer would double-count it |
| `GiProbeBlendUniforms` | **compute** set 2, b0 | 32 | shared by `gi_probe_blend` and `gi_probe_blend_visibility`: `probeCount`, `rayCount`, `distanceClamp` (1.5 × spacing, visibility only), `rayMaxDistance`, `frameIndex` — the last is load-bearing, the blends **re-derive** each ray's direction from it to weight that ray's radiance, so it must be the value the trace ran with |
| `GiProbeClassifyUniforms` | **compute** set 2, b0 | 32 | `probeCount`, `rayCount`, `cascadeIndex` (which counter slot to accumulate into), `maxOffset`, `probeSpacing`, `frameIndex` |
| `HzbCopyUniforms` | **compute** set 2, b0 | 16 | `mipWidth`/`mipHeight` — mip 0 of the Hi-Z pyramid is the depth buffer's size, so one extent serves both source and destination; two named trailing pads round the block to 16 |
| `HzbReduceUniforms` | **compute** set 2, b0 | 16 | `dstWidth`/`dstHeight`, `srcWidth`/`srcHeight`. Both extents are sent rather than the destination alone: the SOURCE's parity decides whether the last destination texel gathers two or three texels along an axis, and re-deriving `src = dst * 2` in the shader gets exactly the edge the pyramid must not drop wrong |
| `GiScreenProbePlaceUniforms` | **compute** set 2, b0 | 112 | R4b's screen-probe placement: `invViewProj` (the JITTERED draw matrix — it unprojects the depth buffer, and the unjittered one would offset every anchor by the jitter's world equivalent in a direction that changes every frame), camera position, `frameIndex`, `probeGrid`, `screenSize`, `missCapacity` |
| `GiScreenProbeMarchUniforms` | **compute** set 2, b0 | 192 | the Hi-Z march: `viewProj` (jittered — the depth buffer's) at 0 and `prevViewProjUnjittered` at 64 (the pair the velocity buffer and the TAA resolve were built from, so a hit's reprojection into the history agrees with how the history was resolved), camera position, the `depthA`/`depthB` linearization pair, `probeGrid`/`screenSize`, `rayMaxDistance` (the SCREEN ray's range), `normalBias`, `historyValid` (0 sends every ray straight to the BVH pass — there is no other source of screen radiance), `frameIndex`, `mipCount`, `missCapacity` |
| `GiScreenProbeArgsUniforms` | **compute** set 2, b0 | 16 | `missCapacity` and `threadsPerGroup` — the latter is the BVH pass's `[numthreads].x`, sent rather than duplicated as a literal in a shader computing another shader's workgroup count |
| `GiScreenProbeTraceUniforms` | **compute** set 2, b0 | 512 | the handed-off BVH rays. Like `GiProbeTraceUniforms`, the **first 336 bytes are `ShadowDataUniforms` field for field** so the pass includes `shadow_common.hlsli` and calls the forward pass's own `calcShadow()`; then camera position/forward, `rayMaxDistance` = 600 (the CLAMP, not the screen march's range), `normalBias`, sun direction/colour, `bounceScale`, the hemisphere fallback + `useSkyCube`, `rt_hit_shade.hlsli`'s two ambient flags (`giAmbientFallback` pinned to 1 here, unlike the probe field's 0 — a ray that escapes the clamp outside the field's coverage has to be lit by something), `cascadeBase[3]`, the TLAS/light/miss counts, `probeGrid`, `frameIndex` and `horizonBlend` |
| `GiScreenProbeFilterUniforms` | **compute** set 2, b0 | 32 | probe-space temporal: `probeGrid`, `screenSize`, `alpha = 0.1`, `historyValid`, `fireflyMax = 8` (a multiple of the probe's OWN mean luminance, so the clamp survives the day/night cycle's ~10³ swing), `frameIndex` |
| `GiScreenProbeProjectUniforms` | **compute** set 2, b0 | 16 | `probeGrid`, `normalPower = 8` (the 3×3 spatial filter's normal agreement exponent), `frameIndex` |
| `BloomDownsampleUniforms` | fragment set 3, b0 | 32 | `threshold = 0.8` compared against **post-exposure** luminance; `isFirstPass = 1`, `exposure = 1` |
| `BloomUpsampleUniforms` | fragment set 3, b0 | 16 | `bloomRadius = 1` |
| `VolumetricFogUniforms` | fragment set 3, b0 | 544 | per-cascade matrices/bias for god rays; defaults `fogDensity = 0.0015`, `scatterStrength = 0.25`, `fogHeight = 100`, `fogFalloff = 0.005`, `startDist = 100`, `maxDist = 3000`, `godRaysEnabled = 1` |
| (clustered lighting) | fragment set 3, b2 | 192 | defined by `render::lighting::ClusterParams` (`light_cluster.hpp`) — `SceneRenderer` pushes that struct directly; `gpu_uniforms.hpp` deliberately carries no mirror |

Storage-buffer record structs (tight std430-style packing, no `alignas(16)`, no cbuffer padding rules):

| Struct | Size | Consumer |
|---|---|---|
| `InstanceData` | 224 (`model` + `normalMatrix` + `tint` + `noFog` + 3 pads + `prevModel`) | `model_instanced.vert.hlsl` + `prepass_instanced.vert.hlsl` instance SSBO (`prevModel` = previous-frame model matrix; == `model` for statics) |
| `ShadowInstanceData` | 64 (`model`) | `shadow_depth_instanced.vert.hlsl` |
| `ParticleInstanceData` | 80 (`model` + `color`; `color.a` = opacity folded into rgb by the VS — additive blending) | `particle.vert.hlsl` |
| `GiScreenProbeHeader` | 32 (`worldPosition` + `flags` + `normal` + `cameraDistance`) | R4b's probe record, written by `gi_screen_probe_place.comp` and read by the four passes after it; `flags` carries the validity bit AND the anchor's in-tile pixel offset (four bits per axis, exactly a 16-pixel tile), so the temporal filter reprojects the pixel the anchor actually sat on and not the tile centre |
| `GiScreenProbeMiss` | 16 (`probeIndex` + `texelIndex` + `resumeDistance` + pad) | the compacted handoff buffer. Records start at byte 16 — the first 16 bytes are a header whose word 0 is the record COUNT, which lives there rather than in the counters buffer because `gi_screen_probe_trace.comp` sits at SDL3 GPU's eight read-only storage buffers exactly |
| `GiScreenProbeSh` | 64 (`shR` + `shG` + `shB` + `meta`) | the published L1 spherical harmonics, one per probe: three coefficient quads (one per colour channel) and one metadata quad (anchor normal + camera distance; `w == 0` means "no probe here"). Bound as the scene shaders' FIFTH fragment storage buffer, because SDL3 GPU caps a stage at 16 samplers and they are already there |
| `GiScreenProbeCounters` | 16 (`screenHits` + `worldHits` + `fallbacks` + `dropped`) | R4b's diagnostics, read back through the same fenced/polled download `ProbeVolume` uses. `dropped` should always be 0 — the miss buffer is sized for every traced ray missing |

Skinned palettes: there is **no GPU-side bone-count constant**; palettes are packed in the `FramePacket` at build time and uploaded by `SceneRenderer` in `palettes()` order into one per-frame `StructuredBuffer<float4x4>` (vertex set 0, binding 0) indexed by `paletteOffset` (= `BonePalette::matrix_offset`). `MAX_BONES = 128` caps only the CPU-side arrays (`animation/animation_types.hpp`); `MAX_BONE_PALETTES = 1024` (`scene/frame_packet.hpp`) caps palettes per frame, enforced at build time in `alloc_bone_palette`.

### Shared types (`src/bracken/gpu/gpu_types.hpp`)

Vertex formats (layouts pinned by `gpu_types.test.cpp`):

| Struct | Size | Fields (offset) | Attribute helper |
|---|---|---|---|
| `Vertex3D` | 64 | position vec3 (0), normal vec3 (12), texcoord vec2 (24), color vec4 (32), tangent vec4 (48; w = glTF bitangent sign) | `get_vertex3d_attributes()` — 5 attrs, locations 0–4, all FLOATn |
| `SkinnedVertex` | 84 | Vertex3D fields + `uint8_t joints[4]` (64, `UBYTE4`), `float weights[4]` (68, FLOAT4) | `get_skinned_vertex_attributes()` — 7 attrs |
| `Vertex2D` | 32 | position vec2 (0), texcoord vec2 (8), color vec4 (16) | `get_vertex2d_attributes()` — 3 attrs |
| `GrassInstance` | 24 | position vec3, rotation, scale, color_variation | none — **unreferenced outside this header** |

All attributes bind buffer slot 0; `get_*_buffer_desc()` returns `{slot 0, pitch = sizeof(struct), VERTEXINPUTRATE_VERTEX, step 0}`.

`enum class TextureFormat` → `to_sdl_format()` mapping: `RGBA8`→`R8G8B8A8_UNORM`, `RGBA8_SRGB`→`R8G8B8A8_UNORM_SRGB`, `BGRA8`→`B8G8R8A8_UNORM`, `R8`→`R8_UNORM`, `R16`→`R16_UNORM` (heightmaps), `D32F`→`D32_FLOAT`, `D24S8`→`D24_UNORM_S8_UINT`; default case falls back to `R8G8B8A8_UNORM`.

`enum class BlendMode` → `get_blend_state()` (all modes write full RGBA):

| Mode | Color (src, dst, op) | Alpha (src, dst, op) |
|---|---|---|
| `None` | blending disabled | — |
| `Alpha` | `SRC_ALPHA`, `ONE_MINUS_SRC_ALPHA`, ADD | `ONE`, `ONE_MINUS_SRC_ALPHA`, ADD |
| `Additive` | `SRC_ALPHA`, `ONE`, ADD | `ONE`, `ONE`, ADD |
| `Multiply` | `DST_COLOR`, `ZERO`, ADD | `DST_ALPHA`, `ZERO`, ADD |

Note `Additive` is `(SRC_ALPHA, ONE)`, not `(ONE, ONE)`; the particle and bloom-upsample shaders write alpha 1.0 to make it effectively `(ONE, ONE)` (the `PipelineType::Particle` doc comment's "ONE/ONE additive" describes this effective behavior).

Legacy: the `CameraUniforms`, `ModelUniforms`, `LightUniforms`, `TimeUniforms` structs in this header (commented "std140 compatible") are referenced nowhere else in the repository; the live uniform contract is `gpu_uniforms.hpp`.

### Debug tooling (`src/bracken/gpu/gpu_debug.hpp`)

`GPUDebugGroup(cb, name)` — RAII `SDL_PushGPUDebugGroup`/`SDL_PopGPUDebugGroup`; null `cb`/`name` degrades to a no-op. Non-copyable/non-movable. Macro `BRACKEN_GPU_SCOPE(cb, name)` instantiates one with a line-unique identifier. `insert_debug_label(cb, name)` / `BRACKEN_GPU_LABEL` wraps `SDL_InsertGPUDebugLabel`.

`GPUTimestampPool` — CPU-record timings are real (`core::PassTimeline` behind `write_begin`/`write_end`); the GPU half is **absent by necessity**: SDL3 GPU (as of SDL 3.4) exposes no per-pass timestamp queries, so `is_supported()` is false and `PassTiming::gpu_ms` is `nullopt`. The API shape is preserved so call sites remain identical when support lands.

```cpp
[[nodiscard]] bool init(GPUDevice&, size_t pair_count);   // stores capacity, supported_ = false, returns true
void shutdown();
int reserve_pair(const char* pass_name);                  // -1 if name null or capacity exhausted
void write_begin(SDL_GPUCommandBuffer*, int);             // no-op
void write_end(SDL_GPUCommandBuffer*, int);               // no-op
void read_results(std::vector<std::pair<std::string, float>>& ms_out);  // names in reserve order, all 0.0f
void begin_frame();                                       // clears the name list
void end_frame(SDL_GPUCommandBuffer*);                    // no-op (no resolve)
bool is_supported() const noexcept;                       // always false currently
```

Lifetime contract: `pass_name` is stored as a raw `const char*` (no per-frame string allocation) and must outlive the frame — every `BRACKEN_GPU_TIMER` call site passes a string literal. `GPUTimerScope` / `BRACKEN_GPU_TIMER(pool, cb, name)` is the RAII begin/end wrapper. `SceneRenderer` initializes the pool with `pair_count = 64` and brackets every major pass; consequently any per-pass GPU timings surfaced in `RenderStats` are placeholders (zeros). Use Tracy (CPU) or RenderDoc/Nsight (GPU) for real timings.

### Threading and lifetime contracts (module-wide)

- One `GPUDevice`; it must outlive every `GPUBuffer`/`GPUTexture`/`GPUSampler`/`GPUShader`/`GPUPipeline`/`PipelineRegistry` created from it (their destructors call `device_->release_*`). `GPUDevice`, `PipelineRegistry`, `ShaderManager`, `RenderContext` are non-copyable/non-movable; the resource wrappers are non-copyable but movable.
- Swapchain acquisition is main-thread-only (asserted). The `ShaderManager` cache and `PipelineRegistry` are not synchronized; all GPU-layer calls are expected on the render/main thread. `GPUShader::s_compiler_initialized` is unsynchronized process-global state.
- `GPUBuffer::Mapping` must not outlive its buffer; a mapped buffer must not be updated through other paths until unmapped.
- Dynamic-buffer updates rely on SDL `cycle=true` semantics for intra-frame safety; static-buffer creation submits its own command buffer immediately.

### Error handling conventions

This module predates the `core::Result` convention: everything is `bool`/`nullptr` + `SDL_Log` (or `BRACKEN_LOG_*` in the registry's hot-reload path). Factories return null `unique_ptr`; `init()`-style methods return `bool`; per-frame mutators (`GPUBuffer::update`, `GPUTexture::upload`) fail as logged no-ops (silent drops from the caller's perspective). Per the house rules these stay as-is until touched.

### Known limitations, stubs, and silent behaviors

- `GPUTimestampPool` returns zeroed durations; `is_supported()` is always false (SDL3 limitation).
- `GPUBuffer::update` / `Mapping::upload` / `GPUTexture::upload` drop out-of-bounds or mismatched requests with only a log.
- `GPUSampler::create` cannot create compare samplers (`enable_compare` hardwired false).
- `GPUTexture::upload` releases its temporary transfer buffer immediately after recording (before submit); the in-source comment flags the deferred-release assumption and suggests a staging pool for async submission paths.
- Pipeline creation failures are retried on every `get_pipeline` call (not negatively cached) — a missing `.spv` logs once per access.

---

## Shader Authoring & Compilation

Technical specification of Bracken's shader system: the build-time HLSL→SPIR-V pipeline (`src/bracken/shaders/CMakeLists.txt`), the runtime shader layer (`src/bracken/gpu/gpu_shader.{hpp,cpp}`), pipeline construction (`src/bracken/gpu/gpu_pipeline.{hpp,cpp}`, `src/bracken/gpu/pipeline_registry.{hpp,cpp}`), the SDL3 GPU binding-order contract, and the CPU↔GPU uniform contract (`src/bracken/gpu/gpu_uniforms.hpp`).

### Source inventory and naming contract

`src/bracken/shaders/` contains 49 compilable `.hlsl` files and 8 `.hlsli` includes (`pbr.hlsli`, `lighting.hlsli`, `cluster_lighting.hlsli`, `clusters.hlsli`, `shadow_common.hlsli`, `fog.hlsli`, `sky_common.hlsli`, `grass_blade.hlsli`). Stage and entry point are derived purely from the filename:

- `{name}.vert.hlsl` → stage `vertex`, entry `VSMain`, output `{name}.vert.spv`
- `{name}.frag.hlsl` → stage `fragment`, entry `PSMain`, output `{name}.frag.spv`
- `{name}.comp.hlsl` → stage `compute`, entry `CSMain`, output `{name}.comp.spv`
- any other `*.hlsl` → `message(WARNING "Unknown shader type: ..., skipping")` and `continue()` — a configure-time warning, not an error.

There is no geometry/tessellation support anywhere in the chain. Compute modules do not become `SDL_GPUShader`s: they load through `gpu::GPUComputePipeline` (which compiles them straight into `SDL_GPUComputePipeline`s), never through `GPUShader`/`ShaderManager`.

Interpolator semantics are `TEXCOORD0..N` throughout (SDL3 GPU D3D12-backend assumption, per `src/bracken/shaders/README.md`); vertex inputs carry `[[vk::location(n)]]` matching the pipeline vertex-attribute locations, and some fragment inputs (e.g. `ui.frag.hlsl`) also use explicit `[[vk::location]]` to match the paired `VSOutput`.

### Build-time compilation — `src/bracken/shaders/CMakeLists.txt`

#### SDL_shadercross fetch

`FetchContent` clones `https://github.com/libsdl-org/SDL_shadercross.git` at pinned commit `7b7365a86611b2a7b6462e521cf1c43a037d0970` with recursive submodules (vendored DXC + SPIRV-Tools). Cache options forced before `MakeAvailable`: `SDLSHADERCROSS_CLI=ON`, `SDLSHADERCROSS_SHARED=OFF`, `SDLSHADERCROSS_STATIC=ON`, `SDLSHADERCROSS_INSTALL=OFF`, `SDLSHADERCROSS_VENDORED=ON`, `SDLSHADERCROSS_SPIRVCROSS_SHARED=ON`, `SDLSHADERCROSS_DXC=ON`, plus test suppression for the vendored deps (`SPIRV_SKIP_TESTS`, `SPIRV_SKIP_EXECUTABLES`, `SKIP_SPIRV_TOOLS_INSTALL`, `SPIRV_WERROR=OFF`, `LLVM_INCLUDE_TESTS=OFF`). `CMAKE_POSITION_INDEPENDENT_CODE=ON` is set as a plain directory-scoped variable — it covers the fetched dependencies without leaking into a consuming super-project's cache.

This provides two artifacts: the `shadercross` CLI (build-time tool) and the `SDL3_shadercross::SDL3_shadercross-static` library, which the `bracken` target links `PUBLIC` (`src/bracken/CMakeLists.txt`) for the runtime SPIR-V→backend conversion. `${SDL_shadercross_SOURCE_DIR}/include` is a PUBLIC include directory of `bracken`.

#### CLI selection

`SHADERCROSS_COMMAND = "$<TARGET_FILE:shadercross>"` — the freshly built, pinned CLI is always used, with the `shadercross` target in every custom command's `DEPENDS`. A system `shadercross` on `PATH` is deliberately ignored (no `find_program`), so the build-time compiler can never version-skew against the runtime SDL3_shadercross library.

#### Per-shader custom commands

- Output directory: `set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/spv")` — a dedicated `.spv`-only directory, `build/src/bracken/shaders/spv/` in this repo. Exported to the parent scope and cached as `BRACKEN_SHADER_OUTPUT_DIR` (INTERNAL) in `src/bracken/CMakeLists.txt`. At configure time, orphaned `.spv` files (no matching `.hlsl` source) are removed from it.
- Sources: `file(GLOB_RECURSE ... CONFIGURE_DEPENDS "*.hlsl")`; includes: `file(GLOB_RECURSE ... CONFIGURE_DEPENDS "*.hlsli")`. `CONFIGURE_DEPENDS` re-runs CMake when the file set changes, so adding/removing a shader needs only an incremental build.
- One `add_custom_command` per shader:

```bash
shadercross <src>.hlsl -s HLSL -d SPIRV -t {vertex|fragment} -e {VSMain|PSMain} \
    -I <src/bracken/shaders> -o <SHADER_OUTPUT_DIR>/<name>.{vert|frag}.spv
```

- `DEPENDS "${SHADER_SOURCE}" ${SHADER_INCLUDES} ${SHADERCROSS_DEPENDS}` — **every `.hlsli` is a dependency of every shader**, so touching any include recompiles all 49 shaders (20 `.vert` + 27 `.frag` + 2 `.comp`). There is no per-shader include scanning.
- `add_custom_target(compile_shaders ALL DEPENDS ${COMPILED_SHADERS})`; `src/bracken/CMakeLists.txt` adds `add_dependencies(bracken compile_shaders)` so shaders always build before/with the engine.

#### Staging — `bracken_copy_shaders(target)`

The function definition (an `ALL` custom target running `cmake -E rm -rf` on `$<TARGET_FILE_DIR:target>/shaders` followed by `cmake -E copy_directory` from `${BRACKEN_SHADER_OUTPUT_DIR}`, ordered after `compile_shaders`, with the consumer target depending on the copy) is reproduced in [Getting Started](#getting-started-building--integrating-bracken) §"`bracken_copy_shaders(target)` — the consumer staging function", along with the runtime shader-path lookup contract it satisfies.

Semantics: the copy target is `ALL` with no `OUTPUT`, so it runs on **every build** (cheap remove + directory copy). Because the staging directory is deleted before the copy, removed/renamed shaders never linger as stale `.spv` files next to the executable. The only in-tree consumer is `examples/playground/CMakeLists.txt` (`bracken_copy_shaders(playground)`).

### Runtime shader layer — `src/bracken/gpu/gpu_shader.{hpp,cpp}`

#### `ShaderResources`

```cpp
struct ShaderResources {
    uint32_t num_samplers = 0;
    uint32_t num_storage_textures = 0;
    uint32_t num_storage_buffers = 0;
    uint32_t num_uniform_buffers = 0;
    friend bool operator==(const ShaderResources&, const ShaderResources&) = default;
};
```

Forwarded into `SDL_ShaderCross_GraphicsShaderResourceInfo` — but first **validated against SPIR-V reflection**: `create_from_spirv` reflects the bytecode via `SDL_ShaderCross_ReflectGraphicsSPIRV` (`GPUShader::reflect_spirv_resources`) and compares the caller-declared counts. On mismatch it logs an error (`declared resource counts do not match SPIR-V`, listing declared/reflected per category) and **the reflected counts win**, so a stale hand-maintained count can no longer silently break rendering. `GPUShader::resources()` returns the effective (baked) counts. A `[[vk::combinedImageSampler]]` texture+sampler pair counts as one sampler.

#### `GPUShader`

Move-only RAII wrapper over `SDL_GPUShader*`; destructor calls `device_->release_shader(shader_)`.

Key API (`gpu_shader.hpp`):

- `[[nodiscard]] static std::unique_ptr<GPUShader> load_spirv(GPUDevice&, const std::string& path, ShaderStage, const std::string& entry_point, const ShaderResources& = {})` — reads the file (binary, whole-file; empty/missing file → log + `nullptr`), then delegates to `create_from_spirv`. **No default entry point** (deliberate — a wrong entry point fails late and confusingly on Vulkan; the build emits `VSMain`/`PSMain`).
- `[[nodiscard]] static std::unique_ptr<GPUShader> create_from_spirv(GPUDevice&, const std::vector<uint8_t>& spirv, ShaderStage, const std::string& entry_point, const ShaderResources& = {})` — lazily calls `init_compiler()` if needed, reflects the SPIR-V and reconciles the declared `ShaderResources` (error log + reflected counts on mismatch), fills `SDL_ShaderCross_SPIRV_Info` (`shader_stage` mapped to `SDL_SHADERCROSS_SHADERSTAGE_VERTEX/FRAGMENT`), and calls `SDL_ShaderCross_CompileGraphicsShaderFromSPIRV(device.handle(), &spirv_info, &resource_info, 0)`. Failure → `SDL_Log` with `SDL_GetError()` + `nullptr`.
- `handle() -> SDL_GPUShader*`, `stage() -> ShaderStage`, `resources() -> const ShaderResources&` (effective baked counts).
- Compiler lifecycle: `static bool init_compiler()` (`SDL_ShaderCross_Init`, idempotent via `static bool s_compiler_initialized`), `shutdown_compiler()` (`SDL_ShaderCross_Quit`), `is_compiler_available()`. Note the state is a process-global static bool — no reference counting.

Backend behavior: Vulkan consumes the SPIR-V directly; Metal/D3D12 are transpiled at load time via SPIRV-Cross (inside SDL_shadercross).

#### `ShaderManager`

In-memory cache keyed by `make_cache_key(path, stage, entry_point)` — a length-prefixed string `"<len>:<path>|<stage int>|<len>:<entry>"` to avoid delimiter collisions. API:

- `get(path, stage, entry_point, resources)` / `get_vertex(...)` (entry default `VSMain`) / `get_fragment(...)` (entry default `PSMain`) — cache hit returns the cached raw pointer, first verifying the requested `ShaderResources` against the counts baked at first load (mismatch ⇒ error log naming both sets; the cached shader is still returned); miss loads via `GPUShader::load_spirv`, caches, returns raw pointer; `nullptr` + log on failure (failures are not cached — retried every call).
- `clear_cache()`; `[[nodiscard]] bool reload(path)` removes all cache entries whose key path matches (parses the length prefix with `std::stoull`; returns whether anything was removed); `int reload_all()` clears and returns the prior count. Neither eagerly reloads — reload happens on next `get`.
- Constructor calls `GPUShader::init_compiler()` (return value deliberately discarded — each later load fails and logs individually if init failed). Destructor clears the cache.
- Non-copyable, non-movable. Cache mutation (loading/reload/clear) is **main-thread tier** — no internal locking; asserted in debug per the Phase-1 threading work. Read-only lookups between init/hot-reload fences are job-safe.

Lifetime: returned `GPUShader*` are owned by the cache; `clear_cache()`/`reload*()` invalidates them. `PipelineRegistry` only uses them transiently during pipeline creation.

### Pipeline construction — `src/bracken/gpu/gpu_pipeline.{hpp,cpp}`

#### `PipelineConfig` defaults and fluent builders

The full `PipelineConfig` default table and the fluent builder inventory (`with_vertex3d()`, `opaque()`, `alpha_blended()`, `additive()`, `no_depth()`, `no_cull()`, `cull_front()`, `with_depth_bias(...)`, …) are specified in [GPU Abstraction Layer](#gpu-abstraction-layer) §"PipelineConfig / GPUPipeline"; defaults are pinned by `gpu_pipeline.test.cpp::PipelineConfigDefaults`. Each builder returns `*this` and later calls override earlier ones. The load-bearing default for shader authors is `depth_compare_op = SDL_GPU_COMPAREOP_GREATER_OR_EQUAL` (reversed-Z scene convention); empty `vertex_buffers`/`vertex_attributes` is valid and means no vertex input (fullscreen-triangle pipelines).

#### `GPUPipeline::create(GPUDevice&, const PipelineConfig&)`

Straight translation to `SDL_GPUGraphicsPipelineCreateInfo`. Details:

- Missing shaders → log + `nullptr`.
- `multisample.sample_mask = 0` (required by SDL3 GPU).
- `config.color_format == SDL_GPU_TEXTUREFORMAT_INVALID` ⇒ depth-only pipeline (`num_color_targets = 0`); otherwise exactly one color target with `get_blend_state(config.blend_mode)`.
- On `SDL_CreateGPUGraphicsPipeline` failure (via `device.create_graphics_pipeline`): log with `SDL_GetError()` + `nullptr`.
- `bind(SDL_GPURenderPass*)` no-ops on null pass or null pipeline; otherwise `SDL_BindGPUGraphicsPipeline`. Move-only; destructor releases via `device_->release_graphics_pipeline`.

#### `BlendMode` (`src/bracken/gpu/gpu_types.hpp`)

`None` (blend disabled), `Alpha` (color: SRC_ALPHA / ONE_MINUS_SRC_ALPHA; alpha: ONE / ONE_MINUS_SRC_ALPHA), `Additive` (color: SRC_ALPHA / ONE; alpha: ONE / ONE), `Multiply` (color: DST_COLOR / ZERO; alpha: DST_ALPHA / ZERO). All ops `ADD`; color write mask always RGBA. `Multiply` has no `PipelineConfig` builder and no registry user. Because `Additive` is (SRC_ALPHA, ONE), shaders using it (`particle.frag`, `bloom_upsample.frag`) write alpha 1.0 to make the effective color blend (ONE, ONE); fade is encoded by scaling rgb.

#### Vertex layouts (`src/bracken/gpu/gpu_types.hpp`)

All single-buffer, slot 0, `SDL_GPU_VERTEXINPUTRATE_VERTEX`, pitch `sizeof(T)`:

- `Vertex3D` — loc 0 `FLOAT3` position, 1 `FLOAT3` normal, 2 `FLOAT2` texcoord, 3 `FLOAT4` color, 4 `FLOAT4` tangent (xyz direction, w bitangent sign per glTF).
- `SkinnedVertex` — `Vertex3D` attributes + loc 5 `UBYTE4` joints (`uint8_t[4]`, read as `uint4 : BLENDINDICES` in HLSL), loc 6 `FLOAT4` weights.
- `Vertex2D` — loc 0 `FLOAT2` position, 1 `FLOAT2` texcoord, 2 `FLOAT4` color.

Custom layouts are named `gpu::VertexLayout` cases applied in `pipeline_catalog.cpp`: `Terrain` `pos3/uv2/color4/normal3` (pitch 48 B = 12 floats), `Grid` `pos3/color4` (pitch 28 B, `LINELIST` primitive), `Position3` (skybox, pitch 12 B), `Text` `pos2/uv2` (pitch 16 B), `Position2` (water grid, pitch 8 B), plus the position-only / position+texcoord / position+skin prepass subsets of the shared vertex formats.

### `PipelineRegistry` — `src/bracken/gpu/pipeline_registry.{hpp,cpp}`

#### Lifecycle & state

- `[[nodiscard]] bool init(GPUDevice&)` — rejects double-init (`false`); captures `device.swapchain_format()`; sets `shader_path_ = SDL_GetBasePath() + "shaders/"` (fallback: relative `"shaders/"` + log); constructs the owned `ShaderManager`. `shutdown()` clears pipelines then the shader manager (also run by the destructor). Non-copyable, non-movable.
- Formats (member defaults): `swapchain_format_ = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM` (overwritten from the device at `init`), `scene_color_format_ = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT` (HDR scene target), `depth_format_ = SDL_GPU_TEXTUREFORMAT_D32_FLOAT`. Accessors: `get_swapchain_format()`, `get_scene_color_format()`, `shader_path()`.
- `GPUPipeline* get_pipeline(PipelineType)` — `nullptr` + log if uninitialized; cache hit returns; otherwise dispatches a `switch` to the matching `create_*` method, logs success/failure, caches on success. `cached_pipeline_count()` exposes the cache size.
- `[[nodiscard]] bool preload_all_pipelines()` — iterates `0 .. PipelineType::Count`, returns `false` if **any** failed (but continues through all; each failure logged).
- `invalidate_all()` — clears the pipeline cache **and** the shader cache.
- `set_swapchain_format(fmt)` — no-op if unchanged; otherwise erases only the swapchain-dependent pipelines, which it derives from the catalogue (`pipeline_targets_swapchain(type)`, i.e. `desc.color == TargetFormat::Swapchain`) — today `{UI, Text, Composite, Fxaa}`, pinned by `gpu/pipeline_catalog.test.cpp`. Called each frame-ish by `SceneRenderer` (`scene/scene_renderer.cpp:120`). A new swapchain-targeting pipeline type needs no edit here: giving its row `TargetFormat::Swapchain` is enough.
- `enable_hot_reload(core::asset::FileWatcher&)` — warns if called pre-`init`; idempotent (ignores second watcher); `watch_directory(shader_path_, ".spv", cb)` where the callback logs and calls `reload_all_pipelines()` (= `invalidate_all()` + `preload_all_pipelines()`, warning on partial failure). `reload_all_pipelines()` is public for tests/diagnostics. `Application::init_renderer` wires this (plus model hot-reload) whenever the global `FileWatcher` is running, which `Application::init_engine` starts only when env `BRACKEN_HOT_RELOAD=1` (`src/bracken/application.cpp`). `FileWatcher` marshals callbacks onto the main thread via `poll_main_thread()` (run in `Application::run()`), so pipeline rebuilds safely touch GPU state.

#### `PipelineType` — 56 values

31 base types (`Model`, `SkinnedModel`, `Terrain`, `Skybox`, `Grid`, `UI`, `Text`, `Particle`, `Grass`, `InstancedModel`, `InstancedShadowModel`, `ShadowModel`, `ShadowSkinnedModel`, `ShadowTerrain`, `LocalShadowClear`, `GTAO`, `GTAOTemporal`, `GTAOUpsample`, `BlurAO`, `RtReflectionResolve`, `Composite`, `BloomDownsample`, `BloomUpsample`, `VolumetricFog`, `Fxaa`, `TaaResolve`, `TaaSharpen`, `SkyCapture`, `IblSpecularPrefilter`, `IblIrradiance`, `IblBrdfLut`) + 7 model-family material variants (`ModelDoubleSided`, `ModelBlend`, `ModelBlendDoubleSided`, `SkinnedModelDoubleSided`, `SkinnedModelBlend`, `SkinnedModelBlendDoubleSided`, `InstancedModelDoubleSided`) + 8 `DepthPrepass*` + 8 `*DepthEqual` variants + 2 masked-prepass variants + 10 G-buffer twins + `Water`, + `Count` (67 pipelines). The enum and its per-type data live in `gpu/pipeline_catalog.hpp`, not in the registry: each value has exactly one `PipelineDesc` row (shader pair, declared `ShaderResources`, `VertexLayout`, `TargetFormat`, cull/blend/depth state), `static_assert`ed to cover `Count`, and `pipeline_type_to_string()` is that row's `name`. Variants share their base row's shader pair and differ only in fixed-function state (`Blend` ⇒ `BlendMode::Alpha` with depth write off, host sorts back-to-front; `DoubleSided` ⇒ `SDL_GPU_CULLMODE_NONE`; `DepthEqual` ⇒ `SDL_GPU_COMPAREOP_EQUAL` with depth write off). glTF alphaMode MASK is *not* a pipeline variant for forward draws: it is the `alphaCutoff` field of the lighting cbuffer (0 disables the discard) — only the prepass has dedicated `*Masked` rows. There are no per-type convenience getters; everything goes through `get_pipeline(PipelineType)`.

#### Pipeline table (shaders, declared resources, key state)

VS/FS resource counts below are the `ShaderResources` declared in each catalogue row (`gpu/pipeline_catalog.cpp`, order: uniform buffers `u`, samplers `s`, storage buffers `b`); the table there is the source of truth and `gpu/pipeline_catalog.test.cpp` pins it against a captured golden. Color format: `scene` = RGBA16F, `swap` = swapchain, `R8` = R8_UNORM, `RGBA16F` = bloom/fog chain, `—` = depth-only (`INVALID`).

| Pipeline | VS / FS | VS res | FS res | State highlights | Color |
|---|---|---|---|---|---|
| Model (+DS/Blend) | `model.vert` / `model.frag` | u1 | u3 s16 b4 | vertex3d; opaque or alpha_blended; no_cull if DS | scene |
| SkinnedModel (+variants) | `skinned_model.vert` / `model.frag` | u1 b1 | u3 s16 b4 | skinned_vertex | scene |
| InstancedModel (+DS) | `model_instanced.vert` / `model_instanced.frag` | u1 b1 | u3 s16 b4 | vertex3d, opaque | scene |
| Terrain | `terrain.vert` / `terrain.frag` | u1 | u3 s14 b4 | custom 48-B layout | scene |
| Grass | `grass.vert` / `grass.frag` | u1 s2 b1 | u3 s12 b4 | vertex3d, opaque, no_cull | scene |
| Skybox | `skybox.vert` / `skybox.frag` | — | u1 | pos-only layout; depth write off; GREATER_OR_EQUAL; z=0 in VS | scene |
| Grid | `grid.vert` / `grid.frag` | u1 | — | 28-B layout, LINELIST, alpha, no_cull | scene |
| Particle | `particle.vert` / `particle.frag` | u2 b1 | u1 s1 | vertex3d, additive, depth write off, GREATER_OR_EQUAL | scene |
| UI | `ui.vert` / `ui.frag` | u1 | u1 s1 | vertex2d, alpha, no_depth, no_cull | swap |
| Text | `text.vert` / `text.frag` | u1 | u1 s1 | pos2/uv2 layout, alpha, no_depth | swap |
| ShadowModel | `shadow_depth.vert` / `shadow_depth.frag` | u1 | u1 s1 | cull_front, bias (1.25, 1.75), depth clip off, LESS_OR_EQUAL | — |
| InstancedShadowModel | `shadow_depth_instanced.vert` / `shadow_depth.frag` | u1 b1 | u1 s1 | same shadow state | — |
| ShadowSkinnedModel | `shadow_skinned.vert` / `shadow_depth_opaque.frag` | u1 b1 | — | skinned_vertex + shadow state | — |
| ShadowTerrain | `shadow_terrain.vert` / `shadow_depth_opaque.frag` | u1 | — | terrain layout, **no_cull**, shadow state | — |
| LocalShadowClear | `fullscreen.vert` / `shadow_depth_opaque.frag` | — | — | no vertex input, `ALWAYS` compare, depth write on | — |
| GTAO | `fullscreen.vert` / `gtao.frag` | — | u1 s1 | no vertex input, no_depth | RGBA8 |
| GTAOTemporal | `fullscreen.vert` / `gtao_temporal.frag` | — | u1 s5 | no_depth | RGBA16F + RG16F (MRT) |
| GTAOUpsample | `fullscreen.vert` / `gtao_upsample.frag` | — | u1 s2 | no_depth | RGBA8 |
| BlurAO | `fullscreen.vert` / `blur_ao.frag` | — | u1 s2 | no_depth | R8 |
| RtReflectionResolve | `fullscreen.vert` / `rt_reflection_resolve.frag` | — | u1 s3 | no_depth; RGB additive, destination alpha preserved | RGBA16F |
| Water | `water.vert` / `water.frag` | u1 s1 | u1 s5 | position2; opaque/no-cull, reversed-Z depth write | scene |
| BloomDownsample | `fullscreen.vert` / `bloom_downsample.frag` | — | u1 s1 | no_depth | RGBA16F |
| BloomUpsample | `fullscreen.vert` / `bloom_upsample.frag` | — | u1 s1 | **additive** (LOADOP_LOAD accumulate), no_depth | RGBA16F |
| VolumetricFog | `fullscreen.vert` / `volumetric_fog.frag` | — | u1 s4 | no_depth | RGBA16F |
| Composite | `fullscreen.vert` / `composite.frag` | — | u1 s4 | no_depth | swap |
| Fxaa | `fullscreen.vert` / `fxaa.frag` | — | u1 s1 | no_depth | swap |
| SkyCapture | `fullscreen.vert` / `sky_capture.frag` | — | u1 | no_depth; targets one capture-cube face | RGBA16F |
| IblSpecularPrefilter | `fullscreen.vert` / `ibl_prefilter.frag` | — | u1 s1 | no_depth; targets one specular-cube (face, mip) | RGBA16F |
| IblIrradiance | `fullscreen.vert` / `ibl_irradiance.frag` | — | u1 s1 | no_depth; targets one irradiance-cube face | RGBA16F |
| IblBrdfLut | `fullscreen.vert` / `ibl_brdf_lut.frag` | — | — | no_depth; rendered once at init | RG16F |

Shared shaders: `model.frag` serves both static and skinned model pipelines; `fullscreen.vert` (bufferless triangle from `SV_VertexID`: verts (-1,-1), (3,-1), (-1,3)) serves every post-process pipeline; `shadow_depth.frag` (alpha-test capable, `ShadowFragUniforms` + 1 sampler) serves static + instanced shadow casters while `shadow_depth_opaque.frag` (zero resources) serves skinned + terrain casters.

Depth convention: scene pipelines inherit the reversed-Z default `GREATER_OR_EQUAL` (near→1, far→0, clear 0.0); the four shadow pipelines override to `LESS_OR_EQUAL` with standard-Z ortho light projections, front-face culling (except terrain: no_cull), `with_depth_bias(1.25f, 1.75f)`, and `enable_depth_clip = false` (near-plane caster pancaking).

### SDL3 GPU binding-order contract

Expressed in HLSL with `[[vk::binding(slot, set)]]`; set assignment is fixed by SDL3 GPU:

- **Vertex:** set 0 = sampled textures, then storage textures, then storage buffers; set 1 = uniform buffers.
- **Fragment:** set 2 = sampled textures, then storage textures, then storage buffers; set 3 = uniform buffers.

Within a set, storage-buffer slots begin immediately after the shader's sampled textures: `grass.vert.hlsl` (2 vertex samplers) binds its chunk SSBO at `(2, 0)`; `skinned_model.vert.hlsl` / `model_instanced.vert.hlsl` / `particle.vert.hlsl` (no vertex textures) bind theirs at `(0, 0)`. Textures use `[[vk::combinedImageSampler]]` with the `Texture2D` and `SamplerState`/`SamplerComparisonState` sharing one binding (see `model.frag.hlsl`, which pairs the shadow atlas twice — once with a comparison sampler at `(1,2)`, once raw at `(2,2)`).

CPU-side slot mapping: `SDL_PushGPUVertexUniformData(cmd, i, ...)` → `(i, 1)`; `SDL_PushGPUFragmentUniformData(cmd, i, ...)` → `(i, 3)`; `SDL_BindGPUVertexStorageBuffers` / `SDL_BindGPUFragmentStorageBuffers(pass, first, ...)` → set-0/set-2 storage slots relative to the sampler count; `SDL_BindGPUFragmentSamplers(pass, first, ...)` → `(first.., 2)`.

`cluster_lighting.hlsli` codifies this for clustered lighting: the including fragment shader must `#define CLUSTER_SSBO_BINDING_BASE <its sampler count PLUS its storage-texture count>` first (compile-time `#error` otherwise), and `CLUSTER_SAMPLER_COUNT` to the sampler count alone when the two differ — they stopped being the same number at R5c, because storage textures sit between samplers and storage buffers in SDL's set-2 layout and the Full-DI tier declares two of them (`local_di.hlsli`). The include then declares `ClusterParamsCB` at `(2, 3)`, the local-shadow atlas + comparison sampler at the shader's LAST sampler slot (`CLUSTER_SAMPLER_COUNT-1`, set 2), and four SSBOs (`lightDataBuf` ByteAddressBuffer, `clusterOffsetsBuf StructuredBuffer<uint2>`, `lightIndicesBuf StructuredBuffer<uint>`, `localShadowSlots StructuredBuffer<LocalShadowSlot>`) at set-2 slots `BASE+0..3`; the CPU call binds FIVE, the fifth being the R4b screen-probe SH at `BASE+4` (`gi_screen_sample.hlsli`). Slot `(2,3)`, the last sampler, and those five set-2 bindings are reserved in every cluster-lit shader (`CLUSTER_SAMPLER_COUNT` 16 / 16 / 14 / 12 and `CLUSTER_SSBO_BINDING_BASE` 19 / 19 / 17 / 15 for `model.frag` / `model_instanced.frag` / `terrain.frag` / `grass.frag`). The three slots preceding the local-shadow atlas are the probe-field GI atlases `gi_sample.hlsli` reads — irradiance and visibility (linear clamp) plus the per-probe metadata plane (point clamp) — at 11–13 / 11–13 / 9–11 / 7–9 respectively; below those is the RT sun-visibility mask (10/10/8/6), with a white dummy while disabled.

### CPU↔GPU uniform contract — `src/bracken/gpu/gpu_uniforms.hpp`

Every cbuffer has an `alignas(16)` CPU mirror annotated with its `(set, binding)`; layouts must match byte-for-byte. Size, alignment and **every** field offset are pinned by the `BRACKEN_HLSL_CBUFFER`/`BRACKEN_HLSL_FIELD` `static_assert` block that follows each struct in the header (`gpu/hlsl_packing.hpp`); the sizes below are additionally checked against the compiled shaders by `src/bracken/gpu/uniform_layout.test.cpp`. Inventory:

| CPU struct | (set, binding) | Shader cbuffer | Size |
|---|---|---|---|
| `ModelTransformUniforms` | (1, b0) | `model.vert` / `skinned_model.vert` `TransformUniforms` | 288 |
| `ModelLightingUniforms` | (3, b0) | `model.frag` `LightingUniforms` | 304 (`iblEnabled` at 216, `giCascadeBase` at 240, `giEnabled` at 288) |
| `InstancedCameraUniforms` | (1, b0) | `model_instanced.vert` / `particle.vert` `CameraUniforms` | 144 |
| `InstancedLightingUniforms` | (3, b0) | `model_instanced.frag` (no `tintColor` — per-instance) | 288 (`iblEnabled` at 200, `giCascadeBase` at 224, `giEnabled` at 272) |
| `SkyboxFragmentUniforms` | (3, b0) | `skybox.frag` `SkyUniforms` | 192 |
| `SkyCaptureUniforms` | (3, b0) | `sky_capture.frag` `SkyCaptureUniforms` | 176 |
| `IblFilterUniforms` | (3, b0) | `ibl_prefilter.frag` / `ibl_irradiance.frag` `IblFilterUniforms` | 48 |
| `GridVertexUniforms` | (1, b0) | `grid.vert` | 64 |
| `UIScreenUniforms` | (1, b0) | `ui.vert` | 16 |
| `UIFragmentUniforms` | (3, b0) | `ui.frag` `UIFragmentUniforms` | 16 |
| `PrepassModelUniforms` | (1, b0) | `prepass_model.vert` / `prepass_skinned.vert` `PrepassTransformUniforms` | 400 |
| `PrepassCameraUniforms` | (1, b0) | `prepass_instanced.vert` / `prepass_terrain.vert` `PrepassCameraUniforms` | 256 |
| `TaaResolveUniforms` | (3, b0) | `taa_resolve.frag` | 176 |
| `TaaSharpenUniforms` | (3, b0) | `taa_sharpen.frag` | 16 |
| `ShadowTransformUniforms` | (1, b0) | `shadow_depth.vert` | 128 |
| `ShadowSkinnedTransformUniforms` | (1, b0) | `shadow_skinned.vert` | 144 |
| `ShadowTerrainUniforms` / `InstancedShadowUniforms` | (1, b0) | `shadow_terrain.vert` / `shadow_depth_instanced.vert` | 64 |
| `ShadowFragUniforms` | (3, b0) | `shadow_depth.frag` | 16 |
| `ShadowDataUniforms` | (3, b1) | every shadow-receiving frag (`ShadowUniforms`) | 336 |
| `GTAOUniforms` | (3, b0) | `gtao.frag` | 240 |
| `GtaoTemporalUniforms` | (3, b0) | `gtao_temporal.frag` | 32 |
| `GtaoUpsampleUniforms` | (3, b0) | `gtao_upsample.frag` | 32 |
| `BlurUniforms` | (3, b0) | `blur_ao.frag` | 32 |
| `RtReflectionResolveUniforms` | (3, b0) | `rt_reflection_resolve.frag` | 112 |
| `RtReflectionUniforms` | compute (2, b0) | `rt_reflection.comp` | 672 |
| `RtShadowUniforms` | compute (2, b0) | `rt_shadow.comp` | 128 |
| `RtLocalShadowUniforms` | compute (2, b0) | `rt_local_shadow.comp` | 160 |
| `render::lighting::ClusterParams` | compute (2, b1) | `rt_local_shadow.comp` | 208 |
| `RtShadowClassifyUniforms` | compute (2, b0) | `rt_shadow_denoise_classify.comp` | 16 |
| `RtShadowTemporalUniforms` | compute (2, b0) | `rt_shadow_denoise_temporal.comp` | 128 |
| `RtShadowGradientUniforms` | compute (2, b0) | `rt_shadow_gradient.comp` | 144 |
| `RtLocalShadowGradientUniforms` | compute (2, b0) | `rt_local_shadow_gradient.comp` | 112 |
| `render::lighting::ClusterParams` | compute (2, b1) | `rt_local_shadow_gradient.comp` | 208 |
| `RtGradientReconstructUniforms` | compute (2, b0) | `rt_gradient_reconstruct.comp` | 16 |
| `RtShadowFilterUniforms` | compute (2, b0) | `rt_shadow_denoise_filter.comp` | 48 |
| `RtReflectionTemporalUniforms` | compute (2, b0) | `rt_reflection_denoise_temporal.comp` | 240 |
| `GiProbeTraceUniforms` | compute (2, b0) | `gi_probe_trace.comp` | 528 |
| `GiProbeBlendUniforms` | compute (2, b0) | `gi_probe_blend.comp` / `gi_probe_blend_visibility.comp` | 32 |
| `GiProbeClassifyUniforms` | compute (2, b0) | `gi_probe_classify.comp` | 32 |
| `HzbCopyUniforms` | compute (2, b0) | `hzb_copy.comp` | 16 |
| `HzbReduceUniforms` | compute (2, b0) | `hzb_reduce.comp` | 16 |
| `GiScreenProbePlaceUniforms` | compute (2, b0) | `gi_screen_probe_place.comp` | 112 |
| `GiScreenProbeMarchUniforms` | compute (2, b0) | `gi_screen_probe_march.comp` | 192 |
| `GiScreenProbeArgsUniforms` | compute (2, b0) | `gi_screen_probe_args.comp` | 16 |
| `GiScreenProbeTraceUniforms` | compute (2, b0) | `gi_screen_probe_trace.comp` | 512 |
| `GiScreenProbeFilterUniforms` | compute (2, b0) | `gi_screen_probe_filter.comp` | 48 |
| `GiScreenProbeProjectUniforms` | compute (2, b0) | `gi_screen_probe_project.comp` | 16 |
| `CompositeUniforms` | (3, b0) | `composite.frag` | 112 |
| `FxaaUniforms` | (3, b0) | `fxaa.frag` | 16 |
| `BloomDownsampleUniforms` | (3, b0) | `bloom_downsample.frag` | 32 |
| `BloomUpsampleUniforms` | (3, b0) | `bloom_upsample.frag` | 16 |
| `VolumetricFogUniforms` | (3, b0) | `volumetric_fog.frag` | 544 |
| `ExposureHistogramUniforms` | compute (2, b0) | `exposure_histogram.comp` | 16 |
| `ExposureAdaptUniforms` | compute (2, b0) | `exposure_adapt.comp` | 48 |
| `ParticleFragmentUniforms` | (3, b0) | `particle.frag` | 16 |
| `ParticleBatchUniforms` | (1, b1) | `particle.vert` `BatchUniforms` | 16 |
| `render::lighting::ClusterParams` (`light_cluster.hpp` — no `gpu_uniforms.hpp` mirror by design) | (3, b2) | `cluster_lighting.hlsli` `ClusterParamsCB` | 192 |

Storage-buffer record mirrors (tight packing, no cbuffer rules): `InstanceData` (mat4 `model` + mat4 `normalMatrix` + vec4 `tint` + float `noFog` + `float _pad[3]` + mat4 `prevModel` = 224 bytes; read by both the forward and depth-prepass instanced shaders), `ShadowInstanceData` (mat4 `model`, 64 bytes), `ParticleInstanceData` (mat4 + vec4 = 80).

Packing rules the mirrors follow (and any new struct must): every `glm::vec3` is chased by a scalar or named pad so no field straddles a 16-byte register; padding is **named scalars, never arrays** (HLSL cbuffer arrays stride at 16 B — a `float _pad[3]` on the HLSL side would declare a 52-byte block; see the comment in `ui.frag.hlsl` and the `ShadowFragUniformsLayout` test); total size a multiple of 16.

Notable defaults living in these structs (pushed as-is unless the renderer overrides): `GTAOUniforms{radius 4.0, falloffRange 0.615, occluderThickness 1.0, maxDistance 12000}`, `GtaoTemporalUniforms{feedbackMax 0.9, depthRejectScale 10}`, `BlurUniforms{sharpness 40, maxDistance 12000}` (the two `maxDistance`s must agree — asserted in the tests), `CompositeUniforms{bloomStrength 0.25, exposure 1, tonemapMode 0, all exposure/grading gates neutral}`, `ExposureAdaptUniforms{percentiles 0.60/0.05, speeds 1.5/2.5 EV/s, EV clamp ±4, log2TargetGray −1.80}`, `BloomDownsampleUniforms{threshold 0.8 (post-exposure luminance), exposure 1}`, `VolumetricFogUniforms{fogDensity 0.0015, scatterStrength 0.25, fogHeight 100, fogFalloff 0.005, startDist 100, maxDist 3000, godRaysEnabled 1}`, `ModelLightingUniforms{roughnessFactor 0.85, ambientStrength 0.5, alphaCutoff 0 (=alpha test off)}`. Retained-geometry model upload also bakes `Mesh::rt_triangle_base_colors` (one linear RGBA value per triangle, sampled from the sRGB base-colour texture at centroid UV and multiplied by averaged vertex colour plus `baseColorFactor`); RT reflection attributes consume this compact proxy after source pixels have been released.

### Skinned/instanced storage-buffer contracts

- **Bone palettes:** `skinned_model.vert.hlsl` and `shadow_skinned.vert.hlsl` read `StructuredBuffer<float4x4> bonePalettes` at vertex `(0, 0)` — the frame's palettes, packed tightly at build time in the `FramePacket` and uploaded in `palettes()` order, indexed by the `paletteOffset` uniform (in matrices; = `BonePalette::matrix_offset`). **No GPU-side bone-count constant**; `MAX_BONES = 128` (`animation/animation_types.hpp`) caps only the CPU arrays, and `MAX_BONE_PALETTES = 1024` (`scene/frame_packet.hpp`) caps palettes per frame, enforced at build time in `alloc_bone_palette`.
- **Instances:** `model_instanced.vert.hlsl` reads `InstanceData` records at `(0, 0)`; `shadow_depth_instanced.vert.hlsl` reads `ShadowInstanceData`.
- **Particles:** `particle.vert.hlsl` reads `ParticleInstance` at `(0, 0)`, offset by `BatchUniforms::instanceOffset` (set 1, b1), one instanced draw per (emitter, mesh) batch; opacity is folded into rgb because the additive blend ignores alpha (`effect_renderer.cpp`).
- **Grass:** `grass.vert.hlsl` reads visible chunk origins at `(2, 0)` after its heightmap/splatmap samplers.
- **Cluster lights:** `lightDataBuf` layout is `[LightDataHeader 16B][LightHeader×cap 8B][PointLight×cap 48B][SpotLight×cap 80B]` with `cap = clusterParams.maxPerCluster.y` (the spot stride is 80, not 64, since R6 — the sizes and offsets in this paragraph are all `static_assert`ed at the top of `light_cluster.cpp`, which is the source of truth); the shader clamps the per-cluster count to `maxPerCluster.x` to bound the loop against stale/partial uploads, and reads each light's engine-assigned `shadow_slot` (point: byte 36 of the 48-B record; spot: byte 56) for local-shadow sampling (`cluster_lighting.hlsli`).

### Threading & lifetime contracts

- `ShaderManager` cache and `PipelineRegistry` are single-threaded (main/render thread). `FileWatcher` hot-reload callbacks are queued and executed on the main thread by `poll_main_thread()` (contract in `core/asset/file_watcher.hpp`), so `reload_all_pipelines()` runs on the main thread.
- `GPUShader`/`GPUPipeline` are move-only RAII over SDL handles, released through their `GPUDevice`. Cached shaders live as long as the `ShaderManager` entry; cached pipelines as long as the registry entry. On hot reload, "live pipelines from the previous frame are released by SDL once the in-flight commands finish" (header comment on `enable_hot_reload`).
- `GPUShader::s_compiler_initialized` is a plain process-global static bool (no atomics) — compiler init/shutdown must also stay single-threaded.

### Error handling summary

Uniform pattern: `nullptr` return + `SDL_Log`/`BRACKEN_LOG_*` (no `core::Result` in this layer; these are boot-time `bool`/pointer APIs per the repo convention). Specific behaviors: unreadable/empty `.spv` → `nullptr`; SPIR-V→backend compile failure → `nullptr` with `SDL_GetError()`; `get_pipeline` on uninitialized registry → `nullptr`; `preload_all_pipelines` returns `false` but attempts every type; hot-reload rebuild failure → warning log, missing pipeline until next reload; misnamed `.hlsl` → configure-time warning only; wrong `ShaderResources` counts → error log + reflected counts used (see `ShaderResources` above).

### Performance characteristics

- Build: cold configure builds SDL_shadercross (DXC) — minutes; steady-state, one `shadercross` process per changed shader; touching any `.hlsli` recompiles all 49 shaders (coarse dependency model).
- Runtime: shader loads are file-read + SPIRV cross-compile, cached per (path, stage, entry); pipeline creation is driver-side compile, mitigated by `preload_all_pipelines()` at load time (53 graphics + 2 compute pipelines). Uniform pushes are SDL push-constants-style (`SDL_PushGPU*UniformData`), re-pushed per draw/pass as needed; per-frame bulk data rides storage buffers uploaded once per frame in the `FrameUploads` copy pass.
- Hot reload is all-or-nothing: any `.spv` change rebuilds every shader + pipeline.

### Known limitations, stubs, and silent behaviors

- **Compute shaders are pipelines, not `GPUShader`s** — `.comp.hlsl` compiles through the same CMake integration, but at runtime the module goes straight into an `SDL_GPUComputePipeline` via `gpu::GPUComputePipeline`; `ShaderManager` never sees it.
- **`GPUShader` load APIs have no default entry point** — engine-built SPIR-V uses `VSMain`/`PSMain`; callers must pass it explicitly (a wrong entry point fails late and confusingly on Vulkan, so there is deliberately no default). `ShaderManager::get_vertex`/`get_fragment` default to `VSMain`/`PSMain`.
- **`ShaderManager::get` verifies `ShaderResources` on cache hits** — resource counts are baked at first load per (path, stage, entry); a later caller requesting different counts gets the cached shader plus an error log naming both sets of counts.
- **Misnamed shader files are skipped with a warning**, not an error.
- **Hot reload watches `.spv`, not `.hlsl`** — an HLSL edit needs `cmake --build build` before the watcher fires; a shader that fails to reload leaves its pipeline missing (draws that check for `nullptr` skip; `GPUPipeline::bind` no-ops on null).
- **GPU-side per-pass timings do not exist** (`gpu/gpu_debug.hpp`: `PassTiming::gpu_ms` is `nullopt`; SDL3 has no timestamp queries; `cpu_ms` measures command recording, not shader cost) — irrelevant to shader authoring but affects shader perf measurement; use RenderDoc/Nsight.

---

## Scene: Cameras, Culling, FramePacket & SceneRenderer

Module `bracken::scene`, sources under `src/bracken/scene/`. This layer defines the host↔renderer contract — the immutable per-frame snapshot `FramePacket`, its sole write interface `FrameBuilder`, and the SPSC handoff ring `PacketRing` (Phase 2 of `docs/multithreading_plan.md`) — plus `CameraState`, the CPU culling primitives (`Plane`, `Frustum`, `transform_math.hpp`), and the frame orchestrator (`SceneRenderer`) that consumes one sealed packet per frame and records the entire GPU frame. Unit tests: `camera_state.test.cpp`, `frustum.test.cpp`, `plane.test.cpp`, `frame_packet.test.cpp`, `frame_builder.test.cpp`, `frame_packet_ring.test.cpp`, `transform_math.test.cpp`; benchmarks in `scene.bench.cpp` (frustum extract/cull, normal matrix, heightmap sampling).

### CameraState and the projection convention

`src/bracken/contracts/camera_state.hpp` — a POD aggregate, no methods:

```cpp
struct CameraState {
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 view_projection;  // convention: projection * view
    glm::vec3 position;
};
```

Producers: `systems::CameraController::get_camera_state()` (`src/bracken/systems/camera_controller.hpp`) builds it from `get_view_matrix()` / `get_projection_matrix()` / `get_position()` with `view_projection = projection * view`. The engine camera (`src/bracken/systems/camera_system.cpp`, `compute_matrices()`) uses `glm::lookAt(pos, look_at, {0,1,0})` and:

```cpp
projection_matrix_ = glm::perspective(glm::radians(current_fov_), aspect, 15000.0f, 5.0f);
```

i.e. **reversed-Z via swapped near/far arguments** in an `RH_ZO` perspective (`GLM_FORCE_DEPTH_ZERO_TO_ONE` is a PUBLIC compile definition on the `bracken` target — it propagates to consumers). Resulting properties (pinned by `camera_state.test.cpp`):

- Right-handed: view-space points in front of the camera (negative z) project to positive clip `w`.
- Near plane → NDC z = 1, far plane → NDC z = 0; interior points strictly in (0, 1).
- Scene depth targets clear to **0.0**; scene pipeline depth compare defaults to `GREATER_OR_EQUAL` (`gpu/gpu_pipeline.hpp` `PipelineConfig`).
- Near-plane recovery from a reversed-Z RH_ZO matrix: `n = m[3][2] / (m[2][2] + 1)` — which is exactly the depth linearization below evaluated at device depth 1, and is spelled that way in `scene_renderer.cpp` so there is one formula rather than two identities to keep in step.
- Depth linearization `viewZ = |B / (depth + A)|` with `A = proj[2][2]`, `B = proj[3][2]` holds for both conventions; `SceneRenderer` passes these as `CompositeParams::depth_a/depth_b`. **Single source: `contracts/depth_linear.hpp`** (`scene::linear_view_depth` / `scene::depth_linear_terms`), transliterated by `shaders/depth_linear.hlsli`, which every depth-linearizing shader includes rather than open-coding. Pinned against the inverse-projection reference by `contracts/depth_linear.test.cpp`. The `+ A` is load-bearing: four traced passes carried `- A` until 2026-08-02, which evaluates to `t / (1 - 2t/far)` — +33% at 500 units, +100% at 1000, and a single saturated constant past `far/2`.
- Exception: shadow cascades stay standard-Z (see `render/shadow_map.hpp`; outside this chapter).

`SceneRenderer` consumes `CameraState` by value semantics per frame: `view_projection` feeds frustum extraction; `view`/`projection` feed all draw uniforms; `position` feeds distance culling, fog, transparent sorting, and specular. `render_frame` caches `inverse(projection)` and only recomputes when `camera.projection` changes bit-wise (`cached_projection_` / `cached_inv_projection_` members).

### Plane and Frustum

`src/bracken/contracts/frustum.hpp` / `frustum.cpp`.

```cpp
struct Plane {
    glm::vec3 normal;
    float distance;
    void normalize();                                        // no-op guard for zero-length normal
    float distance_to_point(const glm::vec3& point) const;   // dot(normal, p) + distance
};

class Frustum {
public:
    void extract_from_matrix(const glm::mat4& vp);
    bool intersects_aabb(const glm::vec3& min_point, const glm::vec3& max_point) const;
    bool intersects_sphere(const glm::vec3& center, float radius) const;
private:
    std::array<Plane, 6> planes_;
};
```

Algorithm: Gribb-Hartmann extraction on `transpose(vp)` (glm is column-major; the transpose gives clean row access). Plane order: `[0]` left (`row3+row0`), `[1]` right (`row3−row0`), `[2]` bottom (`row3+row1`), `[3]` top (`row3−row1`), `[4]` `row2` (bounds clip.z ≥ 0), `[5]` `row3−row2` (bounds clip.z ≤ w). Under `[0,1]` clip depth, `[4]`/`[5]` are near/far for a standard-Z matrix and swap roles under reversed-Z; all tests treat the six planes symmetrically, so the frustum is closed and convention-agnostic (verified against a standard-Z frustum on sample points in `frustum.test.cpp`). Works for orthographic matrices as well (shadow-cascade light frustums use it).

- `intersects_sphere`: outside iff `distance_to_point(center) < -radius` for any plane. Zero radius = point test.
- `intersects_aabb`: positive-vertex (p-vertex) test — per plane pick the corner farthest along the normal; outside iff that corner is behind the plane.

Both are conservative plane-set tests: objects outside the frustum volume but not fully behind any single plane (large objects near corners) are reported as intersecting — false positives only, never false negatives. Performance: 6 dot products per shape; `extract_from_matrix` is one transpose + 6 normalizations (benchmarked in `scene.bench.cpp`).

### transform_math.hpp

Free inline helpers (`src/bracken/scene/transform_math.hpp`):

- `bool has_uniform_scale(const glm::mat4&)` — compares squared column lengths of the 3×3 linear part with relative tolerance `max_s * 0.001f` (~0.05% on linear scale).
- `glm::mat4 compute_normal_matrix(const glm::mat4& model)` — returns `model` unchanged when scale is uniform (shader re-normalization makes `M/s²` equivalent), otherwise `transpose(inverse(model))`. The uniform path skips an ~80-flop 4×4 inverse; called per non-instanced draw and per instance during batching.
- `float max_scale_factor(const glm::mat4&)` — `sqrt(max(|c0|², |c1|², |c2|²))`; used to scale model bounding-sphere radii into world space (one sqrt instead of three).

### FramePacket — the immutable frame snapshot

`src/bracken/scene/frame_packet.hpp` / `frame_packet.cpp`. One frame's complete description: model/skinned draws, bone palettes, lights, billboards, debug lines, effect spawns, UI commands, camera + environment snapshots, draw toggles, and dt/time/frame-index. Non-copyable/movable; `explicit FramePacket(std::size_t initial_arena_bytes = k_default_arena_bytes /* 256 KB */)`. All storage is carved from the packet's own `memory::Arena`; frames that outgrow it grow the arena once and the memory is retained across resets (steady state: zero heap allocations). `MAX_BONE_PALETTES = 1024` is defined here and enforced at **build time** in `alloc_bone_palette`.

Lifecycle state machine (`FramePacket::State`, an acquire-loaded atomic): `Retired → Filling → Sealed → Consuming → Retired`. Writes happen only during `Filling` and only through `FrameBuilder`; every read accessor debug-asserts `Sealed|Consuming`. The `Filling → Sealed` release store in `FrameBuilder::seal()` is THE payload publish edge (non-atomic content + one atomic publish, house style); after seal, packet reads are tier `worker-safe` — plain loads, no atomics on the read path.

#### Command types (trivially copyable, padding-free via explicit zeroed `_pad`; everything by value or packet-allocated)

```cpp
struct BonePaletteId { uint32_t value = k_invalid /*0xFFFFFFFF*/; bool valid() const; };

struct ModelDrawCommand {
    bracken::ModelHandle model_handle;   // 0 = invalid
    uint32_t motion_id = 0;              // nonzero = moving object, pairs N with N-1 in MotionHistory
    glm::mat4 transform;
    glm::vec4 tint = {1,1,1,1};          // LINEAR RGBA multiplier, not a packed core::Color
    DrawFlags flags = DrawFlags::None;   // ForceNonInstanced | NoFog (scene/draw_flags.hpp)
};                                       // NOTE: field order differs from the retired ModelCommand

struct SkinnedDrawCommand {
    bracken::ModelHandle model_handle;
    uint32_t motion_id = 0;
    glm::mat4 transform;
    glm::vec4 tint = {1,1,1,1};
    BonePaletteId palette;               // packet palette — replaces the old non-owning bone span
    DrawFlags flags = DrawFlags::None;   // NoFog applies; ForceNonInstanced is inert (never instanced)
};

struct BonePalette {                     // one packed palette of a sealed frame
    const glm::mat4* matrices;           // packet memory
    uint32_t joint_count;
    uint32_t matrix_offset;              // prefix-summed first-matrix index == the GPU palette offset
};

struct BillboardDrawCommand { float world_x, world_y, world_z, width, fill_ratio;
                              uint32_t fill_color, bg_color, frame_color; };
struct DebugLineDrawCommand { glm::vec3 start, end; uint32_t color; };
struct EffectSpawnCommand   { const bracken::EffectDefinition* definition;
                              glm::vec3 position{0,0,0}, direction{1,0,0}; };
```

UI commands: `UiFilledRectCommand`, `UiRectOutlineCommand`, `UiCircleCommand`, `UiCircleOutlineCommand`, `UiLineCommand`, `UiTextCommand`, `UiButtonCommand` — `UiCommandData` is a `std::variant` of those seven (in that order), wrapped in `struct UiCommand { UiCommandData data; is<T>(); get<T>(); }`. `UiTextCommand::text` / `UiButtonCommand::label` are `std::string_view`s over **packet-arena copies** made at emit time — no `std::string`, no heap, valid until the packet retires. All color fields are `core::Color` — the packed-ARGB word of `bracken/core/color.hpp`, implicitly constructible from a `uint32_t` literal.

`BonePaletteId` semantics: opaque while the frame is Filling (an internal lane/allocation-order encoding — any lane may use an id allocated on another lane, provided the allocation happened-before the use). At seal every id stored in a command is remapped to a **dense index into `FramePacket::palettes()`**, assigned in deterministic (lane-index, allocation-order) order. **Ids held by the host across `seal()` are stale** — only re-read them from sealed commands.

#### Sealed-frame read API (const, assert-sealed, worker-safe)

`std::span<const T>` accessors: `model_commands()`, `skinned_commands()`, `billboards()`, `debug_lines()`, `effect_spawns()`, `point_lights()`, `spot_lights()`, `ui_commands()`, `palettes()`; plus `palette_matrices(BonePaletteId)` (one palette's matrices) and `palette_matrix_total()` (total mat4 count across `palettes()` — the packed palette stream size a renderer uploads). Singular state: `camera()`/`has_camera()` and `environment()`/`has_environment()` (the `has_*` flag is `false` when the host never called the setter this frame — the consumer keeps its own current camera/environment); `should_draw_skybox/ground/grass()` (default `true`, reset each frame); `dt()`, `time()`, `frame_index()`; `has_3d_content()` (same formula as the retired `RenderScene`: any draw toggle on, or any model/skinned/billboard/debug-line/light present; effect spawns excluded — billboard-only or debug-line-only frames run the full 3D pass); `bytes_used()`/`bytes_capacity()` (arena introspection).

Internal API (builder/ring/coordinator — public with a banner, not for host code): `begin_fill()`, `begin_consume()`, `reset()` (any state → Retired; arena reset with memory retained; the caller supplies the frame-fence synchronization), `reserve_block(size, align ≤ 128, pow2)` (thread-safe during fill: one relaxed `fetch_add` per block on the active 64 KB slab, mutex-guarded arena refill on the cold path, aborts on OOM), `take_palette_slot()` (relaxed budget counter).

### FrameSink — the host half of the write interface

`src/bracken/scene/frame_sink.hpp` (header-only). Tier `sim-stage`. A pointer-sized (`sizeof(void*)`, trivially copyable) non-owning view of a `FrameBuilder` that forwards **only** the host surface: `lane_count()`, `lane(i)`, `draw_model`, `draw_skinned`, `alloc_bone_palette`, `add_point_light`, `add_spot_light`, `add_billboard_3d`, `add_debug_line/sphere/box`, `spawn_effect`, `ui()`, `set_camera`, `set_environment`, `set_draw_skybox/ground/grass`. `SimFrame::frame()` returns one.

The engine's sim-stage tail is not a member of it and is **private** on `FrameBuilder`; friends are `bracken::Application`, `bracken::systems::EffectSystem` (`snapshot_into` calls `alloc_effect_snapshot` and is not reached through `Application`) and `bracken::testing::FrameAccess`. `src/bracken/testing/frame_test.test.cpp` pins the split with `requires`-expression detectors (a private member and an absent member both make the expression ill-formed, so one check covers both).

### FrameBuilder — the only write interface

`src/bracken/scene/frame_builder.hpp` / `frame_builder.cpp`. Tier `sim-stage`. `explicit FrameBuilder(FramePacket&, uint32_t lane_count = 1)` performs `Retired → Filling`; lanes live in the packet arena (construction is cheap — one builder per frame); non-copyable/movable; after `seal()` the builder is **spent** — discard it, never reuse it after the packet resets. `k_max_lanes = 4096`.

Emission is multi-lane: `Lane& lane(uint32_t i)` returns one writer's private stream (tier `owned` — one thread at a time; cache-line(128)-padded header; chunked storage with one `reserve_block` `fetch_add` per chunk and private bumps per item). `FrameBuilder` itself mirrors every `Lane` emit plus `ui()` as `lane(0)` forwards — single-threaded hosts never see a lane. Concurrent emission fans out one lane per job; **determinism**: `seal()` merges lanes in (lane-index, emission-order) order, so the merged frame is byte-identical at any worker count *if lanes are keyed by job/chunk index, never thread id*.

`Lane` emit API (mirrors the retired `RenderScene` one-for-one; colors are `core::Color`):

- `draw_model(handle, transform, tint = {1,1,1,1}, DrawFlags flags = DrawFlags::None, motion_id = 0)`
- `draw_skinned(handle, transform, BonePaletteId palette, tint = {1,1,1,1}, DrawFlags flags = DrawFlags::None, motion_id = 0)` — an invalid id drops the draw with a warning.
- `[[nodiscard]] alloc_bone_palette(joint_count /* 1..=animation::MAX_BONES */)` → `BonePaletteAllocation { std::span<glm::mat4> matrices; BonePaletteId id; bool valid(); }` — 16-byte-aligned **uninitialized** packet memory; the host writes every matrix before seal. Budget `FramePacket::MAX_BONE_PALETTES = 1024` per frame; over-budget allocations log an error and return an invalid allocation (empty span) — the visible build-time failure replacing the old render-time silent drop. One palette may be shared by many draws, across lanes (allocation must happen-before use) — reusing the id replaces the renderer's old pointer-dedupe of shared palettes.
- `add_point_light(const PointLightDesc&)` / `add_spot_light(const SpotLightDesc&)` — host descriptors, not the GPU PODs (`src/bracken/contracts/light.hpp`). `PointLightDesc` = position/radius/color (linear vec3)/intensity + `casts_shadows` opt-in + `stable_id` (persistent identity, `0` = engine assigns — the default) + `source_radius` (R6 area-light dial: the EMITTER's radius in world units, `0` = use `RtLocalShadowSettings::source_radius`; affects TRACED local shadows only, and `0` does not mean "point light"); `SpotLightDesc` adds direction + `core::Degrees inner_angle`/`outer_angle` (half-angles from the axis). The builder converts to `PointLight`/`SpotLight`, taking `std::cos(angle.radians())` for the cone, copying `stable_id` and `source_radius` across, and leaving `_pad0`/`_pad1`/`shadow_slot` at their engine-owned defaults — a host never names those.
- `add_billboard_3d(wx, wy, wz, width, fill_ratio, fill_color, bg_color, frame_color)`
- `add_debug_line(start, end, color)`; `add_debug_sphere(center, radius, color, segments = 16)` (3 × segments circle-outline lines in the XZ/XY/YZ planes); `add_debug_box(min, max, color)` (12 AABB edges).
- `spawn_effect(const EffectDefinition*, position, direction = {1,0,0})` — consumed exactly once from the sealed packet.
- `[[nodiscard]] UiLane ui()` — the 2D surface (below).

`UiLane` (a small value handle bound to a lane, mirror of the retired `UIScene`): `add_filled_rect(x,y,w,h,color)`, `add_rect_outline(x,y,w,h,color,line_width=2)`, `add_circle(x,y,radius,color,segments=24)`, `add_circle_outline(x,y,radius,color,line_width=2,segments=24)`, `add_line(x1,y1,x2,y2,color,line_width=2)`, `add_text(std::string_view, x, y, scale=1, color=0xFFFFFFFF)`, `add_text_wrapped(text, x, y, max_width, scale=1, color=0xFFFFFFFF, line_height=0)` (word-wrap on the same approximate 8 px × scale glyph width as the retired `UIScene`; returns total height consumed), `add_button(x,y,w,h,label,color,selected=false)`. Note: text parameters are `std::string_view` (was `const std::string&`) and are copied into packet memory at emit time — temporaries are safe.

Singular frame state (single-writer — call from the frame's root code, not fanned-out jobs; last write wins): `set_camera(const CameraState&)`, `set_environment(const render::EnvironmentState&)`, `set_draw_skybox/ground/grass(bool)`. `set_frame_timing(float dt, double time, uint64_t frame_index)` is part of the private tail — the engine stamps it at kick.

**Engine-internal sim-stage API** — `private`, friended to `Application`, `systems::EffectSystem` and `testing::FrameAccess` (used by `Application`'s sim-job tail, single-threaded after `on_simulate` returned and every emission job joined, before `seal()`):

- `for_each_effect_spawn(fn)` — walks every `EffectSpawnCommand` emitted this fill in deterministic (lane-index, emission-order) order (the same order `seal()` merges), so the sim stage can apply spawns to the `EffectSystem` before its update.
- `alloc_effect_snapshot(emitter_count, particle_count)` → `EffectSnapshotAllocation { std::span<EffectEmitterSnapshot> emitters; std::span<EffectParticleSnapshot> particles; }` — **uninitialized** packet memory, already registered as the packet's `effect_emitters()`/`effect_particles()`; call at most once per fill (zero counts register empty spans). Filled by `EffectSystem::snapshot_into(FrameBuilder&)`.
- `set_engine_frame(scene_time, sky_time, effect_instance_count)` — records the sim-advanced clocks + live-effect count; sets `has_engine_frame()`.
- `has_staged_environment()` / `staged_environment()` — reads of the staged singular state during Filling (the sky-clock advance needs the staged environment's `cloud_speed`).
- `set_frame_timing(dt, time, frame_index)`, `seal()`, `packet()`.

`FramePacket`'s lifecycle and raw allocator are private under the same rule: `begin_fill()`, `begin_consume()`, `reset()`, `reserve_block(size, align)`, `take_palette_slot()`, `palette_slots_taken()` and the three `count_drop_*` counters. Friends are `FrameBuilder`, `PacketRing`, `Application` (abandon-on-throw) and `testing::FrameAccess`.

**`bracken/testing/frame_test.hpp`** (header-only, public) is where the capability is published. `testing::FrameAccess` is a struct of static forwarders for every private member listed above. `testing::TestFrame` is the ergonomic fixture built on it: `TestFrame(lane_count = 1, arena_bytes = k_default_arena_bytes)` owns a packet, `TestFrame(FramePacket&, lane_count = 1)` borrows one; `sink()`, `builder()`, `packet()`, `set_frame_timing`, `seal()`, `sealed()`, typed read-back (`draws()`, `skinned_draws()`, `point_lights()`, `spot_lights()`, `billboards()`, `debug_lines()`, `effect_spawns()`, `ui_commands()`, `ui_text()`), and `build_diagnostics()`.

**Effect snapshot contract** (`frame_packet.hpp`): `EffectEmitterSnapshot { const EmitterDefinition* definition; uint32_t first_particle, particle_count; }` — `definition` is a long-lived host asset pointer (same contract as `EffectSpawnCommand`), and these two pointers are the packet's **only** host-lifetime exceptions: an `EffectDefinition` handed to `spawn_effect` must outlive every instance spawned from it, since the `EffectSystem` retains the emitter definitions across frames. Null is a `BRACKEN_CONTRACT` at emit (`FrameBuilder::Lane::spawn_effect`) and at consume (`Application`'s spawn drain); the particle range indexes `effect_particles()`. `EffectParticleSnapshot { vec3 position; float scale; vec3 rotation; float opacity; vec4 color; }` — exactly the fields the effect renderer's transform build + instance packing consume. The renderer reads the snapshot from the SEALED packet only; the live `EffectSystem` is sim-stage owned and mutates concurrently with rendering.

`seal()`: PRECONDITION — every job that emitted into a lane has been joined (`TaskGroup` wait). Merges every lane into contiguous packet spans in deterministic order, builds the `BonePalette` records with prefix-summed `matrix_offset`, remaps every `SkinnedDrawCommand::palette` to its dense merged index, then release-publishes `Sealed`. Call exactly once. Debug enforcement everywhere: every emit/setter/`reserve_block` asserts `Filling` (write-after-seal aborts); every packet read asserts sealed.

### PacketRing — SPSC frame handoff

`src/bracken/scene/frame_packet_ring.hpp` (header-only). `explicit PacketRing(uint32_t depth = 2, std::size_t initial_arena_bytes = FramePacket::k_default_arena_bytes)` (depth ≥ 1). Contract: one producing stage per frame (`acquire_for_fill` → `FrameBuilder` → `seal` → `publish`) and one consuming stage (`try_acquire_sealed` → const reads → `retire`); the stages may live on different threads. The ring never blocks — backpressure is the caller's job (`peek_fill().state() == Retired`; with depth D the fill side may run at most D−1 frames ahead of retirement).

- `acquire_for_fill()` — round-robin next slot; asserts it is Retired (pipeline overrun otherwise); the returned packet is still Retired — the `FrameBuilder` ctor does the transition.
- `publish(FramePacket&)` — asserts Sealed; ONE release store of the slot index (the frame's single publish; SPSC mailbox, asserts the previous publication was taken).
- `has_sealed()` (relaxed hint, either side may poll) / `try_acquire_sealed()` — ONE acquire exchange; `nullptr` when nothing pending; transitions Sealed → Consuming.
- `retire(FramePacket&)` — accepts Consuming **or** Sealed (the latter for the sequential no-publish flow); calls `packet.reset()` (arena memory retained). Must happen-before the fill side's next acquire of that slot.
- Introspection: `depth()`, `packet(slot)`.

The Phase-2 sequential frame skips the mailbox entirely: `FramePacket& p = ring.acquire_for_fill(); FrameBuilder fb(p); …emit…; fb.seal(); renderer.render_frame(p); ring.retire(p);`.

#### Lifetime/threading contract

A sealed `FramePacket` carries **no host-lifetime contracts**: every command is by value, bone palettes and UI text live in packet memory, and the packet outlives its own consumption by construction. The exceptions are long-lived asset pointers (`EffectSpawnCommand::definition` must outlive spawned instances; `ModelHandle`s resolve against the `ModelManager`). During fill, a lane is single-owner; `FramePacket::reserve_block`/`take_palette_slot` are safe from any fill-stage thread; `seal()` runs on one thread after all emitters are joined.

### SceneRenderer

`src/bracken/scene/scene_renderer.hpp` plus **six** implementation files. `SceneRenderer` was a single `.cpp` of ~3.3k lines until Phase 6 split it by frame stage. The class, its header and its API are unchanged — this table is purely where to look:

| File | What is in it |
|---|---|
| `scene_renderer.cpp` | the frame's spine: construction, `init`/`shutdown`, settings application, and `render_frame`, which calls into the rest |
| `scene_renderer_upload.cpp` | the single consolidated `FrameUploads` copy pass |
| `scene_renderer_shadow.cpp` | CSM cascades + the local (point/spot) shadow atlas |
| `scene_renderer_prepass.cpp` | depth prepass and motion vectors |
| `scene_renderer_draw.cpp` | the forward/transparent draw walk, billboards, debug lines |
| `scene_renderer_capture.cpp` | screenshot readback |

The Prepare stage is *not* one of these: it was extracted in Phase 6 as its own type, `ScenePrep` (`scene/scene_prep.hpp` / `.cpp`), so that the renderer's one thread-tier boundary is expressed by a device-free, stateless type that structurally cannot reach a main-thread API. Pass selection likewise became `RenderPlan` (`scene/render_plan.hpp` / `.cpp`), which is pure and tested headless.

Non-copyable, non-movable. Owns (as `unique_ptr` members): `gpu::PipelineRegistry`, `render::TerrainRenderer`, `render::WorldRenderer` (skybox), `render::UIRenderer`, `render::EffectRenderer`, `ModelManager`, `render::GrassRenderer`, `render::ShadowMap`, `render::PostProcessStack`, `render::GTAO`, `render::TAA`, `render::AutoExposure`, `render::Bloom`, `render::VolumetricFog`, plus `render::lighting::ClusterGrid` by value. (The `systems::EffectSystem` is **not** owned here anymore — it is sim-stage owned by `Application`; the renderer consumes the packet's effect-instance snapshot.)

#### Public API

```cpp
[[nodiscard]] bool init(render::RenderContext& context,
                        float world_width = 32000.0f, float world_height = 32000.0f);
void shutdown();
void render_frame(const FramePacket& frame);   // consumes one sealed packet per frame

void set_heightmap(const Heightmap&);
void set_anisotropic_filter(int level);      // 2^level x, clamped to 16; recreates default sampler
void set_graphics_settings(const GraphicsSettings&);
void set_vsync_mode(int mode);  int max_vsync_mode() const;
void set_screen_size(int width, int height);

void set_collect_stats(bool);  const bracken::RenderStats& render_stats() const;
void request_screenshot(std::string path);   // one-shot, blocking capture of the next frame

render::RenderContext* context();
render::TerrainRenderer& terrain();
ModelManager& models();
render::GrassRenderer* grass();
gpu::PipelineRegistry* pipeline_registry();
float get_terrain_height(float x, float z);
EnvironmentState& environment();                                  // live-editable; applied next frame
uint32_t frame_effect_count() const;   // live EffectInstances at the rendered packet's snapshot time (DevUI Stats)
uint32_t frame_particle_count() const; // particles in the rendered packet's snapshot (DevUI Stats)
int text_width(std::string_view text, float scale = 1.0f) const;  // UI glyph metrics; 0 pre-init
int text_height(float scale = 1.0f) const;                        // or without a loaded font

using PostUICallback = std::function<void(SDL_GPUCommandBuffer*, SDL_GPUTexture*)>;
void set_post_ui_callback(PostUICallback cb);                     // single slot
```

#### init() behavior

`init` static-asserts `MAX_SHADOW_CASCADES == render::CSM_MAX_CASCADES` (the header mirrors the constant because it only forward-declares `ShadowMap`). Hard failures (return `false`): pipeline registry, terrain, world, UI, effect renderer, depth texture. Soft failures (warning to `std::cerr`, feature disabled): pipeline preload, grass, shadow map, local shadow map (point/spot lights render unshadowed), post-process stack, GTAO, bloom, volumetric fog, TAA, IBL (sky image-based lighting falls back to hemisphere ambient), timestamp pool, cluster grid (`cluster_grid_ready_ = false` disables dynamic lighting). Shadow resolution comes from `GraphicsSettings::shadow_resolution` via table `{512, 1024, 2048, 4096}`; active cascade count = `shadow_cascades + 1`; the local-shadow atlas comes from `local_shadow_resolution` via table `{1024, 2048, 4096}`. `init_pipelines()` creates the default sampler (linear, repeat, 16× anisotropy, `max_lod = 1000` so mip chains stay reachable) and four fallback textures: 1×1 white (base-color/MR slots), 1×1 flat normal `(128,128,255,255)`, 1×1 black (emissive), and a 1×1×6 cube (the IBL cube slots when `render::IBL` is unavailable — a `TextureCube`-declared slot must receive a cube-typed texture; never sampled because `iblEnabled` is 0 then). The `RenderContext` must outlive the `SceneRenderer`.

#### render_frame — exact order

Preconditions: main thread (`BRACKEN_ASSERT_MAIN_THREAD()`); `frame` is a sealed packet, consumed exactly once (the caller retires it afterwards). Camera = `frame.camera()` (when `has_camera()` is false the renderer skips the 3D pass rather than draw with a stale/default camera); when `frame.has_environment()` the packet's environment snapshot is applied, otherwise the renderer keeps its own `env_` (preserving the DevUI live-edit path). Frame-level state cached once: `frame_fog_active_ = gfx.fog_enabled`, `frame_draw_dist_sq_ = draw_distance²`, `frame_do_frustum_cull_ = gfx.frustum_culling`.

1. **Clock adoption + environment propagation** — when `frame.has_engine_frame()`: `scene_time_ = frame.scene_time()` and `world_->set_sky_time(frame.sky_time())` — the clocks are **advanced by the sim stage** and adopted here; the renderer no longer integrates dt (hand-built packets without an engine section keep the previous clocks). `frame_effect_count_`/`frame_particle_count_` cached from the packet for the DevUI Stats tab. Then `light_dir_ = env_.sun_dir`; `env_` pushed to world/terrain/grass/water renderers.
2. **Effect snapshot pre-upload** — `effects_->ensure_models_uploaded(frame.effect_emitters())` GPU-uploads any model referenced by the packet's effect-instance snapshot, **before** any pass (and before the swapchain/minimized early-outs — uploads can't run inside a render pass; the mid-pass path is a logged skip). The snapshot is the render side's **only** view of the particle system: spawns were applied and the live `EffectSystem` updated on the sim side before the packet sealed — there is no renderer-side spawn drain or particle update anymore.
3. **begin_frame** — acquires the command buffer; minimized windows (`SDL_WINDOW_MINIMIZED`) skip all recording but still submit (balanced lifecycle).
4. If `frame.has_3d_content()`:
   a. `frame_frustum_.extract_from_matrix(camera.view_projection)`; refresh `cached_inv_projection_` if the projection changed.
   b. **Shadows** (when `shadow_map_->is_ready() && gfx.shadow_mode != ShadowMode::Off`): recover camera near via `m[3][2]/(m[2][2]+1)` (min 0.1); `shadow_far = min(draw_distance, shadow_distance)` (settings field, default 4000); `shadow_map_->update(...)`; extract a `Frustum` per active cascade from `cascade.light_view_projection`.
   c. **CPU prep**: instance batching over `frame.model_commands()`, skinned-command resolve over `frame.skinned_commands()` (palette records read straight from `frame.palettes()`), grass chunk prep, local-shadow selection (`local_shadows_->update(...)` when `gfx.local_shadows` and the scene has dynamic lights: `LocalShadowSelector` picks the budgeted casters, assigns tiles, marks dirty tiles, and writes `shadow_slot` into the frame's light copies; otherwise every `shadow_slot` is forced to −1), cluster-grid build (`begin_frame`/`add_point_light`/`add_spot_light`/`build`) — skipped entirely when the scene has zero dynamic lights, in which case `frame_cluster_params_ = {}` publishes `gridDim.w == 0` so shaders early-out instead of reading stale counts (the SSBOs keep last frame's contents). The instance-batching, skinned-resolve, grass-prep, and cluster-build loops here are the Phase-3 Prepare fan-out: they run as `parallel_for`/`TaskGroup` chunks over the sealed packet through stage helpers taking `JobSystem&` (call sites pass `JobSystem::instance()`), per-chunk partials merged single-threaded in chunk-index order — bit-identical at any worker count; grain guards keep small scenes to one inline chunk (see the Job System chapter). Local-shadow selection (a small budgeted caster pick) stays serial.
   d. **`FrameUploads`** — a single `SDL_GPUCopyPass`: `upload_instance_buffers`, `upload_bone_palettes`, `effects_->upload_particle_instances(copy_pass, frame.effect_emitters(), frame.effect_particles())` (packs the packet snapshot's per-particle transforms/tints into the instance buffer), `upload_debug_lines`, grass `upload_chunks`, `cluster_grid_.upload`, `local_shadows_->upload` (slot SSBO, only when dirty).
   e. **`IBL`** (only when `gfx.ibl_enabled` and `render::IBL` is ready) — `ibl_->update(cmd, pipelines, env_, world_->sky_time())`: re-snapshot the sky-relevant environment fields, and when a refresh is in flight record up to `IBL::UNITS_PER_FRAME` prefilter passes (usually records nothing). See the IBL section in the Environment chapter.
   f. **`ShadowPass`** — see below. Then **`LocalShadowPass`** (when local shadows are active this frame; independent of the CSM toggle): `render_local_shadow_passes` re-renders this frame's dirty tiles into the 8×8-tile atlas — see the local-shadow section in the Lighting chapter.
   g. Cache `frame_shadow_uniforms_ = shadow_map_->get_shadow_uniforms(mode)`; hand cluster buffers + params (and the local-shadow slot SSBO) to terrain/grass.
   h. **HDR path** (whenever `post_->is_ready()`, regardless of `ao_mode`): resize post targets (incl. `gtao_->set_half_res`/`resize` and `taa_->resize`, all of which invalidate their history on change); offscreen clear = `env_.fog_color_horizon`; **`DepthPrepass`** (velocity + opaque scene depth, plus the RGBA8 oct-normal/roughness/metallic G-buffer when RT reflections are enabled); **`GTAO`** (only when `ao_mode != Off` and the prepass ran: `render::GTAO::render` — raw horizon-based GTAO from the prepass depth, temporal accumulation against the velocity target, bilateral blur, joint-bilateral upsample into the full-res RGBA8 AO + bent-cone texture the forward pass samples; otherwise `gtao_->invalidate_history()` and the forward pass gets the 1×1 white dummy); **RT compute** records shadows and reduced-resolution raw reflection radiance using sampled depth/G-buffer bindings (per visibility signal, in order: trace → A-SVGF gradient trace → gradient reconstruct → tile classify → temporal → filter+upsample. R5b target formats per signal: packed visibility bits `R32_UINT` at tile resolution, **ping-ponged** so the gradient can compare against the previous frame's raw bits; tile classes `R8_UINT`; main history `RGBA16F` ping-pong (mean, second moment, count, linear view depth); **fast history `RG16F` ping-pong** (four-frame mean, count) for the dual-history clamp; **gradient `R16_FLOAT` pair** at stratum resolution = ceil(traced / 3), [0] raw and [1] dilated; full-res `R8_UNORM` mask. Plus one shared 8-uint `ComputeStorage` diagnostics buffer, never reset and differenced across lagged readbacks, which is what `RtStats::denoise` reports); `HDROffscreen` renders `render_3d_scene` into the linear-HDR target (forward shaders min-combine the screen AO with material ORM AO in their ambient term, every opaque receiver including terrain/grass uses the traced sun mask when enabled, and eligible opaque model materials smoothly surrender IBL specular to tracing); **`RtReflectionResolve`** adds the depth/normal/material-guided full-resolution receiver-BRDF result to HDR; when water is active, **`RtWaterReflection`** then recycles the raw target for a wave-aware analytic water receiver and the water shader consumes it as the SSR-gap signal; **`TAA`** (only when `aa_mode == AAMode::TAA` and the depth prepass ran this frame: `render::TAA::render` resolves the jittered offscreen color + velocity + depth into the double-buffered RGBA16F history and writes the sharpened result back over the offscreen color; if TAA is active but the prepass/velocity is unavailable, the history is invalidated instead); **`AutoExposure`** (only when `auto_exposure` and the system is ready: `render::AutoExposure::render` records the histogram + adapt compute dispatches over the TAA-resolved offscreen color, with speeds/clamps from the settings; any frame it cannot record calls `auto_exposure_->invalidate()` so the next run snaps to the metered target); `Bloom` (if enabled + ready — reads the TAA-stabilized color when TAA ran; thresholds against the MANUAL exposure only, the adapted EV is not visible to it); `VolumetricFog` (if `volumetric_fog || god_rays`; god rays force-off when shadows are off — the atlas would be garbage); **late swapchain acquire**; `Composite` (bloom strength, exposure = manual bias with the adapted-EV SSBO + `auto_exposure` flag, tonemap mode, contrast, saturation, white-balance coefficients from `render::white_balance_coefficients(env_.grade_temperature, env_.grade_tint)` + the CDL triple from `env_.grade_slope/offset/power`, `depth_a = proj[2][2]`, `depth_b = proj[3][2]` for the fog upsample — neither AO nor RT reflections are composited here) into the swapchain, or into an LDR intermediate followed by `FXAA` only when `aa_mode == AAMode::FXAA` (TAA and Off composite straight to the swapchain). A failed late acquire sets `swapchain_unavailable` — composite + UI are skipped, the buffer still submits.
   i. **Fallback** (post stack not ready): single `MainScene` pass direct to swapchain via `begin_main_pass` (color clear `{0.35, 0.45, 0.6, 1}`, depth clear **0.0** with explicit `LOADOP_CLEAR`); no prepass, no AO (white dummy bound).
5. **`UI`** — `begin_ui` (acquires swapchain if not yet held), replay of `frame.ui_commands()` (a `std::visit` over each `UiCommand` onto the matching `render::UIRenderer::draw_*` call, in submission order), then `draw_billboard_3d` per `frame.billboards()` entry, `end_ui` (clears the background iff no main pass ran this frame).
6. **`PostUICallback`** — invoked with `(command_buffer, swapchain_texture)` only when a swapchain image is held.
7. `gpu_timer_pool_.end_frame`; if `collect_stats_`, `read_results` into `render_stats_.pass_times_ms` (zeros — documented stub) + Tracy plots.
8. **Screenshot** (when `pending_screenshot_path_` non-empty and swapchain held): `capture_screenshot_and_end_frame()` — copy pass `SDL_DownloadFromGPUTexture` of the swapchain into a `DOWNLOAD` transfer buffer, `context_->end_frame_and_wait()` (fence submit + blocking wait), map, wrap in an `SDL_Surface` (BGRA/RGBA UNORM/sRGB swapchain formats handled; unknown formats assume RGBA8 with a logged warning), convert to `RGB24` (drops undefined swapchain alpha), `IMG_SavePNG`. Otherwise plain `end_frame()`. (The only other GPU→CPU readback is `render::AutoExposure`'s fenced 16-byte diagnostic download — non-blocking.)

#### render_3d_scene — in-pass draw order

Within the main/offscreen pass (every scene draw also binds the frame's screen-AO texture and the 3-slot sky-IBL chain — real maps or typed dummies): (1) terrain (if `should_draw_ground()`; shadow atlas bound twice — compare sampler for hardware PCF + raw sampler for PCSS blocker search); (2) grass (if flag + `gfx.grass_enabled`); (3) instanced static models; (4) non-instanced static models (`DrawFlags::ForceNonInstanced` survivors); (5) skinned models (resolved list; per-draw XZ distance + sphere frustum cull; pipeline/shadow/palette SSBO bound once lazily before the first visible draw); (6) skybox (if flag + `gfx.skybox_enabled`) — drawn after opaques so only unwritten pixels shade; (7) sorted transparent models; (8) particle effects (additive HDR, depth-test on / write off); (9) debug lines. `transparent_draws_` is cleared at entry to drop stale entries from an aborted pass.

#### Instanced batching (static models)

`build_instance_batches`: per-command pipeline —

1. XZ-plane distance cull: `dx² + dz² > frame_draw_dist_sq_` → `entities_distance_culled++`, skip (height ignored).
2. Bounds resolve (only when frustum culling or shadows are active): `world_center = transform * model->bounding_center`, `world_radius = model->bounding_half_diag * max_scale_factor(transform)`. Unresolvable model ⇒ no bounds ⇒ no frustum cull (command passes through).
3. Main-view sphere cull. A command that fails main-view culling **still bins into shadow cascades** whose light frustum intersects it (off-screen casters still cast).
4. `DrawFlags::ForceNonInstanced` survivors collect into `non_instanced_commands_` (drawn individually and shadow-rendered via the non-instanced shadow path).
5. Otherwise emit `gpu::InstanceData{model, normalMatrix (via compute_normal_matrix), tint, noFog, pad, prevModel}` into the per-handle bin (`prevModel` from `MotionHistory` via the command's `motion_id`, or `model` when static), and `gpu::ShadowInstanceData{model}` into each intersecting cascade's bin.

Data structure: `instance_batches_` is a dense `std::vector<std::vector<InstanceData>>` indexed directly by `ModelHandle` (handles are dense indices into `ModelManager`'s slot array); `instance_batches_touched_` records first-touched handles so per-frame clearing is O(touched), capacity preserved. Shadow batches replicate this per cascade (`shadow_instance_batches_[4]`, `shadow_batches_touched_[4]`; `MAX_SHADOW_CASCADES = 4`, static-asserted equal to `render::CSM_MAX_CASCADES`).

`upload_instance_buffers` (inside `FrameUploads`): main instances packed **in ascending handle order** straight into the mapped transfer buffer (no CPU pack scratch); the same order is replayed by `render_instanced_models`, so per-batch `base_instance` offsets line up. Storage buffers grow geometrically (`required * 2`); creation failure logs and disables the path. Shadow instances pack all cascades contiguously — `shadow_cascade_base_instance_[c]` records each cascade's first-instance offset.

`render_instanced_models`: binds `PipelineType::InstancedModel` (or `InstancedModelDoubleSided` per-mesh via glTF `doubleSided`, rebinding pass state each switch), pushes `InstancedCameraUniforms` once, binds the instance SSBO at vertex storage slot 0, shadow data (fragment samplers 1–2 + fragment uniform b1) and cluster lights (fragment storage buffers 0–2 + `ClusterParams` at fragment uniform b2), then per model × mesh issues one `SDL_DrawGPUIndexedPrimitives(index_count, instance_count, 0, 0, base_instance)`. Per-mesh `InstancedLightingUniforms` pushes are deduped via `memcmp` against the last-pushed copy. Meshes with `shadow_only`, missing buffers, or zero indices are skipped. glTF `BLEND` meshes cannot be depth-sorted within one instanced draw, so each instance is re-queued through the transparent path individually.

#### Skinned path and bone palettes

Palette layout happens at **build time** in the `FrameBuilder` (allocation, budget accounting, sharing-by-id, and the prefix-summed `matrix_offset` per `BonePalette` record — see the FramePacket section). The renderer's per-frame skinned resolve is therefore a straight walk over `frame.skinned_commands()`:

- Per command: skip invalid palette ids (already warned at emit) and unresolvable models; `const BonePalette& rec = frame.palettes()[cmd.palette.value]`; the palette GPU offset is `rec.matrix_offset` (no dedupe table, no running total — both are dead code since Phase 2); drawn joints clamp to `min(model joints, rec.joint_count)`.
- World bounds resolved once (center/radius as above) and reused by the main view *and* every shadow cascade.
- Output: `skinned_resolved_` — `{cmd*, Model*, world_center, world_radius, palette_offset}` — shared by the main pass and all shadow cascades. `MotionHistory` records `(cmd.motion_id, cmd.transform, rec.matrix_offset)` for the N/N−1 velocity pairing exactly as before.

Palette upload (inside `FrameUploads`): storage buffer sized `frame.palette_matrix_total() * sizeof(glm::mat4)`, grown geometrically with a **64 KB floor**; each `BonePalette` record memcpys `joint_count` matrices from **packet memory** into the mapped transfer buffer, in `palettes()` order (which is exactly `matrix_offset` order). No host storage is touched — the old span-lifetime contract is gone. GPU contract: one `StructuredBuffer<float4x4>` at **vertex storage slot 0**; draws index it via the `paletteOffset` field of `gpu::ModelTransformUniforms` / `gpu::ShadowSkinnedTransformUniforms` (`src/bracken/gpu/gpu_uniforms.hpp`). There is no GPU-side bone-count constant.

Main-pass skinned draws (`render_skinned_model_command_inner`): `ModelTransformUniforms{model, view, projection, cameraPos, normalMatrix, paletteOffset}` at vertex b0; per-mesh `ModelLightingUniforms` (memcmp-deduped) at fragment b0; material textures at fragment sampler slots 0 (base color), 3 (normal), 4 (metallic-roughness), 5 (emissive), with 1×1 fallbacks; shadow atlas at slots 1–2. `doubleSided` switches to `PipelineType::SkinnedModelDoubleSided`; `BLEND` meshes defer to the transparent queue.

#### Shadow passes

`render_shadow_passes`: one `SDL_GPURenderPass` over the whole 2×2 atlas; per active cascade, `set_cascade_viewport` then, in order: instanced casters (`render_instanced_shadow_models` — `InstancedShadowUniforms{lightViewProjection}` at vertex b0, per-cascade `base_instance` into the packed shadow SSBO, `ShadowFragUniforms{hasTexture}` + texture binding deduped across meshes; only `mesh.cast_shadows` meshes draw), non-instanced static casters (pre-resolved into `shadow_static_scratch_`, per-cascade sphere cull against the cascade light frustum, `ShadowTransformUniforms{lightViewProjection, model}`), skinned casters (reuse `skinned_resolved_` + the shared palette SSBO; depth-only pipeline, no fragment state; `ShadowSkinnedTransformUniforms` carries `paletteOffset`), then terrain. Per-cascade culling means near cascades skip distant casters entirely.

#### Transparent queue

`queue_transparent_mesh` records `{mesh*, transform, normal_matrix, tint, palette_offset (INVALID_PALETTE_OFFSET = 0xFFFFFFFF ⇒ static), no_fog, view_dist_sq}` keyed on the *model's* bounding-sphere center. `render_transparent_models` sorts descending by `view_dist_sq` (back-to-front) and draws with one of four pipeline variants (`ModelBlend[DoubleSided]` / `SkinnedModelBlend[DoubleSided]`), rebinding shadow/cluster/palette state on each pipeline switch. Depth test on, depth writes off. Limitations: no OIT; sort granularity is one mesh per command/instance; skinned entries silently skip if the palette buffer is missing. The standalone `render_model_command` path flushes the queue immediately after its command.

#### Debug lines

`upload_debug_lines`: vertex format = 7 floats (position xyz + color rgba via `core::unpack_color` — engine-wide ARGB), 2 vertices per line, built in a reused `std::vector<float>` scratch; buffer capacity grows to `max(line_count, 256)` doubling thereafter. `render_debug_lines`: grid pipeline, one `SDL_PushGPUVertexUniformData` of `projection * view`, single `SDL_DrawGPUPrimitives` for all lines.

#### Billboards

`draw_billboard_3d` (runs during the UI phase, after `render_ui_commands`): clip = `projection * view * world_pos`; rejected when `clip.w <= 0.01` or NDC outside `[-1.5, 1.5]` in x/y or `[-1, 1]` in z; screen position from NDC with y flipped; `distance_scale = clamp(100 / clip.w, 0.3, 1.5)`; bar size = `width*2*scale × width*0.4*scale`; draws three `ui_->draw_filled_rect` calls (frame at ±1 px, background, fill = `bar_w * fill_ratio`).

#### Uniform/binding summary (main-pass model family)

- Vertex b0: `ModelTransformUniforms` (per draw) or `InstancedCameraUniforms` (per pass); vertex storage slot 0: bone palette SSBO (skinned) or instance SSBO (instanced).
- Fragment b0: `ModelLightingUniforms`/`InstancedLightingUniforms` (lazy, memcmp-deduped; invalidated by `reset_model_bind_state()` after every pipeline bind); b1: `ShadowDataUniforms` (frame-cached); b2: `ClusterParams` (slot 2 reserved across all main-pass fragment shaders).
- Fragment samplers: 0 base color, 1–2 shadow atlas (compare + raw), 3 normal, 4 metallic-roughness, 5 emissive, 6 screen AO (1×1 white dummy when AO is off/BLEND), 7–9 sky-IBL chain (irradiance cube / prefiltered specular cube / BRDF LUT; typed dummies when IBL is off), 10 local-shadow atlas + comparison sampler (CSM atlas as filler when local shadows are unavailable). Slot 0 is always bound (unbound slots are UB); per-slot last-bound texture tracking (`last_frag_tex_`) skips redundant binds.
- CPU structs in `gpu/gpu_uniforms.hpp` must match their HLSL cbuffers byte-for-byte (compile-time `static_assert` block per struct; SPIR-V cross-check in `uniform_layout.test.cpp`).

#### RenderStats

`bracken::RenderStats` (`src/bracken/contracts/render_stats.hpp`): `draw_calls`, `triangle_count`, `entities_rendered`, `entities_distance_culled`, `entities_frustum_culled`, `jobs_pending`, `instances_drawn`, `ui_vertices`/`ui_draws`, `local_shadow_tiles_rendered`, the `*_used`/`*_budget` pairs, and `pass_times_ms` (`std::vector<PassTiming>` — `cpu_ms` real, `gpu_ms` `nullopt`; SDL3 has no per-pass timestamps). Collected only when `set_collect_stats(true)`; reset at the top of each `render_frame`. `SceneRenderer::frame_diagnostics()` (`FrameDiagnostics`) is reset and accumulated every frame regardless of that flag.

#### Threading & lifetime

`render_frame` asserts the main thread; all GPU work is recorded there. Contracts: `RenderContext` outlives the renderer; the consumed `FramePacket` is sealed, read-only, and carries no host-lifetime obligations (the caller retires it after the call); GPU resource owners are destroyed in `shutdown()` before the device. `shadow_static_scratch_` is a member (not `thread_local`) so it follows object lifetime and is resettable in tests.

#### Performance characteristics

Per-frame CPU cost is linear in command count: one cull + bin pass over static commands, one resolve pass over skinned commands, one sort over transparent draws (`std::sort`, typically tiny). Allocation-free steady state: bins/scratch preserve capacity; O(touched) bin clearing; palettes and instances memcpy straight into mapped transfer memory (no intermediate pack buffers — palettes come pre-packed from the packet); uniform pushes and texture binds are deduped (`memcmp` / pointer compare); `inverse(projection)` cached; normal matrices take the uniform-scale fast path.

### Constants (this chapter)

| Constant | Value | Where |
|---|---|---|
| `MAX_BONE_PALETTES` | 1024 | `scene/frame_packet.hpp` — bone palettes per frame, enforced at build time in `FrameBuilder::alloc_bone_palette` |
| `FramePacket::k_default_arena_bytes` | 256 KB | `scene/frame_packet.hpp` — initial packet arena size (grows once if outgrown; memory retained across resets) |
| `FrameBuilder::k_max_lanes` | 4096 | `scene/frame_builder.hpp` — emission lanes per builder (palette-id encoding reserves 12 bits) |
| `PacketRing` default depth | 2 | `scene/frame_packet_ring.hpp` — fill + consume slots; deeper pipelining is a constructor argument |
| `INVALID_PALETTE_OFFSET` | `0xFFFFFFFF` | `scene/scene_renderer.hpp` — transparent-queue marker for static draws |
| `MAX_SHADOW_CASCADES` | 4 | `scene/scene_renderer.hpp` — static-asserted == `render::CSM_MAX_CASCADES` |
| bone-palette buffer floor | 64 KB | `scene_renderer_upload.cpp` palette upload |
| debug-line buffer floor | 256 lines | `scene_renderer_draw.cpp` `upload_debug_lines` |
| non-AO clear color | `{0.35, 0.45, 0.6, 1.0}` | `scene_renderer.cpp` `begin_main_pass` |
| depth clear | 0.0 (reversed-Z far) | `scene_renderer.cpp` `begin_main_pass` |
| shadow resolution table | `{512, 1024, 2048, 4096}` | `contracts/graphics_settings.hpp` `shadow_map_texels(ShadowResolution)` — a `constexpr` mapping off the enum, not an index into a table at the call site |
| local-shadow atlas table | `{1024, 2048, 4096}` | `contracts/graphics_settings.hpp` `local_shadow_atlas_texels(LocalShadowResolution)` (reinit drops all cached tiles) |
| shadow far | `min(draw_distance, shadow_distance)` — `GraphicsSettings::shadow_distance`, default 4000, clamped [200, 20000] | `scene_renderer.cpp` `render_frame` |
| billboard scale clamp | `100/clip.w`, clamped 0.3–1.5 | `scene_renderer_draw.cpp` `draw_billboard_3d` |
| `MAX_BONES` (referenced) | 128 | `animation/animation_types.hpp` — upper bound of `alloc_bone_palette`'s `joint_count` |
| `MAX_LIGHTS` (referenced) | 256 | `render/lighting/light.hpp` — cluster-grid capacity |

### Known limitations / silent behaviors

- Bone-palette allocations beyond `MAX_BONE_PALETTES` fail at build time (error log + invalid id — no assert, no error return); `draw_skinned` with an invalid id drops the draw with a warning.
- `BonePaletteId`s held by the host across `seal()` are stale (fill-time encoding → dense index remap); only ids re-read from sealed commands are meaningful.
- Skinned commands whose model/skeleton can't be resolved are silently excluded.
- Distance culling is XZ-only; frustum culling is skipped for commands whose model can't be resolved.
- `pass_times_ms` is zeroed (timestamp stub).
- Cluster light SSBOs retain stale contents on zero-light frames; correctness relies on zeroed `ClusterParams` (`gridDim.w == 0`), not on clearing.
- No OIT; transparent sort granularity is per queued mesh keyed on the model bounding-sphere center; `BLEND` meshes in instanced batches degrade to one draw per instance.
- Single `PostUICallback` slot; `Application::init_dev_ui()` claims it.
- Screenshot capture stalls the GPU (fence wait) by design.
- `render_graph/` exists and is tested but `SceneRenderer` does not use it — passes are issued directly.

---

## Model Loading, Heightmaps & the glTF Contract

Scope: `src/bracken/model_loader.hpp` / `.cpp` (glTF loading, GPU upload,
`ModelManager` registry), `src/bracken/model_utils.hpp` / `.cpp` (transform +
tangent helpers), `src/bracken/heightmap.hpp` (header-only terrain heightmap),
and the authoring contract in `docs/skinned_mesh_format.md`. Tests:
`model_loader.test.cpp`, `model_utils.test.cpp`, `heightmap.test.cpp` (same
directory). Everything lives in namespace `bracken` (not a sub-namespace).

### Public API surface

#### Handles and enums

```cpp
using ModelHandle = uint32_t;
static constexpr ModelHandle INVALID_MODEL_HANDLE = 0;

enum class AlphaMode : uint8_t { Opaque, Mask, Blend };

using gpu::SkinnedVertex;   // re-exported vertex types
using gpu::Vertex3D;
```

`ModelHandle` is an index into `ModelManager`'s slot vector; slot 0 is
permanently reserved. Handles are dense, allocated in registration order, and
never recycled.

#### PBRMaterial (`src/bracken/model_loader.hpp`)

Mirrors glTF `pbrMetallicRoughness` plus alpha/cull state and deferred-upload
pixel storage:

| Field | Type | Default | Notes |
|---|---|---|---|
| `base_color_factor` | `glm::vec4` | `(1,1,1,1)` | |
| `metallic_factor` | `float` | `0.0f` | Engine default; **deviates from the glTF spec default of 1.0** — only matters for primitives with *no* material (with a material, tinygltf's spec defaults overwrite it) |
| `roughness_factor` | `float` | `0.85f` | Same deviation caveat (spec default 1.0) |
| `normal_scale` | `float` | `1.0f` | From `normalTexture.scale` |
| `occlusion_strength` | `float` | `1.0f` | From `occlusionTexture.strength` |
| `emissive_factor` | `glm::vec3` | `(0,0,0)` | Pre-multiplied by `KHR_materials_emissive_strength.emissiveStrength` when present |
| `alpha_mode` | `AlphaMode` | `Opaque` | From `alphaMode` string (`"MASK"`/`"BLEND"`) |
| `alpha_cutoff` | `float` | `0.5f` | glTF default |
| `double_sided` | `bool` | `false` | |
| `has_packed_occlusion` | `bool` | `false` | True iff `occlusionTexture.source == metallicRoughnessTexture.source` (ORM packing; AO in R of the MR texture — the **only** supported occlusion layout; a standalone AO map is warned about and dropped at load) |
| `*_pixels` / `*_width` / `*_height` | `std::vector<uint8_t>`, `int` | empty/0 | RGBA8 staging for metallic-roughness, normal, emissive; freed after `upload_to_gpu` |
| `*_texture` | `std::unique_ptr<gpu::GPUTexture>` | null | GPU textures after upload (metallic-roughness, normal, emissive) |

Defaults are asserted by `PBRMaterial.DefaultsMatchGltfSpec` and end-to-end
propagation by `ModelLoaderAlphaFlags.PropagatesAlphaModeCutoffAndDoubleSided`
in `src/bracken/model_loader.test.cpp`.

#### Mesh (`src/bracken/model_loader.hpp`)

One `Mesh` per glTF *primitive*. CPU-side data + GPU resources + flags:

- `std::vector<Vertex3D> vertices` / `std::vector<SkinnedVertex>
  skinned_vertices` (used when `is_skinned`) / `std::vector<uint32_t> indices`.
- `std::unique_ptr<gpu::GPUBuffer> vertex_buffer, index_buffer`;
  `std::unique_ptr<gpu::GPUTexture> texture` (base color); `PBRMaterial
  material`; `bool uploaded`.
- `bool has_texture` — base-color texture present.
- Perf flags: `cast_shadows = true` (false ⇒ shadow passes skip the mesh),
  `shadow_only = false` (true ⇒ main pass skips it; shadow passes render it —
  canopy shadow proxies). Consumers set these on procedural meshes
  (`src/bracken/procedural/tree_generator.cpp:426-474`); `load_glb` never sets
  them.
- `texture_pixels`/`texture_width`/`texture_height` — deferred base-color
  staging.
- `uint32_t cached_index_count` — snapshotted in `upload_to_gpu` before CPU
  index data is freed.
- `uint32_t vertex_count() const` — size of the active vertex vector;
  **returns 0 after upload** (vectors are freed, and no cached count exists
  for vertices).
- `uint32_t index_count() const` — `cached_index_count` if non-zero, else
  `indices.size()`.
- `void bind_buffers(SDL_GPURenderPass*) const`
  (`model_loader.cpp:42`) — binds the vertex buffer at slot 0 and the index
  buffer as `SDL_GPU_INDEXELEMENTSIZE_32BIT`; each bind is skipped if the
  corresponding buffer is null.

Vertex layouts (`src/bracken/gpu/gpu_types.hpp`):

```cpp
struct Vertex3D {                    // 64 bytes
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    glm::vec2 texcoord{0.0f};
    glm::vec4 color{1.0f};
    glm::vec4 tangent{1.0f, 0.0f, 0.0f, 1.0f}; // xyz + bitangent sign (glTF)
};
struct SkinnedVertex {               // Vertex3D fields + skinning
    /* ...same five fields... */
    uint8_t joints[4]{};
    float weights[4]{};
};
```

#### Model (`src/bracken/model_loader.hpp`)

- `std::vector<Mesh> meshes`; `bool loaded`.
- `AABB bounds` — `struct AABB { glm::vec3 min, max; }` with `size()`,
  `center()` and `empty()` (no extent in any axis, which is also the
  default-constructed state). `load_glb` seeds it ±1e10 and accumulates every
  parsed vertex — bounds come from actual vertex data, not accessor
  `min`/`max` metadata.
- Derived: `width()/height()/depth()/max_dimension()`;
  `compute_bounding_sphere()` fills `bounding_center` (AABB center) and
  `bounding_half_diag` (half the AABB diagonal length) — called at the end of
  `load_glb`, and by `register_model` for hand-built models.
- Animation payload: `animation::Skeleton skeleton`,
  `std::vector<animation::AnimationClip> animations`, `bool has_skeleton`,
  `animation::FootIKData foot_ik` (types in
  `src/bracken/animation/animation_types.hpp`; `MAX_BONES = 128` lives
  there).
- `int find_animation(const std::string&) const` — linear scan; `-1` if not
  found.

#### ModelLoader

```cpp
class ModelLoader {
public:
    [[nodiscard]] static bool load_glb(const std::string& path, Model& model);
    static void upload_to_gpu(gpu::GPUDevice& device, Model& model);
    static void free_gpu_resources(Model& model);
};
```

Stateless; all three are static. `load_glb` is CPU-only (usable headless — the
unit tests exercise it without a GPU).

#### ModelManager

```cpp
class ModelManager {                       // non-copyable, non-movable
    void set_device(gpu::GPUDevice* device);
    ModelHandle load_model(const std::string& name, const std::string& path);
    Model* get_model(ModelHandle handle) const;         // O(1)
    Model* get_model(const std::string& name) const;    // hash-map
    ModelHandle get_handle(const std::string& name) const;
    ModelHandle register_model(const std::string& name, std::unique_ptr<Model> model);
    [[nodiscard]] bool reload_model(ModelHandle handle);
    void enable_hot_reload(core::asset::FileWatcher& watcher);
    void unload_all();
};
```

Internal state: `std::vector<std::unique_ptr<Model>> models_` (slot 0 =
nullptr), parallel `std::vector<std::string> model_paths_` (`""` for
procedural models), `std::unordered_map<std::string, ModelHandle>
name_to_handle_`, `gpu::GPUDevice* device_`, `core::asset::FileWatcher*
watcher_`.

Behavior details (`model_loader.cpp:827-942`):

- `load_model`: `load_glb` → on success `upload_to_gpu` (if `device_` set) →
  push slot, record path, map name → if `watcher_` is set, register a
  `watch_file(path, ...)` that calls `reload_model(handle)`. Returns
  `INVALID_MODEL_HANDLE` on parse failure (nothing is registered).
- Duplicate names: `load_model` overwrites `name_to_handle_[name]` — the old
  model stays alive at its old handle but is unreachable by name.
  `register_model` refuses instead (`NameAlreadyRegistered`).
- `get_model(handle)`: null for `INVALID_MODEL_HANDLE` or out-of-range
  (tested).
- `register_model`: returns `core::Result<ModelHandle, ModelRegisterError>`.
  Refuses a model with no geometry (`EmptyModel` — checked BEFORE the upload,
  which frees the CPU-side vertex arrays), a name already registered
  (`NameAlreadyRegistered`), or a mesh that did not reach the GPU
  (`UploadFailed`, only when a device is set). On the way through it derives
  `bounds` from the mesh vertices when they are empty, calls
  `compute_bounding_sphere()` and sets `loaded`. Stores with an empty path.
  `to_string(ModelRegisterError)` gives the message.
- `reload_model`: rejects invalid handles and empty (procedural) paths; parses
  into a *fresh* `Model`; on parse failure logs a
  `hot_reload` warning and returns `false` leaving the old model untouched; on
  success uploads the new model, calls `free_gpu_resources` on the old one,
  then **move-assigns the new contents into the existing `Model` object** —
  raw `Model*` handed to callers stay valid across reloads (the header
  promises pointer stability; only the model's *contents* are replaced —
  pointers *into* the model are still invalidated).
- `enable_hot_reload`: stores the watcher and registers watches for every
  already-loaded file-backed model; future `load_model` calls self-register.
  Watches are never unregistered (documented in-code: "watched for the
  lifetime of the app").
- `unload_all`: `free_gpu_resources` on every model, clears all vectors/maps,
  re-reserves slot 0. Destructor calls it. Note it releases GPU handles via
  RAII resets — callers must ensure the device outlives the manager teardown.

#### model_utils (`src/bracken/model_utils.hpp`)

```cpp
glm::mat4 build_model_transform(const Model& model, const glm::vec3& position,
                                float yaw_radians, float target_size,
                                float attack_tilt = 0.0f);
void compute_tangents_from_uvs(std::vector<gpu::Vertex3D>& vertices,
                               const std::vector<uint32_t>& indices);
```

#### Heightmap (`src/bracken/heightmap.hpp` + `.cpp`)

```cpp
struct WorldBounds { float origin_x = 0.0f, origin_z = 0.0f, size = 0.0f; };
struct HeightRange { float min = 0.0f, max = 0.0f; };

struct Heightmap {
    uint32_t resolution = 0;
    float world_origin_x = 0.0f, world_origin_z = 0.0f, world_size = 0.0f;
    float min_height = -500.0f, max_height = 500.0f;
    std::vector<uint16_t> height_data;   // row-major, z * resolution + x

    static Heightmap from_floats(std::span<const float>, uint32_t resolution, WorldBounds);
    static Heightmap from_floats(std::span<const float>, uint32_t resolution, WorldBounds, HeightRange);
    bool  valid() const;
    float get_height_local(uint32_t local_x, uint32_t local_z) const;
    float get_height_world(float world_x, float world_z) const;
    glm::vec3 get_normal_world(float world_x, float world_z) const;
};
```

- `from_floats` (the only non-inline members, in `heightmap.cpp`): validates
  `resolution >= 2` and `heights.size() == resolution * resolution` — either
  failure logs `BRACKEN_LOG_ERROR("heightmap", ...)` and returns a default
  (invalid) map. The three-argument form scans the data for the range; the
  four-argument form takes one and CLAMPS out-of-range heights. The packing is
  deliberately the arithmetic hosts hand-rolled — `(h - min) / (max - min)`
  (a division, not a reciprocal multiply, which is not bit-identical) and a
  truncating `static_cast<uint16_t>(normalized * 65535.0f)` — so moving a host
  onto it reproduces its terrain sample for sample
  (`heightmap.test.cpp::ExplicitRangeMatchesTheHandRolledQuantizationBitForBit`).
- `valid()`: `resolution >= 2 && height_data.size() == resolution^2`.
  `Application::set_heightmap` refuses anything else with a logged error.

There is deliberately no engine-side heightmap *file* loader: game code
produces world-unit floats and `from_floats` owns the packing. The playground
synthesizes one (`examples/playground/playground.cpp`, `build_heightmap`,
257×257, explicit range `{-160, 150}`).

### load_glb: parsing pipeline

Order of operations in `ModelLoader::load_glb` (`model_loader.cpp:126-703`).
Profiled with `BRACKEN_PROFILE_ZONE` per phase (`parse_mesh`,
`parse_skeleton`, `parse_animation`).

#### tinygltf configuration and image decoding

- tinygltf is compiled into this TU (`TINYGLTF_IMPLEMENTATION`,
  `TINYGLTF_NO_STB_IMAGE`, `TINYGLTF_NO_STB_IMAGE_WRITE` at
  `model_loader.cpp:29-32`). Any other TU including `tiny_gltf.h` must repeat
  the two `NO_STB` defines or link errors follow (documented in
  `model_loader.test.cpp:5-10`).
- **No Draco support** — the bundled tinygltf has no decoder; Draco-compressed
  files fail at `LoadBinaryFromFile`.
- Images decode through a custom loader (`load_image_data`,
  `model_loader.cpp:58`) using SDL3_image (`IMG_Load_IO`) and are converted to
  `SDL_PIXELFORMAT_RGBA32`, so `image.component == 4`, 8 bits/channel, always.
  A writer stub always returns false (writing unsupported).
- `LoadBinaryFromFile` only — the loader is **GLB-binary-only**; `.gltf`
  text files are not loaded (the `ModelManager` hot-reload docs say the
  same).
- Load failure paths: non-empty tinygltf `err` or `ret == false` →
  `BRACKEN_LOG_ERROR("model", ...)`, return `false`. Warnings are logged and
  loading continues.

#### Texture budget and image indexing

Each embedded image larger than **2048×2048** logs a warning (budget guard)
but still loads. Base-color lookups index `gltf.images` directly by the raw
glTF image index (`texture_image_idx = gltf.textures[tex_idx].source`) — like
every other texture type — so an empty image (e.g. a failed external-URI
resolve) can never shift later meshes onto the wrong image; it just warns
(`base-color image N has no pixel data; mesh renders untextured`) and clears
`has_texture` for that mesh.

#### Per-primitive mesh extraction

For each `gltf.meshes[*].primitives[*]` (the node hierarchy is **never
walked**; node TRS is not applied to mesh geometry — bake transforms at
export):

1. **Material** (if `primitive.material` valid): base-color factor →
   `material.base_color_factor`; metallic/roughness factors; normal scale;
   occlusion strength; texture source indices for base color,
   metallic-roughness (glTF packing: G = roughness, B = metallic), normal,
   occlusion (ORM detection: same source as MR ⇒ `has_packed_occlusion`, no
   separate pixel copy; a **standalone occlusion texture is warned about and
   dropped** — unsupported), emissive (factor ×
   `KHR_materials_emissive_strength` when the extension is present);
   `alphaMode`/`alphaCutoff`/`doubleSided`. Pixels for the PBR maps are copied
   straight from `gltf.images[src].image` into the material's staging vectors.
   Pipeline selection from these flags is the renderer's job (data only here).
2. **Accessor guard** (`get_packed_data`, `model_loader.cpp:298`): decoders
   assume tightly packed data. If `bufferView.byteStride != 0` and differs
   from the tight stride (`componentSize × numComponents`), the attribute is
   skipped with a `BRACKEN_LOG_WARN("model", ...)` rather than decoding
   garbage.
3. **Attributes**: `POSITION` (float3, mandatory — primitive skipped entirely
   if absent or stride-rejected); `NORMAL` (float3, default `(0,1,0)`);
   `TEXCOORD_0` (float2, default `(0,0)`); `TANGENT` (float4, default
   `(1,0,0,1)` then recomputed, see below); `COLOR_0` — must be VEC4, either
   float4 or normalized u8×4 (anything else warned + skipped); missing color
   is baked **white** (`baseColorFactor` is applied in the shader on both the
   textured and untextured paths). Every position updates the model AABB.
4. **Indices**: component types UNSIGNED_SHORT and UNSIGNED_INT are decoded;
   any other component type falls through to a byte read (`default:` treats
   the buffer as u8). Unindexed primitives keep `indices` empty.
5. **Tangent fallback**: if `TANGENT` was absent,
   `compute_tangents_from_uvs(out_mesh.vertices, out_mesh.indices)` runs.
6. **Base-color staging**: `texture_pixels/width/height` copied from
   `gltf.images[texture_image_idx]` (raw glTF image index; an empty image
   warns and clears `has_texture` — see "Texture budget and image indexing"
   above).
7. **Skinning**: requires *both* `JOINTS_0` and `WEIGHTS_0`. Constraints:
   `JOINTS_0` componentType must be UNSIGNED_BYTE and `WEIGHTS_0` FLOAT —
   otherwise `BRACKEN_LOG_WARN` and the mesh ships **unskinned** (silent
   downgrade, no failure). On success `is_skinned = true` and every `Vertex3D`
   is copied into a `SkinnedVertex` with 4 joints/weights. Note both vertex
   vectors remain populated until GPU upload (double memory for skinned meshes
   during load).

#### Skeleton (skins[0] only)

If `gltf.skins` is non-empty, **only `skins[0]`** is used; additional skins
are ignored silently.

- Inverse bind matrices are memcpy'd (16 floats each) from the
  `inverseBindMatrices` accessor via the validated float-accessor helper
  (`get_float_accessor` — rejects sparse accessors, non-FLOAT component
  types, and unexpected element types with a warning); joints beyond the
  accessor count (or with an unsupported accessor) get identity.
- `skeleton.joint_node_indices = skin.joints`; a `node → joint` map is built.
- Per joint: name from the node; parent found by scanning **all** nodes'
  children lists (O(joints × nodes × children)) and mapping through
  `node_to_joint` (parents outside the skin ⇒ `parent_index = -1`); local TRS
  read from the node (`translation` default `(0,0,0)`, `rotation` default
  identity — glTF stores xyzw, converted to glm's wxyz — `scale` default
  `(1,1,1)`).
- **Joint-order normalization**: the animation/IK runtime propagates the
  hierarchy in a single index-order pass (parents-before-children required).
  glTF guarantees no ordering of `skin.joints`, so out-of-order skins are
  normalized at load with a **stable topological reorder**; vertex joint
  indices referencing the old order are remapped. A cyclic joint hierarchy is
  a hard load failure (`ERROR`, `load_glb` returns `false`).
- `model.foot_ik.init(model.skeleton)` caches IK bone indices using the
  default `FootIKBoneNames` (`Hips`, `Spine`, `Left/RightUpperLeg`,
  `Left/RightLowerLeg`, `Left/RightFoot` —
  `src/bracken/animation/animation_types.hpp`); `foot_ik.valid` requires all
  eight. Hosts with other rigs must re-init with custom names.

#### Animations

Per `gltf.animations[*]` (`model_loader.cpp:615-689`):

- Channels are grouped **by target joint**: `channel.target_node` is mapped
  through the skin's `node_to_joint`; **channels targeting nodes outside the
  skin's joint list are silently dropped** (`continue`, no log). This is the
  documented behavior in `docs/skinned_mesh_format.md` §"Animation channels".
- Sampler input (times) and output (values) are read through the validated
  float-accessor helper (sparse accessors, non-FLOAT component types —
  including valid-core-glTF normalized u8/u16 quantized outputs — and wrong
  element types are warned and dropped).
- Sampler `interpolation`: `LINEAR` (default) and `STEP` are honored
  (`animation::Interpolation`). `CUBICSPLINE` stores 3 output elements per
  key (in-tangent, value, out-tangent); it is **degraded to linear** by
  reading the middle (value) element per key, with a warning once per model.
- `target_path` handled: `translation` (vec3), `rotation` (quat, xyzw→wxyz),
  `scale` (vec3). Other paths (`weights`/morph targets) are ignored. Clip
  `duration` = max keyframe time seen across all channels.
- A clip whose channels were *all* dropped (empty `clip.channels`) is **not
  added** to `model.animations` — the clip disappears without a log line.

#### Completion

`model.loaded = true`, `compute_bounding_sphere()`, then
`BRACKEN_LOG_INFO("model", "Loaded GLB model: {} ({} meshes) bounds: Y=[{}, {}]", ...)`.
Return `true`.

### upload_to_gpu

`ModelLoader::upload_to_gpu(gpu::GPUDevice&, Model&)`
(`model_loader.cpp:705-810`). Idempotent per mesh (`uploaded` flag); meshes
with no vertices are skipped.

| Resource | Creation call | Format | Mipmaps | Failure behavior |
|---|---|---|---|---|
| Vertex buffer | `gpu::GPUBuffer::create_static(device, Type::Vertex, data, size)` | `SkinnedVertex[]` if `is_skinned && !skinned_vertices.empty()`, else `Vertex3D[]` | — | log error, `continue` (mesh left un-uploaded, CPU data intact) |
| Index buffer | `create_static(..., Type::Index, ...)` | `uint32_t[]` | — | log error, upload continues without it |
| Base color | `gpu::GPUTexture::create_2d(device, w, h, TextureFormat::RGBA8_SRGB, pixels, true)` | sRGB (sampler linearizes) | yes | log error, `has_texture = false` |
| Normal map | `create_2d(..., TextureFormat::RGBA8, ..., true)` | linear UNORM (per glTF spec) | yes | silent (pixels retained) |
| Metallic-roughness | `create_2d(..., TextureFormat::RGBA8, ..., true)` | linear UNORM; G=roughness, B=metallic, R=AO iff `has_packed_occlusion` | yes | silent |
| Emissive | `create_2d(..., TextureFormat::RGBA8_SRGB, ..., true)` | sRGB | yes | silent |

After successful upload per mesh: staging pixel vectors are cleared +
`shrink_to_fit`; `cached_index_count` snapshots `indices.size()`; then
`vertices`, `skinned_vertices`, and `indices` are cleared + shrunk. So
post-upload, CPU geometry is gone (`vertex_count()` → 0; `index_count()` uses
the cache). `Model::skeleton`, `animations`, bounds, and bounding sphere
persist.

glTF *sampler* objects (wrap/filter) are never read; the renderer binds its
own samplers.

### free_gpu_resources

Resets all GPU `unique_ptr`s per mesh (vertex/index buffers, base color
texture, normal/MR/emissive material textures), sets
`uploaded = false` and `has_texture = false`. Used before device teardown and
by `reload_model` on the outgoing model. It does *not* restore CPU-side data —
a freed model can only be revived by re-parsing the file.

### model_utils algorithms

#### build_model_transform (`model_utils.cpp:10-27`)

```cpp
scale = (target_size * 1.5f) / model.max_dimension();   // div-by-zero if bounds unset
M = T(position) · R_y(yaw) · [R_x(attack_tilt) if != 0] · S(scale) · T(-pivot)
pivot = (bounds.center().x, bounds.min.y, bounds.center().z)
```

Guarantees (verified in `model_utils.test.cpp`): the model's bottom-center
maps exactly to `position`; column 3 holds the world image of the model-space
origin; `attack_tilt = 0` produces a bit-identical matrix to omitting the
parameter. The ×1.5 factor is an intentional fudge (models render slightly
larger than `target_size`).

#### compute_tangents_from_uvs (`model_utils.cpp:29-96`)

Classic per-triangle UV-gradient tangent accumulation + per-vertex
Gram-Schmidt:

1. Early-out on `vertices.empty() || indices.size() < 3` (no-op).
2. Per triangle: skip if any index out of range, or if the UV determinant
   `|x1·y2 − x2·y1| < 1e-8`; otherwise accumulate tangent/bitangent
   directions into per-vertex sums.
3. Per vertex: normalize the normal (`len < 1e-6` ⇒ fallback tangent
   `(1,0,0,+1)`); orthogonalize `t = tan − n·dot(n,tan)`; if `‖t‖ < 1e-6`
   pick a deterministic perpendicular (`cross((0,1,0) or (1,0,0), n)`
   depending on `|n.y| < 0.9`); sign `w = dot(cross(n,t), btn) < 0 ? −1 : +1`
   (mirrored UVs → −1, per glTF convention).

Not MikkTSpace-correct across UV seams (declared "fine for v1" in the
header). Handedness/orientation properties are pinned by four tests in
`model_utils.test.cpp`.

### Heightmap specification

#### Data model

Square grid, `resolution` samples per side, row-major
(`height_data[z * resolution + x]`), `uint16_t` per sample. Decode:

```cpp
height = raw / 65535.0f * (max_height - min_height) + min_height;
```

Vertical quantization step = `(max_height − min_height) / 65535` (defaults
−500..+500 ⇒ ≈15.3 mm; the playground's −100..+150 ⇒ ≈3.8 mm). Grid spacing =
`world_size / (resolution − 1)` world units. `min_height == max_height`
degenerates gracefully to a constant field (range 0 ⇒ every raw value decodes
to `min_height`; tested).

#### Sampling semantics

- `get_height_local(x, z)`: bounds-checked; **out-of-range returns `0.0f`**
  (not clamped — a real footgun if terrain edge height ≠ 0; tested behavior).
- `get_height_world(wx, wz)`: normalizes to `u,v ∈ [0,1]` (clamped — world
  queries outside the map return edge values), scales to grid space,
  **bilinearly interpolates** the four surrounding samples. `resolution` must
  be ≥ 1 and `world_size > 0` (u,v become inf/NaN at `world_size == 0`; no
  guard).
- `get_normal_world`: returns `glm::vec3`; central differences with
  `eps = world_size / (resolution − 1)` (one grid cell):
  `n = normalize(hL−hR, 2·eps, hD−hU)`; if `‖n‖ ≤ 1e-4` returns `(0,1,0)`. Always unit length otherwise (tested,
  including at map corners where clamping degrades the difference to
  one-sided).

The sampling methods are `const`, header-inline, allocation-free, and safe to
call from any thread that isn't concurrently mutating the struct;
`from_floats` is a static factory that allocates once.

#### Renderer integration

`Application::set_heightmap(const Heightmap&)`
(`src/bracken/application.cpp:191`) → `SceneRenderer::set_heightmap`
(`src/bracken/scene/scene_renderer.cpp:360`) →
`TerrainRenderer::set_heightmap`
(`src/bracken/render/terrain_renderer.cpp:68`), which:

1. **Deep-copies** the heightmap (`std::make_unique<bracken::Heightmap>(heightmap)`)
   — the caller's struct has no lifetime obligation after the call.
2. Uploads `height_data` as a `resolution × resolution`
   `gpu::TextureFormat::R16` texture (`SDL_GPU_TEXTUREFORMAT_R16_UNORM`, 2
   bytes/texel, no mipmaps), preserving full 16-bit precision.
3. Regenerates the terrain mesh (high LOD + shadow LOD).
4. `SceneRenderer` then forwards `render::HeightmapParams`
   (origin/size/min/max) plus the R16 texture to the grass renderer.

CPU queries against the renderer's copy are exposed as
`Application::get_terrain_height(x, z)`. The playground keeps its own
`Heightmap` member for gameplay math and calls `set_heightmap(heightmap_)`
once in `on_init`.

### Threading & lifetime contracts

- **Main thread only.** Nothing in this area is synchronized. `ModelManager`
  hot-reload callbacks run on the main thread via
  `FileWatcher::poll_main_thread()` (the watcher's API is documented
  main-thread-only in `src/bracken/core/asset/file_watcher.hpp`), which
  `Application` pumps at the top of the frame loop.
- `upload_to_gpu` / `free_gpu_resources` require a live `gpu::GPUDevice`;
  uploads use `GPUBuffer::create_static` / `GPUTexture::create_2d` (copy pass
  + fence internally — not callable inside a render pass).
- `ModelManager` is non-copyable/non-movable so `ModelHandle`s and the slot
  vector stay stable. `Model*` stability holds *except across
  `reload_model`*, which replaces the pointed-to object (re-fetch by handle
  after any possible hot reload).
- `Model` itself is movable/copyable by default rules, but copying an
  uploaded model would double-own GPU resources via `unique_ptr` — copy is in
  fact implicitly deleted by the `unique_ptr` members; moves are fine before
  registration.
- Freeing GPU resources while the GPU may still reference them is the
  caller's problem; `reload_model` frees the old model's resources
  immediately after uploading the new one (no frame-lag protection).

### Error handling summary

This module predates the `core::Result` convention and uses
`bool` + `BRACKEN_LOG_*` throughout (consistent with CLAUDE.md's "existing
bool+log APIs stay as they are"):

| API | Failure signal | Log category |
|---|---|---|
| `ModelLoader::load_glb` | `false` | `model` (error) |
| `ModelLoader::upload_to_gpu` | none (void); per-resource errors logged, mesh skipped or texture dropped | `model` (error) |
| `ModelManager::load_model` | `INVALID_MODEL_HANDLE` | via `load_glb` |
| `ModelManager::reload_model` | `false`, old model untouched | `hot_reload` (warn/info) |
| `get_model` / `get_handle` | `nullptr` / `INVALID_MODEL_HANDLE` | — |
| attribute/skinning decode problems | attribute or skinning silently skipped | `model` (warn) |
| `Heightmap` queries | out-of-range local reads → `0.0f`; world reads clamp | — |

### Performance characteristics

- `load_glb` is single-threaded synchronous parsing; per-phase Tracy zones
  (`BRACKEN_PROFILE_ZONE`) exist for mesh/skeleton/animation parsing and GPU
  upload. Image decode goes through SDL3_image + one full RGBA conversion +
  one memcpy per image; PBR staging copies each image again per *primitive*
  that references it (no cross-mesh texture dedup — N primitives sharing one
  material upload N copies of its textures).
- Joint-parent resolution is O(J × N × C) over joints/nodes/children —
  irrelevant at ≤128 joints.
- `get_model(handle)` is a bounds-checked vector index; `get_model(name)` /
  `get_handle` are one hash lookup.
- `Heightmap::get_height_world` = 4 decodes + 3 lerps; `get_normal_world` = 4
  world samples (16 decodes).
- Memory: skinned meshes hold both `vertices` and `skinned_vertices` between
  parse and upload; everything CPU-side is freed at upload.

### Known limitations / silent behaviors (consolidated)

1. **No Draco** (no decoder in bundled tinygltf) — load fails.
2. **Node hierarchy ignored for meshes** — node TRS never applied to
   geometry; bake at export. (Y-up, `+Y up`/`-Z forward` assets load with no
   rotation applied — `docs/skinned_mesh_format.md`.)
3. **Animation channels targeting non-skin nodes silently dropped**; clips
   with zero surviving channels vanish from `Model::animations` without a
   log.
4. **Only `skins[0]`** becomes the skeleton; other skins ignored.
5. **Skinning downgrade**: non-u8 joints or non-float weights ⇒ mesh ships
   unskinned (warn).
6. **Interleaved accessors unsupported**: non-tight `byteStride` ⇒ attribute
   skipped (warn); on `POSITION`, the primitive is skipped. Inverse-bind and
   animation sampler accessors go through their own validation (sparse,
   non-FLOAT, wrong element type ⇒ warned and dropped).
7. **Separate occlusion textures are warned about and dropped at load**
   (only ORM-packed AO works).
8. **glTF sampler state ignored** (wrap/filter); engine samplers rule.
9. **>2048×2048 embedded images** warn but still load (budget guard).
10. **Index component types** other than u16/u32 are read as u8 (glTF's u8
    indices work; malformed types decode garbage rather than failing).
11. **Handle slots are never recycled**; `unload_all` is the only teardown.
    `load_model` still orphans the old name mapping on a duplicate name;
    `register_model` refuses one (`ModelRegisterError::NameAlreadyRegistered`).
12. **`build_model_transform` divides by `max_dimension()`** — zero for a
    bounds-less model (no guard).
13. **`Heightmap` has no world_size validation**; `world_size == 0` yields
    NaN/inf UVs, `get_height_local` OOB returns `0.0f` rather than clamping.
    Resolution IS validated at the two entry points that matter:
    `Heightmap::from_floats` refuses to build below 2x2 and
    `Application::set_heightmap` refuses to install an invalid map (`valid()`).
14. Engine material defaults for *material-less* primitives
    (`metallic 0.0`, `roughness 0.85`) deviate from glTF spec defaults
    (1.0/1.0); with a material present, spec defaults flow through tinygltf.

---

## Skeletal Animation, State Machines & IK

### Scope and file inventory

Module `src/bracken/animation/`, namespace `bracken::animation`. No GPU code lives here — the module produces CPU-side matrix palettes that `scene::SceneRenderer` uploads (see "Renderer integration" below).

| File | Contents |
|---|---|
| `animation_types.hpp` | constants, `Interpolation`, `AnimationChannel`, `ChannelCursors`, `AnimationClip`, `Joint`, `Skeleton`, `FootIKSmoother`, `FootIKBoneNames`, `FootIKData`, `RotationSmoother`, `ProceduralConfig`, `ParamType`/`ParamValue`, `interpolate_keyframes<T>` (header-only) |
| `animation_player.hpp/.cpp` | `AnimationPlayer` |
| `animation_state_machine.hpp/.cpp` | `TransitionCondition`, `StateTransition`, `AnimState`, `AnimationStateMachine` |
| `ik_solver.hpp/.cpp` | `solve_two_bone_ik`, `apply_foot_ik`, `apply_body_lean` |
| `*.test.cpp` | GoogleTest units (`animation_types.test.cpp`, `animation_player.test.cpp`, `animation_state_machine.test.cpp`, `ik_solver.test.cpp`, `animation_parallel.test.cpp`) |
| `animation.bench.cpp`, `ik.bench.cpp` | Google Benchmark suites |

Related: `src/bracken/model_loader.{hpp,cpp}` (GLB → `Skeleton`/`AnimationClip`), `src/bracken/scene/frame_packet.hpp` (`SkinnedDrawCommand`/`BonePalette`), `src/bracken/scene/frame_builder.hpp` (`alloc_bone_palette`/`draw_skinned`), `src/bracken/scene/scene_renderer.{hpp,cpp}` (palette upload), `docs/skinned_mesh_format.md` (authoring contract), `docs/architecture.md` §Animation.

### Constants

| Constant | Value | Where | Caps |
|---|---|---|---|
| `MAX_BONES` | `128` | `animation_types.hpp:15` (sole definition) | joints per skinned mesh on the CPU side; sizes `std::array<glm::mat4, MAX_BONES>` ×2 in `AnimationPlayer`; also the `joint_count` upper bound in `FrameBuilder::alloc_bone_palette` |
| `MAX_BONE_PALETTES` | `1024` | `src/bracken/scene/frame_packet.hpp` | bone palettes per frame; enforced at build time in `alloc_bone_palette` |
| `IK_THRESHOLD` | `0.1f` | `ik_solver.cpp::apply_foot_ik` (function-local) | world-space deadzone per foot offset |
| `IK_MAX_CORRECTION` | `8.0f` | `ik_solver.cpp::apply_foot_ik` (function-local) | world-space clamp per foot offset |
| reach margins | `0.01f` | `ik_solver.cpp::solve_two_bone_ik` | max reach = `len_ab+len_bc-0.01`, min reach = `|len_ab-len_bc|+0.01` |
| EQ/NE epsilon | `0.001f` | `animation_state_machine.cpp::TransitionCondition::evaluate` | float equality tolerance |
| lean deadzone | `0.001f` | `ik_solver.cpp::apply_body_lean` | skip threshold on both lean angles |

### Core data types (`animation_types.hpp`)

```cpp
enum class Interpolation : uint8_t { Linear, Step };

struct AnimationChannel {
    int bone_index{};                    // index into Skeleton::joints
    std::vector<float> position_times;   std::vector<glm::vec3> positions;
    std::vector<float> rotation_times;   std::vector<glm::quat> rotations;
    std::vector<float> scale_times;      std::vector<glm::vec3> scales;
    Interpolation position_interp = Interpolation::Linear;
    Interpolation rotation_interp = Interpolation::Linear;
    Interpolation scale_interp = Interpolation::Linear;
};
struct ChannelCursors { size_t pos = 0, rot = 0, scale = 0; };
struct AnimationClip { std::string name; float duration{}; std::vector<AnimationChannel> channels; };

struct Joint {
    std::string name;
    int parent_index{};                  // -1 = root
    glm::mat4 inverse_bind_matrix{};
    glm::vec3 local_translation{}; glm::quat local_rotation{}; glm::vec3 local_scale{};
};
struct Skeleton { std::vector<Joint> joints; std::vector<int> joint_node_indices; };
```

- **Cursor ownership**: `ChannelCursors` live on the player, not the clip — clips are const-shareable across any number of players; sharing cursors on the channel was a historical data race / glitch source (`docs/engine_design_notes.md` §Animation correctness).
- `Skeleton::joint_node_indices` records the glTF node index per joint (used at load time to map animation channel targets to joint indices; kept on the type).
- `ParamValue` is a **tagged union** (`ParamType::Float`/`Bool` + `union { float f; bool b; }`), chosen over `std::variant` explicitly to avoid per-frame overhead on the transition hot path (header comment). `as_float()`/`as_bool()` are type-checked and return `0.0f`/`false` on mismatch.

Host-side helper structs (all header-only, POD-ish, no engine consumption):

- `FootIKSmoother`: `smoothed_left_offset`, `smoothed_right_offset`, `smooth_rate = 15.0f`. `update(l, r, dt)` blends both toward targets with factor `min(1, dt * smooth_rate)`.
- `FootIKBoneNames`: defaults `Hips`, `Spine`, `LeftUpperLeg`, `LeftLowerLeg`, `LeftFoot`, `RightUpperLeg`, `RightLowerLeg`, `RightFoot`. Header comment claims "Defaults match Mixamo conventions"; note raw Mixamo exports use `mixamorig:`-prefixed names (`mixamorig:LeftUpLeg` etc.), and `docs/skinned_mesh_format.md` accordingly says such rigs must supply an override — the header comment overstates the defaults.
- `FootIKData`: 8 joint indices (init `-1`) + `bool valid`. `init(skel, names = {})` does a single linear pass over `skel.joints` with exact string equality; `valid` requires all eight indices ≥ 0.
- `RotationSmoother`: `current`, `turn_rate`, `initialized`. `smooth_toward(target, dt, speed = 12.0f)`: first call snaps; otherwise exponential blend `1 - exp(-speed*dt)` along the wrap-aware shortest angular path (`fmod(diff + π, 2π) − π`); `turn_rate = step/dt` guarded by `dt > 0.0001`. `decay_turn_rate(factor = 0.9f)` multiplies `turn_rate`.
- `ProceduralConfig`: `foot_ik = true`, `lean = true`, `forward_lean_factor = 0.015f`, `forward_lean_max = 0.18f`, `lateral_lean_factor = 0.06f`, `lateral_lean_max = 0.15f`, `attack_tilt_max = 0.4f`, `attack_tilt_cooldown = 0.5f`. Referenced only by its own test; the engine and the playground never read it — pure host-side convenience.

### Keyframe interpolation (`interpolate_keyframes<T>`)

```cpp
template<typename T>
T interpolate_keyframes(const std::vector<float>& times, const std::vector<T>& values,
                        float t, size_t& last_cursor, Interpolation interp = Interpolation::Linear);
template<typename T>
T interpolate_keyframes(const std::vector<float>& times, const std::vector<T>& values, float t,
                        Interpolation interp = Interpolation::Linear); // cursor = 0
```

Algorithm (`animation_types.hpp`):

1. Empty inputs → value-initialized `T{}` (verified zero for `glm::vec3` in `animation_types.test.cpp`).
2. Single key or `t <= times.front()` → first value, cursor 0. `t >= times.back()` → last value, cursor `size-2`.
3. **Warm-cursor fast path**: if `times[last_cursor] <= t < times[last_cursor+1]`, reuse the bracket (O(1)); otherwise `std::upper_bound` (O(log n)) and cache the new bracket.
4. `Interpolation::Step` → return the bracket's left value (hold). Otherwise linear factor `f = (t-t0)/(t1-t0)`; `glm::slerp` when `T == glm::quat` (compile-time `if constexpr`), otherwise `glm::mix`.

Steady playback therefore costs O(1) per channel per frame; seeks/wraps cost one binary search. `animation.bench.cpp` benchmarks warm vs cold cursor for both `glm::quat` and `glm::vec3` at 16/256 keys.

### `AnimationPlayer` (`animation_player.hpp/.cpp`)

#### API surface

```cpp
class AnimationPlayer {
public:
    AnimationPlayer();                                    // reset() — all matrices identity
    AnimationPlayer(const AnimationPlayer&) = delete;     // move-only by design
    AnimationPlayer& operator=(const AnimationPlayer&) = delete;
    AnimationPlayer(AnimationPlayer&&) noexcept = default;
    AnimationPlayer& operator=(AnimationPlayer&&) noexcept = default;

    void update(const Skeleton& skeleton, const std::vector<AnimationClip>& clips, float dt);
    void crossfade_to(int clip_index, float duration = 0.2f);
    void reset();

    void set_speed(float);  float speed() const;          // default 1.0
    void set_playing(bool); bool is_playing() const;      // default true
    void set_loop(bool);    bool loops() const;           // default true
    void seek(float t);     float current_time() const;   // raw write, no clamp

    int current_clip() const;   // default 0
    int prev_clip() const;      // -1 when not blending
    float prev_time() const;  float blend_factor() const;  float blend_duration() const;

    std::span<const glm::mat4> bone_matrices() const;      // size == MAX_BONES
    std::span<const glm::mat4> world_transforms() const;   // size == MAX_BONES
    std::span<glm::mat4> mutable_bone_matrices();
    std::span<glm::mat4> mutable_world_transforms();
    size_t used_joint_count() const;                       // written prefix; tail is identity
};
```

#### Memory layout

Per instance: `std::array<glm::mat4, MAX_BONES>` ×2 = 2 × 8192 B = **16 KB** of matrix storage, plus five vectors sized to the skeleton's joint count (`channel_lookup_`, `prev_channel_lookup_` of `const AnimationChannel*`; `current_cursors_`, `prev_cursors_` of `ChannelCursors`; `local_transforms_` of `glm::mat4`), plus scalars. Move-only rationale (header comment): a copied player sharing keyframe cursors with the original was a historical glitch source.

#### `update(skeleton, clips, dt)` algorithm

1. Early return if `clips.empty()` or `!playing_` (pose freezes; no recompute).
2. Out-of-range `current_clip_` is clamped to `0`.
3. `time_ += dt * speed_`. Zero/negative-duration clip: `time_ = 0`, `playing_ = false` (explicit guard — `fmod` by 0 would be UB). Otherwise on `time_ >= duration`: loop → `fmod`; non-loop → clamp to duration and `playing_ = false`.
4. If mid-blend (`blend_factor_ < 1`): `blend_factor_ += dt / blend_duration_` (**raw `dt`, not speed-scaled**; `blend_duration_ <= 0` snaps to 1). On completion `prev_clip_ = -1`. While blending, `prev_time_ += dt * speed_` and wraps with `fmod` at the previous clip's duration **unconditionally** (the `loop_` flag is not consulted for the outgoing clip).
5. `compute_bone_matrices(skeleton, clips)`:
   - **Channel lookup cache**: on clip change (or joint-count change) rebuild `channel_lookup_` — a per-joint `const AnimationChannel*` table (`nullptr` where the clip has no channel; channels with out-of-range `bone_index` are skipped) — and reset that side's cursors. Same for the previous clip when blending (`cached_clip_index_` / `cached_prev_clip_index_` invalidated by `crossfade_to`).
   - **Sampling** (`sample_clip_joint`, file-static): defaults to the joint's bind-pose local TRS, then overrides each of T/R/S independently if the channel has keys for it, using the warm-cursor `interpolate_keyframes`.
   - **Blending**: if `blend_factor_ < 1` and `prev_clip_` valid, sample the previous clip at `prev_time_` and combine with `glm::mix` (T, S) / `glm::slerp` (R) at factor `blend_factor_` (linear ramp).
   - **Local matrix**: built manually — `mat4_cast(rotation)` with columns 0–2 scaled by `scale.xyz` and column 3 set to the translation (equivalent to T·R·S without three matrix multiplies).
   - **Hierarchy propagation**: single pass `for i in [0, count)` with `count = min(num_joints, MAX_BONES)`; `world[i] = world[parent] * local[i]` when `0 <= parent < count`, else `world[i] = local[i]`. **Correctness requires `parent_index < i`** (parents before children in joint order); a forward parent reference would silently read a stale/previous-frame parent transform. Skeletons from `model_loader.cpp` **guarantee** this — out-of-order `skin.joints` are topologically normalized at load; hand-built skeletons must satisfy it themselves.
   - **Skinning matrices**: `bone_matrices_[i] = world_transforms_[i] * joints[i].inverse_bind_matrix` for the same `count` prefix.
   - **Identity-tail maintenance**: only slots in `[count, used_joint_count_)` are rewritten to identity (a shrinking rebind exposes previously-used slots); `reset()` identity-fills once. The old full-tail rewrite was ~8 KB of dead stores per instance per frame (comment at `animation_player.cpp:211`). `used_joint_count_ = count` afterwards.
   - Joints at index ≥ `MAX_BONES` get local transforms computed (the sampling loop runs to `num_joints`) but **no world/bone matrices — silently ignored**.

#### `crossfade_to(clip_index, duration)`

No-op when `clip_index == current_clip_`. Otherwise: `prev_clip_/prev_time_` capture the current clip/time, `blend_factor_ = 0`, `blend_duration_ = duration`, new clip starts at `time_ = 0`, both channel-lookup caches invalidated, `prev_cursors_ = std::move(current_cursors_)`. Does **not** modify `playing_`, `loop_`, or `speed_`. There is no three-way blend: re-crossfading mid-blend discards the old previous pose (potential pop). `clip_index` is not validated here; an invalid index is handled at the next `update` (clamped to 0 / not blended).

Edge case: if a non-looping clip finishes while a blend is still in progress, `playing_` goes false and subsequent `update()`s early-return, freezing the blend at its current factor until something calls `set_playing(true)` (the state machine's `enter_state` does).

#### Profiling

`update` opens Tracy zone `"AnimPlayer::update"` (`BRACKEN_PROFILE_ZONE`, `src/bracken/core/profiler.hpp`).

### `AnimationStateMachine` (`animation_state_machine.hpp/.cpp`)

#### Data structures

```cpp
struct TransitionCondition {
    enum class Op : uint8_t { GT, LT, EQ, NE, IS_TRUE, IS_FALSE };
    std::string param_name;
    int param_index = -1;          // resolved by bind_clips
    Op op = Op::GT;
    float threshold = 0.0f;
    bool evaluate(const std::vector<ParamValue>& params) const;
};
struct StateTransition {
    std::string target_state;  int target_index = -1;      // resolved by bind_clips
    std::vector<TransitionCondition> conditions;           // AND
    std::vector<TransitionCondition> any_of;               // OR, only when non-empty
    float crossfade_duration = 0.2f;
    int priority = 0;                                      // higher evaluated first
};
struct AnimState {
    std::string name, clip_name;  int clip_index = -1;     // resolved by bind_clips
    bool loop = true;  float speed = 1.0f;
    std::vector<StateTransition> transitions;
};
```

Machine state: `std::vector<AnimState> states_` + `unordered_map<string,int> state_name_to_index_`; `std::vector<ParamValue> params_` + `unordered_map<string,int> param_name_to_index_`; `int current_state_ = -1`; `std::string default_state_`; `bool bound_ = false`.

#### `TransitionCondition::evaluate` — exact semantics

Out-of-range `param_index` → `false`. Then **strictly typed**: `IS_TRUE`/`IS_FALSE` require `ParamType::Bool`; `GT`/`LT`/`EQ`/`NE` require `ParamType::Float` (mismatched type → `false`, no coercion). `EQ` is `|f − threshold| < 0.001`; `NE` is `>= 0.001`.

#### Setup API

- `add_state(AnimState)`: sorts the state's transitions by **descending `priority`** (stable order otherwise per `std::sort` — technically unstable, but priorities are the contract), then inserts; a duplicate state name **replaces** the existing state in place (index preserved).
- `set_default_state(name)`: stores the name; resolved at bind.
- `[[nodiscard]] bool bind_clips(const std::vector<AnimationClip>& clips)`: for each state, linear-search clips by `AnimationClip::name == state.clip_name` (first match wins; O(states × clips)); resolves every transition's `target_index` via the state-name map (missing → `-1`); registers every condition's parameter via `ensure_param` (creating it as `ParamValue{0.0f}`, i.e. Float 0). Sets `bound_ = true` **unconditionally**, enters `current_state_` from `default_state_` if resolvable (note: sets the index directly — it does *not* push clip/loop/speed to any player), and returns `false` iff any state's clip was not found. Re-callable (hot-reload safe).

#### Parameter API

- `int param_index(const std::string&)` — resolve-or-create (wraps `ensure_param`).
- `int find_param_index(const std::string&) const` — lookup only, `-1` if absent.
- String overloads `set_float/set_bool` **create** the parameter (via `ensure_param`) and overwrite the value *and type*; string getters do not create.
- Index overloads: setters silently ignore out-of-range indices (they never grow `params_`); getters return `0.0f`/`false` on out-of-range *or type mismatch*.
- Indices are stable for the machine's lifetime and shared between the string and index APIs (tested in `animation_state_machine.test.cpp`).

#### `update(AnimationPlayer&)` algorithm

Tracy zone `"AnimationStateMachine::update"` (color `0x8E44AD`). No-op unless `bound_` and `current_state_` is in range.

1. `clip_ended = !state.loop && !player.is_playing()` — the "one-shot finished" signal.
2. Iterate the state's transitions in stored (priority-descending) order. A transition **passes** if `clip_ended` is true (**conditions are bypassed entirely**), otherwise if all `conditions` evaluate true AND (`any_of` empty OR at least one `any_of` member true).
3. First passing transition: `enter_state(target_index, player, crossfade_duration)` and **return** — at most one transition per `update` call (multi-hop chains take one call per hop; tested).

`enter_state(index, player, crossfade)`: bails silently if the index is out of range **or** the target's `clip_index < 0` (unbound clip) — in both bail cases `update` has already returned, so a passing transition to a broken target consumes the frame's transition and shadows lower-priority candidates. On success: sets `current_state_`, then `player.crossfade_to(clip_index, crossfade)`, `player.set_playing(true)`, `player.set_loop(state.loop)`, `player.set_speed(state.speed)`. Because `crossfade_to` no-ops on the same clip index, two states sharing one clip transition without a blend or time reset (loop/speed/playing are still applied).

`current_state()` returns the state's name, or a static empty string when unbound/invalid.

#### Error handling summary

No exceptions, no `Result` — this API predates the `Result` convention and follows "bool + silent no-op": unbound machine → no-op update; missing clip → unenterable state; missing target → dead transition; bad param index → ignored/default. The only signal is `bind_clips`' `[[nodiscard]] bool`.

### IK solvers (`ik_solver.hpp/.cpp`)

All three functions share the span contract from `ik_solver.hpp`: *"Spans must point to contiguous storage of at least `MAX_BONES` glm::mat4"* — indices are used unchecked against the span length (they are checked against `min(joints, span sizes)` only in the iteration loops). All positions/targets are in **model (skeleton) space**. All three maintain `bone_matrices[i] == world_transforms[i] * joints[i].inverse_bind_matrix` for touched joints (asserted by tests) — with one caveat in `apply_foot_ik`, below. Each opens a Tracy zone (color `0x8E44AD`).

#### `solve_two_bone_ik(bones, world, skeleton, upper_idx, lower_idx, end_idx, target, pole_hint)`

Analytic law-of-cosines solver; **modifies spans in place**; every failure path is a silent no-op that leaves the pose untouched.

1. **Finite guard**: any non-finite element in the three chain world matrices or the target → return (prevents one poisoned matrix from propagating NaN — comment in source).
2. Read joint positions `pos_a/b/c` from world-transform translation columns. Bone lengths `< 0.001` → return.
3. **Target conditioning**: target coincident with the root (`|t−pos_a| < 0.001`) is pulled `0.001` along the current chain direction (or return if that is also degenerate). Reach clamped to `[|len_ab−len_bc|+0.01, len_ab+len_bc−0.01]` along the root→target ray.
4. Upper interior angle from the law of cosines (`cos` clamped to ±1). **Bend plane**: pole hint projected perpendicular to the root→target axis; if degenerate (`< 0.001`), falls back to the current knee offset projected the same way; if still degenerate (straight chain + collinear pole) → return.
5. New mid-joint position `new_b = pos_a + dir_at·cos(A)·len_ab + bend_dir·sin(A)·len_ab`; end effector goes exactly to the (clamped) target.
6. **Rotation application**: computes delta quaternions `rotation_between(old_dir, new_dir)` for the upper and lower segments (Rodrigues-style shortest-arc; antiparallel case uses a perpendicular axis; result defensively normalized) and pre-multiplies each joint's world 3×3 basis, preserving its translation (upper) / setting the new position (lower, end).
7. Recomputes `bone = world * inverse_bind` for the three chain joints, then translates every **direct child of the end effector** (scan of all joints with `parent_index == end_idx`, bounded by `min(joint count, span sizes)`) by the end effector's displacement — translation only, no rotation carry — and recomputes their bone matrices. Grandchildren are not visited.

Cost: O(joint_count) for the child scan; everything else O(1). Benchmarked in `ik.bench.cpp` (`BM_SolveTwoBoneIK`).

#### `apply_foot_ik(bones, world, skeleton, ik, scale, left_offset, right_offset)`

High-level terrain foot placement. `left/right_offset` are **world-space** vertical corrections (desired foot Δy); `scale` is the instance's uniform world scale used to convert to model space (`offset / scale`) — the instance transform itself never participates.

1. `!ik.valid` → hard no-op (indices would be `-1`; indexing them is UB — comment in source).
2. Both `|offset| <= IK_THRESHOLD (0.1)` → no-op. Offsets clamped to `±IK_MAX_CORRECTION (8.0)`.
3. **Pelvis drop**: `pelvis_drop = min(left, right)`; if negative, every joint (bounded by `min(joints, span sizes)`) gets `world[i][3].y += pelvis_drop/scale` and `bone[i][3].y += pelvis_drop/scale`. This is exact for **any** affine inverse-bind matrix (rotated/scaled included): lowering a joint is `world' = translate(0,drop,0) * world`, which only touches the translation column, and the same holds for `bone = world * inverse_bind`. Both offsets are then rebased (`-= pelvis_drop`).
4. **Per-leg solve** (left then right): skipped when the residual `|offset| < IK_THRESHOLD`. Target = current foot position with `y += offset/scale`. Pole = knee position pushed 0.5 units away from the hip↔foot midpoint (keeps the existing bend direction; degenerate for a perfectly straight leg — the playground's authored clips keep knees pre-bent, and the test skeleton pre-bends them for this reason). Delegates to `solve_two_bone_ik`.

Positive-only offsets (step-up) therefore never move the pelvis; equal negative offsets are absorbed entirely by the pelvis drop with no leg solve.

#### `apply_body_lean(bones, world, skeleton, spine_index, forward_lean, lateral_lean)`

No-op when both angles are within ±0.001 rad. Builds `lean_q = angleAxis(forward, +X) * angleAxis(lateral, +Z)` and the pivot transform `T(pivot) · R · T(−pivot)` where pivot = the spine joint's world position. Marks the spine subtree with a single index-order pass over `is_spine_descendant` flags (`i` is a descendant if its parent is flagged) — like the player's hierarchy pass this **relies on parents preceding children in joint order**. Every flagged joint (spine included) gets `world = pivot_xform * world` and `bone = world * inverse_bind` recomputed. Allocates one `std::vector<uint8_t>` of joint_count per call.

### GLB loading behavior (`src/bracken/model_loader.cpp`)

Skeleton (only when `gltf.skins` is non-empty; **only `skins[0]` is read**):

- `joint_node_indices = skin.joints`; joints resized to match; `node → joint` map built.
- Parent resolution: for each joint node, scan all glTF nodes' children lists; a parent that is itself a skin joint sets `parent_index`, otherwise `-1`.
- Local TRS from the node's `translation`/`rotation`/`scale` (defaults: zero / identity / one). Inverse bind matrices from the `skin.inverseBindMatrices` accessor; missing entries default to identity.
- **No `MAX_BONES` validation at load** — a >128-joint skin loads fully; the excess joints are silently ignored later by the player and the palette packer.
- `model.foot_ik.init(model.skeleton)` runs with the **default** `FootIKBoneNames`; success is logged at debug level.

Skinning attributes: requires `JOINTS_0` as `UNSIGNED_BYTE` and `WEIGHTS_0` as `float`; any other component type logs `BRACKEN_LOG_WARN` and the mesh loads **unskinned** ("rather than decode garbage" — comment at `model_loader.cpp:471`).

Animation clips (`model_loader.cpp:614`):

- Channels grouped per target joint; a channel whose `target_node` is not in the skin's joint list is **silently dropped** (`continue`, no log) — the documented "clip doesn't play" gotcha.
- Only `translation` (vec3), `rotation` (quat, glTF `xyzw` reordered to glm `wxyz`), and `scale` (vec3) paths are read; other paths (e.g. `weights`) are ignored. Sampler interpolation mode is ignored (everything is treated as linear).
- `clip.duration` = max keyframe time across all kept channels. A clip with zero kept channels is **not added** to `Model::animations`.
- Clip name = `gltf_anim.name` verbatim (the state machine's string key).

### Renderer integration and GPU contract

The animation module's output crosses into rendering through packet-allocated bone palettes (`src/bracken/scene/frame_packet.hpp` / `frame_builder.hpp`): `auto pal = fb.alloc_bone_palette(joint_count)` returns uninitialized packet memory; the host copies (or computes) the skinning matrices — e.g. the `AnimationPlayer::bone_matrices()` prefix, after any IK passes — into `pal.matrices`, then submits `fb.draw_skinned(handle, transform, pal.id, ...)` → `SkinnedDrawCommand`.

- **Lifetime contract**: none — the palette lives in packet memory and the packet outlives its own consumption by construction. The old non-owning `bone_matrices` span (and its "storage must remain valid and unmodified until `render_frame` returns" rule) is deleted; re-`update()`ing a player after submission is now harmless.
- **Palette packing** happens at build time: palettes are packed **tightly** (`joint_count` matrices each, 1..=`MAX_BONES`, not padded to `MAX_BONES`), `seal()` prefix-sums each `BonePalette::matrix_offset`, and the renderer uploads them in `palettes()` order into one storage buffer in the `FrameUploads` copy pass, shared by the main pass and all shadow cascades. Sharing one palette across many draws is done by reusing the `BonePaletteId` (replaces the old renderer-side pointer dedupe). Budget: `MAX_BONE_PALETTES = 1024` per frame, enforced in `alloc_bone_palette` (error log + invalid allocation — visible at build time, not a render-time drop).
- **GPU side**: the skinned shaders read the packed palettes from a per-frame `StructuredBuffer<float4x4>` (vertex set 0, binding 0 per `docs/skinned_mesh_format.md`) indexed by a per-draw `paletteOffset` uniform; there is **no compile-time GPU bone count** to keep in sync with `MAX_BONES`. Vertex input is `gpu::SkinnedVertex` (`position f3, normal f3, texcoord f2, color f4, tangent f4, joints u8×4, weights f×4`, `src/bracken/gpu/gpu_types.hpp`).

### Threading and lifetime contracts

- `render_frame` is main-thread (`docs/architecture.md` §Threading); animation updates and palette writes are host-side frame-fill work (tier `sim-stage` — under `run()` they happen inside `on_simulate`, overlapped with the previous frame's rendering).
- `AnimationPlayer::update` on **distinct instances** is safe to run concurrently: clips/skeletons are shared read-only, all mutable state (time, blend, cursors, matrices) is per-player. Verified by `animation_parallel.test.cpp`, which ticks 256 players via `core::jobs::parallel_for` and requires bit-comparable results with the serial run. Note the job system is deliberately kept idle by default — parallel animation ticking is a host decision, not engine behavior.
- `AnimationPlayer` is move-only; a moved-from player's spans are dead. `AnimationStateMachine` has no such restriction (default-copyable), but it holds no matrix storage.
- The IK functions read and write only the spans/skeleton passed in; they hold no state.

### Performance characteristics

- Player update: O(joints) with O(1) amortized keyframe lookup per channel (warm cursors); clip switches pay one lookup-table rebuild (O(joints + channels)) and cold cursors for one frame. Mid-crossfade doubles the sampling cost (both clips sampled per joint). Identity-tail rewriting is avoided (`used_joint_count_` bookkeeping; the removed per-update tail rewrite was ~8 KB of dead stores per instance per frame).
- State machine update: O(transitions × conditions) per call over a flat `std::vector<ParamValue>` — no hashing when hosts use the index-keyed parameter API (the string API hashes per call; header directs hot paths to `param_index` + int overloads).
- Benchmarks: `animation.bench.cpp` (`BM_AnimationPlayerUpdate` at 16/64/128 joints, `BM_AnimationCrossfade` likewise, warm/cold `interpolate_keyframes`, state-machine update) and `ik.bench.cpp` (`BM_SolveTwoBoneIK`, `BM_ApplyFootIK`, `BM_ApplyBodyLean`); run via `build/src/bracken/bracken_benchmarks --benchmark_filter=<regex>`.

### Known limitations, stubs, and silent behaviors

- **Joints beyond `MAX_BONES` (128) silently don't animate** — no warning at GLB load, in the player, or in the palette packer (which independently clamps).
- **glTF channels targeting non-skin nodes are silently dropped** at load; clips left with no channels vanish entirely.
- **Sampler interpolation**: `LINEAR` and `STEP` are honored per channel path; `CUBICSPLINE` is degraded to linear at load (middle output element per key) with a one-per-model warning.
- **Only `gltf.skins[0]`** is loaded; multi-skin files lose the rest.
- Hierarchy propagation and spine-descendant marking **require parents to precede children in joint order** — documented in `ik_solver.hpp` and guaranteed for skeletons from `ModelLoader::load_glb` (topological normalization at load); hand-built skeletons that violate it silently produce stale-parent transforms.
- `apply_foot_ik`'s `IK_THRESHOLD`/`IK_MAX_CORRECTION` are absolute world-unit constants, not configurable.
- `ProceduralConfig` is engine-unused (host convenience only).
- Non-looping-state auto-exit bypasses all transition conditions (by design, but easy to trip over); a passing transition to an unresolvable target consumes the frame's single transition slot and does nothing.
- `crossfade_to` on the current clip index is a no-op (no restart primitive; hosts `seek(0)`).
- No three-way blending: re-crossfading mid-blend discards the in-progress pose.
- `FootIKBoneNames` header comment says defaults "match Mixamo conventions" while the format doc requires overrides for raw `mixamorig:`-prefixed Mixamo rigs — the doc is the accurate one.

---

## Terrain & Grass Rendering

This chapter specifies the terrain subsystem (`bracken::render::TerrainRenderer`, `src/bracken/render/terrain_renderer.{hpp,cpp}`), the grass subsystem (`bracken::render::GrassRenderer`, `src/bracken/render/grass_renderer.{hpp,cpp}`), their shared heightmap data structure (`bracken::Heightmap`, `src/bracken/heightmap.hpp`), their HLSL shaders (`src/bracken/shaders/terrain.{vert,frag}.hlsl`, `shadow_terrain.vert.hlsl`, `grass.{vert,frag}.hlsl`), and their pipeline/binding contracts (`src/bracken/gpu/pipeline_catalog.cpp`).

Both renderers are owned by `scene::SceneRenderer` (`src/bracken/scene/scene_renderer.hpp`: members `terrain_`, `grass_renderer_`; accessors `terrain()` → `render::TerrainRenderer&`, `grass()` → `render::GrassRenderer*`). Both are non-copyable and non-movable; all methods are single-threaded (render-thread) with no internal synchronization.

### Heightmap data structure — `bracken::Heightmap`

The shared heightmap aggregate (`src/bracken/heightmap.{hpp,cpp}`, unit-tested in `src/bracken/heightmap.test.cpp`) is specified in full — fields, decode formula, `get_height_local`/`get_height_world`/`get_normal_world` semantics, and edge behaviors — in [Model Loading, Heightmaps & the glTF Contract](#model-loading-heightmaps--the-gltf-contract) §"Heightmap specification". Terrain-relevant summary: `uint16_t` samples, row-major `z*resolution + x`, decoded as `raw/65535 * (max_height - min_height) + min_height`; world-space queries clamp and bilinearly interpolate; normals are central differences over one grid cell with a `(0,1,0)` degenerate fallback. Precision: 16 bits over the default 1000-unit range ≈ 0.015 units per step (round-trip test tolerance 0.02).

### TerrainRenderer

#### Public API surface

```cpp
class TerrainRenderer {                                   // bracken::render
    [[nodiscard]] bool init(gpu::GPUDevice&, gpu::PipelineRegistry&,
                            float world_width, float world_height);
    void set_heightmap(const bracken::Heightmap&);        // deep copy + GPU upload + mesh regen
    void update_splatmap(const uint8_t* data, uint32_t resolution); // RGBA8, res^2*4 bytes
    void shutdown();
    void render(SDL_GPURenderPass*, SDL_GPUCommandBuffer*, const glm::mat4& view,
                const glm::mat4& projection, const glm::vec3& camera_pos,
                const glm::vec3& light_dir,
                const SDL_GPUTextureSamplerBinding* shadow_bindings = nullptr,
                int shadow_binding_count = 0, const scene::Frustum* frustum = nullptr);
    void render_shadow(SDL_GPURenderPass*, SDL_GPUCommandBuffer*,
                       const glm::mat4& light_view_projection,
                       const scene::Frustum* frustum = nullptr);
    void set_cluster_lighting(SDL_GPUBuffer* light_data, SDL_GPUBuffer* cluster_offsets,
                              SDL_GPUBuffer* light_indices, const void* params, size_t params_size);
    float get_height(float x, float z) const;             // CPU heightmap sample; 0 if none
    glm::vec3 get_normal(float x, float z) const;         // (0,1,0) if none
    float world_width() const;  float world_height() const;
    bool has_heightmap() const;
    gpu::GPUTexture* heightmap_texture();                 // R16, may be null
    gpu::GPUTexture* material_array_texture();            // RGBA8_SRGB array(4), mipmapped
    gpu::GPUTexture* splatmap_texture();                  // RGBA8, no mips
    void set_environment(const EnvironmentState& env);    // copied by value, per frame
    void set_lighting_tunables(float ambient_strength, float sun_intensity); // defaults 0.5 / 1.0
    void set_anisotropic_filter(float level);             // clamps [1,16], recreates material sampler
};
```

#### Vertex format and CPU↔GPU uniform contracts

`TerrainVertex` — 48 bytes, tightly packed, pinned by `static_assert(sizeof == 12*sizeof(float))`:

| Field | Type | Offset | Pipeline attribute |
|---|---|---|---|
| `position` | `vec3` | 0 | loc 0, FLOAT3 |
| `texCoord` | `vec2` | 12 | loc 1, FLOAT2 (declared in the pipeline layout but **unused** by `terrain.vert.hlsl` — sampling is world-space in the fragment shader) |
| `color` | `vec4` | 20 | loc 2, FLOAT4 |
| `normal` | `vec3` | 36 | loc 3, FLOAT3 |

`TerrainTransformUniforms` (vertex uniform slot 0, 144 B, `alignas(16)`): `mat4 view` (0), `mat4 projection` (64), `vec3 cameraPos` (128), `float _padding0`. Pinned against `terrain.vert.hlsl` `TransformUniforms` by static_asserts.

`TerrainLightingUniforms` (fragment uniform slot 0, 208 B, `alignas(16)`, pinned by static_asserts) — each vec3+float pair fills one 16-byte register: `fogColorHorizon/fogStart` (0/12), `fogColorSun/fogEnd` (16/28), `lightDir/fogHeightFalloff` (32/44), `cameraPos/screenAOTexelX` (48/60), `ambientSky/ambientStrength` (64/76), `ambientGround/sunIntensity` (80/92), `sunColor/screenAOTexelY` (96/108), `splat_params` (`glm::vec4`, offset 112) — `xy` = splat region origin (world XZ), `zw` = region X/Z size, so non-square/offset worlds map the splatmap correctly — `iblEnabled` (offset 128; 1 = sky-IBL ambient, samplers 5–7), `rtShadowEnabled` (132; traced mask at sampler 8 replaces CSM), 8-byte pad, then the probe-field GI block R1c added: `giCascadeBase[3]` (`glm::ivec4`, each cascade's window-corner probe coordinate) and `giEnabled` + three named pads (144/192).

`gpu::ShadowTerrainUniforms` (`src/bracken/gpu/gpu_uniforms.hpp`, vertex slot 0 of the shadow pass): a single `alignas(16) glm::mat4 lightViewProjection`.

#### Main mesh generation (`generate_terrain_mesh`)

Runs at `init` (flat placeholder, no heightmap) and again on every `set_heightmap`. Constants (all local to the .cpp):

| Constant | Value | Meaning |
|---|---|---|
| `margin` | 500.0 | ring of terrain generated outside `[0, world]` on all sides |
| `cell_size` | 25.0 | grid cell in world units |
| `TILE_CELLS` | 32 | cells per tile side → 800-unit tiles |
| `tex_scale` | 0.01 | written into `texCoord` (unused by the shader) |

Algorithm:

1. Scratch memory is a single `memory::Arena` sized for max vertices/indices/heights (one-shot; buffers are built, uploaded, discarded).
2. Pass 1: one bilinear `get_height` per grid point into a `heights` array.
3. Pass 2: vertices. Normal = central difference over neighboring cached grid heights (clamped at edges) — 1 bilinear sample per vertex instead of ~5. Vertex color is a subtle tint: `dist_factor = min(dist_from_world_center/3000, 1)`, `height_factor = clamp(y/100, 0, 1)`, `r = 0.95 + 0.05*dist_factor`, `g = 1.0 − 0.05*dist_factor − 0.05*height_factor`, `b = 0.9 + 0.05*dist_factor`, `a = 1`.
4. Indices are emitted **per tile** (row-major tiles, two CCW triangles per cell, 32-bit indices); each `TerrainTile{first_index, index_count, aabb_min, aabb_max}` records its contiguous index range and a world AABB with true min/max Y. Empty tiles are dropped.
5. `build_tile_rows` groups tiles sharing a min-Z band into `TileRow{first_tile, tile_count, merged AABB}` (tiles are contiguous per row by construction).
6. Vertex/index data upload via `gpu::GPUBuffer::create_static`; on index-buffer failure the vertex buffer is also reset (logged, non-fatal).

For a 42000-unit world this yields ~52×52 ≈ 2700 tiles (comment in source); the playground's 8000-unit world with margins is 9000/800 → 12×12 tiles.

#### Shadow (low-LOD) mesh (`generate_shadow_terrain_mesh`)

Separate coarse mesh used only by `render_shadow`. Constants:

| Constant | Value | Meaning |
|---|---|---|
| `kShadowCellSize` | 100.0 | 4× coarser than main → ~16× fewer triangles |
| `kShadowMargin` | 500.0 | same margin ring |
| `TILE_CELLS` | 16 | 1600-unit shadow tiles |
| `PROBE_STEP` | 25.0 | fine probe grid spacing |
| `PROBE_RADIUS` | 2 | ±2 probes → 5×5 min window per vertex |
| `SHADOW_SINK` | 5.0 | vertices lowered by 5 units |

Each shadow vertex takes the **minimum** height over its 5×5 probe window (probes pre-sampled once — one bilinear each — then plain array reads), then sinks by `SHADOW_SINK`. This keeps the simplified caster surface at-or-below the true terrain everywhere, preventing phantom shadow blobs over sub-cell depressions. Only `position` is filled; the other vertex attributes are zero. Tiles/rows are built the same way as the main mesh. A summary line is logged via `SDL_Log`.

#### Tile culling and draw-call merging (`cull_tile_runs` + `draw_culled_tiles`)

Shared by main and shadow passes. The culling core is the pure static `cull_tile_runs(JobSystem& js, const Frustum&, tiles, rows, out_runs)` (GPU-free, unit-tested); the member `draw_culled_tiles(JobSystem& js, pass, frustum, …)` runs it into a reused `tile_runs_scratch_` member and issues one draw per run. Call sites (forward pass, depth prepass, every CSM cascade's `render_shadow`) pass `JobSystem::instance()`:

1. For each `TileRow`, one `frustum.intersects_aabb(row.aabb_min, row.aabb_max)` test skips the row's whole tile span (typical case: 80–95% of the world culled).
2. Surviving rows test per-tile AABBs.
3. Visible tiles whose `first_index` continues the current run (`first_index == run_first + run_count`) merge into `TileRun{run_first, run_count}`; each run is drawn with `SDL_DrawGPUIndexedPrimitives(pass, run_count, 1, run_first, 0, 0)`. Comment: typically collapses 90–260 per-tile draws into ~10–30 row runs.
4. If `frustum == nullptr` or there are no tiles, the caller draws the entire mesh in one call; with empty `rows` (unbuilt row acceleration), `cull_tile_runs` falls back to a plain serial per-tile walk.
5. Phase-3 fan-out: the row-level tests chunk over `js` at `CULL_ROW_GRAIN = 32` rows (a 42000-unit world ≈ 52 rows → 2 chunks; playground scale → 1 inline chunk) into per-chunk partial run lists, merged single-threaded in chunk-index order with seam runs coalesced — bit-identical to the serial contiguous-run walk at any worker count. `build_tile_rows` (init-time) fans out the same way at `ROW_BUILD_TILE_GRAIN = 512` tiles, coalescing z-band rows that span a chunk seam (tile-count sum + AABB union).

#### Textures, samplers, formats

| Resource | Format | Source | Fallback |
|---|---|---|---|
| heightmap | `TextureFormat::R16` (16-bit unorm), no mips | `set_heightmap` upload of `height_data` | none (terrain stays flat) |
| material array | `RGBA8_SRGB`, 4 layers (grass, dirt, rock, sand), mipmapped | `<exe dir>/assets/textures/{grass,dirt,rock,sand}_seamless.png` (anchored via `SDL_GetBasePath`, same policy as shaders) via `SDL_image`, converted `SDL_PIXELFORMAT_RGBA32`; all four must match dimensions | procedural 64×64 per-layer noise-shaded colors (base RGB {92,138,70}, {126,96,66}, {126,124,120}, {196,178,128}), also sRGB+mips |
| splatmap | `RGBA8` (linear), no mips | `<exe dir>/assets/textures/terrain_splatmap.png` | 4×4 all-grass (R=255) |
| material sampler | `SamplerConfig::anisotropic(8.0f)` at init — linear+mips, REPEAT, aniso on | recreated by `set_anisotropic_filter(level)` (clamped 1–16; the failure log message says "grass sampler", a misnomer) |
| splatmap sampler | `SamplerConfig::linear_clamp()` | dedicated CLAMP sampler so the splatmap does not wrap across the 500-unit margin ring |

`update_splatmap(data, resolution)` recreates the splatmap texture (`RGBA8`, no mips); silently returns on null device/data; logs on creation failure. Note: nothing in the engine re-points `GrassRenderer::splatmap_texture_` afterwards — see grass limitations.

#### Main-pass binding contract (`render`)

Pipeline: `PipelineType::Terrain` — vertex shader `terrain.vert.spv`, fragment `terrain.frag.spv`; declared resources VS: 1 UBO; FS: 3 UBOs, 10 samplers, 4 storage buffers. Default cull `SDL_GPU_CULLMODE_BACK`, depth write on, `GREATER_OR_EQUAL` reversed-Z, color `RGBA16F` scene format, depth `D32_FLOAT`. (Row in `gpu/pipeline_catalog.cpp`; that table is the source of truth.)

| Slot | Stage | Content | Pushed/bound by |
|---|---|---|---|
| vertex uniform 0 | VS | `TerrainTransformUniforms` | `render()` |
| fragment uniform 0 | FS | `TerrainLightingUniforms` (208 B) | `render()` |
| fragment uniform 1 | FS | `ShadowDataUniforms` block (`shadow_common.hlsli` cbuffer) | **caller** — `SceneRenderer::render_3d_scene` pushes `frame_shadow_uniforms_` before calling `render()` |
| fragment uniform 2 | FS | cluster params (`cluster_lighting.hlsli`) | `render()`, only when all four `set_cluster_lighting` pointers are non-null |
| fragment sampler 0 | FS | material `Texture2DArray` + anisotropic sampler | `render()` |
| fragment sampler 1 | FS | splatmap + CLAMP sampler | `render()` |
| fragment samplers 2–3 | FS | shadow atlas ×2 (comparison sampler, then raw sampler for PCSS blocker search) | `render()` from caller-provided `shadow_bindings` |
| fragment storage 0–2 | FS | cluster light data / cluster offsets / light indices | `render()` (conditional, as above) |

Skip conditions (all silent-with-log or fully silent): missing pipeline registry/buffers/pass/cmd (silent return); textures or sampler not ready (`SDL_Log`, return); pipeline lookup failure (`SDL_Log`, return).

HLSL binding spaces (SDL3 GPU SPIR-V convention): FS combined image samplers in set 2 bindings 0–12 (material array, splatmap, shadow atlas ×2, screen AO, IBL irradiance/prefiltered/BRDF LUT, RT sun mask, GI irradiance/visibility/probe metadata, local-shadow atlas); FS cbuffers in set 3 (`LightingUniforms` b0, `ShadowUniforms` b1); the four cluster SSBOs follow the 13 samplers in set 2 (`CLUSTER_SSBO_BINDING_BASE 13`). VS cbuffer set 1 binding 0.

#### Fragment shading algorithm (`terrain.frag.hlsl`)

Constants: `TERRAIN_METALLIC = 0.0`, `TERRAIN_AO = 1.0`, `TEX_SCALE = 0.01` (base tiling: 100 world units), `DETAIL_SCALE_RATIO = 0.23` (second anti-tiling scale), `SPLAT_EPSILON = 0.005` (layer skip threshold), `FLAT_NORMAL_Y = 0.8` (single-plane fast-path threshold).

1. Splat weights: `splatmapTexture.Sample(splatmapUV = worldPos.xz / worldSize)` — RGBA maps to layers 0–3 (grass, dirt, rock, sand).
2. Anti-tiling: each layer samples the material array at two scales (`TEX_SCALE` and `TEX_SCALE * 0.23`) blended by a low-frequency murmur-hash value noise of `worldPos.xz * 0.004`. The murmur (integer) hash is deliberate: a sin-based hash degrades ~10⁴ units from the origin (fp32 argument reduction).
3. Projection: triplanar (`abs(N)^4` normalized blend weights) for steep surfaces; single XZ-plane fast path when `N.y > 0.8`. All samples use explicit gradients (`SampleGrad`) so divergent layer-skip branches keep well-defined derivatives.
4. Layers with splat weight < 0.005 are skipped; the accumulated color is normalized by `max(weightSum, 1e-3)` and multiplied by the vertex color tint.
5. Lighting: roughness `0.9 + (noise−0.5)*0.1`; `calcShadow` (CSM, from `shadow_common.hlsli` using `viewDepth`), `evaluatePBR` direct + ambient with `ambientLift = lerp(ambientStrength, 1, shadow)` (`ambientIBL` from the sky prefilter chain when `iblEnabled`, hemisphere `ambientPBR` otherwise); clustered lights added when `clusterParams.gridDim.w > 0`; `applyAtmosphericFog` last. Output is linear HDR (`RGBA16F` target); alpha 1.

#### Shadow-pass contract (`render_shadow`)

Pipeline `PipelineType::ShadowTerrain`: VS `shadow_terrain.vert.spv` (1 UBO), FS `shadow_depth_opaque.frag.spv` (no resources), terrain vertex layout, opaque, `SDL_GPU_CULLMODE_NONE`, depth bias (1.25, 1.75), `depth_clip = false` (pancaking), `depth_compare = LESS_OR_EQUAL` (**shadow cascades stay standard-Z**, unlike the reversed-Z scene), color format `SDL_GPU_TEXTUREFORMAT_INVALID` (depth-only), depth `D32_FLOAT`. `no_cull` is required because terrain is single-sided.

`render_shadow` prefers the low-LOD mesh and falls back to the main mesh if the shadow buffers are absent; pushes `ShadowTerrainUniforms` at vertex uniform 0; per-cascade frustum culls the shadow tiles with the same row/run merging. Called once per cascade by `SceneRenderer::render_shadow_passes`.

#### Performance characteristics

- Mesh regen cost: one bilinear heightmap sample per vertex (pass-1 caching) — the source notes this dropped from ~5 samples. Arena allocation avoids two heap-grown vectors.
- Per-frame CPU: row-first AABB culling makes tile culling O(rows + tiles-in-intersecting-rows) rather than O(all tiles).
- Comment-documented expectation: 5–15% of tiles in frustum → 10–20× less geometry drawn; shadow mesh 16× fewer triangles than main.

### GrassRenderer

#### Architecture

GPU-instanced, chunked, fully procedural (header comment, `src/bracken/render/grass_renderer.hpp`): the world is partitioned into fixed 32-unit chunks; the CPU frustum-culls chunks each frame and uploads visible chunk origins into a storage buffer; one instanced draw renders `visible_chunks × 144` blade instances; the vertex shader derives per-blade transform from `instanceID` and world-space hashes, sampling terrain height from the heightmap texture. Smooth LOD fade replaces discrete density bands.

Compile-time constants (private): `CHUNK_SIZE = 32.0f` world units; `BLADES_PER_CHUNK_SIDE = 12`; `BLADES_PER_CHUNK_SQ = 144`. Public: `static constexpr float MAX_VIEW_DISTANCE = 1200.0f`.

Public mutable fields (not methods): `wind_magnitude = 0.8f`, `wind_direction = vec2(1.0f, 0.3f)` (world-XZ; normalized at upload via `effective_wind_direction()`, zero-safe), `grass_view_distance = MAX_VIEW_DISTANCE` (1200), `edge_margin = 0.0f` (grass-free ring inside the world edges), `ambient_strength = 0.5f`, `sun_intensity = 1.0f`. `grass_view_distance` is **respected**: the per-frame effective distance is `effective_view_distance() = min(grass_view_distance, MAX_VIEW_DISTANCE, view_distance_cap_)` where the cap (set via `set_view_distance_cap`; ≤ 0 disables) is written each frame by `SceneRenderer` from `gfx.get_draw_distance()`. `ambient_strength`/`sun_intensity` are overwritten on every `apply_graphics_settings`.

#### Public API surface

```cpp
class GrassRenderer {                                     // bracken::render
    [[nodiscard]] bool init(gpu::GPUDevice&, gpu::PipelineRegistry&,
                            float world_width, float world_height);
    void update(float current_time);   // absolute-time wind clock; no dt param —
                                       // rewinding/holding current_time rewinds/pauses the wind
    void upload_chunks(SDL_GPUCopyPass*, const glm::vec3& camera_pos,
                       const scene::Frustum* frustum);    // must precede render(), copy pass open
    void render(SDL_GPURenderPass*, SDL_GPUCommandBuffer*, const glm::mat4& view,
                const glm::mat4& projection, const glm::vec3& camera_pos,
                const glm::vec3& light_dir,
                const SDL_GPUTextureSamplerBinding* shadow_bindings = nullptr,
                int shadow_binding_count = 0);
    void shutdown();
    void set_heightmap(gpu::GPUTexture* texture, const HeightmapParams& params); // not owned
    void set_splatmap(std::function<gpu::GPUTexture*()> provider);  // provider fetched each frame
                                                                    // at draw time — can't dangle
    void set_exclusion_rects(std::vector<GrassExclusionRect> rects); // no-grass zones; default none
    std::span<const GrassExclusionRect> exclusion_rects() const;
    void set_view_distance_cap(float cap);                // <= 0 disables the cap
    float effective_view_distance() const;
    glm::vec2 effective_wind_direction() const;           // normalized, zero-safe
    void set_environment(const EnvironmentState& env);
    void set_cluster_lighting(SDL_GPUBuffer*, SDL_GPUBuffer*, SDL_GPUBuffer*,
                              const void* params, size_t params_size);
};
```

`GrassExclusionRect` (public struct): `min_x/min_z/max_x/max_z` world-XZ bounds. Applied at chunk granularity — chunks whose *center* lies inside any rect are culled.

`HeightmapParams` (public struct): `world_origin_x/z = 0`, `world_size = 1`, `min_height = -500`, `max_height = 500` — populated by `SceneRenderer::set_heightmap` from the `Heightmap` fields.

#### Initialization

- Heightmap sampler: `SamplerConfig::linear_clamp()`.
- Blade mesh (`generate_blade_mesh`): unit blade, base at Y=0, tip Y=1, 4 segments = 10 `gpu::Vertex3D` vertices / 8 triangles / 24 32-bit indices; half-width tapers `0.5 * (1 − 0.9t)`; per-vertex color gradient linear from base `(0.05, 0.13, 0.03, 1)` to tip `(0.18, 0.44, 0.10, 1)` (authored linear, tuned to the terrain grass layer); normal `(0,0,1)`; static vertex/index buffers.
- Chunk storage buffer: dynamic `Storage` buffer pre-sized to 8192 × `sizeof(glm::vec4)` = 128 KB (comment: at view 1200/chunk 32, ~4400 candidates, usually 1000–2500 visible).
- Idempotent: a second `init` returns `true` immediately (`initialized_` flag).

#### Per-frame chunk culling and upload (`upload_chunks`)

Runs inside the frame's consolidated `FrameUploads` copy pass, before any render pass:

1. Snap camera to the chunk grid; iterate a square of `half_extent = ceil(effective_view_distance() / 32) + 1` chunks in each direction (the static `cull_chunks(JobSystem& js, …)` helper, unit-tested CPU-only; `upload_chunks` passes `JobSystem::instance()`). Phase-3 fan-out: the grid walk chunks over grid rows at `CULL_ROW_GRAIN = 48` rows (max-view-distance grid = 79 rows → 2 chunks; ≤ 770-unit view distance → 1 inline chunk); each chunk appends to its own cache-line-aligned partial vector, merged single-threaded in chunk-index order — preserving the exact serial row-major append order the FNV hash-skip in step 7 depends on.
2. Reject by 2D center distance > the effective view distance.
3. World-bounds cull: reject chunks with no overlap of the grassable rect `[edge_margin, world − edge_margin]` on both axes (with the default margin 0 this only rejects chunks fully outside the world).
4. Reject chunks whose center lies inside any host-configured `GrassExclusionRect`.
5. Optional frustum AABB test with `y ∈ [params.min_height, params.max_height + 20]` (+20 = blade height upper bound).
6. Survivors are appended to `visible_chunks_scratch_` as `vec4(ox, 0, oz, 0)`.
7. **Upload elision**: FNV-1a (offset basis `1469598103934665603`, prime `1099511628211`) over each chunk's `x`/`z` bit patterns plus the count; if it equals `last_chunk_set_hash_`, the SSBO already holds this exact set and the upload is skipped (camera panning within a chunk is the common case).
8. Growth: if the visible set exceeds capacity, a new dynamic buffer of `needed_bytes * 2` is created (legal inside an open copy pass); on failure logs and resets the hash. Otherwise `GPUBuffer::update(copy_pass, data, bytes)`.

#### Draw (`render`)

Early-outs (all silent): not initialized / null pass/cmd; blade buffers missing; **heightmap texture or sampler unset** (grass requires `set_heightmap` first); empty visible-chunk set. Pipeline lookup failure logs.

Pipeline `PipelineType::Grass`: VS `grass.vert.spv` (1 UBO, 2 samplers, 1 storage buffer), FS `grass.frag.spv` (3 UBOs, 8 samplers, 4 storage buffers); `Vertex3D` layout, opaque, `SDL_GPU_CULLMODE_NONE` — grass blades are double-sided; scene `RGBA16F` + `D32_FLOAT` targets. Vertex layout is the shared `gpu::Vertex3D` (position, normal, texcoord, color, tangent) even though the shader consumes only position/normal/texCoord/color.

Binding contract:

| Slot | Stage | Content | Set by |
|---|---|---|---|
| vertex uniform 0 | VS | `GrassVertexUniforms` (304 B) | `render()` |
| vertex samplers 0–1 | VS | heightmap; splatmap (**slot 1 is always bound** — the heightmap texture is substituted when no splatmap is set and the shader ignores it via `splatEnabled == 0`) | `render()` |
| vertex storage 0 | VS | `StructuredBuffer<float4> ChunkOrigins` | `render()` |
| fragment uniform 0 | FS | `GrassLightingUniforms` (192 B) | `render()` |
| fragment uniform 1 | FS | shadow data block | **caller** (`SceneRenderer::render_3d_scene`) |
| fragment uniform 2 | FS | cluster params | `render()` (only when all cluster pointers non-null) |
| fragment samplers 0–1 | FS | shadow atlas (cmp + raw) | `render()` from caller bindings |
| fragment storage 0–2 | FS | cluster light data / offsets / indices | `render()` (conditional) |

Draw call: `SDL_DrawGPUIndexedPrimitives(pass, 24, visible_chunks * 144, 0, 0, 0)` — one instanced draw for the whole meadow.

HLSL binding spaces: VS set 0 = 2 combined image samplers (bindings 0–1) then the storage buffer at binding 2; VS cbuffer set 1 b0. FS samplers set 2 bindings 0–10 (shadow atlas ×2, screen AO, IBL irradiance/prefiltered/BRDF LUT, RT sun mask, GI irradiance/visibility/probe metadata, local-shadow atlas); FS cbuffers set 3 (b0 lighting, b1 shadow); the four cluster SSBOs at `CLUSTER_SSBO_BINDING_BASE 11` in set 2.

#### `GrassVertexUniforms` layout (304 B, pinned by static_asserts; the trailing `curr_vp_no_jitter`/`prev_vp_no_jitter` mat4 pair at offsets 176/240 feeds the depth-prepass velocity output and is ignored by the forward shader)

| Field | Offset | Notes |
|---|---|---|
| `view_projection` | 0 | `projection * view` |
| `camera_pos` / `time` | 64 / 76 | `time` = scene time from `update()` |
| `splat_enabled` | 84 | float 1/0 (1 when the splatmap provider returned a texture) |
| `grass_view_distance` | 92 | `effective_view_distance()` |
| `wind_direction` | 96 | `effective_wind_direction()` — host-configurable, normalized, zero-safe |
| `wind_strength` | 104 | = `wind_magnitude` |
| `chunk_size` / `blade_spacing` | 108 / 112 | spacing = 32/12 ≈ 2.667 |
| `blades_per_chunk_side` / `_sq` | 116 / 120 | 12 / 144 |
| `heightmap_origin_x/z`, `world_size`, `min/max_height` | 128–147 | from `HeightmapParams` |
| `camera_forward` | 148 | `-vec3(view[0][2], view[1][2], view[2][2])`; packs into the last 12 bytes of the register at 144 — the header warns that inserting any scalar before it silently desynchronizes CPU/GPU (`splat_params` then starts a fresh register at 160 on both sides) |
| `splat_params` | 160 | `glm::vec4`: xy = splat region origin (world XZ), zw = region X/Z size — non-square worlds handled |

`GrassLightingUniforms` (192 B) mirrors the terrain lighting block with grass field ordering: `camera_pos/fog_start`, `fog_color_horizon/fog_end`, `fog_color_sun/fog_height_falloff`, `light_dir/ambient_strength`, `sun_color/sun_intensity`, `ambient_sky/screen_ao_texel_x`, `ambient_ground/screen_ao_texel_y`, then `ibl_enabled` (offset 112), `rt_shadow_enabled` (116; traced mask at sampler 6 replaces CSM), 8-byte pad, then the probe-field GI block R1c added: `giCascadeBase[3]` (`glm::ivec4`, each cascade's window-corner probe coordinate) and `giEnabled` + three named pads (128/176) — all pinned by static_asserts.

#### Vertex shader algorithm (`grass.vert.hlsl`)

Per instance: `chunk_idx = instanceID / 144`, `blade_idx = instanceID % 144`; blade cell = chunk origin + `(bx, bz) * bladeSpacing + 0.5*bladeSpacing`; per-blade seed = `int2(floor(cellPos / bladeSpacing))` (world-anchored murmur hash — stable regardless of camera or chunk set). Position jitter = `(hash2 − 0.5) * bladeSpacing`.

Order of evaluation is optimized to skip texture reads for dead blades:

1. **Smooth LOD fade**: `fade_start = 0.70 * grassViewDistance`; `lod_scale = 1 − saturate((dist − fade_start)/(0.30 * viewDist))` — blades shrink over the last 30%, no pop-in.
2. **Clump attenuation**: hash of `floor(cellPos / (bladeSpacing * 6))` scales `lod_scale` by `0.6 + 0.4*clump` (continuous density variation).
3. **Early collapse**: `lod_scale ≤ 0.001` → `deadBlade()` — a degenerate clipped vertex (`z/w = 2`) emitted **before any texture fetch**.
4. **Splat mask** (if `splatEnabled > 0.5`): `grassWeight = splatmap.R` at `saturate(worldXZ / splatWorldSize)`; `lod_scale *= smoothstep(0.15, 0.5, grassWeight)`; collapse again if ≤ 0.001.
5. **Height**: single `SampleLevel` of the heightmap; decode `raw * (max−min) + min` at clamped UV `((world − origin)/world_size)`.
6. **Blade shape**: height `(2.5 + h2*4 + h3²*4) * lod_scale` (≈2.5–10.5 units), width `(0.6 + hash*0.5) * 1.3` (widened ~1.3× so hash-yawed blades read edge-on); stable per-blade yaw over full 2π (no camera-facing billboarding — the meadow does not re-orient/re-light as the camera orbits); quadratic curve toward a hashed direction, magnitude `0.15 * bladeHeight` (length-guarded normalize to avoid NaN when the hash lands on exactly (0.5, 0.5)).
7. **Layered wind**: `windWave = sin(time*2.0 + 0.10x + 0.10z)*0.7 + sin(time*3.7 + 0.07x − 0.13z)*0.3`; XZ offset = `windDirection * windStrength * heightT * windWave` (tip moves, base pinned).
8. **Vertex color AO**: `rgb *= lerp(0.45, 1.0, heightT)`. Shading normal = `normalize(lerp(bladeFwd, up, 0.7))` — leans toward up so lighting reads as a ground surface, not flipping facets. `viewDepth = dot(worldPos − cameraPos, cameraForward)` (view-space depth, matching other mesh shaders for cluster Z-slice / cascade selection).

#### Fragment shading (`grass.frag.hlsl`)

`GRASS_METALLIC = 0.0`, `GRASS_ROUGHNESS = 0.9`. Direct light = `max(evaluatePBR(...), half-Lambert-squared wrap diffuse)` (wrap term rescues grazing angles that PBR's NdotL clamp kills); plus a tip specular `pow(NdotH, 32) * 0.3 * texCoord.y²`; ambient with `ambientLift = lerp(ambientStrength, 1, shadow)` (`ambientIBL` when `ibl_enabled`, hemisphere `ambientPBR` otherwise); clustered lights when `gridDim.w > 0`; `applyAtmosphericFog` (the VS LOD collapse handles the view-distance boundary, so no distance fade in the FS). CSM shadow via `calcShadow` — grass **receives** shadows.

#### Lifetime and threading contracts

- `set_heightmap`'s pointer is **non-owning**; the terrain renderer recreates its heightmap texture on every `TerrainRenderer::set_heightmap`, so holders must be re-pointed (`SceneRenderer::set_heightmap` re-wires it automatically). The splatmap, by contrast, is wired as a **provider callable** fetched each frame at draw time — `update_splatmap` recreating the texture cannot dangle it.
- `set_cluster_lighting`'s `params` pointer is dereferenced at `render()` time; `SceneRenderer` passes `&frame_cluster_params_`, a member with frame lifetime.
- `upload_chunks` must run in a copy pass recorded before the render pass of the same frame; `render` consumes `visible_chunks_scratch_` (size drives the instance count) and the SSBO contents from that upload.
- All calls render-thread only; destructors call `shutdown()`.

#### Known limitations, stubs, and silent behaviors

- **Grass casts no shadows** — there is no grass path in the shadow passes.
- **`set_splatmap` is never called by the engine** (`SceneRenderer` doesn't wire it; the playground doesn't either), so splat-aware placement is dormant unless the host opts in.
- Grass renders only if a heightmap texture was set; otherwise `render()` silently no-ops.
- Splat sampling covers the region described by `splat_params` (origin + X/Z size); the engine currently populates it as origin (0,0) with the full world width/height.

### SceneRenderer integration (frame order)

Per `docs/architecture.md` and `src/bracken/scene/scene_renderer.cpp`:

1. **CPU prep** — `set_environment` propagated to both renderers; `grass_active = frame.should_draw_grass() && gfx.grass_enabled && grass_renderer_`; `grass_renderer_->set_view_distance_cap(gfx.get_draw_distance())` (the renderer's own `grass_view_distance` is respected, not overwritten); `grass_renderer_->update(scene_time_)`.
2. **`FrameUploads` copy pass** — `grass_renderer_->upload_chunks(copy_pass, camera.position, &frame_frustum_)` batched with instance/bone/cluster uploads.
3. **Shadow passes** — `terrain_->render_shadow(...)` once per cascade with that cascade's frustum (comment: cascade 0 draws ~2–5 tiles). Grass is absent.
4. **Cluster wiring** — `set_cluster_lighting` on both renderers with the `ClusterGrid`'s buffers and `frame_cluster_params_`.
5. **Main 3D pass** (`render_3d_scene`) — shadow atlas bound twice (comparison + raw samplers); `SDL_PushGPUFragmentUniformData(cmd, 1, &frame_shadow_uniforms_, ...)` immediately before each of `terrain_->render(...)` (gated on `scene.should_draw_ground()`) and `grass_renderer_->render(...)` (gated on the grass flags).

Settings plumbing: `apply_graphics_settings` forwards `ambient_strength`/`sun_intensity` to both renderers; `set_anisotropic_filter(int level)` maps `level 0..4` to `1/2/4/8/16` (as `1 << level` for level > 0, clamped to 16) before calling the terrain's float overload. `SceneRenderer::set_heightmap` bridges `Heightmap` → `HeightmapParams` and shares the terrain's R16 texture with the grass renderer. Default world extent when the host omits it: 32000 × 32000 (`Application::init_renderer`, `src/bracken/application.hpp`).

---

## Lighting, Shadows & Ambient Occlusion

Engine-developer reference for the three subsystems under
`src/bracken/render/` that produce lit pixels: clustered-forward dynamic
light binning (`lighting/`), cascaded shadow maps (`shadow_map.*`), and
screen-space ambient occlusion (`gtao.*`) plus the HDR-offscreen/composite/
FXAA chain (`post_process_stack.*`). All are single-threaded, render-thread-only
objects orchestrated by `scene::SceneRenderer::render_frame`
(`src/bracken/scene/scene_renderer.cpp`); none are copyable or movable.

### Clustered-forward light binning (`render::lighting`)

#### Purpose

CPU-side binning of dynamic point/spot lights into a view-frustum cluster
grid, uploaded as three SSBOs consumed by the opaque scene fragment shaders
(`model`, `model_instanced`, `terrain`, `grass`). There is no GPU compute
involved; binning is CPU (`ClusterGrid::build`), per-fragment lookup is done
in the shader (`src/bracken/shaders/clusters.hlsli`,
`cluster_lighting.hlsli`).

#### Constants (`src/bracken/render/lighting/light.hpp`)

| Constant | Value |
|---|---|
| `CLUSTER_DIM_X` | 16 |
| `CLUSTER_DIM_Y` | 9 |
| `CLUSTER_DIM_Z` | 24 (exponential depth slices) |
| `CLUSTER_COUNT` | 16·9·24 = 3456 |
| `MAX_LIGHTS` | 256 (point + spot combined, v1 limit) |
| `MAX_LIGHTS_PER_CLUSTER` | 128 |
| `LIGHT_TYPE_POINT` / `LIGHT_TYPE_SPOT` | 0 / 1 |

These are duplicated as `#define`s in `src/bracken/shaders/clusters.hlsli`
and must stay in sync.

#### Data structures and layout

Std140-compatible PODs, sizes static_asserted in
`src/bracken/render/lighting/light_cluster.cpp`:

- `PointLight` — `{vec3 position, float radius, vec3 color, float
  intensity, bool casts_shadows, int32 shadow_slot, uint32 stable_id,
  float source_radius}` = **48 bytes** (`shadow_slot` at byte 36 — read by the
  shader; engine-assigned each frame, -1 = unshadowed; `casts_shadows` is the
  host opt-in and is never read on the GPU).
- `SpotLight` — `{vec3 position, float radius, vec3 direction, float
  inner_cos, vec3 color, float outer_cos, float intensity, bool
  casts_shadows, int32 shadow_slot, uint32 stable_id, float source_radius
  (+pad)}` = **80 bytes** (`shadow_slot` at byte 56, `source_radius` at 64).
- **Sizes were unchanged by R2a and PointLight's is still unchanged.**
  `stable_id` (byte 40 of `PointLight`, byte 60 of `SpotLight`) took over bytes
  that were declared padding, and R6's `source_radius` did the same in
  `PointLight`'s remaining `_pad1` (byte 44). `SpotLight` had nothing spare and
  therefore **grew 64 → 80 bytes**, which moved exactly two shader stride
  literals: `loadSpotLight` in `cluster_light_load.hlsli` and the flat light
  walk in `rt_hit_shade.hlsli`. The alternative — a half float packed into the
  three bytes after `casts_shadows` — keeps the stride at the price of storing
  the one field a host authors in world units at 11 bits of mantissa, and was
  rejected on that basis. Every offset named here is `static_assert`ed. See
  `contracts/light.hpp` for the identity and area-light contracts.
- `LightHeader` — `{uint32 type, uint32 payload_index}` = **8 bytes**;
  `payload_index` indexes the per-type payload array.
- `ClusterParams` (`light_cluster.hpp`, `alignas(16)`) —
  `{mat4 view, mat4 invProjection, vec4 screenSize (w,h,1/w,1/h),
  vec4 zPlanes (near, far, log(far/near), 1/log(far/near)),
  uvec4 gridDim (X,Y,Z,totalLightCount),
  uvec4 maxPerCluster (per-cluster cap, light-buffer capacity, 0, 0)}`.
  This is the **single CPU-side definition** of the clustered-lighting
  cbuffer layout — `SceneRenderer` pushes it directly, and
  `gpu_uniforms.hpp` deliberately carries no mirror; keep it in sync with
  the HLSL (`clusters.hlsli`).

GPU buffers (all `gpu::GPUBuffer::Type::Storage`, dynamic):

1. **Light data buffer** — single contiguous layout with capacity-based
   section offsets (`cap = max_lights_`, default `MAX_LIGHTS`):
   `[LightDataHeader 16B: header_count, point_count, spot_count, _pad]`
   `[LightHeader × cap]` `[PointLight × cap]` `[SpotLight × cap]` →
   16 + 256·(8+48+80) = **34 832 bytes** at default capacity. The shader
   derives the same offsets from `maxPerCluster.y`
   (`pointLightsBase()`/`spotLightsBase()` in `cluster_lighting.hlsli`).
2. **Cluster offsets buffer** — `uvec2 (offset, count)` per cluster, flat
   `2 × CLUSTER_COUNT` u32 = **27 648 bytes**.
3. **Light indices buffer** — capacity `CLUSTER_COUNT ×
   MAX_LIGHTS_PER_CLUSTER` u32 = **1 769 472 bytes**; only the used prefix is
   uploaded.
4. **Light stable-id buffer** — `uint32 × max_lights` = **1 024 bytes**; the
   persistent identity of each light in PACKED (header) order.
5. **Light index remap buffer** — `uint32 × max_lights` = **1 024 bytes**;
   previous frame's packed index → this frame's packed index, `0xFFFFFFFF`
   where the light no longer exists. Both are uploaded in the same copy pass
   as the three above; no shader reads them yet (R2b's froxel reservoirs are
   the consumer — `docs/rt_lighting_plan.md` §5.3).

#### Public API (`render::lighting::ClusterGrid`)

```cpp
[[nodiscard]] bool init(gpu::GPUDevice& device, uint32_t max_lights = MAX_LIGHTS);
void shutdown();
void begin_frame(core::jobs::JobSystem& js, const scene::CameraState& camera,
                 uint32_t screen_w, uint32_t screen_h,
                 float near_plane, float far_plane);
void begin_frame(const scene::CameraState& camera, uint32_t screen_w,
                 uint32_t screen_h, float near_plane, float far_plane);
                 // convenience: delegates to JobSystem::instance()
[[nodiscard]] bool add_point_light(const PointLight& l);  // false = refused, frame cap reached
[[nodiscard]] bool add_spot_light(const SpotLight& l);
[[nodiscard]] uint32_t lights_dropped() const;         // refused since begin_frame()
[[nodiscard]] uint32_t cluster_lights_dropped() const; // insertions the per-cluster cap refused
void build(core::jobs::JobSystem& js);
void build();                                  // convenience: JobSystem::instance()
void upload(SDL_GPUCopyPass* copy_pass);       // caller-owned pass
void reset_light_history();                    // forget last frame's identities
SDL_GPUBuffer* light_data_buffer() const;      // + cluster_offsets_buffer(), light_indices_buffer(),
                                               //   light_stable_ids_buffer(), light_remap_buffer()
const ClusterParams& params() const;
// test/debug: point_lights(), spot_lights(), light_indices(), cluster_offsets(), total_lights(),
//             stable_ids(), index_remap()
```

Staging lives in member scratch (`light_staging_`); upload records into the
caller's copy pass (the engine batches it into the frame's consolidated
`FrameUploads` pass).

#### Algorithm

- `begin_frame` clears per-frame light lists, captures `camera.view`, and
  recomputes `invProjection` **only when the projection changed** (cached in
  `last_projection_`). `zPlanes` is built from linear near/far clamped to
  `near >= 1e-3`, `far >= near + 1e-3`. Cluster view-space AABBs
  (`compute_cluster_bounds`) are recomputed only when projection or near/far
  change; Z slice boundaries are exponential:
  `z_i = near · (far/near)^(i/24)`. Note the near/far passed here are
  *linear distances* used for slicing, independent of whether the camera
  projection is standard or reversed-Z (pinned by
  `ClusterGrid.BinnedClustersCoverAllLitFragmentsReversedZ` in
  `light_cluster.test.cpp`). The cluster-AABB rebuild is a chunked
  `parallel_for(js, …)` fan-out (per-slice z depths precomputed serially;
  disjoint per-cluster writes, no merge).
- `add_*_light` returns `false` once `points + spots >= max_lights_`,
  increments `lights_dropped_` (reset each `begin_frame`, surfaced as
  `FrameDiagnostics::lights_dropped` and printed in the smoke line), and warns
  once per grid via `BRACKEN_LOG_WARN` (`warned_dropped_` is never reset — the
  cap is a standing property of the scene, not a per-frame event).
- `build()` first resolves **persistent light identity**
  (`assign_stable_ids_and_remap`, serial, emission order): host-supplied
  non-zero `stable_id`s are claimed verbatim in one pass, then every light
  that asked the engine (`stable_id == 0`) gets a quantized content hash —
  FNV-1a over `{type, round(position / 5.0), round(color / 0.25),
  round(log2(intensity))}` — with in-frame collisions broken by re-hashing
  with an occurrence counter until the id is unclaimed. Ids are never 0. The
  resolved values are written back into the uploaded light records AND into
  the packed stable-id buffer. The **remap** is then a diff of last frame's
  packed id list against this one: `remap[prev_index] = current_index`, or
  `LIGHT_REMAP_INVALID` (`0xFFFFFFFF`) where the light is gone. Two
  `<= MAX_LIGHTS` passes; no hashing container and no pointer-order
  iteration anywhere, so the result is bit-identical run to run. Identity
  history is renderer-side, main-thread tier — it never rides the packet.
  `reset_light_history()` clears it, and `SceneRenderer` calls it on frames
  it skips binning entirely (zero dynamic lights) so the next build cannot
  remap across the gap. Full contract: `contracts/light.hpp`.
- `build()` then transforms each light center to view space, and does
  **per-light cluster-range insertion**: a conservative cluster-index box
  (Z slices from the inverse exponential mapping; X/Y tiles from the NDC
  bounds of the sphere's view-space AABB corners projected through
  `last_projection_`) followed by an exact refine per cluster in the box —
  O(lights × touched clusters), not O(lights × 3456). Before projecting, the
  AABB is **clipped to `z <= -near`**: no froxel lives nearer than the near
  plane, so the clipped box still contains every sphere point a froxel can
  touch, every corner then has `w = -z >= near > 0`, and a light beside the
  camera can be rejected outright by the sideways test instead of surrendering
  the whole X/Y range. For a light entirely beyond the near plane the clip is
  a no-op. Lights with `radius <= 0` or entirely outside [near, far] are
  skipped.
- The per-cluster refine is a sphere/AABB test **ANDed, for a spot, with a
  conservative cone/AABB test** (the froxel's bounding sphere against the
  infinite cone, trig-free from `cos_outer`/`sin_outer` precomputed per
  light). Without it a spot was binned as its full radius sphere: the
  playground's 32° flashlight landed in 3246 of 3456 froxels instead of 286,
  and every one of those cost the fragment loop an 80-byte `SpotLight` load
  and a `spotConeFactor` that returns zero — and cost the Full-DI tier a RIS
  candidate draw with p̂ ≡ 0, since the froxel list is that pass's candidate
  domain. The cull can only remove froxels the forward loop already shades
  black, so it is image-neutral for the raster path. It is skipped for points
  and for any cone that is not well-formed (`outer_cos <= 0`, i.e. a
  half-angle at or past 90°, or a transposed `inner_cos < outer_cos` pair,
  whose lit set is the COMPLEMENT of the cone) — those fall back to sphere
  binning. Pinned by `ClusterGrid.SpotBinsCoverEveryFragmentInsideTheCone`
  and `ClusterGrid.SpotConeCullRemovesFroxelsTheConeMissesAndKeepsTheOnesItHits`
  in `light_cluster.test.cpp`.
- Insertion order is ascending light index → deterministic output (pinned by
  `ClusterGrid.BuildIsDeterministic`). A bin that reaches
  `MAX_LIGHTS_PER_CLUSTER` does **not** simply refuse the rest: it keeps the
  highest-contribution lights for that froxel, as a streaming replace-min
  (exact top-k) over `luminance × point_falloff(distance to the froxel's
  closest point, radius)` — the same `point_falloff` the forward loop and the
  RIS target function evaluate (`render/lighting/light_target.hpp`). The score
  is per-FROXEL rather than a scene-wide power proxy, because total power
  ranks a distant floodlight above the torch in the player's hand. Ties keep
  the incumbent (strictly-greater replacement), so identical lights still
  resolve to the lowest indices. Scores are filled lazily on a bin's first
  refusal, so a scene that never overflows allocates and evaluates nothing.
  Every refused insertion is still counted in `cluster_lights_dropped_` and
  warned about once per frame via `BRACKEN_LOG_WARN` (`warned_truncation_`,
  reset in `begin_frame`).
- Binning fans out over `js` chunked by **cluster
  index** (the per-light conservative cluster boxes are precomputed once,
  serially, as `LightRange` scratch): each chunk owns a disjoint cluster
  subrange, walks the light list ascending, and inserts only into its own
  clusters' bins (and is the only writer of their score rows) — every bin
  written by exactly one chunk, so per-bin order and which lights survive the
  cap are bit-identical to the serial light-major loop at any worker count
  (chunk-local truncation COUNTS, `chunk_truncated_` `uint32_t`, sum in
  chunk-index order after the join). At ≤ 8 lights or with an uninitialized
  pool the whole range runs as one inline chunk. Note a bin is no longer
  ascending in light index once eviction has run; no consumer assumes it is
  (`accumulateClusterLights`, `clusterFroxelLightCount`/`clusterFroxelLight`
  and the reservoir's uniform froxel draw all address lights by packed index).
  Finally offsets/indices are packed serially, setting `offsets_dirty_` only
  if any (offset,count) pair actually changed.
- Zero-light frames: only re-zero (and re-mark dirty) the offsets table if
  the previous build wrote counts (`offsets_all_zero_` flag).
- `upload()` packs the light-data buffer into member scratch and transfers
  **only up to the last used byte of the farthest populated section**; the
  offsets buffer is skipped when `!offsets_dirty_`; the index buffer uploads
  only the used prefix (zero indices ⇒ nothing uploaded; safe because every
  count is 0). A single contiguous update is required for the light-data
  buffer — multiple region updates to one buffer would cycle the destination.

#### Screen-orientation contract

CPU binning maps cluster row `cy = 0` to NDC `y = -1` (bottom of screen);
the shader lookup starts from a top-left-origin `SV_Position` UV, so
`clusterIndexFromScreen` (`clusters.hlsli`) flips Y (`1 - uv.y`). This is
regression-pinned by
`ClusterGrid.ShaderScreenLookupMatchesCpuBinningOrientation`.

#### Shader bindings (`src/bracken/shaders/cluster_lighting.hlsli`)

- `ClusterParams` cbuffer at **set 3, binding 2** (pushed via
  `SDL_PushGPUFragmentUniformData(cmd, 2, ...)` in
  `SceneRenderer::bind_cluster_lights`).
- **Five** fragment storage buffers bound with
  `SDL_BindGPUFragmentStorageBuffers(pass, 0, bufs, 5)`:
  `lightDataBuf` (ByteAddressBuffer), `clusterOffsetsBuf`
  (`StructuredBuffer<uint2>`), `lightIndicesBuf`
  (`StructuredBuffer<uint>`), `localShadowSlots`
  (`StructuredBuffer<LocalShadowSlot>`, also declared by
  `cluster_lighting.hlsli`), and the R4b screen-probe SH declared by
  `gi_screen_sample.hlsli` at `CLUSTER_SSBO_BINDING_BASE + 4`. Slots 3 and 4
  are backfilled with `light_data_buffer()` when local shadows / screen probes
  are unavailable: the slot must hold a valid buffer, and neither is read then
  (`shadow_slot == -1`, `giScreenEnabled == 0`). SDL GPU lays out fragment
  set 2 as sampled textures, then storage textures, then storage buffers, so
  the including shader must `#define CLUSTER_SSBO_BINDING_BASE <its fragment
  sampler count PLUS its storage-texture count>` before the include (a
  `#error` enforces it), and `CLUSTER_SAMPLER_COUNT` separately when the two
  differ — which they have since R5c, because the Full-DI tier declares two
  storage textures. Current values: `CLUSTER_SAMPLER_COUNT` 16 / 16 / 14 / 12
  and `CLUSTER_SSBO_BINDING_BASE` (= count + 3) 19 / 19 / 17 / 15 for
  `model.frag` / `model_instanced.frag` / `terrain.frag` / `grass.frag`.
- Shader-side hardening: the per-cluster loop count is
  `min(offsets.count, maxPerCluster.x)` so a stale/partial offsets buffer
  cannot spin ~2³² iterations; `gridDim.w == 0` short-circuits the whole
  accumulation.
- Attenuation: `pointFalloff(d,R) = saturate(1-(d/R)^4)^2 / max(d², 1e-4)`;
  spots multiply `smoothstep(outer_cos, inner_cos, dot(spotDir, -L))`. Both
  live in `cluster_index.hlsli` (binding-free by construction, so every
  consumer can share the one copy) and are mirrored on the CPU by
  `render/lighting/light_target.hpp`. Both feed the standard `evaluatePBR`
  BRDF. The loop's three structural zeros — `dist > radius`, `cone <= 0` and
  `dot(N, L) <= 0` — are all tested BEFORE the local-shadow atlas lookup, so a
  back-facing or out-of-cone pixel never pays the matrix transform and four
  `SampleCmpLevelZero` taps for a visibility term that would be multiplied by
  zero. The `N·L` predicate is `evaluatePBR`'s own, verbatim, so the two
  cannot drift.

#### SceneRenderer integration & silent behaviors

- `init` is called with `MAX_LIGHTS`; on failure `cluster_grid_ready_` stays
  false and dynamic lighting is disabled (warning to stderr).
- Per frame (only when the scene has ≥1 dynamic light):
  `begin_frame(camera, w, h, 0.1f, gfx.get_draw_distance())` → add lights
  from `FramePacket::point_lights()/spot_lights()` → `build()` → upload
  recorded into the renderer's single consolidated `FrameUploads` copy pass
  (`upload(SDL_GPUCopyPass*)` overload).
- Zero-light frames skip build+upload; `SceneRenderer` publishes a
  zero-initialized `frame_cluster_params_` so shaders see
  `gridDim.w == 0` — the SSBOs **retain stale contents** (documented in
  `docs/architecture.md`). `bind_cluster_lights` always pushes
  `frame_cluster_params_`, never `cluster_grid_.params()`.
- `FrameBuilder::add_point_light/add_spot_light`
  (`src/bracken/scene/frame_builder.hpp`) are unbounded; the 256 cap is
  enforced at `ClusterGrid::add_*`, which refuses the excess, counts it in
  `FrameDiagnostics::lights_dropped` and warns once per grid — check
  `RenderStats::diagnostics` before believing a missing lamp is a shader bug.

#### Threading / lifetime / testing

- No internal synchronization; the begin/add/build/upload sequence must be a
  single-threaded per-frame protocol. Buffer handles remain valid between
  frames (SDL GPU cycles dynamic buffers internally on update).
- `begin_frame`/`add_*`/`build` never touch GPU state — CPU-only tests and
  the benchmark construct a `ClusterGrid` without `init()`
  (`light_cluster.test.cpp`, `light_cluster.bench.cpp`; the latter measures
  the per-frame binning cost at 16/64/256 lights, including a mixed
  point+spot workload).
- Destructor calls `shutdown()`.

### Local (point/spot) light shadows — `render/lighting/local_shadows.hpp/.cpp`

Budgeted shadow maps for clustered lights the host opts in via
`PointLight/SpotLight::casts_shadows`. Two classes:

- **`LocalShadowSelector`** — CPU-only selection + tile cache (fully
  unit-testable, `local_shadows.test.cpp`). `update(points, spots,
  camera_pos, dynamic_casters, config, tile_resolution)` runs once per frame:
  1. *Scoring*: candidates = lights with `casts_shadows && radius > 0 &&
     intensity > 0`; score = `intensity · radius / max(distance_to_camera, 1)`
     (`local_shadow_light_score`).
  2. *Matching*: incumbents match this frame's lights by nearest position of
     the same type within `0.5 · max(radii)` (immediate-mode scenes carry no
     light identity — positional continuity IS identity; a teleported light
     reads as new). Unmatched incumbents free their tiles.
  3. *Fill + eviction with hysteresis*: free groups take the best unclaimed
     candidates (sorted score desc, submission index asc — deterministic,
     pinned by `DeterministicAcrossIdenticalRuns`); a challenger evicts the
     weakest incumbent only when its score exceeds `weakest · (1 +
     hysteresis)` (default 0.25) so boundary lights don't thrash slots.
  4. *Dirty analysis*: a group's tiles re-render when the light's
     shadow-relevant state changed (position/radius; spots also
     direction/outer_cos — color/intensity do NOT dirty) or a dynamic caster
     sphere (skinned or nonzero-`motion_id` command) intersects the light
     sphere.
  5. *Scheduling under the cap*: `max_tile_updates_per_frame` (default 6)
     tiles per frame, walked round-robin from a persistent ring cursor.
     Fresh assignments render all-or-nothing (an incomplete group publishes
     `shadow_slot = -1` — unshadowed, never garbage); stale groups update
     partially (each tile's SSBO matrix is written only when the tile
     renders, so content and matrix always agree).
  6. *Publication*: fully-valid groups write `shadow_slot` (point = base of
     6 face tiles, spot = single tile) into the frame's light copies.
- **`LocalShadowMap`** — GPU wrapper: one D32 atlas (default 2048², settings
  table {1024, 2048, 4096}; `reinit` invalidates every cached tile), 8×8
  tiles (`LOCAL_SHADOW_TILES_PER_ROW`; point groups own tiles `[g·6, g·6+6)`
  for g < 8, spots tiles 48–63 — `LOCAL_SHADOW_SPOT_TILE_BASE`), a
  LESS_OR_EQUAL comparison sampler, and the 64-entry slot SSBO
  (`LocalShadowSlotGPU`, 96 B: `mat4 view_proj`, `vec4 rect` (atlas-UV
  origin/scale), `vec4 params` (world-texel-per-unit-distance, tile-UV PCF
  inset, near plane, atlas texel) — mirrored as `LocalShadowSlot` in
  `cluster_lighting.hlsli`). `upload()` records the 6 KB slot upload into
  the frame copy pass only when a tile re-rendered. `begin_pass()` LOADs the
  atlas (first-ever pass clears); dirty tiles clear themselves via the
  `LocalShadowClear` pipeline with the tile viewport's depth range pinned to
  [1,1] (`set_tile_viewport(tile, clear_depth_range)`).

Matrices (pure helpers, unit-tested): point faces use
`local_point_face_view_proj` — standard-Z `perspectiveRH_ZO` (near→0,
far = light radius→1), FOV = `2·atan(res/(res−4))` (slightly over 90° so
seam PCF taps stay valid), face order +X −X +Y −Y +Z −Z selected by the
dominant axis of light→fragment (`local_shadow_cube_face` ↔ HLSL
`localShadowCubeFace`); spots use `local_spot_view_proj` — FOV = outer cone
angle ×1.1 (clamped 5°–170°). Near plane = `clamp(0.02·radius, 1, 20)`
(`local_shadow_near_plane`). **Deliberately standard-Z** like the CSM
cascades, so the local passes reuse the CSM caster pipelines
(`ShadowModel`/`ShadowSkinnedModel`: cull front, depth bias 1.25/1.75,
depth clamp) verbatim.

Rendering (`SceneRenderer::render_local_shadow_passes`, debug scope
`LocalShadowPass`, after the CSM `ShadowPass` and independent of
`shadow_mode`): per tile request — clear draw, then model + skinned
commands within the light radius (rebuilt per light; requests for one
light's faces are consecutive) drawn individually with a per-tile frustum
test; `mesh.cast_shadows` respected, alpha-tested where textured. Terrain
and grass do not cast local shadows. Instanced-model commands draw
non-instanced here (per-light caster sets are small).

Receiver sampling (`cluster_lighting.hlsli`): lights with `shadow_slot >= 0`
call `localLightShadow` inside `accumulateClusterLights` — 4
`SampleCmpLevelZero` taps on a half-texel square (each a hardware 2×2 PCF),
world-space receiver bias (normal offset + slope-scaled push toward the
light, both ∝ the tile's world texel size at the receiver distance — no
perspective-depth-unit conversion needed), UV clamped inside the tile so
taps never bleed across, and a smooth fade to unshadowed over the last 25%
of the light radius. The whole light contribution multiplies by the result.

Settings plumbing: `local_shadows` toggle + budgets clamp to
`LOCAL_SHADOW_MAX_POINT_BUDGET = 8` / `LOCAL_SHADOW_MAX_SPOT_BUDGET = 16`
(the 8×8 atlas exactly: 8·6 + 16 = 64 tiles). When the system is off or
unavailable every uploaded light gets `shadow_slot = -1` and the sampler /
4th-SSBO slots are bound with typed fillers (CSM atlas / light-data
buffer) that are never sampled. Introspection for the dev overlay and the
playground smoke line: `selector().tiles_rendered_last_frame()`,
`tiles_rendered_total()`, `shadowed_point_lights()`,
`shadowed_spot_lights()` via `SceneRenderer::local_shadows()`.

### The directional sun

Not part of the cluster system: `EnvironmentState::sun_dir` / `sun_color`
(`src/bracken/contracts/environment.hpp`, `sun_dir` = direction light travels,
default `normalize(-0.5, -0.8, -0.3)`, colors linear) is evaluated
unconditionally per fragment, scaled by `GraphicsSettings::sun_intensity`
(default 3.0). `GraphicsSettings::ambient_strength` (default 0.45) sets the
hemisphere-ambient floor in shadow (`ambientLift = lerp(ambient_strength, 1,
shadow)` in `model.frag.hlsl`). `SceneRenderer` refreshes
`light_dir_ = env_.sun_dir` each frame and feeds it to `ShadowMap::update`.

### Cascaded shadow maps (`render::ShadowMap`)

#### Purpose & storage

Directional-sun CSM. All cascades render into **one D32_FLOAT atlas** of
`(2·resolution)²` texels; cascade *i* occupies quadrant
`((i%2)·res, (i/2)·res)`. Rationale (header comment,
`src/bracken/render/shadow_map.hpp`): SDL3 GPU cannot render to depth-array
layers, and a single atlas lets all cascades share one render pass and one
pair of sampler bindings in receivers. `CSM_MAX_CASCADES = 4`.

#### Public API

```cpp
[[nodiscard]] bool init(gpu::GPUDevice& device, int resolution = 2048);
void shutdown();
bool reinit(int resolution);          // recreates atlas only; false if init() never ran
void set_active_cascades(int count);  // clamped 1..4, no reinit
int  active_cascades() const; int resolution() const;
void update(const glm::mat4& camera_view, const glm::mat4& camera_proj,
            const glm::vec3& light_dir, float near_plane, float far_plane);
SDL_GPURenderPass* begin_shadow_pass(SDL_GPUCommandBuffer* cmd); // clears whole atlas
void set_cascade_viewport(int cascade_index); // viewport + scissor to quadrant
void end_shadow_pass();
const std::array<CascadeData, CSM_MAX_CASCADES>& cascades() const;
SDL_GPUTexture* atlas_texture() const;
bool is_ready() const;
SDL_GPUSampler* shadow_sampler() const;         // raw NEAREST (PCSS blocker search, fog)
SDL_GPUSampler* shadow_compare_sampler() const; // LINEAR compare LESS_OR_EQUAL (hw PCF)
gpu::ShadowDataUniforms get_shadow_uniforms(int shadow_mode = 2) const;
float light_size = 0.02f;   // tan(sun angular radius) — PCSS penumbra scale
float split_lambda = 0.85f; // log/uniform split mix
```

`CascadeData` = `{mat4 light_view_projection, float split_depth (view-space
far of the cascade), float texel_world (world units per shadow texel), float
depth_range (light-space depth extent in world units, = caster_reach + receiver_reach)}`.

#### Algorithms

- **Splits** (`compute_cascade_splits`): practical split scheme —
  `split_i = λ·near·(far/near)^p + (1-λ)·(near + (far-near)·p)` with
  `p = (i+1)/active`, `λ = split_lambda = 0.85`.
- **Fitting** (`fit_cascade`): the camera projection's z-row is rewritten
  with a self-consistent *standard* RH_ZO mapping for `[near_split,
  far_split]` (only the CPU-side `inv_vp` consumes it, so it works whether
  the camera projection is reversed-Z or not); the 8 NDC corners
  (z ∈ [0,1]) are unprojected to world space; the slice's **bounding
  sphere** is used (radius depends only on fov/aspect/splits, never camera
  orientation, so the ortho window size is rotation-stable), rounded up to
  1/16 to defeat FP jitter. The light view is **rotation-only**
  (`glm::lookAt(vec3(0), light_dir, up)`, up flips to +X when
  |dot(light,up)| > 0.99); all translation lives in the ortho window, where
  the sphere center is snapped to whole shadow texels
  (`texel = 2r/resolution`) — this is what makes static shadow edges
  pixel-stable under camera translation and rotation. The ortho depth window
  is **asymmetric**: `glm::orthoRH_ZO(cx±r, cy±r, -(cz+caster_reach),
  -(cz-receiver_reach))` with `caster_reach = r + CASTER_HEIGHT_BUDGET /
  max(|light_dir.y|, MIN_SUN_ELEVATION_SIN)` and `receiver_reach = r +
  Z_RECEIVER_PAD` (file-local constants in `shadow_map.cpp`:
  `CASTER_HEIGHT_BUDGET = 1500`, `MIN_SUN_ELEVATION_SIN = 0.15`,
  `Z_RECEIVER_PAD = 100`). The toward-light reach is elevation-dependent
  because a caster of height h up-sun of the slice shadows it from up to
  h/sin(elevation) away in light-space Z — a fixed pad culls such casters
  (CPU frustum test) and their shadows pop in only when the caster enters
  the slice. `depth_range = caster_reach + receiver_reach` feeds the
  bias/PCSS uniforms, so the shader math stays calibrated automatically.
- **Depth convention**: cascades deliberately stay **standard-Z** while the
  scene is reversed-Z: atlas cleared to `1.0`, all four shadow pipelines and
  the compare sampler use `SDL_GPU_COMPAREOP_LESS_OR_EQUAL`. The cascades'
  orthographic projections distribute depth linearly so reversed-Z buys
  nothing, and the receiver bias/PCF/PCSS math is calibrated for
  near=0/far=1 (comments in `shadow_map.cpp` and
  `docs/architecture.md`).

#### Uniform contract (`gpu::ShadowDataUniforms`, set 3 b1)

`src/bracken/gpu/gpu_uniforms.hpp`: `{mat4 lightViewProjection[4], vec4
cascadeSplits, vec4 cascadeTexelWorld, vec4 cascadeDepthRange, float
shadowMapResolution (quadrant res), float lightSize, float shadowEnabled,
float lastActiveSplit, float fadeStart, float fadeEnd, float _pad0, _pad1}`.
`shadowEnabled` is a **mode, not a bool**: 0 = off, 1 = PCF, 2 = PCSS.
`get_shadow_uniforms` clones the last active cascade into inactive slots
(split equals `lastActiveSplit` so selection can't reach them; texel/range
stay non-zero for safe dynamic indexing) and sets `fadeStart =
shadow_far·0.85`, `fadeEnd = shadow_far` (`shadow_far_` latched from the
last `update()`; member default 1200.0f).

#### Shadow render passes (SceneRenderer)

- Pipelines (`src/bracken/gpu/pipeline_catalog.cpp`): `ShadowModel`
  (`shadow_depth.vert` + alpha-testing `shadow_depth.frag`),
  `InstancedShadowModel` (`shadow_depth_instanced.vert` + same frag),
  `ShadowSkinnedModel` (`shadow_skinned.vert` + no-alpha
  `shadow_depth_opaque.frag`), `ShadowTerrain` (`shadow_terrain.vert` +
  opaque frag). All: depth-only (`color_format = INVALID`), **front-face
  cull**, depth bias `with_depth_bias(1.25f, 1.75f)`,
  `LESS_OR_EQUAL`.
- Vertex uniforms: `gpu::ShadowTransformUniforms` (lightVP + model),
  `ShadowSkinnedTransformUniforms` (+ `paletteOffset` into the frame bone
  SSBO), `ShadowTerrainUniforms` (lightVP only); fragment
  `ShadowFragUniforms{int hasTexture}` for the alpha-test path (padding is
  three named scalars to match HLSL packing).
- One render pass over the whole atlas per frame (`ShadowPass`, GPU-labeled
  `ShadowPass:Cascade0..3`), viewport/scissor per quadrant. Casters are
  **frustum-culled per cascade** against frustums extracted from each
  `light_view_projection`. Meshes with `cast_shadows == false` are skipped.
  Runs only when `shadow_mode != ShadowMode::Off`.
- `SceneRenderer` computes `shadow_far = min(get_draw_distance(), shadow_distance)` (a `GraphicsSettings` field, default 4000)
  and recovers the camera near plane from the reversed-Z projection as
  `m[3][2]/(m[2][2]+1)` clamped to ≥ 0.1.
- Receiver binding (`SceneRenderer::bind_shadow_data`): atlas bound twice at
  consecutive sampler slots — `[slot] = {atlas, compare_sampler}` then
  `[slot+1] = {atlas, raw_sampler}` (slot 1 for the model family; material
  textures occupy 3–5) — plus `ShadowDataUniforms` pushed at fragment
  uniform slot 1 (cached once per frame in `frame_shadow_uniforms_`).

#### Receiver filtering (`src/bracken/shaders/shadow_common.hlsli`)

- Cascade selected by comparing view depth to `cascadeSplits`; early-out
  returns fully lit past `lastActiveSplit`; distance fade
  `saturate((viewDepth - fadeStart)/(fadeEnd - fadeStart))` lerps toward 1.
- Per-tap addressing: normal-offset bias of one `texelWorld` along the
  geometric normal, then slope-scaled receiver bias
  `(0.5 + 1.5·tanθ)·texelWorld/depthRange` with `tanθ` clamped to 3 — the
  bias is exactly as large as the geometry it must hide. Atlas UVs are
  quadrant origin + half-scale, with every tap clamped one texel inside the
  quadrant so filtering never bleeds across cascades.
- Filter modes: **0** = single hardware-PCF tap (used for back-facing
  fragments, whose direct term is zero anyway — saves up to 20 taps on
  ~half of geometry); **1 (PCF)** = 3×3 tent of hardware-PCF compares
  (weights 1-2-1/2-4-2/1-2-1, ÷16); **2 (PCSS)** = 12-sample Poisson disk
  rotated per pixel by interleaved gradient noise; blocker search = 8 raw
  taps over `clamp(sc.z·depthRange·lightSize/texelWorld, 2, 12)` texels,
  penumbra `=(sc.z - avgBlocker)·depthRange·lightSize/texelWorld` clamped to
  [1, 16] texels.
- Cascade transition: blend over the last 15% of each cascade's range; PCF
  samples both cascades and lerps; PCSS instead **dithers** the cascade
  choice with IGN (offset `(17, 59)` to decorrelate from the Poisson
  rotation) so only one cascade is filtered per pixel.

#### Error handling / lifetime / gotchas

- `init` returns bool and logs via `SDL_Log`; `SceneRenderer` degrades to
  shadows-disabled. `reinit` recreates only the atlas (samplers survive);
  returns true immediately if the resolution is unchanged, false if there is
  no device yet (settings applied before `init` are picked up at `init`
  time via `set_active_cascades` + `shadow_map_texels(ShadowResolution)`,
  `contracts/graphics_settings.hpp`).
- `begin_shadow_pass` returns nullptr on failure; `set_cascade_viewport`
  no-ops without a live pass or with an out-of-range index. The class holds
  the current pass pointer (`current_shadow_pass_`) — do not interleave two
  ShadowMaps' passes.
- With `shadow_mode == Off` the atlas is **never rendered** and contains
  garbage; `SceneRenderer` force-disables god rays for that reason. Any
  other consumer of `atlas_texture()` (e.g. `render::VolumetricFog`) must
  gate the same way.
- `update()` must precede `begin_shadow_pass()`; `SceneRenderer` calls it
  early (before instance-batch building) so per-cascade light frustums can
  cull casters.
- Shadow passes alpha-test `BLEND` casters at 0.5 rather than rendering
  translucent shadows (`docs/architecture.md`).

### Ambient occlusion (`render::GTAO`) & post chain (`render::PostProcessStack`)

#### Purpose & scope

The former `AmbientOcclusion` class was split in the AO rearchitecture:

- **`render::PostProcessStack`** (`src/bracken/render/post_process_stack.hpp`)
  owns the **full-resolution offscreen HDR scene target** (RGBA16F color +
  D32 depth) that replaces the direct-to-swapchain main pass whenever the
  subsystem initialized (`use_hdr_path = post_->is_ready()` in
  `scene_renderer.cpp` — even with `AOMode::Off`), the **composite**
  (fog upsample + tonemap) pass, and the optional **FXAA** resolve via a
  lazily created LDR intermediate.
- **`render::GTAO`** (`src/bracken/render/gtao.hpp`) owns the screen-space
  ambient-occlusion chain: horizon-based GTAO (Jimenez et al. 2016) whose
  per-slice horizon pair became a 32-sector **visibility bitmask** in R5a
  (SSILVB, Therrien et al. 2023), computed from the depth-prepass depth
  **before the forward pass**, plus temporal accumulation, bilateral blur,
  and a joint-bilateral upsample to the full-res **RGBA8** texture the forward
  shaders sample. That texture carries R = AO, GB = octahedral world-space
  **bent normal**, A = **cone aperture** as sin²(half-angle); the layout,
  the `A >= 0.998` "fully open / no directional data" escape value and the
  GI gate curve all live in `shaders/bent_cone.hlsli`. Screen AO attenuates
  the forward **ambient term only** (min-combined with material ORM AO in
  the `ao` argument of `ambientPBR`/`ambientIBL`), and the diffuse GI term is
  gated by the cone aperture INSTEAD of that scalar, never by both; direct sun
  light, speculars, and cluster lights are never attenuated.

#### `render::GTAO` public API

```cpp
[[nodiscard]] bool init(gpu::GPUDevice&, int width, int height, bool half_res = true);
void shutdown();
void resize(int width, int height);   // no-op when unchanged; invalidates history
void set_half_res(bool);              // AOMode::GTAO_Half vs GTAO_Full working res
bool is_ready() const;
void invalidate_history();            // next render() outputs the current frame
SDL_GPUTexture* ao_texture() const;   // full-res RGBA8, white-cleared at init/resize
struct FrameParams { glm::mat4 projection;      // the DRAW (jittered) projection
                     glm::mat4 inv_projection;
                     glm::mat4 inv_view;        // bent normal is published in WORLD space
                     float radius_scale = 1.0f; // GraphicsSettings::ao_radius
                     float strength = 1.0f;     // GraphicsSettings::ao_strength
                     uint64_t frame_index = 0; };
void render(SDL_GPUCommandBuffer*, gpu::PipelineRegistry&,
            gpu::GPUTexture& depth, gpu::GPUTexture& velocity, const FrameParams&);
```

Plus the test-exposed CPU reference in `detail::`, which is the authority the
shader transliterates (all pinned by `gtao.test.cpp`):

| Function | What it is |
|---|---|
| `gtao_slice_visibility(h0, h1, n)` | the pre-R5a cosine-weighted arc integral; kept as the analytic ground truth the 32 sector weights must sum to |
| `gtao_sector_angle(k, n)` / `gtao_arc_antiderivative` / `gtao_arc_cumulative` | the hemicircle parameterisation and the antiderivative `A(θ) = ∫₀^θ sin φ·cos(φ−n) dφ` |
| `gtao_sector_weight(k, n)` | exact cosine weight of sector `k`; strictly positive, and the 32 of them sum to the unoccluded arc value |
| `gtao_mark_sectors(lo, hi)` | which bits a normalised angular span sets (width rounds outward, start inward) |
| `gtao_slab_span(front, back, side, n)` | the constant-thickness occluder's span, normalised onto the hemicircle |
| `gtao_slice_visibility_bitmask(mask, n)` | visibility of the clear sectors |
| `gtao_cone_cos_from_resultant(r)` / `gtao_cone_aperture(r)` | the cone fitted to the visibility lobe's first moment, and its stored sin²α |

`GTAO_SECTOR_COUNT = 32` (one `uint`, `countbits` is plain ALU — no wave
intrinsics in the pass). The tests include a **brute-force gate**: 200 random
synthetic depth profiles integrated exactly over the true visible set at 20k
samples, bounded against the bitmask result by the sector quantisation.

#### `render::PostProcessStack` public API

```cpp
[[nodiscard]] bool init(gpu::GPUDevice& device, int width, int height);
void shutdown();
void resize(int width, int height);              // no-op when unchanged
void set_clear_color(glm::vec3 linear);          // offscreen clear; track fog_color_horizon
SDL_GPURenderPass* begin_offscreen_pass(SDL_GPUCommandBuffer* cmd, bool load_depth = false);
void end_offscreen_pass();
struct CompositeParams { float bloom_strength = 0.25f;
    float exposure = 1.0f; int tonemap_mode = 0; float contrast = 1.0f;
    float saturation = 1.0f; float depth_a = 0.0f; float depth_b = 0.0f;
    // Auto exposure: flag + the AutoExposure state SSBO (null ⇒ a zeroed
    // dummy keeps the composite shader's SSBO slot valid, flag forced off).
    bool auto_exposure = false; SDL_GPUBuffer* exposure_buffer = nullptr;
    // Color grading; all-neutral defaults ⇒ pixel-identical output (pinned
    // by post_process_stack.test.cpp — the pass gates each block on
    // non-neutral values because LMS/pow round-trips aren't bit-exact).
    glm::vec3 white_balance{1.0f}; glm::vec3 cdl_slope{1.0f};
    glm::vec3 cdl_offset{0.0f};    glm::vec3 cdl_power{1.0f}; };
void render_composite_pass(SDL_GPUCommandBuffer*, gpu::PipelineRegistry&,
                           SDL_GPUTexture* target, SDL_GPUTexture* bloom_texture,
                           SDL_GPUTexture* fog_texture, const CompositeParams&);
void render_fxaa_pass(SDL_GPUCommandBuffer*, SDL_GPUTexture* swapchain_target);
bool is_ready() const;
gpu::GPUTexture* offscreen_color();  gpu::GPUTexture* offscreen_depth();
SDL_GPUTexture* ldr_target();        // lazy, swapchain-format, swapchain-sized
```

#### Textures, formats, resolutions

PostProcessStack:

- `offscreen_color_`: full-res `R16G16B16A16_FLOAT`, COLOR_TARGET|SAMPLER.
- `offscreen_depth_`: full-res `D32_FLOAT` (via `GPUTexture::create_depth`),
  cleared to **0.0** (reversed-Z far) by `begin_offscreen_pass` unless
  `load_depth` (post-prepass) LOADs the prepass depth.
- `ldr_target_`: lazily (re)created at current size and
  `device.swapchain_format()`; composite destination when FXAA is on.

GTAO (working res = `max(1, w/2) × max(1, h/2)` for `GTAO_Half`, full res
for `GTAO_Full`):

- `ao_raw_`, `blur_tmp_`, `ao_blurred_`: working-res `R8G8B8A8_UNORM`
  (AO, oct.x, oct.y, aperture).
- `history_[2]`: working-res `R16G16B16A16_FLOAT` double buffer — the same
  four channels. 16F rather than 8-bit because a 0.9-feedback accumulation
  stalls against an 8-bit grid (a per-frame delta under half an LSB never
  lands).
- `history_depth_[2]`: working-res `R16G16_FLOAT` double buffer — R = that
  frame's linear view depth (disocclusion rejection input). A SECOND target
  because the four channels above are full, all five values are linearly
  filtered under the history's linear sampler, and therefore none of them can
  be bit-packed. `GTAOTemporal` writes both in one MRT pass.
- `ao_full_`: **full-res** `R8G8B8A8_UNORM`, the texture the forward pass
  samples; white-cleared at init/resize/quality change via a one-off command
  buffer so it is safe to bind before the first render() — white decodes as
  "AO 1, cone fully open, bent normal = shading normal", exactly the pre-R5a
  no-AO behaviour, which is why no uniform flag gates the bent cone.
- Both classes own a `nearest_clamp_sampler_` + `linear_clamp_sampler_`.

#### Uniform contracts (`src/bracken/gpu/gpu_uniforms.hpp`, all set 3 b0)

- `GTAOUniforms` (**240 B** since R5a): `{mat4 projection, mat4 invProjection,
  mat4 invView, vec2 screenSize (working res), vec2 invScreenSize, float
  radius = 4.0, radiusScale = 1.0, falloffRange = 0.615,
  occluderThickness = 1.0, temporalRotation, temporalOffset,
  maxDistance = 12000.0, pad}`. `invView` is view→world, needed because the
  pass reconstructs in view space and publishes the bent normal in WORLD space
  (no consumer of the AO texture carries a view matrix). `radius` is in world
  units, tuned for 25–35-unit-tall characters; `radiusScale` carries
  `GraphicsSettings::ao_radius`; **`occluderThickness` is the SSILVB slab
  depth along the view ray, in world units**, scaled by `radiusScale`
  alongside the radius — it replaces the unitless
  `thinOccluderCompensation = 0.2` horizon-decay heuristic. The temporal
  fields carry the per-frame golden-ratio jitter; `maxDistance` is an explicit
  linear-view-distance AO cutoff faded over its last 10% in-shader and
  **must equal `BlurUniforms::maxDistance`** (pinned by `gtao.test.cpp`).
  Slice/step/sector counts are shader compile-time constants
  (`SLICE_COUNT = 3`, `STEPS_PER_SLICE = 4`, `SECTOR_COUNT = 32`).
- `GtaoTemporalUniforms` (32 B, unchanged): `{vec2 invScreenSize, float
  depthA, depthB, historyValid, feedbackMax = 0.9, depthRejectScale = 10.0,
  pad}`. The pass now declares **5 samplers** (raw, history, velocity, depth,
  depth history) and **2 colour targets**.
- `GtaoUpsampleUniforms` (32 B, unchanged): `{vec2 lowResTexelSize, float
  depthA, depthB, strength, pads}` — `strength` = `ao_strength`, applied as
  `pow(x, strength)` to BOTH the AO and the aperture channel, so
  `ao_strength = 0` disables the bent-cone gate along with AO instead of
  leaving it gating on its own.
- `BlurUniforms`: `{vec2 direction, vec2 invScreenSize, float sharpness =
  40.0, float depthA, float depthB, float maxDistance = 12000.0}`.
- `CompositeUniforms` (112 B): `{float bloomStrength = 0.25,
  volumetricFogEnabled, exposure = 1.0, int tonemapMode = 0,
  contrast = 1.0, saturation = 1.0, vec2 lowResTexelSize, float depthA,
  depthB, autoExposureEnabled = 0, whiteBalanceEnabled = 0,
  vec3 whiteBalance = 1, float cdlEnabled = 0, vec3 cdlSlope = 1 + pad,
  vec3 cdlOffset = 0 + pad, vec3 cdlPower = 1 + pad}` — no AO fields;
  `lowResTexelSize` is the half-res fog grid; the enabled flags are computed
  by `render_composite_pass` from the params (never trusted from callers).
- `FxaaUniforms`: `{vec2 invScreenSize, float _pad0, _pad1}`.

Depth linearization used by the temporal/blur/upsample/composite passes:
`viewZ = |depthB / (depth + depthA)|` with `depthA = proj[2][2]`,
`depthB = proj[3][2]` (glm RH_ZO) — valid for standard **and** reversed-Z
projections. Zeros degrade to plain bilinear weighting in-shader. The
expression itself is NOT open-coded here or anywhere else: it lives in
`contracts/depth_linear.hpp` and `shaders/depth_linear.hlsli`, which is the
whole point (a sign typo in a copy is self-consistent and therefore invisible —
see the 2026-08-02 entry under reversed-Z above).

#### Pass algorithms

- **GTAO** (`gtao.frag.hlsl`): sky skip on `depth <= 0.0`; view position via
  `invProjection` (Vulkan Y flip); **edge-aware depth-reconstructed
  normals** — 4 extra depth taps (left/right/up/down), per axis picking the
  neighbor whose depth is closer to the center so the derivative never
  crosses a silhouette (chosen over a prepass normal MRT: cheaper than a
  full-res normal write+sample, and AO only needs geometric normals);
  screen-space radius `effRadius·proj[0][0]·0.5/viewZ` clamped to
  `[2·invScreenSize.x, 0.2]`; for each of 3 slices (angles jittered
  per-pixel by IGN + per-frame by a golden-ratio sequence), 4 samples per
  side with squared step distribution update two horizon cosines with an
  XeGTAO-style linear distance falloff (`falloffRange` outer fraction of the
  radius) and thin-occluder decay (`lerp(horizon, sample, 0.2)` when the
  sample is below the horizon); the two horizons are clamped to the
  hemisphere around the projected normal and integrated analytically
  (cosine-weighted arc, `projLen`-weighted). Output
  `lerp(1, visibility, distanceFade)` to `ao_raw_`.
- **Temporal** (`gtao_temporal.frag.hlsl`): reprojects the previous history
  at `prev_uv = uv - velocity` (engine velocity convention,
  `contracts/velocity_math.hpp`); rejects on relative linear-depth mismatch
  against the depth stored in the history G channel (`feedback =
  feedbackMax · saturate(1 - relDiff · depthRejectScale)`); off-screen
  prev_uv or `historyValid == 0` fall back to the current AO. Writes
  `{blended AO, current linear depth}` to the write history slot.
- **Blur** (`blur_ao.frag.hlsl`, unchanged): separable 7-tap Gaussian, edge
  weight `exp(-|Δ viewZ|/centerZ · sharpness)` on *relative* linear view-Z.
  Runs history → `blur_tmp_` → `ao_blurred_`; the history itself stays
  unfiltered so the kernel never compounds across frames.
- **Upsample** (`gtao_upsample.frag.hlsl`): joint-bilateral 4-tap upsample
  (bilinear × relative-depth similarity, `DEPTH_SHARPNESS = 40` — same
  metric as the blur and the composite fog upsample) from `ao_blurred_` to
  the full-res `ao_full_`; degenerates to an exact passthrough when working
  res == full res; applies `strength` as a power curve.
- **Composite** (`render_composite_pass` + `composite.frag.hlsl`):
  fullscreen triangle; 4 sampler bindings — 0 scene HDR color (linear),
  1 bloom, 2 fog, 3 full-res depth (nearest) — plus fragment storage
  buffer 0 = the auto-exposure state SSBO (`CompositeParams::
  exposure_buffer`, or the init-created 16-byte zeroed
  `dummy_exposure_buffer_` when null; `is_ready()` requires it). Null
  `bloom_texture`/`fog_texture` are replaced by the scene color texture as a
  safe dummy with `bloomStrength = 0` / `volumetricFogEnabled = 0`. Shader
  order: fog blend → bloom add → `color *= exposure ·
  (autoExposureEnabled ? exp2(state[0].x) : 1)` → flag-gated white balance
  (linear→LMS (CIECAM02 matrices in the shader) × `whiteBalance` gains →
  back, clamped ≥ 0) → tonemap (`tonemapMode` = `TonemapMode` enum:
  0 ACESFitted, 1 ACESNarkowicz, 2 Reinhard, 3 None) → flag-gated ASC-CDL
  `pow(max(c·slope + offset, 0), power)` → contrast → saturation → gamma →
  IGN dither, to swapchain-format `target`. White balance is deliberately
  PRE-tonemap (it models the scene illuminant, so it must see
  scene-referred light); the CDL is display-referred, POST-tonemap.
  **AO is not sampled here.**
- **FXAA** (`render_fxaa_pass`): samples `ldr_target_` and writes the
  swapchain. **Precondition:** `render_composite_pass` must have run earlier
  in the same frame — it caches `cached_pipelines_`, which FXAA (and only
  FXAA) reads; otherwise the pass silently no-ops.

All fullscreen passes draw 3 vertices (`fullscreen.vert`), use
`LOADOP_DONT_CARE`, and return silently if their pipeline is unavailable.

#### Forward-shader consumption

`model.frag` / `model_instanced.frag` bind the AO texture at fragment
sampler slot 6, `terrain.frag` at slot 4, `grass.frag` at slot 2 (the
three sky-IBL samplers follow, so the cluster SSBO bases are 10/8/6).
UV = `SV_Position.xy × screenAOTexel`
(fields appended to each lighting cbuffer). Model-family draws gate the
sample on `screenAOEnabled` (1 only for OPAQUE meshes — MASK/BLEND are
absent from the depth prepass, so the AO texel at their screen position
describes the geometry BEHIND them; sampling it would stamp the
background's AO onto foliage). The sampled value is
`min(matAO, screenAO)` and feeds the `ao` parameter of
`ambientPBR`/`ambientIBL` — the single place the ambient AO factor lives. `SceneRenderer::bind_screen_ao(slot)`
binds the frame's AO texture (or the 1×1 white dummy when AO is off /
didn't run / for BLEND draws) on every model-family pipeline bind;
terrain/grass receive the binding + texel size as `render()` arguments.

#### SceneRenderer integration

Frame order (`docs/architecture.md` §"Pass order"): depth prepass → GTAO
chain (only when `ao_mode != Off` **and** the prepass ran; otherwise
`invalidate_history()`) → offscreen 3D scene (forward samples the AO
texture) → TAA → AutoExposure (compute; see its section below) → Bloom →
VolumetricFog → *late swapchain acquire* →
Composite (`bloom_strength` zeroed when bloom off; auto-exposure SSBO +
grading params plumbed from settings/environment) → optional FXAA → UI.
`resize()`/`set_half_res()` are called every HDR-path frame from the current
window size + `ao_mode`, and `set_clear_color(env_.fog_color_horizon)` keeps
uncovered pixels matching the atmosphere. If `PostProcessStack::init`
failed, the whole path collapses to a single direct-to-swapchain
`MainScene` pass (no prepass, no AO, no bloom, no fog, no FXAA, no
tonemapping stage).

#### Error handling / lifetime

- Both `init`s return bool (logged); `is_ready()` checks every texture +
  both samplers. `shutdown()` (also run by the destructors) releases
  samplers via the device and resets textures; `device_` must outlive the
  object until then.
- `GTAO::clear_full_res_texture` acquires its own one-off command buffer
  (init/resize happen outside the frame loop) and logs on failure.
- Render-pass methods silently return when prerequisites are missing
  (`!is_ready()`, null pipeline, null target) — an AO failure never crashes
  the frame, it just leaves AO white.
- History invalidation mirrors TAA: first frame, `resize`, `set_half_res`
  change, and any frame where the chain didn't run (AO off, prepass
  missing).

### Histogram auto exposure (`render::AutoExposure`)

`src/bracken/render/auto_exposure.hpp/.cpp` — the engine's first
compute-shader consumer. Two dispatches per frame over the TAA-resolved
(pre-bloom) HDR scene color; the composite pass reads the result as a
read-only fragment SSBO. No screen-size resources (the histogram dispatch is
sized purely by uniforms), hence no `resize()`.

Constants (header, mirrored by the shaders): `EXPOSURE_HISTOGRAM_BINS = 256`,
`EXPOSURE_MIN_LOG2_LUMA = −10`, `EXPOSURE_MAX_LOG2_LUMA = +16` (≈ 0.102 EV
per bin; bin 0 = below-range "black": counts toward the percentile total,
contributes no luminance), `DEFAULT_LOW_PERCENTILE = 0.60`,
`DEFAULT_HIGH_PERCENTILE = 0.05`, `DEFAULT_LOG2_TARGET_GRAY = −1.80`
(calibrated so the default playground converges to ≈ EV 0, i.e. auto ON
matches pre-auto brightness), `SNAP_FRAMES = 3`,
`DEFAULT_READBACK_INTERVAL = 1.0` s.

Public API:

```cpp
[[nodiscard]] bool init(gpu::GPUDevice&);   // creates the two ComputeStorage buffers + sampler
void shutdown();  bool is_ready() const;
void invalidate();                          // next SNAP_FRAMES render()s snap to the metered target
struct FrameParams { float dt; float speed_up = 1.5f, speed_down = 2.5f;
                     float min_ev = -4.0f, max_ev = 4.0f; };  // overwritten from GraphicsSettings
void render(SDL_GPUCommandBuffer*, gpu::PipelineRegistry&, gpu::GPUTexture& scene_color, const FrameParams&);
SDL_GPUBuffer* state_buffer() const;        // float4 {current EV, target EV, avg log2 luma, valid}
void set_percentiles(float low, float high); void set_log2_target_gray(float);
struct ExposureState { float current_ev, target_ev, avg_log2_luma; };
const std::optional<ExposureState>& latest_state() const;  // fenced debug readback, ≥1 frame stale
void set_readback_interval(float seconds);  // 0 = every render()
```

GPU resources: `histogram_` (256 × uint32, `GPUBuffer::Type::ComputeStorage`,
created zero-filled — the per-frame reset is the adapt pass zeroing each bin
after reading it) and `state_` (16 B float4, same type — compute read-write
plus `GRAPHICS_STORAGE_READ` so composite samples it directly). Both are
bound with `cycle = false` always (cycling would swap away the accumulated
state).

Algorithm (CPU reference functions in the header, pinned by
`auto_exposure.test.cpp`; the adapt shader reduces serially on thread 0 so
GPU and CPU stay operation-identical):

1. `exposure_histogram.comp` (16×16 threadgroups over the image): per-pixel
   Rec. 709 luminance → `exposure_bin_from_luma` → LDS `InterlockedAdd`,
   then one global atomic per **non-zero** LDS bin.
2. `exposure_adapt.comp` (single 256-thread workgroup, dispatch (1,1,1)):
   percentile-clipped weighted mean of the bins' log2 luminance
   (`exposure_target_ev`: cumulative-count window [low·total,
   (1−high)·total], straddling bins contribute their inside portion);
   `target_ev = clamp(log2TargetGray − avg, minEv, maxEv)`; adaptation
   `current += clamp(target − current, −speedDown·dt, +speedUp·dt)`
   (`exposure_adapt`), landing exactly on target within one step;
   `firstFrame ≠ 0` snaps. Writes `{current, target, avg, 1}`.

The first `SNAP_FRAMES` (3) metered frames after `init()`/`invalidate()`
snap rather than adapt — the first frames after init routinely meter a
half-warmed scene (pending uploads), and a bogus initial EV would otherwise
decay at wall-clock speed. `SceneRenderer` calls `invalidate()` on any frame
the passes cannot record (`auto_exposure` off, fallback path, no 3D
content), mirroring the TAA/GTAO history policy.

Debug readback: at most every `readback_interval_` seconds, `render()`
records a 16-byte `SDL_DownloadFromGPUBuffer` on its **own** command buffer
(submitted before the frame's, so it reads the previous frame's state) with
`SDL_SubmitGPUCommandBufferAndAcquireFence`; the fence is polled
non-blocking (`SDL_QueryGPUFence`) on subsequent calls. Consumers: the dev
overlay's "Adapted EV" readout and the smoke line's `auto_ev=`.

Known limitations: the bloom prefilter thresholds against the MANUAL
exposure only (the adapted EV lives on the GPU; `bloom_downsample.frag`
doesn't read the SSBO); metering is percentile-band, not center-weighted —
in open scenes the surviving 60–95% band is mostly sky, so the EV tracks sky
brightness far more than ground light (intended for day/night; use
`set_percentiles`/`set_log2_target_gray` for subject-locked exposure).

### Known limitations & footguns (summary)

- `ClusterParams` is defined once on the CPU (`light_cluster.hpp`) and
  mirrored by `clusters.hlsli` — keep the pair byte-identical.
- Light count caps are enforced silently (256) or with a single log per
  frame (128/cluster); there is no per-light diagnostics path.
- Cluster SSBOs keep stale contents on zero-light frames (guarded by zeroed
  params, not cleared buffers).
- The shadow atlas contains garbage when `shadow_mode == Off`; consumers
  must gate on shadows being enabled.
- `shadowEnabled` is a float-encoded mode (0/1/2), not a bool.
- `GTAOUniforms` tunables other than `radiusScale`/`strength`
  (`radius`, `falloffRange`, `thinOccluderCompensation`, `maxDistance`, the
  shader slice/step counts) are compile-time defaults — not exposed through
  `GraphicsSettings`.
- Screen AO is computed from the prepass depth, which covers OPAQUE and
  alpha-tested MASK geometry (masked prepass variants replicate the forward
  alpha test exactly, coverage boost included). BLEND transparents — and
  skinned MASK meshes, which have no masked prepass variant — neither
  occlude nor receive their own AO.
- GPU-side per-pass timings around these passes (`BRACKEN_GPU_TIMER` reports CPU record time only) are unavailable —
  SDL3 has no timestamp queries; use RenderDoc/Nsight.

---

## Model PBR Surface Shading

This chapter specifies the shading algorithm applied to every model draw in the main HDR pass — static (`model.vert.hlsl` + `model.frag.hlsl`), instanced (`model_instanced.vert.hlsl` + `model_instanced.frag.hlsl`), skinned (`skinned_model.vert.hlsl` + the shared `model.frag.hlsl`), and their DoubleSided/Blend pipeline variants — plus the shared shader libraries `pbr.hlsli` and `lighting.hlsli` and the CPU-side per-draw uniform/texture population in `src/bracken/scene/scene_renderer.cpp`. The uniform ABI mirrors live in `src/bracken/gpu/gpu_uniforms.hpp`, are layout-`static_assert`ed there, and are checked against the compiled shaders in `src/bracken/gpu/uniform_layout.test.cpp`. All lighting math is linear-space; output is linear HDR radiance into the `RGBA16F` scene target (tonemapping happens in the composite pass).

### File map and include graph

| File | Role |
|---|---|
| `src/bracken/shaders/pbr.hlsli` | BRDF library: GGX NDF, Smith visibility, Schlick Fresnel (+roughness-aware), Burley diffuse, `evaluatePBR`, `ambientPBR`, `ambientIBL` (sky-IBL ambient; `IBL_SPECULAR_MIPS = 7` must match `render::IBL::SPECULAR_MIPS`), `specularAARoughness` |
| `src/bracken/shaders/lighting.hlsli` | Shared helpers: luminance, exact sRGB↔linear, interleaved gradient noise, ACES curves, depth→view/world position reconstruction |
| `src/bracken/shaders/model.frag.hlsl` | Fragment stage for static + skinned model pipelines (shared) |
| `src/bracken/shaders/model_instanced.frag.hlsl` | Fragment stage for instanced model pipelines (per-instance tint/noFog variant) |
| `src/bracken/shaders/model.vert.hlsl` | Static model vertex stage |
| `src/bracken/shaders/model_instanced.vert.hlsl` | Instanced vertex stage (SSBO of `InstanceData`, `SV_InstanceID`) |
| `src/bracken/shaders/skinned_model.vert.hlsl` | Skinned vertex stage (4-bone LBS from a packed palette SSBO) |
| `src/bracken/scene/scene_renderer.cpp` | Per-draw uniform population (`make_model_lighting`, `apply_mesh_material`), fallback texture wiring (`bind_mesh_material_textures`), pipeline-variant selection, transparent re-routing |
| `src/bracken/gpu/gpu_uniforms.hpp` | `ModelTransformUniforms`, `ModelLightingUniforms`, `InstancedCameraUniforms`, `InstancedLightingUniforms`, `InstanceData` CPU mirrors |

Include chains (relevant to symbol availability):

- `model.frag.hlsl` and `model_instanced.frag.hlsl` include, in order: `shadow_common.hlsli` (which itself includes `lighting.hlsli` for `interleavedGradientNoise`), `fog.hlsli`, `lighting.hlsli`, `pbr.hlsli`, then — after `#define CLUSTER_SSBO_BINDING_BASE 15` (= the fragment sampler count; the include itself declares the 15th sampler, the local-shadow atlas, at slot 14) — `gi_sample.hlsli` and `cluster_lighting.hlsli`.
- Direct `#include "lighting.hlsli"` consumers: `model.frag.hlsl`, `model_instanced.frag.hlsl`, `terrain.frag.hlsl`, `grass.frag.hlsl`, `skybox.frag.hlsl`, `composite.frag.hlsl`, and `shadow_common.hlsli` (which propagates it to every shadow-receiving shader). Both `.hlsli` libraries are include-guarded (`PBR_HLSLI`, `LIGHTING_HLSLI`), so the double inclusion via `shadow_common.hlsli` is harmless.
- Direct `pbr.hlsli` consumer besides `clusters.hlsli`: none in the model family; `terrain.frag.hlsl` and `grass.frag.hlsl` also reach it via `cluster_lighting.hlsli`.

Documentation discrepancy: `src/bracken/shaders/README.md` describes `pbr.hlsli` as containing the "ACES tonemap"; the ACES curves (`acesNarkowicz`, `acesFitted`) actually live in `lighting.hlsli`.

### pbr.hlsli — BRDF library

All inputs linear; all direction vectors world-space unit vectors. `PI_PBR = 3.14159265358979323846` (guarded `#ifndef PI_PBR`).

#### ndfGGX — GGX/Trowbridge-Reitz normal distribution

```hlsl
float ndfGGX(float NdotH, float roughness)
```

`a = roughness²`, `a2 = a²`, `d = NdotH²·(a2−1)+1`; returns `a2 / max(π·d², 1e-7)`. Uses the Disney/glTF squared-perceptual-roughness parameterization. The `1e-7` clamp prevents division by zero at `roughness = 0`, `NdotH = 1`.

#### visibilitySmithGGX — height-correlated Smith visibility

```hlsl
float visibilitySmithGGX(float NdotV, float NdotL, float roughness)
```

Returns the *pre-divided* visibility term `V = G2 / (4·NdotV·NdotL)` — callers multiply by `D` and `F` only. Implementation: `ggxV = NdotL·√(NdotV²·(1−a2)+a2)`, `ggxL = NdotV·√(NdotL²·(1−a2)+a2)`, result `0.5 / max(ggxV+ggxL, 1e-5)`.

#### fresnelSchlick / fresnelSchlickRoughness

```hlsl
float3 fresnelSchlick(float cosTheta, float3 F0)                      // F0 + (1-F0)·(1-cosθ)^5
float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
```

The roughness-aware variant clamps the grazing reflectance to `max((1−roughness).xxx, F0)` — used by `ambientPBR` so rough surfaces don't get a bright Fresnel rim from the analytic environment term.

#### diffuseBurley — Disney diffuse

```hlsl
float3 diffuseBurley(float3 baseColor, float roughness, float NdotV, float NdotL, float LdotH)
```

`fd90 = 0.5 + 2·LdotH²·roughness`; light and view scatter terms `1+(fd90−1)·(1−cosθ)^5`; returns `baseColor · lightScatter · viewScatter / π`. Energy-conserving retro-reflective wrap that darkens grazing angles for smooth surfaces.

#### evaluatePBR — single directional-light contribution

```hlsl
float3 evaluatePBR(float3 normal, float3 view, float3 light, float3 lightColor,
                   float3 baseColor, float metallic, float roughness, float ao)
```

Contract (from the header comment plus code):

- `view` points surface→camera, `light` points surface→light. `lightColor` is linear radiance at the surface, *already shadow-modulated by the caller if desired*.
- `NdotV = max(dot(N,V), 1e-4)`; `NdotL = max(dot(N,L), 0)` with a **hard early-out returning `float3(0,0,0)` when `NdotL <= 0`** — no wrap lighting, back faces receive nothing.
- The half-vector is computed *after* the early-out (deliberate: `V == −L` is the only `normalize(0)` input and the rejected path must never form it), with a residual guard: `H = (V+L)/max(length(V+L), 1e-5)`.
- `F0 = lerp(float3(0.04,0.04,0.04), baseColor, metallic)` — fixed 4% dielectric reflectance, no specular/IOR extension support.
- Specular: `D·V·F` with `F = fresnelSchlick(LdotH, F0)`.
- Diffuse energy split: `kd = (1−F)·(1−metallic)` — **metals have no diffuse**. Diffuse = `kd · diffuseBurley(...) · ao`; **`ao` multiplies the diffuse term only**, never the specular.
- Return: `(diff + spec) · lightColor · NdotL`.

The clustered-lighting wrappers `evaluatePBRPoint` / `evaluatePBRSpot` (`src/bracken/shaders/clusters.hlsli`) fold attenuation (and cone factor) into `lightColor` and call this exact function, so punctual lights share the identical BRDF.

#### ambientPBR — hemisphere "IBL" ambient

```hlsl
float3 ambientPBR(float3 N, float3 V, float3 baseColor, float metallic, float roughness,
                  float ao, float3 skyColor, float3 groundColor)
```

- Diffuse irradiance: hemisphere gradient `lerp(groundColor, skyColor, N.y·0.5+0.5)` (Y-up), split by `kd = (1−F)·(1−metallic)` with `F = fresnelSchlickRoughness(NdotV, F0, roughness)`. No Burley here — irradiance is treated as already cosine-integrated. `NdotV` clamped to `1e-4`; `F0` as in `evaluatePBR`.
- Specular: Karis mobile split-sum fit (UE4 `EnvBRDFApprox`) with exact constants `c0 = (−1.0, −0.0275, −0.572, 0.022)`, `c1 = (1.0, 0.0425, 1.04, −0.04)`; `r = roughness·c0 + c1`; `a004 = min(r.x², exp2(−9.28·NdotV))·r.x + r.y`; `AB = (−1.04, 1.04)·a004 + r.zw`. Environment color is the same hemisphere gradient sampled along the reflection vector `R = reflect(−V, N)`: `envColor = lerp(groundColor, skyColor, R.y·0.5+0.5)`. Specular = `envColor·(F0·AB.x + AB.y)`.
- Return: `(diffuse + specular) · ao` — **AO multiplies the entire ambient**, diffuse and specular alike (unlike `evaluatePBR`).

#### ambientIBL — sky image-based ambient

```hlsl
float3 ambientIBL(float3 N, float3 Ngeo, float3 V, float3 baseColor, float metallic,
                  float roughness, float ao, float lift,
                  TextureCube<float4> irradianceMap, SamplerState irradianceSampler,
                  TextureCube<float4> specularMap, SamplerState specularSampler,
                  Texture2D<float4> brdfLut, SamplerState brdfLutSampler)
```

Replaces `ambientPBR`'s hemisphere terms when the scene shader's `iblEnabled`
uniform flag is on (`GraphicsSettings::ibl_enabled` + `render::IBL` ready):

- Diffuse: `kd · baseColor · irradianceMap.SampleLevel(N, 0)` with the same
  `kd = (1−F)(1−metallic)` roughness-aware Fresnel split as `ambientPBR`. The
  irradiance cube stores `(1/π)·∫L·cosθ dω` (a uniform sky convolves to
  itself), so no extra `1/π`.
- Specular: `specularMap.SampleLevel(R, roughness · (IBL_SPECULAR_MIPS − 1))`
  × the split-sum LUT `(F0·A + B)` sampled at `(NdotV, roughness)`, × a
  Frostbite horizon-occlusion guard `saturate(1 + dot(R, Ngeo))²` (a no-op
  when `N == Ngeo`; fades reflection lobes that normal mapping pushed below
  the geometric surface).
- Return: `(diffuse + specular) · ao · lift` — the same combined
  (material ∧ screen) AO and shadow-driven `ambientLift` multiply the whole
  ambient exactly as on the hemisphere path.

`IBL_SPECULAR_MIPS = 7.0` is a compile-time contract with
`render::IBL::SPECULAR_MIPS` (static_asserted CPU-side).

#### specularAARoughness — Kaplanyan-style specular anti-aliasing

```hlsl
float specularAARoughness(float3 N, float roughness) // fragment-shader only (ddx/ddy)
```

`variance = 0.25·(dot(ddx(N),ddx(N)) + dot(ddy(N),ddy(N)))`; `kernel = min(variance, 0.18)`; returns `clamp(sqrt(roughness² + kernel), roughness, 1.0)`. Widens perceptual roughness by the screen-space normal variance, clamped so the widening never exceeds the `0.18` kernel and the result never goes below the authored roughness or above 1. Applied unconditionally by both model fragment shaders after MR-texture modulation and before all BRDF evaluation (direct, ambient, and clustered lights all see the widened roughness).

### lighting.hlsli — shared helper library

Constants and functions (all math linear unless documented):

- `PI_ = 3.14159265358979323846` (`static const float`).
- `float luminance(float3 c)` — Rec. 709 weights `(0.2126, 0.7152, 0.0722)`.
- `float3 linearToSrgb(float3 c)` — exact piecewise transfer function (threshold `0.0031308`, linear segment slope `12.92`, gamma segment inverse of `((c+0.055)/1.055)^2.4` with a `max(c, 1e-5)` guard). "Used sparingly where sampler gamma doesn't apply." (An unused `srgbToLinear` counterpart was removed.)
- `float interleavedGradientNoise(float2 screenPos)` — Jimenez 2014: `frac(52.9829189 · frac(dot(screenPos, (0.06711056, 0.00583715))))`.
- `float3 acesNarkowicz(float3 x)` — 3-coefficient rational fit, `a=2.51, b=0.03, c=2.43, d=0.59, e=0.14`, saturated.
- `float3 acesFitted(float3 x)` — Hill fitted ACES: input/output 3×3 matrices (`ACESInputMat`, `ACESOutputMat` with the exact coefficients in the file) around the RRT+ODT rational `a = x·(x+0.0245786) − 0.000090537`, `b = x·(0.983729·x+0.4329510)+0.238081`; saturated. Comment: higher quality, desaturates highlights correctly.
- `float3 viewPosFromDepth(float2 uv, float depth, float4x4 invProjection)` / `float3 worldPosFromDepth(float2 uv, float depth, float4x4 invViewProjection)` — clip-space reconstruction assuming Vulkan-style `[0,1]` z with flipped Y (`clip.y = −clip.y`), perspective divide by `w`.

Consumption as of this source tree: `model.frag` / `model_instanced.frag` include it (and `pbr.hlsli` directly); `composite.frag.hlsl` uses `luminance`, `acesFitted`, `acesNarkowicz`, `linearToSrgb`, `interleavedGradientNoise`; `skybox.frag.hlsl` uses `interleavedGradientNoise`; `shadow_common.hlsli` uses `interleavedGradientNoise` (filter rotation + cascade-blend dither); `gtao.frag.hlsl` includes it for `viewPosFromDepth` + `interleavedGradientNoise`; `volumetric_fog.frag.hlsl` includes it as well.

### Vertex stages — interpolant contract

All three vertex shaders read `TransformUniforms`/`CameraUniforms` from **set 1, binding 0** (SDL3 GPU vertex-uniform set) and feed the fragment stage the same core interpolants. Vertex input locations match `get_vertex3d_attributes()` / `get_skinned_vertex_attributes()` in `src/bracken/gpu/gpu_types.hpp`:

| Location | Semantic | Format | Field |
|---|---|---|---|
| 0 | POSITION | FLOAT3 | `Vertex3D::position` |
| 1 | NORMAL | FLOAT3 | `Vertex3D::normal` |
| 2 | TEXCOORD0 | FLOAT2 | `Vertex3D::texcoord` |
| 3 | COLOR0 | FLOAT4 | `Vertex3D::color` (linear) |
| 4 | TANGENT | FLOAT4 | `Vertex3D::tangent` (xyz tangent, w bitangent sign) |
| 5 (skinned only) | BLENDINDICES | UBYTE4 | `SkinnedVertex::joints` (read as `uint4`) |
| 6 (skinned only) | BLENDWEIGHT | FLOAT4 | `SkinnedVertex::weights` |

#### model.vert.hlsl (static)

`TransformUniforms` (mirrored by `gpu::ModelTransformUniforms`, 288 bytes): `model`, `view`, `projection` (mat4 each), `cameraPos` + pad, `normalMatrix` (mat4, **CPU-precomputed inverse-transpose of model** — `compute_normal_matrix` in the renderer), `paletteOffset` (uint at offset 272, unused here; layout shared with `skinned_model.vert.hlsl`), 12 bytes pad.

Outputs:

- `SV_Position` = `projection · view · model · pos`.
- `TEXCOORD0 fragPos` — world position (`(model·pos).xyz`).
- `TEXCOORD1 normal` — `normalize((float3x3)normalMatrix · normal)`.
- `TEXCOORD2 texCoord`, `TEXCOORD3 vertexColor` — passthrough.
- `TEXCOORD4 viewDepth` — `−viewPos.z` (positive distance along camera forward; consumed by cascade selection and cluster z-binning).
- `TEXCOORD5 tangent` — `float4(normalize((float3x3)model · tangent.xyz), tangent.w)`. The tangent deliberately uses the model matrix, not the normal matrix ("rigid-body transform preserves direction"); the bitangent sign `w` passes through untransformed.

#### model_instanced.vert.hlsl

Camera data comes from `CameraUniforms` (mirror: `gpu::InstancedCameraUniforms`; `view`, `projection`, `cameraPos`+pad — pushed once per pass). Per-instance transforms come from `StructuredBuffer<InstanceData> instances` at **vertex set 0, binding 0**, indexed by `SV_InstanceID` (SDL3 GPU: `SDL_DrawGPUIndexedPrimitives`'s `first_instance` offsets the ID). `InstanceData` (HLSL and `gpu::InstanceData` mirror, 160-byte stride): `model` (mat4), `normalMatrix` (mat4), `tint` (float4), `noFog` (float) + 3 pad floats.

Extra outputs beyond the static set: `TEXCOORD5 instanceTint` (= `inst.tint`), `TEXCOORD6 noFog` (= `inst.noFog`), and `tangent` moves to `TEXCOORD7`. Transform math is identical to the static shader but sourced per instance.

#### skinned_model.vert.hlsl

Same `TransformUniforms` layout as the static shader; `paletteOffset` is live here — it is the start index (in matrices) of this draw's palette inside the frame-global `StructuredBuffer<float4x4> bonePalettes` at **vertex set 0, binding 0** (all skinned draws' palettes tightly packed, uploaded once per frame). Skinning is 4-bone linear blend:

```hlsl
float4x4 skinMatrix = weights.x * bonePalettes[paletteOffset + joints.x] + ... (y,z,w);
```

Position, normal (`w = 0`), and tangent (`w = 0`) are all transformed by the same blended `skinMatrix`, then position by `model`, normal by `(float3x3)normalMatrix`, tangent by `(float3x3)model`; bitangent sign is preserved from the source asset. There is no dedicated inverse-transpose for the skin matrix — non-uniformly scaled bones would skew normals (linear-blend approximation). Output interpolants are identical to `model.vert.hlsl` (fragment stage is shared `model.frag.hlsl`).

### Fragment stages — material evaluation algorithm

`model.frag.hlsl` (static + skinned) and `model_instanced.frag.hlsl` are near-identical; the differences are called out inline below. Resource bindings (SDL3 GPU fragment sets — sampled textures in set 2, uniforms in set 3):

| Binding | Resource |
|---|---|
| set 3, b0 | `LightingUniforms` (mirrors `gpu::ModelLightingUniforms` / `gpu::InstancedLightingUniforms`) |
| set 3, b1 | `ShadowUniforms` (mirrors `gpu::ShadowDataUniforms` — see lighting/shadows chapter) |
| set 3, b2 | `ClusterParamsCB` (declared by `cluster_lighting.hlsli`; mirrors `render::lighting::ClusterParams`) |
| set 2, sampler 0 | `baseColorTexture` (combined image sampler; `RGBA8_SRGB` view — sampler linearizes) |
| set 2, sampler 1 | `shadowAtlas` + `SamplerComparisonState` (hardware PCF) |
| set 2, sampler 2 | `shadowAtlasRaw` + plain sampler (PCSS blocker search) |
| set 2, sampler 3 | `normalMapTexture` (`RGBA8` UNORM) |
| set 2, sampler 4 | `metallicRoughnessTexture` (`RGBA8` UNORM) |
| set 2, sampler 5 | `emissiveTexture` (`RGBA8_SRGB`) |
| set 2, sampler 6 | `screenAOTexture` (full-res RGBA8 GTAO: AO + octahedral bent normal + cone aperture; 1×1 white dummy when AO off / BLEND draws) |
| set 2, samplers 7–9 | `iblIrradianceMap` / `iblSpecularMap` (cubes) + `iblBrdfLut` (typed dummies when IBL off) |
| set 2, sampler 10 | `rtShadowTexture` (R8 ray-traced sun visibility, 1 = lit, at the traced pass's OWN resolution — sample with a normalized UV; 1×1 white dummy when ray tracing is off, and read only when `rtShadowEnabled`) |
| set 2, samplers 11–13 | probe-field GI: `giIrradianceAtlas` + `giVisibilityAtlas` (linear clamp) and `giProbeMeta` (point clamp), read by `gi_sample.hlsli`; 1×1 black dummies and `giEnabled == 0` when GI is off |
| set 2, sampler 14 | `localShadowAtlas` + `SamplerComparisonState` (declared by `cluster_lighting.hlsli` at `CLUSTER_SSBO_BINDING_BASE − 1`; CSM atlas as filler when local shadows are unavailable) |
| set 2, storage buffers 0–3 | cluster light data / offsets / indices / local-shadow slots (`CLUSTER_SSBO_BINDING_BASE = 15` for `model.frag` / `model_instanced.frag`, i.e. immediately after their 15 samplers per SDL GPU's set-2 layout: sampled textures, then storage textures, then storage buffers. Slot 10 is the ray-traced sun-visibility mask, 11–13 the probe field, 14 the local-shadow atlas) |

Per-fragment algorithm, in order (`PSMain`):

1. **Vectors** — `N_geo = normalize(input.normal)`; `V = normalize(cameraPos − fragPos)`; `L = normalize(−lightDir)` (`lightDir` is the direction light travels, per `EnvironmentState::sun_dir`).
2. **Normal mapping** — gated on `hasNormalMap == 1`: sample `.xyz·2−1`, scale `.xy` by `normalScale`, build TBN from `T_ws = normalize(tangent.xyz)`, `B_ws = cross(N_geo, T_ws)·tangent.w`, and set `N = normalize(T·x + B·y + N_geo·z)`. Otherwise `N = N_geo`. The CPU binds a 1×1 flat-normal fallback `(128,128,255)` (decodes to `(0,0,1)`) so the slot is always valid; note the renderer comment says this serves a "branch-free path", but the shader as written does branch on `hasNormalMap` — the fallback texture is a safety/no-op either way.
3. **Base color** — `hasTexture == 1`: `baseColorTexture.Sample(...) × baseColorFactor × tintColor` (static/skinned) or `× baseColorFactor × input.instanceTint` (instanced); else `vertexColor × baseColorFactor × tintColor` / `× instanceTint`. `baseColorFactor` applies on **both** paths (glTF semantics: it multiplies `COLOR_0`, which the loader bakes white when absent).
4. **Alpha test** — for MASK draws (`alphaCutoff > 0`) the shader first applies a mip-scaled coverage boost, `baseColor.a *= 1 + max(mip, 0) * 0.35` with `mip = baseColorTexture.CalculateLevelOfDetailUnclamped(...)`, then `if (baseColor.a < alphaCutoff) discard;`. Box-filtered mips average alpha downward, so minified alpha-tested foliage would otherwise erode into sub-pixel holes that AA resolves into a see-through wash (classic alpha-coverage fix). The CPU sends `alphaCutoff > 0` only for glTF MASK materials (`apply_mesh_material`: `u.alphaCutoff = (mat.alpha_mode == AlphaMode::Mask) ? mat.alpha_cutoff : 0.0f`), so OPAQUE/BLEND fragments never take the discard or the boost.
5. **Metallic/roughness/occlusion** — start from `metallicFactor` / `roughnessFactor`, `matAO = 1`. If `hasMRMap > 0.5`: `roughness *= mr.g; metallic *= mr.b;` and, only if `hasPackedOcclusion > 0.5`, `matAO = lerp(1.0, mr.r, occlusionStrength)` (glTF ORM packing: R = occlusion). Then `roughness = specularAARoughness(N, roughness)` unconditionally.
6. **Shadow** — `shadow = calcShadow(fragPos, N_geo, L, viewDepth, SV_Position.xy)` (`src/bracken/shaders/shadow_common.hlsli`; returns raw [0,1] visibility, geometric normal deliberately used). 
7. **Direct** — `direct = evaluatePBR(N, V, L, lightColor·sunIntensity·shadow, baseColor.rgb, metallic, roughness, /*ao=*/1.0)`. Material AO is *not* applied to direct light.
8. **Ambient** — `ambientLift = lerp(ambientStrength, 1.0, shadow)`; with `iblEnabled > 0.5`: `ambient = ambientIBL(N, N_geo, V, baseColor.rgb, metallic, roughness, aoCombined, ambientLift, …IBL samplers…)`; else the hemisphere fallback `ambientPBR(N, V, baseColor.rgb, metallic, roughness, aoCombined, ambientSky·ambientLift, ambientGround·ambientLift)`. The lift dims ambient inside shadow to avoid a flat fill-light look; `ambientStrength` comes from `GraphicsSettings::ambient_strength` (default 0.45).
9. **Clustered lights** — gated on `clusterParams.gridDim.w > 0u` (total light count): `screen_uv = SV_Position.xy · screenSize.zw`; `result += accumulateClusterLights(fragPos, N, V, baseColor.rgb, metallic, roughness, /*ao=*/1.0, viewDepth, screen_uv)` (`src/bracken/shaders/cluster_lighting.hlsli`; same BRDF, per-cluster loop clamped to `maxPerCluster.x`).
10. **Emissive** — `emissive = emissiveFactor`, `*= emissiveTexture.Sample(...).rgb` when `hasEmissiveMap > 0.5`; added to the result.
11. **Fog** — static/skinned: `if (fogEnabled == 1)`; instanced: `if (fogEnabled == 1 && input.noFog < 0.5)` (per-instance exemption). Applies `applyAtmosphericFog(result, fragPos, cameraPos, lightDir, fogColor, fogSunColor, fogStart, fogEnd, fogHeightFalloff)` (`src/bracken/shaders/fog.hlsli` — distance factor with exp shaping, analytic height-integral attenuation, sun-directed color blend).
12. **Output** — `float4(result, baseColor.a)`. Alpha is meaningful only for the Blend pipeline variants; opaque variants have no blend state and ignore it.

### CPU-side uniform and texture population (SceneRenderer)

#### make_model_lighting — per-frame/per-view fields

```cpp
gpu::ModelLightingUniforms SceneRenderer::make_model_lighting(const CameraState& camera, bool fog_active) const;
```

Populates from `EnvironmentState env_` (accessor `SceneRenderer::environment()`) and `GraphicsSettings`: `lightDir = env_.sun_dir`, `lightColor = env_.sun_color`, `ambientSky/ambientGround = env_.ambient_sky/ambient_ground`, `tintColor = vec4(1)`, fog fields from `env_.fog_color_horizon/fog_color_sun/fog_start/fog_end/fog_height_falloff`, `fogEnabled = fog_active`, `cameraPos = camera.position`, `ambientStrength = gfx.ambient_strength` (default 0.45), `sunIntensity = gfx.sun_intensity` (default 3.0). `frame_fog_active_` is latched once per frame from `gfx.fog_enabled`; per-draw `fog_active` is `!has(cmd.flags, DrawFlags::NoFog) && frame_fog_active_` for static, skinned, and transparent draws, and global-with-per-instance-`noFog` for instanced.

#### apply_mesh_material — per-mesh material fields

Template over both uniform structs (identical field names):

```cpp
template <typename U> static inline void apply_mesh_material(U& u, const Mesh& mesh);
```

Sets `hasTexture = (mesh.has_texture && mesh.texture) ? 1 : 0` (both the flag *and* an uploaded `GPUTexture` required), `hasNormalMap = mat.normal_texture ? 1 : 0`, `normalScale`, `baseColorFactor`, `metallicFactor`, `roughnessFactor`, `occlusionStrength`, `hasMRMap = mat.metallic_roughness_texture ? 1.0f : 0.0f`, `emissiveFactor`, `hasEmissiveMap`, `hasPackedOcclusion = mat.has_packed_occlusion ? 1.0f : 0.0f`, and the MASK-only `alphaCutoff` gate described above. Note the flag types are heterogeneous by design: `hasTexture`/`hasNormalMap` are `int` (shader compares `== 1`), the rest are `float` (shader compares `> 0.5`).

#### Redundant-push elision

Each draw path composes `make_model_lighting` + tint + `apply_mesh_material`, then pushes fragment uniforms only when the 208-byte struct differs from the last pushed one (`std::memcmp` against `last_model_lighting_` guarded by `model_lighting_valid_`; the instanced path keeps its own `last_pushed`/`lighting_pushed` pair). `reset_model_bind_state()` invalidates the cache and clears `last_frag_tex_` after every pipeline bind, because pushed uniform data survives pipeline binds but the tracking is conservatively reset.

#### bind_mesh_material_textures — fallback texture wiring

```cpp
void SceneRenderer::bind_mesh_material_textures(const Mesh& mesh);
void SceneRenderer::bind_material_texture(uint32_t slot, SDL_GPUTexture* texture); // dedupes via last_frag_tex_[slot]
```

Slot 0 is *always* bound (the shader declares `baseColorTexture` regardless of the `hasTexture` branch; an unbound or stale-incompatible descriptor is undefined behavior): mesh texture, else `dummy_white_texture_` (1×1 white `0xFFFFFFFF`, `RGBA8`). Slot 3: `mat.normal_texture` else `default_normal_texture_` (1×1 `(128,128,255,255)` flat tangent-space normal). Slot 4: `mat.metallic_roughness_texture` else `dummy_white_texture_` (white is safe *and* neutral: sampling is gated by `hasMRMap`, and white would multiply factors by 1 anyway). Slot 5: `mat.emissive_texture` else `dummy_black_texture_` (1×1 `0xFF000000`). All three fallbacks are created once at boot in `SceneRenderer::init_pipelines` (`scene/scene_renderer.cpp`); slots 1–2 (shadow atlas ×2) are bound separately by `bind_shadow_data` (`scene/scene_renderer_draw.cpp`). `bind_material_texture` skips rebinds when `last_frag_tex_[slot]` already holds the texture and silently no-ops when the texture or `default_sampler_` is null. All material slots share `default_sampler_` (linear min/mag/mip, repeat addressing, anisotropy up to 16, `max_lod = 1000`).

#### Pipeline-variant selection and draw routing

Model-family pipeline types (`src/bracken/gpu/pipeline_registry.hpp`): `Model`, `ModelDoubleSided`, `ModelBlend`, `ModelBlendDoubleSided`, `SkinnedModel`, `SkinnedModelDoubleSided`, `SkinnedModelBlend`, `SkinnedModelBlendDoubleSided`, `InstancedModel`, `InstancedModelDoubleSided`. There is deliberately **no instanced Blend variant**.

- Static draws: `bind_model_pipeline(double_sided)` binds the pipeline, resets bind state, then `bind_shadow_data(..., 1)` and `bind_cluster_lights(..., 2)`. Per mesh, `double_sided` mismatches trigger a variant re-bind mid-model.
- Skinned draws: `bind_skinned_pipeline(double_sided)` additionally binds the bone-palette SSBO to vertex set 0 binding 0; returns false (draw skipped) when `bone_palette_buffer_` doesn't exist.
- Instanced draws (`render_instanced_models`): one pipeline bind, camera uniforms + instance SSBO + shadow + cluster binds once; per-mesh double-sided switches re-establish all pass state. One `SDL_DrawGPUIndexedPrimitives(pass, index_count, instance_count, 0, 0, base_instance)` per (model, mesh); `base_instance` follows ascending-`ModelHandle` packing order, which must match `upload_instance_buffers`' packing.
- **Blend re-routing**: any mesh with `alpha_mode == AlphaMode::Blend` encountered in the static, skinned, or instanced walk is queued via `queue_transparent_mesh` (instanced: one queue entry per instance, losing instancing) and drawn by `render_transparent_models` after the skybox — sorted back-to-front on squared distance from camera to the *model's* world bounding-sphere center, depth-test on / depth-write off, using the (Skinned)ModelBlend(DoubleSided) variants. Skinned transparent draws carry their `palette_offset` (sentinel `INVALID_PALETTE_OFFSET = 0xFFFFFFFFu` marks static) and are skipped if the palette buffer is missing. No OIT, no intra-mesh sorting (documented limitation).
- Meshes with `shadow_only = true` are skipped by all main-pass walks (shadow-pass-only proxies).

#### Bone palettes (skinned vertex-stage input)

The skinned resolve walks each `SkinnedDrawCommand`: invalid palette ids and unresolvable models are skipped; `const BonePalette& rec = frame.palettes()[cmd.palette.value]`; drawn joints clamp to `min(model joints, rec.joint_count)`; the palette GPU offset is `rec.matrix_offset` (prefix-summed at seal — no renderer-side dedupe or running total; palette sharing is by-id in the builder). Budget `MAX_BONE_PALETTES = 1024` is enforced at build time in `alloc_bone_palette` (`scene/frame_packet.hpp`). Palette upload memcpys each `BonePalette` record's matrices from packet memory straight into the mapped transfer buffer, in `palettes()` order, into a storage buffer grown geometrically with a 64 KB floor (`frame.palette_matrix_total()` matrices total). No lifetime contract — the matrices live in the packet.

### Uniform ABI — exact layouts

All structs `alignas(16)`; layouts locked by `static_assert` in `gpu_uniforms.hpp` itself (`BRACKEN_HLSL_CBUFFER`/`BRACKEN_HLSL_FIELD`) and cross-checked against the compiled SPIR-V by `src/bracken/gpu/uniform_layout.test.cpp`. Every `vec3` is followed by a scalar so no field straddles a 16-byte HLSL register — a rule the `BRACKEN_HLSL_FIELD` macro now enforces directly.

#### gpu::ModelLightingUniforms — 224 bytes (model.frag.hlsl `LightingUniforms`, set 3 b0)

| Offset | Field | Type | Default / source |
|---|---|---|---|
| 0 | `lightDir` | vec3 | `env_.sun_dir` (direction light travels) |
| 16 | `lightColor` | vec3 | `env_.sun_color` |
| 32 | `ambientSky` | vec3 | `env_.ambient_sky` |
| 48 | `ambientGround` | vec3 | `env_.ambient_ground` |
| 64 | `tintColor` | vec4 | command tint (default 1,1,1,1) |
| 80 | `baseColorFactor` | vec4 | material (default 1,1,1,1) |
| 96 | `fogColor` + `fogStart`(108) | vec3+f | `env_.fog_color_horizon`, `env_.fog_start` |
| 112 | `fogSunColor` + `fogEnd`(124) | vec3+f | `env_.fog_color_sun`, `env_.fog_end` |
| 128 | `emissiveFactor` + `fogHeightFalloff`(140) | vec3+f | material, `env_.fog_height_falloff` |
| 144 | `cameraPos` + `normalScale`(156) | vec3+f | camera, material (default 1.0) |
| 160 | `hasTexture` | int | mesh flag |
| 164 | `fogEnabled` | int | per-draw fog gate |
| 168 | `hasNormalMap` | int | material |
| 172 | `ambientStrength` | float | `GraphicsSettings` (0.45; struct default 0.5 is always overwritten) |
| 176 | `sunIntensity` | float | `GraphicsSettings` (3.0) |
| 180 | `metallicFactor` | float | material (0.0) |
| 184 | `roughnessFactor` | float | material (0.85) |
| 188 | `occlusionStrength` | float | material (1.0) |
| 192 | `hasMRMap` | float | material |
| 196 | `hasEmissiveMap` | float | material |
| 200 | `hasPackedOcclusion` | float | material |
| 204 | `alphaCutoff` | float | MASK-gated (0 disables) |
| 208 | `screenAOTexelX` + `screenAOTexelY`(212) | float×2 | 1 / render-target size (SV_Position → screen-AO UV) |
| 216 | `iblEnabled` | float | 1 = sky-IBL ambient (samplers 7–9), 0 = hemisphere fallback |
| 220 | `screenAOEnabled` | float | 1 = OPAQUE/MASK (both in the prepass; modulate ambient by screen AO); 0 = BLEND (no prepass depth — the AO texel under it belongs to the geometry behind) |

#### gpu::InstancedLightingUniforms — 208 bytes (model_instanced.frag.hlsl, set 3 b0)

Identical to `ModelLightingUniforms` minus the `tintColor` vec4 (tint is per-instance); every field from `baseColorFactor` (offset 64) onward shifts down 16 bytes (`alphaCutoff` at 188, `screenAOTexelX/Y` at 192/196, `iblEnabled` at 200).

#### gpu::ModelTransformUniforms (model/skinned vertex, set 1 b0)

288 bytes (size-tested): `model`(0), `view`(64), `projection`(128) mat4s; `cameraPos`(192)+pad; `normalMatrix`(208) mat4 (CPU inverse-transpose); `paletteOffset`(272) uint32 (read only by the skinned shader); 12-byte pad. `gpu::InstancedCameraUniforms` (instanced vertex, set 1 b0): `view`, `projection`, `cameraPos`+pad — pushed once per pass and shared with `particle.vert.hlsl`.

#### gpu::InstanceData (vertex set 0 binding 0 SSBO)

`model` mat4 (0), `normalMatrix` mat4 (64), `tint` vec4 (128), `noFog` float (144) + 3 pad floats, `prevModel` mat4 (160) = 224-byte stride, matching the HLSL `InstanceData` struct field-for-field (shared by the forward and depth-prepass instanced shaders). Populated in `build_instance_batches` (`normalMatrix = compute_normal_matrix(cmd.transform)`, `tint = cmd.tint`, `noFog = has(cmd.flags, DrawFlags::NoFog) ? 1 : 0`).

### Threading, lifetime, and error handling

- The entire path is main-thread-only, inside `SceneRenderer::render_frame` (see the scene chapter for the frame contract). No API here is thread-safe.
- No bone-matrix lifetime contract: palettes are packet memory (`FrameBuilder::alloc_bone_palette`; see above).
- Silent-skip behaviors: draws with a null model handle, un-uploaded meshes (debug-asserted then skipped), missing vertex/index buffers, or zero indices are skipped; skinned commands with an invalid palette id or an unresolvable model/skeleton are excluded at resolve (over-budget palettes already failed visibly at `alloc_bone_palette`); transparent skinned draws with no palette buffer are skipped; pipeline-fetch failures skip the mesh/draw. Buffer-creation failures log to stderr and zero the cached capacity.
- Texture-format contract (from `src/bracken/model_loader.cpp` `upload_to_gpu`): base color `RGBA8_SRGB` + mips; normal map `RGBA8` UNORM + mips; metallic-roughness `RGBA8` UNORM + mips; emissive `RGBA8_SRGB` + mips. **Separate (non-ORM) occlusion images are warned about and dropped at load** — only R-channel-of-MR (ORM-packed) occlusion reaches the shader.

### Performance characteristics

- Fragment uniform pushes are deduped by 224/208-byte `memcmp` per mesh; texture binds deduped per slot; both caches reset on every pipeline bind. Meshes sharing materials inside one pipeline variant cost one push + zero rebinds.
- Sun shadowing costs up to ~20 atlas taps per fragment (PCSS path); `calcShadow` collapses to a single tap for back-facing fragments because `evaluatePBR` rejects them anyway and the result only feeds `ambientLift` (comment in `shadow_common.hlsli`).
- `specularAARoughness` is two derivative instructions + arithmetic, always on.
- The cluster-light loop is bounded by `min(count, maxPerCluster.x)` (`MAX_LIGHTS_PER_CLUSTER = 128` CPU-side) and skipped entirely when the frame has zero lights (`gridDim.w == 0`).
- Instanced double-sided/single-sided interleaving re-binds the pipeline and re-establishes vertex SSBO + shadow + cluster state per switch — group materials to avoid ping-ponging.

### Known limitations and stubs

- The hemisphere `ambientPBR` (two-color gradient + analytic env-BRDF) survives only as the `ibl_enabled = false` fallback; the default ambient path is the prefiltered-sky IBL (`ambientIBL`, `render::IBL`). Metals mirror the *procedural sky only* — there are no scene reflections (no SSR/probes); indoor or occluded areas still see the open sky.
- Fixed `F0 = 0.04` for dielectrics — no `KHR_materials_specular`/IOR support.
- Material AO: ambient-only, ORM-packed-only; standalone occlusion textures silently dropped at upload.
- BLEND: back-to-front per-draw sort on model bounding-sphere centers; no OIT, no per-triangle sorting; instanced BLEND devolves to per-instance draws.
- Skinned normals/tangents use the linear-blended skin matrix (no per-bone inverse-transpose).
- README/`scene_renderer.cpp` comment discrepancies noted above (ACES attributed to `pbr.hlsli`; "branch-free" normal-map path that actually branches).

---

## Water (`src/bracken/render/water_renderer.hpp`, `.cpp`)

Surface renderer + compute wave simulation. Owned by `SceneRenderer`
(`water()` accessor); inert until `set_region(WaterRegion)`.

**API**: `init(GPUDevice&, PipelineRegistry&)`; `set_region(const WaterRegion&)`
(builds the surface grid; normalizes `flow`; defaults `sim_center` to the
region centre); `active()`; `add_ripple(world_pos, radius, strength)` (queued;
ignored outside the sim window); `set_environment(const EnvironmentState&)`;
`simulate(SDL_GPUCommandBuffer*, float dt)` (compute; must be outside render
passes); `render(pass, cmd, view_proj, projection, camera_pos, time,
screen_w, screen_h, scene_color_copy, scene_depth_copy, ibl_specular_binding,
ibl_enabled, rt_reflection_texture, rt_reflection_enabled,
rt_reflection_intensity, sun_intensity)`; `simulation_state_texture()` exposes
the current wave field read-only for the renderer's analytic RT receiver.

**WaterRegion fields**: rect (`min_x/min_z/max_x/max_z`), `height`, `flow`
(normalized XZ), `flow_speed` (units/s), colors `deep_color`/`shallow_color`
(linear), `absorption` (1/units, Beer-Lambert), `foam_depth` (units),
`ssr_strength`/`ssr_max_distance`/`ssr_thickness` (screen-space reflection
march; strength 0 disables), `reflection_roughness` (0..1 → IBL mip ×6), `reflection_strength` (sky-reflection
dimmer, default 0.75 — the HDR sky otherwise swamps the refracted bed),
`refraction_strength` (screen-uv),
`detail_wave_strength` (max analytic slope, default 0.12), `sim_displacement`
(units per sim-height unit), `sim_normal_strength`, `ripple_foam_gain`,
`sim_center`/`sim_extent` (square window), `wave_speed` (units/s),
`wave_damping` (per substep).

**Simulation** (`water_sim.comp.hlsl`, `ComputePipelineType::WaterSim`):
512² (`SIM_RESOLUTION`) RG16F ping-pong pair storing (height, prev height);
usage `SAMPLER | COMPUTE_STORAGE_READ/WRITE | COLOR_TARGET` (color-target only
for the one-time lazy clear pass — there is no float-texture upload path).
Fixed substeps `SIM_SUBSTEP = 1/60`, `MAX_SUBSTEPS_PER_FRAME = 4`, dt clamped
at 0.25 s. Per substep: semi-Lagrangian back-trace along `flow`
(`advectTexels`, manual bilinear), 5-point-Laplacian Verlet wave step with
`waveC2Dt2 = min((wave_speed·dt/texel)², 0.24)` (Courant clamp), amplitude
`damping`, ≤`MAX_IMPULSES = 8` Gaussian impulse splats (first substep of the
frame only), and a 24-texel absorbing border band. Uniforms:
`WaterRenderer::SimUniforms` (160 B, pinned by `water_renderer.test.cpp`).

**Surface pass** (`water.vert.hlsl` / `water.frag.hlsl`, `PipelineType::Water`): grid of
float2 world-XZ vertices (`grid_cells_for`: ≤160 cells on the longer axis,
aspect-scaled short axis), lifted to `height` and displaced by the sim field
in the VS (displacement and sim normals fade at the window edge, 0.06 uv
fade). Pipeline: opaque, no cull, depth test `GREATER_OR_EQUAL` + write ON,
into RGBA16F + D32. `SceneRenderer` integration: sim recorded after IBL
upkeep; the pass runs between `HDROffscreen` and TAA — HDR color + depth are
copied via `SDL_CopyGPUTextureToTexture` into screen-sized SAMPLER textures
(`water_scene_copy_`/`water_depth_copy_`, lazily (re)created), then the live
targets re-bind with `LOAD`. Fragment: reversed-Z linearization
(`|depthB/(d+depthA)|`), depth-proportional refraction offset with
front-geometry rejection, Beer-Lambert absorption, screen-space
reflections (24-step geometric-growth march of the reflected ray against the
depth copy — snapshotted pre-water, so no self-hits — with one midpoint
refinement, distance-growing thickness, screen-edge/distance confidence
fade), plus a wave-aware RT texture when `rt_reflections` is enabled. Before
the water pass, `RtRenderer` recycles its quarter-resolution raw target after
the opaque resolve: camera rays intersect the water plane, refine twice
against simulation displacement, reject the region exterior/opaque cover,
and trace with the simulation central-difference + analytic detail normal.
The hybrid hierarchy keeps confident full-detail SSR, uses RT for off-screen,
occluded and screen-edge gaps, and falls through to the IBL specular cube on
RT misses or range fade (`R.y = abs(R.y)`, horizon-color fallback when IBL
is off). Schlick
fresnel (F0 = 0.02), fresnel(H)-scaled Blinn sun glint (pow 280), shoreline +
sim-crest foam, 7-unit depth-feathered waterline, shared `fog.hlsli`
atmospheric fog. Uniforms: `VertexUniforms` (96 B), `FragmentUniforms`
(272 B, including `viewProjSSR`, SSR controls and RT enable/intensity).

**Known limitations**: renders after transparents/particles (they are
overdrawn where water covers them on screen); without RT reflections,
off-screen content falls back from SSR to sky IBL; one region; no independent
water RT toggle (`GraphicsSettings::rt_reflections` controls opaque and water
tracing together; unset region = water off); water
draws are absent from the depth prepass (no velocity — TAA treats it via
camera reprojection); the flat plane floods terrain dips below `height`
inside the region rect (the host must keep a dry margin, e.g. the
playground's heightmap clamp).

---

## Environment, Sky, Fog, Bloom & Post-Processing

Technical specification for the atmosphere and post-processing subsystems:
`EnvironmentState`, `render::WorldRenderer` (skybox),
`render::VolumetricFog`, `render::Bloom`, the HDR-scene/composite contract,
and `render::RenderContext`. Source of truth:

- `src/bracken/contracts/environment.hpp`
- `src/bracken/render/world_renderer.hpp` / `.cpp`
- `src/bracken/render/volumetric_fog.hpp` / `.cpp` / `.test.cpp`
- `src/bracken/render/bloom.hpp` / `.cpp`
- `src/bracken/render/render_context.hpp` / `.cpp`
- `src/bracken/gpu/gpu_uniforms.hpp` (CPU↔GPU uniform mirrors)
- `src/bracken/shaders/skybox.vert.hlsl`, `skybox.frag.hlsl`,
  `grid.vert.hlsl`, `grid.frag.hlsl`, `volumetric_fog.frag.hlsl`,
  `bloom_downsample.frag.hlsl`, `bloom_upsample.frag.hlsl`,
  `composite.frag.hlsl`, `fog.hlsli`
- Integration points in `src/bracken/scene/scene_renderer.cpp`,
  `src/bracken/render/gtao.hpp` / `.cpp` (the AO chain) and
  `src/bracken/render/post_process_stack.hpp` / `.cpp` (composite/FXAA
  passes and the offscreen HDR targets live there)

General contracts inherited from the GPU layer: SDL3 GPU binding-set layout
(vertex: set 0 = sampled textures/storage, set 1 = uniform buffers; fragment:
set 2 = sampled textures/storage, set 3 = uniform buffers — see
`src/bracken/shaders/README.md`), and every HLSL cbuffer has a byte-for-byte
`alignas(16)` CPU mirror in `src/bracken/gpu/gpu_uniforms.hpp`.

### EnvironmentState

`src/bracken/contracts/environment.hpp`. Purpose: single source of truth for
sun, sky gradient, clouds/haze, hemisphere ambient, distance fog, volumetric
fog, and color grading.

The struct is declared in `namespace bracken::render`
(`bracken::render::EnvironmentState`), matching its `render/` path and the
references in `CLAUDE.md` / `docs/architecture.md`.

Plain aggregate; trivially copyable; no methods. All colors linear; defaults
model a clear summer midday (exact values in the field listing below; the
defaults are additionally locked by
`VolumetricFogEnvironment.DefaultsMatchLegacyHardcodedValues` in
`volumetric_fog.test.cpp`, because the volumetric defaults must reproduce the
pass's original hard-coded look).

Fields (glm types; world units where applicable):

```cpp
glm::vec3 sun_dir{glm::normalize(glm::vec3(-0.5f, -0.8f, -0.3f))}; // direction light TRAVELS
glm::vec3 sun_color{1.0f, 0.95f, 0.9f};
glm::vec3 sky_zenith{0.20f, 0.37f, 0.66f};
glm::vec3 sky_horizon{0.45f, 0.56f, 0.74f};
glm::vec3 ground_color{0.16f, 0.14f, 0.11f};
glm::vec3 ambient_sky{0.35f, 0.42f, 0.55f};
glm::vec3 ambient_ground{0.22f, 0.20f, 0.17f};
glm::vec3 fog_color_horizon{0.45f, 0.56f, 0.74f};
glm::vec3 fog_color_sun{0.95f, 0.75f, 0.50f};
float fog_start = 1000.0f;
float fog_end = 8000.0f;
float fog_height_falloff = 0.002f;
// Procedural sky clouds + horizon haze (sky_common.hlsli — skybox AND IBL
// capture). coverage/density/scale/haze dirty the IBL refresh
// (IblSkyParams); cloud_speed only scales the sim-stage sky-clock advance
// (time-like, never refreshes) — note it is read from the environment the
// host STAGES via set_environment, not from live DevUI edits.
float cloud_coverage = 0.35f; // sky fraction reading as cloud (0 clear .. ~1 overcast)
float cloud_density = 1.0f;   // opacity/edge weight (<1 wispy, >1 heavy)
float cloud_scale = 1.0f;     // noise footprint multiplier (<1 = larger formations)
float cloud_speed = 1.0f;     // scroll rate multiplier (phase accumulates; live-safe)
float haze_strength = 1.0f;   // horizon haze amount (color comes from the fog colors)
glm::vec3 fog_volumetric_sun_color{1.0f, 0.9f, 0.7f};
float fog_volumetric_scatter = 0.25f;
float fog_volumetric_density = 0.0015f;
float fog_volumetric_height = 100.0f;
float fog_volumetric_falloff = 0.005f;
float fog_volumetric_start = 100.0f;
float fog_volumetric_end = 3000.0f;
// Color grading (composite pass; look-style, live-editable — deliberately
// NOT GraphicsSettings). All-neutral defaults = pixel-identical output.
float grade_temperature = 0.0f;  // white balance [-1, 1], warm(+)/cool(-)
float grade_tint = 0.0f;         // white balance [-1, 1], magenta(+)/green(-)
glm::vec3 grade_slope{1.0f};     // ASC CDL slope (gain); post-tonemap
glm::vec3 grade_offset{0.0f};    // ASC CDL offset (lift)
glm::vec3 grade_power{1.0f};     // ASC CDL power (gamma)
```

#### Propagation and consumers

`scene::SceneRenderer` holds the authoritative copy
(`SceneRenderer::env_`), exposed mutable via
`EnvironmentState& SceneRenderer::environment()`. At the top of each
`render_frame` (`scene_renderer.cpp`):

- `light_dir_ = env_.sun_dir` (feeds CSM cascade fitting);
- `world_->set_environment(env_)`, `terrain_->set_environment(env_)`,
  `grass_renderer_->set_environment(env_)` — each sub-renderer stores a copy
  (the skybox uniforms carry the cloud/haze fields; the cloud scroll phase
  is advanced by the SIM stage — `dt * cloud_speed` of the host-staged
  environment — and adopted via `world_->set_sky_time(frame.sky_time())`);
- `ao_->set_clear_color(env_.fog_color_horizon)` — the offscreen HDR scene
  target clears to the horizon fog color;
- `make_model_lighting()` copies `sun_dir`, `sun_color`, `ambient_sky`,
  `ambient_ground`, `fog_color_horizon`, `fog_color_sun`, `fog_start`,
  `fog_end`, `fog_height_falloff` into `gpu::ModelLightingUniforms`, together
  with `GraphicsSettings::ambient_strength` (default `0.45f`) and
  `sun_intensity` (default `3.0f`);
- `VolumetricFog::render`/`build_uniforms` receives `env_` by const ref;
- the composite params take the grading fields each frame:
  `white_balance_coefficients(grade_temperature, grade_tint)`
  (`render/color_grading.hpp` — exact `(1,1,1)` at 0/0, otherwise the
  CIE/LMS white-point math; pinned by `color_grading.test.cpp`) plus the
  CDL triple verbatim.

Mutations therefore take effect on the next `render_frame` call. Since Phase
2 a frame packet may also carry an environment snapshot: when the host calls
`FrameBuilder::set_environment(env)` the renderer adopts that snapshot for
the frame (overwriting `env_`); when `frame.has_environment()` is false the
renderer keeps its own `env_` — which is exactly what keeps the mutable
`environment()` accessor and the DevUI live-edit path working. There is no
change notification or validation; fields are raw floats. HDR values (> 1.0)
are legal and expected. `fog_volumetric_sun_color` is used verbatim —
`sun_intensity` is *not* applied (documented in the header; hosts running a
day/night cycle must track their sun tint into this field).

Threading: the `environment()` reference is main-thread tier (all consumers
read it on the main thread during `render_frame`, which asserts
`BRACKEN_ASSERT_MAIN_THREAD()`); sim code stages its own copy via
`FrameBuilder::set_environment` instead. Note the sim stage also reads the
staged environment's `cloud_speed` for the sky-clock advance — live
`environment()` edits do not affect the scroll rate for hosts that stage.

### WorldRenderer (skybox)

`src/bracken/render/world_renderer.hpp/.cpp`, class
`bracken::render::WorldRenderer`. Non-copyable, non-movable. Renders the
procedural skybox (fullscreen triangle + `skybox.frag.hlsl`; no cubemap) and
holds the per-frame `EnvironmentState` copy the sky shader reads.

#### API surface

```cpp
[[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry);
void shutdown();
void set_environment(const EnvironmentState& env);       // stores a copy
const EnvironmentState& environment() const;
void set_sky_time(float t);     // adopt the packet's sim-advanced cloud-scroll phase (replaces the old update(dt))
float sky_time() const;   // accumulated cloud-scroll phase; drives ONLY the cloud scroll
void render_skybox(SDL_GPURenderPass*, SDL_GPUCommandBuffer*, const glm::mat4& view, const glm::mat4& projection);
glm::vec3 sun_direction() const;   // returns -env_.sun_dir (direction TO the sun)
const glm::vec3& light_dir() const; // env_.sun_dir (direction light travels)
```

#### Resources created in init()

- Skybox vertex buffer: 3 vertices × float3, a fullscreen clip-space
  triangle `(-1,-1) (3,-1) (-1,3)` (z unused), created via
  `gpu::GPUBuffer::create_static`. **Creation failure fails `init()`**
  (logged, returns `false`).

#### render_skybox algorithm

1. Early-out (silent) if vertex buffer / registry / pass / cmd null, or the
   `PipelineType::Skybox` pipeline is unavailable.
2. Compute `view_no_translation = mat4(mat3(view))`; the inverse
   view-projection `inverse(projection * view_no_translation)` is cached and
   recomputed only when rotation or projection changes
   (`cached_sky_view_` / `cached_sky_projection_` / `cached_sky_inv_vp_`).
3. Fill `gpu::SkyboxFragmentUniforms` (set 3, b0; mirrors
   `skybox.frag.hlsl` `SkyUniforms`): `invViewProjection`,
   `sunDirection = -env_.sun_dir`, `time = skybox_time_` (the pre-scaled
   scroll phase), then `sunColor`, `zenithColor`, `horizonColor`,
   `groundColor`, `fogHorizonColor`, `fogSunColor` each padded to 16 bytes,
   and a final `cloudCoverage / cloudDensity / cloudScale / hazeStrength`
   row from the `EnvironmentState` cloud/haze fields. Pushed with
   `SDL_PushGPUFragmentUniformData(cmd, 0, ...)`.
4. Bind the vertex buffer, `SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0)`.

`skybox.vert.hlsl` emits `z = 0.0` — the reversed-Z far plane — so with the
scene's `GREATER_OR_EQUAL` depth compare the sky passes only where the depth
buffer still holds the 0.0 clear (nothing drawn). The skybox is therefore
drawn *after* opaques in `render_3d_scene`, gated on
`scene.should_draw_skybox() && gfx.skybox_enabled`
(`scene_renderer.cpp:1015`).

`skybox.frag.hlsl` (all linear HDR; per-pixel ray from `invViewProjection`).
The entire radiance function lives in `sky_common.hlsli`
(`evaluateSkyRadiance`), SHARED with the IBL sky-capture pass so the
prefiltered environment can never drift from the visible sky; the skybox
pass adds only the per-pixel ray reconstruction and the IGN dither:

- `dayFactor = saturate(sunDir.y * 2.0 + 0.2)` scales the whole sky through
  dusk (zenith×`lerp(0.03,1,dayFactor)`, horizon×`lerp(0.04,…)`,
  ground×`lerp(0.05,…)`).
- 3-stop gradient: `lerp(horizon, zenith, pow(saturate(up), 0.42))`, then
  ground below via `1 − smoothstep(−0.08, 0, up)`.
- Clouds (all shaping from the `EnvironmentState` cloud fields via the
  uniform row): one **domain-warped, flat-spectrum FBM** deck on the virtual
  plane `dir.xz / (up + 0.18) * cloudScale`. The warp is a 2-channel
  3-octave FBM at 0.7/unit (scrolling at 0.8× wind so formations evolve);
  density = `cloudFbm(planeUV·0.7 + wind + warp·0.35)` where `cloudFbm` is
  6 octaves at persistence **0.62** (a flat spectrum — with classic 0.5 the
  first octave's near-bilinear iso-contours read as straight sheet edges).
  Noise primitives: quintic-fade value noise over a **PCG2D integer lattice
  hash** (the old `frac(p.x·p.y)` float hash quantizes under float32 at
  upper-octave coordinate magnitudes and rasterizes as straight creases);
  octaves rotate by `FBM_ROT` (±36.87°) to break grid alignment.
  Coverage: `threshold = 1 − coverage_eff` with
  `coverage_eff = cloudCoverage · lerp(0.55, 1, smoothstep(0.005, 0.16, up))`
  (perspective thinning toward the horizon); mask
  `smoothstep(threshold, +0.24)`, core `smoothstep(+0.08, +0.42)`.
  Lighting: bright tops `(1.16,1.13,1.09)` vs shadowed base
  `(0.46,0.50,0.58)` weighted by `core · lerp(1, 0.45, litFace)` where
  `litFace = saturate((density − densityTowardSun)·3 + 0.5)` (one extra
  `cloudFbm` sample offset 0.15 toward the sun — cheap directional
  self-shadowing); body night floor `lerp(0.035, 1, dayFactor)`. Scattering:
  rim `clouds·(1−core)` gets a `pow(cosSun,12)` silver lining scaled
  `(1.2 + 1.3·lowSun)` and lit faces a broad `pow(cosSun·0.5+0.5,4)·0.35`
  bleed, both tinted `lerp((1,0.96,0.90), (1.2,0.62,0.32), lowSun)` (warm at
  golden hour; `lowSun = 1 − saturate(toSun.y·3)`) and faded by
  `duskFade = saturate(toSun.y·8 + 1)` (underlit sunset bases survive to
  ~−7°, then clouds go dark). Alpha:
  `saturate(clouds·(0.72 + 0.28·core)·cloudDensity) · smoothstep(0.005, 0.14, up)`
  (soft edges, near-solid cores, horizon thinning). Wind scroll
  `(0.012, 0.005) × time` where `time` is the CPU-accumulated phase
  (`cloud_speed` never reaches the shader).
- Sun: disc `smoothstep(0.0175, 0.0115, angle)` at intensity **5.0 ×
  sunColor** (this is what feeds bloom), corona
  `exp(−θ²/0.015) * 0.5` tinted `(1.0,0.88,0.65)`, glare
  `exp(−θ/0.35) * 0.15 * 0.8 * dayFactor`; disc/corona occluded by cloud
  alpha (`sunVis = 1 − cloudAlpha * 0.9`).
- Aerial haze: `hazeSunColor(dir, toSun, fogHorizonColor, fogSunColor)`
  (fog.hlsli — SHARED with `applyAtmosphericFog`, see the distance-fog
  contract) blended with weight
  `saturate(exp(−|up| * 8) * 0.85 * hazeStrength)`.
- Stars: only when `sunDir.y < 0`; hash-cell field at threshold
  `step(0.998, h)`, faded in by `saturate(−sunDir.y * 4)`, occluded by
  `(1 − cloudAlpha)`.
- Night skyglow (same `sunDir.y < 0` block): horizon-weighted faint blue
  (`≈(0.015, 0.021, 0.038)` HDR at the horizon) added once the sun sets.
  Load-bearing for night scenes: the IBL capture shares this function and
  replaces the hemisphere ambient, so the glow IS the night ambient.
- Moon: a full moon exactly opposite the sun (`moonDir = −toSun` — no extra
  uniforms; astronomically correct for the full phase, so it rises at
  sunset). ~1.4 HDR disc with limb darkening + value-noise mottle + gaussian
  glow, faded by the same night ramp as the stars and occluded by clouds.
- IGN dither ±0.5/255 with screen offset `+71.0` to decorrelate from the
  composite pass's post-encode dither.

Threading/lifetime: main thread; `device`/`pipeline_registry` must outlive
the renderer; `shutdown()` (also run by the destructor) releases the skybox
buffer.

### Image-based lighting (`render::IBL`)

`src/bracken/render/ibl.{hpp,cpp}` + `sky_capture.frag.hlsl` /
`ibl_prefilter.frag.hlsl` / `ibl_irradiance.frag.hlsl` /
`ibl_brdf_lut.frag.hlsl` + `ambientIBL` in `pbr.hlsli`. Owned by
`SceneRenderer` (`ibl_`, read-only accessor `ibl()`); toggled by
`GraphicsSettings::ibl_enabled` (default ON; OFF branches every scene shader
back to the hemisphere `ambientPBR` — the IBL sampler slots are then fed
typed dummies and never sampled).

Resources (constants in `ibl.hpp`):

| Texture | Spec | Content |
|---|---|---|
| capture cube | 64px RGBA16F, full mip chain (`SPECULAR_SIZE`) | `evaluateSkyRadiance` per face (`sky_capture.frag`); mips via `SDL_GenerateMipmapsForGPUTexture` — the prefilter passes' mip-biased source |
| specular cube | 64px RGBA16F, `SPECULAR_MIPS = 7` mips | GGX-filtered importance sampling of the capture cube (`ibl_prefilter.frag`, 32 Hammersley samples, N=V=R, solid-angle mip bias); mip roughness = `mip / (SPECULAR_MIPS − 1)` (`IBL::mip_roughness`); mip 0 is a passthrough |
| irradiance cube | 16px RGBA16F (`IRRADIANCE_SIZE`) | cosine-hemisphere convolution of capture mip 2 (`ibl_irradiance.frag`, 128 samples); stores `(1/π)·∫L·cosθ` so `diffuse = kd·albedo·sample(N)` with no extra π |
| BRDF LUT | 256px RG16F (`BRDF_LUT_SIZE`) | Karis split-sum scale/bias `(A, B)` over `(NdotV, roughness)` (`ibl_brdf_lut.frag`, 1024 samples, Smith-Schlick `k = α/2`); rendered ONCE at init, sky-independent |

Face-basis contract: every pass renders a fullscreen triangle into one
`(face, mip)` via `SDL_GPUColorTargetInfo::layer_or_depth_plane`/`mip_level`;
the fragment reconstructs its direction from a CPU-supplied
`{forward, right, up}` basis (`SkyCaptureUniforms`/`IblFilterUniforms`).
`render::ibl_face_direction(face, u, v)` is the tested CPU mirror of the
Vulkan/D3D cube-lookup convention (`ibl.test.cpp` round-trips it against a
reference implementation of the spec table).

Regeneration (`IBL::update`, called by `render_frame` after `FrameUploads`,
before the shadow passes, only when `ibl_enabled`):

- `IblSkyParams` snapshots the sky-relevant `EnvironmentState` fields
  (`sun_dir`, `sun_color`, `sky_zenith`, `sky_horizon`, `ground_color`,
  `fog_color_horizon`, `fog_color_sun`, `cloud_coverage`, `cloud_density`,
  `cloud_scale`, `haze_strength`) — deliberately NOT `sky_time` or
  `cloud_speed` (both time-like: speed only scales the scroll-phase
  accumulation), so cloud scroll alone never triggers a refresh (clouds
  freeze at the snapshot time inside the maps).
- On change, the chain re-renders as `TOTAL_UNITS = 55` work units
  (6 capture faces → 1 capture mipgen → 42 specular (face, mip) passes →
  6 irradiance faces; `IBL::decode_unit` is pure and test-pinned), amortized
  at `UNITS_PER_FRAME = 9` tiny raster passes per frame ⇒ a full refresh
  spans 7 frames with no hitch (all passes ≤ 64×64).
- A change arriving mid-refresh finishes the in-flight snapshot first, then
  starts over — a continuously animating sun lags by ≤ ~2 refresh spans.
- `init()` builds the whole chain synchronously against a default
  `EnvironmentState` (plus the one-time BRDF LUT) on a one-off command
  buffer, so the maps are complete before the first sampled frame;
  `refresh_count()` counts completed builds (init = 1 — the playground
  smoke prints it as `ibl_refreshes=`).

Consumption: `SceneRenderer::bind_ibl(7)` binds irradiance/specular/LUT at
fragment sampler slots 7–9 on every model-family pipeline bind
(BLEND draws included — IBL is view/normal-based, unlike screen AO);
terrain/grass receive the same 3-element binding array + an `ibl_enabled`
flag through `render()` parameters (slots 5–7 / 3–5). The local-shadow
atlas + comparison sampler follow at the very last slot
(`bind_local_shadow_atlas(10)`; terrain 8 / grass 6 via a `render()`
parameter). The per-frame flag
`frame_ibl_active_ = ibl_enabled && IBL ready` drives the `iblEnabled`
cbuffer field of all four lighting UBOs.

Known limitations: the maps mirror the *procedural sky only* (no scene
geometry/probes — occluded interiors still see open sky); clouds in the
reflection are frozen at the last refresh; the specular mip count is a
compile-time contract with `pbr.hlsli` (`IBL_SPECULAR_MIPS`,
static_asserted); prefilter passes are natural compute-shader migrations
once the GPU layer grows compute support.

### Distance fog contract

`src/bracken/shaders/fog.hlsli`, applied by the model/terrain/grass fragment
shaders (not a pass). Helpers:

- `linearFog(d, start, end) = saturate((d − start) / max(end − start, 1e-5))`
- `exponentialFog(d, ρ) = 1 − exp(−dρ)`; `exponentialSquaredFog` analogous.
- `distanceFogFactor(d, start, end) = 1 − exp(−linearFog(...) * 2.0)`
  (legacy-compatible shape).
- `hazeSunColor(viewDir, toSun, fogHorizonColor, fogSunColor)` — the
  **shared sky↔fog agreement contract** (the skybox aerial haze calls the
  same function): `lowSun = 1 − saturate(toSun.y·3)` (grazing-sun factor),
  `duskFade = saturate(toSun.y·8 + 1)` (afterglow window to ~−7°); warm lobe
  `pow(saturate(dot(viewDir,toSun)), lerp(8, 2.5, lowSun))` lerps
  horizon→sun color × `duskFade`; Mie brightening
  `× (1 + pow(cosSun,6)·1.4·lowSun·duskFade)`; finally day-dimmed by
  `lerp(0.04, 1, saturate(toSun.y·2 + 0.2))` — the same ramp the sky
  gradient uses, so surface fog darkens through dusk with the sky. At
  midday this reduces exactly to the historical
  `lerp(fogHorizon, fogSun, pow(cos, 8))`.
- `applyAtmosphericFog(color, fragPos, cameraPos, sunDir, fogHorizonColor,
  fogSunColor, fogStart, fogEnd, heightFalloff)`:
  `factor = distanceFogFactor(dist)` × the analytic ray-average of
  `exp(−y · heightFalloff)` between camera and fragment heights
  (`h0 (1 − e^{−kΔy}) / (kΔy)`, with the `|kΔy| ≤ 1e-4` limit `h0`); fog
  color = `hazeSunColor(viewDir, −sunDir, ...)`;
  result `lerp(color, fogColor, factor)`. `sunDir` here is the travel
  direction (`EnvironmentState::sun_dir`).

CPU plumbing: `gpu::ModelLightingUniforms` carries `fogColor`, `fogSunColor`,
`fogStart`, `fogEnd`, `fogHeightFalloff`, `fogEnabled` (int). Per-draw
disable: `DrawFlags::NoFog`; global: `GraphicsSettings::fog_enabled`
(cached per frame as `SceneRenderer::frame_fog_active_`).

### VolumetricFog

`src/bracken/render/volumetric_fog.hpp/.cpp`, class
`bracken::render::VolumetricFog`. Half-resolution ray-marched height fog with
CSM-sampled god rays. Non-copyable/non-movable.

#### API

```cpp
[[nodiscard]] bool init(gpu::GPUDevice& device, int width, int height);
void shutdown();
void resize(int width, int height);            // no-op when unchanged
bool is_ready() const;                          // texture + sampler exist
void render(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& registry,
            SDL_GPUTexture* depth_texture, const ShadowMap& shadow_map,
            const scene::CameraState& camera, const EnvironmentState& env,
            bool god_rays_enabled = true, bool fog_enabled = true,
            float density_multiplier = 1.0f);
gpu::VolumetricFogUniforms build_uniforms(const ShadowMap&, const scene::CameraState&,
            const EnvironmentState&, bool god_rays_enabled, bool fog_enabled,
            float density_multiplier);          // pure CPU, public for unit tests
SDL_GPUTexture* fog_texture() const;            // nullptr until init
```

#### Resources

- `fog_texture_`: `max(1, width/2) × max(1, height/2)`,
  `SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT`,
  usage `COLOR_TARGET | SAMPLER`. RGB = inscattered light, A = 1 −
  transmittance (premultiplied-alpha contract for composite).
- `linear_sampler_`: linear min/mag, nearest mip, CLAMP_TO_EDGE.

`init` logs `SDL_Log("VolumetricFog: Initialized %dx%d (fog: %dx%d)")` and
returns false if either resource failed. `resize` recreates the texture only.

#### Uniform block

`gpu::VolumetricFogUniforms` (set 3, b0; mirrors
`volumetric_fog.frag.hlsl`), `alignas(16)`:

```cpp
glm::mat4 invViewProjection;
glm::mat4 shadowLightViewProjection[4];  // per-cascade light-space matrices
glm::vec4 cascadeSplits;                 // view-space far distance per cascade
glm::vec4 shadowBias;                    // per-cascade depth-compare bias, light-space [0,1]
glm::vec3 lightDir;        float fogDensity;        // default 0.0015f
glm::vec3 lightColor;      float scatterStrength;   // (1.0,0.9,0.7) / 0.25f
glm::vec3 fogColor;        float fogHeight;         // (0.45,0.56,0.74) / 100.0f
glm::vec3 cameraPos;       float fogFalloff;        // 0.005f
float startDist = 100.0f;  float maxDist = 3000.0f;
float godRaysEnabled = 1.0f; float densityMultiplier = 1.0f;
float shadowAtlasTexel = 0.0f; // 0.5 / ShadowMap::resolution() == 1 / atlas width
float lastActiveSplit = 0.0f;  // far distance of last active cascade
float _padding0, _padding1;
```

`build_uniforms` mapping (verified by `volumetric_fog.test.cpp`):

- `invViewProjection = inverse(camera.view_projection)`, cached against
  `camera.view_projection` so the ~80-flop inverse is skipped on idle frames.
- Cascades: for `i in [0, CSM_MAX_CASCADES)`,
  `src = cascades[min(i, active_cascades − 1)]` — inactive cascades **clone
  the last active one** (same scheme as `ShadowMap::get_shadow_uniforms`) so
  dynamic indexing is always safe.
  `shadowBias[i] = 1.5 * src.texel_world / max(src.depth_range, 1e-6)`.
- `lastActiveSplit = cascades[last].split_depth`;
  `shadowAtlasTexel = 0.5f / resolution` (the atlas is a 2×2 quadrant grid,
  so this equals one texel of the full atlas width; the shader comment says
  "1 / atlas width" — consistent).
- `lightDir = env.sun_dir`; `lightColor = env.fog_volumetric_sun_color`;
  `fogColor = env.fog_color_horizon`; `scatterStrength/fogHeight/fogFalloff/
  startDist/maxDist` from the corresponding `fog_volumetric_*` fields.
- `fogDensity = fog_enabled ? env.fog_volumetric_density : 0.0f` — disabling
  fog zeroes **only** density (god rays can still run through zero-density
  fog contributing nothing except via the ambient term gated by density).
- `godRaysEnabled = god_rays_enabled ? 1.0f : 0.0f`;
  `densityMultiplier` passed through.

#### render() pass

Preconditions: no render pass active; the offscreen scene render and shadow
passes must already be recorded this frame. Silent no-op when
`!is_ready() || !cmd || !depth_texture` or the `PipelineType::VolumetricFog`
pipeline is null. Records one render pass to `fog_texture_`
(`LOADOP_DONT_CARE`/`STOREOP_STORE`, no depth), binds fragment samplers:
slot 0 = scene depth + `linear_sampler_`, slot 1 = `shadow_map.atlas_texture()`
+ `shadow_map.shadow_sampler()` (raw, non-compare sampling in the shader),
pushes the uniform block, draws a fullscreen triangle (3 verts).

#### Shader algorithm (`volumetric_fog.frag.hlsl`)

- World-position reconstruction from depth via `invViewProjection` with a
  Vulkan Y-flip.
- `isSky = depth <= 0.0` (reversed-Z far/clear); sky pixels march the full
  `maxDist` so shafts silhouette against the sky.
- Phase function: `0.25 + 0.75 * pow(max(0, cos), 8)` (simplified forward
  Mie), against `−lightDir`.
- **`numSteps = 24`** fixed (the documented cost knob; not a uniform).
  `endDist = isSky ? maxDist : min(rayLength, maxDist)`; early return
  transparent black if `endDist <= startDist`. Step start jittered by
  interleaved gradient noise.
- Density: `fogDensity * densityMultiplier *
  exp(−(y/max(fogHeight,1))² * fogFalloff * 10)`; steps with density ≤ 0.001
  are skipped.
- Per step: ambient term `fogColor * 0.2`; god rays add
  `shadow * lightColor * phase * scatterStrength` where `shadow` is a single
  raw depth compare (no PCF — the march dither hides it) against the cascade
  selected by **camera distance of the sample** (distance ≥ view depth, so
  at worst a farther cascade is picked early). Samples past
  `lastActiveSplit` are fully lit; out-of-cascade-bounds projections are lit.
  Cascade `i` occupies atlas quadrant `(i & 1, i >> 1)`; UVs are clamped one
  atlas texel inside the quadrant to prevent bleed.
- Beer–Lambert accumulation: `stepT = exp(−density * stepSize)`;
  `fog += transmittance * (inScatter + ambient) * (1 − stepT)`;
  output `float4(fog, 1 − transmittance)`.

#### SceneRenderer integration

(`scene_renderer.cpp` post chain) Runs when
`(gfx.volumetric_fog || god_rays_active) && volumetric_fog_->is_ready() &&
shadow_map_->is_ready() && ao_->offscreen_depth()`, where
`god_rays_active = gfx.god_rays && shadows_on` — **god rays are force-disabled
when `shadow_mode == ShadowMode::Off`** because the atlas contents would be
stale/garbage. `density_multiplier = gfx.fog_density / 0.05f` (the
`GraphicsSettings::fog_density` default of `0.05f` maps to a 1× multiplier
on `env.fog_volumetric_density`). Defaults: `volumetric_fog = false`,
`god_rays = false`.

Performance: 24 density samples per half-res pixel (sky included), +1 shadow
tap per step with god rays on. Half-res target keeps this ~¼ the cost of
full res; composite upsamples depth-aware.

### Bloom

`src/bracken/render/bloom.hpp/.cpp`, class `bracken::render::Bloom`.
Threshold-extract → progressive downsample → progressive **additive**
upsample (Jimenez 2014 "Next Generation Post Processing in Call of Duty").
Non-copyable/non-movable.

#### API

```cpp
[[nodiscard]] bool init(gpu::GPUDevice& device, int width, int height);
void shutdown();
void resize(int width, int height);            // no-op when unchanged
bool is_ready() const;
void render(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& registry,
            SDL_GPUTexture* scene_color, float threshold = 1.0f, float exposure = 1.0f);
SDL_GPUTexture* bloom_texture() const;          // mip 0 (largest), the final result
```

#### Mip chain

- `static constexpr int MAX_MIPS = 6`; active count
  `mip_count_ = clamp(int(log2(min(w,h))) − 7, 4, 6)` (`compute_mip_count`).
  1080p/1440p/4K → 4 mips; 8K → 5. Rationale: wider glow radius at very high
  resolutions without extra cost at lower ones.
- `mip_textures_[i]`: `SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT`,
  `COLOR_TARGET | SAMPLER`; mip 0 is `max(1, width/2) × max(1, height/2)`,
  each subsequent mip halves (floor, min 1). Unused slots are reset.
- Sampler: linear min/mag, nearest mip, CLAMP_TO_EDGE.

#### render() algorithm

Silent no-op if `!is_ready() || !cmd || !scene_color` or either pipeline
(`PipelineType::BloomDownsample`, `PipelineType::BloomUpsample`) is missing.

Downsample loop (`i = 0 … mip_count_−1`): source is `scene_color` for i = 0
else `mip[i−1]`; target `mip[i]` with `LOADOP_DONT_CARE`. Uniforms
`gpu::BloomDownsampleUniforms` (set 3, b0):
`srcTexelSize = 1/srcDim`, `threshold`, `isFirstPass = (i==0)`, `exposure`.
Shader (`bloom_downsample.frag.hlsl`): 13-tap filter (center + 4 half-texel
inner ring + 8 one-texel outer ring) combined as five 2×2 groups. First pass
only: groups are **Karis-average weighted** (`1/(1+maxComponent)`) for
firefly suppression, then the soft-knee threshold runs:

```hlsl
lum  = dot(c, (0.2126, 0.7152, 0.0722)) * exposure;   // POST-exposure decision
knee = threshold * 0.5;
soft = clamp(lum - threshold + knee, 0, 2*knee);
soft = soft^2 / (4*knee + 1e-5);
contribution = max(soft, lum - threshold) / max(lum, 1e-5);
return color * max(contribution, 0);
```

The threshold decision is post-exposure so "what blooms" is stable as
exposure changes; the returned color stays in **pre-exposure scene units**
(composite multiplies the summed result by exposure exactly once). Subsequent
mips use plain Jimenez weights (center group 0.5, corner groups 0.125 each).

Upsample loop (`i = mip_count_−1 … 1`): source `mip[i]`, target `mip[i−1]`
with **`LOADOP_LOAD` + an additive pipeline** — each pass tent-filters the
smaller mip and *accumulates* onto the larger, producing both a tight core
and wide falloff (a code comment records the previous bug: `DONT_CARE` +
opaque overwrote each mip, leaving only the smallest mip blurred up).
Uniforms `gpu::BloomUpsampleUniforms`: `srcTexelSize`,
`bloomRadius = 1.0f` (hard-coded at the call site; the uniform supports other
radii). Shader: 9-tap tent (corners 1, edges 2, center 4, ÷16).

Failed `SDL_BeginGPURenderPass` calls `continue` (skip that mip, keep going).

#### Integration

(`scene_renderer.cpp`) Runs when `gfx.bloom_enabled && bloom_->is_ready()`
with `scene_in = post_->offscreen_color()`, `threshold = gfx.bloom_threshold`
(default **1.1**), `exposure = gfx.exposure`. Composite receives
`bloom_texture()` and multiplies by `gfx.bloom_strength` (default **0.25** —
`PostProcessStack::CompositeParams::bloom_strength` and
`gpu::CompositeUniforms::bloomStrength` default to the same value, tracking
`GraphicsSettings::bloom_strength` as the source of truth, pinned by a test).
When bloom is disabled the chain is skipped and `bloom_strength` is forced
to 0.

### HDR scene target, composite and tonemap

Formats (`src/bracken/gpu/pipeline_registry.hpp`):

- swapchain: `SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM` (initial assumption;
  actual format queried from the device — swapchain-format pipelines
  UI/Text/Composite/Fxaa are invalidated if it changes),
- HDR scene color: `SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT`,
- depth: `SDL_GPU_TEXTUREFORMAT_D32_FLOAT`.

The full-resolution offscreen color/depth pair is owned by
`render::PostProcessStack` (`offscreen_color()` / `offscreen_depth()` in
`src/bracken/render/post_process_stack.hpp`), which also hosts
`render_composite_pass` and `render_fxaa_pass` — the composite is the
engine's general post-resolve. The offscreen clear color is set per frame to
`env.fog_color_horizon` via
`PostProcessStack::set_clear_color(glm::vec3 linear)`.

Frame placement (see `docs/architecture.md` §"Pass order"): whenever the
post stack initialized (`post_->is_ready()`), the whole 3D scene renders into
the HDR target; the swapchain is acquired **late** (after GTAO/TAA/bloom/fog)
so the GPU can start on offscreen work without waiting for a presentable
image. If the late acquire fails, composite/FXAA/UI are skipped but the
command buffer still submits. If the post stack never initialized,
everything collapses to a single direct-to-swapchain `MainScene` pass — no
prepass/AO/bloom/fog/tonemap — with a hardcoded clear color
`{0.35, 0.45, 0.6, 1.0}` and reversed-Z depth cleared to 0.0
(`begin_main_pass`, `scene_renderer.cpp`).

#### Composite pass

`PostProcessStack::render_composite_pass(cmd, pipelines, target,
bloom_texture, fog_texture, const CompositeParams&)`. `CompositeParams`:
`bloom_strength = 0.25f`, `exposure = 1.0f`,
`tonemap_mode = 0`, `contrast = 1.0f`, `saturation = 1.0f`,
`depth_a = 0.0f`, `depth_b = 0.0f` (projection constants for depth
linearization: `depth_a = proj[2][2]`, `depth_b = proj[3][2]`; zeros degrade
the bilateral weights to plain bilinear in-shader), plus the auto-exposure
pair (`auto_exposure = false`, `exposure_buffer = nullptr`) and the neutral
grading defaults (`white_balance`/`cdl_slope`/`cdl_offset`/`cdl_power`).

CPU-side null policy: `bloomStrength` forced to 0 when `bloom_texture` is
null; `volumetricFogEnabled = fog_texture ? 1 : 0`; null bloom/fog slots are
back-filled with the scene color texture (any valid texture — the shader
ignores them); a null `exposure_buffer` binds the zeroed dummy SSBO with
`autoExposureEnabled = 0`. `lowResTexelSize = 1 / (half-res fog grid)`.

Uniforms `gpu::CompositeUniforms` (set 3, b0, 112 B): `bloomStrength`,
`volumetricFogEnabled`, `exposure`, `tonemapMode` (int), `contrast`,
`saturation`, `lowResTexelSize` (vec2), `depthA`, `depthB`,
`autoExposureEnabled`, `whiteBalanceEnabled`, `whiteBalance` (vec3),
`cdlEnabled`, `cdlSlope`/`cdlOffset`/`cdlPower` (vec3 + pad each). The
enabled flags are derived in `render_composite_pass` (white balance ⇔
coefficients ≠ 1, CDL ⇔ any of slope/offset/power non-neutral) so the
all-neutral default path executes the exact ungraded arithmetic.

Fragment bindings (set 2): 0 = HDR scene color (linear-clamp), 1 = bloom,
2 = fog, 3 = full-res depth (nearest-clamp); fragment storage buffer 0 =
exposure state (`StructuredBuffer<float4>`; `[0].x` = adapted EV).

Shader (`composite.frag.hlsl`) sequence:

1. `bilateralTaps`: 4 nearest half-res texel centers, weight = bilinear ×
   `exp(−relDiff × 40.0)` where `relDiff = |tapZ − centerZ| / max(centerZ, 1e-3)`
   on linearized view depth `|depthB / (depth + depthA)|` (valid for standard
   and reversed-Z RH_ZO). Degenerate sum ≤ 1e-4 falls back to plain
   bilinear. (Screen AO is applied in the forward pass, never here.)
2. Volumetric fog (if enabled): premultiplied blend
   `color = color * (1 − vfog.a) + vfog.rgb`, linear.
3. `color += bloom.rgb * bloomStrength` (plain bilinear full-screen sample).
4. `color *= exposure * (autoExposureEnabled ? exp2(exposureState[0].x) : 1)`.
5. White balance (if `whiteBalanceEnabled`): linear→LMS (CIECAM02 matrices,
   in-shader constants) × `whiteBalance` gains → linear, clamped ≥ 0.
   Pre-tonemap on purpose — it models the scene illuminant.
6. Tonemap by `tonemapMode`: 0 = `acesFitted` (default), 1 =
   `acesNarkowicz`, 2 = `reinhard` (`c/(c+1)`), else `saturate` (clamp) —
   matching `bracken::TonemapMode` (`graphics_settings.hpp`: `ACESFitted=0`,
   `ACESNarkowicz=1`, `Reinhard=2`, `None=3`; explicit values are the
   serialized settings format).
7. ASC-CDL (if `cdlEnabled`): `pow(max(color·cdlSlope + cdlOffset, 0),
   cdlPower)` on the display-referred image.
8. `color = (color − 0.5) * contrast + 0.5`, then
   `lerp(luminance, color, saturation)`, then `saturate`.
9. `linearToSrgb` gamma encode, then IGN dither ±0.5/255 **after** the
   encode (a true half-LSB before UNORM quantization; in linear space the
   encode slope would over-dither darks ~13×).

Render pass: `LOADOP_DONT_CARE`/`STOREOP_STORE`, fullscreen triangle. Target
is the swapchain directly (`aa_mode == AAMode::Off`) or the lazily created
swapchain-format `ldr_target()`, in which case
`render_fxaa_pass(cmd, swapchain)` (FXAA 3.11, `FxaaUniforms.invScreenSize`)
must run afterwards in the same frame — the composite pass caches the
pipeline registry pointer FXAA uses.

Settings → params mapping (`scene_renderer.cpp:741`):
`bloom_strength = bloom_enabled ? gfx.bloom_strength : 0`,
`ao_strength = ao_mode != Off ? gfx.ao_strength : 0`, `exposure`,
`tonemap_mode = (int)gfx.tonemap_mode`, `contrast`, `saturation`,
`depth_a/b` from the live camera projection.

### RenderContext

`src/bracken/render/render_context.hpp/.cpp`, class
`bracken::render::RenderContext`. Owns `SDL_Window*` + `gpu::GPUDevice`;
non-copyable/non-movable; destructor calls `shutdown()`.

#### API and semantics

```cpp
[[nodiscard]] bool init(int width, int height, const std::string& title);
void shutdown();
void update_viewport();                       // re-reads pixel size
void begin_frame();                           // legacy: stores cmd internally
SDL_GPUCommandBuffer* begin_frame_cmd();      // returns cmd (nullptr on failure)
void end_frame();                             // legacy pair of begin_frame()
void end_frame(SDL_GPUCommandBuffer* cmd);    // submit
[[nodiscard]] bool end_frame_and_wait();      // fence submit + block (readback only)
SDL_GPUCommandBuffer* current_command_buffer() const;
SDL_Window* window() const;   gpu::GPUDevice& device();
int width() const; int height() const; float aspect_ratio();
SDL_GPUTextureFormat swapchain_format() const;
SDL_GPUTexture* acquire_swapchain_texture(SDL_GPUCommandBuffer*, uint32_t* w = nullptr, uint32_t* h = nullptr);
void set_vsync_mode(int mode);  int max_vsync_mode() const;
void set_window_mode(int window_mode, int resolution_index = 0);
struct DisplayMode { int w, h; float refresh_rate; float pixel_density; };
const std::vector<DisplayMode>& available_resolutions() const;
void query_display_modes();
```

- `init`: `SDL_CreateWindow(title, w, h, SDL_WINDOW_RESIZABLE)` (no GL flag —
  SDL3 GPU owns the context), then `device_.init(window_)`; on device failure
  the window is destroyed and false returned. Logs backend driver name and
  drawable size.
- **Pixel-size contract**: `width_`/`height_` come from
  `SDL_GetWindowSizeInPixels` (HiDPI-correct); the swapchain and every
  swapchain-sized offscreen target (AO, bloom, fog, LDR) are allocated in
  pixels, not logical units. `begin_frame_cmd()` calls `update_viewport()`
  every frame. `aspect_ratio()` guards `height_ > 0` (returns 1.0 otherwise).
- `begin_frame()` (legacy): if a frame is already in progress, logs a warning
  and submits the previous command buffer to avoid a leak, then acquires a
  new one. `begin_frame_cmd()` maps to `SDL_AcquireGPUCommandBuffer` via
  `GPUDevice::begin_frame()` (logs + returns nullptr on failure).
- `end_frame(cmd)` → `SDL_SubmitGPUCommandBuffer`. `end_frame_and_wait()` →
  `SDL_SubmitGPUCommandBufferAndAcquireFence` + `SDL_WaitForGPUFences` — a
  deliberate pipeline stall used only by the screenshot readback path
  (`SceneRenderer::capture_screenshot_and_end_frame`).
- Vsync: mode `0` = `SDL_GPU_PRESENTMODE_IMMEDIATE`, `1` = `VSYNC`,
  `2` = `MAILBOX`. Requests are clamped to `max_vsync_mode()`
  (probed via `GPUDevice::supports_present_mode`); repeated same-mode calls
  are no-ops; failures log. Initial tracked mode is `1`.
- `query_display_modes()`: `SDL_GetFullscreenDisplayModes` for the window's
  display, deduplicated by (w, h, pixel_density) — SDL lists the same
  resolution at multiple refresh rates. The header comment says the list
  contains native modes with `pixel_density == 1.0`, but the code does not
  filter on density (it only dedupes) — mildly overspecified doc comment.
- `set_window_mode(mode, resolution_index)`: `0` windowed, `1` borderless
  fullscreen, `2` exclusive fullscreen using the display mode matched from
  `available_resolutions_[resolution_index]` (lazily querying modes if
  empty). An unmatched/out-of-range index (stale settings file, changed
  monitor) **logs a warning, clears the fullscreen mode pointer** (discarding
  any previously applied exclusive mode so SDL can't keep a stale
  resolution), and **falls back to borderless deterministically** — the
  recorded `current_window_mode_` becomes borderless, so a later retry of the
  same exclusive request is not swallowed by change detection. Skips entirely
  when both values are unchanged (`current_window_mode_`/`_resolution_index_`
  start at `-1` to force the first call through); ends with
  `update_viewport()`.
- There is no clear-color API on `RenderContext` — the offscreen scene clear
  uses `env.fog_color_horizon` (via `PostProcessStack::set_clear_color`),
  the fallback pass hardcodes `{0.35, 0.45, 0.6, 1.0}`, and the UI
  clear hardcodes `{0.1, 0.1, 0.15, 1.0}`.

Threading: all of it is main-thread; `GPUDevice::acquire_swapchain_texture`
asserts `BRACKEN_ASSERT_MAIN_THREAD()` (SDL contract: the thread that created
the window).

### Threading & lifetime summary

| Object | Thread | Lifetime notes |
|---|---|---|
| `EnvironmentState` (`SceneRenderer::env_`) | main | copied into sub-renderers per frame; mutate any time between frames |
| `WorldRenderer` | main | needs `GPUDevice`/`PipelineRegistry` to outlive it; `shutdown()` idempotent |
| `VolumetricFog` / `Bloom` | main | GPU-texture owners; `render()` requires no active render pass on `cmd`; output textures valid until next `resize`/`shutdown` |
| `RenderContext` | main | owns window + device; must outlive `SceneRenderer` (passed by ref to `SceneRenderer::init`) |

Error-handling style throughout this area is the boot-time `bool` + log
pattern (per `CLAUDE.md` conventions): `init()` returns `[[nodiscard]] bool`;
per-frame paths fail soft (silent or logged no-ops), never throw, never
return errors.

### Known limitations / stubs / silent behaviors

- Skybox cloud coverage/density/scale/speed and haze strength are
  `EnvironmentState` fields; the remaining sky constants (wind direction,
  sun disc size/intensity (5.0), star density, cloud palette) are shader
  constants — not exposed on `EnvironmentState`.
- `PipelineType::Skybox` is annotated "Skybox cubemap rendering" in
  `pipeline_registry.hpp`, but the implementation is fully procedural — no
  cubemap is sampled.
- Volumetric fog step count (24) is a shader constant; quality/cost is not
  runtime-tunable. God-ray shadow sampling is a single raw compare per step
  (no PCF).
- Volumetric fog is half-res; the composite's joint-bilateral upsample uses
  the half-res fog grid (AO has its own upsample inside `render::GTAO`).
- With `aa_mode == FXAA`, the LDR intermediate is lazily created at first
  use (`PostProcessStack::ldr_target()`).
- Bloom mip render-pass begin failures skip that mip silently
  (`continue`), potentially leaving stale texel data in the chain for that
  frame.
- When the AO subsystem is not ready, the entire post stack (HDR target,
  bloom, fog, tonemap) is bypassed — the fallback path renders linear values
  directly to the UNORM swapchain with no gamma encode.

---

## Particle Effects System

Data-driven CPU-simulated, GPU-instanced mesh-particle effects. Three strictly layered modules plus glue in the scene layer:

| Layer | Files | Namespace |
|---|---|---|
| Definitions (POD-style data, header-only) | `src/bracken/effects/effect_definition.hpp` | `bracken` |
| Runtime simulation | `src/bracken/effects/effect_system.hpp/.cpp` | `bracken::systems` |
| Rendering | `src/bracken/effects/effect_renderer.hpp/.cpp` | `bracken::render` |
| Spawn commands + frame glue | `src/bracken/scene/frame_packet.hpp` (`EffectSpawnCommand`), `src/bracken/scene/frame_builder.hpp` (`spawn_effect`), `src/bracken/scene/scene_renderer.cpp` | `bracken::scene` |

Design contract (`docs/effects_system.md`): the engine owns the structs, the simulation, and the rendering; **serialization/parsing of effect files is entirely the host's responsibility** — no effect file I/O exists anywhere in the engine.

### Definition layer (`src/bracken/effects/effect_definition.hpp`)

Header-only, no dependencies beyond glm/`<string>`/`<vector>`. All types in namespace `bracken`.

#### Enums

```cpp
enum class CurveType   { CONSTANT, LINEAR, EASE_IN, EASE_OUT, EASE_IN_OUT, FADE_OUT_LATE };
enum class SpawnMode   { BURST, CONTINUOUS };
enum class VelocityType{ DIRECTIONAL, RADIAL, ORBITAL, CUSTOM, ARC };
```

#### `Curve`

```cpp
struct Curve {
    CurveType type = CurveType::CONSTANT;
    float start_value = 1.0f;
    float end_value = 1.0f;
    float fade_start = 0.8f;          // FADE_OUT_LATE only
    float evaluate(float t) const;    // defined inline in the same header
};
```

`evaluate(t)` clamps `t` to `[0,1]`, then:

- `CONSTANT` → `start_value`
- `LINEAR` → `mix(start, end, t)`
- `EASE_IN` → `mix(start, end, t*t)`
- `EASE_OUT` → `mix(start, end, 1-(1-t)²)`
- `EASE_IN_OUT` → `mix(start, end, s)` with `s = t<0.5 ? 2t² : 1-2(1-t)²`
- `FADE_OUT_LATE` → `start` for `t < fade_start`, else `mix(start, end, (t-fade_start)/(1-fade_start))`
- fall-through return `start_value`

#### `VelocityDefinition`

Fields and defaults (exact): `type = DIRECTIONAL`, `speed = 100.0f` (units/sec), `direction = {1,0,0}` (DIRECTIONAL/CUSTOM), `spread_angle = 0.0f` (degrees), `gravity = {0,0,0}` (accel), `drag = 0.0f` (0–1 damping); ORBITAL: `orbit_radius = 50.0f`, `orbit_speed = 2.0f` (rotations/sec), `orbit_height_base = 0.0f`, `height_variation = 0.0f`; ARC: `arc_radius = 36.0f`, `arc_height_base = 25.0f`, `arc_height_amplitude = 15.0f`, `arc_tilt_amplitude = 0.8f` (radians).

#### `RotationDefinition`

`initial_rotation = {0,0,0}` (Euler degrees), `rotation_rate = {0,0,0}` (degrees/sec), `face_velocity = false`.

#### `AppearanceDefinition`

`scale_over_lifetime` / `opacity_over_lifetime` (both default-constructed `Curve`s, i.e. CONSTANT 1.0), `color_tint = {1,1,1,1}`, `color_end = {1,1,1,1}`, `use_color_gradient = false`. **Alpha channels of `color_tint`/`color_end` are never consumed** — per-particle opacity comes only from `opacity_over_lifetime` (see instance packing below).

#### `EmitterDefinition`

`name`; `model` (ModelManager name — every particle renders as this mesh); `spawn_mode = BURST`; `spawn_count = 1`; `spawn_rate = 10.0f` (CONTINUOUS, particles/sec; must be > 0 and finite — validated at spawn); `particle_lifetime = 1.0f`; `velocity`; `rotation`; `appearance`; `delay = 0.0f`; `duration = -1.0f` (−1 ⇒ effective duration = `particle_lifetime`).

#### `EffectDefinition`

`name`; `emitters` (`std::vector<EmitterDefinition>`); `duration = -1.0f` — effect-level duration override: when `>= 0`, all emitters stop spawning once the effect has been alive this long, and the effect completes (is compacted away) as soon as its remaining particles die; `-1` (default) = emitter-driven completion.

### Runtime layer (`src/bracken/effects/effect_system.hpp/.cpp`)

Namespace `bracken::systems`; re-exports the definition types via `using` declarations.

#### Data structures

```cpp
struct Particle {
    glm::vec3 position{0,0,0}, velocity{0,0,0};
    glm::vec3 rotation{0,0,0};       // current Euler angles, RADIANS
    glm::vec3 rotation_rate{0,0,0};  // radians/sec
    float scale = 1.0f, opacity = 1.0f;
    glm::vec4 color{1,1,1,1};
    float age = 0.0f, lifetime = 1.0f;
    float orbit_angle = 0.0f; int orbit_index = 0; glm::vec3 orbit_origin{0,0,0};
    glm::vec3 arc_origin{0,0,0}, arc_direction{1,0,0};
};
```

(The renderer resolves the model per-emitter — `EffectRenderer::resolve_emitter_model` — so particles carry no model reference.)

```cpp
struct EmitterInstance {
    const EmitterDefinition* definition = nullptr;
    std::vector<Particle> particles;
    float age = 0.0f;
    float next_spawn_time = 0.0f;    // CONTINUOUS cadence; initialized to def.delay
    bool has_spawned_burst = false;
    glm::vec3 origin{0,0,0}, direction{1,0,0};
    bool is_active() const;   // definition && age < (duration<0 ? particle_lifetime : duration)
    bool is_complete() const;  // !is_active() && particles.empty()
};

struct EffectInstance {
    const EffectDefinition* definition = nullptr;
    std::vector<EmitterInstance> emitters;
    float age = 0.0f;          // compared against EffectDefinition::duration when >= 0
    bool is_complete() const;  // effect duration elapsed && all particles dead,
                               // or (duration < 0) all emitters complete
};
```

Lifetime contract: instances hold raw `const` pointers into the `EffectDefinition` **and into elements of its `emitters` vector**. The definition must outlive all instances and must not have `emitters` reallocated while instances are live. Spelled out in `docs/effects_system.md` and in `effect_system.bench.cpp` comments.

#### `EffectSystem` API

```cpp
class EffectSystem {
public:
    int spawn_effect(const EffectDefinition* definition, const glm::vec3& position,
                     const glm::vec3& direction = {1, 0, 0});
    // Stage-helper forms (Phase 3): update() fans out INTEGRATE/COMPACT on js.
    template<typename TerrainHeightFn>
    void update(core::jobs::JobSystem& js, float dt, TerrainHeightFn&& get_terrain_height);
    void update(core::jobs::JobSystem& js, float dt);
    // Compatibility delegators servicing JobSystem::instance():
    template<typename TerrainHeightFn> void update(float dt, TerrainHeightFn&& get_terrain_height);
    void update(float dt);
    void update(float dt, std::nullptr_t);  // forwards to update(dt)
    // Phase 4: write this frame's effect-instance snapshot into the packet
    // via FrameBuilder::alloc_effect_snapshot — live emitters in deterministic
    // (effect, emitter) order + render-relevant per-particle state. Sim-stage
    // tail, single-threaded, after update(); the renderer consumes the sealed
    // spans, never this object.
    void snapshot_into(scene::FrameBuilder& fb) const;
    const std::vector<EffectInstance>& get_effects() const;  // sim-stage/test introspection
    void clear();
    size_t effect_count() const;
    size_t particle_count() const;  // O(effects * emitters) sum
};
```

- `spawn_effect` returns the index of the new effect (`effects_.size() - 1`) or `-1` if `definition == nullptr`. It only builds `EmitterInstance`s (`origin`/`direction` snapshot, `next_spawn_time = def.delay`); **no particles are created at spawn** — they appear in the next `update()`. CONTINUOUS emitters with a non-positive or non-finite `spawn_rate` are rejected here with a warning (once per definition) and stay inert. Indices are positional and invalidated by compaction; nothing in the engine retains them.
- The templated `update` takes any callable `float(float x, float z)` (terrain height); templating avoids `std::function` dispatch in the per-particle loop. The no-terrain overload uses an internal `NoTerrainFn` sentinel with the shared body `update_t<..., HasTerrain=false>`, compiling the clamp out entirely (`if constexpr`). **The callable is invoked from job workers during INTEGRATE** — it must be safe to call concurrently (a pure function of `(x, z)` over state nothing mutates during the call); the engine's terrain sampler qualifies (const bilinear reads of an immutable `Heightmap`).

#### Update algorithm (per `update(js, dt)` call — three phases, `update_t`)

**SPAWN** — serial on the calling thread, in effect/emitter submission order (this fixes the `thread_local` RNG draw sequence, so results are worker-count-invariant). For each effect: `effect.age += dt`; `allow_spawn = effect_allows_spawning(effect)` — false once an effect-level `duration >= 0` has elapsed (`effect.age >= definition->duration`), gating all emitter spawning off; for each emitter, `spawn_phase` (`.cpp`):

1. `emitter.age += dt`.
2. If `allow_spawn && is_active()`:
   - **BURST** (not yet fired): if `age >= delay`, spawn `spawn_count` particles, set `has_spawned_burst`. Consequence: if the active window (`duration`, or `particle_lifetime` when −1) ends before `delay`, the burst never fires.
   - **CONTINUOUS**: `while (age >= next_spawn_time) { spawn 1; next_spawn_time += 1/spawn_rate; }`. Catch-up is unbounded within a frame. The interval is hoisted and validated — only a positive, finite interval enters the loop, guarding independently of `spawn_effect`'s own validation, so a non-positive/non-finite/NaN rate never spawns (and can no longer hang the loop).

**INTEGRATE** — `parallel_for(js, 0, total_particles, …, INTEGRATE_GRAIN = 512)` over a flattened global particle index space (an exclusive-prefix offset table over live emitters is built serially into member scratch `live_emitters_`/`emitter_offsets_` first; each chunk `upper_bound`s its start emitter and walks contiguous ranges). Per particle: the static, pure `update_particle` (below); when `HasTerrain`, clamp `position.y = max(position.y, get_terrain_height(x, z))` — no bounce/velocity kill, particles slide. All writes are disjoint per-particle — no merge, no atomics.

**COMPACT** — `parallel_for` over live emitters (inline below `COMPACT_EMITTER_GRAIN = 16` emitters); each chunk stable erase-removes particles with `age >= lifetime` from only its own emitters' vectors.

After the phases: `compact_effects()` (`.cpp`, serial) erase-removes effects where `is_complete()` (all emitters done and particles dead, or effect-level duration elapsed and particles dead), then emits `BRACKEN_PROFILE_PLOT("effects.particles", particle_count())` (Tracy plot, one sample per update). The grain guards keep the playground's few hundred particles as one inline chunk on the caller.

#### Particle spawn (`spawn_particles`, `.cpp`)

Per particle: `position = emitter.origin`; `lifetime = def.particle_lifetime`; `velocity = calculate_initial_velocity(...)`; `rotation = radians(initial_rotation)`, `rotation_rate = radians(rotation_rate)`; `scale/opacity = curves.evaluate(0)`; `color = color_tint`. ORBITAL extras: `orbit_index = particles.size()` at insertion, `orbit_angle = 2π/spawn_count * orbit_index` (even distribution — note for CONTINUOUS orbital emitters the index keeps growing past `spawn_count`), `orbit_origin = origin`. ARC extras: `arc_origin = origin`, `arc_direction = emitter.direction`.

`calculate_initial_velocity`:

- `DIRECTIONAL`: `normalize(direction) * speed`; if `spread_angle > 0`, rotate by a uniform random angle in ±`radians(spread_angle)` **around the Y axis only** (2D rotation in the XZ plane; comment says "simplified").
- `RADIAL`: random angle in `[0, 2π)`, velocity `(cos·speed, 0, sin·speed)` — horizontal only.
- `ORBITAL`, `ARC`: zero (fully kinematic; position computed per frame).
- `CUSTOM`: `direction * speed` (no normalization).

RNG: `random_float` uses a lazily constructed `thread_local std::mt19937` seeded from `std::random_device` (avoids throwing at static-init; MT-safe without a mutex).

#### Per-particle integration (`update_particle`, `.cpp`)

`age += dt`; `t = age / lifetime`.

- **ARC** (kinematic, replicates the old `draw_melee_slash`): `base_angle = atan2(arc_direction.x, arc_direction.z)`; `swing = -1 + 2t` radians; `rotation = base_angle + swing`; `position.xz = arc_origin.xz + (sin, cos)(rotation) * arc_radius`; `position.y = arc_origin.y + arc_height_base + sin(tπ) * arc_height_amplitude`; particle Euler: `rotation.x = sin(tπ) * arc_tilt_amplitude`, `rotation.y = rotation + π/2`, `rotation.z = -0.5` (fixed roll).
- **ORBITAL** (kinematic): `orbit_angle += orbit_speed * 2π * dt`; `position.xz = orbit_origin.xz + (cos, sin)(orbit_angle) * orbit_radius`; `position.y = orbit_origin.y + orbit_height_base` plus, when `height_variation > 0`, `sin(orbit_angle * 3) * height_variation`.
- **Otherwise** (DIRECTIONAL/RADIAL/CUSTOM): `position += velocity * dt`; gravity applied iff `dot(gravity, gravity) > 0` (squared test, no sqrt); drag iff `drag > 0`: `velocity *= (1 - drag * dt)` (explicit damping — can overshoot/negate for `drag * dt > 1`; not clamped).
- Rotation: if `face_velocity && dot(velocity, velocity) > 1e-4f` (speed > 0.01), `rotation.y = atan2(velocity.x, velocity.z)` (normalize-free; atan2 is scale-invariant); else `rotation += rotation_rate * dt`. Pinned by two dedicated tests.
- Appearance: `scale`/`opacity` from the curves at `t`; if `use_color_gradient`, `color = mix(color_tint, color_end, t)`.

Note that ARC/ORBITAL also overwrite parts of `rotation` regardless of `RotationDefinition` (ARC drives all three axes).

#### Threading & performance

Thread-safety tier: **owned, sim-stage in the engine path** — one owner calls `spawn_effect`/`update`/`snapshot_into`/`clear`; `update()` internally fans out INTEGRATE/COMPACT onto the given `JobSystem` and joins before returning, so nothing is touched by workers between updates. In the engine path the owner is `Application`'s **sim-job tail** (pool thread, per frame: apply the sealed builder's spawn commands → `update(js, dt, terrain sampler)` → `snapshot_into(fb)`); the renderer never touches the live system — it draws the packet's effect-instance snapshot. Output is bit-identical to a serial run at any worker count: spawning (the only RNG consumer) is serial, and the parallel phases write only disjoint per-particle/per-emitter state (pinned by `effect_system_parallel.test.cpp`, `init(1)` vs `init(4)` caller-owned pools). Cost per update is O(live particles) with vector erase-remove compaction per emitter per frame. Micro-benchmarks in `src/bracken/effects/effect_system.bench.cpp`: `BM_EffectUpdate` (steady-state 1k/10k particles), `BM_EffectUpdateParallel` (`{particles, workers}` on a caller-owned pool), `BM_EffectUpdateWithTerrain` (isolates sampler + clamp overhead), `BM_EffectSpawnBurst` (256-particle burst), `BM_CurveEvaluate` per `CurveType`. Profiler zones: `EffectSystem::spawn_effect` / `spawn_particles` / `compact_effects` (color `0x3D6E9E`).

### Spawn commands (`src/bracken/scene/frame_packet.hpp` / `frame_builder.hpp`)

```cpp
struct EffectSpawnCommand {
    const bracken::EffectDefinition* definition = nullptr;   // null spawns are skipped
    glm::vec3 position = {0, 0, 0};
    glm::vec3 direction = {1, 0, 0};
};

void FrameBuilder::spawn_effect(const bracken::EffectDefinition* definition, const glm::vec3& position,
                                const glm::vec3& direction = {1, 0, 0});
std::span<const EffectSpawnCommand> FramePacket::effect_spawns() const;   // sealed read
```

Spawns ride the packet like every other command: emitted during fill, applied exactly once to the sim-owned `EffectSystem` by the engine's sim-job tail (deterministic lane order, before the update), gone when the packet retires. There is no persistent queue, no `clear_particle_effect_spawns`, and no non-const `render_frame` parameter — a packet is consumed exactly once, so "queue a spawn exactly once per event" is enforced by construction. The `EffectDefinition*` is long-lived host asset data and must outlive the spawned instances.

### Engine integration (Phase 4 split: sim side / render side)

The `EffectSystem` is owned by **`Application`** (sim-stage; `effect_system()` protected accessor — `SceneRenderer` has no effect-system accessor anymore). The `EffectRenderer` half stays owned by `SceneRenderer` (`std::unique_ptr<render::EffectRenderer> effects_`; `effects_->init(...)` in `SceneRenderer::init`, failure fatal). The two halves communicate **only** through the packet's effect-instance snapshot.

Sim side, per frame (inside `Application::run_sim_job`, pool thread, after `on_simulate`):

1. `FrameBuilder::for_each_effect_spawn` → `effect_system_->spawn_effect(...)` per command with a non-null definition, in deterministic (lane, emission) order. Because this runs **before** the update, a spawn queued in `on_simulate` produces its first particles in the same sim frame's update.
2. `effect_system_->update(js, dt, terrain sampler)` — the templated overload with a lambda over `SceneRenderer::get_terrain_height` (worker-safe const reads); nested fork-join on the pool.
3. `effect_system_->snapshot_into(fb)` — allocates + fills the packet snapshot; `fb.set_engine_frame(..., effect_count())` records the live-effect count.

Render side, per frame (inside `SceneRenderer::render_frame`, main thread, over the **sealed** packet):

1. `effects_->ensure_models_uploaded(frame.effect_emitters())` — pre-pass, outside any render pass (GPU uploads submit copy work; the mid-frame path never uploads, only skips).
2. Inside the single consolidated `FrameUploads` `SDL_GPUCopyPass` (shared with instance buffers, bone palettes, grass, cluster lights, debug lines): `effects_->upload_particle_instances(copy_pass, frame.effect_emitters(), frame.effect_particles())`.
3. In the main HDR render pass (`render_3d_scene`), after opaques, skybox, and sorted transparent models, before debug lines: `effects_->draw_particle_effects(main_render_pass_, cmd, camera.view, camera.projection, camera.position)`.

Dev-overlay stats come from the packet, not the live system: `SceneRenderer::frame_effect_count()`/`frame_particle_count()` (cached from the rendered packet) feed the DevUI Stats tab's `Particles: N (in M effects)` line.

### Rendering layer (`src/bracken/effects/effect_renderer.hpp/.cpp`)

#### Public API

```cpp
// render/particle_transform.hpp — generic over anything with .position/.rotation/.scale
template <typename ParticleT>
glm::mat4 build_particle_transform_of(const ParticleT& particle, const glm::vec3& center_offset);

// effects/effect_renderer.hpp — the sealed-packet form, which is the render path
glm::mat4 build_particle_transform(const bracken::scene::EffectParticleSnapshot& particle,
                                   const glm::vec3& center_offset);

class EffectRenderer {  // non-copyable, non-movable
public:
    [[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry, ModelManager* model_manager);
    void shutdown();  // also called by the destructor
    void ensure_models_uploaded(std::span<const bracken::scene::EffectEmitterSnapshot> emitters);
    void upload_particle_instances(SDL_GPUCopyPass* copy_pass,
                                   std::span<const bracken::scene::EffectEmitterSnapshot> emitters,
                                   std::span<const bracken::scene::EffectParticleSnapshot> particles);
    void draw_particle_effects(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view,
                               const glm::mat4& projection, const glm::vec3& camera_pos);
};
```

Call ordering contract: `ensure_models_uploaded` outside any pass → `upload_particle_instances` inside the frame's copy pass → `draw_particle_effects` inside the render pass, same frame (the draw consumes `batches_`/`instance_buffer_` rebuilt by the upload; cached `Model*` in batches are valid frame-locally only).

#### `build_particle_transform`

Composes `translate(position) * rotY * rotX * rotZ * scale(s) * translate(-center_offset)` directly: one sin/cos pair per axis, `R = Ry*Rx*Rz` written column-major, columns scaled by `s`, translation `position - R * (center_offset * s)`. `effect_renderer.test.cpp` pins bit-near equivalence (1e-3) against the naive chained `glm::translate/rotate/scale` reference across a position/rotation/scale/center sweep, plus the invariant that the model-space center maps exactly onto `particle.position`, plus exact agreement between the shipping `EffectParticleSnapshot` overload and the template. There is no longer a `systems::Particle` overload: the render path never sees a live particle, so the header does not depend on `systems/` — the sweep instantiates `build_particle_transform_of` against `systems::Particle` directly.

#### `init` / resources

Creates: an `SDL_GPUSampler` (linear min/mag/mip, REPEAT on u/v/w, anisotropy enabled `max_anisotropy = 4.0`, `max_lod = 1000.0` so mip chains stay reachable) and a 1×1 white RGBA8 `gpu::GPUTexture` (`0xFFFFFFFF`) bound for meshes without a base-color texture so bindings never rely on leftover state. Failure of either logs `BRACKEN_LOG_ERROR("effect", ...)` and returns false. `shutdown()` releases both, clears the model cache, batches, scratch, and the instance buffer.

#### Model resolution & upload

`resolve_emitter_model(const EmitterDefinition&)` caches `EmitterDefinition* → ModelHandle` in an `unordered_map` (definitions have stable addresses, so this replaces per-frame string hashing with one pointer-hash lookup per emitter). Empty `def.model` or a `ModelManager::get_handle` miss caches `INVALID_MODEL_HANDLE` and logs `BRACKEN_LOG_WARN("effect", "Effect emitter '{}' references unknown model '{}'", ...)` once. **The cache is never invalidated except by `shutdown()`** — registering the model after the first resolution does not retroactively fix the emitter.

`ensure_models_uploaded` walks the sealed packet's `EffectEmitterSnapshot` span, resolves each emitter's model, and calls `ModelLoader::upload_to_gpu(*device_, *model)` when `meshes[0].uploaded` is false. Must be called outside any render pass; `SceneRenderer` guarantees this pre-pass placement so first use of an effect never hitches mid-pass.

#### Instance packing (`upload_particle_instances`)

Clears `batches_`/`instance_scratch_` (capacity retained across frames), then for each snapshot emitter with a non-null definition and non-empty particle range:

- Resolve model; skip null. If `meshes[0]` is not uploaded, log `BRACKEN_LOG_WARN("effect", "Effect model '{}' not GPU-uploaded; skipping particles this frame", ...)` and skip — the mid-frame path **never uploads**, only drops for one frame.
- Compute the model-center recentering offset once per emitter: `model->bounds.center()`.
- Append one `gpu::ParticleInstanceData { build_particle_transform(particle, center), vec4(color.rgb, opacity) }` per particle (note: `particle.color.a` is discarded; `opacity` takes its place).
- Record a `ParticleBatch { const Model* model; uint32_t first_instance; uint32_t instance_count; }` — all particles of one emitter share the emitter's model, so each emitter is one contiguous range = one draw batch.

Buffer management: a single `gpu::GPUBuffer` of `Type::Storage` created via `GPUBuffer::create_dynamic`, grown geometrically to `max(required * 2, 16 * 1024)` bytes (16 KB floor ≈ 200 particles at 80 B each) whenever capacity is insufficient; the old buffer's release is deferred by SDL until the GPU finishes with it. Creation failure logs an error and clears the batches (frame renders no particles). Upload is one `instance_buffer_->update(copy_pass, data, size)` for the whole frame.

`gpu::ParticleInstanceData` (`src/bracken/gpu/gpu_uniforms.hpp`): `{ glm::mat4 model; glm::vec4 color; }` — 80 bytes, matching the HLSL `ParticleInstance { float4x4 model; float4 color; }` in the `StructuredBuffer`. `color.rgb` = tint, `color.a` = opacity (folded into rgb by the vertex stage).

#### Draw (`draw_particle_effects`)

Early-outs on null pass/cmd/device/registry, empty batches, missing buffer, or missing pipeline. Then:

1. Bind `gpu::PipelineType::Particle` pipeline.
2. Push `gpu::InstancedCameraUniforms { mat4 view; mat4 projection; vec3 cameraPos; float pad; }` once per pass to vertex slot 0 (layout shared with `model_instanced.vert.hlsl`; `cameraPos` is unused by the particle VS — no fog — kept for layout sharing).
3. Push `gpu::ParticleFragmentUniforms { float emissiveBoost; 3×pad; }` once to fragment slot 0 with `emissiveBoost = PARTICLE_EMISSIVE_BOOST = 2.0f` (file-local constexpr in `effect_renderer.cpp`; the struct's own default is 1.0f — the renderer always overrides).
4. Bind the instance storage buffer once (`SDL_BindGPUVertexStorageBuffers(pass, 0, ...)`).
5. Per batch: push 16-byte `gpu::ParticleBatchUniforms { uint32_t instanceOffset; 3×pad; }` to vertex slot 1; per mesh of the batch's model (skipping non-uploaded/empty meshes): bind vertex + 32-bit index buffers, bind the mesh's base-color texture or the dummy white (with a `last_bound_tex` cache to skip redundant binds), then `SDL_DrawGPUIndexedPrimitives(pass, mesh.index_count(), batch.instance_count, 0, 0, 0)`.

No sorting of batches or particles is performed — unnecessary under commutative additive blending with depth writes off.

### GPU contract

#### Pipeline (`PipelineType::Particle`, its row in `src/bracken/gpu/pipeline_catalog.cpp`)

- Shaders: `particle.vert.spv` / `particle.frag.spv` (entry `VSMain`/`PSMain`). VS resources: 2 uniform buffers (camera b0, batch b1), 1 storage buffer. FS resources: 1 uniform buffer, 1 sampler.
- Vertex layout: standard `with_vertex3d()` (locations: 0 position, 1 normal, 2 texcoord, 3 color, 4 tangent); the VS consumes only 0, 2, 3 (particles are unlit).
- Blend: `PipelineConfig::additive()` ⇒ `BlendMode::Additive` = color `(SRC_ALPHA, ONE)`, alpha `(ONE, ONE)` (`src/bracken/gpu/gpu_types.hpp`). Since `particle.frag` writes alpha = 1.0, the effective color blend is exactly `(ONE, ONE)`.
- Depth: test **on**, write **off** (set by `additive()`), compare overridden to `SDL_GPU_COMPAREOP_GREATER_OR_EQUAL` (reversed-Z). **Note:** the class comment in `effect_renderer.hpp` still says "depth test LESS_OR_EQUAL against opaques" — stale relative to the actual pipeline (reversed-Z `GREATER_OR_EQUAL`); the pipeline code + registry comment are authoritative.
- Cull: back-face culling on (PipelineConfig default).
- Targets: color = HDR scene format (`RGBA16F`), depth = `D32_FLOAT`.

#### Shader bindings (`src/bracken/shaders/particle.vert.hlsl`, `particle.frag.hlsl`)

| Binding | Space/set | Contents | CPU struct |
|---|---|---|---|
| `[[vk::binding(0, 1)]]` cbuffer `CameraUniforms` | set 1, b0 | `view`, `projection`, `cameraPos`, pad | `gpu::InstancedCameraUniforms` (144 B, `alignas(16)`) |
| `[[vk::binding(1, 1)]]` cbuffer `BatchUniforms` | set 1, b1 | `uint instanceOffset` + 3 pad | `gpu::ParticleBatchUniforms` (16 B, `alignas(16)`) |
| `[[vk::binding(0, 0)]]` `StructuredBuffer<ParticleInstance>` | set 0 | `{float4x4 model; float4 color;}` per particle, frame-global | `gpu::ParticleInstanceData` (80 B) |
| `[[vk::binding(0, 2)]]` combined image sampler | set 2 | base-color texture (or 1×1 white) | — |
| `[[vk::binding(0, 3)]]` cbuffer `ParticleFragmentUniforms` | set 3, b0 | `float emissiveBoost` + 3 pad | `gpu::ParticleFragmentUniforms` (16 B, `alignas(16)`) |

VS: `inst = instances[instanceOffset + SV_InstanceID]`; `clip = projection * view * inst.model * position`; output color = `float4(input.color.rgb * inst.color.rgb * inst.color.a, 1.0)` — **opacity folded into RGB** because additive blending ignores alpha. FS: `rgb = baseColor.rgb * input.color.rgb * emissiveBoost`, alpha written as 1.0 so `SRC_ALPHA` factors degenerate to `ONE`.

Bloom interplay: `PARTICLE_EMISSIVE_BOOST = 2.0` intentionally pushes typical tints past the bloom threshold (`BloomDownsampleUniforms::threshold = 0.8`, post-exposure luminance), so effects feed the bloom chain. All colors are linear-space; the composite pass tonemaps.

### Error handling summary

| Condition | Behavior |
|---|---|
| `spawn_effect(nullptr, ...)` | returns `-1`, no state change |
| Spawn command with null definition | skipped by the sim-job tail's spawn apply |
| Unknown model name at resolve | `WARN` once per emitter definition; the miss is **not cached** — the lookup retries every call, so a model registered late resolves on the next frame |
| Model not yet GPU-uploaded at pack time | `WARN` per frame, particles skipped that frame (never uploaded mid-frame) |
| Instance buffer creation failure | `ERROR`, batches cleared, no particles that frame |
| Sampler / dummy-texture creation failure in `init` | `ERROR`, `init` returns false (fatal to `SceneRenderer::init`) |
| Missing Particle pipeline at draw | silent early-out |

The layer predates the `core::Result` convention and uses `bool` + `BRACKEN_LOG_*` throughout, which matches the house rule for boot-time init chains; per-frame paths degrade to logged skips, never crashes.

### Known limitations, stubs, and silent behaviors

- **Only mesh particles exist** — there is no sprite/billboard particle type.
- **No loop flag** — completion is emitter-driven (or gated by an effect-level `duration >= 0`); the host re-spawns to loop.
- **`spread_angle` only spreads in the XZ plane** (Y-axis rotation of the initial velocity), so vertical directional emitters get no meaningful cone.
- **BURST never fires when `delay ≥` effective emitter duration** (active window is measured from effect spawn).
- **Terrain clamp is positional only** — no velocity response; particles slide along the ground.
- **One-frame latency** between `spawn_effect` and first visible particles (update runs before spawn consume in `render_frame`).
- **No particle count cap** — instance buffer grows geometrically without limit; the only per-frame plot/telemetry is `effects.particles` (Tracy) and the DevUI stats line.

---

## Text & UI Rendering

### Component map and data flow

Three layers, strictly host → data → renderer:

| Layer | Type | Location | Role |
|---|---|---|---|
| Command recording | `scene::FrameBuilder::UiLane` → `scene::FramePacket::ui_commands()` | `src/bracken/scene/frame_builder.hpp/.cpp`, `frame_packet.hpp` | Host-facing per-frame command stream (variant-based; text copied into the packet arena) |
| Primitive batching + pass execution | `render::UIRenderer` | `src/bracken/render/ui_renderer.hpp/.cpp` | CPU vertex generation for shapes, text-draw queuing, the UI render pass on the swapchain |
| Glyph atlas + text batching | `render::TextRenderer` | `src/bracken/render/text_renderer.hpp/.cpp` | SDL_ttf glyph rasterization into a shelf-packed atlas, batched textured-quad text draws |

`scene::SceneRenderer` owns one `UIRenderer` (`std::unique_ptr`, member `ui_` in `src/bracken/scene/scene_renderer.hpp`; **private, no accessor**), which in turn owns one `TextRenderer` (`text_renderer()` accessor on `UIRenderer`). Per frame, `SceneRenderer::render_frame(const FramePacket&)` runs the UI stage after the composite/FXAA passes and before the optional post-UI callback (`src/bracken/scene/scene_renderer.cpp`):

1. `begin_ui()` — late swapchain acquire if the 3D path didn't already acquire it; `ui_->begin(cmd)`.
2. Replay of `frame.ui_commands()` — `std::visit` over each `UiCommand`, forwarding to the matching `UIRenderer::draw_*`, in submission order.
3. `draw_billboard_3d(...)` for each `frame.billboards()` entry (billboards are 3D packet data drawn as UI rects — there is no billboard GPU pipeline; see `src/bracken/shaders/README.md`).
4. `end_ui()` — `ui_->end()`, then `ui_->execute(cmd, current_swapchain_, clear_background)` with `clear_background = !had_main_pass_this_frame_`.

Threading/lifetime: the whole path is main-thread-only (`BRACKEN_ASSERT_MAIN_THREAD()` at the top of `render_frame`); no class in this area has internal synchronization. The sealed packet backs every queued `string_view` until `execute()` runs (packet text lives in the packet arena, valid until retire — the caller retires the packet after `render_frame` returns).

### UI command recording — `FrameBuilder::UiLane` → `FramePacket::ui_commands()`

Command payload structs (`src/bracken/scene/frame_packet.hpp`), all trivially copyable, all colors `core::Color` (the packed-ARGB word):

```cpp
struct UiFilledRectCommand   { float x, y, w, h; uint32_t color; };
struct UiRectOutlineCommand  { float x, y, w, h; uint32_t color; float line_width = 2.0f; };
struct UiCircleCommand       { float x, y, radius; uint32_t color; int segments = 24; };
struct UiCircleOutlineCommand{ float x, y, radius; uint32_t color; float line_width = 2.0f; int segments = 24; };
struct UiLineCommand         { float x1, y1, x2, y2; uint32_t color; float line_width = 2.0f; };
struct UiTextCommand         { std::string_view text;   // points into the packet arena
                               float x{}, y{}; float scale = 1.0f; uint32_t color = 0xFFFFFFFF; };
struct UiButtonCommand       { float x{}, y{}, w{}, h{}; std::string_view label;  // packet arena
                               uint32_t color{}; bool selected = false; };

using UiCommandData = std::variant<UiFilledRectCommand, UiRectOutlineCommand, UiCircleCommand,
                                   UiCircleOutlineCommand, UiLineCommand, UiTextCommand, UiButtonCommand>;
struct UiCommand {
    UiCommandData data;
    template<typename T> bool is() const;      // holds_alternative
    template<typename T> const T& get() const; // std::get
};
```

Commands are emitted through `FrameBuilder::ui()` (an `UiLane` value handle — see the scene chapter for the full recording API: `add_filled_rect`, `add_rect_outline`, `add_circle`, `add_circle_outline`, `add_line`, `add_text`, `add_text_wrapped`, `add_button`; text parameters are `std::string_view` and are **copied into the packet arena at emit time**, so temporaries are safe). Cross-type submission order is preserved within a lane; multi-lane frames merge in deterministic (lane-index, emission-order) order at `seal()`.

`UiLane::add_text_wrapped` (`src/bracken/scene/frame_builder.cpp`) is a pure layout helper that emits multiple `UiTextCommand`s and returns total height consumed (same algorithm as the retired `UIScene`):
- `char_w = 8.0f * scale` (constant `CHAR_WIDTH_BASE = 8.0f` — an approximation, not glyph metrics; operates on **bytes**, so multi-byte UTF-8 over-counts width).
- `line_height <= 0` → `18.0f * scale`.
- `max_chars = max(1, int(max_width / char_w))`; wrap point = last `' '` at or before `pos + max_chars` (`std::string::rfind`); if none in range, hard break at `max_chars`; the space broken at is skipped.

Consumption: `SceneRenderer` maps `frame.ui_commands()` 1:1 onto `UIRenderer` calls. Note the argument reorder for text: `UiTextCommand{text,x,y,scale,color}` → `UIRenderer::draw_text(text, x, y, color, scale)`.

### `render::UIRenderer` — primitive batcher and UI pass

Declared in `src/bracken/render/ui_renderer.hpp`. Non-copyable and non-movable (all four special members deleted).

```cpp
[[nodiscard]] bool init(gpu::GPUDevice&, gpu::PipelineRegistry&, int width, int height);
void shutdown();
void set_screen_size(int width, int height);
void begin(SDL_GPUCommandBuffer* cmd);   // recording start (no GPU work)
void end();                              // documented no-op; recording end marker
void execute(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchain, bool clear_background = false);
void draw_filled_rect(float x, float y, float w, float h, uint32_t color);
void draw_rect_outline(float x, float y, float w, float h, uint32_t color, float line_width = 2.0f);
void draw_circle(float x, float y, float radius, uint32_t color, int segments = 24);
void draw_circle_outline(float x, float y, float radius, uint32_t color, float line_width = 2.0f, int segments = 24);
void draw_line(float x1, float y1, float x2, float y2, uint32_t color, float line_width = 2.0f);
void draw_textured_quad(SDL_GPUTexture* texture, float x, float y, float w, float h, uint32_t tint = 0xFFFFFFFF,
                        float u0 = 0, float v0 = 0, float u1 = 1, float v1 = 1); // texture not retained
void draw_text(std::string_view text, float x, float y, uint32_t color, float scale = 1.0f); // queues; NOT copied
void draw_button(float x, float y, float w, float h, const std::string& label, uint32_t color, bool selected);
int text_width(std::string_view text, float scale = 1.0f) const;   // 0 without a font
int text_height(float scale = 1.0f) const;
TextRenderer* text_renderer();
int width() const; int height() const;
```

#### Resources created in `init`

- Dynamic vertex buffer: `INITIAL_VERTEX_CAPACITY * sizeof(UIVertex)` = 16384 × 32 B = **512 KiB** via `gpu::GPUBuffer::create_dynamic(device, Type::Vertex, ...)` — grown geometrically inside `execute()` when a frame outgrows it (the CPU batch is unbounded).
- 1×1 white dummy texture, `SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM`, `SAMPLER` usage, uploaded via a throwaway transfer buffer on a **dedicated command buffer submitted inside `init`** (SDL3 GPU requires every declared sampler slot to be bound even when `has_texture == 0`).
- Nearest/clamp sampler for the dummy texture.
- The owned `TextRenderer` — its `init` failure is logged (`BRACKEN_LOG_ERROR("ui", "Failed to initialize text renderer")`) but **does not fail `UIRenderer::init`**; text simply never draws (`is_ready()` false).
- Projection: `glm::ortho(0, width, height, 0, -1, 1)` — Y flipped so screen top-left maps to Vulkan clip top. Rebuilt by `set_screen_size`; `SceneRenderer::begin_frame` calls `ui_->set_screen_size(context_->width(), context_->height())` every frame, and `SceneRenderer::set_screen_size` also forwards on explicit resize. (The projection matrix itself is only consumed by `TextRenderer`; the UI shape shader does its own NDC math from a screen-size uniform.)

#### Vertex format

```cpp
struct UIVertex { float x, y; float u, v; float r, g, b, a; };  // 32 bytes
```

Must match `gpu::Vertex2D` (`src/bracken/gpu/gpu_types.hpp`: `vec2 position, vec2 texcoord, vec4 color`) and the attribute locations from `get_vertex2d_attributes()`: location 0 = FLOAT2 position, 1 = FLOAT2 texcoord, 2 = FLOAT4 color. Texcoords are always written as 0 for solid shapes.

#### Recording phase

`begin(cmd)` stores `cmd`, clears `vertex_batch_` and `queued_text_draws_`, calls `text_renderer_->release_pending_resources()` (frame counter + per-frame warn-flag reset) and `set_projection(projection_)`.

Vertex generation per call (CPU, appends to `std::vector<UIVertex> vertex_batch_`):

| Call | Vertices | Notes |
|---|---|---|
| `draw_filled_rect` | 6 | two CCW-listed triangles, non-indexed |
| `draw_rect_outline` | 24 | four filled strips (top/bottom/left/right), drawn inside the rect |
| `draw_circle` | `segments * 3` | triangle fan around center; `PI = 3.14159265359f` |
| `draw_circle_outline` | `segments * 6` | ring between `radius` and `radius - line_width`; if `inner_radius <= 0` falls back to `draw_circle`; returns doing nothing when `line_width <= 0 || radius <= 0 || segments <= 0` |
| `draw_line` | 6 | quad extruded ±`line_width/2` along the normal; silently dropped when length < 0.001 |
| `draw_button` | 30 + queued text | filled rect + outline (border `0xFFFFFFFF`/3 px when selected, `0xFF888888`/2 px otherwise) + centered label queued via `draw_text` at color `0xFFFFFFFF`, scale 1.0, centered with `TextRenderer::get_text_width/height` |

Color decoding: every packed color/tint is a `core::Color` and decodes through `Color::to_linear()` (== `core::unpack_color`, `bracken/core/color.hpp`) — the **single engine-wide ARGB `0xAARRGGBB`** convention shared by UI primitives, textured-quad tints, text, buttons, 3D billboards, and debug lines. `TextRenderer` still takes the raw `uint32_t` one level down; `UIRenderer::draw_text` passes `color.argb`.

`draw_text` appends `QueuedTextDraw{ std::string_view text; float x, y; uint32_t color; float scale; }` — the view is **not copied**; the characters must remain valid until `execute()` returns. Packet-arena text satisfies this because the sealed `FramePacket` stays alive (Consuming) through `render_frame` and is only retired afterwards.

**Overflow behavior:** the CPU batch is unbounded; `execute()` grows the GPU vertex buffer geometrically when the frame's batch outgrows it (a new `create_dynamic` buffer — legal inside an open copy pass). Only if that growth allocation fails is the batch truncated to the old capacity, with a `Failed to grow UI vertex buffer` warning.

#### `execute(cmd, swapchain, clear_background)` — phase order

Copy passes must precede the render pass (SDL3 GPU forbids nesting), hence the strict phasing:

1. Forward every `QueuedTextDraw` to `TextRenderer::queue_text_draw` (builds glyph quads CPU-side; may rasterize new glyphs into the CPU atlas).
2. Copy passes: `vertex_buffer_->update(...)` for the primitive batch; `text_renderer_->create_pending_textures(cmd)` (= atlas dirty-band upload); `text_renderer_->upload_queued_text(cmd)` (atlas flush no-op + text vertex upload).
3. Render pass on the swapchain: `load_op = clear_background ? CLEAR : LOAD`, `store_op = STORE`, `clear_color = {0.1f, 0.1f, 0.15f, 1.0f}`; full-screen viewport (depth 0–1) and scissor; no depth-stencil target.
4. Bind `PipelineRegistry::get_pipeline(PipelineType::UI)` — if unavailable, logs via `SDL_Log`, ends the pass, clears state, returns (frame keeps whatever was on the swapchain).
5. Push `gpu::UIScreenUniforms{width, height}` (vertex slot 0).
6. Draw the primitive batch as one `SDL_DrawGPUPrimitives` per **texture-switch range** (`DrawRange`): solid primitives and `draw_textured_quad` ranges interleave in submission order; `UIFragmentUniforms{hasTexture}` is re-pushed and the fragment sampler re-bound only when the range's texture state changes (solid ranges bind the 1×1 white dummy).
7. `text_renderer_->draw_queued_text(cmd, render_pass)` (binds the Text pipeline internally).
8. End pass; clear `vertex_batch_`, `draw_ranges_`, `queued_text_draws_`.

Consequence for z-order: **all primitives (including billboards) draw before all text** — text cannot be occluded by a later-submitted shape.

`shutdown()` releases the vertex buffer, dummy texture/sampler, and shuts down/destroys the `TextRenderer`; the destructor calls `shutdown()`.

### `render::TextRenderer` — glyph atlas and batched text

Declared in `src/bracken/render/text_renderer.hpp`. Constants:

| Constant | Value | Meaning |
|---|---|---|
| `ATLAS_SIZE` | 512 | atlas is 512×512 texels, one atlas, one font size |
| `GLYPH_PADDING` | 1 | gap added to each packed cell (linear-filter bleed guard) |
| `VERTICES_PER_QUAD` | 6 | non-indexed two-triangle quads |
| `FLOATS_PER_VERTEX` | 4 | `x, y, u, v` |
| `MAX_TEXT_QUADS` | 4096 | per-frame glyph-quad budget across all strings |
| `font_size_` | 18 | hardcoded member default; no setter |

#### `decode_utf8` (public inline free function, header)

`uint32_t decode_utf8(std::string_view text, size_t& i)` — decodes one codepoint at byte `i`, advancing `i`. Handles 1–4 byte sequences by lead-byte pattern. Malformed input (stray continuation byte, invalid lead, bad continuation) yields U+FFFD and advances exactly one byte (guaranteed progress / resync); a truncated trailing sequence yields U+FFFD and sets `i = text.size()`. No overlong/surrogate/max-codepoint validation. Fully covered by `src/bracken/render/text_renderer.test.cpp` (`DecodeUtf8.*`).

#### `ShelfPacker` (public class, header)

Left-to-right shelf packer for a fixed `width × height` area: rects are placed on the current shelf at `cursor_x_`; when `cursor_x_ + w > width` a new shelf opens at `shelf_y_ += shelf_height_` (shelf height = max rect height on that shelf). `pack(w, h, out_x, out_y)` returns false for non-positive or atlas-exceeding dims, or when `shelf_y_ + h > height` (full). `reset()` reclaims everything. O(1) per pack; freed space is never reclaimed individually — eviction is all-or-nothing. Covered by `ShelfPacker.*` tests in the same file.

#### Initialization and font loading

`init(gpu::GPUDevice&, gpu::PipelineRegistry&)`:
1. `TTF_Init()` — failure is fatal (returns false).
2. Probes a hardcoded font path array in order, `TTF_OpenFont(path, 18)`:
   `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`, `/usr/share/fonts/TTF/DejaVuSans.ttf`, `/usr/share/fonts/dejavu/DejaVuSans.ttf`, `/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf`, `/usr/share/fonts/truetype/freefont/FreeSans.ttf`, `/System/Library/Fonts/Helvetica.ttc`, `C:\Windows\Fonts\arial.ttf`.
   No font found → `BRACKEN_LOG_WARN`, **non-fatal**: init proceeds, `is_ready()` returns false (`initialized_ && font_ != nullptr`), all draw/queue/measure calls no-op.
3. Dynamic vertex buffer: `MAX_TEXT_QUADS * 6 * 4 * sizeof(float)` = **384 KiB**.
4. Atlas sampler: LINEAR min/mag, NEAREST mipmap, CLAMP_TO_EDGE ×3.
5. Atlas texture: 2D, `SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM`, 512×512, 1 level, `SAMPLER` usage. Persistent `UPLOAD` transfer buffer of 512·512·4 = **1 MiB**. CPU mirror `std::vector<uint8_t> atlas_pixels_` (1 MiB) zero-filled; the whole atlas is marked dirty so the first flush uploads defined zeroes everywhere (incl. inter-cell padding).
   GPU-resource failures are fatal (logged, return false).

`shutdown()` releases atlas texture/transfer/sampler, clears CPU state, resets the vertex buffer, `TTF_CloseFont` + `TTF_Quit`. Destructor calls it.

#### Glyph cache (`ensure_glyph`)

`struct Glyph { uint16_t atlas_x, atlas_y, width, height; int advance; bool has_bitmap; }` keyed by codepoint in `std::unordered_map<uint32_t, Glyph> glyphs_`.

On miss (returns `nullptr` only when `font_ == nullptr`):
1. `TTF_GetGlyphMetrics` → `advance`.
2. `TTF_RenderGlyph_Blended(font_, cp, white)` → surface converted to `SDL_PIXELFORMAT_RGBA32`. The blended cell is **full line-height with origin at the pen position at the top of the line box**, so layout is simply "place cell at pen, advance" — no per-glyph bearing math.
3. Alpha-scan: a fully transparent cell (space) is cached with `has_bitmap = false` and consumes **no atlas space** (advance-only).
4. Visible glyphs pack as `(w + 1) × (h + 1)` cells (`GLYPH_PADDING`). On pack failure: `evict_all_glyphs()` — warn log (`Glyph atlas overflow (...) - evicting all; text may glitch for one frame`), clear the map, `packer_.reset()`, zero the CPU pixels, mark all 512 rows dirty — then retry once. Quads already emitted this frame keep stale UVs for one frame.
5. Blit rows into the CPU mirror (`blit_glyph_to_atlas`), extend the dirty row band (`mark_dirty_rows`).
6. The glyph is cached even if rasterization failed (advance-only entry), so failures aren't retried per frame.

#### Atlas upload (`flush_atlas`)

Dirty tracking is a single row band `[dirty_y0_, dirty_y1_)` (merged, not a rect list). Flush (no-op when clean): map the persistent transfer buffer with `cycle = true` (previous frame's upload may still be in flight), memcpy the band, one `SDL_UploadToGPUTexture` in a copy pass with `cycle = false` (preserves previously uploaded regions), reset the band. Exposed as `create_pending_textures(cmd)` (profiled zone) and implicitly via `upload_queued_text`. Must run **outside** render passes.

#### Quad building (`build_string_quads`)

Pen advances in **unscaled** font pixels; per glyph: kerning via `TTF_GetGlyphKerning(prev, cp)` added to pen, then for `has_bitmap` glyphs a quad at `x0 = x + pen*scale, y0 = y`, size `glyph.{width,height} * scale`, UVs = atlas texel coords × `1/512`, appended as 6 vertices × 4 floats to `out`. Stops when `out.size() + 24 > max_floats` — the shared `MAX_TEXT_QUADS` budget — logging `Text quad budget (4096) exhausted - dropping glyphs this frame` at most once per frame (`batch_full_warned_`, reset by `release_pending_resources()`). Returns quads appended. Rasterizes missing glyphs as a side effect.

#### Batched path (the one actually used)

```cpp
void queue_text_draw(std::string_view text, float x, float y, uint32_t color = 0xFFFFFFFF, float scale = 1.0f);
void upload_queued_text(SDL_GPUCommandBuffer* cmd);   // BEFORE the render pass (copy pass)
void draw_queued_text(SDL_GPUCommandBuffer* cmd, SDL_GPURenderPass* pass); // DURING the pass
```

- `queue_text_draw` appends quads to the shared `batch_vertices_` staging vector and records `TextRun{ size_t first_vertex; uint32_t vertex_count; uint32_t color; }`. New glyphs land in the CPU atlas immediately, so **new strings render the same frame** (uploaded by the next copy-pass flush).
- `upload_queued_text` = `flush_atlas` + one `vertex_buffer_->update` for all runs.
- `draw_queued_text` binds `PipelineRegistry::get_pipeline(PipelineType::Text)` once, pushes the projection (vertex uniform slot 0), binds the atlas+sampler (fragment sampler slot 0) and the vertex buffer once, then walks runs **merging contiguous same-color runs into single draws**; the color uniform (fragment slot 0) is pushed only on change. Draws are `SDL_DrawGPUPrimitives(count, 1, first_vertex, 0)`. Clears `queued_runs_` and `batch_vertices_` afterward (also on every early-out).
- Text color decoding: `core::unpack_color(color)` — the engine-wide **ARGB** convention, identical to the shape primitives.

#### Measurement

`get_text_width(text, scale)` = `TTF_GetStringSize(font_, ...)` width × scale, truncated to `int`; `get_text_height(scale)` = `TTF_GetFontHeight(font_)` × scale. Both return 0 with no font. Hosts reach these through `UIRenderer::text_width`/`text_height` and the `SceneRenderer::text_width`/`text_height` passthroughs (0 before init / without a font); `UiLane::add_text_wrapped` deliberately does not use them.

### GPU contracts

#### Pipelines (`src/bracken/gpu/pipeline_registry.hpp/.cpp`)

- `PipelineType::UI` → `create_ui_pipeline()`: shaders `ui.vert.spv` / `ui.frag.spv` (entry `VSMain`/`PSMain`), VS resources: 1 uniform buffer; FS: 1 uniform buffer + 1 sampler. Config: `with_vertex2d().alpha_blended().no_depth().no_cull()`, `color_format = swapchain_format_` (default `SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM`).
- `PipelineType::Text` → `create_text_pipeline()`: shaders `text.vert.spv` / `text.frag.spv`, same resource counts. Explicit vertex layout: one buffer, pitch `4 * sizeof(float)` = 16 B, attributes `{0: FLOAT2 position @0, 1: FLOAT2 texcoord @8}`; `alpha_blended().no_depth().no_cull()`, swapchain color format, no depth target.
- Blend state for `BlendMode::Alpha` (`get_blend_state`, `src/bracken/gpu/gpu_types.hpp`): color `SRC_ALPHA / ONE_MINUS_SRC_ALPHA / ADD`, alpha `ONE / ONE_MINUS_SRC_ALPHA / ADD`. (`alpha_blended()` sets `depth_test_enable = true`, but the subsequent `no_depth()` disables both test and write.)
- Both pipelines are invalidated and lazily recreated when `PipelineRegistry::set_swapchain_format` changes the swapchain format.

#### Shaders and uniform ABI

`src/bracken/shaders/ui.vert.hlsl` — input matches `Vertex2D` via `[[vk::location(0/1/2)]]`. Uniform (set 1, binding 0 — SDL3 GPU vertex-uniform set):

```hlsl
cbuffer UIUniforms { float2 screen_size; float2 _padding; };
```

CPU mirror `gpu::UIScreenUniforms` (`src/bracken/gpu/gpu_uniforms.hpp`): `struct alignas(16) { float width, height; float _padding[2]; }`. The VS maps pixels to NDC itself: `ndc.x = x/(w*0.5) - 1; ndc.y = 1 - y/(h*0.5)`; z = 0.

`src/bracken/shaders/ui.frag.hlsl` — uniform (set 3, binding 0):

```hlsl
cbuffer UIFragmentUniforms { int has_texture; int _padding0; int _padding1; int _padding2; };
```

CPU mirror `gpu::UIFragmentUniforms`: `alignas(16) { int hasTexture; int _padding0..2; }` — padding is three **named scalars**, not an array, because an HLSL cbuffer `int` array would stride at 16 B and break the byte-for-byte contract. Combined image sampler at set 2, binding 0 (`ui_texture`/`ui_sampler`); output = vertex color, ×texture sample when `has_texture == 1`. Solid ranges push `hasTexture = 0` with the 1×1 white dummy bound; `draw_textured_quad` ranges push `hasTexture = 1` with the caller's texture.

`src/bracken/shaders/text.vert.hlsl` — vertex uniform (set 1, b0) `cbuffer TextUniforms { float4x4 projection; }`; pushed as a raw `glm::mat4` (the `UIRenderer` ortho). `text.frag.hlsl` — fragment uniform (set 3, b0) `cbuffer TextColorUniforms { float4 textColor; }`; atlas combined sampler at set 2, b0; output = `textColor * sample` (glyphs are rasterized white, so tinting is a multiply; alpha comes from the glyph coverage).

### 3D billboards (`SceneRenderer::draw_billboard_3d`)

Data: `scene::BillboardDrawCommand { float world_x, world_y, world_z, width, fill_ratio; uint32_t fill_color, bg_color, frame_color; }`, submitted via `FrameBuilder::add_billboard_3d(...)` (`src/bracken/scene/frame_packet.hpp` / `frame_builder.hpp`).

Algorithm (`src/bracken/scene/scene_renderer.cpp`):
1. `clip = P · V · world`; cull when `clip.w <= 0.01` (behind/at camera).
2. NDC = clip.xyz / w; cull when `|ndc.x| > 1.5 || |ndc.y| > 1.5 || ndc.z ∉ [-1, 1]` (generous margins so bars don't pop at screen edges).
3. Screen: `sx = (ndc.x·0.5 + 0.5)·width_px`, `sy = (1 − (ndc.y·0.5 + 0.5))·height_px`.
4. `distance_scale = clamp(100 / clip.w, 0.3, 1.5)`; bar size = `width·2·scale × width·0.4·scale`, centered at (sx, sy).
5. Three `UIRenderer::draw_filled_rect` calls: frame (bar inflated 1 px each side, `frame_color`), background (`bg_color`), fill (`bar_w · fill_ratio`, `fill_color`). Colors are ARGB.

Billboards therefore batch with — and draw after — the frame's 2D UI primitives, and under all text. No textures or labels; fixed 5:1 bar aspect derived from `width`.

### Error handling summary

- `init` chains return `bool` + `BRACKEN_LOG_ERROR` (boot-time-fatal style per project conventions); missing font is the one deliberate non-fatal init condition.
- Runtime failures never throw or crash: null cmd/pass/pipeline, unready text renderer, or missing font all early-return, mostly silently; buffer-overflow and atlas conditions log (`SDL_Log` in `gpu_buffer.cpp`/`ui_renderer.cpp`, `BRACKEN_LOG_WARN/ERROR` in `text_renderer.cpp` — logging style is mixed in this module).
- No `core::Result` usage anywhere in this area.

### Performance characteristics

- All geometry is CPU-generated per frame (no retained widgets); cost is linear in emitted vertices. Uploads: one `SDL_UploadToGPUBuffer` for primitives (≤512 KiB), one for text (≤384 KiB), plus the atlas dirty band only when new glyphs appeared.
- Draw calls: 1 for all primitives + billboards, plus one per color-change/discontinuity among text runs (contiguous same-color runs merge — common case of many same-color labels is 1–3 draws).
- Steady-state text is pure vertex generation: glyph rasterization happens once per codepoint, uploads only on atlas changes.
- Dynamic buffers and the atlas transfer buffer map with `cycle = true`, so per-frame updates don't stall on the previous frame's GPU reads.

### Known limitations, stubs, and silent behaviors

- **Text quad budget 4096/frame**: excess glyphs dropped with a once-per-frame warning.
- **Fixed layering**: all text over all primitives; billboards over the frame's 2D UI shapes.
- **Hardcoded font list, single 18 px size**: no font/size API; `scale` is bitmap magnification; no-font systems silently render no text.
- **Atlas overflow** evicts everything; earlier-emitted quads keep stale UVs for one frame (logged).
- **`add_text_wrapped` uses an 8 px/char byte-count approximation**, not glyph metrics.
- **No per-widget scissor/clipping**: the pass sets one full-screen scissor.
- `UIRenderer::draw_text`'s `string_view` lifetime contract (valid until `execute()`) is invisible at the packet level (packet text outlives the frame) but binding for direct callers; `draw_textured_quad`'s texture pointer likewise must stay valid until `execute()`.

---

## Camera Controller & Camera System

### Purpose and files

The camera subsystem produces per-frame `view` / `projection` matrices and a camera position for `scene::SceneRenderer`, implementing a spring-smoothed third-person orbit/follow camera. It is deliberately game-agnostic: the public interface exposes only generic framing, smoothing, shake/trauma, focus, and FOV-bias primitives; game feel is composed by the host (`docs/engine_design_notes.md`, "Architecture: the engine is game-agnostic").

| File | Contents |
|---|---|
| `src/bracken/systems/camera_controller.hpp` | `CameraController` abstract interface, `CameraModeConfig`, `ShakeType` |
| `src/bracken/systems/camera_system.hpp` | `CameraSystem` concrete implementation, `CameraShake` |
| `src/bracken/systems/camera_system.cpp` | update pipeline, spring math, projection construction |
| `src/bracken/systems/camera_system.test.cpp` | behavioral tests (convergence, shake, clamps, terrain, projection aspect) |
| `src/bracken/contracts/camera_state.hpp` | `scene::CameraState` — the camera→renderer data contract |
| `src/bracken/contracts/camera_state.test.cpp` | pins the reversed-Z projection idiom and near-plane recovery formula |

Everything is in namespace `bracken::systems` except `CameraState` (`bracken::scene`).

### Public API surface

#### `scene::CameraState` (`src/bracken/contracts/camera_state.hpp`)

```cpp
struct CameraState {
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 view_projection;   // convention: projection * view
    glm::vec3 position;
};
```

Plain aggregate, no invariants enforced. Snapshotted by value into the frame via `FrameBuilder::set_camera(const CameraState&)` and read back through `FramePacket::camera()` in `SceneRenderer::render_frame`. `SceneRenderer` recovers the near plane from a reversed-Z projection as `m[3][2] / (m[2][2] + 1)` (pinned in `camera_state.test.cpp`, `ReversedZKeepsDistantDepthsDistinct`).

#### `CameraModeConfig` (`src/bracken/systems/camera_controller.hpp`)

Aggregate with defaults: `distance = 280.0f`, `height_offset = 90.0f`, `shoulder_offset = 40.0f`, `fov = 55.0f`, `position_lag = 0.001f`, `rotation_lag = 0.001f`, `look_ahead_dist = 60.0f`, `pitch_min = -70.0f`, `pitch_max = 70.0f`, `auto_return_speed = 1.5f`, `auto_center_enabled = true`, `auto_center_delay = 2.0f`, `zoom_min = 80.0f`, `zoom_max = 600.0f`, `near_plane = 5.0f`, `far_plane = 15000.0f`, `ground_clearance = 25.0f`, `sprint_fov_bonus = 8.0f`, `look_ahead_speed_ref = 300.0f`, `sprint_speed_ref = 400.0f`. The header notes defaults are tuned for ~1 unit = 1 cm world scale.

The two `*_lag` fields are consumed as the `smoothTime` argument of a critically-damped spring — a *smaller* value means a *faster* (less lagged) response; the defaults (0.001 s) are effectively instantaneous.

#### `ShakeType`

`enum class ShakeType { Impact, Heavy, Directional, Subtle }`.

#### `CameraController` (abstract interface)

Pure-virtual except `get_camera_state()`. Key signatures:

```cpp
class CameraController {
public:
    virtual ~CameraController() = default;
    virtual void set_target(const glm::vec3& position) = 0;
    virtual void set_target_velocity(const glm::vec3& velocity) = 0;
    virtual void set_yaw(float degrees) = 0;
    virtual void set_pitch(float degrees) = 0;
    virtual void rotate_yaw(float delta_degrees) = 0;
    virtual void rotate_pitch(float delta_degrees) = 0;
    virtual void adjust_zoom(float delta) = 0;
    virtual void set_focus_target(const glm::vec3* target) = 0;   // nullptr clears
    virtual void set_focus_strength(float strength_0_1) = 0;
    virtual float get_focus_strength() const = 0;
    virtual void set_config(const CameraModeConfig& config) = 0;
    virtual const CameraModeConfig& get_config() const = 0;
    virtual void add_shake(ShakeType type, float intensity, float duration = 0.3f) = 0;
    virtual void add_directional_shake(const glm::vec3& direction, float intensity, float duration = 0.2f) = 0;
    virtual void set_fov_bias(float fov_delta) = 0;
    virtual float get_fov_bias() const = 0;
    using TerrainHeightFunc = std::function<float(float x, float z)>;
    virtual void set_terrain_height_func(TerrainHeightFunc func) = 0;
    virtual void set_screen_size(int width, int height) = 0;
    virtual void update(float dt) = 0;
    virtual glm::mat4 get_view_matrix() const = 0;
    virtual glm::mat4 get_projection_matrix() const = 0;
    virtual glm::vec3 get_position() const = 0;
    virtual glm::vec3 get_forward() const = 0;
    virtual scene::CameraState get_camera_state() const;  // default impl:
        // view/projection/position from the getters, view_projection = P * V
};
```

The interface has **no yaw/pitch getters**; `dev::detail::draw_camera_tab` (`src/bracken/dev/dev_panels.cpp`) derives them from `get_forward()`: `pitch = degrees(asin(clamp(fwd.y, -1, 1)))`, `yaw = degrees(atan2(-fwd.x, -fwd.z))` — the same yaw convention as `camera_system.cpp`.

#### `CameraSystem` (concrete, `: public CameraController`)

Additional public API beyond the interface:

```cpp
float get_input_yaw() const;          // raw, pre-smoothing
float get_input_pitch() const;
using CollisionCheckFunc = std::function<bool(const glm::vec3& from, const glm::vec3& to, glm::vec3& hit_point)>;
void set_collision_func(CollisionCheckFunc func);   // obstacle/LOS avoidance
glm::vec3 get_target_position() const;  // returns smoothed_target_
glm::vec3 get_right() const;
float get_yaw() const;                  // smoothed
float get_pitch() const;                // smoothed
float get_current_fov() const;
float get_current_distance() const;
```

`Application` (`src/bracken/application.hpp`) owns a private `std::unique_ptr<systems::CameraSystem> camera_` (constructed in `Application::init_renderer`, `src/bracken/application.cpp`) and exposes only `protected: systems::CameraController& camera()`. Consequently `set_collision_func` and all concrete-only getters are unreachable through the `Application` facade.

#### `CameraShake` (`src/bracken/systems/camera_system.hpp`)

```cpp
struct CameraShake {
    ShakeType type;
    float intensity;
    float duration;
    float elapsed;
    glm::vec3 direction;  // directional shakes only
};
```

Per-type oscillation character comes from hard-coded sine multipliers in `update_camera_shake` — there is no per-shake frequency field.

Stored in `std::deque<CameraShake> active_shakes_` with `static constexpr size_t MAX_ACTIVE_SHAKES = 8`.

### Data / state layout

All state is plain members of `CameraSystem` (no heap besides the deque and the two `std::function`s). Angles are stored in degrees. The system keeps parallel *input* vs *current (smoothed)* values plus per-value spring velocities: yaw, pitch, distance, target position, look-at, camera position, FOV, look-ahead offset, soft-lock strength. Initial values of note (all in `camera_system.hpp`): `input_pitch_ = current_pitch_ = 25.0f`, `input_yaw_ = current_yaw_ = 0.0f`, `input_distance_ = current_distance_ = 250.0f` (immediately overwritten by the constructor to `config.distance = 280`), `screen_width_ = 1280`, `screen_height_ = 720`, matrices identity.

The constructor (`camera_system.cpp`) copies the default `CameraModeConfig` and seeds `input_distance_`, `current_distance_`, `current_fov_` from it.

### Update pipeline

`CameraSystem::update(float dt)` (profiled as Tracy zone `"CameraSystem::update"`, color `0x3D6E9E`) first clamps `dt = min(dt, 0.1f)`, then runs fixed stages in order (shake deliberately runs **before** collision avoidance so the terrain/obstacle clamps contain the shaken position):

1. `update_input_smoothing` — tracks `time_since_input_` (reset when `had_input_this_frame_`; set by `rotate_yaw`/`rotate_pitch`/`adjust_zoom` always, and by `set_yaw`/`set_pitch` when the value changes); springs `current_yaw_ → input_yaw_` (`smooth_damp_angle`, smoothTime `config.rotation_lag`), `current_pitch_ →` pitch clamped to `[pitch_min, pitch_max]` (smoothTime `rotation_lag`), and `current_distance_ → input_distance_ + collision_distance_offset_` (smoothTime **0.15**).
2. `update_look_ahead` — `speed = |target_velocity_|`; direction only when `speed > 1`; `look_ahead_factor = min(speed / max(config.look_ahead_speed_ref, 0.001), 1)`; desired offset `= dir * config.look_ahead_dist * factor`; sprung with smoothTime **0.3**.
3. `update_auto_centering` — skipped unless `config.auto_center_enabled`; activates when `time_since_input_ > config.auto_center_delay` (default **2.0 s**); no-op when `speed < 10`; computes `move_yaw = atan2(-v.x, -v.z) * 180 / 3.14159`; blends `input_yaw_ += shortest_diff * config.auto_return_speed * dt`; wraps into `[0, 360)`. (Not framerate-independent: a linear `rate*dt` blend, not a spring.)
4. `update_soft_lock` — when `focus_target_ == nullptr || focus_strength_ <= 0`: springs `soft_lock_strength_ → 0` (smoothTime 0.3) and returns. Otherwise springs `soft_lock_strength_ → focus_strength_` with smoothTime `soft_lock_blend_speed_` (**3.0** — named "speed" but consumed as a smoothTime, i.e. a slow ~3 s ramp); computes target yaw from **unsmoothed** `target_position_` to `*focus_target_` and applies `input_yaw_ += shortest_diff * soft_lock_strength_ * dt * 2.0`.
5. `update_camera_position` — springs `smoothed_target_ → target_position_` (smoothTime `config.position_lag`); `terrain_y = get_terrain_height_(x, z)` or 0; `base_look_at = (smoothed_target_.x, terrain_y + config.height_offset, smoothed_target_.z)` (**target Y discarded**); `look_at_target_ = base_look_at + look_ahead_offset_`; springs `current_look_at_ → look_at_target_` (smoothTime `position_lag * 0.5`). Spherical offset: `horizontal = distance*cos(pitch)`, `vertical = distance*sin(pitch)`, `offset = (sin(yaw)*horizontal, vertical, cos(yaw)*horizontal)` plus shoulder offset along the yaw right-vector `(cos(yaw), 0, -sin(yaw)) * config.shoulder_offset`; `ideal_camera_pos_ = current_look_at_ + offset`; springs `current_camera_pos_ → ideal_camera_pos_` (smoothTime `position_lag`). Yaw 0 places the camera at +Z of the target looking toward −Z; positive pitch raises the camera (looking down).
6. `update_camera_shake` — zeroes `shake_offset_`, iterates the deque, advances `elapsed`, erases expired entries. Falloff: `intensity_eff = intensity * (1 − progress²)`. Per-type offset with `time = elapsed`:
   - `Impact`: `x = sin(40t)·i·0.5`, `y = cos(45t)·i·0.3`, `z = sin(35t + 1)·i·0.2`
   - `Heavy`: `x = sin(12t)·i·0.4`, `y = cos(10t)·i·0.3`, `z = sin(14t + 0.5)·i·0.2`
   - `Directional`: `direction * sin(25t)·i·0.4`
   - `Subtle`: `x = sin(2t)·i·0.1`, `y = cos(1.5t)·i·0.08` (z stays 0)
   All offsets accumulate into `shake_offset_`, applied to the camera position before the collision stage.
7. `update_collision_avoidance` — starts from `final_camera_pos_ = current_camera_pos_ + shake_offset_`, so the clamps below also contain shake displacement.
   - *Terrain* (only if `get_terrain_height_`): samples clearance at the camera plus 6 points along the look-at→camera ray at `t = 0.2 + 0.15*i` for `i ∈ [0,6)` (0.2, 0.35, 0.5, 0.65, 0.8, 0.95 — integer counter to avoid float-accumulation skipping); if `min_clearance < config.ground_clearance` (default **25.0**) raises `final_camera_pos_.y` by the deficit; then a final absolute floor `final.y ≥ terrain(final.xz) + ground_clearance`.
   - *Obstacles* (only if `check_collision_` set): on a blocked segment `current_look_at_ → final_camera_pos_`, places the camera at `safe_distance = max(hit_distance − 20, config.zoom_min * 0.3)` along the boom; tracks `collision_distance_offset_` via `glm::mix(current, desired, collision_pull_in_speed_ * dt)` (**15.0**) when pulling in, and decays it with `glm::mix(current, 0, collision_push_out_speed_ * dt)` (**5.0**) when clear. With no collision func, the offset decays with factor `3.0 * dt`. (These `mix(a, b, rate*dt)` blends are framerate-dependent and would overshoot if `rate*dt > 1`; with the 0.1 s dt clamp, max factors are 1.5/0.5/0.3 — the pull-in can overshoot below ~10 FPS.)
8. `update_dynamic_fov` — `target_fov = config.fov + config.sprint_fov_bonus * f²` where `f = min(speed / max(config.sprint_speed_ref, 0.001), 1)`, plus `fov_bias_`; sprung into `current_fov_` with smoothTime **0.2**.
9. `compute_matrices` — `camera_forward_ = normalize(current_look_at_ − final_camera_pos_)`; `camera_right_ = normalize(cross(forward, (0,1,0)))` (degenerate if forward ≈ ±Y; pitch clamps at ±70° prevent this in practice, but a shake or terrain lift could in principle align them); `camera_up_ = normalize(cross(right, forward))`; `view_matrix_ = glm::lookAt(final_camera_pos_, current_look_at_, (0,1,0))`; `projection_matrix_ = glm::perspective(radians(current_fov_), aspect, config.far_plane, config.near_plane)` — **near/far deliberately swapped** (reversed-Z; see below).

Finally `had_input_this_frame_ = false`.

#### Input mutators

- `rotate_yaw(d)`: wraps `input_yaw_` into `[0, 360)`; always marks input. `set_yaw(v)`: same wrap; marks input only when the wrapped value differs from the current `input_yaw_` — hosts that pipe absolute mouse angles through `set_yaw` every frame no longer permanently suppress auto-centering.
- `rotate_pitch(d)`: clamps to `[config.pitch_min, config.pitch_max]`; always marks input. `set_pitch(v)`: same clamp; marks input only on change.
- `adjust_zoom(d)`: `input_distance_ = clamp(input_distance_ + d, config.zoom_min, config.zoom_max)`; marks input.
- `set_config(c)`: replaces `current_config_` and sets `input_distance_ = clamp(c.distance, c.zoom_min, c.zoom_max)` (silently discarding user zoom).
- `set_target` / `set_target_velocity`: raw stores.
- `add_shake(type, intensity, duration = 0.3f)`: if `size() >= 8`, `pop_front()` (oldest evicted); pushes a shake with `direction = 0`.
- `add_directional_shake(dir, intensity, duration = 0.2f)`: same eviction; `direction = |dir| > 0.001 ? normalize(dir) : (1,0,0)`.
- `set_screen_size(w, h)`: clamps both to ≥ 1 (prevents NaN aspect; pinned by test `SetScreenSizeClampsDegenerateDimensions`).
- `set_focus_target(ptr)`: raw store. `set_focus_strength(s)`: **clamped to [0, 1]** (values above 1 would over-rotate and oscillate); `get_focus_strength()` reads it back.
- `set_fov_bias(f)` / `set_terrain_height_func(f)` / `set_collision_func(f)`: raw stores; `get_fov_bias()` reads the bias back.

### Spring-damper algorithm

`smooth_damp` (vec3), `smooth_damp_float`, `smooth_damp_angle` (`camera_system.cpp`) implement the Game Programming Gems 4 critically-damped spring:

```cpp
smoothTime = max(0.0001f, smoothTime);
omega = 2 / smoothTime;
x = omega * dt;
exp_factor = 1 / (1 + x + 0.48*x² + 0.235*x³);   // Padé approx of e^-x
delta = current - target;
temp = (velocity + delta * omega) * dt;
velocity = (velocity - temp * omega) * exp_factor;
return target + (delta + temp) * exp_factor;
```

`smooth_damp_angle` first reduces the target into the shortest signed arc (±180°) from `current`, runs the float version, then renormalizes the result into `[0, 360)`. Velocity state lives in caller-owned reference parameters (the per-value `*_velocity_` members). No max-speed clamp (unlike Unity's `SmoothDamp`).

### Projection / depth contract

`compute_matrices` builds the **only** projection the engine's scene path uses: glm `RH_ZO` (via the PUBLIC `GLM_FORCE_DEPTH_ZERO_TO_ONE` define), FOV = smoothed dynamic FOV (degrees→radians), aspect = `screen_width_ / screen_height_`, and **near/far swapped**: `glm::perspective(fov, aspect, 15000.0f, 5.0f)`. Near plane 5 maps to device depth 1, far plane 15000 to depth 0 (reversed-Z), spreading float32 precision across the range; scene pipelines clear depth to 0.0 and compare `GREATER_OR_EQUAL` (`docs/architecture.md`, "Coordinate, depth, and color conventions"). `camera_state.test.cpp` pins: reversed depth mapping, distinct float32 depths for surfaces 1 unit apart at 10 000 units, and the near-plane recovery formula `n = m[3][2] / (m[2][2] + 1)` used by `scene_renderer.cpp`. Near/far come from `config.near_plane` / `config.far_plane` (defaults 5 / 15000).

### Constants (hard-coded in `camera_system.hpp` / `.cpp` unless marked config)

| Constant | Value | Where / role |
|---|---|---|
| `MAX_ACTIVE_SHAKES` | 8 | `camera_system.hpp`; deque cap, oldest evicted |
| zoom clamp | `config.zoom_min` / `config.zoom_max` (80 / 600) | config |
| auto-center delay | `config.auto_center_delay` (2.0 s) | config; idle time before auto-center |
| auto-center speed threshold | 10 u/s | below it, no centering |
| look-ahead speed normalizer | `config.look_ahead_speed_ref` (300 u/s) | config |
| look-ahead smoothTime | 0.3 s | stage 2 |
| distance smoothTime | 0.15 s | stage 1 |
| FOV smoothTime | 0.2 s | stage 8 |
| sprint FOV bonus | `config.sprint_fov_bonus` (8.0°) | config; speed-based FOV widening |
| FOV speed normalizer | `config.sprint_speed_ref` (400 u/s) | config; quadratic factor |
| ground clearance | `config.ground_clearance` (25.0) | config; terrain avoidance |
| terrain ray samples | 6 at t = 0.2 + 0.15·i | stage 7 |
| obstacle standoff | 20.0 | pull-in distance before hit point |
| `collision_pull_in_speed_` / `collision_push_out_speed_` | 15.0 / 5.0 | `mix` rates |
| no-collision-func decay rate | 3.0 | `mix` rate |
| `soft_lock_blend_speed_` | 3.0 | smoothTime for strength ramp |
| soft-lock decay smoothTime | 0.3 s | when focus cleared |
| soft-lock yaw gain | ×2.0·dt | stage 4 |
| dt clamp | 0.1 s | `update()` entry |
| near / far plane | `config.near_plane` / `config.far_plane` (5.0 / 15000.0) | config; swapped in `glm::perspective` |
| default screen | 1280×720 | pre-`set_screen_size` |
| initial pitch | 25° | member initializer |
| projection default FOV | 55° | `CameraModeConfig::fov` |

### Integration points

- **`Application`** (`src/bracken/application.cpp`): `init_renderer()` constructs the `CameraSystem`; `shutdown` resets it. The main loop does **not** call `camera_->update()` — the host does, from `on_simulate` (tier `sim-stage` under `run()`; the DevUI Camera tab's main-thread reads tear at worst — display-only). The host snapshots `camera().get_camera_state()` into the frame via `FrameBuilder::set_camera(...)` each frame (`examples/playground/playground.cpp`).
- **`InputHandler`** (`src/bracken/input_handler.hpp`): raw-input-only by design; exposes `get_camera_yaw()`, `get_camera_pitch()`, `get_camera_zoom_delta()` (zoom delta reset each frame by `Application::run` via `reset_camera_deltas()`), and consumes `set_camera_forward(x, z)` for camera-relative movement axes. The playground bridges the aiming half (`set_yaw(in.camera_yaw)`, `set_pitch(...)`, `adjust_zoom(in.camera_zoom_delta)`); `set_camera_forward` is fed by `Application`'s sim-stage tail from the camera it drives.
- **Dev overlay** (`src/bracken/dev/dev_panels.cpp`): `Application::init_dev_ui()` passes `&camera()` in `DevUI::Bindings`; the Camera tab renders readouts (yaw/pitch derived from forward), shake-test buttons (all four types), an FOV-bias slider (±30°), and a focus-strength slider (0–1; effective only if the host set a focus target). The sliders re-read the controller's live values each frame via `get_fov_bias()`/`get_focus_strength()`, so host-side changes aren't overwritten by stale panel state.
- **GPU**: none directly. The camera is pure CPU math; `CameraState` matrices flow into uniform buffers inside `SceneRenderer` (see `gpu/gpu_uniforms.hpp` contracts documented elsewhere).

### Threading and lifetime contracts

- **Single-threaded, main loop only.** No synchronization anywhere.
- `set_focus_target(const glm::vec3*)` stores a raw non-owning pointer dereferenced during every `update()` while `focus_strength_ > 0`. The pointee must outlive the focus period; clear with `nullptr`.
- `TerrainHeightFunc` and `CollisionCheckFunc` are `std::function`s copied into the camera and invoked (up to 8 terrain calls and 1 collision call) inside every `update()`. Captured state must remain valid for the camera's lifetime or until the callback is replaced.
- `get_config()` returns a reference into the object; invalidated only by destruction.

### Error handling

There is none by design: every setter accepts its input (clamping/normalizing where documented), `update()` cannot fail, and no `Result`/`bool` returns exist in this module. Degenerate inputs are sanitized silently: dt clamped to 0.1 s, screen size clamped to ≥1×1, zero-length shake directions replaced with +X, `smoothTime` floored at 0.0001, shake overflow evicts the oldest entry silently.

### Performance characteristics

`update()` is O(active_shakes) + a constant amount of scalar/vec3 spring math — trivially cheap (a Tracy zone exists for it). The only allocation is `std::deque` node churn on shake add/expire (bounded at 8 entries). Callback cost is host-defined: terrain callback is invoked up to 8× per update when set.

### Known limitations and ambiguities

- The auto-center and collision `mix(a, b, rate*dt)` blends are framerate-dependent (unlike the spring-damped values).
- `update_soft_lock` measures direction from the raw `target_position_` (not `smoothed_target_`), a minor inconsistency with the rest of the framing math.
- `set_collision_func` and the concrete-only getters remain unreachable through the `Application` facade (it exposes the abstract `CameraController&` only).

---

## Procedural Generation: Trees & Rocks

### Scope and source layout

| File | Contents |
|---|---|
| `src/bracken/procedural/tree_generator.hpp` | `TreeParams`, `TreeGenerator` (7 presets + `generate`) |
| `src/bracken/procedural/tree_generator.cpp` | `TreeRNG`, `TreeBuilder` (branch/leaf/proxy emission), texture loader, presets |
| `src/bracken/procedural/rock_generator.hpp` | `RockParams`, `RockGenerator` (5 presets + `generate`) |
| `src/bracken/procedural/rock_generator.cpp` | marching-cubes tables, SDF/noise functions, mesh extraction, presets |
| `src/bracken/procedural/tree_generator.test.cpp` | 15 GoogleTest cases (determinism, mesh presence, AABB, normals, presets) |
| `src/bracken/procedural/procedural.bench.cpp` | `BM_TreeOak`, `BM_TreePine`, `BM_RockBoulder/{40,80}`, `BM_ComputeTangents` |

Namespace: `bracken::procedural`. Both generators are stateless classes exposing only `[[nodiscard]] static` factory functions returning `std::unique_ptr<bracken::Model>`. There is no `rock_generator.test.cpp`; rocks are covered only by the benchmark and the playground integration run.

Attribution (from source comments): tree algorithm "based on the ez-tree algorithm by dgreenheck"; rock algorithm "based on the SDF + marching cubes approach from przemyslawzaworski/Unity-Procedural-Rock-Generation (MIT)".

### Common output contract

Both generators emit `bracken::Model` / `bracken::Mesh` (`src/bracken/model_loader.hpp`) with vertices of type `gpu::Vertex3D` (`src/bracken/gpu/gpu_types.hpp`):

```cpp
struct Vertex3D {
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    glm::vec2 texcoord{0.0f};
    glm::vec4 color{1.0f};
    glm::vec4 tangent{1.0f, 0.0f, 0.0f, 1.0f};
};
```

- Generators write `position`, `normal`, `texcoord`, `color`; **`tangent` is left at its default** `(1,0,0,1)` — `compute_tangents_from_uvs` (`src/bracken/model_utils.hpp`) is *not* invoked. This is inert because generated meshes carry no normal map (`hasNormalMap = 0` in the shaders).
- Indices are `uint32_t` triangle lists.
- Colors are linear-space (engine-wide convention).
- On the returned `Model`: `bounds` populated, `compute_bounding_sphere()` called, `loaded = true`, no skeleton/animations.

### TreeGenerator

#### Public API (`src/bracken/procedural/tree_generator.hpp`)

```cpp
class TreeGenerator {
public:
    [[nodiscard]] static std::unique_ptr<Model> generate(const TreeParams& params);
    // Presets. texture_base_path = directory containing bark/ and leaves/ subdirs.
    [[nodiscard]] static std::unique_ptr<Model> generate_oak(uint32_t seed = 0, const std::string& texture_base_path = "");
    [[nodiscard]] static std::unique_ptr<Model> generate_pine(uint32_t seed = 0, const std::string& texture_base_path = "");
    [[nodiscard]] static std::unique_ptr<Model> generate_dead(uint32_t seed = 0, const std::string& texture_base_path = "");
    [[nodiscard]] static std::unique_ptr<Model> generate_willow(uint32_t seed = 0, const std::string& texture_base_path = "");
    [[nodiscard]] static std::unique_ptr<Model> generate_birch(uint32_t seed = 0, const std::string& texture_base_path = "");
    [[nodiscard]] static std::unique_ptr<Model> generate_maple(uint32_t seed = 0, const std::string& texture_base_path = "");
    [[nodiscard]] static std::unique_ptr<Model> generate_aspen(uint32_t seed = 0, const std::string& texture_base_path = "");
};
```

`generate` always returns a non-null model (the trunk always produces geometry).

#### TreeParams (defaults from the header)

`static constexpr int MAX_LEVELS = 4`. Per-level arrays are plain C arrays of 4; index = branch level (0 = trunk).

| Field | Type | Default | Units / notes |
|---|---|---|---|
| `seed` | `uint32_t` | `0` | `TreeRNG` maps 0 → 1 |
| `levels` | `int` | `3` | recursion depth; branch level reaches `levels` inclusive, so valid range is 0..`MAX_LEVELS - 1` (3). Out-of-range values (and non-positive `sections`/`segments`, negative `children`/leaf counts) are **clamped by `sanitize_params` with a warning** — no out-of-bounds reads |
| `length[4]` | `float` | `{20, 20, 10, 1}` | world units |
| `radius[4]` | `float` | `{1.5, 0.7, 0.7, 0.7}` | level 0 absolute; levels ≥ 1 multiply the parent's interpolated section radius |
| `taper[4]` | `float` | `{0.7, 0.7, 0.7, 0.7}` | fraction of radius lost at tip (linear per section); ignored when `evergreen` |
| `children[4]` | `int` | `{7, 7, 5, 0}` | children spawned per branch at that level |
| `angle[4]` | `float` | `{0, 70, 60, 60}` | **degrees**, child tilt about local X |
| `gnarliness[4]` | `float` | `{0.15, 0.2, 0.3, 0.02}` | per-section perturbation amplitude (radians-ish; small-angle Euler quats) |
| `twist[4]` | `float` | `{0, 0, 0, 0}` | **radians** per section about local Y |
| `sections[4]` | `int` | `{12, 10, 8, 6}` | rings per branch |
| `segments[4]` | `int` | `{8, 6, 4, 3}` | radial vertices per ring (a seam duplicate is added) |
| `start[4]` | `float` | `{0, 0.4, 0.3, 0.3}` | child attach position sampled `U[start[level], 1]` (indexed by the *child's* level) |
| `force_direction` | `glm::vec3` | `{0, 1, 0}` | normalized before use |
| `force_strength` | `float` | `0.01` | per-section slerp amount = `force_strength / max(section_radius, 0.01)`, clamped to 1 |
| `leaf_count` | `int` | `1` | leaf cards per terminal branch (level == `levels`) |
| `leaf_size` | `float` | `2.5` | square card side |
| `leaf_size_variance` | `float` | `0.7` | `size = leaf_size * (1 + U(-v, +v))` |
| `leaf_angle` | `float` | `10` | degrees from branch |
| `leaf_start` | `float` | `0` | attach position `U[leaf_start, 1]` |
| `double_billboard` | `bool` | `true` | second quad rotated `half_pi` about card-local Y |
| `bark_color` | `glm::vec4` | `{0.45, 0.32, 0.22, 1}` | vertex color when untextured |
| `leaf_color` | `glm::vec4` | `{0.2, 0.5, 0.15, 1}` | vertex color when untextured |
| `bark_texture_path` / `leaf_texture_path` | `std::string` | `""` | empty = vertex colors |
| `evergreen` | `bool` | `false` | full linear taper; `child_length *= (1 - child_start)` |

#### RNG and determinism

`TreeRNG` (`tree_generator.cpp`) is a Park–Miller/MINSTD LCG on `uint32_t`:

```cpp
state_ = (state_ * 16807u) % 2147483647u;
float t = state_ / 2147483647.0f;      // random(max=1, min=0) → min + t*(max-min)
```

Constructor: `state_(seed == 0 ? 1 : seed)`. All branch/leaf randomness draws from this single sequential stream, so geometry is a pure function of `TreeParams` (asserted bit-exact by `SeededRNGIsDeterministic`; different seeds diverge per `DifferentSeedsProduceDifferentTrees`). The canopy proxy uses a *separate* murmur-style avalanche hash seeded `params.seed * 0x9E3779B9u + 0x85EBCA6Bu` (xorshift/multiply by `0x85EBCA6B`, `0xC2B2AE35`; 24-bit mantissa output) for per-clump rotation/size, so it does not perturb the main stream.

#### Branch geometry algorithm (`TreeBuilder::build` / `generate_branch`)

Breadth-first: a `std::queue<BranchData>` seeded with the trunk (origin `(0,0,0)`, identity orientation, `length[0]`, `radius[0]`, `sections[0]`, `segments[0]`). Each dequeued branch:

1. Walks `section_count + 1` rings. Ring radius: base radius × linear taper (`1 - taper[level] * i/sections`; evergreen: `1 - i/sections`); the final ring of a terminal-level branch is pinched to `0.001`.
2. Each ring emits `segments + 1` vertices (`cos/sin` circle in the section's local XZ, transformed by the section quaternion; the +1 duplicates segment 0 with `u = 1` for the UV seam). UVs: `u = j/segments`, `v = (i % 2 == 0) ? 0 : 1` — v ping-pongs 0/1 per ring rather than accumulating arclength. Normals: rotated radial directions (unit length; verified by `NormalsAreUnitLength`). Color: `bark_color`.
3. Advances the section origin by `orientation * (0,1,0) * section_length` (`section_length = length / sections`; note lines 110–115 of `tree_generator.cpp` contain a self-cancelling divide/multiply then reassignment — deliberate dead code, net no-op), then composes onto the orientation: gnarliness (small-angle Euler quat, X/Z each `U(-g, g)` where `g = max(1, 1/sqrt(max(radius, 0.001))) * gnarliness[level]`), twist (`angleAxis(twist[level], Y)`), and growth force (slerp toward `rotation(current_up, force_dir) * orientation` by `min(force_strength / max(radius, 0.01), 1)`).
4. Indexing: per quad `(v1, v3, v2), (v2, v3, v4)` with row stride `N = segments + 1`. Per branch: `(sections+1) * (segments+1)` vertices, `sections * segments * 6` indices.
5. If `level < levels`: enqueues `children[level]` children. Attach point `child_start = U[start[level+1], 1]` is mapped to a parent section interval; origin/radius lerped, orientation slerped between bracketing `SectionInfo`s. Child orientation = `parent * angleAxis(radial, Y) * angleAxis(radians(angle[level+1]), X)` where radial angles are evenly spaced with one random per-branch phase offset. Child radius = `radius[level+1] * lerp(ring radii)`; child length = `length[level+1]` (× `(1 - child_start)` if evergreen).
6. If `level == levels`: emits leaves (below).

#### Leaf geometry (`generate_leaves` / `generate_leaf_quad`)

Per terminal branch: `leaf_count` cards placed like child branches (`U[leaf_start, 1]`, evenly-spaced radial + random phase, tilt `leaf_angle` degrees). Each card: 4 vertices spanning local `x ∈ [-W/2, W/2]`, `y ∈ [0, L]` (`W = L = size`), plane normal `+Z` (all 4 verts share it), UVs `{0,1},{0,0},{1,0},{1,1}`, color `leaf_color`; 2 triangles `(0,1,2),(0,2,3)`. `double_billboard` emits a second card rotated `glm::half_pi` about the card's Y (exactly 2× vertices — asserted by `DoubleBillboardDoubleLeafQuads`). The per-branch cluster AABB of emitted leaf verts is recorded in `TreeBuilder::leaf_clumps` (`center`, `half_extent`).

#### Canopy shadow proxy mesh

Built in `TreeGenerator::generate` when `leaf_clumps` is non-empty. Purpose (per source comments): the leaf mesh is ~60–70% of tree triangles but contributes mostly dappled shadow coverage, so leaves get `cast_shadows = false` and the proxy — `shadow_only = true, cast_shadows = true` — carries the shadow silhouette at far lower cost (~10× less alpha-test overdraw).

Per clump: three mutually perpendicular unit quads (XY/XZ/YZ planes, spanning ±1 pre-scale), scaled by `half = max(clump.half_extent * 1.05, vec3(0.8)) * size_var` with `size_var ∈ [0.95, 1.30)`, rotated by a hash-driven quat (`yaw ∈ [0, 2π)`, `pitch/roll ∈ [-0.75, 0.75]`), positioned at the clump centroid. Each quad is emitted with **both winding orders** (shadow pipelines cull front faces — `SDL_GPU_CULLMODE_FRONT` with depth bias `1.25/1.75` in `src/bracken/gpu/pipeline_catalog.cpp`), so per clump: 12 vertices, 36 indices (12 triangles); the `reserve` calls match. The proxy shares `leaf_texture_path` (loaded again into its own `texture_pixels`) so `shadow_depth.frag.hlsl` can alpha-test (`if (hasTexture == 1) ... if (alpha < 0.5) discard`); untextured proxies cast solid quads. Vertex color is white.

#### Model assembly

Mesh order: `[0]` branches (if non-empty — always true), `[1]` leaves (if any leaf verts), `[2]` proxy (if any clumps). If a texture path is set for branches/leaves, `load_texture_from_file` runs first and vertex colors are whitened **only after a successful load** — failure warns (`BRACKEN_LOG_WARN`, "falling back to vertex colors") and keeps the authored vertex colors. Textured leaves additionally get `alpha_mode = Mask` + `alpha_cutoff = 0.5`; leaf meshes are always `double_sided`. Texture loading: `IMG_Load` → `SDL_ConvertSurface(SDL_PIXELFORMAT_RGBA32)` → copy into `Mesh::texture_pixels` + `texture_width/height`, `has_texture = true`.

AABB is computed over all meshes' vertices, then `min_y` is clamped to `0.0f` ("the centering transform doesn't lift the tree off the ground when branches droop below origin"). Bounding sphere from AABB. `loaded = true`.

#### Preset parameter matrix (exact values, `tree_generator.cpp`)

| Param | oak | pine | dead | willow | birch | maple | aspen |
|---|---|---|---|---|---|---|---|
| levels / evergreen | 3 / no | 3 / yes | 2 / no | 3 / no | 3 / no | 3 / no | 3 / no |
| length | 10,8,5,1 | 15,6,3,1 | 8,6,3,1 | 8,10,8,2 | 12,6,4,1 | 9,7,5,1.5 | 14,5,3,1 |
| radius | 1,.5,.4,.3 | .8,.3,.2,.15 | .7,.35,.2,.1 | 1.2,.4,.2,.1 | .5,.25,.15,.1 | .9,.45,.3,.2 | .4,.2,.12,.08 |
| taper | .6,.7,.8,.9 | .8,.7,.8,.9 | .5,.6,.8,.9 | .5,.6,.8,.9 | .7,.75,.8,.9 | .6,.65,.75,.85 | .75,.7,.8,.9 |
| children | 5,4,3,0 | 8,4,2,0 | 4,3,0,0 | 6,5,3,0 | 5,4,2,0 | 6,5,4,0 | 7,3,2,0 |
| angle | 0,50,55,60 | 0,75,60,50 | 0,45,50,0 | 0,60,80,85 | 0,40,50,55 | 0,55,50,55 | 0,35,40,45 |
| gnarliness | .1,.15,.2,.02 | .05,.08,.1,.02 | .3,.4,.3,0 | .08,.12,.15,.02 | .05,.1,.15,.02 | .12,.18,.22,.02 | .03,.06,.1,.02 |
| sections | 6,5,3,2 | 7,4,3,2 | 6,4,3,2 | 6,5,4,3 | 7,4,3,2 | 6,5,4,2 | 7,4,3,2 |
| segments | 6,4,3,3 | 5,3,3,3 | 5,3,3,3 | 5,4,3,3 | 5,3,3,3 | 5,4,3,3 | 4,3,3,3 |
| start | 0,.5,.3,.2 | 0,.2,.2,.2 | 0,.4,.3,.3 | 0,.4,.2,.1 | 0,.5,.3,.2 | 0,.35,.25,.2 | 0,.3,.2,.2 |
| force dir / strength | +Y / .02 | +Y / .03 | +Y / .005 | (0,−.3,0) / .015 | +Y / .025 | +Y / .02 | +Y / .035 |
| leaf count/size/var/angle | 3/3/.5/30 | 4/2/.3/15 | 0 (none) | 4/2.5/.6/45 | 3/2/.4/25 | 5/2.8/.5/35 | 3/1.5/.3/20 (`leaf_start=.2`) |
| bark color | .4,.28,.18 | .35,.22,.12 | .35,.3,.25 | .35,.3,.2 | .85,.82,.78 | .38,.3,.22 | .8,.78,.7 |
| leaf color | .15,.45,.12 | .08,.3,.06 | — | .2,.5,.15 | .2,.55,.1 | .6,.2,.05 | .55,.65,.1 |
| bark texture | oak_color_1k.jpg | pine_color_1k.jpg | birch_color_1k.jpg | willow_color_1k.jpg | birch_color_1k.jpg | oak_color_1k.jpg | birch_color_1k.jpg |
| leaf texture | oak_color.png | pine_color.png | — | ash_color.png | aspen_color.png | oak_color.png | aspen_color.png |

Texture paths are joined as `texture_base_path + "/bark/<file>"` and `"/leaves/<file>"`, only when `texture_base_path` is non-empty. No such assets exist in this repository. Comments in `generate_oak` note the low `sections/segments` values are a deliberate ~40% branch-triangle reduction vs the visual reference.

### RockGenerator

#### Public API (`src/bracken/procedural/rock_generator.hpp`)

```cpp
class RockGenerator {
public:
    [[nodiscard]] static std::unique_ptr<Model> generate(const RockParams& params);
    [[nodiscard]] static std::unique_ptr<Model> generate_boulder(uint32_t seed = 0);
    [[nodiscard]] static std::unique_ptr<Model> generate_slate(uint32_t seed = 0);
    [[nodiscard]] static std::unique_ptr<Model> generate_spire(uint32_t seed = 0);
    [[nodiscard]] static std::unique_ptr<Model> generate_cluster(uint32_t seed = 0);
    [[nodiscard]] static std::unique_ptr<Model> generate_mossy(uint32_t seed = 0);
};
```

`generate` returns `nullptr` if marching cubes yields zero vertices. Rocks are always vertex-colored — every rock mesh has `has_texture = false`; there is no texture parameter.

`RockParams` defaults: `seed = 880`, `resolution = 40`, `scale = 1.0`, `steps = 20`, `smoothness = 0.05`, `displacement_scale = 0.15`, `displacement_spread = 10.0`, `color = {0.5, 0.48, 0.45, 1}`. (The defaults equal the boulder preset, as noted in `procedural.bench.cpp`.) The seed is reduced by an integer avalanche hash masked to `[0, 2^20)` before entering the sin()-based SDF hash, so the full `uint32` range yields distinct rocks with float-exact behavior.

#### Scalar field (`scalar_field`, `rock_sdf`, `fbm`)

Sampled in "unit space" `p ∈ [-0.7, 0.7]³` (`grid_extent = 0.7`, chosen "larger than SDF radius to avoid boundary clipping"):

1. **Base clip:** `p.y < -0.05` → field value 0 (outside) — produces the flat underside.
2. **SDF** `rock_sdf(p * 2, steps, float(seed), smoothness)`: starts from a sphere `length(p) - 0.95`, then `steps` iterations each derive a pseudo-random radius `r = 2.5 + fract(sin(j*727.1)*435.545)` and unit direction from `sin`-hash of `j = i + seed`, and smooth-blend (`polynomial smooth-min` with `k = smoothness`, using `mix(a, -b, h) + k*h*(1-h)`) a sphere of radius `r*0.8` at offset `v*r` into the running distance. **Note:** `seed` participates as a `float`, so seeds ≥ 2²⁴ lose precision and large distinct seeds can collide.
3. **Displacement** (skipped when `displacement_scale <= 0`): SDF normal from central differences (`eps = 1e-4`, six extra `rock_sdf` calls), squared/normalized into triplanar weights; three 6-octave value-noise FBM evaluations (`fbm` over rotated/scaled `glm::vec2` domains at frequency `displacement_spread`, hash via `fract(sin(dot(...))*43758.5453)`) blended by those weights; `d += displacement_scale * length(disp)`.
4. **Density mapping:** `clamp((0.05 - d) / 0.05, 0, 1)` — a narrow density ramp around the surface.

#### Marching cubes extraction

Grid: `(res+1)³` floats precomputed into a `std::vector<float>` (`res = resolution`; 41³ ≈ 276 KB at 40, 81³ ≈ 2.1 MB at 80), cell size `1.4/res`. Classic tables: `edge_table[256]` and `tri_table[256][16]` are embedded in `rock_generator.cpp`. Iso-threshold `0.5`; edge vertices via linear interpolation `t = (0.5 - v1)/(v2 - v1)` (guard: `|v1 - v2| < 1e-6` returns `p1`). Corner positions are multiplied by `params.scale` at extraction.

Per output triangle, three **unshared** vertices are appended (no dedup/welding; index buffer is sequential `base, base+1, base+2`), each with:

- **Normal:** central-difference gradient of `scalar_field` at the vertex (unit-space position `tri[j] / scale`, `eps = 0.01`), normalized; falls back to the triangle face normal when gradient length ≤ 1e-6 (and face normal itself falls back to `(0,1,0)` on NaN).
- **UV:** triplanar by the dominant axis of the *face* normal, from `np = tri[j]/scale + 0.5`: X-dominant → `(np.z, np.y)`, Y-dominant → `(np.x, np.z)`, else `(np.x, np.y)`.
- **Color:** `params.color`.

#### Output post-processing

- All vertices shifted up by `-min_y` so the flat base sits at `Y = 0`.
- Single `Mesh` (`has_texture = false`; appearance comes from per-vertex `params.color`).
- AABB computed, then **forced symmetric in X/Z** (`bounds.min.x = -max(|min.x|, |max.x|)` etc.) and `bounds.min.y = 0`, "so `build_model_transform` centering is zero (procedural rocks are already centered at XZ origin)". Bounding sphere computed; `loaded = true`.

#### Presets (exact values, `rock_generator.cpp`)

All: `resolution = 40`, `scale = 1.0`. Passing `seed == 0` selects the preset's default seed.

| Preset | seed default | steps | smoothness | displacement scale / spread | color |
|---|---|---|---|---|---|
| boulder | 880 | 20 | 0.05 | 0.15 / 10 | (0.5, 0.48, 0.45) |
| slate | 123 | 15 | 0.03 | 0.08 / 15 | (0.42, 0.4, 0.38) |
| spire | 456 | 25 | 0.08 | 0.1 / 8 | (0.55, 0.52, 0.48) |
| cluster | 789 | 30 | 0.12 | 0.2 / 12 | (0.48, 0.45, 0.42) |
| mossy | 321 | 18 | 0.06 | 0.18 / 7 | (0.35, 0.42, 0.32) |

### Renderer integration contract

- **Registration:** `ModelManager::register_model(name, std::unique_ptr<Model>)` (`src/bracken/model_loader.hpp/.cpp`) takes ownership, calls `ModelLoader::upload_to_gpu` when a device is set (static vertex/index `GPUBuffer`s; `texture_pixels` → `RGBA8_SRGB` texture with mipmaps), then **frees CPU-side vertices/indices/texture pixels** (`cached_index_count` preserves the draw count). `model_paths_` gets an empty entry, so `reload_model` returns `false` and hot-reload never touches procedural models.
- **Submission:** `FrameBuilder::draw_model(handle, transform, tint, flags, motion_id)` (`src/bracken/scene/frame_builder.hpp`). `SceneRenderer::build_instance_batches` bins static commands per `ModelHandle`, frustum-culls with `Model::bounding_center/half_diag`, and produces per-cascade shadow instance batches; rendering is one `SDL_DrawGPUIndexedPrimitives(..., instance_count, ...)` per mesh per model type via the `model_instanced` pipelines.
- **Mesh flag consumption** (`src/bracken/scene/scene_renderer.cpp`): `shadow_only` meshes are skipped in every main-pass path (non-instanced ~line 1169, instanced ~1626, skinned ~1810); `cast_shadows == false` meshes are skipped in every shadow path (~1735, ~2215, ~2283). This is exactly the tree's leaf/proxy split.
- **Material handling:** generated meshes use a default-constructed `PBRMaterial` (`metallic_factor = 0`, `roughness_factor = 0.85`, `alpha_mode = Opaque`, `double_sided = false`). `apply_mesh_material` therefore sends `alphaCutoff = 0` (no discard in `model.frag.hlsl`/`model_instanced.frag.hlsl`) and the back-face-culling pipeline variant. The shadow passes independently alpha-test any mesh with a bound texture at a hardcoded 0.5 (`gpu::ShadowFragUniforms::hasTexture`, `shadow_depth.frag.hlsl`).
- **Untextured color path:** shaders use `input.vertexColor * baseColorFactor * tintColor` when `hasTexture == 0`.

### Threading and lifetime

- Generation is pure CPU with no shared mutable state; the engine's single-threaded contract (`docs/architecture.md`, "Threading & lifetime contracts") applies to the *upload/registration* side (`register_model` reaches SDL GPU APIs), not to `generate` itself. `TreeGenerator` calls SDL_image (`IMG_Load`) and `SDL_Log` when texture paths are set.
- Both generators wrap their entry points in `BRACKEN_PROFILE_ZONE` (Tracy CPU zones, `src/bracken/core/profiler.hpp`).
- Returned `Model` ownership transfers to the caller; `ModelManager` is non-copyable/non-movable and keeps `Model*`/handles stable for the app's lifetime.

### Error handling

Per the pattern catalogued in `docs/audits/modernization.md` (static factories returning `unique_ptr`, reason only in the log): no `core::Result` here.

- `TreeGenerator::generate` cannot fail; texture-load failures log `"TreeGenerator: Failed to load texture %s"` / conversion failures via `SDL_Log` and degrade to untextured (with vertex colors already forced white — a silent visual defect).
- `RockGenerator::generate` returns `nullptr` when extraction produces no vertices; presets use known-good parameters. Callers registering the result must not pass null to `register_model` (it dereferences the model unconditionally).

### Performance characteristics

Measured with `build/src/bracken/bracken_benchmarks` (`--benchmark_filter='BM_Tree|BM_Rock'`) on a **Debug** build (order-of-magnitude only):

| Benchmark | Time |
|---|---|
| `BM_TreeOak` | ~1.8 ms |
| `BM_TreePine` | ~1.9 ms |
| `BM_RockBoulder/40` | ~4.6 s |
| `BM_RockBoulder/80` | ~28 s |

Cost model: trees are O(total sections × segments) with trivial per-vertex math. Rocks are dominated by scalar-field evaluations — `(res+1)³` grid samples, each running the `steps`-iteration SDF plus (when displacement is on) 6 more SDF calls and 3 six-octave FBMs; then 6 more `scalar_field` calls *per output vertex* for gradient normals. No parallelism is used (consistent with the engine's "jobs stay idle" policy). Rock vertex count equals 3× triangle count due to unshared vertices, which also inflates GPU vertex-fetch cost versus a welded mesh.

### Known limitations, quirks, and silent behaviors

1. **Out-of-range `TreeParams` are clamped with a warning** (`sanitize_params`): `levels` to `[0, 3]`, `sections` raised to ≥ 1, `segments` to ≥ 3, negative `children`/`leaf_count` zeroed.
2. **Untextured leaf cards are `Opaque`**; textured leaves are alpha-masked (`Mask` + cutoff 0.5) automatically, and leaf meshes are always `double_sided`.
3. **Branch UV `v` ping-pongs 0/1 per ring** instead of accumulating arclength; bark textures repeat once per two rings and stretch with section length.
4. **Determinism is per-build**: bit-exactness is only guaranteed for the same binary/compiler/platform (float-heavy code, `sin`-based hashing). The integer seed hash changed rock shapes for a given seed relative to older builds.
5. Rocks are covered by `rock_generator.test.cpp`; trees by `tree_generator.test.cpp`.

---

## Render Graph (Available, Not Yet Wired)

### Purpose and status

`src/bracken/render_graph/` (namespace `bracken::render_graph`) implements a
frame-graph style pass scheduler: declarative pass registration with
read/write resource sets, Kahn topological scheduling with cycle detection,
interval-based transient texture allocation with within-frame memory aliasing
backed by a cross-frame `SDL_GPUTexture` pool, optional graph-managed
`SDL_BeginGPURenderPass`/`SDL_EndGPURenderPass` bracketing, and Graphviz DOT
export.

**Status:** in-tree and unit-tested (`render_graph.test.cpp`, 8 tests) but
**not wired into the frame path**. `scene::SceneRenderer` issues passes
directly; nothing under `examples/` references the module. Documented in
three places: `docs/architecture.md` (module table row: "in-tree, tested, not
wired into `SceneRenderer`"; stubs list: "The graph is kept (with its tests)
for when pass count or transient aliasing justifies it"),
`docs/engine_design_notes.md` (the "infrastructure without load" trap:
per-frame infrastructure cost must be absorbed by a measured workload; wiring
triggers are pass count past hand-tractable size, TAA history buffers —
"the first concrete reason to wire up `render_graph/`" — transient aliasing,
multi-camera), and `README.md`.

Doc discrepancy: `docs/engine_design_notes.md` line 75 credits the module
with "debug-group integration", but `render_graph.cpp` contains no
`SDL_PushGPUDebugGroup`/`gpu::GPUDebugGroup` usage. The claim is aspirational
or stale.

### File inventory

| File | Contents |
|---|---|
| `src/bracken/render_graph/render_graph.hpp` | `RenderGraph` class (umbrella header; includes all others) |
| `src/bracken/render_graph/render_graph.cpp` | Implementations of `RenderGraph`, `PassBuilder`, `RenderPassContext` |
| `src/bracken/render_graph/render_graph_resource.hpp` | `TextureDesc`, `ResourceHandle`, `ResourceNode` |
| `src/bracken/render_graph/render_graph_pass.hpp` | `ColorAttachment`, `DepthAttachment`, `PassExecuteFn`, `PassNode` |
| `src/bracken/render_graph/pass_builder.hpp` | `PassBuilder` |
| `src/bracken/render_graph/render_graph_context.hpp` | `RenderPassContext` |
| `src/bracken/render_graph/transient_allocator.hpp/.cpp` | `TransientAllocator` texture pool |
| `src/bracken/render_graph/render_graph.test.cpp` | GoogleTest coverage (auto-globbed into `bracken_tests`) |

### Resource model (`render_graph_resource.hpp`)

The graph manages **textures only** (a former buffer-resource scaffold —
`ResourceType`, `BufferDesc`, `get_buffer` — had no declare/import API and was
removed; re-add alongside a real buffer API).

#### `TextureDesc`
Plain struct; the pool key. Fields with defaults: `uint32_t width = 0`,
`uint32_t height = 0`,
`SDL_GPUTextureFormat format = SDL_GPU_TEXTUREFORMAT_INVALID`,
`SDL_GPUTextureUsageFlags usage = 0`, `uint32_t mip_levels = 1`,
`uint32_t array_layers = 1`, `uint32_t sample_count = 1`. `operator==`
compares all seven fields exactly; pool reuse requires exact equality.

#### `ResourceHandle`
Type-safe index wrapper: `uint32_t idx_` with sentinel
`INVALID = std::numeric_limits<uint32_t>::max()`. API: default ctor
(invalid), `explicit constexpr ResourceHandle(uint32_t idx)`,
`valid()`, `index()`, `static invalid()`, `==`/`!=`. Handles index directly
into `RenderGraph::resources_` and are invalidated by `reset()` /
`begin_frame()`.

#### `ResourceNode`
Per-resource record stored by value in `std::vector<ResourceNode>`:

- Identity: `std::string name`.
- Transient state: `bool transient = false`, `TextureDesc texture_desc`.
- Imported state (graph does not own): `SDL_GPUTexture* imported_texture`,
  `imported_width`, `imported_height`,
  `SDL_GPUTextureFormat imported_format = SDL_GPU_TEXTUREFORMAT_INVALID`.
- Compile-time resolution: `SDL_GPUTexture* resolved_texture`
  (transients only).
- Lifetime (compiled-order step indices, `UINT32_MAX` = unset):
  `uint32_t first_write_pass`, `uint32_t last_read_pass`.

### Pass model (`render_graph_pass.hpp`)

A pass is a callback plus declared reads/writes. Whether the graph wraps the
callback in an SDL render pass is driven entirely by `managed_render_pass` +
the declared attachments; there is **no pass-type enum** (a former `PassType`
field was inert metadata and was removed — compute/blit passes just use the
raw command buffer from `RenderPassContext`).

- `struct ColorAttachment { ResourceHandle handle; SDL_GPUColorTargetInfo info{}; }`
- `struct DepthAttachment { ResourceHandle handle; SDL_GPUDepthStencilTargetInfo info{}; }`
- `using PassExecuteFn = std::function<void(RenderPassContext&)>;`
- `struct PassNode`: `std::string name`;
  `std::vector<ResourceHandle> reads, writes`;
  `std::vector<ColorAttachment> color_attachments`;
  `DepthAttachment depth_attachment{}`; `bool has_depth = false`;
  `bool managed_render_pass = false`; `PassExecuteFn execute`.
  There is at most one depth attachment per pass (later
  `attach_depth`/`create_depth_target` calls overwrite it); color attachments
  accumulate. Pass names are not deduplicated (unlike resource names).

### `PassBuilder` (`pass_builder.hpp`, impl in `render_graph.cpp`)

Non-owning view (`RenderGraph* graph_; PassNode* node_`) valid only inside a
setup lambda. Signatures:

```cpp
ResourceHandle create_color_target(const std::string& name, const TextureDesc& desc,
                                   SDL_GPULoadOp loadop = SDL_GPU_LOADOP_CLEAR,
                                   glm::vec4 clear_color = glm::vec4(0.0f));
ResourceHandle create_depth_target(const std::string& name, const TextureDesc& desc,
                                   SDL_GPULoadOp loadop = SDL_GPU_LOADOP_CLEAR,
                                   float clear_depth = 0.0f);
ResourceHandle import_texture(const std::string& name, SDL_GPUTexture* tex,
                              uint32_t width, uint32_t height, SDL_GPUTextureFormat format);
ResourceHandle lookup(const std::string& name) const;
ResourceHandle read(const std::string& name);
ResourceHandle write(const std::string& name);
void read(ResourceHandle h);
void write(ResourceHandle h);
void attach_color(ResourceHandle h, SDL_GPULoadOp loadop = SDL_GPU_LOADOP_LOAD,
                  glm::vec4 clear_color = glm::vec4(0.0f));
void attach_depth(ResourceHandle h, SDL_GPULoadOp loadop = SDL_GPU_LOADOP_CLEAR,
                  float clear_depth = 0.0f);
void set_managed_render_pass(bool managed) noexcept;
```

Behavioral contract:

- `create_color_target` ORs `SDL_GPU_TEXTUREUSAGE_COLOR_TARGET` into
  `desc.usage` if absent, declares the transient, records it as a write,
  appends a `ColorAttachment` with `store_op = SDL_GPU_STOREOP_STORE`, and
  sets `managed_render_pass = true`. `create_depth_target` is the analog with
  `SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET`,
  `stencil_load_op = SDL_GPU_LOADOP_DONT_CARE`,
  `stencil_store_op = SDL_GPU_STOREOP_DONT_CARE`, `clear_stencil = 0`,
  `cycle = false`. Store ops and stencil ops are **not configurable**.
- Depth clear defaults to `0.0f` per the engine's reversed-Z scene
  convention (far = 0); standard-Z passes (shadows) must pass `1.0f`
  explicitly (documented in the header comments).
- `attach_color` default load op is `SDL_GPU_LOADOP_LOAD`;
  `create_color_target`'s is `SDL_GPU_LOADOP_CLEAR` — an asymmetry to note.
- `read(name)`/`write(name)` resolve via `RenderGraph::find_resource`; an
  unknown name returns `ResourceHandle::invalid()` and records **nothing**
  (silent no-op, no log).
- `read(h)`/`write(h)` ignore invalid handles and deduplicate within the
  pass via linear `std::find` over the pass's vectors.
- A pass that both reads and writes the same resource does not self-edge
  (explicit `p == i` skip in `topo_sort`), so read-modify-write is legal.

### `RenderGraph` (`render_graph.hpp/.cpp`)

Non-copyable. Members: `gpu::GPUDevice* device_`, `SDL_GPUCommandBuffer* cb_`,
`TransientAllocator allocator_`, `std::vector<PassNode> passes_`,
`std::vector<ResourceNode> resources_`,
`std::unordered_map<std::string, uint32_t> name_to_resource_`,
`std::vector<size_t> compiled_order_`, `bool compiled_ = false`, and stats
`last_compiled_passes_`, `last_transient_count_`, `last_reuse_count_`
(all `uint32_t`, zero-initialized).

#### Lifecycle API

- `void init(gpu::GPUDevice& device)` — stores the device and binds it to the
  `TransientAllocator`. Optional: without `init()`, the allocator runs in
  test mode (see below).
- `void shutdown()` — clears passes/resources/name map/compiled order, calls
  `allocator_.shutdown()` (evicts pooled textures), nulls `device_`. Also run
  by the destructor.
- `void begin_frame(SDL_GPUCommandBuffer* cb)` — stores `cb` (nullable for
  tests), calls `reset()` and `allocator_.begin_frame()` (zeroes per-frame
  acquire/reuse counters).
- `void reset()` — clears `passes_`, `resources_`, `name_to_resource_`,
  `compiled_order_`; sets `compiled_ = false`. **Does not** touch the texture
  pool.
- `void end_frame()` — `allocator_.release_all()` (marks every pooled entry
  free), nulls `resolved_texture`/`resolved_buffer` on all transient
  resources, nulls `cb_`.
- `void evict_pool()` — forwards to `allocator_.evict_all()`; intended for
  resize (header comment) since pooled textures of stale dimensions would
  otherwise persist unused.

#### Pass declaration

- Typed template:
  `template<typename Data, typename SetupFn, typename ExecuteFn> void add_pass(const std::string& name, SetupFn&& setup, ExecuteFn&& execute)`
  — `std::make_shared<Data>()` per pass; setup is invoked as
  `setup(builder, *data)`; the execute closure captures the `shared_ptr` and
  the forwarded functor, invoking `exec(ctx, *data)`.
- Untyped:
  `void add_pass(const std::string& name, std::function<void(PassBuilder&)> setup, PassExecuteFn execute)`
  — `setup` may be null (skipped). Both overloads are three-argument
  (name, setup, execute).
- `create_pass_node` does `passes_.emplace_back()`; a nested `add_pass` from
  within a setup lambda can reallocate `passes_` and dangle the outer call's
  `PassNode&` (the typed overload assigns `node.execute` *after* setup
  returns). Not guarded — caller contract.

#### Resource declaration

- `ResourceHandle declare_transient_texture(const std::string& name, const TextureDesc& desc)`
  — name-deduplicating: an existing name returns the existing handle and the
  new `desc` is **silently ignored**.
- `ResourceHandle declare_imported_texture(const std::string& name, SDL_GPUTexture* tex, uint32_t w, uint32_t h, SDL_GPUTextureFormat fmt)`
  — an existing name has its imported pointer/dims/format **overwritten** and
  `transient` forced to `false` (can flip a previously transient resource).
- `ResourceHandle find_resource(const std::string& name) const` — invalid
  handle if absent.
- `ResourceNode& resource(ResourceHandle h)` (+ const overload) — `assert`s
  validity and bounds; **no release-build check** (out-of-range index is UB).

#### `compile()`: scheduling + allocation

`[[nodiscard]] bool compile()` runs three phases:

1. **Topological sort** (`topo_sort`, `render_graph.cpp:271`):
   - Builds `producers[resource] = {passes that write it}`.
   - Edge rule: for every pass `i` reading resource `R`, add edge
     `p → i` for **every** producer `p` of `R` (`p != i` skipped;
     per-producer `std::unordered_set<size_t> seen_edges` deduplicates
     edges). There is no versioning — a reader depends on all writers of the
     name regardless of declaration order.
   - Kahn's algorithm with a FIFO `std::queue`, so mutually independent
     passes retain insertion order (deterministic).
   - If `out_order.size() != n`, a cycle exists: `compile()` logs
     `BRACKEN_LOG_ERROR("rg", "compile: cycle detected in render graph (passes={}); skipping execute", ...)`,
     clears `compiled_order_`, sets `compiled_ = false`, returns `false`.
     `execute()` then no-ops. There is no diagnostic of *which* passes form
     the cycle.
   - Complexity: O(P + R + E) with E deduplicated edges; edge construction is
     O(Σ reads × producers-per-resource).

2. **Lifetime computation** (`compute_lifetimes`): resets all lifetimes to
   `UINT32_MAX`, then walks the compiled order. For each step `s`: every
   written resource gets `first_write_pass = min-first-seen(s)` and
   `last_read_pass = max(last_read_pass, s)` (writes extend lifetime too, so
   a written-but-never-read resource lives exactly at its writer); every read
   resource gets `last_read_pass = s`. Note reads assign (not max) — with a
   topologically valid order the read step is monotonically the latest use.

3. **Transient allocation** (`allocate_transients`): interval-based aliasing.
   For each compiled step, first **acquire** every transient whose
   `first_write_pass == step` and `resolved_texture == nullptr`
   (`allocator_.acquire_texture(texture_desc, name.c_str())`), then
   **release** every transient whose `last_read_pass == step`
   (`allocator_.release_texture(...)`) — without nulling
   `resolved_texture`, so the releasing pass can still use the texture at its
   own step while making it available to acquires at step ≥ `step + 1`.
   Because acquires precede releases within a step, two transients with
   identical descs can only alias if their lifetimes are strictly disjoint in
   compiled-step terms. Complexity: **O(steps × resources)** (full resource
   scan per step) — fine at current scale, quadratic-ish for very large
   graphs. Stats: `last_transient_count_` = acquires performed;
   `last_reuse_count_` = allocator reuse hits during this compile
   (delta of `reuses_this_frame()`).

   **Failure behavior:** if texture creation fails inside the allocator,
   `compile()` logs `compile: transient allocation failed`, releases every
   already-acquired transient back to the pool (nulling the resolved
   pointers so no pass can see a stale one), clears `compiled_order_`, and
   returns `false` — `execute()` then no-ops (tests
   `CompileFailsWhenTransientAllocationFails`,
   `PartialTransientAllocationFailureReleasesAcquired`).

Finally `last_compiled_passes_ = compiled_order_.size()`,
`compiled_ = true`, return `true`.

#### `execute()`

No-op unless `compiled_`. For each step, for the pass at
`compiled_order_[step]`:

1. If `cb_ != nullptr && pass.managed_render_pass && (!color_attachments.empty() || has_depth)`:
   copy each attachment's stored `SDL_GPUColorTargetInfo` /
   `SDL_GPUDepthStencilTargetInfo`, patch in `info.texture` resolved as
   `transient ? resolved_texture : imported_texture`, and call
   `SDL_BeginGPURenderPass(cb_, colors.data(), colors.size(), depth_ptr)`.
   A temporary `std::vector<SDL_GPUColorTargetInfo>` is built per pass per
   frame (heap allocation on the hot path — acceptable at current scale).
2. Construct `RenderPassContext ctx(*this, cb_, sdl_pass)` and invoke
   `pass.execute(ctx)` if set.
3. `SDL_EndGPURenderPass` if one was begun.

No pipeline barriers, layout transitions, or fences are emitted — SDL3 GPU
handles hazards internally. No dead-pass elimination: every declared pass
executes, even if nothing consumes its outputs.

#### Diagnostics

- `size_t pass_count() const noexcept`, `size_t resource_count() const noexcept`,
  `const std::vector<size_t>& compiled_order() const noexcept`.
- `uint32_t last_compiled_passes() / last_transient_count() / last_reuse_count() const noexcept`.
- `PassNode& pass_at(size_t i)` (+ const) — commented "For tests";
  unchecked indexing.
- `void dump_dot(std::ostream& os) const` — emits
  `digraph RenderGraph { rankdir=LR; ... }` with passes as `shape=box` nodes,
  resources as `shape=ellipse` nodes labeled with `" (T)"` (transient) or
  `" (I)"` (imported), write edges `passN -> resM`, read edges
  `resM -> passN`. Names are DOT-escaped (`dot_escape` — quotes/backslashes
  can't corrupt the output; test `DumpDotEscapesLabels`). Works on the
  declared graph; after a successful `compile()` each pass label is
  additionally annotated `[step N]` with its compiled execution step (test
  `DumpDotAnnotatesCompiledSteps`).

### `RenderPassContext` (`render_graph_context.hpp`)

Trivial non-owning view constructed per pass invocation:
`RenderPassContext(RenderGraph&, SDL_GPUCommandBuffer*, SDL_GPURenderPass*)`.

- `SDL_GPUCommandBuffer* command_buffer() const noexcept` — the frame's cb
  (null under `begin_frame(nullptr)`).
- `SDL_GPURenderPass* render_pass() const noexcept` — non-null only inside
  managed graphics passes with attachments and a non-null cb.
- `SDL_GPUTexture* get_texture(ResourceHandle) const noexcept` — null-safe
  (null graph or invalid handle → `nullptr`); resolves
  `transient ? resolved_texture : imported_texture`.

### `TransientAllocator` (`transient_allocator.hpp/.cpp`)

Pools `SDL_GPUTexture`s keyed by exact `TextureDesc`. Non-copyable.

Data structures: `std::unordered_map<DescKey, std::vector<Entry>, DescKeyHash> pool_`
where `Entry { SDL_GPUTexture* tex; bool in_use; }`. `DescKeyHash` mixes all
seven desc fields with the boost-style combiner
`h ^= std::hash<uint64_t>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2)`.

API and behavior:

- `[[nodiscard]] bool init(gpu::GPUDevice& device)` — stores pointer,
  always returns `true`.
- `void shutdown()` — `evict_all()` + null device. Also run by destructor.
- `SDL_GPUTexture* acquire_texture(const TextureDesc& desc, const char* debug_name)`
  — increments `acquires_`; linear-scans the desc's bucket for a free entry
  (marks `in_use`, increments `reuses_`, returns it). On miss with a device:
  builds `SDL_GPUTextureCreateInfo` — `type` is
  `SDL_GPU_TEXTURETYPE_2D_ARRAY` when `array_layers > 1` else
  `SDL_GPU_TEXTURETYPE_2D` (no 3D/cube support); `sample_count` maps
  2→`SDL_GPU_SAMPLECOUNT_2`, 4→`_4`, 8→`_8`, anything else→`_1`;
  `props = 0` — then `SDL_CreateGPUTexture(device_->handle(), &info)`.
  On creation failure: `BRACKEN_LOG_ERROR("rg", "TransientAllocator: SDL_CreateGPUTexture failed for '{}'", ...)`
  and returns `nullptr` (failure is **not** cached in the bucket). On
  success: `SDL_SetGPUTextureName` with `debug_name` (if non-null), push
  `Entry{tex, true}`, return.
  **Test mode** (no `init()`, `device_ == nullptr`): fabricates a stable
  non-null sentinel `reinterpret_cast<SDL_GPUTexture*>(uintptr_t(&bucket) + bucket.size() + 1)`
  and pools it like a real texture, so reuse/aliasing logic is unit-testable
  headlessly. (Sentinels derive from the bucket's address; a map rehash
  between frames changes future sentinels, which is irrelevant to tests.)
- `void release_texture(const TextureDesc& desc, SDL_GPUTexture* tex)` —
  finds the entry by pointer in the desc bucket and clears `in_use`;
  silently ignores null/unknown textures or unknown descs.
- `void release_all()` — clears `in_use` on every entry (end-of-frame).
- `void evict_all()` — `SDL_ReleaseGPUTexture` on every pooled texture (or
  just clears the map in test mode), then clears the pool.
- Stats: `acquires_this_frame()`, `reuses_this_frame()` (reset by
  `begin_frame()`), `pool_size()` (total entries across buckets, O(buckets)).

The pool is unbounded: it grows to the high-water mark of same-desc
concurrent transients and never shrinks until `evict_all()`. After a
resolution change, old-desc textures are never matched again and their GPU
memory persists until eviction — hence `RenderGraph::evict_pool()` on
resize.

### Threading and lifetime contracts

- **No internal synchronization anywhere in the module** (no mutexes/atomics).
  All `RenderGraph`/`TransientAllocator` methods must be called from a single
  thread — in practice the render thread that owns the
  `SDL_GPUCommandBuffer`.
- `gpu::GPUDevice&` passed to `init()` is stored as a raw pointer and must
  outlive the graph/allocator (used in `acquire_texture` and `evict_all`).
- Imported textures/buffers are never owned, freed, or lifetime-tracked by
  the graph; the caller keeps them valid through `execute()`.
- Execute closures are stored as `std::function` in `PassNode`; reference
  captures must be valid when `execute()` runs. The typed `add_pass<Data>`
  keeps `Data` alive via `shared_ptr` captured in the closure (one heap
  allocation per typed pass per frame).
- `ResourceHandle`/pass indices are frame-scoped; `reset()` (and therefore
  `begin_frame()`) invalidates them.
- `PassBuilder` is valid only for the duration of its setup lambda; nested
  `add_pass` during setup is undefined (vector reallocation).

### Error handling summary

| Condition | Behavior |
|---|---|
| Dependency cycle | `compile()` logs error (tag `"rg"`), returns `false`; `execute()` no-ops |
| `read`/`write` by unknown name | Silent no-op, returns invalid handle |
| Re-declared transient name | Existing handle returned, new desc silently ignored |
| Re-imported name | Imported pointer/dims overwritten, `transient` forced false |
| `SDL_CreateGPUTexture` failure | Logged; `compile()` returns `false`, already-acquired transients released |
| `resource(h)` with bad handle | `assert` (debug); unchecked UB in release |
| `execute()` before successful `compile()` | Silent no-op |
| `begin_frame(nullptr)` | Valid test mode: callbacks run, no SDL render passes begun |

### Performance characteristics

- The graph is rebuilt from scratch every frame (immediate-mode declaration
  over a retained pool): per-frame `std::string` names, `std::function`
  setup/execute closures, `shared_ptr` per typed pass, `unordered_map` name
  lookups, and a per-managed-pass `std::vector<SDL_GPUColorTargetInfo>`.
  This constant cost is precisely the documented reason it is not in the
  frame path yet (`docs/engine_design_notes.md`).
- `topo_sort`: O(P + R + E); `compute_lifetimes`: O(total reads+writes);
  `allocate_transients`: O(steps × resources).
- Pool acquire is O(bucket length) linear scan; release is
  O(bucket length) pointer search.

### Test coverage (`src/bracken/render_graph/render_graph.test.cpp`)

All tests run headless (`RenderGraph` without `init()`, `begin_frame(nullptr)`):

| Test | Verifies |
|---|---|
| `RenderGraph.EmptyGraphCompiles` | Zero-pass graph compiles/executes/ends cleanly |
| `RenderGraph.LinearTopoOrder_ABC` | A→B→C chain via `create_color_target` + `read(name)` executes in order |
| `RenderGraph.DiamondTopoOrder` | Diamond A→{B,C}→D: A first, D last, both partial orders respected |
| `RenderGraph.TransientReuseSharesUnderlyingTexture` | Disjoint-lifetime, same-desc transients resolve to the same underlying texture; `last_reuse_count() >= 1` |
| `RenderGraph.ImportedTextureSurfacesUnchanged` | `import_texture` + `read` surfaces the exact imported pointer via `ctx.get_texture` |
| `RenderGraph.CycleDetectedAndSkipped` | Forced cycle (mutating `pass_at(0).reads`): `compile()` false, no execute callback runs |
| `RenderGraph.CompileFailsWhenTransientAllocationFails` | Failed transient texture creation fails `compile()` |
| `RenderGraph.PartialTransientAllocationFailureReleasesAcquired` | A mid-compile allocation failure releases already-acquired transients back to the pool |
| `RenderGraph.DumpDotProducesDigraph` | DOT output contains `digraph RenderGraph`, pass and resource names |
| `RenderGraph.DumpDotEscapesLabels` | Quotes/backslashes in names are escaped in DOT output |
| `RenderGraph.DumpDotAnnotatesCompiledSteps` | Post-compile DOT labels carry `[step N]` |
| `RenderGraph.TypedAddPassPassesDataThrough` | `add_pass<Data>` carries setup-written handle + value into execute |

Run: `ctest --test-dir build -R RenderGraph` or
`build/src/bracken/bracken_tests --gtest_filter='RenderGraph.*'`. There are
no benchmarks (`*.bench.cpp`) for this module.

### Known limitations, stubs, and silent behaviors

- **Not in the frame path** — by design; see Purpose and status. Nothing in
  `src/bracken/scene/` or `examples/` constructs a `RenderGraph`.
- **Textures only** — there is no buffer-resource API (declare or import);
  a former buffer scaffold was removed.
- **No resource versioning** — readers depend on all writers of a name
  regardless of declaration order.
- **No dead-pass culling, no barrier management, no subresource views**
  (whole-texture attachments only), no MSAA resolve handling, no
  mip/layer-targeted attachments (`SDL_GPUColorTargetInfo.mip_level/layer`
  left zero-initialized).
- **No per-pass GPU debug groups.**
- **Pool never self-evicts** — call `evict_pool()` on resize or stale-desc
  textures persist in GPU memory.

---

## DevUI Developer Overlay

### Scope and source layout

The developer overlay is the entire `src/bracken/dev/` module, namespace
`bracken::dev`:

| File | Contents |
|---|---|
| `src/bracken/dev/dev_ui.hpp` | `class DevUI` — public API, `Bindings`, pimpl declaration |
| `src/bracken/dev/dev_ui.cpp` | Full implementation **and** the no-op stub, selected by `#if defined(BRACKEN_DEV_UI) && BRACKEN_DEV_UI` |
| `src/bracken/dev/dev_panels.hpp` | Internal seam: `bracken::dev::detail::draw_builtin_panels(const DevUI::Bindings&, DevUI::PanelState&)` |
| `src/bracken/dev/dev_panels.cpp` | Built-in "Bracken Dev" window (tabs: Stats, Environment, Graphics, Camera); stubbed together with `dev_ui.cpp` |
| `src/bracken/dev/dev_ui.test.cpp` | Headless GoogleTest suite: 6 `DevUI` no-op-contract tests, valid in both configurations |
| `src/bracken/dev/dev_panels.test.cpp` | Headless GoogleTest suite: 16 ON-only `DevPanels.*` panel-body tests driving `draw_builtin_panels` with a real ImGui context and no GPU backend |

Integration points outside the module: `Application::init_dev_ui()` /
`Application::dev_ui()` (`src/bracken/application.{hpp,cpp}`), the
`SceneRenderer` post-UI callback (`src/bracken/scene/scene_renderer.{hpp,cpp}`),
and `InputHandler::set_keyboard_captured` (`src/bracken/input_handler.hpp`).
Canonical consumer: `examples/playground/playground.cpp`.

### Build configuration

Root `CMakeLists.txt`:

```cmake
option(BRACKEN_ENABLE_DEV_UI "Build the Dear ImGui developer overlay" ON)
```

Default **ON**. When ON, `FetchContent` pulls Dear ImGui at pinned tag
**`v1.92.2`** (ImGui ships no CMakeLists; `FetchContent_MakeAvailable` only
populates `imgui_SOURCE_DIR`). `src/bracken/CMakeLists.txt` then builds a
companion static library:

```cmake
add_library(bracken_imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdlgpu3.cpp
)
```

with `PUBLIC` include dirs (`${imgui_SOURCE_DIR}`,
`${imgui_SOURCE_DIR}/backends`), linked against SDL3
(static or shared per `VENDORED_SDL`), then
`target_link_libraries(bracken PUBLIC bracken_imgui)` and
`target_compile_definitions(bracken PUBLIC BRACKEN_DEV_UI=1)`. Both the
include paths and the define propagate to every consumer of `bracken`, which
is what lets host code `#include <imgui.h>` and `#ifdef BRACKEN_DEV_UI`.

When OFF: ImGui is not fetched, `bracken_imgui` does not exist,
`BRACKEN_DEV_UI` is undefined, and both `dev_ui.cpp` and `dev_panels.cpp`
compile their `#else` stub halves. Note `imgui_demo.cpp` is compiled in on
purpose in ON builds — `ImGui::ShowDemoWindow()` is callable from host panels.

### Public API surface (`src/bracken/dev/dev_ui.hpp`)

```cpp
namespace bracken::dev {

class DevUI {
public:
    DevUI();
    ~DevUI();                       // calls shutdown()
    // non-copyable, non-movable (all four special members = delete)

    struct Bindings {
        scene::SceneRenderer* scene_renderer = nullptr;
        systems::CameraController* camera = nullptr;
        GraphicsSettings* settings = nullptr;
        std::function<void(const GraphicsSettings&)> apply_settings;
        const float* fps = nullptr;
    };

    [[nodiscard]] bool init(render::RenderContext& context, Bindings bindings);
    void shutdown();
    [[nodiscard]] bool initialized() const;   // impl_ != nullptr

    bool process_event(const SDL_Event& event);
    void set_toggle_key(SDL_Scancode key);    // default SDL_SCANCODE_UNKNOWN

    [[nodiscard]] bool wants_keyboard() const;
    [[nodiscard]] bool wants_mouse() const;

    void render(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchain);

    void set_visible(bool visible);
    [[nodiscard]] bool visible() const;
    void toggle();

    void add_panel(std::string name, std::function<void()> draw);
};

} // namespace bracken::dev
```

Header dependencies are deliberately light: SDL event/GPU headers plus
forward declarations of `render::RenderContext`, `scene::SceneRenderer`,
`systems::CameraController`, `bracken::GraphicsSettings`. **No ImGui types
appear in the header**, so it is includable in stub builds.

#### Data members / memory layout

```cpp
struct Impl;                          // pimpl; nullptr until init and in stub builds
std::unique_ptr<Impl> impl_;
Bindings bindings_;
SDL_Scancode toggle_key_ = SDL_SCANCODE_UNKNOWN;
bool visible_ = false;
std::vector<std::pair<std::string, std::function<void()>>> panels_;
```

`impl_ != nullptr` is the single "initialized" signal (`initialized()`
returns exactly that). In the full build:

```cpp
struct DevUI::Impl {
    ImGuiContext* context = nullptr;   // owned ImGui context
    SDL_GPUDevice* device = nullptr;   // non-owning; for the idle wait at shutdown
    bool backends_initialized = false; // SDL3 + SDLGPU3 backends are up
};
```

In the stub build `struct DevUI::Impl {};` exists only to complete
`unique_ptr<Impl>`'s deleter; it is never instantiated.

`set_visible` / `visible` / `toggle` / `set_toggle_key` / `add_panel` /
`initialized` are inline in the header and therefore behave identically in
both configurations (pure state mutation, no backend contact).

#### `Bindings` semantics

All members are **non-owning** and must outlive the `DevUI`. Null members
hide the corresponding built-in panel section/tab (see tab-visibility rules
below). Under `Application::init_dev_ui()` the bindings are:
`scene_renderer_.get()`, `&camera()` (the `CameraController` facade),
`&graphics_settings_`, a lambda forwarding to
`apply_all_graphics_settings(s)`, and `&fps_` (the app's once-per-second FPS
counter). `bindings` is taken by value and moved into `bindings_` at `init`.

### Lifecycle

#### `init(render::RenderContext&, Bindings)` — full build

1. If `impl_` already exists → return `true` (idempotent; bindings are *not*
   re-bound on a second call).
2. `bindings_ = std::move(bindings)` — note this happens **before** backend
   init, so bindings are stored even if init subsequently fails.
3. `IMGUI_CHECKVERSION()`; `ImGui::CreateContext()`;
   `impl->device = context.device().handle()`.
4. `io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard`;
   `ImGui::StyleColorsDark()`.
5. `ImGui_ImplSDL3_InitForSDLGPU(context.window())` — on failure:
   `BRACKEN_LOG_ERROR("devui", ...)`, destroy the ImGui context, return
   `false`.
6. `ImGui_ImplSDLGPU3_Init` with `ImGui_ImplSDLGPU3_InitInfo{ .Device =
   context.device().handle(), .ColorTargetFormat =
   context.swapchain_format(), .MSAASamples = SDL_GPU_SAMPLECOUNT_1 }` — on
   failure: log, `ImGui_ImplSDL3_Shutdown()`, destroy context, return
   `false`.
7. Mark `backends_initialized = true`, publish `impl_`, log
   `BRACKEN_LOG_INFO("devui", "Dev UI initialized (Dear ImGui {})", IMGUI_VERSION)`,
   return `true`.

The overlay renders directly into the swapchain, so its pipeline target
format is the swapchain format (`RGBA8_UNORM` per
`src/bracken/gpu/pipeline_registry.hpp`), sample count 1. Error handling is
`bool` + log (boot-time init chain convention), with full unwind of partial
state on each failure path.

#### `shutdown()`

Idempotent (`!impl_` → return). If backends were initialized:
`SDL_WaitForGPUIdle(impl_->device)` first — the last frame's ImGui draw data
may still be in flight and the backends destroy their GPU objects
immediately — then `ImGui_ImplSDL3_Shutdown()`,
`ImGui_ImplSDLGPU3_Shutdown()`. Finally `ImGui::DestroyContext` and
`impl_.reset()`. The destructor calls `shutdown()`.

**Ordering contract:** the `DevUI` owns GPU resources and must be destroyed
before the GPU device. `Application::shutdown_renderer()` enforces this by
shutting down and resetting `dev_ui_` before `scene_renderer_` / `context_`
(`src/bracken/application.cpp`).

### Event routing — `process_event` (full build)

Algorithm, in order:

1. `!impl_` → return `false` (uninitialized never consumes; pinned by test
   `ProcessEventReturnsFalseWhenUninitialized`, including with
   `visible_ == true`).
2. `ImGui_ImplSDL3_ProcessEvent(&event)` — **unconditionally**, even while
   hidden, so ImGui's mouse/keyboard/window state is coherent the moment the
   overlay opens. (The resulting ImGui input queue is drained by `render()`
   every frame — see below.)
3. Toggle key: if `toggle_key_ != SDL_SCANCODE_UNKNOWN` and the event is
   `SDL_EVENT_KEY_DOWN` with `!event.key.repeat` and
   `event.key.scancode == toggle_key_` → `toggle()` and return `true`
   (consumed). Handled here rather than by the host so the overlay can
   always be closed even when ImGui keyboard capture (e.g. a focused
   `InputText`) would swallow a host binding.
4. `!visible_` → return `false` (hidden overlay never consumes non-toggle
   events).
5. Consumption switch while visible:
   - `SDL_EVENT_MOUSE_MOTION`, `SDL_EVENT_MOUSE_BUTTON_DOWN`,
     `SDL_EVENT_MOUSE_WHEEL` → return `io.WantCaptureMouse`.
   - `SDL_EVENT_KEY_DOWN`, `SDL_EVENT_TEXT_INPUT`,
     `SDL_EVENT_TEXT_EDITING`, `SDL_EVENT_TEXT_EDITING_CANDIDATES` →
     return `io.WantCaptureKeyboard`.
   - `SDL_EVENT_MOUSE_BUTTON_UP`, `SDL_EVENT_KEY_UP`, and everything else →
     return `false`. **Releases are never consumed by design:**
     `WantCapture*` lags one frame behind ImGui's click ownership, and a
     swallowed `KEY_UP`/`BUTTON_UP` would wedge `InputHandler`'s edge state
     (e.g. right-mouse relative mode stuck on).

`wants_keyboard()` / `wants_mouse()` return `false` when `!impl_ ||
!visible_`, otherwise `ImGui::GetIO().WantCaptureKeyboard` /
`.WantCaptureMouse`. They exist because polled input
(`SDL_GetKeyboardState`) bypasses event consumption entirely; hosts must
mute polled game input from these.

#### `Application` loop integration (`src/bracken/application.cpp`)

Per-event order in `Application::run`: `SDL_EVENT_QUIT` (handled before
everything) → `dev_ui_->process_event(event)` (skip rest if consumed) →
virtual `on_event(event)` (subclass; skip InputHandler if it returns
`true`) → `input_.process_event(event)`. After the poll loop, every frame:

```cpp
input_.set_keyboard_captured(dev_ui_ && dev_ui_->wants_keyboard());
```

`InputHandler::update_input_from_keyboard`
(`src/bracken/input_handler.cpp`) gates the polled WASD/arrow movement axes
on `!keyboard_captured_`; controller input is unaffected.
`set_keyboard_captured` is deliberately separate from
`set_game_input_enabled` so the two owners can't stomp each other.

### Frame rendering — `render(SDL_GPUCommandBuffer*, SDL_GPUTexture*)`

Execution context: must run inside the `SceneRenderer` post-UI callback —
i.e. during command-buffer recording, **outside any render pass** (the
backend records copy passes for uploads). Contracted main-thread-only, like
the whole frame path.

Full-build algorithm:

1. `!impl_` → return (no-op; pinned by `RenderIsNoOpWhenUninitialized` even
   with null arguments).
2. **Hidden/invalid-target path** (`!visible_ || cmd == nullptr ||
   swapchain == nullptr`): run
   `ImGui_ImplSDLGPU3_NewFrame(); ImGui_ImplSDL3_NewFrame(); ImGui::NewFrame(); ImGui::EndFrame();`
   and return. `EndFrame` without `Render` is legal and discards the frame.
   Purpose: `process_event` feeds ImGui unconditionally and only `NewFrame`
   consumes the queued input — without this drain, hours of hidden-overlay
   input would accumulate unboundedly and replay as phantom clicks the
   moment the overlay opens.
3. Visible path (`BRACKEN_PROFILE_ZONE("DevUI::render")`):
   - backend + ImGui `NewFrame`s;
   - `detail::draw_builtin_panels(bindings_)`;
   - for each registered panel `(name, draw)`: `ImGui::Begin(name.c_str())`,
     call `draw()` only if `Begin` returned `true` (false while collapsed —
     body skipped), always `ImGui::End()`;
   - `ImGui::Render()`; fetch `ImGui::GetDrawData()`; return early if null
     or `DisplaySize` is non-positive (minimized window — nothing to upload
     or draw);
   - `ImGui_ImplSDLGPU3_PrepareDrawData(draw_data, cmd)` — **mandatory
     before the render pass**: records vertex/index/texture uploads as copy
     passes on `cmd`;
   - one `SDL_GPUColorTargetInfo{ .texture = swapchain, .load_op =
     SDL_GPU_LOADOP_LOAD, .store_op = SDL_GPU_STOREOP_STORE }`;
     `SDL_BeginGPURenderPass(cmd, &target_info, 1, nullptr)` (no depth
     target), `ImGui_ImplSDLGPU3_RenderDrawData`, end pass. `LOADOP_LOAD` —
     the overlay composites over the finished frame, never clears.

#### Host draw hook: the `SceneRenderer` post-UI callback

`src/bracken/scene/scene_renderer.hpp`:

```cpp
using PostUICallback = std::function<void(SDL_GPUCommandBuffer*, SDL_GPUTexture*)>;
void set_post_ui_callback(PostUICallback cb);   // single slot, last set wins
```

Invoked in `SceneRenderer::render_frame` (`scene_renderer.cpp`) after the
`UI` pass, only when both `post_ui_callback_` and `current_swapchain_` are
non-null — the swapchain can still be null there (acquire failure on the
no-content / non-AO paths), and the callback is skipped rather than handed a
null target. The invocation is wrapped in `BRACKEN_PROFILE_ZONE_COLOR`,
`BRACKEN_GPU_SCOPE`, and `BRACKEN_GPU_TIMER` under the label
`"PostUICallback"`. It runs before the screenshot readback, so
`Application::request_screenshot` captures the overlay.

**Single-slot limitation:** `Application::init_dev_ui()` claims this slot
with `[this](cmd, swapchain) { dev_ui_->render(cmd, swapchain); }`; a host
cannot use both the overlay and its own post-UI callback
(documented in `docs/architecture.md`).

Consequence of the skip-on-null-swapchain behavior: on frames where the
callback is skipped, `render()`'s hidden-path input drain also does not run.
This is bounded (swapchain acquire failures / minimized windows), unlike the
unbounded accumulation the drain exists to prevent.

### Built-in panels (`src/bracken/dev/dev_panels.cpp`)

Internal seam: `namespace bracken::dev::detail`,
`void draw_builtin_panels(const DevUI::Bindings&, DevUI::PanelState&)` —
declared in `dev_panels.hpp`, called by `dev_ui.cpp` with the instance's own
`PanelState` (and directly by the ON-only panel tests), both halves compiled
out together in stub builds (stub is an empty function body). `PanelState`
(public nested struct on `DevUI`) holds the FPS plot ring buffer and the
Camera tab's shake sliders per instance — two `DevUI` instances never share
widget state.

Window: `ImGui::SetNextWindowSize(ImVec2(420.0f, 540.0f),
ImGuiCond_FirstUseEver)`; `ImGui::Begin("Bracken Dev")` (body skipped while
collapsed); tab bar id `"##bracken_dev_tabs"`. Each tab body is wrapped in
`ImGui::PushID` (`"stats"`, `"environment"`, `"graphics"`, `"camera"`).

Tab visibility conditions:

| Tab | Shown when |
|---|---|
| Stats | `bindings.fps != nullptr \|\| bindings.scene_renderer != nullptr` |
| Environment | `bindings.scene_renderer != nullptr` |
| Graphics | `bindings.settings != nullptr && bindings.apply_settings` |
| Camera | `bindings.camera != nullptr` |

The Graphics tab requires **both** the settings pointer and the apply
callback — without an apply path, edits could never take effect, so the tab
hides instead (pinned by `DevUIPanels.GraphicsTabHiddenWithoutApplyCallback`).

#### Stats tab (`draw_stats_tab`)

- FPS block (only when `fps` bound): 120-sample scrolling history in the
  instance's `PanelState::fps_history` ring buffer (per-instance — pinned by
  `DevUIPanels.FpsHistoryIsPerInstanceState`). Rendered via
  `ImGui::PlotLines` with a `%.1f fps` overlay and a derived
  `1000/fps` frame-time text.
- Render block (only when `scene_renderer` bound): calls
  `scene_renderer->set_collect_stats(true)` **every frame the tab is
  drawn**, and deliberately never resets it — hosts (e.g. the playground
  smoke driver) enable collection themselves and auto-disabling on overlay
  close would stomp their setting. Reads `SceneRenderer::render_stats()`
  (`bracken::RenderStats`, `src/bracken/contracts/render_stats.hpp`: `draw_calls`,
  `triangle_count`, `entities_rendered`, `entities_distance_culled`,
  `entities_frustum_culled`; also `jobs_pending` and `pass_times_ms`, which
  the tab does not display). Shows `Particles: N (in M effects)` from
  `SceneRenderer::frame_particle_count()` / `frame_effect_count()` — the
  rendered packet's snapshot counts (the live `EffectSystem` is sim-stage
  owned and is never read from the overlay).
  Ends with disabled-text disclosure that per-pass GPU times are stubbed
  (SDL3 limitation; `PassTiming::gpu_ms` is `nullopt` — `cpu_ms` is command-record time, not GPU cost).

#### Environment tab (`draw_environment_tab`)

Mutates `EnvironmentState& env = renderer.environment()` **in place** — the
intended live-edit path, since `render_frame` re-propagates `env_` to every
sub-renderer each frame (note: a host that calls
`FrameBuilder::set_environment` every frame overwrites these edits — the
live-edit path relies on the renderer keeping its own `env_` when
`has_environment()` is false). Constant:

```cpp
constexpr ImGuiColorEditFlags HDR_COLOR_FLAGS =
    ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR;
```

(all colors are linear HDR, tonemapped at composite; pickers must not clamp
to 0..1). Sun direction editing: `env.sun_dir` is the direction light
*travels*; the tab edits azimuth/elevation of the normalized direction *to*
the sun (`to_sun = -env.sun_dir`), re-decomposed **every frame** so external
writers (a host day/night cycle) keep the sliders in sync; degenerate
near-zero `to_sun` falls back to `(0,1,0)`. Recomposition on slider change:
`dir_to_sun = {cos(el)·cos(az), sin(el), cos(el)·sin(az)}`,
`env.sun_dir = -normalize(dir_to_sun)`. Slider ranges: azimuth −180…180°,
elevation −89…89°.

Fields edited (all on `bracken::render::EnvironmentState`,
`src/bracken/contracts/environment.hpp`): `sun_color`; `sky_zenith`,
`sky_horizon`, `ground_color`, `ambient_sky`, `ambient_ground`; a
"Clouds & haze" block with `cloud_coverage` (0–1), `cloud_density` (0–2),
`cloud_scale` (0.25–4), `cloud_speed` (0–8), `haze_strength` (0–1.5) and a
disabled-text note that the haze color follows the fog colors;
`fog_color_horizon`, `fog_color_sun`, `fog_start` (0–32000), `fog_end`
(0–32000), `fog_height_falloff` (0–0.05); `fog_volumetric_sun_color`,
`fog_volumetric_scatter` (0–2), `fog_volumetric_density` (0–0.02),
`fog_volumetric_height` (1–2000), `fog_volumetric_falloff` (0–0.05),
`fog_volumetric_start` (0–4000), `fog_volumetric_end` (100–32000). A
disabled-text note states the volumetric pass runs only when "Volumetric
fog" is enabled on the Graphics tab. A "Color grading" block edits
`grade_temperature`/`grade_tint` (sliders −1…1) and the CDL triples
`grade_slope` (0–4)/`grade_offset` (−1…1)/`grade_power` (0.1–4) as
`DragFloat3`s, with a "Reset grading" button restoring the all-neutral
defaults.

#### Graphics tab (`draw_graphics_tab`)

Precondition: `bindings.settings != nullptr` (dereferenced unconditionally —
the tab is only reachable when the visibility condition held). Uses a local
helper:

```cpp
template <typename E>
bool combo_enum(const char* label, E& value, const char* const items[], int count);
```

for `enum class E : int` with contiguous 0-based values (clamps the current
value into range, `static_cast`s back on change). Widgets OR their changed
flags into one `changed` bool; at the end:

```cpp
if (changed && bindings.apply_settings) {
    bindings.apply_settings(settings);
}
```

— push-on-change-only (the host apply path diffs internally, but the panel
avoids calling it every frame). Settings edited (`bracken::GraphicsSettings`,
`src/bracken/contracts/graphics_settings.hpp`):

- Enum combos: `shadow_mode` (`ShadowMode` Off/PCF/PCSS), `ao_mode`
  (`AOMode` Off/GTAO Half/GTAO Full), `aa_mode` (`AAMode` Off/FXAA/TAA), `tonemap_mode`
  (`TonemapMode` ACESFitted/ACESNarkowicz/Reinhard/None).
- Int-index combos: `shadow_cascades` (labels 1–4), `shadow_resolution`
  (512/1024/2048/4096), `local_shadow_resolution` ("Local shadow atlas",
  1024/2048/4096), `anisotropic_filter` (Off/2x/4x/8x/16x),
  `vsync_mode` (Immediate/VSync/Mailbox), `window_mode`
  (Windowed/Borderless/Exclusive), `draw_distance`
  (500…32000, 7 entries).
- Int sliders: `local_shadow_point_budget` ("Shadowed point lights", 0–8),
  `local_shadow_spot_budget` ("Shadowed spot lights", 0–16).
- Bool checkboxes: `local_shadows` ("Local light shadows"), `ibl_enabled`
  ("Image-based lighting"), `auto_exposure` ("Auto exposure"),
  `bloom_enabled`, `volumetric_fog`, `god_rays`,
  `fog_enabled`, `grass_enabled`, `skybox_enabled`, `trees_enabled`,
  `rocks_enabled`, `frustum_culling`, `show_fps`, `show_debug_hud`.
- Float sliders: `exposure` ("Exposure bias", 0.1–4),
  `auto_exposure_speed_up`/`_down` ("Adapt speed up/down", 0–10 EV/s),
  `contrast` (0.5–2), `saturation`
  (0–2), `ambient_strength` (0–1), `sun_intensity` (0–10),
  `bloom_strength` (0–1), `bloom_threshold` (0–4), `ao_strength` (0–2),
  `ao_radius` (0.1–4), `fog_density` (0–1). The EV clamp window
  (`auto_exposure_min/max_ev`) is deliberately not exposed here.
- Read-only: "Adapted EV: … (target …)" from
  `scene_renderer->auto_exposure()->latest_state()` (the ~once-per-second
  fenced readback; the line is absent until the first readback lands).

Under `Application`, `settings` points at the app's persisted
`graphics_settings_` and `apply_settings` is
`apply_all_graphics_settings`; since `shutdown_renderer()` auto-saves to
`settings_file_path()`, Graphics-tab edits persist across runs.

#### Camera tab (`draw_camera_tab`)

Operates on the abstract `systems::CameraController`
(`src/bracken/systems/camera_controller.hpp`). Readout: `get_position()`,
`get_forward()`, and — because the abstract controller exposes no yaw/pitch
getters — both derived from forward with the yaw convention of
`camera_system.cpp`:

```cpp
pitch_deg = degrees(asin(clamp(forward.y, -1, 1)));
yaw_deg   = degrees(atan2(-forward.x, -forward.z));
```

plus `get_config().distance` / `.fov` (`systems::CameraModeConfig`).
Interactive testers:

- Shake: intensity slider 0–3 (default 1.0), duration slider 0.05–2 s
  (default 0.3) — fire-and-forget parameters stored in the instance's
  `PanelState`; buttons call `add_shake(ShakeType::Impact | Heavy | Subtle,
  intensity, duration)` and `add_directional_shake(forward, intensity,
  duration)` (`ShakeType` enum: Impact, Heavy, Directional, Subtle).
- Framing: FOV-bias slider −30…30° → `set_fov_bias(bias)`, Reset button →
  `set_fov_bias(0)`; focus-strength slider 0–1 → `set_focus_strength(s)`.
  Both sliders **re-read the controller each frame** via `get_fov_bias()` /
  `get_focus_strength()`, so external setter calls move the sliders (same
  live-sync behavior as the Environment tab's sun sliders). Only the shake
  test parameters are panel-local — they are inputs to fire-and-forget
  calls, with no controller value to read back.

### `Application` integration API

`src/bracken/application.hpp` (protected facade):

```cpp
bool init_dev_ui();      // call after init_renderer(); false if compiled out / init failed
dev::DevUI& dev_ui();    // valid after init_dev_ui() — including when it returned false
```

`init_dev_ui()` behavior (`application.cpp`):

1. Always constructs `dev_ui_ = std::make_unique<dev::DevUI>()` first — even
   when the renderer is down or the overlay is compiled out — so `dev_ui()`
   and downstream `process_event()`/`render()` calls remain safe.
2. Returns `false` if `context_`, `scene_renderer_`, or `camera_` is null
   (renderer not initialized).
3. Builds `Bindings{ scene_renderer_.get(), &camera(), &graphics_settings_,
   [this](const GraphicsSettings& s) { apply_all_graphics_settings(s); },
   &fps_ }` and calls `dev_ui_->init(*context_, std::move(bindings))`;
   returns `false` on failure (stub or backend error).
4. On success, claims the single post-UI slot:
   `scene_renderer_->set_post_ui_callback([this](cmd, swapchain) {
   dev_ui_->render(cmd, swapchain); })` and returns `true`.

`dev_ui()` asserts (`BRACKEN_ASSERT`) that `init_dev_ui()` was called —
calling it beforehand is a programming error, not a soft failure.

Teardown: `shutdown_renderer()` shuts down and resets `dev_ui_` **before**
tearing down the camera, scene renderer, and render context ("Dev overlay
owns GPU resources — it must die before the device").

### Stub configuration (`BRACKEN_ENABLE_DEV_UI=OFF`)

`dev_ui.cpp` `#else` half — exact behavior of each out-of-line entry point:

| Method | Stub behavior |
|---|---|
| `init` | returns `false`; ignores arguments; `impl_` stays null |
| `shutdown` | empty |
| `process_event` | returns `false` always — **the toggle key is inert in stub builds** |
| `wants_keyboard` / `wants_mouse` | return `false` |
| `render` | empty |

Header-inline methods still mutate state (`set_visible(true)` makes
`visible()` return `true`), but nothing draws and no events are consumed —
e.g. the playground smoke driver's `dev_ui().set_visible(true)` at frame 170
is a documented harmless no-op in stub builds.
`detail::draw_builtin_panels` is an empty function in the stub half of
`dev_panels.cpp`.

### Headless test approach

Split across two files. `src/bracken/dev/dev_ui.test.cpp` holds the six
`DevUI` lifecycle tests — no `init()`, no GPU, no window, no ImGui backend
contact — so **every expectation must hold identically in both
configurations** (full and stub). They pin the no-op-safety contract:

| Test | Contract pinned |
|---|---|
| `ConstructsAndDestructsUninitialized` | fresh `DevUI`: `initialized() == false`, `visible() == false` |
| `AddPanelBeforeInitDoesNotCrashOrDraw` | `add_panel` before `init` stores only; callbacks never invoked |
| `VisibilityStateMachine` | `set_visible` idempotent; `toggle()` flips both ways |
| `ProcessEventReturnsFalseWhenUninitialized` | uninitialized never consumes, even when `visible_ == true` |
| `RenderIsNoOpWhenUninitialized` | `render(nullptr, nullptr)` safe, visible or not |
| `ShutdownWithoutInitIsSafe` | `shutdown()` idempotent without prior `init` |

`src/bracken/dev/dev_panels.test.cpp` holds the panel-body tests. They are
**ON-only** (`BRACKEN_ENABLE_DEV_UI=ON`): they drive `draw_builtin_panels`
through a real ImGui context (core only — `RendererHasTextures` makes
`NewFrame` software-build the font atlas; `EndFrame` without `Render()`
discards the frame) with no SDL/GPU backend, no window and no device. Tabs
other than the first only draw their BODY when selected, so the harness
forces the tab under test selected and then asserts it really is selected —
a "drawing this tab changes nothing" test cannot pass by never drawing it.
The bound `SceneRenderer` is default-constructed and never `init()`ed.

| Test | Contract pinned |
|---|---|
| `NoBindingsDrawsNoTabs` | every tab's bindings are its precondition |
| `AllFourTabsAppearWhenFullyBound` | Stats / Environment / Graphics / Camera all surface |
| `StatsTabAppearsWithFpsAlone` | Stats has two independent reasons to exist (fps OR renderer) |
| `GraphicsTabHiddenWithoutApplyCallback` | Graphics tab hides when `apply_settings` is empty |
| `GraphicsTabShownWithSettingsAndApply` | Graphics tab shows with both settings + apply bound |
| `FpsHistoryIsPerInstanceState` | FPS plot history lives in `PanelState`, not shared statics |
| `FpsHistoryCursorWrapsWithoutOverrunning` | the 120-sample plot is a ring, cursor never leaves the array |
| `StatsTabEnablesCollectionAndNeverDisablesIt` | Stats turns `set_collect_stats(true)` on and never back off |
| `PartialTimingBindingsDrawNothingAndDoNotCrash` | phase breakdown is all-or-nothing across the four timing pointers |
| `FullTimingBindingsDraw` | all four bound ⇒ the breakdown draws |
| `EnvironmentTabDrawsWithoutMutatingEnvironment` | drawing leaves `EnvironmentState` byte-identical (no sun round-trip drift) |
| `EnvironmentTabSurvivesDegenerateSunDirection` | zero-length `sun_dir` produces no NaN |
| `GraphicsTabDoesNotApplyOrEditWithoutInteraction` | no `apply_settings` call and no struct edit without a widget change |
| `GraphicsTabLeavesOutOfRangeEnumsAlone` | `combo_enum` clamps for display only, never writes back |
| `CameraTabDrawsWithoutPushingFramingSetters` | Camera tab re-reads controller values rather than pushing stale slider state; fires no shake |
| `CameraTabSurvivesStraightDownForward` | the asin clamp in the yaw/pitch readout |

Tests are auto-globbed into `bracken_tests` per the repo convention.
Anything requiring `init()` is integration-test territory: the playground
`--smoke` run flips the overlay visible at frame 170 and self-verifies a
screenshot (`playground_smoke.png`) that includes the overlay, because the
post-UI callback runs before the screenshot readback.

### Threading, lifetime, and performance summary

- **Main thread only** (header contract: "main thread only, like the rest of
  the frame path"). Stats/Camera widget state lives in the per-instance
  `PanelState`, so multiple `DevUI` instances don't share it.
- `Bindings` members are non-owning and must outlive the `DevUI`.
- `DevUI` must be shut down before the `SDL_GPUDevice`; `shutdown()` blocks
  on `SDL_WaitForGPUIdle` before destroying backend GPU objects.
- Panel `draw` callbacks run during command-buffer recording within the
  ImGui frame; a collapsed window skips the body.
- Hidden-but-initialized steady-state cost per frame: one
  `ImGui_ImplSDL3_ProcessEvent` per SDL event plus one discarded
  `NewFrame`/`EndFrame` pair in `render()` (the input-queue drain).
- Visible cost: one CPU ImGui frame + one copy-pass upload + one
  load/store swapchain render pass, profiled under `DevUI::render` (CPU) and
  `PostUICallback` (GPU scope/timer — but note GPU timer results are
  stubbed zeros).

### Known limitations and silent behaviors

- **Single post-UI callback slot** — `init_dev_ui()` claims it; a later
  `set_post_ui_callback` silently replaces the overlay's draw.
- **Toggle key does nothing in stub builds** (stub `process_event` returns
  `false` unconditionally) and when `init` failed (`impl_` null).
- **`init()` on an already-initialized overlay warns and returns `false`**
  — re-init is an error, not a rebind; `shutdown()` first to re-point.
- **Stats collection is turned on by the Stats tab and never turned off.**
- **Per-pass GPU timings are zeros** (`GPUTimestampPool` stub; SDL3
  limitation) — surfaced as disabled text in the Stats tab.
- **Release events always pass through** to the host, including releases
  whose matching press the overlay consumed (one-frame `WantCapture*` lag by
  design).
- **Graphics tab requires both `settings` and `apply_settings`** — it hides
  when either is missing.
- `imgui_demo.cpp` is compiled into `bracken_imgui`, adding its footprint to
  dev builds (and enabling `ImGui::ShowDemoWindow()` in host panels).

---

## Physics (Jolt)

`src/bracken/physics/` — rigid-body simulation over vendored **Jolt Physics
v5.2.0** (FetchContent in `src/bracken/CMakeLists.txt`, `SOURCE_SUBDIR
Build`, linked **PRIVATE** into `bracken`; sample/test/viewer targets, LTO,
and `-Werror` are disabled on the Jolt build). Public surface:
`physics_world.hpp` only — glm + POD types over a pimpl; Jolt types and
`JPH_*` compile definitions never cross into consumer code.
`physics/internal/jolt_job_system.hpp` is an internal header (it includes Jolt;
only `physics/*.cpp` may include it). `<module>/internal/` is the engine-wide
convention for headers outside the public surface, and
`public_headers.test.cpp` enforces it: no header outside such a directory may
transitively reach Jolt.

### Configuration (`PhysicsConfig`)

| Field | Default | Meaning |
|---|---|---|
| `gravity` | `{0, -9.81, 0}` | World gravity, m/s² |
| `max_bodies` | 4096 | Hard body cap (`add_*` fails past it, logged) |
| `max_body_pairs` | 4096 | Broadphase pair buffer per step |
| `max_contact_constraints` | 2048 | Contact buffer per step (overflow → one-time warning; contacts dropped by Jolt) |
| `fixed_dt` | 1/60 s | Simulation step size |
| `max_steps_per_update` | 4 | Step cap per `update()` call |

Each world also owns a 16 MB Jolt `TempAllocatorImpl` for step-internal
allocations. Jolt's process-globals (default allocator, `Factory`, type
registry, trace/assert hooks routed to `BRACKEN_LOG_*`) are registered once
via `std::call_once` and never unregistered.

### Fixed-step semantics

`update(dt)` accumulates `dt` and steps `PhysicsSystem::Update(fixed_dt, 1,
…)` while `accumulator >= fixed_dt`, at most `max_steps_per_update` times.
Remaining debt beyond one further `fixed_dt` is **clamped away** (`fmod`)
with a once-per-world warning — the simulation slows under sustained
overload instead of feeding back into longer frames. Calling `update` with
`dt == fixed_dt` steps exactly once per call (float-exact: the same value is
added and subtracted). Body adds mark the broadphase dirty;
`OptimizeBroadPhase` runs at the top of the next `update`.

### Layer model

Two object layers with matching broadphase layers, wired through the
standard Jolt trio (`BroadPhaseLayerInterface`,
`ObjectVsBroadPhaseLayerFilter`, `ObjectLayerPairFilter`):

| Object layer | Broadphase layer | Members | Collides with |
|---|---|---|---|
| `static` (0) | `static` (0) | `BodyType::Static`, heightfields | moving only |
| `moving` (1) | `moving` (1) | `BodyType::Dynamic` **and** `Kinematic` | everything |

### Bodies & shapes

`BodyId` is the raw `JPH::BodyID` u32 (index + sequence number; stale ids of
destroyed bodies never alias live ones — queries on them return identity
values). Dynamic bodies with `mass > 0` use Jolt's `CalculateInertia`
override (shape inertia distribution scaled to the given mass). Box shapes
clamp Jolt's convex radius to half the smallest half-extent. Heightfields:
`HeightFieldShapeSettings` with block size 2 — the sample count must be a
multiple of the block size and ≥ 4, so non-conforming grids are padded by
duplicating the last row/column (surface extends < 1 cell past `world_size`
on +x/+z); grid spacing is `world_size / (resolution − 1)`; samples are
world heights (y-scale 1); the body sits at `origin`.

### Queries

`cast_ray(origin, direction_and_length)` — closest hit against any body
(`RayHit`: position, surface normal, fraction, body).
`collide_capsule(center, half_height_of_cylinder, radius, out_contacts,
include_moving = false, ignore_body = INVALID_BODY_ID)` — overlap-tests an
upright capsule (Jolt `NarrowPhaseQuery::CollideShape`) and writes one
`ShapeContact { normal, penetration, body }` per overlapping body into the
caller's span, returning the count. Contacts are sorted deepest-first with a
`BodyId` tiebreak, so output is deterministic regardless of broadphase
traversal order and truncation keeps the contacts that matter. `normal` is
the unit push direction for the *queried* capsule (Jolt's
`-mPenetrationAxis`, normalized); moving the capsule `penetration` along it
separates the pair. Static-only by default (broadphase + object layer
filters pinned to the static layer); `include_moving` widens to all layers;
`ignore_body` maps to `IgnoreSingleBodyFilter` (pass a terrain heightfield
the caller grounds against separately, or the caller's own body when
`include_moving`). Returns 0 on an uninitialized world, empty span, or
non-positive dimensions. The playground's character controller is the
reference use: XZ-only depenetration against statics each sim frame, with
the heightfield excluded so slope contacts can't shove the capsule.

### Job-system bridge & threading

`JoltJobSystem : JPH::JobSystemWithBarrier` mirrors
`JPH::JobSystemThreadPool` (pooled `Job` records in a `FixedSizeFreeList`;
`QueueJob(s)` AddRef + submit a `job->Execute(); job->Release();` thunk;
`GetMaxConcurrency()` = bracken worker count) but submits onto
`core::jobs::JobSystem` — one pool owns the cores, per
`docs/multithreading_plan.md` §7. Thunks use the zero-handle
`TaskGroup` submit path; the bridge destructor drains the group with a
helping wait. Jolt's `WaitForJobs` executes ready barrier jobs on the
waiting thread, so `update()` progresses even from a busy or 1-worker pool.
`PhysicsWorld` itself is tier **owned / sim-stage** — single logical owner,
no internal synchronization of host calls, no GPU/SDL/main-thread coupling
anywhere in the module.

### Determinism

Identical construction sequence + identical `update()` sequence ⇒
bit-identical body transforms, independent of worker count (Jolt's
determinism guarantee; pinned by
`physics_world.test.cpp::DeterministicAcrossWorldsOnSamePool`). The claim
holds per binary/platform — not across compilers, CPUs with different FMA
behavior, or Jolt versions.

### Known limitations

- No collision-event/contact-listener surface yet; poll `is_active`/ray
  casts, or extend the API when a host needs callbacks.
- No shape sharing across bodies (each `add_*` builds its own shape) and no
  compound/mesh/convex-hull shapes yet.
- `cast_ray` is closest-hit only, no layer filtering.
- Queries return identity values rather than errors for stale/invalid ids.
