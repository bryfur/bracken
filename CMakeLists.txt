cmake_minimum_required(VERSION 3.20)
project(bracken VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(BrackenWarnings)

include(FetchContent)

# Build all vendored (FetchContent) dependencies as static libraries. A plain
# (non-cache) variable: it shadows any cache value within this directory tree
# and everything added below it (including all FetchContent deps), but does
# not leak into a consuming super-project's own targets or override its
# BUILD_SHARED_LIBS choice.
set(BUILD_SHARED_LIBS OFF)

# Release-live contracts (host-boundary preconditions). ON by default: these
# guard the API boundary a game calls into — packet state, seal-once, lane
# index, facade-before-init — where the failure mode without them is a
# segfault or silent corruption in a shipped build, not a debug-only mistake.
# Each is one compare and one branch, off any per-item loop; hot per-item
# checks stay on BRACKEN_ASSERT and remain debug-only. Turn OFF only with a
# measurement showing a contract on your profile.
option(BRACKEN_ENABLE_CONTRACTS "Compile host-boundary contracts into release builds" ON)

# Tracy profiler (opt-in; OFF means zero overhead)
option(BRACKEN_ENABLE_TRACY "Enable Tracy profiler instrumentation" OFF)
if(BRACKEN_ENABLE_TRACY)
    FetchContent_Declare(
        tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG v0.11.1
    )
    set(TRACY_ENABLE ON CACHE BOOL "" FORCE)
    set(TRACY_ON_DEMAND ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(tracy)
endif()

# Unit tests
option(BRACKEN_BUILD_TESTS "Build unit tests" ON)

# Optional ctest registration of `playground --smoke` (see examples/playground).
# Declared here — not in examples/ — so enable_testing() below also covers
# integration-only configurations (-DBRACKEN_BUILD_TESTS=OFF
# -DBRACKEN_PLAYGROUND_SMOKE_TEST=ON), where ctest would otherwise silently
# find no tests and exit 0.
option(BRACKEN_PLAYGROUND_SMOKE_TEST "Register 'playground --smoke' as a ctest (requires GPU/display)" OFF)

if(BRACKEN_BUILD_TESTS OR BRACKEN_PLAYGROUND_SMOKE_TEST)
    enable_testing()
endif()

if(BRACKEN_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Google Benchmark (micro-benchmarks for CPU-side systems)
option(BRACKEN_BUILD_BENCHMARKS "Build bracken_benchmarks (Google Benchmark)" ON)
if(BRACKEN_BUILD_BENCHMARKS AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(WARNING
        "BRACKEN_BUILD_BENCHMARKS=ON with no CMAKE_BUILD_TYPE: benchmarks will compile "
        "at -O0 with arena poisoning and live thread-tier asserts. Numbers measured in "
        "this tree are not meaningful. Use `cmake --preset bench` (build-bench/) instead.")
endif()
if(BRACKEN_BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    # benchmark v1.9.1 builds itself with -Werror -pedantic-errors, which
    # clang ≥ 22 trips on (__COUNTER__ in benchmark.h is a C2y extension).
    # Vendored code — its warnings are not ours to fix: drop -Werror, and
    # silence the C2y diag that -pedantic-errors upgrades on its own.
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(benchmark PRIVATE -Wno-c2y-extensions)
        target_compile_options(benchmark_main PRIVATE -Wno-c2y-extensions)
    endif()
endif()

# Dear ImGui developer overlay (bracken::dev::DevUI). ON by default for
# development; turn OFF for shipping builds — the DevUI class compiles to a
# no-op stub and ImGui is neither fetched nor linked.
option(BRACKEN_ENABLE_DEV_UI "Build the Dear ImGui developer overlay" ON)
if(BRACKEN_ENABLE_DEV_UI)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.92.2
    )
    # ImGui ships no CMakeLists; MakeAvailable just populates imgui_SOURCE_DIR
    # and src/bracken/CMakeLists.txt compiles the sources it needs.
    FetchContent_MakeAvailable(imgui)
endif()

# The engine library, its shader-compilation pipeline, and its third-party
# dependencies (SDL3, glm, tinygltf, SDL_shadercross) are self-contained in
# this subdirectory.
add_subdirectory(src/bracken)

# Example apps (playground — the example game / integration test)
option(BRACKEN_BUILD_EXAMPLES "Build example apps" ON)
if(BRACKEN_BUILD_EXAMPLES)
    add_subdirectory(examples)
elseif(BRACKEN_PLAYGROUND_SMOKE_TEST)
    message(WARNING "BRACKEN_PLAYGROUND_SMOKE_TEST=ON has no effect: the "
                    "playground is not built (BRACKEN_BUILD_EXAMPLES=OFF)")
endif()
