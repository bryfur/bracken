cmake_minimum_required(VERSION 3.20)
project(bracken VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# Static linking for all vendored dependencies
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

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
if(BRACKEN_BUILD_TESTS)
    enable_testing()

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
if(BRACKEN_BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
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
endif()
