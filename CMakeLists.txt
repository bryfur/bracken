cmake_minimum_required(VERSION 3.20)
project(mmo_engine VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# Static linking for all vendored dependencies
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Tracy profiler (opt-in; OFF means zero overhead)
option(MMO_ENABLE_TRACY "Enable Tracy profiler instrumentation" OFF)
if(MMO_ENABLE_TRACY)
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
option(MMO_BUILD_TESTS "Build unit tests" ON)
if(MMO_BUILD_TESTS)
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
option(MMO_BUILD_BENCHMARKS "Build mmo_engine_benchmarks (Google Benchmark)" ON)
if(MMO_BUILD_BENCHMARKS)
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

# The engine library, its shader-compilation pipeline, and its third-party
# dependencies (SDL3, glm, tinygltf, SDL_shadercross) are self-contained in
# this subdirectory.
add_subdirectory(src/engine)

# Example apps (playground — the example game / integration test)
option(MMO_BUILD_EXAMPLES "Build example apps" ON)
if(MMO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
