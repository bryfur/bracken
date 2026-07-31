# Required SDL versions
set(SDL3_REQUIRED_VERSION "3.4.0")
set(SDL3_TTF_REQUIRED_VERSION "3.2.2")
set(SDL3_IMAGE_REQUIRED_VERSION "3.4.0")

# Try to find system SDL3 libraries at the required versions
find_package(SDL3 ${SDL3_REQUIRED_VERSION} QUIET)
find_package(SDL3_ttf ${SDL3_TTF_REQUIRED_VERSION} QUIET)
find_package(SDL3_image ${SDL3_IMAGE_REQUIRED_VERSION} QUIET)

if(SDL3_FOUND AND SDL3_ttf_FOUND AND SDL3_image_FOUND)
    set(VENDORED_SDL OFF CACHE INTERNAL "Whether SDL is vendored from source")
    message(STATUS "Using system SDL3 (${SDL3_VERSION}), SDL3_ttf (${SDL3_ttf_VERSION}), SDL3_image (${SDL3_image_VERSION})")
else()
    set(VENDORED_SDL ON CACHE INTERNAL "Whether SDL is vendored from source")
    message(STATUS "System SDL3 libs not found or wrong version, vendoring from source")

    # SDL3
    FetchContent_Declare(
        SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG release-${SDL3_REQUIRED_VERSION}
    )
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_X11_XTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(SDL3)

    # SDL3_ttf
    FetchContent_Declare(
        SDL3_ttf
        GIT_REPOSITORY https://github.com/libsdl-org/SDL_ttf.git
        GIT_TAG release-${SDL3_TTF_REQUIRED_VERSION}
    )
    set(SDLTTF_VENDORED ON CACHE BOOL "" FORCE)
    set(SDLTTF_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(SDL3_ttf)

    # SDL3_image
    FetchContent_Declare(
        SDL3_image
        GIT_REPOSITORY https://github.com/libsdl-org/SDL_image.git
        GIT_TAG release-${SDL3_IMAGE_REQUIRED_VERSION}
    )
    set(SDLIMAGE_VENDORED ON CACHE BOOL "" FORCE)
    set(SDLIMAGE_AVIF OFF CACHE BOOL "" FORCE)
    set(SDLIMAGE_DEPS_SHARED OFF CACHE BOOL "" FORCE)
    set(SDLIMAGE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(SDL3_image)
endif()

# Fetch tinygltf (model loading)
FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG v2.9.7
)
set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinygltf)

# GLM (header-only math library)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.3
)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glm)

# Jolt Physics (rigid-body backend for bracken::physics). The CMake project
# lives in the repo's Build/ subdirectory. Linked PRIVATE below and never
# included from a public bracken header (physics_world.hpp is pimpl +
# glm-only), so Jolt types and its JPH_* compile definitions stay inside the
# engine's own translation units.
FetchContent_Declare(
    JoltPhysics
    GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
    GIT_TAG v5.2.0
    SOURCE_SUBDIR "Build"
)
set(TARGET_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(TARGET_HELLO_WORLD OFF CACHE BOOL "" FORCE)
set(TARGET_PERFORMANCE_TEST OFF CACHE BOOL "" FORCE)
set(TARGET_SAMPLES OFF CACHE BOOL "" FORCE)
set(TARGET_VIEWER OFF CACHE BOOL "" FORCE)
set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
set(ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
# Jolt defaults to -fno-rtti; bracken builds with RTTI and derives from
# JPH::JobSystemWithBarrier (physics/jolt_job_system.hpp), which needs the
# base's typeinfo emitted — keep Jolt's RTTI ABI matched to ours.
set(CPP_RTTI_ENABLED ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(JoltPhysics)

# Compile shaders (HLSL -> SPIRV) at build time
# Also fetches SDL_shadercross for both CLI (build-time) and library (runtime)
add_subdirectory(shaders)

file(GLOB_RECURSE BRACKEN_SOURCES CONFIGURE_DEPENDS "*.cpp")
# Exclude test and benchmark files from the engine library
list(FILTER BRACKEN_SOURCES EXCLUDE REGEX ".*\\.(test|bench)\\.cpp$")
add_library(bracken STATIC ${BRACKEN_SOURCES})
bracken_target_warnings(bracken)

# PROJECT_SOURCE_DIR (the engine root, where project(bracken) is declared) —
# not CMAKE_SOURCE_DIR, which resolves to the super-project's root when the
# engine is consumed via add_subdirectory.
target_include_directories(bracken PUBLIC ${PROJECT_SOURCE_DIR}/src)

# SYSTEM: tinygltf's bundled json.hpp trips -Wdeprecated-literal-operator, and a
# vendored header must never be able to fail our -Werror build.
target_include_directories(bracken SYSTEM PUBLIC ${tinygltf_SOURCE_DIR})
# NOTE: no SDL_shadercross include dir here. FetchContent populates
# SDL3_shadercross_SOURCE_DIR (note the "3"), so the old
# ${SDL_shadercross_SOURCE_DIR}/include entry expanded to a literal "/include"
# and put a bogus -I on all 139 TUs. The real headers arrive via the
# SDL3_shadercross::SDL3_shadercross-static INTERFACE include dirs below.

target_compile_definitions(bracken PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)

# PUBLIC: BRACKEN_CONTRACT lives in headers a host includes (frame_packet.hpp,
# frame_builder.hpp), so engine and host must agree on whether it is compiled.
target_compile_definitions(bracken PUBLIC
    BRACKEN_CONTRACTS_ENABLED=$<IF:$<BOOL:${BRACKEN_ENABLE_CONTRACTS}>,1,0>)

target_link_libraries(bracken PUBLIC
    $<IF:$<BOOL:${VENDORED_SDL}>,SDL3::SDL3-static,SDL3::SDL3>
    $<IF:$<BOOL:${VENDORED_SDL}>,SDL3_ttf::SDL3_ttf-static,SDL3_ttf::SDL3_ttf>
    $<IF:$<BOOL:${VENDORED_SDL}>,SDL3_image::SDL3_image-static,SDL3_image::SDL3_image>
    SDL3_shadercross::SDL3_shadercross-static
    glm::glm
)

# PRIVATE by design: Jolt headers (and their JPH_* definitions) are an
# implementation detail of src/bracken/physics/ — consumers see only the
# glm-typed bracken::physics API. Static archives still propagate the link
# requirement ($<LINK_ONLY>), so executables linking `bracken` get Jolt's
# symbols without its include paths.
target_link_libraries(bracken PRIVATE Jolt)

if(BRACKEN_ENABLE_TRACY)
    target_link_libraries(bracken PUBLIC TracyClient)
    target_compile_definitions(bracken PUBLIC BRACKEN_ENABLE_TRACY=1)
endif()

# Dear ImGui developer overlay: compile ImGui core + SDL3/SDLGPU3 backends into
# a companion static lib. When OFF, dev/dev_ui.cpp compiles to a no-op stub and
# nothing here is built or linked.
if(BRACKEN_ENABLE_DEV_UI)
    add_library(bracken_imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_sdlgpu3.cpp
    )
    target_include_directories(bracken_imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )
    target_link_libraries(bracken_imgui PUBLIC
        $<IF:$<BOOL:${VENDORED_SDL}>,SDL3::SDL3-static,SDL3::SDL3>
    )
    target_link_libraries(bracken PUBLIC bracken_imgui)
    target_compile_definitions(bracken PUBLIC BRACKEN_DEV_UI=1)
endif()

# Ensure shaders are compiled before engine
add_dependencies(bracken compile_shaders)

# Store shader output dir in cache so it's globally accessible
set(BRACKEN_SHADER_OUTPUT_DIR "${SHADER_OUTPUT_DIR}" CACHE INTERNAL "Path to compiled shaders")

# Function for executables that link bracken to stage shaders next to their
# binary. The copy_shaders target always runs after compile_shaders; the
# staging directory is removed and recreated each build so deleted/renamed
# shaders don't keep shipping as stale .spv files.
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

# Micro-benchmarks — live next to their unit as *.bench.cpp, auto-globbed.
# Run: ./build/src/bracken/bracken_benchmarks [--benchmark_filter=...]
if(BRACKEN_BUILD_BENCHMARKS)
    file(GLOB_RECURSE BRACKEN_BENCH_SOURCES CONFIGURE_DEPENDS "*.bench.cpp")
    if(BRACKEN_BENCH_SOURCES)
        add_executable(bracken_benchmarks ${BRACKEN_BENCH_SOURCES})
        bracken_target_warnings(bracken_benchmarks)
        target_link_libraries(bracken_benchmarks PRIVATE bracken benchmark::benchmark_main)
        target_include_directories(bracken_benchmarks PRIVATE ${PROJECT_SOURCE_DIR}/src)
    endif()
endif()

# Unit tests
if(BRACKEN_BUILD_TESTS)
    file(GLOB_RECURSE BRACKEN_TEST_SOURCES CONFIGURE_DEPENDS "*.test.cpp")
    if(BRACKEN_TEST_SOURCES)
        add_executable(bracken_tests ${BRACKEN_TEST_SOURCES})
        bracken_target_warnings(bracken_tests)
        target_link_libraries(bracken_tests PRIVATE bracken GTest::gtest_main)
        target_include_directories(bracken_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
        # Source-scanning tests (logger_discipline) need the tree they are
        # policing. Absent (e.g. a relocated test binary) they GTEST_SKIP.
        target_compile_definitions(bracken_tests PRIVATE BRACKEN_REPO_ROOT="${PROJECT_SOURCE_DIR}")
        # uniform_layout.test.cpp reflects the COMPILED SPIR-V to check that
        # every CPU uniform struct matches the cbuffer the shader declares.
        # Absent (relocated binary) it GTEST_SKIPs.
        target_compile_definitions(bracken_tests PRIVATE BRACKEN_SHADER_SPV_DIR="${BRACKEN_SHADER_OUTPUT_DIR}")
        add_dependencies(bracken_tests compile_shaders)

        include(GoogleTest)
        gtest_discover_tests(bracken_tests)
    endif()
endif()
