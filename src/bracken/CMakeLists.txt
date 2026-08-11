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
# SYSTEM, same reasoning as tinygltf below: Jolt/Core/Array.h trips -Wshadow
# (GCC flags its `pointer` typedef against the inherited member), and a
# vendored header must never be able to fail our -Werror build.
get_target_property(BRACKEN_JOLT_INCLUDES Jolt INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(Jolt PROPERTIES
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${BRACKEN_JOLT_INCLUDES}")

# Compile shaders (HLSL -> SPIRV) at build time
# Also fetches SDL_shadercross for both CLI (build-time) and library (runtime)
add_subdirectory(shaders)

file(GLOB_RECURSE BRACKEN_SOURCES CONFIGURE_DEPENDS "*.cpp")
# Exclude test and benchmark files from the engine library
list(FILTER BRACKEN_SOURCES EXCLUDE REGEX ".*\\.(test|bench)\\.cpp$")
add_library(bracken STATIC ${BRACKEN_SOURCES})
bracken_target_warnings(bracken)

# Namespaced alias. Consumers should write `bracken::bracken` whether they got
# the engine via add_subdirectory or find_package: a missing find_package fails
# at configure time with a clear message, whereas a bare `bracken` silently
# becomes a `-lbracken` link flag against nothing.
add_library(bracken::bracken ALIAS bracken)

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
#
# NOTE, and it is the constraint bracken_compile_shaders below is built around:
# the `rm -rf` here wipes `<exe dir>/shaders` on EVERY build. Anything a host
# stages into that same directory races the wipe and disappears intermittently,
# which is why host shaders get their own STAGE_DIR.
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

# ---------------------------------------------------------------------------
# bracken_compile_shaders(<target> ...) — compile a GAME's own HLSL with the
# engine's pinned SDL_shadercross/DXC and stage the SPIR-V next to its exe.
#
#   bracken_compile_shaders(my_game
#       DIRECTORY    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
#       [SOURCES     a.vert.hlsl b.frag.hlsl ...]
#       [STAGE_DIR   shaders_my_game]      # default: shaders_<target>
#       [INCLUDE_DIRS <dir> ...]
#       [WAVE_OPS    c.comp.hlsl ...])     # bare file names, see below
#
# At runtime the game points the registry at the staged directory:
#   registry.add_shader_search_path(std::string(SDL_GetBasePath()) + "shaders_my_game/");
#
# Same toolchain as the engine, deliberately: the build-time compiler and the
# runtime SDL3_shadercross library the engine links must be the same version, so
# this uses the `shadercross` target built from the pinned FetchContent tree
# rather than anything found on PATH.
#
# TWO DIRECTORIES THAT MUST NOT BE REUSED, both learned from a real failure:
#  * the .spv output NEVER goes in ${BRACKEN_SHADER_OUTPUT_DIR}. The engine's
#    own suite (gpu/uniform_layout.test.cpp) enumerates every .spv in there and
#    fails on any uniform block it has no row for, so a host cbuffer landing
#    there fails the ENGINE's tests.
#  * STAGE_DIR must not be "shaders". bracken_copy_shaders rm -rf's that
#    directory next to the exe on every build.
#
# WAVE_OPS is the host equivalent of the engine's SHADER_WAVE_OPS list and
# carries the same warning: the shadercross CLI never passes -fspv-target-env,
# so DXC defaults to vulkan1.0 where every Wave* intrinsic is a hard compile
# error. Listing a file here routes it through the vendored `dxc` at
# vulkan1.1 with byte-identical -fspv-* flags — and RAISES THE VULKAN FLOOR for
# your game's pipelines that use it. It is an explicit list, not a convention,
# so that hardware requirement is visible in a diff.
function(bracken_compile_shaders target)
    cmake_parse_arguments(BCS "" "DIRECTORY;STAGE_DIR" "SOURCES;INCLUDE_DIRS;WAVE_OPS" ${ARGN})
    if(BCS_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "bracken_compile_shaders(${target}): unknown argument(s): ${BCS_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT TARGET ${target})
        message(FATAL_ERROR "bracken_compile_shaders: '${target}' is not a target")
    endif()
    if(NOT TARGET shadercross)
        message(FATAL_ERROR
            "bracken_compile_shaders(${target}): the `shadercross` CLI target does not exist. It is built "
            "as part of bracken's shader subtree, so call this AFTER add_subdirectory(<bracken>).")
    endif()

    if(NOT BCS_STAGE_DIR)
        set(BCS_STAGE_DIR "shaders_${target}")
    endif()
    if(BCS_STAGE_DIR STREQUAL "shaders")
        message(FATAL_ERROR
            "bracken_compile_shaders(${target}): STAGE_DIR must not be \"shaders\" — bracken_copy_shaders "
            "deletes and re-copies that directory on every build, so your .spv would vanish intermittently.")
    endif()

    # Sources: an explicit list, a globbed directory, or both.
    set(_bcs_sources ${BCS_SOURCES})
    set(_bcs_includes "")
    set(_bcs_header_deps "")
    if(BCS_DIRECTORY)
        file(GLOB_RECURSE _bcs_dir_sources CONFIGURE_DEPENDS "${BCS_DIRECTORY}/*.hlsl")
        file(GLOB_RECURSE _bcs_dir_headers CONFIGURE_DEPENDS "${BCS_DIRECTORY}/*.hlsli")
        list(APPEND _bcs_sources ${_bcs_dir_sources})
        list(APPEND _bcs_header_deps ${_bcs_dir_headers})
        # DIRECTORY goes FIRST — see the one-include-dir note below; it is the
        # root a host's own .hlsli lives in, so it is the one that must win.
        list(APPEND _bcs_includes "${BCS_DIRECTORY}")
    endif()
    list(APPEND _bcs_includes ${BCS_INCLUDE_DIRS})
    if(NOT _bcs_sources)
        message(FATAL_ERROR "bracken_compile_shaders(${target}): no .hlsl sources (pass DIRECTORY and/or SOURCES)")
    endif()

    # The engine's .hlsli root, last. Note that editing an engine .hlsli does
    # NOT retrigger host compilation: the engine's own glob owns that
    # dependency, and re-running the configure step is what picks it up.
    if(BRACKEN_SHADER_INCLUDE_DIR)
        list(APPEND _bcs_includes "${BRACKEN_SHADER_INCLUDE_DIR}")
    endif()
    list(REMOVE_DUPLICATES _bcs_includes)

    # THE SHADERCROSS CLI ACCEPTS EXACTLY ONE -I ("ERROR: '-I' can only be used
    # once"), and it does NOT resolve a quoted #include relative to the source
    # file either — it hands the source to DXC as an in-memory buffer named
    # hlsl.hlsl, so an include that is not under the single -I root is simply
    # not found. The engine's own shader build never noticed because it has one
    # root; a host has at least two the moment the engine's .hlsli root is in
    # play. So: the shadercross path gets the FIRST root only (the host's own
    # DIRECTORY by default), and the DXC/WAVE_OPS path — real DXC, which does
    # accept repeats — gets all of them.
    set(_bcs_include_flags "")
    foreach(_dir IN LISTS _bcs_includes)
        list(APPEND _bcs_include_flags -I "${_dir}")
    endforeach()
    set(_bcs_primary_include_flag "")
    list(LENGTH _bcs_includes _bcs_include_count)
    if(_bcs_include_count GREATER 0)
        list(GET _bcs_includes 0 _bcs_primary_include)
        set(_bcs_primary_include_flag -I "${_bcs_primary_include}")
    endif()
    if(_bcs_include_count GREATER 1)
        message(STATUS "bracken_compile_shaders(${target}): shadercross takes ONE include dir; using "
                       "${_bcs_primary_include} (the rest apply only to WAVE_OPS shaders, which go through dxc)")
    endif()

    if(BCS_WAVE_OPS AND NOT TARGET dxc)
        message(FATAL_ERROR
            "bracken_compile_shaders(${target}): WAVE_OPS is non-empty but the vendored DXC CLI target `dxc` "
            "does not exist (SDLSHADERCROSS_DXC=OFF?). Wave-intrinsic shaders cannot be compiled without it.")
    endif()

    set(_bcs_out_dir "${CMAKE_CURRENT_BINARY_DIR}/spv_${target}")
    if(_bcs_out_dir STREQUAL "${BRACKEN_SHADER_OUTPUT_DIR}")
        message(FATAL_ERROR "bracken_compile_shaders(${target}): output dir collides with the engine's own")
    endif()
    file(MAKE_DIRECTORY "${_bcs_out_dir}")

    set(_bcs_outputs "")
    foreach(_src IN LISTS _bcs_sources)
        cmake_path(GET _src FILENAME _name)
        # Stage + entry point come from the file name, exactly as the engine's
        # own shader build derives them.
        if(_name MATCHES "\\.vert\\.hlsl$")
            set(_stage "vertex")
            set(_entry "VSMain")
            set(_profile "vs_6_0")
            string(REPLACE ".vert.hlsl" ".vert.spv" _out_name "${_name}")
        elseif(_name MATCHES "\\.frag\\.hlsl$")
            set(_stage "fragment")
            set(_entry "PSMain")
            set(_profile "ps_6_0")
            string(REPLACE ".frag.hlsl" ".frag.spv" _out_name "${_name}")
        elseif(_name MATCHES "\\.comp\\.hlsl$")
            set(_stage "compute")
            set(_entry "CSMain")
            set(_profile "cs_6_0")
            string(REPLACE ".comp.hlsl" ".comp.spv" _out_name "${_name}")
        else()
            message(WARNING "bracken_compile_shaders(${target}): unknown shader type '${_name}', skipping "
                            "(expected {name}.vert.hlsl / .frag.hlsl / .comp.hlsl)")
            continue()
        endif()

        set(_out "${_bcs_out_dir}/${_out_name}")
        if(_name IN_LIST BCS_WAVE_OPS)
            add_custom_command(
                OUTPUT "${_out}"
                COMMAND "$<TARGET_FILE:dxc>"
                    -T ${_profile}
                    -E ${_entry}
                    -spirv
                    -fspv-target-env=vulkan1.1
                    -fspv-flatten-resource-arrays
                    -fspv-preserve-bindings
                    -fspv-preserve-interface
                    ${_bcs_include_flags}
                    "${_src}"
                    -Fo "${_out}"
                DEPENDS "${_src}" ${_bcs_header_deps} dxc
                COMMENT "Compiling ${_name} for ${target} (wave ops, vulkan1.1)"
                VERBATIM
            )
        else()
            add_custom_command(
                OUTPUT "${_out}"
                COMMAND "$<TARGET_FILE:shadercross>"
                    "${_src}"
                    -s HLSL
                    -d SPIRV
                    -t ${_stage}
                    -e ${_entry}
                    ${_bcs_primary_include_flag}
                    -o "${_out}"
                DEPENDS "${_src}" ${_bcs_header_deps} shadercross
                COMMENT "Compiling ${_name} for ${target}"
                VERBATIM
            )
        endif()
        list(APPEND _bcs_outputs "${_out}")
    endforeach()

    if(NOT _bcs_outputs)
        message(FATAL_ERROR "bracken_compile_shaders(${target}): no shader matched the .vert/.frag/.comp convention")
    endif()

    # Prune stale outputs at configure time, same reasoning as the engine's:
    # deleting or renaming a .hlsl re-runs configure (CONFIGURE_DEPENDS), and an
    # orphaned .spv would otherwise keep shipping.
    file(GLOB _bcs_stale "${_bcs_out_dir}/*.spv")
    if(_bcs_stale)
        list(REMOVE_ITEM _bcs_stale ${_bcs_outputs})
    endif()
    if(_bcs_stale)
        file(REMOVE ${_bcs_stale})
    endif()

    add_custom_target(${target}_compile_shaders ALL DEPENDS ${_bcs_outputs})
    add_custom_target(${target}_stage_shaders ALL
        COMMAND ${CMAKE_COMMAND} -E rm -rf
            "$<TARGET_FILE_DIR:${target}>/${BCS_STAGE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${_bcs_out_dir}"
            "$<TARGET_FILE_DIR:${target}>/${BCS_STAGE_DIR}"
        COMMENT "Staging ${target} host shaders to ${BCS_STAGE_DIR}"
    )
    add_dependencies(${target}_stage_shaders ${target}_compile_shaders)
    add_dependencies(${target} ${target}_stage_shaders)

    list(LENGTH _bcs_outputs _bcs_count)
    message(STATUS "bracken_compile_shaders(${target}): ${_bcs_count} shader(s) -> ${BCS_STAGE_DIR}/")
endfunction()

# Every header in the public umbrella must compile STANDALONE — on its own, in
# a TU with nothing included before it. That is a compiler question, not
# something a source scan can answer, so generate one TU per header and build
# them. The header list is read out of bracken.hpp itself (one source of
# truth); CMAKE_CONFIGURE_DEPENDS re-runs configure when the umbrella changes.
#
# public_headers.test.cpp checks the include GRAPH (no Jolt/ImGui/tinygltf/stb_
# outside <module>/internal/, the umbrella covers the reference consumer);
# this target checks compilability. Neither substitutes for the other.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/bracken.hpp")
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/bracken.hpp" BRACKEN_UMBRELLA_LINES
     REGEX "^[ \t]*#[ \t]*include[ \t]*\"bracken/")
set(BRACKEN_PUBLIC_HEADER_TUS "")
foreach(line IN LISTS BRACKEN_UMBRELLA_LINES)
    string(REGEX REPLACE "^[ \t]*#[ \t]*include[ \t]*\"(bracken/[^\"]+)\".*$" "\\1" hdr "${line}")
    string(REGEX REPLACE "[/.]" "_" slug "${hdr}")
    set(tu "${CMAKE_CURRENT_BINARY_DIR}/public_headers/${slug}.cpp")
    # Included twice on purpose: the second include proves the include guard.
    file(CONFIGURE OUTPUT "${tu}" CONTENT
"// GENERATED by src/bracken/CMakeLists.txt from bracken.hpp — do not edit.
// Proves that \"${hdr}\" is self-sufficient: nothing included before it.
#include \"${hdr}\"
#include \"${hdr}\"
namespace { [[maybe_unused]] constexpr int bracken_standalone_${slug} = 0; }
" @ONLY)
    list(APPEND BRACKEN_PUBLIC_HEADER_TUS "${tu}")
endforeach()
list(LENGTH BRACKEN_PUBLIC_HEADER_TUS BRACKEN_PUBLIC_HEADER_COUNT)
if(BRACKEN_PUBLIC_HEADER_COUNT LESS 20)
    message(FATAL_ERROR
        "src/bracken/bracken.hpp parsed as only ${BRACKEN_PUBLIC_HEADER_COUNT} public headers — "
        "the umbrella or the parser above is broken")
endif()
add_library(bracken_public_headers OBJECT ${BRACKEN_PUBLIC_HEADER_TUS})
bracken_target_warnings(bracken_public_headers)
target_link_libraries(bracken_public_headers PRIVATE bracken)

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
