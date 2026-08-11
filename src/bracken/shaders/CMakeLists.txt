# Shader compilation - HLSL to SPIRV at build time

# Fetch SDL_shadercross for shader compilation
# Provides: shadercross CLI (build-time), SDL3_shadercross library (runtime)
include(FetchContent)
FetchContent_Declare(
    SDL_shadercross
    GIT_REPOSITORY https://github.com/libsdl-org/SDL_shadercross.git
    GIT_TAG 7b7365a86611b2a7b6462e521cf1c43a037d0970
    GIT_SUBMODULES_RECURSE TRUE
)
set(SDLSHADERCROSS_CLI ON CACHE BOOL "" FORCE)
set(SDLSHADERCROSS_SHARED OFF CACHE BOOL "" FORCE)
set(SDLSHADERCROSS_STATIC ON CACHE BOOL "" FORCE)
set(SDLSHADERCROSS_INSTALL OFF CACHE BOOL "" FORCE)
set(SDLSHADERCROSS_VENDORED ON CACHE BOOL "" FORCE)
set(SDLSHADERCROSS_SPIRVCROSS_SHARED ON CACHE BOOL "" FORCE)
set(SDLSHADERCROSS_DXC ON CACHE BOOL "" FORCE)
# The vendored DXC/SPIRV static archives get linked into shared outputs
# (libdxcompiler.so, libspirv-cross-c-shared.so), so they must be PIC. A plain
# (non-cache) variable: it scopes to this directory and the fetched subtrees
# below it, without forcing PIC on the rest of the engine or on a consuming
# super-project.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Disable tests from vendored dependencies (SPIRV-Tools, DXC, etc.)
set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
# libc++ ≥ 22 marks standard type traits [[clang::no_specializations]], which
# hard-errors on DXC's vendored LLVM-3.7-era std::is_nothrow_constructible
# specializations (llvm/ADT/StringRef.h). Plain variable, same scoping as
# CMAKE_POSITION_INDEPENDENT_CODE above: fetched subtrees only.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-invalid-specialization")
endif()
FetchContent_MakeAvailable(SDL_shadercross)

# Always compile with the CLI pinned and built above — never a system binary —
# so the build-time compiler and the runtime SDL3_shadercross library the
# engine links stay the same version. (A SHADERCROSS_EXECUTABLE entry cached
# by older checkouts' find_program is intentionally ignored.)
set(SHADERCROSS_COMMAND "$<TARGET_FILE:shadercross>")

# Output directory for compiled shaders. Dedicated to .spv files only —
# bracken_copy_shaders mirrors it verbatim next to consuming executables.
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/spv")
file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

# Collect all HLSL source files (CONFIGURE_DEPENDS re-runs CMake when files change)
file(GLOB_RECURSE SHADER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hlsl"
)

# Collect shared include files (.hlsli) as dependencies for recompilation.
# GLOB_RECURSE to match the source glob above — the compiler is invoked with
# -I ${CMAKE_CURRENT_SOURCE_DIR}, so nested includes compile fine and must
# also trigger recompilation.
file(GLOB_RECURSE SHADER_INCLUDES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hlsli"
)

# ---------------------------------------------------------------------------
# Shaders that need SM6 wave (subgroup) intrinsics.
#
# The shadercross CLI never passes -fspv-target-env, so DXC defaults to
# vulkan1.0, under which every Wave* intrinsic is a hard compile error
# ("Vulkan 1.1 is required for Wave Operation but not permitted to use").
# There is no CLI flag or SDL_ShaderCross property to raise it.
#
# So these shaders go through the DXC CLI directly — the same DXC, already
# built as part of the vendored SDL_shadercross tree, just its `dxc` target
# rather than the library — with -fspv-target-env=vulkan1.1 and otherwise
# byte-identical flags to what shadercross passes, so the binding model the
# runtime reflection sees is unchanged.
#
# THIS IS AN EXPLICIT LIST, NOT A CONVENTION, because adding a file to it
# raises the Vulkan floor for the pipelines that use it from 1.0 to 1.1. That
# is a hardware-support decision and it should be visible in a diff, not the
# silent consequence of typing WaveActiveBallot in a new shader.
#
# The Metal path needs nothing here: SPIRV-Cross maps these to simd_* and the
# runtime already asks for MSL 2.2.0 (gpu_compute.cpp / gpu_shader.cpp), above
# the 2.1 floor Metal requires for subgroup ops beyond broadcast/shuffle.
# Keep this list to shaders that ACTUALLY use a Wave* intrinsic. It is the
# record of which pipelines require Vulkan 1.1, so a speculative entry is a
# false claim about the engine's hardware requirements.
set(SHADER_WAVE_OPS
    rt_shadow_denoise_filter.comp.hlsl
)

if(SHADER_WAVE_OPS AND NOT TARGET dxc)
    message(FATAL_ERROR
        "SHADER_WAVE_OPS is non-empty but the vendored DXC CLI target `dxc` does not "
        "exist. It comes from SDL_shadercross's DirectXShaderCompiler subtree and "
        "requires SDLSHADERCROSS_DXC=ON. Without it, wave-intrinsic shaders cannot be "
        "compiled: the shadercross CLI targets vulkan1.0, where Wave* is rejected.")
endif()

set(COMPILED_SHADERS "")

foreach(SHADER_SOURCE IN LISTS SHADER_SOURCES)
    cmake_path(GET SHADER_SOURCE FILENAME SHADER_NAME)

    # Determine shader stage and entry point from filename convention:
    #   name.vert.hlsl -> vertex shader (VSMain)
    #   name.frag.hlsl -> fragment shader (PSMain)
    #   name.comp.hlsl -> compute shader (CSMain)
    if(SHADER_NAME MATCHES "\\.vert\\.hlsl$")
        set(SHADER_STAGE "vertex")
        set(ENTRY_POINT "VSMain")
        string(REPLACE ".vert.hlsl" ".vert.spv" OUTPUT_NAME "${SHADER_NAME}")
    elseif(SHADER_NAME MATCHES "\\.frag\\.hlsl$")
        set(SHADER_STAGE "fragment")
        set(ENTRY_POINT "PSMain")
        string(REPLACE ".frag.hlsl" ".frag.spv" OUTPUT_NAME "${SHADER_NAME}")
    elseif(SHADER_NAME MATCHES "\\.comp\\.hlsl$")
        set(SHADER_STAGE "compute")
        set(ENTRY_POINT "CSMain")
        string(REPLACE ".comp.hlsl" ".comp.spv" OUTPUT_NAME "${SHADER_NAME}")
    else()
        message(WARNING "Unknown shader type: ${SHADER_NAME}, skipping")
        continue()
    endif()

    set(OUTPUT_PATH "${SHADER_OUTPUT_DIR}/${OUTPUT_NAME}")

    if(SHADER_NAME IN_LIST SHADER_WAVE_OPS)
        # DXC directly, at vulkan1.1. The -fspv-* flags below are exactly the
        # ones SDL_shadercross passes (SDL_shadercross.c), so these blobs are
        # indistinguishable from the rest to the runtime reflection; only the
        # target environment differs.
        if(SHADER_STAGE STREQUAL "vertex")
            set(DXC_PROFILE "vs_6_0")
        elseif(SHADER_STAGE STREQUAL "fragment")
            set(DXC_PROFILE "ps_6_0")
        else()
            set(DXC_PROFILE "cs_6_0")
        endif()
        add_custom_command(
            OUTPUT "${OUTPUT_PATH}"
            COMMAND "$<TARGET_FILE:dxc>"
                -T ${DXC_PROFILE}
                -E ${ENTRY_POINT}
                -spirv
                -fspv-target-env=vulkan1.1
                -fspv-flatten-resource-arrays
                -fspv-preserve-bindings
                -fspv-preserve-interface
                -I "${CMAKE_CURRENT_SOURCE_DIR}"
                "${SHADER_SOURCE}"
                -Fo "${OUTPUT_PATH}"
            DEPENDS "${SHADER_SOURCE}" ${SHADER_INCLUDES} dxc
            COMMENT "Compiling ${SHADER_NAME} (wave ops, vulkan1.1)"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${OUTPUT_PATH}"
            COMMAND ${SHADERCROSS_COMMAND}
                "${SHADER_SOURCE}"
                -s HLSL
                -d SPIRV
                -t ${SHADER_STAGE}
                -e ${ENTRY_POINT}
                -I "${CMAKE_CURRENT_SOURCE_DIR}"
                -o "${OUTPUT_PATH}"
            DEPENDS "${SHADER_SOURCE}" ${SHADER_INCLUDES} shadercross
            COMMENT "Compiling ${SHADER_NAME}"
            VERBATIM
        )
    endif()

    list(APPEND COMPILED_SHADERS "${OUTPUT_PATH}")
endforeach()

# Prune stale outputs: deleting/renaming a .hlsl re-runs the configure step
# (CONFIGURE_DEPENDS above), so orphaned .spv files are removed here and can't
# keep shipping via bracken_copy_shaders.
file(GLOB STALE_SHADERS "${SHADER_OUTPUT_DIR}/*.spv")
if(COMPILED_SHADERS)
    list(REMOVE_ITEM STALE_SHADERS ${COMPILED_SHADERS})
endif()
if(STALE_SHADERS)
    file(REMOVE ${STALE_SHADERS})
    list(LENGTH STALE_SHADERS STALE_SHADER_COUNT)
    message(STATUS "Removed ${STALE_SHADER_COUNT} stale compiled shader(s) from ${SHADER_OUTPUT_DIR}")
endif()

add_custom_target(compile_shaders ALL DEPENDS ${COMPILED_SHADERS})

list(LENGTH COMPILED_SHADERS SHADER_COUNT)
message(STATUS "Configured ${SHADER_COUNT} shaders for compilation")

# Export paths to parent scope for runtime library linking and shader lookup
set(SDL_shadercross_SOURCE_DIR ${SDL_shadercross_SOURCE_DIR} PARENT_SCOPE)
set(SHADER_OUTPUT_DIR "${SHADER_OUTPUT_DIR}" PARENT_SCOPE)

# The engine's .hlsli include root, for bracken_compile_shaders (a consuming
# game's own HLSL compiled with this same pinned toolchain). Cached rather than
# PARENT_SCOPE because the consumer's CMakeLists is not this directory's parent.
set(BRACKEN_SHADER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL
    "Engine .hlsli include root for bracken_compile_shaders")
