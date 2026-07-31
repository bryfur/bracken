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
