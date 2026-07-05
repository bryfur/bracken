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

# Compile shaders (HLSL -> SPIRV) at build time
# Also fetches SDL_shadercross for both CLI (build-time) and library (runtime)
add_subdirectory(shaders)

file(GLOB_RECURSE BRACKEN_SOURCES CONFIGURE_DEPENDS "*.cpp")
# Exclude test and benchmark files from the engine library
list(FILTER BRACKEN_SOURCES EXCLUDE REGEX ".*\\.(test|bench)\\.cpp$")
add_library(bracken STATIC ${BRACKEN_SOURCES})

target_include_directories(bracken PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${tinygltf_SOURCE_DIR}
    ${SDL_shadercross_SOURCE_DIR}/include
)

target_compile_definitions(bracken PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)

target_link_libraries(bracken PUBLIC
    $<IF:$<BOOL:${VENDORED_SDL}>,SDL3::SDL3-static,SDL3::SDL3>
    $<IF:$<BOOL:${VENDORED_SDL}>,SDL3_ttf::SDL3_ttf-static,SDL3_ttf::SDL3_ttf>
    $<IF:$<BOOL:${VENDORED_SDL}>,SDL3_image::SDL3_image-static,SDL3_image::SDL3_image>
    SDL3_shadercross::SDL3_shadercross-static
    glm::glm
)

if(BRACKEN_ENABLE_TRACY)
    target_link_libraries(bracken PUBLIC TracyClient)
    target_compile_definitions(bracken PUBLIC BRACKEN_ENABLE_TRACY=1)
endif()

# Ensure shaders are compiled before engine
add_dependencies(bracken compile_shaders)

# Store shader output dir in cache so it's globally accessible
set(BRACKEN_SHADER_OUTPUT_DIR "${SHADER_OUTPUT_DIR}" CACHE INTERNAL "Path to compiled shaders")

# Function for executables that link bracken to copy shaders next to their binary.
# The copy_shaders target always runs after compile_shaders and copies the directory.
function(bracken_copy_shaders target)
    add_custom_target(${target}_copy_shaders ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${BRACKEN_SHADER_OUTPUT_DIR}"
            "$<TARGET_FILE_DIR:${target}>/shaders"
        COMMENT "Copying shaders to ${target} output directory"
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
        target_link_libraries(bracken_benchmarks PRIVATE bracken benchmark::benchmark_main)
        target_include_directories(bracken_benchmarks PRIVATE ${CMAKE_SOURCE_DIR}/src)
    endif()
endif()

# Unit tests
if(BRACKEN_BUILD_TESTS)
    file(GLOB_RECURSE BRACKEN_TEST_SOURCES CONFIGURE_DEPENDS "*.test.cpp")
    if(BRACKEN_TEST_SOURCES)
        add_executable(bracken_tests ${BRACKEN_TEST_SOURCES})
        target_link_libraries(bracken_tests PRIVATE bracken GTest::gtest_main)
        target_include_directories(bracken_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

        include(GoogleTest)
        gtest_discover_tests(bracken_tests)
    endif()
endif()
