# Playground — the example game / integration test. See the header comment in
# playground.hpp for what it covers.
add_executable(playground
    main.cpp
    playground.cpp
    custom_pass.cpp
    scene_layout.cpp
    level_export.cpp
    character.cpp
    day_weather.cpp
    music.cpp
    sfx.cpp
)
bracken_target_warnings(playground)
# The namespaced alias, deliberately: this is the reference consumer, so it
# links the engine the way a game should (docs/user_guide.md "Consuming the
# engine from a game project").
target_link_libraries(playground PRIVATE bracken::bracken)
bracken_copy_shaders(playground)

# The reference consumer compiles its OWN HLSL with the engine's pinned
# shadercross/DXC and stages the SPIR-V into `shaders_playground/` next to the
# exe — a different directory from `shaders/`, which bracken_copy_shaders wipes
# on every build. Compiled by every build so the toolchain is exercised
# continuously rather than discovered broken by a game; REGISTERED at runtime
# only under PLAYGROUND_CUSTOM_PASS=1, because `--smoke` is the screenshot
# regression gate and custom_pass.cpp paints on the frame. custom_pass.{hpp,cpp}
# is the worked example of the whole loop; docs/gap_pipelines.md walks it.
bracken_compile_shaders(playground DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

# Medieval village assets — 52 .glb, 4 terrain textures and village_layout.h,
# all tracked in git, all referenced by name from playground.cpp. They are
# staged next to the exe because the playground resolves them at runtime
# relative to its own directory.
#
# The EXISTS guard is not for the normal case (the files are in the repo); it
# is so a sparse/partial checkout still configures and builds, falling back to
# the procedural-only scene. village_layout.h is guarded by __has_include for
# the same reason.
set(PLAYGROUND_ASSETS ${CMAKE_CURRENT_SOURCE_DIR}/assets)
if(EXISTS ${PLAYGROUND_ASSETS}/models/village.glb)
    target_include_directories(playground PRIVATE ${PLAYGROUND_ASSETS})
    add_custom_command(TARGET playground POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${PLAYGROUND_ASSETS}/textures $<TARGET_FILE_DIR:playground>/assets/textures
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${PLAYGROUND_ASSETS}/models $<TARGET_FILE_DIR:playground>/assets/models
        COMMENT "Staging playground village assets")
else()
    message(STATUS "playground: assets/models/village.glb not found — building the "
                   "procedural-only fallback scene")
endif()

# Optional ctest registration: needs a GPU + display, so OFF by default.
# Enable with -DBRACKEN_PLAYGROUND_SMOKE_TEST=ON where a GPU is available.
# The option itself is declared in the top-level CMakeLists.txt so that
# enable_testing() is called even when BRACKEN_BUILD_TESTS is OFF.
if(BRACKEN_PLAYGROUND_SMOKE_TEST)
    add_test(NAME playground_smoke
             COMMAND playground --smoke 240
             WORKING_DIRECTORY $<TARGET_FILE_DIR:playground>)
    set_tests_properties(playground_smoke PROPERTIES LABELS "integration" TIMEOUT 180)
endif()
