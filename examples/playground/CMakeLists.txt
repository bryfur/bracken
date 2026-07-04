# Playground — the example game / integration test. See the header comment in
# playground.hpp for what it covers.
add_executable(playground
    main.cpp
    playground.cpp
    character.cpp
)
target_link_libraries(playground PRIVATE mmo_engine)
mmo_engine_copy_shaders(playground)

# Optional ctest registration: needs a GPU + display, so OFF by default.
# Enable with -DMMO_PLAYGROUND_SMOKE_TEST=ON where a GPU is available.
option(MMO_PLAYGROUND_SMOKE_TEST "Register 'playground --smoke' as a ctest (requires GPU/display)" OFF)
if(MMO_PLAYGROUND_SMOKE_TEST)
    add_test(NAME playground_smoke
             COMMAND playground --smoke 240
             WORKING_DIRECTORY $<TARGET_FILE_DIR:playground>)
    set_tests_properties(playground_smoke PROPERTIES LABELS "integration" TIMEOUT 180)
endif()
