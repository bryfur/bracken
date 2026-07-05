# Project Rules

## Build Rules
- NEVER run a full clean rebuild (`--clean-first`, `cmake --build build --clean-first`, `rm -rf build`, or reconfiguring over an existing `build/`). Incremental builds only.
- The first configure fetches and builds SDL_shadercross (DXC + SPIRV-Tools) from source — this is the slow part of a cold build. Never throw it away.
- Always use `cmake --build build` (or `ninja -C build`) for incremental builds.
- New source files under `src/bracken/` are auto-discovered by CMake `GLOB_RECURSE` — just rebuild incrementally, no need to reconfigure for a new file (a new file does trigger a CMake re-run, but not a from-scratch dependency rebuild).
