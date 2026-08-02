# Project Rules

## Build Rules
- There is ONE build tree: `build/`. Do not create `build-tsan/`, `build-rel/`, or any other parallel tree — sanitizers and build types are `-D` flags on `build/`.
- NEVER run a full clean rebuild (`--clean-first`, `cmake --build build --clean-first`, `rm -rf build`). Incremental builds only. Re-running `cmake -S . -B build` with different `-D` flags is fine: it reconfigures in place and keeps the fetched dependencies.
- The first configure fetches and builds SDL_shadercross (DXC + SPIRV-Tools) from source — this is the slow part of a cold build. Never throw it away.
- Always use `cmake --build build` (or `ninja -C build`) for incremental builds.
- New source files under `src/bracken/` are auto-discovered by CMake `GLOB_RECURSE` — just rebuild incrementally, no need to reconfigure for a new file (a new file does trigger a CMake re-run, but not a from-scratch dependency rebuild).
