#pragma once

// Profiling
// ---------
// Enable Tracy at CMake configure: -DMMO_ENABLE_TRACY=ON (default OFF, zero cost).
// Run the game, launch tracy-profiler GUI, click Connect (loopback by default).
//
// Macros:
//   ENGINE_PROFILE_FRAME()          - mark a frame boundary (call once per frame)
//   ENGINE_PROFILE_FRAME_NAMED(n)   - named frame boundary (string literal)
//   ENGINE_PROFILE_ZONE(name)       - scoped zone with literal name
//   ENGINE_PROFILE_ZONE_COLOR(n, c) - scoped zone with literal name + 0xRRGGBB color
//   ENGINE_PROFILE_ZONE_TAG(t)      - scoped zone, dynamic tag string attached
//   ENGINE_PROFILE_PLOT(n, v)       - emit a numeric sample on plot n
//   ENGINE_PROFILE_MESSAGE(m)       - one-shot string message on the timeline
//   ENGINE_PROFILE_ALLOC(ptr, sz)   - track an allocation (memory view)
//   ENGINE_PROFILE_FREE(ptr)        - track a free (pair with ENGINE_PROFILE_ALLOC)
//   ENGINE_PROFILE_THREAD(name)     - name the current thread in captures
//
// GPU note: SDL3 GPU does not expose backend queues/devices, so Tracy's GPU
// zones (TracyVkZone etc.) cannot be wired up. GPU timing = RenderDoc/Nsight.
//
// Suggested zone colors (keep passes visually consistent in captures):
//   0x3D6E9E update  |  0x8E44AD animation  |  0x27AE60 culling/scene
//   0xE67E22 shadows |  0xC0392B main pass  |  0x16A085 post-processing

#if defined(MMO_ENABLE_TRACY) && MMO_ENABLE_TRACY

#include <cstring>
#include <tracy/Tracy.hpp>

#define ENGINE_PROFILE_FRAME() FrameMark
#define ENGINE_PROFILE_FRAME_NAMED(n) FrameMarkNamed(n)
#define ENGINE_PROFILE_ZONE(name) ZoneScopedN(name)
#define ENGINE_PROFILE_ZONE_COLOR(name, color) ZoneScopedNC(name, color)
#define ENGINE_PROFILE_ZONE_TAG(t) \
    ZoneScoped;                    \
    ZoneText((t), std::strlen(t))
#define ENGINE_PROFILE_PLOT(n, v) TracyPlot(n, v)
#define ENGINE_PROFILE_MESSAGE(m) TracyMessage(m, std::strlen(m))
#define ENGINE_PROFILE_ALLOC(ptr, sz) TracyAlloc(ptr, sz)
#define ENGINE_PROFILE_FREE(ptr) TracyFree(ptr)
#define ENGINE_PROFILE_THREAD(name) tracy::SetThreadName(name)

#else

#define ENGINE_PROFILE_FRAME() ((void)0)
#define ENGINE_PROFILE_FRAME_NAMED(n) ((void)0)
#define ENGINE_PROFILE_ZONE(name) ((void)0)
#define ENGINE_PROFILE_ZONE_COLOR(name, color) ((void)0)
#define ENGINE_PROFILE_ZONE_TAG(t) ((void)0)
#define ENGINE_PROFILE_PLOT(n, v) ((void)0)
#define ENGINE_PROFILE_MESSAGE(m) ((void)0)
#define ENGINE_PROFILE_ALLOC(ptr, sz) ((void)0)
#define ENGINE_PROFILE_FREE(ptr) ((void)0)
#define ENGINE_PROFILE_THREAD(name) ((void)0)

#endif
