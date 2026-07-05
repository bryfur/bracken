#pragma once

// Profiling
// ---------
// Enable Tracy at CMake configure: -DBRACKEN_ENABLE_TRACY=ON (default OFF, zero cost).
// Run the game, launch tracy-profiler GUI, click Connect (loopback by default).
//
// Macros:
//   BRACKEN_PROFILE_FRAME()          - mark a frame boundary (call once per frame)
//   BRACKEN_PROFILE_FRAME_NAMED(n)   - named frame boundary (string literal)
//   BRACKEN_PROFILE_ZONE(name)       - scoped zone with literal name
//   BRACKEN_PROFILE_ZONE_COLOR(n, c) - scoped zone with literal name + 0xRRGGBB color
//   BRACKEN_PROFILE_ZONE_TAG(t)      - scoped zone, dynamic tag string attached
//   BRACKEN_PROFILE_PLOT(n, v)       - emit a numeric sample on plot n
//   BRACKEN_PROFILE_MESSAGE(m)       - one-shot string message on the timeline
//   BRACKEN_PROFILE_ALLOC(ptr, sz)   - track an allocation (memory view)
//   BRACKEN_PROFILE_FREE(ptr)        - track a free (pair with BRACKEN_PROFILE_ALLOC)
//   BRACKEN_PROFILE_THREAD(name)     - name the current thread in captures
//
// GPU note: SDL3 GPU does not expose backend queues/devices, so Tracy's GPU
// zones (TracyVkZone etc.) cannot be wired up. GPU timing = RenderDoc/Nsight.
//
// Suggested zone colors (keep passes visually consistent in captures):
//   0x3D6E9E update  |  0x8E44AD animation  |  0x27AE60 culling/scene
//   0xE67E22 shadows |  0xC0392B main pass  |  0x16A085 post-processing

#if defined(BRACKEN_ENABLE_TRACY) && BRACKEN_ENABLE_TRACY

#include <cstring>
#include <tracy/Tracy.hpp>

#define BRACKEN_PROFILE_FRAME() FrameMark
#define BRACKEN_PROFILE_FRAME_NAMED(n) FrameMarkNamed(n)
#define BRACKEN_PROFILE_ZONE(name) ZoneScopedN(name)
#define BRACKEN_PROFILE_ZONE_COLOR(name, color) ZoneScopedNC(name, color)
#define BRACKEN_PROFILE_ZONE_TAG(t) \
    ZoneScoped;                    \
    ZoneText((t), std::strlen(t))
#define BRACKEN_PROFILE_PLOT(n, v) TracyPlot(n, v)
#define BRACKEN_PROFILE_MESSAGE(m) TracyMessage(m, std::strlen(m))
#define BRACKEN_PROFILE_ALLOC(ptr, sz) TracyAlloc(ptr, sz)
#define BRACKEN_PROFILE_FREE(ptr) TracyFree(ptr)
#define BRACKEN_PROFILE_THREAD(name) tracy::SetThreadName(name)

#else

#define BRACKEN_PROFILE_FRAME() ((void)0)
#define BRACKEN_PROFILE_FRAME_NAMED(n) ((void)0)
#define BRACKEN_PROFILE_ZONE(name) ((void)0)
#define BRACKEN_PROFILE_ZONE_COLOR(name, color) ((void)0)
#define BRACKEN_PROFILE_ZONE_TAG(t) ((void)0)
#define BRACKEN_PROFILE_PLOT(n, v) ((void)0)
#define BRACKEN_PROFILE_MESSAGE(m) ((void)0)
#define BRACKEN_PROFILE_ALLOC(ptr, sz) ((void)0)
#define BRACKEN_PROFILE_FREE(ptr) ((void)0)
#define BRACKEN_PROFILE_THREAD(name) ((void)0)

#endif
