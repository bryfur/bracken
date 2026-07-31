#pragma once

#include "bracken/core/logger.hpp"

#include <cstdio>
#include <cstdlib>
#include <format>
#include <string>
#include <string_view>
#include <utility>


namespace bracken::core::detail {

[[noreturn]] inline void abort_with(LogLevel level, const char* file, int line, const char* expr,
                                    std::string_view msg) {
    std::string full = std::format("{}:{}: assertion `{}` failed: {}", file, line, expr ? expr : "<unreachable>", msg);
    // Bypass the min-level filter: an aborting contract violation must never be
    // silenced by set_min_level(Off). Sinks may still be cleared, so also print
    // to stderr as a last-resort channel before aborting.
    Logger::instance().dispatch(level, "assert", full);
    std::fprintf(stderr, "%s\n", full.c_str());
    std::abort();
}

template<typename... Args>
[[noreturn]] inline void abort_format(LogLevel level, const char* file, int line, const char* expr,
                                      std::format_string<Args...> fmt, Args&&... args) {
    std::string msg = std::format(fmt, std::forward<Args>(args)...);
    abort_with(level, file, line, expr, msg);
}

[[noreturn]] inline void abort_plain(LogLevel level, const char* file, int line, const char* expr) {
    abort_with(level, file, line, expr, "");
}

} // namespace bracken::core::detail

namespace bracken::core {

/**
 * Record the calling thread as the engine main thread for
 * BRACKEN_ASSERT_MAIN_THREAD(). Called explicitly from the true main thread at
 * engine boot (Application::init_engine, FileWatcher::init); if never called,
 * the first is_main_thread() query captures its calling thread instead.
 */
void bind_main_thread() noexcept;

/** True when the calling thread is the recorded main thread (captures the
 *  calling thread on first use if none was bound). */
bool is_main_thread() noexcept;

} // namespace bracken::core


#define BRACKEN_VERIFY(cond, ...)                                                                                       \
    do {                                                                                                               \
        if (!(cond)) [[unlikely]] {                                                                                    \
            __VA_OPT__(::bracken::core::detail::abort_format(::bracken::core::LogLevel::Fatal, __FILE__,       \
                                                                 __LINE__, #cond, __VA_ARGS__);)                       \
            ::bracken::core::detail::abort_plain(::bracken::core::LogLevel::Fatal, __FILE__, __LINE__, #cond); \
        }                                                                                                              \
    } while (0)

// Release builds keep the condition type-checked (never evaluated) via
// sizeof, so asserted expressions can't bit-rot against renamed symbols.
// Corollary: conditions must be side-effect-free.
#ifdef NDEBUG
#define BRACKEN_ASSERT(cond, ...) ((void)sizeof(!(cond)))
#else
#define BRACKEN_ASSERT(cond, ...) BRACKEN_VERIFY(cond __VA_OPT__(, ) __VA_ARGS__)
#endif

// BRACKEN_CONTRACT — a precondition on the HOST-FACING API boundary, live in
// release builds too.
//
// The distinction from BRACKEN_ASSERT is about who makes the mistake and what
// happens next. BRACKEN_ASSERT guards internal invariants: if one trips, the
// engine has a bug, the developer sees it in a debug build, and compiling it
// out of release costs nothing because a shipped engine is not supposed to
// violate its own invariants. BRACKEN_CONTRACT guards what a GAME does to the
// engine — reading a packet before seal, sealing twice, emitting into a lane
// two threads share, calling models() before init_renderer(). Those are
// mistakes a host makes in ITS release build, and compiling the check out
// converts a named abort into a segfault or, worse, silent memory corruption
// that surfaces three subsystems away.
//
// Cost: one compare and one predictable branch. Every use below is on a
// per-frame or per-call path, never inside a per-item loop — a contract in a
// 20,000-iteration instance loop is a BRACKEN_ASSERT, not a contract.
//
// BRACKEN_CONTRACTS_ENABLED=0 (cmake -DBRACKEN_ENABLE_CONTRACTS=OFF) compiles
// them out; the condition stays type-checked, exactly like BRACKEN_ASSERT.
// Defaults to on when the macro is undefined so headers used outside the
// engine's own build still behave.
#ifndef BRACKEN_CONTRACTS_ENABLED
#define BRACKEN_CONTRACTS_ENABLED 1
#endif

#if BRACKEN_CONTRACTS_ENABLED
#define BRACKEN_CONTRACT(cond, ...) BRACKEN_VERIFY(cond __VA_OPT__(, ) __VA_ARGS__)
#else
#define BRACKEN_CONTRACT(cond, ...) ((void)sizeof(!(cond)))
#endif

// Enforces the documented main-thread-only contracts (FileWatcher entry
// points, swapchain acquire, render_frame). Zero cost in release builds.
#ifdef NDEBUG
#define BRACKEN_ASSERT_MAIN_THREAD() ((void)0)
#else
#define BRACKEN_ASSERT_MAIN_THREAD() \
    BRACKEN_VERIFY(::bracken::core::is_main_thread(), "must be called from the main thread")
#endif

#define BRACKEN_UNREACHABLE(...)                                                                                        \
    do {                                                                                                               \
        __VA_OPT__(::bracken::core::detail::abort_format(::bracken::core::LogLevel::Fatal, __FILE__, __LINE__, \
                                                             "unreachable", __VA_ARGS__);)                             \
        ::bracken::core::detail::abort_plain(::bracken::core::LogLevel::Fatal, __FILE__, __LINE__,             \
                                                 "unreachable");                                                       \
    } while (0)
