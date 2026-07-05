#pragma once

#include "bracken/core/logger.hpp"

#include <cstdlib>
#include <format>
#include <string>
#include <string_view>
#include <utility>


namespace bracken::core::detail {

[[noreturn]] inline void abort_with(LogLevel level, const char* file, int line, const char* expr,
                                    std::string_view msg) {
    std::string full = std::format("{}:{}: assertion `{}` failed: {}", file, line, expr ? expr : "<unreachable>", msg);
    log_plain(level, "assert", full);
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

#ifdef NDEBUG
#define BRACKEN_ASSERT(cond, ...) ((void)0)
#else
#define BRACKEN_ASSERT(cond, ...) BRACKEN_VERIFY(cond __VA_OPT__(, ) __VA_ARGS__)
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
