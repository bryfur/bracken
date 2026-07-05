#pragma once

#include "engine/core/logger.hpp"

#include <cstdlib>
#include <format>
#include <string>
#include <string_view>
#include <utility>


namespace mmo::engine::core::detail {

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

} // namespace mmo::engine::core::detail

namespace mmo::engine::core {

/**
 * Record the calling thread as the engine main thread for
 * ENGINE_ASSERT_MAIN_THREAD(). Called explicitly from the true main thread at
 * engine boot (Application::init_engine, FileWatcher::init); if never called,
 * the first is_main_thread() query captures its calling thread instead.
 */
void bind_main_thread() noexcept;

/** True when the calling thread is the recorded main thread (captures the
 *  calling thread on first use if none was bound). */
bool is_main_thread() noexcept;

} // namespace mmo::engine::core


#define ENGINE_VERIFY(cond, ...)                                                                                       \
    do {                                                                                                               \
        if (!(cond)) [[unlikely]] {                                                                                    \
            __VA_OPT__(::mmo::engine::core::detail::abort_format(::mmo::engine::core::LogLevel::Fatal, __FILE__,       \
                                                                 __LINE__, #cond, __VA_ARGS__);)                       \
            ::mmo::engine::core::detail::abort_plain(::mmo::engine::core::LogLevel::Fatal, __FILE__, __LINE__, #cond); \
        }                                                                                                              \
    } while (0)

#ifdef NDEBUG
#define ENGINE_ASSERT(cond, ...) ((void)0)
#else
#define ENGINE_ASSERT(cond, ...) ENGINE_VERIFY(cond __VA_OPT__(, ) __VA_ARGS__)
#endif

// Enforces the documented main-thread-only contracts (FileWatcher entry
// points, swapchain acquire, render_frame). Zero cost in release builds.
#ifdef NDEBUG
#define ENGINE_ASSERT_MAIN_THREAD() ((void)0)
#else
#define ENGINE_ASSERT_MAIN_THREAD() \
    ENGINE_VERIFY(::mmo::engine::core::is_main_thread(), "must be called from the main thread")
#endif

#define ENGINE_UNREACHABLE(...)                                                                                        \
    do {                                                                                                               \
        __VA_OPT__(::mmo::engine::core::detail::abort_format(::mmo::engine::core::LogLevel::Fatal, __FILE__, __LINE__, \
                                                             "unreachable", __VA_ARGS__);)                             \
        ::mmo::engine::core::detail::abort_plain(::mmo::engine::core::LogLevel::Fatal, __FILE__, __LINE__,             \
                                                 "unreachable");                                                       \
    } while (0)
