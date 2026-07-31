#pragma once

#include <atomic>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bracken::core {

enum class LogLevel : uint8_t {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Fatal = 5,
    Off = 6,
};

const char* to_string(LogLevel level) noexcept;

class LogSink {
public:
    LogSink() = default;
    virtual ~LogSink() = default;

    // Polymorphic base: copying/moving through the base slices derived sinks.
    LogSink(const LogSink&) = delete;
    LogSink& operator=(const LogSink&) = delete;
    LogSink(LogSink&&) = delete;
    LogSink& operator=(LogSink&&) = delete;

    virtual void write(LogLevel level, const char* category, std::string_view msg) = 0;
};

class StdoutSink final : public LogSink {
public:
    void write(LogLevel level, const char* category, std::string_view msg) override;
};

class Logger {
public:
    using SinkHandle = uint32_t;

    static Logger& instance();

    void set_min_level(LogLevel level) noexcept;
    LogLevel min_level() const noexcept;

    SinkHandle add_sink(std::shared_ptr<LogSink> sink);
    void remove_sink(SinkHandle handle);
    void clear_sinks();

    bool should_log(LogLevel level) const noexcept;

    void dispatch(LogLevel level, const char* category, std::string_view msg);

private:
    Logger();

    struct Entry {
        SinkHandle id;
        std::shared_ptr<LogSink> sink;
    };

    // Guards sinks_ and next_id_ only. min_level_ is deliberately outside it:
    // should_log() is on every log macro's fast path and is called from every
    // worker thread, so taking a global mutex there is both a serialization
    // point and (inside a noexcept function) a latent std::terminate.
    mutable std::mutex mutex_;
    std::vector<Entry> sinks_;
    SinkHandle next_id_ = 1;

    // Protocol: publishes the host's chosen verbosity to every logging thread.
    // Written only by set_min_level(), read by should_log()/min_level().
    // Both ends are memory_order_relaxed: the value is self-contained (a single
    // enum, no other state is published with it), and the only guarantee a
    // caller needs is that a level change eventually becomes visible — losing a
    // race means one extra or one missing line around the switch, which is
    // exactly the semantics a level knob has anyway. No release/acquire pairing
    // partner exists by design; do not add one without also publishing state.
    std::atomic<LogLevel> min_level_{LogLevel::Info};
};

namespace detail {

template<typename... Args>
inline void log_formatted(LogLevel level, const char* category, std::format_string<Args...> fmt, Args&&... args) {
    auto& logger = Logger::instance();
    if (!logger.should_log(level)) {
        return;
    }
    std::string msg = std::format(fmt, std::forward<Args>(args)...);
    logger.dispatch(level, category, msg);
}

inline void log_plain(LogLevel level, const char* category, std::string_view msg) {
    auto& logger = Logger::instance();
    if (!logger.should_log(level)) {
        return;
    }
    logger.dispatch(level, category, msg);
}

} // namespace detail

} // namespace bracken::core

#define BRACKEN_LOG_TRACE(category, ...) \
    ::bracken::core::detail::log_formatted(::bracken::core::LogLevel::Trace, (category), __VA_ARGS__)

#define BRACKEN_LOG_DEBUG(category, ...) \
    ::bracken::core::detail::log_formatted(::bracken::core::LogLevel::Debug, (category), __VA_ARGS__)

#define BRACKEN_LOG_INFO(category, ...) \
    ::bracken::core::detail::log_formatted(::bracken::core::LogLevel::Info, (category), __VA_ARGS__)

#define BRACKEN_LOG_WARN(category, ...) \
    ::bracken::core::detail::log_formatted(::bracken::core::LogLevel::Warn, (category), __VA_ARGS__)

#define BRACKEN_LOG_ERROR(category, ...) \
    ::bracken::core::detail::log_formatted(::bracken::core::LogLevel::Error, (category), __VA_ARGS__)

#define BRACKEN_LOG_FATAL(category, ...) \
    ::bracken::core::detail::log_formatted(::bracken::core::LogLevel::Fatal, (category), __VA_ARGS__)
