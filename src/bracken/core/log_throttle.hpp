#pragma once

#include "bracken/core/logger.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>

namespace bracken::core {

/**
 * Per-call-site emission gate for diagnostics that sit on a PER-FRAME path.
 *
 * A `BRACKEN_LOG_WARN` inside `render()` or an emit method is not a warning,
 * it is a 60 Hz denial-of-service on the log: the first line (the one that
 * says what broke) scrolls away before anyone reads it, and the cost of
 * formatting it is paid every frame forever. The two gates here are the house
 * answer:
 *
 *   BRACKEN_LOG_WARN_ONCE(tag, fmt, ...)               — first occurrence only
 *   BRACKEN_LOG_WARN_THROTTLED(secs, tag, fmt, ...)    — at most one per `secs`
 *
 * Each expands to a function-local `static LogThrottle`. The type is
 * constexpr-constructible, so the static is constant-initialized: no
 * thread-safe-static guard, no lazy-init race, nothing to construct at
 * startup. Every atomic is `relaxed` — the state published is a single
 * self-contained bool/timestamp with no other data riding on it, and the only
 * guarantee a rate limiter owes is that *approximately* one message per
 * period escapes. Losing a race means one extra line or one line attributed
 * to the wrong thread, never a missing first line and never a torn read.
 *
 * Suppressed occurrences are counted (`suppressed()`), so a test — or a
 * future "and N more" epilogue — can prove the gate is doing its job rather
 * than the condition having stopped.
 *
 * Choosing between them: use ONCE for a condition that is a standing fact
 * about the frame ("no camera was ever set", "UI pipeline not available") —
 * repeating it adds nothing. Use THROTTLED for a condition that can come and
 * go and whose *recurrence* is the information (budget exhaustion, atlas
 * eviction), where silence after the first line would read as "fixed".
 */
class LogThrottle {
public:
    constexpr LogThrottle() = default;

    LogThrottle(const LogThrottle&) = delete;
    LogThrottle& operator=(const LogThrottle&) = delete;
    LogThrottle(LogThrottle&&) = delete;
    LogThrottle& operator=(LogThrottle&&) = delete;

    /** True exactly once over the process lifetime of this call site. */
    [[nodiscard]] bool take_once() noexcept {
        if (fired_.exchange(true, std::memory_order_relaxed)) {
            suppressed_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        return true;
    }

    /** True when `interval_s` has elapsed since the last true (and the first
     *  time it is ever called). Exactly one caller wins each edge. */
    [[nodiscard]] bool take_interval(double interval_s) noexcept {
        using clock = std::chrono::steady_clock;
        const int64_t now_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
        const int64_t period_ns = static_cast<int64_t>(interval_s * 1e9);
        // last_ns_ == 0 means "never fired"; a real steady_clock reading of
        // exactly 0 would at worst let one extra line through, once.
        int64_t last = last_ns_.load(std::memory_order_relaxed);
        if (last != 0 && now_ns - last < period_ns) {
            suppressed_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (!last_ns_.compare_exchange_strong(last, now_ns, std::memory_order_relaxed)) {
            suppressed_.fetch_add(1, std::memory_order_relaxed); // another thread took this edge
            return false;
        }
        fired_.store(true, std::memory_order_relaxed);
        return true;
    }

    /** How many occurrences this gate swallowed. */
    [[nodiscard]] uint64_t suppressed() const noexcept { return suppressed_.load(std::memory_order_relaxed); }

    /** True once anything has been emitted through this gate. */
    [[nodiscard]] bool fired() const noexcept { return fired_.load(std::memory_order_relaxed); }

    /** Re-arm (tests only — a live call site's gate is never reset). */
    void reset() noexcept {
        fired_.store(false, std::memory_order_relaxed);
        last_ns_.store(0, std::memory_order_relaxed);
        suppressed_.store(0, std::memory_order_relaxed);
    }

private:
    std::atomic<bool> fired_{false};
    std::atomic<int64_t> last_ns_{0};
    std::atomic<uint64_t> suppressed_{0};
};

} // namespace bracken::core

#define BRACKEN_LOG_THROTTLE_CAT2(a, b) a##b
#define BRACKEN_LOG_THROTTLE_CAT(a, b) BRACKEN_LOG_THROTTLE_CAT2(a, b)

/** Emit `level` for this call site once, ever. See LogThrottle. */
#define BRACKEN_LOG_ONCE_AT(level_macro, category, ...)                                        \
    do {                                                                                       \
        static ::bracken::core::LogThrottle BRACKEN_LOG_THROTTLE_CAT(_bracken_throttle_, __LINE__); \
        if (BRACKEN_LOG_THROTTLE_CAT(_bracken_throttle_, __LINE__).take_once()) [[unlikely]] {  \
            level_macro(category, __VA_ARGS__);                                                 \
        }                                                                                      \
    } while (0)

/** Emit `level` for this call site at most once per `interval_s`. */
#define BRACKEN_LOG_THROTTLED_AT(level_macro, interval_s, category, ...)                            \
    do {                                                                                            \
        static ::bracken::core::LogThrottle BRACKEN_LOG_THROTTLE_CAT(_bracken_throttle_, __LINE__);  \
        if (BRACKEN_LOG_THROTTLE_CAT(_bracken_throttle_, __LINE__).take_interval(interval_s))        \
            [[unlikely]] {                                                                          \
            level_macro(category, __VA_ARGS__);                                                      \
        }                                                                                           \
    } while (0)

#define BRACKEN_LOG_WARN_ONCE(category, ...) BRACKEN_LOG_ONCE_AT(BRACKEN_LOG_WARN, category, __VA_ARGS__)
#define BRACKEN_LOG_ERROR_ONCE(category, ...) BRACKEN_LOG_ONCE_AT(BRACKEN_LOG_ERROR, category, __VA_ARGS__)

#define BRACKEN_LOG_WARN_THROTTLED(interval_s, category, ...) \
    BRACKEN_LOG_THROTTLED_AT(BRACKEN_LOG_WARN, interval_s, category, __VA_ARGS__)
#define BRACKEN_LOG_ERROR_THROTTLED(interval_s, category, ...) \
    BRACKEN_LOG_THROTTLED_AT(BRACKEN_LOG_ERROR, interval_s, category, __VA_ARGS__)
