#include "bracken/core/log_throttle.hpp"
#include "bracken/core/logger.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace bracken::core {
namespace {

// The gates only pay off if they gate the WHOLE call, formatting included, so
// the tests below assert on what a sink actually receives.
class CountingSink final : public LogSink {
public:
    void write(LogLevel, const char*, std::string_view msg) override {
        ++count;
        last = std::string(msg);
    }
    std::atomic<int> count{0};
    std::string last;
};

struct SinkScope {
    SinkScope() {
        sink = std::make_shared<CountingSink>();
        handle = Logger::instance().add_sink(sink);
        prev_level = Logger::instance().min_level();
        Logger::instance().set_min_level(LogLevel::Trace);
    }
    ~SinkScope() {
        Logger::instance().remove_sink(handle);
        Logger::instance().set_min_level(prev_level);
    }
    SinkScope(const SinkScope&) = delete;
    SinkScope& operator=(const SinkScope&) = delete;

    std::shared_ptr<CountingSink> sink;
    Logger::SinkHandle handle{};
    LogLevel prev_level{};
};

TEST(LogThrottle, TakeOnceFiresExactlyOnce) {
    LogThrottle t;
    EXPECT_TRUE(t.take_once());
    for (int i = 0; i < 99; ++i) {
        EXPECT_FALSE(t.take_once());
    }
    EXPECT_TRUE(t.fired());
    EXPECT_EQ(t.suppressed(), 99u);
}

TEST(LogThrottle, TakeIntervalFiresOnceInsideThePeriod) {
    LogThrottle t;
    EXPECT_TRUE(t.take_interval(3600.0)); // first call always fires
    for (int i = 0; i < 50; ++i) {
        EXPECT_FALSE(t.take_interval(3600.0));
    }
    EXPECT_EQ(t.suppressed(), 50u);
}

TEST(LogThrottle, TakeIntervalFiresAgainOnceThePeriodElapses) {
    LogThrottle t;
    EXPECT_TRUE(t.take_interval(0.0)); // a zero period never suppresses
    EXPECT_TRUE(t.take_interval(0.0));
    EXPECT_EQ(t.suppressed(), 0u);
}

TEST(LogThrottle, ConcurrentTakeOnceElectsExactlyOneWinner) {
    LogThrottle t;
    std::atomic<int> winners{0};
    std::vector<std::thread> threads;
    threads.reserve(8);
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&] {
            for (int k = 0; k < 200; ++k) {
                if (t.take_once()) {
                    winners.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_EQ(winners.load(), 1);
    EXPECT_EQ(t.suppressed(), 8u * 200u - 1u);
}

// The macro half: a per-frame warning must reach the sink once, not 500 times.
TEST(LogThrottle, WarnOnceMacroEmitsOneLine) {
    SinkScope scope;
    for (int frame = 0; frame < 500; ++frame) {
        BRACKEN_LOG_WARN_ONCE("test", "atlas overflow on frame {}", frame);
    }
    EXPECT_EQ(scope.sink->count.load(), 1);
    EXPECT_EQ(scope.sink->last, "atlas overflow on frame 0");
}

TEST(LogThrottle, ThrottledMacroEmitsOneLinePerPeriod) {
    SinkScope scope;
    for (int frame = 0; frame < 500; ++frame) {
        BRACKEN_LOG_WARN_THROTTLED(3600.0, "test", "budget exhausted ({})", frame);
    }
    EXPECT_EQ(scope.sink->count.load(), 1);
}

// Two call sites must not share a gate, even on the same line-adjacent code.
TEST(LogThrottle, DistinctCallSitesHaveDistinctGates) {
    SinkScope scope;
    for (int i = 0; i < 10; ++i) {
        BRACKEN_LOG_WARN_ONCE("test", "site A");
        BRACKEN_LOG_WARN_ONCE("test", "site B");
    }
    EXPECT_EQ(scope.sink->count.load(), 2);
}

} // namespace
} // namespace bracken::core
