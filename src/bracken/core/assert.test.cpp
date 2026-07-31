#include <gtest/gtest.h>

#include "bracken/core/assert.hpp"
#include "bracken/core/logger.hpp"

#include <thread>

using namespace bracken::core;

// Suite name ends in "DeathTest" so gtest schedules it before threaded tests.
TEST(AssertDeathTest, FailureMessageReachesStderrWhenLoggerSilenced) {
    // abort_with must bypass the logger's min-level filter and, with all sinks
    // cleared, still print the diagnostic to stderr before aborting.
    EXPECT_DEATH(
        {
            Logger::instance().set_min_level(LogLevel::Off);
            Logger::instance().clear_sinks();
            BRACKEN_VERIFY(false, "diagnostic {}", 42);
        },
        "assertion `false` failed: diagnostic 42");
}

TEST(Assert, ConditionEvaluationMatchesBuildMode) {
    int calls = 0;
    auto observed = [&calls]() {
        ++calls;
        return true;
    };
    BRACKEN_ASSERT(observed(), "never fires");
#ifdef NDEBUG
    // Release: the condition is type-checked via sizeof but never evaluated.
    EXPECT_EQ(calls, 0);
#else
    EXPECT_EQ(calls, 1);
#endif
}

TEST(MainThreadAssert, BoundThreadIsMainThread) {
    bind_main_thread();
    EXPECT_TRUE(is_main_thread());
}

TEST(MainThreadAssert, OtherThreadIsNotMainThread) {
    bind_main_thread();
    bool other_is_main = true;
    std::thread t([&] { other_is_main = is_main_thread(); });
    t.join();
    EXPECT_FALSE(other_is_main);
    // The failed query from the worker must not have rebound the id.
    EXPECT_TRUE(is_main_thread());
}

TEST(MainThreadAssert, RebindMovesMainThread) {
    bind_main_thread();
    bool worker_was_main = false;
    std::thread t([&] {
        bind_main_thread();
        worker_was_main = is_main_thread();
    });
    t.join();
    EXPECT_TRUE(worker_was_main);
    EXPECT_FALSE(is_main_thread());
    // Restore for any later test relying on the gtest main thread.
    bind_main_thread();
    EXPECT_TRUE(is_main_thread());
}

TEST(MainThreadAssert, MacroPassesOnMainThread) {
    bind_main_thread();
    // Must not abort.
    BRACKEN_ASSERT_MAIN_THREAD();
    SUCCEED();
}
