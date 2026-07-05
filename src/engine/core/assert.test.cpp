#include <gtest/gtest.h>

#include "engine/core/assert.hpp"

#include <thread>

using namespace mmo::engine::core;

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
    ENGINE_ASSERT_MAIN_THREAD();
    SUCCEED();
}
