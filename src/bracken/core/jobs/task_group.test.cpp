#include <gtest/gtest.h>

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/task_group.hpp"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace bracken::core::jobs;

TEST(TaskGroupTest, WaitOnFreshGroupReturnsImmediately) {
    TaskGroup g;
    EXPECT_EQ(g.pending(), 0u);
    g.wait(); // nothing submitted; must not block or throw
}

TEST(TaskGroupTest, CountsToZeroAndRunsAllJobs) {
    JobSystem js;
    js.init(2);
    TaskGroup g;
    constexpr int N = 64;
    std::atomic<int> ran{0};
    for (int i = 0; i < N; ++i) {
        js.submit(g, [&]() { ran.fetch_add(1, std::memory_order_relaxed); });
    }
    js.wait(g);
    EXPECT_EQ(ran.load(), N);
    EXPECT_EQ(g.pending(), 0u);
    EXPECT_EQ(js.pending_count(), 0u); // group jobs also drain the system count
    js.shutdown();
}

TEST(TaskGroupTest, PendingReflectsInFlightJobs) {
    JobSystem js;
    js.init(1);
    TaskGroup g;
    std::atomic<bool> started{false};
    std::atomic<bool> gate{false};
    js.submit(g, [&]() {
        started.store(true, std::memory_order_release);
        started.notify_one();
        gate.wait(false, std::memory_order_acquire);
    });
    started.wait(false, std::memory_order_acquire);
    EXPECT_EQ(g.pending(), 1u);
    gate.store(true, std::memory_order_release);
    gate.notify_all();
    js.wait(g);
    EXPECT_EQ(g.pending(), 0u);
    js.shutdown();
}

// The join is unconditional: wait() rethrows the member exception only AFTER
// every member job has finished — no member may still be running (or queued)
// once wait() has thrown.
TEST(TaskGroupTest, ExceptionRethrownOnlyAfterAllMembersFinish) {
    JobSystem js;
    js.init(2);
    TaskGroup g;
    constexpr int N = 32;
    std::atomic<int> completed{0};
    for (int i = 0; i < N; ++i) {
        const bool thrower = (i == 0);
        js.submit(g, [&, thrower]() {
            if (thrower) {
                throw std::runtime_error("member boom");
            }
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    }
    EXPECT_THROW(js.wait(g), std::runtime_error);
    // All members joined before the rethrow: the non-throwers all completed.
    EXPECT_EQ(completed.load(), N - 1);
    EXPECT_EQ(g.pending(), 0u);
    EXPECT_EQ(js.pending_count(), 0u);
    js.shutdown();
}

// Multiple throwing members: exactly one exception (the first captured) is
// rethrown; the CAS-guarded slot drops the rest. The group is clean and
// reusable afterwards.
TEST(TaskGroupTest, FirstExceptionWinsAndGroupIsReusableAfter) {
    JobSystem js;
    js.init(2);
    TaskGroup g;
    for (int i = 0; i < 8; ++i) {
        js.submit(g, [i]() { throw std::runtime_error("boom " + std::to_string(i)); });
    }
    int caught = 0;
    try {
        js.wait(g);
    } catch (const std::runtime_error& e) {
        ++caught;
        EXPECT_EQ(std::string(e.what()).rfind("boom ", 0), 0u);
    }
    EXPECT_EQ(caught, 1);

    // Reuse: the consumed error must not leak into the next batch.
    std::atomic<int> ran{0};
    for (int i = 0; i < 8; ++i) {
        js.submit(g, [&]() { ran.fetch_add(1, std::memory_order_relaxed); });
    }
    js.wait(g); // must not throw
    EXPECT_EQ(ran.load(), 8);
    js.shutdown();
}

// Reusable across frames: the same group object cycles submit→wait many
// times without residue (counter, error slot).
TEST(TaskGroupTest, ReusableAcrossManyIterations) {
    JobSystem js;
    js.init(3);
    TaskGroup g;
    std::atomic<long long> total{0};
    constexpr int FRAMES = 200;
    constexpr int JOBS = 8;
    for (int f = 0; f < FRAMES; ++f) {
        for (int j = 0; j < JOBS; ++j) {
            js.submit(g, [&]() { total.fetch_add(1, std::memory_order_relaxed); });
        }
        js.wait(g);
        ASSERT_EQ(g.pending(), 0u) << "residue after frame " << f;
    }
    EXPECT_EQ(total.load(), static_cast<long long>(FRAMES) * JOBS);
    js.shutdown();
}

// Exception + reuse under iteration: alternating failing/clean batches must
// rethrow exactly once per failing batch and never into a clean one.
TEST(TaskGroupStressTest, ExceptionAndReuseCycles) {
    JobSystem js;
    js.init(2);
    TaskGroup g;
    std::atomic<int> ran{0};
    int throws_seen = 0;
    constexpr int CYCLES = 100;
    for (int c = 0; c < CYCLES; ++c) {
        const bool failing = (c % 2 == 0);
        for (int j = 0; j < 8; ++j) {
            const bool thrower = failing && j == 3;
            js.submit(g, [&, thrower]() {
                if (thrower) {
                    throw std::runtime_error("cycle boom");
                }
                ran.fetch_add(1, std::memory_order_relaxed);
            });
        }
        try {
            js.wait(g);
        } catch (const std::runtime_error&) {
            ++throws_seen;
        }
    }
    EXPECT_EQ(throws_seen, CYCLES / 2);
    EXPECT_EQ(ran.load(), CYCLES * 8 - CYCLES / 2);
    js.shutdown();
}

// Concurrent waiters: all parked waiters wake when the group drains; exactly
// one of them rethrows the captured error (single-consumer rethrow — see the
// TaskGroup class comment), the others return normally.
TEST(TaskGroupTest, ConcurrentWaitersAllWakeExactlyOneRethrows) {
    JobSystem js;
    js.init(2);
    TaskGroup g;

    std::atomic<bool> gate{false};
    constexpr int JOBS = 4;
    for (int i = 0; i < JOBS; ++i) {
        const bool thrower = (i == 0);
        js.submit(g, [&, thrower]() {
            gate.wait(false, std::memory_order_acquire);
            if (thrower) {
                throw std::runtime_error("gated boom");
            }
        });
    }

    constexpr int WAITERS = 4;
    std::atomic<int> woke{0};
    std::atomic<int> caught{0};
    std::vector<std::thread> waiters;
    waiters.reserve(WAITERS);
    for (int w = 0; w < WAITERS; ++w) {
        waiters.emplace_back([&]() {
            try {
                g.wait(); // park-only external wait
            } catch (const std::runtime_error&) {
                caught.fetch_add(1, std::memory_order_relaxed);
            }
            woke.fetch_add(1, std::memory_order_relaxed);
        });
    }

    // Let the waiters reach the park before releasing the jobs (best-effort;
    // correctness does not depend on the timing).
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    gate.store(true, std::memory_order_release);
    gate.notify_all();

    for (auto& t : waiters) t.join();
    EXPECT_EQ(woke.load(), WAITERS);
    EXPECT_EQ(caught.load(), 1);
    EXPECT_EQ(g.pending(), 0u);
    js.shutdown();
}

// Group jobs queued at shutdown are discarded with an error and still drain
// the group counter — a group join must never hang across shutdown.
TEST(TaskGroupTest, ShutdownDiscardsQueuedGroupJobsAndDrainsCounter) {
    JobSystem js;
    js.init(1);
    TaskGroup g;

    std::atomic<bool> started{false};
    std::atomic<bool> gate{false};
    js.submit(g, [&]() {
        started.store(true, std::memory_order_release);
        started.notify_one();
        gate.wait(false, std::memory_order_acquire);
    });
    started.wait(false, std::memory_order_acquire); // worker gated

    std::atomic<bool> victim_ran{false};
    js.submit(g, [&]() { victim_ran.store(true, std::memory_order_release); });

    std::thread killer([&]() { js.shutdown(); });
    // The discard pass drains the queued group job while the worker is gated.
    while (g.pending() > 1) {
        std::this_thread::yield();
    }
    gate.store(true, std::memory_order_release);
    gate.notify_all();
    killer.join();

    EXPECT_FALSE(victim_ran.load(std::memory_order_acquire));
    EXPECT_EQ(g.pending(), 0u);
    // The discard error was captured into the group; consume it.
    EXPECT_THROW(g.wait(), std::runtime_error);
}
