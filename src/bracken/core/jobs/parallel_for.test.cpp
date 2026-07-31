#include <gtest/gtest.h>

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace bracken::core::jobs;

namespace {

class ParallelForTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!JobSystem::instance().is_initialized()) {
            JobSystem::instance().init();
        }
    }
};

} // namespace

TEST_F(ParallelForTest, EmptyRangeIsNoop) {
    int touched = 0;
    parallel_for(5, 5, [&](std::size_t, std::size_t) { ++touched; });
    EXPECT_EQ(touched, 0);
}

TEST_F(ParallelForTest, FillsTenThousandIndices) {
    constexpr std::size_t N = 10'000;
    std::vector<int> data(N, 0);
    parallel_for(0, N, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) data[i] = static_cast<int>(i);
    });
    long long sum = std::accumulate(data.begin(), data.end(), 0LL);
    long long expected = static_cast<long long>(N) * (N - 1) / 2;
    EXPECT_EQ(sum, expected);
}

TEST_F(ParallelForTest, SerialFallbackForLargeGrain) {
    constexpr std::size_t N = 1000;
    std::atomic<int> chunks{0};
    parallel_for(0, N, [&](std::size_t, std::size_t) { chunks.fetch_add(1); }, /*grain=*/N * 2);
    EXPECT_EQ(chunks.load(), 1);
}

TEST_F(ParallelForTest, AutomaticGrainSplitsRange) {
    auto& js = JobSystem::instance();
    if (js.worker_count() < 2) {
        GTEST_SKIP() << "Need at least 2 workers";
    }
    constexpr std::size_t N = 4096;
    std::atomic<int> chunks{0};
    parallel_for(0, N, [&](std::size_t, std::size_t) { chunks.fetch_add(1); });
    EXPECT_GT(chunks.load(), 1);
}

TEST_F(ParallelForTest, AccumulatesAtomicCounter) {
    constexpr std::size_t N = 50'000;
    std::atomic<long long> sum{0};
    parallel_for(0, N, [&](std::size_t b, std::size_t e) {
        long long local = 0;
        for (std::size_t i = b; i < e; ++i) local += static_cast<long long>(i);
        sum.fetch_add(local, std::memory_order_relaxed);
    });
    long long expected = static_cast<long long>(N) * (N - 1) / 2;
    EXPECT_EQ(sum.load(), expected);
}

TEST_F(ParallelForTest, StressOneThousandTrivialTasksAllRun) {
    constexpr std::size_t N = 1000;
    std::atomic<int> seen{0};
    parallel_for(
        0, N,
        [&](std::size_t b, std::size_t e) {
            for (std::size_t i = b; i < e; ++i) seen.fetch_add(1, std::memory_order_relaxed);
        },
        /*grain=*/1);
    EXPECT_EQ(seen.load(), static_cast<int>(N));
}

// Historically flaky; now deterministic by contract: every chunk decrements
// the system pending count BEFORE the group join is released, so a returned
// parallel_for implies a drained pending_count (see complete_node_).
TEST_F(ParallelForTest, PendingCountIsZeroOnceDrained) {
    constexpr std::size_t N = 256;
    parallel_for(0, N, [](std::size_t, std::size_t) {});
    EXPECT_EQ(JobSystem::instance().pending_count(), 0u);
}

// M32 regression: a throwing chunk must not unwind parallel_for while other
// chunks are still queued or running (fn captures caller stack state by
// reference — stragglers would be use-after-scope). All chunks are joined
// before the first exception is rethrown, so no chunk may execute after
// parallel_for returns.
TEST_F(ParallelForTest, ThrowingChunkJoinsAllChunksBeforeRethrow) {
    auto& js = JobSystem::instance();
    constexpr std::size_t N = 512;
    std::atomic<int> chunks_run{0};
    std::atomic<bool> caller_unwound{false};
    std::atomic<bool> ran_after_unwind{false};

    bool threw = false;
    try {
        parallel_for(
            0, N,
            [&](std::size_t b, std::size_t) {
                if (caller_unwound.load(std::memory_order_acquire)) {
                    ran_after_unwind.store(true, std::memory_order_release);
                }
                chunks_run.fetch_add(1, std::memory_order_relaxed);
                if (b < 8) {
                    throw std::runtime_error("chunk boom");
                }
            },
            /*grain=*/1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    caller_unwound.store(true, std::memory_order_release);
    const int chunks_at_return = chunks_run.load(std::memory_order_acquire);

    js.wait_all(); // flush any stragglers (there must be none)
    EXPECT_TRUE(threw);
    EXPECT_EQ(chunks_run.load(std::memory_order_acquire), chunks_at_return);
    EXPECT_FALSE(ran_after_unwind.load(std::memory_order_acquire));
    EXPECT_EQ(js.pending_count(), 0u);
}

TEST_F(ParallelForTest, BusySpinIsFasterThanSerial) {
    auto& js = JobSystem::instance();
    if (js.worker_count() < 2) {
        GTEST_SKIP() << "Need at least 2 workers";
    }
    constexpr std::size_t N = 64;

    auto busy_chunk = [](std::size_t b, std::size_t e) {
        volatile std::uint64_t acc = 0;
        for (std::size_t i = b; i < e; ++i) {
            for (int k = 0; k < 200'000; ++k) {
                acc += static_cast<std::uint64_t>(i) * k;
            }
        }
        (void)acc;
    };

    auto t0 = std::chrono::steady_clock::now();
    busy_chunk(0, N);
    auto t1 = std::chrono::steady_clock::now();
    auto serial_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    auto p0 = std::chrono::steady_clock::now();
    parallel_for(0, N, busy_chunk, /*grain=*/1);
    auto p1 = std::chrono::steady_clock::now();
    auto par_us = std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count();

    EXPECT_LT(par_us, serial_us);
}

// ---- parallel_for(JobSystem&, ...) — the explicit-pool overload ----

// Parity: the explicit-pool overload produces bit-identical results to the
// serial single-chunk path.
TEST(ParallelForJobSystemOverloadTest, MatchesSerialResult) {
    JobSystem js;
    js.init(4);

    constexpr std::size_t N = 10'000;
    auto body = [](std::vector<std::uint64_t>& out, std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) {
            out[i] = (i * 2654435761ull) ^ (i >> 3);
        }
    };

    std::vector<std::uint64_t> serial(N, 0);
    body(serial, 0, N);

    std::vector<std::uint64_t> parallel(N, 0);
    parallel_for(js, 0, N, [&](std::size_t b, std::size_t e) { body(parallel, b, e); }, /*grain=*/64);

    EXPECT_EQ(parallel, serial);
    EXPECT_EQ(js.pending_count(), 0u);
    js.shutdown();
}

// An uninitialized caller-owned pool runs the whole range inline as one
// chunk on the caller (same code path, granularity control).
TEST(ParallelForJobSystemOverloadTest, UninitializedPoolRunsInlineAsOneChunk) {
    JobSystem js; // never init()ed
    int chunks = 0;
    std::size_t covered = 0;
    parallel_for(js, 0, 1000, [&](std::size_t b, std::size_t e) {
        ++chunks;
        covered += e - b;
    }, /*grain=*/1);
    EXPECT_EQ(chunks, 1);
    EXPECT_EQ(covered, 1000u);
}

// Fork-join from inside a job on a 1-worker pool: the chunks land in the
// worker's own deque, and the group join must help (drain own deque) rather
// than park — a parking join would livelock deterministically.
TEST(ParallelForJobSystemOverloadTest, NestedFromJobOnSingleWorkerPoolDoesNotLivelock) {
    JobSystem js;
    js.init(1);
    ASSERT_EQ(js.worker_count(), 1u);

    std::atomic<int> total{0};
    auto h = js.submit([&]() {
        parallel_for(js, 0, 64, [&](std::size_t b, std::size_t e) {
            total.fetch_add(static_cast<int>(e - b), std::memory_order_relaxed);
        }, /*grain=*/1);
    });
    js.wait(h);
    EXPECT_EQ(total.load(), 64);
    js.shutdown();
}

// Explicit-pool exception contract mirrors the singleton one: join ALL
// chunks, then rethrow the first captured error.
TEST(ParallelForJobSystemOverloadTest, ThrowingChunkStillJoinsEverything) {
    JobSystem js;
    js.init(2);
    constexpr std::size_t N = 128;
    std::atomic<int> chunks_run{0};
    bool threw = false;
    try {
        parallel_for(js, 0, N, [&](std::size_t b, std::size_t) {
            chunks_run.fetch_add(1, std::memory_order_relaxed);
            if (b == 0) {
                throw std::runtime_error("boom");
            }
        }, /*grain=*/1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
    EXPECT_EQ(chunks_run.load(), static_cast<int>(N)); // every chunk ran before the rethrow
    EXPECT_EQ(js.pending_count(), 0u);
    js.shutdown();
}

// Deterministic re-check of the drained-pending contract under iteration —
// the shape that used to flake (KNOWN FLAKY ParallelForTest.
// PendingCountIsZeroOnceDrained) now pinned tightly on a caller-owned pool.
TEST(ParallelForJobSystemOverloadTest, PendingCountZeroAfterEveryIteration) {
    JobSystem js;
    js.init(3);
    for (int it = 0; it < 200; ++it) {
        parallel_for(js, 0, 512, [](std::size_t, std::size_t) {}, /*grain=*/32);
        ASSERT_EQ(js.pending_count(), 0u) << "iteration " << it;
    }
    js.shutdown();
}
