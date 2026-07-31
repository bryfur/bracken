#include <gtest/gtest.h>

#include "bracken/core/jobs/job_system.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace bracken::core::jobs;

namespace {

class JobSystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!JobSystem::instance().is_initialized()) {
            JobSystem::instance().init();
        }
    }
};

} // namespace

TEST_F(JobSystemTest, InitIsIdempotent) {
    auto& js = JobSystem::instance();
    const unsigned wc = js.worker_count();
    js.init();
    EXPECT_EQ(js.worker_count(), wc);
    EXPECT_TRUE(js.is_initialized());
    EXPECT_GE(wc, 1u);
}

TEST_F(JobSystemTest, SubmitSingleTaskCompletes) {
    auto& js = JobSystem::instance();
    std::atomic<int> counter{0};
    auto h = js.submit([&]() { counter.fetch_add(1); });
    js.wait(h);
    EXPECT_EQ(counter.load(), 1);
    EXPECT_TRUE(h.is_done());
}

TEST_F(JobSystemTest, SubmitManyTasksAllRun) {
    auto& js = JobSystem::instance();
    constexpr int N = 256;
    std::atomic<int> counter{0};
    std::vector<TaskHandle> handles;
    handles.reserve(N);
    for (int i = 0; i < N; ++i) {
        handles.push_back(js.submit([&]() { counter.fetch_add(1); }));
    }
    for (auto& h : handles) js.wait(h);
    EXPECT_EQ(counter.load(), N);
}

TEST_F(JobSystemTest, WaitAllDrainsEverything) {
    auto& js = JobSystem::instance();
    constexpr int N = 1024;
    std::atomic<int> counter{0};
    for (int i = 0; i < N; ++i) {
        (void)js.submit([&]() { counter.fetch_add(1); });
    }
    js.wait_all();
    EXPECT_EQ(counter.load(), N);
}

TEST_F(JobSystemTest, WaitByHelpingDrainsNestedSubmits) {
    auto& js = JobSystem::instance();
    std::atomic<int> outer{0};
    std::atomic<int> inner{0};

    std::vector<TaskHandle> outer_handles;
    for (int i = 0; i < 16; ++i) {
        outer_handles.push_back(js.submit([&]() {
            outer.fetch_add(1);
            std::vector<TaskHandle> inner_handles;
            for (int k = 0; k < 8; ++k) {
                inner_handles.push_back(js.submit([&]() { inner.fetch_add(1); }));
            }
            for (auto& ih : inner_handles) JobSystem::instance().wait(ih);
        }));
    }
    for (auto& h : outer_handles) js.wait(h);
    EXPECT_EQ(outer.load(), 16);
    EXPECT_EQ(inner.load(), 16 * 8);
}

TEST_F(JobSystemTest, ExceptionPropagatesViaWait) {
    auto& js = JobSystem::instance();
    auto h = js.submit([]() { throw std::runtime_error("boom"); });
    EXPECT_THROW(js.wait(h), std::runtime_error);
    EXPECT_TRUE(h.is_done());
    EXPECT_TRUE(h.has_error());
}

TEST_F(JobSystemTest, ExceptionDoesNotKillPool) {
    auto& js = JobSystem::instance();
    auto bad = js.submit([]() { throw std::logic_error("x"); });
    try {
        js.wait(bad);
    } catch (...) {}

    std::atomic<int> ok{0};
    auto good = js.submit([&]() { ok.fetch_add(1); });
    js.wait(good);
    EXPECT_EQ(ok.load(), 1);
}

TEST_F(JobSystemTest, StressOneHundredKTasks) {
    auto& js = JobSystem::instance();
    constexpr int N = 100'000;
    std::atomic<int> counter{0};
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        (void)js.submit([&]() { counter.fetch_add(1, std::memory_order_relaxed); });
    }
    js.wait_all();
    auto t1 = std::chrono::steady_clock::now();
    EXPECT_EQ(counter.load(), N);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    EXPECT_LT(ms, 30'000);
}

TEST_F(JobSystemTest, MultiThreadSubmitters) {
    auto& js = JobSystem::instance();
    constexpr int THREADS = 4;
    constexpr int PER_THREAD = 1000;
    std::atomic<int> counter{0};

    std::vector<std::thread> submitters;
    for (int t = 0; t < THREADS; ++t) {
        submitters.emplace_back([&]() {
            std::vector<TaskHandle> handles;
            handles.reserve(PER_THREAD);
            for (int i = 0; i < PER_THREAD; ++i) {
                handles.push_back(js.submit([&]() { counter.fetch_add(1, std::memory_order_relaxed); }));
            }
            for (auto& h : handles) JobSystem::instance().wait(h);
        });
    }
    for (auto& t : submitters) t.join();
    EXPECT_EQ(counter.load(), THREADS * PER_THREAD);
}

TEST_F(JobSystemTest, WorkerCountReportedCorrectly) {
    EXPECT_GE(JobSystem::instance().worker_count(), 1u);
}

// Handle copies are independent references to the same task.
TEST_F(JobSystemTest, HandleCopiesShareCompletionState) {
    auto& js = JobSystem::instance();
    auto h = js.submit([]() {});
    TaskHandle copy = h;
    js.wait(h);
    EXPECT_TRUE(h.is_done());
    EXPECT_TRUE(copy.is_done());
    EXPECT_FALSE(copy.has_error());
    h = TaskHandle{}; // dropping one reference must not invalidate the other
    EXPECT_TRUE(copy.is_done());
    js.wait(copy); // no-op, already done
}

// The tests below use caller-owned pools (JobSystem is publicly constructible
// by design) so forced worker counts and shutdown sequences don't perturb the
// shared singleton the fixture tests run against.

// H2 regression: nested submit+wait (fork-join) on a forced 1-worker pool.
// The worker's sub-jobs land in its own deque and nothing can steal them;
// wait() must drain the caller's own deque or this livelocks deterministically.
TEST(JobSystemSingleWorkerTest, NestedSubmitWaitDoesNotLivelock) {
    JobSystem js;
    js.init(1);
    ASSERT_EQ(js.worker_count(), 1u);

    std::atomic<int> outer{0};
    std::atomic<int> inner{0};

    std::vector<TaskHandle> outer_handles;
    for (int i = 0; i < 4; ++i) {
        outer_handles.push_back(js.submit([&]() {
            outer.fetch_add(1);
            std::vector<TaskHandle> inner_handles;
            for (int k = 0; k < 8; ++k) {
                inner_handles.push_back(js.submit([&]() { inner.fetch_add(1); }));
            }
            for (auto& ih : inner_handles) js.wait(ih);
        }));
    }
    for (auto& h : outer_handles) js.wait(h);
    EXPECT_EQ(outer.load(), 4);
    EXPECT_EQ(inner.load(), 4 * 8);
    js.shutdown();
}

// M33 regression: shutdown() must complete queued-but-unstarted jobs with a
// "discarded" error so retained handles report done + error instead of a
// wait() hanging forever.
TEST(JobSystemShutdownTest, ShutdownCompletesDiscardedJobsWithError) {
    JobSystem js;
    js.init(1);

    std::atomic<bool> started{false};
    std::atomic<bool> gate{false};
    auto blocker = js.submit([&]() {
        started.store(true, std::memory_order_release);
        started.notify_one();
        gate.wait(false, std::memory_order_acquire);
    });
    started.wait(false, std::memory_order_acquire); // the only worker is now gated inside `blocker`

    std::atomic<bool> victim_ran{false};
    auto victim = js.submit([&]() { victim_ran.store(true, std::memory_order_release); });

    std::thread killer([&]() { js.shutdown(); });

    // shutdown()'s pre-join discard pass completes the queued victim while
    // the worker is still gated; if that pass (or its done-publish) broke,
    // this loop would never exit and the test would time out.
    while (!victim.is_done()) {
        std::this_thread::yield();
    }

    gate.store(true, std::memory_order_release);
    gate.notify_all();
    killer.join();

    EXPECT_FALSE(victim_ran.load(std::memory_order_acquire));
    EXPECT_TRUE(victim.is_done());
    EXPECT_TRUE(victim.has_error());
    EXPECT_NE(victim.error(), nullptr);
    EXPECT_THROW(js.wait(victim), std::runtime_error);

    EXPECT_TRUE(blocker.is_done());
    EXPECT_FALSE(blocker.has_error());
    EXPECT_FALSE(js.is_initialized());
    EXPECT_EQ(js.pending_count(), 0u);
}

// L65: an external waiter with nothing to help with (the only job is
// executing on the gated worker) parks on the task's done flag; the
// completion notify must wake it — a lost wakeup would hang the test.
TEST(JobSystemWaitTest, ExternalWaiterWakesWhenGatedJobCompletes) {
    JobSystem js;
    js.init(1);

    std::atomic<bool> started{false};
    std::atomic<bool> gate{false};
    auto h = js.submit([&]() {
        started.store(true, std::memory_order_release);
        started.notify_one();
        gate.wait(false, std::memory_order_acquire);
    });
    started.wait(false, std::memory_order_acquire);

    std::thread waiter([&]() { js.wait(h); });

    EXPECT_FALSE(h.is_done());
    gate.store(true, std::memory_order_release);
    gate.notify_all();

    waiter.join();
    EXPECT_TRUE(h.is_done());
    js.shutdown();
}

// M31: error() is valid only after is_done() — it must report nullptr while
// the task is in flight and expose the exception once done.
TEST(JobSystemErrorTest, ErrorIsNullUntilDoneThenExposesException) {
    JobSystem js;
    js.init(1);

    std::atomic<bool> started{false};
    std::atomic<bool> gate{false};
    auto h = js.submit([&]() {
        started.store(true, std::memory_order_release);
        started.notify_one();
        gate.wait(false, std::memory_order_acquire);
        throw std::runtime_error("late failure");
    });
    started.wait(false, std::memory_order_acquire);

    EXPECT_FALSE(h.is_done());
    EXPECT_EQ(h.error(), nullptr);
    EXPECT_FALSE(h.has_error());

    gate.store(true, std::memory_order_release);
    gate.notify_all();

    EXPECT_THROW(js.wait(h), std::runtime_error);
    EXPECT_TRUE(h.is_done());
    EXPECT_TRUE(h.has_error());
    EXPECT_NE(h.error(), nullptr);
    js.shutdown();
}

// L66: shutdown()+init() re-initializes the pool; pin it so the supported
// re-init path doesn't rot.
TEST(JobSystemLifecycleTest, InitAfterShutdownReinitializes) {
    JobSystem js;
    js.init(2);
    EXPECT_TRUE(js.is_initialized());
    EXPECT_EQ(js.worker_count(), 2u);

    std::atomic<int> ran{0};
    js.wait(js.submit([&]() { ran.fetch_add(1); }));
    EXPECT_EQ(ran.load(), 1);

    js.shutdown();
    EXPECT_FALSE(js.is_initialized());
    EXPECT_EQ(js.worker_count(), 0u);

    js.init(2);
    EXPECT_TRUE(js.is_initialized());
    EXPECT_EQ(js.worker_count(), 2u);
    js.wait(js.submit([&]() { ran.fetch_add(1); }));
    EXPECT_EQ(ran.load(), 2);
    js.shutdown();
    EXPECT_FALSE(js.is_initialized());
}

// Handles issued before a shutdown()/init() cycle stay valid and queryable
// afterwards (pooled node storage persists until JobSystem destruction).
TEST(JobSystemLifecycleTest, HandlesSurviveShutdownReinitCycle) {
    JobSystem js;
    js.init(2);
    auto h = js.submit([]() {});
    js.wait(h);
    js.shutdown();
    EXPECT_TRUE(h.is_done());
    EXPECT_FALSE(h.has_error());

    js.init(2);
    EXPECT_TRUE(h.is_done());
    js.wait(h); // completed long ago; returns immediately
    js.shutdown();
}

// Generation/ABA safety: a stale handle fabricated with an outdated
// generation for a recycled-and-reused node must report done (that task DID
// complete), never an in-flight state, an aliased error, or UB. The current
// occupant's handle is held alive throughout, so the fabricated reference is
// safe by the TaskHandle(node, gen) liveness contract.
TEST(JobSystemGenerationTest, StaleHandleReportsDoneAfterNodeReuse) {
    JobSystem js;
    js.init(1);

    auto first = js.submit([]() {});
    js.wait(first);
    JobNode* node = first.node();
    const std::uint32_t old_gen = first.generation();
    first = TaskHandle{}; // drop the handle ref; the node recycles once the pool cycles it

    // Churn single submits until the pool hands the same node to a new task
    // (external-pool freelist reuse makes this quick), keeping THAT task's
    // handle alive so the node stays referenced below.
    TaskHandle current;
    bool reused = false;
    for (int i = 0; i < 100'000 && !reused; ++i) {
        TaskHandle t = js.submit([]() {});
        js.wait(t);
        if (t.node() == node && t.generation() != old_gen) {
            current = t;
            reused = true;
        }
    }
    ASSERT_TRUE(reused) << "pool never reused the node; freelist recycling is broken";

    TaskHandle stale(node, old_gen); // internal ctor: fabricated stale handle
    EXPECT_TRUE(stale.is_done());
    EXPECT_FALSE(stale.has_error());
    EXPECT_EQ(stale.error(), nullptr);
    js.wait(stale); // must return immediately, not park on the reused node

    stale = TaskHandle{};   // release order: fabricated ref first...
    current = TaskHandle{}; // ...then the live occupant's
    js.shutdown();
}

// Park/wake stress: rapid submit bursts from an external thread against a
// pool that goes idle (parks) between bursts. A lost wakeup in the epoch
// protocol would hang a wait_all; a miscounted pending_ would fail the sum.
TEST(JobSystemParkStressTest, RapidExternalBurstsAgainstIdlePool) {
    JobSystem js;
    js.init(2);

    std::atomic<int> ran{0};
    int expected = 0;
    for (int burst = 0; burst < 300; ++burst) {
        const int burst_size = 1 + (burst % 7);
        for (int i = 0; i < burst_size; ++i) {
            (void)js.submit([&]() { ran.fetch_add(1, std::memory_order_relaxed); });
        }
        expected += burst_size;
        js.wait_all();
        ASSERT_EQ(js.pending_count(), 0u);
        if (burst % 3 == 0) {
            // Give the workers time to spin down and actually park, so the
            // next burst exercises the futex wake path, not the spin path.
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
    EXPECT_EQ(ran.load(), expected);
    js.shutdown();
}

// Regression (TSan, 2026-07-19): refcount resurrection through stale handles.
// With generation and refcount in separate atomics, a stale handle's add_ref
// could interleave with submit's refcount initialization on a recycled node;
// the stale copy's later drop then drove the LIVE task's count to zero while
// it was still queued/executing — nodes recycled under running jobs and
// submit() reassigned `fn` mid-invocation. The fix packs {generation,
// refcount} into one atomic word (JobNode::state) so every ref op is
// generation-checked in a single CAS and stale ops no-op.
//
// This test churns copies/destructions of aging handles from multiple
// threads while the pool rapidly recycles nodes for new jobs, and asserts
// every job executes exactly once. Under TSan this must also be race-clean.
TEST(JobSystemStaleChurnTest, StaleHandleChurnCannotResurrectNodes) {
    JobSystem js;
    js.init(3);

    constexpr int k_rounds = 200;
    constexpr int k_jobs_per_round = 32;
    std::array<std::atomic<int>, k_rounds * k_jobs_per_round> exec_counts{};

    // Aging handles shared with churn threads. The mutex only guards the
    // vector structure; the churned TaskHandle copies themselves are the
    // lock-free machinery under test.
    std::mutex handles_mutex;
    std::vector<TaskHandle> aging;
    std::atomic<bool> stop_churn{false};

    std::vector<std::thread> churners;
    churners.reserve(3);
    for (int c = 0; c < 3; ++c) {
        churners.emplace_back([&]() {
            while (!stop_churn.load(std::memory_order_acquire)) {
                TaskHandle copy;
                {
                    std::lock_guard<std::mutex> lock(handles_mutex);
                    if (!aging.empty()) {
                        copy = aging[aging.size() / 2]; // copy: generation-checked add_ref
                    }
                }
                // Query through the copy (done/error paths), then drop it —
                // the drop must never deflate a NEW task's refcount.
                (void)copy.is_done();
                (void)copy.has_error();
            }
        });
    }

    for (int r = 0; r < k_rounds; ++r) {
        std::vector<TaskHandle> round;
        round.reserve(k_jobs_per_round);
        for (int j = 0; j < k_jobs_per_round; ++j) {
            const int idx = r * k_jobs_per_round + j;
            round.push_back(js.submit([&exec_counts, idx]() {
                exec_counts[static_cast<std::size_t>(idx)].fetch_add(1, std::memory_order_relaxed);
            }));
        }
        for (auto& h : round) {
            js.wait(h);
        }
        {
            // Retire this round's handles into the churn set; cap its size so
            // most nodes recycle (dropping the oldest handles) while churners
            // still race copies against those recycles.
            std::lock_guard<std::mutex> lock(handles_mutex);
            for (auto& h : round) {
                aging.push_back(std::move(h));
            }
            if (aging.size() > 64) {
                aging.erase(aging.begin(), aging.end() - 48);
            }
        }
    }

    stop_churn.store(true, std::memory_order_release);
    for (auto& t : churners) {
        t.join();
    }
    {
        std::lock_guard<std::mutex> lock(handles_mutex);
        aging.clear();
    }
    js.wait_all();

    for (int i = 0; i < k_rounds * k_jobs_per_round; ++i) {
        ASSERT_EQ(exec_counts[static_cast<std::size_t>(i)].load(), 1)
            << "job " << i << " executed " << exec_counts[static_cast<std::size_t>(i)].load()
            << " times — node lifecycle corruption";
    }
    js.shutdown();
}
