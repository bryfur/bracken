// Phase 4: the sim-stage tier, tested against the three cases a thread_local
// depth counter gets wrong.
//
// The claim being tested is not "a flag can be set and read" — it is that the
// stage follows the JOB rather than the thread, so it survives work stealing,
// nested fork-join, and a helping wait that runs a sim-stage job ON THE MAIN
// THREAD. That last one is why BRACKEN_ASSERT_MAIN_THREAD alone was never
// enough: during Application::join_sim the main thread executes stolen sim
// jobs, and a GPU call from one of them would pass a main-thread check.

#include <gtest/gtest.h>

#include "bracken/core/jobs/job_stage.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/core/jobs/task_group.hpp"

#include <atomic>
#include <vector>

using namespace bracken::core::jobs;

namespace {

TEST(JobStage, DefaultsToNoneOnAPlainThread) {
    EXPECT_EQ(current_stage(), Stage::None);
}

TEST(JobStage, ScopeRestoresThePreviousValue) {
    ASSERT_EQ(current_stage(), Stage::None);
    {
        StageScope outer(Stage::Sim);
        EXPECT_EQ(current_stage(), Stage::Sim);
        {
            // Nesting the other way round is legal and must unwind: a sim job
            // that helps execute a main-stage job goes back to Sim after.
            StageScope inner(Stage::None);
            EXPECT_EQ(current_stage(), Stage::None);
        }
        EXPECT_EQ(current_stage(), Stage::Sim);
    }
    EXPECT_EQ(current_stage(), Stage::None);
}

TEST(JobStage, JobsInheritTheSubmittersStage) {
    JobSystem js;
    js.init(4);

    std::atomic<int> sim_seen{0};
    std::atomic<int> none_seen{0};

    // Root job carries no stage (submitted from a plain thread)...
    TaskHandle plain = js.submit([&] {
        if (current_stage() == Stage::None) {
            none_seen.fetch_add(1, std::memory_order_relaxed);
        }
    });
    js.wait(plain);

    // ...while a job submitted from inside a Sim scope carries Sim, and so do
    // its own children, transitively, wherever they end up running.
    TaskHandle root = js.submit([&] {
        StageScope sim(Stage::Sim);
        TaskGroup group;
        for (int i = 0; i < 32; ++i) {
            js.submit(group, [&] {
                if (current_stage() == Stage::Sim) {
                    sim_seen.fetch_add(1, std::memory_order_relaxed);
                }
                // Grandchildren inherit too.
                TaskGroup inner;
                js.submit(inner, [&] {
                    if (current_stage() == Stage::Sim) {
                        sim_seen.fetch_add(1, std::memory_order_relaxed);
                    }
                });
                js.wait(inner);
            });
        }
        js.wait(group);
    });
    js.wait(root);
    js.shutdown();

    EXPECT_EQ(none_seen.load(), 1);
    EXPECT_EQ(sim_seen.load(), 64);
}

TEST(JobStage, ParallelForChunkBodiesRunAtTheSubmittersStage) {
    JobSystem js;
    js.init(4);

    std::atomic<int> wrong{0};
    std::atomic<int> total{0};
    TaskHandle root = js.submit([&] {
        StageScope sim(Stage::Sim);
        // parallel_for dispatches chunk bodies onto OTHER worker threads —
        // exactly the case a per-thread counter cannot cover.
        parallel_for(js, std::size_t{0}, std::size_t{4096}, [&](std::size_t b, std::size_t e) {
            if (current_stage() != Stage::Sim) {
                wrong.fetch_add(1, std::memory_order_relaxed);
            }
            total.fetch_add(static_cast<int>(e - b), std::memory_order_relaxed);
        }, 64);
    });
    js.wait(root);
    js.shutdown();

    EXPECT_EQ(wrong.load(), 0);
    EXPECT_EQ(total.load(), 4096);
}

TEST(JobStage, HelpingWaitOnTheCallingThreadDoesNotLeakTheStage) {
    JobSystem js;
    js.init(1); // one worker: the waiter is forced to help

    std::atomic<bool> saw_sim_inside{false};
    TaskGroup group;
    {
        StageScope sim(Stage::Sim);
        for (int i = 0; i < 8; ++i) {
            js.submit(group, [&] {
                if (current_stage() == Stage::Sim) {
                    saw_sim_inside.store(true, std::memory_order_relaxed);
                }
            });
        }
    }
    // The submitting thread is back to None here, yet the jobs it queued still
    // carry Sim; it will execute some of them itself inside this wait and must
    // end up back at None afterwards.
    ASSERT_EQ(current_stage(), Stage::None);
    js.wait(group);
    EXPECT_EQ(current_stage(), Stage::None);
    EXPECT_TRUE(saw_sim_inside.load());

    js.shutdown();
}

} // namespace
