#pragma once

#include "bracken/core/assert.hpp"

#include <cstdint>

namespace bracken::core::jobs {

/**
 * Which pipeline stage the calling code is executing on behalf of.
 *
 * This is the mechanism behind BRACKEN_ASSERT_SIM_STAGE /
 * BRACKEN_ASSERT_NOT_SIM_STAGE, and the reason it is not simply a
 * `thread_local` depth counter set around on_simulate:
 *
 *   1. `parallel_for` dispatches chunk bodies onto OTHER worker threads. A
 *      counter on the submitting thread says nothing about them, so a chunk
 *      body that reaches for the GPU would sail past the check.
 *   2. `JobSystem::wait` is a HELPING wait — it executes stolen jobs on
 *      whatever thread is waiting, including the main thread inside
 *      Application::join_sim(). A per-thread flag set once would then mark
 *      the main thread as sim-stage for the rest of the frame, or (worse) a
 *      sim-stage chunk running on the main thread as main-thread tier.
 *
 * So the stage travels WITH THE JOB. Each JobNode carries the stage of the
 * context that submitted it, and the executor installs that stage around the
 * invoke and restores the previous value afterwards. The result is correct
 * under work stealing, under nested fork-join, and on the main thread while it
 * helps — the three cases the naive version gets wrong.
 *
 * The stage is a debug-build diagnostic: the accessors are always compiled
 * (they are trivial), the assertion macros are not.
 */
enum class Stage : uint8_t {
    /// Main thread outside a sim job, engine boot/shutdown, render recording,
    /// and pool jobs submitted from any of those (e.g. the renderer's Prepare
    /// fan-out).
    None = 0,
    /// Inside Application's sim job or anything it submitted, transitively.
    /// Tier `sim-stage`: no GPU, no SDL, no main-thread-tier engine API.
    Sim = 1,
};

namespace detail {

/// One thread_local per thread, shared across TUs (inline function statics are
/// a single entity). Written only by StageScope and the job executor.
inline Stage& current_stage_ref() noexcept {
    static thread_local Stage stage = Stage::None;
    return stage;
}

} // namespace detail

/// The stage the calling thread is currently executing for.
[[nodiscard]] inline Stage current_stage() noexcept {
    return detail::current_stage_ref();
}

/// Install `stage` for the calling thread until scope exit, restoring exactly
/// what was there before (so nesting — a sim job that helps execute a
/// main-stage job, or vice versa — unwinds correctly).
class StageScope {
public:
    explicit StageScope(Stage stage) noexcept : previous_(detail::current_stage_ref()) {
        detail::current_stage_ref() = stage;
    }
    ~StageScope() noexcept { detail::current_stage_ref() = previous_; }

    StageScope(const StageScope&) = delete;
    StageScope& operator=(const StageScope&) = delete;
    StageScope(StageScope&&) = delete;
    StageScope& operator=(StageScope&&) = delete;

private:
    Stage previous_;
};

} // namespace bracken::core::jobs

// Enforces the documented sim-stage tier. Zero cost in release builds, like
// BRACKEN_ASSERT_MAIN_THREAD.
#ifdef NDEBUG
#define BRACKEN_ASSERT_SIM_STAGE() ((void)0)
#define BRACKEN_ASSERT_NOT_SIM_STAGE() ((void)0)
#else
#define BRACKEN_ASSERT_SIM_STAGE()                                                              \
    BRACKEN_VERIFY(::bracken::core::jobs::current_stage() == ::bracken::core::jobs::Stage::Sim, \
                   "must be called from the sim stage (on_simulate or a job it submitted)")
// Uses __func__ at the CALL site, so the abort names the API that was called
// rather than the macro.
#define BRACKEN_ASSERT_NOT_SIM_STAGE()                                                          \
    BRACKEN_VERIFY(::bracken::core::jobs::current_stage() != ::bracken::core::jobs::Stage::Sim, \
                   "{}() is not reachable from the sim stage: on_simulate runs on a pool "      \
                   "thread and may not touch GPU/SDL/main-thread-tier APIs (see "               \
                   "docs/architecture.md, \"Threading & lifetime contracts\")",                 \
                   __func__)
#endif
