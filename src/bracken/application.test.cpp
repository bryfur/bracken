#include "bracken/application.hpp"

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/sim_frame.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <utility>

using bracken::Application;
using bracken::GraphicsSettings;
using bracken::SimFrame;
using bracken::core::jobs::JobSystem;

// Headless pipeline-coordinator tests: no init_engine(), no renderer, no
// window. run()'s loop only needs the job-system singleton (initialized here,
// the parallel_for.test.cpp pattern) — event pumping and input polling are
// no-ops without SDL video, rendering is skipped without a renderer, and the
// packet ring is plain memory. This exercises the Phase-4 schedule mechanics:
// kick → join → fence → retire, quit from the sim job, and the sim-error
// rethrow-at-join policy.

namespace {

class HeadlessApp : public Application {
public:
    int sim_frames = 0;
    int main_frames = 0;
    int quit_at_frame = -1;  // quit() from sim when sim_frames reaches this
    int throw_at_frame = -1; // throw from sim when sim_frames reaches this
    int inits = 0;
    int shutdowns = 0;
    bool init_result = true;

protected:
    bool on_init() override {
        ++inits;
        return init_result;
    }

    void on_shutdown() override { ++shutdowns; }

public:
    // set_camera_driven_by_engine / camera_driven_by_engine are protected
    // facade; expose them for the test below.
    void take_the_camera() { set_camera_driven_by_engine(false); }
    using Application::camera_driven_by_engine;
    using Application::edit_graphics_settings;
    using Application::graphics_settings;

protected:

    void on_simulate(SimFrame& sim) override {
        ++sim_frames;
        // Touch the frame surface so the sealed packet is non-trivial.
        sim.frame().set_draw_skybox(false);
        if (sim_frames == throw_at_frame) {
            throw std::runtime_error("sim boom");
        }
        if (sim_frames == quit_at_frame) {
            quit();
        }
    }

    void on_main_frame(bracken::MainFrame& mf) override {
        ++main_frames;
        (void)mf;
    }
};

class ApplicationPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!JobSystem::instance().is_initialized()) {
            JobSystem::instance().init();
        }
    }
};

} // namespace

TEST_F(ApplicationPipelineTest, QuitFromSimJobJoinsAndReturns) {
    HeadlessApp app;
    app.quit_at_frame = 3;
    app.run(); // must return (in-flight sim joined) — not hang or crash
    // quit() lands in sim frame 3; the loop may kick at most one more frame
    // before observing the flag (pipeline depth 2).
    EXPECT_GE(app.sim_frames, 3);
    EXPECT_LE(app.sim_frames, 4);
    // The fence ran for every joined frame before the quit was observed.
    EXPECT_GE(app.main_frames, 1);
}

TEST_F(ApplicationPipelineTest, SimExceptionIsRethrownOnTheMainThreadAtJoin) {
    HeadlessApp app;
    app.throw_at_frame = 2;
    EXPECT_THROW(app.run(), std::runtime_error);
    EXPECT_EQ(app.sim_frames, 2);
    // The pipeline abandoned the failed frame; a fresh run() must work
    // (packet ring back to Retired, no sim in flight).
    app.throw_at_frame = -1;
    app.quit_at_frame = 4;
    app.run();
    EXPECT_GE(app.sim_frames, 4);
}

// ===== Boot: start()/stop() drive the protected hooks =====

TEST_F(ApplicationPipelineTest, StartRunsOnInitAndStopRunsOnShutdownExactlyOnce) {
    HeadlessApp app;
    EXPECT_TRUE(app.start());
    EXPECT_EQ(app.inits, 1);
    EXPECT_EQ(app.shutdowns, 0);
    app.stop();
    EXPECT_EQ(app.shutdowns, 1);
    app.stop(); // idempotent: a second teardown must not re-run on_shutdown
    EXPECT_EQ(app.shutdowns, 1);
}

TEST_F(ApplicationPipelineTest, StopIsANoOpWhenStartFailed) {
    HeadlessApp app;
    app.init_result = false;
    EXPECT_FALSE(app.start());
    EXPECT_EQ(app.inits, 1);
    // The canonical main() bails to shutdown_engine() here; a host that calls
    // stop() anyway must not unwind a game that never came up.
    app.stop();
    EXPECT_EQ(app.shutdowns, 0);
}

// ===== Camera: the engine drives the rig unless the host takes it back =====

TEST_F(ApplicationPipelineTest, CameraIsEngineDrivenByDefaultAndTheHatchToggles) {
    HeadlessApp app;
    // The default matters: a host that never thinks about the camera gets one
    // that updates, lands in the packet, and feeds the movement basis.
    EXPECT_TRUE(app.camera_driven_by_engine());
    app.take_the_camera();
    EXPECT_FALSE(app.camera_driven_by_engine());
    // Without a renderer there is no rig at all, so the drive is a no-op —
    // run() must still complete every frame.
    app.quit_at_frame = 2;
    app.run();
    EXPECT_GE(app.sim_frames, 2);
}


// ===== Settings: the scoped edit commits itself, or does nothing =====

TEST_F(ApplicationPipelineTest, GraphicsEditIsDeferredNotImmediate) {
    HeadlessApp app;
    const bool before = app.graphics_settings().bloom_enabled;
    {
        auto gs = app.edit_graphics_settings();
        gs->bloom_enabled = !before;
        EXPECT_EQ(gs->bloom_enabled, !before);
    }
    // The edit is a REQUEST: applying it is GPU/SDL work that happens at the
    // frame fence, so the persisted copy does not change under the caller.
    EXPECT_EQ(app.graphics_settings().bloom_enabled, before);
}

TEST_F(ApplicationPipelineTest, UnchangedGraphicsEditRequestsNothing) {
    HeadlessApp app;
    {
        auto gs = app.edit_graphics_settings();
        (void)gs->bloom_enabled; // read-only: nothing to commit
    }
    // Proof by consequence: applying settings needs a renderer (the facade
    // setters are BRACKEN_CONTRACT-guarded), so a spurious request from an
    // untouched edit would abort this renderer-less run at the first fence.
    app.quit_at_frame = 2;
    app.run();
    EXPECT_GE(app.sim_frames, 2);
}

TEST_F(ApplicationPipelineTest, MovedFromGraphicsEditDoesNotAlsoCommit) {
    HeadlessApp app;
    auto gs = app.edit_graphics_settings();
    gs->bloom_enabled = !gs->bloom_enabled;
    {
        // Exactly one handle may commit; the moved-from one is disarmed. A
        // double commit here would be two requests, the second of which is
        // the stale copy.
        bracken::GraphicsSettingsEdit moved = std::move(gs);
        (void)moved;
    }
    EXPECT_EQ(app.graphics_settings().bloom_enabled, GraphicsSettings{}.bloom_enabled);
}
