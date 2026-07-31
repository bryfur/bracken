#pragma once

#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/core/jobs/task_group.hpp"
#include "bracken/input_handler.hpp" // InputSnapshot
#include "bracken/scene/frame_sink.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

namespace bracken {

class Application;

/**
 * SimFeedback — deferred main-thread requests a sim job records through its
 * SimFrame, applied by Application::run at the NEXT frame fence (after the
 * sim job is joined, before events are pumped). Engine-internal: owned by
 * Application, written only by the in-flight sim job, read only at the fence
 * — ordering rides the sim-job kick(submit publish)/join(TaskHandle wait)
 * edges, so no atomics are needed (single writer per phase).
 */
struct SimFeedback {
    bool has_camera_forward = false;
    float camera_forward_x = 0.0f;
    float camera_forward_z = -1.0f;
};

/**
 * SimFrame — the host's view of one simulation frame, passed to
 * Application::on_simulate (docs/multithreading_plan.md §3.1).
 *
 * Thread-safety tier: `sim-stage`. on_simulate RUNS AS A JOB ON THE POOL,
 * overlapped with the previous frame's rendering on the main thread.
 * Everything reachable through this object is safe from the sim job (and,
 * via parallel_for/run_async, from any job it fans out to — respecting the
 * FrameBuilder lane contract). NOTHING reachable from the sim stage may
 * touch the GPU, SDL, or any `main-thread`-tier engine API (swapchain,
 * ImGui/DevUI, GPU create/destroy, FileWatcher, the live InputHandler) — the
 * existing BRACKEN_ASSERT_MAIN_THREAD checks on those entry points enforce
 * it in debug. Main-thread work goes in Application::on_main_frame.
 *
 * The SimFrame is valid only for the duration of on_simulate — never store
 * a pointer/reference to it (or to input()) beyond the callback.
 */
class SimFrame {
public:
    /** Engine-internal constructor (Application's sim job + tests): binds
     *  the frame contexts. Host code never constructs SimFrames. */
    SimFrame(scene::FrameBuilder& builder, const InputSnapshot& input, float dt, double time, uint64_t frame_index,
             SimFeedback* feedback = nullptr)
        : sink_(builder), input_(&input), feedback_(feedback), dt_(dt), time_(time), frame_index_(frame_index) {}

    SimFrame(const SimFrame&) = delete;
    SimFrame& operator=(const SimFrame&) = delete;
    SimFrame(SimFrame&&) = delete;
    SimFrame& operator=(SimFrame&&) = delete;

    /** Frame delta time in seconds (computed at kick time, clamped to 0.1 s;
     *  identical to frame().packet dt). */
    [[nodiscard]] float dt() const { return dt_; }

    /** Accumulated engine time in seconds (sum of every frame's dt). */
    [[nodiscard]] double time() const { return time_; }

    /** Monotonically increasing sim-frame index (0 on the first frame). */
    [[nodiscard]] uint64_t frame_index() const { return frame_index_; }

    /**
     * The frame's immutable input snapshot, taken on the main thread at this
     * frame's kick. By value in the pipeline sense: no reference back into
     * the live InputHandler. LATENCY: this snapshot was taken one pipeline
     * slot before this frame presents (see Application's class comment).
     */
    [[nodiscard]] const InputSnapshot& input() const { return *input_; }

    /**
     * The frame's sink — the ONLY way render state leaves the game (draws,
     * lights, UI, camera, environment, effect spawns). Emit from the frame's
     * root code, or fan emission out across lanes with your own run_async
     * jobs (one lane per job — see FrameBuilder's lane contract).
     *
     * scene::FrameSink is the HOST half of the frame builder and nothing
     * else. The engine's sim-stage tail — seal(), set_frame_timing(), the
     * effect snapshot, the engine-frame stamp — is not reachable through it.
     * That rule used to be this comment ("Do NOT call seal()"), enforced by a
     * debug assert; a shipped game that sealed early republished a packet the
     * renderer was already consuming, silently. Now it does not compile.
     */
    [[nodiscard]] scene::FrameSink& frame() { return sink_; }

    /**
     * Data-parallel loop over [begin, end): body(i) per index, fanned out on
     * the engine pool and JOINED before returning — fork-join from inside
     * the sim job is supported. Same code path at every worker count; small
     * ranges run inline on the caller (granularity control, not a second
     * code path). The body must not emit into a FrameBuilder lane another
     * job is using (give each chunk its own lane or emit from the root).
     * `grain` = indices per chunk (0 = auto).
     */
    template<typename Fn> void parallel_for(std::size_t begin, std::size_t end, Fn&& body, std::size_t grain = 0) {
        core::jobs::parallel_for(
            core::jobs::JobSystem::instance(), begin, end,
            [&body](std::size_t chunk_begin, std::size_t chunk_end) {
                for (std::size_t i = chunk_begin; i < chunk_end; ++i) {
                    body(i);
                }
            },
            grain);
    }

    /**
     * Range form: body(element&) for each element of a caller-stable,
     * random-access container (the container must not resize while the loop
     * runs — pass the container itself, never a temporary).
     */
    template<typename Range, typename Fn> void parallel_for(Range& range, Fn&& body, std::size_t grain = 0) {
        parallel_for(std::size_t{0}, static_cast<std::size_t>(range.size()),
                     [&range, &body](std::size_t i) { body(range[i]); }, grain);
    }

    /**
     * Submit fire-and-forget work against a host-owned TaskGroup (e.g.
     * pathfinding that spans frames). The group MUST outlive the job and the
     * host MUST join it (core::jobs::JobSystem::instance().wait(group))
     * before touching the job's results — and before destroying the data the
     * job captures. Frame-scoped groups must be joined before on_simulate
     * returns; cross-frame groups are explicitly owned by the host.
     */
    void run_async(core::jobs::TaskGroup& group, std::function<void()> fn) {
        core::jobs::JobSystem::instance().submit(group, std::move(fn));
    }

    /**
     * Record the camera's ground-plane forward direction for the input
     * system's camera-relative movement basis (mirror of the retired
     * InputHandler::set_camera_forward call from host update code). Deferred:
     * applied to the InputHandler at the next frame fence, so it shapes the
     * NEXT snapshot's InputState::move_dir — the same one-frame latency the
     * pre-pipelined loop had. No-op when the engine did not attach feedback
     * (hand-built test frames).
     */
    void set_camera_forward(float x, float z) {
        if (feedback_ != nullptr) {
            feedback_->has_camera_forward = true;
            feedback_->camera_forward_x = x;
            feedback_->camera_forward_z = z;
        }
    }

private:
    scene::FrameSink sink_;
    const InputSnapshot* input_;
    SimFeedback* feedback_;
    float dt_;
    double time_;
    uint64_t frame_index_;
};

/**
 * MainFrame — the host's view of the frame FENCE, passed to the optional
 * Application::on_main_frame hook (docs/multithreading_plan.md §3.1).
 *
 * Thread-safety tier: `main-thread`. It runs between frames with NO sim job
 * in flight — the only point in the pipelined loop where the main thread may
 * safely read live game state (e.g. to snapshot values for DevUI panels,
 * which draw later, DURING rendering, while the next sim runs). Keep it
 * short: everything spent here lengthens the frame. Deliberately exposes no
 * GPU/renderer surface — describe frames in on_simulate; use the fence for
 * window-ish main-thread work (screenshots, snapshots, settings).
 */
class MainFrame {
public:
    MainFrame(const MainFrame&) = delete;
    MainFrame& operator=(const MainFrame&) = delete;
    MainFrame(MainFrame&&) = delete;
    MainFrame& operator=(MainFrame&&) = delete;

    /** Index of the sim frame just joined (the frame about to render). */
    [[nodiscard]] uint64_t frame_index() const { return frame_index_; }

    /** The Application's FPS counter (updated at the fence, once per second). */
    [[nodiscard]] float fps() const;

    /** Window client size in pixels (0 before the renderer is up). */
    [[nodiscard]] int screen_width() const;
    [[nodiscard]] int screen_height() const;

    /** Capture the frame about to render (UI included) to a PNG —
     *  passthrough to Application::request_screenshot. */
    void request_screenshot(std::string path);

private:
    friend class Application;
    MainFrame(Application& app, uint64_t frame_index) : app_(&app), frame_index_(frame_index) {}

    Application* app_;
    uint64_t frame_index_;
};

} // namespace bracken
