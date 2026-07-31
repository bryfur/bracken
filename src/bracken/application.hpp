#pragma once

#include "bracken/core/assert.hpp"
#include "bracken/graphics_settings.hpp"
#include "bracken/input_handler.hpp"
#include "bracken/sim_frame.hpp"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <SDL3/SDL_events.h>
#include <optional>
#include <string>
#include <vector>

#include "bracken/core/jobs/task_handle.hpp"
#include "bracken/render_stats.hpp"

namespace bracken {

namespace dev {
class DevUI;
}
namespace render {
class RenderContext;
}
namespace scene {
class SceneRenderer;
class FramePacket;
class FrameBuilder;
class PacketRing;
} // namespace scene
namespace systems {
class CameraController;
class CameraSystem;
class EffectSystem;
} // namespace systems

class ModelManager;
struct Heightmap;

/**
 * Base application class that owns the SDL lifecycle, the pipelined main
 * loop, frame timing, input handling, and core engine subsystems.
 *
 * The engine is natively multithreaded: init_engine() unconditionally brings
 * up the core::jobs::JobSystem worker pool as engine infrastructure — it is
 * not opt-in. The pool spans the whole engine lifetime (workers are joined
 * before GPU teardown in shutdown_renderer()). Worker count defaults to
 * hardware_concurrency()-1 (clamped >= 1); the BRACKEN_WORKERS environment
 * variable overrides it, with BRACKEN_WORKERS=1 as the documented
 * single-worker debug mode (same code path, one worker).
 *
 * THE FRAME PIPELINE (docs/multithreading_plan.md Phase 4, depth 2 —
 * Simulate(i+1) overlaps Render(i)). run() is the coordinator; each loop
 * iteration i:
 *
 *   1. JOIN the sim job for frame i (the main thread helps the pool while
 *      waiting). An exception escaping on_simulate is fatal-logged and
 *      RETHROWN here on the main thread — host bugs must not vanish.
 *   2. FENCE — no sim in flight: FileWatcher::poll_main_thread (hot-reload
 *      applies here, while no packet referencing the old resource is being
 *      built), pending graphics-settings application
 *      (request_graphics_settings), deferred sim requests (screenshot,
 *      camera-forward), FPS accounting, then the on_main_frame(MainFrame&)
 *      hook. This is the ONLY point in the loop where the main thread may
 *      safely read live game state (snapshot it here for DevUI panels —
 *      panels draw during rendering, while the next sim runs).
 *   3. Pump SDL events (on_event / DevUI first, then InputHandler) and take
 *      the frame's by-value InputSnapshot.
 *   4. Acquire packet i+1's FrameBuilder from the ring and KICK sim job i+1
 *      on the pool: SimFrame construction → on_simulate(sim) → the engine's
 *      sim-stage tail (apply the builder's effect spawns to the EffectSystem,
 *      EffectSystem::update on the pool, write the effect-instance snapshot
 *      + sim-advanced scene/sky clocks into the packet's engine section) →
 *      seal. Frame timing (dt) is computed at kick time and carried in the
 *      packet.
 *   5. render_frame(packet i) on the main thread (Prepare fan-out + Record +
 *      present).
 *   6. Retire packet i.
 *
 * The packet handoff needs no extra synchronization: seal happens inside the
 * sim job and the TaskHandle join publishes it to the main thread.
 *
 * INPUT LATENCY: the snapshot taken at kick(i+1) is simulated in frame i+1
 * and presents at frame i+1 — one pipeline slot between input and photons.
 * That single frame of added latency is inherent to pipelining (and stated
 * here on purpose); on_main_frame exists for the rare truly-latency-critical
 * main-thread need.
 *
 * Game-specific subclasses override on_simulate (and optionally
 * on_main_frame/on_event) and interact with the engine through the protected
 * facade. There is no on_update/on_render: the host does not "render" — it
 * DESCRIBES a frame through SimFrame::frame() and the engine draws it.
 *
 * Lifecycle is host-driven: run() drives only the per-frame hooks; the host
 * must itself call init_engine() then on_init() before run(), and
 * on_shutdown() then shutdown_engine() after it (as the playground's
 * start()/stop() wrappers do). on_init/on_shutdown run on the main thread
 * with the GPU available and no sim in flight — their contracts are
 * unchanged.
 */
class Application {
public:
    Application();
    virtual ~Application();

    /** Initialize SDL subsystems + the job-system pool. The host must call
     *  this, then its own on_init(), before run() — Application never
     *  invokes on_init() itself. Main thread (binds the main-thread assert). */
    [[nodiscard]] bool init_engine();

    /** Run the pipelined main loop until quit() is called. Drives only
     *  on_simulate/on_main_frame/on_event — never on_init/on_shutdown. Joins
     *  any in-flight sim job before returning, so teardown never races the
     *  pipeline. Rethrows the first exception a sim job raised. */
    void run();

    /** Shut down SDL. The host must call its own on_shutdown() first —
     *  Application never invokes on_shutdown() itself. */
    void shutdown_engine();

    /** Request the main loop to stop. Tier: worker-safe — callable from the
     *  sim job (or any thread); the loop observes it at the next iteration. */
    void quit() { running_.store(false, std::memory_order_relaxed); }

    /** Frames per second, updated once per second at the frame fence. Safe to
     *  read from the sim job: it is never written while a sim is in flight. */
    float fps() const { return fps_; }

    /**
     * Where the frame's wall time actually went, in milliseconds, smoothed
     * with the same exponential average the per-pass timings use (a raw
     * per-frame number at 100+ fps is unreadable and not actionable).
     *
     * The four cover the pipelined loop's four phases:
     *   sim_ms()     — the whole sim job, including its own fan-out.
     *   render_ms()  — Prepare + Record on the main thread, overlapping the
     *                  NEXT sim (depth 2), so sim_ms + render_ms routinely
     *                  exceeds the frame time. That is the pipeline working.
     *   fence_ms()   — the quiescent point: hot-reload poll, deferred
     *                  settings/screenshot, on_main_frame.
     *   present_wait_ms() — CPU time blocked in swapchain acquire + submit,
     *                  i.e. waiting for the GPU. THIS is the one to read
     *                  first: with no GPU timestamp API anywhere in SDL3 it
     *                  is the only in-engine signal distinguishing a
     *                  GPU-bound frame from a CPU-bound one. It is included
     *                  in render_ms().
     *
     * Written at the fence, so sim code may read them (never written while a
     * sim is in flight).
     */
    float sim_ms() const { return sim_ms_; }
    float render_ms() const { return render_ms_; }
    float fence_ms() const { return fence_ms_; }
    float present_wait_ms() const { return present_wait_ms_; }

protected:
    /** Game-specific initialization (renderer, assets, scene). Not called by
     *  Application — the host must invoke it after init_engine(), before
     *  run(). Main thread, GPU available, no sim in flight. */
    virtual bool on_init() = 0;

    /** Game-specific shutdown. Not called by Application — the host must
     *  invoke it after run() returns, before shutdown_engine(). Main thread,
     *  GPU available, no sim in flight. */
    virtual void on_shutdown() {}

    /**
     * The frame. RUNS AS A JOB ON THE POOL, overlapped with the previous
     * frame's rendering (see the class comment's schedule). Simulate the
     * game with sim.dt()/sim.input(), then DESCRIBE the frame through
     * sim.frame(); fan out heavy loops with sim.parallel_for. Tier:
     * sim-stage — nothing reachable from here may touch the GPU, SDL, or any
     * main-thread-tier engine API (those entry points assert). Do not seal
     * the builder or set frame timing — the engine does both.
     */
    virtual void on_simulate(SimFrame& sim) = 0;

    /**
     * Optional main-thread hook, called at the frame fence (step 2: after
     * the previous sim joined, before events are pumped — NO sim in flight).
     * The one safe place to read live game state from the main thread; use
     * it for window/screenshot work and for snapshotting state DevUI panels
     * display. Most games never override it.
     */
    virtual void on_main_frame(MainFrame& mf) { (void)mf; }

    /** Called for each SDL event before InputHandler processes it — on the
     *  main thread, during the event pump (no sim in flight, so touching
     *  host state here is safe). Return true to consume the event (prevent
     *  InputHandler from seeing it). */
    virtual bool on_event([[maybe_unused]] const SDL_Event& event) { return false; }

    /** The live input handler. Tier: main-thread (on_event/on_main_frame/
     *  init paths only) — sim code reads SimFrame::input() instead. */
    InputHandler& input() { return input_; }
    const InputHandler& input() const { return input_; }

    // ========== Rendering facade ==========

    /** Initialize the rendering subsystems (window, GPU, scene renderer,
     *  camera). Main thread, before run(). */
    [[nodiscard]] bool init_renderer(int width, int height, const std::string& title, float world_width = 32000.0f,
                       float world_height = 32000.0f);

    /** Shut down all rendering subsystems. */
    void shutdown_renderer();

    /**
     * Render a complete frame from a SEALED FramePacket. Main thread. Under
     * run() this is driven by the pipeline (step 5) — hosts that drive their
     * own loop instead of run() may call it directly with a packet they
     * built and sealed (the Phase-2 sequential flow still works: acquire →
     * build → seal → render_frame → retire).
     */
    void render_frame(const scene::FramePacket& frame);

    /**
     * The application's frame-packet ring (created on first use, depth 2 —
     * one slot rendering while the other is being simulated). run() owns the
     * acquire/retire cycle; ring-driving by the host is only for loops that
     * bypass run().
     */
    scene::PacketRing& frame_packets();

    void set_heightmap(const Heightmap& heightmap);

    /**
     * GRAPHICS SETTINGS APPLICATION (window mode, vsync, sampler rebuilds —
     * GPU/SDL work): main-thread tier, no sim in flight — init paths and
     * on_main_frame only. From on_simulate, use request_graphics_settings().
     */
    void set_graphics_settings(const GraphicsSettings& settings);
    void set_anisotropic_filter(AnisotropicFilter level);
    void set_vsync_mode(VSyncMode mode);
    int max_vsync_mode() const;
    void set_window_mode(WindowMode window_mode, int resolution_index = 0);

    /** Apply all graphics settings (scene renderer, vsync, anisotropic,
     *  window mode). Main-thread tier, no sim in flight (see above). */
    void apply_all_graphics_settings(const GraphicsSettings& settings);

    /**
     * Request that `settings` be applied at the next frame fence (schedule
     * step 2). Tier: sim-stage or main thread — the safe way for on_simulate
     * (and the DevUI Graphics tab) to change graphics settings; the actual
     * GPU/SDL work happens on the main thread with no sim in flight.
     * Last request before a fence wins.
     */
    void request_graphics_settings(const GraphicsSettings& settings);

    // ========== Dev overlay ==========

    /** Create the developer overlay and, if the renderer is up, initialize it
     *  bound to the engine subsystems (scene renderer, camera, graphics
     *  settings, fps). On success this claims the SceneRenderer's single
     *  post-UI callback slot to draw the overlay — hosts that call
     *  init_dev_ui() must not set their own post-UI callback. Call after
     *  init_renderer(). Returns false when the dev UI is compiled out
     *  (BRACKEN_ENABLE_DEV_UI=OFF) or backend init fails; dev_ui() remains
     *  safe to use either way. Panel data safety in the pipelined loop is
     *  documented on dev::DevUI (short version: render-side data or
     *  on_main_frame snapshots only). */
    bool init_dev_ui();

    /** Access the dev overlay (toggle visibility, add host panels).
     *  Valid after init_dev_ui() — including when it returned false. */
    dev::DevUI& dev_ui();

    /** The persisted graphics settings. Tier: owned by the host's frame code
     *  — under run(), mutate it from on_simulate (and apply via
     *  request_graphics_settings) or from on_main_frame, but not both
     *  concurrently. The engine itself only touches it at init/shutdown and
     *  the frame fence. */
    GraphicsSettings& graphics_settings() { return graphics_settings_; }
    const GraphicsSettings& graphics_settings() const { return graphics_settings_; }

    /** Get the platform-appropriate settings file path. */
    std::string settings_file_path() const;

    struct DisplayMode {
        int w, h;
    };
    std::vector<DisplayMode> available_resolutions() const;

    /** Model manager. Loading/mutation is main-thread tier (init paths, the
     *  hot-reload fence); const lookups are worker-safe. */
    ModelManager& models();

    /** Terrain height sample — worker-safe (const bilinear reads of the
     *  immutable heightmap; set_heightmap only runs at init/fence time), so
     *  sim code may call it freely. */
    float get_terrain_height(float x, float z);

    /** Window client size. Safe from sim code: values change only during the
     *  event pump / fence, never while a sim is in flight. 0 without a
     *  renderer. */
    int screen_width() const;
    int screen_height() const;

    scene::SceneRenderer& scene_renderer() {
        BRACKEN_CONTRACT(scene_renderer_ != nullptr, "scene_renderer() requires init_renderer() first");
        return *scene_renderer_;
    }

    /** The particle system. Tier: sim-stage owned — the engine drives it
     *  from the sim-job tail (spawns from the builder, update, packet
     *  snapshot); hosts normally only spawn via SimFrame::frame()'s
     *  spawn_effect. Touch it only from on_simulate / the fence / init. */
    systems::EffectSystem& effect_system() {
        BRACKEN_CONTRACT(effect_system_ != nullptr, "effect_system() requires init_engine() first");
        return *effect_system_;
    }

    void set_collect_render_stats(bool enabled);

    /**
     * The last rendered frame's counters, or nullopt when collection is off
     * (set_collect_render_stats(false), the default).
     *
     * The optional is the point: the old signature returned a reference to an
     * all-zero struct when collection was off, which is indistinguishable
     * from a genuinely empty frame — "0 draw calls" read as a renderer bug
     * rather than as "you never asked".
     */
    std::optional<RenderStats> render_stats() const;

    /**
     * What the last rendered frame threw away: unknown model handles, skinned
     * draws with no skeleton, over-budget bone palettes, effect emitters
     * naming missing models, lights past MAX_LIGHTS, cluster bins that
     * truncated, and whether the frame had 3D content but no camera.
     *
     * Always accumulated — unlike render_stats() this does not require
     * set_collect_render_stats(true), because a silently dropped draw is
     * never something a host opted out of being told about. Main-thread /
     * fence read (the renderer writes it during render_frame).
     */
    const FrameDiagnostics& frame_diagnostics() const;
    std::string gpu_driver_name() const;

    /** Pin the simulation timestep, decoupling simulated time from wall clock.
     *
     *  Normally `dt` is measured from the wall clock, so N rendered frames
     *  cover a different amount of simulated time on every run — the machine's
     *  load decides how far the world advances. That makes a fixed-frame-count
     *  run (an integration test, a recorded demo, a golden-image capture)
     *  irreproducible: a slower run simulates *further*, not just slower.
     *
     *  With a value set, every frame advances exactly `seconds` regardless of
     *  how long it actually took to render, so frame N is always at simulated
     *  time `N * seconds`. Rendering is unaffected — this changes what the sim
     *  is told, not how fast the loop runs.
     *
     *  Pass `std::nullopt` to restore wall-clock timing (the default).
     *  Tier: main-thread (set it in `on_init`, not mid-flight). */
    void set_fixed_timestep(std::optional<float> seconds);

    /** The pinned timestep, or `std::nullopt` when timing is wall-clock. */
    [[nodiscard]] std::optional<float> fixed_timestep() const { return fixed_timestep_; }

    /** Accumulate a running fingerprint of every frame this run describes.
     *
     *  Each sealed packet is hashed (see scene/packet_digest.hpp) after the
     *  sim job joins and before it renders, and folded into a running value.
     *  Two runs that describe the same frames produce the same number, so it
     *  works as an exact regression gate — unlike render statistics or a
     *  screenshot, which sit downstream of the GPU and move for driver
     *  reasons that have nothing to do with a code change.
     *
     *  Pair it with set_fixed_timestep(): without a pinned timestep the frames
     *  themselves differ run-to-run and the digest is expected to differ too.
     *
     *  Off by default — it walks every command of every frame, which a
     *  shipping game should not pay for. Tier: main-thread. */
    void set_packet_digest_enabled(bool enabled) { packet_digest_enabled_ = enabled; }

    /** The accumulated frame fingerprint, or 0 if digesting was never enabled.
     *  Tier: main-thread, or sim-stage/fence (never written mid-render). */
    [[nodiscard]] uint64_t packet_digest() const { return packet_digest_accum_; }

    /** Capture the next rendered frame (UI included) to a PNG. The capturing
     *  frame blocks on the GPU — integration tests and debug only. Tier:
     *  sim-stage or main thread — the request is deferred and forwarded to
     *  the renderer at the frame fence (a request from sim frame i captures
     *  frame i's rendering). */
    void request_screenshot(const std::string& path);

    // ========== Camera facade ==========

    /** The host camera rig. Tier: sim-stage under run() (update it in
     *  on_simulate); main-thread reads (DevUI Camera tab) tear at worst —
     *  see dev::DevUI's panel-data-safety note. */
    systems::CameraController& camera();
    const systems::CameraController& camera() const;

private:
    friend class MainFrame; // fence-scoped passthroughs (fps/window/screenshot)

    // ---- Pipeline steps (see the class comment's schedule) ----

    /** Steps 3: pump SDL events into on_event/DevUI/InputHandler. May clear
     *  running_ on SDL_EVENT_QUIT. Main thread, no sim in flight. */
    void pump_events();

    /** Step 4: compute dt, stamp packet timing, take the fill slot and
     *  submit the sim job. Main thread, no sim in flight on entry. */
    void kick_sim();

    /** Step 1: join the in-flight sim job (helping the pool). Returns the
     *  sealed packet, or nullptr when nothing was in flight. On a sim
     *  exception: fatal-log, abandon the partial packet, rethrow. */
    scene::FramePacket* join_sim();

    /** Step 2: the frame fence — FileWatcher poll, pending settings/
     *  screenshot/camera-forward application, FPS accounting, on_main_frame. */
    void frame_fence(uint64_t joined_frame_index);

    /** The sim job body (pool thread): SimFrame → on_simulate → engine
     *  sim-stage tail (effects, clocks, engine packet section) → seal. */
    void run_sim_job();

    InputHandler input_;

    // running_ — loop-continue flag. Protocol: relaxed stores in quit() (any
    // thread, including the in-flight sim job) and the pump's QUIT handling ↔
    // relaxed loads at the loop head. Publishes nothing — it is a pure flag;
    // all cross-thread data movement rides the sim kick/join edges.
    std::atomic<bool> running_{false};

    uint64_t last_frame_time_ = 0; // nanoseconds (SDL_GetTicksNS) at last kick
    // When set, kick_sim() reports this dt instead of the wall-clock delta,
    // making simulated time a pure function of frame index. See
    // set_fixed_timestep(). Main-thread only; read at kick.
    std::optional<float> fixed_timestep_;

    // Rolling frame fingerprint. Zero until enabled, then folded once per
    // joined packet — see set_packet_digest_enabled().
    bool packet_digest_enabled_ = false;
    uint64_t packet_digest_accum_ = 0;
    float fps_ = 0.0f;
    // Phase timings (EMA-smoothed, alpha 0.2 — see sim_ms()). Written at the
    // fence on the main thread with no sim in flight; read from anywhere.
    float sim_ms_ = 0.0f;
    float render_ms_ = 0.0f;
    float fence_ms_ = 0.0f;
    float present_wait_ms_ = 0.0f;
    // Raw last-frame samples handed to the smoother at the next fence.
    float last_sim_ms_ = 0.0f;
    float last_render_ms_ = 0.0f;
    float last_present_wait_ms_ = 0.0f;
    int frame_count_ = 0;
    uint64_t fps_timer_ = 0; // nanoseconds (SDL_GetTicksNS)

    // ---- Sim pipeline state ----
    // Owned by the coordinator (main thread). The builder/packet pointers and
    // the accumulators below are handed to the sim job at kick and back at
    // join; ordering rides the kick(submit publish)/join(TaskHandle wait)
    // edges — single writer per phase, no atomics (see each note).
    std::unique_ptr<scene::FrameBuilder> sim_builder_; // outlives the job; reset after join
    scene::FramePacket* sim_packet_ = nullptr;         // packet being filled / sealed at join
    core::jobs::TaskHandle sim_task_;
    bool sim_in_flight_ = false; // main-thread bookkeeping only

    /** FrameBuilder lane budget per sim frame. Fixed (never derived from the
     *  worker count — §6.2 determinism: lane assignment must be stable):
     *  lane 0 serves the builder's single-threaded conveniences; hosts that
     *  fan out emission assign lanes 0..k_sim_lanes-1 by JOB index. Lanes
     *  are cheap (one cache-line-padded header each in the packet arena). */
    static constexpr uint32_t k_sim_lanes = 16;

    // Written at kick (main thread, no sim in flight), read by the job.
    InputSnapshot sim_input_;
    float sim_dt_ = 0.0f;
    double sim_time_ = 0.0;          // accumulated dt
    uint64_t sim_frame_index_ = 0;   // next frame index to kick
    uint64_t sim_frame_current_ = 0; // index of the in-flight / last-kicked frame
    // Written INSIDE the sim job (sim-stage tail), read/reset only at
    // fence/join time: sim-advanced clocks + the last host-staged
    // environment (its cloud_speed scales the sky clock; complete type via
    // sim_frame.hpp → frame_packet.hpp → render/environment.hpp).
    float sim_scene_time_ = 0.0f;
    float sim_sky_time_ = 0.0f;
    render::EnvironmentState sim_env_;
    SimFeedback sim_feedback_;

    // Deferred fence requests (see request_graphics_settings /
    // request_screenshot), consumed at the fence. The settings request has
    // one slot per legal caller phase because they can overlap (sim job vs
    // the DevUI apply callback on the main thread mid-render); each slot is
    // single-writer — sim-slot ordering rides the kick/join edges.
    GraphicsSettings pending_graphics_settings_;      // sim-phase writer
    bool has_pending_graphics_settings_ = false;      // sim-phase writer
    GraphicsSettings pending_graphics_settings_main_; // main-thread writer
    bool has_pending_graphics_settings_main_ = false; // main-thread writer
    std::string pending_screenshot_path_;             // sim-phase or fence writer

    // Owned engine subsystems (opaque to subclasses)
    std::unique_ptr<render::RenderContext> context_;
    // Frame-packet ring (lazily created by frame_packets(); depth 2 = one
    // packet rendering while the next is being simulated).
    std::unique_ptr<scene::PacketRing> packet_ring_;
    std::unique_ptr<scene::SceneRenderer> scene_renderer_;
    std::unique_ptr<systems::CameraSystem> camera_;
    std::unique_ptr<systems::EffectSystem> effect_system_; // sim-stage owned (see effect_system())
    std::unique_ptr<dev::DevUI> dev_ui_; // created by init_dev_ui(); destroyed before the renderer

    GraphicsSettings graphics_settings_;

    // Last settings actually pushed to the setters, for diffing in
    // apply_all_graphics_settings().
    GraphicsSettings last_applied_settings_;
    bool has_applied_settings_ = false;
};

} // namespace bracken
