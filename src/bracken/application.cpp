#include "application.hpp"
#include "bracken/core/assert.hpp"

#include <chrono>
#include "bracken/core/asset/file_watcher.hpp"
#include "bracken/core/jobs/job_stage.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/dev/dev_ui.hpp"
#include "bracken/effects/effect_system.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/render/render_context.hpp"
#include "bracken/scene/frame_packet_ring.hpp"
#include "bracken/scene/packet_digest.hpp"
#include "bracken/scene/scene_renderer.hpp"
#include "bracken/sim_frame.hpp"
#include "bracken/systems/camera_system.hpp"
#include "SDL3/SDL_error.h"
#include "SDL3/SDL_filesystem.h"
#include "SDL3/SDL_init.h"
#include "SDL3/SDL_timer.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

namespace bracken {

using namespace bracken::render;
using namespace bracken::scene;
using namespace bracken::systems;

Application::Application() : effect_system_(std::make_unique<systems::EffectSystem>()) {}

Application::~Application() = default;

bool Application::init_engine() {
    // Record the engine main thread for BRACKEN_ASSERT_MAIN_THREAD() checks
    // (FileWatcher entry points, swapchain acquire, render_frame).
    core::bind_main_thread();

    // The job-system worker pool is always-on engine infrastructure (see
    // docs/multithreading_plan.md §4.1): it comes up before every other
    // subsystem so any of them may submit jobs, and goes down first at
    // shutdown (top of shutdown_renderer()) so no job outlives the
    // subsystems it might touch. BRACKEN_WORKERS overrides the worker
    // count: unset/0 = hardware_concurrency()-1 clamped >= 1;
    // an explicit positive value is used as-is (BRACKEN_WORKERS=1 is the
    // single-worker debug mode — same code path, one worker).
    {
        unsigned workers = 0; // 0 = JobSystem's default (hw-1, clamped >= 1)
        // getenv is single-threaded at engine boot, before any worker thread is spawned.
        // NOLINTNEXTLINE(concurrency-mt-unsafe)
        if (const char* env = std::getenv("BRACKEN_WORKERS"); env && env[0] != '\0') {
            char* end = nullptr;
            const long parsed = std::strtol(env, &end, 10);
            if (parsed > 0) {
                workers = static_cast<unsigned>(parsed);
            } else if (parsed < 0 || end == env) { // negative or non-numeric
                BRACKEN_LOG_WARN("app", "Ignoring invalid BRACKEN_WORKERS='{}'; using default", env);
            }
        }
        core::jobs::JobSystem::instance().init(workers);
        BRACKEN_LOG_INFO("app", "Job system running with {} worker(s)",
                         core::jobs::JobSystem::instance().worker_count());
    }

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        BRACKEN_LOG_FATAL("app", "Failed to initialize SDL: {}", SDL_GetError());
        return false;
    }

    // getenv is single-threaded at engine boot, before any worker thread is spawned.
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    if (const char* env = std::getenv("BRACKEN_HOT_RELOAD"); env && env[0] == '1') {
        if (core::asset::FileWatcher::instance().init()) {
            BRACKEN_LOG_INFO("app", "Hot-reload enabled (BRACKEN_HOT_RELOAD=1)");
        }
    }
    return true;
}

bool Application::start() {
    BRACKEN_ASSERT_MAIN_THREAD();
    started_ = on_init();
    return started_;
}

void Application::stop() {
    BRACKEN_ASSERT_MAIN_THREAD();
    if (!started_) {
        return; // never came up (or already taken down) — nothing to unwind
    }
    started_ = false;
    on_shutdown();
}

void Application::pump_events() {
    BRACKEN_PROFILE_ZONE("App::poll_events");
    input_.reset_camera_deltas();
    input_.clear_menu_inputs();
    input_.clear_key_edges();
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_EVENT_QUIT) {
            running_.store(false, std::memory_order_relaxed);
            return;
        }
        // Dev overlay sees events first (ImGui tracks state even
        // while hidden; consumes only when visible + wants input).
        if (dev_ui_ && dev_ui_->process_event(event)) {
            continue;
        }
        // Let subclass consume the event
        if (!on_event(event)) {
            input_.process_event(event);
        }
    }
    // Polled movement keys bypass event consumption — mute them while
    // the dev overlay owns the keyboard (e.g. typing in a panel).
    input_.set_keyboard_captured(dev_ui_ && dev_ui_->wants_keyboard());
    input_.post_process_events();
}

void Application::kick_sim() {
    BRACKEN_ASSERT(!sim_in_flight_, "kick_sim with a sim job already in flight");

    // Frame timing is computed AT KICK and carried in the packet.
    // Nanosecond ticks: whole-millisecond timing quantizes dt to 0/1 ms steps
    // at high frame rates, juddering animation and camera motion.
    const uint64_t now = SDL_GetTicksNS();
    float dt = static_cast<float>(static_cast<double>(now - last_frame_time_) /
                                  static_cast<double>(SDL_NS_PER_SECOND));
    last_frame_time_ = now;
    dt = std::min(dt, 0.1f); // clamp huge jumps (debugger stalls, window drags)

    // A pinned timestep replaces the measured delta outright, so simulated
    // time becomes a pure function of frame index and a fixed-frame-count run
    // is reproducible. last_frame_time_ is still updated above: restoring
    // wall-clock timing mid-run must not hand the next frame a huge delta.
    if (fixed_timestep_) {
        dt = *fixed_timestep_;
    }

    sim_dt_ = dt;
    sim_time_ += dt;
    sim_frame_current_ = sim_frame_index_++;

    // Take the input snapshot for this sim frame (main thread; the sim job
    // reads it via the submit publish edge), plus the viewport it may not
    // read for itself (see sim_screen_width_).
    sim_input_ = input_.snapshot();
    sim_screen_width_ = screen_width();
    sim_screen_height_ = screen_height();
    sim_feedback_.has_camera_forward = false;

    // Acquire the fill slot + builder (Retired → Filling) and stamp timing.
    FramePacket& packet = frame_packets().acquire_for_fill();
    sim_builder_ = std::make_unique<FrameBuilder>(packet, k_sim_lanes);
    sim_builder_->set_frame_timing(dt, sim_time_, sim_frame_current_);
    sim_packet_ = &packet;

    sim_task_ = core::jobs::JobSystem::instance().submit([this] { run_sim_job(); });
    sim_in_flight_ = true;
}

void Application::run_sim_job() {
    BRACKEN_PROFILE_ZONE("App::simulate");
    const auto sim_begin = std::chrono::steady_clock::now();
    // Root of the sim stage. Every job submitted from here (host parallel_for
    // fan-outs, the EffectSystem update, Jolt's step jobs) inherits it through
    // JobNode::stage, so the tier assertions hold on any worker that ends up
    // running them — and stop holding the moment this scope exits.
    const core::jobs::StageScope sim_stage(core::jobs::Stage::Sim);
    FrameBuilder& fb = *sim_builder_;

    {
        BRACKEN_PROFILE_ZONE("App::on_simulate");
        SimFrame sim(fb, sim_input_, sim_dt_, sim_time_, sim_frame_current_, &sim_feedback_, sim_screen_width_,
                     sim_screen_height_);
        on_simulate(sim);
    }

    // ---- Engine sim-stage tail (docs/multithreading_plan.md §4.2/§4.3) ----
    // Camera. Application owns the rig, so it drives it: the host aims the
    // camera during on_simulate and the engine does the rest, HERE — after
    // the frame is fully staged, so this frame's target move and this frame's
    // shakes are both in. The viewport comes from the kick-time snapshot, not
    // screen_width(): RenderContext is main-thread tier and may be
    // reconfiguring the swapchain while this job runs. A host with its own rig
    // opts out via set_camera_driven_by_engine(false) and owes the frame all
    // four calls itself.
    if (camera_driven_by_engine_ && camera_) {
        camera_->set_screen_size(sim_screen_width_, sim_screen_height_);
        camera_->update(sim_dt_);
        fb.set_camera(camera_->get_camera_state());
        if (!sim_feedback_.has_camera_forward) {
            // Only when the host did not override it (SimFrame::set_camera_forward).
            const glm::vec3 forward = camera_->get_forward();
            sim_feedback_.has_camera_forward = true;
            sim_feedback_.camera_forward = {forward.x, forward.z};
        }
    }

    // Apply the frame's queued effect spawns to the sim-owned EffectSystem in
    // deterministic (lane, emission) order — the drain the renderer used to
    // do, moved to the sim side.
    fb.for_each_effect_spawn([this](const EffectSpawnCommand& spawn) {
        // Consume side of the packet's host-lifetime exception. The emitter
        // instances the EffectSystem builds from this pointer outlive the
        // packet, so a bad one surfaces later, on the render thread, far from
        // the frame that emitted it. Loud here instead.
        BRACKEN_CONTRACT(spawn.definition != nullptr, "FramePacket effect spawn with a null EffectDefinition");
        effect_system_->spawn_effect(spawn.definition, spawn.position, spawn.direction);
    });

    // Particle update on the pool (nested fork-join from inside this job is
    // supported by JobSystem::wait's helping). The terrain sampler is
    // worker-safe: const bilinear reads of the immutable heightmap.
    auto& js = core::jobs::JobSystem::instance();
    if (scene_renderer_) {
        effect_system_->update(js, sim_dt_,
                               [this](float x, float z) { return scene_renderer_->get_terrain_height(x, z); });
    } else {
        effect_system_->update(js, sim_dt_);
    }

    // Effect-instance snapshot: the renderer draws THIS, never the live
    // EffectSystem (which the next sim frame mutates concurrently).
    effect_system_->snapshot_into(fb);

    // Scene/sky clock advance (moved out of the renderer). The sky clock is
    // a cloud scroll PHASE: dt scaled by the environment's cloud_speed — use
    // the environment the host staged this frame (or the last one it ever
    // staged; defaults otherwise).
    if (fb.has_staged_environment()) {
        sim_env_ = fb.staged_environment();
    }
    sim_scene_time_ += sim_dt_;
    sim_sky_time_ += sim_dt_ * sim_env_.cloud_speed;
    fb.set_engine_frame(sim_scene_time_, sim_sky_time_, static_cast<uint32_t>(effect_system_->effect_count()));

    // Filling → Sealed. The TaskHandle join publishes the sealed packet to
    // the main thread — no further handoff synchronization needed.
    fb.seal();

    // Written by the job, read at the fence after the join publishes it — the
    // same single-writer-per-phase discipline as the rest of the sim state.
    last_sim_ms_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - sim_begin).count();
}

scene::FramePacket* Application::join_sim() {
    if (!sim_in_flight_) {
        return nullptr;
    }
    sim_in_flight_ = false;
    FramePacket* frame = sim_packet_;
    sim_packet_ = nullptr;
    try {
        BRACKEN_PROFILE_ZONE("App::join_sim");
        // Helping wait: the main thread executes pool jobs (including this
        // frame's own sim fan-out) instead of idling.
        core::jobs::JobSystem::instance().wait(sim_task_);
    } catch (...) {
        // Sim-job error policy: a host exception out of on_simulate must not
        // vanish into the pool — fatal-log and rethrow on the main thread.
        BRACKEN_LOG_FATAL("app", "on_simulate threw; abandoning frame {} and rethrowing on the main thread",
                          sim_frame_current_);
        sim_task_ = {};
        sim_builder_.reset();
        if (frame != nullptr) {
            frame->reset(); // abandon the partial fill (any state → Retired)
        }
        throw;
    }
    sim_task_ = {};
    sim_builder_.reset(); // spent after seal
    return frame;         // Sealed
}

void Application::frame_fence(uint64_t joined_frame_index) {
    BRACKEN_PROFILE_ZONE("App::frame_fence");
    const auto fence_begin = std::chrono::steady_clock::now();
    // No sim in flight here — the pipeline's quiescent point.
    BRACKEN_ASSERT(!sim_in_flight_, "frame_fence with a sim job in flight");

    // Hot-reload applies at the fence: no packet referencing the old
    // resource is being built or rendered.
    if (core::asset::FileWatcher::instance().is_initialized()) {
        core::asset::FileWatcher::instance().poll_main_thread();
    }

    // Deferred graphics-settings application (request_graphics_settings; the
    // main-thread slot — DevUI — wins over the sim slot when both are set).
    if (has_pending_graphics_settings_ || has_pending_graphics_settings_main_) {
        const GraphicsSettings& s =
            has_pending_graphics_settings_main_ ? pending_graphics_settings_main_ : pending_graphics_settings_;
        apply_all_graphics_settings(s);
        has_pending_graphics_settings_ = false;
        has_pending_graphics_settings_main_ = false;
    }

    // Camera-forward feedback from the joined sim frame: shapes the movement
    // basis of the NEXT input snapshot (same one-frame latency as the old
    // sequential loop).
    if (sim_feedback_.has_camera_forward) {
        input_.set_camera_forward(sim_feedback_.camera_forward.x, sim_feedback_.camera_forward.y);
        sim_feedback_.has_camera_forward = false;
    }

    // FPS accounting lives at the fence so fps_ is never written while a sim
    // job (which may read fps()) is in flight.
    const uint64_t now = SDL_GetTicksNS();
    frame_count_++;
    if (now - fps_timer_ >= static_cast<uint64_t>(SDL_NS_PER_SECOND)) {
        fps_ = static_cast<float>(frame_count_);
        frame_count_ = 0;
        fps_timer_ = now;
    }

    // Host main-thread hook — the only safe place to read live game state
    // from the main thread (see application.hpp).
    {
        BRACKEN_PROFILE_ZONE("App::on_main_frame");
        MainFrame mf(*this, joined_frame_index);
        on_main_frame(mf);
    }

    // Forward the deferred screenshot request (from sim or on_main_frame)
    // so the frame about to render is the one captured.
    if (!pending_screenshot_path_.empty() && scene_renderer_) {
        scene_renderer_->request_screenshot(std::move(pending_screenshot_path_));
        pending_screenshot_path_.clear();
    }

    // Phase timings settle here, at the pipeline's quiescent point: the sim
    // just joined (so last_sim_ms_ is published), the previous frame's render
    // is long finished, and nothing may read these while a sim runs.
    // Exponentially smoothed — a raw per-frame number at 100+ fps flickers
    // too fast to read and one hitching frame would own the readout.
    constexpr float k_alpha = 0.2f;
    const float fence_ms =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - fence_begin).count();
    sim_ms_ += k_alpha * (last_sim_ms_ - sim_ms_);
    render_ms_ += k_alpha * (last_render_ms_ - render_ms_);
    present_wait_ms_ += k_alpha * (last_present_wait_ms_ - present_wait_ms_);
    fence_ms_ += k_alpha * (fence_ms - fence_ms_);
}

void Application::run() {
    BRACKEN_ASSERT_MAIN_THREAD();
    running_.store(true, std::memory_order_relaxed);
    last_frame_time_ = SDL_GetTicksNS();
    fps_timer_ = last_frame_time_;

    // Bootstrap: pump once so frame 0 sims real input, then kick sim job 0.
    // No overlap for frame 0 — the loop joins it immediately.
    pump_events();
    if (running_.load(std::memory_order_relaxed)) {
        kick_sim();
    }

    // Steady state, iteration i (see application.hpp for the schedule):
    // join sim i → fence → pump → kick sim i+1 → render i → retire i.
    // Simulate(i+1) overlaps Render(i); pipeline depth 2.
    while (running_.load(std::memory_order_relaxed)) {
        FramePacket* frame = join_sim(); // (1) — rethrows sim exceptions
        if (frame == nullptr) {
            break; // defensive: nothing in flight
        }
        const uint64_t frame_index = frame->frame_index();

        // Digest the packet here: it is sealed, and the renderer has not yet
        // written the engine-owned fields (light shadow slots) that would make
        // the value depend on whether the frame had been drawn.
        if (packet_digest_enabled_) {
            packet_digest_accum_ = packet_digest_accum_ * 1099511628211ULL ^ scene::packet_digest(*frame);
        }

        frame_fence(frame_index); // (2)

        pump_events(); // (3)
        if (!running_.load(std::memory_order_relaxed)) {
            // Quit during the pump (or from sim i): frame i is sealed but
            // will never render — retire it and leave with no sim in flight.
            frame_packets().retire(*frame);
            break;
        }

        kick_sim(); // (4) — Simulate(i+1) now runs on the pool...

        // (5) ...while the main thread renders frame i (Prepare fan-out +
        // Record + present, unchanged from Phase 3).
        if (scene_renderer_) {
            BRACKEN_PROFILE_ZONE("App::render");
            const auto render_begin = std::chrono::steady_clock::now();
            if (context_) {
                context_->device().reset_present_wait();
            }
            render_frame(*frame);
            last_render_ms_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() -
                                                                       render_begin)
                                  .count();
            // present_wait is a SUBSET of render_ms: the part of it the CPU
            // spent blocked rather than working.
            last_present_wait_ms_ = context_ ? context_->device().present_wait_ms() : 0.0f;
        }

        frame_packets().retire(*frame); // (6)

        BRACKEN_PROFILE_FRAME();
    }

    // Shutdown: join any in-flight sim before teardown — quit() may have
    // been observed with a freshly kicked job still running, and nothing
    // (renderer, effect system, packets) may be torn down under it. A sim
    // exception here still rethrows out of run() (fatal-logged in join_sim).
    if (sim_in_flight_) {
        FramePacket* frame = join_sim();
        if (frame != nullptr) {
            frame_packets().retire(*frame);
        }
    }
}

void Application::shutdown_engine() {
    core::asset::FileWatcher::instance().shutdown();
    // Safety net for hosts that never brought up the renderer: normally the
    // pool is already down (top of shutdown_renderer(), before GPU teardown);
    // JobSystem::shutdown() is idempotent.
    core::jobs::JobSystem::instance().shutdown();
    SDL_Quit();
}

// ========== Rendering facade ==========

bool Application::init_renderer(int width, int height, const std::string& title, float world_width,
                                float world_height) {
    context_ = std::make_unique<RenderContext>();
    if (!context_->init(width, height, title)) {
        return false;
    }
    context_->query_display_modes();

    scene_renderer_ = std::make_unique<SceneRenderer>();
    if (!scene_renderer_->init(*context_, world_width, world_height)) {
        return false;
    }

    if (auto* pr = scene_renderer_->pipeline_registry()) {
        pr->enable_hot_reload(core::asset::FileWatcher::instance());
    }
    scene_renderer_->models().enable_hot_reload(core::asset::FileWatcher::instance());

    camera_ = std::make_unique<CameraSystem>();

    // Auto-load settings from disk
    auto path = settings_file_path();
    if (!path.empty() && !graphics_settings_.load(path)) {
        // Normal on first run: no settings file yet, defaults apply.
        BRACKEN_LOG_INFO("app", "No graphics settings at '{}'; using defaults", path);
    }
    apply_all_graphics_settings(graphics_settings_);

    return true;
}

void Application::shutdown_renderer() {
    // Stop the worker pool before any GPU teardown: joining the workers here
    // guarantees no in-flight or queued job can touch a subsystem (renderer,
    // device, model manager) that is about to be destroyed.
    core::jobs::JobSystem::instance().shutdown();

    // Auto-save settings on shutdown
    auto path = settings_file_path();
    if (!path.empty()) {
        if (!graphics_settings_.save(path)) {
            BRACKEN_LOG_ERROR("settings", "Failed to save settings to {}", path);
        }
    }

    // Dev overlay owns GPU resources — it must die before the device.
    if (dev_ui_) {
        dev_ui_->shutdown();
        dev_ui_.reset();
    }

    camera_.reset();
    if (scene_renderer_) {
        scene_renderer_->shutdown();
    }
    if (context_) {
        context_->shutdown();
    }
    scene_renderer_.reset();
    context_.reset();
}

void Application::render_frame(const FramePacket& frame) {
    // The rendering facade below dereferences scene_renderer_/context_
    // unconditionally: before init_renderer() every one of these was a null
    // dereference, and the rule was inferable only by reading application.cpp
    // (screen_width() null-checks, its neighbours did not). Contracts, not
    // asserts — a host calls these from ITS release build.
    BRACKEN_CONTRACT(scene_renderer_ != nullptr, "render_frame() requires init_renderer() first");
    scene_renderer_->render_frame(frame);
}

scene::PacketRing& Application::frame_packets() {
    // Lazily created so hosts that own their own ring (or never render)
    // don't pay for idle packet arenas. Depth 2 (PacketRing's default): the
    // pipelined loop renders one slot while the sim job fills the other.
    if (!packet_ring_) {
        packet_ring_ = std::make_unique<scene::PacketRing>();
    }
    return *packet_ring_;
}

void Application::set_heightmap(const Heightmap& heightmap) {
    BRACKEN_CONTRACT(scene_renderer_ != nullptr, "set_heightmap() requires init_renderer() first");
    scene_renderer_->set_heightmap(heightmap);
}

void Application::set_graphics_settings(const GraphicsSettings& settings) {
    BRACKEN_CONTRACT(scene_renderer_ != nullptr, "set_graphics_settings() requires init_renderer() first");
    scene_renderer_->set_graphics_settings(settings);
}

void Application::set_anisotropic_filter(AnisotropicFilter level) {
    BRACKEN_CONTRACT(scene_renderer_ != nullptr, "set_anisotropic_filter() requires init_renderer() first");
    scene_renderer_->set_anisotropic_filter(static_cast<int>(level));
}

void Application::set_vsync_mode(VSyncMode mode) {
    BRACKEN_CONTRACT(scene_renderer_ != nullptr, "set_vsync_mode() requires init_renderer() first");
    scene_renderer_->set_vsync_mode(static_cast<int>(mode));
}

int Application::max_vsync_mode() const {
    BRACKEN_CONTRACT(scene_renderer_ != nullptr, "max_vsync_mode() requires init_renderer() first");
    return scene_renderer_->max_vsync_mode();
}

void Application::set_window_mode(WindowMode window_mode, int resolution_index) {
    BRACKEN_CONTRACT(context_ != nullptr, "set_window_mode() requires init_renderer() first");
    context_->set_window_mode(static_cast<int>(window_mode), resolution_index);
}

std::vector<Application::DisplayMode> Application::available_resolutions() const {
    BRACKEN_CONTRACT(context_ != nullptr, "available_resolutions() requires init_renderer() first");
    std::vector<DisplayMode> result;
    for (const auto& m : context_->available_resolutions()) {
        result.push_back({m.w, m.h});
    }
    return result;
}

ModelManager& Application::models() {
    BRACKEN_CONTRACT(scene_renderer_ != nullptr, "models() requires init_renderer() first");
    return scene_renderer_->models();
}

float Application::get_terrain_height(float x, float z) {
    BRACKEN_CONTRACT(scene_renderer_ != nullptr, "get_terrain_height() requires init_renderer() first");
    return scene_renderer_->get_terrain_height(x, z);
}

int Application::screen_width() const {
    return context_ ? context_->width() : 0;
}

int Application::screen_height() const {
    return context_ ? context_->height() : 0;
}

void Application::set_fixed_timestep(std::optional<float> seconds) {
    BRACKEN_ASSERT(!seconds || *seconds > 0.0f, "fixed timestep must be positive");
    fixed_timestep_ = seconds;
}

void Application::set_collect_render_stats(bool enabled) {
    BRACKEN_CONTRACT(scene_renderer_ != nullptr, "set_collect_render_stats() requires init_renderer() first");
    scene_renderer_->set_collect_stats(enabled);
}

std::optional<RenderStats> Application::render_stats() const {
    BRACKEN_CONTRACT(scene_renderer_ != nullptr, "render_stats() requires init_renderer() first");
    if (!scene_renderer_->collect_stats()) {
        return std::nullopt; // "you never asked", not "the frame was empty"
    }
    return scene_renderer_->render_stats();
}

const FrameDiagnostics& Application::frame_diagnostics() const {
    BRACKEN_CONTRACT(scene_renderer_ != nullptr, "frame_diagnostics() requires init_renderer() first");
    return scene_renderer_->frame_diagnostics();
}

std::string Application::gpu_driver_name() const {
    BRACKEN_CONTRACT(context_ != nullptr, "gpu_driver_name() requires init_renderer() first");
    return context_->device().driver_name();
}

void Application::request_screenshot(const std::string& path) {
    // Deferred: forwarded to the renderer at the frame fence, so a request
    // from sim frame i captures frame i's rendering — and the sim job never
    // touches the renderer's screenshot state while a frame is recording.
    pending_screenshot_path_ = path;
}

void Application::request_graphics_settings(const GraphicsSettings& settings) {
    // Deferred: applied by frame_fence() on the main thread with no sim in
    // flight (the GPU/SDL work in apply_all_graphics_settings is main-thread
    // tier). Two slots because the two legal callers can be CONCURRENT: the
    // sim job (host frame code) and the main thread (the DevUI Graphics tab's
    // apply callback fires during rendering, while a sim is in flight). Each
    // slot is single-writer — sim-slot ordering rides the sim-job kick/join
    // edges, the main slot never leaves the main thread — so no atomics.
    // At the fence the main-thread slot wins (the DevUI edit is the more
    // deliberate action).
    if (core::is_main_thread()) {
        pending_graphics_settings_main_ = settings;
        has_pending_graphics_settings_main_ = true;
    } else {
        pending_graphics_settings_ = settings;
        has_pending_graphics_settings_ = true;
    }
}

GraphicsSettingsEdit::GraphicsSettingsEdit(GraphicsSettingsEdit&& other) noexcept
    : app_(other.app_), original_(other.original_), edited_(other.edited_) {
    other.app_ = nullptr; // disarmed: exactly one handle may commit
}

GraphicsSettingsEdit::~GraphicsSettingsEdit() {
    if (app_ != nullptr && !(edited_ == original_)) {
        app_->request_graphics_settings(edited_);
    }
}

void Application::apply_all_graphics_settings(const GraphicsSettings& settings) {
    graphics_settings_ = settings;
    set_graphics_settings(settings);
    // Only invoke the expensive setters (sampler rebuilds, swapchain reconfig,
    // fullscreen transitions) when their specific inputs actually changed.
    if (!has_applied_settings_ || settings.anisotropic_filter != last_applied_settings_.anisotropic_filter) {
        set_anisotropic_filter(settings.anisotropic_filter);
    }
    if (!has_applied_settings_ || settings.vsync_mode != last_applied_settings_.vsync_mode) {
        set_vsync_mode(settings.vsync_mode);
    }
    if (!has_applied_settings_ || settings.window_mode != last_applied_settings_.window_mode ||
        settings.resolution_index != last_applied_settings_.resolution_index) {
        set_window_mode(settings.window_mode, settings.resolution_index);
    }
    last_applied_settings_ = settings;
    has_applied_settings_ = true;
}

// ========== Dev overlay ==========

bool Application::init_dev_ui() {
    // Always construct, even when the renderer is down or the overlay is
    // compiled out: dev_ui() and process_event()/render() stay safe to call.
    dev_ui_ = std::make_unique<dev::DevUI>();
    if (!context_ || !scene_renderer_ || !camera_) {
        return false;
    }

    dev::DevUI::Bindings bindings{
        scene_renderer_.get(),
        &camera(),
        &graphics_settings_,
        // The overlay draws DURING rendering, while the next sim runs —
        // route its apply through the deferred request so the GPU/SDL work
        // lands at the frame fence (see dev_ui.hpp's panel-data-safety note).
        [this](const GraphicsSettings& s) { request_graphics_settings(s); },
        &fps_,
        &sim_ms_,
        &render_ms_,
        &fence_ms_,
        &present_wait_ms_,
    };
    if (!dev_ui_->init(*context_, std::move(bindings))) {
        return false; // compiled out (stub) or backend init failed
    }

    // init_dev_ui() claims the SceneRenderer's single post-UI callback slot;
    // hosts using the dev UI must not install their own.
    scene_renderer_->set_post_ui_callback(
        [this](SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchain) { dev_ui_->render(cmd, swapchain); });
    return true;
}

dev::DevUI& Application::dev_ui() {
    BRACKEN_CONTRACT(dev_ui_ != nullptr, "dev_ui() requires init_dev_ui() to have been called");
    return *dev_ui_;
}

std::string Application::settings_file_path() const {
    char* pref_path = SDL_GetPrefPath("bracken", "bracken");
    if (!pref_path) {
        return {};
    }
    std::string result = std::string(pref_path) + "settings.cfg";
    SDL_free(pref_path);
    return result;
}

// ========== Camera facade ==========

CameraController& Application::camera() {
    BRACKEN_CONTRACT(camera_ != nullptr, "camera() requires init_renderer() first");
    return *camera_;
}

const CameraController& Application::camera() const {
    BRACKEN_CONTRACT(camera_ != nullptr, "camera() requires init_renderer() first");
    return *camera_;
}

} // namespace bracken
