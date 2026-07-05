#include "application.hpp"
#include "bracken/core/assert.hpp"
#include "bracken/core/asset/file_watcher.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/dev/dev_ui.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/render/render_context.hpp"
#include "bracken/scene/scene_renderer.hpp"
#include "bracken/systems/camera_system.hpp"
#include "SDL3/SDL_error.h"
#include "SDL3/SDL_filesystem.h"
#include "SDL3/SDL_init.h"
#include "SDL3/SDL_timer.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

namespace bracken {

using namespace bracken::render;
using namespace bracken::scene;
using namespace bracken::systems;

Application::Application() = default;

Application::~Application() = default;

bool Application::init_engine() {
    // Record the engine main thread for BRACKEN_ASSERT_MAIN_THREAD() checks
    // (FileWatcher entry points, swapchain acquire, render_frame).
    core::bind_main_thread();
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

void Application::run() {
    running_ = true;
    last_frame_time_ = SDL_GetTicksNS();
    fps_timer_ = last_frame_time_;

    while (running_) {
        // Nanosecond ticks: whole-millisecond timing quantizes dt to 0/1 ms steps
        // at high frame rates, juddering animation and camera motion.
        uint64_t current_time = SDL_GetTicksNS();
        float dt = static_cast<float>(static_cast<double>(current_time - last_frame_time_) /
                                      static_cast<double>(SDL_NS_PER_SECOND));
        last_frame_time_ = current_time;

        // Clamp delta time to avoid huge jumps
        dt = std::min(dt, 0.1f);

        // FPS counter
        frame_count_++;
        if (current_time - fps_timer_ >= static_cast<uint64_t>(SDL_NS_PER_SECOND)) {
            fps_ = static_cast<float>(frame_count_);
            frame_count_ = 0;
            fps_timer_ = current_time;
        }

        if (core::asset::FileWatcher::instance().is_initialized()) {
            core::asset::FileWatcher::instance().poll_main_thread();
        }

        // Process input - poll events, let subclass see them first
        {
            BRACKEN_PROFILE_ZONE("App::poll_events");
            input_.reset_camera_deltas();
            input_.clear_menu_inputs();
            input_.clear_key_edges();
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_EVENT_QUIT) {
                    running_ = false;
                    break;
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
            if (!running_) {
                break;
            }
            input_.post_process_events();
        }

        {
            BRACKEN_PROFILE_ZONE("App::update");
            on_update(dt);
        }
        {
            BRACKEN_PROFILE_ZONE("App::render");
            on_render();
        }

        BRACKEN_PROFILE_FRAME();
    }
}

void Application::shutdown_engine() {
    core::asset::FileWatcher::instance().shutdown();
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

void Application::render_frame(RenderScene& scene, const UIScene& ui_scene, const CameraState& camera, float dt) {
    scene_renderer_->render_frame(scene, ui_scene, camera, dt);
}

void Application::set_heightmap(const Heightmap& heightmap) {
    scene_renderer_->set_heightmap(heightmap);
}

void Application::set_graphics_settings(const GraphicsSettings& settings) {
    scene_renderer_->set_graphics_settings(settings);
}

void Application::set_anisotropic_filter(int level) {
    scene_renderer_->set_anisotropic_filter(level);
}

void Application::set_vsync_mode(int mode) {
    scene_renderer_->set_vsync_mode(mode);
}

int Application::max_vsync_mode() const {
    return scene_renderer_->max_vsync_mode();
}

void Application::set_window_mode(int window_mode, int resolution_index) {
    context_->set_window_mode(window_mode, resolution_index);
}

std::vector<Application::DisplayMode> Application::available_resolutions() const {
    std::vector<DisplayMode> result;
    for (const auto& m : context_->available_resolutions()) {
        result.push_back({m.w, m.h});
    }
    return result;
}

ModelManager& Application::models() {
    return scene_renderer_->models();
}

float Application::get_terrain_height(float x, float z) {
    return scene_renderer_->get_terrain_height(x, z);
}

int Application::screen_width() const {
    return context_->width();
}

int Application::screen_height() const {
    return context_->height();
}

void Application::set_collect_render_stats(bool enabled) {
    scene_renderer_->set_collect_stats(enabled);
}

const RenderStats& Application::render_stats() const {
    return scene_renderer_->render_stats();
}

std::string Application::gpu_driver_name() const {
    return context_->device().driver_name();
}

void Application::request_screenshot(const std::string& path) {
    scene_renderer_->request_screenshot(path);
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
        [this](const GraphicsSettings& s) { apply_all_graphics_settings(s); },
        &fps_,
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
    BRACKEN_ASSERT(dev_ui_ != nullptr, "dev_ui() requires init_dev_ui() to have been called");
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
    return *camera_;
}

const CameraController& Application::camera() const {
    return *camera_;
}

} // namespace bracken
