#include "render_context.hpp"
#include "bracken/core/logger.hpp"
#include "SDL3/SDL_error.h"
#include "SDL3/SDL_gpu.h"
#include "SDL3/SDL_video.h"
#include <algorithm>
#include <cstdint>
#include <string>

namespace bracken::render {

namespace {
constexpr const char* LOG_TAG = "render";
} // namespace


RenderContext::~RenderContext() {
    shutdown();
}

bool RenderContext::init(int width, int height, const std::string& title) {
    width_ = width;
    height_ = height;

    // Create window without OpenGL flag - SDL3 GPU handles its own context
    window_ = SDL_CreateWindow(title.c_str(), width, height, SDL_WINDOW_RESIZABLE);
    if (!window_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "RenderContext::init: Failed to create window: {}", SDL_GetError());
        return false;
    }

    // Initialize the GPU device with the window
    if (!device_.init(window_)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "RenderContext::init: Failed to initialize GPU device");
        SDL_DestroyWindow(window_);
        window_ = nullptr;
        return false;
    }

    // Cache the drawable size in PIXELS (HiDPI-correct); the swapchain and all
    // offscreen targets are pixel-sized, not logical-unit-sized.
    update_viewport();

    BRACKEN_LOG_INFO(LOG_TAG, "RenderContext::init: Initialized with {} backend", device_.driver_name().c_str());
    BRACKEN_LOG_INFO(LOG_TAG, "RenderContext::init: Drawable size: {}x{} pixels", width_, height_);

    return true;
}

void RenderContext::shutdown() {
    current_cmd_ = nullptr;
    device_.shutdown();

    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }

    width_ = 0;
    height_ = 0;
    current_window_mode_ = -1;
    current_resolution_index_ = -1;
}

void RenderContext::update_viewport() {
    if (window_) {
        // Pixel size, not logical window size: under HiDPI scaling these
        // differ and the swapchain (plus every swapchain-sized offscreen
        // target: AO, bloom, fog, LDR) is allocated in pixels.
        SDL_GetWindowSizeInPixels(window_, &width_, &height_);
        // Note: In SDL3 GPU, viewport is set per render pass, not globally
    }
}

void RenderContext::begin_frame() {
    if (current_cmd_ != nullptr) {
        BRACKEN_LOG_WARN(LOG_TAG, "RenderContext::begin_frame: Warning - called while a frame is already in progress. "
                                  "Submitting previous frame to avoid command buffer leak.");
        end_frame(current_cmd_);
    }
    current_cmd_ = begin_frame_cmd();
}

SDL_GPUCommandBuffer* RenderContext::begin_frame_cmd() {
    update_viewport();
    return device_.begin_frame();
}

void RenderContext::end_frame() {
    end_frame(current_cmd_);
    current_cmd_ = nullptr;
}

void RenderContext::end_frame(SDL_GPUCommandBuffer* cmd) {
    device_.end_frame(cmd);
}

bool RenderContext::end_frame_and_wait() {
    bool ok = device_.end_frame_and_wait(current_cmd_);
    current_cmd_ = nullptr;
    return ok;
}

SDL_GPUTexture* RenderContext::acquire_swapchain_texture(SDL_GPUCommandBuffer* cmd, uint32_t* out_width,
                                                         uint32_t* out_height) {
    return device_.acquire_swapchain_texture(cmd, out_width, out_height);
}

int RenderContext::max_vsync_mode() const {
    if (device_.supports_present_mode(SDL_GPU_PRESENTMODE_MAILBOX)) {
        return 2;
    }
    if (device_.supports_present_mode(SDL_GPU_PRESENTMODE_VSYNC)) {
        return 1;
    }
    return 0;
}

void RenderContext::set_vsync_mode(int mode) {
    // Clamp to what the swapchain actually supports (e.g. mailbox is not
    // available on all platforms/drivers); settings may request more.
    mode = std::min(mode, max_vsync_mode());
    if (vsync_mode_ == mode) {
        return;
    }
    vsync_mode_ = mode;

    // Map our mode to SDL present modes:
    // 0 = off (immediate), 1 = vsync, 2 = triple buffer (mailbox)
    SDL_GPUPresentMode present_mode = SDL_GPU_PRESENTMODE_IMMEDIATE;
    switch (mode) {
        case 1:
            present_mode = SDL_GPU_PRESENTMODE_VSYNC;
            break;
        case 2:
            present_mode = SDL_GPU_PRESENTMODE_MAILBOX;
            break;
        default:
            present_mode = SDL_GPU_PRESENTMODE_IMMEDIATE;
            break;
    }
    if (!device_.set_swapchain_parameters(present_mode)) {
        BRACKEN_LOG_WARN(LOG_TAG, "RenderContext::set_vsync_mode: failed to set present mode {}", mode);
    }
}

void RenderContext::query_display_modes() {
    available_resolutions_.clear();
    if (!window_) {
        return;
    }

    SDL_DisplayID display = SDL_GetDisplayForWindow(window_);
    int count = 0;
    const SDL_DisplayMode* const* modes = SDL_GetFullscreenDisplayModes(display, &count);
    if (!modes) {
        return;
    }

    for (int i = 0; i < count; i++) {
        // Deduplicate by (w, h, pixel_density) — SDL may list the same
        // resolution at different refresh rates; the first listed mode wins.
        // All pixel densities are kept (exclusive-mode matching in
        // set_window_mode keys on this density-inclusive triple).
        bool duplicate = false;
        for (const auto& existing : available_resolutions_) {
            if (existing.w == modes[i]->w && existing.h == modes[i]->h &&
                existing.pixel_density == modes[i]->pixel_density) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            available_resolutions_.push_back(
                {modes[i]->w, modes[i]->h, modes[i]->refresh_rate, modes[i]->pixel_density});
        }
    }
}

void RenderContext::set_window_mode(int window_mode, int resolution_index) {
    if (!window_) {
        return;
    }

    // Fullscreen/window transitions are expensive; skip when the request
    // matches the last applied state.
    if (window_mode == current_window_mode_ && resolution_index == current_resolution_index_) {
        return;
    }

    if (window_mode == 0) {
        // Windowed
        SDL_SetWindowFullscreenMode(window_, nullptr);
        SDL_SetWindowFullscreen(window_, false);
    } else if (window_mode == 1) {
        // Borderless fullscreen
        SDL_SetWindowFullscreenMode(window_, nullptr);
        SDL_SetWindowFullscreen(window_, true);
    } else {
        // Exclusive fullscreen — use selected display mode
        if (available_resolutions_.empty()) {
            query_display_modes();
        }

        SDL_DisplayID display = SDL_GetDisplayForWindow(window_);
        int count = 0;
        const SDL_DisplayMode* const* modes = SDL_GetFullscreenDisplayModes(display, &count);
        const SDL_DisplayMode* selected = nullptr;

        if (modes && resolution_index >= 0 && resolution_index < static_cast<int>(available_resolutions_.size())) {
            const auto& res = available_resolutions_[resolution_index];
            for (int i = 0; i < count; i++) {
                if (modes[i]->w == res.w && modes[i]->h == res.h && modes[i]->pixel_density == res.pixel_density) {
                    selected = modes[i];
                    break;
                }
            }
        }

        if (selected) {
            SDL_SetWindowFullscreenMode(window_, selected);
        } else {
            // Stale resolution_index (settings file from another monitor, or
            // the mode list changed): fall back to borderless deterministically.
            // Clearing the fullscreen mode also discards any exclusive mode a
            // previous call applied — without this, SDL would keep the OLD
            // exclusive resolution.
            BRACKEN_LOG_WARN(LOG_TAG,
                             "RenderContext::set_window_mode: resolution_index {} matches no display mode "
                             "({} available); falling back to borderless fullscreen",
                             resolution_index, static_cast<int>(available_resolutions_.size()));
            SDL_SetWindowFullscreenMode(window_, nullptr);
            window_mode = 1;
        }
        SDL_SetWindowFullscreen(window_, true);
    }

    // Record what was actually applied — the exclusive no-match path demotes
    // window_mode to borderless above, so a later retry of the same exclusive
    // request is not swallowed by the change-detection early-out.
    current_window_mode_ = window_mode;
    current_resolution_index_ = resolution_index;
    update_viewport();
}

} // namespace bracken::render
