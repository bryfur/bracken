#pragma once

#include <functional>
#include <memory>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_gpu.h>
#include <string>
#include <utility>
#include <vector>

namespace bracken::render {
class RenderContext;
}
namespace bracken::scene {
class SceneRenderer;
}
namespace bracken::systems {
class CameraController;
}
namespace bracken {
struct GraphicsSettings;
}

namespace bracken::dev {

/**
 * Immediate-mode developer overlay (Dear ImGui, SDL3 GPU backend).
 *
 * Compiled to a no-op stub when the CMake option BRACKEN_ENABLE_DEV_UI is OFF
 * (`init()` returns false, everything else is safe to call), so host code
 * needs no #ifdefs. `Application::init_dev_ui()` wires everything for you:
 * it binds the built-in panels, claims the SceneRenderer post-UI callback,
 * and feeds SDL events before game input. Hosts add their own windows with
 * `add_panel`.
 *
 * Built-in panels (one window, tabbed): Stats (fps, draw calls, triangles,
 * particles), Environment (sun, fog, volumetric params — live), Graphics
 * (GraphicsSettings editors applied through the host's apply path), Camera
 * (readouts + shake/fov test buttons).
 *
 * Threading: main thread only, like the rest of the frame path.
 */
class DevUI {
public:
    DevUI();
    ~DevUI();
    DevUI(const DevUI&) = delete;
    DevUI& operator=(const DevUI&) = delete;
    DevUI(DevUI&&) = delete;
    DevUI& operator=(DevUI&&) = delete;

    /** Everything the built-in panels read or mutate. Non-owning; must
     *  outlive the DevUI. Null members hide their panel sections. */
    struct Bindings {
        scene::SceneRenderer* scene_renderer = nullptr; ///< environment, render stats, effects
        systems::CameraController* camera = nullptr;    ///< readouts + shake/fov primitives
        GraphicsSettings* settings = nullptr;           ///< edited in place...
        std::function<void(const GraphicsSettings&)> apply_settings; ///< ...then pushed through this
        const float* fps = nullptr;                     ///< Application's fps counter
    };

    /** Create the ImGui context + SDL3/SDLGPU3 backends. Call once, after the
     *  renderer is up. Returns false if compiled out or backend init fails. */
    [[nodiscard]] bool init(render::RenderContext& context, Bindings bindings);
    void shutdown();
    [[nodiscard]] bool initialized() const { return impl_ != nullptr; }

    /** Feed every SDL event before game input sees it. Returns true when the
     *  overlay consumed the event (visible and ImGui wants mouse/keyboard). */
    bool process_event(const SDL_Event& event);

    /** Build and render the overlay. Must be called from the SceneRenderer
     *  post-UI callback (command buffer recording, outside any pass). No-op
     *  while hidden or uninitialized. */
    void render(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchain);

    void set_visible(bool visible) { visible_ = visible; }
    [[nodiscard]] bool visible() const { return visible_; }
    void toggle() { visible_ = !visible_; }

    /** Register a host panel, drawn as its own collapsible window while the
     *  overlay is visible. `draw` runs inside the ImGui frame between
     *  Begin(name)/End — body-only ImGui:: calls. Safe to call before init. */
    void add_panel(std::string name, std::function<void()> draw) {
        panels_.emplace_back(std::move(name), std::move(draw));
    }

private:
    struct Impl; // ImGui/backend state; nullptr until init and in stub builds
    std::unique_ptr<Impl> impl_;
    Bindings bindings_;
    bool visible_ = false;
    std::vector<std::pair<std::string, std::function<void()>>> panels_;
};

} // namespace bracken::dev
