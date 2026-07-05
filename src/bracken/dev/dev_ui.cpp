#include "bracken/dev/dev_ui.hpp"

#if defined(BRACKEN_DEV_UI) && BRACKEN_DEV_UI

// ============================================================================
// Full implementation — Dear ImGui with the SDL3 platform backend and the
// SDLGPU3 renderer backend, compiled when BRACKEN_ENABLE_DEV_UI is ON.
// ============================================================================

#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/dev/dev_panels.hpp"
#include "bracken/render/render_context.hpp"

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlgpu3.h>

#include <utility>

namespace bracken::dev {

/// ImGui/backend state. Only exists between a successful init() and
/// shutdown(); `impl_ != nullptr` is the "initialized" signal.
struct DevUI::Impl {
    ImGuiContext* context = nullptr;   ///< owned ImGui context
    SDL_GPUDevice* device = nullptr;   ///< non-owning; for the idle wait at shutdown
    bool backends_initialized = false; ///< SDL3 + SDLGPU3 backends are up
};

DevUI::DevUI() = default;

DevUI::~DevUI() {
    shutdown();
}

bool DevUI::init(render::RenderContext& context, Bindings bindings) {
    if (impl_) {
        return true; // already initialized
    }
    bindings_ = std::move(bindings);

    auto impl = std::make_unique<Impl>();
    IMGUI_CHECKVERSION();
    impl->context = ImGui::CreateContext();
    impl->device = context.device().handle();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    if (!ImGui_ImplSDL3_InitForSDLGPU(context.window())) {
        BRACKEN_LOG_ERROR("devui", "ImGui SDL3 platform backend init failed");
        ImGui::DestroyContext(impl->context);
        return false;
    }

    ImGui_ImplSDLGPU3_InitInfo init_info = {};
    init_info.Device = context.device().handle();
    init_info.ColorTargetFormat = context.swapchain_format();
    init_info.MSAASamples = SDL_GPU_SAMPLECOUNT_1;
    if (!ImGui_ImplSDLGPU3_Init(&init_info)) {
        BRACKEN_LOG_ERROR("devui", "ImGui SDLGPU3 renderer backend init failed");
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext(impl->context);
        return false;
    }

    impl->backends_initialized = true;
    impl_ = std::move(impl);
    BRACKEN_LOG_INFO("devui", "Dev UI initialized (Dear ImGui {})", IMGUI_VERSION);
    return true;
}

void DevUI::shutdown() {
    if (!impl_) {
        return; // never initialized, or already shut down — both fine
    }
    if (impl_->backends_initialized) {
        // The last frame's draw data may still be in flight; the backends
        // destroy their GPU objects immediately, so drain the device first.
        if (impl_->device) {
            SDL_WaitForGPUIdle(impl_->device);
        }
        ImGui_ImplSDL3_Shutdown();
        ImGui_ImplSDLGPU3_Shutdown();
    }
    if (impl_->context) {
        ImGui::DestroyContext(impl_->context);
    }
    impl_.reset();
}

bool DevUI::process_event(const SDL_Event& event) {
    if (!impl_) {
        return false;
    }
    // Always feed the backend: ImGui tracks mouse/keyboard/window state even
    // while the overlay is hidden, so it is coherent the moment it opens.
    // (render() drains the resulting input queue every frame, visible or not.)
    ImGui_ImplSDL3_ProcessEvent(&event);

    // The toggle key is handled here, not by the host, so the overlay can
    // always be closed — ImGui's keyboard capture (e.g. a focused InputText)
    // would otherwise swallow the host's toggle binding.
    if (toggle_key_ != SDL_SCANCODE_UNKNOWN && event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat &&
        event.key.scancode == toggle_key_) {
        toggle();
        return true;
    }

    if (!visible_) {
        return false;
    }
    const ImGuiIO& io = ImGui::GetIO();
    switch (event.type) {
    case SDL_EVENT_MOUSE_MOTION:
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_WHEEL:
        return io.WantCaptureMouse;
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_TEXT_INPUT:
    case SDL_EVENT_TEXT_EDITING:
    case SDL_EVENT_TEXT_EDITING_CANDIDATES:
        return io.WantCaptureKeyboard;
    // Releases are never consumed: WantCapture* lags one frame behind ImGui's
    // click ownership, and a swallowed KEY_UP / BUTTON_UP would wedge
    // InputHandler's edge state (e.g. right-mouse relative mode stuck on).
    case SDL_EVENT_MOUSE_BUTTON_UP:
    case SDL_EVENT_KEY_UP:
    default:
        return false;
    }
}

bool DevUI::wants_keyboard() const {
    if (!impl_ || !visible_) {
        return false;
    }
    return ImGui::GetIO().WantCaptureKeyboard;
}

bool DevUI::wants_mouse() const {
    if (!impl_ || !visible_) {
        return false;
    }
    return ImGui::GetIO().WantCaptureMouse;
}

void DevUI::render(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* swapchain) {
    if (!impl_) {
        return;
    }
    if (!visible_ || cmd == nullptr || swapchain == nullptr) {
        // Drain the ImGui input queue even while hidden: process_event feeds
        // events unconditionally, and only NewFrame consumes them — without
        // this, hours of hidden-overlay input accumulate unboundedly and
        // replay as phantom clicks the moment the overlay opens.
        ImGui_ImplSDLGPU3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        ImGui::EndFrame(); // legal without Render(); discards the frame
        return;
    }
    BRACKEN_PROFILE_ZONE("DevUI::render");

    ImGui_ImplSDLGPU3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    detail::draw_builtin_panels(bindings_);
    for (auto& [name, draw] : panels_) {
        if (ImGui::Begin(name.c_str())) { // false while collapsed — skip the body
            draw();
        }
        ImGui::End(); // must be called regardless of Begin's return value
    }

    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();
    if (draw_data == nullptr || draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f) {
        return; // minimized window: nothing to upload or draw
    }

    // Mandatory before the render pass: records the vertex/index/texture
    // uploads on the command buffer (copy passes — must be outside any pass).
    ImGui_ImplSDLGPU3_PrepareDrawData(draw_data, cmd);

    // Composite over the finished frame: LOAD, never clear.
    SDL_GPUColorTargetInfo target_info = {};
    target_info.texture = swapchain;
    target_info.load_op = SDL_GPU_LOADOP_LOAD;
    target_info.store_op = SDL_GPU_STOREOP_STORE;
    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &target_info, 1, nullptr);
    if (pass != nullptr) {
        ImGui_ImplSDLGPU3_RenderDrawData(draw_data, cmd, pass);
        SDL_EndGPURenderPass(pass);
    }
}

} // namespace bracken::dev

#else // !BRACKEN_DEV_UI

// ============================================================================
// Stub — compiled when BRACKEN_ENABLE_DEV_UI is OFF. Every entry point is a
// safe no-op so host code needs no #ifdefs; init() reports failure.
// ============================================================================

namespace bracken::dev {

struct DevUI::Impl {}; // never instantiated; completes unique_ptr<Impl>'s deleter

DevUI::DevUI() = default;

DevUI::~DevUI() {
    shutdown();
}

bool DevUI::init(render::RenderContext& /*context*/, Bindings /*bindings*/) {
    return false;
}

void DevUI::shutdown() {}

bool DevUI::process_event(const SDL_Event& /*event*/) {
    return false;
}

bool DevUI::wants_keyboard() const {
    return false;
}

bool DevUI::wants_mouse() const {
    return false;
}

void DevUI::render(SDL_GPUCommandBuffer* /*cmd*/, SDL_GPUTexture* /*swapchain*/) {}

} // namespace bracken::dev

#endif // BRACKEN_DEV_UI
