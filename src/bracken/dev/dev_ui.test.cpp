#include <gtest/gtest.h>

#include "bracken/dev/dev_ui.hpp"

#include <SDL3/SDL_events.h>

using bracken::dev::DevUI;

// All tests are headless: no init(), no GPU, no window. The DevUI tests in
// this first section touch no ImGui code at all, so every expectation holds
// identically in both configurations (BRACKEN_ENABLE_DEV_UI ON and the no-op
// stub). The DevUIPanels section further down drives the built-in panels
// through a bare ImGui context (core only, no SDL/GPU backends) and therefore
// only compiles in the ON configuration.

TEST(DevUI, ConstructsAndDestructsUninitialized) {
    DevUI ui;
    EXPECT_FALSE(ui.initialized());
    EXPECT_FALSE(ui.visible());
}

TEST(DevUI, AddPanelBeforeInitDoesNotCrashOrDraw) {
    DevUI ui;
    int draws = 0;
    ui.add_panel("Test", [&draws] { ++draws; });
    ui.add_panel("Another", [] {});
    // Panels are only stored; nothing draws without init() + a frame.
    EXPECT_EQ(draws, 0);
    EXPECT_FALSE(ui.initialized());
}

TEST(DevUI, VisibilityStateMachine) {
    DevUI ui;
    EXPECT_FALSE(ui.visible());
    ui.set_visible(true);
    EXPECT_TRUE(ui.visible());
    ui.set_visible(true); // idempotent
    EXPECT_TRUE(ui.visible());
    ui.toggle();
    EXPECT_FALSE(ui.visible());
    ui.toggle();
    EXPECT_TRUE(ui.visible());
    ui.set_visible(false);
    EXPECT_FALSE(ui.visible());
}

TEST(DevUI, ProcessEventReturnsFalseWhenUninitialized) {
    DevUI ui;
    SDL_Event event{};
    event.type = SDL_EVENT_KEY_DOWN;
    EXPECT_FALSE(ui.process_event(event));
    ui.set_visible(true); // visibility alone must not consume events
    EXPECT_FALSE(ui.process_event(event));
}

TEST(DevUI, RenderIsNoOpWhenUninitialized) {
    DevUI ui;
    ui.render(nullptr, nullptr);
    ui.set_visible(true); // visible but uninitialized: still a no-op
    ui.render(nullptr, nullptr);
    EXPECT_FALSE(ui.initialized());
}

TEST(DevUI, ShutdownWithoutInitIsSafe) {
    DevUI ui;
    ui.shutdown();
    EXPECT_FALSE(ui.initialized());
}

// NOTE: the init()-twice contract (second call returns false with a warning,
// existing bindings untouched) cannot be pinned headlessly — a successful
// first init needs a real window + GPU device. It is documented on
// DevUI::init() and guarded by the impl_ check at the top of init().

#if defined(BRACKEN_DEV_UI) && BRACKEN_DEV_UI

// ============================================================================
// Built-in panel tests — drive dev_panels.cpp through a bare ImGui context.
// ImGui core runs fully headless: setting RendererHasTextures makes NewFrame
// software-build the font atlas, and EndFrame without Render() discards the
// frame. No SDL platform/renderer backend is involved, so no window or GPU.
// ============================================================================

#include "bracken/dev/dev_panels.hpp"
#include "bracken/graphics_settings.hpp"
#include "bracken/systems/camera_controller.hpp"

#include <cstring>
#include <functional>
#include <imgui.h>
#include <imgui_internal.h>

namespace {

/// Owns a throwaway ImGui context for one test. No backends, no files.
class HeadlessImGui {
public:
    HeadlessImGui() {
        context_ = ImGui::CreateContext();
        ImGui::SetCurrentContext(context_);
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(1280.0f, 720.0f);
        io.DeltaTime = 1.0f / 60.0f;
        io.IniFilename = nullptr; // don't read/write imgui.ini from tests
        io.LogFilename = nullptr;
        // Pretend the (absent) renderer manages textures so NewFrame builds
        // the default font atlas in software instead of asserting.
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    }
    ~HeadlessImGui() { ImGui::DestroyContext(context_); }
    HeadlessImGui(const HeadlessImGui&) = delete;
    HeadlessImGui& operator=(const HeadlessImGui&) = delete;

    /// Run `frames` full ImGui frames, calling `body` inside each. Tab-bar
    /// contents only draw from the second frame (layout selects the first
    /// tab one frame after it is submitted), so tests use >= 3 frames.
    void run_frames(int frames, const std::function<void()>& body) {
        for (int i = 0; i < frames; ++i) {
            ImGui::NewFrame();
            body();
            ImGui::EndFrame(); // legal without Render(); discards the frame
        }
    }

private:
    ImGuiContext* context_ = nullptr;
};

/// True if any tab bar in the current context contains a tab named `name`.
bool any_tab_named(const char* name) {
    ImGuiContext& g = *ImGui::GetCurrentContext();
    for (int n = 0; n < g.TabBars.GetMapSize(); ++n) {
        if (ImGuiTabBar* tab_bar = g.TabBars.TryGetMapData(n)) {
            for (ImGuiTabItem& tab : tab_bar->Tabs) {
                if (const char* tab_name = ImGui::TabBarGetTabName(tab_bar, &tab)) {
                    if (std::strcmp(tab_name, name) == 0) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

/// Minimal CameraController so the Camera tab can draw headless. Records the
/// framing setters so tests can assert the panel pushes nothing unprompted.
class FakeCamera final : public bracken::systems::CameraController {
public:
    void set_target(const glm::vec3&) override {}
    void set_target_velocity(const glm::vec3&) override {}
    void set_yaw(float) override {}
    void set_pitch(float) override {}
    void rotate_yaw(float) override {}
    void rotate_pitch(float) override {}
    void adjust_zoom(float) override {}
    void set_focus_target(const glm::vec3*) override {}
    void set_focus_strength(float s) override {
        ++focus_strength_calls;
        focus_strength_ = s;
    }
    float get_focus_strength() const override { return focus_strength_; }
    void set_config(const bracken::systems::CameraModeConfig& config) override { config_ = config; }
    const bracken::systems::CameraModeConfig& get_config() const override { return config_; }
    void add_shake(bracken::systems::ShakeType, float, float) override {}
    void add_directional_shake(const glm::vec3&, float, float) override {}
    void set_fov_bias(float f) override {
        ++fov_bias_calls;
        fov_bias_ = f;
    }
    float get_fov_bias() const override { return fov_bias_; }
    void set_terrain_height_func(TerrainHeightFunc) override {}
    void set_screen_size(int, int) override {}
    void update(float) override {}
    glm::mat4 get_view_matrix() const override { return glm::mat4(1.0f); }
    glm::mat4 get_projection_matrix() const override { return glm::mat4(1.0f); }
    glm::vec3 get_position() const override { return glm::vec3(0.0f); }
    glm::vec3 get_forward() const override { return glm::vec3(0.0f, 0.0f, -1.0f); }

    int fov_bias_calls = 0;
    int focus_strength_calls = 0;

private:
    bracken::systems::CameraModeConfig config_;
    float fov_bias_ = 0.0f;
    float focus_strength_ = 0.0f;
};

} // namespace

// L86: a bound GraphicsSettings without an apply path must not surface the
// Graphics tab — editing the struct with no way to push it to the renderer
// would silently desync the two.
TEST(DevUIPanels, GraphicsTabHiddenWithoutApplyCallback) {
    HeadlessImGui imgui;
    bracken::GraphicsSettings settings;
    DevUI::Bindings bindings;
    bindings.settings = &settings; // apply_settings deliberately left empty
    DevUI::PanelState state;
    imgui.run_frames(3, [&] { bracken::dev::detail::draw_builtin_panels(bindings, state); });
    EXPECT_FALSE(any_tab_named("Graphics"));
}

TEST(DevUIPanels, GraphicsTabShownWithSettingsAndApply) {
    HeadlessImGui imgui;
    bracken::GraphicsSettings settings;
    DevUI::Bindings bindings;
    bindings.settings = &settings;
    bindings.apply_settings = [](const bracken::GraphicsSettings&) {};
    DevUI::PanelState state;
    imgui.run_frames(3, [&] { bracken::dev::detail::draw_builtin_panels(bindings, state); });
    EXPECT_TRUE(any_tab_named("Graphics"));
}

// L88: the FPS history lives on the DevUI instance, not in function-local
// statics — two panel states driven with different fps values must never
// interleave samples.
TEST(DevUIPanels, FpsHistoryIsPerInstanceState) {
    HeadlessImGui imgui;
    const float fps_a = 60.0f;
    const float fps_b = 30.0f;
    DevUI::Bindings bindings_a;
    bindings_a.fps = &fps_a;
    DevUI::Bindings bindings_b;
    bindings_b.fps = &fps_b;
    DevUI::PanelState state_a;
    DevUI::PanelState state_b;

    imgui.run_frames(3, [&] { bracken::dev::detail::draw_builtin_panels(bindings_a, state_a); });
    imgui.run_frames(3, [&] { bracken::dev::detail::draw_builtin_panels(bindings_b, state_b); });

    // Tab contents draw from the second frame on, so at least one sample
    // must have landed in each instance's history.
    EXPECT_GE(state_a.fps_next_sample, 1u);
    EXPECT_GE(state_b.fps_next_sample, 1u);
    // Every written sample carries the instance's own fps — with the old
    // process-wide statics, b's samples would have appended into a's buffer.
    for (std::size_t i = 0; i < state_a.fps_next_sample; ++i) {
        EXPECT_FLOAT_EQ(state_a.fps_history[i], fps_a);
    }
    for (std::size_t i = 0; i < state_b.fps_next_sample; ++i) {
        EXPECT_FLOAT_EQ(state_b.fps_history[i], fps_b);
    }
}

// L87: the Camera tab draws headless, re-derives its framing sliders from the
// controller's getters each frame, and pushes to the controller only when a
// widget actually changes — merely drawing the panel must not stomp the
// host's fov-bias/focus settings.
TEST(DevUIPanels, CameraTabDrawsWithoutPushingFramingSetters) {
    HeadlessImGui imgui;
    FakeCamera camera;
    // Host-set values must be reflected (not overwritten) by the panel.
    camera.set_fov_bias(7.5f);
    camera.set_focus_strength(0.4f);
    const int fov_calls_before = camera.fov_bias_calls;
    const int focus_calls_before = camera.focus_strength_calls;
    DevUI::Bindings bindings;
    bindings.camera = &camera;
    DevUI::PanelState state;
    imgui.run_frames(3, [&] { bracken::dev::detail::draw_builtin_panels(bindings, state); });

    EXPECT_TRUE(any_tab_named("Camera"));
    EXPECT_EQ(camera.fov_bias_calls, fov_calls_before);
    EXPECT_EQ(camera.focus_strength_calls, focus_calls_before);
    // The controller's values survive untouched without interaction.
    EXPECT_FLOAT_EQ(camera.get_fov_bias(), 7.5f);
    EXPECT_FLOAT_EQ(camera.get_focus_strength(), 0.4f);
}

#endif // BRACKEN_DEV_UI
