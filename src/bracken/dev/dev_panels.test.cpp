// Built-in dev-overlay panels (dev_panels.cpp) driven headless.
//
// dev_panels.cpp is 460 lines that only ever ran behind F10 in the playground:
// every tab-visibility rule, every "re-derive from the owner each frame rather
// than cache" decision, and every "don't push unless a widget changed" guard
// was enforced by nothing. Those are exactly the rules whose violation is
// invisible — a panel that silently stomps the host's GraphicsSettings, or
// re-seeds the sun direction from its own rounded sliders every frame, looks
// completely normal on screen.
//
// The harness runs ImGui core fully headless: setting RendererHasTextures
// makes NewFrame software-build the font atlas, and EndFrame without Render()
// discards the frame. No SDL platform/renderer backend, no window, no GPU.
// (These tests moved here from dev_ui.test.cpp, which keeps the DevUI
// lifecycle/visibility tests that hold in both build configurations.)

#include <gtest/gtest.h>

#include "bracken/dev/dev_ui.hpp"

#if defined(BRACKEN_DEV_UI) && BRACKEN_DEV_UI

#include "bracken/dev/dev_panels.hpp"
#include "bracken/graphics_settings.hpp"
#include "bracken/render/environment.hpp"
#include "bracken/scene/scene_renderer.hpp"
#include "bracken/systems/camera_controller.hpp"

#include <cstddef>
#include <cstring>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <imgui.h>
#include <imgui_internal.h>

using bracken::dev::DevUI;

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
    /// contents only draw from the second frame (layout selects the first tab
    /// one frame after it is submitted), so tests use >= 3 frames.
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

/// Select a tab by name, so its BODY (not just its header) draws. Tab bars
/// otherwise only ever draw the first visible tab's contents, which would
/// leave three of the four panels untested.
void select_tab(const char* name) {
    ImGuiContext& g = *ImGui::GetCurrentContext();
    for (int n = 0; n < g.TabBars.GetMapSize(); ++n) {
        if (ImGuiTabBar* tab_bar = g.TabBars.TryGetMapData(n)) {
            for (ImGuiTabItem& tab : tab_bar->Tabs) {
                if (const char* tab_name = ImGui::TabBarGetTabName(tab_bar, &tab)) {
                    if (std::strcmp(tab_name, name) == 0) {
                        tab_bar->SelectedTabId = tab.ID;
                        tab_bar->NextSelectedTabId = tab.ID;
                        return;
                    }
                }
            }
        }
    }
}

/// True if `name` is the currently selected tab — i.e. BeginTabItem(name)
/// returned true on the last frame and its BODY drew.
bool tab_is_selected(const char* name) {
    ImGuiContext& g = *ImGui::GetCurrentContext();
    for (int n = 0; n < g.TabBars.GetMapSize(); ++n) {
        if (ImGuiTabBar* tab_bar = g.TabBars.TryGetMapData(n)) {
            for (ImGuiTabItem& tab : tab_bar->Tabs) {
                if (tab.ID != tab_bar->SelectedTabId) {
                    continue;
                }
                const char* tab_name = ImGui::TabBarGetTabName(tab_bar, &tab);
                if (tab_name && std::strcmp(tab_name, name) == 0) {
                    return true;
                }
            }
        }
    }
    return false;
}

/// Draw the panels for `frames` frames with `tab` forced selected from the
/// second frame on, so the named tab's body is exercised — and verify it
/// actually was, so a test asserting "drawing this tab changes nothing" cannot
/// pass vacuously by never drawing the tab.
void draw_with_tab(HeadlessImGui& imgui, const DevUI::Bindings& bindings, DevUI::PanelState& state, const char* tab,
                   int frames = 5) {
    imgui.run_frames(frames, [&] {
        select_tab(tab);
        bracken::dev::detail::draw_builtin_panels(bindings, state);
        select_tab(tab);
    });
    EXPECT_TRUE(tab_is_selected(tab)) << "tab '" << tab << "' never became the selected tab; its body did not draw";
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
    void add_shake(bracken::systems::ShakeType, float, float) override { ++shake_calls; }
    void add_directional_shake(const glm::vec3&, float, float) override { ++shake_calls; }
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
    glm::vec3 get_position() const override { return position; }
    glm::vec3 get_forward() const override { return forward; }

    glm::vec3 position{10.0f, 20.0f, 30.0f};
    glm::vec3 forward{0.0f, 0.0f, -1.0f};
    int fov_bias_calls = 0;
    int focus_strength_calls = 0;
    int shake_calls = 0;

private:
    bracken::systems::CameraModeConfig config_;
    float fov_bias_ = 0.0f;
    float focus_strength_ = 0.0f;
};

} // namespace

// ============================================================================
// Tab visibility — each tab's bindings are its precondition
// ============================================================================

TEST(DevPanels, NoBindingsDrawsNoTabs) {
    HeadlessImGui imgui;
    DevUI::Bindings bindings; // everything null
    DevUI::PanelState state;
    imgui.run_frames(3, [&] { bracken::dev::detail::draw_builtin_panels(bindings, state); });
    EXPECT_FALSE(any_tab_named("Stats"));
    EXPECT_FALSE(any_tab_named("Environment"));
    EXPECT_FALSE(any_tab_named("Graphics"));
    EXPECT_FALSE(any_tab_named("Camera"));
}

TEST(DevPanels, AllFourTabsAppearWhenFullyBound) {
    HeadlessImGui imgui;
    bracken::scene::SceneRenderer renderer; // never init()ed: no device, no GPU
    FakeCamera camera;
    bracken::GraphicsSettings settings;
    const float fps = 60.0f;
    DevUI::Bindings bindings;
    bindings.scene_renderer = &renderer;
    bindings.camera = &camera;
    bindings.settings = &settings;
    bindings.apply_settings = [](const bracken::GraphicsSettings&) {};
    bindings.fps = &fps;
    DevUI::PanelState state;
    imgui.run_frames(3, [&] { bracken::dev::detail::draw_builtin_panels(bindings, state); });

    EXPECT_TRUE(any_tab_named("Stats"));
    EXPECT_TRUE(any_tab_named("Environment"));
    EXPECT_TRUE(any_tab_named("Graphics"));
    EXPECT_TRUE(any_tab_named("Camera"));
}

// The Stats tab is the one tab with two independent reasons to exist: an fps
// pointer alone must be enough (a host with no SceneRenderer binding still
// gets the frame-time plot).
TEST(DevPanels, StatsTabAppearsWithFpsAlone) {
    HeadlessImGui imgui;
    const float fps = 144.0f;
    DevUI::Bindings bindings;
    bindings.fps = &fps;
    DevUI::PanelState state;
    imgui.run_frames(3, [&] { bracken::dev::detail::draw_builtin_panels(bindings, state); });
    EXPECT_TRUE(any_tab_named("Stats"));
    EXPECT_FALSE(any_tab_named("Environment")); // needs the renderer
}

// L86: a bound GraphicsSettings without an apply path must not surface the
// Graphics tab — editing the struct with no way to push it to the renderer
// would silently desync the two.
TEST(DevPanels, GraphicsTabHiddenWithoutApplyCallback) {
    HeadlessImGui imgui;
    bracken::GraphicsSettings settings;
    DevUI::Bindings bindings;
    bindings.settings = &settings; // apply_settings deliberately left empty
    DevUI::PanelState state;
    imgui.run_frames(3, [&] { bracken::dev::detail::draw_builtin_panels(bindings, state); });
    EXPECT_FALSE(any_tab_named("Graphics"));
}

TEST(DevPanels, GraphicsTabShownWithSettingsAndApply) {
    HeadlessImGui imgui;
    bracken::GraphicsSettings settings;
    DevUI::Bindings bindings;
    bindings.settings = &settings;
    bindings.apply_settings = [](const bracken::GraphicsSettings&) {};
    DevUI::PanelState state;
    imgui.run_frames(3, [&] { bracken::dev::detail::draw_builtin_panels(bindings, state); });
    EXPECT_TRUE(any_tab_named("Graphics"));
}

// ============================================================================
// Stats tab
// ============================================================================

// L88: the FPS history lives on the DevUI instance, not in function-local
// statics — two panel states driven with different fps values must never
// interleave samples.
TEST(DevPanels, FpsHistoryIsPerInstanceState) {
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

// The plot is a ring buffer: the write cursor must wrap instead of running off
// the end of the fixed array.
TEST(DevPanels, FpsHistoryCursorWrapsWithoutOverrunning) {
    HeadlessImGui imgui;
    const float fps = 90.0f;
    DevUI::Bindings bindings;
    bindings.fps = &fps;
    DevUI::PanelState state;
    const std::size_t capacity = state.fps_history.size();
    imgui.run_frames(static_cast<int>(capacity) * 2 + 5,
                     [&] { bracken::dev::detail::draw_builtin_panels(bindings, state); });
    EXPECT_LT(state.fps_next_sample, capacity);
    for (float sample : state.fps_history) {
        EXPECT_FLOAT_EQ(sample, fps); // every slot written, none out of range
    }
}

// Documented side effect: the Stats tab turns render-stats collection ON while
// it is drawn (the counters are otherwise never accumulated and the readouts
// would be frozen zeros), and deliberately never turns it back off.
TEST(DevPanels, StatsTabEnablesCollectionAndNeverDisablesIt) {
    HeadlessImGui imgui;
    bracken::scene::SceneRenderer renderer;
    DevUI::Bindings bindings;
    bindings.scene_renderer = &renderer;
    DevUI::PanelState state;
    ASSERT_FALSE(renderer.collect_stats());

    draw_with_tab(imgui, bindings, state, "Stats");
    EXPECT_TRUE(renderer.collect_stats());

    // Drawing a different tab must not switch it back off — a host that turned
    // collection on itself (the smoke driver does) must keep it.
    draw_with_tab(imgui, bindings, state, "Environment");
    EXPECT_TRUE(renderer.collect_stats());
}

// The phase breakdown is all-or-nothing: with any timing pointer null the
// block is skipped rather than printing a partial, misleading row set.
TEST(DevPanels, PartialTimingBindingsDrawNothingAndDoNotCrash) {
    HeadlessImGui imgui;
    const float fps = 60.0f;
    const float sim = 1.0f;
    const float render = 2.0f;
    DevUI::Bindings bindings;
    bindings.fps = &fps;
    bindings.sim_ms = &sim;
    bindings.render_ms = &render;
    // fence_ms / present_wait_ms deliberately left null
    DevUI::PanelState state;
    draw_with_tab(imgui, bindings, state, "Stats");
    EXPECT_TRUE(any_tab_named("Stats"));
}

TEST(DevPanels, FullTimingBindingsDraw) {
    HeadlessImGui imgui;
    const float fps = 60.0f;
    const float sim = 0.4f;
    const float render = 7.9f;
    const float fence = 0.01f;
    const float present = 3.6f;
    DevUI::Bindings bindings;
    bindings.fps = &fps;
    bindings.sim_ms = &sim;
    bindings.render_ms = &render;
    bindings.fence_ms = &fence;
    bindings.present_wait_ms = &present;
    DevUI::PanelState state;
    draw_with_tab(imgui, bindings, state, "Stats");
    EXPECT_TRUE(any_tab_named("Stats"));
}

// ============================================================================
// Environment tab
// ============================================================================

// The sun sliders decompose sun_dir into azimuth/elevation every frame and
// recompose ONLY when a slider moves. Merely drawing the tab must leave the
// environment bit-identical — otherwise a host day/night cycle would fight the
// panel's rounding every frame the overlay is open.
TEST(DevPanels, EnvironmentTabDrawsWithoutMutatingEnvironment) {
    HeadlessImGui imgui;
    bracken::scene::SceneRenderer renderer;
    // A non-default, non-axis-aligned sun so a re-composition round-trip would
    // visibly move it.
    renderer.environment().sun_dir = glm::normalize(glm::vec3(0.31f, -0.72f, 0.62f));
    renderer.environment().cloud_coverage = 0.42f;
    renderer.environment().grade_slope = glm::vec3(1.1f, 0.95f, 1.02f);
    const bracken::render::EnvironmentState before = renderer.environment();

    DevUI::Bindings bindings;
    bindings.scene_renderer = &renderer;
    DevUI::PanelState state;
    draw_with_tab(imgui, bindings, state, "Environment");

    const bracken::render::EnvironmentState& after = renderer.environment();
    EXPECT_EQ(std::memcmp(&before, &after, sizeof(before)), 0)
        << "drawing the Environment tab mutated EnvironmentState";
}

// A degenerate sun_dir (zero length) must not produce NaNs in the derived
// azimuth/elevation — the panel substitutes straight up.
TEST(DevPanels, EnvironmentTabSurvivesDegenerateSunDirection) {
    HeadlessImGui imgui;
    bracken::scene::SceneRenderer renderer;
    renderer.environment().sun_dir = glm::vec3(0.0f);
    DevUI::Bindings bindings;
    bindings.scene_renderer = &renderer;
    DevUI::PanelState state;
    draw_with_tab(imgui, bindings, state, "Environment");
    const glm::vec3 sun = renderer.environment().sun_dir;
    EXPECT_FALSE(glm::any(glm::isnan(sun)));
    EXPECT_EQ(sun, glm::vec3(0.0f)); // untouched: no slider moved
}

// ============================================================================
// Graphics tab
// ============================================================================

// apply_settings pushes to the renderer; drawing without interaction must not
// fire it, and must not edit the host's struct.
TEST(DevPanels, GraphicsTabDoesNotApplyOrEditWithoutInteraction) {
    HeadlessImGui imgui;
    bracken::GraphicsSettings settings;
    settings.shadow_mode = bracken::ShadowMode::PCSS;
    settings.exposure = 1.7f;
    settings.bloom_enabled = false;
    const bracken::GraphicsSettings before = settings;

    int applies = 0;
    DevUI::Bindings bindings;
    bindings.settings = &settings;
    bindings.apply_settings = [&applies](const bracken::GraphicsSettings&) { ++applies; };
    DevUI::PanelState state;
    draw_with_tab(imgui, bindings, state, "Graphics");

    EXPECT_EQ(applies, 0);
    EXPECT_EQ(std::memcmp(&before, &settings, sizeof(before)), 0) << "the Graphics tab edited settings unprompted";
}

// combo_enum clamps out-of-range enum values for display, but must not write
// the clamped value back when the widget did not change.
TEST(DevPanels, GraphicsTabLeavesOutOfRangeEnumsAlone) {
    HeadlessImGui imgui;
    bracken::GraphicsSettings settings;
    settings.shadow_cascades = static_cast<bracken::ShadowCascades>(99);
    DevUI::Bindings bindings;
    bindings.settings = &settings;
    bindings.apply_settings = [](const bracken::GraphicsSettings&) {};
    DevUI::PanelState state;
    draw_with_tab(imgui, bindings, state, "Graphics");
    EXPECT_EQ(static_cast<int>(settings.shadow_cascades), 99);
}

// ============================================================================
// Camera tab
// ============================================================================

// L87: the Camera tab draws headless, re-derives its framing sliders from the
// controller's getters each frame, and pushes to the controller only when a
// widget actually changes — merely drawing the panel must not stomp the
// host's fov-bias/focus settings, or fire a shake.
TEST(DevPanels, CameraTabDrawsWithoutPushingFramingSetters) {
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
    draw_with_tab(imgui, bindings, state, "Camera");

    EXPECT_TRUE(any_tab_named("Camera"));
    EXPECT_EQ(camera.fov_bias_calls, fov_calls_before);
    EXPECT_EQ(camera.focus_strength_calls, focus_calls_before);
    EXPECT_EQ(camera.shake_calls, 0);
    // The controller's values survive untouched without interaction.
    EXPECT_FLOAT_EQ(camera.get_fov_bias(), 7.5f);
    EXPECT_FLOAT_EQ(camera.get_focus_strength(), 0.4f);
    // Shake parameters are panel state and keep their defaults.
    EXPECT_FLOAT_EQ(state.shake_intensity, 1.0f);
    EXPECT_FLOAT_EQ(state.shake_duration, 0.3f);
}

// The yaw/pitch readout is derived from the forward vector; a straight-down
// camera is the degenerate case (asin at the clamp) and must not produce NaN
// or trip an assert.
TEST(DevPanels, CameraTabSurvivesStraightDownForward) {
    HeadlessImGui imgui;
    FakeCamera camera;
    camera.forward = glm::vec3(0.0f, -1.0f, 0.0f);
    DevUI::Bindings bindings;
    bindings.camera = &camera;
    DevUI::PanelState state;
    draw_with_tab(imgui, bindings, state, "Camera");
    EXPECT_TRUE(any_tab_named("Camera"));
    EXPECT_EQ(camera.shake_calls, 0);
}

#endif // BRACKEN_DEV_UI
