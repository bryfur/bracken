#include <gtest/gtest.h>

#include "bracken/dev/dev_ui.hpp"

#include <SDL3/SDL_events.h>

using bracken::dev::DevUI;

// All tests are headless: no init(), no GPU, no window. Nothing here touches
// ImGui, so every expectation holds identically in both configurations
// (BRACKEN_ENABLE_DEV_UI ON and the no-op stub).
//
// The BUILT-IN PANELS are tested in dev_panels.test.cpp, which drives
// dev_panels.cpp through a bare ImGui context and therefore only compiles in
// the ON configuration.

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
