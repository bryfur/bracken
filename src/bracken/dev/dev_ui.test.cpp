#include <gtest/gtest.h>

#include "bracken/dev/dev_ui.hpp"

#include <SDL3/SDL_events.h>

using bracken::dev::DevUI;

// All tests here are headless: no init(), no GPU, no window. Nothing below
// touches the ImGui backend, so every expectation must hold identically in
// both configurations (BRACKEN_ENABLE_DEV_UI ON and the no-op stub).

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
