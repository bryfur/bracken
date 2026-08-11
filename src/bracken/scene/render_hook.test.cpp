// The ordered, multi-slot host-pass registry — headless.
//
// RenderHookRegistry is deliberately device-free (no GPU, no SDL, no
// SceneRenderer) precisely so the two properties that matter can be pinned
// here: that the execution order inside a slot is a TOTAL order determined by
// what the host asked for, and that a handle whose registration is gone can
// never remove somebody else's hook.
//
// The thing this replaces is SceneRenderer::set_post_ui_callback, a single
// assignable slot that Application::init_dev_ui already owned — so a game that
// used it silently turned the developer overlay off. "Many per slot" is the
// whole point, and SlotsAreIndependent + HooksRunInOrderThenRegistrationOrder
// are what say so mechanically.

#include "bracken/scene/render_hook.hpp"

#include "bracken/core/assert.hpp"

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace bracken::scene {
namespace {

/// Run a slot the way SceneRenderer::run_render_hooks does — under a
/// DispatchScope, walking entries() in order — and record who ran.
std::vector<std::string> dispatch(RenderHookRegistry& registry, RenderHookSlot slot) {
    std::vector<std::string> visited;
    RenderHookContext ctx;
    ctx.slot = slot;
    RenderHookRegistry::DispatchScope guard(registry);
    for (RenderHookRegistry::Entry* entry : registry.entries(slot)) {
        entry->fn(ctx);
        visited.push_back(entry->name);
    }
    return visited;
}

RenderHookFn noop() {
    return [](const RenderHookContext&) {};
}

TEST(RenderHookRegistryTest, EverySlotIsEmptyOnAFreshRegistry) {
    core::bind_main_thread();
    RenderHookRegistry registry;
    EXPECT_TRUE(registry.empty());
    EXPECT_EQ(registry.size(), 0u);
    // The frame path's early-out reads empty(slot) per slot; all of them must
    // answer true before anything is registered, or an engine with no host
    // passes would start building contexts.
    for (size_t i = 0; i < k_render_hook_slot_count; ++i) {
        EXPECT_TRUE(registry.empty(static_cast<RenderHookSlot>(i))) << "slot " << i;
        EXPECT_TRUE(registry.entries(static_cast<RenderHookSlot>(i)).empty()) << "slot " << i;
    }
}

TEST(RenderHookRegistryTest, HooksRunInOrderThenRegistrationOrder) {
    core::bind_main_thread();
    RenderHookRegistry registry;
    // Orders {10, -5, 10}: the negative one first, then the two tens in the
    // order they were registered. Asserting the exact SEQUENCE, not the set —
    // "they all ran" would pass with a hash-ordered container.
    const auto ten_a = registry.add(RenderHookSlot::AfterOpaque, "ten_a", 10, noop());
    const auto minus_five = registry.add(RenderHookSlot::AfterOpaque, "minus_five", -5, noop());
    const auto ten_b = registry.add(RenderHookSlot::AfterOpaque, "ten_b", 10, noop());
    ASSERT_TRUE(ten_a.is_valid());
    ASSERT_TRUE(minus_five.is_valid());
    ASSERT_TRUE(ten_b.is_valid());

    const std::vector<std::string> expected{"minus_five", "ten_a", "ten_b"};
    EXPECT_EQ(dispatch(registry, RenderHookSlot::AfterOpaque), expected);
    EXPECT_EQ(registry.size(), 3u);
    EXPECT_FALSE(registry.empty());
    EXPECT_FALSE(registry.empty(RenderHookSlot::AfterOpaque));
}

TEST(RenderHookRegistryTest, SlotsAreIndependent) {
    core::bind_main_thread();
    RenderHookRegistry registry;
    (void)registry.add(RenderHookSlot::AfterOpaque, "opaque", 0, noop());
    (void)registry.add(RenderHookSlot::BeforeTaa, "taa", 0, noop());
    (void)registry.add(RenderHookSlot::AfterUi, "ui", 0, noop());

    EXPECT_EQ(dispatch(registry, RenderHookSlot::AfterOpaque), std::vector<std::string>{"opaque"});
    EXPECT_EQ(dispatch(registry, RenderHookSlot::BeforeTaa), std::vector<std::string>{"taa"});
    EXPECT_EQ(dispatch(registry, RenderHookSlot::AfterUi), std::vector<std::string>{"ui"});
    EXPECT_TRUE(registry.empty(RenderHookSlot::PreForward));
    EXPECT_TRUE(registry.empty(RenderHookSlot::AfterComposite));
    EXPECT_EQ(registry.size(), 3u);
}

TEST(RenderHookRegistryTest, TheSlotTheHookRanInIsInItsContext) {
    core::bind_main_thread();
    RenderHookRegistry registry;
    RenderHookSlot seen = RenderHookSlot::PreForward;
    (void)registry.add(RenderHookSlot::AfterComposite, "watcher", 0,
                       [&seen](const RenderHookContext& ctx) { seen = ctx.slot; });
    (void)dispatch(registry, RenderHookSlot::AfterComposite);
    EXPECT_EQ(seen, RenderHookSlot::AfterComposite);
}

TEST(RenderHookRegistryTest, RemoveInvalidatesTheHandleAndKeepsNeighbourOrder) {
    core::bind_main_thread();
    RenderHookRegistry registry;
    (void)registry.add(RenderHookSlot::AfterOpaque, "first", 0, noop());
    const auto middle = registry.add(RenderHookSlot::AfterOpaque, "middle", 1, noop());
    (void)registry.add(RenderHookSlot::AfterOpaque, "last", 2, noop());

    ASSERT_TRUE(registry.remove(middle));
    EXPECT_EQ(dispatch(registry, RenderHookSlot::AfterOpaque), (std::vector<std::string>{"first", "last"}));
    EXPECT_EQ(registry.size(), 2u);
    // A retained copy of the handle is now a no-op, not a removal of whatever
    // took the slot.
    EXPECT_FALSE(registry.remove(middle));
    EXPECT_EQ(registry.size(), 2u);
}

TEST(RenderHookRegistryTest, AReusedSlotDoesNotAnswerToTheOldHandle) {
    core::bind_main_thread();
    RenderHookRegistry registry;
    const auto first = registry.add(RenderHookSlot::AfterOpaque, "first", 0, noop());
    ASSERT_TRUE(registry.remove(first));
    const auto second = registry.add(RenderHookSlot::AfterOpaque, "second", 0, noop());
    EXPECT_EQ(second.index, first.index) << "the freed slot should be reused";
    EXPECT_NE(second.generation, first.generation);

    EXPECT_FALSE(registry.remove(first)) << "the stale handle must not remove the new registration";
    EXPECT_EQ(dispatch(registry, RenderHookSlot::AfterOpaque), std::vector<std::string>{"second"});
}

TEST(RenderHookRegistryTest, EntryNameAddressesAreStableAcrossRegistration) {
    core::bind_main_thread();
    RenderHookRegistry registry;
    (void)registry.add(RenderHookSlot::AfterOpaque, "pinned", -100, noop());
    ASSERT_FALSE(registry.entries(RenderHookSlot::AfterOpaque).empty());
    RenderHookRegistry::Entry* entry = registry.entries(RenderHookSlot::AfterOpaque)[0];
    const char* name_address = entry->name.c_str();

    for (int i = 0; i < 64; ++i) {
        (void)registry.add(RenderHookSlot::AfterOpaque, "filler" + std::to_string(i), i, noop());
    }

    // The direct test of what the GPU debug-label API needs: it stores the pass
    // name as a raw const char* and reads it later in the frame, so a
    // vector-of-values entry store would dangle here on reallocation — and only
    // for non-SSO names, i.e. intermittently, in a game, never in the suite.
    ASSERT_EQ(registry.entries(RenderHookSlot::AfterOpaque)[0], entry);
    EXPECT_EQ(entry->name.c_str(), name_address);
}

TEST(RenderHookRegistryTest, TheInternedNameOutLivesTheRegistrationItCameFrom) {
    core::bind_main_thread();
    RenderHookRegistry registry;
    // Entry::stable_name exists because GPUTimestampPool::reserve_pair stores
    // its pass name as a RAW POINTER and RenderStats::pass_times_ms keeps a
    // string_view over it until the NEXT frame's read_results. A host may
    // legally remove a hook at the frame fence — after the frame that recorded
    // the timing and before the read of it — so entry->name.c_str() would be a
    // use-after-free there. This is the direct statement of the fix.
    const auto handle = registry.add(RenderHookSlot::AfterOpaque,
                                     "a_name_far_too_long_for_the_small_string_optimization_to_hide_the_bug", 0,
                                     noop());
    ASSERT_TRUE(handle.is_valid());
    ASSERT_FALSE(registry.entries(RenderHookSlot::AfterOpaque).empty());
    const char* recorded = registry.entries(RenderHookSlot::AfterOpaque)[0]->stable_name;
    ASSERT_NE(recorded, nullptr);
    const std::string expected = recorded;

    ASSERT_TRUE(registry.remove(handle));
    // The registration is gone; the pointer the timer pool is still holding is
    // not. (ASan/UBSan is the detector that makes this test load-bearing.)
    EXPECT_EQ(std::string(recorded), expected);

    // Re-registering the same name reuses the same interned node, which is what
    // bounds the pool for a host that adds and removes every frame.
    const auto again = registry.add(RenderHookSlot::BeforeTaa, expected, 0, noop());
    ASSERT_TRUE(again.is_valid());
    EXPECT_EQ(registry.entries(RenderHookSlot::BeforeTaa)[0]->stable_name, recorded);
}

TEST(RenderHookRegistryDeathTest, AnOutOfRangeSlotIsANamedAbortNotAnOutOfBoundsRead) {
    core::bind_main_thread();
    RenderHookRegistry registry;
    // RenderHookSlot is a scoped enum a host supplies, and the backing store is
    // a fixed array. The contract is live in release; nothing here is a debug
    // -only check.
    EXPECT_DEATH((void)registry.entries(static_cast<RenderHookSlot>(k_render_hook_slot_count)), "");
    EXPECT_DEATH((void)registry.empty(static_cast<RenderHookSlot>(200)), "");
}

TEST(RenderHookRegistryTest, AnEmptyCallbackIsRefused) {
    core::bind_main_thread();
    RenderHookRegistry registry;
    const auto handle = registry.add(RenderHookSlot::AfterOpaque, "empty", 0, RenderHookFn{});
    EXPECT_FALSE(handle.is_valid());
    EXPECT_TRUE(registry.empty());
}

TEST(RenderHookRegistryTest, ClearRemovesEverythingAndStalesEveryHandle) {
    core::bind_main_thread();
    RenderHookRegistry registry;
    const auto a = registry.add(RenderHookSlot::PreForward, "a", 0, noop());
    const auto b = registry.add(RenderHookSlot::AfterUi, "b", 0, noop());
    registry.clear();
    EXPECT_TRUE(registry.empty());
    EXPECT_EQ(registry.size(), 0u);
    EXPECT_FALSE(registry.remove(a));
    EXPECT_FALSE(registry.remove(b));
    for (size_t i = 0; i < k_render_hook_slot_count; ++i) {
        EXPECT_TRUE(registry.entries(static_cast<RenderHookSlot>(i)).empty()) << "slot " << i;
    }
}

TEST(RenderHookRegistryTest, DispatchScopeMarksTheRegistryBusyAndReleasesIt) {
    core::bind_main_thread();
    RenderHookRegistry registry;
    EXPECT_FALSE(registry.is_dispatching());
    {
        RenderHookRegistry::DispatchScope guard(registry);
        EXPECT_TRUE(registry.is_dispatching());
    }
    EXPECT_FALSE(registry.is_dispatching());
    // …and a hook can still be registered once dispatch is over.
    EXPECT_TRUE(registry.add(RenderHookSlot::AfterOpaque, "after", 0, noop()).is_valid());
}

TEST(RenderHookRegistryTest, SlotNamesAreDistinctAndOutOfRangeIsUnknown) {
    std::vector<std::string> names;
    for (size_t i = 0; i < k_render_hook_slot_count; ++i) {
        std::string name = render_hook_slot_to_string(static_cast<RenderHookSlot>(i));
        EXPECT_NE(name, "Unknown") << "slot " << i << " has no name";
        for (const std::string& seen : names) {
            EXPECT_NE(seen, name) << "duplicate slot name " << name;
        }
        names.push_back(std::move(name));
    }
    EXPECT_EQ(names.size(), k_render_hook_slot_count);
    EXPECT_STREQ(render_hook_slot_to_string(static_cast<RenderHookSlot>(k_render_hook_slot_count)), "Unknown");
}

// A hook that registers another hook reallocates the very list the dispatcher
// is walking. That is a host mistake, so it is a BRACKEN_CONTRACT — live in
// release, a named abort rather than a use-after-free three passes later.
TEST(RenderHookRegistryDeathTest, MutatingFromInsideADispatchAborts) {
    core::bind_main_thread();
    RenderHookRegistry registry;
    EXPECT_DEATH(
        {
            RenderHookRegistry::DispatchScope guard(registry);
            (void)registry.add(RenderHookSlot::AfterOpaque, "reentrant", 0, noop());
        },
        "");
    RenderHookRegistry other;
    const auto handle = other.add(RenderHookSlot::AfterOpaque, "victim", 0, noop());
    EXPECT_DEATH(
        {
            RenderHookRegistry::DispatchScope guard(other);
            (void)other.remove(handle);
        },
        "");
    EXPECT_DEATH(
        {
            RenderHookRegistry::DispatchScope guard(other);
            other.clear();
        },
        "");
}

// =============================================================================
// The developer overlay keeps its slot.
//
// The bug this whole feature exists to close is that SceneRenderer had exactly
// one host extension point, Application::init_dev_ui had already taken it, and
// a game calling set_post_ui_callback replaced the overlay without a word. Two
// properties have to hold for that not to come back in a new shape, and neither
// is expressible as a type, so they are pinned here.
// =============================================================================

TEST(RenderHookRegistryTest, RegisteringNeverDisplacesAnExistingRegistration) {
    core::bind_main_thread();
    RenderHookRegistry registry;

    // The mechanical statement of "many per slot": 64 registrations into ONE
    // slot, all with the same order and the same name, and all 64 survive.
    // `set_post_ui_callback` would have kept the last one. If a future
    // "optimization" ever deduplicates by name, this fails.
    std::vector<RenderHookHandle> handles;
    for (int i = 0; i < 64; ++i) {
        handles.push_back(registry.add(RenderHookSlot::AfterUi, "overlay", 0, noop()));
        ASSERT_TRUE(handles.back().is_valid()) << "registration " << i << " was refused";
    }
    EXPECT_EQ(registry.entries(RenderHookSlot::AfterUi).size(), 64u);
    EXPECT_EQ(registry.size(), 64u);

    // And every handle still resolves to its OWN entry: removing the first
    // leaves 63, not 0, and does not invalidate the rest.
    EXPECT_TRUE(registry.remove(handles.front()));
    EXPECT_EQ(registry.entries(RenderHookSlot::AfterUi).size(), 63u);
    for (size_t i = 1; i < handles.size(); ++i) {
        EXPECT_TRUE(registry.remove(handles[i])) << "handle " << i << " went stale when its neighbour was removed";
    }
    EXPECT_TRUE(registry.empty());
}

/// Whole text of a repo-relative file, or empty when it cannot be read.
std::string read_repo_file(const std::string& relative) {
#ifndef BRACKEN_REPO_ROOT
    (void)relative;
    return {};
#else
    std::ifstream in(std::filesystem::path(BRACKEN_REPO_ROOT) / relative);
    if (!in) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
#endif
}

TEST(RenderHookRegistryTest, TheAfterUiSlotRunsUnderTheDeveloperOverlayNotInsteadOfIt) {
    const std::string source = read_repo_file("src/bracken/scene/scene_renderer.cpp");
    if (source.empty()) {
        GTEST_SKIP() << "BRACKEN_REPO_ROOT not defined or scene_renderer.cpp unreadable";
    }

    const size_t after_ui = source.find("run_render_hooks(RenderHookSlot::AfterUi");
    ASSERT_NE(after_ui, std::string::npos) << "the AfterUi dispatch is gone — hosts have lost the swapchain slot";
    const size_t overlay = source.find("post_ui_callback_(post_cmd");
    ASSERT_NE(overlay, std::string::npos)
        << "the post-UI callback invocation was renamed; the developer overlay rides on it, so update this test "
           "deliberately rather than deleting it";

    // ORDER IS THE CONTRACT. AfterUi hooks record first, so a game's overlay
    // ends up UNDER the developer overlay. Swap these two and F10 starts
    // drawing behind the game's HUD — visible, but only to whoever happens to
    // open the overlay in a build with a host overlay installed.
    EXPECT_LT(after_ui, overlay) << "RenderHookSlot::AfterUi must be dispatched BEFORE post_ui_callback_";

    // Exactly one dispatch per slot: a second call site would run every hook in
    // that slot twice, which for a blended overlay is a doubled image rather
    // than an error.
    EXPECT_EQ(source.find("run_render_hooks(RenderHookSlot::AfterUi", after_ui + 1), std::string::npos);

    // Taking the legacy single slot from under the overlay is still POSSIBLE
    // (the setter is public and one caller legitimately re-installs), but it may
    // not be SILENT — that was the half of the original bug that made it cost a
    // debugging session rather than a glance at the log.
    EXPECT_NE(source.find("set_post_ui_callback replaced an already-installed callback"), std::string::npos)
        << "the overwrite warning is gone; a game can once again turn F10 off without a word";
}

TEST(RenderHookRegistryTest, TheDeveloperOverlayOwnsADedicatedSlotNotAHookRegistration) {
    const std::string app = read_repo_file("src/bracken/application.cpp");
    if (app.empty()) {
        GTEST_SKIP() << "BRACKEN_REPO_ROOT not defined or application.cpp unreadable";
    }

    // init_dev_ui still uses the dedicated single-assignment callback.
    EXPECT_NE(app.find("set_post_ui_callback"), std::string::npos)
        << "the developer overlay no longer claims the post-UI callback";

    // And it deliberately does NOT register itself as a render hook. If it did,
    // it would be one entry in a list a host can reorder (by `order`) or empty
    // outright (`clear()`), which is the same failure as before with the arrow
    // pointing the other way. The overlay's slot has to be one nothing else can
    // reach.
    EXPECT_EQ(app.find("render_hooks()"), std::string::npos)
        << "Application must not put the developer overlay in the shared hook registry — a host's clear() would "
           "remove it";
}

} // namespace
} // namespace bracken::scene
