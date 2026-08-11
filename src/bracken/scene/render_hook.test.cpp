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

} // namespace
} // namespace bracken::scene
