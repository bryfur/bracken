#include <gtest/gtest.h>

#include "bracken/action_map.hpp"

#include <cstdio>
#include <fstream>
#include <filesystem>
#include <string>

using namespace bracken;

namespace {

InputSnapshot with_keys(std::initializer_list<SDL_Scancode> down) {
    InputSnapshot s;
    for (SDL_Scancode sc : down) {
        s.keys_held[static_cast<std::size_t>(sc)] = true;
    }
    return s;
}

std::string temp_path(const char* leaf) {
    return (std::filesystem::temp_directory_path() / leaf).string();
}

} // namespace

TEST(ActionMap, AddIsIdempotent) {
    ActionMap m;
    const ActionId a = m.add("jump");
    const ActionId b = m.add("jump");
    EXPECT_EQ(a, b);
    EXPECT_EQ(m.action_count(), 1u);
}

TEST(ActionMap, UnknownActionIsSafeAndAlwaysFalse) {
    ActionMap m;
    const ActionId missing = m.find("never_registered");
    EXPECT_FALSE(missing.valid());
    // A typo must degrade to "never triggers", not to UB.
    m.update(with_keys({SDL_SCANCODE_SPACE}));
    EXPECT_FALSE(m.held(missing));
    EXPECT_FALSE(m.pressed(missing));
    EXPECT_FALSE(m.released(missing));
    EXPECT_FLOAT_EQ(m.axis(missing, missing), 0.0f);
}

TEST(ActionMap, HeldReflectsBoundKey) {
    ActionMap m;
    const ActionId jump = m.add("jump");
    m.bind(jump, Binding::key(SDL_SCANCODE_SPACE));

    m.update(with_keys({}));
    EXPECT_FALSE(m.held(jump));

    m.update(with_keys({SDL_SCANCODE_SPACE}));
    EXPECT_TRUE(m.held(jump));
}

TEST(ActionMap, EdgesFireExactlyOnce) {
    ActionMap m;
    const ActionId jump = m.add("jump");
    m.bind(jump, Binding::key(SDL_SCANCODE_SPACE));

    m.update(with_keys({}));
    EXPECT_FALSE(m.pressed(jump));

    m.update(with_keys({SDL_SCANCODE_SPACE}));
    EXPECT_TRUE(m.pressed(jump));
    EXPECT_FALSE(m.released(jump));

    m.update(with_keys({SDL_SCANCODE_SPACE})); // still held
    EXPECT_FALSE(m.pressed(jump)) << "press edge must not repeat while held";

    m.update(with_keys({}));
    EXPECT_TRUE(m.released(jump));
    EXPECT_FALSE(m.held(jump));
}

// The reason an action has a LIST of bindings: a host should not write `||`
// between a key and a pad button at every call site.
TEST(ActionMap, AnyOfSeveralBindingsTriggers) {
    ActionMap m;
    const ActionId jump = m.add("jump");
    m.bind(jump, Binding::key(SDL_SCANCODE_SPACE));
    m.bind(jump, Binding::key(SDL_SCANCODE_KP_0));

    m.update(with_keys({SDL_SCANCODE_KP_0}));
    EXPECT_TRUE(m.held(jump));
}

TEST(ActionMap, BindIsDeduplicated) {
    ActionMap m;
    const ActionId a = m.add("fire");
    m.bind(a, Binding::key(SDL_SCANCODE_F));
    m.bind(a, Binding::key(SDL_SCANCODE_F));
    EXPECT_EQ(m.bindings(a).size(), 1u) << "re-running setup must not accumulate duplicates";
}

// The defect a rebinding screen produces if the engine does not prevent it:
// two actions on one key, so pressing E does two things.
TEST(ActionMap, RebindExclusiveStealsFromOtherActions) {
    ActionMap m;
    const ActionId use = m.add("use");
    const ActionId drop = m.add("drop");
    m.bind(use, Binding::key(SDL_SCANCODE_E));
    m.bind(drop, Binding::key(SDL_SCANCODE_Q));

    m.rebind_exclusive(drop, Binding::key(SDL_SCANCODE_E));

    m.update(with_keys({SDL_SCANCODE_E}));
    EXPECT_TRUE(m.held(drop));
    EXPECT_FALSE(m.held(use)) << "the key must have been taken away from 'use'";
    EXPECT_TRUE(m.bindings(use).empty());
}

TEST(ActionMap, AxisCancelsOpposedKeys) {
    ActionMap m;
    const ActionId l = m.add("left");
    const ActionId r = m.add("right");
    m.bind(l, Binding::key(SDL_SCANCODE_A));
    m.bind(r, Binding::key(SDL_SCANCODE_D));

    m.update(with_keys({SDL_SCANCODE_D}));
    EXPECT_FLOAT_EQ(m.axis(l, r), 1.0f);

    m.update(with_keys({SDL_SCANCODE_A}));
    EXPECT_FLOAT_EQ(m.axis(l, r), -1.0f);

    m.update(with_keys({SDL_SCANCODE_A, SDL_SCANCODE_D}));
    EXPECT_FLOAT_EQ(m.axis(l, r), 0.0f) << "opposed keys should cancel";
}

// The oldest bug in keyboard movement: diagonals 1.41x faster than straight.
TEST(ActionMap, DiagonalVectorIsNotFasterThanStraight) {
    ActionMap m;
    const ActionId l = m.add("left");
    const ActionId r = m.add("right");
    const ActionId d = m.add("down");
    const ActionId u = m.add("up");
    m.bind(l, Binding::key(SDL_SCANCODE_A));
    m.bind(r, Binding::key(SDL_SCANCODE_D));
    m.bind(d, Binding::key(SDL_SCANCODE_S));
    m.bind(u, Binding::key(SDL_SCANCODE_W));

    m.update(with_keys({SDL_SCANCODE_W}));
    EXPECT_FLOAT_EQ(glm::length(m.vector(l, r, d, u)), 1.0f);

    m.update(with_keys({SDL_SCANCODE_W, SDL_SCANCODE_D}));
    EXPECT_NEAR(glm::length(m.vector(l, r, d, u)), 1.0f, 1e-5f);

    m.update(with_keys({}));
    EXPECT_FLOAT_EQ(glm::length(m.vector(l, r, d, u)), 0.0f);
}

TEST(ActionMap, ControllerBindingIgnoredWithoutController) {
    ActionMap m;
    const ActionId fire = m.add("fire");
    m.bind(fire, Binding::controller(ControllerButton::Primary));

    InputSnapshot s;
    s.has_controller = false;
    s.controller_primary = true; // stale value; must not count
    m.update(s);
    EXPECT_FALSE(m.held(fire));

    s.has_controller = true;
    m.update(s);
    EXPECT_TRUE(m.held(fire));
}

TEST(ActionMap, SaveLoadRoundTrip) {
    const std::string path = temp_path("bracken_actions_roundtrip.cfg");

    ActionMap saved;
    const ActionId jump = saved.add("jump");
    const ActionId fire = saved.add("fire");
    saved.bind(jump, Binding::key(SDL_SCANCODE_SPACE));
    saved.bind(jump, Binding::controller(ControllerButton::Primary));
    saved.bind(fire, Binding::mouse(MouseButton::Left));
    ASSERT_TRUE(saved.save(path));

    ActionMap loaded;
    const ActionId l_jump = loaded.add("jump");
    const ActionId l_fire = loaded.add("fire");
    loaded.bind(l_jump, Binding::key(SDL_SCANCODE_Z)); // a default to be replaced
    ASSERT_TRUE(loaded.load(path));

    ASSERT_EQ(loaded.bindings(l_jump).size(), 2u);
    EXPECT_EQ(loaded.bindings(l_jump)[0], Binding::key(SDL_SCANCODE_SPACE));
    EXPECT_EQ(loaded.bindings(l_jump)[1], Binding::controller(ControllerButton::Primary));
    ASSERT_EQ(loaded.bindings(l_fire).size(), 1u);
    EXPECT_EQ(loaded.bindings(l_fire)[0], Binding::mouse(MouseButton::Left));

    std::remove(path.c_str());
}

TEST(ActionMap, MissingFileLeavesDefaultsAlone) {
    ActionMap m;
    const ActionId jump = m.add("jump");
    m.bind(jump, Binding::key(SDL_SCANCODE_SPACE));
    EXPECT_FALSE(m.load(temp_path("bracken_actions_does_not_exist.cfg")));
    EXPECT_EQ(m.bindings(jump).size(), 1u);
}

// A settings file from an older build naming an action this build removed must
// not discard the rest of the player's setup.
TEST(ActionMap, UnknownActionInFileIsSkippedButRestLoads) {
    const std::string path = temp_path("bracken_actions_stale.cfg");
    {
        std::ofstream f(path);
        f << "retired_action=key:20\n";
        f << "jump=key:44\n"; // SDL_SCANCODE_SPACE
    }

    ActionMap m;
    const ActionId jump = m.add("jump");
    m.bind(jump, Binding::key(SDL_SCANCODE_Z));
    EXPECT_TRUE(m.load(path));

    ASSERT_EQ(m.bindings(jump).size(), 1u);
    EXPECT_EQ(m.bindings(jump)[0].code, static_cast<int>(SDL_SCANCODE_SPACE));

    std::remove(path.c_str());
}

// A half-applied rebind is worse than none: the action would end up partly on
// its defaults and partly on the file.
TEST(ActionMap, MalformedLineKeepsExistingBindings) {
    const std::string path = temp_path("bracken_actions_malformed.cfg");
    {
        std::ofstream f(path);
        f << "jump=key:notanumber\n";
    }

    ActionMap m;
    const ActionId jump = m.add("jump");
    m.bind(jump, Binding::key(SDL_SCANCODE_Z));
    EXPECT_TRUE(m.load(path));

    ASSERT_EQ(m.bindings(jump).size(), 1u);
    EXPECT_EQ(m.bindings(jump)[0].code, static_cast<int>(SDL_SCANCODE_Z));

    std::remove(path.c_str());
}

TEST(ActionMap, ActionAddedAfterFileWasWrittenKeepsItsDefault) {
    const std::string path = temp_path("bracken_actions_partial.cfg");
    {
        ActionMap old;
        const ActionId jump = old.add("jump");
        old.bind(jump, Binding::key(SDL_SCANCODE_SPACE));
        ASSERT_TRUE(old.save(path));
    }

    ActionMap m;
    const ActionId jump = m.add("jump");
    const ActionId sprint = m.add("sprint"); // new in this build
    m.bind(jump, Binding::key(SDL_SCANCODE_Z));
    m.bind(sprint, Binding::key(SDL_SCANCODE_LSHIFT));
    ASSERT_TRUE(m.load(path));

    EXPECT_EQ(m.bindings(jump)[0], Binding::key(SDL_SCANCODE_SPACE)) << "loaded";
    ASSERT_EQ(m.bindings(sprint).size(), 1u);
    EXPECT_EQ(m.bindings(sprint)[0], Binding::key(SDL_SCANCODE_LSHIFT)) << "default kept";

    std::remove(path.c_str());
}
