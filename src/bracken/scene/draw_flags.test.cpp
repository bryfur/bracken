#include <gtest/gtest.h>

#include "bracken/scene/draw_flags.hpp"

#include <cstdint>
#include <type_traits>

using bracken::scene::DrawFlags;
using bracken::scene::has;

// DrawFlags replaced two adjacent `bool` parameters on draw_model/draw_skinned
// whose transposition compiled silently and changed the image. These pin the
// two properties that make the replacement safe: the bits are distinct and
// stable (the word is stored in the frame packet), and `has()` answers about
// exactly one of them at a time.

TEST(DrawFlags, BitsAreDistinctAndStable) {
    // Pinned values, not just distinctness: the word is stored in the frame
    // packet and hashed, so renumbering it is a gate-visible change.
    EXPECT_EQ(static_cast<uint32_t>(DrawFlags::None), 0u);
    EXPECT_EQ(static_cast<uint32_t>(DrawFlags::ForceNonInstanced), 1u);
    EXPECT_EQ(static_cast<uint32_t>(DrawFlags::NoFog), 2u);
    static_assert(std::is_same_v<std::underlying_type_t<DrawFlags>, uint32_t>);
}

TEST(DrawFlags, HasIsPerBit) {
    constexpr DrawFlags both = DrawFlags::ForceNonInstanced | DrawFlags::NoFog;
    static_assert(has(both, DrawFlags::ForceNonInstanced));
    static_assert(has(both, DrawFlags::NoFog));

    static_assert(has(DrawFlags::NoFog, DrawFlags::NoFog));
    static_assert(!has(DrawFlags::NoFog, DrawFlags::ForceNonInstanced));
    static_assert(!has(DrawFlags::ForceNonInstanced, DrawFlags::NoFog));
    SUCCEED();
}

TEST(DrawFlags, NoneIsNeverPresent) {
    // "Is nothing set?" is `f == None`, never `has(f, None)` — the latter
    // would be vacuously true for a mask test and silently wrong here.
    static_assert(!has(DrawFlags::None, DrawFlags::None));
    static_assert(!has(DrawFlags::NoFog, DrawFlags::None));
    EXPECT_EQ(DrawFlags::None | DrawFlags::None, DrawFlags::None);
    SUCCEED();
}

TEST(DrawFlags, OrAndAndCompose) {
    DrawFlags f = DrawFlags::None;
    EXPECT_FALSE(has(f, DrawFlags::NoFog));
    f |= DrawFlags::NoFog;
    EXPECT_TRUE(has(f, DrawFlags::NoFog));
    EXPECT_FALSE(has(f, DrawFlags::ForceNonInstanced));
    f |= DrawFlags::ForceNonInstanced;
    EXPECT_EQ(f & DrawFlags::NoFog, DrawFlags::NoFog);
    EXPECT_EQ(f & DrawFlags::None, DrawFlags::None);
}
