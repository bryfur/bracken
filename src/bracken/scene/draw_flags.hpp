#pragma once

#include <cstdint>

namespace bracken::scene {

/**
 * Per-draw options for FrameBuilder::draw_model / draw_skinned.
 *
 * These were two trailing `bool` parameters. Swapping them compiled silently
 * and changed the image, and a call site reading
 * `draw_model(h, xf, tint, false, false, id)` said nothing about which `false`
 * was which — the flagship defect in the flagship API. A named bit set costs
 * the same bytes in the command (one `uint32_t` replaces the two bools and
 * their explicit pad) and makes the transposition unrepresentable.
 *
 * `has()` is the only way to test a bit; there is no implicit conversion to
 * bool or to an integer, so `if (flags)` does not compile. `has(f, None)` is
 * false by construction — asking "is nothing set?" is `f == DrawFlags::None`.
 *
 * The stored word is part of the frame packet's byte-deterministic layout, so
 * the bit values are pinned by draw_flags.test.cpp and must not be renumbered.
 */
enum class DrawFlags : uint32_t {
    /** Default: instanced when the renderer can batch it, fogged with the frame. */
    None = 0,

    /**
     * Draw this command through the non-instanced path even when other
     * commands share its model — for a one-off that must not be merged into
     * an instance batch (and, in the shadow walk, one that casts only while
     * main-visible). Ignored by draw_skinned: skinned draws never instance.
     */
    ForceNonInstanced = 1u << 0,

    /**
     * Exclude this draw from the frame's distance/height fog. Reaches the GPU
     * as gpu::InstanceData::noFog / the non-instanced model lighting uniform.
     */
    NoFog = 1u << 1,
};

[[nodiscard]] constexpr DrawFlags operator|(DrawFlags a, DrawFlags b) noexcept {
    return static_cast<DrawFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

[[nodiscard]] constexpr DrawFlags operator&(DrawFlags a, DrawFlags b) noexcept {
    return static_cast<DrawFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

constexpr DrawFlags& operator|=(DrawFlags& a, DrawFlags b) noexcept {
    a = a | b;
    return a;
}

/** True when every bit of `flag` is set in `set`. False for DrawFlags::None. */
[[nodiscard]] constexpr bool has(DrawFlags set, DrawFlags flag) noexcept {
    return flag != DrawFlags::None && (set & flag) == flag;
}

} // namespace bracken::scene
