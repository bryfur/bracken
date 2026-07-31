#pragma once

#include <cstdint>
#include <glm/vec4.hpp>

namespace bracken::core {

// Engine-wide packed-color convention
// ===================================
// Every packed `uint32_t` color parameter in the engine — UI primitives,
// textured-quad tints, text, buttons, 3D billboards, debug lines — is
// **ARGB**: `0xAARRGGBB` (alpha in bits 31-24, red 23-16, green 15-8,
// blue 7-0). `0xFFFF0000` is opaque red, `0xFF00FF00` opaque green,
// `0x80000000` half-transparent black.
//
// All packing/unpacking must go through the helpers below — do not
// hand-roll shifts at call sites, and never decode the same word with a
// different byte order in a renderer.

/// Pack 8-bit channels into the canonical ARGB (0xAARRGGBB) word.
[[nodiscard]] constexpr uint32_t pack_color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xFF) noexcept {
    return (static_cast<uint32_t>(a) << 24) | (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(g) << 8) |
           static_cast<uint32_t>(b);
}

/// Unpack a canonical ARGB (0xAARRGGBB) word into normalized (r, g, b, a),
/// each channel in [0, 1]. This is the single decode used by every renderer.
[[nodiscard]] constexpr glm::vec4 unpack_color(uint32_t argb) noexcept {
    return {static_cast<float>((argb >> 16) & 0xFFu) / 255.0f, static_cast<float>((argb >> 8) & 0xFFu) / 255.0f,
            static_cast<float>(argb & 0xFFu) / 255.0f, static_cast<float>((argb >> 24) & 0xFFu) / 255.0f};
}

} // namespace bracken::core
