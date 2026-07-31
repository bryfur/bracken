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

/**
 * Color — the packed-ARGB convention above, as a type.
 *
 * The engine's host-facing API used to hand out three unrelated color
 * conventions as three anonymous scalars: a packed `uint32_t` (UI, text,
 * buttons, billboards, debug lines), a LINEAR `glm::vec4` draw tint, and a
 * LINEAR `glm::vec3` light color. Nothing at a call site said which, and the
 * two float ones are not interchangeable with the packed one — they carry
 * more than 8 bits per channel and are allowed to exceed 1.0. Naming the
 * packed one is what makes the other two readable as "not that".
 *
 * `Color` is exactly the 0xAARRGGBB word — 4 bytes, trivially copyable,
 * standard layout — so it drops into the frame packet's byte-deterministic
 * command structs in place of the `uint32_t` it replaces. Construction from
 * the packed word is implicit on purpose: `0xFF30FF60` IS a color literal,
 * and forcing `Color{...}` around forty existing ones would buy nothing that
 * the parameter type does not already say.
 *
 * NO sRGB transfer is applied anywhere in this file. `to_linear()` is
 * channel/255: UI, debug and billboard colors are authored directly in the
 * engine's linear space, and adding a decode here would shift every one of
 * them. (This is why there is no `srgb8` factory — the name would document a
 * conversion that does not happen.)
 */
struct Color {
    uint32_t argb = 0xFF000000u; // opaque black

    constexpr Color() noexcept = default;
    /// Implicit: the packed 0xAARRGGBB literal hosts already write.
    constexpr Color(uint32_t packed_argb) noexcept : argb(packed_argb) {} // NOLINT(google-explicit-constructor)

    /// From 8-bit channels, no transfer function (see the class comment).
    [[nodiscard]] static constexpr Color rgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xFF) noexcept {
        return Color{pack_color(r, g, b, a)};
    }

    /// From normalized components. Quantizes to 8 bits per channel — do not
    /// route an HDR draw tint or a light color through this.
    [[nodiscard]] static constexpr Color from_linear(const glm::vec4& c) noexcept {
        return Color{pack_color(quantize(c.x), quantize(c.y), quantize(c.z), quantize(c.w))};
    }
    [[nodiscard]] static constexpr Color rgb(float r, float g, float b) noexcept {
        return from_linear({r, g, b, 1.0f});
    }
    [[nodiscard]] static constexpr Color rgba(float r, float g, float b, float a) noexcept {
        return from_linear({r, g, b, a});
    }

    /// The single decode every renderer uses: channel/255, in [0, 1].
    [[nodiscard]] constexpr glm::vec4 to_linear() const noexcept { return unpack_color(argb); }

    [[nodiscard]] static constexpr Color white() noexcept { return Color{0xFFFFFFFFu}; }
    [[nodiscard]] static constexpr Color black() noexcept { return Color{0xFF000000u}; }
    [[nodiscard]] static constexpr Color transparent() noexcept { return Color{0x00000000u}; }

    friend constexpr bool operator==(Color, Color) noexcept = default;

private:
    // Round-to-nearest with clamping, so from_linear(to_linear(c)) == c for
    // every one of the 2^32 words (a truncating cast does not round-trip).
    [[nodiscard]] static constexpr uint8_t quantize(float v) noexcept {
        const float scaled = v * 255.0f + 0.5f;
        return scaled <= 0.0f ? uint8_t{0} : scaled >= 255.0f ? uint8_t{255} : static_cast<uint8_t>(scaled);
    }
};

static_assert(sizeof(Color) == sizeof(uint32_t), "Color must stay the packed word it replaces");

} // namespace bracken::core
