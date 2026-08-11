#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace bracken::core {

/**
 * Decode a single UTF-8 codepoint from `text` starting at byte index `i`.
 * Advances `i` past the decoded sequence. Malformed input yields U+FFFD and
 * advances by one byte so decoding always makes progress.
 *
 * This lived inline in `render/text_renderer.hpp` until the font-metrics
 * mirror needed it: `contracts/` is a LEAF layer and may include only
 * `core/`, `contracts/`, glm and `<std>` (core/module_layering.test.cpp
 * "ContractsAreALeafLayer"), so a shared decoder has to sink to `core/`.
 * `render::decode_utf8` is a using-declaration onto this one — there is
 * exactly one decoder in the tree, and both spellings name it.
 */
inline uint32_t decode_utf8(std::string_view text, std::size_t& i) {
    const uint8_t b0 = static_cast<uint8_t>(text[i]);
    if (b0 < 0x80) {
        i += 1;
        return b0;
    }

    uint32_t cp = 0;
    std::size_t len = 0;
    if ((b0 & 0xE0) == 0xC0) {
        cp = b0 & 0x1F;
        len = 2;
    } else if ((b0 & 0xF0) == 0xE0) {
        cp = b0 & 0x0F;
        len = 3;
    } else if ((b0 & 0xF8) == 0xF0) {
        cp = b0 & 0x07;
        len = 4;
    } else {
        // Stray continuation byte or invalid lead byte
        i += 1;
        return 0xFFFD;
    }

    if (len > text.size() - i) {
        // Truncated sequence at end of string
        i = text.size();
        return 0xFFFD;
    }
    for (std::size_t k = 1; k < len; ++k) {
        const uint8_t bk = static_cast<uint8_t>(text[i + k]);
        if ((bk & 0xC0) != 0x80) {
            i += 1;
            return 0xFFFD;
        }
        cp = (cp << 6) | (bk & 0x3F);
    }
    i += len;
    return cp;
}

} // namespace bracken::core
