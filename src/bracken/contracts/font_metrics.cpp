#include "bracken/contracts/font_metrics.hpp"

#include <algorithm>

namespace bracken {

namespace {

/// Binary search over a sorted (key, value) vector. Sorted vectors rather
/// than hash maps throughout this file: layout must be a pure, deterministic
/// function of the table, and an unordered container's iteration order is
/// neither pinned nor reproducible across libstdc++ versions.
template<typename Key, typename Value>
[[nodiscard]] const Value* lookup(const std::vector<std::pair<Key, Value>>& sorted, Key key) noexcept {
    const auto it = std::lower_bound(sorted.begin(), sorted.end(), key,
                                     [](const std::pair<Key, Value>& e, Key k) { return e.first < k; });
    if (it == sorted.end() || it->first != key) {
        return nullptr;
    }
    return &it->second;
}

} // namespace

bool FontMetrics::has_advance(uint32_t cp) const noexcept {
    if (!valid()) {
        return false;
    }
    if (cp >= k_first_cp && cp <= k_last_cp) {
        return advance[cp - k_first_cp] != k_absent;
    }
    return lookup(sparse_advance, cp) != nullptr;
}

int32_t FontMetrics::advance_of(uint32_t cp) const noexcept {
    if (!valid()) {
        return 0;
    }
    if (cp >= k_first_cp && cp <= k_last_cp) {
        const int16_t a = advance[cp - k_first_cp];
        return a == k_absent ? fallback_advance : static_cast<int32_t>(a);
    }
    if (const int16_t* a = lookup(sparse_advance, cp); a != nullptr) {
        return static_cast<int32_t>(*a);
    }
    return fallback_advance;
}

int32_t FontMetrics::kerning_of(uint32_t prev, uint32_t cur) const noexcept {
    if (!valid() || kerning.empty()) {
        return 0;
    }
    const uint64_t key = (static_cast<uint64_t>(prev) << 32) | static_cast<uint64_t>(cur);
    if (const int16_t* k = lookup(kerning, key); k != nullptr) {
        return static_cast<int32_t>(*k);
    }
    return 0;
}

float FontMetrics::measure_width(std::string_view utf8, float scale) const {
    if (!valid() || utf8.empty()) {
        return 0.0f;
    }
    // The authoritative pen walk, mirrored from
    // render/text_renderer.cpp build_string_quads: kerning BEFORE the glyph,
    // then the glyph's advance. Accumulated in integer font pixels and scaled
    // exactly once, so the result cannot drift with string length.
    int32_t pen = 0;
    uint32_t prev = 0;
    for (std::size_t i = 0; i < utf8.size();) {
        const uint32_t cp = core::decode_utf8(utf8, i);
        if (prev != 0) {
            pen += kerning_of(prev, cp);
        }
        pen += advance_of(cp);
        prev = cp;
    }
    return static_cast<float>(pen) * scale;
}

uint32_t FontMetrics::count_fallbacks(std::string_view utf8) const {
    if (!valid()) {
        return 0;
    }
    uint32_t n = 0;
    for (std::size_t i = 0; i < utf8.size();) {
        if (!has_advance(core::decode_utf8(utf8, i))) {
            ++n;
        }
    }
    return n;
}

std::size_t FontMetrics::fit_prefix(std::string_view utf8, float max_width, float scale, float* out_width) const {
    if (out_width != nullptr) {
        *out_width = 0.0f;
    }
    if (!valid() || utf8.empty() || max_width <= 0.0f || scale <= 0.0f) {
        return 0;
    }
    // Walk codepoints and commit the prefix only when the WHOLE glyph fits;
    // the byte index after the last committed codepoint is the answer, so a
    // multi-byte sequence is never cut in half.
    int32_t pen = 0;
    uint32_t prev = 0;
    std::size_t fit_bytes = 0;
    float fit_width = 0.0f;
    for (std::size_t i = 0; i < utf8.size();) {
        const uint32_t cp = core::decode_utf8(utf8, i);
        int32_t next = pen;
        if (prev != 0) {
            next += kerning_of(prev, cp);
        }
        next += advance_of(cp);
        const float w = static_cast<float>(next) * scale;
        if (w > max_width) {
            // `i` has already advanced past the rejected codepoint, which is
            // exactly why the committed answer is `fit_bytes` and not `i`.
            break;
        }
        pen = next;
        fit_bytes = i;
        fit_width = w;
        prev = cp;
    }
    if (out_width != nullptr) {
        *out_width = fit_width;
    }
    return fit_bytes;
}

FontMetrics::WrapPoint FontMetrics::wrap_point(std::string_view utf8, float max_width, float scale) const {
    if (utf8.empty()) {
        return {};
    }
    if (!valid()) {
        // No font: one line, everything on it. Refusing to wrap is the honest
        // answer when there are no metrics — a guess is what this replaced.
        return {utf8.size(), utf8.size()};
    }
    const std::size_t fit = fit_prefix(utf8, max_width, scale);
    if (fit >= utf8.size()) {
        return {utf8.size(), utf8.size()};
    }
    if (fit == 0) {
        // Not even one glyph fits. Emit one codepoint anyway rather than
        // looping forever on a box narrower than a character.
        std::size_t i = 0;
        (void)core::decode_utf8(utf8, i);
        return {i, i};
    }
    // Prefer the last space inside the fitting prefix. `fit` is a boundary, so
    // searching [0, fit) cannot land inside a multi-byte sequence.
    const std::size_t space = utf8.rfind(' ', fit - 1);
    if (space != std::string_view::npos && space > 0) {
        return {space + 1, space}; // swallow the break space, draw up to it
    }
    if (space == 0) {
        return {1, 0}; // leading space: consume it, draw nothing
    }
    return {fit, fit}; // no space to break on — hard break at the fit boundary
}

} // namespace bracken
