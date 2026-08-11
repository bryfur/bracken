#include "bracken/ui/ui_types.hpp"

namespace bracken::ui {

namespace {

/// One round of FNV-1a 64 over eight bytes. Used to fold ids into each other
/// so that combining is associative-free but stable: the SAME parent/child
/// pair always yields the same value, on every platform, at every worker
/// count. Deliberately not std::hash — libstdc++'s is not specified to be
/// stable across versions, and a widget id that moves between compiler
/// releases loses every player's scroll position.
[[nodiscard]] constexpr uint64_t fold(uint64_t h, uint64_t v) noexcept {
    for (int byte = 0; byte < 8; ++byte) {
        h ^= (v >> (byte * 8)) & 0xFFull;
        h *= core::k_fnv1a_prime_64;
    }
    return h;
}

} // namespace

WidgetId id(std::string_view label) noexcept {
    if (label.empty()) {
        return {};
    }
    const uint64_t h = core::fnv1a64(label);
    // 0 is the invalid sentinel; map the astronomically unlikely collision to
    // 1 rather than letting a real widget claim "no id".
    return WidgetId{h == 0 ? 1ull : h};
}

WidgetId id(WidgetId parent, uint64_t index) noexcept {
    const uint64_t h = fold(parent.valid() ? parent.value : core::k_fnv1a_offset_basis_64, index);
    return WidgetId{h == 0 ? 1ull : h};
}

WidgetId combine(WidgetId parent, WidgetId child) noexcept {
    if (!parent.valid()) {
        return child;
    }
    if (!child.valid()) {
        return parent;
    }
    const uint64_t h = fold(parent.value, child.value);
    return WidgetId{h == 0 ? 1ull : h};
}

} // namespace bracken::ui
