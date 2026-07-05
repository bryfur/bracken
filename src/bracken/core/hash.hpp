#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>

namespace bracken::core {

constexpr uint32_t k_fnv1a_offset_basis_32 = 0x811c9dc5u;
constexpr uint32_t k_fnv1a_prime_32 = 0x01000193u;
constexpr uint64_t k_fnv1a_offset_basis_64 = 0xcbf29ce484222325ull;
constexpr uint64_t k_fnv1a_prime_64 = 0x100000001b3ull;

constexpr uint32_t fnv1a(std::string_view s) noexcept {
    uint32_t h = k_fnv1a_offset_basis_32;
    for (char c : s) {
        h ^= static_cast<uint32_t>(static_cast<unsigned char>(c));
        h *= k_fnv1a_prime_32;
    }
    return h;
}

constexpr uint64_t fnv1a64(std::string_view s) noexcept {
    uint64_t h = k_fnv1a_offset_basis_64;
    for (char c : s) {
        h ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
        h *= k_fnv1a_prime_64;
    }
    return h;
}

struct StringId {
    uint32_t value = 0;

    constexpr StringId() = default;
    constexpr explicit StringId(uint32_t v) noexcept : value(v) {}
    constexpr StringId(std::string_view s) noexcept : value(fnv1a(s)) {}

    constexpr friend bool operator==(StringId a, StringId b) noexcept { return a.value == b.value; }
    constexpr friend bool operator!=(StringId a, StringId b) noexcept { return a.value != b.value; }
    constexpr friend bool operator<(StringId a, StringId b) noexcept { return a.value < b.value; }
};

constexpr StringId operator""_sid(const char* s, std::size_t n) noexcept {
    return {std::string_view(s, n)};
}

} // namespace bracken::core

namespace std {
template<> struct hash<::bracken::core::StringId> {
    size_t operator()(::bracken::core::StringId id) const noexcept { return static_cast<size_t>(id.value); }
};
} // namespace std
