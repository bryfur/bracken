#pragma once

#include <cstdint>

namespace bracken::core {

/**
 * A small, fast, deterministic random stream (SplitMix64).
 *
 * DETERMINISTIC BY SEED, NOT BY DERIVATION. The distinction matters, because
 * getting it wrong is how you end up with something that is reproducible and
 * also wrong. The tempting shortcut is to derive "randomness" from state you
 * already have — a position, a time, an index — which is reproducible, but the
 * value then correlates with that state: a footstep pitch derived from the
 * player's X coordinate is constant while the player runs along Z, so every
 * step sounds identical, which is the exact artifact the scatter existed to
 * prevent. An Rng seeded from a constant gives real variation AND repeats
 * exactly, which is what you actually want.
 *
 * Explicitly NOT seeded from a clock or std::random_device by default: a
 * stream that differs per run puts whatever it drives outside any regression
 * gate. A host that genuinely wants run-to-run variation should seed it
 * itself, deliberately.
 *
 * Not for cryptography. It is a game RNG: fast, tiny, well-distributed.
 *
 * Tier: owned. An Rng is a plain value with no shared state, so give each
 * system (or each emitter, or each thread) its own rather than sharing one —
 * a shared stream makes output depend on who drew from it first, which is
 * exactly how the effect system's old thread_local generator became
 * unreproducible at different worker counts.
 */
class Rng {
public:
    constexpr Rng() = default;
    constexpr explicit Rng(uint64_t seed) noexcept : state_(seed) {}

    /** Uniform in [0, 2^64). */
    constexpr uint64_t next_u64() noexcept {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    /** Uniform in [0, 1). Top 24 bits, which are exactly representable in a
     *  float mantissa, so the mapping carries no rounding surprise. */
    constexpr float next_unit() noexcept {
        return static_cast<float>(next_u64() >> 40) / 16777216.0f;
    }

    /** Uniform in [min, max). */
    constexpr float next_range(float min, float max) noexcept {
        return min + next_unit() * (max - min);
    }

    /** Uniform in [0, n). Returns 0 for n == 0 rather than dividing by it. */
    constexpr uint32_t next_below(uint32_t n) noexcept {
        return n == 0 ? 0u : static_cast<uint32_t>(next_u64() % n);
    }

    [[nodiscard]] constexpr uint64_t state() const noexcept { return state_; }
    constexpr void reseed(uint64_t seed) noexcept { state_ = seed; }

private:
    uint64_t state_ = 0x9E3779B97F4A7C15ULL;
};

} // namespace bracken::core
