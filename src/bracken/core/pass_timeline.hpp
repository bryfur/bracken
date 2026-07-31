#pragma once

#include <array>
#include <chrono>
#include <cstddef>

namespace bracken::core {

/**
 * Fixed-capacity CPU wall-clock timeline for named, possibly nested scopes.
 *
 * What it measures is the cost of RECORDING a pass — building and submitting
 * its command stream on the CPU. It is emphatically NOT GPU execution time:
 * SDL3 exposes no timestamp query API, so nothing in this engine can produce
 * a real GPU-side number (see gpu/gpu_debug.hpp). The distinction has to
 * survive all the way to the UI, which is why the value is surfaced as
 * `PassTiming::cpu_ms` under a column labelled "CPU (record)" while
 * `gpu_ms` stays nullopt.
 *
 * Slot discipline: slots are indexed by the caller (the timestamp pool's pair
 * index), so a pass keeps the same slot for as long as the frame's pass
 * ORDER is stable — which is what makes the exponential moving average
 * meaningful. When a slot's name pointer changes (a conditional pass came or
 * went and shifted the ordering) that slot's average is discarded rather than
 * blended across two different passes. Names are compared BY POINTER: every
 * call site passes a string literal, and two different literals are two
 * different passes even if a compiler happened to pool identical spellings.
 *
 * No allocation, no atomics: this is main-thread render-recording state. All
 * of it is a couple of `steady_clock::now()` calls per pass, which is why it
 * runs unconditionally rather than behind the stats flag.
 */
class PassTimeline {
public:
    /// Matches gpu::GPUTimestampPool's pair capacity; passes beyond this are
    /// simply untimed rather than a failure.
    static constexpr std::size_t k_max_passes = 64;

    /// New frame: forget which slots are live (averages are retained).
    void begin_frame() noexcept;

    /// Bind `name` to `idx` for this frame. A name change resets the slot's
    /// moving average. Out-of-range indices are ignored.
    void reserve(std::size_t idx, const char* name) noexcept;

    /// Start the clock for `idx` (re-entrant starts just restart it).
    void begin_pass(std::size_t idx) noexcept;

    /// Stop the clock for `idx` and fold the sample into its average.
    void end_pass(std::size_t idx) noexcept;

    /// Smoothed record time of slot `idx`, in milliseconds. Zero for a slot
    /// that has never completed a pass.
    [[nodiscard]] float pass_ms(std::size_t idx) const noexcept;

    /// True once slot `idx` has produced at least one completed sample.
    [[nodiscard]] bool has_sample(std::size_t idx) const noexcept;

private:
    // Weight of the newest sample in the moving average. 0.2 settles in
    // ~15 frames — fast enough to show a regression while you are looking at
    // it, slow enough that a single hitching frame does not own the readout.
    static constexpr float k_ema_alpha = 0.2f;

    struct Slot {
        const char* name = nullptr;
        std::chrono::steady_clock::time_point start{};
        float ema_ms = 0.0f;
        bool open = false;
        bool primed = false;
    };

    std::array<Slot, k_max_passes> slots_{};
};

} // namespace bracken::core
