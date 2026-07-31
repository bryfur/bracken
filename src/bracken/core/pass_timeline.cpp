#include "bracken/core/pass_timeline.hpp"

namespace bracken::core {

void PassTimeline::begin_frame() noexcept {
    for (auto& s : slots_) {
        s.open = false;
    }
}

void PassTimeline::reserve(std::size_t idx, const char* name) noexcept {
    if (idx >= k_max_passes) {
        return;
    }
    Slot& s = slots_[idx];
    if (s.name != name) {
        // A different pass now owns this slot — averaging across the two
        // would report a number that describes neither.
        s.name = name;
        s.ema_ms = 0.0f;
        s.primed = false;
    }
}

void PassTimeline::begin_pass(std::size_t idx) noexcept {
    if (idx >= k_max_passes) {
        return;
    }
    Slot& s = slots_[idx];
    s.start = std::chrono::steady_clock::now();
    s.open = true;
}

void PassTimeline::end_pass(std::size_t idx) noexcept {
    if (idx >= k_max_passes) {
        return;
    }
    Slot& s = slots_[idx];
    if (!s.open) {
        return; // end without a begin (or after begin_frame): no sample
    }
    s.open = false;
    const auto delta = std::chrono::steady_clock::now() - s.start;
    const float ms = std::chrono::duration<float, std::milli>(delta).count();
    if (!s.primed) {
        s.ema_ms = ms; // first sample IS the average; no ramp from zero
        s.primed = true;
    } else {
        s.ema_ms += k_ema_alpha * (ms - s.ema_ms);
    }
}

float PassTimeline::pass_ms(std::size_t idx) const noexcept {
    return idx < k_max_passes ? slots_[idx].ema_ms : 0.0f;
}

bool PassTimeline::has_sample(std::size_t idx) const noexcept {
    return idx < k_max_passes && slots_[idx].primed;
}

} // namespace bracken::core
