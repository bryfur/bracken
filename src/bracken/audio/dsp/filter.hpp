#pragma once

#include "bracken/audio/audio_format.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace bracken::audio::dsp {

enum class FilterMode : uint8_t {
    Off,
    LowPass,
    HighPass,
    BandPass,
};

/**
 * Chamberlin state-variable filter — 12 dB/octave, all three responses
 * available from the same two integrators.
 *
 * Chosen over a biquad because the cutoff can be swept per-sample without
 * recomputing coefficients, which is what makes an envelope-driven filter
 * (the thing that makes a synth patch sound like an instrument rather than a
 * buzzer) cheap. The tradeoff is a stability limit: the topology blows up as
 * cutoff approaches a quarter of the sample rate, so `set_cutoff` clamps well
 * below Nyquist rather than trusting callers.
 */
class StateVariableFilter {
public:
    void set_mode(FilterMode m) noexcept { mode_ = m; }
    [[nodiscard]] FilterMode mode() const noexcept { return mode_; }

    void set_cutoff(float hz) noexcept {
        // Hard ceiling at ~1/5 of the rate: past roughly a quarter the
        // Chamberlin form self-oscillates into a NaN and takes the whole
        // mix bus with it. 20 Hz floor keeps f_ from underflowing to 0.
        constexpr float k_max_hz = static_cast<float>(k_sample_rate) * 0.2f;
        const float clamped = std::clamp(hz, 20.0f, k_max_hz);
        f_ = 2.0f * std::sin(std::numbers::pi_v<float> * clamped / static_cast<float>(k_sample_rate));
    }

    /** Resonance as 0..1, mapped to the topology's damping term. 1.0 sits
     *  just below self-oscillation. */
    void set_resonance(float r) noexcept {
        q_ = 1.0f - std::clamp(r, 0.0f, 0.98f);
    }

    void reset() noexcept {
        low_ = 0.0f;
        band_ = 0.0f;
    }

    [[nodiscard]] float process(float in) noexcept {
        if (mode_ == FilterMode::Off) {
            return in;
        }

        const float high = in - low_ - q_ * band_;
        band_ += f_ * high;
        low_ += f_ * band_;

        switch (mode_) {
        case FilterMode::LowPass: return low_;
        case FilterMode::HighPass: return high;
        case FilterMode::BandPass: return band_;
        case FilterMode::Off: break;
        }
        return in;
    }

private:
    FilterMode mode_ = FilterMode::Off;
    float f_ = 0.5f;
    float q_ = 0.5f;
    float low_ = 0.0f;
    float band_ = 0.0f;
};

} // namespace bracken::audio::dsp
