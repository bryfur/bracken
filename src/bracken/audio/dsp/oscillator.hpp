#pragma once

#include "bracken/audio/audio_format.hpp"

#include <cmath>
#include <cstdint>
#include <numbers>

namespace bracken::audio::dsp {

enum class Waveform : uint8_t {
    Sine,
    Triangle,
    Saw,
    Square,
    Noise,
};

/**
 * A single-voice oscillator.
 *
 * Saw and square are band-limited with PolyBLEP rather than generated naively.
 * That is not polish: a naive saw is a sum of harmonics extending past
 * Nyquist, and those fold back down as inharmonic tones, so a rising melody
 * line acquires a descending metallic shimmer underneath it. PolyBLEP
 * subtracts a polynomial approximation of the band-limited step around each
 * discontinuity, which removes most of that for a couple of arithmetic ops per
 * sample — cheap enough to leave on always.
 *
 * Phase is kept in [0,1) rather than radians so the discontinuity tests are
 * plain comparisons against the per-sample increment.
 */
class Oscillator {
public:
    void set_waveform(Waveform w) noexcept { waveform_ = w; }
    [[nodiscard]] Waveform waveform() const noexcept { return waveform_; }

    void set_frequency(float hz) noexcept {
        // Guard the increment rather than the input: a zero or negative
        // frequency is a legitimate thing for a modulated patch to reach
        // momentarily, and it should go silent, not produce NaN.
        increment_ = hz > 0.0f ? hz / static_cast<float>(k_sample_rate) : 0.0f;
    }

    /** Restart the waveform. Called on note-on so repeated notes are
     *  identical rather than starting wherever the last one left off. */
    void reset_phase(float phase01 = 0.0f) noexcept { phase_ = phase01; }

    /** Advance one frame and return the sample, nominally in [-1, 1]. */
    [[nodiscard]] float next() noexcept {
        if (waveform_ == Waveform::Noise) {
            return next_noise();
        }

        const float t = phase_;
        const float dt = increment_;
        float out = 0.0f;

        switch (waveform_) {
        case Waveform::Sine:
            out = std::sin(t * 2.0f * std::numbers::pi_v<float>);
            break;
        case Waveform::Triangle:
            // Integrating a square would need state; the direct form is
            // exact and triangle's harmonics roll off fast enough (1/n^2)
            // that aliasing is inaudible without band-limiting.
            out = 4.0f * std::fabs(t - 0.5f) - 1.0f;
            break;
        case Waveform::Saw:
            out = 2.0f * t - 1.0f;
            out -= poly_blep(t, dt);
            break;
        case Waveform::Square:
            out = t < 0.5f ? 1.0f : -1.0f;
            // Two discontinuities per cycle: the wrap at 0 and the flip at
            // 0.5. Correcting only one leaves half the aliasing.
            out += poly_blep(t, dt);
            out -= poly_blep(std::fmod(t + 0.5f, 1.0f), dt);
            break;
        case Waveform::Noise:
            break; // handled above
        }

        phase_ += dt;
        if (phase_ >= 1.0f) {
            phase_ -= 1.0f;
        }
        return out;
    }

private:
    // Polynomial band-limited step. Within one sample either side of a
    // discontinuity, blend in the residual between the ideal step and the
    // sampled one.
    [[nodiscard]] static float poly_blep(float t, float dt) noexcept {
        if (dt <= 0.0f) {
            return 0.0f;
        }
        if (t < dt) {
            const float x = t / dt;
            return x + x - x * x - 1.0f;
        }
        if (t > 1.0f - dt) {
            const float x = (t - 1.0f) / dt;
            return x * x + x + x + 1.0f;
        }
        return 0.0f;
    }

    // xorshift32: deterministic per-voice noise. Deliberately NOT a shared or
    // thread_local generator — the effect system already learned that lesson
    // (see systems/effect_system.cpp): a shared stream makes output depend on
    // which thread ran, and puts it outside any regression gate.
    [[nodiscard]] float next_noise() noexcept {
        noise_state_ ^= noise_state_ << 13;
        noise_state_ ^= noise_state_ >> 17;
        noise_state_ ^= noise_state_ << 5;
        // Top 24 bits → [-1, 1); exactly representable in a float mantissa.
        return static_cast<float>(noise_state_ >> 8) * (2.0f / 16777216.0f) - 1.0f;
    }

    Waveform waveform_ = Waveform::Sine;
    float phase_ = 0.0f;
    float increment_ = 0.0f;
    uint32_t noise_state_ = 0x1234567u;
};

} // namespace bracken::audio::dsp
