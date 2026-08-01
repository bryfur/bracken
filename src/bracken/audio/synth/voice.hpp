#pragma once

#include "bracken/audio/audio_format.hpp"
#include "bracken/audio/dsp/envelope.hpp"
#include "bracken/audio/dsp/filter.hpp"
#include "bracken/audio/dsp/oscillator.hpp"
#include "bracken/audio/synth/instrument.hpp"

#include <cmath>
#include <cstdint>

namespace bracken::audio::synth {

/**
 * One sounding note: two oscillators, a filter and an amplitude envelope.
 *
 * A voice holds a COPY of the instrument it was started with, not a pointer.
 * A patch edited from the dev overlay while a note is ringing would otherwise
 * mutate the filter coefficients under the audio thread — a data race in the
 * one place in the engine where a race is guaranteed to be audible. Copying
 * ~80 bytes at note-on is the cheapest possible fix.
 *
 * Tier: audio-thread. Every method here runs in the device callback; none of
 * them allocate, lock, or log.
 */
class Voice {
public:
    void note_on(const Instrument& instrument, uint8_t note, float velocity) noexcept {
        instrument_ = instrument;
        note_ = note;
        velocity_ = velocity;
        active_ = true;

        // Deterministic per-note detune: same note always drifts the same
        // way, so a song renders identically on every run and stays inside
        // the regression gate. A random source here would not.
        const float jitter = instrument_.pitch_jitter_cents *
                             (static_cast<float>((note * 2654435761u) >> 24) / 128.0f - 1.0f);

        base_hz_ = hz_from_midi(static_cast<float>(note) + jitter / 100.0f);

        osc1_.set_waveform(instrument_.osc1);
        osc1_.set_frequency(base_hz_);
        osc1_.reset_phase(0.0f);

        osc2_.set_waveform(instrument_.osc2);
        osc2_.set_frequency(osc2_hz());
        // Quarter-cycle offset: starting both oscillators at phase 0 makes
        // the attack transient of a detuned pair sum to a single louder
        // click before they drift apart.
        osc2_.reset_phase(0.25f);

        filter_.set_mode(instrument_.filter);
        filter_.set_resonance(instrument_.resonance);
        filter_.reset();

        env_.configure(instrument_.amp);
        env_.note_on();
    }

    void note_off() noexcept { env_.note_off(); }

    /** Reclaim immediately, for voice stealing. */
    void kill() noexcept {
        env_.hard_reset();
        active_ = false;
    }

    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] uint8_t note() const noexcept { return note_; }
    [[nodiscard]] bool releasing() const noexcept {
        return env_.stage() == dsp::Envelope::Stage::Release;
    }
    /** Current envelope level — the voice stealer prefers quiet voices. */
    [[nodiscard]] float level() const noexcept { return env_.level(); }

    /** Render one frame, panned. Returns silence when inactive. */
    [[nodiscard]] StereoFrame next() noexcept {
        if (!active_) {
            return {};
        }

        const float amp = env_.next();
        if (env_.finished()) {
            active_ = false;
            return {};
        }

        float s = osc1_.next() * (1.0f - instrument_.osc2_mix);
        if (instrument_.osc2_mix > 0.0f) {
            s += osc2_.next() * instrument_.osc2_mix;
        }

        if (instrument_.filter != dsp::FilterMode::Off) {
            // Envelope-tracked cutoff, in octaves so the sweep sounds the
            // same at every pitch.
            const float sweep = instrument_.filter_env_amount * amp;
            filter_.set_cutoff(instrument_.cutoff_hz * std::exp2(sweep));
            s = filter_.process(s);
        }

        s *= amp * velocity_ * instrument_.gain;

        // Constant-power pan: a centered voice keeps the same perceived
        // loudness as a hard-panned one, which linear panning does not.
        const float p = (instrument_.pan + 1.0f) * 0.5f;
        const float angle = p * (std::numbers::pi_v<float> * 0.5f);
        return {s * std::cos(angle), s * std::sin(angle)};
    }

private:
    [[nodiscard]] float osc2_hz() const noexcept {
        const float cents = instrument_.osc2_detune_cents / 1200.0f;
        return base_hz_ * std::exp2(static_cast<float>(instrument_.osc2_octave) + cents);
    }

    Instrument instrument_{};
    dsp::Oscillator osc1_;
    dsp::Oscillator osc2_;
    dsp::StateVariableFilter filter_;
    dsp::Envelope env_;

    float base_hz_ = 440.0f;
    float velocity_ = 1.0f;
    uint8_t note_ = 0;
    bool active_ = false;
};

} // namespace bracken::audio::synth
