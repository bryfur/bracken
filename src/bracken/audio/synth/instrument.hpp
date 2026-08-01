#pragma once

#include "bracken/audio/audio_format.hpp"
#include "bracken/audio/dsp/envelope.hpp"
#include "bracken/audio/dsp/filter.hpp"
#include "bracken/audio/dsp/oscillator.hpp"

#include <cstdint>
#include <string>

namespace bracken::audio::synth {

/**
 * A patch: everything needed to turn a note number into sound.
 *
 * Plain data, like `EffectDefinition` — a host builds these in code or loads
 * them from its own file format (parsing is the host's job; the engine's job
 * is to play what it is handed). Copying one is cheap and it holds no
 * device-side state, so an instrument can be built on any thread and handed
 * to the audio system by value.
 */
struct Instrument {
    std::string name;

    // --- Oscillators -------------------------------------------------------
    // Two, because one is a tone and two is an instrument: detuning a second
    // oscillator a few cents against the first produces the slow beating that
    // makes a pad sound wide, and dropping it an octave is how a bass gets
    // weight. `osc2_mix` at 0 makes this a one-oscillator patch at no cost.
    dsp::Waveform osc1 = dsp::Waveform::Saw;
    dsp::Waveform osc2 = dsp::Waveform::Saw;
    float osc2_mix = 0.0f;         // 0 = osc1 only, 1 = osc2 only
    float osc2_detune_cents = 0.0f; // ±cents against osc1
    int osc2_octave = 0;            // whole-octave offset for osc2

    // --- Amplitude envelope ------------------------------------------------
    dsp::Envelope::Settings amp{};

    // --- Filter ------------------------------------------------------------
    dsp::FilterMode filter = dsp::FilterMode::Off;
    float cutoff_hz = 8000.0f;
    float resonance = 0.0f;
    // The filter envelope reuses the amplitude envelope's shape scaled by
    // `filter_env_amount` octaves. A separate envelope would be more flexible
    // and twice the parameters; this covers the patches that actually get
    // written, and a host wanting more can layer two instruments.
    float filter_env_amount = 0.0f; // octaves of cutoff sweep, may be negative

    // --- Output ------------------------------------------------------------
    float gain = 0.5f;  // linear, applied per voice
    float pan = 0.0f;   // -1 left, 0 center, +1 right

    /** Per-note pitch drift in cents, applied deterministically from the note
     *  number. A tiny amount stops repeated notes sounding machine-stamped;
     *  it is not random, so a song renders identically every run. */
    float pitch_jitter_cents = 0.0f;
};

/** A few ready-made patches, so a host can make sound in one line before
 *  learning what any of the parameters mean. */
namespace patches {

[[nodiscard]] inline Instrument soft_pad() {
    Instrument i;
    i.name = "soft_pad";
    i.osc1 = dsp::Waveform::Saw;
    i.osc2 = dsp::Waveform::Saw;
    i.osc2_mix = 0.5f;
    i.osc2_detune_cents = 7.0f;
    i.amp = {.attack_s = 0.6f, .decay_s = 0.4f, .sustain = 0.75f, .release_s = 1.2f};
    i.filter = dsp::FilterMode::LowPass;
    i.cutoff_hz = 1400.0f;
    i.resonance = 0.15f;
    i.filter_env_amount = 1.2f;
    i.gain = 0.22f;
    return i;
}

[[nodiscard]] inline Instrument plucked_string() {
    Instrument i;
    i.name = "plucked_string";
    i.osc1 = dsp::Waveform::Saw;
    i.osc2 = dsp::Waveform::Triangle;
    i.osc2_mix = 0.35f;
    i.osc2_detune_cents = -4.0f;
    i.amp = {.attack_s = 0.004f, .decay_s = 0.45f, .sustain = 0.0f, .release_s = 0.25f};
    i.filter = dsp::FilterMode::LowPass;
    i.cutoff_hz = 2600.0f;
    i.resonance = 0.25f;
    i.filter_env_amount = 2.2f;
    i.gain = 0.30f;
    return i;
}

[[nodiscard]] inline Instrument sine_bass() {
    Instrument i;
    i.name = "sine_bass";
    i.osc1 = dsp::Waveform::Sine;
    i.osc2 = dsp::Waveform::Triangle;
    i.osc2_mix = 0.25f;
    i.osc2_octave = 0;
    i.amp = {.attack_s = 0.01f, .decay_s = 0.25f, .sustain = 0.55f, .release_s = 0.2f};
    i.filter = dsp::FilterMode::LowPass;
    i.cutoff_hz = 900.0f;
    i.gain = 0.40f;
    return i;
}

[[nodiscard]] inline Instrument reed() {
    Instrument i;
    i.name = "reed";
    i.osc1 = dsp::Waveform::Square;
    i.osc2 = dsp::Waveform::Saw;
    i.osc2_mix = 0.3f;
    i.osc2_detune_cents = 5.0f;
    i.amp = {.attack_s = 0.08f, .decay_s = 0.2f, .sustain = 0.6f, .release_s = 0.3f};
    i.filter = dsp::FilterMode::LowPass;
    i.cutoff_hz = 1800.0f;
    i.resonance = 0.3f;
    i.filter_env_amount = 1.0f;
    i.gain = 0.18f;
    return i;
}

/** Noise + fast envelope: a hi-hat / shaker, no pitch. */
[[nodiscard]] inline Instrument hat() {
    Instrument i;
    i.name = "hat";
    i.osc1 = dsp::Waveform::Noise;
    i.osc2_mix = 0.0f;
    i.amp = {.attack_s = 0.001f, .decay_s = 0.05f, .sustain = 0.0f, .release_s = 0.05f};
    i.filter = dsp::FilterMode::HighPass;
    i.cutoff_hz = 6000.0f;
    i.gain = 0.10f;
    return i;
}

} // namespace patches

} // namespace bracken::audio::synth
