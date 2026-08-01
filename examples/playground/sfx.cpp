#include "sfx.hpp"

#if BRACKEN_SYNTH

#include "bracken/audio/synth/render.hpp"

namespace playground {
namespace {

using namespace bracken::audio;
using bracken::audio::synth::Instrument;

/** A footstep: a short noise burst, band-limited by a filter to suggest what
 *  is underfoot. Dry grass is mid-heavy and very short. */
Instrument grass_step_patch() {
    Instrument i;
    i.name = "footstep_grass";
    i.osc1 = dsp::Waveform::Noise;
    i.osc2_mix = 0.0f;
    i.amp = {.attack_s = 0.001f, .decay_s = 0.06f, .sustain = 0.0f, .release_s = 0.04f};
    i.filter = dsp::FilterMode::BandPass;
    i.cutoff_hz = 1500.0f;
    i.resonance = 0.35f;
    // Sweeping the band down over the burst is what turns a click into a
    // "crunch" — the spectral movement is the whole cue.
    i.filter_env_amount = 1.6f;
    i.gain = 0.5f;
    return i;
}

/** Wet footfall: lower, longer, with more low end than the grass step. */
Instrument water_step_patch() {
    Instrument i = grass_step_patch();
    i.name = "footstep_water";
    i.amp = {.attack_s = 0.002f, .decay_s = 0.12f, .sustain = 0.0f, .release_s = 0.10f};
    i.cutoff_hz = 700.0f;
    i.resonance = 0.2f;
    i.filter_env_amount = 2.4f;
    i.gain = 0.55f;
    return i;
}

/** A bigger splash for entering the river — same idea, more of it. */
Instrument splash_patch() {
    Instrument i;
    i.name = "splash";
    i.osc1 = dsp::Waveform::Noise;
    i.osc2_mix = 0.0f;
    i.amp = {.attack_s = 0.004f, .decay_s = 0.35f, .sustain = 0.0f, .release_s = 0.25f};
    i.filter = dsp::FilterMode::LowPass;
    i.cutoff_hz = 900.0f;
    i.resonance = 0.4f;
    i.filter_env_amount = 3.0f; // opens bright then closes: water settling
    i.gain = 0.6f;
    return i;
}

/** Orb pickup: a bright pitched blip. Tonal rather than noisy so it reads as
 *  a reward against a scene full of broadband ambience. */
Instrument pickup_patch() {
    Instrument i;
    i.name = "orb_pickup";
    i.osc1 = dsp::Waveform::Triangle;
    i.osc2 = dsp::Waveform::Sine;
    i.osc2_mix = 0.45f;
    i.osc2_octave = 1; // an octave above: sparkle
    i.amp = {.attack_s = 0.002f, .decay_s = 0.28f, .sustain = 0.0f, .release_s = 0.18f};
    i.filter = dsp::FilterMode::LowPass;
    i.cutoff_hz = 4000.0f;
    i.gain = 0.5f;
    return i;
}

} // namespace

void SfxBank::build(AudioEngine& audio) {
    // Noise patches ignore the note number, so pitch only matters for the
    // pickup; A4 elsewhere is a placeholder.
    footstep_grass = audio.register_clip(
        synth::render_note(grass_step_patch(), A4, 0.02f, 0.12f, "footstep_grass"));
    footstep_water = audio.register_clip(
        synth::render_note(water_step_patch(), A4, 0.03f, 0.22f, "footstep_water"));
    splash = audio.register_clip(
        synth::render_note(splash_patch(), A4, 0.06f, 0.60f, "splash"));
    orb_pickup = audio.register_clip(
        synth::render_note(pickup_patch(), A5, 0.05f, 0.45f, "orb_pickup"));
}

} // namespace playground

#endif // BRACKEN_SYNTH
