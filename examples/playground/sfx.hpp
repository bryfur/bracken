#pragma once

#if BRACKEN_SYNTH

#include "bracken/audio/audio_engine.hpp"

namespace playground {

/**
 * The village's sound effects — generated, not loaded.
 *
 * Every clip here is baked at boot from a synthesizer patch (see
 * `bracken::audio::synth::render_note`), which is why this repo still contains
 * no audio assets. A footstep is a filtered noise burst with a fast envelope;
 * a splash is the same thing wetter and longer. Keeping them as code means
 * they diff, they version, and they are tweakable without an editor.
 */
struct SfxBank {
    bracken::audio::SoundId footstep_grass{};
    bracken::audio::SoundId footstep_water{};
    bracken::audio::SoundId orb_pickup{};
    bracken::audio::SoundId splash{};

    /** Synthesize every clip and register it. Main thread, once, at init. */
    void build(bracken::audio::AudioEngine& audio);
};

} // namespace playground

#endif // BRACKEN_SYNTH
