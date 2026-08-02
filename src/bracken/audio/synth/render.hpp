#pragma once


#include "bracken/audio/audio_clip.hpp"
#include "bracken/audio/audio_format.hpp"
#include "bracken/audio/synth/synth.hpp"

#include <string>
#include <vector>

namespace bracken::audio::synth {

/**
 * Render one note of an instrument to a clip, offline.
 *
 * Turns the synthesizer into a sound-effect *generator*: a game can bake its
 * SFX at boot from patches instead of shipping WAVs. That is not a novelty —
 * a noise oscillator through a filter and a fast envelope is what a footstep,
 * a UI blip and a splash actually are, and generating them keeps them
 * tweakable in code, versionable as a diff, and free of binary assets.
 *
 * `hold_seconds` is how long the note is held before release; the render then
 * continues for `tail_seconds` so the release is captured rather than cut off.
 * A clip that ends mid-release clicks on every playback.
 *
 * Deterministic: the same instrument and note always produce identical
 * samples, so baked SFX are stable across runs and machines.
 *
 * Tier: caller-owned. Do this at load time, not per frame.
 */
[[nodiscard]] inline AudioClip render_note(const Instrument& instrument, uint8_t note,
                                           float hold_seconds, float tail_seconds,
                                           std::string name) {
    const int hold_frames = frames_from_seconds(hold_seconds);
    const int tail_frames = frames_from_seconds(tail_seconds);

    Synth s;
    s.note_on(instrument, note);

    std::vector<StereoFrame> frames(static_cast<std::size_t>(hold_frames + tail_frames));
    if (hold_frames > 0) {
        s.render_additive(std::span<StereoFrame>(frames).subspan(0, static_cast<std::size_t>(hold_frames)));
    }
    s.note_off(note);
    if (tail_frames > 0) {
        s.render_additive(std::span<StereoFrame>(frames).subspan(
            static_cast<std::size_t>(hold_frames), static_cast<std::size_t>(tail_frames)));
    }

    return AudioClip::from_samples(std::move(frames), std::move(name));
}

} // namespace bracken::audio::synth

