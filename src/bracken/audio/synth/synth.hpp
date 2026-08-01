#pragma once

#include "bracken/audio/audio_format.hpp"
#include "bracken/audio/synth/instrument.hpp"
#include "bracken/audio/synth/voice.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace bracken::audio::synth {

/** Voices are a fixed array, never a vector: the audio thread must not
 *  allocate, and an unbounded polyphony is a way to turn a stuck note into a
 *  CPU spiral rather than a quiet mix. 32 is far past what a game score uses
 *  at once and still trivially cheap. */
inline constexpr int k_max_voices = 32;

/**
 * A polyphonic synthesizer: note events in, stereo frames out.
 *
 * Entirely headless and device-free — it is a pure function of the note
 * events it has been given, which is what makes it unit-testable and what
 * lets the playground's music be rendered offline and compared byte-for-byte
 * in a test.
 *
 * Tier: audio-thread for render()/note_on()/note_off(). The owning
 * AudioEngine is what marshals note events across from the game.
 */
class Synth {
public:
    void note_on(const Instrument& instrument, uint8_t note, float velocity = 1.0f) noexcept {
        Voice* v = allocate_voice(note);
        if (v != nullptr) {
            v->note_on(instrument, note, velocity);
        }
    }

    /** Release the newest still-held voice for this note. Newest rather than
     *  any, so overlapping repeats of the same pitch release in the order a
     *  player would expect. */
    void note_off(uint8_t note) noexcept {
        for (int i = k_max_voices - 1; i >= 0; --i) {
            Voice& v = voices_[static_cast<std::size_t>(i)];
            if (v.active() && v.note() == note && !v.releasing()) {
                v.note_off();
                return;
            }
        }
    }

    /** Release everything that is still held. */
    void all_notes_off() noexcept {
        for (Voice& v : voices_) {
            if (v.active()) {
                v.note_off();
            }
        }
    }

    /** Cut everything immediately — for a hard stop, where a release tail
     *  would be wrong (scene teardown, a skipped cutscene). */
    void panic() noexcept {
        for (Voice& v : voices_) {
            v.kill();
        }
    }

    [[nodiscard]] int active_voices() const noexcept {
        int n = 0;
        for (const Voice& v : voices_) {
            n += v.active() ? 1 : 0;
        }
        return n;
    }

    /** Sum every voice into `out`, ADDING rather than overwriting so several
     *  synths (or a synth and sampled audio) can share one bus. */
    void render_additive(std::span<StereoFrame> out) noexcept {
        for (Voice& v : voices_) {
            if (!v.active()) {
                continue;
            }
            for (StereoFrame& f : out) {
                f += v.next();
            }
        }
    }

private:
    // Free voice if there is one; otherwise steal. Stealing order matters
    // more than it looks: taking the quietest RELEASING voice first means a
    // dense passage drops reverb tails rather than notes the player is
    // actually hearing.
    [[nodiscard]] Voice* allocate_voice(uint8_t note) noexcept {
        for (Voice& v : voices_) {
            if (!v.active()) {
                return &v;
            }
        }

        Voice* best = nullptr;
        float best_level = 2.0f;
        for (Voice& v : voices_) {
            if (!v.releasing()) {
                continue;
            }
            if (v.level() < best_level) {
                best_level = v.level();
                best = &v;
            }
        }
        if (best != nullptr) {
            return best;
        }

        // Everything is held. Prefer re-using the same pitch (a retrigger),
        // else take the quietest.
        for (Voice& v : voices_) {
            if (v.note() == note) {
                return &v;
            }
        }
        best_level = 2.0f;
        for (Voice& v : voices_) {
            if (v.level() < best_level) {
                best_level = v.level();
                best = &v;
            }
        }
        return best;
    }

    std::array<Voice, k_max_voices> voices_{};
};

} // namespace bracken::audio::synth
