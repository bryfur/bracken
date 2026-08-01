#pragma once

#include "bracken/audio/audio_format.hpp"
#include "bracken/audio/synth/instrument.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace bracken::audio::music {

/**
 * One cell of a pattern. A tracker row, essentially: what to play and for how
 * long, or nothing at all.
 */
struct Step {
    uint8_t note = k_note_rest; // k_note_rest = keep silent / let the previous note ring
    float velocity = 1.0f;
    /** How many steps the note is held before release. 0 means "until the
     *  next note on this track", which is what you want for a legato pad and
     *  is a nuisance to express any other way. */
    uint8_t gate_steps = 0;
};

/** A bar (or several) of steps for a single voice. Patterns are referenced by
 *  index from a track's arrangement so a repeated section is stored once. */
struct Pattern {
    std::string name;
    std::vector<Step> steps;
};

/**
 * One monophonic line: an instrument plus the order its patterns play in.
 *
 * Monophonic per track is a deliberate limit — chords are written as several
 * tracks sharing an instrument. It keeps note-off bookkeeping to a single
 * slot per track, which is what lets the sequencer run on the audio thread
 * without allocating.
 */
struct Track {
    std::string name;
    synth::Instrument instrument;
    std::vector<Pattern> patterns;
    /** Indices into `patterns`, in play order. An empty arrangement plays
     *  pattern 0 forever, which is the common case while sketching. */
    std::vector<uint32_t> arrangement;
    float gain = 1.0f;
    bool muted = false;
};

/**
 * A complete piece. Plain data with no device or engine coupling: a host can
 * build one in code (as the playground does), generate it, or load it from
 * its own format.
 */
struct Song {
    std::string name;
    float bpm = 100.0f;
    /** Steps per beat. 4 = sixteenth notes at 4/4, the usual tracker grid. */
    int steps_per_beat = 4;
    std::vector<Track> tracks;
    bool loop = true;

    [[nodiscard]] int frames_per_step() const noexcept {
        const float steps_per_second = (bpm / 60.0f) * static_cast<float>(steps_per_beat);
        return steps_per_second > 0.0f
                   ? static_cast<int>(static_cast<float>(k_sample_rate) / steps_per_second + 0.5f)
                   : k_sample_rate;
    }

    /** Length of the longest track in steps — the point at which the song
     *  loops or stops. Tracks shorter than this simply run out and stay
     *  silent, so a one-bar drum loop under an eight-bar melody needs no
     *  padding. */
    [[nodiscard]] int length_steps() const noexcept {
        int longest = 0;
        for (const Track& t : tracks) {
            int n = 0;
            if (t.arrangement.empty()) {
                n = t.patterns.empty() ? 0 : static_cast<int>(t.patterns[0].steps.size());
            } else {
                for (uint32_t idx : t.arrangement) {
                    if (idx < t.patterns.size()) {
                        n += static_cast<int>(t.patterns[idx].steps.size());
                    }
                }
            }
            longest = n > longest ? n : longest;
        }
        return longest;
    }
};

} // namespace bracken::audio::music
