#include "music.hpp"

#include "bracken/audio/audio_engine.hpp"
#include "bracken/audio/synth/instrument.hpp"
#include "bracken/audio/wav_writer.hpp"

#include <cstdint>
#include <vector>

#if BRACKEN_SYNTH

namespace playground {
namespace {

using namespace bracken::audio;
using bracken::audio::music::Pattern;
using bracken::audio::music::Song;
using bracken::audio::music::Step;
using bracken::audio::music::Track;

constexpr int k_steps_per_bar = 16; // 4/4 at sixteenth-note resolution
constexpr int k_bars = 4;
constexpr int k_song_steps = k_steps_per_bar * k_bars;

/** A note placed on the grid, written the way you would say it out loud:
 *  "A4, on step 8, held for a beat". */
struct Placement {
    int step = 0;
    uint8_t note = k_note_rest;
    uint8_t gate = 4;
    float velocity = 1.0f;
};

/** Build a full-length pattern of rests, then drop the notes in. Writing the
 *  sparse thing sparsely keeps the music readable as music; a literal 64-entry
 *  array of mostly-zero is not something anyone can proofread. */
Pattern lay(const char* name, const std::vector<Placement>& notes, int length = k_song_steps) {
    Pattern p;
    p.name = name;
    p.steps.assign(static_cast<std::size_t>(length), Step{});
    for (const Placement& n : notes) {
        if (n.step >= 0 && n.step < length) {
            p.steps[static_cast<std::size_t>(n.step)] =
                Step{.note = n.note, .velocity = n.velocity, .gate_steps = n.gate};
        }
    }
    return p;
}

Track make_track(const char* name, const synth::Instrument& instrument, Pattern pattern,
                 float gain) {
    Track t;
    t.name = name;
    t.instrument = instrument;
    t.patterns.push_back(std::move(pattern));
    t.gain = gain;
    return t;
}

// Bar starts, for readability below.
constexpr int b1 = 0;
constexpr int b2 = 16;
constexpr int b3 = 32;
constexpr int b4 = 48;

} // namespace

Song build_village_theme() {
    Song song;
    song.name = "Hearthsmoke";
    // Slow enough to sit under ambience rather than compete with it. At 76 BPM
    // with 16th steps a bar is a little over three seconds, so the four-bar
    // loop runs ~12.6 s — long enough not to feel like a ringtone.
    song.bpm = 76.0f;
    song.steps_per_beat = 4;
    song.loop = true;

    // Harmony: Dm - C - F - Dm. Dorian, so the C is natural and there is no
    // C#: that missing leading tone is the whole reason this reads as folk
    // rather than as a cadence heading somewhere.

    // --- Bass: root of each bar, held nearly the full bar -----------------
    auto bass = synth::patches::sine_bass();
    bass.pan = 0.0f;
    Pattern bass_p = lay("bass", {
        {.step = b1, .note = D2, .gate = 14},
        {.step = b2, .note = C2, .gate = 14},
        {.step = b3, .note = F2, .gate = 14},
        {.step = b4, .note = D2, .gate = 14},
    });

    // --- Pad: the triad, one track per chord tone -------------------------
    // Three monophonic tracks rather than one polyphonic one, because a Track
    // is monophonic by design (see song.hpp) — and it means each voice can be
    // panned separately, which is what gives the pad its width.
    auto pad = synth::patches::soft_pad();

    auto pad_voice = [&](float pan) {
        synth::Instrument i = pad;
        i.pan = pan;
        return i;
    };

    Pattern pad_low = lay("pad_low", {
        {.step = b1, .note = D4, .gate = 15, .velocity = 0.85f},
        {.step = b2, .note = C4, .gate = 15, .velocity = 0.85f},
        {.step = b3, .note = C4, .gate = 15, .velocity = 0.85f},
        {.step = b4, .note = D4, .gate = 15, .velocity = 0.85f},
    });
    Pattern pad_mid = lay("pad_mid", {
        {.step = b1, .note = F4, .gate = 15, .velocity = 0.75f},
        {.step = b2, .note = E4, .gate = 15, .velocity = 0.75f},
        {.step = b3, .note = F4, .gate = 15, .velocity = 0.75f},
        {.step = b4, .note = F4, .gate = 15, .velocity = 0.75f},
    });
    Pattern pad_high = lay("pad_high", {
        {.step = b1, .note = A4, .gate = 15, .velocity = 0.7f},
        {.step = b2, .note = G4, .gate = 15, .velocity = 0.7f},
        {.step = b3, .note = A4, .gate = 15, .velocity = 0.7f},
        {.step = b4, .note = A4, .gate = 15, .velocity = 0.7f},
    });

    // --- Melody: plucked, mostly quarters and eighths ---------------------
    // Shaped as a question in bars 1-2 rising to C5, answered in bars 3-4
    // falling back to D. Rests are as deliberate as the notes: the gaps are
    // where the village ambience shows through.
    auto lead = synth::patches::plucked_string();
    lead.pan = -0.15f;
    Pattern lead_p = lay("lead", {
        // Bar 1 — Dm
        {.step = b1 + 0,  .note = A4, .gate = 3},
        {.step = b1 + 4,  .note = D5, .gate = 3},
        {.step = b1 + 8,  .note = C5, .gate = 3},
        {.step = b1 + 12, .note = A4, .gate = 6},
        // Bar 2 — C
        {.step = b2 + 0,  .note = G4, .gate = 3},
        {.step = b2 + 6,  .note = A4, .gate = 2},
        {.step = b2 + 8,  .note = C5, .gate = 7},
        // Bar 3 — F
        {.step = b3 + 0,  .note = F4, .gate = 3},
        {.step = b3 + 4,  .note = A4, .gate = 3},
        {.step = b3 + 8,  .note = G4, .gate = 3},
        {.step = b3 + 12, .note = F4, .gate = 3},
        // Bar 4 — Dm, settling
        {.step = b4 + 0,  .note = E4, .gate = 3},
        {.step = b4 + 4,  .note = D4, .gate = 11},
    });

    // --- Counter-melody: a sparse high answer, panned opposite the lead ---
    auto counter = synth::patches::reed();
    counter.pan = 0.35f;
    Pattern counter_p = lay("counter", {
        {.step = b2 + 12, .note = D5, .gate = 3, .velocity = 0.55f},
        {.step = b3 + 6,  .note = C5, .gate = 3, .velocity = 0.5f},
        {.step = b4 + 8,  .note = A4, .gate = 6, .velocity = 0.45f},
    });

    // --- Shaker: offbeat eighths, quiet, just to keep time moving ---------
    auto hat = synth::patches::hat();
    hat.pan = 0.2f;
    std::vector<Placement> hats;
    for (int step = 2; step < k_song_steps; step += 4) {
        // Slight accent every other hit so it breathes instead of ticking.
        const float v = ((step / 4) % 2 == 0) ? 0.5f : 0.32f;
        hats.push_back({.step = step, .note = A4, .gate = 1, .velocity = v});
    }
    Pattern hat_p = lay("hat", hats);

    song.tracks = {
        make_track("bass", bass, std::move(bass_p), 0.9f),
        make_track("pad_low", pad_voice(-0.4f), std::move(pad_low), 0.8f),
        make_track("pad_mid", pad_voice(0.0f), std::move(pad_mid), 0.8f),
        make_track("pad_high", pad_voice(0.4f), std::move(pad_high), 0.8f),
        make_track("lead", lead, std::move(lead_p), 1.0f),
        make_track("counter", counter, std::move(counter_p), 0.9f),
        make_track("hat", hat, std::move(hat_p), 0.7f),
    };

    return song;
}


bool render_music_to_wav(const std::string& path, float seconds) {
    // Through the real AudioEngine, not a bare Sequencer: this way the dump
    // includes the master gain and the soft limiter, so what lands in the file
    // is what the device would have played.
    AudioEngine engine;
    engine.set_music_gain(0.55f); // same as the playground sets in on_init
    engine.play_music(build_village_theme());
    engine.set_music_playing(true);

    std::vector<StereoFrame> frames(
        static_cast<std::size_t>(frames_from_seconds(seconds)));
    engine.render_offline(frames);
    return write_wav(path, frames);
}

} // namespace playground
#endif // BRACKEN_SYNTH
