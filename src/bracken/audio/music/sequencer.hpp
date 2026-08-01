#pragma once

#include "bracken/audio/audio_format.hpp"
#include "bracken/audio/music/song.hpp"
#include "bracken/audio/synth/synth.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace bracken::audio::music {

/** One synth per track, so tracks do not steal each other's voices: a busy
 *  arpeggio must not silence the bass. */
inline constexpr int k_max_tracks = 16;

/**
 * Plays a Song by driving one Synth per track.
 *
 * Sample-accurate: the step boundary is found by counting frames, and a
 * buffer that straddles one is split rather than rounded. Rounding to buffer
 * boundaries is the classic way a sequencer acquires audible timing jitter
 * that changes with the device's buffer size — the same class of bug as the
 * variable-timestep problem the smoke test had.
 *
 * The Song is held BY VALUE. The audio thread reads it every step, so a host
 * mutating its own copy mid-playback would be a race; this is the same
 * no-host-lifetime-contracts rule the FramePacket follows.
 *
 * Tier: audio-thread for render(). Construction and set_song() happen on the
 * owning thread before playback starts.
 */
class Sequencer {
public:
    void set_song(Song song) {
        song_ = std::move(song);
        rewind();
    }

    /**
     * Swap in a new song and hand the old one back.
     *
     * Exists so the audio thread never frees memory: it moves the outgoing
     * song out (a pointer steal, no deallocation) and the owning thread
     * destroys it later. `set_song` would destroy the old song in place,
     * which on the audio thread means calling free() inside the device
     * callback — the classic source of an intermittent dropout that only
     * reproduces on someone else's machine.
     */
    [[nodiscard]] Song swap_song(Song next) {
        Song previous = std::move(song_);
        song_ = std::move(next);
        rewind();
        return previous;
    }

    [[nodiscard]] const Song& song() const noexcept { return song_; }

    void rewind() noexcept {
        step_ = 0;
        frame_in_step_ = 0;
        frames_per_step_ = song_.frames_per_step();
        length_steps_ = song_.length_steps();
        finished_ = length_steps_ == 0;
        for (auto& s : track_state_) {
            s = TrackState{};
        }
        for (auto& s : synths_) {
            s.panic();
        }
    }

    void set_playing(bool playing) noexcept {
        playing_ = playing;
        if (!playing) {
            // Release rather than cut: pausing should not click.
            for (auto& s : synths_) {
                s.all_notes_off();
            }
        }
    }

    [[nodiscard]] bool playing() const noexcept { return playing_ && !finished_; }
    [[nodiscard]] int current_step() const noexcept { return step_; }
    [[nodiscard]] bool finished() const noexcept { return finished_; }

    [[nodiscard]] int active_voices() const noexcept {
        int n = 0;
        for (const auto& s : synths_) {
            n += s.active_voices();
        }
        return n;
    }

    /** Render `out` frames, ADDING into whatever is already there. Voices
     *  keep ringing after the song finishes, so a release tail is not cut. */
    void render_additive(std::span<StereoFrame> out) noexcept {
        std::size_t done = 0;
        while (done < out.size()) {
            if (playing() && frame_in_step_ == 0) {
                trigger_step();
            }

            // Render up to the next step boundary in one go.
            std::size_t chunk = out.size() - done;
            if (playing()) {
                const auto until_boundary =
                    static_cast<std::size_t>(frames_per_step_ - frame_in_step_);
                chunk = chunk < until_boundary ? chunk : until_boundary;
            }

            auto slice = out.subspan(done, chunk);
            for (std::size_t t = 0; t < synths_.size(); ++t) {
                if (t < song_.tracks.size() && !song_.tracks[t].muted) {
                    synths_[t].render_additive(slice);
                }
            }

            done += chunk;
            if (!playing()) {
                continue;
            }

            frame_in_step_ += static_cast<int>(chunk);
            if (frame_in_step_ >= frames_per_step_) {
                frame_in_step_ = 0;
                advance_step();
            }
        }
    }

private:
    struct TrackState {
        uint8_t sounding_note = k_note_rest;
        int gate_remaining = -1; // <0 = hold until the next note on this track
        bool has_note = false;
    };

    /** Resolve a track's absolute step index through its arrangement. */
    [[nodiscard]] const Step* step_for(const Track& track, int absolute_step) const noexcept {
        if (track.patterns.empty()) {
            return nullptr;
        }
        if (track.arrangement.empty()) {
            const auto& steps = track.patterns[0].steps;
            if (steps.empty()) {
                return nullptr;
            }
            // A short pattern with no arrangement loops on its own length, so
            // a 16-step drum figure sits under a 64-step melody unpadded.
            return &steps[static_cast<std::size_t>(absolute_step) % steps.size()];
        }

        int remaining = absolute_step;
        for (uint32_t idx : track.arrangement) {
            if (idx >= track.patterns.size()) {
                continue;
            }
            const auto& steps = track.patterns[idx].steps;
            const auto n = static_cast<int>(steps.size());
            if (remaining < n) {
                return &steps[static_cast<std::size_t>(remaining)];
            }
            remaining -= n;
        }
        return nullptr; // this track has run out; stays silent
    }

    void trigger_step() noexcept {
        const auto track_count = song_.tracks.size() < synths_.size() ? song_.tracks.size()
                                                                      : synths_.size();
        for (std::size_t t = 0; t < track_count; ++t) {
            const Track& track = song_.tracks[t];
            TrackState& st = track_state_[t];

            // Expire a timed gate before considering a new note, so a step
            // that both ends one note and starts another does it in order.
            if (st.has_note && st.gate_remaining == 0) {
                synths_[t].note_off(st.sounding_note);
                st.has_note = false;
            }
            if (st.gate_remaining > 0) {
                --st.gate_remaining;
            }

            const Step* s = step_for(track, step_);
            if (s == nullptr || s->note == k_note_rest) {
                continue;
            }

            // A new note on a track that is holding one releases it first —
            // this is what makes a track monophonic.
            if (st.has_note) {
                synths_[t].note_off(st.sounding_note);
            }

            synths_[t].note_on(track.instrument, s->note, s->velocity * track.gain);
            st.sounding_note = s->note;
            st.has_note = true;
            st.gate_remaining = s->gate_steps == 0 ? -1 : static_cast<int>(s->gate_steps);
        }
    }

    void advance_step() noexcept {
        ++step_;
        if (step_ < length_steps_) {
            return;
        }
        if (song_.loop) {
            step_ = 0;
        } else {
            finished_ = true;
            for (std::size_t t = 0; t < track_state_.size(); ++t) {
                if (track_state_[t].has_note) {
                    synths_[t].note_off(track_state_[t].sounding_note);
                    track_state_[t].has_note = false;
                }
            }
        }
    }

    Song song_{};
    std::array<synth::Synth, k_max_tracks> synths_{};
    std::array<TrackState, k_max_tracks> track_state_{};

    int step_ = 0;
    int frame_in_step_ = 0;
    int frames_per_step_ = k_sample_rate;
    int length_steps_ = 0;
    bool playing_ = false;
    bool finished_ = true;
};

} // namespace bracken::audio::music
