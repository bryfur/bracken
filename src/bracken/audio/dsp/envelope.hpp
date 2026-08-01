#pragma once

#include "bracken/audio/audio_format.hpp"

#include <algorithm>

namespace bracken::audio::dsp {

/**
 * ADSR envelope in linear gain.
 *
 * Stages advance per frame rather than by a timer, so an envelope is exactly
 * as long as its parameters say regardless of buffer size — a note is not
 * allowed to get longer because the device asked for bigger chunks.
 *
 * Release always starts from the CURRENT level, not from sustain. Releasing
 * during the attack of a very short note would otherwise jump the gain up to
 * sustain first, which is an audible click on exactly the notes (staccato,
 * fast arpeggios) where it is most obvious.
 */
class Envelope {
public:
    struct Settings {
        float attack_s = 0.005f;
        float decay_s = 0.08f;
        float sustain = 0.7f; // linear gain held while the note is down
        float release_s = 0.15f;
    };

    enum class Stage : uint8_t { Idle, Attack, Decay, Sustain, Release };

    void configure(const Settings& s) noexcept {
        // One frame minimum: a zero-length attack must still be a ramp, or
        // the first sample steps from 0 to full scale and clicks.
        attack_frames_ = std::max(1, frames_from_seconds(s.attack_s));
        decay_frames_ = std::max(1, frames_from_seconds(s.decay_s));
        release_frames_ = std::max(1, frames_from_seconds(s.release_s));
        sustain_ = std::clamp(s.sustain, 0.0f, 1.0f);
    }

    void note_on() noexcept {
        stage_ = Stage::Attack;
        frame_in_stage_ = 0;
        stage_start_level_ = level_; // retrigger from wherever we are: no click
    }

    void note_off() noexcept {
        if (stage_ == Stage::Idle || stage_ == Stage::Release) {
            return;
        }
        stage_ = Stage::Release;
        frame_in_stage_ = 0;
        stage_start_level_ = level_;
    }

    /** Cut to silence immediately — used when a voice is stolen. */
    void hard_reset() noexcept {
        stage_ = Stage::Idle;
        level_ = 0.0f;
        frame_in_stage_ = 0;
        stage_start_level_ = 0.0f;
    }

    [[nodiscard]] Stage stage() const noexcept { return stage_; }
    [[nodiscard]] float level() const noexcept { return level_; }

    /** True once the envelope has finished releasing — the voice may be
     *  reclaimed. Idle counts: a voice that never started is free. */
    [[nodiscard]] bool finished() const noexcept { return stage_ == Stage::Idle; }

    /** Advance one frame and return the new gain. */
    [[nodiscard]] float next() noexcept {
        switch (stage_) {
        case Stage::Idle:
            level_ = 0.0f;
            break;

        case Stage::Attack: {
            const float t = progress(attack_frames_);
            level_ = stage_start_level_ + (1.0f - stage_start_level_) * t;
            if (++frame_in_stage_ >= attack_frames_) {
                stage_ = Stage::Decay;
                frame_in_stage_ = 0;
                stage_start_level_ = 1.0f;
                level_ = 1.0f;
            }
            break;
        }

        case Stage::Decay: {
            const float t = progress(decay_frames_);
            level_ = 1.0f + (sustain_ - 1.0f) * t;
            if (++frame_in_stage_ >= decay_frames_) {
                stage_ = Stage::Sustain;
                level_ = sustain_;
            }
            break;
        }

        case Stage::Sustain:
            level_ = sustain_;
            break;

        case Stage::Release: {
            const float t = progress(release_frames_);
            level_ = stage_start_level_ * (1.0f - t);
            if (++frame_in_stage_ >= release_frames_) {
                stage_ = Stage::Idle;
                level_ = 0.0f;
            }
            break;
        }
        }
        return level_;
    }

private:
    [[nodiscard]] float progress(int stage_frames) const noexcept {
        return static_cast<float>(frame_in_stage_) / static_cast<float>(stage_frames);
    }

    Stage stage_ = Stage::Idle;
    float level_ = 0.0f;
    float stage_start_level_ = 0.0f;
    int frame_in_stage_ = 0;

    int attack_frames_ = 1;
    int decay_frames_ = 1;
    int release_frames_ = 1;
    float sustain_ = 0.7f;
};

} // namespace bracken::audio::dsp
