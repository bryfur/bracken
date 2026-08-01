#pragma once

#include <cmath>
#include <cstdint>

namespace bracken::audio {

/**
 * The engine mixes and synthesizes in one format and converts once, at the
 * device boundary: 32-bit float, stereo, interleaved. Float because DSP that
 * clips intermediate results is a class of bug nobody should debug; stereo
 * interleaved because that is what SDL's stream wants, so the mixer's output
 * buffer can be handed over without a repack.
 */
inline constexpr int k_channels = 2;

/** The mixer's internal rate. The device may run at something else — the SDL
 *  stream resamples — but every DSP object here is built against this. */
inline constexpr int k_sample_rate = 48000;

/** One stereo sample pair. The mixer works in frames, never in raw samples,
 *  because "how many samples is that buffer" is otherwise ambiguous by a
 *  factor of two and gets it wrong silently. */
struct StereoFrame {
    float left = 0.0f;
    float right = 0.0f;

    constexpr StereoFrame& operator+=(StereoFrame o) noexcept {
        left += o.left;
        right += o.right;
        return *this;
    }

    constexpr StereoFrame operator*(float g) const noexcept { return {left * g, right * g}; }

    friend constexpr bool operator==(StereoFrame, StereoFrame) noexcept = default;
};

/** Seconds → frames at the mixer rate, rounded to nearest. */
[[nodiscard]] inline constexpr int frames_from_seconds(float seconds) noexcept {
    return static_cast<int>(seconds * static_cast<float>(k_sample_rate) + 0.5f);
}

/** Frames → seconds at the mixer rate. */
[[nodiscard]] inline constexpr float seconds_from_frames(int frames) noexcept {
    return static_cast<float>(frames) / static_cast<float>(k_sample_rate);
}

/**
 * Decibels → linear gain. Audio APIs that take a bare 0..1 "volume" mislead:
 * 0.5 is not half as loud, it is about -6 dB. Hosts that want a perceptual
 * fade should step dB, not the linear factor.
 */
[[nodiscard]] inline float gain_from_db(float db) noexcept {
    return std::pow(10.0f, db / 20.0f);
}

/**
 * MIDI note number → frequency in Hz, A4 (note 69) = 440 Hz. Note numbers
 * rather than raw Hz are what music data is written in, and equal temperament
 * is one exp2 away — so the engine does it instead of every song table
 * carrying a column of pre-computed frequencies.
 */
[[nodiscard]] inline float hz_from_midi(float note) noexcept {
    return 440.0f * std::exp2((note - 69.0f) / 12.0f);
}

/** Named notes, so song data reads as music instead of as integers. Octave
 *  numbering is scientific pitch notation: C4 = middle C = MIDI 60. */
enum : uint8_t {
    // clang-format off
    C1 = 24, Cs1, D1, Ds1, E1, F1, Fs1, G1, Gs1, A1, As1, B1,
    C2 = 36, Cs2, D2, Ds2, E2, F2, Fs2, G2, Gs2, A2, As2, B2,
    C3 = 48, Cs3, D3, Ds3, E3, F3, Fs3, G3, Gs3, A3, As3, B3,
    C4 = 60, Cs4, D4, Ds4, E4, F4, Fs4, G4, Gs4, A4, As4, B4,
    C5 = 72, Cs5, D5, Ds5, E5, F5, Fs5, G5, Gs5, A5, As5, B5,
    C6 = 84, Cs6, D6, Ds6, E6, F6, Fs6, G6, Gs6, A6, As6, B6,
    // clang-format on
};

/** Sentinel for "no note in this step" in pattern data. */
inline constexpr uint8_t k_note_rest = 0;

} // namespace bracken::audio
