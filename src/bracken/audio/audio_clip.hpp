#pragma once

#include "bracken/audio/audio_format.hpp"
#include "bracken/core/result.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace bracken::audio {

enum class ClipLoadError : uint8_t {
    FileNotFound,
    UnsupportedFormat,
    ConversionFailed,
};

[[nodiscard]] const char* to_string(ClipLoadError e) noexcept;

/**
 * Decoded PCM, already converted to the mixer's format (float, stereo,
 * k_sample_rate).
 *
 * Conversion happens once at load rather than per-voice at playback: a clip
 * played by forty overlapping voices would otherwise resample forty times a
 * frame, on the audio thread, where there is no budget for it.
 *
 * Clips are immutable once loaded, so any number of voices can read one
 * concurrently without synchronization.
 */
class AudioClip {
public:
    /** Load a WAV. The engine deliberately supports exactly one container:
     *  compressed formats need a decoder dependency, and a game that wants
     *  Ogg/Opus can decode with its own library and hand over the PCM via
     *  from_samples(). */
    [[nodiscard]] static core::Result<AudioClip, ClipLoadError> load_wav(const std::string& path);

    /** Adopt already-decoded interleaved stereo float samples at
     *  k_sample_rate — the escape hatch for hosts with their own decoder. */
    [[nodiscard]] static AudioClip from_samples(std::vector<StereoFrame> frames, std::string name);

    [[nodiscard]] const std::vector<StereoFrame>& frames() const noexcept { return frames_; }
    [[nodiscard]] std::size_t frame_count() const noexcept { return frames_.size(); }
    [[nodiscard]] float duration_seconds() const noexcept {
        return seconds_from_frames(static_cast<int>(frames_.size()));
    }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] bool empty() const noexcept { return frames_.empty(); }

private:
    std::vector<StereoFrame> frames_;
    std::string name_;
};

} // namespace bracken::audio
