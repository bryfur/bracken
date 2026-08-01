#pragma once

#include "bracken/audio/audio_format.hpp"

#include <span>
#include <string>

namespace bracken::audio {

/**
 * Write stereo float frames to a 16-bit PCM WAV.
 *
 * Exists so audio can be inspected rather than only heard: a rendered song or
 * a captured mix can be opened in any editor, diffed between builds, or
 * eyeballed as a waveform. That matters more here than in most engines
 * because the music is *generated* — when a patch sounds wrong there is no
 * source asset to compare against, so being able to dump the actual output is
 * the equivalent of the renderer's screenshot.
 *
 * 16-bit rather than float WAV because every tool opens it. Samples are
 * clamped, not wrapped: a hot mix should sound clipped, not inverted.
 *
 * Returns false and logs on IO failure.
 */
[[nodiscard]] bool write_wav(const std::string& path, std::span<const StereoFrame> frames,
                             int sample_rate = k_sample_rate);

} // namespace bracken::audio
