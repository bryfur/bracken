#include "bracken/audio/audio_clip.hpp"

#include "bracken/core/logger.hpp"

#include <SDL3/SDL_audio.h>
#include <SDL3/SDL_stdinc.h>

#include <cstring>
#include <utility>

namespace bracken::audio {

const char* to_string(ClipLoadError e) noexcept {
    switch (e) {
    case ClipLoadError::FileNotFound: return "file not found or unreadable";
    case ClipLoadError::UnsupportedFormat: return "unsupported WAV format";
    case ClipLoadError::ConversionFailed: return "sample-format conversion failed";
    }
    return "unknown";
}

AudioClip AudioClip::from_samples(std::vector<StereoFrame> frames, std::string name) {
    AudioClip clip;
    clip.frames_ = std::move(frames);
    clip.name_ = std::move(name);
    return clip;
}

core::Result<AudioClip, ClipLoadError> AudioClip::load_wav(const std::string& path) {
    SDL_AudioSpec spec{};
    Uint8* wav_buffer = nullptr;
    Uint32 wav_len = 0;

    if (!SDL_LoadWAV(path.c_str(), &spec, &wav_buffer, &wav_len)) {
        BRACKEN_LOG_ERROR("audio", "AudioClip::load_wav('{}'): {}", path, SDL_GetError());
        return core::Err(ClipLoadError::FileNotFound);
    }

    // Convert once, here, into the mixer's format. Doing it at playback would
    // put a resampler on the audio thread for every voice.
    const SDL_AudioSpec dst{.format = SDL_AUDIO_F32, .channels = k_channels, .freq = k_sample_rate};

    Uint8* converted = nullptr;
    int converted_len = 0;
    const bool ok = SDL_ConvertAudioSamples(&spec, wav_buffer, static_cast<int>(wav_len), &dst,
                                            &converted, &converted_len);
    SDL_free(wav_buffer);

    if (!ok || converted == nullptr) {
        BRACKEN_LOG_ERROR("audio", "AudioClip::load_wav('{}'): conversion failed: {}", path,
                          SDL_GetError());
        return core::Err(ClipLoadError::ConversionFailed);
    }

    AudioClip clip;
    clip.name_ = path;
    const auto frame_count = static_cast<std::size_t>(converted_len) / sizeof(StereoFrame);
    clip.frames_.resize(frame_count);
    if (frame_count > 0) {
        std::memcpy(clip.frames_.data(), converted, frame_count * sizeof(StereoFrame));
    }
    SDL_free(converted);

    BRACKEN_LOG_INFO("audio", "Loaded clip '{}' ({} frames, {:.2f}s)", path, frame_count,
                     clip.duration_seconds());
    return clip;
}

} // namespace bracken::audio
