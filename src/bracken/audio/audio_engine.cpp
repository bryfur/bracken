#include "bracken/audio/audio_engine.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/logger.hpp"

#include <SDL3/SDL_audio.h>
#include <SDL3/SDL_init.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <numbers>
#include <optional>

namespace bracken::audio {
namespace {

/** Simultaneous one-shot/looping clip voices. Fixed, for the same reason the
 *  synth's polyphony is fixed: an unbounded voice count turns a runaway
 *  spawner into a CPU spiral inside the device callback. */
constexpr int k_max_sound_voices = 64;

/** Commands in flight from the game to the mixer. Two frames' worth of a very
 *  chatty scene, rounded up to a power of two. */
constexpr std::size_t k_command_capacity = 512;

/** Mix buffer high-water. SDL asks in device-sized chunks; anything larger is
 *  split across several mix() calls rather than reallocating. */
constexpr std::size_t k_max_mix_frames = 4096;

enum class CommandType : uint8_t {
    PlaySound,
    StopSound,
    SetListener,
    SetMusicPlaying,
};

/** Trivially copyable by construction — it travels through the lock-free ring,
 *  so it may not own memory. */
struct Command {
    CommandType type = CommandType::PlaySound;
    SoundId id{};
    SoundParams params{};
    glm::vec3 listener_pos{0.0f};
    glm::vec3 listener_forward{0.0f, 0.0f, -1.0f};
    glm::vec3 listener_up{0.0f, 1.0f, 0.0f};
    bool flag = false;
};

struct SoundVoice {
    const AudioClip* clip = nullptr;
    SoundId id{};
    double position = 0.0; // fractional frame cursor, for pitch shifting
    float gain = 1.0f;
    float pitch = 1.0f;
    SoundSpace space = SoundSpace::UI;
    glm::vec3 world_pos{0.0f};
    float min_distance = 200.0f;
    float max_distance = 4000.0f;
    bool looping = false;
    bool active = false;
};

} // namespace

struct AudioEngine::Mixer {
    // --- shared, written by the game thread, read by the audio thread ------
    CommandRing<Command, k_command_capacity> commands;
    std::atomic<float> master_gain{1.0f};
    std::atomic<float> music_gain{1.0f};
    std::atomic<float> sound_gain{1.0f};

#if BRACKEN_SYNTH
    // Song handoff. The mutex is taken with try_lock on the audio side, so
    // the callback is never blocked by a host building a song.
    std::mutex song_mutex;
    std::optional<music::Song> pending_song;
    std::optional<music::Song> retired_song; // freed by the game thread
    bool pending_song_valid = false;
    music::Sequencer sequencer;
#endif

    // --- audio-thread-only -------------------------------------------------
    std::array<SoundVoice, k_max_sound_voices> voices{};
    glm::vec3 listener_pos{0.0f};
    glm::vec3 listener_right{1.0f, 0.0f, 0.0f};

    // --- diagnostics (atomics; read from anywhere) ------------------------
    std::atomic<int> active_sound_voices{0};
    std::atomic<int> active_music_voices{0};
    std::atomic<uint64_t> underruns{0};
    std::atomic<float> peak_level{0.0f};
    std::atomic<bool> music_playing{false};

    // Clips are append-only and never removed, so the audio thread can hold
    // raw pointers into this vector for the process lifetime. Registration
    // happens before playback; `reserve` keeps the backing store stable.
    std::vector<AudioClip> clips;
};

AudioEngine::AudioEngine() : mixer_(std::make_unique<Mixer>()) {
    mixer_->clips.reserve(256);
}

AudioEngine::~AudioEngine() {
    shutdown();
}

bool AudioEngine::init() {
    if (device_open_) {
        return true;
    }

    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        BRACKEN_LOG_WARN("audio", "SDL audio subsystem unavailable ({}); running silent",
                         SDL_GetError());
        return false;
    }

    scratch_.resize(k_max_mix_frames);

    const SDL_AudioSpec spec{
        .format = SDL_AUDIO_F32, .channels = k_channels, .freq = k_sample_rate};

    stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec,
                                        &AudioEngine::stream_callback, this);
    if (stream_ == nullptr) {
        // Not fatal: a headless CI box or a machine with no output device
        // should still run the game, just silently.
        BRACKEN_LOG_WARN("audio", "No audio playback device ({}); running silent", SDL_GetError());
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        return false;
    }

    if (!SDL_ResumeAudioStreamDevice(stream_)) {
        BRACKEN_LOG_WARN("audio", "Could not start audio device: {}", SDL_GetError());
        SDL_DestroyAudioStream(stream_);
        stream_ = nullptr;
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        return false;
    }

    device_open_ = true;
    BRACKEN_LOG_INFO("audio", "Audio device open ({} Hz, {} ch, float32)", k_sample_rate,
                     k_channels);
    return true;
}

void AudioEngine::shutdown() {
    if (stream_ != nullptr) {
        // Destroying the stream joins the callback, so nothing below can race
        // with a mix in progress.
        SDL_DestroyAudioStream(stream_);
        stream_ = nullptr;
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        BRACKEN_LOG_INFO("audio", "Audio device closed");
    }
    device_open_ = false;
}

SoundId AudioEngine::register_clip(AudioClip clip) {
    if (clip.empty()) {
        BRACKEN_LOG_WARN("audio", "register_clip('{}'): clip has no samples; ignored", clip.name());
        return {};
    }
    if (mixer_->clips.size() >= mixer_->clips.capacity()) {
        // Growing would move the backing store while the audio thread holds
        // pointers into it. Refuse loudly rather than corrupt the mix.
        BRACKEN_LOG_ERROR("audio", "register_clip('{}'): clip budget ({}) exhausted",
                          clip.name(), mixer_->clips.capacity());
        return {};
    }
    const auto index = static_cast<uint32_t>(mixer_->clips.size());
    mixer_->clips.push_back(std::move(clip));
    return SoundId{index};
}

SoundId AudioEngine::load_sound(const std::string& path) {
    auto result = AudioClip::load_wav(path);
    if (!result) {
        BRACKEN_LOG_WARN("audio", "load_sound('{}') failed: {}", path,
                         to_string(result.error()));
        return {};
    }
    return register_clip(std::move(result).value());
}

void AudioEngine::play_sound(SoundId id, const SoundParams& params) {
    if (!id.valid() || id.value >= mixer_->clips.size()) {
        return;
    }
    Command c;
    c.type = CommandType::PlaySound;
    c.id = id;
    c.params = params;
    (void)mixer_->commands.push(c); // dropped-on-full is counted in diagnostics
}

void AudioEngine::stop_sound(SoundId id) {
    Command c;
    c.type = CommandType::StopSound;
    c.id = id;
    (void)mixer_->commands.push(c);
}

void AudioEngine::set_listener(const glm::vec3& position, const glm::vec3& forward,
                               const glm::vec3& up) {
    Command c;
    c.type = CommandType::SetListener;
    c.listener_pos = position;
    c.listener_forward = forward;
    c.listener_up = up;
    (void)mixer_->commands.push(c);
}

void AudioEngine::set_master_gain(float gain) {
    mixer_->master_gain.store(std::max(0.0f, gain), std::memory_order_relaxed);
}
void AudioEngine::set_music_gain(float gain) {
    mixer_->music_gain.store(std::max(0.0f, gain), std::memory_order_relaxed);
}
void AudioEngine::set_sound_gain(float gain) {
    mixer_->sound_gain.store(std::max(0.0f, gain), std::memory_order_relaxed);
}

#if BRACKEN_SYNTH
void AudioEngine::play_music(music::Song song) {
    {
        std::lock_guard lock(mixer_->song_mutex);
        // Free the previously retired song HERE, on the calling thread — this
        // is the whole point of the retire slot.
        mixer_->retired_song.reset();
        mixer_->pending_song = std::move(song);
        mixer_->pending_song_valid = true;
    }
    mixer_->music_playing.store(true, std::memory_order_relaxed);
    Command c;
    c.type = CommandType::SetMusicPlaying;
    c.flag = true;
    (void)mixer_->commands.push(c);
}

#endif // BRACKEN_SYNTH

void AudioEngine::stop_music() {
    set_music_playing(false);
}

void AudioEngine::set_music_playing(bool playing) {
    mixer_->music_playing.store(playing, std::memory_order_relaxed);
    Command c;
    c.type = CommandType::SetMusicPlaying;
    c.flag = playing;
    (void)mixer_->commands.push(c);
}

bool AudioEngine::music_playing() const noexcept {
    return mixer_->music_playing.load(std::memory_order_relaxed);
}

AudioDiagnostics AudioEngine::diagnostics() const {
    AudioDiagnostics d;
    d.active_sound_voices = mixer_->active_sound_voices.load(std::memory_order_relaxed);
    d.active_music_voices = mixer_->active_music_voices.load(std::memory_order_relaxed);
    d.commands_dropped = mixer_->commands.dropped();
    d.buffer_underruns = mixer_->underruns.load(std::memory_order_relaxed);
    d.peak_level = mixer_->peak_level.load(std::memory_order_relaxed);
    d.device_open = device_open_;
    return d;
}

void SDLCALL AudioEngine::stream_callback(void* userdata, SDL_AudioStream* stream,
                                          int additional_amount, int /*total_amount*/) {
    auto* self = static_cast<AudioEngine*>(userdata);
    if (additional_amount <= 0) {
        return;
    }

    int frames_needed = additional_amount / static_cast<int>(sizeof(StereoFrame));
    while (frames_needed > 0) {
        const int chunk = std::min(frames_needed, static_cast<int>(k_max_mix_frames));
        std::span<StereoFrame> out(self->scratch_.data(), static_cast<std::size_t>(chunk));
        self->mix(out);
        SDL_PutAudioStreamData(stream, out.data(),
                               chunk * static_cast<int>(sizeof(StereoFrame)));
        frames_needed -= chunk;
    }
}

void AudioEngine::render_offline(std::span<StereoFrame> out) {
    // Same path the device takes, minus the device. Chunked identically so an
    // offline render is sample-identical to a live one.
    std::size_t done = 0;
    while (done < out.size()) {
        const std::size_t chunk = std::min(k_max_mix_frames, out.size() - done);
        mix(out.subspan(done, chunk));
        done += chunk;
    }
}

void AudioEngine::mix(std::span<StereoFrame> out) {
    Mixer& m = *mixer_;

    // --- drain commands ---------------------------------------------------
    Command c;
    while (m.commands.pop(c)) {
        switch (c.type) {
        case CommandType::PlaySound: {
            SoundVoice* slot = nullptr;
            for (SoundVoice& v : m.voices) {
                if (!v.active) {
                    slot = &v;
                    break;
                }
            }
            if (slot == nullptr) {
                break; // all voices busy; drop the newest, never a live one
            }
            *slot = SoundVoice{};
            slot->clip = &m.clips[c.id.value];
            slot->id = c.id;
            slot->gain = c.params.gain;
            slot->pitch = std::max(0.01f, c.params.pitch);
            slot->space = c.params.space;
            slot->world_pos = c.params.position;
            slot->min_distance = std::max(1.0f, c.params.min_distance);
            slot->max_distance = std::max(slot->min_distance + 1.0f, c.params.max_distance);
            slot->looping = c.params.looping;
            slot->active = true;
            break;
        }
        case CommandType::StopSound:
            for (SoundVoice& v : m.voices) {
                if (v.active && v.id.value == c.id.value) {
                    v.active = false;
                }
            }
            break;
        case CommandType::SetListener: {
            m.listener_pos = c.listener_pos;
            const glm::vec3 f = c.listener_forward;
            const glm::vec3 u = c.listener_up;
            const glm::vec3 r = glm::cross(f, u);
            const float len = glm::length(r);
            // A degenerate basis (forward parallel to up) must not produce
            // NaN pan values and blow up the whole mix.
            m.listener_right = len > 1e-5f ? r / len : glm::vec3(1.0f, 0.0f, 0.0f);
            break;
        }
        case CommandType::SetMusicPlaying:
#if BRACKEN_SYNTH
            m.sequencer.set_playing(c.flag);
#endif
            break;
        }
    }

    // --- adopt a pending song without ever freeing on this thread ---------
#if BRACKEN_SYNTH
    if (m.pending_song_valid && m.song_mutex.try_lock()) {
        if (m.pending_song_valid && m.pending_song.has_value()) {
            // swap_song moves the old song out; the game thread frees it.
            m.retired_song = m.sequencer.swap_song(std::move(*m.pending_song));
            m.pending_song.reset();
            m.pending_song_valid = false;
            m.sequencer.set_playing(m.music_playing.load(std::memory_order_relaxed));
        }
        m.song_mutex.unlock();
    }
#endif

    // --- clear, then sum --------------------------------------------------
    for (StereoFrame& f : out) {
        f = StereoFrame{};
    }

    const float music_gain = m.music_gain.load(std::memory_order_relaxed);
#if BRACKEN_SYNTH
    if (music_gain > 0.0f) {
        // The sequencer adds into the bus; scale afterwards so its own
        // internal mix stays in range.
        m.sequencer.render_additive(out);
        if (music_gain != 1.0f) {
            for (StereoFrame& f : out) {
                f = f * music_gain;
            }
        }
    }
    m.active_music_voices.store(m.sequencer.active_voices(), std::memory_order_relaxed);
#else
    (void)music_gain;
#endif

    const float sound_gain = m.sound_gain.load(std::memory_order_relaxed);
    int active = 0;
    for (SoundVoice& v : m.voices) {
        if (!v.active) {
            continue;
        }
        ++active;

        float gain_l = v.gain * sound_gain;
        float gain_r = gain_l;
        if (v.space == SoundSpace::World) {
            const glm::vec3 to_source = v.world_pos - m.listener_pos;
            const float distance = glm::length(to_source);

            // Inverse-distance rolloff clamped at min_distance, faded to
            // nothing by max_distance so a distant emitter costs a voice for
            // a bounded time rather than forever.
            float attenuation = 1.0f;
            if (distance > v.min_distance) {
                attenuation = v.min_distance / distance;
                const float fade = 1.0f - std::clamp((distance - v.min_distance) /
                                                         (v.max_distance - v.min_distance),
                                                     0.0f, 1.0f);
                attenuation *= fade;
            }

            float pan = 0.0f;
            if (distance > 1e-4f) {
                pan = std::clamp(glm::dot(to_source / distance, m.listener_right), -1.0f, 1.0f);
            }
            const float p = (pan + 1.0f) * 0.5f;
            const float angle = p * (std::numbers::pi_v<float> * 0.5f);
            gain_l = v.gain * sound_gain * attenuation * std::cos(angle);
            gain_r = v.gain * sound_gain * attenuation * std::sin(angle);
        }

        const auto& frames = v.clip->frames();
        const auto frame_count = static_cast<double>(frames.size());
        for (StereoFrame& o : out) {
            if (v.position >= frame_count) {
                if (v.looping) {
                    v.position = std::fmod(v.position, frame_count);
                } else {
                    v.active = false;
                    break;
                }
            }
            // Linear interpolation: cheap, and adequate for the modest pitch
            // range game SFX use. Anything more needs a real resampler.
            const auto i0 = static_cast<std::size_t>(v.position);
            const std::size_t i1 = (i0 + 1 < frames.size()) ? i0 + 1 : (v.looping ? 0 : i0);
            const auto frac = static_cast<float>(v.position - static_cast<double>(i0));
            const StereoFrame a = frames[i0];
            const StereoFrame b = frames[i1];
            const float l = a.left + (b.left - a.left) * frac;
            const float r = a.right + (b.right - a.right) * frac;
            o.left += l * gain_l;
            o.right += r * gain_r;
            v.position += static_cast<double>(v.pitch);
        }
    }
    m.active_sound_voices.store(active, std::memory_order_relaxed);

    // --- master gain + peak, with a soft limiter --------------------------
    const float master = m.master_gain.load(std::memory_order_relaxed);
    float peak = 0.0f;
    for (StereoFrame& f : out) {
        f = f * master;
        // tanh soft-clip: a dense chord plus several effects can exceed unity,
        // and hard clipping there is a much worse artifact than the gentle
        // compression this applies only to the peaks that need it.
        f.left = std::tanh(f.left);
        f.right = std::tanh(f.right);
        peak = std::max(peak, std::max(std::fabs(f.left), std::fabs(f.right)));
    }
    m.peak_level.store(peak, std::memory_order_relaxed);
}

} // namespace bracken::audio
