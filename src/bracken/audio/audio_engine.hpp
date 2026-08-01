#pragma once

#include "bracken/audio/audio_clip.hpp"
#include "bracken/audio/audio_format.hpp"
#include "bracken/audio/command_ring.hpp"
#if BRACKEN_SYNTH
#include "bracken/audio/music/sequencer.hpp"
#endif

#include <atomic>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <vector>

// For SDL_AudioStream and SDLCALL on the device callback. SDL is already part
// of the engine's public surface (application.hpp includes SDL_events.h) and
// is not on public_headers.test.cpp's forbidden list, unlike Jolt/ImGui.
#include <SDL3/SDL_audio.h>

namespace bracken::audio {

/** How a sound is positioned. */
enum class SoundSpace : uint8_t {
    /** Fixed in the mix — UI clicks, narration, music stings. Not attenuated
     *  or panned by listener pose. */
    UI,
    /** Placed in the world: panned and attenuated relative to the listener. */
    World,
};

struct SoundParams {
    float gain = 1.0f;      // linear
    float pitch = 1.0f;     // playback rate multiplier; 1 = as recorded
    SoundSpace space = SoundSpace::UI;
    glm::vec3 position{0.0f};
    /** Distance at which the sound is at full volume; beyond it, gain falls
     *  off. Named rather than hard-coded because the right value depends
     *  entirely on world scale, and this engine's worlds are in the
     *  thousands of units. */
    float min_distance = 200.0f;
    float max_distance = 4000.0f;
    bool looping = false;
};

/** Handle to a clip registered with the engine. */
struct SoundId {
    static constexpr uint32_t k_invalid = 0xFFFFFFFFu;
    uint32_t value = k_invalid;
    [[nodiscard]] bool valid() const noexcept { return value != k_invalid; }
};

/** Snapshot of what the audio thread is doing, for the dev overlay. */
struct AudioDiagnostics {
    int active_sound_voices = 0;
    int active_music_voices = 0;
    uint64_t commands_dropped = 0;  // ring overflow: game out-producing the mixer
    uint64_t buffer_underruns = 0;  // callback asked for more than we had
    float peak_level = 0.0f;        // post-master, for a level meter
    bool device_open = false;
};

/**
 * The engine's audio system: an SDL3 device, a mixer, and the game-facing API
 * for making noise.
 *
 * THREADING. This is the only subsystem in the engine with a thread it does
 * not own: SDL calls back on its own audio thread whenever the device wants
 * more samples. That thread may never block, allocate, or log, so everything
 * crossing into it goes through a bounded lock-free ring (command_ring.hpp)
 * and every buffer it touches is pre-sized. The consequence for hosts is
 * pleasant: play_sound / play_music / set_listener are **worker-safe** and can
 * be called straight from on_simulate with no locking and no ordering rules.
 *
 * OWNERSHIP. Clips are registered once and referred to by SoundId. The audio
 * thread reads clip sample data directly, so a registered clip lives until
 * shutdown — the engine never hands the audio thread a pointer whose lifetime
 * the host controls, the same rule FramePacket follows.
 *
 * The synthesizer and the sequencer are compiled in unless
 * BRACKEN_ENABLE_SYNTH=OFF, in which case play_music is a no-op and a game
 * that only plays recorded audio pays nothing for them.
 */
class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();

    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;

    /** Open the default playback device and start the mixer. Returns false
     *  (and logs) if no device is available — a machine with no sound card,
     *  or a CI box, is not a fatal condition and the rest of the engine runs
     *  normally with every audio call becoming a no-op.
     *  Tier: main-thread. */
    [[nodiscard]] bool init();

    /** Stop the device and release everything. Safe to call twice.
     *  Tier: main-thread. */
    void shutdown();

    [[nodiscard]] bool device_open() const noexcept { return device_open_; }

    // ========== Asset registration (main-thread, before playback) ==========

    /** Register a clip and take ownership of its samples. */
    [[nodiscard]] SoundId register_clip(AudioClip clip);

    /** Convenience: load a WAV and register it. Returns an invalid SoundId
     *  and logs on failure — a missing sound effect should not stop a game
     *  from booting. */
    [[nodiscard]] SoundId load_sound(const std::string& path);

    // ========== Playback (worker-safe: callable from on_simulate) =========

    /** Fire a one-shot (or start a loop). Returns immediately; the sound
     *  starts at the next mixer buffer. Dropped with a diagnostic if the
     *  command ring is full or the id is invalid. */
    void play_sound(SoundId id, const SoundParams& params = {});

    /** Stop every voice playing this clip. Mainly for looping sounds. */
    void stop_sound(SoundId id);

    /** Where the listener is. Usually the camera; call it once per frame.
     *  `forward` and `up` orient the stereo field. */
    void set_listener(const glm::vec3& position, const glm::vec3& forward, const glm::vec3& up);

    /** Master output gain, linear. Applied after everything else. */
    void set_master_gain(float gain);
    void set_music_gain(float gain);
    void set_sound_gain(float gain);

    // ========== Music (the synthesizer) ==================================

#if BRACKEN_SYNTH
    /** Hand the sequencer a song and start it. The song is copied — the
     *  audio thread reads it, so a host must not retain a reference. */
    void play_music(music::Song song);
#endif
    void stop_music();
    void set_music_playing(bool playing);
    [[nodiscard]] bool music_playing() const noexcept;

    // ========== Diagnostics ==============================================

    [[nodiscard]] AudioDiagnostics diagnostics() const;

    /**
     * Render `frames` of the full mix into `out` WITHOUT a device, exactly as
     * the audio callback would.
     *
     * This is what makes the audio system testable and what lets a song be
     * rendered to a file for inspection. It is also how the engine's own
     * tests verify the mixer without requiring a sound card on CI.
     * Tier: caller-owned — do not call while a device is open.
     */
    void render_offline(std::span<StereoFrame> out);

private:
    // The audio thread's view. Everything here is touched ONLY by the audio
    // callback (or by render_offline, which is documented as device-free).
    struct Mixer;

    static void SDLCALL stream_callback(void* userdata, SDL_AudioStream* stream,
                                        int additional_amount, int total_amount);
    void mix(std::span<StereoFrame> out);

    std::unique_ptr<Mixer> mixer_;
    SDL_AudioStream* stream_ = nullptr;
    bool device_open_ = false;

    // Scratch the callback mixes into before handing bytes to SDL. Sized at
    // init and never reallocated, because the audio thread must not allocate.
    std::vector<StereoFrame> scratch_;
};

} // namespace bracken::audio
