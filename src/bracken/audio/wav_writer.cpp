#include "bracken/audio/wav_writer.hpp"

#include "bracken/core/logger.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace bracken::audio {
namespace {

void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(static_cast<uint8_t>(v & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

void put_u16(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back(static_cast<uint8_t>(v & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

void put_tag(std::vector<uint8_t>& b, const char (&tag)[5]) {
    b.insert(b.end(), tag, tag + 4);
}

int16_t to_pcm16(float sample) {
    // Clamp before scaling: a value past ±1 would otherwise wrap around to
    // the opposite polarity and turn a hot passage into digital noise.
    const float clamped = std::clamp(sample, -1.0f, 1.0f);
    return static_cast<int16_t>(std::lround(clamped * 32767.0f));
}

} // namespace

bool write_wav(const std::string& path, std::span<const StereoFrame> frames, int sample_rate) {
    constexpr uint16_t k_bits = 16;
    const auto frame_count = static_cast<uint32_t>(frames.size());
    const uint32_t data_bytes = frame_count * k_channels * (k_bits / 8);

    std::vector<uint8_t> header;
    header.reserve(44);
    put_tag(header, "RIFF");
    put_u32(header, 36 + data_bytes);
    put_tag(header, "WAVE");
    put_tag(header, "fmt ");
    put_u32(header, 16);                                     // PCM chunk size
    put_u16(header, 1);                                      // format = PCM
    put_u16(header, static_cast<uint16_t>(k_channels));
    put_u32(header, static_cast<uint32_t>(sample_rate));
    put_u32(header, static_cast<uint32_t>(sample_rate * k_channels * (k_bits / 8))); // byte rate
    put_u16(header, static_cast<uint16_t>(k_channels * (k_bits / 8)));               // block align
    put_u16(header, k_bits);
    put_tag(header, "data");
    put_u32(header, data_bytes);

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        BRACKEN_LOG_ERROR("audio", "write_wav('{}'): could not open for writing", path);
        return false;
    }
    out.write(reinterpret_cast<const char*>(header.data()),
              static_cast<std::streamsize>(header.size()));

    std::vector<int16_t> pcm;
    pcm.reserve(static_cast<std::size_t>(frame_count) * k_channels);
    for (const StereoFrame& f : frames) {
        pcm.push_back(to_pcm16(f.left));
        pcm.push_back(to_pcm16(f.right));
    }
    out.write(reinterpret_cast<const char*>(pcm.data()),
              static_cast<std::streamsize>(pcm.size() * sizeof(int16_t)));

    if (!out) {
        BRACKEN_LOG_ERROR("audio", "write_wav('{}'): write failed", path);
        return false;
    }
    BRACKEN_LOG_INFO("audio", "Wrote {} ({} frames, {:.2f}s)", path, frame_count,
                     static_cast<float>(frame_count) / static_cast<float>(sample_rate));
    return true;
}

} // namespace bracken::audio
