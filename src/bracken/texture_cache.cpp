#include "bracken/texture_cache.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_stage.hpp"
#include "bracken/core/logger.hpp"

namespace bracken {

namespace {
constexpr const char* LOG_TAG = "model";

// FNV-1a 64 over the pixel bytes and the parameters that change what gets
// uploaded. Not a cryptographic hash and does not need to be: a collision would
// have to match byte-for-byte on dimensions AND format AND mipmap flag, and the
// cost of being wrong is a shared texture rather than memory corruption. A
// stronger hash would cost more than the upload it is avoiding.
uint64_t hash_image(int width, int height, gpu::TextureFormat format, std::span<const uint8_t> pixels,
                    bool generate_mipmaps) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            h ^= (v >> (i * 8)) & 0xFF;
            h *= 1099511628211ULL;
        }
    };
    mix(static_cast<uint64_t>(width));
    mix(static_cast<uint64_t>(height));
    mix(static_cast<uint64_t>(format));
    mix(generate_mipmaps ? 1ULL : 0ULL);
    mix(pixels.size());
    for (uint8_t byte : pixels) {
        h ^= byte;
        h *= 1099511628211ULL;
    }
    return h;
}
} // namespace

gpu::GPUTexture* TextureCache::acquire(gpu::GPUDevice& device, int width, int height, gpu::TextureFormat format,
                                       std::span<const uint8_t> pixels, bool generate_mipmaps) {
    // Creates GPU textures, so it carries the same tier as the upload it
    // replaces.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (pixels.empty() || width <= 0 || height <= 0) {
        return nullptr;
    }

    const uint64_t key = hash_image(width, height, format, pixels, generate_mipmaps);
    if (auto it = entries_.find(key); it != entries_.end()) {
        ++hits_;
        return it->second.get();
    }

    auto texture = gpu::GPUTexture::create_2d(device, width, height, format, pixels.data(), generate_mipmaps);
    if (!texture) {
        BRACKEN_LOG_ERROR(LOG_TAG, "TextureCache: failed to create {}x{} texture", width, height);
        return nullptr;
    }
    gpu::GPUTexture* raw = texture.get();
    entries_.emplace(key, std::move(texture));
    return raw;
}

void TextureCache::clear() {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    entries_.clear();
    hits_ = 0;
}

} // namespace bracken
