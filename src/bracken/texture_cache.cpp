#include "bracken/texture_cache.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_stage.hpp"
#include "bracken/core/log_throttle.hpp"
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
        ++it->second.refs;
        return it->second.texture.get();
    }

    auto texture = gpu::GPUTexture::create_2d(device, width, height, format, pixels.data(), generate_mipmaps);
    if (!texture) {
        BRACKEN_LOG_ERROR(LOG_TAG, "TextureCache: failed to create {}x{} texture", width, height);
        return nullptr;
    }
    gpu::GPUTexture* raw = texture.get();
    Entry entry;
    entry.bytes = texture_bytes_estimate(width, height, format, generate_mipmaps);
    entry.refs = 1;
    entry.texture = std::move(texture);
    bytes_resident_ += entry.bytes;
    entries_.emplace(key, std::move(entry));
    key_of_texture_.emplace(raw, key);
    return raw;
}

void TextureCache::release(gpu::GPUTexture* texture) {
    // Symmetric with acquire(): it may destroy a GPU texture.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (texture == nullptr) {
        return;
    }
    auto rev = key_of_texture_.find(texture);
    if (rev == key_of_texture_.end()) {
        // Not ours. Eviction walks mesh slots that may point at an image an
        // earlier eviction already dropped, or at a texture some other
        // subsystem owns outright — counted, warned once, never fatal.
        ++unknown_releases_;
        BRACKEN_LOG_WARN_ONCE(LOG_TAG, "TextureCache::release: pointer is not owned by this cache (ignored)");
        return;
    }
    auto it = entries_.find(rev->second);
    if (it == entries_.end()) {
        ++unknown_releases_;
        return;
    }
    Entry& entry = it->second;
    if (entry.refs == 0) {
        // Released more often than acquired. Clamped rather than wrapped: an
        // underflowed count is a count that will free a texture somebody is
        // still bound to, which is the exact failure this class must not have.
        BRACKEN_LOG_WARN(LOG_TAG, "TextureCache::release: refcount already zero; the acquire/release pairing is wrong");
        return;
    }
    --entry.refs;
    if (entry.refs != 0) {
        return;
    }
    ++released_to_zero_;
    if (!destroy_on_release_) {
        // The default. The count is exact and observable, the texture stays
        // alive, and every existing borrower keeps working — see the class
        // comment for why this switch exists at all.
        return;
    }
    bytes_resident_ -= entry.bytes;
    key_of_texture_.erase(rev);
    entries_.erase(it);
    ++destroyed_;
}

std::size_t TextureCache::refcount(const gpu::GPUTexture* texture) const noexcept {
    if (texture == nullptr) {
        return 0;
    }
    auto rev = key_of_texture_.find(texture);
    if (rev == key_of_texture_.end()) {
        return 0;
    }
    auto it = entries_.find(rev->second);
    return it == entries_.end() ? 0 : it->second.refs;
}

std::size_t TextureCache::texture_bytes(const gpu::GPUTexture* texture) const noexcept {
    if (texture == nullptr) {
        return 0;
    }
    auto rev = key_of_texture_.find(texture);
    if (rev == key_of_texture_.end()) {
        return 0;
    }
    auto it = entries_.find(rev->second);
    return it == entries_.end() ? 0 : it->second.bytes;
}

TextureCache::Stats TextureCache::stats() const noexcept {
    Stats s;
    s.images = entries_.size();
    s.bytes_resident = bytes_resident_;
    for (const auto& [key, entry] : entries_) {
        (void)key;
        s.references += entry.refs;
    }
    s.released_to_zero = released_to_zero_;
    s.destroyed = destroyed_;
    s.unknown_releases = unknown_releases_;
    return s;
}

void TextureCache::clear() {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    entries_.clear();
    key_of_texture_.clear();
    hits_ = 0;
    bytes_resident_ = 0;
    released_to_zero_ = 0;
    destroyed_ = 0;
    unknown_releases_ = 0;
}

} // namespace bracken
