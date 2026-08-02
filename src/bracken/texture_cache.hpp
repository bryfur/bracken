#pragma once

#include "bracken/gpu/gpu_texture.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>

namespace bracken {

/**
 * Content-addressed GPU texture cache: one upload per distinct image, shared by
 * every mesh of every model that references it.
 *
 * WHY THIS EXISTS. `Mesh` used to own its textures outright
 * (`std::unique_ptr<gpu::GPUTexture>` per slot), so N models built from one art
 * kit uploaded that kit's atlases N times. A village of 40 buildings sharing
 * eight material atlases paid for 320 texture uploads' worth of VRAM to show
 * eight images.
 *
 * That single limitation is why `examples/playground` shipped its village as
 * ONE merged 16 MB `.glb`: merging per-material was the only way to make the
 * kit upload each atlas once. The merge in turn destroyed the per-building node
 * hierarchy, which then had to be smuggled back in as a generated C++ header of
 * `constexpr` rects. Two workarounds, one root cause — this class.
 *
 * KEYING. The key is a 64-bit FNV-1a over the decoded pixel bytes plus
 * width/height/format/mipmap flag. Content, not file path: the same atlas
 * arriving through two different `.glb`s is one entry, and two genuinely
 * different images that happen to share a URI are two. Hashing megabytes per
 * image is far cheaper than uploading them, and it happens once per distinct
 * image rather than once per reference.
 *
 * LIFETIME. The cache owns every texture and outlives all models — it is a
 * member of `ModelManager`, destroyed after the models that borrow from it.
 * `Mesh` therefore holds NON-OWNING `gpu::GPUTexture*`, which is also why
 * `Mesh::shared_texture` no longer exists: an LOD level asking for the same
 * pixels as its base level is just an ordinary cache hit, not a special case
 * with its own borrowed-pointer slot and its own lifetime argument.
 *
 * Tier: main-thread (creates and destroys GPU textures).
 */
class TextureCache {
public:
    TextureCache() = default;
    ~TextureCache() = default;
    TextureCache(const TextureCache&) = delete;
    TextureCache& operator=(const TextureCache&) = delete;
    TextureCache(TextureCache&&) = delete;
    TextureCache& operator=(TextureCache&&) = delete;

    /**
     * Return the texture for these pixels, uploading it only if it is new.
     *
     * Returns null if creation fails. The returned pointer is owned by the
     * cache and stays valid until `clear()`.
     */
    [[nodiscard]] gpu::GPUTexture* acquire(gpu::GPUDevice& device, int width, int height,
                                           gpu::TextureFormat format, std::span<const uint8_t> pixels,
                                           bool generate_mipmaps);

    /** Release every cached texture. Invalidates every pointer handed out. */
    void clear();

    /** Distinct images currently held. */
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    /** Uploads avoided so far — references served minus images created. Purely
     *  diagnostic; it is what tells you whether an art kit is actually sharing
     *  atlases or just looks like it should be. */
    [[nodiscard]] std::size_t hits() const noexcept { return hits_; }

private:
    std::unordered_map<uint64_t, std::unique_ptr<gpu::GPUTexture>> entries_;
    std::size_t hits_ = 0;
};

} // namespace bracken
