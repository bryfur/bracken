#pragma once

#include "bracken/gpu/gpu_texture.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>

namespace bracken {

/** Bytes one texel of `format` occupies. */
[[nodiscard]] constexpr std::size_t texture_format_bytes(gpu::TextureFormat format) noexcept {
    switch (format) {
        case gpu::TextureFormat::R8:
            return 1;
        case gpu::TextureFormat::R16:
            return 2;
        case gpu::TextureFormat::RG16F:
            return 4;
        case gpu::TextureFormat::RGBA8:
        case gpu::TextureFormat::RGBA8_SRGB:
        case gpu::TextureFormat::BGRA8:
        case gpu::TextureFormat::D32F:
        case gpu::TextureFormat::D24S8:
            return 4;
    }
    return 4;
}

/**
 * Estimated GPU bytes of a 2D texture, mip chain included when `mipmapped`.
 *
 * ESTIMATE, and the word is load-bearing: SDL3 exposes no allocation size, and
 * tiling, alignment and format promotion are the driver's business. What this
 * gives a residency budget is a stable, monotone, content-derived number —
 * twice the pixels really is twice the cost — which is all a budget needs.
 * Quoting it as VRAM usage would be a lie.
 *
 * The mip chain is charged as the exact sum over levels rather than the 4/3
 * limit: a 1-pixel-tall strip's chain is nothing like 4/3 of its base.
 */
[[nodiscard]] constexpr std::size_t texture_bytes_estimate(int width, int height, gpu::TextureFormat format,
                                                           bool mipmapped) noexcept {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    const std::size_t texel = texture_format_bytes(format);
    std::size_t total = 0;
    int w = width;
    int h = height;
    for (;;) {
        total += static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * texel;
        if (!mipmapped || (w == 1 && h == 1)) {
            break;
        }
        w = w > 1 ? w / 2 : 1;
        h = h > 1 ? h / 2 : 1;
    }
    return total;
}

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
 * REFERENCE COUNTS, AND WHY THEY DO NOT FREE ANYTHING YET.
 *
 * Streaming needs the opposite of the lifetime argument above: evicting a
 * model has to be able to return its images' VRAM. So `acquire()` now counts,
 * `release()` uncounts, and `bytes_resident()` reports — but the count
 * reaching zero DESTROYS the texture only when `set_destroy_on_release(true)`
 * has been called, and the default is `false`.
 *
 * That default is not timidity, it is the risk profile. `Mesh` and
 * `PBRMaterial` hold FOUR non-owning `gpu::GPUTexture*` each and there is no
 * central registry of who borrowed what; a single missed `release()` pairing
 * — an LOD level, an EffectRenderer upload, a hot-reload that replaced a model
 * in place — leaves a live descriptor pointing at a destroyed texture. The
 * failure mode is a GPU fault or garbage pixels three subsystems away, not an
 * assert. With destruction off the counts are still exact and still
 * observable (`refcount()`, `bytes_resident()`, `Stats::released_to_zero`), so
 * a project can VERIFY the pairing on its own content before flipping the
 * switch. Turning it on is a deliberate, measured step; leaving it off
 * reproduces today's never-evicted behaviour byte for byte.
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

    /** Counters a host or dev panel can read without knowing the layout. */
    struct Stats {
        std::size_t images = 0;          ///< distinct images held
        std::size_t bytes_resident = 0;  ///< estimated VRAM of the images held
        std::size_t references = 0;      ///< sum of every entry's refcount
        std::size_t released_to_zero = 0;///< times a refcount hit zero (ever)
        std::size_t destroyed = 0;       ///< of those, how many actually freed
        std::size_t unknown_releases = 0;///< release() calls naming a foreign pointer
    };

    /**
     * Return the texture for these pixels, uploading it only if it is new, and
     * take ONE reference to it.
     *
     * Returns null if creation fails. The returned pointer is owned by the
     * cache and stays valid until `clear()` — or, once
     * `set_destroy_on_release(true)` is in force, until the matching number of
     * `release()` calls have been made.
     */
    [[nodiscard]] gpu::GPUTexture* acquire(gpu::GPUDevice& device, int width, int height,
                                           gpu::TextureFormat format, std::span<const uint8_t> pixels,
                                           bool generate_mipmaps);

    /**
     * Drop one reference taken by `acquire()`.
     *
     * A null pointer, or one this cache does not own, is a no-op counted in
     * `Stats::unknown_releases` (with one throttled warning) rather than an
     * abort: `release()` sits on the eviction path, which walks meshes whose
     * texture slots may legitimately be null or may point at an image an
     * earlier eviction already dropped.
     *
     * Releasing more times than acquired is a caller bug and is clamped at
     * zero with a warning — it is the shape of bug that would otherwise free a
     * texture another model is still bound to.
     *
     * Main-thread. Destroys the texture at zero ONLY when
     * `set_destroy_on_release(true)`; see the class comment for why that is
     * off by default.
     */
    void release(gpu::GPUTexture* texture);

    /**
     * Let a zero refcount actually free the GPU texture.
     *
     * OFF by default. Turn it on only once the project has verified — with
     * `stats()` — that every `acquire()` its content path performs has exactly
     * one matching `release()`. Turning it on retroactively frees nothing:
     * entries already sitting at zero stay alive, because a pointer already
     * handed out may already be bound. Only a subsequent zero transition
     * frees.
     */
    void set_destroy_on_release(bool destroy) noexcept { destroy_on_release_ = destroy; }
    [[nodiscard]] bool destroys_on_release() const noexcept { return destroy_on_release_; }

    /** Release every cached texture. Invalidates every pointer handed out. */
    void clear();

    /** Distinct images currently held. */
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    /** Uploads avoided so far — references served minus images created. Purely
     *  diagnostic; it is what tells you whether an art kit is actually sharing
     *  atlases or just looks like it should be. */
    [[nodiscard]] std::size_t hits() const noexcept { return hits_; }

    /** Estimated bytes of GPU memory the held images occupy, mip chain
     *  included as the EXACT per-level sum (`texture_bytes_estimate` — not the
     *  4/3 limit; a one-pixel-tall strip's chain is nothing like 4/3 of its
     *  base). An estimate and labelled as one:
     *  SDL3 exposes no allocation size, and a driver's tiling and alignment are
     *  its own business. It is a budget input, not an accounting figure. */
    [[nodiscard]] std::size_t bytes_resident() const noexcept { return bytes_resident_; }

    /** Live reference count of a texture this cache owns; 0 for a foreign or
     *  null pointer. Diagnostic — the thing a project checks before enabling
     *  `set_destroy_on_release`. */
    [[nodiscard]] std::size_t refcount(const gpu::GPUTexture* texture) const noexcept;

    /** Estimated bytes of one held image; 0 for a foreign or null pointer.
     *  What ModelManager::gpu_bytes charges a model for its texture slots. */
    [[nodiscard]] std::size_t texture_bytes(const gpu::GPUTexture* texture) const noexcept;

    [[nodiscard]] Stats stats() const noexcept;

private:
    struct Entry {
        std::unique_ptr<gpu::GPUTexture> texture;
        std::size_t refs = 0;
        std::size_t bytes = 0;
    };

    std::unordered_map<uint64_t, Entry> entries_;
    /// Reverse index so release() can find an entry from the borrowed pointer
    /// without rehashing megabytes of pixels the caller no longer has.
    std::unordered_map<const gpu::GPUTexture*, uint64_t> key_of_texture_;
    std::size_t hits_ = 0;
    std::size_t bytes_resident_ = 0;
    std::size_t released_to_zero_ = 0;
    std::size_t destroyed_ = 0;
    std::size_t unknown_releases_ = 0;
    bool destroy_on_release_ = false;
};

} // namespace bracken
