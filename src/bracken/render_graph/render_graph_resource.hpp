#pragma once

#include <cstdint>
#include <limits>
#include <SDL3/SDL_gpu.h>
#include <string>

namespace bracken::render_graph {

struct TextureDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    SDL_GPUTextureFormat format = SDL_GPU_TEXTUREFORMAT_INVALID;
    SDL_GPUTextureUsageFlags usage = 0;
    uint32_t mip_levels = 1;
    uint32_t array_layers = 1;
    uint32_t sample_count = 1;

    bool operator==(const TextureDesc& other) const noexcept {
        return width == other.width && height == other.height && format == other.format && usage == other.usage &&
               mip_levels == other.mip_levels && array_layers == other.array_layers &&
               sample_count == other.sample_count;
    }
};

class ResourceHandle {
public:
    constexpr ResourceHandle() noexcept = default;
    explicit constexpr ResourceHandle(uint32_t idx) noexcept : idx_(idx) {}

    constexpr bool valid() const noexcept { return idx_ != INVALID; }
    constexpr uint32_t index() const noexcept { return idx_; }

    static constexpr ResourceHandle invalid() noexcept { return ResourceHandle{}; }

    constexpr bool operator==(const ResourceHandle& o) const noexcept { return idx_ == o.idx_; }
    constexpr bool operator!=(const ResourceHandle& o) const noexcept { return idx_ != o.idx_; }

private:
    static constexpr uint32_t INVALID = std::numeric_limits<uint32_t>::max();
    uint32_t idx_ = INVALID;
};

// The graph manages textures only. (Buffer resources were scaffolding with no
// declare/import API and were removed; re-add alongside a real buffer API.)
struct ResourceNode {
    std::string name;

    // For transient resources owned by the graph.
    bool transient = false;
    TextureDesc texture_desc{};

    // For imported resources (graph does not own).
    SDL_GPUTexture* imported_texture = nullptr;
    uint32_t imported_width = 0;
    uint32_t imported_height = 0;
    SDL_GPUTextureFormat imported_format = SDL_GPU_TEXTUREFORMAT_INVALID;

    // Resolved at compile-time (transient resources): the real GPU object.
    SDL_GPUTexture* resolved_texture = nullptr;

    // Lifetime: indices into compiled pass list. UINT32_MAX = unset.
    uint32_t first_write_pass = std::numeric_limits<uint32_t>::max();
    uint32_t last_read_pass = std::numeric_limits<uint32_t>::max();
};

} // namespace bracken::render_graph
