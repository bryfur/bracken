#pragma once

#include "bracken/render_graph/render_graph_resource.hpp"
#include <cstdint>
#include <SDL3/SDL_gpu.h>
#include <unordered_map>
#include <vector>

namespace bracken::gpu {
class GPUDevice;
}

namespace bracken::render_graph {

// Pools transient SDL_GPUTextures keyed by their TextureDesc.
// On acquire(), pops a free texture matching desc, or creates a new one.
// On release(), pushes back into the pool. release_all() returns everything.
// Underlying SDL textures persist across frames until evict_all() (e.g. on resize).
class TransientAllocator {
public:
    TransientAllocator() = default;
    ~TransientAllocator();

    TransientAllocator(const TransientAllocator&) = delete;
    TransientAllocator& operator=(const TransientAllocator&) = delete;

    // Bind the GPU device used to create textures. Cannot fail. May be skipped
    // for device-less tests: acquire_texture then returns non-null sentinels.
    void init(gpu::GPUDevice& device);
    void shutdown();

    // Acquire a texture matching desc. If the pool has a free one, reuses it;
    // otherwise creates one. Returns nullptr if creation fails (logged).
    SDL_GPUTexture* acquire_texture(const TextureDesc& desc, const char* debug_name);

    // Return a texture to the pool for reuse on next acquire.
    void release_texture(const TextureDesc& desc, SDL_GPUTexture* tex);

    // Stats for telemetry.
    uint32_t acquires_this_frame() const noexcept { return acquires_; }
    uint32_t reuses_this_frame() const noexcept { return reuses_; }
    uint32_t pool_size() const noexcept;

    void begin_frame() noexcept {
        acquires_ = 0;
        reuses_ = 0;
    }

    // Release all textures back into pool keyed by desc (called at end_frame()).
    void release_all();

    // Destroy every pooled texture (called on resize / shutdown).
    void evict_all();

    // Test seam: cap the number of texture *creations* (pool reuses are
    // unaffected). Once the budget is exhausted, acquire_texture returns
    // nullptr as if SDL_CreateGPUTexture had failed. -1 (default) = unlimited.
    void set_texture_create_budget_for_testing(int64_t budget) noexcept { create_budget_ = budget; }

private:
    struct DescKey {
        TextureDesc desc;
        bool operator==(const DescKey& o) const noexcept { return desc == o.desc; }
    };
    struct DescKeyHash {
        size_t operator()(const DescKey& k) const noexcept;
    };

    struct Entry {
        SDL_GPUTexture* tex = nullptr;
        bool in_use = false;
    };

    gpu::GPUDevice* device_ = nullptr;
    std::unordered_map<DescKey, std::vector<Entry>, DescKeyHash> pool_;

    uint32_t acquires_ = 0;
    uint32_t reuses_ = 0;
    int64_t create_budget_ = -1; // test seam; -1 = unlimited
};

} // namespace bracken::render_graph
