#pragma once

#include "bracken/core/pass_timeline.hpp"
#include "bracken/contracts/render_stats.hpp" // PassTiming

#include <cstddef>
#include <SDL3/SDL_gpu.h>
#include <vector>

namespace bracken::gpu {

class GPUDevice;

class GPUDebugGroup {
public:
    GPUDebugGroup(SDL_GPUCommandBuffer* cb, const char* name) noexcept;
    ~GPUDebugGroup() noexcept;
    GPUDebugGroup(const GPUDebugGroup&) = delete;
    GPUDebugGroup& operator=(const GPUDebugGroup&) = delete;
    GPUDebugGroup(GPUDebugGroup&&) = delete;
    GPUDebugGroup& operator=(GPUDebugGroup&&) = delete;

private:
    SDL_GPUCommandBuffer* cb_ = nullptr;
};

inline void insert_debug_label(SDL_GPUCommandBuffer* cb, const char* name) noexcept {
    if (cb && name) {
        SDL_InsertGPUDebugLabel(cb, name);
    }
}

// Per-pass timing for BRACKEN_GPU_TIMER scopes.
//
// TWO clocks, one API. The CPU clock is real: every scope is wall-clock timed
// through core::PassTimeline, so all BRACKEN_GPU_TIMER sites report the cost
// of RECORDING their pass with no call-site change. The GPU clock does not
// exist — SDL3 exposes no timestamp query API (as of SDL 3.4), so
// is_supported() is false, write_begin/write_end are no-ops, and read_results
// reports PassTiming::gpu_ms as nullopt. That nullopt is the point: the old
// API returned 0.0 ms, which reads as "this pass is free" rather than "this
// number does not exist". For real GPU-side cost use RenderDoc/Nsight.
class GPUTimestampPool {
public:
    [[nodiscard]] bool init(GPUDevice& device, size_t pair_count);
    void shutdown();

    // `pass_name` is stored as a raw pointer (no per-frame string alloc):
    // it must be a string literal or otherwise outlive the frame (i.e. stay
    // valid until the next begin_frame()/read_results). Every BRACKEN_GPU_TIMER
    // call site passes a literal.
    int reserve_pair(const char* pass_name);
    void write_begin(SDL_GPUCommandBuffer* cb, int pair_idx);
    void write_end(SDL_GPUCommandBuffer* cb, int pair_idx);

    // Fill `out` with one entry per pass reserved this frame, in reservation
    // order: cpu_ms always present, gpu_ms present only if is_supported().
    void read_results(std::vector<PassTiming>& out);

    void begin_frame();
    void end_frame(SDL_GPUCommandBuffer* cb);

    bool is_supported() const noexcept { return supported_; }

private:
    bool supported_ = false;
    size_t capacity_ = 0;
    std::vector<const char*> pass_names_; // interned literals, cleared per frame
    core::PassTimeline timeline_;         // CPU record-time clock (always live)
};

class GPUTimerScope {
public:
    GPUTimerScope(GPUTimestampPool& pool, SDL_GPUCommandBuffer* cb, const char* pass_name) noexcept;
    /// Nullable overload, for a subsystem that holds an OPTIONAL pool pointer
    /// rather than owning one (render::RtRenderer, whose passes belong to the
    /// scene renderer's timeline but which is constructible without it). A null
    /// pool times nothing and reserves nothing.
    GPUTimerScope(GPUTimestampPool* pool, SDL_GPUCommandBuffer* cb, const char* pass_name) noexcept;
    ~GPUTimerScope() noexcept;
    GPUTimerScope(const GPUTimerScope&) = delete;
    GPUTimerScope& operator=(const GPUTimerScope&) = delete;

private:
    GPUTimestampPool* pool_ = nullptr;
    SDL_GPUCommandBuffer* cb_ = nullptr;
    int pair_idx_ = -1;
};

} // namespace bracken::gpu

#define BRACKEN_GPU_DBG_CAT2(a, b) a##b
#define BRACKEN_GPU_DBG_CAT(a, b) BRACKEN_GPU_DBG_CAT2(a, b)

#define BRACKEN_GPU_SCOPE(cb, name) \
    ::bracken::gpu::GPUDebugGroup BRACKEN_GPU_DBG_CAT(_gpu_dbg_, __LINE__)((cb), (name))

#define BRACKEN_GPU_LABEL(cb, name) ::bracken::gpu::insert_debug_label((cb), (name))

#define BRACKEN_GPU_TIMER(pool, cb, name) \
    ::bracken::gpu::GPUTimerScope BRACKEN_GPU_DBG_CAT(_gpu_tm_, __LINE__)((pool), (cb), (name))
