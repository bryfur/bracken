#include "gpu_device.hpp"
#include "bracken/core/assert.hpp"

#include <chrono>
#include "bracken/core/jobs/job_stage.hpp"
#include "bracken/core/logger.hpp"
#include "SDL3/SDL_error.h"
#include "SDL3/SDL_gpu.h"
#include "SDL3/SDL_stdinc.h"
#include "SDL3/SDL_video.h"
#include <cstdint>
#include <cstdlib>
#include <SDL3/SDL_properties.h>
#include <string>

namespace bracken::gpu {

namespace {
constexpr const char* LOG_TAG = "gpu";
} // namespace

GPUDevice::~GPUDevice() {
    shutdown();
}

bool GPUDevice::init(SDL_Window* window, bool prefer_low_power) {
    // GPU device lifecycle is main-thread tier (asserted). Single-threaded
    // test binaries bind "main" on first use — see core::bind_main_thread.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (device_) {
        BRACKEN_LOG_WARN(LOG_TAG, "GPUDevice::init: Already initialized");
        return false;
    }

    window_ = window;

    // GPU validation costs real CPU time per submit, so it is off unless asked
    // for: set_validation_enabled() before init(), or BRACKEN_GPU_DEBUG in the
    // environment (which wins, so it can be flipped without a rebuild). It does
    // NOT key off NDEBUG — what build you are running is the wrong thing to
    // hang a per-submit cost on.
    // getenv is single-threaded at device creation, before any worker thread is spawned.
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char* debug_env = std::getenv("BRACKEN_GPU_DEBUG");
    bool debug_mode = validation_;
    if (debug_env) {
        debug_mode = debug_env[0] != '\0' && SDL_strcmp(debug_env, "0") != 0 &&
                     SDL_strcasecmp(debug_env, "false") != 0 && SDL_strcasecmp(debug_env, "off") != 0;
    }
    BRACKEN_LOG_INFO(LOG_TAG, "GPUDevice::init: GPU validation {} ({})", debug_mode ? "enabled" : "disabled",
                     debug_env ? "from BRACKEN_GPU_DEBUG" : "set_validation_enabled");

    // Create the GPU device with automatic backend selection
    // SDL3 will choose the best available backend (Metal on macOS, Vulkan/D3D12 on others).
    // Properties-based creation so prefer_low_power actually reaches SDL
    // (SDL_CreateGPUDevice has no low-power knob).
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_DEBUGMODE_BOOLEAN, debug_mode);
    SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_PREFERLOWPOWER_BOOLEAN, prefer_low_power);
    SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_SPIRV_BOOLEAN, true);
    SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_METALLIB_BOOLEAN, true);
    SDL_SetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_DXIL_BOOLEAN, true);
    device_ = SDL_CreateGPUDeviceWithProperties(props);
    SDL_DestroyProperties(props);

    if (prefer_low_power) {
        BRACKEN_LOG_INFO(LOG_TAG, "GPUDevice::init: prefer_low_power requested (energy-efficient adapter preferred)");
    }

    if (!device_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUDevice::init: Failed to create GPU device: {}", SDL_GetError());
        return false;
    }

    // Claim the window for the GPU device
    if (!SDL_ClaimWindowForGPUDevice(device_, window_)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUDevice::init: Failed to claim window: {}", SDL_GetError());
        SDL_DestroyGPUDevice(device_);
        device_ = nullptr;
        return false;
    }

    BRACKEN_LOG_INFO(LOG_TAG, "GPUDevice::init: Initialized with driver '{}'", driver_name().c_str());
    return true;
}

void GPUDevice::shutdown() {
    if (device_) {
        // GPU device destruction is main-thread tier.
        BRACKEN_ASSERT_MAIN_THREAD();
        BRACKEN_ASSERT_NOT_SIM_STAGE();
        // Wait for all GPU operations to complete
        SDL_WaitForGPUIdle(device_);

        if (window_) {
            SDL_ReleaseWindowFromGPUDevice(device_, window_);
        }

        SDL_DestroyGPUDevice(device_);
        device_ = nullptr;
        window_ = nullptr;

        BRACKEN_LOG_INFO(LOG_TAG, "GPUDevice::shutdown: Device destroyed");
    }
}

SDL_GPUCommandBuffer* GPUDevice::begin_frame() {
    if (!device_) {
        return nullptr;
    }

    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device_);
    if (!cmd) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUDevice::begin_frame: Failed to acquire command buffer: {}", SDL_GetError());
        return nullptr;
    }

    return cmd;
}

void GPUDevice::end_frame(SDL_GPUCommandBuffer* cmd) {
    if (!cmd) {
        return;
    }

    // Submit can block behind a full queue; it belongs in the same bucket as
    // the swapchain wait (time the CPU spent waiting on the GPU, not working).
    const auto begin = std::chrono::steady_clock::now();
    SDL_SubmitGPUCommandBuffer(cmd);
    present_wait_ns_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
}

bool GPUDevice::end_frame_and_wait(SDL_GPUCommandBuffer* cmd) {
    if (!cmd || !device_) {
        return false;
    }

    const auto begin = std::chrono::steady_clock::now();
    SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (!fence) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUDevice::end_frame_and_wait: submit failed: {}", SDL_GetError());
        return false;
    }
    SDL_WaitForGPUFences(device_, true, &fence, 1);
    SDL_ReleaseGPUFence(device_, fence);
    present_wait_ns_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
    return true;
}

SDL_GPUTexture* GPUDevice::acquire_swapchain_texture(SDL_GPUCommandBuffer* cmd, uint32_t* out_width,
                                                     uint32_t* out_height) {
    // SDL contract: must run on the thread that created the window.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    // SDL_WaitAndAcquireGPUSwapchainTexture BLOCKS until the presentation
    // engine hands back an image, i.e. until the GPU has caught up. With no
    // timestamp query API in SDL3, this stall is the only in-engine signal
    // separating "GPU-bound" from "CPU-bound", so it is measured rather than
    // guessed at. See present_wait_ms().
    const auto wait_begin = std::chrono::steady_clock::now();
    const struct WaitTimer {
        GPUDevice* self;
        std::chrono::steady_clock::time_point begin;
        ~WaitTimer() {
            self->present_wait_ns_ +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
        }
    } wait_timer{this, wait_begin};
    if (!cmd || !window_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUDevice::acquire_swapchain_texture: cmd={} window={}",
                          reinterpret_cast<void*>(cmd), reinterpret_cast<void*>(window_));
        return nullptr;
    }

    SDL_GPUTexture* swapchain_texture = nullptr;
    uint32_t w = 0;
    uint32_t h = 0;

    // Use waiting version to avoid memory bloat from accumulated command buffers
    // See: https://wiki.libsdl.org/SDL3/SDL_WaitAndAcquireGPUSwapchainTexture
    if (!SDL_WaitAndAcquireGPUSwapchainTexture(cmd, window_, &swapchain_texture, &w, &h)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUDevice::acquire_swapchain_texture: SDL call failed: {}", SDL_GetError());
        return nullptr;
    }

    // NULL texture with success means window is minimized or not ready
    if (!swapchain_texture) {
        return nullptr;
    }

    if (out_width) {
        *out_width = w;
    }
    if (out_height) {
        *out_height = h;
    }

    return swapchain_texture;
}

// =============================================================================
// Resource Creation
// =============================================================================
// All SDL_CreateGPU*/SDL_ReleaseGPU* calls funnel through the methods below,
// so the main-thread contract for GPU resource create/destroy (thread-safety
// tier "main-thread" — see docs/multithreading_plan.md §6.1) is asserted at
// this single chokepoint. Until native streaming (Phase 6) introduces a
// main-thread publish queue, creating or destroying a GPU resource off the
// main thread is a contract violation.

SDL_GPUBuffer* GPUDevice::create_buffer(const SDL_GPUBufferCreateInfo& info) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (!device_) {
        return nullptr;
    }
    return SDL_CreateGPUBuffer(device_, &info);
}

SDL_GPUTransferBuffer* GPUDevice::create_transfer_buffer(const SDL_GPUTransferBufferCreateInfo& info) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (!device_) {
        return nullptr;
    }
    return SDL_CreateGPUTransferBuffer(device_, &info);
}

SDL_GPUTexture* GPUDevice::create_texture(const SDL_GPUTextureCreateInfo& info) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (!device_) {
        return nullptr;
    }
    return SDL_CreateGPUTexture(device_, &info);
}

SDL_GPUSampler* GPUDevice::create_sampler(const SDL_GPUSamplerCreateInfo& info) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (!device_) {
        return nullptr;
    }
    return SDL_CreateGPUSampler(device_, &info);
}

SDL_GPUGraphicsPipeline* GPUDevice::create_graphics_pipeline(const SDL_GPUGraphicsPipelineCreateInfo& info) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (!device_) {
        return nullptr;
    }
    return SDL_CreateGPUGraphicsPipeline(device_, &info);
}

SDL_GPUShader* GPUDevice::create_shader(const SDL_GPUShaderCreateInfo& info) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (!device_) {
        return nullptr;
    }
    return SDL_CreateGPUShader(device_, &info);
}

// =============================================================================
// Resource Destruction
// =============================================================================
// Main-thread asserted — see the Resource Creation note above.

void GPUDevice::release_buffer(SDL_GPUBuffer* buffer) {
    if (device_ && buffer) {
        BRACKEN_ASSERT_MAIN_THREAD();
        BRACKEN_ASSERT_NOT_SIM_STAGE();
        SDL_ReleaseGPUBuffer(device_, buffer);
    }
}

void GPUDevice::release_transfer_buffer(SDL_GPUTransferBuffer* buffer) {
    if (device_ && buffer) {
        BRACKEN_ASSERT_MAIN_THREAD();
        BRACKEN_ASSERT_NOT_SIM_STAGE();
        SDL_ReleaseGPUTransferBuffer(device_, buffer);
    }
}

void GPUDevice::release_texture(SDL_GPUTexture* texture) {
    if (device_ && texture) {
        BRACKEN_ASSERT_MAIN_THREAD();
        BRACKEN_ASSERT_NOT_SIM_STAGE();
        SDL_ReleaseGPUTexture(device_, texture);
    }
}

void GPUDevice::release_sampler(SDL_GPUSampler* sampler) {
    if (device_ && sampler) {
        BRACKEN_ASSERT_MAIN_THREAD();
        BRACKEN_ASSERT_NOT_SIM_STAGE();
        SDL_ReleaseGPUSampler(device_, sampler);
    }
}

void GPUDevice::release_graphics_pipeline(SDL_GPUGraphicsPipeline* pipeline) {
    if (device_ && pipeline) {
        BRACKEN_ASSERT_MAIN_THREAD();
        BRACKEN_ASSERT_NOT_SIM_STAGE();
        SDL_ReleaseGPUGraphicsPipeline(device_, pipeline);
    }
}

void GPUDevice::release_shader(SDL_GPUShader* shader) {
    if (device_ && shader) {
        BRACKEN_ASSERT_MAIN_THREAD();
        BRACKEN_ASSERT_NOT_SIM_STAGE();
        SDL_ReleaseGPUShader(device_, shader);
    }
}

void GPUDevice::release_compute_pipeline(SDL_GPUComputePipeline* pipeline) {
    if (device_ && pipeline) {
        BRACKEN_ASSERT_MAIN_THREAD();
        BRACKEN_ASSERT_NOT_SIM_STAGE();
        SDL_ReleaseGPUComputePipeline(device_, pipeline);
    }
}

// =============================================================================
// Transfer Operations
// =============================================================================

void* GPUDevice::map_transfer_buffer(SDL_GPUTransferBuffer* buffer, bool cycle) {
    if (!device_ || !buffer) {
        return nullptr;
    }
    return SDL_MapGPUTransferBuffer(device_, buffer, cycle);
}

void GPUDevice::unmap_transfer_buffer(SDL_GPUTransferBuffer* buffer) {
    if (device_ && buffer) {
        SDL_UnmapGPUTransferBuffer(device_, buffer);
    }
}

// =============================================================================
// Accessors
// =============================================================================

int GPUDevice::width() const {
    if (!window_) {
        return 0;
    }
    int w = 0;
    SDL_GetWindowSize(window_, &w, nullptr);
    return w;
}

int GPUDevice::height() const {
    if (!window_) {
        return 0;
    }
    int h = 0;
    SDL_GetWindowSize(window_, nullptr, &h);
    return h;
}

SDL_GPUTextureFormat GPUDevice::swapchain_format() const {
    if (!device_ || !window_) {
        return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM; // Default fallback
    }
    return SDL_GetGPUSwapchainTextureFormat(device_, window_);
}

std::string GPUDevice::driver_name() const {
    if (!device_) {
        return "none";
    }
    const char* name = SDL_GetGPUDeviceDriver(device_);
    return name ? name : "unknown";
}

bool GPUDevice::supports_format(SDL_GPUTextureFormat format, SDL_GPUTextureType type,
                                SDL_GPUTextureUsageFlags usage) const {
    if (!device_) {
        return false;
    }
    return SDL_GPUTextureSupportsFormat(device_, format, type, usage);
}

bool GPUDevice::supports_present_mode(SDL_GPUPresentMode mode) const {
    if (!device_ || !window_) {
        return false;
    }
    return SDL_WindowSupportsGPUPresentMode(device_, window_, mode);
}

bool GPUDevice::set_swapchain_parameters(SDL_GPUPresentMode present_mode) {
    if (!device_ || !window_) {
        return false;
    }

    if (!SDL_SetGPUSwapchainParameters(device_, window_, SDL_GPU_SWAPCHAINCOMPOSITION_SDR, present_mode)) {
        BRACKEN_LOG_WARN(LOG_TAG, "GPUDevice::set_swapchain_parameters: present_mode={} not supported: {}",
                         static_cast<int>(present_mode), SDL_GetError());
        return false;
    }
    BRACKEN_LOG_INFO(LOG_TAG, "GPUDevice::set_swapchain_parameters: present_mode={}", static_cast<int>(present_mode));
    return true;
}

} // namespace bracken::gpu
