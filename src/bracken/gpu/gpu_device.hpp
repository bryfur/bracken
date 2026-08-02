#pragma once

#include <memory>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_video.h>
#include <cstdint>
#include <string>

namespace bracken::gpu {

/**
 * @brief GPU Device wrapper for SDL3 GPU API
 *
 * This class manages the GPU device lifecycle and provides a clean interface
 * for frame management and resource creation. It encapsulates the SDL3 GPU
 * device and handles the command buffer acquire/submit cycle.
 *
 * Usage:
 *   GPUDevice device;
 *   device.init(window);
 *
 *   // Each frame:
 *   auto* cmd = device.begin_frame();
 *   // ... render commands ...
 *   device.end_frame(cmd);
 *
 *   device.shutdown();
 */
class GPUDevice {
public:
    GPUDevice() = default;
    ~GPUDevice();

    // Non-copyable, non-movable by design:
    // 1. GPUDevice owns the SDL_GPUDevice which manages global GPU state
    // 2. Other GPU resources (buffers, textures, etc.) hold raw pointers to GPUDevice
    // 3. Moving would invalidate those pointers and complicate lifetime management
    // 4. There should typically be only one GPUDevice per application
    GPUDevice(const GPUDevice&) = delete;
    GPUDevice& operator=(const GPUDevice&) = delete;
    GPUDevice(GPUDevice&&) = delete;
    GPUDevice& operator=(GPUDevice&&) = delete;

    /**
     * @brief Initialize the GPU device
     * @param window The SDL window to render to
     * @param prefer_low_power If true, prefer the energy-efficient adapter
     *        (typically the integrated GPU) over maximum performance; wired to
     *        SDL_PROP_GPU_DEVICE_CREATE_PREFERLOWPOWER_BOOLEAN
     * @return true on success, false on failure
     */
    [[nodiscard]] bool init(SDL_Window* window, bool prefer_low_power = false);

    /**
     * @brief Enable the backend's validation layer. Call before init().
     *
     * Off by default and a runtime setting, not a build-time one: validation
     * costs real CPU time per submit, but which build you happen to be running
     * is the wrong thing to hang that on — a Debug build that never validates
     * and a release build that cannot are both worse than one switch.
     */
    void set_validation_enabled(bool enabled) noexcept { validation_ = enabled; }
    [[nodiscard]] bool validation_enabled() const noexcept { return validation_; }

    /**
     * @brief Shutdown and release all GPU resources
     */
    void shutdown();

    /**
     * @brief Check if the device is initialized
     */
    bool is_initialized() const { return device_ != nullptr; }

    // =========================================================================
    // Frame Lifecycle
    // =========================================================================

    /**
     * @brief Begin a new frame and acquire a command buffer
     * @return Command buffer for recording commands, or nullptr on failure
     */
    SDL_GPUCommandBuffer* begin_frame();

    /**
     * @brief End the frame and submit the command buffer
     * @param cmd The command buffer to submit
     */
    // Non-static (was static): the submit stall is accumulated into
    // present_wait_ms(). Every caller already goes through an instance.
    void end_frame(SDL_GPUCommandBuffer* cmd);

    /**
     * @brief Submit the command buffer and block until the GPU has executed it.
     *
     * Stalls the pipeline — readback paths only (screenshots), never the frame
     * loop. @return false if submission failed.
     */
    [[nodiscard]] bool end_frame_and_wait(SDL_GPUCommandBuffer* cmd);

    /**
     * @brief Acquire the swapchain texture for the current frame
     * @param cmd The command buffer
     * @param out_width Output width of the swapchain texture
     * @param out_height Output height of the swapchain texture
     * @return The swapchain texture, or nullptr if unavailable (e.g., window minimized)
     *
     * @warning Must be called from the thread that created the window.
     *          See: https://wiki.libsdl.org/SDL3/SDL_WaitAndAcquireGPUSwapchainTexture
     */
    SDL_GPUTexture* acquire_swapchain_texture(SDL_GPUCommandBuffer* cmd, uint32_t* out_width = nullptr,
                                              uint32_t* out_height = nullptr);

    // =========================================================================
    // Resource Creation (Factory Methods)
    //
    // Thread-safety tier: main-thread (asserted). Every create_*/release_*
    // below funnels an SDL_CreateGPU*/SDL_ReleaseGPU* call and must run on
    // the engine main thread — see docs/multithreading_plan.md §6.1. Native
    // streaming (Phase 6) will keep this contract via a main-thread publish
    // queue rather than relaxing it.
    // =========================================================================

    /**
     * @brief Create a GPU buffer
     */
    SDL_GPUBuffer* create_buffer(const SDL_GPUBufferCreateInfo& info);

    /**
     * @brief Create a transfer buffer for uploads/downloads
     */
    SDL_GPUTransferBuffer* create_transfer_buffer(const SDL_GPUTransferBufferCreateInfo& info);

    /**
     * @brief Create a GPU texture
     */
    SDL_GPUTexture* create_texture(const SDL_GPUTextureCreateInfo& info);

    /**
     * @brief Create a GPU sampler
     */
    SDL_GPUSampler* create_sampler(const SDL_GPUSamplerCreateInfo& info);

    /**
     * @brief Create a graphics pipeline
     */
    SDL_GPUGraphicsPipeline* create_graphics_pipeline(const SDL_GPUGraphicsPipelineCreateInfo& info);

    /**
     * @brief Create a shader from bytecode
     */
    SDL_GPUShader* create_shader(const SDL_GPUShaderCreateInfo& info);

    // =========================================================================
    // Resource Destruction (main-thread, asserted — see Resource Creation)
    // =========================================================================

    void release_buffer(SDL_GPUBuffer* buffer);
    void release_transfer_buffer(SDL_GPUTransferBuffer* buffer);
    void release_texture(SDL_GPUTexture* texture);
    void release_sampler(SDL_GPUSampler* sampler);
    void release_graphics_pipeline(SDL_GPUGraphicsPipeline* pipeline);
    void release_shader(SDL_GPUShader* shader);
    void release_compute_pipeline(SDL_GPUComputePipeline* pipeline);

    // =========================================================================
    // Transfer Operations
    // =========================================================================

    /**
     * @brief Map a transfer buffer for CPU access
     */
    void* map_transfer_buffer(SDL_GPUTransferBuffer* buffer, bool cycle);

    /**
     * @brief Unmap a transfer buffer
     */
    void unmap_transfer_buffer(SDL_GPUTransferBuffer* buffer);

    // =========================================================================
    // Accessors
    // =========================================================================

    /**
     * @brief Check if a swapchain present mode is supported
     */
    bool supports_present_mode(SDL_GPUPresentMode mode) const;

    /**
     * @brief Set swapchain present mode and composition
     * @param present_mode SDL_GPU_PRESENTMODE_VSYNC, IMMEDIATE, or MAILBOX
     * @return true if the mode was set, false if unsupported
     */
    [[nodiscard]] bool set_swapchain_parameters(SDL_GPUPresentMode present_mode);

    SDL_GPUDevice* handle() const { return device_; }
    SDL_Window* window() const { return window_; }

    int width() const;
    int height() const;

    /**
     * @brief Get the swapchain texture format
     */
    SDL_GPUTextureFormat swapchain_format() const;

    /**
     * @brief Get the device driver name (e.g., "metal", "vulkan", "d3d12")
     */
    std::string driver_name() const;

    /**
     * @brief Check if a texture format is supported
     */
    bool supports_format(SDL_GPUTextureFormat format, SDL_GPUTextureType type, SDL_GPUTextureUsageFlags usage) const;

    /**
     * Milliseconds the CPU spent BLOCKED on the GPU since the last
     * reset_present_wait(): swapchain acquire (which waits for the
     * presentation engine to return an image) plus command-buffer submit.
     *
     * SDL3 exposes no GPU timestamp query, so no engine-side number can say
     * how long a pass took on the device. This is the one thing that CAN be
     * measured honestly, and it answers the first question anyone asks about a
     * depth-2 pipelined frame: a large value means the frame is GPU-bound and
     * the CPU is idling on it; a near-zero value means the CPU is the limit
     * and optimizing shaders will change nothing. Main-thread tier.
     */
    [[nodiscard]] float present_wait_ms() const noexcept {
        return static_cast<float>(static_cast<double>(present_wait_ns_) * 1e-6);
    }
    void reset_present_wait() noexcept { present_wait_ns_ = 0; }

private:
    SDL_GPUDevice* device_ = nullptr;
    bool validation_ = false; // see set_validation_enabled()
    SDL_Window* window_ = nullptr;
    // Accumulated CPU-blocked-on-GPU nanoseconds; plain integer, main-thread
    // only (every accumulating call site is main-thread asserted).
    int64_t present_wait_ns_ = 0;
};

} // namespace bracken::gpu
