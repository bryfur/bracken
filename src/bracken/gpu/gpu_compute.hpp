#pragma once

#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace bracken::gpu {

/**
 * @brief Compute-pipeline resource requirements
 *
 * Unlike graphics shaders, SDL3 GPU compute distinguishes read-only from
 * read-write storage resources (they live in different binding sets — see the
 * binding-order note on GPUComputePipeline). At pipeline creation these
 * declared counts are cross-checked against SPIR-V reflection
 * (SDL_ShaderCross_ReflectComputeSPIRV). On mismatch an error is logged and
 * the reflected counts win, so a stale hand-maintained count can no longer
 * silently misbind resources.
 */
struct ComputeResources {
    uint32_t num_samplers = 0;
    uint32_t num_readonly_storage_textures = 0;
    uint32_t num_readonly_storage_buffers = 0;
    uint32_t num_readwrite_storage_textures = 0;
    uint32_t num_readwrite_storage_buffers = 0;
    uint32_t num_uniform_buffers = 0;

    friend bool operator==(const ComputeResources&, const ComputeResources&) = default;
};

/**
 * @brief GPU compute pipeline wrapper for pre-compiled SPIR-V shaders
 *
 * Compute shaders (`{name}.comp.hlsl`, entry point `CSMain`) are compiled to
 * SPIR-V at build time like the graphics stages. At runtime SDL_shadercross
 * converts the module to the backend format and builds the whole
 * SDL_GPUComputePipeline in one step (there is no SDL_GPUShader for compute).
 *
 * Creation REQUIRES SPIR-V reflection to succeed: the threadgroup dimensions
 * baked into the pipeline come from the module's [numthreads] declaration and
 * SDL has no way to take them from the caller. (Graphics shaders degrade to
 * caller-declared counts when reflection fails; compute cannot.)
 *
 * Binding order (SPIR-V sets / HLSL registers) — differs from graphics:
 * - set 0 / (t[n]+s[n], space0): sampled textures, then READ-ONLY storage
 *   textures, then READ-ONLY storage buffers
 * - set 1 / (u[n], space1): READ-WRITE storage textures, then READ-WRITE
 *   storage buffers (bound up front by SDL_BeginGPUComputePass — see
 *   ComputePass)
 * - set 2 / (b[n], space2): uniform buffers
 *
 * Usage:
 *   auto pipeline = GPUComputePipeline::load_spirv(device,
 *       "shaders/exposure_histogram.comp.spv", "CSMain", resources);
 *   ...
 *   auto pass = ComputePass::begin(cmd, {}, rw_buffers);
 *   pass.bind_pipeline(*pipeline);
 *   pass.push_uniforms(0, &uniforms, sizeof(uniforms));
 *   pass.dispatch_for(*pipeline, width, height);
 *   pass.end();
 */
class GPUComputePipeline {
public:
    ~GPUComputePipeline();

    // Non-copyable, movable
    GPUComputePipeline(const GPUComputePipeline&) = delete;
    GPUComputePipeline& operator=(const GPUComputePipeline&) = delete;
    GPUComputePipeline(GPUComputePipeline&& other) noexcept;
    GPUComputePipeline& operator=(GPUComputePipeline&& other) noexcept;

    /**
     * @brief Load a compute pipeline from a pre-compiled SPIR-V file
     *
     * @param device The GPU device
     * @param path Path to the compiled .comp.spv file
     * @param entry_point Entry point function name. Engine-built compute
     *                    SPIR-V always uses "CSMain".
     * @param resources Declared resource counts (validated against SPIR-V
     *                  reflection; see ComputeResources)
     * @return Unique pointer to the pipeline, or nullptr on failure
     */
    [[nodiscard]] static std::unique_ptr<GPUComputePipeline> load_spirv(GPUDevice& device, const std::string& path,
                                                                        const std::string& entry_point,
                                                                        const ComputeResources& resources = {});

    /**
     * @brief Create a compute pipeline from SPIR-V bytecode in memory
     */
    [[nodiscard]] static std::unique_ptr<GPUComputePipeline> create_from_spirv(GPUDevice& device,
                                                                               const std::vector<uint8_t>& spirv_bytecode,
                                                                               const std::string& entry_point,
                                                                               const ComputeResources& resources = {});

    /**
     * @brief Reflect the resource counts + threadgroup size a compute SPIR-V
     * module actually declares.
     *
     * Wraps SDL_ShaderCross_ReflectComputeSPIRV. Initializes the shader
     * cross-compilation system on demand. Pure CPU work — no GPU device is
     * required.
     *
     * @param spirv_bytecode The SPIR-V module
     * @param out Filled with the reflected counts on success
     * @param out_threadcount Optional; filled with the [numthreads] x/y/z
     * @return true on success, false if reflection failed (outputs untouched)
     */
    [[nodiscard]] static bool reflect_spirv_resources(const std::vector<uint8_t>& spirv_bytecode, ComputeResources& out,
                                                      uint32_t out_threadcount[3] = nullptr);

    // =========================================================================
    // Accessors
    // =========================================================================

    SDL_GPUComputePipeline* handle() const { return pipeline_; }

    /** Resource counts reflected from the SPIR-V (what SDL binds against). */
    const ComputeResources& resources() const { return resources_; }

    /** Threadgroup dimensions from the shader's [numthreads] declaration. */
    uint32_t threadcount_x() const { return threadcount_[0]; }
    uint32_t threadcount_y() const { return threadcount_[1]; }
    uint32_t threadcount_z() const { return threadcount_[2]; }

    /** Workgroups needed to cover `items` elements along each axis (ceil-div
     *  by the shader's threadgroup size). */
    uint32_t group_count_x(uint32_t items) const { return (items + threadcount_[0] - 1) / threadcount_[0]; }
    uint32_t group_count_y(uint32_t items) const { return (items + threadcount_[1] - 1) / threadcount_[1]; }
    uint32_t group_count_z(uint32_t items) const { return (items + threadcount_[2] - 1) / threadcount_[2]; }

private:
    GPUComputePipeline() = default;

    GPUDevice* device_ = nullptr;
    SDL_GPUComputePipeline* pipeline_ = nullptr;
    ComputeResources resources_{};
    uint32_t threadcount_[3] = {1, 1, 1};
};

/**
 * @brief RAII helper around one SDL_GPUComputePass
 *
 * SDL3 GPU binds READ-WRITE storage resources at pass BEGIN (they define the
 * pass's writable footprint); read-only resources (samplers, storage
 * textures/buffers) bind after the pipeline like a render pass. The
 * destructor ends a still-open pass, so early returns cannot leak one.
 *
 * The `cycle` flag of every read-write binding passed to begin() matters:
 * cycle=false preserves the buffer/texture contents (required for
 * accumulators and persistent state), cycle=true hands SDL a fresh backing
 * allocation when the resource is still referenced by in-flight frames.
 */
class ComputePass {
public:
    /**
     * @brief Begin a compute pass (no render/copy pass may be open on `cmd`)
     *
     * @param cmd Command buffer to record into
     * @param rw_textures Read-write storage-texture bindings (set 1 first)
     * @param rw_buffers Read-write storage-buffer bindings (set 1, after the
     *                   read-write textures)
     * @return Pass helper; falsy if SDL_BeginGPUComputePass failed
     */
    [[nodiscard]] static ComputePass begin(SDL_GPUCommandBuffer* cmd,
                                           std::span<const SDL_GPUStorageTextureReadWriteBinding> rw_textures,
                                           std::span<const SDL_GPUStorageBufferReadWriteBinding> rw_buffers);

    ComputePass() = default;
    ~ComputePass();

    // Non-copyable, movable
    ComputePass(const ComputePass&) = delete;
    ComputePass& operator=(const ComputePass&) = delete;
    ComputePass(ComputePass&& other) noexcept;
    ComputePass& operator=(ComputePass&& other) noexcept;

    explicit operator bool() const { return pass_ != nullptr; }
    SDL_GPUComputePass* handle() const { return pass_; }

    void bind_pipeline(const GPUComputePipeline& pipeline);

    /** Sampled textures (set 0, from `first_slot`). */
    void bind_samplers(uint32_t first_slot, std::span<const SDL_GPUTextureSamplerBinding> bindings);

    /** READ-ONLY storage textures (set 0, after the sampled textures). */
    void bind_storage_textures(uint32_t first_slot, std::span<SDL_GPUTexture* const> textures);

    /** READ-ONLY storage buffers (set 0, after the read-only storage textures). */
    void bind_storage_buffers(uint32_t first_slot, std::span<SDL_GPUBuffer* const> buffers);

    /** Push uniform data (set 2, b<slot>). Records onto the command buffer. */
    void push_uniforms(uint32_t slot, const void* data, uint32_t size);

    /** Dispatch an explicit workgroup grid. */
    void dispatch(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);

    /** Dispatch enough workgroups of `pipeline` to cover items_x × items_y ×
     *  items_z ELEMENTS (ceil-div by the shader's threadgroup size). */
    void dispatch_for(const GPUComputePipeline& pipeline, uint32_t items_x, uint32_t items_y = 1,
                      uint32_t items_z = 1);

    /**
     * @brief Dispatch with the workgroup counts read from a GPU buffer
     *
     * Wraps SDL_DispatchGPUComputeIndirect. What SDL requires of the caller,
     * read off SDL_gpu.h rather than assumed:
     *
     * - `args` must carry SDL_GPU_BUFFERUSAGE_INDIRECT — i.e. it must have
     *   been created as GPUBuffer::Type::Indirect.
     * - The 12 bytes at `offset` must be one SDL_GPUIndirectDispatchCommand:
     *   three consecutive Uint32 group counts (x, y, z), in that order. There
     *   is no count/stride parameter — this is exactly one dispatch. SDL states
     *   no alignment requirement beyond the struct's own (4 bytes), but the
     *   underlying APIs do: Vulkan's vkCmdDispatchIndirect and D3D12's
     *   ExecuteIndirect both require the argument offset to be a multiple of 4,
     *   so keep `offset` 4-byte aligned. Pack argument records at 16 bytes if
     *   you want them to also be addressable as std140 storage-buffer uints.
     * - A pipeline must already be bound (`bind_pipeline`) — SDL says so
     *   explicitly, and the counts are workgroups, so the shader's [numthreads]
     *   is what they multiply against.
     * - The counts are NOT synchronized against the pass that wrote them.
     *   SDL's rule for compute is that a dispatch depending on a previous
     *   dispatch's output needs a pass boundary; the producing pass must
     *   therefore have ENDED before the pass containing this call begins.
     *
     * NOT EXERCISED AT RUNTIME YET (R4a). Nothing in the engine dispatches
     * indirectly; the consumer is R4b's adaptive screen-probe placement, which
     * is the first pass whose workgroup count only the GPU knows. The mapping
     * this package can and does test CPU-side is Type::Indirect ->
     * SDL_GPU_BUFFERUSAGE_INDIRECT (gpu_resource_lifetime.test.cpp); the SDL
     * call itself is a one-line forward that no test can drive without a
     * device, and it is honest to say so here rather than imply coverage.
     *
     * @param args Buffer holding the dispatch command (Type::Indirect)
     * @param offset Byte offset of the command within `args` (4-byte aligned)
     */
    void dispatch_indirect(const GPUBuffer& args, uint32_t offset = 0);

    /** End the pass (idempotent; also called by the destructor). */
    void end();

private:
    SDL_GPUCommandBuffer* cmd_ = nullptr;
    SDL_GPUComputePass* pass_ = nullptr;
};

} // namespace bracken::gpu
