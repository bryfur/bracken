#pragma once

#include "gpu_device.hpp"
#include <cstddef>
#include <memory>

namespace bracken::gpu {

/**
 * @brief GPU Buffer abstraction for vertex, index, uniform, and storage buffers
 *
 * This class simplifies buffer creation and data upload by handling the transfer
 * buffer lifecycle automatically. It supports both static (upload once) and
 * dynamic (update frequently) buffer patterns.
 *
 * Usage:
 *   // Static buffer (geometry that doesn't change)
 *   auto vbo = GPUBuffer::create_static(device, GPUBuffer::Type::Vertex,
 *                                        vertices.data(), vertices.size() * sizeof(Vertex));
 *
 *   // Dynamic buffer (UI, particles, uniforms that change every frame)
 *   auto ubo = GPUBuffer::create_dynamic(device, GPUBuffer::Type::Uniform, sizeof(Uniforms));
 *   ubo->update(cmd, &uniforms, sizeof(uniforms));
 */
class GPUBuffer {
public:
    enum class Type {
        Vertex,  ///< Vertex buffer for geometry data
        Index,   ///< Index buffer for indexed drawing
        Uniform, ///< Uniform buffer for shader constants
        Storage, ///< Read-only storage buffer for graphics stages
        ComputeStorage, ///< Compute read-write storage (histogram/reduction
                        ///< targets); also readable as a graphics-stage
                        ///< read-only SSBO (COMPUTE_STORAGE_READ | WRITE |
                        ///< GRAPHICS_STORAGE_READ)
    };

    ~GPUBuffer();

    // Non-copyable, movable
    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;
    GPUBuffer(GPUBuffer&& other) noexcept;
    GPUBuffer& operator=(GPUBuffer&& other) noexcept;

    /**
     * @brief Create a static buffer with initial data
     *
     * Static buffers are optimized for data that is uploaded once and used many times.
     * The data is uploaded immediately via a transfer buffer.
     *
     * @param device The GPU device
     * @param type Buffer type (Vertex, Index, Uniform, Storage)
     * @param data Pointer to the initial data
     * @param size Size of the data in bytes
     * @return Unique pointer to the buffer, or nullptr on failure
     */
    [[nodiscard]] static std::unique_ptr<GPUBuffer> create_static(GPUDevice& device, Type type, const void* data, size_t size);

    /**
     * @brief Create a dynamic buffer for frequent updates
     *
     * Dynamic buffers maintain an internal transfer buffer for efficient updates.
     * Use update() to upload new data each frame.
     *
     * @param device The GPU device
     * @param type Buffer type
     * @param size Maximum size of the buffer in bytes
     * @return Unique pointer to the buffer, or nullptr on failure
     */
    [[nodiscard]] static std::unique_ptr<GPUBuffer> create_dynamic(GPUDevice& device, Type type, size_t size);

    /**
     * @brief Update buffer contents (for dynamic buffers)
     *
     * Begins and ends its own copy pass. Prefer the copy-pass overload when
     * updating several buffers in one frame so they share a single pass.
     *
     * @param cmd Command buffer for the copy operation
     * @param data Pointer to the new data
     * @param size Size of the data in bytes
     * @param offset Offset into the buffer (default 0)
     */
    void update(SDL_GPUCommandBuffer* cmd, const void* data, size_t size, size_t offset = 0);

    /**
     * @brief Update buffer contents within a caller-owned copy pass
     *
     * Records the upload into an already-open copy pass, letting multiple
     * buffer updates batch into one pass instead of each opening its own.
     *
     * @param copy_pass Open copy pass to record the upload into
     * @param data Pointer to the new data
     * @param size Size of the data in bytes
     * @param offset Offset into the buffer (default 0)
     */
    void update(SDL_GPUCopyPass* copy_pass, const void* data, size_t size, size_t offset = 0);

    /**
     * @brief RAII mapping of a dynamic buffer's persistent transfer buffer
     *
     * Lets callers write upload data directly into the mapped transfer memory
     * (no intermediate CPU staging vector), then record the GPU copy:
     *
     *   auto mapping = buffer->map();
     *   if (mapping) {
     *       std::memcpy(mapping.data(), src, size);
     *       mapping.upload(copy_pass, size);
     *   }
     *
     * upload() unmaps before recording the copy. If upload() is never called
     * (e.g. an early error exit), the destructor unmaps without recording a
     * copy. The mapping must not outlive the buffer.
     */
    class Mapping {
    public:
        Mapping() = default;
        ~Mapping();
        Mapping(Mapping&& other) noexcept;
        Mapping& operator=(Mapping&& other) noexcept;
        Mapping(const Mapping&) = delete;
        Mapping& operator=(const Mapping&) = delete;

        /// Mapped write pointer (nullptr if the map failed / already uploaded)
        void* data() const { return data_; }
        explicit operator bool() const { return data_ != nullptr; }

        /**
         * @brief Unmap and record the upload into an open copy pass
         * @param copy_pass Open copy pass to record the upload into
         * @param size Bytes to upload
         * @param offset Byte offset (applies to transfer and GPU buffer alike)
         */
        void upload(SDL_GPUCopyPass* copy_pass, size_t size, size_t offset = 0);

    private:
        friend class GPUBuffer;
        Mapping(GPUBuffer* buffer, void* data) : buffer_(buffer), data_(data) {}
        void release(); // unmap without uploading
        GPUBuffer* buffer_ = nullptr;
        void* data_ = nullptr;
    };

    /**
     * @brief Map the transfer buffer for direct writes (dynamic buffers only)
     *
     * Maps with cycle=true — the same double-buffering semantics as update() —
     * so a full per-frame rewrite is safe. Returns an empty Mapping on failure.
     */
    [[nodiscard]] Mapping map();

    /**
     * @brief Get the raw SDL GPU buffer handle
     */
    SDL_GPUBuffer* handle() const { return buffer_; }

    /**
     * @brief Get the buffer size in bytes
     */
    size_t size() const { return size_; }

    /**
     * @brief Get the buffer type
     */
    Type type() const { return type_; }

    /**
     * @brief Check if this is a dynamic buffer
     */
    bool is_dynamic() const { return transfer_buffer_ != nullptr; }

private:
    GPUBuffer() = default;

    GPUDevice* device_ = nullptr;
    SDL_GPUBuffer* buffer_ = nullptr;
    SDL_GPUTransferBuffer* transfer_buffer_ = nullptr; // For dynamic buffers
    size_t size_ = 0;
    Type type_ = Type::Vertex;

    static SDL_GPUBufferUsageFlags get_usage_flags(Type type);
};

} // namespace bracken::gpu
