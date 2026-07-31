#pragma once

#include "gpu_device.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace bracken::gpu {

/**
 * @brief Shader stage types
 *
 * Compute exists for API completeness (build naming, cache keys) but does NOT
 * go through GPUShader: SDL3 GPU has no SDL_GPUShader for compute — a compute
 * SPIR-V module compiles directly into an SDL_GPUComputePipeline. Use
 * gpu::GPUComputePipeline (gpu_compute.hpp) instead; GPUShader creation
 * rejects ShaderStage::Compute.
 */
enum class ShaderStage {
    Vertex,
    Fragment,
    Compute,
};

/**
 * @brief Shader resource requirements for pipeline creation
 *
 * At shader creation these declared counts are cross-checked against SPIR-V
 * reflection (SDL_ShaderCross_ReflectGraphicsSPIRV). On mismatch an error is
 * logged and the reflected counts win, so a stale hand-maintained count can
 * no longer silently misbind resources. Counts are still not validated
 * against hardware limits; a shader exceeding device capabilities fails at
 * shader/pipeline creation with an SDL error.
 */
struct ShaderResources {
    uint32_t num_samplers = 0;
    uint32_t num_storage_textures = 0;
    uint32_t num_storage_buffers = 0;
    uint32_t num_uniform_buffers = 0;

    friend bool operator==(const ShaderResources&, const ShaderResources&) = default;
};

/**
 * @brief GPU Shader wrapper for pre-compiled SPIRV shaders
 *
 * Shaders are compiled from HLSL to SPIRV at build time by CMake.
 * At runtime, SDL_shadercross converts SPIRV to the backend format
 * (Vulkan uses SPIRV directly, Metal/D3D12 get transpiled).
 *
 * Usage:
 *   auto vs = GPUShader::load_spirv(device, "shaders/model.vert.spv",
 *                                    ShaderStage::Vertex, "VSMain", resources);
 */
class GPUShader {
public:
    ~GPUShader();

    // Non-copyable, movable
    GPUShader(const GPUShader&) = delete;
    GPUShader& operator=(const GPUShader&) = delete;
    GPUShader(GPUShader&& other) noexcept;
    GPUShader& operator=(GPUShader&& other) noexcept;

    /**
     * @brief Load a shader from pre-compiled SPIRV file
     *
     * Uses SDL_shadercross to convert SPIRV to backend format at runtime.
     *
     * @param device The GPU device
     * @param path Path to the compiled .spv file
     * @param stage Shader stage (Vertex or Fragment)
     * @param entry_point Entry point function name. Engine-built SPIR-V always
     *                    uses "VSMain" (vertex) / "PSMain" (fragment); there is
     *                    deliberately no default because a wrong entry point
     *                    fails late and confusingly on Vulkan.
     * @param resources Shader resource requirements (validated against SPIR-V
     *                  reflection; see ShaderResources)
     * @return Unique pointer to the shader, or nullptr on failure
     */
    [[nodiscard]] static std::unique_ptr<GPUShader> load_spirv(GPUDevice& device, const std::string& path, ShaderStage stage,
                                                 const std::string& entry_point,
                                                 const ShaderResources& resources = {});

    /**
     * @brief Create a shader from SPIRV bytecode in memory
     */
    [[nodiscard]] static std::unique_ptr<GPUShader> create_from_spirv(GPUDevice& device, const std::vector<uint8_t>& spirv_bytecode,
                                                        ShaderStage stage, const std::string& entry_point,
                                                        const ShaderResources& resources = {});

    /**
     * @brief Reflect the resource counts a SPIR-V module actually declares.
     *
     * Wraps SDL_ShaderCross_ReflectGraphicsSPIRV. Initializes the shader
     * cross-compilation system on demand. Pure CPU work — no GPU device is
     * required.
     *
     * @param spirv_bytecode The SPIR-V module
     * @param out Filled with the reflected counts on success
     * @return true on success, false if reflection failed (out untouched)
     */
    [[nodiscard]] static bool reflect_spirv_resources(const std::vector<uint8_t>& spirv_bytecode, ShaderResources& out);

    // =========================================================================
    // Accessors
    // =========================================================================

    SDL_GPUShader* handle() const { return shader_; }
    ShaderStage stage() const { return stage_; }

    /**
     * @brief Resource counts this shader was actually created with
     *
     * These are the counts baked into the SDL_GPUShader (reflected counts when
     * reflection succeeded, caller-declared counts otherwise).
     */
    const ShaderResources& resources() const { return resources_; }

    // =========================================================================
    // Global Configuration
    // =========================================================================

    /**
     * @brief Initialize the shader cross-compilation system
     * Must be called before any shader loading.
     */
    [[nodiscard]] static bool init_compiler();

    /**
     * @brief Shutdown the shader cross-compilation system
     */
    static void shutdown_compiler();

    /**
     * @brief Check if the compiler is available
     */
    static bool is_compiler_available();

private:
    GPUShader() = default;

    GPUDevice* device_ = nullptr;
    SDL_GPUShader* shader_ = nullptr;
    ShaderStage stage_ = ShaderStage::Vertex;
    ShaderResources resources_{};

    static std::vector<uint8_t> read_file(const std::string& path);

    static bool s_compiler_initialized;
};

/**
 * @brief High-level shader manager with in-memory caching
 *
 * This is the recommended way to load shaders. It provides:
 * - Automatic SPIRV loading with backend conversion
 * - In-memory caching (avoid reloading within session)
 *
 * Thread-safety tiers (see docs/multithreading_plan.md §6.1):
 * - **main-thread (asserted)**: every cache-mutating entry point — get()/
 *   get_vertex()/get_fragment() on a cache MISS (first load of a shader),
 *   clear_cache(), reload(), reload_all(), and destruction.
 * - **worker-safe after init**: get() on an already-cached shader is a
 *   read-only lookup, safe from any thread once all shaders are loaded
 *   (pipeline preload does this) and no mutation is in flight.
 *
 * Usage:
 *   ShaderManager shaders(device);
 *
 *   auto* vs = shaders.get_vertex("shaders/model.vert.spv", "VSMain");
 *   auto* fs = shaders.get_fragment("shaders/model.frag.spv", "PSMain");
 */
class ShaderManager {
public:
    /**
     * @brief Create a shader manager
     * @param device The GPU device
     */
    explicit ShaderManager(GPUDevice& device);
    ~ShaderManager();

    // Non-copyable
    ShaderManager(const ShaderManager&) = delete;
    ShaderManager& operator=(const ShaderManager&) = delete;
    ShaderManager(ShaderManager&&) = delete;
    ShaderManager& operator=(ShaderManager&&) = delete;

    /**
     * @brief Load or get a cached vertex shader
     *
     * Default entry point matches the engine's build-time convention (VSMain).
     */
    GPUShader* get_vertex(const std::string& path, const std::string& entry_point = "VSMain",
                          const ShaderResources& resources = {});

    /**
     * @brief Load or get a cached fragment shader
     *
     * Default entry point matches the engine's build-time convention (PSMain).
     */
    GPUShader* get_fragment(const std::string& path, const std::string& entry_point = "PSMain",
                            const ShaderResources& resources = {});

    /**
     * @brief Load or get a shader (generic)
     *
     * The cache key is (path, stage, entry_point). On a cache hit the
     * requested resource counts are compared against the counts the cached
     * shader was created with; a mismatch logs an error (the cached shader is
     * returned unchanged — resource counts are baked in at first load).
     *
     * A cache hit is a read-only lookup (worker-safe after init); a cache
     * miss loads and inserts, which is main-thread tier — asserted.
     */
    GPUShader* get(const std::string& path, ShaderStage stage, const std::string& entry_point,
                   const ShaderResources& resources = {});

    /**
     * @brief Clear all cached shaders. Main-thread (asserted).
     */
    void clear_cache();

    /**
     * @brief Reload a shader (clear from cache, reloads on next get).
     * Main-thread (asserted).
     * @return true if shader was in cache
     */
    [[nodiscard]] bool reload(const std::string& path);

    /**
     * @brief Reload all shaders. Main-thread (asserted).
     * @return Number of shaders cleared
     */
    int reload_all();

private:
    GPUDevice& device_;

    // Thread-safety: mutation is main-thread tier and asserted at every
    // mutating entry point (first-load insert, clear_cache, reload*);
    // lookups of already-cached shaders are const-safe after init. No lock —
    // the contract is enforced, not guarded.
    std::unordered_map<std::string, std::unique_ptr<GPUShader>> cache_;

    static std::string make_cache_key(const std::string& path, ShaderStage stage, const std::string& entry_point);
};

} // namespace bracken::gpu
