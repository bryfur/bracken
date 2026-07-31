#include "gpu_shader.hpp"
#include "bracken/core/assert.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/pipeline_golden_capture.hpp" // TEMPORARY: phase 6c commit 1
#include "SDL3/SDL_error.h"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <SDL3/SDL_stdinc.h>
#include <SDL3_shadercross/SDL_shadercross.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace bracken::gpu {

namespace {
constexpr const char* LOG_TAG = "gpu";
} // namespace

// Static member initialization
bool GPUShader::s_compiler_initialized = false;

// =============================================================================
// GPUShader Implementation
// =============================================================================

GPUShader::~GPUShader() {
    if (device_ && shader_) {
        device_->release_shader(shader_);
    }
}

GPUShader::GPUShader(GPUShader&& other) noexcept
    : device_(other.device_),
      shader_(other.shader_),
      stage_(other.stage_),
      resources_(other.resources_) {
    other.device_ = nullptr;
    other.shader_ = nullptr;
}

GPUShader& GPUShader::operator=(GPUShader&& other) noexcept {
    if (this != &other) {
        if (device_ && shader_) {
            device_->release_shader(shader_);
        }

        device_ = other.device_;
        shader_ = other.shader_;
        stage_ = other.stage_;
        resources_ = other.resources_;

        other.device_ = nullptr;
        other.shader_ = nullptr;
    }
    return *this;
}

bool GPUShader::init_compiler() {
    if (s_compiler_initialized) {
        return true;
    }

    if (!SDL_ShaderCross_Init()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUShader::init_compiler: Failed to initialize SDL_shadercross: {}",
                          SDL_GetError());
        return false;
    }

    s_compiler_initialized = true;
    BRACKEN_LOG_INFO(LOG_TAG, "GPUShader::init_compiler: SDL_shadercross initialized");
    return true;
}

void GPUShader::shutdown_compiler() {
    if (s_compiler_initialized) {
        SDL_ShaderCross_Quit();
        s_compiler_initialized = false;
        BRACKEN_LOG_INFO(LOG_TAG, "GPUShader::shutdown_compiler: SDL_shadercross shutdown");
    }
}

bool GPUShader::is_compiler_available() {
    return s_compiler_initialized;
}

std::vector<uint8_t> GPUShader::read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUShader::read_file: Failed to open '{}'", path.c_str());
        return {};
    }

    std::streamsize size = file.tellg();
    if (size <= 0) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUShader::read_file: Empty file '{}'", path.c_str());
        return {};
    }

    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(static_cast<size_t>(size));

    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUShader::read_file: Failed to read '{}'", path.c_str());
        return {};
    }

    return buffer;
}

std::unique_ptr<GPUShader> GPUShader::load_spirv(GPUDevice& device, const std::string& path, ShaderStage stage,
                                                 const std::string& entry_point, const ShaderResources& resources) {

    std::vector<uint8_t> bytecode = read_file(path);
    if (bytecode.empty()) {
        return nullptr;
    }

    auto shader = create_from_spirv(device, bytecode, stage, entry_point, resources);
    if (shader) {
        BRACKEN_LOG_DEBUG(LOG_TAG, "GPUShader::load_spirv: Loaded '{}' successfully", path.c_str());
    }
    return shader;
}

bool GPUShader::reflect_spirv_resources(const std::vector<uint8_t>& spirv_bytecode, ShaderResources& out) {
    if (!s_compiler_initialized && !init_compiler()) {
        return false;
    }
    if (spirv_bytecode.empty()) {
        return false;
    }

    SDL_ShaderCross_GraphicsShaderMetadata* meta =
        SDL_ShaderCross_ReflectGraphicsSPIRV(spirv_bytecode.data(), spirv_bytecode.size(), 0);
    if (!meta) {
        return false;
    }

    out.num_samplers = meta->resource_info.num_samplers;
    out.num_storage_textures = meta->resource_info.num_storage_textures;
    out.num_storage_buffers = meta->resource_info.num_storage_buffers;
    out.num_uniform_buffers = meta->resource_info.num_uniform_buffers;
    SDL_free(meta);
    return true;
}

std::unique_ptr<GPUShader> GPUShader::create_from_spirv(GPUDevice& device, const std::vector<uint8_t>& spirv_bytecode,
                                                        ShaderStage stage, const std::string& entry_point,
                                                        const ShaderResources& resources) {

    if (!s_compiler_initialized) {
        if (!init_compiler()) {
            return nullptr;
        }
    }

    // GPU resource creation is main-thread tier (SDL_CreateGPUShader under
    // the hood). Single-threaded test binaries bind "main" on first use.
    BRACKEN_ASSERT_MAIN_THREAD();

    if (spirv_bytecode.empty()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUShader::create_from_spirv: Empty bytecode");
        return nullptr;
    }

    if (stage == ShaderStage::Compute) {
        // SDL3 GPU compute shaders are whole pipelines, not SDL_GPUShaders.
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUShader::create_from_spirv: compute shaders do not produce SDL_GPUShaders; "
                                   "use GPUComputePipeline (gpu_compute.hpp)");
        return nullptr;
    }

    auto shader = std::unique_ptr<GPUShader>(new GPUShader());
    shader->device_ = &device;
    shader->stage_ = stage;

    // Cross-check the caller-declared counts against what the SPIR-V actually
    // declares. Stale hand-maintained counts are the documented most-common way
    // to silently break rendering, so a mismatch is loud — and the reflected
    // counts win, since they are by definition what SDL needs to bind.
    ShaderResources effective = resources;
    ShaderResources reflected;
    if (reflect_spirv_resources(spirv_bytecode, reflected)) {
        if (reflected != resources) {
            BRACKEN_LOG_ERROR(LOG_TAG,
                              "GPUShader::create_from_spirv: declared resource counts do not match SPIR-V "
                              "(entry '{}'): samplers {}/{}, storage_textures {}/{}, storage_buffers {}/{}, "
                              "uniform_buffers {}/{} (declared/reflected). Using reflected counts; fix the caller.",
                              entry_point.c_str(), resources.num_samplers, reflected.num_samplers,
                              resources.num_storage_textures, reflected.num_storage_textures,
                              resources.num_storage_buffers, reflected.num_storage_buffers,
                              resources.num_uniform_buffers, reflected.num_uniform_buffers);
            effective = reflected;
        }
    } else {
        BRACKEN_LOG_WARN(LOG_TAG,
                         "GPUShader::create_from_spirv: SPIR-V reflection failed ({}); using declared resource counts",
                         SDL_GetError());
    }
    shader->resources_ = effective;

    // Set up resource info
    SDL_ShaderCross_GraphicsShaderResourceInfo resource_info = {};
    resource_info.num_samplers = effective.num_samplers;
    resource_info.num_storage_textures = effective.num_storage_textures;
    resource_info.num_storage_buffers = effective.num_storage_buffers;
    resource_info.num_uniform_buffers = effective.num_uniform_buffers;

    // Set up SPIRV info
    SDL_ShaderCross_SPIRV_Info spirv_info = {};
    spirv_info.bytecode = spirv_bytecode.data();
    spirv_info.bytecode_size = spirv_bytecode.size();
    spirv_info.entrypoint = entry_point.c_str();
    spirv_info.shader_stage =
        (stage == ShaderStage::Vertex) ? SDL_SHADERCROSS_SHADERSTAGE_VERTEX : SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT;
    // SPIRV-Cross emits MSL 1.2 by default, but CalculateLevelOfDetail*
    // (OpImageQueryLod) in the model/masked-prepass shaders needs MSL 2.2
    // (macOS 10.15+). Ignored by the non-Metal backends.
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetStringProperty(props, SDL_SHADERCROSS_PROP_SPIRV_MSL_VERSION_STRING, "2.2.0");
    spirv_info.props = props;

    // Compile SPIRV to backend shader (Vulkan uses SPIRV directly,
    // Metal/D3D12 get transpiled via SPIRV-Cross)
    shader->shader_ = SDL_ShaderCross_CompileGraphicsShaderFromSPIRV(device.handle(), &spirv_info, &resource_info, 0);
    SDL_DestroyProperties(props);

    if (!shader->shader_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUShader::create_from_spirv: SPIRV->GPU failed: {}", SDL_GetError());
        return nullptr;
    }

    return shader;
}

// =============================================================================
// ShaderManager Implementation
// =============================================================================

ShaderManager::ShaderManager(GPUDevice& device) : device_(device) {
    // Initialize the cross-compilation system. Failure is logged inside and
    // every subsequent shader load fails (and logs) individually.
    (void)GPUShader::init_compiler();
}

ShaderManager::~ShaderManager() {
    clear_cache();
}

std::string ShaderManager::make_cache_key(const std::string& path, ShaderStage stage, const std::string& entry_point) {
    // Use length-prefixed fields to avoid collisions
    std::ostringstream oss;
    oss << path.size() << ':' << path << '|' << static_cast<int>(stage) << '|' << entry_point.size() << ':'
        << entry_point;
    return oss.str();
}

GPUShader* ShaderManager::get_vertex(const std::string& path, const std::string& entry_point,
                                     const ShaderResources& resources) {
    return get(path, ShaderStage::Vertex, entry_point, resources);
}

GPUShader* ShaderManager::get_fragment(const std::string& path, const std::string& entry_point,
                                       const ShaderResources& resources) {
    return get(path, ShaderStage::Fragment, entry_point, resources);
}

GPUShader* ShaderManager::get(const std::string& path, ShaderStage stage, const std::string& entry_point,
                              const ShaderResources& resources) {
    // TEMPORARY (refactor phase 6c commit 1): golden capture — records which
    // shader each hand-written creator asks for, cache hits included.
    if (golden::g_shader_hook) {
        golden::g_shader_hook(path, stage, resources);
    }
    std::string key = make_cache_key(path, stage, entry_point);

    // Check cache
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // Resource counts are baked into the SDL_GPUShader at first load; a
        // second caller requesting different counts would otherwise silently
        // get the first-baked shader (load-order dependent misbinding).
        const ShaderResources& cached = it->second->resources();
        if (!(cached == resources)) {
            BRACKEN_LOG_ERROR(
                LOG_TAG,
                "ShaderManager::get: '{}' requested with resource counts differing from the cached "
                "shader: samplers {}/{}, storage_textures {}/{}, storage_buffers {}/{}, uniform_buffers {}/{} "
                "(requested/cached). Returning the cached shader; reconcile the call sites.",
                path.c_str(), resources.num_samplers, cached.num_samplers, resources.num_storage_textures,
                cached.num_storage_textures, resources.num_storage_buffers, cached.num_storage_buffers,
                resources.num_uniform_buffers, cached.num_uniform_buffers);
        }
        return it->second.get();
    }

    // Cache miss: load + insert mutate the cache — main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    auto shader = GPUShader::load_spirv(device_, path, stage, entry_point, resources);
    if (!shader) {
        BRACKEN_LOG_ERROR(LOG_TAG, "ShaderManager: Failed to load '{}'", path.c_str());
        return nullptr;
    }

    // Cache and return
    GPUShader* result = shader.get();
    cache_[key] = std::move(shader);
    return result;
}

void ShaderManager::clear_cache() {
    // Cache mutation + GPU shader release: main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    cache_.clear();
}

bool ShaderManager::reload(const std::string& path) {
    BRACKEN_ASSERT_MAIN_THREAD();
    // Find and remove all cache entries for this path
    auto matches_path = [&path](const std::string& key) {
        size_t colon_pos = key.find(':');
        if (colon_pos == std::string::npos) {
            return false;
        }
        size_t path_len = std::stoull(key.substr(0, colon_pos));
        std::string key_path = key.substr(colon_pos + 1, path_len);
        return key_path == path;
    };

    std::vector<std::string> keys_to_remove;
    for (const auto& [key, _] : cache_) {
        if (matches_path(key)) {
            keys_to_remove.push_back(key);
        }
    }

    for (const auto& key : keys_to_remove) {
        cache_.erase(key);
    }

    BRACKEN_LOG_INFO(LOG_TAG, "ShaderManager: Marked '{}' for reload ({} variants)", path.c_str(),
                     keys_to_remove.size());
    return !keys_to_remove.empty();
}

int ShaderManager::reload_all() {
    int count = static_cast<int>(cache_.size());
    clear_cache();
    BRACKEN_LOG_INFO(LOG_TAG, "ShaderManager: Cleared {} shaders for reload", count);
    return count;
}

} // namespace bracken::gpu
