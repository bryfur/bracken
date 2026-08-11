#include "gpu_compute.hpp"
#include "bracken/core/assert.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/gpu/gpu_shader.hpp"
#include "SDL3/SDL_error.h"
#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <SDL3/SDL_stdinc.h>
#include <SDL3_shadercross/SDL_shadercross.h>
#include <string>
#include <utility>
#include <vector>

namespace bracken::gpu {

namespace {
constexpr const char* LOG_TAG = "gpu";
} // namespace

// =============================================================================
// GPUComputePipeline
// =============================================================================

GPUComputePipeline::~GPUComputePipeline() {
    if (device_ && pipeline_) {
        device_->release_compute_pipeline(pipeline_);
    }
}

GPUComputePipeline::GPUComputePipeline(GPUComputePipeline&& other) noexcept
    : device_(other.device_),
      pipeline_(other.pipeline_),
      resources_(other.resources_),
      threadcount_{other.threadcount_[0], other.threadcount_[1], other.threadcount_[2]} {
    other.device_ = nullptr;
    other.pipeline_ = nullptr;
}

GPUComputePipeline& GPUComputePipeline::operator=(GPUComputePipeline&& other) noexcept {
    if (this != &other) {
        if (device_ && pipeline_) {
            device_->release_compute_pipeline(pipeline_);
        }

        device_ = other.device_;
        pipeline_ = other.pipeline_;
        resources_ = other.resources_;
        threadcount_[0] = other.threadcount_[0];
        threadcount_[1] = other.threadcount_[1];
        threadcount_[2] = other.threadcount_[2];

        other.device_ = nullptr;
        other.pipeline_ = nullptr;
    }
    return *this;
}

bool GPUComputePipeline::reflect_spirv_resources(const std::vector<uint8_t>& spirv_bytecode, ComputeResources& out,
                                                 uint32_t out_threadcount[3]) {
    if (!GPUShader::is_compiler_available() && !GPUShader::init_compiler()) {
        return false;
    }
    if (spirv_bytecode.empty()) {
        return false;
    }

    SDL_ShaderCross_ComputePipelineMetadata* meta =
        SDL_ShaderCross_ReflectComputeSPIRV(spirv_bytecode.data(), spirv_bytecode.size(), 0);
    if (!meta) {
        return false;
    }

    out.num_samplers = meta->num_samplers;
    out.num_readonly_storage_textures = meta->num_readonly_storage_textures;
    out.num_readonly_storage_buffers = meta->num_readonly_storage_buffers;
    out.num_readwrite_storage_textures = meta->num_readwrite_storage_textures;
    out.num_readwrite_storage_buffers = meta->num_readwrite_storage_buffers;
    out.num_uniform_buffers = meta->num_uniform_buffers;
    if (out_threadcount != nullptr) {
        out_threadcount[0] = meta->threadcount_x;
        out_threadcount[1] = meta->threadcount_y;
        out_threadcount[2] = meta->threadcount_z;
    }
    SDL_free(meta);
    return true;
}

bool GPUComputePipeline::reflect_spirv_file(const std::string& path, ComputeResources& out) {
    // Registration-time only (host pipelines); the extra read of a file
    // load_spirv is about to read again is not worth avoiding.
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUComputePipeline::reflect_spirv_file: Failed to open '{}'", path.c_str());
        return false;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUComputePipeline::reflect_spirv_file: Empty file '{}'", path.c_str());
        return false;
    }
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytecode(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(bytecode.data()), size)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUComputePipeline::reflect_spirv_file: Failed to read '{}'", path.c_str());
        return false;
    }
    if (!reflect_spirv_resources(bytecode, out)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUComputePipeline::reflect_spirv_file: reflection failed for '{}': {}",
                          path.c_str(), SDL_GetError());
        return false;
    }
    return true;
}

std::unique_ptr<GPUComputePipeline> GPUComputePipeline::load_spirv(GPUDevice& device, const std::string& path,
                                                                   const std::string& entry_point,
                                                                   const ComputeResources& resources) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUComputePipeline::load_spirv: Failed to open '{}'", path.c_str());
        return nullptr;
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUComputePipeline::load_spirv: Empty file '{}'", path.c_str());
        return nullptr;
    }
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytecode(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(bytecode.data()), size)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUComputePipeline::load_spirv: Failed to read '{}'", path.c_str());
        return nullptr;
    }

    auto pipeline = create_from_spirv(device, bytecode, entry_point, resources);
    if (pipeline) {
        BRACKEN_LOG_DEBUG(LOG_TAG, "GPUComputePipeline::load_spirv: Loaded '{}' successfully", path.c_str());
    }
    return pipeline;
}

std::unique_ptr<GPUComputePipeline> GPUComputePipeline::create_from_spirv(GPUDevice& device,
                                                                          const std::vector<uint8_t>& spirv_bytecode,
                                                                          const std::string& entry_point,
                                                                          const ComputeResources& resources) {
    // GPU pipeline creation is main-thread tier (compiles straight into an
    // SDL_GPUComputePipeline). Destruction asserts inside
    // GPUDevice::release_compute_pipeline.
    BRACKEN_ASSERT_MAIN_THREAD();
    if (!GPUShader::is_compiler_available() && !GPUShader::init_compiler()) {
        return nullptr;
    }
    if (spirv_bytecode.empty()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUComputePipeline::create_from_spirv: Empty bytecode");
        return nullptr;
    }

    auto pipeline = std::unique_ptr<GPUComputePipeline>(new GPUComputePipeline());
    pipeline->device_ = &device;

    // Reflection is REQUIRED for compute: the threadgroup dimensions baked
    // into the pipeline come from the module's [numthreads] and there is no
    // caller-side fallback for them. The declared counts are cross-checked
    // exactly like the graphics stages — a mismatch is loud and the reflected
    // counts win, since they are by definition what SDL needs to bind.
    ComputeResources reflected;
    uint32_t threadcount[3] = {1, 1, 1};
    if (!reflect_spirv_resources(spirv_bytecode, reflected, threadcount)) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "GPUComputePipeline::create_from_spirv: SPIR-V reflection failed ({}); compute pipelines "
                          "cannot be created without reflection (threadgroup size is only known to the module)",
                          SDL_GetError());
        return nullptr;
    }
    if (reflected != resources) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "GPUComputePipeline::create_from_spirv: declared resource counts do not match SPIR-V "
                          "(entry '{}'): samplers {}/{}, ro_storage_textures {}/{}, ro_storage_buffers {}/{}, "
                          "rw_storage_textures {}/{}, rw_storage_buffers {}/{}, uniform_buffers {}/{} "
                          "(declared/reflected). Using reflected counts; fix the caller.",
                          entry_point.c_str(), resources.num_samplers, reflected.num_samplers,
                          resources.num_readonly_storage_textures, reflected.num_readonly_storage_textures,
                          resources.num_readonly_storage_buffers, reflected.num_readonly_storage_buffers,
                          resources.num_readwrite_storage_textures, reflected.num_readwrite_storage_textures,
                          resources.num_readwrite_storage_buffers, reflected.num_readwrite_storage_buffers,
                          resources.num_uniform_buffers, reflected.num_uniform_buffers);
    }
    pipeline->resources_ = reflected;
    pipeline->threadcount_[0] = threadcount[0];
    pipeline->threadcount_[1] = threadcount[1];
    pipeline->threadcount_[2] = threadcount[2];

    SDL_ShaderCross_ComputePipelineMetadata metadata = {};
    metadata.num_samplers = reflected.num_samplers;
    metadata.num_readonly_storage_textures = reflected.num_readonly_storage_textures;
    metadata.num_readonly_storage_buffers = reflected.num_readonly_storage_buffers;
    metadata.num_readwrite_storage_textures = reflected.num_readwrite_storage_textures;
    metadata.num_readwrite_storage_buffers = reflected.num_readwrite_storage_buffers;
    metadata.num_uniform_buffers = reflected.num_uniform_buffers;
    metadata.threadcount_x = threadcount[0];
    metadata.threadcount_y = threadcount[1];
    metadata.threadcount_z = threadcount[2];

    SDL_ShaderCross_SPIRV_Info spirv_info = {};
    spirv_info.bytecode = spirv_bytecode.data();
    spirv_info.bytecode_size = spirv_bytecode.size();
    spirv_info.entrypoint = entry_point.c_str();
    spirv_info.shader_stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE;
    // Same MSL floor as GPUShader::create_from_spirv — SPIRV-Cross defaults
    // to MSL 1.2, which predates features the engine's shaders rely on.
    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetStringProperty(props, SDL_SHADERCROSS_PROP_SPIRV_MSL_VERSION_STRING, "2.2.0");
    spirv_info.props = props;

    pipeline->pipeline_ =
        SDL_ShaderCross_CompileComputePipelineFromSPIRV(device.handle(), &spirv_info, &metadata, 0);
    SDL_DestroyProperties(props);
    if (!pipeline->pipeline_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GPUComputePipeline::create_from_spirv: SPIRV->GPU failed: {}", SDL_GetError());
        return nullptr;
    }

    return pipeline;
}

// =============================================================================
// ComputePass
// =============================================================================

ComputePass ComputePass::begin(SDL_GPUCommandBuffer* cmd,
                               std::span<const SDL_GPUStorageTextureReadWriteBinding> rw_textures,
                               std::span<const SDL_GPUStorageBufferReadWriteBinding> rw_buffers) {
    ComputePass pass;
    if (!cmd) {
        return pass;
    }
    pass.cmd_ = cmd;
    pass.pass_ = SDL_BeginGPUComputePass(cmd, rw_textures.data(), static_cast<Uint32>(rw_textures.size()),
                                         rw_buffers.data(), static_cast<Uint32>(rw_buffers.size()));
    if (!pass.pass_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "ComputePass::begin: SDL_BeginGPUComputePass failed: {}", SDL_GetError());
    }
    return pass;
}

ComputePass::~ComputePass() {
    end();
}

ComputePass::ComputePass(ComputePass&& other) noexcept : cmd_(other.cmd_), pass_(other.pass_) {
    other.cmd_ = nullptr;
    other.pass_ = nullptr;
}

ComputePass& ComputePass::operator=(ComputePass&& other) noexcept {
    if (this != &other) {
        end();
        cmd_ = other.cmd_;
        pass_ = other.pass_;
        other.cmd_ = nullptr;
        other.pass_ = nullptr;
    }
    return *this;
}

void ComputePass::bind_pipeline(const GPUComputePipeline& pipeline) {
    if (pass_ && pipeline.handle()) {
        SDL_BindGPUComputePipeline(pass_, pipeline.handle());
    }
}

void ComputePass::bind_samplers(uint32_t first_slot, std::span<const SDL_GPUTextureSamplerBinding> bindings) {
    if (pass_ && !bindings.empty()) {
        SDL_BindGPUComputeSamplers(pass_, first_slot, bindings.data(), static_cast<Uint32>(bindings.size()));
    }
}

void ComputePass::bind_storage_textures(uint32_t first_slot, std::span<SDL_GPUTexture* const> textures) {
    if (pass_ && !textures.empty()) {
        SDL_BindGPUComputeStorageTextures(pass_, first_slot, textures.data(), static_cast<Uint32>(textures.size()));
    }
}

void ComputePass::bind_storage_buffers(uint32_t first_slot, std::span<SDL_GPUBuffer* const> buffers) {
    if (pass_ && !buffers.empty()) {
        SDL_BindGPUComputeStorageBuffers(pass_, first_slot, buffers.data(), static_cast<Uint32>(buffers.size()));
    }
}

void ComputePass::push_uniforms(uint32_t slot, const void* data, uint32_t size) {
    if (cmd_ && data && size > 0) {
        SDL_PushGPUComputeUniformData(cmd_, slot, data, size);
    }
}

void ComputePass::dispatch(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    if (pass_) {
        SDL_DispatchGPUCompute(pass_, groups_x, groups_y, groups_z);
    }
}

void ComputePass::dispatch_for(const GPUComputePipeline& pipeline, uint32_t items_x, uint32_t items_y,
                               uint32_t items_z) {
    dispatch(pipeline.group_count_x(items_x), pipeline.group_count_y(items_y), pipeline.group_count_z(items_z));
}

void ComputePass::dispatch_indirect(const GPUBuffer& args, uint32_t offset) {
    if (!pass_) {
        return;
    }
    if (args.type() != GPUBuffer::Type::Indirect) {
        // SDL would fail its own validation here, but only in a debug device
        // build and only as a driver-flavoured message. Say which buffer type
        // was needed instead.
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "ComputePass::dispatch_indirect: buffer is not GPUBuffer::Type::Indirect "
                          "(SDL_GPU_BUFFERUSAGE_INDIRECT is required); dispatch skipped");
        return;
    }
    if (args.handle() == nullptr) {
        return;
    }
    SDL_DispatchGPUComputeIndirect(pass_, args.handle(), offset);
}

void ComputePass::end() {
    if (pass_) {
        SDL_EndGPUComputePass(pass_);
        pass_ = nullptr;
    }
    cmd_ = nullptr;
}

} // namespace bracken::gpu
