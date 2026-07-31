// TEMPORARY — refactor phase 6c, commit 1 of 4. See the header.

#include "bracken/gpu/pipeline_golden_capture.hpp"

#include "bracken/gpu/gpu_types.hpp"
#include <format>

namespace bracken::gpu::golden {

void (*g_config_hook)(const PipelineConfig&) = nullptr;
void (*g_shader_hook)(const std::string&, ShaderStage, const ShaderResources&) = nullptr;
void (*g_compute_hook)(const std::string&, const std::string&, const ComputeResources&) = nullptr;

namespace {

const char* primitive_name(SDL_GPUPrimitiveType v) {
    switch (v) {
        case SDL_GPU_PRIMITIVETYPE_TRIANGLELIST: return "TRIANGLELIST";
        case SDL_GPU_PRIMITIVETYPE_TRIANGLESTRIP: return "TRIANGLESTRIP";
        case SDL_GPU_PRIMITIVETYPE_LINELIST: return "LINELIST";
        case SDL_GPU_PRIMITIVETYPE_LINESTRIP: return "LINESTRIP";
        case SDL_GPU_PRIMITIVETYPE_POINTLIST: return "POINTLIST";
    }
    return "?";
}

const char* fill_name(SDL_GPUFillMode v) {
    switch (v) {
        case SDL_GPU_FILLMODE_FILL: return "FILL";
        case SDL_GPU_FILLMODE_LINE: return "LINE";
    }
    return "?";
}

const char* cull_name(SDL_GPUCullMode v) {
    switch (v) {
        case SDL_GPU_CULLMODE_NONE: return "NONE";
        case SDL_GPU_CULLMODE_FRONT: return "FRONT";
        case SDL_GPU_CULLMODE_BACK: return "BACK";
    }
    return "?";
}

const char* front_face_name(SDL_GPUFrontFace v) {
    switch (v) {
        case SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE: return "COUNTER_CLOCKWISE";
        case SDL_GPU_FRONTFACE_CLOCKWISE: return "CLOCKWISE";
    }
    return "?";
}

const char* compare_name(SDL_GPUCompareOp v) {
    switch (v) {
        case SDL_GPU_COMPAREOP_INVALID: return "INVALID";
        case SDL_GPU_COMPAREOP_NEVER: return "NEVER";
        case SDL_GPU_COMPAREOP_LESS: return "LESS";
        case SDL_GPU_COMPAREOP_EQUAL: return "EQUAL";
        case SDL_GPU_COMPAREOP_LESS_OR_EQUAL: return "LESS_OR_EQUAL";
        case SDL_GPU_COMPAREOP_GREATER: return "GREATER";
        case SDL_GPU_COMPAREOP_NOT_EQUAL: return "NOT_EQUAL";
        case SDL_GPU_COMPAREOP_GREATER_OR_EQUAL: return "GREATER_OR_EQUAL";
        case SDL_GPU_COMPAREOP_ALWAYS: return "ALWAYS";
    }
    return "?";
}

const char* sample_count_name(SDL_GPUSampleCount v) {
    switch (v) {
        case SDL_GPU_SAMPLECOUNT_1: return "1";
        case SDL_GPU_SAMPLECOUNT_2: return "2";
        case SDL_GPU_SAMPLECOUNT_4: return "4";
        case SDL_GPU_SAMPLECOUNT_8: return "8";
    }
    return "?";
}

const char* blend_name(BlendMode v) {
    switch (v) {
        case BlendMode::None: return "None";
        case BlendMode::Alpha: return "Alpha";
        case BlendMode::Additive: return "Additive";
        case BlendMode::Multiply: return "Multiply";
    }
    return "?";
}

const char* texture_format_name(SDL_GPUTextureFormat v) {
    switch (v) {
        case SDL_GPU_TEXTUREFORMAT_INVALID: return "INVALID";
        case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
        case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
        case SDL_GPU_TEXTUREFORMAT_R8_UNORM: return "R8_UNORM";
        case SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT: return "R16G16_FLOAT";
        case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT: return "R16G16B16A16_FLOAT";
        case SDL_GPU_TEXTUREFORMAT_D32_FLOAT: return "D32_FLOAT";
        default: break;
    }
    return "OTHER";
}

const char* vertex_element_name(SDL_GPUVertexElementFormat v) {
    switch (v) {
        case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT: return "FLOAT";
        case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2: return "FLOAT2";
        case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3: return "FLOAT3";
        case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4: return "FLOAT4";
        case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4: return "UBYTE4";
        case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM: return "UBYTE4_NORM";
        default: break;
    }
    return "OTHER";
}

const char* input_rate_name(SDL_GPUVertexInputRate v) {
    switch (v) {
        case SDL_GPU_VERTEXINPUTRATE_VERTEX: return "VERTEX";
        case SDL_GPU_VERTEXINPUTRATE_INSTANCE: return "INSTANCE";
    }
    return "?";
}

const char* stage_name(ShaderStage v) {
    switch (v) {
        case ShaderStage::Vertex: return "vs";
        case ShaderStage::Fragment: return "fs";
        case ShaderStage::Compute: return "cs";
    }
    return "?";
}

std::string basename(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

} // namespace

std::string describe(const PipelineConfig& config) {
    std::string out;
    out += std::format("  raster    primitive={} fill={} cull={} front={}\n", primitive_name(config.primitive_type),
                       fill_name(config.fill_mode), cull_name(config.cull_mode), front_face_name(config.front_face));
    out += std::format("  bias      enable={} constant={:.6g} slope={:.6g} clamp={:.6g} depth_clip={}\n",
                       config.enable_depth_bias, config.depth_bias_constant, config.depth_bias_slope,
                       config.depth_bias_clamp, config.enable_depth_clip);
    out += std::format("  depth     test={} write={} compare={} stencil={}\n", config.depth_test_enable,
                       config.depth_write_enable, compare_name(config.depth_compare_op), config.stencil_test_enable);
    out += std::format("  targets   blend={} color={} has_depth={} depth={} samples={}\n", blend_name(config.blend_mode),
                       texture_format_name(config.color_format), config.has_depth_target,
                       texture_format_name(config.depth_format), sample_count_name(config.sample_count));
    for (const auto& vb : config.vertex_buffers) {
        out += std::format("  vbuffer   slot={} pitch={} rate={} step={}\n", vb.slot, vb.pitch,
                           input_rate_name(vb.input_rate), vb.instance_step_rate);
    }
    for (const auto& va : config.vertex_attributes) {
        out += std::format("  vattr     location={} slot={} format={} offset={}\n", va.location, va.buffer_slot,
                           vertex_element_name(va.format), va.offset);
    }
    return out;
}

std::string describe_shader(const std::string& path, ShaderStage stage, const ShaderResources& resources) {
    return std::format("  {}        {} samplers={} storage_textures={} storage_buffers={} uniform_buffers={}\n",
                       stage_name(stage), basename(path), resources.num_samplers, resources.num_storage_textures,
                       resources.num_storage_buffers, resources.num_uniform_buffers);
}

std::string describe_compute(const std::string& path, const std::string& entry_point,
                             const ComputeResources& resources) {
    return std::format("  cs        {} entry={} samplers={} ro_textures={} ro_buffers={} rw_textures={} "
                       "rw_buffers={} uniform_buffers={}\n",
                       basename(path), entry_point, resources.num_samplers, resources.num_readonly_storage_textures,
                       resources.num_readonly_storage_buffers, resources.num_readwrite_storage_textures,
                       resources.num_readwrite_storage_buffers, resources.num_uniform_buffers);
}

} // namespace bracken::gpu::golden
