#include "bracken/gpu/host_pipeline.hpp"

namespace bracken::gpu {

PipelineConfig build_host_pipeline_config(const HostGraphicsPipelineDesc& desc, const PipelineFormats& formats) {
    // Project onto a catalogue row and reuse build_pipeline_config rather than
    // re-deriving the same state here. The whole point is that a host pipeline
    // and a catalogue pipeline cannot disagree about reversed-Z, about the
    // depth-target force-off, or about what a TargetFormat resolves to — and
    // "they call the same function" is the only version of that claim which
    // survives a future edit to either side.
    //
    // The row's three `const char*` name fields stay empty: build_pipeline_config
    // never reads them (the registry resolves shaders itself), so there is no
    // lifetime question to answer here.
    PipelineDesc row{};
    row.vertex_layout = desc.vertex_layout;
    row.primitive_type = desc.primitive_type;
    row.cull_mode = desc.cull_mode;
    row.blend_mode = desc.blend_mode;
    row.depth_test = desc.depth_test;
    row.depth_write = desc.depth_write;
    row.depth_compare = desc.depth_compare;
    row.depth_clip = desc.depth_clip;
    row.depth_bias = desc.depth_bias;
    row.color = desc.color;
    row.color2 = desc.color2;
    row.depth_target = desc.depth_target;

    PipelineConfig config = build_pipeline_config(row, formats);

    // Explicit overrides, applied after — INVALID means "keep what the
    // TargetFormat resolved to". Applying them last is what makes an override
    // win over the enum without the enum having to grow a Custom case.
    if (desc.color_format_override != SDL_GPU_TEXTUREFORMAT_INVALID) {
        config.color_format = desc.color_format_override;
    }
    if (desc.color2_format_override != SDL_GPU_TEXTUREFORMAT_INVALID) {
        config.color_format2 = desc.color2_format_override;
    }
    if (desc.depth_format_override != SDL_GPU_TEXTUREFORMAT_INVALID) {
        config.depth_format = desc.depth_format_override;
    }

    // An explicit vertex format REPLACES the named layout, attributes and all.
    // Taken as a pair deliberately: a host that supplies buffers and forgets
    // attributes wants an empty attribute list, not the engine's leftovers
    // from a layout it also named.
    if (!desc.buffers.empty()) {
        config.vertex_buffers = desc.buffers;
        config.vertex_attributes = desc.attributes;
    }

    return config;
}

} // namespace bracken::gpu
