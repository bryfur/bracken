// The pipeline catalogue's golden test — the gate for refactor phase 6c.
//
// gpu/pipeline_config_golden.inc was captured from the 38 hand-written pipeline
// creators that used to live in pipeline_registry.cpp, before any of them was
// touched (see the phase 6c(1) commit). It records, for every PipelineType and
// ComputePipelineType, the shader pair, the declared resource counts and every
// field of the resulting PipelineConfig.
//
// This test rebuilds all of that from the catalogue and demands a byte-for-byte
// match. It is the only mechanical check in the tree that sees GPU *state*: a
// row with the wrong depth compare, or a prepass variant that writes depth,
// shows up nowhere else until it surfaces as a subtly wrong image.
//
// If you deliberately change a pipeline, the golden has to be updated in the
// same change and the new block reviewed by eye — especially the `depth`
// line. The engine renders reversed-Z (GREATER_OR_EQUAL, clear 0.0); the only
// legitimate departures are the shadow cascades' standard-Z LESS_OR_EQUAL, the
// post-prepass EQUAL variants and the local-shadow tile clear's ALWAYS.

#include "bracken/gpu/pipeline_catalog.hpp"

#include "bracken/gpu/gpu_types.hpp"

#include <format>
#include <gtest/gtest.h>
#include <fstream>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace bracken::gpu {
namespace {

constexpr std::string_view k_golden =
#include "bracken/gpu/pipeline_config_golden.inc"
    ;

// The formats the golden was captured with. Deliberately not the live device
// formats: the swapchain format is whatever the driver reported, so pinning it
// keeps the golden a property of the catalogue rather than of the host.
constexpr PipelineFormats k_canonical_formats{
    .swapchain = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
    .scene_color = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
    .velocity = SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT,
    .depth = SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
};

// =============================================================================
// Symbolic rendering of a PipelineConfig — identical to the capture harness's
// formatter, so a diff against the golden is a diff in the data, never in the
// presentation.
// =============================================================================

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
        case BlendMode::AdditivePreserveAlpha: return "AdditivePreserveAlpha";
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
    // Emitted only when present, so adding MRT did not churn every golden row.
    if (config.color_format2 != SDL_GPU_TEXTUREFORMAT_INVALID) {
        out += std::format("  targets2  color2={}\n", texture_format_name(config.color_format2));
    }
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

std::string describe_shader(const char* stage, const std::string& path, const ShaderResources& resources) {
    return std::format("  {}        {} samplers={} storage_textures={} storage_buffers={} uniform_buffers={}\n", stage,
                       path, resources.num_samplers, resources.num_storage_textures, resources.num_storage_buffers,
                       resources.num_uniform_buffers);
}

std::string describe_compute(const std::string& path, const std::string& entry_point,
                             const ComputeResources& resources) {
    return std::format("  cs        {} entry={} samplers={} ro_textures={} ro_buffers={} rw_textures={} "
                       "rw_buffers={} uniform_buffers={}\n",
                       path, entry_point, resources.num_samplers, resources.num_readonly_storage_textures,
                       resources.num_readonly_storage_buffers, resources.num_readwrite_storage_textures,
                       resources.num_readwrite_storage_buffers, resources.num_uniform_buffers);
}

// =============================================================================
// Golden parsing
// =============================================================================
// The golden is a flat text file: an unindented line names a pipeline, the
// two-space-indented lines that follow are its record.

struct Golden {
    std::map<std::string, std::string> blocks;
    std::vector<std::string> order;
};

Golden parse_golden() {
    Golden golden;
    std::string current;
    size_t pos = 0;
    while (pos <= k_golden.size()) {
        const size_t nl = k_golden.find('\n', pos);
        const std::string_view line = k_golden.substr(pos, (nl == std::string_view::npos ? k_golden.size() : nl) - pos);
        pos = (nl == std::string_view::npos) ? k_golden.size() + 1 : nl + 1;

        if (line.empty()) {
            continue;
        }
        if (line.starts_with("  ")) {
            if (!current.empty()) {
                golden.blocks[current] += std::string(line) + "\n";
            }
            continue;
        }
        if (line.starts_with("Canonical formats:")) {
            continue;
        }
        current = std::string(line);
        golden.order.push_back(current);
        golden.blocks[current]; // create empty
    }
    return golden;
}

// Regenerating the golden. It is a RECORD of reviewed GPU state, so this is
// deliberately not automatic: set the variable, run the suite, diff what it
// wrote, and check in only the changes you meant to make.
//
//   BRACKEN_REGEN_PIPELINE_GOLDEN=src/bracken/gpu/pipeline_config_golden.inc ./build/src/bracken/bracken_tests --gtest_filter='PipelineCatalog.*'
//
// A diff bigger than the row you touched means the change was not local.
TEST(PipelineCatalog, RegenerateGoldenOnRequest) {
    const char* out_path = std::getenv("BRACKEN_REGEN_PIPELINE_GOLDEN");
    if (out_path == nullptr) {
        GTEST_SKIP() << "set BRACKEN_REGEN_PIPELINE_GOLDEN=<path> to rewrite the golden";
    }
    std::string out = "// GENERATED by BRACKEN_DUMP_PIPELINE_GOLDEN — do not edit by hand.\n";
    out += "// The GPU state every pipeline in the catalogue must produce.\n";
    out += "R\"GOLDEN(\n";
    out += std::format("Canonical formats: swapchain={} scene={} velocity={} depth={}\n",
                       texture_format_name(k_canonical_formats.swapchain),
                       texture_format_name(k_canonical_formats.scene_color),
                       texture_format_name(k_canonical_formats.velocity),
                       texture_format_name(k_canonical_formats.depth));
    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        const PipelineDesc& desc = pipeline_desc(static_cast<PipelineType>(i));
        out += std::string(desc.name) + "\n";
        out += describe_shader("vs", desc.vertex_shader, desc.vs);
        out += describe_shader("fs", desc.fragment_shader, desc.fs);
        out += describe(build_pipeline_config(desc, k_canonical_formats));
    }
    for (int i = 0; i < static_cast<int>(ComputePipelineType::Count); ++i) {
        const ComputePipelineDesc& desc = compute_pipeline_desc(static_cast<ComputePipelineType>(i));
        out += std::string(desc.name) + "\n";
        out += describe_compute(desc.shader, desc.entry_point, desc.resources);
    }
    out += ")GOLDEN\"\n";
    std::ofstream file(out_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(file) << "cannot write " << out_path;
    file << out;
}

TEST(PipelineCatalog, GoldenParsesIntoOnePerTypeBlock) {
    const Golden golden = parse_golden();
    ASSERT_EQ(golden.order.size(),
              static_cast<size_t>(PipelineType::Count) + static_cast<size_t>(ComputePipelineType::Count))
        << "gpu/pipeline_config_golden.inc does not cover the enumeration";
    for (const auto& [name, block] : golden.blocks) {
        EXPECT_FALSE(block.empty()) << "empty golden block for " << name;
    }
}

TEST(PipelineCatalog, EveryGraphicsPipelineMatchesTheGolden) {
    const Golden golden = parse_golden();
    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        const auto type = static_cast<PipelineType>(i);
        const PipelineDesc& desc = pipeline_desc(type);

        // The golden's block order is the enum order, so a row that drifted to
        // the wrong index is caught here rather than by a confusing field diff.
        ASSERT_LT(static_cast<size_t>(i), golden.order.size());
        EXPECT_EQ(golden.order[static_cast<size_t>(i)], std::string(desc.name))
            << "catalogue row " << i << " is out of enum order";

        std::string actual;
        actual += describe_shader("vs", desc.vertex_shader, desc.vs);
        actual += describe_shader("fs", desc.fragment_shader, desc.fs);
        actual += describe(build_pipeline_config(desc, k_canonical_formats));

        const auto it = golden.blocks.find(desc.name);
        ASSERT_NE(it, golden.blocks.end()) << "no golden block for " << desc.name;
        EXPECT_EQ(actual, it->second) << "pipeline " << desc.name << " no longer builds the recorded GPU state";
    }
}

TEST(PipelineCatalog, ComputeDescriptorOrderMatchesTheEnum) {
    // THE BUG THIS EXISTS FOR, found the hard way on 2026-08-08. The descriptor
    // array is indexed BY THE ENUM, so inserting a row at the wrong position
    // silently re-points every pipeline after it: four raster-GI rows added
    // above `GiProbeTrace` instead of below made GI probe tracing dispatch a
    // voxel kernel, and the only symptom was that one screenshot pin moved by
    // 5.6 luma. Nothing else caught it — the golden test happily recorded the
    // wrong mapping, because a golden regenerated from a broken table is a
    // faithful record of the break.
    //
    // Names are unique and the enum is the authority, so each type's descriptor
    // must be the one whose name matches. That single property is enough to
    // pin the ordering.
    for (int i = 0; i < static_cast<int>(ComputePipelineType::Count); ++i) {
        const auto type = static_cast<ComputePipelineType>(i);
        const ComputePipelineDesc& desc = compute_pipeline_desc(type);
        ASSERT_NE(desc.name, nullptr) << "index " << i;
        EXPECT_STRNE(desc.name, "Unknown")
            << "index " << i << " has no descriptor — the array is shorter than the enum";
        EXPECT_STRNE(desc.shader, "")
            << "index " << i << " (" << desc.name << ") names no shader";
    }

    // And no two entries may share a name, which is what makes the check above
    // meaningful rather than merely non-empty.
    std::set<std::string_view> seen;
    for (int i = 0; i < static_cast<int>(ComputePipelineType::Count); ++i) {
        const ComputePipelineDesc& desc =
            compute_pipeline_desc(static_cast<ComputePipelineType>(i));
        EXPECT_TRUE(seen.insert(desc.name).second)
            << "duplicate compute pipeline name '" << desc.name << "' at index " << i
            << " — two enum values map to one descriptor, so one of them dispatches the wrong "
               "shader";
    }
}

TEST(PipelineCatalog, EveryComputePipelineMatchesTheGolden) {
    const Golden golden = parse_golden();
    for (int i = 0; i < static_cast<int>(ComputePipelineType::Count); ++i) {
        const auto type = static_cast<ComputePipelineType>(i);
        const ComputePipelineDesc& desc = compute_pipeline_desc(type);

        const std::string actual = describe_compute(desc.shader, desc.entry_point, desc.resources);
        const auto it = golden.blocks.find(desc.name);
        ASSERT_NE(it, golden.blocks.end()) << "no golden block for compute " << desc.name;
        EXPECT_EQ(actual, it->second) << "compute pipeline " << desc.name << " no longer matches the recorded module";
    }
}

// =============================================================================
// Properties the table now makes checkable, which the creators never were
// =============================================================================

TEST(PipelineCatalog, SwapchainDependentSetComesFromTheTable) {
    // This used to be a hand-written 4-entry list sitting next to 56 creators,
    // consulted by set_swapchain_format. Now it is a query; the expected set
    // is spelled out here so that adding a swapchain-targeting pipeline has to
    // be a deliberate, reviewed change.
    const std::set<PipelineType> expected{PipelineType::UI, PipelineType::Text, PipelineType::Composite,
                                          PipelineType::Fxaa};
    std::set<PipelineType> actual;
    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        const auto type = static_cast<PipelineType>(i);
        if (pipeline_targets_swapchain(type)) {
            actual.insert(type);
        }
    }
    EXPECT_EQ(actual, expected);
}

TEST(PipelineCatalog, SwapchainTargetsFollowTheSuppliedFormat) {
    PipelineFormats formats = k_canonical_formats;
    formats.swapchain = SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;

    const PipelineConfig composite = build_pipeline_config(pipeline_desc(PipelineType::Composite), formats);
    EXPECT_EQ(composite.color_format, SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM);

    // ...and nothing else moves: scene/HDR/post pipelines render to fixed-format
    // offscreen targets, which is why set_swapchain_format rebuilds only four.
    const PipelineConfig model = build_pipeline_config(pipeline_desc(PipelineType::Model), formats);
    EXPECT_EQ(model.color_format, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT);
}

TEST(PipelineCatalog, DepthOnlyRowsHaveNoColorTarget) {
    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        const auto type = static_cast<PipelineType>(i);
        const PipelineDesc& desc = pipeline_desc(type);
        const PipelineConfig config = build_pipeline_config(desc, k_canonical_formats);
        const bool depth_only = desc.color == TargetFormat::DepthOnly;
        EXPECT_EQ(config.color_format == SDL_GPU_TEXTUREFORMAT_INVALID, depth_only) << desc.name;
        // A caster with no color target must still have a depth target —
        // otherwise it writes nothing at all.
        if (depth_only) {
            EXPECT_TRUE(config.has_depth_target) << desc.name;
        }
    }
}

TEST(PipelineCatalog, NoDepthTargetForcesTestAndWriteOff) {
    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        const auto type = static_cast<PipelineType>(i);
        const PipelineDesc& desc = pipeline_desc(type);
        if (desc.depth_target) {
            continue;
        }
        const PipelineConfig config = build_pipeline_config(desc, k_canonical_formats);
        EXPECT_FALSE(config.depth_test_enable) << desc.name;
        EXPECT_FALSE(config.depth_write_enable) << desc.name;
    }
}

TEST(PipelineCatalog, OnlyShadowCastersDepartFromReversedZ) {
    // The reversed-Z contract, as a checkable fact instead of a comment: every
    // scene pipeline compares GREATER_OR_EQUAL, and the three families that do
    // not are named here.
    const std::set<std::string> standard_z{"InstancedShadowModel", "ShadowModel", "ShadowSkinnedModel",
                                           "ShadowTerrain"};
    const std::set<std::string> depth_equal{
        "ModelDepthEqual",       "ModelDoubleSidedDepthEqual",          "SkinnedModelDepthEqual",
        "SkinnedModelDoubleSidedDepthEqual", "InstancedModelDepthEqual", "InstancedModelDoubleSidedDepthEqual",
        "TerrainDepthEqual",     "GrassDepthEqual"};

    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        const PipelineDesc& desc = pipeline_desc(static_cast<PipelineType>(i));
        const std::string name = desc.name;
        if (standard_z.contains(name)) {
            EXPECT_EQ(desc.depth_compare, SDL_GPU_COMPAREOP_LESS_OR_EQUAL) << name;
            // Cascade casters also pancake geometry in front of the near plane.
            EXPECT_FALSE(desc.depth_clip) << name;
            EXPECT_TRUE(desc.depth_bias.enabled) << name;
        } else if (depth_equal.contains(name)) {
            EXPECT_EQ(desc.depth_compare, SDL_GPU_COMPAREOP_EQUAL) << name;
            EXPECT_FALSE(desc.depth_write) << name << ": the prepass already wrote this depth";
        } else if (name == "LocalShadowClear") {
            EXPECT_EQ(desc.depth_compare, SDL_GPU_COMPAREOP_ALWAYS) << name;
        } else {
            EXPECT_EQ(desc.depth_compare, SDL_GPU_COMPAREOP_GREATER_OR_EQUAL) << name;
        }
    }
}

TEST(PipelineCatalog, OutOfRangeYieldsThePlaceholderRow) {
    EXPECT_STREQ(pipeline_desc(PipelineType::Count).name, "Unknown");
    EXPECT_STREQ(pipeline_desc(PipelineType::Count).vertex_shader, "");
    EXPECT_STREQ(compute_pipeline_desc(ComputePipelineType::Count).name, "Unknown");
    EXPECT_FALSE(pipeline_targets_swapchain(PipelineType::Count));
}

} // namespace
} // namespace bracken::gpu
