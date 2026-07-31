#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_shader.hpp"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3_shadercross/SDL_shadercross.h>
#include <vector>

using namespace bracken::gpu;

// =============================================================================
// ComputeResources equality (used by the declared-vs-reflected cross-check)
// =============================================================================

TEST(ComputeResources, EqualityComparesAllFields) {
    ComputeResources a{.num_samplers = 1,
                       .num_readonly_storage_textures = 2,
                       .num_readonly_storage_buffers = 3,
                       .num_readwrite_storage_textures = 4,
                       .num_readwrite_storage_buffers = 5,
                       .num_uniform_buffers = 6};
    ComputeResources b = a;
    EXPECT_EQ(a, b);

    b.num_samplers = 9;
    EXPECT_NE(a, b);
    b = a;
    b.num_readonly_storage_textures = 9;
    EXPECT_NE(a, b);
    b = a;
    b.num_readonly_storage_buffers = 9;
    EXPECT_NE(a, b);
    b = a;
    b.num_readwrite_storage_textures = 9;
    EXPECT_NE(a, b);
    b = a;
    b.num_readwrite_storage_buffers = 9;
    EXPECT_NE(a, b);
    b = a;
    b.num_uniform_buffers = 9;
    EXPECT_NE(a, b);
}

// =============================================================================
// Compute SPIR-V reflection (GPUComputePipeline::reflect_spirv_resources)
//
// Pure CPU path: compiles a small compute HLSL shader to SPIR-V via the
// vendored SDL_shadercross (bundled DXC, no GPU device needed) and reflects
// resource counts + [numthreads] threadgroup dimensions.
// =============================================================================

namespace {

std::vector<uint8_t> compile_hlsl_to_spirv(const char* source, const char* entry_point) {
    SDL_ShaderCross_HLSL_Info info = {};
    info.source = source;
    info.entrypoint = entry_point;
    info.shader_stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE;

    size_t size = 0;
    void* blob = SDL_ShaderCross_CompileSPIRVFromHLSL(&info, &size);
    if (!blob || size == 0) {
        return {};
    }
    std::vector<uint8_t> bytecode(size);
    std::memcpy(bytecode.data(), blob, size);
    SDL_free(blob);
    return bytecode;
}

// Mirrors the engine's compute authoring convention (see
// exposure_histogram.comp.hlsl): sampled textures + read-only storage in
// set 0, read-write storage in set 1, uniforms in set 2.
constexpr const char* k_comp_source = R"(
[[vk::binding(0, 2)]]
cbuffer TestUniforms {
    uint width;
    uint height;
    float scale;
    float offset;
};

[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
Texture2D inputTexture;
[[vk::combinedImageSampler]][[vk::binding(0, 0)]]
SamplerState inputSampler;

[[vk::binding(1, 0)]]
StructuredBuffer<float> readOnlyData;

[[vk::binding(0, 1)]]
RWStructuredBuffer<uint> bins;

groupshared uint lds[64];

[numthreads(16, 8, 2)]
void CSMain(uint3 dtid : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    if (gi < 64) {
        lds[gi] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (dtid.x < width && dtid.y < height) {
        float v = inputTexture.SampleLevel(inputSampler, float2(0.5, 0.5), 0).r;
        uint bin = (uint)saturate(v * scale + offset + readOnlyData[0]) * 63;
        InterlockedAdd(bins[bin], 1u);
    }
}
)";

} // namespace

TEST(ComputeReflection, CountsAndThreadgroupMatchDeclarations) {
    if (!GPUShader::init_compiler()) {
        GTEST_SKIP() << "SDL_shadercross unavailable in this environment";
    }

    std::vector<uint8_t> spirv = compile_hlsl_to_spirv(k_comp_source, "CSMain");
    ASSERT_FALSE(spirv.empty()) << "HLSL->SPIR-V compilation failed";

    ComputeResources reflected;
    uint32_t threadcount[3] = {0, 0, 0};
    ASSERT_TRUE(GPUComputePipeline::reflect_spirv_resources(spirv, reflected, threadcount));

    EXPECT_EQ(reflected.num_samplers, 1u);
    EXPECT_EQ(reflected.num_readonly_storage_textures, 0u);
    EXPECT_EQ(reflected.num_readonly_storage_buffers, 1u);
    EXPECT_EQ(reflected.num_readwrite_storage_textures, 0u);
    EXPECT_EQ(reflected.num_readwrite_storage_buffers, 1u);
    EXPECT_EQ(reflected.num_uniform_buffers, 1u);

    // [numthreads(16, 8, 2)]
    EXPECT_EQ(threadcount[0], 16u);
    EXPECT_EQ(threadcount[1], 8u);
    EXPECT_EQ(threadcount[2], 2u);
}

TEST(ComputeReflection, EmptyBytecodeFails) {
    ComputeResources reflected{.num_samplers = 99};
    std::vector<uint8_t> empty;
    EXPECT_FALSE(GPUComputePipeline::reflect_spirv_resources(empty, reflected));
    // Output untouched on failure.
    EXPECT_EQ(reflected.num_samplers, 99u);
}

TEST(ComputeReflection, GarbageBytecodeFails) {
    if (!GPUShader::init_compiler()) {
        GTEST_SKIP() << "SDL_shadercross unavailable in this environment";
    }
    std::vector<uint8_t> garbage(64, 0xAB);
    ComputeResources reflected;
    EXPECT_FALSE(GPUComputePipeline::reflect_spirv_resources(garbage, reflected));
}

// group_count_* ceil-division helpers need no GPU.
TEST(ComputePipelineHelpers, GroupCountCeilDivision) {
    // Exercise the arithmetic through a value-constructed pipeline is not
    // possible (private ctor); the helpers are trivial wrappers over the
    // threadcounts, so pin the formula directly.
    constexpr auto ceil_div = [](uint32_t items, uint32_t tc) { return (items + tc - 1) / tc; };
    EXPECT_EQ(ceil_div(1600u, 16u), 100u);
    EXPECT_EQ(ceil_div(1601u, 16u), 101u);
    EXPECT_EQ(ceil_div(900u, 16u), 57u);
    EXPECT_EQ(ceil_div(1u, 256u), 1u);
    EXPECT_EQ(ceil_div(256u, 256u), 1u);
    EXPECT_EQ(ceil_div(257u, 256u), 2u);
}
