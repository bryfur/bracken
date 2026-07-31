#include "bracken/gpu/gpu_shader.hpp"
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3_shadercross/SDL_shadercross.h>
#include <vector>

using namespace bracken::gpu;

// =============================================================================
// ShaderResources equality (used by the ShaderManager cache-hit mismatch check)
// =============================================================================

TEST(ShaderResources, EqualityComparesAllFields) {
    ShaderResources a{.num_samplers = 1, .num_storage_textures = 2, .num_storage_buffers = 3, .num_uniform_buffers = 4};
    ShaderResources b = a;
    EXPECT_EQ(a, b);

    b.num_samplers = 9;
    EXPECT_NE(a, b);
    b = a;
    b.num_storage_textures = 9;
    EXPECT_NE(a, b);
    b = a;
    b.num_storage_buffers = 9;
    EXPECT_NE(a, b);
    b = a;
    b.num_uniform_buffers = 9;
    EXPECT_NE(a, b);
}

// =============================================================================
// SPIR-V reflection (GPUShader::reflect_spirv_resources)
//
// Pure CPU path: compiles a small HLSL shader to SPIR-V via the vendored
// SDL_shadercross (bundled DXC, no GPU device needed) and reflects it.
// =============================================================================

namespace {

std::vector<uint8_t> compile_hlsl_to_spirv(const char* source, const char* entry_point,
                                           SDL_ShaderCross_ShaderStage stage) {
    SDL_ShaderCross_HLSL_Info info = {};
    info.source = source;
    info.entrypoint = entry_point;
    info.shader_stage = stage;

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

// Mirrors the engine's shader authoring convention (see ui.frag.hlsl):
// fragment textures/samplers in set 2, fragment uniforms in set 3.
constexpr const char* k_frag_source = R"(
struct PSInput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 texcoord : TEXCOORD0;
};

[[vk::binding(0, 3)]]
cbuffer TestUniforms {
    float4 tint;
};

[[vk::combinedImageSampler]]
[[vk::binding(0, 2)]]
Texture2D test_texture;
[[vk::combinedImageSampler]]
[[vk::binding(0, 2)]]
SamplerState test_sampler;

float4 PSMain(PSInput input) : SV_Target {
    return test_texture.Sample(test_sampler, input.texcoord) * tint;
}
)";

// Vertex uniforms in set 1; no samplers or storage resources.
constexpr const char* k_vert_source = R"(
struct VSInput {
    [[vk::location(0)]] float3 position : POSITION;
};

struct VSOutput {
    float4 position : SV_Position;
};

[[vk::binding(0, 1)]]
cbuffer VertUniforms {
    float4x4 mvp;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.position = mul(mvp, float4(input.position, 1.0));
    return output;
}
)";

} // namespace

TEST(ShaderReflection, FragmentCountsMatchDeclarations) {
    if (!GPUShader::init_compiler()) {
        GTEST_SKIP() << "SDL_shadercross unavailable in this environment";
    }

    std::vector<uint8_t> spirv =
        compile_hlsl_to_spirv(k_frag_source, "PSMain", SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    ASSERT_FALSE(spirv.empty()) << "HLSL->SPIR-V compilation failed";

    ShaderResources reflected;
    ASSERT_TRUE(GPUShader::reflect_spirv_resources(spirv, reflected));

    EXPECT_EQ(reflected.num_samplers, 1u);
    EXPECT_EQ(reflected.num_uniform_buffers, 1u);
    EXPECT_EQ(reflected.num_storage_textures, 0u);
    EXPECT_EQ(reflected.num_storage_buffers, 0u);
}

TEST(ShaderReflection, VertexCountsMatchDeclarations) {
    if (!GPUShader::init_compiler()) {
        GTEST_SKIP() << "SDL_shadercross unavailable in this environment";
    }

    std::vector<uint8_t> spirv = compile_hlsl_to_spirv(k_vert_source, "VSMain", SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    ASSERT_FALSE(spirv.empty()) << "HLSL->SPIR-V compilation failed";

    ShaderResources reflected;
    ASSERT_TRUE(GPUShader::reflect_spirv_resources(spirv, reflected));

    EXPECT_EQ(reflected.num_samplers, 0u);
    EXPECT_EQ(reflected.num_uniform_buffers, 1u);
    EXPECT_EQ(reflected.num_storage_textures, 0u);
    EXPECT_EQ(reflected.num_storage_buffers, 0u);
}

TEST(ShaderReflection, EmptyBytecodeFails) {
    ShaderResources reflected{.num_samplers = 99};
    std::vector<uint8_t> empty;
    EXPECT_FALSE(GPUShader::reflect_spirv_resources(empty, reflected));
    // Output untouched on failure.
    EXPECT_EQ(reflected.num_samplers, 99u);
}

TEST(ShaderReflection, GarbageBytecodeFails) {
    if (!GPUShader::init_compiler()) {
        GTEST_SKIP() << "SDL_shadercross unavailable in this environment";
    }
    std::vector<uint8_t> garbage(64, 0xAB);
    ShaderResources reflected;
    EXPECT_FALSE(GPUShader::reflect_spirv_resources(garbage, reflected));
}
