#include "engine/gpu/gpu_uniforms.hpp"
#include <cstddef>
#include <gtest/gtest.h>

using namespace mmo::engine::gpu;

// =============================================================================
// Alignment tests - all uniform structs must be 16-byte aligned for GPU cbuffers
// =============================================================================

TEST(UniformAlignment, AllStructsAre16ByteAligned) {
    EXPECT_EQ(alignof(ModelTransformUniforms), 16u);
    EXPECT_EQ(alignof(ModelLightingUniforms), 16u);
    EXPECT_EQ(alignof(SkyboxFragmentUniforms), 16u);
    EXPECT_EQ(alignof(GridVertexUniforms), 16u);
    EXPECT_EQ(alignof(UIScreenUniforms), 16u);
    EXPECT_EQ(alignof(UIFragmentUniforms), 16u);
    EXPECT_EQ(alignof(InstancedCameraUniforms), 16u);
    EXPECT_EQ(alignof(InstancedLightingUniforms), 16u);
    EXPECT_EQ(alignof(InstancedShadowUniforms), 16u);
    EXPECT_EQ(alignof(ShadowTransformUniforms), 16u);
    EXPECT_EQ(alignof(ShadowSkinnedTransformUniforms), 16u);
    EXPECT_EQ(alignof(ShadowTerrainUniforms), 16u);
    EXPECT_EQ(alignof(ShadowFragUniforms), 16u);
    EXPECT_EQ(alignof(ShadowDataUniforms), 16u);
    EXPECT_EQ(alignof(GTAOUniforms), 16u);
    EXPECT_EQ(alignof(BlurUniforms), 16u);
    EXPECT_EQ(alignof(CompositeUniforms), 16u);
    EXPECT_EQ(alignof(FxaaUniforms), 16u);
    EXPECT_EQ(alignof(ParticleFragmentUniforms), 16u);
    EXPECT_EQ(alignof(ParticleBatchUniforms), 16u);
    EXPECT_EQ(alignof(BloomDownsampleUniforms), 16u);
    EXPECT_EQ(alignof(BloomUpsampleUniforms), 16u);
    EXPECT_EQ(alignof(VolumetricFogUniforms), 16u);
}

// =============================================================================
// Size tests - sizes must be multiples of 16 and match shader cbuffer layout
// =============================================================================

TEST(UniformSizes, SizesAreMultiplesOf16) {
    // GPU uniform buffers require 16-byte aligned sizes
    EXPECT_EQ(sizeof(ModelTransformUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(ModelLightingUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(SkyboxFragmentUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(GridVertexUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(UIScreenUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(UIFragmentUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(InstancedCameraUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(InstancedLightingUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(InstancedShadowUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(ShadowTransformUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(ShadowSkinnedTransformUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(ShadowTerrainUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(ShadowFragUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(ShadowDataUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(GTAOUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(BlurUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(CompositeUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(FxaaUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(ParticleFragmentUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(ParticleBatchUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(BloomDownsampleUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(BloomUpsampleUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(VolumetricFogUniforms) % 16, 0u);
}

// =============================================================================
// ModelTransformUniforms - matches model.vert.hlsl TransformUniforms
// =============================================================================

TEST(ModelTransformUniformsLayout, SizeIs288Bytes) {
    // 3 mat4 (192) + vec3+pad (16) + mat4 (64) + int+uint+pad[2] (16) = 288
    EXPECT_EQ(sizeof(ModelTransformUniforms), 288u);
}

TEST(ModelTransformUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(ModelTransformUniforms, model), 0u);
    EXPECT_EQ(offsetof(ModelTransformUniforms, view), 64u);
    EXPECT_EQ(offsetof(ModelTransformUniforms, projection), 128u);
    EXPECT_EQ(offsetof(ModelTransformUniforms, cameraPos), 192u);
    EXPECT_EQ(offsetof(ModelTransformUniforms, normalMatrix), 208u);
    EXPECT_EQ(offsetof(ModelTransformUniforms, useSkinning), 272u);
    EXPECT_EQ(offsetof(ModelTransformUniforms, paletteOffset), 276u);
}

// =============================================================================
// ShadowSkinnedTransformUniforms - matches shadow_skinned.vert.hlsl
// =============================================================================

TEST(ShadowSkinnedTransformUniformsLayout, SizeIs144Bytes) {
    // 2 mat4 (128) + uint+pad[3] (16) = 144
    EXPECT_EQ(sizeof(ShadowSkinnedTransformUniforms), 144u);
}

TEST(ShadowSkinnedTransformUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(ShadowSkinnedTransformUniforms, lightViewProjection), 0u);
    EXPECT_EQ(offsetof(ShadowSkinnedTransformUniforms, model), 64u);
    EXPECT_EQ(offsetof(ShadowSkinnedTransformUniforms, paletteOffset), 128u);
}

// =============================================================================
// ShadowFragUniforms - matches shadow_depth.frag.hlsl ShadowFragUniforms
// =============================================================================

TEST(ShadowFragUniformsLayout, SizeIs16Bytes) {
    // int + 3 named scalar pads = 16. The HLSL side must pad with scalars too:
    // a cbuffer float[3] array would stride at 16 bytes and declare a 52-byte
    // block that no longer matches this struct.
    EXPECT_EQ(sizeof(ShadowFragUniforms), 16u);
}

TEST(ShadowFragUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(ShadowFragUniforms, hasTexture), 0u);
    EXPECT_EQ(offsetof(ShadowFragUniforms, _pad0), 4u);
}

// =============================================================================
// UIFragmentUniforms - matches ui.frag.hlsl UIFragmentUniforms
// =============================================================================

TEST(UIFragmentUniformsLayout, SizeIs16Bytes) {
    // int + 3 named scalar pads = 16 (see ShadowFragUniformsLayout.SizeIs16Bytes).
    EXPECT_EQ(sizeof(UIFragmentUniforms), 16u);
}

TEST(UIFragmentUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(UIFragmentUniforms, hasTexture), 0u);
    EXPECT_EQ(offsetof(UIFragmentUniforms, _padding0), 4u);
}

// =============================================================================
// ModelLightingUniforms - matches model.frag.hlsl LightingUniforms
// =============================================================================

TEST(ModelLightingUniformsLayout, SizeIs208Bytes) {
    // 4 vec3+pad (64) + 2 vec4 (32) + 4 vec3+float (64) + 12 scalars (48) = 208
    EXPECT_EQ(sizeof(ModelLightingUniforms), 208u);
}

TEST(ModelLightingUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(ModelLightingUniforms, lightDir), 0u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, lightColor), 16u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, ambientSky), 32u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, ambientGround), 48u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, tintColor), 64u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, baseColorFactor), 80u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, fogColor), 96u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, fogStart), 108u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, fogSunColor), 112u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, fogEnd), 124u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, emissiveFactor), 128u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, fogHeightFalloff), 140u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, cameraPos), 144u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, normalScale), 156u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, hasTexture), 160u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, fogEnabled), 164u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, hasNormalMap), 168u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, ambientStrength), 172u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, sunIntensity), 176u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, metallicFactor), 180u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, roughnessFactor), 184u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, occlusionStrength), 188u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, hasMRMap), 192u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, hasEmissiveMap), 196u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, hasPackedOcclusion), 200u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, alphaCutoff), 204u);
}

// =============================================================================
// InstancedLightingUniforms - matches model_instanced.frag.hlsl LightingUniforms
// =============================================================================

TEST(InstancedLightingUniformsLayout, SizeIs192Bytes) {
    // ModelLightingUniforms layout minus the vec4 tintColor (tint is per-instance)
    EXPECT_EQ(sizeof(InstancedLightingUniforms), 192u);
}

TEST(InstancedLightingUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(InstancedLightingUniforms, lightDir), 0u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, lightColor), 16u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, ambientSky), 32u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, ambientGround), 48u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, baseColorFactor), 64u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, fogColor), 80u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, fogStart), 92u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, fogSunColor), 96u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, fogEnd), 108u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, emissiveFactor), 112u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, fogHeightFalloff), 124u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, cameraPos), 128u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, normalScale), 140u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, hasTexture), 144u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, fogEnabled), 148u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, hasNormalMap), 152u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, ambientStrength), 156u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, sunIntensity), 160u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, metallicFactor), 164u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, roughnessFactor), 168u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, occlusionStrength), 172u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, hasMRMap), 176u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, hasEmissiveMap), 180u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, hasPackedOcclusion), 184u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, alphaCutoff), 188u);
}

// =============================================================================
// SkyboxFragmentUniforms - matches skybox.frag.hlsl SkyUniforms
// =============================================================================

TEST(SkyboxFragmentUniformsLayout, SizeIs176Bytes) {
    // mat4 (64) + 7 * (vec3 + float) rows (112) = 176
    EXPECT_EQ(sizeof(SkyboxFragmentUniforms), 176u);
}

TEST(SkyboxFragmentUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(SkyboxFragmentUniforms, invViewProjection), 0u);
    EXPECT_EQ(offsetof(SkyboxFragmentUniforms, sunDirection), 64u);
    EXPECT_EQ(offsetof(SkyboxFragmentUniforms, time), 76u);
    EXPECT_EQ(offsetof(SkyboxFragmentUniforms, sunColor), 80u);
    EXPECT_EQ(offsetof(SkyboxFragmentUniforms, zenithColor), 96u);
    EXPECT_EQ(offsetof(SkyboxFragmentUniforms, horizonColor), 112u);
    EXPECT_EQ(offsetof(SkyboxFragmentUniforms, groundColor), 128u);
    EXPECT_EQ(offsetof(SkyboxFragmentUniforms, fogHorizonColor), 144u);
    EXPECT_EQ(offsetof(SkyboxFragmentUniforms, fogSunColor), 160u);
}

// =============================================================================
// CompositeUniforms - matches composite.frag.hlsl
// =============================================================================

TEST(CompositeUniformsLayout, SizeIs48Bytes) {
    // 8 floats (ao/bloom/fog + exposure/tonemap/contrast/saturation) +
    // vec2 lowResTexelSize + depthA + depthB = 48 bytes (3 16-byte blocks)
    EXPECT_EQ(sizeof(CompositeUniforms), 48u);
}

TEST(CompositeUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(CompositeUniforms, aoStrength), 0u);
    EXPECT_EQ(offsetof(CompositeUniforms, bloomStrength), 4u);
    EXPECT_EQ(offsetof(CompositeUniforms, volumetricFogEnabled), 8u);
    EXPECT_EQ(offsetof(CompositeUniforms, exposure), 16u);
    EXPECT_EQ(offsetof(CompositeUniforms, tonemapMode), 20u);
    EXPECT_EQ(offsetof(CompositeUniforms, contrast), 24u);
    EXPECT_EQ(offsetof(CompositeUniforms, saturation), 28u);
    EXPECT_EQ(offsetof(CompositeUniforms, lowResTexelSize), 32u);
    EXPECT_EQ(offsetof(CompositeUniforms, depthA), 40u);
    EXPECT_EQ(offsetof(CompositeUniforms, depthB), 44u);
}

TEST(CompositeUniformsLayout, DefaultValues) {
    CompositeUniforms u;
    EXPECT_FLOAT_EQ(u.aoStrength, 1.0f);
    EXPECT_FLOAT_EQ(u.bloomStrength, 0.35f);
    EXPECT_FLOAT_EQ(u.volumetricFogEnabled, 0.0f);
    EXPECT_FLOAT_EQ(u.exposure, 1.0f);
    EXPECT_EQ(u.tonemapMode, 0);
    EXPECT_FLOAT_EQ(u.contrast, 1.0f);
    EXPECT_FLOAT_EQ(u.saturation, 1.0f);
    // depthA/depthB of 0 degrade to plain bilinear weights in the shader.
    EXPECT_FLOAT_EQ(u.depthA, 0.0f);
    EXPECT_FLOAT_EQ(u.depthB, 0.0f);
}

// =============================================================================
// VolumetricFogUniforms - matches volumetric_fog.frag.hlsl
// =============================================================================

TEST(VolumetricFogUniformsLayout, SizeIs448Bytes) {
    // 5 mat4 (320) + 2 vec4 (32) + 4 * vec3+float (64) + 8 floats (32) = 448
    EXPECT_EQ(sizeof(VolumetricFogUniforms), 448u);
}

TEST(VolumetricFogUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(VolumetricFogUniforms, invViewProjection), 0u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, shadowLightViewProjection), 64u);
    // 4 mat4 = 256 bytes
    EXPECT_EQ(offsetof(VolumetricFogUniforms, cascadeSplits), 320u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, shadowBias), 336u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, lightDir), 352u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, fogDensity), 364u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, lightColor), 368u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, scatterStrength), 380u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, fogColor), 384u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, fogHeight), 396u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, cameraPos), 400u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, fogFalloff), 412u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, startDist), 416u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, maxDist), 420u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, godRaysEnabled), 424u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, densityMultiplier), 428u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, shadowAtlasTexel), 432u);
    EXPECT_EQ(offsetof(VolumetricFogUniforms, lastActiveSplit), 436u);
}

TEST(VolumetricFogUniformsLayout, DefaultValues) {
    VolumetricFogUniforms u;
    EXPECT_FLOAT_EQ(u.fogDensity, 0.0015f);
    EXPECT_FLOAT_EQ(u.scatterStrength, 0.25f);
    EXPECT_FLOAT_EQ(u.fogHeight, 100.0f);
    EXPECT_FLOAT_EQ(u.fogFalloff, 0.005f);
    EXPECT_FLOAT_EQ(u.startDist, 100.0f);
    EXPECT_FLOAT_EQ(u.maxDist, 3000.0f);
    EXPECT_FLOAT_EQ(u.godRaysEnabled, 1.0f);
}

// =============================================================================
// ShadowDataUniforms - cascade shadow maps
// =============================================================================

TEST(ShadowDataUniformsLayout, SizeIs336Bytes) {
    // 4 mat4 (256) + 3 vec4 (48) + 8 floats (32) = 336
    EXPECT_EQ(sizeof(ShadowDataUniforms), 336u);
}

TEST(ShadowDataUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(ShadowDataUniforms, lightViewProjection), 0u);
    // 4 mat4 = 256 bytes
    EXPECT_EQ(offsetof(ShadowDataUniforms, cascadeSplits), 256u);
    EXPECT_EQ(offsetof(ShadowDataUniforms, cascadeTexelWorld), 272u);
    EXPECT_EQ(offsetof(ShadowDataUniforms, cascadeDepthRange), 288u);
    EXPECT_EQ(offsetof(ShadowDataUniforms, shadowMapResolution), 304u);
    EXPECT_EQ(offsetof(ShadowDataUniforms, lightSize), 308u);
    EXPECT_EQ(offsetof(ShadowDataUniforms, shadowEnabled), 312u);
    EXPECT_EQ(offsetof(ShadowDataUniforms, lastActiveSplit), 316u);
    EXPECT_EQ(offsetof(ShadowDataUniforms, fadeStart), 320u);
    EXPECT_EQ(offsetof(ShadowDataUniforms, fadeEnd), 324u);
}

// =============================================================================
// GTAOUniforms
// =============================================================================

TEST(GTAOUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(GTAOUniforms, projection), 0u);
    EXPECT_EQ(offsetof(GTAOUniforms, invProjection), 64u);
    EXPECT_EQ(offsetof(GTAOUniforms, screenSize), 128u);
    EXPECT_EQ(offsetof(GTAOUniforms, invScreenSize), 136u);
    EXPECT_EQ(offsetof(GTAOUniforms, radius), 144u);
    EXPECT_EQ(offsetof(GTAOUniforms, bias), 148u);
    EXPECT_EQ(offsetof(GTAOUniforms, numDirections), 152u);
    EXPECT_EQ(offsetof(GTAOUniforms, numSteps), 156u);
    EXPECT_EQ(offsetof(GTAOUniforms, radiusScale), 160u);
    EXPECT_EQ(offsetof(GTAOUniforms, maxDistance), 164u);
    EXPECT_EQ(offsetof(GTAOUniforms, sampleKernel), 176u);
}

TEST(GTAOUniformsLayout, SizeIs560Bytes) {
    // 2 mat4 (128) + 2 vec2 (16) + 8 scalars (32) + 24 vec4 kernel (384) = 560
    EXPECT_EQ(sizeof(GTAOUniforms), 560u);
}

TEST(GTAOUniformsLayout, DefaultValues) {
    GTAOUniforms u;
    EXPECT_FLOAT_EQ(u.radius, 4.0f);
    EXPECT_FLOAT_EQ(u.radiusScale, 1.0f);
    EXPECT_EQ(u.numDirections, 6);
    EXPECT_EQ(u.numSteps, 3);
    EXPECT_FLOAT_EQ(u.maxDistance, 12000.0f);
    // The blur pass must agree with the AO passes on the far cutoff.
    EXPECT_FLOAT_EQ(u.maxDistance, BlurUniforms{}.maxDistance);
}

// =============================================================================
// BlurUniforms
// =============================================================================

TEST(BlurUniformsLayout, SizeIs32Bytes) {
    EXPECT_EQ(sizeof(BlurUniforms), 32u);
}

TEST(BlurUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(BlurUniforms, direction), 0u);
    EXPECT_EQ(offsetof(BlurUniforms, invScreenSize), 8u);
    EXPECT_EQ(offsetof(BlurUniforms, sharpness), 16u);
    EXPECT_EQ(offsetof(BlurUniforms, depthA), 20u);
    EXPECT_EQ(offsetof(BlurUniforms, depthB), 24u);
    EXPECT_EQ(offsetof(BlurUniforms, maxDistance), 28u);
}

// =============================================================================
// FxaaUniforms
// =============================================================================

TEST(FxaaUniformsLayout, SizeIs16Bytes) {
    EXPECT_EQ(sizeof(FxaaUniforms), 16u);
}

TEST(FxaaUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(FxaaUniforms, invScreenSize), 0u);
}

// =============================================================================
// ParticleFragmentUniforms - matches particle.frag.hlsl
// =============================================================================

TEST(ParticleFragmentUniformsLayout, SizeIs16Bytes) {
    // 4 scalars (16); per-particle tint/opacity live in ParticleInstanceData
    EXPECT_EQ(sizeof(ParticleFragmentUniforms), 16u);
}

TEST(ParticleFragmentUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(ParticleFragmentUniforms, emissiveBoost), 0u);
}

TEST(ParticleFragmentUniformsLayout, DefaultValues) {
    ParticleFragmentUniforms u;
    EXPECT_FLOAT_EQ(u.emissiveBoost, 1.0f);
}

// =============================================================================
// ParticleBatchUniforms - matches particle.vert.hlsl BatchUniforms
// =============================================================================

TEST(ParticleBatchUniformsLayout, SizeIs16Bytes) {
    EXPECT_EQ(sizeof(ParticleBatchUniforms), 16u);
}

TEST(ParticleBatchUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(ParticleBatchUniforms, instanceOffset), 0u);
}

// =============================================================================
// ParticleInstanceData - matches particle.vert.hlsl ParticleInstance (storage
// buffer record; tight std430-compatible packing, no cbuffer padding rules)
// =============================================================================

TEST(ParticleInstanceDataLayout, SizeIs80Bytes) {
    // mat4 (64) + vec4 (16) = 80
    EXPECT_EQ(sizeof(ParticleInstanceData), 80u);
}

TEST(ParticleInstanceDataLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(ParticleInstanceData, model), 0u);
    EXPECT_EQ(offsetof(ParticleInstanceData, color), 64u);
}

// =============================================================================
// BloomDownsampleUniforms - matches bloom_downsample.frag.hlsl
// =============================================================================

TEST(BloomDownsampleUniformsLayout, SizeIs32Bytes) {
    // vec2 + threshold + isFirstPass (16) + exposure + pad[3] (16) = 32
    EXPECT_EQ(sizeof(BloomDownsampleUniforms), 32u);
}

TEST(BloomDownsampleUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(BloomDownsampleUniforms, srcTexelSize), 0u);
    EXPECT_EQ(offsetof(BloomDownsampleUniforms, threshold), 8u);
    EXPECT_EQ(offsetof(BloomDownsampleUniforms, isFirstPass), 12u);
    EXPECT_EQ(offsetof(BloomDownsampleUniforms, exposure), 16u);
}

TEST(BloomDownsampleUniformsLayout, DefaultValues) {
    BloomDownsampleUniforms u;
    // exposure defaults to neutral: post-exposure thresholding is then
    // byte-identical to the old pre-exposure behavior.
    EXPECT_FLOAT_EQ(u.exposure, 1.0f);
    EXPECT_FLOAT_EQ(u.isFirstPass, 1.0f);
}
