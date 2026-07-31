#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/render/environment.hpp"
#include <cstddef>
#include <gtest/gtest.h>

using namespace bracken::gpu;

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
    EXPECT_EQ(alignof(GtaoTemporalUniforms), 16u);
    EXPECT_EQ(alignof(GtaoUpsampleUniforms), 16u);
    EXPECT_EQ(alignof(BlurUniforms), 16u);
    EXPECT_EQ(alignof(CompositeUniforms), 16u);
    EXPECT_EQ(alignof(FxaaUniforms), 16u);
    EXPECT_EQ(alignof(ParticleFragmentUniforms), 16u);
    EXPECT_EQ(alignof(ParticleBatchUniforms), 16u);
    EXPECT_EQ(alignof(BloomDownsampleUniforms), 16u);
    EXPECT_EQ(alignof(BloomUpsampleUniforms), 16u);
    EXPECT_EQ(alignof(VolumetricFogUniforms), 16u);
    EXPECT_EQ(alignof(PrepassModelUniforms), 16u);
    EXPECT_EQ(alignof(PrepassCameraUniforms), 16u);
    EXPECT_EQ(alignof(TaaResolveUniforms), 16u);
    EXPECT_EQ(alignof(TaaSharpenUniforms), 16u);
    EXPECT_EQ(alignof(SkyCaptureUniforms), 16u);
    EXPECT_EQ(alignof(IblFilterUniforms), 16u);
    EXPECT_EQ(alignof(ExposureHistogramUniforms), 16u);
    EXPECT_EQ(alignof(ExposureAdaptUniforms), 16u);
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
    EXPECT_EQ(sizeof(GtaoTemporalUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(GtaoUpsampleUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(BlurUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(CompositeUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(FxaaUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(ParticleFragmentUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(ParticleBatchUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(BloomDownsampleUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(BloomUpsampleUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(VolumetricFogUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(PrepassModelUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(PrepassCameraUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(TaaResolveUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(TaaSharpenUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(SkyCaptureUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(IblFilterUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(ExposureHistogramUniforms) % 16, 0u);
    EXPECT_EQ(sizeof(ExposureAdaptUniforms) % 16, 0u);
}

// =============================================================================
// ModelTransformUniforms - matches model.vert.hlsl TransformUniforms
// =============================================================================

TEST(ModelTransformUniformsLayout, SizeIs288Bytes) {
    // 3 mat4 (192) + vec3+pad (16) + mat4 (64) + uint+pad[3] (16) = 288
    EXPECT_EQ(sizeof(ModelTransformUniforms), 288u);
}

TEST(ModelTransformUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(ModelTransformUniforms, model), 0u);
    EXPECT_EQ(offsetof(ModelTransformUniforms, view), 64u);
    EXPECT_EQ(offsetof(ModelTransformUniforms, projection), 128u);
    EXPECT_EQ(offsetof(ModelTransformUniforms, cameraPos), 192u);
    EXPECT_EQ(offsetof(ModelTransformUniforms, normalMatrix), 208u);
    EXPECT_EQ(offsetof(ModelTransformUniforms, paletteOffset), 272u);
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

TEST(ModelLightingUniformsLayout, SizeIs224Bytes) {
    // 4 vec3+pad (64) + 2 vec4 (32) + 4 vec3+float (64) + 16 scalars (64) = 224
    EXPECT_EQ(sizeof(ModelLightingUniforms), 224u);
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
    EXPECT_EQ(offsetof(ModelLightingUniforms, screenAOTexelX), 208u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, screenAOTexelY), 212u);
    EXPECT_EQ(offsetof(ModelLightingUniforms, iblEnabled), 216u);
}

// =============================================================================
// InstancedLightingUniforms - matches model_instanced.frag.hlsl LightingUniforms
// =============================================================================

TEST(InstancedLightingUniformsLayout, SizeIs208Bytes) {
    // ModelLightingUniforms layout minus the vec4 tintColor (tint is per-instance)
    EXPECT_EQ(sizeof(InstancedLightingUniforms), 208u);
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
    EXPECT_EQ(offsetof(InstancedLightingUniforms, screenAOTexelX), 192u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, screenAOTexelY), 196u);
    EXPECT_EQ(offsetof(InstancedLightingUniforms, iblEnabled), 200u);
}

// =============================================================================
// SkyboxFragmentUniforms - matches skybox.frag.hlsl SkyUniforms
// =============================================================================

TEST(SkyboxFragmentUniformsLayout, SizeIs192Bytes) {
    // mat4 (64) + 7 * (vec3 + float) rows (112) + cloud/haze row (16) = 192
    EXPECT_EQ(sizeof(SkyboxFragmentUniforms), 192u);
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
    EXPECT_EQ(offsetof(SkyboxFragmentUniforms, cloudCoverage), 176u);
    EXPECT_EQ(offsetof(SkyboxFragmentUniforms, cloudDensity), 180u);
    EXPECT_EQ(offsetof(SkyboxFragmentUniforms, cloudScale), 184u);
    EXPECT_EQ(offsetof(SkyboxFragmentUniforms, hazeStrength), 188u);
}

TEST(SkyboxFragmentUniformsLayout, CloudDefaultsMatchEnvironmentState) {
    // The uniform defaults exist only as a safety net; the real source of
    // truth is EnvironmentState — keep them in lockstep.
    const SkyboxFragmentUniforms u{};
    const bracken::render::EnvironmentState env{};
    EXPECT_EQ(u.cloudCoverage, env.cloud_coverage);
    EXPECT_EQ(u.cloudDensity, env.cloud_density);
    EXPECT_EQ(u.cloudScale, env.cloud_scale);
    EXPECT_EQ(u.hazeStrength, env.haze_strength);
}

// =============================================================================
// SkyCaptureUniforms - matches sky_capture.frag.hlsl SkyCaptureUniforms
// =============================================================================

TEST(SkyCaptureUniformsLayout, SizeIs176Bytes) {
    // 10 * (vec3 + float) rows (3 face-basis + 7 atmosphere) + cloud/haze
    // row = 176
    EXPECT_EQ(sizeof(SkyCaptureUniforms), 176u);
}

TEST(SkyCaptureUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(SkyCaptureUniforms, faceForward), 0u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, faceRight), 16u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, faceUp), 32u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, sunDirection), 48u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, time), 60u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, sunColor), 64u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, zenithColor), 80u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, horizonColor), 96u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, groundColor), 112u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, fogHorizonColor), 128u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, fogSunColor), 144u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, cloudCoverage), 160u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, cloudDensity), 164u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, cloudScale), 168u);
    EXPECT_EQ(offsetof(SkyCaptureUniforms, hazeStrength), 172u);
}

// =============================================================================
// IblFilterUniforms - matches ibl_prefilter.frag.hlsl / ibl_irradiance.frag.hlsl
// =============================================================================

TEST(IblFilterUniformsLayout, SizeIs48Bytes) {
    // 3 * (vec3 + float) rows = 48
    EXPECT_EQ(sizeof(IblFilterUniforms), 48u);
}

TEST(IblFilterUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(IblFilterUniforms, faceForward), 0u);
    EXPECT_EQ(offsetof(IblFilterUniforms, roughness), 12u);
    EXPECT_EQ(offsetof(IblFilterUniforms, faceRight), 16u);
    EXPECT_EQ(offsetof(IblFilterUniforms, sourceSize), 28u);
    EXPECT_EQ(offsetof(IblFilterUniforms, faceUp), 32u);
}

// =============================================================================
// CompositeUniforms - matches composite.frag.hlsl
// =============================================================================

TEST(CompositeUniformsLayout, SizeIs112Bytes) {
    // 3 pre-P6 16-byte blocks (bloom/fog/exposure/tonemap, contrast/
    // saturation/lowResTexelSize, depthA/depthB/autoExposure/wbEnabled) +
    // 4 grading rows (whiteBalance+cdlEnabled, slope, offset, power) = 112
    EXPECT_EQ(sizeof(CompositeUniforms), 112u);
}

TEST(CompositeUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(CompositeUniforms, bloomStrength), 0u);
    EXPECT_EQ(offsetof(CompositeUniforms, volumetricFogEnabled), 4u);
    EXPECT_EQ(offsetof(CompositeUniforms, exposure), 8u);
    EXPECT_EQ(offsetof(CompositeUniforms, tonemapMode), 12u);
    EXPECT_EQ(offsetof(CompositeUniforms, contrast), 16u);
    EXPECT_EQ(offsetof(CompositeUniforms, saturation), 20u);
    EXPECT_EQ(offsetof(CompositeUniforms, lowResTexelSize), 24u);
    EXPECT_EQ(offsetof(CompositeUniforms, depthA), 32u);
    EXPECT_EQ(offsetof(CompositeUniforms, depthB), 36u);
    EXPECT_EQ(offsetof(CompositeUniforms, autoExposureEnabled), 40u);
    EXPECT_EQ(offsetof(CompositeUniforms, whiteBalanceEnabled), 44u);
    EXPECT_EQ(offsetof(CompositeUniforms, whiteBalance), 48u);
    EXPECT_EQ(offsetof(CompositeUniforms, cdlEnabled), 60u);
    EXPECT_EQ(offsetof(CompositeUniforms, cdlSlope), 64u);
    EXPECT_EQ(offsetof(CompositeUniforms, cdlOffset), 80u);
    EXPECT_EQ(offsetof(CompositeUniforms, cdlPower), 96u);
}

TEST(CompositeUniformsLayout, DefaultValues) {
    CompositeUniforms u;
    // Must track GraphicsSettings::bloom_strength (agreement is pinned in
    // post_process_stack.test.cpp).
    EXPECT_FLOAT_EQ(u.bloomStrength, 0.25f);
    EXPECT_FLOAT_EQ(u.volumetricFogEnabled, 0.0f);
    EXPECT_FLOAT_EQ(u.exposure, 1.0f);
    EXPECT_EQ(u.tonemapMode, 0);
    EXPECT_FLOAT_EQ(u.contrast, 1.0f);
    EXPECT_FLOAT_EQ(u.saturation, 1.0f);
    // depthA/depthB of 0 degrade to plain bilinear weights in the shader.
    EXPECT_FLOAT_EQ(u.depthA, 0.0f);
    EXPECT_FLOAT_EQ(u.depthB, 0.0f);
    // Exposure/grading defaults must be NEUTRAL (flags off, identity grade):
    // the ungraded path is pixel-identical to pre-P6 output.
    EXPECT_FLOAT_EQ(u.autoExposureEnabled, 0.0f);
    EXPECT_FLOAT_EQ(u.whiteBalanceEnabled, 0.0f);
    EXPECT_EQ(u.whiteBalance, glm::vec3(1.0f));
    EXPECT_FLOAT_EQ(u.cdlEnabled, 0.0f);
    EXPECT_EQ(u.cdlSlope, glm::vec3(1.0f));
    EXPECT_EQ(u.cdlOffset, glm::vec3(0.0f));
    EXPECT_EQ(u.cdlPower, glm::vec3(1.0f));
}

// =============================================================================
// Exposure compute uniforms - match exposure_histogram.comp.hlsl /
// exposure_adapt.comp.hlsl (compute set 2, b0)
// =============================================================================

TEST(ExposureHistogramUniformsLayout, SizeAndOffsets) {
    // uint2 extent + minLog2Luma + invLog2Range = one 16-byte row
    EXPECT_EQ(sizeof(ExposureHistogramUniforms), 16u);
    EXPECT_EQ(offsetof(ExposureHistogramUniforms, width), 0u);
    EXPECT_EQ(offsetof(ExposureHistogramUniforms, height), 4u);
    EXPECT_EQ(offsetof(ExposureHistogramUniforms, minLog2Luma), 8u);
    EXPECT_EQ(offsetof(ExposureHistogramUniforms, invLog2Range), 12u);
}

TEST(ExposureAdaptUniformsLayout, SizeAndOffsets) {
    // 12 floats = 3 16-byte rows
    EXPECT_EQ(sizeof(ExposureAdaptUniforms), 48u);
    EXPECT_EQ(offsetof(ExposureAdaptUniforms, minLog2Luma), 0u);
    EXPECT_EQ(offsetof(ExposureAdaptUniforms, log2Range), 4u);
    EXPECT_EQ(offsetof(ExposureAdaptUniforms, deltaTime), 8u);
    EXPECT_EQ(offsetof(ExposureAdaptUniforms, firstFrame), 12u);
    EXPECT_EQ(offsetof(ExposureAdaptUniforms, speedUp), 16u);
    EXPECT_EQ(offsetof(ExposureAdaptUniforms, speedDown), 20u);
    EXPECT_EQ(offsetof(ExposureAdaptUniforms, minEv), 24u);
    EXPECT_EQ(offsetof(ExposureAdaptUniforms, maxEv), 28u);
    EXPECT_EQ(offsetof(ExposureAdaptUniforms, lowPercentile), 32u);
    EXPECT_EQ(offsetof(ExposureAdaptUniforms, highPercentile), 36u);
    EXPECT_EQ(offsetof(ExposureAdaptUniforms, log2TargetGray), 40u);
}

TEST(ExposureAdaptUniformsLayout, DefaultsTrackSettingsAndConstants) {
    const ExposureAdaptUniforms u;
    // Speeds/clamps mirror the GraphicsSettings defaults (the frame path
    // overwrites them from settings each frame); percentile/range defaults
    // mirror render/auto_exposure.hpp.
    EXPECT_FLOAT_EQ(u.minLog2Luma, -10.0f);
    EXPECT_FLOAT_EQ(u.log2Range, 26.0f);
    EXPECT_FLOAT_EQ(u.speedUp, 1.5f);
    EXPECT_FLOAT_EQ(u.speedDown, 2.5f);
    EXPECT_FLOAT_EQ(u.minEv, -4.0f);
    EXPECT_FLOAT_EQ(u.maxEv, 4.0f);
    EXPECT_FLOAT_EQ(u.lowPercentile, 0.60f);
    EXPECT_FLOAT_EQ(u.highPercentile, 0.05f);
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
    EXPECT_EQ(offsetof(GTAOUniforms, radiusScale), 148u);
    EXPECT_EQ(offsetof(GTAOUniforms, falloffRange), 152u);
    EXPECT_EQ(offsetof(GTAOUniforms, thinOccluderCompensation), 156u);
    EXPECT_EQ(offsetof(GTAOUniforms, temporalRotation), 160u);
    EXPECT_EQ(offsetof(GTAOUniforms, temporalOffset), 164u);
    EXPECT_EQ(offsetof(GTAOUniforms, maxDistance), 168u);
}

TEST(GTAOUniformsLayout, SizeIs176Bytes) {
    // 2 mat4 (128) + 2 vec2 (16) + 8 scalars (32) = 176
    EXPECT_EQ(sizeof(GTAOUniforms), 176u);
}

TEST(GTAOUniformsLayout, DefaultValues) {
    GTAOUniforms u;
    EXPECT_FLOAT_EQ(u.radius, 4.0f);
    EXPECT_FLOAT_EQ(u.radiusScale, 1.0f);
    // Falloff spans a sane fraction of the radius; thin-occluder decay is a
    // blend factor in [0, 1].
    EXPECT_GT(u.falloffRange, 0.0f);
    EXPECT_LE(u.falloffRange, 1.0f);
    EXPECT_GE(u.thinOccluderCompensation, 0.0f);
    EXPECT_LE(u.thinOccluderCompensation, 1.0f);
    EXPECT_FLOAT_EQ(u.maxDistance, 12000.0f);
    // The blur pass must agree with the AO pass on the far cutoff.
    EXPECT_FLOAT_EQ(u.maxDistance, BlurUniforms{}.maxDistance);
}

// =============================================================================
// GtaoTemporalUniforms - matches gtao_temporal.frag.hlsl
// =============================================================================

TEST(GtaoTemporalUniformsLayout, SizeIs32Bytes) {
    EXPECT_EQ(sizeof(GtaoTemporalUniforms), 32u);
}

TEST(GtaoTemporalUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(GtaoTemporalUniforms, invScreenSize), 0u);
    EXPECT_EQ(offsetof(GtaoTemporalUniforms, depthA), 8u);
    EXPECT_EQ(offsetof(GtaoTemporalUniforms, depthB), 12u);
    EXPECT_EQ(offsetof(GtaoTemporalUniforms, historyValid), 16u);
    EXPECT_EQ(offsetof(GtaoTemporalUniforms, feedbackMax), 20u);
    EXPECT_EQ(offsetof(GtaoTemporalUniforms, depthRejectScale), 24u);
}

TEST(GtaoTemporalUniformsLayout, DefaultValues) {
    GtaoTemporalUniforms u;
    // History invalid until the CPU says otherwise (first frame outputs the
    // current AO unblended). Tuning constants are NSDMI live values.
    EXPECT_FLOAT_EQ(u.historyValid, 0.0f);
    EXPECT_FLOAT_EQ(u.feedbackMax, 0.9f);
    EXPECT_GT(u.depthRejectScale, 0.0f);
}

// =============================================================================
// GtaoUpsampleUniforms - matches gtao_upsample.frag.hlsl
// =============================================================================

TEST(GtaoUpsampleUniformsLayout, SizeIs32Bytes) {
    EXPECT_EQ(sizeof(GtaoUpsampleUniforms), 32u);
}

TEST(GtaoUpsampleUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(GtaoUpsampleUniforms, lowResTexelSize), 0u);
    EXPECT_EQ(offsetof(GtaoUpsampleUniforms, depthA), 8u);
    EXPECT_EQ(offsetof(GtaoUpsampleUniforms, depthB), 12u);
    EXPECT_EQ(offsetof(GtaoUpsampleUniforms, strength), 16u);
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

// =============================================================================
// PrepassModelUniforms - matches prepass_model.vert.hlsl /
// prepass_skinned.vert.hlsl PrepassTransformUniforms
// =============================================================================

TEST(PrepassModelUniformsLayout, SizeIs400Bytes) {
    // 6 mat4 (384) + 2 uint + pad[2] (16) = 400
    EXPECT_EQ(sizeof(PrepassModelUniforms), 400u);
}

TEST(PrepassModelUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(PrepassModelUniforms, model), 0u);
    EXPECT_EQ(offsetof(PrepassModelUniforms, view), 64u);
    EXPECT_EQ(offsetof(PrepassModelUniforms, projection), 128u);
    EXPECT_EQ(offsetof(PrepassModelUniforms, prevModel), 192u);
    EXPECT_EQ(offsetof(PrepassModelUniforms, currViewProjectionNoJitter), 256u);
    EXPECT_EQ(offsetof(PrepassModelUniforms, prevViewProjectionNoJitter), 320u);
    EXPECT_EQ(offsetof(PrepassModelUniforms, paletteOffset), 384u);
    EXPECT_EQ(offsetof(PrepassModelUniforms, prevPaletteOffset), 388u);
}

TEST(PrepassModelUniformsLayout, DefaultPrevPaletteIsSentinel) {
    PrepassModelUniforms u;
    EXPECT_EQ(u.prevPaletteOffset, PrepassModelUniforms::NO_PREV_PALETTE);
}

// =============================================================================
// PrepassCameraUniforms - matches prepass_instanced.vert.hlsl /
// prepass_terrain.vert.hlsl PrepassCameraUniforms
// =============================================================================

TEST(PrepassCameraUniformsLayout, SizeIs256Bytes) {
    EXPECT_EQ(sizeof(PrepassCameraUniforms), 256u);
}

TEST(PrepassCameraUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(PrepassCameraUniforms, view), 0u);
    EXPECT_EQ(offsetof(PrepassCameraUniforms, projection), 64u);
    EXPECT_EQ(offsetof(PrepassCameraUniforms, currViewProjectionNoJitter), 128u);
    EXPECT_EQ(offsetof(PrepassCameraUniforms, prevViewProjectionNoJitter), 192u);
}

// =============================================================================
// InstanceData - matches model_instanced.vert.hlsl + prepass_instanced.vert.hlsl
// =============================================================================

TEST(InstanceDataLayout, SizeIs224Bytes) {
    // 2 mat4 (128) + vec4 (16) + noFog+pad[3] (16) + prevModel mat4 (64) = 224
    EXPECT_EQ(sizeof(InstanceData), 224u);
}

TEST(InstanceDataLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(InstanceData, model), 0u);
    EXPECT_EQ(offsetof(InstanceData, normalMatrix), 64u);
    EXPECT_EQ(offsetof(InstanceData, tint), 128u);
    EXPECT_EQ(offsetof(InstanceData, noFog), 144u);
    EXPECT_EQ(offsetof(InstanceData, prevModel), 160u);
}

// =============================================================================
// TaaResolveUniforms - matches taa_resolve.frag.hlsl
// =============================================================================

TEST(TaaResolveUniformsLayout, SizeIs176Bytes) {
    // 2 mat4 (128) + vec2+vec2 (16) + vec3+float (16) + 4 floats (16) = 176
    EXPECT_EQ(sizeof(TaaResolveUniforms), 176u);
}

TEST(TaaResolveUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(TaaResolveUniforms, currInvViewProjUnjittered), 0u);
    EXPECT_EQ(offsetof(TaaResolveUniforms, prevViewProjUnjittered), 64u);
    EXPECT_EQ(offsetof(TaaResolveUniforms, screenSize), 128u);
    EXPECT_EQ(offsetof(TaaResolveUniforms, invScreenSize), 136u);
    EXPECT_EQ(offsetof(TaaResolveUniforms, cameraPos), 144u);
    EXPECT_EQ(offsetof(TaaResolveUniforms, historyValid), 156u);
    EXPECT_EQ(offsetof(TaaResolveUniforms, feedbackMax), 160u);
    EXPECT_EQ(offsetof(TaaResolveUniforms, varianceGamma), 164u);
}

TEST(TaaResolveUniformsLayout, DefaultValues) {
    TaaResolveUniforms u;
    // History invalid until the CPU says otherwise (first frame outputs
    // the current color unblended).
    EXPECT_FLOAT_EQ(u.historyValid, 0.0f);
    // Engine tuning constants — the NSDMI defaults are the live values.
    EXPECT_FLOAT_EQ(u.feedbackMax, 0.9f);
    EXPECT_FLOAT_EQ(u.varianceGamma, 1.0f);
}

// =============================================================================
// TaaSharpenUniforms - matches taa_sharpen.frag.hlsl
// =============================================================================

TEST(TaaSharpenUniformsLayout, SizeIs16Bytes) {
    EXPECT_EQ(sizeof(TaaSharpenUniforms), 16u);
}

TEST(TaaSharpenUniformsLayout, FieldOffsets) {
    EXPECT_EQ(offsetof(TaaSharpenUniforms, invScreenSize), 0u);
    EXPECT_EQ(offsetof(TaaSharpenUniforms, sharpenStrength), 8u);
}
