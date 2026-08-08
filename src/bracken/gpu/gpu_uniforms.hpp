#pragma once

#include "bracken/gpu/hlsl_packing.hpp"
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>

namespace bracken::gpu {

/**
 * Model vertex shader uniforms - matches model.vert.hlsl /
 * skinned_model.vert.hlsl TransformUniforms (set 1, b0).
 * Used by: scene_renderer.cpp
 * paletteOffset indexes the frame's bone-palette SSBO (in matrices); it is
 * only read by skinned_model.vert.hlsl.
 */
struct alignas(16) ModelTransformUniforms {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 cameraPos;
    float _padding0 = 0.0f;
    glm::mat4 normalMatrix;
    uint32_t paletteOffset = 0;
    float _padding1[3] = {0.0f, 0.0f, 0.0f};
};
BRACKEN_HLSL_CBUFFER(ModelTransformUniforms, 288);
BRACKEN_HLSL_FIELD(ModelTransformUniforms, model, 0);
BRACKEN_HLSL_FIELD(ModelTransformUniforms, view, 64);
BRACKEN_HLSL_FIELD(ModelTransformUniforms, projection, 128);
BRACKEN_HLSL_FIELD(ModelTransformUniforms, cameraPos, 192);
BRACKEN_HLSL_FIELD(ModelTransformUniforms, _padding0, 204);
BRACKEN_HLSL_FIELD(ModelTransformUniforms, normalMatrix, 208);
BRACKEN_HLSL_FIELD(ModelTransformUniforms, paletteOffset, 272);
BRACKEN_HLSL_FIELD(ModelTransformUniforms, _padding1, 276);

/**
 * Model fragment shader uniforms - matches model.frag.hlsl LightingUniforms (set 3, b0)
 * Used by: scene_renderer.cpp
 * Field order mirrors the HLSL cbuffer exactly; every vec3 is followed by a
 * scalar so no field straddles a 16-byte register boundary.
 */
struct alignas(16) ModelLightingUniforms {
    glm::vec3 lightDir;
    float _padding0 = 0.0f;
    glm::vec3 lightColor;
    float _padding1 = 0.0f;
    glm::vec3 ambientSky;
    float _padding2 = 0.0f;
    glm::vec3 ambientGround;
    float _padding3 = 0.0f;
    glm::vec4 tintColor;
    glm::vec4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec3 fogColor; // horizon fog color
    float fogStart;
    glm::vec3 fogSunColor;
    float fogEnd;
    glm::vec3 emissiveFactor;
    float fogHeightFalloff = 0.0f;
    glm::vec3 cameraPos;
    float normalScale = 1.0f;
    int hasTexture;
    int fogEnabled;
    int hasNormalMap = 0;
    float ambientStrength = 0.5f;
    float sunIntensity = 1.0f;
    float metallicFactor = 0.0f;
    float roughnessFactor = 0.85f;
    float occlusionStrength = 1.0f;
    float hasMRMap = 0.0f;
    float hasEmissiveMap = 0.0f;
    float hasPackedOcclusion = 0.0f;
    // glTF alphaMode MASK cutoff; 0 disables the alpha test (OPAQUE/BLEND).
    float alphaCutoff = 0.0f;
    // 1 / render-target size: SV_Position.xy * texel = screen UV for the
    // full-res screen-space AO texture (fragment sampler slot 6). The AO
    // factor multiplies the AMBIENT term only, min-combined with material AO.
    float screenAOTexelX = 0.0f;
    float screenAOTexelY = 0.0f;
    // 1 = ambient comes from the sky IBL chain (irradiance/prefiltered/BRDF
    // LUT at fragment samplers 7-9); 0 = hemisphere ambient fallback.
    float iblEnabled = 0.0f;
    // 1 = modulate ambient by the screen-space GTAO texture. 0 for non-OPAQUE
    // meshes: they are excluded from the depth prepass, so the AO texel at
    // their screen position belongs to whatever is BEHIND them — sampling it
    // stamps the background's AO onto foliage ("see-through" canopies).
    float screenAOEnabled = 1.0f;
    // 1 = take the sun shadow from the ray-traced mask instead of the cascades
    // (render/rt_renderer.hpp). Off by default; the CSM path is unchanged.
    float rtShadowEnabled = 0.0f;
    /// The traced-reflection handover window, matching
    /// render::RtReflectionSettings::fade_start / max_roughness. The forward
    /// pass scales its IBL specular lobe by 1 - rtReflectionWeight(...) and
    /// the compute pass writes that same weight into the traced target's
    /// alpha, so the two always sum to one (shaders/reflection_gate.hlsli).
    /// rtReflectionMaxRoughness == 0 is the "reflections off" encoding.
    float rtReflectionMaxRoughness = 0.0f;
    float rtReflectionFadeStart = 0.0f;
    /// `GraphicsSettings::rt_local_shadows` as a float, because a cbuffer has
    /// no enums: 0 = Off, 1 = Ratio, 2 = Full-DI. Rides in what used to be
    /// `_rtPad2`, so the cbuffer neither grew nor moved a field.
    ///
    /// 0 leaves accumulateClusterLights' arithmetic bit-identical — the mask is
    /// not sampled at all, exactly as `giEnabled == 0` skips the probe tap
    /// rather than blending it at weight zero. 1 multiplies the whole
    /// point+spot subtotal by the traced visibility ratio. 2 (R5c) does not
    /// call accumulateClusterLights AT ALL and reads the two demodulated DI
    /// storage textures instead — see shaders/local_di.hlsli.
    float rtLocalShadowMode = 0.0f;
    /// Probe-field GI (render/gi/), R1c. `giCascadeBase[c].xyz` is
    /// `render::gi::cascade_base()` of the position ProbeVolume was scrolled
    /// with — sent rather than re-derived from cameraPos in the shader, so the
    /// two can never disagree about which probe answers for which point in the
    /// world (`.w` is unused; an array cannot carry a named pad). `giEnabled`
    /// gates every tap: 0 restores the pre-R1c ambient term exactly.
    glm::ivec4 giCascadeBase[3] = {};
    float giEnabled = 0.0f;
    /// Screen-probe gather (R4b, render/gi/screen_probes.hpp). These three were
    /// the trailing GI pads, so the cbuffer neither grew nor moved a field.
    /// `giScreenEnabled` gates every tap in `gi_screen_sample.hlsli`; 0 leaves the
    /// pure probe-field ambient of R1c. The grid rides as floats because the
    /// pads were floats, and the shader casts them back.
    float giScreenEnabled = 0.0f;
    float giScreenProbesX = 0.0f;
    float giScreenProbesY = 0.0f;
};
BRACKEN_HLSL_CBUFFER(ModelLightingUniforms, 304);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, lightDir, 0);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, _padding0, 12);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, lightColor, 16);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, _padding1, 28);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, ambientSky, 32);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, _padding2, 44);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, ambientGround, 48);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, _padding3, 60);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, tintColor, 64);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, baseColorFactor, 80);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, fogColor, 96);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, fogStart, 108);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, fogSunColor, 112);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, fogEnd, 124);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, emissiveFactor, 128);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, fogHeightFalloff, 140);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, cameraPos, 144);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, normalScale, 156);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, hasTexture, 160);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, fogEnabled, 164);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, hasNormalMap, 168);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, ambientStrength, 172);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, sunIntensity, 176);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, metallicFactor, 180);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, roughnessFactor, 184);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, occlusionStrength, 188);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, hasMRMap, 192);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, hasEmissiveMap, 196);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, hasPackedOcclusion, 200);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, alphaCutoff, 204);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, screenAOTexelX, 208);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, screenAOTexelY, 212);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, iblEnabled, 216);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, screenAOEnabled, 220);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, rtShadowEnabled, 224);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, rtReflectionMaxRoughness, 228);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, rtReflectionFadeStart, 232);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, rtLocalShadowMode, 236);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, giCascadeBase, 240);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, giEnabled, 288);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, giScreenEnabled, 292);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, giScreenProbesX, 296);
BRACKEN_HLSL_FIELD(ModelLightingUniforms, giScreenProbesY, 300);

/**
 * Skybox fragment shader uniforms - matches skybox.frag.hlsl SkyUniforms (set 3, b0)
 * Colors are linear, sourced from EnvironmentState (render/environment.hpp).
 */
struct alignas(16) SkyboxFragmentUniforms {
    glm::mat4 invViewProjection;
    glm::vec3 sunDirection; // direction TO the sun
    float time;             // cloud scroll phase (sky time pre-scaled by cloud_speed)
    glm::vec3 sunColor;
    float _padding0 = 0.0f;
    glm::vec3 zenithColor;
    float _padding1 = 0.0f;
    glm::vec3 horizonColor;
    float _padding2 = 0.0f;
    glm::vec3 groundColor;
    float _padding3 = 0.0f;
    glm::vec3 fogHorizonColor;
    float _padding4 = 0.0f;
    glm::vec3 fogSunColor;
    float _padding5 = 0.0f;
    // Cloud/haze shaping (EnvironmentState cloud_* / haze_strength).
    float cloudCoverage = 0.35f;
    float cloudDensity = 1.0f;
    float cloudScale = 1.0f;
    float hazeStrength = 1.0f;
};
BRACKEN_HLSL_CBUFFER(SkyboxFragmentUniforms, 192);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, invViewProjection, 0);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, sunDirection, 64);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, time, 76);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, sunColor, 80);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, _padding0, 92);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, zenithColor, 96);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, _padding1, 108);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, horizonColor, 112);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, _padding2, 124);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, groundColor, 128);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, _padding3, 140);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, fogHorizonColor, 144);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, _padding4, 156);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, fogSunColor, 160);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, _padding5, 172);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, cloudCoverage, 176);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, cloudDensity, 180);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, cloudScale, 184);
BRACKEN_HLSL_FIELD(SkyboxFragmentUniforms, hazeStrength, 188);

/**
 * IBL sky-capture pass fragment uniforms - matches sky_capture.frag.hlsl
 * (set 3, b0). Face basis: with u,v the target UV (v down),
 * dir = normalize(faceForward + (2u-1)*faceRight + (2v-1)*faceUp) — CPU
 * reference render::ibl_face_direction (render/ibl.hpp). The atmosphere
 * fields mirror SkyboxFragmentUniforms; `time` is the sky-time snapshot taken
 * when the prefilter chain (re)started, so an amortized rebuild renders every
 * face with one consistent cloud state.
 */
struct alignas(16) SkyCaptureUniforms {
    glm::vec3 faceForward;
    float _pad0 = 0.0f;
    glm::vec3 faceRight;
    float _pad1 = 0.0f;
    glm::vec3 faceUp;
    float _pad2 = 0.0f;
    glm::vec3 sunDirection; // direction TO the sun
    float time = 0.0f;
    glm::vec3 sunColor;
    float _pad3 = 0.0f;
    glm::vec3 zenithColor;
    float _pad4 = 0.0f;
    glm::vec3 horizonColor;
    float _pad5 = 0.0f;
    glm::vec3 groundColor;
    float _pad6 = 0.0f;
    glm::vec3 fogHorizonColor;
    float _pad7 = 0.0f;
    glm::vec3 fogSunColor;
    float _pad8 = 0.0f;
    // Cloud/haze shaping snapshot (IblSkyParams cloud_* / haze_strength).
    float cloudCoverage = 0.35f;
    float cloudDensity = 1.0f;
    float cloudScale = 1.0f;
    float hazeStrength = 1.0f;
};
BRACKEN_HLSL_CBUFFER(SkyCaptureUniforms, 176);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, faceForward, 0);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, _pad0, 12);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, faceRight, 16);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, _pad1, 28);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, faceUp, 32);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, _pad2, 44);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, sunDirection, 48);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, time, 60);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, sunColor, 64);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, _pad3, 76);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, zenithColor, 80);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, _pad4, 92);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, horizonColor, 96);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, _pad5, 108);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, groundColor, 112);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, _pad6, 124);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, fogHorizonColor, 128);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, _pad7, 140);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, fogSunColor, 144);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, _pad8, 156);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, cloudCoverage, 160);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, cloudDensity, 164);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, cloudScale, 168);
BRACKEN_HLSL_FIELD(SkyCaptureUniforms, hazeStrength, 172);

/**
 * IBL prefilter pass fragment uniforms - matches ibl_prefilter.frag.hlsl AND
 * ibl_irradiance.frag.hlsl (set 3, b0; the irradiance pass ignores
 * roughness). Same face-basis convention as SkyCaptureUniforms. sourceSize is
 * the mip-0 face size of the sampled capture cube (solid-angle mip bias).
 */
struct alignas(16) IblFilterUniforms {
    glm::vec3 faceForward;
    float roughness = 0.0f;
    glm::vec3 faceRight;
    float sourceSize = 64.0f;
    glm::vec3 faceUp;
    float _pad0 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(IblFilterUniforms, 48);
BRACKEN_HLSL_FIELD(IblFilterUniforms, faceForward, 0);
BRACKEN_HLSL_FIELD(IblFilterUniforms, roughness, 12);
BRACKEN_HLSL_FIELD(IblFilterUniforms, faceRight, 16);
BRACKEN_HLSL_FIELD(IblFilterUniforms, sourceSize, 28);
BRACKEN_HLSL_FIELD(IblFilterUniforms, faceUp, 32);
BRACKEN_HLSL_FIELD(IblFilterUniforms, _pad0, 44);

/**
 * Grid vertex shader uniforms (set 1, b0)
 */
struct alignas(16) GridVertexUniforms {
    glm::mat4 viewProjection;
};
BRACKEN_HLSL_CBUFFER(GridVertexUniforms, 64);
BRACKEN_HLSL_FIELD(GridVertexUniforms, viewProjection, 0);

/**
 * UI vertex shader uniforms (set 1, b0)
 */
struct alignas(16) UIScreenUniforms {
    float width{};
    float height{};
    float _padding[2] = {0.0f, 0.0f};
};
BRACKEN_HLSL_CBUFFER(UIScreenUniforms, 16);
BRACKEN_HLSL_FIELD(UIScreenUniforms, width, 0);
BRACKEN_HLSL_FIELD(UIScreenUniforms, height, 4);
BRACKEN_HLSL_FIELD(UIScreenUniforms, _padding, 8);

/**
 * UI fragment shader uniforms (set 3, b0)
 * Padding is three named scalars to mirror ui.frag.hlsl byte-for-byte (an
 * array would stride at 16 bytes on the HLSL side).
 */
struct alignas(16) UIFragmentUniforms {
    int hasTexture{};
    int _padding0 = 0;
    int _padding1 = 0;
    int _padding2 = 0;
};
BRACKEN_HLSL_CBUFFER(UIFragmentUniforms, 16);
BRACKEN_HLSL_FIELD(UIFragmentUniforms, hasTexture, 0);
BRACKEN_HLSL_FIELD(UIFragmentUniforms, _padding0, 4);
BRACKEN_HLSL_FIELD(UIFragmentUniforms, _padding1, 8);
BRACKEN_HLSL_FIELD(UIFragmentUniforms, _padding2, 12);

/**
 * Instanced model vertex shader camera uniforms - matches the CameraUniforms
 * cbuffer of model_instanced.vert.hlsl and particle.vert.hlsl (set 1, b0)
 */
struct alignas(16) InstancedCameraUniforms {
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 cameraPos;
    float _padding0 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(InstancedCameraUniforms, 144);
BRACKEN_HLSL_FIELD(InstancedCameraUniforms, view, 0);
BRACKEN_HLSL_FIELD(InstancedCameraUniforms, projection, 64);
BRACKEN_HLSL_FIELD(InstancedCameraUniforms, cameraPos, 128);
BRACKEN_HLSL_FIELD(InstancedCameraUniforms, _padding0, 140);

/**
 * Instanced model fragment shader uniforms - matches model_instanced.frag.hlsl (set 3, b0)
 * Same layout as ModelLightingUniforms minus tintColor (tint is per-instance).
 */
struct alignas(16) InstancedLightingUniforms {
    glm::vec3 lightDir;
    float _padding0 = 0.0f;
    glm::vec3 lightColor;
    float _padding1 = 0.0f;
    glm::vec3 ambientSky;
    float _padding2 = 0.0f;
    glm::vec3 ambientGround;
    float _padding3 = 0.0f;
    glm::vec4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec3 fogColor; // horizon fog color
    float fogStart;
    glm::vec3 fogSunColor;
    float fogEnd;
    glm::vec3 emissiveFactor;
    float fogHeightFalloff = 0.0f;
    glm::vec3 cameraPos;
    float normalScale = 1.0f;
    int hasTexture;
    int fogEnabled;
    int hasNormalMap = 0;
    float ambientStrength = 0.5f;
    float sunIntensity = 1.0f;
    float metallicFactor = 0.0f;
    float roughnessFactor = 0.85f;
    float occlusionStrength = 1.0f;
    float hasMRMap = 0.0f;
    float hasEmissiveMap = 0.0f;
    float hasPackedOcclusion = 0.0f;
    // glTF alphaMode MASK cutoff; 0 disables the alpha test (OPAQUE/BLEND).
    float alphaCutoff = 0.0f;
    // 1 / render-target size — see ModelLightingUniforms::screenAOTexelX.
    float screenAOTexelX = 0.0f;
    float screenAOTexelY = 0.0f;
    // Sky IBL flag — see ModelLightingUniforms::iblEnabled.
    float iblEnabled = 0.0f;
    // Screen-AO gate — see ModelLightingUniforms::screenAOEnabled.
    float screenAOEnabled = 1.0f;
    // Ray-traced sun shadows — see ModelLightingUniforms::rtShadowEnabled.
    // The instanced path carries this for the same reason the non-instanced
    // one does: without it the village (every building, crate, tree and rock
    // is an instanced draw) keeps its cascade shadows while the character
    // alone gets traced ones, which reads as "ray tracing does nothing".
    float rtShadowEnabled = 0.0f;
    /// Traced-reflection handover window — see
    /// ModelLightingUniforms::rtReflectionMaxRoughness.
    float rtReflectionMaxRoughness = 0.0f;
    float rtReflectionFadeStart = 0.0f;
    /// `GraphicsSettings::rt_local_shadows` as a float, because a cbuffer has
    /// no enums: 0 = Off, 1 = Ratio, 2 = Full-DI. Rides in what used to be
    /// `_rtPad2`, so the cbuffer neither grew nor moved a field.
    ///
    /// 0 leaves accumulateClusterLights' arithmetic bit-identical — the mask is
    /// not sampled at all, exactly as `giEnabled == 0` skips the probe tap
    /// rather than blending it at weight zero. 1 multiplies the whole
    /// point+spot subtotal by the traced visibility ratio. 2 (R5c) does not
    /// call accumulateClusterLights AT ALL and reads the two demodulated DI
    /// storage textures instead — see shaders/local_di.hlsli.
    float rtLocalShadowMode = 0.0f;
    /// Probe-field GI (render/gi/), R1c — see ModelLightingUniforms::giCascadeBase.
    glm::ivec4 giCascadeBase[3] = {};
    float giEnabled = 0.0f;
    /// Screen-probe gather (R4b, render/gi/screen_probes.hpp). These three were
    /// the trailing GI pads, so the cbuffer neither grew nor moved a field.
    /// `giScreenEnabled` gates every tap in `gi_screen_sample.hlsli`; 0 leaves the
    /// pure probe-field ambient of R1c. The grid rides as floats because the
    /// pads were floats, and the shader casts them back.
    float giScreenEnabled = 0.0f;
    float giScreenProbesX = 0.0f;
    float giScreenProbesY = 0.0f;
};
BRACKEN_HLSL_CBUFFER(InstancedLightingUniforms, 288);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, lightDir, 0);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, _padding0, 12);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, lightColor, 16);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, _padding1, 28);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, ambientSky, 32);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, _padding2, 44);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, ambientGround, 48);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, _padding3, 60);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, baseColorFactor, 64);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, fogColor, 80);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, fogStart, 92);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, fogSunColor, 96);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, fogEnd, 108);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, emissiveFactor, 112);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, fogHeightFalloff, 124);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, cameraPos, 128);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, normalScale, 140);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, hasTexture, 144);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, fogEnabled, 148);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, hasNormalMap, 152);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, ambientStrength, 156);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, sunIntensity, 160);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, metallicFactor, 164);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, roughnessFactor, 168);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, occlusionStrength, 172);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, hasMRMap, 176);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, hasEmissiveMap, 180);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, hasPackedOcclusion, 184);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, alphaCutoff, 188);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, screenAOTexelX, 192);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, screenAOTexelY, 196);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, iblEnabled, 200);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, screenAOEnabled, 204);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, rtShadowEnabled, 208);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, rtReflectionMaxRoughness, 212);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, rtReflectionFadeStart, 216);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, rtLocalShadowMode, 220);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, giCascadeBase, 224);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, giEnabled, 272);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, giScreenEnabled, 276);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, giScreenProbesX, 280);
BRACKEN_HLSL_FIELD(InstancedLightingUniforms, giScreenProbesY, 284);

/**
 * Per-instance data for instanced model rendering (storage buffer).
 * Matches the InstanceData struct in model_instanced.vert.hlsl AND
 * prepass_instanced.vert.hlsl (both index the same per-frame buffer).
 * prevModel is the previous frame's model matrix (motion-vector prepass);
 * statics submit prevModel == model so their velocity is camera-only.
 */
struct InstanceData {
    glm::mat4 model;
    glm::mat4 normalMatrix;
    glm::vec4 tint;
    float noFog;
    float _pad[3] = {0.0f, 0.0f, 0.0f};
    glm::mat4 prevModel{1.0f};
};
BRACKEN_HLSL_STRUCT(InstanceData, 224);
BRACKEN_HLSL_FIELD(InstanceData, model, 0);
BRACKEN_HLSL_FIELD(InstanceData, normalMatrix, 64);
BRACKEN_HLSL_FIELD(InstanceData, tint, 128);
BRACKEN_HLSL_FIELD(InstanceData, noFog, 144);
BRACKEN_HLSL_FIELD(InstanceData, _pad, 148);
BRACKEN_HLSL_FIELD(InstanceData, prevModel, 160);

/**
 * Per-instance data for instanced shadow rendering (storage buffer).
 */
struct ShadowInstanceData {
    glm::mat4 model;
};
BRACKEN_HLSL_STRUCT(ShadowInstanceData, 64);
BRACKEN_HLSL_FIELD(ShadowInstanceData, model, 0);

/**
 * Instanced shadow vertex uniforms (set 1, b0)
 */
struct alignas(16) InstancedShadowUniforms {
    glm::mat4 lightViewProjection;
};
BRACKEN_HLSL_CBUFFER(InstancedShadowUniforms, 64);
BRACKEN_HLSL_FIELD(InstancedShadowUniforms, lightViewProjection, 0);

/**
 * Depth-prepass vertex uniforms for non-instanced static + skinned models —
 * matches prepass_model.vert.hlsl / prepass_skinned.vert.hlsl
 * PrepassTransformUniforms (set 1, b0).
 *
 * SV_Position is produced from model/view/projection with the exact same
 * operation order as the forward shaders so the forward pass can depth-test
 * EQUAL against the prepass depth. `projection` is the JITTERED projection
 * when TAA jitter is on; the *ViewProjectionNoJitter pair feeds the velocity
 * output (velocity must never see jitter).
 *
 * prevPaletteOffset == NO_PREV_PALETTE tells the skinned prepass shader to
 * reuse the current-frame palette for the previous position (zero skin
 * motion; used for motion_id 0, first frames, and pruned/missing history).
 */
/// Fragment uniforms for the alpha-MASK depth-prepass variants: the alpha
/// test must reproduce the forward draw's cutoff exactly so both passes
/// discard the same texels.
struct alignas(16) PrepassMaskedUniforms {
    float alphaCutoff = 0.5f;
    float _padding0[3] = {0.0f, 0.0f, 0.0f};
};
BRACKEN_HLSL_CBUFFER(PrepassMaskedUniforms, 16);
BRACKEN_HLSL_FIELD(PrepassMaskedUniforms, alphaCutoff, 0);
BRACKEN_HLSL_FIELD(PrepassMaskedUniforms, _padding0, 4);

struct alignas(16) PrepassModelUniforms {
    static constexpr uint32_t NO_PREV_PALETTE = 0xFFFFFFFFu;

    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection; // jittered when TAA jitter is enabled
    glm::mat4 prevModel;
    glm::mat4 currViewProjectionNoJitter;
    glm::mat4 prevViewProjectionNoJitter;
    uint32_t paletteOffset = 0;                     // skinned only
    uint32_t prevPaletteOffset = NO_PREV_PALETTE;   // skinned only
    float _padding0[2] = {0.0f, 0.0f};
    /// Inverse-transpose of `model`, read only by the depth-prepass G-buffer
    /// variants (ray-traced reflections). It lives in this shared block rather
    /// than a second binding so every prepass model pipeline — plain, masked
    /// and G-buffer — pushes one identical uniform layout.
    glm::mat4 normalMatrix{1.0f};
};
BRACKEN_HLSL_CBUFFER(PrepassModelUniforms, 464);
BRACKEN_HLSL_FIELD(PrepassModelUniforms, model, 0);
BRACKEN_HLSL_FIELD(PrepassModelUniforms, view, 64);
BRACKEN_HLSL_FIELD(PrepassModelUniforms, projection, 128);
BRACKEN_HLSL_FIELD(PrepassModelUniforms, prevModel, 192);
BRACKEN_HLSL_FIELD(PrepassModelUniforms, currViewProjectionNoJitter, 256);
BRACKEN_HLSL_FIELD(PrepassModelUniforms, prevViewProjectionNoJitter, 320);
BRACKEN_HLSL_FIELD(PrepassModelUniforms, paletteOffset, 384);
BRACKEN_HLSL_FIELD(PrepassModelUniforms, prevPaletteOffset, 388);
BRACKEN_HLSL_FIELD(PrepassModelUniforms, _padding0, 392);
BRACKEN_HLSL_FIELD(PrepassModelUniforms, normalMatrix, 400);

/**
 * @brief Depth-prepass G-buffer fragment uniforms (roughness).
 *
 * Matches the PrepassGBufferUniforms cbuffer in prepass_gbuffer.frag.hlsl.
 * Only the roughness FACTOR is sent: the reflection pass uses it as a
 * threshold test ("is this smooth enough to mirror?"), which does not justify
 * a metallic-roughness texture fetch per pixel in the prepass to refine.
 */
struct alignas(16) PrepassGBufferUniforms {
    /// The material's roughness/metallic FACTORS. The shader multiplies them
    /// by the metallicRoughness texture exactly as the forward pass does
    /// (glTF: G = roughness, B = metallic), because in real content the
    /// factors carry almost no information: of the 79 materials in the staged
    /// medieval/nature packs, 77 have roughnessFactor == 1.0 and all 79 have
    /// metallicFactor == 0.0. Classifying on the factor alone marked every
    /// piece of pack art "fully rough", so no stone, brick, plaster, tile or
    /// wood pixel could ever receive a traced reflection.
    float roughness = 1.0f;
    /// Needed by the reflection pass to build F0: a dielectric reflects ~4% of
    /// the environment, a metal reflects nearly all of it. Without this every
    /// traced surface is weighted as a mirror and grass looks like glass.
    float metallic = 0.0f;
    /// 1 = a metallicRoughness texture is bound at fragment sampler 0 and the
    /// factors above are modulated by it; 0 = the 1x1 white dummy is bound and
    /// the factors stand alone. Mirrors ModelLightingUniforms::hasMRMap.
    float hasMRMap = 0.0f;
    float _pad2 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(PrepassGBufferUniforms, 16);
BRACKEN_HLSL_FIELD(PrepassGBufferUniforms, roughness, 0);
BRACKEN_HLSL_FIELD(PrepassGBufferUniforms, metallic, 4);
BRACKEN_HLSL_FIELD(PrepassGBufferUniforms, hasMRMap, 8);

/**
 * Depth-prepass vertex uniforms shared by the instanced-model and terrain
 * prepass shaders — matches prepass_instanced.vert.hlsl /
 * prepass_terrain.vert.hlsl PrepassCameraUniforms (set 1, b0).
 * Same jitter/no-jitter split as PrepassModelUniforms.
 */
struct alignas(16) PrepassCameraUniforms {
    glm::mat4 view;
    glm::mat4 projection; // jittered when TAA jitter is enabled
    glm::mat4 currViewProjectionNoJitter;
    glm::mat4 prevViewProjectionNoJitter;
};
BRACKEN_HLSL_CBUFFER(PrepassCameraUniforms, 256);
BRACKEN_HLSL_FIELD(PrepassCameraUniforms, view, 0);
BRACKEN_HLSL_FIELD(PrepassCameraUniforms, projection, 64);
BRACKEN_HLSL_FIELD(PrepassCameraUniforms, currViewProjectionNoJitter, 128);
BRACKEN_HLSL_FIELD(PrepassCameraUniforms, prevViewProjectionNoJitter, 192);

/**
 * Shadow depth pass - vertex uniforms for static models (set 1, b0)
 */
struct alignas(16) ShadowTransformUniforms {
    glm::mat4 lightViewProjection;
    glm::mat4 model;
};
BRACKEN_HLSL_CBUFFER(ShadowTransformUniforms, 128);
BRACKEN_HLSL_FIELD(ShadowTransformUniforms, lightViewProjection, 0);
BRACKEN_HLSL_FIELD(ShadowTransformUniforms, model, 64);

/**
 * Shadow depth pass - vertex uniforms for skinned models (set 1, b0)
 * Matches shadow_skinned.vert.hlsl ShadowTransformUniforms. paletteOffset
 * indexes the frame's bone-palette SSBO (in matrices).
 */
struct alignas(16) ShadowSkinnedTransformUniforms {
    glm::mat4 lightViewProjection;
    glm::mat4 model;
    uint32_t paletteOffset = 0;
    float _padding0[3] = {0.0f, 0.0f, 0.0f};
};
BRACKEN_HLSL_CBUFFER(ShadowSkinnedTransformUniforms, 144);
BRACKEN_HLSL_FIELD(ShadowSkinnedTransformUniforms, lightViewProjection, 0);
BRACKEN_HLSL_FIELD(ShadowSkinnedTransformUniforms, model, 64);
BRACKEN_HLSL_FIELD(ShadowSkinnedTransformUniforms, paletteOffset, 128);
BRACKEN_HLSL_FIELD(ShadowSkinnedTransformUniforms, _padding0, 132);

/**
 * Shadow depth pass - vertex uniforms for terrain (set 1, b0)
 * Terrain vertices are already in world space, no model matrix needed.
 */
struct alignas(16) ShadowTerrainUniforms {
    glm::mat4 lightViewProjection;
};
BRACKEN_HLSL_CBUFFER(ShadowTerrainUniforms, 64);
BRACKEN_HLSL_FIELD(ShadowTerrainUniforms, lightViewProjection, 0);

/**
 * Shadow depth pass - fragment uniforms for alpha-tested casters (set 3, b0)
 * Matches shadow_depth.frag.hlsl ShadowFragUniforms. Padding is three named
 * scalars to mirror the HLSL cbuffer byte-for-byte (an array would stride at
 * 16 bytes on the HLSL side).
 */
struct alignas(16) ShadowFragUniforms {
    int hasTexture = 0;
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
    float _pad2 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(ShadowFragUniforms, 16);
BRACKEN_HLSL_FIELD(ShadowFragUniforms, hasTexture, 0);
BRACKEN_HLSL_FIELD(ShadowFragUniforms, _pad0, 4);
BRACKEN_HLSL_FIELD(ShadowFragUniforms, _pad1, 8);
BRACKEN_HLSL_FIELD(ShadowFragUniforms, _pad2, 12);

/**
 * Shadow data for fragment shaders in the main pass (set 3, b1)
 * All cascades live in one depth atlas laid out as a 2x2 quadrant grid;
 * cascade i occupies quadrant (i%2, i/2). Matches the ShadowUniforms cbuffer
 * declared by every shadow-receiving fragment shader.
 */
struct alignas(16) ShadowDataUniforms {
    glm::mat4 lightViewProjection[4]; // Per-cascade light-space matrices
    glm::vec4 cascadeSplits;          // View-space far depth per cascade
    glm::vec4 cascadeTexelWorld;      // World units per shadow texel, per cascade
    glm::vec4 cascadeDepthRange;      // Light-space depth extent in world units, per cascade
    float shadowMapResolution;        // Quadrant (per-cascade) resolution in texels
    float lightSize;                  // tan(sun angular radius): world penumbra width per world unit of blocker-receiver distance
    float shadowEnabled;              // 0=off, 1=PCF, 2=PCSS
    float lastActiveSplit;            // Far depth of the last active cascade (shader early-out)
    float fadeStart;                  // View depth where shadow distance fade begins
    float fadeEnd;                    // View depth where shadows are fully faded out
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(ShadowDataUniforms, 336);
BRACKEN_HLSL_FIELD(ShadowDataUniforms, lightViewProjection, 0);
BRACKEN_HLSL_FIELD(ShadowDataUniforms, cascadeSplits, 256);
BRACKEN_HLSL_FIELD(ShadowDataUniforms, cascadeTexelWorld, 272);
BRACKEN_HLSL_FIELD(ShadowDataUniforms, cascadeDepthRange, 288);
BRACKEN_HLSL_FIELD(ShadowDataUniforms, shadowMapResolution, 304);
BRACKEN_HLSL_FIELD(ShadowDataUniforms, lightSize, 308);
BRACKEN_HLSL_FIELD(ShadowDataUniforms, shadowEnabled, 312);
BRACKEN_HLSL_FIELD(ShadowDataUniforms, lastActiveSplit, 316);
BRACKEN_HLSL_FIELD(ShadowDataUniforms, fadeStart, 320);
BRACKEN_HLSL_FIELD(ShadowDataUniforms, fadeEnd, 324);
BRACKEN_HLSL_FIELD(ShadowDataUniforms, _pad0, 328);
BRACKEN_HLSL_FIELD(ShadowDataUniforms, _pad1, 332);

/**
 * GTAO pass fragment uniforms (set 3, b0) — matches gtao.frag.hlsl.
 *
 * Horizon-based ground-truth AO (Jimenez et al. 2016) upgraded in R5a to a
 * per-slice 32-sector VISIBILITY BITMASK (SSILVB, Therrien et al. 2023) that
 * also publishes a bent normal and cone aperture. Slice/step/sector counts are
 * compile-time constants in the shader (SLICE_COUNT / STEPS_PER_SLICE /
 * SECTOR_COUNT).
 *
 * Tuning: render::GTAO assigns projection/view/screen sizes, radiusScale
 * (GraphicsSettings::ao_radius), and the temporal jitter per frame. The
 * remaining NSDMI defaults ARE the live values (GTAO-style engine constants);
 * changing them requires a rebuild. maxDistance must stay equal to
 * BlurUniforms::maxDistance (pinned by a test in gtao.test.cpp).
 */
struct alignas(16) GTAOUniforms {
    glm::mat4 projection;
    glm::mat4 invProjection;
    // View -> world. The pass works in view space and publishes the bent
    // normal in WORLD space, because no consumer of the AO texture carries a
    // view matrix to undo a view-space one with.
    glm::mat4 invView;
    glm::vec2 screenSize;    // AO working resolution (half or full res)
    glm::vec2 invScreenSize;
    // World/view-unit sample radius. Characters here are 25-35 units tall, so
    // sub-2.0 radii only catch sub-ankle contact; 4.0 reads as proper corner
    // and crevice darkening. Scaled by the ao_radius settings multiplier.
    float radius = 4.0f;
    float radiusScale = 1.0f;
    // Fraction of the radius over which sample occlusion fades to zero
    // (XeGTAO-style linear falloff at the edge of the sampling sphere).
    float falloffRange = 0.615f;
    // OCCLUDER THICKNESS, in world units, along the view ray. Every depth
    // sample occludes the sectors subtended by a slab this deep and nothing
    // past it (render/gtao.hpp detail::gtao_slab_span). Scaled by radiusScale
    // alongside `radius`, so the slab stays a fixed fraction of the sampling
    // sphere at any ao_radius.
    //
    // 1.0 is a quarter of the default radius and, at this project's scale
    // (characters 25-35 units tall, so ~6 cm per unit), about the thickness of
    // a fence plank or a door — the smallest silhouette in the village that
    // should still cast real contact occlusion. Smaller under-occludes solid
    // walls seen edge-on; larger brings back exactly the over-darkening of
    // railings and grass that the bitmask exists to remove.
    //
    // This REPLACES `thinOccluderCompensation` (0.2, unitless), which blended
    // a lowered horizon back down by a fixed fraction: a guess at thickness
    // with no length in it, identical for a plank and a barn wall, and unable
    // to express an occluder that blocks a band and nothing beyond it.
    float occluderThickness = 1.0f;
    // Per-frame slice-rotation / step-offset jitter in [0,1) (golden-ratio
    // sequence, render::GTAO::render). The temporal accumulation pass
    // integrates the rotating sample pattern over time.
    float temporalRotation = 0.0f;
    float temporalOffset = 0.0f;
    // Explicit far cutoff (linear view distance) for AO, faded over the last
    // 10% in-shader. Replaces the old `depth >= 0.9999` "sky skip" whose
    // world-space cutoff moved with the camera's near plane; 12000 roughly
    // matches where that test landed with the built-in near=5/far=15000 camera.
    float maxDistance = 12000.0f;
    float _padding0 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(GTAOUniforms, 240);
BRACKEN_HLSL_FIELD(GTAOUniforms, projection, 0);
BRACKEN_HLSL_FIELD(GTAOUniforms, invProjection, 64);
BRACKEN_HLSL_FIELD(GTAOUniforms, invView, 128);
BRACKEN_HLSL_FIELD(GTAOUniforms, screenSize, 192);
BRACKEN_HLSL_FIELD(GTAOUniforms, invScreenSize, 200);
BRACKEN_HLSL_FIELD(GTAOUniforms, radius, 208);
BRACKEN_HLSL_FIELD(GTAOUniforms, radiusScale, 212);
BRACKEN_HLSL_FIELD(GTAOUniforms, falloffRange, 216);
BRACKEN_HLSL_FIELD(GTAOUniforms, occluderThickness, 220);
BRACKEN_HLSL_FIELD(GTAOUniforms, temporalRotation, 224);
BRACKEN_HLSL_FIELD(GTAOUniforms, temporalOffset, 228);
BRACKEN_HLSL_FIELD(GTAOUniforms, maxDistance, 232);
BRACKEN_HLSL_FIELD(GTAOUniforms, _padding0, 236);

/**
 * GTAO temporal accumulation pass fragment uniforms (set 3, b0) — matches
 * gtao_temporal.frag.hlsl. Reprojects the AO history with the prepass
 * velocity (prev_uv = uv - velocity, scene/velocity_math.hpp convention) and
 * rejects on relative linear-depth mismatch (the history stores last frame's
 * linear view depth in its G channel). historyValid is 0 on the first frame
 * and after resize/mode gaps — the shader then outputs the current AO
 * unblended, mirroring TAA's invalidation policy.
 */
struct alignas(16) GtaoTemporalUniforms {
    glm::vec2 invScreenSize; // AO working resolution
    // Depth linearization (viewZ = |depthB / (depth + depthA)|, glm RH_ZO):
    // depthA = proj[2][2], depthB = proj[3][2].
    float depthA = 0.0f;
    float depthB = 0.0f;
    float historyValid = 0.0f;
    // History weight ceiling. 0.9 ≈ 10-frame convergence of the rotating
    // GTAO sample pattern.
    float feedbackMax = 0.9f;
    // Disocclusion rejection: feedback scales to 0 as the relative depth
    // difference |prevZ - currZ| / currZ approaches 1 / depthRejectScale.
    float depthRejectScale = 10.0f;
    float _pad0 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(GtaoTemporalUniforms, 32);
BRACKEN_HLSL_FIELD(GtaoTemporalUniforms, invScreenSize, 0);
BRACKEN_HLSL_FIELD(GtaoTemporalUniforms, depthA, 8);
BRACKEN_HLSL_FIELD(GtaoTemporalUniforms, depthB, 12);
BRACKEN_HLSL_FIELD(GtaoTemporalUniforms, historyValid, 16);
BRACKEN_HLSL_FIELD(GtaoTemporalUniforms, feedbackMax, 20);
BRACKEN_HLSL_FIELD(GtaoTemporalUniforms, depthRejectScale, 24);
BRACKEN_HLSL_FIELD(GtaoTemporalUniforms, _pad0, 28);

/**
 * GTAO upsample/finalize pass fragment uniforms (set 3, b0) — matches
 * gtao_upsample.frag.hlsl. Joint-bilateral (depth-aware) upsample of the
 * working-res blurred AO + bent cone to the full-res RGBA8 texture the
 * forward pass samples;
 * degrades to an exact passthrough when working res == full res (GTAO_Full).
 * strength = GraphicsSettings::ao_strength, applied as pow(ao, strength).
 */
struct alignas(16) GtaoUpsampleUniforms {
    glm::vec2 lowResTexelSize; // 1 / AO working resolution
    float depthA = 0.0f;
    float depthB = 0.0f;
    float strength = 1.0f;
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
    float _pad2 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(GtaoUpsampleUniforms, 32);
BRACKEN_HLSL_FIELD(GtaoUpsampleUniforms, lowResTexelSize, 0);
BRACKEN_HLSL_FIELD(GtaoUpsampleUniforms, depthA, 8);
BRACKEN_HLSL_FIELD(GtaoUpsampleUniforms, depthB, 12);
BRACKEN_HLSL_FIELD(GtaoUpsampleUniforms, strength, 16);
BRACKEN_HLSL_FIELD(GtaoUpsampleUniforms, _pad0, 20);
BRACKEN_HLSL_FIELD(GtaoUpsampleUniforms, _pad1, 24);
BRACKEN_HLSL_FIELD(GtaoUpsampleUniforms, _pad2, 28);

/**
 * Bilateral blur pass fragment uniforms (set 3, b0)
 * depthA/depthB linearize the non-linear depth buffer in the shader:
 * viewZ = |depthB / (depth + depthA)| with depthA = proj[2][2], depthB =
 * proj[3][2] (glm RH_ZO; the formula holds for the engine's reversed-Z
 * projection too). Weighting on relative linear view-Z keeps edge rejection
 * consistent at distance (raw depth differences collapse far away -> halos).
 */
struct alignas(16) BlurUniforms {
    glm::vec2 direction;
    glm::vec2 invScreenSize;
    float sharpness = 40.0f;
    float depthA = 0.0f;
    float depthB = 0.0f;
    // Linear view-distance far cutoff; must match GTAOUniforms::maxDistance
    // (past it the AO passes faded to 1.0 and the blur passes through).
    float maxDistance = 12000.0f;
};
BRACKEN_HLSL_CBUFFER(BlurUniforms, 32);
BRACKEN_HLSL_FIELD(BlurUniforms, direction, 0);
BRACKEN_HLSL_FIELD(BlurUniforms, invScreenSize, 8);
BRACKEN_HLSL_FIELD(BlurUniforms, sharpness, 16);
BRACKEN_HLSL_FIELD(BlurUniforms, depthA, 20);
BRACKEN_HLSL_FIELD(BlurUniforms, depthB, 24);
BRACKEN_HLSL_FIELD(BlurUniforms, maxDistance, 28);

/**
 * Composite pass fragment uniforms (set 3, b0)
 * AO is no longer composited here — screen-space AO is applied to the ambient
 * term in the forward shaders (see GTAOUniforms above).
 *
 * Exposure: `exposure` is the manual bias (GraphicsSettings::exposure, linear
 * multiplier); with autoExposureEnabled != 0 the shader additionally
 * multiplies exp2(current EV) read from the exposure-state SSBO (fragment
 * storage buffer 0 — render::AutoExposure::state_buffer()).
 *
 * Color grading: white balance (LMS-space coefficients from
 * render::white_balance_coefficients) applies PRE-tonemap in linear light;
 * the ASC-CDL block (pow(color * slope + offset, power)) applies POST-tonemap
 * before the existing contrast/saturation trims. The *Enabled flags gate the
 * math entirely so all-neutral defaults are PIXEL-IDENTICAL to the ungraded
 * path (pinned by tests — pow/matrix round-trips are not bit-exact even at
 * neutral parameters).
 */
struct alignas(16) CompositeUniforms {
    // Default tracks GraphicsSettings::bloom_strength (the source of truth);
    // pinned by tests in gpu_uniforms.test.cpp and post_process_stack.test.cpp.
    float bloomStrength = 0.25f;
    float volumetricFogEnabled = 0.0f;
    float exposure = 1.0f;
    int tonemapMode = 0;
    float contrast = 1.0f;
    float saturation = 1.0f;
    // Depth-aware (joint bilateral) upsample of the half-res volumetric-fog
    // texture: lowResTexelSize = 1 / half-res fog target size; depthA/depthB
    // linearize the depth buffer (viewZ = |depthB / (depth + depthA)|, glm
    // RH_ZO, standard or reversed-Z). depthA/B of 0 degrade gracefully to
    // plain bilinear weights in-shader.
    glm::vec2 lowResTexelSize{0.0f, 0.0f};
    float depthA = 0.0f;
    float depthB = 0.0f;
    float autoExposureEnabled = 0.0f;
    float whiteBalanceEnabled = 0.0f;
    glm::vec3 whiteBalance{1.0f, 1.0f, 1.0f}; // linear-RGB gains via LMS (neutral = 1,1,1)
    float cdlEnabled = 0.0f;
    glm::vec3 cdlSlope{1.0f, 1.0f, 1.0f};
    float _pad0 = 0.0f;
    glm::vec3 cdlOffset{0.0f, 0.0f, 0.0f};
    float _pad1 = 0.0f;
    glm::vec3 cdlPower{1.0f, 1.0f, 1.0f};
    float _pad2 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(CompositeUniforms, 112);
BRACKEN_HLSL_FIELD(CompositeUniforms, bloomStrength, 0);
BRACKEN_HLSL_FIELD(CompositeUniforms, volumetricFogEnabled, 4);
BRACKEN_HLSL_FIELD(CompositeUniforms, exposure, 8);
BRACKEN_HLSL_FIELD(CompositeUniforms, tonemapMode, 12);
BRACKEN_HLSL_FIELD(CompositeUniforms, contrast, 16);
BRACKEN_HLSL_FIELD(CompositeUniforms, saturation, 20);
BRACKEN_HLSL_FIELD(CompositeUniforms, lowResTexelSize, 24);
BRACKEN_HLSL_FIELD(CompositeUniforms, depthA, 32);
BRACKEN_HLSL_FIELD(CompositeUniforms, depthB, 36);
BRACKEN_HLSL_FIELD(CompositeUniforms, autoExposureEnabled, 40);
BRACKEN_HLSL_FIELD(CompositeUniforms, whiteBalanceEnabled, 44);
BRACKEN_HLSL_FIELD(CompositeUniforms, whiteBalance, 48);
BRACKEN_HLSL_FIELD(CompositeUniforms, cdlEnabled, 60);
BRACKEN_HLSL_FIELD(CompositeUniforms, cdlSlope, 64);
BRACKEN_HLSL_FIELD(CompositeUniforms, _pad0, 76);
BRACKEN_HLSL_FIELD(CompositeUniforms, cdlOffset, 80);
BRACKEN_HLSL_FIELD(CompositeUniforms, _pad1, 92);
BRACKEN_HLSL_FIELD(CompositeUniforms, cdlPower, 96);
BRACKEN_HLSL_FIELD(CompositeUniforms, _pad2, 108);

/**
 * FXAA pass fragment uniforms (set 3, b0)
 * Operates on the post-tonemap gamma-encoded LDR image.
 */
struct alignas(16) FxaaUniforms {
    glm::vec2 invScreenSize;
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(FxaaUniforms, 16);
BRACKEN_HLSL_FIELD(FxaaUniforms, invScreenSize, 0);
BRACKEN_HLSL_FIELD(FxaaUniforms, _pad0, 8);
BRACKEN_HLSL_FIELD(FxaaUniforms, _pad1, 12);

/**
 * TAA resolve pass fragment uniforms - matches taa_resolve.frag.hlsl (set 3, b0).
 *
 * Both matrices are the UNJITTERED view-projections (velocity_math.hpp
 * convention — TAA jitter must never leak into reprojection). historyValid
 * is 0 on the first frame after init/resize/mode-toggle: the shader then
 * outputs the current frame unchanged. feedbackMax and varianceGamma are
 * engine tuning constants (the NSDMI defaults ARE the live values, GTAO-style);
 * render::TAA pushes them verbatim.
 */
struct alignas(16) TaaResolveUniforms {
    glm::mat4 currInvViewProjUnjittered;
    glm::mat4 prevViewProjUnjittered;
    glm::vec2 screenSize;
    glm::vec2 invScreenSize;
    glm::vec3 cameraPos;
    float historyValid = 0.0f;
    // History weight ceiling. 0.9 ≈ 10-frame convergence; disocclusion
    // heuristics (velocity length, clip amount) reduce it per pixel.
    float feedbackMax = 0.9f;
    // Stddev multiplier for the YCoCg variance-clip box. 1.0 is tight
    // (ghost-resistant); raise toward 1.5 for more temporal stability.
    float varianceGamma = 1.0f;
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(TaaResolveUniforms, 176);
BRACKEN_HLSL_FIELD(TaaResolveUniforms, currInvViewProjUnjittered, 0);
BRACKEN_HLSL_FIELD(TaaResolveUniforms, prevViewProjUnjittered, 64);
BRACKEN_HLSL_FIELD(TaaResolveUniforms, screenSize, 128);
BRACKEN_HLSL_FIELD(TaaResolveUniforms, invScreenSize, 136);
BRACKEN_HLSL_FIELD(TaaResolveUniforms, cameraPos, 144);
BRACKEN_HLSL_FIELD(TaaResolveUniforms, historyValid, 156);
BRACKEN_HLSL_FIELD(TaaResolveUniforms, feedbackMax, 160);
BRACKEN_HLSL_FIELD(TaaResolveUniforms, varianceGamma, 164);
BRACKEN_HLSL_FIELD(TaaResolveUniforms, _pad0, 168);
BRACKEN_HLSL_FIELD(TaaResolveUniforms, _pad1, 172);

/**
 * TAA sharpen pass fragment uniforms - matches taa_sharpen.frag.hlsl (set 3, b0).
 * Applied to the resolved output only (history stays unsharpened, so the
 * kernel never compounds). 0 disables (pass-through copy).
 */
struct alignas(16) TaaSharpenUniforms {
    glm::vec2 invScreenSize;
    float sharpenStrength = 0.25f;
    float _pad0 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(TaaSharpenUniforms, 16);
BRACKEN_HLSL_FIELD(TaaSharpenUniforms, invScreenSize, 0);
BRACKEN_HLSL_FIELD(TaaSharpenUniforms, sharpenStrength, 8);
BRACKEN_HLSL_FIELD(TaaSharpenUniforms, _pad0, 12);

/**
 * Particle fragment shader uniforms - matches particle.frag.hlsl (set 3, b0).
 * Pushed once per pass; per-particle tint/opacity ride in the instance
 * storage buffer (ParticleInstanceData). Blend is configured (SRC_ALPHA, ONE);
 * the particle fragment shader writes alpha 1.0 so the effective color blend
 * is (ONE, ONE) additive. Fade is encoded by scaling rgb.
 */
struct alignas(16) ParticleFragmentUniforms {
    float emissiveBoost = 1.0f;
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
    float _pad2 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(ParticleFragmentUniforms, 16);
BRACKEN_HLSL_FIELD(ParticleFragmentUniforms, emissiveBoost, 0);
BRACKEN_HLSL_FIELD(ParticleFragmentUniforms, _pad0, 4);
BRACKEN_HLSL_FIELD(ParticleFragmentUniforms, _pad1, 8);
BRACKEN_HLSL_FIELD(ParticleFragmentUniforms, _pad2, 12);

/**
 * Particle batch vertex uniforms - matches particle.vert.hlsl BatchUniforms
 * (set 1, b1). instanceOffset is the batch's first record in the per-frame
 * particle instance storage buffer; camera data goes through
 * InstancedCameraUniforms (set 1, b0), pushed once per pass.
 */
struct alignas(16) ParticleBatchUniforms {
    uint32_t instanceOffset = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
    uint32_t _pad2 = 0;
};
BRACKEN_HLSL_CBUFFER(ParticleBatchUniforms, 16);
BRACKEN_HLSL_FIELD(ParticleBatchUniforms, instanceOffset, 0);
BRACKEN_HLSL_FIELD(ParticleBatchUniforms, _pad0, 4);
BRACKEN_HLSL_FIELD(ParticleBatchUniforms, _pad1, 8);
BRACKEN_HLSL_FIELD(ParticleBatchUniforms, _pad2, 12);

/**
 * Per-particle data for instanced particle rendering (storage buffer).
 * Matches particle.vert.hlsl ParticleInstance: color.rgb tints, color.a is
 * opacity (folded into rgb by the vertex stage - additive blending).
 */
struct ParticleInstanceData {
    glm::mat4 model;
    glm::vec4 color;
};
BRACKEN_HLSL_STRUCT(ParticleInstanceData, 80);
BRACKEN_HLSL_FIELD(ParticleInstanceData, model, 0);
BRACKEN_HLSL_FIELD(ParticleInstanceData, color, 64);

/**
 * Bloom downsample pass fragment uniforms (set 3, b0)
 * threshold is compared against POST-exposure luminance (the shader multiplies
 * scene luminance by `exposure` before thresholding), so what blooms is stable
 * when the exposure setting changes; the extracted color stays in pre-exposure
 * scene units because the composite pass applies exposure to the summed bloom.
 */
struct alignas(16) BloomDownsampleUniforms {
    glm::vec2 srcTexelSize;
    float threshold = 0.8f;
    float isFirstPass = 1.0f;
    float exposure = 1.0f;
    float _padding[3] = {0.0f, 0.0f, 0.0f};
};
BRACKEN_HLSL_CBUFFER(BloomDownsampleUniforms, 32);
BRACKEN_HLSL_FIELD(BloomDownsampleUniforms, srcTexelSize, 0);
BRACKEN_HLSL_FIELD(BloomDownsampleUniforms, threshold, 8);
BRACKEN_HLSL_FIELD(BloomDownsampleUniforms, isFirstPass, 12);
BRACKEN_HLSL_FIELD(BloomDownsampleUniforms, exposure, 16);
BRACKEN_HLSL_FIELD(BloomDownsampleUniforms, _padding, 20);

/**
 * Bloom upsample pass fragment uniforms (set 3, b0)
 */
struct alignas(16) BloomUpsampleUniforms {
    glm::vec2 srcTexelSize;
    float bloomRadius = 1.0f;
    float _padding = 0.0f;
};
BRACKEN_HLSL_CBUFFER(BloomUpsampleUniforms, 16);
BRACKEN_HLSL_FIELD(BloomUpsampleUniforms, srcTexelSize, 0);
BRACKEN_HLSL_FIELD(BloomUpsampleUniforms, bloomRadius, 8);
BRACKEN_HLSL_FIELD(BloomUpsampleUniforms, _padding, 12);

/**
 * Volumetric fog pass fragment uniforms (set 3, b0)
 * God rays sample the whole CSM atlas: the shader picks a cascade per
 * ray-march sample from its camera distance (atlas quadrant i%2, i/2).
 * Defaults mirror the EnvironmentState volumetric-fog defaults.
 */
struct alignas(16) VolumetricFogUniforms {
    glm::mat4 invViewProjection{};
    glm::mat4 shadowLightViewProjection[4]{}; // per-cascade light-space matrices
    glm::vec4 cascadeSplits{};                // view-space far distance per cascade
    glm::vec4 shadowBias{};                   // per-cascade depth compare bias in light-space [0,1] units
    glm::vec3 lightDir{};
    float fogDensity = 0.0015f;
    glm::vec3 lightColor{1.0f, 0.9f, 0.7f};
    float scatterStrength = 0.25f;
    glm::vec3 fogColor{0.45f, 0.56f, 0.74f};
    float fogHeight = 100.0f;
    glm::vec3 cameraPos{};
    float fogFalloff = 0.005f;
    float startDist = 100.0f;      // near dead zone: no volumetric fog closer than this
    float maxDist = 3000.0f;       // ray-march distance cap in world units
    float godRaysEnabled = 1.0f;
    float densityMultiplier = 1.0f;
    float shadowAtlasTexel = 0.0f; // 1 / shadow atlas width
    float lastActiveSplit = 0.0f;  // far distance of the last active cascade (lit beyond it)
    float _padding0 = 0.0f;
    float _padding1 = 0.0f;
    /// The hemisphere ambient (EnvironmentState). Needed by this pass only
    /// since R1c: it is both the probe field's fallback and the reference the
    /// GI gain is measured against, so the two must be the same numbers.
    glm::vec3 ambientSky{};
    float _ambientPad0 = 0.0f;
    glm::vec3 ambientGround{};
    float _ambientPad1 = 0.0f;
    /// Probe-field GI (render/gi/), R1c — see ModelLightingUniforms::giCascadeBase.
    glm::ivec4 giCascadeBase[3] = {};
    float giEnabled = 0.0f;
    float _giPad0 = 0.0f;
    float _giPad1 = 0.0f;
    float _giPad2 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(VolumetricFogUniforms, 544);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, invViewProjection, 0);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, shadowLightViewProjection, 64);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, cascadeSplits, 320);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, shadowBias, 336);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, lightDir, 352);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, fogDensity, 364);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, lightColor, 368);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, scatterStrength, 380);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, fogColor, 384);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, fogHeight, 396);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, cameraPos, 400);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, fogFalloff, 412);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, startDist, 416);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, maxDist, 420);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, godRaysEnabled, 424);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, densityMultiplier, 428);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, shadowAtlasTexel, 432);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, lastActiveSplit, 436);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, _padding0, 440);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, _padding1, 444);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, ambientSky, 448);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, _ambientPad0, 460);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, ambientGround, 464);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, _ambientPad1, 476);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, giCascadeBase, 480);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, giEnabled, 528);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, _giPad0, 532);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, _giPad1, 536);
BRACKEN_HLSL_FIELD(VolumetricFogUniforms, _giPad2, 540);

/**
 * Auto-exposure histogram COMPUTE uniforms - matches
 * exposure_histogram.comp.hlsl (compute set 2, b0). Defaults mirror the
 * render/auto_exposure.hpp constants (EV -10..+16 over bins 1-255); the
 * mapping's CPU reference is exposure_bin_from_luma().
 */
struct alignas(16) ExposureHistogramUniforms {
    uint32_t width = 0;  // scene color extent in pixels
    uint32_t height = 0;
    float minLog2Luma = -10.0f;         // EXPOSURE_MIN_LOG2_LUMA
    float invLog2Range = 1.0f / 26.0f;  // 1 / (max - min log2 luma)
};
BRACKEN_HLSL_CBUFFER(ExposureHistogramUniforms, 16);
BRACKEN_HLSL_FIELD(ExposureHistogramUniforms, width, 0);
BRACKEN_HLSL_FIELD(ExposureHistogramUniforms, height, 4);
BRACKEN_HLSL_FIELD(ExposureHistogramUniforms, minLog2Luma, 8);
BRACKEN_HLSL_FIELD(ExposureHistogramUniforms, invLog2Range, 12);

/**
 * Auto-exposure adapt COMPUTE uniforms - matches exposure_adapt.comp.hlsl
 * (compute set 2, b0). Percentile/target defaults mirror the
 * render::AutoExposure constants; speeds/clamps track GraphicsSettings.
 * firstFrame != 0 snaps the adapted EV straight to the target.
 */
struct alignas(16) ExposureAdaptUniforms {
    float minLog2Luma = -10.0f;
    float log2Range = 26.0f;
    float deltaTime = 0.0f;
    float firstFrame = 0.0f;
    float speedUp = 1.5f;   // EV/s toward brighter
    float speedDown = 2.5f; // EV/s toward darker
    float minEv = -4.0f;
    float maxEv = 4.0f;
    float lowPercentile = 0.60f;
    float highPercentile = 0.05f;
    float log2TargetGray = -1.80f; // AutoExposure::DEFAULT_LOG2_TARGET_GRAY
    float _pad0 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(ExposureAdaptUniforms, 48);
BRACKEN_HLSL_FIELD(ExposureAdaptUniforms, minLog2Luma, 0);
BRACKEN_HLSL_FIELD(ExposureAdaptUniforms, log2Range, 4);
BRACKEN_HLSL_FIELD(ExposureAdaptUniforms, deltaTime, 8);
BRACKEN_HLSL_FIELD(ExposureAdaptUniforms, firstFrame, 12);
BRACKEN_HLSL_FIELD(ExposureAdaptUniforms, speedUp, 16);
BRACKEN_HLSL_FIELD(ExposureAdaptUniforms, speedDown, 20);
BRACKEN_HLSL_FIELD(ExposureAdaptUniforms, minEv, 24);
BRACKEN_HLSL_FIELD(ExposureAdaptUniforms, maxEv, 28);
BRACKEN_HLSL_FIELD(ExposureAdaptUniforms, lowPercentile, 32);
BRACKEN_HLSL_FIELD(ExposureAdaptUniforms, highPercentile, 36);
BRACKEN_HLSL_FIELD(ExposureAdaptUniforms, log2TargetGray, 40);
BRACKEN_HLSL_FIELD(ExposureAdaptUniforms, _pad0, 44);

// NOTE: the clustered-lighting cbuffer layout (clusters.hlsli) is defined by
// render::lighting::ClusterParams (light_cluster.hpp) — SceneRenderer pushes
// that struct directly; there is deliberately no mirror here.

/**
 * Ray-traced shadow compute uniforms — matches rt_shadow.comp.hlsl
 * RtShadowUniforms (set 2, b0). Used by: render::RtShadowRenderer.
 */
struct alignas(16) RtShadowUniforms {
    glm::mat4 invViewProj;    // clip -> world, for depth reconstruction
    glm::vec3 sunDirection;   // direction light TRAVELS
    float rayMaxDistance;     // ray length; shorter is cheaper and misses less
    glm::vec3 cameraPosition;
    float normalBias;         // offset along the normal against self-shadowing
    glm::uvec2 outputSize;    // traced resolution (may be half the screen)
    uint32_t tlasNodeCount;   // 0 = no acceleration structure this frame
    float depthBias;
    /// cos of the sun's angular radius; 1.0 is a point light (hard shadow).
    float sunCosThetaMax;
    uint32_t frameIndex;      // advances the sample sequence
    uint32_t hasGBuffer;      // 0 = reconstruct the normal from depth taps
    /// Analytic capsule occluders bound at t6 (docs/rt_lighting_plan.md §5.8).
    /// Took the slot that was `_pad0`, so the cbuffer's extent is unchanged.
    /// 0 collapses the capsule loop and reproduces the pre-capsule trace.
    uint32_t shadowCapsuleCount = 0;
};

BRACKEN_HLSL_CBUFFER(RtShadowUniforms, 128);
BRACKEN_HLSL_FIELD(RtShadowUniforms, invViewProj, 0);
BRACKEN_HLSL_FIELD(RtShadowUniforms, sunDirection, 64);
BRACKEN_HLSL_FIELD(RtShadowUniforms, rayMaxDistance, 76);
BRACKEN_HLSL_FIELD(RtShadowUniforms, cameraPosition, 80);
BRACKEN_HLSL_FIELD(RtShadowUniforms, normalBias, 92);
BRACKEN_HLSL_FIELD(RtShadowUniforms, outputSize, 96);
BRACKEN_HLSL_FIELD(RtShadowUniforms, tlasNodeCount, 104);
BRACKEN_HLSL_FIELD(RtShadowUniforms, depthBias, 108);
BRACKEN_HLSL_FIELD(RtShadowUniforms, sunCosThetaMax, 112);
BRACKEN_HLSL_FIELD(RtShadowUniforms, frameIndex, 116);
BRACKEN_HLSL_FIELD(RtShadowUniforms, hasGBuffer, 120);
BRACKEN_HLSL_FIELD(RtShadowUniforms, shadowCapsuleCount, 124);

/**
 * Froxel-RIS local-light shadow trace — matches rt_local_shadow.comp.hlsl
 * (set 2, b0). Used by: render::RtRenderer. The pass ALSO takes ClusterParams
 * at set 2, b1 — that cbuffer is ClusterGrid's and is pushed straight from
 * `ClusterGrid::params()`, so it has no mirror here.
 */
struct alignas(16) RtLocalShadowUniforms {
    glm::mat4 invViewProj;
    glm::uvec2 outputSize;
    uint32_t tlasNodeCount = 0;
    float normalBias = 1.5f;
    float depthBias = 0.5f;
    /// Emitter radius in world units for the sphere-cap shadow ray. One value
    /// for every light until per-light authoring lands in R6.
    float lightSourceRadius = 10.0f;
    /// Cap on carried-over reservoir M, as a multiple of this frame's candidate
    /// count.
    float historyMaxMRatio = 20.0f;
    uint32_t frameIndex = 0;
    uint32_t hasGBuffer = 0;
    uint32_t historyValid = 0;
    uint32_t candidateCount = 6;
    /// NO depthA/depthB pair, deliberately: the froxel lookup needs an ABSOLUTE
    /// view depth that agrees with the CPU binner's, and the shader takes it
    /// from `clusterParams.view` — the matrix ClusterGrid binned against —
    /// rather than from a reversed-Z linearization that only has to be
    /// self-consistent everywhere else it is used.
    ///
    /// `LocalRtShadowMode` as a uint: 0/1 = Ratio (packed visibility bits
    /// only), 2 = Full-DI (also the two demodulated radiance targets). Took the
    /// slot that was `_pad0` through R5b.
    uint32_t diMode = 0;
    /// The eye, for the Full-DI tier's specular half. A visibility bit has no
    /// view dependence, which is why the Ratio tier never needed this and R5c
    /// is where it arrives.
    glm::vec3 cameraPosition{0.0f};
    /// R6, emissive mesh lights. Records live, the alias table's normalizing
    /// constant, M_e, and whether an emissive index in last frame's reservoir
    /// still names the same triangle (0 after a TLAS topology rebuild).
    ///
    /// `emissiveCount` is 0 in the Ratio tier BY CONSTRUCTION, which is the
    /// whole tier split: see the header comment of rt_local_shadow.comp.hlsl.
    /// Zero also makes every RIS weight in that shader collapse to the R5c
    /// expression bit for bit, which is what keeps the Ratio baselines exact.
    uint32_t emissiveCount = 0;
    float emissiveTotalPower = 0.0f;
    uint32_t emissiveCandidateCount = 0;
    uint32_t emissiveHistoryValid = 0;
    /// Analytic capsule occluders, bound at u6 in SET 1 for this pass — the
    /// eight read-only slots were spent before capsules existed, so a read-only
    /// use of a read-write binding is what the budget leaves (R2b's precedent).
    /// Took the slot that was `_pad1`, so the extent is unchanged.
    uint32_t shadowCapsuleCount = 0;
    /// R7a. 0 = RESAMPLE (froxel candidates + the temporal merge), 1 = REUSE
    /// (the pairwise-MIS spatial merge) + the ray + the DI block. The pass is
    /// dispatched twice per frame with the same bindings and different reservoir
    /// buffers in the two slots; rt_local_shadow.comp.hlsl's header carries the
    /// argument for why the split is forced rather than chosen.
    uint32_t stage = 0;
    /// k — spatial neighbours attempted. 0 makes stage 1's merge the identity
    /// and reproduces the pre-R7a image bit for bit.
    uint32_t spatialNeighbors = 0;
    /// Neighbour ring radius, in traced texels.
    float spatialRadius = 8.0f;
    /// R8 — q_f, the share of each emissive candidate draw taken from the
    /// pixel's PER-FROXEL emissive list rather than from the scene-wide power
    /// alias table (docs/rt_lighting_plan.md §5.7 follow-up 2). Took the slot
    /// that was `_pad0`, so the extent is unchanged.
    ///
    /// 0 collapses the mixture to its global component and reproduces R6's draw
    /// bit for bit — which is both the A/B control and what the CPU sends on any
    /// frame the froxel table is not this frame's.
    float emissiveFroxelShare = 0.0f;
};

BRACKEN_HLSL_CBUFFER(RtLocalShadowUniforms, 160);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, invViewProj, 0);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, outputSize, 64);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, tlasNodeCount, 72);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, normalBias, 76);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, depthBias, 80);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, lightSourceRadius, 84);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, historyMaxMRatio, 88);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, frameIndex, 92);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, hasGBuffer, 96);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, historyValid, 100);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, candidateCount, 104);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, diMode, 108);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, cameraPosition, 112);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, emissiveCount, 124);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, emissiveTotalPower, 128);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, emissiveCandidateCount, 132);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, emissiveHistoryValid, 136);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, shadowCapsuleCount, 140);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, stage, 144);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, spatialNeighbors, 148);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, spatialRadius, 152);
BRACKEN_HLSL_FIELD(RtLocalShadowUniforms, emissiveFroxelShare, 156);

/**
 * The Full-DI tier's temporal denoiser — matches
 * rt_di_denoise_temporal.comp.hlsl (set 2, b0). R5c, docs/rt_lighting_plan.md
 * §5.3.
 */
struct alignas(16) RtDiTemporalUniforms {
    glm::uvec2 tracedSize{0, 0};
    /// Gradient dimensions (traced / render::GRADIENT_STRATUM).
    glm::uvec2 stratumCount{0, 0};
    float maxAccumFrames = 32.0f;
    float clampSigma = 1.5f;
    /// R5b's analytic sun gradient, already a global hysteresis floor. 0 is the
    /// identity.
    float lightingReset = 0.0f;
    uint32_t historyValid = 0;
    uint32_t gradientValid = 0;
    uint32_t exposureEnabled = 0;
    float manualExposureEv = 0.0f;
    uint32_t _pad0 = 0;
};

BRACKEN_HLSL_CBUFFER(RtDiTemporalUniforms, 48);
BRACKEN_HLSL_FIELD(RtDiTemporalUniforms, tracedSize, 0);
BRACKEN_HLSL_FIELD(RtDiTemporalUniforms, stratumCount, 8);
BRACKEN_HLSL_FIELD(RtDiTemporalUniforms, maxAccumFrames, 16);
BRACKEN_HLSL_FIELD(RtDiTemporalUniforms, clampSigma, 20);
BRACKEN_HLSL_FIELD(RtDiTemporalUniforms, lightingReset, 24);
BRACKEN_HLSL_FIELD(RtDiTemporalUniforms, historyValid, 28);
BRACKEN_HLSL_FIELD(RtDiTemporalUniforms, gradientValid, 32);
BRACKEN_HLSL_FIELD(RtDiTemporalUniforms, exposureEnabled, 36);
BRACKEN_HLSL_FIELD(RtDiTemporalUniforms, manualExposureEv, 40);
BRACKEN_HLSL_FIELD(RtDiTemporalUniforms, _pad0, 44);

/**
 * The Full-DI tier's spatial denoiser — matches rt_di_denoise_spatial.comp.hlsl
 * (set 2, b0). R5c.
 */
struct alignas(16) RtDiSpatialUniforms {
    glm::uvec2 tracedSize{0, 0};
    float depthA = 0.0f;
    float depthB = 0.0f;
    /// Radius in traced texels for a one-sample pixel; the curve down to
    /// render::lighting::DI_FILTER_MIN_RADIUS is `di_filter_radius`.
    float maxRadius = 3.0f;
    float depthSigma = 0.05f;
    float normalPower = 8.0f;
    /// R7c: a-trous tap spacing, 1 on the first iteration and 2 on the second.
    /// Above 1 also enables the converged-pixel pass-through, which is what
    /// keeps a steady-state image sharp while the wide pass still hides a
    /// disocclusion.
    uint32_t stepSize = 1;
    /// 1 on the frame's LAST iteration — the one that converts the diffuse alpha
    /// from the accumulation count to the linear view depth the forward
    /// reconstruction reads. An intermediate iteration must pass the count on.
    uint32_t isFinal = 1;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
    uint32_t _pad2 = 0;
};

BRACKEN_HLSL_CBUFFER(RtDiSpatialUniforms, 48);
BRACKEN_HLSL_FIELD(RtDiSpatialUniforms, tracedSize, 0);
BRACKEN_HLSL_FIELD(RtDiSpatialUniforms, depthA, 8);
BRACKEN_HLSL_FIELD(RtDiSpatialUniforms, depthB, 12);
BRACKEN_HLSL_FIELD(RtDiSpatialUniforms, maxRadius, 16);
BRACKEN_HLSL_FIELD(RtDiSpatialUniforms, depthSigma, 20);
BRACKEN_HLSL_FIELD(RtDiSpatialUniforms, normalPower, 24);
BRACKEN_HLSL_FIELD(RtDiSpatialUniforms, stepSize, 28);
BRACKEN_HLSL_FIELD(RtDiSpatialUniforms, isFinal, 32);
BRACKEN_HLSL_FIELD(RtDiSpatialUniforms, _pad0, 36);
BRACKEN_HLSL_FIELD(RtDiSpatialUniforms, _pad1, 40);
BRACKEN_HLSL_FIELD(RtDiSpatialUniforms, _pad2, 44);

/**
 * The R7 temporal-noise probe — matches rt_di_noise.comp.hlsl (set 2, b0).
 *
 * A diagnostic, not part of the frame: dispatched only when
 * `render::RtLocalShadowSettings::di_noise_probe` is set.
 */
struct alignas(16) RtDiNoiseUniforms {
    glm::uvec2 tracedSize{0, 0};
    /// First of this probe's three uints in the shared denoiser counter buffer.
    uint32_t counterBase = 0;
    /// 0 whenever `prevLuma` does not hold a measurement of the previous frame
    /// of THIS image — the first probed frame, a resize, or any gap in the tier.
    uint32_t historyValid = 0;
};

BRACKEN_HLSL_CBUFFER(RtDiNoiseUniforms, 16);
BRACKEN_HLSL_FIELD(RtDiNoiseUniforms, tracedSize, 0);
BRACKEN_HLSL_FIELD(RtDiNoiseUniforms, counterBase, 8);
BRACKEN_HLSL_FIELD(RtDiNoiseUniforms, historyValid, 12);

/**
 * Shadow-denoiser tile classification — matches
 * rt_shadow_denoise_classify.comp.hlsl (set 2, b0).
 */
struct alignas(16) RtShadowClassifyUniforms {
    glm::uvec2 tileCount;
    glm::uvec2 _pad0{0, 0};
};

BRACKEN_HLSL_CBUFFER(RtShadowClassifyUniforms, 16);
BRACKEN_HLSL_FIELD(RtShadowClassifyUniforms, tileCount, 0);
BRACKEN_HLSL_FIELD(RtShadowClassifyUniforms, _pad0, 8);

/**
 * Shadow-denoiser temporal accumulation — matches
 * rt_shadow_denoise_temporal.comp.hlsl (set 2, b0).
 *
 * Shared by BOTH stochastic visibility signals (sun and froxel-RIS local
 * lights), which go through the same three denoiser pipelines — see
 * RtRenderer::VisibilityDenoiser.
 */
struct alignas(16) RtShadowTemporalUniforms {
    glm::uvec2 tracedSize;
    glm::uvec2 tileCount;
    /// Dimensions of the reconstructed A-SVGF gradient (traced / 3, rounded up).
    glm::uvec2 stratumCount{0, 0};
    float depthA = 0.0f;
    float depthB = 0.0f;
    float maxAccumFrames = 32.0f;
    /// Relative linear-view-depth difference above which history is rejected.
    /// NOT retuned when the linearization was fixed (2026-08-02,
    /// contracts/depth_linear.hpp) and that is a decision, not an oversight:
    /// the old effective threshold was `0.05 * (1 - 2t/far)` — 0.0475 at 100
    /// units, 0.0375 at 500, 0.025 at 1000, and "never reject" past far/2,
    /// where every surface linearized to the same saturated constant. No
    /// constant reproduces a distance-dependent curve, and 5% is what this
    /// field has always claimed to mean. Must stay equal to
    /// RtShadowGradientUniforms::depthRejectRelative.
    float depthRejectRelative = 0.05f;
    /// R5b: the analytic sun gradient's global hysteresis floor, [0, 1]
    /// (render/rt_denoise_math.hpp's `lighting_reset_floor`). 0 is the identity
    /// and the value for every frame whose sun did not move.
    float lightingReset = 0.0f;
    uint32_t historyValid = 0;
    /// 0 when no traced gradient was produced this frame, in which case the
    /// bound gradient texture is a valid stand-in that must not be read.
    uint32_t gradientValid = 0;
    /// First of this signal's four uint slots in the shared, never-reset
    /// diagnostics buffer (sun 0..3, local 4..7).
    uint32_t counterBase = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
};

BRACKEN_HLSL_CBUFFER(RtShadowTemporalUniforms, 64);
BRACKEN_HLSL_FIELD(RtShadowTemporalUniforms, tracedSize, 0);
BRACKEN_HLSL_FIELD(RtShadowTemporalUniforms, tileCount, 8);
BRACKEN_HLSL_FIELD(RtShadowTemporalUniforms, stratumCount, 16);
BRACKEN_HLSL_FIELD(RtShadowTemporalUniforms, depthA, 24);
BRACKEN_HLSL_FIELD(RtShadowTemporalUniforms, depthB, 28);
BRACKEN_HLSL_FIELD(RtShadowTemporalUniforms, maxAccumFrames, 32);
BRACKEN_HLSL_FIELD(RtShadowTemporalUniforms, depthRejectRelative, 36);
BRACKEN_HLSL_FIELD(RtShadowTemporalUniforms, lightingReset, 40);
BRACKEN_HLSL_FIELD(RtShadowTemporalUniforms, historyValid, 44);
BRACKEN_HLSL_FIELD(RtShadowTemporalUniforms, gradientValid, 48);
BRACKEN_HLSL_FIELD(RtShadowTemporalUniforms, counterBase, 52);
BRACKEN_HLSL_FIELD(RtShadowTemporalUniforms, _pad0, 56);
BRACKEN_HLSL_FIELD(RtShadowTemporalUniforms, _pad1, 60);

/**
 * A-SVGF gradient trace for the SUN visibility signal — matches
 * rt_shadow_gradient.comp.hlsl (set 2, b0). Phase R5b, §5.6.
 *
 * One thread per 3x3 stratum of the traced resolution; each re-traces this
 * frame's scene with the PREVIOUS frame's sample (`frameIndex - 1` through the
 * same replayable `rtSample2D`) and compares against the previous frame's raw
 * bit. The difference is "the lighting at this surface changed", with motion
 * and sampling noise both factored out.
 */
struct alignas(16) RtShadowGradientUniforms {
    glm::mat4 invViewProj{1.0f};
    glm::vec3 sunDirection{0.0f, -1.0f, 0.0f};
    float rayMaxDistance = 6000.0f;
    glm::uvec2 tracedSize{0, 0};
    glm::uvec2 stratumCount{0, 0};
    float normalBias = 1.5f;
    float depthBias = 0.5f;
    float sunCosThetaMax = 1.0f;
    /// THIS frame's counter — the replay uses frameIndex - 1.
    uint32_t frameIndex = 0;
    uint32_t tlasNodeCount = 0;
    uint32_t hasGBuffer = 0;
    uint32_t historyValid = 0;
    float depthA = 0.0f;
    float depthB = 0.0f;
    float depthRejectRelative = 0.05f;
    /// §5.8 — the capsule set the trace used. The replay must see the SAME one
    /// or every capsule-shadowed stratum fires a gradient forever. Took the
    /// slot that was `_pad0`, so the extent is unchanged.
    uint32_t shadowCapsuleCount = 0;
    uint32_t _pad1 = 0;
    /**
     * Previous frame's unjittered view-projection, for the replay's depth test.
     *
     * The test asks "is this the same surface the previous frame saw here?", and
     * the only honest way to ask it is to project THIS frame's world position
     * with the PREVIOUS frame's camera and compare that against the depth the
     * history recorded. Comparing the history's depth against this frame's view
     * depth instead measures the camera's own per-frame depth change, which at
     * the playground's 420-unit camera boom is 2-5 world units against a
     * tolerance of 0.17 — so the gradient collapsed under any camera motion and
     * history was never invalidated for a moving occluder's shadow. Measured
     * before the fix: `local_grad_fire` 0.0278 pinned vs 0.0054 orbiting, while
     * `di noise` went the other way, 0.00014 vs 0.00400.
     */
    glm::mat4 prevViewProjUnjittered{1.0f};
};

BRACKEN_HLSL_CBUFFER(RtShadowGradientUniforms, 208);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, invViewProj, 0);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, sunDirection, 64);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, rayMaxDistance, 76);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, tracedSize, 80);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, stratumCount, 88);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, normalBias, 96);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, depthBias, 100);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, sunCosThetaMax, 104);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, frameIndex, 108);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, tlasNodeCount, 112);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, hasGBuffer, 116);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, historyValid, 120);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, depthA, 124);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, depthB, 128);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, depthRejectRelative, 132);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, shadowCapsuleCount, 136);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, _pad1, 140);
BRACKEN_HLSL_FIELD(RtShadowGradientUniforms, prevViewProjUnjittered, 144);

/**
 * A-SVGF gradient trace for the LOCAL visibility signal — matches
 * rt_local_shadow_gradient.comp.hlsl (set 2, b0). Phase R5b, §5.6.
 *
 * Same shape as the sun's, plus the emitter radius: the replay has to reproduce
 * WHICH LIGHT the previous frame's reservoir picked (resolved through R2a's
 * prev->current remap), not merely which direction it sampled.
 */
struct alignas(16) RtLocalShadowGradientUniforms {
    glm::mat4 invViewProj{1.0f};
    glm::uvec2 tracedSize{0, 0};
    glm::uvec2 stratumCount{0, 0};
    float normalBias = 1.5f;
    float depthBias = 0.5f;
    float lightSourceRadius = 10.0f;
    uint32_t frameIndex = 0;
    uint32_t tlasNodeCount = 0;
    uint32_t hasGBuffer = 0;
    uint32_t historyValid = 0;
    /// R6 — the emissive record set the PREVIOUS frame's reservoirs were
    /// written against, and whether it is still the same set. A flagged
    /// reservoir index that cannot be resolved is a full gradient, exactly as a
    /// vanished analytic light is.
    uint32_t emissiveCount = 0;
    uint32_t emissiveHistoryValid = 0;
    /// §5.8 — same contract as the sun gradient's: the replay must see the
    /// trace's capsules. Took the slot that was `_pad0`.
    uint32_t shadowCapsuleCount = 0;
    /// `LOCAL_GRADIENT_DEPTH_REJECT_RELATIVE`. A uniform since 2026-08-08 —
    /// it used to be a bare `#define` in the shader, which is how it silently
    /// diverged from the sun's after that constant was re-measured. The
    /// divergence is correct; having two definitions of it was not.
    float depthRejectRelative = 0.0005f;
    uint32_t _pad2 = 0;
    /**
     * Previous frame's VIEW matrix, for the replay's depth test.
     *
     * The sun gradient needs a view-PROJECTION here because its history stores a
     * linearized device depth; this one needs the plain view matrix, because the
     * reservoir's `depth` lane is a true view depth
     * (`-mul(clusterParams.view, float4(worldPos, 1)).z`) and never passes
     * through `depth_linear.hlsli` at all. Same defect either way: comparing
     * last frame's recorded depth against THIS frame's view depth measures the
     * camera's motion, not the surface's.
     */
    glm::mat4 prevView{1.0f};
};

BRACKEN_HLSL_CBUFFER(RtLocalShadowGradientUniforms, 192);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, invViewProj, 0);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, tracedSize, 64);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, stratumCount, 72);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, normalBias, 80);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, depthBias, 84);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, lightSourceRadius, 88);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, frameIndex, 92);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, tlasNodeCount, 96);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, hasGBuffer, 100);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, historyValid, 104);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, emissiveCount, 108);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, emissiveHistoryValid, 112);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, shadowCapsuleCount, 116);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, depthRejectRelative, 120);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, _pad2, 124);
BRACKEN_HLSL_FIELD(RtLocalShadowGradientUniforms, prevView, 128);

/**
 * Dense reconstruction of the sparse gradient — matches
 * rt_gradient_reconstruct.comp.hlsl (set 2, b0). Both visibility signals run
 * this one pipeline against their own pair of stratum-resolution targets.
 */
struct alignas(16) RtGradientReconstructUniforms {
    glm::uvec2 stratumCount{0, 0};
    glm::uvec2 _pad0{0, 0};
};

BRACKEN_HLSL_CBUFFER(RtGradientReconstructUniforms, 16);
BRACKEN_HLSL_FIELD(RtGradientReconstructUniforms, stratumCount, 0);
BRACKEN_HLSL_FIELD(RtGradientReconstructUniforms, _pad0, 8);

/**
 * Shadow-denoiser filter + upsample — matches
 * rt_shadow_denoise_filter.comp.hlsl (set 2, b0).
 */
struct alignas(16) RtShadowFilterUniforms {
    glm::uvec2 outputSize;
    glm::uvec2 tracedSize;
    glm::uvec2 tileCount;
    float depthA = 0.0f;
    float depthB = 0.0f;
    float maxRadius = 3.0f;
    /// Relative depth sigma of the bilateral weight. Same call as
    /// RtShadowTemporalUniforms::depthRejectRelative above: NOT retuned when
    /// the linearization was fixed, because the old effective sigma was
    /// `0.05 * (1 - 2t/far)` and degenerated into a uniform-weight box filter
    /// past far/2.
    float depthSigma = 0.05f;
    glm::uvec2 _pad0{0, 0};
};

BRACKEN_HLSL_CBUFFER(RtShadowFilterUniforms, 48);
BRACKEN_HLSL_FIELD(RtShadowFilterUniforms, outputSize, 0);
BRACKEN_HLSL_FIELD(RtShadowFilterUniforms, tracedSize, 8);
BRACKEN_HLSL_FIELD(RtShadowFilterUniforms, tileCount, 16);
BRACKEN_HLSL_FIELD(RtShadowFilterUniforms, depthA, 24);
BRACKEN_HLSL_FIELD(RtShadowFilterUniforms, depthB, 28);
BRACKEN_HLSL_FIELD(RtShadowFilterUniforms, maxRadius, 32);
BRACKEN_HLSL_FIELD(RtShadowFilterUniforms, depthSigma, 36);
BRACKEN_HLSL_FIELD(RtShadowFilterUniforms, _pad0, 40);

/**
 * Ray-traced reflection compute uniforms — matches rt_reflection.comp.hlsl
 * RtReflectionUniforms (set 2, b0). Used by: render::RtRenderer.
 */
struct alignas(16) RtReflectionUniforms {
    /// R3: the leading 336 bytes ARE `ShadowDataUniforms`, field for field and
    /// in order, exactly as `GiProbeTraceUniforms` carries them and for exactly
    /// the same reason — the shader declares them under the names
    /// `shadow_common.hlsli` requires and calls the forward pass's own
    /// `calcShadow()` at the hit, so a reflected facade standing in a shadow
    /// reflects as shadowed. Reimplementing the cascade math here is named as a
    /// trap in `docs/rt_lighting_plan.md` §8; this is the answer to it.
    ShadowDataUniforms shadow{};
    glm::mat4 invViewProj{1.0f};
    glm::vec3 cameraPosition{0.0f};
    float rayMaxDistance = 0.0f;
    /// Unit view direction — `calcShadow` selects a cascade by VIEW depth, and
    /// a reflection-ray hit has to be projected onto this to have one.
    glm::vec3 cameraForward{0.0f, 0.0f, -1.0f};
    float maxRoughness = 0.0f; // above this the IBL probe is used instead of a ray
    glm::vec3 sunDirection{0.0f, -1.0f, 0.0f};
    float normalBias = 0.0f;
    /// ALREADY multiplied by GraphicsSettings::sun_intensity. Before R3 it was
    /// not, and the resulting factor-of-three deficit happened to cancel the
    /// missing Lambertian 1/pi at the playground's sun_intensity of 3.
    glm::vec3 sunColor{0.0f};
    /// Roughness at which the traced contribution starts fading out; it
    /// reaches zero at maxRoughness. Shared with the forward shaders through
    /// shaders/reflection_gate.hlsli so traced + IBL specular sum to one.
    /// NOT `fadeStart`: since R3 that name belongs to `shadow`'s cascade
    /// distance fade, which shadow_common.hlsli reads by name.
    float gateFadeStart = 0.0f;
    glm::vec3 ambientSky{0.0f};
    /// 1 when the IBL sky-capture cube is bound and valid; 0 falls back to the
    /// ambientSky/ambientGround hemisphere on a ray miss.
    float useSkyCube = 0.0f;
    glm::vec3 ambientGround{0.0f};
    /// Scale on the probe-field term at the hit; 1 is the field as stored.
    float bounceScale = 1.0f;
    /// 1 when the probe atlases hold a live field. 0 binds the engine's dummies
    /// and the hit's ambient falls to the hemisphere above — which is what
    /// makes reflections improve with GI off as well as on.
    float giEnabled = 0.0f;
    /// Always 1 here: a reflected point outside the field's coverage must be
    /// lit by SOMETHING, and the hemisphere is the engine's documented no-GI
    /// ambient tier. Probe rays pass 0. See `rtHitAmbient`.
    float giAmbientFallback = 1.0f;
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
    /// Absolute lattice coordinate of each cascade's corner probe (xyz; w
    /// unused), sent rather than re-derived — see `gi_sample.hlsli`.
    glm::ivec4 giCascadeBase[3]{};
    /// Optional analytic water receiver. receiverMode == 0 reads the regular
    /// depth/G-buffer pair; == 1 intersects the camera ray with this displaced
    /// water surface and reads the simulation state from sceneNormal.
    glm::vec4 waterBounds{0.0f};    // minX, minZ, maxX, maxZ
    glm::vec4 waterSurface{0.0f};   // height, roughness, displacement, normal strength
    glm::vec4 waterSimWindow{0.0f}; // minX, minZ, inverse extent, edge fade width
    glm::vec4 waterFlowTime{0.0f};  // flow XZ, speed, absolute time
    glm::vec4 waterDetail{0.0f};    // detail strength, receiver mode, visibility bias, unused
    glm::uvec2 outputSize{0, 0};
    uint32_t tlasNodeCount = 0;
    /// Advances the GGX sample sequence. Must change every frame or the
    /// temporal pass averages the same lobe sample forever.
    uint32_t frameIndex = 0;
    uint32_t giLightCount = 0;
    /// The light buffer's per-array capacity (MAX_LIGHTS) — the payload strides
    /// are derived from it exactly as cluster_lighting.hlsli derives them.
    uint32_t giLightCapacity = 0;
    uint32_t _pad2 = 0;
    uint32_t _pad3 = 0;
};

BRACKEN_HLSL_CBUFFER(RtReflectionUniforms, 672);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, shadow, 0);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, invViewProj, 336);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, cameraPosition, 400);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, rayMaxDistance, 412);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, cameraForward, 416);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, maxRoughness, 428);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, sunDirection, 432);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, normalBias, 444);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, sunColor, 448);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, gateFadeStart, 460);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, ambientSky, 464);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, useSkyCube, 476);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, ambientGround, 480);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, bounceScale, 492);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, giEnabled, 496);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, giAmbientFallback, 500);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, _pad0, 504);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, _pad1, 508);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, giCascadeBase, 512);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, waterBounds, 560);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, waterSurface, 576);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, waterSimWindow, 592);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, waterFlowTime, 608);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, waterDetail, 624);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, outputSize, 640);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, tlasNodeCount, 648);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, frameIndex, 652);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, giLightCount, 656);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, giLightCapacity, 660);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, _pad2, 664);
BRACKEN_HLSL_FIELD(RtReflectionUniforms, _pad3, 668);

/**
 * Reflection-denoiser temporal accumulation — matches
 * rt_reflection_denoise_temporal.comp.hlsl (set 2, b0).
 */
struct alignas(16) RtReflectionTemporalUniforms {
    /// R3, virtual-hit-point reprojection. `invViewProj` reconstructs the
    /// RECEIVER's world position from the depth buffer (so it must be the same
    /// matrix the trace used, jitter and all); the two unjittered view-projs
    /// then bracket the VIRTUAL point's motion, and their difference is a
    /// virtual motion vector directly comparable to the velocity buffer's —
    /// which is itself built from unjittered matrices (velocity_math.hpp), so
    /// the jitter cancels on both sides rather than having to be removed.
    glm::mat4 invViewProj{1.0f};
    glm::mat4 viewProjUnjittered{1.0f};
    glm::mat4 prevViewProjUnjittered{1.0f};
    glm::vec3 cameraPosition{0.0f};
    /// The trace's ray range, which is what its R16F hitT target is normalized
    /// by. Multiplying it back is the only way to recover metres here.
    float rayMaxDistance = 0.0f;
    glm::uvec2 tracedSize{0, 0};
    float maxAccumFrames = 16.0f;
    float clampSigma = 1.5f;
    uint32_t historyValid = 0;
    /// 1 when the hit-distance, depth and G-buffer textures are all real. 0
    /// binds valid-but-meaningless stand-ins and takes the pure surface-motion
    /// path, which is exactly the pre-R3 behaviour.
    uint32_t virtualValid = 0;
    /// R5b: 1 when `exposureState` holds a live adapted EV. 0 binds the buffer
    /// anyway (a descriptor may not be empty) and takes `manualExposureEv`.
    uint32_t exposureEnabled = 0;
    /// GraphicsSettings::exposure as an EV (log2 of the multiplier), so the
    /// clamps mean the same thing with auto exposure off as with it on.
    float manualExposureEv = 0.0f;
};

BRACKEN_HLSL_CBUFFER(RtReflectionTemporalUniforms, 240);
BRACKEN_HLSL_FIELD(RtReflectionTemporalUniforms, invViewProj, 0);
BRACKEN_HLSL_FIELD(RtReflectionTemporalUniforms, viewProjUnjittered, 64);
BRACKEN_HLSL_FIELD(RtReflectionTemporalUniforms, prevViewProjUnjittered, 128);
BRACKEN_HLSL_FIELD(RtReflectionTemporalUniforms, cameraPosition, 192);
BRACKEN_HLSL_FIELD(RtReflectionTemporalUniforms, rayMaxDistance, 204);
BRACKEN_HLSL_FIELD(RtReflectionTemporalUniforms, tracedSize, 208);
BRACKEN_HLSL_FIELD(RtReflectionTemporalUniforms, maxAccumFrames, 216);
BRACKEN_HLSL_FIELD(RtReflectionTemporalUniforms, clampSigma, 220);
BRACKEN_HLSL_FIELD(RtReflectionTemporalUniforms, historyValid, 224);
BRACKEN_HLSL_FIELD(RtReflectionTemporalUniforms, virtualValid, 228);
BRACKEN_HLSL_FIELD(RtReflectionTemporalUniforms, exposureEnabled, 232);
BRACKEN_HLSL_FIELD(RtReflectionTemporalUniforms, manualExposureEv, 236);

/**
 * Full-resolution material/depth/normal-guided resolve of the sparse traced
 * reflection signal into the HDR scene. It runs before TAA/bloom/exposure, so
 * all later image processing sees one coherent lighting result.
 */
struct alignas(16) RtReflectionResolveUniforms {
    glm::mat4 invViewProj{1.0f};
    glm::vec3 cameraPosition{0.0f};
    float maxRoughness = 0.0f;
    glm::vec2 outputSize{1.0f};
    glm::vec2 reflectionSize{1.0f};
    float depthA = 0.0f;
    float depthB = 0.0f;
    float fadeStart = 0.0f;
    float intensity = 1.0f;
};
BRACKEN_HLSL_CBUFFER(RtReflectionResolveUniforms, 112);
BRACKEN_HLSL_FIELD(RtReflectionResolveUniforms, invViewProj, 0);
BRACKEN_HLSL_FIELD(RtReflectionResolveUniforms, cameraPosition, 64);
BRACKEN_HLSL_FIELD(RtReflectionResolveUniforms, maxRoughness, 76);
BRACKEN_HLSL_FIELD(RtReflectionResolveUniforms, outputSize, 80);
BRACKEN_HLSL_FIELD(RtReflectionResolveUniforms, reflectionSize, 88);
BRACKEN_HLSL_FIELD(RtReflectionResolveUniforms, depthA, 96);
BRACKEN_HLSL_FIELD(RtReflectionResolveUniforms, depthB, 100);
BRACKEN_HLSL_FIELD(RtReflectionResolveUniforms, fadeStart, 104);
BRACKEN_HLSL_FIELD(RtReflectionResolveUniforms, intensity, 108);

// ===========================================================================
// Probe-field GI (phase R1b of docs/rt_lighting_plan.md)
// ===========================================================================

/**
 * One probe scheduled for update this frame — the record of
 * `gi_probe_trace/blend/classify`'s update-list StructuredBuffer.
 *
 * The CPU sends the probe's LATTICE world position; the shader adds the
 * relocation offset it reads out of the offset texture, so a probe that was
 * nudged out of a wall on a previous update keeps its nudge without the CPU
 * ever having to know about it.
 */
struct alignas(16) GiProbeUpdateRecord {
    glm::vec3 worldPosition{0.0f};
    /// Storage slot inside the cascade (`gi_math.hpp`'s torus wrap already
    /// applied), which is also the atlas tile index.
    uint32_t slot = 0;
    /// Bit 0: the probe was PROBE_UNINITIALIZED when it was scheduled, so the
    /// blend passes take the new estimate wholesale instead of blending into
    /// whatever the slot's previous occupant left.
    uint32_t flags = 0;
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
    float _pad2 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(GiProbeUpdateRecord, 32);
BRACKEN_HLSL_FIELD(GiProbeUpdateRecord, worldPosition, 0);
BRACKEN_HLSL_FIELD(GiProbeUpdateRecord, slot, 12);
BRACKEN_HLSL_FIELD(GiProbeUpdateRecord, flags, 16);
BRACKEN_HLSL_FIELD(GiProbeUpdateRecord, _pad0, 20);
BRACKEN_HLSL_FIELD(GiProbeUpdateRecord, _pad1, 24);
BRACKEN_HLSL_FIELD(GiProbeUpdateRecord, _pad2, 28);

inline constexpr uint32_t GI_UPDATE_FLAG_UNINITIALIZED = 1u;

/// What the classify pass counts, per cascade, into its read-write buffer.
/// Read back through a fenced download (see render::gi::ProbeVolume), never
/// synchronously — these are diagnostics, not a frame dependency.
struct alignas(16) GiProbeCounters {
    uint32_t relocated = 0;
    uint32_t insideGeometry = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
};
BRACKEN_HLSL_CBUFFER(GiProbeCounters, 16);
BRACKEN_HLSL_FIELD(GiProbeCounters, relocated, 0);
BRACKEN_HLSL_FIELD(GiProbeCounters, insideGeometry, 4);
BRACKEN_HLSL_FIELD(GiProbeCounters, _pad0, 8);
BRACKEN_HLSL_FIELD(GiProbeCounters, _pad1, 12);

/**
 * GiProbeTraceUniforms (set 2, b0) — gi_probe_trace.comp.hlsl.
 *
 * The leading 336 bytes ARE `ShadowDataUniforms`, field for field and in order,
 * and the shader declares them under exactly the names `shadow_common.hlsli`
 * requires. That is not a coincidence and not a copy: it lets the trace shader
 * `#include "shadow_common.hlsli"` and call `calcShadow()` — the same code the
 * forward pass runs — so the cascade selection, the standard-Z atlas
 * addressing, the normal-offset and slope-scaled biases and the distance fade
 * cannot drift between the raster and probe paths. `docs/rt_lighting_plan.md`
 * §8 calls the alternative out by name as a trap; this is the answer to it.
 */
struct alignas(16) GiProbeTraceUniforms {
    ShadowDataUniforms shadow{};
    glm::vec3 cameraPosition{0.0f};
    float rayMaxDistance = 4000.0f;
    /// Unit view direction: `calcShadow` selects a cascade by VIEW depth, and a
    /// probe-ray hit has to be projected onto this to have one.
    glm::vec3 cameraForward{0.0f, 0.0f, -1.0f};
    float probeSpacing = 150.0f;
    glm::vec3 sunDirection{0.0f, -1.0f, 0.0f}; ///< direction light TRAVELS
    float normalBias = 1.0f;
    glm::vec3 sunColor{0.0f};
    /// Scale on the previous-frame probe irradiance fed back at each hit. 1.0
    /// is true infinite bounce; below 1 it is a geometric series that converges
    /// faster, which is the knob for a scene bright enough to ring.
    float bounceScale = 1.0f;
    glm::vec3 ambientSky{0.0f};
    /// 1 when the IBL sky-capture cube is bound and valid; 0 falls back to the
    /// ambientSky/ambientGround hemisphere on a ray miss.
    float useSkyCube = 0.0f;
    glm::vec3 ambientGround{0.0f};
    /// PROBE_MAX_OFFSET_FRACTION * probeSpacing, precomputed.
    float maxOffset = 0.0f;
    /// Absolute lattice coordinate of each cascade's corner probe (xyz; w
    /// unused). All three, because a ray fired from cascade 2 can land inside
    /// cascade 0 and must take its bounce from the finest field that covers it.
    glm::ivec4 cascadeBase[3]{};
    uint32_t cascadeIndex = 0;
    uint32_t rayCount = 0;
    uint32_t probeCount = 0;
    uint32_t tlasNodeCount = 0;
    uint32_t lightCount = 0;
    /// The light buffer's per-array capacity (MAX_LIGHTS) — the payload strides
    /// are derived from it exactly as cluster_lighting.hlsli derives them.
    uint32_t lightCapacity = 0;
    /// Advances the ray-set rotation. Must change every frame or every update
    /// of a probe re-samples the same directions and the estimate is biased.
    uint32_t frameIndex = 0;
    uint32_t _pad0 = 0;
    /// `shaders/rt_hit_shade.hlsli`'s two ambient flags, pinned by
    /// `ProbeVolume::trace` to (1, 0). The probe field is by definition
    /// available to a probe ray, and a probe's own bounce term must NOT fall
    /// back to the hemisphere where the field has no answer: the probe already
    /// collects sky through its ray MISSES, so a hemisphere on top would
    /// double-count it and break the bounce series' convergence argument. The
    /// reflection pass passes (giEnabled, 1) for the opposite reason — see
    /// `rtHitAmbient`.
    float giEnabled = 1.0f;
    float giAmbientFallback = 0.0f;
    float _pad1 = 0.0f;
    float _pad2 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(GiProbeTraceUniforms, 528);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, shadow, 0);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, cameraPosition, 336);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, rayMaxDistance, 348);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, cameraForward, 352);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, probeSpacing, 364);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, sunDirection, 368);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, normalBias, 380);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, sunColor, 384);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, bounceScale, 396);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, ambientSky, 400);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, useSkyCube, 412);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, ambientGround, 416);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, maxOffset, 428);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, cascadeBase, 432);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, cascadeIndex, 480);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, rayCount, 484);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, probeCount, 488);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, tlasNodeCount, 492);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, lightCount, 496);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, lightCapacity, 500);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, frameIndex, 504);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, _pad0, 508);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, giEnabled, 512);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, giAmbientFallback, 516);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, _pad1, 520);
BRACKEN_HLSL_FIELD(GiProbeTraceUniforms, _pad2, 524);

/**
 * GiProbeBlendUniforms (set 2, b0) — shared by gi_probe_blend.comp.hlsl and
 * gi_probe_blend_visibility.comp.hlsl. The two differ only in payload size,
 * which is a compile-time constant in each shader (and therefore not a uniform
 * that could disagree with the atlas it writes).
 */
/**
 * VoxelVoxelizeUniforms (set 2, b0) — voxel_voxelize.comp.hlsl.
 *
 * The RASTER GI path's voxelization dispatch (RG1). One dispatch per cascade,
 * flat over (instance, triangle) pairs.
 */
struct alignas(16) VoxelVoxelizeUniforms {
    /// Which cascade this dispatch fills.
    int32_t cascade = 0;
    /// Its base corner in whole voxels. Every write is tested against this
    /// window — torus addressing would otherwise wrap a triangle from the far
    /// side of the map onto a texel near the camera, with full confidence.
    glm::ivec3 base{0, 0, 0};
    /// Total (instance, triangle) pairs — the dispatch's flat extent.
    uint32_t workCount = 0;
    /// Instances, for the shader's binary search over the prefix sums.
    uint32_t instanceCount = 0;
    /// Threads per row of the 2D dispatch (groups_x * 64). The dispatch is 2D
    /// only to stay inside Vulkan's guaranteed 65535 groups per dimension — a
    /// village is millions of (instance, triangle) pairs, and the flat 1D form
    /// made the driver segfault rather than report the limit.
    uint32_t dispatchWidth = 0;
    uint32_t _pad1 = 0;
};

BRACKEN_HLSL_CBUFFER(VoxelVoxelizeUniforms, 32);
BRACKEN_HLSL_FIELD(VoxelVoxelizeUniforms, cascade, 0);
BRACKEN_HLSL_FIELD(VoxelVoxelizeUniforms, base, 4);
BRACKEN_HLSL_FIELD(VoxelVoxelizeUniforms, workCount, 16);
BRACKEN_HLSL_FIELD(VoxelVoxelizeUniforms, instanceCount, 20);
BRACKEN_HLSL_FIELD(VoxelVoxelizeUniforms, dispatchWidth, 24);
BRACKEN_HLSL_FIELD(VoxelVoxelizeUniforms, _pad1, 28);

/**
 * VoxelMipUniforms (set 2, b0) — voxel_mip.comp.hlsl.
 *
 * One dispatch per level. Exists because SDL_GenerateMipmapsForGPUTexture
 * FAULTS on a 3D texture — see the shader.
 */
struct alignas(16) VoxelMipUniforms {
    /// Resolution of the level being written.
    glm::uvec3 dstSize{0, 0, 0};
    /// Level being read; the destination is this + 1.
    float srcLevel = 0.0f;
};

BRACKEN_HLSL_CBUFFER(VoxelMipUniforms, 16);
BRACKEN_HLSL_FIELD(VoxelMipUniforms, dstSize, 0);
BRACKEN_HLSL_FIELD(VoxelMipUniforms, srcLevel, 12);

/**
 * VoxelInjectUniforms (set 2, b0) — voxel_inject.comp.hlsl.
 *
 * Leading 176 bytes are the shadow_common.hlsli block, embedded verbatim and in
 * the same order GiProbeTraceUniforms carries it — that file names its uniforms
 * and the including shader owns their slots, so every consumer repeats the
 * block.
 */
struct alignas(16) VoxelInjectUniforms {
    glm::mat4 shadowLightViewProjection[4]{};
    glm::vec4 cascadeSplits{0.0f};
    glm::vec4 cascadeTexelWorld{0.0f};
    glm::vec4 cascadeDepthRange{0.0f};
    float shadowMapResolution = 0.0f;
    float lightSize = 0.0f;
    float shadowEnabled = 0.0f;
    float lastActiveSplit = 0.0f;
    float fadeStart = 0.0f;
    float fadeEnd = 0.0f;
    float _shadowPad0 = 0.0f;
    float _shadowPad1 = 0.0f;

    int32_t cascade = 0;
    glm::ivec3 base{0};
    /// A voxel has no pixel and therefore no view depth; giving it the same
    /// projection a fragment at that world point would get is what makes the
    /// two select the same shadow cascade.
    glm::vec3 cameraPosition{0.0f};
    float sunIntensity = 1.0f;
    glm::vec3 cameraForward{0.0f, 0.0f, -1.0f};
    float skyFloor = 0.0f;
    glm::vec3 sunDirection{0.0f, -1.0f, 0.0f};
    float _pad0 = 0.0f;
    glm::vec3 sunColor{1.0f};
    float _pad1 = 0.0f;
    glm::vec3 skyColor{0.0f};
    /// Lights in the bound cluster buffer, and the capacity its offsets were
    /// built with — the two differ, because the layout is sized by capacity.
    uint32_t lightCount = 0;
    uint32_t lightCapacity = 0;
    uint32_t _pad2 = 0;
    glm::vec2 _pad3{0.0f};
};

BRACKEN_HLSL_CBUFFER(VoxelInjectUniforms, 448);
BRACKEN_HLSL_FIELD(VoxelInjectUniforms, shadowLightViewProjection, 0);
BRACKEN_HLSL_FIELD(VoxelInjectUniforms, cascadeSplits, 256);
BRACKEN_HLSL_FIELD(VoxelInjectUniforms, cascadeTexelWorld, 272);
BRACKEN_HLSL_FIELD(VoxelInjectUniforms, cascadeDepthRange, 288);
BRACKEN_HLSL_FIELD(VoxelInjectUniforms, cascade, 336);
BRACKEN_HLSL_FIELD(VoxelInjectUniforms, base, 340);
BRACKEN_HLSL_FIELD(VoxelInjectUniforms, cameraPosition, 352);
BRACKEN_HLSL_FIELD(VoxelInjectUniforms, sunIntensity, 364);
BRACKEN_HLSL_FIELD(VoxelInjectUniforms, cameraForward, 368);
BRACKEN_HLSL_FIELD(VoxelInjectUniforms, skyFloor, 380);
BRACKEN_HLSL_FIELD(VoxelInjectUniforms, sunDirection, 384);
BRACKEN_HLSL_FIELD(VoxelInjectUniforms, sunColor, 400);
BRACKEN_HLSL_FIELD(VoxelInjectUniforms, skyColor, 416);

/**
 * VoxelConeTraceUniforms (set 2, b0) — voxel_cone_trace.comp.hlsl.
 *
 * The raster GI path's replacement for GiProbeTraceUniforms. It writes the same
 * ray-results contract into the same buffer, so `rayCount` and `rayMaxDistance`
 * must carry the ray-traced path's values or the visibility moments would
 * change meaning between GI modes.
 */
struct alignas(16) VoxelConeTraceUniforms {
    uint32_t probeCount = 0;
    uint32_t rayCount = 0;
    uint32_t frameIndex = 0;
    float rayMaxDistance = 6000.0f;

    glm::vec3 cameraPosition{0.0f};
    float maxDistance = 0.0f;

    glm::vec3 skyColor{0.0f};
    float bounceScale = 1.0f;

    /// The three cascade base corners; xyz used, w padding.
    glm::ivec4 cascadeBases[3]{};
};

BRACKEN_HLSL_CBUFFER(VoxelConeTraceUniforms, 96);
BRACKEN_HLSL_FIELD(VoxelConeTraceUniforms, probeCount, 0);
BRACKEN_HLSL_FIELD(VoxelConeTraceUniforms, rayCount, 4);
BRACKEN_HLSL_FIELD(VoxelConeTraceUniforms, frameIndex, 8);
BRACKEN_HLSL_FIELD(VoxelConeTraceUniforms, rayMaxDistance, 12);
BRACKEN_HLSL_FIELD(VoxelConeTraceUniforms, cameraPosition, 16);
BRACKEN_HLSL_FIELD(VoxelConeTraceUniforms, maxDistance, 28);
BRACKEN_HLSL_FIELD(VoxelConeTraceUniforms, skyColor, 32);
BRACKEN_HLSL_FIELD(VoxelConeTraceUniforms, bounceScale, 44);
BRACKEN_HLSL_FIELD(VoxelConeTraceUniforms, cascadeBases, 48);

struct alignas(16) GiProbeBlendUniforms {
    uint32_t probeCount = 0;
    uint32_t rayCount = 0;
    /// Distances entering the visibility moments are clamped here
    /// (PROBE_VISIBILITY_DISTANCE_SCALE * spacing). Ignored by the irradiance
    /// blend, which carries the field so both passes take one struct.
    float distanceClamp = 0.0f;
    /// What a miss recorded as its distance, so the visibility blend can treat
    /// "no occluder" as "the clamp" instead of as a real surface at 4000 units.
    float rayMaxDistance = 0.0f;
    /// The trace's frame counter. The blend passes RE-DERIVE each ray's
    /// direction from it to weight that ray's radiance — so this must be the
    /// value the trace ran with, or every texel weights the wrong ray.
    uint32_t frameIndex = 0;
    /// Which cascade's BAND of the shared atlas this dispatch writes. Since
    /// R1c every cascade lives in one texture (gi_math.hpp,
    /// ATLAS_ROWS_ALL_CASCADES), so the tile origin needs it.
    uint32_t cascadeIndex = 0;
    /// R5b: the analytic sun gradient's global hysteresis floor, [0, 1]. Read
    /// by the IRRADIANCE blend only — the visibility moments describe geometry
    /// — but declared in both shaders because they share this struct. 0 is the
    /// identity, so a static sun leaves the field bit-identical.
    float lightingReset = 0.0f;
    uint32_t _pad0 = 0;
};
BRACKEN_HLSL_CBUFFER(GiProbeBlendUniforms, 32);
BRACKEN_HLSL_FIELD(GiProbeBlendUniforms, probeCount, 0);
BRACKEN_HLSL_FIELD(GiProbeBlendUniforms, rayCount, 4);
BRACKEN_HLSL_FIELD(GiProbeBlendUniforms, distanceClamp, 8);
BRACKEN_HLSL_FIELD(GiProbeBlendUniforms, rayMaxDistance, 12);
BRACKEN_HLSL_FIELD(GiProbeBlendUniforms, frameIndex, 16);
BRACKEN_HLSL_FIELD(GiProbeBlendUniforms, cascadeIndex, 20);
BRACKEN_HLSL_FIELD(GiProbeBlendUniforms, lightingReset, 24);
BRACKEN_HLSL_FIELD(GiProbeBlendUniforms, _pad0, 28);

/** GiProbeClassifyUniforms (set 2, b0) — gi_probe_classify.comp.hlsl. */
struct alignas(16) GiProbeClassifyUniforms {
    uint32_t probeCount = 0;
    uint32_t rayCount = 0;
    /// Which cascade's counter slot this dispatch accumulates into.
    uint32_t cascadeIndex = 0;
    /// PROBE_MAX_OFFSET_FRACTION * spacing.
    float maxOffset = 0.0f;
    float probeSpacing = 0.0f;
    /// Same contract as GiProbeBlendUniforms::frameIndex — this pass rebuilds
    /// the backface rays' directions to decide which way to nudge the probe.
    uint32_t frameIndex = 0;
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
};
BRACKEN_HLSL_CBUFFER(GiProbeClassifyUniforms, 32);
BRACKEN_HLSL_FIELD(GiProbeClassifyUniforms, probeCount, 0);
BRACKEN_HLSL_FIELD(GiProbeClassifyUniforms, rayCount, 4);
BRACKEN_HLSL_FIELD(GiProbeClassifyUniforms, cascadeIndex, 8);
BRACKEN_HLSL_FIELD(GiProbeClassifyUniforms, maxOffset, 12);
BRACKEN_HLSL_FIELD(GiProbeClassifyUniforms, probeSpacing, 16);
BRACKEN_HLSL_FIELD(GiProbeClassifyUniforms, frameIndex, 20);
BRACKEN_HLSL_FIELD(GiProbeClassifyUniforms, _pad0, 24);
BRACKEN_HLSL_FIELD(GiProbeClassifyUniforms, _pad1, 28);

/**
 * HzbCopyUniforms (set 2, b0) — hzb_copy.comp.hlsl.
 *
 * Mip 0 of the hierarchical-Z pyramid is the same size as the prepass depth it
 * copies, so one extent serves both. The trailing pad is named because HLSL
 * rounds the cbuffer to 16 bytes and uniform_layout.test.cpp compares the
 * declared extent against sizeof.
 */
struct alignas(16) HzbCopyUniforms {
    uint32_t mipWidth = 0;
    uint32_t mipHeight = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
};
BRACKEN_HLSL_CBUFFER(HzbCopyUniforms, 16);
BRACKEN_HLSL_FIELD(HzbCopyUniforms, mipWidth, 0);
BRACKEN_HLSL_FIELD(HzbCopyUniforms, mipHeight, 4);
BRACKEN_HLSL_FIELD(HzbCopyUniforms, _pad0, 8);
BRACKEN_HLSL_FIELD(HzbCopyUniforms, _pad1, 12);

/**
 * HzbReduceUniforms (set 2, b0) — hzb_reduce.comp.hlsl.
 *
 * Both extents are sent rather than the destination alone: the source's parity
 * is what decides whether the last destination texel gathers two or three
 * texels along an axis (render/hzb.hpp), and re-deriving `src = dst * 2` in the
 * shader gets that wrong for exactly the edge the pyramid must not drop.
 */
struct alignas(16) HzbReduceUniforms {
    uint32_t dstWidth = 0;
    uint32_t dstHeight = 0;
    uint32_t srcWidth = 0;
    uint32_t srcHeight = 0;
};
BRACKEN_HLSL_CBUFFER(HzbReduceUniforms, 16);
BRACKEN_HLSL_FIELD(HzbReduceUniforms, dstWidth, 0);
BRACKEN_HLSL_FIELD(HzbReduceUniforms, dstHeight, 4);
BRACKEN_HLSL_FIELD(HzbReduceUniforms, srcWidth, 8);
BRACKEN_HLSL_FIELD(HzbReduceUniforms, srcHeight, 12);

// ===========================================================================
// Screen-probe gather (render/gi/screen_probes.hpp, R4b —
// docs/rt_lighting_plan.md §5.5). Six passes, six cbuffers, three records.
// ===========================================================================

/**
 * One screen probe, as the placement pass writes it and every later pass reads
 * it. 32 B, mirrored by `GiScreenProbe` in shaders/gi_screen_probe.hlsli.
 *
 * `flags` carries the validity bit AND the anchor's in-tile pixel offset (four
 * bits per axis, which is exactly a 16-pixel tile). The offset is there so the
 * temporal filter can reproject the probe through the VELOCITY buffer at the
 * pixel the anchor actually sat on, rather than at the tile centre — which for
 * a probe anchored on a foreground silhouette is a different object.
 */
struct alignas(16) GiScreenProbeHeader {
    glm::vec3 worldPosition{0.0f};
    uint32_t flags = 0;
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    /// Distance from the camera, world units. The bilateral weight's depth term
    /// on both sides (spatial filter and forward gather). Zero means "no probe
    /// here" — a tile that was all sky.
    float cameraDistance = 0.0f;
};
BRACKEN_HLSL_CBUFFER(GiScreenProbeHeader, 32);
BRACKEN_HLSL_FIELD(GiScreenProbeHeader, worldPosition, 0);
BRACKEN_HLSL_FIELD(GiScreenProbeHeader, flags, 12);
BRACKEN_HLSL_FIELD(GiScreenProbeHeader, normal, 16);
BRACKEN_HLSL_FIELD(GiScreenProbeHeader, cameraDistance, 28);

/**
 * One ray the screen march could not resolve, appended by the march pass and
 * consumed by the INDIRECT-dispatched BVH pass. 16 B.
 *
 * The buffer's first 16 bytes are a header whose first word is the record
 * count; records start at byte 16 (`giSpMissRecordOffset`). That is why the
 * count is not in a buffer of its own: stage B is at SDL3 GPU's eight
 * read-only storage buffers exactly, and a ninth for a single uint would not
 * fit.
 */
/// Bytes reserved at the head of the miss buffer, before record 0. Word 0 is
/// the record count; the other three keep the records 16-byte aligned.
/// Mirrored as GI_SP_MISS_HEADER_BYTES in shaders/gi_screen_probe.hlsli.
inline constexpr uint32_t GI_SCREEN_PROBE_MISS_HEADER_BYTES = 16;

struct alignas(16) GiScreenProbeMiss {
    uint32_t probeIndex = 0;
    uint32_t texelIndex = 0;
    /// World distance already covered on screen before the handoff. The BVH ray
    /// starts here, so a ray that walked 30 m across a courtyard before leaving
    /// the frustum does not re-trace those 30 m.
    float resumeDistance = 0.0f;
    uint32_t _pad0 = 0;
};
BRACKEN_HLSL_CBUFFER(GiScreenProbeMiss, 16);
BRACKEN_HLSL_FIELD(GiScreenProbeMiss, probeIndex, 0);
BRACKEN_HLSL_FIELD(GiScreenProbeMiss, texelIndex, 4);
BRACKEN_HLSL_FIELD(GiScreenProbeMiss, resumeDistance, 8);
BRACKEN_HLSL_FIELD(GiScreenProbeMiss, _pad0, 12);

/**
 * The gather's diagnostic counters, zeroed by the placement pass, incremented
 * once per threadgroup by the march and trace passes, and read back through the
 * same fenced/polled download `ProbeVolume` uses. Lagged and honestly so —
 * nothing in the frame waits on them.
 */
struct alignas(16) GiScreenProbeCounters {
    uint32_t screenHits = 0;
    uint32_t worldHits = 0;
    uint32_t fallbacks = 0;
    /// Miss records the capacity clamp DROPPED. Should be zero: the buffer is
    /// sized for every traced ray missing (screen_probe_math.hpp), so a nonzero
    /// value means the sizing rule and the shader disagree.
    uint32_t dropped = 0;
};
BRACKEN_HLSL_CBUFFER(GiScreenProbeCounters, 16);
BRACKEN_HLSL_FIELD(GiScreenProbeCounters, screenHits, 0);
BRACKEN_HLSL_FIELD(GiScreenProbeCounters, worldHits, 4);
BRACKEN_HLSL_FIELD(GiScreenProbeCounters, fallbacks, 8);
BRACKEN_HLSL_FIELD(GiScreenProbeCounters, dropped, 12);

/**
 * The published per-probe L1 spherical harmonics: three coefficient quads (one
 * per colour channel) and one metadata quad. 64 B, mirrored by
 * `GiScreenProbeSh` in shaders/gi_screen_probe.hlsli.
 *
 * A STORAGE BUFFER, NOT A TEXTURE, and the reason is a hard platform limit
 * rather than taste. The design this implements called for one new fragment
 * SAMPLER in each scene shader (model/instanced 16 -> 17). SDL3 GPU refuses:
 * `SDL_CreateGPUShader` asserts "Shader sampler count cannot be higher than
 * 16!" (SDL_gpu.c:1208) and the pipeline is never created. Seventeen is DXC's
 * legalization ceiling, which is a different ceiling from SDL's, and SDL's is
 * the binding one.
 *
 * A buffer is also the better fit once forced: this is probe-INDEXED data read
 * with a plain index and never filtered — which is exactly what a texture bound
 * POINT-clamp and read with Load() was pretending not to be — and the scene
 * shaders had storage-buffer headroom (four of SDL's eight) where they had no
 * sampler headroom at all.
 */
struct alignas(16) GiScreenProbeSh {
    /// (c0, c1, c2, c3) of the red / green / blue channel.
    glm::vec4 shR{0.0f};
    glm::vec4 shG{0.0f};
    glm::vec4 shB{0.0f};
    /// xyz = the anchor's world normal, w = its camera distance. w == 0 is the
    /// "no probe here" signal (a tile that was all sky) and the ONLY one the
    /// forward gather gets, so it is written even for an invalid probe rather
    /// than left holding the previous frame's occupant.
    glm::vec4 meta{0.0f};
};
BRACKEN_HLSL_CBUFFER(GiScreenProbeSh, 64);
BRACKEN_HLSL_FIELD(GiScreenProbeSh, shR, 0);
BRACKEN_HLSL_FIELD(GiScreenProbeSh, shG, 16);
BRACKEN_HLSL_FIELD(GiScreenProbeSh, shB, 32);
BRACKEN_HLSL_FIELD(GiScreenProbeSh, meta, 48);

/** GiScreenProbePlaceUniforms (set 2, b0) — gi_screen_probe_place.comp.hlsl. */
struct alignas(16) GiScreenProbePlaceUniforms {
    /// Clip -> world of the DRAW (jittered) camera. It must be the jittered one:
    /// the depth buffer this pass unprojects was rendered with the jitter, and
    /// unprojecting it with the unjittered matrix offsets every anchor by the
    /// jitter's world-space equivalent — small, but systematically biased in a
    /// direction that changes every frame, which is the one thing a temporally
    /// filtered probe cannot absorb.
    glm::mat4 invViewProj{1.0f};
    glm::vec3 cameraPosition{0.0f};
    uint32_t frameIndex = 0;
    glm::ivec2 probeGrid{0};
    glm::ivec2 screenSize{0};
    uint32_t missCapacity = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
    uint32_t _pad2 = 0;
};
BRACKEN_HLSL_CBUFFER(GiScreenProbePlaceUniforms, 112);
BRACKEN_HLSL_FIELD(GiScreenProbePlaceUniforms, invViewProj, 0);
BRACKEN_HLSL_FIELD(GiScreenProbePlaceUniforms, cameraPosition, 64);
BRACKEN_HLSL_FIELD(GiScreenProbePlaceUniforms, frameIndex, 76);
BRACKEN_HLSL_FIELD(GiScreenProbePlaceUniforms, probeGrid, 80);
BRACKEN_HLSL_FIELD(GiScreenProbePlaceUniforms, screenSize, 88);
BRACKEN_HLSL_FIELD(GiScreenProbePlaceUniforms, missCapacity, 96);

/** GiScreenProbeMarchUniforms (set 2, b0) — gi_screen_probe_march.comp.hlsl. */
struct alignas(16) GiScreenProbeMarchUniforms {
    /// World -> clip, JITTERED: the march walks the depth buffer, and the depth
    /// buffer is the jittered projection's.
    glm::mat4 viewProj{1.0f};
    /// UNJITTERED previous frame, the pair the velocity buffer was built from —
    /// so a hit's reprojection into the TAA history agrees with how the history
    /// itself was resolved.
    glm::mat4 prevViewProjUnjittered{1.0f};
    glm::vec3 cameraPosition{0.0f};
    /// projection[2][2] of the draw projection (reversed-Z linearization).
    float depthA = 0.0f;
    glm::ivec2 probeGrid{0};
    glm::ivec2 screenSize{0};
    /// projection[3][2] of the draw projection.
    float depthB = 0.0f;
    /// How far a screen ray may travel in world units before it is handed off.
    float rayMaxDistance = 0.0f;
    /// Offset along the anchor normal before projecting the ray origin, world
    /// units. Without it every ray's first sample is the anchor's own surface.
    float normalBias = 0.0f;
    /// 1 when the TAA history holds a resolved previous frame. 0 sends EVERY
    /// ray straight to the BVH pass: there is no other source of screen
    /// radiance, and marching a pyramid to sample a texture that means nothing
    /// would be a confident wrong answer rather than a handoff.
    float historyValid = 0.0f;
    uint32_t frameIndex = 0;
    uint32_t mipCount = 0;
    uint32_t missCapacity = 0;
    uint32_t _pad0 = 0;
};
BRACKEN_HLSL_CBUFFER(GiScreenProbeMarchUniforms, 192);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, viewProj, 0);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, prevViewProjUnjittered, 64);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, cameraPosition, 128);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, depthA, 140);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, probeGrid, 144);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, screenSize, 152);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, depthB, 160);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, rayMaxDistance, 164);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, normalBias, 168);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, historyValid, 172);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, frameIndex, 176);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, mipCount, 180);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, missCapacity, 184);
BRACKEN_HLSL_FIELD(GiScreenProbeMarchUniforms, _pad0, 188);

/** GiScreenProbeArgsUniforms (set 2, b0) — gi_screen_probe_args.comp.hlsl. */
struct alignas(16) GiScreenProbeArgsUniforms {
    uint32_t missCapacity = 0;
    /// The BVH pass's [numthreads].x. Sent rather than duplicated as a literal:
    /// the args pass computes ceil(count / threadsPerGroup) for a DIFFERENT
    /// shader's threadgroup size, and hardcoding another shader's constant is
    /// exactly the kind of thing that survives a refactor of that shader.
    uint32_t threadsPerGroup = 64;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
};
BRACKEN_HLSL_CBUFFER(GiScreenProbeArgsUniforms, 16);
BRACKEN_HLSL_FIELD(GiScreenProbeArgsUniforms, missCapacity, 0);
BRACKEN_HLSL_FIELD(GiScreenProbeArgsUniforms, threadsPerGroup, 4);
BRACKEN_HLSL_FIELD(GiScreenProbeArgsUniforms, _pad0, 8);
BRACKEN_HLSL_FIELD(GiScreenProbeArgsUniforms, _pad1, 12);

/**
 * GiScreenProbeTraceUniforms (set 2, b0) — gi_screen_probe_trace.comp.hlsl.
 *
 * Leading 336 bytes ARE `ShadowDataUniforms`, for the same reason
 * `GiProbeTraceUniforms`'s are: the pass includes `shadow_common.hlsli` and
 * calls the forward pass's own `calcShadow()` rather than transliterating the
 * standard-Z cascade math a second time.
 */
struct alignas(16) GiScreenProbeTraceUniforms {
    ShadowDataUniforms shadow{};
    glm::vec3 cameraPosition{0.0f};
    /// GI_SP_WORLD_RAY_CLAMP — the handed-off ray's own length, NOT the screen
    /// march's range.
    float rayMaxDistance = 600.0f;
    glm::vec3 cameraForward{0.0f, 0.0f, -1.0f};
    float normalBias = 1.0f;
    glm::vec3 sunDirection{0.0f, -1.0f, 0.0f};
    float bounceScale = 1.0f;
    glm::vec3 sunColor{0.0f};
    float useSkyCube = 0.0f;
    glm::vec3 ambientSky{0.0f};
    float giEnabled = 1.0f;
    glm::vec3 ambientGround{0.0f};
    /// Pinned to 1: a screen-probe ray that escapes the clamp is shaded by the
    /// sky/field blend below, and where the field has no answer the engine's
    /// documented no-GI ambient tier — the hemisphere — is what is left.
    float giAmbientFallback = 1.0f;
    glm::ivec4 cascadeBase[3]{};
    uint32_t tlasNodeCount = 0;
    uint32_t lightCount = 0;
    uint32_t lightCapacity = 0;
    uint32_t missCapacity = 0;
    glm::ivec2 probeGrid{0};
    uint32_t frameIndex = 0;
    /// Strength of the downward blend from sky radiance toward the probe
    /// field's mean radiance on a ray that escaped the clamp. 1 = the full
    /// heuristic; 0 = sky only. See rtScreenProbeFallback.
    float horizonBlend = 1.0f;
};
BRACKEN_HLSL_CBUFFER(GiScreenProbeTraceUniforms, 512);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, shadow, 0);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, cameraPosition, 336);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, rayMaxDistance, 348);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, cameraForward, 352);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, normalBias, 364);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, sunDirection, 368);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, bounceScale, 380);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, sunColor, 384);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, useSkyCube, 396);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, ambientSky, 400);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, giEnabled, 412);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, ambientGround, 416);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, giAmbientFallback, 428);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, cascadeBase, 432);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, tlasNodeCount, 480);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, lightCount, 484);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, lightCapacity, 488);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, missCapacity, 492);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, probeGrid, 496);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, frameIndex, 504);
BRACKEN_HLSL_FIELD(GiScreenProbeTraceUniforms, horizonBlend, 508);

/** GiScreenProbeFilterUniforms (set 2, b0) — gi_screen_probe_filter.comp.hlsl. */
struct alignas(16) GiScreenProbeFilterUniforms {
    glm::ivec2 probeGrid{0};
    glm::ivec2 screenSize{0};
    float alpha = 0.1f;
    /// 0 on the first frame at a size and after any reallocation: there is no
    /// previous atlas, so the EMA takes this frame whole.
    float historyValid = 0.0f;
    float fireflyMax = 8.0f;
    uint32_t frameIndex = 0;
    /// R5b: 1 when `exposureState` holds a live adapted EV, 0 to take
    /// `manualExposureEv`. The firefly ceiling's absolute floor is a
    /// POST-EXPOSURE luminance, so it has to be told which exposure.
    uint32_t exposureEnabled = 0;
    float manualExposureEv = 0.0f;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
};
BRACKEN_HLSL_CBUFFER(GiScreenProbeFilterUniforms, 48);
BRACKEN_HLSL_FIELD(GiScreenProbeFilterUniforms, probeGrid, 0);
BRACKEN_HLSL_FIELD(GiScreenProbeFilterUniforms, screenSize, 8);
BRACKEN_HLSL_FIELD(GiScreenProbeFilterUniforms, alpha, 16);
BRACKEN_HLSL_FIELD(GiScreenProbeFilterUniforms, historyValid, 20);
BRACKEN_HLSL_FIELD(GiScreenProbeFilterUniforms, fireflyMax, 24);
BRACKEN_HLSL_FIELD(GiScreenProbeFilterUniforms, frameIndex, 28);
BRACKEN_HLSL_FIELD(GiScreenProbeFilterUniforms, exposureEnabled, 32);
BRACKEN_HLSL_FIELD(GiScreenProbeFilterUniforms, manualExposureEv, 36);
BRACKEN_HLSL_FIELD(GiScreenProbeFilterUniforms, _pad0, 40);
BRACKEN_HLSL_FIELD(GiScreenProbeFilterUniforms, _pad1, 44);

/** GiScreenProbeProjectUniforms (set 2, b0) — gi_screen_probe_project.comp.hlsl. */
struct alignas(16) GiScreenProbeProjectUniforms {
    glm::ivec2 probeGrid{0};
    float normalPower = 8.0f;
    uint32_t frameIndex = 0;
};
BRACKEN_HLSL_CBUFFER(GiScreenProbeProjectUniforms, 16);
BRACKEN_HLSL_FIELD(GiScreenProbeProjectUniforms, probeGrid, 0);
BRACKEN_HLSL_FIELD(GiScreenProbeProjectUniforms, normalPower, 8);
BRACKEN_HLSL_FIELD(GiScreenProbeProjectUniforms, frameIndex, 12);

} // namespace bracken::gpu
