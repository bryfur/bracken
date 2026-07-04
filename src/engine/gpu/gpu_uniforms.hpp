#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace mmo::engine::gpu {

/**
 * Model vertex shader uniforms - matches model.vert.hlsl /
 * skinned_model.vert.hlsl TransformUniforms (set 1, b0).
 * Used by: renderer.cpp, world_renderer.cpp, effect_renderer.cpp
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
    int useSkinning = 0;
    uint32_t paletteOffset = 0;
    float _padding1[2] = {0.0f, 0.0f};
};

/**
 * Model fragment shader uniforms - matches model.frag.hlsl LightingUniforms (set 3, b0)
 * Used by: scene_renderer.cpp, effect_renderer.cpp
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
    float _padding4 = 0.0f;
};

/**
 * Skybox fragment shader uniforms - matches skybox.frag.hlsl SkyUniforms (set 3, b0)
 * Colors are linear, sourced from EnvironmentState (render/environment.hpp).
 */
struct alignas(16) SkyboxFragmentUniforms {
    glm::mat4 invViewProjection;
    glm::vec3 sunDirection; // direction TO the sun
    float time;
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
};

/**
 * Grid vertex shader uniforms (set 1, b0)
 */
struct alignas(16) GridVertexUniforms {
    glm::mat4 viewProjection;
};

/**
 * UI vertex shader uniforms (set 1, b0)
 */
struct alignas(16) UIScreenUniforms {
    float width{};
    float height{};
    float _padding[2] = {0.0f, 0.0f};
};

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

/**
 * Instanced model vertex shader camera uniforms - matches model_instanced.vert.hlsl (set 1, b0)
 */
struct alignas(16) InstancedCameraUniforms {
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 cameraPos;
    float _padding0 = 0.0f;
};

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
    float _padding4 = 0.0f;
};

/**
 * Per-instance data for instanced model rendering (storage buffer).
 */
struct InstanceData {
    glm::mat4 model;
    glm::mat4 normalMatrix;
    glm::vec4 tint;
    float noFog;
    float _pad[3] = {0.0f, 0.0f, 0.0f};
};

/**
 * Per-instance data for instanced shadow rendering (storage buffer).
 */
struct ShadowInstanceData {
    glm::mat4 model;
};

/**
 * Instanced shadow vertex uniforms (set 1, b0)
 */
struct alignas(16) InstancedShadowUniforms {
    glm::mat4 lightViewProjection;
};

/**
 * Shadow depth pass - vertex uniforms for static models (set 1, b0)
 */
struct alignas(16) ShadowTransformUniforms {
    glm::mat4 lightViewProjection;
    glm::mat4 model;
};

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

/**
 * Shadow depth pass - vertex uniforms for terrain (set 1, b0)
 * Terrain vertices are already in world space, no model matrix needed.
 */
struct alignas(16) ShadowTerrainUniforms {
    glm::mat4 lightViewProjection;
};

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

/**
 * GTAO pass fragment uniforms (set 3, b0). Shared by ssao.frag and gtao.frag.
 *
 * sampleKernel is precomputed once on the CPU (3 hashes + acos/sin/cos per
 * sample per pixel was the old in-shader cost). Contents differ per pass:
 * - SSAO: xyz = hemisphere direction * per-sample length in [0,1], w unused.
 * - GTAO: xy = evenly spaced unit-circle direction, zw unused (first
 *   numDirections entries used). Both passes rotate per-pixel via IGN.
 */
struct alignas(16) GTAOUniforms {
    static constexpr int KERNEL_SIZE = 24;

    glm::mat4 projection;
    glm::mat4 invProjection;
    glm::vec2 screenSize;
    glm::vec2 invScreenSize;
    // World/view-unit sample radius. Characters here are 25-35 units tall, so
    // sub-2.0 radii only catch sub-ankle contact; 4.0 reads as proper corner
    // and crevice darkening. Scaled by the ao_radius settings multiplier.
    float radius = 4.0f;
    float bias = 0.01f;
    int numDirections = 6;
    int numSteps = 3;
    float radiusScale = 1.0f;
    float _padding[3] = {0.0f, 0.0f, 0.0f};
    glm::vec4 sampleKernel[KERNEL_SIZE]{};
};

/**
 * Bilateral blur pass fragment uniforms (set 3, b0)
 * depthA/depthB linearize the non-linear depth buffer in the shader:
 * viewZ = depthB / (depth + depthA) with depthA = proj[2][2], depthB = proj[3][2]
 * (glm RH_ZO). Weighting on relative linear view-Z keeps edge rejection
 * consistent at distance (raw depth differences collapse far away -> halos).
 */
struct alignas(16) BlurUniforms {
    glm::vec2 direction;
    glm::vec2 invScreenSize;
    float sharpness = 40.0f;
    float depthA = 0.0f;
    float depthB = 0.0f;
    float _padding = 0.0f;
};

/**
 * Composite pass fragment uniforms (set 3, b0)
 */
struct alignas(16) CompositeUniforms {
    float aoStrength = 1.0f;
    float bloomStrength = 0.35f;
    float volumetricFogEnabled = 0.0f;
    float _padding = 0.0f;
    float exposure = 1.0f;
    int tonemapMode = 0;
    float contrast = 1.0f;
    float saturation = 1.0f;
};

/**
 * FXAA pass fragment uniforms (set 3, b0)
 * Operates on the post-tonemap gamma-encoded LDR image.
 */
struct alignas(16) FxaaUniforms {
    glm::vec2 invScreenSize;
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
};

/**
 * Particle fragment shader uniforms - matches particle.frag.hlsl (set 3, b0).
 * Additive (ONE, ONE) blending: fade is encoded by scaling rgb via opacity,
 * the alpha output is unused.
 */
struct alignas(16) ParticleFragmentUniforms {
    glm::vec4 tintColor{1.0f, 1.0f, 1.0f, 1.0f};
    float emissiveBoost = 1.0f;
    float opacity = 1.0f;
    float _pad0 = 0.0f;
    float _pad1 = 0.0f;
};

/**
 * Bloom downsample pass fragment uniforms (set 3, b0)
 */
struct alignas(16) BloomDownsampleUniforms {
    glm::vec2 srcTexelSize;
    float threshold = 0.8f;
    float isFirstPass = 1.0f;
};

/**
 * Bloom upsample pass fragment uniforms (set 3, b0)
 */
struct alignas(16) BloomUpsampleUniforms {
    glm::vec2 srcTexelSize;
    float bloomRadius = 1.0f;
    float _padding = 0.0f;
};

/**
 * Volumetric fog pass fragment uniforms (set 3, b0)
 */
struct alignas(16) VolumetricFogUniforms {
    glm::mat4 invViewProjection{};
    glm::mat4 shadowLightViewProjection{};
    glm::vec3 lightDir{};
    float fogDensity = 0.02f;
    glm::vec3 lightColor{};
    float scatterStrength = 0.3f;
    glm::vec3 fogColor{};
    float fogHeight = 50.0f;
    glm::vec3 cameraPos{};
    float fogFalloff = 0.01f;
    float nearPlane = 0.1f;
    float farPlane = 5000.0f;
    float godRaysEnabled = 1.0f;
    float densityMultiplier = 1.0f;
    float shadowAtlasTexel = 0.0f; // 1 / shadow atlas width (cascade 0 = top-left quadrant)
    float shadowBias = 0.001f;     // depth compare bias in light-space [0,1] units
    float _padding0 = 0.0f;
    float _padding1 = 0.0f;
};

/**
 * Clustered lighting fragment uniforms (set 3, b2).
 * Mirrors render::lighting::ClusterParams. Held here so it can be referenced
 * without dragging the lighting module into headers that only need GPU layouts.
 */
struct alignas(16) ClusterParamsUniforms {
    glm::mat4 view;
    glm::mat4 invProjection;
    glm::vec4 screenSize;     // (width, height, 1/width, 1/height)
    glm::vec4 zPlanes;        // (near, far, log(far/near), 1/log(far/near))
    glm::uvec4 gridDim;       // (X, Y, Z, totalLightCount)
    glm::uvec4 maxPerCluster; // (max_lights_per_cluster, _, _, _)
};

} // namespace mmo::engine::gpu
