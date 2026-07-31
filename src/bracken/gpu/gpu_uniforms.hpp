#pragma once

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
};

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
 * Instanced model vertex shader camera uniforms - matches the CameraUniforms
 * cbuffer of model_instanced.vert.hlsl and particle.vert.hlsl (set 1, b0)
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
    // glTF alphaMode MASK cutoff; 0 disables the alpha test (OPAQUE/BLEND).
    float alphaCutoff = 0.0f;
    // 1 / render-target size — see ModelLightingUniforms::screenAOTexelX.
    float screenAOTexelX = 0.0f;
    float screenAOTexelY = 0.0f;
    // Sky IBL flag — see ModelLightingUniforms::iblEnabled.
    float iblEnabled = 0.0f;
    // Screen-AO gate — see ModelLightingUniforms::screenAOEnabled.
    float screenAOEnabled = 1.0f;
};

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
};

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
 * GTAO pass fragment uniforms (set 3, b0) — matches gtao.frag.hlsl.
 *
 * Horizon-based ground-truth AO (Jimenez et al. 2016): per-slice horizon
 * search with cosine-weighted arc integration. Slice/step counts are
 * compile-time constants in the shader (SLICE_COUNT / STEPS_PER_SLICE).
 *
 * Tuning: render::GTAO assigns projection/screen sizes, radiusScale
 * (GraphicsSettings::ao_radius), and the temporal jitter per frame. The
 * remaining NSDMI defaults ARE the live values (GTAO-style engine constants);
 * changing them requires a rebuild. maxDistance must stay equal to
 * BlurUniforms::maxDistance (pinned by a test in gtao.test.cpp).
 */
struct alignas(16) GTAOUniforms {
    glm::mat4 projection;
    glm::mat4 invProjection;
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
    // Thickness heuristic: when a sample lowers the running horizon, blend
    // toward it by this factor — models finite occluder thickness so thin
    // features don't cast unbounded occlusion. 0 = pure max (infinitely
    // thick occluders).
    float thinOccluderCompensation = 0.2f;
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

/**
 * GTAO upsample/finalize pass fragment uniforms (set 3, b0) — matches
 * gtao_upsample.frag.hlsl. Joint-bilateral (depth-aware) upsample of the
 * working-res blurred AO to the full-res R8 texture the forward pass samples;
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

/**
 * Per-particle data for instanced particle rendering (storage buffer).
 * Matches particle.vert.hlsl ParticleInstance: color.rgb tints, color.a is
 * opacity (folded into rgb by the vertex stage - additive blending).
 */
struct ParticleInstanceData {
    glm::mat4 model;
    glm::vec4 color;
};

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
};

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

// NOTE: the clustered-lighting cbuffer layout (clusters.hlsli) is defined by
// render::lighting::ClusterParams (light_cluster.hpp) — SceneRenderer pushes
// that struct directly; there is deliberately no mirror here.

} // namespace bracken::gpu
