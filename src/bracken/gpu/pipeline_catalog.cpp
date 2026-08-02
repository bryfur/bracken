#include "pipeline_catalog.hpp"

#include "bracken/gpu/gpu_types.hpp"
#include <array>
#include <cstddef>

namespace bracken::gpu {

namespace {

// =============================================================================
// Shared resource declarations
// =============================================================================
// Named once so that the counts a shader family needs cannot drift between the
// pipelines that share the shader. `model.frag` in particular is used by five
// rows below; its sampler inventory used to be written out (with its comment)
// three separate times.

constexpr ShaderResources k_none{};
constexpr ShaderResources k_ubo1{.num_uniform_buffers = 1};
constexpr ShaderResources k_ubo1_sb1{.num_storage_buffers = 1, .num_uniform_buffers = 1};
constexpr ShaderResources k_ubo1_sb2{.num_storage_buffers = 2, .num_uniform_buffers = 1};
constexpr ShaderResources k_ubo2_sb1{.num_storage_buffers = 1, .num_uniform_buffers = 2};
constexpr ShaderResources k_ubo1_samp1{.num_samplers = 1, .num_uniform_buffers = 1};
constexpr ShaderResources k_ubo1_samp2{.num_samplers = 2, .num_uniform_buffers = 1};
constexpr ShaderResources k_ubo1_samp4{.num_samplers = 4, .num_uniform_buffers = 1};
constexpr ShaderResources k_ubo1_samp5{.num_samplers = 5, .num_uniform_buffers = 1};

/// `model.frag` / `model_instanced.frag`: baseColor + shadow atlas (cmp + raw)
/// + normal + metallic-roughness + emissive + screen AO + IBL (irradiance /
/// prefiltered / BRDF LUT) + traced sun-shadow mask + GI (irradiance /
/// visibility / probe metadata) + traced LOCAL-light mask + local-shadow atlas
/// = 16 samplers; light data
/// + cluster offsets + light indices + local-shadow slots = 4 storage buffers;
/// lighting + shadow data + cluster params = 3 uniform buffers.
///
/// The RT sun mask is slot 10, the probe field occupies 11-13, R2b's traced
/// LOCAL-light visibility ratio is 14, and the rasterized local-shadow atlas
/// moved to 15 (`CLUSTER_SSBO_BINDING_BASE` = 16). The
/// atlas stays LAST because cluster_lighting.hlsli declares it at
/// `CLUSTER_SSBO_BINDING_BASE - 1`, which is the contract every one of these
/// three rows depends on.
///
/// The RT mask was previously multiplexed into the screen-AO texture's G
/// channel on the belief that a 12th sampler failed DXC legalization; that does
/// not reproduce — model.frag compiles clean at 12, and was verified up to 17,
/// which is also the headroom R1c's three GI slots were spent out of.
constexpr ShaderResources k_fs_lit_model{
    .num_samplers = 16, .num_storage_buffers = 4, .num_uniform_buffers = 3};

/// `terrain.frag`: material array + splatmap + shadow atlas (cmp + raw) +
/// screen AO + IBL (irradiance / prefiltered / BRDF LUT) + traced sun mask +
/// GI (irradiance / visibility / probe metadata) + traced LOCAL-light mask +
/// local-shadow atlas.
constexpr ShaderResources k_fs_lit_terrain{
    .num_samplers = 14, .num_storage_buffers = 4, .num_uniform_buffers = 3};

/// `grass.frag`: shadow atlas (cmp + raw) + screen AO + IBL (irradiance /
/// prefiltered / BRDF LUT) + traced sun mask + GI (irradiance / visibility /
/// probe metadata) + traced LOCAL-light mask + local-shadow atlas.
constexpr ShaderResources k_fs_lit_grass{
    .num_samplers = 12, .num_storage_buffers = 4, .num_uniform_buffers = 3};

/// `grass.vert` / `prepass_grass.vert`: heightmap + splatmap samplers, plus the
/// visible-chunk origin buffer for the chunked LOD.
constexpr ShaderResources k_vs_grass{
    .num_samplers = 2, .num_storage_buffers = 1, .num_uniform_buffers = 1};

/// `composite.frag`: color + bloom + volumetric fog + depth, plus the
/// auto-exposure state buffer. Reflections have already been resolved into
/// scene colour before TAA; keeping them out of this final pass is what lets
/// TAA/bloom/exposure see the traced signal.
// This count MUST match the shader — a
// stale value here does not fail loudly, it misbinds the storage buffer that
// follows the samplers and quietly breaks tonemapping.
constexpr ShaderResources k_fs_composite{
    .num_samplers = 4, .num_storage_buffers = 1, .num_uniform_buffers = 1};

// =============================================================================
// Shared fixed-function constants
// =============================================================================

/// The one shadow-caster rasterizer bias. Receiver-side slope/normal-offset
/// biasing carries most of the acne fight, so this stays small to avoid
/// peter-panning. Four creators used to carry their own copy of the pair.
constexpr DepthBias k_shadow_bias{.constant = 1.25f, .slope = 1.75f, .enabled = true};

/// Terrain vertex stride: position(3) + texcoord(2) + color(4) + normal(3).
constexpr uint32_t k_terrain_vertex_stride = sizeof(float) * 12;

// =============================================================================
// The catalogue
// =============================================================================
// Rows are in PipelineType declaration order — the table is indexed by the
// enum, so a row out of place is a wrong pipeline, and the static_assert below
// catches a row that is missing entirely.

constexpr std::array<PipelineDesc, static_cast<size_t>(PipelineType::Count)> k_pipeline_descs{{
    // --- 3D scene ------------------------------------------------------------
    {.name = "Model",
     .vertex_shader = "model.vert.spv",
     .fragment_shader = "model.frag.spv",
     .vs = k_ubo1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::Vertex3D},
    {.name = "SkinnedModel",
     .vertex_shader = "skinned_model.vert.spv",
     .fragment_shader = "model.frag.spv",
     .vs = k_ubo1_sb1, // transform uniforms + tightly packed bone palettes
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::SkinnedVertex},
    {.name = "Terrain",
     .vertex_shader = "terrain.vert.spv",
     .fragment_shader = "terrain.frag.spv",
     .vs = k_ubo1,
     .fs = k_fs_lit_terrain,
     .vertex_layout = VertexLayout::Terrain},
    // Reversed-Z: skybox.vert emits z = 0.0 (the far plane) and
    // GREATER_OR_EQUAL passes only where the depth buffer still holds the 0.0
    // clear value, i.e. where no geometry was drawn.
    {.name = "Skybox",
     .vertex_shader = "skybox.vert.spv",
     .fragment_shader = "skybox.frag.spv",
     .vs = k_none, // fullscreen-ish cube, no transforms
     .fs = k_ubo1,
     .vertex_layout = VertexLayout::Position3,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .depth_write = false,
     .depth_compare = SDL_GPU_COMPAREOP_GREATER_OR_EQUAL},
    {.name = "Grid",
     .vertex_shader = "grid.vert.spv",
     .fragment_shader = "grid.frag.spv",
     .vs = k_ubo1,
     .fs = k_none,
     .vertex_layout = VertexLayout::Grid,
     .primitive_type = SDL_GPU_PRIMITIVETYPE_LINELIST,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .blend_mode = BlendMode::Alpha,
     .depth_write = false},

    // --- 2D / UI (swapchain targets, no depth) --------------------------------
    {.name = "UI",
     .vertex_shader = "ui.vert.spv",
     .fragment_shader = "ui.frag.spv",
     .vs = k_ubo1,       // screen size
     .fs = k_ubo1_samp1, // has_texture flag + texture
     .vertex_layout = VertexLayout::Vertex2D,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .blend_mode = BlendMode::Alpha,
     .color = TargetFormat::Swapchain,
     .depth_target = false},
    {.name = "Text",
     .vertex_shader = "text.vert.spv",
     .fragment_shader = "text.frag.spv",
     .vs = k_ubo1,       // projection
     .fs = k_ubo1_samp1, // text color + font atlas
     .vertex_layout = VertexLayout::Text,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .blend_mode = BlendMode::Alpha,
     .color = TargetFormat::Swapchain,
     .depth_target = false},

    // --- Effects --------------------------------------------------------------
    // BlendMode::Additive is (SRC_ALPHA, ONE); particle.frag writes alpha 1.0
    // so the effective color blend is exactly (ONE, ONE). Depth test on with
    // writes off so particles occlude against opaques but not each other.
    {.name = "Particle",
     .vertex_shader = "particle.vert.spv",
     .fragment_shader = "particle.frag.spv",
     .vs = k_ubo2_sb1,   // camera (b0) + batch instance offset (b1) + instances
     .fs = k_ubo1_samp1, // ParticleFragmentUniforms + baseColor
     .vertex_layout = VertexLayout::Vertex3D,
     .blend_mode = BlendMode::Additive,
     .depth_write = false,
     .depth_compare = SDL_GPU_COMPAREOP_GREATER_OR_EQUAL},
    {.name = "Grass",
     .vertex_shader = "grass.vert.spv",
     .fragment_shader = "grass.frag.spv",
     .vs = k_vs_grass,
     .fs = k_fs_lit_grass,
     .vertex_layout = VertexLayout::Vertex3D,
     .cull_mode = SDL_GPU_CULLMODE_NONE}, // blades are double-sided

    // --- Instanced ------------------------------------------------------------
    {.name = "InstancedModel",
     .vertex_shader = "model_instanced.vert.spv",
     .fragment_shader = "model_instanced.frag.spv",
     .vs = k_ubo1_sb1, // camera uniforms + instance data
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::Vertex3D},
    {.name = "InstancedShadowModel",
     .vertex_shader = "shadow_depth_instanced.vert.spv",
     .fragment_shader = "shadow_depth.frag.spv",
     .vs = k_ubo1_sb1,   // lightViewProjection + instance model matrices
     .fs = k_ubo1_samp1, // hasTexture flag + alpha texture
     .vertex_layout = VertexLayout::Vertex3D,
     .cull_mode = SDL_GPU_CULLMODE_FRONT,
     .depth_compare = SDL_GPU_COMPAREOP_LESS_OR_EQUAL, // cascades stay standard-Z
     .depth_clip = false,
     .depth_bias = k_shadow_bias,
     .color = TargetFormat::DepthOnly},

    // --- Shadow casters (standard-Z, depth-only) ------------------------------
    // The cascades' orthographic light projections have no reversed-Z precision
    // problem, and the atlas clear, comparison sampler and receiver math all
    // assume near=0/far=1 — so every row here is LESS_OR_EQUAL by design.
    {.name = "ShadowModel",
     .vertex_shader = "shadow_depth.vert.spv",
     .fragment_shader = "shadow_depth.frag.spv",
     .vs = k_ubo1,       // lightVP + model
     .fs = k_ubo1_samp1, // hasTexture flag + alpha texture
     .vertex_layout = VertexLayout::Vertex3D,
     .cull_mode = SDL_GPU_CULLMODE_FRONT,
     .depth_compare = SDL_GPU_COMPAREOP_LESS_OR_EQUAL,
     .depth_clip = false,
     .depth_bias = k_shadow_bias,
     .color = TargetFormat::DepthOnly},
    {.name = "ShadowSkinnedModel",
     .vertex_shader = "shadow_skinned.vert.spv",
     .fragment_shader = "shadow_depth_opaque.frag.spv",
     .vs = k_ubo1_sb1, // lightVP + model + palette offset, + bone palettes
     .fs = k_none,
     .vertex_layout = VertexLayout::SkinnedVertex,
     .cull_mode = SDL_GPU_CULLMODE_FRONT,
     .depth_compare = SDL_GPU_COMPAREOP_LESS_OR_EQUAL,
     .depth_clip = false,
     .depth_bias = k_shadow_bias,
     .color = TargetFormat::DepthOnly},
    // Terrain is single-sided (upward-facing only), so culling has to be off
    // for its triangles to reach the shadow map at all.
    {.name = "ShadowTerrain",
     .vertex_shader = "shadow_terrain.vert.spv",
     .fragment_shader = "shadow_depth_opaque.frag.spv",
     .vs = k_ubo1,
     .fs = k_none,
     .vertex_layout = VertexLayout::Terrain,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .depth_compare = SDL_GPU_COMPAREOP_LESS_OR_EQUAL,
     .depth_clip = false,
     .depth_bias = k_shadow_bias,
     .color = TargetFormat::DepthOnly},
    // Local-shadow atlas tiles persist across frames (the atlas pass LOADs), so
    // a dirty tile is "cleared" by drawing the fullscreen triangle over its
    // viewport with the depth range pinned to [1,1] and an ALWAYS depth test.
    {.name = "LocalShadowClear",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "shadow_depth_opaque.frag.spv",
     .vs = k_none,
     .fs = k_none,
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .depth_compare = SDL_GPU_COMPAREOP_ALWAYS,
     .color = TargetFormat::DepthOnly},

    // --- Post-process (fullscreen triangle, no vertex input, no depth) --------
    {.name = "GTAO",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "gtao.frag.spv",
     .vs = k_none,
     .fs = k_ubo1_samp1, // GTAOUniforms + depth
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::R8,
     .depth_target = false},
    // RG16F history: R = accumulated AO, G = linear view depth (disocclusion).
    {.name = "GTAOTemporal",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "gtao_temporal.frag.spv",
     .vs = k_none,
     .fs = k_ubo1_samp4, // raw AO + history + velocity + depth
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::RG16F,
     .depth_target = false},
    {.name = "GTAOUpsample",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "gtao_upsample.frag.spv",
     .vs = k_none,
     .fs = k_ubo1_samp2, // blurred AO + depth
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::R8,
     .depth_target = false},
    {.name = "BlurAO",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "blur_ao.frag.spv",
     .vs = k_none,
     .fs = k_ubo1_samp2, // AO + depth
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::R8,
     .depth_target = false},
    {.name = "RtReflectionResolve",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "rt_reflection_resolve.frag.spv",
     .vs = k_none,
     .fs = {.num_samplers = 3, .num_uniform_buffers = 1},
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .blend_mode = BlendMode::AdditivePreserveAlpha,
     .color = TargetFormat::RGBA16F,
     .depth_target = false},
    {.name = "Composite",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "composite.frag.spv",
     .vs = k_none,
     .fs = k_fs_composite,
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Swapchain,
     .depth_target = false},
    {.name = "BloomDownsample",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "bloom_downsample.frag.spv",
     .vs = k_none,
     .fs = k_ubo1_samp1,
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::RGBA16F,
     .depth_target = false},
    // Progressive upsample: each pass tent-filters the smaller mip and ADDS it
    // onto the larger one (LOADOP_LOAD on the target). The shader writes alpha
    // 1.0 so Additive (SRC_ALPHA, ONE) is effectively (ONE, ONE).
    {.name = "BloomUpsample",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "bloom_upsample.frag.spv",
     .vs = k_none,
     .fs = k_ubo1_samp1,
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .blend_mode = BlendMode::Additive,
     .color = TargetFormat::RGBA16F,
     .depth_target = false},
    {.name = "VolumetricFog",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "volumetric_fog.frag.spv",
     .vs = k_none,
     .fs = k_ubo1_samp5, // depth + shadow atlas + the GI field's three atlases
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::RGBA16F,
     .depth_target = false},
    {.name = "Fxaa",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "fxaa.frag.spv",
     .vs = k_none,
     .fs = k_ubo1_samp1, // LDR composite output
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Swapchain,
     .depth_target = false},
    // Writes the history buffer, same format as the HDR scene target.
    {.name = "TaaResolve",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "taa_resolve.frag.spv",
     .vs = k_none,
     .fs = k_ubo1_samp4, // scene color + history + velocity + depth
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::SceneColor,
     .depth_target = false},
    // Writes back over the HDR scene target for bloom/composite.
    {.name = "TaaSharpen",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "taa_sharpen.frag.spv",
     .vs = k_none,
     .fs = k_ubo1_samp1, // resolved history
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::SceneColor,
     .depth_target = false},

    // --- IBL prefilter chain --------------------------------------------------
    {.name = "SkyCapture",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "sky_capture.frag.spv",
     .vs = k_none,
     .fs = k_ubo1,
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::RGBA16F, // one face of the capture cube
     .depth_target = false},
    {.name = "IblSpecularPrefilter",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "ibl_prefilter.frag.spv",
     .vs = k_none,
     .fs = k_ubo1_samp1, // IblFilterUniforms + sky-capture cube
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::RGBA16F, // one (face, mip) of the specular cube
     .depth_target = false},
    {.name = "IblIrradiance",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "ibl_irradiance.frag.spv",
     .vs = k_none,
     .fs = k_ubo1_samp1,
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::RGBA16F, // one face of the irradiance cube
     .depth_target = false},
    {.name = "IblBrdfLut",
     .vertex_shader = "fullscreen.vert.spv",
     .fragment_shader = "ibl_brdf_lut.frag.spv",
     .vs = k_none,
     .fs = k_none, // fully procedural
     .vertex_layout = VertexLayout::None,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::RG16F, // split-sum (A, B) pair per texel
     .depth_target = false},

    // --- Model-family material variants ---------------------------------------
    // Same shader pair as their base row; glTF doubleSided drops culling and
    // alphaMode BLEND turns on alpha blending with depth writes off (the host
    // sorts these back-to-front after all opaque content).
    {.name = "ModelDoubleSided",
     .vertex_shader = "model.vert.spv",
     .fragment_shader = "model.frag.spv",
     .vs = k_ubo1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::Vertex3D,
     .cull_mode = SDL_GPU_CULLMODE_NONE},
    {.name = "ModelBlend",
     .vertex_shader = "model.vert.spv",
     .fragment_shader = "model.frag.spv",
     .vs = k_ubo1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::Vertex3D,
     .blend_mode = BlendMode::Alpha,
     .depth_write = false},
    {.name = "ModelBlendDoubleSided",
     .vertex_shader = "model.vert.spv",
     .fragment_shader = "model.frag.spv",
     .vs = k_ubo1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::Vertex3D,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .blend_mode = BlendMode::Alpha,
     .depth_write = false},
    {.name = "SkinnedModelDoubleSided",
     .vertex_shader = "skinned_model.vert.spv",
     .fragment_shader = "model.frag.spv",
     .vs = k_ubo1_sb1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::SkinnedVertex,
     .cull_mode = SDL_GPU_CULLMODE_NONE},
    {.name = "SkinnedModelBlend",
     .vertex_shader = "skinned_model.vert.spv",
     .fragment_shader = "model.frag.spv",
     .vs = k_ubo1_sb1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::SkinnedVertex,
     .blend_mode = BlendMode::Alpha,
     .depth_write = false},
    {.name = "SkinnedModelBlendDoubleSided",
     .vertex_shader = "skinned_model.vert.spv",
     .fragment_shader = "model.frag.spv",
     .vs = k_ubo1_sb1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::SkinnedVertex,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .blend_mode = BlendMode::Alpha,
     .depth_write = false},
    {.name = "InstancedModelDoubleSided",
     .vertex_shader = "model_instanced.vert.spv",
     .fragment_shader = "model_instanced.frag.spv",
     .vs = k_ubo1_sb1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::Vertex3D,
     .cull_mode = SDL_GPU_CULLMODE_NONE},

    // --- Depth prepass --------------------------------------------------------
    // All prepass rows share prepass_velocity.frag (zero fragment resources),
    // render into the RG16F velocity target plus the scene depth target, and
    // keep the scene's reversed-Z GREATER_OR_EQUAL compare with depth writes
    // ON. Vertex strides match the forward pass; only the position (and
    // skinning) attributes are declared.
    {.name = "DepthPrepassModel",
     .vertex_shader = "prepass_model.vert.spv",
     .fragment_shader = "prepass_velocity.frag.spv",
     .vs = k_ubo1, // PrepassModelUniforms
     .fs = k_none,
     .vertex_layout = VertexLayout::Vertex3DPositionOnly,
     .color = TargetFormat::Velocity},
    {.name = "DepthPrepassModelDoubleSided",
     .vertex_shader = "prepass_model.vert.spv",
     .fragment_shader = "prepass_velocity.frag.spv",
     .vs = k_ubo1,
     .fs = k_none,
     .vertex_layout = VertexLayout::Vertex3DPositionOnly,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Velocity},
    {.name = "DepthPrepassSkinnedModel",
     .vertex_shader = "prepass_skinned.vert.spv",
     .fragment_shader = "prepass_velocity.frag.spv",
     .vs = k_ubo1_sb2, // current + previous bone palettes
     .fs = k_none,
     .vertex_layout = VertexLayout::SkinnedVertexPositionSkin,
     .color = TargetFormat::Velocity},
    {.name = "DepthPrepassSkinnedModelDoubleSided",
     .vertex_shader = "prepass_skinned.vert.spv",
     .fragment_shader = "prepass_velocity.frag.spv",
     .vs = k_ubo1_sb2,
     .fs = k_none,
     .vertex_layout = VertexLayout::SkinnedVertexPositionSkin,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Velocity},
    {.name = "DepthPrepassInstancedModel",
     .vertex_shader = "prepass_instanced.vert.spv",
     .fragment_shader = "prepass_velocity.frag.spv",
     .vs = k_ubo1_sb1, // PrepassCameraUniforms + shared InstanceData buffer
     .fs = k_none,
     .vertex_layout = VertexLayout::Vertex3DPositionOnly,
     .color = TargetFormat::Velocity},
    {.name = "DepthPrepassInstancedModelDoubleSided",
     .vertex_shader = "prepass_instanced.vert.spv",
     .fragment_shader = "prepass_velocity.frag.spv",
     .vs = k_ubo1_sb1,
     .fs = k_none,
     .vertex_layout = VertexLayout::Vertex3DPositionOnly,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Velocity},
    {.name = "DepthPrepassTerrain",
     .vertex_shader = "prepass_terrain.vert.spv",
     .fragment_shader = "prepass_velocity.frag.spv",
     .vs = k_ubo1,
     .fs = k_none,
     .vertex_layout = VertexLayout::TerrainPositionOnly,
     .color = TargetFormat::Velocity},
    // Same vertex-stage resources as the forward grass pipeline: the prepass
    // shader runs the identical blade placement math (grass_blade.hlsli), and
    // no_cull matches so the EQUAL depth coverage is identical.
    {.name = "DepthPrepassGrass",
     .vertex_shader = "prepass_grass.vert.spv",
     .fragment_shader = "prepass_velocity.frag.spv",
     .vs = k_vs_grass,
     .fs = k_none,
     .vertex_layout = VertexLayout::Vertex3DPositionOnly,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Velocity},

    // --- Post-prepass forward variants ----------------------------------------
    // Depth was laid down by the prepass, so these test EQUAL and skip the
    // redundant depth write. Everything else matches their base row.
    {.name = "ModelDepthEqual",
     .vertex_shader = "model.vert.spv",
     .fragment_shader = "model.frag.spv",
     .vs = k_ubo1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::Vertex3D,
     .depth_write = false,
     .depth_compare = SDL_GPU_COMPAREOP_EQUAL},
    {.name = "ModelDoubleSidedDepthEqual",
     .vertex_shader = "model.vert.spv",
     .fragment_shader = "model.frag.spv",
     .vs = k_ubo1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::Vertex3D,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .depth_write = false,
     .depth_compare = SDL_GPU_COMPAREOP_EQUAL},
    {.name = "SkinnedModelDepthEqual",
     .vertex_shader = "skinned_model.vert.spv",
     .fragment_shader = "model.frag.spv",
     .vs = k_ubo1_sb1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::SkinnedVertex,
     .depth_write = false,
     .depth_compare = SDL_GPU_COMPAREOP_EQUAL},
    {.name = "SkinnedModelDoubleSidedDepthEqual",
     .vertex_shader = "skinned_model.vert.spv",
     .fragment_shader = "model.frag.spv",
     .vs = k_ubo1_sb1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::SkinnedVertex,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .depth_write = false,
     .depth_compare = SDL_GPU_COMPAREOP_EQUAL},
    {.name = "InstancedModelDepthEqual",
     .vertex_shader = "model_instanced.vert.spv",
     .fragment_shader = "model_instanced.frag.spv",
     .vs = k_ubo1_sb1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::Vertex3D,
     .depth_write = false,
     .depth_compare = SDL_GPU_COMPAREOP_EQUAL},
    {.name = "InstancedModelDoubleSidedDepthEqual",
     .vertex_shader = "model_instanced.vert.spv",
     .fragment_shader = "model_instanced.frag.spv",
     .vs = k_ubo1_sb1,
     .fs = k_fs_lit_model,
     .vertex_layout = VertexLayout::Vertex3D,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .depth_write = false,
     .depth_compare = SDL_GPU_COMPAREOP_EQUAL},
    {.name = "TerrainDepthEqual",
     .vertex_shader = "terrain.vert.spv",
     .fragment_shader = "terrain.frag.spv",
     .vs = k_ubo1,
     .fs = k_fs_lit_terrain,
     .vertex_layout = VertexLayout::Terrain,
     .depth_write = false,
     .depth_compare = SDL_GPU_COMPAREOP_EQUAL},
    {.name = "GrassDepthEqual",
     .vertex_shader = "grass.vert.spv",
     .fragment_shader = "grass.frag.spv",
     .vs = k_vs_grass,
     .fs = k_fs_lit_grass,
     .vertex_layout = VertexLayout::Vertex3D,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .depth_write = false,
     .depth_compare = SDL_GPU_COMPAREOP_EQUAL},

    // --- Alpha-MASK prepass ---------------------------------------------------
    // Texcoord passthrough + alpha-tested fragment so foliage writes depth
    // (GTAO + velocity coverage). Always no-cull: MASK foliage is double-sided
    // in practice.
    {.name = "DepthPrepassModelMasked",
     .vertex_shader = "prepass_model_masked.vert.spv",
     .fragment_shader = "prepass_velocity_masked.frag.spv",
     .vs = k_ubo1,
     .fs = k_ubo1_samp1, // PrepassMaskedUniforms (alpha cutoff) + base color
     .vertex_layout = VertexLayout::Vertex3DPositionTexcoord,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Velocity},
    {.name = "DepthPrepassInstancedModelMasked",
     .vertex_shader = "prepass_instanced_masked.vert.spv",
     .fragment_shader = "prepass_velocity_masked.frag.spv",
     .vs = k_ubo1_sb1,
     .fs = k_ubo1_samp1,
     .vertex_layout = VertexLayout::Vertex3DPositionTexcoord,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Velocity},

    // --- Depth-prepass G-buffer variants --------------------------------------
    // Same geometry, same depth, plus an oct-normal/roughness/metallic attachment for
    // ray-traced reflections (render/rt_renderer.hpp). Only created when a game
    // turns reflections on. Grass and MASK foliage write a constant roughness
    // of 1 -- they are never mirror surfaces, so the reflection pass skips them
    // without tracing, and they need no normal plumbed through their vertex
    // stage. `.fs` gains one uniform buffer on the rows that send a real
    // roughness factor.
    {.name = "DepthPrepassModelGBuffer",
     .vertex_shader = "prepass_model_gbuffer.vert.spv",
     .fragment_shader = "prepass_gbuffer.frag.spv",
     .vs = k_ubo1,
     .fs = k_ubo1_samp1, // + the metallicRoughness map (see prepass_gbuffer.frag.hlsl)
     .vertex_layout = VertexLayout::Vertex3DPositionNormalTexcoord,
     .color = TargetFormat::Velocity,
     .color2 = TargetFormat::NormalRoughness},
    {.name = "DepthPrepassModelDoubleSidedGBuffer",
     .vertex_shader = "prepass_model_gbuffer.vert.spv",
     .fragment_shader = "prepass_gbuffer.frag.spv",
     .vs = k_ubo1,
     .fs = k_ubo1_samp1, // + the metallicRoughness map (see prepass_gbuffer.frag.hlsl)
     .vertex_layout = VertexLayout::Vertex3DPositionNormalTexcoord,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Velocity,
     .color2 = TargetFormat::NormalRoughness},
    {.name = "DepthPrepassSkinnedModelGBuffer",
     .vertex_shader = "prepass_skinned_gbuffer.vert.spv",
     .fragment_shader = "prepass_gbuffer.frag.spv",
     .vs = k_ubo1_sb2,
     .fs = k_ubo1_samp1, // + the metallicRoughness map
     .vertex_layout = VertexLayout::SkinnedVertexPositionSkinNormalTexcoord,
     .color = TargetFormat::Velocity,
     .color2 = TargetFormat::NormalRoughness},
    {.name = "DepthPrepassSkinnedModelDoubleSidedGBuffer",
     .vertex_shader = "prepass_skinned_gbuffer.vert.spv",
     .fragment_shader = "prepass_gbuffer.frag.spv",
     .vs = k_ubo1_sb2,
     .fs = k_ubo1_samp1, // + the metallicRoughness map
     .vertex_layout = VertexLayout::SkinnedVertexPositionSkinNormalTexcoord,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Velocity,
     .color2 = TargetFormat::NormalRoughness},
    {.name = "DepthPrepassInstancedModelGBuffer",
     .vertex_shader = "prepass_instanced_gbuffer.vert.spv",
     .fragment_shader = "prepass_gbuffer.frag.spv",
     .vs = k_ubo1_sb1,
     .fs = k_ubo1_samp1, // + the metallicRoughness map
     .vertex_layout = VertexLayout::Vertex3DPositionNormalTexcoord,
     .color = TargetFormat::Velocity,
     .color2 = TargetFormat::NormalRoughness},
    {.name = "DepthPrepassInstancedModelDoubleSidedGBuffer",
     .vertex_shader = "prepass_instanced_gbuffer.vert.spv",
     .fragment_shader = "prepass_gbuffer.frag.spv",
     .vs = k_ubo1_sb1,
     .fs = k_ubo1_samp1, // + the metallicRoughness map
     .vertex_layout = VertexLayout::Vertex3DPositionNormalTexcoord,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Velocity,
     .color2 = TargetFormat::NormalRoughness},
    // .fs = k_none is load-bearing: this row used to share prepass_gbuffer.frag
    // (which declares a fragment UBO) while TerrainRenderer::render_prepass
    // only ever pushed the VERTEX uniforms, so the fragment block read
    // recycled pool memory and classified the whole terrain as a mirror. Its
    // own uniform-free shader makes that unrepresentable.
    {.name = "DepthPrepassTerrainGBuffer",
     .vertex_shader = "prepass_terrain_gbuffer.vert.spv",
     .fragment_shader = "prepass_gbuffer_terrain.frag.spv",
     .vs = k_ubo1,
     .fs = k_none,
     .vertex_layout = VertexLayout::TerrainPositionNormal,
     .color = TargetFormat::Velocity,
     .color2 = TargetFormat::NormalRoughness},
    {.name = "DepthPrepassGrassGBuffer",
     .vertex_shader = "prepass_grass.vert.spv",
     .fragment_shader = "prepass_gbuffer_rough.frag.spv",
     .vs = k_vs_grass,
     .fs = k_none,
     .vertex_layout = VertexLayout::Vertex3DPositionOnly,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Velocity,
     .color2 = TargetFormat::NormalRoughness},
    {.name = "DepthPrepassModelMaskedGBuffer",
     .vertex_shader = "prepass_model_masked.vert.spv",
     .fragment_shader = "prepass_gbuffer_masked.frag.spv",
     .vs = k_ubo1,
     .fs = k_ubo1_samp1,
     .vertex_layout = VertexLayout::Vertex3DPositionTexcoord,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Velocity,
     .color2 = TargetFormat::NormalRoughness},
    {.name = "DepthPrepassInstancedModelMaskedGBuffer",
     .vertex_shader = "prepass_instanced_masked.vert.spv",
     .fragment_shader = "prepass_gbuffer_masked.frag.spv",
     .vs = k_ubo1_sb1,
     .fs = k_ubo1_samp1,
     .vertex_layout = VertexLayout::Vertex3DPositionTexcoord,
     .cull_mode = SDL_GPU_CULLMODE_NONE,
     .color = TargetFormat::Velocity,
     .color2 = TargetFormat::NormalRoughness},

    // --- Water ----------------------------------------------------------------
    // Refraction/reflection blending is composited in-shader from the scene
    // copies, so the pipeline itself is opaque; depth test + write stay on so
    // fog, TAA reprojection and later transparents see the water surface.
    {.name = "Water",
     .vertex_shader = "water.vert.spv",
     .fragment_shader = "water.frag.spv",
     .vs = k_ubo1_samp1, // VertexUniforms + sim height state (displacement)
     .fs = k_ubo1_samp5, // + raw wave-aware RT reflection (dummy while disabled)
     .vertex_layout = VertexLayout::Position2,
     .cull_mode = SDL_GPU_CULLMODE_NONE},
}};

static_assert(k_pipeline_descs.size() == static_cast<size_t>(PipelineType::Count),
              "every PipelineType needs exactly one catalogue row, in enum order");

constexpr std::array<ComputePipelineDesc, static_cast<size_t>(ComputePipelineType::Count)> k_compute_descs{{
    {.name = "ExposureHistogram",
     .shader = "exposure_histogram.comp.spv",
     .entry_point = "CSMain",
     .resources = {.num_samplers = 1,                  // HDR scene color
                   .num_readwrite_storage_buffers = 1, // histogram bins
                   .num_uniform_buffers = 1}},
    {.name = "ExposureAdapt",
     .shader = "exposure_adapt.comp.spv",
     .entry_point = "CSMain",
     .resources = {.num_readwrite_storage_buffers = 2, // bins (read + reset) + exposure state
                   .num_uniform_buffers = 1}},
    {.name = "WaterSim",
     .shader = "water_sim.comp.spv",
     .entry_point = "CSMain",
     .resources = {.num_readonly_storage_textures = 1,  // previous wave state
                   .num_readwrite_storage_textures = 1, // next wave state
                   .num_uniform_buffers = 1}},
    {.name = "RtShadow",
     .shader = "rt_shadow.comp.spv",
     .entry_point = "CSMain",
     // Depth + the prepass normal G-buffer, then the four acceleration-
     // structure buffers. No attribute buffer: a shadow ray asks a yes/no
     // question and never shades what it hits, which is why it costs 48 bytes
     // per triangle and reflections cost 112. The output is one BIT per pixel
     // packed into 8x4 tiles, not an R8 mask — the denoiser's classifier reads
     // 32 pixels per fetch off the back of that.
     .resources = {.num_samplers = 2,
                   .num_readonly_storage_buffers = 4,
                   .num_readwrite_storage_textures = 1,
                   .num_uniform_buffers = 1}},
    {.name = "RtShadowClassify",
     .shader = "rt_shadow_denoise_classify.comp.spv",
     .entry_point = "CSMain",
     .resources = {.num_readonly_storage_textures = 1,  // packed visibility bits
                   .num_readwrite_storage_textures = 1, // per-tile verdict
                   .num_uniform_buffers = 1}},
    {.name = "RtShadowTemporal",
     .shader = "rt_shadow_denoise_temporal.comp.spv",
     .entry_point = "CSMain",
     .resources = {.num_samplers = 3,                   // depth, velocity, history
                   .num_readonly_storage_textures = 2,  // packed bits, tile verdicts
                   .num_readwrite_storage_textures = 1, // accumulated history out
                   .num_uniform_buffers = 1}},
    {.name = "RtShadowFilter",
     .shader = "rt_shadow_denoise_filter.comp.spv",
     .entry_point = "CSMain",
     .resources = {.num_samplers = 2,                   // history, depth
                   .num_readonly_storage_textures = 1,  // tile verdicts
                   .num_readwrite_storage_textures = 1, // full-res R8 mask
                   .num_uniform_buffers = 1}},
    {.name = "RtReflection",
     .shader = "rt_reflection.comp.spv",
     .entry_point = "CSMain",
     // Same, plus per-triangle attributes to shade the hit. Two RW targets:
     // raw radiance, and the normalized hit distance R3's reprojection wants.
     .resources = {.num_samplers = 2,
                   .num_readonly_storage_buffers = 5,
                   .num_readwrite_storage_textures = 2,
                   .num_uniform_buffers = 1}},
    {.name = "RtReflectionTemporal",
     .shader = "rt_reflection_denoise_temporal.comp.spv",
     .entry_point = "CSMain",
     .resources = {.num_samplers = 3,                   // raw trace, velocity, history
                   .num_readwrite_storage_textures = 1, // accumulated history out
                   .num_uniform_buffers = 1}},
    {.name = "RtLocalShadow",
     .shader = "rt_local_shadow.comp.spv",
     .entry_point = "CSMain",
     // Sampled: depth, the prepass normal G-buffer, and velocity — the last
     // because this pass reprojects its own RESERVOIR state, which is a
     // different thing from the denoiser reprojecting the denoised signal.
     // Read-only storage buffers: the four acceleration-structure buffers, the
     // FIVE cluster buffers cluster_compute.hlsli claims (light data, froxel
     // offsets, froxel indices, stable ids, prev->current remap), and the
     // previous frame's reservoirs. Read-write: the packed visibility bits and
     // this frame's reservoirs. Two uniform buffers, not one: b0 is this pass's
     // own, b1 is ClusterGrid's own ClusterParams, pushed verbatim rather than
     // copied field by field into a second struct that could drift from it.
     // EIGHT read-only storage buffers, and eight is not a round number here —
     // it is SDL3 GPU's hard ceiling for a compute pipeline ("Compute pipeline
     // readonly storage buffer count cannot be higher than 8"). The four
     // acceleration-structure buffers are non-negotiable, which leaves exactly
     // four for the cluster grid: light data, froxel offsets, froxel indices,
     // and the prev->current remap. The fifth buffer cluster_compute.hlsli can
     // declare — the packed stable ids — is therefore OPT-IN and this pass does
     // not take it: a reservoir carries identity ACROSS frames, which is the
     // remap's job, not the ids'.
     //
     // The reservoir ping-pong consequently rides in set 1 as two READ-WRITE
     // buffers, of which one is only ever read. That is the budget talking, and
     // it is written down at the binding site too.
     .resources = {.num_samplers = 3,
                   .num_readonly_storage_buffers = 8,
                   .num_readwrite_storage_textures = 1,
                   .num_readwrite_storage_buffers = 2,
                   .num_uniform_buffers = 2}},
    {.name = "GiProbeTrace",
     .shader = "gi_probe_trace.comp.spv",
     .entry_point = "CSMain",
     // The widest binding block in the engine, and every entry earns its slot.
     // Sampled: the CSM atlas twice (comparison + raw, because it includes
     // shadow_common.hlsli verbatim rather than reimplementing the cascade
     // math), the IBL sky-capture cube for ray misses, and the BANDED
     // irradiance atlas for the previous-frame bounce — one texture holding
     // every cascade since R1c (gi_math.hpp), where R1b bound three.
     // Read-only storage textures: the relocation/classification plane and the
     // probe-state plane, both banded for the same reason, so the bounce can
     // reject a slot whose previous occupant has not been overwritten yet.
     // Read-only storage buffers: the four acceleration-structure buffers, the
     // per-triangle attributes that make a hit shadeable, the
     // clustered-lighting byte buffer, and the frame's update list. One
     // read-write buffer out: the per-ray radiance + signed distance the three
     // passes below consume.
     .resources = {.num_samplers = 4,
                   .num_readonly_storage_textures = 2,
                   .num_readonly_storage_buffers = 7,
                   .num_readwrite_storage_buffers = 1,
                   .num_uniform_buffers = 1}},
    {.name = "GiProbeBlend",
     .shader = "gi_probe_blend.comp.spv",
     .entry_point = "CSMain",
     .resources = {.num_readonly_storage_textures = 1,  // relocation/classification
                   .num_readonly_storage_buffers = 2,   // ray results, update list
                   .num_readwrite_storage_textures = 1, // irradiance atlas (read-modify-write)
                   .num_uniform_buffers = 1}},
    {.name = "GiProbeBlendVisibility",
     .shader = "gi_probe_blend_visibility.comp.spv",
     .entry_point = "CSMain",
     .resources = {.num_readonly_storage_textures = 1,
                   .num_readonly_storage_buffers = 2,
                   .num_readwrite_storage_textures = 1, // visibility moments
                   .num_uniform_buffers = 1}},
    {.name = "GiProbeClassify",
     .shader = "gi_probe_classify.comp.spv",
     .entry_point = "CSMain",
     .resources = {.num_readonly_storage_buffers = 2,
                   .num_readwrite_storage_textures = 1, // per-probe offset + verdict
                   .num_readwrite_storage_buffers = 1,  // diagnostic counters
                   .num_uniform_buffers = 1}},
}};

static_assert(k_compute_descs.size() == static_cast<size_t>(ComputePipelineType::Count),
              "every ComputePipelineType needs exactly one catalogue row, in enum order");

constexpr PipelineDesc k_unknown_pipeline{.name = "Unknown"};
constexpr ComputePipelineDesc k_unknown_compute{.name = "Unknown"};

SDL_GPUVertexBufferDescription buffer_desc(uint32_t pitch) {
    return {.slot = 0, .pitch = pitch, .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX, .instance_step_rate = 0};
}

void apply_vertex_layout(PipelineConfig& config, VertexLayout layout) {
    switch (layout) {
        case VertexLayout::None:
            // Fullscreen triangle synthesized from SV_VertexID: no input at all.
            return;
        case VertexLayout::Vertex3D:
            config.with_vertex3d();
            return;
        case VertexLayout::Vertex3DPositionOnly:
            config.vertex_buffers = {get_vertex3d_buffer_desc()};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(Vertex3D, position)},
            };
            return;
        case VertexLayout::Vertex3DPositionTexcoord:
            config.vertex_buffers = {get_vertex3d_buffer_desc()};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(Vertex3D, position)},
                {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, offsetof(Vertex3D, texcoord)},
            };
            return;
        case VertexLayout::Vertex3DPositionNormal:
            config.vertex_buffers = {get_vertex3d_buffer_desc()};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(Vertex3D, position)},
                {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(Vertex3D, normal)},
            };
            return;
        case VertexLayout::Vertex3DPositionNormalTexcoord:
            config.vertex_buffers = {get_vertex3d_buffer_desc()};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(Vertex3D, position)},
                {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(Vertex3D, normal)},
                {2, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, offsetof(Vertex3D, texcoord)},
            };
            return;
        case VertexLayout::SkinnedVertex:
            config.with_skinned_vertex();
            return;
        case VertexLayout::SkinnedVertexPositionSkin:
            config.vertex_buffers = {get_skinned_vertex_buffer_desc()};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(SkinnedVertex, position)},
                {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4, offsetof(SkinnedVertex, joints)},
                {2, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, offsetof(SkinnedVertex, weights)},
            };
            return;
        case VertexLayout::SkinnedVertexPositionSkinNormal:
            config.vertex_buffers = {get_skinned_vertex_buffer_desc()};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(SkinnedVertex, position)},
                {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4, offsetof(SkinnedVertex, joints)},
                {2, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, offsetof(SkinnedVertex, weights)},
                {3, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(SkinnedVertex, normal)},
            };
            return;
        case VertexLayout::SkinnedVertexPositionSkinNormalTexcoord:
            config.vertex_buffers = {get_skinned_vertex_buffer_desc()};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(SkinnedVertex, position)},
                {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4, offsetof(SkinnedVertex, joints)},
                {2, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, offsetof(SkinnedVertex, weights)},
                {3, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(SkinnedVertex, normal)},
                {4, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, offsetof(SkinnedVertex, texcoord)},
            };
            return;
        case VertexLayout::Vertex2D:
            config.with_vertex2d();
            return;
        case VertexLayout::Terrain:
            config.vertex_buffers = {buffer_desc(k_terrain_vertex_stride)};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0},                 // position
                {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, sizeof(float) * 3}, // texcoord
                {2, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, sizeof(float) * 5}, // color
                {3, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, sizeof(float) * 9}, // normal
            };
            return;
        case VertexLayout::TerrainPositionOnly:
            config.vertex_buffers = {buffer_desc(k_terrain_vertex_stride)};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0},
            };
            return;
        case VertexLayout::TerrainPositionNormal:
            config.vertex_buffers = {buffer_desc(k_terrain_vertex_stride)};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0},                 // position
                {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, sizeof(float) * 9}, // normal
            };
            return;
        case VertexLayout::Grid:
            config.vertex_buffers = {buffer_desc(sizeof(float) * 7)};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0},                 // position
                {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, sizeof(float) * 3}, // color
            };
            return;
        case VertexLayout::Text:
            // Color is a uniform, not per-vertex.
            config.vertex_buffers = {buffer_desc(sizeof(float) * 4)};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, 0},                 // position
                {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, sizeof(float) * 2}, // texcoord
            };
            return;
        case VertexLayout::Position3:
            config.vertex_buffers = {buffer_desc(sizeof(float) * 3)};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0},
            };
            return;
        case VertexLayout::Position2:
            config.vertex_buffers = {buffer_desc(sizeof(float) * 2)};
            config.vertex_attributes = {
                {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, 0},
            };
            return;
    }
}

SDL_GPUTextureFormat resolve_target_format(TargetFormat target, const PipelineFormats& formats) {
    switch (target) {
        case TargetFormat::DepthOnly:
            return SDL_GPU_TEXTUREFORMAT_INVALID;
        case TargetFormat::Swapchain:
            return formats.swapchain;
        case TargetFormat::SceneColor:
            return formats.scene_color;
        case TargetFormat::Velocity:
            return formats.velocity;
        case TargetFormat::R8:
            return SDL_GPU_TEXTUREFORMAT_R8_UNORM;
        case TargetFormat::RG16F:
            return SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT;
        case TargetFormat::RGBA16F:
            return SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
        case TargetFormat::NormalRoughness:
            return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    }
    return SDL_GPU_TEXTUREFORMAT_INVALID;
}

bool in_range(PipelineType type) {
    return static_cast<size_t>(type) < k_pipeline_descs.size();
}

bool in_range(ComputePipelineType type) {
    return static_cast<size_t>(type) < k_compute_descs.size();
}

} // namespace

const PipelineDesc& pipeline_desc(PipelineType type) {
    return in_range(type) ? k_pipeline_descs[static_cast<size_t>(type)] : k_unknown_pipeline;
}

const ComputePipelineDesc& compute_pipeline_desc(ComputePipelineType type) {
    return in_range(type) ? k_compute_descs[static_cast<size_t>(type)] : k_unknown_compute;
}

const char* pipeline_type_to_string(PipelineType type) {
    return pipeline_desc(type).name;
}

const char* compute_pipeline_type_to_string(ComputePipelineType type) {
    return compute_pipeline_desc(type).name;
}

bool pipeline_targets_swapchain(PipelineType type) {
    return in_range(type) && k_pipeline_descs[static_cast<size_t>(type)].color == TargetFormat::Swapchain;
}

PipelineConfig build_pipeline_config(const PipelineDesc& desc, const PipelineFormats& formats) {
    PipelineConfig config;
    apply_vertex_layout(config, desc.vertex_layout);

    config.primitive_type = desc.primitive_type;
    config.cull_mode = desc.cull_mode;
    config.blend_mode = desc.blend_mode;

    config.depth_test_enable = desc.depth_test;
    config.depth_write_enable = desc.depth_write;
    config.depth_compare_op = desc.depth_compare;
    config.enable_depth_clip = desc.depth_clip;
    config.enable_depth_bias = desc.depth_bias.enabled;
    config.depth_bias_constant = desc.depth_bias.constant;
    config.depth_bias_slope = desc.depth_bias.slope;
    config.depth_bias_clamp = 0.0f;

    config.color_format = resolve_target_format(desc.color, formats);
    config.color_format2 = resolve_target_format(desc.color2, formats);
    config.depth_format = formats.depth;
    config.has_depth_target = desc.depth_target;
    if (!desc.depth_target) {
        // No depth attachment: the test/write bits must be off regardless of
        // what the row says (PipelineConfig::no_depth()).
        config.depth_test_enable = false;
        config.depth_write_enable = false;
    }
    return config;
}

} // namespace bracken::gpu
