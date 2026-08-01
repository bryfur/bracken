#pragma once

#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/gpu_shader.hpp"
#include <cstdint>

namespace bracken::gpu {

/**
 * @file pipeline_catalog.hpp
 * @brief The static pipeline catalogue: what every PipelineType *is*.
 *
 * This is data, not machinery. `PipelineRegistry` (pipeline_registry.hpp) owns
 * the cache, the pre-creation contract and the hot-reload lifecycle; it names
 * no individual pipeline. Everything a pipeline is made of — its shader pair,
 * the resource counts those shaders declare, its vertex layout, its render
 * target and its fixed-function state — lives in one row of the table here,
 * and `build_pipeline_config` turns a row plus the frame's texture formats
 * into the `PipelineConfig` the GPU layer consumes.
 *
 * Two consequences worth knowing:
 * - A missing row is a **compile error** (`static_assert` on the table size in
 *   pipeline_catalog.cpp), not a silent "Unknown" at runtime the way two
 *   hand-maintained 56-case switches used to be.
 * - The reversed-Z story is one reviewable column. The engine renders the
 *   scene reversed-Z (near -> 1, far -> 0, clear 0.0) and therefore compares
 *   `GREATER_OR_EQUAL`; the exceptions — shadow cascades staying standard-Z
 *   with `LESS_OR_EQUAL`, the post-prepass `EQUAL` variants, the local-shadow
 *   tile clear's `ALWAYS` — are visible as `depth_compare` overrides in the
 *   table rather than scattered across a thousand lines of creators.
 *
 * Resource counts are still declared by hand here. They are cross-checked
 * against SPIR-V reflection at shader creation (see ShaderResources), and
 * reflecting them instead of declaring them is a deliberate follow-on, not
 * part of this split.
 */

/**
 * @brief Pipeline type enumeration for all supported rendering pipelines
 */
enum class PipelineType {
    // 3D rendering pipelines
    Model,        ///< Static 3D models
    SkinnedModel, ///< Animated/skeletal models
    Terrain,      ///< Terrain rendering with splatmap
    Skybox,       ///< Procedural sky (fullscreen triangle, no textures/cubemap)
    Grid,         ///< Debug grid overlay

    // 2D/UI pipelines
    UI,   ///< 2D UI elements
    Text, ///< Text rendering with font atlas

    // Effect pipelines
    Particle, ///< HDR particles (instanced; particle.vert + particle.frag; blend configured
              ///< (SRC_ALPHA, ONE) — the fragment shader writes alpha 1.0, so the effective
              ///< color blend is (ONE, ONE) additive)
    Grass,    ///< Instanced grass rendering

    // Instanced rendering pipelines
    InstancedModel,       ///< Instanced static models (rocks, trees)
    InstancedShadowModel, ///< Instanced shadow depth pass

    // Shadow pipelines (depth-only)
    ShadowModel,        ///< Shadow depth pass for static models
    ShadowSkinnedModel, ///< Shadow depth pass for skinned models
    ShadowTerrain,      ///< Shadow depth pass for terrain
    LocalShadowClear,   ///< Fullscreen depth clear of one local-shadow atlas tile
                        ///< (viewport depth range pinned to [1,1]; compare ALWAYS)

    // Post-processing pipelines (fullscreen triangle, no vertex input)
    GTAO,            ///< GTAO (horizon-based) ambient occlusion from prepass depth
    GTAOTemporal,    ///< GTAO temporal accumulation (velocity reprojection + depth reject)
    GTAOUpsample,    ///< GTAO joint-bilateral upsample to the full-res R8 AO texture
    BlurAO,          ///< Bilateral blur for AO
    Composite,       ///< Final composite (scene + bloom + fog, tonemap) to swapchain
    BloomDownsample, ///< Bloom brightness extract + downsample
    BloomUpsample,   ///< Bloom tent-filter upsample (additive accumulate)
    VolumetricFog,   ///< Volumetric fog with god rays
    Fxaa,            ///< FXAA 3.11 on the LDR composite output -> swapchain
    TaaResolve,      ///< TAA resolve: HDR scene + history + velocity + depth -> new history
    TaaSharpen,      ///< Post-TAA unsharp mask, resolved history -> HDR scene target

    // IBL prefilter pipelines (fullscreen triangle into one cubemap face/mip;
    // render::IBL drives them — see render/ibl.hpp).
    SkyCapture,           ///< Procedural sky radiance -> capture cube face (RGBA16F)
    IblSpecularPrefilter, ///< GGX importance-sampled prefilter of the capture cube, per mip
    IblIrradiance,        ///< Cosine-hemisphere irradiance convolution of the capture cube
    IblBrdfLut,           ///< Karis split-sum BRDF LUT (RG16F, rendered once at init)

    // Model-family material variants for glTF flags: DoubleSided disables
    // back-face culling; Blend enables alpha blending (depth test on, depth
    // write off — drawn back-to-front after opaques by SceneRenderer).
    ModelDoubleSided,             ///< Static models, doubleSided
    ModelBlend,                   ///< Static models, alphaMode BLEND
    ModelBlendDoubleSided,        ///< Static models, BLEND + doubleSided
    SkinnedModelDoubleSided,      ///< Skinned models, doubleSided
    SkinnedModelBlend,            ///< Skinned models, alphaMode BLEND
    SkinnedModelBlendDoubleSided, ///< Skinned models, BLEND + doubleSided
    InstancedModelDoubleSided,    ///< Instanced static models, doubleSided

    // Depth-prepass pipelines: position-only vertex work, velocity (RG16F)
    // color target + scene depth, reversed-Z GREATER_OR_EQUAL with depth
    // writes ON. OPAQUE geometry uses the position-only variants; MASK
    // geometry uses the *Masked alpha-tested variants below (BLEND keeps
    // its forward depth behavior).
    DepthPrepassModel,                    ///< Prepass: static models
    DepthPrepassModelDoubleSided,         ///< Prepass: static models, doubleSided
    DepthPrepassSkinnedModel,             ///< Prepass: skinned models (curr + prev palettes)
    DepthPrepassSkinnedModelDoubleSided,  ///< Prepass: skinned models, doubleSided
    DepthPrepassInstancedModel,           ///< Prepass: instanced models
    DepthPrepassInstancedModelDoubleSided,///< Prepass: instanced models, doubleSided
    DepthPrepassTerrain,                  ///< Prepass: terrain
    DepthPrepassGrass,                    ///< Prepass: grass blades

    // "DepthEqual" forward variants for geometry whose depth was laid down by
    // the prepass: depth test EQUAL, depth writes OFF, otherwise identical to
    // their base pipeline. MASK/BLEND/skybox/particles keep GREATER_OR_EQUAL.
    ModelDepthEqual,                   ///< Static models after prepass
    ModelDoubleSidedDepthEqual,        ///< Static models, doubleSided, after prepass
    SkinnedModelDepthEqual,            ///< Skinned models after prepass
    SkinnedModelDoubleSidedDepthEqual, ///< Skinned models, doubleSided, after prepass
    InstancedModelDepthEqual,          ///< Instanced models after prepass
    InstancedModelDoubleSidedDepthEqual,///< Instanced models, doubleSided, after prepass
    TerrainDepthEqual,                 ///< Terrain after prepass
    GrassDepthEqual,                   ///< Grass after prepass

    DepthPrepassModelMasked,           ///< Static MASK meshes (alpha-tested prepass, no cull)
    DepthPrepassInstancedModelMasked,  ///< Instanced MASK meshes (alpha-tested prepass, no cull)

    // --- Depth-prepass G-buffer variants -------------------------------------
    // Identical to the rows above but writing a SECOND attachment (world
    // normal + roughness) for ray-traced reflections. They exist as separate
    // pipelines rather than a flag because a render pass and every pipeline
    // used inside it must agree on the colour-attachment count — so enabling
    // reflections swaps the whole prepass over to this set, and a game that
    // leaves them off never creates these at all.
    DepthPrepassModelGBuffer,
    DepthPrepassModelDoubleSidedGBuffer,
    DepthPrepassSkinnedModelGBuffer,
    DepthPrepassSkinnedModelDoubleSidedGBuffer,
    DepthPrepassInstancedModelGBuffer,
    DepthPrepassInstancedModelDoubleSidedGBuffer,
    DepthPrepassTerrainGBuffer,
    DepthPrepassGrassGBuffer,
    DepthPrepassModelMaskedGBuffer,
    DepthPrepassInstancedModelMaskedGBuffer,

    Water,                             ///< Water surface (own pass after opaques + skybox)

    // Count for iteration
    Count
};

/**
 * @brief Compute pipeline enumeration (managed alongside the graphics
 * pipelines: same cache, eager preload and hot-reload paths).
 * Compute shaders are `{name}.comp.hlsl` (entry `CSMain`) compiled to
 * `.comp.spv` — see gpu_compute.hpp for the binding conventions.
 */
enum class ComputePipelineType {
    ExposureHistogram, ///< 256-bin log2-luminance histogram of the HDR scene (16x16 tiles, LDS-merged)
    ExposureAdapt,     ///< Histogram reduce -> target EV -> temporal adaptation (single workgroup)
    WaterSim,          ///< Wave-equation water simulation substep (ping-pong RG16F state)
    RtShadow,          ///< Ray-traced sun shadows over the software BVH (any-hit, half-res mask)
    RtReflection,      ///< Ray-traced specular reflections over the same BVH (closest-hit)

    // Count for iteration
    Count
};

/**
 * @brief Vertex input layout a pipeline declares.
 *
 * Named rather than spelled out per pipeline: the terrain stride in
 * particular used to be written out three times and its 4-attribute layout
 * twice, which is exactly the kind of copy that drifts.
 */
enum class VertexLayout : uint8_t {
    None,                      ///< No vertex input (fullscreen triangle from SV_VertexID)
    Vertex3D,                  ///< gpu::Vertex3D: position, normal, texcoord, color, tangent
    Vertex3DPositionOnly,      ///< Vertex3D stride, position attribute only (depth prepass)
    Vertex3DPositionTexcoord,  ///< Vertex3D stride, position + texcoord (alpha-MASK prepass)
    Vertex3DPositionNormal,    ///< Vertex3D stride, position + normal (prepass G-buffer)
    SkinnedVertex,             ///< gpu::SkinnedVertex: Vertex3D fields + joints + weights
    SkinnedVertexPositionSkin, ///< SkinnedVertex stride, position + joints + weights
    SkinnedVertexPositionSkinNormal, ///< …plus normal (prepass G-buffer)
    Vertex2D,                  ///< gpu::Vertex2D: position, texcoord, color (UI)
    Terrain,                   ///< 12 floats: position(3), texcoord(2), color(4), normal(3)
    TerrainPositionOnly,       ///< Terrain stride, position attribute only
    TerrainPositionNormal,     ///< Terrain stride, position + normal (prepass G-buffer)
    Grid,                      ///< 7 floats: position(3), color(4)
    Text,                      ///< 4 floats: position(2), texcoord(2); color is a uniform
    Position3,                 ///< 3 floats, tightly packed (skybox cube)
    Position2,                 ///< 2 floats, tightly packed (water grid world XZ)
};

/**
 * @brief Which render target a pipeline writes, resolved at build time.
 *
 * Four of these are runtime values, not constants — `Swapchain` in particular
 * is whatever the driver handed us and can change while the app is running
 * (PipelineRegistry::set_swapchain_format). Tabulating the *intent* is what
 * lets the registry ask the table which pipelines a format change affects
 * instead of keeping a hand-written list beside it.
 */
enum class TargetFormat : uint8_t {
    DepthOnly,  ///< No color target (SDL_GPU_TEXTUREFORMAT_INVALID) — shadow casters
    Swapchain,  ///< The presentable surface format
    SceneColor, ///< The HDR offscreen scene target
    Velocity,   ///< The motion-vector target written by the depth prepass
    R8,         ///< Single-channel unorm (ambient occlusion)
    RG16F,      ///< Two-channel half float (AO history, BRDF LUT)
    RGBA16F,    ///< Four-channel half float (bloom chain, fog, IBL cubes)
    /// RGBA8: world normal encoded to [0,1] in RGB, linear roughness in A.
    /// The optional second prepass attachment that feeds ray-traced
    /// reflections; see render/rt_renderer.hpp.
    NormalRoughness,
};

/// Rasterizer depth bias. `enabled == false` is "no bias" regardless of the
/// magnitudes, matching PipelineConfig::enable_depth_bias.
struct DepthBias {
    float constant = 0.0f;
    float slope = 0.0f;
    bool enabled = false;
};

/**
 * @brief One row of the catalogue: everything a pipeline is.
 *
 * Fields left at their defaults describe the common case — a back-face-culled,
 * opaque, reversed-Z scene pipeline writing the HDR scene target. Anything a
 * row spells out is, by construction, a deliberate deviation.
 */
struct PipelineDesc {
    const char* name = "";             ///< Debug name; also what pipeline_type_to_string returns
    const char* vertex_shader = "";    ///< File name under PipelineRegistry::shader_path()
    const char* fragment_shader = "";  ///< File name under PipelineRegistry::shader_path()
    ShaderResources vs{};              ///< Resources the vertex stage declares
    ShaderResources fs{};              ///< Resources the fragment stage declares

    VertexLayout vertex_layout = VertexLayout::None;
    SDL_GPUPrimitiveType primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    SDL_GPUCullMode cull_mode = SDL_GPU_CULLMODE_BACK;
    BlendMode blend_mode = BlendMode::None;

    bool depth_test = true;
    bool depth_write = true;
    /// Reversed-Z default. See the file comment for the four kinds of override.
    SDL_GPUCompareOp depth_compare = SDL_GPU_COMPAREOP_GREATER_OR_EQUAL;
    /// false pancakes casters in front of the cascade near plane.
    bool depth_clip = true;
    DepthBias depth_bias{};

    TargetFormat color = TargetFormat::SceneColor;
    /// Second colour attachment, or DepthOnly for "none" (the usual case).
    /// See PipelineConfig::color_format2 for why this is a variant.
    TargetFormat color2 = TargetFormat::DepthOnly;
    /// false ⇒ no depth attachment at all; depth test and write are forced off
    /// whatever the two fields above say (this is PipelineConfig::no_depth()).
    bool depth_target = true;
};

/// One row of the compute catalogue. A compute module is the whole pipeline —
/// there is no vertex/fragment pair and no fixed-function state.
struct ComputePipelineDesc {
    const char* name = "";
    const char* shader = "";     ///< File name under PipelineRegistry::shader_path()
    const char* entry_point = ""; ///< Always "CSMain" for engine-built SPIR-V
    ComputeResources resources{};
};

/**
 * @brief The texture formats a catalogue row is resolved against.
 *
 * Defaults are the engine's canonical set; PipelineRegistry substitutes the
 * live swapchain format, which is the only one that changes at runtime.
 */
struct PipelineFormats {
    SDL_GPUTextureFormat swapchain = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    SDL_GPUTextureFormat scene_color = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
    SDL_GPUTextureFormat velocity = SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT;
    SDL_GPUTextureFormat depth = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
};

/// The catalogue row for `type`. Out-of-range (including `Count`) returns a
/// placeholder row named "Unknown" with no shaders — callers that build from
/// it get a config with no shaders, which GPUPipeline::create rejects.
[[nodiscard]] const PipelineDesc& pipeline_desc(PipelineType type);

/// The compute catalogue row for `type`. Same out-of-range contract.
[[nodiscard]] const ComputePipelineDesc& compute_pipeline_desc(ComputePipelineType type);

/// Debug name for a pipeline type ("Unknown" out of range).
[[nodiscard]] const char* pipeline_type_to_string(PipelineType type);

/// Debug name for a compute pipeline type ("Unknown" out of range).
[[nodiscard]] const char* compute_pipeline_type_to_string(ComputePipelineType type);

/// True when `type`'s color target is the swapchain, i.e. the pipeline has to
/// be rebuilt when the presentable format changes.
[[nodiscard]] bool pipeline_targets_swapchain(PipelineType type);

/**
 * @brief Turn a catalogue row + the live formats into a PipelineConfig.
 *
 * Pure: no device, no shader loading, no global state — which is what makes
 * the whole catalogue testable headless against a checked-in golden. The
 * returned config has null `vertex_shader`/`fragment_shader`; the caller
 * resolves those from `desc.vertex_shader` / `desc.fragment_shader`.
 */
[[nodiscard]] PipelineConfig build_pipeline_config(const PipelineDesc& desc, const PipelineFormats& formats);

} // namespace bracken::gpu
