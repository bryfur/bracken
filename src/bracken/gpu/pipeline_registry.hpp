#pragma once

#include "gpu_compute.hpp"
#include "gpu_device.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_shader.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace bracken::core::asset {
class FileWatcher;
}


namespace bracken::gpu {

class ShaderManager;

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

    // Count for iteration
    Count
};

/**
 * @brief Convert pipeline type to string for debugging
 */
inline const char* pipeline_type_to_string(PipelineType type) {
    switch (type) {
        case PipelineType::Model:
            return "Model";
        case PipelineType::SkinnedModel:
            return "SkinnedModel";
        case PipelineType::Terrain:
            return "Terrain";
        case PipelineType::Skybox:
            return "Skybox";
        case PipelineType::Grid:
            return "Grid";
        case PipelineType::UI:
            return "UI";
        case PipelineType::Text:
            return "Text";
        case PipelineType::Particle:
            return "Particle";
        case PipelineType::Grass:
            return "Grass";
        case PipelineType::InstancedModel:
            return "InstancedModel";
        case PipelineType::InstancedShadowModel:
            return "InstancedShadowModel";
        case PipelineType::ShadowModel:
            return "ShadowModel";
        case PipelineType::ShadowSkinnedModel:
            return "ShadowSkinnedModel";
        case PipelineType::ShadowTerrain:
            return "ShadowTerrain";
        case PipelineType::LocalShadowClear:
            return "LocalShadowClear";
        case PipelineType::GTAO:
            return "GTAO";
        case PipelineType::GTAOTemporal:
            return "GTAOTemporal";
        case PipelineType::GTAOUpsample:
            return "GTAOUpsample";
        case PipelineType::BlurAO:
            return "BlurAO";
        case PipelineType::Composite:
            return "Composite";
        case PipelineType::BloomDownsample:
            return "BloomDownsample";
        case PipelineType::BloomUpsample:
            return "BloomUpsample";
        case PipelineType::VolumetricFog:
            return "VolumetricFog";
        case PipelineType::Fxaa:
            return "Fxaa";
        case PipelineType::TaaResolve:
            return "TaaResolve";
        case PipelineType::TaaSharpen:
            return "TaaSharpen";
        case PipelineType::SkyCapture:
            return "SkyCapture";
        case PipelineType::IblSpecularPrefilter:
            return "IblSpecularPrefilter";
        case PipelineType::IblIrradiance:
            return "IblIrradiance";
        case PipelineType::IblBrdfLut:
            return "IblBrdfLut";
        case PipelineType::ModelDoubleSided:
            return "ModelDoubleSided";
        case PipelineType::ModelBlend:
            return "ModelBlend";
        case PipelineType::ModelBlendDoubleSided:
            return "ModelBlendDoubleSided";
        case PipelineType::SkinnedModelDoubleSided:
            return "SkinnedModelDoubleSided";
        case PipelineType::SkinnedModelBlend:
            return "SkinnedModelBlend";
        case PipelineType::SkinnedModelBlendDoubleSided:
            return "SkinnedModelBlendDoubleSided";
        case PipelineType::InstancedModelDoubleSided:
            return "InstancedModelDoubleSided";
        case PipelineType::DepthPrepassModel:
            return "DepthPrepassModel";
        case PipelineType::DepthPrepassModelDoubleSided:
            return "DepthPrepassModelDoubleSided";
        case PipelineType::DepthPrepassSkinnedModel:
            return "DepthPrepassSkinnedModel";
        case PipelineType::DepthPrepassSkinnedModelDoubleSided:
            return "DepthPrepassSkinnedModelDoubleSided";
        case PipelineType::DepthPrepassInstancedModel:
            return "DepthPrepassInstancedModel";
        case PipelineType::DepthPrepassInstancedModelDoubleSided:
            return "DepthPrepassInstancedModelDoubleSided";
        case PipelineType::DepthPrepassTerrain:
            return "DepthPrepassTerrain";
        case PipelineType::DepthPrepassGrass:
            return "DepthPrepassGrass";
        case PipelineType::ModelDepthEqual:
            return "ModelDepthEqual";
        case PipelineType::ModelDoubleSidedDepthEqual:
            return "ModelDoubleSidedDepthEqual";
        case PipelineType::SkinnedModelDepthEqual:
            return "SkinnedModelDepthEqual";
        case PipelineType::SkinnedModelDoubleSidedDepthEqual:
            return "SkinnedModelDoubleSidedDepthEqual";
        case PipelineType::InstancedModelDepthEqual:
            return "InstancedModelDepthEqual";
        case PipelineType::InstancedModelDoubleSidedDepthEqual:
            return "InstancedModelDoubleSidedDepthEqual";
        case PipelineType::TerrainDepthEqual:
            return "TerrainDepthEqual";
        case PipelineType::GrassDepthEqual:
            return "GrassDepthEqual";
        case PipelineType::DepthPrepassModelMasked:
            return "DepthPrepassModelMasked";
        case PipelineType::DepthPrepassInstancedModelMasked:
            return "DepthPrepassInstancedModelMasked";
        case PipelineType::Water:
            return "Water";
        default:
            return "Unknown";
    }
}

/**
 * @brief Convert compute pipeline type to string for debugging
 */
inline const char* compute_pipeline_type_to_string(ComputePipelineType type) {
    switch (type) {
        case ComputePipelineType::ExposureHistogram:
            return "ExposureHistogram";
        case ComputePipelineType::ExposureAdapt:
            return "ExposureAdapt";
        case ComputePipelineType::WaterSim:
            return "WaterSim";
        default:
            return "Unknown";
    }
}

/**
 * @brief Registry for managing and caching GPU pipelines
 *
 * The PipelineRegistry provides a central location for creating and caching
 * all graphics + compute pipelines used by the renderer. It handles:
 * - Eager pre-creation of the full static set (preload_all_pipelines)
 * - Pipeline caching to avoid recreation
 * - Shader compilation and management
 * - Proper resource cleanup on shutdown
 *
 * Thread-safety tiers (see docs/multithreading_plan.md §6.1):
 * - **main-thread (asserted)**: init, shutdown, preload_all_pipelines,
 *   invalidate_all, set_swapchain_format, enable_hot_reload,
 *   reload_all_pipelines, and any get_pipeline/get_compute_pipeline call
 *   that has to CREATE a pipeline (cache miss).
 * - **worker-safe after preload**: get_pipeline/get_compute_pipeline on
 *   already-cached entries are read-only unordered_map lookups. The intended
 *   protocol is: preload the full enumeration at init (SceneRenderer does),
 *   after which the hot path never mutates the cache. A cache miss after
 *   preload is logged — it means the static enumeration is incomplete.
 *
 * Usage:
 *   PipelineRegistry registry;
 *   registry.init(device);
 *   registry.preload_all_pipelines();   // pre-create the full static set
 *
 *   // Read-only lookup on the hot path:
 *   auto* model_pipeline = registry.get_pipeline(PipelineType::Model);
 *
 *   // In render loop:
 *   model_pipeline->bind(render_pass);
 *
 *   registry.shutdown();
 */
class PipelineRegistry {
public:
    PipelineRegistry() = default;
    ~PipelineRegistry();

    // Non-copyable, non-movable (owns GPU resources)
    PipelineRegistry(const PipelineRegistry&) = delete;
    PipelineRegistry& operator=(const PipelineRegistry&) = delete;
    PipelineRegistry(PipelineRegistry&&) = delete;
    PipelineRegistry& operator=(PipelineRegistry&&) = delete;

    /**
     * @brief Initialize the pipeline registry
     *
     * Must be called before any pipelines can be retrieved. Main-thread
     * (asserted). Call preload_all_pipelines() afterwards (after
     * set_swapchain_format) so hot-path lookups stay read-only.
     *
     * @param device The GPU device to create pipelines on
     * @return true on success, false on failure
     */
    [[nodiscard]] bool init(GPUDevice& device);

    /**
     * @brief Shutdown and release all cached pipelines. Main-thread (asserted).
     */
    void shutdown();

    /**
     * @brief Check if the registry is initialized
     */
    bool is_initialized() const { return device_ != nullptr; }

    /**
     * @brief Get a pipeline by type
     *
     * After preload_all_pipelines() this is a read-only cache lookup
     * (worker-safe). A cache miss falls back to a guarded creation path:
     * main-thread asserted, and logged when it happens after a preload
     * (a lazy creation on the hot path means the static enumeration missed
     * a variant — a contract violation once recording is parallel).
     * Returns nullptr if the pipeline could not be created.
     *
     * @param type The pipeline type to retrieve
     * @return Pointer to the pipeline, or nullptr on failure
     */
    GPUPipeline* get_pipeline(PipelineType type);

    // Convenience accessors for common pipeline types
    GPUPipeline* get_model_pipeline() { return get_pipeline(PipelineType::Model); }
    GPUPipeline* get_skinned_model_pipeline() { return get_pipeline(PipelineType::SkinnedModel); }
    GPUPipeline* get_terrain_pipeline() { return get_pipeline(PipelineType::Terrain); }
    GPUPipeline* get_skybox_pipeline() { return get_pipeline(PipelineType::Skybox); }
    GPUPipeline* get_ui_pipeline() { return get_pipeline(PipelineType::UI); }
    GPUPipeline* get_grass_pipeline() { return get_pipeline(PipelineType::Grass); }
    GPUPipeline* get_particle_pipeline() { return get_pipeline(PipelineType::Particle); }
    GPUPipeline* get_text_pipeline() { return get_pipeline(PipelineType::Text); }
    GPUPipeline* get_grid_pipeline() { return get_pipeline(PipelineType::Grid); }
    GPUPipeline* get_instanced_model_pipeline() { return get_pipeline(PipelineType::InstancedModel); }
    GPUPipeline* get_instanced_shadow_model_pipeline() { return get_pipeline(PipelineType::InstancedShadowModel); }
    GPUPipeline* get_shadow_model_pipeline() { return get_pipeline(PipelineType::ShadowModel); }
    GPUPipeline* get_shadow_skinned_model_pipeline() { return get_pipeline(PipelineType::ShadowSkinnedModel); }
    GPUPipeline* get_shadow_terrain_pipeline() { return get_pipeline(PipelineType::ShadowTerrain); }
    GPUPipeline* get_gtao_pipeline() { return get_pipeline(PipelineType::GTAO); }
    GPUPipeline* get_gtao_temporal_pipeline() { return get_pipeline(PipelineType::GTAOTemporal); }
    GPUPipeline* get_gtao_upsample_pipeline() { return get_pipeline(PipelineType::GTAOUpsample); }
    GPUPipeline* get_blur_ao_pipeline() { return get_pipeline(PipelineType::BlurAO); }
    GPUPipeline* get_composite_pipeline() { return get_pipeline(PipelineType::Composite); }
    GPUPipeline* get_bloom_downsample_pipeline() { return get_pipeline(PipelineType::BloomDownsample); }
    GPUPipeline* get_bloom_upsample_pipeline() { return get_pipeline(PipelineType::BloomUpsample); }
    GPUPipeline* get_volumetric_fog_pipeline() { return get_pipeline(PipelineType::VolumetricFog); }
    GPUPipeline* get_fxaa_pipeline() { return get_pipeline(PipelineType::Fxaa); }
    GPUPipeline* get_taa_resolve_pipeline() { return get_pipeline(PipelineType::TaaResolve); }
    GPUPipeline* get_taa_sharpen_pipeline() { return get_pipeline(PipelineType::TaaSharpen); }
    GPUPipeline* get_sky_capture_pipeline() { return get_pipeline(PipelineType::SkyCapture); }
    GPUPipeline* get_ibl_specular_prefilter_pipeline() { return get_pipeline(PipelineType::IblSpecularPrefilter); }
    GPUPipeline* get_ibl_irradiance_pipeline() { return get_pipeline(PipelineType::IblIrradiance); }
    GPUPipeline* get_ibl_brdf_lut_pipeline() { return get_pipeline(PipelineType::IblBrdfLut); }

    /**
     * @brief Get a compute pipeline by type
     *
     * Same contract as get_pipeline: read-only lookup after preload
     * (worker-safe), guarded main-thread-asserted creation on a miss,
     * rebuilt by the same hot-reload/invalidation paths as the graphics
     * pipelines. Returns nullptr if the pipeline could not be created.
     */
    GPUComputePipeline* get_compute_pipeline(ComputePipelineType type);

    /**
     * @brief Pre-create the full static pipeline set (graphics + compute)
     *
     * Main-thread (asserted). Call during loading, after set_swapchain_format.
     * Once this returns, every enumerated PipelineType/ComputePipelineType is
     * cached and get_pipeline()/get_compute_pipeline() are read-only lookups —
     * this is what deletes the lazy first-touch creation race.
     * Returns false if any pipeline fails to create.
     *
     * @return true if all pipelines created successfully
     */
    [[nodiscard]] bool preload_all_pipelines();

    /**
     * @brief Get the number of cached pipelines (graphics + compute)
     */
    size_t cached_pipeline_count() const { return pipelines_.size() + compute_pipelines_.size(); }

    /**
     * @brief Total pipelines created since init (graphics + compute)
     *
     * Monotonic creation counter — unlike cached_pipeline_count() it is not
     * reduced by invalidation. Lets tests (and diagnostics) verify that no
     * lazy creation happened during a get_pipeline sequence: if the count is
     * unchanged, every get was a pure cache lookup.
     */
    size_t creation_count() const { return creation_count_; }

    /**
     * @brief True once preload_all_pipelines() has completed
     *
     * While true, any creation inside get_pipeline/get_compute_pipeline is
     * logged as an enumeration gap.
     */
    bool is_preloaded() const { return preloaded_; }

    /**
     * @brief Invalidate all cached pipelines
     *
     * Main-thread (asserted). Call this if the shader binaries changed
     * (hot reload). Follow with preload_all_pipelines() — the registry drops
     * back to not-preloaded until then, and lookups may hit the guarded
     * creation path.
     */
    void invalidate_all();

    /**
     * @brief Set the swapchain color format
     *
     * Main-thread (asserted). Ideally called before pipelines are created. If
     * called after, only the pipelines that actually target the swapchain
     * format (UI, Text, Composite, Fxaa) are affected; scene/HDR/post
     * pipelines render to fixed-format offscreen targets and stay cached.
     * When the registry is already preloaded the affected pipelines are
     * re-created eagerly here (not lazily on next access), so hot-path
     * lookups stay read-only.
     */
    void set_swapchain_format(SDL_GPUTextureFormat format);

    /**
     * @brief Get the current swapchain format
     */
    SDL_GPUTextureFormat get_swapchain_format() const { return swapchain_format_; }

    /**
     * @brief Get the HDR scene color format (used for offscreen scene target).
     * Pipelines that render the 3D scene (Model, Terrain, Grass, Skybox, ...)
     * are created with this format; the composite pass then tonemaps to swapchain.
     */
    SDL_GPUTextureFormat get_scene_color_format() const { return scene_color_format_; }

    /**
     * @brief Get the velocity (motion-vector) target format written by the
     * depth-prepass pipelines. The SceneRenderer's velocity texture must be
     * created with this format.
     */
    SDL_GPUTextureFormat get_velocity_format() const { return velocity_format_; }

    /**
     * @brief Enable shader hot-reload by watching the SPIRV directory.
     *
     * Main-thread (asserted) — like all FileWatcher entry points. On any .spv
     * change, all cached pipelines are invalidated and eagerly pre-loaded
     * again (the watcher applies callbacks on the main thread at the frame
     * fence). Coarse but correct; live pipelines from the previous frame are
     * released by SDL once the in-flight commands finish.
     */
    void enable_hot_reload(core::asset::FileWatcher& watcher);

    /**
     * @brief Force a hot-reload pass right now (mainly for tests/diagnostics).
     *
     * Main-thread (asserted). Invalidates and eagerly re-creates the full
     * set — never leaves lazy creation for the hot path.
     */
    void reload_all_pipelines();

    /**
     * @brief Path on disk where SPIRV shaders are loaded from.
     */
    const std::string& shader_path() const { return shader_path_; }

private:
    // Guarded creation path (cache-miss side of get_pipeline /
    // get_compute_pipeline, plus the eager recreation in
    // set_swapchain_format). Main-thread asserted; bumps creation_count_.
    GPUPipeline* create_and_cache_pipeline(PipelineType type);
    GPUComputePipeline* create_and_cache_compute_pipeline(ComputePipelineType type);

    // Dispatch to the per-type creator (the static enumeration).
    std::unique_ptr<GPUPipeline> create_pipeline_uncached(PipelineType type);
    std::unique_ptr<GPUComputePipeline> create_compute_pipeline_uncached(ComputePipelineType type);

    // Pipeline creation methods. The model-family creators take the glTF
    // material variant flags (alphaMode BLEND -> alpha_blend, doubleSided ->
    // no back-face culling). depth_equal selects the post-prepass forward
    // variant (depth test EQUAL, writes off); mutually exclusive with
    // alpha_blend (BLEND draws are excluded from the prepass).
    std::unique_ptr<GPUPipeline> create_model_pipeline(bool alpha_blend = false, bool double_sided = false,
                                                       bool depth_equal = false);
    std::unique_ptr<GPUPipeline> create_skinned_model_pipeline(bool alpha_blend = false, bool double_sided = false,
                                                               bool depth_equal = false);
    std::unique_ptr<GPUPipeline> create_terrain_pipeline(bool depth_equal = false);
    std::unique_ptr<GPUPipeline> create_skybox_pipeline();
    std::unique_ptr<GPUPipeline> create_grid_pipeline();
    std::unique_ptr<GPUPipeline> create_ui_pipeline();
    std::unique_ptr<GPUPipeline> create_text_pipeline();
    std::unique_ptr<GPUPipeline> create_particle_pipeline();
    std::unique_ptr<GPUPipeline> create_grass_pipeline(bool depth_equal = false);
    std::unique_ptr<GPUPipeline> create_instanced_model_pipeline(bool double_sided = false, bool depth_equal = false);

    // Depth-prepass creators (position-only VS + shared velocity fragment).
    std::unique_ptr<GPUPipeline> create_prepass_model_pipeline(bool double_sided = false);
    std::unique_ptr<GPUPipeline> create_prepass_skinned_model_pipeline(bool double_sided = false);
    std::unique_ptr<GPUPipeline> create_prepass_instanced_model_pipeline(bool double_sided = false);
    std::unique_ptr<GPUPipeline> create_prepass_terrain_pipeline();
    // Alpha-MASK prepass variants: texcoord passthrough + alpha-tested
    // fragment so foliage writes depth (GTAO + velocity coverage). Always
    // no-cull — MASK foliage is double-sided in practice.
    std::unique_ptr<GPUPipeline> create_prepass_model_masked_pipeline();
    std::unique_ptr<GPUPipeline> create_prepass_instanced_model_masked_pipeline();
    std::unique_ptr<GPUPipeline> create_prepass_grass_pipeline();
    std::unique_ptr<GPUPipeline> create_instanced_shadow_model_pipeline();
    std::unique_ptr<GPUPipeline> create_shadow_model_pipeline();
    std::unique_ptr<GPUPipeline> create_shadow_skinned_model_pipeline();
    std::unique_ptr<GPUPipeline> create_shadow_terrain_pipeline();
    std::unique_ptr<GPUPipeline> create_local_shadow_clear_pipeline();
    std::unique_ptr<GPUPipeline> create_gtao_pipeline();
    std::unique_ptr<GPUPipeline> create_gtao_temporal_pipeline();
    std::unique_ptr<GPUPipeline> create_gtao_upsample_pipeline();
    std::unique_ptr<GPUPipeline> create_blur_ao_pipeline();
    std::unique_ptr<GPUPipeline> create_composite_pipeline();
    std::unique_ptr<GPUPipeline> create_bloom_downsample_pipeline();
    std::unique_ptr<GPUPipeline> create_bloom_upsample_pipeline();
    std::unique_ptr<GPUPipeline> create_volumetric_fog_pipeline();
    std::unique_ptr<GPUPipeline> create_fxaa_pipeline();
    std::unique_ptr<GPUPipeline> create_taa_resolve_pipeline();
    std::unique_ptr<GPUPipeline> create_taa_sharpen_pipeline();
    std::unique_ptr<GPUPipeline> create_sky_capture_pipeline();
    std::unique_ptr<GPUPipeline> create_ibl_specular_prefilter_pipeline();
    std::unique_ptr<GPUPipeline> create_ibl_irradiance_pipeline();
    std::unique_ptr<GPUPipeline> create_ibl_brdf_lut_pipeline();
    std::unique_ptr<GPUPipeline> create_water_pipeline();

    // Compute pipeline creators (declared ComputeResources are validated
    // against SPIR-V reflection at creation — see gpu_compute.hpp).
    std::unique_ptr<GPUComputePipeline> create_exposure_histogram_pipeline();
    std::unique_ptr<GPUComputePipeline> create_exposure_adapt_pipeline();
    std::unique_ptr<GPUComputePipeline> create_water_sim_pipeline();

    GPUDevice* device_ = nullptr;
    SDL_GPUTextureFormat swapchain_format_ = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    // HDR scene target format: lighting/bloom/fog compute in linear HDR, composite tonemaps.
    SDL_GPUTextureFormat scene_color_format_ = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
    SDL_GPUTextureFormat depth_format_ = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    // Motion-vector target written by the depth-prepass pipelines (TAA input).
    SDL_GPUTextureFormat velocity_format_ = SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT;

    // Pipeline caches (compute shares the graphics invalidation/hot-reload
    // lifecycle; cached_pipeline_count() counts both). Mutated only on the
    // main thread (asserted on every insert/erase path); read-only lookups
    // are worker-safe once preloaded_ is true and no mutation is in flight.
    std::unordered_map<PipelineType, std::unique_ptr<GPUPipeline>> pipelines_;
    std::unordered_map<ComputePipelineType, std::unique_ptr<GPUComputePipeline>> compute_pipelines_;

    // Monotonic count of pipelines created since init (see creation_count()).
    // Plain size_t, not atomic: every writer is on the main-thread-asserted
    // creation path.
    size_t creation_count_ = 0;

    // True after preload_all_pipelines(); reset by invalidate_all/shutdown.
    // While true, a cache miss in get_*_pipeline logs an enumeration gap.
    bool preloaded_ = false;

    // Shader manager for loading shaders from files
    std::unique_ptr<ShaderManager> shader_manager_;

    // Base path for compiled SPIRV shader files (relative to working directory)
    std::string shader_path_ = "shaders/";

    // Hot-reload watcher (non-owning). uint32_t handle from FileWatcher.
    core::asset::FileWatcher* watcher_ = nullptr;
    uint32_t shader_watch_handle_ = 0;
};

} // namespace bracken::gpu
