#pragma once

#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/gpu_shader.hpp"
#include "bracken/gpu/pipeline_catalog.hpp"
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

    // TEMPORARY (refactor phase 6c commit 1): walk every creator with canonical
    // formats and write the resulting GPU state to `out_path`, so the tabulated
    // catalogue that replaces the creators can be diffed against it field by
    // field. Driven by BRACKEN_DUMP_PIPELINE_GOLDEN; deleted in commit 3.
    void dump_config_golden(const std::string& out_path);

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
