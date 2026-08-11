#pragma once

#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/gpu_shader.hpp"
#include "bracken/gpu/host_pipeline.hpp"
#include "bracken/gpu/pipeline_catalog.hpp"
#include <cstdint>
#include <memory>
#include <span>
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
 * What each pipeline *is* lives in the catalogue (`gpu/pipeline_catalog.hpp`),
 * one constexpr row per PipelineType; this class names no individual pipeline
 * and holds no per-pipeline knowledge beyond the cache key. Adding a pipeline
 * is adding a row.
 *
 * HOST-REGISTERED PIPELINES (`gpu/host_pipeline.hpp`) are the second, open
 * half of that. A game describes a pipeline at runtime and gets back an opaque
 * handle; the catalogue enums are untouched and stay the closed, golden-pinned
 * set they are. Both halves live in ONE cache, keyed by PipelineCacheKey, so
 * invalidation, the swapchain-format rebuild, hot-reload and shutdown cover
 * host registrations by construction rather than by remembering them.
 *
 * Thread-safety tiers (see docs/multithreading_plan.md §6.1):
 * - **main-thread (asserted)**: init, shutdown, preload_all_pipelines,
 *   invalidate_all, set_swapchain_format, enable_hot_reload,
 *   reload_all_pipelines, add_shader_search_path, register_graphics_pipeline,
 *   register_compute_pipeline, unregister_pipeline, and any
 *   get_pipeline/get_compute_pipeline call that has to CREATE a pipeline
 *   (cache miss).
 * - **worker-safe after preload**: get_pipeline/get_compute_pipeline on
 *   already-cached entries are read-only unordered_map lookups. The intended
 *   protocol is: preload the full enumeration at init (SceneRenderer does),
 *   after which the hot path never mutates the cache. A cache miss after
 *   preload is logged — it means the static enumeration is incomplete. The
 *   handle-taking overloads are the same read-only lookup plus a generation
 *   check, and they NEVER create: a host registration that is not in the cache
 *   returns nullptr rather than dragging pipeline creation onto a worker.
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
     *
     * Also unwatches every shader root and releases every HOST registration —
     * those own GPU objects that die with the device, and their handles go
     * stale rather than dangling. The shader SEARCH ROOTS are deliberately
     * KEPT: they are configuration, not a resource, so a host that re-inits
     * does not have to re-add them, and init() rewrites element 0 in place so
     * the engine root stays first. `pipeline_registry.test.cpp`'s
     * ShutdownReleasesHostRegistrationsAndKeepsTheSearchRoots pins the split.
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
     * format are affected — which the catalogue answers
     * (`pipeline_targets_swapchain`), today {UI, Text, Composite, Fxaa};
     * scene/HDR/post pipelines render to fixed-format offscreen targets and
     * stay cached.
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
     * @brief Path on disk where the ENGINE's SPIRV shaders are loaded from.
     *
     * Always the first shader search root, and that ordering is load-bearing:
     * if a host root came first, a game shipping its own `model.frag.spv`
     * would silently replace PipelineType::Model's shader.
     */
    const std::string& shader_path() const { return shader_path_roots_.front(); }

    // =========================================================================
    // Host extension surface (gpu/host_pipeline.hpp)
    // =========================================================================

    /**
     * @brief Add a directory the shader loader searches, AFTER shader_path().
     *
     * Main-thread (asserted). Roots are tried in registration order with the
     * engine root always first. A trailing '/' is added if missing, so
     * "shaders_mygame" and "shaders_mygame/" are the same root.
     *
     * When hot-reload is already enabled the new root is watched immediately;
     * otherwise enable_hot_reload() picks it up along with every other root.
     *
     * Only HOST descs resolve through the search roots. Catalogue rows keep
     * concatenating shader_path() directly — unchanged, so a host cannot
     * perturb engine shader resolution even by accident.
     *
     * @return false if `dir` is empty or already a root (both are no-ops).
     */
    [[nodiscard]] bool add_shader_search_path(std::string dir);

    /// Every shader search root, engine root first, in resolution order.
    [[nodiscard]] std::span<const std::string> shader_search_paths() const { return shader_path_roots_; }

    /**
     * @brief Register a host graphics pipeline and create it now.
     *
     * Main-thread (asserted). Returns an INVALID handle — logging an ERROR and
     * without bumping creation_count() — when the registry has no device, when
     * `desc.name` is empty or duplicates a live registration, when either
     * shader name is empty or resolves against no search root, or when
     * pipeline creation fails.
     *
     * Creation here is deliberately NOT reported as an enumeration gap the way
     * a post-preload catalogue miss is: a host registration is by definition
     * outside the static enumeration, and hosts register from on_init() which
     * runs after SceneRenderer::init has already preloaded.
     *
     * The registry keeps `desc` by value; that copy is what lets
     * invalidate_all() / reload_all_pipelines() / set_swapchain_format()
     * rebuild the pipeline. Consequently the GPUPipeline* returned by
     * get_pipeline(handle) is invalidated by any of those three —
     * RE-FETCH BY HANDLE EVERY FRAME, exactly as every engine call site
     * re-fetches by enum.
     */
    [[nodiscard]] HostPipelineHandle register_graphics_pipeline(HostGraphicsPipelineDesc desc);

    /// Compute twin of register_graphics_pipeline. Same contract; note that
    /// compute creation additionally requires SPIR-V reflection to succeed
    /// (the threadgroup size lives only in the module).
    [[nodiscard]] HostComputePipelineHandle register_compute_pipeline(HostComputePipelineDesc desc);

    /**
     * @brief Look up a host pipeline by handle.
     *
     * Read-only and worker-safe once the pipeline exists — a map lookup plus a
     * generation check, no creation, no cache mutation. Returns nullptr for a
     * default-constructed handle, a handle whose registration was released
     * (its generation was bumped), or a registration whose pipeline failed to
     * rebuild after a hot reload.
     */
    [[nodiscard]] GPUPipeline* get_pipeline(HostPipelineHandle handle);
    [[nodiscard]] GPUComputePipeline* get_compute_pipeline(HostComputePipelineHandle handle);

    /// The desc a live handle was registered with, or nullptr. Read-only.
    [[nodiscard]] const HostGraphicsPipelineDesc* host_pipeline_desc(HostPipelineHandle handle) const;
    [[nodiscard]] const HostComputePipelineDesc* host_compute_pipeline_desc(HostComputePipelineHandle handle) const;

    /**
     * @brief Release a host registration and its pipeline. Main-thread
     * (asserted). Releasing an already-released or never-valid handle is a
     * logged no-op, not an abort — a host tearing down in an unknown order is
     * not a contract violation.
     */
    void unregister_pipeline(HostPipelineHandle handle);
    void unregister_pipeline(HostComputePipelineHandle handle);

    /// Live host registrations (graphics + compute). Not affected by
    /// invalidate_all(), which drops the PIPELINES and keeps the descs.
    [[nodiscard]] size_t host_pipeline_count() const {
        return host_graphics_.live_count() + host_compute_.live_count();
    }

private:
    // Guarded creation path (cache-miss side of get_pipeline /
    // get_compute_pipeline, plus the eager recreation in
    // set_swapchain_format). Main-thread asserted; bumps creation_count_.
    GPUPipeline* create_and_cache_pipeline(PipelineType type);
    GPUComputePipeline* create_and_cache_compute_pipeline(ComputePipelineType type);

    // Build one pipeline from its catalogue row (pipeline_catalog.hpp): resolve
    // the row's shaders through the ShaderManager, then hand the built config
    // to the GPU layer. The registry itself names no pipeline.
    std::unique_ptr<GPUPipeline> create_pipeline_uncached(PipelineType type);
    std::unique_ptr<GPUComputePipeline> create_compute_pipeline_uncached(ComputePipelineType type);

    // Host siblings of the two above. Separate functions rather than branches
    // inside them: the catalogue path must stay byte-identical, including its
    // shader-path concatenation and its enumeration-gap WARN.
    std::unique_ptr<GPUPipeline> create_host_pipeline_uncached(const HostGraphicsPipelineDesc& desc);
    std::unique_ptr<GPUComputePipeline> create_host_compute_uncached(const HostComputePipelineDesc& desc);

    // (Re)create host slot `slot` into the cache. Main-thread asserted; bumps
    // creation_count_ on success. Used by registration, preload and the
    // swapchain rebuild, so there is one definition of "build this one".
    bool create_and_cache_host_pipeline(uint32_t slot);
    bool create_and_cache_host_compute(uint32_t slot);

    // First search root under which `file` exists, joined with `file`. Empty
    // (and logged) when no root has it. HOST descs only — see shader_path().
    [[nodiscard]] std::string resolve_shader_path(const std::string& file) const;

    // Watch every search root that is not watched yet. No-op without a
    // watcher. Returns the number of roots newly watched.
    size_t watch_unwatched_roots();

    // The formats catalogue rows are resolved against. Only the swapchain
    // entry can change while the app is running (set_swapchain_format).
    PipelineFormats formats() const {
        return {.swapchain = swapchain_format_,
                .scene_color = scene_color_format_,
                .velocity = velocity_format_,
                .depth = depth_format_};
    }

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
    //
    // Keyed by PipelineCacheKey rather than by the enums so that catalogue and
    // host pipelines share one lifecycle. The two ordinal spaces are kept
    // disjoint by the key's `origin` tag.
    std::unordered_map<PipelineCacheKey, std::unique_ptr<GPUPipeline>> pipelines_;
    std::unordered_map<PipelineCacheKey, std::unique_ptr<GPUComputePipeline>> compute_pipelines_;

    // Host registrations. The DESCS live here and outlive invalidation; the
    // built pipelines live in the two caches above.
    HostPipelineTable<HostGraphicsPipelineTag, HostGraphicsPipelineDesc> host_graphics_;
    HostPipelineTable<HostComputePipelineTag, HostComputePipelineDesc> host_compute_;

    // Monotonic count of pipelines created since init (see creation_count()).
    // Plain size_t, not atomic: every writer is on the main-thread-asserted
    // creation path.
    size_t creation_count_ = 0;

    // True after preload_all_pipelines(); reset by invalidate_all/shutdown.
    // While true, a cache miss in get_*_pipeline logs an enumeration gap.
    bool preloaded_ = false;

    // Shader manager for loading shaders from files
    std::unique_ptr<ShaderManager> shader_manager_;

    // Shader search roots, in resolution order. Element 0 is the ENGINE root
    // (what shader_path() returns) and is never removed, which is why every
    // access can use front() without a guard; init() rewrites it to the
    // executable-relative path. Elements 1.. are host roots.
    std::vector<std::string> shader_path_roots_{std::string("shaders/")};

    // Hot-reload watcher (non-owning). Parallel to shader_path_roots_ — one watch
    // handle per root, k_invalid_handle for a root that is not watched yet
    // (either because the FileWatcher was not ready or because the root was
    // added after enable_hot_reload). Kept the same length as shader_path_roots_.
    core::asset::FileWatcher* watcher_ = nullptr;
    std::vector<uint32_t> shader_watch_handles_{0};
};

} // namespace bracken::gpu
