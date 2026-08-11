#include "pipeline_registry.hpp"
#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_stage.hpp"
#include "bracken/core/asset/file_watcher.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/gpu_shader.hpp"
#include "SDL3/SDL_gpu.h"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <SDL3/SDL_filesystem.h>
#include <string>
#include <utility>

namespace bracken::gpu {

namespace {
constexpr const char* LOG_TAG = "gpu";

// shader_watch_handles_ is initialized in the header with a literal 0 so the
// header does not have to include the FileWatcher. Pin that the literal really
// is the watcher's "no handle" value.
static_assert(core::asset::FileWatcher::k_invalid_handle == 0,
              "PipelineRegistry::shader_watch_handles_ seeds unwatched roots with a literal 0");

/// Normalize a search root to a directory prefix that can be concatenated with
/// a file name. Empty stays empty (the caller rejects it).
std::string with_trailing_separator(std::string dir) {
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') {
        dir.push_back('/');
    }
    return dir;
}
} // namespace

// =============================================================================
// PipelineRegistry Implementation
// =============================================================================

PipelineRegistry::~PipelineRegistry() {
    shutdown();
}

bool PipelineRegistry::init(GPUDevice& device) {
    // GPU resource creation/destruction is main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (device_) {
        BRACKEN_LOG_WARN(LOG_TAG, "PipelineRegistry: Already initialized");
        return false;
    }

    device_ = &device;

    // Get swapchain format from device
    swapchain_format_ = device_->swapchain_format();

    // Set shader path relative to executable location. Element 0 of
    // shader_path_roots_ IS the engine root — rewritten, never appended, so host
    // roots can never be searched ahead of it.
    const char* base_path = SDL_GetBasePath();
    if (base_path) {
        shader_path_roots_[0] = std::string(base_path) + "shaders/";
    } else {
        shader_path_roots_[0] = "shaders/";
        BRACKEN_LOG_WARN(LOG_TAG, "PipelineRegistry: SDL_GetBasePath failed, using relative path");
    }

    // Create shader manager (shaders are pre-compiled to SPIRV at build time)
    shader_manager_ = std::make_unique<ShaderManager>(device);

    BRACKEN_LOG_INFO(LOG_TAG, "PipelineRegistry: Initialized with shader path: {}", shader_path().c_str());

    return true;
}

void PipelineRegistry::shutdown() {
    if (!device_ && pipelines_.empty() && compute_pipelines_.empty() && !shader_manager_) {
        return; // Already shut down
    }
    // GPU resource destruction is main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    // Stop watching before the caches go: a watch callback firing into a
    // half-torn-down registry would rebuild against a null device.
    if (watcher_ != nullptr) {
        for (uint32_t& handle : shader_watch_handles_) {
            if (handle != core::asset::FileWatcher::k_invalid_handle) {
                watcher_->unwatch(handle);
                handle = core::asset::FileWatcher::k_invalid_handle;
            }
        }
        watcher_ = nullptr;
    }
    pipelines_.clear();
    compute_pipelines_.clear();
    host_graphics_.clear();
    host_compute_.clear();
    shader_manager_.reset();
    device_ = nullptr;
    preloaded_ = false;
    BRACKEN_LOG_INFO(LOG_TAG, "PipelineRegistry: Shutdown complete");
}

GPUPipeline* PipelineRegistry::get_pipeline(PipelineType type) {
    if (!device_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry: Not initialized");
        return nullptr;
    }

    // Hot path: read-only cache lookup — worker-safe once the full set was
    // preloaded (no mutation happens on a hit).
    auto it = pipelines_.find(PipelineCacheKey::catalog(type));
    if (it != pipelines_.end()) {
        return it->second.get();
    }

    return create_and_cache_pipeline(type);
}

GPUPipeline* PipelineRegistry::create_and_cache_pipeline(PipelineType type) {
    // Guarded creation path: cache mutation + SDL pipeline creation are
    // main-thread tier. From Phase 5 onward (parallel recording) reaching
    // this off the main thread is a contract violation, not just a race.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (preloaded_) {
        BRACKEN_LOG_WARN(LOG_TAG,
                         "PipelineRegistry: lazy creation of {} pipeline after preload — "
                         "the static pipeline enumeration is missing this variant",
                         pipeline_type_to_string(type));
    }

    std::unique_ptr<GPUPipeline> pipeline = create_pipeline_uncached(type);

    if (!pipeline) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry: Failed to create {} pipeline", pipeline_type_to_string(type));
        return nullptr;
    }

    BRACKEN_LOG_DEBUG(LOG_TAG, "PipelineRegistry: Created {} pipeline", pipeline_type_to_string(type));

    ++creation_count_;
    auto* result = pipeline.get();
    pipelines_[PipelineCacheKey::catalog(type)] = std::move(pipeline);
    return result;
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_pipeline_uncached(PipelineType type) {
    // Everything a pipeline *is* lives in one catalogue row; this function only
    // resolves the row's shaders and hands the built config to the GPU layer.
    const PipelineDesc& desc = pipeline_desc(type);
    if (desc.vertex_shader[0] == '\0' || desc.fragment_shader[0] == '\0') {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry: Unknown pipeline type {}", static_cast<int>(type));
        return nullptr;
    }

    auto* vs = shader_manager_->get(shader_path() + desc.vertex_shader, ShaderStage::Vertex, "VSMain", desc.vs);
    auto* fs = shader_manager_->get(shader_path() + desc.fragment_shader, ShaderStage::Fragment, "PSMain", desc.fs);
    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config = build_pipeline_config(desc, formats());
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();

    return GPUPipeline::create(*device_, config);
}

GPUComputePipeline* PipelineRegistry::get_compute_pipeline(ComputePipelineType type) {
    if (!device_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry: Not initialized");
        return nullptr;
    }

    // Hot path: read-only cache lookup — worker-safe once preloaded.
    auto it = compute_pipelines_.find(PipelineCacheKey::catalog(type));
    if (it != compute_pipelines_.end()) {
        return it->second.get();
    }

    return create_and_cache_compute_pipeline(type);
}

GPUComputePipeline* PipelineRegistry::create_and_cache_compute_pipeline(ComputePipelineType type) {
    // Same guarded creation contract as create_and_cache_pipeline.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (preloaded_) {
        BRACKEN_LOG_WARN(LOG_TAG,
                         "PipelineRegistry: lazy creation of {} compute pipeline after preload — "
                         "the static pipeline enumeration is missing this variant",
                         compute_pipeline_type_to_string(type));
    }

    std::unique_ptr<GPUComputePipeline> pipeline = create_compute_pipeline_uncached(type);

    if (!pipeline) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry: Failed to create {} compute pipeline",
                          compute_pipeline_type_to_string(type));
        return nullptr;
    }

    BRACKEN_LOG_DEBUG(LOG_TAG, "PipelineRegistry: Created {} compute pipeline", compute_pipeline_type_to_string(type));

    ++creation_count_;
    auto* result = pipeline.get();
    compute_pipelines_[PipelineCacheKey::catalog(type)] = std::move(pipeline);
    return result;
}

std::unique_ptr<GPUComputePipeline>
PipelineRegistry::create_compute_pipeline_uncached(ComputePipelineType type) {
    // A compute module is the whole pipeline: no fixed-function state to build,
    // and the declared ComputeResources are validated against SPIR-V reflection
    // inside load_spirv (see gpu_compute.hpp).
    const ComputePipelineDesc& desc = compute_pipeline_desc(type);
    if (desc.shader[0] == '\0') {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry: Unknown compute pipeline type {}", static_cast<int>(type));
        return nullptr;
    }
    return GPUComputePipeline::load_spirv(*device_, shader_path() + desc.shader, desc.entry_point, desc.resources);
}

bool PipelineRegistry::preload_all_pipelines() {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (!device_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry: preload_all_pipelines called before init");
        return false;
    }
    BRACKEN_LOG_INFO(LOG_TAG, "PipelineRegistry: Preloading all pipelines...");

    bool success = true;
    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        auto type = static_cast<PipelineType>(i);
        if (!get_pipeline(type)) {
            BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry: Failed to preload {} pipeline",
                              pipeline_type_to_string(type));
            success = false;
        }
    }
    for (int i = 0; i < static_cast<int>(ComputePipelineType::Count); ++i) {
        auto type = static_cast<ComputePipelineType>(i);
        if (!get_compute_pipeline(type)) {
            BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry: Failed to preload {} compute pipeline",
                              compute_pipeline_type_to_string(type));
            success = false;
        }
    }

    // The full enumeration has been walked; from here on, get_pipeline /
    // get_compute_pipeline are read-only lookups and any creation is logged
    // as an enumeration gap. Set even on partial failure — a variant that
    // failed to create here will fail identically on a lazy retry.
    preloaded_ = true;

    // Host registrations come AFTER the latch, and deliberately so: they are
    // outside the static enumeration by definition, so rebuilding them must
    // not be reported as an enumeration gap. This is also the path that puts
    // host pipelines back after invalidate_all() — which is what makes
    // reload_all_pipelines() (hot reload) cover them for free.
    for (uint32_t slot = 0; slot < static_cast<uint32_t>(host_graphics_.slots().size()); ++slot) {
        const auto& entry = host_graphics_.slots()[slot];
        if (!entry->live || pipelines_.contains(PipelineCacheKey::host(slot))) {
            continue;
        }
        if (!create_and_cache_host_pipeline(slot)) {
            BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry: Failed to preload host pipeline '{}'",
                              entry->desc.name);
            success = false;
        }
    }
    for (uint32_t slot = 0; slot < static_cast<uint32_t>(host_compute_.slots().size()); ++slot) {
        const auto& entry = host_compute_.slots()[slot];
        if (!entry->live || compute_pipelines_.contains(PipelineCacheKey::host(slot))) {
            continue;
        }
        if (!create_and_cache_host_compute(slot)) {
            BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry: Failed to preload host compute pipeline '{}'",
                              entry->desc.name);
            success = false;
        }
    }

    BRACKEN_LOG_INFO(LOG_TAG, "PipelineRegistry: Preloaded {} pipelines",
                     pipelines_.size() + compute_pipelines_.size());

    return success;
}

void PipelineRegistry::invalidate_all() {
    // Cache mutation + (deferred) GPU pipeline release: main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    // One cache holds catalogue and host pipelines alike, so this drops both.
    // The host DESCS are untouched: preload_all_pipelines() rebuilds from
    // them, which is the whole reason the registry keeps them by value.
    pipelines_.clear();
    compute_pipelines_.clear();
    if (shader_manager_) {
        shader_manager_->clear_cache();
    }
    preloaded_ = false; // caller must preload_all_pipelines() again
    BRACKEN_LOG_INFO(LOG_TAG, "PipelineRegistry: All pipelines invalidated");
}

void PipelineRegistry::set_swapchain_format(SDL_GPUTextureFormat format) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (format == swapchain_format_) {
        return;
    }
    swapchain_format_ = format;

    // Which pipelines depend on the swapchain format is a property of the
    // catalogue, not a list maintained next to it: everything else renders to
    // fixed-format offscreen targets (HDR scene, R8 AO, depth) and would be
    // rebuilt for nothing.
    std::vector<PipelineType> swapchain_dependent;
    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        const auto candidate = static_cast<PipelineType>(i);
        if (pipeline_targets_swapchain(candidate)) {
            swapchain_dependent.push_back(candidate);
        }
    }
    size_t invalidated = 0;
    for (PipelineType type : swapchain_dependent) {
        invalidated += pipelines_.erase(PipelineCacheKey::catalog(type));
    }

    // Host registrations that target the swapchain, found by a SECOND walk.
    // Deliberately not by widening pipeline_targets_swapchain(): that query is
    // pinned by gpu/pipeline_catalog.test.cpp to answer for exactly four enum
    // values, and it must keep meaning "a catalogue row targets the swapchain".
    //
    // Only TargetFormat::Swapchain qualifies, and it qualifies on EITHER colour
    // attachment: a host that names it as `color2` is exactly as dependent on
    // the presentable format as one that names it as `color`. A host that
    // hard-codes the presentable format through the matching *_format_override
    // has opted out of being rebuilt, because the registry cannot tell that
    // format apart from any other one the host chose.
    std::vector<uint32_t> host_swapchain_dependent;
    for (uint32_t slot = 0; slot < static_cast<uint32_t>(host_graphics_.slots().size()); ++slot) {
        const auto& entry = host_graphics_.slots()[slot];
        if (!entry->live) {
            continue;
        }
        const HostGraphicsPipelineDesc& desc = entry->desc;
        const bool color_depends =
            desc.color == TargetFormat::Swapchain && desc.color_format_override == SDL_GPU_TEXTUREFORMAT_INVALID;
        const bool color2_depends =
            desc.color2 == TargetFormat::Swapchain && desc.color2_format_override == SDL_GPU_TEXTUREFORMAT_INVALID;
        if (color_depends || color2_depends) {
            host_swapchain_dependent.push_back(slot);
        }
    }
    for (uint32_t slot : host_swapchain_dependent) {
        invalidated += pipelines_.erase(PipelineCacheKey::host(slot));
    }

    BRACKEN_LOG_INFO(LOG_TAG,
                     "PipelineRegistry: Swapchain format changed, invalidated {} swapchain-dependent pipelines",
                     invalidated);

    // Keep the pre-creation contract: if the registry was already preloaded,
    // re-create the affected pipelines now instead of leaving lazy creation
    // for the next hot-path lookup. Drop preloaded_ for the duration so this
    // intentional eager re-creation is not logged as an enumeration gap.
    if (preloaded_ && invalidated > 0) {
        preloaded_ = false;
        for (PipelineType type : swapchain_dependent) {
            if (!create_and_cache_pipeline(type)) {
                BRACKEN_LOG_ERROR(LOG_TAG,
                                  "PipelineRegistry: Failed to re-create {} pipeline after swapchain format change",
                                  pipeline_type_to_string(type));
            }
        }
        for (uint32_t slot : host_swapchain_dependent) {
            if (!create_and_cache_host_pipeline(slot)) {
                BRACKEN_LOG_ERROR(
                    LOG_TAG, "PipelineRegistry: Failed to re-create host pipeline '{}' after swapchain format change",
                    host_graphics_.slots()[slot]->desc.name);
            }
        }
        preloaded_ = true;
    }
}

size_t PipelineRegistry::watch_unwatched_roots() {
    // Callers all sit behind a main-thread assert; this is a private helper.
    if (watcher_ == nullptr) {
        return 0;
    }
    shader_watch_handles_.resize(shader_path_roots_.size(), core::asset::FileWatcher::k_invalid_handle);
    size_t newly_watched = 0;
    for (size_t i = 0; i < shader_path_roots_.size(); ++i) {
        if (shader_watch_handles_[i] != core::asset::FileWatcher::k_invalid_handle) {
            continue; // already watched — never double-watch, that is two rebuilds per save
        }
        const uint32_t handle =
            watcher_->watch_directory(shader_path_roots_[i], ".spv", [this](const std::filesystem::path& p) {
                BRACKEN_LOG_INFO("hot_reload", "Shader changed: {} -- rebuilding pipelines", p.filename().string());
                reload_all_pipelines();
            });
        if (handle == core::asset::FileWatcher::k_invalid_handle) {
            // Leave the slot invalid so a later enable_hot_reload (e.g. after
            // the host initializes the FileWatcher, or after a host root's
            // directory is created) retries THIS root without disturbing the
            // roots that did take.
            BRACKEN_LOG_WARN("hot_reload",
                             "PipelineRegistry: watch_directory failed for {} (FileWatcher not initialized, or the "
                             "directory does not exist); shader hot-reload NOT enabled for that root",
                             shader_path_roots_[i]);
            continue;
        }
        shader_watch_handles_[i] = handle;
        ++newly_watched;
        BRACKEN_LOG_INFO("hot_reload", "PipelineRegistry hot-reload enabled at {}", shader_path_roots_[i]);
    }
    return newly_watched;
}

void PipelineRegistry::enable_hot_reload(core::asset::FileWatcher& watcher) {
    // FileWatcher entry points are main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (!device_) {
        BRACKEN_LOG_WARN("hot_reload", "PipelineRegistry::enable_hot_reload called before init");
        return;
    }
    if (watcher_ != nullptr && watcher_ != &watcher) {
        BRACKEN_LOG_WARN("hot_reload",
                         "PipelineRegistry::enable_hot_reload called with a second FileWatcher; ignoring it "
                         "(the registry watches its roots through exactly one watcher)");
        return;
    }
    // The old single-root latch (`if (watcher_) return;`) doubled as the retry
    // path when the FileWatcher was not yet initialized. With N roots the
    // retry has to be PER ROOT — an already-watched root is skipped inside
    // watch_unwatched_roots(), so calling this again is cheap and idempotent,
    // and a root added later by add_shader_search_path still gets picked up.
    watcher_ = &watcher;
    watch_unwatched_roots();
    const bool any_watched =
        std::any_of(shader_watch_handles_.begin(), shader_watch_handles_.end(),
                    [](uint32_t h) { return h != core::asset::FileWatcher::k_invalid_handle; });
    if (!any_watched) {
        // Nothing took: drop the watcher again so the next call is a clean
        // retry rather than a no-op behind a latch. This is the behaviour the
        // single-root version had, preserved.
        watcher_ = nullptr;
    }
}

void PipelineRegistry::reload_all_pipelines() {
    // Hot-reload apply is main-thread tier (the FileWatcher hands callbacks
    // to the main thread); re-creation is eager, never left to lazy lookup.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (!device_) {
        return;
    }
    invalidate_all();
    if (!preload_all_pipelines()) {
        BRACKEN_LOG_WARN("hot_reload", "PipelineRegistry: one or more pipelines failed to rebuild");
    }
}

// =============================================================================
// Host-registered pipelines (gpu/host_pipeline.hpp)
// =============================================================================

bool PipelineRegistry::add_shader_search_path(std::string dir) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (dir.empty()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry::add_shader_search_path: empty directory");
        return false;
    }
    std::string root = with_trailing_separator(std::move(dir));
    if (std::find(shader_path_roots_.begin(), shader_path_roots_.end(), root) != shader_path_roots_.end()) {
        // Not an error: a host that calls this twice (or names the engine root)
        // gets the same resolution order either way. Refusing keeps the list a
        // set, which is what makes double-watching impossible.
        BRACKEN_LOG_DEBUG(LOG_TAG, "PipelineRegistry::add_shader_search_path: '{}' is already a search root", root);
        return false;
    }
    shader_path_roots_.push_back(std::move(root));
    shader_watch_handles_.resize(shader_path_roots_.size(), core::asset::FileWatcher::k_invalid_handle);
    // Watch it now if hot-reload is already live; otherwise enable_hot_reload
    // picks it up with everything else.
    watch_unwatched_roots();
    BRACKEN_LOG_INFO(LOG_TAG, "PipelineRegistry: shader search root {} added at position {}",
                     shader_path_roots_.back(), shader_path_roots_.size() - 1);
    return true;
}

std::string PipelineRegistry::resolve_shader_path(const std::string& file) const {
    std::error_code ec;
    for (const std::string& root : shader_path_roots_) {
        std::string candidate = root + file;
        if (std::filesystem::exists(std::filesystem::path(candidate), ec)) {
            return candidate;
        }
    }
    BRACKEN_LOG_ERROR(LOG_TAG,
                      "PipelineRegistry: shader '{}' not found under any of the {} shader search root(s); "
                      "did the host call add_shader_search_path and does the build stage the .spv there?",
                      file, shader_path_roots_.size());
    return {};
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_host_pipeline_uncached(const HostGraphicsPipelineDesc& desc) {
    const std::string vs_path = resolve_shader_path(desc.vertex_shader);
    const std::string fs_path = resolve_shader_path(desc.fragment_shader);
    if (vs_path.empty() || fs_path.empty()) {
        return nullptr; // resolve_shader_path already logged which one
    }

    // Reflected mode: read the counts out of the module and hand them BACK in
    // as the declared ones. Without this, a host that (reasonably) does not
    // hand-count its samplers gets a declared/reflected mismatch ERROR logged
    // on every single load, which trains people to ignore that error — and it
    // is the engine's own most valuable shader diagnostic.
    ShaderResources vs_resources = desc.vs;
    ShaderResources fs_resources = desc.fs;
    if (desc.reflect_resources) {
        if (!GPUShader::reflect_spirv_file(vs_path, vs_resources) ||
            !GPUShader::reflect_spirv_file(fs_path, fs_resources)) {
            BRACKEN_LOG_ERROR(LOG_TAG,
                              "PipelineRegistry: host pipeline '{}' asked for reflected resource counts and "
                              "reflection failed; set reflect_resources=false and declare them to proceed",
                              desc.name);
            return nullptr;
        }
    }

    auto* vs = shader_manager_->get(vs_path, ShaderStage::Vertex, desc.vertex_entry, vs_resources);
    auto* fs = shader_manager_->get(fs_path, ShaderStage::Fragment, desc.fragment_entry, fs_resources);
    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config = build_host_pipeline_config(desc, formats());
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUComputePipeline>
PipelineRegistry::create_host_compute_uncached(const HostComputePipelineDesc& desc) {
    const std::string path = resolve_shader_path(desc.shader);
    if (path.empty()) {
        return nullptr;
    }
    ComputeResources resources = desc.resources;
    if (desc.reflect_resources && !GPUComputePipeline::reflect_spirv_file(path, resources)) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "PipelineRegistry: host compute pipeline '{}' asked for reflected resource counts and "
                          "reflection failed",
                          desc.name);
        return nullptr;
    }
    return GPUComputePipeline::load_spirv(*device_, path, desc.entry_point, resources);
}

bool PipelineRegistry::create_and_cache_host_pipeline(uint32_t slot) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    BRACKEN_ASSERT(slot < host_graphics_.slots().size(), "host graphics slot out of range");
    const auto& entry = host_graphics_.slots()[slot];
    // Note: NO enumeration-gap WARN here. A host registration is outside the
    // static enumeration by construction, so creating one after preload is the
    // normal case, not a symptom.
    std::unique_ptr<GPUPipeline> pipeline = create_host_pipeline_uncached(entry->desc);
    if (!pipeline) {
        return false;
    }
    ++creation_count_;
    pipelines_[PipelineCacheKey::host(slot)] = std::move(pipeline);
    BRACKEN_LOG_DEBUG(LOG_TAG, "PipelineRegistry: Created host pipeline '{}'", entry->desc.name);
    return true;
}

bool PipelineRegistry::create_and_cache_host_compute(uint32_t slot) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    BRACKEN_ASSERT(slot < host_compute_.slots().size(), "host compute slot out of range");
    const auto& entry = host_compute_.slots()[slot];
    std::unique_ptr<GPUComputePipeline> pipeline = create_host_compute_uncached(entry->desc);
    if (!pipeline) {
        return false;
    }
    ++creation_count_;
    compute_pipelines_[PipelineCacheKey::host(slot)] = std::move(pipeline);
    BRACKEN_LOG_DEBUG(LOG_TAG, "PipelineRegistry: Created host compute pipeline '{}'", entry->desc.name);
    return true;
}

HostPipelineHandle PipelineRegistry::register_graphics_pipeline(HostGraphicsPipelineDesc desc) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    // Desc validation FIRST, and deliberately ahead of the device check: it is
    // pure (gpu/host_pipeline.hpp), so a malformed desc is refused for the same
    // stated reason whether or not the registry is up — which is also what
    // makes every refusal reason reachable from a device-free test.
    if (const char* why = host_pipeline_desc_error(desc); why != nullptr) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry::register_graphics_pipeline('{}'): {}", desc.name, why);
        return HostPipelineHandle::invalid();
    }
    if (!device_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry::register_graphics_pipeline: registry not initialized");
        return HostPipelineHandle::invalid();
    }
    if (host_graphics_.has_live_name(desc.name)) {
        BRACKEN_LOG_ERROR(LOG_TAG,
                          "PipelineRegistry::register_graphics_pipeline: '{}' is already registered; names must be "
                          "unique so a log line identifies one pipeline",
                          desc.name);
        return HostPipelineHandle::invalid();
    }

    const HostPipelineHandle handle = host_graphics_.add(std::move(desc));
    if (!create_and_cache_host_pipeline(handle.index)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry::register_graphics_pipeline: '{}' failed to create",
                          host_graphics_.slots()[handle.index]->desc.name);
        (void)host_graphics_.remove(handle);
        return HostPipelineHandle::invalid();
    }
    return handle;
}

HostComputePipelineHandle PipelineRegistry::register_compute_pipeline(HostComputePipelineDesc desc) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (const char* why = host_compute_pipeline_desc_error(desc); why != nullptr) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry::register_compute_pipeline('{}'): {}", desc.name, why);
        return HostComputePipelineHandle::invalid();
    }
    if (!device_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry::register_compute_pipeline: registry not initialized");
        return HostComputePipelineHandle::invalid();
    }
    if (host_compute_.has_live_name(desc.name)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry::register_compute_pipeline: '{}' is already registered",
                          desc.name);
        return HostComputePipelineHandle::invalid();
    }

    const HostComputePipelineHandle handle = host_compute_.add(std::move(desc));
    if (!create_and_cache_host_compute(handle.index)) {
        BRACKEN_LOG_ERROR(LOG_TAG, "PipelineRegistry::register_compute_pipeline: '{}' failed to create",
                          host_compute_.slots()[handle.index]->desc.name);
        (void)host_compute_.remove(handle);
        return HostComputePipelineHandle::invalid();
    }
    return handle;
}

GPUPipeline* PipelineRegistry::get_pipeline(HostPipelineHandle handle) {
    // Read-only: generation check + one map lookup. NEVER creates — pipeline
    // creation is main-thread tier and this overload is explicitly documented
    // as worker-safe, so a miss returns nullptr instead of quietly violating
    // the tier from a render job.
    if (host_graphics_.find(handle) == nullptr) {
        return nullptr;
    }
    auto it = pipelines_.find(PipelineCacheKey::host(handle.index));
    return it != pipelines_.end() ? it->second.get() : nullptr;
}

GPUComputePipeline* PipelineRegistry::get_compute_pipeline(HostComputePipelineHandle handle) {
    if (host_compute_.find(handle) == nullptr) {
        return nullptr;
    }
    auto it = compute_pipelines_.find(PipelineCacheKey::host(handle.index));
    return it != compute_pipelines_.end() ? it->second.get() : nullptr;
}

const HostGraphicsPipelineDesc* PipelineRegistry::host_pipeline_desc(HostPipelineHandle handle) const {
    const auto* entry = host_graphics_.find(handle);
    return entry != nullptr ? &entry->desc : nullptr;
}

const HostComputePipelineDesc* PipelineRegistry::host_compute_pipeline_desc(HostComputePipelineHandle handle) const {
    const auto* entry = host_compute_.find(handle);
    return entry != nullptr ? &entry->desc : nullptr;
}

void PipelineRegistry::unregister_pipeline(HostPipelineHandle handle) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (!host_graphics_.remove(handle)) {
        BRACKEN_LOG_DEBUG(LOG_TAG, "PipelineRegistry::unregister_pipeline: handle is not a live registration");
        return;
    }
    pipelines_.erase(PipelineCacheKey::host(handle.index));
}

void PipelineRegistry::unregister_pipeline(HostComputePipelineHandle handle) {
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (!host_compute_.remove(handle)) {
        BRACKEN_LOG_DEBUG(LOG_TAG, "PipelineRegistry::unregister_pipeline: handle is not a live registration");
        return;
    }
    compute_pipelines_.erase(PipelineCacheKey::host(handle.index));
}

} // namespace bracken::gpu
