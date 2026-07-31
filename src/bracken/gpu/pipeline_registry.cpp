#include "pipeline_registry.hpp"
#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/job_stage.hpp"
#include "bracken/core/asset/file_watcher.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/gpu_shader.hpp"
#include "SDL3/SDL_gpu.h"
#include <memory>
#include <SDL3/SDL_filesystem.h>
#include <string>
#include <utility>

namespace bracken::gpu {

namespace {
constexpr const char* LOG_TAG = "gpu";
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

    // Set shader path relative to executable location
    const char* base_path = SDL_GetBasePath();
    if (base_path) {
        shader_path_ = std::string(base_path) + "shaders/";
    } else {
        shader_path_ = "shaders/";
        BRACKEN_LOG_WARN(LOG_TAG, "PipelineRegistry: SDL_GetBasePath failed, using relative path");
    }

    // Create shader manager (shaders are pre-compiled to SPIRV at build time)
    shader_manager_ = std::make_unique<ShaderManager>(device);

    BRACKEN_LOG_INFO(LOG_TAG, "PipelineRegistry: Initialized with shader path: {}", shader_path_.c_str());

    return true;
}

void PipelineRegistry::shutdown() {
    if (!device_ && pipelines_.empty() && compute_pipelines_.empty() && !shader_manager_) {
        return; // Already shut down
    }
    // GPU resource destruction is main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    pipelines_.clear();
    compute_pipelines_.clear();
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
    auto it = pipelines_.find(type);
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
    pipelines_[type] = std::move(pipeline);
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

    auto* vs = shader_manager_->get(shader_path_ + desc.vertex_shader, ShaderStage::Vertex, "VSMain", desc.vs);
    auto* fs = shader_manager_->get(shader_path_ + desc.fragment_shader, ShaderStage::Fragment, "PSMain", desc.fs);
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
    auto it = compute_pipelines_.find(type);
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
    compute_pipelines_[type] = std::move(pipeline);
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
    return GPUComputePipeline::load_spirv(*device_, shader_path_ + desc.shader, desc.entry_point, desc.resources);
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

    BRACKEN_LOG_INFO(LOG_TAG, "PipelineRegistry: Preloaded {} pipelines",
                     pipelines_.size() + compute_pipelines_.size());

    return success;
}

void PipelineRegistry::invalidate_all() {
    // Cache mutation + (deferred) GPU pipeline release: main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
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
        invalidated += pipelines_.erase(type);
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
        preloaded_ = true;
    }
}

void PipelineRegistry::enable_hot_reload(core::asset::FileWatcher& watcher) {
    // FileWatcher entry points are main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    BRACKEN_ASSERT_NOT_SIM_STAGE();
    if (!device_) {
        BRACKEN_LOG_WARN("hot_reload", "PipelineRegistry::enable_hot_reload called before init");
        return;
    }
    if (watcher_ != nullptr) {
        return;
    }
    uint32_t handle = watcher.watch_directory(shader_path_, ".spv", [this](const std::filesystem::path& p) {
        BRACKEN_LOG_INFO("hot_reload", "Shader changed: {} -- rebuilding pipelines", p.filename().string());
        reload_all_pipelines();
    });
    if (handle == core::asset::FileWatcher::k_invalid_handle) {
        // Leave watcher_ null so a later call (e.g. after the host initializes
        // the FileWatcher) can retry instead of being blocked by the latch.
        BRACKEN_LOG_WARN("hot_reload",
                         "PipelineRegistry: watch_directory failed for {} (FileWatcher not initialized?); "
                         "shader hot-reload NOT enabled",
                         shader_path_);
        return;
    }
    watcher_ = &watcher;
    shader_watch_handle_ = handle;
    BRACKEN_LOG_INFO("hot_reload", "PipelineRegistry hot-reload enabled at {}", shader_path_);
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

} // namespace bracken::gpu
