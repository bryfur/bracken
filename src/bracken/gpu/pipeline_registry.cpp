#include "pipeline_registry.hpp"
#include "bracken/core/assert.hpp"
#include "bracken/core/asset/file_watcher.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/gpu_shader.hpp"
#include "SDL3/SDL_gpu.h"
#include <memory>
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_log.h>
#include <string>
#include <utility>

namespace bracken::gpu {

// =============================================================================
// PipelineRegistry Implementation
// =============================================================================

PipelineRegistry::~PipelineRegistry() {
    shutdown();
}

bool PipelineRegistry::init(GPUDevice& device) {
    // GPU resource creation/destruction is main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    if (device_) {
        SDL_Log("PipelineRegistry: Already initialized");
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
        SDL_Log("PipelineRegistry: SDL_GetBasePath failed, using relative path");
    }

    // Create shader manager (shaders are pre-compiled to SPIRV at build time)
    shader_manager_ = std::make_unique<ShaderManager>(device);

    SDL_Log("PipelineRegistry: Initialized with shader path: %s", shader_path_.c_str());

    return true;
}

void PipelineRegistry::shutdown() {
    if (!device_ && pipelines_.empty() && compute_pipelines_.empty() && !shader_manager_) {
        return; // Already shut down
    }
    // GPU resource destruction is main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    pipelines_.clear();
    compute_pipelines_.clear();
    shader_manager_.reset();
    device_ = nullptr;
    preloaded_ = false;
    SDL_Log("PipelineRegistry: Shutdown complete");
}

GPUPipeline* PipelineRegistry::get_pipeline(PipelineType type) {
    if (!device_) {
        SDL_Log("PipelineRegistry: Not initialized");
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
    if (preloaded_) {
        BRACKEN_LOG_WARN("gpu", "PipelineRegistry: lazy creation of {} pipeline after preload — "
                                "the static pipeline enumeration is missing this variant",
                         pipeline_type_to_string(type));
    }

    std::unique_ptr<GPUPipeline> pipeline = create_pipeline_uncached(type);

    if (!pipeline) {
        SDL_Log("PipelineRegistry: Failed to create %s pipeline", pipeline_type_to_string(type));
        return nullptr;
    }

    SDL_Log("PipelineRegistry: Created %s pipeline", pipeline_type_to_string(type));

    ++creation_count_;
    auto* result = pipeline.get();
    pipelines_[type] = std::move(pipeline);
    return result;
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_pipeline_uncached(PipelineType type) {
    std::unique_ptr<GPUPipeline> pipeline;

    switch (type) {
        case PipelineType::Model:
            pipeline = create_model_pipeline();
            break;
        case PipelineType::SkinnedModel:
            pipeline = create_skinned_model_pipeline();
            break;
        case PipelineType::Terrain:
            pipeline = create_terrain_pipeline();
            break;
        case PipelineType::Skybox:
            pipeline = create_skybox_pipeline();
            break;
        case PipelineType::Grid:
            pipeline = create_grid_pipeline();
            break;
        case PipelineType::UI:
            pipeline = create_ui_pipeline();
            break;
        case PipelineType::Text:
            pipeline = create_text_pipeline();
            break;
        case PipelineType::Particle:
            pipeline = create_particle_pipeline();
            break;
        case PipelineType::Grass:
            pipeline = create_grass_pipeline();
            break;
        case PipelineType::InstancedModel:
            pipeline = create_instanced_model_pipeline();
            break;
        case PipelineType::InstancedShadowModel:
            pipeline = create_instanced_shadow_model_pipeline();
            break;
        case PipelineType::ShadowModel:
            pipeline = create_shadow_model_pipeline();
            break;
        case PipelineType::ShadowSkinnedModel:
            pipeline = create_shadow_skinned_model_pipeline();
            break;
        case PipelineType::ShadowTerrain:
            pipeline = create_shadow_terrain_pipeline();
            break;
        case PipelineType::LocalShadowClear:
            pipeline = create_local_shadow_clear_pipeline();
            break;
        case PipelineType::GTAO:
            pipeline = create_gtao_pipeline();
            break;
        case PipelineType::GTAOTemporal:
            pipeline = create_gtao_temporal_pipeline();
            break;
        case PipelineType::GTAOUpsample:
            pipeline = create_gtao_upsample_pipeline();
            break;
        case PipelineType::BlurAO:
            pipeline = create_blur_ao_pipeline();
            break;
        case PipelineType::Composite:
            pipeline = create_composite_pipeline();
            break;
        case PipelineType::BloomDownsample:
            pipeline = create_bloom_downsample_pipeline();
            break;
        case PipelineType::BloomUpsample:
            pipeline = create_bloom_upsample_pipeline();
            break;
        case PipelineType::VolumetricFog:
            pipeline = create_volumetric_fog_pipeline();
            break;
        case PipelineType::Fxaa:
            pipeline = create_fxaa_pipeline();
            break;
        case PipelineType::TaaResolve:
            pipeline = create_taa_resolve_pipeline();
            break;
        case PipelineType::TaaSharpen:
            pipeline = create_taa_sharpen_pipeline();
            break;
        case PipelineType::SkyCapture:
            pipeline = create_sky_capture_pipeline();
            break;
        case PipelineType::IblSpecularPrefilter:
            pipeline = create_ibl_specular_prefilter_pipeline();
            break;
        case PipelineType::IblIrradiance:
            pipeline = create_ibl_irradiance_pipeline();
            break;
        case PipelineType::IblBrdfLut:
            pipeline = create_ibl_brdf_lut_pipeline();
            break;
        case PipelineType::ModelDoubleSided:
            pipeline = create_model_pipeline(false, true);
            break;
        case PipelineType::ModelBlend:
            pipeline = create_model_pipeline(true, false);
            break;
        case PipelineType::ModelBlendDoubleSided:
            pipeline = create_model_pipeline(true, true);
            break;
        case PipelineType::SkinnedModelDoubleSided:
            pipeline = create_skinned_model_pipeline(false, true);
            break;
        case PipelineType::SkinnedModelBlend:
            pipeline = create_skinned_model_pipeline(true, false);
            break;
        case PipelineType::SkinnedModelBlendDoubleSided:
            pipeline = create_skinned_model_pipeline(true, true);
            break;
        case PipelineType::InstancedModelDoubleSided:
            pipeline = create_instanced_model_pipeline(true);
            break;
        case PipelineType::DepthPrepassModel:
            pipeline = create_prepass_model_pipeline(false);
            break;
        case PipelineType::DepthPrepassModelDoubleSided:
            pipeline = create_prepass_model_pipeline(true);
            break;
        case PipelineType::DepthPrepassSkinnedModel:
            pipeline = create_prepass_skinned_model_pipeline(false);
            break;
        case PipelineType::DepthPrepassSkinnedModelDoubleSided:
            pipeline = create_prepass_skinned_model_pipeline(true);
            break;
        case PipelineType::DepthPrepassInstancedModel:
            pipeline = create_prepass_instanced_model_pipeline(false);
            break;
        case PipelineType::DepthPrepassInstancedModelDoubleSided:
            pipeline = create_prepass_instanced_model_pipeline(true);
            break;
        case PipelineType::DepthPrepassTerrain:
            pipeline = create_prepass_terrain_pipeline();
            break;
        case PipelineType::DepthPrepassGrass:
            pipeline = create_prepass_grass_pipeline();
            break;
        case PipelineType::ModelDepthEqual:
            pipeline = create_model_pipeline(false, false, true);
            break;
        case PipelineType::ModelDoubleSidedDepthEqual:
            pipeline = create_model_pipeline(false, true, true);
            break;
        case PipelineType::SkinnedModelDepthEqual:
            pipeline = create_skinned_model_pipeline(false, false, true);
            break;
        case PipelineType::SkinnedModelDoubleSidedDepthEqual:
            pipeline = create_skinned_model_pipeline(false, true, true);
            break;
        case PipelineType::InstancedModelDepthEqual:
            pipeline = create_instanced_model_pipeline(false, true);
            break;
        case PipelineType::InstancedModelDoubleSidedDepthEqual:
            pipeline = create_instanced_model_pipeline(true, true);
            break;
        case PipelineType::TerrainDepthEqual:
            pipeline = create_terrain_pipeline(true);
            break;
        case PipelineType::GrassDepthEqual:
            pipeline = create_grass_pipeline(true);
            break;
        case PipelineType::DepthPrepassModelMasked:
            pipeline = create_prepass_model_masked_pipeline();
            break;
        case PipelineType::DepthPrepassInstancedModelMasked:
            pipeline = create_prepass_instanced_model_masked_pipeline();
            break;
        case PipelineType::Water:
            pipeline = create_water_pipeline();
            break;
        default:
            SDL_Log("PipelineRegistry: Unknown pipeline type %d", static_cast<int>(type));
            return nullptr;
    }

    return pipeline;
}

GPUComputePipeline* PipelineRegistry::get_compute_pipeline(ComputePipelineType type) {
    if (!device_) {
        SDL_Log("PipelineRegistry: Not initialized");
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
    if (preloaded_) {
        BRACKEN_LOG_WARN("gpu", "PipelineRegistry: lazy creation of {} compute pipeline after preload — "
                                "the static pipeline enumeration is missing this variant",
                         compute_pipeline_type_to_string(type));
    }

    std::unique_ptr<GPUComputePipeline> pipeline = create_compute_pipeline_uncached(type);

    if (!pipeline) {
        SDL_Log("PipelineRegistry: Failed to create %s compute pipeline", compute_pipeline_type_to_string(type));
        return nullptr;
    }

    SDL_Log("PipelineRegistry: Created %s compute pipeline", compute_pipeline_type_to_string(type));

    ++creation_count_;
    auto* result = pipeline.get();
    compute_pipelines_[type] = std::move(pipeline);
    return result;
}

std::unique_ptr<GPUComputePipeline> PipelineRegistry::create_compute_pipeline_uncached(ComputePipelineType type) {
    switch (type) {
        case ComputePipelineType::ExposureHistogram:
            return create_exposure_histogram_pipeline();
        case ComputePipelineType::ExposureAdapt:
            return create_exposure_adapt_pipeline();
        case ComputePipelineType::WaterSim:
            return create_water_sim_pipeline();
        default:
            SDL_Log("PipelineRegistry: Unknown compute pipeline type %d", static_cast<int>(type));
            return nullptr;
    }
}

bool PipelineRegistry::preload_all_pipelines() {
    BRACKEN_ASSERT_MAIN_THREAD();
    if (!device_) {
        SDL_Log("PipelineRegistry: preload_all_pipelines called before init");
        return false;
    }
    SDL_Log("PipelineRegistry: Preloading all pipelines...");

    bool success = true;
    for (int i = 0; i < static_cast<int>(PipelineType::Count); ++i) {
        auto type = static_cast<PipelineType>(i);
        if (!get_pipeline(type)) {
            SDL_Log("PipelineRegistry: Failed to preload %s pipeline", pipeline_type_to_string(type));
            success = false;
        }
    }
    for (int i = 0; i < static_cast<int>(ComputePipelineType::Count); ++i) {
        auto type = static_cast<ComputePipelineType>(i);
        if (!get_compute_pipeline(type)) {
            SDL_Log("PipelineRegistry: Failed to preload %s compute pipeline", compute_pipeline_type_to_string(type));
            success = false;
        }
    }

    // The full enumeration has been walked; from here on, get_pipeline /
    // get_compute_pipeline are read-only lookups and any creation is logged
    // as an enumeration gap. Set even on partial failure — a variant that
    // failed to create here will fail identically on a lazy retry.
    preloaded_ = true;

    SDL_Log("PipelineRegistry: Preloaded %zu pipelines", pipelines_.size() + compute_pipelines_.size());
    return success;
}

void PipelineRegistry::invalidate_all() {
    // Cache mutation + (deferred) GPU pipeline release: main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
    pipelines_.clear();
    compute_pipelines_.clear();
    if (shader_manager_) {
        shader_manager_->clear_cache();
    }
    preloaded_ = false; // caller must preload_all_pipelines() again
    SDL_Log("PipelineRegistry: All pipelines invalidated");
}

void PipelineRegistry::set_swapchain_format(SDL_GPUTextureFormat format) {
    BRACKEN_ASSERT_MAIN_THREAD();
    if (format == swapchain_format_) {
        return;
    }
    swapchain_format_ = format;

    // Only pipelines that target the swapchain format depend on it; everything
    // else renders to fixed-format offscreen targets (HDR scene, R8 AO, depth)
    // and would be rebuilt for nothing.
    static constexpr PipelineType swapchain_dependent[] = {
        PipelineType::UI, PipelineType::Text, PipelineType::Composite, PipelineType::Fxaa,
    };
    size_t invalidated = 0;
    for (PipelineType type : swapchain_dependent) {
        invalidated += pipelines_.erase(type);
    }
    SDL_Log("PipelineRegistry: Swapchain format changed, invalidated %zu swapchain-dependent pipelines", invalidated);

    // Keep the pre-creation contract: if the registry was already preloaded,
    // re-create the affected pipelines now instead of leaving lazy creation
    // for the next hot-path lookup. Drop preloaded_ for the duration so this
    // intentional eager re-creation is not logged as an enumeration gap.
    if (preloaded_ && invalidated > 0) {
        preloaded_ = false;
        for (PipelineType type : swapchain_dependent) {
            if (!create_and_cache_pipeline(type)) {
                SDL_Log("PipelineRegistry: Failed to re-create %s pipeline after swapchain format change",
                        pipeline_type_to_string(type));
            }
        }
        preloaded_ = true;
    }
}

void PipelineRegistry::enable_hot_reload(core::asset::FileWatcher& watcher) {
    // FileWatcher entry points are main-thread tier.
    BRACKEN_ASSERT_MAIN_THREAD();
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
    if (!device_) {
        return;
    }
    invalidate_all();
    if (!preload_all_pipelines()) {
        BRACKEN_LOG_WARN("hot_reload", "PipelineRegistry: one or more pipelines failed to rebuild");
    }
}

// =============================================================================
// Pipeline Creation Methods
// =============================================================================

std::unique_ptr<GPUPipeline> PipelineRegistry::create_model_pipeline(bool alpha_blend, bool double_sided,
                                                                     bool depth_equal) {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 3; // lighting + shadow data + cluster params
    fs_resources.num_samplers = 11;       // baseColor + shadow atlas (cmp + raw) + normal + metallic-roughness + emissive + screen AO + IBL (irradiance/prefiltered/BRDF LUT) + local-shadow atlas
    fs_resources.num_storage_buffers = 4; // light data + cluster offsets + light indices + local-shadow slots

    auto* vs = shader_manager_->get(shader_path_ + "model.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "model.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.with_vertex3d();
    if (alpha_blend) {
        // glTF alphaMode BLEND: depth test on, depth write off; draws are
        // sorted back-to-front by SceneRenderer after all opaque content.
        config.alpha_blended();
    } else {
        config.opaque();
    }
    if (double_sided) {
        config.no_cull(); // glTF doubleSided
    }
    if (depth_equal) {
        // Post-prepass forward variant: depth was laid down by the depth
        // prepass, so test EQUAL and skip the redundant depth write.
        config.depth_compare_op = SDL_GPU_COMPAREOP_EQUAL;
        config.depth_write_enable = false;
    }
    config.color_format = scene_color_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_instanced_model_pipeline(bool double_sided, bool depth_equal) {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // camera uniforms
    vs_resources.num_storage_buffers = 1; // instance data

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 3; // lighting + shadow data + cluster params
    fs_resources.num_samplers = 11;       // baseColor + shadow atlas (cmp + raw) + normal + metallic-roughness + emissive + screen AO + IBL (irradiance/prefiltered/BRDF LUT) + local-shadow atlas
    fs_resources.num_storage_buffers = 4; // light data + cluster offsets + light indices + local-shadow slots

    auto* vs =
        shader_manager_->get(shader_path_ + "model_instanced.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "model_instanced.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.with_vertex3d().opaque();
    if (double_sided) {
        config.no_cull(); // glTF doubleSided
    }
    if (depth_equal) {
        config.depth_compare_op = SDL_GPU_COMPAREOP_EQUAL;
        config.depth_write_enable = false;
    }
    config.color_format = scene_color_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_instanced_shadow_model_pipeline() {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // lightViewProjection
    vs_resources.num_storage_buffers = 1; // instance model matrices

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // hasTexture flag
    fs_resources.num_samplers = 1;        // alpha texture for transparency

    auto* vs = shader_manager_->get(shader_path_ + "shadow_depth_instanced.vert.spv", ShaderStage::Vertex, "VSMain",
                                    vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "shadow_depth.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    // Receiver-side slope/normal-offset biasing carries most of the acne
    // fight; rasterizer bias stays small to avoid peter-panning. Depth clamp
    // (clip off) pancakes casters in front of the cascade near plane.
    config.with_vertex3d().opaque().cull_front().with_depth_bias(1.25f, 1.75f);
    config.enable_depth_clip = false;
    // Shadow cascades stay standard-Z (ortho projections have no reversed-Z
    // precision problem); the atlas clear, compare sampler and receiver math
    // all assume near=0/far=1.
    config.depth_compare_op = SDL_GPU_COMPAREOP_LESS_OR_EQUAL;
    config.color_format = SDL_GPU_TEXTUREFORMAT_INVALID; // depth-only
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_skinned_model_pipeline(bool alpha_blend, bool double_sided,
                                                                             bool depth_equal) {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // transform (incl. palette offset)
    vs_resources.num_storage_buffers = 1; // tightly packed bone palettes

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 3; // lighting + shadow data + cluster params
    fs_resources.num_samplers = 11;       // baseColor + shadow atlas (cmp + raw) + normal + metallic-roughness + emissive + screen AO + IBL (irradiance/prefiltered/BRDF LUT) + local-shadow atlas
    fs_resources.num_storage_buffers = 4; // light data + cluster offsets + light indices + local-shadow slots

    auto* vs =
        shader_manager_->get(shader_path_ + "skinned_model.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "model.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.with_skinned_vertex();
    if (alpha_blend) {
        // glTF alphaMode BLEND: depth test on, depth write off; draws are
        // sorted back-to-front by SceneRenderer after all opaque content.
        config.alpha_blended();
    } else {
        config.opaque();
    }
    if (double_sided) {
        config.no_cull(); // glTF doubleSided
    }
    if (depth_equal) {
        config.depth_compare_op = SDL_GPU_COMPAREOP_EQUAL;
        config.depth_write_enable = false;
    }
    config.color_format = scene_color_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_terrain_pipeline(bool depth_equal) {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 3; // lighting + shadow data + cluster params
    fs_resources.num_samplers = 9;        // material array + splatmap + shadow atlas (cmp + raw) + screen AO + IBL (irradiance/prefiltered/BRDF LUT) + local-shadow atlas
    fs_resources.num_storage_buffers = 4; // light data + cluster offsets + light indices + local-shadow slots

    auto* vs = shader_manager_->get(shader_path_ + "terrain.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "terrain.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    // Custom terrain vertex format: position(3), texcoord(2), color(4), normal(3)
    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();

    config.vertex_buffers = {{.slot = 0,
                              .pitch = sizeof(float) * 12, // 3 + 2 + 4 + 3 floats
                              .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
                              .instance_step_rate = 0}};

    config.vertex_attributes = {
        {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0},                 // position
        {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, sizeof(float) * 3}, // texcoord
        {2, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, sizeof(float) * 5}, // color
        {3, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, sizeof(float) * 9}, // normal
    };

    config.opaque();
    if (depth_equal) {
        config.depth_compare_op = SDL_GPU_COMPAREOP_EQUAL;
        config.depth_write_enable = false;
    }
    config.color_format = scene_color_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_skybox_pipeline() {
    ShaderResources vs_resources;
    // Skybox vertex shader has no uniforms (fullscreen triangle, no transforms)

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1;

    auto* vs = shader_manager_->get(shader_path_ + "skybox.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "skybox.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();

    // Skybox uses position-only vertices (float3), not full Vertex3D
    config.vertex_buffers = {
        {.slot = 0, .pitch = sizeof(float) * 3, .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX, .instance_step_rate = 0}};
    config.vertex_attributes = {
        {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0}, // position only
    };

    // Reversed-Z: the skybox vertex shader emits z = 0.0 (the far plane) and
    // GREATER_OR_EQUAL passes only where the depth buffer still holds the
    // 0.0 clear value, i.e. where no geometry was drawn.
    config.opaque().no_cull();
    config.depth_write_enable = false;
    config.depth_compare_op = SDL_GPU_COMPAREOP_GREATER_OR_EQUAL;
    config.color_format = scene_color_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_grid_pipeline() {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1;

    ShaderResources fs_resources;
    // No resources needed

    auto* vs = shader_manager_->get(shader_path_ + "grid.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "grid.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();

    // Custom grid vertex format: position(3) + color(4) = 28 bytes
    config.vertex_buffers = {
        {.slot = 0, .pitch = sizeof(float) * 7, .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX, .instance_step_rate = 0}};
    config.vertex_attributes = {
        {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0},                 // position
        {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, sizeof(float) * 3}, // color
    };

    config.alpha_blended().no_cull();
    config.color_format = scene_color_format_;
    config.depth_format = depth_format_;
    config.primitive_type = SDL_GPU_PRIMITIVETYPE_LINELIST;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_ui_pipeline() {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // screen size uniform

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // has_texture flag
    fs_resources.num_samplers = 1;        // texture sampler (even if not always used)

    SDL_Log("PipelineRegistry: Creating UI shaders with vs_uniforms=%d, fs_uniforms=%d, fs_samplers=%d",
            vs_resources.num_uniform_buffers, fs_resources.num_uniform_buffers, fs_resources.num_samplers);

    auto* vs = shader_manager_->get(shader_path_ + "ui.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "ui.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.with_vertex2d().alpha_blended().no_depth().no_cull();
    config.color_format = swapchain_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_text_pipeline() {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // projection

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // text color
    fs_resources.num_samplers = 1;        // font texture

    auto* vs = shader_manager_->get(shader_path_ + "text.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "text.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    // Text uses 4-float vertex format: position(2), texcoord(2)
    // Color is passed as a uniform, not per-vertex
    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();

    config.vertex_buffers = {{.slot = 0,
                              .pitch = sizeof(float) * 4, // 2 + 2 floats
                              .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
                              .instance_step_rate = 0}};

    config.vertex_attributes = {
        {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, 0},                 // position
        {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, sizeof(float) * 2}, // texcoord
    };

    config.alpha_blended().no_depth().no_cull();
    config.color_format = swapchain_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_particle_pipeline() {
    // Instanced: per-particle transforms + colors come from a per-frame
    // storage buffer (one draw per emitter/mesh batch); fragment is a cheap
    // unlit additive shader writing into the HDR scene target. Depth test
    // GREATER_OR_EQUAL (reversed-Z) with depth writes off so particles occlude
    // correctly against opaques but not against each other.
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 2; // camera (b0) + batch instance offset (b1)
    vs_resources.num_storage_buffers = 1; // per-particle instance data

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // ParticleFragmentUniforms
    fs_resources.num_samplers = 1;        // baseColor (bind 1x1 white when untextured)

    auto* vs = shader_manager_->get(shader_path_ + "particle.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "particle.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    // BlendMode::Additive is (SRC_ALPHA, ONE); particle.frag writes alpha 1.0
    // so the effective color blend is exactly (ONE, ONE). Back-face culling
    // stays on (PipelineConfig default).
    config.with_vertex3d().additive();
    config.depth_compare_op = SDL_GPU_COMPAREOP_GREATER_OR_EQUAL; // reversed-Z
    config.color_format = scene_color_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_grass_pipeline(bool depth_equal) {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1;
    vs_resources.num_samplers = 2;        // heightmap + splatmap textures
    vs_resources.num_storage_buffers = 1; // visible-chunk origins (chunked LOD)

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 3; // lighting + shadow data + cluster params
    fs_resources.num_samplers = 7;        // shadow atlas (cmp + raw) + screen AO + IBL (irradiance/prefiltered/BRDF LUT) + local-shadow atlas
    fs_resources.num_storage_buffers = 4; // light data + cluster offsets + light indices + local-shadow slots

    auto* vs = shader_manager_->get(shader_path_ + "grass.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "grass.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.with_vertex3d().opaque().no_cull();
    if (depth_equal) {
        config.depth_compare_op = SDL_GPU_COMPAREOP_EQUAL;
        config.depth_write_enable = false;
    }
    config.color_format = scene_color_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

// =============================================================================
// Depth-prepass pipelines
// =============================================================================
// All prepass pipelines share prepass_velocity.frag (zero fragment resources),
// render into the RG16F velocity target + the scene depth target, and keep the
// scene's reversed-Z GREATER_OR_EQUAL compare with depth writes ON. Vertex
// buffers are the same as the forward pass; only the position(-and-skinning)
// attributes are declared.

std::unique_ptr<GPUPipeline> PipelineRegistry::create_prepass_model_pipeline(bool double_sided) {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // PrepassModelUniforms

    ShaderResources fs_resources; // velocity output only, no resources

    auto* vs =
        shader_manager_->get(shader_path_ + "prepass_model.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "prepass_velocity.frag.spv", ShaderStage::Fragment, "PSMain",
                                    fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    // Forward Vertex3D stride, position attribute only.
    config.vertex_buffers = {get_vertex3d_buffer_desc()};
    config.vertex_attributes = {
        {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(Vertex3D, position)},
    };
    config.opaque();
    if (double_sided) {
        config.no_cull();
    }
    config.color_format = velocity_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_prepass_skinned_model_pipeline(bool double_sided) {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // PrepassModelUniforms
    vs_resources.num_storage_buffers = 2; // current + previous bone palettes

    ShaderResources fs_resources;

    auto* vs =
        shader_manager_->get(shader_path_ + "prepass_skinned.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "prepass_velocity.frag.spv", ShaderStage::Fragment, "PSMain",
                                    fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    // Forward SkinnedVertex stride; position + joints + weights only.
    config.vertex_buffers = {get_skinned_vertex_buffer_desc()};
    config.vertex_attributes = {
        {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(SkinnedVertex, position)},
        {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4, offsetof(SkinnedVertex, joints)},
        {2, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, offsetof(SkinnedVertex, weights)},
    };
    config.opaque();
    if (double_sided) {
        config.no_cull();
    }
    config.color_format = velocity_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_prepass_instanced_model_pipeline(bool double_sided) {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // PrepassCameraUniforms
    vs_resources.num_storage_buffers = 1; // shared InstanceData buffer

    ShaderResources fs_resources;

    auto* vs =
        shader_manager_->get(shader_path_ + "prepass_instanced.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "prepass_velocity.frag.spv", ShaderStage::Fragment, "PSMain",
                                    fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.vertex_buffers = {get_vertex3d_buffer_desc()};
    config.vertex_attributes = {
        {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(Vertex3D, position)},
    };
    config.opaque();
    if (double_sided) {
        config.no_cull();
    }
    config.color_format = velocity_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_prepass_model_masked_pipeline() {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // PrepassModelUniforms

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // PrepassMaskedUniforms (alpha cutoff)
    fs_resources.num_samplers = 1;        // base color (alpha test)

    auto* vs = shader_manager_->get(shader_path_ + "prepass_model_masked.vert.spv", ShaderStage::Vertex, "VSMain",
                                    vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "prepass_velocity_masked.frag.spv", ShaderStage::Fragment,
                                    "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.vertex_buffers = {get_vertex3d_buffer_desc()};
    config.vertex_attributes = {
        {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(Vertex3D, position)},
        {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, offsetof(Vertex3D, texcoord)},
    };
    config.opaque().no_cull(); // MASK foliage is double-sided in practice
    config.color_format = velocity_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_prepass_instanced_model_masked_pipeline() {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // PrepassCameraUniforms
    vs_resources.num_storage_buffers = 1; // shared InstanceData buffer

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // PrepassMaskedUniforms (alpha cutoff)
    fs_resources.num_samplers = 1;        // base color (alpha test)

    auto* vs = shader_manager_->get(shader_path_ + "prepass_instanced_masked.vert.spv", ShaderStage::Vertex,
                                    "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "prepass_velocity_masked.frag.spv", ShaderStage::Fragment,
                                    "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.vertex_buffers = {get_vertex3d_buffer_desc()};
    config.vertex_attributes = {
        {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(Vertex3D, position)},
        {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, offsetof(Vertex3D, texcoord)},
    };
    config.opaque().no_cull();
    config.color_format = velocity_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_prepass_terrain_pipeline() {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // PrepassCameraUniforms

    ShaderResources fs_resources;

    auto* vs =
        shader_manager_->get(shader_path_ + "prepass_terrain.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "prepass_velocity.frag.spv", ShaderStage::Fragment, "PSMain",
                                    fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    // Forward terrain stride (12 floats), position attribute only.
    config.vertex_buffers = {{.slot = 0,
                              .pitch = sizeof(float) * 12,
                              .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
                              .instance_step_rate = 0}};
    config.vertex_attributes = {
        {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0},
    };
    config.opaque();
    config.color_format = velocity_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_prepass_grass_pipeline() {
    // Same vertex-stage resources as the forward grass pipeline: the prepass
    // shader runs the identical blade placement math (grass_blade.hlsli).
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // GrassUniforms (identical layout)
    vs_resources.num_samplers = 2;        // heightmap + splatmap
    vs_resources.num_storage_buffers = 1; // visible-chunk origins

    ShaderResources fs_resources;

    auto* vs =
        shader_manager_->get(shader_path_ + "prepass_grass.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "prepass_velocity.frag.spv", ShaderStage::Fragment, "PSMain",
                                    fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    // Forward Vertex3D stride, position attribute only; no_cull matches the
    // forward grass pipeline so the EQUAL depth coverage is identical.
    config.vertex_buffers = {get_vertex3d_buffer_desc()};
    config.vertex_attributes = {
        {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(Vertex3D, position)},
    };
    config.opaque().no_cull();
    config.color_format = velocity_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_shadow_model_pipeline() {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // lightVP + model

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // hasTexture flag
    fs_resources.num_samplers = 1;        // alpha texture for transparency

    auto* vs =
        shader_manager_->get(shader_path_ + "shadow_depth.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "shadow_depth.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.with_vertex3d().opaque().cull_front().with_depth_bias(1.25f, 1.75f);
    config.enable_depth_clip = false; // pancake casters in front of the near plane
    config.depth_compare_op = SDL_GPU_COMPAREOP_LESS_OR_EQUAL; // shadow cascades stay standard-Z
    config.color_format = SDL_GPU_TEXTUREFORMAT_INVALID; // depth-only
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_shadow_skinned_model_pipeline() {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // lightVP + model + palette offset
    vs_resources.num_storage_buffers = 1; // tightly packed bone palettes

    ShaderResources fs_resources; // depth-only, no fragment resources

    auto* vs =
        shader_manager_->get(shader_path_ + "shadow_skinned.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "shadow_depth_opaque.frag.spv", ShaderStage::Fragment, "PSMain",
                                    fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.with_skinned_vertex().opaque().cull_front().with_depth_bias(1.25f, 1.75f);
    config.enable_depth_clip = false; // pancake casters in front of the near plane
    config.depth_compare_op = SDL_GPU_COMPAREOP_LESS_OR_EQUAL; // shadow cascades stay standard-Z
    config.color_format = SDL_GPU_TEXTUREFORMAT_INVALID;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_shadow_terrain_pipeline() {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // lightVP

    ShaderResources fs_resources; // depth-only, no fragment resources

    auto* vs =
        shader_manager_->get(shader_path_ + "shadow_terrain.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "shadow_depth_opaque.frag.spv", ShaderStage::Fragment, "PSMain",
                                    fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    // Terrain vertex format: position(3), texcoord(2), color(4), normal(3)
    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();

    config.vertex_buffers = {{.slot = 0,
                              .pitch = sizeof(float) * 12,
                              .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
                              .instance_step_rate = 0}};

    config.vertex_attributes = {
        {0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0},
        {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, sizeof(float) * 3},
        {2, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, sizeof(float) * 5},
        {3, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, sizeof(float) * 9},
    };

    // Terrain is single-sided (upward-facing only), so no_cull is needed
    // to ensure terrain triangles are written to the shadow map.
    config.opaque().no_cull().with_depth_bias(1.25f, 1.75f);
    config.enable_depth_clip = false; // pancake casters in front of the near plane
    config.depth_compare_op = SDL_GPU_COMPAREOP_LESS_OR_EQUAL; // shadow cascades stay standard-Z
    config.color_format = SDL_GPU_TEXTUREFORMAT_INVALID;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_local_shadow_clear_pipeline() {
    // Per-tile depth clear for the local-shadow atlas: cached tiles persist
    // across frames (the atlas pass LOADs), so a dirty tile is "cleared" by
    // drawing the fullscreen triangle over its viewport with the depth range
    // pinned to [1,1] (far plane) and an ALWAYS depth test. No new shaders:
    // fullscreen.vert + the empty opaque shadow fragment.
    ShaderResources vs_resources; // fullscreen triangle from SV_VertexID
    ShaderResources fs_resources; // empty fragment, no outputs

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "shadow_depth_opaque.frag.spv", ShaderStage::Fragment, "PSMain",
                                    fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_cull();
    config.depth_compare_op = SDL_GPU_COMPAREOP_ALWAYS;
    config.color_format = SDL_GPU_TEXTUREFORMAT_INVALID; // depth-only
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_gtao_pipeline() {
    ShaderResources vs_resources;
    // Fullscreen vertex shader: no uniforms, no samplers

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // GTAOUniforms
    fs_resources.num_samplers = 1;        // depth texture

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "gtao.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    // No vertex input (fullscreen triangle from SV_VertexID)
    config.opaque().no_depth().no_cull();
    config.color_format = SDL_GPU_TEXTUREFORMAT_R8_UNORM;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_gtao_temporal_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // GtaoTemporalUniforms
    fs_resources.num_samplers = 4;        // raw AO + history + velocity + depth

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "gtao_temporal.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();
    // RG16F history: R = accumulated AO, G = linear view depth (disocclusion reject).
    config.color_format = SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_gtao_upsample_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // GtaoUpsampleUniforms
    fs_resources.num_samplers = 2;        // blurred AO + depth texture

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "gtao_upsample.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();
    config.color_format = SDL_GPU_TEXTUREFORMAT_R8_UNORM;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_blur_ao_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // BlurUniforms
    fs_resources.num_samplers = 2;        // AO texture + depth texture

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "blur_ao.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();
    config.color_format = SDL_GPU_TEXTUREFORMAT_R8_UNORM;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_composite_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // CompositeUniforms
    fs_resources.num_samplers = 4;        // color + bloom + volumetric fog + depth
    fs_resources.num_storage_buffers = 1; // auto-exposure state (AutoExposure or dummy)

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "composite.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();
    config.color_format = swapchain_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_bloom_downsample_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // BloomDownsampleUniforms
    fs_resources.num_samplers = 1;        // source texture

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "bloom_downsample.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();
    config.color_format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_bloom_upsample_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // BloomUpsampleUniforms
    fs_resources.num_samplers = 1;        // source texture

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "bloom_upsample.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    // Progressive upsample: each pass tent-filters the smaller mip and ADDS it
    // onto the larger mip (LOADOP_LOAD on the target). The shader writes alpha
    // 1.0 so BlendMode::Additive (SRC_ALPHA, ONE) is effectively (ONE, ONE).
    config.additive().no_depth().no_cull();
    config.color_format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_volumetric_fog_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // VolumetricFogUniforms
    fs_resources.num_samplers = 2;        // depth texture + shadow atlas

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "volumetric_fog.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();
    config.color_format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_fxaa_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // FxaaUniforms
    fs_resources.num_samplers = 1;        // LDR composite output

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "fxaa.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();
    config.color_format = swapchain_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_taa_resolve_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // TaaResolveUniforms
    fs_resources.num_samplers = 4;        // scene color + history + velocity + depth

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "taa_resolve.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();
    // Writes the RGBA16F history buffer (same format as the HDR scene target).
    config.color_format = scene_color_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_taa_sharpen_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // TaaSharpenUniforms
    fs_resources.num_samplers = 1;        // resolved history

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "taa_sharpen.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();
    // Writes back over the HDR offscreen scene target for bloom/composite.
    config.color_format = scene_color_format_;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_sky_capture_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // SkyCaptureUniforms

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "sky_capture.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();
    // Targets one face of the RGBA16F sky-capture cube (linear HDR radiance).
    config.color_format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_ibl_specular_prefilter_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // IblFilterUniforms
    fs_resources.num_samplers = 1;        // sky-capture cube

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "ibl_prefilter.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();
    // Targets one (face, mip) of the RGBA16F prefiltered specular cube.
    config.color_format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_ibl_irradiance_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // IblFilterUniforms
    fs_resources.num_samplers = 1;        // sky-capture cube

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "ibl_irradiance.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();
    // Targets one face of the RGBA16F irradiance cube.
    config.color_format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_ibl_brdf_lut_pipeline() {
    ShaderResources vs_resources;

    ShaderResources fs_resources; // fully procedural: no uniforms, no samplers

    auto* vs = shader_manager_->get(shader_path_ + "fullscreen.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs =
        shader_manager_->get(shader_path_ + "ibl_brdf_lut.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);

    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();
    config.opaque().no_depth().no_cull();
    // Split-sum (A, B) pair per texel.
    config.color_format = SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT;

    return GPUPipeline::create(*device_, config);
}

std::unique_ptr<GPUPipeline> PipelineRegistry::create_water_pipeline() {
    ShaderResources vs_resources;
    vs_resources.num_uniform_buffers = 1; // WaterRenderer::VertexUniforms
    vs_resources.num_samplers = 1;        // sim height state (vertex displacement)

    ShaderResources fs_resources;
    fs_resources.num_uniform_buffers = 1; // WaterRenderer::FragmentUniforms
    fs_resources.num_samplers = 4;        // scene color copy, scene depth copy, sim state, IBL specular

    auto* vs = shader_manager_->get(shader_path_ + "water.vert.spv", ShaderStage::Vertex, "VSMain", vs_resources);
    auto* fs = shader_manager_->get(shader_path_ + "water.frag.spv", ShaderStage::Fragment, "PSMain", fs_resources);
    if (!vs || !fs) {
        return nullptr;
    }

    PipelineConfig config;
    config.vertex_shader = vs->handle();
    config.fragment_shader = fs->handle();

    // Minimal grid vertex: float2 world-space XZ (the VS lifts it to the
    // water level and displaces by the simulation height field).
    SDL_GPUVertexBufferDescription vb = {};
    vb.slot = 0;
    vb.pitch = sizeof(float) * 2;
    vb.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
    config.vertex_buffers = {vb};
    SDL_GPUVertexAttribute attr = {};
    attr.location = 0;
    attr.buffer_slot = 0;
    attr.format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2;
    attr.offset = 0;
    config.vertex_attributes = {attr};

    // Refraction/reflection blending is composited in-shader from the scene
    // copies, so the pipeline itself is opaque; depth test + write stay on
    // (reversed-Z default) so fog, TAA reprojection and later transparents
    // see the water surface in the depth buffer.
    config.opaque().no_cull();
    config.color_format = scene_color_format_;
    config.depth_format = depth_format_;

    return GPUPipeline::create(*device_, config);
}

// =============================================================================
// Compute pipelines
// =============================================================================

std::unique_ptr<GPUComputePipeline> PipelineRegistry::create_exposure_histogram_pipeline() {
    ComputeResources resources;
    resources.num_samplers = 1;                  // HDR scene color
    resources.num_readwrite_storage_buffers = 1; // histogram bins
    resources.num_uniform_buffers = 1;           // ExposureHistogramUniforms

    return GPUComputePipeline::load_spirv(*device_, shader_path_ + "exposure_histogram.comp.spv", "CSMain", resources);
}

std::unique_ptr<GPUComputePipeline> PipelineRegistry::create_exposure_adapt_pipeline() {
    ComputeResources resources;
    resources.num_readwrite_storage_buffers = 2; // histogram bins (read + reset) + exposure state
    resources.num_uniform_buffers = 1;           // ExposureAdaptUniforms

    return GPUComputePipeline::load_spirv(*device_, shader_path_ + "exposure_adapt.comp.spv", "CSMain", resources);
}

std::unique_ptr<GPUComputePipeline> PipelineRegistry::create_water_sim_pipeline() {
    ComputeResources resources;
    resources.num_readonly_storage_textures = 1;  // previous wave state
    resources.num_readwrite_storage_textures = 1; // next wave state
    resources.num_uniform_buffers = 1;            // WaterRenderer::SimUniforms

    return GPUComputePipeline::load_spirv(*device_, shader_path_ + "water_sim.comp.spv", "CSMain", resources);
}

} // namespace bracken::gpu
