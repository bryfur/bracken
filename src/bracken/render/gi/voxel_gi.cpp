#include "bracken/render/gi/voxel_gi.hpp"

#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"

#include <array>
#include <cstring>

namespace bracken::render::gi {
namespace {
constexpr const char* LOG_TAG = "gi";

/// Threads per group in voxel_clear / voxel_inject, which walk the volume.
constexpr std::uint32_t VOLUME_GROUP = 4;
/// Threads per group in voxel_voxelize / voxel_cone_trace, which walk a list.
constexpr std::uint32_t LINEAR_GROUP = 64;

[[nodiscard]] constexpr std::uint32_t groups_for(std::uint64_t items, std::uint32_t group) {
    return static_cast<std::uint32_t>((items + group - 1) / group);
}

} // namespace

bool VoxelGi::init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipelines) {
    pipelines_ = &pipelines;
    if (!volume_.init(device)) {
        return false;
    }
    dirty_.fill(true); // nothing has been voxelized yet
    return true;
}

void VoxelGi::shutdown() {
    scene_.shutdown();
    volume_.shutdown();
    pipelines_ = nullptr;
    dirty_.fill(false);
    next_cascade_ = 0;
    stats_ = VoxelGiStats{};
}

void VoxelGi::build_models(const ModelManager& models) {
    scene_.build_models(models);
    // New geometry invalidates every cascade's contents, not just its window.
    dirty_.fill(true);
}

void VoxelGi::set_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances) {
    const std::uint32_t before = scene_.instance_count();
    scene_.set_instances(instances);
    if (scene_.instance_count() != before) {
        dirty_.fill(true);
    }
}

void VoxelGi::upload(gpu::GPUDevice& device) {
    scene_.upload(device);
}

bool VoxelGi::update(SDL_GPUCommandBuffer* cmd, const VoxelGiFrameParams& params) {
    BRACKEN_PROFILE_ZONE("VoxelGi::update");
    stats_.cascade_revoxelized = -1;
    stats_.injected = false;
    stats_.voxelize_work_items = scene_.work_count();

    if (!volume_.ready() || !scene_.ready() || cmd == nullptr || pipelines_ == nullptr) {
        return false;
    }

    // A cascade whose window moved holds contents for the wrong region; a
    // lighting change leaves the geometry valid but the radiance stale. Both
    // land in the same dirty set because both are repaired by the same
    // clear/voxelize/inject sequence — separating them would save the
    // voxelization on a lighting-only change and is the obvious RG4 refinement.
    if (volume_.scroll(params.camera_position)) {
        for (int c = 0; c < VOXEL_CASCADE_COUNT; ++c) {
            if (volume_.stats().per_cascade[static_cast<std::size_t>(c)].voxels_invalidated > 0) {
                dirty_[static_cast<std::size_t>(c)] = true;
            }
        }
    }
    if (params.lighting_changed) {
        dirty_.fill(true);
    }

    std::uint32_t dirty_count = 0;
    for (const bool d : dirty_) {
        dirty_count += d ? 1u : 0u;
    }
    stats_.cascades_dirty = dirty_count;
    if (dirty_count == 0) {
        return true;
    }

    // AT MOST ONE CASCADE PER FRAME, round-robin. The finest cascade scrolls
    // whenever the camera moves 25 units, so servicing every dirty cascade
    // every frame would make the cost of moving unbounded; the cursor is what
    // stops cascade 0 from starving the coarser ones it always beats to the
    // dirty flag.
    for (int i = 0; i < VOXEL_CASCADE_COUNT; ++i) {
        const int c = (next_cascade_ + i) % VOXEL_CASCADE_COUNT;
        if (!dirty_[static_cast<std::size_t>(c)]) {
            continue;
        }
        if (!rebuild_cascade(cmd, c, params)) {
            return false;
        }
        dirty_[static_cast<std::size_t>(c)] = false;
        stats_.cascade_revoxelized = c;
        next_cascade_ = (c + 1) % VOXEL_CASCADE_COUNT;
        break;
    }
    return true;
}

bool VoxelGi::rebuild_cascade(SDL_GPUCommandBuffer* cmd, int cascade,
                              const VoxelGiFrameParams& params) {
    BRACKEN_PROFILE_ZONE("VoxelGi::rebuild_cascade");

    const gpu::GPUComputePipeline* clear =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::VoxelClear);
    const gpu::GPUComputePipeline* voxelize =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::VoxelVoxelize);
    const gpu::GPUComputePipeline* inject =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::VoxelInject);
    if (clear == nullptr || voxelize == nullptr || inject == nullptr) {
        return false;
    }
    gpu::GPUTexture* albedo = volume_.albedo(cascade);
    gpu::GPUTexture* radiance = volume_.radiance(cascade);
    if (albedo == nullptr || radiance == nullptr) {
        return false;
    }
    const glm::ivec3 base = volume_.base(cascade);
    const std::uint32_t volume_groups = groups_for(VOXEL_RESOLUTION, VOLUME_GROUP);

    // ---- 1. clear ---------------------------------------------------------
    // Required, not hygiene: the voxelizer composites with InterlockedMax, which
    // is monotone, so without this a voxel could only ever get more opaque.
    {
        const SDL_GPUStorageTextureReadWriteBinding rw{.texture = albedo->handle(),
                                                       .mip_level = 0, .layer = 0, .cycle = false,
                                                       .padding1 = 0, .padding2 = 0, .padding3 = 0};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (!pass) {
            return false;
        }
        pass.bind_pipeline(*clear);
        pass.dispatch(volume_groups, volume_groups, volume_groups);
        pass.end();
    }

    // ---- 2. voxelize ------------------------------------------------------
    {
        gpu::VoxelVoxelizeUniforms u{};
        u.cascade = cascade;
        u.base = base;
        u.workCount = static_cast<std::uint32_t>(scene_.work_count());
        u.instanceCount = scene_.instance_count();

        const SDL_GPUStorageTextureReadWriteBinding rw{.texture = albedo->handle(),
                                                       .mip_level = 0, .layer = 0, .cycle = false,
                                                       .padding1 = 0, .padding2 = 0, .padding3 = 0};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (!pass) {
            return false;
        }
        pass.bind_pipeline(*voxelize);
        const std::array<SDL_GPUBuffer*, 2> buffers{scene_.triangle_buffer()->handle(),
                                                    scene_.instance_buffer()->handle()};
        pass.bind_storage_buffers(0, buffers);
        pass.push_uniforms(0, &u, sizeof(u));
        pass.dispatch(groups_for(scene_.work_count(), LINEAR_GROUP), 1, 1);
        pass.end();
    }

    // ---- 3. inject --------------------------------------------------------
    if (params.shadow_atlas == nullptr || params.shadow_block == nullptr) {
        // No CSM this frame: the geometry is voxelized but nothing lights it.
        // Reported rather than silently skipped — a clipmap of black voxels is
        // indistinguishable from a broken cone tracer.
        BRACKEN_LOG_WARN(LOG_TAG, "VoxelGi: cascade {} voxelized with no shadow atlas to inject",
                         cascade);
        return true;
    }
    {
        gpu::VoxelInjectUniforms u{};
        // The shadow block is assembled by the caller because it is the same
        // block every shadow consumer receives; copying it verbatim is what
        // guarantees this pass selects the same cascade a fragment would.
        const std::size_t copy = std::min(params.shadow_block_size, offsetof(gpu::VoxelInjectUniforms, cascade));
        std::memcpy(&u, params.shadow_block, copy);
        u.cascade = cascade;
        u.base = base;
        u.cameraPosition = params.camera_position;
        u.cameraForward = params.camera_forward;
        u.sunDirection = params.sun_direction;
        u.sunColor = params.sun_color;
        u.sunIntensity = params.sun_intensity;
        u.skyColor = params.sky_color;
        u.skyFloor = params.sky_floor;

        const std::array<SDL_GPUStorageTextureReadWriteBinding, 2> rw{{
            {.texture = albedo->handle(), .mip_level = 0, .layer = 0, .cycle = false,
             .padding1 = 0, .padding2 = 0, .padding3 = 0},
            {.texture = radiance->handle(), .mip_level = 0, .layer = 0, .cycle = false,
             .padding1 = 0, .padding2 = 0, .padding3 = 0},
        }};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, rw, {});
        if (!pass) {
            return false;
        }
        pass.bind_pipeline(*inject);
        const std::array<SDL_GPUTextureSamplerBinding, 2> textures{{
            {params.shadow_atlas, params.shadow_cmp_sampler},
            {params.shadow_atlas, params.shadow_raw_sampler},
        }};
        pass.bind_samplers(0, textures);
        pass.push_uniforms(0, &u, sizeof(u));
        pass.dispatch(volume_groups, volume_groups, volume_groups);
        pass.end();
        stats_.injected = true;
    }

    // ---- 4. mips ----------------------------------------------------------
    // The cone trace's entire economy is one filtered tap at the mip whose
    // voxels match the cone's width, so the chain is the technique rather than
    // an optimization. SDL3's generator is used rather than a hand-rolled
    // reduction because a box filter IS what a cone wants here — an
    // anisotropic chain (six directional mips, so a cone approaching a wall
    // edge-on does not see through it) is the known quality upgrade and is RG4.
    SDL_GenerateMipmapsForGPUTexture(cmd, radiance->handle());
    return true;
}

bool VoxelGi::cone_trace(SDL_GPUCommandBuffer* cmd, const VoxelConeTraceParams& params) {
    BRACKEN_PROFILE_ZONE("VoxelGi::cone_trace");
    stats_.cone_probes = 0;

    if (!volume_.ready() || cmd == nullptr || pipelines_ == nullptr) {
        return false;
    }
    if (params.probe_updates == nullptr || params.ray_results == nullptr ||
        params.probe_count == 0 || params.ray_count == 0) {
        return false;
    }
    const gpu::GPUComputePipeline* pipeline =
        pipelines_->get_compute_pipeline(gpu::ComputePipelineType::VoxelConeTrace);
    if (pipeline == nullptr) {
        return false;
    }

    gpu::VoxelConeTraceUniforms u{};
    u.probeCount = params.probe_count;
    u.rayCount = params.ray_count;
    u.frameIndex = params.frame_index;
    u.rayMaxDistance = params.ray_max_distance;
    u.cameraPosition = params.camera_position;
    u.bounceScale = params.bounce_scale;
    // A cone may march to the edge of the coarsest cascade and no further —
    // past that there is no data, only the sky term applied on exit.
    u.maxDistance = voxel_cascade_half_extent(VOXEL_CASCADE_COUNT - 1);
    for (int c = 0; c < VOXEL_CASCADE_COUNT; ++c) {
        const glm::ivec3 b = volume_.base(c);
        u.cascadeBases[c] = glm::ivec4(b, 0);
    }

    const SDL_GPUStorageBufferReadWriteBinding rw{.buffer = params.ray_results, .cycle = false,
                                                  .padding1 = 0, .padding2 = 0, .padding3 = 0};
    gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {}, {&rw, 1});
    if (!pass) {
        return false;
    }
    pass.bind_pipeline(*pipeline);
    const std::array<SDL_GPUTextureSamplerBinding, 3> volumes{{
        {volume_.radiance(0)->handle(), volume_.sampler()},
        {volume_.radiance(1)->handle(), volume_.sampler()},
        {volume_.radiance(2)->handle(), volume_.sampler()},
    }};
    pass.bind_samplers(0, volumes);
    SDL_GPUBuffer* updates = params.probe_updates;
    pass.bind_storage_buffers(0, {&updates, 1});
    pass.push_uniforms(0, &u, sizeof(u));
    // x = rays, y = probes. The same decomposition gi_probe_trace uses, so the
    // ray-results buffer is written in the same order by either path.
    pass.dispatch(groups_for(params.ray_count, LINEAR_GROUP), params.probe_count, 1);
    pass.end();

    stats_.cone_probes = params.probe_count;
    return true;
}

} // namespace bracken::render::gi
