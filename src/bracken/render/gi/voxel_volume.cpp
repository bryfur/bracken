#include "bracken/render/gi/voxel_volume.hpp"

#include "bracken/core/logger.hpp"
#include "bracken/gpu/gpu_device.hpp"

namespace bracken::render::gi {
namespace {
constexpr const char* LOG_TAG = "gi";

/// SIMULTANEOUS_READ_WRITE, and the SDL header is explicit that this is "NOT
/// equivalent to READ | WRITE". The voxelizer composites with InterlockedMax,
/// which is a read-modify-write of the bound storage image, and the plain
/// WRITE bit does not cover that. Getting it wrong is not diagnosed: the clear
/// pass (write only) runs fine and the voxelize dispatch faults inside the
/// driver, which looks like a bad dispatch size rather than a bad usage flag.
///
/// No SAMPLER bit: nothing ever filters the albedo volume, so asking for one
/// would only constrain the format choice on some backends.
constexpr SDL_GPUTextureUsageFlags ALBEDO_USAGE =
    SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE |
    SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;

/// Injection writes the base level, the mip pass writes the rest, and the cones
/// SAMPLE it — the one volume that genuinely needs filtering, because the whole
/// technique is "one filtered tap stands in for a cone's cross-section".
constexpr SDL_GPUTextureUsageFlags RADIANCE_USAGE =
    SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE |
    SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_SAMPLER;
} // namespace

VoxelVolume::~VoxelVolume() {
    shutdown();
}

bool VoxelVolume::init(gpu::GPUDevice& device) {
    if (ready_) {
        return true; // idempotent — a host toggling GI modes must not reallocate
    }
    device_ = &device;

    // Ask rather than assume: the SDL header carries a literal
    // "FIXME: Check universal support for SIMULTANEOUS_READ_WRITE", so 3D +
    // R32_UINT + atomics is exactly the combination worth querying before a
    // dispatch faults inside the driver with no diagnostic.
    {
        SDL_GPUDevice* raw = device.handle();
        const bool albedo_ok = SDL_GPUTextureSupportsFormat(
            raw, SDL_GPU_TEXTUREFORMAT_R32_UINT, SDL_GPU_TEXTURETYPE_3D, ALBEDO_USAGE);
        const bool radiance_ok = SDL_GPUTextureSupportsFormat(
            raw, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, SDL_GPU_TEXTURETYPE_3D, RADIANCE_USAGE);
        BRACKEN_LOG_INFO(LOG_TAG, "voxel format support: albedo(R32_UINT 3D rw)={} radiance(RGBA16F 3D rw+sampler)={}",
                         albedo_ok, radiance_ok);
        if (!albedo_ok || !radiance_ok) {
            BRACKEN_LOG_ERROR(LOG_TAG,
                              "VoxelVolume: the driver does not support the 3D storage formats "
                              "voxel GI needs; gi_mode=VoxelProbes cannot run here");
            return false;
        }
    }

    for (int c = 0; c < VOXEL_CASCADE_COUNT; ++c) {
        Cascade& cascade = cascades_[static_cast<std::size_t>(c)];
        cascade.albedo = gpu::GPUTexture::create_3d(
            device, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION,
            // R32_UINT, not RGBA8: the voxelizer's only order-independent
            // write is an INTEGER ATOMIC (InterlockedMax over a packed RGBA8
            // with opacity in the high byte), and an atomic needs an integer
            // view. Same four bytes per texel either way, so the memory table is
            // unchanged — but declaring the shader's image format as r32ui over
            // an RGBA8_UNORM texture is a mismatch the driver does not report,
            // it simply faults inside the dispatch.
            SDL_GPU_TEXTUREFORMAT_R32_UINT, ALBEDO_USAGE, /*mipmaps=*/false);
        cascade.radiance = gpu::GPUTexture::create_3d(
            device, VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION,
            SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, RADIANCE_USAGE, /*mipmaps=*/true);
        if (!cascade.albedo || !cascade.radiance) {
            BRACKEN_LOG_ERROR(LOG_TAG, "VoxelVolume: could not allocate cascade {} volumes", c);
            shutdown();
            return false;
        }

        VoxelCascadeStats& cs = stats_.per_cascade[static_cast<std::size_t>(c)];
        cs.albedo_bytes = voxel_cascade_bytes(4, false);
        cs.radiance_bytes = voxel_cascade_bytes(8, true);
        stats_.total_bytes += cs.albedo_bytes + cs.radiance_bytes;
    }

    // REPEATING, not clamped, and that is a correctness requirement rather than
    // a preference: storage is a torus, so the wrap a repeat sampler performs is
    // exactly the wrap the addressing wants. A clamped sampler would smear the
    // window's edge voxels outward instead of wrapping, and every cone leaving
    // the window would read a streak rather than a neighbour.
    //
    // LINEAR mipmap mode, unlike the probe atlases' NEAREST: a cone's mip is a
    // continuous function of its width, so stepping between levels would band
    // every widening cone. This is the pass that needs the interpolation.
    SDL_GPUSamplerCreateInfo sampler_info = {};
    sampler_info.min_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mag_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
    sampler_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sampler_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sampler_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
    sampler_ = device.create_sampler(sampler_info);
    if (sampler_ == nullptr) {
        BRACKEN_LOG_ERROR(LOG_TAG, "VoxelVolume: could not create the volume sampler");
        shutdown();
        return false;
    }

    ready_ = true;
    BRACKEN_LOG_INFO(LOG_TAG,
                     "voxel clipmap: {} cascades x {}^3, voxel {:.0f}..{:.0f} units, "
                     "extent {:.0f}..{:.0f} units, {:.1f} MB",
                     VOXEL_CASCADE_COUNT, VOXEL_RESOLUTION, static_cast<double>(voxel_size(0)),
                     static_cast<double>(voxel_size(VOXEL_CASCADE_COUNT - 1)),
                     static_cast<double>(voxel_cascade_extent(0)),
                     static_cast<double>(voxel_cascade_extent(VOXEL_CASCADE_COUNT - 1)),
                     static_cast<double>(stats_.total_bytes) / (1024.0 * 1024.0));
    return true;
}

void VoxelVolume::shutdown() {
    for (Cascade& cascade : cascades_) {
        cascade.albedo.reset();
        cascade.radiance.reset();
        cascade.has_base = false;
        cascade.base = glm::ivec3(0);
    }
    if (sampler_ != nullptr && device_ != nullptr) {
        device_->release_sampler(sampler_);
    }
    sampler_ = nullptr;
    ready_ = false;
    stats_ = VoxelVolumeStats{};
    device_ = nullptr;
}

bool VoxelVolume::scroll(const glm::vec3& camera_position) {
    if (!ready_) {
        return false;
    }
    bool any_moved = false;
    for (int c = 0; c < VOXEL_CASCADE_COUNT; ++c) {
        Cascade& cascade = cascades_[static_cast<std::size_t>(c)];
        const glm::ivec3 next = voxel_cascade_base(c, camera_position);

        VoxelCascadeStats& cs = stats_.per_cascade[static_cast<std::size_t>(c)];
        if (cascade.has_base && next == cascade.base) {
            cs.voxels_invalidated = 0;
            continue;
        }

        // How much of the window is stale. On the first scroll, or on a jump
        // larger than the window, that is everything; otherwise it is the shell
        // the camera uncovered, which is the volume difference between the
        // window and its intersection with where it used to be.
        std::uint32_t invalidated = 0;
        const auto window = static_cast<std::uint64_t>(VOXEL_RESOLUTION) * VOXEL_RESOLUTION *
                            VOXEL_RESOLUTION;
        if (!cascade.has_base) {
            invalidated = static_cast<std::uint32_t>(window);
        } else {
            const glm::ivec3 d = glm::abs(next - cascade.base);
            const glm::ivec3 overlap = glm::max(glm::ivec3(VOXEL_RESOLUTION) - d, glm::ivec3(0));
            const auto kept = static_cast<std::uint64_t>(overlap.x) *
                              static_cast<std::uint64_t>(overlap.y) *
                              static_cast<std::uint64_t>(overlap.z);
            invalidated = static_cast<std::uint32_t>(window - kept);
        }

        cascade.base = next;
        cascade.has_base = true;
        cs.base = next;
        cs.voxels_invalidated = invalidated;
        stats_.voxels_invalidated_total += invalidated;
        any_moved = true;
    }
    return any_moved;
}

glm::ivec3 VoxelVolume::base(int cascade) const {
    if (cascade < 0 || cascade >= VOXEL_CASCADE_COUNT) {
        return glm::ivec3(0);
    }
    return cascades_[static_cast<std::size_t>(cascade)].base;
}

gpu::GPUTexture* VoxelVolume::albedo(int cascade) const {
    if (cascade < 0 || cascade >= VOXEL_CASCADE_COUNT) {
        return nullptr;
    }
    return cascades_[static_cast<std::size_t>(cascade)].albedo.get();
}

gpu::GPUTexture* VoxelVolume::radiance(int cascade) const {
    if (cascade < 0 || cascade >= VOXEL_CASCADE_COUNT) {
        return nullptr;
    }
    return cascades_[static_cast<std::size_t>(cascade)].radiance.get();
}

} // namespace bracken::render::gi
