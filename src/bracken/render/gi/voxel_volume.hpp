#pragma once

/**
 * The VOXEL RADIANCE CLIPMAP — the raster GI path's world representation
 * (docs/rt_lighting_plan.md §6, "Raster GI").
 *
 * This is the alternative to tracing rays into a BVH. It owns two 3D textures
 * per cascade and the scroll state that keeps them centred on the camera; the
 * passes that fill and read them live elsewhere. `render/gi/voxel_math.hpp` is
 * the source of truth for every coordinate this class hands out, and is tested
 * with no device.
 *
 * ---------------------------------------------------------------------------
 * WHERE IT PLUGS IN
 * ---------------------------------------------------------------------------
 *
 * Nowhere near the fragment shaders. The ray-traced probe field
 * (`ProbeVolume`) meets its ray source at exactly one buffer —
 * `ray_results_`, a `float4(radiance, distance)` per (probe, ray) — and this
 * module exists to fill that same buffer by CONE TRACING instead. Everything
 * downstream is untouched and shared: the octahedral irradiance and visibility
 * atlases, the blend, the Chebyshev leak test, classification, relocation, the
 * cascade blend, the fog in-scatter, and every shader that samples the field.
 *
 * That is the whole reason this is worth building rather than a second GI
 * system: the expensive, load-bearing, already-debugged half is reused.
 *
 * ---------------------------------------------------------------------------
 * WHAT IS STORED, AND WHAT IS NOT
 * ---------------------------------------------------------------------------
 *
 *   albedo_[c]    R32_UINT, no mip chain. One packed word: RGB = surface albedo
 *                 in the low three bytes, HIGH BYTE = emissive intensity as a
 *                 fraction of VOXEL_EMISSIVE_SCALE. Coverage is implied by
 *                 occupancy rather than stored, which is what frees the byte.
 *                 Written by voxelization, read ONLY by injection — hence no
 *                 mips, which is a third of the memory saved.
 *   radiance_[c]  RGBA16F, WITH a mip chain. rgb = outgoing radiance
 *                 PREMULTIPLIED by a = coverage, which is 1 or 0 at mip 0 and
 *                 fractional above it because the chain is an unweighted box
 *                 average over children whose empty half is black. This is what
 *                 the cones sample, and the mip chain is not an optimization but
 *                 the technique: a cone takes ONE filtered tap at the mip whose
 *                 voxels match its width, standing in for the thousands of rays
 *                 it represents.
 *
 * There is deliberately no normal volume — see `VOXEL_RESOLUTION`.
 *
 * ---------------------------------------------------------------------------
 * SCROLLING
 * ---------------------------------------------------------------------------
 *
 * Torus-addressed, exactly like the probe field. The volume does not move in
 * memory when the camera does: the base corner advances in WHOLE voxels and the
 * addressing wraps, so a voxel still inside the window keeps its texel and only
 * the shell that scrolled in is stale. Snapping to whole voxels is what makes
 * that true — a fractional scroll would move every voxel's world position every
 * frame and invalidate the entire volume continuously.
 *
 * Tier: main-thread (owns GPU resources).
 */

#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/render/gi/voxel_math.hpp"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <memory>

namespace bracken::gpu {
class GPUDevice;
}

namespace bracken::render::gi {

/// Per-cascade scroll state and what it cost, for the dev overlay and the smoke
/// line. Mirrors `GiCascadeStats`'s role for the probe field.
struct VoxelCascadeStats {
    /// Base corner in whole voxels of this cascade — what the shaders index
    /// against, and what `voxel_in_window` tests.
    glm::ivec3 base{0};
    /// Voxels that scrolled in this frame and therefore hold stale contents
    /// until revoxelized. 0 in the steady state of a stationary camera.
    std::uint32_t voxels_invalidated = 0;
    std::uint64_t albedo_bytes = 0;
    std::uint64_t radiance_bytes = 0;
};

struct VoxelVolumeStats {
    std::array<VoxelCascadeStats, VOXEL_CASCADE_COUNT> per_cascade{};
    std::uint64_t total_bytes = 0;
    /// Cumulative, never reset — differenced by the bench harness.
    std::uint64_t voxels_invalidated_total = 0;
};

class VoxelVolume {
public:
    VoxelVolume() = default;
    ~VoxelVolume();
    VoxelVolume(const VoxelVolume&) = delete;
    VoxelVolume& operator=(const VoxelVolume&) = delete;
    VoxelVolume(VoxelVolume&&) = delete;
    VoxelVolume& operator=(VoxelVolume&&) = delete;

    /// Allocate the clipmap. Idempotent: a second call with the volumes already
    /// alive is a no-op, so a host toggling the GI mode does not reallocate.
    [[nodiscard]] bool init(gpu::GPUDevice& device);

    /// Release every volume and the sampler. Safe to call twice, and safe to
    /// call when `init` failed part-way.
    void shutdown();

    [[nodiscard]] bool ready() const { return ready_; }

    /**
     * Re-centre every cascade on `camera_position`, in whole voxels.
     *
     * Returns true when ANY cascade moved, which is the signal that
     * revoxelization has work to do. Device-free — this is pure scroll
     * bookkeeping, so it may run before the frame's command buffer exists.
     */
    bool scroll(const glm::vec3& camera_position);

    /// Base corner of `cascade`, in whole voxels of that cascade.
    [[nodiscard]] glm::ivec3 base(int cascade) const;

    [[nodiscard]] gpu::GPUTexture* albedo(int cascade) const;
    [[nodiscard]] gpu::GPUTexture* radiance(int cascade) const;

    /// The trilinear REPEATING sampler the cones read `radiance_` through.
    /// Repeating rather than clamped is not a detail: storage is a torus, so the
    /// wrap a repeat sampler performs is exactly the wrap the addressing wants,
    /// and a clamped sampler would smear the window's edge voxels outward.
    [[nodiscard]] SDL_GPUSampler* sampler() const { return sampler_; }

    [[nodiscard]] const VoxelVolumeStats& stats() const { return stats_; }

private:
    struct Cascade {
        std::unique_ptr<gpu::GPUTexture> albedo;
        std::unique_ptr<gpu::GPUTexture> radiance;
        glm::ivec3 base{0};
        bool has_base = false;
    };

    gpu::GPUDevice* device_ = nullptr;
    std::array<Cascade, VOXEL_CASCADE_COUNT> cascades_{};
    SDL_GPUSampler* sampler_ = nullptr;
    bool ready_ = false;
    VoxelVolumeStats stats_{};
};

} // namespace bracken::render::gi
