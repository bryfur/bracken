#pragma once

#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/render/lighting/light.hpp"
#include "bracken/scene/camera_state.hpp"
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <SDL3/SDL_gpu.h>
#include <vector>

namespace bracken::core::jobs {
class JobSystem;
}

namespace bracken::render::lighting {

// Std140-compatible layout shared with clusters.hlsli.
// All fields packed contiguously; HLSL cbuffer layout requires 16-byte alignment.
struct alignas(16) ClusterParams {
    glm::mat4 view;
    glm::mat4 invProjection;
    glm::vec4 screenSize;     // (width, height, 1/width, 1/height)
    glm::vec4 zPlanes;        // (near, far, log(far/near), 1/log(far/near))
    glm::uvec4 gridDim;       // (X, Y, Z, totalLightCount)
    // .x = per-cluster light cap (shader clamps its loop with it);
    // .y = light-data buffer capacity in lights (max_lights_) — the shader
    //      derives its payload section offsets from it (cluster_lighting.hlsli).
    glm::uvec4 maxPerCluster;
};

class ClusterGrid {
public:
    ClusterGrid() = default;
    ~ClusterGrid();

    ClusterGrid(const ClusterGrid&) = delete;
    ClusterGrid& operator=(const ClusterGrid&) = delete;
    ClusterGrid(ClusterGrid&&) = delete;
    ClusterGrid& operator=(ClusterGrid&&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device, uint32_t max_lights = MAX_LIGHTS);
    void shutdown();

    // Reset per-frame light lists and capture camera state. When the
    // projection or near/far changed, the memoized cluster-AABB table is
    // rebuilt as a chunked fan-out over `js` (pure disjoint per-cluster
    // writes — no merge; docs/multithreading_plan.md §4.3).
    // Thread-safety: externally synchronized — one caller at a time (the
    // Prepare stage); every fanned-out chunk is joined before return.
    void begin_frame(core::jobs::JobSystem& js, const scene::CameraState& camera, uint32_t screen_w,
                     uint32_t screen_h, float near_plane, float far_plane);
    // Convenience overload servicing the canonical singleton pool
    // (JobSystem::instance()); identical semantics.
    void begin_frame(const scene::CameraState& camera, uint32_t screen_w, uint32_t screen_h, float near_plane,
                     float far_plane);

    void add_point_light(const PointLight& l);
    void add_spot_light(const SpotLight& l);

    // CPU-side bin lights into clusters; populates the per-cluster offset/count
    // table and concatenated light index list. Per-light cluster-range
    // insertion: O(lights x touched clusters), not O(lights x CLUSTER_COUNT).
    //
    // Binning fans out over `js` chunked by CLUSTER index (per-cluster form,
    // §4.3): each chunk owns a disjoint cluster subrange, walks the (small)
    // per-frame light list in ascending index order, and inserts into only
    // its own clusters' bins — every bin is written by exactly one chunk, so
    // per-bin light order (and the truncation cutoff) is identical to the
    // serial light-major loop: bit-identical output at any worker count
    // (§6.2). The conservative per-light cluster-range precompute runs once,
    // serially, before the fan-out; offsets/index packing stays serial
    // (order-dependent dirty diff). At <= 8 dynamic lights or with an
    // uninitialized pool the whole range runs as ONE inline chunk (same code
    // path — granularity control, not a second path).
    // Thread-safety: externally synchronized, like begin_frame().
    void build(core::jobs::JobSystem& js);
    // Convenience overload servicing the canonical singleton pool.
    void build();

    // Record the light data + cluster offsets + light indices SSBO uploads
    // into a caller-owned copy pass (the renderer's consolidated pre-pass
    // upload phase). Only the used prefix of the light-data buffer is
    // transferred, and the offsets buffer is skipped when bin contents are
    // unchanged. Staging lives in member scratch (light_staging_).
    void upload(SDL_GPUCopyPass* copy_pass);

    SDL_GPUBuffer* light_data_buffer() const;
    SDL_GPUBuffer* cluster_offsets_buffer() const;
    SDL_GPUBuffer* light_indices_buffer() const;

    const ClusterParams& params() const { return params_; }

    // Test/debug accessors.
    const std::vector<PointLight>& point_lights() const { return points_; }
    const std::vector<SpotLight>& spot_lights() const { return spots_; }
    const std::vector<uint32_t>& light_indices() const { return light_indices_; }
    // Per-cluster (offset, count) packed as uvec2 — flat array of 2*CLUSTER_COUNT u32.
    const std::vector<uint32_t>& cluster_offsets() const { return cluster_offsets_; }
    uint32_t total_lights() const { return static_cast<uint32_t>(points_.size() + spots_.size()); }

private:
    // Linear cluster -> sphere/AABB params, computed in begin_frame() from camera.
    struct ClusterBounds {
        glm::vec3 view_min;
        glm::vec3 view_max;
    };

    // View-space light position/radius, rebuilt in build() each frame.
    struct ViewLight {
        glm::vec3 view_pos;
        float radius;
    };

    // Conservative cluster-index box of one light (build() scratch,
    // precomputed serially before the binning fan-out): Z slices from the
    // exp_z_slice inverse, X/Y tiles from the projected NDC extent of the
    // sphere's view-space AABB. min_c/max_c are the box's linear-index
    // extremes, used for cheap whole-light rejection per chunk.
    struct LightRange {
        uint32_t x0, x1, y0, y1, z0, z1;
        uint32_t min_c, max_c;
        bool active;
    };

    void compute_cluster_bounds(core::jobs::JobSystem& js);
    static bool sphere_intersects_aabb(const glm::vec3& center, float radius, const glm::vec3& aabb_min,
                                       const glm::vec3& aabb_max);

    gpu::GPUDevice* device_ = nullptr;
    uint32_t max_lights_ = MAX_LIGHTS;

    // Per-frame CPU state.
    std::vector<PointLight> points_;
    std::vector<SpotLight> spots_;
    std::vector<LightHeader> headers_;          // size = points + spots
    std::vector<ClusterBounds> cluster_bounds_; // size = CLUSTER_COUNT
    std::vector<uint32_t> cluster_offsets_;     // 2 * CLUSTER_COUNT (offset, count) flat
    std::vector<uint32_t> light_indices_;       // packed indices into headers_
    std::vector<std::vector<uint32_t>> bins_;   // scratch, CLUSTER_COUNT entries
    std::vector<ViewLight> view_lights_;        // scratch, reused across frames
    std::vector<LightRange> light_ranges_;      // scratch, per-light conservative cluster box
    // Per-chunk truncation partials for the binning fan-out: each chunk
    // writes its own flag ONCE at chunk end (no shared cache-line writes in
    // the per-item loop); merged single-threaded in chunk-index order after
    // the join (§6.2).
    std::vector<uint8_t> chunk_truncated_;
    std::vector<uint8_t> light_staging_;        // scratch for upload() packing

    ClusterParams params_{};
    glm::mat4 last_projection_{0.0f};
    float last_near_ = 0.0f;
    float last_far_ = 0.0f;
    bool bounds_valid_ = false;
    bool warned_truncation_ = false;

    // Dirty flag for cluster_offsets_, set while build() packs the table —
    // skips the ~27KB offsets transfer when bin layout is unchanged
    // frame-over-frame. Cleared by upload().
    bool offsets_dirty_ = true;
    // True when cluster_offsets_ is known to be all zeros — lets the no-lights
    // build() skip re-zeroing (and re-uploading) an already-zero table.
    bool offsets_all_zero_ = false;

    // GPU buffers (storage buffers, dynamic).
    std::unique_ptr<gpu::GPUBuffer> light_data_buf_;
    std::unique_ptr<gpu::GPUBuffer> cluster_offsets_buf_;
    std::unique_ptr<gpu::GPUBuffer> light_indices_buf_;
};

} // namespace bracken::render::lighting
