#pragma once

#include "bracken/contracts/camera_state.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/render/lighting/cluster_index.hpp"
#include "bracken/render/lighting/light.hpp"
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <SDL3/SDL_gpu.h>
#include <span>
#include <vector>

namespace bracken::core::jobs {
class JobSystem;
}

namespace bracken::render::lighting {

// ClusterParams and the froxel index math live in cluster_index.hpp — the CPU
// half of the include the fragment and compute shaders share.

// A remap entry for a light that no longer exists this frame. Mirrors
// LIGHT_REMAP_INVALID in shaders/cluster_compute.hlsli.
inline constexpr uint32_t LIGHT_REMAP_INVALID = 0xFFFFFFFFu;

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

    // Submit a dynamic light for this frame. FALSE means the light was
    // DROPPED because the frame already holds max_lights (MAX_LIGHTS = 256)
    // — a host submitting 300 lights used to lose 44 of them, chosen by
    // lane-merge order, in total silence. Dropped lights are counted
    // (lights_dropped()) and warned about once per grid.
    [[nodiscard]] bool add_point_light(const PointLight& l);
    [[nodiscard]] bool add_spot_light(const SpotLight& l);

    // Lights refused since the last begin_frame() (see add_point_light).
    [[nodiscard]] uint32_t lights_dropped() const { return lights_dropped_; }
    // Light-to-cluster insertions the last build() could not keep in full
    // because a bin was already at MAX_LIGHTS_PER_CLUSTER (see
    // RenderStats/FrameDiagnostics). A refused insertion may still have
    // EVICTED a dimmer incumbent — the bin keeps the highest-contribution
    // lights, not the earliest — but one light's worth of the froxel's
    // lighting was lost either way, which is the number a host acts on.
    [[nodiscard]] uint32_t cluster_lights_dropped() const { return cluster_lights_dropped_; }

    // CPU-side bin lights into clusters; populates the per-cluster offset/count
    // table and concatenated light index list. Per-light cluster-range
    // insertion: O(lights x touched clusters), not O(lights x CLUSTER_COUNT).
    //
    // Binning fans out over `js` chunked by CLUSTER index (per-cluster form,
    // §4.3): each chunk owns a disjoint cluster subrange, walks the (small)
    // per-frame light list in ascending index order, and inserts into only
    // its own clusters' bins — every bin is written by exactly one chunk, so
    // per-bin light order (and which lights survive MAX_LIGHTS_PER_CLUSTER)
    // is identical to the serial light-major loop: bit-identical output at
    // any worker count (§6.2). The conservative per-light cluster-range
    // precompute runs once, serially, before the fan-out; offsets/index
    // packing stays serial
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

    // Forget last frame's light identities. Call this on any frame the grid
    // does NOT build (the renderer skips binning entirely when the scene has
    // no dynamic lights) — otherwise the next build would remap against a
    // "previous frame" that is two or more frames old, and a consumer's
    // temporal history would be reused across a gap where nothing was lit.
    void reset_light_history();

    SDL_GPUBuffer* light_data_buffer() const;
    SDL_GPUBuffer* cluster_offsets_buffer() const;
    SDL_GPUBuffer* light_indices_buffer() const;
    // uint[max_lights]: the stable id of each light in PACKED order — index it
    // with the same header index the cluster light list yields.
    SDL_GPUBuffer* light_stable_ids_buffer() const;
    // uint[max_lights]: PREVIOUS frame's packed index -> this frame's packed
    // index, LIGHT_REMAP_INVALID where the light is gone. Entries at or past
    // the previous frame's light count are LIGHT_REMAP_INVALID too.
    SDL_GPUBuffer* light_remap_buffer() const;

    const ClusterParams& params() const { return params_; }

    // Per-froxel view-space AABBs, CLUSTER_COUNT entries — the memoized table
    // begin_frame() rebuilds when the projection or near/far changes. Exposed
    // because the light binner is no longer the only one: the emissive record
    // binner (render/raytracing/emissive_clusters.hpp) tests the SAME boxes, and
    // recomputing them there would be a second copy of the unprojection that
    // could disagree with this one about which froxel a point is in. EMPTY
    // before the first begin_frame().
    // Thread-safety: main-thread/Prepare, like every other accessor here — valid
    // until the next begin_frame().
    [[nodiscard]] std::span<const ClusterBounds> cluster_bounds() const { return cluster_bounds_; }

    // Test/debug accessors.
    const std::vector<PointLight>& point_lights() const { return points_; }
    const std::vector<SpotLight>& spot_lights() const { return spots_; }
    const std::vector<uint32_t>& light_indices() const { return light_indices_; }
    // Resolved stable ids in packed (header) order — the CPU side of
    // light_stable_ids_buffer(). Never 0 for a light that made it into the
    // frame. See contracts/light.hpp for the identity contract.
    const std::vector<uint32_t>& stable_ids() const { return stable_ids_; }
    // The CPU side of light_remap_buffer(); always max_lights entries.
    const std::vector<uint32_t>& index_remap() const { return remap_; }
    // Per-cluster (offset, count) packed as uvec2 — flat array of 2*CLUSTER_COUNT u32.
    const std::vector<uint32_t>& cluster_offsets() const { return cluster_offsets_; }
    uint32_t total_lights() const { return static_cast<uint32_t>(points_.size() + spots_.size()); }

private:
    // View-space light bounds + the two things the refine needs beyond a
    // sphere, rebuilt in build() each frame.
    //
    // A spot used to be binned as its full radius SPHERE, which is how a 32
    // degree cone ended up in 94% of the froxel grid instead of 8% — the solid
    // angle ratio (1 - cos 32)/2 says 7.6%, and the measurement agreed. Every
    // one of those froxels costs the fragment loop a header + an 80-byte
    // SpotLight load + a `spotConeFactor` that returns zero, and costs the
    // traced Full-DI tier a RIS candidate draw with p̂ = 0 (the froxel list is
    // that pass's candidate domain, rt_local_shadow.comp.hlsl).
    struct ViewLight {
        glm::vec3 view_pos;
        float radius;
        // Spot axis in VIEW space; (0,0,-1) and unused for points.
        glm::vec3 view_dir;
        // Cone half-angle cosine/sine, precomputed once per light so the
        // per-froxel test is trig-free. cos_outer <= 0 means "no cone test" —
        // a point light, a half-angle at or past 90 degrees, or a spot whose
        // cone pair is not well-formed (see build()).
        float cos_outer;
        float sin_outer;
        // Rec.709 luminance of color * intensity — the light's brightness with
        // distance left out, half of the per-froxel eviction score. Computed
        // here because build() already has the payload in hand.
        float lum;
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
    // Resolve every light's stable id (host-supplied wins, else the quantized
    // content hash) and diff this frame's identities against last frame's into
    // remap_. Runs at the top of build(), serially, in emission order — the
    // determinism the identity contract promises comes from that order and
    // nothing else. Two <= MAX_LIGHTS passes; no allocation after warm-up.
    void assign_stable_ids_and_remap();
    // Count + warn-once for a light past the frame cap; always returns false.
    bool reject_over_cap();
    static bool sphere_intersects_aabb(const glm::vec3& center, float radius, const glm::vec3& aabb_min,
                                       const glm::vec3& aabb_max);
    // Conservative spot-cone reject, ANDed with the sphere test above. Only
    // meaningful when vl.cos_outer > 0; see ViewLight and build().
    static bool cone_intersects_aabb(const ViewLight& vl, const ClusterBounds& cb);
    // A light's peak possible contribution anywhere in one froxel: its
    // luminance times the forward loop's own range window evaluated at the
    // froxel's CLOSEST point. Only ever called on the overflow path.
    static float bin_score(const ViewLight& vl, const ClusterBounds& cb);

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
    // Parallel to bins_, and EMPTY for every bin that never overflowed: the
    // eviction scores are filled lazily, on a bin's first refused insertion,
    // so a scene that stays under the cap allocates and evaluates nothing.
    std::vector<std::vector<float>> bin_scores_;
    std::vector<ViewLight> view_lights_;        // scratch, reused across frames
    std::vector<LightRange> light_ranges_;      // scratch, per-light conservative cluster box
    // Per-chunk truncation partials for the binning fan-out: each chunk
    // accumulates a PRIVATE count and writes its own slot ONCE at chunk end
    // (no shared cache-line writes in the per-item loop); merged
    // single-threaded in chunk-index order after the join (§6.2).
    std::vector<uint32_t> chunk_truncated_;
    std::vector<uint8_t> light_staging_;        // scratch for upload() packing

    // Persistent light identity (contracts/light.hpp). All three are plain
    // vectors walked in index order — no hashing container, no pointer-order
    // iteration, so the result is bit-identical run to run.
    std::vector<uint32_t> stable_ids_;      // this frame, packed (header) order
    std::vector<uint32_t> prev_stable_ids_; // last built frame, same order
    std::vector<uint32_t> remap_;           // max_lights_ entries, prev idx -> cur idx

    ClusterParams params_{};
    glm::mat4 last_projection_{0.0f};
    float last_near_ = 0.0f;
    float last_far_ = 0.0f;
    bool bounds_valid_ = false;
    bool warned_truncation_ = false;
    bool warned_dropped_ = false;
    uint32_t lights_dropped_ = 0;         // reset each begin_frame()
    uint32_t cluster_lights_dropped_ = 0; // set by build()

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
    // Identity side-channel: two uint[max_lights] buffers uploaded with the
    // cluster buffers in the same copy pass. They are PARALLEL to the light
    // data rather than fields inside it because LightHeader is exactly 8 bytes
    // on both sides (the shader hardcodes that stride), and because a consumer
    // wants both arrays indexed by PACKED index — one load, no header hop.
    std::unique_ptr<gpu::GPUBuffer> light_stable_ids_buf_;
    std::unique_ptr<gpu::GPUBuffer> light_remap_buf_;
};

} // namespace bracken::render::lighting
