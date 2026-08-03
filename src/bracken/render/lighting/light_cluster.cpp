#include "bracken/render/lighting/light_cluster.hpp"

#include "bracken/core/assert.hpp"
#include "bracken/core/hash.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <limits>

namespace bracken::render::lighting {

namespace {
constexpr const char* LOG_TAG = "lights";

// Std140 layout sanity. PointLight: 3 float4-sized rows in 48B, SpotLight: 4
// in 64B (the third/fourth row carries casts_shadows + the engine-assigned
// shadow_slot the shader reads — cluster_lighting.hlsli loadPointLight/
// loadSpotLight hardcode these strides and the shadow_slot byte offsets).
static_assert(sizeof(PointLight) == 48, "PointLight must be 48 bytes (std140)");
static_assert(offsetof(PointLight, shadow_slot) == 36, "PointLight::shadow_slot read at byte 36 by the shader");
static_assert(sizeof(SpotLight) == 80, "SpotLight must be 80 bytes (std140)");
static_assert(offsetof(SpotLight, shadow_slot) == 56, "SpotLight::shadow_slot read at byte 56 by the shader");
static_assert(sizeof(LightHeader) == 8, "LightHeader must be 8 bytes");
// stable_id took over declared padding (contracts/light.hpp). These pin that
// it stayed padding-shaped: the sizes above are unchanged, and the bytes it
// occupies are past every offset the shader loads.
static_assert(offsetof(PointLight, stable_id) == 40, "PointLight::stable_id must sit in the old _pad1 bytes");
static_assert(offsetof(SpotLight, stable_id) == 60, "SpotLight::stable_id must sit in the old _pad1 bytes");
// R6's source_radius. PointLight had a spare declared pad and took it in place
// (48 bytes, unchanged); SpotLight was full at 64 and GREW to 80, which is why
// its stride literal moved in cluster_light_load.hlsli and rt_hit_shade.hlsli.
static_assert(offsetof(PointLight, source_radius) == 44, "PointLight::source_radius read at byte 44");
static_assert(offsetof(SpotLight, source_radius) == 64, "SpotLight::source_radius read at byte 64");

// Single GPU layout combining headers + point payloads + spot payloads.
// Layout: [u32 headerCount][u32 pointCount][u32 spotCount][u32 _pad]
//         [LightHeader * MAX_LIGHTS]
//         [PointLight  * MAX_LIGHTS]
//         [SpotLight   * MAX_LIGHTS]
struct LightDataHeader {
    uint32_t header_count;
    uint32_t point_count;
    uint32_t spot_count;
    uint32_t _pad;
};

constexpr size_t light_data_buffer_size(uint32_t max_lights) {
    return sizeof(LightDataHeader) + sizeof(LightHeader) * max_lights + sizeof(PointLight) * max_lights +
           sizeof(SpotLight) * max_lights;
}

constexpr size_t cluster_offsets_buffer_size() {
    return static_cast<size_t>(CLUSTER_COUNT) * 2 * sizeof(uint32_t);
}

constexpr size_t light_indices_buffer_size() {
    return static_cast<size_t>(CLUSTER_COUNT) * MAX_LIGHTS_PER_CLUSTER * sizeof(uint32_t);
}

// The two identity side-channel buffers are both uint[max_lights] (1 KB each
// at MAX_LIGHTS = 256).
constexpr size_t light_id_buffer_size(uint32_t max_lights) {
    return static_cast<size_t>(max_lights) * sizeof(uint32_t);
}

// ---------------------------------------------------------------------------
// Persistent light identity (contracts/light.hpp)
// ---------------------------------------------------------------------------

// Round-to-nearest quantization. Round-to-nearest, not floor: floor puts a
// boundary at every multiple of the quantum in each axis independently, which
// triples the odds a jittering light straddles one.
int32_t quantize(float v, float q) {
    return static_cast<int32_t>(std::lround(v / q));
}

// Intensity is snapped in log2 because a flickering torch animates it by a
// RATIO. An absolute quantum here would re-id every torch in the village
// every frame, which is precisely the failure this hash exists to avoid.
int32_t quantize_intensity(float intensity) {
    if (!(intensity > 0.0f)) {
        return std::numeric_limits<int32_t>::min(); // one bucket for "off"
    }
    return static_cast<int32_t>(std::lround(std::log2(intensity) / LIGHT_ID_INTENSITY_OCTAVE));
}

uint32_t hash_step(uint32_t h, uint32_t word) {
    // FNV-1a over the four bytes of `word` — core::fnv1a's mixing, applied to
    // an integer stream instead of a string.
    for (int b = 0; b < 4; ++b) {
        h ^= (word >> (b * 8)) & 0xFFu;
        h *= core::k_fnv1a_prime_32;
    }
    return h;
}

uint32_t hash_step(uint32_t h, int32_t word) {
    return hash_step(h, static_cast<uint32_t>(word));
}

// The quantized content hash: type + snapped position + snapped color +
// snapped intensity. Never returns 0 (0 is the "engine assigns" sentinel).
uint32_t content_hash(uint32_t type, const glm::vec3& position, const glm::vec3& color, float intensity) {
    uint32_t h = core::k_fnv1a_offset_basis_32;
    h = hash_step(h, type);
    h = hash_step(h, quantize(position.x, LIGHT_ID_POSITION_QUANTUM));
    h = hash_step(h, quantize(position.y, LIGHT_ID_POSITION_QUANTUM));
    h = hash_step(h, quantize(position.z, LIGHT_ID_POSITION_QUANTUM));
    h = hash_step(h, quantize(color.r, LIGHT_ID_COLOR_QUANTUM));
    h = hash_step(h, quantize(color.g, LIGHT_ID_COLOR_QUANTUM));
    h = hash_step(h, quantize(color.b, LIGHT_ID_COLOR_QUANTUM));
    h = hash_step(h, quantize_intensity(intensity));
    return h == 0u ? 1u : h;
}

// View-space frustum corner for cluster (cx,cy,cz) at the given Z slice.
glm::vec3 unproject_corner(uint32_t cx, uint32_t cy, float view_z, const glm::mat4& invProj) {
    // Cluster (cx,cy) maps to NDC tile [-1..1] in X and Y: cy = 0 is NDC
    // y = -1, the BOTTOM of the screen. The shader lookup starts from a
    // top-left-origin SV_Position uv, so clusterIndexFromScreen
    // (clusters.hlsli) flips Y to land in this row convention.
    float ndc_x = (static_cast<float>(cx) / CLUSTER_DIM_X) * 2.0f - 1.0f;
    float ndc_y = (static_cast<float>(cy) / CLUSTER_DIM_Y) * 2.0f - 1.0f;
    glm::vec4 clip(ndc_x, ndc_y, 0.0f, 1.0f);
    // Convert NDC to view at z=near via invProj for direction; we then rescale to view_z.
    glm::vec4 view = invProj * clip;
    if (std::abs(view.w) > 1e-6f) {
        view /= view.w;
    }
    // Ray from origin through view; hit plane at z = view_z (negative for forward).
    if (std::abs(view.z) < 1e-6f) {
        return glm::vec3(0.0f);
    }
    float t = view_z / view.z;
    return {view.x * t, view.y * t, view_z};
}

float exp_z_slice(uint32_t z, float near_z, float far_z) {
    return near_z * std::pow(far_z / near_z, static_cast<float>(z) / CLUSTER_DIM_Z);
}

// Fan-out grains (docs/multithreading_plan.md §4.3, cluster light binning).
// Bounds rebuild: ~256 clusters per chunk -> 13-14 chunks over the
// 3,456-cluster grid.
constexpr std::size_t BOUNDS_GRAIN = 256;
// Binning: chunk the cluster range in whole Z-slices (CLUSTER_DIM_X *
// CLUSTER_DIM_Y = 144 clusters per slice); 3 slices per chunk -> 8 chunks.
// Slice alignment keeps a light's row clipping mostly chunk-local.
constexpr std::size_t BIN_GRAIN = static_cast<std::size_t>(CLUSTER_DIM_X) * CLUSTER_DIM_Y * 3;
// Below this many dynamic lights the binning runs as ONE inline chunk (same
// code path — granularity control, §6.2): the playground's handful of lights
// must not pay chunk-dispatch overhead.
constexpr std::size_t BIN_MIN_PARALLEL_LIGHTS = 9;

} // namespace

ClusterGrid::~ClusterGrid() {
    shutdown();
}

bool ClusterGrid::init(gpu::GPUDevice& device, uint32_t max_lights) {
    device_ = &device;
    max_lights_ = max_lights;

    points_.reserve(max_lights_);
    spots_.reserve(max_lights_);
    headers_.reserve(max_lights_);
    cluster_bounds_.assign(CLUSTER_COUNT, ClusterBounds{});
    cluster_offsets_.assign(CLUSTER_COUNT * 2, 0u);
    offsets_all_zero_ = true;
    offsets_dirty_ = true;
    light_indices_.reserve(CLUSTER_COUNT * 4);
    bins_.assign(CLUSTER_COUNT, {});
    for (auto& b : bins_) b.reserve(8);

    light_data_buf_ =
        gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::Storage, light_data_buffer_size(max_lights_));
    cluster_offsets_buf_ =
        gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::Storage, cluster_offsets_buffer_size());
    light_indices_buf_ =
        gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::Storage, light_indices_buffer_size());
    light_stable_ids_buf_ =
        gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::Storage, light_id_buffer_size(max_lights_));
    light_remap_buf_ =
        gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::Storage, light_id_buffer_size(max_lights_));
    if (!light_data_buf_ || !cluster_offsets_buf_ || !light_indices_buf_ || !light_stable_ids_buf_ ||
        !light_remap_buf_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "ClusterGrid::init: failed to allocate storage buffers");
        shutdown();
        return false;
    }
    return true;
}

void ClusterGrid::shutdown() {
    light_data_buf_.reset();
    cluster_offsets_buf_.reset();
    light_indices_buf_.reset();
    light_stable_ids_buf_.reset();
    light_remap_buf_.reset();
    stable_ids_.clear();
    prev_stable_ids_.clear();
    remap_.clear();
    points_.clear();
    spots_.clear();
    headers_.clear();
    cluster_bounds_.clear();
    cluster_offsets_.clear();
    light_indices_.clear();
    bins_.clear();
    view_lights_.clear();
    light_ranges_.clear();
    chunk_truncated_.clear();
    light_staging_.clear();
    offsets_dirty_ = true;
    offsets_all_zero_ = false;
    device_ = nullptr;
}

void ClusterGrid::begin_frame(const scene::CameraState& camera, uint32_t screen_w, uint32_t screen_h, float near_plane,
                              float far_plane) {
    begin_frame(core::jobs::JobSystem::instance(), camera, screen_w, screen_h, near_plane, far_plane);
}

void ClusterGrid::begin_frame(core::jobs::JobSystem& js, const scene::CameraState& camera, uint32_t screen_w,
                              uint32_t screen_h, float near_plane, float far_plane) {
    BRACKEN_PROFILE_ZONE_COLOR("ClusterGrid::begin_frame", 0x27AE60);
    // Allow CPU-only callers (tests) to skip init() — lazily size scratch arrays.
    if (cluster_bounds_.size() != CLUSTER_COUNT) {
        cluster_bounds_.assign(CLUSTER_COUNT, ClusterBounds{});
    }
    if (cluster_offsets_.size() != CLUSTER_COUNT * 2) {
        cluster_offsets_.assign(CLUSTER_COUNT * 2, 0u);
        offsets_all_zero_ = true;
        offsets_dirty_ = true;
    }
    if (bins_.size() != CLUSTER_COUNT) {
        bins_.assign(CLUSTER_COUNT, {});
    }

    points_.clear();
    spots_.clear();
    headers_.clear();
    light_indices_.clear();
    // cluster_offsets_ is NOT cleared here: it keeps last build()'s packing so
    // build() can diff against it (the upload dirty check) and rewrites it.
    for (auto& bin : bins_) bin.clear();
    warned_truncation_ = false;
    lights_dropped_ = 0;
    cluster_lights_dropped_ = 0;

    params_.view = camera.view;
    // The 4x4 inverse (~80 flops) only depends on the projection — recompute it
    // when the matrix actually changes, not every frame.
    const bool projection_changed = camera.projection != last_projection_;
    if (projection_changed) {
        params_.invProjection = glm::inverse(camera.projection);
        last_projection_ = camera.projection;
    }
    params_.screenSize = glm::vec4(static_cast<float>(screen_w), static_cast<float>(screen_h),
                                   screen_w > 0 ? 1.0f / static_cast<float>(screen_w) : 0.0f,
                                   screen_h > 0 ? 1.0f / static_cast<float>(screen_h) : 0.0f);
    float near_safe = std::max(near_plane, 1e-3f);
    float far_safe = std::max(far_plane, near_safe + 1e-3f);
    float log_ratio = std::log(far_safe / near_safe);
    params_.zPlanes = glm::vec4(near_safe, far_safe, log_ratio, log_ratio > 0.0f ? 1.0f / log_ratio : 0.0f);
    params_.gridDim = glm::uvec4(CLUSTER_DIM_X, CLUSTER_DIM_Y, CLUSTER_DIM_Z, 0u);
    // .x caps the shader's per-cluster loop; .y is the light-data buffer
    // capacity the shader derives payload offsets from (cluster_lighting.hlsli).
    params_.maxPerCluster = glm::uvec4(MAX_LIGHTS_PER_CLUSTER, max_lights_, 0u, 0u);

    // Cluster AABBs only depend on the projection + near/far. Recompute when
    // they change; otherwise the previous frame's bounds are still valid.
    if (!bounds_valid_ || projection_changed || near_safe != last_near_ || far_safe != last_far_) {
        compute_cluster_bounds(js);
        last_near_ = near_safe;
        last_far_ = far_safe;
        bounds_valid_ = true;
    }
}

// Shared refusal path: a light past the frame cap is DROPPED, and dropping it
// silently is how a host ends up debugging "why is that lamp black" against a
// renderer that is working exactly as designed. Warn once per grid (the cap
// is a standing property of the scene, not an event) and count every one.
bool ClusterGrid::reject_over_cap() {
    ++lights_dropped_;
    if (!warned_dropped_) {
        BRACKEN_LOG_WARN(LOG_TAG,
                         "ClusterGrid: light budget MAX_LIGHTS={} reached; further lights this frame are dropped "
                         "(submit fewer lights, or raise max_lights at init())",
                         max_lights_);
        warned_dropped_ = true;
    }
    return false;
}

bool ClusterGrid::add_point_light(const PointLight& l) {
    if (points_.size() + spots_.size() >= max_lights_) {
        return reject_over_cap();
    }
    uint32_t payload_idx = static_cast<uint32_t>(points_.size());
    points_.push_back(l);
    headers_.push_back({LIGHT_TYPE_POINT, payload_idx});
    return true;
}

bool ClusterGrid::add_spot_light(const SpotLight& l) {
    if (points_.size() + spots_.size() >= max_lights_) {
        return reject_over_cap();
    }
    uint32_t payload_idx = static_cast<uint32_t>(spots_.size());
    spots_.push_back(l);
    headers_.push_back({LIGHT_TYPE_SPOT, payload_idx});
    return true;
}

void ClusterGrid::reset_light_history() {
    prev_stable_ids_.clear();
    if (remap_.size() != max_lights_) {
        remap_.assign(max_lights_, LIGHT_REMAP_INVALID);
    } else {
        std::fill(remap_.begin(), remap_.end(), LIGHT_REMAP_INVALID);
    }
}

void ClusterGrid::assign_stable_ids_and_remap() {
    BRACKEN_PROFILE_ZONE_COLOR("ClusterGrid::assign_stable_ids", 0x27AE60);
    const std::size_t n = headers_.size();
    BRACKEN_ASSERT(n <= max_lights_, "packed light count cannot exceed the frame cap");
    stable_ids_.assign(n, 0u);

    // Pass 1: host-supplied ids, verbatim. They go FIRST so an auto-assigned
    // hash can never squat on one — the contract says the host wins, and that
    // has to hold regardless of emission order.
    for (std::size_t i = 0; i < n; ++i) {
        const auto& h = headers_[i];
        stable_ids_[i] = (h.type == LIGHT_TYPE_POINT) ? points_[h.payload_index].stable_id
                                                      : spots_[h.payload_index].stable_id;
    }

    // Is `id` already claimed by a light other than `self`? A linear scan over
    // at most MAX_LIGHTS entries, run at most a handful of times — the whole
    // pass is <= 256 * 256 integer compares in the pathological case where
    // every light in the frame is identical, and typically ~n.
    const auto claimed_by_other = [&](uint32_t id, std::size_t self) {
        for (std::size_t j = 0; j < n; ++j) {
            if (j != self && stable_ids_[j] == id) {
                return true;
            }
        }
        return false;
    };

    // Pass 2: hash the lights that asked the engine to (stable_id == 0), in
    // EMISSION order, disambiguating collisions with an occurrence counter.
    // Two identical lanterns therefore get distinct ids; which lantern gets
    // which follows emission order, and the contract documents that.
    for (std::size_t i = 0; i < n; ++i) {
        if (stable_ids_[i] != 0u) {
            continue;
        }
        const auto& h = headers_[i];
        uint32_t base = 0;
        if (h.type == LIGHT_TYPE_POINT) {
            const auto& p = points_[h.payload_index];
            base = content_hash(LIGHT_TYPE_POINT, p.position, p.color, p.intensity);
        } else {
            const auto& s = spots_[h.payload_index];
            base = content_hash(LIGHT_TYPE_SPOT, s.position, s.color, s.intensity);
        }
        uint32_t id = base;
        for (uint32_t occurrence = 1; claimed_by_other(id, i); ++occurrence) {
            id = hash_step(base, occurrence);
            if (id == 0u) {
                id = 1u;
            }
        }
        stable_ids_[i] = id;
    }

    // Write the resolved ids back into the light copies the frame uploads, so
    // "what identity did this light have" is answerable from the light data
    // alone (the parallel GPU buffer below is the fast path, not the record).
    for (std::size_t i = 0; i < n; ++i) {
        const auto& h = headers_[i];
        if (h.type == LIGHT_TYPE_POINT) {
            points_[h.payload_index].stable_id = stable_ids_[i];
        } else {
            spots_[h.payload_index].stable_id = stable_ids_[i];
        }
    }

    // The remap: previous frame's packed index -> this frame's packed index,
    // LIGHT_REMAP_INVALID where the light is gone. Lights that APPEARED this
    // frame simply have no prev entry pointing at them — a consumer detects
    // them by finding no reservoir to reproject, which is the correct answer.
    if (remap_.size() != max_lights_) {
        remap_.assign(max_lights_, LIGHT_REMAP_INVALID);
    } else {
        std::fill(remap_.begin(), remap_.end(), LIGHT_REMAP_INVALID);
    }
    BRACKEN_ASSERT(prev_stable_ids_.size() <= max_lights_, "prev light count cannot exceed the frame cap");
    for (std::size_t p = 0; p < prev_stable_ids_.size(); ++p) {
        const uint32_t prev_id = prev_stable_ids_[p];
        for (std::size_t c = 0; c < n; ++c) {
            if (stable_ids_[c] == prev_id) {
                remap_[p] = static_cast<uint32_t>(c);
                break;
            }
        }
    }
    prev_stable_ids_.assign(stable_ids_.begin(), stable_ids_.end());
}

void ClusterGrid::compute_cluster_bounds(core::jobs::JobSystem& js) {
    BRACKEN_PROFILE_ZONE_COLOR("ClusterGrid::compute_cluster_bounds", 0x27AE60);
    const float near_z = -params_.zPlanes.x;
    const float far_z = -params_.zPlanes.y;
    const glm::mat4& invProj = params_.invProjection;

    // Per-slice view-space depth range, precomputed serially so every chunk
    // reads the exact floats the old per-slice loop produced (no redundant
    // pow() per cluster, and trivially bit-identical, §6.2).
    std::array<glm::vec2, CLUSTER_DIM_Z> slice_z; // .x = zn, .y = zf
    for (uint32_t z = 0; z < CLUSTER_DIM_Z; ++z) {
        // View-space depths are negative (camera looks down -Z).
        float zn = -exp_z_slice(z, -near_z, -far_z);
        float zf = -exp_z_slice(z + 1, -near_z, -far_z);
        // Ensure ordering: zn closer to camera than zf, both negative.
        if (zn < zf) {
            std::swap(zn, zf);
        }
        slice_z[z] = {zn, zf};
    }

    // Chunked fan-out over the 3,456-cluster range (§4.3): each cluster's
    // bounds are a pure function of its index — disjoint per-element writes,
    // no partials, no merge. The guards inside parallel_for (uninitialized
    // pool, one worker) collapse this to one inline chunk on the caller.
    core::jobs::parallel_for(
        js, 0, CLUSTER_COUNT,
        [&](std::size_t chunk_begin, std::size_t chunk_end) {
            for (std::size_t ci = chunk_begin; ci < chunk_end; ++ci) {
                const uint32_t idx = static_cast<uint32_t>(ci);
                const uint32_t z = idx / (CLUSTER_DIM_X * CLUSTER_DIM_Y);
                const uint32_t rem = idx % (CLUSTER_DIM_X * CLUSTER_DIM_Y);
                const uint32_t y = rem / CLUSTER_DIM_X;
                const uint32_t x = rem % CLUSTER_DIM_X;
                const float zn = slice_z[z].x;
                const float zf = slice_z[z].y;

                glm::vec3 corners[8];
                corners[0] = unproject_corner(x, y, zn, invProj);
                corners[1] = unproject_corner(x + 1, y, zn, invProj);
                corners[2] = unproject_corner(x, y + 1, zn, invProj);
                corners[3] = unproject_corner(x + 1, y + 1, zn, invProj);
                corners[4] = unproject_corner(x, y, zf, invProj);
                corners[5] = unproject_corner(x + 1, y, zf, invProj);
                corners[6] = unproject_corner(x, y + 1, zf, invProj);
                corners[7] = unproject_corner(x + 1, y + 1, zf, invProj);

                glm::vec3 mn = corners[0];
                glm::vec3 mx = corners[0];
                for (int i = 1; i < 8; ++i) {
                    mn = glm::min(mn, corners[i]);
                    mx = glm::max(mx, corners[i]);
                }
                cluster_bounds_[ci] = {mn, mx};
            }
        },
        BOUNDS_GRAIN);
}

bool ClusterGrid::sphere_intersects_aabb(const glm::vec3& center, float radius, const glm::vec3& aabb_min,
                                         const glm::vec3& aabb_max) {
    glm::vec3 closest = glm::clamp(center, aabb_min, aabb_max);
    glm::vec3 d = center - closest;
    return glm::dot(d, d) <= radius * radius;
}

void ClusterGrid::build() {
    build(core::jobs::JobSystem::instance());
}

void ClusterGrid::build(core::jobs::JobSystem& js) {
    BRACKEN_PROFILE_ZONE_COLOR("ClusterGrid::build", 0x27AE60);
    const glm::mat4& view = params_.view;

    // Identity first: it must run even on a frame with zero lights, or the
    // remap would still describe the frame before that one.
    assign_stable_ids_and_remap();

    const std::size_t light_count = headers_.size();
    if (light_count == 0) {
        // No dynamic lights: bins/indices are already empty from begin_frame().
        // The offsets table persists across frames for the upload dirty check —
        // it only needs (re)zeroing if the previous build wrote any counts.
        if (!offsets_all_zero_) {
            std::fill(cluster_offsets_.begin(), cluster_offsets_.end(), 0u);
            offsets_all_zero_ = true;
            offsets_dirty_ = true;
        }
        return;
    }
    view_lights_.resize(light_count);
    for (size_t i = 0; i < light_count; ++i) {
        const auto& h = headers_[i];
        glm::vec3 world_pos(0.0f);
        float radius = 0.0f;
        if (h.type == LIGHT_TYPE_POINT) {
            const auto& p = points_[h.payload_index];
            world_pos = p.position;
            radius = p.radius;
        } else {
            const auto& s = spots_[h.payload_index];
            world_pos = s.position;
            radius = s.radius;
        }
        view_lights_[i].view_pos = glm::vec3(view * glm::vec4(world_pos, 1.0f));
        view_lights_[i].radius = radius;
    }

    // Per-light cluster-range insertion: instead of testing every one of the
    // CLUSTER_COUNT clusters against every light (O(clusters x lights)),
    // compute each light's conservative cluster-index box — Z slices from the
    // inverse of exp_z_slice, X/Y tiles from the projected NDC extent of the
    // sphere's view-space AABB — then run the exact sphere/AABB refine only
    // inside it: O(lights x touched clusters). The range precompute below
    // runs ONCE, serially, before the fan-out (no redundant per-chunk sphere
    // math); insertion then chunks the CLUSTER range — see bin_chunk.
    const float near_z = params_.zPlanes.x;
    const float far_z = params_.zPlanes.y;
    // The shared froxel Z math (cluster_index.hpp) — the same function the
    // shaders' cluster_index.hlsli mirrors. Binning and lookup drifting apart
    // is exactly the bug that header exists to make impossible.
    auto z_slice_of = [&](float depth) { return cluster_z_slice(depth, params_); };
    // NDC -> tile index, matching unproject_corner(): index 0 starts at NDC -1
    // (left / bottom of screen).
    auto tile_of = [](float ndc, uint32_t dim) {
        float t = std::floor((ndc * 0.5f + 0.5f) * static_cast<float>(dim));
        return static_cast<uint32_t>(std::clamp(t, 0.0f, static_cast<float>(dim - 1)));
    };

    light_ranges_.resize(light_count);
    for (std::size_t i = 0; i < light_count; ++i) {
        const auto& vl = view_lights_[i];
        auto& lr = light_ranges_[i];
        lr.active = false;
        if (vl.radius <= 0.0f) {
            continue;
        }
        // Depth range covered by the sphere (positive distance forward of camera).
        const float depth_min = -vl.view_pos.z - vl.radius;
        const float depth_max = -vl.view_pos.z + vl.radius;
        if (depth_max < near_z || depth_min > far_z) {
            continue; // entirely in front of the near or beyond the far plane
        }
        const uint32_t z0 = z_slice_of(depth_min);
        const uint32_t z1 = z_slice_of(depth_max);

        // X/Y tile range from the NDC bounds of the sphere's view-space AABB
        // corners (the projective image of a convex box is the hull of its
        // projected corners, so this bounds every lit point). If the sphere
        // reaches the near plane its projection is unbounded — fall back to the
        // full range and let the sphere/AABB refine reject clusters.
        uint32_t x0 = 0, x1 = CLUSTER_DIM_X - 1;
        uint32_t y0 = 0, y1 = CLUSTER_DIM_Y - 1;
        if (depth_min >= near_z) {
            const glm::vec3 bb_min = vl.view_pos - glm::vec3(vl.radius);
            const glm::vec3 bb_max = vl.view_pos + glm::vec3(vl.radius);
            glm::vec2 ndc_min(std::numeric_limits<float>::max());
            glm::vec2 ndc_max(std::numeric_limits<float>::lowest());
            bool projected = true;
            for (int corner = 0; corner < 8; ++corner) {
                const glm::vec4 clip = last_projection_ * glm::vec4((corner & 1) ? bb_max.x : bb_min.x,
                                                                    (corner & 2) ? bb_max.y : bb_min.y,
                                                                    (corner & 4) ? bb_max.z : bb_min.z, 1.0f);
                if (clip.w <= 1e-6f) {
                    projected = false; // degenerate projection — keep full range
                    break;
                }
                const glm::vec2 ndc = glm::vec2(clip) / clip.w;
                ndc_min = glm::min(ndc_min, ndc);
                ndc_max = glm::max(ndc_max, ndc);
            }
            if (projected) {
                if (ndc_max.x < -1.0f || ndc_min.x > 1.0f || ndc_max.y < -1.0f || ndc_min.y > 1.0f) {
                    continue; // fully outside the frustum sideways
                }
                x0 = tile_of(ndc_min.x, CLUSTER_DIM_X);
                x1 = tile_of(ndc_max.x, CLUSTER_DIM_X);
                y0 = tile_of(ndc_min.y, CLUSTER_DIM_Y);
                y1 = tile_of(ndc_max.y, CLUSTER_DIM_Y);
            }
        }
        lr = {x0, x1, y0, y1, z0, z1,
              /*min_c=*/(z0 * CLUSTER_DIM_Y + y0) * CLUSTER_DIM_X + x0,
              /*max_c=*/(z1 * CLUSTER_DIM_Y + y1) * CLUSTER_DIM_X + x1,
              /*active=*/true};
    }

    // Per-cluster insertion chunked by cluster index (§4.3): a chunk owns the
    // disjoint cluster subrange [chunk_begin, chunk_end) and is the ONLY
    // writer of those clusters' bins — no merge, no contention. Within a
    // chunk lights are walked in ascending index order, so each bin receives
    // exactly the sequence the serial light-major loop produced (including
    // the MAX_LIGHTS_PER_CLUSTER cutoff keeping the lowest indices):
    // bit-identical at any worker count / any grain (§6.2).
    const std::size_t chunk_count = (CLUSTER_COUNT + BIN_GRAIN - 1) / BIN_GRAIN;
    chunk_truncated_.assign(chunk_count, 0);
    auto bin_chunk = [&](std::size_t chunk_begin, std::size_t chunk_end) {
        uint32_t truncated_local = 0; // per-chunk partial; published once below
        for (std::size_t i = 0; i < light_count; ++i) {
            const auto& lr = light_ranges_[i];
            if (!lr.active || lr.max_c < chunk_begin || lr.min_c >= chunk_end) {
                continue;
            }
            const auto& vl = view_lights_[i];
            bool past_chunk = false;
            for (uint32_t z = lr.z0; z <= lr.z1 && !past_chunk; ++z) {
                for (uint32_t y = lr.y0; y <= lr.y1; ++y) {
                    const std::size_t row = (static_cast<std::size_t>(z) * CLUSTER_DIM_Y + y) * CLUSTER_DIM_X;
                    if (row + lr.x0 >= chunk_end) {
                        past_chunk = true; // rows only grow with (z, y) — done with this light
                        break;
                    }
                    if (row + lr.x1 < chunk_begin) {
                        continue; // row entirely before this chunk
                    }
                    const std::size_t c_lo = std::max<std::size_t>(row + lr.x0, chunk_begin);
                    const std::size_t c_hi = std::min<std::size_t>(row + lr.x1, chunk_end - 1);
                    for (std::size_t c = c_lo; c <= c_hi; ++c) {
                        const auto& cb = cluster_bounds_[c];
                        if (!sphere_intersects_aabb(vl.view_pos, vl.radius, cb.view_min, cb.view_max)) {
                            continue;
                        }
                        auto& bin = bins_[c];
                        if (bin.size() < MAX_LIGHTS_PER_CLUSTER) {
                            bin.push_back(static_cast<uint32_t>(i));
                        } else {
                            ++truncated_local;
                        }
                    }
                }
            }
        }
        // One write per CHUNK to a chunk-indexed slot — never inside the
        // per-item loop (§7: no shared cache-line writes per item).
        if (truncated_local != 0) {
            chunk_truncated_[chunk_begin / BIN_GRAIN] = truncated_local;
        }
    };

    // Grain guard: tiny light lists (the 1-character playground) and
    // uninitialized pools run the SAME body as one inline chunk.
    if (js.is_initialized() && light_count >= BIN_MIN_PARALLEL_LIGHTS) {
        core::jobs::parallel_for(js, 0, CLUSTER_COUNT, bin_chunk, BIN_GRAIN);
    } else {
        bin_chunk(0, CLUSTER_COUNT);
    }

    // Merge partials single-threaded in chunk-index order (§6.2). Integer
    // addition commutes, but the pattern is the house contract.
    cluster_lights_dropped_ = 0;
    for (std::size_t k = 0; k < chunk_count; ++k) {
        cluster_lights_dropped_ += chunk_truncated_[k];
    }

    if (cluster_lights_dropped_ != 0 && !warned_truncation_) {
        BRACKEN_LOG_WARN(LOG_TAG,
                         "ClusterGrid: {} light insertion(s) dropped by MAX_LIGHTS_PER_CLUSTER={}; those clusters "
                         "are lit by only their first {} lights",
                         cluster_lights_dropped_, MAX_LIGHTS_PER_CLUSTER, MAX_LIGHTS_PER_CLUSTER);
        warned_truncation_ = true;
    }

    // Pack offsets + indices (serial, deterministic order across clusters),
    // detecting changes in-place — this dirty flag replaces the old byte-wise
    // FNV hash over the whole ~27KB table.
    light_indices_.clear();
    uint32_t offset = 0;
    bool offsets_changed = false;
    for (uint32_t c = 0; c < CLUSTER_COUNT; ++c) {
        const uint32_t count = static_cast<uint32_t>(bins_[c].size());
        if (cluster_offsets_[c * 2 + 0] != offset || cluster_offsets_[c * 2 + 1] != count) {
            cluster_offsets_[c * 2 + 0] = offset;
            cluster_offsets_[c * 2 + 1] = count;
            offsets_changed = true;
        }
        light_indices_.insert(light_indices_.end(), bins_[c].begin(), bins_[c].end());
        offset += count;
    }
    if (offsets_changed) {
        offsets_dirty_ = true;
    }
    offsets_all_zero_ = light_indices_.empty();

    params_.gridDim.w = static_cast<uint32_t>(headers_.size());
}

void ClusterGrid::upload(SDL_GPUCopyPass* copy_pass) {
    BRACKEN_PROFILE_ZONE_COLOR("ClusterGrid::upload", 0x27AE60);
    if (!copy_pass || !light_data_buf_) {
        return;
    }

    // Pack the light data buffer into reusable member scratch. Section offsets
    // are capacity-based; the shader derives the same offsets from the capacity
    // in ClusterParams::maxPerCluster.y (cluster_lighting.hlsli). Only the
    // transfer shrinks to the last used byte of the farthest populated section.
    // Bytes between a section's used prefix and its capacity end are stale but
    // never read: the shader bounds every access by the counts in the header.
    // A single contiguous upload is required here — multiple region updates to
    // the same buffer would cycle the destination between regions.
    const size_t headers_off = sizeof(LightDataHeader);
    const size_t points_off = headers_off + sizeof(LightHeader) * max_lights_;
    const size_t spots_off = points_off + sizeof(PointLight) * max_lights_;

    if (light_staging_.size() != light_data_buffer_size(max_lights_)) {
        light_staging_.assign(light_data_buffer_size(max_lights_), 0);
    }
    uint8_t* staging = light_staging_.data();

    LightDataHeader hdr{};
    hdr.header_count = static_cast<uint32_t>(headers_.size());
    hdr.point_count = static_cast<uint32_t>(points_.size());
    hdr.spot_count = static_cast<uint32_t>(spots_.size());
    std::memcpy(staging, &hdr, sizeof(hdr));

    size_t upload_size = sizeof(LightDataHeader);
    if (!headers_.empty()) {
        std::memcpy(staging + headers_off, headers_.data(), headers_.size() * sizeof(LightHeader));
        upload_size = headers_off + headers_.size() * sizeof(LightHeader);
    }
    if (!points_.empty()) {
        std::memcpy(staging + points_off, points_.data(), points_.size() * sizeof(PointLight));
        upload_size = points_off + points_.size() * sizeof(PointLight);
    }
    if (!spots_.empty()) {
        std::memcpy(staging + spots_off, spots_.data(), spots_.size() * sizeof(SpotLight));
        upload_size = spots_off + spots_.size() * sizeof(SpotLight);
    }

    light_data_buf_->update(copy_pass, staging, upload_size);

    // Offsets only need re-uploading when bin layout moved (build() sets the
    // dirty flag while packing the table).
    if (offsets_dirty_) {
        cluster_offsets_buf_->update(copy_pass, cluster_offsets_.data(), cluster_offsets_.size() * sizeof(uint32_t));
        offsets_dirty_ = false;
    }

    // Upload the used index prefix; with zero indices every cluster's count is 0
    // so the shader never reads stale contents.
    const size_t idx_bytes = light_indices_.size() * sizeof(uint32_t);
    if (idx_bytes > 0) {
        light_indices_buf_->update(copy_pass, light_indices_.data(), idx_bytes);
    }

    // The identity side-channel. Both are <= 1 KB and both change every frame
    // a light moves in or out, so there is no dirty flag worth the branch —
    // the whole point of putting them in this copy pass is that they cost
    // nothing next to the light data above.
    if (light_stable_ids_buf_ && !stable_ids_.empty()) {
        light_stable_ids_buf_->update(copy_pass, stable_ids_.data(), stable_ids_.size() * sizeof(uint32_t));
    }
    if (light_remap_buf_ && remap_.size() == max_lights_) {
        light_remap_buf_->update(copy_pass, remap_.data(), remap_.size() * sizeof(uint32_t));
    }
    // Pass ownership stays with the caller — do not end it here.
}

SDL_GPUBuffer* ClusterGrid::light_data_buffer() const {
    return light_data_buf_ ? light_data_buf_->handle() : nullptr;
}
SDL_GPUBuffer* ClusterGrid::cluster_offsets_buffer() const {
    return cluster_offsets_buf_ ? cluster_offsets_buf_->handle() : nullptr;
}
SDL_GPUBuffer* ClusterGrid::light_indices_buffer() const {
    return light_indices_buf_ ? light_indices_buf_->handle() : nullptr;
}
SDL_GPUBuffer* ClusterGrid::light_stable_ids_buffer() const {
    return light_stable_ids_buf_ ? light_stable_ids_buf_->handle() : nullptr;
}
SDL_GPUBuffer* ClusterGrid::light_remap_buffer() const {
    return light_remap_buf_ ? light_remap_buf_->handle() : nullptr;
}

} // namespace bracken::render::lighting
