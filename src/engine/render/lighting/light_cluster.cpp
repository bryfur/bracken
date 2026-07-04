#include "engine/render/lighting/light_cluster.hpp"

#include "engine/core/jobs/parallel_for.hpp"
#include "engine/core/profiler.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_log.h>

namespace mmo::engine::render::lighting {

namespace {

// Std140 layout sanity. PointLight: 2 vec3+float in 32B. SpotLight: 4 vec4 in 64B.
static_assert(sizeof(PointLight) == 32, "PointLight must be 32 bytes (std140)");
static_assert(sizeof(SpotLight) == 64, "SpotLight must be 64 bytes (std140)");
static_assert(sizeof(LightHeader) == 8, "LightHeader must be 8 bytes");

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

// View-space frustum corner for cluster (cx,cy,cz) at the given Z slice.
glm::vec3 unproject_corner(uint32_t cx, uint32_t cy, float view_z, const glm::mat4& invProj) {
    // Cluster (cx,cy) maps to NDC tile [-1..1] in X and Y.
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

uint64_t fnv1a64(const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t hash = 14695981039346656037ull;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

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
    light_indices_.reserve(CLUSTER_COUNT * 4);
    bins_.assign(CLUSTER_COUNT, {});
    for (auto& b : bins_) b.reserve(8);

    light_data_buf_ =
        gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::Storage, light_data_buffer_size(max_lights_));
    cluster_offsets_buf_ =
        gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::Storage, cluster_offsets_buffer_size());
    light_indices_buf_ =
        gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::Storage, light_indices_buffer_size());
    if (!light_data_buf_ || !cluster_offsets_buf_ || !light_indices_buf_) {
        SDL_Log("ClusterGrid::init: failed to allocate storage buffers");
        shutdown();
        return false;
    }
    return true;
}

void ClusterGrid::shutdown() {
    light_data_buf_.reset();
    cluster_offsets_buf_.reset();
    light_indices_buf_.reset();
    points_.clear();
    spots_.clear();
    headers_.clear();
    cluster_bounds_.clear();
    cluster_offsets_.clear();
    light_indices_.clear();
    bins_.clear();
    view_lights_.clear();
    light_staging_.clear();
    last_offsets_hash_ = 0;
    offsets_uploaded_ = false;
    device_ = nullptr;
}

void ClusterGrid::begin_frame(const scene::CameraState& camera, uint32_t screen_w, uint32_t screen_h, float near_plane,
                              float far_plane) {
    ENGINE_PROFILE_ZONE_COLOR("ClusterGrid::begin_frame", 0x27AE60);
    // Allow CPU-only callers (tests) to skip init() — lazily size scratch arrays.
    if (cluster_bounds_.size() != CLUSTER_COUNT) {
        cluster_bounds_.assign(CLUSTER_COUNT, ClusterBounds{});
    }
    if (cluster_offsets_.size() != CLUSTER_COUNT * 2) {
        cluster_offsets_.assign(CLUSTER_COUNT * 2, 0u);
    }
    if (bins_.size() != CLUSTER_COUNT) {
        bins_.assign(CLUSTER_COUNT, {});
    }

    points_.clear();
    spots_.clear();
    headers_.clear();
    light_indices_.clear();
    std::fill(cluster_offsets_.begin(), cluster_offsets_.end(), 0u);
    for (auto& bin : bins_) bin.clear();
    warned_truncation_ = false;

    params_.view = camera.view;
    params_.invProjection = glm::inverse(camera.projection);
    params_.screenSize = glm::vec4(static_cast<float>(screen_w), static_cast<float>(screen_h),
                                   screen_w > 0 ? 1.0f / static_cast<float>(screen_w) : 0.0f,
                                   screen_h > 0 ? 1.0f / static_cast<float>(screen_h) : 0.0f);
    float near_safe = std::max(near_plane, 1e-3f);
    float far_safe = std::max(far_plane, near_safe + 1e-3f);
    float log_ratio = std::log(far_safe / near_safe);
    params_.zPlanes = glm::vec4(near_safe, far_safe, log_ratio, log_ratio > 0.0f ? 1.0f / log_ratio : 0.0f);
    params_.gridDim = glm::uvec4(CLUSTER_DIM_X, CLUSTER_DIM_Y, CLUSTER_DIM_Z, 0u);
    params_.maxPerCluster = glm::uvec4(MAX_LIGHTS_PER_CLUSTER, 0u, 0u, 0u);

    // Cluster AABBs only depend on the projection. Recompute when it changes;
    // otherwise the previous frame's bounds are still valid.
    const bool projection_changed = !bounds_valid_ || params_.invProjection != last_inv_projection_ ||
                                    near_safe != last_near_ || far_safe != last_far_;
    if (projection_changed) {
        compute_cluster_bounds();
        last_inv_projection_ = params_.invProjection;
        last_near_ = near_safe;
        last_far_ = far_safe;
        bounds_valid_ = true;
    }
}

void ClusterGrid::add_point_light(const PointLight& l) {
    if (points_.size() + spots_.size() >= max_lights_) {
        return;
    }
    uint32_t payload_idx = static_cast<uint32_t>(points_.size());
    points_.push_back(l);
    headers_.push_back({LIGHT_TYPE_POINT, payload_idx});
}

void ClusterGrid::add_spot_light(const SpotLight& l) {
    if (points_.size() + spots_.size() >= max_lights_) {
        return;
    }
    uint32_t payload_idx = static_cast<uint32_t>(spots_.size());
    spots_.push_back(l);
    headers_.push_back({LIGHT_TYPE_SPOT, payload_idx});
}

void ClusterGrid::compute_cluster_bounds() {
    ENGINE_PROFILE_ZONE_COLOR("ClusterGrid::compute_cluster_bounds", 0x27AE60);
    const float near_z = -params_.zPlanes.x;
    const float far_z = -params_.zPlanes.y;
    const glm::mat4& invProj = params_.invProjection;

    for (uint32_t z = 0; z < CLUSTER_DIM_Z; ++z) {
        // View-space depths are negative (camera looks down -Z).
        float zn = -exp_z_slice(z, -near_z, -far_z);
        float zf = -exp_z_slice(z + 1, -near_z, -far_z);
        // Ensure ordering: zn closer to camera than zf, both negative.
        if (zn < zf) {
            std::swap(zn, zf);
        }

        for (uint32_t y = 0; y < CLUSTER_DIM_Y; ++y) {
            for (uint32_t x = 0; x < CLUSTER_DIM_X; ++x) {
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
                uint32_t idx = (z * CLUSTER_DIM_Y + y) * CLUSTER_DIM_X + x;
                cluster_bounds_[idx] = {mn, mx};
            }
        }
    }
}

bool ClusterGrid::sphere_intersects_aabb(const glm::vec3& center, float radius, const glm::vec3& aabb_min,
                                         const glm::vec3& aabb_max) {
    glm::vec3 closest = glm::clamp(center, aabb_min, aabb_max);
    glm::vec3 d = center - closest;
    return glm::dot(d, d) <= radius * radius;
}

void ClusterGrid::build() {
    ENGINE_PROFILE_ZONE_COLOR("ClusterGrid::build", 0x27AE60);
    const glm::mat4& view = params_.view;

    const std::size_t light_count = headers_.size();
    if (light_count == 0) {
        // No dynamic lights: bins are already empty from begin_frame(). Skip the
        // 3456-cluster parallel walk entirely — it would just be empty inner loops
        // but the dispatch overhead is real.
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

    std::atomic<bool> truncated{false};

    // Parallelize over clusters. Each cluster owns its bins_[c] vector exclusively.
    core::jobs::parallel_for(0u, static_cast<std::size_t>(CLUSTER_COUNT), [&](std::size_t begin, std::size_t end) {
        for (std::size_t c = begin; c < end; ++c) {
            const auto& cb = cluster_bounds_[c];
            auto& bin = bins_[c];
            for (std::size_t i = 0; i < light_count; ++i) {
                const auto& vl = view_lights_[i];
                if (vl.radius <= 0.0f) {
                    continue;
                }
                if (sphere_intersects_aabb(vl.view_pos, vl.radius, cb.view_min, cb.view_max)) {
                    if (bin.size() < MAX_LIGHTS_PER_CLUSTER) {
                        bin.push_back(static_cast<uint32_t>(i));
                    } else {
                        truncated.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
            }
        }
    });

    if (truncated.load(std::memory_order_relaxed) && !warned_truncation_) {
        SDL_Log("ClusterGrid: at least one cluster exceeded MAX_LIGHTS_PER_CLUSTER=%u, truncating",
                MAX_LIGHTS_PER_CLUSTER);
        warned_truncation_ = true;
    }

    // Pack offsets + indices (serial, deterministic order across clusters).
    light_indices_.clear();
    uint32_t offset = 0;
    for (uint32_t c = 0; c < CLUSTER_COUNT; ++c) {
        uint32_t count = static_cast<uint32_t>(bins_[c].size());
        cluster_offsets_[c * 2 + 0] = offset;
        cluster_offsets_[c * 2 + 1] = count;
        light_indices_.insert(light_indices_.end(), bins_[c].begin(), bins_[c].end());
        offset += count;
    }

    params_.gridDim.w = static_cast<uint32_t>(headers_.size());
}

void ClusterGrid::upload(SDL_GPUCommandBuffer* cb, memory::Arena* /*frame_arena*/) {
    if (!cb || !light_data_buf_) {
        return;
    }

    // One copy pass shared by all cluster buffers.
    SDL_GPUCopyPass* copy_pass = SDL_BeginGPUCopyPass(cb);
    if (!copy_pass) {
        SDL_Log("ClusterGrid::upload: failed to begin copy pass: %s", SDL_GetError());
        return;
    }
    upload(copy_pass);
    SDL_EndGPUCopyPass(copy_pass);
}

void ClusterGrid::upload(SDL_GPUCopyPass* copy_pass) {
    ENGINE_PROFILE_ZONE_COLOR("ClusterGrid::upload", 0x27AE60);
    if (!copy_pass || !light_data_buf_) {
        return;
    }

    // Pack the light data buffer into reusable member scratch. Section offsets
    // stay capacity-based (GPU layout unchanged — shaders untouched); only the
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

    // Offsets only need re-uploading when bin layout moved.
    const uint64_t offsets_hash = fnv1a64(cluster_offsets_.data(), cluster_offsets_.size() * sizeof(uint32_t));
    const bool offsets_dirty = !offsets_uploaded_ || offsets_hash != last_offsets_hash_;

    light_data_buf_->update(copy_pass, staging, upload_size);

    if (offsets_dirty) {
        cluster_offsets_buf_->update(copy_pass, cluster_offsets_.data(), cluster_offsets_.size() * sizeof(uint32_t));
        last_offsets_hash_ = offsets_hash;
        offsets_uploaded_ = true;
    }

    // Upload the used index prefix; with zero indices every cluster's count is 0
    // so the shader never reads stale contents.
    const size_t idx_bytes = light_indices_.size() * sizeof(uint32_t);
    if (idx_bytes > 0) {
        light_indices_buf_->update(copy_pass, light_indices_.data(), idx_bytes);
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

} // namespace mmo::engine::render::lighting
