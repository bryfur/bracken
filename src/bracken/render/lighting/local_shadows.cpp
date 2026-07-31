#include "bracken/render/lighting/local_shadows.hpp"

#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>

namespace bracken::render::lighting {

namespace {
constexpr const char* LOG_TAG = "shadows";

// GPU mirror layout sanity (cluster_lighting.hlsli LocalShadowSlot).
static_assert(sizeof(LocalShadowSlotGPU) == 96, "LocalShadowSlotGPU must be 96 bytes");

constexpr uint8_t POINT_FULL_MASK = 0x3F; // 6 faces
constexpr uint8_t SPOT_FULL_MASK = 0x01;  // 1 tile

// Cube-face bases. Faces look along their axis from the light; the up
// vectors are arbitrary but fixed (the shader derives UVs from view_proj
// directly, so only CPU/GPU face SELECTION must agree — see
// local_shadow_cube_face and localShadowCubeFace).
glm::vec3 face_forward(int face) {
    switch (face) {
        case 0: return {1.0f, 0.0f, 0.0f};
        case 1: return {-1.0f, 0.0f, 0.0f};
        case 2: return {0.0f, 1.0f, 0.0f};
        case 3: return {0.0f, -1.0f, 0.0f};
        case 4: return {0.0f, 0.0f, 1.0f};
        default: return {0.0f, 0.0f, -1.0f};
    }
}

glm::vec3 face_up(int face) {
    // +Y/-Y faces need a non-parallel up.
    return (face == 2 || face == 3) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
}

// Point faces render slightly wider than 90 deg so the 2-texel PCF border at
// every face seam still holds valid depth (tan(fov/2) scaled by
// res/(res-4) — exact 90 deg frusta share edges with zero overlap).
float point_face_fovy(int tile_resolution) {
    const float scale =
        tile_resolution > 4 ? static_cast<float>(tile_resolution) / static_cast<float>(tile_resolution - 4) : 1.0f;
    return 2.0f * std::atan(scale);
}

// Spot maps cover the outer cone widened ~10% so the smoothstep cone edge
// never reads outside the map; clamped away from degenerate extremes.
float spot_fovy(float outer_cos) {
    const float outer = std::acos(std::clamp(outer_cos, -1.0f, 1.0f));
    return std::clamp(2.0f * outer * 1.1f, glm::radians(5.0f), glm::radians(170.0f));
}

// Per-light-type plumbing for the shared assignment algorithm.
template <typename LightT>
struct LightTraits;

template <>
struct LightTraits<PointLight> {
    static constexpr uint8_t FULL = POINT_FULL_MASK;
    static PointLight& stored(auto& g) { return g.point; }
    static const PointLight& stored(const auto& g) { return g.point; }
    // Fields that change the rendered shadow map (color/intensity do not).
    static bool shadow_relevant_changed(const PointLight& a, const PointLight& b) {
        return a.position != b.position || a.radius != b.radius;
    }
};

template <>
struct LightTraits<SpotLight> {
    static constexpr uint8_t FULL = SPOT_FULL_MASK;
    static SpotLight& stored(auto& g) { return g.spot; }
    static const SpotLight& stored(const auto& g) { return g.spot; }
    static bool shadow_relevant_changed(const SpotLight& a, const SpotLight& b) {
        return a.position != b.position || a.radius != b.radius || a.direction != b.direction ||
               a.outer_cos != b.outer_cos;
    }
};

} // namespace

// ============================================================================
// Pure helpers
// ============================================================================

int local_shadow_cube_face(const glm::vec3& d) {
    const glm::vec3 ad = glm::abs(d);
    if (ad.x >= ad.y && ad.x >= ad.z) {
        return d.x > 0.0f ? 0 : 1;
    }
    if (ad.y >= ad.z) {
        return d.y > 0.0f ? 2 : 3;
    }
    return d.z > 0.0f ? 4 : 5;
}

float local_shadow_light_score(float intensity, float radius, float distance) {
    return intensity * radius / std::max(distance, 1.0f);
}

float local_shadow_near_plane(float radius) {
    return std::clamp(0.02f * radius, 1.0f, 20.0f);
}

glm::vec4 local_shadow_tile_rect(int tile) {
    const int col = tile % LOCAL_SHADOW_TILES_PER_ROW;
    const int row = tile / LOCAL_SHADOW_TILES_PER_ROW;
    const float scale = 1.0f / static_cast<float>(LOCAL_SHADOW_TILES_PER_ROW);
    return {static_cast<float>(col) * scale, static_cast<float>(row) * scale, scale, scale};
}

glm::mat4 local_point_face_view_proj(const glm::vec3& light_pos, int face, float near_plane, float far_plane,
                                     int tile_resolution) {
    // Standard-Z RH_ZO perspective: near -> 0, far -> 1 (the scene's
    // reversed-Z comes from swapped near/far, which we deliberately do NOT do
    // here — see the class comment in local_shadows.hpp).
    const glm::mat4 proj = glm::perspectiveRH_ZO(point_face_fovy(tile_resolution), 1.0f,
                                                 std::min(near_plane, far_plane * 0.5f), far_plane);
    const glm::mat4 view = glm::lookAtRH(light_pos, light_pos + face_forward(face), face_up(face));
    return proj * view;
}

glm::mat4 local_spot_view_proj(const glm::vec3& pos, const glm::vec3& dir, float outer_cos, float near_plane,
                               float far_plane) {
    const glm::mat4 proj =
        glm::perspectiveRH_ZO(spot_fovy(outer_cos), 1.0f, std::min(near_plane, far_plane * 0.5f), far_plane);
    glm::vec3 forward = dir;
    const float len = glm::length(forward);
    forward = len > 1e-5f ? forward / len : glm::vec3(0.0f, -1.0f, 0.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    if (std::abs(glm::dot(forward, up)) > 0.99f) {
        up = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    const glm::mat4 view = glm::lookAtRH(pos, pos + forward, up);
    return proj * view;
}

// ============================================================================
// LocalShadowSelector
// ============================================================================

template <typename LightT>
void LocalShadowSelector::assign_groups(std::span<Group> groups, int budget, std::vector<LightT>& lights,
                                        const glm::vec3& camera_pos,
                                        std::span<const LocalShadowDynamicCaster> dynamic_casters, float hysteresis) {
    using T = LightTraits<LightT>;

    // Groups beyond the active budget are dropped (budget setting shrank).
    for (size_t g = static_cast<size_t>(budget); g < groups.size(); ++g) {
        groups[g] = Group{};
    }

    const int light_count = static_cast<int>(lights.size());
    auto candidate_ok = [](const LightT& l) { return l.casts_shadows && l.radius > 0.0f && l.intensity > 0.0f; };

    score_scratch_.assign(static_cast<size_t>(light_count), 0.0f);
    for (int i = 0; i < light_count; ++i) {
        if (candidate_ok(lights[i])) {
            score_scratch_[i] = local_shadow_light_score(lights[i].intensity, lights[i].radius,
                                                         glm::length(lights[i].position - camera_pos));
        }
    }

    // A) Match incumbents to this frame's lights by nearest position of the
    // same type (immediate-mode scenes carry no light identity, so a light
    // that resubmits in roughly the same place IS the same light; a
    // teleported light intentionally reads as a brand-new one). Fixed group
    // order keeps this deterministic.
    claimed_scratch_.assign(static_cast<size_t>(light_count), 0);
    for (int g = 0; g < budget; ++g) {
        Group& grp = groups[g];
        grp.light_index = -1;
        if (!grp.used) {
            continue;
        }
        int best = -1;
        float best_d2 = std::numeric_limits<float>::max();
        for (int i = 0; i < light_count; ++i) {
            if (claimed_scratch_[i] != 0 || !candidate_ok(lights[i])) {
                continue;
            }
            const glm::vec3 d = lights[i].position - T::stored(grp).position;
            const float d2 = glm::dot(d, d);
            if (d2 < best_d2) {
                best_d2 = d2;
                best = i;
            }
        }
        bool matched = false;
        if (best >= 0) {
            const float thresh = 0.5f * std::max(T::stored(grp).radius, lights[best].radius);
            matched = best_d2 <= thresh * thresh;
        }
        if (matched) {
            claimed_scratch_[best] = 1;
            grp.light_index = best;
            grp.score = score_scratch_[best];
        } else {
            grp = Group{}; // no continuation for this light: free the tiles
        }
    }

    // B) Remaining candidates ordered by (score desc, submission index asc).
    order_scratch_.clear();
    for (int i = 0; i < light_count; ++i) {
        if (claimed_scratch_[i] == 0 && candidate_ok(lights[i])) {
            order_scratch_.push_back(i);
        }
    }
    std::sort(order_scratch_.begin(), order_scratch_.end(), [&](int a, int b) {
        if (score_scratch_[a] != score_scratch_[b]) {
            return score_scratch_[a] > score_scratch_[b];
        }
        return a < b;
    });

    // C) Fill free groups (lowest group index first).
    size_t next_cand = 0;
    for (int g = 0; g < budget && next_cand < order_scratch_.size(); ++g) {
        Group& grp = groups[g];
        if (grp.used) {
            continue;
        }
        const int i = order_scratch_[next_cand++];
        grp = Group{};
        grp.used = true;
        grp.light_index = i;
        grp.score = score_scratch_[i];
        grp.dirty_mask = T::FULL; // fresh assignment: every tile must render
    }

    // D) Eviction with hysteresis: a challenger must beat the weakest
    // incumbent's score by the margin, so lights hovering at the boundary
    // don't thrash slots (each swap costs a full re-render).
    for (; next_cand < order_scratch_.size(); ++next_cand) {
        const int i = order_scratch_[next_cand];
        int weakest = -1;
        float weakest_score = std::numeric_limits<float>::max();
        for (int g = 0; g < budget; ++g) {
            if (groups[g].used && groups[g].score < weakest_score) {
                weakest_score = groups[g].score;
                weakest = g;
            }
        }
        if (weakest < 0 || score_scratch_[i] <= weakest_score * (1.0f + hysteresis)) {
            break; // candidates are sorted desc: nobody later can evict either
        }
        Group& grp = groups[weakest];
        grp = Group{};
        grp.used = true;
        grp.light_index = i;
        grp.score = score_scratch_[i];
        grp.dirty_mask = T::FULL;
    }

    // E) Dirty analysis: the tile content is stale when the light's
    // shadow-relevant state changed since the snapshot, or when a dynamic
    // caster (moving/skinned) is inside the light's sphere. Snapshot then
    // refreshed for next frame's matching.
    for (int g = 0; g < budget; ++g) {
        Group& grp = groups[g];
        if (!grp.used || grp.light_index < 0) {
            continue;
        }
        const LightT& cur = lights[grp.light_index];
        if (T::shadow_relevant_changed(cur, T::stored(grp))) {
            grp.dirty_mask = T::FULL;
        } else {
            for (const auto& caster : dynamic_casters) {
                const float r = caster.radius + cur.radius;
                const glm::vec3 d = caster.center - cur.position;
                if (glm::dot(d, d) <= r * r) {
                    grp.dirty_mask = T::FULL;
                    break;
                }
            }
        }
        T::stored(grp) = cur;
    }
}

void LocalShadowSelector::schedule_and_publish(std::vector<PointLight>& points, std::vector<SpotLight>& spots,
                                               int point_budget, int spot_budget, const LocalShadowConfig& config,
                                               int tile_resolution) {
    const float atlas_texel =
        1.0f / static_cast<float>(std::max(tile_resolution, 1) * LOCAL_SHADOW_TILES_PER_ROW);
    const float uv_inset = 2.0f / static_cast<float>(std::max(tile_resolution, 1));
    const float face_texel_per_unit =
        2.0f * std::tan(point_face_fovy(tile_resolution) * 0.5f) / static_cast<float>(std::max(tile_resolution, 1));

    auto render_point_face = [&](int group_index, int face) {
        Group& grp = point_groups_[static_cast<size_t>(group_index)];
        const PointLight& light = grp.point;
        const int tile = group_index * 6 + face;
        const float near_plane = local_shadow_near_plane(light.radius);
        LocalShadowSlotGPU& slot = slot_data_[static_cast<size_t>(tile)];
        slot.view_proj = local_point_face_view_proj(light.position, face, near_plane, light.radius, tile_resolution);
        slot.rect = local_shadow_tile_rect(tile);
        slot.params = {face_texel_per_unit, uv_inset, near_plane, atlas_texel};
        requests_.push_back({tile, slot.view_proj, light.position, light.radius});
    };
    auto render_spot_tile = [&](int group_index) {
        Group& grp = spot_groups_[static_cast<size_t>(group_index)];
        const SpotLight& light = grp.spot;
        const int tile = LOCAL_SHADOW_SPOT_TILE_BASE + group_index;
        const float near_plane = local_shadow_near_plane(light.radius);
        LocalShadowSlotGPU& slot = slot_data_[static_cast<size_t>(tile)];
        slot.view_proj = local_spot_view_proj(light.position, light.direction, light.outer_cos, near_plane,
                                              light.radius);
        slot.rect = local_shadow_tile_rect(tile);
        const float texel_per_unit =
            2.0f * std::tan(spot_fovy(light.outer_cos) * 0.5f) / static_cast<float>(std::max(tile_resolution, 1));
        slot.params = {texel_per_unit, uv_inset, near_plane, atlas_texel};
        requests_.push_back({tile, slot.view_proj, light.position, light.radius});
    };

    // Ring of active groups: point groups first, then spots. The persistent
    // cursor makes the tile-update cap round-robin fair — a starved group
    // catches up on later frames instead of never rendering.
    const int ring_size = point_budget + spot_budget;
    int budget_left = std::max(config.max_tile_updates_per_frame, 0);
    int last_serviced = -1;
    if (ring_size > 0) {
        ring_cursor_ %= ring_size;

        // Two passes: fresh assignments first (they are unshadowed until every
        // tile holds content, so they render all-or-nothing), then stale
        // updates (partial face updates are fine — each tile's slot matrix is
        // written only when the tile renders, so content and matrix always
        // agree even mid-update).
        for (const bool fresh_pass : {true, false}) {
            for (int k = 0; k < ring_size && budget_left > 0; ++k) {
                const int r = (ring_cursor_ + k) % ring_size;
                const bool is_spot = r >= point_budget;
                Group& grp = is_spot ? spot_groups_[static_cast<size_t>(r - point_budget)]
                                     : point_groups_[static_cast<size_t>(r)];
                if (!grp.used) {
                    continue;
                }
                const uint8_t full = is_spot ? SPOT_FULL_MASK : POINT_FULL_MASK;
                const bool incomplete = (grp.valid_mask & full) != full;
                if (fresh_pass != incomplete) {
                    continue;
                }
                uint8_t need = static_cast<uint8_t>(grp.dirty_mask & full);
                if (incomplete) {
                    need |= static_cast<uint8_t>(full & ~grp.valid_mask);
                }
                if (need == 0) {
                    continue;
                }
                if (incomplete && std::popcount(static_cast<unsigned>(need)) > budget_left) {
                    continue; // all-or-nothing for fresh assignments
                }
                const int faces = is_spot ? 1 : 6;
                for (int f = 0; f < faces && budget_left > 0; ++f) {
                    const uint8_t bit = static_cast<uint8_t>(1u << f);
                    if ((need & bit) == 0) {
                        continue;
                    }
                    if (is_spot) {
                        render_spot_tile(r - point_budget);
                    } else {
                        render_point_face(r, f);
                    }
                    grp.dirty_mask &= static_cast<uint8_t>(~bit);
                    grp.valid_mask |= bit;
                    slot_data_dirty_ = true;
                    ++tiles_rendered_total_;
                    --budget_left;
                    last_serviced = r;
                }
            }
        }
        if (last_serviced >= 0) {
            ring_cursor_ = (last_serviced + 1) % ring_size;
        }
    }

    // Publish shadow slots: only groups whose every tile holds content for
    // the current assignment are sampleable (tiles scheduled above render
    // before the forward pass, so same-frame publication is safe).
    shadowed_points_ = 0;
    shadowed_spots_ = 0;
    for (int g = 0; g < point_budget; ++g) {
        const Group& grp = point_groups_[static_cast<size_t>(g)];
        if (grp.used && grp.light_index >= 0 && (grp.valid_mask & POINT_FULL_MASK) == POINT_FULL_MASK) {
            points[static_cast<size_t>(grp.light_index)].shadow_slot = g * 6;
            ++shadowed_points_;
        }
    }
    for (int g = 0; g < spot_budget; ++g) {
        const Group& grp = spot_groups_[static_cast<size_t>(g)];
        if (grp.used && grp.light_index >= 0 && (grp.valid_mask & SPOT_FULL_MASK) == SPOT_FULL_MASK) {
            spots[static_cast<size_t>(grp.light_index)].shadow_slot = LOCAL_SHADOW_SPOT_TILE_BASE + g;
            ++shadowed_spots_;
        }
    }
}

void LocalShadowSelector::update(std::vector<PointLight>& points, std::vector<SpotLight>& spots,
                                 const glm::vec3& camera_pos,
                                 std::span<const LocalShadowDynamicCaster> dynamic_casters,
                                 const LocalShadowConfig& config, int tile_resolution) {
    BRACKEN_PROFILE_ZONE_COLOR("LocalShadowSelector::update", 0xE67E22);
    requests_.clear();
    shadowed_points_ = 0;
    shadowed_spots_ = 0;

    const int point_budget = std::clamp(config.point_budget, 0, LOCAL_SHADOW_MAX_POINT_BUDGET);
    const int spot_budget = std::clamp(config.spot_budget, 0, LOCAL_SHADOW_MAX_SPOT_BUDGET);

    for (auto& l : points) {
        l.shadow_slot = -1;
    }
    for (auto& l : spots) {
        l.shadow_slot = -1;
    }

    assign_groups<PointLight>(point_groups_, point_budget, points, camera_pos, dynamic_casters, config.hysteresis);
    assign_groups<SpotLight>(spot_groups_, spot_budget, spots, camera_pos, dynamic_casters, config.hysteresis);
    schedule_and_publish(points, spots, point_budget, spot_budget, config, tile_resolution);
}

void LocalShadowSelector::invalidate_all_tiles() {
    for (auto& grp : point_groups_) {
        grp.valid_mask = 0;
        if (grp.used) {
            grp.dirty_mask = POINT_FULL_MASK;
        }
    }
    for (auto& grp : spot_groups_) {
        grp.valid_mask = 0;
        if (grp.used) {
            grp.dirty_mask = SPOT_FULL_MASK;
        }
    }
    requests_.clear();
}

// ============================================================================
// LocalShadowMap
// ============================================================================

LocalShadowMap::LocalShadowMap() = default;

LocalShadowMap::~LocalShadowMap() {
    shutdown();
}

bool LocalShadowMap::init(gpu::GPUDevice& device, int atlas_resolution) {
    device_ = &device;
    atlas_resolution_ = atlas_resolution;
    atlas_initialized_ = false;

    atlas_texture_ = gpu::GPUTexture::create_depth(device, atlas_resolution_, atlas_resolution_);
    if (!atlas_texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "LocalShadowMap::init: failed to create {}x{} shadow atlas", atlas_resolution_,
                          atlas_resolution_);
        shutdown();
        return false;
    }

    // LINEAR comparison sampler (LESS_OR_EQUAL): each SampleCmpLevelZero tap
    // is a free 2x2 hardware PCF — same convention as the CSM atlas.
    SDL_GPUSamplerCreateInfo compare_info = {};
    compare_info.min_filter = SDL_GPU_FILTER_LINEAR;
    compare_info.mag_filter = SDL_GPU_FILTER_LINEAR;
    compare_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    compare_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    compare_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    compare_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    compare_info.enable_compare = true;
    compare_info.compare_op = SDL_GPU_COMPAREOP_LESS_OR_EQUAL;
    compare_sampler_ = device.create_sampler(compare_info);
    if (!compare_sampler_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "LocalShadowMap::init: failed to create comparison sampler");
        shutdown();
        return false;
    }

    slot_buffer_ = gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::Storage,
                                                  sizeof(LocalShadowSlotGPU) * LOCAL_SHADOW_MAX_SLOTS);
    if (!slot_buffer_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "LocalShadowMap::init: failed to create slot-data storage buffer");
        shutdown();
        return false;
    }

    BRACKEN_LOG_INFO(LOG_TAG, "LocalShadowMap: initialized {}x{} atlas ({} per tile, {}x{} tiles)", atlas_resolution_,
                     atlas_resolution_, tile_resolution(), LOCAL_SHADOW_TILES_PER_ROW, LOCAL_SHADOW_TILES_PER_ROW);
    return true;
}

void LocalShadowMap::shutdown() {
    if (device_ && compare_sampler_) {
        device_->release_sampler(compare_sampler_);
        compare_sampler_ = nullptr;
    }
    atlas_texture_.reset();
    slot_buffer_.reset();
    current_pass_ = nullptr;
    device_ = nullptr;
}

bool LocalShadowMap::reinit(int atlas_resolution) {
    if (!device_) {
        return false;
    }
    if (atlas_resolution == atlas_resolution_) {
        return true;
    }
    atlas_texture_.reset();
    atlas_resolution_ = atlas_resolution;
    atlas_initialized_ = false;
    atlas_texture_ = gpu::GPUTexture::create_depth(*device_, atlas_resolution_, atlas_resolution_);
    if (!atlas_texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "LocalShadowMap::reinit: failed to create {}x{} shadow atlas", atlas_resolution_,
                          atlas_resolution_);
        return false;
    }
    // Every cached tile just died with the old texture.
    selector_.invalidate_all_tiles();
    BRACKEN_LOG_INFO(LOG_TAG, "LocalShadowMap: reinitialized {}x{} atlas ({} per tile)", atlas_resolution_,
                     atlas_resolution_, tile_resolution());
    return true;
}

bool LocalShadowMap::is_ready() const {
    return atlas_texture_ != nullptr && compare_sampler_ != nullptr && slot_buffer_ != nullptr;
}

void LocalShadowMap::update(std::vector<PointLight>& points, std::vector<SpotLight>& spots,
                            const glm::vec3& camera_pos,
                            std::span<const LocalShadowDynamicCaster> dynamic_casters,
                            const LocalShadowConfig& config) {
    selector_.update(points, spots, camera_pos, dynamic_casters, config, tile_resolution());
}

void LocalShadowMap::upload(SDL_GPUCopyPass* copy_pass) {
    if (!copy_pass || !slot_buffer_ || !selector_.slot_data_dirty()) {
        return;
    }
    slot_buffer_->update(copy_pass, selector_.slot_data().data(),
                         sizeof(LocalShadowSlotGPU) * LOCAL_SHADOW_MAX_SLOTS);
    selector_.clear_slot_data_dirty();
}

SDL_GPURenderPass* LocalShadowMap::begin_pass(SDL_GPUCommandBuffer* cmd) {
    if (!cmd || !atlas_texture_) {
        return nullptr;
    }
    SDL_GPUDepthStencilTargetInfo depth_target = {};
    depth_target.texture = atlas_texture_->handle();
    // Cached tiles persist across frames: LOAD previous contents (the very
    // first pass clears the whole atlas instead — its contents are undefined
    // until then, but no tile is sampled before it has rendered). Dirty tiles
    // are cleared individually via the LocalShadowClear fullscreen draw.
    depth_target.load_op = atlas_initialized_ ? SDL_GPU_LOADOP_LOAD : SDL_GPU_LOADOP_CLEAR;
    depth_target.store_op = SDL_GPU_STOREOP_STORE;
    depth_target.clear_depth = 1.0f; // standard-Z: 1.0 is the far plane
    depth_target.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
    depth_target.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;

    current_pass_ = SDL_BeginGPURenderPass(cmd, nullptr, 0, &depth_target);
    if (!current_pass_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "LocalShadowMap::begin_pass: failed to begin atlas render pass");
        return nullptr;
    }
    atlas_initialized_ = true;
    return current_pass_;
}

void LocalShadowMap::set_tile_viewport(int tile, bool clear_depth_range) {
    if (!current_pass_ || tile < 0 || tile >= LOCAL_SHADOW_MAX_SLOTS) {
        return;
    }
    const int res = tile_resolution();
    const int x = (tile % LOCAL_SHADOW_TILES_PER_ROW) * res;
    const int y = (tile / LOCAL_SHADOW_TILES_PER_ROW) * res;

    SDL_GPUViewport viewport = {};
    viewport.x = static_cast<float>(x);
    viewport.y = static_cast<float>(y);
    viewport.w = static_cast<float>(res);
    viewport.h = static_cast<float>(res);
    // The clear draw reuses the fullscreen triangle (z = 0); pinning the
    // viewport depth range to [1,1] maps it to far-plane depth without a
    // dedicated shader.
    viewport.min_depth = clear_depth_range ? 1.0f : 0.0f;
    viewport.max_depth = 1.0f;
    SDL_SetGPUViewport(current_pass_, &viewport);

    SDL_Rect scissor = {x, y, res, res};
    SDL_SetGPUScissor(current_pass_, &scissor);
}

void LocalShadowMap::end_pass() {
    if (current_pass_) {
        SDL_EndGPURenderPass(current_pass_);
        current_pass_ = nullptr;
    }
}

SDL_GPUTexture* LocalShadowMap::atlas_texture() const {
    return atlas_texture_ ? atlas_texture_->handle() : nullptr;
}

SDL_GPUBuffer* LocalShadowMap::slot_buffer() const {
    return slot_buffer_ ? slot_buffer_->handle() : nullptr;
}

} // namespace bracken::render::lighting
