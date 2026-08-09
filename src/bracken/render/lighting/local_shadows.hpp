#pragma once

#include "bracken/contracts/frustum.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/render/lighting/light.hpp"
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <SDL3/SDL_gpu.h>
#include <span>
#include <vector>

namespace bracken::render::lighting {

namespace gpu = bracken::gpu;

// ============================================================================
// Budgeted local (point/spot) light shadows.
//
// One D32 atlas tiled LOCAL_SHADOW_TILES_PER_ROW^2 (8x8). Point lights render
// 6 perspective faces (~90 deg FOV, slightly widened so PCF taps at face
// seams stay inside the map) into 6 consecutive tiles; spot lights render one
// tile at their cone angle. All local shadow maps are STANDARD-Z perspective
// depth (near -> 0, far = light radius -> 1, LESS_OR_EQUAL compare, clear
// 1.0) — deliberately NOT the scene's reversed-Z, matching the CSM cascades'
// standard-Z convention so the caster pipelines (rasterizer bias, depth
// clamp) are shared verbatim.
//
// Per frame, LocalShadowSelector picks up to a budgeted number of
// shadow-casting lights (host opt-in: casts_shadows) by a deterministic
// priority score with eviction hysteresis, assigns atlas tiles, and decides
// which tiles actually re-render (change detection + a per-frame tile-update
// cap, round-robin at BOTH granularities: which group is serviced first, and
// which of a point group's six faces it starts on — so the worst case
// degrades to a stale tile, never a hitch and never a tile that stops
// re-rendering altogether). LocalShadowMap wraps the selector with the GPU
// resources (atlas, comparison sampler, per-tile slot SSBO). The receiver
// side lives in cluster_lighting.hlsli (LocalShadowSlot mirror + sampling).
// ============================================================================

// ---- pure helpers (CPU mirrors of cluster_lighting.hlsli; unit-tested) ----

/// Cube-face index (0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z) from the light->point
/// vector. Mirrors localShadowCubeFace() in cluster_lighting.hlsli.
[[nodiscard]] int local_shadow_cube_face(const glm::vec3& light_to_point);

/// Bitmask of cube faces (bit i = face i, matching local_shadow_cube_face)
/// whose tile a caster sphere can appear in. `to_caster` is
/// caster_center - light_position. CONSERVATIVE: never drops a face the
/// sphere touches (the tile's FOV widening is folded in); may report an extra
/// face near a frustum corner, because the test is the intersection of the
/// pyramid's four side half-spaces rather than the pyramid itself. Returns
/// POINT_FULL_MASK when the light is inside the caster.
[[nodiscard]] uint8_t local_shadow_caster_face_mask(const glm::vec3& to_caster, float caster_radius,
                                                    int tile_resolution);

/// Priority score for shadow selection: intensity * radius / distance
/// (distance clamped to >= 1 so lights at the camera don't explode).
[[nodiscard]] float local_shadow_light_score(float intensity, float radius, float distance);

// An out-of-view light contributes no pixels (the cluster binner drops it), so
// it must not hold a shadow slot against a visible one. A soft factor on the
// score rather than a cull: a light grazing the frustum edge should not pop.
// The test sphere is DILATED so the 1.0 <-> k_offscreen_score_scale flip
// happens where the light no longer matters — the flip is 4x, far outside the
// 25% eviction hysteresis, and each flip costs a full 6-tile re-render.
inline constexpr float k_offscreen_score_scale = 0.25f;
inline constexpr float k_offscreen_test_dilation = 1.25f;

/// Near plane for a local shadow map, derived from the light radius.
[[nodiscard]] float local_shadow_near_plane(float radius);

/// Atlas-UV rect for a tile: (origin.x, origin.y, scale.x, scale.y).
[[nodiscard]] glm::vec4 local_shadow_tile_rect(int tile);

/// View-projection of one point-light cube face. Standard-Z perspective,
/// FOV slightly wider than 90 deg (border texels) so face seams overlap.
[[nodiscard]] glm::mat4 local_point_face_view_proj(const glm::vec3& light_pos, int face, float near_plane,
                                                   float far_plane, int tile_resolution);

/// View-projection of a spot light's shadow map. Standard-Z perspective;
/// FOV = the outer cone angle widened ~10% so the smoothstep edge is covered.
[[nodiscard]] glm::mat4 local_spot_view_proj(const glm::vec3& pos, const glm::vec3& dir, float outer_cos,
                                             float near_plane, float far_plane);

/// Per-tile slot data uploaded as StructuredBuffer<LocalShadowSlot>
/// (cluster_lighting.hlsli mirror — byte-for-byte, 96 B).
struct alignas(16) LocalShadowSlotGPU {
    glm::mat4 view_proj{1.0f}; // world -> light clip (standard-Z perspective)
    glm::vec4 rect{0.0f};      // atlas UV: xy = tile origin, zw = tile scale
    glm::vec4 params{0.0f};    // x = world texel per unit distance, y = tile-UV
                               // PCF clamp inset, z = near plane, w = atlas texel
};

struct LocalShadowConfig {
    int point_budget = 4; // clamped to [0, LOCAL_SHADOW_MAX_POINT_BUDGET]
    int spot_budget = 2;  // clamped to [0, LOCAL_SHADOW_MAX_SPOT_BUDGET]
    // Per-frame cap on re-rendered tiles (a point light re-render = 6 tiles).
    int max_tile_updates_per_frame = 6;
    // A challenger must beat an incumbent's score by this fraction to evict
    // it — keeps lights from popping in/out at the selection boundary.
    float hysteresis = 0.25f;
};

/// A moving caster this frame (motion_id != 0 model command or any skinned
/// command): any cached tile whose light sphere it touches must re-render.
struct LocalShadowDynamicCaster {
    glm::vec3 center;
    float radius;
};

/// One tile the renderer must draw casters into this frame.
struct LocalShadowTileRequest {
    int tile = -1;
    glm::mat4 view_proj{1.0f};
    glm::vec3 light_pos{0.0f};
    float light_radius = 0.0f;
};

/**
 * CPU-only selection + caching state (no GPU resources — fully unit-testable).
 *
 * update() is deterministic: candidates sort by (score desc, submission index
 * asc); incumbents are matched to this frame's lights by nearest position of
 * the same type (lights are immediate-mode, so continuity is positional);
 * slot iteration for the tile-update cap is a fixed ring walked round-robin
 * from a persistent cursor, and each point group walks its six faces from a
 * persistent cursor of its own.
 */
class LocalShadowSelector {
public:
    /// Runs selection/assignment/caching for one frame and writes each
    /// assigned light's shadow_slot (base tile; -1 stays for the rest).
    /// A light is only marked shadowed once every tile it needs holds
    /// content for the current assignment (fresh assignments render
    /// all-or-nothing under the cap, so they never sample garbage).
    /// `view_frustum` (optional) demotes lights that light nothing on screen —
    /// see k_offscreen_score_scale; nullptr keeps the pure distance ranking.
    void update(std::vector<PointLight>& points, std::vector<SpotLight>& spots, const glm::vec3& camera_pos,
                std::span<const LocalShadowDynamicCaster> dynamic_casters, const LocalShadowConfig& config,
                int tile_resolution, const scene::Frustum* view_frustum = nullptr);

    /// Drop every cached tile (atlas recreated / resolution changed):
    /// assignments survive but re-render before their lights are shadowed.
    void invalidate_all_tiles();

    /// Tiles to render this frame, in draw order (grouped per light).
    [[nodiscard]] const std::vector<LocalShadowTileRequest>& tile_requests() const { return requests_; }

    /// Full slot-data array (LOCAL_SHADOW_MAX_SLOTS entries) + upload dirty flag.
    [[nodiscard]] const std::array<LocalShadowSlotGPU, LOCAL_SHADOW_MAX_SLOTS>& slot_data() const {
        return slot_data_;
    }
    [[nodiscard]] bool slot_data_dirty() const { return slot_data_dirty_; }
    void clear_slot_data_dirty() { slot_data_dirty_ = false; }

    // Introspection (dev overlay / smoke logging / tests).
    [[nodiscard]] int tiles_rendered_last_frame() const { return static_cast<int>(requests_.size()); }
    [[nodiscard]] uint64_t tiles_rendered_total() const { return tiles_rendered_total_; }
    [[nodiscard]] int shadowed_point_lights() const { return shadowed_points_; }
    [[nodiscard]] int shadowed_spot_lights() const { return shadowed_spots_; }

private:
    struct Group {
        bool used = false;
        int light_index = -1; // into this frame's light vector (per-frame)
        PointLight point{};   // stored snapshot (point groups)
        SpotLight spot{};     // stored snapshot (spot groups)
        float score = 0.0f;
        uint8_t dirty_mask = 0; // tiles needing re-render (bit i = face i)
        uint8_t valid_mask = 0; // tiles whose content matches this assignment
        // Next face to service. The per-frame tile cap can cut a point group
        // short; restarting at face 0 every time would re-render faces 0..k
        // forever and starve the rest (a moving light re-dirties all six every
        // frame, so `dirty_mask` is not a record of what is still owed).
        uint8_t face_cursor = 0;
        // Faces a dynamic caster occupied LAST frame. A caster that left a
        // face must dirty the face it left, or its silhouette stays baked in.
        uint8_t prev_caster_mask = 0;
    };

    // Matching + fill + eviction for one light type (defined in the .cpp;
    // only used there).
    template <typename LightT>
    void assign_groups(std::span<Group> groups, int budget, std::vector<LightT>& lights,
                       const glm::vec3& camera_pos, std::span<const LocalShadowDynamicCaster> dynamic_casters,
                       float hysteresis, int tile_resolution, const scene::Frustum* view_frustum);
    // Tile-update cap scheduling (round-robin) + shadow_slot publication.
    void schedule_and_publish(std::vector<PointLight>& points, std::vector<SpotLight>& spots, int point_budget,
                              int spot_budget, const LocalShadowConfig& config, int tile_resolution);

    std::array<Group, LOCAL_SHADOW_MAX_POINT_BUDGET> point_groups_{};
    std::array<Group, LOCAL_SHADOW_MAX_SPOT_BUDGET> spot_groups_{};
    std::array<LocalShadowSlotGPU, LOCAL_SHADOW_MAX_SLOTS> slot_data_{};
    std::vector<LocalShadowTileRequest> requests_;
    std::vector<uint8_t> claimed_scratch_; // per-frame, capacity preserved
    std::vector<int> order_scratch_;
    std::vector<float> score_scratch_;
    bool slot_data_dirty_ = false;
    int ring_cursor_ = 0; // round-robin start for the tile-update cap
    uint64_t tiles_rendered_total_ = 0;
    int shadowed_points_ = 0;
    int shadowed_spots_ = 0;
};

/**
 * GPU wrapper: D32 atlas + LESS_OR_EQUAL comparison sampler + slot-data SSBO
 * around a LocalShadowSelector. The SceneRenderer drives the caster draws;
 * this class only owns the pass begin/end + per-tile viewport plumbing
 * (mirroring render::ShadowMap for the CSM atlas).
 */
class LocalShadowMap {
public:
    LocalShadowMap();
    ~LocalShadowMap();

    LocalShadowMap(const LocalShadowMap&) = delete;
    LocalShadowMap& operator=(const LocalShadowMap&) = delete;
    LocalShadowMap(LocalShadowMap&&) = delete;
    LocalShadowMap& operator=(LocalShadowMap&&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device, int atlas_resolution = 2048);
    void shutdown();

    /// Recreate the atlas at a new resolution; every cached tile is dropped.
    bool reinit(int atlas_resolution);

    [[nodiscard]] bool is_ready() const;

    /// Per-frame selection + caching (see LocalShadowSelector::update).
    void update(std::vector<PointLight>& points, std::vector<SpotLight>& spots, const glm::vec3& camera_pos,
                std::span<const LocalShadowDynamicCaster> dynamic_casters, const LocalShadowConfig& config,
                const scene::Frustum* view_frustum = nullptr);

    /// Record the slot-data SSBO upload into the frame's consolidated copy
    /// pass (no-op when nothing changed).
    void upload(SDL_GPUCopyPass* copy_pass);

    /// Begin the atlas render pass. LOADs previous contents so cached tiles
    /// persist (first-ever pass clears); dirty tiles are cleared individually
    /// by the LocalShadowClear fullscreen draw the SceneRenderer issues.
    SDL_GPURenderPass* begin_pass(SDL_GPUCommandBuffer* cmd);
    /// Viewport + scissor for one tile. clear_depth_range pins the viewport
    /// depth range to [1,1] so the LocalShadowClear draw writes far-plane
    /// depth everywhere; pass false before drawing casters.
    void set_tile_viewport(int tile, bool clear_depth_range);
    void end_pass();

    // Accessors
    [[nodiscard]] SDL_GPUTexture* atlas_texture() const;
    [[nodiscard]] SDL_GPUSampler* compare_sampler() const { return compare_sampler_; }
    [[nodiscard]] SDL_GPUBuffer* slot_buffer() const;
    [[nodiscard]] int atlas_resolution() const { return atlas_resolution_; }
    [[nodiscard]] int tile_resolution() const { return atlas_resolution_ / LOCAL_SHADOW_TILES_PER_ROW; }
    [[nodiscard]] const LocalShadowSelector& selector() const { return selector_; }

private:
    gpu::GPUDevice* device_ = nullptr;
    std::unique_ptr<gpu::GPUTexture> atlas_texture_;
    std::unique_ptr<gpu::GPUBuffer> slot_buffer_;
    SDL_GPUSampler* compare_sampler_ = nullptr;
    SDL_GPURenderPass* current_pass_ = nullptr;
    LocalShadowSelector selector_;
    int atlas_resolution_ = 2048;
    bool atlas_initialized_ = false; // first pass clears, later passes load
};

} // namespace bracken::render::lighting
