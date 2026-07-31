#pragma once

#include "bracken/contracts/environment.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <vector>

namespace bracken::scene {
class Frustum;
}

namespace bracken::core::jobs {
class JobSystem;
}

namespace bracken::render {

namespace gpu = bracken::gpu;

/**
 * Heightmap parameters for GPU-side height sampling
 */
struct HeightmapParams {
    float world_origin_x = 0.0f;
    float world_origin_z = 0.0f;
    float world_size = 1.0f;
    float min_height = -500.0f;
    float max_height = 500.0f;
};

/**
 * Grass vertex uniforms - matches grass.vert.hlsl (new chunked layout).
 * Layout is std140-compatible; padding makes vec3 members 16-byte aligned.
 */
struct alignas(16) GrassVertexUniforms {
    glm::mat4 view_projection;      // 64 bytes
    glm::vec3 camera_pos;           // 12 bytes
    float time;                     // 4  (finishes vec4 slot)
    float _pad2;                    // 4
    float splat_enabled;            // 4  1 when a splatmap is bound at VS slot 1
    float _pad0;                    // 4
    float grass_view_distance;      // 4  (vec4 boundary)
    glm::vec2 wind_direction;       // 8
    float wind_strength;            // 4
    float chunk_size;               // 4  (vec4 boundary)
    float blade_spacing;            // 4
    uint32_t blades_per_chunk_side; // 4
    uint32_t blades_per_chunk_sq;   // 4
    float _pad1;                    // 4  (vec4 boundary)
    float heightmap_origin_x;       // 4
    float heightmap_origin_z;       // 4
    float heightmap_world_size;     // 4
    float heightmap_min_height;     // 4  (vec4 boundary)
    float heightmap_max_height;     // 4
    glm::vec3 camera_forward;       // 12 world-space camera forward (vec4 boundary)
    glm::vec4 splat_params;         // 16 xy = splat region origin (world XZ), zw = region size
    // Unjittered view-projection pair for the depth-prepass velocity output.
    // Declared (and pushed) for BOTH grass shaders so the cbuffer layouts stay
    // byte-identical; grass.vert.hlsl ignores them.
    glm::mat4 curr_vp_no_jitter{1.0f}; // 64
    glm::mat4 prev_vp_no_jitter{1.0f}; // 64
};

// Layout pins against grass.vert.hlsl + prepass_grass.vert.hlsl GrassUniforms.
// camera_forward matches HLSL only because a float3 at 148 exactly fits the
// last 12 bytes of the register starting at 144 — inserting any scalar before
// it silently desynchronizes the two sides (splat_params then starts a fresh
// register at 160 on both sides). Update BOTH shaders in the same change.
static_assert(sizeof(GrassVertexUniforms) == 304, "must match grass.vert.hlsl GrassUniforms (304 B)");
static_assert(offsetof(GrassVertexUniforms, view_projection) == 0);
static_assert(offsetof(GrassVertexUniforms, camera_pos) == 64);
static_assert(offsetof(GrassVertexUniforms, time) == 76);
static_assert(offsetof(GrassVertexUniforms, splat_enabled) == 84);
static_assert(offsetof(GrassVertexUniforms, grass_view_distance) == 92);
static_assert(offsetof(GrassVertexUniforms, wind_direction) == 96);
static_assert(offsetof(GrassVertexUniforms, chunk_size) == 108);
static_assert(offsetof(GrassVertexUniforms, blades_per_chunk_side) == 116);
static_assert(offsetof(GrassVertexUniforms, heightmap_origin_x) == 128);
static_assert(offsetof(GrassVertexUniforms, heightmap_max_height) == 144);
static_assert(offsetof(GrassVertexUniforms, camera_forward) == 148,
              "camera_forward must pack into the last 12 bytes of the register at 144");
static_assert(offsetof(GrassVertexUniforms, splat_params) == 160);
static_assert(offsetof(GrassVertexUniforms, curr_vp_no_jitter) == 176);
static_assert(offsetof(GrassVertexUniforms, prev_vp_no_jitter) == 240);

/**
 * Grass fragment uniforms - matches grass.frag.hlsl cbuffer.
 * Each vec3+float pair fills one 16-byte cbuffer register.
 */
struct alignas(16) GrassLightingUniforms {
    glm::vec3 camera_pos;        // offset 0
    float fog_start;             // offset 12
    glm::vec3 fog_color_horizon; // offset 16
    float fog_end;               // offset 28
    glm::vec3 fog_color_sun;     // offset 32
    float fog_height_falloff;    // offset 44
    glm::vec3 light_dir;         // offset 48
    float ambient_strength;      // offset 60
    glm::vec3 sun_color;         // offset 64
    float sun_intensity;         // offset 76
    glm::vec3 ambient_sky;       // offset 80
    float screen_ao_texel_x;     // offset 92 — 1/render-target size (screen-AO UV from SV_Position)
    glm::vec3 ambient_ground;    // offset 96
    float screen_ao_texel_y;     // offset 108
    float ibl_enabled = 0.0f;    // offset 112 — 1 = sky IBL ambient (samplers 3-5), 0 = hemisphere
    float _pad_ibl[3] = {0.0f, 0.0f, 0.0f};
};

// Layout pins against grass.frag.hlsl GrassLightingUniforms.
static_assert(sizeof(GrassLightingUniforms) == 128, "must match grass.frag.hlsl GrassLightingUniforms (128 B)");
static_assert(offsetof(GrassLightingUniforms, camera_pos) == 0);
static_assert(offsetof(GrassLightingUniforms, fog_start) == 12);
static_assert(offsetof(GrassLightingUniforms, fog_color_horizon) == 16);
static_assert(offsetof(GrassLightingUniforms, fog_color_sun) == 32);
static_assert(offsetof(GrassLightingUniforms, light_dir) == 48);
static_assert(offsetof(GrassLightingUniforms, sun_color) == 64);
static_assert(offsetof(GrassLightingUniforms, ambient_sky) == 80);
static_assert(offsetof(GrassLightingUniforms, ambient_ground) == 96);
static_assert(offsetof(GrassLightingUniforms, ibl_enabled) == 112);

/**
 * Axis-aligned world-space XZ rectangle excluded from grass placement
 * (e.g. buildings, roads, arenas). Applied at chunk granularity
 * (GrassRenderer::CHUNK_SIZE): chunks whose center lies inside any rect are
 * culled. Host-configurable via GrassRenderer::set_exclusion_rects; the
 * default is no exclusions.
 */
struct GrassExclusionRect {
    float min_x = 0.0f;
    float min_z = 0.0f;
    float max_x = 0.0f;
    float max_z = 0.0f;
};

/**
 * GPU chunked grass renderer (SDL3 GPU API)
 *
 * Architecture:
 *  - World is partitioned into fixed-size grass chunks (e.g. 32x32 world units).
 *  - Each chunk contains BLADES_PER_CHUNK_SIDE^2 blades.
 *  - Each frame, CPU frustum-culls chunks against camera and uploads visible
 *    chunk origins into a storage buffer. Shader reads chunk origin via
 *    instance_id / blades_per_chunk_sq.
 *  - Smooth LOD fade replaces discrete density bands → no phase-in pop.
 */
class GrassRenderer {
public:
    GrassRenderer();
    ~GrassRenderer();

    GrassRenderer(const GrassRenderer&) = delete;
    GrassRenderer& operator=(const GrassRenderer&) = delete;
    GrassRenderer(GrassRenderer&&) = delete;
    GrassRenderer& operator=(GrassRenderer&&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry, float world_width, float world_height);

    /// Advance the grass clock. All grass/wind animation is absolute-time
    /// driven (the shader reads `current_time`), so there is no dt parameter —
    /// rewinding or holding `current_time` rewinds/pauses the wind.
    void update(float current_time);

    /// Cull chunks and record the visible chunk storage buffer upload into the
    /// caller's open copy pass (the frame's consolidated pre-pass uploads).
    void upload_chunks(SDL_GPUCopyPass* copy_pass, const glm::vec3& camera_pos, const scene::Frustum* frustum);

    /// Draw the grass. Requires upload_chunks() to have been called this frame.
    /// Must be called inside an active render pass.
    /// `ao_binding` is the full-res screen-space AO texture + sampler
    /// (fragment slot 2; pass the 1x1 white dummy when AO is off — the slot
    /// must always be bound) and `ao_texel` = 1 / render-target size.
    /// `depth_equal` selects the GrassDepthEqual pipeline variant (depth test
    /// EQUAL, writes off) — used when the depth prepass already laid down the
    /// blade depth this frame.
    /// `ibl_bindings` is the 3-entry sky IBL chain (irradiance cube,
    /// prefiltered specular cube, BRDF LUT — fragment slots 3-5; the slots
    /// must always be bound, with cube-typed dummies when IBL is off) and
    /// `ibl_enabled` sets the shader's iblEnabled flag.
    /// `local_shadow_binding` is the local (point/spot) shadow atlas +
    /// comparison sampler at slot 6 (the shader's last sampler; sampled only
    /// for lights with shadow_slot >= 0).
    void render(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view, const glm::mat4& projection,
                const glm::vec3& camera_pos, const glm::vec3& light_dir,
                const SDL_GPUTextureSamplerBinding* shadow_bindings = nullptr, int shadow_binding_count = 0,
                const SDL_GPUTextureSamplerBinding* ao_binding = nullptr, glm::vec2 ao_texel = glm::vec2(0.0f),
                bool depth_equal = false,
                const SDL_GPUTextureSamplerBinding* ibl_bindings = nullptr, bool ibl_enabled = false,
                const SDL_GPUTextureSamplerBinding* local_shadow_binding = nullptr);

    /// Depth-prepass draw (velocity + depth, DepthPrepassGrass pipeline).
    /// Must use the SAME per-frame state as render() (same time/wind/chunks)
    /// so blade positions match bit-for-bit; `projection` is the (possibly
    /// jittered) draw projection, the *_no_jitter matrices feed velocity.
    void render_prepass(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view,
                        const glm::mat4& projection, const glm::vec3& camera_pos,
                        const glm::mat4& curr_vp_no_jitter, const glm::mat4& prev_vp_no_jitter);

    void shutdown();

    void set_heightmap(gpu::GPUTexture* texture, const HeightmapParams& params) {
        heightmap_texture_ = texture;
        heightmap_params_ = params;
    }

    /// Terrain splatmap for splat-aware blade placement (grass layer = R channel).
    /// Takes a provider, invoked once per frame at draw time, so splatmap
    /// recreation (e.g. TerrainRenderer::update_splatmap during editor
    /// painting) can never leave a dangling texture pointer. A null provider —
    /// or a provider returning nullptr — disables splat masking.
    /// Host-opt-in: nothing in the engine wires this by default. Example:
    ///   grass->set_splatmap([&terrain] { return terrain.splatmap_texture(); });
    void set_splatmap(std::function<gpu::GPUTexture*()> provider) { splatmap_provider_ = std::move(provider); }

    /// World-space XZ rectangles where no grass grows (chunk-granular).
    /// Default: empty (grass everywhere inside the world bounds).
    void set_exclusion_rects(std::vector<GrassExclusionRect> rects) { exclusion_rects_ = std::move(rects); }
    [[nodiscard]] std::span<const GrassExclusionRect> exclusion_rects() const { return exclusion_rects_; }

    /// Per-frame environment (sun/ambient/fog). Owned by the scene; call once per frame.
    void set_environment(const EnvironmentState& env) { env_ = env; }

    // local_shadow_slots is the 4th fragment SSBO (per-tile local-shadow slot
    // data); nullptr substitutes the light-data buffer at bind time (declared
    // but never read while every light's shadow_slot is -1).
    void set_cluster_lighting(SDL_GPUBuffer* light_data, SDL_GPUBuffer* cluster_offsets, SDL_GPUBuffer* light_indices,
                              const void* params, size_t params_size, SDL_GPUBuffer* local_shadow_slots = nullptr) {
        cluster_light_data_ = light_data;
        cluster_offsets_ = cluster_offsets;
        cluster_indices_ = light_indices;
        cluster_params_ = params;
        cluster_params_size_ = params_size;
        cluster_local_shadow_slots_ = local_shadow_slots;
    }

    // Wind parameters (host-facing). Direction is a world-XZ vector,
    // normalized at upload (a zero vector disables directional sway);
    // magnitude scales the shader's layered wind waves.
    float wind_magnitude = 0.8f;
    glm::vec2 wind_direction = glm::vec2(1.0f, 0.3f);

    /// wind_direction normalized for upload (zero-safe). Exposed for tests.
    [[nodiscard]] glm::vec2 effective_wind_direction() const {
        const float len = glm::length(wind_direction);
        return (len > 1e-5f) ? wind_direction / len : glm::vec2(0.0f);
    }

    // Grass draw distance in world units (host-facing; direct writes are
    // respected). The effective per-frame distance is
    // min(grass_view_distance, view-distance cap, MAX_VIEW_DISTANCE); the
    // SceneRenderer feeds the scene draw distance as the cap each frame so
    // grass never out-draws the world. The early VS blade collapse keeps the
    // cost linear-ish.
    static constexpr float MAX_VIEW_DISTANCE = 1200.0f;
    float grass_view_distance = MAX_VIEW_DISTANCE;

    /// Upper bound applied on top of grass_view_distance (<= 0 disables the
    /// cap). Called per frame by the SceneRenderer with the scene draw
    /// distance; standalone hosts can ignore it.
    void set_view_distance_cap(float cap) { view_distance_cap_ = cap; }

    /// The view distance actually used for culling and the LOD fade.
    [[nodiscard]] float effective_view_distance() const {
        float dist = std::min(grass_view_distance, MAX_VIEW_DISTANCE);
        if (view_distance_cap_ > 0.0f) {
            dist = std::min(dist, view_distance_cap_);
        }
        return dist;
    }

    /// Grass-free margin (world units) inside the world edges. Default 0:
    /// grass grows to the world bounds; chunks fully outside the world are
    /// always culled.
    float edge_margin = 0.0f;

    // Lighting tunables (fed from GraphicsSettings each frame)
    float ambient_strength = 0.5f;
    float sun_intensity = 1.0f;

    // Chunk geometry constants
    static constexpr float CHUNK_SIZE = 32.0f;            // world units per chunk side
    static constexpr uint32_t BLADES_PER_CHUNK_SIDE = 12; // blades per chunk side
    static constexpr uint32_t BLADES_PER_CHUNK_SQ = BLADES_PER_CHUNK_SIDE * BLADES_PER_CHUNK_SIDE;

    /// Pure CPU chunk culling used by upload_chunks (exposed for unit tests).
    /// Appends visible CHUNK_SIZE-sized chunk origins (world XZ in .x/.z) to
    /// out_chunks. A chunk survives when: its center is within view_distance
    /// of the camera (XZ), it overlaps the world rect inset by edge_margin,
    /// its center is outside every exclusion rect, and (if a frustum is
    /// given) its AABB over [y_min, y_max] intersects the frustum.
    ///
    /// worker-safe (reads only its arguments); the grid walk fans out over
    /// `js` in CULL_ROW_GRAIN-row chunks, each chunk appending to its own
    /// partial, merged single-threaded in chunk-index order — the appended
    /// sequence is bit-identical to the serial row-major walk at any worker
    /// count (upload_chunks' FNV hash of the visible set keys on exact
    /// element order). Call sites pass JobSystem::instance(); tests may pass
    /// caller-owned pools.
    struct ChunkCullParams {
        glm::vec3 camera_pos{0.0f};
        float view_distance = 0.0f;
        float world_width = 0.0f;
        float world_height = 0.0f;
        float edge_margin = 0.0f;
        float y_min = 0.0f;
        float y_max = 0.0f;
    };
    static void cull_chunks(core::jobs::JobSystem& js, const ChunkCullParams& params,
                            std::span<const GrassExclusionRect> exclusions, const scene::Frustum* frustum,
                            std::vector<glm::vec4>& out_chunks);

    /// Fan-out grain for cull_chunks, in grid rows. Sized so typical scenes
    /// stay near-serial: the max-view-distance grid is 79 rows -> 2 chunks;
    /// anything under ~770 world units of view distance is a single inline
    /// chunk (granularity control, not a second code path).
    static constexpr std::size_t CULL_ROW_GRAIN = 48;

private:
    void generate_blade_mesh();

    // Per-frame vertex uniforms shared by render() and render_prepass(); the
    // two passes MUST push identical values (velocity matrices aside) so the
    // shared blade math yields bit-identical positions for the EQUAL depth test.
    GrassVertexUniforms build_vertex_uniforms(const glm::mat4& view, const glm::mat4& projection,
                                              const glm::vec3& camera_pos, bool splat_enabled) const;

    gpu::GPUDevice* device_ = nullptr;
    gpu::PipelineRegistry* pipeline_registry_ = nullptr;

    // Single blade mesh (created once at init)
    std::unique_ptr<gpu::GPUBuffer> blade_vertex_buffer_;
    std::unique_ptr<gpu::GPUBuffer> blade_index_buffer_;
    uint32_t blade_index_count_ = 0;

    // Per-frame chunk storage buffer. Reused + grown via capacity-doubling.
    std::unique_ptr<gpu::GPUBuffer> chunk_storage_buffer_;
    size_t chunk_storage_capacity_bytes_ = 0;
    std::vector<glm::vec4> visible_chunks_scratch_; // reused per frame
    uint64_t last_chunk_set_hash_ = 0;              // skip SSBO upload when the visible set is unchanged

    // Heightmap for GPU height sampling
    gpu::GPUTexture* heightmap_texture_ = nullptr;
    std::unique_ptr<gpu::GPUSampler> heightmap_sampler_;
    HeightmapParams heightmap_params_;

    // Terrain splatmap provider for splat-aware placement; fetched each frame
    // at draw time so texture recreation can't dangle.
    std::function<gpu::GPUTexture*()> splatmap_provider_;

    // Host-configured grass-free zones (chunk-granular).
    std::vector<GrassExclusionRect> exclusion_rects_;

    // Sun/ambient/fog environment, fed per frame by the scene renderer
    EnvironmentState env_{};

    float world_width_ = 0.0f;
    float world_height_ = 0.0f;
    float current_time_ = 0.0f;
    float view_distance_cap_ = 0.0f; // <= 0 = uncapped
    bool initialized_ = false;

    SDL_GPUBuffer* cluster_light_data_ = nullptr;
    SDL_GPUBuffer* cluster_offsets_ = nullptr;
    SDL_GPUBuffer* cluster_indices_ = nullptr;
    SDL_GPUBuffer* cluster_local_shadow_slots_ = nullptr;
    const void* cluster_params_ = nullptr;
    size_t cluster_params_size_ = 0;
};

} // namespace bracken::render
