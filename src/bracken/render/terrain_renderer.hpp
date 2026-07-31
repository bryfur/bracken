#pragma once

#include "bracken/contracts/environment.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/heightmap.hpp"
#include <cstddef>
#include <glm/glm.hpp>
#include <memory>
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
 * Terrain vertex format for SDL3 GPU API
 * Matches the vertex attributes expected by terrain.vert.hlsl
 */
struct TerrainVertex {
    glm::vec3 position; // POSITION
    glm::vec2 texCoord; // TEXCOORD0
    glm::vec4 color;    // COLOR0
    glm::vec3 normal;   // NORMAL
};

// Verify TerrainVertex is tightly packed as expected by pipeline_registry
static_assert(sizeof(TerrainVertex) == sizeof(float) * 12,
              "TerrainVertex must be 12 floats (48 bytes) to match pipeline vertex stride");

/**
 * Terrain transform uniforms - matches terrain.vert.hlsl cbuffer
 */
struct alignas(16) TerrainTransformUniforms {
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 cameraPos;
    float _padding0;
};

// Layout pins against terrain.vert.hlsl TransformUniforms.
static_assert(sizeof(TerrainTransformUniforms) == 144, "must match terrain.vert.hlsl TransformUniforms (144 B)");
static_assert(offsetof(TerrainTransformUniforms, projection) == 64);
static_assert(offsetof(TerrainTransformUniforms, cameraPos) == 128);

/**
 * Terrain lighting uniforms - matches terrain.frag.hlsl cbuffer.
 * Each vec3+float pair fills one 16-byte cbuffer register.
 */
struct alignas(16) TerrainLightingUniforms {
    glm::vec3 fogColorHorizon; // offset 0
    float fogStart;            // offset 12
    glm::vec3 fogColorSun;     // offset 16
    float fogEnd;              // offset 28
    glm::vec3 lightDir;        // offset 32
    float fogHeightFalloff;    // offset 44
    glm::vec3 cameraPos;       // offset 48
    float screenAOTexelX;      // offset 60 — 1/render-target size (screen-AO UV from SV_Position)
    glm::vec3 ambientSky;      // offset 64
    float ambientStrength;     // offset 76
    glm::vec3 ambientGround;   // offset 80
    float sunIntensity;        // offset 92
    glm::vec3 sunColor;        // offset 96
    float screenAOTexelY;      // offset 108
    glm::vec4 splat_params;    // offset 112 — xy = splat origin (world XZ), zw = splat X/Z size
    float iblEnabled = 0.0f;   // offset 128 — 1 = sky IBL ambient (samplers 5-7), 0 = hemisphere
    float _padIbl[3] = {0.0f, 0.0f, 0.0f};
};

// Layout pins against terrain.frag.hlsl LightingUniforms.
static_assert(sizeof(TerrainLightingUniforms) == 144, "must match terrain.frag.hlsl LightingUniforms (144 B)");
static_assert(offsetof(TerrainLightingUniforms, fogColorHorizon) == 0);
static_assert(offsetof(TerrainLightingUniforms, fogColorSun) == 16);
static_assert(offsetof(TerrainLightingUniforms, lightDir) == 32);
static_assert(offsetof(TerrainLightingUniforms, cameraPos) == 48);
static_assert(offsetof(TerrainLightingUniforms, ambientSky) == 64);
static_assert(offsetof(TerrainLightingUniforms, ambientGround) == 80);
static_assert(offsetof(TerrainLightingUniforms, sunColor) == 96);
static_assert(offsetof(TerrainLightingUniforms, splat_params) == 112);
static_assert(offsetof(TerrainLightingUniforms, iblEnabled) == 128);

/**
 * TerrainRenderer handles terrain rendering using server-provided heightmaps.
 * Ported to SDL3 GPU API - replaces OpenGL calls with GPUBuffer and GPUTexture.
 */
class TerrainRenderer {
public:
    TerrainRenderer();
    ~TerrainRenderer();

    // Non-copyable
    TerrainRenderer(const TerrainRenderer&) = delete;
    TerrainRenderer& operator=(const TerrainRenderer&) = delete;
    TerrainRenderer(TerrainRenderer&&) = delete;
    TerrainRenderer& operator=(TerrainRenderer&&) = delete;

    /**
     * Initialize terrain resources.
     * @param device The GPU device for resource creation
     * @param pipeline_registry The pipeline registry for getting terrain pipeline
     * @param world_width World X dimension
     * @param world_height World Z dimension
     */
    [[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry, float world_width, float world_height);

    /**
     * Set heightmap from server data. Uploads to GPU texture.
     */
    void set_heightmap(const bracken::Heightmap& heightmap);

    /**
     * Update splatmap texture from CPU data. Host-opt-in (nothing in the
     * engine calls this): destroys and recreates splatmap_texture(), so any
     * host-cached raw pointer to the old texture is invalid afterwards —
     * GrassRenderer::set_splatmap takes a provider for exactly this reason.
     * @param data RGBA pixel data (must be splatmap_resolution^2 * 4 bytes)
     * @param resolution Splatmap resolution (width and height)
     */
    void update_splatmap(const uint8_t* data, uint32_t resolution);

    /**
     * Clean up terrain resources.
     */
    void shutdown();

    /**
     * Render the terrain mesh using SDL3 GPU API.
     * @param pass The render pass to draw into
     * @param cmd The command buffer for push constants
     * @param view View matrix
     * @param projection Projection matrix
     * @param camera_pos Camera position
     * @param light_dir Light direction
     * @param ao_binding Full-res screen-space AO texture + sampler (slot 4;
     *        pass the 1x1 white dummy when AO is off — the slot must always
     *        be bound). ao_texel = 1 / render-target size for the shader's
     *        SV_Position -> AO UV mapping.
     * @param depth_equal Use the TerrainDepthEqual pipeline variant (depth
     *        test EQUAL, writes off) — set when the depth prepass already
     *        laid down the terrain depth this frame.
     * @param ibl_bindings Sky IBL chain bindings, 3 entries (irradiance cube,
     *        prefiltered specular cube, BRDF LUT — slots 5-7; the slots must
     *        always be bound, with cube-typed dummies when IBL is off).
     * @param ibl_enabled Sets the shader's iblEnabled flag: 1 = sample the
     *        IBL chain for ambient, 0 = hemisphere ambient fallback.
     * @param local_shadow_binding Local (point/spot) shadow atlas +
     *        comparison sampler (slot 8 — the shader's last sampler; sampled
     *        only for lights with shadow_slot >= 0, so any D32 depth texture
     *        keeps the slot valid when local shadows are off).
     */
    void render(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view, const glm::mat4& projection,
                const glm::vec3& camera_pos, const glm::vec3& light_dir,
                const SDL_GPUTextureSamplerBinding* shadow_bindings = nullptr, int shadow_binding_count = 0,
                const SDL_GPUTextureSamplerBinding* ao_binding = nullptr, glm::vec2 ao_texel = glm::vec2(0.0f),
                const scene::Frustum* frustum = nullptr, bool depth_equal = false,
                const SDL_GPUTextureSamplerBinding* ibl_bindings = nullptr, bool ibl_enabled = false,
                const SDL_GPUTextureSamplerBinding* local_shadow_binding = nullptr);

    /**
     * Depth-prepass draw (velocity + depth, DepthPrepassTerrain pipeline).
     * Draws the SAME full-LOD mesh + frustum-culled tile set as render() so
     * the forward pass's EQUAL depth test matches. `uniforms.projection` is
     * the (possibly jittered) draw projection; the no-jitter matrices feed
     * the velocity output.
     */
    void render_prepass(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                        const gpu::PrepassCameraUniforms& uniforms, const scene::Frustum* frustum = nullptr);

    // Per-frame cluster lighting plumbing. Buffers and uniform are owned by
    // the SceneRenderer's ClusterGrid; nullptr disables clustered lighting.
    // local_shadow_slots is the 4th fragment SSBO (per-tile local-shadow slot
    // data, LocalShadowMap::slot_buffer()); pass nullptr to substitute the
    // light-data buffer — the slot is declared by the shader but never read
    // while every light's shadow_slot is -1.
    void set_cluster_lighting(SDL_GPUBuffer* light_data, SDL_GPUBuffer* cluster_offsets, SDL_GPUBuffer* light_indices,
                              const void* params, size_t params_size, SDL_GPUBuffer* local_shadow_slots = nullptr);

    /**
     * Render terrain into a shadow depth map.
     */
    void render_shadow(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& light_view_projection,
                       const scene::Frustum* frustum = nullptr);

    /**
     * Get terrain height at any world position.
     * Samples from CPU-side heightmap data (for physics, placement, etc.)
     */
    float get_height(float x, float z) const;

    /**
     * Get terrain normal at any world position.
     */
    glm::vec3 get_normal(float x, float z) const;

    // Accessors
    float world_width() const { return world_width_; }
    float world_height() const { return world_height_; }
    bool has_heightmap() const { return heightmap_ != nullptr; }
    gpu::GPUTexture* heightmap_texture() { return heightmap_texture_.get(); }
    gpu::GPUTexture* material_array_texture() { return material_array_texture_.get(); }
    gpu::GPUTexture* splatmap_texture() { return splatmap_texture_.get(); }

    // Settings
    /// Per-frame environment (sun/ambient/fog). Owned by the scene; call once per frame.
    void set_environment(const EnvironmentState& env) { env_ = env; }
    void set_lighting_tunables(float ambient_strength, float sun_intensity) {
        ambient_strength_ = ambient_strength;
        sun_intensity_ = sun_intensity;
    }

    /**
     * Set anisotropic filter level for terrain textures.
     * Recreates the terrain material-array sampler with the new anisotropy level.
     */
    void set_anisotropic_filter(float level);

    // --- Pure CPU tile-culling machinery (public + GPU-free for unit tests;
    // --- the private draw path below is the only engine consumer) ---

    // Terrain tiling: each tile is a rectangular region of cells with its own
    // index range and world-space AABB. Per-frame frustum culling skips tiles
    // outside the view frustum — typically 80–95% are culled per pass.
    struct TerrainTile {
        uint32_t first_index = 0;
        uint32_t index_count = 0;
        glm::vec3 aabb_min = glm::vec3(0.0f);
        glm::vec3 aabb_max = glm::vec3(0.0f);
    };

    // Row-level acceleration for tile culling: tiles are generated row-major,
    // so each z-row is a contiguous slice of the tile vector with a merged
    // AABB. Culling tests one row AABB first and only tests per-tile AABBs
    // inside rows that intersect the frustum, turning the per-pass linear
    // scan of every tile into row tests + a handful of intersecting rows.
    struct TileRow {
        size_t first_tile = 0;
        size_t tile_count = 0;
        glm::vec3 aabb_min = glm::vec3(0.0f);
        glm::vec3 aabb_max = glm::vec3(0.0f);
    };

    // A contiguous index-buffer range covering one or more visible tiles —
    // one SDL draw call each.
    struct TileRun {
        uint32_t first_index = 0;
        uint32_t index_count = 0;
        bool operator==(const TileRun&) const = default;
    };

    /// Fan-out grain for cull_tile_runs, in tile rows. A 42000-unit world is
    /// ~52 rows -> 2 chunks; the playground-scale worlds stay a single inline
    /// chunk (granularity control, not a second code path).
    static constexpr std::size_t CULL_ROW_GRAIN = 32;
    /// Fan-out grain for build_tile_rows, in tiles (init-time only).
    static constexpr std::size_t ROW_BUILD_TILE_GRAIN = 512;

    // Rebuilds `rows` from row-major `tiles` (grouped by shared min-z band).
    // worker-safe (pure on its arguments). Fans out over `js` in
    // ROW_BUILD_TILE_GRAIN-tile chunks; each chunk builds its own partial row
    // list, merged single-threaded in chunk-index order with z-band rows
    // spanning a chunk seam coalesced — output identical to the serial scan
    // at any worker count (chunk boundaries depend only on the grain).
    static void build_tile_rows(core::jobs::JobSystem& js, const std::vector<TerrainTile>& tiles,
                                std::vector<TileRow>& rows);

    // Frustum-culls `tiles` (rows AABB-rejected first, then per-tile tests)
    // and merges visible tiles with contiguous index ranges into runs,
    // appended to out_runs (cleared first). worker-safe (pure on its
    // arguments). The per-tile frustum tests fan out over `js` in
    // CULL_ROW_GRAIN-row chunks into per-chunk partial run lists; the merge
    // consumes partials single-threaded in chunk-index order, coalescing runs
    // across chunk seams — bit-identical to the serial walk at any worker
    // count. Engine call sites pass JobSystem::instance(); tests may pass
    // caller-owned pools.
    static void cull_tile_runs(core::jobs::JobSystem& js, const scene::Frustum& frustum,
                               const std::vector<TerrainTile>& tiles, const std::vector<TileRow>& rows,
                               std::vector<TileRun>& out_runs);

private:
    void generate_terrain_mesh();
    void generate_shadow_terrain_mesh(); // low-LOD mesh for shadow passes
    void load_terrain_textures();
    void create_fallback_material_array();
    void create_fallback_splatmap();
    void upload_heightmap_texture();

    gpu::GPUDevice* device_ = nullptr;
    gpu::PipelineRegistry* pipeline_registry_ = nullptr;

    SDL_GPUBuffer* cluster_light_data_ = nullptr;
    SDL_GPUBuffer* cluster_offsets_ = nullptr;
    SDL_GPUBuffer* cluster_indices_ = nullptr;
    SDL_GPUBuffer* cluster_local_shadow_slots_ = nullptr;
    const void* cluster_params_ = nullptr;
    size_t cluster_params_size_ = 0;

    float world_width_ = 0.0f;
    float world_height_ = 0.0f;

    // Server-provided heightmap (CPU side for sampling)
    std::unique_ptr<bracken::Heightmap> heightmap_;

    // GPU resources
    std::unique_ptr<gpu::GPUTexture> heightmap_texture_;

    // Terrain material textures (using Texture2DArray to reduce sampler count)
    std::unique_ptr<gpu::GPUTexture> material_array_texture_; // 4 layers: grass, dirt, rock, sand
    std::unique_ptr<gpu::GPUTexture> splatmap_texture_;

    std::unique_ptr<gpu::GPUSampler> material_sampler_;
    std::unique_ptr<gpu::GPUSampler> splatmap_sampler_; // linear CLAMP — splatmap must not wrap in the margin ring
    std::unique_ptr<gpu::GPUBuffer> vertex_buffer_;
    std::unique_ptr<gpu::GPUBuffer> index_buffer_;
    uint32_t index_count_ = 0;

    // Low-LOD mesh used only for shadow passes. Coarser cell size -> ~16x fewer triangles.
    // Shadow resolution already smooths over the mesh detail, so visual impact is negligible.
    std::unique_ptr<gpu::GPUBuffer> shadow_vertex_buffer_;
    std::unique_ptr<gpu::GPUBuffer> shadow_index_buffer_;
    uint32_t shadow_index_count_ = 0;

    std::vector<TerrainTile> tiles_;
    std::vector<TerrainTile> shadow_tiles_;
    std::vector<TileRow> tile_rows_;
    std::vector<TileRow> shadow_tile_rows_;

    // Reused per draw call by draw_culled_tiles (main + prepass + cascades).
    // main-thread (like the draw path that owns it).
    std::vector<TileRun> tile_runs_scratch_;

    // Frustum-culled tile draw shared by the main and shadow passes: culls
    // via cull_tile_runs (fanned out over `js`), then issues one draw per
    // contiguous run. Call sites pass JobSystem::instance().
    void draw_culled_tiles(core::jobs::JobSystem& js, SDL_GPURenderPass* pass, const scene::Frustum& frustum,
                           const std::vector<TerrainTile>& tiles, const std::vector<TileRow>& rows);

    // Sun/ambient/fog environment, fed per frame by the scene renderer.
    EnvironmentState env_{};

    // Lighting tunables (per-frame from GraphicsSettings)
    float ambient_strength_ = 0.5f;
    float sun_intensity_ = 1.0f;
};

} // namespace bracken::render
