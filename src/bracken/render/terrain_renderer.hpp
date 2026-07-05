#pragma once

#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/heightmap.hpp"
#include "bracken/render/environment.hpp"
#include <cstddef>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace bracken::scene {
class Frustum;
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
    float world_size;          // offset 60
    glm::vec3 ambientSky;      // offset 64
    float ambientStrength;     // offset 76
    glm::vec3 ambientGround;   // offset 80
    float sunIntensity;        // offset 92
    glm::vec3 sunColor;        // offset 96
    float _padding0;           // offset 108 — struct ends 16-byte aligned (112)
};

// Layout pins against terrain.frag.hlsl LightingUniforms.
static_assert(sizeof(TerrainLightingUniforms) == 112, "must match terrain.frag.hlsl LightingUniforms (112 B)");
static_assert(offsetof(TerrainLightingUniforms, fogColorHorizon) == 0);
static_assert(offsetof(TerrainLightingUniforms, fogColorSun) == 16);
static_assert(offsetof(TerrainLightingUniforms, lightDir) == 32);
static_assert(offsetof(TerrainLightingUniforms, cameraPos) == 48);
static_assert(offsetof(TerrainLightingUniforms, world_size) == 60);
static_assert(offsetof(TerrainLightingUniforms, ambientSky) == 64);
static_assert(offsetof(TerrainLightingUniforms, ambientGround) == 80);
static_assert(offsetof(TerrainLightingUniforms, sunColor) == 96);

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
     * Update splatmap texture from CPU data.
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
     */
    void render(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view, const glm::mat4& projection,
                const glm::vec3& camera_pos, const glm::vec3& light_dir,
                const SDL_GPUTextureSamplerBinding* shadow_bindings = nullptr, int shadow_binding_count = 0,
                const scene::Frustum* frustum = nullptr);

    // Per-frame cluster lighting plumbing. Buffers and uniform are owned by
    // the SceneRenderer's ClusterGrid; nullptr disables clustered lighting.
    void set_cluster_lighting(SDL_GPUBuffer* light_data, SDL_GPUBuffer* cluster_offsets, SDL_GPUBuffer* light_indices,
                              const void* params, size_t params_size);

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
     * Recreates the grass sampler with the new anisotropy level.
     */
    void set_anisotropic_filter(float level);

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

    // Terrain tiling: each tile is a rectangular region of cells with its own
    // index range and world-space AABB. Per-frame frustum culling skips tiles
    // outside the view frustum — typically 80–95% are culled per pass.
    struct TerrainTile {
        uint32_t first_index = 0;
        uint32_t index_count = 0;
        glm::vec3 aabb_min = glm::vec3(0.0f);
        glm::vec3 aabb_max = glm::vec3(0.0f);
    };
    std::vector<TerrainTile> tiles_;
    std::vector<TerrainTile> shadow_tiles_;

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
    std::vector<TileRow> tile_rows_;
    std::vector<TileRow> shadow_tile_rows_;

    // Rebuilds `rows` from row-major `tiles` (grouped by shared min-z band).
    static void build_tile_rows(const std::vector<TerrainTile>& tiles, std::vector<TileRow>& rows);
    // Frustum-culled tile draw shared by the main and shadow passes: rows are
    // AABB-tested first; visible tiles merge into contiguous index-range runs.
    static void draw_culled_tiles(SDL_GPURenderPass* pass, const scene::Frustum& frustum,
                                  const std::vector<TerrainTile>& tiles, const std::vector<TileRow>& rows);

    // Sun/ambient/fog environment, fed per frame by the scene renderer.
    EnvironmentState env_{};

    // Lighting tunables (per-frame from GraphicsSettings)
    float ambient_strength_ = 0.5f;
    float sun_intensity_ = 1.0f;
};

} // namespace bracken::render
