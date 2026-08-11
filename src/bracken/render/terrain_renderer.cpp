#include "terrain_renderer.hpp"
#include "bracken/contracts/frustum.hpp"
#include "bracken/render/gi/gi_bindings.hpp"
#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/cache_line.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/core/log_throttle.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_types.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/heightmap.hpp"
#include "bracken/memory/arena.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float2.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"
#include "SDL3/SDL_gpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <SDL3_image/SDL_image.h>
#include <vector>

namespace bracken::render {

namespace {
constexpr const char* LOG_TAG = "terrain";

/// Union of a page's tile AABBs — the page-level reject box.
///
/// An empty tile list yields a degenerate box AT THE ORIGIN, and that box is
/// NOT self-evidently rejected: a point at the world origin is inside the
/// frustum whenever the camera is looking at it. What makes an empty page safe
/// is upstream of the box — a tile is recorded only when its index count is
/// non-zero, so no tiles means `page.index_count == 0`, and all three draw
/// sites skip such a page before they ever consult its AABB.
glm::vec3 tile_bounds_min(const std::vector<TerrainRenderer::TerrainTile>& tiles) {
    if (tiles.empty()) {
        return glm::vec3(0.0f);
    }
    glm::vec3 lo = tiles.front().aabb_min;
    for (const TerrainRenderer::TerrainTile& tile : tiles) {
        lo = glm::min(lo, tile.aabb_min);
    }
    return lo;
}

glm::vec3 tile_bounds_max(const std::vector<TerrainRenderer::TerrainTile>& tiles) {
    if (tiles.empty()) {
        return glm::vec3(0.0f);
    }
    glm::vec3 hi = tiles.front().aabb_max;
    for (const TerrainRenderer::TerrainTile& tile : tiles) {
        hi = glm::max(hi, tile.aabb_max);
    }
    return hi;
}

} // namespace

namespace gpu = bracken::gpu;

TerrainRenderer::TerrainRenderer() = default;

TerrainRenderer::~TerrainRenderer() {
    shutdown();
}

bool TerrainRenderer::init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry, float world_width,
                           float world_height) {
    device_ = &device;
    pipeline_registry_ = &pipeline_registry;
    world_width_ = world_width;
    world_height_ = world_height;

    // Create material sampler with linear filtering and repeat addressing
    material_sampler_ = gpu::GPUSampler::create(device, gpu::SamplerConfig::anisotropic(8.0f));
    if (!material_sampler_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "TerrainRenderer::init: Failed to create material sampler");
        return false;
    }

    // Splatmap gets its own CLAMP sampler: the REPEAT material sampler made the
    // splatmap wrap across the 500-unit margin ring outside the world bounds.
    splatmap_sampler_ = gpu::GPUSampler::create(device, gpu::SamplerConfig::linear_clamp());
    if (!splatmap_sampler_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "TerrainRenderer::init: Failed to create splatmap sampler");
        return false;
    }

    load_terrain_textures();
    // Note: terrain mesh will be generated when heightmap is received
    // For now, generate a flat placeholder — one page, no height data.
    rebuild_pages();
    generate_terrain_mesh();
    generate_shadow_terrain_mesh();

    return true;
}

void TerrainRenderer::set_heightmap(const bracken::Heightmap& heightmap) {
    // The 1x1 adapter. Every pre-paging host call site lands here, and the
    // grid path below reproduces the old mesh/tile/draw stream exactly for a
    // single page — see docs/gap_streaming.md.
    set_heightmap_grid(bracken::HeightmapGrid::from_single(heightmap));
}

void TerrainRenderer::set_heightmap_grid(const bracken::HeightmapGrid& grid) {
    const core::Result<void, std::string> check = grid.validate();
    if (!check) {
        // Refusing is the point: an accepted-but-inconsistent grid renders as
        // a stepped seam with nothing logged anywhere.
        BRACKEN_LOG_ERROR(LOG_TAG, "TerrainRenderer::set_heightmap_grid: refusing the grid — {}", check.error());
        return;
    }
    if (grid.page_count() > 1 && grid.page_size < bracken::HeightmapGrid::k_recommended_min_page_size) {
        // Degradation, not failure: correct but expensive. See
        // set_heightmap_grid's draw-call arithmetic.
        BRACKEN_LOG_WARN(LOG_TAG,
                         "TerrainRenderer::set_heightmap_grid: {}x{} pages of {:.0f} units is below the {:.0f}-unit "
                         "recommendation — tile runs never merge across pages, so every visible page costs at least "
                         "6 draws a frame (main + prepass + 4 cascades)",
                         grid.pages_x, grid.pages_z, grid.page_size,
                         bracken::HeightmapGrid::k_recommended_min_page_size);
    }

    grid_ = grid;
    has_heightmap_ = grid_.resident_pages() > 0;

    rebuild_pages();
    // Order preserved from the pre-paging path: texture first, then the two
    // meshes (both of which sample get_height, i.e. the CPU side).
    upload_heightmap_texture();
    generate_terrain_mesh();
    generate_shadow_terrain_mesh();
}

void TerrainRenderer::rebuild_pages() {
    const uint32_t nx = std::max(grid_.pages_x, 1u);
    const uint32_t nz = std::max(grid_.pages_z, 1u);
    pages_.clear();
    pages_.resize(static_cast<std::size_t>(nx) * nz);
    for (uint32_t pz = 0; pz < nz; ++pz) {
        for (uint32_t px = 0; px < nx; ++px) {
            TerrainPage& page = pages_[static_cast<std::size_t>(pz) * nx + px];
            page.px = static_cast<int>(px);
            page.pz = static_cast<int>(pz);
        }
    }
}

TerrainRenderer::PageRect TerrainRenderer::page_mesh_rect(int px, int pz) const {
    // THE SINGLE-PAGE RULE, and it is a compatibility rule rather than an
    // aesthetic one: before paging the mesh spanned the renderer's init-time
    // world rect (world_width_ x world_height_) plus a 500-unit margin, and
    // that rect has never been derived from the heightmap — a map may be any
    // size and sampling clamps. A 1x1 grid therefore keeps exactly that rect,
    // which is what makes the adapter bit-identical. Only a genuinely PAGED
    // grid switches to grid-derived page rects.
    if (grid_.pages_x <= 1 && grid_.pages_z <= 1) {
        return PageRect{-k_page_margin, -k_page_margin, world_width_ + k_page_margin, world_height_ + k_page_margin};
    }
    const glm::vec2 lo = grid_.page_min_xz(px, pz);
    const glm::vec2 hi = grid_.page_max_xz(px, pz);
    PageRect rect{lo.x, lo.y, hi.x, hi.y};
    // The margin ring belongs to the OUTSIDE of the world only: adding it at
    // an interior seam would overlap the neighbour's geometry and z-fight.
    if (px == 0) {
        rect.start_x -= k_page_margin;
    }
    if (pz == 0) {
        rect.start_z -= k_page_margin;
    }
    if (static_cast<uint32_t>(px) + 1 == grid_.pages_x) {
        rect.end_x += k_page_margin;
    }
    if (static_cast<uint32_t>(pz) + 1 == grid_.pages_z) {
        rect.end_z += k_page_margin;
    }
    return rect;
}

std::size_t TerrainRenderer::resident_pages() const {
    std::size_t count = 0;
    for (const TerrainPage& page : pages_) {
        if (page.has_buffers()) {
            ++count;
        }
    }
    return count;
}

std::size_t TerrainRenderer::page_bytes() const {
    std::size_t bytes = 0;
    for (const TerrainPage& page : pages_) {
        bytes += page.gpu_bytes;
    }
    return bytes + grass_proxy_bytes_;
}

void TerrainRenderer::update_splatmap(const uint8_t* data, uint32_t resolution) {
    if (!device_ || !data) {
        return;
    }

    // Recreate splatmap texture with new data
    splatmap_texture_ = gpu::GPUTexture::create_2d(*device_, resolution, resolution, gpu::TextureFormat::RGBA8, data,
                                                   false // No mipmaps for splatmap
    );

    if (!splatmap_texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "TerrainRenderer::update_splatmap: Failed to update splatmap texture");
    }
}

void TerrainRenderer::upload_heightmap_texture() {
    if (!device_) {
        return;
    }

    for (TerrainPage& page : pages_) {
        // Release the old texture first: a page being re-uploaded must not
        // hold two R16 surfaces at once on a streamed world.
        if (page.heightmap_texture) {
            page.gpu_bytes -= std::min(page.gpu_bytes, page.heightmap_texture_bytes);
        }
        page.heightmap_texture.reset();
        page.heightmap_texture_bytes = 0;

        const bracken::Heightmap* map = grid_.page(page.px, page.pz);
        if (map == nullptr) {
            continue; // page not resident — a flat placeholder mesh is drawn
        }

        // Create R16 texture for heightmap (16-bit unsigned normalized)
        // This preserves full precision from server-provided height data
        page.heightmap_texture = gpu::GPUTexture::create_2d(*device_, map->resolution, map->resolution,
                                                            gpu::TextureFormat::R16, map->height_data.data(),
                                                            false // No mipmaps for heightmap
        );

        if (!page.heightmap_texture) {
            BRACKEN_LOG_ERROR(LOG_TAG,
                              "TerrainRenderer::upload_heightmap_texture: Failed to create heightmap texture for page "
                              "({},{})",
                              page.px, page.pz);
            continue;
        }
        page.heightmap_texture_bytes = static_cast<std::size_t>(map->resolution) * map->resolution * sizeof(uint16_t);
        page.gpu_bytes += page.heightmap_texture_bytes;
    }

    rebuild_grass_heightmap();
}

void TerrainRenderer::rebuild_grass_heightmap() {
    grass_proxy_texture_.reset();
    grass_proxy_bytes_ = 0;
    grass_heightmap_texture_ = nullptr;
    grass_params_ = HeightmapSampleParams{};

    if (pages_.empty() || !device_) {
        return;
    }

    // 1x1: hand grass the page's own texture and the map's own five scalars —
    // byte for byte what SceneRenderer::set_heightmap did before paging.
    if (grid_.pages_x <= 1 && grid_.pages_z <= 1) {
        const bracken::Heightmap* map = grid_.page(0, 0);
        grass_heightmap_texture_ = pages_[0].heightmap_texture.get();
        if (map != nullptr) {
            grass_params_ = HeightmapSampleParams{map->world_origin_x, map->world_origin_z, map->world_size,
                                                  map->min_height, map->max_height};
        }
        return;
    }

    // Paged: grass_blade.hlsli's sampleTerrainHeight hard-codes ONE texture,
    // ONE origin and ONE square world size, so a paged grid gets a coarse
    // world-spanning PROXY resampled through the grid sampler. Grass follows
    // a smoothed terrain — the documented limitation of this landing, not an
    // accident. Grass only draws inside GrassRenderer's view distance, so the
    // error that matters is the near-field one, and near-field terrain is
    // exactly where a per-page texture would have full resolution; revisit
    // this when GrassVertexUniforms can afford a page index.
    const uint32_t page_res = grid_.page_resolution();
    if (page_res < 2) {
        return;
    }
    const uint32_t ideal = page_res * std::max(grid_.pages_x, grid_.pages_z);
    const uint32_t side = std::max(2u, std::min(ideal, k_grass_proxy_max_resolution));
    const float span = std::max(grid_.world_size_x(), grid_.world_size_z());
    const float range_span = grid_.range.max - grid_.range.min;

    std::vector<uint16_t> texels(static_cast<std::size_t>(side) * side);
    for (uint32_t iz = 0; iz < side; ++iz) {
        const float wz = grid_.origin_z + (static_cast<float>(iz) / static_cast<float>(side - 1)) * span;
        for (uint32_t ix = 0; ix < side; ++ix) {
            const float wx = grid_.origin_x + (static_cast<float>(ix) / static_cast<float>(side - 1)) * span;
            const float h = grid_.get_height_world(wx, wz, grid_.range.min);
            // The same packing Heightmap::from_floats uses, because the grass
            // shader decodes with the same raw*(max-min)+min formula.
            float normalized = range_span > 0.0f ? (h - grid_.range.min) / range_span : 0.0f;
            normalized = std::clamp(normalized, 0.0f, 1.0f);
            texels[static_cast<std::size_t>(iz) * side + ix] = static_cast<uint16_t>(normalized * 65535.0f);
        }
    }

    grass_proxy_texture_ =
        gpu::GPUTexture::create_2d(*device_, side, side, gpu::TextureFormat::R16, texels.data(), false);
    if (!grass_proxy_texture_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "TerrainRenderer: failed to create the {}x{} grass heightmap proxy", side, side);
        return;
    }
    grass_proxy_bytes_ = texels.size() * sizeof(uint16_t);
    grass_heightmap_texture_ = grass_proxy_texture_.get();
    grass_params_ = HeightmapSampleParams{grid_.origin_x, grid_.origin_z, span, grid_.range.min, grid_.range.max};
    BRACKEN_LOG_DEBUG(LOG_TAG, "TerrainRenderer: grass heightmap proxy {}x{} over {:.0f} world units", side, side,
                      span);
}

void TerrainRenderer::shutdown() {
    pages_.clear();
    grass_proxy_texture_.reset();
    grass_proxy_bytes_ = 0;
    grass_heightmap_texture_ = nullptr;

    // Release terrain textures
    material_array_texture_.reset();
    splatmap_texture_.reset();

    material_sampler_.reset();
    splatmap_sampler_.reset();
    grid_ = bracken::HeightmapGrid{};
    has_heightmap_ = false;
    device_ = nullptr;
    pipeline_registry_ = nullptr;
}

void TerrainRenderer::load_terrain_textures() {
    if (!device_) {
        return;
    }

    // Material art lives next to the executable (same anchoring as the
    // compiled shaders — SDL_GetBasePath), NOT the process CWD, so the app
    // finds its terrain regardless of where it is launched from.
    const char* base_path = SDL_GetBasePath();
    const std::string prefix = base_path ? base_path : "";
    const std::string texture_paths[4] = {prefix + "assets/textures/grass_seamless.png",
                                          prefix + "assets/textures/dirt_seamless.png",
                                          prefix + "assets/textures/rock_seamless.png",
                                          prefix + "assets/textures/sand_seamless.png"};

    SDL_Surface* loaded_surfaces[4] = {nullptr, nullptr, nullptr, nullptr};
    SDL_Surface* converted_surfaces[4] = {nullptr, nullptr, nullptr, nullptr};
    const void* layer_data[4] = {nullptr, nullptr, nullptr, nullptr};
    int tex_width = 0;
    int tex_height = 0;

    // Load all textures and convert to RGBA8
    bool all_loaded = true;
    for (int i = 0; i < 4; ++i) {
        loaded_surfaces[i] = IMG_Load(texture_paths[i].c_str());
        if (!loaded_surfaces[i]) {
            BRACKEN_LOG_ERROR(LOG_TAG, "TerrainRenderer::load_terrain_textures: Failed to load {}: {}",
                              texture_paths[i].c_str(), SDL_GetError());
            all_loaded = false;
            break;
        }

        // Convert to RGBA8 format
        converted_surfaces[i] = SDL_ConvertSurface(loaded_surfaces[i], SDL_PIXELFORMAT_RGBA32);
        if (!converted_surfaces[i]) {
            BRACKEN_LOG_ERROR(LOG_TAG, "TerrainRenderer::load_terrain_textures: Failed to convert {} to RGBA8: {}",
                              texture_paths[i].c_str(), SDL_GetError());
            all_loaded = false;
            break;
        }

        // Verify all textures have the same dimensions
        if (i == 0) {
            tex_width = converted_surfaces[i]->w;
            tex_height = converted_surfaces[i]->h;
        } else if (converted_surfaces[i]->w != tex_width || converted_surfaces[i]->h != tex_height) {
            BRACKEN_LOG_ERROR(LOG_TAG, "TerrainRenderer::load_terrain_textures: Texture size mismatch for {}",
                              texture_paths[i].c_str());
            all_loaded = false;
            break;
        }

        layer_data[i] = converted_surfaces[i]->pixels;
    }

    // Create Texture2DArray with all 4 material layers.
    // sRGB so the sampler linearizes albedo for lighting math; mipmapped so
    // distant terrain filters instead of shimmering.
    if (all_loaded && tex_width > 0 && tex_height > 0) {
        material_array_texture_ = gpu::GPUTexture::create_2d_array(*device_, tex_width, tex_height,
                                                                   4, // 4 layers: grass, dirt, rock, sand
                                                                   gpu::TextureFormat::RGBA8_SRGB, layer_data, true);

        if (!material_array_texture_) {
            BRACKEN_LOG_ERROR(LOG_TAG,
                              "TerrainRenderer::load_terrain_textures: Failed to create material array texture");
        }
    }

    // Clean up surfaces
    for (int i = 0; i < 4; ++i) {
        if (converted_surfaces[i]) {
            SDL_DestroySurface(converted_surfaces[i]);
        }
        if (loaded_surfaces[i]) {
            SDL_DestroySurface(loaded_surfaces[i]);
        }
    }

    // No material art on disk (the engine ships no assets): fall back to
    // procedural layers so terrain renders out of the box.
    if (!material_array_texture_) {
        create_fallback_material_array();
    }

    // Load splatmap texture (same executable-relative anchoring)
    splatmap_texture_ =
        gpu::GPUTexture::load_from_file(*device_, prefix + "assets/textures/terrain_splatmap.png", false);

    if (!splatmap_texture_) {
        create_fallback_splatmap();
    }
}

void TerrainRenderer::create_fallback_material_array() {
    // Small noise-shaded solid colors per layer: grass, dirt, rock, sand.
    constexpr uint32_t SIZE = 64;
    const std::array<std::array<uint8_t, 3>, 4> base_colors = {{
        {92, 138, 70},   // grass
        {126, 96, 66},   // dirt
        {126, 124, 120}, // rock
        {196, 178, 128}, // sand
    }};
    auto hash = [](uint32_t v) {
        v ^= v >> 16;
        v *= 0x7feb352dU;
        v ^= v >> 15;
        v *= 0x846ca68bU;
        v ^= v >> 16;
        return v;
    };
    std::array<std::vector<uint8_t>, 4> pixels;
    std::array<const void*, 4> layer_data{};
    for (uint32_t layer = 0; layer < 4; ++layer) {
        pixels[layer].resize(static_cast<size_t>(SIZE) * SIZE * 4);
        for (uint32_t y = 0; y < SIZE; ++y) {
            for (uint32_t x = 0; x < SIZE; ++x) {
                // Coarse patches plus per-pixel grain so the surface reads as
                // material rather than flat color.
                uint32_t coarse = hash((layer << 16) | ((y / 8) << 8) | (x / 8));
                uint32_t fine = hash((layer << 20) | (y << 10) | x);
                float brightness = 0.80f + (0.26f * static_cast<float>(coarse & 0xFF) / 255.0f) +
                                   (0.08f * static_cast<float>(fine & 0xFF) / 255.0f);
                size_t idx = ((static_cast<size_t>(y) * SIZE) + x) * 4;
                for (size_t c = 0; c < 3; ++c) {
                    float v = static_cast<float>(base_colors[layer][c]) * brightness;
                    pixels[layer][idx + c] = static_cast<uint8_t>(std::min(v, 255.0f));
                }
                pixels[layer][idx + 3] = 255;
            }
        }
        layer_data[layer] = pixels[layer].data();
    }
    material_array_texture_ = gpu::GPUTexture::create_2d_array(*device_, SIZE, SIZE, 4,
                                                               gpu::TextureFormat::RGBA8_SRGB, layer_data.data(), true);
    BRACKEN_LOG_WARN(LOG_TAG,
                     "TerrainRenderer: no material textures under assets/textures/, using procedural fallback");
}

void TerrainRenderer::create_fallback_splatmap() {
    // Neutral splat: full weight on layer 0 (grass) everywhere.
    constexpr uint32_t SIZE = 4;
    std::array<uint8_t, static_cast<size_t>(SIZE) * SIZE * 4> data{};
    for (size_t i = 0; i < data.size(); i += 4) {
        data[i] = 255;
    }
    splatmap_texture_ = gpu::GPUTexture::create_2d(*device_, SIZE, SIZE, gpu::TextureFormat::RGBA8, data.data());
    BRACKEN_LOG_WARN(LOG_TAG, "TerrainRenderer: no splatmap under assets/textures/, using all-grass fallback");
}

float TerrainRenderer::get_height(float x, float z) const {
    // Sample from the CPU-side page grid if any page is resident. For the 1x1
    // grid the adapter builds this IS Heightmap::get_height_world, with no
    // arithmetic in front of it.
    if (has_heightmap_) {
        return grid_.get_height_world(x, z);
    }
    // Fallback: return 0 (flat) if no heightmap yet
    return 0.0f;
}

glm::vec3 TerrainRenderer::get_normal(float x, float z) const {
    if (has_heightmap_) {
        return grid_.get_normal_world(x, z);
    }
    // Fallback: return up vector
    return {0.0f, 1.0f, 0.0f};
}

void TerrainRenderer::generate_terrain_mesh() {
    BRACKEN_PROFILE_ZONE_COLOR("TerrainRenderer::generate_terrain_mesh", 0x27AE60);
    if (!device_) {
        return;
    }
    for (TerrainPage& page : pages_) {
        generate_page_mesh(page);
    }
}

void TerrainRenderer::generate_page_mesh(TerrainPage& page) {
    if (!device_) {
        return;
    }

    // Reduced margin: 5000 added 10000 units of terrain ring outside the world
    // (400x wider x 400x taller at cell=25 = huge triangle count). 500 is enough
    // for visual continuity beyond world edge; tile culling hides the rest anyway.
    // Paged: the ring is applied only on the grid's OUTER edges — see
    // page_mesh_rect.
    const PageRect rect = page_mesh_rect(page.px, page.pz);
    float start_x = rect.start_x;
    float start_z = rect.start_z;
    float end_x = rect.end_x;
    float end_z = rect.end_z;

    float cell_size = 25.0f;
    int cells_x = static_cast<int>((end_x - start_x) / cell_size);
    int cells_z = static_cast<int>((end_z - start_z) / cell_size);

    // Chunk into tiles of TILE_CELLS × TILE_CELLS cells. At cell=25, tile=32 cells =
    // 800 world units. For a 42000-unit world this yields ~52x52 = ~2700 tiles;
    // typically only 5-15% are in the view frustum = 10–20x less geometry drawn.
    constexpr int TILE_CELLS = 32;
    int tiles_x = (cells_x + TILE_CELLS - 1) / TILE_CELLS;
    int tiles_z = (cells_z + TILE_CELLS - 1) / TILE_CELLS;

    float tex_scale = 0.01f;

    // One-shot setup: vertex+index buffers are built once, uploaded, and discarded.
    // A single Arena chunk replaces two heap-grown std::vectors.
    const int verts_x = cells_x + 1;
    const int verts_z = cells_z + 1;
    const size_t vertex_count_max = static_cast<size_t>(verts_x) * static_cast<size_t>(verts_z);
    const size_t index_count_max = static_cast<size_t>(cells_x) * static_cast<size_t>(cells_z) * 6;
    const size_t arena_bytes = vertex_count_max * (sizeof(TerrainVertex) + sizeof(float)) +
                               index_count_max * sizeof(uint32_t) + 4096;
    memory::Arena scratch(arena_bytes);

    auto* vertices =
        static_cast<TerrainVertex*>(scratch.allocate(vertex_count_max * sizeof(TerrainVertex), alignof(TerrainVertex)));
    auto* indices = static_cast<uint32_t*>(scratch.allocate(index_count_max * sizeof(uint32_t), alignof(uint32_t)));
    auto* heights = static_cast<float*>(scratch.allocate(vertex_count_max * sizeof(float), alignof(float)));
    size_t vertex_count = 0;
    size_t index_count = 0;

    // Pass 1: one bilinear heightmap sample per grid point. Normals are then
    // derived from grid neighbors instead of 4 extra bilinear samples each —
    // drops mesh generation from ~5 to 1 bilinear sample per vertex.
    for (int iz = 0; iz < verts_z; ++iz) {
        for (int ix = 0; ix < verts_x; ++ix) {
            heights[static_cast<size_t>(iz) * verts_x + ix] =
                get_height(start_x + ix * cell_size, start_z + iz * cell_size);
        }
    }

    for (int iz = 0; iz < verts_z; ++iz) {
        for (int ix = 0; ix < verts_x; ++ix) {
            float x = start_x + ix * cell_size;
            float z = start_z + iz * cell_size;
            float y = heights[static_cast<size_t>(iz) * verts_x + ix];

            // Central-difference normal from neighboring grid heights (clamped at edges).
            int ixl = std::max(ix - 1, 0);
            int ixr = std::min(ix + 1, cells_x);
            int izl = std::max(iz - 1, 0);
            int izr = std::min(iz + 1, cells_z);
            float hl = heights[static_cast<size_t>(iz) * verts_x + ixl];
            float hr = heights[static_cast<size_t>(iz) * verts_x + ixr];
            float hd = heights[static_cast<size_t>(izl) * verts_x + ix];
            float hu = heights[static_cast<size_t>(izr) * verts_x + ix];

            TerrainVertex vertex{};
            vertex.position = glm::vec3(x, y, z);
            vertex.texCoord = glm::vec2(x * tex_scale, z * tex_scale);
            vertex.normal = glm::normalize(glm::vec3((hl - hr) / (static_cast<float>(ixr - ixl) * cell_size), 1.0f,
                                                     (hd - hu) / (static_cast<float>(izr - izl) * cell_size)));

            float world_center_x = world_width_ / 2.0f;
            float world_center_z = world_height_ / 2.0f;
            float dx = x - world_center_x;
            float dz = z - world_center_z;
            float dist = std::sqrt(dx * dx + dz * dz);
            float dist_factor = std::min(dist / 3000.0f, 1.0f);
            float height_factor = std::min(std::max(y / 100.0f, 0.0f), 1.0f);
            float r = 0.95f + dist_factor * 0.05f;
            float g = 1.0f - dist_factor * 0.05f - height_factor * 0.05f;
            float b = 0.9f + dist_factor * 0.05f;
            vertex.color = glm::vec4(r, g, b, 1.0f);

            vertices[vertex_count++] = vertex;
        }
    }

    // Build indices per tile, track index ranges + AABBs for frustum culling.
    std::vector<TerrainTile>& tiles_out = page.tiles;
    tiles_out.clear();
    tiles_out.reserve(static_cast<size_t>(tiles_x) * tiles_z);
    for (int tz = 0; tz < tiles_z; ++tz) {
        for (int tx = 0; tx < tiles_x; ++tx) {
            int cell_x_start = tx * TILE_CELLS;
            int cell_z_start = tz * TILE_CELLS;
            int cell_x_end = std::min(cell_x_start + TILE_CELLS, cells_x);
            int cell_z_end = std::min(cell_z_start + TILE_CELLS, cells_z);

            TerrainTile tile;
            tile.first_index = static_cast<uint32_t>(index_count);

            float min_y = std::numeric_limits<float>::infinity();
            float max_y = -std::numeric_limits<float>::infinity();

            for (int iz = cell_z_start; iz < cell_z_end; ++iz) {
                for (int ix = cell_x_start; ix < cell_x_end; ++ix) {
                    uint32_t tl = iz * (cells_x + 1) + ix;
                    uint32_t tr = tl + 1;
                    uint32_t bl = (iz + 1) * (cells_x + 1) + ix;
                    uint32_t br = bl + 1;
                    indices[index_count++] = tl;
                    indices[index_count++] = bl;
                    indices[index_count++] = tr;
                    indices[index_count++] = tr;
                    indices[index_count++] = bl;
                    indices[index_count++] = br;

                    min_y = std::min({min_y, vertices[tl].position.y, vertices[tr].position.y, vertices[bl].position.y,
                                      vertices[br].position.y});
                    max_y = std::max({max_y, vertices[tl].position.y, vertices[tr].position.y, vertices[bl].position.y,
                                      vertices[br].position.y});
                }
            }

            tile.index_count = static_cast<uint32_t>(index_count) - tile.first_index;
            tile.aabb_min = glm::vec3(start_x + cell_x_start * cell_size, min_y, start_z + cell_z_start * cell_size);
            tile.aabb_max = glm::vec3(start_x + cell_x_end * cell_size, max_y, start_z + cell_z_end * cell_size);
            if (tile.index_count > 0) {
                tiles_out.push_back(tile);
            }
        }
    }
    build_tile_rows(core::jobs::JobSystem::instance(), tiles_out, page.rows);
    page.aabb_min = tile_bounds_min(tiles_out);
    page.aabb_max = tile_bounds_max(tiles_out);

    page.index_count = static_cast<uint32_t>(index_count);

    page.vertex_buffer = gpu::GPUBuffer::create_static(*device_, gpu::GPUBuffer::Type::Vertex, vertices,
                                                       vertex_count * sizeof(TerrainVertex));

    if (!page.vertex_buffer) {
        BRACKEN_LOG_ERROR(LOG_TAG, "TerrainRenderer::generate_terrain_mesh: Failed to create vertex buffer");
        return;
    }

    page.index_buffer =
        gpu::GPUBuffer::create_static(*device_, gpu::GPUBuffer::Type::Index, indices, index_count * sizeof(uint32_t));

    if (!page.index_buffer) {
        BRACKEN_LOG_ERROR(LOG_TAG, "TerrainRenderer::generate_terrain_mesh: Failed to create index buffer");
        page.vertex_buffer.reset();
        return;
    }
    page.gpu_bytes += vertex_count * sizeof(TerrainVertex) + index_count * sizeof(uint32_t);
}

void TerrainRenderer::render(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view,
                             const glm::mat4& projection, const glm::vec3& camera_pos, const glm::vec3& light_dir,
                             const SDL_GPUTextureSamplerBinding* shadow_bindings, int shadow_binding_count,
                             const SDL_GPUTextureSamplerBinding* ao_binding, glm::vec2 ao_texel,
                             const scene::Frustum* frustum, bool depth_equal,
                             const SDL_GPUTextureSamplerBinding* ibl_bindings, bool ibl_enabled,
                             const SDL_GPUTextureSamplerBinding* rt_shadow_binding, bool rt_shadows_enabled,
                             const SDL_GPUTextureSamplerBinding* rt_local_shadow_binding,
                             bool rt_local_shadows_enabled,
                             const SDL_GPUTextureSamplerBinding* local_shadow_binding,
                             const gi::GiBindings* gi) {
    BRACKEN_PROFILE_ZONE_COLOR("TerrainRenderer::render", 0xC0392B);
    // Skip rendering if required resources aren't available. The per-page
    // predicate is deliberately the pre-paging one (buffers, not index count)
    // so a 1x1 grid takes the identical early-out.
    if (!pipeline_registry_ || !pass || !cmd || !any_page(&TerrainPage::has_buffers)) {
        return;
    }
    if (!material_array_texture_ || !splatmap_texture_ || !material_sampler_) {
        // Once: same standing condition, evaluated every frame at 60 Hz.
        BRACKEN_LOG_WARN_ONCE(LOG_TAG, "TerrainRenderer::render: Terrain textures/sampler not ready, skipping");
        return;
    }

    // Get terrain pipeline (EQUAL/write-off variant after a depth prepass)
    auto* pipeline = pipeline_registry_->get_pipeline(depth_equal ? gpu::PipelineType::TerrainDepthEqual
                                                                  : gpu::PipelineType::Terrain);
    if (!pipeline) {
        BRACKEN_LOG_ERROR(LOG_TAG, "TerrainRenderer::render: Failed to get terrain pipeline");
        return;
    }

    // Bind pipeline
    pipeline->bind(pass);

    // Push vertex uniforms (transform data)
    TerrainTransformUniforms transform_uniforms{};
    transform_uniforms.view = view;
    transform_uniforms.projection = projection;
    transform_uniforms.cameraPos = camera_pos;
    transform_uniforms._padding0 = 0.0f;

    SDL_PushGPUVertexUniformData(cmd, 0, &transform_uniforms, sizeof(transform_uniforms));

    // Push fragment uniforms (lighting + environment data)
    TerrainLightingUniforms lighting_uniforms{};
    lighting_uniforms.fogColorHorizon = env_.fog_color_horizon;
    lighting_uniforms.fogStart = env_.fog_start;
    lighting_uniforms.fogColorSun = env_.fog_color_sun;
    lighting_uniforms.fogEnd = env_.fog_end;
    lighting_uniforms.lightDir = light_dir;
    lighting_uniforms.fogHeightFalloff = env_.fog_height_falloff;
    lighting_uniforms.cameraPos = camera_pos;
    lighting_uniforms.ambientSky = env_.ambient_sky;
    lighting_uniforms.ambientStrength = ambient_strength_;
    lighting_uniforms.ambientGround = env_.ambient_ground;
    lighting_uniforms.sunIntensity = sun_intensity_;
    lighting_uniforms.sunColor = env_.sun_color;
    lighting_uniforms.screenAOTexelX = ao_texel.x;
    lighting_uniforms.screenAOTexelY = ao_texel.y;
    // Splat region: the splatmap spans the world rect, anchored at the origin
    // (the terrain mesh itself is generated over [0, world] on both axes).
    // Separate X/Z sizes support non-square worlds.
    lighting_uniforms.splat_params =
        glm::vec4(0.0f, 0.0f, std::max(world_width_, 1.0f), std::max(world_height_, 1.0f));
    lighting_uniforms.iblEnabled = (ibl_enabled && ibl_bindings) ? 1.0f : 0.0f;
    lighting_uniforms.rtShadowEnabled =
        (rt_shadows_enabled && rt_shadow_binding) ? 1.0f : 0.0f;
    // 2 = Full-DI (R5c: the analytic point/spot loop does not run and the two
    // demodulated storage textures below supply the local lighting), 1 = Ratio,
    // 0 = the rasterized atlas. Gated on the bindings actually being there for
    // the same reason iblEnabled is.
    lighting_uniforms.rtLocalShadowMode =
        (local_di_active_ && local_di_diffuse_ && local_di_specular_ && local_di_direction_)
            ? 2.0f
        : ((rt_local_shadows_enabled && rt_local_shadow_binding) ? 1.0f : 0.0f);
    // Probe-field GI (R1c): the caller's one snapshot of the field. Gated on
    // the bindings actually being there for the same reason iblEnabled is —
    // a flag without its textures samples an undefined descriptor.
    if (gi != nullptr && gi->textures[0].texture != nullptr) {
        lighting_uniforms.giCascadeBase[0] = gi->cascade_base[0];
        lighting_uniforms.giCascadeBase[1] = gi->cascade_base[1];
        lighting_uniforms.giCascadeBase[2] = gi->cascade_base[2];
        lighting_uniforms.giEnabled = gi->enabled;
        // The R4b screen-probe tier, gated on its own texture for the same
        // reason: a flag without its atlas samples an undefined descriptor.
        if (gi->screen_sh != nullptr) {
            lighting_uniforms.giScreenEnabled = gi->screen_enabled;
            lighting_uniforms.giScreenProbesX = gi->screen_probes.x;
            lighting_uniforms.giScreenProbesY = gi->screen_probes.y;
        }
    }

    SDL_PushGPUFragmentUniformData(cmd, 0, &lighting_uniforms, sizeof(lighting_uniforms));

    // Bind material texture array at slot 0 (4 layers: grass, dirt, rock, sand)
    SDL_GPUTextureSamplerBinding material_binding = {material_array_texture_->handle(), material_sampler_->handle()};
    SDL_BindGPUFragmentSamplers(pass, 0, &material_binding, 1);

    // Bind splatmap texture at slot 1 (dedicated CLAMP sampler)
    SDL_GPUTextureSamplerBinding splatmap_binding = {splatmap_texture_->handle(), splatmap_sampler_->handle()};
    SDL_BindGPUFragmentSamplers(pass, 1, &splatmap_binding, 1);

    // Bind shadow atlas at slots 2-3 (comparison sampler + raw sampler)
    if (shadow_bindings && shadow_binding_count > 0) {
        SDL_BindGPUFragmentSamplers(pass, 2, shadow_bindings, shadow_binding_count);
    }

    // Bind screen-space AO at slot 4 (white dummy when AO is off).
    if (ao_binding) {
        SDL_BindGPUFragmentSamplers(pass, 4, ao_binding, 1);
    }

    // Bind the sky IBL chain at slots 5-7 (irradiance cube, prefiltered
    // specular cube, BRDF LUT — cube-typed dummies when IBL is off).
    if (ibl_bindings) {
        SDL_BindGPUFragmentSamplers(pass, 5, ibl_bindings, 3);
    }

    // Ray-traced sun visibility at slot 8. A white dummy is passed while RT
    // is disabled so the descriptor remains valid on every backend.
    if (rt_shadow_binding) {
        SDL_BindGPUFragmentSamplers(pass, 8, rt_shadow_binding, 1);
    }

    // Probe-field GI atlases at 9-11, the traced LOCAL-light visibility ratio
    // at 12, and the rasterized local-shadow ATLAS at 13 — which keeps its
    // contract of sitting at this shader's LAST sampler slot. R4b's
    // screen-probe SH rides the storage buffers below, not a sampler.
    if (gi != nullptr && gi->textures[0].texture != nullptr) {
        SDL_BindGPUFragmentSamplers(pass, 9, gi->textures.data(), 3);
    }
    if (rt_local_shadow_binding) {
        SDL_BindGPUFragmentSamplers(pass, 12, rt_local_shadow_binding, 1);
    }
    if (local_shadow_binding) {
        SDL_BindGPUFragmentSamplers(pass, 13, local_shadow_binding, 1);
    }

    if (cluster_light_data_ && cluster_offsets_ && cluster_indices_ && cluster_params_) {
        // 4th SSBO = local-shadow slot data; substitute the light-data buffer
        // when local shadows are unavailable (declared but never read while
        // every light's shadow_slot is -1).
        SDL_GPUBuffer* slots = cluster_local_shadow_slots_ ? cluster_local_shadow_slots_ : cluster_light_data_;
        // 5th SSBO = R4b's screen-probe SH (gi_screen_sample.hlsli). Same
        // substitution rule as the 4th: the slot must hold a valid buffer, and
        // giScreenEnabled == 0 is what stops it being read.
        SDL_GPUBuffer* screen_sh =
            (gi != nullptr && gi->screen_sh != nullptr) ? gi->screen_sh : cluster_light_data_;
        SDL_GPUBuffer* bufs[5] = {cluster_light_data_, cluster_offsets_, cluster_indices_, slots,
                                  screen_sh};
        SDL_BindGPUFragmentStorageBuffers(pass, 0, bufs, 5);
        SDL_PushGPUFragmentUniformData(cmd, 2, cluster_params_, static_cast<Uint32>(cluster_params_size_));
    }

    // R5c's Full-DI pair plus R7b's direction, as READ-ONLY STORAGE textures.
    // Bound unconditionally — an unbound storage-texture descriptor is
    // undefined, so the caller hands over its 1x1 stand-ins when the tier is off
    // and `rt_local_shadow_mode < 2` is what stops the shader reading them.
    if (local_di_diffuse_ != nullptr && local_di_specular_ != nullptr &&
        local_di_direction_ != nullptr) {
        SDL_GPUTexture* di[3] = {local_di_diffuse_, local_di_specular_, local_di_direction_};
        SDL_BindGPUFragmentStorageTextures(pass, 0, di, 3);
    }

    // Tile-culled draw, per page: the page's own AABB is rejected first (one
    // test that skips a whole page's buffers and cull), then whole rows are
    // AABB-rejected, then surviving tiles are tested individually. Visible
    // tiles with contiguous index ranges (tiles are stored row-major,
    // back-to-back in the page's index buffer) merge into a single draw —
    // typically collapsing 90-260 per-tile draws into ~10-30 row runs. Runs
    // never merge ACROSS pages: different index buffers.
    // Fallback: if no frustum provided or no tiles, draw the whole page mesh.
    for (const TerrainPage& page : pages_) {
        if (!page.has_buffers() || page.index_count == 0) {
            continue;
        }
        const bool cull = (frustum != nullptr) && !page.tiles.empty();
        if (cull && !frustum->intersects_aabb(page.aabb_min, page.aabb_max)) {
            continue;
        }

        SDL_GPUBufferBinding vb_binding = {page.vertex_buffer->handle(), 0};
        SDL_BindGPUVertexBuffers(pass, 0, &vb_binding, 1);

        SDL_GPUBufferBinding ib_binding = {page.index_buffer->handle(), 0};
        SDL_BindGPUIndexBuffer(pass, &ib_binding, SDL_GPU_INDEXELEMENTSIZE_32BIT);

        if (cull) {
            draw_culled_tiles(core::jobs::JobSystem::instance(), pass, *frustum, page.tiles, page.rows);
        } else {
            SDL_DrawGPUIndexedPrimitives(pass, page.index_count, 1, 0, 0, 0);
        }
    }
}

void TerrainRenderer::build_tile_rows(core::jobs::JobSystem& js, const std::vector<TerrainTile>& tiles,
                                      std::vector<TileRow>& rows) {
    rows.clear();
    if (tiles.empty()) {
        return;
    }

    // Deterministic chunked fan-out (multithreading_plan §4.3/§6.2): tiles
    // are grouped into rows per chunk (each chunk writes only its OWN
    // cache-line-aligned partial), then partials merge single-threaded in
    // chunk-index order, coalescing a z-band row that spans a chunk seam.
    // Chunk boundaries depend only on the grain, so the result is identical
    // to the serial scan at any worker count (the 1-worker / uninitialized
    // pool runs the same lambda as one chunk into partials[0]).
    const std::size_t grain = ROW_BUILD_TILE_GRAIN;
    const std::size_t fanout_chunks = (tiles.size() + grain - 1) / grain;

    struct alignas(core::jobs::k_cache_line) RowPartial {
        std::vector<TileRow> rows;
    };
    std::vector<RowPartial> partials(fanout_chunks);

    core::jobs::parallel_for(
        js, 0, tiles.size(),
        [&](std::size_t begin, std::size_t end) {
            const std::size_t chunk_index = begin / grain;
            BRACKEN_ASSERT(chunk_index < partials.size());
            std::vector<TileRow>& out = partials[chunk_index].rows;
            float row_z = 0.0f;
            for (std::size_t i = begin; i < end; ++i) {
                const TerrainTile& tile = tiles[i];
                // Tiles are generated row-major, so tiles sharing a min-z band
                // are contiguous; a z change starts a new row.
                if (out.empty() || tile.aabb_min.z != row_z) {
                    out.push_back({i, 0, tile.aabb_min, tile.aabb_max});
                    row_z = tile.aabb_min.z;
                }
                TileRow& row = out.back();
                ++row.tile_count;
                row.aabb_min = glm::min(row.aabb_min, tile.aabb_min);
                row.aabb_max = glm::max(row.aabb_max, tile.aabb_max);
            }
        },
        grain);

    // Single-threaded merge in chunk-index order. Every tile in a row shares
    // the row's aabb_min.z (that's the grouping key), so a partial's first
    // row continues the previous chunk's last row iff their z bands match.
    for (const RowPartial& partial : partials) {
        for (const TileRow& row : partial.rows) {
            if (!rows.empty() && row.aabb_min.z == rows.back().aabb_min.z) {
                TileRow& prev = rows.back();
                prev.tile_count += row.tile_count;
                prev.aabb_min = glm::min(prev.aabb_min, row.aabb_min);
                prev.aabb_max = glm::max(prev.aabb_max, row.aabb_max);
            } else {
                rows.push_back(row);
            }
        }
    }
}

void TerrainRenderer::cull_tile_runs(core::jobs::JobSystem& js, const scene::Frustum& frustum,
                                     const std::vector<TerrainTile>& tiles, const std::vector<TileRow>& rows,
                                     std::vector<TileRun>& out_runs) {
    out_runs.clear();

    // Appends a visible tile to a run list, extending the tail run when the
    // tile's index range is contiguous with it (tiles are stored row-major,
    // back-to-back in the index buffer).
    auto merge_tile = [](std::vector<TileRun>& runs, const TerrainTile& tile) {
        if (!runs.empty() && tile.first_index == runs.back().first_index + runs.back().index_count) {
            runs.back().index_count += tile.index_count;
        } else {
            runs.push_back({tile.first_index, tile.index_count});
        }
    };

    if (rows.empty()) {
        // No row acceleration (unbuilt rows): plain serial tile walk.
        for (const TerrainTile& tile : tiles) {
            if (frustum.intersects_aabb(tile.aabb_min, tile.aabb_max)) {
                merge_tile(out_runs, tile);
            }
        }
        return;
    }

    // Deterministic chunked fan-out over tile rows (multithreading_plan
    // §4.3/§6.2): each chunk culls its rows into its OWN cache-line-aligned
    // partial run list; the contiguous-run merge below consumes partials
    // single-threaded in chunk-index order, coalescing runs across chunk
    // seams. Chunk boundaries depend only on the grain, so the run list is
    // bit-identical to the serial walk at any worker count (1-worker /
    // uninitialized pools run the same lambda as one chunk — granularity
    // control, not a second code path).
    const std::size_t grain = CULL_ROW_GRAIN;
    const std::size_t fanout_chunks = (rows.size() + grain - 1) / grain;

    struct alignas(core::jobs::k_cache_line) RunPartial {
        std::vector<TileRun> runs;
    };
    std::vector<RunPartial> partials(fanout_chunks);

    core::jobs::parallel_for(
        js, 0, rows.size(),
        [&](std::size_t begin, std::size_t end) {
            const std::size_t chunk_index = begin / grain;
            BRACKEN_ASSERT(chunk_index < partials.size());
            std::vector<TileRun>& runs = partials[chunk_index].runs;
            for (std::size_t r = begin; r < end; ++r) {
                const TileRow& row = rows[r];
                // One AABB test skips the row's whole tile span when it lies
                // outside the frustum (the common case: 80-95% of the world).
                if (!frustum.intersects_aabb(row.aabb_min, row.aabb_max)) {
                    continue;
                }
                const std::size_t row_end = row.first_tile + row.tile_count;
                for (std::size_t i = row.first_tile; i < row_end; ++i) {
                    const TerrainTile& tile = tiles[i];
                    if (frustum.intersects_aabb(tile.aabb_min, tile.aabb_max)) {
                        merge_tile(runs, tile);
                    }
                }
            }
        },
        grain);

    // Single-threaded merge in chunk-index order; merge_tile-style coalescing
    // across the seam reproduces the serial run structure exactly (runs
    // within a partial are non-contiguous by construction, so only seam runs
    // can fuse).
    for (const RunPartial& partial : partials) {
        for (const TileRun& run : partial.runs) {
            if (!out_runs.empty() && run.first_index == out_runs.back().first_index + out_runs.back().index_count) {
                out_runs.back().index_count += run.index_count;
            } else {
                out_runs.push_back(run);
            }
        }
    }
}

void TerrainRenderer::draw_culled_tiles(core::jobs::JobSystem& js, SDL_GPURenderPass* pass,
                                        const scene::Frustum& frustum, const std::vector<TerrainTile>& tiles,
                                        const std::vector<TileRow>& rows) {
    cull_tile_runs(js, frustum, tiles, rows, tile_runs_scratch_);
    for (const TileRun& run : tile_runs_scratch_) {
        SDL_DrawGPUIndexedPrimitives(pass, run.index_count, 1, run.first_index, 0, 0);
    }
}

void TerrainRenderer::set_anisotropic_filter(float level) {
    if (!device_) {
        return;
    }

    // Clamp level to reasonable range (1.0 = no anisotropy, 16.0 = max typical)
    level = std::max(1.0f, std::min(16.0f, level));

    // Recreate the terrain material-array sampler with the new anisotropy level
    material_sampler_ = gpu::GPUSampler::create(*device_, gpu::SamplerConfig::anisotropic(level));
    if (!material_sampler_) {
        BRACKEN_LOG_ERROR(
            LOG_TAG, "TerrainRenderer::set_anisotropic_filter: Failed to recreate material sampler with level {:.1f}",
            level);
    }
}

void TerrainRenderer::set_cluster_lighting(SDL_GPUBuffer* light_data, SDL_GPUBuffer* cluster_offsets,
                                           SDL_GPUBuffer* light_indices, const void* params, size_t params_size,
                                           SDL_GPUBuffer* local_shadow_slots) {
    cluster_light_data_ = light_data;
    cluster_offsets_ = cluster_offsets;
    cluster_indices_ = light_indices;
    cluster_params_ = params;
    cluster_params_size_ = params_size;
    cluster_local_shadow_slots_ = local_shadow_slots;
}

void TerrainRenderer::render_prepass(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                                     const gpu::PrepassCameraUniforms& uniforms, gpu::PipelineType pipeline_type,
                                     const scene::Frustum* frustum) {
    BRACKEN_PROFILE_ZONE_COLOR("TerrainRenderer::render_prepass", 0xC0392B);
    if (!pipeline_registry_ || !pass || !cmd || !any_page(&TerrainPage::drawable)) {
        return;
    }

    auto* pipeline = pipeline_registry_->get_pipeline(pipeline_type);
    if (!pipeline) {
        return;
    }

    pipeline->bind(pass);
    SDL_PushGPUVertexUniformData(cmd, 0, &uniforms, sizeof(uniforms));

    // Full-LOD mesh + the same page reject and tile culling as render(): the
    // forward pass depth-tests EQUAL, so the prepass must cover exactly the
    // same geometry — which is why the page reject has to be the identical
    // predicate on the identical AABB, not an approximation of it.
    for (const TerrainPage& page : pages_) {
        if (!page.drawable()) {
            continue;
        }
        const bool cull = (frustum != nullptr) && !page.tiles.empty();
        if (cull && !frustum->intersects_aabb(page.aabb_min, page.aabb_max)) {
            continue;
        }

        SDL_GPUBufferBinding vb_binding = {page.vertex_buffer->handle(), 0};
        SDL_BindGPUVertexBuffers(pass, 0, &vb_binding, 1);

        SDL_GPUBufferBinding ib_binding = {page.index_buffer->handle(), 0};
        SDL_BindGPUIndexBuffer(pass, &ib_binding, SDL_GPU_INDEXELEMENTSIZE_32BIT);

        if (cull) {
            draw_culled_tiles(core::jobs::JobSystem::instance(), pass, *frustum, page.tiles, page.rows);
        } else {
            SDL_DrawGPUIndexedPrimitives(pass, page.index_count, 1, 0, 0, 0);
        }
    }
}

void TerrainRenderer::render_shadow(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd,
                                    const glm::mat4& light_view_projection, const scene::Frustum* frustum) {
    BRACKEN_PROFILE_ZONE_COLOR("TerrainRenderer::render_shadow", 0xE67E22);
    if (!pipeline_registry_ || !pass || !cmd || !any_page(&TerrainPage::shadow_drawable)) {
        return;
    }

    auto* pipeline = pipeline_registry_->get_pipeline(gpu::PipelineType::ShadowTerrain);
    if (!pipeline) {
        return;
    }

    pipeline->bind(pass);

    gpu::ShadowTerrainUniforms shadow_uniforms = {};
    shadow_uniforms.lightViewProjection = light_view_projection;
    SDL_PushGPUVertexUniformData(cmd, 0, &shadow_uniforms, sizeof(shadow_uniforms));

    // Per-page, per-cascade tile culling: cascade 0 (tight near) draws maybe
    // 2-5 tiles, cascade 3 (far) draws many but still skips half the world.
    // The page AABB is rejected first, then rows, then tiles; contiguous
    // visible runs merge into single draws (same as the main pass).
    for (const TerrainPage& page : pages_) {
        // Prefer the low-LOD shadow mesh (4x coarser -> 16x fewer triangles
        // per cascade). Decided per page, which for a 1x1 grid is the same
        // decision the pre-paging code made once.
        const bool using_shadow_mesh =
            (page.shadow_vertex_buffer && page.shadow_index_buffer && page.shadow_index_count > 0);
        gpu::GPUBuffer* vb = using_shadow_mesh ? page.shadow_vertex_buffer.get() : page.vertex_buffer.get();
        gpu::GPUBuffer* ib = using_shadow_mesh ? page.shadow_index_buffer.get() : page.index_buffer.get();
        const uint32_t idx_count = using_shadow_mesh ? page.shadow_index_count : page.index_count;
        const std::vector<TerrainTile>& tile_list = using_shadow_mesh ? page.shadow_tiles : page.tiles;
        const std::vector<TileRow>& rows = using_shadow_mesh ? page.shadow_rows : page.rows;
        if (!vb || !ib || idx_count == 0) {
            continue;
        }

        const bool cull = (frustum != nullptr) && !tile_list.empty();
        if (cull) {
            const glm::vec3& lo = using_shadow_mesh ? page.shadow_aabb_min : page.aabb_min;
            const glm::vec3& hi = using_shadow_mesh ? page.shadow_aabb_max : page.aabb_max;
            if (!frustum->intersects_aabb(lo, hi)) {
                continue;
            }
        }

        SDL_GPUBufferBinding vb_binding = {vb->handle(), 0};
        SDL_BindGPUVertexBuffers(pass, 0, &vb_binding, 1);

        SDL_GPUBufferBinding ib_binding = {ib->handle(), 0};
        SDL_BindGPUIndexBuffer(pass, &ib_binding, SDL_GPU_INDEXELEMENTSIZE_32BIT);

        if (cull) {
            draw_culled_tiles(core::jobs::JobSystem::instance(), pass, *frustum, tile_list, rows);
        } else {
            SDL_DrawGPUIndexedPrimitives(pass, idx_count, 1, 0, 0, 0);
        }
    }
}

bool TerrainRenderer::any_page(bool (TerrainPage::*predicate)() const) const {
    for (const TerrainPage& page : pages_) {
        if ((page.*predicate)()) {
            return true;
        }
    }
    return false;
}

void TerrainRenderer::generate_shadow_terrain_mesh() {
    BRACKEN_PROFILE_ZONE_COLOR("TerrainRenderer::generate_shadow_terrain_mesh", 0xE67E22);
    if (!device_) {
        return;
    }
    std::size_t total_shadow_indices = 0;
    std::size_t total_shadow_tiles = 0;
    std::size_t total_main_indices = 0;
    std::size_t total_main_tiles = 0;
    for (TerrainPage& page : pages_) {
        generate_page_shadow_mesh(page);
        total_shadow_indices += page.shadow_index_count;
        total_shadow_tiles += page.shadow_tiles.size();
        total_main_indices += page.index_count;
        total_main_tiles += page.tiles.size();
    }
    BRACKEN_LOG_INFO(LOG_TAG,
                     "TerrainRenderer: {} page(s), shadow {} tiles -> {} triangles (main: {} tris, {} tiles)",
                     pages_.size(), total_shadow_tiles, total_shadow_indices / 3, total_main_indices / 3,
                     total_main_tiles);
}

void TerrainRenderer::generate_page_shadow_mesh(TerrainPage& page) {
    if (!device_) {
        return;
    }

    // Coarser mesh: 4x the cell size of the main terrain (25 -> 100).
    // Shadow maps are 2048 at most; 100-unit cell spacing is finer than the
    // per-texel shadow resolution at typical camera ranges, so visual impact
    // is imperceptible while triangle count drops 16x.
    constexpr float kShadowCellSize = 100.0f;
    constexpr int TILE_CELLS = 16; // 16 cells @ 100 units = 1600 world units per tile

    // The same rect as the main mesh — the margin ring is k_page_margin on
    // both, which is what the two private 500s were before paging.
    const PageRect rect = page_mesh_rect(page.px, page.pz);
    float start_x = rect.start_x;
    float start_z = rect.start_z;
    float end_x = rect.end_x;
    float end_z = rect.end_z;

    int cells_x = static_cast<int>((end_x - start_x) / kShadowCellSize);
    int cells_z = static_cast<int>((end_z - start_z) / kShadowCellSize);
    if (cells_x <= 0 || cells_z <= 0) {
        return;
    }

    int tiles_x = (cells_x + TILE_CELLS - 1) / TILE_CELLS;
    int tiles_z = (cells_z + TILE_CELLS - 1) / TILE_CELLS;

    std::vector<TerrainVertex> vertices;
    std::vector<uint32_t> indices;
    vertices.reserve(static_cast<size_t>(cells_x + 1) * (cells_z + 1));

    // Conservative MIN sample: each shadow vertex takes the minimum of the
    // high-detail heightmap over its cell neighborhood (±half-cell). Keeps
    // the simplified shadow surface at-or-below real terrain everywhere,
    // preventing phantom shadow blobs in sub-cell depressions. Small downward
    // sink covers floating-point + interpolation slack across the planar quad.
    constexpr float PROBE_STEP = 25.0f;
    constexpr int PROBE_RADIUS = 2;
    constexpr float SHADOW_SINK = 5.0f;

    // Probe grid sampled once (one bilinear per point), then each vertex takes
    // the min over its 5x5 window of plain array reads — instead of 25 bilinear
    // heightmap samples per vertex.
    constexpr int FINE_PER_CELL = static_cast<int>(kShadowCellSize / PROBE_STEP);
    const int fine_nx = cells_x * FINE_PER_CELL + 2 * PROBE_RADIUS + 1;
    const int fine_nz = cells_z * FINE_PER_CELL + 2 * PROBE_RADIUS + 1;
    std::vector<float> fine_heights(static_cast<size_t>(fine_nx) * fine_nz);
    for (int fz = 0; fz < fine_nz; ++fz) {
        float z = start_z + static_cast<float>(fz - PROBE_RADIUS) * PROBE_STEP;
        for (int fx = 0; fx < fine_nx; ++fx) {
            float x = start_x + static_cast<float>(fx - PROBE_RADIUS) * PROBE_STEP;
            fine_heights[static_cast<size_t>(fz) * fine_nx + fx] = get_height(x, z);
        }
    }

    for (int iz = 0; iz <= cells_z; ++iz) {
        for (int ix = 0; ix <= cells_x; ++ix) {
            float x = start_x + ix * kShadowCellSize;
            float z = start_z + iz * kShadowCellSize;
            int cfx = ix * FINE_PER_CELL + PROBE_RADIUS;
            int cfz = iz * FINE_PER_CELL + PROBE_RADIUS;
            float y = std::numeric_limits<float>::infinity();
            for (int dz = -PROBE_RADIUS; dz <= PROBE_RADIUS; ++dz) {
                for (int dx = -PROBE_RADIUS; dx <= PROBE_RADIUS; ++dx) {
                    y = std::min(y, fine_heights[static_cast<size_t>(cfz + dz) * fine_nx + (cfx + dx)]);
                }
            }
            TerrainVertex v{};
            v.position = glm::vec3(x, y - SHADOW_SINK, z);
            vertices.push_back(v);
        }
    }

    std::vector<TerrainTile>& shadow_tiles_out = page.shadow_tiles;
    shadow_tiles_out.clear();
    shadow_tiles_out.reserve(static_cast<size_t>(tiles_x) * tiles_z);
    for (int tz = 0; tz < tiles_z; ++tz) {
        for (int tx = 0; tx < tiles_x; ++tx) {
            int cx0 = tx * TILE_CELLS;
            int cz0 = tz * TILE_CELLS;
            int cx1 = std::min(cx0 + TILE_CELLS, cells_x);
            int cz1 = std::min(cz0 + TILE_CELLS, cells_z);

            TerrainTile tile;
            tile.first_index = static_cast<uint32_t>(indices.size());
            float min_y = std::numeric_limits<float>::infinity();
            float max_y = -std::numeric_limits<float>::infinity();

            for (int iz = cz0; iz < cz1; ++iz) {
                for (int ix = cx0; ix < cx1; ++ix) {
                    uint32_t tl = iz * (cells_x + 1) + ix;
                    uint32_t tr = tl + 1;
                    uint32_t bl = tl + (cells_x + 1);
                    uint32_t br = bl + 1;
                    indices.push_back(tl);
                    indices.push_back(bl);
                    indices.push_back(tr);
                    indices.push_back(tr);
                    indices.push_back(bl);
                    indices.push_back(br);
                    min_y = std::min({min_y, vertices[tl].position.y, vertices[tr].position.y, vertices[bl].position.y,
                                      vertices[br].position.y});
                    max_y = std::max({max_y, vertices[tl].position.y, vertices[tr].position.y, vertices[bl].position.y,
                                      vertices[br].position.y});
                }
            }
            tile.index_count = static_cast<uint32_t>(indices.size()) - tile.first_index;
            tile.aabb_min = glm::vec3(start_x + cx0 * kShadowCellSize, min_y, start_z + cz0 * kShadowCellSize);
            tile.aabb_max = glm::vec3(start_x + cx1 * kShadowCellSize, max_y, start_z + cz1 * kShadowCellSize);
            if (tile.index_count > 0) {
                shadow_tiles_out.push_back(tile);
            }
        }
    }
    build_tile_rows(core::jobs::JobSystem::instance(), shadow_tiles_out, page.shadow_rows);
    page.shadow_aabb_min = tile_bounds_min(shadow_tiles_out);
    page.shadow_aabb_max = tile_bounds_max(shadow_tiles_out);

    page.shadow_vertex_buffer = gpu::GPUBuffer::create_static(*device_, gpu::GPUBuffer::Type::Vertex, vertices.data(),
                                                              vertices.size() * sizeof(TerrainVertex));
    page.shadow_index_buffer = gpu::GPUBuffer::create_static(*device_, gpu::GPUBuffer::Type::Index, indices.data(),
                                                             indices.size() * sizeof(uint32_t));
    page.shadow_index_count = static_cast<uint32_t>(indices.size());
    page.gpu_bytes += vertices.size() * sizeof(TerrainVertex) + indices.size() * sizeof(uint32_t);
}

} // namespace bracken::render
