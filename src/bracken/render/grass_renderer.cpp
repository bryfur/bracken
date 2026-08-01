#include "grass_renderer.hpp"
#include "bracken/contracts/frustum.hpp"
#include "bracken/core/assert.hpp"
#include "bracken/core/jobs/cache_line.hpp"
#include "bracken/core/jobs/job_system.hpp"
#include "bracken/core/jobs/parallel_for.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_types.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "SDL3/SDL_gpu.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace bracken::render {

namespace {
constexpr const char* LOG_TAG = "grass";
} // namespace

namespace gpu = bracken::gpu;

GrassRenderer::GrassRenderer() = default;

GrassRenderer::~GrassRenderer() {
    shutdown();
}

bool GrassRenderer::init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry, float world_width,
                         float world_height) {
    if (initialized_) {
        return true;
    }

    BRACKEN_LOG_INFO(LOG_TAG, "Initializing grass renderer (chunked instanced, AAA-quality)...");

    device_ = &device;
    pipeline_registry_ = &pipeline_registry;
    world_width_ = world_width;
    world_height_ = world_height;

    // Heightmap sampler: linear filtering for smooth terrain-height transitions.
    heightmap_sampler_ = gpu::GPUSampler::create(device, gpu::SamplerConfig::linear_clamp());
    if (!heightmap_sampler_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GrassRenderer::init: Failed to create heightmap sampler");
        return false;
    }

    generate_blade_mesh();
    if (!blade_vertex_buffer_ || !blade_index_buffer_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GrassRenderer::init: Failed to create blade mesh");
        return false;
    }

    // Pre-size chunk storage buffer for a reasonable upper bound of visible chunks.
    // At view_dist=1200 and chunk=32, (1200/32)^2*pi ≈ 4400 candidate chunks;
    // usually 1000-2500 visible after frustum cull. 8192 * vec4 = 128 KB.
    constexpr size_t initial_capacity_chunks = 8192;
    chunk_storage_capacity_bytes_ = initial_capacity_chunks * sizeof(glm::vec4);
    chunk_storage_buffer_ =
        gpu::GPUBuffer::create_dynamic(device, gpu::GPUBuffer::Type::Storage, chunk_storage_capacity_bytes_);
    if (!chunk_storage_buffer_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GrassRenderer::init: Failed to create chunk storage buffer");
        return false;
    }
    visible_chunks_scratch_.reserve(initial_capacity_chunks);

    initialized_ = true;
    BRACKEN_LOG_INFO(LOG_TAG, "Grass renderer initialized (blade mesh: {} indices, chunks: {}x{}, blades/chunk: {})",
                     blade_index_count_, BLADES_PER_CHUNK_SIDE, BLADES_PER_CHUNK_SIDE, BLADES_PER_CHUNK_SQ);
    return true;
}

void GrassRenderer::update(float current_time) {
    current_time_ = current_time;
}

void GrassRenderer::generate_blade_mesh() {
    // Unit blade: base at Y=0, tip at Y=1, width tapers toward tip.
    // 4 segments = 5 rows of 2 vertices = 10 vertices, 8 triangles.
    const int segments = 4;
    const float base_width = 0.5f;

    std::vector<gpu::Vertex3D> vertices;
    std::vector<uint32_t> indices;

    // Base-to-tip color gradient (linear). Tuned toward the terrain grass
    // layer's meadow green so blades read as part of the ground, not on it.
    glm::vec4 base_color(0.05f, 0.13f, 0.03f, 1.0f);
    glm::vec4 tip_color(0.18f, 0.44f, 0.10f, 1.0f);

    for (int i = 0; i <= segments; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(segments);
        float w = base_width * (1.0f - t * 0.9f);
        float y = t;

        glm::vec3 normal(0.0f, 0.0f, 1.0f);
        glm::vec4 color = glm::mix(base_color, tip_color, t);

        gpu::Vertex3D left;
        left.position = glm::vec3(-w, y, 0.0f);
        left.normal = normal;
        left.texcoord = glm::vec2(0.0f, t);
        left.color = color;
        vertices.push_back(left);

        gpu::Vertex3D right;
        right.position = glm::vec3(w, y, 0.0f);
        right.normal = normal;
        right.texcoord = glm::vec2(1.0f, t);
        right.color = color;
        vertices.push_back(right);
    }

    for (int i = 0; i < segments; ++i) {
        uint32_t base = i * 2;
        indices.push_back(base);
        indices.push_back(base + 2);
        indices.push_back(base + 1);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
    }

    blade_vertex_buffer_ = gpu::GPUBuffer::create_static(*device_, gpu::GPUBuffer::Type::Vertex, vertices.data(),
                                                         vertices.size() * sizeof(gpu::Vertex3D));

    blade_index_buffer_ = gpu::GPUBuffer::create_static(*device_, gpu::GPUBuffer::Type::Index, indices.data(),
                                                        indices.size() * sizeof(uint32_t));

    blade_index_count_ = static_cast<uint32_t>(indices.size());
}

void GrassRenderer::cull_chunks(core::jobs::JobSystem& js, const ChunkCullParams& params,
                                std::span<const GrassExclusionRect> exclusions, const scene::Frustum* frustum,
                                std::vector<glm::vec4>& out_chunks) {
    const float chunk_size = CHUNK_SIZE;
    const float view_dist = params.view_distance;
    const float view_dist_sq = view_dist * view_dist;

    const float cam_chunk_x = std::floor(params.camera_pos.x / chunk_size) * chunk_size;
    const float cam_chunk_z = std::floor(params.camera_pos.z / chunk_size) * chunk_size;
    const int half_extent = static_cast<int>(std::ceil(view_dist / chunk_size)) + 1;

    const float margin = std::max(params.edge_margin, 0.0f);

    // Serial body over a contiguous band of grid rows [row_begin, row_end),
    // appending survivors in row-major order (the serial iteration order).
    auto cull_rows = [&](std::size_t row_begin, std::size_t row_end, std::vector<glm::vec4>& out) {
        for (std::size_t row = row_begin; row < row_end; ++row) {
            const int iz = static_cast<int>(row) - half_extent;
            for (int ix = -half_extent; ix <= half_extent; ++ix) {
                float ox = cam_chunk_x + static_cast<float>(ix) * chunk_size;
                float oz = cam_chunk_z + static_cast<float>(iz) * chunk_size;
                float cx = ox + chunk_size * 0.5f;
                float cz = oz + chunk_size * 0.5f;

                float dx = cx - params.camera_pos.x;
                float dz = cz - params.camera_pos.z;
                float dist_sq = dx * dx + dz * dz;
                if (dist_sq > view_dist_sq) {
                    continue;
                }

                // World-bounds cull: skip chunks with no overlap of the grassable
                // rect [margin, world - margin] on both axes. With the default
                // margin of 0 this only rejects chunks fully outside the world.
                if (ox + chunk_size < margin || ox > params.world_width - margin || oz + chunk_size < margin ||
                    oz > params.world_height - margin) {
                    continue;
                }

                // Host-configured exclusion zones (chunk-granular, center test).
                bool excluded = false;
                for (const GrassExclusionRect& rect : exclusions) {
                    if (cx >= rect.min_x && cx <= rect.max_x && cz >= rect.min_z && cz <= rect.max_z) {
                        excluded = true;
                        break;
                    }
                }
                if (excluded) {
                    continue;
                }

                if (frustum) {
                    glm::vec3 aabb_min(ox, params.y_min, oz);
                    glm::vec3 aabb_max(ox + chunk_size, params.y_max, oz + chunk_size);
                    if (!frustum->intersects_aabb(aabb_min, aabb_max)) {
                        continue;
                    }
                }

                out.emplace_back(ox, 0.0f, oz, 0.0f);
            }
        }
    };

    // Deterministic chunked fan-out (multithreading_plan §4.3/§6.2): each
    // fan-out chunk appends into its OWN partial (headers cache-line-aligned —
    // no shared writes inside the per-chunk loop), then partials merge
    // single-threaded in chunk-index order below. Chunk boundaries depend only
    // on the grain, never on worker count or thread ids, so the merged output
    // is bit-identical to the serial walk at any worker count (including the
    // 1-worker / uninitialized-pool inline path, which runs as one chunk into
    // partials[0] — same code path, granularity control only).
    const std::size_t row_count = static_cast<std::size_t>(2 * half_extent + 1);
    const std::size_t grain = CULL_ROW_GRAIN;
    const std::size_t fanout_chunks = (row_count + grain - 1) / grain;

    struct alignas(core::jobs::k_cache_line) RowPartial {
        std::vector<glm::vec4> chunks;
    };
    std::vector<RowPartial> partials(fanout_chunks);

    core::jobs::parallel_for(
        js, 0, row_count,
        [&](std::size_t begin, std::size_t end) {
            // parallel_for splits at multiples of the grain, so begin/grain
            // is this chunk's index; each chunk owns exactly one partial.
            const std::size_t chunk_index = begin / grain;
            BRACKEN_ASSERT(chunk_index < partials.size());
            cull_rows(begin, end, partials[chunk_index].chunks);
        },
        grain);

    // Single-threaded merge in chunk-index order == serial append order.
    for (const RowPartial& partial : partials) {
        out_chunks.insert(out_chunks.end(), partial.chunks.begin(), partial.chunks.end());
    }
}

void GrassRenderer::upload_chunks(SDL_GPUCopyPass* copy_pass, const glm::vec3& camera_pos,
                                  const scene::Frustum* frustum) {
    BRACKEN_PROFILE_ZONE_COLOR("GrassRenderer::upload_chunks", 0x27AE60);
    if (!initialized_ || !copy_pass) {
        return;
    }

    // === CPU chunk culling ===
    ChunkCullParams cull_params;
    cull_params.camera_pos = camera_pos;
    cull_params.view_distance = effective_view_distance();
    cull_params.world_width = world_width_;
    cull_params.world_height = world_height_;
    cull_params.edge_margin = edge_margin;
    cull_params.y_min = heightmap_params_.min_height;
    cull_params.y_max = heightmap_params_.max_height + 20.0f; // + blade height upper bound

    visible_chunks_scratch_.clear();
    cull_chunks(core::jobs::JobSystem::instance(), cull_params, exclusion_rects_, frustum, visible_chunks_scratch_);

    if (visible_chunks_scratch_.empty()) {
        return;
    }

    // Skip the SSBO re-upload when the visible chunk set is unchanged — the
    // camera pans within a chunk most frames, so the set is usually identical.
    uint64_t set_hash = 1469598103934665603ull; // FNV-1a offset basis
    auto fnv_mix = [&set_hash](uint32_t v) { set_hash = (set_hash ^ v) * 1099511628211ull; };
    for (const auto& chunk : visible_chunks_scratch_) {
        fnv_mix(std::bit_cast<uint32_t>(chunk.x));
        fnv_mix(std::bit_cast<uint32_t>(chunk.z));
    }
    fnv_mix(static_cast<uint32_t>(visible_chunks_scratch_.size()));
    if (set_hash == last_chunk_set_hash_) {
        return; // buffer already holds this exact set
    }

    // Grow storage buffer if needed (buffer creation is legal inside an open copy pass).
    size_t needed_bytes = visible_chunks_scratch_.size() * sizeof(glm::vec4);
    if (needed_bytes > chunk_storage_capacity_bytes_) {
        chunk_storage_capacity_bytes_ = needed_bytes * 2;
        chunk_storage_buffer_ =
            gpu::GPUBuffer::create_dynamic(*device_, gpu::GPUBuffer::Type::Storage, chunk_storage_capacity_bytes_);
        if (!chunk_storage_buffer_) {
            BRACKEN_LOG_ERROR(LOG_TAG, "GrassRenderer::upload_chunks: Failed to grow chunk storage buffer");
            last_chunk_set_hash_ = 0;
            return;
        }
    }

    chunk_storage_buffer_->update(copy_pass, visible_chunks_scratch_.data(), needed_bytes);
    last_chunk_set_hash_ = set_hash;
}

GrassVertexUniforms GrassRenderer::build_vertex_uniforms(const glm::mat4& view, const glm::mat4& projection,
                                                         const glm::vec3& camera_pos, bool splat_enabled) const {
    const float chunk_size = CHUNK_SIZE;
    const float blade_spacing = chunk_size / static_cast<float>(BLADES_PER_CHUNK_SIDE);

    GrassVertexUniforms vu{};
    vu.view_projection = projection * view;
    vu.camera_pos = camera_pos;
    vu.time = current_time_;
    vu.splat_enabled = splat_enabled ? 1.0f : 0.0f;
    vu.grass_view_distance = effective_view_distance();
    vu.wind_direction = effective_wind_direction();
    vu.wind_strength = wind_magnitude;
    vu.chunk_size = chunk_size;
    vu.blade_spacing = blade_spacing;
    vu.blades_per_chunk_side = BLADES_PER_CHUNK_SIDE;
    vu.blades_per_chunk_sq = BLADES_PER_CHUNK_SQ;
    vu.heightmap_origin_x = heightmap_params_.world_origin_x;
    vu.heightmap_origin_z = heightmap_params_.world_origin_z;
    vu.heightmap_world_size = heightmap_params_.world_size;
    vu.heightmap_min_height = heightmap_params_.min_height;
    vu.heightmap_max_height = heightmap_params_.max_height;
    // Splat region: the terrain splatmap spans the world rect, which the
    // TerrainRenderer anchors at the origin. Separate X/Z sizes support
    // non-square worlds; the origin slots future-proof the contract.
    vu.splat_params = glm::vec4(0.0f, 0.0f, std::max(world_width_, 1.0f), std::max(world_height_, 1.0f));
    // World-space camera forward (third row of the view rotation, negated):
    // the shader computes viewDepth = dot(worldPos - cameraPos, forward),
    // i.e. -viewPos.z, matching the other mesh shaders.
    vu.camera_forward = -glm::vec3(view[0][2], view[1][2], view[2][2]);
    return vu;
}

void GrassRenderer::render(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view,
                           const glm::mat4& projection, const glm::vec3& camera_pos, const glm::vec3& light_dir,
                           const SDL_GPUTextureSamplerBinding* shadow_bindings, int shadow_binding_count,
                           const SDL_GPUTextureSamplerBinding* ao_binding, glm::vec2 ao_texel, bool depth_equal,
                           const SDL_GPUTextureSamplerBinding* ibl_bindings, bool ibl_enabled,
                           const SDL_GPUTextureSamplerBinding* local_shadow_binding) {
    BRACKEN_PROFILE_ZONE_COLOR("GrassRenderer::render", 0xC0392B);
    if (!initialized_ || !pipeline_registry_ || !pass || !cmd) {
        return;
    }
    if (!blade_vertex_buffer_ || !blade_index_buffer_ || blade_index_count_ == 0) {
        return;
    }
    if (!heightmap_texture_ || !heightmap_sampler_) {
        return;
    }
    if (visible_chunks_scratch_.empty() || !chunk_storage_buffer_) {
        return;
    }

    auto* pipeline = pipeline_registry_->get_pipeline(depth_equal ? gpu::PipelineType::GrassDepthEqual
                                                                  : gpu::PipelineType::Grass);
    if (!pipeline) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GrassRenderer::render: Failed to get grass pipeline");
        return;
    }

    // Splatmap fetched through the provider every frame — TerrainRenderer
    // recreates the texture on update_splatmap(), so caching a raw pointer
    // across frames would dangle.
    gpu::GPUTexture* splat_tex = splatmap_provider_ ? splatmap_provider_() : nullptr;

    // === Draw ===
    pipeline->bind(pass);

    // Vertex uniforms — must match render_prepass() exactly (shared builder).
    GrassVertexUniforms vu = build_vertex_uniforms(view, projection, camera_pos, splat_tex != nullptr);

    SDL_PushGPUVertexUniformData(cmd, 0, &vu, sizeof(vu));

    // Fragment uniforms (environment fog + lighting).
    GrassLightingUniforms fu{};
    fu.camera_pos = camera_pos;
    fu.fog_start = env_.fog_start;
    fu.fog_color_horizon = env_.fog_color_horizon;
    fu.fog_end = env_.fog_end;
    fu.fog_color_sun = env_.fog_color_sun;
    fu.fog_height_falloff = env_.fog_height_falloff;
    fu.light_dir = light_dir;
    fu.ambient_strength = ambient_strength;
    fu.sun_color = env_.sun_color;
    fu.sun_intensity = sun_intensity;
    fu.ambient_sky = env_.ambient_sky;
    fu.ambient_ground = env_.ambient_ground;
    fu.screen_ao_texel_x = ao_texel.x;
    fu.screen_ao_texel_y = ao_texel.y;
    fu.ibl_enabled = (ibl_enabled && ibl_bindings) ? 1.0f : 0.0f;
    SDL_PushGPUFragmentUniformData(cmd, 0, &fu, sizeof(fu));

    // Vertex samplers: slot 0 = heightmap, slot 1 = splatmap (both CLAMP).
    // Slot 1 must always be bound; fall back to the heightmap texture when no
    // splatmap is set — the shader ignores it when splatEnabled == 0.
    if (!splat_tex) {
        splat_tex = heightmap_texture_;
    }
    SDL_GPUTextureSamplerBinding vertex_bindings[2] = {
        {heightmap_texture_->handle(), heightmap_sampler_->handle()},
        {splat_tex->handle(), heightmap_sampler_->handle()},
    };
    SDL_BindGPUVertexSamplers(pass, 0, vertex_bindings, 2);

    // Vertex storage buffer slot 0: visible-chunk origins.
    SDL_GPUBuffer* chunk_buf = chunk_storage_buffer_->handle();
    SDL_BindGPUVertexStorageBuffers(pass, 0, &chunk_buf, 1);

    // Fragment samplers slots 0-1: shadow atlas (cmp + raw).
    if (shadow_bindings && shadow_binding_count > 0) {
        SDL_BindGPUFragmentSamplers(pass, 0, shadow_bindings, shadow_binding_count);
    }

    // Fragment sampler slot 2: screen-space AO (white dummy when AO is off).
    if (ao_binding) {
        SDL_BindGPUFragmentSamplers(pass, 2, ao_binding, 1);
    }

    // Fragment sampler slots 3-5: sky IBL chain (irradiance cube, prefiltered
    // specular cube, BRDF LUT — cube-typed dummies when IBL is off).
    if (ibl_bindings) {
        SDL_BindGPUFragmentSamplers(pass, 3, ibl_bindings, 3);
    }

    // Fragment sampler slot 6: local (point/spot) shadow atlas + comparison
    // sampler.
    if (local_shadow_binding) {
        SDL_BindGPUFragmentSamplers(pass, 6, local_shadow_binding, 1);
    }

    if (cluster_light_data_ && cluster_offsets_ && cluster_indices_ && cluster_params_) {
        // 4th SSBO = local-shadow slot data; substitute the light-data buffer
        // when local shadows are unavailable (declared but never read while
        // every light's shadow_slot is -1).
        SDL_GPUBuffer* slots = cluster_local_shadow_slots_ ? cluster_local_shadow_slots_ : cluster_light_data_;
        SDL_GPUBuffer* bufs[4] = {cluster_light_data_, cluster_offsets_, cluster_indices_, slots};
        SDL_BindGPUFragmentStorageBuffers(pass, 0, bufs, 4);
        SDL_PushGPUFragmentUniformData(cmd, 2, cluster_params_, static_cast<Uint32>(cluster_params_size_));
    }

    // Blade mesh.
    SDL_GPUBufferBinding vb_binding = {blade_vertex_buffer_->handle(), 0};
    SDL_BindGPUVertexBuffers(pass, 0, &vb_binding, 1);

    SDL_GPUBufferBinding ib_binding = {blade_index_buffer_->handle(), 0};
    SDL_BindGPUIndexBuffer(pass, &ib_binding, SDL_GPU_INDEXELEMENTSIZE_32BIT);

    uint32_t instance_count = static_cast<uint32_t>(visible_chunks_scratch_.size()) * BLADES_PER_CHUNK_SQ;
    SDL_DrawGPUIndexedPrimitives(pass, blade_index_count_, instance_count, 0, 0, 0);
}

void GrassRenderer::render_prepass(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view,
                                   const glm::mat4& projection, const glm::vec3& camera_pos,
                                   const glm::mat4& curr_vp_no_jitter, const glm::mat4& prev_vp_no_jitter,
                                   gpu::PipelineType pipeline_type) {
    BRACKEN_PROFILE_ZONE_COLOR("GrassRenderer::render_prepass", 0xC0392B);
    if (!initialized_ || !pipeline_registry_ || !pass || !cmd) {
        return;
    }
    if (!blade_vertex_buffer_ || !blade_index_buffer_ || blade_index_count_ == 0) {
        return;
    }
    if (!heightmap_texture_ || !heightmap_sampler_) {
        return;
    }
    if (visible_chunks_scratch_.empty() || !chunk_storage_buffer_) {
        return;
    }

    auto* pipeline = pipeline_registry_->get_pipeline(pipeline_type);
    if (!pipeline) {
        BRACKEN_LOG_ERROR(LOG_TAG, "GrassRenderer::render_prepass: Failed to get grass prepass pipeline");
        return;
    }

    gpu::GPUTexture* splat_tex = splatmap_provider_ ? splatmap_provider_() : nullptr;

    pipeline->bind(pass);

    // Same uniform values as render() (shared builder) so blade positions are
    // bit-identical; the velocity matrices are only read by the prepass shader.
    GrassVertexUniforms vu = build_vertex_uniforms(view, projection, camera_pos, splat_tex != nullptr);
    vu.curr_vp_no_jitter = curr_vp_no_jitter;
    vu.prev_vp_no_jitter = prev_vp_no_jitter;
    SDL_PushGPUVertexUniformData(cmd, 0, &vu, sizeof(vu));

    // Vertex samplers: slot 0 = heightmap, slot 1 = splatmap (heightmap
    // placeholder when unset — shader ignores it when splatEnabled == 0).
    if (!splat_tex) {
        splat_tex = heightmap_texture_;
    }
    SDL_GPUTextureSamplerBinding vertex_bindings[2] = {
        {heightmap_texture_->handle(), heightmap_sampler_->handle()},
        {splat_tex->handle(), heightmap_sampler_->handle()},
    };
    SDL_BindGPUVertexSamplers(pass, 0, vertex_bindings, 2);

    SDL_GPUBuffer* chunk_buf = chunk_storage_buffer_->handle();
    SDL_BindGPUVertexStorageBuffers(pass, 0, &chunk_buf, 1);

    SDL_GPUBufferBinding vb_binding = {blade_vertex_buffer_->handle(), 0};
    SDL_BindGPUVertexBuffers(pass, 0, &vb_binding, 1);

    SDL_GPUBufferBinding ib_binding = {blade_index_buffer_->handle(), 0};
    SDL_BindGPUIndexBuffer(pass, &ib_binding, SDL_GPU_INDEXELEMENTSIZE_32BIT);

    uint32_t instance_count = static_cast<uint32_t>(visible_chunks_scratch_.size()) * BLADES_PER_CHUNK_SQ;
    SDL_DrawGPUIndexedPrimitives(pass, blade_index_count_, instance_count, 0, 0, 0);
}

void GrassRenderer::shutdown() {
    blade_vertex_buffer_.reset();
    blade_index_buffer_.reset();
    chunk_storage_buffer_.reset();
    chunk_storage_capacity_bytes_ = 0;
    visible_chunks_scratch_.clear();
    last_chunk_set_hash_ = 0;
    heightmap_sampler_.reset();
    heightmap_texture_ = nullptr;
    splatmap_provider_ = nullptr;
    device_ = nullptr;
    pipeline_registry_ = nullptr;
    blade_index_count_ = 0;
    initialized_ = false;
}

} // namespace bracken::render
