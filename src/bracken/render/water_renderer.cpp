#include "bracken/render/water_renderer.hpp"

#include "bracken/core/logger.hpp"
#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace bracken::render {

namespace {
constexpr const char* LOG_TAG = "water";
} // namespace

WaterRenderer::~WaterRenderer() {
    shutdown();
}

bool WaterRenderer::init(gpu::GPUDevice& device, gpu::PipelineRegistry& registry) {
    device_ = &device;
    registry_ = &registry;

    linear_clamp_sampler_ = gpu::GPUSampler::create(device, gpu::SamplerConfig::linear_clamp());
    if (!linear_clamp_sampler_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "failed to create sampler");
        return false;
    }

    // Ping-pong wave state. COLOR_TARGET is only for the one-time lazy clear
    // (there is no texture-upload path for float textures); the per-frame
    // uses are compute read/write + sampled reads from the water shaders.
    for (auto& tex : sim_state_) {
        tex = gpu::GPUTexture::create_2d(
            device, static_cast<int>(SIM_RESOLUTION), static_cast<int>(SIM_RESOLUTION),
            SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT,
            SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET);
        if (!tex) {
            BRACKEN_LOG_ERROR(LOG_TAG, "failed to create {}x{} RG16F sim state", SIM_RESOLUTION, SIM_RESOLUTION);
            return false;
        }
    }

    initialized_ = true;
    return true;
}

void WaterRenderer::shutdown() {
    vertex_buffer_.reset();
    index_buffer_.reset();
    sim_state_[0].reset();
    sim_state_[1].reset();
    linear_clamp_sampler_.reset();
    pending_impulses_.clear();
    initialized_ = false;
    has_region_ = false;
    device_ = nullptr;
    registry_ = nullptr;
}

glm::ivec2 WaterRenderer::grid_cells_for(float size_x, float size_z, int max_cells) {
    const float sx = std::max(size_x, 1.0f);
    const float sz = std::max(size_z, 1.0f);
    const float longest = std::max(sx, sz);
    const int nx = std::max(1, static_cast<int>(std::lround(static_cast<float>(max_cells) * sx / longest)));
    const int nz = std::max(1, static_cast<int>(std::lround(static_cast<float>(max_cells) * sz / longest)));
    return {nx, nz};
}

void WaterRenderer::set_region(const WaterRegion& region) {
    region_ = region;
    if (region_.sim_center == glm::vec2(0.0f)) {
        region_.sim_center = {(region_.min_x + region_.max_x) * 0.5f, (region_.min_z + region_.max_z) * 0.5f};
    }
    if (glm::length(region_.flow) > 1e-4f) {
        region_.flow = glm::normalize(region_.flow);
    }
    has_region_ = region_.max_x > region_.min_x && region_.max_z > region_.min_z;
    if (initialized_ && has_region_) {
        build_surface_mesh();
    }
}

void WaterRenderer::build_surface_mesh() {
    const float size_x = region_.max_x - region_.min_x;
    const float size_z = region_.max_z - region_.min_z;
    const glm::ivec2 cells = grid_cells_for(size_x, size_z);

    std::vector<float> verts;
    verts.reserve(static_cast<size_t>(cells.x + 1) * (cells.y + 1) * 2);
    for (int iz = 0; iz <= cells.y; ++iz) {
        for (int ix = 0; ix <= cells.x; ++ix) {
            verts.push_back(region_.min_x + size_x * static_cast<float>(ix) / static_cast<float>(cells.x));
            verts.push_back(region_.min_z + size_z * static_cast<float>(iz) / static_cast<float>(cells.y));
        }
    }
    std::vector<uint32_t> indices;
    indices.reserve(static_cast<size_t>(cells.x) * cells.y * 6);
    const auto vtx = [&](int ix, int iz) { return static_cast<uint32_t>(iz * (cells.x + 1) + ix); };
    for (int iz = 0; iz < cells.y; ++iz) {
        for (int ix = 0; ix < cells.x; ++ix) {
            indices.push_back(vtx(ix, iz));
            indices.push_back(vtx(ix, iz + 1));
            indices.push_back(vtx(ix + 1, iz));
            indices.push_back(vtx(ix + 1, iz));
            indices.push_back(vtx(ix, iz + 1));
            indices.push_back(vtx(ix + 1, iz + 1));
        }
    }

    vertex_buffer_ = gpu::GPUBuffer::create_static(*device_, gpu::GPUBuffer::Type::Vertex, verts.data(),
                                                   verts.size() * sizeof(float));
    index_buffer_ = gpu::GPUBuffer::create_static(*device_, gpu::GPUBuffer::Type::Index, indices.data(),
                                                  indices.size() * sizeof(uint32_t));
    index_count_ = static_cast<uint32_t>(indices.size());
    if (!vertex_buffer_ || !index_buffer_) {
        BRACKEN_LOG_ERROR(LOG_TAG, "failed to build surface mesh ({}x{} cells)", cells.x, cells.y);
        has_region_ = false;
        return;
    }
    BRACKEN_LOG_INFO(LOG_TAG, "water region [{:.0f},{:.0f}]x[{:.0f},{:.0f}] level {:.1f}, grid {}x{}",
                     region_.min_x, region_.max_x, region_.min_z, region_.max_z, region_.height, cells.x,
                     cells.y);
}

void WaterRenderer::add_ripple(const glm::vec3& world_pos, float radius, float strength) {
    if (!active()) {
        return;
    }
    const float window = std::max(region_.sim_extent, 1.0f);
    const glm::vec2 min = region_.sim_center - 0.5f * window;
    const glm::vec2 uv = (glm::vec2(world_pos.x, world_pos.z) - min) / window;
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
        return; // outside the simulated window
    }
    const float texels_per_unit = static_cast<float>(SIM_RESOLUTION) / window;
    pending_impulses_.push_back({uv * static_cast<float>(SIM_RESOLUTION),
                                 std::max(radius * texels_per_unit, 1.0f), strength});
}

void WaterRenderer::clear_sim_textures(SDL_GPUCommandBuffer* cmd) {
    for (auto& tex : sim_state_) {
        SDL_GPUColorTargetInfo target = {};
        target.texture = tex->handle();
        target.load_op = SDL_GPU_LOADOP_CLEAR;
        target.store_op = SDL_GPU_STOREOP_STORE;
        target.clear_color = {0.0f, 0.0f, 0.0f, 0.0f};
        SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &target, 1, nullptr);
        if (pass) {
            SDL_EndGPURenderPass(pass);
        }
    }
    sim_cleared_ = true;
}

void WaterRenderer::simulate(SDL_GPUCommandBuffer* cmd, float dt) {
    if (!active() || cmd == nullptr) {
        return;
    }
    auto* pipeline = registry_->get_compute_pipeline(gpu::ComputePipelineType::WaterSim);
    if (pipeline == nullptr) {
        return;
    }
    if (!sim_cleared_) {
        clear_sim_textures(cmd);
    }

    sim_time_accum_ += std::min(dt, 0.25f); // clamp hitches; the sim is cosmetic
    int substeps = static_cast<int>(sim_time_accum_ / SIM_SUBSTEP);
    substeps = std::min(substeps, MAX_SUBSTEPS_PER_FRAME);
    sim_time_accum_ = std::min(sim_time_accum_ - static_cast<float>(substeps) * SIM_SUBSTEP, SIM_SUBSTEP);
    if (substeps == 0) {
        return;
    }

    const float window = std::max(region_.sim_extent, 1.0f);
    const float texel_world = window / static_cast<float>(SIM_RESOLUTION);
    // Courant number for the 5-point stencil; keep comfortably below 0.5.
    const float courant = region_.wave_speed * SIM_SUBSTEP / texel_world;

    SimUniforms uniforms;
    uniforms.resolution = SIM_RESOLUTION;
    uniforms.waveC2Dt2 = std::min(courant * courant, 0.24f);
    uniforms.damping = region_.wave_damping;
    uniforms.advectTexels = region_.flow_speed * SIM_SUBSTEP / texel_world;
    uniforms.flowDir = region_.flow;
    uniforms.borderTexels = 24.0f;

    SDL_PushGPUDebugGroup(cmd, "Water Sim");
    for (int step = 0; step < substeps; ++step) {
        // Impulses feed the first substep only (they are per-frame events).
        uniforms.impulseCount = 0;
        if (step == 0 && !pending_impulses_.empty()) {
            const size_t n = std::min(pending_impulses_.size(), static_cast<size_t>(MAX_IMPULSES));
            for (size_t i = 0; i < n; ++i) {
                const PendingImpulse& imp = pending_impulses_[i];
                uniforms.impulses[i] = {imp.pos_texels.x, imp.pos_texels.y, imp.radius_texels, imp.strength};
            }
            uniforms.impulseCount = static_cast<uint32_t>(n);
            pending_impulses_.clear();
        }

        SDL_GPUStorageTextureReadWriteBinding rw = {};
        rw.texture = sim_state_[sim_read_ ^ 1]->handle();
        rw.cycle = false; // every texel is overwritten; no need to cycle

        auto pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (!pass) {
            break;
        }
        pass.bind_pipeline(*pipeline);
        SDL_GPUTexture* read_tex = sim_state_[sim_read_]->handle();
        pass.bind_storage_textures(0, {&read_tex, 1});
        pass.push_uniforms(0, &uniforms, sizeof(uniforms));
        pass.dispatch_for(*pipeline, SIM_RESOLUTION, SIM_RESOLUTION);
        pass.end();

        sim_read_ ^= 1;
    }
    SDL_PopGPUDebugGroup(cmd);
}

void WaterRenderer::render(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view_proj,
                           const glm::mat4& projection, const glm::vec3& camera_pos, float time_s,
                           int screen_width, int screen_height, SDL_GPUTexture* scene_color_copy,
                           SDL_GPUTexture* scene_depth_copy, const SDL_GPUTextureSamplerBinding& ibl_specular,
                           bool ibl_enabled, float sun_intensity) {
    if (!active() || pass == nullptr || index_count_ == 0 || scene_color_copy == nullptr ||
        scene_depth_copy == nullptr) {
        return;
    }
    auto* pipeline = registry_->get_pipeline(gpu::PipelineType::Water);
    if (pipeline == nullptr) {
        return;
    }
    pipeline->bind(pass);

    const float window = std::max(region_.sim_extent, 1.0f);
    const glm::vec2 sim_min = region_.sim_center - 0.5f * window;
    const float texel_world = window / static_cast<float>(SIM_RESOLUTION);

    VertexUniforms vu;
    vu.viewProj = view_proj;
    vu.simWindow = {sim_min.x, sim_min.y, 1.0f / window, 0.0f};
    vu.params = {region_.height, region_.sim_displacement, 0.06f, 0.0f};
    SDL_PushGPUVertexUniformData(cmd, 0, &vu, sizeof(vu));

    FragmentUniforms fu;
    fu.cameraPosTime = {camera_pos, time_s};
    fu.sunDirIntensity = {env_.sun_dir, sun_intensity};
    fu.sunColorIbl = {env_.sun_color, ibl_enabled ? 1.0f : 0.0f};
    fu.deepColor = {region_.deep_color, region_.absorption};
    fu.shallowColor = {region_.shallow_color, region_.foam_depth};
    fu.fogHorizon = {env_.fog_color_horizon, env_.fog_start};
    fu.fogSun = {env_.fog_color_sun, env_.fog_end};
    fu.screenDepthAB = {1.0f / static_cast<float>(std::max(screen_width, 1)),
                        1.0f / static_cast<float>(std::max(screen_height, 1)), projection[2][2],
                        projection[3][2]};
    fu.flowDetail = {region_.flow.x, region_.flow.y, region_.flow_speed, region_.detail_wave_strength};
    fu.simParams = {texel_world, region_.sim_normal_strength, region_.sim_displacement,
                    env_.fog_height_falloff};
    fu.misc = {region_.refraction_strength, region_.reflection_roughness, region_.ripple_foam_gain,
               region_.reflection_strength};
    fu.viewProjSSR = view_proj;
    fu.ssrParams = {region_.ssr_strength, region_.ssr_max_distance, region_.ssr_thickness, 0.0f};
    SDL_PushGPUFragmentUniformData(cmd, 0, &fu, sizeof(fu));

    // Vertex stage: the sim state displaces the grid.
    SDL_GPUTextureSamplerBinding sim_binding = {sim_state_[sim_read_]->handle(), linear_clamp_sampler_->handle()};
    SDL_BindGPUVertexSamplers(pass, 0, &sim_binding, 1);

    // Fragment stage: scene copies + sim + IBL specular cube.
    SDL_GPUTextureSamplerBinding frag_bindings[4] = {
        {scene_color_copy, linear_clamp_sampler_->handle()},
        {scene_depth_copy, linear_clamp_sampler_->handle()},
        {sim_state_[sim_read_]->handle(), linear_clamp_sampler_->handle()},
        ibl_specular,
    };
    SDL_BindGPUFragmentSamplers(pass, 0, frag_bindings, 4);

    SDL_GPUBufferBinding vb = {vertex_buffer_->handle(), 0};
    SDL_BindGPUVertexBuffers(pass, 0, &vb, 1);
    SDL_GPUBufferBinding ib = {index_buffer_->handle(), 0};
    SDL_BindGPUIndexBuffer(pass, &ib, SDL_GPU_INDEXELEMENTSIZE_32BIT);
    SDL_DrawGPUIndexedPrimitives(pass, index_count_, 1, 0, 0, 0);
}

} // namespace bracken::render
