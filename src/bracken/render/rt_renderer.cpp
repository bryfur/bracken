#include "bracken/render/rt_renderer.hpp"

#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"

#include <algorithm>
#include <array>

namespace bracken::render {

RtRenderer::~RtRenderer() {
    shutdown();
}

bool RtRenderer::init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipelines) {
    device_ = &device;
    pipelines_ = &pipelines;
    return true;
}

void RtRenderer::shutdown() {
    shadow_mask_.reset();
    reflection_.reset();
    device_ = nullptr;
    pipelines_ = nullptr;
}

bool RtRenderer::ensure_target(gpu::GPUDevice& device, std::unique_ptr<gpu::GPUTexture>& tex,
                               int width, int height, SDL_GPUTextureFormat format,
                               const char* label) {
    if (width <= 0 || height <= 0) {
        return false;
    }
    if (tex && tex->width() == width && tex->height() == height) {
        return true;
    }
    tex = gpu::GPUTexture::create_2d(device, width, height, format,
                                     SDL_GPU_TEXTUREUSAGE_SAMPLER |
                                         SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                                         SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE);
    if (!tex) {
        BRACKEN_LOG_ERROR("rt", "RtRenderer: could not create {} ({}x{})", label, width, height);
        return false;
    }
    return true;
}

void RtRenderer::resize(gpu::GPUDevice& device, int width, int height) {
    screen_width_ = width;
    screen_height_ = height;
    // Targets are (re)created lazily in render(), where the resolution setting
    // is known — a game that never enables ray tracing should not allocate a
    // screen-sized target for it.
    shadow_mask_.reset();
    reflection_.reset();
    (void)device;
}

void RtRenderer::build_scene(gpu::GPUDevice& device, const ModelManager& models,
                             bool want_attributes) {
    scene_.build_blas_for_models(models, want_attributes);
    scene_.upload(device);
    built_with_attributes_ = want_attributes;
    stats_.acceleration_bytes = scene_.stats().gpu_bytes;
}

void RtRenderer::set_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances) {
    scene_.set_instances(instances);
    if (device_ != nullptr) {
        scene_.upload(*device_);
        stats_.acceleration_bytes = scene_.stats().gpu_bytes;
    }
}

void RtRenderer::render(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth, gpu::GPUTexture& normal,
                        const FrameParams& params, const RtShadowSettings& shadows,
                        const RtReflectionSettings& reflections) {
    stats_.shadows_traced = false;
    stats_.reflections_traced = false;
    stats_.shadow_rays = 0;
    stats_.reflection_rays = 0;

    const bool want_any = shadows.enabled || reflections.enabled;
    if (!want_any || cmd == nullptr || device_ == nullptr || pipelines_ == nullptr) {
        return; // both off: not a single byte of work
    }
    if (!scene_.ready()) {
        return; // no acceleration structure yet
    }

    // ---- shadows ---------------------------------------------------------
    if (shadows.enabled) {
        BRACKEN_PROFILE_ZONE("RtRenderer::shadows");
        const int div = rt_resolution_divisor(shadows.resolution);
        const int w = std::max(1, screen_width_ / div);
        const int h = std::max(1, screen_height_ / div);
        if (shadow_res_ != shadows.resolution) {
            shadow_mask_.reset();
            shadow_res_ = shadows.resolution;
        }

        if (ensure_target(*device_, shadow_mask_, w, h, SDL_GPU_TEXTUREFORMAT_R8_UNORM,
                          "shadow mask")) {
            const gpu::GPUComputePipeline* pipeline =
                pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtShadow);
            if (pipeline != nullptr) {
                gpu::RtShadowUniforms u{};
                u.invViewProj = params.inv_view_proj;
                u.sunDirection = params.environment.sun_dir;
                u.rayMaxDistance = shadows.max_distance;
                u.cameraPosition = params.camera_position;
                u.normalBias = shadows.normal_bias;
                u.outputSize = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
                u.tlasNodeCount = scene_.tlas_node_count();
                u.depthBias = shadows.depth_bias;

                const SDL_GPUStorageTextureReadWriteBinding rw{.texture = shadow_mask_->handle(),
                                                               .mip_level = 0,
                                                               .layer = 0,
                                                               .cycle = true,
                                                               .padding1 = 0, .padding2 = 0, .padding3 = 0};
                gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
                if (pass) {
                    pass.bind_pipeline(*pipeline);
                    const std::array<SDL_GPUTexture*, 2> textures{depth.handle(), normal.handle()};
                    pass.bind_storage_textures(0, textures);
                    const std::array<SDL_GPUBuffer*, 4> buffers{
                        scene_.tlas_nodes(), scene_.tlas_instances(), scene_.blas_nodes(),
                        scene_.blas_triangles()};
                    pass.bind_storage_buffers(0, buffers);
                    pass.push_uniforms(0, &u, sizeof(u));
                    pass.dispatch_for(*pipeline, static_cast<uint32_t>(w),
                                      static_cast<uint32_t>(h));
                    stats_.shadows_traced = true;
                    stats_.shadow_rays = static_cast<uint32_t>(w * h);
                }
            }
        }
    }

    // ---- reflections -----------------------------------------------------
    if (reflections.enabled) {
        BRACKEN_PROFILE_ZONE("RtRenderer::reflections");
        if (!scene_.has_attributes()) {
            // Attributes are gathered at build time; enabling reflections
            // afterwards cannot conjure them. Say so once rather than
            // rendering black reflections forever.
            if (!warned_missing_attributes_) {
                BRACKEN_LOG_ERROR("rt",
                                  "Reflections are enabled but the acceleration structure was "
                                  "built without triangle attributes. Rebuild with "
                                  "build_scene(..., want_attributes=true).");
                warned_missing_attributes_ = true;
            }
            return;
        }

        const int div = rt_resolution_divisor(reflections.resolution);
        const int w = std::max(1, screen_width_ / div);
        const int h = std::max(1, screen_height_ / div);
        if (reflection_res_ != reflections.resolution) {
            reflection_.reset();
            reflection_res_ = reflections.resolution;
        }

        if (!ensure_target(*device_, reflection_, w, h, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                           "reflection target")) {
            return;
        }
        const gpu::GPUComputePipeline* pipeline =
            pipelines_->get_compute_pipeline(gpu::ComputePipelineType::RtReflection);
        if (pipeline == nullptr) {
            return;
        }

        gpu::RtReflectionUniforms u{};
        u.invViewProj = params.inv_view_proj;
        u.cameraPosition = params.camera_position;
        u.rayMaxDistance = reflections.max_distance;
        u.sunDirection = params.environment.sun_dir;
        u.maxRoughness = reflections.max_roughness;
        u.sunColor = params.environment.sun_color;
        u.normalBias = reflections.normal_bias;
        u.skyColor = params.environment.sky_horizon;
        u.intensity = reflections.intensity;
        u.outputSize = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
        u.tlasNodeCount = scene_.tlas_node_count();

        const SDL_GPUStorageTextureReadWriteBinding rw{
            .texture = reflection_->handle(), .mip_level = 0, .layer = 0, .cycle = true,
            .padding1 = 0, .padding2 = 0, .padding3 = 0};
        gpu::ComputePass pass = gpu::ComputePass::begin(cmd, {&rw, 1}, {});
        if (pass) {
            pass.bind_pipeline(*pipeline);
            const std::array<SDL_GPUTexture*, 2> textures{depth.handle(), normal.handle()};
            pass.bind_storage_textures(0, textures);
            const std::array<SDL_GPUBuffer*, 5> buffers{
                scene_.tlas_nodes(), scene_.tlas_instances(), scene_.blas_nodes(),
                scene_.blas_triangles(), scene_.triangle_attributes()};
            pass.bind_storage_buffers(0, buffers);
            pass.push_uniforms(0, &u, sizeof(u));
            pass.dispatch_for(*pipeline, static_cast<uint32_t>(w), static_cast<uint32_t>(h));
            stats_.reflections_traced = true;
            stats_.reflection_rays = static_cast<uint32_t>(w * h);
        }
    }
}

gpu::GPUTexture* RtRenderer::shadow_mask() const {
    return shadow_mask_.get();
}

gpu::GPUTexture* RtRenderer::reflection_target() const {
    return reflection_.get();
}

} // namespace bracken::render
