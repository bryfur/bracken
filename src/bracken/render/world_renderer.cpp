#include "bracken/render/world_renderer.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "glm/ext/matrix_float3x3.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/matrix.hpp"
#include "SDL3/SDL_gpu.h"

namespace bracken::render {

namespace gpu = bracken::gpu;

WorldRenderer::WorldRenderer() = default;

WorldRenderer::~WorldRenderer() {
    shutdown();
}

bool WorldRenderer::init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipeline_registry) {
    device_ = &device;
    pipeline_registry_ = &pipeline_registry;

    create_skybox_mesh();

    return skybox_vertex_buffer_ != nullptr;
}

void WorldRenderer::shutdown() {
    skybox_vertex_buffer_.reset();

    device_ = nullptr;
    pipeline_registry_ = nullptr;
}

void WorldRenderer::create_skybox_mesh() {
    if (!device_) {
        return;
    }

    // Fullscreen triangle in clip space (3 vertices that cover the entire screen)
    // z component is unused by shader but included for float3 vertex format
    float vertices[] = {
        -1.0f, -1.0f, 0.0f, 3.0f, -1.0f, 0.0f, -1.0f, 3.0f, 0.0f,
    };

    skybox_vertex_buffer_ =
        gpu::GPUBuffer::create_static(*device_, gpu::GPUBuffer::Type::Vertex, vertices, sizeof(vertices));

    if (!skybox_vertex_buffer_) {
        BRACKEN_LOG_ERROR("render", "Failed to create skybox vertex buffer");
    }
}

void WorldRenderer::render_skybox(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view,
                                  const glm::mat4& projection) {
    BRACKEN_PROFILE_ZONE_COLOR("WorldRenderer::render_skybox", 0xC0392B);
    if (!skybox_vertex_buffer_ || !pipeline_registry_ || !pass || !cmd) {
        return;
    }

    auto* pipeline = pipeline_registry_->get_skybox_pipeline();
    if (!pipeline) {
        return;
    }

    pipeline->bind(pass);

    // Cache the inverse view-projection: glm::inverse is not free per-frame
    // and the camera rotation/projection are stable across many frames.
    const glm::mat4 view_no_translation = glm::mat4(glm::mat3(view));
    if (view_no_translation != cached_sky_view_ || projection != cached_sky_projection_) {
        cached_sky_view_ = view_no_translation;
        cached_sky_projection_ = projection;
        cached_sky_inv_vp_ = glm::inverse(projection * view_no_translation);
    }

    // Push fragment uniforms - invVP for per-pixel ray, time, and atmosphere
    gpu::SkyboxFragmentUniforms fs_uniforms = {};
    fs_uniforms.invViewProjection = cached_sky_inv_vp_;
    fs_uniforms.sunDirection = -env_.sun_dir; // shader expects direction TO the sun
    fs_uniforms.time = skybox_time_;
    fs_uniforms.sunColor = env_.sun_color;
    fs_uniforms.zenithColor = env_.sky_zenith;
    fs_uniforms.horizonColor = env_.sky_horizon;
    fs_uniforms.groundColor = env_.ground_color;
    fs_uniforms.fogHorizonColor = env_.fog_color_horizon;
    fs_uniforms.fogSunColor = env_.fog_color_sun;
    fs_uniforms.cloudCoverage = env_.cloud_coverage;
    fs_uniforms.cloudDensity = env_.cloud_density;
    fs_uniforms.cloudScale = env_.cloud_scale;
    fs_uniforms.hazeStrength = env_.haze_strength;
    SDL_PushGPUFragmentUniformData(cmd, 0, &fs_uniforms, sizeof(fs_uniforms));

    // Bind vertex buffer
    SDL_GPUBufferBinding vb_binding = {};
    vb_binding.buffer = skybox_vertex_buffer_->handle();
    vb_binding.offset = 0;
    SDL_BindGPUVertexBuffers(pass, 0, &vb_binding, 1);

    // Draw fullscreen triangle (3 vertices)
    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
}

} // namespace bracken::render
