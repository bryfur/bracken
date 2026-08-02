#include "volumetric_fog.hpp"
#include "bracken/render/gi/gi_bindings.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include <algorithm>
#include <glm/gtc/matrix_inverse.hpp>
#include <SDL3/SDL_gpu.h>

namespace bracken::render {

namespace {
constexpr const char* LOG_TAG = "fog";
} // namespace

VolumetricFog::VolumetricFog() = default;

VolumetricFog::~VolumetricFog() {
    shutdown();
}

bool VolumetricFog::init(gpu::GPUDevice& device, int width, int height) {
    device_ = &device;
    width_ = width;
    height_ = height;
    fog_width_ = std::max(1, width / 2);
    fog_height_ = std::max(1, height / 2);

    create_textures(width, height);
    create_sampler();

    if (!is_ready()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "VolumetricFog: Failed to initialize");
        return false;
    }

    BRACKEN_LOG_INFO(LOG_TAG, "VolumetricFog: Initialized {}x{} (fog: {}x{})", width, height, fog_width_, fog_height_);
    return true;
}

void VolumetricFog::shutdown() {
    fog_texture_.reset();

    if (device_ && linear_sampler_) {
        device_->release_sampler(linear_sampler_);
        linear_sampler_ = nullptr;
    }

    device_ = nullptr;
}

void VolumetricFog::resize(int width, int height) {
    if (width == width_ && height == height_) {
        return;
    }
    if (!device_) {
        return;
    }

    width_ = width;
    height_ = height;
    fog_width_ = std::max(1, width / 2);
    fog_height_ = std::max(1, height / 2);

    create_textures(width, height);
    BRACKEN_LOG_INFO(LOG_TAG, "VolumetricFog: Resized to {}x{} (fog: {}x{})", width, height, fog_width_, fog_height_);
}

void VolumetricFog::create_textures(int width, int height) {
    (void)width;
    (void)height;
    fog_texture_ =
        gpu::GPUTexture::create_2d(*device_, fog_width_, fog_height_, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                                   SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER);
}

void VolumetricFog::create_sampler() {
    SDL_GPUSamplerCreateInfo info = {};
    info.min_filter = SDL_GPU_FILTER_LINEAR;
    info.mag_filter = SDL_GPU_FILTER_LINEAR;
    info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    linear_sampler_ = device_->create_sampler(info);
}

bool VolumetricFog::is_ready() const {
    return fog_texture_ && (linear_sampler_ != nullptr);
}

SDL_GPUTexture* VolumetricFog::fog_texture() const {
    if (fog_texture_) {
        return fog_texture_->handle();
    }
    return nullptr;
}

gpu::VolumetricFogUniforms VolumetricFog::build_uniforms(const ShadowMap& shadow_map,
                                                         const scene::CameraState& camera,
                                                         const EnvironmentState& env, bool god_rays_enabled,
                                                         bool fog_enabled, float density_multiplier,
                                                         const gi::GiBindings* gi) {
    gpu::VolumetricFogUniforms uniforms = {};
    if (camera.view_projection != cached_view_projection_) {
        cached_view_projection_ = camera.view_projection;
        cached_inv_view_projection_ = glm::inverse(camera.view_projection);
    }
    uniforms.invViewProjection = cached_inv_view_projection_;

    // God rays sample every cascade; the shader picks one per march sample by
    // camera distance. Inactive cascades clone the last active one (same
    // scheme as ShadowMap::get_shadow_uniforms) so selection never reaches
    // them past the lastActiveSplit early-out.
    const auto& cascades = shadow_map.cascades();
    const int last = std::max(shadow_map.active_cascades() - 1, 0);
    for (int i = 0; i < CSM_MAX_CASCADES; ++i) {
        const CascadeData& src = cascades[std::min(i, last)];
        uniforms.shadowLightViewProjection[i] = src.light_view_projection;
        uniforms.cascadeSplits[i] = src.split_depth;
        uniforms.shadowBias[i] = 1.5f * src.texel_world / std::max(src.depth_range, 1e-6f);
    }
    uniforms.lastActiveSplit = cascades[last].split_depth;
    uniforms.shadowAtlasTexel = 0.5f / static_cast<float>(shadow_map.resolution());

    uniforms.lightDir = env.sun_dir;
    uniforms.fogDensity = fog_enabled ? env.fog_volumetric_density : 0.0f;
    uniforms.lightColor = env.fog_volumetric_sun_color;
    uniforms.scatterStrength = env.fog_volumetric_scatter;
    uniforms.fogColor = env.fog_color_horizon;
    uniforms.fogHeight = env.fog_volumetric_height;
    uniforms.cameraPos = camera.position;
    uniforms.fogFalloff = env.fog_volumetric_falloff;
    uniforms.startDist = env.fog_volumetric_start;
    uniforms.maxDist = env.fog_volumetric_end;
    uniforms.godRaysEnabled = god_rays_enabled ? 1.0f : 0.0f;
    uniforms.densityMultiplier = density_multiplier;
    // R1c: both the probe field's fallback and the reference its per-channel
    // gain is measured against, so the fog's ambient term is bit-identical to
    // the pre-R1c one wherever the field has no answer.
    uniforms.ambientSky = env.ambient_sky;
    uniforms.ambientGround = env.ambient_ground;
    if (gi != nullptr && gi->textures[0].texture != nullptr) {
        uniforms.giCascadeBase[0] = gi->cascade_base[0];
        uniforms.giCascadeBase[1] = gi->cascade_base[1];
        uniforms.giCascadeBase[2] = gi->cascade_base[2];
        uniforms.giEnabled = gi->enabled;
    }
    return uniforms;
}

void VolumetricFog::render(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& registry, SDL_GPUTexture* depth_texture,
                           const ShadowMap& shadow_map, const scene::CameraState& camera, const EnvironmentState& env,
                           bool god_rays_enabled, bool fog_enabled, float density_multiplier,
                           const gi::GiBindings* gi) {
    BRACKEN_PROFILE_ZONE_COLOR("VolumetricFog::render", 0x16A085);
    if (!is_ready() || !cmd || !depth_texture) {
        return;
    }

    auto* pipeline = registry.get_pipeline(gpu::PipelineType::VolumetricFog);
    if (!pipeline) {
        return;
    }

    gpu::VolumetricFogUniforms uniforms =
        build_uniforms(shadow_map, camera, env, god_rays_enabled, fog_enabled, density_multiplier, gi);

    // Begin render pass to fog texture
    SDL_GPUColorTargetInfo color_target = {};
    color_target.texture = fog_texture_->handle();
    color_target.load_op = SDL_GPU_LOADOP_DONT_CARE;
    color_target.store_op = SDL_GPU_STOREOP_STORE;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, nullptr);
    if (!pass) {
        return;
    }

    pipeline->bind(pass);
    SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));

    // Depth (slot 0), shadow atlas with the raw sampler (slot 1), and the
    // probe field's three atlases (slots 2-4). The GI slots are declared
    // unconditionally by the shader, so they are bound even with GI off —
    // giEnabled is what decides whether they are read.
    SDL_GPUTextureSamplerBinding bindings[5] = {};
    bindings[0].texture = depth_texture;
    bindings[0].sampler = linear_sampler_;
    bindings[1].texture = shadow_map.atlas_texture();
    bindings[1].sampler = shadow_map.shadow_sampler();
    int binding_count = 2;
    if (gi != nullptr && gi->textures[0].texture != nullptr) {
        bindings[2] = gi->textures[0];
        bindings[3] = gi->textures[1];
        bindings[4] = gi->textures[2];
        binding_count = 5;
    }
    SDL_BindGPUFragmentSamplers(pass, 0, bindings, static_cast<Uint32>(binding_count));

    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
}

} // namespace bracken::render
