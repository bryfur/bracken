#include "shadow_map.hpp"
#include "bracken/core/profiler.hpp"
#include "SDL3/SDL_gpu.h"
#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <SDL3/SDL_log.h>

namespace bracken::render {

namespace {
// Tallest caster height (world units) above its receivers that the cascades
// account for. A caster up-sun of a cascade's view slice still shadows it;
// the ortho volume must reach h / sin(elevation) toward the light to include
// it, or its shadow silently pops in only when the caster itself enters the
// slice. 1500 covers ~12 m buildings/trees plus terrain relief with margin.
constexpr float CASTER_HEIGHT_BUDGET = 1500.0f;
// Floor on |light_dir.y| so the toward-light reach stays bounded (~10x
// budget) as the sun grazes the horizon.
constexpr float MIN_SUN_ELEVATION_SIN = 0.15f;
// Small pad on the away-from-light side: receivers only need depth ordering,
// not reach, so this side stays tight to preserve depth precision.
constexpr float Z_RECEIVER_PAD = 100.0f;
} // namespace

ShadowMap::ShadowMap() = default;

ShadowMap::~ShadowMap() {
    shutdown();
}

bool ShadowMap::init(gpu::GPUDevice& device, int resolution) {
    device_ = &device;
    resolution_ = resolution;

    atlas_texture_ = gpu::GPUTexture::create_depth(device, resolution_ * 2, resolution_ * 2);
    if (!atlas_texture_) {
        SDL_Log("ShadowMap::init: Failed to create shadow atlas texture");
        return false;
    }

    // Raw NEAREST sampler: PCSS blocker search + volumetric fog compares.
    SDL_GPUSamplerCreateInfo sampler_info = {};
    sampler_info.min_filter = SDL_GPU_FILTER_NEAREST;
    sampler_info.mag_filter = SDL_GPU_FILTER_NEAREST;
    sampler_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    sampler_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sampler_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sampler_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sampler_info.enable_compare = false;
    sampler_info.enable_anisotropy = false;

    shadow_sampler_ = device.create_sampler(sampler_info);
    if (!shadow_sampler_) {
        SDL_Log("ShadowMap::init: Failed to create shadow sampler");
        return false;
    }

    // LINEAR comparison sampler: each SampleCmpLevelZero tap gets free 2x2
    // hardware PCF.
    SDL_GPUSamplerCreateInfo compare_info = sampler_info;
    compare_info.min_filter = SDL_GPU_FILTER_LINEAR;
    compare_info.mag_filter = SDL_GPU_FILTER_LINEAR;
    compare_info.enable_compare = true;
    compare_info.compare_op = SDL_GPU_COMPAREOP_LESS_OR_EQUAL;

    shadow_compare_sampler_ = device.create_sampler(compare_info);
    if (!shadow_compare_sampler_) {
        SDL_Log("ShadowMap::init: Failed to create shadow comparison sampler");
        return false;
    }

    SDL_Log("ShadowMap: Initialized %dx%d atlas (%d per cascade, %d cascades)", resolution_ * 2, resolution_ * 2,
            resolution_, active_cascades_);
    return true;
}

void ShadowMap::shutdown() {
    if (device_ && shadow_sampler_) {
        device_->release_sampler(shadow_sampler_);
        shadow_sampler_ = nullptr;
    }
    if (device_ && shadow_compare_sampler_) {
        device_->release_sampler(shadow_compare_sampler_);
        shadow_compare_sampler_ = nullptr;
    }
    atlas_texture_.reset();
    device_ = nullptr;
}

bool ShadowMap::reinit(int resolution) {
    if (!device_) {
        return false;
    }
    if (resolution == resolution_) {
        return true;
    }

    atlas_texture_.reset();
    resolution_ = resolution;

    atlas_texture_ = gpu::GPUTexture::create_depth(*device_, resolution_ * 2, resolution_ * 2);
    if (!atlas_texture_) {
        SDL_Log("ShadowMap::reinit: Failed to create shadow atlas texture");
        return false;
    }

    SDL_Log("ShadowMap: Reinitialized %dx%d atlas (%d per cascade)", resolution_ * 2, resolution_ * 2, resolution_);
    return true;
}

void ShadowMap::set_active_cascades(int count) {
    active_cascades_ = std::clamp(count, 1, CSM_MAX_CASCADES);
}

void ShadowMap::compute_cascade_splits(float near_plane, float far_plane) {
    float ratio = far_plane / near_plane;
    for (int i = 0; i < active_cascades_; ++i) {
        float p = static_cast<float>(i + 1) / static_cast<float>(active_cascades_);
        float log_split = near_plane * std::pow(ratio, p);
        float uniform_split = near_plane + (far_plane - near_plane) * p;
        cascades_[i].split_depth = split_lambda * log_split + (1.0f - split_lambda) * uniform_split;
    }
}

void ShadowMap::fit_cascade(int cascade_index, const glm::mat4& camera_view, const glm::mat4& camera_proj,
                            const glm::vec3& light_dir, float near_split, float far_split) {
    BRACKEN_PROFILE_ZONE_COLOR("ShadowMap::fit_cascade", 0xE67E22);
    glm::mat4 slice_proj = camera_proj;

    // The z-row is rewritten with a self-consistent STANDARD RH_ZO mapping for
    // [near_split, far_split]; only inv_vp below consumes it (never the GPU),
    // so it works unchanged whether camera_proj itself is reversed-Z or not.
    float n = near_split;
    float f = far_split;
    slice_proj[2][2] = f / (n - f);
    slice_proj[3][2] = -(f * n) / (f - n);

    glm::mat4 inv_vp = glm::inverse(slice_proj * camera_view);

    // 8 NDC corners: x,y in [-1,1], z in [0,1] (depth zero to one)
    static const glm::vec4 ndc_corners[8] = {
        {-1, -1, 0, 1}, {1, -1, 0, 1}, {-1, 1, 0, 1}, {1, 1, 0, 1},
        {-1, -1, 1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}, {1, 1, 1, 1},
    };

    glm::vec3 world_corners[8];
    glm::vec3 center(0.0f);
    for (int i = 0; i < 8; ++i) {
        glm::vec4 wc = inv_vp * ndc_corners[i];
        world_corners[i] = glm::vec3(wc) / wc.w;
        center += world_corners[i];
    }
    center /= 8.0f;

    // Bounding sphere of the slice: the radius depends only on the slice
    // shape (fov/aspect/splits), not camera orientation, so the ortho window
    // never changes size as the camera rotates. Round it up slightly so FP
    // jitter can't change the texel size frame to frame.
    float radius = 0.0f;
    for (const auto& corner : world_corners) {
        radius = std::max(radius, glm::length(corner - center));
    }
    radius = std::ceil(radius * 16.0f) / 16.0f;

    glm::vec3 light_direction = glm::normalize(light_dir);
    glm::vec3 up = glm::vec3(0, 1, 0);
    if (std::abs(glm::dot(light_direction, up)) > 0.99f) {
        up = glm::vec3(1, 0, 0);
    }

    // Rotation-only light view (eye at origin): all translation lives in the
    // ortho window below, where it gets snapped to whole texels.
    glm::mat4 light_view = glm::lookAt(glm::vec3(0.0f), light_direction, up);

    const float texel = 2.0f * radius / static_cast<float>(resolution_);
    glm::vec3 center_ls = glm::vec3(light_view * glm::vec4(center, 1.0f));
    center_ls.x = std::floor(center_ls.x / texel) * texel;
    center_ls.y = std::floor(center_ls.y / texel) * texel;
    center_ls.z = std::floor(center_ls.z / texel) * texel;

    // Asymmetric depth window: reach far toward the light for out-of-slice
    // casters (elevation-dependent), stay tight on the receiver side.
    const float caster_reach =
        radius + CASTER_HEIGHT_BUDGET / std::max(std::abs(light_direction.y), MIN_SUN_ELEVATION_SIN);
    const float receiver_reach = radius + Z_RECEIVER_PAD;
    glm::mat4 light_proj =
        glm::orthoRH_ZO(center_ls.x - radius, center_ls.x + radius, center_ls.y - radius, center_ls.y + radius,
                        -(center_ls.z + caster_reach), -(center_ls.z - receiver_reach));

    cascades_[cascade_index].light_view_projection = light_proj * light_view;
    cascades_[cascade_index].texel_world = texel;
    cascades_[cascade_index].depth_range = caster_reach + receiver_reach;
}

void ShadowMap::update(const glm::mat4& camera_view, const glm::mat4& camera_proj, const glm::vec3& light_dir,
                       float near_plane, float far_plane) {
    BRACKEN_PROFILE_ZONE_COLOR("ShadowMap::update", 0xE67E22);
    shadow_far_ = far_plane;
    compute_cascade_splits(near_plane, far_plane);

    float prev_split = near_plane;
    for (int i = 0; i < active_cascades_; ++i) {
        fit_cascade(i, camera_view, camera_proj, light_dir, prev_split, cascades_[i].split_depth);
        prev_split = cascades_[i].split_depth;
    }
}

SDL_GPURenderPass* ShadowMap::begin_shadow_pass(SDL_GPUCommandBuffer* cmd) {
    if (!cmd || !atlas_texture_) {
        return nullptr;
    }

    SDL_GPUDepthStencilTargetInfo depth_target = {};
    depth_target.texture = atlas_texture_->handle();
    depth_target.load_op = SDL_GPU_LOADOP_CLEAR;
    depth_target.store_op = SDL_GPU_STOREOP_STORE;
    // Shadow cascades deliberately stay standard-Z (clear 1.0, LESS_OR_EQUAL)
    // while the main scene is reversed-Z: the cascades' orthographic
    // projections distribute depth linearly, so they have no precision problem
    // to fix, and receiver bias/PCF math is calibrated for near=0/far=1.
    depth_target.clear_depth = 1.0f;
    depth_target.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
    depth_target.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;

    current_shadow_pass_ = SDL_BeginGPURenderPass(cmd, nullptr, 0, &depth_target);
    if (!current_shadow_pass_) {
        SDL_Log("ShadowMap::begin_shadow_pass: Failed to begin atlas render pass");
        return nullptr;
    }

    return current_shadow_pass_;
}

void ShadowMap::set_cascade_viewport(int cascade_index) {
    if (!current_shadow_pass_ || cascade_index < 0 || cascade_index >= active_cascades_) {
        return;
    }

    const int x = (cascade_index % 2) * resolution_;
    const int y = (cascade_index / 2) * resolution_;

    SDL_GPUViewport viewport = {};
    viewport.x = static_cast<float>(x);
    viewport.y = static_cast<float>(y);
    viewport.w = static_cast<float>(resolution_);
    viewport.h = static_cast<float>(resolution_);
    viewport.min_depth = 0.0f;
    viewport.max_depth = 1.0f;
    SDL_SetGPUViewport(current_shadow_pass_, &viewport);

    SDL_Rect scissor = {};
    scissor.x = x;
    scissor.y = y;
    scissor.w = resolution_;
    scissor.h = resolution_;
    SDL_SetGPUScissor(current_shadow_pass_, &scissor);
}

void ShadowMap::end_shadow_pass() {
    if (current_shadow_pass_) {
        SDL_EndGPURenderPass(current_shadow_pass_);
        current_shadow_pass_ = nullptr;
    }
}

SDL_GPUTexture* ShadowMap::atlas_texture() const {
    return atlas_texture_ ? atlas_texture_->handle() : nullptr;
}

bool ShadowMap::is_ready() const {
    return atlas_texture_ && shadow_sampler_ != nullptr && shadow_compare_sampler_ != nullptr;
}

gpu::ShadowDataUniforms ShadowMap::get_shadow_uniforms(int shadow_mode) const {
    gpu::ShadowDataUniforms uniforms = {};
    const int last = active_cascades_ - 1;
    const float last_split = cascades_[last].split_depth;

    // Inactive cascades clone the last active one: their split equals
    // lastActiveSplit so selection never reaches them past the early-out,
    // and texel/range stay non-zero for safe dynamic indexing.
    for (int i = 0; i < CSM_MAX_CASCADES; ++i) {
        const CascadeData& src = cascades_[std::min(i, last)];
        uniforms.lightViewProjection[i] = src.light_view_projection;
        uniforms.cascadeSplits[i] = src.split_depth;
        uniforms.cascadeTexelWorld[i] = src.texel_world;
        uniforms.cascadeDepthRange[i] = src.depth_range;
    }
    uniforms.shadowMapResolution = static_cast<float>(resolution_);
    uniforms.lightSize = light_size;
    uniforms.shadowEnabled = static_cast<float>(shadow_mode);
    uniforms.lastActiveSplit = last_split;
    uniforms.fadeStart = shadow_far_ * 0.85f;
    uniforms.fadeEnd = shadow_far_;
    return uniforms;
}

} // namespace bracken::render
