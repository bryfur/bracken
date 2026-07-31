#include "ibl.hpp"
#include "bracken/core/logger.hpp"
#include "bracken/core/profiler.hpp"
#include "bracken/gpu/gpu_pipeline.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include <algorithm>
#include <SDL3/SDL_gpu.h>

namespace bracken::render {

namespace {
constexpr const char* LOG_TAG = "render";
} // namespace

IBL::IBL() = default;

IBL::~IBL() {
    shutdown();
}

IBL::UnitDesc IBL::decode_unit(int unit) {
    if (unit < CAPTURE_UNITS) {
        return {UnitKind::CaptureFace, unit, 0};
    }
    if (unit == CAPTURE_UNITS) {
        return {UnitKind::CaptureMips, 0, 0};
    }
    const int spec_index = unit - CAPTURE_UNITS - 1;
    if (spec_index < SPECULAR_UNITS) {
        return {UnitKind::SpecularFaceMip, spec_index % 6, spec_index / 6};
    }
    return {UnitKind::IrradianceFace, unit - CAPTURE_UNITS - 1 - SPECULAR_UNITS, 0};
}

bool IBL::init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipelines) {
    device_ = &device;

    constexpr SDL_GPUTextureUsageFlags USAGE = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
    capture_ = gpu::GPUTexture::create_cube(device, SPECULAR_SIZE, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, USAGE,
                                            /*mipmaps=*/true);
    specular_ = gpu::GPUTexture::create_cube(device, SPECULAR_SIZE, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, USAGE,
                                             /*mipmaps=*/true);
    irradiance_ = gpu::GPUTexture::create_cube(device, IRRADIANCE_SIZE, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                                               USAGE, /*mipmaps=*/false);
    brdf_lut_ = gpu::GPUTexture::create_2d(device, BRDF_LUT_SIZE, BRDF_LUT_SIZE, SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT,
                                           USAGE);

    // The specular cube's mip count is a compile-time contract with
    // pbr.hlsli's IBL_SPECULAR_MIPS (roughness -> mip mapping).
    static_assert(SPECULAR_MIPS == 7 && SPECULAR_SIZE == 64, "update pbr.hlsli IBL_SPECULAR_MIPS with this");

    SDL_GPUSamplerCreateInfo cube_info = {};
    cube_info.min_filter = SDL_GPU_FILTER_LINEAR;
    cube_info.mag_filter = SDL_GPU_FILTER_LINEAR;
    cube_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
    cube_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    cube_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    cube_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    cube_info.max_lod = 1000.0f; // VK_LOD_CLAMP_NONE — mips must stay reachable
    cube_sampler_ = device.create_sampler(cube_info);

    SDL_GPUSamplerCreateInfo lut_info = cube_info;
    lut_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    lut_info.max_lod = 0.0f;
    lut_sampler_ = device.create_sampler(lut_info);

    if (!is_ready()) {
        BRACKEN_LOG_ERROR(LOG_TAG, "IBL: Failed to initialize");
        return false;
    }

    // One-off command buffer: BRDF LUT (once, sky-independent) + a full
    // synchronous chain build against the default environment, so the maps
    // are complete before anything samples them. If the live environment
    // differs, the first update() detects it and refreshes amortized.
    SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device.handle());
    if (!cmd) {
        BRACKEN_LOG_ERROR(LOG_TAG, "IBL: Failed to acquire init command buffer: {}", SDL_GetError());
        return false;
    }
    render_brdf_lut(cmd, pipelines);
    snapshot_ = IblSkyParams::from_environment(EnvironmentState{});
    snapshot_time_ = 0.0f;
    for (int unit = 0; unit < TOTAL_UNITS; ++unit) {
        execute_unit(cmd, pipelines, unit);
    }
    next_unit_ = TOTAL_UNITS;
    refresh_count_ = 1;
    SDL_SubmitGPUCommandBuffer(cmd);

    BRACKEN_LOG_INFO(LOG_TAG,
                     "IBL initialized: {}px specular cube x{} mips, {}px irradiance cube, {}px BRDF LUT "
                     "(full chain built; refreshes amortize {} passes at {}/frame)",
                     SPECULAR_SIZE, SPECULAR_MIPS, IRRADIANCE_SIZE, BRDF_LUT_SIZE, TOTAL_UNITS, UNITS_PER_FRAME);

    return true;
}

void IBL::shutdown() {
    capture_.reset();
    specular_.reset();
    irradiance_.reset();
    brdf_lut_.reset();

    if (device_) {
        if (cube_sampler_) {
            device_->release_sampler(cube_sampler_);
            cube_sampler_ = nullptr;
        }
        if (lut_sampler_) {
            device_->release_sampler(lut_sampler_);
            lut_sampler_ = nullptr;
        }
    }

    device_ = nullptr;
    brdf_lut_rendered_ = false;
    next_unit_ = TOTAL_UNITS;
}

bool IBL::is_ready() const {
    return capture_ && specular_ && irradiance_ && brdf_lut_ && (cube_sampler_ != nullptr) &&
           (lut_sampler_ != nullptr);
}

void IBL::update(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, const EnvironmentState& env,
                 float sky_time) {
    BRACKEN_PROFILE_ZONE_COLOR("IBL::update", 0x16A085);
    if (!is_ready() || !cmd) {
        return;
    }

    // Start a refresh only between chains: a mid-flight parameter change
    // finishes the current (consistent) snapshot first, then re-runs.
    if (chain_complete()) {
        const IblSkyParams current = IblSkyParams::from_environment(env);
        if (current != snapshot_) {
            snapshot_ = current;
            snapshot_time_ = sky_time;
            next_unit_ = 0;
            BRACKEN_LOG_DEBUG(LOG_TAG, "IBL: sky changed - refreshing prefilter chain ({} passes, {}/frame)",
                              TOTAL_UNITS, UNITS_PER_FRAME);
        }
    }

    if (chain_complete()) {
        return;
    }

    const int end = std::min(next_unit_ + UNITS_PER_FRAME, TOTAL_UNITS);
    for (; next_unit_ < end; ++next_unit_) {
        execute_unit(cmd, pipelines, next_unit_);
    }
    if (chain_complete()) {
        ++refresh_count_;
    }
}

void IBL::render_brdf_lut(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines) {
    if (brdf_lut_rendered_) {
        return;
    }
    auto* pipeline = pipelines.get_ibl_brdf_lut_pipeline();
    if (!pipeline) {
        return;
    }

    SDL_GPUColorTargetInfo target = {};
    target.texture = brdf_lut_->handle();
    target.load_op = SDL_GPU_LOADOP_DONT_CARE;
    target.store_op = SDL_GPU_STOREOP_STORE;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &target, 1, nullptr);
    if (!pass) {
        return;
    }
    pipeline->bind(pass);
    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
    brdf_lut_rendered_ = true;
}

void IBL::execute_unit(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, int unit) {
    const UnitDesc desc = decode_unit(unit);

    if (desc.kind == UnitKind::CaptureMips) {
        // Fills the capture cube's mip chain across all faces; must run
        // outside a render pass, which holds by construction here.
        SDL_GenerateMipmapsForGPUTexture(cmd, capture_->handle());
        return;
    }

    const IblFaceBasis basis = ibl_face_basis(desc.face);

    gpu::GPUPipeline* pipeline = nullptr;
    SDL_GPUTexture* target_tex = nullptr;
    int target_mip = 0;
    bool bind_capture = false;

    switch (desc.kind) {
        case UnitKind::CaptureFace:
            pipeline = pipelines.get_sky_capture_pipeline();
            target_tex = capture_->handle();
            break;
        case UnitKind::SpecularFaceMip:
            pipeline = pipelines.get_ibl_specular_prefilter_pipeline();
            target_tex = specular_->handle();
            target_mip = desc.mip;
            bind_capture = true;
            break;
        case UnitKind::IrradianceFace:
            pipeline = pipelines.get_ibl_irradiance_pipeline();
            target_tex = irradiance_->handle();
            bind_capture = true;
            break;
        default:
            return;
    }
    if (!pipeline || !target_tex) {
        return;
    }

    SDL_GPUColorTargetInfo target = {};
    target.texture = target_tex;
    target.mip_level = static_cast<Uint32>(target_mip);
    target.layer_or_depth_plane = static_cast<Uint32>(desc.face);
    target.load_op = SDL_GPU_LOADOP_DONT_CARE;
    target.store_op = SDL_GPU_STOREOP_STORE;

    SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(cmd, &target, 1, nullptr);
    if (!pass) {
        return;
    }
    pipeline->bind(pass);

    if (desc.kind == UnitKind::CaptureFace) {
        gpu::SkyCaptureUniforms uniforms = {};
        uniforms.faceForward = basis.forward;
        uniforms.faceRight = basis.right;
        uniforms.faceUp = basis.up;
        uniforms.sunDirection = -snapshot_.sun_dir; // shader expects direction TO the sun
        uniforms.time = snapshot_time_;
        uniforms.sunColor = snapshot_.sun_color;
        uniforms.zenithColor = snapshot_.sky_zenith;
        uniforms.horizonColor = snapshot_.sky_horizon;
        uniforms.groundColor = snapshot_.ground_color;
        uniforms.fogHorizonColor = snapshot_.fog_color_horizon;
        uniforms.fogSunColor = snapshot_.fog_color_sun;
        uniforms.cloudCoverage = snapshot_.cloud_coverage;
        uniforms.cloudDensity = snapshot_.cloud_density;
        uniforms.cloudScale = snapshot_.cloud_scale;
        uniforms.hazeStrength = snapshot_.haze_strength;
        SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));
    } else {
        gpu::IblFilterUniforms uniforms = {};
        uniforms.faceForward = basis.forward;
        uniforms.faceRight = basis.right;
        uniforms.faceUp = basis.up;
        uniforms.roughness = desc.kind == UnitKind::SpecularFaceMip ? mip_roughness(desc.mip) : 0.0f;
        uniforms.sourceSize = static_cast<float>(SPECULAR_SIZE);
        SDL_PushGPUFragmentUniformData(cmd, 0, &uniforms, sizeof(uniforms));
    }

    if (bind_capture) {
        SDL_GPUTextureSamplerBinding binding = {capture_->handle(), cube_sampler_};
        SDL_BindGPUFragmentSamplers(pass, 0, &binding, 1);
    }

    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
}

} // namespace bracken::render
