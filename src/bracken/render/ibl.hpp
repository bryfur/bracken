#pragma once

#include "bracken/contracts/environment.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>

namespace bracken::render {

namespace gpu = bracken::gpu;

/**
 * Cube-face basis for the IBL raster passes. With u,v the face UV in [0,1]
 * (v grows downward, texel-centered), the sampled direction is
 * normalize(forward + (2u-1)*right + (2v-1)*up). The table encodes the
 * Vulkan/D3D cube lookup convention (faces +X -X +Y -Y +Z -Z = layers 0..5),
 * so a direction written through this basis reads back from the same texel
 * via TextureCube sampling. Mirrored by sky_capture / ibl_prefilter /
 * ibl_irradiance shaders — keep in sync (pinned by ibl.test.cpp).
 */
struct IblFaceBasis {
    glm::vec3 forward;
    glm::vec3 right;
    glm::vec3 up;
};

[[nodiscard]] inline IblFaceBasis ibl_face_basis(int face) {
    switch (face) {
        case 0: return {{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, -1.0f, 0.0f}};  // +X
        case 1: return {{-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, -1.0f, 0.0f}};  // -X
        case 2: return {{0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};    // +Y
        case 3: return {{0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}};  // -Y
        case 4: return {{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}};   // +Z
        default: return {{0.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}}; // -Z
    }
}

/** Direction through face texel UV (u,v in [0,1], v down) — the CPU mirror of
 *  the direction math in the IBL fragment shaders. */
[[nodiscard]] inline glm::vec3 ibl_face_direction(int face, float u, float v) {
    const IblFaceBasis b = ibl_face_basis(face);
    return glm::normalize(b.forward + (2.0f * u - 1.0f) * b.right + (2.0f * v - 1.0f) * b.up);
}

/** Van der Corput radical inverse (base 2) — CPU mirror of radicalInverseVdC
 *  in the IBL shaders (Hammersley sequence). */
[[nodiscard]] inline float ibl_radical_inverse(uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

/**
 * The EnvironmentState subset the procedural sky consumes (sky_common.hlsli).
 * IBL compares this snapshot each frame to detect sky changes; sky_time is
 * deliberately EXCLUDED so the scrolling clouds alone never trigger a
 * refresh (the chain freezes clouds at its snapshot time). cloud_speed is
 * likewise excluded — it is time-like (it only scales the scroll-phase
 * accumulation in WorldRenderer) and never reaches the capture shader.
 */
struct IblSkyParams {
    glm::vec3 sun_dir{0.0f};
    glm::vec3 sun_color{0.0f};
    glm::vec3 sky_zenith{0.0f};
    glm::vec3 sky_horizon{0.0f};
    glm::vec3 ground_color{0.0f};
    glm::vec3 fog_color_horizon{0.0f};
    glm::vec3 fog_color_sun{0.0f};
    float cloud_coverage = 0.0f;
    float cloud_density = 0.0f;
    float cloud_scale = 0.0f;
    float haze_strength = 0.0f;

    bool operator==(const IblSkyParams&) const = default;

    [[nodiscard]] static IblSkyParams from_environment(const EnvironmentState& env) {
        return {env.sun_dir,          env.sun_color,     env.sky_zenith,    env.sky_horizon,
                env.ground_color,     env.fog_color_horizon, env.fog_color_sun,
                env.cloud_coverage,   env.cloud_density, env.cloud_scale,   env.haze_strength};
    }
};

/**
 * Image-based lighting from the procedural sky (raster prefilter chain — no
 * compute). Owns:
 *  - a 64px RGBA16F sky-capture cube (full mip chain; the shared
 *    sky_common.hlsli radiance rendered per face, mips via
 *    SDL_GenerateMipmapsForGPUTexture),
 *  - a 64px RGBA16F GGX-prefiltered specular cube (SPECULAR_MIPS mips,
 *    roughness = mip / (mips-1); pbr.hlsli IBL_SPECULAR_MIPS must match),
 *  - a 16px RGBA16F diffuse-irradiance cube (cosine convolution),
 *  - a 256px RG16F Karis split-sum BRDF LUT (rendered once at init).
 *
 * The forward shaders sample all three in ambientIBL() (pbr.hlsli) when the
 * `ibl_enabled` graphics setting is on; the SceneRenderer binds them at the
 * last three fragment sampler slots of every scene shader.
 *
 * REGENERATION: init() builds the whole chain synchronously against the
 * default EnvironmentState. Afterwards update() re-snapshots the sky-relevant
 * environment fields each frame; on change the chain re-renders AMORTIZED at
 * UNITS_PER_FRAME face/mip passes per frame (~0.1 ms of tiny raster passes —
 * capture faces first, then the capture mip chain, then specular mips,
 * then irradiance), completing in TOTAL_UNITS / UNITS_PER_FRAME ≈ 7 frames.
 * A change arriving mid-refresh finishes the in-flight snapshot first, then
 * starts a fresh pass over the new one — a continuously animating sun
 * therefore lags the visible sky by at most ~2 refresh spans, never hitches.
 */
class IBL {
public:
    static constexpr int SPECULAR_SIZE = 64;
    static constexpr uint32_t SPECULAR_MIPS = 7; // == log2(64) + 1; pbr.hlsli IBL_SPECULAR_MIPS
    static constexpr int IRRADIANCE_SIZE = 16;
    static constexpr int BRDF_LUT_SIZE = 256;

    // Work-unit plan for one chain build, in execution order:
    //   [0, 6)                    sky-capture faces (mip 0)
    //   6                         capture-cube mip generation
    //   [7, 7 + 6 * SPECULAR_MIPS) specular prefilter, unit -> (mip, face)
    //   last 6                    irradiance faces
    static constexpr int CAPTURE_UNITS = 6;
    static constexpr int SPECULAR_UNITS = static_cast<int>(SPECULAR_MIPS) * 6;
    static constexpr int IRRADIANCE_UNITS = 6;
    static constexpr int TOTAL_UNITS = CAPTURE_UNITS + 1 + SPECULAR_UNITS + IRRADIANCE_UNITS;

    // Amortization budget: passes recorded per frame while a refresh is in
    // flight. 9 units ≈ 64x64-and-smaller draws, well under 0.1 ms on any
    // hardware that runs the engine; a full refresh spans ceil(55/9) = 7
    // frames.
    static constexpr int UNITS_PER_FRAME = 9;

    // What a work unit does — decode is pure so tests can pin the plan.
    enum class UnitKind { CaptureFace, CaptureMips, SpecularFaceMip, IrradianceFace };
    struct UnitDesc {
        UnitKind kind;
        int face; // CaptureFace/SpecularFaceMip/IrradianceFace
        int mip;  // SpecularFaceMip only
    };
    [[nodiscard]] static UnitDesc decode_unit(int unit);

    /** Roughness rendered into a specular mip: mip / (SPECULAR_MIPS - 1). */
    [[nodiscard]] static float mip_roughness(int mip) {
        return static_cast<float>(mip) / static_cast<float>(SPECULAR_MIPS - 1);
    }

    IBL();
    ~IBL();

    IBL(const IBL&) = delete;
    IBL& operator=(const IBL&) = delete;
    IBL(IBL&&) = delete;
    IBL& operator=(IBL&&) = delete;

    /**
     * Create textures/samplers, render the BRDF LUT, and build the full
     * prefilter chain synchronously against a default EnvironmentState —
     * all on a one-off command buffer, so the maps are complete before the
     * first frame samples them.
     */
    [[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipelines);
    void shutdown();
    bool is_ready() const;

    /**
     * Per-frame upkeep: snapshot the sky-relevant EnvironmentState fields,
     * start a refresh when they changed, and record up to UNITS_PER_FRAME
     * prefilter passes on `cmd` (must be outside any render pass; call
     * BEFORE the forward pass so same-frame samples see the newest faces).
     */
    void update(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, const EnvironmentState& env,
                float sky_time);

    SDL_GPUTexture* irradiance_map() const { return irradiance_ ? irradiance_->handle() : nullptr; }
    SDL_GPUTexture* specular_map() const { return specular_ ? specular_->handle() : nullptr; }
    SDL_GPUTexture* brdf_lut() const { return brdf_lut_ ? brdf_lut_->handle() : nullptr; }
    SDL_GPUSampler* cube_sampler() const { return cube_sampler_; }   // linear + mips, clamp
    SDL_GPUSampler* lut_sampler() const { return lut_sampler_; }     // linear, no mips, clamp

    // Introspection (dev overlay / tests / smoke logging).
    bool chain_complete() const { return next_unit_ >= TOTAL_UNITS; }
    int pending_units() const { return chain_complete() ? 0 : TOTAL_UNITS - next_unit_; }
    /** Completed chain builds (the synchronous init build counts as 1). */
    uint64_t refresh_count() const { return refresh_count_; }

private:
    void execute_unit(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, int unit);
    void render_brdf_lut(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines);

    gpu::GPUDevice* device_ = nullptr;

    std::unique_ptr<gpu::GPUTexture> capture_;    // 64px cube, full mips (sky radiance source)
    std::unique_ptr<gpu::GPUTexture> specular_;   // 64px cube, SPECULAR_MIPS mips
    std::unique_ptr<gpu::GPUTexture> irradiance_; // 16px cube
    std::unique_ptr<gpu::GPUTexture> brdf_lut_;   // 256px RG16F

    SDL_GPUSampler* cube_sampler_ = nullptr;
    SDL_GPUSampler* lut_sampler_ = nullptr;

    // Refresh state machine.
    IblSkyParams snapshot_{};      // params the in-flight/last chain rendered
    float snapshot_time_ = 0.0f;   // sky time frozen into that chain
    int next_unit_ = TOTAL_UNITS;  // == TOTAL_UNITS when idle/complete
    bool brdf_lut_rendered_ = false;
    uint64_t refresh_count_ = 0;
};

} // namespace bracken::render
