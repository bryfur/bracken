#pragma once

#include "bracken/contracts/environment.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include <SDL3/SDL_gpu.h>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace bracken::gpu {
class GPUDevice;
class PipelineRegistry;
} // namespace bracken::gpu

namespace bracken::render {

/**
 * A single water surface: a flat plane at `height` covering the world rect,
 * drawn as a displaced grid. Terrain above the water level hides the plane
 * via the depth test, so the rect can generously cover a carved riverbed —
 * only the channel shows water.
 *
 * The interactive simulation runs in a square window (`sim_center` /
 * `sim_extent`) that should cover the stretch of water the player can reach;
 * outside the window the surface falls back to the analytic detail waves.
 */
struct WaterRegion {
    float min_x = 0.0f;
    float min_z = 0.0f;
    float max_x = 0.0f;
    float max_z = 0.0f;
    float height = 0.0f;             // water level (world Y)
    glm::vec2 flow{0.0f, -1.0f};     // normalized flow direction (world XZ)
    float flow_speed = 50.0f;        // world units/s

    // Look (linear-space colors, matching the scene's PBR pipeline).
    glm::vec3 deep_color{0.02f, 0.07f, 0.09f};
    glm::vec3 shallow_color{0.20f, 0.42f, 0.40f};
    float absorption = 0.008f;          // Beer-Lambert extinction, 1/world-units
    float foam_depth = 20.0f;           // shoreline foam band (units of water depth)
    float reflection_roughness = 0.14f; // 0..1 into the IBL prefilter mips
    float reflection_strength = 0.75f;  // sky-reflection dimmer (artistic; a bright
                                        // dusk sky swamps the bed even at low fresnel)
    float refraction_strength = 0.035f; // screen-uv distortion scale
    // Screen-space reflections: geometry (trees, buildings, the player)
    // mirrored in the water, ray-marched against the scene depth copy.
    // Falls back to the sky IBL where rays miss or leave the screen.
    float ssr_strength = 1.0f;          // 0 disables the march entirely
    float ssr_max_distance = 2400.0f;   // world-units ray length
    float ssr_thickness = 30.0f;        // hit tolerance (world units, grows with t)
    float detail_wave_strength = 0.12f; // analytic micro-wave max slope (~7 deg)
    float sim_displacement = 6.0f;      // world units of surface lift per sim unit
    float sim_normal_strength = 0.7f;   // sim ripple normal contribution
    float ripple_foam_gain = 0.25f;     // foam from energetic sim crests

    glm::vec2 sim_center{0.0f};
    float sim_extent = 2048.0f;         // sim window edge length (world units)

    // Wave-equation tuning.
    float wave_speed = 90.0f;           // ripple propagation (world units/s)
    float wave_damping = 0.994f;        // amplitude retention per substep
};

/**
 * Water surface renderer + compute wave simulation.
 *
 * simulate() runs the fixed-step wave equation (ping-pong RG16F state,
 * ripple impulses via add_ripple, downstream advection) in a compute pass —
 * call it once per frame outside any render pass. render() draws the
 * surface in its own pass after the opaque scene + skybox, sampling copies
 * of the scene color/depth for refraction and shoreline softness (the live
 * depth buffer stays bound for real depth testing).
 */
class WaterRenderer {
public:
    static constexpr uint32_t SIM_RESOLUTION = 512;
    static constexpr int MAX_IMPULSES = 8;
    static constexpr float SIM_SUBSTEP = 1.0f / 60.0f;
    static constexpr int MAX_SUBSTEPS_PER_FRAME = 4;

    WaterRenderer() = default;
    ~WaterRenderer();

    WaterRenderer(const WaterRenderer&) = delete;
    WaterRenderer& operator=(const WaterRenderer&) = delete;
    WaterRenderer(WaterRenderer&&) = delete;
    WaterRenderer& operator=(WaterRenderer&&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& registry);
    void shutdown();

    /** Configure the (single) water region and build its surface grid.
     *  sim_center defaults to the region centre when left at zero. */
    void set_region(const WaterRegion& region);
    [[nodiscard]] bool active() const { return initialized_ && has_region_; }
    const WaterRegion& region() const { return region_; }

    /** Queue a ripple impulse (world position; radius/strength in world
     *  units). Impulses outside the sim window are ignored. Cheap; call
     *  freely (wakes, footsteps, splashes, rain). */
    void add_ripple(const glm::vec3& world_pos, float radius, float strength);

    void set_environment(const EnvironmentState& env) { env_ = env; }

    /** Advance the wave simulation (fixed substeps). Records compute passes
     *  on `cmd`; must be called OUTSIDE any render/copy pass. */
    void simulate(SDL_GPUCommandBuffer* cmd, float dt);

    /** Draw the water surface. The pass must have the HDR scene color bound
     *  (LOAD) plus the real scene depth (LOAD, reversed-Z test). The copies
     *  passed in are what refraction/shore softness sample. */
    void render(SDL_GPURenderPass* pass, SDL_GPUCommandBuffer* cmd, const glm::mat4& view_proj,
                const glm::mat4& projection, const glm::vec3& camera_pos, float time_s,
                int screen_width, int screen_height, SDL_GPUTexture* scene_color_copy,
                SDL_GPUTexture* scene_depth_copy, const SDL_GPUTextureSamplerBinding& ibl_specular,
                bool ibl_enabled, float sun_intensity);

    // ========== CPU/GPU uniform contracts (match the HLSL byte-for-byte) ==========

    struct alignas(16) SimUniforms {
        uint32_t resolution = SIM_RESOLUTION;
        float waveC2Dt2 = 0.0f;
        float damping = 0.9965f;
        float advectTexels = 0.0f;
        glm::vec2 flowDir{0.0f};
        uint32_t impulseCount = 0;
        float borderTexels = 24.0f;
        glm::vec4 impulses[MAX_IMPULSES]{};
    };
    static_assert(sizeof(SimUniforms) == 160, "must match water_sim.comp.hlsl cbuffer");

    struct alignas(16) VertexUniforms {
        glm::mat4 viewProj{1.0f};
        glm::vec4 simWindow{0.0f}; // xy = window min, z = 1/window size, w = unused
        glm::vec4 params{0.0f};    // x = water level, y = displacement, z = edge fade (uv)
    };
    static_assert(sizeof(VertexUniforms) == 96, "must match water.vert.hlsl cbuffer");

    struct alignas(16) FragmentUniforms {
        glm::vec4 cameraPosTime{0.0f};
        glm::vec4 sunDirIntensity{0.0f};
        glm::vec4 sunColorIbl{0.0f};
        glm::vec4 deepColor{0.0f};
        glm::vec4 shallowColor{0.0f};
        glm::vec4 fogHorizon{0.0f};
        glm::vec4 fogSun{0.0f};
        glm::vec4 screenDepthAB{0.0f};
        glm::vec4 flowDetail{0.0f};
        glm::vec4 simParams{0.0f};
        glm::vec4 misc{0.0f};
        glm::mat4 viewProjSSR{1.0f}; // world -> clip for the SSR march (draw VP)
        glm::vec4 ssrParams{0.0f};   // x = strength, y = max distance, z = thickness
    };
    static_assert(sizeof(FragmentUniforms) == 256, "must match water.frag.hlsl cbuffer");

    /** Grid density: cells along the longer region axis (the shorter axis
     *  scales by aspect, min 1). Exposed for the headless tests. */
    static glm::ivec2 grid_cells_for(float size_x, float size_z, int max_cells = 160);

private:
    void build_surface_mesh();
    void clear_sim_textures(SDL_GPUCommandBuffer* cmd);

    gpu::GPUDevice* device_ = nullptr;
    gpu::PipelineRegistry* registry_ = nullptr;
    bool initialized_ = false;
    bool has_region_ = false;
    WaterRegion region_{};
    EnvironmentState env_{};

    std::unique_ptr<gpu::GPUBuffer> vertex_buffer_;
    std::unique_ptr<gpu::GPUBuffer> index_buffer_;
    uint32_t index_count_ = 0;

    // Ping-pong wave state (RG16F: height, previous height).
    std::array<std::unique_ptr<gpu::GPUTexture>, 2> sim_state_;
    int sim_read_ = 0;         // texture read this substep; written one is ^1
    bool sim_cleared_ = false; // both textures zeroed (lazy, first simulate)
    float sim_time_accum_ = 0.0f;

    std::unique_ptr<gpu::GPUSampler> linear_clamp_sampler_;

    struct PendingImpulse {
        glm::vec2 pos_texels;
        float radius_texels;
        float strength;
    };
    std::vector<PendingImpulse> pending_impulses_;
};

} // namespace bracken::render
