#pragma once

#include "bracken/contracts/environment.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/render/raytracing/rt_scene.hpp"

#include <memory>

namespace bracken::render {

/** Resolution the traced signal is produced at, relative to the screen. */
enum class RtResolution : uint8_t {
    Full = 0,
    Half = 1,
    Quarter = 2,
};

[[nodiscard]] constexpr int rt_resolution_divisor(RtResolution r) {
    switch (r) {
    case RtResolution::Full: return 1;
    case RtResolution::Half: return 2;
    case RtResolution::Quarter: return 4;
    }
    return 2;
}

struct RtShadowSettings {
    bool enabled = false;
    RtResolution resolution = RtResolution::Half;
    /// Ray length. Shorter is cheaper AND misses distant occluders — the
    /// single most effective quality/cost dial here, because traversal cost
    /// scales with how much of the tree a ray can reach.
    float max_distance = 6000.0f;
    float normal_bias = 1.5f;
    float depth_bias = 0.5f;
};

struct RtReflectionSettings {
    bool enabled = false;
    RtResolution resolution = RtResolution::Quarter;
    float max_distance = 4000.0f;
    /// The traced/probe handover window, in linear roughness. At or below
    /// `fade_start` the traced reflection supplies the whole specular lobe; it
    /// falls to zero at `max_roughness`, above which the prefiltered IBL probe
    /// is indistinguishable and far cheaper. The forward shaders scale their
    /// IBL lobe by exactly one minus that weight, using the same function
    /// (`shaders/reflection_gate.hlsli`), so the two always sum to one.
    ///
    /// A hard cutoff put a visible seam through any surface whose roughness
    /// crossed the threshold — a full mirror on one side, the probe on the
    /// other. It also turned any small disagreement between the prepass
    /// G-buffer and the forward pass into a full-strength flip.
    ///
    /// 0.35 was the old cutoff, chosen when the G-buffer classified on the
    /// material's roughness FACTOR. Now that it samples the metallicRoughness
    /// map like the forward pass, the numbers it produces are the authored
    /// ones — and authored stone/brick sits at 0.93+, so nothing in a typical
    /// pack would ever have reflected at 0.35. 0.55 lets semi-glossy plaster
    /// and worn wood participate at partial weight without turning masonry
    /// into a mirror.
    float fade_start = 0.25f;
    float max_roughness = 0.55f;
    float normal_bias = 1.0f;
    float intensity = 1.0f;
};

/** Analytic receiver description for the displaced WaterRenderer surface.
 *  Keeping it here avoids putting water geometry into the opaque depth buffer
 *  before refraction has captured the scene behind it. */
struct RtWaterReceiver {
    glm::vec2 bounds_min{0.0f};
    glm::vec2 bounds_max{0.0f};
    float height = 0.0f;
    float roughness = 0.14f;
    glm::vec2 sim_min{0.0f};
    float sim_extent = 1.0f;
    float sim_displacement = 0.0f;
    float sim_normal_strength = 1.0f;
    glm::vec2 flow{0.0f};
    float flow_speed = 0.0f;
    float detail_wave_strength = 0.0f;
    float time = 0.0f;
};

struct RtStats {
    bool shadows_traced = false;
    bool reflections_traced = false;
    uint32_t shadow_rays = 0;      ///< dispatched threads, not rays actually cast
    uint32_t reflection_rays = 0;  ///< dispatched opaque + water threads
    std::size_t acceleration_bytes = 0;
};

/**
 * Compute-shader ray tracing for shadows and reflections.
 *
 * BOTH ARE INDEPENDENTLY OPTIONAL and both default OFF. Software ray tracing
 * is a large cost with a taste-dependent benefit, so it is opt-in per feature:
 * a game can trace shadows and keep IBL reflections, trace reflections and
 * keep cascaded shadow maps, take both, or take neither and pay nothing —
 * the acceleration structure is not even built unless something needs it.
 *
 * They share one RtScene, so enabling the second costs a second dispatch
 * rather than a second acceleration structure. Reflections additionally need
 * per-triangle attributes (normals and colours to shade a hit), which more
 * than doubles triangle memory — so those are gathered only when reflections
 * are enabled, and turning them off later does not silently keep paying.
 *
 * Tier: main-thread (owns GPU resources and records compute passes).
 */
class RtRenderer {
public:
    RtRenderer() = default;
    ~RtRenderer();
    RtRenderer(const RtRenderer&) = delete;
    RtRenderer& operator=(const RtRenderer&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device, gpu::PipelineRegistry& pipelines);
    void shutdown();

    /** Screen size changed: reallocate the traced targets. */
    void resize(gpu::GPUDevice& device, int width, int height);

    /**
     * Bring the acceleration structure up to date.
     *
     * `want_attributes` must be true if reflections may be enabled during this
     * session; flipping it later requires a rebuild, which this reports rather
     * than doing silently mid-frame.
     */
    void build_scene(gpu::GPUDevice& device, const ModelManager& models, bool want_attributes);
    void set_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances);

    struct FrameParams {
        glm::mat4 inv_view_proj{1.0f};
        glm::vec3 camera_position{0.0f};
        EnvironmentState environment{};
        int screen_width = 0;
        int screen_height = 0;
        float depth_a = 0.0f;
        float depth_b = 0.0f;
    };

    /** Record the traced passes. Each is skipped entirely when its feature is
     *  off or the acceleration structure is not ready. */
    /// `normal_rough` is the depth-prepass G-buffer (RG = octahedral world
    /// normal, B = roughness, A = metallic). Reflections REQUIRE it and are
    /// skipped when it is null; shadows ignore it entirely.
    ///
    /// The shadow pass used to also take the screen-space AO texture and copy
    /// it through into R of an RG8 mask, so the lighting stage could read AO
    /// and sun visibility from one sampler. It no longer does: the mask is R8
    /// visibility only. See rt_shadow.comp.hlsl for what that packing cost.
    void render(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                SDL_GPUTexture* normal_rough, const FrameParams& params, const RtShadowSettings& shadows,
                const RtReflectionSettings& reflections);

    /** Trace the visible part of a displaced water surface into the same raw
     * reflection target after the opaque resolve consumed it. The dispatch
     * analytically intersects camera rays with the water rectangle, refines
     * against the simulation displacement, rejects pixels hidden by opaque
     * depth, and derives the same wave normal as WaterRenderer. */
    [[nodiscard]] bool render_water_reflections(SDL_GPUCommandBuffer* cmd,
                                                gpu::GPUTexture& opaque_depth,
                                                gpu::GPUTexture& simulation_state,
                                                const FrameParams& params,
                                                const RtReflectionSettings& reflections,
                                                const RtWaterReceiver& water);

    /** Material/depth/normal-guided resolve into HDR. Call after forward
     * lighting and before water/TAA/bloom/exposure. */
    [[nodiscard]] bool resolve_reflections(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& hdr_scene,
                                           gpu::GPUTexture& depth, SDL_GPUTexture* normal_rough,
                                           const FrameParams& params,
                                           const RtReflectionSettings& reflections);

    /** Half/quarter-res R8 visibility mask (1 = lit). Null when shadows are
     *  off. Sample it with a NORMALIZED UV — its resolution is independent of
     *  both the screen and the depth buffer. */
    [[nodiscard]] gpu::GPUTexture* shadow_mask() const;
    /** Reduced-resolution raw incoming radiance. Alpha is the trace validity
     *  gate consumed by resolve_reflections() or the specialized water pass;
     *  hosts should not composite this target directly. */
    [[nodiscard]] gpu::GPUTexture* reflection_target() const;

    [[nodiscard]] const RtStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const rt::RtScene& scene() const noexcept { return scene_; }

private:
    [[nodiscard]] bool ensure_target(gpu::GPUDevice& device, std::unique_ptr<gpu::GPUTexture>& tex,
                                     int width, int height, SDL_GPUTextureFormat format,
                                     const char* label);
    [[nodiscard]] bool trace_reflections(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                                         SDL_GPUTexture* receiver_data,
                                         const FrameParams& params,
                                         const RtReflectionSettings& reflections,
                                         const RtWaterReceiver* water);

    rt::RtScene scene_;
    gpu::GPUDevice* device_ = nullptr;
    gpu::PipelineRegistry* pipelines_ = nullptr;
    SDL_GPUSampler* nearest_clamp_sampler_ = nullptr;

    std::unique_ptr<gpu::GPUTexture> shadow_mask_;
    std::unique_ptr<gpu::GPUTexture> reflection_;
    int screen_width_ = 0;
    int screen_height_ = 0;
    RtResolution shadow_res_ = RtResolution::Half;
    RtResolution reflection_res_ = RtResolution::Quarter;
    bool built_with_attributes_ = false;
    bool warned_missing_attributes_ = false;
    RtStats stats_{};
};

} // namespace bracken::render
