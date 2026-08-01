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
    /// Surfaces rougher than this keep using the prefiltered IBL probe, which
    /// is indistinguishable at that point and far cheaper.
    float max_roughness = 0.35f;
    float normal_bias = 1.0f;
    float intensity = 1.0f;
};

struct RtStats {
    bool shadows_traced = false;
    bool reflections_traced = false;
    uint32_t shadow_rays = 0;      ///< dispatched threads, not rays actually cast
    uint32_t reflection_rays = 0;
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
    };

    /** Record the traced passes. Each is skipped entirely when its feature is
     *  off or the acceleration structure is not ready. */
    /// `ao_input` is the screen-space AO the shadow pass passes through in R
    /// (see rt_shadow.comp.hlsl); raw SDL handle because that is how the
    /// renderer already carries this texture between passes.
    void render(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth, SDL_GPUTexture* ao_input,
                const FrameParams& params, const RtShadowSettings& shadows,
                const RtReflectionSettings& reflections);

    /** Half/quarter-res visibility mask (1 = lit). Null when shadows are off. */
    [[nodiscard]] gpu::GPUTexture* shadow_mask() const;
    /** Traced reflection colour; alpha 0 means "use the IBL probe here". */
    [[nodiscard]] gpu::GPUTexture* reflection_target() const;

    [[nodiscard]] const RtStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const rt::RtScene& scene() const noexcept { return scene_; }

private:
    [[nodiscard]] bool ensure_target(gpu::GPUDevice& device, std::unique_ptr<gpu::GPUTexture>& tex,
                                     int width, int height, SDL_GPUTextureFormat format,
                                     const char* label);

    rt::RtScene scene_;
    gpu::GPUDevice* device_ = nullptr;
    gpu::PipelineRegistry* pipelines_ = nullptr;

    std::unique_ptr<gpu::GPUTexture> shadow_mask_;
    std::unique_ptr<gpu::GPUTexture> reflection_;
    int screen_width_ = 0;
    int screen_height_ = 0;
    RtResolution shadow_res_ = RtResolution::Half;
    RtResolution reflection_res_ = RtResolution::Quarter;
    bool built_with_attributes_ = false;
    bool warned_missing_attributes_ = false;
    bool warned_no_normal_buffer_ = false;
    RtStats stats_{};
};

} // namespace bracken::render
