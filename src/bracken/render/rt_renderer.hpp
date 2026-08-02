#pragma once

#include "bracken/contracts/environment.hpp"
#include "bracken/contracts/graphics_settings.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/render/raytracing/rt_scene.hpp"

#include <array>
#include <memory>

namespace bracken::gpu {
// Forward-declared rather than including gpu_debug.hpp: this header only holds
// a pointer to the scene renderer's pass timeline (see set_timer_pool).
class GPUTimestampPool;
} // namespace bracken::gpu

namespace bracken::render {

// RtResolution is a GraphicsSettings field (contracts/graphics_settings.hpp),
// so it lives in the leaf layer both the host and this renderer can agree on
// rather than being defined here and mirrored there.

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
    /// Angular RADIUS of the sun disk, in degrees. The real sun is ~0.265;
    /// larger is softer. 0 collapses the cone and reproduces the old hard
    /// shadow exactly, which is what makes an A/B against it possible.
    ///
    /// This is the knob that turns one ray per pixel into a penumbra: the ray
    /// is jittered across the disk, and the denoiser integrates the result
    /// over time. Widening it costs nothing to trace and only makes the
    /// denoiser work harder.
    float sun_angular_radius_degrees = 0.35f;
    /// Frames of temporal accumulation to converge over. Higher is cleaner and
    /// lags further behind moving occluders.
    float max_accum_frames = 32.0f;
    /// Spatial filter radius, in traced texels, at maximum variance.
    float filter_radius = 3.0f;
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
    ///
    /// 0.55 was the ceiling while the trace cast a single MIRROR ray, because
    /// a mirror ray is only the right answer for a mirror — on anything
    /// rougher it returns one arbitrary point of a wide lobe, which reads as
    /// a too-sharp, too-bright reflection. Now that the trace importance-
    /// samples the GGX lobe and the result is temporally accumulated, a rough
    /// surface gets the lobe it actually has, so the handover can move up to
    /// where the prefiltered probe genuinely becomes indistinguishable.
    float fade_start = 0.35f;
    float max_roughness = 0.75f;
    float normal_bias = 1.0f;
    float intensity = 1.0f;
    /// Frames of temporal accumulation for the GGX-sampled signal. Shorter
    /// than the shadow denoiser's: reflected content moves at roughly twice
    /// the parallax rate of the surface carrying it, so a long tail smears
    /// more than it cleans.
    float max_accum_frames = 16.0f;
    /// How far reprojected history may sit outside its neighbourhood's colour
    /// standard deviation before being clamped back. Larger keeps more history
    /// (cleaner, more ghosting behind moving reflections).
    float clamp_sigma = 1.5f;
};

/**
 * Ray-traced shadows for LOCAL (point/spot) lights — the Ratio tier of
 * docs/rt_lighting_plan.md §5.3.
 *
 * ONE ray per pixel, aimed by a froxel-RIS reservoir rather than by a loop, so
 * the cost does not scale with the light count and neither the atlas's 8-point /
 * 16-spot budgets nor its 6-tile-per-frame refresh cap apply. Requires the
 * frame's clustered-lighting buffers (RtClusterBindings) and the prepass
 * G-buffer; the pass is skipped rather than approximated when either is absent.
 */
struct RtLocalShadowSettings {
    bool enabled = false;
    RtResolution resolution = RtResolution::Half;
    float normal_bias = 1.5f;
    float depth_bias = 0.5f;
    /// Emitter radius in WORLD UNITS — the sphere the shadow ray samples a
    /// point on, and therefore the penumbra-width dial. 10 units is 10 cm at
    /// the playground's 100 units/metre: a lantern flame or a torch head. One
    /// value for every light until per-light authoring arrives with the area
    /// lights of R6; a per-light radius belongs on PointLightDesc, not here.
    float source_radius = 10.0f;
    /// M — candidates drawn from the pixel's froxel before one survives. The
    /// froxel list is short (the playground's densest is single digits), so
    /// beyond ~8 the extra draws mostly re-pick lights already considered.
    uint32_t candidate_count = 6;
    /// Cap on carried-over reservoir M, as a multiple of `candidate_count`.
    /// 20x is the ReSTIR-standard figure: enough inertia to behave like dozens
    /// of rays, little enough that a light that moves is followed within a
    /// handful of frames.
    float history_max_m_ratio = 20.0f;
    /// Denoiser knobs, same meaning as RtShadowSettings' — the two signals go
    /// through the same three pipelines.
    float max_accum_frames = 32.0f;
    float filter_radius = 3.0f;
};

/**
 * The frame's clustered-lighting buffers, as the local-shadow trace needs them.
 *
 * Borrowed for the duration of one render() call and never retained. `params`
 * is a `render::lighting::ClusterParams` passed as opaque bytes — the same
 * arrangement TerrainRenderer::set_cluster_lighting uses, and for the same
 * reason: this header has no business dragging the light-binning module into
 * everything that includes it.
 */
struct RtClusterBindings {
    SDL_GPUBuffer* light_data = nullptr;
    SDL_GPUBuffer* cluster_offsets = nullptr;
    SDL_GPUBuffer* light_indices = nullptr;
    /// NOT the packed stable ids. They are ClusterGrid's fifth cluster buffer
    /// and nothing here binds them: SDL3 GPU allows a compute pipeline eight
    /// read-only storage buffers, the four acceleration ones are mandatory, and
    /// a temporal reservoir's question is the REMAP's — "where did the light I
    /// picked last frame go?" — not the ids'. See cluster_compute.hlsli's
    /// CLUSTER_COMPUTE_WANT_STABLE_IDS.
    SDL_GPUBuffer* light_remap = nullptr;
    const void* params = nullptr;
    uint32_t params_size = 0;
    /// Lights the frame actually binned. Zero means the CPU skipped binning, in
    /// which case every buffer above holds LAST frame's contents.
    uint32_t total_lights = 0;

    [[nodiscard]] bool complete() const {
        return light_data != nullptr && cluster_offsets != nullptr && light_indices != nullptr &&
               light_remap != nullptr && params != nullptr &&
               params_size > 0 && total_lights > 0;
    }
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

/**
 * What the traced signals cost this frame, and what the structure they walk
 * costs to hold.
 *
 * The byte figure used to be one number, `acceleration_bytes`, which answered
 * "how much" but never "of what" — and the three components have completely
 * different levers. BLAS bytes are a property of the loaded models; attribute
 * bytes exist only because reflections are on (64 B per unique triangle, the
 * single largest item when they are); TLAS bytes scale with the INSTANCE count
 * and are therefore the part `rt_max_instance_range` moves. `acceleration_bytes`
 * is kept as their sum so existing readers keep working and so the split can be
 * checked against it.
 */
struct RtStats {
    bool shadows_traced = false;
    bool reflections_traced = false;
    /// Whether the froxel-RIS local-light shadow pass ran. False whenever the
    /// mode is Off, the cluster buffers were incomplete, or the scene had no
    /// binned lights — which is also the signal that the rasterized atlas is
    /// still the thing shadowing local lights this frame.
    bool local_shadows_traced = false;
    uint32_t shadow_rays = 0;      ///< dispatched threads, not rays actually cast
    uint32_t reflection_rays = 0;  ///< dispatched opaque + water threads
    /// Dispatched threads of the local-light pass. One RESERVOIR update each;
    /// the ray itself is cast only where the froxel held a candidate, so the
    /// true ray count is lower and this is the same honest over-estimate
    /// `shadow_rays` is.
    uint32_t local_shadow_rays = 0;
    /// Instances in the TLAS — i.e. AFTER the renderer's range filter, so this
    /// is what the rays actually see rather than what the frame drew.
    uint32_t instances = 0;
    /// How many times the TLAS has been rebuilt since startup. A PERFORMANCE
    /// contract, not a curiosity: a scene whose instance SET is stable must
    /// hold at 1 no matter how many frames pass or how much moves, and a finite
    /// rt_max_instance_range is the documented way to make it climb. Mirrors
    /// rt::RtSceneStats::tlas_builds.
    uint32_t tlas_builds = 0;
    /// How many times it was refitted instead — same instances, new transforms.
    /// Only meaningful next to `tlas_builds`: builds≈1 with refits≈frames is
    /// the healthy reading for a live scene. Mirrors
    /// rt::RtSceneStats::tlas_refits.
    uint32_t tlas_refits = 0;
    std::size_t blas_bytes = 0;       ///< concatenated BLAS nodes + triangles
    std::size_t tlas_bytes = 0;       ///< TLAS nodes + instance records
    std::size_t attribute_bytes = 0;  ///< per-triangle shading attributes; 0 without reflections
    /// blas_bytes + tlas_bytes + attribute_bytes.
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

    /**
     * Adopt the frame's per-pass timeline, so the traced passes report into
     * `RenderStats::pass_times_ms` alongside every other pass.
     *
     * OPTIONAL, and null is the default: this renderer is constructible and
     * usable without one (tests, a host driving it directly), in which case its
     * timers are no-ops. The pool is owned by the caller — SceneRenderer — and
     * must outlive this object.
     */
    void set_timer_pool(gpu::GPUTimestampPool* pool) noexcept { timers_ = pool; }

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
    /// @param cmd The frame's command buffer, so a changed TLAS is written into
    ///   it rather than through its own submit. Null is valid (host calling
    ///   outside a frame); see RtScene::upload.
    void set_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances,
                       SDL_GPUCommandBuffer* cmd = nullptr);

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
    ///
    /// `velocity` is the prepass velocity buffer, which the shadow denoiser
    /// reprojects its history through. Without it the denoiser still runs but
    /// cannot accumulate across frames, so the result is the raw one-sample
    /// estimate spatially filtered — noticeably noisier, and it says so once
    /// in the log rather than looking like a shader bug.
    ///
    /// `local_shadows` additionally needs `clusters` — the frame's binned light
    /// buffers. It is skipped, not degraded, when they are incomplete: a
    /// reservoir drawn from a stale froxel list would shadow against lights
    /// that are not in the frame.
    void render(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                SDL_GPUTexture* normal_rough, SDL_GPUTexture* velocity,
                const FrameParams& params, const RtShadowSettings& shadows,
                const RtReflectionSettings& reflections,
                const RtLocalShadowSettings& local_shadows = {},
                const RtClusterBindings& clusters = {});

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
    /** Full-res R8 local-light visibility ratio (1 = lit). Null when the local
     *  pass is off or has not produced a mask. The forward shaders multiply
     *  their whole point+spot subtotal by it — see cluster_lighting.hlsli. */
    [[nodiscard]] gpu::GPUTexture* local_shadow_mask() const;
    /** Reduced-resolution raw incoming radiance. Alpha is the trace validity
     *  gate consumed by resolve_reflections() or the specialized water pass;
     *  hosts should not composite this target directly. */
    [[nodiscard]] gpu::GPUTexture* reflection_target() const;
    /** Per-pixel normalized reflection hit distance (`hit.t / max_distance`,
     *  1.0 on a miss), R16F at the reflection target's resolution. Null until
     *  a reflection trace has run. See the member for why it exists. */
    [[nodiscard]] gpu::GPUTexture* reflection_hit_distance() const;

    [[nodiscard]] const RtStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const rt::RtScene& scene() const noexcept { return scene_; }

private:
    /// Copy RtScene's accounting (instances, TLAS builds, the byte split) into
    /// stats_. Called after every upload so the two can never disagree.
    void publish_scene_stats();
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
    /// Borrowed, may be null — see set_timer_pool().
    gpu::GPUTimestampPool* timers_ = nullptr;
    SDL_GPUSampler* nearest_clamp_sampler_ = nullptr;
    /// Bilinear, for the denoiser's sub-texel history reprojection only.
    SDL_GPUSampler* linear_clamp_sampler_ = nullptr;

    /**
     * Every target and every flag one STOCHASTIC VISIBILITY signal needs on its
     * way from packed bits to a full-resolution R8 mask.
     *
     * There are two such signals — the sun, and the froxel-RIS local lights —
     * and they are the same mechanism, not two mechanisms that resemble each
     * other: both produce one bit per pixel packed 8x4 per uint, and both are
     * turned back into an image by the SAME three pipelines
     * (RtShadowClassify/Temporal/Filter), dispatched twice against two of these
     * structs. Forking the denoiser would have meant three more pipelines,
     * three more catalogue rows and three more shaders to keep in step with the
     * originals, in exchange for nothing: nothing in those passes is
     * sun-specific. If something ever needs to be, it becomes a uniform.
     */
    struct VisibilityDenoiser {
        /// One bit of raw traced visibility per pixel, 8x4 per uint (R32_UINT),
        /// so the classifier reads 32 pixels' verdicts in one fetch.
        std::unique_ptr<gpu::GPUTexture> bits;
        /// Per-tile verdict: 0 uniformly lit, 1 uniformly shadowed, 2 mixed.
        std::unique_ptr<gpu::GPUTexture> tile_class;
        /// Ping-ponged accumulation (RGBA16F): mean, second moment, sample
        /// count, linear depth. Two because a temporal filter cannot read and
        /// write the same texture in one dispatch.
        std::array<std::unique_ptr<gpu::GPUTexture>, 2> history;
        /// Full-res R8 output — the denoiser's last pass fuses the upsample into
        /// the filter, so the forward shaders sample it 1:1.
        std::unique_ptr<gpu::GPUTexture> mask;
        int history_index = 0;
        /// False until a history has been written at this size. Cleared on
        /// resize, on a resolution change, and whenever the signal was off last
        /// frame — in all three cases the history describes a different image.
        bool history_valid = false;
        RtResolution resolution = RtResolution::Half;
        /// Advances the sample sequence. Wraps freely; only its low bits matter
        /// to the R2 sequence, and wrapping is what keeps it exact in a float.
        /// PER SIGNAL, so the two never sample in lockstep.
        uint32_t frame_index = 0;

        void release_targets();
        void release();
    };

    /// Records the three denoiser dispatches for one signal. Split out of
    /// render() because the trace is one dispatch and the denoiser is most of
    /// the pass.
    void denoise_visibility(SDL_GPUCommandBuffer* cmd, VisibilityDenoiser& d, gpu::GPUTexture& depth,
                            SDL_GPUTexture* velocity, const FrameParams& params, float max_accum_frames,
                            float filter_radius, int traced_w, int traced_h);
    /// Trace + denoise the froxel-RIS local-light signal. Returns whether the
    /// trace ran.
    [[nodiscard]] bool render_local_shadows(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                                            SDL_GPUTexture* normal_rough, SDL_GPUTexture* velocity,
                                            const FrameParams& params,
                                            const RtLocalShadowSettings& settings,
                                            const RtClusterBindings& clusters);
    /// Temporal accumulation of the GGX-sampled reflection signal. Opaque path
    /// only — see reflection_denoised_this_frame_.
    void denoise_reflections(SDL_GPUCommandBuffer* cmd, SDL_GPUTexture* velocity,
                             const RtReflectionSettings& reflections);

    std::unique_ptr<gpu::GPUTexture> reflection_;
    /// NOTHING CONSUMES THIS YET. The reflection trace writes each pixel's
    /// normalized hit distance here (R16F, alongside `reflection_`) because it
    /// is the one place in the frame where that number is free, and R3 of the
    /// ray-traced lighting plan needs it: virtual-hit-point reprojection
    /// reprojects the REFLECTED point — the eye ray extended through the
    /// surface by the hit distance — instead of the surface carrying it, which
    /// is what stops low-roughness reflections smearing under camera motion.
    /// Producing it now means that lands as a denoiser change rather than as a
    /// change to the traversal everything else depends on.
    std::unique_ptr<gpu::GPUTexture> reflection_hit_distance_;

    // ---- the two stochastic visibility signals ----
    VisibilityDenoiser sun_shadow_;
    VisibilityDenoiser local_shadow_;
    /// Ping-ponged per-pixel reservoirs at the LOCAL pass's traced resolution,
    /// 16 B each (render/lighting/reservoir.hpp). Not textures: the record is
    /// four heterogeneous lanes, one of which is a light INDEX that must not be
    /// filtered, converted or clamped by anything.
    std::array<std::unique_ptr<gpu::GPUBuffer>, 2> local_reservoirs_;
    int local_reservoir_index_ = 0;
    /// Element count the pair is currently sized for; a change reallocates and
    /// invalidates the history, because reservoir N describes a different pixel.
    int local_reservoir_capacity_ = 0;

    // ---- reflection denoiser state ----
    /// Ping-ponged accumulation (RGBA16F): rgb = radiance, a = sample count.
    /// The count lives in alpha because rt_reflection_resolve.frag treats
    /// alpha purely as a validity test and never as a weight, so "0 samples"
    /// and "not reflective" are the same statement there.
    std::array<std::unique_ptr<gpu::GPUTexture>, 2> reflection_history_;
    int reflection_history_index_ = 0;
    bool reflection_history_valid_ = false;
    /// Whether the temporal pass ran THIS frame. The water reflection path
    /// writes the raw target with different alpha semantics (an SSR/IBL
    /// hand-off weight, not a count), so it is deliberately not denoised and
    /// the resolve must fall back to the raw target for it.
    bool reflection_denoised_this_frame_ = false;
    uint32_t reflection_frame_index_ = 0;

    int screen_width_ = 0;
    int screen_height_ = 0;
    RtResolution reflection_res_ = RtResolution::Quarter;
    bool built_with_attributes_ = false;
    bool warned_missing_attributes_ = false;
    RtStats stats_{};
};

} // namespace bracken::render
