#pragma once

#include "bracken/contracts/environment.hpp"
#include "bracken/contracts/graphics_settings.hpp"
#include "bracken/contracts/shadow_capsule.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_compute.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include "bracken/render/raytracing/rt_scene.hpp"

#include <array>
#include <memory>
#include <span>

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
    /**
     * The Full-DI tier (R5c): the same reservoir and the same ray, but the pass
     * ALSO writes two demodulated radiance targets and the forward shaders drop
     * the analytic point/spot loop entirely.
     *
     * Requires exactly what `enabled` requires and nothing more — the
     * acceleration structure and the prepass G-buffer — so there is no separate
     * ensure path and no way for this to be on while the trace is off. It is a
     * second flag rather than a mode enum here because `enabled` is what gates
     * the whole pass and every caller already sets it.
     */
    bool full_di = false;
    RtResolution resolution = RtResolution::Half;
    float normal_bias = 1.5f;
    float depth_bias = 0.5f;
    /// Emitter radius in WORLD UNITS — the sphere the shadow ray samples a
    /// point on, and therefore the penumbra-width dial. 10 units is 10 cm at
    /// the playground's 100 units/metre: a lantern flame or a torch head.
    ///
    /// Since R6 this is the DEFAULT rather than the only value: a host that
    /// sets `PointLightDesc::source_radius` (or the spot equivalent) overrides
    /// it per light, and 0 there means "use this". Every pre-R6 scene leaves
    /// the desc field at 0 and therefore sees exactly what it saw before.
    float source_radius = 10.0f;
    /**
     * M_e — emissive-mesh light candidates drawn per pixel, on top of the
     * `candidate_count` analytic ones (R6, docs/rt_lighting_plan.md §5.7).
     *
     * FULL TIER ONLY. The Ratio tier's mask is a scalar multiplying the forward
     * pass's analytic subtotal, and no value of that scalar can ADD a light the
     * subtotal does not contain — so emissive records are simply not sent when
     * `full_di` is false, and this field is inert there. The tier note in
     * rt_local_shadow.comp.hlsl's header carries the full argument.
     *
     * 2 is the shipped default and the plan's lower bound. Each draw is already
     * an importance-sampled choice rather than a uniform one (see
     * `emissive_froxel_share`), so past ~4 the extra draws mostly re-pick the
     * same bright emitters and the reservoir's temporal reuse is the cheaper
     * way to buy the same convergence.
     *
     * **0 turns the emissive candidate stream off**, which is the A/B control
     * §5.7's cost table is measured against: with no records sent every RIS
     * weight in the pass collapses to its R5c form bit for bit.
     */
    uint32_t emissive_candidate_count = 2;
    /**
     * Fraction of each emissive candidate draw taken from the pixel's PER-FROXEL
     * emissive list rather than from the scene-wide power-weighted alias table
     * (R8, docs/rt_lighting_plan.md §5.7 follow-up 2).
     *
     * The two are a MIXTURE over the same domain, not two streams: a record can
     * be reachable both ways and its sampling pdf is the sum of the two paths'
     * contributions. That is what keeps the estimator unbiased over the FULL
     * record set while still aiming three draws in four at emitters that are
     * actually near the pixel — which is what shortens the shadow ray.
     *
     * **0 reproduces R6's purely global draw exactly** (the mixture collapses to
     * its second component), and is the before column of that measurement. 1
     * would drop the global component and with it the coverage guarantee for an
     * emitter no froxel list holds, so it is not offered as a quality level.
     */
    float emissive_froxel_share = 1.0f;
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

    // ---- Full-DI tier only (R5c) ----

    /// Frames the DI radiance denoiser accumulates over. Longer than the
    /// reflection denoiser's 16: direct lighting is painted on the surface
    /// carrying it, so it does not move at twice the parallax rate and a long
    /// tail costs no smear.
    float di_max_accum_frames = 32.0f;
    /// Neighbourhood standard deviations the reprojected DI history may sit
    /// outside before being clamped back — the reflection denoiser's dial, same
    /// meaning and the same default.
    float di_clamp_sigma = 1.5f;
    /// Radius in TRACED texels of the DI spatial gather at one accumulated
    /// sample. `render::lighting::di_filter_radius` maps the count down from
    /// here to its floor.
    float di_filter_radius = 3.0f;

    // ---- R7: spatial reservoir reuse, a second filter iteration, the probe ----

    /**
     * k — spatial reservoir-reuse neighbours per pixel (R7a,
     * docs/rt_lighting_plan.md §5.3's deferred "3-5 neighbours, pairwise MIS").
     *
     * APPLIES TO BOTH TIERS, deliberately. The reuse improves WHICH LIGHT the
     * reservoir picks, and both tiers spend that choice: Ratio's mask is the
     * winner's visibility bit, Full's estimate is the winner's radiance. Gating
     * it to Full would have meant two sampling paths behind one gradient replay,
     * which is the arrangement §5.3 spent R2b avoiding.
     *
     * 0 turns it off and reproduces the pre-R7a image BIT for bit — the
     * pairwise merge returns the canonical unchanged and the ray's sample
     * stream is untouched. 3 is the plan's lower bound and the shipped default;
     * past ~5 the neighbours start agreeing with each other faster than they add
     * information, and each one costs three target-function evaluations plus a
     * surface reconstruction.
     */
    uint32_t spatial_neighbors = 3;
    /// Radius of the spatial neighbour ring, in TRACED texels. At half res that
    /// is ~16 screen pixels — wide enough to reach a genuinely different
    /// candidate set, narrow enough that the depth/normal gate usually passes.
    float spatial_radius = 8.0f;
    /**
     * A-trous iterations of the DI spatial filter (R7c). 1 reproduces the R5c
     * chain; 2 adds a stride-2 pass that only touches texels whose accumulation
     * is still below `render::lighting::DI_FILTER_CONVERGED_COUNT`, so a
     * converged image stays exactly as sharp as it was.
     */
    uint32_t di_spatial_iterations = 2;
    /**
     * Run the temporal-noise probe over the filtered DI targets (R7).
     *
     * OFF by default and meant to stay that way outside a measurement: it is a
     * whole extra dispatch plus an R16F target, spent to produce one number the
     * frame does not use. `--rt-bench` turns it on; nothing else should.
     * Reported through `RtStats::DenoiseStats::di_noise*`.
     */
    bool di_noise_probe = false;
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
    /// R8's per-froxel EMISSIVE candidate table
    /// (render/raytracing/emissive_clusters.hpp), or null when it was not built
    /// this frame. Optional, deliberately: null makes the trace send
    /// `emissiveFroxelShare = 0`, which is R6's global draw — correct, merely
    /// noisier and longer-rayed. It is therefore NOT part of `complete()`.
    SDL_GPUBuffer* emissive_froxels = nullptr;
    /// Froxel-list entries the table actually holds. Zero is treated as null.
    uint32_t emissive_froxel_entries = 0;
    /// The other two halves of `rt::EmissiveClusterStats`, carried here for one
    /// reason only: the binner belongs to the caller (it needs the light grid's
    /// froxel bounds) and the stats LINE belongs to this renderer, so they ride
    /// the struct that already crosses that boundary rather than growing a
    /// second path. Reported verbatim in `RtStats`.
    uint32_t emissive_froxels_filled = 0;
    uint32_t emissive_froxel_dropped = 0;
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

/**
 * Everything `shaders/rt_hit_shade.hlsli` needs to shade a reflection-ray hit
 * honestly — phase R3 of docs/rt_lighting_plan.md §5.4.
 *
 * Before R3 a reflection hit was `albedo * (sunColor * N·L + skyColor * 0.25)`:
 * an unshadowed sun and a flat ambient constant. Every field below exists to
 * replace one of those two with the answer the rest of the frame already has —
 * the CSM atlas the forward pass filters, the light buffer it loops, the probe
 * field it samples for ambient. All of it is already built by the time the
 * traced passes run; none of it is retained past one `render()` call.
 *
 * The GI half degrades rather than branching: with `gi_mode == Off` the caller
 * binds this renderer's own 1×1 dummies and leaves `gi_enabled` at 0, the probe
 * term is skipped, and the hit's ambient falls to the `EnvironmentState`
 * hemisphere. That is deliberately still an improvement on the flat constant —
 * reflections get shadowed sun and local lights with GI off as well as on.
 *
 * Tier: main-thread (live GPU handles, valid for exactly one frame).
 */
struct RtHitShadeBindings {
    /// This frame's CSM uniforms, exactly as the forward pass received them.
    /// The trace shader embeds them and calls the shared `calcShadow`.
    gpu::ShadowDataUniforms shadow{};
    SDL_GPUTexture* shadow_atlas = nullptr;
    SDL_GPUSampler* shadow_compare_sampler = nullptr;
    SDL_GPUSampler* shadow_raw_sampler = nullptr;

    /// IBL sky-capture cube (raw radiance) for ray misses, or null to fall back
    /// to the EnvironmentState hemisphere. `sky_cube_fallback` must always be a
    /// valid cube texture — a binding slot cannot be left empty.
    SDL_GPUTexture* sky_cube = nullptr;
    SDL_GPUTexture* sky_cube_fallback = nullptr;
    SDL_GPUSampler* sky_sampler = nullptr;

    /// The probe field's irradiance atlas and per-probe state plane, banded
    /// (render/gi/gi_math.hpp). Null on both means "GI is off" and this
    /// renderer substitutes its own dummies; `gi_enabled` is the flag the
    /// shader actually reads.
    SDL_GPUTexture* gi_irradiance = nullptr;
    SDL_GPUSampler* gi_irradiance_sampler = nullptr;
    SDL_GPUTexture* gi_probe_state = nullptr;
    /// `render::gi::cascade_base()` per cascade, xyz used. Sent rather than
    /// re-derived, for the reason `gi_sample.hlsli` spells out.
    std::array<glm::ivec4, 3> gi_cascade_base{};

    /// The clustered-lighting byte buffer and its counts. A ray hit reads the
    /// FLAT list, not the froxel grid — a world-space point has no froxel, and
    /// the grid would not fit inside SDL3 GPU's eight read-only storage buffers
    /// next to the four acceleration ones.
    SDL_GPUBuffer* light_data = nullptr;
    std::uint32_t light_count = 0;
    std::uint32_t light_capacity = 0;

    /// Whether the probe atlases above are a live field. Zero is a valid,
    /// tested configuration, not a failure.
    [[nodiscard]] bool gi_ready() const {
        return gi_irradiance != nullptr && gi_probe_state != nullptr &&
               gi_irradiance_sampler != nullptr;
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
    /**
     * Whether the local pass ran in the Full-DI tier this frame — i.e. whether
     * the two demodulated radiance targets below are live and the forward
     * shaders dropped their analytic point/spot loop.
     *
     * Distinct from `local_shadows_traced`, which is true in both traced tiers:
     * this is the one that says which of the two the frame you are looking at
     * actually used. Both targets are RGBA16F at the LOCAL pass's traced
     * resolution — RGB demodulated radiance, and the alpha lanes carrying the
     * texel's linear view depth and the Schlick fraction the forward shaders
     * rebuild a metal's Fresnel from (render/lighting/local_di.hpp).
     */
    bool local_di_traced = false;
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
    std::size_t emissive_bytes = 0;   ///< emissive light records; 0 without emitters
    /// blas_bytes + tlas_bytes + attribute_bytes + emissive_bytes.
    std::size_t acceleration_bytes = 0;

    /// R6, emissive mesh lights (docs/rt_lighting_plan.md §5.7). Mirrors
    /// rt::RtSceneStats: how many MODEL triangles cleared the power threshold,
    /// how many scene-space records they flattened into, and how many the cap
    /// threw away (a running total — non-zero means the budget is binding).
    uint32_t emissive_triangles = 0;
    uint32_t emissive_records = 0;
    uint32_t emissive_records_dropped = 0;
    /// Whether emissive records actually took part in this frame's candidate
    /// draw. FALSE in the Ratio tier even when records exist: see the tier note
    /// on RtLocalShadowSettings::full_di — a ratio estimator cannot represent a
    /// light the analytic forward loop does not evaluate.
    bool emissive_di_sampled = false;
    /// R8, per-froxel emissive candidate lists (§5.7 follow-up 2). Mirrors
    /// rt::EmissiveClusterStats: froxels holding at least one record, entries
    /// packed across all of them, and record-froxel associations that did not
    /// fit the per-froxel cap. A non-zero drop count is a QUALITY figure and not
    /// an error — those records stay reachable through the global half of the
    /// mixture — but it is reported rather than swallowed, per the no-silent-
    /// drops rule. All three are 0 whenever the table was not built.
    uint32_t emissive_froxels_filled = 0;
    uint32_t emissive_froxel_entries = 0;
    uint32_t emissive_froxel_dropped = 0;
    /// Whether the trace actually drew through the froxel lists this frame
    /// (`emissiveFroxelShare > 0`). False means R6's purely global draw.
    bool emissive_froxel_local = false;

    /// A-SVGF gradient rays dispatched this frame, per signal — one thread per
    /// 3x3 stratum of the traced resolution, so exactly ceil(w/3)*ceil(h/3).
    /// The same honest over-estimate `shadow_rays` is: a stratum whose
    /// reprojection failed, whose surface changed, or whose reservoir held no
    /// light returns without casting.
    uint32_t shadow_gradient_rays = 0;
    uint32_t local_shadow_gradient_rays = 0;

    /**
     * What the two visibility denoisers actually did, from a LAGGED readback of
     * the GPU-side counters (`RtRenderer::set_stats_readback_interval`).
     *
     * These are the only numbers that can show a denoiser lagging: a static
     * screenshot cannot, because a lagging denoiser and a converged one look
     * identical on a frame where nothing has changed for thirty frames. Under a
     * sun sweep the accumulation counts collapse and the fire fraction climbs;
     * with a static sun they must not move at all, which is the regression
     * test.
     *
     * `*_valid` is false until the first readback has landed (a second or so
     * after the signal is first enabled) and whenever the interval between two
     * readbacks contained no dispatch of that signal.
     */
    struct DenoiseStats {
        /// Mean accumulated sample count over pixels in MIXED tiles — the only
        /// pixels that accumulate at all, since a uniform tile writes the
        /// converged answer directly.
        float shadow_mean_accum = 0.0f;
        float local_mean_accum = 0.0f;
        /// Fraction of traced pixels whose reconstructed gradient cleared
        /// `render::GRADIENT_FIRE_THRESHOLD`, i.e. whose accumulation was cut.
        float shadow_gradient_fire = 0.0f;
        float local_gradient_fire = 0.0f;
        bool shadow_valid = false;
        bool local_valid = false;

        /// R7's temporal-noise probe over the FILTERED Full-DI targets, valid
        /// only when `RtLocalShadowSettings::di_noise_probe` was set. The one
        /// number the smoke test structurally cannot produce: mean absolute
        /// frame-to-frame delta of the demodulated local-light estimate, and the
        /// mean level it is a delta of, so the figure reads both absolutely and
        /// as a fraction. Under a pinned camera this IS the temporal noise.
        float di_noise_mean_delta = 0.0f;
        float di_noise_mean_level = 0.0f;
        bool di_noise_valid = false;
    };
    DenoiseStats denoise{};
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
    /// @param skinned_first Index of the first SKINNED instance in a list
    ///   partitioned static-then-skinned; see RtScene::set_instances. Skinned
    ///   instances are excluded from the ANY-HIT walks (their BLAS is the bind
    ///   pose) and stand in through upload_shadow_capsules() instead.
    void set_instances(std::span<const std::pair<ModelHandle, glm::mat4>> instances,
                       SDL_GPUCommandBuffer* cmd = nullptr,
                       uint32_t skinned_first = 0xFFFFFFFFu);

    /**
     * Take this frame's analytic capsule occluders (docs/rt_lighting_plan.md
     * §5.8) and record their upload into an ALREADY-OPEN copy pass.
     *
     * Call once per frame from the renderer's batched frame-upload pass,
     * before render(). The span is copied; nothing is retained. An empty span
     * is valid and is the normal state for a host with no animated characters
     * on screen — the count uniform goes to zero and the trace's capsule loop
     * collapses to nothing.
     *
     * The buffer is grow-only: a frame whose capsule count fits the existing
     * allocation reallocates nothing, which is the steady state, since the
     * count is a function of how many characters are on screen and not of what
     * they are doing.
     */
    void upload_shadow_capsules(SDL_GPUCopyPass* copy_pass,
                                std::span<const rt::ShadowCapsule> capsules);

    struct FrameParams {
        glm::mat4 inv_view_proj{1.0f};
        /// UNJITTERED, current and previous — the exact pair the velocity
        /// buffer was built from (contracts/velocity_math.hpp). R3's
        /// virtual-hit-point reprojection subtracts one projection of the same
        /// world point from the other, so the TAA jitter cancels inside the
        /// difference instead of having to be removed from either.
        glm::mat4 view_proj_unjittered{1.0f};
        glm::mat4 prev_view_proj_unjittered{1.0f};
        /// Previous frame's VIEW matrix. The local gradient replay needs this
        /// specifically: the reservoir stores a true view depth, not a
        /// linearized device depth, so its "was this the same surface?" test
        /// re-measures the world point with the previous camera directly rather
        /// than going back through a projection.
        glm::mat4 prev_view{1.0f};
        glm::vec3 camera_position{0.0f};
        /// Unit view direction. `calcShadow` selects a cascade by VIEW depth,
        /// and a reflection-ray hit has to be projected onto this to have one.
        glm::vec3 camera_forward{0.0f, 0.0f, -1.0f};
        EnvironmentState environment{};
        /// GraphicsSettings::sun_intensity. It lives outside EnvironmentState
        /// and every consumer multiplies it in at its own use site, so this
        /// pass has to be told. Getting it wrong makes a reflected surface
        /// disagree with the same surface drawn directly by a factor of three
        /// in the playground — which is exactly what happened before R3, where
        /// it cancelled the equally-missing Lambertian 1/pi.
        float sun_intensity = 1.0f;
        int screen_width = 0;
        int screen_height = 0;
        float depth_a = 0.0f;
        float depth_b = 0.0f;
        /// What the reflection trace shades its hits with. Ignored by the
        /// shadow and local-shadow passes, which never shade anything.
        RtHitShadeBindings hit_shade{};

        // ---- R5b: lighting-change invalidation (§5.6) ---------------------

        /**
         * The ANALYTIC sun gradient, already reduced to a global hysteresis
         * floor in [0, 1] by `render::lighting_reset_floor`.
         *
         * The caller owns the previous frame's sun state, so it — not this
         * renderer — is where the delta is measured; the probe-field blend
         * takes the same number from the same place, which is what makes one
         * sun jump collapse every accumulation window in the frame at once
         * rather than each subsystem discovering it separately.
         *
         * 0 is the identity: every temporal pass reduces to its pre-R5b
         * arithmetic, bit for bit.
         */
        float lighting_reset = 0.0f;

        /**
         * `render::AutoExposure::state_buffer()` — a float4 {current EV, target
         * EV, avg log2 luma, valid}, bound READ-ONLY so the reflection
         * denoiser's absolute thresholds can be expressed in post-exposure
         * luminance. Null is valid and means "no auto exposure": the pass binds
         * a stand-in and reads `manual_exposure_ev` instead.
         */
        SDL_GPUBuffer* exposure_state = nullptr;
        /// log2 of `GraphicsSettings::exposure`, the manual bias, used whenever
        /// `exposure_state` is null or auto exposure is off.
        float manual_exposure_ev = 0.0f;
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
    /** The Full-DI tier's two denoised, demodulated local-light targets, at the
     *  LOCAL pass's traced resolution (R5c). Null unless the last render() ran
     *  with `full_di`. The scene shaders bind them as READ-ONLY STORAGE
     *  textures, never as samplers — they sit at SDL's 16-sampler ceiling —
     *  and reconstruct with `localDiSample`. See shaders/local_di.hlsli. */
    [[nodiscard]] gpu::GPUTexture* local_di_diffuse() const;
    [[nodiscard]] gpu::GPUTexture* local_di_specular() const;
    /** R7b's third target: the winning light's octahedral direction (RG16F), so
     *  the forward pass can re-apply N.L with its own normal-mapped normal.
     *  Live exactly when the other two are. */
    [[nodiscard]] gpu::GPUTexture* local_di_direction() const;
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
    /// Create (once) the 1x1 probe-field stand-ins bound when GI is off.
    [[nodiscard]] bool ensure_gi_dummies(SDL_GPUCommandBuffer* cmd);
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
        ///
        /// PING-PONGED since R5b: the A-SVGF gradient pass compares this
        /// frame's replayed trace against the PREVIOUS frame's raw bit, so the
        /// bits have to survive one extra frame. That is the whole memory cost
        /// of the gradient mechanism — 22 KB at half res — and it is why the
        /// bits were worth packing to one bit per pixel in the first place.
        std::array<std::unique_ptr<gpu::GPUTexture>, 2> bits;
        /// Per-tile verdict: 0 uniformly lit, 1 uniformly shadowed, 2 mixed.
        std::unique_ptr<gpu::GPUTexture> tile_class;
        /// Ping-ponged accumulation (RGBA16F): mean, second moment, sample
        /// count, linear depth. Two because a temporal filter cannot read and
        /// write the same texture in one dispatch.
        std::array<std::unique_ptr<gpu::GPUTexture>, 2> history;
        /// The R5b FAST history (RG16F: mean, count) — a four-frame mean of the
        /// same samples, far too noisy to display and never displayed. It is a
        /// change DETECTOR: the main mean is clamped toward it every frame by a
        /// margin derived from its own sampling error
        /// (`render::fast_history_margin`), which catches the changes the
        /// sparse gradient cannot see because no ray landed there.
        std::array<std::unique_ptr<gpu::GPUTexture>, 2> fast_history;
        /// The A-SVGF gradient at STRATUM resolution (traced / 3), R16F.
        /// [0] is the sparse trace's raw output, [1] the dilated reconstruction
        /// the temporal pass reads. Not a ping-pong: the two have different
        /// meanings and the pass between them is not iterated.
        std::array<std::unique_ptr<gpu::GPUTexture>, 2> gradient;
        /// Full-res R8 output — the denoiser's last pass fuses the upsample into
        /// the filter, so the forward shaders sample it 1:1.
        std::unique_ptr<gpu::GPUTexture> mask;
        int history_index = 0;
        /// Which half of `bits` THIS frame's trace wrote. The other half is the
        /// previous frame's, which is what the gradient compares against.
        int bits_index = 0;
        /// False until a history has been written at this size. Cleared on
        /// resize, on a resolution change, and whenever the signal was off last
        /// frame — in all three cases the history describes a different image.
        bool history_valid = false;
        /// False until a gradient has been reconstructed at this size. Tracked
        /// separately from `history_valid` because the gradient chain can be
        /// unavailable (a missing pipeline) on a frame whose history is fine,
        /// and the temporal pass must then read a bound-but-meaningless texture
        /// rather than a stale one.
        bool gradient_valid = false;
        RtResolution resolution = RtResolution::Half;
        /// Advances the sample sequence. Wraps freely; only its low bits matter
        /// to the R2 sequence, and wrapping is what keeps it exact in a float.
        /// PER SIGNAL, so the two never sample in lockstep.
        uint32_t frame_index = 0;
        /// First of this signal's four slots in `denoise_counters_`.
        uint32_t counter_base = 0;

        void release_targets();
        void release();
    };

    /// Records the three denoiser dispatches for one signal. Split out of
    /// render() because the trace is one dispatch and the denoiser is most of
    /// the pass.
    void denoise_visibility(SDL_GPUCommandBuffer* cmd, VisibilityDenoiser& d, gpu::GPUTexture& depth,
                            SDL_GPUTexture* velocity, const FrameParams& params, float max_accum_frames,
                            float filter_radius, int traced_w, int traced_h);
    /// Stratum grid for a traced resolution (ceil / GRADIENT_STRATUM).
    [[nodiscard]] static int gradient_stratum_count(int traced);
    /// Allocate/keep this signal's gradient pair and the fast-history pair.
    [[nodiscard]] bool ensure_gradient_targets(VisibilityDenoiser& d, int traced_w, int traced_h,
                                               const char* label);
    /// The sun's sparse gradient trace, into `d.gradient[0]`.
    [[nodiscard]] bool trace_sun_gradient(SDL_GPUCommandBuffer* cmd, VisibilityDenoiser& d,
                                          gpu::GPUTexture& depth, SDL_GPUTexture* normal_rough,
                                          SDL_GPUTexture* velocity, const FrameParams& params,
                                          const RtShadowSettings& shadows, int traced_w, int traced_h);
    /// The local signal's sparse gradient trace, into `d.gradient[0]`.
    [[nodiscard]] bool trace_local_gradient(SDL_GPUCommandBuffer* cmd, VisibilityDenoiser& d,
                                            gpu::GPUTexture& depth, SDL_GPUTexture* normal_rough,
                                            SDL_GPUTexture* velocity, const FrameParams& params,
                                            const RtLocalShadowSettings& settings,
                                            const RtClusterBindings& clusters, int prev_reservoir,
                                            int traced_w, int traced_h);
    /// gradient[0] -> gradient[1], the weighted-max soft dilation.
    [[nodiscard]] bool reconstruct_gradient(SDL_GPUCommandBuffer* cmd, VisibilityDenoiser& d,
                                            int traced_w, int traced_h);
    /// Create (once) the 8-uint diagnostics buffer both temporal dispatches
    /// accumulate into, and the 1x1 stand-in bound where an exposure state
    /// buffer is wanted and none exists.
    [[nodiscard]] bool ensure_denoise_diagnostics();
    /// Poll the lagged counter readback and, when one lands, difference it
    /// against the previous to fill `stats_.denoise`. Called once per render().
    void update_denoise_stats();
    /// Create (once) the 1x1 stand-ins the trace binds into its two DI slots in
    /// Ratio mode. Size-independent, so they outlive a resolution change.
    [[nodiscard]] bool ensure_di_dummies();
    /// Create (once) the one-element stand-in the trace binds into its emissive
    /// record slot when the scene has no emitter. A storage-buffer descriptor
    /// may not be empty; `emissiveCount = 0` is what stops it being indexed.
    [[nodiscard]] bool ensure_emissive_dummy();
    /// Same, for R8's per-froxel emissive candidate table. One zeroed froxel
    /// block; `emissiveFroxelShare = 0` is what stops it being indexed.
    [[nodiscard]] bool ensure_emissive_froxel_dummy();
    /// Create (once) the one-element stand-in bound into the capsule slot on a
    /// frame with no capsules. Same rule as the emissive dummy: a storage-buffer
    /// descriptor may not be empty, and `shadowCapsuleCount = 0` is what stops
    /// it being indexed.
    [[nodiscard]] bool ensure_capsule_dummy();
    /// The buffer the four any-hit passes bind, and the count they read — the
    /// real one when the frame had capsules, the stand-in when it did not.
    [[nodiscard]] SDL_GPUBuffer* capsule_binding();
    /// Allocate/keep the Full-DI tier's six RGBA16F targets at the local
    /// signal's traced resolution. Invalidates the DI history on any change.
    [[nodiscard]] bool ensure_di_targets(int traced_w, int traced_h);
    /// The DI denoiser chain: the temporal EMA, then one or two a-trous
    /// iterations of the accumulation-driven edge-aware gather, then R7's
    /// probe when the caller asked for it. No-op when the trace did not write
    /// the raw targets.
    void denoise_local_di(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                          SDL_GPUTexture* normal_rough, SDL_GPUTexture* velocity,
                          const FrameParams& params, const RtLocalShadowSettings& settings,
                          int traced_w, int traced_h);
    /// R7's temporal-noise probe over the filtered DI targets. Off unless
    /// `RtLocalShadowSettings::di_noise_probe`; costs a dispatch and an R16G16F
    /// target when on.
    void probe_di_noise(SDL_GPUCommandBuffer* cmd, int traced_w, int traced_h);
    /// Trace + denoise the froxel-RIS local-light signal. Returns whether the
    /// trace ran.
    [[nodiscard]] bool render_local_shadows(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                                            SDL_GPUTexture* normal_rough, SDL_GPUTexture* velocity,
                                            const FrameParams& params,
                                            const RtLocalShadowSettings& settings,
                                            const RtClusterBindings& clusters);
    /// Temporal accumulation of the GGX-sampled reflection signal. Opaque path
    /// only — see reflection_denoised_this_frame_.
    void denoise_reflections(SDL_GPUCommandBuffer* cmd, gpu::GPUTexture& depth,
                             SDL_GPUTexture* normal_rough, SDL_GPUTexture* velocity,
                             const FrameParams& params, const RtReflectionSettings& reflections);

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
    /// 1x1 stand-ins for the probe field, bound whenever `gi_mode` is Off so
    /// the reflection pass's slots are valid rather than dangling. Owned here
    /// rather than borrowed from SceneRenderer because the probe-state plane is
    /// a READ-ONLY STORAGE texture and the engine's shared dummies carry
    /// sampler usage only. The state texel is zero, which is
    /// GI_PROBE_UNINITIALIZED, so even a shader that ignored `gi_enabled` and
    /// read them would get no contribution rather than garbage.
    std::unique_ptr<gpu::GPUTexture> gi_dummy_irradiance_;
    std::unique_ptr<gpu::GPUTexture> gi_dummy_probe_state_;

    // ---- the two stochastic visibility signals ----
    VisibilityDenoiser sun_shadow_;
    VisibilityDenoiser local_shadow_;
    /// Ping-ponged per-pixel reservoirs at the LOCAL pass's traced resolution,
    /// 16 B each (render/lighting/reservoir.hpp). Not textures: the record is
    /// four heterogeneous lanes, one of which is a light INDEX that must not be
    /// filtered, converted or clamped by anything.
    std::array<std::unique_ptr<gpu::GPUBuffer>, 2> local_reservoirs_;
    /// R7a's third reservoir buffer: stage 0's output, which stage 1 reads as
    /// its spatial neighbourhood. NOT part of the ping-pong — it never crosses a
    /// frame boundary, and the ping-pong's two halves are still exactly "last
    /// frame's final" and "this frame's final", which is what the gradient
    /// replay and the temporal merge both depend on.
    std::unique_ptr<gpu::GPUBuffer> local_reservoir_temp_;

    /**
     * The Full-DI tier's targets (R5c), all RGBA16F at the LOCAL signal's
     * traced resolution.
     *
     * A RADIANCE denoiser, not a visibility one, and that is why it is a
     * separate block rather than two more fields on VisibilityDenoiser: there
     * is no bit packing, no tile classification and no second moment here — the
     * quantity is unbounded and continuous, so the reflection denoiser's
     * sigma-clamped EMA is the right shape and the visibility chain's is not.
     *
     * `raw` is what the trace wrote; `history` is the ping-ponged temporal
     * accumulation; `filtered` is what the forward shaders read. Three targets
     * per stage since R7b — diffuse, specular and the winning light's DIRECTION,
     * the last RG16F where the first two are RGBA16F.
     *
     * R7c'S SECOND A-TROUS ITERATION COSTS NO NEW MEMORY, and that is the reason
     * the triple is arranged this way: the first iteration writes into `raw`,
     * which the temporal dispatch has already consumed by then and which nothing
     * reads again until next frame's trace overwrites it, and the second reads
     * `raw` and writes `filtered`. With one iteration configured, it writes
     * `filtered` directly and `raw` is never re-entered.
     */
    struct RadianceDenoiser {
        /// [0] diffuse, [1] specular, [2] direction — the same order everywhere
        /// in this struct, and the order the shaders' bindings are in.
        std::array<std::unique_ptr<gpu::GPUTexture>, 3> raw;
        std::array<std::array<std::unique_ptr<gpu::GPUTexture>, 3>, 2> history; ///< [ping][signal]
        std::array<std::unique_ptr<gpu::GPUTexture>, 3> filtered;
        /// 1x1 stand-ins bound into the trace's three DI slots in Ratio mode. A
        /// storage-image descriptor may not be empty, and `diMode` is what
        /// stops anything being written through them. Typed to match: the
        /// direction slot's stand-in is RG16F, because a storage image read or
        /// written without a format must still agree on the channel count.
        std::array<std::unique_ptr<gpu::GPUTexture>, 3> dummy;
        int history_index = 0;
        bool history_valid = false;
        int width = 0;
        int height = 0;

        void release();
    };
    RadianceDenoiser local_di_;
    /// Whether the last render() produced live `local_di_.filtered` targets.
    bool local_di_ready_ = false;
    /// R7's noise probe: last frame's per-texel scalar (R16F, traced res), and
    /// whether it holds a measurement of the previous frame of THIS image.
    std::unique_ptr<gpu::GPUTexture> di_noise_prev_;
    bool di_noise_history_valid_ = false;
    int local_reservoir_index_ = 0;
    /// Element count the pair is currently sized for; a change reallocates and
    /// invalidates the history, because reservoir N describes a different pixel.
    int local_reservoir_capacity_ = 0;
    /// The emissive record epoch the PREVIOUS frame's reservoirs were written
    /// against (R6). Equal means an emissive index in the history still names
    /// the same triangle; different means the record set was renumbered by a
    /// TLAS topology rebuild and emissive history must be dropped.
    uint32_t local_emissive_epoch_ = 0;
    /// What the TRACE decided this frame, so the gradient replay one dispatch
    /// later reaches the identical verdict. Recomputing it there would work and
    /// would be a second place for the tier gate and the epoch test to disagree.
    uint32_t local_emissive_count_ = 0;
    uint32_t local_emissive_history_valid_ = 0;
    SDL_GPUBuffer* local_emissive_bound_ = nullptr;
    std::unique_ptr<gpu::GPUBuffer> local_emissive_dummy_;
    std::unique_ptr<gpu::GPUBuffer> local_emissive_froxel_dummy_;

    // ---- analytic capsule occluders (docs/rt_lighting_plan.md §5.8) ----
    /// The frame's ShadowCapsules, uploaded once in the renderer's copy pass
    /// and bound by all four any-hit passes. Grow-only: `capsule_capacity_` is
    /// the element count it is sized for, and a scene with a stable cast never
    /// reallocates. ComputeStorage rather than Storage because rt_local_shadow
    /// and its gradient bind it to a READ-WRITE slot — their eight read-only
    /// slots were spent before capsules existed — while the two sun passes bind
    /// the very same buffer read-only. One buffer, two binding kinds.
    std::unique_ptr<gpu::GPUBuffer> capsule_buffer_;
    std::size_t capsule_capacity_ = 0;
    /// Capsules the last upload actually wrote — the value every one of the
    /// four `shadowCapsuleCount` uniforms is set from, so the trace and its
    /// replay cannot disagree about the occluder set.
    uint32_t capsule_count_ = 0;
    std::unique_ptr<gpu::GPUBuffer> capsule_dummy_;

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

    // ---- R5b diagnostics ----
    /**
     * Twelve uints. Four per visibility signal: {accumulated-count sum, pixels
     * counted, gradient-fired pixels, in-bounds pixels}. Then three for R7's DI
     * noise probe at base 8: {fixed-point |delta| sum, fixed-point level sum,
     * texels counted}, and one spare.
     *
     * NEVER RESET, and that is the design rather than an omission. Zeroing them
     * would cost a copy pass in every frame to serve a statistic nobody reads
     * most frames. Instead both temporal dispatches accumulate monotonically
     * and the CPU DIFFERENCES two lagged readbacks — which is exact under
     * unsigned wraparound, since the difference of two uint32s is correct
     * whenever fewer than 2^32 were added between them (here: ~360 k per frame,
     * so ~12000 frames of headroom against a readback interval of well under a
     * second).
     */
    std::unique_ptr<gpu::GPUBuffer> denoise_counters_;
    /// 1x1 float4 {0,0,0,0} bound where a pass wants the exposure state and the
    /// caller has none. The `w` lane is the adapt pass's validity flag, so a
    /// shader that ignored `exposureEnabled` would read "not adapted yet" and
    /// fall back rather than read undefined memory.
    std::unique_ptr<gpu::GPUBuffer> exposure_dummy_;

    SDL_GPUTransferBuffer* counter_transfer_ = nullptr;
    SDL_GPUFence* counter_fence_ = nullptr;
    /// Frames between readback REQUESTS. Ten rather than one: the download is
    /// 32 bytes but it rides its own command-buffer submit, and the statistic
    /// it serves is an average over a window anyway. Ten also gives a
    /// 90-frame --rt-bench sample window nine independent measurements, which
    /// is what the harness averages.
    static constexpr int COUNTER_READBACK_FRAMES = 10;
    int frames_since_counter_readback_ = COUNTER_READBACK_FRAMES;
    std::array<uint32_t, 12> counter_previous_{};
    bool counter_previous_valid_ = false;

    int screen_width_ = 0;
    int screen_height_ = 0;
    RtResolution reflection_res_ = RtResolution::Quarter;
    bool built_with_attributes_ = false;
    bool warned_missing_attributes_ = false;
    RtStats stats_{};
};

} // namespace bracken::render
