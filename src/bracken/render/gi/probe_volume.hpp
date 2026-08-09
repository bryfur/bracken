#pragma once

#include "bracken/contracts/environment.hpp"
#include "bracken/contracts/graphics_settings.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/render/gi/gi_math.hpp"
#include "bracken/render/gi/probe_schedule.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace bracken::gpu {
class PipelineRegistry;
} // namespace bracken::gpu

namespace bracken::render::rt {
class RtScene;
} // namespace bracken::render::rt

namespace bracken::render::gi {

class VoxelGi;

namespace gpu = bracken::gpu;

/** Per-cascade accounting: what the structure costs, and what it traced. */
struct GiCascadeStats {
    float spacing = 0.0f;              ///< world units between adjacent probes
    glm::ivec3 base{0};                ///< absolute coord of the (0,0,0) corner probe
    std::uint32_t probes = 0;          ///< always PROBES_PER_CASCADE once allocated
    std::uint32_t probes_invalidated = 0; ///< entered the window THIS frame
    std::uint32_t probes_updated = 0;  ///< traced THIS frame (0 on a frame the rota skipped)
    std::size_t atlas_bytes = 0;       ///< irradiance + visibility + state + offsets
};

/**
 * What the probe field costs and what it did this frame.
 *
 * `probes_invalidated` is the number that matters while nothing consumes the
 * field: it is the only observable proof the scroll is working, and it is what
 * R1b's ray budget will be spent on. A stationary camera must report 0 after
 * the first frame; a moving one must report a plane's worth, not a field's.
 */
struct GiStats {
    bool active = false;
    std::uint32_t cascades = 0;
    std::uint32_t probes = 0;
    std::uint32_t probes_invalidated = 0;
    /// Running total since init(). The per-frame figure is zero on most frames
    /// (a cascade only scrolls when the camera crosses a whole probe spacing),
    /// so a run's evidence that the scroll happened at all has to be cumulative
    /// — sampling `probes_invalidated` at an arbitrary frame proves nothing.
    std::uint64_t probes_invalidated_total = 0;
    std::size_t atlas_bytes = 0;

    // ---- what R1b's update passes did ----
    /// Rays dispatched this frame, summed over the cascades whose turn it was.
    /// Zero on no frame once GI is on: cascade 0 traces every frame.
    std::uint32_t rays_this_frame = 0;
    /// Probes traced this frame. The rota (probe_schedule.hpp) puts at most two
    /// cascades in any one frame, so this is at most 2x a cascade's budget.
    std::uint32_t probes_updated = 0;
    /// Of those, how many the classify pass nudged, and how many it wrote off
    /// as sitting inside geometry.
    ///
    /// LAGGED, and honestly so: both are GPU-computed and reach the CPU through
    /// the same fenced, polled, never-blocking download `render::AutoExposure`
    /// uses for its exposure state. They describe a recent frame's scheduled
    /// set, not necessarily this one's, and they are diagnostics — nothing in
    /// the frame waits on them.
    std::uint32_t probes_relocated = 0;
    std::uint32_t probes_inside_geometry = 0;
    std::uint64_t rays_total = 0;
    std::uint64_t probes_updated_total = 0;

    std::array<GiCascadeStats, CASCADE_COUNT> per_cascade{};
};

/**
 * Everything the update passes need from the rest of the frame.
 *
 * All of it is already built by the time the GI block runs: the CSM atlas was
 * rendered, the cluster light buffer uploaded, the IBL chain refreshed, and the
 * TLAS brought up to date — see `SceneRenderer::render_frame`'s ordering, which
 * is why the GI dispatches sit where they do and not earlier.
 */
struct ProbeUpdateParams {
    gpu::PipelineRegistry* pipelines = nullptr;
    /// The acceleration structure. The RAY-TRACED source requires it, with
    /// attributes, because probe rays shade what they hit. Null (or not ready)
    /// skips the whole update — UNLESS `voxel` below is set, in which case there
    /// is no ray tracing in this path at all and no structure is needed.
    const rt::RtScene* scene = nullptr;
    /**
     * The RASTER ray source (GiMode::VoxelProbes). When non-null the trace step
     * cone-traces a voxel clipmap instead of walking the BVH, writing the same
     * `float4(radiance, distance)` contract into the same buffer — so every
     * other step of the update is byte-for-byte the path it always was.
     *
     * Mutually exclusive with `scene` in practice, and deliberately not enforced
     * here: `trace` picks the voxel source whenever it is present, so a caller
     * that supplies both gets the raster answer rather than an assertion.
     */
    VoxelGi* voxel = nullptr;

    GiQuality quality = GiQuality::Medium;
    glm::vec3 camera_position{0.0f};
    /// Unit view direction — `calcShadow` picks a cascade by view depth.
    glm::vec3 camera_forward{0.0f, 0.0f, -1.0f};
    EnvironmentState environment{};
    /// GraphicsSettings::sun_intensity. It lives outside EnvironmentState and
    /// every consumer multiplies it in at its own use site, so the probe pass
    /// has to be told rather than being able to read it off `environment` —
    /// getting this wrong makes bounce light disagree with direct light by a
    /// factor of three in the playground.
    float sun_intensity = 1.0f;

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

    /// The clustered-lighting byte buffer and its counts. Probe rays read the
    /// flat list, not the froxel grid (a world-space hit has no froxel).
    SDL_GPUBuffer* light_data = nullptr;
    std::uint32_t light_count = 0;
    std::uint32_t light_capacity = 0;

    /// How far a probe ray may see, as a CEILING. Longer catches distant sky
    /// occlusion and costs traversal; the default is the reflection trace's
    /// range. Each cascade then takes the smaller of this and
    /// `PROBE_RAY_REACH_PROBES` of its own spacing (gi_math.hpp), so the finest
    /// cascade does not pay for reach its own window never uses — only the
    /// coarsest one actually sees this number.
    float ray_max_distance = 4000.0f;

    /**
     * R5b's analytic sun gradient, already reduced to a hysteresis floor in
     * [0, 1] by `render::lighting_reset_floor`.
     *
     * The blend's own change-adaptive hysteresis can only react to a delta a
     * probe has already OBSERVED, and a probe on a coarse cascade traces once
     * every four frames — so a sun jump reaches the field several frames after
     * it reaches the shadow denoisers, and reaches it as a large delta that
     * then takes tens of updates to work through a 0.97 EMA. This floor is what
     * lets the field collapse its window on the frame the sun moved, from the
     * same number the visibility denoisers get.
     *
     * 0 is the identity — a static sun leaves the field bit-identical.
     */
    float lighting_reset = 0.0f;
};

/**
 * The scrolled probe clipmap: three camera-centred cascades of octahedral
 * irradiance + visibility probes, and the CPU addressing that says which texel
 * belongs to which point in the world.
 *
 * FOUR 2D TEXTURES FOR THE WHOLE FIELD, each holding every cascade as a band of
 * rows (`gi_math.hpp`, ATLAS_ROWS_ALL_CASCADES) over the tiling defined there:
 *   - irradiance, RGBA16F, 8x8 texels/probe (6x6 payload + 1 border ring)
 *     -> 2304 x 576, 10.125 MB
 *   - visibility, RG16F, 16x16 texels/probe (14x14 payload + border), storing
 *     mean and mean-squared ray distance for the Chebyshev leak test
 *     -> 4608 x 1152, 20.25 MB
 *   - probe offset, RGBA16F, one texel/probe -> 288 x 72, 162 KB
 *   - probe state, R8_UINT, one texel/probe -> 288 x 72, 20.25 KB
 * i.e. ~30.6 MB for the field, unchanged by the banding (the same texels, in
 * three textures instead of nine). `docs/rt_lighting_plan.md` §6 budgets ~35 MB.
 *
 * Banding is a CONSUMPTION decision: the forward shaders sample the first three
 * of those, and per-cascade textures would have cost them nine sampler slots
 * where they have three to spare. See gi_math.hpp for the full argument.
 *
 * Tier: main-thread (owns GPU resources, records copy passes).
 */
class ProbeVolume {
public:
    ProbeVolume() = default;
    ~ProbeVolume();
    ProbeVolume(const ProbeVolume&) = delete;
    ProbeVolume& operator=(const ProbeVolume&) = delete;
    ProbeVolume(ProbeVolume&&) = delete;
    ProbeVolume& operator=(ProbeVolume&&) = delete;

    /// Allocate every cascade's atlases. Fails (and allocates nothing) if any
    /// texture cannot be created — a half-built field would be worse than none.
    [[nodiscard]] bool init(gpu::GPUDevice& device);
    void shutdown();
    [[nodiscard]] bool is_ready() const { return ready_; }

    /**
     * Follow the camera: re-snap every cascade and mark entering probes.
     *
     * `cmd` may be null (a host or test driving the CPU side alone), in which
     * case the state textures are left stale and re-uploaded on the next call
     * that has a command buffer. NO RENDER PASS may be active — the state
     * upload opens its own copy pass.
     */
    void update(SDL_GPUCommandBuffer* cmd, const glm::vec3& camera_position);

    /**
     * Trace, blend and classify this frame's share of the field.
     *
     * Call AFTER update() and outside any render pass; each cascade whose turn
     * the rota gives it records four compute dispatches — trace, irradiance
     * blend, visibility blend, classify — in that order. A cascade the rota
     * skips costs nothing at all.
     *
     * Silently does nothing (and reports zero rays) when the acceleration
     * structure is not ready or a pipeline is missing: GI is a feature that can
     * be switched on before its prerequisites exist, and a half-traced field is
     * worse than an untraced one.
     */
    void trace(SDL_GPUCommandBuffer* cmd, const ProbeUpdateParams& params);

    [[nodiscard]] const GiStats& stats() const noexcept { return stats_; }

    /**
     * The three atlases the forward pass and the fog march sample, and the
     * fourth the update passes need. Each holds EVERY cascade, banded — see
     * `gi_math.hpp`'s ATLAS_ROWS_ALL_CASCADES. Null before init().
     *
     * `irradiance` and `visibility` want a LINEAR clamp sampler (the border
     * ring exists so a bilinear tap at a payload edge wraps the octahedron);
     * `probe_offset` and `probe_state` are one texel per probe and want a
     * POINT sampler or a Load.
     */
    [[nodiscard]] gpu::GPUTexture* irradiance() const { return irradiance_.get(); }
    [[nodiscard]] gpu::GPUTexture* visibility() const { return visibility_.get(); }
    [[nodiscard]] gpu::GPUTexture* probe_state() const { return state_.get(); }
    /**
     * Per-probe relocation offset (xyz, world units) + classification verdict
     * (w, one of the PROBE_CLASS_* values). RGBA16F, one texel per probe.
     *
     * R1c CONTRACT, both halves. A probe's irradiance describes the world at
     * `lattice position + offset.xyz`, not at its lattice position: a sampler
     * that ignores this reads a probe that was nudged out of a wall as if it
     * were still in it, which is precisely the leak the nudge existed to
     * prevent. And `w == PROBE_CLASS_INSIDE_GEOMETRY` is the ONLY "do not use
     * this probe" signal a consumer gets — it covers both "buried" and "never
     * filled", because init() seeds the whole plane with it. See gi_math.hpp.
     */
    [[nodiscard]] gpu::GPUTexture* probe_offset() const { return offset_.get(); }

    /// The bilinear clamp sampler the atlases want. Null before init().
    [[nodiscard]] SDL_GPUSampler* linear_sampler() const { return linear_clamp_sampler_; }
    /// The point clamp sampler the per-probe planes want. Null before init().
    [[nodiscard]] SDL_GPUSampler* point_sampler() const { return point_clamp_sampler_; }

    /// Absolute lattice coordinate of each cascade's (0,0,0) corner probe —
    /// what a consumer needs to turn a world position into probe space.
    /// Unplaced cascades report their last base (zero before the first update).
    [[nodiscard]] std::array<glm::ivec3, CASCADE_COUNT> cascade_bases() const;

    /// CPU-side scroll state, for tests and the dev overlay.
    [[nodiscard]] const ProbeCascade& cascade(int index) const;

private:
    struct Cascade {
        ProbeCascade cpu;
    };

    /// Record one cascade's four dispatches. Returns probes traced.
    std::uint32_t trace_cascade(SDL_GPUCommandBuffer* cmd, const ProbeUpdateParams& params,
                                int cascade_index, std::uint32_t probe_budget,
                                std::uint32_t rays_per_probe);
    void poll_counter_readback();
    void request_counter_readback();

    gpu::GPUDevice* device_ = nullptr;
    std::array<Cascade, CASCADE_COUNT> cascades_;
    /// ONE texture each, all cascades banded (gi_math.hpp).
    std::unique_ptr<gpu::GPUTexture> irradiance_;
    std::unique_ptr<gpu::GPUTexture> visibility_;
    std::unique_ptr<gpu::GPUTexture> state_;
    std::unique_ptr<gpu::GPUTexture> offset_;
    /// CPU mirror of the whole banded state plane. The scroll re-uploads all of
    /// it whenever ANY cascade's window moved: it is 20.25 KB, which is the
    /// worst case R1a already budgeted for (three cascades scrolling at once),
    /// and one whole-texture upload is what `GPUTexture::upload` offers.
    std::vector<std::uint8_t> state_mirror_;
    bool ready_ = false;
    GiStats stats_{};

    /// Advances the ray-set rotation and drives the cascade rota. Persistent,
    /// deterministic, and the only clock either depends on.
    std::uint64_t frame_counter_ = 0;

    /// Bilinear clamp, for the previous-frame irradiance the bounce samples and
    /// for every consumer of the octahedral payloads.
    SDL_GPUSampler* linear_clamp_sampler_ = nullptr;
    /// Nearest clamp, for the one-texel-per-probe planes. A linear tap on those
    /// would blend two unrelated probes' offsets.
    SDL_GPUSampler* point_clamp_sampler_ = nullptr;

    /// Per-ray radiance + signed distance, sized for the largest single-cascade
    /// turn at any quality so a quality change never reallocates. Reused by
    /// every cascade in the frame — GPU passes serialize on one queue, so the
    /// second cascade's trace cannot start before the first's classify ends.
    std::unique_ptr<gpu::GPUBuffer> ray_results_;
    /// This turn's probe list — ONE PER CASCADE, not one shared. Two cascades
    /// can trace in the same frame, and a dynamic buffer's transfer memory is
    /// written by the CPU immediately while its copy is only recorded: writing
    /// the second cascade's list into the same staging bytes would race the
    /// first cascade's copy, which has not executed yet. 55 KB each.
    std::array<std::unique_ptr<gpu::GPUBuffer>, CASCADE_COUNT> update_lists_;
    /// Classification counters, four uints per cascade, zeroed every frame.
    std::unique_ptr<gpu::GPUBuffer> counters_;

    /// Fenced, polled, never-blocking counter download — the same shape as
    /// render::AutoExposure's debug readback, for the same reason: these are
    /// diagnostics and no part of the frame may wait on them.
    SDL_GPUTransferBuffer* readback_transfer_ = nullptr;
    SDL_GPUFence* readback_fence_ = nullptr;

    /// Scratch, reused per dispatch so the per-frame path allocates nothing.
    std::vector<ProbeUpdateSlot> scratch_slots_;
    std::vector<gpu::GiProbeUpdateRecord> scratch_records_;
};

} // namespace bracken::render::gi
