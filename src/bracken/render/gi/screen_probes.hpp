#pragma once

#include "bracken/contracts/environment.hpp"
#include "bracken/gpu/gpu_buffer.hpp"
#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/render/gi/gi_math.hpp"
#include "bracken/render/gi/screen_probe_math.hpp"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>

namespace bracken::gpu {
class PipelineRegistry;
} // namespace bracken::gpu

namespace bracken::render::rt {
class RtScene;
} // namespace bracken::render::rt

namespace bracken::render::gi {

namespace gpu = bracken::gpu;

/** What the screen-probe gather cost and what it resolved, this frame. */
struct GiScreenProbeStats {
    bool active = false;
    glm::ivec2 grid{0};
    std::uint32_t probes = 0;
    std::size_t bytes = 0;

    /// Rays resolved against the depth buffer, rays handed to the BVH, and of
    /// those, how many hit geometry versus fell through to the sky/field
    /// blend. LAGGED — same fenced, polled, never-blocking download
    /// `ProbeVolume` uses for its relocation counters, and for the same reason:
    /// these are diagnostics and no part of the frame waits on them.
    std::uint32_t screen_hits = 0;
    std::uint32_t world_rays = 0;
    std::uint32_t world_hits = 0;
    std::uint32_t fallbacks = 0;
    /// Miss records the capacity clamp dropped. Should be zero always; see
    /// `screen_probe_miss_capacity`.
    std::uint32_t dropped = 0;
};

/** Everything the six passes need from the rest of the frame. */
struct ScreenProbeParams {
    gpu::PipelineRegistry* pipelines = nullptr;
    /// The acceleration structure, WITH attributes: the handed-off rays shade
    /// what they hit. Null or unready skips the whole block — a half-gathered
    /// probe field is worse than none.
    const rt::RtScene* scene = nullptr;

    /// Hi-Z pyramid over THIS frame's prepass depth (render/hzb.hpp). Mip 0 is
    /// the depth itself, which is why no separate depth binding exists.
    SDL_GPUTexture* hzb = nullptr;
    int hzb_mip_count = 0;
    /// Prepass G-buffer (octahedral world normal + roughness).
    SDL_GPUTexture* gbuffer_normal = nullptr;
    /// Previous frame's resolved linear-HDR scene colour (render::TAA history),
    /// or null when TAA is off / the history is invalid — in which case every
    /// ray is handed to the BVH rather than reading a texture that means
    /// nothing.
    SDL_GPUTexture* scene_history = nullptr;
    /// This frame's UV-space motion vectors, for the temporal filter's anchor
    /// reprojection.
    SDL_GPUTexture* velocity = nullptr;

    int screen_width = 0;
    int screen_height = 0;

    /// The DRAW (jittered) camera's matrices — they must be the ones the depth
    /// buffer was rendered with.
    glm::mat4 view_projection{1.0f};
    glm::mat4 inv_view_projection{1.0f};
    /// UNJITTERED previous frame, the pair the velocity buffer and the TAA
    /// resolve were built from.
    glm::mat4 prev_view_projection_unjittered{1.0f};
    glm::vec3 camera_position{0.0f};
    glm::vec3 camera_forward{0.0f, 0.0f, -1.0f};
    /// projection[2][2] / projection[3][2] of the draw projection.
    float depth_a = 0.0f;
    float depth_b = 0.0f;

    EnvironmentState environment{};
    float sun_intensity = 1.0f;

    /// This frame's CSM uniforms and atlas, exactly as the forward pass got
    /// them: the BVH pass calls the same `calcShadow()`.
    gpu::ShadowDataUniforms shadow{};
    SDL_GPUTexture* shadow_atlas = nullptr;
    SDL_GPUSampler* shadow_compare_sampler = nullptr;
    SDL_GPUSampler* shadow_raw_sampler = nullptr;

    SDL_GPUTexture* sky_cube = nullptr;
    SDL_GPUTexture* sky_cube_fallback = nullptr;
    SDL_GPUSampler* sky_sampler = nullptr;

    SDL_GPUBuffer* light_data = nullptr;
    std::uint32_t light_count = 0;
    std::uint32_t light_capacity = 0;

    /// The world probe field, for the handed-off rays' hit shading and for the
    /// escaped-ray fallback. Null disables both (the fallback becomes pure sky).
    SDL_GPUTexture* gi_irradiance = nullptr;
    SDL_GPUSampler* gi_irradiance_sampler = nullptr;
    SDL_GPUTexture* gi_probe_state = nullptr;
    std::array<glm::ivec4, CASCADE_COUNT> gi_cascade_base{};
    float gi_enabled = 0.0f;

    /// How far a screen ray may travel before it is handed off, world units.
    /// The screen march's range, NOT the BVH ray's — that one is pinned to
    /// `SCREEN_PROBE_WORLD_RAY_CLAMP`.
    float screen_ray_max_distance = 4000.0f;
};

/**
 * The screen-probe gather: one 8x8 octahedral probe per 16x16 screen tile,
 * traced screen-first and BVH-second, filtered in probe space, and published as
 * an L1 spherical-harmonic atlas the scene shaders sample.
 *
 * `GiMode::ProbesScreen` only. This is the QUALITY TIER above the world probe
 * field (render/gi/probe_volume.hpp), not a replacement for it: the field is
 * still traced, still lights the fog, and is still what every pixel this
 * gather cannot answer for falls back to.
 *
 * SIX PASSES, in this order, all inside one `GIScreenProbes` timer scope:
 *
 *   1. place    — anchor each tile's probe on the depth buffer; zero the
 *                 frame's counters and the miss-buffer header.
 *   2. march    — one thread per probe texel; walk the Hi-Z pyramid; sample the
 *                 TAA history where it hits; COMPACT everything else into the
 *                 miss buffer with one atomic per threadgroup.
 *   3. args     — one thread; turn the miss count into an
 *                 SDL_GPUIndirectDispatchCommand.
 *   4. trace    — INDIRECT dispatch over the misses; short BVH rays shaded by
 *                 rt_hit_shade.hlsli; sky/field blend beyond the clamp.
 *   5. filter   — probe-space temporal: firefly clamp + EMA against the
 *                 reprojected previous probe (ping-ponged atlases).
 *   6. project  — 3x3 probe-space spatial filter fused with the L1 projection,
 *                 into the SH atlas the forward pass reads.
 *
 * Each is its OWN compute pass because SDL3 GPU has no intra-pass
 * synchronization and every one of them reads the previous one's output — and
 * pass 4 additionally requires it, since indirect arguments are not
 * synchronized against the pass that wrote them (gpu/gpu_compute.hpp).
 *
 * Tier: main-thread (owns GPU resources, records passes).
 */
class ScreenProbeGather {
public:
    ScreenProbeGather() = default;
    ~ScreenProbeGather();
    ScreenProbeGather(const ScreenProbeGather&) = delete;
    ScreenProbeGather& operator=(const ScreenProbeGather&) = delete;
    ScreenProbeGather(ScreenProbeGather&&) = delete;
    ScreenProbeGather& operator=(ScreenProbeGather&&) = delete;

    /// Allocate for a `width` x `height` framebuffer. Fails (and allocates
    /// nothing) if any resource cannot be created.
    [[nodiscard]] bool init(gpu::GPUDevice& device, int width, int height);
    void shutdown();
    [[nodiscard]] bool is_ready() const { return ready_; }

    /// Reallocate at a new framebuffer size; no-op at the same size. Drops the
    /// temporal history — every atlas is indexed by probe, and a resize moves
    /// every probe.
    void resize(int width, int height);

    /** Drop the accumulated probe history: the next frame's EMA takes its raw
     *  samples whole. Called on resize and whenever the gather did not run.
     *
     *  Also clears `stats().active`, which is what the renderer's binding
     *  snapshot tests: an atlas that was not rewritten this frame must not be
     *  presented to the forward pass as if it had been. */
    void invalidate_history() {
        history_valid_ = false;
        stats_.active = false;
    }

    /**
     * Record the six passes. No render/copy pass may be open on `cmd`.
     *
     * Silently does nothing (reporting `active = false`) when a prerequisite is
     * missing — no pyramid, no G-buffer, no acceleration structure, a missing
     * pipeline. Same policy as `ProbeVolume::trace`: this is a feature that can
     * be switched on before its prerequisites exist, and a half-filled atlas
     * would be worse than an empty one.
     */
    void render(SDL_GPUCommandBuffer* cmd, const ScreenProbeParams& params);

    [[nodiscard]] const GiScreenProbeStats& stats() const noexcept { return stats_; }

    /// The published SH, one `gpu::GiScreenProbeSh` per probe, row-major. Bound
    /// by the scene shaders as their FIFTH fragment storage buffer — SDL3 GPU
    /// caps a stage at 16 samplers and they are already there. Null before
    /// init().
    [[nodiscard]] SDL_GPUBuffer* sh_buffer() const { return sh_ ? sh_->handle() : nullptr; }

    [[nodiscard]] glm::ivec2 grid() const { return grid_; }

private:
    void create_resources();
    void poll_counter_readback();
    void request_counter_readback();

    gpu::GPUDevice* device_ = nullptr;
    bool ready_ = false;
    int width_ = 0;
    int height_ = 0;
    glm::ivec2 grid_{0};

    /// Stage A/B write here; the filter reads it. One texture, not ping-ponged
    /// — it is fully rewritten every frame.
    std::unique_ptr<gpu::GPUTexture> raw_;
    /// The temporally filtered atlas, PING-PONGED: `filtered_[write_]` receives
    /// this frame while the other holds last frame's.
    std::array<std::unique_ptr<gpu::GPUTexture>, 2> filtered_;
    int write_ = 0;
    bool history_valid_ = false;
    /// The published L1 SH, one 64-byte record per probe.
    std::unique_ptr<gpu::GPUBuffer> sh_;

    /// Probe headers, PING-PONGED for the same reason the atlases are: the
    /// temporal filter has to ask whether the PREVIOUS frame's probe at the
    /// reprojected tile described a compatible surface, and this frame's
    /// headers have already overwritten that answer.
    std::array<std::unique_ptr<gpu::GPUBuffer>, 2> probes_;
    /// Compacted miss records; word 0 of the buffer is the count.
    std::unique_ptr<gpu::GPUBuffer> miss_;
    /// One SDL_GPUIndirectDispatchCommand, written by the args pass.
    std::unique_ptr<gpu::GPUBuffer> args_;
    /// Diagnostic counters (screen hits / world hits / fallbacks / dropped).
    std::unique_ptr<gpu::GPUBuffer> counters_;

    SDL_GPUSampler* point_clamp_sampler_ = nullptr;
    SDL_GPUSampler* linear_clamp_sampler_ = nullptr;

    SDL_GPUTransferBuffer* readback_transfer_ = nullptr;
    SDL_GPUFence* readback_fence_ = nullptr;

    std::uint64_t frame_counter_ = 0;
    GiScreenProbeStats stats_{};
};

} // namespace bracken::render::gi
