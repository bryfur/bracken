#pragma once

#include "bracken/gpu/gpu_device.hpp"
#include "bracken/gpu/gpu_texture.hpp"
#include "bracken/gpu/gpu_uniforms.hpp"
#include "bracken/gpu/pipeline_registry.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>

namespace bracken::render {

namespace gpu = bracken::gpu;

namespace detail {

constexpr float GTAO_PI = 3.14159265358979323846f;
constexpr float GTAO_HALF_PI = GTAO_PI * 0.5f;

/**
 * Sectors per slice hemicircle — the width of the visibility bitmask.
 *
 * 32 because the mask is a `uint32_t`/`uint` and `countbits` is plain ALU on
 * every backend. Each sector spans π/32 ≈ 5.6°.
 */
inline constexpr int GTAO_SECTOR_COUNT = 32;

/// Angular width of one sector, in radians.
inline constexpr float GTAO_SECTOR_DELTA = GTAO_PI / static_cast<float>(GTAO_SECTOR_COUNT);

/**
 * CPU reference for the GTAO cosine-weighted horizon arc integration — the
 * closed form of Jimenez et al. 2016 eq. 9 / XeGTAO.
 *
 * Since R5a the shader no longer evaluates this directly: it integrates the
 * SAME weight over 32 discrete sectors instead (see `gtao_sector_weight`).
 * It stays because it is the analytic ground truth the sector weights must
 * sum to — `GtaoSectorWeights.SumMatchesArcIntegral` pins exactly that — and
 * because the tests below document what the arc integral means.
 *
 * `horizon_cos0/1` are the horizon cosines relative to the view vector for
 * the negative/positive slice direction; `n` is the signed angle of the
 * projected normal from the view vector; returns visibility in [0, 1]
 * BEFORE the projected-normal-length weighting the shader applies per slice.
 * Unoccluded horizons (cos == cos(n ± π/2)) yield 1; horizons raised to the
 * view vector (cos == 1) yield 0.
 */
[[nodiscard]] inline float gtao_slice_visibility(float horizon_cos0, float horizon_cos1, float n) {
    const float cos_n = std::cos(n);
    float h0 = -std::acos(std::clamp(horizon_cos0, -1.0f, 1.0f));
    float h1 = std::acos(std::clamp(horizon_cos1, -1.0f, 1.0f));
    h0 = n + std::clamp(h0 - n, -GTAO_HALF_PI, GTAO_HALF_PI);
    h1 = n + std::clamp(h1 - n, -GTAO_HALF_PI, GTAO_HALF_PI);
    const float sin_n = std::sin(n);
    const float iarc0 = (cos_n + 2.0f * h0 * sin_n - std::cos(2.0f * h0 - n)) * 0.25f;
    const float iarc1 = (cos_n + 2.0f * h1 * sin_n - std::cos(2.0f * h1 - n)) * 0.25f;
    return iarc0 + iarc1;
}

// ---------------------------------------------------------------------------
// SSILVB — the visibility bitmask (Therrien et al. 2023, "Screen Space
// Indirect Lighting with Visibility Bitmask", The Visual Computer 39).
// ---------------------------------------------------------------------------
//
// The slice's hemicircle — the angular range [n - π/2, n + π/2] around the
// projected normal, with angles measured FROM THE VIEW VECTOR and signed
// positive toward the slice's +ω march direction — is cut into
// GTAO_SECTOR_COUNT equal sectors. Every depth sample sets the bits its
// occluder covers; visibility is what is left, weighted by the same cosine
// integrand the arc integral uses.

/// Signed angle (from the view vector) of sector boundary `k` ∈ [0, 32].
[[nodiscard]] inline float gtao_sector_angle(int k, float n) {
    return n - GTAO_HALF_PI + static_cast<float>(k) * GTAO_SECTOR_DELTA;
}

/**
 * A(θ) = ∫₀^θ sin φ · cos(φ − n) dφ, the antiderivative the GTAO arc integral
 * is built from (differentiate `gtao_slice_visibility`'s iarc term and apply
 * sin(2h−n) + sin n = 2 sin h cos(h−n) to see it).
 *
 * `sin φ` is the slice's arc-length Jacobian and `cos(φ − n)` the cosine lobe
 * about the projected normal. A ≥ 0 over the whole hemicircle, with A(0) = 0.
 */
[[nodiscard]] inline float gtao_arc_antiderivative(float theta, float n) {
    return 0.25f * (-std::cos(2.0f * theta - n) + std::cos(n) + 2.0f * theta * std::sin(n));
}

/**
 * Cumulative cosine-weighted visibility up to θ, i.e. ∫ |sin φ| cos(φ − n) dφ
 * measured from the hemicircle's start. `sign(θ)` is what turns A — which
 * integrates the SIGNED `sin φ` and therefore runs backwards on the negative
 * side — into a monotonically increasing cumulative.
 *
 * Only differences of this are used, so the additive constant is dropped.
 */
[[nodiscard]] inline float gtao_arc_cumulative(float theta, float n) {
    const float a = gtao_arc_antiderivative(theta, n);
    return theta < 0.0f ? -a : a;
}

/**
 * Exact cosine-weighted weight of sector `k` ∈ [0, 32). Strictly positive, and
 * the 32 of them sum to `gtao_slice_visibility` with unoccluded horizons —
 * which is what makes the bitmask pass an upgrade of the SAME integral rather
 * than a differently-normalised one.
 */
[[nodiscard]] inline float gtao_sector_weight(int k, float n) {
    return gtao_arc_cumulative(gtao_sector_angle(k + 1, n), n) -
           gtao_arc_cumulative(gtao_sector_angle(k, n), n);
}

/**
 * Bits covering the normalised angular span [lo, hi], both in [0, 1] over the
 * hemicircle (0 = the n − π/2 edge, 1 = the n + π/2 edge).
 *
 * Rounds OUTWARD on the width (`ceil`) and inward on the start (`floor`): a
 * sub-sector occluder still costs a whole sector rather than vanishing, which
 * is the conservative direction for an occlusion estimate. A zero-width span
 * marks nothing — that is the path a sample outside the sampling radius, a
 * background sample below the tangent plane, and a coplanar neighbour all take.
 */
[[nodiscard]] inline uint32_t gtao_mark_sectors(float lo, float hi) {
    const float span = (hi - lo) * static_cast<float>(GTAO_SECTOR_COUNT);
    if (!(span > 0.0f)) {
        return 0u;
    }
    const auto count = static_cast<uint32_t>(
        std::min(static_cast<float>(GTAO_SECTOR_COUNT), std::ceil(span)));
    const auto start = std::min(static_cast<uint32_t>(lo * static_cast<float>(GTAO_SECTOR_COUNT)),
                                static_cast<uint32_t>(GTAO_SECTOR_COUNT - 1));
    // 0xFFFFFFFF >> (32 - count) rather than (1 << count) - 1: count == 32 is
    // reachable and a 32-bit shift is undefined in both C++ and HLSL.
    const uint32_t bits = 0xFFFFFFFFu >> (static_cast<uint32_t>(GTAO_SECTOR_COUNT) - count);
    return bits << start;
}

/// Normalised sector-space span of one occluder, as produced by `gtao_slab_span`.
struct GtaoSlabSpan {
    float lo = 0.0f;
    float hi = 0.0f;
};

/**
 * The angular span a CONSTANT-THICKNESS SLAB occluder covers, normalised onto
 * the hemicircle.
 *
 * `front_angle` is the unsigned angle from the view vector to the sampled
 * surface point; `back_angle` the same for that point pushed `thickness` world
 * units further along the view ray (the slab's far face). `side` is +1 for the
 * +ω march and −1 for the −ω march, which is the only thing that signs the
 * span. `n` is the projected normal's angle.
 *
 * WHAT THIS REPLACES. Before R5a a sample only ever RAISED a running horizon,
 * and `thinOccluderCompensation` (0.2, unitless) let the horizon decay back
 * down when a later sample sat lower — a guess at "the occluder was thin"
 * with no relation to how thick the occluder actually was, applied identically
 * to a fence plank and to a barn wall, and unable to express a foreground
 * sliver occluding a middle band and nothing beyond it. The slab states the
 * geometry instead: a sample occludes exactly [front, back] and nothing past
 * it, so a plank costs a plank's worth of hemisphere. Both edges of the
 * hemicircle then fall out for free — a background sample below the tangent
 * plane has front_angle past the edge and clamps to zero width, and a coplanar
 * neighbour sits exactly on the edge.
 */
[[nodiscard]] inline GtaoSlabSpan gtao_slab_span(float front_angle, float back_angle, float side,
                                                 float n) {
    const float lo_angle = side > 0.0f ? front_angle : -back_angle;
    const float hi_angle = side > 0.0f ? back_angle : -front_angle;
    const float base = n - GTAO_HALF_PI;
    GtaoSlabSpan span;
    span.lo = std::clamp((lo_angle - base) / GTAO_PI, 0.0f, 1.0f);
    span.hi = std::clamp((hi_angle - base) / GTAO_PI, 0.0f, 1.0f);
    return span;
}

/**
 * Per-slice visibility from an occlusion bitmask: the cosine-weighted sum of
 * the sectors whose bit is CLEAR. An all-clear mask reproduces
 * `gtao_slice_visibility` with unoccluded horizons exactly; an all-set mask is
 * 0. As with the arc integral this is BEFORE the projected-normal-length
 * weighting the shader applies per slice.
 */
[[nodiscard]] inline float gtao_slice_visibility_bitmask(uint32_t occluded, float n) {
    float visibility = 0.0f;
    for (int k = 0; k < GTAO_SECTOR_COUNT; ++k) {
        if (((occluded >> k) & 1u) == 0u) {
            visibility += gtao_sector_weight(k, n);
        }
    }
    return visibility;
}

// ---------------------------------------------------------------------------
// Bent cone
// ---------------------------------------------------------------------------

/**
 * Cone half-angle cosine fitted to a visibility lobe of MEAN RESULTANT LENGTH
 * `r` = |Σ wᵢ dᵢ| / Σ wᵢ, the length of the (cosine-weighted) mean unoccluded
 * direction before normalisation.
 *
 * For a cone of half-angle α the cosine-weighted mean resultant length is
 *
 *   r(α) = ∫₀^α cos θ · cos θ sin θ dθ / ∫₀^α cos θ sin θ dθ
 *        = (2/3) · (1 − c³)/(1 − c²)  =  (2/3) · (1 + c + c²)/(1 + c),  c = cos α
 *
 * which runs monotonically from 2/3 at α = 90° (the open hemisphere) to 1 at
 * α = 0. Inverting it is a quadratic: with m = 3r/2 − 1 ∈ [0, ½],
 *
 *   c² + (1 − u)c + (1 − u) = 0,  u = 3r/2   ⇒   c = (m + √(m² + 4m)) / 2.
 *
 * r below 2/3 means the visible set is SPLIT across the hemisphere (two open
 * lobes on opposite sides), which no cone models; α clamps to 90° there, the
 * widest cone, which is the non-committal answer.
 */
[[nodiscard]] inline float gtao_cone_cos_from_resultant(float r) {
    const float m = std::max(1.5f * r - 1.0f, 0.0f);
    return std::clamp(0.5f * (m + std::sqrt(m * m + 4.0f * m)), 0.0f, 1.0f);
}

/**
 * The stored aperture: sin²α, the fraction of the hemisphere's PROJECTED solid
 * angle the fitted cone covers — 1 for a fully open cone, 0 for a closed one.
 * This is both the channel-A encoding of the half-angle and, directly, the
 * factor the diffuse GI term is scaled by (shaders/bent_cone.hlsli).
 */
[[nodiscard]] inline float gtao_cone_aperture(float resultant_length) {
    const float c = gtao_cone_cos_from_resultant(resultant_length);
    return std::clamp(1.0f - c * c, 0.0f, 1.0f);
}

// ---------------------------------------------------------------------------
// Per-frame temporal jitter
// ---------------------------------------------------------------------------

/**
 * 2-D Roberts/R2 low-discrepancy sequence (Roberts 2018): 1/ρ and 1/ρ² for the
 * plastic number ρ. The SAME pair as `shaders/rt_sampling.hlsli` and
 * `render/gi/screen_probe_math.hpp`.
 *
 * The two scales must be rationally INDEPENDENT together with 1, and that is
 * the whole point of them. The previous pair (φ⁻¹, 1 − φ⁻¹) summed to exactly
 * 1.0 in binary32 (Sterbenz), so `temporalOffset == 1 − temporalRotation` on
 * every frame and the shader's (noiseSlice, noiseStep) pair was pinned to the
 * anti-diagonal of the unit square: noiseSlice + noiseStep == frac(IGN1 + IGN2),
 * a per-pixel CONSTANT. Each axis passed any uniformity test on its own while
 * the temporal accumulation converged to a line integral indexed by that
 * constant instead of to the 2-D integral it is estimating, leaving a
 * stationary per-pixel residual no number of frames removes. 1/ρ has an
 * irreducible cubic minimal polynomial, so no such relation exists for R2.
 */
inline constexpr float TEMPORAL_R2_A = 0.7548776662466927f; // 1/ρ
inline constexpr float TEMPORAL_R2_B = 0.5698402909980532f; // 1/ρ²

/// `frac(frame * scale)`, the per-frame jitter both AO axes are offset by.
[[nodiscard]] inline float temporal_sequence(uint64_t frame_index, float scale) {
    // Accumulate in double: at frame 4095 the float product is ~3000, where one
    // ulp is 2.4e-4 and the low bits of the fractional part are already gone.
    const double x = static_cast<double>(frame_index % 4096u) * static_cast<double>(scale);
    return static_cast<float>(x - std::floor(x));
}

} // namespace detail

/**
 * Screen-space ground-truth ambient occlusion with a VISIBILITY BITMASK and a
 * bent cone (GTAO, Jimenez et al. 2016; SSILVB, Therrien et al. 2023).
 *
 * Runs AFTER the depth prepass and BEFORE the forward pass, from the prepass
 * depth alone (glTF alphaMode MASK/BLEND geometry is not in the prepass, so
 * it neither receives its own depth nor occludes). The forward shaders sample
 * the full-res result (`ao_texture()`) and use it in their AMBIENT term only —
 * direct sun light and speculars are never attenuated.
 *
 * SINCE R5a THE OUTPUT IS RGBA, NOT A SCALAR. Same horizon loop, three answers
 * (shaders/bent_cone.hlsli owns the encoding and is what consumers include):
 *   R  ambient occlusion, 1 = unoccluded
 *   GB world-space bent normal, octahedral
 *   A  cone aperture as sin²(half-angle) — the diffuse-GI gate
 * The AOMode setting is unchanged in meaning: this is an upgrade of the same
 * pass, not a new mode.
 *
 * Per-frame pass chain (render(), all raster fullscreen passes at the AO
 * working resolution = half res for AOMode::GTAO_Half, full res for
 * AOMode::GTAO_Full):
 *   1. GTAO           depth -> raw AO+cone (RGBA8); per slice a 32-sector
 *                     occlusion bitmask marked by constant-thickness slabs,
 *                     integrated with the cosine weight of the arc integral it
 *                     replaces; per-pixel IGN + per-frame R2 jitter
 *                     (detail::temporal_sequence), depth-reconstructed
 *                     edge-aware normals.
 *   2. GTAOTemporal   raw + history + velocity + depth -> TWO targets: the new
 *                     RGBA16F history (AO, oct.xy, aperture) and an RG16F
 *                     depth history (linear view Z) for next frame's
 *                     disocclusion test. The two are separate because five
 *                     quantities do not fit in four filterable channels, and
 *                     every one of them is filtered.
 *   3-4. BlurAO       depth-aware separable blur of all four channels
 *                     (history stays unfiltered so the blur never compounds).
 *   5. GTAOUpsample   joint-bilateral upsample to the full-res RGBA8 AO
 *                     texture (+ ao_strength); exact passthrough at full
 *                     working res.
 * The bent normal is decoded, filtered as a direction and re-encoded at every
 * one of those stages — averaging octahedral coordinates across the fold seam
 * is not averaging directions. That is also why NO stage reads the AO chain
 * through a hardware LINEAR sampler: stage 2's reprojection builds its bilinear
 * filter from 4 nearest taps by hand for exactly this reason. The aperture, a
 * projected solid angle, filters linearly as it stands.
 *
 * History is INVALID (temporal pass outputs the current frame) on the first
 * frame and after resize()/set_half_res()/invalidate_history() — the
 * SceneRenderer invalidates whenever AO was off or the prepass didn't run,
 * mirroring TAA's policy.
 */
class GTAO {
public:
    GTAO();
    ~GTAO();

    GTAO(const GTAO&) = delete;
    GTAO& operator=(const GTAO&) = delete;
    GTAO(GTAO&&) = delete;
    GTAO& operator=(GTAO&&) = delete;

    [[nodiscard]] bool init(gpu::GPUDevice& device, int width, int height, bool half_res = true);
    void shutdown();
    void resize(int width, int height); // no-op at same size; otherwise invalidates history
    void set_half_res(bool half_res);   // no-op at same quality; otherwise invalidates history
    bool half_res() const { return half_res_; }

    bool is_ready() const;

    /** Drop accumulated history: the next render() outputs the current frame
     *  unblended. Called on resize/quality change and by the SceneRenderer
     *  whenever AO was inactive for a frame. */
    void invalidate_history() { history_valid_ = false; }
    bool history_valid() const { return history_valid_; }

    /** Full-res RGBA8 AO texture the forward pass samples: R = AO
     *  (1 = unoccluded), GB = octahedral world bent normal, A = cone aperture
     *  (shaders/bent_cone.hlsli). Cleared to WHITE at init/resize, which
     *  decodes to "unoccluded, no directional data" — so it is safe to bind
     *  before the first render(). Null until init(). */
    SDL_GPUTexture* ao_texture() const { return ao_full_ ? ao_full_->handle() : nullptr; }

    struct FrameParams {
        // The DRAW projection (jittered under TAA — it must match the depth
        // buffer contents) and its inverse.
        glm::mat4 projection{1.0f};
        glm::mat4 inv_projection{1.0f};
        // Inverse of the DRAW view matrix. The pass reconstructs everything in
        // view space and publishes the bent normal in WORLD space, because no
        // consumer of the AO texture has a view matrix to undo it with.
        glm::mat4 inv_view{1.0f};
        float radius_scale = 1.0f; // GraphicsSettings::ao_radius
        float strength = 1.0f;     // GraphicsSettings::ao_strength
        uint64_t frame_index = 0;  // drives the temporal rotation/offset jitter
    };

    /**
     * Record the AO pass chain (no render pass may be active). `depth` is the
     * full-res prepass scene depth; `velocity` the full-res prepass RG16F
     * velocity target — both must hold THIS frame's contents
     * (SceneRenderer::depth_prepass_active()).
     */
    void render(SDL_GPUCommandBuffer* cmd, gpu::PipelineRegistry& pipelines, gpu::GPUTexture& depth,
                gpu::GPUTexture& velocity, const FrameParams& params);

private:
    void create_textures();
    void clear_full_res_texture();

    gpu::GPUDevice* device_ = nullptr;

    // Working-resolution chain. Every RGBA texel is (AO, oct.x, oct.y,
    // aperture) — see shaders/bent_cone.hlsli.
    std::unique_ptr<gpu::GPUTexture> ao_raw_;     // RGBA8: this frame's raw GTAO
    std::unique_ptr<gpu::GPUTexture> blur_tmp_;   // RGBA8: horizontal blur ping
    std::unique_ptr<gpu::GPUTexture> ao_blurred_; // RGBA8: blurred accumulated result
    // Double-buffered history: [history_write_] receives this frame's
    // accumulation while the other slot supplies last frame's. RGBA16F rather
    // than RGBA8 because an 8-bit grid stalls a 0.9-feedback accumulation —
    // a per-frame delta under half an LSB never lands.
    std::array<std::unique_ptr<gpu::GPUTexture>, 2> history_;
    // The disocclusion half of the history: R = last frame's linear view
    // depth. Its own target because the four channels above are full and all
    // five values are linearly filtered, so nothing can be packed.
    std::array<std::unique_ptr<gpu::GPUTexture>, 2> history_depth_;
    int history_write_ = 0;
    bool history_valid_ = false;

    // Full-resolution output the forward pass samples.
    std::unique_ptr<gpu::GPUTexture> ao_full_;

    SDL_GPUSampler* nearest_clamp_sampler_ = nullptr;
    SDL_GPUSampler* linear_clamp_sampler_ = nullptr;

    int width_ = 0;  // full resolution
    int height_ = 0;
    int ao_width_ = 0; // working resolution (== full res unless half_res_)
    int ao_height_ = 0;
    bool half_res_ = true;
};

} // namespace bracken::render
