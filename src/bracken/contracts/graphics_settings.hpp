#pragma once

#include <algorithm>
#include <limits>
#include <string>

namespace bracken {

// Rendering mode enums. The explicit underlying values are the integers
// written by save()/load() — keep them stable so existing settings files
// keep their meaning. Serialize with static_cast<int> at the file and
// GPU-uniform boundaries only.

enum class ShadowMode : int {
    Off = 0,
    PCF = 1,
    PCSS = 2,
};

// Screen-space ambient occlusion: horizon-based GTAO computed from the depth
// prepass and applied to the forward pass's AMBIENT term only. Half/Full is
// the AO compute resolution. Numeric values were redefined in the AO
// rearchitecture (1 was "SSAO", 2 was a fake "GTAO"); old settings files map
// 1 -> GTAO_Half and 2 -> GTAO_Full, which preserves their relative cost.
enum class AOMode : int {
    Off = 0,
    GTAO_Half = 1,
    GTAO_Full = 2,
};

enum class AAMode : int {
    Off = 0,
    FXAA = 1,
    TAA = 2,
};

enum class TonemapMode : int {
    ACESFitted = 0,
    ACESNarkowicz = 1,
    Reinhard = 2,
    None = 3,
};

// The enums below replace bare ints whose meaning lived in a trailing comment.
// The failure mode that motivated them is invisible rather than loud: writing
// `shadow_resolution = 2048` looks obviously right, silently clamps to 3, and
// yields 4096; `shadow_cascades = 4` clamps to 3, which also happens to mean
// four cascades. Naming the values makes both a compile error.

/** Far-plane / culling distance in world units. */
enum class DrawDistance : int {
    D500 = 0,
    D1000 = 1,
    D2000 = 2,
    D4000 = 3,
    D8000 = 4,
    D16000 = 5,
    D32000 = 6,
};

/** Texture anisotropy. `X2` = 2x, and so on. */
enum class AnisotropicFilter : int {
    Off = 0,
    X2 = 1,
    X4 = 2,
    X8 = 3,
    X16 = 4,
};

/** Present mode. Clamped to the max the swapchain supports at apply time. */
enum class VSyncMode : int {
    Immediate = 0,
    Vsync = 1,
    Mailbox = 2,
};

/** Cascade count for the sun's CSM. Use cascade_count() for the actual number
 *  — the enumerator values are indices, so `Four` is 3. */
enum class ShadowCascades : int {
    One = 0,
    Two = 1,
    Three = 2,
    Four = 3,
};

/**
 * Resolution the traced ray signals are produced at, relative to the screen.
 *
 * The dominant cost dial for both RT features, because it sets how many rays
 * are cast. The denoisers upsample back to full resolution, so dropping a step
 * costs sharpness at silhouettes rather than producing a visibly blocky image.
 */
enum class RtResolution : int {
    Full = 0,
    Half = 1,
    Quarter = 2,
};

/**
 * How far from the camera an instance may sit and still be traced.
 *
 * The traversal/TLAS lever: instances whose bounds lie ENTIRELY beyond the
 * cutoff never enter the TLAS, so they cost neither a rebuild nor a node
 * visit. Distant sun shadows keep coming from the cascade maps, which already
 * outrange the traced mask.
 *
 * `Unlimited` is the default and means "trace everything the frame drew" —
 * the behaviour every release before this enum had.
 */
enum class RtRange : int {
    Near = 0,
    Medium = 1,
    Far = 2,
    Unlimited = 3,
};

/**
 * Ray-traced shadows for LOCAL (point/spot) lights.
 *
 * `Off` is the rasterized 8x8-tile shadow atlas the engine has always had
 * (`render/lighting/local_shadows.hpp`): hard edges, and hard budgets — 8 point
 * lights, 16 spot lights, 6 tile re-renders a frame. Light #9 gets nothing.
 *
 * `Ratio` replaces all of that with ONE stochastic shadow ray per pixel,
 * whichever light it lands on. Per pixel the pass draws candidates from the
 * froxel the pixel falls in, picks one by weighted-reservoir sampling against
 * the light's own unshadowed contribution, merges the pick with the reprojected
 * previous-frame reservoir, and traces a single sphere-cap ray at it. The
 * denoised result is a scalar visibility RATIO the forward pass multiplies its
 * whole point+spot subtotal by (Heitz 2018) — so the ANALYTIC lighting stays
 * sharp and only the shadowing is stochastic, and every light in the frame
 * casts a soft shadow for the price of the one ray. Requires the ray-traced
 * scene and the prepass G-buffer, exactly as `rt_shadows` does; in this mode
 * the atlas is not updated and no tile is rendered at all.
 *
 * `Full` (R5c) is the same one ray, spent differently. Instead of a scalar
 * ratio the trace evaluates the WINNER'S OWN BRDF and writes two DEMODULATED
 * radiance estimates — diffuse without albedo, specular without Fresnel-F0 —
 * which are denoised as radiance and remodulated per pixel in the forward
 * shaders. The point and spot lights then leave the analytic cluster loop
 * ENTIRELY: nothing multiplies them, because nothing sums them.
 *
 * What that buys over `Ratio`, and it is not "more accurate shadows" — the
 * shadow ray is the same ray:
 *
 *  - **Colour.** A ratio is one scalar for the whole point+spot subtotal, so a
 *    red lantern and a blue one shadowing the same pixel dim each other's
 *    colour. Full carries the winner's radiance, so a shadow cast out of the
 *    red lantern leaves the blue light's colour behind it.
 *  - **Specular.** The ratio multiplies diffuse and specular alike, which is
 *    wrong whenever the light that dominates the highlight is not the light
 *    that dominates the irradiance. Full estimates the two lobes separately.
 *
 * What it costs: the estimate is stochastic in the SURFACE now, not only in
 * the shadowing, so it is noisier than `Ratio` at the same ray count and leans
 * harder on the denoiser; and the traced pass runs at its own resolution, so
 * local lighting is reconstructed by a depth-aware bilinear rather than
 * evaluated per pixel. `Ratio` remains the default recommendation and the iGPU
 * answer. Full needs everything Ratio needs — the acceleration structure and
 * the prepass G-buffer — and nothing more.
 */
enum class LocalRtShadowMode : int {
    Off = 0,
    Ratio = 1,
    Full = 2,
};

/**
 * Diffuse global illumination.
 *
 * `Probes` is the scrolled probe clipmap (`render/gi/probe_volume.hpp`): three
 * camera-centred cascades of octahedral irradiance + visibility probes, updated
 * from a fixed ray budget per frame rather than per pixel, which is what makes
 * GI affordable on hardware that cannot afford a ray per pixel.
 *
 * `ProbesScreen` layers phase R4's screen-probe gather
 * (`render/gi/screen_probes.hpp`) over that world field: probes placed on the
 * depth prepass march the Hi-Z pyramid (`render/hzb.hpp`), and the result is
 * weight-blended back toward the field wherever the screen has no answer. It
 * therefore REQUIRES the depth prepass, and falls back to plain `Probes`
 * behaviour (with a warning) if the gather cannot allocate.
 *
 * `VoxelProbes` fills the SAME probe field from a voxel radiance clipmap
 * instead of from rays (`render/gi/voxel_gi.hpp`). It is the RASTER
 * alternative: no BVH, no acceleration structure, and it runs with
 * `rt_shadows` and `rt_reflections` off — the sun's visibility comes from the
 * cascaded shadow map the raster path already renders, which is the whole
 * economy of the technique. Everything downstream of the probe field is
 * identical to `Probes`, because the two modes differ only in what fills the
 * ray-results buffer. Expect softer, leakier light than `Probes` (a voxel is a
 * volume element and has no normal, so the cosine is an isotropic mean) in
 * exchange for needing no ray tracing at all.
 *
 * `Off` is the default and is the pre-GI engine exactly: ambient comes from the
 * IBL sky probe or the hemisphere fallback, as it always has.
 */
enum class GiMode : int {
    Off = 0,
    Probes = 1,
    ProbesScreen = 2,
    VoxelProbes = 3,
};

/**
 * Probe-field budget tier: rays per probe per update and how often a probe is
 * revisited.
 *
 * Selects `render/gi/probe_schedule.hpp`'s `gi_budget()` row — rays per probe
 * (32 / 64 / 96) and the fraction of a cascade refreshed per turn (1/8 at Low,
 * 1/4 at Medium and High). `ProbeVolume::Config::quality` carries it and the
 * volume's update pass reads it, so this is the field that decides both the
 * per-frame trace cost and how fast the field converges after a light moves.
 */
enum class GiQuality : int {
    Low = 0,
    Medium = 1,
    High = 2,
};

/** Per-cascade shadow map edge length in texels. */
enum class ShadowResolution : int {
    R512 = 0,
    R1024 = 1,
    R2048 = 2,
    R4096 = 3,
};

/** Edge length of the local (point/spot) shadow atlas in texels. */
enum class LocalShadowResolution : int {
    R1024 = 0,
    R2048 = 1,
    R4096 = 2,
};

enum class WindowMode : int {
    Windowed = 0,
    BorderlessFullscreen = 1,
    ExclusiveFullscreen = 2,
};

/** Number of cascades, i.e. 1..4 — not the enumerator's index value. */
[[nodiscard]] constexpr int cascade_count(ShadowCascades c) {
    return std::clamp(static_cast<int>(c), 0, 3) + 1;
}

// The lookups below SATURATE rather than fall back to a default. The fields
// are public and these enums are int-backed, so a corrupt settings file or a
// static_cast can still present an out-of-range value; saturating keeps a
// too-large value large and a too-small value small, which is the behavior
// these accessors had when the fields were bare ints.
/** Shadow map edge length in texels. */
[[nodiscard]] constexpr int shadow_map_texels(ShadowResolution r) {
    constexpr int table[] = {512, 1024, 2048, 4096};
    return table[std::clamp(static_cast<int>(r), 0, 3)];
}

/** Local shadow atlas edge length in texels. */
[[nodiscard]] constexpr int local_shadow_atlas_texels(LocalShadowResolution r) {
    constexpr int table[] = {1024, 2048, 4096};
    return table[std::clamp(static_cast<int>(r), 0, 2)];
}

/** Far distance in world units. */
[[nodiscard]] constexpr float draw_distance_units(DrawDistance d) {
    constexpr float table[] = {500.0f, 1000.0f, 2000.0f, 4000.0f, 8000.0f, 16000.0f, 32000.0f};
    return table[std::clamp(static_cast<int>(d), 0, 6)];
}

/**
 * Ray-tracing instance cutoff in WORLD UNITS — the same scale as
 * draw_distance_units(), not a second one.
 *
 * The three finite steps are DrawDistance table entries on purpose, so the two
 * knobs can be read against each other: the traced range is always a subset of
 * what is drawn, and `Far` is exactly the D4000 default draw distance, i.e.
 * "trace everything a default frame draws". `Medium` (2000) brackets the
 * playground village's flat plateau (PLATEAU_RADIUS = 2400 units) — the whole
 * settlement, none of the outlying hills — and `Near` (1000) keeps roughly the
 * buildings fronting the plaza. At the playground's 100 units/metre
 * (`VILLAGE_SCALE`) that reads as 10 m / 20 m / 40 m across a ~200 m map.
 *
 * `Unlimited` returns infinity rather than a large number, so the use site
 * tests `std::isfinite` and skips the filter entirely instead of comparing
 * every instance against a sentinel.
 */
[[nodiscard]] constexpr float rt_range_units(RtRange r) {
    constexpr float table[] = {1000.0f, 2000.0f, 4000.0f, std::numeric_limits<float>::infinity()};
    return table[std::clamp(static_cast<int>(r), 0, 3)];
}

/** Anisotropy as a sample-count multiplier (1, 2, 4, 8, 16). */
[[nodiscard]] constexpr int anisotropy_samples(AnisotropicFilter f) {
    constexpr int table[] = {1, 2, 4, 8, 16};
    return table[std::clamp(static_cast<int>(f), 0, 4)];
}

struct GraphicsSettings {
    // Bumped when default semantics change (e.g. the wave-3 lighting model
    // retuned sun/ambient/exposure/bloom). Files saved without this key were
    // authored against the old lighting model and get their lighting-related
    // values migrated to the new defaults on load().
    static constexpr int CURRENT_SETTINGS_VERSION = 2;
    int settings_version = CURRENT_SETTINGS_VERSION;

    bool fog_enabled = true;
    bool grass_enabled = true;
    bool skybox_enabled = true;
    bool trees_enabled = true;
    bool rocks_enabled = true;
    bool show_fps = false;
    bool show_debug_hud = false;

    // Culling settings
    DrawDistance draw_distance = DrawDistance::D4000;
    bool frustum_culling = true;
    /**
     * Hi-Z occlusion culling for model draws (docs/gap_streaming.md).
     *
     * OFF by default, and it must stay that way until somebody measures it.
     * The depth it tests against is a FRAME OLD by construction — the Prepare
     * stage runs before this frame's depth prepass and SDL3 has no
     * non-blocking map — so under camera motion the test is a heuristic rather
     * than a proof, and a false cull is a prop missing behind a wall: a few
     * hundred pixels, invisible in `mean_luma`, exactly the failure mode
     * CLAUDE.md warns about twice.
     *
     * It also costs a SECOND full mip pyramid — culling needs the opposite
     * reduction to the screen-probe march — plus one texture readback and one
     * deterministic fence wait per frame.
     */
    bool occlusion_culling = false;

    // Quality settings
    AnisotropicFilter anisotropic_filter = AnisotropicFilter::X16;
    VSyncMode vsync_mode = VSyncMode::Mailbox; // clamped to max supported at apply
    ShadowMode shadow_mode = ShadowMode::PCSS;
    ShadowCascades shadow_cascades = ShadowCascades::Four;
    ShadowResolution shadow_resolution = ShadowResolution::R2048;
    // Sun (CSM) shadow range in world units, additionally clamped by draw
    // distance at the use site; the shader fades shadows out over the last
    // 15% instead of popping. Texel density scales inversely — raising it
    // softens distant shadows at a given shadow_resolution.
    float shadow_distance = 4000.0f;
    // Local (point/spot) light shadows: budgeted shadow maps for lights the
    // host opted in via PointLight/SpotLight::casts_shadows.
    bool local_shadows = true;                // master toggle for local-light shadows
    int local_shadow_point_budget = 4;        // shadowed point lights per frame (0-8)
    int local_shadow_spot_budget = 2;         // shadowed spot lights per frame (0-16)
    LocalShadowResolution local_shadow_resolution = LocalShadowResolution::R2048;
    AOMode ao_mode = AOMode::GTAO_Half;
    // TAA is the default for fresh settings; files that stored a mode keep it.
    AAMode aa_mode = AAMode::TAA;
    bool ibl_enabled = true;                  // Image-based lighting: ambient from the prefiltered
                                              // procedural sky; OFF falls back to hemisphere ambient
    bool bloom_enabled = true;                // Bloom post-processing
    float bloom_strength = 0.25f;             // Bloom intensity (0.0 - 1.0)
    // --- Compute-shader ray tracing -------------------------------------
    // BOTH OFF BY DEFAULT and independently switchable. There is no hardware
    // ray tracing in SDL3 GPU, so these traverse a software BVH the engine
    // builds; that is a large cost for a taste-dependent gain, and neither
    // even builds an acceleration structure unless it is on.
    //
    // rt_reflections additionally needs per-triangle attributes, which more
    // than doubles the structure's memory — so enabling it after the scene was
    // built without them logs an error rather than rendering black.
    bool rt_shadows = false;     ///< traced sun shadows instead of / alongside CSM
    bool rt_reflections = false; ///< traced specular for smooth surfaces
    /// Rays cast per frame, relative to the screen. The main perf dial for
    /// each feature; the denoiser upsamples whatever it is given.
    /// Both default to FULL. The half/quarter defaults were inherited from
    /// when these signals were a raw binary mask and a raw mirror ray, where
    /// tracing sparsely was the only thing that made them affordable. Now that
    /// both are denoised, the reduced settings mainly cost silhouette
    /// sharpness — and on the hardware this was developed against, Full is
    /// visibly better for no framerate the eye can find. Weaker GPUs have the
    /// dial; it is no longer the default position.
    RtResolution rt_shadow_resolution = RtResolution::Full;
    RtResolution rt_reflection_resolution = RtResolution::Full;
    /// Angular RADIUS of the sun disk in degrees — the shadow softness knob.
    /// The real sun is ~0.265; larger is softer. 0 collapses the cone to a
    /// point and reproduces a hard shadow exactly, which is the honest way to
    /// A/B the soft-shadow path. Costs nothing to trace: the ray count is the
    /// same either way, only the denoiser works harder.
    float rt_sun_softness_deg = 0.35f;
    /// Roughness at which traced specular hands over completely to the IBL
    /// probe. Raising it makes more of the frame reflect; because the pass is
    /// bound by wave divergence rather than by traced-pixel count, that is
    /// nearly free once anything is tracing at all (measured 0.45 -> 0.75 for
    /// ~2 fps on the playground village).
    float rt_reflection_max_roughness = 0.75f;
    /// How far an instance may be from the camera and still be traced. This is
    /// the only RT dial that cuts the ACCELERATION STRUCTURE rather than the
    /// ray count: instances entirely beyond it are dropped before the TLAS is
    /// hashed, so they cost neither traversal nor a rebuild.
    ///
    /// Default `Unlimited` — behaviour is exactly what it was before the knob
    /// existed until someone dials it. The price of a finite setting is stated
    /// where it is consumed (SceneRenderer::refresh_raytracing_instances):
    /// crossing the threshold changes the instance list, so a moving camera can
    /// re-trigger full TLAS rebuilds that a static scene otherwise never pays
    /// — a moving instance only costs a refit, but an instance dropping in or
    /// out of range changes the topology and there is no tree to refit onto.
    RtRange rt_max_instance_range = RtRange::Unlimited;
    /// Ray-traced local (point/spot) light shadows — see LocalRtShadowMode.
    /// OFF BY DEFAULT: `Ratio` builds the acceleration structure (like
    /// rt_shadows), costs one more traced pass plus its denoiser, and RETIRES
    /// the rasterized atlas for the frames it runs. `Off` is the atlas path
    /// byte for byte.
    LocalRtShadowMode rt_local_shadows = LocalRtShadowMode::Off;
    /// Resolution the local-shadow ray is traced at. Half by default rather
    /// than Full — unlike the sun, this signal picks a DIFFERENT light per
    /// pixel per frame, so its raw variance is higher and the denoiser is
    /// doing more work regardless; spending twice the rays to sharpen a
    /// silhouette it is going to filter anyway is the wrong end of the trade.
    RtResolution rt_local_shadow_resolution = RtResolution::Half;
    // --- Probe-field global illumination ---------------------------------
    /// Diffuse GI mode. OFF BY DEFAULT: the probe field costs ~30 MB of atlases
    /// the moment it is switched on, and (until R1b) buys nothing. `Off` is the
    /// pre-GI path byte for byte.
    GiMode gi_mode = GiMode::Off;
    /// Probe ray budget tier. Stored and persisted now, consumed by R1b — see
    /// the enum. Medium is the default because it is the tier the plan's
    /// mid-range budget (§6 of docs/rt_lighting_plan.md) is written against.
    GiQuality gi_quality = GiQuality::Medium;
    bool volumetric_fog = false;              // Atmospheric fog haze (expensive, off by default)
    bool god_rays = false;                    // Light shafts through shadows (requires volumetric_fog)
    WindowMode window_mode = WindowMode::Windowed;
    int resolution_index = 0;                 // index into RenderContext::available_resolutions(); an index past
                                              // that list falls back to borderless at apply time (with a warning)

    // Tone, exposure, color grading
    // With auto_exposure on, `exposure` is the manual BIAS multiplied on top
    // of the adapted exp2(EV); with it off, the whole exposure (pre-P6
    // behavior). Look-style grading (white balance, CDL) lives in
    // render::EnvironmentState, not here.
    float exposure = 1.0f; // linear scene multiplier before tonemap; 1.0 = neutral
    TonemapMode tonemap_mode = TonemapMode::ACESFitted;
    float contrast = 1.0f;   // post-tonemap contrast; 1.0 = neutral, 1.2 = punchy
    float saturation = 1.0f; // post-tonemap saturation; 1.0 = neutral, 0.0 = grayscale

    // Histogram auto exposure (render::AutoExposure; needs the HDR path).
    // The adapted EV is clamped to [min_ev, max_ev]; the default scene is
    // calibrated to converge near EV 0, so 0/0 would pin it off entirely.
    bool auto_exposure = true;
    float auto_exposure_min_ev = -4.0f;    // clamp on the adapted EV (clamped to [-12, 12] on load)
    float auto_exposure_max_ev = 4.0f;
    float auto_exposure_speed_up = 1.5f;   // EV/s toward brighter (scene got darker)
    float auto_exposure_speed_down = 2.5f; // EV/s toward darker (scene got brighter)

    // Lighting feel
    float ambient_strength = 0.45f; // shadow-region lift; 0.0 = full black in shadow, 1.0 = fully lit
    float sun_intensity = 3.0f;     // multiplier on directional sun light (HDR: >1 so bloom/tonemap have headroom)

    // Bloom
    // POST-exposure HDR luminance above which bloom contributes. The prefilter
    // thresholds scene luminance against the SAME product the composite pass
    // applies — `exposure * exp2(adapted EV)` — so what blooms is what will be
    // bright on screen, at any adaptation. It reads the auto-exposure state
    // buffer to do it; with auto exposure off the factor is the manual bias
    // alone, and at exposure = 1.0 / EV = 0 this is the plain pre-exposure cut.
    float bloom_threshold = 1.1f;

    // AO
    float ao_strength = 1.0f; // AO darkening intensity (0..1+, 1 = stock)
    float ao_radius = 1.0f;   // multiplier on AO sample radius

    // Volumetric fog
    float fog_density = 0.05f; // density multiplier

    /** Field-by-field equality. Exact (floats compared with ==) on purpose:
     *  the question it answers is "did anything change", and a tolerance
     *  would swallow a deliberate hairline edit. */
    friend bool operator==(const GraphicsSettings&, const GraphicsSettings&) = default;

    float get_draw_distance() const { return draw_distance_units(draw_distance); }

    /**
     * Clamp every enum-like/index field to its valid range. Applied by load()
     * so file contents are normalized once at the trust boundary; call it
     * yourself after mutating fields from untrusted input.
     */
    void clamp_to_valid_ranges();

    [[nodiscard]] bool save(const std::string& path) const;
    [[nodiscard]] bool load(const std::string& path);
};

} // namespace bracken
