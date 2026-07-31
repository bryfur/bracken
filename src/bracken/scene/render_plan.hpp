#pragma once

#include "bracken/graphics_settings.hpp"
#include "bracken/scene/frame_packet.hpp"

namespace bracken::scene {

/**
 * The renderer-side facts the pass-selection tree needs and that neither the
 * graphics settings nor the sealed packet can supply: whether a sub-renderer
 * is actually alive this frame, and whether a camera was EVER set.
 *
 * Every one of these is a "can this even run" answer, not a preference. The
 * settings say what the host wants; these say what the renderer can deliver.
 * Asking for TAA on a renderer whose post chain never initialized has to
 * resolve to "off", and that decision cannot be made from settings alone —
 * which is exactly why this struct exists instead of plan_frame() reaching
 * into sub-renderers it would then have to own.
 */
struct RendererCaps {
    bool post_ready = false;  ///< PostProcessStack::is_ready() — gates the whole HDR path
    bool taa_ready = false;   ///< TAA::is_ready() — history targets alive
    bool ibl_ready = false;   ///< IBL::is_ready() — sky prefilter chain alive
    bool camera_seen = false; ///< a camera was set on THIS or an earlier frame (sticky)
    bool shadows_ready = false;  ///< ShadowMap::is_ready() — the CSM atlas exists
    int shadow_cascades = 0;     ///< ShadowMap::active_cascades() (a settings-driven count)
};

/**
 * Which passes run this frame, and with what frame-scoped constants —
 * the authoritative answer, derived once at the top of render_frame().
 *
 * IMMUTABLE HALF ONLY, and deliberately so. Three frame-scoped renderer
 * members are written MID-frame, after the camera/frustum state is final:
 * `frame_prepass_active_` (the prepass reports whether it actually recorded),
 * `frame_ao_texture_` (GTAO publishes its output) and
 * `frame_shadow_uniforms_` (filled after the cascades update). They cannot
 * live in a const struct built before those passes run, so they are NOT here.
 * Their eventual home is a separate mutable FramePassState — named in
 * docs/refactor_plan.md Phase 6 so the next pass does not re-derive it.
 */
struct RenderPlan {
    /// 3D passes run at all: the packet describes geometry AND a camera exists.
    bool has_content = false;
    /// Render 3D through the HDR offscreen target + post chain (vs straight
    /// to the swapchain). Everything downstream of the composite hangs off it.
    bool use_hdr_path = false;
    /// TAA resolve runs — which is also what enables the sub-pixel camera
    /// jitter. Jitter without a resolve to integrate the samples just
    /// shimmers, so this is one decision, not two.
    bool taa_active = false;
    /// Sky IBL feeds the scene shaders' iblEnabled flag (the sampler slots
    /// are bound either way — typed dummies when inactive).
    bool ibl_active = false;
    /// Distance fog term in the model/instanced lighting uniforms.
    bool fog_active = false;
    /// Frustum-cull draws against the main view frustum during prep.
    bool do_frustum_cull = false;
    /// Squared draw distance, the prep cull radius (settings distance²).
    float draw_dist_sq = 0.0f;
    /// CSM cascades to bin, render and sample this frame; 0 == no shadow
    /// passes at all. `active_cascades > 0` IS the "shadows on" predicate:
    /// ShadowMap::active_cascades() is clamped to [1, CSM_MAX_CASCADES], so
    /// the count is zero exactly when the atlas is unavailable or the host
    /// set ShadowMode::Off.
    int active_cascades = 0;

    // ---- Reported, not performed -------------------------------------
    // The prologue interleaves state mutation with decisions: the TAA history
    // and the auto-exposure adaptation must be dropped on exactly the frames
    // their decision flips false, or a re-enable ghosts/pops against stale
    // history. plan_frame() is pure, so it REPORTS these and render_frame()
    // makes the calls. Do not "simplify" by invalidating from inside the
    // planner — it would make a pure function that mutates two sub-renderers.

    /// TAA is not resolving this frame: drop the accumulated history.
    bool invalidate_taa_history = false;
    /// Auto exposure cannot record this frame: the adapted EV is stale, so
    /// snap to the metered target when it comes back.
    bool invalidate_auto_exposure = false;
};

/**
 * Decide the frame's pass set from the host's settings, the sealed packet and
 * what the renderer can currently do. Pure: no renderer state is read or
 * written, which is what makes the whole tree testable headless
 * (scene/render_plan.test.cpp).
 *
 * Two decisions in render_frame() are deliberately NOT here:
 *  - the minimized-window early-out, which must sit between begin_frame() and
 *    end_frame() so the command-buffer lifecycle stays balanced; and
 *  - the mid-frame TAA history drop for a jittered frame whose prepass or
 *    velocity target turned out to be unavailable — that is a fact discovered
 *    while recording, not a plan.
 */
[[nodiscard]] RenderPlan plan_frame(const GraphicsSettings& gfx, const FramePacket& frame, RendererCaps caps);

} // namespace bracken::scene
