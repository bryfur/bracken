#include "bracken/scene/render_plan.hpp"

namespace bracken::scene {

RenderPlan plan_frame(const GraphicsSettings& gfx, const FramePacket& frame, RendererCaps caps) {
    RenderPlan plan;

    // 3D needs BOTH content and a camera. `camera_seen` is sticky across
    // frames: a camera-less frame reuses the previous camera, but a host that
    // never set one renders nothing at all (diagnosed in render_frame).
    plan.has_content = frame.has_3d_content() && caps.camera_seen;

    // Frame-scoped cull constants, straight off the settings.
    plan.fog_active = gfx.fog_enabled;
    plan.draw_dist_sq = gfx.get_draw_distance() * gfx.get_draw_distance();
    plan.do_frustum_cull = gfx.frustum_culling;

    // The HDR offscreen path is not a preference — it is available exactly
    // when the post chain initialized. Scene lighting stays in linear HDR and
    // the composite pass tonemaps to the swapchain.
    plan.use_hdr_path = caps.post_ready;

    // TAA: asked for AND resolvable (offscreen HDR path + live history).
    plan.taa_active = gfx.aa_mode == AAMode::TAA && plan.use_hdr_path && caps.taa_ready;
    plan.invalidate_taa_history = !plan.taa_active;

    // Sky IBL: setting on AND the prefilter chain alive.
    plan.ibl_active = gfx.ibl_enabled && caps.ibl_ready;

    // Auto exposure records only on HDR-path frames that actually drew
    // something; any other frame stales the adapted EV.
    plan.invalidate_auto_exposure = !gfx.auto_exposure || !plan.use_hdr_path || !plan.has_content;

    // Shadows: the atlas must exist and the host must not have turned them
    // off. The count is what binning, the cascade loop and the sampler all
    // agree on, so it is computed once here.
    plan.active_cascades = (caps.shadows_ready && gfx.shadow_mode != ShadowMode::Off) ? caps.shadow_cascades : 0;

    return plan;
}

} // namespace bracken::scene
