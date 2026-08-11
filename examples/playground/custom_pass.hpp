#pragma once

#include "bracken/gpu/host_pipeline.hpp"
#include "bracken/scene/render_hook.hpp"
#include "bracken/scene/scene_renderer.hpp"

#include <cstdint>
#include <glm/glm.hpp>

namespace playground {

/**
 * @brief THE WORKED EXAMPLE for host-extensible rendering — a game's own
 *        render pass, drawn with a game's own pipeline, from a game's own HLSL.
 *
 * Copy this file. It is deliberately the whole loop and nothing else:
 *
 *  1. `examples/playground/shaders/playground_tint.frag.hlsl` is compiled by
 *     the ENGINE's pinned shadercross/DXC through `bracken_compile_shaders`
 *     (see `examples/playground/CMakeLists.txt`) and staged into
 *     `shaders_playground/` next to the executable.
 *  2. `install()` points the registry at that directory
 *     (`add_shader_search_path`) and registers ONE graphics pipeline
 *     (`register_graphics_pipeline`), getting back an opaque handle. Note that
 *     the VERTEX shader is the engine's own `fullscreen.vert.spv`: search roots
 *     resolve engine-first, so a host pipeline can reuse an engine shader
 *     without knowing where it lives.
 *  3. `install()` registers ONE hook in `RenderHookSlot::AfterOpaque`, which is
 *     the point in the frame where scene colour and scene depth both hold this
 *     frame's result and neither is bound.
 *  4. `draw()` records a render pass into the frame's command buffer and — the
 *     one rule that is not optional — ENDS it before returning.
 *
 * OFF BY DEFAULT, and that is load-bearing rather than shy: `--smoke` is the
 * engine's screenshot regression gate, so anything that paints on the frame has
 * to be opt-in or the pinned `mean_luma` moves and every downstream table in
 * CLAUDE.md loses its reference. Turn it on with `PLAYGROUND_CUSTOM_PASS=1`.
 *
 * Tier: main-thread throughout. `install()`/`uninstall()` are `on_init` /
 * `on_shutdown` calls; `draw()` runs on the main thread inside the renderer's
 * frame. Nothing here may be touched from `on_simulate`.
 */
class CustomTintPass {
public:
    /// True when `PLAYGROUND_CUSTOM_PASS` is set to anything but "0".
    /// Separate from install() so the caller can say so in its log line.
    [[nodiscard]] static bool enabled_by_environment();

    /**
     * Register the pipeline and the hook. Call from `on_init` AFTER
     * `init_renderer()` — the registry has no device before that and would
     * refuse.
     *
     * @return false if the shader could not be resolved/compiled into a
     *         pipeline, or if the hook was refused. Both cases log; the object
     *         is left uninstalled and `uninstall()` on it is a no-op.
     */
    [[nodiscard]] bool install(bracken::scene::SceneRenderer& renderer);

    /// Remove the hook and release the pipeline. Safe to call when nothing was
    /// installed. Must run BEFORE the renderer shuts down, i.e. from
    /// `on_shutdown`, which is exactly where the Application lifecycle puts it.
    void uninstall(bracken::scene::SceneRenderer& renderer);

    [[nodiscard]] bool installed() const noexcept { return hook_.is_valid(); }

    /// Linear HDR colour added to the scene, weighted by the vignette. Alpha is
    /// the overall strength; the blend mode ignores the shader's alpha output
    /// (see the .cpp for why).
    void set_tint(const glm::vec4& tint) noexcept { tint_ = tint; }

    /// How many times the hook actually recorded a pass. The number a smoke run
    /// prints: "the hook ran" and "the hook drew something" are different
    /// claims, and this is the first one.
    [[nodiscard]] uint64_t executions() const noexcept { return executions_; }

private:
    void draw(const bracken::scene::RenderHookContext& ctx);

    bracken::gpu::HostPipelineHandle pipeline_{};
    bracken::scene::RenderHookHandle hook_{};
    /// Warm amber, ~1/3 strength. Linear, not sRGB — the scene target is
    /// RGBA16F linear HDR and the composite tonemaps afterwards.
    glm::vec4 tint_{0.65f, 0.32f, 0.08f, 0.35f};
    uint64_t executions_ = 0;
};

} // namespace playground
