#pragma once

#include <cstdint>

namespace bracken {

/**
 * UiTextureId — a stable, packet-safe name for a UI texture.
 *
 * `render::UIRenderer::draw_textured_quad` takes a raw `SDL_GPUTexture*`,
 * which a sim job may never hold: the packet's rule is that everything in it
 * is by value or packet-allocated (`scene/frame_packet.hpp`), and SDL types
 * are `main-thread` tier. That is why nine-slice sprites were impossible from
 * `on_simulate` — there was no way to NAME a texture.
 *
 * The host registers the texture once on the main thread
 * (`SceneRenderer::register_ui_texture`), keeps the id, and references it from
 * the sim stage. The renderer resolves it back to the pointer at consume time;
 * an id that resolves to nothing is COUNTED
 * (`FrameDiagnostics::ui_textures_unresolved`) and degrades to a solid tint
 * quad, never a silent no-draw.
 *
 * Ids are never reused within a process run: `unregister_ui_texture` clears
 * the slot but does not recycle the value, so a stale id held across an
 * unload is diagnosable rather than a wrong texture.
 */
struct UiTextureId {
    static constexpr uint32_t k_invalid = 0u;

    uint32_t value = k_invalid;

    [[nodiscard]] constexpr bool valid() const noexcept { return value != k_invalid; }

    friend constexpr bool operator==(UiTextureId a, UiTextureId b) noexcept = default;
};

static_assert(sizeof(UiTextureId) == sizeof(uint32_t), "UiTextureId must stay a bare handle word");

} // namespace bracken
