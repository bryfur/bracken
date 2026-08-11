#pragma once

#include "bracken/ui/ui_types.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <span>

namespace bracken::ui {

/**
 * A nine-slice sprite: a texture whose four border bands stay at their
 * authored size while the middle stretches, so one 48x48 png can draw a panel
 * at any size without the corners smearing.
 */
struct NineSlice {
    /// Source texture size in pixels. Used only to turn `border` into UVs.
    glm::vec2 texture_size{1.0f, 1.0f};
    /// Border widths in SOURCE-TEXTURE pixels.
    Edges border{};
    /// When false the middle patch is not emitted (a frame with a hole).
    bool draw_center = true;
};

/** One of the nine patches: a destination rectangle and its UV window. */
struct NineSliceQuad {
    Rect dst{};
    float u0 = 0.0f;
    float v0 = 0.0f;
    float u1 = 1.0f;
    float v1 = 1.0f;
};

/**
 * Build the patches that tile `dst`, writing them into `out` and returning
 * how many were written (9, or 8 with `draw_center` false; 0 for an empty
 * `dst`).
 *
 * The patches EXACTLY tile `dst`: the seams are shared edges, so their areas
 * sum to `dst`'s area and no two overlap. That is the property the test pins,
 * and it is why the destination band widths are computed by differencing
 * absolute edge positions rather than by adding widths.
 *
 * BORDERS WIDER THAN `dst` are scaled down proportionally rather than
 * allowed to overlap or invert. A 16 px border on an 8 px-wide box becomes
 * two 4 px borders with a zero-width middle — degenerate, but never a quad
 * with negative width, which is what an unguarded subtraction produces and
 * what makes a rasterizer draw the sprite inside out.
 *
 * Pure math: no packet, no GPU, no texture. Emitting the quads is the
 * caller's job (`ui::Context::panel_sprite` turns them into
 * `UiTexturedQuadCommand`s).
 */
[[nodiscard]] uint32_t build_nine_slice(const NineSlice& slice, const Rect& dst, std::span<NineSliceQuad, 9> out);

} // namespace bracken::ui
