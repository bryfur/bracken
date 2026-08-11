#include "bracken/ui/nine_slice.hpp"

#include <algorithm>

namespace bracken::ui {

uint32_t build_nine_slice(const NineSlice& slice, const Rect& dst, std::span<NineSliceQuad, 9> out) {
    if (dst.empty()) {
        return 0;
    }

    // Destination band widths. Scaled down proportionally when the authored
    // borders do not fit, so the middle collapses to zero instead of a quad
    // inverting (see the header).
    float dl = std::max(slice.border.l, 0.0f);
    float dr = std::max(slice.border.r, 0.0f);
    float dt = std::max(slice.border.t, 0.0f);
    float db = std::max(slice.border.b, 0.0f);
    if (dl + dr > dst.w) {
        const float f = (dl + dr) > 0.0f ? dst.w / (dl + dr) : 0.0f;
        dl *= f;
        dr *= f;
    }
    if (dt + db > dst.h) {
        const float f = (dt + db) > 0.0f ? dst.h / (dt + db) : 0.0f;
        dt *= f;
        db *= f;
    }

    // Absolute edge positions, then DIFFERENCE them for the widths. Adding
    // widths instead lets rounding put the right column a fraction past
    // dst.right() and open a hairline seam at the panel edge.
    const float xs[4] = {dst.x, dst.x + dl, dst.right() - dr, dst.right()};
    const float ys[4] = {dst.y, dst.y + dt, dst.bottom() - db, dst.bottom()};

    // UVs come from the SOURCE border, never the scaled destination one: the
    // texture's corner is the corner whatever size the box ended up.
    const float tw = slice.texture_size.x > 0.0f ? slice.texture_size.x : 1.0f;
    const float th = slice.texture_size.y > 0.0f ? slice.texture_size.y : 1.0f;
    //
    // The two inner columns are also forced MONOTONE. Clamping them
    // independently is not enough: authored borders whose sum exceeds the
    // source size (a 40 px border on a 48 px texture, or a texture_size the
    // caller got wrong) put the third column BELOW the second, and a quad with
    // u1 < u0 samples its band mirrored — the corner art drawn inside out,
    // which reads as a texture bug rather than as bad authoring. Collapsing the
    // middle band to zero width is the same degenerate-but-correct answer the
    // destination bands above already give.
    const float u_lo = std::clamp(slice.border.l / tw, 0.0f, 1.0f);
    const float u_hi = std::max(u_lo, std::clamp(1.0f - slice.border.r / tw, 0.0f, 1.0f));
    const float v_lo = std::clamp(slice.border.t / th, 0.0f, 1.0f);
    const float v_hi = std::max(v_lo, std::clamp(1.0f - slice.border.b / th, 0.0f, 1.0f));
    const float us[4] = {0.0f, u_lo, u_hi, 1.0f};
    const float vs[4] = {0.0f, v_lo, v_hi, 1.0f};

    uint32_t n = 0;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            if (!slice.draw_center && row == 1 && col == 1) {
                continue;
            }
            NineSliceQuad q{};
            q.dst = Rect{xs[col], ys[row], xs[col + 1] - xs[col], ys[row + 1] - ys[row]};
            q.u0 = us[col];
            q.u1 = us[col + 1];
            q.v0 = vs[row];
            q.v1 = vs[row + 1];
            out[n++] = q;
        }
    }
    return n;
}

} // namespace bracken::ui
