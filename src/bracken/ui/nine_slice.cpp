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
    const float us[4] = {0.0f, std::clamp(slice.border.l / tw, 0.0f, 1.0f),
                         std::clamp(1.0f - slice.border.r / tw, 0.0f, 1.0f), 1.0f};
    const float vs[4] = {0.0f, std::clamp(slice.border.t / th, 0.0f, 1.0f),
                         std::clamp(1.0f - slice.border.b / th, 0.0f, 1.0f), 1.0f};

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
